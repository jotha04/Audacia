"""Servicios del carril 4 — Haku Inventario, API Interna y Voz Web.

Responsabilidades clave:
1. Cola segura y Lock para convivir con Haku Visitante sin colisionar en GPU.
2. Normalización fonética para síntesis de voz (Piper TTS: 'AudacIA' -> 'Audacia').
3. Generación de audio WAV para el cliente (suena exclusivamente en el navegador).
4. Transcripción de voz (faster-whisper) desde el navegador.
5. Procesamiento de comandos de inventario en lenguaje natural (consultas y mutaciones).
6. Motor de pre-confirmación en dos pasos para mutaciones de inventario.
7. Ejecución atómica y segura de movimientos con select_for_update(), idempotencia
   y auditoría con marca via_haku=True.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import io
import json
import logging
import re
import secrets
import struct
import threading
import time
import unicodedata
import uuid
import wave
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

from django.conf import settings
from django.core.exceptions import PermissionDenied, ValidationError
from django.db import IntegrityError, OperationalError, models, transaction
from django.db.models import Case, F, Q, Value, When
from django.utils import timezone

from cuentas.models import UsuarioAudacIA
from inventario.models import (
    Almacen,
    Auditoria,
    Categoria,
    DevolucionPrestamo,
    Existencia,
    Movimiento,
    Prestamo,
    Producto,
    Ubicacion,
)
from .telemetria import registrar_timing
from .servicios_sesion import (
    ErrorConflictoOCC,
    ErrorSesionInvalida,
    ErrorTokenSesionInvalido,
    capturar_estado_sesion_occ,
    ejecutar_guardado_occ,
    obtener_o_crear_session_token,
    persistir_propuesta_sesion_occ,
)

logger = logging.getLogger("integraciones.haku")

VERSION_FIRMA_PROPUESTA = "v2"

HAKU_PROPUESTA_TIMEOUT_SEGUNDOS = getattr(
    settings, "HAKU_PROPOSAL_TIMEOUT", getattr(settings, "HAKU_PROPUESTA_TIMEOUT_SEGUNDOS", 1800)
)  # 30 minutos por defecto (transaccional)

# ═════════════════════════════════════════════════════════════════════════════
# MATRIZ CANÓNICA DE PERMISOS RBAC PARA HAKU INVENTARIO
# ═════════════════════════════════════════════════════════════════════════════

MAPA_PERMISOS_OPERACIONES = {
    # Mutaciones de inventario
    "REGISTRAR_ENTRADA": "movimientos.entrada",
    "ENTRADA": "movimientos.entrada",
    "REGISTRAR_SALIDA": "movimientos.dotacion",
    "SALIDA": "movimientos.dotacion",
    "DOTACION": "movimientos.dotacion",
    "PRESTAMO": "movimientos.prestamo",
    "DEVOLUCION": "movimientos.devolucion",
    "TRASLADAR_PRODUCTO": "movimientos.traslado_almacen",
    "TRASLADO_ENTRE": "movimientos.traslado_almacen",
    "TRASLADO": "movimientos.traslado_almacen",
    "TRASLADO_INTERNO": "movimientos.traslado_interno",
    "BAJA": "movimientos.baja",
    # Catálogo
    "CREAR_PRODUCTO": "catalogo.crear",
    "EDITAR_PRODUCTO": "catalogo.editar",
    "INHABILITAR_PRODUCTO": "catalogo.eliminar",
    "HABILITAR_PRODUCTO": "catalogo.editar",
    "BUSCAR_PRODUCTO": "catalogo.ver",
    "CONSULTAR_STOCK": "catalogo.ver",
    "LISTAR_CATALOGO": "catalogo.ver",
    "PRODUCTOS_CON_STOCK": "catalogo.ver",
    "CONSULTAR_UBICACION": "catalogo.ver",
    "CONSULTAR_UBICACION_COLECCION": "catalogo.ver",
    "LISTAR_UBICACIONES": "catalogo.ver",
    "LISTAR_ALMACENES": "catalogo.ver",
    # Analítica y alertas
    "ALERTAS_STOCK": "analitica.ver_alertas",
    "PRODUCTOS_SIN_STOCK": "analitica.ver_alertas",
    "PRODUCTOS_BAJO_STOCK": "analitica.ver_alertas",
    "CONSULTAR_FALTANTES_REPOSICION": "analitica.ver_alertas",
    "CONSULTAR_METRICAS_MOVIMIENTOS": "analitica.ver_reportes",
    "CONSULTAR_RESUMEN_INVENTARIO": "analitica.ver_reportes",
    # Historial y préstamos activos
    "PRESTAMOS_ACTIVOS": "movimientos.ver",
    "CONSULTAR_PRODUCTO_MOVIMIENTOS": "movimientos.ver",
}


def usuario_tiene_permiso_operativo(usuario: Any, codigo_permiso: str) -> bool:
    """Verifica si el usuario cuenta con el permiso operativo requerido en el modelo RBAC."""
    if not usuario or not getattr(usuario, "is_authenticated", False):
        return False
    if not getattr(usuario, "is_active", True) or getattr(usuario, "eliminado_logicamente", False):
        return False
    if getattr(usuario, "es_administrador", False) or getattr(usuario, "is_superuser", False):
        return True
    if hasattr(usuario, "tiene_permiso"):
        return usuario.tiene_permiso(codigo_permiso)
    return False

# 1. CONTROL DE CONCURRENCIA INTRA-PROCESO PARA TRANSCRIPCIÓN Y AUDIO (DJANGO)
# ═════════════════════════════════════════════════════════════════════════════

# Lock reentrante intra-proceso para serializar llamadas de transcripción/audio dentro de los hilos de Django.
# NOTA TÉCNICA: threading.RLock() es exclusivo del proceso del worker de Django. No coordina
# procesos independientes de Windows (como run_hacu.py o llama-server.exe). La serialización global
# del LLM se gobierna a nivel de red en llama-server.exe (--parallel 1/2), y la coexistencia con
# Faster-Whisper depende del driver CUDA y el fallback a CPU int8 en HakuTranscriber.
HAKU_GPU_LOCK = threading.RLock()


class HakuOcupadoError(Exception):
    """Lanzada cuando otro hilo del servidor web retiene el lock de transcripción de audio."""
    pass


@contextmanager
def adquirir_lock_haku(timeout: float = 12.0):
    """Context manager para adquirir el lock de Haku con timeout seguro."""
    adquirido = HAKU_GPU_LOCK.acquire(timeout=timeout)
    if not adquirido:
        logger.warning("Timeout esperando lock de Haku (ocupado por Visitante o hilo previo)")
        raise HakuOcupadoError("Hacu Inventario está atendiendo otra solicitud. Por favor, inténtelo nuevamente en unos instantes.")
    try:
        yield
    finally:
        HAKU_GPU_LOCK.release()



# ═════════════════════════════════════════════════════════════════════════════
# 2. NORMALIZACIÓN FONÉTICA PARA SÍNTESIS DE VOZ (PIPER TTS)
# ═════════════════════════════════════════════════════════════════════════════

NUMEROS_TEXTO_ES = {
    0: "cero", 1: "un", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco",
    6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
    11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince",
    16: "dieciséis", 17: "diecisiete", 18: "dieciocho", 19: "diecinueve",
    20: "veinte", 21: "veintiuno", 22: "veintidós", 23: "veintitrés",
    24: "veinticuatro", 25: "veinticinco", 26: "veintiséis", 27: "veintisiete",
    28: "veintiocho", 29: "veintinueve", 30: "treinta", 40: "cuarenta",
    50: "cincuenta", 60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa",
    100: "cien",
}


def _numero_a_letras(n: int) -> str:
    """Convierte un número entero positivo a palabras en español."""
    if n in NUMEROS_TEXTO_ES:
        return NUMEROS_TEXTO_ES[n]
    if n < 100:
        dec = (n // 10) * 10
        uni = n % 10
        return f"{NUMEROS_TEXTO_ES[dec]} y {NUMEROS_TEXTO_ES[uni]}"
    if n < 1000:
        cen = (n // 100) * 100
        resto = n % 100
        pref = "ciento" if cen == 100 else {
            200: "doscientos", 300: "trescientos", 400: "cuatrocientos",
            500: "quinientos", 600: "seiscientos", 700: "setecientos",
            800: "ochocientos", 900: "novecientos"
        }.get(cen, str(cen))
        return f"{pref} {_numero_a_letras(resto)}" if resto > 0 else pref
    if n < 1000000:
        miles = n // 1000
        resto = n % 1000
        pref = "mil" if miles == 1 else f"{_numero_a_letras(miles)} mil"
        return f"{pref} {_numero_a_letras(resto)}" if resto > 0 else pref
    return str(n)


DIGITOS_A_LETRAS = {
    "0": "cero",
    "1": "uno",
    "2": "dos",
    "3": "tres",
    "4": "cuatro",
    "5": "cinco",
    "6": "seis",
    "7": "siete",
    "8": "ocho",
    "9": "nueve",
}


def _formatear_codigo_tts(m: re.Match) -> str:
    """Convierte códigos de inventario (ej. CAB-0001 -> código CAB guion cero cero cero uno)."""
    prefijo = m.group(1)
    digitos = m.group(2)
    digitos_leidos = " ".join(DIGITOS_A_LETRAS.get(d, d) for d in digitos)
    return f"código {prefijo} guion {digitos_leidos}"


def normalizar_para_tts(texto: str) -> str:
    """Normaliza texto formal para que Piper TTS lo pronuncie con prosodia profesional.

    Reglas de calidad e idempotencia:
    - Remueve markdown (negritas, cursivas, encabezados, viñetas, bloques de código).
    - Remueve emojis y símbolos especiales no verbales.
    - Convierte códigos de inventario (CAB-0001 -> código CAB guion cero cero cero uno) sin duplicar 'código código'.
    - Pronunciación de AudacIA -> Audacia.
    - Expansión contextual de baterías AA -> doble A, y AAA -> triple A.
    - Conversión de decimales a español hablado (1,5 -> uno coma cinco, 2,50 -> dos coma cincuenta).
    - Expansión de números cardinales aislados de cualquier magnitud.
    - Conversión de ubicaciones alfanuméricas (Estante A1 -> Estante A uno, Cajón B-2 -> Cajón B dos).
    """
    if not texto:
        return ""

    resultado = texto

    # 1. Limpieza de emojis y caracteres gráficos no verbales
    resultado = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50-\u2b55]", "", resultado)

    # 2. Reemplazo fonético para AudacIA y Haku/Hacu (evita h muda -> "aku" en Piper TTS)
    resultado = re.sub(r"\bAudacIA\b", "Audacia", resultado, flags=re.IGNORECASE)
    resultado = re.sub(r"\b(?:Haku|Hacu)\b", "Jacu", resultado, flags=re.IGNORECASE)

    # 3. Limpieza de markdown
    resultado = re.sub(r"\*\*([^*]+)\*\*", r"\1", resultado)
    resultado = re.sub(r"\*([^*]+)\*", r"\1", resultado)
    resultado = re.sub(r"`([^`]+)`", r"\1", resultado)
    resultado = re.sub(r"^#+\s*", "", resultado, flags=re.MULTILINE)
    resultado = re.sub(r"^\s*[-*•]\s*", "", resultado, flags=re.MULTILINE)

    # 3-bis. Contexto de baterías para TTS: 'baterías AA' -> 'baterías doble A', 'pilas AAA' -> 'pilas triple A'
    resultado = re.sub(r"(?i)\b(bater[ií]as?|pilas?)\s+AAA\b", r"\1 triple A", resultado)
    resultado = re.sub(r"(?i)\b(bater[ií]as?|pilas?)\s+AA\b", r"\1 doble A", resultado)

    # 4. Códigos de inventario (evitando duplicar 'código código' si ya venía en el texto)
    resultado = re.sub(
        r"(?i)(?:\bc[oó]digo\s+)?\b([A-Z]{2,6})-(\d{1,6})\b",
        _formatear_codigo_tts,
        resultado,
    )
    resultado = re.sub(r"(?i)\bc[oó]digo\s+c[oó]digo\b", "código", resultado)

    # 4-bis. Transformación contextual para ubicaciones numéricas y alfanuméricas (ej. Cajón 1 -> Cajón uno, Estante A1 -> Estante A uno, Cajón B-2 -> Cajón B dos)
    def _reemplazar_ubicacion(m: re.Match) -> str:
        pref = m.group(1)
        letra = m.group(2)
        num_str = m.group(3)
        try:
            num = int(num_str)
            palabras = _numero_a_letras(num)
            if palabras == "un" or palabras.endswith(" y un"):
                palabras = palabras + "o"
            if letra:
                return f"{pref} {letra.upper()} {palabras}"
            return f"{pref} {palabras}"
        except ValueError:
            return m.group(0)

    resultado = re.sub(
        r"(?i)\b(caj[oó]n|estante|vitrina|pasillo|gaveta|bodega|almac[eé]n|repisa|secci[oó]n|nivel|fila|columna)\s+(?:([A-Za-z])[\s\-_]?)?(\d{1,4})\b",
        _reemplazar_ubicacion,
        resultado,
    )

    # 4-ter. Concordancia fonética de unidades
    resultado = re.sub(r"(?i)\b1\s+unidades\b", "una unidad", resultado)
    resultado = re.sub(r"(?i)\b1\s+unidad\b", "una unidad", resultado)

    # 4-cuater. Normalización fonética de fechas a español natural (ej. 22/10/2026 -> veintidós de octubre de 2026)
    MESES_ESP = {
        1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
        7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
    }
    def _reemplazar_fecha(m: re.Match) -> str:
        d_str, m_str, y_str = m.group(1), m.group(2), m.group(3)
        try:
            d = int(d_str)
            mes = int(m_str)
            y = int(y_str)
            if 1 <= d <= 31 and 1 <= mes <= 12 and 1900 <= y <= 2100:
                mes_palabra = MESES_ESP.get(mes, str(mes))
                d_palabra = _numero_a_letras(d) if d > 1 else "primero"
                if d_palabra == "un":
                    d_palabra = "uno"
                return f"{d_palabra} de {mes_palabra} de {y}"
        except Exception:
            pass
        return m.group(0)

    resultado = re.sub(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", _reemplazar_fecha, resultado)

    # 4-quinto. Decimales en cantidades y medidas (ej. 1,5 -> uno coma cinco, 2,50 -> dos coma cincuenta)
    def _reemplazar_decimal(m: re.Match) -> str:
        ent = int(m.group(1))
        d_str = m.group(2)
        ent_txt = _numero_a_letras(ent)
        if ent_txt == "un":
            ent_txt = "uno"
        if d_str.startswith("0") and len(d_str) > 1 and int(d_str) > 0:
            d_txt = " ".join(DIGITOS_A_LETRAS.get(d, d) for d in d_str)
        else:
            d_val = int(d_str)
            d_txt = _numero_a_letras(d_val)
            if d_txt == "un":
                d_txt = "uno"
        return f"{ent_txt} coma {d_txt}"

    resultado = re.sub(r"(?<!\d[/.-])\b(\d+)[,.](\d{1,3})\b(?![/.-]\d)", _reemplazar_decimal, resultado)

    # 5. Expansión de números enteros cardinales aislados (evitando códigos con guion)
    def _reemplazar_numero(m: re.Match) -> str:
        num_str = m.group(1)
        if len(num_str) > 1 and num_str.startswith("0"):
            return num_str  # Códigos con ceros a la izquierda se dejan para pronunciar dígito a dígito
        try:
            val = int(num_str)
            return _numero_a_letras(val)
        except ValueError:
            pass
        return num_str

    resultado = re.sub(r"(?<!guion\s)(?<!código\s)\b(\d+)\b", _reemplazar_numero, resultado)

    # 6. Colapsar espacios múltiples y asegurar idempotencia de 'código'
    resultado = re.sub(r"(?i)\bc[oó]digo\s+c[oó]digo\b", "código", resultado)
    resultado = re.sub(r"\s+", " ", resultado).strip()
    return resultado


# ═════════════════════════════════════════════════════════════════════════════
# 3. SÍNTESIS DE AUDIO PARA EL CLIENTE (AUDIO EN NAVEGADOR EXCLUSIVO)
# ═════════════════════════════════════════════════════════════════════════════


def sintetizar_audio_bytes(texto: str, interaction_id: Optional[str] = None) -> bytes:
    """Sintetiza texto a audio WAV en memoria usando HakuSynthesizer.

    REGLAS INNEGOCIABLES:
    1. Capa fina: texto -> HakuSynthesizer -> WAV bytes.
    2. NO reproduce en altavoces del servidor físico (reproducción exclusiva en el cliente).
    3. NO depende de hacu/ (Haku Visitante).
    4. NO devuelve tonos sinusoidales de fallback; lanza HakuSynthesizerError en caso de fallo.
    """
    texto_normalizado = normalizar_para_tts(texto)
    if not texto_normalizado:
        from .audio.synthesizer import HakuSynthesizerError
        raise HakuSynthesizerError("Texto de entrada vacío para síntesis de voz.")

    from .audio.synthesizer import HakuSynthesizer
    synth = HakuSynthesizer.obtener_instancia()
    return synth.sintetizar(texto_normalizado, interaction_id=interaction_id)


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRANSCRIPCIÓN DE VOZ CLIENTE (FASTER-WHISPER)
# ═════════════════════════════════════════════════════════════════════════════


def procesar_audio_y_transcribir(audio_bytes: bytes, interaction_id: Optional[str] = None) -> dict[str, Any]:
    """Procesa el buffer de audio (pipeline completo: decode -> analyzer -> VAD -> normalizer)
    y transcribe con faster-whisper únicamente si se detecta voz válida.

    Retorna un diccionario con:
    - estado: "VOZ_DETECTADA" | "SIN_VOZ"
    - motivo: "VOZ_DETECTADA" | "AUDIO_SILENCIOSO" | "SOLO_RUIDO" | "ERROR_DECODIFICACION"
    - texto: str
    - metricas: dict
    """
    if not audio_bytes:
        return {
            "estado": "SIN_VOZ",
            "motivo": "BUFFER_VACIO",
            "texto": "",
            "metricas": {},
        }

    t0_vad = time.perf_counter()
    from .audio.pipeline import AudioPipeline
    pipeline = AudioPipeline.desde_settings()
    resultado_proc = pipeline.procesar(audio_bytes)
    dt_vad = (time.perf_counter() - t0_vad) * 1000.0
    registrar_timing(interaction_id, "stt_vad_decode", dt_vad)

    if resultado_proc.estado == "SIN_VOZ":
        logger.info("Audio descartado por pipeline: %s", resultado_proc.motivo)
        return {
            "estado": "SIN_VOZ",
            "motivo": resultado_proc.motivo,
            "texto": "",
            "metricas": resultado_proc.to_dict(),
        }

    # Si se detectó voz válida, procedemos a transcribir usando HakuTranscriber
    samples_float32 = resultado_proc.audio_limpio
    texto_transcrito = ""

    from .audio.transcriber import HakuTranscriber, TranscripcionError

    transcriptor = HakuTranscriber.obtener_instancia()
    try:
        ya_filtrado_vad = bool(pipeline.vad_activado)
        # Esperar carga inicial del modelo FUERA del lock compartido para evitar que una
        # descarga o arranque en frío de Whisper bloquee HAKU_GPU_LOCK durante minutos.
        if not transcriptor.modelo_listo():
            if not transcriptor.esperar_modelo_listo(timeout=15.0, interaction_id=interaction_id):
                logger.info(
                    "[HakuSTT] interaction=%s modelo STT aún inicializando; respondiendo MOTOR_CARGANDO sin retener lock.",
                    interaction_id,
                )
                return {
                    "estado": "MOTOR_CARGANDO",
                    "motivo": "MODELO_CARGANDO",
                    "texto": "",
                    "error": "El motor de voz aún está terminando de iniciar. Espera unos segundos e intenta de nuevo.",
                    "metricas": resultado_proc.to_dict(),
                }
        # Serializar únicamente la inferencia rápida de CTranslate2/Whisper para evitar
        # colisiones concurrentes cuando varios dispositivos de la LAN dictan al mismo tiempo.
        with adquirir_lock_haku(timeout=30.0):
            texto_transcrito = transcriptor.transcribir(
                samples_float32,
                interaction_id=interaction_id,
                ya_filtrado_vad=ya_filtrado_vad,
            )
    except HakuOcupadoError:
        logger.warning("Haku ocupado por otro proceso al intentar transcribir")
        raise
    except TranscripcionError as err:
        logger.error("Error en motor STT de Haku: %s", err)
        return {
            "estado": "ERROR_TRANSCRIPCION",
            "motivo": "ERROR_STT",
            "texto": "",
            "error": "Fallo en motor de transcripción de voz.",
            "metricas": resultado_proc.to_dict(),
        }
    except Exception as exc:
        logger.exception("Error inesperado en transcripción faster-whisper: %s", exc)
        return {
            "estado": "ERROR_TRANSCRIPCION",
            "motivo": "ERROR_STT",
            "texto": "",
            "error": "Error interno en procesamiento de audio.",
            "metricas": resultado_proc.to_dict(),
        }

    dt_infer = getattr(transcriptor, "ultimo_tiempo_inferencia_ms", 0.0) or 0.0
    dt_model_load = getattr(transcriptor, "tiempo_carga_modelo_ms", None)
    metricas_dict = resultado_proc.to_dict()
    metricas_dict["tiempos"] = {
        "vad_decode_ms": round(dt_vad, 2),
        "inference_ms": round(dt_infer, 2),
        "model_load_ms": round(dt_model_load, 2) if dt_model_load is not None else None,
        "stt_compute_total_ms": round(dt_vad + dt_infer, 2),
    }

    return {
        "estado": "VOZ_DETECTADA",
        "motivo": "VOZ_DETECTADA",
        "texto": texto_transcrito,
        "metricas": metricas_dict,
    }


def transcribir_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe un archivo de audio (WAV/WebM) recibido del navegador cliente.

    Usa faster-whisper con el lock de Haku para evitar conflictos de VRAM con el LLM.
    """
    res = procesar_audio_y_transcribir(audio_bytes)
    return res.get("texto", "")


# ═════════════════════════════════════════════════════════════════════════════
# 5. PARSER Y PROCESADOR DE COMANDOS DE INVENTARIO EN LENGUAJE NATURAL
# ═════════════════════════════════════════════════════════════════════════════


def pluralizar(cantidad: int, singular: str, plural: str) -> str:
    """Retorna '1 unidad' o 'X unidades' según la cantidad (regla de concordancia de número en español)."""
    return f"{cantidad} {singular}" if cantidad == 1 else f"{cantidad} {plural}"


def pluralizar_md(cantidad: int, singular: str, plural: str) -> str:
    """Retorna '**1** unidad' o '**X** unidades' con el número en negrita para Markdown."""
    sufijo = singular if cantidad == 1 else plural
    return f"**{cantidad}** {sufijo}"


MAPEO_ALIAS_FAMILIAS = {
    # Familia Cables (prefijo CAB)
    "cable": "CAB",
    "cables": "CAB",
    # Familia Herramientas (prefijo HER)
    "herramienta": "HER",
    "herramientas": "HER",
    "ponchadora": "HER",
    "ponchadoras": "HER",
    # Familia Equipos de Red (prefijo RED)
    "switch": "RED",
    "switches": "RED",
    "router": "RED",
    "routers": "RED",
    "equipo de red": "RED",
    "equipos de red": "RED",
    "red": "RED",
    "redes": "RED",
}

PREFIJOS_INCOMPATIBLES = {
    "CAB": ("HER-", "RED-", "TST-"),
    "HER": ("CAB-", "RED-", "TST-"),
    "RED": ("CAB-", "HER-", "TST-"),
}

CATEGORIAS_PURAS_FAMILIA = {
    "CAB": ("cables",),
    "HER": ("herramientas",),
    "RED": ("redes", "equipos de red"),
}


def es_categoria_compuesta(nombre_cat: str) -> bool:
    """Detecta si una categoría es compuesta (agrupa múltiples familias mediante ' y ', comas o barras)."""
    if not nombre_cat:
        return False
    nom = nombre_cat.lower()
    return " y " in nom or "," in nom or "/" in nom or " & " in nom


def construir_filtro_familia_o_tipo(filtro_nombre: str, prefix_relacion: str = "") -> Q:
    """Construye un filtro Q determinista para productos por familia, tipo o nombre.
    
    Reglas de negocio y seguridad:
    1. Normalización explícita de alias de familias conocidas:
       - Cables: 'cable', 'cables' -> prefijo oficial 'CAB-'
       - Herramientas: 'herramienta', 'herramientas', 'ponchadora' -> prefijo oficial 'HER-'
       - Equipos de red: 'switch', 'switches', 'router', 'routers', 'equipo de red', 'redes' -> prefijo 'RED-'
    2. Exclusión estricta de prefijos incompatibles:
       - Una herramienta (código HER-) NUNCA se clasifica como cable, aunque su nombre contenga 'cable'
         (ej. 'Ponchadora para cable UTP') o su categoría sea 'Cables y Herramientas'.
       - Un cable (código CAB-) NUNCA se clasifica como switch o herramienta, aunque su nombre contenga 'switch'
         (ej. 'Cable para switch').
       - Códigos de prueba (TST-) nunca se clasifican en familias estándar.
    3. Categorías compuestas vs puras:
       - Categorías compuestas (ej. 'Cables y Herramientas') nunca se asumen homogéneas:
         los productos dentro de ellas deben discriminarse por su código o nombre directo.
       - Categorías puras solo coinciden si el producto no tiene un código incompatible.
    4. Términos no pertenecientes a familias estándar:
       - No se deducen prefijos arbitrarios tomando las 3 primeras letras.
       - Se realiza búsqueda por categoría exacta o coincidencia en el nombre del producto.
    """
    if not filtro_nombre:
        return Q()

    pr = prefix_relacion
    termino = filtro_nombre.strip().lower()

    # Resolver prefijo de familia mediante mapeo canónico explícito
    prefijo_familia = MAPEO_ALIAS_FAMILIAS.get(termino)
    if not prefijo_familia:
        if termino.endswith("es") and len(termino) > 4:
            prefijo_familia = MAPEO_ALIAS_FAMILIAS.get(termino[:-2])
        elif termino.endswith("s") and len(termino) > 3:
            prefijo_familia = MAPEO_ALIAS_FAMILIAS.get(termino[:-1])

    if prefijo_familia:
        # 1. Cláusula de exclusión de prefijos incompatibles (ej. HER- y RED- fuera de consultas de cables)
        incompatibles = PREFIJOS_INCOMPATIBLES.get(prefijo_familia, ())
        q_incomp = Q()
        for p in incompatibles:
            q_incomp |= Q(**{f"{pr}codigo_interno__istartswith": p})

        # 2. Categorías puras de la familia (nunca compuestas)
        q_cat_pura = Q()
        for cat_nom in CATEGORIAS_PURAS_FAMILIA.get(prefijo_familia, ()):
            q_cat_pura |= Q(**{f"{pr}categoria__nombre__iexact": cat_nom})

        # 3. Construcción de condición base según la familia o subtipo solicitado
        if termino in ("switch", "switches"):
            q_base = (
                (Q(**{f"{pr}codigo_interno__istartswith": "RED-"}) & Q(**{f"{pr}nombre__icontains": "switch"}))
                | Q(**{f"{pr}nombre__icontains": "switch"})
            )
        elif termino in ("router", "routers"):
            q_base = (
                (Q(**{f"{pr}codigo_interno__istartswith": "RED-"}) & Q(**{f"{pr}nombre__icontains": "router"}))
                | Q(**{f"{pr}nombre__icontains": "router"})
            )
        elif termino in ("ponchadora", "ponchadoras"):
            q_base = (
                (Q(**{f"{pr}codigo_interno__istartswith": "HER-"}) & Q(**{f"{pr}nombre__icontains": "ponchadora"}))
                | Q(**{f"{pr}nombre__icontains": "ponchadora"})
            )
        elif prefijo_familia == "CAB":
            q_base = (
                Q(**{f"{pr}codigo_interno__istartswith": "CAB-"})
                | Q(**{f"{pr}nombre__icontains": "cable"})
                | q_cat_pura
            )
        elif prefijo_familia == "HER":
            q_base = (
                Q(**{f"{pr}codigo_interno__istartswith": "HER-"})
                | Q(**{f"{pr}nombre__icontains": "herramienta"})
                | q_cat_pura
            )
        else:  # RED general ('red', 'redes', 'equipo de red', 'equipos de red')
            q_base = (
                Q(**{f"{pr}codigo_interno__istartswith": "RED-"})
                | Q(**{f"{pr}nombre__icontains": "red"})
                | q_cat_pura
            )

        return q_base & ~q_incomp
    else:
        # Término no es un alias de familia estándar: coincidencia por categoría exacta o por nombre
        q_cat = Q(**{f"{pr}categoria__nombre__iexact": filtro_nombre.strip()})
        q_nom = Q(**{f"{pr}nombre__icontains": termino})
        return q_cat | q_nom


from .lenguaje.numeros import extraer_numero_espanol as _extraer_numero


def _buscar_producto(termino: str, usuario: UsuarioAudacIA) -> Producto | None:
    """Busca un producto por código interno, código de barras o coincidencia en nombre."""
    if not termino:
        return None

    # Limpiar signos de puntuación e interrogación
    termino_limpio = re.sub(r"[¿?¡!.,:;()\[\]\"'`]", " ", termino).strip()
    if not termino_limpio:
        return None

    from .lenguaje.normalizador import NormalizadorLenguajeHaku
    prod, _, _ = NormalizadorLenguajeHaku._resolver_producto_en_texto(termino_limpio, usuario)
    return prod


def _validar_acceso_almacen(usuario: UsuarioAudacIA, almacen_intencion: Almacen | None) -> tuple[Almacen | None, dict[str, Any] | None]:
    """Valida los permisos de acceso al almacén solicitado.
    
    Retorna:
    - (almacen_filtrado, None) si el acceso es válido.
    - (None, dict_error) si el usuario normal intentó consultar un almacén no autorizado
      o si no tiene almacén asignado.
    """
    if almacen_intencion:
        if not usuario.tiene_acceso_almacen(almacen_intencion):
            msg = f"No cuenta con permiso para consultar el almacén '{almacen_intencion.nombre}'."
            return None, {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"error_permisos": True, "almacen_denegado": almacen_intencion.nombre},
            }
        return almacen_intencion, None

    if usuario.es_usuario_normal:
        if usuario.acceso_todos_almacenes:
            return None, None
        if usuario.almacen_asignado and usuario.tiene_acceso_almacen(usuario.almacen_asignado):
            return usuario.almacen_asignado, None
        primer_multi = usuario.almacenes_asignados.filter(activo=True, eliminado=False).first()
        if primer_multi:
            return primer_multi, None
        else:
            msg = "Su cuenta de usuario no tiene un almacén asignado activo para realizar esta consulta."
            return None, {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"error_permisos": True, "sin_almacen_asignado": True},
            }

    # Administrador o Superusuario sin almacén explícito: acceso global
    return None, None


def _buscar_almacen_por_nombre(nombre_str: str | None) -> Almacen | None:
    """Busca un almacén actual por nombre completo o por una variante inequívoca."""
    if not nombre_str:
        return None
    n = nombre_str.strip()
    normalizar_nombre = lambda valor: re.sub(
        r"\s+", " ", "".join(
            c for c in unicodedata.normalize("NFKD", str(valor).casefold())
            if not unicodedata.combining(c)
        ),
    ).strip()
    buscado = normalizar_nombre(n)
    almacenes = list(Almacen.objects.filter(activo=True, eliminado=False))

    exactos = [alm for alm in almacenes if normalizar_nombre(alm.nombre) == buscado]
    if len(exactos) == 1:
        return exactos[0]

    # Acepta un nombre sin el prefijo de tipo solo cuando identifica un único
    # almacén registrado. El resultado siempre proviene del catálogo actual.
    alias = re.sub(r"^(?:almacen|bodega)\s+", "", buscado).strip()
    if alias:
        if len(alias) == 1:
            candidatos = [alm for alm in almacenes if normalizar_nombre(alm.nombre) == f"almacen {alias}"]
        elif len(alias) >= 3:
            candidatos = [
                alm for alm in almacenes
                if normalizar_nombre(alm.nombre).endswith(f" {alias}")
            ]
        else:
            candidatos = []
        if len(candidatos) == 1:
            return candidatos[0]
    return None


def _detectar_almacen_destino_en_texto(texto: str) -> Almacen | None:
    """Detecta un destino citado en el turno usando los nombres existentes en BD."""
    texto_norm = "".join(
        c for c in unicodedata.normalize("NFKD", str(texto or "").casefold())
        if not unicodedata.combining(c)
    )
    texto_norm = re.sub(r"[^a-z0-9]+", " ", texto_norm).strip()
    candidatos = []
    for almacen in Almacen.objects.filter(activo=True, eliminado=False):
        nombre_norm = "".join(
            c for c in unicodedata.normalize("NFKD", almacen.nombre.casefold())
            if not unicodedata.combining(c)
        )
        nombre_norm = re.sub(r"[^a-z0-9]+", " ", nombre_norm).strip()
        variantes = {nombre_norm}
        alias = re.sub(r"^(?:almacen|bodega)\s+", "", nombre_norm).strip()
        if len(alias) >= 3:
            variantes.add(alias)
        for variante in variantes:
            patron = rf"(?<!\w){re.escape(variante)}(?!\w)"
            for match in re.finditer(patron, texto_norm):
                prefijo = texto_norm[max(0, match.start() - 24):match.start()]
                enlace = re.search(r"\b(?:a|al|hacia|para)\s+(?:(?:el|la)\s+)?$", prefijo)
                if enlace:
                    candidatos.append((match.start(), len(variante), almacen))
    if not candidatos:
        return None
    candidatos.sort(key=lambda item: (item[0], item[1]))
    return candidatos[-1][2]


def _resolver_almacen_para_operacion(
    usuario: UsuarioAudacIA,
    almacen_intencion: Almacen | None,
    producto: Producto,
    operacion: str = "operación",
) -> tuple[Almacen | None, dict[str, Any] | None]:
    """Resuelve determinísticamente el almacén para operaciones de mutación (entrada, salida, préstamo).
    
    Reglas:
    1. Si se indicó un almacén explícito:
       - Si el usuario tiene acceso y el almacén está activo: se usa.
       - Si no tiene acceso o está inactivo: se deniega con error de permisos.
    2. Si no se indicó almacén:
       - Si el usuario normal tiene un único almacén autorizado (sea por asignado o multi): se usa ese almacén.
       - Si el usuario tiene acceso a múltiples almacenes (o acceso a todos) y el producto está enlazado a su categoría:
         se usa el almacén principal de la categoría del producto si el usuario tiene acceso a él.
       - Si no se puede determinar sin ambigüedad: se solicita aclaración breve preguntando en qué almacén operar.
    """
    if almacen_intencion:
        if not usuario.tiene_acceso_almacen(almacen_intencion):
            msg = f"No cuenta con autorización para realizar la {operacion} en el almacén '{almacen_intencion.nombre}'."
            return None, {
                "tipo": "error",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"error_permisos": True, "almacen_denegado": almacen_intencion.nombre},
            }
        return almacen_intencion, None

    # Caso sin almacén explícito:
    alms_accesibles = list(usuario.obtener_almacenes_accesibles())
    if not alms_accesibles:
        msg = f"Su cuenta de usuario no tiene almacenes activos autorizados para registrar {operacion}."
        return None, {
            "tipo": "error",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"sin_almacen": True},
        }

    # Si solo tiene 1 almacén accesible
    if len(alms_accesibles) == 1:
        return alms_accesibles[0], None

    # Si tiene múltiples almacenes accesibles o acceso global:
    # Comprobar si el almacén principal de la categoría del producto está entre sus almacenes autorizados
    alm_cat = getattr(getattr(producto, "categoria", None), "almacen_principal", None)
    if alm_cat and usuario.tiene_acceso_almacen(alm_cat):
        return alm_cat, None

    # Si el producto tiene stock en un único almacén accesible para el usuario (relevante para salidas/préstamos)
    exs_con_stock = Existencia.objects.filter(
        producto=producto,
        almacen__in=alms_accesibles,
        stock_disponible__gt=0,
    )
    if exs_con_stock.count() == 1:
        return exs_con_stock.first().almacen, None

    # Ante ambigüedad con múltiples almacenes autorizados, solicitar aclaración breve
    nombres_alms = ", ".join(f"**{a.nombre}**" for a in alms_accesibles[:3])
    msg = f"Cuenta con acceso a varios almacenes ({nombres_alms}). ¿En cuál almacén desea registrar la {operacion} de **{producto.nombre}**?"
    return None, {
        "tipo": "aclaracion",
        "mensaje_pantalla": msg,
        "mensaje_verbal": normalizar_para_tts(f"Cuenta con acceso a varios almacenes. ¿En cuál desea registrar la {operacion} de {producto.nombre}?"),
        "datos": {"requiere_almacen": True},
    }


def _clasificar_turno_con_operacion_pendiente(texto: str, op_pendiente: dict | None) -> str:
    """Clasifica un turno entrante cuando existe `estado_operativo.operacion_pendiente`.

    Retorna:
    - "cancelar": si el usuario cancela explícitamente la operación pendiente.
    - "consulta_intermedia": si el usuario formula una pregunta o consulta informativa de lectura.
    - "nueva_operacion": si el usuario inicia explícitamente una operación de escritura distinta.
    - "ranura": si el mensaje debe procesarse como respuesta a la ranura esperada.
    """
    if not op_pendiente or not texto:
        return "ranura"

    t_raw = texto.strip()
    t_low = t_raw.lower()
    t_clean = re.sub(r"[^\w\sñáéíóú]", " ", t_low)
    t_clean = re.sub(r"\s+", " ", t_clean).strip()
    t_cancel = re.sub(r"\bpor\s+favor\b", "", t_clean).strip()

    # 0. Corrección explícita de cantidad en medio de una operación pendiente (ej. «ya no quiero 2 sino 4», «olvídalo, ponle 5»)
    esperando_slot = str(op_pendiente.get("esperando") or "").lower()
    if esperando_slot in ("cantidad", "destinatario", "destino", "ubicacion_destino", "almacen", "confirmar_menor_cantidad"):
        from .lenguaje.numeros import extraer_modificacion_cantidad_propuesta
        if extraer_modificacion_cantidad_propuesta(t_raw) is not None:
            return "ranura"

    # 1. Cancelación explícita (no aplica si «olvídalo»/«mejor» introduce una corrección de campo como «Olvídalo, que estén en Almacén B»)
    if re.search(
        r"^(?:no|cancelar|cancela|cancelado|canc[eé]lalo|olv[ií]dalo|olvidate|olv[ií]date|descarta|descartar|"
        r"ya\s+no|no\s+quiero|(?:no\s+)?d[eé]jal(?:o|a|os|as)\s+as[ií]|mejor\s+no|no\s+hagas\s+nada|"
        r"deja\s+eso\s+quieto|d[eé]jalo\s+quieto|ya\s+no\s+hagas\s+(?:nada|eso)|nombe(?:\s+ya\s+no)?|ninguna)$",
        t_cancel,
    ) or re.search(
        r"\b(cancela\s+(?:la\s+operaci[oó]n|el\s+traslado|el\s+pr[eé]stamo|la\s+salida|la\s+entrada|el\s+registro|eso)|"
        r"olv[ií]dalo\s+cancela|cancela\s+olv[ií]dalo|olv[ií]date\s+de\s+eso|no\s+hacer\s+nada|"
        r"mejor\s+cancela|ya\s+no\s+quiero|deja\s+eso\s+quieto|d[eé]jalo\s+quieto|(?:no\s+)?d[eé]jal(?:o|a|os|as)\s+as[ií])\b",
        t_cancel,
    ):
        return "cancelar"

    # 2. Consulta informativa intermedia (prioridad sobre llenado de ranura)
    if esperando_slot in ("stock_inicial", "stock_minimo", "cantidad"):
        # Respuesta numérica corta (incluye «que diez», «¿Qué diez?», «que sean diez», «diez unidades»)
        t_num_cand = re.sub(
            r"^(?:(?:y\s+)?(?:mejor\s+)?(?:qu[eé]\s+(?:sean?\s+)?|pon(?:le|er)?\s+|sean?\s+|son\s+|unos?\s+|unas?\s+))+"
            r"|\s+(?:unidades?|piezas?|art[ií]culos?)$",
            " ",
            t_clean,
        ).strip()
        if t_num_cand and _extraer_numero(t_num_cand, permitir_cero=True, detectar_negativo=True, solo_numero_exacto=True) is not None:
            return "ranura"

    es_alta_producto_pend = str(op_pendiente.get("tipo") or "").upper() == "CREAR_PRODUCTO"
    tiene_signo_pregunta = "?" in t_raw or "¿" in t_raw
    tiene_patron_prestable = bool(
        re.search(
            r"\b(?:prestable(?:s)?|pr[eé]stable(?:s)?|consumible(?:s)?|retornable(?:s)?|se\s+puede\s+prestar|se\s+presta)\b",
            t_clean,
        )
    )
    # Durante el alta de un producto, declarar «es prestable» / «es consumible» es un atributo del producto nuevo,
    # NO una consulta informativa intermedia (a menos que sea una pregunta explícita con «?» o «cuáles/qué/dime si»).
    es_consulta_prestable = tiene_patron_prestable and (
        not es_alta_producto_pend
        or tiene_signo_pregunta
        or bool(re.search(r"\b(?:cu[aá]l(?:es)?|qu[eé]\s+productos?|dime\s+si|me\s+confirm[ae]s?\s+si|conf[ií]rmame\s+si)\b", t_clean))
    )
    tiene_patron_consulta = es_consulta_prestable or bool(
        re.search(
            r"\b("
            r"cu[aá]nt[oa]s?|d[oó]nde\s+est[aá]|d[oó]nde\s+se\s+encuentra|d[oó]nde\s+queda|"
            r"en\s+qu[eé]\s+(?:ubicaci[oó]n|ubicaciones|estantes?|almac[eé]n|almacenes|bodegas?|categor[ií]as?)|"
            r"cu[aá]l\s+es\s+(?:su|la|el)\s+(?:ubicaci[oó]n|stock|c[oó]digo|categor[ií]a|almac[eé]n)|"
            r"cu[aá]les?\s+(?:son|hay|existen|tenemos|est[aá]n)|"
            r"qu[eé]\s+(?:stock|ubicaci[oó]n|ubicaciones|estantes?|almac[eé]n|almacenes|bodegas?|productos?|referencias?|categor[ií]as?|entradas?|salidas?|movimientos?|pr[eé]stamos?|devoluciones?|traslados?)|"
            r"(?:lista|listado)\s+de\s+(?:almacenes|bodegas|ubicaciones|estantes|categor[ií]as|productos|pr[eé]stamos|movimientos|entradas|salidas|traslados|devoluciones)|"
            r"(?:mis\s+)?(?:almacenes|bodegas)\s+(?:asignad[oa]s?|disponibles?|del\s+sistema|activ[oa]s?)|"
            r"(?:quiero|necesito)\s+(?:ver|revisar|consultar|saber)\b|"
            r"qui[eé]n(?:es)?\s+(?:lo\s+|la\s+|los\s+|las\s+)?tiene(?:n)?|a\s+qui[eé]n\s+est[aá]\s+prest|"
            r"hay\s+disponibles?|quedan?\s+disponibles?|"
            r"stock\s+(?:actual|disponible|de|del|en)|existencias?\s+(?:de|del|en)|"
            r"me\s+confirm[ae]s?\s+si|conf[ií]rmame\s+si|"
            r"agotad[oa]s?|sin\s+stock|bajo\s+stock|por\s+debajo\s+del?\s+(?:stock\s+)?m[ií]nimo|pelad[oa]s?|escas[oa]s?|"
            r"pr[eé]stamos?\s+(?:activos?|vigentes?|pendientes?)|"
            r"ficha\s+de|informaci[oó]n\s+de|detalles?\s+de|"
            r"antes\s+dime|primero\s+dime|dime\s+(?:cu[aá]nt[oa]s?|d[oó]nde|qu[eé]|cu[aá]l|qui[eé]n|si\s+)|"
            r"mu[eé]strame|mostrame|m[aáeé]s?trame|b[uú]scame|consulta(?:r)?\s+(?:el\s+stock|la\s+ubicaci[oó]n|cu[aá]nto|los\s+movimientos?)|"
            r"saber\s+(?:cu[aá]nt[oa]s?|d[oó]nde|qu[eé]|cu[aá]l(?:es)?|si\s+)|revisa(?:r)?\s+cu[aá]nt[oa]s?|"
            r"(?:entradas?|salidas?|movimientos?|pr[eé]stamos?|devoluciones?|traslados?)\s+(?:hubo|de\s+hoy|de\s+ayer|de\s+esta\s+semana|de\s+este\s+mes)"
            r")\b",
            t_clean,
        )
    )
    if tiene_patron_consulta or (
        tiene_signo_pregunta
        and re.search(
            r"\b(hay|quedan?|queda|tiene|tienen|est[aá]|est[aá]n|disponibles?|stock|ubicaci[oó]n|ubicaciones|estantes?|almac[eé]n|almacenes|bodegas?|"
            r"prestable|pr[eé]stable|consumible|retornable|prestar|qui[eé]n|cu[aá]nt[oa]s?|d[oó]nde|cu[aá]l(?:es)?|qu[eé]|"
            r"entradas?|salidas?|movimientos?|pr[eé]stamos?|categor[ií]as?|c[oó]digo)\b",
            t_clean,
        )
    ):
        return "consulta_intermedia"

    # 3. Nueva operación explícita (de tipo distinto, o un comando completo nuevo que reinicia el flujo)
    if not re.match(r"^(?:qu[eé]|cu[aá]nt[oa]s?|cu[aá]les?|c[oó]mo|puedo|podemos|se\s+puede[n]?|no\s+(?:quiero|deseo|hagas?|saques?|retires?|ingreses?|registres?|prestes?|traslades?|devuelvas?))\b", t_clean):
        tipo_actual = str(op_pendiente.get("tipo") or "").upper()
        mapa_nuevas_ops = [
            ("REGISTRAR_ENTRADA", r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:(?:registr[aeoó]\w*|gener[aeoó]\w*|hac(?:er|e)|da(?:r|me)|cre[aeoó]\w*)\s+(?:una?\s+)?entrada|ingres[aeoó]\w*\s+(?:stock|\d+|un[oa]?\b|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez))\b"),
            ("REGISTRAR_SALIDA", r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:(?:registr[aeoó]\w*|gener[aeoó]\w*|hac(?:er|e)|da(?:r|me)|cre[aeoó]\w*)\s+(?:una?\s+)?salida|(?:sacar|retirar|despachar)\s+\d+)\b"),
            ("PRESTAMO", r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:(?:hacer|hac[eé]|registr[aeoó]\w*|gener[aeoó]\w*)\s+un\s+pr[eé]stamo|prestar(?:le)?\b)"),
            ("TRASLADO_ENTRE", r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:(?:hacer|registr[aeoó]\w*|gener[aeoó]\w*)\s+un\s+traslado|trasladar\b)"),
            ("DEVOLUCION", r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:(?:registr[aeoó]\w*|gener[aeoó]\w*|hacer)\s+(?:una?\s+)?devoluci[oó]n|devolver\b)"),
            ("INHABILITAR_PRODUCTO", r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:inhabilitar|desactivar|archivar)\b"),
            ("HABILITAR_PRODUCTO", r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:habilitar|reactivar)\b"),
            ("CREAR_PRODUCTO", r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:crear\s+(?:un\s+)?(?:nuevo\s+)?producto|dar\s+de\s+alta|nuevo\s+producto|registrar\s+(?:un\s+)?nuevo\s+producto)\b"),
        ]
        es_comando_completo_nuevo = bool(
            re.search(
                r"\b(?:gener[aeoó]\w*|registr[aeoó]\w*|cre[aeoó]\w*)\s+(?:una?\s+)?(?:entrada|salida|pr[eé]stamo|traslado|devoluci[oó]n)\b",
                t_clean,
            )
            and (
                re.search(r"\b(?:\d+|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s+unidades?\b", t_clean)
                or re.search(r"\b[a-z]{2,5}[-_\s]\d{1,5}\b", t_low)
            )
        )
        for tipo_cand, pat_op in mapa_nuevas_ops:
            if re.search(pat_op, t_clean):
                if tipo_cand != tipo_actual or es_comando_completo_nuevo:
                    return "nueva_operacion"

    return "ranura"


def procesar_comando_inventario(
    usuario: UsuarioAudacIA,
    mensaje: str,
    propuesta_activa: dict | None = None,
    sesion: dict | None = None,
    estado_operativo: Any = None,
    interaction_id: str | None = None,
    conversacion_id: str | None = None,
    cliente_llm: Any = None,
) -> dict[str, Any]:
    """Interpreta el mensaje del usuario mediante la capa lingüística desacoplada.

    Flujo obligatorio:
    Audio -> Whisper -> Normalización lingüística -> Intención -> Confirmación Haku -> API Inventario

    Retorna un diccionario con:
    - tipo: "consulta" | "propuesta" | "aclaracion" | "accion_confirmar" | "accion_cancelar" | "error" | "conversacion"
    - mensaje_pantalla: texto formateado para la interfaz
    - mensaje_verbal: texto optimizado para síntesis de voz (sin markdown)
    - propuesta: dict con datos de la propuesta (si tipo == "propuesta")
    - datos: dict adicional para renderizado visual
    """
    from .contexto import HakuOperationalContext

    if conversacion_id and sesion is not None:
        if isinstance(sesion, dict):
            sesion["haku_conversacion_activa_id"] = conversacion_id
        elif hasattr(sesion, "__setitem__"):
            try:
                sesion["haku_conversacion_activa_id"] = conversacion_id
            except Exception:
                pass

    if estado_operativo is None and usuario and getattr(usuario, "id", None):
        session_key = None
        if sesion is not None:
            if hasattr(sesion, "session_key") and sesion.session_key:
                session_key = str(sesion.session_key)
            elif isinstance(sesion, dict) and sesion.get("session_key"):
                session_key = str(sesion.get("session_key"))
            else:
                if hasattr(sesion, "save") and callable(getattr(sesion, "save", None)):
                    try:
                        sesion.save()
                        session_key = getattr(sesion, "session_key", None)
                    except Exception:
                        pass
                if not session_key:
                    session_key = f"sess_{usuario.id}_{uuid.uuid4().hex[:12]}"
                    if isinstance(sesion, dict):
                        sesion["session_key"] = session_key
                    elif hasattr(sesion, "session_key"):
                        try:
                            sesion.session_key = session_key
                        except Exception:
                            pass
        else:
            session_key = f"sess_transient_{usuario.id}_{uuid.uuid4().hex[:12]}"
        almacen_id = getattr(usuario, "almacen_asignado_id", None)
        estado_operativo = HakuOperationalContext.obtener_o_crear(usuario.id, session_key, almacen_id)

    ctx_resguardo: dict[str, Any] = {
        "es_consulta_intermedia": False,
        "op_pendiente_resguardo": None,
        "foco_resguardo": None,
    }
    resultado = _procesar_comando_inventario_interno(
        usuario=usuario,
        mensaje=mensaje,
        propuesta_activa=propuesta_activa,
        sesion=sesion,
        estado_operativo=estado_operativo,
        interaction_id=interaction_id,
        ctx_resguardo=ctx_resguardo,
        conversacion_id=conversacion_id,
        cliente_llm=cliente_llm,
    )
    if ctx_resguardo.get("es_consulta_intermedia") and ctx_resguardo.get("op_pendiente_resguardo"):
        op_resg = ctx_resguardo["op_pendiente_resguardo"]
        foco_resg = ctx_resguardo.get("foco_resguardo") or {}

        # C-04: Verificar si op_resg corresponda a una propuesta ya terminada (CANCELADA / EJECUTADA / SUSTITUIDA)
        prop_id_resg = op_resg.get("propuesta_id")
        op_obsoleta = False
        if sesion is not None:
            historial = sesion.get("haku_historial_propuestas_terminales") or {}
            ultimo_evento = sesion.get("haku_ultimo_evento") or {}
            if prop_id_resg:
                if prop_id_resg in historial:
                    op_obsoleta = True
                elif ultimo_evento.get("propuesta_id") == prop_id_resg and ultimo_evento.get("estado") in {"CANCELADA", "EJECUTADA", "SUSTITUIDA"}:
                    op_obsoleta = True

        if isinstance(resultado, dict) and resultado.get("tipo") == "propuesta":
            # Una consulta informativa intermedia jamás debe generar una propuesta de escritura
            if estado_operativo:
                estado_operativo.propuesta_activa = None
            if sesion is not None and isinstance(sesion, dict):
                if propuesta_activa:
                    sesion["haku_propuesta_activa"] = propuesta_activa
                else:
                    sesion["haku_propuesta_activa"] = None
            resultado = {
                "tipo": "aclaracion",
                "esperando": op_resg.get("esperando"),
                "mensaje_pantalla": "Para continuar con la operación pendiente, por favor responda al dato solicitado o indique si desea cancelarla.",
                "mensaje_verbal": normalizar_para_tts("Para continuar con la operación pendiente, responda al dato solicitado o indique si desea cancelarla."),
                "datos": {},
            }
        if op_obsoleta:
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
        else:
            if estado_operativo:
                estado_operativo.operacion_pendiente = op_resg
                if op_resg.get("producto_id") and not estado_operativo.producto_id:
                    p_orig = Producto.objects.filter(id=op_resg.get("producto_id")).first()
                    if p_orig:
                        estado_operativo.producto_id = p_orig.id
                        estado_operativo.producto_codigo = p_orig.codigo_interno
                        estado_operativo.producto_nombre = p_orig.nombre
                elif foco_resg.get("producto_id") and not estado_operativo.producto_id:
                    estado_operativo.producto_id = foco_resg["producto_id"]
                    estado_operativo.producto_codigo = foco_resg["producto_codigo"]
                    estado_operativo.producto_nombre = foco_resg["producto_nombre"]
                HakuOperationalContext.guardar(estado_operativo)
            if isinstance(resultado, dict):
                if "datos" not in resultado or not isinstance(resultado["datos"], dict):
                    resultado["datos"] = {}
                resultado["datos"]["operacion_pendiente_preservada"] = True
                resultado["datos"]["esperando"] = op_resg.get("esperando")
    if isinstance(resultado, dict) and resultado.get("tipo") in ("consulta", "exito"):
        datos_respuesta = resultado.get("datos")
        if isinstance(datos_respuesta, dict) and not isinstance(datos_respuesta.get("productos"), list):
            producto_id = datos_respuesta.get("producto_id")
            producto = Producto.objects.filter(pk=producto_id).first() if producto_id else None
            if producto is not None:
                # Mantener el contrato estructurado también en la ruta
                # semántica heredada usada por selecciones ordinales.
                from .agente.herramientas import FiltrosConsulta, HerramientasInventario

                resultado_catalogo = HerramientasInventario(usuario).buscar_productos(
                    FiltrosConsulta(codigos=[producto.codigo_interno])
                )
                if resultado_catalogo.total == 1:
                    datos_respuesta.update({
                        "productos": resultado_catalogo.items,
                        "total": 1,
                        "resultados_completos": True,
                        "operacion_consulta": "detalle",
                    })
    return resultado


def _posicion_coincidencia_entidad(
    nombre: str,
    texto_lower: str,
    texto_norm: str = "",
    *,
    clave_canonica: str = "",
    norm_fn=None,
) -> tuple[int, int] | None:
    """Busca coincidencia por límites estrictos de palabra para evitar falsos positivos
    de subcadena (ej. «Almacén A» dentro de «almacén al», o «A1» dentro de «A10»).

    Retorna ``(fin_match, longitud_match)`` de la coincidencia más específica, o ``None``.
    """
    mejor: tuple[int, int] | None = None
    patrones: list[tuple[str, str]] = []
    nom_low = (nombre or "").strip().lower()
    if nom_low and texto_lower:
        patrones.append((nom_low, texto_lower))
    if norm_fn is not None and nom_low and texto_norm:
        nom_norm = (norm_fn(nom_low) or "").strip()
        if nom_norm:
            patrones.append((nom_norm, texto_norm))
    clave = (clave_canonica or "").strip().lower()
    if clave and texto_norm:
        patrones.append((clave, texto_norm))
    if clave and texto_lower and texto_lower != texto_norm:
        patrones.append((clave, texto_lower))

    for objetivo, fuente in patrones:
        for m in re.finditer(rf"(?<![\w]){re.escape(objetivo)}(?![\w])", fuente):
            cand = (m.end(), len(objetivo))
            if mejor is None or cand[1] > mejor[1] or (cand[1] == mejor[1] and cand[0] > mejor[0]):
                mejor = cand
    return mejor


def _seleccionar_entidad_en_texto(
    candidatos: list[Any],
    texto_lower: str,
    texto_norm: str = "",
    *,
    norm_fn=None,
    usar_clave_canonica: bool = False,
    preferir_ultimo: bool = False,
    excluir_id: Any = None,
):
    """Selecciona la entidad (Almacén, Categoría o Ubicación) que mejor coincide en el texto."""
    coincidencias: list[tuple[tuple[int, int], Any]] = []
    for item in candidatos:
        nombre = getattr(item, "nombre", "") or ""
        clave = (getattr(item, "clave_canonica", "") or "") if usar_clave_canonica else ""
        pos = _posicion_coincidencia_entidad(
            nombre,
            texto_lower,
            texto_norm,
            clave_canonica=clave,
            norm_fn=norm_fn,
        )
        if pos is not None:
            coincidencias.append((pos, item))
    if not coincidencias:
        return None
    if excluir_id is not None:
        sin_excluido = [c for c in coincidencias if getattr(c[1], "id", None) != excluir_id]
        if sin_excluido:
            coincidencias = sin_excluido
    if preferir_ultimo:
        coincidencias.sort(key=lambda par: (par[0][0], par[0][1]), reverse=True)
    else:
        coincidencias.sort(key=lambda par: (par[0][1], par[0][0]), reverse=True)
    return coincidencias[0][1]


def _resolver_ubicacion_en_almacen(
    texto_ubicacion: str,
    almacen_id: int | None = None,
) -> tuple[Any | None, list[Any], str]:
    """Resuelve una ubicación física por nombre administrable o clave canónica dentro de un almacén.

    Retorna (ubicacion, candidatos, estado):
    - ('ok', [ubi]) si hay coincidencia única.
    - ('ambigua', [ubi1, ubi2, ...]) si hay más de una coincidencia en el almacén.
    - ('otro_almacen', [ubi_otro]) si no está en el almacén pero existe en otro.
    - ('no_encontrada', []) si no existe.
    """
    from inventario.models import Ubicacion
    from inventario.utils import normalizar_texto as norm_txt

    if not texto_ubicacion or not isinstance(texto_ubicacion, str) or not texto_ubicacion.strip():
        return None, [], "no_encontrada"

    txt_raw = texto_ubicacion.strip()
    txt_low = txt_raw.lower()
    txt_norm = norm_txt(txt_raw)

    qs_base = Ubicacion.objects.filter(activo=True, eliminado=False)
    if almacen_id:
        qs_alm = list(qs_base.filter(almacen_id=almacen_id))
    else:
        qs_alm = list(qs_base)

    # 1. Coincidencia exacta por clave_canonica o nombre o nombre_normalizado
    exactos = [
        u for u in qs_alm
        if (getattr(u, "clave_canonica", None) and u.clave_canonica.lower() == txt_low)
        or u.nombre.lower() == txt_low
        or (getattr(u, "nombre_normalizado", None) and u.nombre_normalizado.lower() == txt_low)
        or (txt_norm and getattr(u, "nombre_normalizado", None) and norm_txt(u.nombre_normalizado) == txt_norm)
    ]
    if len(exactos) == 1:
        return exactos[0], [], "ok"
    if len(exactos) > 1:
        return None, exactos, "ambigua"

    # 2. Coincidencia por límites de palabra dentro del almacén (evita «A1» dentro de «A10»)
    parciales_exactos_en_texto: list[tuple[int, Any]] = []
    parciales_subfrase: list[Any] = []
    for u in qs_alm:
        u_nom_low = u.nombre.lower()
        u_norm = norm_txt(u.nombre)
        u_clave = (getattr(u, "clave_canonica", "") or "").lower()
        pos_en_txt = _posicion_coincidencia_entidad(
            u.nombre,
            txt_low,
            txt_norm,
            clave_canonica=u_clave,
            norm_fn=norm_txt,
        )
        if pos_en_txt is not None:
            parciales_exactos_en_texto.append((pos_en_txt[1], u))
            continue
        if len(txt_low) >= 3 and (
            re.search(rf"(?<![\w]){re.escape(txt_low)}(?![\w])", u_nom_low)
            or (txt_norm and len(txt_norm) >= 3 and u_norm and re.search(rf"(?<![\w]){re.escape(txt_norm)}(?![\w])", u_norm))
        ):
            parciales_subfrase.append(u)

    if parciales_exactos_en_texto:
        max_len_u = max(l for l, _ in parciales_exactos_en_texto)
        mejores_u = [u for l, u in parciales_exactos_en_texto if l == max_len_u]
        if len(mejores_u) == 1:
            return mejores_u[0], [], "ok"
        return None, mejores_u, "ambigua"

    if len(parciales_subfrase) == 1:
        return parciales_subfrase[0], [], "ok"
    if len(parciales_subfrase) > 1:
        return None, parciales_subfrase, "ambigua"

    # 3. Si no se encontró dentro del almacén pero se especificó almacen_id, verificar si está en otro almacén
    if almacen_id:
        otras = list(qs_base.exclude(almacen_id=almacen_id))
        otras_match = [
            u for u in otras
            if _posicion_coincidencia_entidad(
                u.nombre,
                txt_low,
                txt_norm,
                clave_canonica=(getattr(u, "clave_canonica", "") or ""),
                norm_fn=norm_txt,
            ) is not None
            or (
                len(txt_low) >= 3
                and (
                    re.search(rf"(?<![\w]){re.escape(txt_low)}(?![\w])", u.nombre.lower())
                    or (txt_norm and len(txt_norm) >= 3 and norm_txt(u.nombre) and re.search(rf"(?<![\w]){re.escape(txt_norm)}(?![\w])", norm_txt(u.nombre)))
                )
            )
        ]
        if otras_match:
            return None, otras_match, "otro_almacen"

    return None, [], "no_encontrada"


def _extraer_es_prestable_en_texto(texto: str) -> bool | None:
    """Detecta si el usuario indicó explícitamente que un producto es prestable/retornable o consumible."""
    t = (texto or "").strip().lower()
    if not t:
        return None
    if re.search(
        r"\b(?:no\s+(?:es\s+|sea\s+)?(?:prestable|retornable)|"
        r"no\s+se\s+(?:puede\s+)?prest\w+|no\s+admite\s+pr[eé]stamos?|"
        r"(?:es\s+)?consumible|desechable|uso\s+definitivo)\b",
        t,
    ):
        return False
    if re.search(
        r"\b(?:(?:es\s+)?prestable|(?:es\s+)?retornable|"
        r"se\s+(?:puede\s+)?prest\w+|admite\s+pr[eé]stamos?|"
        r"(?:para|de)\s+pr[eé]stamo)\b",
        t,
    ):
        return True
    return None


def _procesar_alta_producto_conversacional(
    *,
    texto: str,
    usuario: Any,
    estado_operativo: Any = None,
    sesion: Any = None,
    semantica: Any = None,
    op_pendiente: dict | None = None,
    conversacion_id: str | None = None,
) -> dict[str, Any]:
    """Gestiona de forma dinámica y segura el alta conversacional de productos (C-02).
    
    Aplica rigurosamente:
    - Validación de permisos RBAC: catalogo.crear y acceso al almacén.
    - Resolución dinámica de almacén, categoría y ubicación (regla DN-01).
    - Desambiguación de ubicaciones entre almacenes.
    - Prohibición de stock negativo (inicial o mínimo < 0).
    - Soporte explícito de stock inicial 0.
    - Prohibición de asumir stock mínimo 0 sin confirmación explícita del usuario.
    - Construcción de propuesta canónica firmada con HMAC v2.
    """
    from inventario.models import Almacen, Categoria, Ubicacion
    from inventario.utils import normalizar_texto as norm_txt
    from .contexto import HakuOperationalContext

    # 1. Validación previa de permiso catalogo.crear (C02-12)
    if not usuario_tiene_permiso_operativo(usuario, "catalogo.crear"):
        raise PermissionDenied("No cuentas con el permiso requerido (catalogo.crear) para registrar nuevos productos.")

    texto_strip = (texto or "").strip()
    texto_norm = norm_txt(texto_strip)
    texto_lower = texto_strip.lower()

    # 2. Verificación y rechazo inmediato de stock negativo (C02-19)
    pat_neg_ini = re.search(
        r"\b(?:stock\s+inicial|inicial|cantidad\s+inicial)\s*(?:de|en|:)?\s*((?:-\s*\d+|(?:menos|negativo)\s+[^\.,;\n]+))",
        texto_strip,
        re.I,
    )
    pat_neg_min = re.search(
        r"\b(?:stock\s+m[ií]nimo|m[ií]nimo|umbral\s+m[ií]nimo)\s*(?:de|en|:)?\s*((?:-\s*\d+|(?:menos|negativo)\s+[^\.,;\n]+))",
        texto_strip,
        re.I,
    )
    val_neg_ini = _extraer_numero(pat_neg_ini.group(1), detectar_negativo=True, solo_al_inicio=True) if pat_neg_ini else None
    val_neg_min = _extraer_numero(pat_neg_min.group(1), detectar_negativo=True, solo_al_inicio=True) if pat_neg_min else None
    if (val_neg_ini is not None and val_neg_ini < 0) or (val_neg_min is not None and val_neg_min < 0):
        msg = "El stock inicial y el stock mínimo deben ser números enteros mayores o iguales a cero (no se permiten números negativos)."
        return {
            "tipo": "error",
            "mensaje_pantalla": f"⚠️ {msg}",
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"error": "stock_negativo"},
        }

    esperando_previo = op_pendiente.get("esperando") if op_pendiente else None
    if esperando_previo in ("stock_inicial", "stock_minimo"):
        val_esperado_neg = _extraer_numero(texto_strip, permitir_cero=True, detectar_negativo=True)
        if val_esperado_neg is not None and val_esperado_neg < 0:
            msg = "El stock inicial y el stock mínimo deben ser números enteros mayores o iguales a cero (no se permiten números negativos)."
            return {
                "tipo": "error",
                "mensaje_pantalla": f"⚠️ {msg}",
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"error": "stock_negativo"},
            }

    if semantica is not None:
        if semantica.cantidad is not None and semantica.cantidad < 0:
            msg = "El stock inicial y el stock mínimo deben ser números enteros mayores o iguales a cero (no se permiten números negativos)."
            return {
                "tipo": "error",
                "mensaje_pantalla": f"⚠️ {msg}",
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"error": "stock_negativo"},
            }
        de_s_pre = getattr(semantica, "datos_extra", {}) or {}
        if de_s_pre.get("stock_minimo") is not None and int(de_s_pre["stock_minimo"]) < 0:
            msg = "El stock inicial y el stock mínimo deben ser números enteros mayores o iguales a cero (no se permiten números negativos)."
            return {
                "tipo": "error",
                "mensaje_pantalla": f"⚠️ {msg}",
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"error": "stock_negativo"},
            }

    # 3. Recuperar datos acumulados de la sesión / estado operativo
    datos: dict[str, Any] = dict(op_pendiente.get("datos_producto", {})) if op_pendiente else {}
    nota_correccion_campo = ""

    # 3-bis. Corrección dinámica en medio del flujo (ej. «Olvídalo, que estén en Almacén B», «Mejor en Almacén C»)
    cambio_almacen_aplicado = False
    if esperando_previo in ("categoria", "ubicacion", "stock_inicial", "stock_minimo"):
        # Detectar si el usuario nombra explícitamente otro almacén para corregir su selección previa
        menciona_almacen_o_cambio = bool(
            re.search(
                r"\b(?:almac[eé]n|bodega|olv[ií]dalo|mejor|cambi[aeoá]\w*|pon(?:lo|la|los|las)?|"
                r"que\s+est[eé]n?|en\s+vez\s+de|no[,\s]+(?:en|que\s+est[eé]n?))\b",
                texto_lower,
            )
        )
        if menciona_almacen_o_cambio:
            alms_todos = list(Almacen.objects.filter(activo=True, eliminado=False))
            alm_cambio = _seleccionar_entidad_en_texto(
                alms_todos,
                texto_lower,
                texto_norm,
                norm_fn=norm_txt,
                preferir_ultimo=True,
                excluir_id=datos.get("almacen_id"),
            )
            if alm_cambio is not None and alm_cambio.id != datos.get("almacen_id"):
                if not usuario.tiene_acceso_almacen(alm_cambio):
                    raise PermissionDenied(f"No cuenta con acceso al almacén '{alm_cambio.nombre}'.")
                datos["almacen_id"] = alm_cambio.id
                datos["almacen_nombre"] = alm_cambio.nombre
                datos.pop("categoria_id", None)
                datos.pop("categoria_nombre", None)
                datos.pop("ubicacion_id", None)
                datos.pop("ubicacion_nombre", None)
                datos.pop("_ubicacion_ambigua", None)
                cambio_almacen_aplicado = True
                nota_correccion_campo = f"Entendido, cambié el almacén a {alm_cambio.nombre}. "

    # 4. Procesar ranura específica si estábamos esperando un campo particular
    if cambio_almacen_aplicado:
        pass
    elif esperando_previo == "nombre":
        n_cand = texto_strip.strip('"\' \t\r\n.,;:!?¿¡')
        marcador = re.search(
            r"\b(?:se\s+llama(?:r[aá])?|se\s+va\s+a\s+llamar|llamad[oa]|denominad[oa]|"
            r"(?:el\s+)?nombre\s+(?:es|ser[ií]a)|de\s+nombre|con\s+nombre|cuyo\s+nombre\s+sea)\s+(.+)",
            n_cand,
            flags=re.IGNORECASE,
        )
        if marcador:
            n_cand = marcador.group(1).strip('"\' \t\r\n.,;:!?¿¡')
        cortes_campos = [
            re.search(patron, n_cand, flags=re.IGNORECASE)
            for patron in (
                r"[,;\.]\s*(?:y\s+)?(?:lo\s+vas\s+a|vas\s+a|gu[aá]rdal[oa]|ponl[oa]|ub[ií]cal[oa]|m[eé]tel[oa]|va\s+(?:en|para)|est[aá](?:r[aá])?\s+en|la\s+categor[ií]a|el\s+almac[eé]n|la\s+ubicaci[oó]n|el\s+stock)\b",
                r"\.\s+",
                r"\s+y\s+(?:lo\s+vas\s+a|vas\s+a|la\s+categor[ií]a|el\s+almac[eé]n|la\s+ubicaci[oó]n|el\s+stock|con\s+stock)\b",
                r"[,;]\s*(?=(?:con\s+)?\d+\s+(?:unidades?\s+)?inicial(?:es)?\b)",
                r"\b(?:stock\s+(?:inicial|m[ií]nimo)|cantidad\s+inicial|"
                r"existencias?\s+iniciales?|m[ií]nimo|umbral\s+m[ií]nimo)\b",
                r"(?:[,;]\s*|\s+)(?:(?:y\s+)?(?:no\s+)?(?:es\s+)?(?:prestable|retornable|consumible)|se\s+puede\s+prestar)\b",
                r"(?:[,;]\s*|\s+)(?:en\s+(?:(?:la|el)\s+)?)?(?:categor[ií]a|almac[eé]n|ubicaci[oó]n)\b",
            )
        ]
        limites_nombre = [m.start() for m in cortes_campos if m]
        if limites_nombre:
            n_cand = n_cand[:min(limites_nombre)].strip(" ,;:-.!?")
        if len(n_cand) >= 2:
            datos["nombre"] = n_cand
    elif esperando_previo == "almacen":
        alms_acc = list(usuario.obtener_almacenes_accesibles() if hasattr(usuario, "obtener_almacenes_accesibles") else Almacen.objects.filter(activo=True, eliminado=False))
        alm_match = _seleccionar_entidad_en_texto(
            alms_acc,
            texto_lower,
            texto_norm,
            norm_fn=norm_txt,
            preferir_ultimo=True,
        )
        if not alm_match and texto_strip.isdigit():
            idx = int(texto_strip) - 1
            if 0 <= idx < len(alms_acc):
                alm_match = alms_acc[idx]
        if alm_match:
            datos["almacen_id"] = alm_match.id
            datos["almacen_nombre"] = alm_match.nombre
        else:
            # Revisar si nombró un almacén existente al cual no tiene permiso (C02-13)
            alm_inacc = _seleccionar_entidad_en_texto(
                list(Almacen.objects.filter(activo=True, eliminado=False)),
                texto_lower,
                texto_norm,
                norm_fn=norm_txt,
                preferir_ultimo=True,
            )
            if alm_inacc is not None and not usuario.tiene_acceso_almacen(alm_inacc):
                raise PermissionDenied(f"No cuenta con acceso al almacén '{alm_inacc.nombre}'.")

            # Inferencia dinámica por Categoría o Ubicación: si el usuario indica directamente una
            # categoría o ubicación física (porque no recuerda en qué almacén se encuentra),
            # deducir automáticamente su almacén desde la BD respetando RBAC.
            cats_match = [
                c
                for c in Categoria.objects.filter(activo=True, eliminado=False).select_related("almacen_principal")
                if _posicion_coincidencia_entidad(c.nombre, texto_lower, texto_norm, norm_fn=norm_txt) is not None
            ]
            if len(cats_match) > 1:
                max_len_c = max(len(c.nombre) for c in cats_match)
                cats_match = [c for c in cats_match if len(c.nombre) == max_len_c]
            if len(cats_match) == 1 and cats_match[0].almacen_principal:
                cat_inf = cats_match[0]
                alm_inf = cat_inf.almacen_principal
                if not usuario.tiene_acceso_almacen(alm_inf):
                    raise PermissionDenied(
                        f"La categoría '{cat_inf.nombre}' pertenece a '{alm_inf.nombre}', almacén al cual no cuenta con acceso."
                    )
                datos["almacen_id"] = alm_inf.id
                datos["almacen_nombre"] = alm_inf.nombre
                datos["categoria_id"] = cat_inf.id
                datos["categoria_nombre"] = cat_inf.nombre
                nota_correccion_campo = (
                    f"Asigné la categoría '{cat_inf.nombre}' y automáticamente su almacén ({alm_inf.nombre}). "
                )
            else:
                ubis_match = [
                    u
                    for u in Ubicacion.objects.filter(activo=True, eliminado=False).select_related("almacen")
                    if _posicion_coincidencia_entidad(
                        u.nombre,
                        texto_lower,
                        texto_norm,
                        clave_canonica=(getattr(u, "clave_canonica", "") or ""),
                        norm_fn=norm_txt,
                    ) is not None
                ]
                if len(ubis_match) > 1:
                    max_len_u = max(len(u.nombre) for u in ubis_match)
                    ubis_match = [u for u in ubis_match if len(u.nombre) == max_len_u]
                if len(ubis_match) == 1 and ubis_match[0].almacen:
                    ubi_inf = ubis_match[0]
                    alm_inf = ubi_inf.almacen
                    if not usuario.tiene_acceso_almacen(alm_inf):
                        raise PermissionDenied(
                            f"La ubicación '{ubi_inf.nombre}' pertenece a '{alm_inf.nombre}', almacén al cual no cuenta con acceso."
                        )
                    datos["almacen_id"] = alm_inf.id
                    datos["almacen_nombre"] = alm_inf.nombre
                    datos["ubicacion_id"] = ubi_inf.id
                    datos["ubicacion_nombre"] = ubi_inf.nombre
                    nota_correccion_campo = (
                        f"Asigné la ubicación '{ubi_inf.nombre}' y automáticamente su almacén ({alm_inf.nombre}). "
                    )
                else:
                    noms_a = [a.nombre for a in alms_acc]
                    msg = (
                        f"El almacén '{texto_strip}' no existe en esta sede o no cuenta con acceso. "
                        f"Almacenes disponibles: {', '.join(noms_a)}. También puede indicar directamente la categoría."
                    )
                    return {
                        "tipo": "aclaracion",
                        "esperando": "almacen",
                        "mensaje_pantalla": msg,
                        "mensaje_verbal": normalizar_para_tts(msg),
                        "datos": {
                            "esperando": "almacen",
                            "operacion_pendiente": "CREAR_PRODUCTO",
                            "datos_producto": {k: v for k, v in datos.items() if not str(k).startswith("_")},
                            "candidatos": [{"id": a.id, "nombre": a.nombre} for a in alms_acc],
                        },
                    }
    elif esperando_previo == "categoria":
        alm_id_actual = datos.get("almacen_id")
        if alm_id_actual:
            cats_qs = list(Categoria.objects.filter(almacen_principal_id=alm_id_actual, activo=True, eliminado=False))
        else:
            cats_qs = list(Categoria.objects.filter(activo=True, eliminado=False))
        cat_match = _seleccionar_entidad_en_texto(
            cats_qs,
            texto_lower,
            texto_norm,
            norm_fn=norm_txt,
        )
        if not cat_match and texto_strip.isdigit():
            idx = int(texto_strip) - 1
            if 0 <= idx < len(cats_qs):
                cat_match = cats_qs[idx]
        if cat_match:
            datos["categoria_id"] = cat_match.id
            datos["categoria_nombre"] = cat_match.nombre
        else:
            # Verificar si existe en otro almacén (DN-01 / C02-11)
            cat_otra = _seleccionar_entidad_en_texto(
                list(Categoria.objects.filter(activo=True, eliminado=False).select_related("almacen_principal")),
                texto_lower,
                texto_norm,
                norm_fn=norm_txt,
            )
            if cat_otra and alm_id_actual and cat_otra.almacen_principal_id != alm_id_actual:
                alm_nom_act = datos.get("almacen_nombre")
                noms_c = [c.nombre for c in cats_qs]
                msg = (
                    f"La categoría '{cat_otra.nombre}' pertenece al almacén '{cat_otra.almacen_principal.nombre}', "
                    f"pero estás registrando en '{alm_nom_act}'. "
                    f"Categorías disponibles para '{alm_nom_act}': {', '.join(noms_c)}."
                )
                return {
                    "tipo": "aclaracion",
                    "esperando": "categoria",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts(msg),
                    "datos": {"esperando": "categoria", "candidatos": [{"id": c.id, "nombre": c.nombre} for c in cats_qs]},
                }
            noms_c = [c.nombre for c in cats_qs]
            msg = f"No reconocí la categoría '{texto_strip}'. Categorías disponibles para {datos.get('almacen_nombre', 'el almacén')}: {', '.join(noms_c)}."
            return {
                "tipo": "aclaracion",
                "esperando": "categoria",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"esperando": "categoria", "candidatos": [{"id": c.id, "nombre": c.nombre} for c in cats_qs]},
            }
    elif esperando_previo == "ubicacion":
        alm_id_actual = datos.get("almacen_id")
        u_res, cands, st = _resolver_ubicacion_en_almacen(texto_strip, alm_id_actual)
        if st == "ok" and u_res:
            datos["ubicacion_id"] = u_res.id
            datos["ubicacion_nombre"] = u_res.nombre
            datos.pop("_ubicacion_ambigua", None)
        elif st == "ambigua":
            noms_c = [u.nombre for u in cands]
            msg = f"Encontré varias ubicaciones posibles: {', '.join(noms_c)}. Por favor especifique exactamente a cuál se refiere."
            if estado_operativo:
                estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "ubicacion", "datos_producto": datos}
                HakuOperationalContext.guardar(estado_operativo)
            return {
                "tipo": "aclaracion",
                "esperando": "ubicacion",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"esperando": "ubicacion", "candidatos": [{"id": u.id, "nombre": u.nombre} for u in cands]},
            }
        elif st == "otro_almacen":
            u_otra = cands[0]
            ubis_qs = list(Ubicacion.objects.filter(almacen_id=alm_id_actual, activo=True, eliminado=False)) if alm_id_actual else []
            noms_u = [u.nombre for u in ubis_qs]
            msg = f"La ubicación '{u_otra.nombre}' pertenece a '{u_otra.almacen.nombre}', no al almacén '{datos.get('almacen_nombre')}'. Ubicaciones disponibles en {datos.get('almacen_nombre')}: {', '.join(noms_u)}."
            return {
                "tipo": "aclaracion",
                "esperando": "ubicacion",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"esperando": "ubicacion", "candidatos": [{"id": u.id, "nombre": u.nombre} for u in ubis_qs]},
            }
        else:
            ubis_qs = list(Ubicacion.objects.filter(almacen_id=alm_id_actual, activo=True, eliminado=False)) if alm_id_actual else []
            if texto_strip.isdigit():
                idx = int(texto_strip) - 1
                if 0 <= idx < len(ubis_qs):
                    datos["ubicacion_id"] = ubis_qs[idx].id
                    datos["ubicacion_nombre"] = ubis_qs[idx].nombre
                    datos.pop("_ubicacion_ambigua", None)
            if not datos.get("ubicacion_id"):
                noms_u = [u.nombre for u in ubis_qs]
                msg = f"No reconocí la ubicación '{texto_strip}'. Ubicaciones disponibles en {datos.get('almacen_nombre', 'el almacén')}: {', '.join(noms_u)}."
                return {
                    "tipo": "aclaracion",
                    "esperando": "ubicacion",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts(msg),
                    "datos": {"esperando": "ubicacion", "candidatos": [{"id": u.id, "nombre": u.nombre} for u in ubis_qs]},
                }
    elif esperando_previo == "stock_inicial":
        if re.search(r"\b(?:0|cero|ningun[oa]|nada|sin\s+stock|no\s+(?:hay|tengo|tenemos))\b", texto_strip, re.I):
            datos["stock_inicial"] = 0
        else:
            num_ini = _extraer_numero(texto_strip, permitir_cero=True)
            if num_ini is not None and num_ini >= 0:
                datos["stock_inicial"] = num_ini
    elif esperando_previo == "stock_minimo":
        if re.search(r"\b(?:0|cero)\b", texto_strip, re.I):
            datos["stock_minimo"] = 0
        else:
            num_min = _extraer_numero(texto_strip, permitir_cero=True)
            if num_min is not None and num_min >= 0:
                datos["stock_minimo"] = num_min

    # 5. Extracción complementaria desde semántica y texto completo
    if semantica is not None:
        if not datos.get("nombre") and semantica.producto:
            datos["nombre"] = semantica.producto.strip().strip('"\' \t\r\n.,;:!?¿¡')
        if not datos.get("almacen_id") and semantica.almacen:
            alm_s = _buscar_almacen_por_nombre(semantica.almacen)
            if alm_s:
                if not usuario.tiene_acceso_almacen(alm_s):
                    raise PermissionDenied(f"No cuenta con acceso al almacén '{alm_s.nombre}'.")
                datos["almacen_id"] = alm_s.id
                datos["almacen_nombre"] = alm_s.nombre
        de_s = getattr(semantica, "datos_extra", {}) or {}
        if not datos.get("categoria_id") and de_s.get("categoria"):
            c_nom = str(de_s["categoria"]).strip()
            cat_s = Categoria.objects.filter(nombre__iexact=c_nom, activo=True, eliminado=False).select_related("almacen_principal").first()
            if not cat_s and norm_txt(c_nom):
                c_nom_norm = norm_txt(c_nom)
                cat_s = next(
                    (
                        c for c in Categoria.objects.filter(activo=True, eliminado=False).select_related("almacen_principal")
                        if norm_txt(c.nombre) == c_nom_norm
                    ),
                    None,
                )
            if cat_s:
                datos["categoria_id"] = cat_s.id
                datos["categoria_nombre"] = cat_s.nombre
                if not datos.get("almacen_id") and cat_s.almacen_principal:
                    if not usuario.tiene_acceso_almacen(cat_s.almacen_principal):
                        raise PermissionDenied(
                            f"La categoría '{cat_s.nombre}' pertenece a '{cat_s.almacen_principal.nombre}', almacén al cual no cuenta con acceso."
                        )
                    datos["almacen_id"] = cat_s.almacen_principal_id
                    datos["almacen_nombre"] = cat_s.almacen_principal.nombre
        if not datos.get("ubicacion_id") and de_s.get("ubicacion"):
            u_nom = str(de_s["ubicacion"]).strip()
            u_res, cands, st = _resolver_ubicacion_en_almacen(u_nom, datos.get("almacen_id"))
            if st == "ok" and u_res:
                datos["ubicacion_id"] = u_res.id
                datos["ubicacion_nombre"] = u_res.nombre
                if not datos.get("almacen_id") and u_res.almacen:
                    if not usuario.tiene_acceso_almacen(u_res.almacen):
                        raise PermissionDenied(
                            f"La ubicación '{u_res.nombre}' pertenece a '{u_res.almacen.nombre}', almacén al cual no cuenta con acceso."
                        )
                    datos["almacen_id"] = u_res.almacen_id
                    datos["almacen_nombre"] = u_res.almacen.nombre
            elif st == "ambigua":
                datos["_ubicacion_ambigua"] = cands
        if datos.get("stock_inicial") is None:
            if de_s.get("stock_inicial") is not None:
                datos["stock_inicial"] = int(de_s["stock_inicial"])
            elif semantica.cantidad is not None and semantica.tipo_operacion == "CREAR_PRODUCTO":
                datos["stock_inicial"] = int(semantica.cantidad)
        if datos.get("stock_minimo") is None and de_s.get("stock_minimo") is not None:
            datos["stock_minimo"] = int(de_s["stock_minimo"])
        if de_s.get("codigo_barras"):
            datos["codigo_barras"] = str(de_s["codigo_barras"]).strip()
        if isinstance(de_s.get("es_prestable"), bool):
            datos["es_prestable"] = de_s["es_prestable"]

    # Extracción por patrones directos en texto
    # A) Almacén
    if not datos.get("almacen_id"):
        alm_directo = _seleccionar_entidad_en_texto(
            list(Almacen.objects.filter(activo=True, eliminado=False)),
            texto_lower,
            texto_norm,
            norm_fn=norm_txt,
            preferir_ultimo=True,
        )
        if alm_directo is not None:
            if not usuario.tiene_acceso_almacen(alm_directo):
                raise PermissionDenied(f"No cuenta con acceso al almacén '{alm_directo.nombre}'.")
            datos["almacen_id"] = alm_directo.id
            datos["almacen_nombre"] = alm_directo.nombre

    # B) Categoría
    if not datos.get("categoria_id"):
        cats_pool = Categoria.objects.filter(activo=True, eliminado=False).select_related("almacen_principal")
        if datos.get("almacen_id"):
            cats_pool = cats_pool.filter(almacen_principal_id=datos["almacen_id"])
        cat_directa = _seleccionar_entidad_en_texto(
            list(cats_pool),
            texto_lower,
            texto_norm,
            norm_fn=norm_txt,
        )
        if cat_directa is not None:
            datos["categoria_id"] = cat_directa.id
            datos["categoria_nombre"] = cat_directa.nombre
            if not datos.get("almacen_id") and cat_directa.almacen_principal:
                if not usuario.tiene_acceso_almacen(cat_directa.almacen_principal):
                    raise PermissionDenied(
                        f"La categoría '{cat_directa.nombre}' pertenece a '{cat_directa.almacen_principal.nombre}', almacén al cual no cuenta con acceso."
                    )
                datos["almacen_id"] = cat_directa.almacen_principal_id
                datos["almacen_nombre"] = cat_directa.almacen_principal.nombre

    # C) Ubicación
    if not datos.get("ubicacion_id") and not datos.get("_ubicacion_ambigua"):
        m_ubi = re.search(
            r"\b(?:en\s+la\s+ubicaci[oó]n|en\s+ubicaci[oó]n|ubicaci[oó]n)\s+([a-zA-Z0-9ñÑáéíóúÁÉÍÓÚ\s\-_]+?)(?:\s+(?:con\s+stock|con\s+un\s+stock|stock\s+inicial|stock\s+m[ií]nimo|en\s+categor[ií]a|en\s+almac[eé]n)|$|\.|\,)",
            texto_strip,
            re.I,
        )
        candidato_txt = m_ubi.group(1).strip() if m_ubi else texto_strip
        u_res, cands, st = _resolver_ubicacion_en_almacen(candidato_txt, datos.get("almacen_id"))
        if st == "ok" and u_res:
            if not datos.get("almacen_id") and u_res.almacen:
                if not usuario.tiene_acceso_almacen(u_res.almacen):
                    raise PermissionDenied(
                        f"La ubicación '{u_res.nombre}' pertenece a '{u_res.almacen.nombre}', almacén al cual no cuenta con acceso."
                    )
                datos["almacen_id"] = u_res.almacen_id
                datos["almacen_nombre"] = u_res.almacen.nombre
            datos["ubicacion_id"] = u_res.id
            datos["ubicacion_nombre"] = u_res.nombre
        elif st == "ambigua":
            datos["_ubicacion_ambigua"] = cands

    # D) Stock inicial
    if datos.get("stock_inicial") is None:
        m_ini = re.search(r"\b(?:stock\s+inicial|inicial|cantidad\s+inicial)\s*(?:es|ser[ií]a|de|en|:)?\s*([^\.,;\n]+)", texto_strip, re.I)
        val_ini = _extraer_numero(m_ini.group(1), permitir_cero=True, solo_al_inicio=True) if m_ini else None
        if val_ini is None:
            m_ini_pre = re.search(r"\b(\d+|[a-zA-ZáéíóúÁÉÍÓÚ]+(?:\s+y\s+[a-zA-ZáéíóúÁÉÍÓÚ]+)?)\s+(?:unidades?\s+)?inicial(?:es)?\b", texto_strip, re.I)
            if m_ini_pre:
                val_ini = _extraer_numero(m_ini_pre.group(1), permitir_cero=True, solo_al_inicio=True)
        if val_ini is not None:
            datos["stock_inicial"] = val_ini
        elif re.search(r"\b(?:con\s+)?(?:stock\s+inicial\s+de\s+0|0\s+de\s+stock\s+inicial|stock\s+inicial\s+cero)\b", texto_strip, re.I):
            datos["stock_inicial"] = 0

    # E) Stock mínimo
    if datos.get("stock_minimo") is None:
        m_min = re.search(r"\b(?:stock\s+m[ií]nimo|m[ií]nimo|umbral\s+m[ií]nimo)\s*(?:es|ser[ií]a|de|en|:)?\s*([^\.,;\n]+)", texto_strip, re.I)
        val_min = _extraer_numero(m_min.group(1), permitir_cero=True, solo_al_inicio=True) if m_min else None
        if val_min is not None:
            datos["stock_minimo"] = val_min
        elif re.search(r"\b(?:con\s+)?(?:stock\s+m[ií]nimo\s+de\s+0|0\s+de\s+stock\s+m[ií]nimo|stock\s+m[ií]nimo\s+cero)\b", texto_strip, re.I):
            datos["stock_minimo"] = 0

    # F) Nombre del producto
    if not datos.get("nombre"):
        m_nom = re.search(
            r"\b(?:crear|registrar|dar\s+de\s+alta)\s+(?:un\s+)?(?:nuevo\s+)?(?:producto|art[ií]culo)\s+(?:llamado\s+|denominado\s+|de\s+nombre\s+|con\s+nombre\s+)([a-zA-Z0-9ñÑáéíóúÁÉÍÓÚ\s\-_]+?)(?:\s+(?:en\s+la\s+categor[ií]a|en\s+categor[ií]a|en\s+el\s+almac[eé]n|en\s+almac[eé]n|en\s+la\s+ubicaci[oó]n|en\s+ubicaci[oó]n|con\s+stock|con\s+un\s+stock|con\s+stock\s+inicial|con\s+stock\s+m[ií]nimo|(?:y\s+)?(?:no\s+)?(?:es\s+)?(?:prestable|retornable|consumible)|categor[ií]a|almac[eé]n|ubicaci[oó]n)|$|\.|\,)",
            texto_strip,
            re.I,
        )
        if m_nom:
            cand = m_nom.group(1).strip()
            if cand.lower() not in ("un", "nuevo", "de", "el", "la"):
                datos["nombre"] = cand

    # G) Naturaleza prestable/retornable vs consumible (opcional; por defecto False = consumible)
    es_prest_txt = _extraer_es_prestable_en_texto(texto_strip)
    if es_prest_txt is not None:
        datos["es_prestable"] = es_prest_txt

    # Validación de valores negativos para stock inicial y stock mínimo (Corrección 1 - C-02)
    if (
        re.search(r"\b(?:stock\s+inicial|inicial|cantidad\s+inicial)\s*(?:de|en|:)?\s*-\d+\b", texto_strip, re.I)
        or re.search(r"\b(?:stock\s+m[ií]nimo|m[ií]nimo|umbral\s+m[ií]nimo)\s*(?:de|en|:)?\s*-\d+\b", texto_strip, re.I)
        or (datos.get("stock_inicial") is not None and datos["stock_inicial"] < 0)
        or (datos.get("stock_minimo") is not None and datos["stock_minimo"] < 0)
    ):
        if estado_operativo:
            estado_operativo.operacion_pendiente = None
            HakuOperationalContext.guardar(estado_operativo)
        msg = "No se permiten números negativos para el stock inicial ni para el stock mínimo. Por favor indique valores iguales o mayores a 0."
        return {
            "tipo": "error",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"error": "valor_negativo_invalido"},
        }

    # 6. Revalidar consistencia relacional y regla DN-01 (C02-11)
    if datos.get("_ubicacion_ambigua"):
        cands = datos.pop("_ubicacion_ambigua")
        noms_c = [u.nombre for u in cands]
        msg = f"Encontré varias ubicaciones posibles: {', '.join(noms_c)}. Por favor especifique a cuál se refiere."
        if estado_operativo:
            estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "ubicacion", "datos_producto": datos}
            HakuOperationalContext.guardar(estado_operativo)
        return {
            "tipo": "aclaracion",
            "esperando": "ubicacion",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"esperando": "ubicacion", "candidatos": [{"id": u.id, "nombre": u.nombre} for u in cands]},
        }

    if datos.get("categoria_id") and datos.get("almacen_id"):
        cat_obj = Categoria.objects.filter(id=datos["categoria_id"]).first()
        alm_obj = Almacen.objects.filter(id=datos["almacen_id"]).first()
        if cat_obj and alm_obj and cat_obj.almacen_principal_id != alm_obj.id:
            msg = (
                f"La categoría '{cat_obj.nombre}' pertenece al almacén '{cat_obj.almacen_principal.nombre}', "
                f"pero seleccionaste el almacén '{alm_obj.nombre}'. "
                f"Por favor selecciona una categoría correspondiente a '{alm_obj.nombre}' o cambia el almacén."
            )
            cats_disp = list(Categoria.objects.filter(almacen_principal_id=alm_obj.id, activo=True, eliminado=False))
            datos.pop("categoria_id", None)
            datos.pop("categoria_nombre", None)
            if estado_operativo:
                estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "categoria", "datos_producto": datos}
                HakuOperationalContext.guardar(estado_operativo)
            return {
                "tipo": "aclaracion",
                "esperando": "categoria",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"esperando": "categoria", "candidatos": [{"id": c.id, "nombre": c.nombre} for c in cats_disp]},
            }

    # Revalidar pertenencia de la ubicación al almacén (C02-09)
    if datos.get("ubicacion_id") and datos.get("almacen_id"):
        ubi_obj = Ubicacion.objects.filter(id=datos["ubicacion_id"]).first()
        alm_obj = Almacen.objects.filter(id=datos["almacen_id"]).first()
        if ubi_obj and alm_obj and ubi_obj.almacen_id != alm_obj.id:
            msg = f"La ubicación '{ubi_obj.nombre}' pertenece a '{ubi_obj.almacen.nombre}', no a '{alm_obj.nombre}'. Ubicaciones disponibles en '{alm_obj.nombre}': {', '.join(u.nombre for u in Ubicacion.objects.filter(almacen_id=alm_obj.id, activo=True, eliminado=False))}."
            ubis_disp = list(Ubicacion.objects.filter(almacen_id=alm_obj.id, activo=True, eliminado=False))
            datos.pop("ubicacion_id", None)
            datos.pop("ubicacion_nombre", None)
            if estado_operativo:
                estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "ubicacion", "datos_producto": datos}
                HakuOperationalContext.guardar(estado_operativo)
            return {
                "tipo": "aclaracion",
                "esperando": "ubicacion",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"esperando": "ubicacion", "candidatos": [{"id": u.id, "nombre": u.nombre} for u in ubis_disp]},
            }

    # 7. Solicitar aclaración de campos faltantes en orden lógico (C02-02 a C02-06)
    datos_publicos = {k: v for k, v in datos.items() if not str(k).startswith("_")}

    if not datos.get("nombre"):
        msg = "¿Cómo se llama el producto que desea registrar?"
        if estado_operativo:
            estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "nombre", "datos_producto": datos}
            HakuOperationalContext.guardar(estado_operativo)
        return {
            "tipo": "aclaracion",
            "esperando": "nombre",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"esperando": "nombre", "operacion_pendiente": "CREAR_PRODUCTO", "datos_producto": datos_publicos},
        }

    if not datos.get("almacen_id"):
        alms = list(usuario.obtener_almacenes_accesibles() if hasattr(usuario, "obtener_almacenes_accesibles") else Almacen.objects.filter(activo=True, eliminado=False))
        msg = f"¿En qué almacén se registrará '{datos['nombre']}'? Almacenes disponibles: {', '.join(a.nombre for a in alms)}. También puede indicar directamente la categoría."
        if estado_operativo:
            estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "almacen", "datos_producto": datos}
            HakuOperationalContext.guardar(estado_operativo)
        return {
            "tipo": "aclaracion",
            "esperando": "almacen",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"esperando": "almacen", "operacion_pendiente": "CREAR_PRODUCTO", "datos_producto": datos_publicos, "candidatos": [{"id": a.id, "nombre": a.nombre} for a in alms]},
        }

    if not datos.get("categoria_id"):
        cats = list(Categoria.objects.filter(almacen_principal_id=datos["almacen_id"], activo=True, eliminado=False))
        nota_cat_invalida = ""
        m_cat_menc = re.search(
            r"\b(?:la\s+)?categor[ií]a\s+(?:es|ser[ií]a|de|:)\s+([A-Za-zÁ-ÿñÑ0-9\s\-_]{2,40}?)(?:\.|,|;|\s+(?:la\s+ubicaci[oó]n|ubicaci[oó]n|el\s+almac[eé]n|almac[eé]n|el\s+stock|con\s+stock|stock)\b|$)",
            texto_strip,
            flags=re.IGNORECASE,
        )
        if m_cat_menc:
            cat_sol = m_cat_menc.group(1).strip(" .,;:!?")
            if cat_sol:
                nota_cat_invalida = f"No encontré la categoría '{cat_sol}' en {datos['almacen_nombre']}. "
        msg = f"{nota_correccion_campo}{nota_cat_invalida}¿A qué categoría pertenece '{datos['nombre']}'? Categorías disponibles para {datos['almacen_nombre']}: {', '.join(c.nombre for c in cats)}."
        if estado_operativo:
            estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "categoria", "datos_producto": datos}
            HakuOperationalContext.guardar(estado_operativo)
        return {
            "tipo": "aclaracion",
            "esperando": "categoria",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"esperando": "categoria", "operacion_pendiente": "CREAR_PRODUCTO", "datos_producto": datos_publicos, "candidatos": [{"id": c.id, "nombre": c.nombre} for c in cats]},
        }

    if not datos.get("ubicacion_id"):
        ubis = list(Ubicacion.objects.filter(almacen_id=datos["almacen_id"], activo=True, eliminado=False))
        msg = f"{nota_correccion_campo}¿En qué ubicación física de {datos['almacen_nombre']} estará '{datos['nombre']}'? Ubicaciones disponibles: {', '.join(u.nombre for u in ubis)}."
        if estado_operativo:
            estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "ubicacion", "datos_producto": datos}
            HakuOperationalContext.guardar(estado_operativo)
        return {
            "tipo": "aclaracion",
            "esperando": "ubicacion",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"esperando": "ubicacion", "operacion_pendiente": "CREAR_PRODUCTO", "datos_producto": datos_publicos, "candidatos": [{"id": u.id, "nombre": u.nombre} for u in ubis]},
        }

    if datos.get("stock_inicial") is None:
        msg = f"¿Cuánto stock inicial desea registrar para '{datos['nombre']}'? (Puede indicar 0 si no cuenta con unidades actualmente)."
        if estado_operativo:
            estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "stock_inicial", "datos_producto": datos}
            HakuOperationalContext.guardar(estado_operativo)
        return {
            "tipo": "aclaracion",
            "esperando": "stock_inicial",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"esperando": "stock_inicial", "operacion_pendiente": "CREAR_PRODUCTO", "datos_producto": datos_publicos},
        }

    if datos.get("stock_minimo") is None:
        msg = f"¿Cuál es el umbral de stock mínimo para '{datos['nombre']}'? Por favor indique la cantidad mínima para alertas de inventario."
        if estado_operativo:
            estado_operativo.operacion_pendiente = {"tipo": "CREAR_PRODUCTO", "esperando": "stock_minimo", "datos_producto": datos}
            HakuOperationalContext.guardar(estado_operativo)
        return {
            "tipo": "aclaracion",
            "esperando": "stock_minimo",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"esperando": "stock_minimo", "operacion_pendiente": "CREAR_PRODUCTO", "datos_producto": datos_publicos},
        }

    # 8. Todos los 6 campos están completos: construir propuesta formal (C02-01, C02-14, C02-15)
    if estado_operativo:
        estado_operativo.operacion_pendiente = None

    cid_res, thash_res = _resolver_contexto_contrato(sesion, conversacion_id)
    ahora = time.time()
    propuesta_id = str(uuid.uuid4())

    es_prestable_val = bool(datos.get("es_prestable", False))
    tipo_art_lbl = "Prestable / retornable" if es_prestable_val else "Consumible (no prestable)"
    tipo_art_voz = "prestable" if es_prestable_val else "consumible"

    propuesta = {
        "id": propuesta_id,
        "tipo_operacion": "CREAR_PRODUCTO",
        "producto_id": None,
        "producto_codigo": None,
        "producto_nombre": datos["nombre"],
        "categoria_id": datos["categoria_id"],
        "categoria_nombre": datos["categoria_nombre"],
        "almacen_origen_id": datos["almacen_id"],
        "almacen_origen_nombre": datos["almacen_nombre"],
        "ubicacion_origen_id": datos["ubicacion_id"],
        "ubicacion_origen_nombre": datos["ubicacion_nombre"],
        "cantidad": datos["stock_inicial"],
        "stock_minimo": datos["stock_minimo"],
        "es_prestable": es_prestable_val,
        "codigo_barras": datos.get("codigo_barras") or "",
        "destinatario": "",
        "motivo": "Alta de producto vía HAKU",
        "usuario_id": getattr(usuario, "id", None),
        "creado_en": ahora,
        "expira_en": round(ahora + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4),
        "version_contrato": VERSION_FIRMA_PROPUESTA,
        "conversacion_id": cid_res,
        "session_token_hash": thash_res,
        "datos_extra": {
            "stock_minimo": datos["stock_minimo"],
            "es_prestable": es_prestable_val,
            "codigo_barras": datos.get("codigo_barras") or "",
            "categoria_nombre": datos["categoria_nombre"],
            "ubicacion_origen_nombre": datos["ubicacion_nombre"],
        },
    }
    calcular_huella_propuesta(propuesta)

    if estado_operativo:
        estado_operativo.propuesta_activa = propuesta
        estado_operativo.registrar_accion(
            "PROPUESTA_CREAR_PRODUCTO",
            producto=datos["nombre"],
            cantidad=datos["stock_inicial"],
            detalles={
                "almacen": datos["almacen_nombre"],
                "stock_minimo": datos["stock_minimo"],
                "es_prestable": es_prestable_val,
            },
        )
        HakuOperationalContext.guardar(estado_operativo)

    if sesion is not None:
        token_sesion = sesion.get("haku_session_token")
        if token_sesion:
            propuesta["session_token"] = token_sesion
        sesion["haku_propuesta_activa"] = propuesta

    try:
        from .diagnostico import HakuDiagnostico
        HakuDiagnostico.registrar_propuesta("generada")
    except Exception:
        pass

    verbal = (
        f"Va a registrar el nuevo producto {datos['nombre']} en la categoría {datos['categoria_nombre']}, "
        f"almacén {datos['almacen_nombre']}, ubicación {datos['ubicacion_nombre']}, "
        f"marcado como {tipo_art_voz}, con {datos['stock_inicial']} unidades de stock inicial y stock mínimo de {datos['stock_minimo']}. "
        f"¿Desea confirmar la creación? También puede indicar si es prestable o consumible."
    )
    pantalla = (
        f"### Propuesta pendiente de confirmación\n"
        f"- **Operación:** Creación de nuevo producto\n"
        f"- **Producto:** {datos['nombre']}\n"
        f"- **Categoría:** {datos['categoria_nombre']}\n"
        f"- **Tipo de artículo:** {tipo_art_lbl}\n"
        f"- **Almacén:** {datos['almacen_nombre']}\n"
        f"- **Ubicación:** {datos['ubicacion_nombre']}\n"
        f"- **Stock inicial:** {datos['stock_inicial']} unidades\n"
        f"- **Stock mínimo:** {datos['stock_minimo']} unidades\n\n"
        f"¿Desea confirmar esta operación? *(También puedes decir «es prestable» o «es consumible» antes de confirmar).*"
    )
    return {
        "tipo": "propuesta",
        "requiere_confirmacion": True,
        "propuesta": propuesta,
        "mensaje_pantalla": pantalla,
        "mensaje_verbal": normalizar_para_tts(verbal),
        "datos": {
            "propuesta_id": propuesta_id,
            "tipo_operacion": "CREAR_PRODUCTO",
            "producto": datos["nombre"],
            "stock_inicial": datos["stock_inicial"],
            "stock_minimo": datos["stock_minimo"],
            "es_prestable": es_prestable_val,
        },
    }


def _procesar_comando_inventario_interno(
    usuario: UsuarioAudacIA,
    mensaje: str,
    propuesta_activa: dict | None = None,
    sesion: dict | None = None,
    estado_operativo: Any = None,
    interaction_id: str | None = None,
    ctx_resguardo: dict[str, Any] | None = None,
    conversacion_id: str | None = None,
    cliente_llm: Any = None,
) -> dict[str, Any]:
    from .cache import HakuCacheManager
    from .contexto import HakuOperationalContext
    from .diagnostico import HakuDiagnostico
    from .lenguaje.normalizador import IntencionResultado, IntencionTipo, NormalizadorLenguajeHaku
    from .lenguaje.siglas import NormalizadorSiglasTecnicas

    texto = NormalizadorSiglasTecnicas.normalizar(mensaje.strip())
    texto_lower = texto.lower()

    # Sublote 2.2: request.session es la única fuente canónica de verdad.
    # Prohibido expresamente cualquier fallback o rehidratación de propuestas desde RAM.

    intencion = None

    # No escoger silenciosamente una referencia para una operación de escritura:
    # «uno de esos» requiere una selección explícita cuando hay varios productos.
    candidatos_contexto = []
    if estado_operativo:
        candidatos_contexto = (
            getattr(estado_operativo, "candidatos", None)
            or getattr(estado_operativo, "coleccion_productos", None)
            or []
        )
    referencia_plural_operacion = bool(
        len(candidatos_contexto) > 1
        and re.search(r"\b(?:uno|una)\s+de\s+(?:es[oa]s|ellos|ellas|los\s+anteriores|las\s+anteriores)\b", texto_lower)
        and re.search(r"\b(?:presta|pr[eé]stamo|ingresa|ingresar|entrada|saca|retira|salida|traslada|trasladar|inhabilita|habilita)\b", texto_lower)
    )
    if referencia_plural_operacion:
        opciones = [
            {"id": c.get("id"), "codigo": c.get("codigo"), "nombre": c.get("nombre")}
            for c in candidatos_contexto[:5]
        ]
        opciones_txt = ", ".join(f"{c['nombre']} ({c['codigo']})" for c in opciones if c.get("nombre") and c.get("codigo"))
        msg = f"¿Cuál producto desea seleccionar para la operación? Opciones: {opciones_txt}."
        return {
            "tipo": "aclaracion",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"candidatos": opciones, "requiere_aclaracion": True},
        }

    # 0-pre. Selección determinista de candidatos pendientes de desambiguación («El primero», «El segundo», «El último», código, nombre)
    if intencion is None and estado_operativo and estado_operativo.candidatos and not propuesta_activa:
        cand_sel, error_cand = estado_operativo.resolver_seleccion_candidato(texto)
        if cand_sel:
            prod_sel = Producto.objects.filter(id=cand_sel["id"]).first()
            if prod_sel:
                # Validar permisos de acceso sobre el producto seleccionado
                if usuario and usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
                    alms_user = usuario.obtener_almacenes_accesibles()
                    tiene_acceso = prod_sel.existencias.filter(almacen__in=alms_user).exists() or (prod_sel.categoria and prod_sel.categoria.almacen_principal in alms_user)
                    if not tiene_acceso:
                        msg_denegado = f"No cuenta con permisos de acceso para consultar el producto **{prod_sel.nombre}**."
                        return {
                            "tipo": "error",
                            "mensaje_pantalla": msg_denegado,
                            "mensaje_verbal": normalizar_para_tts(msg_denegado),
                            "datos": {"error_permisos": True},
                        }

                int_pendiente = estado_operativo.intencion_pendiente_aclaracion or IntencionTipo.CONSULTAR_STOCK.value
                datos_pendientes = estado_operativo.datos_intencion_pendiente or {}
                estado_operativo.candidatos = []
                estado_operativo.intencion_pendiente_aclaracion = None
                estado_operativo.datos_intencion_pendiente = None
                estado_operativo.asignar_producto(prod_sel.id, prod_sel.codigo_interno, prod_sel.nombre)
                HakuOperationalContext.guardar(estado_operativo)

                # Continuar la consulta original según la intención pendiente
                if int_pendiente == IntencionTipo.CONSULTAR_UBICACION.value:
                    intencion = IntencionResultado(
                        tipo=IntencionTipo.CONSULTAR_UBICACION,
                        producto=prod_sel,
                        codigo_producto=prod_sel.codigo_interno,
                        almacen=datos_pendientes.get("almacen"),
                    )
                elif int_pendiente == IntencionTipo.CONSULTAR_PRODUCTO_MOVIMIENTOS.value:
                    intencion = IntencionResultado(
                        tipo=IntencionTipo.CONSULTAR_PRODUCTO_MOVIMIENTOS,
                        producto=prod_sel,
                        codigo_producto=prod_sel.codigo_interno,
                    )
                elif int_pendiente == IntencionTipo.EDITAR_PRODUCTO.value or int_pendiente == "EDITAR_PRODUCTO":
                    intencion = IntencionResultado(
                        tipo=IntencionTipo.EDITAR_PRODUCTO,
                        producto=prod_sel,
                        codigo_producto=prod_sel.codigo_interno,
                        datos_extra=datos_pendientes,
                        requiere_confirmacion=True,
                    )
                elif int_pendiente in ("consultar_atributo", "CONSULTAR_ATRIBUTO"):
                    intencion = IntencionResultado(
                        tipo=IntencionTipo.CONSULTAR_STOCK,
                        producto=prod_sel,
                        codigo_producto=prod_sel.codigo_interno,
                        datos_extra=datos_pendientes,
                    )
                elif int_pendiente in (
                    IntencionTipo.PRESTAMO.value,
                    IntencionTipo.REGISTRAR_SALIDA.value,
                    IntencionTipo.REGISTRAR_ENTRADA.value,
                    IntencionTipo.TRASLADAR_PRODUCTO.value,
                    IntencionTipo.DEVOLUCION.value,
                ):
                    intencion = IntencionResultado(
                        tipo=IntencionTipo(int_pendiente),
                        producto=prod_sel,
                        codigo_producto=prod_sel.codigo_interno,
                        cantidad=datos_pendientes.get("cantidad"),
                        destinatario=datos_pendientes.get("destinatario"),
                        motivo=datos_pendientes.get("motivo"),
                        almacen=datos_pendientes.get("almacen"),
                        datos_extra=datos_pendientes,
                        requiere_confirmacion=True,
                    )
                else:
                    intencion = IntencionResultado(
                        tipo=IntencionTipo.CONSULTAR_STOCK,
                        producto=prod_sel,
                        codigo_producto=prod_sel.codigo_interno,
                        almacen=datos_pendientes.get("almacen"),
                    )
        elif error_cand == "fuera_de_rango":
            opciones_str = ", ".join(f"{c['nombre']} ({c['codigo']})" for c in estado_operativo.candidatos)
            msg_rango = f"Solo se ofrecieron {len(estado_operativo.candidatos)} opciones: {opciones_str}. Por favor indique cuál de ellas desea consultar."
            return {
                "tipo": "aclaracion",
                "mensaje_pantalla": msg_rango,
                "mensaje_verbal": normalizar_para_tts(f"Solo hay {len(estado_operativo.candidatos)} opciones disponibles. Por favor elige una de ellas."),
                "datos": {"requiere_aclaracion": True},
            }
        elif error_cand == "ambiguo":
            opciones_str = ", ".join(f"{c['nombre']} ({c['codigo']})" for c in estado_operativo.candidatos)
            msg_amb = f"Hay ambigüedad en la selección. Las opciones disponibles son: {opciones_str}. ¿Cuál de ellas desea consultar?"
            return {
                "tipo": "aclaracion",
                "mensaje_pantalla": msg_amb,
                "mensaje_verbal": normalizar_para_tts("Hay ambigüedad en la selección. Por favor indique claramente el código o la opción deseada."),
                "datos": {"requiere_aclaracion": True},
            }
        else:
            # Si el usuario introduce una consulta completamente independiente, limpiar candidatos previos
            estado_operativo.candidatos = []
            estado_operativo.intencion_pendiente_aclaracion = None
            estado_operativo.datos_intencion_pendiente = None

    # Corrección 1 - C-02: Rechazo inmediato de stock negativo en creación de productos
    if re.search(r"\b(?:crear|registrar|alta|nuevo)\s+(?:un\s+)?(?:producto|art[ií]culo)\b", texto_lower) and (
        re.search(r"\b(?:stock\s+inicial|inicial|cantidad\s+inicial)\s*(?:de|en|:)?\s*-\d+\b", texto_lower)
        or re.search(r"\b(?:stock\s+m[ií]nimo|m[ií]nimo|umbral\s+m[ií]nimo)\s*(?:de|en|:)?\s*-\d+\b", texto_lower)
    ):
        if estado_operativo:
            estado_operativo.operacion_pendiente = None
            HakuOperationalContext.guardar(estado_operativo)
        msg = "No se permiten números negativos para el stock inicial ni para el stock mínimo. Por favor indique valores iguales o mayores a 0."
        return {
            "tipo": "error",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"error": "valor_negativo_invalido"},
        }

    # 0. Multiturno prioritario: Si hay una operación pendiente esperando ranuras (destinatario, cantidad, confirmación stock menor, ubicación destino, etc.)
    op_pendiente = getattr(estado_operativo, "operacion_pendiente", None) if estado_operativo else None
    op_pendiente_resguardo = None
    foco_resguardo = None
    es_consulta_intermedia = False

    if op_pendiente and op_pendiente.get("esperando"):
        import copy
        clasif_op = _clasificar_turno_con_operacion_pendiente(texto, op_pendiente)
        if clasif_op == "cancelar":
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            return {
                "tipo": "accion_cancelar",
                "mensaje_pantalla": "Operación pendiente cancelada. No se realizaron cambios en el inventario.",
                "mensaje_verbal": normalizar_para_tts("Operación cancelada. No se realizaron cambios en el inventario."),
            }
        elif clasif_op == "nueva_operacion":
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            op_pendiente = None
        elif clasif_op == "consulta_intermedia":
            es_consulta_intermedia = True
            op_pendiente_resguardo = copy.deepcopy(op_pendiente)
            if estado_operativo:
                foco_resguardo = {
                    "producto_id": estado_operativo.producto_id,
                    "producto_codigo": estado_operativo.producto_codigo,
                    "producto_nombre": estado_operativo.producto_nombre,
                }
                # Si no hay producto en foco, exponer temporalmente el producto de la operación pendiente para consultas elípticas («¿Dónde está ubicado?»)
                if not estado_operativo.producto_id and op_pendiente.get("producto_id"):
                    p_tmp = Producto.objects.filter(id=op_pendiente.get("producto_id")).first()
                    if p_tmp:
                        estado_operativo.producto_id = p_tmp.id
                        estado_operativo.producto_codigo = p_tmp.codigo_interno
                        estado_operativo.producto_nombre = p_tmp.nombre
                estado_operativo.operacion_pendiente = None
            if ctx_resguardo is not None:
                ctx_resguardo["es_consulta_intermedia"] = True
                ctx_resguardo["op_pendiente_resguardo"] = op_pendiente_resguardo
                ctx_resguardo["foco_resguardo"] = foco_resguardo
            op_pendiente = None

    if op_pendiente and op_pendiente.get("esperando") == "ubicacion_destino":
        from .lenguaje.numeros import extraer_modificacion_cantidad_propuesta as _ext_cant_ub_dest
        cant_mod_ub = _ext_cant_ub_dest(texto)
        if re.search(r"\b(cancelar|cancela|cancelalo|olv[ií]dalo|descarta|descartar|no quiero|ninguna)\b", texto_lower) and cant_mod_ub is None and not re.search(r"\b(?:sino|en\s+vez\s+de|en\s+lugar\s+de)\b", texto_lower):
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            return {
                "tipo": "accion_cancelar",
                "mensaje_pantalla": "Operación cancelada. El traslado ha sido descartado.",
                "mensaje_verbal": normalizar_para_tts("Operación cancelada. No se realizaron cambios en el inventario."),
            }

        alm_dest_id = op_pendiente.get("almacen_destino_id")
        ubicaciones_disp = list(
            Ubicacion.objects.filter(almacen_id=alm_dest_id, activo=True, eliminado=False).order_by("nombre")
        )

        # Permitir ajuste de cantidad en medio del flujo antes de elegir la ubicación destino
        if cant_mod_ub is not None and cant_mod_ub > 0:
            prod_id_ub = op_pendiente.get("producto_id")
            alm_orig_id_ub = op_pendiente.get("almacen_id")
            ex_orig_ub = Existencia.objects.filter(producto_id=prod_id_ub, almacen_id=alm_orig_id_ub).first()
            disp_orig_ub = ex_orig_ub.stock_disponible if ex_orig_ub else 0
            if disp_orig_ub < cant_mod_ub:
                msg_stk_ub = (
                    f"Stock insuficiente para el traslado: solo hay **{disp_orig_ub}** unidades disponibles "
                    f"en el almacén de origen (solicitaste **{cant_mod_ub}**)."
                )
                return {
                    "tipo": "error",
                    "mensaje_pantalla": msg_stk_ub,
                    "mensaje_verbal": normalizar_para_tts(f"Stock insuficiente. Solo hay {disp_orig_ub} unidades disponibles."),
                }
            op_pendiente["cantidad"] = int(cant_mod_ub)
            if estado_operativo:
                estado_operativo.operacion_pendiente = op_pendiente
                HakuOperationalContext.guardar(estado_operativo)

        ub_seleccionada = None
        texto_limpio = texto_lower.strip()
        if texto_limpio.isdigit() and cant_mod_ub is None:
            idx = int(texto_limpio) - 1
            if 0 <= idx < len(ubicaciones_disp):
                ub_seleccionada = ubicaciones_disp[idx]

        if not ub_seleccionada:
            from inventario.utils import normalizar_texto as norm_txt
            t_norm = norm_txt(texto_limpio)
            ub_seleccionada = _seleccionar_entidad_en_texto(
                ubicaciones_disp,
                texto_lower,
                t_norm,
                norm_fn=norm_txt,
                usar_clave_canonica=True,
                preferir_ultimo=True,
            )

        if ub_seleccionada:
            prod_id = op_pendiente.get("producto_id")
            prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
            alm_orig_id = op_pendiente.get("almacen_id")
            alm_orig = Almacen.objects.filter(id=alm_orig_id).first() if alm_orig_id else None
            alm_dest = Almacen.objects.filter(id=alm_dest_id).first() if alm_dest_id else None

            datos_extra = {
                "almacen_destino_id": alm_dest.id if alm_dest else alm_dest_id,
                "almacen_destino_nombre": alm_dest.nombre if alm_dest else op_pendiente.get("almacen_destino_nombre"),
                "ubicacion_destino_id": ub_seleccionada.id,
                "ubicacion_destino_nombre": ub_seleccionada.nombre,
            }
            if op_pendiente.get("observacion"):
                datos_extra["observacion"] = op_pendiente["observacion"]
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)

            intencion = IntencionResultado(
                tipo=IntencionTipo.TRASLADAR_PRODUCTO,
                producto=prod_obj,
                codigo_producto=prod_obj.codigo_interno if prod_obj else None,
                cantidad=op_pendiente.get("cantidad"),
                almacen=alm_orig,
                datos_extra=datos_extra,
                requiere_confirmacion=True,
            )
        elif cant_mod_ub is not None and cant_mod_ub > 0:
            alm_dest_nom = op_pendiente.get("almacen_destino_nombre", "el almacén destino")
            msg_p = (
                f"Entendido, actualicé la cantidad a **{cant_mod_ub} unidades**.\n"
                f"¿En cuál ubicación física de **{alm_dest_nom}** desea colocar el producto?\n"
                + "\n".join(f"- **{u.nombre}**" for u in ubicaciones_disp)
            )
            return {
                "tipo": "aclaracion",
                "esperando": "ubicacion_destino",
                "mensaje_pantalla": msg_p,
                "mensaje_verbal": normalizar_para_tts(
                    f"Actualicé la cantidad a {cant_mod_ub} unidades. ¿En cuál ubicación de {alm_dest_nom} desea colocar el producto?"
                ),
                "datos": {
                    "esperando": "ubicacion_destino",
                    "requiere_ubicacion_destino": True,
                    "cantidad": cant_mod_ub,
                    "candidatos": [{"id": u.id, "nombre": u.nombre} for u in ubicaciones_disp],
                },
            }
        else:
            alm_dest_nom = op_pendiente.get("almacen_destino_nombre", "el almacén destino")
            msg_p = (
                f"No reconocí la ubicación '{texto.strip()}'. Por favor selecciona una de las ubicaciones físicas disponibles en **{alm_dest_nom}**:\n"
                + "\n".join(f"- **{u.nombre}**" for u in ubicaciones_disp)
                + f"\n\n¿En cuál de estas ubicaciones desea colocar el producto?"
            )
            nombres_v = [u.nombre for u in ubicaciones_disp[:3]]
            return {
                "tipo": "aclaracion",
                "esperando": "ubicacion_destino",
                "mensaje_pantalla": msg_p,
                "mensaje_verbal": normalizar_para_tts(f"No reconocí la ubicación {texto}. Las ubicaciones son {', '.join(nombres_v)}. Por favor selecciona una."),
                "datos": {
                    "esperando": "ubicacion_destino",
                    "requiere_ubicacion_destino": True,
                    "candidatos": [{"id": u.id, "nombre": u.nombre} for u in ubicaciones_disp],
                },
            }

    elif op_pendiente and op_pendiente.get("tipo") == "CREAR_PRODUCTO":
        return _procesar_alta_producto_conversacional(
            texto=texto,
            usuario=usuario,
            estado_operativo=estado_operativo,
            sesion=sesion,
            op_pendiente=op_pendiente,
            conversacion_id=conversacion_id,
        )

    elif op_pendiente and op_pendiente.get("esperando") in ("destinatario", "cantidad", "confirmar_menor_cantidad", "referencia_prestamo", "destino", "producto", "identificacion"):
        int_slot = NormalizadorLenguajeHaku.interpretar(
            mensaje=texto,
            usuario=usuario,
            contexto_sesion=sesion,
            estado_operativo=estado_operativo,
        )
        if int_slot and int_slot.tipo not in (IntencionTipo.SOLICITUD_NO_COMPRENDIDA, IntencionTipo.NO_INTENCION_INVENTARIO):
            intencion = int_slot

    # 0-bis. Acciones administrativas explícitas sobre producto en foco/pronombre sin propuesta activa
    if intencion is None and not propuesta_activa and "?" not in texto and "¿" not in texto:
        from .agente.planificador import (
            _es_negacion_operacion,
            _es_pregunta_como_operar,
            _es_pregunta_permiso_capacidad,
        )
        es_consulta_o_negacion_admin = bool(
            _es_pregunta_como_operar(texto)
            or _es_pregunta_permiso_capacidad(texto)
            or _es_negacion_operacion(texto)
            or re.match(
                r"^(?:c[oó]mo|qu[eé]|cu[aá]l(?:es)?|cu[aá]nt[oa]s?|qui[eé]n(?:es)?|d[oó]nde|puedo|podemos|se\s+puede[n]?|no)\b",
                texto_lower.strip(),
            )
        )
        es_accion_inhab = (not es_consulta_o_negacion_admin) and bool(
            re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|lo\s+quiero\s+)?(?:ir\s+a\s+)?(?:inhabilitar|desactivar|archivar|dar\s+de\s+baja)\b", texto_lower)
            or re.search(r"\b(?:inhabilit[aá]r?lo|desact[ií]v[aá]r?lo|arch[ií]v[aá]r?lo|inhabil[ií]talo|desact[ií]valo|arch[ií]valo)\b", texto_lower)
        )
        es_accion_hab = (not es_consulta_o_negacion_admin) and bool(
            re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|lo\s+quiero\s+)?(?:ir\s+a\s+)?(?:habilitar|activar|reactivar)\b", texto_lower)
            or re.search(r"\b(?:habilit[aá]r?lo|activ[aá]r?lo|reactiv[aá]r?lo|hab[ií]l[ií]talo|act[ií]valo|react[ií]valo)\b", texto_lower)
        )
        if es_accion_inhab or es_accion_hab:
            int_admin = NormalizadorLenguajeHaku.interpretar(
                mensaje=texto,
                usuario=usuario,
                contexto_sesion=sesion,
                estado_operativo=estado_operativo,
            )
            if int_admin and int_admin.tipo in (IntencionTipo.INHABILITAR_PRODUCTO, IntencionTipo.HABILITAR_PRODUCTO):
                intencion = int_admin

    # 0-ter. Detección temprana de referencias anafóricas sin foco en contexto
    if intencion is None and not propuesta_activa:
        tiene_foco = bool(
            (estado_operativo and (estado_operativo.producto_id or estado_operativo.coleccion_productos))
            or (sesion and (sesion.get("haku_ultimo_producto_id") if isinstance(sesion, dict) else False))
        )
        if not tiene_foco:
            from .lenguaje.costeno import InterpretadorCosteno
            analisis_dem = InterpretadorCosteno.normalizar(texto)
            if analisis_dem.demostrativo_detectado:
                int_dem = NormalizadorLenguajeHaku.interpretar(
                    mensaje=texto,
                    usuario=usuario,
                    contexto_sesion=sesion if isinstance(sesion, dict) else None,
                    estado_operativo=estado_operativo,
                )
                if int_dem and int_dem.tipo == IntencionTipo.SOLICITAR_ACLARACION:
                    intencion = int_dem

    # ─────────────────────────────────────────────────────────────────────────
    # AGENTE BASADO EN EL CEREBRO CENTRAL (planificar → consultar → redactar)
    # Solo sin propuesta activa ni recolección de ranuras pendiente: esos estados
    # transaccionales conservan su flujo de seguridad dedicado.
    # ─────────────────────────────────────────────────────────────────────────
    from .lenguaje.interprete_semantico import ExtraccionSemantica, InterpreteSemanticoHaku
    semantica_agente = None
    agente_infirio = False
    salida_ag = None
    op_pend_activa = bool(op_pendiente and op_pendiente.get("esperando"))

    # C-03: Si hay una propuesta activa previa, verificar primero si el mensaje es una acción sobre ella (confirmar, cancelar, modificar).
    # Si lo es, se resuelve de forma determinista sin consultar al agente ni al LLM.
    if propuesta_activa and intencion is None:
        res_prop = InterpreteSemanticoHaku._analizar_en_contexto_propuesta(texto, propuesta_activa)
        if res_prop is not None:
            semantica_agente = res_prop

    if (
        intencion is None
        and semantica_agente is None
        and not op_pend_activa
        and InterpreteSemanticoHaku._evaluar_fast_path_social(texto) is None
        and getattr(settings, "HAKU_AGENTE_HABILITADO", True)
    ):
        from .agente import AgenteInventarioHaku
        t0_ag = time.perf_counter()
        salida_ag = AgenteInventarioHaku.procesar(
            texto=texto,
            usuario=usuario,
            estado_operativo=estado_operativo,
            sesion=sesion,
            interaction_id=interaction_id,
            cliente_llm=cliente_llm,
        )
        agente_infirio = salida_ag.inferencia_realizada
        if estado_operativo is not None:
            llm_agente_ms = (
                float(salida_ag.plan.llm_ms)
                if salida_ag.plan is not None
                and salida_ag.inferencia_realizada
                and float(salida_ag.plan.llm_ms or 0.0) > 0.0
                else None
            )
            llm_cola_agente_ms = (
                float(salida_ag.plan.llm_cola_ms)
                if salida_ag.plan is not None
                and salida_ag.inferencia_realizada
                else None
            )
            estado_operativo.metricas_tiempo = {
                "semantica_ms": None,
                "llm_ms": round(llm_agente_ms, 2) if llm_agente_ms is not None else None,
                "llm_cola_ms": round(llm_cola_agente_ms, 2) if llm_cola_agente_ms is not None else None,
            }
        if salida_ag.plan is not None:
            registrar_timing(interaction_id, "llm", salida_ag.plan.llm_ms)
        registrar_timing(interaction_id, "agente_total", (time.perf_counter() - t0_ag) * 1000.0)

        # C-03: Si el agente devolvió respuesta (éxito o error terminal)
        if salida_ag.respuesta is not None:
            # Si hay una propuesta activa previa y el agente arrojó un error terminal,
            # preservar la propuesta activa y advertir al usuario que sigue pendiente.
            if propuesta_activa and getattr(salida_ag, "error_terminal", False):
                salida_ag.respuesta["propuesta_activa"] = True
                if "datos" not in salida_ag.respuesta or not isinstance(salida_ag.respuesta["datos"], dict):
                    salida_ag.respuesta["datos"] = {}
                salida_ag.respuesta["datos"]["propuesta_preservada"] = True
                msg_nota = "\n\n*Nota: Su propuesta anterior sigue pendiente de confirmación o cancelación.*"
                if msg_nota.strip() not in salida_ag.respuesta.get("mensaje_pantalla", ""):
                    salida_ag.respuesta["mensaje_pantalla"] = (
                        (salida_ag.respuesta.get("mensaje_pantalla") or "") + msg_nota
                    )

            if not getattr(salida_ag, "error_terminal", False):
                HakuDiagnostico.registrar_intencion(
                    f"AGENTE_{(salida_ag.plan.operacion or salida_ag.plan.tipo).upper() if salida_ag.plan else 'CONSULTA'}",
                    latencia_ms=max((time.perf_counter() - t0_ag) * 1000.0, 0.01),
                )
                if "datos" not in salida_ag.respuesta or salida_ag.respuesta["datos"] is None:
                    salida_ag.respuesta["datos"] = {}
                if estado_operativo and estado_operativo.producto_id:
                    salida_ag.respuesta["datos"].setdefault("producto_id", str(estado_operativo.producto_id))
                    salida_ag.respuesta["datos"].setdefault("producto_codigo", estado_operativo.producto_codigo)
                    salida_ag.respuesta["datos"].setdefault("producto_nombre", estado_operativo.producto_nombre)
            return salida_ag.respuesta

        if salida_ag.intencion is not None:
            intencion = salida_ag.intencion
        if salida_ag.extraccion is not None:
            semantica_agente = salida_ag.extraccion

        # C-03: Firewall contra Zero-LLM fallback: Si hubo error terminal o intento de inferencia física
        # y no se produjo ninguna intención ni extracción operativa válida, NUNCA delegar en Zero-LLM.
        if intencion is None and semantica_agente is None:
            if getattr(salida_ag, "error_terminal", False) or agente_infirio or getattr(salida_ag, "intento_inferencia_fisica", False):
                estado_t = getattr(salida_ag, "estado_terminal", "LLM_ERROR_TERMINAL")
                cod_http = getattr(salida_ag, "codigo_http", 502)
                msg_p = AgenteInventarioHaku._obtener_mensaje_error_pantalla(estado_t)
                msg_v = AgenteInventarioHaku._obtener_mensaje_error_verbal(estado_t)
                resp_err = {
                    "tipo": "error",
                    "mensaje_pantalla": msg_p,
                    "mensaje_verbal": msg_v,
                    "datos": {
                        "error": estado_t,
                        "codigo": estado_t,
                        "codigo_http": cod_http,
                        "error_terminal": True,
                        "llamadas_fisicas": 1 if getattr(salida_ag, "intento_inferencia_fisica", False) else 0,
                    },
                }
                if propuesta_activa:
                    resp_err["propuesta_activa"] = True
                    resp_err["datos"]["propuesta_preservada"] = True
                    resp_err["mensaje_pantalla"] += "\n\n*Nota: Su propuesta anterior sigue pendiente de confirmación o cancelación.*"
                return resp_err

    # ─────────────────────────────────────────────────────────────────────────
    # GENERACIÓN DE INTENCIÓN SEMÁNTICA (CAPA LINGÜÍSTICA HÍBRIDA / CEREBRO CENTRAL)
    # ─────────────────────────────────────────────────────────────────────────
    t0_sem = time.perf_counter()
    if semantica_agente is not None:
        semantica = semantica_agente
    elif intencion is None:
        if agente_infirio:
            # C-03: Fallback Zero-LLM prohibido terminantemente tras fallo de inferencia del LLM
            return {
                "tipo": "error",
                "mensaje_pantalla": "⚠️ **Servicio no disponible**\n\nEl servicio central de comprensión lingüística falló y no está permitido el fallback silencioso.",
                "mensaje_verbal": "El servicio central de procesamiento lingüístico no está disponible en este momento.",
                "datos": {"error": "LLM_ERROR_TERMINAL", "codigo": "LLM_ERROR_TERMINAL", "codigo_http": 502, "error_terminal": True},
            }
        semantica = InterpreteSemanticoHaku.interpretar(
            mensaje=texto,
            contexto_operativo=estado_operativo,
            usuario=usuario,
        )
    else:
        semantica = ExtraccionSemantica(intencion="desconocido", confianza=1.0)
    dt_sem = (time.perf_counter() - t0_sem) * 1000.0
    llm_ms = semantica.datos_extra.get("llm_ms") if hasattr(semantica, "datos_extra") and semantica.datos_extra else None
    llm_cola_ms = semantica.datos_extra.get("llm_cola_ms") if hasattr(semantica, "datos_extra") and semantica.datos_extra else None
    if llm_ms:
        registrar_timing(interaction_id, "llm", llm_ms)
    registrar_timing(interaction_id, "semantica_total", dt_sem)
    if estado_operativo:
        estado_operativo.metricas_tiempo = {
            "semantica_ms": round(dt_sem, 2),
            "llm_ms": round(llm_ms, 2) if llm_ms else None,
            "llm_cola_ms": round(llm_cola_ms, 2) if llm_cola_ms is not None else None,
        }

    # 1. Seguridad Transaccional: Modificación, confirmación o cancelación de propuesta activa
    if propuesta_activa:
        def _marcar_modificacion_sin_resolver(
            msg_pantalla: str,
            msg_verbal: str | None = None,
            datos_extra_resp: dict[str, Any] | None = None,
            tipo_resp: str = "aclaracion",
        ) -> dict[str, Any]:
            propuesta_activa["modificacion_sin_resolver"] = True
            calcular_huella_propuesta(propuesta_activa)
            if estado_operativo:
                estado_operativo.propuesta_activa = propuesta_activa
                HakuOperationalContext.guardar(estado_operativo)
            if sesion is not None:
                sesion["haku_propuesta_activa"] = propuesta_activa
            datos_out: dict[str, Any] = {
                "modificacion_pendiente": True,
                "actualizar_propuesta_sesion": True,
            }
            if datos_extra_resp:
                datos_out.update(datos_extra_resp)
            return {
                "tipo": tipo_resp,
                "mensaje_pantalla": msg_pantalla,
                "mensaje_verbal": normalizar_para_tts(msg_verbal or msg_pantalla),
                "propuesta_activa": True,
                "datos": datos_out,
            }

        # Si estamos esperando desambiguar un producto dentro de una modificación de propuesta
        if propuesta_activa.get("modificacion_sin_resolver") and estado_operativo and estado_operativo.candidatos:
            cand_sel_mod, _ = estado_operativo.resolver_seleccion_candidato(texto)
            if cand_sel_mod:
                semantica = ExtraccionSemantica(
                    intencion="modificar_propuesta",
                    campo_modificar="producto",
                    nuevo_valor=cand_sel_mod.get("codigo") or cand_sel_mod.get("nombre"),
                    producto=cand_sel_mod.get("codigo") or cand_sel_mod.get("nombre"),
                    confianza=0.99,
                )
                estado_operativo.candidatos = []

        if semantica.intencion == "modificar_propuesta_incompleta":
            msg_inc = (
                "Entendí que desea corregir la propuesta activa, pero no identifiqué qué dato cambiar. "
                "Por favor indique claramente el producto (código o nombre), la cantidad, el destinatario o el almacén que desea actualizar, "
                "o di **«Cancelar»** para descartar la operación."
            )
            return _marcar_modificacion_sin_resolver(
                msg_inc,
                "No identifiqué qué dato desea corregir en la propuesta. Por favor indique el producto, la cantidad o el destinatario correcto, o diga cancelar.",
            )

        if semantica.intencion == "modificar_propuesta":
            # Verificar expiración de la propuesta activa (30 minutos)
            if time.time() > propuesta_activa.get("expira_en", 0):
                HakuDiagnostico.registrar_propuesta("expirada")
                if estado_operativo:
                    estado_operativo.propuesta_activa = None
                    HakuOperationalContext.guardar(estado_operativo)
                if sesion is not None:
                    sesion["haku_propuesta_activa"] = None
                return {
                    "tipo": "aclaracion",
                    "mensaje_pantalla": "La propuesta previa ha caducado por superar los 30 minutos. Por favor genera una nueva solicitud.",
                    "mensaje_verbal": "La propuesta ha caducado. Por favor solicita la operación de nuevo.",
                    "propuesta_activa": False,
                }

            # Prohibir modificación de propuestas ya confirmadas
            if propuesta_activa.get("confirmada") or propuesta_activa.get("estado") == "confirmada":
                return {
                    "tipo": "error",
                    "mensaje_pantalla": "Esta propuesta ya fue confirmada y no puede ser modificada.",
                    "mensaje_verbal": normalizar_para_tts("Esta propuesta ya fue confirmada y no puede ser modificada."),
                }

            campo = semantica.campo_modificar
            nuevo_val = semantica.nuevo_valor
            tipo_op_actual = str(propuesta_activa.get("tipo_operacion") or propuesta_activa.get("tipo") or "").upper()

            if campo == "producto":
                if tipo_op_actual in (Movimiento.Tipo.DEVOLUCION, "DEVOLUCION"):
                    return _marcar_modificacion_sin_resolver(
                        "No se puede cambiar el producto de una devolución porque está asociada a un préstamo existente. Cancele esta propuesta y solicite la devolución del producto correcto.",
                        "No se puede cambiar el producto de una devolución existente. Cancele la propuesta y solicite la devolución del producto correcto.",
                        tipo_resp="error",
                    )
                termino_prod = (semantica.producto or str(nuevo_val or "")).strip()
                if not termino_prod:
                    return _marcar_modificacion_sin_resolver(
                        "¿Por cuál producto (nombre o código) desea sustituir el producto de la propuesta?",
                        "¿Por cuál producto desea cambiar la propuesta?",
                    )
                if tipo_op_actual == "CREAR_PRODUCTO":
                    if len(termino_prod) < 2:
                        return _marcar_modificacion_sin_resolver(
                            "El nombre del nuevo producto debe tener al menos 2 caracteres.",
                            "El nombre del producto debe tener al menos dos caracteres.",
                            tipo_resp="error",
                        )
                    propuesta_activa["producto_nombre"] = termino_prod[:200]
                    nuevo_val = propuesta_activa["producto_nombre"]
                else:
                    incluir_inact = (tipo_op_actual == "HABILITAR_PRODUCTO")
                    prod_nuevo, msg_amb_p, cands_p = NormalizadorLenguajeHaku._resolver_producto_en_texto(
                        termino_prod, usuario, incluir_inactivos=incluir_inact
                    )
                    if not prod_nuevo and cands_p:
                        if len(cands_p) == 1:
                            prod_nuevo = Producto.objects.filter(id=cands_p[0]["id"]).first()
                        else:
                            if estado_operativo:
                                estado_operativo.candidatos = cands_p
                                HakuOperationalContext.guardar(estado_operativo)
                            cands_str = ", ".join(f"**{c['nombre']}** (`{c['codigo']}`)" for c in cands_p[:5])
                            msg_amb = (
                                f"Encontré varios productos que coinciden con '{termino_prod}': {cands_str}.\n\n"
                                "¿A cuál de ellos desea cambiar la propuesta? (Indique el código exacto)."
                            )
                            return _marcar_modificacion_sin_resolver(
                                msg_amb,
                                f"Encontré varios productos para {termino_prod}. Por favor indique el código exacto.",
                                datos_extra_resp={"candidatos": cands_p, "requiere_aclaracion": True},
                            )
                    if not prod_nuevo:
                        return _marcar_modificacion_sin_resolver(
                            f"No encontré ningún producto que coincida con **{termino_prod}**. Por favor indique un código o nombre válido, o cancele la propuesta.",
                            f"No encontré el producto {termino_prod}. Por favor indique un código o nombre válido.",
                        )

                    # Validar estado activo/no archivado para operaciones de inventario
                    if tipo_op_actual == "HABILITAR_PRODUCTO":
                        if prod_nuevo.activo and not prod_nuevo.archivado:
                            return _marcar_modificacion_sin_resolver(
                                f"El producto **{prod_nuevo.nombre}** (`{prod_nuevo.codigo_interno}`) ya se encuentra activo y habilitado en el sistema.",
                                f"El producto {prod_nuevo.nombre} ya se encuentra activo.",
                                tipo_resp="error",
                            )
                    elif not prod_nuevo.activo or prod_nuevo.archivado:
                        return _marcar_modificacion_sin_resolver(
                            f"El producto **{prod_nuevo.nombre}** (`{prod_nuevo.codigo_interno}`) se encuentra inactivo o archivado y no admite esta operación.",
                            f"El producto {prod_nuevo.nombre} está inactivo y no admite esta operación.",
                            tipo_resp="error",
                        )

                    # Validar condición prestable si la propuesta es un préstamo
                    if tipo_op_actual in (Movimiento.Tipo.PRESTAMO, "PRESTAMO") and not getattr(prod_nuevo, "es_prestable", False):
                        return _marcar_modificacion_sin_resolver(
                            f"El producto **{prod_nuevo.nombre}** (`{prod_nuevo.codigo_interno}`) es un artículo consumible y no admite préstamos.",
                            f"El producto {prod_nuevo.nombre} es consumible y no admite préstamos.",
                            tipo_resp="error",
                        )

                    # Revalidar almacén y permisos sobre el nuevo producto
                    alm_actual_id = propuesta_activa.get("almacen_origen_id")
                    alm_actual_obj = Almacen.objects.filter(id=alm_actual_id, activo=True).first() if alm_actual_id else None
                    op_str_map = {
                        Movimiento.Tipo.ENTRADA: "entrada",
                        Movimiento.Tipo.SALIDA: "salida",
                        Movimiento.Tipo.PRESTAMO: "préstamo",
                        Movimiento.Tipo.TRASLADO_ENTRE: "traslado",
                    }
                    if tipo_op_actual in op_str_map:
                        alm_res, err_alm = _resolver_almacen_para_operacion(
                            usuario=usuario,
                            almacen_intencion=alm_actual_obj,
                            producto=prod_nuevo,
                            operacion=op_str_map[tipo_op_actual],
                        )
                        if err_alm:
                            msg_err_alm = err_alm.get("mensaje_pantalla") if isinstance(err_alm, dict) else str(err_alm)
                            return _marcar_modificacion_sin_resolver(
                                msg_err_alm,
                                msg_err_alm,
                                tipo_resp="error",
                            )
                        alm_actual_obj = alm_res

                    if tipo_op_actual == Movimiento.Tipo.TRASLADO_ENTRE and alm_actual_obj:
                        if propuesta_activa.get("almacen_destino_id") and alm_actual_obj.id == propuesta_activa.get("almacen_destino_id"):
                            return _marcar_modificacion_sin_resolver(
                                f"El almacén de origen (**{alm_actual_obj.nombre}**) y el almacén destino no pueden ser el mismo.",
                                "El almacén de origen y el de destino no pueden ser el mismo.",
                                tipo_resp="error",
                            )

                    # Validar stock suficiente si es salida, préstamo o traslado
                    cant_actual = int(propuesta_activa.get("cantidad") or 1)
                    if tipo_op_actual in (Movimiento.Tipo.SALIDA, Movimiento.Tipo.PRESTAMO, Movimiento.Tipo.TRASLADO_ENTRE) and alm_actual_obj:
                        ex_nueva = Existencia.objects.filter(producto=prod_nuevo, almacen=alm_actual_obj).first()
                        disp_nuevo = ex_nueva.stock_disponible if ex_nueva else 0
                        if disp_nuevo < cant_actual:
                            return _marcar_modificacion_sin_resolver(
                                f"Stock insuficiente para **{prod_nuevo.nombre}** (`{prod_nuevo.codigo_interno}`): solo hay **{disp_nuevo}** unidades disponibles en **{alm_actual_obj.nombre}** (se requieren **{cant_actual}**).",
                                f"Stock insuficiente para {prod_nuevo.nombre}. Solo hay {disp_nuevo} unidades disponibles en {alm_actual_obj.nombre}.",
                                tipo_resp="error",
                            )

                    propuesta_activa["producto_id"] = prod_nuevo.id
                    propuesta_activa["producto_nombre"] = prod_nuevo.nombre
                    propuesta_activa["producto_codigo"] = prod_nuevo.codigo_interno
                    if alm_actual_obj:
                        propuesta_activa["almacen_origen_id"] = alm_actual_obj.id
                        propuesta_activa["almacen_origen_nombre"] = alm_actual_obj.nombre
                        propuesta_activa["almacen_nombre"] = alm_actual_obj.nombre
                    if estado_operativo:
                        estado_operativo.asignar_producto(prod_nuevo.id, prod_nuevo.codigo_interno, prod_nuevo.nombre)
                    if sesion is not None:
                        sesion["haku_ultimo_producto_id"] = str(prod_nuevo.id)
                    nuevo_val = f"{prod_nuevo.nombre} ({prod_nuevo.codigo_interno})"

            elif campo == "destinatario":
                if tipo_op_actual in (Movimiento.Tipo.DEVOLUCION, "DEVOLUCION"):
                    return _marcar_modificacion_sin_resolver(
                        "No se puede cambiar el responsable de una devolución porque está asociada a un préstamo existente. Si se trata de otra persona, cancele esta propuesta y solicite la devolución correspondiente.",
                        "No se puede cambiar el responsable de una devolución existente. Cancele la propuesta y solicite la devolución del préstamo correcto.",
                        tipo_resp="error",
                    )
                if tipo_op_actual not in (Movimiento.Tipo.PRESTAMO, "PRESTAMO", Movimiento.Tipo.SALIDA, "SALIDA"):
                    return _marcar_modificacion_sin_resolver(
                        "La operación actual no utiliza un campo de destinatario o responsable.",
                        "Esta operación no utiliza destinatario ni responsable.",
                        tipo_resp="error",
                    )
                dest_raw = str(nuevo_val or "").strip()
                if not dest_raw:
                    return _marcar_modificacion_sin_resolver(
                        "¿A nombre de qué persona desea registrar la operación?",
                        "¿A nombre de qué persona desea registrar la operación?",
                    )
                from .servicios_personas import HakuPersonasService
                origen_voz = bool(sesion and sesion.get("haku_ultimo_origen") == "voz")
                nombre_canonico, confianza_p, cands_persona, req_escritura = HakuPersonasService.resolver_destinatario(
                    dest_raw, origen_voz=origen_voz
                )
                if cands_persona and len(cands_persona) > 1:
                    cand_str = ", ".join(cands_persona[:4])
                    return _marcar_modificacion_sin_resolver(
                        f"Encontré varias personas con nombres similares ({cand_str}). Por favor indique el nombre y apellido completo para actualizar el destinatario con seguridad.",
                        f"Encontré varias personas con ese nombre: {cand_str}. Por favor indique el nombre completo.",
                        datos_extra_resp={"candidatos": cands_persona, "requiere_escritura": True},
                    )
                if not nombre_canonico:
                    return _marcar_modificacion_sin_resolver(
                        "No pude identificar con seguridad el nombre del destinatario. Por favor indícalo con nombre y apellido.",
                        "No pude identificar el nombre del destinatario. Por favor indícalo claramente.",
                    )
                dest_anterior = propuesta_activa.get("destinatario") or ""
                propuesta_activa["destinatario"] = nombre_canonico
                nuevo_val = nombre_canonico
                motivo_act = str(propuesta_activa.get("motivo") or "")
                if dest_anterior and dest_anterior in motivo_act:
                    propuesta_activa["motivo"] = motivo_act.replace(dest_anterior, nombre_canonico)
                elif tipo_op_actual == Movimiento.Tipo.SALIDA and motivo_act.startswith("Salida registrada por Haku"):
                    propuesta_activa["motivo"] = f"Salida registrada por Haku para {nombre_canonico}"
                elif tipo_op_actual == Movimiento.Tipo.PRESTAMO and motivo_act.startswith("Préstamo registrado por Haku"):
                    propuesta_activa["motivo"] = f"Préstamo registrado por Haku a {nombre_canonico}"

            elif campo == "cantidad":
                if tipo_op_actual in ("HABILITAR_PRODUCTO", "INHABILITAR_PRODUCTO"):
                    return _marcar_modificacion_sin_resolver(
                        "Esta operación de catálogo no requiere ni admite cantidad.",
                        "Esta operación de catálogo no admite cantidad.",
                        tipo_resp="error",
                    )
                nueva_cant = semantica.cantidad
                permite_cero_mod = (tipo_op_actual == "CREAR_PRODUCTO")
                if nueva_cant is None or (int(nueva_cant) < 0 if permite_cero_mod else int(nueva_cant) <= 0):
                    msg_c_err = (
                        "El stock inicial debe ser un número entero mayor o igual a cero."
                        if permite_cero_mod else
                        "La cantidad debe ser un número entero mayor a cero. ¿Cuántas unidades desea registrar?"
                    )
                    return _marcar_modificacion_sin_resolver(msg_c_err, msg_c_err, tipo_resp="error")
                nueva_cant = int(nueva_cant)
                if tipo_op_actual in (Movimiento.Tipo.DEVOLUCION, "DEVOLUCION"):
                    prestamo_id_act = propuesta_activa.get("prestamo_id")
                    if prestamo_id_act:
                        prest_act = Prestamo.objects.filter(id=prestamo_id_act).first()
                        if prest_act and nueva_cant > int(prest_act.cantidad_pendiente_devolucion or 0):
                            pend_act = int(prest_act.cantidad_pendiente_devolucion or 0)
                            nom_p_act = propuesta_activa.get("producto_nombre", "el producto")
                            return _marcar_modificacion_sin_resolver(
                                f"Cantidad excedida: el préstamo activo de **{nom_p_act}** solo tiene **{pend_act}** unidades pendientes de devolución (solicitaste **{nueva_cant}**).",
                                f"La cantidad solicitada supera las {pend_act} unidades pendientes de devolución de {nom_p_act}.",
                                tipo_resp="error",
                            )
                elif tipo_op_actual in (Movimiento.Tipo.SALIDA, Movimiento.Tipo.PRESTAMO, Movimiento.Tipo.TRASLADO_ENTRE):
                    prod_id_act = propuesta_activa.get("producto_id")
                    alm_id_act = propuesta_activa.get("almacen_origen_id")
                    if prod_id_act and alm_id_act:
                        ex_act = Existencia.objects.filter(producto_id=prod_id_act, almacen_id=alm_id_act).first()
                        disp_act = ex_act.stock_disponible if ex_act else 0
                        if disp_act < nueva_cant:
                            nom_p_act = propuesta_activa.get("producto_nombre", "el producto")
                            return _marcar_modificacion_sin_resolver(
                                f"Stock insuficiente: **{nom_p_act}** solo tiene **{disp_act}** unidades disponibles (solicitaste **{nueva_cant}**).",
                                f"Stock insuficiente. {nom_p_act} solo tiene {disp_act} unidades disponibles.",
                                tipo_resp="error",
                            )
                propuesta_activa["cantidad"] = nueva_cant
                nuevo_val = nueva_cant

            elif campo == "stock_minimo":
                if tipo_op_actual != "CREAR_PRODUCTO":
                    return _marcar_modificacion_sin_resolver(
                        "Solo las propuestas de creación de producto admiten modificar el stock mínimo.",
                        "Solo las propuestas de creación de producto admiten modificar el stock mínimo.",
                        tipo_resp="error",
                    )
                stk_min_raw = (semantica.datos_extra or {}).get("stock_minimo")
                if stk_min_raw is None:
                    stk_min_raw = semantica.cantidad if semantica.cantidad is not None else nuevo_val
                try:
                    nuevo_min = int(stk_min_raw)
                except (TypeError, ValueError):
                    nuevo_min = -1
                if nuevo_min < 0:
                    return _marcar_modificacion_sin_resolver(
                        "El stock mínimo debe ser un número entero mayor o igual a cero.",
                        "El stock mínimo debe ser cero o mayor.",
                        tipo_resp="error",
                    )
                propuesta_activa["stock_minimo"] = nuevo_min
                if not isinstance(propuesta_activa.get("datos_extra"), dict):
                    propuesta_activa["datos_extra"] = {}
                propuesta_activa["datos_extra"]["stock_minimo"] = nuevo_min
                nuevo_val = nuevo_min

            elif campo == "almacen_destino":
                if tipo_op_actual != Movimiento.Tipo.TRASLADO_ENTRE:
                    return _marcar_modificacion_sin_resolver(
                        "Solo las propuestas de traslado entre almacenes tienen almacén de destino.",
                        "Solo los traslados entre almacenes tienen almacén destino.",
                        tipo_resp="error",
                    )
                alm_dest_txt = (semantica.almacen or str(nuevo_val or "")).strip()
                alm_dest_match = _buscar_almacen_por_nombre(alm_dest_txt)
                if not alm_dest_match:
                    return _marcar_modificacion_sin_resolver(
                        f"No encontré el almacén destino **{alm_dest_txt}**. Por favor indique un almacén válido.",
                        f"No encontré el almacén destino {alm_dest_txt}.",
                        tipo_resp="error",
                    )
                if not usuario.tiene_acceso_almacen(alm_dest_match):
                    return _marcar_modificacion_sin_resolver(
                        f"No cuenta con permisos para trasladar hacia el almacén **{alm_dest_match.nombre}**.",
                        f"No cuenta con permisos para el almacén {alm_dest_match.nombre}.",
                        tipo_resp="error",
                    )
                if alm_dest_match.id == propuesta_activa.get("almacen_origen_id"):
                    return _marcar_modificacion_sin_resolver(
                        f"El almacén de destino (**{alm_dest_match.nombre}**) no puede ser el mismo que el almacén de origen.",
                        "El almacén destino y el origen no pueden ser el mismo.",
                        tipo_resp="error",
                    )
                ubis_dest = list(
                    Ubicacion.objects.filter(almacen=alm_dest_match, activo=True, eliminado=False).order_by("nombre")
                )
                if not ubis_dest:
                    return _marcar_modificacion_sin_resolver(
                        f"El almacén destino **{alm_dest_match.nombre}** no tiene ubicaciones físicas activas registradas.",
                        f"El almacén {alm_dest_match.nombre} no tiene ubicaciones físicas registradas.",
                        tipo_resp="error",
                    )
                from inventario.utils import normalizar_texto as _norm_ub_mod
                t_low_ub = (texto or "").lower()
                t_norm_ub = _norm_ub_mod(t_low_ub)
                ub_sel_mod = _seleccionar_entidad_en_texto(
                    ubis_dest,
                    t_low_ub,
                    t_norm_ub,
                    norm_fn=_norm_ub_mod,
                    usar_clave_canonica=True,
                    preferir_ultimo=True,
                )
                if not ub_sel_mod and len(ubis_dest) == 1:
                    ub_sel_mod = ubis_dest[0]

                if not ub_sel_mod:
                    prev_id_tr = str(propuesta_activa.get("id") or "").strip()
                    if estado_operativo:
                        estado_operativo.propuesta_activa = None
                        op_tr_pend = {
                            "tipo": "TRASLADO_ENTRE",
                            "producto_id": propuesta_activa.get("producto_id"),
                            "almacen_id": propuesta_activa.get("almacen_origen_id"),
                            "almacen_destino_id": alm_dest_match.id,
                            "almacen_destino_nombre": alm_dest_match.nombre,
                            "cantidad": propuesta_activa.get("cantidad"),
                            "esperando": "ubicacion_destino",
                            "candidatos_ubicacion": [{"id": u.id, "nombre": u.nombre} for u in ubis_dest],
                        }
                        if propuesta_activa.get("observacion"):
                            op_tr_pend["observacion"] = propuesta_activa.get("observacion")
                        estado_operativo.operacion_pendiente = op_tr_pend
                        HakuOperationalContext.guardar(estado_operativo)
                    if sesion is not None:
                        sesion["haku_propuesta_activa"] = None
                        if prev_id_tr:
                            sesion["haku_propuesta_sustituida_id"] = prev_id_tr
                    nom_p_tr = propuesta_activa.get("producto_nombre", "el producto")
                    cod_p_tr = propuesta_activa.get("producto_codigo", "")
                    cant_tr = propuesta_activa.get("cantidad", 1)
                    msg_ub_p = (
                        f"Destino actualizado a **{alm_dest_match.nombre}** para **{nom_p_tr}** (`{cod_p_tr}`, {cant_tr} unidades).\n\n"
                        f"Por favor selecciona la ubicación física de destino en **{alm_dest_match.nombre}**:\n"
                        + "\n".join(f"- **{u.nombre}**" for u in ubis_dest)
                        + "\n\n¿En cuál de estas ubicaciones desea colocar el producto?"
                    )
                    nombres_ub_v = ", ".join(u.nombre for u in ubis_dest[:3])
                    return {
                        "tipo": "aclaracion",
                        "esperando": "ubicacion_destino",
                        "propuesta_activa": False,
                        "mensaje_pantalla": msg_ub_p,
                        "mensaje_verbal": normalizar_para_tts(
                            f"Destino cambiado a {alm_dest_match.nombre}. Tiene las ubicaciones {nombres_ub_v}. ¿En cuál desea colocar el producto?"
                        ),
                        "datos": {
                            "esperando": "ubicacion_destino",
                            "requiere_ubicacion_destino": True,
                            "candidatos": [{"id": u.id, "nombre": u.nombre} for u in ubis_dest],
                        },
                    }

                propuesta_activa["almacen_destino_id"] = alm_dest_match.id
                propuesta_activa["almacen_destino_nombre"] = alm_dest_match.nombre
                propuesta_activa["ubicacion_destino_id"] = ub_sel_mod.id
                propuesta_activa["ubicacion_destino_nombre"] = ub_sel_mod.nombre
                alm_orig_nom_m = propuesta_activa.get("almacen_origen_nombre") or "almacén origen"
                propuesta_activa["motivo"] = f"Traslado vía Haku de {alm_orig_nom_m} a {alm_dest_match.nombre} ({ub_sel_mod.nombre})"
                nuevo_val = f"{alm_dest_match.nombre} ({ub_sel_mod.nombre})"

            elif campo == "ubicacion_destino":
                if tipo_op_actual != Movimiento.Tipo.TRASLADO_ENTRE:
                    return _marcar_modificacion_sin_resolver(
                        "Solo las propuestas de traslado entre almacenes tienen ubicación de destino.",
                        "Solo los traslados entre almacenes tienen ubicación de destino.",
                        tipo_resp="error",
                    )
                alm_dest_id_u = propuesta_activa.get("almacen_destino_id")
                ubis_dest_u = list(
                    Ubicacion.objects.filter(almacen_id=alm_dest_id_u, activo=True, eliminado=False).order_by("nombre")
                ) if alm_dest_id_u else []
                from inventario.utils import normalizar_texto as _norm_ub_only
                val_ub_txt = str(nuevo_val or texto or "").strip()
                val_ub_low = val_ub_txt.lower()
                val_ub_norm = _norm_ub_only(val_ub_low)
                t_ub_low = (texto or "").lower()
                t_ub_norm = _norm_ub_only(t_ub_low)
                ub_match_only = next(
                    (u_c for u_c in ubis_dest_u if u_c.nombre.lower() == val_ub_low),
                    None,
                )
                if not ub_match_only:
                    ub_match_only = _seleccionar_entidad_en_texto(
                        ubis_dest_u,
                        val_ub_low,
                        val_ub_norm,
                        norm_fn=_norm_ub_only,
                        usar_clave_canonica=True,
                        preferir_ultimo=True,
                    )
                if not ub_match_only:
                    ub_match_only = _seleccionar_entidad_en_texto(
                        ubis_dest_u,
                        t_ub_low,
                        t_ub_norm,
                        norm_fn=_norm_ub_only,
                        usar_clave_canonica=True,
                        preferir_ultimo=True,
                    )
                if not ub_match_only:
                    alm_dest_nom_u = propuesta_activa.get("almacen_destino_nombre") or "el almacén destino"
                    lista_u = ", ".join(f"**{u.nombre}**" for u in ubis_dest_u) or "sin ubicaciones activas"
                    return _marcar_modificacion_sin_resolver(
                        f"No encontré esa ubicación en **{alm_dest_nom_u}**. Ubicaciones disponibles: {lista_u}.",
                        f"No encontré esa ubicación en {alm_dest_nom_u}.",
                        tipo_resp="error",
                    )
                propuesta_activa["ubicacion_destino_id"] = ub_match_only.id
                propuesta_activa["ubicacion_destino_nombre"] = ub_match_only.nombre
                alm_orig_nom_u = propuesta_activa.get("almacen_origen_nombre") or "almacén origen"
                alm_dest_nom_u = propuesta_activa.get("almacen_destino_nombre") or "almacén destino"
                propuesta_activa["motivo"] = f"Traslado vía Haku de {alm_orig_nom_u} a {alm_dest_nom_u} ({ub_match_only.nombre})"
                nuevo_val = ub_match_only.nombre

            elif campo == "almacen" and semantica.almacen:
                if tipo_op_actual in (Movimiento.Tipo.DEVOLUCION, "DEVOLUCION"):
                    return _marcar_modificacion_sin_resolver(
                        "No se puede cambiar el almacén de una devolución porque pertenece al almacén del préstamo original.",
                        "No se puede cambiar el almacén de una devolución existente.",
                        tipo_resp="error",
                    )
                if tipo_op_actual in ("HABILITAR_PRODUCTO", "INHABILITAR_PRODUCTO"):
                    return _marcar_modificacion_sin_resolver(
                        "Esta operación de catálogo no utiliza cambio de almacén.",
                        "Esta operación de catálogo no permite cambiar de almacén.",
                        tipo_resp="error",
                    )
                if tipo_op_actual == "CREAR_PRODUCTO":
                    return _marcar_modificacion_sin_resolver(
                        "Para cambiar el almacén al crear un producto, cancele esta propuesta e indique el nuevo almacén, categoría y ubicación, ya que pertenecen a ese almacén.",
                        "Para cambiar de almacén al crear un producto, cancele la propuesta e indique el nuevo almacén, categoría y ubicación.",
                        tipo_resp="error",
                    )
                alm_match = _buscar_almacen_por_nombre(semantica.almacen)
                if not alm_match:
                    return _marcar_modificacion_sin_resolver(
                        f"No encontré el almacén **{semantica.almacen}**. Por favor indique un almacén válido.",
                        f"No encontré el almacén {semantica.almacen}.",
                        tipo_resp="error",
                    )
                if not usuario.tiene_acceso_almacen(alm_match):
                    return _marcar_modificacion_sin_resolver(
                        f"No cuenta con permisos para operar en el almacén **{alm_match.nombre}**.",
                        f"No cuenta con permisos para operar en el almacén {alm_match.nombre}.",
                        tipo_resp="error",
                    )
                if tipo_op_actual == Movimiento.Tipo.TRASLADO_ENTRE and alm_match.id == propuesta_activa.get("almacen_destino_id"):
                    return _marcar_modificacion_sin_resolver(
                        f"El almacén de origen (**{alm_match.nombre}**) no puede ser el mismo que el almacén de destino.",
                        "El almacén de origen y destino no pueden ser el mismo.",
                        tipo_resp="error",
                    )
                prod_id_act = propuesta_activa.get("producto_id")
                if prod_id_act and tipo_op_actual in (Movimiento.Tipo.ENTRADA, Movimiento.Tipo.SALIDA, Movimiento.Tipo.PRESTAMO, Movimiento.Tipo.TRASLADO_ENTRE):
                    prod_obj_alm = Producto.objects.select_related("categoria__almacen_principal").filter(id=prod_id_act).first()
                    if prod_obj_alm and prod_obj_alm.categoria and prod_obj_alm.categoria.almacen_principal_id != alm_match.id:
                        alm_real_nom = prod_obj_alm.categoria.almacen_principal.nombre
                        return _marcar_modificacion_sin_resolver(
                            (
                                f"Regla de almacén único: **{prod_obj_alm.nombre}** pertenece exclusivamente a **{alm_real_nom}** "
                                f"y no puede operar en **{alm_match.nombre}**. Si desea cambiarlo de sede, cancele y solicite un traslado entre almacenes."
                            ),
                            f"{prod_obj_alm.nombre} pertenece a {alm_real_nom} y no puede operar en {alm_match.nombre}.",
                            tipo_resp="error",
                        )
                    otra_ex = Existencia.objects.select_related("almacen").filter(producto_id=prod_id_act).exclude(almacen=alm_match).first()
                    if otra_ex:
                        return _marcar_modificacion_sin_resolver(
                            (
                                f"Regla de almacén único: **{propuesta_activa.get('producto_nombre', 'el producto')}** ya reside en "
                                f"**{otra_ex.almacen.nombre}** y no puede registrar existencias en **{alm_match.nombre}**."
                            ),
                            f"El producto ya reside en {otra_ex.almacen.nombre} y no puede operar en {alm_match.nombre}.",
                            tipo_resp="error",
                        )
                if tipo_op_actual in (Movimiento.Tipo.SALIDA, Movimiento.Tipo.PRESTAMO, Movimiento.Tipo.TRASLADO_ENTRE):
                    cant_act = int(propuesta_activa.get("cantidad") or 1)
                    if prod_id_act:
                        ex_alm = Existencia.objects.filter(producto_id=prod_id_act, almacen=alm_match).first()
                        disp_alm = ex_alm.stock_disponible if ex_alm else 0
                        if disp_alm < cant_act:
                            nom_p_act = propuesta_activa.get("producto_nombre", "el producto")
                            return _marcar_modificacion_sin_resolver(
                                f"Stock insuficiente en **{alm_match.nombre}**: **{nom_p_act}** solo tiene **{disp_alm}** unidades disponibles allí.",
                                f"Stock insuficiente en {alm_match.nombre}. Solo hay {disp_alm} unidades disponibles.",
                                tipo_resp="error",
                            )
                propuesta_activa["almacen_origen_id"] = alm_match.id
                propuesta_activa["almacen_origen_nombre"] = alm_match.nombre
                propuesta_activa["almacen_nombre"] = alm_match.nombre
                if tipo_op_actual == Movimiento.Tipo.TRASLADO_ENTRE:
                    alm_dest_nom_o = propuesta_activa.get("almacen_destino_nombre") or "almacén destino"
                    ub_dest_nom_o = propuesta_activa.get("ubicacion_destino_nombre") or ""
                    propuesta_activa["motivo"] = f"Traslado vía Haku de {alm_match.nombre} a {alm_dest_nom_o}" + (f" ({ub_dest_nom_o})" if ub_dest_nom_o else "")
                nuevo_val = alm_match.nombre

            elif campo == "motivo":
                motivo_val = (nuevo_val or "").strip()
                if "dotacion" in motivo_val.lower() or "dotación" in motivo_val.lower():
                    motivo_val = "Dotación"
                propuesta_activa["motivo"] = motivo_val
                nuevo_val = motivo_val

            elif campo == "observacion":
                accion_obs = str((semantica.datos_extra or {}).get("accion_observacion") or "establecer").lower()
                if accion_obs == "limpiar":
                    propuesta_activa["observacion"] = ""
                    nuevo_val = "(sin observación)"
                else:
                    obs_val = str(nuevo_val or "").strip()[:500]
                    if not obs_val:
                        return _marcar_modificacion_sin_resolver(
                            "¿Qué observación desea registrar en esta operación?",
                            "¿Qué observación desea registrar?",
                        )
                    propuesta_activa["observacion"] = obs_val
                    nuevo_val = obs_val
            elif campo == "es_prestable" and tipo_op_actual == "CREAR_PRODUCTO":
                es_prest_nuevo = bool((semantica.datos_extra or {}).get("es_prestable", False))
                propuesta_activa["es_prestable"] = es_prest_nuevo
                if not isinstance(propuesta_activa.get("datos_extra"), dict):
                    propuesta_activa["datos_extra"] = {}
                propuesta_activa["datos_extra"]["es_prestable"] = es_prest_nuevo
                nuevo_val = "Prestable / retornable" if es_prest_nuevo else "Consumible (no prestable)"
            else:
                return _marcar_modificacion_sin_resolver(
                    "No identifiqué el cambio solicitado sobre la propuesta. Indique claramente qué dato (producto, cantidad, destinatario, almacén u observación) desea modificar.",
                    "No identifiqué el cambio solicitado. Indique qué dato desea modificar.",
                )

            # Regenerar ID / UUID para invalidar la versión previa y registrar linaje (Garantía Sección 7 y Fase F)
            prev_id = str(propuesta_activa.get("id") or "").strip()
            if prev_id:
                propuesta_activa["origen_modificacion_id"] = prev_id
                if sesion is not None and hasattr(sesion, "__setitem__"):
                    sesion["haku_propuesta_sustituida_id"] = prev_id
                    sesion["haku_ultimo_evento"] = {
                        "propuesta_id": prev_id,
                        "estado": "SUSTITUIDA",
                        "fecha": round(time.time(), 4),
                    }
            propuesta_activa.pop("modificacion_sin_resolver", None)
            propuesta_activa["id"] = str(uuid.uuid4())
            propuesta_activa["expira_en"] = round(time.time() + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4)
            propuesta_activa["creado_en"] = round(time.time(), 4)
            propuesta_activa["version_contrato"] = VERSION_FIRMA_PROPUESTA

            cid_ctx, thash_ctx = _resolver_contexto_contrato(sesion, conversacion_id or propuesta_activa.get("conversacion_id"))
            propuesta_activa["conversacion_id"] = cid_ctx
            propuesta_activa["session_token_hash"] = thash_ctx
            if sesion is not None and hasattr(sesion, "get"):
                token_s = sesion.get("haku_session_token")
                if token_s:
                    propuesta_activa["session_token"] = token_s

            calcular_huella_propuesta(propuesta_activa)

            if prev_id and usuario and getattr(usuario, "id", None):
                try:
                    from .servicios_sesion import registrar_auditoria_modificacion
                    registrar_auditoria_modificacion(
                        usuario=usuario,
                        propuesta_anterior_id=prev_id,
                        propuesta_nueva=propuesta_activa,
                        campos_modificados=[campo] if campo else ["propuesta"],
                    )
                except Exception:
                    pass

            if estado_operativo:
                estado_operativo.propuesta_activa = propuesta_activa
                HakuOperationalContext.guardar(estado_operativo)
            if sesion is not None:
                sesion["haku_propuesta_activa"] = propuesta_activa

            nombre_p = propuesta_activa.get("producto_nombre", "el producto")
            cod_p = propuesta_activa.get("producto_codigo", "")
            cant = propuesta_activa.get("cantidad", 1)
            dest = propuesta_activa.get("destinatario")
            alm_nom = propuesta_activa.get("almacen_origen_nombre") or propuesta_activa.get("almacen_nombre") or "el almacén"
            tipo_op = propuesta_activa.get("tipo", propuesta_activa.get("tipo_operacion", "operación")).lower()
            motivo_p = propuesta_activa.get("motivo")
            obs_p = (propuesta_activa.get("observacion") or "").strip()
            prod_label = f"{nombre_p} (`{cod_p}`)" if cod_p else nombre_p

            if tipo_op_actual == "CREAR_PRODUCTO":
                cat_nom_p = propuesta_activa.get("categoria_nombre") or (propuesta_activa.get("datos_extra") or {}).get("categoria_nombre") or ""
                ubi_nom_p = propuesta_activa.get("ubicacion_origen_nombre") or (propuesta_activa.get("datos_extra") or {}).get("ubicacion_origen_nombre") or ""
                stk_min_p = propuesta_activa.get("stock_minimo")
                if stk_min_p is None:
                    stk_min_p = (propuesta_activa.get("datos_extra") or {}).get("stock_minimo", 0)
                es_prest_p = bool(
                    propuesta_activa.get("es_prestable")
                    if "es_prestable" in propuesta_activa
                    else (propuesta_activa.get("datos_extra") or {}).get("es_prestable", False)
                )
                tipo_art_p = "Prestable / retornable" if es_prest_p else "Consumible (no prestable)"
                mapa_et_crear = {
                    "es_prestable": "Tipo de artículo",
                    "cantidad": "Stock inicial",
                    "stock_minimo": "Stock mínimo",
                    "producto": "Nombre del producto",
                    "observacion": "Observación",
                }
                etiqueta_campo_p = mapa_et_crear.get(campo, campo.capitalize())
                v_txt = (
                    f"He actualizado {etiqueta_campo_p.lower()} a {nuevo_val}. "
                    f"Producto {nombre_p} ({'prestable' if es_prest_p else 'consumible'}), "
                    f"stock inicial {cant}, stock mínimo {stk_min_p}, en {alm_nom}. ¿Desea confirmar la creación?"
                )
                p_txt = (
                    f"⚠️ **Propuesta Actualizada** ({etiqueta_campo_p}: **{nuevo_val}**)\n\n"
                    f"- **Operación:** Creación de nuevo producto\n"
                    f"- **Producto:** {prod_label}\n"
                    + (f"- **Categoría:** {cat_nom_p}\n" if cat_nom_p else "")
                    + f"- **Tipo de artículo:** {tipo_art_p}\n"
                    + f"- **Almacén:** {alm_nom}\n"
                    + (f"- **Ubicación:** {ubi_nom_p}\n" if ubi_nom_p else "")
                    + f"- **Stock inicial:** {cant} unidades\n"
                    + f"- **Stock mínimo:** {stk_min_p} unidades\n"
                    + (f"- **Observación:** {obs_p}\n" if obs_p else "")
                    + "\n¿Desea confirmar la operación con estos datos actualizados?"
                )
            elif tipo_op_actual in ("HABILITAR_PRODUCTO", "INHABILITAR_PRODUCTO"):
                op_cat_lbl = "Habilitar / reactivar producto" if tipo_op_actual == "HABILITAR_PRODUCTO" else "Inhabilitar / archivar producto"
                etiqueta_campo_p = "Observación" if campo == "observacion" else campo.capitalize()
                v_txt = f"He actualizado {etiqueta_campo_p.lower()} a {nuevo_val}. Propuesta de {op_cat_lbl.lower()}: {nombre_p}. ¿Desea confirmar la operación?"
                p_txt = (
                    f"⚠️ **Propuesta Actualizada** ({etiqueta_campo_p}: **{nuevo_val}**)\n\n"
                    f"- **Operación:** {op_cat_lbl}\n"
                    f"- **Producto:** {prod_label}\n"
                    + (f"- **Observación:** {obs_p}\n" if obs_p else "")
                    + "\n¿Desea confirmar la operación con estos datos actualizados?"
                )
            elif tipo_op_actual == Movimiento.Tipo.TRASLADO_ENTRE:
                alm_dest_nom = propuesta_activa.get("almacen_destino_nombre") or "almacén destino"
                ubi_dest_nom = propuesta_activa.get("ubicacion_destino_nombre") or ""
                mapa_et_tr = {
                    "almacen_destino": "Almacén destino",
                    "ubicacion_destino": "Ubicación destino",
                    "almacen": "Almacén origen",
                    "observacion": "Observación",
                }
                etiqueta_campo_p = mapa_et_tr.get(campo, campo.capitalize())
                if campo == "observacion" and not obs_p:
                    v_txt = f"He eliminado la observación. Traslado de {cant} unidad(es) de {nombre_p} desde {alm_nom} hacia {alm_dest_nom}"
                else:
                    v_txt = f"He actualizado {etiqueta_campo_p.lower()} a {nuevo_val}. Traslado de {cant} unidad(es) de {nombre_p} desde {alm_nom} hacia {alm_dest_nom}"
                if ubi_dest_nom:
                    v_txt += f", en la ubicación {ubi_dest_nom}"
                v_txt += "."
                if obs_p:
                    v_txt += f" Observación: {obs_p}."
                v_txt += " ¿Desea confirmar la operación?"
                p_txt = (
                    f"⚠️ **Propuesta Actualizada** ({etiqueta_campo_p}: **{nuevo_val}**)\n\n"
                    f"- **Operación:** Traslado entre almacenes\n"
                    f"- **Producto:** {prod_label}\n"
                    f"- **Cantidad:** {cant}\n"
                    f"- **Almacén origen:** {alm_nom}\n"
                    f"- **Almacén destino:** {alm_dest_nom}\n"
                    + (f"- **Ubicación destino:** {ubi_dest_nom}\n" if ubi_dest_nom else "")
                    + (f"- **Motivo:** {motivo_p}\n" if motivo_p else "")
                    + (f"- **Observación:** {obs_p}\n" if obs_p else "")
                    + "\n¿Desea confirmar la operación con estos datos actualizados?"
                )
            else:
                if campo == "observacion" and not obs_p:
                    v_txt = f"He eliminado la observación. Propuesta de {tipo_op}: {cant} unidad(es) de {nombre_p}"
                else:
                    v_txt = f"He actualizado {campo} a {nuevo_val}. Propuesta de {tipo_op}: {cant} unidad(es) de {nombre_p}"
                if dest:
                    v_txt += f" para {dest}"
                v_txt += f" en {alm_nom}."
                if motivo_p:
                    v_txt += f" Motivo: {motivo_p}."
                if obs_p:
                    v_txt += f" Observación: {obs_p}."
                v_txt += " ¿Desea confirmar la operación?"

                etiqueta_campo_p = "Observación" if campo == "observacion" else campo.capitalize()
                p_txt = (
                    f"⚠️ **Propuesta Actualizada** ({etiqueta_campo_p}: **{nuevo_val}**)\n\n"
                    f"- **Operación:** {tipo_op.capitalize()}\n"
                    f"- **Producto:** {prod_label}\n"
                    f"- **Cantidad:** {cant}\n"
                )
                if dest:
                    etiqueta_dest_p = "Responsable" if tipo_op_actual in (Movimiento.Tipo.DEVOLUCION, "DEVOLUCION") else "Destinatario"
                    p_txt += f"- **{etiqueta_dest_p}:** {dest}\n"
                p_txt += f"- **Almacén:** {alm_nom}\n"
                if motivo_p:
                    p_txt += f"- **Motivo:** {motivo_p}\n"
                if obs_p:
                    p_txt += f"- **Observación:** {obs_p}\n"
                p_txt += "\n¿Desea confirmar la operación con estos datos actualizados?"

            return {
                "tipo": "propuesta",
                "requiere_confirmacion": True,
                "propuesta": propuesta_activa,
                "mensaje_pantalla": p_txt,
                "mensaje_verbal": normalizar_para_tts(v_txt),
                "datos": {
                    "propuesta_actualizada": True,
                    "propuesta_anterior_id": prev_id,
                    "campo_modificado": campo,
                    "nuevo_valor": nuevo_val,
                },
            }

        elif semantica.intencion == "confirmar":
            if time.time() > propuesta_activa.get("expira_en", 0):
                HakuDiagnostico.registrar_propuesta("expirada")
                if estado_operativo:
                    estado_operativo.propuesta_activa = None
                    HakuOperationalContext.guardar(estado_operativo)
                if sesion is not None:
                    sesion["haku_propuesta_activa"] = None
                return {
                    "tipo": "aclaracion",
                    "mensaje_pantalla": "La propuesta previa ha caducado por superar los 30 minutos. Por favor genera una nueva solicitud.",
                    "mensaje_verbal": "La propuesta ha caducado. Por favor solicita la operación de nuevo.",
                    "propuesta_activa": False,
                }
            if propuesta_activa.get("modificacion_sin_resolver"):
                msg_bloq = (
                    "Hay una corrección pendiente de aclarar sobre esta propuesta. "
                    "Antes de confirmar, por favor indique el dato exacto que desea corregir (producto, cantidad o destinatario) "
                    "o di **«Cancelar»** para descartar la operación."
                )
                return {
                    "tipo": "aclaracion",
                    "mensaje_pantalla": msg_bloq,
                    "mensaje_verbal": normalizar_para_tts(
                        "Hay una corrección sin resolver en la propuesta. Por favor aclare el dato que desea cambiar antes de confirmar, o cancele la operación."
                    ),
                    "propuesta_activa": True,
                    "datos": {"modificacion_pendiente": True, "confirmacion_bloqueada": True},
                }
            resp_conf = {
                "tipo": "accion_confirmar",
                "propuesta_id": propuesta_activa["id"],
                "mensaje_pantalla": "Confirmación recibida. Procediendo con la transacción...",
                "mensaje_verbal": "Confirmación recibida. Ejecutando la operación.",
            }
            if semantica is not None and getattr(semantica, "datos_extra", None):
                obs_inline_conf = semantica.datos_extra.get("observacion")
                if obs_inline_conf:
                    resp_conf["observacion_usuario"] = str(obs_inline_conf).strip()[:500]
            return resp_conf

        elif semantica.intencion == "cancelar":
            HakuDiagnostico.registrar_propuesta("cancelada")
            if estado_operativo:
                estado_operativo.propuesta_activa = None
                HakuOperationalContext.guardar(estado_operativo)
            if sesion is not None:
                sesion["haku_propuesta_activa"] = None
            return {
                "tipo": "accion_cancelar",
                "propuesta_id": propuesta_activa["id"],
                "mensaje_pantalla": "Operación cancelada. La propuesta ha sido descartada.",
                "mensaje_verbal": "Operación cancelada. No se realizaron cambios en el inventario.",
            }

    # 2. Descarte o ruptura de tema ("olvídate de X", "cambiemos de tema")
    if semantica.intencion == "cambio_tema":
        if estado_operativo:
            estado_operativo.limpiar_foco_conversacional()
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = None
        msg = "Entendido, cambiamos de tema. ¿Qué consulta u operación desea realizar?"
        return {
            "tipo": "conversacion",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
        }

    # 3. Selección contextual de alternativa ("el otro", "la otra")
    if semantica.intencion == "seleccionar_alternativa":
        if estado_operativo and getattr(estado_operativo, "candidatos", None):
            otro = estado_operativo.obtener_otro_candidato()
            if otro:
                prod_otro = Producto.objects.filter(id=otro["id"]).first()
                if prod_otro:
                    estado_operativo.asignar_producto(prod_otro.id, prod_otro.codigo_interno, prod_otro.nombre)
                    HakuOperationalContext.guardar(estado_operativo)
                    intencion = IntencionResultado(tipo=IntencionTipo.CONSULTAR_STOCK, producto=prod_otro)
            elif len(estado_operativo.candidatos) > 2:
                cands_str = ", ".join(f"{c['nombre']} ({c['codigo']})" for c in estado_operativo.candidatos)
                msg = f"Tenemos {len(estado_operativo.candidatos)} opciones pendientes: {cands_str}. ¿Cuál de ellos desea elegir?"
                return {
                    "tipo": "aclaracion",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts("Tenemos varias opciones pendientes. ¿Cuál de ellos desea elegir?"),
                    "datos": {"requiere_aclaracion": True},
                }

    # 4. Consultas de Atributos Específicos (Sección 5.3)
    if semantica.intencion == "consultar_atributo":
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar información del catálogo o existencias."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar el catálogo."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }
        prod_obj = None
        qs_prod = Producto.objects.all() if (usuario and getattr(usuario, "es_administrador", False)) else Producto.objects.filter(activo=True, archivado=False)
        if usuario and getattr(usuario, "es_usuario_normal", False) and not getattr(usuario, "acceso_todos_almacenes", False):
            alms_accesibles = usuario.obtener_almacenes_accesibles()
            if alms_accesibles.exists():
                qs_prod = qs_prod.filter(
                    Q(categoria__almacen_principal__in=alms_accesibles) |
                    Q(existencias__almacen__in=alms_accesibles)
                ).distinct()
            else:
                qs_prod = qs_prod.none()
        if semantica.producto:
            nom_prod = re.sub(r"^(?:el|la|los|las|un|una|unos|unas|del?|de\s+la)\s+", "", semantica.producto, flags=re.IGNORECASE).strip()
            from .lenguaje.normalizador import NormalizadorLenguajeHaku
            prod_obj, cod_p, cands = NormalizadorLenguajeHaku._resolver_producto_en_texto(
                nom_prod, usuario
            )
            if not prod_obj and cands:
                if len(cands) == 1:
                    prod_obj = qs_prod.filter(id=cands[0]["id"]).first()
                else:
                    cands_str = ", ".join(f"{c['nombre']} ({c['codigo']})" for c in cands[:3])
                    int_orig = IntencionTipo.CONSULTAR_UBICACION.value if semantica.atributo == "ubicacion" else IntencionTipo.CONSULTAR_STOCK.value
                    if estado_operativo:
                        estado_operativo.asignar_candidatos(
                            cands,
                            intencion_original=int_orig,
                            datos_extra={"atributo": semantica.atributo},
                        )
                        HakuOperationalContext.guardar(estado_operativo)
                    msg_cands = f"Encontré varias coincidencias para su consulta: {cands_str}. ¿Cuál de ellos desea consultar?"
                    return {
                        "tipo": "aclaracion",
                        "mensaje_pantalla": msg_cands,
                        "mensaje_verbal": normalizar_para_tts(msg_cands),
                        "datos": {
                            "requiere_aclaracion": True,
                            "candidatos": [
                                {"id": c["id"], "nombre": c["nombre"], "codigo": c["codigo"]}
                                for c in cands
                            ],
                        },
                    }
            if not prod_obj:
                m_cod = re.search(r"\b([A-Z]{2,5}-\d{1,4})\b", nom_prod.upper())
                if m_cod:
                    prod_obj = qs_prod.filter(codigo_interno__iexact=m_cod.group(1)).first()
                if not prod_obj:
                    prod_obj = qs_prod.filter(
                        Q(nombre__icontains=nom_prod) | Q(codigo_interno__iexact=nom_prod)
                    ).first()
        elif estado_operativo and estado_operativo.producto_id:
            prod_obj = qs_prod.filter(id=estado_operativo.producto_id).first()
        elif sesion and sesion.get("haku_ultimo_producto_id"):
            prod_obj = qs_prod.filter(id=sesion.get("haku_ultimo_producto_id")).first()

        if not prod_obj:
            msg = "¿De cuál producto desea conocer esa información?"
            return {
                "tipo": "aclaracion",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
            }

        if estado_operativo:
            estado_operativo.asignar_producto(prod_obj.id, prod_obj.codigo_interno, prod_obj.nombre)
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(prod_obj.id)
            sesion["haku_ultimo_acceso"] = time.time()

        if not semantica.atributo or semantica.atributo in ("stock", "general", "informacion", "detalle"):
            intencion = IntencionResultado(tipo=IntencionTipo.CONSULTAR_STOCK, producto=prod_obj)
        elif semantica.atributo == "categoria":
            p_msg = f"El producto **{prod_obj.nombre}** (`{prod_obj.codigo_interno}`) pertenece a la categoría **{prod_obj.categoria.nombre}**."
            v_msg = f"{prod_obj.nombre} pertenece a la categoría {prod_obj.categoria.nombre}."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": p_msg,
                "mensaje_verbal": normalizar_para_tts(v_msg),
                "datos": {"producto_id": str(prod_obj.id), "categoria": prod_obj.categoria.nombre},
            }
        elif semantica.atributo == "stock_minimo":
            p_msg = f"El stock mínimo registrado para **{prod_obj.nombre}** (`{prod_obj.codigo_interno}`) es de {pluralizar_md(prod_obj.stock_minimo, 'unidad', 'unidades')}."
            v_msg = f"El stock mínimo de {prod_obj.nombre} es de {pluralizar(prod_obj.stock_minimo, 'unidad', 'unidades')}."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": p_msg,
                "mensaje_verbal": normalizar_para_tts(v_msg),
                "datos": {"producto_id": str(prod_obj.id), "stock_minimo": prod_obj.stock_minimo},
            }
        elif semantica.atributo == "codigo":
            p_msg = f"El código interno de **{prod_obj.nombre}** es **{prod_obj.codigo_interno}**."
            v_msg = f"El código de {prod_obj.nombre} es {prod_obj.codigo_interno}."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": p_msg,
                "mensaje_verbal": normalizar_para_tts(v_msg),
                "datos": {"producto_id": str(prod_obj.id), "codigo": prod_obj.codigo_interno},
            }
        elif semantica.atributo == "prestados":
            if not (usuario_tiene_permiso_operativo(usuario, "prestamos.ver") or usuario_tiene_permiso_operativo(usuario, "movimientos.ver") or usuario_tiene_permiso_operativo(usuario, "movimientos.prestamo")):
                msg = "No cuentas con el permiso requerido (prestamos.ver) para consultar préstamos activos."
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar los préstamos activos."),
                    "datos": {"error_permiso": True, "permiso_requerido": "prestamos.ver"},
                }
            alms_acc = usuario.obtener_almacenes_accesibles() if hasattr(usuario, "obtener_almacenes_accesibles") else None
            qs_p = Prestamo.objects.filter(
                producto=prod_obj,
                estado__in=[Prestamo.Estado.ABIERTO, Prestamo.Estado.DEVUELTO_PARCIAL],
            )
            if alms_acc is not None:
                qs_p = qs_p.filter(almacen__in=alms_acc)
            qs_p = qs_p.select_related("usuario")
            tot_p = sum(p.cantidad_pendiente_devolucion for p in qs_p)
            if tot_p > 0:
                borrowers = [f"{p.destinatario or (p.usuario.get_full_name() if p.usuario else 'Usuario')} ({p.cantidad_pendiente_devolucion} u.)" for p in qs_p]
                p_msg = f"Actualmente hay {pluralizar_md(tot_p, 'unidad', 'unidades')} de **{prod_obj.nombre}** en préstamo:\n\n" + "\n".join(f"- {b}" for b in borrowers)
                v_msg = f"Hay {pluralizar(tot_p, 'unidad', 'unidades')} de {prod_obj.nombre} en préstamo."
            else:
                p_msg = f"No hay préstamos activos de **{prod_obj.nombre}** en este momento."
                v_msg = f"No hay préstamos activos de {prod_obj.nombre}."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": p_msg,
                "mensaje_verbal": normalizar_para_tts(v_msg),
                "datos": {"producto_id": str(prod_obj.id), "total_prestado": tot_p, "total_unidades": tot_p},
            }
        elif semantica.atributo == "ubicacion":
            intencion = IntencionResultado(tipo=IntencionTipo.CONSULTAR_UBICACION, producto=prod_obj)
        elif semantica.atributo == "historial":
            if not usuario_tiene_permiso_operativo(usuario, "movimientos.ver"):
                msg = "No cuentas con el permiso requerido (movimientos.ver) para consultar el historial de movimientos."
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar el historial de movimientos."),
                    "datos": {"error_permiso": True, "permiso_requerido": "movimientos.ver"},
                }
            intencion = IntencionResultado(tipo=IntencionTipo.CONSULTAR_PRODUCTO_MOVIMIENTOS, producto=prod_obj)

    elif semantica.intencion in ("crear_producto", "alta_producto", "nuevo_producto"):
        return _procesar_alta_producto_conversacional(
            texto=texto,
            usuario=usuario,
            estado_operativo=estado_operativo,
            sesion=sesion,
            semantica=semantica,
            op_pendiente=op_pendiente,
            conversacion_id=conversacion_id,
        )

    elif semantica.intencion in ("inhabilitar_producto", "inhabilitar") or (
        semantica.intencion == "modificar_propuesta"
        and semantica.campo_modificar == "estado"
        and ("inhab" in str(semantica.nuevo_valor).lower() or "desact" in str(semantica.nuevo_valor).lower())
    ):
        from .agente.planificador import tiene_referencia_contextual_operacion
        prod_inhab = None
        qs_admin = Producto.objects.all() if (usuario and getattr(usuario, "es_administrador", False)) else Producto.objects.filter(activo=True, archivado=False)
        tiene_ref_ctx = tiene_referencia_contextual_operacion(texto)
        if semantica.producto and not tiene_ref_ctx:
            if estado_operativo and estado_operativo.producto_nombre and (
                estado_operativo.producto_nombre.lower() in semantica.producto.lower()
                or semantica.producto.lower() in estado_operativo.producto_nombre.lower()
            ):
                if not any(tok in texto_lower for tok in semantica.producto.lower().split() if len(tok) > 3):
                    semantica.producto = None
        if semantica.producto:
            from .lenguaje.normalizador import NormalizadorLenguajeHaku
            prod_inhab, _, _ = NormalizadorLenguajeHaku._resolver_producto_en_texto(semantica.producto, usuario)
        elif tiene_ref_ctx:
            if estado_operativo and estado_operativo.producto_id:
                prod_inhab = qs_admin.filter(id=estado_operativo.producto_id).first()
            elif sesion and sesion.get("haku_ultimo_producto_id"):
                prod_inhab = qs_admin.filter(id=sesion.get("haku_ultimo_producto_id")).first()
        intencion = IntencionResultado(
            tipo=IntencionTipo.INHABILITAR_PRODUCTO,
            producto=prod_inhab,
            requiere_confirmacion=True,
        )

    elif semantica.intencion in ("habilitar_producto", "habilitar") or (
        semantica.intencion == "modificar_propuesta"
        and semantica.campo_modificar == "estado"
        and ("hab" in str(semantica.nuevo_valor).lower() or "act" in str(semantica.nuevo_valor).lower())
    ):
        from .agente.planificador import tiene_referencia_contextual_operacion
        prod_hab = None
        qs_admin = Producto.objects.all() if (usuario and getattr(usuario, "es_administrador", False)) else Producto.objects.filter(activo=True, archivado=False)
        tiene_ref_ctx = tiene_referencia_contextual_operacion(texto)
        if semantica.producto and not tiene_ref_ctx:
            if estado_operativo and estado_operativo.producto_nombre and (
                estado_operativo.producto_nombre.lower() in semantica.producto.lower()
                or semantica.producto.lower() in estado_operativo.producto_nombre.lower()
            ):
                if not any(tok in texto_lower for tok in semantica.producto.lower().split() if len(tok) > 3):
                    semantica.producto = None
        if semantica.producto:
            from .lenguaje.normalizador import NormalizadorLenguajeHaku
            prod_hab, _, _ = NormalizadorLenguajeHaku._resolver_producto_en_texto(
                semantica.producto, usuario, incluir_inactivos=True,
            )
        elif tiene_ref_ctx:
            if estado_operativo and estado_operativo.producto_id:
                prod_hab = qs_admin.filter(id=estado_operativo.producto_id).first()
            elif sesion and sesion.get("haku_ultimo_producto_id"):
                prod_hab = qs_admin.filter(id=sesion.get("haku_ultimo_producto_id")).first()
        intencion = IntencionResultado(
            tipo=IntencionTipo.HABILITAR_PRODUCTO,
            producto=prod_hab,
            requiere_confirmacion=True,
        )

    # 5. Consultas de Catálogo (Global y Almacén)
    elif semantica.intencion == "consultar_catalogo":
        intencion = IntencionResultado(tipo=IntencionTipo.LISTAR_CATALOGO)
    elif semantica.intencion == "consultar_catalogo_almacen":
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar el catálogo del almacén."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar el catálogo."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }
        alm_obj = _buscar_almacen_por_nombre(semantica.almacen)
        if alm_obj:
            if not usuario.tiene_acceso_almacen(alm_obj):
                p_msg = f"No cuenta con permisos para consultar el almacén **{alm_obj.nombre}**."
                v_msg = f"No cuenta con permiso para consultar el {alm_obj.nombre}."
                return {
                    "tipo": "error",
                    "mensaje_pantalla": p_msg,
                    "mensaje_verbal": normalizar_para_tts(v_msg),
                }
            existencias = Existencia.objects.filter(almacen=alm_obj).select_related("producto")
            total_ref = existencias.count()
            con_stock_qs = existencias.filter(stock_disponible__gt=0)
            con_stock_count = con_stock_qs.count()
            total_unidades = sum(e.stock_disponible for e in con_stock_qs)
            agotados_count = existencias.filter(stock_disponible=0).count()

            p_msg = f"### Catálogo de {alm_obj.nombre}\n\nHay **{total_ref}** productos registrados en total: **{con_stock_count}** con existencias disponibles ({total_unidades} unidades en total) y **{agotados_count}** agotados.\n\n"
            if con_stock_count > 0:
                p_msg += "Algunos productos disponibles:\n"
                for ex in con_stock_qs[:10]:
                    p_msg += f"- **{ex.producto.nombre}** (`{ex.producto.codigo_interno}`): {ex.stock_disponible} u.\n"

            v_msg = f"En el {alm_obj.nombre} hay {total_ref} productos registrados en total: {con_stock_count} con existencias disponibles y {agotados_count} agotados."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": p_msg,
                "mensaje_verbal": normalizar_para_tts(v_msg),
                "datos": {
                    "almacen": alm_obj.nombre,
                    "total_referencias": total_ref,
                    "con_stock": con_stock_count,
                    "agotados": agotados_count,
                    "total_unidades": total_unidades,
                },
            }
        else:
            p_msg = f"El almacén '{semantica.almacen}' no existe en el sistema."
            v_msg = f"No encontré el almacén {semantica.almacen}."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": p_msg,
                "mensaje_verbal": normalizar_para_tts(v_msg),
            }

    elif semantica.intencion == "prestamos_activos":
        intencion = IntencionResultado(
            tipo=IntencionTipo.PRESTAMOS_ACTIVOS,
            almacen=_buscar_almacen_por_nombre(semantica.almacen) if semantica.almacen else None,
            datos_extra={"metrica": semantica.datos_extra.get("metrica")} if semantica.datos_extra else {},
        )

    elif semantica.intencion == "consultar_reposicion":
        if semantica.filtro_reposicion == "agotados":
            intencion = IntencionResultado(tipo=IntencionTipo.PRODUCTOS_SIN_STOCK)
        else:
            intencion = IntencionResultado(tipo=IntencionTipo.CONSULTAR_FALTANTES_REPOSICION)

    elif semantica.intencion == "consultar_movimientos":
        intencion = IntencionResultado(tipo=IntencionTipo.CONSULTAR_METRICAS_MOVIMIENTOS)

    elif semantica.intencion == "comparar_ubicaciones":
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para comparar ubicaciones de inventario."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar información del catálogo."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }
        loc1_raw = semantica.ubicacion or ""
        loc2_raw = semantica.destino or ""
        loc1 = re.sub(r"^(?:el\s+|la\s+|en\s+el\s+|en\s+la\s+|estante\s+|estanter[ií]a\s+|caj[oó]n\s+|repisa\s+|sitio\s+|lugar\s+)", "", loc1_raw, flags=re.IGNORECASE).strip().upper()
        loc2 = re.sub(r"^(?:el\s+|la\s+|en\s+el\s+|en\s+la\s+|estante\s+|estanter[ií]a\s+|caj[oó]n\s+|repisa\s+|sitio\s+|lugar\s+)", "", loc2_raw, flags=re.IGNORECASE).strip().upper()
        ub1 = Ubicacion.objects.filter(Q(nombre__iexact=loc1) | Q(clave_canonica=loc1.lower())).first()
        ub2 = Ubicacion.objects.filter(Q(nombre__iexact=loc2) | Q(clave_canonica=loc2.lower())).first()

        if not ub1:
            p_msg = f"No se encontró la ubicación física **{loc1}** en el inventario."
            v_msg = f"No encontré la ubicación {loc1}."
            return {"tipo": "consulta", "mensaje_pantalla": p_msg, "mensaje_verbal": normalizar_para_tts(v_msg)}
        if not ub2:
            p_msg = f"No se encontró la ubicación física **{loc2}** en el inventario."
            v_msg = f"No encontré la ubicación {loc2}."
            return {"tipo": "consulta", "mensaje_pantalla": p_msg, "mensaje_verbal": normalizar_para_tts(v_msg)}

        if not usuario.tiene_acceso_almacen(ub1.almacen):
            p_msg = f"No cuenta con permisos para consultar el almacén **{ub1.almacen.nombre}** (ubicación {ub1.nombre})."
            v_msg = f"No cuenta con permiso para consultar el almacén {ub1.almacen.nombre}."
            return {"tipo": "error", "mensaje_pantalla": p_msg, "mensaje_verbal": normalizar_para_tts(v_msg)}
        if not usuario.tiene_acceso_almacen(ub2.almacen):
            p_msg = f"No cuenta con permisos para consultar el almacén **{ub2.almacen.nombre}** (ubicación {ub2.nombre})."
            v_msg = f"No cuenta con permiso para consultar el almacén {ub2.almacen.nombre}."
            return {"tipo": "error", "mensaje_pantalla": p_msg, "mensaje_verbal": normalizar_para_tts(v_msg)}

        c1 = ub1.productos.filter(activo=True, archivado=False).count()
        c2 = ub2.productos.filter(activo=True, archivado=False).count()
        p_msg = f"### Comparación entre {ub1.nombre} y {ub2.nombre}\n\n- **{ub1.nombre}** ({ub1.almacen.nombre}): **{c1}** productos registrados (asignación provisional).\n- **{ub2.nombre}** ({ub2.almacen.nombre}): **{c2}** productos registrados.\n"
        v_msg = f"En {ub1.nombre} hay {c1} productos registrados y en {ub2.nombre} hay {c2}."
        return {
            "tipo": "consulta",
            "mensaje_pantalla": p_msg,
            "mensaje_verbal": normalizar_para_tts(v_msg),
            "datos": {"ub1": ub1.nombre, "c1": c1, "ub2": ub2.nombre, "c2": c2},
        }

    elif semantica.intencion == "consultar_ubicaciones":
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar ubicaciones de inventario."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar las ubicaciones del inventario."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }
        if semantica.ubicacion:
            loc_clean = re.sub(r"^(?:el\s+|la\s+|en\s+el\s+|en\s+la\s+|estante\s+|estanter[ií]a\s+|caj[oó]n\s+|repisa\s+|sitio\s+|lugar\s+)", "", semantica.ubicacion, flags=re.IGNORECASE).strip().upper()
            ub_obj = Ubicacion.objects.filter(
                Q(nombre__iexact=loc_clean) | Q(nombre__iexact=semantica.ubicacion) | Q(clave_canonica=loc_clean.lower())
            ).first()

            if ub_obj:
                # Verificación estricta de permisos sobre el almacén propietario
                if not usuario.tiene_acceso_almacen(ub_obj.almacen):
                    p_msg = f"No cuenta con permisos para consultar el almacén **{ub_obj.almacen.nombre}** (ubicación {ub_obj.nombre})."
                    v_msg = f"No cuenta con permiso para consultar el almacén {ub_obj.almacen.nombre}."
                    return {
                        "tipo": "error",
                        "mensaje_pantalla": p_msg,
                        "mensaje_verbal": normalizar_para_tts(v_msg),
                    }

                metrica = semantica.datos_extra.get("metrica", "")

                # Caso 1: Consulta de unidades físicas en la ubicación
                if metrica == "unidades":
                    p_msg = f"No es posible determinar las unidades físicas almacenadas en **{ub_obj.nombre}**; en el sistema las existencias se registran por almacén (**{ub_obj.almacen.nombre}**), no por estante o ubicación individual."
                    v_msg = f"No puedo determinar las unidades de {ub_obj.nombre}; el stock se registra por almacén."
                    return {
                        "tipo": "consulta",
                        "mensaje_pantalla": p_msg,
                        "mensaje_verbal": normalizar_para_tts(v_msg),
                        "datos": {"ubicacion": ub_obj.nombre, "almacen": ub_obj.almacen.nombre, "stock_por_ubicacion": False},
                    }

                # Caso 2: Conteo de productos distintos asociados
                prods_count = ub_obj.productos.filter(activo=True, archivado=False).count()
                if metrica == "productos":
                    if prods_count > 0:
                        p_msg = f"En **{ub_obj.nombre}** ({ub_obj.almacen.nombre}) hay **{prods_count}** productos registrados (asignación provisional)."
                        v_msg = f"En {ub_obj.nombre} hay {prods_count} productos registrados."
                    else:
                        p_msg = f"No hay productos registrados en **{ub_obj.nombre}** ({ub_obj.almacen.nombre})."
                        v_msg = f"No hay productos registrados en {ub_obj.nombre}."
                    return {
                        "tipo": "consulta",
                        "mensaje_pantalla": p_msg,
                        "mensaje_verbal": normalizar_para_tts(v_msg),
                        "datos": {"ubicacion": ub_obj.nombre, "total_productos": prods_count},
                    }

                # Caso 3: Contenido general (¿Qué hay en X?)
                prods = list(ub_obj.productos.filter(activo=True, archivado=False))
                if prods:
                    p_msg = f"### Ubicación: {ub_obj.nombre} ({ub_obj.almacen.nombre})\n\n"
                    p_msg += f"Hay **{len(prods)}** productos registrados (asignación provisional):\n"
                    for pr in prods[:10]:
                        p_msg += f"- **{pr.nombre}** (`{pr.codigo_interno}`)\n"
                    v_msg = f"En {ub_obj.nombre} hay {len(prods)} productos registrados. Te muestro la lista en pantalla."
                else:
                    p_msg = f"No hay productos registrados en **{ub_obj.nombre}** ({ub_obj.almacen.nombre})."
                    v_msg = f"No hay productos registrados en {ub_obj.nombre}."
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": p_msg,
                    "mensaje_verbal": normalizar_para_tts(v_msg),
                    "datos": {"ubicacion": ub_obj.nombre, "total_productos": len(prods), "productos": [pr.nombre for pr in prods]},
                }
            else:
                p_msg = f"No se encontró la ubicación física **{semantica.ubicacion}** en el inventario."
                v_msg = f"No encontré la ubicación {semantica.ubicacion}."
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": p_msg,
                    "mensaje_verbal": normalizar_para_tts(v_msg),
                }
        else:
            intencion = IntencionResultado(tipo=IntencionTipo.LISTAR_UBICACIONES)

    elif semantica.intencion == "consultar_stock":
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar existencias o stock de productos."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar el stock de productos."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }
        # A. Caso familia o categoría (ej: "cables", "herramientas", "redes")
        if semantica.datos_extra.get("es_familia"):
            termino_fam = semantica.producto or ""
            prods_fam = Producto.objects.filter(
                Q(categoria__nombre__icontains=termino_fam) | Q(nombre__icontains=termino_fam),
                activo=True,
                archivado=False,
            )
            if not prods_fam.exists():
                p_msg = f"No hay **{termino_fam}** registrados en el inventario."
                v_msg = f"No hay {termino_fam} registrados en el inventario."
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": p_msg,
                    "mensaje_verbal": normalizar_para_tts(v_msg),
                    "datos": {"familia": termino_fam, "encontrados": 0},
                }
            alms_acc = usuario.obtener_almacenes_accesibles()
            existencias = Existencia.objects.filter(producto__in=prods_fam, almacen__in=alms_acc)
            tot_ref = prods_fam.count()
            tot_units = sum(e.stock_disponible for e in existencias)
            p_msg = f"De **{termino_fam}** tenemos **{tot_ref}** referencias con un total de **{tot_units}** unidades disponibles en sus almacenes autorizados."
            v_msg = f"De {termino_fam} tenemos {tot_ref} referencias con {tot_units} unidades disponibles."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": p_msg,
                "mensaje_verbal": normalizar_para_tts(v_msg),
                "datos": {"familia": termino_fam, "total_referencias": tot_ref, "total_unidades": tot_units},
            }

        # B. Consulta de producto específico (por código o nombre)
        prod_obj = None
        qs_prod = Producto.objects.all() if (usuario and getattr(usuario, "es_administrador", False)) else Producto.objects.filter(activo=True, archivado=False)
        if usuario and getattr(usuario, "es_usuario_normal", False) and not getattr(usuario, "acceso_todos_almacenes", False):
            alms_accesibles = usuario.obtener_almacenes_accesibles()
            if alms_accesibles.exists():
                qs_prod = qs_prod.filter(
                    Q(categoria__almacen_principal__in=alms_accesibles) |
                    Q(existencias__almacen__in=alms_accesibles)
                ).distinct()
            else:
                qs_prod = qs_prod.none()
        if semantica.producto:
            from .lenguaje.siglas import NormalizadorSiglasTecnicas
            prod_norm = NormalizadorSiglasTecnicas.normalizar(semantica.producto)
            nom_p = re.sub(r"^(?:el|la|los|las|un|una|unos|unas|del?|de\s+la)\s+", "", prod_norm, flags=re.IGNORECASE).strip()
            # Búsqueda exacta por código interno
            prod_obj = qs_prod.filter(codigo_interno__iexact=nom_p).first()
            if not prod_obj:
                from .lenguaje.normalizador import NormalizadorLenguajeHaku
                prod_obj, cod_p, cands = NormalizadorLenguajeHaku._resolver_producto_en_texto(nom_p, usuario)
            if not prod_obj:
                if cands:
                    if estado_operativo:
                        estado_operativo.asignar_candidatos(
                            cands,
                            intencion_original=semantica.intencion,
                            datos_extra={"almacen": semantica.almacen},
                        )
                        from .contexto import HakuOperationalContext
                        HakuOperationalContext.guardar(estado_operativo)
                    msg_cands = cod_p or "Hay varios productos que coinciden con su consulta. ¿Cuál de ellos desea consultar?"
                    return {
                        "tipo": "aclaracion",
                        "mensaje_pantalla": msg_cands,
                        "mensaje_verbal": normalizar_para_tts(msg_cands),
                        "datos": {"candidatos": cands, "requiere_aclaracion": True},
                    }
                # Prioridad estricta al código interno para evitar colisiones por substring en nombre
                m_cod = re.search(r"\b([A-Z]{2,5}-\d{1,4})\b", nom_p.upper())
                if m_cod:
                    prod_obj = qs_prod.filter(codigo_interno__iexact=m_cod.group(1)).first()
                if not prod_obj:
                    if len(nom_p) >= 4:
                        prod_obj = qs_prod.filter(nombre__icontains=nom_p).first()
                    else:
                        for p_cand in qs_prod:
                            if re.search(rf"\b{re.escape(nom_p)}\b", p_cand.nombre, re.IGNORECASE):
                                prod_obj = p_cand
                                break
        elif estado_operativo and getattr(estado_operativo, "producto_id", None):
            prod_obj = qs_prod.filter(id=estado_operativo.producto_id).first()
        elif sesion and sesion.get("haku_ultimo_producto_id"):
            prod_obj = qs_prod.filter(id=sesion.get("haku_ultimo_producto_id")).first()

        if not prod_obj:
            # Comprobar si coincide con una familia antes de pedir aclaración
            if semantica.producto:
                prods_fam = Producto.objects.filter(
                    Q(categoria__nombre__icontains=semantica.producto) | Q(nombre__icontains=semantica.producto),
                    activo=True,
                    archivado=False,
                )
                if not prods_fam.exists():
                    p_msg = f"No hay **{semantica.producto}** registrados en el inventario."
                    v_msg = f"No hay {semantica.producto} registrados en el inventario."
                    return {
                        "tipo": "consulta",
                        "mensaje_pantalla": p_msg,
                        "mensaje_verbal": normalizar_para_tts(v_msg),
                        "datos": {"familia": semantica.producto, "encontrados": 0},
                    }
            msg = "¿De cuál producto desea consultar el stock?"
            return {
                "tipo": "aclaracion",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
            }

        alms_acc = usuario.obtener_almacenes_accesibles()
        if semantica.almacen:
            alm_filtro = _buscar_almacen_por_nombre(semantica.almacen)
            if alm_filtro:
                alm_val, err_val = _validar_acceso_almacen(usuario, alm_filtro)
                if err_val:
                    return err_val
                alms_acc = alms_acc.filter(id=alm_filtro.id)

        existencias = Existencia.objects.filter(producto=prod_obj, almacen__in=alms_acc).select_related("almacen")
        tot_disp = sum(e.stock_disponible for e in existencias)
        tot_prest = sum(e.stock_en_prestamo for e in existencias)
        ubis_noms = list(prod_obj.ubicaciones.filter(almacen__in=alms_acc).values_list("nombre", flat=True))
        ubi_txt = f" Ubicación: {', '.join(ubis_noms)}." if ubis_noms else ""

        if len(existencias) == 1:
            alm_nom_res = existencias[0].almacen.nombre
            p_msg = f"De **{prod_obj.codigo_interno}** (**{prod_obj.nombre}**) quedan {tot_disp} unidades disponibles en **{alm_nom_res}**.{ubi_txt}"
            v_msg = f"De {prod_obj.codigo_interno}, {prod_obj.nombre}, quedan {tot_disp} unidades disponibles en {alm_nom_res}."
        elif len(existencias) > 1:
            p_msg = f"De **{prod_obj.codigo_interno}** (**{prod_obj.nombre}**) hay un total de **{tot_disp}** unidades disponibles en sus almacenes autorizados.{ubi_txt}"
            v_msg = f"De {prod_obj.codigo_interno}, {prod_obj.nombre}, quedan {tot_disp} unidades disponibles."
        else:
            p_msg = f"De **{prod_obj.codigo_interno}** (**{prod_obj.nombre}**) quedan **0** unidades disponibles en sus almacenes autorizados."
            v_msg = f"De {prod_obj.codigo_interno}, {prod_obj.nombre}, quedan 0 unidades disponibles."

        if estado_operativo:
            estado_operativo.asignar_producto(prod_obj.id, prod_obj.codigo_interno, prod_obj.nombre)
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(prod_obj.id)
            sesion["haku_ultimo_acceso"] = time.time()

        return {
            "tipo": "consulta",
            "mensaje_pantalla": p_msg,
            "mensaje_verbal": normalizar_para_tts(v_msg),
            "datos": {
                "producto_id": str(prod_obj.id),
                "codigo": prod_obj.codigo_interno,
                "nombre": prod_obj.nombre,
                "stock_disponible": tot_disp,
                "stock_en_prestamo": tot_prest,
            },
        }

    # 6. Consumo Seguro de Propuestas de Operación Semántica (Contrato HAKU)
    elif semantica.intencion == "proponer_operacion":
        tipo_op_sem = (semantica.tipo_operacion or "").strip().lower()
        if tipo_op_sem in ("crear_producto", "crear", "alta_producto", "nuevo_producto"):
            return _procesar_alta_producto_conversacional(
                texto=texto,
                usuario=usuario,
                estado_operativo=estado_operativo,
                sesion=sesion,
                semantica=semantica,
                op_pendiente=op_pendiente,
                conversacion_id=conversacion_id,
            )
        tipo_int_map = {
            "prestamo": IntencionTipo.PRESTAMO,
            "registrar_prestamo": IntencionTipo.PRESTAMO,
            "devolucion": IntencionTipo.DEVOLUCION,
            "registrar_devolucion": IntencionTipo.DEVOLUCION,
            "entrada": IntencionTipo.REGISTRAR_ENTRADA,
            "ingreso": IntencionTipo.REGISTRAR_ENTRADA,
            "registrar_entrada": IntencionTipo.REGISTRAR_ENTRADA,
            "salida": IntencionTipo.REGISTRAR_SALIDA,
            "retiro": IntencionTipo.REGISTRAR_SALIDA,
            "registrar_salida": IntencionTipo.REGISTRAR_SALIDA,
            "dotacion": IntencionTipo.REGISTRAR_SALIDA,
            "salida_dotacion": IntencionTipo.REGISTRAR_SALIDA,
            "traslado": IntencionTipo.TRASLADAR_PRODUCTO,
            "trasladar": IntencionTipo.TRASLADAR_PRODUCTO,
            "mover_producto": IntencionTipo.TRASLADAR_PRODUCTO,
            "trasladar_producto": IntencionTipo.TRASLADAR_PRODUCTO,
        }
        tipo_intencion = tipo_int_map.get(tipo_op_sem)
        if tipo_intencion:
            # Validación de cantidad: debe ser mayor a cero si fue especifiqueda
            if semantica.cantidad is not None and semantica.cantidad <= 0:
                msg = "La cantidad debe ser un número entero mayor a cero."
                return {
                    "tipo": "error",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts(msg),
                    "datos": {"cantidad_invalida": semantica.cantidad},
                }

            # Resolución y validación del producto
            from .agente.planificador import tiene_referencia_contextual_operacion
            prod_obj = None
            candidatos = None

            # Regla de aislamiento estricto: Una nueva operación sin referencia contextual explícita NO hereda el producto previo
            tiene_ref = tiene_referencia_contextual_operacion(texto)

            if not tiene_ref and semantica.producto:
                from .agente.herramientas import normalizar_codigos_hablados_en_texto
                texto_cods_lower = normalizar_codigos_hablados_en_texto(texto).lower()
                if estado_operativo and estado_operativo.producto_nombre and (
                    estado_operativo.producto_nombre.lower() in semantica.producto.lower()
                    or semantica.producto.lower() in estado_operativo.producto_nombre.lower()
                    or (estado_operativo.producto_codigo and estado_operativo.producto_codigo.lower() == semantica.producto.lower())
                ):
                    if not any(token in texto_cods_lower for token in semantica.producto.lower().split() if len(token) > 3) and (
                        not estado_operativo.producto_codigo or estado_operativo.producto_codigo.lower() not in texto_cods_lower
                    ):
                        semantica.producto = None

            if semantica.producto:
                from .lenguaje.normalizador import NormalizadorLenguajeHaku
                prod_obj, cod_p, cands = NormalizadorLenguajeHaku._resolver_producto_en_texto(
                    semantica.producto, usuario
                )
                if not prod_obj and cands:
                    if len(cands) == 1:
                        prod_obj = Producto.objects.filter(id=cands[0]["id"], activo=True, archivado=False).first()
                    else:
                        candidatos = cands
            elif tiene_ref and estado_operativo:
                cands_ctx = getattr(estado_operativo, "candidatos", None) or []
                col_ctx = getattr(estado_operativo, "coleccion_productos", None) or []
                if len(cands_ctx) > 1:
                    candidatos = cands_ctx
                elif len(col_ctx) > 1 and not getattr(estado_operativo, "producto_id", None):
                    candidatos = col_ctx
                elif getattr(estado_operativo, "producto_id", None):
                    prod_obj = Producto.objects.filter(
                        id=estado_operativo.producto_id, activo=True, archivado=False
                    ).first()

            if not prod_obj and candidatos and len(candidatos) > 1:
                cands_str = ", ".join(f"{c['nombre']} ({c['codigo']})" for c in candidatos[:3])
                alm_cand_obj = None
                if semantica.almacen and "dotaci" not in semantica.almacen.lower():
                    alm_cand_obj = _buscar_almacen_por_nombre(semantica.almacen)
                de_cand = dict(semantica.datos_extra) if semantica.datos_extra else {}
                de_cand["tipo_operacion_pendiente"] = tipo_intencion.value
                dest_cand_nom = getattr(semantica, "destino", None) or de_cand.get("destino") or de_cand.get("almacen_destino")
                if not dest_cand_nom and tipo_intencion == IntencionTipo.TRASLADAR_PRODUCTO:
                    _alm_det_c = _detectar_almacen_destino_en_texto(texto)
                    dest_cand_nom = _alm_det_c.nombre if _alm_det_c else None
                if dest_cand_nom:
                    _alm_dest_c = _buscar_almacen_por_nombre(dest_cand_nom)
                    if _alm_dest_c:
                        de_cand["almacen_destino_id"] = _alm_dest_c.id
                        de_cand["almacen_destino_nombre"] = _alm_dest_c.nombre
                intencion = IntencionResultado(
                    tipo=IntencionTipo.SOLICITAR_ACLARACION,
                    cantidad=semantica.cantidad,
                    destinatario=semantica.destinatario,
                    almacen=alm_cand_obj,
                    motivo=de_cand.get("motivo"),
                    candidatos=candidatos,
                    datos_extra=de_cand,
                    mensaje_aclaracion=f"Encontré varias opciones para '{semantica.producto}': {cands_str}. ¿Cuál de ellos desea usar?",
                )
            else:
                # Resolución y validación del almacén de origen
                alm_obj = None
                if semantica.almacen:
                    if "dotaci" in semantica.almacen.lower():
                        semantica.almacen = None
                    else:
                        alm_obj = _buscar_almacen_por_nombre(semantica.almacen)
                        if not alm_obj:
                            msg = f"El almacén '{semantica.almacen}' no existe en el sistema."
                            return {
                                "tipo": "error",
                                "mensaje_pantalla": msg,
                                "mensaje_verbal": normalizar_para_tts(msg),
                                "datos": {"almacen_invalido": semantica.almacen},
                            }
                        if not usuario.tiene_acceso_almacen(alm_obj):
                            msg = f"No cuenta con autorización para operar en el almacén '{alm_obj.nombre}'."
                            return {
                                "tipo": "error",
                                "mensaje_pantalla": msg,
                                "mensaje_verbal": normalizar_para_tts(msg),
                                "datos": {"error_permisos": True, "almacen_denegado": alm_obj.nombre},
                            }

                # Datos extra y almacén destino para traslado
                datos_extra = dict(semantica.datos_extra) if semantica.datos_extra else {}
                destino_nom = getattr(semantica, "destino", None) or datos_extra.get("destino") or datos_extra.get("almacen_destino")
                if not destino_nom and tipo_intencion == IntencionTipo.TRASLADAR_PRODUCTO:
                    almacen_detectado = _detectar_almacen_destino_en_texto(texto)
                    destino_nom = almacen_detectado.nombre if almacen_detectado else None
                if destino_nom:
                    alm_dest = _buscar_almacen_por_nombre(destino_nom)
                    if alm_dest:
                        datos_extra["almacen_destino_id"] = alm_dest.id
                        datos_extra["almacen_destino_nombre"] = alm_dest.nombre

                # Motivo para dotación o salida
                motivo_op = None
                if tipo_op_sem in ("dotacion", "salida_dotacion") or (semantica.campo_modificar == "motivo" and semantica.nuevo_valor):
                    motivo_op = "Dotación"
                elif semantica.nuevo_valor and "dotacion" in str(semantica.nuevo_valor).lower():
                    motivo_op = "Dotación"
                elif re.search(r"\b(?:por|como|en\s+calidad\s+de)?\s*dotaci[oó]n\b", texto, re.IGNORECASE):
                    motivo_op = "Dotación"
                elif datos_extra.get("motivo"):
                    motivo_op = datos_extra.get("motivo")

                intencion = IntencionResultado(
                    tipo=tipo_intencion,
                    producto=prod_obj,
                    codigo_producto=prod_obj.codigo_interno if prod_obj else None,
                    cantidad=semantica.cantidad,
                    destinatario=semantica.destinatario,
                    motivo=motivo_op,
                    almacen=alm_obj,
                    candidatos=candidatos,
                    datos_extra=datos_extra,
                    requiere_confirmacion=True,
                )

    if intencion is None and semantica and semantica.intencion == "saludo":
        intencion = IntencionResultado(tipo=IntencionTipo.SALUDO)
    elif intencion is None and semantica and semantica.intencion == "agradecimiento":
        intencion = IntencionResultado(tipo=IntencionTipo.AGRADECIMIENTO)

    if intencion is None:
        _t0_interp = time.time()
        intencion = NormalizadorLenguajeHaku.interpretar(
            mensaje=texto,
            usuario=usuario,
            contexto_sesion=sesion,
            estado_operativo=estado_operativo,
        )
        _latencia_ms = max((time.time() - _t0_interp) * 1000.0, 0.01)
        HakuDiagnostico.registrar_intencion(intencion.tipo.value, latencia_ms=_latencia_ms)

    # Actualizar producto en foco conversacional en RAM
    if intencion.producto and estado_operativo:
        estado_operativo.asignar_producto(
            intencion.producto.id,
            intencion.producto.codigo_interno,
            intencion.producto.nombre,
        )
        estado_operativo.ultima_intencion = intencion.tipo.value
        estado_operativo.ultima_busqueda = texto
        HakuOperationalContext.guardar(estado_operativo)
    if intencion.producto and sesion is not None:
        sesion["haku_ultimo_producto_id"] = str(intencion.producto.id)
        sesion["haku_ultimo_acceso"] = time.time()

    # 0. Protocolo social: Saludos y Agradecimientos
    if intencion.tipo == IntencionTipo.SALUDO:
        _tl_sal = (texto or "").lower()
        if re.search(r"\bbuenos?\s+d[ií]as?\b", _tl_sal):
            _pref_sal = "Buenos días."
        elif re.search(r"\bbuenas?\s+tardes?\b", _tl_sal):
            _pref_sal = "Buenas tardes."
        elif re.search(r"\bbuenas?\s+noches?\b", _tl_sal):
            _pref_sal = "Buenas noches."
        else:
            _h_sal = timezone.localtime().hour
            _pref_sal = "Buenos días." if _h_sal < 12 else ("Buenas tardes." if _h_sal < 19 else "Buenas noches.")
        msg = f"{_pref_sal} Hacu Inventario está listo para atenderle. ¿Qué consulta u operación desea realizar?"
        return {
            "tipo": "saludo",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
        }

    if intencion.tipo == IntencionTipo.AGRADECIMIENTO:
        msg = "Con gusto. El sistema permanece disponible para nuevas consultas u operaciones."
        return {
            "tipo": "agradecimiento",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
        }

    if intencion.tipo == IntencionTipo.SOLICITUD_NO_COMPRENDIDA:
        msg = "Disculpe, no entendí la solicitud. ¿Podría repetirla?"
        return {
            "tipo": "no_comprendido",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"no_comprendido": True},
        }

    # 0-bis. Acciones de Confirmación y Cancelación desacopladas
    if intencion.tipo == IntencionTipo.ACCION_CONFIRMAR:
        if propuesta_activa:
            if time.time() > propuesta_activa.get("expira_en", 0):
                HakuDiagnostico.registrar_propuesta("expirada")
                if estado_operativo:
                    estado_operativo.propuesta_activa = None
                    HakuOperationalContext.guardar(estado_operativo)
                if sesion is not None:
                    sesion["haku_propuesta_activa"] = None
                return {
                    "tipo": "aclaracion",
                    "mensaje_pantalla": "La propuesta previa ha caducado por superar los 30 minutos. Por favor genera una nueva solicitud.",
                    "mensaje_verbal": "La propuesta ha caducado. Por favor solicita la operación de nuevo.",
                    "propuesta_activa": False,
                }
            resp_conf = {
                "tipo": "accion_confirmar",
                "propuesta_id": propuesta_activa["id"],
                "mensaje_pantalla": "Confirmación recibida. Procediendo con la transacción...",
                "mensaje_verbal": "Confirmación recibida. Ejecutando la operación.",
            }
            if semantica is not None and getattr(semantica, "datos_extra", None):
                obs_inline_conf = semantica.datos_extra.get("observacion")
                if obs_inline_conf:
                    resp_conf["observacion_usuario"] = str(obs_inline_conf).strip()[:500]
            return resp_conf
        return {
            "tipo": "accion_confirmar",
            "mensaje_pantalla": "No hay ninguna transacción pendiente de confirmación en este momento.",
            "mensaje_verbal": normalizar_para_tts("No hay ninguna transacción pendiente de confirmación en este momento."),
        }

    if intencion.tipo == IntencionTipo.ACCION_CANCELAR:
        if propuesta_activa:
            HakuDiagnostico.registrar_propuesta("cancelada")
            if estado_operativo:
                estado_operativo.propuesta_activa = None
                HakuOperationalContext.guardar(estado_operativo)
            if sesion is not None:
                sesion["haku_propuesta_activa"] = None
            return {
                "tipo": "accion_cancelar",
                "propuesta_id": propuesta_activa["id"],
                "mensaje_pantalla": "Operación cancelada. La propuesta ha sido descartada.",
                "mensaje_verbal": "Operación cancelada. No se realizaron cambios en el inventario.",
            }
        return {
            "tipo": "accion_cancelar",
            "mensaje_pantalla": "Entendido. No hay operaciones pendientes en ejecución.",
            "mensaje_verbal": normalizar_para_tts("Entendido. No hay operaciones pendientes en ejecución."),
        }

    # Descartar propuesta previa al conmutar a una nueva operación independiente
    if propuesta_activa and intencion.tipo in (
        IntencionTipo.REGISTRAR_ENTRADA,
        IntencionTipo.REGISTRAR_SALIDA,
        IntencionTipo.PRESTAMO,
        IntencionTipo.DEVOLUCION,
        IntencionTipo.TRASLADAR_PRODUCTO,
        IntencionTipo.CREAR_PRODUCTO,
        IntencionTipo.INHABILITAR_PRODUCTO,
        IntencionTipo.HABILITAR_PRODUCTO,
    ):
        if estado_operativo:
            estado_operativo.propuesta_activa = None
            estado_operativo.operacion_pendiente = None
            HakuOperationalContext.guardar(estado_operativo)
        if sesion and "haku_propuesta_activa" in sesion:
            del sesion["haku_propuesta_activa"]
        propuesta_activa = None

    # 1. Petición de aclaración por ambigüedad o múltiples códigos (Reglas 3 y 5)
    if intencion.tipo == IntencionTipo.SOLICITAR_ACLARACION:
        if intencion.candidatos and estado_operativo:
            int_orig = (intencion.datos_extra or {}).get("tipo_operacion_pendiente")
            if not int_orig:
                int_orig = IntencionTipo.CONSULTAR_STOCK.value
                if re.search(r"\b(d[oó]nde|ubicaci[oó]n|estante|caj[oó]n|repisa|vitrina|bodega)\b", texto_lower):
                    int_orig = IntencionTipo.CONSULTAR_UBICACION.value
                elif re.search(r"\b(movimientos?|qu[eé]\s+pas[oó]|despachos?|historial)\b", texto_lower):
                    int_orig = IntencionTipo.CONSULTAR_PRODUCTO_MOVIMIENTOS.value
                elif re.search(r"\b(pr[eé]sta(?:r|le|lo|me)?|pr[eé]stamo)\b", texto_lower):
                    int_orig = IntencionTipo.PRESTAMO.value
                elif re.search(r"\b(s[aá]ca(?:r|me)?|retira(?:r|me)?|salida|despacha(?:r)?|dotaci[oó]n)\b", texto_lower):
                    int_orig = IntencionTipo.REGISTRAR_SALIDA.value
                elif re.search(r"\b(ingresa(?:r)?|entrada|recib[ií]|agrega(?:r)?)\b", texto_lower):
                    int_orig = IntencionTipo.REGISTRAR_ENTRADA.value
                elif re.search(r"\b(devuelve|devolver|devoluci[oó]n|regresa)\b", texto_lower):
                    int_orig = IntencionTipo.DEVOLUCION.value
                elif re.search(r"\b(traslada(?:r)?|traslado)\b", texto_lower):
                    int_orig = IntencionTipo.TRASLADAR_PRODUCTO.value

            datos_extra = dict(intencion.datos_extra) if intencion.datos_extra else {}
            if intencion.almacen:
                datos_extra["almacen"] = intencion.almacen
            if intencion.cantidad:
                datos_extra["cantidad"] = intencion.cantidad
            if intencion.destinatario:
                datos_extra["destinatario"] = intencion.destinatario
            if intencion.motivo:
                datos_extra["motivo"] = intencion.motivo

            estado_operativo.asignar_candidatos(
                intencion.candidatos,
                intencion_original=int_orig,
                datos_extra=datos_extra,
            )
            estado_operativo.ultima_intencion = intencion.tipo.value
            estado_operativo.ultima_busqueda = texto
            HakuOperationalContext.guardar(estado_operativo)

        msg_aclaracion = intencion.mensaje_aclaracion or "¿Podría especificar a qué producto o acción se refiere?"
        return {
            "tipo": "aclaracion",
            "mensaje_pantalla": msg_aclaracion,
            "mensaje_verbal": normalizar_para_tts(msg_aclaracion),
            "datos": {"requiere_aclaracion": True},
        }

    # 2. Métricas y consultas generales de movimientos por intervalo temporal y multialmacén
    if intencion.tipo == IntencionTipo.CONSULTAR_METRICAS_MOVIMIENTOS:
        if not usuario_tiene_permiso_operativo(usuario, "analitica.ver_reportes"):
            msg = "No cuentas con el permiso requerido (analitica.ver_reportes) para consultar métricas de inventario."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar métricas de movimientos."),
                "datos": {"error_permiso": True, "permiso_requerido": "analitica.ver_reportes"},
            }

        from .servicios_temporales import resolver_intervalo_temporal, obtener_ahora_local
        intervalo = resolver_intervalo_temporal(texto)
        if not intervalo:
            ahora = obtener_ahora_local()
            hoy_date = ahora.date()
            import datetime
            inicio = datetime.datetime.combine(hoy_date, datetime.time.min, tzinfo=ahora.tzinfo)
            fin = ahora
            desc_periodo = f"hoy ({hoy_date.strftime('%d/%m/%Y')})"
        else:
            inicio = intervalo.inicio
            fin = intervalo.fin
            desc_periodo = intervalo.descripcion_legible

        qs_mov = Movimiento.objects.filter(
            estado=Movimiento.Estado.CONFIRMADO,
            fecha_hora__gte=inicio,
            fecha_hora__lt=fin,
        ).exclude(tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL).select_related("producto", "almacen_origen", "almacen_destino", "usuario")

        # Filtrado de almacén: si se especificó almacén en la intención, validar acceso
        if intencion.almacen:
            almacen_filtrado, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
            if error_resp:
                return error_resp
            qs_mov = qs_mov.filter(Q(almacen_origen=almacen_filtrado) | Q(almacen_destino=almacen_filtrado)).distinct()
            desc_periodo += f" en {almacen_filtrado.nombre}"
        elif usuario.es_usuario_normal:
            if usuario.acceso_todos_almacenes:
                pass
            else:
                alms = usuario.obtener_almacenes_accesibles()
                if not alms.exists():
                    msg = "Su cuenta de usuario no tiene un almacén asignado activo para consultar movimientos."
                    return {
                        "tipo": "consulta",
                        "mensaje_pantalla": msg,
                        "mensaje_verbal": normalizar_para_tts(msg),
                        "datos": {"total": 0, "sin_almacen_asignado": True},
                    }
                qs_mov = qs_mov.filter(Q(almacen_origen__in=alms) | Q(almacen_destino__in=alms)).distinct()

        total_mov = qs_mov.count()
        entradas = qs_mov.filter(tipo=Movimiento.Tipo.ENTRADA).count()
        salidas = qs_mov.filter(tipo=Movimiento.Tipo.SALIDA).count()
        prestamos = qs_mov.filter(tipo=Movimiento.Tipo.PRESTAMO).count()
        devoluciones = qs_mov.filter(tipo=Movimiento.Tipo.DEVOLUCION).count()
        traslados = qs_mov.filter(tipo__in=[Movimiento.Tipo.TRASLADO_ENTRE, Movimiento.Tipo.TRASLADO_INTERNO]).count()
        iniciales = qs_mov.filter(tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL).count()
        otros = max(0, total_mov - (entradas + salidas + prestamos + devoluciones + traslados + iniciales))

        if estado_operativo:
            estado_operativo.registrar_accion(
                "CONSULTAR_METRICAS_MOVIMIENTOS",
                cantidad=total_mov,
            )
            HakuOperationalContext.guardar(estado_operativo)

        es_pregunta_entradas = bool(re.search(r"\bentradas?\b", texto_lower) and not re.search(r"\bsalidas?\b", texto_lower))
        es_pregunta_salidas = bool(re.search(r"\bsalidas?\b", texto_lower) and not re.search(r"\bentradas?\b", texto_lower))
        es_pregunta_traslados = bool(re.search(r"\btraslados?\b", texto_lower))
        es_pregunta_prestamos = bool(re.search(r"\bpr[eé]stamos?\b", texto_lower) and not re.search(r"\b(entradas?|salidas?|devoluciones?)\b", texto_lower))
        es_pregunta_devoluciones = bool(re.search(r"\bdevoluciones?\b", texto_lower) and not re.search(r"\b(entradas?|salidas?|pr[eé]stamos?)\b", texto_lower))
        ultimos_movs = list(qs_mov.select_related("producto", "almacen_origen", "almacen_destino")[:10]) if total_mov > 0 else []

        entradas_txt = pluralizar(entradas, "entrada", "entradas")
        salidas_txt = pluralizar(salidas, "salida", "salidas")
        prestamos_txt = pluralizar(prestamos, "préstamo", "préstamos")
        devoluciones_txt = pluralizar(devoluciones, "devolución", "devoluciones")
        traslados_txt = pluralizar(traslados, "traslado", "traslados")
        total_mov_txt = pluralizar(total_mov, "movimiento", "movimientos")

        entradas_md = pluralizar_md(entradas, "entrada", "entradas")
        salidas_md = pluralizar_md(salidas, "salida", "salidas")
        prestamos_md = pluralizar_md(prestamos, "préstamo", "préstamos")
        devoluciones_md = pluralizar_md(devoluciones, "devolución", "devoluciones")
        traslados_md = pluralizar_md(traslados, "traslado", "traslados")
        total_mov_md = pluralizar_md(total_mov, "movimiento", "movimientos")

        if total_mov == 0:
            if desc_periodo.strip().lower().startswith("hoy"):
                pantalla = (
                    f"Hoy no se registraron movimientos de inventario.\n\n"
                    f"- **0** entradas\n- **0** salidas\n- **0** préstamos\n- **0** devoluciones\n- **0** traslados\n\n"
                    f"¿Desea revisar algún producto específico?"
                )
                verbal = "Hoy no se registraron movimientos de inventario. ¿Desea revisar algún producto específico?"
            else:
                pantalla = (
                    f"Durante el periodo consultado (**{desc_periodo}**) no se registraron movimientos de inventario.\n\n"
                    f"- **0** entradas\n- **0** salidas\n- **0** préstamos\n- **0** devoluciones\n- **0** traslados\n\n"
                    f"¿Desea revisar algún producto específico?"
                )
                verbal = f"Durante el periodo consultado, {desc_periodo}, no se registraron movimientos de inventario. ¿Desea revisar algún producto específico?"
        elif es_pregunta_entradas:
            pantalla = f"Se registraron {entradas_md} durante **{desc_periodo}** (de un total de {total_mov_md}):\n- {entradas_md}\n- {salidas_md}\n- {prestamos_md}\n- {devoluciones_md}\n- {traslados_md}\n\n¿Desea revisar algún producto específico?"
            verbal = f"Se registraron {entradas_txt} durante {desc_periodo}, de un total de {total_mov_txt}. ¿Desea revisar algún producto específico?"
        elif es_pregunta_salidas:
            pantalla = f"Se registraron {salidas_md} durante **{desc_periodo}** (de un total de {total_mov_md}):\n- {entradas_md}\n- {salidas_md}\n- {prestamos_md}\n- {devoluciones_md}\n- {traslados_md}\n\n¿Desea revisar algún producto específico?"
            verbal = f"Se registraron {salidas_txt} durante {desc_periodo}, de un total de {total_mov_txt}. ¿Desea revisar algún producto específico?"
        elif es_pregunta_traslados:
            pantalla = f"Se registraron {traslados_md} durante **{desc_periodo}** (de un total de {total_mov_md}):\n- {entradas_md}\n- {salidas_md}\n- {prestamos_md}\n- {devoluciones_md}\n- {traslados_md}\n\n¿Desea revisar algún producto específico?"
            verbal = f"Se registraron {traslados_txt} durante {desc_periodo}, de un total de {total_mov_txt}. ¿Desea revisar algún producto específico?"
        elif es_pregunta_prestamos:
            movs_p = qs_mov.filter(tipo=Movimiento.Tipo.PRESTAMO)
            destinatarios = [m.destinatario for m in movs_p if m.destinatario]
            if destinatarios:
                dest_unicos = list(dict.fromkeys(destinatarios))
                pantalla = f"Se registraron {prestamos_md} durante **{desc_periodo}**.\n\nDestinatarios:\n" + "\n".join(f"- **{d}**" for d in dest_unicos)
                verbal = f"Se registraron {prestamos_txt} durante {desc_periodo}. Entre los beneficiarios: {', '.join(dest_unicos[:3])}."
            else:
                pantalla = f"Se registraron {prestamos_md} durante **{desc_periodo}** (de un total de {total_mov_md})."
                verbal = f"Se registraron {prestamos_txt} durante {desc_periodo}."
        elif es_pregunta_devoluciones:
            pantalla = f"Se registraron {devoluciones_md} durante **{desc_periodo}** (de un total de {total_mov_md})."
            verbal = f"Se registraron {devoluciones_txt} durante {desc_periodo}."
        else:
            # Si se solicita listado o detalle («muéstrame los movimientos de ayer»)
            detalle_movs = []
            for m in ultimos_movs:
                fecha_str = m.fecha_hora.astimezone(obtener_ahora_local().tzinfo).strftime("%d/%m %H:%M")
                alm_txt = m.almacen_origen.nombre if m.almacen_origen else ""
                if m.almacen_destino:
                    alm_txt += f" ➔ {m.almacen_destino.nombre}"
                detalle_movs.append(f"- `{fecha_str}` | **{m.get_tipo_display()}**: {pluralizar(m.cantidad, 'ud', 'uds')} de {m.producto.nombre} ({alm_txt})")

            bloque_detalle = "\n".join(detalle_movs)
            extra_paginacion = f"\n\n*(Mostrando primeros 10 de {total_mov} movimientos registrados)*" if total_mov > 10 else ""

            partes_desglose_md = [entradas_md, salidas_md, prestamos_md, devoluciones_md, traslados_md]
            partes_desglose_txt = [entradas_txt, salidas_txt, prestamos_txt, devoluciones_txt, traslados_txt]
            if iniciales > 0:
                partes_desglose_md.append(pluralizar_md(iniciales, "movimiento inicial", "movimientos iniciales"))
                partes_desglose_txt.append(pluralizar(iniciales, "movimiento inicial", "movimientos iniciales"))
            if otros > 0:
                partes_desglose_md.append(pluralizar_md(otros, "otro ajuste", "otros ajustes"))
                partes_desglose_txt.append(pluralizar(otros, "otro ajuste", "otros ajustes"))

            resumen_verbal_desglose = (
                ", ".join(partes_desglose_txt[:-1]) + f" y {partes_desglose_txt[-1]}"
                if len(partes_desglose_txt) > 1
                else partes_desglose_txt[0]
            )

            pantalla = (
                f"### Movimientos de inventario — {desc_periodo}\n"
                f"Durante el periodo ({desc_periodo}) se registraron movimientos de inventario correspondientes a las operaciones realizadas.\n"
                f"Se registraron {total_mov_md} en total:\n"
                f"- {' | '.join(partes_desglose_md)}\n\n"
                f"**Detalle cronológico:**\n{bloque_detalle}{extra_paginacion}\n\n"
                f"¿Desea revisar algún producto específico?"
            )
            verbal = (
                f"Durante {desc_periodo} se registraron {total_mov_txt} en total: "
                f"{resumen_verbal_desglose}. "
                f"¿Desea revisar algún producto específico?"
            )

        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
            "datos": {
                "periodo": desc_periodo,
                "inicio": inicio.isoformat(),
                "fin": fin.isoformat(),
                "total": total_mov,
                "entradas": entradas,
                "salidas": salidas,
                "prestamos": prestamos,
                "devoluciones": devoluciones,
                "traslados": traslados,
                "movimientos": [
                    {
                        "id": str(m.id),
                        "tipo": m.tipo,
                        "producto": m.producto.nombre,
                        "cantidad": m.cantidad,
                        "almacen_origen": m.almacen_origen.nombre if m.almacen_origen else None,
                        "almacen_destino": m.almacen_destino.nombre if m.almacen_destino else None,
                        "motivo": m.motivo,
                    }
                    for m in ultimos_movs
                ],
            },
        }

    # 3. Resumen de inventario / almacén
    if intencion.tipo == IntencionTipo.CONSULTAR_RESUMEN_INVENTARIO:
        if not usuario_tiene_permiso_operativo(usuario, "analitica.ver_reportes"):
            msg = "No cuentas con el permiso requerido (analitica.ver_reportes) para consultar el resumen analítico de inventario."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar el resumen de inventario."),
                "datos": {"error_permiso": True, "permiso_requerido": "analitica.ver_reportes"},
            }

        hoy = timezone.now().date()
        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        qs_prod = Producto.objects.filter(activo=True, archivado=False)
        if almacen:
            qs_prod = qs_prod.filter(categoria__almacen_principal=almacen)
        total_productos = qs_prod.count()

        qs_exist = Existencia.objects.filter(producto__activo=True, producto__archivado=False)
        if almacen:
            qs_exist = qs_exist.filter(almacen=almacen)
        elif usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
            alms_usuario = usuario.obtener_almacenes_accesibles()
            qs_exist = qs_exist.filter(almacen__in=alms_usuario)

        sin_stock_count = sum(1 for e in qs_exist if e.sin_stock)
        bajo_stock_count = sum(1 for e in qs_exist if e.stock_bajo)
        total_alertas = sin_stock_count + bajo_stock_count

        qs_mov = Movimiento.objects.filter(estado=Movimiento.Estado.CONFIRMADO, fecha_hora__date=hoy).exclude(tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL)
        if almacen:
            qs_mov = qs_mov.filter(almacen_origen=almacen)
        elif usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
            alms_usuario = usuario.obtener_almacenes_accesibles()
            qs_mov = qs_mov.filter(Q(almacen_origen__in=alms_usuario) | Q(almacen_destino__in=alms_usuario)).distinct()
        movimientos_hoy = qs_mov.count()

        qs_prest = Prestamo.objects.filter(
            estado__in=[Prestamo.Estado.ABIERTO, Prestamo.Estado.DEVUELTO_PARCIAL]
        )
        if almacen:
            qs_prest = qs_prest.filter(almacen=almacen)
        elif usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
            alms_usuario = usuario.obtener_almacenes_accesibles()
            qs_prest = qs_prest.filter(almacen__in=alms_usuario)
        prestamos_activos = qs_prest.count()

        if estado_operativo:
            estado_operativo.registrar_accion(
                "CONSULTAR_RESUMEN_INVENTARIO",
                cantidad=total_productos,
            )
            HakuOperationalContext.guardar(estado_operativo)

        nombre_almacen = almacen.nombre if almacen else "el inventario general"
        pantalla = (
            f"### Resumen de inventario ({nombre_almacen})\n"
            f"- **Cantidad de productos registrados:** {total_productos}\n"
            f"- **Productos con bajo stock:** {bajo_stock_count}\n"
            f"- **Productos sin stock (agotados):** {sin_stock_count}\n"
            f"- **Movimientos del día:** {movimientos_hoy}\n"
            f"- **Alertas activas:** {total_alertas}\n"
            f"- **Préstamos activos:** {prestamos_activos}"
        )
        verbal = (
            f"En {nombre_almacen} hay {total_productos} productos registrados. "
            f"Hay {bajo_stock_count} con bajo stock y {sin_stock_count} agotados. "
            f"Hoy van {movimientos_hoy} movimientos del día y {prestamos_activos} préstamos activos."
        )
        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
            "datos": {
                "total_productos": total_productos,
                "bajo_stock": bajo_stock_count,
                "sin_stock": sin_stock_count,
                "movimientos_hoy": movimientos_hoy,
                "alertas": total_alertas,
                "prestamos_activos": prestamos_activos,
            },
        }

    # 3-bis A. Listar almacenes / bodegas
    if intencion.tipo == IntencionTipo.LISTAR_ALMACENES:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar almacenes del sistema."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar almacenes."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }

        qs_almacenes = Almacen.objects.filter(activo=True, eliminado=False).order_by("nombre")
        total_almacenes = qs_almacenes.count()

        if total_almacenes == 0:
            msg = "Actualmente no hay almacenes activos registrados en el sistema."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"almacenes": []},
            }

        alms_autorizados_ids = set(usuario.obtener_almacenes_accesibles().values_list("id", flat=True)) if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes else set()

        lineas_pantalla = ["### Almacenes activos en el sistema:\n"]
        lineas_voz = [f"Hay {total_almacenes} almacenes registrados:"]
        lista_datos = []

        for alm in qs_almacenes:
            es_propio = bool(alm.id in alms_autorizados_ids)
            marca_asignado = " *(Su almacén asignado)*" if (usuario.almacen_asignado and alm.id == usuario.almacen_asignado.id) else (" *(Autorizado)*" if es_propio else "")
            desc = f" — {alm.descripcion}" if alm.descripcion else ""
            lineas_pantalla.append(f"- **{alm.nombre}**{marca_asignado}{desc}")
            lineas_voz.append(f"{alm.nombre}.")
            lista_datos.append({"id": alm.id, "nombre": alm.nombre, "descripcion": alm.descripcion, "es_asignado": es_propio})

        if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes and not alms_autorizados_ids:
            lineas_pantalla.append("\n*Aviso: Su cuenta de usuario no tiene ningún almacén asignado actualmente.*")

        if estado_operativo:
            estado_operativo.registrar_accion(
                "LISTAR_ALMACENES",
                cantidad=total_almacenes,
            )
            HakuOperationalContext.guardar(estado_operativo)

        pantalla = "\n".join(lineas_pantalla)
        verbal = " ".join(lineas_voz)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
            "datos": {"almacenes": lista_datos, "total": total_almacenes},
        }

    # 3-bis A-2. Listar ubicaciones registradas
    if intencion.tipo == IntencionTipo.LISTAR_UBICACIONES:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar las ubicaciones de inventario."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar ubicaciones."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        qs_ubicaciones = Ubicacion.objects.filter(activo=True, eliminado=False).select_related("almacen").order_by("almacen__nombre", "nombre")
        if almacen:
            qs_ubicaciones = qs_ubicaciones.filter(almacen=almacen)

        total_ubicaciones = qs_ubicaciones.count()
        if total_ubicaciones == 0:
            nombre_alm = f" en {almacen.nombre}" if almacen else ""
            msg = f"Actualmente no hay ubicaciones registradas{nombre_alm} en el sistema."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"ubicaciones": [], "total": 0},
            }

        alms_autorizados_ids = set(usuario.obtener_almacenes_accesibles().values_list("id", flat=True)) if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes else None
        if alms_autorizados_ids is not None:
            qs_ubicaciones = qs_ubicaciones.filter(almacen_id__in=alms_autorizados_ids)
            total_ubicaciones = qs_ubicaciones.count()

        encabezado = f"### Ubicaciones registradas en {almacen.nombre}:\n" if almacen else "### Ubicaciones registradas en el sistema:\n"
        lineas_pantalla = [encabezado]
        lineas_voz = [f"Hay {total_ubicaciones} ubicaciones registradas:"]
        lista_datos = []

        for ub in qs_ubicaciones:
            desc = f" — {ub.descripcion}" if ub.descripcion else ""
            lineas_pantalla.append(f"- **{ub.nombre}** ({ub.almacen.nombre}){desc}")
            lineas_voz.append(f"{ub.nombre} en {ub.almacen.nombre}.")
            lista_datos.append({"id": ub.id, "nombre": ub.nombre, "almacen_id": ub.almacen_id, "almacen_nombre": ub.almacen.nombre})

        if estado_operativo:
            estado_operativo.registrar_accion(
                "LISTAR_UBICACIONES",
                cantidad=total_ubicaciones,
            )
            HakuOperationalContext.guardar(estado_operativo)

        pantalla = "\n".join(lineas_pantalla)
        verbal = " ".join(lineas_voz)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
            "datos": {"ubicaciones": lista_datos, "total": total_ubicaciones},
        }

    # 3-bis B. Listar catálogo general de productos
    if intencion.tipo == IntencionTipo.LISTAR_CATALOGO:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar el catálogo de productos."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar el catálogo."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        qs_prod = Producto.objects.filter(activo=True, archivado=False).select_related("categoria")
        if almacen:
            qs_prod = qs_prod.filter(categoria__almacen_principal=almacen)

        filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
        if filtro_nombre:
            filtro_q = construir_filtro_familia_o_tipo(filtro_nombre, prefix_relacion="")
            qs_prod = qs_prod.filter(filtro_q)

        qs_prod = qs_prod.order_by("codigo_interno")

        total_productos = qs_prod.count()
        if total_productos == 0:
            if filtro_nombre:
                msg = f"No hay productos de la categoría o tipo '{filtro_nombre}' registrados en el catálogo."
            else:
                msg = "No hay productos registrados en el catálogo en este momento."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"productos": [], "total": 0},
            }

        productos_muestra = list(qs_prod[:10])
        prods_txt = pluralizar(total_productos, "producto", "productos")
        lineas_pantalla = [f"### Catálogo de productos ({total_productos} registrados):\n"]

        lista_datos = []
        nombres_voz = []
        for p in productos_muestra:
            lineas_pantalla.append(f"- **{p.nombre}** (`{p.codigo_interno}`) — Categoría: {p.categoria.nombre}")
            nombres_voz.append(p.nombre)
            lista_datos.append({"id": str(p.id), "codigo": p.codigo_interno, "nombre": p.nombre, "categoria": p.categoria.nombre})

        if total_productos > 10:
            lineas_pantalla.append(f"\n*Mostrando 10 de {total_productos} productos registrados.*")

        if total_productos > 3:
            lineas_voz = [f"El catálogo cuenta con {total_productos} {prods_txt} registrados. Te muestro el detalle en pantalla."]
        else:
            lineas_voz = [f"El catálogo cuenta con {total_productos} {prods_txt} registrados: {', '.join(nombres_voz)}."]

        if estado_operativo:
            estado_operativo.asignar_coleccion(lista_datos)
            estado_operativo.registrar_accion(
                "LISTAR_CATALOGO",
                cantidad=total_productos,
            )
            estado_operativo.producto_id = None
            estado_operativo.producto_codigo = None
            estado_operativo.producto_nombre = None
            HakuOperationalContext.guardar(estado_operativo)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": "\n".join(lineas_pantalla),
            "mensaje_verbal": normalizar_para_tts(" ".join(lineas_voz)),
            "datos": {"productos": lista_datos, "total": total_productos},
        }

    # 3-bis C. Productos con existencias disponibles en stock
    if intencion.tipo == IntencionTipo.PRODUCTOS_CON_STOCK:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar productos con existencias."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar existencias."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        almacenes_accesibles = usuario.obtener_almacenes_accesibles()

        # Si se seleccionaron candidatos plurales específicos (ej. "de los tres")
        cands_plurales = intencion.datos_extra.get("candidatos_plurales") if intencion.datos_extra else None
        if cands_plurales:
            ids_cands = [c["id"] for c in cands_plurales if "id" in c]
            qs_exist = Existencia.objects.filter(
                producto_id__in=ids_cands,
                producto__activo=True,
                producto__archivado=False,
                stock_disponible__gt=0,
                almacen__in=almacenes_accesibles,
            ).select_related("producto", "almacen")
        else:
            qs_exist = Existencia.objects.filter(
                producto__activo=True,
                producto__archivado=False,
                stock_disponible__gt=0,
                almacen__in=almacenes_accesibles,
            ).select_related("producto", "almacen").order_by("producto__nombre", "-stock_disponible")

            if almacen:
                qs_exist = qs_exist.filter(almacen=almacen)

            filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
            if filtro_nombre:
                filtro_q = construir_filtro_familia_o_tipo(filtro_nombre, prefix_relacion="producto__")
                qs_exist = qs_exist.filter(filtro_q)

        # Agrupar existencias por producto único para evitar duplicación entre almacenes
        productos_map = {}
        for e in qs_exist:
            pid = e.producto_id
            if pid not in productos_map:
                productos_map[pid] = {
                    "producto": e.producto,
                    "total_disponible": 0,
                    "desglose": [],
                }
            productos_map[pid]["total_disponible"] += e.stock_disponible
            productos_map[pid]["desglose"].append((e.almacen.nombre, e.stock_disponible))

        total_productos = len(productos_map)
        if total_productos == 0:
            filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
            if filtro_nombre:
                msg = f"No hay productos de la categoría o tipo '{filtro_nombre}' con existencias disponibles en este momento."
            else:
                msg = "No hay productos con existencias disponibles en este momento."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"productos": [], "existencias": [], "total": 0},
            }

        productos_muestra = list(productos_map.values())[:10]

        if almacen:
            lineas_pantalla = [f"### Productos con existencias disponibles en {almacen.nombre} ({total_productos} en total):\n"]
        else:
            lineas_pantalla = [f"### Productos con existencias disponibles ({total_productos} en total):\n"]

        lista_datos = []
        for item in productos_muestra:
            p = item["producto"]
            tot = item["total_disponible"]
            desglose = item["desglose"]
            unid_disp = "unidad disponible" if tot == 1 else "unidades disponibles"

            if almacen:
                lineas_pantalla.append(
                    f"- **{p.nombre}** (`{p.codigo_interno}`): **{tot}** {unid_disp}"
                )
            elif len(desglose) > 1:
                desglose_str = ", ".join(f"{cant} en {alm}" for alm, cant in desglose)
                lineas_pantalla.append(
                    f"- **{p.nombre}** (`{p.codigo_interno}`): **{tot}** {unid_disp} ({desglose_str})"
                )
            else:
                alm_nom, cant = desglose[0]
                lineas_pantalla.append(
                    f"- **{p.nombre}** (`{p.codigo_interno}`): **{tot}** {unid_disp} en {alm_nom}"
                )

            lista_datos.append({
                "id": p.id,
                "producto_id": str(p.id),
                "codigo": p.codigo_interno,
                "nombre": p.nombre,
                "disponible": tot,
                "almacenes": [a for a, _ in desglose],
                "desglose": [{"almacen": a, "cantidad": c} for a, c in desglose],
            })

        if total_productos > 10:
            lineas_pantalla.append(f"\n*Mostrando los 10 productos con mayor disponibilidad.*")

        # Respuesta verbal concisa para listados (Sección 7)
        filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
        if filtro_nombre == "cable":
            termino_voz = "cable" if total_productos == 1 else "cables"
        elif filtro_nombre:
            termino_voz = f"producto de tipo {filtro_nombre}" if total_productos == 1 else f"productos de tipo {filtro_nombre}"
        else:
            termino_voz = "producto" if total_productos == 1 else "productos"

        if almacen:
            lineas_voz = [f"En {almacen.nombre} encontré {total_productos} {termino_voz} con existencias disponibles. Te muestro las cantidades en pantalla."]
        else:
            lineas_voz = [f"Encontré {total_productos} {termino_voz} con existencias disponibles. Te muestro las cantidades y los almacenes en pantalla."]

        if estado_operativo:
            estado_operativo.asignar_coleccion(lista_datos)
            estado_operativo.registrar_accion(
                "PRODUCTOS_CON_STOCK",
                cantidad=total_productos,
            )
            estado_operativo.producto_id = None
            estado_operativo.producto_codigo = None
            estado_operativo.producto_nombre = None
            HakuOperationalContext.guardar(estado_operativo)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": "\n".join(lineas_pantalla),
            "mensaje_verbal": normalizar_para_tts(" ".join(lineas_voz)),
            "datos": {"productos": lista_datos, "existencias": lista_datos, "total": total_productos},
        }

    # 3-bis D. Productos sin stock / agotados (stock == 0)
    if intencion.tipo == IntencionTipo.PRODUCTOS_SIN_STOCK:
        if not usuario_tiene_permiso_operativo(usuario, "analitica.ver_alertas"):
            msg = "No cuentas con el permiso requerido (analitica.ver_alertas) para consultar productos sin stock."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar alertas de productos sin stock."),
                "datos": {"error_permiso": True, "permiso_requerido": "analitica.ver_alertas"},
            }

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        almacenes_accesibles = usuario.obtener_almacenes_accesibles()
        qs_exist = Existencia.objects.filter(
            producto__activo=True,
            producto__archivado=False,
            stock_disponible=0,
            almacen__in=almacenes_accesibles,
        ).select_related("producto", "almacen").order_by("producto__nombre")

        if almacen:
            qs_exist = qs_exist.filter(almacen=almacen)

        filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
        if filtro_nombre:
            filtro_q = construir_filtro_familia_o_tipo(filtro_nombre, prefix_relacion="producto__")
            qs_exist = qs_exist.filter(filtro_q)

        filtrar_coleccion = bool(intencion.datos_extra and intencion.datos_extra.get("filtrar_coleccion"))
        if filtrar_coleccion:
            coleccion = estado_operativo.coleccion_productos if estado_operativo else []
            if not coleccion:
                msg = "No tengo una lista previa de productos en la conversación activa para verificar cuáles se acabaron."
                return {
                    "tipo": "aclaracion",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts(msg),
                    "datos": {"sin_stock": [], "total": 0},
                }
            ids = [c.get("id") or c.get("producto_id") for c in coleccion if (c.get("id") or c.get("producto_id"))]
            qs_exist = qs_exist.filter(producto_id__in=ids)

        total_sin_stock = qs_exist.count()
        if total_sin_stock == 0:
            if filtro_nombre:
                msg = f"No hay productos de la categoría o tipo '{filtro_nombre}' agotados en este momento."
            elif filtrar_coleccion:
                msg = "De las referencias que acabamos de revisar, ninguna se ha agotado. Todas tienen stock disponible."
            else:
                msg = "No hay productos agotados en este momento. Todas las referencias cuentan con existencias disponibles."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"sin_stock": [], "total": 0},
            }

        items_muestra = list(qs_exist[:10])
        # Mensaje verbal conciso para síntesis de voz (ej. "Hay nueve productos agotados.")
        if total_sin_stock == 0:
            lineas_voz = ["No hay productos agotados en el inventario."]
        elif total_sin_stock == 1:
            if filtrar_coleccion:
                lineas_voz = ["De las referencias revisadas, hay un producto agotado."]
            elif almacen:
                lineas_voz = [f"En {almacen.nombre} hay un producto agotado."]
            else:
                lineas_voz = ["Hay un producto agotado."]
        else:
            palabra_cant = _numero_a_letras(total_sin_stock)
            if filtrar_coleccion:
                lineas_voz = [f"De las referencias revisadas, hay {palabra_cant} productos agotados."]
            elif almacen:
                lineas_voz = [f"En {almacen.nombre} hay {palabra_cant} productos agotados."]
            else:
                lineas_voz = [f"Hay {palabra_cant} productos agotados."]

        if filtrar_coleccion:
            lineas_pantalla = [f"### De las referencias revisadas, las siguientes están agotadas ({total_sin_stock} en total):\n"]
        elif almacen:
            lineas_pantalla = [f"### Productos agotados en {almacen.nombre} ({total_sin_stock} en total):\n"]
        else:
            lineas_pantalla = [f"### Productos agotados sin existencias ({total_sin_stock} en total):\n"]
        lista_datos = []

        for e in items_muestra:
            lineas_pantalla.append(f"- **{e.producto.nombre}** (`{e.producto.codigo_interno}`) en {e.almacen.nombre}: **0 unidades disponibles**")
            lista_datos.append({
                "id": e.producto.id,
                "producto_id": str(e.producto.id),
                "codigo": e.producto.codigo_interno,
                "nombre": e.producto.nombre,
                "disponible": 0,
                "almacen": e.almacen.nombre,
            })

        if total_sin_stock > 10:
            lineas_pantalla.append(f"\n*Mostrando 10 de {total_sin_stock} productos agotados.*")

        if estado_operativo:
            estado_operativo.asignar_coleccion(lista_datos, total_original=total_sin_stock)
            estado_operativo.registrar_accion(
                "PRODUCTOS_SIN_STOCK",
                cantidad=total_sin_stock,
            )
            estado_operativo.producto_id = None
            estado_operativo.producto_codigo = None
            estado_operativo.producto_nombre = None
            HakuOperationalContext.guardar(estado_operativo)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": "\n".join(lineas_pantalla),
            "mensaje_verbal": normalizar_para_tts(" ".join(lineas_voz)),
            "datos": {"sin_stock": lista_datos, "total": total_sin_stock},
        }

    # 3-bis E. Productos con bajo stock (por debajo del mínimo)
    if intencion.tipo == IntencionTipo.PRODUCTOS_BAJO_STOCK:
        if not usuario_tiene_permiso_operativo(usuario, "analitica.ver_alertas"):
            msg = "No cuentas con el permiso requerido (analitica.ver_alertas) para consultar productos con bajo stock."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar alertas de bajo stock."),
                "datos": {"error_permiso": True, "permiso_requerido": "analitica.ver_alertas"},
            }

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        almacenes_accesibles = usuario.obtener_almacenes_accesibles()
        qs_exist = Existencia.objects.filter(
            producto__activo=True,
            producto__archivado=False,
            stock_disponible__gt=0,
            almacen__in=almacenes_accesibles,
        ).select_related("producto", "almacen")

        if almacen:
            qs_exist = qs_exist.filter(almacen=almacen)

        filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
        if filtro_nombre:
            filtro_q = construir_filtro_familia_o_tipo(filtro_nombre, prefix_relacion="producto__")
            qs_exist = qs_exist.filter(filtro_q)

        bajo_stock_items = [e for e in qs_exist if e.stock_bajo]
        total_bajo_stock = len(bajo_stock_items)

        if total_bajo_stock == 0:
            if filtro_nombre:
                msg = f"No hay productos de la categoría o tipo '{filtro_nombre}' por debajo del stock mínimo configurado."
            else:
                msg = "No hay productos con existencias por debajo del stock mínimo configurado."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"bajo_stock": [], "total": 0},
            }

        items_muestra = bajo_stock_items[:10]
        lineas_pantalla = [f"### Productos por debajo del stock mínimo ({total_bajo_stock} en total):\n"]
        lineas_voz = [f"Hay {total_bajo_stock} productos por debajo del stock mínimo:"]
        lista_datos = []
        nombres_voz = []

        for e in items_muestra:
            lineas_pantalla.append(
                f"- **{e.producto.nombre}** (`{e.producto.codigo_interno}`) en {e.almacen.nombre}: "
                f"{pluralizar_md(e.stock_disponible, 'unidad', 'unidades')} (mínimo: {e.producto.stock_minimo})"
            )
            nombres_voz.append(f"{e.producto.nombre} con {e.stock_disponible} de {e.producto.stock_minimo}")
            lista_datos.append({
                "id": e.producto.id,
                "producto_id": str(e.producto.id),
                "codigo": e.producto.codigo_interno,
                "nombre": e.producto.nombre,
                "disponible": e.stock_disponible,
                "minimo": e.producto.stock_minimo,
                "almacen": e.almacen.nombre,
            })

        if total_bajo_stock > 10:
            lineas_pantalla.append(f"\n*Mostrando 10 de {total_bajo_stock} productos con bajo stock.*")
            lineas_voz.append(f"Entre ellos: {', '.join(nombres_voz[:4])}.")
        else:
            lineas_voz.append(f"{', '.join(nombres_voz)}.")

        if estado_operativo:
            estado_operativo.asignar_coleccion(lista_datos, total_original=total_bajo_stock)
            estado_operativo.registrar_accion(
                "PRODUCTOS_BAJO_STOCK",
                cantidad=total_bajo_stock,
            )
            estado_operativo.producto_id = None
            estado_operativo.producto_codigo = None
            estado_operativo.producto_nombre = None
            HakuOperationalContext.guardar(estado_operativo)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": "\n".join(lineas_pantalla),
            "mensaje_verbal": normalizar_para_tts(" ".join(lineas_voz)),
            "datos": {"bajo_stock": lista_datos, "total": total_bajo_stock},
        }

    # 3-bis F. Consulta de reposición integral («¿Qué hace falta por pedir?»)
    if intencion.tipo == IntencionTipo.CONSULTAR_FALTANTES_REPOSICION:
        if not usuario_tiene_permiso_operativo(usuario, "analitica.ver_alertas"):
            msg = "No cuentas con el permiso requerido (analitica.ver_alertas) para consultar faltantes y reposición de inventario."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar faltantes de stock."),
                "datos": {"error_permiso": True, "permiso_requerido": "analitica.ver_alertas"},
            }

        if intencion.almacen:
            almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
            if error_resp:
                return error_resp
            scope_user = f"alm_{almacen.id}"
        elif usuario.es_administrador or usuario.acceso_todos_almacenes:
            almacen = None
            scope_user = f"global_rol_{usuario.rol}"
        else:
            almacen = None
            alms_ids = sorted(list(usuario.obtener_almacenes_accesibles().values_list("id", flat=True)))
            if not alms_ids:
                msg = "Su cuenta de usuario no tiene un almacén asignado activo para realizar esta consulta."
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts(msg),
                    "datos": {"error_permisos": True, "sin_almacen_asignado": True},
                }
            alms_firma = "-".join(map(str, alms_ids))
            scope_user = f"user_{usuario.id}_alms_{alms_firma}"

        # Revisar filtro textual y caché en RAM
        filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
        cached_data = HakuCacheManager.obtener_faltantes(
            almacen_id=almacen.id if almacen else None,
            scope_id=scope_user,
            filtro_nombre=filtro_nombre,
        )
        if cached_data is not None:
            return {
                "tipo": "consulta",
                "mensaje_pantalla": cached_data["pantalla"],
                "mensaje_verbal": cached_data["verbal"],
                "datos": cached_data["datos"],
            }

        # Query de existencias activas
        qs_exist = Existencia.objects.filter(
            producto__activo=True,
            producto__archivado=False,
        ).select_related("producto", "producto__categoria", "almacen")

        if almacen:
            qs_exist = qs_exist.filter(almacen=almacen)
        elif usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
            alms_acc = usuario.obtener_almacenes_accesibles()
            qs_exist = qs_exist.filter(almacen__in=alms_acc)

        filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
        if filtro_nombre:
            filtro_q = construir_filtro_familia_o_tipo(filtro_nombre, prefix_relacion="producto__")
            qs_exist = qs_exist.filter(filtro_q)

        agotados_items = []
        bajo_stock_items = []
        items_datos = []

        for e in qs_exist:
            prod = e.producto
            disponible = e.stock_disponible
            prestado = e.stock_en_prestamo
            minimo = prod.stock_minimo
            fisico = disponible + prestado

            if disponible == 0:
                motivo = "Sin existencias disponibles"
                if prestado > 0:
                    motivo += f" ({prestado} unidad(es) prestadas)"
                sug_compra = max(0, minimo - disponible)
                agotados_items.append({
                    "existencia": e,
                    "motivo": motivo,
                    "sugerido": sug_compra if sug_compra > 0 else (1 if minimo == 0 else minimo),
                })
                items_datos.append({
                    "id": prod.id,
                    "codigo": prod.codigo_interno,
                    "nombre": prod.nombre,
                    "categoria": prod.categoria.nombre,
                    "almacen": e.almacen.nombre,
                    "stock_fisico": fisico,
                    "stock_prestado": prestado,
                    "stock_disponible": disponible,
                    "stock_minimo": minimo,
                    "estado_alerta": "AGOTADO",
                    "motivo_alerta": motivo,
                    "cantidad_sugerida_reposicion": sug_compra if sug_compra > 0 else (1 if minimo == 0 else minimo),
                    "consumo_estimado": "Sin historial suficiente para predicción de consumo",
                })
            elif e.stock_bajo:
                motivo = f"Por debajo del mínimo ({disponible} de {minimo})"
                sug_compra = max(0, minimo - disponible)
                bajo_stock_items.append({
                    "existencia": e,
                    "motivo": motivo,
                    "sugerido": sug_compra,
                })
                items_datos.append({
                    "id": prod.id,
                    "codigo": prod.codigo_interno,
                    "nombre": prod.nombre,
                    "categoria": prod.categoria.nombre,
                    "almacen": e.almacen.nombre,
                    "stock_fisico": fisico,
                    "stock_prestado": prestado,
                    "stock_disponible": disponible,
                    "stock_minimo": minimo,
                    "estado_alerta": "BAJO_STOCK",
                    "motivo_alerta": motivo,
                    "cantidad_sugerida_reposicion": sug_compra,
                    "consumo_estimado": "Sin historial suficiente para predicción de consumo",
                })

        total_faltantes = len(items_datos)

        if total_faltantes == 0:
            msg = (
                f"No hay productos que requieran reposición en {almacen.nombre}."
                if almacen else "No hay productos que requieran reposición en los almacenes accesibles."
            )
            return {
                "tipo": "consulta",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"faltantes": [], "total": 0},
            }

        lineas_pantalla = ["### Reporte de Reposición y Necesidades de Compra\n"]
        lineas_voz = []

        if agotados_items:
            lineas_pantalla.append(f"**Agotados sin stock disponible ({len(agotados_items)}):**")
            lineas_voz.append(f"Hay {len(agotados_items)} productos agotados.")
            for it in agotados_items[:6]:
                ex = it["existencia"]
                lineas_pantalla.append(
                    f"- 🔴 **{ex.producto.nombre}** (`{ex.producto.codigo_interno}`) en {ex.almacen.nombre}: "
                    f"0 disponibles (físico: {ex.stock_total}, prestados: {ex.stock_en_prestamo}, mínimo: {ex.producto.stock_minimo}). "
                    f"*Sugerido reponer:* {it['sugerido']} ud(s)."
                )

        if bajo_stock_items:
            lineas_pantalla.append(f"\n**Stock bajo por debajo del mínimo ({len(bajo_stock_items)}):**")
            lineas_voz.append(f"{len(bajo_stock_items)} productos están por debajo de su stock mínimo.")
            for it in bajo_stock_items[:6]:
                ex = it["existencia"]
                lineas_pantalla.append(
                    f"- 🟡 **{ex.producto.nombre}** (`{ex.producto.codigo_interno}`) en {ex.almacen.nombre}: "
                    f"**{ex.stock_disponible}** disponibles de **{ex.producto.stock_minimo}** requeridos. "
                    f"*Sugerido reponer:* {it['sugerido']} ud(s)."
                )

        lineas_pantalla.append(
            "\n*Nota: Las sugerencias de compra se basan estrictamente en alcanzar el stock mínimo configurado. "
            "No se calculan plazos de agotamiento ficticios al no disponer de tasas de consumo histórico.*"
        )

        nombres_destacados = [it["existencia"].producto.nombre for it in (agotados_items + bajo_stock_items)[:3]]
        lineas_voz.append(f"Entre ellos: {', '.join(nombres_destacados)}.")

        txt_pantalla = "\n".join(lineas_pantalla)
        txt_verbal = normalizar_para_tts(" ".join(lineas_voz))

        if estado_operativo:
            estado_operativo.asignar_coleccion(items_datos, total_original=total_faltantes)
            estado_operativo.registrar_accion("CONSULTAR_FALTANTES_REPOSICION", cantidad=total_faltantes)
            estado_operativo.producto_id = None
            estado_operativo.producto_codigo = None
            estado_operativo.producto_nombre = None
            HakuOperationalContext.guardar(estado_operativo)

        # Guardar en caché
        HakuCacheManager.guardar_faltantes(
            {"pantalla": txt_pantalla, "verbal": txt_verbal, "datos": {"faltantes": items_datos, "total": total_faltantes}},
            almacen_id=almacen.id if almacen else None,
            scope_id=scope_user,
            filtro_nombre=filtro_nombre,
        )

        return {
            "tipo": "consulta",
            "mensaje_pantalla": txt_pantalla,
            "mensaje_verbal": txt_verbal,
            "datos": {"faltantes": items_datos, "total": total_faltantes},
        }

    # 4. Movimientos por producto específico
    if intencion.tipo == IntencionTipo.CONSULTAR_PRODUCTO_MOVIMIENTOS:
        if not usuario_tiene_permiso_operativo(usuario, "movimientos.ver"):
            msg = "No cuentas con el permiso requerido (movimientos.ver) para consultar el historial de movimientos."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar el historial de movimientos."),
                "datos": {"error_permiso": True, "permiso_requerido": "movimientos.ver"},
            }

        producto = intencion.producto
        if not producto and sesion and sesion.get("haku_ultimo_producto_id"):
            producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
        elif not producto and estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
            pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
            producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        if not producto:
            msg = "¿De cuál producto desea consultar los movimientos?"
            return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(producto.id)
            sesion["haku_ultimo_acceso"] = time.time()

        from .servicios_temporales import resolver_intervalo_temporal, obtener_ahora_local
        intervalo = resolver_intervalo_temporal(texto)
        if not intervalo:
            ahora = obtener_ahora_local()
            desc_periodo = "el periodo consultado"
            filtro_tiempo = {}
        else:
            desc_periodo = intervalo.descripcion_legible
            filtro_tiempo = {
                "fecha_hora__gte": intervalo.inicio,
                "fecha_hora__lt": intervalo.fin,
            }

        qs_mov = Movimiento.objects.filter(
            producto=producto,
            estado=Movimiento.Estado.CONFIRMADO,
            **filtro_tiempo,
        ).exclude(tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL).select_related("almacen_origen", "almacen_destino")

        # Filtrar almacenes según permisos
        if intencion.almacen:
            almacen_filtrado, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
            if error_resp:
                return error_resp
            qs_mov = qs_mov.filter(Q(almacen_origen=almacen_filtrado) | Q(almacen_destino=almacen_filtrado)).distinct()
            desc_periodo += f" en {almacen_filtrado.nombre}"
        elif usuario.es_usuario_normal:
            if usuario.acceso_todos_almacenes:
                pass
            else:
                alms = usuario.obtener_almacenes_accesibles()
                if not alms.exists():
                    msg = "Su cuenta de usuario no tiene un almacén asignado para consultar movimientos."
                    return {"tipo": "consulta", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg), "datos": {}}
                qs_mov = qs_mov.filter(Q(almacen_origen__in=alms) | Q(almacen_destino__in=alms)).distinct()

        total_mov = qs_mov.count()
        entradas = qs_mov.filter(tipo=Movimiento.Tipo.ENTRADA).count()
        salidas = qs_mov.filter(tipo=Movimiento.Tipo.SALIDA).count()
        prestamos = qs_mov.filter(tipo=Movimiento.Tipo.PRESTAMO).count()
        devoluciones = qs_mov.filter(tipo=Movimiento.Tipo.DEVOLUCION).count()
        traslados = qs_mov.filter(tipo__in=[Movimiento.Tipo.TRASLADO_ENTRE, Movimiento.Tipo.TRASLADO_INTERNO]).count()
        iniciales = qs_mov.filter(tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL).count()
        otros = max(0, total_mov - (entradas + salidas + prestamos + devoluciones + traslados + iniciales))

        if estado_operativo:
            estado_operativo.registrar_accion(
                "CONSULTAR_PRODUCTO_MOVIMIENTOS",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
                cantidad=total_mov,
            )
            HakuOperationalContext.guardar(estado_operativo)

        total_mov_txt = pluralizar(total_mov, "movimiento", "movimientos")
        entradas_txt = pluralizar(entradas, "entrada", "entradas")
        salidas_txt = pluralizar(salidas, "salida", "salidas")
        prestamos_txt = pluralizar(prestamos, "préstamo", "préstamos")
        devoluciones_txt = pluralizar(devoluciones, "devolución", "devoluciones")
        traslados_txt = pluralizar(traslados, "traslado", "traslados")

        total_mov_md = pluralizar_md(total_mov, "movimiento", "movimientos")
        entradas_md = pluralizar_md(entradas, "entrada", "entradas")
        salidas_md = pluralizar_md(salidas, "salida", "salidas")
        prestamos_md = pluralizar_md(prestamos, "préstamo", "préstamos")
        devoluciones_md = pluralizar_md(devoluciones, "devolución", "devoluciones")
        traslados_md = pluralizar_md(traslados, "traslado", "traslados")

        if total_mov == 0:
            pantalla = f"El producto {producto.nombre} (`{producto.codigo_interno}`) no registra movimientos de inventario en {desc_periodo}."
            verbal = f"El producto {producto.nombre} no registra movimientos de inventario en {desc_periodo}."
        else:
            detalle_lineas = []
            for m in qs_mov[:8]:
                fecha_str = m.fecha_hora.astimezone(obtener_ahora_local().tzinfo).strftime("%d/%m %H:%M")
                alm_txt = m.almacen_origen.nombre if m.almacen_origen else ""
                if m.almacen_destino:
                    alm_txt += f" ➔ {m.almacen_destino.nombre}"
                detalle_lineas.append(f"- `{fecha_str}` | **{m.get_tipo_display()}**: {pluralizar(m.cantidad, 'ud', 'uds')} ({alm_txt})")

            bloque_det = "\n".join(detalle_lineas)
            extra_pag = f"\n\n*(Mostrando primeros 8 de {total_mov_txt})*" if total_mov > 8 else ""

            pantalla = (
                f"### Movimientos de {producto.nombre} (`{producto.codigo_interno}`) — {desc_periodo}\n"
                f"El {producto.nombre} tuvo {total_mov_md}:\n"
                f"- {entradas_md} | {salidas_md} | {prestamos_md} | {devoluciones_md} | {traslados_md}\n\n"
                f"**Detalle de actividad:**\n{bloque_det}{extra_pag}"
            )
            verbal = (
                f"El producto {producto.nombre} tuvo {total_mov_txt} durante {desc_periodo}: "
                f"{entradas_txt}, {salidas_txt}, {prestamos_txt}, {devoluciones_txt} y {traslados_txt}."
            )
        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
            "datos": {
                "producto_id": str(producto.id),
                "codigo": producto.codigo_interno,
                "nombre": producto.nombre,
                "periodo": desc_periodo,
                "total": total_mov,
                "entradas": entradas,
                "salidas": salidas,
                "prestamos": prestamos,
                "devoluciones": devoluciones,
                "traslados": traslados,
            },
        }

    # 5. Alertas de stock bajo / agotado
    if intencion.tipo == IntencionTipo.ALERTAS_STOCK:
        if not usuario_tiene_permiso_operativo(usuario, "analitica.ver_alertas"):
            msg = "No cuentas con el permiso requerido (analitica.ver_alertas) para consultar las alertas de existencias."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar alertas de inventario."),
                "datos": {"error_permiso": True, "permiso_requerido": "analitica.ver_alertas"},
            }

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        qs_existencias = Existencia.objects.filter(
            producto__activo=True,
            producto__archivado=False,
        ).select_related("producto", "almacen")

        if almacen:
            qs_existencias = qs_existencias.filter(almacen=almacen)

        filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None
        if filtro_nombre:
            filtro_q = construir_filtro_familia_o_tipo(filtro_nombre, prefix_relacion="producto__")
            qs_existencias = qs_existencias.filter(filtro_q)

        sin_stock = [e for e in qs_existencias if e.sin_stock][:5]
        bajo_stock = [e for e in qs_existencias if e.stock_bajo][:5]

        if not sin_stock and not bajo_stock:
            if almacen:
                res = f"No hay alertas activas de stock en {almacen.nombre}. Todos los productos tienen existencias suficientes."
            elif filtro_nombre:
                res = f"No hay alertas activas de stock para '{filtro_nombre}'. Todos los productos tienen existencias suficientes."
            else:
                res = "No hay alertas activas de stock en este momento. Todos los productos tienen existencias suficientes."
            return {"tipo": "consulta", "mensaje_pantalla": res, "mensaje_verbal": res, "datos": {}}

        lineas = []
        lineas_voz = []
        if sin_stock:
            nombres = ", ".join(f"{e.producto.nombre} ({e.almacen.nombre})" for e in sin_stock)
            lineas.append(f"**Agotados ({len(sin_stock)}):** {nombres}")
            lineas_voz.append(f"Hay {len(sin_stock)} productos agotados: {nombres}.")
        if bajo_stock:
            nombres = ", ".join(f"{e.producto.nombre} ({e.stock_disponible}/{e.producto.stock_minimo})" for e in bajo_stock)
            lineas.append(f"**Stock bajo ({len(bajo_stock)}):** {nombres}")
            lineas_voz.append(f"Hay {len(bajo_stock)} productos con stock bajo.")

        if estado_operativo:
            lista_alertas = []
            for e in sin_stock + bajo_stock:
                lista_alertas.append({
                    "id": e.producto.id,
                    "producto_id": str(e.producto.id),
                    "codigo": e.producto.codigo_interno,
                    "nombre": e.producto.nombre,
                    "almacen": e.almacen.nombre,
                })
            estado_operativo.asignar_coleccion(lista_alertas)
            estado_operativo.registrar_accion(
                "ALERTAS_STOCK",
                cantidad=len(sin_stock) + len(bajo_stock),
            )
            estado_operativo.producto_id = None
            estado_operativo.producto_codigo = None
            estado_operativo.producto_nombre = None
            HakuOperationalContext.guardar(estado_operativo)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": "\n\n".join(lineas),
            "mensaje_verbal": " ".join(lineas_voz),
            "datos": {
                "sin_stock": [{"codigo": e.producto.codigo_interno, "nombre": e.producto.nombre, "almacen": e.almacen.nombre} for e in sin_stock],
                "bajo_stock": [
                    {"codigo": e.producto.codigo_interno, "nombre": e.producto.nombre, "almacen": e.almacen.nombre, "disponible": e.stock_disponible, "minimo": e.producto.stock_minimo}
                    for e in bajo_stock
                ],
            },
        }

    # 3. Préstamos activos
    if intencion.tipo == IntencionTipo.PRESTAMOS_ACTIVOS:
        if not usuario_tiene_permiso_operativo(usuario, "movimientos.ver"):
            msg = "No cuentas con el permiso requerido (movimientos.ver) para consultar los préstamos activos."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar préstamos activos."),
                "datos": {"error_permiso": True, "permiso_requerido": "movimientos.ver"},
            }

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        qs_prestamos = Prestamo.objects.filter(
            estado__in=[Prestamo.Estado.ABIERTO, Prestamo.Estado.DEVUELTO_PARCIAL]
        ).select_related("producto", "almacen")

        if almacen:
            qs_prestamos = qs_prestamos.filter(almacen=almacen)
        else:
            alms_acc = usuario.obtener_almacenes_accesibles() if hasattr(usuario, "obtener_almacenes_accesibles") else None
            if alms_acc is not None:
                qs_prestamos = qs_prestamos.filter(almacen__in=alms_acc)

        producto = intencion.producto
        if not producto and sesion and sesion.get("haku_ultimo_producto_id"):
            producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
        elif not producto and estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
            pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
            producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        filtro_nombre = intencion.datos_extra.get("filtro_nombre") if intencion.datos_extra else None

        if producto:
            if not getattr(producto, "es_prestable", False):
                msg = f"El producto {producto.nombre} es un artículo consumible y no admite préstamos."
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": msg,
                    "mensaje_verbal": normalizar_para_tts(msg),
                    "datos": {"prestamos": [], "total": 0, "total_prestamos": 0, "total_unidades": 0, "total_productos_distintos": 0, "no_prestable": True},
                }
            qs_prestamos = qs_prestamos.filter(producto=producto)
            if sesion is not None:
                sesion["haku_ultimo_producto_id"] = str(producto.id)
                sesion["haku_ultimo_acceso"] = time.time()
        elif filtro_nombre:
            filtro_q = construir_filtro_familia_o_tipo(filtro_nombre, prefix_relacion="producto__")
            qs_prestamos = qs_prestamos.filter(filtro_q)

        from django.db.models import Sum
        total_prestamos = qs_prestamos.count()
        agg_unidades = qs_prestamos.aggregate(Sum("cantidad_pendiente_devolucion"))
        total_unidades = agg_unidades.get("cantidad_pendiente_devolucion__sum") or 0
        total_productos_distintos = qs_prestamos.values("producto").distinct().count()
        prestamos = list(qs_prestamos[:10])
        metrica = intencion.datos_extra.get("metrica", "GENERAL") if intencion.datos_extra else "GENERAL"

        if not prestamos:
            if producto:
                if almacen:
                    msg = f"El producto {producto.nombre} no tiene préstamos activos pendientes en {almacen.nombre}."
                else:
                    msg = f"El producto {producto.nombre} no tiene préstamos activos pendientes."
            elif filtro_nombre:
                msg = f"No hay préstamos activos para productos de la categoría o tipo '{filtro_nombre}'."
            elif almacen:
                msg = f"No hay préstamos activos pendientes en {almacen.nombre}."
            else:
                msg = "No hay préstamos activos pendientes en este almacén." if usuario.es_usuario_normal else "No hay préstamos activos pendientes en el inventario."
            return {"tipo": "consulta", "mensaje_pantalla": msg, "mensaje_verbal": msg, "datos": {"prestamos": [], "total": 0, "total_prestamos": 0, "total_unidades": 0, "total_productos_distintos": 0}}

        items = []
        prest_txt = pluralizar(total_prestamos, "préstamo activo", "préstamos activos")
        unid_txt = pluralizar(total_unidades, "unidad", "unidades")
        prod_dist_txt = pluralizar(total_productos_distintos, "producto distinto", "productos distintos")

        if producto:
            pend_txt = "unidad pendiente" if total_unidades == 1 else "unidades pendientes"
            lineas_voz = [f"El {producto.nombre} tiene {total_prestamos} {pluralizar(total_prestamos, 'préstamo activo', 'préstamos activos')} con {total_unidades} {pend_txt} de devolución:"]
            for p in prestamos:
                items.append(f"- **{p.cantidad_pendiente_devolucion} ud(s)** prestadas a **{p.destinatario}** ({p.almacen.nombre})")
                lineas_voz.append(f"{pluralizar(p.cantidad_pendiente_devolucion, 'unidad', 'unidades')} a {p.destinatario}.")
            if total_prestamos > 10:
                items.append(f"\n*Mostrando 10 de {total_prestamos} préstamos activos.*")
            pantalla = f"### Préstamos activos de {producto.nombre} (`{producto.codigo_interno}`):\n- **Total préstamos:** {total_prestamos}\n- **Unidades pendientes:** {total_unidades}\n\n" + "\n".join(items)
        elif metrica == "CONTEO_REGISTROS":
            lineas_voz = [f"Hay {total_prestamos} {prest_txt} en este momento, correspondientes a {total_unidades} {unid_txt} de {total_productos_distintos} {prod_dist_txt}."]
            for p in prestamos:
                items.append(f"- **{p.producto.nombre}**: {p.cantidad_pendiente_devolucion} ud(s) prestadas a **{p.destinatario}** ({p.almacen.nombre})")
            if total_prestamos > 10:
                items.append(f"\n*Mostrando 10 de {total_prestamos} préstamos activos.*")
            pantalla = f"### Conteo de préstamos activos ({total_prestamos} en total):\n- **Total préstamos activos:** {total_prestamos}\n- **Total unidades prestadas:** {total_unidades}\n- **Productos distintos involucrados:** {total_productos_distintos}\n\n" + "\n".join(items)
        elif metrica == "CONTEO_UNIDADES":
            lineas_voz = [f"Hay {total_unidades} {pluralizar(total_unidades, 'unidad prestada', 'unidades prestadas')} en total, distribuidas en {total_prestamos} {prest_txt}."]
            for p in prestamos:
                items.append(f"- **{p.producto.nombre}**: {p.cantidad_pendiente_devolucion} ud(s) prestadas a **{p.destinatario}** ({p.almacen.nombre})")
            if total_prestamos > 10:
                items.append(f"\n*Mostrando 10 de {total_prestamos} préstamos activos.*")
            pantalla = f"### Unidades activas en préstamo ({total_unidades} en total):\n- **Total unidades pendientes de devolución:** {total_unidades}\n- **Total préstamos activos:** {total_prestamos}\n- **Productos distintos involucrados:** {total_productos_distintos}\n\n" + "\n".join(items)
        elif metrica == "LISTA_RESPONSABLES":
            resp_str = ", ".join(f"{p.destinatario} ({pluralizar(p.cantidad_pendiente_devolucion, 'ud', 'uds')} de {p.producto.nombre})" for p in prestamos[:3])
            lineas_voz = [f"Hay {total_prestamos} {prest_txt} a cargo de: {resp_str}."]
            for p in prestamos:
                items.append(f"- **{p.destinatario}**: {p.cantidad_pendiente_devolucion} ud(s) de **{p.producto.nombre}** ({p.almacen.nombre})")
            if total_prestamos > 10:
                items.append(f"\n*Mostrando 10 de {total_prestamos} préstamos activos.*")
            pantalla = f"### Préstamos activos por responsable ({total_prestamos} en total):\n" + "\n".join(items)
        else:
            lineas_voz = [f"Hay {total_prestamos} {prest_txt}:"]
            for p in prestamos:
                items.append(f"- **{p.producto.nombre}**: {p.cantidad_pendiente_devolucion} ud(s) prestadas a **{p.destinatario}** ({p.almacen.nombre})")
                lineas_voz.append(f"{p.cantidad_pendiente_devolucion} de {p.producto.nombre} a {p.destinatario}.")
            if total_prestamos > 10:
                items.append(f"\n*Mostrando 10 de {total_prestamos} préstamos activos.*")
            pantalla = f"### Préstamos activos ({total_prestamos} en total):\n" + "\n".join(items)

        if estado_operativo:
            if producto:
                estado_operativo.registrar_accion(
                    "PRESTAMOS_ACTIVOS_PRODUCTO",
                    producto=producto.nombre,
                    producto_id=producto.id,
                    codigo=producto.codigo_interno,
                    cantidad=total_prestamos,
                )
            else:
                lista_prestamos = [
                    {
                        "id": p.producto.id,
                        "producto_id": str(p.producto.id),
                        "codigo": p.producto.codigo_interno,
                        "nombre": p.producto.nombre,
                        "almacen": p.almacen.nombre,
                        "destinatario": p.destinatario,
                        "cantidad_prestada": p.cantidad_pendiente_devolucion,
                    }
                    for p in prestamos
                ]
                estado_operativo.asignar_coleccion(lista_prestamos, total_original=total_prestamos)
                estado_operativo.registrar_accion("PRESTAMOS_ACTIVOS", cantidad=total_prestamos)
                estado_operativo.producto_id = None
                estado_operativo.producto_codigo = None
                estado_operativo.producto_nombre = None
            HakuOperationalContext.guardar(estado_operativo)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(" ".join(lineas_voz)),
            "datos": {
                "prestamos": [
                    {
                        "id": p.id,
                        "producto": p.producto.nombre,
                        "codigo": p.producto.codigo_interno,
                        "destinatario": p.destinatario,
                        "cantidad": p.cantidad_pendiente_devolucion,
                        "almacen": p.almacen.nombre,
                    }
                    for p in prestamos
                ],
                "total": total_prestamos,
                "total_unidades": total_unidades,
            },
        }

    # 4-bis. Consulta de ubicación de colección o conjunto de productos
    if intencion.tipo == IntencionTipo.CONSULTAR_UBICACION_COLECCION:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar ubicaciones de productos."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar ubicaciones."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }

        coleccion = []
        if intencion.candidatos:
            coleccion = intencion.candidatos
        elif intencion.datos_extra and intencion.datos_extra.get("candidatos_plurales"):
            coleccion = intencion.datos_extra["candidatos_plurales"]
        elif estado_operativo and estado_operativo.coleccion_productos:
            coleccion = estado_operativo.coleccion_productos

        if not coleccion:
            msg = "No tengo una lista previa de productos en la conversación activa. Puede consultar el catálogo o los productos con stock primero."
            return {
                "tipo": "aclaracion",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {"productos": []},
            }

        total_orig = (estado_operativo.coleccion_total_original if estado_operativo else 0) or len(coleccion)
        if total_orig < len(coleccion):
            total_orig = len(coleccion)
        es_parcial = total_orig > len(coleccion)

        lineas_pantalla = ["### Ubicación de las referencias consultadas:\n"]
        if es_parcial:
            lineas_voz = [f"Ubicación de {len(coleccion)} de las {total_orig} referencias consultadas:"]
        else:
            lineas_voz = [f"Ubicación de {len(coleccion)} productos:"]
        lista_datos = []

        alms_usuario = list(usuario.obtener_almacenes_accesibles()) if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes else []

        for item in coleccion:
            prod_id = item.get("id") or item.get("producto_id")
            prod = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first() if prod_id else None
            if not prod:
                continue

            alm = None
            if alms_usuario:
                alm_cat = getattr(prod.categoria, "almacen_principal", None)
                if alm_cat and alm_cat in alms_usuario:
                    alm = alm_cat
                else:
                    alm = alms_usuario[0]
            else:
                alm = getattr(prod.categoria, "almacen_principal", None)
            ubicaciones = HakuCacheManager.obtener_o_cargar_ubicacion(prod, almacen_id=alm.id if alm else None, usuario=usuario)
            nombre_alm = alm.nombre if alm else "Sin almacén"

            lineas_pantalla.append(f"- **{prod.nombre}** (`{prod.codigo_interno}`): **{ubicaciones}** en *{nombre_alm}*")
            lineas_voz.append(f"{prod.nombre} en {ubicaciones}, {nombre_alm}.")
            lista_datos.append({
                "producto_id": str(prod.id),
                "codigo": prod.codigo_interno,
                "nombre": prod.nombre,
                "ubicaciones": ubicaciones,
                "almacen": nombre_alm,
            })

        if es_parcial:
            lineas_pantalla.append(
                f"\n*Aviso: Mostrando la ubicación de las {len(lista_datos)} referencias conservadas en el foco conversacional activo (de {total_orig} productos devueltos por la consulta anterior).*"
            )

        if estado_operativo:
            estado_operativo.registrar_accion(
                "CONSULTAR_UBICACION_COLECCION",
                cantidad=len(lista_datos),
            )
            HakuOperationalContext.guardar(estado_operativo)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": "\n".join(lineas_pantalla),
            "mensaje_verbal": normalizar_para_tts(" ".join(lineas_voz)),
            "datos": {
                "productos": lista_datos,
                "total": len(lista_datos),
                "total_original": total_orig,
                "es_parcial": es_parcial,
            },
        }

    # 4. Consulta de ubicación específica
    if intencion.tipo == IntencionTipo.CONSULTAR_UBICACION:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar la ubicación de productos."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar la ubicación del producto."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }

        producto = intencion.producto
        if not producto and sesion and sesion.get("haku_ultimo_producto_id"):
            producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
        elif not producto and estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
            pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
            producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        if not producto:
            msg = "¿De cuál producto desea conocer la ubicación?"
            return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(producto.id)
            sesion["haku_ultimo_acceso"] = time.time()

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        if almacen:
            qs_ub = producto.ubicaciones.filter(almacen=almacen)
            ubicaciones = ", ".join(u.nombre for u in qs_ub) if qs_ub.exists() else f"Sin ubicación registrada en {almacen.nombre}"
            cached_stock = HakuCacheManager.obtener_o_cargar_stock(producto.id, almacen.id, usuario=usuario)
            disponible = cached_stock["disponible"]
            pantalla = (
                f"### Ubicación de {producto.nombre} (`{producto.codigo_interno}`)\n"
                f"- **Ubicación:** {ubicaciones}\n"
                f"- **Almacén:** {almacen.nombre}\n"
                f"- **Stock disponible:** {pluralizar(disponible, 'unidad', 'unidades')}"
            )
            voz = f"{producto.nombre} está ubicado en {ubicaciones}, en {almacen.nombre}."
        else:
            qs_ub = producto.ubicaciones.filter(almacen__activo=True).select_related("almacen")
            if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
                qs_ub = qs_ub.filter(almacen__in=usuario.obtener_almacenes_accesibles())

            alms_con_ub = {u.almacen for u in qs_ub}
            if len(alms_con_ub) > 1:
                detalles = []
                detalles_voz = []
                for alm_i in sorted(alms_con_ub, key=lambda a: a.nombre):
                    ubs_alm = [u.nombre for u in qs_ub if u.almacen_id == alm_i.id]
                    st = Existencia.objects.filter(producto=producto, almacen=alm_i).first()
                    disp_alm = st.stock_disponible if st else 0
                    detalles.append(f"- **{alm_i.nombre}:** {', '.join(ubs_alm)} ({pluralizar(disp_alm, 'unidad disponible', 'unidades disponibles')})")
                    detalles_voz.append(f"en {alm_i.nombre}, {', '.join(ubs_alm)}")
                ubicaciones = "; ".join(f"{alm_i.nombre}: {', '.join(u.nombre for u in qs_ub if u.almacen_id == alm_i.id)}" for alm_i in sorted(alms_con_ub, key=lambda a: a.nombre))
                pantalla = (
                    f"### Ubicaciones de {producto.nombre} (`{producto.codigo_interno}`)\n"
                    + "\n".join(detalles)
                )
                voz = f"{producto.nombre} está ubicado: {', '.join(detalles_voz)}."
            elif len(alms_con_ub) == 1:
                alm_i = list(alms_con_ub)[0]
                ubs_alm = [u.nombre for u in qs_ub if u.almacen_id == alm_i.id]
                ubicaciones = ", ".join(ubs_alm)
                st = Existencia.objects.filter(producto=producto, almacen=alm_i).first()
                disponible = st.stock_disponible if st else 0
                pantalla = (
                    f"### Ubicación de {producto.nombre} (`{producto.codigo_interno}`)\n"
                    f"- **Ubicación:** {ubicaciones}\n"
                    f"- **Almacén:** {alm_i.nombre}\n"
                    f"- **Stock disponible:** {pluralizar(disponible, 'unidad', 'unidades')}"
                )
                voz = f"{producto.nombre} está ubicado en {ubicaciones}, en {alm_i.nombre}."
            else:
                alm_cat = getattr(producto.categoria, "almacen_principal", None)
                ubicaciones = f"Sin ubicación registrada en {alm_cat.nombre if alm_cat else 'el inventario'}"
                pantalla = (
                    f"### Ubicación de {producto.nombre} (`{producto.codigo_interno}`)\n"
                    f"- **Ubicación:** {ubicaciones}\n"
                    f"- **Almacén:** {alm_cat.nombre if alm_cat else 'Sin almacén'}\n"
                    f"- **Stock disponible:** 0 unidades"
                )
                voz = f"{producto.nombre} no tiene una ubicación registrada en el inventario."

        if estado_operativo:
            estado_operativo.registrar_accion(
                "CONSULTAR_UBICACION",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
            )
            HakuOperationalContext.guardar(estado_operativo)

        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(voz),
            "datos": {"producto_id": str(producto.id), "codigo": producto.codigo_interno, "ubicaciones": ubicaciones},
        }

    # 5. Consulta de stock o búsqueda de producto
    if intencion.tipo in (IntencionTipo.CONSULTAR_STOCK, IntencionTipo.BUSCAR_PRODUCTO):
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.ver"):
            msg = "No cuentas con el permiso requerido (catalogo.ver) para consultar el catálogo o existencias."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con permisos suficientes para consultar el catálogo de inventario."),
                "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
            }

        producto = intencion.producto
        if not producto and sesion and sesion.get("haku_ultimo_producto_id"):
            producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
        elif not producto and estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
            pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
            producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        if not producto:
            msg = "¿De cuál producto desea consultar el stock o existencias?"
            return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(producto.id)
            sesion["haku_ultimo_acceso"] = time.time()

        almacen, error_resp = _validar_acceso_almacen(usuario, intencion.almacen)
        if error_resp:
            return error_resp

        otro_almacen = bool(intencion.datos_extra and intencion.datos_extra.get("otro_almacen"))

        if not almacen and not otro_almacen:
            qs_ex = Existencia.objects.filter(producto=producto, almacen__activo=True).select_related("almacen")
            if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
                qs_ex = qs_ex.filter(almacen__in=usuario.obtener_almacenes_accesibles())

            lista_existencias = list(qs_ex)
            if len(lista_existencias) > 1:
                total_disp = sum(e.stock_disponible for e in lista_existencias)
                total_prest = sum(e.stock_en_prestamo for e in lista_existencias)
                lineas_alm = []
                lineas_v = []
                es_prestable = getattr(producto, "es_prestable", False)
                for e in lista_existencias:
                    qs_ub = producto.ubicaciones.filter(almacen=e.almacen)
                    ub_txt = f" (Ubicación: {', '.join(u.nombre for u in qs_ub)})" if qs_ub.exists() else ""
                    disp_alm_txt = pluralizar(e.stock_disponible, "disponible", "disponibles")
                    prest_alm = f", {pluralizar(e.stock_en_prestamo, 'en préstamo', 'en préstamo')}" if es_prestable and e.stock_en_prestamo else ""
                    lineas_alm.append(f"- **{e.almacen.nombre}:** {e.stock_disponible} {disp_alm_txt}{prest_alm}{ub_txt}")
                    lineas_v.append(f"{e.stock_disponible} en {e.almacen.nombre}")

                disp_global_txt = pluralizar(total_disp, "unidad", "unidades")
                prest_global_txt = pluralizar(total_prest, "unidad", "unidades")
                voz_disp_txt = pluralizar(total_disp, "unidad disponible", "unidades disponibles")
                pantalla = (
                    f"### {producto.nombre} (`{producto.codigo_interno}`)\n"
                    f"- **Stock global disponible:** {disp_global_txt}\n"
                    + (f"- **Total en préstamo:** {prest_global_txt}\n" if es_prestable else "")
                    + f"- **Stock mínimo:** {producto.stock_minimo}\n\n"
                    f"**Distribución por almacén:**\n"
                    + "\n".join(lineas_alm)
                )
                voz = f"{producto.nombre} cuenta con {voz_disp_txt} en total: {', '.join(lineas_v)}."
                if estado_operativo:
                    estado_operativo.registrar_accion(
                        "CONSULTAR_STOCK",
                        producto=producto.nombre,
                        producto_id=producto.id,
                        codigo=producto.codigo_interno,
                        cantidad=total_disp,
                    )
                    HakuOperationalContext.guardar(estado_operativo)
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": pantalla,
                    "mensaje_verbal": normalizar_para_tts(voz),
                    "datos": {
                        "producto_id": str(producto.id),
                        "codigo": producto.codigo_interno,
                        "disponible_global": total_disp,
                        "distribucion": [{"almacen": e.almacen.nombre, "disponible": e.stock_disponible, "prestamo": e.stock_en_prestamo} for e in lista_existencias],
                    },
                }
            elif len(lista_existencias) == 1:
                almacen = lista_existencias[0].almacen
            else:
                almacen = getattr(producto.categoria, "almacen_principal", None)

        if otro_almacen:
            if usuario.es_usuario_normal:
                msg = f"Su cuenta de usuario solo tiene acceso al almacén {almacen.nombre if almacen else 'asignado'}. No cuenta con permisos para consultar existencias en otros almacenes."
                return {"tipo": "consulta", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg), "datos": {"producto_id": str(producto.id), "codigo": producto.codigo_interno}}

            existencias_otros = Existencia.objects.filter(
                producto=producto,
                almacen__activo=True
            ).exclude(almacen=almacen).select_related("almacen")

            if not existencias_otros.exists():
                pantalla = f"### {producto.nombre} (`{producto.codigo_interno}`)\nNo hay existencias registradas de este producto en otros almacenes diferentes a {almacen.nombre if almacen else 'el principal'}."
                voz = f"No hay existencias registradas de {producto.nombre} en otros almacenes."
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": pantalla,
                    "mensaje_verbal": normalizar_para_tts(voz),
                    "datos": {"producto_id": str(producto.id), "codigo": producto.codigo_interno, "otros_almacenes": []},
                }

            lineas_p = [f"### Existencias de {producto.nombre} (`{producto.codigo_interno}`) en otros almacenes:\n"]
            lineas_v = [f"En otros almacenes, de {producto.nombre}:"]
            datos_otros = []
            for eo in existencias_otros:
                disp_eo_txt = pluralizar(eo.stock_disponible, "disponible", "disponibles")
                prest_eo = f" ({pluralizar(eo.stock_en_prestamo, 'en préstamo', 'en préstamo')})" if eo.stock_en_prestamo else ""
                lineas_p.append(f"- **{eo.almacen.nombre}:** {eo.stock_disponible} {disp_eo_txt}{prest_eo}")
                lineas_v.append(f"En {eo.almacen.nombre}, {pluralizar(eo.stock_disponible, 'unidad disponible', 'unidades disponibles')}.")
                datos_otros.append({
                    "almacen": eo.almacen.nombre,
                    "disponible": eo.stock_disponible,
                    "prestado": eo.stock_en_prestamo,
                })

            if estado_operativo:
                estado_operativo.registrar_accion(
                    "CONSULTAR_STOCK_OTROS_ALMACENES",
                    producto=producto.nombre,
                    producto_id=producto.id,
                    codigo=producto.codigo_interno,
                )
                HakuOperationalContext.guardar(estado_operativo)

            return {
                "tipo": "consulta",
                "mensaje_pantalla": "\n".join(lineas_p),
                "mensaje_verbal": normalizar_para_tts(" ".join(lineas_v)),
                "datos": {
                    "producto_id": str(producto.id),
                    "codigo": producto.codigo_interno,
                    "nombre": producto.nombre,
                    "otros_almacenes": datos_otros,
                },
            }

        cached_stock = HakuCacheManager.obtener_o_cargar_stock(producto.id, almacen.id if almacen else None, usuario=usuario)
        disponible = cached_stock["disponible"]
        prestado = cached_stock["prestado"]
        if almacen:
            qs_ub = producto.ubicaciones.filter(almacen=almacen)
            ubicaciones = ", ".join(u.nombre for u in qs_ub) if qs_ub.exists() else f"Sin ubicación registrada en {almacen.nombre}"
        else:
            ubicaciones = HakuCacheManager.obtener_o_cargar_ubicacion(producto, almacen_id=almacen.id if almacen else None, usuario=usuario)

        if estado_operativo:
            estado_operativo.registrar_accion(
                "CONSULTAR_STOCK",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
                cantidad=disponible,
            )
            HakuOperationalContext.guardar(estado_operativo)

        es_prestable = getattr(producto, "es_prestable", False)
        linea_prestamo_pantalla = f"- **En préstamo:** {pluralizar(prestado, 'unidad', 'unidades')}\n" if es_prestable else ""
        frase_prestamo_voz = f", y {pluralizar(prestado, 'en préstamo', 'en préstamo')}" if es_prestable else ""
        disp_txt = pluralizar(disponible, "unidad disponible", "unidades disponibles")

        pantalla = (
            f"### {producto.nombre} (`{producto.codigo_interno}`)\n"
            f"- **Stock disponible:** {pluralizar(disponible, 'unidad', 'unidades')}\n"
            f"{linea_prestamo_pantalla}"
            f"- **Stock mínimo:** {producto.stock_minimo}\n"
            f"- **Almacén:** {almacen.nombre if almacen else 'Sin almacén'}\n"
            f"- **Ubicación:** {ubicaciones}"
        )
        voz = (
            f"De {producto.nombre}, hay {disp_txt} en {almacen.nombre if almacen else 'el almacén'}"
            f"{frase_prestamo_voz}. Ubicación: {ubicaciones}."
        )
        # Proyección de saldo si el usuario preguntó «¿Y cuánto queda después de eso?»
        if intencion.datos_extra and intencion.datos_extra.get("proyeccion_post_operacion"):
            prop_activa = getattr(estado_operativo, "propuesta_activa", None) if estado_operativo else None
            if prop_activa and prop_activa.get("producto_id") == producto.id:
                prop_cant = prop_activa.get("cantidad", 0)
                prop_op = prop_activa.get("tipo_operacion")
                cant_txt = pluralizar(prop_cant, 'unidad', 'unidades')
                if prop_op in (Movimiento.Tipo.SALIDA, Movimiento.Tipo.PRESTAMO):
                    restante = max(0, disponible - prop_cant)
                    rest_md = pluralizar_md(restante, 'unidad disponible', 'unidades disponibles')
                    rest_voz = pluralizar(restante, 'unidad disponible', 'unidades disponibles')
                    pantalla += f"\n\n> [!NOTE]\n> **Proyección post-operación:** Si se confirma la propuesta actual ({cant_txt}), quedarán {rest_md}."
                    voz += f" Si se confirma la operación de {cant_txt}, quedarán {rest_voz}."
                elif prop_op in (Movimiento.Tipo.ENTRADA, Movimiento.Tipo.DEVOLUCION):
                    restante = disponible + prop_cant
                    rest_md = pluralizar_md(restante, 'unidad disponible', 'unidades disponibles')
                    rest_voz = pluralizar(restante, 'unidad disponible', 'unidades disponibles')
                    pantalla += f"\n\n> [!NOTE]\n> **Proyección post-operación:** Si se confirma la propuesta actual (+{cant_txt}), quedarán {rest_md}."
                    voz += f" Si se confirma la operación de {cant_txt}, quedarán {rest_voz}."

        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(voz),
            "datos": {
                "producto_id": str(producto.id),
                "codigo": producto.codigo_interno,
                "nombre": producto.nombre,
                "disponible": disponible,
                "en_prestamo": prestado,
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # C. MUTACIONES DE INVENTARIO (PRE-CONFIRMACIÓN OBLIGATORIA EN DOS PASOS)
    # ─────────────────────────────────────────────────────────────────────────

    # 6. Entrada de inventario
    if intencion.tipo == IntencionTipo.REGISTRAR_ENTRADA:
        if not usuario_tiene_permiso_operativo(usuario, "movimientos.entrada"):
            msg = "No cuentas con el permiso requerido (movimientos.entrada) para registrar entradas de stock."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con los permisos requeridos para registrar entradas de stock."),
                "datos": {"error_permiso": True, "permiso_requerido": "movimientos.entrada"},
                "propuesta": None,
            }

        producto = intencion.producto
        from .agente.planificador import tiene_referencia_contextual_operacion
        if not producto and tiene_referencia_contextual_operacion(texto):
            if sesion and sesion.get("haku_ultimo_producto_id"):
                producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
            elif estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
                pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
                producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        from .lenguaje.observaciones import separar_observacion_operacion
        _, obs_ent_extraida = separar_observacion_operacion(texto or "")
        obs_ent_previa = str((intencion.datos_extra or {}).get("observacion") or "").strip()[:500]
        obs_ent_final = obs_ent_extraida or obs_ent_previa

        cantidad = intencion.cantidad

        if not producto:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "REGISTRAR_ENTRADA",
                    "esperando": "producto",
                    "cantidad": cantidad,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                    "motivo": intencion.motivo or "",
                    "observacion": obs_ent_final,
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = "Indique el nombre o código del producto para registrar la entrada."
            return {"tipo": "aclaracion", "esperando": "producto", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if not cantidad or cantidad <= 0:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "REGISTRAR_ENTRADA",
                    "esperando": "cantidad",
                    "producto_id": producto.id,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                    "motivo": intencion.motivo or "",
                    "observacion": obs_ent_final,
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"¿Cuántas unidades de {producto.nombre} va a ingresar?"
            return {"tipo": "aclaracion", "esperando": "cantidad", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        almacen, error_almacen = _resolver_almacen_para_operacion(
            usuario=usuario,
            almacen_intencion=intencion.almacen,
            producto=producto,
            operacion="entrada",
        )
        if error_almacen:
            return error_almacen

        if not usuario.tiene_acceso_almacen(almacen):
            raise PermissionDenied("No cuenta con acceso al almacén de este producto.")

        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(producto.id)
            sesion["haku_ultimo_acceso"] = time.time()

        motivo_ent = (intencion.motivo or "").strip() or "Entrada registrada por Haku"

        ahora = round(time.time(), 4)
        cid_res, thash_res = _resolver_contexto_contrato(sesion)
        propuesta = {
            "id": str(uuid.uuid4()),
            "tipo_operacion": Movimiento.Tipo.ENTRADA,
            "producto_id": producto.id,
            "producto_nombre": producto.nombre,
            "producto_codigo": producto.codigo_interno,
            "cantidad": cantidad,
            "almacen_origen_id": almacen.id,
            "almacen_origen_nombre": almacen.nombre,
            "destinatario": "",
            "motivo": motivo_ent,
            "observacion": obs_ent_final,
            "usuario_id": getattr(usuario, "id", None),
            "creado_en": ahora,
            "expira_en": round(ahora + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4),
            "version_contrato": VERSION_FIRMA_PROPUESTA,
            "conversacion_id": cid_res,
            "session_token_hash": thash_res,
        }
        calcular_huella_propuesta(propuesta)

        if estado_operativo:
            estado_operativo.operacion_pendiente = None
            estado_operativo.propuesta_activa = propuesta
            estado_operativo.registrar_accion(
                "PROPUESTA_ENTRADA",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
                cantidad=cantidad,
            )
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_propuesta_activa"] = propuesta
        HakuDiagnostico.registrar_propuesta("generada")

        u_ent = "unidad" if cantidad == 1 else "unidades"
        verbal = (
            f"Va a registrar la entrada de {cantidad} {u_ent} de {producto.nombre} "
            f"al almacén {almacen.nombre}"
            + (f", con observación: {obs_ent_final}." if obs_ent_final else ".")
            + " ¿Desea confirmar la operación?"
        )
        linea_obs_ent = f"- **Observación:** {obs_ent_final}\n" if obs_ent_final else ""
        pantalla = (
            f"### Propuesta pendiente de confirmación\n"
            f"- **Operación:** Entrada de stock\n"
            f"- **Producto:** {producto.nombre} (`{producto.codigo_interno}`)\n"
            f"- **Cantidad:** +{cantidad} {u_ent}\n"
            f"- **Almacén:** {almacen.nombre}\n"
            f"{linea_obs_ent}\n"
            f"*Di 'Confirmar' o presiona el botón para ejecutar la transacción.*"
        )
        return {
            "tipo": "propuesta",
            "requiere_confirmacion": True,
            "propuesta": propuesta,
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
        }

    # 7. Salida de inventario
    if intencion.tipo == IntencionTipo.REGISTRAR_SALIDA:
        motivo_temp = str(intencion.motivo or "").lower()
        es_baja = any(w in motivo_temp for w in ("baja", "daño", "avería", "averia", "pérdida", "perdida"))
        perm_salida = "movimientos.baja" if es_baja else "movimientos.dotacion"
        if not usuario_tiene_permiso_operativo(usuario, perm_salida):
            msg = f"No cuentas con el permiso requerido ({perm_salida}) para registrar salidas o dotaciones de stock."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con los permisos requeridos para registrar salidas o dotaciones de stock."),
                "datos": {"error_permiso": True, "permiso_requerido": perm_salida},
                "propuesta": None,
            }

        producto = intencion.producto
        from .agente.planificador import tiene_referencia_contextual_operacion
        if not producto and tiene_referencia_contextual_operacion(texto):
            if sesion and sesion.get("haku_ultimo_producto_id"):
                producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
            elif estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
                pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
                producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        from .lenguaje.observaciones import separar_observacion_operacion
        texto_sal_base, obs_sal_extraida = separar_observacion_operacion(texto or "")
        obs_sal_previa = str((intencion.datos_extra or {}).get("observacion") or "").strip()[:500]
        obs_sal_final = obs_sal_extraida or obs_sal_previa

        cantidad = intencion.cantidad
        destinatario = intencion.destinatario or ""
        if destinatario:
            dest_limpio_s, obs_dest_s = separar_observacion_operacion(destinatario)
            if obs_dest_s:
                destinatario = dest_limpio_s.strip(" ,;:-")
                if not obs_sal_final:
                    obs_sal_final = obs_dest_s
        if not destinatario and (texto_sal_base or texto):
            m_dest_sal = re.search(
                r"\b(?:para|a\s+nombre\s+de|entregar\s+a|entregado\s+a|destino)\s+"
                r"([A-Za-zÁ-ÿñÑ0-9\s'-]+?)"
                r"(?:\s+(?:por\s+motivo|con\s+motivo|motivo\s+|en\s+(?:el\s+)?almac[eé]n|del?\s+almac[eé]n)|[\.,;!?]|$)",
                texto_sal_base or texto,
                flags=re.IGNORECASE,
            )
            if m_dest_sal:
                cand_sal = m_dest_sal.group(1).strip('"\' \t\r\n.,;:!?')
                cand_sal = re.sub(r"^(?:el|la)\s+", "", cand_sal, flags=re.IGNORECASE).strip()
                if (
                    len(cand_sal) >= 2
                    and not re.search(r"\b(?:almac[eé]n|bodega|unidades?|piezas?|stock|inventario)\b", cand_sal, flags=re.IGNORECASE)
                ):
                    destinatario = cand_sal
        motivo = intencion.motivo or ""

        if not producto:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "REGISTRAR_SALIDA",
                    "esperando": "producto",
                    "cantidad": cantidad,
                    "destinatario": destinatario or "",
                    "motivo": motivo or "",
                    "observacion": obs_sal_final,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = "Por favor indique qué producto va a retirar del inventario."
            return {"tipo": "aclaracion", "esperando": "producto", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if not cantidad or cantidad <= 0:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "REGISTRAR_SALIDA",
                    "esperando": "cantidad",
                    "producto_id": producto.id,
                    "destinatario": destinatario or "",
                    "motivo": motivo or "",
                    "observacion": obs_sal_final,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"¿Cuántas unidades de {producto.nombre} va a retirar?"
            return {"tipo": "aclaracion", "esperando": "cantidad", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if not destinatario:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "REGISTRAR_SALIDA",
                    "esperando": "destinatario",
                    "producto_id": producto.id,
                    "cantidad": cantidad,
                    "motivo": motivo or "",
                    "observacion": obs_sal_final,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"Para registrar la salida de {producto.nombre}, ¿a quién se le entregan las unidades?"
            return {"tipo": "aclaracion", "esperando": "destinatario", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if not motivo:
            motivo = f"Salida registrada por Haku para {destinatario}"

        almacen, error_resp = _resolver_almacen_para_operacion(usuario, intencion.almacen, producto, operacion="salida")
        if error_resp:
            return error_resp

        existencia = Existencia.objects.filter(producto=producto, almacen=almacen).first()
        disponible = existencia.stock_disponible if existencia else 0
        if disponible < cantidad:
            if disponible > 0:
                if estado_operativo:
                    estado_operativo.operacion_pendiente = {
                        "tipo": "REGISTRAR_SALIDA",
                        "producto_id": producto.id,
                        "cantidad": disponible,
                        "destinatario": destinatario,
                        "motivo": motivo,
                        "observacion": obs_sal_final,
                        "almacen_id": almacen.id,
                        "esperando": "confirmar_menor_cantidad",
                    }
                    HakuOperationalContext.guardar(estado_operativo)
                msg_pantalla = (
                    f"Stock insuficiente: **{producto.nombre}** solo tiene **{disponible}** unidades disponibles en {almacen.nombre}.\n\n"
                    f"¿Desea registrar la salida por las **{disponible}** unidades disponibles?"
                )
                msg_verbal = normalizar_para_tts(
                    f"Stock insuficiente. {producto.nombre} solo tiene {disponible} unidades disponibles en {almacen.nombre}. ¿Desea retirar esas {disponible} unidades?"
                )
                return {
                    "tipo": "aclaracion",
                    "esperando": "confirmar_menor_cantidad",
                    "mensaje_pantalla": msg_pantalla,
                    "mensaje_verbal": msg_verbal,
                    "datos": {"disponible": disponible, "solicitado": cantidad},
                }
            else:
                if estado_operativo:
                    estado_operativo.operacion_pendiente = None
                    HakuOperationalContext.guardar(estado_operativo)
                msg = f"Stock insuficiente: {producto.nombre} no tiene existencias disponibles en {almacen.nombre if almacen else 'el almacén'}."
                return {"tipo": "error", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(producto.id)
            sesion["haku_ultimo_acceso"] = time.time()

        ahora = round(time.time(), 4)
        cid_res, thash_res = _resolver_contexto_contrato(sesion)
        propuesta = {
            "id": str(uuid.uuid4()),
            "tipo_operacion": Movimiento.Tipo.SALIDA,
            "producto_id": producto.id,
            "producto_nombre": producto.nombre,
            "producto_codigo": producto.codigo_interno,
            "cantidad": cantidad,
            "almacen_origen_id": almacen.id,
            "almacen_origen_nombre": almacen.nombre,
            "destinatario": destinatario,
            "motivo": motivo,
            "observacion": obs_sal_final,
            "usuario_id": getattr(usuario, "id", None),
            "creado_en": ahora,
            "expira_en": round(ahora + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4),
            "version_contrato": VERSION_FIRMA_PROPUESTA,
            "conversacion_id": cid_res,
            "session_token_hash": thash_res,
        }
        calcular_huella_propuesta(propuesta)

        if estado_operativo:
            estado_operativo.operacion_pendiente = None
            estado_operativo.propuesta_activa = propuesta
            estado_operativo.registrar_accion(
                "PROPUESTA_SALIDA",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
                cantidad=cantidad,
            )
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_propuesta_activa"] = propuesta
        HakuDiagnostico.registrar_propuesta("generada")

        es_dotacion = bool(
            motivo and ("dotación" in motivo.lower() or "dotacion" in motivo.lower() or "entrega definitiva" in motivo.lower())
        )
        tipo_op_label = "Dotación / Entrega definitiva (consumo oficial)" if es_dotacion else "Salida de stock"

        u_sal = "unidad" if cantidad == 1 else "unidades"
        verbal = (
            f"Va a registrar la dotación de {cantidad} {u_sal} de {producto.nombre} para {destinatario}. "
            f"Es una salida definitiva de inventario. ¿Desea confirmar la operación?"
            if es_dotacion else
            f"Va a registrar la salida de {cantidad} {u_sal} de {producto.nombre} para {destinatario}, "
            f"con motivo {motivo}. ¿Desea confirmar la operación?"
        )
        linea_obs_sal = f"- **Observación:** {obs_sal_final}\n" if obs_sal_final else ""
        pantalla = (
            f"### Propuesta pendiente de confirmación\n"
            f"- **Operación:** {tipo_op_label}\n"
            f"- **Producto:** {producto.nombre} (`{producto.codigo_interno}`)\n"
            f"- **Cantidad:** {cantidad} {u_sal}\n"
            f"- **Almacén:** {almacen.nombre}\n"
            f"- **Destinatario:** {destinatario}\n"
            f"- **Motivo:** {motivo}\n"
            f"{linea_obs_sal}\n"
            f"*Di 'Confirmar' o presiona el botón para ejecutar la transacción.*"
        )
        return {
            "tipo": "propuesta",
            "requiere_confirmacion": True,
            "propuesta": propuesta,
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
        }

    # 8. Devolución de préstamo
    if intencion.tipo == IntencionTipo.DEVOLUCION:
        if not usuario_tiene_permiso_operativo(usuario, "movimientos.devolucion"):
            msg = "No cuentas con el permiso requerido (movimientos.devolucion) para registrar devoluciones de préstamos."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con los permisos requeridos para registrar devoluciones."),
                "datos": {"error_permiso": True, "permiso_requerido": "movimientos.devolucion"},
                "propuesta": None,
            }

        destinatario = intencion.destinatario or ""
        qs_prestamos = Prestamo.objects.filter(
            estado__in=[Prestamo.Estado.ABIERTO, Prestamo.Estado.DEVUELTO_PARCIAL]
        ).select_related("producto", "almacen")
        if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
            alms_usuario = usuario.obtener_almacenes_accesibles()
            qs_prestamos = qs_prestamos.filter(almacen__in=alms_usuario)

        prestamo_especifico_id = (intencion.datos_extra or {}).get("prestamo_id")
        obs_previa_dev = str((intencion.datos_extra or {}).get("observacion") or "").strip()[:500]
        from .lenguaje.observaciones import separar_observacion_operacion
        texto_dev_base, obs_dev_extraida = separar_observacion_operacion(texto or "")
        obs_dev_opcional = obs_dev_extraida or obs_previa_dev
        if obs_dev_extraida and obs_previa_dev and obs_dev_extraida.lower() != obs_previa_dev.lower():
            obs_dev_opcional = f"{obs_previa_dev}; {obs_dev_extraida}"[:500]

        prestamo = None

        if prestamo_especifico_id:
            prestamo = qs_prestamos.filter(
                Q(id__istartswith=prestamo_especifico_id) |
                Q(movimiento__id__istartswith=prestamo_especifico_id)
            ).first()

        if not prestamo:
            from .servicios_personas import emparejar_nombre_contra_candidatos

            producto_dev = intencion.producto
            # Si el LLM devolvió DEVOLUCION sin extraer producto o destinatario, recuperarlos
            # dinámicamente usando el comando base sin observación (para que palabras de la observación
            # como "sin cable" o "lo revisó Juan" nunca contaminen el producto ni el responsable).
            if not producto_dev and texto:
                prod_res_dev = _buscar_producto(texto_dev_base, usuario) if texto_dev_base else None
                if not prod_res_dev:
                    prod_res_dev = _buscar_producto(texto, usuario)
                    if prod_res_dev and obs_dev_extraida and obs_dev_extraida.lower() in prod_res_dev.nombre.lower():
                        obs_dev_opcional = obs_previa_dev
                        texto_dev_base = texto
                if prod_res_dev:
                    producto_dev = prod_res_dev
                else:
                    # Si el nombre es parcial o ambiguo en el catálogo general (ej. "el taladro"),
                    # acotar la coincidencia estrictamente a los productos que tienen préstamos activos.
                    from .agente.herramientas import normalizar as _norm_dev, tokens_significativos as _toks_dev
                    stopwords_dev = {
                        "quiero", "necesito", "deseo", "devolver", "devolucion", "devolvio", "devuelve",
                        "registra", "registrar", "prestamo", "regresar", "regreso", "retornar", "reintegrar",
                        "trajo", "entrego", "producto", "articulo", "equipo", "unidad", "unidades",
                    }
                    toks_msg_dev = {
                        t for t in _toks_dev(texto_dev_base or texto)
                        if len(t) >= 4 and t not in stopwords_dev
                    }
                    if toks_msg_dev:
                        prods_prestados: dict[int, Any] = {}
                        for pr in qs_prestamos:
                            if pr.producto_id not in prods_prestados:
                                prods_prestados[pr.producto_id] = pr.producto
                        coincidentes_prest = []
                        for p_act in prods_prestados.values():
                            toks_p_act = set(_toks_dev(f"{p_act.nombre} {p_act.codigo_interno}"))
                            if toks_msg_dev & toks_p_act:
                                coincidentes_prest.append(p_act)
                        if len(coincidentes_prest) == 1:
                            producto_dev = coincidentes_prest[0]
                        elif len(coincidentes_prest) > 1:
                            lista_p = "\n".join(
                                f"- **{p_c.nombre}** (`{p_c.codigo_interno}`)" for p_c in coincidentes_prest[:5]
                            )
                            msg_p = (
                                f"Encontré varios productos con préstamos activos que coinciden:\n{lista_p}\n\n"
                                "¿Cuál de ellos desea devolver?"
                            )
                            if estado_operativo:
                                estado_operativo.operacion_pendiente = {
                                    "tipo": "DEVOLUCION",
                                    "esperando": "producto",
                                    "destinatario": destinatario or "",
                                    "cantidad": intencion.cantidad,
                                    "observacion": obs_dev_opcional,
                                }
                                HakuOperationalContext.guardar(estado_operativo)
                            return {
                                "tipo": "aclaracion",
                                "esperando": "producto",
                                "mensaje_pantalla": msg_p,
                                "mensaje_verbal": normalizar_para_tts(
                                    f"Encontré varios productos en préstamo que coinciden: {', '.join(p_c.nombre for p_c in coincidentes_prest[:3])}. ¿Cuál desea devolver?"
                                ),
                            }

            if not destinatario and texto_dev_base:
                qs_base_cands = qs_prestamos.filter(producto=producto_dev) if producto_dev else qs_prestamos
                cands_pre = list({p.destinatario for p in qs_base_cands if p.destinatario})
                match_fon_pre = emparejar_nombre_contra_candidatos(texto_dev_base, cands_pre)
                if match_fon_pre:
                    destinatario = match_fon_pre
                else:
                    # Si la frase nombra explícitamente un sujeto antes del verbo de devolución
                    # (ej. "Yan Carlos quiere devolver...") y no empató con ningún préstamo activo,
                    # conservar ese nombre como destinatario para evitar asignar el préstamo de otra persona.
                    m_suj_dev = re.match(
                        r"^\s*([A-Za-zÁ-ÿñÑ]+(?:\s+[A-Za-zÁ-ÿñÑ]+){0,3})\s+"
                        r"(?:ya\s+)?(?:quiere|va\s+a|viene\s+a|necesita|desea|devolvi[oó]|regres[oó]|retorn[oó]|reintegr[oó]|trajo|entreg[oó])\b",
                        texto_dev_base.strip(),
                        flags=re.IGNORECASE,
                    )
                    if m_suj_dev:
                        suj_cand = m_suj_dev.group(1).strip()
                        if suj_cand.lower() not in (
                            "yo", "el", "la", "se", "me", "te", "nos", "hoy", "ya", "quiero", "voy",
                            "vengo", "necesito", "deseo", "por favor", "haku", "hacu",
                        ):
                            destinatario = suj_cand

            if producto_dev and destinatario:
                qs_match = qs_prestamos.filter(producto=producto_dev, destinatario__icontains=destinatario)
                if not qs_match.exists():
                    qs_prod_all = qs_prestamos.filter(producto=producto_dev)
                    cands_prod = [p.destinatario for p in qs_prod_all if p.destinatario]
                    match_fon = emparejar_nombre_contra_candidatos(destinatario, cands_prod)
                    if match_fon:
                        destinatario = match_fon
                        qs_match = qs_prod_all.filter(destinatario__iexact=match_fon)
                if not qs_match.exists():
                    # Solo emparejar por tokens si TODOS los tokens significativos del nombre aparecen en el prestatario
                    # (nunca descartar "Yan" de "Yan Carlos" para empatar erróneamente con "Profesor Carlos Mendoza").
                    toks_dest = [w for w in destinatario.split() if len(w) >= 2]
                    if toks_dest:
                        q_todos = qs_prestamos.filter(producto=producto_dev)
                        for w in toks_dest:
                            q_todos = q_todos.filter(destinatario__icontains=w)
                        qs_match = q_todos
                if qs_match.count() == 1:
                    prestamo = qs_match.first()
                elif qs_match.count() > 1:
                    # Si el usuario especificó una cantidad exacta que coincide con solo uno de los préstamos
                    if intencion.cantidad and qs_match.filter(cantidad_pendiente_devolucion=intencion.cantidad).count() == 1:
                        prestamo = qs_match.filter(cantidad_pendiente_devolucion=intencion.cantidad).first()
                    else:
                        lista_prods = [
                            f"Préstamo #{str(p.id)[:8]} ({p.cantidad_pendiente_devolucion} uds pendientes en {p.almacen.nombre})"
                            for p in qs_match[:5]
                        ]
                        msg_pantalla = (
                            f"**{destinatario}** tiene **{qs_match.count()}** préstamos activos distintos de **{producto_dev.nombre}**:\n"
                            + "\n".join(f"- {lp}" for lp in lista_prods)
                            + f"\n\n¿Cuál de estos préstamos desea devolver? (Indique la cantidad o la referencia)."
                        )
                        msg_verbal = normalizar_para_tts(
                            f"{destinatario} tiene {qs_match.count()} préstamos activos de {producto_dev.nombre}. ¿Cuál desea devolver?"
                        )
                        if estado_operativo:
                            estado_operativo.operacion_pendiente = {
                                "tipo": "DEVOLUCION",
                                "producto_id": producto_dev.id,
                                "destinatario": destinatario,
                                "cantidad": intencion.cantidad,
                                "esperando": "referencia_prestamo",
                                "observacion": obs_dev_opcional,
                            }
                            HakuOperationalContext.guardar(estado_operativo)
                        return {
                            "tipo": "aclaracion",
                            "esperando": "referencia_prestamo",
                            "mensaje_pantalla": msg_pantalla,
                            "mensaje_verbal": msg_verbal,
                            "datos": {"requiere_referencia": True, "candidatos": [str(p.id)[:8] for p in qs_match]},
                        }
                else:
                    msg = f"No se encontró ningún préstamo activo de {producto_dev.nombre} a nombre de {destinatario}."
                    return {"tipo": "conversacion", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

            elif producto_dev:
                qs_prod = qs_prestamos.filter(producto=producto_dev)
                cant_prestamos = qs_prod.count()
                if cant_prestamos == 0:
                    msg = f"No hay préstamos activos del producto {producto_dev.nombre} para devolver."
                    return {"tipo": "conversacion", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}
                elif cant_prestamos == 1:
                    prestamo = qs_prod.first()
                else:
                    cands_dest_list = [p.destinatario for p in qs_prod]
                    lista_dest = [f"**{p.destinatario}** ({p.cantidad_pendiente_devolucion} uds en {p.almacen.nombre})" for p in qs_prod[:5]]
                    msg_pantalla = (
                        f"El producto **{producto_dev.nombre}** tiene varios préstamos activos:\n"
                        + "\n".join(f"- {d}" for d in lista_dest)
                        + f"\n\n¿A nombre de cuál persona desea registrar la devolución?"
                    )
                    nombres_v = [p.destinatario for p in qs_prod[:3]]
                    msg_verbal = normalizar_para_tts(
                        f"El producto {producto_dev.nombre} está prestado a varias personas, entre ellas {', '.join(nombres_v)}. ¿Quién realiza la devolución?"
                    )
                    if estado_operativo:
                        estado_operativo.operacion_pendiente = {
                            "tipo": "DEVOLUCION",
                            "producto_id": producto_dev.id,
                            "cantidad": intencion.cantidad,
                            "esperando": "destinatario",
                            "candidatos_destinatario": cands_dest_list,
                            "observacion": obs_dev_opcional,
                        }
                        HakuOperationalContext.guardar(estado_operativo)
                    return {
                        "tipo": "aclaracion",
                        "esperando": "destinatario",
                        "mensaje_pantalla": msg_pantalla,
                        "mensaje_verbal": msg_verbal,
                        "datos": {"requiere_destinatario": True, "candidatos": cands_dest_list},
                    }

            elif destinatario:
                qs_dest = qs_prestamos.filter(destinatario__icontains=destinatario)
                if not qs_dest.exists():
                    cands_all_dest = list({p.destinatario for p in qs_prestamos if p.destinatario})
                    match_all = emparejar_nombre_contra_candidatos(destinatario, cands_all_dest)
                    if match_all:
                        destinatario = match_all
                        qs_dest = qs_prestamos.filter(destinatario__iexact=match_all)
                if not qs_dest.exists():
                    toks_d = [w for w in destinatario.split() if len(w) >= 2]
                    if toks_d:
                        q_todos_d = qs_prestamos
                        for w in toks_d:
                            q_todos_d = q_todos_d.filter(destinatario__icontains=w)
                        qs_dest = q_todos_d
                cant_dest = qs_dest.count()
                if cant_dest == 0:
                    msg = f"No encontré ningún préstamo activo a nombre de '{destinatario}' para devolver."
                    return {"tipo": "conversacion", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}
                elif cant_dest == 1:
                    prestamo = qs_dest.first()
                else:
                    lista_prods = [f"**{p.producto.nombre}** (`{p.producto.codigo_interno}`) — {p.cantidad_pendiente_devolucion} uds ({p.almacen.nombre})" for p in qs_dest[:5]]
                    msg_pantalla = (
                        f"**{destinatario}** tiene varios préstamos activos pendientes:\n"
                        + "\n".join(f"- {p}" for p in lista_prods)
                        + f"\n\n¿Cuál de estos productos desea devolver?"
                    )
                    prods_v = [p.producto.nombre for p in qs_dest[:3]]
                    msg_verbal = normalizar_para_tts(
                        f"{destinatario} tiene varios préstamos activos: {', '.join(prods_v)}. ¿Cuál desea devolver?"
                    )
                    if estado_operativo:
                        estado_operativo.operacion_pendiente = {
                            "tipo": "DEVOLUCION",
                            "destinatario": destinatario,
                            "cantidad": intencion.cantidad,
                            "esperando": "producto",
                            "observacion": obs_dev_opcional,
                        }
                        HakuOperationalContext.guardar(estado_operativo)
                    return {
                        "tipo": "aclaracion",
                        "esperando": "producto",
                        "mensaje_pantalla": msg_pantalla,
                        "mensaje_verbal": msg_verbal,
                        "datos": {"requiere_producto": True, "candidatos": [{"id": p.producto.id, "nombre": p.producto.nombre} for p in qs_dest]},
                    }

            else:
                total_activos = qs_prestamos.count()
                if total_activos == 0:
                    msg = "No hay préstamos activos en el inventario para registrar devoluciones."
                    return {"tipo": "conversacion", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}
                elif total_activos == 1:
                    prestamo = qs_prestamos.first()
                else:
                    msg = "Por favor indique qué producto o a nombre de qué persona desea registrar la devolución."
                    if estado_operativo:
                        estado_operativo.operacion_pendiente = {
                            "tipo": "DEVOLUCION",
                            "cantidad": intencion.cantidad,
                            "esperando": "identificacion",
                            "observacion": obs_dev_opcional,
                        }
                        HakuOperationalContext.guardar(estado_operativo)
                    return {
                        "tipo": "aclaracion",
                        "esperando": "identificacion",
                        "mensaje_pantalla": msg,
                        "mensaje_verbal": normalizar_para_tts(msg),
                        "datos": {"requiere_identificacion": True},
                    }

        if not prestamo:
            msg = "No encontré ningún préstamo activo con los datos suministrados para devolver."
            return {"tipo": "conversacion", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        es_solicitud_parcial = bool(
            re.search(r"\b(parte|parcial|parcialmente|una\s+parte)\b", texto_lower)
            and not re.search(r"\b(tod[oa]s?|completo|completa|total)\b", texto_lower)
        )

        if es_solicitud_parcial and (not intencion.cantidad or intencion.cantidad <= 0):
            msg_pantalla = (
                f"**{prestamo.destinatario}** tiene **{prestamo.cantidad_pendiente_devolucion}** unidades pendientes de **{prestamo.producto.nombre}** ({prestamo.almacen.nombre}).\n\n"
                f"¿Cuántas unidades desea devolver?"
            )
            msg_verbal = normalizar_para_tts(
                f"{prestamo.destinatario} tiene {prestamo.cantidad_pendiente_devolucion} unidades pendientes de {prestamo.producto.nombre}. ¿Cuántas unidades desea devolver?"
            )
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "DEVOLUCION",
                    "prestamo_id": str(prestamo.id),
                    "producto_id": prestamo.producto.id,
                    "destinatario": prestamo.destinatario,
                    "cantidad_maxima": prestamo.cantidad_pendiente_devolucion,
                    "almacen_id": prestamo.almacen.id,
                    "esperando": "cantidad",
                    "observacion": obs_dev_opcional,
                }
                HakuOperationalContext.guardar(estado_operativo)
            return {
                "tipo": "aclaracion",
                "mensaje_pantalla": msg_pantalla,
                "mensaje_verbal": msg_verbal,
                "datos": {"requiere_cantidad": True, "saldo_pendiente": prestamo.cantidad_pendiente_devolucion},
            }

        if intencion.cantidad and intencion.cantidad > 0:
            cantidad_dev = intencion.cantidad
        else:
            cantidad_dev = prestamo.cantidad_pendiente_devolucion

        if cantidad_dev > prestamo.cantidad_pendiente_devolucion:
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"La cantidad a devolver ({cantidad_dev}) excede el saldo pendiente del préstamo ({prestamo.cantidad_pendiente_devolucion} unidades)."
            return {"tipo": "error", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(prestamo.producto.id)
            sesion["haku_ultimo_acceso"] = time.time()

        ahora = round(time.time(), 4)
        cid_res, thash_res = _resolver_contexto_contrato(sesion)
        propuesta = {
            "id": str(uuid.uuid4()),
            "tipo_operacion": Movimiento.Tipo.DEVOLUCION,
            "prestamo_id": str(prestamo.id),
            "producto_id": prestamo.producto.id,
            "producto_nombre": prestamo.producto.nombre,
            "producto_codigo": prestamo.producto.codigo_interno,
            "cantidad": cantidad_dev,
            "almacen_origen_id": prestamo.almacen.id,
            "almacen_origen_nombre": prestamo.almacen.nombre,
            "destinatario": prestamo.destinatario,
            "motivo": f"Devolución de préstamo por {prestamo.destinatario}",
            "observacion": obs_dev_opcional,
            "usuario_id": getattr(usuario, "id", None),
            "creado_en": ahora,
            "expira_en": round(ahora + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4),
            "version_contrato": VERSION_FIRMA_PROPUESTA,
            "conversacion_id": cid_res,
            "session_token_hash": thash_res,
        }
        calcular_huella_propuesta(propuesta)

        if estado_operativo:
            estado_operativo.operacion_pendiente = None
            estado_operativo.propuesta_activa = propuesta
            estado_operativo.registrar_accion(
                "PROPUESTA_DEVOLUCION",
                producto=prestamo.producto.nombre,
                producto_id=prestamo.producto.id,
                codigo=prestamo.producto.codigo_interno,
                cantidad=cantidad_dev,
            )
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_propuesta_activa"] = propuesta
        HakuDiagnostico.registrar_propuesta("generada")

        verbal = (
            f"Va a registrar la devolución de {cantidad_dev} unidades de {prestamo.producto.nombre} "
            f"prestadas a {prestamo.destinatario}"
            + (f", con observación: {obs_dev_opcional}." if obs_dev_opcional else ".")
            + " ¿Desea confirmar?"
        )
        linea_obs_pantalla = f"- **Observación:** {obs_dev_opcional}\n" if obs_dev_opcional else ""
        pantalla = (
            f"### Propuesta pendiente de confirmación\n"
            f"- **Operación:** Devolución de préstamo\n"
            f"- **Producto:** {prestamo.producto.nombre} (`{prestamo.producto.codigo_interno}`)\n"
            f"- **Cantidad devuelta:** +{cantidad_dev} unidades\n"
            f"- **Responsable:** {prestamo.destinatario}\n"
            f"- **Almacén:** {prestamo.almacen.nombre}\n"
            f"{linea_obs_pantalla}\n"
            f"*Di 'Confirmar' o presiona el botón para ejecutar la transacción.*"
        )
        return {
            "tipo": "propuesta",
            "requiere_confirmacion": True,
            "propuesta": propuesta,
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
        }

    # 9. Préstamo
    if intencion.tipo == IntencionTipo.PRESTAMO:
        if not usuario_tiene_permiso_operativo(usuario, "movimientos.prestamo"):
            msg = "No cuentas con el permiso requerido (movimientos.prestamo) para registrar préstamos de herramientas o artículos."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con los permisos requeridos para registrar préstamos."),
                "datos": {"error_permiso": True, "permiso_requerido": "movimientos.prestamo"},
                "propuesta": None,
            }

        producto = intencion.producto
        from .agente.planificador import tiene_referencia_contextual_operacion
        if not producto and tiene_referencia_contextual_operacion(texto):
            if sesion and sesion.get("haku_ultimo_producto_id"):
                producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
            elif estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
                pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
                producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        from .lenguaje.observaciones import separar_observacion_operacion
        texto_pres_base, obs_pres_extraida = separar_observacion_operacion(texto or "")
        obs_pres_previa = str((intencion.datos_extra or {}).get("observacion") or "").strip()[:500]
        obs_pres_final = obs_pres_extraida or obs_pres_previa

        cantidad = intencion.cantidad
        destinatario = intencion.destinatario or ""
        if destinatario:
            dest_limpio_p, obs_dest_p = separar_observacion_operacion(destinatario)
            if obs_dest_p:
                destinatario = dest_limpio_p.strip(" ,;:-")
                if not obs_pres_final:
                    obs_pres_final = obs_dest_p
        if not destinatario and (texto_pres_base or texto):
            base_pr = (texto_pres_base or texto).strip()
            m_dest_pr = re.search(
                r"\b(?:para|a\s+nombre\s+de|entregar\s+a)\s+"
                r"([A-Za-zÁ-ÿñÑ\s'-]{2,60}?)"
                r"(?:\s+(?:por\s+motivo|con\s+motivo|motivo\s+|en\s+(?:el\s+)?almac[eé]n|del?\s+almac[eé]n)|[\.,;!?]|$)",
                base_pr,
                flags=re.IGNORECASE,
            ) or re.search(
                r"\s+a\s+([A-Za-zÁ-ÿñÑ]+(?:\s+[A-Za-zÁ-ÿñÑ]+){0,3})\s*$",
                base_pr.rstrip(".,;:!? "),
                flags=re.IGNORECASE,
            )
            if m_dest_pr:
                cand_pr = m_dest_pr.group(1).strip('"\' \t\r\n.,;:!?')
                cand_pr = re.sub(r"^(?:el|la)\s+", "", cand_pr, flags=re.IGNORECASE).strip()
                if len(cand_pr) >= 2 and not re.search(r"\b(?:almac[eé]n|bodega|unidades?|piezas?|stock|inventario)\b", cand_pr, flags=re.IGNORECASE):
                    destinatario = cand_pr

        if not producto:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "PRESTAMO",
                    "esperando": "producto",
                    "cantidad": cantidad,
                    "destinatario": destinatario or "",
                    "observacion": obs_pres_final,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = "Por favor indique qué producto desea prestar."
            return {"tipo": "aclaracion", "esperando": "producto", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if not getattr(producto, "es_prestable", False):
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"El producto {producto.nombre} es un artículo consumible y no admite préstamos. Solo puede registrarse como salida de inventario."
            return {"tipo": "error", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        if not cantidad or cantidad <= 0:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "PRESTAMO",
                    "esperando": "cantidad",
                    "producto_id": producto.id,
                    "destinatario": destinatario or "",
                    "observacion": obs_pres_final,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"¿Cuántas unidades de {producto.nombre} se van a prestar?"
            return {"tipo": "aclaracion", "esperando": "cantidad", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        # Validación temprana de stock cuando el almacén ya es inequívoco:
        # evitar pedir aclaración de destinatario si físicamente no hay unidades suficientes.
        almacen_pre, error_resp_pre = _resolver_almacen_para_operacion(usuario, intencion.almacen, producto, operacion="préstamo")
        if almacen_pre is not None:
            existencia_pre = Existencia.objects.filter(producto=producto, almacen=almacen_pre).first()
            disponible_pre = existencia_pre.stock_disponible if existencia_pre else 0
            if disponible_pre < cantidad:
                if estado_operativo:
                    estado_operativo.operacion_pendiente = None
                    HakuOperationalContext.guardar(estado_operativo)
                msg = f"Stock insuficiente para préstamo: {producto.nombre} solo tiene {disponible_pre} disponibles."
                return {"tipo": "error", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        # Validación y normalización robusta del destinatario
        from .servicios_personas import HakuPersonasService
        
        # Determinar si el origen fue voz
        origen_voz = bool(sesion and sesion.get("haku_ultimo_origen") == "voz")

        nombre_canonico, confianza, candidatos_similares, requiere_escritura = HakuPersonasService.resolver_destinatario(
            destinatario, origen_voz=origen_voz
        )

        if not destinatario or requiere_escritura or not nombre_canonico:
            unidad_solicitada = "1 unidad" if cantidad == 1 else f"{cantidad} unidades"
            verbo_prestar = "prestará" if cantidad == 1 else "prestarán"
            if not destinatario:
                msg = (
                    f"¿A quién se le {verbo_prestar} {unidad_solicitada} de {producto.nombre}? Escribe el nombre de la persona para registrarlo correctamente."
                )
            elif candidatos_similares and len(candidatos_similares) > 1:
                cand_str = ", ".join(candidatos_similares[:3])
                msg = (
                    f"Encontré varias personas con nombres similares ({cand_str}). "
                    "Por favor escribe el nombre completo de la persona para evitar registrar a otra persona por error."
                )
            else:
                msg = (
                    "No pude identificar con seguridad el nombre. Escríbelo para evitar registrar a otra persona por error."
                )

            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "PRESTAMO",
                    "esperando": "destinatario",
                    "producto_id": producto.id,
                    "cantidad": cantidad,
                    "observacion": obs_pres_final,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                }
                HakuOperationalContext.guardar(estado_operativo)
            return {
                "tipo": "aclaracion",
                "esperando": "destinatario",
                "requiere_escritura": True,
                "candidatos": candidatos_similares,
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(msg),
                "datos": {
                    "esperando": "destinatario",
                    "requiere_escritura": True,
                    "candidatos": candidatos_similares,
                },
            }

        # Asignar el nombre canónico oficial verificado
        destinatario = nombre_canonico

        almacen, error_resp = _resolver_almacen_para_operacion(usuario, intencion.almacen, producto, operacion="préstamo")
        if error_resp:
            return error_resp

        existencia = Existencia.objects.filter(producto=producto, almacen=almacen).first()
        disponible = existencia.stock_disponible if existencia else 0
        if disponible < cantidad:
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"Stock insuficiente para préstamo: {producto.nombre} solo tiene {disponible} disponibles."
            return {"tipo": "error", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        if sesion is not None:
            sesion["haku_ultimo_producto_id"] = str(producto.id)
            sesion["haku_ultimo_acceso"] = time.time()

        ahora = round(time.time(), 4)
        cid_res, thash_res = _resolver_contexto_contrato(sesion)
        propuesta = {
            "id": str(uuid.uuid4()),
            "tipo_operacion": Movimiento.Tipo.PRESTAMO,
            "producto_id": producto.id,
            "producto_nombre": producto.nombre,
            "producto_codigo": producto.codigo_interno,
            "cantidad": cantidad,
            "almacen_origen_id": almacen.id,
            "almacen_origen_nombre": almacen.nombre,
            "destinatario": destinatario,
            "motivo": f"Préstamo registrado por Haku a {destinatario}",
            "observacion": obs_pres_final,
            "usuario_id": getattr(usuario, "id", None),
            "creado_en": ahora,
            "expira_en": round(ahora + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4),
            "version_contrato": VERSION_FIRMA_PROPUESTA,
            "conversacion_id": cid_res,
            "session_token_hash": thash_res,
        }
        calcular_huella_propuesta(propuesta)

        if estado_operativo:
            estado_operativo.propuesta_activa = propuesta
            estado_operativo.operacion_pendiente = None
            estado_operativo.registrar_accion(
                "PROPUESTA_PRESTAMO",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
                cantidad=cantidad,
            )
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_propuesta_activa"] = propuesta
        HakuDiagnostico.registrar_propuesta("generada")

        unidad_txt = "1 unidad" if cantidad == 1 else f"{cantidad} unidades"
        verbal = (
            f"Va a prestar {unidad_txt} de {producto.nombre} a {destinatario}. "
            f"Quedará registrada como préstamo pendiente. ¿Desea confirmar?"
            if cantidad == 1 else
            f"Va a prestar {cantidad} unidades de {producto.nombre} a {destinatario}. "
            f"Quedarán registradas como préstamo pendiente. ¿Desea confirmar?"
        )
        linea_obs_pres = f"- **Observación:** {obs_pres_final}\n" if obs_pres_final else ""
        pantalla = (
            f"### Propuesta pendiente de confirmación\n"
            f"- **Operación:** Préstamo de stock\n"
            f"- **Producto:** {producto.nombre} (`{producto.codigo_interno}`)\n"
            f"- **Cantidad:** {unidad_txt}\n"
            f"- **Almacén:** {almacen.nombre}\n"
            f"- **Responsable/Destinatario:** {destinatario}\n"
            f"{linea_obs_pres}\n"
            f"*Di 'Confirmar' o presiona el botón para ejecutar la transacción.*"
        )
        return {
            "tipo": "propuesta",
            "requiere_confirmacion": True,
            "propuesta": propuesta,
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
        }

    # 10. Creación de producto con formulario interactivo
    # 9-bis. Creación de usuario con formulario interactivo
    if intencion.tipo == IntencionTipo.CREAR_USUARIO:
        if usuario.es_usuario_normal:
            raise PermissionDenied("No cuenta con permisos para administrar usuarios. Contacte al administrador.")

        almacenes_qs = Almacen.objects.filter(activo=True, eliminado=False).values("id", "nombre")
        # Filtrar roles permitidos: Administrador solo puede crear USUARIO; Superusuario puede crear ADMINISTRADOR, USUARIO y SUPERUSUARIO
        roles_disponibles = [{"codigo": "USUARIO", "valor": "USUARIO", "nombre": "Usuario (Operador de almacén)"}]
        if usuario.es_superusuario_inventario:
            roles_disponibles.insert(0, {"codigo": "ADMINISTRADOR", "valor": "ADMINISTRADOR", "nombre": "Administrador"})
            roles_disponibles.insert(0, {"codigo": "SUPERUSUARIO", "valor": "SUPERUSUARIO", "nombre": "Superusuario (Control total)"})

        return {
            "tipo": "formulario_crear_usuario",
            "mensaje_pantalla": "### Registro de Nuevo Usuario\nCompleta los datos requeridos en el formulario a continuación.",
            "mensaje_verbal": normalizar_para_tts("Abriendo formulario para registrar un nuevo usuario."),
            "datos": {
                "roles": roles_disponibles,
                "almacenes": list(almacenes_qs),
                "es_superusuario": usuario.es_superusuario_inventario,
                "usuario_actor_rol": usuario.rol,
            },
        }

    # 9-ter. Edición de usuario con formulario interactivo
    if intencion.tipo == IntencionTipo.EDITAR_USUARIO:
        if usuario.es_usuario_normal:
            raise PermissionDenied("No cuenta con permisos para modificar usuarios. Contacte al administrador.")

        almacenes_qs = Almacen.objects.filter(activo=True, eliminado=False).values("id", "nombre")
        roles_disponibles = [{"codigo": "USUARIO", "valor": "USUARIO", "nombre": "Usuario (Operador de almacén)"}]
        if usuario.es_superusuario_inventario:
            roles_disponibles.insert(0, {"codigo": "ADMINISTRADOR", "valor": "ADMINISTRADOR", "nombre": "Administrador"})
            roles_disponibles.insert(0, {"codigo": "SUPERUSUARIO", "valor": "SUPERUSUARIO", "nombre": "Superusuario (Control total)"})

        # Lista de usuarios que el solicitante puede editar
        from cuentas.models import UsuarioAudacIA
        qs_usuarios = UsuarioAudacIA.objects.filter(eliminado_logicamente=False).order_by("username")
        if not usuario.es_superusuario_inventario:
            # Administrador no puede editar Superusuarios ni elevar privilegios
            qs_usuarios = qs_usuarios.exclude(rol=UsuarioAudacIA.Rol.SUPERUSUARIO)

        usuarios_lista = [
            {
                "id": u.id,
                "username": u.username,
                "nombre_completo": u.nombre_completo,
                "rol": u.rol,
                "almacen_asignado_id": u.almacen_asignado_id,
                "almacenes_asignados_ids": list(u.almacenes_asignados.values_list("id", flat=True)),
                "acceso_todos_almacenes": u.acceso_todos_almacenes,
                "is_active": u.is_active,
            }
            for u in qs_usuarios
        ]

        return {
            "tipo": "formulario_editar_usuario",
            "mensaje_pantalla": "### Gestión de Usuario\nSelecciona y modifica el usuario en el formulario a continuación.",
            "mensaje_verbal": normalizar_para_tts("Abriendo formulario para editar usuarios."),
            "datos": {
                "usuarios": usuarios_lista,
                "roles": roles_disponibles,
                "almacenes": list(almacenes_qs),
                "es_superusuario": usuario.es_superusuario_inventario,
            },
        }

    if intencion.tipo == IntencionTipo.CREAR_PRODUCTO:
        return _procesar_alta_producto_conversacional(
            texto=texto,
            usuario=usuario,
            estado_operativo=estado_operativo,
            sesion=sesion,
            semantica=semantica,
            op_pendiente=op_pendiente,
            conversacion_id=conversacion_id,
        )

    # 11. Edición de producto con formulario interactivo
    if intencion.tipo == IntencionTipo.EDITAR_PRODUCTO:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.editar"):
            raise PermissionDenied("No cuentas con el permiso requerido (catalogo.editar) para modificar productos.")

        producto = intencion.producto
        from .agente.planificador import tiene_referencia_contextual_operacion
        if not producto and tiene_referencia_contextual_operacion(texto):
            if sesion and sesion.get("haku_ultimo_producto_id"):
                producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
            elif estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
                pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
                producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        if not producto:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "EDITAR_PRODUCTO",
                    "esperando": "producto",
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = "¿Cuál producto desea editar o modificar?"
            return {"tipo": "aclaracion", "esperando": "producto", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
            alms_usuario = set(usuario.obtener_almacenes_accesibles().values_list("id", flat=True))
            tiene_acceso = producto.existencias.filter(almacen_id__in=alms_usuario).exists() or (
                producto.categoria and producto.categoria.almacen_principal_id in alms_usuario
            )
            if not tiene_acceso:
                raise PermissionDenied("No cuenta con acceso al almacén de este producto.")

        valores_propuestos = dict((intencion.datos_extra or {}).get("valores_propuestos") or {})
        if not valores_propuestos.get("nombre"):
            m_renom = re.search(
                r"\b(?:cambia(?:r)?|renombra(?:r)?|actualiza(?:r)?|edita(?:r)?|modifica(?:r)?)\s+"
                r"(?:el\s+)?nombre\s+(?:de|del)\s+.+?\s+(?:a|por)\s+(.+)",
                texto or "",
                flags=re.IGNORECASE,
            )
            if m_renom:
                nuevo_nom = m_renom.group(1).strip('"\' \t\r\n.,;:!?¿¡')
                if len(nuevo_nom) >= 2:
                    valores_propuestos["nombre"] = nuevo_nom

        if "es_prestable" not in valores_propuestos:
            es_prest_edit = _extraer_es_prestable_en_texto(texto or "")
            if es_prest_edit is not None:
                valores_propuestos["es_prestable"] = es_prest_edit

        prod_dict = {
            "id": producto.id,
            "codigo_interno": producto.codigo_interno,
            "nombre": producto.nombre,
            "codigo_barras": producto.codigo_barras or "",
            "categoria_id": producto.categoria_id,
            "categoria_nombre": producto.categoria.nombre if producto.categoria else "",
            "stock_minimo": producto.stock_minimo,
            "es_prestable": bool(producto.es_prestable),
            "nombre_empaque": producto.nombre_empaque or "",
            "unidades_por_empaque": producto.unidades_por_empaque or "",
            "descripcion": producto.descripcion or "",
            "activo": producto.activo,
        }
        if valores_propuestos:
            for k, v in valores_propuestos.items():
                if k in prod_dict:
                    prod_dict[k] = v

        cats_qs_edit = Categoria.objects.filter(activo=True, eliminado=False)
        if producto.categoria_id and producto.categoria.almacen_principal_id:
            cats_qs_edit = cats_qs_edit.filter(almacen_principal_id=producto.categoria.almacen_principal_id)
        categorias_list = [
            {
                "id": c.id,
                "nombre": c.nombre,
                "prefijo_codigo": c.prefijo_codigo,
            }
            for c in cats_qs_edit.order_by("nombre")
        ]
        res_payload = {
            "tipo": "formulario_editar_producto",
            "mensaje_pantalla": f"### Edición de Producto: {producto.nombre} (`{producto.codigo_interno}`)\nModifica los campos permitidos y confirma los cambios.",
            "mensaje_verbal": normalizar_para_tts(f"Abriendo formulario para editar {producto.nombre}."),
            "datos": {
                "producto": prod_dict,
                "categorias": categorias_list,
            },
        }
        if valores_propuestos:
            res_payload["datos"]["valores_propuestos"] = valores_propuestos
        return res_payload

    # 12. Inhabilitación de producto
    # ─────────────────────────────────────────────────────────────────────────
    # C-6. TRASLADO ENTRE ALMACENES (TRASLADAR_PRODUCTO)
    # ─────────────────────────────────────────────────────────────────────────
    if intencion.tipo == IntencionTipo.TRASLADAR_PRODUCTO:
        if not usuario_tiene_permiso_operativo(usuario, "movimientos.traslado_almacen"):
            msg = "No cuentas con el permiso requerido (movimientos.traslado_almacen) para registrar traslados entre almacenes."
            return {
                "tipo": "error_permiso",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts("No cuenta con los permisos requeridos para registrar traslados de almacén."),
                "datos": {"error_permiso": True, "permiso_requerido": "movimientos.traslado_almacen"},
                "propuesta": None,
            }

        producto = intencion.producto
        from .agente.planificador import tiene_referencia_contextual_operacion
        if not producto and tiene_referencia_contextual_operacion(texto):
            if sesion and sesion.get("haku_ultimo_producto_id"):
                producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
            elif estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
                pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
                producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        from .lenguaje.observaciones import separar_observacion_operacion
        _, obs_tr_extraida = separar_observacion_operacion(texto or "")
        obs_tr_previa = str((intencion.datos_extra or {}).get("observacion") or "").strip()[:500]
        obs_tr_final = obs_tr_extraida or obs_tr_previa

        if not producto:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "TRASLADO_ENTRE",
                    "esperando": "producto",
                    "cantidad": intencion.cantidad,
                    "almacen_id": intencion.almacen.id if intencion.almacen else None,
                    "almacen_destino_id": (intencion.datos_extra or {}).get("almacen_destino_id"),
                    "almacen_destino_nombre": (intencion.datos_extra or {}).get("almacen_destino_nombre"),
                    "observacion": obs_tr_final,
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = "Indique el nombre o código del producto que desea trasladar entre almacenes."
            return {"tipo": "aclaracion", "esperando": "producto", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        almacen_origen, acceso_msg = _validar_acceso_almacen(usuario, intencion.almacen)
        if acceso_msg:
            if isinstance(acceso_msg, dict):
                return acceso_msg
            return {"tipo": "error", "mensaje_pantalla": acceso_msg, "mensaje_verbal": normalizar_para_tts(acceso_msg)}

        # Determinar almacén destino
        almacen_destino_id = (intencion.datos_extra or {}).get("almacen_destino_id")
        almacen_destino_nombre = (intencion.datos_extra or {}).get("almacen_destino_nombre")

        # Si el LLM asignó el mismo almacén a origen y destino cuando el usuario solo indicó
        # el destino («Traslada 2 de PAP-030 al Almacén B»), descartar ese origen espurio para
        # resolver el almacén real donde reside el producto.
        if almacen_origen and almacen_destino_id and almacen_origen.id == almacen_destino_id:
            if not re.search(rf"\bdesde\s+(?:el\s+)?{re.escape(almacen_origen.nombre)}\b", texto or "", re.IGNORECASE):
                almacen_origen = None

        if not almacen_origen:
            alms_acc = list(usuario.obtener_almacenes_accesibles())
            if len(alms_acc) == 1:
                almacen_origen = alms_acc[0]
            elif estado_operativo and getattr(estado_operativo, "almacen_id", None) and estado_operativo.almacen_id != almacen_destino_id:
                cand_orig = Almacen.objects.filter(id=estado_operativo.almacen_id, activo=True).first()
                if cand_orig and Existencia.objects.filter(producto=producto, almacen=cand_orig, stock_disponible__gt=0).exists():
                    almacen_origen = cand_orig
            if not almacen_origen:
                # Seleccionar almacén donde el producto tenga stock disponible diferente al destino
                ex_qs = Existencia.objects.filter(producto=producto, stock_disponible__gt=0)
                if usuario.es_usuario_normal and not usuario.acceso_todos_almacenes:
                    ex_qs = ex_qs.filter(almacen__in=alms_acc)
                if almacen_destino_id:
                    ex_qs = ex_qs.exclude(almacen_id=almacen_destino_id)
                ex_con_stock = ex_qs.first()
                if ex_con_stock:
                    almacen_origen = ex_con_stock.almacen
                else:
                    alm_cat = getattr(producto.categoria, "almacen_principal", None)
                    if alm_cat and usuario.tiene_acceso_almacen(alm_cat):
                        almacen_origen = alm_cat

        if not almacen_origen:
            msg = f"¿Desde cuál almacén desea trasladar el producto **{producto.nombre}**?"
            return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        if not almacen_destino_id:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "TRASLADO_ENTRE",
                    "producto_id": producto.id,
                    "almacen_id": almacen_origen.id,
                    "cantidad": intencion.cantidad,
                    "observacion": obs_tr_final,
                    "esperando": "destino",
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = (
                f"Producto: **{producto.nombre}** (`{producto.codigo_interno}`).\n"
                f"Almacén origen: **{almacen_origen.nombre}**.\n\n"
                f"¿A cuál almacén o bodega desea trasladar este producto?"
            )
            return {
                "tipo": "aclaracion",
                "esperando": "destino",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(
                    f"El producto {producto.nombre} está en el almacén {almacen_origen.nombre}. ¿A cuál almacén desea trasladarlo?"
                ),
            }

        # Validar que destino existe y es diferente al origen
        try:
            almacen_destino = Almacen.objects.get(id=almacen_destino_id, activo=True)
        except Almacen.DoesNotExist:
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"El almacén destino '{almacen_destino_nombre or almacen_destino_id}' no existe o fue desactivado."
            return {"tipo": "error", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        if not usuario.tiene_acceso_almacen(almacen_destino):
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            msg = f"No cuenta con autorización para trasladar mercancía hacia el almacén '{almacen_destino.nombre}'."
            return {"tipo": "error", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        if almacen_destino.id == almacen_origen.id:
            msg = "El almacén de origen y destino son el mismo. Selecciona un almacén diferente como destino."
            return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        cantidad = intencion.cantidad
        if not cantidad or cantidad <= 0:
            ex_pre_tr = Existencia.objects.filter(producto=producto, almacen=almacen_origen).first()
            stk_pre_tr = ex_pre_tr.stock_disponible if ex_pre_tr else 0
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "TRASLADO_ENTRE",
                    "producto_id": producto.id,
                    "almacen_id": almacen_origen.id,
                    "almacen_destino_id": almacen_destino.id,
                    "almacen_destino_nombre": almacen_destino.nombre,
                    "observacion": obs_tr_final,
                    "esperando": "cantidad",
                }
                HakuOperationalContext.guardar(estado_operativo)
            nota_total_tr = (
                f" Actualmente hay **{stk_pre_tr}** unidades disponibles en {almacen_origen.nombre} "
                f"(por regla de almacén único, para mover el producto completo debe trasladar las {stk_pre_tr} unidades)."
                if stk_pre_tr > 0 else ""
            )
            msg = f"¿Cuántas unidades de {producto.nombre} desea trasladar de {almacen_origen.nombre} a {almacen_destino.nombre}?{nota_total_tr}"
            msg_v = (
                f"¿Cuántas unidades de {producto.nombre} desea trasladar de {almacen_origen.nombre} a {almacen_destino.nombre}? "
                + (f"Hay {stk_pre_tr} disponibles y el traslado entre almacenes mueve la totalidad." if stk_pre_tr > 0 else "")
            )
            return {"tipo": "aclaracion", "esperando": "cantidad", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg_v)}

        # Verificar stock disponible antes de proponer
        existencia = Existencia.objects.filter(producto=producto, almacen=almacen_origen).first()
        stock_actual = existencia.stock_disponible if existencia else 0
        if stock_actual < cantidad:
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            msg = (
                f"Stock insuficiente para el traslado. "
                f"El producto **{producto.nombre}** tiene **{stock_actual}** unidades disponibles "
                f"en **{almacen_origen.nombre}**, pero se solicitaron **{cantidad}**."
            )
            return {"tipo": "error", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(
                f"Stock insuficiente. Solo hay {stock_actual} unidades disponibles de {producto.nombre} en {almacen_origen.nombre}."
            )}

        # Consultar ubicaciones físicas activas del almacén destino (DN-01)
        ubicaciones_destino = list(
            Ubicacion.objects.filter(
                almacen=almacen_destino, activo=True, eliminado=False
            ).order_by("nombre")
        )
        if not ubicaciones_destino:
            msg = (
                f"El almacén destino **{almacen_destino.nombre}** no tiene ubicaciones físicas registradas o activas. "
                f"Para trasladar productos hacia este almacén, primero debes registrar al menos una ubicación física."
            )
            if estado_operativo:
                estado_operativo.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_operativo)
            return {
                "tipo": "error",
                "mensaje_pantalla": msg,
                "mensaje_verbal": normalizar_para_tts(
                    f"El almacén {almacen_destino.nombre} no tiene ubicaciones físicas registradas. Registra una ubicación primero."
                ),
                "datos": {"almacen_sin_ubicaciones": True, "almacen_id": almacen_destino.id},
            }

        # Comprobar si ya se seleccionó una ubicación física destino válida
        ub_dest_id = (intencion.datos_extra or {}).get("ubicacion_destino_id")
        ub_dest = None
        if ub_dest_id:
            ub_dest = next((u for u in ubicaciones_destino if u.id == ub_dest_id), None)

        if not ub_dest:
            ub_nombre_extra = (intencion.datos_extra or {}).get("ubicacion_destino_nombre", "")
            from inventario.utils import normalizar_texto as norm_txt
            t_norm = norm_txt(texto_lower)
            if ub_nombre_extra:
                ub_dest = next(
                    (u for u in ubicaciones_destino if u.nombre.lower() == ub_nombre_extra.lower()),
                    None,
                )
            if not ub_dest:
                ub_dest = _seleccionar_entidad_en_texto(
                    ubicaciones_destino,
                    texto_lower,
                    t_norm,
                    norm_fn=norm_txt,
                    usar_clave_canonica=True,
                    preferir_ultimo=True,
                )

        if not ub_dest:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "TRASLADO_ENTRE",
                    "producto_id": producto.id,
                    "almacen_id": almacen_origen.id,
                    "almacen_destino_id": almacen_destino.id,
                    "almacen_destino_nombre": almacen_destino.nombre,
                    "cantidad": cantidad,
                    "observacion": obs_tr_final,
                    "esperando": "ubicacion_destino",
                    "candidatos_ubicacion": [{"id": u.id, "nombre": u.nombre} for u in ubicaciones_destino],
                }
                HakuOperationalContext.guardar(estado_operativo)

            msg_pantalla = (
                f"Producto: **{producto.nombre}** (`{producto.codigo_interno}`).\n"
                f"Traslado hacia: **{almacen_destino.nombre}** ({cantidad} unidades).\n\n"
                f"Por favor selecciona la ubicación física de destino en **{almacen_destino.nombre}**:\n"
                + "\n".join(f"- **{u.nombre}**" for u in ubicaciones_destino)
                + f"\n\n¿En cuál de estas ubicaciones desea colocar el producto?"
            )
            nombres_v = [u.nombre for u in ubicaciones_destino[:3]]
            msg_verbal = normalizar_para_tts(
                f"El almacén {almacen_destino.nombre} tiene las ubicaciones {', '.join(nombres_v)}. ¿En cuál desea colocar el producto?"
            )
            return {
                "tipo": "aclaracion",
                "esperando": "ubicacion_destino",
                "mensaje_pantalla": msg_pantalla,
                "mensaje_verbal": msg_verbal,
                "datos": {
                    "esperando": "ubicacion_destino",
                    "requiere_ubicacion_destino": True,
                    "candidatos": [{"id": u.id, "nombre": u.nombre} for u in ubicaciones_destino],
                },
            }

        ahora = round(time.time(), 4)
        cid_res, thash_res = _resolver_contexto_contrato(sesion)
        propuesta = {
            "id": str(uuid.uuid4()),
            "tipo_operacion": Movimiento.Tipo.TRASLADO_ENTRE,
            "producto_id": producto.id,
            "producto_nombre": producto.nombre,
            "producto_codigo": producto.codigo_interno,
            "almacen_origen_id": almacen_origen.id,
            "almacen_origen_nombre": almacen_origen.nombre,
            "almacen_destino_id": almacen_destino.id,
            "almacen_destino_nombre": almacen_destino.nombre,
            "ubicacion_destino_id": ub_dest.id,
            "ubicacion_destino_nombre": ub_dest.nombre,
            "cantidad": cantidad,
            "motivo": f"Traslado vía Haku de {almacen_origen.nombre} a {almacen_destino.nombre} ({ub_dest.nombre})",
            "observacion": obs_tr_final,
            "usuario_id": getattr(usuario, "id", None),
            "creado_en": ahora,
            "expira_en": round(ahora + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4),
            "version_contrato": VERSION_FIRMA_PROPUESTA,
            "conversacion_id": cid_res,
            "session_token_hash": thash_res,
        }
        calcular_huella_propuesta(propuesta)

        if estado_operativo:
            estado_operativo.operacion_pendiente = None
            estado_operativo.propuesta_activa = propuesta
            estado_operativo.registrar_accion(
                "PROPUESTA_TRASLADO",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
                cantidad=cantidad,
                detalles={
                    "almacen_origen": almacen_origen.nombre,
                    "almacen_destino": almacen_destino.nombre,
                    "ubicacion_destino": ub_dest.nombre,
                },
            )
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_propuesta_activa"] = propuesta
        HakuDiagnostico.registrar_propuesta("generada")

        linea_obs_tr = f"- **Observación:** {obs_tr_final}\n" if obs_tr_final else ""
        aviso_parcial_tr = (
            f"\n> **Nota (Regla de almacén único):** Este producto tiene **{stock_actual}** unidades disponibles en {almacen_origen.nombre}. "
            f"Al confirmar un traslado entre almacenes se exige mover la totalidad ({stock_actual} unidades). "
            f"Puedes decir *«Cambia la cantidad a {stock_actual}»* antes de confirmar.\n"
            if cantidad < stock_actual else ""
        )
        u_tr = "unidad" if cantidad == 1 else "unidades"
        u_tr_orig = "unidad" if stock_actual == 1 else "unidades"
        pantalla = (
            f"### Propuesta de traslado pendiente de confirmación\n"
            f"- **Operación:** Traslado entre almacenes\n"
            f"- **Producto:** {producto.nombre} (`{producto.codigo_interno}`)\n"
            f"- **Cantidad:** {cantidad} {u_tr}\n"
            f"- **Origen:** {almacen_origen.nombre}\n"
            f"- **Destino:** {almacen_destino.nombre}\n"
            f"- **Ubicación destino:** {ub_dest.nombre}\n"
            f"- **Stock actual en origen:** {stock_actual} {u_tr_orig}\n"
            f"{linea_obs_tr}{aviso_parcial_tr}\n"
            f"¿Confirmas el traslado?"
        )
        verbal = (
            f"¿Confirmas trasladar {cantidad} {u_tr} de {producto.nombre} "
            f"desde {almacen_origen.nombre} hacia {almacen_destino.nombre}, en la ubicación {ub_dest.nombre}?"
            + (
                f" Ten en cuenta que hay {stock_actual} unidades en origen y el traslado entre almacenes requiere mover la totalidad."
                if cantidad < stock_actual else ""
            )
        )
        return {
            "tipo": "propuesta",
            "requiere_confirmacion": True,
            "propuesta": propuesta,
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
        }

    if intencion.tipo == IntencionTipo.INHABILITAR_PRODUCTO:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.eliminar"):
            raise PermissionDenied("No cuentas con permisos suficientes (catalogo.eliminar) para inhabilitar o archivar productos.")

        producto = intencion.producto
        from .agente.planificador import tiene_referencia_contextual_operacion
        if not producto and tiene_referencia_contextual_operacion(texto):
            if sesion and sesion.get("haku_ultimo_producto_id"):
                producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id"), activo=True).first()
            elif estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
                pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
                producto = Producto.objects.filter(id=pid_ctx, activo=True).first()
        if not producto:
            termino = re.sub(r"^(?:inhabilitar|desactivar|archivar|dar\s+de\s+baja)\s+(?:el\s+producto\s+|el\s+|la\s+|los\s+|las\s+|un\s+|una\s+)?", "", texto, flags=re.IGNORECASE).strip()
            if termino:
                from .lenguaje.normalizador import NormalizadorLenguajeHaku
                p_res, _, cands_res = NormalizadorLenguajeHaku._resolver_producto_en_texto(termino, usuario)
                if p_res:
                    producto = p_res
                elif cands_res and len(cands_res) == 1:
                    producto = Producto.objects.filter(id=cands_res[0]["id"]).first()
        if not producto:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "INHABILITAR_PRODUCTO",
                    "esperando": "producto",
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = "¿Cuál producto desea inhabilitar o archivar del inventario?"
            return {"tipo": "aclaracion", "esperando": "producto", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        # Comprobar estado real: si ya está inactivo o archivado, informar
        if not producto.activo or producto.archivado:
            msg = f"El producto {producto.nombre} ({producto.codigo_interno}) ya se encuentra inhabilitado en el sistema."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": f"⚠️ {msg}",
                "mensaje_verbal": normalizar_para_tts(msg),
            }

        ahora = round(time.time(), 4)
        cid_res, thash_res = _resolver_contexto_contrato(sesion)
        propuesta = {
            "id": str(uuid.uuid4()),
            "tipo_operacion": "INHABILITAR_PRODUCTO",
            "producto_id": producto.id,
            "producto_nombre": producto.nombre,
            "producto_codigo": producto.codigo_interno,
            "usuario_id": getattr(usuario, "id", None),
            "creado_en": ahora,
            "expira_en": round(ahora + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4),
            "version_contrato": VERSION_FIRMA_PROPUESTA,
            "conversacion_id": cid_res,
            "session_token_hash": thash_res,
        }
        calcular_huella_propuesta(propuesta)

        if estado_operativo:
            estado_operativo.operacion_pendiente = None
            estado_operativo.propuesta_activa = propuesta
            estado_operativo.registrar_accion(
                "PROPUESTA_INHABILITAR",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
            )
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_propuesta_activa"] = propuesta
        HakuDiagnostico.registrar_propuesta("generada")

        verbal = f"¿Desea confirmar la inhabilitación del producto {producto.nombre} con código {producto.codigo_interno}?"
        pantalla = (
            f"### Propuesta pendiente de confirmación\n"
            f"- **Operación:** Inhabilitar / archivar producto\n"
            f"- **Producto:** {producto.nombre} (`{producto.codigo_interno}`)\n"
            f"- **Estado actual:** {'Activo' if producto.activo else 'Inactivo'}\n\n"
            f"*Esta acción ocultará el producto de las listas operativas sin borrar su historial.*"
        )
        return {
            "tipo": "propuesta",
            "requiere_confirmacion": True,
            "propuesta": propuesta,
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
        }

    if intencion.tipo == IntencionTipo.HABILITAR_PRODUCTO:
        if not usuario_tiene_permiso_operativo(usuario, "catalogo.editar"):
            raise PermissionDenied("No cuentas con permisos suficientes (catalogo.editar) para habilitar o reactivar productos.")

        producto = intencion.producto
        from .agente.planificador import tiene_referencia_contextual_operacion
        if not producto and tiene_referencia_contextual_operacion(texto):
            if sesion and sesion.get("haku_ultimo_producto_id"):
                producto = Producto.objects.filter(id=sesion.get("haku_ultimo_producto_id")).first()
            elif estado_operativo and (getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)):
                pid_ctx = getattr(estado_operativo, "ultimo_producto_id", None) or getattr(estado_operativo, "producto_id", None)
                producto = Producto.objects.filter(id=pid_ctx).first()
        if not producto:
            termino = re.sub(r"^(?:habilitar|activar|reactivar|dar\s+de\s+alta)\s+(?:el\s+producto\s+|el\s+|la\s+|los\s+|las\s+|un\s+|una\s+)?", "", texto, flags=re.IGNORECASE).strip()
            if termino:
                from .lenguaje.normalizador import NormalizadorLenguajeHaku
                p_res, _, cands_res = NormalizadorLenguajeHaku._resolver_producto_en_texto(
                    termino, usuario, incluir_inactivos=True,
                )
                if p_res:
                    producto = p_res
                elif cands_res and len(cands_res) == 1:
                    producto = Producto.objects.filter(id=cands_res[0]["id"]).first()
        if not producto:
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "HABILITAR_PRODUCTO",
                    "esperando": "producto",
                }
                HakuOperationalContext.guardar(estado_operativo)
            msg = "¿Cuál producto desea habilitar o reactivar en el inventario?"
            return {"tipo": "aclaracion", "esperando": "producto", "mensaje_pantalla": msg, "mensaje_verbal": normalizar_para_tts(msg)}

        # Comprobar estado real: si ya está activo y no archivado, informar
        if producto.activo and not producto.archivado:
            msg = f"El producto {producto.nombre} ({producto.codigo_interno}) ya se encuentra activo y habilitado en el sistema."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": f"ℹ️ {msg}",
                "mensaje_verbal": normalizar_para_tts(msg),
            }

        ahora = round(time.time(), 4)
        cid_res, thash_res = _resolver_contexto_contrato(sesion)
        propuesta = {
            "id": str(uuid.uuid4()),
            "tipo_operacion": "HABILITAR_PRODUCTO",
            "producto_id": producto.id,
            "producto_nombre": producto.nombre,
            "producto_codigo": producto.codigo_interno,
            "usuario_id": getattr(usuario, "id", None),
            "creado_en": ahora,
            "expira_en": round(ahora + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4),
            "version_contrato": VERSION_FIRMA_PROPUESTA,
            "conversacion_id": cid_res,
            "session_token_hash": thash_res,
        }
        calcular_huella_propuesta(propuesta)

        if estado_operativo:
            estado_operativo.operacion_pendiente = None
            estado_operativo.propuesta_activa = propuesta
            estado_operativo.registrar_accion(
                "PROPUESTA_HABILITAR",
                producto=producto.nombre,
                producto_id=producto.id,
                codigo=producto.codigo_interno,
            )
            HakuOperationalContext.guardar(estado_operativo)
        if sesion is not None:
            sesion["haku_propuesta_activa"] = propuesta
        HakuDiagnostico.registrar_propuesta("generada")

        verbal = f"¿Desea confirmar la habilitación y reactivación del producto {producto.nombre} con código {producto.codigo_interno}?"
        pantalla = (
            f"### Propuesta pendiente de confirmación\n"
            f"- **Operación:** Habilitar / reactivar producto\n"
            f"- **Producto:** {producto.nombre} (`{producto.codigo_interno}`)\n"
            f"- **Estado actual:** {'Activo' if producto.activo else 'Inactivo / Archivado'}\n\n"
            f"*Esta acción reactivará el producto para operaciones ordinarias en el inventario.*"
        )
        return {
            "tipo": "propuesta",
            "requiere_confirmacion": True,
            "propuesta": propuesta,
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": normalizar_para_tts(verbal),
        }


    # ─────────────────────────────────────────────────────────────────────────
    # D. LOCUCIÓN FUERA DE DOMINIO (NO_INTENCION_INVENTARIO / CONVERSACION_GENERAL)
    # Haku Inventario NO es un chatbot general: solo gestiona almacén e inventario.
    # ─────────────────────────────────────────────────────────────────────────
    if intencion.tipo == IntencionTipo.SOLICITUD_NO_COMPRENDIDA:
        msg = "Disculpe, no entendí la solicitud. ¿Podría repetirla?"
        return {
            "tipo": "no_comprendido",
            "mensaje_pantalla": msg,
            "mensaje_verbal": normalizar_para_tts(msg),
            "datos": {"no_comprendido": True},
        }

    msg_fuera_dominio = (
        "Solo gestiono operaciones y consultas del inventario "
        "(stock, ubicaciones, movimientos, préstamos y métricas del almacén)."
    )
    msg_verbal = (
        "Esa consulta está fuera de mis funciones. Soy HAKU, el asistente de inventario de AudacIA. "
        "Puedo ayudarte a consultar productos, existencias o movimientos de inventario."
    )
    return {
        "tipo": "fuera_de_dominio",
        "mensaje_pantalla": msg_fuera_dominio,
        "mensaje_verbal": normalizar_para_tts(msg_verbal),
        "datos": {"fuera_de_dominio": True},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. EJECUCIÓN ATÓMICA DE LA CONFIRMACIÓN DE UNA PROPUESTA
# ═════════════════════════════════════════════════════════════════════════════

class IdempotencyConflictError(ValidationError):
    """Lanzada cuando se reutiliza una clave de idempotencia con una propuesta cuyo contenido semántico difiere."""

    def __init__(self, mensaje="La clave de idempotencia ya fue utilizada para una propuesta con diferente contenido."):
        super().__init__(mensaje)


# VERSION_FIRMA_PROPUESTA definida al inicio del módulo ("v2")


class ErrorConfiguracionCriptografica(ValidationError):
    """Lanzada cuando falta la configuración del secreto criptográfico del servidor."""

    def __init__(self, mensaje="Error de configuración del servidor: secreto criptográfico no disponible."):
        super().__init__(mensaje)


def _obtener_secreto_propuestas() -> bytes:
    """Obtiene la clave secreta exclusiva del servidor para firmar propuestas HAKU.

    Busca preferentemente 'HAKU_PROPUESTA_SECRET' en settings; si no está definida,
    utiliza 'SECRET_KEY' de Django. Si ninguna existe o es vacía, lanza ErrorConfiguracionCriptografica.
    """
    secreto = None
    try:
        secreto = getattr(settings, "HAKU_PROPUESTA_SECRET", None)
    except Exception:
        secreto = None

    if not secreto:
        try:
            secreto = getattr(settings, "SECRET_KEY", None)
        except Exception:
            secreto = None

    if not secreto or not isinstance(secreto, (str, bytes)):
        raise ErrorConfiguracionCriptografica(
            "Error de configuración del servidor: secreto criptográfico no disponible."
        )

    if isinstance(secreto, str):
        secreto = secreto.strip().encode("utf-8")

    if not secreto:
        raise ErrorConfiguracionCriptografica(
            "Error de configuración del servidor: secreto criptográfico no disponible."
        )

    return secreto


def calcular_hash_token_sesion(session_token: str) -> str:
    """Calcula el hash HMAC-SHA256 del token de sesión para inclusión segura en el contrato v2.

    Usa el secreto de propuestas del servidor y el prefijo de dominio 'haku-session-token-v2:'.
    Retorna 64 caracteres hexadecimales en minúsculas.
    """
    if not session_token or not isinstance(session_token, str):
        return ""
    token_limpio = session_token.strip()
    if not token_limpio:
        return ""
    secreto = _obtener_secreto_propuestas()
    payload = b"haku-session-token-v2:" + token_limpio.encode("utf-8")
    return hmac.new(secreto, payload, hashlib.sha256).hexdigest().lower()


def _resolver_contexto_contrato(sesion: Any | None, conversacion_id: str | None = None) -> tuple[str, str]:
    """Resuelve conversacion_id y session_token_hash a partir de la sesión activa o parámetros."""
    cid = conversacion_id
    if not cid and sesion is not None:
        if isinstance(sesion, dict):
            cid = sesion.get("haku_conversacion_activa_id")
        elif hasattr(sesion, "get"):
            cid = sesion.get("haku_conversacion_activa_id")
    if not cid:
        cid = str(uuid.uuid4())
    else:
        try:
            uuid.UUID(str(cid).strip(), version=4)
            cid = str(cid).strip()
        except (ValueError, TypeError, AttributeError):
            cid = str(uuid.uuid4())

    token = None
    if sesion is not None:
        if isinstance(sesion, dict):
            token = sesion.get("haku_session_token")
        elif hasattr(sesion, "get"):
            token = sesion.get("haku_session_token")
        if not token or not isinstance(token, str):
            from .servicios_sesion import obtener_o_crear_session_token
            token = obtener_o_crear_session_token(sesion)

    if not token or not isinstance(token, str):
        token = secrets.token_hex(32)

    thash = calcular_hash_token_sesion(token)
    return str(cid), thash


def _normalizar_valor_canonico(val: Any) -> Any:
    """Normaliza un valor arbitrario de forma recursiva, determinista y respetando tipos."""
    if val is None:
        return None
    # En Python, bool es subclase de int: comprobar bool antes de int
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, (float, Decimal)):
        try:
            return round(float(val), 4)
        except (ValueError, TypeError):
            return 0.0
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        return {
            str(k).strip(): _normalizar_valor_canonico(v)
            for k, v in sorted(val.items(), key=lambda item: str(item[0]))
        }
    if isinstance(val, (list, tuple)):
        return [_normalizar_valor_canonico(x) for x in val]
    if isinstance(val, set):
        items_norm = [_normalizar_valor_canonico(x) for x in val]
        return sorted(items_norm, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    return str(val).strip()


CAMPOS_PROTEGIDOS_CONTRATO = (
    "id",
    "usuario_id",
    "tipo_operacion",
    "producto_id",
    "producto_codigo",
    "producto_nombre",
    "cantidad",
    "almacen_origen_id",
    "almacen_origen_nombre",
    "almacen_destino_id",
    "almacen_destino_nombre",
    "ubicacion_origen_id",
    "ubicacion_destino_id",
    "categoria_id",
    "destinatario",
    "motivo",
    "prestamo_id",
    "creado_en",
    "expira_en",
    "version_contrato",
    "conversacion_id",
    "session_token_hash",
    "datos_extra",
)


CLAVES_EXTRA_PERMITIDAS_PROPUESTA = frozenset({
    "sustituye",
    "ubicacion_destino_nombre",
    "origen_modificacion_id",
    "stock_minimo",
    "es_prestable",
    "codigo_barras",
    "descripcion",
    "observacion",
    "categoria_nombre",
    "ubicacion_origen_nombre",
    "tipo_categoria_id",
})
MAX_BIGINT_VAL = 9223372036854775807


def validar_uuid4_canonico_estricto(val: Any) -> bool:
    """Valida que una cadena represente estrictamente un UUID4 canónico en minúsculas conforme a RFC 4122.
    
    Sin aplicar strip() ni lower() y sin pasar version=4 al constructor de Python para no mutar los bits.
    """
    if type(val) is not str or len(val) != 36:
        return False
    try:
        u = uuid.UUID(val)
        return str(u) == val and u.version == 4 and u.variant == uuid.RFC_4122
    except (ValueError, TypeError, AttributeError):
        return False


def validar_usuario_id_envoltorio(val: Any) -> bool:
    """Valida que usuario_id sea una cadena decimal ASCII no vacía en rango BigAutoField sin ceros iniciales."""
    if type(val) is not str or len(val) < 1 or len(val) > 19:
        return False
    if not all(c in "0123456789" for c in val):
        return False
    if val == "0" or val.startswith("0"):
        return False
    try:
        n = int(val)
        return 1 <= n <= MAX_BIGINT_VAL
    except (ValueError, TypeError, OverflowError):
        return False


def validar_cantidad_operativa_estricta(v: Any, obligatoria: bool = True, permitir_cero: bool = False) -> int | None:
    """Valida que la cantidad operativa sea un entero discreto estrictamente positivo (o >= 0 si permitir_cero)."""
    if v is None:
        if obligatoria:
            raise ValidationError("La cantidad operativa es obligatoria y no puede ser None.")
        return None
    if type(v) is bool:
        raise ValidationError("La cantidad operativa no puede ser un valor booleano.")
    if isinstance(v, (float, Decimal)):
        raise ValidationError("La cantidad operativa debe ser un número entero discreto sin decimales.")
    if type(v) is int:
        if v < 0 or (v == 0 and not permitir_cero):
            raise ValidationError("La cantidad operativa debe ser un entero estrictamente mayor a cero." if not permitir_cero else "La cantidad no puede ser un número negativo.")
        return v
    if type(v) is str:
        s = v.strip()
        if s.isdigit():
            val = int(s)
            if val < 0 or (val == 0 and not permitir_cero):
                raise ValidationError("La cantidad operativa debe ser un entero estrictamente mayor a cero." if not permitir_cero else "La cantidad no puede ser un número negativo.")
            return val
        raise ValidationError(f"Formato no numérico o decimal en cantidad operativa: '{v}'")
    raise ValidationError(f"Tipo no convertible a cantidad entera: {type(v).__name__}")


def canonicalizar_contenido_propuesta(propuesta: dict[str, Any]) -> bytes:
    """Produce una representación binaria canónica, determinista e inmutable de la propuesta.

    Normaliza tipos de datos, ordenamiento lexicográfico de claves, codificación UTF-8
    y preservación explícita de valores nulos, estructuras anidadas y decimales.
    Cubre todos los campos protegidos de la propuesta y aplica lista blanca estricta
    para prevenir inyección de campos no autorizados (F-02).
    """
    if type(propuesta) is not dict:
        raise ValidationError("La propuesta debe ser un diccionario válido.")

    def _norm_id(v: Any) -> int | str | None:
        if v is None:
            return None
        if type(v) is bool:
            raise ValidationError("Los identificadores relacionales no pueden ser booleanos.")
        if type(v) is int:
            return v
        if isinstance(v, uuid.UUID):
            return str(v)
        s = str(v).strip()
        if not s:
            return None
        if s.isdigit():
            try:
                return int(s)
            except ValueError:
                return s
        return s

    def _norm_str(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s

    def _norm_time(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return round(float(v), 4)
        except (ValueError, TypeError):
            return None

    # 1. Contrato estricto de datos_extra (F-02)
    datos_extra_raw = propuesta.get("datos_extra")
    de_dict = {}
    if datos_extra_raw is not None:
        if type(datos_extra_raw) is not dict:
            raise ValidationError(
                f"El campo 'datos_extra' debe ser un diccionario o None, recibido {type(datos_extra_raw).__name__}."
            )
        for k, v in datos_extra_raw.items():
            if type(k) is not str:
                raise ValidationError(f"Clave en 'datos_extra' no es string: {type(k).__name__}")
            if k != k.strip() or any(c.isspace() for c in k):
                raise ValidationError(f"Clave en 'datos_extra' no es canónica: '{k}'")
            if k not in CLAVES_EXTRA_PERMITIDAS_PROPUESTA:
                raise ValidationError(f"Clave extra no permitida en 'datos_extra': '{k}'")
            de_dict[k] = v

    # 2. Claves en nivel superior y validación contra inyección no autorizada
    for k, v in propuesta.items():
        if type(k) is not str:
            raise ValidationError(f"Clave de propuesta no es string: {type(k).__name__}")
        if k != k.strip() or any(c.isspace() for c in k):
            raise ValidationError(f"Clave en propuesta no es canónica: '{k}'")
        if k in CAMPOS_PROTEGIDOS_CONTRATO or k in (
            "firma_hmac",
            "huella_payload",
            "almacen_nombre",
            "session_token",
            "modificacion_sin_resolver",
            "categoria_nombre",
            "ubicacion_origen_nombre",
        ):
            continue
        if k in CLAVES_EXTRA_PERMITIDAS_PROPUESTA:
            if k in de_dict:
                if str(de_dict[k]).strip() != str(v).strip():
                    raise ValidationError(f"Clave duplicada ambigua '{k}' presente en nivel superior y datos_extra con valores divergentes.")
            else:
                de_dict[k] = v
        else:
            raise ValidationError(f"Clave no autorizada en propuesta: '{k}'")

    # 3. Validación y normalización estricta de valores dentro de datos_extra
    datos_extra_canonico = {}
    if de_dict:
        for k in sorted(de_dict.keys()):
            v = de_dict[k]
            if k in ("sustituye", "origen_modificacion_id"):
                if not validar_uuid4_canonico_estricto(v):
                    raise ValidationError(f"El campo '{k}' debe ser un UUID4 canónico en minúsculas.")
                datos_extra_canonico[k] = str(v)
            elif k == "ubicacion_destino_nombre":
                if type(v) is not str:
                    raise ValidationError("El campo 'ubicacion_destino_nombre' debe ser una cadena de texto.")
                if any(unicodedata.category(c) == "Cc" for c in v):
                    raise ValidationError("Caracter de control Unicode (categoría Cc) detectado en 'ubicacion_destino_nombre'.")
                if len(v) < 1 or len(v) > 100:
                    raise ValidationError("Longitud de 'ubicacion_destino_nombre' fuera de rango (1-100 caracteres).")
                if v != v.strip():
                    raise ValidationError("Espacios iniciales o finales no permitidos en 'ubicacion_destino_nombre'.")
                datos_extra_canonico["ubicacion_destino_nombre"] = v
            elif k == "stock_minimo":
                if type(v) is bool:
                    raise ValidationError("El stock mínimo no puede ser un valor booleano.")
                try:
                    s_min = int(str(v).strip()) if isinstance(v, (int, str)) and str(v).strip().lstrip("-").isdigit() else None
                    if s_min is None or s_min < 0:
                        raise ValidationError("El stock mínimo debe ser un número entero mayor o igual a cero.")
                    datos_extra_canonico["stock_minimo"] = s_min
                except (ValueError, TypeError):
                    raise ValidationError("El stock mínimo debe ser un número entero mayor o igual a cero.")
            elif k == "codigo_barras":
                if v is None:
                    datos_extra_canonico["codigo_barras"] = ""
                elif type(v) is str:
                    datos_extra_canonico["codigo_barras"] = v.strip()
                else:
                    raise ValidationError("El campo 'codigo_barras' debe ser una cadena de texto.")
            elif k == "descripcion":
                datos_extra_canonico["descripcion"] = str(v).strip() if v is not None else ""
            elif k == "categoria_nombre":
                datos_extra_canonico["categoria_nombre"] = str(v).strip() if v is not None else ""
            elif k == "ubicacion_origen_nombre":
                datos_extra_canonico["ubicacion_origen_nombre"] = str(v).strip() if v is not None else ""
            elif k == "tipo_categoria_id":
                datos_extra_canonico["tipo_categoria_id"] = _norm_id(v)
            elif k == "es_prestable":
                if type(v) is not bool:
                    raise ValidationError("El campo 'es_prestable' debe ser un valor booleano.")
                datos_extra_canonico["es_prestable"] = v

    # 4. Resolución estricta de alias histórico almacen_nombre vs almacen_origen_nombre
    s_orig = _norm_str(propuesta.get("almacen_origen_nombre"))
    s_alias = _norm_str(propuesta.get("almacen_nombre"))
    if s_orig is not None and s_alias is not None:
        if s_orig != s_alias:
            raise ValidationError(
                f"Conflicto por duplicidad ambigua entre almacen_origen_nombre ('{s_orig}') y almacen_nombre ('{s_alias}')"
            )
        alm_origen_nom = s_orig
    elif s_orig is not None:
        alm_origen_nom = s_orig
    elif s_alias is not None:
        alm_origen_nom = s_alias
    else:
        alm_origen_nom = None

    # 5. Cantidad operativa discreta y obligatoria según tipo de operación
    tipo_op = _norm_str(propuesta.get("tipo_operacion") or "").upper() if propuesta.get("tipo_operacion") is not None else None
    es_sin_cant = tipo_op in ("INHABILITAR_PRODUCTO", "HABILITAR_PRODUCTO")
    es_crear = (tipo_op == "CREAR_PRODUCTO")
    cant_canon = None if es_sin_cant else validar_cantidad_operativa_estricta(propuesta.get("cantidad"), obligatoria=True, permitir_cero=es_crear)

    if es_crear:
        prod_nom = _norm_str(propuesta.get("producto_nombre"))
        if not prod_nom:
            raise ValidationError("El campo 'producto_nombre' es obligatorio para CREAR_PRODUCTO.")
        if _norm_id(propuesta.get("categoria_id")) is None:
            raise ValidationError("El campo 'categoria_id' es obligatorio para CREAR_PRODUCTO.")
        if _norm_id(propuesta.get("almacen_origen_id")) is None:
            raise ValidationError("El campo 'almacen_origen_id' es obligatorio para CREAR_PRODUCTO.")
        if _norm_id(propuesta.get("ubicacion_origen_id")) is None:
            raise ValidationError("El campo 'ubicacion_origen_id' es obligatorio para CREAR_PRODUCTO.")
        if "stock_minimo" not in datos_extra_canonico:
            raise ValidationError("El campo 'stock_minimo' es obligatorio para CREAR_PRODUCTO.")

    contenido_canonico: dict[str, Any] = {
        "id": _norm_str(propuesta.get("id")),
        "usuario_id": _norm_id(propuesta.get("usuario_id")),
        "tipo_operacion": tipo_op,
        "producto_id": _norm_id(propuesta.get("producto_id")),
        "producto_codigo": _norm_str(propuesta.get("producto_codigo")),
        "producto_nombre": _norm_str(propuesta.get("producto_nombre")),
        "cantidad": cant_canon if cant_canon is not None else 0,
        "almacen_origen_id": _norm_id(propuesta.get("almacen_origen_id")),
        "almacen_origen_nombre": alm_origen_nom,
        "almacen_destino_id": _norm_id(propuesta.get("almacen_destino_id")),
        "almacen_destino_nombre": _norm_str(propuesta.get("almacen_destino_nombre")),
        "ubicacion_origen_id": _norm_id(propuesta.get("ubicacion_origen_id")),
        "ubicacion_destino_id": _norm_id(propuesta.get("ubicacion_destino_id")),
        "categoria_id": _norm_id(propuesta.get("categoria_id")),
        "destinatario": _norm_str(propuesta.get("destinatario")),
        "motivo": _norm_str(propuesta.get("motivo")),
        "prestamo_id": _norm_str(propuesta.get("prestamo_id")),
        "creado_en": _norm_time(propuesta.get("creado_en")),
        "expira_en": _norm_time(propuesta.get("expira_en")),
        "version_contrato": _norm_str(propuesta.get("version_contrato") or VERSION_FIRMA_PROPUESTA),
        "conversacion_id": _norm_str(propuesta.get("conversacion_id")),
        "session_token_hash": _norm_str(propuesta.get("session_token_hash")),
        "datos_extra": datos_extra_canonico if datos_extra_canonico else None,
    }

    json_str = json.dumps(
        contenido_canonico,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return json_str.encode("utf-8")


def calcular_snapshot_sha256(propuesta_snapshot: dict[str, Any]) -> str:
    """Calcula el digest SHA-256 de la proyección binaria canónica del snapshot.

    Utiliza exactamente la misma función canonicalizar_contenido_propuesta().
    """
    canon = canonicalizar_contenido_propuesta(propuesta_snapshot)
    return hashlib.sha256(canon).hexdigest().lower()


def calcular_firma_propuesta(propuesta: dict[str, Any]) -> str:
    """Calcula la firma HMAC-SHA256 versionada sobre la representación canónica de la propuesta.

    Retorna una cadena con formato: 'v2:<64_caracteres_hexadecimales>'.
    Lanza ErrorConfiguracionCriptografica si el servidor carece de secreto seguro configurado.
    """
    secreto = _obtener_secreto_propuestas()
    canonico = canonicalizar_contenido_propuesta(propuesta)
    firma_hex = hmac.new(secreto, canonico, hashlib.sha256).hexdigest()
    return f"{VERSION_FIRMA_PROPUESTA}:{firma_hex}"


def verificar_firma_propuesta(propuesta: dict[str, Any]) -> bool:
    """Verifica que la propuesta cuente con una firma HMAC-SHA256 v2 válida emitida por el servidor.

    Comprueba:
    1. Que la propuesta sea un diccionario con firma presente en 'firma_hmac' o 'huella_payload'.
    2. Que la versión sea reconocida ('v2:').
    3. Que la firma tenga formato hexadecimal de 64 caracteres.
    4. Que 'conversacion_id' sea obligatorio y un UUID4 sintácticamente válido.
    5. Que 'session_token_hash' coincida con 'session_token' si este último está presente en memoria.
    6. Que coincida con la firma esperada calculada con el secreto del servidor usando hmac.compare_digest().

    Retorna True si es auténtica e inalterada, False en caso contrario.
    """
    if not isinstance(propuesta, dict):
        return False

    firma_recibida = propuesta.get("firma_hmac") or propuesta.get("huella_payload")
    if not firma_recibida or not isinstance(firma_recibida, str):
        return False

    firma_recibida = firma_recibida.strip()
    prefix = f"{VERSION_FIRMA_PROPUESTA}:"
    if not firma_recibida.startswith(prefix):
        return False

    partes = firma_recibida.split(":", 1)
    if len(partes) != 2:
        return False

    hex_recibido = partes[1].strip()
    if len(hex_recibido) != 64 or not all(c in "0123456789abcdefABCDEF" for c in hex_recibido):
        return False

    cid = propuesta.get("conversacion_id")
    if not cid:
        return False
    try:
        uuid.UUID(str(cid).strip(), version=4)
    except (ValueError, TypeError, AttributeError):
        return False

    token_hash = propuesta.get("session_token_hash")
    if not token_hash or not isinstance(token_hash, str) or len(token_hash.strip()) != 64 or not all(c in "0123456789abcdefABCDEF" for c in token_hash.strip()):
        return False
    raw_token = propuesta.get("session_token")
    if raw_token and isinstance(raw_token, str):
        calc_hash = calcular_hash_token_sesion(raw_token)
        if not hmac.compare_digest(token_hash.lower(), calc_hash.lower()):
            return False

    try:
        firma_esperada = calcular_firma_propuesta(propuesta)
    except Exception:
        return False
    return hmac.compare_digest(firma_recibida.lower(), firma_esperada.lower())


def calcular_huella_propuesta(propuesta: dict[str, Any]) -> str:
    """Calcula la firma HMAC-SHA256 autenticada y versionada de una propuesta HAKU v2.

    Asigna automáticamente 'version_contrato', 'conversacion_id', 'session_token_hash',
    'firma_hmac' y 'huella_payload' en el diccionario de la propuesta.
    """
    if not isinstance(propuesta, dict):
        return ""

    if not propuesta.get("version_contrato"):
        propuesta["version_contrato"] = VERSION_FIRMA_PROPUESTA

    if not propuesta.get("conversacion_id"):
        propuesta["conversacion_id"] = str(uuid.uuid4())

    if not propuesta.get("session_token_hash"):
        if propuesta.get("session_token"):
            propuesta["session_token_hash"] = calcular_hash_token_sesion(propuesta["session_token"])
        else:
            import secrets
            tok_gen = secrets.token_hex(32)
            propuesta["session_token"] = tok_gen
            propuesta["session_token_hash"] = calcular_hash_token_sesion(tok_gen)

    firma = calcular_firma_propuesta(propuesta)
    propuesta["firma_hmac"] = firma
    propuesta["huella_payload"] = firma
    return firma


_HEX_SET = set("0123456789abcdef")


def comparar_firma_canonica(firma_almacenada: str, firma_esperada: str) -> bool:
    """Compara firmas HMAC versionadas canónicas sin mutaciones ni normalizaciones.

    Exige estructura exacta 'v2:<64 hex minúsculos>' o 'v1:<64 hex minúsculos>' (longitud 67).
    """
    if not isinstance(firma_almacenada, str) or not isinstance(firma_esperada, str):
        return False
    if len(firma_almacenada) != 67 or len(firma_esperada) != 67:
        return False
    prefix_alm = firma_almacenada[:3]
    prefix_esp = firma_esperada[:3]
    if prefix_alm not in ("v2:", "v1:") or prefix_esp not in ("v2:", "v1:"):
        return False
    if prefix_alm != prefix_esp:
        return False
    cuerpo_alm = firma_almacenada[3:]
    cuerpo_esp = firma_esperada[3:]
    if any(c not in _HEX_SET for c in cuerpo_alm) or any(c not in _HEX_SET for c in cuerpo_esp):
        return False
    return hmac.compare_digest(firma_almacenada, firma_esperada)


def comparar_hash_canonico(hash_almacenado: str, hash_esperado: str) -> bool:
    """Compara digests SHA-256 canónicos sin mutaciones ni normalizaciones.

    Exige estructura exacta de 64 caracteres hexadecimales minúsculos.
    """
    if not isinstance(hash_almacenado, str) or not isinstance(hash_esperado, str):
        return False
    if len(hash_almacenado) != 64 or len(hash_esperado) != 64:
        return False
    if any(c not in _HEX_SET for c in hash_almacenado) or any(c not in _HEX_SET for c in hash_esperado):
        return False
    return hmac.compare_digest(hash_almacenado, hash_esperado)


class _PropuestaLock:
    """Cerrojo por propuesta con conteo de usuarios activos para prevenir desalojos prematuros."""

    def __init__(self, key: str):
        self.key = key
        self._raw_lock = threading.Lock()
        self.usuarios_activos = 0
        self.obtenido_en = time.time()
        self.ultimo_acceso = time.time()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._raw_lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._raw_lock.release()

    def locked(self) -> bool:
        return self._raw_lock.locked()

    def __enter__(self):
        with _propuesta_locks_lock:
            self.usuarios_activos += 1
            self.ultimo_acceso = time.time()
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.release()
        finally:
            with _propuesta_locks_lock:
                self.usuarios_activos = max(0, self.usuarios_activos - 1)
                self.obtenido_en = 0.0
                self.ultimo_acceso = time.time()


_propuesta_locks_lock = threading.RLock()
_propuesta_locks: dict[str, _PropuestaLock] = {}
_MAX_PROPUESTA_LOCKS = 512


def _desvincular_uso_lock(propuesta_id: str) -> None:
    """Libera el registro de uso activo de un cerrojo de propuesta."""
    with _propuesta_locks_lock:
        lock_obj = _propuesta_locks.get(propuesta_id)
        if lock_obj is not None:
            lock_obj.usuarios_activos = max(0, lock_obj.usuarios_activos - 1)
            lock_obj.obtenido_en = 0.0
            lock_obj.ultimo_acceso = time.time()


def _obtener_lock_propuesta(propuesta_id: str) -> _PropuestaLock:
    """Obtiene o crea un cerrojo canónico en memoria por propuesta.

    Garantías de ciclo de vida (Fase 1F):
    - No desaloja cerrojos mientras tengan usuarios activos o esperadores (usuarios_activos > 0).
    - No desaloja cerrojos bloqueados (lk.locked() == True).
    - Protege la ventana de adquisición inicial mediante marca temporal obtenido_en.
    - Desaloja según LRU únicamente cuando se alcanza _MAX_PROPUESTA_LOCKS sobre entradas inactivas.
    """
    with _propuesta_locks_lock:
        ahora = time.time()
        if len(_propuesta_locks) >= _MAX_PROPUESTA_LOCKS:
            candidatos = [
                k for k, lk in _propuesta_locks.items()
                if lk.usuarios_activos <= 0 and not lk.locked() and (ahora - lk.obtenido_en >= 2.0 or lk.obtenido_en == 0.0)
            ]
            candidatos.sort(key=lambda k: _propuesta_locks[k].ultimo_acceso)
            for k in candidatos[: _MAX_PROPUESTA_LOCKS // 2]:
                _propuesta_locks.pop(k, None)

        if propuesta_id not in _propuesta_locks:
            _propuesta_locks[propuesta_id] = _PropuestaLock(propuesta_id)

        lock_obj = _propuesta_locks[propuesta_id]
        lock_obj.obtenido_en = ahora
        lock_obj.ultimo_acceso = ahora
        return lock_obj


def _buscar_ejecucion_previa_catalogo(
    *,
    propuesta: dict[str, Any],
    propuesta_id: str,
    clave_canonica: str,
    idempotency_key: str,
    usuario: Any,
    tipo_op: str,
    firma_propuesta: str,
    huella_actual: str,
) -> dict[str, Any] | None:
    """Recuperación canónica de idempotencia para operaciones de catálogo.

    Aplica rigurosamente B-01 a B-04:
    - B-01: Ámbito lógico global, reconociendo la ausencia de UNIQUE física en Auditoria.
    - B-02: 7 factores obligatorios con coincidencia exacta (c_idem_key == idempotency_key).
    - B-03: Validación estricta c.tipo_entidad == 'Producto' y id_entidad.
    - B-04: Rechazo Fail-Closed si c.usuario_id is None o difiere de usuario.id.
    """
    if not idempotency_key or not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise IdempotencyConflictError("La solicitud carece de una clave de idempotencia válida.")

    # 1. Detección de colisiones con el dominio de Movimiento (Caso K)
    q_mov = Q(clave_idempotencia=clave_canonica) | Q(clave_idempotencia=idempotency_key)
    q_mov |= Q(observacion__contains=f"[client_idempotency_key:{idempotency_key}]")
    q_mov |= Q(observacion__contains=f"[client_idempotency:{idempotency_key}]")
    q_mov |= Q(observacion__contains=f"[propuesta_id:{propuesta_id}]")
    q_mov |= Q(motivo__contains=f"[propuesta_id:{propuesta_id}]")

    if tipo_op == "CREAR_PRODUCTO":
        mov_colision = Movimiento.objects.filter(q_mov).exclude(tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL).first()
    else:
        mov_colision = Movimiento.objects.filter(q_mov).first()
    if mov_colision:
        if str(mov_colision.usuario_id) != str(usuario.id):
            raise PermissionDenied("La propuesta o clave solicitada pertenece a otro usuario.")
        raise IdempotencyConflictError(
            "Colisión entre dominios: la clave o propuesta ya fue ejecutada en una operación de movimiento de inventario."
        )

    # 2. Búsqueda exhaustiva en Auditoria (campos estructurados y texto delimitado con corchetes)
    q_aud = (
        Q(valores_nuevos__propuesta_id=propuesta_id)
        | Q(motivo__contains=f"[propuesta_id:{propuesta_id}]")
        | Q(motivo__contains=f"[clave_canonica:{clave_canonica}]")
        | Q(valores_nuevos__idempotency_key=idempotency_key)
        | Q(motivo__contains=f"[client_idempotency:{idempotency_key}]")
        | Q(motivo__contains=f"[client_idempotency_key:{idempotency_key}]")
    )

    # 3. Consulta atómica única acotada a 51 registros (B-01 / Consistencia)
    # Excluir únicamente el registro de linaje PROPUESTA_MODIFICADA (P1 -> P2) que apunta al ID de la
    # nueva propuesta antes de ser confirmada; cualquier otra colisión de entidad sigue rechazándose Fail-Closed.
    candidatos = list(
        Auditoria.objects.filter(q_aud)
        .exclude(tipo_entidad="PropuestaHaku", accion="PROPUESTA_MODIFICADA")
        .order_by("fecha_hora", "id")[:51]
    )
    total = len(candidatos)

    # Caso A: Sin candidatos
    if total == 0:
        return None

    # Caso H: Exceso del límite operativo (> 50 candidatos)
    if total == 51:
        raise IdempotencyConflictError(
            "Conjunto de candidatos ambiguo o excesivo (> 50 registros). Límite operativo superado. "
            "Rechazo preventivo Fail-Closed por imposibilidad de evaluar exhaustividad."
        )

    # 4. Evaluación exhaustiva de los candidatos recuperados
    for c in candidatos:
        # Validación B-04: Propietario ausente o ajeno
        if c.usuario_id is None:
            raise IdempotencyConflictError(
                "Registro histórico de auditoría sin propietario verificable. "
                "Rechazo preventivo Fail-Closed: no es posible certificar identidad ni autorizar ejecución idempotente."
            )
        if str(c.usuario_id) != str(usuario.id):
            # Caso D o E: Pertenece a otro usuario
            raise PermissionDenied("La propuesta o clave solicitada pertenece a otro usuario.")

        # Validación B-03: Tipo de entidad obligatorio
        if c.tipo_entidad != "Producto":
            # Caso I o L: Colisión con otra entidad (e.g. Almacen #42 frente a Producto #42)
            raise IdempotencyConflictError(
                f"Colisión de identificador entre entidades: la clave coincide con una auditoría "
                f"de tipo '{c.tipo_entidad}', no de 'Producto'."
            )

        vn = c.valores_nuevos if isinstance(c.valores_nuevos, dict) else {}

        # Extracción estricta de factores
        c_prop_id = vn.get("propuesta_id")
        if not c_prop_id and c.motivo:
            m = re.search(r'\[propuesta_id:([^\]]+)\]', c.motivo)
            c_prop_id = m.group(1) if m else None

        c_idem_key = vn.get("idempotency_key")
        if not c_idem_key and c.motivo:
            m = re.search(r'\[client_idempotency(?:_key)?:([^\]]+)\]', c.motivo)
            c_idem_key = m.group(1) if m else None

        c_firma = vn.get("payload_firma")
        if not c_firma and c.motivo:
            m = re.search(r'\[payload_firma:([^\]]+)\]', c.motivo)
            c_firma = m.group(1) if m else None

        c_hash = vn.get("payload_hash")
        if not c_hash and c.motivo:
            m = re.search(r'\[payload_hash:([^\]]+)\]', c.motivo)
            c_hash = m.group(1) if m else None

        c_cid = vn.get("conversacion_id")
        if not c_cid and c.motivo:
            m_c = re.search(r'\[conversacion_id:([^\]]+)\]', c.motivo)
            c_cid = m_c.group(1) if m_c else None
        cid_prop = propuesta.get("conversacion_id")
        if cid_prop and c_cid and str(c_cid).strip().lower() != str(cid_prop).strip().lower():
            raise IdempotencyConflictError(
                "Conflicto de idempotencia: la auditoría previa pertenece a otra conversación."
            )

        # Caso G: Registro histórico incompleto (falta alguno de los factores obligatorios)
        if not c_prop_id or not c_idem_key or not c_firma or not c_hash:
            raise IdempotencyConflictError(
                "Registro histórico incompleto: carece de factores obligatorios de autenticidad o clave. "
                "Rechazo preventivo Fail-Closed."
            )

        # Validación B-02: Verificación estricta de los 7 factores sin omisiones
        coincide_prop = (c_prop_id == propuesta_id)
        coincide_user = (str(c.usuario_id) == str(usuario.id))
        if tipo_op == "CREAR_PRODUCTO":
            coincide_prod = bool(c.id_entidad) and Producto.objects.filter(id=c.id_entidad).exists()
        else:
            coincide_prod = (str(c.id_entidad) == str(propuesta.get("producto_id")))
        coincide_op = (c.accion == tipo_op)
        coincide_key = (c_idem_key == idempotency_key)  # B-02: Comparación binaria exacta obligatoria
        coincide_firma = comparar_firma_canonica(c_firma, firma_propuesta)
        coincide_hash = comparar_hash_canonico(c_hash, huella_actual)

        # Caso F: Discrepancia en cualquiera de los factores
        if not (
            coincide_prop
            and coincide_user
            and coincide_prod
            and coincide_op
            and coincide_key
            and coincide_firma
            and coincide_hash
        ):
            raise IdempotencyConflictError(
                "Conflicto de idempotencia: la clave o propuesta ya fue utilizada para una operación "
                "con diferente contenido, producto o acción."
            )

    # Caso C: Múltiples equivalentes idénticos
    if total > 1:
        logger.warning(
            "[HAKU] Múltiples auditorías equivalentes detectadas (%d) para propuesta %s. "
            "Recuperando recibo idempotente.",
            total,
            propuesta_id,
        )

    # Caso B: Recuperación exitosa
    if tipo_op == "CREAR_PRODUCTO":
        prod = Producto.objects.filter(id=c.id_entidad).first()
        prod_nom = prod.nombre if prod else str(propuesta.get("producto_nombre", ""))
        prod_cod = prod.codigo_interno if prod else ""
        stock_tot = sum(e.stock_total for e in prod.existencias.all()) if prod else int(propuesta.get("cantidad") or 0)
        return {
            "exito": True,
            "idempotente": True,
            "estado": "EJECUTADA",
            "producto_id": str(c.id_entidad),
            "producto_codigo": prod_cod,
            "movimiento_id": None,
            "stock_nuevo": stock_tot,
            "mensaje_pantalla": f"El producto {prod_nom} ({prod_cod}) ya fue creado previamente.",
            "mensaje_verbal": "Esta operación ya había sido confirmada previamente.",
        }

    op_label = "inhabilitación" if tipo_op == "INHABILITAR_PRODUCTO" else "habilitación"
    return {
        "exito": True,
        "idempotente": True,
        "estado": "EJECUTADA",
        "movimiento_id": None,
        "mensaje_pantalla": f"La {op_label} de este producto ya fue confirmada y ejecutada previamente.",
        "mensaje_verbal": "Esta operación ya había sido confirmada previamente.",
    }


def ejecutar_confirmacion_propuesta(
    usuario: UsuarioAudacIA,
    propuesta: dict,
    idempotency_key: str | None = None,
    ip_origen: str = "127.0.0.1",
    observacion_usuario: str | None = None,
) -> dict[str, Any]:
    """Ejecuta de forma atómica y segura una propuesta pre-confirmada por el usuario.

    Reglas obligatorias:
    - transaction.atomic() + select_for_update() (sección 9)
    - Prohibición estricta de stock negativo
    - Idempotencia garantizada por propuesta e idempotency_key con detección de discrepancia de carga útil
    - Protección contra doble ejecución bajo concurrencia independiente de bloqueos de existencias
    - Auditoría inmutable con via_haku=True
    - Movimiento con via_haku=True
    """
    if not propuesta:
        raise ValidationError("No hay propuesta válida para confirmar.")

    if propuesta.get("modificacion_sin_resolver"):
        raise ValidationError(
            "La propuesta tiene una solicitud de corrección pendiente de aclarar. "
            "Por favor resuelva la corrección o cancele la propuesta antes de confirmar."
        )

    # 1. Revalidar usuario activo y autorizado
    if not usuario or not usuario.is_active:
        raise PermissionDenied("Usuario inactivo o no autorizado.")

    # 1.1 Revalidar que la propuesta pertenezca al usuario que la confirma (prevención de usurpación entre usuarios)
    propuesta_usuario_id = propuesta.get("usuario_id")
    if not propuesta_usuario_id or str(propuesta_usuario_id) != str(usuario.id):
        raise PermissionDenied("Esta propuesta pertenece a otro usuario o no cuenta con usuario_id válido.")

    # 2. Verificar expiración de la propuesta (1800 segundos / 30 minutos)
    ahora = time.time()
    if ahora > propuesta.get("expira_en", 0):
        raise ValidationError("La propuesta ha caducado por superar el límite de 30 minutos. Por favor genera una nueva solicitud.")

    tipo_op = propuesta.get("tipo_operacion")
    es_sin_cant = tipo_op in ("INHABILITAR_PRODUCTO", "HABILITAR_PRODUCTO")
    permitir_cero = tipo_op == "CREAR_PRODUCTO"
    cantidad = validar_cantidad_operativa_estricta(propuesta.get("cantidad"), obligatoria=(not es_sin_cant), permitir_cero=permitir_cero)

    propuesta_id = str(propuesta.get("id") or "").strip()
    if not propuesta_id:
        raise ValidationError("La propuesta no tiene un identificador válido.")

    # Clave canónica inmutable respaldada por restricción UNIQUE de BD (sección 9.1)
    clave_canonica = f"haku-prop-{propuesta_id}"[:100]

    # Preservar o asignar idempotency_key de cliente para correlación
    if not idempotency_key:
        idempotency_key = clave_canonica

    # 2.1 Verificación criptográfica de autenticidad e integridad (Sublote 2.1 - HMAC-SHA256)
    firma_propuesta = propuesta.get("firma_hmac") or propuesta.get("huella_payload")
    if not firma_propuesta or not isinstance(firma_propuesta, str) or not firma_propuesta.strip():
        raise ValidationError(
            "Propuesta no autenticada: la propuesta carece de firma digital del servidor."
        )

    firma_propuesta = firma_propuesta.strip()
    if not firma_propuesta.startswith(f"{VERSION_FIRMA_PROPUESTA}:"):
        if len(firma_propuesta) == 64 and all(c in "0123456789abcdefABCDEF" for c in firma_propuesta):
            raise ValidationError(
                "Firma de propuesta no autenticada o en formato legado (SHA-256 plano sin secreto). "
                "La operación fue rechazada por seguridad."
            )
        raise ValidationError(
            "Versión o formato de firma de propuesta desconocido o no soportado. "
            "La operación fue rechazada por seguridad."
        )

    partes_firma = firma_propuesta.split(":", 1)
    if len(partes_firma) != 2 or len(partes_firma[1].strip()) != 64 or not all(c in "0123456789abcdefABCDEF" for c in partes_firma[1].strip()):
        raise ValidationError(
            "Formato de firma digital inválido. La operación fue rechazada por seguridad."
        )

    if not verificar_firma_propuesta(propuesta):
        raise ValidationError(
            "Firma de propuesta inválida o alterada. El contenido de la propuesta fue modificado "
            "o no fue emitido por este servidor."
        )

    firma_hex = partes_firma[1].strip().lower()
    huella_actual = firma_hex

    # 1.2 Revalidación estricta de permisos granulares vigentes en base de datos al momento de confirmar
    if hasattr(usuario, "limpiar_permisos_cache"):
        usuario.limpiar_permisos_cache()

    permiso_requerido = None
    if tipo_op in (Movimiento.Tipo.ENTRADA, "ENTRADA"):
        permiso_requerido = "movimientos.entrada"
    elif tipo_op in (Movimiento.Tipo.PRESTAMO, "PRESTAMO"):
        permiso_requerido = "movimientos.prestamo"
    elif tipo_op in (Movimiento.Tipo.DEVOLUCION, "DEVOLUCION"):
        permiso_requerido = "movimientos.devolucion"
    elif tipo_op in (Movimiento.Tipo.TRASLADO_ENTRE, "TRASLADO", "TRASLADO_ENTRE", "TRASLADAR_PRODUCTO"):
        permiso_requerido = "movimientos.traslado_almacen"
    elif tipo_op == "HABILITAR_PRODUCTO":
        permiso_requerido = "catalogo.editar"
    elif tipo_op == "INHABILITAR_PRODUCTO":
        permiso_requerido = "catalogo.eliminar"
    elif tipo_op == "CREAR_PRODUCTO":
        permiso_requerido = "catalogo.crear"
    elif tipo_op in (Movimiento.Tipo.SALIDA, "SALIDA"):
        motivo_p = str(propuesta.get("motivo", "")).lower()
        if any(w in motivo_p for w in ("baja", "daño", "avería", "averia", "pérdida", "perdida")):
            permiso_requerido = "movimientos.baja"
        else:
            permiso_requerido = "movimientos.dotacion"

    if permiso_requerido and not usuario_tiene_permiso_operativo(usuario, permiso_requerido):
        raise PermissionDenied(f"Permiso revocado o insuficiente: se requiere '{permiso_requerido}' para confirmar esta operación.")

    def _buscar_ejecucion_previa() -> dict[str, Any] | None:
        if tipo_op in ("INHABILITAR_PRODUCTO", "HABILITAR_PRODUCTO", "CREAR_PRODUCTO"):
            return _buscar_ejecucion_previa_catalogo(
                propuesta=propuesta,
                propuesta_id=propuesta_id,
                clave_canonica=clave_canonica,
                idempotency_key=idempotency_key,
                usuario=usuario,
                tipo_op=tipo_op,
                firma_propuesta=firma_propuesta,
                huella_actual=huella_actual,
            )
        else:
            q_filtro = (
                Q(clave_idempotencia=clave_canonica) |
                Q(observacion__contains=f"propuesta_id:{propuesta_id}") |
                Q(motivo__contains=f"propuesta_id:{propuesta_id}")
            )
            if idempotency_key:
                q_filtro |= (
                    Q(clave_idempotencia=idempotency_key) |
                    Q(observacion__contains=f"[client_idempotency_key:{idempotency_key}]")
                )
            existente = Movimiento.objects.filter(q_filtro).first()
            if existente:
                # Verificación estricta de aislamiento de conversación
                cid_esperada = propuesta.get("conversacion_id")
                if cid_esperada and existente.observacion and "[conversacion_id:" in existente.observacion:
                    m_cid = re.search(r'\[conversacion_id:([a-f0-9\-]+)\]', existente.observacion)
                    if m_cid and m_cid.group(1).lower() != str(cid_esperada).strip().lower():
                        raise IdempotencyConflictError(
                            "Conflicto de idempotencia: el registro previo pertenece a otra conversación."
                        )

                huella_existente = getattr(existente, "huella_payload", "")
                if not huella_existente and existente.observacion and "[payload_hash:" in existente.observacion:
                    m = re.search(r'\[payload_hash:([a-f0-9]{64})\]', existente.observacion)
                    if m:
                        huella_existente = m.group(1)

                if huella_existente:
                    coincide = False
                    if huella_existente.startswith(("v1:", "v2:")):
                        coincide = hmac.compare_digest(huella_existente.lower(), firma_propuesta.lower())
                    else:
                        coincide = hmac.compare_digest(huella_existente.lower(), firma_hex.lower())
                    if not coincide:
                        raise IdempotencyConflictError(
                            "La clave de idempotencia ya fue utilizada para una propuesta con diferente contenido."
                        )
                else:
                    if (
                        existente.tipo != tipo_op
                        or str(existente.producto_id) != str(propuesta.get("producto_id"))
                        or existente.cantidad != int(propuesta.get("cantidad") or 0)
                    ):
                        raise IdempotencyConflictError(
                            "La clave de idempotencia coincide con un registro histórico de diferente producto, tipo o cantidad."
                        )

                return {
                    "exito": True,
                    "idempotente": True,
                    "estado": "EJECUTADA",
                    "movimiento_id": str(existente.id),
                    "stock_nuevo": existente.stock_disponible_posterior,
                    "mensaje_pantalla": f"Operación previamente confirmada (Movimiento #{existente.id}).",
                    "mensaje_verbal": "Esta operación ya había sido confirmada previamente.",
                }
        return None

    # Pre-check de idempotencia (por clave y por propuesta)
    res_previo = _buscar_ejecucion_previa()
    if res_previo:
        return res_previo

    # Lock de propuesta canónica: serializa la ejecución bajo concurrencia intra-proceso por propuesta
    lock_key = f"prop_{propuesta_id}"
    lock_propuesta = _obtener_lock_propuesta(lock_key)
    with lock_propuesta:
        # Double-check una vez adquirido el lock
        res_previo = _buscar_ejecucion_previa()
        if res_previo:
            return res_previo

        # Operaciones canónicas de catálogo (HABILITAR / INHABILITAR)
        if tipo_op == "HABILITAR_PRODUCTO":
            try:
                producto = Producto.objects.get(id=propuesta["producto_id"])
            except Producto.DoesNotExist:
                raise ValidationError("El producto asociado a la propuesta no existe.")

            from inventario.servicios.datos_maestros import habilitar_producto
            cid_prop = str(propuesta.get("conversacion_id") or "").strip()
            meta_aud = {
                "payload_firma": firma_propuesta,
                "payload_hash": huella_actual,
                "propuesta_id": propuesta_id,
                "conversacion_id": cid_prop,
                "idempotency_key": idempotency_key,
                "motivo_extra": f"[propuesta_id:{propuesta_id}] [conversacion_id:{cid_prop}] [client_idempotency_key:{idempotency_key}] [payload_firma:{firma_propuesta}] [payload_hash:{huella_actual}] desde IP {ip_origen}",
            }
            try:
                prod_actualizado = habilitar_producto(
                    producto=producto,
                    usuario=usuario,
                    ip_origen=ip_origen,
                    via_haku=True,
                    metadatos_auditoria=meta_aud,
                )
            except ValidationError as val_err:
                res_previo = _buscar_ejecucion_previa()
                if res_previo:
                    return res_previo
                raise val_err

            from .cache import HakuCacheManager
            transaction.on_commit(HakuCacheManager.invalidar_catalogo)

            msg_ok = f"El producto {prod_actualizado.nombre} ({prod_actualizado.codigo_interno}) ha sido habilitado y reactivado correctamente."
            return {
                "exito": True,
                "movimiento_id": None,
                "mensaje_pantalla": msg_ok,
                "mensaje_verbal": normalizar_para_tts(f"El producto {prod_actualizado.nombre} fue reactivado correctamente."),
            }

        if tipo_op == "INHABILITAR_PRODUCTO":
            try:
                producto = Producto.objects.get(id=propuesta["producto_id"])
            except Producto.DoesNotExist:
                raise ValidationError("El producto asociado a la propuesta no existe.")

            from inventario.servicios.datos_maestros import inhabilitar_producto
            cid_prop = str(propuesta.get("conversacion_id") or "").strip()
            meta_aud = {
                "payload_firma": firma_propuesta,
                "payload_hash": huella_actual,
                "propuesta_id": propuesta_id,
                "conversacion_id": cid_prop,
                "idempotency_key": idempotency_key,
                "motivo_extra": f"[propuesta_id:{propuesta_id}] [conversacion_id:{cid_prop}] [client_idempotency_key:{idempotency_key}] [payload_firma:{firma_propuesta}] [payload_hash:{huella_actual}] desde IP {ip_origen}",
            }
            try:
                prod_actualizado = inhabilitar_producto(
                    producto=producto,
                    usuario=usuario,
                    ip_origen=ip_origen,
                    via_haku=True,
                    metadatos_auditoria=meta_aud,
                )
            except ValidationError as val_err:
                res_previo = _buscar_ejecucion_previa()
                if res_previo:
                    return res_previo
                raise val_err

            from .cache import HakuCacheManager
            transaction.on_commit(HakuCacheManager.invalidar_catalogo)

            msg_ok = f"El producto {prod_actualizado.nombre} ({prod_actualizado.codigo_interno}) ha sido inhabilitado y archivado correctamente."
            return {
                "exito": True,
                "movimiento_id": None,
                "mensaje_pantalla": msg_ok,
                "mensaje_verbal": normalizar_para_tts(f"El producto {prod_actualizado.nombre} fue inhabilitado correctamente."),
            }

        if tipo_op == "CREAR_PRODUCTO":
            alm_id = propuesta.get("almacen_origen_id")
            cat_id = propuesta.get("categoria_id")
            ubi_id = propuesta.get("ubicacion_origen_id")
            if not alm_id or not cat_id or not ubi_id:
                raise ValidationError("La propuesta de creación carece de almacén, categoría o ubicación.")

            almacen = Almacen.objects.filter(id=alm_id, activo=True, eliminado=False).first()
            if not almacen:
                raise ValidationError("El almacén asignado a la propuesta no existe o está inactivo.")

            if not usuario.tiene_acceso_almacen(almacen):
                raise PermissionDenied(f"No cuenta con acceso al almacén '{almacen.nombre}'.")

            categoria = Categoria.objects.filter(id=cat_id, activo=True, eliminado=False).first()
            if not categoria:
                raise ValidationError("La categoría asignada a la propuesta no existe o está inactiva.")

            if categoria.almacen_principal_id != almacen.id:
                raise ValidationError(
                    f"La categoría '{categoria.nombre}' pertenece a '{categoria.almacen_principal.nombre}', "
                    f"pero la propuesta indica '{almacen.nombre}'."
                )

            ubicacion = Ubicacion.objects.filter(id=ubi_id, activo=True, eliminado=False).first()
            if not ubicacion:
                raise ValidationError("La ubicación asignada a la propuesta no existe o está inactiva.")

            if ubicacion.almacen_id != almacen.id:
                raise ValidationError(
                    f"La ubicación '{ubicacion.nombre}' pertenece a '{ubicacion.almacen.nombre}', "
                    f"pero el almacén de destino es '{almacen.nombre}'."
                )

            prod_nombre = str(propuesta.get("producto_nombre") or "").strip()
            if not prod_nombre:
                raise ValidationError("El nombre del producto no puede estar vacío.")

            if propuesta.get("cantidad") is None:
                raise ValidationError("El stock inicial es obligatorio para crear el producto.")
            stk_inicial = int(propuesta["cantidad"])
            if stk_inicial < 0:
                raise ValidationError("El stock inicial no puede ser negativo.")

            stk_minimo = propuesta.get("stock_minimo")
            if stk_minimo is None and isinstance(propuesta.get("datos_extra"), dict):
                stk_minimo = propuesta["datos_extra"].get("stock_minimo")
            if stk_minimo is None:
                raise ValidationError("El stock mínimo es obligatorio para crear el producto.")
            stk_minimo = int(stk_minimo)
            if stk_minimo < 0:
                raise ValidationError("El stock mínimo no puede ser negativo.")

            cod_barras = propuesta.get("codigo_barras") or (propuesta.get("datos_extra") or {}).get("codigo_barras") or ""
            desc = propuesta.get("descripcion") or (propuesta.get("datos_extra") or {}).get("descripcion") or ""
            es_prest_prop = (
                propuesta.get("es_prestable")
                if "es_prestable" in propuesta
                else (propuesta.get("datos_extra") or {}).get("es_prestable", False)
            )
            es_prest_prop = bool(es_prest_prop)

            from inventario.servicios.datos_maestros import crear_producto as servicio_crear_producto
            from inventario.auditoria import registrar_auditoria

            cid_prop = str(propuesta.get("conversacion_id") or "").strip()

            try:
                with transaction.atomic():
                    # Comprobación de carrera concurrente (B-01 / B-02): si otra transacción ya completó
                    # esta propuesta, abortar inmediatamente para no duplicar el producto.
                    aud_previa = (
                        Auditoria.objects.filter(
                            Q(valores_nuevos__propuesta_id=propuesta_id)
                            | Q(motivo__contains=f"[propuesta_id:{propuesta_id}]")
                        )
                        .exclude(tipo_entidad="PropuestaHaku", accion="PROPUESTA_MODIFICADA")
                        .first()
                    )
                    if aud_previa:
                        raise ValidationError(f"Propuesta {propuesta_id} ya ejecutada concurrentemente.")

                    nuevo_prod = servicio_crear_producto(
                        nombre=prod_nombre,
                        categoria=categoria,
                        stock_inicial=stk_inicial,
                        stock_minimo=stk_minimo,
                        es_prestable=es_prest_prop,
                        codigo_barras=str(cod_barras).strip(),
                        descripcion=str(desc).strip(),
                        ubicaciones=[ubicacion.id],
                        usuario=usuario,
                        ip_origen=ip_origen,
                    )

                    meta_aud = {
                        "propuesta_id": propuesta_id,
                        "conversacion_id": cid_prop,
                        "idempotency_key": idempotency_key,
                        "payload_firma": firma_propuesta,
                        "payload_hash": huella_actual,
                        "producto_codigo": nuevo_prod.codigo_interno,
                        "producto_nombre": nuevo_prod.nombre,
                        "categoria_id": categoria.id,
                        "almacen_id": almacen.id,
                        "ubicacion_id": ubicacion.id,
                        "stock_inicial": nuevo_prod.stock_inicial,
                        "stock_minimo": nuevo_prod.stock_minimo,
                        "es_prestable": nuevo_prod.es_prestable,
                    }
                    motivo_aud = (
                        f"Creación de producto vía HAKU "
                        f"[propuesta_id:{propuesta_id}] [conversacion_id:{cid_prop}] "
                        f"[client_idempotency_key:{idempotency_key}] [payload_firma:{firma_propuesta}] "
                        f"[payload_hash:{huella_actual}]"
                    )

                    registrar_auditoria(
                        usuario=usuario,
                        accion="CREAR_PRODUCTO",
                        tipo_entidad="Producto",
                        id_entidad=str(nuevo_prod.id),
                        valores_anteriores=None,
                        valores_nuevos=meta_aud,
                        motivo=motivo_aud,
                        resultado="OK",
                        via_haku=True,
                        ip_origen=ip_origen,
                    )

                    mov_id = None
                    if stk_inicial > 0:
                        mov_inicial = Movimiento.objects.filter(
                            producto=nuevo_prod,
                            tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL,
                        ).order_by("-id").first()
                        if mov_inicial:
                            mov_inicial.via_haku = True
                            mov_inicial.clave_idempotencia = clave_canonica
                            mov_inicial.huella_payload = huella_actual
                            mov_inicial.observacion = (
                                f"{mov_inicial.observacion or ''} "
                                f"[propuesta_id:{propuesta_id}] [conversacion_id:{cid_prop}] "
                                f"[client_idempotency_key:{idempotency_key}] [payload_hash:{huella_actual}]"
                            ).strip()
                            mov_inicial.save(update_fields=["via_haku", "clave_idempotencia", "huella_payload", "observacion"])
                            mov_id = str(mov_inicial.id)

            except ValidationError as val_err:
                res_previo = _buscar_ejecucion_previa()
                if res_previo:
                    return res_previo
                raise val_err

            from .cache import HakuCacheManager
            transaction.on_commit(HakuCacheManager.invalidar_catalogo)

            msg_ok = f"El producto {nuevo_prod.nombre} ({nuevo_prod.codigo_interno}) ha sido creado exitosamente en {almacen.nombre} ({ubicacion.nombre})."
            return {
                "exito": True,
                "producto_id": str(nuevo_prod.id),
                "producto_codigo": nuevo_prod.codigo_interno,
                "movimiento_id": mov_id,
                "stock_nuevo": nuevo_prod.stock_inicial,
                "mensaje_pantalla": msg_ok,
                "mensaje_verbal": normalizar_para_tts(f"El producto {nuevo_prod.nombre} fue creado exitosamente."),
            }

        intentos_lock = 12
        for intento in range(intentos_lock):
            try:
                with transaction.atomic():

                    # Bloquear producto con select_for_update() y revalidar que sigue activo para operaciones de stock
                    try:
                        producto = (
                            Producto.objects.select_for_update()
                            .get(id=propuesta["producto_id"], activo=True, archivado=False)
                        )
                    except Producto.DoesNotExist:
                        raise ValidationError("El producto asociado a la propuesta ya no existe o fue dado de baja.")

                    if tipo_op == Movimiento.Tipo.TRASLADO_ENTRE:
                        # Para traslados: validar ambos almacenes y bloquear en orden ascendente estricto por ID
                        # Previene ciclos de interbloqueo (deadlocks) en PostgreSQL ante traslados opuestos A->B y B->A
                        alm_orig_id = propuesta.get("almacen_origen_id")
                        alm_dest_id = propuesta.get("almacen_destino_id")
                        if not alm_orig_id or not alm_dest_id:
                            raise ValidationError("El traslado requiere almacén de origen y almacén de destino.")
                        if alm_orig_id == alm_dest_id:
                            raise ValidationError("El almacén de origen y destino son el mismo.")

                        id_primero, id_segundo = sorted([alm_orig_id, alm_dest_id])

                        # 1. Bloqueo consistente de Almacenes en orden ascendente
                        alms_bloqueados = {
                            alm.id: alm
                            for alm in Almacen.objects.select_for_update()
                            .filter(id__in=[id_primero, id_segundo], activo=True)
                            .order_by("id")
                        }
                        if alm_orig_id not in alms_bloqueados:
                            raise ValidationError("El almacén asociado a la propuesta ya no existe o fue desactivado.")
                        if alm_dest_id not in alms_bloqueados:
                            raise ValidationError("El almacén destino ya no existe o fue desactivado.")

                        almacen = alms_bloqueados[alm_orig_id]
                        almacen_destino = alms_bloqueados[alm_dest_id]

                        if not usuario.tiene_acceso_almacen(almacen):
                            raise PermissionDenied("No cuenta con permisos para operar en este almacén.")
                        if not usuario.tiene_acceso_almacen(almacen_destino):
                            raise PermissionDenied("No cuenta con permisos para operar en el almacén de destino.")

                        # Las existencias son gestionadas exclusivamente por el servicio canónico (DN-01)
                        existencia = Existencia.objects.filter(producto=producto, almacen=almacen).first()
                        existencia_destino = None

                    else:
                        # Bloquear almacén con select_for_update() y revalidar acceso para operaciones ordinarias
                        try:
                            almacen = Almacen.objects.select_for_update().get(id=propuesta["almacen_origen_id"], activo=True)
                        except Almacen.DoesNotExist:
                            raise ValidationError("El almacén asociado a la propuesta ya no existe o fue desactivado.")

                        if not usuario.tiene_acceso_almacen(almacen):
                            raise PermissionDenied("No cuenta con permisos para operar en este almacén.")

                        existencia, _ = Existencia.objects.select_for_update().get_or_create(
                            producto=producto,
                            almacen=almacen,
                            defaults={"stock_disponible": 0, "stock_en_prestamo": 0},
                        )

                    # Double-check atómico dentro de la transacción con fila bloqueada:
                    # Detecta si otro proceso o worker concurrente commiteó mientras esperábamos el bloqueo de fila
                    res_previo_tx = _buscar_ejecucion_previa()
                    if res_previo_tx:
                        return res_previo_tx

                    stock_ant = existencia.stock_disponible if existencia else 0
                    stock_post = stock_ant
                    movimiento = None
                    alerta_operacion = None

                    cid_prop = str(propuesta.get("conversacion_id") or "").strip()
                    obs_idemp = f"propuesta_id:{propuesta_id} [conversacion_id:{cid_prop}] [client_idempotency_key:{idempotency_key}] [payload_firma:{firma_propuesta}] [payload_hash:{huella_actual}]"
                    obs_usuario_op = (observacion_usuario if observacion_usuario is not None else propuesta.get("observacion") or "").strip()[:500]
                    obs_mov_final = f"{obs_usuario_op} | {obs_idemp}" if obs_usuario_op else obs_idemp
                    motivo_con_prop = f"{propuesta.get('motivo', '')} [propuesta_id:{propuesta_id}] [conversacion_id:{cid_prop}] [payload_firma:{firma_propuesta}] [payload_hash:{huella_actual}]".strip()

                    if tipo_op == Movimiento.Tipo.ENTRADA:
                        from inventario.servicios.operaciones import registrar_entrada
                        ubicacion_prop = None
                        ub_id = propuesta.get("ubicacion_origen_id") or propuesta.get("ubicacion_destino_id")
                        if ub_id:
                            ubicacion_prop = Ubicacion.objects.filter(id=ub_id, almacen=almacen).first()

                        existencia, movimiento = registrar_entrada(
                            producto=producto,
                            almacen=almacen,
                            cantidad=cantidad,
                            usuario=usuario,
                            ubicacion=ubicacion_prop,
                            es_empaque=bool(propuesta.get("es_empaque", False)),
                            cantidad_empaques=propuesta.get("cantidad_empaques"),
                            clave_idempotencia=clave_canonica,
                            motivo=motivo_con_prop or "Entrada vía Haku",
                            observacion=obs_mov_final,
                            ip_origen=ip_origen,
                            via_haku=True,
                            huella_payload=huella_actual,
                        )
                        stock_post = existencia.stock_disponible
                        stock_ant = stock_post - cantidad

                    elif tipo_op == Movimiento.Tipo.SALIDA:
                        from inventario.servicios.operaciones import registrar_salida
                        ubicacion_prop = None
                        ub_id = propuesta.get("ubicacion_origen_id")
                        if ub_id:
                            ubicacion_prop = Ubicacion.objects.filter(id=ub_id, almacen=almacen).first()

                        motivo_salida = propuesta.get("motivo", "").strip() or "Salida de inventario vía Haku"
                        destinatario_salida = (propuesta.get("destinatario") or "").strip()

                        existencia, movimiento, alerta = registrar_salida(
                            producto=producto,
                            almacen=almacen,
                            cantidad=cantidad,
                            motivo=motivo_con_prop or motivo_salida,
                            destinatario=destinatario_salida,
                            usuario=usuario,
                            ubicacion=ubicacion_prop,
                            clave_idempotencia=clave_canonica,
                            observacion=obs_mov_final,
                            ip_origen=ip_origen,
                            via_haku=True,
                            huella_payload=huella_actual,
                        )
                        alerta_operacion = alerta
                        stock_post = existencia.stock_disponible
                        stock_ant = stock_post + cantidad

                    elif tipo_op == Movimiento.Tipo.PRESTAMO:
                        if not getattr(producto, "es_prestable", False):
                            raise ValidationError(f"El producto '{producto.nombre}' no es prestable y no admite préstamos.")
                        from inventario.servicios.operaciones import registrar_prestamo
                        destinatario_prest = (propuesta.get("destinatario") or "").strip()
                        if not destinatario_prest:
                            raise ValidationError("El destinatario o responsable del préstamo es obligatorio.")

                        ubicacion_prop = None
                        ub_id = propuesta.get("ubicacion_origen_id")
                        if ub_id:
                            ubicacion_prop = Ubicacion.objects.filter(id=ub_id, almacen=almacen).first()

                        motivo_prestamo = propuesta.get("motivo", "").strip() or "Préstamo vía Haku"

                        prestamo_obj, movimiento, alerta = registrar_prestamo(
                            producto=producto,
                            almacen=almacen,
                            cantidad=cantidad,
                            destinatario=destinatario_prest,
                            usuario=usuario,
                            ubicacion=ubicacion_prop,
                            motivo=motivo_con_prop or motivo_prestamo,
                            observacion=obs_mov_final,
                            clave_idempotencia=clave_canonica,
                            ip_origen=ip_origen,
                            via_haku=True,
                            huella_payload=huella_actual,
                        )
                        alerta_operacion = alerta
                        existencia.refresh_from_db()
                        stock_post = existencia.stock_disponible
                        stock_ant = stock_post + cantidad

                    elif tipo_op == Movimiento.Tipo.DEVOLUCION:
                        from inventario.servicios.operaciones import registrar_devolucion_prestamo
                        prestamo_id = propuesta.get("prestamo_id")
                        if not prestamo_id:
                            raise ValidationError("La propuesta no especifique un identificador de préstamo válido.")

                        try:
                            prestamo = Prestamo.objects.select_for_update().get(id=prestamo_id)
                        except Prestamo.DoesNotExist:
                            raise ValidationError(f"El préstamo #{prestamo_id} no existe o fue eliminado.")

                        motivo_devolucion = propuesta.get("motivo", "").strip() or "Devolución vía Haku"

                        prestamo_db, devolucion_obj, movimiento = registrar_devolucion_prestamo(
                            prestamo=prestamo,
                            cantidad=cantidad,
                            usuario=usuario,
                            motivo=motivo_con_prop or motivo_devolucion,
                            observacion=obs_mov_final,
                            clave_idempotencia=clave_canonica,
                            ip_origen=ip_origen,
                            via_haku=True,
                            huella_payload=huella_actual,
                        )
                        existencia.refresh_from_db()
                        stock_post = existencia.stock_disponible
                        stock_ant = stock_post - cantidad

                    elif tipo_op == Movimiento.Tipo.TRASLADO_ENTRE:
                        from inventario.servicios.operaciones import registrar_traslado_entre_almacenes

                        ub_dest_id = propuesta.get("ubicacion_destino_id")
                        if not ub_dest_id:
                            raise ValidationError("La propuesta de traslado carece de ubicación física de destino obligatoria.")

                        ubicacion_dest = Ubicacion.objects.filter(
                            id=ub_dest_id,
                            almacen=almacen_destino,
                            activo=True,
                            eliminado=False,
                        ).first()
                        if not ubicacion_dest:
                            raise ValidationError("La ubicación destino no existe, está inactiva o no pertenece al almacén destino.")

                        cat_dest_id = propuesta.get("categoria_destino_id")
                        categoria_dest = None
                        if cat_dest_id:
                            categoria_dest = Categoria.objects.filter(id=cat_dest_id, almacen_principal=almacen_destino).first()
                            if not categoria_dest:
                                raise ValidationError("La categoría destino no pertenece al almacén destino.")

                        motivo_traslado = propuesta.get("motivo", "").strip() or "Traslado entre almacenes vía Haku"

                        ex_orig, ex_dest, movimiento = registrar_traslado_entre_almacenes(
                            producto=producto,
                            almacen_origen=almacen,
                            almacen_destino=almacen_destino,
                            cantidad=cantidad,
                            usuario=usuario,
                            ubicacion_destino=ubicacion_dest,
                            categoria_destino=categoria_dest,
                            motivo=motivo_con_prop or motivo_traslado,
                            observacion=obs_mov_final,
                            clave_idempotencia=clave_canonica,
                            ip_origen=ip_origen,
                            via_haku=True,
                            huella_payload=huella_actual,
                        )

                        stock_post = ex_dest.stock_disponible
                        stock_origen_post = ex_orig.stock_disponible if ex_orig else 0

                        # Invalidar caché de stock para ambos almacenes
                        from .cache import HakuCacheManager
                        from .diagnostico import HakuDiagnostico
                        transaction.on_commit(lambda p_id=producto.id, a_id=almacen.id: HakuCacheManager.invalidar_stock(p_id, a_id))
                        transaction.on_commit(lambda p_id=producto.id, ad_id=almacen_destino.id: HakuCacheManager.invalidar_stock(p_id, ad_id))
                        HakuDiagnostico.registrar_propuesta("confirmada")
                        HakuDiagnostico.registrar_traslado()

                        mensaje_exito = (
                            f"Traslado confirmado. Movimiento #{movimiento.id} registrado. "
                            f"{cantidad} unidades de {producto.nombre} trasladadas de {almacen.nombre} a {almacen_destino.nombre}. "
                            f"Stock en origen: {stock_origen_post}. Stock en destino: {stock_post}."
                        )
                        return {
                            "exito": True,
                            "movimiento_id": movimiento.id,
                            "stock_nuevo": stock_origen_post,
                            "mensaje_pantalla": mensaje_exito,
                            "mensaje_verbal": normalizar_para_tts(
                                f"Traslado confirmado. {cantidad} unidades trasladadas. "
                                f"Stock en origen: {stock_origen_post}. Stock en destino: {stock_post}."
                            ),
                        }

                    else:
                        raise ValidationError(f"Tipo de operación '{tipo_op}' no soportado vía Haku.")

                    if tipo_op not in (Movimiento.Tipo.ENTRADA, Movimiento.Tipo.SALIDA, Movimiento.Tipo.PRESTAMO, Movimiento.Tipo.DEVOLUCION, Movimiento.Tipo.TRASLADO_ENTRE):
                        # Registro inmutable de auditoría para operaciones ordinarias no delegadas aún
                        Auditoria.objects.create(
                            usuario=usuario,
                            accion=tipo_op,
                            tipo_entidad="Producto",
                            id_entidad=str(producto.id),
                            valores_anteriores={"stock_disponible": stock_ant},
                            valores_nuevos={"stock_disponible": stock_post, "payload_hash": huella_actual, "conversacion_id": cid_prop},
                            motivo=f"Operación confirmada vía Haku desde IP {ip_origen}: {propuesta.get('motivo', '')} [propuesta_id:{propuesta_id}] [conversacion_id:{cid_prop}] [client_idempotency:{idempotency_key}] [payload_hash:{huella_actual}]",
                            resultado="OK",
                            via_haku=True,
                        )
                    break
            except IntegrityError:
                # Colisión atómica por concurrencia multi-proceso respaldada por la restricción UNIQUE en base de datos.
                res_previo = _buscar_ejecucion_previa()
                if res_previo:
                    return res_previo
                raise
            except OperationalError as op_err:
                err_str = str(op_err).lower()
                if (
                    ("locked" in err_str or "busy" in err_str or "deadlock" in err_str)
                    and intento < intentos_lock - 1
                ):
                    time.sleep(0.08 * (intento + 1))
                    continue
                raise

        # Invalidar caché de stock mediante hook on_commit tras confirmación exitosa
        from .cache import HakuCacheManager
        from .diagnostico import HakuDiagnostico
        transaction.on_commit(lambda p_id=producto.id, a_id=almacen.id: HakuCacheManager.invalidar_stock(p_id, a_id))
        HakuDiagnostico.registrar_propuesta("confirmada")

        mensaje_exito = (
            f"Transacción confirmada con éxito. Movimiento #{movimiento.id} registrado. "
            f"Nuevo stock disponible de {producto.nombre}: {stock_post} unidades."
        )
        mensaje_verbal = normalizar_para_tts(
            f"Operación confirmada. Se registraron {cantidad} unidades. "
            f"Nuevo stock disponible: {stock_post}."
        )

        if alerta_operacion and alerta_operacion.get("nivel") in ("CRITICO", "BAJO"):
            mensaje_exito += f"\n\n{alerta_operacion['mensaje']}"
            mensaje_verbal = normalizar_para_tts(
                f"Operación confirmada. Se registraron {cantidad} unidades. "
                f"Nuevo stock disponible: {stock_post}. "
                f"Advertencia: {alerta_operacion['mensaje']}"
            )

        res_final = {
            "exito": True,
            "movimiento_id": movimiento.id,
            "stock_nuevo": stock_post,
            "mensaje_pantalla": mensaje_exito,
            "mensaje_verbal": mensaje_verbal,
        }
        if alerta_operacion and alerta_operacion.get("nivel"):
            res_final["alerta"] = alerta_operacion
        if tipo_op == Movimiento.Tipo.PRESTAMO and "prestamo_obj" in locals():
            res_final["prestamo_id"] = str(prestamo_obj.id)
        elif tipo_op == Movimiento.Tipo.DEVOLUCION and "prestamo_db" in locals():
            res_final["prestamo_id"] = str(prestamo_db.id)
            res_final["saldo_pendiente"] = prestamo_db.cantidad_pendiente_devolucion
            res_final["estado_prestamo"] = prestamo_db.estado
        return res_final
