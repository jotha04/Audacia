"""Motor de transcripción Speech-to-Text (STT) para Haku Inventario.

Usa faster-whisper (CTranslate2) de forma 100% autónoma y local con aceleración GPU (CUDA).
REGLAS ARQUITECTÓNICAS:
- Módulo confinado al dominio Haku Inventario (Integrante 4).
- NO importa ni depende de 'hacu' (Haku Visitante).
- No ejecuta inferencia de LLM ni genera diálogo conversacional general.
- Ejecuta en memoria sobre arrays numpy float32 calibrados a 16kHz mono.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
from django.conf import settings

from ..telemetria import registrar_timing

logger = logging.getLogger("integraciones.haku.audio")


def configurar_rutas_cuda_windows() -> None:
    """Configura las rutas de bibliotecas dinámicas (DLL) para NVIDIA CUDA en Windows.

    Python 3.8+ en Windows no busca DLLs en las carpetas de site-packages de forma predeterminada.
    CTranslate2 requiere 'cublas64_12.dll' y bibliotecas afines provistas por los paquetes
    oficiales nvidia-cublas-cu12 y nvidia-cuda-nvrtc-cu12, o por CUDA_PATH del sistema.
    """
    if sys.platform != "win32":
        return

    raiz_repo = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )
    rutas_candidatas = [
        # Binarios CUDA locales del proveedor de modelos del repositorio (cublas64_12.dll, etc.)
        os.path.join(raiz_repo, "cerebro_audacia", "proveedor_modelo", "bin"),
        # Entorno virtual activo (site-packages nvidia)
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cuda_nvrtc", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    ]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        rutas_candidatas.append(os.path.join(cuda_path, "bin"))

    for ruta in rutas_candidatas:
        if os.path.isdir(ruta):
            # Agregar a PATH del proceso para llamadas nativas LoadLibrary de CTranslate2
            path_actual = os.environ.get("PATH", "")
            if ruta not in path_actual:
                os.environ["PATH"] = ruta + os.pathsep + path_actual
            # Registrar directorio en el cargador de DLLs de Python 3.8+
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(ruta)
                except Exception as exc:
                    logger.debug("No se pudo agregar directorio DLL '%s': %s", ruta, exc)


class TranscripcionError(Exception):
    """Error originado en la carga o ejecución del motor STT."""
    pass


import re


class HakuTranscriber:
    """Transcriptor Speech-to-Text local basado en faster-whisper con soporte CUDA y fallback CPU."""

    _instancia: Optional[HakuTranscriber] = None
    _lock = threading.Lock()
    _vocab_cache_lock = threading.Lock()
    _vocab_cache_valor: Optional[tuple[str, str]] = None
    _vocab_cache_ts: float = 0.0
    _VOCAB_CACHE_TTL_SEGUNDOS: float = 120.0

    @classmethod
    def invalidar_cache_vocabulario(cls) -> None:
        """Invalida el prompt contextual en caché cuando cambian catálogo, almacenes o categorías."""
        with cls._vocab_cache_lock:
            cls._vocab_cache_valor = None
            cls._vocab_cache_ts = 0.0

    def __init__(
        self,
        modelo_nombre: Optional[str] = None,
        dispositivo: Optional[str] = None,
        computo: Optional[str] = None,
    ) -> None:
        self.modelo_nombre = modelo_nombre or getattr(settings, "HAKU_STT_MODEL", "small")
        self.dispositivo_solicitado = dispositivo or getattr(settings, "HAKU_STT_DEVICE", "cuda")
        self.computo_solicitado = computo or getattr(settings, "HAKU_STT_COMPUTE", "float16")
        self.dispositivo_efectivo: Optional[str] = None
        self.computo_efectivo: Optional[str] = None
        self.motivo_fallback: Optional[str] = None
        self.tiempo_carga_modelo_ms: Optional[float] = None
        self.ultimo_tiempo_inferencia_ms: Optional[float] = None
        self._model = None
        self._init_lock = threading.Lock()

    @classmethod
    def obtener_instancia(cls) -> HakuTranscriber:
        """Obtiene o crea la instancia singleton del transcriptor."""
        if cls._instancia is None:
            with cls._lock:
                if cls._instancia is None:
                    cls._instancia = cls()
        return cls._instancia

    @classmethod
    def reset_instancia(cls) -> None:
        """Reinicia la instancia (útil para pruebas unitarias)."""
        with cls._lock:
            cls._instancia = None

    def modelo_listo(self) -> bool:
        """Indica sin bloquear si el modelo faster-whisper ya está cargado en memoria."""
        return self._model is not None

    def esperar_modelo_listo(self, timeout: float = 15.0, interaction_id: Optional[str] = None) -> bool:
        """Espera hasta `timeout` segundos a que el modelo STT esté listo sin tomar HAKU_GPU_LOCK.

        Si otro hilo (como el warmup de arranque) sigue descargando o inicializando el modelo
        pasado el timeout, retorna False de forma limpia para no bloquear el lock compartido.
        """
        if self._model is not None:
            return True
        adquirido = self._init_lock.acquire(timeout=max(0.1, float(timeout)))
        if not adquirido:
            return self._model is not None
        try:
            if self._model is not None:
                return True
            self._cargar_modelo_bloqueado(interaction_id=interaction_id)
            return self._model is not None
        finally:
            self._init_lock.release()

    def _asegurar_modelo(self, interaction_id: Optional[str] = None):
        """Carga perezosa del modelo faster-whisper con fallback seguro CPU int8."""
        if self._model is not None:
            return self._model

        with self._init_lock:
            if self._model is not None:
                return self._model
            return self._cargar_modelo_bloqueado(interaction_id=interaction_id)

    def _cargar_modelo_bloqueado(self, interaction_id: Optional[str] = None):
        """Ejecuta la carga real del modelo (debe invocarse con `self._init_lock` adquirido)."""
        t0_load = time.perf_counter()

        # Configurar rutas DLL si estamos en Windows
        configurar_rutas_cuda_windows()

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscripcionError(
                "Falta la biblioteca faster-whisper. Ejecuta 'pip install faster-whisper'."
            ) from exc

        import ctranslate2

        # Intentar CUDA si está disponible y solicitado
        num_cuda = 0
        try:
            num_cuda = ctranslate2.get_cuda_device_count()
        except Exception as exc_count:
            logger.debug("Error consultando dispositivos CUDA: %s", exc_count)

        def _cargar_whisper(device: str, compute_type: str):
            modelos_intento = [self.modelo_nombre]
            if self.modelo_nombre != "base":
                modelos_intento.append("base")
            ultimo_error = None
            for candidato in modelos_intento:
                try:
                    modelo = WhisperModel(
                        candidato,
                        device=device,
                        compute_type=compute_type,
                        local_files_only=True,
                    )
                    if candidato != self.modelo_nombre:
                        logger.warning(
                            "Modelo STT '%s' no disponible localmente; usando fallback '%s'.",
                            self.modelo_nombre,
                            candidato,
                        )
                        self.modelo_nombre = candidato
                    return modelo
                except Exception:
                    try:
                        modelo = WhisperModel(
                            candidato,
                            device=device,
                            compute_type=compute_type,
                        )
                        if candidato != self.modelo_nombre:
                            logger.warning(
                                "Modelo STT '%s' no disponible; usando fallback '%s'.",
                                self.modelo_nombre,
                                candidato,
                            )
                            self.modelo_nombre = candidato
                        return modelo
                    except Exception as exc_cand:
                        ultimo_error = exc_cand
            raise ultimo_error

        if num_cuda > 0 and self.dispositivo_solicitado in ("auto", "cuda"):
            computo = "float16" if self.computo_solicitado in ("auto", "float16") else self.computo_solicitado
            try:
                logger.info(
                    "Iniciando carga de faster-whisper '%s' en GPU CUDA (dispositivos=%d, compute_type=%s)...",
                    self.modelo_nombre,
                    num_cuda,
                    computo,
                )
                modelo_cuda = _cargar_whisper("cuda", computo)

                # Verificación real obligatoria mediante inferencia con array mínimo (0.1s)
                # Un array de 1s (16000 muestras) causaba ~34.5s de primera inferencia.
                # 1600 muestras son suficientes para validar la pipeline CUDA.
                dummy = np.zeros(1600, dtype=np.float32)
                segs, _ = modelo_cuda.transcribe(dummy, language="es")
                _ = list(segs)

                self.dispositivo_efectivo = "cuda"
                self.computo_efectivo = computo
                self.motivo_fallback = None
                self._model = modelo_cuda
                self.tiempo_carga_modelo_ms = (time.perf_counter() - t0_load) * 1000.0
                registrar_timing(interaction_id, "stt_model_load", self.tiempo_carga_modelo_ms, device="cuda", compute=computo)
                logger.info(
                    "Modelo faster-whisper verificado y activo en GPU CUDA (device=cuda, compute_type=%s, carga=%.2f ms).",
                    computo,
                    self.tiempo_carga_modelo_ms,
                )
                return self._model
            except Exception as exc_cuda:
                self.motivo_fallback = str(exc_cuda)
                logger.warning(
                    "Aceleración CUDA solicitada pero no operativa (%s). Activando fallback controlado a CPU int8.",
                    exc_cuda,
                )

        # Fallback a CPU int8
        try:
            logger.info(
                "Cargando faster-whisper '%s' en CPU (int8)%s...",
                self.modelo_nombre,
                f" [Motivo fallback: {self.motivo_fallback}]" if self.motivo_fallback else "",
            )
            self._model = _cargar_whisper("cpu", "int8")
            self.dispositivo_efectivo = "cpu"
            self.computo_efectivo = "int8"
            self.tiempo_carga_modelo_ms = (time.perf_counter() - t0_load) * 1000.0
            registrar_timing(interaction_id, "stt_model_load", self.tiempo_carga_modelo_ms, device="cpu", compute="int8")
            logger.info(
                "Modelo faster-whisper cargado con éxito en CPU (device=cpu, compute_type=int8, carga=%.2f ms).",
                self.tiempo_carga_modelo_ms,
            )
            return self._model
        except Exception as exc_cpu:
            raise TranscripcionError(
                f"No se pudo cargar el modelo faster-whisper '{self.modelo_nombre}': {exc_cpu}"
            ) from exc_cpu

    def es_cuda(self) -> bool:
        """Indica si el transcriptor activo está corriendo sobre GPU CUDA."""
        self._asegurar_modelo()
        return self.dispositivo_efectivo == "cuda"

    _RE_ALUCINACION_STT = re.compile(
        r"^(?:"
        r"[\[\(\*].+[\]\)\*]|"
        r"h[ae][ck]u\s+inventario\s*:.*|"
        r"prefijos\s+de\s+c[oó]digo\s+vigentes\s*:.*|"
        r"(?:subt[ií]tulos|transcripci[oó]n)\s+(?:realizados\s+)?por\b.*|"
        r".*\bamara\.org\b.*|"
        r"suscr[ií]b(?:ete|anse|ir(?:se)?)\b.*|"
        r"gracias\s+por\s+(?:ver|mirar|escuchar|su\s+atenci[oó]n)\b.*|"
        r"dale\s+a?\s*like\b.*|"
        r"activa\s+la\s+campanita\b.*"
        r")$",
        re.IGNORECASE,
    )
    _RE_PREFIJOS_CRUDOS = re.compile(r"^(?:[A-Z]{2,5}(?:\s*,\s*|\s+)){2,}[A-Z]{2,5}\.?$")
    _RE_ALUCINACION_OUTRO_BAJA_CONFIANZA = re.compile(
        r"^(?:"
        r"eso\s+es\s+todo|"
        r"es\s+todo|"
        r"todo\s+eso|"
        r"y\s+eso\s+es\s+todo|"
        r"bueno\s*,?\s*eso\s+es\s+todo|"
        r"muchas\s+gracias|"
        r"gracias|"
        r"hasta\s+(?:luego|pronto|la\s+pr[oó]xima)|"
        r"nos\s+vemos|"
        r"m[uú]sica|"
        r"aplausos|"
        r"am[eé]n"
        r")$",
        re.IGNORECASE,
    )
    _RE_STT_STOCK = re.compile(r"\b(?:es\s+to[ck]k?|estoc[oa]?|est[oó]k|est[oó]c|estock|stoc|stok)\b", re.IGNORECASE)
    _RE_STT_AGOTADO = re.compile(r"\babotad([oa]s?)\b", re.IGNORECASE)
    _RE_CUENTA_SERVICIO = re.compile(
        r"\b(?:administrador|admin|operador|consulta|soporte|sistema|usuario|qa|test|prueba|root|bot|servicio)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _es_segmento_alucinacion_stt(
        cls,
        segmento,
        texto_limpio: str,
        duracion_audio_seg: Optional[float] = None,
    ) -> bool:
        """Detecta segmentos espurios, silencios o ecos del prompt emitidos por Whisper."""
        if not texto_limpio:
            return True
        sin_puntuacion = re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]+", "", texto_limpio)
        if not sin_puntuacion:
            return True

        no_speech = getattr(segmento, "no_speech_prob", None)
        avg_logprob = getattr(segmento, "avg_logprob", None)
        compression = getattr(segmento, "compression_ratio", None)
        if isinstance(no_speech, (int, float)) and no_speech >= 0.72:
            return True
        if isinstance(avg_logprob, (int, float)):
            if avg_logprob <= -1.35:
                return True
            if avg_logprob <= -1.0 and isinstance(no_speech, (int, float)) and no_speech >= 0.45:
                return True
        if isinstance(compression, (int, float)) and compression >= 2.6:
            return True

        t_strip = texto_limpio.strip(" \t\r\n¿?¡!.,;:")
        if cls._RE_ALUCINACION_STT.match(texto_limpio.strip()) or cls._RE_ALUCINACION_STT.match(t_strip):
            return True
        if cls._RE_PREFIJOS_CRUDOS.match(texto_limpio.strip()):
            return True
        # Filtrar cierres fantasma típicos de Whisper en español ('Eso es todo.', 'Gracias.')
        # solo cuando la confianza acústica es débil o el clip es demasiado corto para esa frase.
        if cls._RE_ALUCINACION_OUTRO_BAJA_CONFIANZA.match(t_strip):
            if isinstance(no_speech, (int, float)) and no_speech >= 0.30:
                return True
            if isinstance(avg_logprob, (int, float)) and avg_logprob <= -0.65:
                return True
            if duracion_audio_seg is not None and duracion_audio_seg < 0.70 and len(t_strip.split()) >= 3:
                return True
        return False

    @classmethod
    def _formatear_y_desduplicar_segmentos(
        cls,
        segmentos,
        duracion_audio_seg: Optional[float] = None,
    ) -> str:
        """Formatea los segmentos de Whisper y descarta alucinaciones, silencios y bucles."""
        textos = []
        ultimo_norm = ""
        for s in segmentos:
            t = s.text.strip() if hasattr(s, "text") else str(s).strip()
            if cls._es_segmento_alucinacion_stt(s, t, duracion_audio_seg=duracion_audio_seg):
                continue
            # Normalizar para detectar repeticiones consecutivas emitidas por Whisper
            t_norm = re.sub(r"[¿?¡!.,:;]", "", t).lower().strip()
            if t_norm and t_norm == ultimo_norm:
                continue
            textos.append(t)
            ultimo_norm = t_norm

        texto_unido = " ".join(textos).strip()
        if not texto_unido:
            return ""
        # Colapsar oraciones idénticas consecutivas producidas por alucinaciones cíclicas
        patron_rep = re.compile(r"(\b.+?[.?!])\s+\1", re.IGNORECASE)
        for _ in range(3):
            if patron_rep.search(texto_unido):
                texto_unido = patron_rep.sub(r"\1", texto_unido)
            else:
                break
        # Colapsar bucles repetitivos separados por coma (ej. "sí, sí, sí, sí" o "el mismo, el mismo, el mismo")
        patron_comas = re.compile(r"\b([^,.;!?]{1,40})(?:\s*,\s*\1){2,}", re.IGNORECASE)
        for _ in range(3):
            if patron_comas.search(texto_unido):
                texto_unido = patron_comas.sub(r"\1", texto_unido)
            else:
                break
        return texto_unido.strip()

    @classmethod
    def _normalizar_lexico_stt(cls, texto: str) -> str:
        """Corrige grafías fonéticas frecuentes de Whisper para términos operativos (p. ej. 'estoc' -> 'stock')."""
        if not texto:
            return texto
        texto = cls._RE_STT_STOCK.sub(
            lambda m: "Stock" if m.group(0)[:1].isupper() else "stock",
            texto,
        )
        texto = cls._RE_STT_AGOTADO.sub(
            lambda m: ("Agotad" if m.group(0)[:1].isupper() else "agotad") + m.group(1).lower(),
            texto,
        )
        return texto

    @staticmethod
    def _normalizar_referencias_catalogo(texto: str) -> str:
        """Corrige referencias habladas solo con equivalencias únicas del catálogo vivo."""
        try:
            from ..agente.herramientas import normalizar_codigos_hablados_en_texto
            return normalizar_codigos_hablados_en_texto(texto)
        except Exception:
            logger.debug("No fue posible normalizar referencias dictadas contra el catálogo", exc_info=True)
            return texto

    @staticmethod
    def _obtener_prefijos_catalogo() -> set[str]:
        """Obtiene prefijos de código del catálogo sin realizar consultas adicionales a la BD.

        Prioriza la caché en RAM (HakuCacheManager) y recurre a la función estándar
        obtener_prefijos_catalogo_dinamicos() con manejo de excepciones y fallback seguro.
        """
        try:
            from ..cache import HakuCacheManager
            cat = HakuCacheManager.obtener_catalogo()
            if cat:
                return {p["codigo_interno"].split("-")[0].upper() for p in cat if "-" in p.get("codigo_interno", "")}
        except Exception:
            pass

        try:
            from ..agente.herramientas import obtener_prefijos_catalogo_dinamicos
            return obtener_prefijos_catalogo_dinamicos()
        except Exception:
            pass

        return set()

    @staticmethod
    def _obtener_almacenes_catalogo() -> list[str]:
        """Obtiene nombres de almacenes activos vigentes sin hardcodear ninguno."""
        try:
            from inventario.models import Almacen
            return sorted(
                str(nombre).strip()
                for nombre in Almacen.objects.filter(activo=True, eliminado=False).values_list("nombre", flat=True)[:10]
                if str(nombre).strip()
            )
        except Exception:
            return []

    @staticmethod
    def _obtener_categorias_catalogo() -> list[str]:
        """Obtiene nombres de categorías activas vigentes sin hardcodear ninguna."""
        try:
            from inventario.models import Categoria
            return sorted(
                str(nombre).strip()
                for nombre in Categoria.objects.filter(activo=True, eliminado=False).values_list("nombre", flat=True)[:15]
                if str(nombre).strip()
            )
        except Exception:
            return []

    @classmethod
    def _obtener_destinatarios_activos(cls) -> list[str]:
        """Obtiene personas reales del inventario (priorizando préstamos activos) sin cuentas de servicio/QA."""
        vistos: set[str] = set()
        nombres: list[str] = []

        def _agregar(valor: str) -> None:
            nom = str(valor or "").strip()
            if not nom or len(nom) < 3:
                return
            if cls._RE_CUENTA_SERVICIO.search(nom):
                return
            clave = nom.lower()
            if clave not in vistos and len(nombres) < 8:
                vistos.add(clave)
                nombres.append(nom)

        try:
            from inventario.models import Prestamo
            for dest in (
                Prestamo.objects.filter(
                    estado__in=[Prestamo.Estado.ABIERTO, Prestamo.Estado.DEVUELTO_PARCIAL]
                )
                .exclude(destinatario__isnull=True)
                .exclude(destinatario="")
                .values_list("destinatario", flat=True)[:12]
            ):
                _agregar(dest)
        except Exception:
            pass

        if len(nombres) < 8:
            try:
                from inventario.models import Movimiento
                for dest in (
                    Movimiento.objects.exclude(destinatario__isnull=True)
                    .exclude(destinatario="")
                    .order_by("-fecha_hora")
                    .values_list("destinatario", flat=True)[:20]
                ):
                    _agregar(dest)
            except Exception:
                pass

        if len(nombres) < 8:
            try:
                from cuentas.models import UsuarioAudacIA
                for u in UsuarioAudacIA.objects.filter(is_active=True, eliminado_logicamente=False)[:15]:
                    nom_u = (getattr(u, "nombre_completo", "") or u.get_full_name() or "").strip()
                    if " " in nom_u:
                        _agregar(nom_u)
            except Exception:
                pass

        return nombres

    def construir_vocabulario_contextual(self, *, forzar_recarga: bool = False) -> tuple[str, str]:
        """Construye initial_prompt compacto (<= 224 tokens de Whisper) y hotwords del catálogo vivo.

        Cachea el resultado en RAM (TTL 120s, invalidado por señales de catálogo) para evitar
        consultas ORM repetidas en cada audio dictado. Coloca los prefijos al inicio y el
        vocabulario operativo crítico al final para que Whisper nunca trunque términos esenciales.
        """
        ahora = time.monotonic()
        cls = type(self)
        if not forzar_recarga:
            with cls._vocab_cache_lock:
                if (
                    cls._vocab_cache_valor is not None
                    and (ahora - cls._vocab_cache_ts) <= cls._VOCAB_CACHE_TTL_SEGUNDOS
                ):
                    return cls._vocab_cache_valor
        prefijos_disponibles = sorted(self._obtener_prefijos_catalogo())
        limite_configurado = os.environ.get("HAKU_STT_MAX_PREFIX_HINTS")
        if limite_configurado is None or not str(limite_configurado).strip():
            prefijos = prefijos_disponibles
        else:
            try:
                limite_prefijos = max(0, int(limite_configurado))
            except (TypeError, ValueError):
                limite_prefijos = len(prefijos_disponibles)
            prefijos = prefijos_disponibles[:limite_prefijos]
        prefijos_contextuales = ", ".join(prefijos)
        almacenes_vigentes = self._obtener_almacenes_catalogo()
        almacenes_txt = ", ".join(almacenes_vigentes)
        categorias_vigentes = self._obtener_categorias_catalogo()
        destinatarios_activos = self._obtener_destinatarios_activos()[:4]
        destinatarios_txt = ", ".join(destinatarios_activos)

        cola_dominio = (
            "Hacu Inventario: almacén, bodega, estante, stock, existencias, disponibles, "
            "agotados, mínimo, préstamo, devolución, entrada, salida, traslado, dotación, "
            "movimientos, prestable, consumible, conectividad, confirmar, cancelar."
        )
        partes_fijas = []
        if prefijos_contextuales:
            partes_fijas.append(f"Prefijos de código vigentes: {prefijos_contextuales}.")
        if almacenes_txt:
            partes_fijas.append(f"Almacenes: {almacenes_txt}.")
        if destinatarios_txt:
            partes_fijas.append(f"Responsables: {destinatarios_txt}.")
        partes_fijas.append(cola_dominio)

        # Ajustar categorías al presupuesto dinámico restante para mantener el prompt <= 220 tokens (~775 chars)
        base_len = len(" ".join(partes_fijas))
        presupuesto_cats = max(0, 775 - base_len - len(" Categorías: ."))
        cats_acotadas: list[str] = []
        largo_cats = 0
        for cat in categorias_vigentes:
            extra = len(cat) + (2 if cats_acotadas else 0)
            if largo_cats + extra > presupuesto_cats:
                break
            cats_acotadas.append(cat)
            largo_cats += extra
        categorias_txt = ", ".join(cats_acotadas)

        partes_prompt = []
        if prefijos_contextuales:
            partes_prompt.append(f"Prefijos de código vigentes: {prefijos_contextuales}.")
        if almacenes_txt:
            partes_prompt.append(f"Almacenes: {almacenes_txt}.")
        if categorias_txt:
            partes_prompt.append(f"Categorías: {categorias_txt}.")
        if destinatarios_txt:
            partes_prompt.append(f"Responsables: {destinatarios_txt}.")
        partes_prompt.append(cola_dominio)
        prompt_vocabulario = " ".join(partes_prompt)

        hotwords_vocabulario = (
            f"Hacu inventario almacén existencias disponibles stock mínimo agotados búscame buscar "
            f"muéstrame lista productos préstamos activos prestable consumible conectividad movimientos entradas salidas "
            f"devoluciones hubo semana mes producto "
            f"{' '.join(almacenes_vigentes)} {' '.join(prefijos)}"
        ).strip()
        resultado_vocab = (prompt_vocabulario, hotwords_vocabulario)
        with cls._vocab_cache_lock:
            cls._vocab_cache_valor = resultado_vocab
            cls._vocab_cache_ts = time.monotonic()
        return resultado_vocab

    @staticmethod
    def _filtrar_hotwords_para_inferencia(hotwords: str) -> str:
        """Retira tokens cortos en MAYÚSCULAS puras (prefijos de código) del parámetro hotwords.

        CTranslate2/Whisper aplica boost a cada subword BPE dentro de hotwords; incluir decenas
        de prefijos en mayúsculas sesga la búsqueda hacia texto en MAYÚSCULAS y palabras rotas.
        Los prefijos ya viajan de forma natural en initial_prompt.
        """
        tokens = [
            tok for tok in (hotwords or "").split()
            if not (2 <= len(tok) <= 6 and tok.isalpha() and tok.isupper())
        ]
        return " ".join(tokens).strip()

    def transcribir(
        self,
        samples_16k: np.ndarray,
        idioma: str = "es",
        beam_size: int = 5,
        interaction_id: Optional[str] = None,
        ya_filtrado_vad: bool = False,
    ) -> str:
        """Transcribe un array de audio float32 a 16.000 Hz mono.

        Retorna texto limpio (sin espacios residuales ni alucinaciones de audio corto).
        """
        if samples_16k is None or len(samples_16k) == 0:
            return ""

        # Descarta audio inferior a 150 ms (clic o ruido ultracorto)
        if len(samples_16k) < 16000 * 0.15:
            return ""

        modelo = self._asegurar_modelo(interaction_id=interaction_id)

        prompt_vocabulario, _hotwords_vocabulario = self.construir_vocabulario_contextual()

        duracion_seg = len(samples_16k) / 16000.0
        # Colchón digital de 150 ms a cada extremo para no cortar ataques/sílabas finales rápidas.
        pad_muestras = int(16000 * 0.15)
        audio_entrada = np.pad(
            samples_16k.astype(np.float32),
            (pad_muestras, pad_muestras),
            mode="constant",
        )

        # Evitar doble recorte Silero VAD en clips cortos ya filtrados por AudioPipeline;
        # mantener VAD de Whisper activo en llamadas directas o grabaciones largas (>15s).
        usar_vad_whisper = (not ya_filtrado_vad) or (duracion_seg > 15.0)

        # Usar únicamente initial_prompt (sin hotwords de CTranslate2): el boost BPE de
        # hotwords en habla rápida introduce sustituciones artificiales (p. ej. «búsqueme»
        # sobre «hubo este mes»), mientras initial_prompt guía el dominio sin distorsionar.
        kwargs_transcribe = {
            "language": idioma,
            "task": "transcribe",
            "beam_size": beam_size,
            "temperature": [0.0, 0.2],
            "initial_prompt": prompt_vocabulario,
            "vad_filter": usar_vad_whisper,
            "vad_parameters": {
                "threshold": 0.38,
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 400,
            },
            "condition_on_previous_text": False,
            "repetition_penalty": 1.0,
            "compression_ratio_threshold": 2.4,
            "no_speech_threshold": 0.6,
        }

        t0_infer = time.perf_counter()
        try:
            segmentos, _info = modelo.transcribe(
                audio_entrada,
                **kwargs_transcribe,
            )
            resultado = self._formatear_y_desduplicar_segmentos(segmentos, duracion_audio_seg=duracion_seg)
            resultado = self._normalizar_referencias_catalogo(resultado)
            resultado = self._normalizar_lexico_stt(resultado)
            dt_infer = (time.perf_counter() - t0_infer) * 1000.0
            self.ultimo_tiempo_inferencia_ms = dt_infer
            registrar_timing(interaction_id, "stt_inference", dt_infer, device=self.dispositivo_efectivo)
            return resultado
        except RuntimeError as exc:
            if "cublas" in str(exc).lower() or "cuda" in str(exc).lower():
                logger.error(
                    "Fallo de GPU/CUDA en inferencia runtime (%s). Recayendo en CPU int8.",
                    exc,
                )
                self.dispositivo_efectivo = "cpu"
                self.computo_efectivo = "int8"
                self.motivo_fallback = str(exc)
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.modelo_nombre, device="cpu", compute_type="int8")
                segs, _ = self._model.transcribe(
                    audio_entrada,
                    **kwargs_transcribe,
                )
                resultado = self._formatear_y_desduplicar_segmentos(segs, duracion_audio_seg=duracion_seg)
                resultado = self._normalizar_referencias_catalogo(resultado)
                resultado = self._normalizar_lexico_stt(resultado)
                dt_infer = (time.perf_counter() - t0_infer) * 1000.0
                self.ultimo_tiempo_inferencia_ms = dt_infer
                registrar_timing(interaction_id, "stt_inference", dt_infer, device="cpu", fallback=True)
                return resultado
            raise TranscripcionError(f"Error en inferencia Whisper: {exc}") from exc
        except Exception as exc:
            logger.exception("Fallo durante la transcripción en faster-whisper: %s", exc)
            raise TranscripcionError(f"Error en inferencia Whisper: {exc}") from exc
