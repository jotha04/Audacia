"""Motor de Interpretación Semántica Oficial para HAKU Inventario.

Conecta el carril conversacional de inventario con el Cerebro Central de AudacIA
(cerebro_audacia.conexion.cliente.obtener_cliente_cerebro_central) preservando:
1. Arquitectura Híbrida: Si el LLM está activo en llama-server, aprovecha su capacidad
   para interpretar lenguaje natural flexible mediante esquema JSON estructurado.
2. Marco Semántico Resiliente (Zero-LLM): Si el LLM está temporalmente fuera de línea
   (ej. suite de pruebas automatizadas o arranque en frío), un motor semántico desacoplado
   resuelve intenciones, ranuras, alternativas ("el otro"), modificaciones de propuesta
   y consultas de atributos sin depender de listas cerradas de frases preestablecidas.
3. Seguridad Transaccional: El intérprete NUNCA ejecuta mutaciones ni escribe en la base de datos.
   Solo extrae y clasifica intenciones y parámetros para que los servicios autorizados
   gobiernen las reglas de negocio, confirmaciones e idempotencia.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("integraciones.haku.semantica")


@dataclass
class ExtraccionSemantica:
    """Resultado estructurado emitido por el Intérprete Semántico."""

    intencion: str
    producto: Optional[str] = None
    almacen: Optional[str] = None
    origen: Optional[str] = None
    destino: Optional[str] = None
    ubicacion: Optional[str] = None
    cantidad: Optional[int] = None
    destinatario: Optional[str] = None
    atributo: Optional[str] = None  # "categoria" | "stock_minimo" | "ubicacion" | "codigo" | "prestados" | "historial"
    tipo_operacion: Optional[str] = None  # "prestamo" | "entrada" | "salida" | "devolucion" | "traslado"
    campo_modificar: Optional[str] = None  # "destinatario" | "cantidad" | "almacen" | "motivo"
    nuevo_valor: Optional[str] = None
    filtro_reposicion: Optional[str] = None  # "agotados" | "bajo_stock" | "todos"
    filtro_movimiento: Optional[str] = None  # "entradas" | "salidas" | "prestamos" | "devoluciones" | "traslados" | "todos"
    periodo: Optional[str] = None  # "hoy" | "ayer" | "semana" | "mes"
    confianza: float = 1.0
    fuente: str = "reglas"  # "llm" | "marco_semantico"
    datos_extra: Dict[str, Any] = field(default_factory=dict)


def _limpiar_str(val: Any) -> Optional[str]:
    """Normaliza un valor devuelto por el LLM a string limpio o None."""
    if val is None or isinstance(val, bool):
        return None
    val_str = str(val).strip()
    if not val_str or val_str.casefold() in (
        "null", "none", "n/a", "undefined", "ninguno", "ninguna",
    ):
        return None
    return val_str


def _limpiar_entero(val: Any) -> Optional[int]:
    """Normaliza un campo numérico devuelto por el LLM a int o None."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val if val >= 0 else None
    if isinstance(val, float):
        return int(val) if val >= 0 else None
    if isinstance(val, str):
        val_clean = val.strip()
        if val_clean.isdigit():
            return int(val_clean)
    return None


def _obtener_cliente_cerebro():
    """Obtiene de forma segura el cliente del Cerebro Central de AudacIA si está disponible."""
    try:
        repo_root = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from cerebro_audacia.conexion import obtener_cliente_cerebro_central
        return obtener_cliente_cerebro_central()
    except Exception as e:
        logger.debug("No se pudo cargar cliente de Cerebro Central: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────
# FAST-PATH SOCIAL DETERMINISTA (OPT-01)
# ─────────────────────────────────────────────────────────────────────────

_PALABRAS_OPERATIVAS_FAST_PATH = re.compile(
    r"\b("
    r"stock|producto|productos|art[ií]culo|art[ií]culos|cat[aá]logo|cat[aá]logos|"
    r"almac[eé]n|almacenes|bodega|bodegas|estante|estantes|caj[oó]n|cajones|repisa|repisas|"
    r"vitrina|vitrinas|ubicaci[oó]n|ubicaciones|posici[oó]n|posiciones|"
    r"existencia|existencias|disponible|disponibles|agotado|agotados|m[ií]nimo|"
    r"equipo|equipos|herramienta|herramientas|"
    r"prestar|pr[eé]stamo|pr[eé]stamos|devolver|devoluci[oó]n|devoluciones|"
    r"trasladar|traslado|traslados|entrada|entradas|salida|salidas|ingreso|ingresos|"
    r"mover|cambiar|c[aá]mbialo|c[aá]mbiala|registrar|guardar|sacar|entregar|"
    r"cu[aá]nto|cu[aá]ntos|cu[aá]nta|cu[aá]ntas|qu[eé]\s+hay|qu[eé]\s+tenemos|d[oó]nde|"
    r"s[ií]|no|cancela|cancelar|confirma|confirmar|proceder|otro|otra|segundo|segunda|"
    r"dale|listo\b(?!\s*,\s*gracias)|ese\s+mismo|devuelve"
    r")\b",
    re.IGNORECASE,
)

_PATRON_SALUDO_PURO = re.compile(
    r"^(?:(?:hola(?:\s+ha[ck]u|\s+audacia|\s+amigo)?\s*,?\s*)?(?:buenos?\s+d[ií]as?|buenas?\s+tardes?|buenas?\s+noches?|buen\s+d[ií]a|qu[eé]\s+tal)(?:\s+ha[ck]u|\s+a\s+todos|\s+amigo)?|"
    r"hola(?:\s+ha[ck]u|\s+audacia|\s+amigo)?|"
    r"buenas(?:\s+buenas|\s+ha[ck]u)?|"
    r"saludos(?:\s+cordiales|\s+ha[ck]u)?|"
    r"qu[eé]\s+hubo(?:\s+ha[ck]u)?|quihubo(?:\s+ha[ck]u)?)$",
    re.IGNORECASE,
)

_PATRON_AGRADECIMIENTO_PURO = re.compile(
    r"^(?:muchas\s+gracias(?:\s+ha[ck]u|\s+por\s+la\s+ayuda|\s+por\s+todo)?|"
    r"gracias(?:\s+ha[ck]u|\s+amigo|\s+totales|\s+por\s+todo|\s+por\s+la\s+ayuda)?|"
    r"te\s+(?:lo\s+)?agradezco(?:\s+mucho)?|"
    r"muy\s+amable(?:\s+gracias)?|"
    r"listo\s*,?\s*gracias|"
    r"vale\s*,?\s*gracias|"
    r"mil\s+gracias|"
    r"agradecido|"
    r"adi[oó]s(?:\s+ha[ck]u)?|"
    r"hasta\s+luego(?:\s+ha[ck]u)?|"
    r"hasta\s+pronto(?:\s+ha[ck]u)?|"
    r"hasta\s+mañana(?:\s+ha[ck]u)?|"
    r"chao(?:\s+ha[ck]u|\s+pues)?|"
    r"nos\s+vemos(?:\s+luego)?|"
    r"que\s+tengas?\s+buen\s+d[ií]a|"
    r"que\s+est[eé]s?\s+bien)$",
    re.IGNORECASE,
)


_CLIENTE_POR_DEFECTO = object()


class InterpreteSemanticoHaku:
    """Intérprete semántico híbrido con integración a Cerebro Central y fallback robusto."""

    @classmethod
    def _evaluar_fast_path_social(cls, texto: str) -> Optional[ExtraccionSemantica]:
        """OPT-01: Evalúa de forma determinista O(1) si un mensaje es exclusivamente social.

        Garantiza:
        1. Cero invocación de GPU/LLM para saludos, agradecimientos y despedidas puras.
        2. Rechazo inmediato si contiene términos operativos o contextuales de inventario.
        3. Coincidencia estricta sobre la frase completa sin ambigüedades.
        """
        if not texto:
            return None

        t_clean = re.sub(r"[^\w\sñáéíóúÁÉÍÓÚ]", " ", texto.strip())
        t_clean = re.sub(r"\s+", " ", t_clean).strip()
        if not t_clean:
            return None

        # Limpiar vocativos coloquiales/costeños de apertura antes de evaluar saludo puro
        t_sin_voc = re.sub(
            r"^(?:(?:oye|oiga|eche|aj[aá]|hey|ey|mira)\s+(?:mi\s+llave|llave|compae|compadre|compa|cuadro|vale|mijo|ombe)|"
            r"(?:mi\s+llave|compae|compadre|compa|mijo|ombe))\s+",
            "",
            t_clean,
            flags=re.IGNORECASE,
        ).strip()
        if t_sin_voc:
            t_clean = t_sin_voc

        # Si contiene cualquier término operativo de inventario o transacción, abortar Fast-Path
        if _PALABRAS_OPERATIVAS_FAST_PATH.search(t_clean):
            return None

        # Saludos puros
        if _PATRON_SALUDO_PURO.match(t_clean):
            return ExtraccionSemantica(
                intencion="saludo",
                confianza=1.0,
                fuente="marco_semantico",
                datos_extra={"mecanismo": "fast_path"},
            )

        # Agradecimientos y despedidas puras
        if _PATRON_AGRADECIMIENTO_PURO.match(t_clean):
            return ExtraccionSemantica(
                intencion="agradecimiento",
                confianza=1.0,
                fuente="marco_semantico",
                datos_extra={"mecanismo": "fast_path"},
            )

        return None

    @classmethod
    def interpretar(
        cls,
        mensaje: str,
        contexto_operativo: Any = None,
        usuario: Any = None,
        cliente_llm: Any = _CLIENTE_POR_DEFECTO,
    ) -> ExtraccionSemantica:
        """Punto de entrada principal para análisis semántico de un mensaje de inventario."""
        texto = mensaje.strip() if mensaje else ""
        if not texto:
            return ExtraccionSemantica(intencion="desconocido", confianza=0.0)

        # Normalización fonética y técnica de códigos y siglas antes de la interpretación
        from .siglas import NormalizadorSiglasTecnicas
        texto = NormalizadorSiglasTecnicas.normalizar(texto)

        # 1. Prioridad Transaccional de Seguridad: Si hay una propuesta activa pendiente,
        # resolver con alta prioridad si el usuario confirma, cancela o modifica campos.
        propuesta_activa = getattr(contexto_operativo, "propuesta_activa", None) if contexto_operativo else None
        if propuesta_activa:
            res_propuesta = cls._analizar_en_contexto_propuesta(texto, propuesta_activa)
            if res_propuesta:
                return res_propuesta

        # 2. OPT-01: Fast-Path Determinista para mensajes sociales puros e inequívocos (< 0.05 ms)
        res_fast_path = cls._evaluar_fast_path_social(texto)
        if res_fast_path is not None:
            return res_fast_path

        # 3. Intentar inferencia semántica con el LLM del Cerebro Central si está disponible
        cliente = _obtener_cliente_cerebro() if cliente_llm is _CLIENTE_POR_DEFECTO else cliente_llm
        if cliente is not None:
            try:
                salud = cliente.comprobar_salud()
                if salud.get("disponible"):
                    extraccion = cls._interpretar_con_llm(texto, cliente, contexto_operativo)
                    if extraccion and extraccion.intencion != "desconocido":
                        return extraccion
            except Exception as e:
                logger.warning("Fallo al consultar Cerebro Central LLM: %s. Aplicando marco semántico.", e)

        # 4. Fallback: Marco semántico determinista de alta generalización (Zero-LLM)
        return cls._interpretar_marcos_semanticos(texto, contexto_operativo, usuario)

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN TRANSACCIONAL: GESTIÓN DE PROPUESTAS PENDIENTES
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _analizar_en_contexto_propuesta(
        cls,
        texto: str,
        propuesta: Dict[str, Any],
    ) -> Optional[ExtraccionSemantica]:
        """Evalúa un mensaje emitido mientras una propuesta transaccional está en espera de confirmación."""
        t_raw = texto.strip()
        t_lower = t_raw.lower()
        t_clean = re.sub(r"[^\w\s]", " ", t_lower)
        t_clean = re.sub(r"\s+", " ", t_clean).strip()

        # A. Confirmación explícita
        patrones_confirmacion = [
            r"^(s[ií]|si\s+por\s+favor|s[ií]\s+confirmo|confirmo|confirma|confirmad|confirmar|confirmado|dale|de\s+acuerdo|proceder|ejecutar|adelante|claro|por\s+supuesto|correcto|hacerlo|hazlo|procede)$",
            r"\b(s[ií]\s+confirmo|confirmo\s+(?:(?:la|esa|esta)\s+)?(?:operaci[oó]n|devoluci[oó]n|salida|entrada|propuesta)|confirmo\s+(?:(?:el|ese|este)\s+)?(?:pr[eé]stamo|traslado)|(?:s[ií]\s+)?proced(?:e|er)\s+con\s+(?:(?:la|el|esa|esta|ese|este)\s+)?(?:operaci[oó]n|devoluci[oó]n|pr[eé]stamo|salida|entrada|traslado|propuesta)|ejecut(?:a|ar)\s+(?:(?:la|el|esa|esta|ese|este)\s+)?(?:operaci[oó]n|devoluci[oó]n|pr[eé]stamo|salida|entrada|traslado|propuesta))\b",
        ]
        for p in patrones_confirmacion:
            if re.search(p, t_clean):
                return ExtraccionSemantica(
                    intencion="confirmar",
                    confianza=1.0,
                    fuente="marco_transaccional",
                )

        # B. Cancelación explícita (incluye giros coloquiales y costeños de cancelación)
        # Nunca cancelar si la frase es una corrección contrastiva (ej. «no quiero que sean esas unidades sino 5»,
        # «no quiero 2, quiero 4», «olvídalo, ponle 3», «no quiero a Carlos sino a Miguel», «no es prestable»,
        # «no, stock mínimo 5», «no, al Almacén C»).
        from .numeros import extraer_modificacion_cantidad_propuesta, extraer_numero_espanol
        tipo_op_prop = str(propuesta.get("tipo_operacion") or propuesta.get("tipo") or "").upper()
        _mod_es_prestable: bool | None = None
        _mod_stock_minimo: int | None = None
        if tipo_op_prop == "CREAR_PRODUCTO":
            if re.search(
                r"\b(?:no\s+(?:es\s+|sea\s+|lo\s+dejes\s+)?(?:prestable|retornable)|"
                r"no\s+se\s+(?:puede\s+)?prest\w+|no\s+admite\s+pr[eé]stamos?|"
                r"consumible|desechable|uso\s+definitivo)\b",
                t_clean,
            ):
                _mod_es_prestable = False
            elif re.search(
                r"\b(?:prestable|retornable|se\s+(?:puede\s+)?presta\w*|"
                r"admite\s+pr[eé]stamos?|para\s+pr[eé]stamo)\b",
                t_clean,
            ):
                _mod_es_prestable = True
            m_stk_min_prop = re.search(
                r"\b(?:stock\s+m[ií]nimo|umbral\s+m[ií]nimo|m[ií]nimo)\s*(?:de|en|a|por|es|sea|ser[ií]a|que\s+sea|:)?\s*([^\.,;\n]+)",
                t_raw,
                flags=re.IGNORECASE,
            )
            if m_stk_min_prop:
                val_sm = extraer_numero_espanol(m_stk_min_prop.group(1), permitir_cero=True, solo_al_inicio=True)
                if val_sm is not None and val_sm >= 0:
                    _mod_stock_minimo = val_sm

        _mod_stock_inicial_cero = bool(
            tipo_op_prop == "CREAR_PRODUCTO"
            and re.search(
                r"\b(?:(?:stock\s+inicial|inicial|cantidad)\s*(?:de|en|a|es|sea|:)?\s*(?:0|cero)|"
                r"(?:mejor|pon(?:le|lo)?|que\s+sea[ns]?|c[aá]mbial[oa]\s+a|sin\s+stock)\s*(?:0|cero)?\b)",
                t_clean,
            )
            and re.search(r"\b(?:0|cero|sin\s+stock)\b", t_clean)
            and _mod_stock_minimo is None
        )

        # Detección temprana de almacén o ubicación destino para evitar falso cancelamiento contrastivo
        _menciona_alm_o_ub_mod = bool(
            re.search(r"\b(?:almac[eé]n|bodega|ubicaci[oó]n|estante|vitrina|zona|caj[oó]n|repisa|destino)\b", t_clean)
        )
        _es_correccion_contrastiva = bool(
            extraer_modificacion_cantidad_propuesta(t_raw) is not None
            or _mod_es_prestable is not None
            or _mod_stock_minimo is not None
            or _mod_stock_inicial_cero
            or (_menciona_alm_o_ub_mod and re.search(r"\b(?:no\b|mejor|cambi\w*|al\b|hacia|para|en|desde|destino|ubicaci[oó]n)\b", t_clean))
            or re.search(r"\b(?:sino|en\s+vez\s+de|en\s+lugar\s+de)\b", t_clean)
        )
        t_cancel = re.sub(
            r"^(?:nombe|eche|aj[aá]|oye|oiga|compae|compadre|compa|cuadro|mi\s+llave|llave|mijo|ombe)\s+",
            "",
            t_clean,
        ).strip() or t_clean
        patrones_cancelacion = [
            r"^(no|nombe|cancela(?:r)?(?:\s+por\s+favor)?|cancelado|olv[ií]dalo|olvidate|olv[ií]date|descarta|descartar|ya\s+no|no\s+lo\s+hagas|ya\s+no\s+hagas\s+eso|no\s+hagas\s+(?:eso|nada)|(?:no\s+)?d[eé]jal(?:o|a|os|as)\s+(?:as[ií]|quieto)|deja\s+eso\s+(?:quieto|as[ií])|para|detenlo|rechazar|rechaza)$",
            r"\b(cancela(?:r)?\s+(?:(?:el|la|los|las|esa|este|esta|dicha)\s+)?(?:operaci[oó]n|propuesta|solicitud|eso|todo)|(?:ya\s+)?no\s+(?:lo\s+)?(?:hagas(?:\s+eso)?|hacer\s+nada|quiero(?:\s+(?:eso|esto|hacer(?:lo)?|nada|continuar|seguir))?(?!\s+(?:que|a|para|\d|un|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)))|(?:no\s+)?d[eé]jal(?:o|a|os|as)\s+(?:as[ií]|quieto)|deja\s+eso\s+(?:quieto|as[ií])|olv[ií]dalo|descarta(?:r)?\s+(?:(?:el|la|los|las|esa|este|esta|dicha)\s+)?(?:propuesta|operaci[oó]n|eso)|no\s*,?\s*mejor\s+no|no\s*,?\s*cancela(?:r)?(?:lo)?|dejar\s+as[ií])\b",
        ]
        if not _es_correccion_contrastiva:
            for p in patrones_cancelacion:
                if re.search(p, t_cancel):
                    return ExtraccionSemantica(
                        intencion="cancelar",
                        confianza=1.0,
                        fuente="marco_transaccional",
                    )

        # B.2 Confirmación con observación en una sola frase (ej. "Confirma y ponle que llegó sin caja")
        # y modificación/agregado/borrado de observación por voz mientras la propuesta está activa.
        from .observaciones import (
            extraer_confirmacion_con_observacion,
            extraer_modificacion_observacion_propuesta,
        )
        obs_en_confirmacion = extraer_confirmacion_con_observacion(t_raw, tipo_operacion=tipo_op_prop)
        if obs_en_confirmacion:
            return ExtraccionSemantica(
                intencion="confirmar",
                confianza=1.0,
                fuente="marco_transaccional",
                datos_extra={"observacion": obs_en_confirmacion},
            )

        mod_obs = extraer_modificacion_observacion_propuesta(t_raw, tipo_operacion=tipo_op_prop)
        if mod_obs and _mod_es_prestable is None and _mod_stock_minimo is None and not _mod_stock_inicial_cero:
            accion_obs, val_obs = mod_obs
            return ExtraccionSemantica(
                intencion="modificar_propuesta",
                campo_modificar="observacion",
                nuevo_valor=val_obs,
                confianza=0.98,
                fuente="marco_transaccional",
                datos_extra={"accion_observacion": accion_obs},
            )

        if _mod_es_prestable is not None:
            return ExtraccionSemantica(
                intencion="modificar_propuesta",
                campo_modificar="es_prestable",
                nuevo_valor="Prestable / retornable" if _mod_es_prestable else "Consumible (no prestable)",
                confianza=0.98,
                fuente="marco_transaccional",
                datos_extra={"es_prestable": _mod_es_prestable},
            )

        if _mod_stock_minimo is not None:
            return ExtraccionSemantica(
                intencion="modificar_propuesta",
                campo_modificar="stock_minimo",
                nuevo_valor=str(_mod_stock_minimo),
                cantidad=_mod_stock_minimo,
                confianza=0.98,
                fuente="marco_transaccional",
                datos_extra={"stock_minimo": _mod_stock_minimo},
            )

        if _mod_stock_inicial_cero:
            return ExtraccionSemantica(
                intencion="modificar_propuesta",
                campo_modificar="cantidad",
                nuevo_valor="0",
                cantidad=0,
                confianza=0.98,
                fuente="marco_transaccional",
            )

        # C.1 Corrección contrastiva explícita de Destinatario
        # Ej: "No, a Carlos no, a Jhan Carlos", "No era para María, era para Pedro",
        # "A Carlos no, sino a Jhan Carlos", "En vez de Carlos, a Jhan Carlos", "No, espérate, no era Carlos; era Carlos Andrés"
        patrones_dest_contrastivos = [
            r"\b(?:no\s*,?\s*)?(?:esp[eé]rate\s*,?\s*|espera\s*,?\s*|perd[oó]n\s*,?\s*|disculpa\s*,?\s*)?(?:a|para)\s+[a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]{2,30}?\s+no\b[\s,;:]*(?:sino\s+|mejor\s+|es\s+|era\s+)?(?:a|para|a\s+nombre\s+de)?\s*([a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]{2,40})[\s.,;:!?]*$",
            r"\b(?:no\s*,?\s*)?(?:esp[eé]rate\s*,?\s*|espera\s*,?\s*|perd[oó]n\s*,?\s*|disculpa\s*,?\s*)?no\s+(?:era|es)\s+(?:para\s+|a\s+|de\s+)?[a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]{2,30}?[\s,;]+(?:sino\s+|mejor\s+)?(?:era|es)?\s*(?:para\s+|a\s+|a\s+nombre\s+de\s+)?([a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]{2,40})[\s.,;:!?]*$",
            r"\ben\s+(?:vez|lugar)\s+de\s+(?:a\s+|para\s+)?[a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]{2,30}?[\s,;]+(?:que\s+sea\s+|ponlo\s+|c[aá]mbialo\s+)?(?:para\s+|a\s+|a\s+nombre\s+de\s+)([a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]{2,40})[\s.,;:!?]*$",
        ]
        for pat_c in patrones_dest_contrastivos:
            m_dc = re.search(pat_c, t_raw, re.IGNORECASE)
            if m_dc:
                nombre_crudo = m_dc.group(1).strip()
                nombre_crudo = re.sub(r"^(?:a|para|nombre|de|el|la)\s+", "", nombre_crudo, flags=re.IGNORECASE).strip()
                from .numeros import extraer_numero_espanol
                es_cant_en_contraste = bool(
                    re.search(r"\b(?:unidades?|piezas?|art[ií]culos?|cantidad)\b", t_raw, re.IGNORECASE)
                    or extraer_numero_espanol(
                        re.sub(r"\b(?:unidades?|piezas?|art[ií]culos?)\b", "", nombre_crudo, flags=re.IGNORECASE).strip(),
                        permitir_cero=True,
                        solo_numero_exacto=True,
                    ) is not None
                )
                if (
                    len(nombre_crudo) >= 2
                    and not es_cant_en_contraste
                    and not re.search(r"\b([A-Za-z]{2,5}-\d{1,4})\b", nombre_crudo)
                ):
                    from integraciones.haku.servicios_personas import capitalizar_nombre_propio
                    return ExtraccionSemantica(
                        intencion="modificar_propuesta",
                        campo_modificar="destinatario",
                        nuevo_valor=capitalizar_nombre_propio(nombre_crudo),
                        confianza=0.99,
                        fuente="marco_transaccional",
                    )

        # C.2 Sustitución de Producto o Código en propuesta activa
        # Ej: "No, perdón, era PAP-031", "No, era de PAP-031", "Cámbialo por PAP-031",
        # "Mejor del producto PAP-031", "No, perdón, era la batería 9V", "Cambia el producto a PAP-031"
        m_cod_prop = re.search(r"\b([A-Za-z]{2,5}-\d{1,4})\b", t_raw)
        if m_cod_prop:
            cod_detectado = m_cod_prop.group(1).upper()
            # Evitar capturar si es una consulta explícita de stock/ubicación sobre otro código
            es_pregunta_lectura = bool(
                re.search(
                    r"\b(cu[aá]nt[oa]s?|d[oó]nde|ubicaci[oó]n|qu[eé]\s+stock|quedan?|hay\s+disponibles?|consultar|b[uú]scame|mu[eé]strame|dime)\b",
                    t_clean,
                )
            )
            if not es_pregunta_lectura:
                return ExtraccionSemantica(
                    intencion="modificar_propuesta",
                    campo_modificar="producto",
                    nuevo_valor=cod_detectado,
                    producto=cod_detectado,
                    confianza=0.99,
                    fuente="marco_transaccional",
                )

        # Sustitución de producto por nombre sin código explícito
        patron_prod_nombre = re.compile(
            r"^(?:y\s+|pero\s+|adem[aá]s\s+|tambi[eé]n\s+)?(?:no\s*,?\s*)?(?:perd[oó]n\s*,?\s*|disculpa\s*,?\s*|espera\s*,?\s*|esp[eé]rate\s*,?\s*|mejor\s+)?"
            r"(?:c[aá]mbia(?:lo)?\s+por\s+|cambia\s+el\s+producto\s+(?:a|por)\s+|el\s+producto\s+(?:es|era|ser[ií]a)\s+|"
            r"no\s+era\s+ese\s+producto\s*,?\s*(?:era|es)\s+|era\s+del?\s+producto\s+|que\s+sea\s+del?\s+producto\s+|"
            r"(?:no\s*,?\s*)?(?:perd[oó]n\s*,?\s*|disculpa\s*,?\s*)(?:era|es)\s+(?:el\s+|la\s+|los\s+|las\s+|un\s+|una\s+|de\s+|del\s+)?)"
            r"([a-zA-Z0-9ñÑáéíóúÁÉÍÓÚ\s'-]{3,50})$",
            re.IGNORECASE,
        )
        m_prod_nom = patron_prod_nombre.search(t_raw)
        if m_prod_nom:
            cand_prod = m_prod_nom.group(1).strip()
            cand_prod = re.sub(r"^(?:el|la|los|las|un|una|de|del|producto|art[ií]culo)\s+", "", cand_prod, flags=re.IGNORECASE).strip()
            # No confundir con números puros ("era 5") ni con "para <Persona>"
            es_solo_numero = bool(re.match(r"^(\d+|un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)(\s+unidades?)?$", cand_prod, re.IGNORECASE))
            empieza_preposicion_persona = bool(re.match(r"^(?:para|a\s+nombre\s+de)\b", cand_prod, re.IGNORECASE))
            if len(cand_prod) >= 3 and not es_solo_numero and not empieza_preposicion_persona:
                # Si el usuario dijo explícitamente "producto" o si coincide con un producto del catálogo
                menciona_palabra_prod = bool(re.search(r"\b(producto|art[ií]culo|referencia|c[aá]mbialo\s+por)\b", t_clean))
                if menciona_palabra_prod:
                    return ExtraccionSemantica(
                        intencion="modificar_propuesta",
                        campo_modificar="producto",
                        nuevo_valor=cand_prod,
                        producto=cand_prod,
                        confianza=0.96,
                        fuente="marco_transaccional",
                    )
                else:
                    # Comprobar si cand_prod resuelve a un producto o candidatos de producto en catálogo
                    try:
                        from .normalizador import NormalizadorLenguajeHaku
                        p_obj, _, p_cands = NormalizadorLenguajeHaku._resolver_producto_en_texto(cand_prod, None)
                        if p_obj or p_cands:
                            return ExtraccionSemantica(
                                intencion="modificar_propuesta",
                                campo_modificar="producto",
                                nuevo_valor=cand_prod,
                                producto=cand_prod,
                                confianza=0.96,
                                fuente="marco_transaccional",
                            )
                    except Exception:
                        pass

        # F. Modificación de Motivo (evaluar antes de destinatario para que "para dotación" o "que sea por dotación" no se tome como persona)
        m_mot = re.search(
            r"\b(?:el\s+motivo\s+(?:es|ser[ií]a|debe\s+ser)\s+|c[aá]mbial[eo]\s+el\s+motivo\s+a\s+|"
            r"es\s+por\s+|que\s+sea\s+por\s+|por\s+|motivo\s+|para\s+)(dotaci[oó]n|entrega\s+definitiva|consumo\s+interno|mantenimiento|reparaci[oó]n|donaci[oó]n)\b",
            t_clean,
        )
        if m_mot:
            val_mot = m_mot.group(1).strip()
            if "dotaci" in val_mot.lower():
                val_mot = "dotacion"
            return ExtraccionSemantica(
                intencion="modificar_propuesta",
                campo_modificar="motivo",
                nuevo_valor=val_mot,
                confianza=0.95,
                fuente="marco_transaccional",
            )

        # E. Modificación de Almacén / Destino / Ubicación destino (resolución dinámica contra BD)
        es_pregunta_alm_ub = bool(re.search(r"\b(cu[aá]nt[oa]s?|d[oó]nde|qu[eé]\s+hay|quedan?|disponibles?)\b", t_clean))
        if not es_pregunta_alm_ub:
            # E.1 Cambio de ubicación física destino en TRASLADO_ENTRE
            if tipo_op_prop == "TRASLADO_ENTRE" and propuesta.get("almacen_destino_id"):
                try:
                    from inventario.models import Ubicacion
                    from inventario.utils import normalizar_texto as _norm_ub_txt
                    ubis_dest = list(
                        Ubicacion.objects.filter(
                            almacen_id=propuesta.get("almacen_destino_id"),
                            activo=True,
                            eliminado=False,
                        )
                    )
                    t_ub_norm = _norm_ub_txt(t_clean)
                    ub_match_prop = None
                    for u_c in sorted(ubis_dest, key=lambda x: len(x.nombre), reverse=True):
                        u_n = _norm_ub_txt(u_c.nombre)
                        u_low = u_c.nombre.lower().strip()
                        if (
                            (u_low and re.search(rf"(?<![\w]){re.escape(u_low)}(?![\w])", t_lower))
                            or (u_n and re.search(rf"(?<![\w]){re.escape(u_n)}(?![\w])", t_ub_norm))
                        ):
                            ub_match_prop = u_c
                            break
                        if getattr(u_c, "clave_canonica", None) and re.search(rf"(?<![\w]){re.escape(u_c.clave_canonica)}(?![\w])", t_ub_norm):
                            ub_match_prop = u_c
                            break
                    if ub_match_prop is not None:
                        return ExtraccionSemantica(
                            intencion="modificar_propuesta",
                            campo_modificar="ubicacion_destino",
                            nuevo_valor=ub_match_prop.nombre,
                            confianza=0.98,
                            fuente="marco_transaccional",
                            datos_extra={
                                "ubicacion_destino_id": ub_match_prop.id,
                                "ubicacion_destino_nombre": ub_match_prop.nombre,
                            },
                        )
                except Exception:
                    pass

            # E.2 Cambio de almacén (soporta nombres reales en BD además de «Almacén X», con límites de palabra estrictos)
            nombre_alm_detectado = None
            try:
                from inventario.models import Almacen
                from inventario.utils import normalizar_texto as _norm_alm_txt
                t_alm_norm = _norm_alm_txt(t_clean)
                alms_activos = list(Almacen.objects.filter(activo=True, eliminado=False))
                mejor_pos = -1
                mejor_len = -1
                for a_c in sorted(alms_activos, key=lambda x: len(x.nombre), reverse=True):
                    a_n = _norm_alm_txt(a_c.nombre)
                    a_low = a_c.nombre.lower().strip()
                    m_a = (
                        re.search(rf"(?<![\w]){re.escape(a_low)}(?![\w])", t_lower)
                        if a_low else None
                    ) or (
                        re.search(rf"(?<![\w]){re.escape(a_n)}(?![\w])", t_alm_norm)
                        if a_n else None
                    )
                    if m_a and (m_a.start() > mejor_pos or (m_a.start() == mejor_pos and len(a_c.nombre) > mejor_len)):
                        mejor_pos = m_a.start()
                        mejor_len = len(a_c.nombre)
                        nombre_alm_detectado = a_c.nombre
            except Exception:
                pass
            if not nombre_alm_detectado:
                _stop_alm = {"al", "del", "de", "en", "el", "la", "para", "por", "hacia", "desde", "a"}
                for m_alm in re.finditer(
                    r"\b(?:en\s+el\s+|para\s+el\s+|al\s+|del\s+|desde\s+el\s+|hacia\s+el\s+)?almac[eé]n\s+([a-zA-Z0-9]+)\b",
                    t_clean,
                ):
                    suf_alm = m_alm.group(1).strip()
                    if suf_alm.lower() in _stop_alm and suf_alm.lower() != "a":
                        continue
                    if suf_alm.lower() == "a" and re.match(r"^\s*(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\b", t_clean[m_alm.end():]):
                        continue
                    nombre_alm_detectado = f"Almacén {suf_alm.upper()}"

            if nombre_alm_detectado:
                es_cambio_origen_traslado = bool(
                    re.search(r"\b(?:desde|origen|del\s+almac[eé]n|de\s+la\s+bodega)\b", t_clean)
                    and not re.search(r"\b(?:destino|hacia|al\s+almac[eé]n|para\s+el\s+almac[eé]n|a\s+la\s+bodega)\b", t_clean)
                )
                campo_alm = (
                    "almacen"
                    if (tipo_op_prop != "TRASLADO_ENTRE" or es_cambio_origen_traslado)
                    else "almacen_destino"
                )
                return ExtraccionSemantica(
                    intencion="modificar_propuesta",
                    campo_modificar=campo_alm,
                    nuevo_valor=nombre_alm_detectado,
                    almacen=nombre_alm_detectado,
                    confianza=0.96,
                    fuente="marco_transaccional",
                )

        # C.3 Modificación natural de Destinatario (con conectores "y", "pero", "mejor", "que sea para", "ponlo a nombre de", etc.)
        # Expresiones como: "Y que sea para Pedro", "Que sea para Pedro", "Mejor para Pedro", "No, para Pedro",
        # "a Jhan Carlos", "para Jhan Carlos", "el destinatario es Jhan Carlos", "cámbialo a Jhan Carlos",
        # "no, perdón, era para Pedro", "ponlo para Pedro", "entrégaselo a Pedro", "préstaselo a Pedro"
        patron_destinatario = re.compile(
            r"^(?:s[ií]\s*,?\s*pero\s+|y\s+|pero\s+|adem[aá]s\s+|tambi[eé]n\s+|entonces\s+)?(?:no\s*,?\s*)?"
            r"(?:perd[oó]n\s*,?\s*|disculpa\s*,?\s*|espera\s*,?\s*|esp[eé]rate\s*,?\s*|mejor\s+)?"
            r"(?:que\s+(?:sea|quede)\s+(?:para|a\s+nombre\s+de|a)\s+|"
            r"el\s+destinatario\s+(?:es|era|ser[ií]a|debe\s+ser)\s+|"
            r"c[aá]mbia(?:lo|la|le)?\s+(?:el\s+destinatario\s+|el\s+nombre\s+|el\s+responsable\s+)?(?:a|para|por)\s+|"
            r"corrige\s+(?:el\s+)?(?:destinatario|nombre|responsable)\s+(?:a|para|por)\s+|"
            r"pon(?:lo|la|le)?\s+(?:a\s+nombre\s+de|para|a)\s+|"
            r"an[oó]ta(?:selo|lo|le)\s+a\s+|"
            r"se\s+lo\s+(?:voy\s+a\s+prestar|entrego|doy|presto)\s+a\s+|"
            r"entr[eé]ga(?:selo|lo)\s+a\s+|pr[eé]sta(?:selo|lo)\s+a\s+|"
            r"(?:es|era)\s+para\s+|(?:es|era)\s+a\s+nombre\s+de\s+|"
            r"a\s+nombre\s+de\s+|para\s+|a\s+)\s*([a-zA-Z0-9ñÑáéíóúÁÉÍÓÚ\s'-]{2,40})[\s.,;:!?]*$",
            re.IGNORECASE,
        )
        m_dest = patron_destinatario.search(t_raw)
        if m_dest:
            nombre_crudo = m_dest.group(1).strip()
            nombre_crudo = re.sub(r"^(?:a|para|nombre|de|el|la)\s+", "", nombre_crudo, flags=re.IGNORECASE).strip()
            # Evitar confundir con números ("cámbialo a 5") o sustantivos de inventario
            es_num_dest = bool(re.match(r"^(\d+|un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)(\s+unidades?)?$", nombre_crudo, re.IGNORECASE))
            palabras_no_persona = {"almacen", "almacén", "bodega", "dotacion", "dotación", "mantenimiento", "consumo", "producto", "inventario"}
            if (
                len(nombre_crudo) >= 2
                and not es_num_dest
                and nombre_crudo.lower() not in palabras_no_persona
                and not re.search(r"\b(almac[eé]n|bodega|unidades?|piezas?)\b", nombre_crudo, re.IGNORECASE)
            ):
                from integraciones.haku.servicios_personas import capitalizar_nombre_propio
                return ExtraccionSemantica(
                    intencion="modificar_propuesta",
                    campo_modificar="destinatario",
                    nuevo_valor=capitalizar_nombre_propio(nombre_crudo),
                    confianza=0.98,
                    fuente="marco_transaccional",
                )

        # D. Modificación de Cantidad (cortas, largas, coloquiales y contrastivas)
        # Ej: "5", "mejor 3", "cámbiala a 5", "solo 4 unidades",
        # "hey no quiero que sean esas unidades sino que sean 5 unidades",
        # "no eran 2 sino 5", "en vez de 2 ponle 4", "pon 6 en vez de 2", "no quiero 2, quiero 4"
        from .numeros import extraer_modificacion_cantidad_propuesta
        cant_mod = extraer_modificacion_cantidad_propuesta(t_raw)
        if cant_mod is not None:
            return ExtraccionSemantica(
                intencion="modificar_propuesta",
                campo_modificar="cantidad",
                nuevo_valor=str(cant_mod),
                cantidad=cant_mod,
                confianza=0.98,
                fuente="marco_transaccional",
            )

        # G. Protección ante intento de modificación no reconocido o incompleto
        # Si el usuario expresa intención de corregir/modificar ("no, ...", "perdón...", "cambia...", "corrige...",
        # "mejor...", "en vez de...", "equivocado", "no era...") pero no se pudo extraer con certeza el campo/valor:
        es_pregunta_informativa = bool(
            re.search(
                r"\b(cu[aá]nt[oa]s?|d[oó]nde|ubicaci[oó]n|qu[eé]\s+stock|quedan?|queda|hay\s+disponibles?|qui[eé]n\s+tiene|consultar|b[uú]scame|mu[eé]strame)\b",
                t_clean,
            )
        )
        tiene_indicador_correccion = bool(
            re.search(
                r"^(?:no\b|perd[oó]n\b|disculpa\b|espera\b|esp[eé]rate\b|mejor\b|c[aá]mbia|cambiar\b|corrige\b|corregir\b|modificar?\b|en\s+vez\s+de\b|en\s+lugar\s+de\b|y\s+que\s+sea\b|que\s+sea\b)",
                t_clean,
            )
            or re.search(
                r"\b(no\s+era\b|no\s+es\s+ese\b|c[aá]mbiale\b|c[aá]mbialo\b|corregir\s+la\s+propuesta|modificar\s+la\s+propuesta|est[aá]\s+mal|equivocad[oa])\b",
                t_clean,
            )
        )
        es_inicio_otra_op = bool(
            re.search(
                r"\b(?:quiero\s+|voy\s+a\s+|mejor\s+)?(?:registrar\s+(?:una\s+)?(?:entrada|salida|devoluci[oó]n)|hacer\s+un\s+(?:pr[eé]stamo|traslado)|inhabilitar|habilitar|crear\s+producto)\b",
                t_clean,
            )
        )
        if tiene_indicador_correccion and not es_pregunta_informativa and not es_inicio_otra_op:
            return ExtraccionSemantica(
                intencion="modificar_propuesta_incompleta",
                confianza=0.95,
                fuente="marco_transaccional",
                datos_extra={"texto_original": t_raw},
            )

        # H. Si la propuesta ya tenía una corrección pendiente de aclarar y el usuario proporciona solo un nombre propio (ej. "Carlos Andrés Rodríguez")
        if propuesta.get("modificacion_sin_resolver") and not es_pregunta_informativa and not es_inicio_otra_op:
            if re.match(r"^[a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]{4,50}$", t_raw) and len(t_raw.split()) >= 2:
                from integraciones.haku.servicios_personas import capitalizar_nombre_propio
                return ExtraccionSemantica(
                    intencion="modificar_propuesta",
                    campo_modificar="destinatario",
                    nuevo_valor=capitalizar_nombre_propio(t_raw),
                    confianza=0.95,
                    fuente="marco_transaccional",
                )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN LLM: INTEGRACIÓN DIRECTA CON CEREBRO CENTRAL
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _normalizar_contrato_canonico(cls, datos: Dict[str, Any], texto: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """OPT-02B: Traduce intenciones y ranuras al contrato canónico de HAKU Inventario.

        Garantiza:
        1. Intenciones de mutación directa ('devolucion', 'prestamo', 'traslado', 'entrada', 'salida', 'dotacion')
           se traducen inequívocamente a 'proponer_operacion' con su respectivo 'tipo_operacion'.
        2. 'consultar_catalogo' con almacén se normaliza a 'consultar_catalogo_almacen'.
        3. Consultas de stock sin producto pero con almacén se normalizan a 'consultar_catalogo_almacen'.
        4. Consultas directas de reposición ('agotados', 'bajo_stock') se normalizan a 'consultar_reposicion'.
        5. Consultas directas de atributos de producto (categoría, código, stock_mínimo, prestados)
           se normalizan a 'consultar_atributo'.
        6. Toda intención no reconocida o ambigua se normaliza estrictamente a 'desconocido'.
        7. Jamás inventa entidades (producto, cantidad, destinatario, almacén) ausentes.
        """
        if not datos or not isinstance(datos, dict):
            return "desconocido", {}

        datos_norm = dict(datos)
        intencion_raw = _limpiar_str(datos_norm.get("intencion"))
        if not intencion_raw:
            return "desconocido", datos_norm

        intencion = intencion_raw.lower().strip()
        intencion = (
            intencion.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )

        producto = _limpiar_str(datos_norm.get("producto"))
        almacen = _limpiar_str(datos_norm.get("almacen"))

        # Si el usuario pregunta dónde está un producto o referencia ("¿dónde está ese producto?"),
        # se trata de una consulta de atributo ubicación, no un catálogo general ni de almacén.
        if texto:
            t_low = texto.lower()
            if re.search(r"\b(d[oó]nde\s+est[aá]|d[oó]nde\s+se\s+encuentra|d[oó]nde\s+est[aá]n|d[oó]nde\s+se\s+guardan?|en\s+qu[eé]\s+lugar)\b", t_low):
                if re.search(r"\b(ese|este|aquel)\s*(?:producto|art[ií]culo)\b", t_low) or (
                    re.search(r"\b(?:el\s+producto|el\s+art[ií]culo)\b", t_low) and not producto
                ):
                    intencion = "consultar_atributo"
                    datos_norm["atributo"] = "ubicacion"
                    datos_norm["producto"] = None
                    producto = None
                    datos_norm["ubicacion"] = None
                    datos_norm["almacen"] = None
                    almacen = None
                elif producto:
                    intencion = "consultar_atributo"
                    datos_norm["atributo"] = "ubicacion"

        # Detección de comparativa entre ubicaciones
        _patron_ubi_codigo = re.compile(r"\b([a-cA-C][1-6]|[a-zA-Z][0-9]{1,3})\b", re.IGNORECASE)
        ubis_encontradas = _patron_ubi_codigo.findall(texto) if texto else []
        if texto and re.search(r"\b(diferencia|compara|comparar|comparativa|vs|versus)\b", texto.lower()) and len(ubis_encontradas) >= 2:
            intencion = "comparar_ubicaciones"
            datos_norm["ubicacion"] = ubis_encontradas[0].upper()
            datos_norm["destino"] = ubis_encontradas[1].upper()

        # Detección de métrica en consultas de ubicación
        if texto and (intencion == "consultar_ubicaciones" or datos_norm.get("ubicacion")):
            t_low = texto.lower()
            if re.search(r"\b(cu[aá]ntas?\s+(?:unidades?|piezas?|existencias?|cantidad)|cu[aá]nto\s+(?:stock|inventario)|total\s+de\s+unidades|unidades\s+f[ií]sicas)\b", t_low):
                datos_norm["metrica"] = "unidades"
            elif re.search(r"\b(cu[aá]ntos\s+productos|cu[aá]ntas\s+referencias|cu[aá]ntos\s+[ií]tems?|cu[aá]ntas?\s+cosas|n[uú]mero\s+de\s+productos|qu[eé]\s+productos)\b", t_low):
                datos_norm["metrica"] = "productos"
            elif not datos_norm.get("metrica"):
                datos_norm["metrica"] = "contenido"

        # Detección de consultas de familias de inventario (cables, herramientas, redes)
        if texto and intencion in ("consultar_stock", "consultar_atributo", "consultar_catalogo", "consultar_catalogo_almacen"):
            t_low = texto.lower()
            prod_val = (datos_norm.get("producto") or "").lower()
            for fam in ("cables", "cable", "herramientas", "herramienta", "redes", "red", "switches", "switch"):
                if re.search(rf"\b{fam}\b", t_low) or prod_val == fam:
                    if not re.search(r"\b(rj45|utp|hdmi|gigabit|taladro|mult[ií]metro)\b", t_low):
                        intencion = "consultar_stock"
                        datos_norm["producto"] = fam
                        datos_norm["es_familia"] = True
                        break

        # Si el modelo extrajo un código de ubicación física (ej: A1..A6, B1..B6, C1..C6)
        # en lugar de un almacén, normalizarlo como 'ubicacion' e intención 'consultar_ubicaciones'
        if almacen and re.match(r"^[A-Ca-c][1-6]$", almacen.strip()):
            datos_norm["ubicacion"] = almacen.strip().upper()
            datos_norm["almacen"] = None
            almacen = None
            if intencion in ("consultar_catalogo_almacen", "consultar_catalogo", "consultar_stock", "desconocido"):
                intencion = "consultar_ubicaciones"

        # Si no hay producto y se especificó una ubicación física (ej: A1..C6), normalizar a consultar_ubicaciones
        if not producto and datos_norm.get("ubicacion") and re.match(r"^[A-Ca-c][1-6]$", str(datos_norm.get("ubicacion")).strip()):
            if intencion in ("consultar_catalogo_almacen", "consultar_catalogo", "consultar_stock", "desconocido"):
                intencion = "consultar_ubicaciones"

        if almacen and ("dotaci" in almacen.lower()):
            almacen = None
            datos_norm["almacen"] = None
            if not datos_norm.get("tipo_operacion") or datos_norm.get("tipo_operacion") == "salida":
                datos_norm["tipo_operacion"] = "salida"
                datos_norm["campo_modificar"] = "motivo"
                datos_norm["nuevo_valor"] = "dotacion"

        if producto and ("dotaci" in producto.lower()):
            producto = None
            datos_norm["producto"] = None
            if not datos_norm.get("tipo_operacion") or datos_norm.get("tipo_operacion") == "salida":
                datos_norm["tipo_operacion"] = "salida"
                datos_norm["campo_modificar"] = "motivo"
                datos_norm["nuevo_valor"] = "dotacion"

        if producto and producto.lower() in (
            "producto",
            "productos",
            "articulo",
            "articulos",
            "item",
            "items",
            "referencia",
            "referencias",
            "cosas",
            "cosa",
            "stock",
            "existencias",
            "cantidad",
            "producto desconocido",
            "desconocido",
            "desconocida",
            "producto no especificado",
            "no especificado",
            "no especificada",
            "ese producto",
            "este producto",
            "aquel producto",
            "el producto",
            "ninguno",
            "n/a",
            "null",
            "none",
        ):
            producto = None
            datos_norm["producto"] = None

        origen = _limpiar_str(datos_norm.get("origen"))
        destino = _limpiar_str(datos_norm.get("destino"))
        if origen and not almacen:
            almacen = origen
            datos_norm["almacen"] = origen
        ubicacion = _limpiar_str(datos_norm.get("ubicacion"))
        if ubicacion and ubicacion.lower() in (
            "ubicacion desconocida",
            "desconocida",
            "desconocido",
            "no especificada",
            "no especificado",
            "n/a",
            "null",
            "none",
        ):
            ubicacion = None
            datos_norm["ubicacion"] = None

        # Si el modelo extrajo una ubicación que no existe en el texto ni en el contexto, descartar alucinación
        if ubicacion and texto:
            _m_ubi_text = re.search(rf"\b{re.escape(ubicacion)}\b", texto, re.IGNORECASE)
            if not _m_ubi_text and not ("contexto" in texto.lower() and ubicacion.lower() in texto.lower()):
                datos_norm["ubicacion"] = None
                ubicacion = None

        # Si el modelo extrajo un almacén que no existe en el texto, ni en el contexto, ni se deriva de la ubicación
        if almacen and texto:
            _m_alm_text = re.search(r"\b(almac[eé]n|bodega|dep[oó]sito)\b", texto, re.IGNORECASE) or (
                re.search(r"\b(almac[eé]n\s+[a-cA-C]|en\s+el\s+[a-cA-C]\b)", texto, re.IGNORECASE)
            )
            ubi_val = datos_norm.get("ubicacion") or ubicacion
            ubi_corresponde = False
            if ubi_val and re.match(r"^[A-Ca-c][1-6]$", str(ubi_val).strip()):
                letra_ubi = str(ubi_val).strip()[0].upper()
                if f"almacén {letra_ubi}".lower() in almacen.lower() or letra_ubi in almacen.upper():
                    ubi_corresponde = True

            if not _m_alm_text and not ubi_corresponde and not ("contexto" in texto.lower() and almacen.lower() in texto.lower()):
                datos_norm["almacen"] = None
                almacen = None
        atributo = _limpiar_str(datos_norm.get("atributo"))
        tipo_op = _limpiar_str(datos_norm.get("tipo_operacion"))
        filtro_rep = _limpiar_str(datos_norm.get("filtro_reposicion"))

        # 1. Mutaciones operativas directas -> contrato canónico proponer_operacion
        if intencion in ("devolucion", "registrar_devolucion"):
            intencion = "proponer_operacion"
            datos_norm["tipo_operacion"] = tipo_op or "devolucion"

        elif intencion in ("prestamo", "registrar_prestamo"):
            intencion = "proponer_operacion"
            datos_norm["tipo_operacion"] = tipo_op or "prestamo"

        elif intencion in ("traslado", "trasladar", "mover_producto", "trasladar_producto"):
            intencion = "proponer_operacion"
            datos_norm["tipo_operacion"] = tipo_op or "traslado"

        elif intencion in ("entrada", "ingreso", "registrar_entrada"):
            intencion = "proponer_operacion"
            datos_norm["tipo_operacion"] = tipo_op or "entrada"

        elif intencion in ("salida", "retiro", "registrar_salida"):
            intencion = "proponer_operacion"
            datos_norm["tipo_operacion"] = tipo_op or "salida"

        elif intencion in ("dotacion", "salida_dotacion"):
            intencion = "proponer_operacion"
            datos_norm["tipo_operacion"] = tipo_op or "salida"
            if not datos_norm.get("campo_modificar"):
                datos_norm["campo_modificar"] = "motivo"
                datos_norm["nuevo_valor"] = "dotacion"

        elif intencion in ("inhabilitar", "inhabilitar_producto", "desactivar", "desactivar_producto"):
            intencion = "inhabilitar_producto"

        elif intencion in ("habilitar", "habilitar_producto", "activar", "activar_producto", "reactivar"):
            intencion = "habilitar_producto"

        # 2. Catálogo general vs catálogo de almacén específico vs consulta de producto
        elif intencion in ("consultar_catalogo", "catalogo", "listar_catalogo"):
            if almacen:
                intencion = "consultar_catalogo_almacen"
            elif producto:
                intencion = "consultar_stock"
            else:
                intencion = "consultar_catalogo"

        elif intencion == "consultar_stock" and not producto and almacen:
            intencion = "consultar_catalogo_almacen"

        # 3. Reposición y agotamiento
        elif intencion in ("agotados", "productos_agotados", "sin_stock", "productos_sin_stock"):
            intencion = "consultar_reposicion"
            datos_norm["filtro_reposicion"] = filtro_rep or "agotados"

        elif intencion in ("bajo_stock", "productos_bajo_stock", "stock_bajo"):
            intencion = "consultar_reposicion"
            datos_norm["filtro_reposicion"] = filtro_rep or "bajo_stock"

        elif intencion in ("reposicion", "consultar_faltantes"):
            intencion = "consultar_reposicion"

        # 4. Consultas directas de atributos de producto
        elif intencion in ("categoria", "consultar_categoria"):
            intencion = "consultar_atributo"
            datos_norm["atributo"] = atributo or "categoria"

        elif intencion in ("codigo", "consultar_codigo"):
            intencion = "consultar_atributo"
            datos_norm["atributo"] = atributo or "codigo"

        elif intencion in ("stock_minimo", "consultar_stock_minimo"):
            intencion = "consultar_atributo"
            datos_norm["atributo"] = atributo or "stock_minimo"

        elif intencion in ("quien_tiene", "prestados", "consultar_prestados"):
            intencion = "consultar_atributo"
            datos_norm["atributo"] = atributo or "prestados"

        elif intencion in ("historial", "consultar_historial"):
            intencion = "consultar_atributo"
            datos_norm["atributo"] = atributo or "historial"

        # 5. Ubicaciones físicas
        elif intencion in ("ubicacion", "ubicaciones", "consultar_ubicacion", "consultar_ubicaciones"):
            if (
                atributo == "ubicacion"
                or producto
                or not ubicacion
                or (ubicacion and ubicacion.lower() in ("no especificada", "no especificado"))
            ):
                intencion = "consultar_atributo"
                datos_norm["atributo"] = "ubicacion"
                if ubicacion and ubicacion.lower() in ("no especificada", "no especificado"):
                    datos_norm["ubicacion"] = None
                if producto and producto.lower() in ("producto no especificado", "no especificado", "ese producto", "este producto", "aquel producto"):
                    datos_norm["producto"] = None
            else:
                intencion = "consultar_ubicaciones"

        # 6. Movimientos
        elif intencion in ("movimientos", "consultar_metricas_movimientos"):
            intencion = "consultar_movimientos"

        # 7. Confirmaciones y cancelaciones desacopladas
        elif intencion in ("confirmar", "confirmacion", "accion_confirmar"):
            intencion = "confirmar"

        elif intencion in ("cancelar", "cancelacion", "accion_cancelar"):
            intencion = "cancelar"

        # 8. Otras intenciones estándar válidas del contrato HAKU
        elif intencion in (
            "consultar_catalogo",
            "consultar_catalogo_almacen",
            "consultar_stock",
            "consultar_atributo",
            "consultar_ubicaciones",
            "comparar_ubicaciones",
            "consultar_reposicion",
            "consultar_movimientos",
            "proponer_operacion",
            "modificar_propuesta",
            "inhabilitar_producto",
            "habilitar_producto",
            "seleccionar_alternativa",
            "cambio_tema",
            "saludo",
            "agradecimiento",
        ):
            pass

        else:
            intencion = "desconocido"

        # Detección contextual de consultas sobre agotamiento / reposición cuando no hay un producto específico
        if intencion in ("consultar_ubicaciones", "consultar_stock", "consultar_catalogo", "consultar_reposicion", "consultar_atributo"):
            if texto and re.search(r"\b(agotad[ao]s?|sin\s+stock|sin\s+existencias?|en\s+cero|se\s+acabar?on|qu[eé]\s+nos\s+hace\s+falta)\b", texto.lower()):
                if not producto or producto.lower() in (
                    "referencias en cero", "productos en cero", "articulos en cero", "en cero",
                    "cosas agotadas", "articulos agotados", "productos agotados", "referencias agotadas",
                    "artículos sin stock", "articulos sin stock", "productos sin stock", "cosas sin stock",
                    "productos que se acabaron", "articulos que se acabaron", "cosas que se acabaron",
                ):
                    intencion = "consultar_reposicion"
                    datos_norm["filtro_reposicion"] = "agotados"
                    datos_norm["producto"] = None
                    producto = None

        # Guardia determinista contra alucinación del LLM en comandos explícitos de mutación:
        # 1) Si el usuario ordena explícitamente registrar una entrada/salida/préstamo/devolución/traslado
        #    (p. ej. "Quiero registrar una entrada de ARD-001 con observación: Lote 3"), impedir que el LLM
        #    lo degrade a una consulta de lectura (consultar_atributo/historial/movimientos).
        # 2) Si la intención resultante es proponer_operacion y el LLM inventó una cantidad tomada de un
        #    código (TAL-002), especificación (20V) u observación (Lote 3), limpiarla.
        if texto:
            msg_actual = (
                texto.split("[Mensaje del usuario]")[-1].strip()
                if "[Mensaje del usuario]" in texto
                else texto.strip()
            )
            from .observaciones import separar_observacion_operacion
            from .numeros import extraer_cantidad_operacion_texto
            msg_base_llm, _ = separar_observacion_operacion(msg_actual)
            msg_base_low = msg_base_llm.lower()

            es_pregunta_lectura_llm = bool(
                "?" in msg_base_llm
                or "¿" in msg_base_llm
                or re.search(
                    r"\b(?:cu[aá]nt[oa]s?|qu[eé]\s+(?:entradas?|salidas?|pr[eé]stamos?|devoluciones?|traslados?|movimientos?)|"
                    r"hubo|se\s+(?:hicieron|registraron)|historial|mu[eé]strame|l[ií]stame|qui[eé]n\s+tiene|ver\s+los)\b",
                    msg_base_low,
                )
            )
            if intencion.startswith("consultar_") and not es_pregunta_lectura_llm:
                tipo_op_forzado = None
                if re.search(r"\b(?:registrar|hacer|generar|dar|ingresar|meter|agregar)\s+(?:una?\s+)?entrada\b", msg_base_low):
                    tipo_op_forzado = "entrada"
                elif re.search(r"\b(?:registrar|hacer|generar|dar)\s+(?:una?\s+)?(?:salida|dotaci[oó]n)\b", msg_base_low):
                    tipo_op_forzado = "salida"
                elif re.search(r"\b(?:registrar|hacer|generar|dar)\s+(?:un\s+)?pr[eé]stamo\b", msg_base_low):
                    tipo_op_forzado = "prestamo"
                elif re.search(r"\b(?:registrar|hacer|generar|dar)\s+(?:una?\s+)?devoluci[oó]n\b", msg_base_low):
                    tipo_op_forzado = "devolucion"
                elif re.search(r"\b(?:registrar|hacer|generar|dar)\s+(?:un\s+)?traslado\b", msg_base_low):
                    tipo_op_forzado = "traslado"
                if tipo_op_forzado:
                    intencion = "proponer_operacion"
                    datos_norm["tipo_operacion"] = tipo_op_forzado
                    datos_norm.pop("atributo", None)

            if intencion == "proponer_operacion" and datos_norm.get("cantidad") is not None:
                cant_real_msg = extraer_cantidad_operacion_texto(msg_actual)
                if cant_real_msg is None:
                    datos_norm["cantidad"] = None

        return intencion, datos_norm

    @classmethod
    def _interpretar_con_llm(
        cls,
        texto: str,
        cliente_llm: Any,
        contexto_operativo: Any,
    ) -> Optional[ExtraccionSemantica]:
        """Solicita al Cerebro Central la extracción de intención y ranuras con esquema compacto (OPT-02A)."""
        foco_producto = None
        candidatos_txt = None
        propuesta_txt = None

        if contexto_operativo:
            if getattr(contexto_operativo, "producto_nombre", None):
                foco_producto = f"{contexto_operativo.producto_codigo or ''} - {contexto_operativo.producto_nombre}".strip(" -")
            if getattr(contexto_operativo, "candidatos", None):
                cands = [f"{c.get('codigo', '')}: {c.get('nombre', '')}" for c in contexto_operativo.candidatos]
                candidatos_txt = "; ".join(cands)
            if getattr(contexto_operativo, "propuesta_activa", None):
                p = contexto_operativo.propuesta_activa
                propuesta_txt = f"{p.get('tipo', '')} de {p.get('cantidad', 1)} {p.get('producto_nombre', '')} a {p.get('destinatario', 'N/A')}"

        prompt_sistema = (
            "Eres el intérprete semántico de HACU Inventario de la Universidad AudacIA.\n"
            "Analiza el mensaje y extrae la intención y ranuras en un JSON COMPACTO.\n"
            "REGLAS INNEGOCIABLES:\n"
            "1. NO inventes productos, personas, cantidades, almacenes ni ubicaciones.\n"
            "2. Emite SOLAMENTE la clave 'intencion' y las ranuras que tengan valor real en el mensaje o contexto.\n"
            "3. NUNCA emitas claves con valor null ni campos irrelevantes para la intención.\n"
            "4. Si hay ambigüedad o el tema no es de inventario, usa intencion 'desconocido'.\n"
            "5. NO respondas conversacionalmente, entrega ÚNICAMENTE el objeto JSON sin formato adicional.\n\n"
            "Intenciones:\n"
            "- consultar_catalogo: catálogo general sin almacén específico.\n"
            "- consultar_catalogo_almacen: catálogo o existencias de un almacén (slot: almacen).\n"
            "- consultar_stock: existencias de un producto o familia (slots: producto, almacen opcional).\n"
            "- consultar_atributo: ubicación (dónde está guardado un producto), categoría, código, stock_mínimo, prestados o historial de un producto (slots: producto, atributo ['ubicacion'|'categoria'|'codigo'|'stock_minimo'|'prestados'|'historial']).\n"
            "- consultar_ubicaciones: conteo o contenido de un estante o código de ubicación física explícito como A1-C6 (slots: ubicacion, metrica ['productos'|'unidades'|'contenido'], almacen opcional). NUNCA inventes una ubicación si no está mencionada en el mensaje.\n"
            "- comparar_ubicaciones: comparar existencias o referencias entre dos ubicaciones (slots: ubicacion, destino).\n"
            "- consultar_reposicion: productos agotados o con stock bajo (slot: filtro_reposicion ['agotados'|'bajo_stock']).\n"
            "- consultar_movimientos: historial general de movimientos (slots: filtro_movimiento, periodo).\n"
            "- proponer_operacion: préstamo, devolución, entrada, salida o traslado (slots: tipo_operacion ['prestamo'|'devolucion'|'entrada'|'salida'|'traslado'], producto, cantidad, destinatario, almacen).\n"
            "- modificar_propuesta: actualizar un campo de la propuesta activa (slots: campo_modificar, nuevo_valor).\n"
            "- confirmar | cancelar | seleccionar_alternativa | cambio_tema | saludo | agradecimiento | desconocido.\n\n"
            "Ranuras válidas (incluir SOLO si están presentes explícitamente en el mensaje):\n"
            "producto (nombre literal), almacen/origen/destino (nombre literal de una entidad existente), ubicacion (etiqueta literal de ubicación existente), metrica (productos|unidades|contenido), cantidad (entero explícito), destinatario (cadena), "
            "atributo (ubicacion|categoria|stock_minimo|codigo|prestados|historial), tipo_operacion (prestamo|entrada|salida|devolucion|traslado), "
            "campo_modificar (destinatario|cantidad|almacen|motivo), nuevo_valor (cadena), filtro_reposicion (agotados|bajo_stock|todos), "
            "filtro_movimiento (entradas|salidas|prestamos|devoluciones|traslados|todos), periodo (hoy|ayer|semana|mes)."
        )

        contexto_lineas = []
        if foco_producto:
            contexto_lineas.append(f"Producto en foco actual: {foco_producto}")
        if candidatos_txt:
            contexto_lineas.append(f"Opciones previas pendientes de desambiguación: {candidatos_txt}")
        if propuesta_txt:
            contexto_lineas.append(f"Propuesta activa pendiente de confirmación: {propuesta_txt}")

        mensaje_usuario = texto
        if contexto_lineas:
            mensaje_usuario = f"[Contexto]\n" + "\n".join(contexto_lineas) + f"\n\n[Mensaje del usuario]\n{texto}"

        mensajes = [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": mensaje_usuario},
        ]

        # OPT-02B: Calibración a max_tokens=96 (suficiente para JSON compacto de extracción)
        # Timeout calibrado a 6.0s para evitar bloqueos excesivos antes del fallback determinista
        resp = cliente_llm.generar(
            mensajes=mensajes,
            formato_json=True,
            temperatura=0.1,
            max_tokens=96,
            timeout=6.0,
        )

        datos = resp.datos_json if resp is not None else None
        if not datos or not isinstance(datos, dict):
            return None

        # OPT-02B: Normalización canónica de intenciones y ranuras hacia el contrato de HAKU
        intencion, datos_norm = cls._normalizar_contrato_canonico(datos, texto=mensaje_usuario)
        if intencion == "desconocido":
            return None

        return ExtraccionSemantica(
            intencion=intencion,
            producto=_limpiar_str(datos_norm.get("producto")),
            almacen=_limpiar_str(datos_norm.get("almacen")),
            origen=_limpiar_str(datos_norm.get("origen")),
            destino=_limpiar_str(datos_norm.get("destino")),
            ubicacion=_limpiar_str(datos_norm.get("ubicacion")),
            cantidad=_limpiar_entero(datos_norm.get("cantidad")),
            destinatario=_limpiar_str(datos_norm.get("destinatario")),
            atributo=_limpiar_str(datos_norm.get("atributo")),
            tipo_operacion=_limpiar_str(datos_norm.get("tipo_operacion")),
            campo_modificar=_limpiar_str(datos_norm.get("campo_modificar")),
            nuevo_valor=_limpiar_str(datos_norm.get("nuevo_valor")),
            filtro_reposicion=_limpiar_str(datos_norm.get("filtro_reposicion")),
            filtro_movimiento=_limpiar_str(datos_norm.get("filtro_movimiento")),
            periodo=_limpiar_str(datos_norm.get("periodo")),
            confianza=0.92,
            fuente="llm",
            datos_extra={
                "mecanismo": "prompt_compacto_opt02",
                "llm_ms": getattr(resp, "duracion_ms", 0.0) if resp else 0.0,
                "llm_cola_ms": getattr(resp, "espera_cola_ms", 0.0) if resp else 0.0,
                "metrica": _limpiar_str(datos_norm.get("metrica")),
                "es_familia": bool(datos_norm.get("es_familia")),
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN MARCO SEMÁNTICO GENERALIZADO (ZERO-LLM / FALLBACK)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _interpretar_marcos_semanticos(
        cls,
        texto: str,
        contexto_operativo: Any,
        usuario: Any,
    ) -> ExtraccionSemantica:
        """Motor determinista de marcos semánticos con alta cobertura lingüística."""
        t_raw = texto.strip()
        t_lower = t_raw.lower()
        t_clean = re.sub(r"[^\w\sñáéíóúÁÉÍÓÚ]", " ", t_lower)
        t_clean = re.sub(r"\s+", " ", t_clean).strip()

        # 0. Protocolo social
        if re.search(r"^(hola|buenos?\s+d[ií]as?|buenas?\s+tardes?|buenas?\s+noches?|qu[eé]\s+tal|buenas?)\b", t_clean):
            return ExtraccionSemantica(intencion="saludo", fuente="marco_semantico")
        if re.search(r"^(gracias|muchas\s+gracias|te\s+lo\s+agradezco|vale\s+gracias|listo\s+gracias)\b", t_clean):
            return ExtraccionSemantica(intencion="agradecimiento", fuente="marco_semantico")

        # 0-bis. Fuera de dominio temático explícito (preguntas de cultura general, astronomía, viajes, etc.)
        from .normalizador import NormalizadorLenguajeHaku
        if NormalizadorLenguajeHaku._es_fuera_de_dominio(t_clean):
            return ExtraccionSemantica(intencion="no_intencion_inventario", fuente="marco_semantico")

        # 1. Ruptura de tema o descarte explícito ("olvídate de X", "cambiemos de tema")
        if re.search(r"\b(olv[ií]date\s+de|olvida\s+el|olvida\s+la|no\s+le\s+pares\s+bola\s+a|cambiemos\s+de\s+tema|otro\s+tema)\b", t_clean):
            return ExtraccionSemantica(intencion="cambio_tema", fuente="marco_semantico")

        # 2. Selección de alternativa ("el otro", "la otra", "el segundo")
        if re.search(r"\b(el\s+otro|la\s+otra|el\s+segundo|la\s+segunda|la\s+otra\s+opci[oó]n|el\s+otro\s+producto)\b", t_clean):
            return ExtraccionSemantica(
                intencion="seleccionar_alternativa",
                nuevo_valor="el_otro",
                fuente="marco_semantico",
            )

        # 3. Consultas y Comparaciones de Ubicaciones Físicas (A1-C6, Estantes, etc.)
        # Distinguir: (a) listar ubicaciones, (b) comparar ubicaciones, (c) conteo o contenido de una ubicación
        if re.search(r"\b(qu[eé]\s+ubicaciones\s+(?:tenemos|hay|existen)|lista\s+(?:de\s+)?ubicaciones|cu[aá]les\s+ubicaciones\s+hay)\b", t_clean):
            return ExtraccionSemantica(intencion="consultar_ubicaciones", fuente="marco_semantico")

        _patron_ubi_codigo = re.compile(
            r"\b(?:en\s+(?:el\s+|la\s+)?|(?:el\s+|la\s+)?(?:estante|estanter[ií]a|repisa|caj[oó]n|sitio|lugar|ubicaci[oó]n|secci[oó]n)\s+(?:en\s+)?(?:el\s+|la\s+)?|(?:en\s+el\s+|en\s+la\s+)?)\b([a-cA-C][1-6]|[a-zA-Z][0-9]{1,3})\b",
            re.IGNORECASE,
        )
        ubis_encontradas = _patron_ubi_codigo.findall(t_raw)

        # (a) Comparación entre dos ubicaciones
        if re.search(r"\b(diferencia|compara|comparar|comparativa|vs|versus)\b", t_clean) and len(ubis_encontradas) >= 2:
            return ExtraccionSemantica(
                intencion="comparar_ubicaciones",
                ubicacion=ubis_encontradas[0].upper(),
                destino=ubis_encontradas[1].upper(),
                fuente="marco_semantico",
            )

        # (b) Consulta sobre una ubicación específica (A1-C6 o genérica)
        loc_codigo = None
        if ubis_encontradas:
            loc_codigo = ubis_encontradas[0].upper()
        else:
            m_ub_gen = re.search(
                r"\b(?:en\s+(?:el\s+|la\s+)?|(?:el\s+|la\s+)?(?:estante|estanter[ií]a|repisa|caj[oó]n|sitio|lugar|ubicaci[oó]n|secci[oó]n)\s+(?:el\s+|la\s+)?)(estante\s+\d+|caj[oó]n\s+\d+|vitrina\s+\d+|repisa\s+\d+|bodega\s+\d+)\b",
                t_clean,
            )
            if m_ub_gen:
                loc_codigo = m_ub_gen.group(1).title()

        if loc_codigo:
            # Identificar métrica solicitada: unidades físicas vs productos distintos vs contenido
            if re.search(r"\b(unidades?|stock|existencias?|cantidad|piezas?)\b", t_clean):
                metrica_tipo = "unidades"
            elif re.search(r"\b(productos?|referencias?|art[ií]culos?|[ií]tems?|cosas?)\b", t_clean):
                metrica_tipo = "productos"
            else:
                metrica_tipo = "contenido"

            m_alm_nom = re.search(r"\balmac[eé]n\s+([a-zA-Z0-9]+)\b", t_clean)
            alm_ubi = f"Almacén {m_alm_nom.group(1).upper()}" if m_alm_nom else None

            return ExtraccionSemantica(
                intencion="consultar_ubicaciones",
                ubicacion=loc_codigo,
                almacen=alm_ubi,
                fuente="marco_semantico",
                datos_extra={"metrica": metrica_tipo},
            )

        # 4. Reposición y Agotamiento (Sección 5.5)
        # (a) Agotados (Stock == 0)
        _patron_agotados = (
            r"\b("
            r"cu[aá]les\s+est[aá]n\s+agotad[ao]s?|"
            r"qu[eé]\s+(?:cosas?|art[ií]culos?|productos?|referencias?|[ií]tems?|materiales?)\s+(?:que\s+)?(?:est[aá]n\s+)?agotad[ao]s?|"
            r"qu[eé]\s+(?:cosas?|art[ií]culos?|productos?|referencias?|[ií]tems?)\s+(?:est[aá]n\s+)?en\s+cero|"
            r"qu[eé]\s+(?:cosas?|art[ií]culos?|productos?|referencias?|[ií]tems?)\s+(?:no\s+tienen?|est[aá]n\s+sin)\s+stock|"
            r"qu[eé]\s+(?:cosas?|art[ií]culos?|productos?|referencias?|[ií]tems?)\s+se\s+acabar?on|"
            r"qu[eé]\s+est[aá]n?\s+agotad[ao]s?|"
            r"qu[eé]\s+est[aá]n?\s+en\s+cero|"
            r"qu[eé]\s+se\s+acab[oó]|"
            r"qu[eé]\s+se\s+acabaron|"
            r"productos\s+agotados?|"
            r"art[ií]culos\s+agotados?|"
            r"cosas\s+agotadas?|"
            r"referencias\s+agotadas?|"
            r"sin\s+existencias?|"
            r"sin\s+stock|"
            r"no\s+tienen?\s+stock|"
            r"est[aá]n\s+en\s+cero|"
            r"en\s+cero|"
            r"hay\s+agotad[ao]s?|"
            r"est[aá]n\s+agotad[ao]s?|"
            r"agotad[ao]s?"
            r")\b"
        )
        if re.search(_patron_agotados, t_clean):
            return ExtraccionSemantica(
                intencion="consultar_reposicion",
                filtro_reposicion="agotados",
                fuente="marco_semantico",
            )
        # (b) Necesitan reposición / compra / stock bajo / faltantes
        _patron_bajo_stock = (
            r"\b("
            r"qu[eé]\s+(?:nos\s+)?hace\s+falta|"
            r"qu[eé]\s+falta\s+(?:por\s+comprar|pedir|reponer)|"
            r"qu[eé]\s+productos\s+necesitan\s+reposici[oó]n|"
            r"qu[eé]\s+deber[ií]amos\s+comprar|"
            r"por\s+debajo\s+de\s+(?:su\s+)?stock\s+m[ií]nimo|"
            r"pr[oó]xim[oa]s?\s+a\s+agotarse|"
            r"bajo\s+stock|"
            r"stock\s+bajo"
            r")\b"
        )
        if re.search(_patron_bajo_stock, t_clean):
            return ExtraccionSemantica(
                intencion="consultar_reposicion",
                filtro_reposicion="bajo_stock",
                fuente="marco_semantico",
            )

        # 5. Catálogo Filtrado por Almacén vs Catálogo Global (Secciones 5.1 y 5.2)
        m_alm_catalogo = re.search(
            r"\b(?:qu[eé]\s+(?:productos?|art[ií]culos?)\s+hay|dame\s+los\s+nombres|qu[eé]\s+tenemos\s+guardado|mu[eé]strame\s+lo\s+que\s+hay|qu[eé]\s+hay|qu[eé]\s+tenemos|productos|art[ií]culos|cat[aá]logo)\b.*?\balmac[eé]n\s+([a-zA-Z0-9]+)\b",
            t_clean,
        )
        if m_alm_catalogo:
            alm_letra = m_alm_catalogo.group(1).upper()
            return ExtraccionSemantica(
                intencion="consultar_catalogo_almacen",
                almacen=f"Almacén {alm_letra}",
                fuente="marco_semantico",
            )

        if re.search(
            r"\b(cu[eé]ntame\s+los\s+productos|mu[eé]strame\s+los\s+productos|hazme\s+una\s+lista\s+de\s+los\s+art[ií]culos|quiero\s+ver\s+el\s+cat[aá]logo\s+completo|qu[eé]\s+tenemos\s+en\s+el\s+inventario|qu[eé]\s+art[ií]culos\s+manejamos|lista\s+de\s+productos|cat[aá]logo\s+general)\b",
            t_clean,
        ):
            return ExtraccionSemantica(intencion="consultar_catalogo", fuente="marco_semantico")

        # 5b. Consultas de Familias o Categorías ("¿Cuántos cables tenemos?")
        m_fam = re.search(
            r"\b(?:cu[aá]ntos?|hay|tenemos)\s+(cables?|herramientas?|redes|papeler[ií]a|pegantes?|bater[ií]as?|qu[ií]micos?|epp|conectores?|adaptadores?)(?:\s+(?:tenemos|hay|en\s+el\s+inventario|en\s+total|registrados?))?\??$",
            t_clean,
        )
        if m_fam:
            fam_nom = m_fam.group(1).strip()
            return ExtraccionSemantica(
                intencion="consultar_stock",
                producto=fam_nom,
                fuente="marco_semantico",
                datos_extra={"es_familia": True},
            )

        # 5b-bis. Préstamos globales y métricas diferenciadas
        if re.search(r"\b(cu[aá]ntos\s+pr[eé]stamos|n[uú]mero\s+de\s+pr[eé]stamos|cantidad\s+de\s+pr[eé]stamos|total\s+de\s+pr[eé]stamos)\b", t_clean):
            return ExtraccionSemantica(
                intencion="prestamos_activos",
                fuente="marco_semantico",
                datos_extra={"metrica": "CONTEO_REGISTROS"},
            )
        if re.search(r"\b(cu[aá]ntas\s+unidades\s+(?:hay\s+|tenemos\s+)?presta(?:d[oa]|o)s?|total\s+de\s+unidades\s+presta(?:d[oa]|o)s?|cantidad\s+de\s+unidades\s+presta(?:d[oa]|o)s?)\b", t_clean):
            return ExtraccionSemantica(
                intencion="prestamos_activos",
                fuente="marco_semantico",
                datos_extra={"metrica": "CONTEO_UNIDADES"},
            )
        if re.search(r"\b(qui[eé]n\s+tiene\s+(?:cosas|productos|material|art[ií]culos|\w+)?\s*presta(?:d[oa]|o)s?|a\s+qui[eé]n\s+(?:se\s+le\s+prest[oó]|est[aá]\s+prestado)|responsables?\s+de\s+pr[eé]stamos?)\b", t_clean) and not re.search(r"\b(eso|ese|este|aquel|su|el|la|dicho)\b", t_clean):
            return ExtraccionSemantica(
                intencion="prestamos_activos",
                fuente="marco_semantico",
                datos_extra={"metrica": "LISTA_RESPONSABLES"},
            )
        if re.search(r"\b(pr[eé]stamos?\s+activos?|pr[eé]stamos?\s+pendientes?|qu[eé]\s+pr[eé]stamos\s+hay|cu[aá]les\s+son\s+los\s+pr[eé]stamos|qu[eé]\s+tenemos\s+prestado|qu[eé]\s+est[aá]\s+prestado)\b", t_clean):
            return ExtraccionSemantica(
                intencion="prestamos_activos",
                fuente="marco_semantico",
            )

        # 5b-ter. Movimientos y despachos (Sección 5.6)
        if re.search(r"\b(movimientos|entradas|salidas|despachos|pr[eé]stamos|devoluciones|traslados)\b", t_clean) and re.search(r"\b(hoy|ayer|semana|mes|se\s+hicieron|hubo|se\s+registraron)\b", t_clean):
            filtro_mov = "todos"
            if re.search(r"\bentradas?\b", t_clean):
                filtro_mov = "entradas"
            elif re.search(r"\b(salidas?|despachos?)\b", t_clean):
                filtro_mov = "salidas"
            elif re.search(r"\bpr[eé]stamos?\b", t_clean):
                filtro_mov = "prestamos"
            elif re.search(r"\bdevoluciones?\b", t_clean):
                filtro_mov = "devoluciones"

            periodo = "hoy" if "hoy" in t_clean else ("ayer" if "ayer" in t_clean else None)
            return ExtraccionSemantica(
                intencion="consultar_movimientos",
                filtro_movimiento=filtro_mov,
                periodo=periodo,
                fuente="marco_semantico",
            )

        # 5c. Consultas de Stock / Disponibilidad de Producto (Resiliencia Zero-LLM)
        if (
            re.search(r"\b(cu[aá]ntos?|qu[eé]\s+stock|cu[aá]nto\s+stock|stock\s+de|quedan?|hay\s+disponibles?|tenemos\s+en\s+bodega|tenemos\s+en\s+el\s+almac[eé]n)\b", t_clean)
            and not re.search(r"\b(presta(?:d[oa]|o)s?|en\s+pr[eé]stamo|despachos?|movimientos?|entradas?|salidas?)\b", t_clean)
        ):
            prod_m = cls._extraer_producto_mencionado(t_raw)
            m_alm = re.search(r"\b(?:en\s+el\s+|para\s+el\s+|del\s+)?((?:almac[eé]n|bodega)\s+[a-zA-Z0-9]+)\b", t_clean)
            alm_nom = m_alm.group(1).title() if m_alm else None
            return ExtraccionSemantica(
                intencion="consultar_stock",
                producto=prod_m,
                almacen=alm_nom,
                fuente="marco_semantico",
            )

        # 6. Consultas de Atributos de Producto (Sección 5.3)
        # Categoría
        if re.search(r"\b(en\s+qu[eé]\s+categor[ií]a|a\s+qu[eé]\s+categor[ií]a\s+pertenece|dime\s+(?:solamente\s+)?(?:su\s+|la\s+)?categor[ií]a|cu[aá]l\s+es\s+(?:su\s+|la\s+)?categor[ií]a)\b", t_clean):
            prod_m = cls._extraer_producto_mencionado(t_raw)
            return ExtraccionSemantica(
                intencion="consultar_atributo",
                atributo="categoria",
                producto=prod_m,
                fuente="marco_semantico",
            )

        # Stock mínimo
        if re.search(r"\b(cu[aá]l\s+es\s+(?:su\s+|el\s+)?stock\s+m[ií]nimo|stock\s+m[ií]nimo\s+de|cu[aá]nto\s+es\s+el\s+m[ií]nimo)\b", t_clean):
            prod_m = cls._extraer_producto_mencionado(t_raw)
            return ExtraccionSemantica(
                intencion="consultar_atributo",
                atributo="stock_minimo",
                producto=prod_m,
                fuente="marco_semantico",
            )

        # Ubicación física registrada del producto
        if re.search(r"\b(d[oó]nde\s+est[aá]\s+registrada\s+(?:la\s+)?ubicaci[oó]n|d[oó]nde\s+est[aá]\s+(?:guardad[oa]|ubicad[oa])|d[oó]nde\s+se\s+(?:encuentra|ubica)|d[oó]nde\s+est[aá]\s+(?:el|la|los|las)|d[oó]nde\s+est[aá](?:\s+actualmente|\s+eso|\s+ese)?\s*\??$|cu[aá]l\s+es\s+(?:su\s+|la\s+)?ubicaci[oó]n)\b", t_clean):
            prod_m = cls._extraer_producto_mencionado(t_raw)
            return ExtraccionSemantica(
                intencion="consultar_atributo",
                atributo="ubicacion",
                producto=prod_m,
                fuente="marco_semantico",
            )

        # Código del producto
        if re.search(r"\b(qu[eé]\s+c[oó]digo\s+tiene|cu[aá]l\s+es\s+(?:su\s+|el\s+)?c[oó]digo)\b", t_clean):
            prod_m = cls._extraer_producto_mencionado(t_raw)
            return ExtraccionSemantica(
                intencion="consultar_atributo",
                atributo="codigo",
                producto=prod_m,
                fuente="marco_semantico",
            )

        # Total prestado / quién lo tiene
        if re.search(r"\b(qui[eé]n\s+tiene|cu[aá]ntos?\s+.*?\s*est[aá]n?\s+presta(?:d[oa]|o)s?|est[aá]n?\s+presta(?:d[oa]|o)s?|hay\s+presta(?:d[oa]|o)s?|en\s+pr[eé]stamo)\b", t_clean):
            prod_m = cls._extraer_producto_mencionado(t_raw)
            return ExtraccionSemantica(
                intencion="consultar_atributo",
                atributo="prestados",
                producto=prod_m,
                fuente="marco_semantico",
            )

        # Historial de movimientos del producto
        if re.search(r"\b(historial\s+(?:de\s+movimientos\s+)?de\s+(?:este\s+producto|este|el|la)|movimientos\s+de\s+este\s+producto)\b", t_clean):
            prod_m = cls._extraer_producto_mencionado(t_raw)
            return ExtraccionSemantica(
                intencion="consultar_atributo",
                atributo="historial",
                producto=prod_m,
                fuente="marco_semantico",
            )

        # 7. Movimientos (Sección 5.6)
        if re.search(r"\b(movimientos|entradas|salidas|pr[eé]stamos|devoluciones|traslados)\b", t_clean) and re.search(r"\b(hoy|ayer|semana|mes|se\s+hicieron|hubo|se\s+registraron)\b", t_clean):
            filtro_mov = "todos"
            if re.search(r"\bentradas?\b", t_clean):
                filtro_mov = "entradas"
            elif re.search(r"\bsalidas?\b", t_clean):
                filtro_mov = "salidas"
            elif re.search(r"\bpr[eé]stamos?\b", t_clean):
                filtro_mov = "prestamos"
            elif re.search(r"\bdevoluciones?\b", t_clean):
                filtro_mov = "devoluciones"

            periodo = "hoy" if "hoy" in t_clean else ("ayer" if "ayer" in t_clean else None)
            return ExtraccionSemantica(
                intencion="consultar_movimientos",
                filtro_movimiento=filtro_mov,
                periodo=periodo,
                fuente="marco_semantico",
            )

        # 8. Operaciones de Inventario (Sección 5.7)
        # Préstamo
        m_prest = re.search(r"\b(?:quiero\s+prestar|voy\s+a\s+prestar|prestar|pr[eé]stale|presta)\s+(?:a\s+)?(?:(\d+|un[oa]|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s+)?(?:unidades?\s+de\s+|de\s+)?([a-zA-Z0-9\s'-]+?)(?:\s+a\s+([a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]+))?$", t_raw, re.IGNORECASE)
        if m_prest:
            cant_str = (m_prest.group(1) or "").lower().strip()
            _num_map = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
            cant_val = int(cant_str) if cant_str.isdigit() else _num_map.get(cant_str, 1)
            prod_txt = m_prest.group(2).strip()
            dest_txt = m_prest.group(3).strip() if m_prest.group(3) else None
            return ExtraccionSemantica(
                intencion="proponer_operacion",
                tipo_operacion="prestamo",
                producto=prod_txt if len(prod_txt) >= 2 else None,
                destinatario=dest_txt,
                cantidad=cant_val,
                fuente="marco_semantico",
            )

        # Entrada
        if re.search(r"\b(quiero\s+registrar\s+una\s+entrada|voy\s+a\s+ingresar|registrar\s+entrada|dar\s+entrada)\b", t_clean):
            return ExtraccionSemantica(
                intencion="proponer_operacion",
                tipo_operacion="entrada",
                fuente="marco_semantico",
            )

        # Salida
        if re.search(r"\b(voy\s+a\s+sacar|quiero\s+sacar|registrar\s+salida|dar\s+salida|retirar)\b", t_clean):
            m_cant = re.search(r"\b(\d+|dos|tres|cuatro|cinco)\s+unidades?\b", t_clean)
            c_val = None
            if m_cant:
                w_map = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}
                c_val = int(m_cant.group(1)) if m_cant.group(1).isdigit() else w_map.get(m_cant.group(1))
            return ExtraccionSemantica(
                intencion="proponer_operacion",
                tipo_operacion="salida",
                cantidad=c_val,
                fuente="marco_semantico",
            )

        # Devolución
        if re.search(r"\b(voy\s+a\s+devolver|quiero\s+devolver|registrar\s+devoluci[oó]n|devolver)\b", t_clean):
            return ExtraccionSemantica(
                intencion="proponer_operacion",
                tipo_operacion="devolucion",
                fuente="marco_semantico",
            )

        # Traslado
        if re.search(r"\b(mueve|mover|trasladar|pasa|pasar)\s+(\d+|dos|tres|cuatro|cinco)?\s*unidades?\s+al\s+almac[eé]n\s+([a-zA-Z0-9]+)\b", t_clean):
            return ExtraccionSemantica(
                intencion="proponer_operacion",
                tipo_operacion="traslado",
                fuente="marco_semantico",
            )

        return ExtraccionSemantica(intencion="desconocido", confianza=0.0, fuente="marco_semantico")

    @classmethod
    def _extraer_producto_mencionado(cls, texto: str) -> Optional[str]:
        """Intenta extraer el sintagma nominal correspondiente al producto en la pregunta."""
        # Limpieza de puntuación intermedia que separa cláusulas o especificaciones (ej. ponchadora, RJ45)
        texto_limpio = re.sub(r"[,;]", " ", texto.strip())
        # Si es una pregunta elíptica con filtro de almacén («¿cuántos hay en el Almacén A?»),
        # remover el sintagma preposicional del almacén para no confundirlo con nombre de producto
        texto_limpio = re.sub(
            r"\s+(?:en\s+el|en\s+la|del|de\s+la|para\s+el)\s+(?:almac[eé]n|bodega|dep[oó]sito)\s+[a-zA-Z0-9]+\??$",
            "",
            texto_limpio,
            flags=re.IGNORECASE,
        ).strip()
        texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()

        # Prioridad 0: Detección directa de código canónico de inventario (ej. PAP-032, QUI-002, CAB-0001)
        m_code = re.search(r"\b([A-Za-z]{2,5}-\d{1,4})\b", texto_limpio)
        if m_code:
            return m_code.group(1).upper()

        # Limpieza de patrones interrogativos
        patrones = [
            r"\b(?:cu[aá]nto|cu[aá]ntos?|qu[eé])\s+(?:queda[n]?|hay|tenemos)\s+(?:de\s+|del\s+)?([a-zA-Z0-9\s'-]{3,50})\??$",
            r"\bcu[aá]ntos?\s+([a-zA-Z0-9\s'-]{3,50}?)\s+(?:quedan?|tenemos|hay|en|del?|est[aá]n?\s+presta(?:d[oa]|o)s?)\b",
            r"\b(?:d[oó]nde\s+est[aá]\s+(?:registrada\s+(?:la\s+)?ubicaci[oó]n\s+de\s+|(?:guardad[oa]|ubicad[oa])\s+(?:el|la|los|las)?\s*|)|ubicaci[oó]n\s+(?:del?|de\s+la)\s*)(?:el|la|los|las)?\s*([a-zA-Z0-9\s'-]{3,50})\??$",
            r"\bstock\s+(?:del?|de\s+la)\s+([a-zA-Z0-9\s'-]{3,50})\??$",
            r"(?:del|de\s+la|de|sobre\s+el|sobre\s+la)\s+([a-zA-Z0-9\s'-]{3,50})\??$",
            r"(?:el|la|los|las)\s+([a-zA-Z0-9\s'-]{3,50})\??$",
        ]
        _PALABRAS_BLOQUEADAS_PRODUCTO = {
            "este", "ese", "aquel", "este producto", "ese producto", "otro", "otra", "el otro", "la otra",
            "almacen", "almacén", "bodega", "bodegas",
            "producto", "productos", "articulo", "articulos", "artículo", "artículos",
            "item", "items", "ítem", "ítems", "referencia", "referencias", "cosa", "cosas",
            "unidad", "unidades", "stock", "existencia", "existencias", "cantidad", "piezas",
            "estante", "estantes", "estanteria", "estantería", "cajon", "cajón", "cajones",
            "repisa", "repisas", "vitrina", "vitrinas", "ubicacion", "ubicación", "ubicaciones",
            "sitio", "sitios", "lugar", "lugares", "seccion", "sección",
            "queda", "quedan", "hay", "tenemos", "tienen", "disponible", "disponibles",
            "ubicado", "ubicada", "ubicados", "ubicadas", "guardado", "guardada", "guardados", "guardadas",
            "ubicado actualmente", "guardado actualmente", "actualmente", "ahora",
            "cuanto", "cuantos", "cuanta", "cuantas", "cuesta",
            "a1", "a2", "a3", "a4", "a5", "a6",
            "b1", "b2", "b3", "b4", "b5", "b6",
            "c1", "c2", "c3", "c4", "c5", "c6",
        }
        for p in patrones:
            m = re.search(p, texto_limpio, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                cand = re.sub(r"^(?:el|la|los|las|un|una|unos|unas|del|de\s+la|de)\s+", "", cand, flags=re.IGNORECASE).strip()
                c_low = cand.lower()
                if (
                    c_low not in _PALABRAS_BLOQUEADAS_PRODUCTO
                    and not any(c_low.startswith(b) for b in ("estante ", "estanteria ", "estantería ", "cajon ", "cajón ", "ubicacion ", "ubicación ", "sitio ", "lugar "))
                ):
                    return cand
        return None
