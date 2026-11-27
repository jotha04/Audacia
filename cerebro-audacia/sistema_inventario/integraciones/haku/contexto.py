"""Capa de memoria temporal en RAM para Haku Inventario — HakuOperationalContext.

Responsabilidades clave:
1. Almacenar el estado operativo mínimo de interacción (foco del producto, última intención,
   propuesta activa, candidatos).
2. Desacoplado 100% de Django Session (la autenticación web del usuario sigue en DB/cookie por 7 días;
   el contexto conversacional reside en RAM mediante LocMemCache con TTL de 7200 segundos / 2 horas).
   Nota: Las propuestas transaccionales independientes caducan a los 1800 segundos (30 minutos).
3. NO almacena audios, transcripciones completas, conversaciones completas ni respuestas generadas.
4. Límite de 20 contextos simultáneos con política de desalojo LRU (óptimo para 10 usuarios concurrentes).
5. Versión de contexto (version_contexto) para invalidar de forma segura esquemas obsoletos.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.core.cache import caches

logger = logging.getLogger("integraciones.haku.contexto")

CACHE_ALIAS = "haku_contexto"
VERSION_CONTEXTO_ACTUAL = 2
MAX_CANDIDATOS = 5
MAX_ULTIMAS_ACCIONES = 5
TIMEOUT_INACTIVIDAD_SEGUNDOS = getattr(settings, "HAKU_CONTEXT_TIMEOUT", 7200)  # 2 horas (7200 s) de inactividad conversacional en RAM
TIMEOUT_FOCO_IMPLICITO_SEGUNDOS = getattr(settings, "HAKU_IMPLICIT_FOCUS_TIMEOUT", 900)  # 15 minutos de foco implícito sin actividad


@dataclass
class EstadoOperativoHaku:
    """Estado operativo efímero mínimo de Haku Inventario."""

    usuario_id: int
    session_key: str
    almacen_id: Optional[int] = None

    # 1. Producto actualmente en contexto conversacional
    producto_id: Optional[int] = None
    producto_codigo: Optional[str] = None
    producto_nombre: Optional[str] = None

    # 2. Última interacción registrada
    ultima_intencion: Optional[str] = None
    ultima_busqueda: Optional[str] = None

    # 3. Productos candidatos para desambiguación rápida (máximo 3)
    # Formato: [{"id": 101, "nombre": "Cable HDMI 2 metros", "codigo": "CAB-001"}]
    candidatos: List[Dict[str, Any]] = field(default_factory=list)
    intencion_pendiente_aclaracion: Optional[str] = None
    slot_pendiente_aclaracion: Optional[str] = None
    datos_intencion_pendiente: Optional[Dict[str, Any]] = None

    # 4. Operación pendiente de pre-confirmación en 2 pasos
    propuesta_activa: Optional[Dict[str, Any]] = None

    # 4-bis. Operación pendiente en recolección de ranuras (ej. esperando destinatario de préstamo o edición)
    operacion_pendiente: Optional[Dict[str, Any]] = None

    # 5. Memoria operacional reciente (máximo 5 elementos)
    # Formato: [{"accion": "CONSULTAR_STOCK", "producto": "Cable HDMI", "producto_id": 101, "codigo": "CAB-001", "cantidad": 15, "fecha": ...}]
    # NO almacena chat history, ni audios, ni conversaciones completas.
    ultimas_acciones: List[Dict[str, Any]] = field(default_factory=list)

    # 5-bis. Colección de productos resultante de la última consulta de lista (máximo 10 elementos)
    # Formato: [{"id": 101, "codigo": "CAB-001", "nombre": "Cable HDMI"}]
    coleccion_productos: List[Dict[str, Any]] = field(default_factory=list)
    coleccion_total_original: int = 0

    # 6. Timestamp de control y versionado
    ultimo_acceso: float = field(default_factory=time.time)
    foco_actualizado_en: float = 0.0
    version_contexto: int = VERSION_CONTEXTO_ACTUAL

    def esta_expirado(self, timeout_segundos: int = TIMEOUT_INACTIVIDAD_SEGUNDOS) -> bool:
        """Determina si han transcurrido más de 7200 segundos (2 horas) sin actividad."""
        return (time.time() - self.ultimo_acceso) > timeout_segundos

    def foco_implicito_expirado(self, timeout_segundos: int = TIMEOUT_FOCO_IMPLICITO_SEGUNDOS) -> bool:
        """Determina si el foco implícito de producto/lista caducó por inactividad breve sin cerrar la sesión."""
        if self.foco_actualizado_en <= 0:
            return False
        if self.propuesta_activa or self.operacion_pendiente:
            return False
        return (time.time() - self.foco_actualizado_en) > timeout_segundos

    def limpiar_foco_implicito(self) -> None:
        """Limpia solo el foco implícito viejo (producto/candidatos/colección) conservando sesión y acciones recientes."""
        self.producto_id = None
        self.producto_codigo = None
        self.producto_nombre = None
        self.candidatos = []
        self.intencion_pendiente_aclaracion = None
        self.slot_pendiente_aclaracion = None
        self.datos_intencion_pendiente = None
        self.coleccion_productos = []
        self.coleccion_total_original = 0
        self.foco_actualizado_en = 0.0

    def actualizar_actividad(self) -> None:
        """Actualiza el timestamp de última interacción activa."""
        self.ultimo_acceso = time.time()

    def limpiar_foco_conversacional(self) -> None:
        """Purga el foco del producto, propuestas y memoria operacional sin cerrar sesión web."""
        self.producto_id = None
        self.producto_codigo = None
        self.producto_nombre = None
        self.ultima_intencion = None
        self.ultima_busqueda = None
        self.candidatos = []
        self.intencion_pendiente_aclaracion = None
        self.slot_pendiente_aclaracion = None
        self.datos_intencion_pendiente = None
        self.coleccion_productos = []
        self.coleccion_total_original = 0
        self.propuesta_activa = None
        self.operacion_pendiente = None
        self.ultimas_acciones = []
        self.foco_actualizado_en = 0.0

    def asignar_producto(self, producto_id: int, codigo: str, nombre: str) -> None:
        """Establece el producto actualmente referenciado en el foco operativo."""
        self.producto_id = producto_id
        self.producto_codigo = codigo
        self.producto_nombre = nombre
        self.foco_actualizado_en = time.time()
        self.actualizar_actividad()

    def asignar_coleccion(self, lista_items: List[Dict[str, Any]], total_original: Optional[int] = None) -> None:
        """Guarda hasta 10 productos estructurados resultantes de una consulta colectiva, registrando el total original."""
        limpios = []
        for item in lista_items[:10]:
            prod_id = item.get("id") or item.get("producto_id")
            if prod_id:
                limpios.append({
                    "id": int(prod_id),
                    "codigo": str(item.get("codigo", "")),
                    "nombre": str(item.get("nombre", "")),
                })
        self.coleccion_productos = limpios
        self.coleccion_total_original = int(total_original) if total_original is not None else len(lista_items)
        self.foco_actualizado_en = time.time()
        self.actualizar_actividad()

    def asignar_candidatos(
        self,
        lista_candidatos: List[Dict[str, Any]],
        intencion_original: Optional[str] = None,
        slot_pendiente: Optional[str] = None,
        datos_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Guarda hasta MAX_CANDIDATOS (5) opciones mínimas útiles para desambiguación y preserva la intención original."""
        limpios = []
        for c in lista_candidatos[:MAX_CANDIDATOS]:
            limpios.append({
                "id": c.get("id"),
                "nombre": str(c.get("nombre", "")),
                "codigo": str(c.get("codigo", "")),
            })
        self.candidatos = limpios
        if intencion_original:
            self.intencion_pendiente_aclaracion = intencion_original
        if slot_pendiente:
            self.slot_pendiente_aclaracion = slot_pendiente
        if datos_extra:
            self.datos_intencion_pendiente = dict(datos_extra)
        self.foco_actualizado_en = time.time()
        self.actualizar_actividad()

    def resolver_seleccion_candidato(self, texto: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Resuelve determinísticamente la selección de un candidato de la lista ofrecida en el turno anterior.

        Soporta:
        - Referencias ordinales: «el primero», «primero», «la primera», «opción 1», «el 1», «1» -> candidatos[0]
        - «el segundo», «segundo», «la segunda», «opción 2», «el 2», «2» -> candidatos[1]
        - «el tercero», «tercero», «la tercera», «opción 3», «el 3», «3» -> candidatos[2]
        - «el último», «la última», «último», «última» -> candidatos[-1]
        - Código de la opción: exacto o contenido (ej. "CAB-0001", "CAB-1001", "1001")
        - Nombre de la opción: coincidencia sustancial de nombre

        Retorna: (candidato_seleccionado, error_o_motivo)
        """
        if not self.candidatos:
            return None, None

        import re
        t_clean = re.sub(r"[¿?¡!.,:;()\[\]\"'`]", " ", texto.lower()).strip()
        t_clean = re.sub(r"\s+", " ", t_clean)
        num_cands = len(self.candidatos)

        # 0. Si el usuario menciona explícitamente un código de producto que NO está en los candidatos,
        # se trata de una solicitud nueva sobre otro artículo: no secuestrar el turno.
        codigos_cands = {str(c.get("codigo", "")).lower() for c in self.candidatos if c.get("codigo")}
        codigos_en_texto = re.findall(r"\b([a-z]{2,5}[-_]\d{1,5})\b", t_clean)
        if codigos_en_texto and not any(cod in codigos_cands for cod in codigos_en_texto):
            return None, None

        # 0b. Si el mensaje es una operación o pregunta nueva completa (ej. «Quiero hacer un préstamo del TAL-002»,
        # «¿Dónde está el TAL-002?», «Devuelve el TAL-001»), NO secuestrarlo como selección de una consulta previa.
        es_nueva_operacion_o_pregunta = bool(
            "?" in texto or "¿" in texto
            or re.search(
                r"\b(?:pr[eé]stamo|prestar|pr[eé]st\w*|devoluci[oó]n|devolver|devuelv\w*|"
                r"entrada|ingresar|ingres\w*|salida|sacar|retirar|despachar|dotaci[oó]n|"
                r"traslado|trasladar|crear\s+(?:un\s+)?producto|inhabilitar|habilitar|"
                r"qu[eé]\s+stock|cu[aá]nt[oa]s?|d[oó]nde\s+est[aá]|qui[eé]n\s+tiene|movimientos?|historial)\b",
                t_clean,
                flags=re.IGNORECASE,
            )
        )
        if es_nueva_operacion_o_pregunta:
            return None, None

        # 1. Ordinales exactos y relativos (estrictamente acotados a frases de selección,
        # nunca sobre comandos nuevos como "primero dime...", "último movimiento", "saca 5 unidades...")
        _pref_sel = (
            r"(?:(?:dame|quiero|elige|escoge|selecciona|prefiero|usa|usar|ser[ií]a|es|era|me\s+refiero\s+a|hablo\s+de)\s+)?"
            r"(?:el\s+|la\s+)?"
        )
        _suf_sel = r"(?:\s+(?:opci[oó]n|producto|art[ií]culo|candidato|equipo|por\s+favor))?"

        # Primero
        if re.fullmatch(
            rf"{_pref_sel}(?:primer[oa]|opci[oó]n\s+1|n[uú]mero\s+1|1|uno|una){_suf_sel}",
            t_clean,
        ):
            return self.candidatos[0], None

        # Segundo
        if re.fullmatch(
            rf"{_pref_sel}(?:segund[oa]|opci[oó]n\s+2|n[uú]mero\s+2|2|dos){_suf_sel}",
            t_clean,
        ):
            if num_cands >= 2:
                return self.candidatos[1], None
            return None, "fuera_de_rango"

        # Tercero
        if re.fullmatch(
            rf"{_pref_sel}(?:tercer[oa]|opci[oó]n\s+3|n[uú]mero\s+3|3|tres){_suf_sel}",
            t_clean,
        ):
            if num_cands >= 3:
                return self.candidatos[2], None
            return None, "fuera_de_rango"

        # Cuarto
        if re.fullmatch(
            rf"{_pref_sel}(?:cuart[oa]|opci[oó]n\s+4|n[uú]mero\s+4|4|cuatro){_suf_sel}",
            t_clean,
        ):
            if num_cands >= 4:
                return self.candidatos[3], None
            return None, "fuera_de_rango"

        # Quinto
        if re.fullmatch(
            rf"{_pref_sel}(?:quint[oa]|opci[oó]n\s+5|n[uú]mero\s+5|5|cinco){_suf_sel}",
            t_clean,
        ):
            if num_cands >= 5:
                return self.candidatos[4], None
            return None, "fuera_de_rango"

        # Último
        if re.fullmatch(rf"{_pref_sel}[uú]ltim[oa]{_suf_sel}", t_clean):
            return self.candidatos[-1], None

        # Ordinales superiores fuera de rango (solo si es una selección explícita de opción)
        if re.fullmatch(
            rf"{_pref_sel}(?:sext[oa]|s[eé]ptim[oa]|octav[oa]|noven[oa]|opci[oó]n\s+[6-9]|n[uú]mero\s+[6-9]|[6-9]){_suf_sel}",
            t_clean,
        ):
            return None, "fuera_de_rango"

        # 2. Búsqueda por código de opción (CAB-0001, etc.)
        for cand in self.candidatos:
            cod = str(cand.get("codigo", "")).lower()
            if cod and (cod == t_clean or re.search(rf"(?<!\w){re.escape(cod)}(?!\w)", t_clean)):
                return cand, None

        # 3. Búsqueda por nombre o rasgo diferenciador (con normalización de tildes, medidas y fronteras de palabra;
        # evita falsos positivos de subcadena como "si" en "resistencia" o "no" en "conector" y preserva tokens cortos como "5m", "AA", "1/2")
        import unicodedata
        from .agente.herramientas import normalizar as _norm_haku, preprocesar_texto_inventario as _pre_haku, _raiz as _raiz_haku

        def _norm_sel(s: str) -> str:
            base = unicodedata.normalize("NFKD", str(s or ""))
            base = "".join(ch for ch in base if not unicodedata.combining(ch)).lower()
            base = re.sub(r"[^\w/\-\s]", " ", base)
            return re.sub(r"\s+", " ", base).strip()

        _palabras_relleno_sel = {
            "si", "no", "ok", "vale", "ya", "el", "la", "los", "las", "de", "del", "en",
            "a", "al", "y", "o", "un", "una", "unos", "unas", "ese", "esa", "eso", "esos",
            "esas", "este", "esta", "esto", "estos", "estas", "aquel", "aquella", "por",
            "favor", "dame", "quiero", "elige", "escoge", "selecciona", "prefiero", "usa",
            "usar", "seria", "es", "era", "me", "refiero", "hablo", "opcion", "producto",
            "articulo", "candidato", "equipo",
        }
        t_sel_norm = _norm_sel(texto)
        if not t_sel_norm or t_sel_norm in _palabras_relleno_sel:
            return None, None

        t_pre_tokens = [
            _raiz_haku(w)
            for w in _norm_haku(_pre_haku(texto)).split()
            if w and w not in _palabras_relleno_sel
        ]
        t_lit_tokens = [
            _raiz_haku(w)
            for w in _norm_haku(texto).split()
            if w and w not in _palabras_relleno_sel
        ]
        tokens_busqueda = t_pre_tokens or t_lit_tokens
        if not tokens_busqueda:
            return None, None

        coincidentes_nombre = []
        for cand in self.candidatos:
            nom_raw = str(cand.get("nombre", ""))
            nom_norm = _norm_sel(nom_raw)
            if not nom_norm:
                continue
            if nom_norm == t_sel_norm or re.search(rf"(?<!\w){re.escape(nom_norm)}(?!\w)", t_sel_norm):
                coincidentes_nombre.append(cand)
                continue
            cand_tokens = {
                _raiz_haku(w)
                for w in (_norm_haku(_pre_haku(nom_raw)).split() + _norm_haku(nom_raw).split())
                if w and w not in _palabras_relleno_sel
            }
            if cand_tokens and all(
                tok in cand_tokens or any(len(tok) >= 3 and ct.startswith(tok) for ct in cand_tokens)
                for tok in tokens_busqueda
            ):
                coincidentes_nombre.append(cand)

        if len(coincidentes_nombre) == 1:
            return coincidentes_nombre[0], None
        elif len(coincidentes_nombre) > 1:
            return None, "ambiguo"

        return None, None

    def obtener_otro_candidato(self, producto_id_actual: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Si hay exactamente 2 candidatos, retorna la alternativa que no es el producto actual.

        Si hay 3 o más candidatos, o menos de 2, retorna None para exigir desambiguación explícita.
        """
        if len(self.candidatos) != 2:
            return None
        prod_ref = producto_id_actual or self.producto_id
        if prod_ref:
            for c in self.candidatos:
                if c.get("id") != prod_ref:
                    return c
        return self.candidatos[1]

    def registrar_accion(
        self,
        accion: str,
        producto: Optional[str] = None,
        producto_id: Optional[int] = None,
        codigo: Optional[str] = None,
        cantidad: Optional[int] = None,
        detalles: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra una acción operacional en la memoria reciente (máximo 5 elementos).

        NO almacena mensajes de texto de la conversación, ni transcripciones, ni audios.
        """
        item = {
            "accion": accion,
            "producto": producto,
            "producto_id": producto_id,
            "codigo": codigo,
            "fecha": time.time(),
        }
        if cantidad is not None:
            item["cantidad"] = cantidad
        if detalles:
            claves_prohibidas = {
                "texto", "mensaje", "audio", "conversacion", "prompt", "respuesta",
                "mensaje_usuario", "transcripcion", "raw_audio", "wav", "pcm", "samples",
                "dialogo", "historial", "user_message", "transcript",
            }
            for k, v in detalles.items():
                if k.lower() not in claves_prohibidas and not isinstance(v, (bytes, bytearray)):
                    item[k] = v

        self.ultimas_acciones.append(item)
        if len(self.ultimas_acciones) > MAX_ULTIMAS_ACCIONES:
            self.ultimas_acciones = self.ultimas_acciones[-MAX_ULTIMAS_ACCIONES:]
        self.actualizar_actividad()

    def obtener_producto_anterior(self) -> Optional[Dict[str, Any]]:
        """Recupera el producto de las acciones operacionales anteriores."""
        if not self.ultimas_acciones:
            return None
        # Buscar primero una acción previa que tuviera un producto diferente al actual
        for item in reversed(self.ultimas_acciones):
            if item.get("producto_id") and (self.producto_id is None or item["producto_id"] != self.producto_id):
                return item
        # Si no hay diferente, retornar la última con producto
        for item in reversed(self.ultimas_acciones):
            if item.get("producto_id"):
                return item
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario para almacenamiento seguro en caché."""
        return {
            "usuario_id": self.usuario_id,
            "session_key": self.session_key,
            "almacen_id": self.almacen_id,
            "producto_id": self.producto_id,
            "producto_codigo": self.producto_codigo,
            "producto_nombre": self.producto_nombre,
            "ultima_intencion": self.ultima_intencion,
            "ultima_busqueda": self.ultima_busqueda,
            "candidatos": self.candidatos,
            "intencion_pendiente_aclaracion": self.intencion_pendiente_aclaracion,
            "slot_pendiente_aclaracion": self.slot_pendiente_aclaracion,
            "datos_intencion_pendiente": self.datos_intencion_pendiente,
            "coleccion_productos": self.coleccion_productos,
            "coleccion_total_original": self.coleccion_total_original,
            "propuesta_activa": self.propuesta_activa,
            "operacion_pendiente": self.operacion_pendiente,
            "ultimas_acciones": self.ultimas_acciones,
            "ultimo_acceso": self.ultimo_acceso,
            "foco_actualizado_en": self.foco_actualizado_en,
            "version_contexto": self.version_contexto,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EstadoOperativoHaku:
        """Reconstruye el objeto desde diccionario validando versión."""
        return cls(
            usuario_id=data.get("usuario_id", 0),
            session_key=data.get("session_key", ""),
            almacen_id=data.get("almacen_id"),
            producto_id=data.get("producto_id"),
            producto_codigo=data.get("producto_codigo"),
            producto_nombre=data.get("producto_nombre"),
            ultima_intencion=data.get("ultima_intencion"),
            ultima_busqueda=data.get("ultima_busqueda"),
            candidatos=data.get("candidatos", []),
            intencion_pendiente_aclaracion=data.get("intencion_pendiente_aclaracion"),
            slot_pendiente_aclaracion=data.get("slot_pendiente_aclaracion"),
            datos_intencion_pendiente=data.get("datos_intencion_pendiente"),
            coleccion_productos=data.get("coleccion_productos", []),
            coleccion_total_original=data.get("coleccion_total_original", 0),
            propuesta_activa=data.get("propuesta_activa"),
            operacion_pendiente=data.get("operacion_pendiente"),
            ultimas_acciones=data.get("ultimas_acciones", []),
            ultimo_acceso=data.get("ultimo_acceso", time.time()),
            foco_actualizado_en=float(data.get("foco_actualizado_en", 0.0) or 0.0),
            version_contexto=data.get("version_contexto", VERSION_CONTEXTO_ACTUAL),
        )


class HakuOperationalContext:
    """Gestor de almacenamiento en RAM de contextos operativos con LocMemCache."""

    _lock = threading.Lock()
    _context_hits: int = 0
    _context_misses: int = 0

    @classmethod
    def _obtener_cache(cls):
        """Retorna el backend de caché configurado para contexto o default si no existe."""
        try:
            return caches[CACHE_ALIAS]
        except Exception:
            return caches["default"]

    @classmethod
    def _generar_clave(cls, usuario_id: int, session_key: str) -> str:
        """Genera la clave única y aislada por usuario y dispositivo/sesión."""
        return f"haku:ctx:{usuario_id}:{session_key}"

    @classmethod
    def obtener(cls, usuario_id: int, session_key: str) -> Optional[EstadoOperativoHaku]:
        """Recupera el contexto activo de la RAM.

        Si transcurrieron más de 7200 segundos (2 horas) sin actividad, limpia el foco y retorna None.
        """
        if not usuario_id or not session_key:
            return None

        cache = cls._obtener_cache()
        clave = cls._generar_clave(usuario_id, session_key)
        data = cache.get(clave)

        if data is None:
            with cls._lock:
                cls._context_misses += 1
            return None

        # Reconstruir estado
        try:
            estado = EstadoOperativoHaku.from_dict(data)
        except Exception as exc:
            logger.warning("Error deserializando estado de contexto (%s); descartando", exc)
            cache.delete(clave)
            with cls._lock:
                cls._context_misses += 1
            return None

        # Verificar versión de estructura
        if estado.version_contexto != VERSION_CONTEXTO_ACTUAL:
            logger.info("Versión de contexto obsoleta (%s vs %s); invalidando", estado.version_contexto, VERSION_CONTEXTO_ACTUAL)
            cache.delete(clave)
            with cls._lock:
                cls._context_misses += 1
            return None

        # Control de expiración por inactividad (7200 segundos = 2 horas por defecto)
        if estado.esta_expirado(TIMEOUT_INACTIVIDAD_SEGUNDOS):
            logger.info("Contexto conversacional expirado (> %s s) para usuario %s", TIMEOUT_INACTIVIDAD_SEGUNDOS, usuario_id)
            estado.limpiar_foco_conversacional()
            cls.limpiar(usuario_id, session_key)
            with cls._lock:
                cls._context_misses += 1
            return None

        with cls._lock:
            cls._context_hits += 1

        # Si el foco implícito de producto/lista lleva >15 min inactivo (sin propuesta ni operación abierta),
        # despejarlo para evitar responder sobre un producto viejo cuando el usuario cambia de tema.
        if estado.foco_implicito_expirado(TIMEOUT_FOCO_IMPLICITO_SEGUNDOS):
            logger.debug("Foco implícito expirado (> %s s) para usuario %s; despejando foco previo.", TIMEOUT_FOCO_IMPLICITO_SEGUNDOS, usuario_id)
            estado.limpiar_foco_implicito()

        estado.actualizar_actividad()
        cls.guardar(estado)
        return estado

    @classmethod
    def guardar(cls, estado: EstadoOperativoHaku) -> None:
        """Persiste el estado operativo en LocMemCache con TTL de 7200 segundos (2 horas)."""
        if not estado or not estado.usuario_id or not estado.session_key:
            return

        cache = cls._obtener_cache()
        clave = cls._generar_clave(estado.usuario_id, estado.session_key)
        cache.set(clave, estado.to_dict(), timeout=TIMEOUT_INACTIVIDAD_SEGUNDOS)

    @classmethod
    def obtener_o_crear(
        cls,
        usuario_id: int,
        session_key: str,
        almacen_id: Optional[int] = None,
    ) -> EstadoOperativoHaku:
        """Obtiene el contexto existente o crea uno nuevo en RAM."""
        estado = cls.obtener(usuario_id, session_key)
        if estado is None:
            estado = EstadoOperativoHaku(
                usuario_id=usuario_id,
                session_key=session_key,
                almacen_id=almacen_id,
            )
            cls.guardar(estado)
        elif almacen_id and estado.almacen_id != almacen_id:
            estado.almacen_id = almacen_id
            cls.guardar(estado)
        return estado

    @classmethod
    def limpiar(cls, usuario_id: int, session_key: str) -> None:
        """Elimina el contexto operativo de la memoria."""
        cache = cls._obtener_cache()
        clave = cls._generar_clave(usuario_id, session_key)
        cache.delete(clave)

    @classmethod
    def limpiar_todo(cls) -> None:
        """Limpia todo el caché de contextos (útil para pruebas)."""
        cache = cls._obtener_cache()
        cache.clear()

    @classmethod
    def obtener_metricas(cls) -> Dict[str, int]:
        """Retorna las métricas internas de aciertos y fallos de contexto."""
        with cls._lock:
            return {
                "context_hits": cls._context_hits,
                "context_misses": cls._context_misses,
            }

    @classmethod
    def reset_metricas(cls) -> None:
        """Reinicia los contadores de métricas."""
        with cls._lock:
            cls._context_hits = 0
            cls._context_misses = 0


def conectar_senales_contexto():
    """Conecta el listener para limpiar el contexto en RAM cuando el usuario cierra sesión."""
    from django.contrib.auth.signals import user_logged_out

    def _on_user_logged_out(sender, request, user, **kwargs):
        if user and getattr(user, "id", None):
            session_key = getattr(request, "session", None) and getattr(request.session, "session_key", None)
            if session_key:
                HakuOperationalContext.limpiar(user.id, session_key)

    user_logged_out.connect(_on_user_logged_out, dispatch_uid="haku_logout_cleanup")

