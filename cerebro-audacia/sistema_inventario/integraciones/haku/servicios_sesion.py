"""Módulo de persistencia OCC (Optimistic Concurrency Control) para propuestas HAKU sobre django_session.

Implementación autorizada para la Fase F, Sublote 2.2:
- Control de concurrencia optimista sobre session_data sin modificar esquema de BD.
- Una única propuesta activa por sesión con aislamiento de usuario y sesión.
- Detección de colisiones y neutralización de sobreescrituras tardías del SessionMiddleware.
- Prevención de resurrección de propuestas canceladas, ejecutadas o sustituidas.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import math
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from django.contrib.sessions.backends.db import SessionStore
from django.contrib.sessions.models import Session
from django.utils import timezone

logger = logging.getLogger("integraciones.haku.sesion")


class ErrorConflictoOCC(Exception):
    """Lanzada cuando un intento de actualizar la sesión falla por modificación concurrente."""
    pass


class EstadoPropuestaHaku:
    """Constantes de estado formal para el ciclo de vida de propuestas HAKU (Fase F, Sublote 2.3)."""
    PENDIENTE = "PENDIENTE"
    EJECUTADA = "EJECUTADA"
    CANCELADA = "CANCELADA"
    SUSTITUIDA = "SUSTITUIDA"
    CADUCADA = "CADUCADA"
    RECHAZADA = "RECHAZADA"
    INEXISTENTE = "INEXISTENTE"


class ErrorSesionInvalida(Exception):
    """Lanzada cuando la sesión no existe, expiró o fue eliminada de la base de datos."""
    pass


class ErrorTokenSesionInvalido(Exception):
    """Lanzada cuando la propuesta no coincide con el token criptográfico de la sesión."""
    pass


class ErrorRegistroEvidenciaTerminal(Exception):
    """Lanzada cuando no se puede construir, validar o registrar la evidencia terminal obligatoria (F-01)."""
    pass


DOMINIO_SELLO_TERMINAL = "haku-evidencia-terminal-v2"
ESTADOS_TERMINALES_PERMITIDOS = frozenset({"EJECUTADA", "CANCELADA", "SUSTITUIDA", "CADUCADA", "RECHAZADA"})
VERSION_EVIDENCIA_TERMINAL = "ev2"

# C-04: Claves protegidas por control de concurrencia optimista (OCC)
# Estas claves representan el estado transaccional estricto de propuestas y ciclo de vida;
# jamás deben ser sobreescritas por una petición que tenga un snapshot desactualizado.
CLAVES_PROTEGIDAS_OCC = frozenset({
    "haku_propuesta_activa",
    "haku_propuesta_sustituida_id",
    "haku_ultimo_evento",
    "haku_ultima_ejecucion",
    "haku_historial_propuestas_terminales",
    "haku_bloqueo_modificacion",
    "haku_session_token",
})

# C-04: Claves de metadatos conversacionales legítimos
# Estas claves pueden y deben ser fusionadas/actualizadas durante consultas informativas
# o diálogo general sin destruir el estado de propuestas activas ni terminales.
CLAVES_METADATOS_CONVERSACIONALES = frozenset({
    "haku_ultimo_producto_id",
    "haku_ultimo_almacen_id",
    "haku_ultimo_acceso",
    "haku_conversacion_activa_id",
    "haku_conversaciones",
    "haku_conversacion",
    "haku_contexto_expirado",
})


def neutralizar_sesion_en_memoria(request: Any) -> None:
    """Neutraliza la sesión en memoria para evitar que SessionMiddleware persista un estado corrupto (F-01.3).

    Compatible con Django 4.2.30: desmarca modified y elimina _session_cache
    si existe para que no se use un objeto a medio mutar.
    """
    if request is not None and hasattr(request, "session"):
        request.session.modified = False
        if hasattr(request.session, "_session_cache"):
            del request.session._session_cache


def _es_timestamp_real_valido(val: Any) -> bool:
    """Verifica estrictamente que val sea un float/int finito y positivo (no bool)."""
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(val)
        and val > 0.0
    )


def validar_intervalo_temporal_entidad(
    creado_en: Any,
    expira_en: Any,
    ahora: Optional[float] = None,
) -> bool:
    """Valida la consistencia matemática y vigencia del intervalo [creado_en, expira_en]."""
    if not _es_timestamp_real_valido(creado_en) or not _es_timestamp_real_valido(expira_en):
        return False
    t_creado = float(creado_en)
    t_expira = float(expira_en)
    if t_creado >= t_expira:
        return False
    t_ahora = ahora if ahora is not None and _es_timestamp_real_valido(ahora) else time.time()
    # creado_en no debe estar en el futuro distante (tolerancia de 60s por desfase de reloj)
    if t_creado > t_ahora + 60.0:
        return False
    return True


def construir_bloqueo_modificacion_propuesta(
    propuesta_id_bloqueada: str,
    conversacion_id: str,
    usuario_id: Any,
    session_token_hash: str,
    firma_hmac: str,
    creado_en: float,
    expira_en: float,
    motivo_bloqueo: str = "esperando_aclaracion_modificacion",
) -> Dict[str, Any]:
    """Construye la estructura canónica del cerrojo de modificación para la sesión."""
    return {
        "propuesta_id_bloqueada": str(propuesta_id_bloqueada).strip(),
        "conversacion_id": str(conversacion_id).strip(),
        "usuario_id": str(usuario_id).strip() if usuario_id is not None else "",
        "session_token_hash": str(session_token_hash).strip(),
        "firma_hmac": str(firma_hmac).strip(),
        "creado_en": creado_en,
        "expira_en": expira_en,
        "motivo_bloqueo": str(motivo_bloqueo).strip(),
    }


def canonicalizar_metadatos_envoltorio(meta: dict[str, Any]) -> bytes:
    """Canonicaliza los metadatos terminales del envoltorio para el cálculo del sello terminal (F-02).
    
    Proyecta exclusivamente las 10 claves terminales con ordenamiento lexicográfico,
    separadores compactos, UTF-8 y prefijo de separación de dominio.
    """
    sust_id = meta.get("sustituida_por_id")
    proyeccion = {
        "cerrado_en": round(float(meta["cerrado_en"]), 4),
        "conversacion_id": str(meta["conversacion_id"]).strip().lower(),
        "estado_terminal": str(meta["estado_terminal"]).strip().upper(),
        "firma_hmac": str(meta["firma_hmac"]).strip(),
        "propuesta_id": str(meta["propuesta_id"]).strip().lower(),
        "session_token_hash": str(meta["session_token_hash"]).strip().lower(),
        "snapshot_sha256": str(meta["snapshot_sha256"]).strip().lower(),
        "sustituida_por_id": str(sust_id).strip().lower() if sust_id is not None else None,
        "usuario_id": str(meta["usuario_id"]).strip(),
        "version_evidencia": str(meta.get("version_evidencia", VERSION_EVIDENCIA_TERMINAL)).strip(),
    }
    json_bytes = json.dumps(
        proyeccion,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"{DOMINIO_SELLO_TERMINAL}:".encode("utf-8") + json_bytes


def calcular_sello_terminal(meta: dict[str, Any]) -> str:
    """Calcula el sello HMAC-SHA256 con separación de dominio para metadatos terminales."""
    from .services import _obtener_secreto_propuestas
    secreto = _obtener_secreto_propuestas()
    canon_bytes = canonicalizar_metadatos_envoltorio(meta)
    sello_hex = hmac.new(secreto, canon_bytes, hashlib.sha256).hexdigest().lower()
    return f"ev2:{sello_hex}"


def _validar_fase1_estructura_y_tipos(ev: Any) -> bool:
    """Fase 1: Valida tipos, formatos y presencia estricta de las 12 claves del envoltorio."""
    if type(ev) is not dict:
        return False

    claves_requeridas = {
        "propuesta_id",
        "conversacion_id",
        "usuario_id",
        "session_token_hash",
        "firma_hmac",
        "estado_terminal",
        "sustituida_por_id",
        "cerrado_en",
        "snapshot_sha256",
        "version_evidencia",
        "sello_terminal",
        "propuesta_snapshot",
    }
    if not claves_requeridas.issubset(ev.keys()):
        return False

    if ev.get("version_evidencia") != VERSION_EVIDENCIA_TERMINAL:
        return False

    from .services import (
        validar_uuid4_canonico_estricto,
        validar_usuario_id_envoltorio,
    )

    p_id = ev.get("propuesta_id")
    if not validar_uuid4_canonico_estricto(p_id):
        return False

    cid = ev.get("conversacion_id")
    if not validar_uuid4_canonico_estricto(cid):
        return False

    uid = ev.get("usuario_id")
    if not validar_usuario_id_envoltorio(uid):
        return False

    thash = ev.get("session_token_hash")
    if type(thash) is not str or len(thash) != 64 or not all(c in "0123456789abcdef" for c in thash):
        return False

    firma = ev.get("firma_hmac")
    if (
        type(firma) is not str
        or not firma.startswith("v2:")
        or len(firma) != 67
        or not all(c in "0123456789abcdef" for c in firma[3:])
    ):
        return False

    estado = ev.get("estado_terminal")
    if type(estado) is not str or estado not in ESTADOS_TERMINALES_PERMITIDOS:
        return False

    sustituida_id = ev.get("sustituida_por_id")
    if estado == "SUSTITUIDA":
        if not validar_uuid4_canonico_estricto(sustituida_id) or sustituida_id == p_id:
            return False
    else:
        if sustituida_id is not None:
            return False

    cerrado_en = ev.get("cerrado_en")
    if not _es_timestamp_real_valido(cerrado_en):
        return False

    snap_sha = ev.get("snapshot_sha256")
    if type(snap_sha) is not str or len(snap_sha) != 64 or not all(c in "0123456789abcdef" for c in snap_sha):
        return False

    sello = ev.get("sello_terminal")
    if (
        type(sello) is not str
        or not sello.startswith("ev2:")
        or len(sello) != 68
        or not all(c in "0123456789abcdef" for c in sello[4:])
    ):
        return False

    snap = ev.get("propuesta_snapshot")
    if type(snap) is not dict:
        return False

    return True


def _validar_fase2_coherencia(ev: dict[str, Any], usuario_id: Any = None) -> bool:
    """Fase 2: Valida consistencia cruzada entre envoltorio, snapshot y usuario filtrado."""
    snap = ev.get("propuesta_snapshot")
    if not isinstance(snap, dict):
        return False

    if str(snap.get("id")) != ev.get("propuesta_id"):
        return False
    if str(snap.get("conversacion_id")) != ev.get("conversacion_id"):
        return False
    if str(snap.get("usuario_id")) != ev.get("usuario_id"):
        return False
    if str(snap.get("session_token_hash")) != ev.get("session_token_hash"):
        return False

    firma_snap = snap.get("firma_hmac") or snap.get("huella_payload")
    if str(firma_snap) != ev.get("firma_hmac"):
        return False

    if usuario_id is not None:
        u_str = str(usuario_id)
        if ev.get("usuario_id") != u_str:
            return False

    return True


def _validar_fase3_sello_terminal(ev: dict[str, Any]) -> bool:
    """Fase 3: Recalcula y verifica el sello HMAC del envoltorio con separación de dominio."""
    sello_registrado = ev.get("sello_terminal")
    if not isinstance(sello_registrado, str):
        return False
    try:
        sello_esperado = calcular_sello_terminal(ev)
        return hmac.compare_digest(sello_registrado.lower(), sello_esperado.lower())
    except Exception:
        return False


def _validar_fase4_hmac_snapshot(ev: dict[str, Any]) -> bool:
    """Fase 4: Verifica la integridad SHA-256 del snapshot y la firma HMAC v2 de la propuesta original."""
    snap = ev.get("propuesta_snapshot")
    if not isinstance(snap, dict):
        return False

    from .services import calcular_snapshot_sha256, verificar_firma_propuesta

    try:
        sha_calculado = calcular_snapshot_sha256(snap)
        if not hmac.compare_digest(str(ev.get("snapshot_sha256")).lower(), sha_calculado.lower()):
            return False

        return verificar_firma_propuesta(snap)
    except Exception:
        return False


def registrar_evidencia_terminal_sesion(
    datos_sesion: Any,
    propuesta: Dict[str, Any],
    estado_terminal: str,
    sustituida_por_id: Optional[str] = None,
) -> None:
    """Registra evidencia verificable de una propuesta que entra a estado terminal en sesión.

    Construye la evidencia localmente, calcula snapshot_sha256 y sello_terminal,
    valida localmente las 4 fases de integridad y sólo entonces persiste en el historial.
    Lanza ErrorRegistroEvidenciaTerminal ante cualquier anomalía (F-01 / F-02).
    """
    if not hasattr(datos_sesion, "get") or not hasattr(datos_sesion, "__setitem__"):
        raise ErrorRegistroEvidenciaTerminal("datos_sesion no es accesible o no admite asignación.")

    if type(propuesta) is not dict:
        raise ErrorRegistroEvidenciaTerminal("La propuesta para evidencia terminal debe ser un diccionario.")

    from .services import (
        MAX_BIGINT_VAL,
        calcular_hash_token_sesion,
        calcular_snapshot_sha256,
        validar_usuario_id_envoltorio,
        validar_uuid4_canonico_estricto,
    )

    estado = str(estado_terminal).strip().upper() if isinstance(estado_terminal, str) else ""
    if estado not in ESTADOS_TERMINALES_PERMITIDOS:
        raise ErrorRegistroEvidenciaTerminal(f"Estado terminal no permitido: '{estado_terminal}'.")

    p_id = propuesta.get("id")
    if not validar_uuid4_canonico_estricto(p_id):
        raise ErrorRegistroEvidenciaTerminal(f"propuesta_id inválido o no canónico en evidencia: {p_id}")

    cid = propuesta.get("conversacion_id")
    if not validar_uuid4_canonico_estricto(cid):
        raise ErrorRegistroEvidenciaTerminal(f"conversacion_id inválido o no canónico en evidencia: {cid}")

    uid_raw = propuesta.get("usuario_id")
    if type(uid_raw) is int and type(uid_raw) is not bool:
        if not (1 <= uid_raw <= MAX_BIGINT_VAL):
            raise ErrorRegistroEvidenciaTerminal(f"usuario_id entero fuera de rango: {uid_raw}")
        uid = str(uid_raw)
    elif type(uid_raw) is str:
        if not validar_usuario_id_envoltorio(uid_raw):
            raise ErrorRegistroEvidenciaTerminal(f"usuario_id str no canónico o inválido: '{uid_raw}'")
        uid = uid_raw
    else:
        raise ErrorRegistroEvidenciaTerminal(f"Tipo de usuario_id inválido: {type(uid_raw).__name__}")

    token_hash = propuesta.get("session_token_hash")
    if not token_hash and propuesta.get("session_token"):
        token_hash = calcular_hash_token_sesion(propuesta["session_token"])

    if type(token_hash) is not str or len(token_hash) != 64 or not all(c in "0123456789abcdefABCDEF" for c in token_hash):
        raise ErrorRegistroEvidenciaTerminal(f"session_token_hash ausente o inválido: {token_hash}")
    token_hash = token_hash.lower()

    firma = propuesta.get("firma_hmac") or propuesta.get("huella_payload")
    if (
        type(firma) is not str
        or not firma.startswith("v2:")
        or len(firma) != 67
        or not all(c in "0123456789abcdefABCDEF" for c in firma[3:])
    ):
        raise ErrorRegistroEvidenciaTerminal(f"firma_hmac ausente o con formato no reconocido: {firma}")
    firma = f"v2:{firma[3:].lower()}"

    sust_id_canon: Optional[str] = None
    if estado == "SUSTITUIDA":
        if (
            not sustituida_por_id
            or not validar_uuid4_canonico_estricto(sustituida_por_id)
            or sustituida_por_id == p_id
        ):
            raise ErrorRegistroEvidenciaTerminal(
                f"sustituida_por_id es obligatorio, debe ser UUID4 válido y diferente de propuesta_id para estado SUSTITUIDA: {sustituida_por_id}"
            )
        sust_id_canon = sustituida_por_id
    else:
        sust_id_canon = None

    propuesta_snapshot = copy.deepcopy(propuesta)
    try:
        snapshot_sha256 = calcular_snapshot_sha256(propuesta_snapshot)
    except Exception as exc:
        raise ErrorRegistroEvidenciaTerminal(f"Error al calcular SHA-256 canónico del snapshot: {exc}") from exc

    cerrado_en = round(time.time(), 4)

    evidencia_candidata = {
        "propuesta_id": p_id,
        "conversacion_id": cid,
        "usuario_id": uid,
        "session_token_hash": token_hash,
        "firma_hmac": firma,
        "estado_terminal": estado,
        "sustituida_por_id": sust_id_canon,
        "cerrado_en": cerrado_en,
        "snapshot_sha256": snapshot_sha256,
        "version_evidencia": VERSION_EVIDENCIA_TERMINAL,
        "propuesta_snapshot": propuesta_snapshot,
    }

    try:
        sello = calcular_sello_terminal(evidencia_candidata)
        evidencia_candidata["sello_terminal"] = sello
    except Exception as exc:
        raise ErrorRegistroEvidenciaTerminal(f"Error al calcular sello terminal: {exc}") from exc

    # Validación local de las 4 fases antes de alterar el historial de sesión
    if not _validar_fase1_estructura_y_tipos(evidencia_candidata):
        raise ErrorRegistroEvidenciaTerminal("La evidencia terminal local no superó la Fase 1 (estructura y tipos).")
    if not _validar_fase2_coherencia(evidencia_candidata, uid):
        raise ErrorRegistroEvidenciaTerminal("La evidencia terminal local no superó la Fase 2 (coherencia de snapshot).")
    if not _validar_fase3_sello_terminal(evidencia_candidata):
        raise ErrorRegistroEvidenciaTerminal("La evidencia terminal local no superó la Fase 3 (sello terminal).")
    if not _validar_fase4_hmac_snapshot(evidencia_candidata):
        raise ErrorRegistroEvidenciaTerminal("La evidencia terminal local no superó la Fase 4 (HMAC snapshot).")

    # Modificar el historial de sesión de manera segura
    historial = datos_sesion.get("haku_historial_propuestas_terminales")
    if type(historial) is not list:
        historial = []
        datos_sesion["haku_historial_propuestas_terminales"] = historial

    idx_existente = -1
    for i, item in enumerate(historial):
        if isinstance(item, dict) and item.get("propuesta_id") == p_id:
            idx_existente = i
            break

    if idx_existente >= 0:
        historial[idx_existente] = evidencia_candidata
    else:
        historial.append(evidencia_candidata)

    CAP_MAXIMA = 12
    if len(historial) > CAP_MAXIMA:
        bloqueo = datos_sesion.get("haku_bloqueo_modificacion")
        prop_protegida = str(bloqueo.get("propuesta_id_bloqueada") or "").strip() if isinstance(bloqueo, dict) else ""

        candidato = 0
        while len(historial) > CAP_MAXIMA and candidato < len(historial):
            item = historial[candidato]
            if isinstance(item, dict) and prop_protegida and item.get("propuesta_id") == prop_protegida:
                candidato += 1
            else:
                historial.pop(candidato)


def obtener_evidencia_verificable_propuesta_terminal(
    datos_sesion: Optional[Any],
    propuesta_id: str,
    usuario_id: Any = None,
) -> Optional[Dict[str, Any]]:
    """Obtiene y verifica criptográficamente la evidencia terminal de una propuesta.

    Orden de verificación estricto (Fail-Closed, F-03):
    1. Validar propuesta_id (UUID4 canónico RFC 4122 en minúsculas, sin coerción).
    2. Validar usuario_id cuando esté presente (int o string ASCII en rango BigAutoField).
    3. Acceder a datos de sesión (sin lecturas previas).
    4. Validar historial exacto list.
    5. Validar cada evidencia exacta dict.
    6. Validar estructura y coherencia (Fase 1 y Fase 2).
    7. Verificar sello terminal (Fase 3).
    8. Verificar HMAC del snapshot (Fase 4).
    """
    from .services import (
        MAX_BIGINT_VAL,
        validar_usuario_id_envoltorio,
        validar_uuid4_canonico_estricto,
    )

    # Paso 1: Validar propuesta_id sin tocar datos_sesion
    if not validar_uuid4_canonico_estricto(propuesta_id):
        return None

    # Paso 2: Validar usuario_id si se proporciona
    usuario_id_canonico = None
    if usuario_id is not None:
        if type(usuario_id) is int and type(usuario_id) is not bool:
            if not (1 <= usuario_id <= MAX_BIGINT_VAL):
                return None
            usuario_id_canonico = str(usuario_id)
        elif type(usuario_id) is str:
            if not validar_usuario_id_envoltorio(usuario_id):
                return None
            usuario_id_canonico = usuario_id
        else:
            return None

    # Paso 3: Acceder a datos de sesión
    if not hasattr(datos_sesion, "get"):
        return None

    # Paso 4: Validar historial exacto list
    historial = datos_sesion.get("haku_historial_propuestas_terminales")
    if type(historial) is not list:
        return None

    # Paso 5 a 8: Iterar evidencias y aplicar las 4 fases
    for ev in reversed(historial):
        if type(ev) is not dict:
            continue
        if ev.get("propuesta_id") == propuesta_id:
            # Fase 1: Estructura y tipos
            if not _validar_fase1_estructura_y_tipos(ev):
                return None
            # Fase 2: Coherencia de snapshot y usuario
            if not _validar_fase2_coherencia(ev, usuario_id_canonico):
                return None
            # Fase 3: Sello terminal independiente
            if not _validar_fase3_sello_terminal(ev):
                return None
            # Fase 4: HMAC y SHA-256 del snapshot
            if not _validar_fase4_hmac_snapshot(ev):
                return None
            return ev

    return None


def evaluar_bloqueo_modificacion_activo(
    datos_sesion: Any,
    usuario: Any = None,
    session_token: Optional[str] = None,
    ahora: Optional[float] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Evalúa si existe un cerrojo de modificación activo y válido bajo política Fail-Closed.

    Retorna (esta_activo: bool, razon: str, bloqueo: Optional[dict]).
    """
    if not hasattr(datos_sesion, "get"):
        return False, "sin_datos_sesion", None

    bloqueo = datos_sesion.get("haku_bloqueo_modificacion")
    if not bloqueo:
        return False, "sin_bloqueo", None

    if not isinstance(bloqueo, dict):
        return False, "estructura_bloqueo_invalida", None

    prop_id = bloqueo.get("propuesta_id_bloqueada")
    cid = bloqueo.get("conversacion_id")
    uid = bloqueo.get("usuario_id")
    thash = bloqueo.get("session_token_hash")
    firma = bloqueo.get("firma_hmac")
    creado_en = bloqueo.get("creado_en")
    expira_en = bloqueo.get("expira_en")

    if not (prop_id and cid and uid and thash and firma and creado_en and expira_en):
        return False, "campos_bloqueo_incompletos", None

    user_id_str = str(getattr(usuario, "id", None) or (datos_sesion.get("_auth_user_id") if hasattr(datos_sesion, "get") else "") or "")
    if user_id_str and str(uid) != user_id_str:
        return False, "usuario_discordante", None

    tok = session_token or (datos_sesion.get("haku_session_token") if hasattr(datos_sesion, "get") else None)
    if tok and isinstance(tok, str):
        from .services import calcular_hash_token_sesion
        expected_thash = calcular_hash_token_sesion(tok)
        if not hmac.compare_digest(str(thash).lower(), expected_thash.lower()):
            return False, "token_sesion_discordante", None

    try:
        uuid.UUID(str(cid).strip(), version=4)
    except (ValueError, TypeError):
        return False, "conversacion_id_invalida", None

    t_ahora = ahora if ahora is not None and _es_timestamp_real_valido(ahora) else time.time()
    if not validar_intervalo_temporal_entidad(creado_en, expira_en, t_ahora):
        return False, "intervalo_temporal_invalido", None

    if t_ahora >= float(expira_en):
        return False, "bloqueo_expirado", None

    evidencia = obtener_evidencia_verificable_propuesta_terminal(datos_sesion, prop_id, user_id_str)
    if not evidencia:
        return False, "evidencia_terminal_p0_no_verificable", None

    if (
        evidencia.get("conversacion_id") != str(cid).strip()
        or evidencia.get("session_token_hash") != str(thash).strip()
        or (user_id_str and evidencia.get("usuario_id") != user_id_str)
    ):
        return False, "discordancia_evidencia_p0", None

    return True, "bloqueo_activo", bloqueo


def resolver_o_crear_conversacion_turno(
    datos_sesion: Any,
    usuario: Any,
    conversacion_id_cliente: Optional[str] = None,
    ahora: Optional[float] = None,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """Resuelve o crea el identificador de conversación para la pestaña actual (aislamiento por pestaña).

    Capacidad máxima acotada: 20 conversaciones por sesión.
    Política de desalojo protegida: nunca desaloja la conversación de la propuesta activa ni del cerrojo.
    Retorna (conversacion_id: str, fue_creada: bool, error_o_motivo: Optional[str]).
    """
    if not hasattr(datos_sesion, "get") or not hasattr(datos_sesion, "__setitem__"):
        return None, False, "sin_datos_sesion"

    conversaciones = datos_sesion.get("haku_conversaciones")
    if not isinstance(conversaciones, dict):
        conversaciones = {}
        datos_sesion["haku_conversaciones"] = conversaciones

    user_id_str = str(getattr(usuario, "id", None) or "")
    t_ahora = ahora if ahora is not None and _es_timestamp_real_valido(ahora) else time.time()

    if conversacion_id_cliente:
        cid_limpio = str(conversacion_id_cliente).strip()
        try:
            uuid.UUID(cid_limpio, version=4)
        except (ValueError, TypeError):
            return None, False, "conversacion_id_formato_invalido"

        if cid_limpio in conversaciones:
            entrada = conversaciones[cid_limpio]
            if isinstance(entrada, dict) and entrada.get("usuario_id") and str(entrada.get("usuario_id")) != user_id_str:
                return None, False, "conversacion_usuario_discordante"
            entrada["ultimo_acceso"] = round(t_ahora, 4)
            datos_sesion["haku_conversacion_activa_id"] = cid_limpio
            return cid_limpio, False, None

        CAP_MAX = 20
        if len(conversaciones) >= CAP_MAX:
            prop_activa = datos_sesion.get("haku_propuesta_activa")
            cid_prop = str(prop_activa.get("conversacion_id") or "").strip() if isinstance(prop_activa, dict) else ""
            bloqueo = datos_sesion.get("haku_bloqueo_modificacion")
            cid_bloq = str(bloqueo.get("conversacion_id") or "").strip() if isinstance(bloqueo, dict) else ""

            claves_ordenadas = sorted(
                conversaciones.keys(),
                key=lambda k: conversaciones[k].get("ultimo_acceso", 0) if isinstance(conversaciones[k], dict) else 0,
            )
            candidato_desalojado = None
            for k in claves_ordenadas:
                if k != cid_prop and k != cid_bloq:
                    candidato_desalojado = k
                    break

            if not candidato_desalojado:
                return None, False, "limite_conversaciones_alcanzado"

            conversaciones.pop(candidato_desalojado, None)

        conversaciones[cid_limpio] = {
            "usuario_id": user_id_str,
            "creado_en": round(t_ahora, 4),
            "ultimo_acceso": round(t_ahora, 4),
        }
        datos_sesion["haku_conversacion_activa_id"] = cid_limpio
        return cid_limpio, True, None

    # Si el cliente no especificó conversacion_id, reutilizar la conversación activa de la sesión si existe
    cid_existente = datos_sesion.get("haku_conversacion_activa_id")
    if cid_existente and isinstance(cid_existente, str) and cid_existente in conversaciones:
        entrada = conversaciones[cid_existente]
        if isinstance(entrada, dict) and (not entrada.get("usuario_id") or str(entrada.get("usuario_id")) == user_id_str):
            entrada["ultimo_acceso"] = round(t_ahora, 4)
            return cid_existente, False, None

    prop_activa = datos_sesion.get("haku_propuesta_activa")
    if isinstance(prop_activa, dict) and prop_activa.get("conversacion_id"):
        cid_prop = str(prop_activa.get("conversacion_id")).strip()
        if cid_prop in conversaciones:
            entrada = conversaciones[cid_prop]
            if isinstance(entrada, dict) and (not entrada.get("usuario_id") or str(entrada.get("usuario_id")) == user_id_str):
                entrada["ultimo_acceso"] = round(t_ahora, 4)
                datos_sesion["haku_conversacion_activa_id"] = cid_prop
                return cid_prop, False, None

    nuevo_cid = str(uuid.uuid4())
    CAP_MAX = 20
    if len(conversaciones) >= CAP_MAX:
        prop_activa = datos_sesion.get("haku_propuesta_activa")
        cid_prop = str(prop_activa.get("conversacion_id") or "").strip() if isinstance(prop_activa, dict) else ""
        bloqueo = datos_sesion.get("haku_bloqueo_modificacion")
        cid_bloq = str(bloqueo.get("conversacion_id") or "").strip() if isinstance(bloqueo, dict) else ""

        claves_ordenadas = sorted(
            conversaciones.keys(),
            key=lambda k: conversaciones[k].get("ultimo_acceso", 0) if isinstance(conversaciones[k], dict) else 0,
        )
        candidato_desalojado = None
        for k in claves_ordenadas:
            if k != cid_prop and k != cid_bloq:
                candidato_desalojado = k
                break

        if not candidato_desalojado:
            return None, False, "limite_conversaciones_alcanzado"

        conversaciones.pop(candidato_desalojado, None)

    conversaciones[nuevo_cid] = {
        "usuario_id": user_id_str,
        "creado_en": round(t_ahora, 4),
        "ultimo_acceso": round(t_ahora, 4),
    }
    datos_sesion["haku_conversacion_activa_id"] = nuevo_cid
    return nuevo_cid, True, None


def es_propuesta_contrato_anterior(propuesta: Optional[Dict[str, Any]]) -> bool:
    """Determina si la propuesta fue emitida con un contrato previo (v1) o carece de campos v2."""
    if not propuesta or not isinstance(propuesta, dict):
        return False
    v_contrato = str(propuesta.get("version_contrato") or "").strip().lower()
    if v_contrato != "v2":
        return True
    firma = str(propuesta.get("firma_hmac") or propuesta.get("huella_payload") or "").strip()
    if not firma.startswith("v2:"):
        return True
    cid = propuesta.get("conversacion_id")
    if not cid:
        return True
    try:
        uuid.UUID(str(cid).strip(), version=4)
    except (ValueError, TypeError):
        return True
    thash = propuesta.get("session_token_hash")
    if not thash or len(str(thash).strip()) != 64:
        return True
    return False


def evaluar_vigencia_propuesta(expira_en_raw: Any) -> Tuple[bool, bool]:
    """Evalúa la vigencia temporal de una propuesta bajo política Fail-Closed.

    Regla matemática obligatoria:
    - now < expira_en: vigente (es_valida=True, esta_caducada=False)
    - now >= expira_en: caducada legítima (es_valida=True, esta_caducada=True)

    Cualquier valor nulo, ausente, booleano, no numérico, NaN, infinito o <= 0
    representa un error de integridad temporal (es_valida=False, esta_caducada=False).
    Captura explícitamente TypeError, ValueError, OverflowError sin lanzar excepciones.
    """
    if expira_en_raw is None or expira_en_raw == "":
        return False, False

    if isinstance(expira_en_raw, bool):
        # En Python float(True) == 1.0; se rechaza explícitamente
        return False, False

    try:
        val_exp = float(expira_en_raw)
        if math.isnan(val_exp) or math.isinf(val_exp) or val_exp <= 0:
            return False, False
        esta_caducada = time.time() >= val_exp
        return True, esta_caducada
    except (TypeError, ValueError, OverflowError):
        return False, False


def sanitizar_operacion_declarada(op: Any) -> str:
    """Sanitiza el código de operación declarado por el cliente a un máximo de 30 caracteres alfanuméricos."""
    if not op:
        return "DESCONOCIDA"
    return re.sub(r'[^A-Z0-9_]', '', str(op).upper())[:30]


def sanitizar_identificador_diagnostico(propuesta_id_raw: Any) -> str:
    """Genera un identificador de diagnóstico seguro para solicitudes no verificadas."""
    if propuesta_id_raw:
        try:
            val_uuid = uuid.UUID(str(propuesta_id_raw).strip())
            return f"no-verificada:{val_uuid}"[:100]
        except (ValueError, TypeError, AttributeError):
            pass
    return f"intento-invalido:{uuid.uuid4()}"[:100]


def registrar_auditoria_rechazo_seguridad(
    usuario: Optional[Any],
    propuesta_id: Optional[str],
    codigo_resultado: str,
    motivo: str = "",
    operacion_declarada: Optional[str] = None,
    producto_id_declarado: Optional[Any] = None,
    direccion_peer: Optional[str] = None,
    propuesta_verificada: bool = False,
) -> Optional[Any]:
    """Persiste un evento PROPUESTA_RECHAZADA en una transacción atómica independiente y limpia."""
    from django.db import transaction
    from inventario.models import Auditoria

    try:
        with transaction.atomic(using="default"):
            user_inst = usuario if getattr(usuario, "is_authenticated", False) else None

            if propuesta_verificada and propuesta_id:
                safe_id = str(propuesta_id).strip()[:100]
            else:
                safe_id = sanitizar_identificador_diagnostico(propuesta_id)

            val_ant: Dict[str, Any] = {"tipo_rechazo": codigo_resultado}
            if operacion_declarada:
                val_ant["operacion_declarada"] = sanitizar_operacion_declarada(operacion_declarada)
            if producto_id_declarado is not None and not isinstance(producto_id_declarado, bool):
                try:
                    val_ant["producto_id_declarado"] = int(producto_id_declarado)
                except (ValueError, TypeError, OverflowError):
                    val_ant["producto_id_declarado"] = None
            if direccion_peer:
                val_ant["direccion_peer"] = str(direccion_peer)[:45]

            motivo_limpio = f"[propuesta_id:{safe_id}] Rechazo de seguridad: {codigo_resultado}"
            if motivo:
                motivo_limpio += f" - {str(motivo)[:150]}"

            aud = Auditoria.objects.create(
                usuario=user_inst,
                accion="PROPUESTA_RECHAZADA",
                tipo_entidad="PropuestaHaku",
                id_entidad=safe_id,
                resultado=str(codigo_resultado)[:50],
                motivo=motivo_limpio,
                valores_anteriores=val_ant,
                valores_nuevos={"estado": "RECHAZADA"},
                via_haku=True,
            )
            return aud
    except Exception as exc:
        logger.error("No se pudo persistir auditoría de rechazo: %s", exc)
        return None


def registrar_auditoria_cancelacion(
    usuario: Any,
    propuesta_id: str,
    propuesta_dict: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Registra la cancelación persistente en Auditoria de forma idempotente."""
    from inventario.models import Auditoria

    safe_id = str(propuesta_id).strip()
    user_id = getattr(usuario, "id", None)

    aud_prev = Auditoria.objects.filter(
        tipo_entidad="PropuestaHaku",
        accion="PROPUESTA_CANCELADA",
        id_entidad=safe_id,
        usuario_id=user_id,
    ).first()
    if aud_prev:
        return aud_prev

    val_ant: Dict[str, Any] = {"propuesta_id": safe_id}
    val_nue: Dict[str, Any] = {"propuesta_id": safe_id, "estado": "CANCELADA"}
    if propuesta_dict and isinstance(propuesta_dict, dict):
        for k in ("tipo_operacion", "producto_id", "cantidad", "destinatario", "conversacion_id", "session_token_hash"):
            if k in propuesta_dict and propuesta_dict[k] is not None:
                val_ant[k] = propuesta_dict[k]
        if propuesta_dict.get("conversacion_id"):
            val_nue["conversacion_id"] = propuesta_dict["conversacion_id"]
        if propuesta_dict.get("session_token_hash"):
            val_nue["session_token_hash"] = propuesta_dict["session_token_hash"]
        firma = propuesta_dict.get("firma_hmac") or propuesta_dict.get("huella_payload")
        if firma:
            val_ant["firma_hmac"] = firma
            val_nue["firma_hmac"] = firma

    return Auditoria.objects.create(
        usuario=usuario if getattr(usuario, "is_authenticated", False) else None,
        accion="PROPUESTA_CANCELADA",
        tipo_entidad="PropuestaHaku",
        id_entidad=safe_id,
        resultado="CANCELADA",
        motivo=f"[propuesta_id:{safe_id}] Cancelación explícita de propuesta",
        valores_anteriores=val_ant,
        valores_nuevos=val_nue,
        via_haku=True,
    )


def registrar_auditoria_modificacion(
    usuario: Any,
    propuesta_anterior_id: str,
    propuesta_nueva: Dict[str, Any],
    campos_modificados: Optional[List[str]] = None,
) -> Optional[Any]:
    """Registra la sustitución P1 -> P2 en Auditoria preservando el contrato de linaje."""
    from inventario.models import Auditoria

    p_prev_id = str(propuesta_anterior_id).strip()
    p_nueva_id = str(propuesta_nueva.get("id") or "").strip()
    user_id = getattr(usuario, "id", None)

    aud_prev = Auditoria.objects.filter(
        tipo_entidad="PropuestaHaku",
        accion="PROPUESTA_MODIFICADA",
        id_entidad=p_nueva_id,
        usuario_id=user_id,
    ).first()
    if aud_prev:
        return aud_prev

    campos = campos_modificados or ["destinatario"]
    val_ant: Dict[str, Any] = {"propuesta_id": p_prev_id}
    val_nue: Dict[str, Any] = {
        "propuesta_id": p_nueva_id,
        "sustituye": p_prev_id,
        "campos_modificados": campos,
    }
    if propuesta_nueva and isinstance(propuesta_nueva, dict):
        if propuesta_nueva.get("conversacion_id"):
            val_nue["conversacion_id"] = propuesta_nueva["conversacion_id"]
            val_ant["conversacion_id"] = propuesta_nueva["conversacion_id"]
        if propuesta_nueva.get("session_token_hash"):
            val_nue["session_token_hash"] = propuesta_nueva["session_token_hash"]
            val_ant["session_token_hash"] = propuesta_nueva["session_token_hash"]
        firma = propuesta_nueva.get("firma_hmac") or propuesta_nueva.get("huella_payload")
        if firma:
            val_nue["firma_hmac"] = firma

    return Auditoria.objects.create(
        usuario=usuario if getattr(usuario, "is_authenticated", False) else None,
        accion="PROPUESTA_MODIFICADA",
        tipo_entidad="PropuestaHaku",
        id_entidad=p_nueva_id,
        resultado="OK",
        motivo=f"[propuesta_id:{p_nueva_id}] [sustituye:{p_prev_id}] Modificación de propuesta ({', '.join(campos)})",
        valores_anteriores=val_ant,
        valores_nuevos=val_nue,
        via_haku=True,
    )


def registrar_auditoria_caducidad(
    usuario: Any,
    propuesta_id: str,
    propuesta_dict: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Registra la caducidad legítima en Auditoria de forma idempotente."""
    from inventario.models import Auditoria

    safe_id = str(propuesta_id).strip()
    user_id = getattr(usuario, "id", None)

    aud_prev = Auditoria.objects.filter(
        tipo_entidad="PropuestaHaku",
        accion="PROPUESTA_CADUCADA",
        id_entidad=safe_id,
        usuario_id=user_id,
    ).first()
    if aud_prev:
        return aud_prev

    val_ant: Dict[str, Any] = {"propuesta_id": safe_id}
    val_nue: Dict[str, Any] = {"propuesta_id": safe_id, "estado": "EXPIRADA"}
    if propuesta_dict and isinstance(propuesta_dict, dict):
        val_ant["expira_en"] = propuesta_dict.get("expira_en")
        if propuesta_dict.get("conversacion_id"):
            val_nue["conversacion_id"] = propuesta_dict["conversacion_id"]
            val_ant["conversacion_id"] = propuesta_dict["conversacion_id"]
        if propuesta_dict.get("session_token_hash"):
            val_nue["session_token_hash"] = propuesta_dict["session_token_hash"]
            val_ant["session_token_hash"] = propuesta_dict["session_token_hash"]
        firma = propuesta_dict.get("firma_hmac") or propuesta_dict.get("huella_payload")
        if firma:
            val_nue["firma_hmac"] = firma
            val_ant["firma_hmac"] = firma

    return Auditoria.objects.create(
        usuario=usuario if getattr(usuario, "is_authenticated", False) else None,
        accion="PROPUESTA_CADUCADA",
        tipo_entidad="PropuestaHaku",
        id_entidad=safe_id,
        resultado="EXPIRADA",
        motivo=f"[propuesta_id:{safe_id}] Propuesta caducada por vencimiento de tiempo",
        valores_anteriores=val_ant,
        valores_nuevos=val_nue,
        via_haku=True,
    )


def consultar_estado_propuesta_persistente(
    propuesta_id: str,
    usuario: Optional[Any] = None,
    sesion_o_request: Optional[Any] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Reconstruye de forma canónica el estado de la propuesta discriminando los siete estados.

    Orden jerárquico de fuentes de verdad:
    1. EJECUTADA en Ledger (Movimiento o Auditoría de catálogo).
    2. Eventos persistentes en Auditoria (CANCELADA, SUSTITUIDA, CADUCADA, RECHAZADA) con detección
       estricta de duplicados, contradicciones semánticas y bifurcaciones de linaje.
    3. PENDIENTE en sesión (verificada con firma HMAC, propiedad, token y vigencia temporal).
    4. Metadatos de sesión (haku_propuesta_sustituida_id, haku_ultimo_evento) subordinados a la BD.
    5. INEXISTENTE si no hay rastros.
    """
    if not propuesta_id:
        return EstadoPropuestaHaku.INEXISTENTE, None

    from django.db.models import Q
    from inventario.models import Auditoria

    p_id_str = str(propuesta_id).strip()
    user_id = getattr(usuario, "id", None)

    # 1. Comprobar si fue confirmada y ejecutada en el ledger persistente
    res_ledger = buscar_ejecucion_existente_ledger(p_id_str, usuario)

    # 2. Consultar eventos de ciclo de vida en Auditoria para esta propuesta
    qs_aud = Auditoria.objects.filter(tipo_entidad="PropuestaHaku")
    if user_id:
        qs_aud = qs_aud.filter(usuario_id=user_id)

    eventos = list(
        qs_aud.filter(
            Q(id_entidad=p_id_str) | Q(valores_nuevos__sustituye=p_id_str)
        ).order_by("fecha_hora")
    )

    # Si hay ejecución en el ledger y coexisten eventos terminales contradictorios en Auditoria
    if res_ledger:
        if eventos:
            acciones_contradictorias = [
                e.accion for e in eventos
                if e.accion in ("PROPUESTA_CANCELADA", "PROPUESTA_CADUCADA", "PROPUESTA_MODIFICADA")
            ]
            if acciones_contradictorias:
                logger.critical(
                    "Anomalía crítica de integridad en propuesta %s: ejecución en ledger coexiste con auditorías terminales: %s",
                    p_id_str,
                    acciones_contradictorias,
                )
                return "CONFLICTO_INTEGRIDAD", {
                    "error": "conflicto_integridad_historial",
                    "mensaje": "Se detectó una inconsistencia crítica en el historial de la propuesta.",
                    "acciones": acciones_contradictorias,
                }
        return EstadoPropuestaHaku.EJECUTADA, res_ledger

    # Si no hay ejecución en ledger, analizar los eventos de Auditoria
    if eventos:
        cancelaciones = [e for e in eventos if e.accion == "PROPUESTA_CANCELADA" and e.id_entidad == p_id_str]
        caducidades = [e for e in eventos if e.accion == "PROPUESTA_CADUCADA" and e.id_entidad == p_id_str]
        rechazos = [e for e in eventos if e.accion == "PROPUESTA_RECHAZADA" and e.id_entidad == p_id_str]
        sustituciones = [
            e for e in eventos
            if e.accion == "PROPUESTA_MODIFICADA" and isinstance(e.valores_nuevos, dict) and str(e.valores_nuevos.get("sustituye") or "").strip() == p_id_str
        ]

        # Contradicción semántica entre distintas acciones terminales
        categorias_presentes = []
        if cancelaciones:
            categorias_presentes.append("PROPUESTA_CANCELADA")
        if caducidades:
            categorias_presentes.append("PROPUESTA_CADUCADA")
        if sustituciones:
            categorias_presentes.append("PROPUESTA_MODIFICADA")

        if len(categorias_presentes) > 1:
            logger.critical(
                "Inconsistencia semántica en propuesta %s: múltiples acciones terminales incompatibles: %s",
                p_id_str,
                categorias_presentes,
            )
            return "CONFLICTO_INTEGRIDAD", {
                "error": "conflicto_integridad_historial",
                "mensaje": "Se detectaron estados terminales contradictorios para la propuesta.",
                "acciones": categorias_presentes,
            }

        # Bifurcación de linaje: dos o más sucesores para la misma propuesta anterior
        if len(sustituciones) >= 2:
            sucesores = [e.id_entidad for e in sustituciones]
            logger.critical(
                "Anomalía crítica de linaje en propuesta %s: múltiples sucesores detectados: %s",
                p_id_str,
                sucesores,
            )
            return "CONFLICTO_INTEGRIDAD_LINAJE", {
                "error": "conflicto_integridad_linaje",
                "mensaje": "Se detectó una bifurcación de linaje anómala con múltiples sucesores.",
                "sucesores": sucesores,
            }

        # Duplicados anómalos de la misma acción terminal
        if len(cancelaciones) >= 2:
            logger.warning("Duplicación anómala detectada en propuesta %s: múltiples registros para PROPUESTA_CANCELADA", p_id_str)
            return EstadoPropuestaHaku.CANCELADA, {"auditoria_id": str(cancelaciones[0].id), "duplicados": len(cancelaciones)}

        if len(caducidades) >= 2:
            logger.warning("Duplicación anómala detectada en propuesta %s: múltiples registros para PROPUESTA_CADUCADA", p_id_str)
            return EstadoPropuestaHaku.CADUCADA, {"auditoria_id": str(caducidades[0].id), "duplicados": len(caducidades)}

        if len(sustituciones) == 1:
            return EstadoPropuestaHaku.SUSTITUIDA, {
                "sucesora_id": str(sustituciones[0].id_entidad),
                "auditoria_id": str(sustituciones[0].id),
            }

        if len(cancelaciones) == 1:
            return EstadoPropuestaHaku.CANCELADA, {"auditoria_id": str(cancelaciones[0].id)}

        if len(caducidades) == 1:
            return EstadoPropuestaHaku.CADUCADA, {"auditoria_id": str(caducidades[0].id)}

        if len(rechazos) >= 1:
            return EstadoPropuestaHaku.RECHAZADA, {
                "auditoria_id": str(rechazos[0].id),
                "resultado": rechazos[0].resultado,
            }

    # 3. Comprobar en sesión activa (PENDIENTE)
    store = None
    if sesion_o_request is not None:
        if hasattr(sesion_o_request, "session"):
            store = sesion_o_request.session
        elif hasattr(sesion_o_request, "datos_sesion"):
            store = sesion_o_request.datos_sesion
        else:
            store = sesion_o_request

    if store and hasattr(store, "get"):
        prop_activa = store.get("haku_propuesta_activa")
        if isinstance(prop_activa, dict) and str(prop_activa.get("id") or "").strip() == p_id_str:
            # Validar propiedad
            if user_id and str(prop_activa.get("usuario_id") or "") != str(user_id):
                return EstadoPropuestaHaku.RECHAZADA, {"motivo": "usuario_discordante"}

            # Validar token de sesión
            token_sesion = store.get("haku_session_token")
            token_prop = prop_activa.get("session_token")
            if (
                not token_sesion
                or not token_prop
                or not isinstance(token_sesion, str)
                or not isinstance(token_prop, str)
                or not hmac.compare_digest(str(token_prop).strip(), str(token_sesion).strip())
            ):
                return EstadoPropuestaHaku.RECHAZADA, {"motivo": "token_invalido"}

            # Validar firma HMAC v1
            from .services import verificar_firma_propuesta
            if not verificar_firma_propuesta(prop_activa):
                return EstadoPropuestaHaku.RECHAZADA, {"motivo": "firma_invalida"}

            # Validar vigencia temporal
            es_valida, esta_caducada = evaluar_vigencia_propuesta(prop_activa.get("expira_en"))
            if not es_valida:
                return EstadoPropuestaHaku.RECHAZADA, {"motivo": "integridad_temporal_invalida"}
            if esta_caducada:
                return EstadoPropuestaHaku.CADUCADA, prop_activa

            return EstadoPropuestaHaku.PENDIENTE, prop_activa

        # 4. Metadatos de sesión (fallback subordinado a la BD)
        sust_id = str(store.get("haku_propuesta_sustituida_id") or "").strip()
        if sust_id and sust_id == p_id_str:
            return EstadoPropuestaHaku.SUSTITUIDA, {"origen": "sesion"}

        ult_ev = store.get("haku_ultimo_evento")
        if isinstance(ult_ev, dict):
            ev_p_id = str(ult_ev.get("propuesta_id") or "").strip()
            if ev_p_id and ev_p_id == p_id_str:
                ev_user = ult_ev.get("usuario_id")
                if ev_user and user_id and str(ev_user) != str(user_id):
                    return EstadoPropuestaHaku.INEXISTENTE, None

                ev_estado = ult_ev.get("estado")
                if ev_estado in (
                    EstadoPropuestaHaku.CANCELADA,
                    EstadoPropuestaHaku.SUSTITUIDA,
                    EstadoPropuestaHaku.CADUCADA,
                    EstadoPropuestaHaku.RECHAZADA,
                    EstadoPropuestaHaku.EJECUTADA,
                ):
                    return ev_estado, ult_ev

        ult_ejec = store.get("haku_ultima_ejecucion")
        if isinstance(ult_ejec, dict):
            ejec_p_id = str(ult_ejec.get("propuesta_id") or "").strip()
            if ejec_p_id and ejec_p_id == p_id_str:
                return EstadoPropuestaHaku.EJECUTADA, ult_ejec

    return EstadoPropuestaHaku.INEXISTENTE, None


def obtener_o_crear_session_token(sesion: Any) -> str:
    """Obtiene o inicializa de forma segura el token criptográfico de sesión (128 bits de entropía)."""
    if sesion is None:
        return ""
    token = sesion.get("haku_session_token")
    if not token or not isinstance(token, str):
        token = secrets.token_hex(16)
        sesion["haku_session_token"] = token
        if hasattr(sesion, "modified"):
            sesion.modified = True
    return token


def obtener_estado_propuesta_sesion(
    sesion_o_request: Any,
    usuario: Optional[Any] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Determina de forma canónica el estado formal de la propuesta en sesión.

    Regla criptográfica innegociable:
    El estado 'PENDIENTE' es un estado DERIVADO en tiempo de ejecución.
    Bajo NINGUNA circunstancia muta el diccionario firmado de la propuesta.
    """
    if hasattr(sesion_o_request, "session"):
        store = sesion_o_request.session
    elif hasattr(sesion_o_request, "datos_sesion"):
        store = sesion_o_request.datos_sesion
    else:
        store = sesion_o_request

    if store is None or not hasattr(store, "get"):
        return EstadoPropuestaHaku.INEXISTENTE, None

    propuesta_activa = store.get("haku_propuesta_activa")
    if isinstance(propuesta_activa, dict) and propuesta_activa.get("id"):
        p_id = str(propuesta_activa["id"]).strip()
        return consultar_estado_propuesta_persistente(p_id, usuario, store)

    # Si no hay propuesta activa, consultar metadatos del último evento auxiliar
    ultimo_evento = store.get("haku_ultimo_evento")
    if isinstance(ultimo_evento, dict) and ultimo_evento.get("propuesta_id"):
        ev_id = str(ultimo_evento["propuesta_id"]).strip()
        return consultar_estado_propuesta_persistente(ev_id, usuario, store)

    ultima_ejec = store.get("haku_ultima_ejecucion")
    if isinstance(ultima_ejec, dict) and ultima_ejec.get("propuesta_id"):
        ej_id = str(ultima_ejec["propuesta_id"]).strip()
        return consultar_estado_propuesta_persistente(ej_id, usuario, store)

    if isinstance(ultimo_evento, dict) and ultimo_evento.get("estado"):
        return str(ultimo_evento["estado"]), None

    return EstadoPropuestaHaku.INEXISTENTE, None


@dataclass
class EstadoSesionOCC:
    session_key: str
    raw_original: str
    datos_sesion: Dict[str, Any]
    expire_date: Any


def capturar_estado_sesion_occ(session_key: str, request: Optional[Any] = None) -> Optional[EstadoSesionOCC]:
    """Lee la fila vigente de django_session y decodifica su contenido, capturando el raw_original exacto."""
    if not session_key:
        return None

    ahora = timezone.now()
    session_obj = Session.objects.filter(
        session_key=session_key,
        expire_date__gt=ahora,
    ).first()

    if not session_obj:
        return None

    raw_original = session_obj.session_data
    store = getattr(request, "session", None) if request else None
    if store is None or not hasattr(store, "decode"):
        store = SessionStore(session_key)

    try:
        datos = store.decode(raw_original)
    except Exception as exc:
        logger.warning("Error decodificando session_data en OCC (%s): %s", session_key, exc)
        datos = {}

    if not isinstance(datos, dict):
        datos = {}

    return EstadoSesionOCC(
        session_key=session_key,
        raw_original=raw_original,
        datos_sesion=datos,
        expire_date=session_obj.expire_date,
    )


def ejecutar_guardado_occ(
    session_key: str,
    raw_original: str,
    nuevos_datos: Dict[str, Any],
    expire_date: Any,
    request: Optional[Any] = None,
) -> int:
    """Ejecuta el UPDATE condicional atómico sobre django_session usando raw_original en el WHERE.

    Retorna el número de filas afectadas (1 si éxito, 0 si hubo colisión concurrente).
    """
    store = getattr(request, "session", None) if request else None
    if store is None or not hasattr(store, "encode"):
        store = SessionStore(session_key)

    nuevo_raw = store.encode(nuevos_datos)

    filas = Session.objects.filter(
        session_key=session_key,
        session_data=raw_original,
    ).update(
        session_data=nuevo_raw,
        expire_date=expire_date,
    )

    if filas == 1 and request is not None and hasattr(request, "session"):
        # 1. Sincronizar memoria interna de Django Session
        request.session._session_cache = nuevos_datos
        # 2. DESARMAR SessionMiddleware para neutralizar el guardado tardío redundante
        request.session.modified = False

    return filas


def reconciliar_y_guardar_metadatos_occ(
    session_key: str,
    raw_original: Optional[str],
    nuevos_datos: Dict[str, Any],
    request: Optional[Any] = None,
    max_reintentos: int = 3,
) -> bool:
    """Reconcilia y persiste metadatos conversacionales sobre django_session usando OCC (C-04).

    Garantías C-04:
    - Si raw_original coincide con el session_data en base de datos, ejecuta UPDATE atómico condicional.
    - Si hubo cambios concurrentes (colisión OCC), recupera la fila fresca de django_session.
    - PRESERVA estrictamente todas las CLAVES_PROTEGIDAS_OCC de la fila fresca de la BD:
      una propuesta cancelada, ejecutada o sustituida NUNCA resucita ni vuelve a pendiente.
    - Fusiona selectivamente las CLAVES_METADATOS_CONVERSACIONALES (haku_ultimo_acceso,
      haku_conversaciones, haku_ultimo_producto_id) sin perder datos legítimos de turno.
    - Reintenta hasta max_reintentos en caso de colisión persistente.
    - Actualiza request.session._session_cache y desmarca request.session.modified = False.
    - Si falla tras reintentos, llama a neutralizar_sesion_en_memoria(request) para no corromper la BD.

    Retorna True si se persistió con éxito, False en caso contrario.
    """
    if not session_key:
        return False

    store = getattr(request, "session", None) if request else None
    if store is None or not hasattr(store, "encode"):
        store = SessionStore(session_key)

    expire_date = store.get_expiry_date()

    # Intento 0: Fast-path si raw_original coincide
    if raw_original is not None:
        nuevo_raw = store.encode(nuevos_datos)
        filas = Session.objects.filter(
            session_key=session_key,
            session_data=raw_original,
        ).update(
            session_data=nuevo_raw,
            expire_date=expire_date,
        )
        if filas == 1:
            if request is not None and hasattr(request, "session"):
                request.session._session_cache = nuevos_datos
                request.session.modified = False
                request._haku_raw_original_inicial = nuevo_raw
            return True

    # Camino de Reconciliación con reintentos
    for intento in range(max_reintentos):
        ses_obj = Session.objects.filter(session_key=session_key).first()
        if not ses_obj:
            neutralizar_sesion_en_memoria(request)
            return False

        fresh_raw = ses_obj.session_data
        try:
            datos_frescos = store.decode(fresh_raw)
        except Exception:
            datos_frescos = {}

        datos_reconciliados = dict(datos_frescos)

        # 1. CLAVES PROTEGIDAS OCC: Intocables desde la memoria local desactualizada.
        # Siempre prevalece el valor fresco de la base de datos.
        for k in CLAVES_PROTEGIDAS_OCC:
            if k in datos_frescos:
                datos_reconciliados[k] = datos_frescos[k]
            elif k in datos_reconciliados:
                del datos_reconciliados[k]

        # 2. CLAVES DE METADATOS CONVERSACIONALES: Fusionar inteligentemente
        for k in CLAVES_METADATOS_CONVERSACIONALES:
            if k in nuevos_datos:
                val_local = nuevos_datos[k]
                val_fresco = datos_frescos.get(k)
                if k == "haku_ultimo_acceso":
                    t_local = val_local if _es_timestamp_real_valido(val_local) else 0.0
                    t_fresco = val_fresco if _es_timestamp_real_valido(val_fresco) else 0.0
                    datos_reconciliados[k] = max(t_local, t_fresco) if (t_local > 0 or t_fresco > 0) else val_local
                elif k == "haku_conversaciones":
                    c_frescas = val_fresco if isinstance(val_fresco, dict) else {}
                    c_locales = val_local if isinstance(val_local, dict) else {}
                    c_merged = dict(c_frescas)
                    c_merged.update(c_locales)
                    datos_reconciliados[k] = c_merged
                elif k == "haku_conversacion":
                    if isinstance(val_local, list) and val_local:
                        datos_reconciliados[k] = val_local
                    else:
                        datos_reconciliados[k] = val_fresco
                elif k == "haku_contexto_expirado":
                    datos_reconciliados[k] = bool(val_local or val_fresco)
                elif k in ("haku_ultimo_producto_id", "haku_ultimo_almacen_id"):
                    datos_reconciliados[k] = val_local if val_local is not None else val_fresco
                elif k == "haku_conversacion_activa_id":
                    datos_reconciliados[k] = val_local if val_local else val_fresco
                else:
                    datos_reconciliados[k] = val_local

        # 3. Intentar UPDATE atómico condicional sobre fresh_raw
        raw_reconciliado = store.encode(datos_reconciliados)
        filas = Session.objects.filter(
            session_key=session_key,
            session_data=fresh_raw,
        ).update(
            session_data=raw_reconciliado,
            expire_date=expire_date,
        )
        if filas == 1:
            if request is not None and hasattr(request, "session"):
                request.session._session_cache = datos_reconciliados
                request.session.modified = False
                request._haku_raw_original_inicial = raw_reconciliado
            return True

    # Agotados los reintentos: neutralizar memoria local para impedir UPDATE ciego de SessionMiddleware
    neutralizar_sesion_en_memoria(request)
    logger.warning("Conflicto OCC en reconciliacion de metadatos de sesion para key %s", session_key)
    return False


def proteger_sesion_occ(session_store: Any, request: Optional[Any] = None) -> None:
    """Envuelve session_store.save para que cualquier guardado (explícito o de SessionMiddleware)
    utilice reconciliación OCC en lugar de un UPDATE ciego incondicional (C-04)."""
    if session_store is None:
        return
    if getattr(session_store, "_haku_occ_protegido", False):
        return

    orig_save = getattr(session_store, "save", None)
    if not callable(orig_save):
        return

    def safe_save(must_create: bool = False) -> None:
        if must_create:
            return orig_save(must_create=True)
        session_key = getattr(session_store, "session_key", None)
        if not session_key:
            return orig_save(must_create=must_create)

        raw_orig = getattr(request, "_haku_raw_original_inicial", None) if request else None
        cache = getattr(session_store, "_session_cache", None)
        if cache is None:
            return orig_save(must_create=must_create)

        exito = reconciliar_y_guardar_metadatos_occ(
            session_key=session_key,
            raw_original=raw_orig,
            nuevos_datos=cache,
            request=request,
        )
        if not exito:
            session_store.modified = False

    session_store.save = safe_save
    session_store._haku_occ_protegido = True


def persistir_propuesta_sesion_occ(
    request: Any,
    nueva_propuesta: Optional[Dict[str, Any]],
    accion: str = "ACTUALIZAR",  # "CREAR", "ACTUALIZAR", "CANCELAR", "CONFIRMAR_PURGA"
    propuesta_id_esperada: Optional[str] = None,
    estado_occ_previo: Optional[EstadoSesionOCC] = None,
    recibo_ejecucion: Optional[Dict[str, Any]] = None,
    motivo_evento: Optional[str] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Persiste cambios en la propuesta activa utilizando control de concurrencia optimista (OCC).

    Garantías:
    - Comprueba sesión viva y no expirada.
    - Valida token de sesión y propiedad de usuario de forma obligatoria y estricta.
    - En confirmación/cancelación/actualización: comprueba que propuesta_id_esperada coincida con la activa en disco.
    - Si la propuesta activa ya fue sustituida por otra, rechaza sin destruir la nueva propuesta.
    - Ejecuta UPDATE condicional WHERE session_data = raw_original (basado en snapshot previo si se provee).
    - Si tiene éxito (1 fila), neutraliza SessionMiddleware con request.session.modified = False.
    - Sincroniza HakuOperationalContext en RAM como espejo secundario.
    - Registra metadatos de ciclo de vida (haku_ultimo_evento, haku_ultima_ejecucion) vinculados al UUID.

    Retorna: (exito: bool, propuesta_resultante: Optional[dict], motivo_o_error: str)
    """
    if not hasattr(request, "session"):
        return False, None, "sin_sesion_en_request"

    session_key = getattr(request.session, "session_key", None)
    if not session_key:
        request.session.save()
        session_key = request.session.session_key
        if not session_key:
            return False, None, "no_se_pudo_crear_session_key"

    if estado_occ_previo is not None:
        estado_occ = estado_occ_previo
    else:
        estado_occ = capturar_estado_sesion_occ(session_key, request)

    if not estado_occ:
        return False, None, "sesion_inexistente_o_expirada"

    token_sesion = obtener_o_crear_session_token(estado_occ.datos_sesion)
    propuesta_actual = estado_occ.datos_sesion.get("haku_propuesta_activa")

    user_id_actual = str(getattr(getattr(request, "user", None), "id", ""))

    import time

    if accion == "CONFIRMAR_PURGA":
        if not propuesta_actual:
            return False, None, "sin_propuesta_activa"

        actual_id = str(propuesta_actual.get("id") or "")
        if propuesta_id_esperada and actual_id != str(propuesta_id_esperada):
            sustituida_id = estado_occ.datos_sesion.get("haku_propuesta_sustituida_id")
            if sustituida_id and str(sustituida_id) == str(propuesta_id_esperada):
                return False, propuesta_actual, "propuesta_sustituida"
            return False, propuesta_actual, "propuesta_id_discordante"

        if user_id_actual and str(propuesta_actual.get("usuario_id") or "") != user_id_actual:
            return False, propuesta_actual, "usuario_invalido"

        # Validación obligatoria de token criptográfico de sesión
        prop_token = propuesta_actual.get("session_token")
        if (
            not token_sesion
            or not prop_token
            or not isinstance(token_sesion, str)
            or not isinstance(prop_token, str)
            or not hmac.compare_digest(str(prop_token).strip(), str(token_sesion).strip())
        ):
            return False, propuesta_actual, "token_invalido"

        estado_occ.datos_sesion["haku_propuesta_activa"] = None
        estado_occ.datos_sesion["haku_ultimo_evento"] = {
            "propuesta_id": actual_id,
            "estado": motivo_evento or "EJECUTADA",
            "fecha": round(time.time(), 4),
        }
        if recibo_ejecucion and isinstance(recibo_ejecucion, dict):
            estado_occ.datos_sesion["haku_ultima_ejecucion"] = {
                k: (str(v) if isinstance(v, uuid.UUID) else v)
                for k, v in recibo_ejecucion.items()
            }
        propuesta_retorno = None

    elif accion == "CANCELAR":
        if not propuesta_actual:
            return True, None, "ya_cancelada"

        actual_id = str(propuesta_actual.get("id") or "")
        if propuesta_id_esperada and actual_id != str(propuesta_id_esperada):
            sustituida_id = estado_occ.datos_sesion.get("haku_propuesta_sustituida_id")
            if sustituida_id and str(sustituida_id) == str(propuesta_id_esperada):
                return False, propuesta_actual, "propuesta_sustituida"
            return False, propuesta_actual, "propuesta_id_discordante"

        if user_id_actual and str(propuesta_actual.get("usuario_id") or "") != user_id_actual:
            return False, propuesta_actual, "usuario_invalido"

        # Validación obligatoria de token criptográfico de sesión
        prop_token = propuesta_actual.get("session_token")
        if (
            not token_sesion
            or not prop_token
            or not isinstance(token_sesion, str)
            or not isinstance(prop_token, str)
            or not hmac.compare_digest(str(prop_token).strip(), str(token_sesion).strip())
        ):
            return False, propuesta_actual, "token_invalido"

        estado_occ.datos_sesion["haku_propuesta_activa"] = None
        estado_occ.datos_sesion["haku_ultimo_evento"] = {
            "propuesta_id": actual_id,
            "estado": "CANCELADA",
            "fecha": round(time.time(), 4),
        }
        propuesta_retorno = None

    elif accion in ("CREAR", "ACTUALIZAR"):
        if not isinstance(nueva_propuesta, dict):
            return False, None, "nueva_propuesta_invalida"

        if accion == "ACTUALIZAR" and propuesta_id_esperada:
            if not propuesta_actual:
                return False, None, "sin_propuesta_activa"
            actual_id = str(propuesta_actual.get("id") or "")
            if actual_id != str(propuesta_id_esperada):
                sustituida_id = estado_occ.datos_sesion.get("haku_propuesta_sustituida_id")
                if sustituida_id and str(sustituida_id) == str(propuesta_id_esperada):
                    return False, propuesta_actual, "propuesta_sustituida"
                return False, propuesta_actual, "propuesta_id_discordante"

            # Validación obligatoria de token criptográfico en propuesta previa a actualizar
            prop_token_prev = propuesta_actual.get("session_token")
            if (
                not token_sesion
                or not prop_token_prev
                or not isinstance(token_sesion, str)
                or not isinstance(prop_token_prev, str)
                or not hmac.compare_digest(str(prop_token_prev).strip(), str(token_sesion).strip())
            ):
                return False, propuesta_actual, "token_invalido"

            # Marcar la versión previa como sustituida
            estado_occ.datos_sesion["haku_propuesta_sustituida_id"] = propuesta_id_esperada
            estado_occ.datos_sesion["haku_ultimo_evento"] = {
                "propuesta_id": propuesta_id_esperada,
                "estado": "SUSTITUIDA",
                "fecha": round(time.time(), 4),
            }

        if not token_sesion or not isinstance(token_sesion, str) or len(token_sesion.strip()) < 16:
            return False, None, "token_sesion_invalido"

        # Si había una propuesta activa con ID diferente, registrarla como sustituida
        if propuesta_actual and propuesta_actual.get("id"):
            prev_id = str(propuesta_actual.get("id"))
            new_id = str(nueva_propuesta.get("id") or "")
            if prev_id and prev_id != new_id:
                estado_occ.datos_sesion["haku_propuesta_sustituida_id"] = prev_id
                estado_occ.datos_sesion["haku_ultimo_evento"] = {
                    "propuesta_id": prev_id,
                    "estado": "SUSTITUIDA",
                    "fecha": round(time.time(), 4),
                }

        # Adjuntar session_token a la nueva propuesta (exento de canonicalización HMAC)
        nueva_propuesta["session_token"] = token_sesion
        estado_occ.datos_sesion["haku_propuesta_activa"] = nueva_propuesta
        propuesta_retorno = nueva_propuesta

    else:
        return False, None, f"accion_desconocida_{accion}"

    filas = ejecutar_guardado_occ(
        session_key=session_key,
        raw_original=estado_occ.raw_original,
        nuevos_datos=estado_occ.datos_sesion,
        expire_date=estado_occ.expire_date,
        request=request,
    )

    if filas == 1:
        # Sincronizar espejo secundario en RAM (HakuOperationalContext)
        try:
            from .contexto import HakuOperationalContext
            user_id_int = getattr(getattr(request, "user", None), "id", None)
            if user_id_int:
                estado_op = HakuOperationalContext.obtener(user_id_int, session_key)
                if estado_op:
                    estado_op.propuesta_activa = estado_occ.datos_sesion.get("haku_propuesta_activa")
                    HakuOperationalContext.guardar(estado_op)
        except Exception as exc:
            logger.warning("Error sincronizando espejo RAM tras OCC (%s): %s", session_key, exc)

        return True, propuesta_retorno, "ok"

    return False, None, "conflicto_occ"


# ═════════════════════════════════════════════════════════════════════════════
# PROTOCOLO TRANSACCIONAL ÚNICO (FASE F, SUBLOTE 2.4-B)
# ═════════════════════════════════════════════════════════════════════════════


def es_error_bloqueo_sqlite(exc: Exception) -> bool:
    """Clasifica de forma precisa si una excepción corresponde a contención transitoria o colisión en SQLite."""
    for obj in (exc, getattr(exc, "__cause__", None)):
        if obj is None:
            continue
        errcode = getattr(obj, "sqlite_errorcode", None)
        errname = getattr(obj, "sqlite_errorname", None)
        # 5: SQLITE_BUSY, 6: SQLITE_LOCKED, 261: SQLITE_BUSY_RECOVERY, 517: SQLITE_BUSY_SNAPSHOT
        if errname in ("SQLITE_BUSY", "SQLITE_BUSY_SNAPSHOT", "SQLITE_LOCKED"):
            return True
        if errcode in (5, 6, 261, 517):
            return True
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def bloquear_y_leer_sesion(
    session_key: str,
    request: Optional[Any] = None,
) -> Tuple[Optional[Session], Dict[str, Any], str]:
    """Lee y bloquea la fila de django_session dentro de la transacción activa (Sublote 2.4-B).

    - En PostgreSQL: ejecuta select_for_update() para exclusión preventiva a nivel de fila.
    - En SQLite: ejecuta lectura y captura raw_original para posterior validación OCC en la primera escritura.
    """
    from django.db import connection

    if not session_key:
        return None, {}, ""

    ahora = timezone.now()
    if connection.vendor == "postgresql":
        session_obj = Session.objects.select_for_update().filter(
            session_key=session_key,
            expire_date__gt=ahora,
        ).first()
    else:
        session_obj = Session.objects.filter(
            session_key=session_key,
            expire_date__gt=ahora,
        ).first()

    if not session_obj:
        return None, {}, ""

    raw_original = session_obj.session_data
    store = getattr(request, "session", None) if request else None
    if store is None or not hasattr(store, "decode"):
        store = SessionStore(session_key)

    try:
        datos = store.decode(raw_original)
    except Exception as exc:
        logger.warning("Error decodificando session_data transaccional (%s): %s", session_key, exc)
        datos = {}

    if not isinstance(datos, dict):
        datos = {}

    return session_obj, datos, raw_original


def guardar_sesion_transaccional(
    session_obj: Session,
    nuevos_datos: Dict[str, Any],
    raw_original: str,
    request: Optional[Any] = None,
) -> bool:
    """Actualiza django_session dentro de la transacción activa respetando el motor (Sublote 2.4-B).

    - En PostgreSQL: guarda directamente sobre session_obj (fila ya bloqueada por select_for_update).
    - En SQLite: ejecuta UPDATE condicional WHERE session_data = raw_original para verificar que no hubo colisión de snapshot.
    - Sincroniza request.session._session_cache y desactiva request.session.modified para desarmar SessionMiddleware.
    """
    from django.db import connection

    store = getattr(request, "session", None) if request else None
    if store is None or not hasattr(store, "encode"):
        store = SessionStore(session_obj.session_key)

    # Serializar UUIDs a cadenas en nuevos_datos para compatibilidad total con el serializador JSON de Django
    if "haku_ultima_ejecucion" in nuevos_datos and isinstance(nuevos_datos["haku_ultima_ejecucion"], dict):
        nuevos_datos["haku_ultima_ejecucion"] = {
            k: (str(v) if isinstance(v, uuid.UUID) else v)
            for k, v in nuevos_datos["haku_ultima_ejecucion"].items()
        }

    nuevo_raw = store.encode(nuevos_datos)

    if connection.vendor == "postgresql":
        session_obj.session_data = nuevo_raw
        session_obj.save(update_fields=["session_data"])
        filas = 1
    else:
        filas = Session.objects.filter(
            session_key=session_obj.session_key,
            session_data=raw_original,
        ).update(session_data=nuevo_raw)

    if filas == 1:
        if request is not None and hasattr(request, "session"):
            request.session._session_cache = nuevos_datos
            request.session.modified = False
        return True

    raise ErrorConflictoOCC(f"Conflicto OCC en sesión {session_obj.session_key}: session_data fue modificado concurrentemente.")


def buscar_ejecucion_existente_ledger(
    propuesta_id: str,
    usuario: Optional[Any] = None,
    firma_hmac_esperada: Optional[str] = None,
    conversacion_id_esperada: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Consulta el ledger inmutable (Movimiento / Auditoria) para verificar si la propuesta ya fue confirmada y ejecutada.

    Garantiza idempotencia estricta ante reintentos de cliente tras caídas de red post-commit.
    """
    if not propuesta_id:
        return None

    import re
    import time
    from django.db import OperationalError
    from inventario.models import Movimiento, Auditoria
    from django.db.models import Q

    max_intentos = 5
    for intento in range(max_intentos):
        try:
            clave_canonica = f"haku-prop-{propuesta_id}"[:100]
            q_mov = (
                Q(clave_idempotencia=clave_canonica)
                | Q(observacion__contains=f"propuesta_id:{propuesta_id}")
                | Q(motivo__contains=f"propuesta_id:{propuesta_id}")
            )
            mov = Movimiento.objects.filter(q_mov).first()
            if mov:
                if usuario is not None and getattr(usuario, "id", None):
                    if str(mov.usuario_id) != str(usuario.id):
                        return None

                # Verificación estricta de aislamiento de conversación
                if conversacion_id_esperada:
                    cid_mov = None
                    if mov.observacion and "[conversacion_id:" in mov.observacion:
                        m_c = re.search(r'\[conversacion_id:([a-f0-9\-]+)\]', mov.observacion)
                        if m_c:
                            cid_mov = m_c.group(1).strip()
                    if cid_mov and cid_mov.lower() != str(conversacion_id_esperada).strip().lower():
                        return None

                # Si se suministró la firma HMAC esperada, verificar correspondencia con la huella
                if firma_hmac_esperada:
                    huella_existente = getattr(mov, "huella_payload", "")
                    if not huella_existente and mov.observacion and "[payload_hash:" in mov.observacion:
                        m = re.search(r'\[payload_hash:([a-f0-9]{64})\]', mov.observacion)
                        if m:
                            huella_existente = m.group(1)
                    # Comparar firma o hash
                    if huella_existente:
                        partes = firma_hmac_esperada.split(":", 1)
                        firma_hex = partes[1].strip().lower() if len(partes) == 2 else firma_hmac_esperada.strip().lower()
                        if not (
                            hmac.compare_digest(huella_existente.lower(), firma_hmac_esperada.lower())
                            or hmac.compare_digest(huella_existente.lower(), firma_hex)
                        ):
                            return None

                return {
                    "exito": True,
                    "idempotente": True,
                    "estado": "EJECUTADA",
                    "mensaje_pantalla": f"Operación previamente confirmada (Movimiento #{mov.id}).",
                    "mensaje_verbal": "Esta operación ya había sido confirmada previamente.",
                    "movimiento_id": str(mov.id),
                    "stock_nuevo": mov.stock_disponible_posterior,
                }

            # Comprobar operaciones de catálogo en Auditoria
            clave_canonica = f"haku-prop-{propuesta_id}"[:100]
            q_aud = (
                Q(valores_nuevos__propuesta_id=propuesta_id)
                | Q(motivo__contains=f"propuesta_id:{propuesta_id}")
                | Q(motivo__contains=clave_canonica)
            )
            aud = Auditoria.objects.filter(tipo_entidad="Producto").filter(q_aud).first()
            if aud:
                if (
                    not aud.usuario_id
                    or usuario is None
                    or not getattr(usuario, "id", None)
                    or str(aud.usuario_id) != str(usuario.id)
                ):
                    return None

                # Verificación estricta de aislamiento de conversación en Auditoria
                if conversacion_id_esperada:
                    cid_aud = None
                    if isinstance(aud.valores_nuevos, dict) and aud.valores_nuevos.get("conversacion_id"):
                        cid_aud = str(aud.valores_nuevos["conversacion_id"]).strip()
                    elif aud.motivo and "[conversacion_id:" in aud.motivo:
                        m_c = re.search(r'\[conversacion_id:([^\]]+)\]', aud.motivo)
                        if m_c:
                            cid_aud = m_c.group(1).strip()
                    if cid_aud and cid_aud.lower() != str(conversacion_id_esperada).strip().lower():
                        return None

                # Si se suministró la firma HMAC esperada, verificar correspondencia con la huella registrada en Auditoria
                if firma_hmac_esperada:
                    huella_existente = ""
                    if isinstance(aud.valores_nuevos, dict):
                        huella_existente = aud.valores_nuevos.get("payload_firma") or aud.valores_nuevos.get("payload_hash", "")
                    if not huella_existente and aud.motivo and "[payload_firma:" in aud.motivo:
                        m_f = re.search(r'\[payload_firma:([^\]]+)\]', aud.motivo)
                        if m_f:
                            huella_existente = m_f.group(1).strip()
                    if not huella_existente and aud.motivo and "[payload_hash:" in aud.motivo:
                        m = re.search(r'\[payload_hash:([a-f0-9]{64})\]', aud.motivo)
                        if m:
                            huella_existente = m.group(1).strip()

                    if huella_existente:
                        partes = firma_hmac_esperada.split(":", 1)
                        firma_hex = partes[1].strip().lower() if len(partes) == 2 else firma_hmac_esperada.strip().lower()
                        if not (
                            hmac.compare_digest(huella_existente.lower(), firma_hmac_esperada.lower())
                            or hmac.compare_digest(huella_existente.lower(), firma_hex)
                        ):
                            return None

                if aud.accion in ("CREAR_PRODUCTO", "CREAR"):
                    prod = Producto.objects.filter(id=aud.id_entidad).first()
                    prod_nom = prod.nombre if prod else "Producto"
                    prod_cod = prod.codigo_interno if prod else (aud.valores_nuevos.get("producto_codigo") if isinstance(aud.valores_nuevos, dict) else "")
                    stock_tot = sum(e.stock_total for e in prod.existencias.all()) if prod else 0
                    return {
                        "exito": True,
                        "idempotente": True,
                        "estado": "EJECUTADA",
                        "producto_id": str(aud.id_entidad),
                        "producto_codigo": prod_cod,
                        "movimiento_id": None,
                        "stock_nuevo": stock_tot,
                        "mensaje_pantalla": f"El producto {prod_nom} ({prod_cod}) ya fue creado previamente.",
                        "mensaje_verbal": "Esta operación ya había sido confirmada previamente.",
                    }

                op_label = "inhabilitación" if aud.accion == "INHABILITAR_PRODUCTO" else "habilitación"
                return {
                    "exito": True,
                    "idempotente": True,
                    "estado": "EJECUTADA",
                    "movimiento_id": None,
                    "mensaje_pantalla": f"La {op_label} de este producto ya fue confirmada y ejecutada previamente.",
                    "mensaje_verbal": "Esta operación ya había sido confirmada previamente.",
                }

            return None
        except OperationalError as err:
            if es_error_bloqueo_sqlite(err) and intento < max_intentos - 1:
                time.sleep(0.05 * (2 ** intento))
                continue
            raise


def reconciliar_sesion_propuesta_ejecutada(
    session_key: str,
    request: Any,
    propuesta_id: str,
    res_ledger: Dict[str, Any],
    max_intentos: int = 4,
) -> bool:
    """Reconcilia la sesión cuando una propuesta ya fue confirmada y ejecutada en el ledger (H-01).

    Garantías de seguridad y consistencia:
    - Comprueba el estado de la sesión dentro de una transacción atómica protegida.
    - Si haku_propuesta_activa corresponde exactamente al propuesta_id ejecutado:
      - Purga la propuesta activa (haku_propuesta_activa = None).
      - Registra haku_ultimo_evento: EJECUTADA.
      - Registra haku_ultima_ejecucion con el recibo auxiliar (movimiento_id, timestamp).
      - Registra evidencia terminal verificable en haku_historial_propuestas_terminales.
      - Libera el cerrojo de modificación si está vinculado a esta propuesta o su P0 terminal probada.
      - Persiste usando protocolo transaccional seguro (select_for_update en PostgreSQL, OCC en SQLite).
    - Si la sesión contiene una propuesta DIFERENTE (e.g. una solicitud posterior nueva P2) o no hay propuesta activa:
      - Conserva la sesión íntegramente y no altera la propuesta nueva.
    - En caso de contención de base de datos o colisión OCC transitoria:
      - Reintenta hasta max_intentos veces con backoff exponencial y relectura fresca de la sesión.
    - Si se agotan los reintentos, retorna False sin destruir datos.
    """
    if not session_key or not propuesta_id:
        return True

    import time
    from django.db import transaction, OperationalError

    propuesta_id_str = str(propuesta_id).strip()

    for intento in range(max_intentos):
        try:
            with transaction.atomic(using="default"):
                session_obj, datos_sesion, raw_orig = bloquear_y_leer_sesion(session_key, request)
                if not session_obj:
                    return True

                prop_activa = datos_sesion.get("haku_propuesta_activa")
                if not prop_activa or not isinstance(prop_activa, dict):
                    # No hay propuesta activa que purgar
                    return True

                actual_id = str(prop_activa.get("id") or "").strip()
                if actual_id != propuesta_id_str:
                    # La propuesta activa en sesión es distinta (ej. una nueva propuesta P2 posterior).
                    # Preservarla intacta sin tocarla ni sobrescribir metadatos.
                    return True

                # Coincidencia exacta: purgar propuesta activa y reconciliar recibo de ejecución
                datos_sesion["haku_propuesta_activa"] = None
                datos_sesion["haku_ultimo_evento"] = {
                    "propuesta_id": propuesta_id_str,
                    "estado": "EJECUTADA",
                    "fecha": round(time.time(), 4),
                }

                mov_id = res_ledger.get("movimiento_id")
                datos_sesion["haku_ultima_ejecucion"] = {
                    "tipo_registro": "RECIBO_AUXILIAR",
                    "propuesta_id": propuesta_id_str,
                    "movimiento_id": str(mov_id) if mov_id else None,
                    "ejecutado_en": round(time.time(), 4),
                }

                # Registrar evidencia terminal en sesión
                registrar_evidencia_terminal_sesion(datos_sesion, prop_activa, "EJECUTADA")

                # Gestión segura de cerrojo de modificación: solo liberar si vinculado a esta propuesta
                bloqueo = datos_sesion.get("haku_bloqueo_modificacion")
                if isinstance(bloqueo, dict):
                    prop_bloq = str(bloqueo.get("propuesta_id_bloqueada") or "").strip()
                    if prop_bloq == propuesta_id_str:
                        datos_sesion["haku_bloqueo_modificacion"] = None
                    else:
                        uid_b = bloqueo.get("usuario_id")
                        ev_p0 = obtener_evidencia_verificable_propuesta_terminal(datos_sesion, prop_bloq, uid_b)
                        if ev_p0 and ev_p0.get("estado_terminal") in ("EJECUTADA", "CANCELADA", "SUSTITUIDA", "CADUCADA", "RECHAZADA"):
                            datos_sesion["haku_bloqueo_modificacion"] = None

                guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                return True

        except (OperationalError, ErrorConflictoOCC) as err:
            es_occ = isinstance(err, ErrorConflictoOCC)
            es_bloqueo = isinstance(err, OperationalError) and es_error_bloqueo_sqlite(err)
            if es_occ or es_bloqueo:
                if intento < max_intentos - 1:
                    time.sleep(0.05 * (2 ** intento))
                    continue
                logger.warning(
                    "No se pudo reconciliar la sesión para propuesta %s tras %d intentos: %s",
                    propuesta_id_str,
                    max_intentos,
                    err,
                )
                return False
            raise
        except Exception as exc:
            logger.warning("Error inesperado reconciliando sesión (%s): %s", session_key, exc)
            return False

    return False


def invalidar_propuesta_pre_despliegue_en_transaccion(
    session_key: str,
    request: Any,
    propuesta_id: str,
    max_intentos: int = 4,
) -> Tuple[bool, str]:
    """Invalida de manera atómica una propuesta emitida con un contrato previo sin afectar movimientos ni operaciones confirmadas.

    Verifica las precondiciones R-06 si hay un cerrojo adjunto.
    Si el cerrojo es ajeno o corrupto, aborta con conflicto (fail-closed).
    Si es válido o no hay cerrojo, purga la propuesta activa y registra CADUCADA.
    """
    if not session_key or not propuesta_id:
        return False, "parametros_invalidos"

    from django.db import transaction, OperationalError
    p_id_str = str(propuesta_id).strip()

    for intento in range(max_intentos):
        try:
            with transaction.atomic(using="default"):
                session_obj, datos_sesion, raw_orig = bloquear_y_leer_sesion(session_key, request)
                if not session_obj:
                    return False, "sesion_inexistente"

                prop_activa = datos_sesion.get("haku_propuesta_activa")
                if not prop_activa or str(prop_activa.get("id") or "").strip() != p_id_str:
                    return True, "ya_no_activa"

                # Evaluar cerrojo adjunto si existe (Regla R-06)
                bloqueo = datos_sesion.get("haku_bloqueo_modificacion")
                if isinstance(bloqueo, dict):
                    cid_bloq = str(bloqueo.get("conversacion_id") or "").strip()
                    uid_bloq = str(bloqueo.get("usuario_id") or "").strip()
                    user_actual_id = str(getattr(getattr(request, "user", None), "id", "") or "")

                    if uid_bloq and user_actual_id and uid_bloq != user_actual_id:
                        return False, "conflicto_integridad_bloqueo"

                    cid_prop = str(prop_activa.get("conversacion_id") or "").strip()
                    if cid_prop and cid_bloq and cid_prop != cid_bloq:
                        return False, "conflicto_integridad_bloqueo"

                    datos_sesion["haku_bloqueo_modificacion"] = None

                datos_sesion["haku_propuesta_activa"] = None
                datos_sesion["haku_ultimo_evento"] = {
                    "propuesta_id": p_id_str,
                    "usuario_id": str(getattr(getattr(request, "user", None), "id", "") or ""),
                    "estado": "CADUCADA",
                    "fecha": round(time.time(), 4),
                }

                registrar_evidencia_terminal_sesion(datos_sesion, prop_activa, "CADUCADA")
                guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)

                user_inst = getattr(request, "user", None)
                if user_inst and getattr(user_inst, "is_authenticated", False):
                    registrar_auditoria_caducidad(user_inst, p_id_str, prop_activa)

                return True, "ok"

        except (OperationalError, ErrorConflictoOCC) as err:
            es_occ = isinstance(err, ErrorConflictoOCC)
            es_bloqueo = isinstance(err, OperationalError) and es_error_bloqueo_sqlite(err)
            if (es_occ or es_bloqueo) and intento < max_intentos - 1:
                time.sleep(0.05 * (2 ** intento))
                continue
            return False, "conflicto_concurrencia"
        except Exception as exc:
            logger.warning("Error invalidando propuesta pre-despliegue (%s): %s", p_id_str, exc)
            return False, "error_inesperado"

    return False, "agotados_reintentos"
