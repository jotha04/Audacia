"""Módulo de instrumentación y telemetría de latencia end-to-end para HAKU Inventario.

Garantías obligatorias:
1. Estrictamente sin datos sensibles: NUNCA registra contenido de audio,
   transcripciones de texto, credenciales ni identificadores de usuario.
2. Bajo impacto: mediciones de alta precisión con time.perf_counter().
3. Formato estructurado uniforme:
   [HAKU-TIMING] interaction=<id> stage=<etapa> duration_ms=<ms> [clave=valor ...]
4. Sin interferencia en lógica transaccional ni mutaciones en base de datos.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger("integraciones.haku.timing")


def registrar_timing(
    interaction_id: Optional[str],
    stage: str,
    duration_ms: float,
    **extra: Any,
) -> str:
    """Emite un registro estructurado de telemetría de latencia.

    Formato:
    [HAKU-TIMING] interaction=<id> stage=<etapa> duration_ms=<ms> [k=v ...]
    """
    int_id = interaction_id or "int_anonima"
    extra_items = []
    for k, v in extra.items():
        if v is not None:
            # Formatear números flotantes a 2 decimales
            if isinstance(v, float):
                extra_items.append(f"{k}={v:.2f}")
            else:
                extra_items.append(f"{k}={v}")

    extra_str = (" " + " ".join(extra_items)) if extra_items else ""
    linea = f"[HAKU-TIMING] interaction={int_id} stage={stage} duration_ms={duration_ms:.2f}{extra_str}"

    # Se emite como warning para asegurar su impresión en la consola de Django runserver
    # independientemente del nivel configurado para el logger raíz en desarrollo.
    logger.warning(linea)
    return linea


@contextmanager
def cronometrar_etapa(
    interaction_id: Optional[str],
    stage: str,
    metricas_dict: Optional[Dict[str, float]] = None,
    **extra: Any,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager para medir con precisión el tiempo de ejecución de un bloque de código."""
    datos_contexto: Dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        yield datos_contexto
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if metricas_dict is not None:
            metricas_dict[stage] = round(dt_ms, 2)
        extra_combinado = {**extra, **datos_contexto}
        registrar_timing(interaction_id, stage, dt_ms, **extra_combinado)


def obtener_ids_haku(request: Any) -> tuple[str, str]:
    """Extrae o genera (interaction_id, request_id) desde los encabezados HTTP.

    Garantías de seguridad y privacidad:
    - Identificadores puramente aleatorios sin datos personales ni del usuario.
    - Headers soportados:
      * X-Haku-Interaction-ID (HTTP_X_HAKU_INTERACTION_ID)
      * X-Haku-Request-ID (HTTP_X_HAKU_REQUEST_ID)
    """
    import uuid

    meta = getattr(request, "META", {}) if request else {}
    int_id = meta.get("HTTP_X_HAKU_INTERACTION_ID")
    if not int_id:
        int_id = f"int_{uuid.uuid4().hex[:12]}"

    req_id = meta.get("HTTP_X_HAKU_REQUEST_ID")
    if not req_id:
        req_id = f"req_{uuid.uuid4().hex[:12]}"

    return str(int_id), str(req_id)


def adjuntar_headers_haku(
    response: Any,
    interaction_id: str,
    request_id: str,
    **duraciones_ms: Optional[float],
) -> Any:
    """Adjunta los encabezados de telemetría a una HttpResponse de Django."""
    if response is None:
        return response
    response["X-Haku-Interaction-ID"] = interaction_id
    response["X-Haku-Request-ID"] = request_id
    for k, v in duraciones_ms.items():
        if v is not None:
            header_name = f"X-Haku-{k.replace('_', '-').title()}-Ms"
            response[header_name] = f"{v:.2f}"
    return response

