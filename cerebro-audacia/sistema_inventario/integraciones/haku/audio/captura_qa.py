"""Captura temporal y acotada de audios en la base QA humana aislada."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from django.conf import settings


logger = logging.getLogger("integraciones.haku.audio.qa")
_LOCK = threading.RLock()
_PREFIX = "hakuqa_audio_"
_EXTENSIONES_MIME = {
    "audio/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/opus": ".opus",
}


def _configuracion_captura() -> tuple[Path, timedelta] | None:
    """Restringe la captura a una carpeta dedicada junto a la BD QA en %TEMP%."""
    if not getattr(settings, "HAKU_HUMAN_QA_SAVE_AUDIO", False):
        return None

    try:
        db_path = Path(settings.DATABASES["default"]["NAME"]).resolve(strict=True)
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
        qa_root = db_path.parent
        capture_dir = Path(settings.HAKU_HUMAN_QA_AUDIO_DIR).resolve(strict=False)
        retention_hours = int(settings.HAKU_HUMAN_QA_AUDIO_RETENTION_HOURS)
    except Exception as exc:
        raise RuntimeError("La captura QA está habilitada, pero falta su configuración aislada") from exc

    if (
        db_path.name != "fase_g.sqlite3"
        or qa_root.parent != temp_root
        or not qa_root.name.startswith("haku-fase-g-human-qa-")
        or capture_dir.parent != qa_root
        or capture_dir.name != "audios_temporales"
    ):
        raise RuntimeError("La captura de audio solo se permite junto a la base QA humana bajo %TEMP%")
    if not 1 <= retention_hours <= 168:
        raise RuntimeError("La retención de audio QA debe estar entre 1 y 168 horas")

    return capture_dir, timedelta(hours=retention_hours)


def limpiar_capturas_expiradas() -> int:
    """Elimina audios y metadatos propios de captura que ya vencieron."""
    try:
        configuracion = _configuracion_captura()
        if configuracion is None:
            return 0
        capture_dir, retention = configuracion
        if not capture_dir.is_dir():
            return 0

        vencimiento = (datetime.now(timezone.utc) - retention).timestamp()
        eliminados = 0
        with _LOCK:
            for path in capture_dir.glob(f"{_PREFIX}*"):
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    if path.stat().st_mtime < vencimiento:
                        path.unlink()
                        eliminados += 1
                except FileNotFoundError:
                    continue
        return eliminados
    except Exception:
        logger.warning("No se pudieron limpiar audios QA vencidos", exc_info=True)
        return 0


def _escribir_metadatos(path: Path, datos: dict[str, Any]) -> None:
    temporal = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporal.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporal, path)


def guardar_audio_qa(
    audio_bytes: bytes,
    *,
    interaction_id: str,
    request_id: str,
    content_type: str,
) -> dict[str, str] | None:
    """Guarda el archivo original y un manifiesto correlacionable, solo en QA."""
    try:
        configuracion = _configuracion_captura()
        if configuracion is None or not audio_bytes:
            return None
        capture_dir, _retention = configuracion
        limpiar_capturas_expiradas()
        capture_dir.mkdir(parents=True, exist_ok=True)

        mime = (content_type or "").split(";", 1)[0].strip().lower()
        extension = _EXTENSIONES_MIME.get(mime, ".bin")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_interaction = re.sub(r"[^A-Za-z0-9_-]", "_", interaction_id or "sin-interaction")[:80]
        unique = uuid.uuid4().hex[:10]
        stem = f"{_PREFIX}{stamp}_{safe_interaction}_{unique}"
        audio_path = capture_dir / f"{stem}{extension}"
        metadata_path = capture_dir / f"{stem}.json"

        metadata = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "interaction_id": interaction_id,
            "request_id": request_id,
            "content_type": mime or "unknown",
            "audio_file": audio_path.name,
            "audio_bytes": len(audio_bytes),
            "status": "recibido",
        }
        with _LOCK:
            audio_path.write_bytes(audio_bytes)
            _escribir_metadatos(metadata_path, metadata)
        return {"audio_path": str(audio_path), "metadata_path": str(metadata_path)}
    except Exception:
        logger.warning("No se pudo guardar el audio temporal de QA", exc_info=True)
        return None


def actualizar_metadatos_audio_qa(
    captura: dict[str, str] | None,
    *,
    status: str,
    transcript: str,
    metrics: dict[str, Any],
) -> None:
    """Asocia el resultado STT con el audio temporal sin interrumpir la consulta."""
    if not captura:
        return
    try:
        metadata_path = Path(captura["metadata_path"])
        configuracion = _configuracion_captura()
        if configuracion is None or metadata_path.parent != configuracion[0].resolve(strict=False):
            return
        with _LOCK:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "status": status,
                    "transcript": transcript,
                    "metrics": metrics,
                    "processed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _escribir_metadatos(metadata_path, metadata)
    except Exception:
        logger.warning("No se pudieron asociar métricas al audio temporal de QA", exc_info=True)
