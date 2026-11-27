"""Serializadores y validadores para la API interna de Haku Inventario.

Valida peticiones entrantes sin requerir Django REST Framework, usando
formularios y validadores nativos de Django para mantener cero dependencias
externas innecesarias.
"""

import json
import uuid
from decimal import Decimal

from django.core.exceptions import ValidationError


class ValidadorComando:
    """Valida los comandos de texto o voz enviados a Haku."""

    @staticmethod
    def validar(datos: dict) -> dict:
        mensaje = datos.get("mensaje", "").strip()
        if not mensaje:
            raise ValidationError({"mensaje": "El mensaje no puede estar vacío."})
        if len(mensaje) > 1000:
            raise ValidationError({"mensaje": "El mensaje excede los 1000 caracteres."})

        conversacion_id_raw = datos.get("conversacion_id")
        conversacion_id = None
        if conversacion_id_raw is not None and str(conversacion_id_raw).strip():
            cid_str = str(conversacion_id_raw).strip()
            try:
                val_u = uuid.UUID(cid_str, version=4)
                conversacion_id = str(val_u)
            except (ValueError, TypeError):
                raise ValidationError({"conversacion_id": "El conversacion_id debe ser un UUID4 válido."})

        return {
            "mensaje": mensaje,
            "origen": datos.get("origen", "texto"),  # "texto" o "voz"
            "conversacion_id": conversacion_id,
        }


def validar_conversacion_id_estricto(conversacion_id_raw) -> str:
    """Valida que conversacion_id sea obligatorio y cumpla formato canónico UUID4 en minúsculas."""
    if conversacion_id_raw is None or (isinstance(conversacion_id_raw, str) and not conversacion_id_raw.strip()):
        raise ValidationError({"conversacion_id": "conversacion_id_obligatorio"})
    if type(conversacion_id_raw) is not str or len(conversacion_id_raw) != 36:
        raise ValidationError({"conversacion_id": "El conversacion_id debe ser un UUID4 válido."})
    try:
        val_u = uuid.UUID(conversacion_id_raw)
        if val_u.version != 4 or val_u.variant != uuid.RFC_4122 or str(val_u) != conversacion_id_raw:
            raise ValidationError({"conversacion_id": "El conversacion_id debe ser un UUID4 canónico en minúsculas."})
        return conversacion_id_raw
    except (ValueError, TypeError, AttributeError):
        raise ValidationError({"conversacion_id": "El conversacion_id debe ser un UUID4 válido."})


def validar_propuesta_id_estricto(propuesta_id_raw) -> str:
    """Valida que propuesta_id sea obligatorio y cumpla formato canónico UUID4 en minúsculas."""
    if propuesta_id_raw is None or (isinstance(propuesta_id_raw, str) and not propuesta_id_raw.strip()):
        raise ValidationError({"propuesta_id": "Debe proporcionar el ID de la propuesta."})
    if type(propuesta_id_raw) is not str or len(propuesta_id_raw) != 36:
        raise ValidationError({"propuesta_id": "El propuesta_id debe ser un UUID4 válido."})
    try:
        val_u = uuid.UUID(propuesta_id_raw)
        if val_u.version != 4 or val_u.variant != uuid.RFC_4122 or str(val_u) != propuesta_id_raw:
            raise ValidationError({"propuesta_id": "El propuesta_id debe ser un UUID4 canónico en minúsculas."})
        return propuesta_id_raw
    except (ValueError, TypeError, AttributeError):
        raise ValidationError({"propuesta_id": "El propuesta_id debe ser un UUID4 válido."})


class ValidadorConfirmacion:
    """Valida la confirmación de una propuesta pendiente (H-02)."""

    @staticmethod
    def validar(datos: dict) -> dict:
        propuesta_id = validar_propuesta_id_estricto(datos.get("propuesta_id"))
        conversacion_id = validar_conversacion_id_estricto(datos.get("conversacion_id"))

        idempotency_key = datos.get("idempotency_key", "")
        if isinstance(idempotency_key, str):
            idempotency_key = idempotency_key.strip()
        else:
            idempotency_key = str(idempotency_key) if idempotency_key is not None else ""

        return {
            "propuesta_id": propuesta_id,
            "conversacion_id": conversacion_id,
            "idempotency_key": idempotency_key or None,
        }


class ValidadorCancelacion:
    """Valida la cancelación de una propuesta pendiente (H-02)."""

    @staticmethod
    def validar(datos: dict) -> dict:
        propuesta_id = validar_propuesta_id_estricto(datos.get("propuesta_id"))
        conversacion_id = validar_conversacion_id_estricto(datos.get("conversacion_id"))
        return {
            "propuesta_id": propuesta_id,
            "conversacion_id": conversacion_id,
        }


class ValidadorActualizarDestinatario:
    """Valida la actualización de destinatario de una propuesta pendiente (H-02)."""

    @staticmethod
    def validar(datos: dict) -> dict:
        propuesta_id = validar_propuesta_id_estricto(datos.get("propuesta_id"))
        destinatario = str(datos.get("destinatario", "")).strip()
        if not destinatario or len(destinatario) < 2:
            raise ValidationError({"destinatario": "El nombre del destinatario debe contener al menos 2 caracteres."})

        conversacion_id = validar_conversacion_id_estricto(datos.get("conversacion_id"))
        return {
            "propuesta_id": propuesta_id,
            "destinatario": destinatario,
            "conversacion_id": conversacion_id,
        }


class ValidadorConsultaStock:
    """Valida parámetros para consulta directa de stock."""

    @staticmethod
    def validar(datos: dict) -> dict:
        termino = datos.get("termino", "").strip()
        almacen_id = datos.get("almacen_id")
        return {
            "termino": termino,
            "almacen_id": int(almacen_id) if almacen_id else None,
        }


class ValidadorSintesisVoz:
    """Valida la petición de síntesis de voz."""

    @staticmethod
    def validar(datos: dict) -> dict:
        texto = datos.get("texto", "").strip()
        if not texto:
            raise ValidationError({"texto": "El texto para síntesis no puede estar vacío."})
        if len(texto) > 2000:
            raise ValidationError({"texto": "El texto excede el límite de 2000 caracteres."})
        return {"texto": texto}
