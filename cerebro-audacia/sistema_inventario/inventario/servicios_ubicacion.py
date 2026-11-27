"""
Módulo de servicios para la normalización de ubicaciones físicas y prevención estricta de duplicados.

Arquitectura:
1. Normalización visual limpia (Unicode NFC, trim de espacios y colapso de espacios múltiples).
2. Clave canónica insensible a mayúsculas, minúsculas, tildes y espacios.
3. Validación en capa de servicio y modelo que previene duplicados concurrentes e interactivos.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Optional, Tuple

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction

logger = logging.getLogger("inventario.ubicaciones")


# Mapeo exhaustivo de palabras numéricas cardinales en español (con variantes de género)
MAPA_PALABRAS_NUMERICAS = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiun": 21,
    "veintiuno": 21,
    "veintiuna": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "doscientas": 200,
    "trescientos": 300,
    "trescientas": 300,
    "cuatrocientos": 400,
    "cuatrocientas": 400,
    "quinientos": 500,
    "quinientas": 500,
    "seiscientos": 600,
    "seiscientas": 600,
    "setecientos": 700,
    "setecientas": 700,
    "ochocientos": 800,
    "ochocientas": 800,
    "novecientos": 900,
    "novecientas": 900,
    "mil": 1000,
}


def normalizar_nombre_ubicacion(nombre: str) -> str:
    """Limpia y estandariza la representación visible del nombre de una ubicación (Unicode NFC y espacios colapsados)."""
    if not nombre:
        return ""
    # Normalización canónica Unicode NFC
    texto = unicodedata.normalize("NFC", str(nombre))
    # Colapsar espacios redundantes y trim
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def normalizar_texto_canonico(texto: str) -> str:
    """
    Normaliza deterministamente una cadena para comparación canónica:
    1. Unicode NFC y colapso de espacios.
    2. Eliminación de tildes/marcas diacríticas (NFD + Mn strip).
    3. Minúsculas (casefold).
    4. Conversión de palabras numéricas simples y compuestas ("treinta y dos" -> "32", "ciento cinco" -> "105").
    5. Normalización de números con ceros a la izquierda ("01" -> "1").
    """
    if not texto:
        return ""
    limpio = normalizar_nombre_ubicacion(texto)
    # Descomponer diacríticos
    descompuesto = unicodedata.normalize("NFD", limpio)
    sin_diacriticos = "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")
    casefolded = sin_diacriticos.lower().strip()

    tokens = re.findall(r"\w+|\S", casefolded)
    i = 0
    tokens_canonicos = []
    centenas_valores = {100, 200, 300, 400, 500, 600, 700, 800, 900}
    decenas_valores = {30, 40, 50, 60, 70, 80, 90}

    while i < len(tokens):
        tok = tokens[i]
        val_tok = MAPA_PALABRAS_NUMERICAS.get(tok)

        # 1. Comprobar centenas compuestas (ej: "ciento cinco", "ciento veinticinco", "ciento treinta y dos")
        if val_tok in centenas_valores:
            # Caso 1a: Centena + Decena + 'y' + Unidad (ej: "ciento treinta y dos" -> 132)
            if (
                i + 3 < len(tokens)
                and tokens[i+1] in MAPA_PALABRAS_NUMERICAS
                and tokens[i+2] == "y"
                and tokens[i+3] in MAPA_PALABRAS_NUMERICAS
            ):
                val_dec = MAPA_PALABRAS_NUMERICAS[tokens[i+1]]
                val_uni = MAPA_PALABRAS_NUMERICAS[tokens[i+3]]
                if val_dec in decenas_valores and 1 <= val_uni <= 9:
                    tokens_canonicos.append(str(val_tok + val_dec + val_uni))
                    i += 4
                    continue

            # Caso 1b: Centena + palabra numérica 1..99 (ej: "ciento cinco" -> 105, "ciento veinticinco" -> 125)
            if i + 1 < len(tokens) and tokens[i+1] in MAPA_PALABRAS_NUMERICAS:
                val_sig = MAPA_PALABRAS_NUMERICAS[tokens[i+1]]
                if 1 <= val_sig < 100:
                    tokens_canonicos.append(str(val_tok + val_sig))
                    i += 2
                    continue

            # Caso 1c: Centena + dígito 1..99 (ej: "ciento 5" -> 105)
            if i + 1 < len(tokens) and tokens[i+1].isdigit():
                val_dig = int(tokens[i+1])
                if 1 <= val_dig < 100:
                    tokens_canonicos.append(str(val_tok + val_dig))
                    i += 2
                    continue

            tokens_canonicos.append(str(val_tok))
            i += 1
            continue

        # 2. Comprobar decenas compuestas tipo 'treinta y dos' o 'cuarenta y cinco'
        if i + 2 < len(tokens) and tok in MAPA_PALABRAS_NUMERICAS and tokens[i+1] == "y" and tokens[i+2] in MAPA_PALABRAS_NUMERICAS:
            val_dec = MAPA_PALABRAS_NUMERICAS[tok]
            val_uni = MAPA_PALABRAS_NUMERICAS[tokens[i+2]]
            if val_dec in decenas_valores and 1 <= val_uni <= 9:
                tokens_canonicos.append(str(val_dec + val_uni))
                i += 3
                continue

        # 3. Comprobar palabras numéricas simples o dígitos
        if val_tok is not None:
            tokens_canonicos.append(str(val_tok))
        elif tok.isdigit():
            tokens_canonicos.append(str(int(tok)))
        else:
            tokens_canonicos.append(tok)
        i += 1

    return " ".join(tokens_canonicos)


def generar_clave_canonica_ubicacion(almacen_id: int, nombre: str) -> str:
    """
    Genera la clave canónica de deduplicación insensible a mayúsculas, tildes,
    espacios redundantes, ceros a la izquierda y palabras numéricas cardinales.
    Formato: alm_{almacen_id}:{texto_canonico}
    """
    canonico = normalizar_texto_canonico(nombre)
    return f"alm_{almacen_id}:{canonico}"


class UbicacionService:
    """Servicio de dominio para la gestión, validación y búsqueda de ubicaciones sin duplicados."""

    @classmethod
    def validar_no_duplicado(
        cls,
        almacen_id: int,
        nombre: str,
        excluir_id: Optional[int] = None,
    ) -> None:
        """
        Verifica si existe ya una ubicación equivalente en el mismo almacén (insensible a mayúsculas, tildes, números).
        Lanza ValidationError si detecta duplicidad.
        """
        from inventario.models import Ubicacion

        texto_canonico = normalizar_texto_canonico(nombre)
        
        # Buscar en las ubicaciones del almacén por clave canónica persistida o calculada
        qs = Ubicacion.objects.filter(almacen_id=almacen_id, clave_canonica=texto_canonico)
        if excluir_id:
            qs = qs.exclude(id=excluir_id)

        coincidente = qs.first()
        if coincidente:
            raise ValidationError(
                f"Ya existe la ubicación '{coincidente.nombre}' en este almacén. "
                f"No se permite crear duplicados por diferencias de escritura, mayúsculas, tildes o números ('{nombre}')."
            )

    @classmethod
    def obtener_o_crear_ubicacion(
        cls,
        almacen,
        nombre: str,
        descripcion: str = "",
    ) -> Tuple[Any, bool]:
        """
        Obtiene una ubicación existente que coincida canónicamente o crea una nueva con el nombre normalizado.
        Garantiza idempotencia y previene duplicados.
        """
        from inventario.models import Ubicacion

        if not almacen:
            raise ValidationError("El almacén es obligatorio para registrar una ubicación.")

        nombre_limpio = normalizar_nombre_ubicacion(nombre)
        if not nombre_limpio:
            raise ValidationError("El nombre de la ubicación no puede estar vacío.")

        texto_canonico = normalizar_texto_canonico(nombre_limpio)

        with transaction.atomic():
            # 1. Búsqueda por clave canónica persistida
            existente = Ubicacion.objects.filter(almacen=almacen, clave_canonica=texto_canonico).first()
            if existente:
                return existente, False

            # 2. Respaldo por nombre limpio si clave_canonica aún no estuviera migrada
            existente_nombre = Ubicacion.objects.filter(almacen=almacen, nombre=nombre_limpio).first()
            if existente_nombre:
                if not existente_nombre.clave_canonica:
                    existente_nombre.clave_canonica = texto_canonico
                    existente_nombre.save(update_fields=["clave_canonica"])
                return existente_nombre, False

            # 3. Crear nueva garantizando clave canónica
            nueva = Ubicacion.objects.create(
                almacen=almacen,
                nombre=nombre_limpio,
                clave_canonica=texto_canonico,
                descripcion=descripcion or "",
                activo=True,
                eliminado=False,
            )
            return nueva, True
