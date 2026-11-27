"""Normalización de siglas técnicas, términos hablados y formatos de códigos."""

from __future__ import annotations

import re


class NormalizadorSiglasTecnicas:
    """Convierte transcripciones fonéticas de términos técnicos y códigos a su forma estándar."""

    # 1. Siglas técnicas habladas en telecomunicaciones e informática
    REGLAS_SIGLAS = [
        (r"\b(?:erre\s+jota|re\s+jota|rejota|ra\s*jota|rajota|rj)\s*(?:cuarenta\s+y\s+cinco|45)\b", "RJ45"),
        (r"\b(?:erre\s+jota|re\s+jota|rejota|ra\s*jota|rajota|rj)\s*(?:once|11)\b", "RJ11"),
        (r"\b(?:hache\s+de\s+eme\s+i|hache\s+de\s+mi)\b", "HDMI"),
        (r"\b(?:u\s+te\s+pe)\b", "UTP"),
        (r"\b(?:ef\s+te\s+pe|efe\s+te\s+pe)\b", "FTP"),
        (r"\b(?:ese\s+te\s+pe)\b", "STP"),
        (r"\b(?:ve\s+ge\s+a|uve\s+ge\s+a)\b", "VGA"),
        (r"\b(?:u\s+ese\s+be)\b", "USB"),
        (r"\b(?:cat|categor[ií]a)\s*(?:seis|6)\b", "Cat 6"),
        (r"\b(?:cat|categor[ií]a)\s*(?:cinco|5)(?:\s*(?:e|letra\s+e))?\b", "Cat 5e"),
        (r"\b(?:patch\s*cord|pach\s*cor|pascor)\b", "Patch Cord"),
        (r"\b(?:switch|suich|swich)\b", "Switch"),
        (r"\b(?:router|ruter)\b", "Router"),
        (r"\b(?:dos\s+punto\s+cero|2\s+punto\s+cero|2\s+punto\s+0)\b", "2.0"),
        (r"\b(?:uno\s+punto\s+cuatro|1\s+punto\s+cuatro|1\s+punto\s+4)\b", "1.4"),
        (r"\b(?:de\s+)?(?:ocho|8)\s+puertos\b", "8 Puertos"),
        (r"\b(?:de\s+)?(?:cuatro|4)\s+puertos\b", "4 Puertos"),
        (r"\b(?:de\s+)?(?:diecis[eé]is|16)\s+puertos\b", "16 Puertos"),
        (r"\b(?:de\s+)?(?:veinticuatro|24)\s+puertos\b", "24 Puertos"),
        (r"\b(?:de\s+)?(?:cuarenta\s+y\s+ocho|48)\s+puertos\b", "48 Puertos"),
    ]

    # Pronunciaciones de letras del español. Los prefijos aplicables se derivan
    # en tiempo de ejecución de los códigos presentes en el catálogo.
    NOMBRES_LETRAS = {
        "A": ("a",), "B": ("be", "b"), "C": ("ce", "c"), "D": ("de", "d"),
        "E": ("e",), "F": ("efe", "f"), "G": ("ge", "g"), "H": ("hache", "h"),
        "I": ("i",), "J": ("jota", "j"), "K": ("ka", "k"), "L": ("ele", "l"),
        "M": ("eme", "m"), "N": ("ene", "n"), "O": ("o",), "P": ("pe", "p"),
        "Q": ("cu", "q"), "R": ("erre", "ere", "r"), "S": ("ese", "s"), "T": ("te", "t"),
        "U": ("u",), "V": ("uve", "ve", "v"), "W": ("doble uve", "doble ve", "doble u", "w"),
        "X": ("equis", "x"), "Y": ("ye", "i griega", "y"), "Z": ("zeta", "z"),
    }

    @classmethod
    def _prefijos_catalogo(cls) -> list[str]:
        """Consulta prefijos vigentes, sin mantener una lista fija de familias."""
        try:
            from ..agente.herramientas import obtener_prefijos_catalogo_dinamicos
            return sorted(
                (p for p in obtener_prefijos_catalogo_dinamicos() if 2 <= len(p) <= 5),
                key=lambda p: (-len(p), p),
            )
        except Exception:
            return []

    @classmethod
    def _normalizar_prefijos_hablados(cls, texto: str, prefijos: list[str]) -> str:
        resultado = texto
        for prefijo in prefijos:
            partes = []
            valido = True
            for letra in prefijo:
                alternativas = cls.NOMBRES_LETRAS.get(letra)
                if not alternativas:
                    valido = False
                    break
                partes.append("(?:" + "|".join(re.escape(v) for v in alternativas) + ")")
            if not valido:
                continue
            patron = r"\b" + r"\s+".join(partes) + r"\b"
            resultado = re.sub(patron, prefijo, resultado, flags=re.IGNORECASE)
        return resultado

    # 3. Mapeo de dígitos y números hablados (sin "o" suelta para no corromper disyunciones como "dos o tres")
    DIGITOS_MAP = {
        "cero": "0",
        "uno": "1", "un": "1",
        "dos": "2", "tres": "3", "cuatro": "4",
        "cinco": "5", "seis": "6", "siete": "7",
        "ocho": "8", "nueve": "9",
    }

    DECENAS_COMPUESTAS = {
        "diez": "10", "once": "11", "doce": "12", "trece": "13", "catorce": "14", "quince": "15",
        "dieciseis": "16", "dieciséis": "16", "diecisiete": "17", "dieciocho": "18", "diecinueve": "19",
        "veinte": "20", "veintiuno": "21", "veintiun": "21", "veintiún": "21",
        "veintidos": "22", "veintidós": "22", "veintitres": "23", "veintitrés": "23",
        "veinticuatro": "24", "veinticinco": "25", "veintiseis": "26", "veintiséis": "26",
        "veintisiete": "27", "veintiocho": "28", "veintinueve": "29",
        "treinta": "30", "cuarenta": "40", "cincuenta": "50", "sesenta": "60",
        "setenta": "70", "ochenta": "80", "noventa": "90",
    }

    DECENAS_BASE = {
        "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
        "setenta": 70, "ochenta": 80, "noventa": 90,
    }

    # 4. Medidas comunes en inventario (sin romper formas compuestas como «un metro y medio» / «dos metros y cuarto»)
    _NO_MIXTA = r"(?!\s+y\s+(?:medi[oa]|(?:un\s+|tres\s+)?cuartos?)\b)"
    REGLAS_MEDIDAS = [
        (r"\b(?:un|1)\s+metro\b" + _NO_MIXTA, "1m"),
        (r"\b(?:dos|2)\s+metros\b" + _NO_MIXTA, "2m"),
        (r"\b(?:tres|3)\s+metros\b" + _NO_MIXTA, "3m"),
        (r"\b(?:cinco|5)\s+metros\b" + _NO_MIXTA, "5m"),
        (r"\b(?:diez|10)\s+metros\b" + _NO_MIXTA, "10m"),
    ]

    @classmethod
    def _normalizar_secuencias_numericas(cls, texto: str) -> str:
        """Convierte secuencias habladas de dígitos o compuestos con cero a representación numérica."""
        resultado = texto

        # a. Cero + decena y unidad: ej. 'cero treinta y dos' -> '032'
        patron_decena_y_unidad = re.compile(
            r"\bcero\s+(treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa)\s+y\s+(uno|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)\b",
            re.IGNORECASE,
        )
        def _sub_decena_y_u(m: re.Match) -> str:
            dec = cls.DECENAS_BASE[m.group(1).lower()]
            uni = int(cls.DIGITOS_MAP[m.group(2).lower()])
            return f"0{dec + uni}"
        resultado = patron_decena_y_unidad.sub(_sub_decena_y_u, resultado)

        # b. Cero + número compuesto (10-29 o decena redonda): ej. 'cero quince' -> '015', 'cero veinte' -> '020'
        patron_cero_compuesto = re.compile(
            r"\bcero\s+(" + "|".join(cls.DECENAS_COMPUESTAS.keys()) + r")\b",
            re.IGNORECASE,
        )
        def _sub_cero_comp(m: re.Match) -> str:
            return f"0{cls.DECENAS_COMPUESTAS[m.group(1).lower()]}"
        resultado = patron_cero_compuesto.sub(_sub_cero_comp, resultado)

        # c. Secuencias de dígitos hablados: 4, 3, o 2 dígitos (ej. 'cero tres dos' -> '032', 'cero cero uno' -> '001', 'cero o cero uno' -> '0001')
        dig_words = "|".join(cls.DIGITOS_MAP.keys())
        p4 = re.compile(rf"\b({dig_words})\s+({dig_words})\s+({dig_words})\s+({dig_words})\b", re.IGNORECASE)
        resultado = p4.sub(lambda m: "".join(cls.DIGITOS_MAP[g.lower()] for g in m.groups()), resultado)

        p3 = re.compile(rf"\b({dig_words})\s+({dig_words})\s+({dig_words})\b", re.IGNORECASE)
        resultado = p3.sub(lambda m: "".join(cls.DIGITOS_MAP[g.lower()] for g in m.groups()), resultado)

        p2 = re.compile(rf"\b(cero)\s+({dig_words})\b", re.IGNORECASE)
        resultado = p2.sub(lambda m: f"0{cls.DIGITOS_MAP[m.group(2).lower()]}", resultado)

        return resultado

    @classmethod
    def normalizar(cls, texto: str) -> str:
        """Aplica las transformaciones fonéticas conservando nombres y términos."""
        if not texto:
            return ""

        resultado = texto

        # 0. Limpieza de prefijos de consulta como "código PAP-030" o "código PAP030"
        resultado = re.sub(
            r"\b(?:c[oó]digo|referencia|ref\.?)\s+([A-Za-z]{2,5})[\s\-_]*(\d{1,5})(?!\s*,\s*\d)\b",
            lambda m: f"{m.group(1).upper()}-{m.group(2)}",
            resultado,
            flags=re.IGNORECASE,
        )

        # 0-bis. Variantes de letras y números repetidos hablados (ej. "doble A" -> "AA", "triple A" -> "AAA", "doble cero" -> "00")
        # Antes de la regla general "doble <letra>" -> "<LL>", convertir medidas de potencia dictadas ("60 doble u" / "60 doble ve" -> "60W").
        resultado = re.sub(
            r"(?<![\w-])(\d+(?:[.,]\d+)?)\s*(?:doble\s*(?:u|uve|ve)|dobleu|dobleve)\b",
            r"\1W",
            resultado,
            flags=re.IGNORECASE,
        )
        resultado = re.sub(r"\b(?:cu[aá]druple|cuadruple)\s+([a-zA-Z])\b", lambda m: m.group(1).upper() * 4, resultado, flags=re.IGNORECASE)
        resultado = re.sub(r"\btriple\s+([a-zA-Z])\b", lambda m: m.group(1).upper() * 3, resultado, flags=re.IGNORECASE)
        resultado = re.sub(r"\bdoble\s+([a-zA-Z])\b", lambda m: m.group(1).upper() * 2, resultado, flags=re.IGNORECASE)
        resultado = re.sub(r"\bdoble\s+cero\b", "00", resultado, flags=re.IGNORECASE)
        resultado = re.sub(r"\btriple\s+cero\b", "000", resultado, flags=re.IGNORECASE)

        # 1. Aplicar reglas de siglas
        for patron, reemplazo in cls.REGLAS_SIGLAS:
            resultado = re.sub(patron, reemplazo, resultado, flags=re.IGNORECASE)

        # Convención técnica para revisiones de hardware y producto: «R tres»
        # equivale a R3. La regla general no depende de marcas ni referencias.
        digitos_revision = "|".join(re.escape(p) for p in cls.DIGITOS_MAP)
        resultado = re.sub(
            rf"\br\s+({digitos_revision}|\d)\b",
            lambda m: "R" + cls.DIGITOS_MAP.get(m.group(1).lower(), m.group(1)),
            resultado,
            flags=re.IGNORECASE,
        )

        # 2. Interpretar el deletreo de cualquier prefijo que exista actualmente.
        prefijos_catalogo = cls._prefijos_catalogo()
        resultado = cls._normalizar_prefijos_hablados(resultado, prefijos_catalogo)

        # 3. Aplicar normalización de secuencias numéricas habladas (ej. 'cero tres dos' -> '032')
        resultado = cls._normalizar_secuencias_numericas(resultado)

        # 4. Aplicar reglas de medidas
        for patron, reemplazo in cls.REGLAS_MEDIDAS:
            resultado = re.sub(patron, reemplazo, resultado, flags=re.IGNORECASE)

        # 5. Normalizar códigos usando solo prefijos registrados en el catálogo.
        if prefijos_catalogo:
            patron_prefijo = "(?:" + "|".join(re.escape(p) for p in prefijos_catalogo) + ")"
            resultado = re.sub(
                rf"\b({patron_prefijo})[\s\-]+(\d),\s*(\d{{2,3}})\b",
                r"\1-\2\3",
                resultado,
                flags=re.IGNORECASE,
            )
            marcador_guion = r"(?:guion|gui[oó]n|dion|di[oó]n|raya|rayita|menos)"
            resultado = re.sub(
                rf"\b({patron_prefijo})\s*{marcador_guion}\s*(\d{{1,5}})\b",
                r"\1-\2",
                resultado,
                flags=re.IGNORECASE,
            )
            resultado = re.sub(
                rf"\b({patron_prefijo})[\s\-_]*{marcador_guion}(\d{{1,5}})\b",
                r"\1-\2",
                resultado,
                flags=re.IGNORECASE,
            )
            resultado = re.sub(
                rf"\b({patron_prefijo})[\s,]+(\d{{1,5}})\b",
                r"\1-\2",
                resultado,
                flags=re.IGNORECASE,
            )

        return resultado.strip()
