"""Herramientas de LECTURA del agente HAKU Inventario.

Ejecutan consultas reales sobre el ORM respetando estrictamente:
- Los almacenes accesibles del usuario (``obtener_almacenes_accesibles``).
- La visibilidad de productos inactivos/archivados (solo administradores).
- La inmutabilidad: este módulo jamás escribe en la base de datos.

No interpretan lenguaje natural más allá de la coincidencia léxica de nombres;
la comprensión de la pregunta la realiza el Cerebro Central (planificador).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Any, Dict, List, Optional

from django.db.models import Count, Prefetch, Q

MARGEN_PROXIMIDAD_STOCK = 2  # Margen por defecto para considerar un producto "próximo" al mínimo

CONDICIONES_VALIDAS = {
    "agotado",
    "bajo_minimo",
    "requiere_reposicion",
    "en_minimo",
    "cerca_minimo",
    "suficiente",
    "con_stock",
    "minimo_cero",
    "con_prestamos",
}

ESTADOS_STOCK_PRIMARIOS = (
    "agotado",
    "bajo_minimo",
    "en_minimo",
    "cerca_minimo",
    "suficiente",
)

# Palabras gramaticales o genéricas que no identifican un producto.
_PALABRAS_VACIAS = {
    "de", "del", "la", "las", "el", "los", "y", "o", "para", "con", "en", "un", "una",
    "unos", "unas", "al", "a", "que", "por", "sus", "su", "mis", "tus", "este", "esta",
    "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella", "como", "cual", "cuales",
    "donde", "esta", "estan", "guardado", "guardada", "guardados", "guardadas",
    "ubicado", "ubicada", "ubicados", "ubicadas", "almacenado", "almacenada",
    "producto", "productos", "referencia", "referencias", "articulo", "articulos",
    "item", "items", "cosa", "cosas", "tipo", "tipos", "todas", "todos", "toda", "todo",
    "registrado", "registrados", "registrada", "registradas", "inventario", "catalogo",
    "sistema", "actual", "actuales", "disponible", "disponibles", "existente", "existentes",
    "hay", "tener", "tiene", "tienen", "tendra", "tendran", "habra", "algun", "alguno", "alguna",
    "algunos", "algunas", "ningun", "ninguno", "ninguna", "otro", "otra", "otros", "otras",
    "buen", "bueno", "buena", "buenos", "buenas",
    "muestra", "muestrame", "mostrar", "ver", "dame", "dar", "decir", "dime", "enseña", "enseñame",
    "podria", "podrias", "indicarme", "indicame", "disponibilidad", "detalle", "detalles", "ficha", "informacion", "datos",
    "fue", "fueron", "ha", "han", "sido", "sera", "seran", "diferente", "diferentes", "distinto", "distintos",
}

# Verbos y sustantivos de encuadre funcional que acompañan consultas por uso («qué cables sirven para conectar este equipo»)
# pero no forman parte literal del nombre del producto.
_PALABRAS_FUNCIONALES = {
    "servir", "sirve", "sirven", "usar", "usa", "usan", "utilizar", "utiliza", "utilizan",
    "funcionar", "funciona", "funcionan", "conectar", "conectan", "conexion", "conexiones",
    "equipo", "equipos", "dispositivo", "dispositivos", "aparato", "aparatos", "maquina", "maquinas",
    "elemento", "elementos", "insumo", "insumos", "material", "materiales", "herramienta", "herramientas",
}

# Perífrasis descriptivas o dimensionales que se normalizan a su adjetivo/sustantivo canónico.
_PERIFRASIS_CANONICAS = (
    (r"\b(?:de\s+)?menor\s+(?:tama[nñ]o|medida|dimensi[oó]n)\b", "pequena"),
    (r"\bm[aá]s\s+peque[nñ][oa]s?\b", "pequena"),
    (r"\b(?:chiquit[oa]s?|chic[oa]s?)\b", "pequena"),
    (r"\btama[nñ]o\s+peque[nñ][oa]\b", "pequena"),
    (r"\b(?:de\s+)?mayor\s+(?:tama[nñ]o|medida|dimensi[oó]n)\b", "grande"),
    (r"\bm[aá]s\s+grandes?\b", "grande"),
    (r"\btama[nñ]o\s+grande\b", "grande"),
    (r"\btama[nñ]o\s+median[oa]\b", "mediana"),
    (r"\b(?:grapitas?|engrapaditas?)\b", "grapas"),
    (r"\bganchos?\s+de\s+(?:cosedora|grapadora|engrapadora)\b", "grapas"),
    (r"\bbarras?\s+de\s+(?:pegante|pegamento|colb[oó]n|goma)\b", "pegante barra"),
    (r"\b(?:bater[ií]as?|pilas?)\s+doble\s+[aA]\b", "baterias aa"),
    (r"\b(?:bater[ií]as?|pilas?)\s+triple\s+[aA]\b", "baterias aaa"),
)

# Grupos de sinónimos léxicos verdaderos (equivalencia de producto en español / uso universitario colombiano).
# NUNCA incluye productos distintos pero relacionados (ej. grapadora != grapa).
_GRUPOS_SINONIMOS = (
    {"bateria", "pila"},
    {"grapadora", "cosedora", "engrapadora", "abrochadora"},
    {"pegante", "pegamento", "colbon", "adhesivo", "goma"},
    {"boligrafo", "esfero", "lapicero", "pluma"},
    {"marcador", "plumon", "rotulador"},
    {"resaltador", "destacador", "marcatexto"},
    {"carpeta", "folder", "legajador", "archivador"},
    {"libreta", "cuaderno", "bloc", "agenda"},
    {"sacapuntas", "tajapiz", "tajalapiz", "afilalapiz"},
    {"regleta", "multitoma", "extension"},
    {"computador", "computadora", "ordenador", "pc", "portatil", "laptop"},
    {"pantalla", "monitor"},
    {"proyector", "videobeam"},
    {"destornillador", "atornillador"},
    {"alicate", "pinza", "tenaza"},
    {"exacto", "bisturi", "cuter"},
    {"pequena", "pequeno", "chica", "chico", "menor"},
    {"grande", "mayor"},
    {"conectar", "conexion"},
    {"switch", "suich", "conmutador"},
)

# Relaciones conceptuales/funcionales entre productos distintos (herramienta <-> insumo).
# Solo se usan para sugerir alternativas relacionadas cuando el producto buscado NO existe (total == 0).
_RELACIONES_CONCEPTUALES: Dict[str, List[str]] = {
    "grapadora": ["grapa"],
    "cosedora": ["grapa"],
    "engrapadora": ["grapa"],
    "abrochadora": ["grapa"],
    "grapa": ["grapadora", "cosedora", "engrapadora"],
    "impresora": ["toner", "cartucho", "tinta", "resma", "papel"],
    "toner": ["impresora"],
    "cartucho": ["impresora"],
    "taladro": ["broca", "chazo"],
    "broca": ["taladro"],
    "tablero": ["marcador", "borrador"],
}

# Mapeo de conceptos funcionales a familias de productos para preguntas puramente funcionales.
_DESCRIPCIONES_FUNCIONALES: Dict[str, List[str]] = {
    "conectar": ["cable", "utp", "hdmi", "vga", "adaptador", "conector", "rj45", "switch", "router"],
    "conexion": ["cable", "utp", "hdmi", "vga", "adaptador", "conector", "rj45", "switch", "router"],
    "red": ["cable", "utp", "rj45", "switch", "router", "patch"],
    "pegar": ["pegante", "pegamento", "colbon", "cinta", "adhesivo", "silicona", "goma"],
    "adherir": ["pegante", "pegamento", "colbon", "cinta", "adhesivo", "silicona", "goma"],
    "grapar": ["grapadora", "cosedora", "grapa"],
    "coser": ["grapadora", "cosedora", "grapa"],
    "engrapar": ["grapadora", "cosedora", "grapa"],
    "perforar": ["perforadora", "taladro", "broca"],
    "borrar": ["borrador"],
    "cortar": ["tijera", "exacto", "bisturi", "cuter"],
    "escribir": ["lapiz", "lapicero", "boligrafo", "esfero", "marcador", "tiza"],
    "pintar": ["tempera", "pintura", "vinilo", "pincel"],
    "limpiar": ["alcohol", "cloruro", "clinex"],
    "proteger": ["guante", "tapaboca", "mascara", "careta", "gafa"],
    "corregir": ["corrector", "liquid", "borrador"],
    "archivar": ["carpeta", "folder", "legajador", "archivador"],
}

_UNIDADES_HABLADAS = {
    "cero": 0, "un": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}
_DECENAS_HABLADAS = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90,
}

# Unidades físicas comunes: se usan solo para comparar cantidades equivalentes
# dentro de una búsqueda; nunca se derivan nombres ni valores del catálogo.
_ALIAS_UNIDAD_MEDIDA = {
    "pulgada": r"pulgadas?|pulg\.?|inch(?:es)?|in",
    "centimetro": r"cent[ií]\s*metros?|centimeters?|centimetres?|cm",
    "milimetro_cuadrado": r"mil[ií]\s*metros?\s+(?:al\s+)?cuadrados?|mm[2²]|mm\s+al\s+cuadrado|mm\s+2(?!\s+(?:unidades?|piezas?|art[ií]culos?|cajas?|paquetes?|bolsas?|rollos?)\b)",
    "milimetro": r"mil[ií]\s*metros?|millimeters?|millimetres?|mm",
    "micrometro": r"micr[oó]\s*metros?|micras?|[μµu]m",
    "kilometro": r"kil[oó]\s*metros?|kilometers?|kilometres?|km",
    "metro": r"metros?|meters?|metres?|m",
    "pie": r"pies|feet|foot|ft",
    "kilogramo": r"kilo\s*gramos?|kilograms?|kg",
    "miligramo": r"mili\s*gramos?|milligrams?|mg",
    "gramaje": r"gramos?\s+(?:por\s+)?metros?\s+cuadrados?|g/m[2²]|gr/m[2²]|gsm",
    "gramo": r"gramos?|grams?|g",
    "onza": r"onzas?|oz",
    "libra": r"libras?|lbs?",
    "galon": r"galones?|gal[oó]n|gal",
    "mililitro": r"mili\s*litros?|milliliters?|millilitres?|ml",
    "litro": r"litros?|liters?|litres?|l",
    "kilovoltio": r"kilo\s*voltios?|kilovolts?|kv",
    "milivoltio": r"mili\s*voltios?|millivolts?|mv",
    "voltio": r"voltios?|volts?|v",
    "kilovatio": r"kilo\s*vatios?|kilo\s*watts?|kw",
    "milivatio": r"mili\s*vatios?|mili\s*watts?|mw",
    "vatio": r"vatios?|watts?|doble\s*(?:u|uve|ve)|dobleu|dobleve|w",
    "gigahercio": r"giga\s*hercios?|giga\s*hertz|ghz",
    "megahercio": r"mega\s*hercios?|mega\s*hertz|mhz",
    "kilohercio": r"kilo\s*hercios?|kilo\s*hertz|khz",
    "hercio": r"hercios?|hertz|hz",
    "miliamperio": r"mili\s*amperios?|milli\s*amperes?|ma",
    "amperio": r"amperios?|amperes?|amps?",
    "megaohmio": r"(?:mega|m)\s*ohmios?|(?:mega|meg|m)\s*ohms?|m[ωΩ]",
    "kiloohmio": r"(?:kilo|k)\s*ohmios?|(?:kilo|kil|k)\s*ohms?|k[ωΩ]",
    "ohmio": r"ohmios?|ohms?|[ωΩ]",
    "microfaradio": r"micro\s*faradios?|[μµu]f",
    "nanofaradio": r"nano\s*faradios?|nf",
    "picofaradio": r"pico\s*faradios?|pf",
    "decibelio": r"decibelios?|decibeles?|db",
    "bar": r"bares?|bar",
    "psi": r"psi",
    "kilopascal": r"kilo\s*pascales?|kpa",
    "calibre": r"calibre|awg",
    "porciento": r"por\s+ciento|porciento|%",
    "terabyte": r"tera\s*bytes?|teras?|tb",
    "gigabyte": r"giga\s*bytes?|gigas?|gb",
    "megabyte": r"mega\s*bytes?|megas?|mb",
    "rpm": r"revoluciones(?:\s+por\s+minuto)?|rpm",
}
_SIMBOLOS_UNIDAD_EXACTOS = {
    "%": "porciento",
    "ω": "ohmio",
    "kω": "kiloohmio",
    "mω": "megaohmio",
    "μf": "microfaradio",
    "µf": "microfaradio",
    "uf": "microfaradio",
    "μm": "micrometro",
    "µm": "micrometro",
    "um": "micrometro",
    "mm2": "milimetro_cuadrado",
    "mm²": "milimetro_cuadrado",
    "g/m2": "gramaje",
    "g/m²": "gramaje",
    "gr/m2": "gramaje",
    "gr/m²": "gramaje",
    "gsm": "gramaje",
    '"': "pulgada",
    "″": "pulgada",
    "''": "pulgada",
}
# Unidades en las que "A/B <unidad>" denota un rango o especificación dual (ej. 50/60 Hz, 110/220V),
# jamás una fracción matemática reducible.
_UNIDADES_SIN_FRACCION_BARRA = frozenset({
    "voltio", "kilovoltio", "milivoltio",
    "hercio", "kilohercio", "megahercio", "gigahercio",
    "amperio", "miliamperio",
    "vatio", "kilovatio", "milivatio",
    "ohmio", "kiloohmio", "megaohmio",
    "microfaradio", "nanofaradio", "picofaradio",
    "decibelio", "bar", "psi", "kilopascal",
    "calibre", "porciento", "gramaje", "rpm",
    "terabyte", "gigabyte", "megabyte",
})
_GLIFOS_FRACCIONARIOS = {
    "½": Fraction(1, 2), "⅓": Fraction(1, 3), "⅔": Fraction(2, 3),
    "¼": Fraction(1, 4), "¾": Fraction(3, 4), "⅛": Fraction(1, 8),
    "⅜": Fraction(3, 8), "⅝": Fraction(5, 8), "⅞": Fraction(7, 8),
}
_DENOMINADORES_HABLADOS = {
    "medio": 2, "medios": 2, "media": 2, "medias": 2,
    "tercio": 3, "tercios": 3, "tercia": 3, "tercias": 3,
    "cuarto": 4, "cuartos": 4, "cuarta": 4, "cuartas": 4,
    "quinto": 5, "quintos": 5, "quinta": 5, "quintas": 5,
    "sexto": 6, "sextos": 6, "sexta": 6, "sextas": 6,
    "octavo": 8, "octavos": 8, "octava": 8, "octavas": 8,
    "decimo": 10, "decimos": 10, "decima": 10, "decimas": 10,
    "dieciseisavo": 16, "dieciseisavos": 16, "dieciseisava": 16, "dieciseisavas": 16,
    "treintaidosavo": 32, "treintaidosavos": 32, "treintaidosava": 32, "treintaidosavas": 32,
    "sesentaicuatroavo": 64, "sesentaicuatroavos": 64,
}
_DENOMINADORES_FRACCION_SUELTA = frozenset({2, 3, 4, 8, 16, 32, 64})


_DENOMINADORES_PARTIDOS_REGEX = (
    r"diecis[eé]is\s+avos?|16\s*avos?|"
    r"treinta\s+y\s+dos\s+avos?|treintaidos\s+avos?|32\s*avos?|"
    r"sesenta\s+y\s+cuatro\s+avos?|sesentaicuatro\s+avos?|64\s*avos?"
)


def _crear_patron_cantidad_unidad() -> re.Pattern:
    unidades = "|".join(f"(?:{patron})" for patron in _ALIAS_UNIDAD_MEDIDA.values())
    cardinales = sorted(
        set(_UNIDADES_HABLADAS) | set(_DECENAS_HABLADAS) | {"una"},
        key=len,
        reverse=True,
    )
    numeradores = "|".join(re.escape(palabra) for palabra in cardinales)
    decenas_comp = "|".join(re.escape(p) for p in sorted(_DECENAS_HABLADAS, key=len, reverse=True))
    unidades_comp = "|".join(
        re.escape(p)
        for p in ("uno", "una", "un", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve")
    )
    compuesto_hablado = rf"(?:{decenas_comp})\s+y\s+(?:{unidades_comp})"
    denominadores = "|".join(
        re.escape(palabra)
        for palabra in sorted(_DENOMINADORES_HABLADOS, key=len, reverse=True)
    )
    denominadores_ext = rf"(?:{_DENOMINADORES_PARTIDOS_REGEX}|{denominadores})"
    numero_mixto = r"(?:\d+\s+\d+\s*/\s*\d+|\d+\s*[½⅓⅔¼¾⅛⅜⅝⅞])"
    numero = rf"(?:{numero_mixto}|\d+\s*/\s*\d+|\d+(?:[.,]\d+)?|[½⅓⅔¼¾⅛⅜⅝⅞])"
    decimal_hablado = (
        rf"(?:\d+|{compuesto_hablado}|{numeradores})\s+(?:punto|coma)\s+"
        rf"(?:(?:cero\s+)*)(?:\d+|{compuesto_hablado}|{numeradores})"
    )
    decimal_sin_entero = (
        rf"(?:punto|coma)\s+(?:(?:cero\s+)*)(?:\d+|{compuesto_hablado}|{numeradores})"
    )
    fraccion_sobre = rf"(?:{numeradores}|\d+)\s+(?:sobre|barra|entre)\s+(?:{compuesto_hablado}|{numeradores}|\d+)"
    mixto_hablado = rf"(?:{numeradores}|una)\s+y\s+(?:medi[oa]|(?:un\s+)?cuarto|tres\s+cuartos)"
    fraccion_hablada = rf"(?:(?:{numeradores}|\d+)\s+)?(?:{denominadores_ext})"
    cantidad = (
        rf"(?:{decimal_hablado}|{decimal_sin_entero}|{numero}|{mixto_hablado}|"
        rf"{fraccion_sobre}|{fraccion_hablada}|{compuesto_hablado}|{numeradores}|una)"
    )
    # Exigir espacio tras palabras habladas (evita falsos positivos como «medio del» -> «de» + «l»/litro);
    # solo se permite unidad pegada sin espacio tras dígito o glifo fraccionario («12V», «4.7kΩ», «½in»).
    return re.compile(
        rf"(?<![\w-])(?P<cantidad>{cantidad})(?:(?:\s+de)?\s+|(?<=\d|[½⅓⅔¼¾⅛⅜⅝⅞]))(?P<unidad>{unidades})(?!\w)",
        flags=re.IGNORECASE,
    )


_PATRON_CANTIDAD_UNIDAD = _crear_patron_cantidad_unidad()
# Símbolo de pulgadas pegado a número o fracción (ej. 1/2", 3/8", 1 1/2", 2", 3/4″, 1/2'')
_PATRON_PULGADA_SIMBOLO = re.compile(
    r'(?<![\w-])(?P<cantidad>\d+\s+\d+\s*/\s*\d+|\d+\s*[½⅓⅔¼¾⅛⅜⅝⅞]|\d+\s*/\s*\d+|\d+(?:[.,]\d+)?|[½⅓⅔¼¾⅛⅜⅝⅞])\s*(?:"|″|\'\')(?!\w)'
)
# Forma española "[entero] <unidad> y media/medio/cuarto/tres cuartos"
# (ej. "pulgada y media", "pulgada y medio", "un metro y media", "dos pulgadas y media", "2 metros y cuarto")
_PATRON_UNIDAD_Y_MEDIA = re.compile(
    r"(?<![\w-])(?:(?P<entero>\d+|una|uno|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|doce|quince|veinte)\s+)?"
    r"(?P<unidad>pulgadas?|metros?|cent[ií]\s*metros?|mil[ií]\s*metros?|litros?|kilos?|kilo\s*gramos?|libras?|pies?)"
    r"\s+y\s+(?P<frac>medi[oa]|(?:un\s+)?cuarto|tres\s+cuartos)\b",
    flags=re.IGNORECASE,
)
# Fracción mecánica suelta sin unidad explícita (ej. "Tornillo 1/2", "Broca 5/16", "Llave 1 1/2", "½")
# Excluye fechas (26/11/2026, 12/05) restringiendo a denominadores mecánicos estándar sin cero inicial.
_PATRON_FRACCION_SUELTA = re.compile(
    r"(?<![\w/-])(?P<cantidad>\d+\s+\d+\s*/\s*[1-9]\d*|\d+\s*[½⅓⅔¼¾⅛⅜⅝⅞]|\d+\s*/\s*[1-9]\d*|[½⅓⅔¼¾⅛⅜⅝⅞])(?![/\w])"
)
# Fracción hablada en español sin unidad explícita:
# - "de media" / "de medio" / "de un medio" exige "de" previo y excluye locuciones temporales/espaciales.
# - Fracciones mecánicas explícitas ("tres cuartos", "cinco dieciséis avos", "5 16 avos", "5 sobre 16",
#   "tres treinta y dos avos", etc.) funcionan con o sin "de".
_NUM_FRAC_SUELTA = (
    r"(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|\d+)"
)
_PATRON_FRACCION_HABLADA_SIN_UNIDAD = re.compile(
    r"(?:"
    r"(?<=\bde\s)(?P<cant_media>un\s+medio|medi[oa])"
    r"(?!\s*(?:horas?|docenas?|tardes?|ma[ñn]anas?|noches?|vidas?|lunas?|vueltas?|d[ií]as?|semanas?|meses?|a[ñn]os?|"
    r"ambiente|tiempo|cuerpo|camino|lado|paso|piso|estante|almac[eé]n|bodega|del?|al?)\b)"
    r"|"
    r"(?<![\w-])(?P<cant_frac>"
    r"(?:un|tres|1|3)\s+cuartos?|"
    r"(?:un|tres|cinco|siete|1|3|5|7)\s+octavos?|"
    rf"{_NUM_FRAC_SUELTA}\s+(?:dieciseisavos?|diecis[eé]is\s+avos?|16\s*avos?)|"
    rf"{_NUM_FRAC_SUELTA}\s+(?:treintaidosavos?|treinta\s+y\s+dos\s+avos?|treintaidos\s+avos?|32\s*avos?)|"
    rf"{_NUM_FRAC_SUELTA}\s+(?:sesentaicuatroavos?|sesenta\s+y\s+cuatro\s+avos?|sesentaicuatro\s+avos?|64\s*avos?)|"
    rf"(?<=\bde\s){_NUM_FRAC_SUELTA}\s+(?:sobre|barra|entre)\s+(?:dos|tres|cuatro|ocho|diecis[eé]is|treinta\s+y\s+dos|sesenta\s+y\s+cuatro|2|3|4|8|16|32|64)"
    r")"
    r"(?!\s*(?:horas?|docenas?|tardes?|ma[ñn]anas?|noches?|vidas?|lunas?|vueltas?|d[ií]as?|semanas?|meses?|a[ñn]os?)\b)"
    r")(?=$|[\s,.;:?!])",
    flags=re.IGNORECASE,
)
# Amperios abreviados como «A»: solo pegados al número («5A», «2.5a») o «A» mayúscula aislada («5 A»),
# para jamás confundir la preposición española «a» en frases como «prestar 2 a Juan» o «de 1 a 5».
_PATRON_AMPERIO_CORTO = re.compile(
    r"(?<![\w-])(?P<cantidad>\d+(?:[.,]\d+)?)(?:(?P<pegada>[aA])|\s+(?P<mayus>A)(?![a-záéíóúñü]))(?!\w)"
)
# Tallas estándar (S, M, L, XS, XL, XXL, XXXL), incluyendo deletreo Whisper
# («talla eme», «talla ele», «talla ese», «talla equis ele», «talla x l», «talla doble xl»)
# y descriptores en español tras la palabra «talla» («talla pequeña», «talla mediana», «talla grande»).
_REGLAS_TALLA_HABLADA = (
    (r"\btallas?\s+(?:triple\s+(?:equis|x)\s*(?:ele|l)|(?:xxx|x\s+x\s+x)\s*(?:ele|l)|xxxl)\b", "tallaxxxl0"),
    (r"\btallas?\s+(?:(?:doble\s+(?:equis|x)|xx|x\s+x)\s*(?:ele|l)|doble\s+xl|xxl)\b", "tallaxxl0"),
    (r"\btallas?\s+(?:(?:equis|x)\s*(?:ele|l)|extra\s*grande|xl)\b", "tallaxl0"),
    (r"\btallas?\s+(?:(?:equis|x)\s*(?:ese|s)|extra\s*peque[nñ][oa]|xs)\b", "tallaxs0"),
    (r"\btallas?\s+(?:ese|peque[nñ][oa]|s)\b", "tallas0"),
    (r"\btallas?\s+(?:eme|median[oa]|m)\b", "tallam0"),
    (r"\btallas?\s+(?:ele|grande|l)\b", "tallal0"),
)
_PATRONES_TALLA_HABLADA = tuple(
    (re.compile(patron, flags=re.IGNORECASE), canon)
    for patron, canon in _REGLAS_TALLA_HABLADA
)


def _valor_numero_hablado(texto: str) -> Optional[int]:
    """Convierte un número cardinal español aislado o compuesto (0-99) a entero."""
    palabra = normalizar(texto).strip()
    if not palabra:
        return None
    if palabra.isdigit():
        return int(palabra)
    if palabra == "una":
        return 1
    if palabra in _UNIDADES_HABLADAS:
        return _UNIDADES_HABLADAS[palabra]
    if palabra in _DECENAS_HABLADAS:
        return _DECENAS_HABLADAS[palabra]
    partes = palabra.split()
    if len(partes) == 3 and partes[1] == "y" and partes[0] in _DECENAS_HABLADAS:
        uni = 1 if partes[2] == "una" else _UNIDADES_HABLADAS.get(partes[2])
        if uni is not None and 1 <= uni <= 9:
            return _DECENAS_HABLADAS[partes[0]] + uni
    return None


def _cantidad_medida_canonica(cantidad: str) -> Optional[str]:
    """Representa fracciones, números mixtos, decimales y cardinales con una fracción reducida común."""
    valor_original = str(cantidad or "").strip().lower()
    valor = normalizar(valor_original).strip()
    # Unificar denominadores compuestos partidos por Whisper («dieciseis avos», «16 avos», «treinta y dos avos», «32 avos»)
    valor = re.sub(r"\b(?:dieciseis\s+avos?|16\s*avos?)\b", "dieciseisavos", valor)
    valor = re.sub(r"\b(?:treinta\s+y\s+dos\s+avos?|treintaidos\s+avos?|32\s*avos?)\b", "treintaidosavos", valor)
    valor = re.sub(r"\b(?:sesenta\s+y\s+cuatro\s+avos?|sesentaicuatro\s+avos?|64\s*avos?)\b", "sesentaicuatroavos", valor)
    m_mixto_num = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", valor_original)
    m_mixto_glifo = re.fullmatch(r"(\d+)\s*([½⅓⅔¼¾⅛⅜⅝⅞])", valor_original)
    m_dec_hablado = re.fullmatch(r"(.+?)\s+(?:punto|coma)\s+(.+)", valor)
    m_dec_sin_entero = re.fullmatch(r"(?:punto|coma)\s+(.+)", valor)
    m_sobre = re.fullmatch(r"(.+?)\s+(?:sobre|barra|entre)\s+(.+)", valor)
    if m_mixto_num:
        ent, num, den = int(m_mixto_num.group(1)), int(m_mixto_num.group(2)), int(m_mixto_num.group(3))
        if den == 0:
            return None
        fraccion = Fraction(ent, 1) + Fraction(num, den)
    elif m_mixto_glifo:
        ent = int(m_mixto_glifo.group(1))
        fraccion = Fraction(ent, 1) + _GLIFOS_FRACCIONARIOS[m_mixto_glifo.group(2)]
    elif valor_original in _GLIFOS_FRACCIONARIOS:
        fraccion = _GLIFOS_FRACCIONARIOS[valor_original]
    elif re.fullmatch(r"\d+\s*/\s*\d+", valor_original):
        numerador, denominador = re.split(r"\s*/\s*", valor_original, maxsplit=1)
        if int(denominador) == 0:
            return None
        fraccion = Fraction(int(numerador), int(denominador))
    elif re.fullmatch(r"\d+(?:[.,]\d+)?", valor_original):
        fraccion = Fraction(valor_original.replace(",", "."))
    elif m_dec_hablado or m_dec_sin_entero:
        ent_val = 0 if m_dec_sin_entero else _valor_numero_hablado(m_dec_hablado.group(1))
        dec_raw = m_dec_sin_entero.group(1) if m_dec_sin_entero else m_dec_hablado.group(2)
        dec_palabras = dec_raw.split()
        ceros_prefijo = 0
        while len(dec_palabras) > 1 and dec_palabras[0] == "cero":
            ceros_prefijo += 1
            dec_palabras.pop(0)
        dec_resto = " ".join(dec_palabras)
        if dec_resto.isdigit():
            dec_str = ("0" * ceros_prefijo) + dec_resto
        else:
            dec_val = _valor_numero_hablado(dec_resto)
            if dec_val is None:
                return None
            dec_str = ("0" * ceros_prefijo) + str(dec_val)
        if ent_val is None or not dec_str:
            return None
        fraccion = Fraction(f"{ent_val}.{dec_str}")
    elif m_sobre:
        num_s = _valor_numero_hablado(m_sobre.group(1))
        den_s = _valor_numero_hablado(m_sobre.group(2))
        if num_s is None or den_s is None or den_s == 0:
            return None
        fraccion = Fraction(num_s, den_s)
    else:
        palabras = valor.split()
        if len(palabras) >= 3 and palabras[1] == "y" and palabras[2] in ("medio", "media", "cuarto"):
            ent_h = _valor_numero_hablado(palabras[0])
            if ent_h is None:
                return None
            fraccion = Fraction(ent_h, 1) + Fraction(1, 2 if palabras[2].startswith("medi") else 4)
        elif len(palabras) == 4 and palabras[1] == "y" and palabras[2] in ("un", "tres") and palabras[3] in ("cuarto", "cuartos"):
            ent_h = _valor_numero_hablado(palabras[0])
            if ent_h is None:
                return None
            fraccion = Fraction(ent_h, 1) + Fraction(1 if palabras[2] == "un" else 3, 4)
        elif len(palabras) == 3 and palabras[1] == "y" and palabras[0] in _DECENAS_HABLADAS:
            uni = 1 if palabras[2] == "una" else _valor_numero_hablado(palabras[2])
            if uni is None or not (1 <= uni <= 9):
                return None
            fraccion = Fraction(_DECENAS_HABLADAS[palabras[0]] + uni, 1)
        elif len(palabras) == 1 and palabras[0] in _DENOMINADORES_HABLADOS:
            fraccion = Fraction(1, _DENOMINADORES_HABLADOS[palabras[0]])
        elif len(palabras) == 2 and palabras[1] in _DENOMINADORES_HABLADOS:
            numerador = _valor_numero_hablado(palabras[0])
            if numerador is None:
                return None
            fraccion = Fraction(numerador, _DENOMINADORES_HABLADOS[palabras[1]])
        elif len(palabras) == 1:
            entero = 1 if palabras[0] == "una" else _valor_numero_hablado(palabras[0])
            if entero is None:
                return None
            fraccion = Fraction(entero, 1)
        else:
            return None
    return f"q{fraccion.numerator}d{fraccion.denominator}"


def _normalizar_cantidades_con_unidad(texto: str) -> str:
    """Unifica notaciones equivalentes de cantidad+unidad sin conocer productos del catálogo.

    Por ejemplo, ``media pulgada``, ``1/2 pulgada``, ``1/2"`` y ``0,5 in`` comparten
    la misma representación (``q1d2 pulgada``), y ``Tornillo 1/2`` produce ``q1d2``.
    Solo transforma cantidades junto a una unidad o fracciones mecánicas estándar,
    por lo que no confunde códigos ni fechas con dimensiones.
    """
    if not texto:
        return ""

    def _unidad_canonica(unidad_bruta: str) -> str:
        u_low = unidad_bruta.strip().lower()
        exacta = _SIMBOLOS_UNIDAD_EXACTOS.get(u_low)
        if exacta:
            return exacta
        unidad_original = normalizar(u_low)
        return next(
            (
                unidad
                for unidad, patron_alias in _ALIAS_UNIDAD_MEDIDA.items()
                if re.fullmatch(patron_alias, unidad_original, flags=re.IGNORECASE)
            ),
            unidad_original,
        )

    def reemplazar(m: re.Match) -> str:
        cant_raw = m.group("cantidad")
        unidad_canon = _unidad_canonica(m.group("unidad"))
        m_barra = re.fullmatch(r"(\d+)\s*/\s*(\d+)", cant_raw.strip())
        if m_barra:
            den_barra = int(m_barra.group(2))
            if unidad_canon in _UNIDADES_SIN_FRACCION_BARRA or den_barra not in _DENOMINADORES_FRACCION_SUELTA:
                return m.group(0)
        canon = _cantidad_medida_canonica(cant_raw)
        if not canon:
            return m.group(0)
        return f"{canon} {unidad_canon}"

    def reemplazar_pulgada(m: re.Match) -> str:
        canon = _cantidad_medida_canonica(m.group("cantidad"))
        if not canon:
            return m.group(0)
        return f"{canon} pulgada"

    def reemplazar_unidad_y_media(m: re.Match) -> str:
        ent_raw = m.group("entero")
        ent_val = _valor_numero_hablado(ent_raw) if ent_raw else 1
        if ent_val is None:
            return m.group(0)
        frac_norm = normalizar(m.group("frac"))
        if frac_norm.startswith("medi"):
            delta = Fraction(1, 2)
        elif frac_norm.startswith("tres"):
            delta = Fraction(3, 4)
        else:
            delta = Fraction(1, 4)
        total = Fraction(ent_val, 1) + delta
        u_raw = m.group("unidad").strip().lower()
        u_canon = "kilogramo" if u_raw.startswith("kilo") else _unidad_canonica(u_raw)
        return f"q{total.numerator}d{total.denominator} {u_canon}"

    def reemplazar_fraccion_suelta(m: re.Match) -> str:
        raw = m.group("cantidad").strip()
        if raw in _GLIFOS_FRACCIONARIOS or any(g in raw for g in _GLIFOS_FRACCIONARIOS):
            canon = _cantidad_medida_canonica(raw)
            return canon or m.group(0)
        m_mix = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", raw)
        if m_mix:
            num, den = int(m_mix.group(2)), int(m_mix.group(3))
            if den in _DENOMINADORES_FRACCION_SUELTA and 0 < num < den:
                canon = _cantidad_medida_canonica(raw)
                return canon or m.group(0)
            return m.group(0)
        m_f = re.fullmatch(r"(\d+)\s*/\s*(\d+)", raw)
        if m_f:
            num, den = int(m_f.group(1)), int(m_f.group(2))
            if den in _DENOMINADORES_FRACCION_SUELTA and 0 < num < den:
                canon = _cantidad_medida_canonica(raw)
                return canon or m.group(0)
        return m.group(0)

    def reemplazar_amperio(m: re.Match) -> str:
        canon = _cantidad_medida_canonica(m.group("cantidad"))
        if not canon:
            return m.group(0)
        return f"{canon} amperio"

    res = _PATRON_PULGADA_SIMBOLO.sub(reemplazar_pulgada, texto)
    res = _PATRON_UNIDAD_Y_MEDIA.sub(reemplazar_unidad_y_media, res)
    res = _PATRON_CANTIDAD_UNIDAD.sub(reemplazar, res)
    res = _PATRON_AMPERIO_CORTO.sub(reemplazar_amperio, res)
    res = _PATRON_FRACCION_SUELTA.sub(reemplazar_fraccion_suelta, res)
    res = _PATRON_FRACCION_HABLADA_SIN_UNIDAD.sub(
        lambda m: _cantidad_medida_canonica(m.group("cant_media") or m.group("cant_frac") or "") or m.group(0),
        res,
    )
    # Termina en dígito ("0") para que la lematización (_raiz) nunca recorte la "s" de talla S / XS.
    for patron_talla, canon_talla in _PATRONES_TALLA_HABLADA:
        res = patron_talla.sub(canon_talla, res)
    return res

# Prefijos que representan estándares técnicos o palabras gramaticales que preceden a "cero" / números
# y nunca constituyen un código de producto interno por sí mismos.
_NO_PREFIJOS_CODIGO = {
    "RJ", "CAT", "USB", "ISO", "UTF", "NOM", "LED", "LCD", "VGA", "HDMI", "UTP", "WIFI", "N",
    "CON", "MAS", "POR", "SIN", "DEL", "LAS", "LOS", "SON", "HAY", "UNA", "UNO", "DOS", "TRES",
    "ESTA", "ESTAN", "QUE", "CUAL", "COMO", "DONDE", "TODO", "TODA", "TODOS", "TODAS",
    "TIENE", "TIENEN", "QUEDA", "QUEDAN", "ESTE", "ESTOS", "ESTAS", "ESE", "ESOS", "ESAS",
    "AQUEL", "AQUELLOS", "AQUELLAS", "EN", "DE", "AL", "EL", "LA", "ES", "UN", "SE", "SI", "NO", "YA",
    "TOP", "STOCK", "MINIMO", "IGUAL", "VALOR", "NIVEL", "TOPE", "CASI", "DESDE", "HASTA", "ENTRE", "PARA",
    "BUSCA", "DAME", "SACA", "HAZ", "HACER", "PIDE", "PON", "PONER", "LLEVA", "TRAE", "MUEVE", "PRESTA",
    "FALTA", "FALTAN", "SOLO", "CADA", "OTRO", "OTRA", "OTROS", "OTRAS", "UNOS", "UNAS", "ERAN",
    "DION", "GUION", "GION", "BION", "YON", "RAYA", "MENOS", "CODE", "REGION", "LEGION",
}


def _normalizar_numeros_codigo_hablado(texto: str) -> str:
    """Convierte sufijos numéricos hablados tras un prefijo de código en dígitos (ej. 'PAP cero treinta' -> 'PAP-030').

    Regla estructural para evitar convertir lenguaje ordinario ('Busca un', 'Dame dos', 'Saca diez') en códigos:
    Solo se considera código hablado cuando:
      1. La secuencia hablada comienza explícitamente con 'cero' (ej. 'pap cero treinta', 'bat cero cero uno'), O
      2. El prefijo viene precedido por marcador explícito ('código', 'referencia', 'ref'), O
      3. El prefijo está en MAYÚSCULAS sostenidas (ej. 'PAP treinta' tras normalizar 'pe a pe') en un texto que no
         está íntegramente en mayúsculas y el número hablado no es el artículo indefinido ('un'/'una').
    """
    if not texto:
        return ""
    vocab_num = "|".join(list(_UNIDADES_HABLADAS.keys()) + list(_DECENAS_HABLADAS.keys()))
    patron = re.compile(
        rf"\b([A-Za-z]{{2,5}})\s+((?:{vocab_num})(?:\s+(?:y\s+)?(?:{vocab_num}))*)\b",
        flags=re.IGNORECASE,
    )
    letras_totales = [c for c in texto if c.isalpha()]
    texto_todo_mayus = bool(letras_totales) and all(c.isupper() for c in letras_totales)
    prefijos_vigentes = obtener_prefijos_catalogo_dinamicos()

    def _reemplazar(m: re.Match) -> str:
        pref_raw = m.group(1)
        pref = pref_raw.upper()
        if pref in _NO_PREFIJOS_CODIGO and pref not in prefijos_vigentes:
            return m.group(0)
        bloque = unicodedata.normalize("NFKD", m.group(2).lower())
        bloque = "".join(c for c in bloque if not unicodedata.combining(c))
        palabras = [w for w in bloque.split() if w]
        if not palabras:
            return m.group(0)

        prefijo_previo = texto[: m.start()].rstrip().lower()
        tiene_marcador_codigo = bool(re.search(r"\b(?:c[oó]digo|ref(?:erencia)?)$", prefijo_previo))
        empieza_con_cero = palabras[0] == "cero"
        es_sigla_mayus = pref_raw.isupper() and not texto_todo_mayus and palabras not in (["un"], ["una"])

        if not (empieza_con_cero or tiene_marcador_codigo or es_sigla_mayus or pref in prefijos_vigentes):
            return m.group(0)

        digitos: List[str] = []
        i = 0
        while i < len(palabras):
            w = palabras[i]
            if w == "y":
                i += 1
                continue
            if w == "cero":
                digitos.append("0")
                i += 1
            elif w in _DECENAS_HABLADAS:
                val = _DECENAS_HABLADAS[w]
                if i + 2 < len(palabras) and palabras[i + 1] == "y" and palabras[i + 2] in _UNIDADES_HABLADAS:
                    val += _UNIDADES_HABLADAS[palabras[i + 2]]
                    i += 3
                else:
                    i += 1
                digitos.append(str(val))
            elif w in _UNIDADES_HABLADAS:
                digitos.append(str(_UNIDADES_HABLADAS[w]))
                i += 1
            else:
                break
        if not digitos:
            return m.group(0)
        num_str = "".join(digitos)
        if pref in prefijos_vigentes:
            # «CAB uno» puede referirse al código almacenado como CAB-001.
            # Solo se corrige si el sufijo identifica un código actual único.
            codigo_catalogo = _resolver_codigo_catalogo_por_sufijo(pref, num_str)
            if codigo_catalogo:
                return codigo_catalogo
            # Si el prefijo existe, pero el sufijo no distingue una referencia
            # vigente, no inventar el relleno de ceros: el índice de alias
            # posterior conservará los candidatos o la mención sin resolver.
            return m.group(0)
        if len(num_str) < 3 and num_str.isdigit():
            num_str = f"{int(num_str):03d}"
        return f"{pref}-{num_str}"

    return patron.sub(_reemplazar, texto)


def obtener_codigos_catalogo_dinamicos() -> set[str]:
    """Obtiene el conjunto de códigos de inventario existentes en el catálogo activo o en caché (ej. 'BAT-001', 'PAP-030')."""
    try:
        from ..cache import HakuCacheManager
        cat = HakuCacheManager.obtener_catalogo()
        if cat is None:
            cat = HakuCacheManager.obtener_o_cargar_catalogo()
        if cat:
            return {p["codigo_interno"].upper() for p in cat if "codigo_interno" in p}
    except Exception:
        pass
    try:
        from inventario.models import Producto
        return {
            str(codigo).strip().upper()
            for codigo in Producto.objects.filter(activo=True, archivado=False)
            .exclude(codigo_interno="")
            .values_list("codigo_interno", flat=True)
            if codigo
        }
    except Exception:
        return set()


def _codigos_catalogo_coincidentes_por_sufijo(prefijo: str, numero: str) -> List[str]:
    """Devuelve referencias vigentes compatibles con un prefijo y sufijo numérico."""
    if not prefijo or not str(numero).isdigit():
        return []
    prefijo = str(prefijo).upper()
    try:
        objetivo = int(numero)
    except (TypeError, ValueError):
        return []
    codigos = obtener_codigos_catalogo_dinamicos()
    coincidencias = {
        str(codigo).upper()
        for codigo in codigos
        if (coincidencia := re.fullmatch(rf"{re.escape(prefijo)}-(\d+)", str(codigo).upper()))
        and int(coincidencia.group(1)) == objetivo
    }
    return sorted(coincidencias)


def _resolver_codigo_catalogo_por_sufijo(prefijo: str, numero: str) -> Optional[str]:
    """Resuelve un sufijo a su código vigente solo cuando identifica uno único."""
    coincidencias = _codigos_catalogo_coincidentes_por_sufijo(prefijo, numero)
    return coincidencias[0] if len(coincidencias) == 1 else None


def _alias_fusionado_codigo(codigo: str) -> set[str]:
    """Crea alias de voz a partir del formato real del código, nunca de ejemplos fijos."""
    coincidencia = re.fullmatch(r"([A-Z]{2,5})-(\d{1,5})", str(codigo or "").upper())
    if not coincidencia:
        return set()
    prefijo, sufijo = coincidencia.groups()
    numero = int(sufijo)
    alias = {f"{prefijo.lower()}{sufijo}", f"{prefijo.lower()}{numero}"}
    palabra_numero = next(
        (palabra for palabra, valor in _UNIDADES_HABLADAS.items() if valor == numero and palabra != "un"),
        None,
    )
    if palabra_numero is None:
        decena = next(
            ((palabra, valor) for palabra, valor in _DECENAS_HABLADAS.items() if valor <= numero < valor + 10),
            None,
        )
        if decena:
            decena_nombre, valor_decena = decena
            unidad_nombre = next(
                (palabra for palabra, valor in _UNIDADES_HABLADAS.items() if valor == numero - valor_decena),
                None,
            )
            palabra_numero = decena_nombre + (f"y{unidad_nombre}" if unidad_nombre and numero > valor_decena else "")
    if palabra_numero:
        alias.add(f"{prefijo.lower()}{palabra_numero.replace(' ', '')}")
    return alias


def _distancia_edicion_max_uno(a: str, b: str) -> Optional[int]:
    """Distancia Levenshtein acotada a 1 para corregir errores ASR muy cercanos."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 1:
        return None
    i = j = diferencias = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diferencias += 1
        if diferencias > 1:
            return None
        if len(a) > len(b):
            i += 1
        elif len(b) > len(a):
            j += 1
        else:
            i += 1
            j += 1
    diferencias += (len(a) - i) + (len(b) - j)
    return diferencias if diferencias <= 1 else None


def _codigos_por_alias_hablado(texto: str, codigos_catalogo: set[str]) -> List[str]:
    """Devuelve alias únicos corregibles y alias exactos ambiguos como candidatos."""
    alias_a_codigos = _indice_aliases_codigo(frozenset(str(c).upper() for c in codigos_catalogo))
    encontrados = [codigo for _, _, codigo in _coincidencias_alias_hablado(texto, alias_a_codigos)]

    # Un alias exacto que corresponde a varios códigos no se corrige a uno,
    # pero sí se conserva como conjunto de candidatos para que el agente pida
    # desambiguación en vez de perder la referencia o elegir por orden.
    tokens = list(re.finditer(r"[\wáéíóúñü]+", texto or "", flags=re.IGNORECASE))
    for indice in range(len(tokens)):
        for ancho in (3, 2, 1):
            fin = indice + ancho
            if fin > len(tokens):
                continue
            if any(
                re.search(r"[-_]", (texto or "")[tokens[pos].end():tokens[pos + 1].start()])
                for pos in range(indice, fin - 1)
            ):
                # Un código escrito con separador explícito se interpreta en
                # la extracción literal, no como alias de voz fusionado.
                continue
            fragmento = "".join(
                normalizar(token.group(0)).replace("v", "b")
                for token in tokens[indice:fin]
            )
            candidatos = alias_a_codigos.get(fragmento, frozenset())
            if len(candidatos) > 1:
                encontrados.extend(sorted(candidatos))

    return list(dict.fromkeys(encontrados))


def normalizar_codigos_hablados_en_texto(texto: str) -> str:
    """Corrige solo menciones de código que el catálogo actual resuelve de forma única."""
    if not texto:
        return texto or ""
    codigos = obtener_codigos_catalogo_dinamicos()
    indice_aliases = _indice_aliases_codigo(frozenset(str(c).upper() for c in codigos))
    coincidencias = _coincidencias_alias_hablado(texto, indice_aliases)
    resultado = texto
    for inicio, fin, codigo in reversed(coincidencias):
        resultado = resultado[:inicio] + codigo + resultado[fin:]
    return resultado


def _coincidencias_alias_hablado(texto: str, alias_a_codigos: Dict[str, frozenset[str]]) -> List[tuple[int, int, str]]:
    """Encuentra ventanas contiguas de una a tres palabras que forman un alias único."""
    tokens = list(re.finditer(r"[\wáéíóúñü]+", texto or "", flags=re.IGNORECASE))
    encontrados: List[tuple[int, int, str]] = []
    indice = 0
    while indice < len(tokens):
        coincidencia = None
        # Probar la ventana más larga primero para no confundir el prefijo con
        # el sufijo dictado en palabras separadas.
        for ancho in (3, 2, 1):
            fin_indice = indice + ancho
            if fin_indice > len(tokens):
                continue
            fragmento = "".join(normalizar(token.group(0)).replace("v", "b") for token in tokens[indice:fin_indice])
            fragmento_literal = "".join(token.group(0) for token in tokens[indice:fin_indice])
            if not fragmento:
                continue
            candidatos = set(alias_a_codigos.get(fragmento, ()))
            prefijo_texto = normalizar((texto or "")[:tokens[indice].start()])
            marcador_codigo = bool(re.search(r"\b(?:codigo|referencia|ref)\s*$", prefijo_texto))
            primer_token_solo_digitos = tokens[indice].group(0).isdigit()
            forma_codigo_explicita = bool(fragmento_literal) and fragmento_literal.isupper() and not primer_token_solo_digitos
            tramo_original = (texto or "")[tokens[indice].start():tokens[fin_indice - 1].end()]
            m_sintaxis_codigo = re.fullmatch(r"([A-Za-z]{2,5})[-_](\d{1,5})", tramo_original)
            # Una corrección difusa de una sola palabra puede convertir un
            # sustantivo real (p. ej. «estaño») en el alias hablado de una
            # referencia del catálogo. Reservamos la tolerancia difusa para
            # un contexto explícito de código o una forma escrita como código;
            # los alias hablados exactos siguen admitidos sin esa señal.
            if not candidatos and m_sintaxis_codigo is not None:
                pref_frag = normalizar(m_sintaxis_codigo.group(1)).replace("v", "b")
                num_frag = int(m_sintaxis_codigo.group(2))
                candidatos_prefijo = set()
                candidatos_par_fonetico = set()
                _pares_sonoridad = {frozenset(("p", "b")), frozenset(("t", "d")), frozenset(("c", "g")), frozenset(("k", "g"))}
                for codigos in alias_a_codigos.values():
                    for codigo in codigos:
                        m_cod = re.fullmatch(r"([A-Za-z]{2,5})-(\d{1,5})", codigo)
                        if not m_cod:
                            continue
                        if int(m_cod.group(2)) != num_frag:
                            continue
                        pref_cod = normalizar(m_cod.group(1)).replace("v", "b")
                        if _distancia_edicion_max_uno(pref_frag, pref_cod) == 1:
                            candidatos_prefijo.add(codigo)
                            if len(pref_frag) == len(pref_cod):
                                difs = [(a, b) for a, b in zip(pref_frag, pref_cod) if a != b]
                                if len(difs) == 1 and frozenset(difs[0]) in _pares_sonoridad:
                                    candidatos_par_fonetico.add(codigo)
                if len(candidatos_prefijo) == 1:
                    candidatos = candidatos_prefijo
                elif len(candidatos_par_fonetico) == 1:
                    candidatos = candidatos_par_fonetico
            if not candidatos and not primer_token_solo_digitos and len(fragmento) >= 5 and (marcador_codigo or forma_codigo_explicita):
                candidatos = {
                    codigo
                    for alias, codigos in alias_a_codigos.items()
                    if len(alias) >= 5 and _distancia_edicion_max_uno(fragmento, alias) == 1
                    for codigo in codigos
                }
            if len(candidatos) == 1:
                codigo = next(iter(candidatos))
                coincidencia = (tokens[indice].start(), tokens[fin_indice - 1].end(), codigo)
                indice = fin_indice
                break
        if coincidencia:
            encontrados.append(coincidencia)
        else:
            indice += 1
    return encontrados


@lru_cache(maxsize=4)
def _indice_aliases_codigo(codigos_catalogo: frozenset[str]) -> Dict[str, frozenset[str]]:
    """Índice pequeño cacheado por versión del conjunto vivo de códigos."""
    alias_a_codigos: Dict[str, set[str]] = {}
    for codigo in codigos_catalogo:
        for alias in _alias_fusionado_codigo(codigo):
            clave = normalizar(alias).replace("v", "b")
            alias_a_codigos.setdefault(clave, set()).add(str(codigo).upper())
    return {alias: frozenset(codigos) for alias, codigos in alias_a_codigos.items()}


def _normalizar_colisiones_foneticas_codigo(texto: str) -> str:
    """Corrige prefijos de catálogo absorbidos por la preposición «de» en una transcripción.

    Protección estricta contra falsos positivos:
    1. Si la forma transcrita está precedida por un determinante o artículo, se preserva intacta.
    2. Solo se activa si la secuencia numérica posterior (1-5 dígitos) corresponde a un producto
       existente en el catálogo activo (o catálogo dinámico en caché).
    3. Los prefijos se derivan del catálogo actual; no se conserva un conjunto fijo de referencias.
    """
    if not texto:
        return texto

    codigos_cat = obtener_codigos_catalogo_dinamicos()
    prefijos = sorted(
        {codigo.split("-", 1)[0] for codigo in codigos_cat if re.fullmatch(r"[A-Z]{2,5}-\d{1,5}", codigo)},
        key=len,
        reverse=True,
    )
    if not prefijos:
        return texto

    alternativos = "|".join(re.escape(prefijo.lower()) for prefijo in prefijos)
    patron = re.compile(
        rf"(?:(\b(?:el|un|este|ese|aquel|al|del)\s+))?(?:(c[oó]digo|referencia|ref\.?)\s+)?\bde({alternativos})e?\s+(\d{{1,5}})\b",
        re.IGNORECASE,
    )

    def _reemplazar(m: re.Match) -> str:
        articulo, marcador, prefijo_raw, num_str = m.groups()
        if articulo:
            return m.group(0)
        prefijo = prefijo_raw.upper()
        numero = int(num_str)
        codigos_prefijo = {
            codigo for codigo in codigos_cat
            if codigo.startswith(f"{prefijo}-")
        }
        exacto = f"{prefijo}-{num_str}"
        existente = any(
            int(coincidencia.group(1)) == numero
            for codigo in codigos_prefijo
            if (coincidencia := re.fullmatch(rf"{re.escape(prefijo)}-(\d+)", codigo))
        )
        if not existente:
            return m.group(0)
        if not marcador and not (num_str.startswith("0") or len(num_str) >= 3) and exacto not in codigos_cat:
            return m.group(0)
        return f"de {prefijo} {num_str}"

    return patron.sub(_reemplazar, texto)


_IMPERATIVOS_CONSULTA_ENCLITICOS: Dict[str, str] = {
    "muestrame": "muéstrame",
    "muestreme": "muéstreme",
    "ensename": "enséñame",
    "buscame": "búscame",
    "listame": "lístame",
    "consultame": "consúltame",
    "indicame": "indícame",
    "digame": "dígame",
}

_PATRON_VERBO_CONSULTA_INICIAL = re.compile(
    r"^([¿¡\s]*(?:(?:oye|oiga|hola|hacu|haku|por\s+favor)\b[,:\s]*)*)"
    r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{6,})"
    r"(\s+(?:el|la|los|las|un|una|unos|unas|en|de|del|qu[eé]|cu[aá]l(?:es)?|cu[aá]nt[oa]s?|tod[oa]s?)\b)",
    re.IGNORECASE,
)


def _normalizar_verbo_consulta_inicial(texto: str) -> str:
    """Repara distorsiones fonéticas de 1 edición en imperativos iniciales de consulta (p. ej. 'Máestrame').

    Solo actúa al inicio de la frase (o tras un vocativo) antes de un determinante o
    interrogativo, y únicamente si existe una coincidencia única a distancia de edición 1
    contra el conjunto cerrado de imperativos enclíticos del español.
    """
    if not texto:
        return texto
    m = _PATRON_VERBO_CONSULTA_INICIAL.match(texto)
    if not m:
        return texto
    prefijo, token_raw, sufijo = m.groups()
    token_norm = normalizar(token_raw)
    if len(token_norm) < 6 or token_norm in _IMPERATIVOS_CONSULTA_ENCLITICOS:
        return texto
    candidatos = [
        canon for clave, canon in _IMPERATIVOS_CONSULTA_ENCLITICOS.items()
        if _distancia_edicion_max_uno(token_norm, clave) == 1
    ]
    if len(candidatos) != 1:
        return texto
    reemplazo = candidatos[0]
    if token_raw[:1].isupper():
        reemplazo = reemplazo[:1].upper() + reemplazo[1:]
    return f"{prefijo}{reemplazo}{sufijo}{texto[m.end():]}"


def preprocesar_texto_inventario(
    texto: Optional[str],
    *,
    normalizar_medidas: bool = True,
) -> str:
    """Aplica normalización fonética Whisper, siglas técnicas, códigos hablados y perífrasis dimensionales."""
    if not texto:
        return ""
    t = str(texto)
    try:
        from ..lenguaje.costeno import InterpretadorCosteno
        from ..lenguaje.siglas import NormalizadorSiglasTecnicas

        t = InterpretadorCosteno.normalizar_fonetica_whisper(t)
        t = NormalizadorSiglasTecnicas.normalizar(t)
    except Exception:
        pass
    t = _normalizar_verbo_consulta_inicial(t)
    t = _normalizar_numeros_codigo_hablado(t)
    t = _normalizar_colisiones_foneticas_codigo(t)
    if normalizar_medidas:
        t = _normalizar_cantidades_con_unidad(t)
    for patron, reemplazo in _PERIFRASIS_CANONICAS:
        t = re.sub(patron, reemplazo, t, flags=re.IGNORECASE)
    return t


def normalizar(texto: Optional[str]) -> str:
    """Minúsculas, sin tildes y con espacios simples."""
    if not texto:
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[^a-z0-9\-\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def obtener_prefijos_catalogo_dinamicos() -> set[str]:
    """Obtiene los prefijos de códigos existentes en el catálogo desde caché o base de datos.

    Aprovecha el catálogo dinámico para evitar diccionarios rígidos y permitir generalización a nuevas categorías.
    """
    try:
        from ..cache import HakuCacheManager
        cat = HakuCacheManager.obtener_catalogo()
        if cat is None:
            cat = HakuCacheManager.obtener_o_cargar_catalogo()
        if cat:
            return {p["codigo_interno"].split("-")[0].upper() for p in cat if "-" in p.get("codigo_interno", "")}
    except Exception:
        pass
    try:
        from inventario.models import Producto
        codigos = (
            Producto.objects.filter(activo=True, archivado=False)
            .exclude(codigo_interno="")
            .values_list("codigo_interno", flat=True)
        )
        return {codigo.split("-")[0].upper() for codigo in codigos if "-" in codigo}
    except Exception:
        # Sin un catálogo disponible no se presume ningún prefijo de producto.
        return set()


def normalizar_codigo_producto(texto: Optional[str]) -> Optional[str]:
    """Si el texto representa únicamente un código de producto (ej. 'pap 030', 'PAP030', 'PAP-030'), devuelve 'PAP-030'."""
    if not texto:
        return None
    limpio = preprocesar_texto_inventario(texto).strip()
    tras_marcador = bool(re.match(r"^(?:c[oó]digo|referencia|ref\.?)\s+", limpio, flags=re.IGNORECASE))
    limpio = re.sub(r"^(?:c[oó]digo|referencia|ref\.?)\s+", "", limpio, flags=re.IGNORECASE).strip().upper()
    m = re.match(r"^([A-Z]{2,5})([\s\-_]?)(\d{1,5})$", limpio)
    if m:
        pref, sep, num = m.group(1), m.group(2), m.group(3)
        tiene_guion = sep in ("-", "_")
        prefijos_cat = obtener_prefijos_catalogo_dinamicos()
        es_prefijo = pref in prefijos_cat
        if pref in _NO_PREFIJOS_CODIGO and not es_prefijo:
            return None
        codigo_catalogo = _resolver_codigo_catalogo_por_sufijo(pref, num) if es_prefijo else None
        guion_literal_en_entrada = bool(
            re.search(rf"\b{re.escape(pref)}\s*[-_]\s*{re.escape(num)}\b", str(texto), flags=re.IGNORECASE)
        )
        if es_prefijo and tiene_guion and guion_literal_en_entrada:
            codigo_literal = f"{pref}-{num}"
            if codigo_literal in obtener_codigos_catalogo_dinamicos():
                codigo_catalogo = codigo_literal
        if es_prefijo and len(num) < 3 and not codigo_catalogo:
            # Los sufijos cortos solo se aceptan al identificar una referencia
            # vigente única; ni el guion ni el marcador de voz eliminan colisiones.
            return None
        # Si no tiene guión ni marcador explícito de código, y no es prefijo del catálogo, no es un código
        if not tiene_guion and not tras_marcador and not es_prefijo:
            return None
        if not tiene_guion and not tras_marcador:
            # Una sigla técnica seguida por un número (p. ej. STM32) no es un
            # código solo porque su prefijo exista: la combinación debe estar
            # presente en el catálogo para evitar convertir nombres de modelo.
            if not codigo_catalogo:
                return None
        if not es_prefijo and not tras_marcador and len(num) < 3:
            from inventario.models import Producto
            cand = f"{pref}-{num}"
            if not Producto.objects.filter(codigo_interno__iexact=cand).exists():
                return None
        if codigo_catalogo:
            return codigo_catalogo
        if not tiene_guion and len(num) < 3 and not tras_marcador and not codigo_catalogo:
            return None
        if tiene_guion or len(num) >= 2 or es_prefijo or tras_marcador:
            return f"{pref}-{num}"
    return None


def extraer_codigos_de_texto(texto: Optional[str]) -> List[str]:
    """Extrae códigos de inventario (XXX-NNN, XXX0NN, XXXNNN) evitando falsos positivos con cantidades ordinarias y palabras comunes."""
    if not texto:
        return []
    texto_original = str(texto or "")
    pre = preprocesar_texto_inventario(texto_original)
    prefijos_cat = obtener_prefijos_catalogo_dinamicos()

    # Algunos reconocedores separan cada dígito con guiones. Solo unirlos si
    # el resultado existe en el catálogo actual y no hay un código literal
    # diferente con esa misma grafía; nunca se inventa una referencia.
    codigos_catalogo = obtener_codigos_catalogo_dinamicos()

    def _unir_digitos_dictados(m: re.Match) -> str:
        prefijo, digitos = m.group(1).upper(), m.group(2)
        if prefijo not in prefijos_cat:
            return m.group(0)
        literal = f"{prefijo}-{digitos}".upper()
        unido = f"{prefijo}-{digitos.replace('-', '')}".upper()
        if unido in codigos_catalogo and literal not in codigos_catalogo:
            return unido
        return m.group(0)

    pre = re.sub(
        r"\b([A-Za-z]{2,5})\s*[,\- ]\s*(\d(?:-\d){1,4})(?![-\d])",
        _unir_digitos_dictados,
        pre,
    )

    # 1. Separar preposiciones fonéticas pegadas (ej. 'debat 001' -> 'de BAT 001')
    # Solo cuando 'de'/'del' no forma parte de un código explícito con guion o precedido de 'código'
    def _separar_prep(m: re.Match, p_upper: str) -> str:
        marcador = m.group(1)
        prep = m.group(2)
        num = m.group(4)
        if marcador:
            return m.group(0)
        return f"{prep} {p_upper} {num}"

    for p_cat in prefijos_cat:
        patron_prep = rf"\b(?:(c[oó]digo|referencia|ref\.?)\s+)?(de|del)({p_cat.lower()})\s*(\d{{1,5}})\b"
        pre = re.sub(patron_prep, lambda m, p=p_cat: _separar_prep(m, p), pre, flags=re.IGNORECASE)

    encontrados: List[str] = []
    for fuente in (pre,):
        for m in re.finditer(r"\b([A-Za-z]{2,5})([\s\-_]?)(\d{1,5})\b(?![-_]\d)", fuente):
            pref_raw = m.group(1)
            pref = pref_raw.upper()
            sep = m.group(2)
            num = m.group(3)
            prev = fuente[: m.start()].rstrip().lower()
            tras_codigo = bool(re.search(r"\b(?:c[oó]digo|ref(?:erencia)?\.?)$", prev))
            tiene_guion = sep in ("-", "_")
            es_prefijo_catalogo = pref in prefijos_cat
            if pref in _NO_PREFIJOS_CODIGO and not es_prefijo_catalogo and not tras_codigo:
                continue

            if es_prefijo_catalogo and len(num) < 3 and not num.startswith("0"):
                coincidencias_sufijo = _codigos_catalogo_coincidentes_por_sufijo(pref, num)
                codigo_literal = f"{pref}-{num}"
                guion_literal_en_entrada = bool(
                    re.search(
                        rf"\b{re.escape(pref)}\s*[-_]\s*{re.escape(num)}\b",
                        texto_original,
                        flags=re.IGNORECASE,
                    )
                )
                if tiene_guion and guion_literal_en_entrada and codigo_literal in obtener_codigos_catalogo_dinamicos():
                    coincidencias_sufijo = [codigo_literal]
                if len(coincidencias_sufijo) > 1:
                    encontrados.extend(coincidencias_sufijo)
                elif len(coincidencias_sufijo) == 1:
                    encontrados.append(coincidencias_sufijo[0])
                # No aproximar un sufijo corto sin coincidencia en el catálogo.
                continue

            # Si NO tiene guión ni está precedido explícitamente por 'código'/'referencia':
            # NO convertir palabras comunes seguidas de número (ej. 'CAJA 100', 'PISO 200') en códigos
            # a menos que sea un prefijo reconocido del catálogo (ej. 'BAT 001', 'PAP 030')
            if not tiene_guion and not tras_codigo:
                if not es_prefijo_catalogo:
                    continue
                # Los sufijos cortos solo se aceptan si el catálogo actual los
                # resuelve sin colisiones, sin depender de mayúsculas dictadas.
                if sep.isspace() and len(num) < 3 and not num.startswith("0") and not _resolver_codigo_catalogo_por_sufijo(pref, num):
                    continue

            if not es_prefijo_catalogo and not tras_codigo and len(num) < 3:
                from inventario.models import Producto
                cand = f"{pref}-{num}"
                if not Producto.objects.filter(codigo_interno__iexact=cand).exists():
                    continue

            # Un nombre como «STM32» puede compartir prefijo con códigos del
            # catálogo, pero solo es código si esa combinación existe hoy.
            # La forma con guion o el marcador «código» siguen siendo
            # menciones explícitas y se validan en la resolución normal.
            if not sep and not tras_codigo and normalizar_codigo_producto(f"{pref_raw}{num}") != f"{pref}-{num}":
                continue

            canon = _resolver_codigo_catalogo_por_sufijo(pref, num) or f"{pref}-{num}"
            if canon not in encontrados:
                encontrados.append(canon)
    for codigo in _codigos_por_alias_hablado(pre, obtener_codigos_catalogo_dinamicos()):
        if codigo not in encontrados:
            encontrados.append(codigo)
    return encontrados


def _raiz(palabra: str) -> str:
    """Reducción morfológica mínima singular/plural y diminutivos del español."""
    p = palabra
    if len(p) > 5 and p.endswith(("itas", "itos", "illa", "illo")):
        p = p[:-4]
    elif len(p) > 4 and p.endswith(("ita", "ito", "ico", "ica")):
        p = p[:-3]
    elif len(p) > 4 and p.endswith("es") and not p.endswith(("bles", "tres")):
        p = p[:-2]
    elif len(p) > 3 and p.endswith("s"):
        p = p[:-1]
    return p


def _lexema(palabra: str) -> str:
    """Obtiene el lexema base eliminando vocal temática final átona tras reducción de plural/diminutivo."""
    r = _raiz(palabra)
    if len(r) > 4 and r[-1] in ("a", "o", "e"):
        return r[:-1]
    return r


def _es_agentivo(palabra: str) -> bool:
    """True si la palabra tiene sufijo instrumental/agentivo (-dor / -dora), ej. 'grapadora', 'cosedora', 'perforadora'."""
    r = _raiz(palabra)
    return len(r) >= 6 and r.endswith(("dor", "dora"))


def _sinonimos_de(token: str) -> set[str]:
    """Devuelve el conjunto de raíces equivalentes para un token según _GRUPOS_SINONIMOS."""
    r = _raiz(token)
    l = _lexema(token)
    equivalentes = {r}
    for grupo in _GRUPOS_SINONIMOS:
        raices_grupo = {_raiz(x) for x in grupo}
        lexemas_grupo = {_lexema(x) for x in grupo}
        if r in raices_grupo or l in lexemas_grupo:
            equivalentes |= raices_grupo
    return equivalentes


def _distancia_edicion_max1(a: str, b: str) -> bool:
    """True si las cadenas a y b están a distancia de Levenshtein <= 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    # Asegurar la <= lb
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = diffs = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
    return True


def clave_fonetica_es(palabra: str) -> str:
    """Clave fonética ligera para español hablado (betacismo b/v, seseo s/c/z, yeísmo ll/y, g(e,i)/j, qu/k/c, h muda)."""
    s = normalizar(palabra)
    if not s:
        return ""
    s = s.replace("ch", "\x01")
    s = re.sub(r"c(?=[ei])", "s", s)
    s = s.replace("z", "s").replace("v", "b").replace("ll", "y")
    s = re.sub(r"g(?=[ei])", "j", s)
    s = re.sub(r"qu(?=[ei])", "k", s)
    s = s.replace("c", "k").replace("h", "")
    return s.replace("\x01", "ch")


def tokens_significativos(texto: Optional[str]) -> List[str]:
    pre = preprocesar_texto_inventario(texto)
    return [_raiz(t) for t in normalizar(pre).split() if t not in _PALABRAS_VACIAS and len(t) > 1]


@dataclass
class FiltrosConsulta:
    texto: Optional[str] = None
    codigos: List[str] = field(default_factory=list)
    categoria: Optional[str] = None
    almacen: Optional[str] = None
    ubicacion: Optional[str] = None
    condicion: Optional[str] = None
    umbral: Optional[int] = None
    categoria_explicita: bool = False
    categoria_generica: bool = False
    es_prestable: Optional[bool] = None

    def vacio(self) -> bool:
        return not any([
            self.texto,
            self.codigos,
            self.categoria,
            self.almacen,
            self.ubicacion,
            self.condicion,
            self.categoria_generica,
            self.es_prestable is not None,
        ])


@dataclass
class ResultadoHerramienta:
    items: List[Dict[str, Any]] = field(default_factory=list)
    total: int = 0
    error: Optional[str] = None  # "permiso_denegado" | "almacen_no_autorizado" | "almacen_inexistente" | "categoria_inexistente"
    limitacion: Optional[str] = None  # "umbral_no_definido"
    coincidencia: str = "exacta"  # "exacta" | "aproximada" | "sinonimo" | "funcional" | "categoria" | "relacionada" | "descriptiva"
    almacen_filtrado: Optional[str] = None
    detalle_filtros: Dict[str, Any] = field(default_factory=dict)
    relacionados: List[Dict[str, Any]] = field(default_factory=list)


class ListaResultadoHerramienta(list):
    """Lista de resultados de herramienta con metadatos de error y permiso."""

    def __init__(
        self,
        iterable=(),
        error: Optional[str] = None,
        permiso_requerido: Optional[str] = None,
        total_eventos: Optional[int] = None,
        unidades_evento: Optional[int] = None,
    ):
        super().__init__(iterable)
        self.error = error
        self.permiso_requerido = permiso_requerido
        self.permiso_denegado = error == "permiso_denegado"
        self.total_eventos = total_eventos
        self.unidades_evento = unidades_evento


class HerramientasInventario:
    """Fachada de consultas de solo lectura con alcance por usuario."""

    def __init__(self, usuario: Any):
        self.usuario = usuario
        self.es_admin = bool(getattr(usuario, "es_administrador", False) or getattr(usuario, "is_superuser", False))
        try:
            self.almacenes = list(usuario.obtener_almacenes_accesibles())
        except Exception:
            self.almacenes = []
        self._ids_almacenes = {a.id for a in self.almacenes}

    def tiene_permiso(self, *codigos: str) -> bool:
        """Verifica si el usuario cuenta con al menos uno de los permisos operativos solicitados."""
        if not self.usuario or not getattr(self.usuario, "is_authenticated", True):
            return False
        if not getattr(self.usuario, "is_active", True) or getattr(self.usuario, "eliminado_logicamente", False):
            return False
        if self.es_admin:
            return True
        fn = getattr(self.usuario, "tiene_permiso", None)
        if callable(fn):
            return any(fn(c) for c in codigos)
        return False

    # ── Catálogos auxiliares para dar contexto al planificador ──────────────
    def nombres_categorias(self) -> List[str]:
        from inventario.models import Categoria

        qs = Categoria.objects.filter(activo=True, eliminado=False)
        if not self.es_admin:
            if not self._ids_almacenes:
                return []
            qs = qs.filter(
                Q(almacen_principal_id__in=self._ids_almacenes)
                | Q(productos__existencias__almacen_id__in=self._ids_almacenes)
            ).distinct()
        return list(qs.values_list("nombre", flat=True))

    def nombres_almacenes(self) -> List[str]:
        return [a.nombre for a in self.almacenes]

    def nombres_ubicaciones(self) -> List[str]:
        from inventario.models import Ubicacion

        return list(
            Ubicacion.objects.filter(
                almacen_id__in=self._ids_almacenes,
                activo=True,
                eliminado=False,
            ).order_by("nombre").values_list("nombre", flat=True).distinct()
        )

    def candidatos_producto_para_mensaje(self, texto: str, limite: int = 5) -> List[Dict[str, str]]:
        """Recupera nombres actuales y visibles que pueden aclarar una mención indirecta.

        El ranking usa el índice dinámico del catálogo y conserva el alcance RBAC.
        Los candidatos son contexto para el modelo, no una decisión automática ni
        una lista de sinónimos mantenida en código.
        """
        if not texto or not self.tiene_permiso("catalogo.ver"):
            return []
        from .planificador import tiene_referencia_contextual_operacion

        if tiene_referencia_contextual_operacion(texto):
            _no_entidad_op = {
                "presta", "prestar", "prestame", "prestale", "devuelve", "devolver", "regresa",
                "regresar", "saca", "sacar", "retira", "retirar", "ingresa", "ingresar",
                "agrega", "agregar", "traslada", "trasladar", "mueve", "mover", "pasa", "pasar",
                "inhabilita", "inhabilitar", "habilita", "habilitar", "desactiva", "activar",
                "uno", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
                "unidad", "unidades", "de", "del", "a", "al", "en", "para", "por", "el", "la", "los", "las",
                "un", "ese", "esa", "esos", "esas", "este", "esta", "estos", "estas",
                "aquel", "aquella", "aquellos", "aquellas", "mismo", "misma", "dicho", "dicha",
                "producto", "articulo", "referencia", "equipo", "herramienta", "anterior",
                "quiero", "necesito", "voy", "deseo", "favor", "nombre",
            }
            tokens_restantes = [
                t for t in normalizar(texto).split()
                if t and not t.isdigit() and t not in _no_entidad_op
            ]
            if not tokens_restantes:
                return []
        from ..buscador import HakuSearchIndex

        indice = HakuSearchIndex.obtener()
        almacen_id = next(iter(self._ids_almacenes)) if len(self._ids_almacenes) == 1 else None
        candidatos = indice.buscar(
            texto=texto,
            usuario=self.usuario,
            almacen_id=almacen_id,
            top_k=max(1, min(int(limite or 5), 8)),
        )
        return [
            {
                "codigo": str(item.get("codigo") or "").strip(),
                "nombre": str(item.get("nombre") or "").strip(),
                "categoria": str(item.get("categoria_nombre") or "").strip(),
                "descripcion": str(item.get("descripcion") or "").strip(),
            }
            for item in candidatos
            if item.get("codigo") and item.get("nombre")
        ]

    def codigos_productos_mencionados(self, texto: str) -> List[str]:
        """Devuelve códigos actuales de nombres completos citados por el usuario.

        Usa el índice dinámico y el mismo alcance de almacenes de la consulta;
        nunca añade nombres de productos al código ni expone productos ajenos al RBAC.
        """
        if not self.tiene_permiso("catalogo.ver"):
            return []
        from ..buscador import HakuSearchIndex

        indice = HakuSearchIndex.obtener()
        almacen_id = next(iter(self._ids_almacenes)) if len(self._ids_almacenes) == 1 else None
        productos = indice.mencionar_productos_literales(
            texto,
            usuario=self.usuario,
            almacen_id=almacen_id,
        )
        return [producto["codigo"] for producto in productos if producto.get("codigo")]

    # ── Resolución de almacén ───────────────────────────────────────────────
    def _resolver_almacen(self, nombre: str):
        from inventario.models import Almacen

        n = normalizar(nombre).replace("almacen", "").replace("bodega", "").strip()
        candidatos = Almacen.objects.filter(eliminado=False)
        for a in candidatos:
            na = normalizar(a.nombre)
            if na == normalizar(nombre) or na.replace("almacen", "").strip() == n:
                return a
        return None

    # ── Herramienta principal: búsqueda de productos ───────────────────────
    def buscar_productos(self, filtros: FiltrosConsulta) -> ResultadoHerramienta:
        from inventario.models import Existencia, Producto, Ubicacion
        from django.db.models import Prefetch

        res = ResultadoHerramienta()
        if not self.tiene_permiso("catalogo.ver"):
            res.error = "permiso_denegado"
            res.detalle_filtros["permiso"] = "catalogo.ver"
            return res

        almacenes_ids = set(self._ids_almacenes)
        almacenes_autorizados_ids = set(almacenes_ids)

        if filtros.almacen:
            partes = [p.strip() for p in filtros.almacen.split("|") if p.strip()]
            resueltos = [self._resolver_almacen(parte) for parte in partes]
            if not partes or any(alm is None for alm in resueltos):
                res.error = "almacen_inexistente"
                res.detalle_filtros["almacen"] = filtros.almacen
                return res
            no_autorizados = [alm.nombre for alm in resueltos if alm.id not in almacenes_ids]
            if no_autorizados:
                res.error = "almacen_no_autorizado"
                res.detalle_filtros["almacen"] = no_autorizados
                return res
            almacenes_ids &= {alm.id for alm in resueltos}
            res.almacen_filtrado = ", ".join(alm.nombre for alm in resueltos)

        qs = Producto.objects.select_related("categoria", "ubicacion", "ubicacion__almacen").prefetch_related(
            Prefetch(
                "existencias",
                queryset=Existencia.objects.filter(almacen_id__in=almacenes_ids).select_related("almacen"),
                to_attr="exist_visibles",
            ),
        )
        if not self.es_admin:
            qs = qs.filter(activo=True, archivado=False)
        # Alcance: solo productos con existencia en almacenes visibles
        qs = qs.filter(existencias__almacen_id__in=almacenes_ids).distinct()

        if not filtros.codigos and filtros.texto:
            cod_directo = normalizar_codigo_producto(filtros.texto)
            if cod_directo:
                filtros.codigos = [cod_directo]
                filtros.texto = None

        cands_idx = None
        if filtros.codigos:
            codigos_raw = [c.strip().upper() for c in filtros.codigos if c and c.strip()]
            q = Q()
            for c in codigos_raw:
                q |= Q(codigo_interno__iexact=c)
                # Formatos normalizados equivalentes (p. ej. PAP030 o PAP 030 <-> PAP-030)
                limpio = re.sub(r"[\s\-_]", "", c)
                m = re.match(r"^([A-Z]+)(\d+)$", limpio)
                if m:
                    pref_c, num_c = m.group(1), m.group(2)
                    q |= Q(codigo_interno__iexact=f"{pref_c}-{num_c}")
                    q |= Q(codigo_interno__iexact=f"{pref_c} {num_c}")
                    q |= Q(codigo_interno__iexact=f"{pref_c}{num_c}")
                    if len(num_c) < 5:
                        q |= Q(codigo_interno__iexact=f"{pref_c}-{int(num_c):02d}")
                        q |= Q(codigo_interno__iexact=f"{pref_c}-{int(num_c):03d}")
                        q |= Q(codigo_interno__iexact=f"{pref_c}-{int(num_c):04d}")
            qs = qs.filter(q)
        elif filtros.texto:
            # Optimización Fase II-B.4: Poda O(1)/O(k) con Índice Invertido Ligero RBAC
            try:
                from ..buscador import HakuSearchIndex
                indice = HakuSearchIndex.obtener()
                alm_target = next(iter(almacenes_ids)) if len(almacenes_ids) == 1 else None
                cands_idx = indice.buscar(
                    texto=filtros.texto,
                    usuario=self.usuario,
                    almacen_id=alm_target,
                    categoria=filtros.categoria,
                    top_k=50,
                )
                if cands_idx:
                    pids_cands = [c["id"] for c in cands_idx]
                    qs = qs.filter(id__in=pids_cands)
                else:
                    # El índice reduce el universo, pero no es una fuente de
                    # verdad: variantes morfológicas o entradas recién creadas
                    # pueden no estar indexadas todavía. Se deja que el filtro
                    # dinámico del catálogo completo acotado por permisos resuelva.
                    cands_idx = None
            except Exception as exc:
                logger.debug("Fallback directo de búsqueda: %s", exc)

        productos = list(qs.order_by("codigo_interno"))
        if cands_idx:
            id_order = {c["id"]: i for i, c in enumerate(cands_idx)}
            productos.sort(key=lambda p: id_order.get(p.id, 999))
        universo_base = list(productos)

        if filtros.categoria:
            cat_tokens = tokens_significativos(filtros.categoria)
            filtrados = [
                p for p in productos
                if p.categoria and cat_tokens
                and all(any(_raiz(w).startswith(t) for w in normalizar(p.categoria.nombre).split()) for t in cat_tokens)
            ]
            if not filtrados:
                if filtros.categoria_explicita:
                    # La ausencia de productos en el almacén solicitado no
                    # demuestra que la categoría no exista. Verifica su
                    # presencia en el catálogo visible para la cuenta, sin
                    # ampliar permisos ni consultar categorías de almacenes
                    # ajenos.
                    categorias_visibles = Producto.objects.filter(
                        existencias__almacen_id__in=almacenes_autorizados_ids,
                    )
                    if not self.es_admin:
                        categorias_visibles = categorias_visibles.filter(activo=True, archivado=False)
                    nombres_categorias_visibles = (
                        categorias_visibles.exclude(categoria__isnull=True)
                        .values_list("categoria__nombre", flat=True)
                        .distinct()
                    )
                    categoria_en_catalogo_visible = any(
                        cat_tokens
                        and all(
                            any(_raiz(palabra).startswith(token) for palabra in normalizar(nombre).split())
                            for token in cat_tokens
                        )
                        for nombre in nombres_categorias_visibles
                    )
                    if categoria_en_catalogo_visible:
                        res.detalle_filtros["categoria_sin_resultados_en_alcance"] = filtros.categoria
                    else:
                        res.error = "categoria_inexistente"
                        res.detalle_filtros["categoria_sin_coincidencia"] = filtros.categoria
                    productos = []
                elif productos and filtros.texto:
                    # Si el LLM repitió el nombre del producto como categoría,
                    # no descartar una búsqueda textual ya resuelta. Solo una
                    # categoría explícita del usuario puede estrechar el filtro.
                    res.detalle_filtros["categoria_ignorada_no_catalogada"] = filtros.categoria
                    filtros.categoria = None
                elif productos and not filtros.texto:
                    # Si el LLM clasificó un sustantivo de producto en el campo categoría sin que el usuario
                    # pidiera explícitamente una categoría, resolverlo como búsqueda de producto
                    prod_texto, modo = self._filtrar_por_texto(productos, filtros.categoria)
                    if prod_texto:
                        productos = prod_texto
                        res.coincidencia = modo
                        res.detalle_filtros["categoria_resuelta_como_producto"] = filtros.categoria
                        # Normaliza el plan para que el redactor no afirme que
                        # el nombre del producto es una categoría del catálogo.
                        filtros.texto = filtros.categoria
                        filtros.categoria = None
                    else:
                        rels = self._buscar_relacionados(universo_base, filtros.categoria)
                        if rels:
                            res.relacionados = [self._serializar(p, filtros.umbral) for p in rels]
                            res.coincidencia = "relacionada"
                        res.detalle_filtros["categoria_sin_coincidencia"] = filtros.categoria
                        productos = []
                else:
                    res.detalle_filtros["categoria_sin_coincidencia"] = filtros.categoria
                    productos = []
            else:
                productos = filtrados

        if filtros.texto and not filtros.codigos and "categoria_resuelta_como_producto" not in res.detalle_filtros:
            productos, modo = self._filtrar_por_texto(productos, filtros.texto)
            res.coincidencia = modo
            if not productos:
                rels = self._buscar_relacionados(universo_base, filtros.texto)
                if rels:
                    res.relacionados = [self._serializar(p, filtros.umbral) for p in rels]
                    res.coincidencia = "relacionada"

        if filtros.ubicacion:
            u = normalizar(filtros.ubicacion)
            ubicaciones_activas = set(
                Ubicacion.objects.filter(
                    almacen_id__in=almacenes_ids,
                    nombre__iexact=filtros.ubicacion,
                    activo=True,
                    eliminado=False,
                    almacen__activo=True,
                    almacen__eliminado=False,
                ).values_list("id", flat=True)
            )
            productos = [
                p for p in productos
                if getattr(p, "ubicacion_id", None) in ubicaciones_activas
                or any(
                    getattr(r, "ubicacion_id", None) in ubicaciones_activas
                    for r in getattr(p, "ubic_rel", [])
                )
            ]

        if filtros.es_prestable is not None:
            productos = [
                producto for producto in productos
                if bool(producto.es_prestable) is filtros.es_prestable
            ]

        items = [self._serializar(p, filtros.umbral) for p in productos]

        if filtros.condicion:
            items = [i for i in items if self._cumple_condicion(i, filtros.condicion, filtros.umbral)]

        res.items = items
        res.total = len(items)
        return res

    def _filtrar_por_texto(self, productos, texto: str, _permitir_fonetica: bool = True):
        # Fusión dinámica de tokens técnicos con guion presentes en los nombres del catálogo actual
        # (ej. «USB C» -> «USB-C», «RJ 45» -> «RJ-45», «Cat 6» -> «Cat-6», «HC SR04» -> «HC-SR04»),
        # sin codificar ninguna sigla fija ni alterar códigos internos.
        texto_busqueda = texto or ""
        if texto_busqueda and productos:
            compuestos_guion: set[str] = set()
            for p_cand in productos:
                for w_nom in normalizar(getattr(p_cand, "nombre", "") or "").split():
                    if "-" in w_nom and len(w_nom) >= 3:
                        partes = [pt for pt in w_nom.split("-") if pt]
                        if len(partes) == 2 and all(pt.isalnum() for pt in partes):
                            compuestos_guion.add(w_nom)
            if compuestos_guion:
                t_pre = preprocesar_texto_inventario(texto_busqueda)
                for comp in sorted(compuestos_guion, key=len, reverse=True):
                    p1, p2 = comp.split("-", 1)
                    t_pre_nuevo = re.sub(
                        rf"(?<![\w-]){re.escape(p1)}\s+{re.escape(p2)}(?![\w-])",
                        comp,
                        t_pre,
                        flags=re.IGNORECASE,
                    )
                    if t_pre_nuevo != t_pre:
                        t_pre = t_pre_nuevo
                if t_pre != preprocesar_texto_inventario(texto_busqueda):
                    texto_busqueda = t_pre

        tokens_brutos = tokens_significativos(texto_busqueda)
        if not tokens_brutos:
            return productos, "exacta"

        # Si hay sustantivos nucleares de producto junto con palabras de encuadre funcional
        # (ej. «cables sirven para conectar equipo»), conservar los sustantivos nucleares.
        raices_funcionales = {_raiz(w) for w in _PALABRAS_FUNCIONALES}
        tokens_nucleares = [t for t in tokens_brutos if t not in raices_funcionales]
        tokens = tokens_nucleares if tokens_nucleares else tokens_brutos

        def palabras(p):
            nombre_codigo = _normalizar_cantidades_con_unidad(f"{p.nombre} {p.codigo_interno}")
            return [_raiz(w) for w in normalizar(nombre_codigo).split()]

        # El filtro puede conservar palabras del preámbulo oral que el modelo
        # no separó de la entidad (incluidos regionalismos no enumerados aquí).
        # Usa el catálogo candidato vigente para quitar solo el prefijo que no
        # puede pertenecer a ningún nombre; los calificadores posteriores se
        # mantienen para no ampliar silenciosamente una petición más precisa.
        palabras_catalogo = []
        for producto in productos:
            palabras_catalogo.extend(palabras(producto))
            descripcion = preprocesar_texto_inventario(getattr(producto, "descripcion", "") or "")
            palabras_catalogo.extend(
                _raiz(palabra)
                for palabra in normalizar(descripcion).split()
                if palabra not in _PALABRAS_VACIAS and len(palabra) > 1
            )

        def token_anclado(token):
            variantes = {token, *(_sinonimos_de(token) or set())}
            return any(
                palabra == variante
                or (len(variante) >= 3 and palabra.replace("-", "") == variante.replace("-", ""))
                or (len(variante) >= 4 and palabra.startswith(variante))
                or (len(palabra) >= 4 and variante.startswith(palabra))
                or (len(_lexema(variante)) >= 3 and _lexema(palabra) == _lexema(variante))
                for palabra in palabras_catalogo
                for variante in variantes
            )

        # El plan del LLM a veces deja al final un verbo plural del marco de
        # consulta («resistencia aparecen»). Se elimina solo una forma verbal
        # morfológica que no existe en el catálogo; calificadores desconocidos
        # como «azul» se conservan para no ensanchar la búsqueda.
        while (
            len(tokens) > 1
            and not token_anclado(tokens[-1])
            and len(tokens[-1]) >= 6
            and tokens[-1].endswith(("an", "en"))
        ):
            tokens.pop()

        tokens_anclados = []
        for token in tokens:
            tokens_anclados.append(token_anclado(token))
        primer_ancla = next((i for i, anclado in enumerate(tokens_anclados) if anclado), None)
        if primer_ancla is not None and all(tokens_anclados[primer_ancla:]):
            tokens = tokens[primer_ancla:]

        def asignar_tokens(p, compatible):
            disponibles = palabras(p)
            idx_tok = 0
            while idx_tok < len(tokens):
                token = tokens[idx_tok]
                indice = next((i for i, palabra in enumerate(disponibles) if compatible(palabra, token)), None)
                if indice is not None:
                    disponibles.pop(indice)
                    idx_tok += 1
                    continue
                # Tolerancia bidireccional espacio <-> guion/unión (ej. producto «USB C» vs consulta «USB-C»/«USBC»,
                # o consulta «RJ 45» / «HC SR04» vs producto «RJ45» / «HC-SR04»).
                par_cat = next(
                    (
                        i
                        for i in range(len(disponibles) - 1)
                        if compatible(f"{disponibles[i]}-{disponibles[i + 1]}", token)
                        or compatible(f"{disponibles[i]}{disponibles[i + 1]}", token)
                    ),
                    None,
                )
                if par_cat is not None:
                    disponibles.pop(par_cat + 1)
                    disponibles.pop(par_cat)
                    idx_tok += 1
                    continue
                if idx_tok + 1 < len(tokens):
                    siguiente = tokens[idx_tok + 1]
                    par_q = next(
                        (
                            i
                            for i, palabra in enumerate(disponibles)
                            if compatible(palabra, f"{token}-{siguiente}")
                            or compatible(palabra, f"{token}{siguiente}")
                        ),
                        None,
                    )
                    if par_q is not None:
                        disponibles.pop(par_q)
                        idx_tok += 2
                        continue
                return False
            return True

        exactos = [p for p in productos if asignar_tokens(p, lambda palabra, token: palabra == token)]
        if exactos:
            return exactos, "exacta"

        # Coincidencia por prefijo de familia de código (p. ej. "PEG" o "familia PAP" -> códigos PEG-*)
        if len(tokens) == 1 and re.match(r"^[a-z]{2,5}$", tokens[0]):
            pref = f"{tokens[0]}-"
            por_prefijo_cod = [p for p in productos if normalizar(p.codigo_interno).startswith(pref)]
            if por_prefijo_cod:
                return por_prefijo_cod, "aproximada"

        # Coincidencia morfológica controlada: lexema compartido, extensión derivativa sobre token completo
        # o tolerancia ortográfica acotada (distancia de edición <= 1 en palabras largas con mismo inicio).
        # Bloquea confusión entre insumo y herramienta agentiva (-dor/-dora), p. ej. 'grapa' != 'grapadora'.
        def _palabra_compatible(w: str, t: str) -> bool:
            if "-" in t or "-" in w:
                # Tolerancia de guiones en siglas/modelos técnicos también en el nombre del producto
                # (ej. usb-c <-> usbc, rj-45 <-> rj45, cat-6 <-> cat6, hc-sr04 <-> hcsr04).
                # Preserva el rechazo estricto de códigos inexistentes distintos (ej. peg-999 != peg-001).
                w_sin_guion = w.replace("-", "")
                t_sin_guion = t.replace("-", "")
                if w_sin_guion == t_sin_guion and len(t_sin_guion) >= 3:
                    return True
                if re.match(r"^[a-z]{2,5}-\d+$", t):
                    return w == t
                return w == t or w.startswith(f"{t}-")
            if w == t:
                return True
            if _es_agentivo(w) != _es_agentivo(t):
                return False
            lw, lt = _lexema(w), _lexema(t)
            if len(lw) >= 3 and lw == lt:
                return True
            if len(t) >= 4 and w.startswith(t):
                return True
            if len(w) >= 5 and len(t) >= 5 and _distancia_edicion_max1(lw, lt):
                if w[:3] == t[:3] or (w[0] in ("s", "c", "z") and t[0] in ("s", "c", "z") and _distancia_edicion_max1(w[1:4], t[1:4])):
                    return True
            if (
                len(w) >= 6
                and len(t) >= 6
                and w[0] == t[0]
                and abs(len(lw) - len(lt)) <= 2
                and not (w != t and t in _sinonimos_de(w))
            ):
                from difflib import SequenceMatcher
                if SequenceMatcher(None, lw, lt).ratio() >= 0.80:
                    return True
            # Equivalencia fonética española (betacismo b<->v, seseo s/c/z, yeísmo ll<->y,
            # g(e,i)<->j, qu/k<->c, h muda preservando 'ch') en palabras significativas.
            if min(len(w), len(t)) >= 3 and max(len(w), len(t)) >= 4:
                fw, ft = clave_fonetica_es(w), clave_fonetica_es(t)
                if fw == ft or clave_fonetica_es(lw) == clave_fonetica_es(lt):
                    return True
                # Compatibilidad con reducción -illo/-illa en _raiz() frente a transcripción yeísta -iyo/-iya (ej. tornillo <-> torniyo)
                if (len(ft) > 5 and ft.endswith(("iyo", "iya")) and ft[:-3] == fw) or (
                    len(fw) > 5 and fw.endswith(("iyo", "iya")) and fw[:-3] == ft
                ):
                    return True
            return False

        aprox = [p for p in productos if asignar_tokens(p, _palabra_compatible)]
        if aprox:
            return aprox, "aproximada"

        # Equivalencia por sinónimo léxico controlado (ej. colbón/pegamento <-> pegante, cosedora <-> grapadora)
        def _palabra_sinonima(w: str, t: str) -> bool:
            for eq in _sinonimos_de(t):
                if _palabra_compatible(w, eq):
                    return True
            return False

        por_sinonimo = [p for p in productos if asignar_tokens(p, _palabra_sinonima)]
        if por_sinonimo:
            return por_sinonimo, "sinonimo"

        # Si el usuario indicó una fracción con unidad implícita (ej. "media pulgada" -> q1d2 pulgada)
        # pero el registro del catálogo guardó solo la fracción sin el texto "pulgada" ni el símbolo "
        # (ej. "Tornillo hexagonal 1/2" -> q1d2), reintentar conservando la fracción exacta y omitiendo
        # únicamente el token de unidad 'pulgada' solo cuando no hubo coincidencia más específica.
        tiene_fraccion_no_entera = any(re.fullmatch(r"q\d+d(?!1\b)\d+", tok) for tok in tokens)
        if tiene_fraccion_no_entera and "pulgada" in tokens and len(tokens) >= 3:
            tokens_sin_pulgada = [tok for tok in tokens if tok != "pulgada"]
            tokens_orig_ref = tokens
            tokens = tokens_sin_pulgada
            try:
                por_frac_impl = [p for p in productos if asignar_tokens(p, _palabra_sinonima)]
                if por_frac_impl:
                    return por_frac_impl, "aproximada"
            finally:
                tokens = tokens_orig_ref

        # Si la consulta fue puramente funcional (ej. «conectar equipo», «para pegar»), buscar por función
        claves_func = set()
        for c in _DESCRIPCIONES_FUNCIONALES:
            claves_func.add(_raiz(c))
            claves_func.add(_lexema(c))
        tokens_especificos = [t for t in tokens_nucleares if t not in claves_func]

        objetivos_func: set[str] = set()
        for t in tokens_brutos:
            for clave_f, lista_f in _DESCRIPCIONES_FUNCIONALES.items():
                if _raiz(clave_f) == t or _lexema(clave_f) == _lexema(t):
                    objetivos_func.update(_raiz(x) for x in lista_f)
        if objetivos_func:
            por_funcion = [
                p for p in productos
                if any(any(_palabra_compatible(w, obj) for w in palabras(p)) for obj in objetivos_func)
                and (not tokens_especificos or all(any(_palabra_compatible(w, t) for w in palabras(p)) for t in tokens_especificos))
            ]
            if por_funcion:
                return por_funcion, "funcional"

        # Si el texto corresponde a una categoría, usarla
        por_cat = [
            p for p in productos
            if p.categoria and all(any(_raiz(w).startswith(t) for w in normalizar(p.categoria.nombre).split()) for t in tokens)
        ]
        if por_cat:
            return por_cat, "categoria"

        # ── NIVEL 3: Coincidencias descriptivas ─────────────────────────────
        # Solo se activa si existen tokens nucleares sustantivos (evita falsos positivos con frases
        # genéricas o puramente funcionales como «conectar equipo» o «aparato»)
        if tokens_nucleares:
            def _palabra_compatible_desc(w: str, t: str) -> bool:
                if w == t:
                    return True
                # Tolerancia de guiones en siglas técnicas/acrónimos (ej. usb-c <-> usbc, rj-45 <-> rj45)
                if "-" in w or "-" in t:
                    if w.replace("-", "") == t.replace("-", "") and len(t) >= 3:
                        return True
                return _palabra_compatible(w, t)

            def palabras_desc(p) -> List[str]:
                desc = getattr(p, "descripcion", "") or ""
                if not desc:
                    return []
                desc_pre = preprocesar_texto_inventario(desc)
                desc_norm = normalizar(desc_pre)
                return [_raiz(w) for w in desc_norm.split() if w not in _PALABRAS_VACIAS and len(w) > 1]

            def palabras_totales(p) -> List[str]:
                return palabras(p) + palabras_desc(p)

            descriptivos = [
                p for p in productos
                if any(any(_palabra_compatible_desc(w, t) for w in palabras_desc(p)) for t in tokens_nucleares)
                and all(any(_palabra_compatible_desc(w, t) for w in palabras_totales(p)) for t in tokens_nucleares)
            ]
            if descriptivos:
                return descriptivos, "descriptiva"

        # ── NIVEL 4: Equivalencia fonética de unidades B<->V en STT (ej. «20B» / «20 be» <-> «20V», «9B» <-> «9V») ──
        # Solo se activa cuando las búsquedas literales/aproximadas devolvieron 0 resultados, preservando
        # coincidencias literales como «Lápiz 2B» si existen en el catálogo.
        if _permitir_fonetica and texto:
            variantes_fon = []
            # «20B», «20 b», «20 be», «20 ve», «20 volt» -> «20v»
            t_b_a_v = re.sub(
                r"(?<![\w-])(\d+(?:[.,]\d+)?)\s*(?:b|be|ve)\b",
                r"\1v",
                texto,
                flags=re.IGNORECASE,
            )
            if t_b_a_v != texto:
                variantes_fon.append(t_b_a_v)
            # Inverso: «2V» -> «2b» (ej. «lápiz 2V» -> «Lápiz 2B»)
            t_v_a_b = re.sub(
                r"(?<![\w-])(\d+(?:[.,]\d+)?)\s*(?:v|ve)\b",
                r"\1b",
                texto,
                flags=re.IGNORECASE,
            )
            if t_v_a_b != texto and t_v_a_b not in variantes_fon:
                variantes_fon.append(t_v_a_b)
            for variante in variantes_fon:
                prods_fon, _ = self._filtrar_por_texto(productos, variante, _permitir_fonetica=False)
                if prods_fon:
                    return prods_fon, "aproximada"

        return [], "exacta"

    def _buscar_relacionados(self, productos, texto: str):
        """Busca productos conceptualmente relacionados (ej. grapadora -> grapas) sin confundirlos con el producto pedido."""
        tokens = tokens_significativos(texto)
        if not tokens:
            return []
        raices_rel: set[str] = set()
        for t in tokens:
            for clave_r, destinos_r in _RELACIONES_CONCEPTUALES.items():
                if _raiz(clave_r) == t or _lexema(clave_r) == _lexema(t):
                    raices_rel.update(_raiz(d) for d in destinos_r)
        if not raices_rel:
            return []

        def palabras(p):
            return [_raiz(w) for w in normalizar(f"{p.nombre} {p.codigo_interno}").split()]

        return [
            p for p in productos
            if any(r in palabras(p) for r in raices_rel)
        ]

    @staticmethod
    def estado_stock(disponible: int, stock_minimo: int, umbral: Optional[int] = None) -> str:
        """Clasifica el stock disponible en uno de los 5 estados primarios mutuamente excluyentes:
        - 'agotado': disponible == 0
        - 'bajo_minimo': stock_minimo > 0 y 0 < disponible < stock_minimo
        - 'requiere_reposicion': agotado o por debajo del mínimo
        - 'en_minimo': stock_minimo > 0 y disponible == stock_minimo
        - 'cerca_minimo': stock_minimo > 0 y stock_minimo < disponible <= stock_minimo + margen
        - 'suficiente': (stock_minimo > 0 y disponible > stock_minimo + margen) o (stock_minimo == 0 y disponible > 0)
        """
        margen = umbral if umbral is not None else MARGEN_PROXIMIDAD_STOCK
        if disponible <= 0:
            return "agotado"
        if stock_minimo > 0:
            if disponible < stock_minimo:
                return "bajo_minimo"
            if disponible == stock_minimo:
                return "en_minimo"
            if disponible <= stock_minimo + margen:
                return "cerca_minimo"
        return "suficiente"

    @classmethod
    def _cumple_condicion(cls, item: Dict[str, Any], condicion: str, umbral: Optional[int]) -> bool:
        disp = item["disponible"]
        minimo = item["stock_minimo"]

        if condicion == "bajo_minimo":
            # Estados primarios disjuntos: los agotados se cuentan solo como
            # agotados, aunque matemáticamente su disponibilidad sea menor al mínimo.
            return minimo > 0 and 0 < disp < minimo
        if condicion == "requiere_reposicion":
            # Necesidad de abastecimiento incluye agotados y productos con
            # disponibilidad positiva por debajo de un mínimo configurado.
            return disp <= 0 or (minimo > 0 and disp < minimo)
        if condicion in ESTADOS_STOCK_PRIMARIOS:
            return cls.estado_stock(disp, minimo, umbral) == condicion
        if condicion == "con_stock":
            return disp > 0
        if condicion == "minimo_cero":
            return minimo == 0
        if condicion == "con_prestamos":
            return item["en_prestamo"] > 0
        return True

    def _serializar(self, p, umbral: Optional[int] = None) -> Dict[str, Any]:
        exist = getattr(p, "exist_visibles", [])
        ubic = []
        if getattr(p, "ubicacion", None):
            u = p.ubicacion
            if (
                u.activo
                and not u.eliminado
                and u.almacen.activo
                and not u.almacen.eliminado
                and (u.almacen_id in self._ids_almacenes or u.almacen_id is None)
            ):
                ubic.append({"nombre": u.nombre, "almacen": u.almacen.nombre if u.almacen_id else None})
        elif hasattr(p, "ubic_rel"):
            ubic = [
                {"nombre": r.ubicacion.nombre, "almacen": r.ubicacion.almacen.nombre if r.ubicacion.almacen_id else None}
                for r in getattr(p, "ubic_rel", [])
                if (r.ubicacion.almacen_id in self._ids_almacenes or r.ubicacion.almacen_id is None)
            ]
        disp = sum(e.stock_disponible for e in exist)
        en_prest = sum(e.stock_en_prestamo for e in exist)
        return {
            "id": p.id,
            "codigo": p.codigo_interno,
            "nombre": p.nombre,
            "descripcion": getattr(p, "descripcion", "") or "",
            "categoria": p.categoria.nombre if p.categoria_id else None,
            "stock_minimo": p.stock_minimo,
            "disponible": disp,
            "en_prestamo": en_prest,
            "estado_stock": self.estado_stock(disp, p.stock_minimo, umbral),
            "almacenes": [
                {"nombre": e.almacen.nombre, "disponible": e.stock_disponible, "en_prestamo": e.stock_en_prestamo}
                for e in exist
            ],
            "ubicaciones": ubic,
            "es_prestable": p.es_prestable,
            "empaque": (f"{p.nombre_empaque} x{p.unidades_por_empaque}" if p.nombre_empaque and p.unidades_por_empaque else None),
            "codigo_barras": p.codigo_barras or None,
            "activo": p.activo and not p.archivado,
        }

    # ── Herramienta: movimientos recientes de un producto ──────────────────
    def movimientos_producto(
        self,
        producto_id: int,
        limite: int = 5,
        almacen: Optional[str] = None,
        tipo: Optional[str] = None,
        destinatario: Optional[str] = None,
        orden: Optional[str] = None,
        fecha_desde: Optional[str] = None,
        fecha_hasta: Optional[str] = None,
    ) -> ListaResultadoHerramienta:
        from inventario.models import Movimiento
        from datetime import date, datetime, time, timedelta
        from django.utils import timezone

        if not self.tiene_permiso("movimientos.ver"):
            return ListaResultadoHerramienta(error="permiso_denegado", permiso_requerido="movimientos.ver")

        almacenes_ids = set(self._ids_almacenes)
        if almacen:
            alm = self._resolver_almacen(almacen)
            if alm and alm.id in almacenes_ids:
                almacenes_ids = {alm.id}

        qs = (
            Movimiento.objects.select_related("prestamo")
            .filter(producto_id=producto_id)
            .filter(Q(almacen_origen_id__in=almacenes_ids) | Q(almacen_destino_id__in=almacenes_ids))
            .filter(estado=Movimiento.Estado.CONFIRMADO)
            .exclude(tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL)
        )

        fechas = {}
        try:
            if fecha_desde:
                fechas["desde"] = date.fromisoformat(str(fecha_desde))
            if fecha_hasta:
                fechas["hasta"] = date.fromisoformat(str(fecha_hasta))
            if fechas.get("desde") and fechas.get("hasta") and fechas["desde"] > fechas["hasta"]:
                return ListaResultadoHerramienta(error="rango_temporal_invalido")
            tz = timezone.get_current_timezone()
            if fechas.get("desde"):
                inicio = timezone.make_aware(datetime.combine(fechas["desde"], time.min), tz)
                qs = qs.filter(fecha_hora__gte=inicio)
            if fechas.get("hasta"):
                fin_exclusivo = fechas["hasta"] + timedelta(days=1)
                fin = timezone.make_aware(datetime.combine(fin_exclusivo, time.min), tz)
                qs = qs.filter(fecha_hora__lt=fin)
        except (TypeError, ValueError, OverflowError):
            return ListaResultadoHerramienta(error="rango_temporal_invalido")

        tipo_valido = {
            codigo for codigo, _ in Movimiento.Tipo.choices
            if codigo != Movimiento.Tipo.MOVIMIENTO_INICIAL
        }
        if tipo in tipo_valido:
            qs = qs.filter(tipo=tipo)
        else:
            tipo = None

        if destinatario:
            qs = qs.filter(destinatario__icontains=str(destinatario).strip())

        def _fmt_dt_local(dt) -> str:
            if not dt:
                return ""
            if timezone.is_aware(dt):
                dt = timezone.localtime(dt)
            return dt.strftime("%d/%m/%Y %H:%M")

        def _estado_prestamo_mov(mov_obj) -> Dict[str, Any]:
            try:
                prest = mov_obj.prestamo
            except Exception:
                prest = None
            if prest is None:
                return {}
            return {
                "estado_prestamo": prest.estado,
                "pendiente_devolucion": int(prest.cantidad_pendiente_devolucion or 0),
                "fecha_devolucion": _fmt_dt_local(prest.fecha_cierre) if prest.fecha_cierre else None,
            }

        if orden in {"asc", "desc"}:
            evento = qs.order_by("fecha_hora" if orden == "asc" else "-fecha_hora").first()
            resultado = ListaResultadoHerramienta(
                [] if evento is None else [{
                    "tipo": evento.get_tipo_display() if hasattr(evento, "get_tipo_display") else evento.tipo,
                    "cantidad": evento.cantidad,
                    "fecha": _fmt_dt_local(evento.fecha_hora),
                    "destinatario": evento.destinatario or None,
                    "stock_disponible_anterior": evento.stock_disponible_anterior,
                    "stock_disponible_posterior": evento.stock_disponible_posterior,
                    **_estado_prestamo_mov(evento),
                }]
            )
            resultado.total_eventos = 1 if evento is not None else 0
            resultado.unidades_evento = evento.cantidad if evento is not None else 0
            return resultado

        total_eventos = None
        unidades_evento = None
        if tipo or fechas:
            from django.db.models import Sum
            agregado = qs.aggregate(total=Count("pk"), unidades=Sum("cantidad"))
            total_eventos = agregado["total"] or 0
            unidades_evento = agregado["unidades"] or 0

        consulta = qs.order_by("-fecha_hora")
        if limite is not None:
            consulta = consulta[:max(0, int(limite))]
        resultado = ListaResultadoHerramienta(
            {
                "tipo": m.get_tipo_display() if hasattr(m, "get_tipo_display") else m.tipo,
                "cantidad": m.cantidad,
                "fecha": _fmt_dt_local(m.fecha_hora),
                "destinatario": m.destinatario or None,
                "stock_disponible_anterior": m.stock_disponible_anterior,
                "stock_disponible_posterior": m.stock_disponible_posterior,
                **_estado_prestamo_mov(m),
            }
            for m in consulta
        )
        resultado.total_eventos = total_eventos
        resultado.unidades_evento = unidades_evento
        resultado.periodo_descripcion = (
            f"entre {fechas['desde'].strftime('%d/%m/%Y')} y {fechas['hasta'].strftime('%d/%m/%Y')}"
            if fechas.get("desde") and fechas.get("hasta")
            else f"desde {fechas['desde'].strftime('%d/%m/%Y')}"
            if fechas.get("desde")
            else f"hasta {fechas['hasta'].strftime('%d/%m/%Y')}"
            if fechas.get("hasta")
            else None
        )
        return resultado

    # ── Herramienta: movimientos globales con alcance RBAC y filtros ───────
    def buscar_movimientos(
        self,
        limite: int = 20,
        almacen: Optional[str] = None,
        categoria: Optional[str] = None,
        tipo: Optional[str] = None,
        destinatario: Optional[str] = None,
        orden: Optional[str] = None,
        fecha_desde: Optional[str] = None,
        fecha_hasta: Optional[str] = None,
    ) -> ListaResultadoHerramienta:
        """Consulta movimientos del inventario visible respetando almacenes y permisos."""
        from datetime import date, datetime, time, timedelta
        from django.db.models import Sum
        from django.utils import timezone
        from inventario.models import Movimiento

        if not self.tiene_permiso("movimientos.ver"):
            return ListaResultadoHerramienta(error="permiso_denegado", permiso_requerido="movimientos.ver")

        almacenes_ids = set(self._ids_almacenes)
        almacen_filtrado = None
        if almacen:
            alm = self._resolver_almacen(almacen)
            if alm is None:
                res_err = ListaResultadoHerramienta(error="almacen_inexistente")
                res_err.almacen_filtrado = almacen
                return res_err
            if alm.id not in almacenes_ids:
                res_err = ListaResultadoHerramienta(error="almacen_no_autorizado", permiso_requerido="almacen.asignado")
                res_err.almacen_filtrado = alm.nombre
                return res_err
            almacenes_ids = {alm.id}
            almacen_filtrado = alm.nombre

        if not almacenes_ids:
            resultado_vacio = ListaResultadoHerramienta([], total_eventos=0, unidades_evento=0)
            resultado_vacio.periodo_descripcion = None
            resultado_vacio.almacen_filtrado = almacen_filtrado
            return resultado_vacio

        qs = (
            Movimiento.objects.select_related("producto", "almacen_origen", "almacen_destino", "prestamo")
            .filter(Q(almacen_origen_id__in=almacenes_ids) | Q(almacen_destino_id__in=almacenes_ids))
            .filter(estado=Movimiento.Estado.CONFIRMADO)
            .exclude(tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL)
        )

        if categoria:
            from inventario.models import Categoria

            cat_norm = normalizar(categoria)
            cats_qs = Categoria.objects.filter(activo=True, eliminado=False)
            if not self.es_admin:
                cats_qs = cats_qs.filter(
                    Q(almacen_principal_id__in=almacenes_ids)
                    | Q(productos__existencias__almacen_id__in=almacenes_ids)
                ).distinct()
            cats_activas = list(cats_qs)
            cat_obj = next((c for c in cats_activas if normalizar(c.nombre) == cat_norm), None)
            if cat_obj is None and cat_norm:
                coincidencias_cat = [
                    c for c in cats_activas
                    if cat_norm in normalizar(c.nombre) or normalizar(c.nombre) in cat_norm
                ]
                if len(coincidencias_cat) == 1:
                    cat_obj = coincidencias_cat[0]
            if cat_obj is None:
                res_cat = ListaResultadoHerramienta(error="categoria_inexistente")
                res_cat.categoria_sin_coincidencia = categoria
                return res_cat
            qs = qs.filter(producto__categoria_id=cat_obj.id)

        fechas = {}
        try:
            if fecha_desde:
                fechas["desde"] = date.fromisoformat(str(fecha_desde))
            if fecha_hasta:
                fechas["hasta"] = date.fromisoformat(str(fecha_hasta))
            if fechas.get("desde") and fechas.get("hasta") and fechas["desde"] > fechas["hasta"]:
                return ListaResultadoHerramienta(error="rango_temporal_invalido")
            tz = timezone.get_current_timezone()
            if fechas.get("desde"):
                inicio = timezone.make_aware(datetime.combine(fechas["desde"], time.min), tz)
                qs = qs.filter(fecha_hora__gte=inicio)
            if fechas.get("hasta"):
                fin_exclusivo = fechas["hasta"] + timedelta(days=1)
                fin = timezone.make_aware(datetime.combine(fin_exclusivo, time.min), tz)
                qs = qs.filter(fecha_hora__lt=fin)
        except (TypeError, ValueError, OverflowError):
            return ListaResultadoHerramienta(error="rango_temporal_invalido")

        tipos_validos = {
            codigo for codigo, _ in Movimiento.Tipo.choices
            if codigo != Movimiento.Tipo.MOVIMIENTO_INICIAL
        }
        if tipo in tipos_validos:
            qs = qs.filter(tipo=tipo)
        elif tipo == "TRASLADO":
            qs = qs.filter(tipo__in=[Movimiento.Tipo.TRASLADO_INTERNO, Movimiento.Tipo.TRASLADO_ENTRE])
        else:
            tipo = None

        if destinatario:
            qs = qs.filter(destinatario__icontains=str(destinatario).strip())

        agregado = qs.aggregate(total=Count("pk"), unidades=Sum("cantidad"))
        total_eventos = int(agregado["total"] or 0)
        unidades_evento = int(agregado["unidades"] or 0)

        consulta = qs.order_by("fecha_hora" if orden == "asc" else "-fecha_hora")
        if limite is not None:
            consulta = consulta[:max(0, int(limite))]

        def _fmt_dt_local(dt) -> str:
            if not dt:
                return ""
            if timezone.is_aware(dt):
                dt = timezone.localtime(dt)
            return dt.strftime("%d/%m/%Y %H:%M")

        def _estado_prestamo_mov(mov_obj) -> Dict[str, Any]:
            try:
                prest = mov_obj.prestamo
            except Exception:
                prest = None
            if prest is None:
                return {}
            return {
                "estado_prestamo": prest.estado,
                "pendiente_devolucion": int(prest.cantidad_pendiente_devolucion or 0),
                "fecha_devolucion": _fmt_dt_local(prest.fecha_cierre) if prest.fecha_cierre else None,
            }

        resultado = ListaResultadoHerramienta(
            {
                "id": str(m.id),
                "producto_id": m.producto_id,
                "producto_codigo": m.producto.codigo_interno if m.producto_id else "",
                "producto_nombre": m.producto.nombre if m.producto_id else "",
                "tipo": m.get_tipo_display() if hasattr(m, "get_tipo_display") else m.tipo,
                "tipo_codigo": m.tipo,
                "cantidad": m.cantidad,
                "fecha": _fmt_dt_local(m.fecha_hora),
                "destinatario": m.destinatario or None,
                "almacen": (
                    m.almacen_origen.nombre if m.almacen_origen_id
                    else (m.almacen_destino.nombre if m.almacen_destino_id else "")
                ),
                "stock_disponible_anterior": m.stock_disponible_anterior,
                "stock_disponible_posterior": m.stock_disponible_posterior,
                **_estado_prestamo_mov(m),
            }
            for m in consulta
        )
        resultado.total_eventos = total_eventos
        resultado.unidades_evento = unidades_evento
        resultado.almacen_filtrado = almacen_filtrado
        if fechas.get("desde") and fechas.get("hasta"):
            if fechas["desde"] == fechas["hasta"]:
                resultado.periodo_descripcion = f"el {fechas['desde'].strftime('%d/%m/%Y')}"
            else:
                resultado.periodo_descripcion = (
                    f"entre {fechas['desde'].strftime('%d/%m/%Y')} y {fechas['hasta'].strftime('%d/%m/%Y')}"
                )
        elif fechas.get("desde"):
            resultado.periodo_descripcion = f"desde {fechas['desde'].strftime('%d/%m/%Y')}"
        elif fechas.get("hasta"):
            resultado.periodo_descripcion = f"hasta {fechas['hasta'].strftime('%d/%m/%Y')}"
        else:
            resultado.periodo_descripcion = None
        return resultado

    # ── Herramienta: préstamos activos de un producto ──────────────────────
    def prestamos_activos(self, producto_id: int, limite: int = 5, almacen: Optional[str] = None) -> ListaResultadoHerramienta:
        """Consulta los préstamos activos reales del producto."""
        from inventario.models import Prestamo

        if not self.tiene_permiso("prestamos.ver", "movimientos.ver", "movimientos.prestamo"):
            return ListaResultadoHerramienta(error="permiso_denegado", permiso_requerido="prestamos.ver")

        almacenes_ids = set(self._ids_almacenes)
        if almacen:
            alm = self._resolver_almacen(almacen)
            if alm and alm.id in almacenes_ids:
                almacenes_ids = {alm.id}

        qs = (
            Prestamo.objects.filter(producto_id=producto_id, almacen_id__in=almacenes_ids)
            .filter(estado__in=["ABIERTO", "DEVUELTO_PARCIAL"], cantidad_pendiente_devolucion__gt=0)
            .order_by("-fecha_creacion")[:limite]
        )
        from django.utils import timezone as _tz
        return ListaResultadoHerramienta(
            {
                "destinatario": p.destinatario,
                "cantidad": p.cantidad_pendiente_devolucion,
                "almacen": p.almacen.nombre if p.almacen_id else "",
                "fecha": (
                    _tz.localtime(p.fecha_creacion).strftime("%d/%m/%Y")
                    if (p.fecha_creacion and _tz.is_aware(p.fecha_creacion))
                    else (p.fecha_creacion.strftime("%d/%m/%Y") if p.fecha_creacion else "")
                ),
            }
            for p in qs
        )

    def sumar_unidades_pendientes_devolucion(
        self,
        productos_ids: List[int],
        almacen: Optional[str] = None,
    ) -> tuple[Optional[int], Optional[str]]:
        """Suma el saldo pendiente de préstamos abiertos para productos visibles.

        La fuente canónica es Prestamo.cantidad_pendiente_devolucion, no el
        agregado de existencias. Así el total sigue siendo correcto si ambos
        saldos quedaron temporalmente desincronizados.
        """
        from django.db.models import Sum
        from inventario.models import Prestamo

        if not self.tiene_permiso("prestamos.ver", "movimientos.ver", "movimientos.prestamo"):
            return None, "permiso_denegado"

        ids = {int(producto_id) for producto_id in productos_ids if producto_id is not None}
        if not ids:
            return 0, None

        almacenes_ids = set(self._ids_almacenes)
        if almacen:
            almacen_obj = self._resolver_almacen(almacen)
            if almacen_obj is None:
                return None, "almacen_inexistente"
            if almacen_obj.id not in almacenes_ids:
                return None, "almacen_no_autorizado"
            almacenes_ids = {almacen_obj.id}

        saldo = (
            Prestamo.objects.filter(
                producto_id__in=ids,
                almacen_id__in=almacenes_ids,
                estado__in=["ABIERTO", "DEVUELTO_PARCIAL"],
                cantidad_pendiente_devolucion__gt=0,
            ).aggregate(total=Sum("cantidad_pendiente_devolucion"))["total"]
            or 0
        )
        return int(saldo), None
