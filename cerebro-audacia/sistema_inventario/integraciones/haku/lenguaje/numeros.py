"""Parser determinista de números cardinales en español (dígitos y palabras, 0..999_999)."""

from __future__ import annotations

import re
import unicodedata

_UNIDADES_ES = {
    "un": 1, "una": 1, "uno": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}
_ESPECIALES_ES = {
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintiun": 21, "veintiuna": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}
_DECENAS_ES = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
}
_CENTENAS_ES = {
    "cien": 100, "ciento": 100,
    "doscientos": 200, "doscientas": 200,
    "trescientos": 300, "trescientas": 300,
    "cuatrocientos": 400, "cuatrocientas": 400,
    "quinientos": 500, "quinientas": 500,
    "seiscientos": 600, "seiscientas": 600,
    "setecientos": 700, "setecientas": 700,
    "ochocientos": 800, "ochocientas": 800,
    "novecientos": 900, "novecientas": 900,
}
_PALABRAS_NUM_ES = (
    {"cero", "mil"}
    | set(_UNIDADES_ES)
    | set(_ESPECIALES_ES)
    | set(_DECENAS_ES)
    | set(_CENTENAS_ES)
)


def _normalizar_sin_tildes(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (texto or "").lower())
        if not unicodedata.combining(c)
    )


def _parsear_tokens_numero_es(tokens: list[str], inicio: int) -> tuple[int | None, int, bool]:
    """Parsea un cardinal español (0..999_999) desde tokens[inicio].

    Retorna (valor, indice_fin_exclusivo, es_articulo_un).
    """
    n = len(tokens)
    if inicio >= n or tokens[inicio] not in _PALABRAS_NUM_ES:
        return None, inicio, False

    if tokens[inicio] == "cero":
        return 0, inicio + 1, False

    i = inicio
    total = 0
    parcial = 0
    consumidos = 0

    while i < n:
        tok = tokens[i]
        if tok in _CENTENAS_ES:
            parcial += _CENTENAS_ES[tok]
            consumidos += 1
            i += 1
        elif tok in _DECENAS_ES:
            parcial += _DECENAS_ES[tok]
            consumidos += 1
            i += 1
            if i + 1 < n and tokens[i] == "y" and tokens[i + 1] in _UNIDADES_ES:
                parcial += _UNIDADES_ES[tokens[i + 1]]
                consumidos += 2
                i += 2
        elif tok in _ESPECIALES_ES:
            parcial += _ESPECIALES_ES[tok]
            consumidos += 1
            i += 1
        elif tok in _UNIDADES_ES:
            parcial += _UNIDADES_ES[tok]
            consumidos += 1
            i += 1
        elif tok == "mil":
            total += (parcial if parcial > 0 else 1) * 1000
            parcial = 0
            consumidos += 1
            i += 1
        else:
            break

    if consumidos == 0:
        return None, inicio, False

    valor = total + parcial
    es_articulo_un = (
        consumidos == 1
        and tokens[inicio] in ("un", "una")
        and i < n
    )
    return valor, i, es_articulo_un


def extraer_numero_espanol(
    texto: str,
    *,
    permitir_cero: bool = True,
    detectar_negativo: bool = False,
    solo_al_inicio: bool = False,
    solo_numero_exacto: bool = False,
) -> int | None:
    """Extrae un número entero en dígitos o palabras en español (0..999_999).

    Parámetros:
    - permitir_cero: si False, devuelve None cuando el número resuelto es 0.
    - detectar_negativo: si True, devuelve valores negativos (-5, "menos cinco");
      si False, un número negativo explícito retorna None en vez de convertirse a positivo.
    - solo_al_inicio: solo acepta un número al comienzo del fragmento.
    - solo_numero_exacto: solo acepta si TODO el fragmento es una expresión numérica pura.
    """
    if not texto:
        return None

    t_norm = _normalizar_sin_tildes(texto).strip()
    if not t_norm:
        return None

    # Modismos cuantitativos universales del español hablado (p. ej. "media docena" -> 6, "una docena" -> 12, "un par de" -> 2)
    t_norm = re.sub(r"\bmedia\s+docena\b", "6", t_norm)
    t_norm = re.sub(r"\b(?:una|1)\s+docena\b", "12", t_norm)
    t_norm = re.sub(
        r"\b(dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|\d+)\s+docenas\b",
        lambda m: str((int(m.group(1)) if m.group(1).isdigit() else _UNIDADES_ES.get(m.group(1), _ESPECIALES_ES.get(m.group(1), 1))) * 12),
        t_norm,
    )
    t_norm = re.sub(r"\bun\s+par(?:\s+de\b|$)", lambda m: "2 de" if "de" in m.group(0) else "2", t_norm)

    if solo_numero_exacto:
        t_clean = re.sub(r"[^\w\s-]", " ", t_norm)
        t_clean = re.sub(r"\s+", " ", t_clean).strip()
        if not t_clean:
            return None
        m_neg_d = re.fullmatch(r"(?:-\s*|menos\s+|negativo\s+)(\d+)", t_clean)
        if m_neg_d:
            val = -int(m_neg_d.group(1))
            return val if detectar_negativo else None
        m_d = re.fullmatch(r"(\d+)", t_clean)
        if m_d:
            val = int(m_d.group(1))
            return val if (val > 0 or permitir_cero) else None
        tokens = re.findall(r"[a-z]+", t_clean)
        if not tokens:
            return None
        es_neg = False
        idx = 0
        if tokens[0] in ("menos", "negativo") and len(tokens) > 1:
            es_neg = True
            idx = 1
        val, fin, _ = _parsear_tokens_numero_es(tokens, idx)
        if val is None or fin != len(tokens):
            return None
        if es_neg:
            return -val if detectar_negativo else None
        return val if (val > 0 or permitir_cero) else None

    if solo_al_inicio:
        m_neg_d = re.match(r"^(?:-\s*|menos\s+|negativo\s+)(\d+)\b", t_norm)
        if m_neg_d:
            val = -int(m_neg_d.group(1))
            return val if detectar_negativo else None
        m_d = re.match(r"^(\d+)\b", t_norm)
        if m_d:
            val = int(m_d.group(1))
            return val if (val > 0 or permitir_cero) else None
        tokens = re.findall(r"[a-z]+", t_norm)
        if not tokens:
            return None
        es_neg = False
        idx = 0
        if tokens[0] in ("menos", "negativo") and len(tokens) > 1:
            es_neg = True
            idx = 1
        val, _, _ = _parsear_tokens_numero_es(tokens, idx)
        if val is None:
            return None
        if es_neg:
            return -val if detectar_negativo else None
        return val if (val > 0 or permitir_cero) else None

    # 1. Negativos en dígitos o "menos <dígitos>"
    m_neg_d = re.search(r"(?<!\w)-\s*(\d+)\b|\b(?:menos|negativo)\s+(\d+)\b", t_norm)
    if m_neg_d:
        dig = m_neg_d.group(1) or m_neg_d.group(2)
        return -int(dig) if detectar_negativo else None

    # 2. Dígitos positivos o cero
    m_d = re.search(r"\b(\d+)\b", t_norm)
    if m_d:
        val = int(m_d.group(1))
        if val > 0 or permitir_cero:
            return val

    # 3. Palabras numéricas en español (cardinales simples y compuestos)
    tokens = re.findall(r"[a-z]+", t_norm)
    candidatos: list[tuple[int, bool, bool]] = []
    i = 0
    while i < len(tokens):
        es_neg = i > 0 and tokens[i - 1] in ("menos", "negativo")
        val, fin, es_art = _parsear_tokens_numero_es(tokens, i)
        if val is not None:
            candidatos.append((val, es_art, es_neg))
            i = fin
        else:
            i += 1

    if not candidatos:
        return None

    elegido = next((c for c in candidatos if not c[1]), candidatos[0])
    val, _, es_neg = elegido
    if es_neg:
        return -val if detectar_negativo else None
    if val == 0 and not permitir_cero:
        return None

    return val


# Patrones estructurales que contienen dígitos o artículos pero NO son cantidades de operación
_RE_CODIGO_PRODUCTO = re.compile(r"\b[a-z]{2,8}(?:[-_]\d{1,6}|\s+0\d{1,5})\b", re.IGNORECASE)
_RE_FECHA_HORA = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b\d{1,2}:\d{2}\b")
_RE_FRACCION = re.compile(r"\b\d+\s*/\s*\d+\b")
_RE_FRACCION_HABLADA = re.compile(
    r"\b(?:(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|\d+)\s+y\s+)?"
    r"(?:(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|\d+)\s+)?"
    r"(?:tercios?|tercias?|cuartos?|cuartas?|quintos?|quintas?|sextos?|sextas?|"
    r"octavos?|octavas?|decimos?|decimas?|"
    r"dieciseisavos?|dieciseisavas?|dieciseis\s+avos?|16\s*avos?|"
    r"treintaidosavos?|treintaidosavas?|treinta\s+y\s+dos\s+avos?|treintaidos\s+avos?|32\s*avos?|"
    r"sesentaicuatroavos?|sesenta\s+y\s+cuatro\s+avos?|sesentaicuatro\s+avos?|64\s*avos?)"
    r"(?:\s+de\s+pulgadas?)?\b|"
    r"\b(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|\d+)\s+y\s+medi[oa](?:\s+pulgadas?)?\b|"
    r"\b(?:un\s+medio|medi[oa])\s+(?:pulgadas?|metros?|centi\s*metros?|mili\s*metros?|litros?|kilos?|kilo\s*gramos?|libras?)\b|"
    r"\bde\s+(?:un\s+medio|medi[oa])(?!\s+(?:docenas?|horas?)\b)\b|"
    r"\b(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|\d+)\s+(?:sobre|barra|entre)\s+"
    r"(?:dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|dieciseis|treinta\s+y\s+dos|sesenta\s+y\s+cuatro|\d+)\b|"
    r"\b(?:(?:una|uno|un|dos|tres|cuatro|cinco|\d+)\s+)?(?:pulgadas?|metros?|centi\s*metros?|mili\s*metros?|litros?|kilos?|kilo\s*gramos?|libras?)\s+y\s+(?:medi[oa]|(?:un\s+|tres\s+)?cuartos?)\b",
    re.IGNORECASE,
)
_PAT_NUM_SPEC = (
    r"(?:\d+|cero|"
    r"(?:cien|ciento|(?:dos|tres|cuatro|quini|seis|sete|ocho|nove)cient[oa]s)|"
    r"(?:treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa)(?:\s+y\s+(?:uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve))?|"
    r"veintiuno|veintiuna|veintiun|veintidos|veintitres|veinticuatro|veinticinco|veintiseis|veintisiete|veintiocho|veintinueve|veinte|"
    r"dieciseis|diecisiete|dieciocho|diecinueve|diez|once|doce|trece|catorce|quince|"
    r"uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)"
)
_RE_DECIMAL_HABLADO_SPEC = re.compile(
    rf"\b(?:{_PAT_NUM_SPEC}\s+)?(?:punto|coma)\s+(?:cero\s+)*{_PAT_NUM_SPEC}\b",
    re.IGNORECASE,
)
_RE_SPEC_HABLADA_CON_DE = re.compile(
    rf"\bde\s+{_PAT_NUM_SPEC}\s+(?:"
    r"ohms?|ohmios?|(?:kilo|k)\s*ohmios?|(?:kilo|k)\s*ohms?|(?:mega|m)\s*ohmios?|(?:mega|m)\s*ohms?|"
    r"voltios?|volts?|mili\s*voltios?|kilo\s*voltios?|"
    r"vatios?|watios?|watts?|kilo\s*vatios?|mili\s*vatios?|amperios?|mili\s*amperios?|"
    r"hercios?|hertz|kilo\s*hercios?|mega\s*hercios?|giga\s*hercios?|"
    r"micro\s*faradios?|nano\s*faradios?|pico\s*faradios?|"
    r"gigas?|megas?|teras?|giga\s*bytes?|mega\s*bytes?|tera\s*bytes?|kilo\s*bytes?|"
    r"por\s+ciento|revoluciones(?:\s+por\s+minuto)?|"
    r"pulgadas?|metros?|centi\s*metros?|mili\s*metros?(?:\s+(?:al\s+)?cuadrados?)?|litros?|mili\s*litros?|"
    r"gramos?(?:\s+(?:por\s+)?metros?\s+cuadrados?)?|kilo\s*gramos?|kilos?|libras?|onzas?|bares?|psi|kilo\s*pascales?|"
    r"puertos?|canales?|polos?|pines?|hilos?|puntas?|bits?|hojas?"
    r")\b",
    re.IGNORECASE,
)
_RE_ESPECIFICACION_TECNICA = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:"
    r"v|kv|mv|b|w|mw|kw|a|ma|hz|khz|mhz|ghz|mm[2²]|mm\s+al\s+cuadrado|mm|cm|m|km|in|ft|g/m[2²]|gr/m[2²]|gsm|g|kg|mg|l|ml|oz|"
    r"gb|mb|kb|tb|rpm|%|bar|psi|kpa|awg|uf|nf|pf|(?:k|m)\s*ohms?|(?:k|m)\s*ohmios?|"
    r"ohms?|ohmios?|kilo\s*ohmios?|mega\s*ohmios?|voltios?|volts?|mili\s*voltios?|kilo\s*voltios?|"
    r"vatios?|watios?|watts?|kilo\s*vatios?|mili\s*vatios?|"
    r"amperios?|mili\s*amperios?|hercios?|hertz|kilo\s*hercios?|mega\s*hercios?|giga\s*hercios?|"
    r"micro\s*faradios?|nano\s*faradios?|pico\s*faradios?|"
    r"gigas?|megas?|teras?|giga\s*bytes?|mega\s*bytes?|tera\s*bytes?|kilo\s*bytes?|"
    r"por\s+ciento|revoluciones(?:\s+por\s+minuto)?|doble\s+(?:u|ve|b)|"
    r"pulgadas?|metros?|centi\s*metros?|mili\s*metros?(?:\s+(?:al\s+)?cuadrados?)?|litros?|mili\s*litros?|"
    r"gramos?(?:\s+(?:por\s+)?metros?\s+cuadrados?)?|kilo\s*gramos?|hojas?|"
    r"puertos?|canales?|polos?|pines?|hilos?|puntas?|bits?"
    r")\b|"
    r"\b\d+(?:[.,]\d+)?\s*(?:k?[ωΩ]|[μµ]f|[μµ]m|\"|″)",
    re.IGNORECASE,
)
_RE_ESTANDAR_UBICACION = re.compile(
    r"\b(?:cat|rj|usb|hdmi|vga|iso|ip|estante|cajon|vitrina|pasillo|gaveta|almacen|bodega|"
    r"repisa|nivel|fila|columna|bloque|zona|piso|lote|serie|modelo|tipo|grado|calibre|clase|version)"
    r"\s*[-_#]?\s*[a-z]?\s*\d+[a-z]?\b",
    re.IGNORECASE,
)
_RE_ARTICULO_OPERACION = re.compile(
    r"\b(?:un|una)\s+(?:nuev[oa]\s+)?"
    r"(?:prestamo|entrada|salida|devolucion|traslado|dotacion|movimiento|ingreso|retiro|"
    r"baja|conteo|ajuste|producto|articulo|operacion|registro|cambio|favor|momento|segundo|poco)\b",
    re.IGNORECASE,
)
_RE_ARTICULO_PREPOSICIONAL = re.compile(
    r"\b(?:en|de|a|al|por|con|para|sobre|desde|hacia)\s+(?:un|una)\s+[a-z]{2,}\b",
    re.IGNORECASE,
)
_RE_CODIGO_BARRAS_LARGO = re.compile(r"\b\d{6,}\b")

# Palabra numérica compuesta en español (para reconocer cláusulas contrastivas)
_PAT_SUB_NUM = (
    r"(?:uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|"
    r"once|doce|trece|catorce|quince|dieciseis|diecisiete|dieciocho|diecinueve|"
    r"veinte|veintiuno|veintiuna|veintiun|veintidos|veintitres|veinticuatro|veinticinco|"
    r"veintiseis|veintisiete|veintiocho|veintinueve|"
    r"treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|"
    r"cien|ciento|(?:dos|tres|cuatro|quini|seis|sete|ocho|nove)cient[oa]s|y)"
)
_PAT_NUM_TOK = (
    r"(?:\d+|cero|"
    r"(?:(?:dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|veinte|treinta|cuarenta|cincuenta|cien)\s+)?mil(?:\s+" + _PAT_SUB_NUM + r"){0,5}|"
    r"(?:cien|ciento|(?:dos|tres|cuatro|quini|seis|sete|ocho|nove)cient[oa]s)(?:\s+" + _PAT_SUB_NUM + r"){0,3}|"
    r"(?:treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa)(?:\s+y\s+(?:uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve))?|"
    r"veintiuno|veintiuna|veintiun|veintidos|veintitres|veinticuatro|veinticinco|veintiseis|veintisiete|veintiocho|veintinueve|veinte|"
    r"dieciseis|diecisiete|dieciocho|diecinueve|diez|once|doce|trece|catorce|quince|"
    r"media\s+docena|(?:una|dos|tres|cuatro|cinco)\s+docenas?|un\s+par|"
    r"uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)"
)


def limpiar_no_cantidades_operacion(texto: str) -> str:
    """Remueve cláusulas de observación, códigos, especificaciones técnicas, fechas y artículos
    de sustantivos de operación.

    Garantiza que expresiones como «TAL-002», «20V», «20B», «24 puertos», «1/2 pulgada»,
    «Estante A1», «hacer un préstamo» o «con observación: Lote 3» jamás se confundan con la
    cantidad a mover.
    """
    if not texto:
        return ""
    texto_sin_obs = texto
    try:
        from .observaciones import separar_observacion_operacion
        base_obs, obs_ext = separar_observacion_operacion(texto)
        if obs_ext and base_obs:
            # Si hay marcador explícito de observación/nota/comentario, excluir siempre su contenido.
            # Si proviene de una cláusula adversativa/natural (", pero ..."), excluir salvo que
            # contenga una autocorrección contrastiva de cantidad (", pero mejor 5 unidades").
            es_marcador_explicito = bool(
                re.search(
                    r"\b(?:observaci[oó]n(?:es)?|comentario(?:s)?|nota(?:s)?|anotaci[oó]n(?:es)?|"
                    r"detalle(?:s)?|novedad(?:es)?|aclaraci[oó]n(?:es)?|reporte|"
                    r"ponle\s+que|agr[eé]gale\s+que|anota\s+que|dejando\s+constancia)\b",
                    texto,
                    flags=re.IGNORECASE,
                )
            )
            parece_autocorreccion_cant = bool(
                re.search(
                    r"\b(?:mejor|sino|que\s+sea[ns]?|que\s+quede[ns]?|pon(?:le|lo)?|"
                    r"c[aá]mbi\w*|unidades?|piezas?|art[ií]culos?)\b",
                    obs_ext,
                    flags=re.IGNORECASE,
                )
            )
            if es_marcador_explicito or not parece_autocorreccion_cant:
                texto_sin_obs = base_obs
    except Exception:
        texto_sin_obs = texto

    t = _normalizar_sin_tildes(texto_sin_obs)
    t = _RE_CODIGO_PRODUCTO.sub(" ", t)
    t = _RE_FECHA_HORA.sub(" ", t)
    t = _RE_FRACCION.sub(" ", t)
    t = _RE_FRACCION_HABLADA.sub(" ", t)
    t = _RE_DECIMAL_HABLADO_SPEC.sub(" ", t)
    t = _RE_SPEC_HABLADA_CON_DE.sub(" ", t)
    t = _RE_ESTANDAR_UBICACION.sub(" ", t)
    t = _RE_ESPECIFICACION_TECNICA.sub(" ", t)
    t = _RE_CODIGO_BARRAS_LARGO.sub(" ", t)
    t = _RE_ARTICULO_OPERACION.sub(" ", t)
    t = _RE_ARTICULO_PREPOSICIONAL.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _limpiar_clausulas_negadas_cantidad(t_norm: str) -> str:
    """Elimina la cantidad descartada en autocorrecciones contrastivas.

    Soporta ambos órdenes:
    - «en vez de 2 que sean 5» -> «que sean 5»
    - «ponle 5 en vez de 2» -> «ponle 5»
    - «no quiero que sean 2 unidades sino que sean 5» -> «sino que sean 5»
    - «no eran dos, eran cinco» -> «, eran cinco»
    """
    t = t_norm
    # 1. «en vez de <num> [unidades]» / «en lugar de <num> [unidades]»
    t = re.sub(
        rf"\ben\s+(?:vez|lugar)\s+de\s+(?:las?\s+|los?\s+|esas?\s+|esos?\s+)?{_PAT_NUM_TOK}\s*(?:unidades?|piezas?|articulos?)?\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    # 2. «no [quiero|eran|son|es|prestes|saques|...] [que sean] <num> [unidades]»
    t = re.sub(
        rf"\bno\s+(?:(?:quiero|quisiera|deseo|necesito|voy\s+a\s+\w+|me\s+lleves?|prest\w*|saqu\w*|sac\w*|retir\w*|ingres\w*|met\w*|devuelv\w*|devolv\w*|traslad\w*|pon\w*|pong\w*|dej\w*|dame|me\s+des|son|eran?|es|sea[ns]?)\s+){{0,3}}"
        rf"(?:que\s+sea[ns]?\s+)?(?:las?\s+|los?\s+|esas?\s+|esos?\s+)?{_PAT_NUM_TOK}\s*(?:unidades?|piezas?|articulos?)?\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    # 3. «<num> [unidades] no» (ej. «2 unidades no, mejor 5»)
    t = re.sub(
        rf"\b{_PAT_NUM_TOK}\s*(?:unidades?|piezas?|articulos?)?\s+no\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", t).strip()


def extraer_cantidad_operacion_texto(texto: str, *, permitir_cero: bool = False) -> int | None:
    """Extrae la cantidad operativa explícita mencionada por el usuario sin falsos positivos.

    Ignora números provenientes de códigos (TAL-002), medidas/especificaciones (20V, 20B,
    24 puertos, 1/2 pulgada, 100MHz), ubicaciones (A1) y artículos indefinidos de acción
    («hacer un préstamo», «registrar una entrada»). Si el usuario se autocorrige dentro de
    la misma frase («presta 2, no mejor 5»), retorna la cantidad corregida (5).
    """
    t = limpiar_no_cantidades_operacion(texto)
    if not t:
        return None

    # Si hay autocorrección contrastiva dentro de la misma frase, descartar el valor negado
    tiene_contraste = bool(
        re.search(
            r"\b(?:sino|en\s+vez\s+de|en\s+lugar\s+de|mejor|perdon|disculpa)\b|,\s*no\b|\bno\s+(?:eran?|son|es|quiero)\b",
            t,
        )
    )
    if tiene_contraste:
        t = _limpiar_clausulas_negadas_cantidad(t)
        # Si aparece «sino ...» / «mejor ...» / «perdon ...» con número a la derecha, priorizar esa cola
        m_cola = re.search(r"\b(?:sino|mejor|perdon|disculpa)\b(.*)$", t)
        if m_cola:
            val_cola = extraer_numero_espanol(m_cola.group(1), permitir_cero=permitir_cero)
            if val_cola is not None:
                return val_cola

    # Priorizar dígitos explícitos restantes
    digitos = re.findall(r"(?<!-)\b(\d+)\b", t)
    if digitos:
        candidatos = [int(d) for d in digitos if int(d) > 0 or permitir_cero]
        if candidatos:
            return candidatos[-1] if tiene_contraste else candidatos[0]

    # Luego cardinales en español (respetando que «un préstamo» ya fue removido)
    return extraer_numero_espanol(t, permitir_cero=permitir_cero)


def extraer_modificacion_cantidad_propuesta(texto: str) -> int | None:
    """Detecta y extrae una corrección o cambio de cantidad en lenguaje natural.

    Cubre variaciones cortas, largas, coloquiales y contrastivas:
    - Cortas: «5», «cinco», «3 unidades», «solo 1», «mejor 4», «ponle 6», «que sean 3».
    - Largas: «hey no quiero que sean esas unidades sino que sean 5 unidades»,
      «oye espera, no eran 2, mejor cámbialo a 4 unidades por favor»,
      «me equivoqué, necesito 3 en vez de 2», «no prestes 2, presta 5».

    Retorna:
    - int > 0 si hay una nueva cantidad válida.
    - 0 o int < 0 si el usuario pidió explícitamente cero o un número negativo (para dar error claro).
    - None si la frase no corresponde a un cambio de cantidad (p. ej. pregunta, código o cambio de almacén).
    """
    if not texto or not texto.strip():
        return None

    t_norm = _normalizar_sin_tildes(texto).strip()
    if not t_norm:
        return None

    # Permitir signos ¿? espurios del STT en respuestas cortas como «¿Qué diez?» o «¿Mejor 5?»,
    # pero rechazar preguntas reales del dominio.
    if ("?" in texto or "¿" in texto) and re.search(
        r"\b(?:cuant[oa]s?|donde|ubicacion|stock|existencias?|quedan?|queda|"
        r"hay|quien|cual(?:es)?|como|puedo|puede[ns]?|se\s+puede|prestable|consumible|movimientos?|historial|prestamos?)\b",
        t_norm,
    ):
        return None

    # Excluir consultas informativas o códigos de otro producto
    if re.search(
        r"\b(?:cuant[oa]s?|donde|ubicacion|stock|existencias?|quedan?|queda|"
        r"hay\s+disponibles?|quien\s+tiene|consultar?|buscame|muestrame|dime|"
        r"confirm[ao]\w*|cancel\w*|descart\w*|olvida\s+todo|anula\w*)\b",
        t_norm,
    ):
        return None
    if _RE_CODIGO_PRODUCTO.search(t_norm):
        return None
    # Si se trata claramente de una nueva operación completa sobre otro producto/tipo (sin contraste de cantidad)
    if re.search(r"\b(?:crear\s+(?:un\s+)?producto|inhabilitar|habilitar|trasladar|hacer\s+un\s+traslado)\b", t_norm):
        return None
    # Si solo pide cambiar de almacén (sin mencionar cantidad/unidades/números)
    if re.search(r"\b(?:almacen|bodega)\b", t_norm) and not re.search(r"\b(?:cantidad|unidades?|piezas?)\b", t_norm):
        return None

    # Limpiar especificaciones técnicas y cláusulas negadas/contrastivas
    t_sin_spec = limpiar_no_cantidades_operacion(t_norm)
    t_limpio = _limpiar_clausulas_negadas_cantidad(t_sin_spec)

    # 1. Forma pura o corta: «5», «cinco», «que diez», «¿Qué diez?», «5 unidades», «solo 3», «mejor 4», «una sola», «0», «-2»
    m_corta = re.fullmatch(
        r"[¿¡\s]*(?:(?:hey|ey|oye|oiga|eh|bueno|mira|si|no|y|pero|ademas|tambien|entonces|"
        r"perdon|disculpa|espera|esperate|mejor|solo|solamente|unicamente|nada\s+mas|"
        r"en\s+realidad|la\s+verdad|que(?:\s+sea[ns]?|\s+quede[ns]?)?|son|eran?|es|"
        r"cambial[aoe]|cambia|cambiar|modific[aeo]\w*|corrig[aeo]\w*|actualiz[aeo]\w*|"
        r"pon(?:le|lo|la|me)?|ponga[ns]?|dejal[aoe]|dejame|dame|quiero|prefiero|necesito|"
        r"prest[aeo]\w*|sac[aeo]\w*|saqu\w*|retir\w*|ingres\w*|devuelv\w*|devolv\w*|"
        r"la\s+cantidad|cantidad|unidades?|a|por|en|de|sino)[\s,;:.!¡¿?-]*)+"
        r"(-?\s*\d+|[a-z]+(?:\s+y\s+[a-z]+)?)\s*(?:sol[oa]s?|unidades?|piezas?|articulos?|por\s+favor|mejor|en\s+realidad|en\s+total)?[\s.,;:!¡?]*",
        t_limpio,
    )
    if m_corta:
        val = extraer_numero_espanol(
            m_corta.group(1),
            permitir_cero=True,
            detectar_negativo=True,
            solo_numero_exacto=True,
        )
        if val is not None:
            return val

    # 1b. Número exacto pelado («5», «cinco», «5 unidades», «una sola», «0», «-3»)
    t_pelado = re.sub(
        r"\b(?:solo|solamente|unicamente|nada\s+mas|sol[oa]s?|unidades?|piezas?|articulos?|por\s+favor)\b",
        " ",
        t_limpio,
    )
    t_pelado = re.sub(r"[^\w\s-]", " ", t_pelado)
    t_pelado = re.sub(r"\s+", " ", t_pelado).strip()
    if t_pelado:
        val_pelado = extraer_numero_espanol(
            t_pelado,
            permitir_cero=True,
            detectar_negativo=True,
            solo_numero_exacto=True,
        )
        if val_pelado is not None:
            return val_pelado

    # 2. Frases largas naturales con marcador de cantidad, corrección o contraste
    tiene_marca_cant_o_corr = bool(
        re.search(
            r"\b(?:cantidad|unidades?|piezas?|sino|mejor|en\s+vez\s+de|en\s+lugar\s+de|"
            r"cambi\w*|modific\w*|corrig\w*|actualiz\w*|pon(?:le|lo|la|me)?|ponga[ns]?|"
            r"dejal[aoe]|que\s+sea[ns]?|que\s+quede[ns]?|no\s+quiero|no\s+eran?|no\s+son|"
            r"me\s+equivoque|equivocad[oa]|prefiero|necesito|quiero|dame|solo|solamente|unicamente|"
            r"nada\s+mas|espera|esperate|perdon|disculpa)\b",
            t_norm,
        )
    )
    if not tiene_marca_cant_o_corr:
        return None

    # Evitar capturar una corrección de observación que por casualidad contenga un número (ej. "ponle observación: falta 1 tornillo")
    if re.search(r"\b(?:observaci[oó]n|nota|comentario|motivo)\b", t_norm):
        return None

    # Priorizar número después de «sino», «que sean», «a», «por», «en», «quiero», «mejor», «ponle», «son»
    m_objetivo = re.search(
        r"(?:\bsino\s+(?:que\s+sea[ns]?\s+)?|"
        r"\bque\s+(?:sea[ns]?|quede[ns]?)\s+(?:solo\s+|en\s+)?|"
        r"\b(?:cambi\w*|modific\w*|corrig\w*|actualiz\w*)(?:\s+(?:la\s+)?cantidad)?\s+(?:a|por|en)\s+|"
        r"\b(?:pon(?:le|lo|la|me)?|ponga[ns]?|dejal[aoe])\s+(?:en\s+|solo\s+)?|"
        r"\b(?:mejor|solo|solamente|unicamente|nada\s+mas|quiero|prefiero|necesito|dame|son|eran?|presta\w*|saca\w*|saque\w*|ingresa\w*|devuelve\w*)\s+(?:que\s+sea[ns]?\s+|solo\s+)?)"
        rf"(-?\s*\d+|{_PAT_NUM_TOK})\b",
        t_limpio,
        flags=re.IGNORECASE,
    )
    if m_objetivo:
        val_obj = extraer_numero_espanol(
            m_objetivo.group(1),
            permitir_cero=True,
            detectar_negativo=True,
            solo_numero_exacto=True,
        )
        if val_obj is not None:
            return val_obj

    # Fallback sobre el texto limpio tras quitar cláusulas negadas (si menciona explícitamente unidades o cantidad)
    if re.search(r"\b(?:unidades?|piezas?|cantidad)\b", t_limpio):
        val_fb = extraer_numero_espanol(t_limpio, permitir_cero=True, detectar_negativo=True)
        if val_fb is not None:
            return val_fb

    return None
