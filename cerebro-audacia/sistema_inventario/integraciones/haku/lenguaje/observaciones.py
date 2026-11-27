"""Extracción general y libre de hardcoding de observaciones en español para operaciones de HAKU.

Soporta de manera escalable las formas naturales en que un usuario expresa observaciones,
novedades o condiciones físicas de un artículo (especialmente en devoluciones):
1. Marcadores explícitos ("con observación...", "comentario:", "nota:", "ponle que...",
   "agrega como comentario que...", "deja constancia de que...", "reportando que...", etc.).
2. Cláusulas adversativas o aclaratorias (", pero viene rayado", ", aunque sin caja",
   ", solo que le falta un cable", ", eso sí, está sucio").
3. Cláusulas naturales de estado físico o completitud sin palabra clave ("en buen estado",
   "en óptimas condiciones", "sin caja", "sin novedades", "con la pantalla rota",
   "llegó golpeado", "le falta una punta", "no enciende", "todo completo").
4. Modificación por voz sobre propuesta activa ("agrégale observación...", "ponle que...",
   "cambia la observación a...", "quita la observación") y confirmación con observación en un turno
   ("confirma con observación...", "sí, confirma y ponle que...").
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


# Comandos de inventario positivos que nunca deben absorberse como observación de una cláusula adversativa
# (permitiendo observaciones negativas reales como "pero no devolvió el cable" o "pero no trajo el estuche")
_RE_COMANDO_INVENTARIO = re.compile(
    r"(?<!\bno\s)\b(?:devolv(?:er|i[oó]|ieron)|devuelv\w*|prest(?:ar|[oó]|ame|ale)|"
    r"registr(?:a|ar|[oó])\s+(?:una?\s+)?(?:entrada|salida|pr[eé]stamo|traslado|devoluci[oó]n)|"
    r"traslad(?:a|ar|[oó])|cre(?:a|ar)\s+(?:un\s+)?producto)\b",
    re.IGNORECASE,
)

# Tier 1A: Marcadores nominales explícitos de observación/nota/comentario
_RE_MARCADOR_NOMINAL = re.compile(
    r"(?:[,;]\s*|\s+y\s+|\s+)"
    r"(?:con\s+(?:la\s+|el\s+|una?\s+|las\s+|los\s+)?)?"
    r"(?:observaci[oó]n(?:es)?|comentario(?:s)?|nota(?:s)?|anotaci[oó]n(?:es)?|"
    r"detalle(?:s)?|novedad(?:es)?|aclaraci[oó]n(?:es)?|reporte)"
    r"(?:\s+(?:adicional|opcional))?"
    r"\s*(?:de\s+que\b|que\s+diga\b|que\b|de\b|:|-)?\s*(.+)$",
    re.IGNORECASE,
)

# Tier 1B: Verbos imperativos o de reporte para añadir una observación
_RE_MARCADOR_VERBAL = re.compile(
    r"(?:[,;]\s*|\s+y\s+|\s+)"
    r"(?:"
    r"(?:agr[eé]g(?:a|ale|arle|ar|ando)|a[nñ][aá]d(?:e|ele|ir(?:le)?|iendo)|an[oó]t(?:a|ale|ar(?:le)?|ando)|"
    r"ap[uú]nt(?:a|ale|ar(?:le)?|ando)|d[eé]j(?:a|ale|ando)\s+(?:constancia|registro|nota|escrito|observaci[oó]n)|"
    r"aclar(?:a|ando)|indic(?:a|ando)|coment(?:a|ando)|report(?:a|ando)|especific(?:a|ando)|"
    r"diciendo|avisando|mencionando|se[nñ]alando)"
    r"(?:\s+(?:como\s+|en\s+(?:la\s+|las\s+)?)?(?:observaci[oó]n(?:es)?|nota(?:s)?|comentario(?:s)?))?"
    r"\s*(?:de\s+que\b|que\s+diga\b|que\b|:|-)"
    r"|"
    r"p[oó]n(?:le|er(?:le)?)?\s+(?:(?:como\s+|en\s+(?:la\s+|las\s+)?)?(?:observaci[oó]n(?:es)?|nota(?:s)?|comentario(?:s)?)\s*(?:de\s+que\b|que\b|:|-)?|que\b)"
    r")\s*(.+)$",
    re.IGNORECASE,
)

# Tier 2: Cláusulas adversativas / matices naturales ("..., pero viene rayado", "... aunque le falta el cable")
_RE_CLAUSULA_ADVERSATIVA = re.compile(
    r"(?:[,;]\s*|\s+)(?:pero|aunque|s[oó]lo\s+que|sin\s+embargo|eso\s+s[ií]|"
    r"nada\s+m[aá]s\s+que|nom[aá]s\s+que|salvo\s+que|excepto\s+que)\s*,?\s*(.+)$",
    re.IGNORECASE,
)

# Tier 3: Cláusulas naturales de condición física / completitud / novedad (sin palabra meta)
_PARTES_BIEN = (
    r"pantalla|carcasa|tapa|caja|estuche|empaque|funda|cable(?:s)?|punta(?:s)?|conector(?:es)?|"
    r"bot[oó]n(?:es)?|tecla(?:s)?|lente(?:s)?|bater[ií]a(?:s)?|pila(?:s)?|display|visor|equipo|"
    r"piezas?|accesorios?|cargador|adaptador|fuente|antena|sensor(?:es)?|soporte|base|mango|"
    r"aguja|fusible|sonda(?:s)?|pinza(?:s)?|manual(?:es)?|tornillos?|control|componentes?"
)
_ESTADOS_FISICOS = (
    r"golpead[oa]s?|da[nñ]ad[oa]s?|rayad[oa]s?|rot[oa]s?|quebrad[oa]s?|doblad[oa]s?|"
    r"suci[oa]s?|mojad[oa]s?|quemad[oa]s?|averiad[oa]s?|defectuos[oa]s?|descompuest[oa]s?|"
    r"descalibrad[oa]s?|descargad[oa]s?|incomplet[oa]s?|suelt[oa]s?|floj[oa]s?|faltante(?:s)?|"
    r"rajad[oa]s?|partid[oa]s?|oxidado(?:s)?|oxidada(?:s)?|desgastad[oa]s?|pelad[oa]s?"
)

_RE_CONDICION_NATURAL = re.compile(
    r"(?:[,;]\s*|\s+)("
    # a) "en buen/perfecto/mal/... estado" o "en buenas/óptimas/... condiciones"
    r"en\s+(?:muy\s+)?(?:buen|perfecto|excelente|[oó]ptimo|mal|regular|p[eé]simo)\s+estado(?:\s+[^.;?!]*)?|"
    r"en\s+(?:muy\s+)?(?:buenas|perfectas|excelentes|[oó]ptimas|malas|regulares|p[eé]simas)\s+condiciones(?:\s+[^.;?!]*)?|"
    # b) "sin novedades / sin daños / sin caja / sin cargador / sin accesorios..."
    rf"sin\s+(?:novedad(?:es)?|da[nñ]os?|fallas?|problemas?|detalles?|golpes?|rayones?|{_PARTES_BIEN})(?:\s+[^.;?!]*)?|"
    # c) "con todos sus accesorios / con daño / con la pantalla rayada..."
    r"con\s+(?:todos?\s+(?:sus|los)\s+accesorios|accesorios\s+completos?|caja\s+completa|estuche\s+completo|"
    r"(?:un[oa]?\s+)?(?:da[nñ]o(?:s)?|golpe(?:s)?|ray[oó]n(?:es)?|falla(?:s)?|defecto(?:s)?|desgaste|"
    r"rotura|fisura|grieta|aver[ií]a|corto|bater[ií]a\s+baja|pila(?:s)?\s+descargada(?:s)?)|"
    rf"(?:la\s+|el\s+|las\s+|los\s+|un\s+|una\s+)?(?:{_PARTES_BIEN})\s+(?:{_ESTADOS_FISICOS}))(?:\s+[^.;?!]*)?|"
    # d) Predicado de estado: "llegó golpeado", "viene en buen estado", "está roto", "vino sin caja"
    r"(?:lleg[oó]|vino|viene|est[aá]|qued[oó]|entr[oó]|lo\s+trajo|se\s+recibi[oó]|se\s+entreg[oó])\s+"
    rf"(?:en\s+(?:muy\s+)?(?:buen|perfecto|mal|excelente|[oó]ptimo|regular)\s+estado|"
    r"en\s+(?:buenas|perfectas|malas|excelentes|[oó]ptimas)\s+condiciones|"
    r"con\s+[^.;?!]+|sin\s+[^.;?!]+|"
    rf"(?:un\s+poco\s+|muy\s+|bastante\s+|medio\s+)?(?:{_ESTADOS_FISICOS}|"
    r"complet[oa]s?|intact[oa]s?|impecable(?:s)?|perfect[oa]s?|bien|mal|operativ[oa]s?|"
    r"funcionando(?:\s+(?:bien|mal|correctamente|perfectamente|perfect[oa]|normal|sin\s+problemas))?))(?:\s+[^.;?!]*)?|"
    # e) Faltantes o mal funcionamiento: "le falta una punta", "no enciende", "tiene un golpe"
    r"(?:le\s+|les\s+)?falta(?:n)?\s+(?:el\s+|la\s+|los\s+|las\s+|un\s+|una\s+|unos\s+|unas\s+|\d+\s+)?[^.;?!]+|"
    r"no\s+(?:enciende|prende|funciona|sirve|carga|mide|marca|arranca|trae\s+[^.;?!]+|tiene\s+[^.;?!]+)(?:\s+[^.;?!]*)?|"
    r"tiene\s+(?:un[oa]?\s+)?(?:golpe(?:s)?|ray[oó]n(?:es)?|falla(?:s)?|da[nñ]o(?:s)?|defecto(?:s)?|"
    rf"grieta(?:s)?|fisura(?:s)?|corto|problema(?:s)?|desgaste|bater[ií]a\s+baja|"
    rf"(?:la\s+|el\s+)?(?:{_PARTES_BIEN})\s+(?:{_ESTADOS_FISICOS}))(?:\s+[^.;?!]*)?|"
    # f) Descriptores de estado al cierre (evitando 'completo' suelto por 'préstamo completo')
    r"todo\s+completo|accesorios\s+completos?|funcionando\s+(?:bien|correctamente|perfectamente|perfect[oa]|normal|sin\s+problemas)|"
    rf"(?:{_ESTADOS_FISICOS})"
    r")[\s.,;:!?]*$",
    re.IGNORECASE,
)

# Condición natural standalone (cuando el usuario agrega observación por voz con propuesta abierta)
_RE_CONDICION_STANDALONE = re.compile(
    r"^(?:"
    r"en\s+(?:muy\s+)?(?:buen|perfecto|excelente|[oó]ptimo|mal|regular|p[eé]simo)\s+estado\b|"
    r"en\s+(?:muy\s+)?(?:buenas|perfectas|excelentes|[oó]ptimas|malas|regulares)\s+condiciones\b|"
    rf"sin\s+(?:novedad(?:es)?|da[nñ]os?|fallas?|problemas?|detalles?|golpes?|{_PARTES_BIEN})\b|"
    r"con\s+(?:todos?\s+(?:sus|los)\s+accesorios|accesorios\s+completos?|caja\s+completa|estuche\s+completo|"
    r"(?:un[oa]?\s+)?(?:da[nñ]o|golpe|ray[oó]n|falla|defecto|desgaste|rotura|fisura|grieta|aver[ií]a|bater[ií]a\s+baja)|"
    rf"(?:la\s+|el\s+|las\s+|los\s+|un\s+|una\s+)?(?:{_PARTES_BIEN})\s+(?:{_ESTADOS_FISICOS}))\b|"
    r"(?:lleg[oó]|vino|viene|est[aá]|qued[oó]|se\s+recibi[oó]|se\s+entreg[oó])\s+|"
    r"(?:le\s+|les\s+)?falta(?:n)?\s+|"
    r"no\s+(?:enciende|prende|funciona|sirve|carga|mide|marca|arranca|trae|tiene)\b|"
    r"tiene\s+(?:un[oa]?\s+)?(?:golpe|ray[oó]n|falla|da[nñ]o|defecto|grieta|fisura|corto|problema|desgaste|bater[ií]a\s+baja)\b|"
    r"todo\s+completo\b|accesorios\s+completos?\b|funcionando\s+(?:bien|correctamente|perfectamente|normal)\b|"
    rf"(?:{_ESTADOS_FISICOS})\b"
    r")",
    re.IGNORECASE,
)


def _limpiar_observacion(texto_obs: str) -> str:
    """Limpia puntuación sobrante y normaliza espacios de una observación extraída."""
    limpia = re.sub(r"\s+", " ", (texto_obs or "")).strip('"\' \t\r\n.,;:!?¿¡-')
    # Quitar conectores introductorios residuales ("que ..." o "de que ...")
    limpia = re.sub(r"^(?:de\s+que|que)\s+", "", limpia, flags=re.IGNORECASE).strip('"\' \t\r\n.,;:!?¿¡-')
    return limpia[:500]


def _limpiar_base(texto_base: str) -> str:
    """Limpia conectores colgantes al final del comando base tras retirar la observación."""
    base = re.sub(r"[\s,;:.\-]+$", "", (texto_base or "")).strip()
    base = re.sub(r"\s+(?:y|con)$", "", base, flags=re.IGNORECASE).strip()
    return base


def separar_observacion_operacion(texto: str) -> Tuple[str, str]:
    """Separa un comando en (texto_base_sin_observacion, observacion_limpia).

    Si no detecta una observación válida, devuelve (texto_original_limpio, "").
    Nunca descarta el comando base: exige que el prefijo conserve al menos 2 palabras.
    """
    raw = (texto or "").strip()
    if not raw:
        return "", ""

    # Tier 1B (verbal) y Tier 1A (nominal): elegir el que comience primero en la frase
    # para capturar toda la cláusula de observación cuando se combinan
    # (ej. "ponle en observaciones que llegó sin caja").
    candidatos_t1 = []
    for patron in (_RE_MARCADOR_VERBAL, _RE_MARCADOR_NOMINAL):
        m = patron.search(raw)
        if m:
            # Si "sin" precede inmediatamente al marcador nominal ("sin novedades ni daños"),
            # dejar que Tier 3 capture la cláusula completa incluyendo "sin".
            if patron is _RE_MARCADOR_NOMINAL and re.search(r"\bsin\s*$", raw[: m.start()], re.IGNORECASE):
                continue
            prefijo = _limpiar_base(raw[: m.start()])
            obs = _limpiar_observacion(m.group(1))
            if len(prefijo.split()) >= 2 and len(obs) >= 2:
                candidatos_t1.append((m.start(), prefijo, obs))

    if candidatos_t1:
        candidatos_t1.sort(key=lambda item: item[0])
        _, prefijo, obs = candidatos_t1[0]
        return prefijo, obs

    # Tier 2: Cláusula adversativa (", pero viene con la pantalla rayada", "aunque falta el cable")
    m_adv = _RE_CLAUSULA_ADVERSATIVA.search(raw)
    if m_adv:
        prefijo = _limpiar_base(raw[: m_adv.start()])
        obs = _limpiar_observacion(m_adv.group(1))
        if (
            len(prefijo.split()) >= 2
            and len(obs) >= 3
            and not _RE_COMANDO_INVENTARIO.search(obs)
        ):
            return prefijo, obs

    # Tier 3: Condición física / completitud natural al cierre del enunciado
    m_cond = _RE_CONDICION_NATURAL.search(raw)
    if m_cond:
        prefijo = _limpiar_base(raw[: m_cond.start()])
        obs = _limpiar_observacion(m_cond.group(1))
        # Evitar confundir "préstamo completo" / "inventario completo" (nunca capturable) o frases de 1 palabra
        if (
            len(prefijo.split()) >= 2
            and len(obs) >= 3
            and not _RE_COMANDO_INVENTARIO.search(obs)
        ):
            return prefijo, obs

    return raw, ""


def extraer_modificacion_observacion_propuesta(
    texto: str,
    tipo_operacion: str = "",
) -> Optional[Tuple[str, str]]:
    """Detecta si el usuario desea agregar, cambiar o quitar la observación de una propuesta activa.

    Retorna:
    - ("limpiar", "") si solicita borrar/quitar la observación.
    - ("establecer", texto_obs) si proporciona una observación nueva o modificada.
    - None si el texto no corresponde a edición de observación.
    """
    raw = (texto or "").strip()
    if not raw:
        return None

    # 1. Quitar / borrar observación explícitamente
    if re.fullmatch(
        r"(?:qu[ií]ta(?:le)?|b[oó]rra(?:le)?|elim[ií]na(?:le)?|l[ií]mpia(?:le)?|remueve)\s+"
        r"(?:la\s+|las\s+|el\s+|los\s+)?(?:observaci[oó]n(?:es)?|nota(?:s)?|comentario(?:s)?)"
        r"|sin\s+(?:observaci[oó]n(?:es)?|nota(?:s)?|comentario(?:s)?)",
        raw.strip(" \t\r\n.,;:!?¿¡"),
        flags=re.IGNORECASE,
    ):
        return ("limpiar", "")

    # 2. Formas explícitas de agregar / cambiar observación por voz
    m_expl = re.match(
        r"^(?:y\s+|adem[aá]s\s+|tambi[eé]n\s+|por\s+favor\s+)?"
        r"(?:"
        r"(?:agr[eé]g(?:a|ale|ar(?:le)?)|a[nñ][aá]d(?:e|ele|ir(?:le)?)|p[oó]n(?:le|er(?:le)?)?|"
        r"c[aá]mbi(?:a|ale|ar(?:le)?)|modif[ií]c(?:a|ale|ar)|actual[ií]z(?:a|ale|ar)|ed[ií]t(?:a|ale|ar)|"
        r"an[oó]t(?:a|ale|ar(?:le)?)|ap[uú]nt(?:a|ale|ar(?:le)?)|escr[ií]b(?:e|ele|ir)|"
        r"d[eé]j(?:a|ale|ar)\s+(?:constancia|registro|nota|escrito))"
        r"\s+(?:(?:una?\s+|la\s+|el\s+|las\s+|los\s+|como\s+|en\s+(?:la\s+|las\s+)?)?"
        r"(?:observaci[oó]n(?:es)?|nota(?:s)?|comentario(?:s)?|anotaci[oó]n|detalle)\s*)?"
        r"(?:que\s+diga\b|de\s+que\b|que\b|a\b|por\b|de\b|:|-)?\s*"
        r"|"
        r"(?:aclar(?:a|ar)|indic(?:a|ar)|coment(?:a|ar)|report(?:a|ar)|especific(?:a|ar)|"
        r"di(?:le)?|se[nñ]al(?:a|ar))\s+(?:de\s+)?(?:que\b|:)\s*"
        r"|"
        r"(?:con\s+(?:la\s+|el\s+|una?\s+)?)?"
        r"(?:observaci[oó]n(?:es)?|comentario(?:s)?|nota(?:s)?|anotaci[oó]n|detalle)\s*"
        r"(?:adicional\s*)?(?:es\b|ser[ií]a\b|de\s+que\b|que\b|de\b|:|-)\s*"
        r")"
        r"(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if m_expl:
        cabecera = raw[: m_expl.start(1)].lower()
        obs = _limpiar_observacion(m_expl.group(1))
        # Evitar falsos positivos con cambios de destinatario/producto/cantidad/almacén
        # ("ponle a Juan", "cambia la cantidad a 5", "cambia el producto a MUL-001", "ponlo en almacén B")
        menciona_obs = bool(
            re.search(
                r"\b(?:observaci[oó]n(?:es)?|nota(?:s)?|comentario(?:s)?|anotaci[oó]n|detalle|"
                r"constancia|registro|escrito|aclar|indic|report|especific|se[nñ]al)\b",
                cabecera,
            )
            or re.search(r"\b(?:p[oó]n(?:le|er(?:le)?)?|agr[eé]g\w*|a[nñ][aá]d\w*|an[oó]t\w*|ap[uú]nt\w*|di(?:le)?)\s+que\b", cabecera)
        )
        es_otro_campo = bool(
            re.search(
                r"\b(?:cantidad|producto|art[ií]culo|destinatario|nombre|persona|almac[eé]n|bodega|"
                r"ubicaci[oó]n|destino|origen|motivo|stock|prestable|consumible|retornable)\b",
                cabecera,
            )
        )
        if menciona_obs and not es_otro_campo and len(obs) >= 2:
            menciona_palabra_obs = bool(
                re.search(
                    r"\b(?:observaci[oó]n(?:es)?|nota(?:s)?|comentario(?:s)?|anotaci[oó]n|detalle|"
                    r"constancia|registro|escrito)\b",
                    cabecera,
                )
            )
            # Si el usuario no dijo explícitamente "observación/nota/comentario", no tragar
            # modificaciones de cantidad, destinatario, prestable/consumible, stock ni almacén/ubicación.
            es_mod_otro_campo_en_obs = (not menciona_palabra_obs) and bool(
                re.match(
                    r"^(?:"
                    r"(?:sean?\s+|son\s+)?(?:\d+|cero|un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|"
                    r"once|doce|trece|catorce|quince|dieci\w+|veint\w+|treinta|cuarenta|cincuenta|cien\w*)\b|"
                    r"(?:sean?\s+|es\s+|est[eé]\s+)?(?:para\b|a\s+nombre\b|en\s+(?:el\s+)?(?:almac[eé]n|bodega)|"
                    r"al\s+(?:almac[eé]n|bodega)|en\s+(?:la\s+)?ubicaci[oó]n|"
                    r"(?:no\s+)?(?:prestable|retornable|consumible)\b)|"
                    r"(?:el\s+)?stock\s+(?:m[ií]nimo|inicial)\b"
                    r")",
                    obs,
                    flags=re.IGNORECASE,
                )
            )
            if not es_mod_otro_campo_en_obs:
                return ("establecer", obs)

    # 3. Para propuestas de DEVOLUCION activas, aceptar también frases naturales de estado físico
    #    (ej. "Llegó en buen estado", "Viene con la pantalla rayada", "Sin caja", "Le falta una punta").
    if str(tipo_operacion or "").upper() == "DEVOLUCION":
        sin_signos = raw.strip(" \t\r\n.,;:!?¿¡")
        # Excluir preguntas o códigos de producto
        if "?" not in raw and "¿" not in raw and not re.search(r"\b[A-Za-z]{2,5}-\d{1,5}\b", raw):
            if _RE_CONDICION_STANDALONE.match(sin_signos):
                obs = _limpiar_observacion(sin_signos)
                if len(obs) >= 3:
                    return ("establecer", obs)

    return None


def extraer_confirmacion_con_observacion(
    texto: str,
    tipo_operacion: str = "",
) -> Optional[str]:
    """Detecta confirmación por voz que incluye una observación en la misma frase.

    Ejemplos:
    - "Confirma con observación devuelto en buen estado."
    - "Sí, confirma y ponle que llegó sin caja."
    - "Confirmar, observación: todo completo."
    - "Dale, anótale que viene con la pantalla rayada."
    - "Sí, confirma en perfecto estado." (solo cuando tipo_operacion == DEVOLUCION)
    """
    raw = (texto or "").strip()
    if not raw:
        return None

    m_conf = re.match(
        r"^(?:s[ií]\s*,?\s*)?(?:confirmo|confirma|confirmad|confirmar|confirmado|dale|de\s+acuerdo|adelante|procede(?:r)?|ejecuta(?:r)?|correcto|ok)"
        r"(?:\s+(?:la\s+operaci[oó]n|la\s+devoluci[oó]n|la\s+propuesta))?"
        r"[\s,;:\-]+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if not m_conf:
        return None

    cola = m_conf.group(1).strip()
    # Limpiar conectores iniciales ("y ...", "pero ...") antes de analizar la cola
    cola_limpia = re.sub(r"^(?:y|pero)\s+", "", cola, flags=re.IGNORECASE).strip()
    if not cola_limpia:
        return None

    res_mod = extraer_modificacion_observacion_propuesta(cola_limpia, tipo_operacion=tipo_operacion)
    if res_mod and res_mod[0] == "establecer" and res_mod[1]:
        return res_mod[1]

    # También soportar "confirma, pero viene rayado" / "confirma en buen estado" en DEVOLUCION
    if str(tipo_operacion or "").upper() == "DEVOLUCION":
        _, obs_sep = separar_observacion_operacion(f"devuelve equipo {cola}")
        if obs_sep:
            return obs_sep

    return None
