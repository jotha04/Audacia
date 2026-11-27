"""Interpretación de expresiones coloquiales de la región Caribe (Barranquilla/Costa) para Haku Inventario.

REGLA DE DISEÑO ARQUITECTÓNICO — SEPARACIÓN ESTRICTA ENTRADA / SALIDA:
1. Capa de Ingesta y Comprensión (Entrada):
   - El dialecto caribeño (modismos: vaina, vuelta, voleo; expresiones coloquiales;
     variaciones fonéticas de Whisper: prestao->prestado, pa->para; elisión de plural)
     pertenece EXCLUSIVAMENTE a la interpretación determinista de intenciones operacionales.
2. Capa de Comunicación y Respuesta (Salida):
   - Haku NUNCA utiliza personalidad costeña ni responde con jerga regional ('ajá', 'compae', 'la vaina', etc.).
   - La respuesta generada (texto en pantalla y voz TTS) es estrictamente profesional, neutral,
     formal y empresarial, adecuada para un sistema de inventario organizacional.
   - Nunca se infiere que comprender el habla regional autorice a responder de manera informal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExpresionAnalizada:
    """Resultado del preprocesamiento dialectal costeño."""
    texto_original: str
    texto_estandar: str
    vocativo_detectado: Optional[str]
    es_ambiguo: bool
    demostrativo_detectado: Optional[str]
    producto_mencionado: Optional[str]
    cantidad_detectada: Optional[int]


def sanitizar_demostrativo_salida(dem: Optional[str]) -> str:
    """Sanitiza cualquier demostrativo o comodín para que nunca contenga regionalismos costeños."""
    if not dem:
        return "ese producto"
    if re.search(r"\b(vaina|vuelta|cosa|movida|voleo|caleta|pelao|pelaa|aj[aá]|compae|compadre|mi\s+llave|cuadro|volado)\b", dem, re.IGNORECASE):
        return "ese producto"
    return dem


class ValidadorSalidaEmpresarial:
    """Validador y blindaje estricto que impide que Haku genere regionalismos en sus respuestas."""

    PALABRAS_PROHIBIDAS = [
        "ajá",
        "compae",
        "compadre",
        "mi llave",
        "cuadro",
        "la vaina",
        "esa vaina",
        "vaina",
        "esa vuelta",
        "la vuelta",
        "pelao",
        "pelaa",
        "voleo",
        "caleta",
        "ombe",
    ]

    PATRONES_PROHIBIDOS = [
        r"\baj[aá]\b",
        r"\bcompae\b",
        r"\bcompadre\b",
        r"\bmi\s+llave\b",
        r"\bcuadro\b",
        r"\b(?:la|esa|esta)\s+vaina\b",
        r"\bvaina\b",
        r"\b(?:la|esa|esta)\s+vuelta\b",
        r"\b(?:est[aá]|qued[oó])\s+pela[oa]\b",
        r"\bpela[oa]\b",
        r"\bpelaa\b",
        r"\bvoleo\b",
        r"\bcaleta\b",
        r"\bombe\b",
    ]

    @classmethod
    def detectar_regionalismos_prohibidos(cls, texto: str, nombres_permitidos: Optional[list[str]] = None) -> list[str]:
        """Detecta términos regionales prohibidos en la salida de Haku."""
        if not texto:
            return []

        texto_evaluar = texto
        if nombres_permitidos:
            for nombre in nombres_permitidos:
                if nombre:
                    texto_evaluar = re.sub(re.escape(nombre), "", texto_evaluar, flags=re.IGNORECASE)

        encontrados = []
        for patron in cls.PATRONES_PROHIBIDOS:
            match = re.search(patron, texto_evaluar, re.IGNORECASE)
            if match:
                encontrados.append(match.group(0))
        return encontrados

    @classmethod
    def asegurar_salida_empresarial(cls, texto: str, nombres_permitidos: Optional[list[str]] = None) -> str:
        """Verifica que el texto no contenga regionalismos. Si los hubiera, lanza un error de integridad."""
        prohibidos = cls.detectar_regionalismos_prohibidos(texto, nombres_permitidos)
        if prohibidos:
            raise ValueError(
                f"Contaminación dialectal detectada en la salida empresarial de Haku: {prohibidos}. "
                "La capa de salida debe ser estrictamente formal y neutral."
            )
        return texto


class InterpretadorCosteno:
    """Interpreta modismos y estructuras gramaticales de la costa caribeña colombiana."""

    # Vocativos y saludos comunes de apertura que no aportan a la intención
    VOCATIVOS = [
        r"^hey\s+ha[ck]u\b",
        r"^ey\s+ha[ck]u\b",
        r"^aj[aá]\s+ha[ck]u\b",
        r"^oye\s+ha[ck]u\b",
        r"^mira\s+ha[ck]u\b",
        r"^viejo\s+ha[ck]u\b",
        r"^ha[ck]u\s+cuadro\b",
        r"^ha[ck]u\s+mi\s+llave\b",
        r"^ha[ck]u\s+compa\b",
        r"^ha[ck]u\s+compae\b",
        r"^ha[ck]u\b",
        r"^compadre\b",
        r"^compae\b",
        r"^compa\b",
        r"^mijo\b",
        r"^mi\s+llave\b",
        r"^(?:oye|eche|aj[aá]|hey|ey|mira|hola|buenas)\s+cuadro\b",
        r"^cuadro\s*,\s*",
        r"^papi\b",
        r"^ombe\b",
        r"^oye\b",
        r"^hey\b",
        r"^ey\b",
        r"^aj[aá]\b",
        r"^ve\s+ac[aá]\b",
        r"^mira\s+ve\b",
        r"^ven\s+ac[aá]\b",
        r"^h[aá]game\s+el\s+favor\b",
        r"^por\s+fa\b",
    ]

    # Normalizaciones fonéticas para transcripciones imperfectas de Faster-Whisper en habla caribeña
    NORMALIZACIONES_FONETICAS = [
        # Vocativos costeños y aperturas coloquiales (solo al inicio cuando anteceden a una frase real)
        (
            r"^(?:oye|oiga|eche|aj[aá]|nombe|hey|ey|mira|hola|buenas)[\s,]+(?:mi\s+llave|llave|compae|compadre|compa|cuadro|vale|mijo|ombe)\b[\s,]*(?=[¿?¡!\w])",
            "",
        ),
        (
            r"^(?:mi\s+llave|compae|compadre|compa|nombe|eche|mijo|ombe)\b[\s,]+(?=[¿?¡!\w])",
            "",
        ),
        (r"^cuadro\s*,\s*", ""),
        (
            r"^(?:[eé]cheme\s+una\s+manito(?:\s+ah[ií])?|(?:h[aá]game|haceme)\s+el\s+favor(?:\s+y(?:\s+me)?)?)[\s,]+(?=[¿?¡!\w])",
            "",
        ),

        # Participios y adjetivos con pérdida de /d/ intervocálica y elisión caribeña
        (r"\bprestaos\b", "prestados"),
        (r"\bprestao\b", "prestado"),
        (r"\bprest[aá]s\b", "prestadas"),
        (r"\bpelao\b", "pelado"),
        (r"\bpelaos\b", "agotados"),
        (r"\bpelaas\b", "agotadas"),
        (r"\b(est[aá]n?|qued(?:[oó]|aron)|tenemos|hay)\s+(?:pelaa|pel[aá]s?)\b", r"\1 agotadas"),
        (r"\b(est[aá]n?|qued(?:[oó]|aron)|tenemos|hay)\s+pelad[oa]s?\b", r"\1 agotados"),
        (r"\bqu[eé]\s+(?:productos?\s+|art[ií]culos?\s+|cosas?\s+)?tenemos\s+agotad[oa]s?\b", "qué productos están agotados"),
        (r"\bagotao\b", "agotado"),
        (r"\bagotaos\b", "agotados"),
        (r"\bsal[ií]o\b", "salido"),
        (r"\bmov[ií]o\b", "movido"),
        (r"\bmet[ií]o\b", "metido"),
        (r"\bdespachao\b", "despachado"),
        (r"\bguardao\b", "guardado"),
        (r"\btirao\b", "tirado"),
        (r"\bacabao\b", "acabado"),
        (r"\bquedao\b", "quedado"),

        # Infinitivos caribeños con elisión de /r/ final tras auxiliar
        (r"\b(se\s+puede|se\s+pueden|puedo|podemos|necesito|quiero|voy\s+a|para)\s+prest[aá]\b", r"\1 prestar"),
        (r"\b(se\s+puede|se\s+pueden|puedo|podemos|necesito|quiero|voy\s+a|para)\s+sac[aá]\b", r"\1 sacar"),

        # Verbo estar con aspiración de /s/ o apócope coloquial
        (r"\beht[aá]\b", "está"),
        (r"\beht[aá]n\b", "están"),
        (r"\b(?:d[oó]nde|d[oó]n|en\s+d[oó]nde)\s+ta\b", "dónde está"),
        (r"\b(?:d[oó]nde|d[oó]n|en\s+d[oó]nde)\s+tan\b", "dónde están"),
        (r"\bta\s+(pelao|pelado|escaso|escasos|escasas|flojo|vac[ií]o|agotao|agotado|bueno|malo|guardado|metido|ubicado)\b", r"está \1"),
        (
            r"\btan\s+(pelao|pelado|escaso|escasos|escasas|flojo|flojos|flojas|cortos|cortas|vac[ií]o|vac[ií]os|agotao|agotaos|agotado|agotados|agotadas|prestado|prestados|prestadas|disponibles)\b",
            r"están \1",
        ),

        # Apócope de preposiciones y cuantificadores (protegiendo prefijos de código como PA-001)
        (r"\bpa\b(?!\s*[-_]\s*\d)(?!\s+\d{2,5}\b)", "para"),
        (r"\bp'a\b", "para"),
        (r"\bt[oó]\s+los?\b", "todos los"),
        (r"\bt[oó]\s+las?\b", "todas las"),

        # Pérdida de /s/ plural en sustantivos, verbos (1ª plural) y artículos clave
        (
            r"\blo\s+(movimiento|pr[eé]stamo|traslado|producto|art[ií]culo|cable|equipo)\b",
            r"los \1s",
        ),
        (
            r"\blo\s+(movimientos|pr[eé]stamos|devoluciones|traslados|productos|art[ií]culos|almacenes|cables|chismes|aparatos|conectores|equipos)\b",
            r"los \1",
        ),
        (
            r"\bla\s+(ponchadoras|devoluciones|entradas|salidas|categor[ií]as|bodegas|herramientas|cosas|vainas)\b",
            r"las \1",
        ),
        (r"\b(cu[aá]les\s+son)\s+la\s+categor[ií]a\b", r"\1 las categorías"),
        (r"\b(manej|necesit|us|llev|sac|prest|guard|registr|busc|compr)amo\b", r"\1amos"),
        (r"\btenemo\b", "tenemos"),
        (r"\bqu[eé]\s+vainas?\s+(est[aá]n?|hay|tenemos|tienen|qued(?:[oó]|aron)|hace\s+falta|hay\s+que)\b", r"qué productos \1"),
        (r"\b(?:eto|ehto)\s+cables?\b", "estos cables"),
        (r"\b(?:eto|ehto)\b", "estos"),

        # Aglutinación de clítico + artículo en transcripción STT ("muéstramelos movimientos" -> "muéstrame los movimientos")
        (
            r"\b(mu[eé]strame|mu[eé]streme|mostrame|ens[eé]ñame|dame|l[ií]stame)los\s+(movimientos?|pr[eé]stamos?|productos?|art[ií]culos?|traslados?)\b",
            r"\1 los \2",
        ),
        (
            r"\b(mu[eé]strame|mu[eé]streme|mostrame|ens[eé]ñame|dame|l[ií]stame)las\s+(devoluciones?|entradas?|salidas?|categor[ií]as?|herramientas?)\b",
            r"\1 las \2",
        ),

        # Voseo imperativo y reducción rápida de diptongo en habla veloz ("muéstrame" -> "métrame"/"méstrame"/"mástrame")
        (r"\b(?:mostrame|m[aáeé]s?trame)\b", "muéstrame"),
        (r"\bdecime\b", "dime"),
        # Confusión fonética frecuente de Whisper: "Almacén D/A/B" -> "Amazon D/A/B", "Quién tiene..." -> "Bien, tiene..."
        (r"\bamazon\s+([a-z0-9]{1,3})\b", r"almacén \1"),
        (r"^bien[\s,]+(tiene[ns]?)\b", r"quién \1"),

        # Enlace fonético en español hablado rápido (s + h muda de «hubo» -> «subo»/«suba»/«a su hubo»/«se usa», «están agotados» -> «de tan agotados»/«otálogo», «stock mínimo» -> «éxito/esto/este stock mínimo»)
        (
            r"\b(entrada|salida|pr[eé]stamo|traslado|movimiento)s?\s+(?:sub[oa]|(?:a\s+)?su\s+hubo|se\s+usa(?=\s+(?:hoy|ayer|anteayer|esta\s+semana|este\s+mes|el\s+mes|la\s+semana)))\b",
            r"\1s hubo",
        ),
        (
            r"\bdevoluci[oó]n(?:es)?\s+(?:sub[oa]|(?:a\s+)?su\s+hubo|se\s+usa(?=\s+(?:hoy|ayer|anteayer|esta\s+semana|este\s+mes|el\s+mes|la\s+semana)))\b",
            "devoluciones hubo",
        ),
        (
            r"\b(y|categor[ií]a(?:\s+de)?)\s+con\s+actividad\b",
            r"\1 conectividad",
        ),
        (
            r"\bde\s+tan\s+(agotad[oa]s?|pela[oa]s?|escas[oa]s?|disponibles?|activos?)\b",
            r"están \1",
        ),
        (
            r"\best[aá]n?\s+(?:en\s+el\s+)?(?:ot[aá]logos?|aut[aá]logos?)\b",
            "están agotados o",
        ),
        (
            r"\bqu[eé]\s+producto\s+est[aá]n\b",
            "qué productos están",
        ),
        (
            r"\b(?:por\s+debajo\s+de(?:l)?|bajo)\s+(?:(?:el|su|este)\s+)?(?:[eé]xito|esto|este\s+stock|toque)\s+m[ií]nimo\b",
            "por debajo del stock mínimo",
        ),
        (
            r"\bde\s+la\s+almac[eé]n\b",
            "del almacén",
        ),

        # Giros costeños y coloquiales de consulta general (solo lectura, acotados al dominio de almacén/inventario)
        (
            r"\b(?:c[oó]mo|qu[eé]\s+tal)\s+(?:va|anda|est[aá]|estuvo|fue|sigue|pinta)\s+(?:la\s+vaina|la\s+cosa|la\s+vuelta|la\s+movida|el\s+voleo)\s+(?:hoy\s+)?en\s+(?:el\s+almac[eé]n|la\s+bodega|el\s+inventario)(?:\s+hoy)?\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bqu[eé]\s+voleo\s+(?:hay|hubo)(?:\s+de)?\s+(hoy|ayer|anteayer|anoche|esta\s+semana|este\s+mes)(?:\s+en\s+(?:el\s+almac[eé]n|la\s+bodega|el\s+inventario))?\b",
            r"qué movimientos hubo \1",
        ),
        (
            r"\b(?:[eé]chale\s+un\s+ojo\s+a|revisa(?:me)?)\s+lo\s+que\s+se\s+movi[oó]\b",
            "muéstrame los movimientos que hubo",
        ),
        (
            r"\bqu[eé]\s+(?:tanto\s+)?se\s+ha\s+movido\s+(?:de\s+)?hoy\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bqu[eé]\s+hay\s+en\s+caleta\b",
            "qué productos hay disponibles",
        ),
        (
            r"\b(?:bajit[oa]s?|floj[oa]s?)\s+de\s+(?:stock|existencias?|unidades?)\b",
            "bajo stock",
        ),
        (
            r"\ben\s+alerta(?:s)?\s+de\s+(?:stock|inventario|reposici[oó]n)\b",
            "que requieren reposición",
        ),
        (
            r"\bqu[eé]\s+(?:cosas?|vainas?|productos?|art[ií]culos?)\s+est[aá]n\s+escas[oa]s?\b",
            "qué productos están por debajo de su stock mínimo",
        ),

        # Prótesis / aglutinación fonética de /s/ final con el verbo tenemos ("cable estenemos" -> "cables tenemos")
        (r"\bcable\s+estenemos\b", "cables tenemos"),
        (r"\bestenemos\b", "tenemos"),

        # Términos técnicos y de inventario con variaciones fonéticas o transcripciones Whisper
        (r"\b(?:cableud|cable\s+ud)\s+(?:de\s+)?(?:pecaceis|peca\s+ceis|peca\s+6)\b", "cable UTP Cat 6"),
        (r"\b(?:cableud|cable\s+ud)\b", "cable UTP"),
        (r"\b(?:pecaceis|peca\s+ceis)\b", "Cat 6"),
        (r"\b(?:barro|barros|bajo)\s+(?:stop|estoco|estock|estoc|top)\b", "bajo stock"),
        (r"\b(?:bajo\s+stop)\b", "bajo stock"),
        (
            r"(?<!bot[oó]n\s)\b(sin|con|de|hay|qu[eé]|cu[aá]nto|cu[aá]ntos|cu[aá]ntas|buen|buenos|cero|tienen?|ver|mostrar|mu[eé]strame|dame|consultar|revisar|el)\s+stop\b(?!\s+de\s+(?:emergencia|seguridad|paro|freno))",
            r"\1 stock",
        ),
        (
            r"\b(sin|con|hay|cu[aá]nto|cu[aá]ntos|cu[aá]ntas|buen|buenos|cero|tienen?)\s+top\b(?!\s+(?:\d+|uno|dos|tres|cuatro|cinco|diez|de\s+))",
            r"\1 stock",
        ),
        (r"\b(?:es\s+to[ck]k?|estoc[oa]?|est[oó]k|est[oó]c|estock|stoc|stok)\b", "stock"),
        (r"\b(?:almas[eé]n|arma[zs][eé]n|armazen)\b", "almacén"),
        (r"\b(?:almasenes|arma[zs]enes|armazenes)\b", "almacenes"),
        (r"\b(el|al|del|en)\s+almacena\b", r"\1 Almacén A"),
        (r"\babotad([oa]s?)\b", r"agotad\1"),
        (r"\by\s+soprop[ií]lic([oa]s?)\b", r"isopropílic\1"),
        (r"\bha[ck]u\s*categor[ií]as?\b", "qué categorías"),
        (r"\bproducto\s+se\s+est[aá]n\b", "productos están"),
        (r"\b(?:volegas|bolegas|bodejas|boderas)\b", "bodegas"),
        (r"\b(?:volega|bolega|bodeja|bodera)\b", "bodega"),
    ]

    # Demostrativos y comodines ambiguos que requieren resolución o aclaración
    DEMOSTRATIVOS_AMBIGUOS = [
        r"\b(?:esa\s+vaina|esta\s+vaina|la\s+vaina)\b",
        r"\b(?:este\s+volado|ese\s+volado)\b",
        r"\b(?:ese\s+chisme|este\s+chisme|los\s+chismes)\b",
        r"\b(?:ese\s+aparato|este\s+aparato)\b",
        r"\b(?:ese\s+cable|este\s+cable)\b",
        r"\b(?:esa\s+caja|esta\s+caja)\b",
        r"\b(?:ese\s+producto|este\s+producto|aquel\s+producto)\b",
        r"\b(?:el\s+anterior|lo\s+anterior|producto\s+anterior)\b",
        r"\b(?:de\s+eso|de\s+esos|de\s+esas|de\s+aquellos)\b",
        r"\b(?:eso\s+all[aá]|eso)\b",
        r"\b(?:el\s+coroto|los\s+corotos)\b",
        r"\b(?:esa\s+cosa|esta\s+cosa)\b",
    ]

    # Mapeos de giros verbales costeños a expresiones estándar de inventario
    GIROS_VERBALES = [
        # 1. Consultas generales de actividad / estado del almacén
        # Patrón: (verbo de estado: cómo va/anda/está/estuvo/fue) + (comodín: vaina/cosa/vuelta/movida/voleo/jugada/trabajo/día/jornada) + (tiempo: hoy/el día/la jornada)?
        (
            r"\b(?:c[oó]mo|qu[eé]\s+tal)\s+(?:va|anda|est[aá]|estuvo|fue|sigue|pinta)\s+(?:la\s+vaina|la\s+cosa|la\s+vuelta|la\s+movida|el\s+voleo|la\s+jugada|el\s+trabajo|el\s+d[ií]a|la\s+jornada|eso)\s*(?:de\s+hoy|hoy|el\s+d[ií]a|esta\s+mañana|ahorita)?\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\b(?:c[oó]mo|qu[eé]\s+tal)\s+(?:estuvo|fue)\s+(?:la\s+vaina|la\s+cosa|la\s+vuelta|la\s+movida|el\s+voleo|la\s+jugada|el\s+trabajo|el\s+d[ií]a|la\s+jornada)\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bc[oó]mo\s+(?:va|anda|est[aá]|sigue)\s+(?:la\s+vaina|la\s+cosa|la\s+vuelta|la\s+movida|el\s+voleo)\s+(?:de\s+)?hoy\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bc[oó]mo\s+va\s+la\s+vaina\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bc[oó]mo\s+est[aá]\s+la\s+vaina\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bc[oó]mo\s+va\s+eso\s+(?:de\s+)?hoy\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bqu[eé]\s+(?:tanto\s+)?se\s+ha\s+movido\s+(?:de\s+)?hoy\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bqu[eé]\s+se\s+ha\s+movido\s+hoy\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bqu[eé]\s+voleo\s+hay\s+(?:de\s+)?hoy\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bc[oó]mo\s+va\s+el\s+d[ií]a(?:\s+de\s+hoy)?\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bqu[eé]\s+tan\s+movid[oa]\s+(?:estuvo|ha\s+estado|anda)\s+(?:el\s+d[ií]a|la\s+cosa)\b",
            "cómo van los movimientos de hoy",
        ),
        (
            r"\bc[oó]mo\s+(?:est[aá]|estuvo|va)\s+el\s+voleo\b",
            "cómo van los movimientos de hoy",
        ),

        # 2. Consultas de inventario y almacén (resumen)
        (r"\bamazon\s+([a-z0-9]{1,3})\b", r"almacén \1"),
        (r"\bdime\s+c[oó]mo\s+va\s+el\s+almac[eé]n\b", "cómo va el almacén"),
        (r"\bc[oó]mo\s+va\s+la\s+cosa\s+en\s+el\s+almac[eé]n\b", "cómo va el almacén"),
        (r"\bc[oó]mo\s+pinta\s+el\s+almac[eé]n\b", "cómo va el almacén"),
        (r"\bc[oó]mo\s+anda\s+el\s+inventario\b", "cómo va el almacén"),
        (r"\bdame\s+el\s+resumen\s+de\s+la\s+bodega\b", "resumen del almacén"),

        # 3. Consultas de movimientos por producto
        (r"\bqu[eé]\s+tanto\s+ha\s+salido\s+del?\b", "qué tanto ha salido de"),
        (r"\bqu[eé]\s+tanto\s+se\s+ha\s+movido\b", "cómo se ha movido"),
        (r"\bc[oó]mo\s+se\s+ha\s+despachado\b", "cómo se ha movido"),
        (r"\bqu[eé]\s+pas[oó]\s+con\s+(?:el|la|los|las|ese|esa|aquel|aquella)\b", "cómo se ha movido"),
        (r"\bqu[eé]\s+movimiento\s+tuvo\b", "cómo se ha movido"),
        (r"\bc[oó]mo\s+ha\s+estado\s+saliendo\b", "cómo se ha movido"),

        # 4. Stock y disponibilidad en bodega
        (r"\bcu[aá]nto\s+queda\s+de\b", "cuánto stock hay de"),
        (r"\bcu[aá]ntos\s+quedan\s+de\b", "cuánto stock hay de"),
        (r"\bcu[aá]ntas\s+quedan\s+de\b", "cuánto stock hay de"),
        (r"\bquedan\s+muchos\b", "cuánto stock queda"),
        (r"\bquedan\s+pocos\b", "cuánto stock queda"),
        (r"\bquedan\s+bastantes\b", "cuánto stock queda"),
        (r"\bhay\s+bastante\b", "cuánto stock queda"),
        (r"\btodav[ií]a\s+hay\b", "cuánto stock queda"),
        (r"\bqued[oó]\s+vac[ií]o\b", "agotado"),
        (r"\bse\s+acab[oó]\b", "agotado"),
        (r"\bno\s+hay\s+nada\b", "agotado"),
        (r"\bqu[eé]\s+hay\s+en\s+caleta\b", "qué hay disponible"),

        # Alertas coloquiales ("está escaso", "están escasas", "está flojo", "está pelado")
        (r"\bqu[eé]\s+cosas\s+est[aá]n\s+escasas?\b", "qué productos tienen bajo stock"),
        (r"\best[aá]n?\s+escas[oa]s?\b", "bajo stock"),
        (r"\best[aá]\s+flojo\b", "alerta bajo stock"),
        (r"\best[aá]\s+pelado\b", "alerta bajo stock"),
        (r"\bqued[oó]\s+pelado\b", "alerta bajo stock"),

        # Consultas de quién tiene un producto
        (r"\bqui[eé]n\s+tiene\s+(?:los?|las?|el|la)?\s*([a-zA-Z0-9\s]+?)\s+ahora\s+mismo\b", r"quién tiene prestado \1"),

        # Ubicación
        (r"\bd[oó]nde\s+(?:dejaron|metieron|tiraron|guardaron)\s+(?:el|la|los|las)\b", "dónde está "),
        (r"\ben\s+qu[eé]\s+lado\s+(?:metieron|dejaron|tiraron|guardaron|est[aá]n?)\b", "dónde está "),

        # PRÉSTAMO: préstame, déjame uno, pásame uno prestado, anótale ese préstamo a Carlos
        (r"\bpr[eé]stame\b", "préstamo de"),
        (r"\bd[eé]jame\s+uno\b", "préstamo de 1"),
        (r"\bd[eé]jame\s+una\b", "préstamo de 1"),
        (r"\bp[aá]same\s+un[oa]\s+prestado\b", "préstamo de 1"),
        (r"\ban[oó]tale\s+((?:ese\s+)?pr[eé]stamo)\s+a\b", r"\1 para"),
        (r"\ban[oó]taselo\s+a\b", "para"),

        # SALIDA / DOTACIÓN DEFINITIVA: sácame, despáchame, entregar definitivamente
        (r"\beso\s+no\s+lo\s+voy\s+a\s+prestar\s*,\s*se\s+lo\s+voy\s+a\s+entregar\s+definitivamente\b", "entrega definitiva de eso"),
        (r"\bno\s+lo\s+voy\s+a\s+prestar\s*,\s*(?:se\s+lo\s+voy\s+a\s+)?entregar\s+definitivamente\b", "entrega definitiva de eso"),
        (r"\bno\s+es\s+pr[eé]stamo\s*,\s*(?:es\s+)?(?:dotaci[oó]n|entrega\s+definitiva)\b", "entrega definitiva de eso"),
        (r"\bs[aá]came\b", "salida de"),
        (r"\bdesp[aá]chame\b", "salida de"),
        (
            r"\bm[aá]ndame\b(?!\s+(?:el\s+|la\s+|los\s+|las\s+)?(?:lista|listado|stock|existencias?|inventario|resumen|alertas?|faltantes?|pr[eé]stamos?|movimientos?|ubicaci[oó]n(?:es)?|almac[eé]n(?:es)?|categor[ií]as?|c[oó]digo|ficha|datos?|informaci[oó]n|reporte)\b)",
            "salida de",
        ),
        (r"\bb[aá]jame\b", "salida de"),
        (r"\bsaca\s+ah[ií]\b", "salida de"),
        (r"\bs[aá]calo\b", "salida de"),
        (r"\bdesp[aá]chalo\b", "salida de"),
        (r"\bb[aá]jalo\b", "salida de"),
        (
            r"\bp[aá]same\b(?!\s+(?:el\s+|la\s+|los\s+|las\s+)?(?:lista|listado|stock|existencias?|inventario|resumen|alertas?|faltantes?|pr[eé]stamos?|movimientos?|ubicaci[oó]n(?:es)?|almac[eé]n(?:es)?|categor[ií]as?|c[oó]digo|ficha|datos?|informaci[oó]n|reporte)\b)",
            "salida de",
        ),

        # ENTRADA: mete, mételo, emboca, embócalo, agrégale, llegó, llegaron
        (r"\b(?:mete|metan|emboca)\b", "entrada de"),
        (r"\bm[eé]telo\b", "entrada de"),
        (r"\bemb[oó]calo\b", "entrada de"),
        (r"\bagr[eé]gale\b", "entrada de"),
        (r"\blleg[oó]\b(?!\s+(?:la\s+devoluci[oó]n|el\s+pr[eé]stamo|a\s+(?:su\s+|el\s+)?(?:stock\s+)?m[ií]nimo|a\s+cero)\b)", "entrada de"),
        (r"\bllegaron\b(?!\s+(?:las\s+devoluciones|los\s+pr[eé]stamos|a\s+(?:su\s+|el\s+)?(?:stock\s+)?m[ií]nimo|a\s+cero)\b)", "entrada de"),

        # DEVOLUCIÓN: te regreso, ya volvió, devuelvo, devuelve uno nada más
        (r"\bdevuelve\s+(?:uno|1)\s+nada\s+m[aá]s\b", "devolución de 1"),
        (r"\bdevuelve\s+solo\s+(?:uno|1)\b", "devolución de 1"),
        (r"\bdevuelve\s+(\d+)\s+nada\s+m[aá]s\b", r"devolución de \1"),
        (r"\bdevuelve\s+uno\b", "devolución de 1"),
        (r"\bdevuelve\s+(\d+)\b", r"devolución de \1"),
        (r"\bte\s+regreso\b", "devolución de"),
        (r"\bya\s+volvi[oó]\b", "devolución de"),
        (r"\bdevuelvo\b", "devolución de"),
        (r"\blo\s+traje\s+otra\s+vez\b", "devolución de"),

        # Búsqueda
        (r"\bb[uú]scame\s+el\s+([a-zA-Z0-9\s]+?)\s+ese\s+([a-zA-Z0-9\s]+)\b", r"buscar \1 \2"),
    ]

    @classmethod
    def normalizar_fonetica_whisper(cls, texto: str) -> str:
        """Corrige errores fonéticos frecuentes de Faster-Whisper en habla caribeña."""
        if not texto:
            return ""
        resultado = texto
        for pat, rep in cls.NORMALIZACIONES_FONETICAS:
            resultado = re.sub(pat, rep, resultado, flags=re.IGNORECASE)
        return resultado

    @classmethod
    def normalizar(cls, texto: str) -> ExpresionAnalizada:
        """Limpia el texto, detecta si es ambiguo y aisla la intención."""
        if not texto:
            return ExpresionAnalizada(
                texto_original="",
                texto_estandar="",
                vocativo_detectado=None,
                es_ambiguo=True,
                demostrativo_detectado=None,
                producto_mencionado=None,
                cantidad_detectada=None,
            )

        t_clean = cls.normalizar_fonetica_whisper(texto).strip()
        vocativo = None

        # 1. Quitar vocativos iniciales (ej. "Haku búscame...", "Oye Haku...")
        for pat in cls.VOCATIVOS:
            m = re.search(pat, t_clean, flags=re.IGNORECASE)
            if m:
                vocativo = m.group(0).strip()
                t_clean = re.sub(pat, "", t_clean, flags=re.IGNORECASE).strip()
                # Limpiar comas iniciales que queden: ", búscame..."
                t_clean = re.sub(r"^[,\s]+", "", t_clean)
                break

        # 2. Aplicar giros verbales costeños
        estandarizado = t_clean
        for pat, reemplazo in cls.GIROS_VERBALES:
            estandarizado = re.sub(pat, reemplazo, estandarizado, flags=re.IGNORECASE)

        # 3. Detectar demostrativos ambiguos ("esa vaina", "ese cable", "de esos", etc.)
        demostrativo = None
        for pat in cls.DEMOSTRATIVOS_AMBIGUOS:
            m = re.search(pat, estandarizado, flags=re.IGNORECASE)
            if m:
                demostrativo = m.group(0).strip()
                break

        # 4. Extraer cantidad operativa real (ignorando códigos como TAL-002, medidas como 20V/20B y artículos como «un préstamo»)
        from .numeros import extraer_cantidad_operacion_texto
        cantidad = extraer_cantidad_operacion_texto(estandarizado)

        # 5. Detección de ambigüedad
        # Si la frase contiene un demostrativo comodín ("esa vaina", "ese cable", "de esos")
        # y NO especifica marcas, modelos o códigos concretos (ej. no dice "HDMI", "UTP", "RJ45", "CAB-0001")
        es_ambiguo = False
        terminos_tecnicos = re.search(
            r"\b(HDMI|UTP|RJ45|Switch|Ponchadora|Patch\s*Cord|VGA|Router|Cat\s*6|CAB-\d+|HER-\d+|RED-\d+)\b",
            estandarizado,
            flags=re.IGNORECASE,
        )

        if demostrativo is not None and not terminos_tecnicos:
            # Ejemplo: "pásame ese cable", "búscame esa vaina", "sácame tres de esos"
            es_ambiguo = True

        return ExpresionAnalizada(
            texto_original=texto,
            texto_estandar=estandarizado,
            vocativo_detectado=vocativo,
            es_ambiguo=es_ambiguo,
            demostrativo_detectado=demostrativo,
            producto_mencionado=terminos_tecnicos.group(0) if terminos_tecnicos else None,
            cantidad_detectada=cantidad,
        )
