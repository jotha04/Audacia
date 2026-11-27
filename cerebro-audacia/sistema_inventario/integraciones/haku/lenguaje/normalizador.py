"""Normalizador lingüístico principal y generador de intenciones para Haku Inventario.

REGLAS INNEGOCIABLES:
1. Nunca ejecuta movimientos ni altera la base de datos: solo produce intenciones estructuradas.
2. Ante ambigüedad ("pásame ese cable", "esa vaina") solicita aclaración sin inventar productos.
3. No asume equivalencias de códigos (CAB-001 vs CAB-0001): busca coincidencias y solicita aclaración si hay más de una.
4. Contexto conversacional efímero (máximo 30 min, usuario autenticado, sin persistencia permanente).
5. Separación dialectal: el dialecto costeño pertenece únicamente a la entrada; todos los mensajes generados hacia el usuario son 100% formales, neutrales y empresariales.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Set


def _quitar_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

from django.core.exceptions import PermissionDenied
from django.db.models import Q

from cuentas.models import UsuarioAudacIA
from inventario.models import Almacen, Categoria, Existencia, Movimiento, Prestamo, Producto

from ..cache import HakuCacheManager
from ..servicios_personas import capitalizar_nombre_propio
from .costeno import InterpretadorCosteno, sanitizar_demostrativo_salida
from .siglas import NormalizadorSiglasTecnicas

PREFIJOS_CONVERSACIONALES_DESTINATARIO = re.compile(
    r"^(?:por\s+favor\s+)?"
    r"(?:"
    r"se\s+lo\s+(?:voy\s+a\s+|va\s+a\s+|vas\s+a\s+)?(?:prestar|entregar|dar)(?:\s+a)?|"
    r"se\s+lo\s+(?:presto|entrego|doy)(?:\s+a)?|"
    r"(?:pr[eé]staselo|pr[eé]stale|entr[eé]gaselo|entr[eé]gale|d[aá]selo|d[aá]le)\s+a|"
    r"a\s+nombre\s+de|"
    r"el\s+destinatario\s+es|"
    r"la\s+persona\s+(?:se\s+llama|es)|"
    r"se\s+llama|"
    r"es\s+para|"
    r"va\s+para|"
    r"para|"
    r"a"
    r")\s+",
    re.IGNORECASE,
)


class AlcanceTipo(str, Enum):
    """Alcance o cardinalidad de la consulta en el inventario."""
    GLOBAL = "GLOBAL"
    ALMACEN = "ALMACEN"
    COLECCION = "COLECCION"
    INDIVIDUAL = "INDIVIDUAL"


class CondicionTipo(str, Enum):
    """Filtro o propiedad de interés sobre el stock de inventario."""
    NINGUNA = "NINGUNA"
    DISPONIBLE = "DISPONIBLE"
    AGOTADO = "AGOTADO"
    BAJO_STOCK = "BAJO_STOCK"
    EN_PRESTAMO = "EN_PRESTAMO"
    ALERTAS = "ALERTAS"


class ConceptoTipo(str, Enum):
    """Concepto u operación objetivo de la interacción."""
    STOCK = "STOCK"
    UBICACION = "UBICACION"
    PRESTAMO = "PRESTAMO"
    DEVOLUCION = "DEVOLUCION"
    MOVIMIENTO = "MOVIMIENTO"
    RESUMEN_GENERAL = "RESUMEN_GENERAL"
    CATALOGO = "CATALOGO"
    ALMACENES = "ALMACENES"
    DESCONOCIDO = "DESCONOCIDO"


class ModalidadTipo(str, Enum):
    """Modalidad de la expresión del operario."""
    CONSULTA = "CONSULTA"
    MUTACION = "MUTACION"
    CONFIRMAR = "CONFIRMAR"
    CANCELAR = "CANCELAR"
    SALUDO = "SALUDO"
    AGRADECIMIENTO = "AGRADECIMIENTO"
    DESCONOCIDO = "DESCONOCIDO"


@dataclass(frozen=True)
class SemanticFrame:
    """Marco semántico estructurado: separa modalidad, alcance, concepto y condición."""
    modalidad: ModalidadTipo = ModalidadTipo.CONSULTA
    alcance: AlcanceTipo = AlcanceTipo.INDIVIDUAL
    concepto: ConceptoTipo = ConceptoTipo.STOCK
    condicion: CondicionTipo = CondicionTipo.NINGUNA
    almacen_especifico: Optional[Almacen] = None
    filtro_coleccion: Optional[str] = None
    producto_objetivo: Optional[Producto] = None
    destinatario: Optional[str] = None
    cantidad: Optional[int] = None
    motivo: Optional[str] = None


class IntencionTipo(str, Enum):
    """Tipos de intención semántica reconocidos por Haku Inventario."""
    BUSCAR_PRODUCTO = "BUSCAR_PRODUCTO"
    CONSULTAR_STOCK = "CONSULTAR_STOCK"
    CONSULTAR_UBICACION = "CONSULTAR_UBICACION"
    CONSULTAR_UBICACION_COLECCION = "CONSULTAR_UBICACION_COLECCION"
    ALERTAS_STOCK = "ALERTAS_STOCK"
    PRODUCTOS_SIN_STOCK = "PRODUCTOS_SIN_STOCK"
    PRODUCTOS_BAJO_STOCK = "PRODUCTOS_BAJO_STOCK"
    CONSULTAR_FALTANTES_REPOSICION = "CONSULTAR_FALTANTES_REPOSICION"
    LISTAR_CATALOGO = "LISTAR_CATALOGO"
    PRODUCTOS_CON_STOCK = "PRODUCTOS_CON_STOCK"
    LISTAR_ALMACENES = "LISTAR_ALMACENES"
    LISTAR_UBICACIONES = "LISTAR_UBICACIONES"
    PRESTAMOS_ACTIVOS = "PRESTAMOS_ACTIVOS"
    REGISTRAR_SALIDA = "REGISTRAR_SALIDA"
    REGISTRAR_ENTRADA = "REGISTRAR_ENTRADA"
    PRESTAMO = "PRESTAMO"
    DEVOLUCION = "DEVOLUCION"
    TRASLADAR_PRODUCTO = "TRASLADAR_PRODUCTO"
    CONSULTAR_METRICAS_MOVIMIENTOS = "CONSULTAR_METRICAS_MOVIMIENTOS"
    CONSULTAR_RESUMEN_INVENTARIO = "CONSULTAR_RESUMEN_INVENTARIO"
    CONSULTAR_PRODUCTO_MOVIMIENTOS = "CONSULTAR_PRODUCTO_MOVIMIENTOS"
    CREAR_PRODUCTO = "CREAR_PRODUCTO"
    EDITAR_PRODUCTO = "EDITAR_PRODUCTO"
    INHABILITAR_PRODUCTO = "INHABILITAR_PRODUCTO"
    HABILITAR_PRODUCTO = "HABILITAR_PRODUCTO"
    CREAR_USUARIO = "CREAR_USUARIO"
    EDITAR_USUARIO = "EDITAR_USUARIO"
    SOLICITAR_ACLARACION = "SOLICITAR_ACLARACION"
    SALUDO = "SALUDO"
    AGRADECIMIENTO = "AGRADECIMIENTO"
    ACCION_CONFIRMAR = "ACCION_CONFIRMAR"
    ACCION_CANCELAR = "ACCION_CANCELAR"
    SOLICITUD_NO_COMPRENDIDA = "SOLICITUD_NO_COMPRENDIDA"
    NO_INTENCION_INVENTARIO = "NO_INTENCION_INVENTARIO"
    CONVERSACION_GENERAL = "CONVERSACION_GENERAL"  # Deprecada: mantener por retrocompatibilidad


@dataclass(frozen=True)
class IntencionResultado:
    """Intención procesada lista para el motor de confirmación o consulta de Haku."""
    tipo: IntencionTipo
    producto: Optional[Producto] = None
    codigo_producto: Optional[str] = None
    cantidad: Optional[int] = None
    destinatario: Optional[str] = None
    motivo: Optional[str] = None
    mensaje_aclaracion: Optional[str] = None
    almacen: Optional[Almacen] = None
    requiere_confirmacion: bool = False
    candidatos: Optional[List[Dict[str, Any]]] = None
    datos_extra: dict[str, Any] = None
    semantic_frame: Optional[SemanticFrame] = None


@dataclass
class RanurasSemanticas:
    """Ranuras semánticas extraídas determinísticamente del texto del operario."""
    acto_de_habla: Optional[str] = None  # "SALUDO", "AGRADECIMIENTO", "CONFIRMAR", "CANCELAR"
    es_afirmacion_pura: bool = False
    es_negacion_pura: bool = False
    es_cambio_de_tema: bool = False
    conceptos_operativos: Set[str] = field(default_factory=set)
    accion: Optional[str] = None
    ventana_temporal: Optional[str] = None
    contexto_fisico: Optional[str] = None
    alcance: str = "INDIVIDUAL"  # "INDIVIDUAL", "COLECCION", "GLOBAL", "ALMACEN"
    condicion: str = "NINGUNA"  # "NINGUNA", "DISPONIBLE", "AGOTADO", "BAJO_STOCK", "EN_PRESTAMO", "ALERTAS"
    metrica_prestamo: str = "GENERAL"  # "GENERAL", "CONTEO_REGISTROS", "CONTEO_UNIDADES", "LISTA_PRODUCTOS", "LISTA_RESPONSABLES"
    destinatario: Optional[str] = None
    motivo: Optional[str] = None
    almacen_especifico: Optional[Almacen] = None
    filtro_coleccion: Optional[str] = None
    semantic_frame: Optional[SemanticFrame] = None


class NormalizadorLenguajeHaku:
    """Analizador lingüístico determinista y sin efectos secundarios en la base de datos."""

    TIMEOUT_CONTEXTO_SEGUNDOS = 1800  # 30 minutos de inactividad conversacional

    @classmethod
    def _resolver_almacen_en_texto(cls, texto: str) -> Optional[Almacen]:
        """Detecta determinísticamente si el texto menciona un almacén o bodega específico por nombre.
        
        Evita falsos positivos con frases genéricas como 'en el almacén' o 'qué almacenes tenemos'.
        """
        texto_norm = _quitar_tildes(texto.lower().strip())
        qs_almacenes = Almacen.objects.filter(activo=True, eliminado=False)
        lista_almacenes = sorted(qs_almacenes, key=lambda a: len(a.nombre), reverse=True)

        for alm in lista_almacenes:
            alm_nom = _quitar_tildes(alm.nombre.lower().strip())
            if re.search(r"\b" + re.escape(alm_nom) + r"\b", texto_norm):
                return alm
            partes = [p for p in alm_nom.split() if p not in ("almacen", "bodega", "de", "el", "la", "los", "las")]
            if partes:
                calificador = " ".join(partes)
                if len(calificador) >= 2 and re.search(r"\b(?:almac[eé]n|bodega)\s+" + re.escape(calificador) + r"\b", texto_norm):
                    return alm
                if len(calificador) >= 4 and re.search(r"\b(?:en|para|de)\s+(?:el\s+|la\s+)?(?:" + re.escape(calificador) + r")\b", texto_norm):
                    if calificador not in cls.PALABRAS_PARADA:
                        return alm
        return None

    @classmethod
    def _detectar_filtro_coleccion(cls, texto: str) -> Optional[str]:
        """Detecta si la consulta se refiere a una colección, familia o categoría de productos.
        
        Retorna la raíz/nombre de la familia (ej. 'cable', 'herramienta') o None.
        Solo aplica si la estructura lingüística es una consulta plural o explícita de colección:
        'qué cables...', 'cuáles herramientas...', 'muéstrame los cables...', 'qué hay de cables...', etc.
        """
        texto_norm = _quitar_tildes(texto.lower().strip())

        # Comandos de acción operativa nunca son consultas de colección
        if re.search(
            r"\b(sacar?|s[aá]came|s[aá]calo|saca|meter?|m[eé]tame|m[eé]telo|mete|prestar?|pr[eé]stame|pr[eé]stalo|presta|despachar?|desp[aá]chame|despacha|devolver?|devu[eé]lveme|devuelve|regresar?|regreso|regresa|traje|traer|trae)\b",
            texto_norm,
        ):
            return None

        patrones = [
            r"\b(?:qu[eé]|cu[aá]les)\s+(?:productos\s+(?:de\s+tipo\s+|de\s+la\s+categor[ií]a\s+)?|art[ií]culos\s+(?:de\s+tipo\s+|de\s+la\s+categor[ií]a\s+)?|tipos\s+de\s+)?([a-z]{3,}(?:\s+de\s+[a-z]{2,})?)\b",
            r"\bmu[eé]strame\s+(?:los|las|el\s+grupo\s+de|la\s+categor[ií]a\s+de)\s+([a-z]{3,}(?:\s+de\s+[a-z]{2,})?)\b",
            r"\blistar?\s+(?:los|las|categor[ií]a\s+de\s+)?([a-z]{3,}(?:\s+de\s+[a-z]{2,})?)\b",
            r"\bqu[eé]\s+hay\s+de\s+([a-z]{3,}(?:\s+de\s+[a-z]{2,})?)\b",
            r"\b(?:hay|tenemos)\s+(?:algunos?|algunas?)\s+([a-z]{3,}(?:\s+de\s+[a-z]{2,})?)\b",
            r"\bcategor[ií]a\s+(?:de\s+)?([a-z]{3,}(?:\s+de\s+[a-z]{2,})?)\b",
            r"\bfamilia\s+(?:de\s+)?([a-z]{3,}(?:\s+de\s+[a-z]{2,})?)\b",
        ]

        palabras_genericas = {
            "producto", "productos", "articulo", "articulos", "item", "items",
            "referencia", "referencias", "cosa", "cosas", "vaina", "vainas",
            "almacen", "almacenes", "bodega", "bodegas", "inventario",
            "stock", "existencia", "existencias", "movimiento", "movimientos",
            "prestamo", "prestamos", "alerta", "alertas", "novedad", "novedades",
            "disponible", "disponibles", "agotado", "agotados", "vacio", "pelado",
            "hoy", "ayer", "mañana", "aqui", "alli", "alla",
            "tipo", "tipos", "clase", "clases", "grupo", "grupos",
            "musica", "noticia", "noticias", "chiste", "chistes", "partido", "partidos",
            "pon", "mete", "saca", "dame", "pasa", "trae", "lleva", "haz",
            "tanto", "saldo", "saldos", "unidad", "unidades", "menos", "mas",
        }

        try:
            from ..services import MAPEO_ALIAS_FAMILIAS
        except ImportError:
            MAPEO_ALIAS_FAMILIAS = {}

        for pat in patrones:
            m = re.search(pat, texto_norm)
            if m:
                cand = m.group(1).lower().strip()
                if cand in palabras_genericas or cand in cls.PALABRAS_PARADA:
                    continue

                if cand in MAPEO_ALIAS_FAMILIAS:
                    return cand

                stems = [cand]
                if cand.endswith("es") and len(cand) > 4:
                    stems.append(cand[:-2])
                elif cand.endswith("s") and len(cand) > 3:
                    stems.append(cand[:-1])

                for stem in stems:
                    if stem in palabras_genericas or stem in cls.PALABRAS_PARADA:
                        continue
                    if stem in MAPEO_ALIAS_FAMILIAS:
                        return stem
                    if Categoria.objects.filter(nombre__icontains=stem, activo=True, eliminado=False).exists():
                        return stem
                    qs_prods = Producto.objects.filter(nombre__icontains=stem, activo=True, archivado=False)
                    if qs_prods.count() >= 1:
                        return stem

        return None

    @classmethod
    def _extraer_ranuras_semanticas(
        cls,
        texto_trabajo: str,
        texto_siglas: str,
        analisis_costeno: Any,
    ) -> RanurasSemanticas:
        """Extrae ranuras semánticas (slots) determinísticamente a partir del texto preprocesado."""
        ranuras = RanurasSemanticas()

        match_dest = re.search(r"\b(?:de|por|a|para|pa)\s+([A-Za-zÁ-ÿ\s]+?)(?:\s*$)", texto_siglas, re.IGNORECASE)
        if match_dest:
            d = match_dest.group(1).strip()
            # Limpiar posible sufijo de almacén (ej: "Carlos Andrés Rodríguez en el Almacén Principal")
            d = re.sub(r"\b(?:en|desde|hacia)\s+(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\b.*$", "", d, flags=re.IGNORECASE).strip()
            # Limpiar sufijo de dotación o motivo (ej: "María Gómez como dotación", "María Gómez por dotación")
            d = re.sub(r"\b(?:como|por|en\s+calidad\s+de)\s+dotaci[oó]n\b.*$", "", d, flags=re.IGNORECASE).strip()
            d = re.sub(r"\b(pr[eé]stamo|pr[eé]stamos|de|el|la|los|las|inventario|almac[eé]n|bodega)\b", "", d, flags=re.IGNORECASE).strip()
            if d:
                ranuras.destinatario = d

        texto_limpio = re.sub(r"[¿?¡!.,:;()\[\]\"'`]", " ", texto_trabajo).strip()

        # 0. Detección de marcadores discursivos de cambio de tema (Topic Shift)
        if re.search(
            r"\b(cambiando\s+de\s+tema|hablando\s+de\s+otra\s+cosa|en\s+otro\s+orden|punto\s+y\s+aparte|aparte\s+de\s+eso|por\s+cierto|otro\s+tema)\b",
            texto_limpio,
        ):
            ranuras.es_cambio_de_tema = True

        # 1. Actos de habla & Afirmación / Negación Pura
        patron_confirmar_token = r"(s[ií]|correcto|adelante|confirma|confirmar|confirmo|dale|listo|h[aá]gale|hazlo|de\s+acuerdo|afirmativo|proceder|ejecutar|ok)"
        patron_cancelar_token = r"(no|nombe|cancelar|cancela|cancela\s+eso|cancelar\s+eso|cancela\s+esa\s+operaci[oó]n|cancelar\s+esa\s+operaci[oó]n|cancela\s+la\s+operaci[oó]n|cancelar\s+la\s+operaci[oó]n|mejor\s+cancela(?:\s+esa\s+operaci[oó]n|\s+eso|\s+la\s+operaci[oó]n)?|anular|anula\s+eso|mejor\s+no|(?:ya\s+)?no\s+(?:lo\s+)?hagas(?:\s+(?:eso|nada))?|olv[ií]dalo|descartar|descarta|descarta\s+eso|descarta\s+esa\s+propuesta|descartar\s+esa\s+propuesta|descarta\s+la\s+propuesta|descartar\s+la\s+propuesta|abortar|d[eé]jalo\s+(?:as[ií]|quieto)|deja\s+eso\s+(?:quieto|as[ií])|det[eé]n|parar)"

        # Texto enfocado en el acto sin partículas de cortesía ni vocativos
        texto_acto = re.sub(r"\b(haku|hacu|por\s+favor|oye|eche|aj[aá]|compae|compadre|compa|cuadro|mi\s+llave)\b", "", texto_limpio).strip()
        texto_acto = re.sub(r"\s+", " ", texto_acto)

        if re.fullmatch(rf"^(?:{patron_confirmar_token}\s*)+$", texto_acto):
            ranuras.es_afirmacion_pura = True
            ranuras.acto_de_habla = "CONFIRMAR"
        elif re.fullmatch(rf"^(?:{patron_cancelar_token}\s*)+$", texto_acto):
            ranuras.es_negacion_pura = True
            ranuras.acto_de_habla = "CANCELAR"
        else:
            if re.search(r"\b(hola(\s+haku)?|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|buen\s+d[ií]a|buenas(\s+buenas)?|saludos|qu[eé]\s+hubo|quihubo)\b", texto_limpio):
                ranuras.acto_de_habla = "SALUDO"
            elif re.search(r"\b(muchas\s+gracias|gracias(\s+haku)?|te\s+agradezco|agradecido|muy\s+amable)\b", texto_limpio):
                ranuras.acto_de_habla = "AGRADECIMIENTO"
            elif re.search(rf"\b{patron_confirmar_token}\b", texto_limpio):
                ranuras.acto_de_habla = "CONFIRMAR"
            elif re.search(rf"\b{patron_cancelar_token}\b", texto_limpio):
                ranuras.acto_de_habla = "CANCELAR"

        # 2. Ventana Temporal
        from ..servicios_temporales import resolver_intervalo_temporal
        intervalo_temp = resolver_intervalo_temporal(texto_limpio)
        if intervalo_temp:
            ranuras.ventana_temporal = intervalo_temp.etiqueta
        elif re.search(r"\b(hoy|d[ií]a|jornada|turno|fecha|hasta\s+ahora|esta\s+mañana|esta\s+tarde|este\s+turno)\b", texto_limpio):
            ranuras.ventana_temporal = "HOY"

        # 3. Contexto Físico de Ubicación Interna
        match_contexto = re.search(
            r"\b(estante|estantes|rack|racks|pasillo|pasillos|gaveta|gavetas|caj[oó]n|cajones|repisa|repisas|secci[oó]n|secciones|posici[oó]n|posiciones|ubicaci[oó]n|ubicaciones|localizaci[oó]n|lugar|lugares)\b",
            texto_limpio,
        )
        if match_contexto:
            ranuras.contexto_fisico = match_contexto.group(1)

        # 4. Almacén Específico y Filtro de Colección
        ranuras.almacen_especifico = cls._resolver_almacen_en_texto(texto_siglas)
        ranuras.filtro_coleccion = cls._detectar_filtro_coleccion(texto_limpio)

        # 5. Detección de Alcance Semántico (Scope)
        es_alcance_coleccion = bool(
            ranuras.filtro_coleccion is not None
            or re.search(
                r"\b(cada\s+(?:producto|referencia|art[ií]culo|uno)|cada\s+uno|todos\s+(?:ellos|los\s+productos|las\s+referencias|los\s+art[ií]culos|los\s+que\s+acabas|esos\s+productos|esos\s+art[ií]culos|los\s+de\s+la\s+lista)|esas\s+referencias|esos\s+art[ií]culos|los\s+art[ií]culos|las\s+referencias|los\s+tres|de\s+los\s+tres|todos\s+ellos|de\s+todos\s+ellos|de\s+todos|en\s+qu[eé]\s+estantes\s+se\s+encuentran|ub[ií]came\s+todos|de\s+cada\s+uno)\b",
                texto_limpio,
            )
            or re.search(
                r"\b(?:las\s+referencias|los\s+art[ií]culos|los\s+productos|lo)\s+que\s+(?:acabas\s+de\s+|dijiste\s+de\s+|listaste\s+|mencionaste\s+)?(?:listar|mencionar|decir|nombrar)\b",
                texto_limpio,
            )
            or re.search(
                r"\b(?:esos|esas)\s+(?:art[ií]culos|referencias|productos)\b",
                texto_limpio,
            )
            or re.search(
                r"\bcu[aá]les\s+de\s+(?:esos|esas|ellos|los\s+productos|las\s+referencias)\b",
                texto_limpio,
            )
        )

        # Detección de préstamos globales y métricas
        es_metrica_prestamo_registros = bool(
            re.search(r"\b(cu[aá]ntos\s+pr[eé]stamos|n[uú]mero\s+de\s+pr[eé]stamos|cantidad\s+de\s+pr[eé]stamos|total\s+de\s+pr[eé]stamos)\b", texto_limpio)
        )
        es_metrica_prestamo_unidades = bool(
            re.search(r"\b(cu[aá]ntas\s+unidades\s+(?:hay\s+|tenemos\s+)?presta(?:d[oa]|o)s?|total\s+de\s+unidades\s+presta(?:d[oa]|o)s?|cantidad\s+de\s+unidades\s+presta(?:d[oa]|o)s?|cu[aá]nto\s+material\s+hay\s+prestado)\b", texto_limpio)
        )
        es_metrica_prestamo_responsables = bool(
            re.search(r"\b(qui[eé]n\s+tiene\s+(?:cosas|productos|material|art[ií]culos|\w+)?\s*presta(?:d[oa]|o)s?|a\s+qui[eé]n\s+(?:se\s+le\s+prest[oó]|est[aá]\s+prestado)|responsables?\s+de\s+pr[eé]stamos?)\b", texto_limpio)
            and not re.search(r"\b(eso|ese|este|aquel|su|el|la|dicho)\b", texto_limpio)
        )

        es_prestamos_global = bool(
            es_metrica_prestamo_registros
            or es_metrica_prestamo_unidades
            or es_metrica_prestamo_responsables
            or re.search(r"\b(hay\s+alg[uú]n\s+producto\s+en\s+pr[eé]stamos?|hay\s+algo\s+prestado|qu[eé]\s+tenemos\s+prestado|qu[eé]\s+est[aá]\s+prestado|qu[eé]\s+(?:productos|art[ií]culos|referencias)\s+est[aá]n\s+prestados?|mu[eé]strame\s+los\s+(?:productos|art[ií]culos)\s+que\s+est[aá]n\s+prestados?|cu[aá]les\s+son\s+los\s+pr[eé]stamos\s+activos?|pr[eé]stamos\s+activos|pr[eé]stamos\s+pendientes|qu[eé]\s+pr[eé]stamos\s+hay)\b", texto_limpio)
            or (re.search(r"\bqui[eé]n\s+tiene\b", texto_limpio) and not re.search(r"\b(eso|ese|este|aquel|su|el|la|dicho)\b", texto_limpio))
            or (re.search(r"\b(pr[eé]stamo|pr[eé]stamos|prestado|prestados)\b", texto_limpio) and re.search(r"\b(hay\s+alg[uú]n|qu[eé]\s+tenemos|qu[eé]\s+hay|cu[aá]les\s+son|todos\s+los|qu[eé]\s+productos)\b", texto_limpio))
        )

        if es_metrica_prestamo_registros:
            ranuras.metrica_prestamo = "CONTEO_REGISTROS"
        elif es_metrica_prestamo_unidades:
            ranuras.metrica_prestamo = "CONTEO_UNIDADES"
        elif es_metrica_prestamo_responsables:
            ranuras.metrica_prestamo = "LISTA_RESPONSABLES"
        elif re.search(r"\b(qu[eé]\s+productos|cu[aá]les\s+productos|qu[eé]\s+art[ií]culos)\b", texto_limpio) and es_prestamos_global:
            ranuras.metrica_prestamo = "LISTA_PRODUCTOS"

        es_agotados_global = bool(
            re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:productos|art[ií]culos|referencias|cosas)\s+(?:no\s+tienen\s+stock|est[aá]n\s+agotad[oa]s?|se\s+agotaron|tienen\s+stock\s+cero|est[aá]n\s+en\s+cero|est[aá]n\s+sin\s+existencias?|sin\s+existencias?|se\s+acabar?on)\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:cosas?|productos?|art[ií]culos?|referencias?)\s+(?:est[aá]n\s+)?agotad[ao]s?\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:cosas?|productos?|art[ií]culos?|referencias?)\s+(?:est[aá]n\s+)?en\s+cero\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:cosas?|productos?|art[ií]culos?|referencias?)\s+(?:no\s+tienen|sin)\s+stock\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:cosas?|productos?|art[ií]culos?|referencias?)\s+se\s+acabar?on\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+(?:productos|art[ií]culos|cosas)\s+(?:no\s+tienen|sin)\s+stock\b", texto_limpio)
            or re.search(r"\b(?:productos|art[ií]culos|cosas|referencias)\s+(?:sin\s+stock|agotad[oa]s?|en\s+cero)\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:est[aá]|est[aá]n|se)\s+agotad[oa]s?\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+se\s+acabar?on\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+no\s+tienen\s+stock\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+est[aá]\s+en\s+cero\b", texto_limpio)
        )

        es_bajo_stock_global = bool(
            re.search(r"\b(?:qu[eé]|cu[aá]les|los\s+que|mu[eé]strame\s+los\s+que)\s+(?:productos\s+|art[ií]culos\s+|cosas\s+)?(?:est[aá]n\s+)?(?:por\s+debajo\s+(?:del?\s+(?:stock\s+)?m[ií]nimo|de\s+lo\s+(?:que\s+)?(?:deber[ií]amos|hay\s+que)\s+tener|de\s+lo\s+necesario)|escas[oa]s?)\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:productos\s+|art[ií]culos\s+)?(?:necesitan?|requieren?)\s+(?:reposici[oó]n|reponer|reabastecimiento|reabastecer)\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:est[aá]n\s+)?por\s+debajo\s+del?\s+m[ií]nimo\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+productos\s+tienen\s+bajo\s+st(?:ock|op)\b", texto_limpio)
            or re.search(r"\b(?:productos|art[ií]culos)\s+(?:con\s+)?bajo\s+st(?:ock|op)\b", texto_limpio)
            or re.search(r"\b(?:alerta|alertas)\s+(?:de\s+)?(?:stock|inventario)\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+alertas\s+(?:hay|tenemos)\b", texto_limpio)
            or re.search(r"\b(?:en\s+)?barro\s+st(?:ock|op)\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+cosas\s+est[aá]n\s+escasas?\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+est[aá]n\s+escas[oa]s?\b", texto_limpio)
        )

        es_catalogo_resumen_global = bool(
            re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:productos|art[ií]culos)\s+(?:hay|tenemos|existen|est[aá]n\s+registrados|manejamos)\b", texto_limpio)
            or re.search(r"\b(?:cat[aá]logo|listado|lista)\s+de\s+productos\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+hay\s+en\s+el\s+(?:cat[aá]logo|inventario|almac[eé]n|bodega)\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:son\s+los\s+)?(?:almacenes|bodegas)\b", texto_limpio)
            or re.search(r"\b(?:dime|listar?|ver|mostrar|cu[aá]les\s+son)\s+(?:los\s+|las\s+)?(?:almacenes|bodegas)(?:\s+disponibles|\s+activ[oa]s)?\b", texto_limpio)
            or re.search(r"\b(?:almacenes|bodegas)\s+(?:disponibles|activ[oa]s)\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+(?:almacenes|bodegas)\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+productos\s+(?:hay|tenemos|est[aá]n|quedan)\s+en\s+stock\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+(?:hay|tenemos|queda)\s+en\s+stock\b", texto_limpio)
            or re.search(r"\b(?:productos|art[ií]culos)\s+(?:en\s+stock|con\s+stock|disponibles)\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+(?:tenemos|hay)\s+disponible\b", texto_limpio)
            or re.search(r"\b(?:resumen|balance|estado\s+del?\s+(?:almac[eé]n|inventario|bodega)|c[oó]mo\s+(?:va|anda|est[aá])\s+(?:el\s+almac[eé]n|el\s+inventario|la\s+bodega))\b", texto_limpio)
            or re.search(r"\b(?:consultar?|revisar?|ver|chequear?|dime|consulta)\s+(?:el\s+)?(?:inventario|almac[eé]n|bodega)\b", texto_limpio)
        )

        meses_y_dias_pat = r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)"
        es_movimientos_global = bool(
            (ranuras.ventana_temporal is not None and "MOVIMIENTO" in ranuras.conceptos_operativos)
            or re.search(r"\b(?:mu[eé]strame|dame|ver|consultar?|qu[eé])\s+(?:los\s+)?(?:movimientos?|entradas?|salidas?|traslados?|despachos?)\b", texto_limpio)
            or re.search(rf"\b(?:movimientos?|entradas?|salidas?|traslados?)\s+(?:de\s+|en\s+|durante\s+)?(?:ayer|hoy|la\s+semana|este\s+mes|el\s+mes|los\s+[uú]ltimos|{meses_y_dias_pat})\b", texto_limpio)
            or re.search(r"\b(cu[aá]nt[ao]s?|qu[eé])\s+(?:entradas?|salidas?|despachos?|movimientos?|operaciones|transacciones?|actividades?|actividad|pr[eé]stamos?|devoluciones?)\s+(?:han\s+habido|hubo|se\s+hicieron|se\s+registraron|tuvimos|tuvieron|hay|tenemos)\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]l|cu[aá]les)\s+(?:\w+\s+){0,3}(?:entr[oó]|entraron|ingres[oó]|ingresaron|lleg[oó]|llegaron|sali[oó]|salieron|se\s+despach[oó]|se\s+movi[oó])\b", texto_limpio)
            or re.search(r"\b(movimientos?\s+(?:de\s+)?(?:hoy|ayer)|movimientos?\s+del\s+d[ií]a|c[oó]mo\s+van\s+los\s+movimientos|c[oó]mo\s+va\s+la\s+movida|de\s+movida\s+la\s+vaina|el\s+voleo\s+(?:de\s+)?hoy|qu[eé]\s+voleo\s+hay|qu[eé]\s+se\s+ha\s+movido|c[oó]mo\s+va\s+el\s+voleo|c[oó]mo\s+estuvo\s+el\s+voleo|c[oó]mo\s+va\s+el\s+trabajo\s+hoy|qu[eé]\s+tal\s+estuvo\s+la\s+jornada|c[oó]mo\s+estuvo\s+la\s+jornada|c[oó]mo\s+estuvo\s+la\s+vuelta|c[oó]mo\s+va\s+la\s+vaina|c[oó]mo\s+anda\s+la\s+cosa|c[oó]mo\s+va\s+la\s+cosa|c[oó]mo\s+va\s+eso|qu[eé]\s+pas[oó]\s+hoy\s+en\s+(?:el\s+almac[eé]n|la\s+bodega)|c[oó]mo\s+va\s+el\s+d[ií]a|qu[eé]\s+tan\s+movido)\b", texto_limpio)
        )

        es_alertas_global = bool(
            (
                re.search(r"\b(?:alerta|alertas)\s+(?:de\s+)?(?:stock|inventario)\b", texto_limpio)
                or re.search(r"\bqu[eé]\s+alertas\s+(?:hay|tenemos)\b", texto_limpio)
                or re.search(r"\b(?:agotad[oa]s?\s+o\s+con\s+bajo\s+st(?:ock|op)|bajo\s+st(?:ock|op)\s+o\s+agotad[oa]s?)\b", texto_limpio)
                or re.search(r"\b(?:en\s+)?barro\s+st(?:ock|op)\b", texto_siglas)
                or re.search(r"\b(?:av[ií]same\s+)?(?:lo\s+que|qu[eé]|cu[aá]les)\s+est[eé]\s+en\s+bajo\s+st(?:ock|op)\b", texto_limpio)
                or re.search(r"\b(?:av[ií]same\s+)?(?:lo\s+que|qu[eé]|cu[aá]les)\s+est[eé]\s+en\s+barro\s+st(?:ock|op)\b", texto_siglas)
            )
            and not re.search(r"\b(pelado|pelao|vacio|vac[ií]o|acab[oó]|acabao|escaso|nada)\b", texto_siglas)
            and not re.search(r"\b(?:qu[eé]|cu[aá]les)\s+productos\s+(?:tienen|est[aá]n\s+en|hay\s+en)\s+bajo\s+st(?:ock|op)\b", texto_limpio)
        )

        es_faltantes_global = bool(
            re.search(
                r"\b(?:qu[eé]|cu[aá]l|cu[aá]les)\s+(?:hace\s+falta|falta|hacen\s+falta)\s+(?:por\s+|para\s+)?(?:pedir|comprar|ordenar|reponer|abastecer)\b",
                texto_limpio,
            )
            or re.search(r"\bqu[eé]\s+(?:tengo|tenemos|hay)\s+que\s+(?:pedir|comprar|ordenar|reponer|mandar\s+a\s+pedir)\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+(?:est[aá]|est[aá]n)\s+por\s+(?:acabarse|terminarse|agotarse)\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+debo\s+reponer\b", texto_limpio)
            or re.search(r"\brecomendaci[oó]n\s+de\s+(?:compra|compras|reposici[oó]n)\b", texto_limpio)
            or re.search(r"\bqu[eé]\s+hace\s+falta\s+(?:en\s+el\s+almac[eé]n|en\s+la\s+bodega|comprar|pedir)\b", texto_limpio)
        )

        menciona_producto_potencial = bool(
            re.search(r"\b[A-Za-z]{2,5}[-_]\d{1,4}\b", texto_siglas)
            or re.search(r"\b(?:del?|sobre|con)\s+(?:el\s+|la\s+|este\s+|ese\s+)?(?:cable|hdmi|utp|bobina|switch|router|ponchadora|taladro|taladros|tornillo|tornillos|conector|rj45|rj-45|patch\s*cord)\b", texto_limpio)
            or re.search(r"\b(hdmi|utp|bobina|ponchadora|patch\s*cord|taladro|taladros|tornillo|tornillos)\b", texto_limpio)
        )
        if ranuras.accion in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER"):
            ranuras.alcance = "INDIVIDUAL"
            ranuras.filtro_coleccion = None
        elif es_alcance_coleccion:
            ranuras.alcance = "COLECCION"
        elif (
            (es_prestamos_global and not menciona_producto_potencial)
            or es_agotados_global
            or es_bajo_stock_global
            or es_faltantes_global
            or es_alertas_global
            or es_catalogo_resumen_global
            or (es_movimientos_global and not menciona_producto_potencial)
            or re.search(r"\b(qu[eé]\s+productos|cu[aá]les\s+productos|todos\s+los\s+productos|cat[aá]logo|almacenes|resumen\s+general)\b", texto_limpio)
        ):
            ranuras.alcance = "GLOBAL"
        elif ranuras.almacen_especifico is not None:
            es_consulta_individual_en_almacen = bool(
                (analisis_costeno and getattr(analisis_costeno, "demostrativo_detectado", None))
                or re.search(r"\b(ese|esa|este|esta|aquel|aquella|de\s+es[eo]|de\s+aquel|dicho|anterior|vaina)\b", texto_limpio)
                or re.search(r"\b[A-Za-z]{3}-\d{3,4}\b", texto_siglas)
                or re.search(r"\b(cu[aá]nto\s+queda|cu[aá]nto\s+hay|cu[aá]ntas\s+unidades|d[oó]nde\s+est[aá]|d[oó]nde\s+se\s+encuentra)\b", texto_limpio)
                or re.search(r"\b(cable|hdmi|utp|bobina|switch|router|ponchadora|conector|rj45|rj-45)\b", texto_limpio)
            )
            if es_consulta_individual_en_almacen:
                ranuras.alcance = "INDIVIDUAL"
            else:
                ranuras.alcance = "ALMACEN"
        else:
            ranuras.alcance = "INDIVIDUAL"

        # 6. Detección de Condición / Filtro sobre stock
        if es_prestamos_global or re.search(r"\b(presta(?:d[oa]|o)s?|en\s+pr[eé]stamos?)\b", texto_limpio):
            ranuras.condicion = "EN_PRESTAMO"
            ranuras.conceptos_operativos.add("PRESTAMO")
        elif es_agotados_global or re.search(r"\b(sin\s+stock|no\s+tienen?\s+stock|agotad[oa]s?|en\s+cero|cero\s+stock|acabad[oa]s?|se\s+acabaron|acabaron|se\s+terminaron|terminaron|sin\s+existencias?)\b", texto_limpio):
            ranuras.condicion = "AGOTADO"
            ranuras.conceptos_operativos.add("STOCK")
            ranuras.conceptos_operativos.add("STOCK_AGOTADO")
        elif es_alertas_global:
            ranuras.condicion = "ALERTAS"
            ranuras.conceptos_operativos.add("ALERTA")
        elif es_bajo_stock_global or re.search(r"\b(bajo\s+st(?:ock|op)|por\s+debajo\s+(?:del?\s+(?:stock\s+)?m[ií]nimo|de\s+lo\s+(?:que\s+)?(?:deber[ií]amos|hay\s+que)\s+tener|de\s+lo\s+necesario)|necesitan?\s+reposici[oó]n|requieren?\s+reposici[oó]n|reponer|reabastecer|stock\s+m[ií]nimo|stock\s+insuficiente|escaso)\b", texto_limpio):
            ranuras.condicion = "BAJO_STOCK"
            ranuras.conceptos_operativos.add("STOCK")
            ranuras.conceptos_operativos.add("STOCK_BAJO")
        elif re.search(r"\b(con\s+stock|en\s+stock|disponible|disponibles|con\s+existencias?)\b", texto_limpio):
            ranuras.condicion = "DISPONIBLE"
            ranuras.conceptos_operativos.add("STOCK")
            ranuras.conceptos_operativos.add("STOCK_DISPONIBLE")
        else:
            ranuras.condicion = "NINGUNA"

        # 7. Conceptos Operativos Adicionales
        # MOVIMIENTO:
        if re.search(r"\b(movimientos?|operaci[oó]n|operaciones|transacci[oó]n|transacciones|actividad|actividades|registro|registros|despacho|despachos|flujo|flujos|movida|voleo|vuelta|trabajo|qu[eé]\s+se\s+ha\s+movido|c[oó]mo\s+va\s+la\s+vaina|qu[eé]\s+pas[oó]\s+hoy)\b", texto_limpio):
            ranuras.conceptos_operativos.add("MOVIMIENTO")

        # RESUMEN_GENERAL:
        if re.search(
            r"\b(resumen|balance|estado\s+del?\s+(almac[eé]n|inventario|bodega)|c[oó]mo\s+(va|anda|est[aá]|pinta)\s+(todo|el\s+almac[eé]n|el\s+inventario|la\s+bodega)|(?:consultar?|revisar?|ver|chequear?|dime)\s+(el\s+)?(?:inventario|almac[eé]n|bodega))\b",
            texto_limpio,
        ) or re.search(r"\bconsulta\s+(?:el\s+)?(?:inventario|almac[eé]n|bodega)\b", texto_limpio):
            ranuras.conceptos_operativos.add("RESUMEN_GENERAL")

        # STOCK:
        if re.search(r"\b(stock|existencias?|disponible|disponibilidad|saldo|saldos|unidades?|piezas?|cantidad|cantidades|quedan?|queda|tenemos|contamos|hay|bastante|bastantes|nada|escaso|pelado|vac[ií]o|agotad[oa]s?|acabad[oa]s?|acabaron|se\s+acabaron|se\s+acab[oó]|terminaron|se\s+terminaron)\b", texto_limpio):
            ranuras.conceptos_operativos.add("STOCK")

        # UBICACION:
        if match_contexto or re.search(r"\b(d[oó]nde|ubicaci[oó]n|localizaci[oó]n|posici[oó]n|secci[oó]n|encuentra|encuentran|guardan?|guardamos|guardad[oa]s?|lado|ub[ií]came)\b", texto_limpio):
            ranuras.conceptos_operativos.add("UBICACION")

        # PRESTAMO:
        # Préstamo temporal:
        if re.search(r"\b(prestar|presta|pr[eé]stale|prestale|preste|presten|pr[eé]stamo|pr[eé]stamos|pr[eé]stame|prestado|prestados|d[eé]jame\s+un[oa]|que\s+(?:lo|los|la|las)\s+devuelva)\b", texto_limpio):
            ranuras.conceptos_operativos.add("PRESTAMO")

        # DEVOLUCION:
        if re.search(r"\b(devolver|devoluci[oó]n|devoluciones|regreso|regresar|devuelvo|ya\s+volvi[oó]|lo\s+traje\s+otra\s+vez)\b", texto_limpio):
            ranuras.conceptos_operativos.add("DEVOLUCION")

        # ALERTA:
        if re.search(r"\b(alerta|alertas|bajo\s+stock|sin\s+stock|qu[eé]\s+falta)\b", texto_limpio):
            ranuras.conceptos_operativos.add("ALERTA")

        # 8. Acciones o verbos
        es_consulta_estadistica = bool(
            re.search(r"\b(cu[aá]nt[ao]s?|qu[eé])\s+(?:entradas?|salidas?|despachos?|movimientos?|operaciones|pr[eé]stamos?|devoluciones?)\b", texto_limpio)
            or re.search(r"\b(?:qu[eé]|cu[aá]l|cu[aá]les)\s+(?:\w+\s+){0,3}(?:entr[oó]|entraron|ingres[oó]|ingresaron|lleg[oó]|llegaron|sali[oó]|salieron|se\s+despach[oó]|se\s+movi[oó])\b", texto_limpio)
            or re.search(r"\b(?:han\s+habido|hubo|se\s+hicieron|se\s+registraron|tuvimos|tuvieron)\b", texto_limpio)
            or re.search(r"\b(?:movimientos?|actividad|voleo|despachos?)\s+(?:de\s+)?hoy\b", texto_limpio)
        )

        if es_consulta_estadistica:
            ranuras.accion = "CONSULTAR"
            ranuras.conceptos_operativos.add("MOVIMIENTO")
        elif (
            re.search(r"\b(prestar|presta|pr[eé]stale|prestale|preste|presten|pr[eé]stamo|pr[eé]stame|d[eé]jame\s+un[oa]|an[oó]tale\s+(?:ese\s+)?pr[eé]stamo|que\s+(?:lo|los|la|las)\s+devuelva)\b", texto_limpio)
            and not re.search(r"\b(no\s+(?:lo\s+)?voy\s+a\s+prestar|no\s+es\s+pr[eé]stamo)\b", texto_limpio)
        ):
            ranuras.accion = "PRESTAR"
            ranuras.conceptos_operativos.add("PRESTAMO")
        elif re.search(r"\b(ingresar|ingreso|llegaron|lleg[oó]|recibir|aumentar|meter|mete|m[eé]telo|m[eé]tame|emboca|emb[oó]calo|agr[eé]gale|agregar|registrar\s+entrada|hacer\s+entrada)\b", texto_limpio) or (re.search(r"\bentrada\b", texto_limpio) and not re.search(r"\b(cu[aá]nt[ao]s?|qu[eé]|cu[aá]les?|ver|consultar|mu[eé]str\w*|hubo|esta\s+semana|este\s+mes|ayer|hoy)\b", texto_limpio)):
            ranuras.accion = "REGISTRAR_ENTRADA"
        elif re.search(r"\b(dotaci[oó]n|entrega\s+definitiva|definitivamente|entregar?\s+definitivamente|entregar?\s+dotaci[oó]n|dar\s+dotaci[oó]n|no\s+(?:lo\s+)?voy\s+a\s+prestar|entrega\s+definitiva\s+de\s+eso)\b", texto_limpio):
            ranuras.accion = "REGISTRAR_SALIDA"
            ranuras.motivo = "Dotación / entrega definitiva"
        elif re.search(r"\b(retirar|sacar|saca|s[aá]came|desp[aá]chame|despacha|desp[aá]chalo|m[aá]ndame|b[aá]jame|b[aá]jalo|entregar|p[aá]same|gastar|registrar\s+salida|hacer\s+salida)\b", texto_limpio) or (re.search(r"\bsalida\b", texto_limpio) and not re.search(r"\b(cu[aá]nt[ao]s?|qu[eé]|cu[aá]les?|ver|consultar|mu[eé]str\w*|hubo|esta\s+semana|este\s+mes|ayer|hoy)\b", texto_limpio)):
            ranuras.accion = "REGISTRAR_SALIDA"
        elif re.search(r"\b(devolver|devoluci[oó]n|devu[eé]lv(?:e|en|o|eme|an)|te\s+regreso|regreso|devuelvo|ya\s+volvi[oó]|lo\s+traje\s+otra\s+vez)\b", texto_limpio):
            ranuras.accion = "DEVOLVER"
        elif re.search(r"\b(trasladar|traslada|trasladom?|trasladem?|traslad[aá]me|transferir|transferencia|transfieram?|mover\s+(?:a\s+)?(?:otro\s+almac[eé]n|otra\s+bodega)|pasar\s+(?:a\s+)?(?:otro\s+almac[eé]n|otra\s+bodega)|llevar\s+a\s+(?:otro\s+almac[eé]n|otra\s+bodega)|enviar\s+a\s+(?:otro\s+almac[eé]n|otra\s+bodega)|mandar\s+(?:a\s+|para\s+)(?:otro\s+almac[eé]n|otra\s+bodega|almac[eé]n|bodega))\b", texto_limpio):
            ranuras.accion = "TRASLADAR"
            ranuras.conceptos_operativos.add("MOVIMIENTO")
        elif re.search(r"\b(prestar|presta|pr[eé]stale|prestale|preste|presten|pr[eé]stamo|pr[eé]stame|d[eé]jame\s+un[oa])\b", texto_limpio):
            ranuras.accion = "PRESTAR"
        elif re.search(r"\b(qu[eé]|c[oó]mo|cu[aá]nto|cu[aá]ntos|cu[aá]ntas|d[oó]nde|ver|dime|decir|consultar|conocer|revisar|saber|hubo|han\s+habido|tuvimos|tuvieron|se\s+hicieron|se\s+registraron|se\s+encuentra|guardan|tienen|hay)\b", texto_limpio):
            ranuras.accion = "CONSULTAR"

        # 9. Construcción del Marco Semántico (SemanticFrame)
        concepto_final = ConceptoTipo.STOCK
        if "PRESTAMO" in ranuras.conceptos_operativos or ranuras.condicion == "EN_PRESTAMO":
            concepto_final = ConceptoTipo.PRESTAMO
        elif "DEVOLUCION" in ranuras.conceptos_operativos or ranuras.accion == "DEVOLVER":
            concepto_final = ConceptoTipo.DEVOLUCION
        elif "MOVIMIENTO" in ranuras.conceptos_operativos:
            concepto_final = ConceptoTipo.MOVIMIENTO
        elif "RESUMEN_GENERAL" in ranuras.conceptos_operativos:
            concepto_final = ConceptoTipo.RESUMEN_GENERAL
        elif "UBICACION" in ranuras.conceptos_operativos:
            concepto_final = ConceptoTipo.UBICACION

        modalidad_final = ModalidadTipo.CONSULTA
        if ranuras.es_afirmacion_pura:
            modalidad_final = ModalidadTipo.CONFIRMAR
        elif ranuras.es_negacion_pura:
            modalidad_final = ModalidadTipo.CANCELAR
        elif ranuras.acto_de_habla == "SALUDO":
            modalidad_final = ModalidadTipo.SALUDO
        elif ranuras.acto_de_habla == "AGRADECIMIENTO":
            modalidad_final = ModalidadTipo.AGRADECIMIENTO
        elif ranuras.accion in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER", "TRASLADAR"):
            modalidad_final = ModalidadTipo.MUTACION

        ranuras.semantic_frame = SemanticFrame(
            modalidad=modalidad_final,
            alcance=AlcanceTipo(ranuras.alcance),
            concepto=concepto_final,
            condicion=CondicionTipo(ranuras.condicion),
            almacen_especifico=ranuras.almacen_especifico,
            filtro_coleccion=ranuras.filtro_coleccion,
            destinatario=ranuras.destinatario,
        )

        return ranuras

    PALABRAS_PARADA = {
        # Vocativos y cortesía
        "haku", "oye", "hola", "saludos", "favor", "gracias", "porfa", "porfavor", "buenas", "buenos", "dias", "tardes", "noches",
        # Artículos y pronombres
        "el", "la", "los", "las", "un", "uno", "una", "unos", "unas",
        "se", "lo", "le", "me", "te", "nos", "les",
        "este", "esta", "estos", "estas", "esto",
        "ese", "esa", "esos", "esas", "eso", "eto", "eta",
        "aquel", "aquella", "aquellos", "aquellas", "aquello",
        "su", "sus", "mi", "mis", "tu", "tus",
        "dicho", "dicha", "mismo", "misma", "otro", "otra", "otros", "otras",
        "anterior", "anteriores", "previo", "previa", "previos", "previas", "pasado", "pasada", "pasados", "pasadas",
        # Preposiciones y conjunciones
        "de", "del", "al", "a", "para", "por", "pa", "en", "con", "sin", "sobre", "entre",
        "y", "e", "o", "u", "ni", "que", "si", "no", "como", "cuando", "donde", "cual", "cuales",
        # Numerales y cuantificadores
        "cero", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
        "once", "doce", "veinte", "treinta", "cincuenta", "cien", "ciento", "mil",
        "cuanto", "cuantos", "cuanta", "cuantas", "mucho", "muchos", "mucha", "muchas",
        "poco", "pocos", "poca", "pocas", "bastante", "bastantes", "algo", "nada", "todo", "todos", "toda", "todas",
        "mas", "menos", "tanto", "demasiado", "demasiados",
        "algun", "alguno", "alguna", "algunos", "algunas", "ningun", "ninguno", "ninguna", "ningunos", "ningunas",
        # Adverbios y estados
        "hoy", "ayer", "ahora", "ya", "todavia", "aun", "bien", "mal", "asi",
        "pelado", "vacio", "vacia", "vacios", "vacias",
        "agotado", "agotados", "agotada", "agotadas", "acabado", "acabados", "terminaron", "agotaron", "acabaron",
        "escaso", "escasos", "escasas", "escasez",
        # Metáforas / dialectal comodín
        "vaina", "vainas", "cosa", "cosas", "tema",
        # Reposición y necesidad
        "reposicion", "reponer", "reabastecer", "reabastecimiento",
        "necesita", "necesitan", "requiere", "requieren", "falta", "faltan",
        # Sustantivos abstractos de inventario
        "producto", "productos", "articulo", "articulos", "item", "items",
        "referencia", "referencias", "bodega", "bodegas", "almacen", "almacenes",
        "stock", "existencia", "existencias", "disponible", "disponibles", "disponibilidad",
        "saldo", "saldos", "unidad", "unidades", "pieza", "piezas", "cantidad", "cantidades",
        "inventario", "inventarios",
        "ubicacion", "ubicaciones", "localizacion", "posicion", "seccion",
        "ubicado", "ubicada", "ubicados", "ubicadas", "localizado", "localizada", "localizados", "localizadas",
        "estante", "estantes", "rack", "racks", "pasillo", "pasillos", "gaveta", "gavetas",
        "cajon", "cajones", "repisa", "repisas", "lugar", "lugares",
        "movimiento", "movimientos", "operacion", "operaciones", "transaccion", "transacciones",
        "actividad", "registro", "registros", "despacho", "despachos", "flujo", "flujos",
        "salida", "salidas", "entrada", "entradas", "prestamo", "prestamos", "devolucion", "devoluciones",
        "pedido", "pedidos", "orden", "ordenes",
        # Verbos conjugados comunes
        "hay", "hubo", "habia", "habido", "ha", "han", "hemos",
        "queda", "quedan", "quedo", "quedaron",
        "esta", "estan", "estuvo", "estado",
        "tiene", "tienen", "tuvo", "tuvimos", "tenemos", "tenido", "contamos",
        "salio", "salieron", "saliendo", "salido", "salidos", "sale", "salen",
        "entro", "entraron", "entrando", "entrado", "entrados", "entra", "entran",
        "llego", "llegaron", "llegando",
        "movio", "movieron", "moviendo", "movido", "movidos", "mueve", "mueven",
        "acabo", "acabaron", "termino",
        "paso", "pasaron", "va", "van",
        "meter", "mete", "metelo", "metele", "metame", "metieron",
        "sacar", "saca", "sacalo", "sacale", "sacame", "sacaron",
        "despachar", "despacha", "despachalo", "despachame", "despacharon",
        "bajar", "baja", "bajalo", "bajame", "bajaron",
        "mandar", "manda", "mandalo", "mandame",
        "pasar", "pasa", "pasalo", "pasame",
        "prestar", "presta", "prestalo", "prestame", "preste", "presten", "prestado", "prestao",
        "dejar", "deja", "dejalo", "dejame", "dejaron",
        "devolver", "devuelve", "devuelvelo", "devuelveme", "devuelvo",
        "regresar", "regresa", "regresalo", "regresame", "regreso",
        "traer", "trae", "traelo", "traeme", "traje", "trajo", "trajeron",
        "llevar", "lleva", "llevalo", "llevame",
        "ver", "dime", "decir", "consultar", "consulta", "consultas", "consultame",
        "buscar", "buscame", "busca", "buscan", "buscando", "saber", "sabe", "revisar", "revisa", "revisame", "mira", "mirame",
        "mencionar", "mencionaste", "mencionadas", "listar", "listadas", "dijiste", "acabas",
    }

    @classmethod
    def interpretar(
        cls,
        mensaje: str,
        usuario: UsuarioAudacIA,
        contexto_sesion: Optional[dict] = None,
        estado_operativo: Any = None,
    ) -> IntencionResultado:
        """Transforma lenguaje natural (incluyendo expresiones costeñas) en una intención segura."""
        def _responder(res: IntencionResultado) -> IntencionResultado:
            HakuCacheManager.registrar_intencion_detectada(res.tipo.value)
            return res

        if not mensaje or not mensaje.strip():
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion="No entendí tu mensaje. ¿Puedes repetirlo?",
            ))

        producto_en_contexto = None

        # 1. Recuperar producto en contexto desde EstadoOperativoHaku (RAM) o diccionario fallback
        if estado_operativo is not None and hasattr(estado_operativo, "esta_expirado"):
            if not estado_operativo.esta_expirado(cls.TIMEOUT_CONTEXTO_SEGUNDOS) and estado_operativo.producto_id:
                qs_ctx = Producto.objects.all() if (usuario and usuario.es_administrador) else Producto.objects.filter(activo=True, archivado=False)
                producto_en_contexto = qs_ctx.filter(id=estado_operativo.producto_id).first()

        if producto_en_contexto is None and contexto_sesion:
            ahora = time.time()
            ultimo_acceso = contexto_sesion.get("haku_ultimo_acceso", ahora)
            if (ahora - ultimo_acceso) <= cls.TIMEOUT_CONTEXTO_SEGUNDOS:
                prod_id = contexto_sesion.get("haku_ultimo_producto_id")
                if prod_id:
                    qs_ctx = Producto.objects.all() if (usuario and usuario.es_administrador) else Producto.objects.filter(activo=True, archivado=False)
                    producto_en_contexto = qs_ctx.filter(id=prod_id).first()

        # 1. Paso 1: Normalización fonética Whisper (resuelve errores dialectales en transcripción)
        texto_fonetico = InterpretadorCosteno.normalizar_fonetica_whisper(mensaje)

        # 2. Paso 2: Normalización de siglas técnicas y códigos hablados
        texto_siglas = NormalizadorSiglasTecnicas.normalizar(texto_fonetico)

        # 3. Paso 3: Análisis dialectal costeño y detección de giros/ambigüedades
        analisis = InterpretadorCosteno.normalizar(texto_siglas)
        texto_trabajo = analisis.texto_estandar.lower()

        # 4. Paso 4: Extracción determinista de ranuras semánticas
        ranuras = cls._extraer_ranuras_semanticas(texto_trabajo, texto_siglas, analisis)

        # Si el usuario introduce una ruptura discursiva de tema ("cambiando de tema", "otro tema", etc.),
        # se purga de raíz el foco conversacional previo para que la nueva consulta inicie limpia.
        if ranuras.es_cambio_de_tema:
            producto_en_contexto = None
            if estado_operativo:
                estado_operativo.limpiar_foco_conversacional()

        # AISLAMIENTO TOTAL DE ALCANCE (FASE 2):
        # Si la consulta es de alcance GLOBAL o ALMACEN, el foco conversacional previo NUNCA se hereda.
        if ranuras.alcance in ("GLOBAL", "ALMACEN"):
            producto_en_contexto = None

        # AISLAMIENTO POR CÓDIGO EXPLÍCITO DISTINTO AL FOCO:
        # Si el mensaje nombra explícitamente un código (p. ej. "CAB-001", "XYZ-999"), jamás
        # se debe sustituir silenciosamente por el producto previo en foco.
        _codigos_explicitos_msg = re.findall(r"\b[A-Za-z]{2,5}[-_]\d{1,5}\b", texto_siglas)
        if _codigos_explicitos_msg and producto_en_contexto is not None:
            _cod_foco_norm = (producto_en_contexto.codigo_interno or "").upper().replace("_", "-")
            if not any(c.upper().replace("_", "-") == _cod_foco_norm for c in _codigos_explicitos_msg):
                producto_en_contexto = None

        # AISLAMIENTO POR CAMBIO DE FLUJO (FASE 3):
        # Las intenciones CRUD (crear/editar usuario o producto) implican un flujo completamente
        # nuevo e independiente del producto anterior en foco. Purgar para evitar contaminación.
        _es_flujo_crud = bool(
            re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|vamos\s+a\s+|necesito\s+)?(?:crear|registrar|ingresar|dar\s+de\s+alta|editar|modificar|actualizar|inhabilitar|desactivar|archivar|dar\s+de\s+baja)\s+(?:un\s+)?(?:nuevo\s+)?(?:usuario|producto)\b", texto_trabajo)
        )
        if _es_flujo_crud:
            producto_en_contexto = None
            if estado_operativo:
                estado_operativo.producto_id = None
                estado_operativo.producto_codigo = None
                estado_operativo.producto_nombre = None
                estado_operativo.candidatos = []

        # ─────────────────────────────────────────────────────────────────────
        # REGLA 0: PROTOCOLO SOCIAL Y CONFIRMACIONES / CANCELACIONES PURAS
        # ─────────────────────────────────────────────────────────────────────
        op_pendiente = getattr(estado_operativo, "operacion_pendiente", None) if estado_operativo else None
        if op_pendiente and op_pendiente.get("esperando") == "confirmar_menor_cantidad":
            prod_id = op_pendiente.get("producto_id")
            prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
            cant = op_pendiente.get("cantidad")
            dest = op_pendiente.get("destinatario")
            mot = op_pendiente.get("motivo")
            alm_id = op_pendiente.get("almacen_id")
            alm_obj = Almacen.objects.filter(id=alm_id).first() if alm_id else None

            if ranuras.es_afirmacion_pura or ranuras.acto_de_habla == "CONFIRMAR" or re.search(r"\b(s[ií]|claro|por\s+supuesto|retira|retirar|saca|sacar|dale|de\s+acuerdo|esas|eso)\b", texto_siglas, re.IGNORECASE):
                if estado_operativo:
                    estado_operativo.operacion_pendiente = None
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.REGISTRAR_SALIDA,
                    producto=prod_obj,
                    cantidad=cant,
                    destinatario=dest,
                    motivo=mot,
                    almacen=alm_obj,
                    requiere_confirmacion=True,
                ))
            elif ranuras.es_negacion_pura or ranuras.acto_de_habla == "CANCELAR":
                if estado_operativo:
                    estado_operativo.operacion_pendiente = None
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.ACCION_CANCELAR,
                    requiere_confirmacion=False,
                ))

        if ranuras.es_afirmacion_pura:
            return _responder(IntencionResultado(tipo=IntencionTipo.ACCION_CONFIRMAR))

        if ranuras.es_negacion_pura:
            return _responder(IntencionResultado(tipo=IntencionTipo.ACCION_CANCELAR))

        # 0-bis. SEGUIMIENTO DE OPERACIÓN PENDIENTE (RECOLECCIÓN MULTITURNO DE CANTIDAD O DESTINATARIO)
        if op_pendiente:
            esperando = op_pendiente.get("esperando")
            from .numeros import extraer_modificacion_cantidad_propuesta as _ext_mod_cant_pend
            _es_mod_cant_en_pend = _ext_mod_cant_pend(texto_siglas) is not None
            if (ranuras.es_negacion_pura or ranuras.acto_de_habla == "CANCELAR") and not _es_mod_cant_en_pend and not re.search(r"\b(?:sino|en\s+vez\s+de|en\s+lugar\s+de)\b", texto_trabajo):
                if estado_operativo:
                    estado_operativo.operacion_pendiente = None
                return _responder(IntencionResultado(tipo=IntencionTipo.ACCION_CANCELAR))
            elif ranuras.es_cambio_de_tema and not _es_mod_cant_en_pend:
                if estado_operativo:
                    estado_operativo.operacion_pendiente = None
            elif esperando == "cantidad":
                # Proteger contra comandos nuevos o consultas que no son una respuesta de cantidad
                _es_comando_ajeno_cant = bool(
                    "?" in texto_siglas or "¿" in texto_siglas
                    or re.search(
                        r"\b(?:prestable|consumible|retornable|se\s+puede\s+prestar|me\s+confirm[ae]s?|"
                        r"(?:gener[aeoó]|registr[aeoó]|cre[aeoó]|hacer|dar)\s+(?:una?\s+)?(?:entrada|salida|pr[eé]stamo|traslado|devoluci[oó]n))\b",
                        texto_siglas,
                        flags=re.IGNORECASE,
                    )
                )
                if _es_comando_ajeno_cant:
                    if estado_operativo:
                        estado_operativo.operacion_pendiente = None
                    op_pendiente = None
                else:
                    texto_cant_limpio = re.sub(r"\b[A-Za-z]{2,8}(?:[-_]\d+|\s+0\d+)\b", " ", texto_siglas)
                    cant_extraida = analisis.cantidad_detectada
                    if not cant_extraida:
                        if re.search(r"\b(tod[oa]s?|completo|completa|total)\b", texto_cant_limpio, flags=re.IGNORECASE):
                            cant_extraida = op_pendiente.get("cantidad_maxima") or op_pendiente.get("cantidad") or 1
                    if not cant_extraida:
                        from .numeros import extraer_numero_espanol
                        cant_extraida = extraer_numero_espanol(texto_cant_limpio, permitir_cero=False)

                    if cant_extraida and cant_extraida > 0:
                        prod_id = op_pendiente.get("producto_id")
                        prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
                        tipo_op = op_pendiente.get("tipo", "PRESTAMO")
                        dest = op_pendiente.get("destinatario")
                        alm_id = op_pendiente.get("almacen_id")
                        alm_obj = Almacen.objects.filter(id=alm_id).first() if alm_id else None
                        mot_pend = op_pendiente.get("motivo")
                        datos_extra = {}
                        if op_pendiente.get("observacion"):
                            datos_extra["observacion"] = op_pendiente["observacion"]

                        if tipo_op == "PRESTAMO":
                            tipo_int = IntencionTipo.PRESTAMO
                        elif tipo_op == "REGISTRAR_SALIDA":
                            tipo_int = IntencionTipo.REGISTRAR_SALIDA
                        elif tipo_op == "REGISTRAR_ENTRADA":
                            tipo_int = IntencionTipo.REGISTRAR_ENTRADA
                        elif tipo_op == "TRASLADO_ENTRE":
                            tipo_int = IntencionTipo.TRASLADAR_PRODUCTO
                            if op_pendiente.get("almacen_destino_id"):
                                datos_extra["almacen_destino_id"] = op_pendiente["almacen_destino_id"]
                                datos_extra["almacen_destino_nombre"] = op_pendiente.get("almacen_destino_nombre")
                        elif tipo_op == "DEVOLUCION":
                            tipo_int = IntencionTipo.DEVOLUCION
                            if op_pendiente.get("prestamo_id"):
                                datos_extra["prestamo_id"] = op_pendiente["prestamo_id"]
                        else:
                            tipo_int = IntencionTipo.PRESTAMO

                        return _responder(IntencionResultado(
                            tipo=tipo_int,
                            producto=prod_obj,
                            cantidad=cant_extraida,
                            destinatario=dest,
                            almacen=alm_obj,
                            motivo=mot_pend,
                            datos_extra=datos_extra or None,
                            requiere_confirmacion=True,
                        ))

            elif esperando == "destinatario":
                # Permitir corrección de cantidad en medio del flujo (ej. «Hey no quiero que sean 2 unidades, que sean 5»
                # o «Mejor 4 unidades para Miguel») antes de descartar la operación pendiente.
                from .numeros import extraer_modificacion_cantidad_propuesta
                cant_corr_dest = extraer_modificacion_cantidad_propuesta(texto_siglas)
                if cant_corr_dest is not None and cant_corr_dest > 0 and op_pendiente.get("producto_id"):
                    prod_obj_cd = Producto.objects.filter(
                        id=op_pendiente.get("producto_id"), activo=True, archivado=False
                    ).first()
                    if prod_obj_cd:
                        tipo_op_cd = op_pendiente.get("tipo", "PRESTAMO")
                        alm_id_cd = op_pendiente.get("almacen_id")
                        alm_obj_cd = Almacen.objects.filter(id=alm_id_cd).first() if alm_id_cd else None
                        dest_junto = None
                        m_dest_junto = re.search(
                            r"\b(?:para|a\s+nombre\s+de|es\s+para|es\s+a)\s+([A-Za-zÁ-ÿñÑ][A-Za-zÁ-ÿñÑ\s'-]{1,40}?)(?:[\.,;!?]|$)",
                            texto_siglas,
                            flags=re.IGNORECASE,
                        )
                        if m_dest_junto:
                            cand_dj = m_dest_junto.group(1).strip(" .,;:!?")
                            if (
                                len(cand_dj) >= 2
                                and not re.search(r"\b(?:unidades?|piezas?|almac[eé]n|bodega|producto|inventario)\b", cand_dj, re.IGNORECASE)
                            ):
                                dest_junto = capitalizar_nombre_propio(cand_dj)
                        tipo_int_cd = (
                            IntencionTipo.DEVOLUCION if tipo_op_cd == "DEVOLUCION"
                            else IntencionTipo.PRESTAMO if tipo_op_cd == "PRESTAMO"
                            else IntencionTipo.REGISTRAR_SALIDA
                        )
                        datos_extra_cd = (
                            {"observacion": op_pendiente["observacion"]}
                            if op_pendiente.get("observacion")
                            else None
                        )
                        return _responder(IntencionResultado(
                            tipo=tipo_int_cd,
                            producto=prod_obj_cd,
                            cantidad=cant_corr_dest,
                            destinatario=dest_junto,
                            almacen=alm_obj_cd,
                            motivo=op_pendiente.get("motivo"),
                            datos_extra=datos_extra_cd,
                            requiere_confirmacion=True,
                        ))

                # Nunca aceptar una orden operativa, código de producto o consulta como nombre de persona
                _es_comando_o_consulta_dest = bool(
                    "?" in texto_siglas or "¿" in texto_siglas
                    or re.search(r"\b[A-Za-z]{2,5}[-_]\d{1,5}\b", texto_siglas)
                    or re.search(
                        r"\b(?:entradas?|salidas?|traslados?|devoluci[oó]n(?:es)?|unidades?|stock|existencias?|"
                        r"almac[eé]n|bodega|categor[ií]a|prestable|consumible|retornable|"
                        r"registr[aeoó]\w*|gener[aeoó]\w*|ingres[aeoó]\w*|sacar|retirar|crear|consultar?|buscar|mu[eé]str\w*)\b",
                        texto_siglas,
                        flags=re.IGNORECASE,
                    )
                )
                if _es_comando_o_consulta_dest:
                    if estado_operativo:
                        estado_operativo.operacion_pendiente = None
                    op_pendiente = None
                else:
                    # Extraer nombre del destinatario del texto (o resolver ordinal sobre candidatos activos)
                    texto_nom = texto_siglas.strip(" \t\r\n.,;:!?¿¡")
                    cands_dest = op_pendiente.get("candidatos_destinatario") or []
                    if cands_dest:
                        t_ord = re.sub(r"[^\w\sáéíóú]", " ", texto_nom.lower()).strip()
                        mapa_ord = {
                            "el primero": 0, "la primera": 0, "primero": 0, "primera": 0, "el 1": 0, "numero 1": 0, "número 1": 0, "1": 0, "uno": 0,
                            "el segundo": 1, "la segunda": 1, "segundo": 1, "segunda": 1, "el 2": 1, "numero 2": 1, "número 2": 1, "2": 1, "dos": 1,
                            "el tercero": 2, "la tercera": 2, "tercero": 2, "tercera": 2, "el 3": 2, "numero 3": 2, "número 3": 2, "3": 2, "tres": 2,
                            "el cuarto": 3, "la cuarta": 3, "cuarto": 3, "cuarta": 3, "el 4": 3, "4": 3, "cuatro": 3,
                            "el quinto": 4, "la quinta": 4, "quinto": 4, "quinta": 4, "el 5": 4, "5": 4, "cinco": 4,
                        }
                        if t_ord in mapa_ord and mapa_ord[t_ord] < len(cands_dest):
                            texto_nom = cands_dest[mapa_ord[t_ord]]
                        else:
                            from ..servicios_personas import emparejar_nombre_contra_candidatos
                            match_cand = emparejar_nombre_contra_candidatos(texto_nom, cands_dest)
                            if match_cand:
                                texto_nom = match_cand
                    # Limpiar prefijos conversacionales como "se lo voy a prestar a", "se lo entrego a", "a nombre de", "para", etc.
                    texto_nom = PREFIJOS_CONVERSACIONALES_DESTINATARIO.sub("", texto_nom).strip().rstrip(".,;:!?")
                    # Si empieza por 'ay' pegado al nombre (ej. 'ayancarlos' -> 'yancarlos')
                    if re.match(r"^ay[a-z]{3,}", texto_nom, flags=re.IGNORECASE):
                        texto_nom = texto_nom[1:].strip()
                    elif re.match(r"^a\s+[a-z]{2,}", texto_nom, flags=re.IGNORECASE):
                        texto_nom = texto_nom[2:].strip()
                    if re.fullmatch(r"confirm[ao]\w*|cancel\w*", texto_nom, flags=re.IGNORECASE):
                        texto_nom = ""
                    else:
                        texto_nom = capitalizar_nombre_propio(texto_nom)

                    prod_id = op_pendiente.get("producto_id")
                    cant = op_pendiente.get("cantidad")
                    prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
                    tipo_op = op_pendiente.get("tipo", "PRESTAMO")
                    alm_id = op_pendiente.get("almacen_id")
                    alm_obj = Almacen.objects.filter(id=alm_id).first() if alm_id else None

                    if tipo_op == "DEVOLUCION":
                        tipo_int = IntencionTipo.DEVOLUCION
                    elif tipo_op == "PRESTAMO":
                        tipo_int = IntencionTipo.PRESTAMO
                    else:
                        tipo_int = IntencionTipo.REGISTRAR_SALIDA

                    if prod_obj and len(texto_nom) >= 2 and len(texto_nom.split()) <= 6:
                        datos_extra_dest = {}
                        if op_pendiente.get("observacion"):
                            datos_extra_dest["observacion"] = op_pendiente["observacion"]
                        return _responder(IntencionResultado(
                            tipo=tipo_int,
                            producto=prod_obj,
                            cantidad=cant,
                            destinatario=texto_nom,
                            almacen=alm_obj,
                            motivo=op_pendiente.get("motivo"),
                            datos_extra=datos_extra_dest or None,
                            requiere_confirmacion=True,
                        ))

            elif esperando == "producto":
                tipo_op_pend = op_pendiente.get("tipo", "PRESTAMO")
                incluir_inactivos_pend = (tipo_op_pend == "HABILITAR_PRODUCTO")
                prod_res, cod_err, cands_prod = cls._resolver_producto_en_texto(
                    texto_siglas,
                    usuario,
                    candidatos_previos=getattr(estado_operativo, "candidatos", None),
                    incluir_inactivos=incluir_inactivos_pend,
                )
                if not prod_res and cands_prod and len(cands_prod) == 1:
                    qs_p_res = Producto.objects.filter(id=cands_prod[0]["id"])
                    if not incluir_inactivos_pend:
                        qs_p_res = qs_p_res.filter(activo=True, archivado=False)
                    prod_res = qs_p_res.first()
                if not prod_res and cands_prod and len(cands_prod) > 1:
                    cands_str = ", ".join(f"{c['nombre']} ({c['codigo']})" for c in cands_prod[:3])
                    return _responder(IntencionResultado(
                        tipo=IntencionTipo.SOLICITAR_ACLARACION,
                        candidatos=cands_prod,
                        mensaje_aclaracion=f"Encontré varios productos posibles ({cands_str}). ¿A cuál de ellos te refieres?",
                    ))
                if prod_res:
                    tipo_op = tipo_op_pend
                    if tipo_op == "INHABILITAR_PRODUCTO":
                        return _responder(IntencionResultado(
                            tipo=IntencionTipo.INHABILITAR_PRODUCTO,
                            producto=prod_res,
                            requiere_confirmacion=True,
                        ))
                    if tipo_op == "HABILITAR_PRODUCTO":
                        return _responder(IntencionResultado(
                            tipo=IntencionTipo.HABILITAR_PRODUCTO,
                            producto=prod_res,
                            requiere_confirmacion=True,
                        ))
                    if tipo_op == "EDITAR_PRODUCTO":
                        return _responder(IntencionResultado(
                            tipo=IntencionTipo.EDITAR_PRODUCTO,
                            producto=prod_res,
                            requiere_confirmacion=False,
                        ))
                    dest = op_pendiente.get("destinatario")
                    if not dest:
                        m_d = re.search(r"\b(?:para|a\s+nombre\s+de|entregar\s+a|a)\s+([A-Za-zÁ-ÿñÑ\s'-]+?)(?:\s+(?:por|motivo|en|del?)\b|\s*$)", texto_siglas, re.IGNORECASE)
                        if m_d:
                            cand_d = m_d.group(1).strip()
                            cand_d = re.sub(r"\b(?:en|desde|hacia)\s+(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\b.*$", "", cand_d, flags=re.IGNORECASE).strip()
                            if prod_res.nombre.lower() in cand_d.lower():
                                cand_d = re.sub(re.escape(prod_res.nombre), "", cand_d, flags=re.IGNORECASE).strip()
                            if prod_res.codigo_interno.lower() in cand_d.lower():
                                cand_d = re.sub(re.escape(prod_res.codigo_interno), "", cand_d, flags=re.IGNORECASE).strip()
                            if len(cand_d) >= 2 and not re.search(r"\b(almac[eé]n|bodega|unidades?|piezas?)\b", cand_d, re.IGNORECASE):
                                dest = capitalizar_nombre_propio(cand_d)
                    cant = op_pendiente.get("cantidad") or analisis.cantidad_detectada
                    alm_id = op_pendiente.get("almacen_id")
                    alm_obj = Almacen.objects.filter(id=alm_id).first() if alm_id else ranuras.almacen_especifico
                    mot = op_pendiente.get("motivo") or ranuras.motivo
                    obs_pend_prod = op_pendiente.get("observacion")

                    if tipo_op == "DEVOLUCION":
                        datos_extra_dev_p = {"observacion": obs_pend_prod} if obs_pend_prod else None
                        return _responder(IntencionResultado(
                            tipo=IntencionTipo.DEVOLUCION,
                            producto=prod_res,
                            destinatario=dest,
                            cantidad=analisis.cantidad_detectada or op_pendiente.get("cantidad"),
                            almacen=alm_obj,
                            datos_extra=datos_extra_dev_p,
                            requiere_confirmacion=True,
                        ))
                    elif tipo_op == "TRASLADO_ENTRE":
                        datos_tr = {}
                        if op_pendiente.get("almacen_destino_id"):
                            datos_tr["almacen_destino_id"] = op_pendiente["almacen_destino_id"]
                            datos_tr["almacen_destino_nombre"] = op_pendiente.get("almacen_destino_nombre")
                        if obs_pend_prod:
                            datos_tr["observacion"] = obs_pend_prod
                        return _responder(IntencionResultado(
                            tipo=IntencionTipo.TRASLADAR_PRODUCTO,
                            producto=prod_res,
                            almacen=alm_obj,
                            cantidad=op_pendiente.get("cantidad") or analisis.cantidad_detectada,
                            datos_extra=datos_tr or None,
                            requiere_confirmacion=True,
                        ))
                    elif tipo_op == "REGISTRAR_SALIDA":
                        datos_sal_p = {"observacion": obs_pend_prod} if obs_pend_prod else None
                        return _responder(IntencionResultado(
                            tipo=IntencionTipo.REGISTRAR_SALIDA,
                            producto=prod_res,
                            destinatario=dest,
                            cantidad=op_pendiente.get("cantidad") or analisis.cantidad_detectada,
                            almacen=alm_obj,
                            motivo=mot,
                            datos_extra=datos_sal_p,
                            requiere_confirmacion=True,
                        ))
                    elif tipo_op == "REGISTRAR_ENTRADA":
                        datos_ent_p = {"observacion": obs_pend_prod} if obs_pend_prod else None
                        return _responder(IntencionResultado(
                            tipo=IntencionTipo.REGISTRAR_ENTRADA,
                            producto=prod_res,
                            cantidad=op_pendiente.get("cantidad") or analisis.cantidad_detectada,
                            almacen=alm_obj,
                            motivo=mot,
                            datos_extra=datos_ent_p,
                            requiere_confirmacion=True,
                        ))
                    else:
                        datos_pr_p = {"observacion": obs_pend_prod} if obs_pend_prod else None
                        return _responder(IntencionResultado(
                            tipo=IntencionTipo.PRESTAMO,
                            producto=prod_res,
                            destinatario=dest,
                            cantidad=op_pendiente.get("cantidad") or analisis.cantidad_detectada,
                            almacen=alm_obj,
                            datos_extra=datos_pr_p,
                            requiere_confirmacion=True,
                        ))

            elif esperando == "destino":
                from .numeros import extraer_cantidad_operacion_texto, extraer_modificacion_cantidad_propuesta
                cant_corr_tr = extraer_modificacion_cantidad_propuesta(texto_siglas)
                if cant_corr_tr is None:
                    cant_corr_tr = extraer_cantidad_operacion_texto(texto_siglas)
                alm_dest = cls._resolver_almacen_en_texto(texto_siglas)
                prod_id = op_pendiente.get("producto_id")
                prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
                alm_orig_id = op_pendiente.get("almacen_id")
                alm_orig = Almacen.objects.filter(id=alm_orig_id).first() if alm_orig_id else None
                cant = cant_corr_tr if (cant_corr_tr is not None and cant_corr_tr > 0) else (op_pendiente.get("cantidad") or analisis.cantidad_detectada)
                obs_dest_tr = op_pendiente.get("observacion")
                if alm_dest:
                    datos_dest_tr = {
                        "almacen_destino_id": alm_dest.id,
                        "almacen_destino_nombre": alm_dest.nombre,
                    }
                    if obs_dest_tr:
                        datos_dest_tr["observacion"] = obs_dest_tr
                    return _responder(IntencionResultado(
                        tipo=IntencionTipo.TRASLADAR_PRODUCTO,
                        producto=prod_obj,
                        almacen=alm_orig,
                        cantidad=cant,
                        datos_extra=datos_dest_tr,
                        requiere_confirmacion=True,
                    ))
                elif cant_corr_tr is not None and cant_corr_tr > 0 and prod_obj:
                    return _responder(IntencionResultado(
                        tipo=IntencionTipo.TRASLADAR_PRODUCTO,
                        producto=prod_obj,
                        almacen=alm_orig,
                        cantidad=cant_corr_tr,
                        datos_extra={"observacion": obs_dest_tr} if obs_dest_tr else None,
                        requiere_confirmacion=True,
                    ))

            elif esperando == "identificacion":
                prod_res, _, _ = cls._resolver_producto_en_texto(texto_siglas, usuario, candidatos_previos=getattr(estado_operativo, "candidatos", None))
                match_dest = re.search(r"\b(?:de|por|a|para)\s+([A-Za-zÁ-ÿ\s]+?)(?:\s*$)", texto_siglas, re.IGNORECASE)
                dest = match_dest.group(1).strip() if match_dest else None
                if dest:
                    dest = re.sub(r"\b(pr[eé]stamo|pr[eé]stamos|de|el|la)\b", "", dest, flags=re.IGNORECASE).strip()
                    if prod_res and prod_res.nombre.lower() in dest.lower():
                        dest = re.sub(re.escape(prod_res.nombre), "", dest, flags=re.IGNORECASE).strip()
                    dest = capitalizar_nombre_propio(dest) if len(dest) >= 2 else None

                if not dest and not prod_res:
                    texto_limpio = texto_siglas.strip()
                    if re.match(r"^[A-Za-zÁ-ÿ\s]{2,30}$", texto_limpio) and not re.search(r"\b(devolver|prestamo|cancelar)\b", texto_limpio, flags=re.IGNORECASE):
                        dest = capitalizar_nombre_propio(texto_limpio)

                if prod_res or dest:
                    datos_extra_id = {"observacion": op_pendiente["observacion"]} if op_pendiente.get("observacion") else None
                    return _responder(IntencionResultado(
                        tipo=IntencionTipo.DEVOLUCION,
                        producto=prod_res,
                        destinatario=dest,
                        cantidad=analisis.cantidad_detectada or op_pendiente.get("cantidad"),
                        datos_extra=datos_extra_id,
                        requiere_confirmacion=True,
                    ))

            elif esperando == "referencia_prestamo":
                cant_extraida = analisis.cantidad_detectada or cls._extraer_cardinal_o_numero(texto_siglas) or op_pendiente.get("cantidad")
                match_ref_pend = re.search(r"\b(?:pr[eé]stamo\s+)?(?:#|id\s+|ref\s+)?([0-9a-fA-F-]{6,36})\b", texto_siglas, re.IGNORECASE)
                ref_pend = match_ref_pend.group(1).lower() if match_ref_pend else None

                prod_id = op_pendiente.get("producto_id")
                prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
                dest = op_pendiente.get("destinatario")
                datos_extra = {}
                if ref_pend:
                    datos_extra["prestamo_id"] = ref_pend
                if op_pendiente.get("observacion"):
                    datos_extra["observacion"] = op_pendiente["observacion"]

                return _responder(IntencionResultado(
                    tipo=IntencionTipo.DEVOLUCION,
                    producto=prod_obj,
                    destinatario=dest,
                    cantidad=cant_extraida,
                    datos_extra=datos_extra or None,
                    requiere_confirmacion=True,
                ))

            elif esperando == "confirmar_menor_cantidad":
                prod_id = op_pendiente.get("producto_id")
                prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
                cant = op_pendiente.get("cantidad")
                dest = op_pendiente.get("destinatario")
                mot = op_pendiente.get("motivo")
                alm_id = op_pendiente.get("almacen_id")
                alm_obj = Almacen.objects.filter(id=alm_id).first() if alm_id else None
                datos_extra_mc = {"observacion": op_pendiente["observacion"]} if op_pendiente.get("observacion") else None

                from .numeros import extraer_modificacion_cantidad_propuesta, extraer_numero_espanol
                cant_alt = extraer_modificacion_cantidad_propuesta(texto_siglas)
                if cant_alt is None:
                    cant_alt = extraer_numero_espanol(texto_siglas, permitir_cero=False, solo_numero_exacto=True)
                if cant_alt is not None and cant_alt > 0 and prod_obj:
                    return _responder(IntencionResultado(
                        tipo=IntencionTipo.REGISTRAR_SALIDA,
                        producto=prod_obj,
                        cantidad=cant_alt,
                        destinatario=dest,
                        motivo=mot,
                        almacen=alm_obj,
                        datos_extra=datos_extra_mc,
                        requiere_confirmacion=True,
                    ))
                if ranuras.es_afirmacion_pura or ranuras.acto_de_habla == "CONFIRMAR" or re.search(r"\b(retira|retirar|saca|sacar|dale|de\s+acuerdo|esas|eso)\b", texto_siglas, re.IGNORECASE):
                    return _responder(IntencionResultado(
                        tipo=IntencionTipo.REGISTRAR_SALIDA,
                        producto=prod_obj,
                        cantidad=cant,
                        destinatario=dest,
                        motivo=mot,
                        almacen=alm_obj,
                        datos_extra=datos_extra_mc,
                        requiere_confirmacion=True,
                    ))
                elif ranuras.es_negacion_pura or ranuras.acto_de_habla == "CANCELAR":
                    return _responder(IntencionResultado(
                        tipo=IntencionTipo.ACCION_CANCELAR,
                        requiere_confirmacion=False,
                    ))

        # 0-ter. CORRECCIÓN O MODIFICACIÓN DE PROPUESTA ACTIVA (DESTINATARIO, ALMACÉN O TIPO DE OPERACIÓN)
        prop_activa = getattr(estado_operativo, "propuesta_activa", None) if estado_operativo else None
        if contexto_sesion is not None and isinstance(contexto_sesion, dict) and not contexto_sesion.get("haku_propuesta_activa"):
            prop_activa = None
        if prop_activa and not ranuras.es_cambio_de_tema:
            tipo_op_prop_actual = str(prop_activa.get("tipo_operacion") or "").upper()
            obs_prop_actual = (prop_activa.get("observacion") or "").strip()
            # 1. Corrección de destinatario solo para PRESTAMO o SALIDA (nunca secuestrar ENTRADA/DEVOLUCION/TRASLADO/CREAR_PRODUCTO)
            match_correccion_dest = re.search(
                r"\b(?:no\s+(?:esp[eé]rate\s+|espera\s+)?no\s+era\s+[A-Za-zÁ-ÿ\s]+[;,]?\s*(?:era|es)|no\s+era\s+[A-Za-zÁ-ÿ\s]+[;,]?\s*(?:era|es)|corrige\s+(?:ese\s+|el\s+)?(?:nombre|destinatario)(?:\s+(?:a|por|es))?|c[aá]mbiale\s+(?:el\s+)?nombre\s+a|an[oó]taselo\s+a|an[oó]tale\s+a)\s+([A-Za-zÁ-ÿ\s]+)",
                texto_siglas,
                re.IGNORECASE,
            )
            if match_correccion_dest and tipo_op_prop_actual in (Movimiento.Tipo.PRESTAMO, Movimiento.Tipo.SALIDA):
                nuevo_dest = match_correccion_dest.group(1).strip()
                nuevo_dest = re.sub(r"\b(?:en|desde|hacia)\s+(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\b.*$", "", nuevo_dest, flags=re.IGNORECASE).strip()
                nuevo_dest = PREFIJOS_CONVERSACIONALES_DESTINATARIO.sub("", nuevo_dest).strip()
                nuevo_dest = capitalizar_nombre_propio(nuevo_dest)
                prod_id = prop_activa.get("producto_id")
                prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
                if prod_obj and len(nuevo_dest) >= 2:
                    tipo_int = IntencionTipo.PRESTAMO if tipo_op_prop_actual == Movimiento.Tipo.PRESTAMO else IntencionTipo.REGISTRAR_SALIDA
                    return _responder(IntencionResultado(
                        tipo=tipo_int,
                        producto=prod_obj,
                        cantidad=prop_activa.get("cantidad", 1),
                        destinatario=nuevo_dest,
                        almacen=Almacen.objects.filter(id=prop_activa.get("almacen_origen_id")).first(),
                        motivo=prop_activa.get("motivo"),
                        datos_extra={"observacion": obs_prop_actual} if obs_prop_actual else None,
                        requiere_confirmacion=True,
                    ))

            # 2. Conmutación a dotación definitiva solo sobre propuesta de PRESTAMO
            es_cambio_a_dotacion = bool(
                re.search(r"\b(?:no\s+(?:lo\s+)?voy\s+a\s+prestar|no\s+es\s+pr[eé]stamo|entrega\s+definitiva\s+de\s+eso|dotaci[oó]n\s+de\s+eso)\b", texto_trabajo)
                or (
                    re.search(r"\b(?:entregar?\s+definitivamente|entrega\s+definitiva|dotaci[oó]n|definitivamente)\b", texto_trabajo)
                    and re.search(r"\b(?:eso|ese|este|aquel|lo|la)\b", texto_trabajo)
                )
            )
            if es_cambio_a_dotacion and tipo_op_prop_actual == Movimiento.Tipo.PRESTAMO:
                prod_id = prop_activa.get("producto_id")
                prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
                if prod_obj:
                    return _responder(IntencionResultado(
                        tipo=IntencionTipo.REGISTRAR_SALIDA,
                        producto=prod_obj,
                        cantidad=prop_activa.get("cantidad", 1),
                        destinatario=prop_activa.get("destinatario"),
                        almacen=Almacen.objects.filter(id=prop_activa.get("almacen_origen_id")).first(),
                        motivo="Dotación / entrega definitiva",
                        datos_extra={"observacion": obs_prop_actual} if obs_prop_actual else None,
                        requiere_confirmacion=True,
                    ))

            # 3. Conmutación de almacén en propuesta activa (solo ENTRADA, SALIDA o PRESTAMO)
            es_cambio_almacen_prop = bool(
                re.search(r"\b(?:ese\s+mismo|el\s+mismo|de\s+ese)\s*,\s*(?:pero\s+)?(?:del?\s+|en\s+)(?:el\s+)?(?:otro\s+almac[eé]n|otra\s+bodega)\b", texto_trabajo)
                or re.search(r"\b(?:del?\s+|en\s+)(?:el\s+)?(?:otro\s+almac[eé]n|otra\s+bodega)\b", texto_trabajo)
            ) and not re.search(r"\b(stock|cu[aá]nto|quedan?|queda)\b", texto_trabajo)
            if es_cambio_almacen_prop and tipo_op_prop_actual in (Movimiento.Tipo.ENTRADA, Movimiento.Tipo.SALIDA, Movimiento.Tipo.PRESTAMO):
                prod_id = prop_activa.get("producto_id")
                prod_obj = Producto.objects.filter(id=prod_id, activo=True, archivado=False).first()
                if prod_obj:
                    tipo_int = IntencionTipo.PRESTAMO if tipo_op_prop_actual == Movimiento.Tipo.PRESTAMO else (
                        IntencionTipo.REGISTRAR_SALIDA if tipo_op_prop_actual == Movimiento.Tipo.SALIDA else IntencionTipo.REGISTRAR_ENTRADA
                    )
                    datos_otro = {"otro_almacen": True}
                    if obs_prop_actual:
                        datos_otro["observacion"] = obs_prop_actual
                    return _responder(IntencionResultado(
                        tipo=tipo_int,
                        producto=prod_obj,
                        cantidad=prop_activa.get("cantidad", 1),
                        destinatario=prop_activa.get("destinatario"),
                        motivo=prop_activa.get("motivo"),
                        datos_extra=datos_otro,
                        requiere_confirmacion=True,
                    ))

        tiene_operacion = bool(
            ranuras.conceptos_operativos.intersection({"STOCK", "UBICACION", "MOVIMIENTO", "RESUMEN_GENERAL", "PRESTAMO", "DEVOLUCION", "ALERTA"})
            or (ranuras.accion in {"REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER", "TRASLADAR"})
        )

        if ranuras.acto_de_habla == "SALUDO" and not tiene_operacion:
            return _responder(IntencionResultado(tipo=IntencionTipo.SALUDO))

        if ranuras.acto_de_habla == "AGRADECIMIENTO" and not tiene_operacion:
            return _responder(IntencionResultado(tipo=IntencionTipo.AGRADECIMIENTO))

        # Creación y edición de usuarios por voz / texto
        if re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|vamos\s+a\s+|necesito\s+)?(?:crear|registrar|ingresar|dar\s+de\s+alta)\s+(?:un\s+)?(?:nuevo\s+)?usuario\b", texto_trabajo):
            return _responder(IntencionResultado(
                tipo=IntencionTipo.CREAR_USUARIO,
                almacen=ranuras.almacen_especifico,
            ))

        if re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|vamos\s+a\s+|necesito\s+)?(?:editar|modificar|actualizar|cambiar\s+(?:los\s+)?datos\s+de|cambiar\s+(?:los\s+)?permisos\s+de)\s+(?:este\s+|un\s+)?usuario\b", texto_trabajo):
            return _responder(IntencionResultado(
                tipo=IntencionTipo.EDITAR_USUARIO,
                almacen=ranuras.almacen_especifico,
            ))

        # Creación de producto nuevo por voz
        if re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|vamos\s+a\s+)?(?:crear|registrar|ingresar|dar\s+de\s+alta)\s+(?:un\s+)?(?:nuevo\s+)?producto\b", texto_trabajo):
            return _responder(IntencionResultado(
                tipo=IntencionTipo.CREAR_PRODUCTO,
                almacen=ranuras.almacen_especifico,
            ))

        # Intención de habilitar o reactivar producto sin producto en texto ni contexto
        es_frase_habilitar = bool(re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|vamos\s+a\s+)?(?:ir\s+a\s+)?(?:habilitar|activar|reactivar|dar\s+de\s+alta)\b", texto_trabajo))
        es_frase_inhabilitar = bool(re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|vamos\s+a\s+)?(?:ir\s+a\s+)?(?:inhabilitar|desactivar|archivar|dar\s+de\s+baja)\b", texto_trabajo))

        tiene_pronombre_o_demostrativo = bool(
            analisis.demostrativo_detectado
            or re.search(r"\b(lo|la|los|las|ese|esa|este|esta|aquel|aquella|dicho|lo\s+quiero|quiero\s+inhabilitarlo|quiero\s+habilitarlo|inhabilitarlo|habilitarlo|desactivarlo|activarlo)\b", texto_trabajo)
            or re.search(r"\b(?:inhabilit[aá]r?lo|desact[ií]v[aá]r?lo|arch[ií]v[aá]r?lo|inhabil[ií]talo|desact[ií]valo|arch[ií]valo|habilit[aá]r?lo|activ[aá]r?lo|reactiv[aá]r?lo|hab[ií]l[ií]talo|act[ií]valo|react[ií]valo)\b", texto_trabajo)
        )

        if es_frase_habilitar and not tiene_pronombre_o_demostrativo and not producto_en_contexto and not re.search(r"\b[A-Za-z]{2,5}[-_]\d{1,4}\b", texto_siglas):
            return _responder(IntencionResultado(
                tipo=IntencionTipo.HABILITAR_PRODUCTO,
                almacen=ranuras.almacen_especifico,
            ))

        if es_frase_inhabilitar and not tiene_pronombre_o_demostrativo and not producto_en_contexto and not re.search(r"\b[A-Za-z]{2,5}[-_]\d{1,4}\b", texto_siglas):
            return _responder(IntencionResultado(
                tipo=IntencionTipo.INHABILITAR_PRODUCTO,
                almacen=ranuras.almacen_especifico,
            ))

        # 5. Paso 5: Búsqueda de producto nombrado explícitamente en el texto actual
        candidatos_previos = getattr(estado_operativo, "candidatos", None) if estado_operativo else None

        prod_mencionado = None
        aclaracion_codigo = None
        lista_candidatos = []

        # Buscar producto individual si el alcance NO es GLOBAL, ALMACEN ni COLECCION con filtro,
        # o si la acción explícita es una mutación/préstamo (ej. "Préstale 2 cinta aislante a Carlos en el Almacén Principal")
        es_accion_mutacion = ranuras.accion in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER", "TRASLADAR") or es_frase_inhabilitar or es_frase_habilitar
        if es_accion_mutacion or (ranuras.alcance not in ("GLOBAL", "ALMACEN") and not (ranuras.alcance == "COLECCION" and ranuras.filtro_coleccion)):
            prod_mencionado, aclaracion_codigo, lista_candidatos = cls._resolver_producto_en_texto(
                texto_siglas,
                usuario,
                candidatos_previos=candidatos_previos,
                incluir_inactivos=es_frase_habilitar,
            )

        if prod_mencionado is not None:
            ranuras.alcance = "INDIVIDUAL"
            ranuras.filtro_coleccion = None
            aclaracion_codigo = None
            # FASE 3: Un producto explícito nuevo sustituye al anterior y limpia candidatos previos.
            if estado_operativo and estado_operativo.candidatos:
                estado_operativo.candidatos = []
        elif aclaracion_codigo and ranuras.alcance == "COLECCION" and ranuras.filtro_coleccion and not ranuras.accion:
            aclaracion_codigo = None

        if aclaracion_codigo:
            es_demostrativo_puro = bool(
                analisis.demostrativo_detectado
                or re.search(r"\b(ese|esa|este|esta|aquel|aquella|eso|esos|esas|estos|estas|aquellos|aquellas|su|sus|de\s+es[eoa]s?|de\s+aquell[oas]+|dicho\s+producto|el\s+producto\s+anterior|el\s+mismo|la\s+misma|esa\s+vaina|esta\s+vaina|la\s+vaina|vaina)\b", texto_trabajo)
            )
            contexto_en_candidatos = bool(
                producto_en_contexto and any(str(c.get("id")) == str(producto_en_contexto.id) for c in lista_candidatos)
            )
            es_accion_contextual = bool(
                contexto_en_candidatos
                and (
                    ranuras.accion in ("DEVOLVER", "PRESTAR", "REGISTRAR_ENTRADA", "REGISTRAR_SALIDA")
                    and re.search(r"\b(prestao|prestado|devoluci[oó]n|regreso|devuelvo|regr[eé]same|devu[eé]lveme)\b", texto_trabajo)
                )
            )
            prestamo_candidato = None
            if not contexto_en_candidatos and ranuras.accion == "DEVOLVER" and ranuras.destinatario:
                qs_p = Prestamo.objects.filter(
                    destinatario__icontains=ranuras.destinatario,
                    estado__in=[Prestamo.Estado.ABIERTO, Prestamo.Estado.DEVUELTO_PARCIAL],
                )
                prestamos_coincidentes = [p for p in qs_p if any(str(c.get("id")) == str(p.producto_id) for c in lista_candidatos)]
                if len(prestamos_coincidentes) == 1:
                    prestamo_candidato = prestamos_coincidentes[0].producto

            if producto_en_contexto and (es_demostrativo_puro or es_accion_contextual):
                prod_mencionado = producto_en_contexto
                aclaracion_codigo = None
                lista_candidatos = []
            elif prestamo_candidato:
                prod_mencionado = prestamo_candidato
                aclaracion_codigo = None
                lista_candidatos = []
            elif analisis.demostrativo_detectado and not producto_en_contexto:
                dem = sanitizar_demostrativo_salida(analisis.demostrativo_detectado)
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.SOLICITAR_ACLARACION,
                    mensaje_aclaracion=(
                        f"Te refieres a '{dem}', pero no tengo un producto asociado actualmente (no indicaste cuál producto). "
                        "Por favor especifica el nombre o código interno del producto."
                    ),
                ))
            else:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.SOLICITAR_ACLARACION,
                    mensaje_aclaracion=aclaracion_codigo,
                    candidatos=lista_candidatos,
                ))

        # 5-bis: Selección plural de candidatos ante aclaración previa ("de los tres", "de todos ellos")
        if lista_candidatos and not aclaracion_codigo and candidatos_previos:
            if "UBICACION" in ranuras.conceptos_operativos or ranuras.contexto_fisico:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_UBICACION_COLECCION,
                    candidatos=lista_candidatos,
                    almacen=ranuras.almacen_especifico,
                    datos_extra={"candidatos_plurales": lista_candidatos},
                ))
            return _responder(IntencionResultado(
                tipo=IntencionTipo.CONSULTAR_UBICACION_COLECCION if (estado_operativo and estado_operativo.ultima_intencion == IntencionTipo.CONSULTAR_UBICACION.value) else IntencionTipo.PRODUCTOS_CON_STOCK,
                candidatos=lista_candidatos,
                almacen=ranuras.almacen_especifico,
                datos_extra={"candidatos_plurales": lista_candidatos},
            ))

        # ─────────────────────────────────────────────────────────────────────
        # CAPA DE INTENCIONES GLOBALES Y DE COLECCIÓN (INMUNE AL PRODUCTO ANTERIOR)
        # ─────────────────────────────────────────────────────────────────────
        # Se activa cuando el alcance es explícitamente GLOBAL, ALMACEN,
        # o cuando es una consulta sobre una COLECCION específica con filtro.
        # NUNCA se activa si el alcance es INDIVIDUAL para no secuestrar consultas
        # dependientes, demostrativos ni solicitudes de aclaración individual.
        if ranuras.alcance in ("GLOBAL", "ALMACEN", "COLECCION"):
            # A. Préstamos activos (Global - sin referencia a producto individual específico)
            es_prestamos_global_cond = bool(
                ranuras.condicion == "EN_PRESTAMO"
                or ranuras.metrica_prestamo != "GENERAL"
                or re.search(r"\b(hay\s+alg[uú]n\s+producto\s+en\s+pr[eé]stamos?|hay\s+algo\s+prestado|qu[eé]\s+tenemos\s+prestado|qu[eé]\s+est[aá]\s+prestado|qu[eé]\s+(?:productos|art[ií]culos|referencias)\s+est[aá]n\s+prestados?|mu[eé]strame\s+los\s+(?:productos|art[ií]culos)\s+que\s+est[aá]n\s+prestados?|cu[aá]les\s+son\s+los\s+pr[eé]stamos\s+activos?|pr[eé]stamos\s+activos|pr[eé]stamos\s+pendientes|qui[eé]n\s+tiene\s+(?:pr[eé]stamos|cosas\s+prestadas)|qu[eé]\s+pr[eé]stamos\s+hay)\b", texto_trabajo)
                or (re.search(r"\bqui[eé]n\s+tiene\b", texto_trabajo) and not re.search(r"\b(eso|ese|este|aquel|su|el|la|dicho)\b", texto_trabajo))
                or (re.search(r"\b(pr[eé]stamo|pr[eé]stamos|prestado|prestados)\b", texto_trabajo) and re.search(r"\b(hay\s+alg[uú]n|qu[eé]\s+tenemos|qu[eé]\s+hay|cu[aá]les\s+son|todos\s+los|qu[eé]\s+productos)\b", texto_trabajo))
            )
            if es_prestamos_global_cond and ranuras.accion not in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "DEVOLVER", "PRESTAR"):
                datos = {}
                if ranuras.filtro_coleccion:
                    datos["filtro_nombre"] = ranuras.filtro_coleccion
                if hasattr(ranuras, "metrica_prestamo") and ranuras.metrica_prestamo:
                    datos["metrica"] = ranuras.metrica_prestamo
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.PRESTAMOS_ACTIVOS,
                    almacen=ranuras.almacen_especifico,
                    datos_extra=datos or None,
                ))

            # B. Productos sin stock / agotados (Global determinista - PRODUCTOS_SIN_STOCK)
            es_consulta_sin_stock = bool(
                ranuras.condicion == "AGOTADO"
                or "STOCK_AGOTADO" in ranuras.conceptos_operativos
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:productos|art[ií]culos|referencias|cosas)\s+(?:no\s+tienen\s+stock|est[aá]n\s+agotad[oa]s?|se\s+agotaron|tienen\s+stock\s+cero|est[aá]n\s+en\s+cero|est[aá]n\s+sin\s+existencias?|sin\s+existencias?|se\s+acabar?on)\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:cosas?|productos?|art[ií]culos?|referencias?)\s+(?:est[aá]n\s+)?agotad[ao]s?\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:cosas?|productos?|art[ií]culos?|referencias?)\s+(?:est[aá]n\s+)?en\s+cero\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:cosas?|productos?|art[ií]culos?|referencias?)\s+(?:no\s+tienen|sin)\s+stock\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:cosas?|productos?|art[ií]culos?|referencias?)\s+se\s+acabar?on\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+(?:productos|art[ií]culos|cosas)\s+(?:no\s+tienen|sin)\s+stock\b", texto_trabajo)
                or re.search(r"\b(?:productos|art[ií]culos|cosas|referencias)\s+(?:sin\s+stock|agotad[oa]s?|en\s+cero)\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:est[aá]|est[aá]n|se)\s+agotad[oa]s?\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+se\s+acabar?on\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+no\s+tienen\s+stock\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+est[aá]\s+en\s+cero\b", texto_trabajo)
                or (ranuras.alcance == "COLECCION" and re.search(r"\b(se\s+acabaron|acabaron|se\s+terminaron|agotad[oa]s?)\b", texto_trabajo))
            )
            if es_consulta_sin_stock:
                datos = {}
                if ranuras.filtro_coleccion:
                    datos["filtro_nombre"] = ranuras.filtro_coleccion
                elif ranuras.alcance == "COLECCION":
                    datos["filtrar_coleccion"] = True
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.PRODUCTOS_SIN_STOCK,
                    almacen=ranuras.almacen_especifico,
                    datos_extra=datos or None,
                ))

            # C. Alertas de inventario / stock generales (Global - ALERTAS_STOCK)
            es_consulta_alertas_global = bool(
                (
                    ranuras.condicion == "ALERTAS"
                    or re.search(r"\b(?:alerta|alertas)\s+(?:de\s+)?(?:stock|inventario)\b", texto_siglas, re.IGNORECASE)
                    or re.search(r"\bqu[eé]\s+alertas\s+(?:hay|tenemos)\b", texto_siglas, re.IGNORECASE)
                    or re.search(r"\b(?:agotad[oa]s?\s+o\s+con\s+bajo\s+st(?:ock|op)|bajo\s+st(?:ock|op)\s+o\s+agotad[oa]s?)\b", texto_siglas, re.IGNORECASE)
                    or re.search(r"\b(?:en\s+)?barro\s+st(?:ock|op)\b", texto_siglas, re.IGNORECASE)
                    or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+productos\s+(?:tienen|est[aá]n\s+en|hay\s+en)\s+bajo\s+st(?:ock|op)\b", texto_siglas, re.IGNORECASE)
                )
                and not re.search(r"\b(pelado|pelao|vacio|vac[ií]o|acab[oó]|escaso|nada)\b", texto_siglas, re.IGNORECASE)
            ) and not re.search(r"\b(por\s+debajo\s+del?\s+m[ií]nimo|no\s+tienen\s+stock|sin\s+stock|est[aá]n\s+agotados|se\s+agotaron|necesitan\s+reposici[oó]n)\b", texto_trabajo)
            if es_consulta_alertas_global:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.ALERTAS_STOCK,
                    almacen=ranuras.almacen_especifico,
                ))

            # D-0. Consulta de faltantes / compras / reposición integral («¿Qué hace falta por pedir?»)
            es_consulta_faltantes = bool(
                re.search(
                    r"\b(?:qu[eé]|cu[aá]l|cu[aá]les)\s+(?:hace\s+falta|falta|hacen\s+falta)\s+(?:por\s+|para\s+)?(?:pedir|comprar|ordenar|reponer|abastecer)\b",
                    texto_trabajo,
                )
                or re.search(r"\bqu[eé]\s+(?:tengo|tenemos|hay)\s+que\s+(?:pedir|comprar|ordenar|reponer|mandar\s+a\s+pedir)\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+(?:est[aá]|est[aá]n)\s+por\s+(?:acabarse|terminarse|agotarse)\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+debo\s+reponer\b", texto_trabajo)
                or re.search(r"\brecomendaci[oó]n\s+de\s+(?:compra|compras|reposici[oó]n)\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+hace\s+falta\s+(?:en\s+el\s+almac[eé]n|en\s+la\s+bodega|comprar|pedir)\b", texto_trabajo)
            )
            if es_consulta_faltantes:
                datos = {}
                if ranuras.filtro_coleccion:
                    datos["filtro_nombre"] = ranuras.filtro_coleccion
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_FALTANTES_REPOSICION,
                    almacen=ranuras.almacen_especifico,
                    datos_extra=datos or None,
                ))

            # D. Productos por debajo del mínimo / reposición (Global determinista - PRODUCTOS_BAJO_STOCK)
            es_consulta_bajo_stock = bool(
                ranuras.condicion == "BAJO_STOCK"
                or "STOCK_BAJO" in ranuras.conceptos_operativos
                or re.search(r"\b(?:qu[eé]|cu[aá]les|los\s+que|mu[eé]strame\s+los\s+que)\s+(?:productos\s+|art[ií]culos\s+)?(?:est[aá]n\s+)?por\s+debajo\s+(?:del?\s+(?:stock\s+)?m[ií]nimo|de\s+lo\s+(?:que\s+)?(?:deber[ií]amos|hay\s+que)\s+tener|de\s+lo\s+necesario)\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:productos\s+|art[ií]culos\s+)?(?:necesitan?|requieren?)\s+(?:reposici[oó]n|reponer|reabastecimiento|reabastecer)\b", texto_trabajo)
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:est[aá]n\s+)?por\s+debajo\s+del?\s+m[ií]nimo\b", texto_trabajo)
                or re.search(r"\b(?:productos|art[ií]culos)\s+(?:con\s+)?bajo\s+st(?:ock|op)\b", texto_trabajo)
                or re.search(r"\bpor\s+debajo\s+(?:del?\s+(?:stock\s+)?m[ií]nimo|de\s+lo\s+(?:que\s+)?(?:deber[ií]amos|hay\s+que)\s+tener)\b", texto_trabajo)
                or re.search(r"\bcu[aá]l\s+es\s+el\s+producto\s+con\s+stock\s+bajo\b", texto_trabajo)
            ) and not re.search(r"\b(agotad[oa]s?|sin\s+stock|no\s+tienen?\s+stock)\b", texto_trabajo)
            if es_consulta_bajo_stock:
                datos = {}
                if ranuras.filtro_coleccion:
                    datos["filtro_nombre"] = ranuras.filtro_coleccion
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.PRODUCTOS_BAJO_STOCK,
                    almacen=ranuras.almacen_especifico,
                    datos_extra=datos or None,
                ))

            # E. Listar almacenes / bodegas
            es_consulta_almacenes = bool(
                re.search(
                    r"\b(?:qu[eé]|cu[aá]les)\s+(?:son\s+los\s+|tenemos\s+de\s+|hay\s+de\s+|existen\s+de\s+)?(?:almacenes|bodegas)\b",
                    texto_trabajo,
                )
                or re.search(r"\bqu[eé]\s+(?:almac[eé]n|bodega)\b", texto_trabajo)
                or re.search(r"\b(?:listar?|ver|mostrar|dime|cu[aá]les\s+son)\s+(?:los\s+|las\s+)?(?:almacenes|bodegas)(?:\s+disponibles|\s+activ[oa]s)?\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+(?:almacenes|bodegas)\s+(?:tenemos|hay|manejamos)\b", texto_trabajo)
                or re.search(r"\b(?:almacenes|bodegas)\s+(?:disponibles|activ[oa]s)\b", texto_trabajo)
            )
            if es_consulta_almacenes and not re.search(r"\b(d[oó]nde|est[aá]|queda|guardan?|guardad[oa]s?|metieron)\b", texto_trabajo) and ranuras.alcance != "COLECCION":
                return _responder(IntencionResultado(tipo=IntencionTipo.LISTAR_ALMACENES))

            # E-bis. Listar ubicaciones registradas en el sistema o por almacén
            es_consulta_ubicaciones = bool(
                re.search(
                    r"\b(?:qu[eé]|cu[aá]les)\s+(?:son\s+(?:las\s+)?)?(?:ubicaciones|estantes|cajones|racks|vitrinas|gavetas|repisas)\s+(?:hay|existen|tenemos|registradas?|activas?|manejamos|disponibles?)\b",
                    texto_trabajo,
                )
                or re.search(r"\b(?:qu[eé]|cu[aá]les)\s+(?:son\s+las\s+)?ubicaciones(?:\s+del?\s+almac[eé]n|\s+de\s+la\s+bodega|\s+registradas?|\s+que\s+existen)?\b", texto_trabajo)
                or re.search(r"\b(?:listar?|ver|mostrar|dime)\s+(?:las\s+)?ubicaciones(?:\s+(?:registradas|disponibles|activas|del?\s+almac[eé]n))?\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+ubicaciones\s+(?:hay|existen|tenemos)\b", texto_trabajo)
                or re.search(r"\bubicaciones\s+(?:disponibles|activas|registradas|existentes)\b", texto_trabajo)
            )
            if es_consulta_ubicaciones and not re.search(r"\b(d[oó]nde\s+est[aá]|d[oó]nde\s+queda|d[oó]nde\s+se\s+guarda|d[oó]nde\s+lo\s+metieron)\b", texto_trabajo) and ranuras.alcance != "COLECCION":
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.LISTAR_UBICACIONES,
                    almacen=ranuras.almacen_especifico,
                ))

            # F. Ubicación de colección / referencias plurales
            es_consulta_ubicacion_coleccion = bool(
                (
                    "UBICACION" in ranuras.conceptos_operativos
                    or re.search(r"\b(d[oó]nde|ubicaci[oó]n|en\s+qu[eé]\s+(?:almac[eé]n|bodega|estante|rack|repisa|pasillo|gaveta|caj[oó]n|secci[oó]n)|d[oó]nde\s+(?:est[aá]n|se\s+guardan|se\s+encuentran|los\s+metieron)|d[oó]nde\s+guardamos|ub[ií]came)\b", texto_trabajo)
                )
                and (
                    ranuras.alcance == "COLECCION"
                    or re.search(r"\b(cada\s+(?:producto|uno|referencia|art[ií]culo)|cada\s+uno|todos\s+(?:ellos|los\s+productos|las\s+referencias|los\s+art[ií]culos|los\s+que\s+acabas|esos)|esas\s+referencias|esos\s+art[ií]culos|en\s+qu[eé]\s+estantes\s+se\s+encuentran|los\s+tres|de\s+los\s+tres|todos\s+ellos|de\s+todos)\b", texto_trabajo)
                )
                and ranuras.condicion not in ("DISPONIBLE", "AGOTADO", "BAJO_STOCK")
                and ranuras.accion not in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER")
            )
            if es_consulta_ubicacion_coleccion:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_UBICACION_COLECCION,
                    almacen=ranuras.almacen_especifico,
                ))

            # G. Listar catálogo general de productos
            es_consulta_catalogo = bool(
                re.search(r"\b(?:cu[aá]l\s+es\s+el\s+)?listado\s+de\s+productos\b", texto_trabajo)
                or re.search(r"\b(?:lista|listado|cat[aá]logo)\s+de\s+productos\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+productos\s+(?:hay|tenemos|existen|est[aá]n\s+registrados|manejamos)(?!\s+en\s+stock)\b", texto_trabajo)
                or (ranuras.alcance == "COLECCION" and ranuras.filtro_coleccion and re.search(r"\b(?:cat[aá]logo|listado|lista)\s+de\s+\w+\b|\bqu[eé]\s+(?:\w+\s+)?(?:manejamos|existen|est[aá]n\s+registrados)\b", texto_trabajo) and not re.search(r"\b(disponibles?|en\s+stock|con\s+stock|tenemos|hay)\b", texto_trabajo))
                or re.search(r"\bver\s+(?:el\s+)?(?:cat[aá]logo|listado|lista)\s+de\s+productos\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+hay\s+en\s+el\s+cat[aá]logo\b", texto_trabajo)
            ) and not re.search(r"\b(bajo\s+stock|en\s+stock|con\s+stock|agotad[oa]s?|sin\s+stock|prestado|prestados)\b", texto_trabajo)
            if es_consulta_catalogo:
                datos = {}
                if ranuras.filtro_coleccion:
                    datos["filtro_nombre"] = ranuras.filtro_coleccion
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.LISTAR_CATALOGO,
                    almacen=ranuras.almacen_especifico,
                    datos_extra=datos or None,
                ))

            # H. Productos con existencias disponibles en stock (Global o por familia/colección)
            es_consulta_disponibles = bool(
                ranuras.condicion == "DISPONIBLE"
                or (ranuras.alcance == "COLECCION" and ranuras.filtro_coleccion and not es_consulta_sin_stock and not es_consulta_bajo_stock)
                or re.search(r"\bqu[eé]\s+productos\s+(?:hay|tenemos|est[aá]n|quedan)\s+en\s+stock\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+(?:hay|tenemos|queda)\s+en\s+stock\b", texto_trabajo)
                or re.search(r"\b(?:productos|art[ií]culos)\s+(?:en\s+stock|con\s+stock|disponibles|con\s+existencias?)\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+productos\s+(?:tienen|cuentan\s+con)\s+(?:stock|existencias?|disponibilidad)\b", texto_trabajo)
                or re.search(r"\bqu[eé]\s+(?:tenemos|hay)\s+disponible\b", texto_trabajo)
                or ("STOCK_DISPONIBLE" in ranuras.conceptos_operativos and ranuras.alcance in ("GLOBAL", "COLECCION", None))
            ) and not re.search(r"\b(bajo\s+stock|agotad[oa]s?|sin\s+stock|no\s+tienen?\s+stock|qu[eé]\s+falta|necesitan\s+reposici[oó]n|prestado|prestados)\b", texto_trabajo)
            if es_consulta_disponibles:
                datos = {}
                if ranuras.filtro_coleccion:
                    datos["filtro_nombre"] = ranuras.filtro_coleccion
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.PRODUCTOS_CON_STOCK,
                    almacen=ranuras.almacen_especifico,
                    datos_extra=datos or None,
                ))

            # I. Resumen general de almacén / inventario («¿Qué hay en el almacén?»)
            es_consulta_resumen = bool(
                "RESUMEN_GENERAL" in ranuras.conceptos_operativos
                or re.search(
                    r"\b(?:dime\s+)?c[oó]mo\s+(?:va|anda|est[aá]|pinta)\s+(?:todo\s+en\s+)?(?:el\s+almac[eé]n|el\s+inventario|la\s+bodega)|resumen\s+del?\s+(?:almac[eé]n|inventario|bodega|d[ií]a|turno|jornada)|estado\s+del?\s+(?:almac[eé]n|inventario|bodega)|(?:consultar?|revisar?|ver|chequear?|dime)\s+(?:el\s+)?(?:inventario|almac[eé]n|bodega)|consulta\s+(?:el\s+)?(?:inventario|almac[eé]n|bodega)\b",
                    texto_trabajo,
                )
                or re.search(r"\bqu[eé]\s+hay\s+en\s+el\s+(?:almac[eé]n|bodega|inventario)\b", texto_trabajo)
                or (ranuras.almacen_especifico and re.search(r"\bqu[eé]\s+hay\s+en\b", texto_trabajo))
            ) and not re.search(r"\b(movimientos?|entradas?|salidas?|prestado|agotados?)\b", texto_trabajo)
            if es_consulta_resumen:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_RESUMEN_INVENTARIO,
                    almacen=ranuras.almacen_especifico,
                ))

            # J. Métricas generales de movimientos del almacén (del día / hoy / temporales)
            es_consulta_movimientos_general = bool(
                ("MOVIMIENTO" in ranuras.conceptos_operativos and ranuras.ventana_temporal is not None)
                or re.search(
                    r"\b(?:mu[eé]strame|dame|ver|consultar?|qu[eé])\s+(?:los\s+)?movimientos?\s+(?:de\s+|en\s+|durante\s+)?(?:ayer|hoy|la\s+semana|este\s+mes|el\s+mes|los\s+[uú]ltimos|[a-z]+)\b",
                    texto_trabajo,
                )
                or re.search(
                    r"\b(cu[aá]nt[ao]s?|qu[eé])\s+(?:entradas?|salidas?|despachos?|movimientos?|operaciones|transacciones?|actividades?|actividad|pr[eé]stamos?|devoluciones?)\s+(?:han\s+habido|hubo|se\s+hicieron|se\s+registraron|tuvimos|tuvieron|hay|tenemos)\b",
                    texto_trabajo,
                )
                or re.search(
                    r"\b(cu[aá]nt[ao]s?)\s+(?:entradas?|salidas?|despachos?|movimientos?|operaciones|transacciones?)\b",
                    texto_trabajo,
                )
                or re.search(
                    r"\b(?:qu[eé]|cu[aá]l|cu[aá]les)\s+(?:\w+\s+){0,3}(?:entr[oó]|entraron|ingres[oó]|ingresaron|lleg[oó]|llegaron|sali[oó]|salieron|se\s+despach[oó]|se\s+movi[oó])\b",
                    texto_trabajo,
                )
                or re.search(
                    r"\b(movimientos?\s+(?:de\s+)?(?:hoy|ayer)|movimientos?\s+del\s+d[ií]a|c[oó]mo\s+van\s+los\s+movimientos|c[oó]mo\s+va\s+la\s+movida|de\s+movida\s+la\s+vaina|el\s+voleo\s+(?:de\s+)?hoy|qu[eé]\s+voleo\s+hay|qu[eé]\s+se\s+ha\s+movido\s+hoy|c[oó]mo\s+va\s+el\s+voleo|c[oó]mo\s+estuvo\s+el\s+voleo|c[oó]mo\s+va\s+el\s+trabajo\s+hoy|qu[eé]\s+tal\s+estuvo\s+la\s+jornada|c[oó]mo\s+estuvo\s+la\s+jornada|c[oó]mo\s+estuvo\s+la\s+vuelta|c[oó]mo\s+va\s+la\s+vaina\s+hoy|c[oó]mo\s+anda\s+la\s+cosa\s+hoy|c[oó]mo\s+va\s+la\s+cosa\s+hoy|c[oó]mo\s+va\s+eso\s+hoy|c[oó]mo\s+va\s+la\s+vaina|c[oó]mo\s+est[aá]\s+la\s+vaina|qu[eé]\s+pas[oó]\s+hoy\s+en\s+(?:el\s+almac[eé]n|la\s+bodega))\b",
                    texto_trabajo,
                )
            )
            if es_consulta_movimientos_general:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_METRICAS_MOVIMIENTOS,
                    almacen=ranuras.almacen_especifico,
                ))

            # K. Creación de producto nuevo por voz
            if re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+|vamos\s+a\s+)?(?:crear|registrar|ingresar|dar\s+de\s+alta)\s+(?:un\s+)?(?:nuevo\s+)?producto\b", texto_trabajo):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CREAR_PRODUCTO,
                    almacen=ranuras.almacen_especifico,
                ))

            # L. Inhabilitar o archivar producto por voz (cuando no se nombró producto específico en texto global)
            if re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+)?(?:inhabilitar|desactivar|archivar|dar\s+de\s+baja)\s+(?:un\s+)?producto\b", texto_trabajo) and not prod_mencionado:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.INHABILITAR_PRODUCTO,
                    almacen=ranuras.almacen_especifico,
                ))

        # ─────────────────────────────────────────────────────────────────────
        # CAPA DE ENTIDAD Y JUSTIFICACIÓN CONTEXTUAL
        # ─────────────────────────────────────────────────────────────────────
        # Solo se hereda el producto anterior si existe justificación conversacional legítima:
        # 1. Referencia demostrativa explícita ("ese producto", "su stock", "cuántos quedan de ese")
        # 2. Respuesta a aclaración pendiente o selección de candidato
        # 3. Continuación elíptica inequívoca de consulta sobre producto ("¿y en el almacén B?", "¿dónde está?")
        # Caso B: Una consulta de pedido u orden NO debe heredar el producto previo
        if not prod_mencionado and re.search(r"\b(pedido|pedidos|orden|ordenes)\b", texto_trabajo) and not ranuras.es_cambio_de_tema:
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion=(
                    "No tengo información de un pedido u orden específica. En este inventario gestionamos "
                    "existencias, entradas, salidas y préstamos. ¿A qué producto o movimiento registrado te refieres?"
                ),
            ))

        producto_objetivo = None
        if prod_mencionado:
            producto_objetivo = prod_mencionado
        else:
            tiene_justificacion = False

            # 1. Referencia demostrativa o pronominal explícita al producto en foco
            es_ref_demostrativa = bool(
                analisis.demostrativo_detectado
                or re.search(r"\b(ese|esa|este|esta|aquel|aquella|eso|esos|esas|estos|estas|aquellos|aquellas|su|sus|de\s+es[eoa]s?|de\s+aquell[oas]+|dicho\s+producto|el\s+producto\s+anterior|el\s+mismo|la\s+misma|esa\s+vaina|esta\s+vaina|la\s+vaina|vaina)\b", texto_trabajo)
                or re.search(r"\b(lo\s+quiero|la\s+quiero|quiero\s+(?:inhabilitarlo|habilitarlo|desactivarlo|activarlo|archivarlo)|inhabilitarlo|habilitarlo|desactivarlo|activarlo|archivarlo|darlo\s+de\s+(?:alta|baja))\b", texto_trabajo)
                or re.search(r"\b(?:inhabilit[aá]r?lo|desact[ií]v[aá]r?lo|arch[ií]v[aá]r?lo|inhabil[ií]talo|desact[ií]valo|arch[ií]valo|habilit[aá]r?lo|activ[aá]r?lo|reactiv[aá]r?lo|hab[ií]l[ií]talo|act[ií]valo|react[ií]valo)\b", texto_trabajo)
                or (re.search(r"\b(?:lo|la)\b", texto_trabajo) and re.search(r"\b(?:inhabilitar|desactivar|habilitar|activar|archivar)\b", texto_trabajo))
                or re.search(r"\b(?:m[eé]telo|m[eé]tela|m[eé]telos|m[eé]telas|s[aá]calo|s[aá]cala|s[aá]calos|s[aá]calas|desp[aá]chalo|desp[aá]chala|desp[aá]chalos|desp[aá]chalas|agr[eé]gale|agr[eé]galos|agr[eé]galas|p[aá]same\s+un[oa]|devu[eé]lv(?:elo|ela|elos|elas|eme|e\s+un[oa]|e\s+\d+|e\s+uno\s+nada\s+m[aá]s)|regr[eé]sa(?:lo|la|los|las|me|\s+un[oa]|\s+\d+)|llegaron\s+\w+\s+(?:pa|para)\s+inventario)\b", f"{texto_siglas} {texto_trabajo}", re.IGNORECASE)
            )

            # 2. Respuesta a aclaración o propuesta pendiente
            responde_aclaracion = bool(
                candidatos_previos
                or (estado_operativo and estado_operativo.ultima_intencion == IntencionTipo.SOLICITAR_ACLARACION.value)
                or (estado_operativo and estado_operativo.propuesta_activa)
            )

            # 3. Continuación elíptica inequívoca cuando ya hay producto en contexto (sea RAM o sesión)
            es_continuacion_eliptica = False
            if producto_en_contexto is not None:
                dest_tokens = set(_quitar_tildes(ranuras.destinatario.lower()).split()) if ranuras.destinatario else set()
                alm_tokens = set(_quitar_tildes(ranuras.almacen_especifico.nombre.lower()).split()) if ranuras.almacen_especifico else set()
                tokens_consulta = [
                    w for w in _quitar_tildes(re.sub(r"[¿?¡!.,:;()\[\]\"'`]", " ", texto_siglas).lower()).split()
                    if w not in cls.PALABRAS_PARADA and not w.isdigit() and w not in dest_tokens and w not in alm_tokens and w not in {"bodega", "almacen", "bodegas", "almacenes"}
                ]
                intento_producto_especifico = bool(tokens_consulta) and not es_ref_demostrativa

                # Solo es elipsis legítima si NO hubo un intento fallido de nombrar otro producto
                if not intento_producto_especifico:
                    es_pregunta_dependiente = bool(
                        re.search(r"\b(d[oó]nde|ubicaci[oó]n|cu[aá]nto|cu[aá]ntos|quedan?|queda|stock|saldo|qui[eé]n\s+lo\s+tiene|qui[eé]n\s+tiene|prestado|prestao)\b", texto_trabajo)
                        or re.search(r"\b(?:y\s+(?:del?|en)\s+(?:el\s+)?(?:otro\s+almac[eé]n|otra\s+bodega|almac[eé]n\s+[A-Za-z0-9]+|bodega\s+[A-Za-z0-9]+))\b", texto_trabajo)
                        or (ranuras.almacen_especifico is not None and re.search(r"\b(?:y\s+en|en)\b", texto_trabajo))
                        or re.search(r"\b(qued[oó]\s+(?:pelado|vac[ií]o)|se\s+acab[oó]|no\s+hay\s+nada|pelado|vac[ií]o)\b", texto_trabajo)
                        or re.search(r"\b(c[oó]mo\s+se\s+ha\s+movido|c[oó]mo\s+ha\s+estado\s+saliendo|qu[eé]\s+movimiento\s+tuvo|qu[eé]\s+tanto\s+ha\s+salido|c[oó]mo\s+se\s+ha\s+despachado|cu[aá]nto\s+ha\s+salido|c[oó]mo\s+va|qu[eé]\s+pas[oó])\b", texto_trabajo)
                    )
                    if es_pregunta_dependiente:
                        es_continuacion_eliptica = True
                else:
                    if estado_operativo:
                        estado_operativo.limpiar_foco_conversacional()

            if es_ref_demostrativa or responde_aclaracion or es_continuacion_eliptica:
                tiene_justificacion = True

            # REGLA ESTRICTA DE AISLAMIENTO:
            # Una operación de escritura («Quiero registrar una entrada», «Registra una salida de 5»,
            # «Haz un préstamo», «Necesito sacar tres unidades», «Quiero hacer un traslado», etc.)
            # sin referencia contextual explícita (demostrativo, enclítico o cláusula anafórica)
            # JAMÁS debe heredar automáticamente el producto del contexto anterior.
            from ..agente.planificador import tiene_referencia_contextual_operacion
            es_accion_escritura = bool(
                ranuras.accion in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER", "TRASLADAR")
                or re.search(
                    r"\b(?:registrar|hacer|nueva|necesito|quiero|voy\s+a)?\s*(?:una?\s+)?(?:entrada|salida|pr[eé]stamo|traslado|devoluci[oó]n|sacar|retirar|prestar|ingresar|trasladar|devolver|inhabilitar|desactivar|habilitar|reactivar|archivar)\b",
                    texto_trabajo,
                )
            )
            if es_accion_escritura and not tiene_referencia_contextual_operacion(f"{texto_siglas} {texto_trabajo}"):
                tiene_justificacion = False

            if tiene_justificacion:
                if producto_en_contexto:
                    producto_objetivo = producto_en_contexto
                elif candidatos_previos and len(candidatos_previos) == 1:
                    cand_id = candidatos_previos[0].get("id")
                    if cand_id:
                        cand_prod = Producto.objects.filter(id=cand_id, activo=True, archivado=False).first()
                        if cand_prod:
                            producto_objetivo = cand_prod

        # ─────────────────────────────────────────────────────────────────────
        # RESOLUCIÓN DE ACCIONES Y CONSULTAS SOBRE EL PRODUCTO
        # ─────────────────────────────────────────────────────────────────────
        if producto_objetivo:
            # 2.A Movimientos de producto específico (incluye "cómo va esa vaina", "qué pasó con esa vaina", etc.)
            if ("MOVIMIENTO" in ranuras.conceptos_operativos and ranuras.accion not in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER", "TRASLADAR")) or re.search(
                r"\b(c[oó]mo\s+se\s+ha\s+movido|qu[eé]\s+tanto\s+(?:ha\s+salido|se\s+ha\s+movido)|c[oó]mo\s+se\s+ha\s+despachado|movimientos?\s+del?\s+|qu[eé]\s+movimientos\s+(?:tiene|ha\s+tenido)|cu[aá]nto\s+ha\s+salido|c[oó]mo\s+va\s+(?:esa\s+vaina|este\s+cable|el\s+cable|este\s+producto|eso|aquello)|qu[eé]\s+pas[oó]\s+con\s+(?:esa\s+vaina|este\s+cable|el\s+cable|este\s+producto|eso|aquello)|qu[eé]\s+movimiento\s+tuvo|c[oó]mo\s+ha\s+estado\s+saliendo)\b",
                texto_trabajo,
            ):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_PRODUCTO_MOVIMIENTOS,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                ))

            # 2.B Consulta de ubicación del producto
            if ("UBICACION" in ranuras.conceptos_operativos or ranuras.contexto_fisico or re.search(
                r"\b(d[oó]nde|ubicaci[oó]n|localizaci[oó]n|estante|caj[oó]n|rack|pasillo|gaveta|repisa|secci[oó]n|posici[oó]n|en\s+qu[eé]\s+lado|d[oó]nde\s+guardan|d[oó]nde\s+se\s+encuentra)\b",
                texto_trabajo,
            )) and ranuras.accion not in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER", "TRASLADAR"):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_UBICACION,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                ))

            # 2.B-bis Consulta de préstamos activos / quién tiene el producto en préstamo
            if (
                ranuras.accion not in ("PRESTAR", "DEVOLVER", "REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "TRASLADAR")
                and not re.search(r"\b(te\s+regreso|regreso|devuelvo|devolver|devoluci[oó]n|d[eé]jame\s+un[oa]|p[aá]same\s+un[oa]\s+prestado)\b", texto_trabajo)
                and (
                    re.search(r"\b(pr[eé]stamos activos|qui[eé]n tiene|pr[eé]stamos pendientes|est[aá]n?\s+presta(?:d[oa]|o)s?|tiene\s+presta(?:d[oa]|o)s?|hay\s+presta(?:d[oa]|o)s?|queda\s+presta(?:d[oa]|o)s?|cu[aá]nt[oa]s?\s+(?:hay\s+|queda\s+|est[aá]n?\s+)?presta(?:d[oa]|o)s?|cu[aá]ntas\s+unidades\s+(?:hay\s+|est[aá]n?\s+)?presta(?:d[oa]|o)s?)\b", texto_trabajo)
                    or (ranuras.condicion == "EN_PRESTAMO" and re.search(r"\b(cu[aá]nto|qu[eé]|qui[eé]n|hay)\b", texto_trabajo))
                )
            ):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.PRESTAMOS_ACTIVOS,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                ))

            # 2.C Consulta de stock / existencias del producto
            es_otro_alm = bool(re.search(r"\b(?:otro\s+almac[eé]n|otra\s+bodega)\b", texto_trabajo))
            es_proyeccion = bool(re.search(r"\b(despu[eé]s\s+de\s+eso|tras\s+(?:la\s+)?operaci[oó]n|despu[eé]s)\b", texto_trabajo))
            if (
                "STOCK" in ranuras.conceptos_operativos
                or es_otro_alm
                or es_proyeccion
                or re.search(
                    r"\b(stock|cu[aá]nto|cu[aá]ntos|cu[aá]ntas|hay|existencias?|quedan?|queda|disponible|disponibilidad|saldo|saldos|unidades?|piezas?|cantidad|cantidades|tenemos|contamos|quedan\s+muchos|quedan\s+pocos|quedan\s+bastantes|hay\s+bastante|hay\s+algo|todav[ií]a\s+hay|qued[oó]\s+pelado|qued[oó]\s+vac[ií]o|se\s+acab[oó]|se\s+acabado|acabado|acabados|acab[oó]|se\s+acabaron|acabaron|agotado|agotados|no\s+hay\s+nada|est[aá]\s+pelado|est[aá]\s+escaso|est[aá]\s+flojo|est[aá]\s+agotado)\b",
                    texto_trabajo,
                )
            ) and ranuras.accion not in ("REGISTRAR_ENTRADA", "REGISTRAR_SALIDA", "PRESTAR", "DEVOLVER", "TRASLADAR"):
                datos = {}
                if es_otro_alm:
                    datos["otro_almacen"] = True
                if es_proyeccion:
                    datos["proyeccion_post_operacion"] = True
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_STOCK,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                    datos_extra=datos or None,
                ))

            # 2.D Devolución de préstamo (evaluar ANTES de préstamo)
            if ranuras.accion == "DEVOLVER" or re.search(r"\b(devolver|devoluci[oó]n|devu[eé]lv(?:e|en|o|eme|an)|te\s+regreso|regreso|devuelvo|ya\s+volvi[oó]|lo\s+traje\s+otra\s+vez)\b", texto_trabajo):
                match_dest = re.search(r"\b(?:de|por|a|para)\s+([A-Za-zÁ-ÿ\s]+?)(?:\s*$)", texto_siglas, re.IGNORECASE)
                dest = match_dest.group(1).strip() if match_dest else None
                if dest:
                    dest = re.sub(r"\b(pr[eé]stamo|pr[eé]stamos|de|el|la)\b", "", dest, flags=re.IGNORECASE).strip()

                match_ref = re.search(r"\b(?:pr[eé]stamo|ref(?:erencia)?|id)\s*[:#]?\s*([0-9a-fA-F-]{6,36})\b", texto_siglas, re.IGNORECASE)
                datos_extra_dev = {"prestamo_id": match_ref.group(1).lower()} if match_ref else None

                return _responder(IntencionResultado(
                    tipo=IntencionTipo.DEVOLUCION,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                    destinatario=dest,
                    cantidad=analisis.cantidad_detectada,
                    datos_extra=datos_extra_dev,
                    requiere_confirmacion=True,
                ))

            # 2.E Préstamo
            if ranuras.accion == "PRESTAR" or (re.search(r"\b(prestar|presta|pr[eé]stale|prestale|preste|presten|pr[eé]stamo|pr[eé]stame|d[eé]jame\s+uno|d[eé]jame\s+una|p[aá]same\s+un[oa]\s+prestado|an[oó]tale\s+(?:ese\s+)?pr[eé]stamo)\b", texto_trabajo) and not re.search(r"\b(devolver|devoluci[oó]n|regreso|devuelvo|no\s+(?:lo\s+)?voy\s+a\s+prestar)\b", texto_trabajo)):
                match_dest = re.search(r"\b(?:para|a)\s+([A-Za-zÁ-ÿ\s]+?)(?:\s+(?:por|motivo)|\s*$)", texto_siglas, re.IGNORECASE)
                dest = match_dest.group(1).strip() if match_dest else None
                if dest:
                    dest = re.sub(r"\b(?:en|desde|hacia)\s+(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\b.*$", "", dest, flags=re.IGNORECASE).strip()
                    # Si el destinatario capturó el nombre del producto (ej. "a Yankar del segundo producto" o "del producto a Yankar")
                    # limpiar el nombre del producto o referencias ordinales
                    dest = re.sub(r"\b(?:del?|de\s+la|el|la)\s+(?:segundo|primer|tercer|segunda|primera|tercera)\s+producto\b", "", dest, flags=re.IGNORECASE).strip()
                    if producto_objetivo and producto_objetivo.nombre.lower() in dest.lower():
                        dest = re.sub(re.escape(producto_objetivo.nombre), "", dest, flags=re.IGNORECASE).strip()

                return _responder(IntencionResultado(
                    tipo=IntencionTipo.PRESTAMO,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                    cantidad=analisis.cantidad_detectada,
                    destinatario=dest or None,
                    requiere_confirmacion=True,
                ))

            # 2.E-bis Edición de producto identificado
            if re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+)?(?:editar|modificar|actualizar)\s+(?:el\s+|este\s+|ese\s+)?producto\b", texto_trabajo) or re.search(r"\b(?:editar|modificar|actualizar)\b", texto_trabajo):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.EDITAR_PRODUCTO,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                ))

            # 2.E-ter Inhabilitar o archivar producto identificado
            if re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+)?(?:ir\s+a\s+)?(?:inhabilitar|desactivar|archivar|dar\s+de\s+baja)\b", texto_trabajo) or re.search(r"\b(?:inhabilit[aá]r?lo|desact[ií]v[aá]r?lo|arch[ií]v[aá]r?lo|inhabil[ií]talo|desact[ií]valo|arch[ií]valo)\b", texto_trabajo):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.INHABILITAR_PRODUCTO,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                    requiere_confirmacion=True,
                ))

            # 2.E-ter-bis Habilitar o reactivar producto identificado
            if re.search(r"\b(?:quiero\s+|deseo\s+|voy\s+a\s+)?(?:ir\s+a\s+)?(?:habilitar|activar|reactivar|dar\s+de\s+alta)\b", texto_trabajo) or re.search(r"\b(?:habilit[aá]r?lo|activ[aá]r?lo|reactiv[aá]r?lo|hab[ií]l[ií]talo|act[ií]valo|react[ií]valo)\b", texto_trabajo):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.HABILITAR_PRODUCTO,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                    requiere_confirmacion=True,
                ))

            # 2.E-quater Traslado entre almacenes con producto identificado
            if ranuras.accion == "TRASLADAR" or re.search(r"\b(trasladar|transferir|transferencia|mover\s+a\s+(?:otro\s+almac[eé]n|otra\s+bodega)|pasar\s+a\s+(?:otro\s+almac[eé]n|otra\s+bodega))\b", texto_trabajo):
                almacen_destino = None
                almacen_origen = None
                texto_norm_completo = _quitar_tildes(texto_trabajo.lower())
                qs_alm = list(Almacen.objects.filter(activo=True, eliminado=False))

                # 1. Coincidencia de almacén destino (precedido por a, al, hacia, para)
                for alm in qs_alm:
                    alm_norm = _quitar_tildes(alm.nombre.lower().strip())
                    patron = rf"\b(?:a|al|hacia|para)\s+(?:el\s+|la\s+)?(?:almac[eé]n\s+|bodega\s+)?{re.escape(alm_norm)}\b"
                    if re.search(patron, texto_norm_completo):
                        almacen_destino = alm
                        break

                if not almacen_destino:
                    match_destino = re.search(
                        r"\b(?:a(?:l)?|hacia|para)\s+(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\s+([A-Za-zÁ-ÿ0-9\s]+?)(?:\s*$|\s+\d|\s+del?\s+)",
                        texto_trabajo,
                    )
                    if match_destino:
                        nombre_dest = match_destino.group(1).strip()
                        texto_norm_dest = _quitar_tildes(nombre_dest.lower())
                        for alm in qs_alm:
                            if _quitar_tildes(alm.nombre.lower()) in texto_norm_dest or texto_norm_dest in _quitar_tildes(alm.nombre.lower()):
                                almacen_destino = alm
                                break

                # 2. Coincidencia de almacén origen (precedido por desde, del, de)
                for alm in qs_alm:
                    alm_norm = _quitar_tildes(alm.nombre.lower().strip())
                    patron_orig = rf"\b(?:desde|del?)\s+(?:el\s+|la\s+)?(?:almac[eé]n\s+|bodega\s+)?{re.escape(alm_norm)}\b"
                    if re.search(patron_orig, texto_norm_completo):
                        almacen_origen = alm
                        break

                # 3. Si no hay origen explícito pero ranuras detectó un almacén y no es el destino:
                if not almacen_origen:
                    if ranuras.almacen_especifico and (not almacen_destino or ranuras.almacen_especifico.id != almacen_destino.id):
                        almacen_origen = ranuras.almacen_especifico

                datos = {}
                if almacen_destino:
                    datos["almacen_destino_id"] = almacen_destino.id
                    datos["almacen_destino_nombre"] = almacen_destino.nombre

                return _responder(IntencionResultado(
                    tipo=IntencionTipo.TRASLADAR_PRODUCTO,
                    producto=producto_objetivo,
                    almacen=almacen_origen,
                    cantidad=analisis.cantidad_detectada,
                    requiere_confirmacion=True,
                    datos_extra=datos or None,
                ))

            # 2.F Entrada de stock
            if ranuras.accion == "REGISTRAR_ENTRADA" or re.search(r"\b(entrada|ingresar|ingreso|llegaron|lleg[oó]|recibir|aumentar|meter|mete|m[eé]telo|m[eé]tame|emboca|emb[oó]calo|agr[eé]gale|agregar)\b", texto_trabajo):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.REGISTRAR_ENTRADA,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                    cantidad=analisis.cantidad_detectada,
                    requiere_confirmacion=True,
                ))

            # 2.G Salida de stock
            if ranuras.accion == "REGISTRAR_SALIDA" or re.search(r"\b(salida|retirar|sacar|saca|s[aá]came|desp[aá]chame|despacha|desp[aá]chalo|m[aá]ndame|b[aá]jame|b[aá]jalo|entregar|p[aá]same|gastar)\b", texto_trabajo):
                match_dest = re.search(r"\b(?:para|a)\s+([A-Za-zÁ-ÿ\s]+?)(?:\s+(?:por|motivo)|\s*$)", texto_siglas, re.IGNORECASE)
                dest = match_dest.group(1).strip() if match_dest else None
                if dest:
                    dest = re.sub(r"\b(?:en|desde|hacia)\s+(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\b.*$", "", dest, flags=re.IGNORECASE).strip()

                match_mot = re.search(r"\b(?:motivo|por)\s+([A-Za-zÁ-ÿ\s0-9]+)$", texto_siglas, re.IGNORECASE)
                mot = match_mot.group(1).strip() if match_mot else ranuras.motivo

                return _responder(IntencionResultado(
                    tipo=IntencionTipo.REGISTRAR_SALIDA,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                    cantidad=analisis.cantidad_detectada,
                    destinatario=dest,
                    motivo=mot,
                    requiere_confirmacion=True,
                ))

            # 2.H Búsqueda o consulta del producto si fue explícitamente nombrado
            # o si el usuario usó un demostrativo explícito hacia él
            if prod_mencionado or (analisis.demostrativo_detectado or re.search(r"\b(ese|este|aquel|eso)\b", texto_trabajo)):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.BUSCAR_PRODUCTO,
                    producto=producto_objetivo,
                    almacen=ranuras.almacen_especifico,
                ))

        # ─────────────────────────────────────────────────────────────────────
        # PRIORIDAD 3: Acciones transaccionales intentadas SIN producto
        # ─────────────────────────────────────────────────────────────────────
        if ranuras.accion == "REGISTRAR_SALIDA" or re.search(r"\b(salida|retirar|sacar|saca|s[aá]came|desp[aá]chame|despacha|desp[aá]chalo|m[aá]ndame|b[aá]jame|b[aá]jalo|entregar|p[aá]same|gastar)\b", texto_trabajo):
            match_dest = re.search(r"\b(?:para|a\s+nombre\s+de|entregar\s+a|a)\s+([A-Za-zÁ-ÿñÑ\s'-]+?)(?:\s+(?:por|motivo)|\s*$)", texto_siglas, re.IGNORECASE)
            dest = match_dest.group(1).strip() if match_dest else None
            if dest:
                dest = re.sub(r"\b(?:en|desde|hacia)\s+(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\b.*$", "", dest, flags=re.IGNORECASE).strip()
            match_mot = re.search(r"\b(?:motivo|por)\s+([A-Za-zÁ-ÿ\s0-9]+)$", texto_siglas, re.IGNORECASE)
            mot = match_mot.group(1).strip() if match_mot else ranuras.motivo
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "REGISTRAR_SALIDA",
                    "esperando": "producto",
                    "cantidad": analisis.cantidad_detectada,
                    "destinatario": dest,
                    "motivo": mot,
                    "almacen_id": ranuras.almacen_especifico.id if ranuras.almacen_especifico else None,
                }
            dem = sanitizar_demostrativo_salida(analisis.demostrativo_detectado) if analisis.demostrativo_detectado else None
            if dem:
                msg = f"No tengo un producto asociado actualmente ({dem}, no indicaste cuál producto). Indícame qué producto deseas retirar."
            else:
                msg = "Por favor indica el nombre o código del producto que deseas retirar del inventario."
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion=msg,
            ))

        if ranuras.accion == "REGISTRAR_ENTRADA" or re.search(r"\b(entrada|ingresar|ingreso|llegaron|lleg[oó]|recibir|aumentar|meter|mete|m[eé]telo|m[eé]tame|emboca|emb[oó]calo|agr[eé]gale|agregar)\b", texto_trabajo):
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "REGISTRAR_ENTRADA",
                    "esperando": "producto",
                    "cantidad": analisis.cantidad_detectada,
                    "almacen_id": ranuras.almacen_especifico.id if ranuras.almacen_especifico else None,
                }
            if producto_en_contexto:
                msg = f"Indica el nombre o código del producto para registrar la entrada y la cantidad (¿te refieres a {producto_en_contexto.nombre} o a otro producto?)."
            else:
                msg = "Indica el nombre o código del producto y la cantidad para registrar la entrada."
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion=msg,
            ))

        if ranuras.accion == "DEVOLVER" or re.search(r"\b(devolver|devoluci[oó]n|te\s+regreso|regreso|devuelvo|ya\s+volvi[oó]|lo\s+traje\s+otra\s+vez)\b", texto_trabajo):
            match_ref = re.search(r"\b(?:pr[eé]stamo|ref(?:erencia)?|id)\s*[:#]?\s*([0-9a-fA-F-]{6,36})\b", texto_siglas, re.IGNORECASE)
            datos_extra_dev = {"prestamo_id": match_ref.group(1).lower()} if match_ref else None

            match_dest = re.search(r"\b(?:de|por|a|para)\s+([A-Za-zÁ-ÿ\s]+?)(?:\s*$)", texto_siglas, re.IGNORECASE)
            dest = match_dest.group(1).strip() if match_dest else None
            if dest:
                dest = re.sub(r"\b(pr[eé]stamo|pr[eé]stamos|de|el|la|ref|referencia)\b", "", dest, flags=re.IGNORECASE).strip()

            if dest or datos_extra_dev:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.DEVOLUCION,
                    producto=None,
                    destinatario=dest,
                    cantidad=analisis.cantidad_detectada,
                    datos_extra=datos_extra_dev,
                    requiere_confirmacion=True,
                ))

            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "DEVOLUCION",
                    "esperando": "identificacion",
                    "cantidad": analisis.cantidad_detectada,
                }
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion="Indica el producto o destinatario del préstamo que se está devolviendo.",
            ))

        if ranuras.accion == "PRESTAR" or (re.search(r"\b(prestar|pr[eé]stamo|pr[eé]stame|d[eé]jame\s+uno|d[eé]jame\s+una|p[aá]same\s+un[oa]\s+prestado)\b", texto_trabajo) and not re.search(r"\b(devolver|devoluci[oó]n|regreso|devuelvo)\b", texto_trabajo)):
            match_dest = re.search(r"\b(?:para|a\s+nombre\s+de|a)\s+([A-Za-zÁ-ÿñÑ\s'-]+?)(?:\s+(?:por|motivo)|\s*$)", texto_siglas, re.IGNORECASE)
            dest = match_dest.group(1).strip() if match_dest else None
            if dest:
                dest = re.sub(r"\b(?:en|desde|hacia)\s+(?:el\s+|la\s+)?(?:almac[eé]n|bodega)\b.*$", "", dest, flags=re.IGNORECASE).strip()
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "PRESTAMO",
                    "esperando": "producto",
                    "cantidad": analisis.cantidad_detectada,
                    "destinatario": dest,
                    "almacen_id": ranuras.almacen_especifico.id if ranuras.almacen_especifico else None,
                }
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion="Por favor indica qué producto deseas prestar.",
            ))

        if ranuras.accion == "TRASLADAR" or re.search(r"\b(trasladar|traslado|transferir|transferencia|mover\s+a\s+(?:otro\s+almac[eé]n|otra\s+bodega)|pasar\s+a\s+(?:otro\s+almac[eé]n|otra\s+bodega))\b", texto_trabajo):
            if estado_operativo:
                estado_operativo.operacion_pendiente = {
                    "tipo": "TRASLADO_ENTRE",
                    "esperando": "producto",
                    "cantidad": analisis.cantidad_detectada,
                    "almacen_id": ranuras.almacen_especifico.id if ranuras.almacen_especifico else None,
                }
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion="Indica el nombre o código del producto que deseas trasladar entre almacenes.",
            ))

        # Manejo contextual de referencias a pedidos u órdenes sin módulo de compras externo
        if re.search(r"\b(pedido|pedidos|orden|ordenes)\b", texto_trabajo) and not ranuras.es_cambio_de_tema:
            # Caso B: Una consulta independiente de pedido NO debe heredar el producto de turnos anteriores
            # a menos que exista una referencia explícita en el mismo texto.
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion=(
                    "No tengo información de un pedido u orden específica. En este inventario gestionamos "
                    "existencias, entradas, salidas y préstamos. ¿A qué producto o movimiento registrado te refieres?"
                ),
            ))

        if re.search(r"\b(c[oó]mo\s+se\s+ha\s+movido|qu[eé]\s+tanto\s+(?:ha\s+salido|se\s+ha\s+movido)|c[oó]mo\s+se\s+ha\s+despachado|movimientos?\s+del?\s+|qu[eé]\s+movimientos\s+(?:tiene|ha\s+tenido)|cu[aá]nto\s+ha\s+salido)\b", texto_trabajo):
            if producto_en_contexto:
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.CONSULTAR_PRODUCTO_MOVIMIENTOS,
                    producto=producto_en_contexto,
                ))
            dem = sanitizar_demostrativo_salida(analisis.demostrativo_detectado) if analisis.demostrativo_detectado else None
            if dem:
                msg = f"Te refieres a '{dem}', pero no tengo un producto asociado actualmente. ¿De cuál producto deseas consultar los movimientos?"
            else:
                msg = "¿De cuál producto deseas consultar los movimientos?"
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion=msg,
            ))

        # PRIORIDAD 4: Referencias demostrativas huérfanas / comodines ambiguos sin producto
        # Solo si no se trata de una consulta específica de ubicación o stock sin producto
        es_consulta_especifica = bool(
            "UBICACION" in ranuras.conceptos_operativos
            or ranuras.contexto_fisico
            or "STOCK" in ranuras.conceptos_operativos
            or re.search(r"\b(d[oó]nde|ubicaci[oó]n|localizaci[oó]n|estante|rack|pasillo|gaveta|repisa|caj[oó]n|stock|cu[aá]nto|cu[aá]ntos|cu[aá]ntas|quedan?|queda|hay|disponible|disponibilidad)\b", texto_trabajo)
        )

        es_referencia_demostrativa = (
            not es_consulta_especifica
            and (
                analisis.es_ambiguo
                or bool(re.search(r"\b(ese|este|aquel|eso|esos|esas|anterior|vaina|volado|de\s+esos)\b", texto_trabajo))
            )
        )
        if es_referencia_demostrativa:
            dem = sanitizar_demostrativo_salida(analisis.demostrativo_detectado)
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion=(
                    f"Te refieres a '{dem}', pero no tengo un producto asociado actualmente (no indicaste cuál producto). "
                    "Por favor especifica el nombre o código interno del producto."
                ),
            ))

        # PRIORIDAD 5: Consultas específicas sin producto asociado (solicitar aclaración)

        # C. Consulta de ubicación sin producto
        if (
            "UBICACION" in ranuras.conceptos_operativos
            or ranuras.contexto_fisico
            or re.search(r"\b(d[oó]nde|ubicaci[oó]n|localizaci[oó]n|estante|caj[oó]n|rack|pasillo|gaveta|repisa|secci[oó]n|posici[oó]n|en qu[eé] lado|d[oó]nde\s+guardan|d[oó]nde\s+se\s+encuentra)\b", texto_trabajo)
        ) and not cls._es_fuera_de_dominio(texto_trabajo):
            if re.search(r"\b(?:qu[eé]|cu[aá]les)\s+.*ubicaciones\b|\bubicaciones\s+(?:hay|existen|registradas|tenemos|disponibles)\b|\blistar?\s+(?:las\s+)?ubicaciones\b", texto_trabajo):
                return _responder(IntencionResultado(
                    tipo=IntencionTipo.LISTAR_UBICACIONES,
                    almacen=ranuras.almacen_especifico,
                ))
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion="¿De cuál producto deseas conocer la ubicación?",
            ))

        # D. Consulta de stock sin producto
        if (
            "STOCK" in ranuras.conceptos_operativos
            or re.search(r"\b(stock|cu[aá]nto|cu[aá]ntos|cu[aá]ntas|hay|existencias?|quedan?|queda|disponible|disponibilidad|saldo|saldos|unidades?|piezas?|cantidad|cantidades|tenemos|contamos)\b", texto_trabajo)
        ) and not cls._es_fuera_de_dominio(texto_trabajo):
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion="¿De cuál producto deseas consultar el stock?",
            ))

        # E. Consulta de préstamo sin producto
        if (
            "PRESTAMO" in ranuras.conceptos_operativos
            or re.search(r"\b(qui[eé]n\s+lo\s+tiene|qui[eé]n\s+tiene\s+presta[doa]|a\s+qui[eé]n\s+se\s+le\s+prest[oó]|est[aá]\s+presta[doa]|qui[eé]n\s+tiene)\b", texto_trabajo)
        ) and not cls._es_fuera_de_dominio(texto_trabajo):
            return _responder(IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion="¿De cuál producto deseas consultar los préstamos?",
            ))

        # PRIORIDAD 6: Fallback lingüístico con separación semántica
        tipo_fallback = cls._clasificar_fallback(texto_trabajo)
        return _responder(IntencionResultado(
            tipo=tipo_fallback,
            datos_extra={"texto": texto_siglas},
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # SEPARACIÓN SEMÁNTICA DE FALLBACK LINGÜÍSTICO (FASE 1)
    # ─────────────────────────────────────────────────────────────────────────

    PATRON_FUERA_DOMINIO = (
        r"\b("
        r"clima|lluvia|temperatura|tiempo\s+meteorol[oó]gico|hierve\s+el\s+agua|"
        r"m[uú]sica|canci[oó]n|canciones|reproduc(?:e|ir)|"
        r"pel[ií]cula|pel[ií]culas|cine|serie|series|"
        r"chiste|chistes|broma|bromas|cu[eé]ntame\s+un\s+chiste|"
        r"historia|historias|cuento|cuentos|"
        r"f[uú]tbol|partido|partidos|gol|goles|liga|campeonato|copa\s+del\s+mundo|"
        r"noticias?|peri[oó]dico|actualidad|qu[eé]\s+est[aá]\s+pasando\s+en\s+las\s+noticias|"
        r"pol[ií]tica|elecciones|presidente|gobierno|"
        r"d[oó]lar|d[oó]lares|euro|euros|tasa\s+de\s+cambio|divisa|divisas|cripto|bitcoin|"
        r"hora|qu[eé]\s+hora\s+es|cu[aá]l\s+es\s+la\s+hora|"
        r"planetas?\s+del\s+sistema\s+solar|astronom[ií]a|"
        r"qu[eé]\s+es\s+(?:el|la|los|las)\s+(?:sol|luna|tierra|universo|vida|amor|filosof[ií]a|muerte)|"
        r"qui[eé]n\s+(?:es|fue|descubri[oó]|invent[oó]|fund[oó]|conquist[oó]|gan[oó])\s+[a-zñáéíóú\s]+|"
        r"c[oó]mo\s+funciona\s+(?:el\s+mundo|el\s+universo|la\s+gravedad)|"
        r"cu[aá]nto\s+cuesta\s+(?:viajar|un\s+vuelo|un\s+hotel|un\s+viaje|ir\s+a)\s+[a-zñáéíóú\s]+|"
        r"viajar\s+a|turismo\s+en|vuelos?\s+a|"
        r"resuelve\s+(?:esta\s+)?(?:ecuaci[oó]n|f[oó]rmula|problema\s+matem[aá]tico)|"
        r"cu[aá]nto\s+es\s+\d+\s*(?:por|\+|\-|x|\/|\*|m[aá]s|menos|dividido\s+(?:por|entre)?)\s*\d+|"
        r"calcula\s+\d+\s*(?:por|\+|\-|x|\/|\*|m[aá]s|menos|dividido\s+(?:por|entre)?)\s*\d+|"
        r"(?:expl[ií]came\s+)?(?:la\s+)?historia\s+de\s+[a-zñáéíóú\s]+|"
        r"qu[eé]\s+pas[oó]\s+hoy\s+en\s+[a-zñáéíóú\s]+|"
        r"receta|recetas|cocinar|ingredientes\s+para"
        r")\b"
    )

    @classmethod
    def _es_fuera_de_dominio(cls, texto: str) -> bool:
        """Determina de forma determinista si la locución corresponde a un tema explícitamente fuera de dominio.
        Garantiza protección estricta si la locución incluye intenciones o términos de inventario.
        """
        # Salvaguarda: si menciona términos operativos o transaccionales del inventario, NO es fuera de dominio
        terminos_inventario = (
            r"\b(stock|existencias?|inventario|almac[eé]n|bodega|pr[eé]stamo|pr[eé]stamos|"
            r"devoluci[oó]n|entrada|salida|cables?|hdmi|utp|rj45|switch|ponchadora|"
            r"pedido|pedidos|orden|ordenes|art[ií]culo|art[ií]culos|producto|productos)\b"
        )
        if re.search(terminos_inventario, texto, flags=re.IGNORECASE) and not re.search(r"\b(chiste|broma|canci[oó]n|pel[ií]cula|noticias)\b", texto, flags=re.IGNORECASE):
            return False
        return bool(re.search(cls.PATRON_FUERA_DOMINIO, texto, flags=re.IGNORECASE))

    @classmethod
    def _clasificar_fallback(cls, texto_trabajo: str) -> IntencionTipo:
        """Clasifica locuciones que no coincidieron con ninguna intención operativa.

        Orden estricto determinista:
        1. Baja comprensión / ruido / transcripción incompleta (SOLICITUD_NO_COMPRENDIDA)
        2. Fuera de dominio temático real (NO_INTENCION_INVENTARIO)
        3. Caso ambiguo o residual no reconocido (SOLICITUD_NO_COMPRENDIDA)
        """
        texto_limpio = re.sub(r"[¿?¡!.,:;()\[\]\"'`]", " ", texto_trabajo).strip().lower()
        texto_limpio = re.sub(r"\s+", " ", texto_limpio)
        tokens = texto_limpio.split()

        # PRIMERO: Entradas de baja comprensión o señales aisladas de Whisper
        tokens_baja_comprension = {
            "menos", "eh", "ah", "mmm", "mm", "um", "uh", "aja", "ajá",
            "que", "qué", "buen"
        }
        if not tokens or len(texto_limpio) <= 2:
            return IntencionTipo.SOLICITUD_NO_COMPRENDIDA

        # Si el texto completo es una sola palabra de baja comprensión o palabra muy corta sin sentido
        if len(tokens) == 1 and (tokens[0] in tokens_baja_comprension or len(tokens[0]) <= 3):
            return IntencionTipo.SOLICITUD_NO_COMPRENDIDA

        # Si todos los tokens pertenecen al vocabulario de baja comprensión / titubeos
        if set(tokens).issubset(tokens_baja_comprension | {"oye", "haku", "este", "bueno"}):
            return IntencionTipo.SOLICITUD_NO_COMPRENDIDA

        # SEGUNDO: Fuera de dominio real (temas no relacionados con inventario)
        if cls._es_fuera_de_dominio(texto_limpio):
            return IntencionTipo.NO_INTENCION_INVENTARIO

        # TERCERO: Detección parcial de intención incompleta (el usuario mencionó algo de
        # inventario pero sin la información mínima para clasificar). Distinguir de producto
        # ambiguo: si el texto menciona términos operativos genéricos, es intención incompleta.
        _mencion_inventario_generica = bool(
            re.search(r"\b(trasladar|mover|transferir|cambiar|pasar)\b", texto_limpio)
            or re.search(r"\b(producto|articulo|referencia|material|equipo|herramienta)\b", texto_limpio)
        )
        if _mencion_inventario_generica:
            return IntencionTipo.SOLICITUD_NO_COMPRENDIDA

        # CUARTO: Cualquier caso ambiguo devuelve SOLICITUD_NO_COMPRENDIDA
        return IntencionTipo.SOLICITUD_NO_COMPRENDIDA

    # ─────────────────────────────────────────────────────────────────────────
    # RESOLUCIÓN NO PRESUNTIVA DE CÓDIGOS Y NOMBRES (REGLA 5)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _resolver_producto_en_texto(
        cls,
        texto: str,
        usuario: UsuarioAudacIA,
        candidatos_previos: Optional[List[Dict[str, Any]]] = None,
        incluir_inactivos: bool = False,
    ) -> tuple[Optional[Producto], Optional[str], List[Dict[str, Any]]]:
        # Consultas normales y operaciones de baja solo resuelven productos activos.
        # La ruta de reactivación puede incluir inactivos, siempre dentro del mismo
        # alcance de almacenes que ya tiene autorizado el usuario.
        qs = (
            Producto.objects.all()
            if usuario.es_administrador or incluir_inactivos
            else Producto.objects.filter(activo=True, archivado=False)
        )
        if usuario.es_usuario_normal:
            if usuario.acceso_todos_almacenes:
                pass
            else:
                alms_accesibles = usuario.obtener_almacenes_accesibles()
                if alms_accesibles.exists():
                    qs = qs.filter(
                        Q(categoria__almacen_principal__in=alms_accesibles) |
                        Q(existencias__almacen__in=alms_accesibles)
                    ).distinct()
                else:
                    qs = qs.none()

        texto_siglas = NormalizadorSiglasTecnicas.normalizar(texto)
        texto_clean = _quitar_tildes(re.sub(r"[¿?¡!.,:;()\[\]\"'`]", " ", texto_siglas).strip().lower())

        # El agente puede entregar aquí el nombre canónico aislado de un producto.
        # Resuélvelo contra el catálogo accesible antes de aplicar el filtro temático:
        # un nombre real puede contener palabras como «temperatura» que, solas,
        # aparecen en peticiones meteorológicas fuera del inventario.
        for texto_candidato in dict.fromkeys((texto.strip(), texto_siglas.strip())):
            if not texto_candidato:
                continue
            coincidencias_exactas = list(qs.filter(nombre__iexact=texto_candidato)[:4])
            if not coincidencias_exactas:
                continue
            activas = [p for p in coincidencias_exactas if p.activo and not p.archivado]
            mejores = activas or coincidencias_exactas
            if len(mejores) == 1:
                return mejores[0], None, []
            return (
                None,
                "Encontré varios productos con ese nombre. ¿Cuál de ellos deseas consultar?",
                [
                    {"id": p.id, "nombre": p.nombre, "codigo": p.codigo_interno}
                    for p in mejores[:3]
                ],
            )

        # Si el texto es explícitamente fuera de dominio, no resolver coincidencias de productos
        if cls._es_fuera_de_dominio(texto_clean):
            return None, None, []

        # 0. Resolución de candidatos previos en contexto (ej. "el primero", "el segundo", "opción 1", "el último", código, nombre)
        if candidatos_previos:
            if re.search(r"\b(primer[oa]|opci[oó]n\s+1|el\s+1|^1$)\b", texto_clean) and len(candidatos_previos) >= 1:
                cand = qs.filter(id=candidatos_previos[0]["id"]).first()
                if cand:
                    return cand, None, []
            elif re.search(r"\b(segund[oa]|opci[oó]n\s+2|el\s+2|^2$)\b", texto_clean) and len(candidatos_previos) >= 2:
                cand = qs.filter(id=candidatos_previos[1]["id"]).first()
                if cand:
                    return cand, None, []
            elif re.search(r"\b(tercer[oa]|opci[oó]n\s+3|el\s+3|^3$)\b", texto_clean) and len(candidatos_previos) >= 3:
                cand = qs.filter(id=candidatos_previos[2]["id"]).first()
                if cand:
                    return cand, None, []
            elif re.search(r"\b([uú]ltim[oa])\b", texto_clean) and len(candidatos_previos) >= 1:
                cand = qs.filter(id=candidatos_previos[-1]["id"]).first()
                if cand:
                    return cand, None, []
            elif re.search(r"\b(de\s+los\s+tres|de\s+todos\s+ellos|de\s+todos|los\s+tres|todos\s+ellos|mu[eé]strame\s+los\s+tres|dime\s+de\s+los\s+tres)\b", texto_clean):
                return None, None, list(candidatos_previos)
            else:
                # Búsqueda por código o nombre directo de los candidatos previos
                for c in candidatos_previos:
                    cod_c = str(c.get("codigo", "")).lower()
                    if cod_c and (cod_c in texto_clean or cod_c.replace("-", "") in texto_clean):
                        cand = qs.filter(id=c["id"]).first()
                        if cand:
                            return cand, None, []
                    nom_c = _quitar_tildes(str(c.get("nombre", "")).lower())
                    toks_nom = [w for w in nom_c.split() if len(w) > 3]
                    if nom_c in texto_clean or (toks_nom and all(tok in texto_clean for tok in toks_nom)):
                        cand = qs.filter(id=c["id"]).first()
                        if cand:
                            return cand, None, []

        # 1. Búsqueda exacta de código por token (prioridad máxima de código exacto)
        palabras_texto = [w.strip() for w in re.split(r"[\s,;¿?¡!]+", texto_siglas) if w.strip()]
        for w in palabras_texto:
            w_code = w.upper().replace("_", "-")
            if "-" in w_code or any(c.isdigit() for c in w_code):
                p_exacto_code = qs.filter(codigo_interno__iexact=w_code).first()
                if p_exacto_code:
                    partes = w_code.split("-")
                    if len(partes) >= 2 and partes[-1].isdigit():
                        num_val = int(partes[-1])
                        candidatos_num = [
                            p for p in qs.filter(codigo_interno__istartswith=f"{partes[0]}-")
                            if p.codigo_interno.split("-")[-1].isdigit() and int(p.codigo_interno.split("-")[-1]) == num_val
                        ]
                        if len(candidatos_num) > 1 and len(partes[-1]) < 4:
                            pass
                        else:
                            return p_exacto_code, None, []
                    else:
                        return p_exacto_code, None, []

        # 1-bis. Búsqueda de patrón de código en el texto (ej. "CAB-0001", "CAB-001", "HER-1")
        match_code = re.search(r"\b([A-Za-z]{2,6})[-_\s]?(\d{1,4})\b", texto_siglas)
        if match_code:
            prefijo = match_code.group(1).upper()
            num_str = match_code.group(2)
            num_int = int(num_str)

            # Si el prefijo absorbió fonéticamente la preposición 'DE' (ej. 'DEBAT' -> 'BAT' o 'DEBATE' -> 'BAT')
            if prefijo.startswith("DE") and len(prefijo) >= 4 and not qs.filter(codigo_interno__istartswith=f"{prefijo}-").exists():
                prefijo_sin_de = prefijo[2:]
                if qs.filter(codigo_interno__istartswith=f"{prefijo_sin_de}-").exists():
                    prefijo = prefijo_sin_de
                elif prefijo_sin_de.endswith("E") and qs.filter(codigo_interno__istartswith=f"{prefijo_sin_de[:-1]}-").exists():
                    # Para colisiones fonéticas contextuales como 'DEBATE 002' -> prefijo 'BAT'
                    # Proteger sustantivos comunes precedidos por artículos/determinantes ('el debate', 'un debate')
                    texto_ant = texto_siglas[:match_code.start()].lower().strip()
                    if not re.search(r"\b(el|un|este|ese|aquel|al|del)\s*$", texto_ant):
                        # Evitar dígitos aislados sin formato de código a menos que coincidan exactamente
                        if num_str.startswith("0") or len(num_str) >= 3 or qs.filter(codigo_interno__iexact=f"{prefijo_sin_de[:-1]}-{num_str}").exists():
                            prefijo = prefijo_sin_de[:-1]

            codigo_formateado = f"{prefijo}-{num_str}"
            
            # Buscar todos los productos del mismo prefijo cuyo sufijo termine o coincida numéricamente
            coincidencias = []
            for p in qs.filter(codigo_interno__istartswith=f"{prefijo}-"):
                try:
                    partes = p.codigo_interno.split("-")
                    if len(partes) >= 2 and int(partes[1]) == num_int:
                        coincidencias.append(p)
                except (ValueError, IndexError):
                    continue

            # Regla 5: Si hay múltiples coincidencias numéricas (ej. CAB-001 y CAB-0001), desambiguar
            if len(coincidencias) > 1:
                coincidencias_str = [p for p in coincidencias if p.codigo_interno.upper() == codigo_formateado.upper()]
                if len(coincidencias_str) == 1 and len(num_str) >= 4:
                    return coincidencias_str[0], None, []
                
                opciones = ", ".join(f"{p.codigo_interno} ({p.nombre})" for p in coincidencias)
                candidatos_lista = [
                    {"id": p.id, "nombre": p.nombre, "codigo": p.codigo_interno}
                    for p in coincidencias[:3]
                ]
                return (
                    None,
                    f"Encontré varias coincidencias para el código '{match_code.group(0)}': {opciones}. "
                    "¿Cuál de ellos deseas consultar?",
                    candidatos_lista,
                )
            if len(coincidencias) == 1:
                return coincidencias[0], None, []

        # 2. Búsqueda por coincidencia de nombre exacto delimitado por palabras
        coincidencias_nombre_exacto = []
        for p in qs:
            p_nom = _quitar_tildes(p.nombre.lower().strip())
            if re.search(r"\b" + re.escape(p_nom) + r"\b", texto_clean):
                coincidencias_nombre_exacto.append(p)

        inactivas_exactas = []
        if coincidencias_nombre_exacto:
            # PRIORIZAR PRODUCTOS ACTIVOS: Si existen coincidencias activas, descartar inactivos
            activas = [p for p in coincidencias_nombre_exacto if p.activo and not p.archivado]
            if activas:
                coincidencias_nombre_exacto = activas
                coincidencias_nombre_exacto.sort(key=lambda p: len(p.nombre), reverse=True)
                mejor_len = len(coincidencias_nombre_exacto[0].nombre)
                mejores = [p for p in coincidencias_nombre_exacto if len(p.nombre) == mejor_len]
                if len(mejores) == 1:
                    return mejores[0], None, []
                elif len(mejores) > 1:
                    opciones = ", ".join(f"{p.nombre} ({p.codigo_interno})" for p in mejores[:3])
                    candidatos_lista = [
                        {"id": p.id, "nombre": p.nombre, "codigo": p.codigo_interno}
                        for p in mejores[:3]
                    ]
                    return (
                        None,
                        f"Encontré varios productos con nombre similar: {opciones}. ¿Cuál de ellos deseas consultar?",
                        candidatos_lista,
                    )
            else:
                # Si todas las coincidencias exactas son inactivas, no bloquear la búsqueda de productos
                # activos en el paso 3 (n-gramas). Guardar como fallback solo si no hay productos activos.
                inactivas_exactas = list(coincidencias_nombre_exacto)

        # 3. N-gramas y puntuación composicional con desambiguación y morfología
        tokens = [w for w in texto_clean.split() if w not in cls.PALABRAS_PARADA and not w.isdigit()]
        if not tokens:
            if inactivas_exactas:
                inactivas_exactas.sort(key=lambda p: len(p.nombre), reverse=True)
                mejor_len = len(inactivas_exactas[0].nombre)
                mejores = [p for p in inactivas_exactas if len(p.nombre) == mejor_len]
                if len(mejores) == 1:
                    return mejores[0], None, []
                opciones = ", ".join(f"{p.nombre} ({p.codigo_interno})" for p in mejores[:3])
                candidatos_lista = [{"id": p.id, "nombre": p.nombre, "codigo": p.codigo_interno} for p in mejores[:3]]
                return None, f"Encontré varios productos con nombre similar: {opciones}. ¿Cuál de ellos deseas consultar?", candidatos_lista
            return None, None, []

        def _variantes(w: str) -> set[str]:
            v = {w}
            if len(w) >= 5 and w.endswith("es"):
                v.add(w[:-2])
            if len(w) >= 4 and w.endswith("s"):
                v.add(w[:-1])
            return v

        prod_data = []
        for p in qs:
            p_nom = _quitar_tildes(p.nombre.lower())
            p_toks = set(re.findall(r"\b[A-Za-z0-9ñ]{2,}\b", p_nom))
            prod_data.append((p, p_nom, p_toks))

        for n in range(len(tokens), 0, -1):
            candidatos_n_grama = []
            for i in range(len(tokens) - n + 1):
                gram_tokens = tokens[i : i + n]
                if not any(len(t) >= 3 for t in gram_tokens):
                    continue

                for p, p_nom, p_toks in prod_data:
                    coincide_gram = True
                    for gt in gram_tokens:
                        if not (_variantes(gt) & p_toks):
                            coincide_gram = False
                            break
                    if coincide_gram:
                        tokens_adicionales = sum(1 for t in tokens if (_variantes(t) & p_toks))
                        candidatos_n_grama.append((n, tokens_adicionales, len(p_toks), p))

            if candidatos_n_grama:
                mejores_por_prod = {}
                for n_gram_len, tok_add, p_toks_len, p in candidatos_n_grama:
                    score = (n_gram_len * 10) + (tok_add * 2)
                    if p.activo and not p.archivado:
                        score += 10
                    if p.id not in mejores_por_prod or score > mejores_por_prod[p.id][0]:
                        mejores_por_prod[p.id] = (score, p)

                lista_ordenada = sorted(mejores_por_prod.values(), key=lambda x: x[0], reverse=True)
                # Priorizar productos activos: si hay activos, descartar inactivos
                activos_ordenados = [(s, p) for s, p in lista_ordenada if p.activo and not p.archivado]
                if activos_ordenados:
                    lista_ordenada = activos_ordenados

                top_score, top_p = lista_ordenada[0]

                candidatos_cercanos = [p for s, p in lista_ordenada if (top_score - s) <= 2]

                if len(candidatos_cercanos) == 1:
                    return top_p, None, []

                cands = [{"id": p.id, "nombre": p.nombre, "codigo": p.codigo_interno} for p in candidatos_cercanos[:3]]
                opciones = ", ".join(f"{p.nombre} ({p.codigo_interno})" for p in candidatos_cercanos[:3])
                return None, f"Encontré varias coincidencias para tu consulta: {opciones}. ¿Cuál de ellos deseas consultar?", cands

        if inactivas_exactas:
            inactivas_exactas.sort(key=lambda p: len(p.nombre), reverse=True)
            mejor_len = len(inactivas_exactas[0].nombre)
            mejores = [p for p in inactivas_exactas if len(p.nombre) == mejor_len]
            if len(mejores) == 1:
                return mejores[0], None, []
            opciones = ", ".join(f"{p.nombre} ({p.codigo_interno})" for p in mejores[:3])
            candidatos_lista = [{"id": p.id, "nombre": p.nombre, "codigo": p.codigo_interno} for p in mejores[:3]]
            return None, f"Encontré varios productos con nombre similar: {opciones}. ¿Cuál de ellos deseas consultar?", candidatos_lista

        return None, None, []
