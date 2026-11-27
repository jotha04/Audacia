"""Planificador semántico del agente HAKU Inventario.

Usa el Cerebro Central compartido (Meta-Llama-3.1-8B vía
``cerebro_audacia.conexion``) para traducir el mensaje a un plan estructurado;
solo pide una reparación breve si ese plan contradice las entidades actuales.
El plan describe QUÉ consultar, nunca la respuesta final.

No contiene catálogos de frases: la comprensión lingüística es del LLM.
El código solo valida el contrato y descarta valores fuera de esquema.
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .herramientas import (
    CONDICIONES_VALIDAS,
    FiltrosConsulta,
    _PALABRAS_VACIAS,
    _raiz,
    _sinonimos_de,
    extraer_codigos_de_texto,
    normalizar,
    normalizar_codigo_producto,
    preprocesar_texto_inventario,
    tokens_significativos,
)
from .contrato import PROMPT_SISTEMA_CANONICO

logger = logging.getLogger("integraciones.haku.agente")

TIPOS_VALIDOS = {"consulta", "operacion", "capacidades", "social", "fuera_dominio", "aclaracion"}
OPERACIONES_VALIDAS = {"listar", "contar", "sumar_unidades", "detalle", "comparar", "explicar_estado", "mayor_stock", "menor_stock"}
REFERENCIAS_VALIDAS = {"ninguna", "foco", "resultados_previos"}
_PALABRAS_NO_FILTRO_GLOBAL = frozenset({
    "podemos", "puede", "pueden", "despachar", "despacho", "entregar", "entrega",
    "aparecen", "aparece", "total", "totales", "articulos", "articulo", "productos",
    "producto", "referencias", "referencia", "items", "item", "material", "materiales",
    "insumos", "insumo", "inventario", "catalogo", "sistema", "todo", "toda", "todos",
    "todas", "general", "completo", "completa", "unidades", "unidad", "disponible",
    "disponibles", "stock", "existencia", "existencias", "el", "la", "los", "las",
    "en", "de", "del", "un", "una", "nuestro", "nuestra",
    "algo", "algun", "alguna", "alguno", "algunos", "algunas",
    "cosa", "cosas", "nada", "vaina", "vainas",
})
CAMPOS_VALIDOS = {
    "almacen", "ubicacion", "stock", "stock_minimo", "categoria", "codigo",
    "codigo_barras", "nombre", "prestamos", "movimientos", "ficha", "empaque", "es_prestable",
}
MEDIDAS_SOLICITADAS_VALIDAS = {
    "disponible", "pendiente_devolucion", "evento", "ambos", "producto", "atributo", "ninguna",
}
ALCANCES_ENTIDAD_VALIDOS = {"producto", "coleccion", "global", "indefinido"}
LIMITE_TOKENS_PLAN_DEFAULT = 384
ACCIONES_VALIDAS = {
    "prestamo", "devolucion", "entrada", "salida", "traslado",
    "cambiar_stock_minimo", "editar_producto", "inhabilitar", "habilitar",
    "crear_producto",
}
ACCIONES_DE_FORMULARIO = {"editar_producto", "crear_producto"}
EJEMPLO_STOCK_INDIVIDUAL = (
    'Elige sumar_unidades cuando se solicite un agregado cuantitativo: stock+disponible suma '
    'existencias; prestamos+pendiente_devolucion suma cantidad_pendiente_devolucion de todos '
    'los préstamos activos dentro del filtro, incluso para un producto. Usa detalle+prestamos '
    'solo cuando se pidan responsables o datos de los préstamos individuales. '
    'El saldo actual es una magnitud agregada de unidades por devolver; que el producto sea '
    'singular no lo convierte en detalle. '
    'Calibración semántica: la suma del saldo de un producto produce '
    '{"tipo":"consulta","operacion":"sumar_unidades","campos":["prestamos"],'
    '"medida_solicitada":"pendiente_devolucion"}; identificar quién tiene los préstamos '
    'produce detalle con campos=["prestamos"]. '
    'Ejemplo de stock: {"tipo":"consulta","operacion":"sumar_unidades","producto":"CODIGO",'
    '"campos":["stock"],"medida_solicitada":"disponible"}.'
)
INSTRUCCION_CAMPOS_COMPUESTOS = (
    "Una consulta que pide atributos de una entidad usa operacion=detalle y enumera todos "
    "los atributos solicitados en campos. sumar_unidades agrega la medida cuantitativa "
    "indicada por campos y medida_solicitada: stock+disponible suma existencias físicas; "
    "prestamos+pendiente_devolucion suma los saldos pendientes de los préstamos activos "
    "que cumplen los filtros. detalle de préstamos enumera responsables/registros; no lo "
    "uses para sustituir un total o saldo agregado. El stock actual y el stock mínimo son "
    "atributos distintos. `campos` contiene cada valor que se pide mostrar, no solo la primera magnitud de la oración. "
    "Si una consulta de una entidad pide el saldo actual y además pregunta por el nivel, límite o cantidad configurada "
    "que sirve de referencia para reponerla, incluye stock y stock_minimo y usa detalle; no reduzcas la solicitud a "
    "sumar_unidades. Por ejemplo, pedir cuánto queda y cuál es el nivel fijado para decidir una reposición solicita "
    "ambos atributos. Una mención del mínimo como referencia solicitada es una columna de salida; solo es filtro si "
    "el mensaje pide seleccionar registros según su relación actual con ese umbral. No conviertas una pregunta por el "
    "valor configurado en un filtro de productos."
)

_PATRONES_CONDICION_STOCK = {
    "bajo_minimo": (
        r"\b(?:"
        r"(?:por\s+debajo\s+de(?:l)?|bajo|inferior(?:es)?\s+a(?:l)?|menor(?:es)?\s+(?:a(?:l)?|que))\s+"
        r"(?:(?:su|el)\s+)?(?:(?:stock|umbral|nivel|l[ií]mite)\s+)?(?:m[ií]nimo|de\s+reposici[oó]n)(?:\s+configurado)?|"
        r"(?:con\s+|tienen?\s+|en\s+)?(?:bajo\s+stock|stock\s+bajo)|"
        r"no\s+(?:alcanza[n]?|llega[n]?|supera[n]?|cumple[n]?)\s+(?:a(?:l)?\s+)?(?:(?:su|el)\s+)?(?:stock\s+)?(?:m[ií]nimo|umbral)(?:\s+configurado)?|"
        r"(?:(?:est[aá][n]?|anda[n]?|qued(?:o|aron))\s+)?cort[oa]s?\s+(?:de\s+(?:stock|existencias?|unidades?)|frente\s+a(?:l)?\s+(?:(?:su|el)\s+)?(?:stock\s+)?m[ií]nimo)|"
        r"(?:stock|existencias?)\s+insuficientes?|"
        r"pocas?\s+unidades?\s+para\s+(?:lo\s+que\s+exige\s+)?(?:(?:su|el)\s+)?(?:stock\s+)?(?:m[ií]nimo|umbral)"
        r")\b"
    ),
    "en_minimo": r"\b(?:justo\s+en|exactamente\s+en|en|igual(?:es)?\s+a(?:l)?)\s+(?:(?:su|el)\s+)?(?:stock\s+)?m[ií]nimo\b",
    "cerca_minimo": r"\b(?:cerca\s+de(?:l)?|proxim[oa]s?\s+a(?:l)?)\s+(?:(?:su|el)\s+)?(?:stock\s+)?m[ií]nimo\b",
    "minimo_cero": r"\b(?:stock\s+)?m[ií]nimo\s+(?:configurado\s+)?(?:(?:igual\s+a|es|en)\s+)?(?:0|cero)\b",
    "agotado": (
        r"(?<!\bno\s)(?<!\bni\s)(?<!\bsin\sestar\s)\b(?:agotad[oa]s?|sin\s+(?:stock|existencias|unidades)|"
        r"no\s+(?:tienen|hay)\s+(?:stock|existencias|unidades)|"
        r"cero\s+(?:unidades|existencias)|"
        r"(?:stock|existencias?|unidades?)\s+(?:(?:en|igual\s+a|es)\s+)?(?:nulos?|nulas?|cero|0)|"
        r"se\s+(?:nos\s+)?(?:acab[oó]|acabaron|termin[oó]|terminaron|gast[oó]|gastaron)|"
        r"ya\s+no\s+(?:nos\s+)?quedan?|qued(?:o|aron)\s+en\s+cero)\b"
    ),
}

_PATRON_REPOSICION_EXPLICITA = re.compile(
    r"\b(?:"
    r"(?:necesita[n]?|requiere[n]?|hace[n]?\s+falta|pendientes?\s+de|urgentes?\s+de)\s+(?:reposici[oó]n|reponer|reabastec\w*)|"
    r"(?:deber[ií]amos|hay\s+que|toca)\s+(?:reponer|reabastecer|volver\s+a\s+abastecer)|"
    r"para\s+(?:reposici[oó]n|reabastecimiento)"
    r")\b"
)


class ErrorInferenciaLLM(Exception):
    """Excepción tipada para fallos durante la inferencia con el LLM del Cerebro Central."""

    def __init__(
        self,
        estado: str,
        mensaje: str,
        codigo_http: int = 502,
        error_original: Optional[Exception] = None,
        duracion_ms: float = 0.0,
    ) -> None:
        super().__init__(mensaje)
        self.estado = estado
        self.mensaje = mensaje
        self.codigo_http = codigo_http
        self.error_original = error_original
        self.duracion_ms = duracion_ms

# Un solo contrato semántico alimenta ambas rutas para evitar divergencias.
# El suplemento conserva reglas de salida que aplican a la interfaz de inventario.
PROMPT_SISTEMA = f"""{PROMPT_SISTEMA_CANONICO}
Elegibilidad es independiente de stock y préstamos activos: filtra por `es_prestable` solo si lo solicitan; en consultas de stock o reposición déjalo null. Para un producto concreto, devuélvela como atributo, sin filtrar.
Alta al catálogo = `operacion/crear_producto` aunque falte nombre. Entrega temporal a persona = `operacion/prestamo`, no elegibilidad. Retorno o devolución de un artículo prestado por una persona («devolvió/trajo/regresó/retornó/reintegró/quiere devolver») = `operacion/devolucion`, nunca consulta de stock.
Incluye solo claves necesarias. {EJEMPLO_STOCK_INDIVIDUAL}
Listados: campos=[] salvo atributos pedidos. No pidas ficha completa. Aclaración solo con intención de inventario clara. Para seleccionar registros por condición, usa listar + alcance_entidad=coleccion; condición va en filtros.condicion.
Cada consulta devuelve relacion_stock=sin_condicion si no hay predicado; si existe, clasifícalo y cópialo en filtros.condicion. Condición colectiva produce lista o conteo, no suma. atributos_stock_solicitados incluye solo stock/stock_minimo pedidos como salida, no el umbral usado para filtrar. filtro_texto_es_entidad=true solo si filtros.texto identifica una entidad real.
Préstamo original=>evento. Para un producto, cantidad actual=stock; si también piden el mínimo configurado, devuelve ambos con detalle.
Estado: bajo_minimo=disponible positivo menor al mínimo; en_minimo=igual; cerca_minimo=por encima dentro del margen; agotado=cero; `requiere_reposicion`=agotados+bajo_minimo. Reposición/reabastecimiento es listado requiere_reposicion; no aclares ni confundas con préstamos. Interpreta el mensaje completo. Tono formal; solo JSON."""

PROMPT_VERIFICADOR_INTENCION_OPERATIVA = """Clasifica la intención principal y extrae solo datos expresados. Devuelve exclusivamente el JSON del esquema. `clase` es `operacion` para un cambio solicitado, `formulario` para crear o editar registros, `consulta` para pedir información y `conversacion` para lo demás.
Solicitar o pedir un préstamo para recibir o llevar temporalmente un artículo significa que se pide la operación `prestamo`, aunque se formule con cortesía, como pregunta o como deseo. También lo es dejar un artículo bajo custodia temporal de otra persona. Preguntar en general qué productos se pueden prestar, qué política aplica o cómo funciona el préstamo es `consulta`.
`prestamo` entrega existencias a una persona para uso temporal; `devolucion` registra que un artículo prestado vuelve al inventario; `entrada` incorpora unidades; `salida` las retira sin préstamo; `traslado` cambia almacén o ubicación; `cambiar_stock_minimo` ajusta el umbral; `habilitar` e `inhabilitar` cambian el estado; `crear_producto` agrega un registro nuevo; `editar_producto` cambia uno existente. Si la persona pide que le devuelvan un artículo a ella, no lo clasifiques como retorno al inventario.
Si `clase` es `formulario`, `accion` debe ser `crear_producto` o `editar_producto`, incluso si falta el nombre. Una solicitud de incorporar un producto al catálogo sigue siendo `crear_producto` aunque use un verbo equivalente a agregar o dar de alta y no incluya todavía nombre; la intención sigue siendo operativa aunque falten campos. No la conviertas en consulta ni fuera de dominio y no ejecutes nada sin confirmación.
`producto` debe ser una entidad concreta; artículo, herramienta, material y otros nombres genéricos no identifican un producto y deben dejar `producto` nulo. `destinatario` es una persona; `destino` es almacén o ubicación. No inventes categorías, almacenes, ubicaciones, cantidades ni destinatarios. Deja los campos no expresados en `null`."""

ESQUEMA_VERIFICADOR_INTENCION_OPERATIVA = {
    "type": "object",
    "properties": {
        "clase": {"type": "string", "enum": ["operacion", "formulario", "consulta", "conversacion"]},
        "accion": {
            "type": ["string", "null"],
            "enum": ["prestamo", "devolucion", "entrada", "salida", "traslado", "cambiar_stock_minimo", "inhabilitar", "habilitar", "crear_producto", "editar_producto", None],
        },
        "producto": {"type": ["string", "null"]},
        "cantidad": {"type": ["integer", "null"]},
        "destinatario": {"type": ["string", "null"]},
        "almacen": {"type": ["string", "null"]},
        "categoria": {"type": ["string", "null"]},
        "ubicacion": {"type": ["string", "null"]},
        "destino": {"type": ["string", "null"]},
        "stock_inicial": {"type": ["integer", "null"]},
        "stock_minimo": {"type": ["integer", "null"]},
    },
    "required": [
        "clase", "accion", "producto", "cantidad", "destinatario", "almacen",
        "categoria", "ubicacion", "destino", "stock_inicial", "stock_minimo",
    ],
    "additionalProperties": False,
}


# Micro-caché acotado con TTL para evitar inferencias repetidas en consultas idénticas sobre el mismo contexto.
_CACHE_PLANES: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL_SEGUNDOS = 300.0
_CACHE_MAX_ENTRADAS = 64

PROMPT_CLASIFICADOR_RELACION_STOCK = """Interpreta la intención completa del mensaje y devuelve solo JSON con `r,m,x,e,a,c`.
La elegibilidad o consumibilidad de un producto es una propiedad del catálogo, no una relación de existencias; ante esa solicitud usa `r=ninguna` y deja que el plan trate la propiedad del producto.
Filtrar por una propiedad del catálogo no solicita mostrar stock; usa `a=[]` salvo que también pidan explícitamente cantidades o atributos.
Clasifica el predicado que la persona pide aplicar a los registros; no exijas que lo afirme como una oración declarativa ni que use las palabras “stock” o “disponible”.
Una condición propuesta por el plan preliminar no es evidencia de que la persona la haya pedido. Si el mensaje solicita la cantidad o los atributos de un producto identificado y no expresa una relación o criterio que limite la búsqueda, usa `ninguna`; no conviertas una pregunta normal por sus existencias en una búsqueda por estado.
`r` es una relación: `aprox`, `igual`, `bajo`, `agotado`, `min0`, `reponer`, `suf`, `stock`, `ninguna` o `ambigua`.
Usa estos significados, sin intercambiar cantidades: disponible=mínimo→`igual`; disponible>0 y <mínimo→`bajo`; disponible=0→`agotado`; mínimo configurado=0→`min0`. `Igual` significa el mismo valor exacto, aunque la persona lo formule coloquialmente; `aprox` significa estar por encima y acercándose, no estar ya en el mínimo. La ausencia o agotamiento actual de unidades significa `agotado`; no infieras préstamos sin custodia temporal o devolución pendiente. La insuficiencia frente al umbral es `bajo` incluso si se pregunta por referencias; disponible=0 no es `bajo`. `min0` exige que el propio umbral sea cero. Si pregunta qué registros necesitan reposición/reabastecimiento o deben recibir más unidades, usa `reponer`: es una selección colectiva de agotados y bajo mínimo, no una pregunta de cuánto stock o préstamo quiere consultar. Una petición de listado de esa condición no es ambigua por no indicar una magnitud. Cercanía o riesgo futuro de cruzar el mínimo usa `aprox` salvo déficit presente. `suf` indica margen amplio, `stock` positivo sin otra condición, `ninguna` ningún filtro de relación.
`m` = margen `cerca`, `amplio` o `sin_dato`. Un margen restante pequeño antes de alcanzar o cruzar el umbral describe proximidad desde arriba aunque no se dé una cantidad ni se diga literalmente «por encima»: fija `m=cerca` y clasifica `aprox`. `bajo` exige que la persona solicite el estado presente de estar por debajo; no infieras un déficit actual a partir de una trayectoria futura o de una holgura pequeña. `x` = disponibilidad que expresa el mensaje: `cero`, `positivo` o `sin_dato`; nunca la infieras del umbral ni del catálogo. Ningún sustantivo genérico es filtro de producto.
`e` = `true` si el mensaje identifica una entidad del inventario, por nombre/código o por una descripción indirecta; `false` para una condición, cuantificador o marco sin entidad.
`a` = TODOS los atributos de salida solicitados, subconjunto de `stock` y `stock_minimo`; un listado condicionado usa `[]` salvo que también pidan mostrar cantidades. Distingue valores pedidos de criterios de selección. Para una entidad, pedir “qué cantidad queda y cuál es su mínimo” o “las existencias actuales y el umbral de reposición configurado” solicita `a=["stock","stock_minimo"]`, `r="ninguna"`: el segundo valor es una respuesta explícita, no contexto ni filtro. Pedir solo el saldo actual solicita `stock`. En cambio, preguntar cuáles registros están por debajo, iguales o cerca del umbral pide una condición de selección, no el campo `stock_minimo`. No infieras atributos desde el plan preliminar.
`c` = un código exacto de la lista de candidatos del catálogo cuando la petición identifica un producto de forma única; usa `ninguno` si la petición es global, describe una colección o los datos no permiten distinguir un candidato. Una descripción indirecta también puede identificar un producto: compara todas sus propiedades mencionadas con los nombres y atributos de todos los candidatos, y descarta los que contradigan alguna propiedad. La entidad propuesta por el plan es una hipótesis útil, no una prueba; consérvala si la descripción la confirma y los candidatos similares no satisfacen el mensaje. No exijas que la persona repita literalmente el nombre, no selecciones por una palabra compartida ni por aparecer primero, y no inventes códigos o propiedades."""

PROMPT_CLASIFICADOR_RELACION_STOCK_COLECTIVO = """Clasifique la condición que la persona pide consultar en el inventario. Interprete el mensaje completo y cualquier forma natural de expresarlo; no dependa de palabras exactas ni del plan preliminar. Devuelva solo JSON con `r,m,x,a`.
La elegibilidad o consumibilidad de un producto es una propiedad del catálogo, no una relación de existencias; ante esa solicitud use `r=ninguna`.
Un filtro de elegibilidad no pide mostrar el stock; use `a=[]` salvo que se solicite explícitamente una cantidad o atributo.
`r`: `aprox`, `igual`, `bajo`, `agotado`, `min0`, `reponer`, `suf`, `stock`, `ninguna` o `ambigua`. Decida por el predicado concreto que pide consultar, no por una categoría más amplia que lo incluya: `bajo` = existencia actual positiva menor que el mínimo; `agotado` = existencia actual cero; `igual` = existencia igual al mínimo; `min0` = mínimo configurado igual a cero; `aprox` = existencia por encima del mínimo y dentro del margen de riesgo; `suf` = supera ese margen. Una cantidad positiva insuficiente sigue siendo `bajo`, nunca `agotado`. Use `reponer` solo cuando la solicitud sea identificar lo que hace falta reabastecer o reponer; no lo use como sustituto de una condición más concreta que la persona pidió consultar. Riesgo futuro desde arriba es `aprox`, no `bajo`.
`m`: margen de holgura respecto al mínimo. Solo describa cercanía o margen amplio cuando el mensaje sitúe la existencia por encima del mínimo (`aprox` o `suf`); si ya es menor, igual, cero, se habla de reposición u otra intención, use `sin_dato`. Si el mensaje afirma insuficiencia actual frente al umbral, use `bajo` aunque no formule una desigualdad matemática; use `aprox` solo si la existencia sigue por encima y expresa riesgo futuro.
`x`: `cero`, `positivo` o `sin_dato`, según lo que el mensaje afirma sobre la existencia; el criterio de comparación con el mínimo no informa por sí mismo la cantidad. `a` = atributos que la persona pide incluir en la respuesta: `stock`, `stock_minimo` o ambos. El mínimo mencionado solo como criterio no es un campo solicitado; si no pide mostrar cantidades o atributos, use `a=[]`. No complete datos ausentes ni infiera un campo desde el plan."""

PROMPT_CLASIFICADOR_LISTADO_GLOBAL = """Classify the user's task as well as its inventory scope. `operacion_inventario` means the person wants Hacu to create or change an inventory record or move custody; return the matching `accion_solicitada` even when required details are missing. Adding a new catalog record is `crear_producto`. For loans, choose exactly one `clasificacion_prestamo`: `operacion` when the user wants a handoff to happen, `elegibilidad` when they ask which records may be lent, `saldo_pendiente` for units still due back, `prestamos_activos` for records already out, or `no_aplica`. A planned handoff is an operation; describing an existing loan or asking whether an item qualifies is a query. Example: wanting to leave an item on loan => `operacion`; asking which items can be loaned => `elegibilidad`. Set `accion_solicitada=prestamo` only with `clasificacion_prestamo=operacion`; otherwise it must be null. Requests to create content unrelated to inventory remain outside Hacu. For every other operation, return its action; for other intents set `accion_solicitada=null`.
A query asking which current catalog records satisfy a product property such as loan eligibility is an inventory-record request, not a question about Hacu capabilities. Set `propiedad_prestable_solicitada=true` only when the message asks whether records are eligible/returnable for loan or are consumable; otherwise set it false and `es_prestable=null`. When requested, set `es_prestable=true` for eligible records and `false` for consumables. Stock level, depletion, replenishment, and loan eligibility are independent properties; never infer one from another, even if the preliminary plan added a property filter. Treat a request to identify records by an attribute as a property filter even if the preliminary plan interpreted the adjective or noun as a product name. A category and a product property can use similar words: select a category only when the user asks for that stored category as a group, using live catalog evidence; do not infer a category from partial word overlap when the sentence asks which records have a property. Do not infer eligibility from current custody: records currently on loan or awaiting return are active-loan records, so set `es_prestable=null`, `propiedad_prestable_solicitada=false`, and `prestamos_activos=true`. Set `pide_saldo_pendiente=true` only if the user asks for the number/total of units still due back; identifying active-loan records alone is not a quantity request. A single product's requested eligibility is an attribute, not this collection filter. If the plan has a text filter, `filtro_texto_es_entidad` must be true only when it names a real requested product/category; set false when it repeats a predicate or task wording, and never return null for a nonempty filter. Examples: asking which products can be loaned selects eligible products; asking which are currently loaned selects active-loan records; asking how many units remain due requests a balance; asking which cannot be loaned selects consumables.
When both loan eligibility and present ability to hand items out are requested, require `es_prestable=true` and positive current stock (`relacion_stock=con_stock`); do not substitute `suficiente`, since that is a comparison with the stock minimum and is unrelated to whether an item is lendable. A temporal cue like “now” does not change the eligibility property or make a low-minimum predicate.
Clasifica el mensaje original para distinguir una solicitud global de inventario, una consulta sobre una entidad concreta, una pregunta por las funciones del agente, una interacción social breve y una pregunta externa. El plan preliminar puede omitir la entidad mencionada o incluir filtros de alcance genéricos y columnas no solicitadas; decide solo por la intención del mensaje. Devuelve únicamente JSON con `intencion`: `capacidades`, `social`, `fuera_dominio`, `consulta_inventario`, `listar_catalogo`, `listar_disponibles`, `consulta_entidad` o `ambigua`, y `relacion_stock`: `aproximacion`, `igualdad_actual`, `inferior_actual`, `agotamiento`, `minimo_cero`, `union_reposicion`, `suficiente`, `con_stock`, `sin_condicion` o `ambigua`.
`social` cubre un saludo o una cortesía breve sobre cómo está la persona o cómo va su día, incluso con expresiones regionales, si no solicita una tarea externa ni datos de inventario.
`capacidades` significa que la persona pregunta por el tipo de información o las funciones que Hacu Inventario puede ofrecer; describe el alcance del agente y no solicita registros actuales. Esta categoría solo aplica si el objeto de la pregunta es Hacu, sus funciones o el tipo de consultas que admite. Si el usuario encarga una tarea externa concreta —por ejemplo, escribir, traducir, resumir o responder una pregunta factual— clasifícala como `fuera_dominio`, aunque el plan preliminar diga capacidades. Ante esa tensión, manda la tarea pedida, no la forma gramatical ni una clasificación previa.
Una solicitud para identificar, contar o listar registros que cumplan una condición actual es una consulta de registros aunque no nombre productos ni almacenes. No la clasifiques como capacidades por falta de una entidad concreta.
`fuera_dominio` es una pregunta factual del mundo general sin relación explícita con productos, existencias, almacenes, ubicaciones, préstamos o movimientos del inventario. Una pregunta general como cuántos elementos tiene un sistema natural o astronómico no se vuelve de inventario solo por usar «cuántos», «hay», «tiene» o «sistema».
Una petición de conocimiento externo o de generar/transformar contenido sigue fuera de dominio aunque nombre objetos que podrían existir como productos; no conviertas esos objetos en filtros ni pidas una entidad de inventario. Una petición clara sobre existencias, ubicación u otros datos de un producto desconocido sigue siendo una consulta de inventario aunque no haya coincidencia en el catálogo.
Usa las entidades vivas de catálogo y el plan preliminar que se adjunten como evidencia semántica, no como una lista cerrada. Si el plan interpreta una petición como cantidad, atributo o listado de una entidad/categoría que coincide con el catálogo actual, conserva la consulta de inventario aunque omita «inventario», «stock» o «almacén» y aunque use habla regional. Cambia a `fuera_dominio` solo cuando el propio mensaje pide una tarea externa concreta (crear, traducir, resumir o responder conocimiento general); no confundas una pregunta de cantidad sobre un producto actual con una pregunta general por no incluir una palabra de inventario.
`consulta_inventario` cubre conteos y sumas de los registros del inventario cuando no se pide enumerarlos.
`listar_catalogo` significa que pide los registros reales del inventario, incluidos los que tienen cero existencias, aunque solicite el catálogo completo.
`listar_disponibles` significa que pide explícitamente solo los registros con existencias positivas. Una negación de disponibilidad significa ausencia/agotamiento, nunca `listar_disponibles`.
`consulta_entidad` significa que solicita información de uno o varios productos, categorías, almacenes o ubicaciones identificables en el mensaje, aunque el plan preliminar haya omitido ese filtro. Conserva el plan para que el catálogo actual resuelva la entidad después.
`ambigua` significa que no queda claro si quiere conocer las funciones del agente o consultar registros. Usa solo el mensaje actual; no respondas ni inventes datos.
`relacion_stock` clasifica la condición que selecciona registros actuales, aunque no se diga literalmente «stock»: igualdad exacta con el mínimo; proximidad por encima del mínimo dentro del margen vigente; déficit positivo inferior al mínimo; agotamiento; mínimo configurado en cero; reposición como unión de agotados y bajo mínimo; suficiente; existencia positiva sin condición; o ninguna. Si preguntan qué referencias necesitan reposición, deben recibir más unidades o están en una situación equivalente, selecciona `union_reposicion`. Si solo preguntan por las funciones, por todo el catálogo o por un producto sin criterio de estado, usa `sin_condicion`. No confundas una condición de listado con una pregunta por cantidad de stock o préstamos."""

PROMPT_CLASIFICADOR_SALDO_PRESTAMOS = """Clasifique la forma de respuesta que solicita el mensaje sobre préstamos del inventario. `saldo_total` significa que pide la magnitud agregada de unidades aún pendientes en los préstamos activos que coincidan con la entidad y filtros; el total debe incluir todos los registros coincidentes. `detalle_individual` significa que necesita identificar responsables o revisar los préstamos por separado. `otra_consulta` significa que pide otra propiedad o tarea del inventario. `ambigua` significa que no se puede decidir cuál de esas salidas pidió. Use la semántica del mensaje completo, no una palabra aislada ni el plan preliminar. Devuelva solo JSON con `modo`: `saldo_total`, `detalle_individual`, `otra_consulta` o `ambigua`. No calcule cantidades ni invente entidades."""

PROMPT_CLASIFICADOR_MEDIDA_CANTIDAD = """Clasifique qué saldo pide el mensaje por la custodia física actual, no por una palabra aislada: disponible=unidades físicamente dentro del inventario y libres para retiro; pendiente_devolucion=unidades que están fuera del inventario bajo custodia temporal o responsabilidad de una persona/tercero y siguen debiendo regresar; ambos=las dos cantidades juntas. La custodia temporal fuera del inventario implica saldo pendiente aunque el mensaje no diga literalmente «préstamo» o «devolución»: preguntar cuántas unidades siguen en poder de usuarios/personas pregunta por el saldo pendiente. No clasifique como disponible una unidad que otra persona conserva fuera del almacén. Un total simple de productos, una pregunta por cuántas unidades quedan en inventario o una descripción de la referencia sin otra dimensión solicitada usa disponible por defecto. El agotamiento o la ausencia de existencias describe disponibilidad cero; no implica que algo esté prestado. Use ambos únicamente cuando la persona solicite las dos cantidades. No infiera préstamos por una cantidad pequeña, una referencia o una forma regional de hablar. Reserve ambigua para mensajes que no solicitan una cantidad actual o cuyo objeto no se puede determinar ni aplicando el valor por defecto. Use el mensaje completo, no el plan preliminar. Devuelva solo JSON con `medida`: `disponible`, `pendiente_devolucion`, `ambos` o `ambigua`. No responda ni calcule valores."""

_RELACION_A_CONDICION_STOCK = {
    "aproximacion": "cerca_minimo",
    "igualdad_actual": "en_minimo",
    "inferior_actual": "bajo_minimo",
    "agotamiento": "agotado",
    "minimo_cero": "minimo_cero",
    "union_reposicion": "requiere_reposicion",
    "suficiente": "suficiente",
    "con_stock": "con_stock",
}
RELACIONES_STOCK_VALIDAS = {
    *_RELACION_A_CONDICION_STOCK.keys(), "sin_condicion", "ambigua",
}
_CONDICIONES_STOCK_PARA_CLASIFICAR = {
    "agotado", "bajo_minimo", "en_minimo", "cerca_minimo", "minimo_cero",
    "requiere_reposicion", "suficiente", "con_stock",
}
_CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA = _CONDICIONES_STOCK_PARA_CLASIFICAR - {"con_stock"}

_ESQUEMA_PLANIFICADOR_JSON = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tipo": {"type": "string", "enum": sorted(TIPOS_VALIDOS)},
        "operacion": {"type": ["string", "null"], "enum": sorted(OPERACIONES_VALIDAS) + [None]},
        "medida_solicitada": {
            "type": "string",
            "enum": sorted(MEDIDAS_SOLICITADAS_VALIDAS),
        },
        "alcance_entidad": {
            "type": ["string", "null"],
            "enum": sorted(ALCANCES_ENTIDAD_VALIDOS) + [None],
        },
        "relacion_stock": {
            "type": ["string", "null"],
            "enum": sorted(RELACIONES_STOCK_VALIDAS) + [None],
        },
        "filtro_texto_es_entidad": {"type": ["boolean", "null"]},
        "atributos_stock_solicitados": {
            "type": "array",
            "items": {"type": "string", "enum": ["stock", "stock_minimo"]},
        },
        "filtros": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "texto": {"type": ["string", "null"]},
                "codigos": {"type": "array", "items": {"type": "string"}},
                "categoria": {"type": ["string", "null"]},
                "almacen": {"type": ["string", "null"]},
                "ubicacion": {"type": ["string", "null"]},
                "condicion": {
                    "type": ["string", "null"],
                    "enum": sorted(CONDICIONES_VALIDAS) + [None],
                },
                "umbral": {"type": ["integer", "null"]},
                "es_prestable": {"type": ["boolean", "null"]},
            },
        },
        "ubicaciones_comparadas": {"type": "array", "items": {"type": "string"}},
        "referencia": {"type": ["string", "null"], "enum": sorted(REFERENCIAS_VALIDAS) + [None]},
        "alcance": {
            "type": ["string", "null"],
            "enum": ["actual", "evento", "ambos", "indefinido", None],
        },
        "tipo_movimiento": {"type": ["string", "null"]},
        "campos": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(CAMPOS_VALIDOS)},
        },
        "accion": {"type": ["string", "null"], "enum": sorted(ACCIONES_VALIDAS) + [None]},
        "producto": {"type": ["string", "null"]},
        "cantidad": {"type": ["integer", "null"]},
        "destinatario": {"type": ["string", "null"]},
        "valor": {"type": ["integer", "null"]},
        "destino": {"type": ["string", "null"]},
        "pregunta": {"type": ["string", "null"]},
        # Ordinales are control data, never free-form prose. Constraining the
        # field keeps a malformed model completion from spending hundreds of
        # tokens on an unrelated explanation that the validator will discard.
        "orden_movimiento": {"type": ["string", "null"], "enum": ["asc", "desc", None]},
        "fecha_desde": {"type": ["string", "null"]},
        "fecha_hasta": {"type": ["string", "null"]},
    },
    "required": [
        "tipo", "medida_solicitada", "relacion_stock",
        "atributos_stock_solicitados",
    ],
}


def _construir_esquema_planificador_campos_compuestos() -> Dict[str, Any]:
    """Construye un esquema tipado que conserva consultas con varios atributos.

    La variante comparte los tipos y enumeraciones del contrato base. La única
    restricción adicional es exigir dos o más campos dentro de la rama de
    consulta; las demás clases de plan conservan sus formas válidas.
    """
    propiedades = _ESQUEMA_PLANIFICADOR_JSON["properties"]
    consulta = {
        clave: copy.deepcopy(propiedades[clave])
        for clave in (
            "tipo", "operacion", "medida_solicitada", "alcance_entidad",
            "relacion_stock", "filtro_texto_es_entidad",
            "atributos_stock_solicitados", "filtros", "ubicaciones_comparadas",
            "referencia", "alcance", "tipo_movimiento", "campos",
            "fecha_desde", "fecha_hasta",
        )
    }
    consulta["tipo"] = {"type": "string", "const": "consulta"}
    consulta["campos"]["minItems"] = 2

    conversacion = {
        "tipo": {"type": "string", "enum": ["social", "capacidades", "fuera_dominio"]},
        "medida_solicitada": copy.deepcopy(propiedades["medida_solicitada"]),
        "relacion_stock": copy.deepcopy(propiedades["relacion_stock"]),
        "filtro_texto_es_entidad": copy.deepcopy(propiedades["filtro_texto_es_entidad"]),
        "atributos_stock_solicitados": copy.deepcopy(propiedades["atributos_stock_solicitados"]),
    }
    aclaracion = {
        "tipo": {"type": "string", "const": "aclaracion"},
        "medida_solicitada": copy.deepcopy(propiedades["medida_solicitada"]),
        "pregunta": copy.deepcopy(propiedades["pregunta"]),
        "filtros": copy.deepcopy(propiedades["filtros"]),
        "referencia": copy.deepcopy(propiedades["referencia"]),
        "relacion_stock": copy.deepcopy(propiedades["relacion_stock"]),
        "filtro_texto_es_entidad": copy.deepcopy(propiedades["filtro_texto_es_entidad"]),
        "atributos_stock_solicitados": copy.deepcopy(propiedades["atributos_stock_solicitados"]),
    }
    mutacion = {
        clave: copy.deepcopy(propiedades[clave])
        for clave in (
            "tipo", "medida_solicitada", "accion", "producto", "cantidad", "destinatario",
            "valor", "destino", "pregunta", "filtros", "alcance",
            "tipo_movimiento", "relacion_stock", "filtro_texto_es_entidad",
            "atributos_stock_solicitados",
        )
    }
    mutacion["tipo"] = {"type": "string", "const": "operacion"}

    def rama(propiedades_rama: Dict[str, Any], requeridas: List[str]) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": propiedades_rama,
            "required": requeridas,
            "additionalProperties": False,
        }

    return {
        "oneOf": [
            rama(consulta, [
                "tipo", "campos", "medida_solicitada", "relacion_stock",
                "atributos_stock_solicitados",
            ]),
            rama(conversacion, ["tipo", "medida_solicitada"]),
            rama(aclaracion, ["tipo", "medida_solicitada"]),
            rama(mutacion, ["tipo", "medida_solicitada"]),
        ],
    }


_ESQUEMA_PLANIFICADOR_CAMPOS_COMPUESTOS = _construir_esquema_planificador_campos_compuestos()

_ESQUEMA_RELACION_STOCK_JSON = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "r": {
            "type": "string",
            "enum": [
                "aprox", "igual", "bajo", "agotado", "min0", "reponer",
                "suf", "stock", "ninguna", "ambigua",
            ],
        },
        "m": {
            "type": "string",
            "enum": ["cerca", "amplio", "sin_dato"],
        },
        "x": {
            "type": "string",
            "enum": ["cero", "positivo", "sin_dato"],
        },
        "e": {"type": "boolean"},
        "a": {
            "type": "array",
            "items": {"type": "string", "enum": ["stock", "stock_minimo"]},
        },
    },
    "required": [
        "r", "m", "x", "e", "a",
    ],
}

_ESQUEMA_RELACION_STOCK_COLECTIVO_JSON = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "r": copy.deepcopy(_ESQUEMA_RELACION_STOCK_JSON["properties"]["r"]),
        "m": copy.deepcopy(_ESQUEMA_RELACION_STOCK_JSON["properties"]["m"]),
        "x": copy.deepcopy(_ESQUEMA_RELACION_STOCK_JSON["properties"]["x"]),
        "a": copy.deepcopy(_ESQUEMA_RELACION_STOCK_JSON["properties"]["a"]),
    },
    "required": ["r", "m", "x", "a"],
}

_RELACION_STOCK_CORTA_A_CANONICA = {
    "aprox": "aproximacion",
    "igual": "igualdad_actual",
    "bajo": "inferior_actual",
    "agotado": "agotamiento",
    "min0": "minimo_cero",
    "reponer": "union_reposicion",
    "suf": "suficiente",
    "stock": "con_stock",
    "ninguna": "sin_condicion",
}
_MARGEN_STOCK_CORTO_A_CANONICO = {
    "cerca": "estrecho",
    "amplio": "holgado",
    "sin_dato": "no_indicado",
}

_ESQUEMA_INTENCION_LISTADO_GLOBAL_JSON = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intencion": {"type": "string", "enum": ["capacidades", "social", "fuera_dominio", "consulta_inventario", "listar_catalogo", "listar_disponibles", "consulta_entidad", "operacion_inventario", "ambigua"]},
        "accion_solicitada": {"type": ["string", "null"], "enum": sorted(ACCIONES_VALIDAS) + [None]},
        "clasificacion_prestamo": {"type": "string", "enum": ["operacion", "elegibilidad", "saldo_pendiente", "prestamos_activos", "no_aplica"]},
        "relacion_stock": {
            "type": "string",
            "enum": [
                "aproximacion", "igualdad_actual", "inferior_actual", "agotamiento",
                "minimo_cero", "union_reposicion", "suficiente", "con_stock",
                "sin_condicion", "ambigua",
            ],
        },
        "es_prestable": {"type": ["boolean", "null"]},
        "propiedad_prestable_solicitada": {"type": "boolean"},
        "filtro_texto_es_entidad": {"type": ["boolean", "null"]},
        "prestamos_activos": {"type": ["boolean", "null"]},
        "pide_saldo_pendiente": {"type": ["boolean", "null"]},
    },
    "required": ["intencion", "accion_solicitada", "clasificacion_prestamo", "relacion_stock", "es_prestable", "propiedad_prestable_solicitada", "filtro_texto_es_entidad", "prestamos_activos", "pide_saldo_pendiente"],
}

_ESQUEMA_SALDO_PRESTAMOS_JSON = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "modo": {
            "type": "string",
            "enum": ["saldo_total", "detalle_individual", "otra_consulta", "ambigua"],
        },
    },
    "required": ["modo"],
}

_ESQUEMA_MEDIDA_CANTIDAD_JSON = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "medida": {
            "type": "string",
            "enum": ["disponible", "pendiente_devolucion", "ambos", "ambigua"],
        },
    },
    "required": ["medida"],
}


@dataclass
class PlanAgente:
    tipo: str
    operacion: Optional[str] = None
    filtros: FiltrosConsulta = field(default_factory=FiltrosConsulta)
    referencia: str = "ninguna"
    alcance: Optional[str] = None
    tipo_movimiento: Optional[str] = None
    campos: List[str] = field(default_factory=list)
    medida_solicitada: Optional[str] = None
    alcance_entidad: Optional[str] = None
    accion: Optional[str] = None
    producto: Optional[str] = None
    cantidad: Optional[int] = None
    destinatario: Optional[str] = None
    valor: Optional[int] = None
    destino: Optional[str] = None
    ubicaciones_comparadas: List[str] = field(default_factory=list)
    pregunta: Optional[str] = None
    llm_ms: float = 0.0
    chars_prompt: int = 0
    tokens_estimados_prompt: int = 0
    tokens_completados: int = 0
    fuente_plan: str = "llm"
    crudo: Dict[str, Any] = field(default_factory=dict)
    relacion_stock_verificada: Optional[str] = None
    filtro_texto_es_entidad_verificado: Optional[bool] = None
    codigo_catalogo_verificado: bool = False
    llm_cola_ms: float = 0.0
    orden_movimiento: Optional[str] = None
    fecha_desde: Optional[str] = None
    fecha_hasta: Optional[str] = None
    entidad_producto_mencionada: bool = False
    descriptor_producto_grounded: bool = False
    alcance_global_confirmado: bool = False
    atributos_stock_verificados: Optional[List[str]] = None

    def resumen(self) -> Dict[str, Any]:
        """Representación segura (sin datos personales extensos) para registro e instrumentación."""
        return {
            "tipo": self.tipo,
            "operacion": self.operacion,
            "filtros": {k: v for k, v in self.filtros.__dict__.items() if v not in (None, [], "")},
            "referencia": self.referencia,
            "alcance": self.alcance,
            "tipo_movimiento": self.tipo_movimiento,
            "orden_movimiento": self.orden_movimiento,
            "fecha_desde": self.fecha_desde,
            "fecha_hasta": self.fecha_hasta,
            "campos": self.campos,
            "medida_solicitada": self.medida_solicitada,
            "alcance_entidad": self.alcance_entidad,
            "alcance_global_confirmado": self.alcance_global_confirmado,
            "accion": self.accion,
            "llm_ms": round(self.llm_ms, 1),
            "llm_cola_ms": round(self.llm_cola_ms, 1),
            "atributos_stock_verificados": self.atributos_stock_verificados,
            "chars_prompt": self.chars_prompt,
            "tokens_estimados_prompt": self.tokens_estimados_prompt,
            "tokens_completados": self.tokens_completados,
            "fuente_plan": self.fuente_plan,
            "ubicaciones_comparadas": self.ubicaciones_comparadas,
        }

    def a_plan_haku(self) -> Any:
        """Adapta el PlanAgente al contrato canónico PlanHaku (Fase II-C.1)."""
        from .contrato import adaptar_plan_agente_a_plan_haku
        return adaptar_plan_agente_a_plan_haku(self)


_ESQUEMA_REVISION_CONTEXTO = {
    "type": "object",
    "properties": {
        "intencion": {"type": "string", "enum": ["consulta_foco", "operacion_foco", "social", "fuera_dominio", "otro", "incierta"]},
        "campos": {"type": "array", "items": {"type": "string", "enum": sorted(CAMPOS_VALIDOS)}},
        "medida": {"type": "string", "enum": ["disponible", "pendiente_devolucion", "ambos", "atributo", "ninguna"]},
        "accion": {"type": ["string", "null"], "enum": sorted(ACCIONES_VALIDAS) + [None]},
        "cantidad": {"type": ["integer", "null"]},
        "valor": {"type": ["integer", "null"]},
        "destinatario": {"type": ["string", "null"]},
        "origen": {"type": ["string", "null"]},
        "destino": {"type": ["string", "null"]},
    },
    "required": ["intencion", "campos", "medida", "accion", "cantidad", "valor", "destinatario", "origen", "destino"],
    "additionalProperties": False,
}


def _revisar_social_con_foco(
    texto, contexto, cliente, plan, timeout, telemetria,
    almacenes_accesibles: Optional[List[str]] = None,
):
    """Resuelve un turno contextual con el mismo modelo y el catálogo vigente."""
    if plan.tipo not in {"social", "aclaracion"} or not re.search(
        r"(?m)^Producto en foco:\s*\S", contexto or ""
    ):
        return plan
    if plan.tipo == "aclaracion" and (plan.accion or any((
        plan.filtros.texto, plan.filtros.codigos, plan.filtros.categoria,
        plan.filtros.almacen, plan.filtros.ubicacion,
    ))):
        return plan
    instrucciones = (
        "Interpreta el mensaje en el contexto de la conversación de inventario. "
        "Clasifica consulta_foco solo si pide información sobre el producto en foco; "
        "operacion_foco solo si pide modificar ese único producto o hacer sobre él una operación; "
        "social si es saludo, agradecimiento o despedida; fuera_dominio si pregunta "
        "por un tema ajeno al inventario sin relación con el producto; otro si "
        "pide un conjunto, otro producto o una operación, "
        "e incierta si no se puede determinar. Nunca infieras una operación ni "
        "selecciones el producto en foco cuando se pida otro producto o un conjunto. "
        "Distingue una pregunta sobre saldo prestado o elegibilidad de una orden para prestar, "
        "devolver, modificar o cambiar estado. La consulta es consulta_foco; una orden es "
        "operacion_foco y requiere una accion válida del contrato. Para esta última, devuelve "
        "cantidad, valor, destinatario, origen y destino solo si se expresaron. Origen y destino "
        "deben coincidir con almacenes actuales incluidos en el contexto; nunca los inventes. "
        "Una pregunta sobre el lugar donde está o donde guardaron el producto pide "
        "su ubicación actual; una pregunta por cambios anteriores pide movimientos. "
        "Para consulta_foco devuelve los atributos solicitados en campos: ubicacion, "
        "almacen, stock, prestamos, stock_minimo, categoria, nombre, codigo, "
        "codigo_barras, movimientos, ficha, empaque o es_prestable. Para cantidades "
        "de existencias usa medida disponible; para unidades prestadas pendientes de "
        "retorno usa pendiente_devolucion y el campo prestamos: preguntar por ese "
        "saldo no solicita registrar una devolución. Para ambos usa ambos; para otros atributos "
        "usa atributo. Si no se identifica lo solicitado, usa incierta. Solo JSON."
    )
    mensajes = [{"role": "system", "content": instrucciones},
                {"role": "user", "content": f"[Contexto]\n{contexto}\n[Mensaje]\n{texto}"}]
    inicio = time.perf_counter()
    telemetria["llamadas_fisicas"] = int(telemetria.get("llamadas_fisicas", 0) or 0) + 1
    try:
        respuesta = cliente.generar(mensajes=mensajes, formato_json=True, temperatura=0.0,
            max_tokens=128, timeout=timeout, json_schema=_ESQUEMA_REVISION_CONTEXTO,
            json_schema_name="haku_revision_contexto")
        datos = getattr(respuesta, "datos_json", None)
        datos = datos if isinstance(datos, dict) else {}
        clase = datos.get("intencion")
        campos = datos.get("campos")
        medida = datos.get("medida")
        plan.llm_cola_ms += float(getattr(respuesta, "espera_cola_ms", 0.0) or 0.0)
        plan.tokens_completados += int(getattr(respuesta, "tokens_completados", 0) or 0)
    except Exception as exc:
        logger.warning("[HakuAgente] No se pudo revisar el contexto: %s", type(exc).__name__)
        datos = {}
        clase, campos, medida = "incierta", [], "ninguna"
    plan.llm_ms += (time.perf_counter() - inicio) * 1000.0
    chars_revision = sum(len(item["content"]) for item in mensajes)
    plan.chars_prompt += chars_revision
    plan.tokens_estimados_prompt += max(1, chars_revision // 4)
    foco = re.search(r"(?m)^Producto en foco:\s*(\S+)", contexto or "")
    if clase == "operacion_foco":
        codigos = set(extraer_codigos_de_texto(texto or ""))
        codigo_foco = foco.group(1).upper() if foco else ""
        accion = datos.get("accion")
        if (
            not foco
            or (codigos and codigos != {codigo_foco})
            or accion not in ACCIONES_VALIDAS
            or accion in ACCIONES_DE_FORMULARIO
        ):
            clase = "incierta"
        else:
            almacenes = list(almacenes_accesibles or [])

            def resolver_almacen(valor):
                candidato = _str(valor)
                if not candidato:
                    return None
                coincidencias = [
                    nombre for nombre in almacenes
                    if normalizar(nombre) == normalizar(candidato)
                ]
                return coincidencias[0] if len(coincidencias) == 1 else None

            def entero_no_negativo(valor, permitir_cero=False):
                if isinstance(valor, bool) or not isinstance(valor, int):
                    return None
                return valor if valor >= (0 if permitir_cero else 1) else None

            plan.tipo = "operacion"
            plan.operacion = None
            plan.accion = accion
            plan.referencia = "foco"
            plan.alcance_entidad = "producto"
            plan.filtros = FiltrosConsulta(almacen=resolver_almacen(datos.get("origen")))
            plan.campos = []
            plan.cantidad = entero_no_negativo(datos.get("cantidad"))
            plan.valor = entero_no_negativo(datos.get("valor"), permitir_cero=True)
            plan.destinatario = _str(datos.get("destinatario"))
            plan.destino = resolver_almacen(datos.get("destino"))
            plan.pregunta = None
            plan.crudo = dict(datos)
            telemetria["revision_contexto"] = "operacion_foco"
            return plan
    if clase == "consulta_foco" and foco and re.search(r"\d", texto or ""):
        codigos = set(extraer_codigos_de_texto(texto))
        if codigos and codigos != {foco.group(1).upper()}:
            clase = "incierta"
    if clase in {"social", "otro", "fuera_dominio"} and campos:
        clase = "incierta"
    if clase == "consulta_foco" and isinstance(campos, list):
        cuantitativos = set(campos).intersection({"stock", "prestamos"})
        if ((medida == "disponible" and cuantitativos == {"prestamos"})
                or (medida == "pendiente_devolucion" and cuantitativos == {"stock"})
                or (medida == "ambos" and cuantitativos != {"stock", "prestamos"})):
            clase = "incierta"
    telemetria["revision_contexto"] = clase
    if clase == "fuera_dominio":
        plan.tipo = "fuera_dominio"
        return plan
    if clase == "consulta_foco" and isinstance(campos, list) and campos and all(c in CAMPOS_VALIDOS for c in campos):
        if medida not in {"disponible", "pendiente_devolucion", "ambos", "atributo", "ninguna"}:
            clase = "incierta"
        else:
            plan.tipo = "consulta"
            plan.filtros = FiltrosConsulta()
            plan.campos = list(dict.fromkeys(campos))
            if not set(plan.campos).intersection({"stock", "prestamos"}):
                medida = "atributo"
            plan.medida_solicitada = medida
            plan.operacion = "sumar_unidades" if medida in {"disponible", "pendiente_devolucion", "ambos"} else "detalle"
            plan.referencia = "foco"
            plan.alcance_entidad = "producto"
            plan.pregunta = None
            return plan
    if clase not in {"social", "otro", "fuera_dominio"} and plan.tipo != "aclaracion":
        plan.tipo = "aclaracion"
        plan.pregunta = "¿Podría precisar qué información desea consultar?"
    return plan


def _consulta_contextual_breve_ejecutable(plan: PlanAgente) -> bool:
    """La vía breve solo ejecuta una consulta de un atributo sin ambigüedad."""
    if plan.tipo != "consulta" or plan.referencia != "foco" or len(plan.campos) != 1:
        return False
    campo = plan.campos[0]
    if campo in {"ficha", "movimientos"}:
        return False
    if campo == "stock":
        return plan.medida_solicitada == "disponible"
    if campo == "prestamos":
        return plan.medida_solicitada == "pendiente_devolucion"
    return plan.medida_solicitada == "atributo"


def _str(v: Any) -> Optional[str]:
    if v is None or isinstance(v, (bool, dict, list)):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("null", "none", "n/a", "ninguno", "ninguna") else None


def _fecha_iso_validada(valor: Any) -> Optional[str]:
    """Acepta solo fechas ISO reales para que el modelo no pueda crear rangos vagos."""
    texto = _str(valor)
    if not texto:
        return None
    try:
        fecha = datetime.date.fromisoformat(texto)
    except (TypeError, ValueError):
        return None
    return fecha.isoformat() if fecha.isoformat() == texto else None


def _fecha_referencia_local() -> str:
    """Fecha local usada por el LLM al convertir periodos relativos en límites."""
    try:
        from django.utils import timezone
        return timezone.localdate().isoformat()
    except Exception:
        return datetime.date.today().isoformat()


def normalizar_decision_intencion_operativa(
    decision: Optional[Dict[str, Any]],
    *,
    categorias_catalogo: Optional[List[str]] = None,
    almacenes_catalogo: Optional[List[str]] = None,
    ubicaciones_catalogo: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Repara discordancias entre campos estructurados con anclaje al contrato y al catálogo.

    El modelo a veces coloca un valor del enum de acción en un campo de entidad,
    o devuelve una acción válida junto con clase ``conversacion``. Solo se
    recupera esa discordancia cuando la clase no es ``consulta`` y el valor no
    corresponde a una categoría real. Las entidades espaciales se conservan
    únicamente si existen en los catálogos actuales.
    """
    if not isinstance(decision, dict):
        return None

    normalizada = copy.deepcopy(decision)
    clase = _str(normalizada.get("clase"))
    accion = _str(normalizada.get("accion"))
    categoria = _str(normalizada.get("categoria"))
    categorias = categorias_catalogo or []
    categorias_norm = {normalizar(nombre) for nombre in categorias if nombre}

    if clase == "conversacion":
        accion_desde_categoria = bool(
            accion not in ACCIONES_VALIDAS
            and categoria
            and categorias_catalogo is not None
            and normalizar(categoria) in ACCIONES_VALIDAS
            and normalizar(categoria) not in categorias_norm
        )
        if accion_desde_categoria:
            accion = normalizar(categoria)
            normalizada["categoria"] = None
    if accion in ACCIONES_VALIDAS:
        normalizada["accion"] = accion
        if clase != "consulta":
            clase_esperada = "formulario" if accion in ACCIONES_DE_FORMULARIO else "operacion"
            if clase in {"conversacion", "operacion", "formulario"}:
                normalizada["clase"] = clase_esperada

    def resolver_catalogo(valor: Any, nombres: Optional[List[str]]) -> Optional[str]:
        candidato = _str(valor)
        if not candidato or nombres is None:
            return candidato
        coincidencias = [nombre for nombre in nombres if normalizar(nombre) == normalizar(candidato)]
        return coincidencias[0] if len(coincidencias) == 1 else None

    normalizada["categoria"] = resolver_catalogo(normalizada.get("categoria"), categorias_catalogo)
    normalizada["almacen"] = resolver_catalogo(normalizada.get("almacen"), almacenes_catalogo)
    normalizada["ubicacion"] = resolver_catalogo(normalizada.get("ubicacion"), ubicaciones_catalogo)

    destino = _str(normalizada.get("destino"))
    if destino and (almacenes_catalogo is not None or ubicaciones_catalogo is not None):
        destinos_validos = list(almacenes_catalogo or []) + list(ubicaciones_catalogo or [])
        normalizada["destino"] = resolver_catalogo(destino, destinos_validos)
    else:
        normalizada["destino"] = destino

    for campo in ("producto", "destinatario"):
        normalizada[campo] = _str(normalizada.get(campo))
    return normalizada


def _resumir_respuesta_modelo(resp: Any, limite_tokens: int) -> Dict[str, Any]:
    """Resume el estado técnico de la salida sin registrar su contenido conversacional."""
    if resp is None:
        texto = ""
        datos_json = None
        finish_reason = "sin_respuesta"
        tokens = 0
    else:
        texto = getattr(resp, "texto", "") or ""
        datos_json = getattr(resp, "datos_json", None)
        finish_reason = str(getattr(resp, "finish_reason", "desconocido") or "desconocido")
        try:
            tokens = max(0, int(getattr(resp, "tokens_completados", 0) or 0))
        except (TypeError, ValueError):
            tokens = 0

    razon_normalizada = finish_reason.strip().lower()
    probable_corte_por_limite = razon_normalizada in {"length", "max_tokens", "max_token"}
    if limite_tokens > 0 and tokens >= limite_tokens:
        probable_corte_por_limite = True

    return {
        "finish_reason": finish_reason[:40],
        "tokens_completados": tokens,
        "caracteres_salida": len(texto) if isinstance(texto, str) else 0,
        "json_parseado": datos_json is not None,
        "probable_corte_por_limite": probable_corte_por_limite,
        "limite_tokens": max(0, int(limite_tokens)),
    }


def _limite_tokens_planificador() -> int:
    """Límite configurable para planes complejos con JSON completo y válido."""
    valor = os.environ.get("HAKU_LLM_PLAN_MAX_TOKENS")
    if not valor:
        return LIMITE_TOKENS_PLAN_DEFAULT
    try:
        limite = int(valor)
    except (TypeError, ValueError):
        return LIMITE_TOKENS_PLAN_DEFAULT
    return max(128, min(512, limite))


def _tipos_movimiento_validos() -> Dict[str, str]:
    """Obtiene códigos y etiquetas del esquema actual, sin duplicar su catálogo aquí."""
    try:
        from inventario.models import Movimiento
        return {str(codigo): str(etiqueta) for codigo, etiqueta in Movimiento.Tipo.choices}
    except Exception:
        return {}


def _normalizar_tipo_movimiento(valor: Any) -> Optional[str]:
    """Resuelve solo códigos o etiquetas reales definidos por el modelo de inventario."""
    texto = _str(valor)
    if not texto:
        return None
    tipos = _tipos_movimiento_validos()
    if texto in tipos:
        return texto
    texto_norm = normalizar(texto)
    return next((codigo for codigo, etiqueta in tipos.items() if normalizar(etiqueta) == texto_norm), None)


_VERBOS_EVENTO_PASADO_A_TIPO: Tuple[Tuple[str, str], ...] = (
    ("PRESTAMO", r"\b(?:se\s+prest(?:o|aron|a|an)|prestaron|presto)\b"),
    ("DEVOLUCION", r"\b(?:se\s+devolv(?:io|ieron)|devolvieron|devolvio|se\s+regres(?:o|aron)|regresaron)\b"),
    ("ENTRADA", r"\b(?:entr(?:o|aron|a|an)|ingres(?:o|aron|a|an)|lleg(?:o|aron|a|an)|se\s+recib(?:io|ieron|e|en)|recibieron)\b"),
    ("SALIDA", r"\b(?:sal(?:io|ieron|e|en|ga|gan)|se\s+despach(?:o|aron|a|an)|despacharon|se\s+entreg(?:o|aron|a|an)|entregaron|se\s+retir(?:o|aron|a|an)|retiraron)\b"),
)


def _inferir_tipo_movimiento_desde_etiqueta(
    texto: str,
    alcance: Optional[str],
    tipo_actual: Optional[str],
) -> Optional[str]:
    """Completa un evento tipado comparando su texto con las etiquetas vivas del esquema.

    La coincidencia tolera flexiones cercanas mediante prefijos compartidos y
    solo decide cuando una etiqueta obtiene una coincidencia única y clara.
    """
    if tipo_actual or alcance not in {"evento", "ambos"}:
        return tipo_actual

    texto_norm = normalizar(texto or "")
    palabras_mensaje = re.findall(r"[a-z0-9]+", texto_norm)
    if not palabras_mensaje:
        return None

    puntuaciones: Dict[str, Tuple[float, int, int, int]] = {}
    palabras_no_discriminantes = {"almacen", "almacenes", "movimiento", "movimientos", "tipo", "tipos"}
    for codigo, etiqueta in _tipos_movimiento_validos().items():
        candidatos_fuente = [etiqueta]
        if "_" not in codigo:
            candidatos_fuente.append(codigo.lower())
        mejor_para_codigo: Optional[Tuple[float, int, int, int]] = None
        for fuente in candidatos_fuente:
            palabras_etiqueta = re.findall(r"[a-z0-9]+", normalizar(fuente))
            palabras_etiqueta = [
                palabra for palabra in palabras_etiqueta
                if len(palabra) >= 4 and palabra not in palabras_no_discriminantes
            ]
            if not palabras_etiqueta:
                continue

            aciertos = 0
            exactos = 0
            prefijo_total = 0
            for palabra_etiqueta in palabras_etiqueta:
                prefijo = 0
                exacta = False
                for palabra_mensaje in palabras_mensaje:
                    if palabra_mensaje == palabra_etiqueta:
                        exacta = True
                        prefijo = len(palabra_etiqueta)
                        break
                    comun = 0
                    for izquierda, derecha in zip(palabra_etiqueta, palabra_mensaje):
                        if izquierda != derecha:
                            break
                        comun += 1
                    if min(len(palabra_etiqueta), len(palabra_mensaje)) >= 6 and comun >= 5:
                        prefijo = max(prefijo, comun)
                if prefijo:
                    aciertos += 1
                    exactos += int(exacta)
                    prefijo_total += prefijo

            # Para subtipos compuestos (p. ej. TRASLADO_INTERNO vs TRASLADO_ENTRE),
            # exige que estén presentes todos los términos discriminantes de la etiqueta.
            if aciertos and ("_" not in codigo or aciertos == len(palabras_etiqueta)):
                punt = (
                    aciertos / len(palabras_etiqueta),
                    aciertos,
                    exactos,
                    prefijo_total,
                )
                if mejor_para_codigo is None or punt > mejor_para_codigo:
                    mejor_para_codigo = punt
        if mejor_para_codigo is not None:
            puntuaciones[codigo] = mejor_para_codigo

    if puntuaciones:
        mejor = max(puntuaciones.values())
        ganadores = [codigo for codigo, puntuacion in puntuaciones.items() if puntuacion == mejor]
        if len(ganadores) == 1:
            return ganadores[0]

    tipos_validos = _tipos_movimiento_validos()
    coincidencias_verbo = [
        codigo for codigo, patron in _VERBOS_EVENTO_PASADO_A_TIPO
        if codigo in tipos_validos and re.search(patron, texto_norm)
    ]
    return coincidencias_verbo[0] if len(coincidencias_verbo) == 1 else None


def _tipo_movimiento_mencionado(texto: str, codigo: Optional[str]) -> bool:
    """Comprueba una clase de evento solo contra códigos y etiquetas vivos del modelo."""
    if not codigo:
        return False
    opciones = _tipos_movimiento_validos()
    etiqueta = opciones.get(codigo)
    if not etiqueta:
        return False
    palabras = {_raiz(token) for token in re.findall(r"[a-z0-9]+", normalizar(texto or ""))}
    opciones_nombre = (codigo.replace("_", " "), etiqueta)
    for opcion in opciones_nombre:
        objetivo = {
            _raiz(token)
            for token in re.findall(r"[a-z0-9]+", normalizar(opcion))
            if token not in {"de", "del", "la", "el", "los", "las", "por", "a", "entre", "mismo"}
        }
        if objetivo and objetivo.issubset(palabras):
            return True
    return False


def _contexto_tipos_movimiento() -> str:
    tipos = _tipos_movimiento_validos()
    return ", ".join(f"{codigo}={etiqueta}" for codigo, etiqueta in tipos.items())


def _int(v: Any) -> Optional[int]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        n = int(float(str(v).strip()))
        return n if n >= 0 else None
    except (TypeError, ValueError):
        return None


def _menciones_catalogo(texto: str, nombres: List[str]) -> List[str]:
    """Devuelve solo nombres actuales del catálogo que aparecen literalmente en el turno."""
    texto_norm = f" {normalizar(texto)} "
    encontrados = []
    for nombre in sorted(set(nombres or []), key=lambda valor: len(normalizar(valor)), reverse=True):
        nombre_norm = normalizar(nombre)
        if nombre_norm and f" {nombre_norm} " in texto_norm:
            encontrados.append(nombre)
    return encontrados


def _menciones_almacenes_catalogo(texto: str, nombres: List[str]) -> List[str]:
    """Resuelve menciones de almacenes vigentes aceptando la sinonimia almacén/bodega."""
    texto_norm = f" {normalizar(texto)} "
    encontrados = []
    for nombre in sorted(set(nombres or []), key=lambda valor: len(normalizar(valor)), reverse=True):
        nombre_norm = normalizar(nombre)
        if not nombre_norm:
            continue
        variantes = [nombre_norm]
        if nombre_norm.startswith("almacen "):
            variantes.append("bodega " + nombre_norm[len("almacen "):])
        elif nombre_norm.startswith("bodega "):
            variantes.append("almacen " + nombre_norm[len("bodega "):])
        if any(f" {var} " in texto_norm for var in variantes):
            encontrados.append(nombre)
    return encontrados


def _texto_menciona_stock_minimo(texto: str) -> bool:
    """True solo si el mensaje refiere explícita o parafrásticamente al umbral/stock mínimo."""
    txt_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    if not txt_norm:
        return False
    return bool(re.search(
        r"\b(?:minim\w*|umbral\w*|limite\w*|tope\w*|"
        r"nivel\s+(?:de\s+reposic\w*|fijad\w*|configurad\w*|de\s+alerta)|"
        r"referencia\s+para\s+repon\w*|punto\s+de\s+(?:reorden|reposic\w*))\b",
        txt_norm,
    ))


def _condicion_stock_tiene_indicio_textual(texto: str, condicion: Optional[str]) -> bool:
    """Verifica evidencia léxica mínima para impedir que el LLM alucine condiciones de stock/mínimo."""
    if not condicion:
        return True
    txt_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    if not txt_norm:
        return False
    if condicion == "minimo_cero":
        return _texto_menciona_stock_minimo(texto) and bool(
            re.search(r"(?<!\d)0+(?!\d)|\b(?:cero|nulo|nula|sin\s+(?:stock\s+)?minim\w*|ningun\w*)\b", txt_norm)
        )
    if condicion == "suficiente":
        return bool(re.search(
            r"\b(?:suficient\w*|sobr\w*|holgur\w*|margen|ampli\w*|por\s+encima|encima\s+de|"
            r"supera\w*|bastante\w*|alcanz\w*|cubr\w*|bien\s+de\s+stock|buen\s+stock|arriba\s+de)\b",
            txt_norm,
        ))
    if condicion == "en_minimo":
        return bool(re.search(
            r"\b(?:minim\w*|umbral\w*|limite\w*|tope\w*)\b",
            txt_norm,
        )) and bool(re.search(
            r"\b(?:justo|exact\w*|igual\w*|alcanz\w*|lleg\w*|tope\w*|en\s+(?:su|el)\s+(?:stock\s+)?minim\w*|al\s+minim\w*)\b",
            txt_norm,
        ))
    if condicion == "cerca_minimo":
        return _texto_menciona_stock_minimo(texto) or bool(re.search(
            r"\b(?:cerca\w*|proxim\w*|a\s+punto|casi|riesgo|margen|rozando|por\s+bajar)\b",
            txt_norm,
        ))
    if condicion == "bajo_minimo":
        return _texto_menciona_stock_minimo(texto) or bool(re.search(
            r"\b(?:baj[oa]s?|debajo|poc[oa]s?|escas\w*|cort[oa]s?|insuficien\w*|"
            r"menor\w*|inferior\w*|criti\w*|alert\w*|falt\w*|repon\w*|reposic\w*|agotand\w*)\b",
            txt_norm,
        ))
    if condicion == "requiere_reposicion":
        return _texto_menciona_stock_minimo(texto) or bool(re.search(
            r"\b(?:repon\w*|reposic\w*|reabast\w*|abastec\w*|pedir|comprar|surtir|"
            r"falt\w*|necesit\w*|requier\w*|urgen\w*|agotad\w*|escas\w*|baj[oa]s?)\b",
            txt_norm,
        ))
    if condicion == "agotado":
        return bool(re.search(
            r"\b(?:agotad\w*|sin\s+(?:stock|existencias?|unidades?)|cero|0|acab\w*|termin\w*|"
            r"gast\w*|no\s+(?:hay|tiene\w*|quedan?)|vaci[oa]s?|nulo|nula|exhaust\w*|ningun\w*)\b",
            txt_norm,
        ))
    if condicion == "con_stock":
        return bool(re.search(
            r"\b(?:disponib\w*|con\s+stock|en\s+stock|con\s+existencias?|hay\s+existencias?|"
            r"hay\s+stock|tienen?\s+stock|tengan\s+stock|quedan|despach\w*|entreg\w*|positiv\w*)\b",
            txt_norm,
        ))
    return True


def _texto_menciona_elegibilidad_prestamo(texto: str) -> bool:
    """True solo si el mensaje refiere a elegibilidad para préstamo o carácter consumible/retornable."""
    txt_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    if not txt_norm:
        return False
    return bool(re.search(
        r"\b(?:prestab\w*|retornab\w*|consumib\w*|devolutiv\w*|elegib\w*|"
        r"(?:pued\w*|podemos|permit\w*|sirv\w*|aplica\w*)\s+(?:dar\s+en\s+prestamo|prestar)|"
        r"admit\w*\s+prestam\w*|para\s+prestar|para\s+prestam\w*|"
        r"uso\s+definitiv\w*|(?:no\s+)?se\s+devuelv\w*|se\s+regres\w*)\b",
        txt_norm,
    ))


def _texto_menciona_prestamo_o_custodia(texto: str) -> bool:
    """True solo si el mensaje refiere a préstamos, devoluciones o custodia temporal fuera del almacén."""
    txt_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    if not txt_norm:
        return False
    return bool(re.search(
        r"\b(?:prest\w*|devol\w*|devuel\w*|regres\w*|retorn\w*|reintegr\w*|"
        r"pendient\w*|afuera|fuera\s+de(?:l)?\s+(?:almacen|bodega|inventario)|"
        r"custodia|en\s+(?:poder|manos)\s+de|terceros?|usuarios?|personas?|responsables?|"
        r"quien(?:es)?\s+(?:(?:lo|la|los|las|se|nos)\s+)?(?:tiene\w*|llev\w*)|"
        r"a\s+quien(?:es)?\s+se\b|se\s+llev\w*|alguien\s+tiene)\b",
        txt_norm,
    ))


def _texto_menciona_verbo_mutacion(texto: str) -> bool:
    """True solo si el mensaje contiene indicio léxico de operación/mutación o alta/edición."""
    txt_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    if not txt_norm:
        return False
    return bool(re.search(
        r"\b(?:prest\w*|devolv\w*|devuelv\w*|devolu\w*|trajo|trajeron|regres\w*|retorn\w*|reintegr\w*|"
        r"ingres\w*|entrad\w*|met(?:er|e|a|i\w*)|sac\w*|salid\w*|retir\w*|despach\w*|entreg\w*|"
        r"traslad\w*|moviliz\w*|reubic\w*|mover|mueve|cambi\w*|modifi\w*|actuali\w*|edit\w*|ajust\w*|"
        r"inhabilit\w*|deshabilit\w*|desactiv\w*|habilit\w*|reactiv\w*|"
        r"crea\w*|registr\w*|agreg\w*|anad\w*|incorpor\w*|cargar|"
        r"dar\s+(?:de\s+)?(?:alta|baja)|dar\s+(?:entrada|salida)|"
        r"alta\s+(?:de\s+|al\s+)?(?:producto|articulo|catalogo)|"
        r"nuev[oa]\s+(?:producto|articulo|referencia)|"
        r"dejar\s+(?:en\s+)?prestamo|llevar(?:me|se|nos)?|"
        r"(?:poner|fijar|subir|bajar)\s+(?:el\s+)?(?:stock\s+)?minimo)\b",
        txt_norm,
    ))


def _candidatos_catalogo_por_solapamiento(texto: str, nombres: List[str], limite: int = 5) -> List[str]:
    """Sugiere nombres actuales del catálogo a partir de tokens compartidos.

    Solo prepara contexto para el LLM; la selección final y el acceso a datos
    siguen pasando por validación y permisos.
    """
    tokens_consulta = set(tokens_significativos(texto or ""))
    if not tokens_consulta:
        return []
    puntuados = []
    for nombre in sorted(set(nombres or [])):
        tokens_nombre = set(tokens_significativos(nombre))
        compartidos = tokens_consulta & tokens_nombre
        if not compartidos:
            continue
        cobertura = len(compartidos) / max(1, len(tokens_nombre))
        puntuados.append((len(compartidos), cobertura, nombre))
    puntuados.sort(key=lambda fila: (-fila[0], -fila[1], normalizar(fila[2])))
    return [nombre for _, _, nombre in puntuados[: max(1, min(limite, 8))]]


def construir_contexto(
    estado: Any,
    categorias: List[str],
    almacenes: List[str],
    texto_usuario: Optional[str] = None,
    ubicaciones: Optional[List[str]] = None,
    productos_candidatos: Optional[List[Dict[str, str]]] = None,
) -> str:
    # El contexto solo incluye entidades mencionadas en el turno. Así no se
    # omiten entidades por su posición ni crece con el tamaño del inventario.
    cats_mencionadas = _menciones_catalogo(texto_usuario or "", categorias or []) if texto_usuario else []
    cats_candidatas = (
        _candidatos_catalogo_por_solapamiento(texto_usuario or "", categorias or [])
        if texto_usuario and not cats_mencionadas else []
    )
    alms_mencionados = _menciones_catalogo(texto_usuario or "", almacenes or []) if texto_usuario else []
    alms_candidatos = (
        _candidatos_catalogo_por_solapamiento(texto_usuario or "", almacenes or [])
        if texto_usuario and not alms_mencionados else []
    )
    if not alms_mencionados and not alms_candidatos and len(set(almacenes or [])) <= 8:
        alms_candidatos = sorted(set(almacenes or []), key=normalizar)
    ubicaciones_mencionadas = _resolver_menciones_ubicacion(texto_usuario or "", ubicaciones or []) if texto_usuario else []
    lineas = [f"Categorías del catálogo mencionadas: {', '.join(cats_mencionadas) or 'ninguna'}"]
    lineas.append(f"Almacenes accesibles mencionados: {', '.join(alms_mencionados) or 'ninguno'}")
    if alms_candidatos:
        lineas.append(f"Almacenes candidatos actuales accesibles: {', '.join(alms_candidatos[:8])}")
    lineas.append(
        "Ubicaciones accesibles mencionadas (entidades físicas, no productos; conserve su nombre exacto): "
        f"{', '.join(ubicaciones_mencionadas) or 'ninguna'}"
    )
    if cats_candidatas:
        nombres_cat = ", ".join(cats_candidatas)
        if len(cats_candidatas) == 1:
            nombres_cat += " (coincidencia única)"
        lineas.append(f"Categorías candidatas actuales: {nombres_cat}")
    if productos_candidatos:
        etiqueta_prod = (
            "Coincidencia única de producto actual accesible"
            if len(productos_candidatos) == 1
            else "Candidatos actuales de producto accesibles (referencias para resolver el mensaje)"
        )
        tokens_consulta = set(tokens_significativos(texto_usuario or ""))
        lineas_productos = []
        for item in productos_candidatos[:8]:
            nombre = str(item.get("nombre") or "").strip()
            categoria = str(item.get("categoria") or "").strip()
            descripcion = str(item.get("descripcion") or "").strip()
            linea = f"{item.get('codigo')} | {nombre}"
            if categoria:
                linea += f" | {categoria}"
            # Cuando la consulta encuentra una referencia por su descripción,
            # expón esa evidencia viva al modelo. No se añaden alias al código;
            # el texto procede del producto accesible del catálogo.
            tokens_nombre = set(tokens_significativos(f"{nombre} {categoria}"))
            tokens_descripcion = set(tokens_significativos(descripcion))
            if descripcion and tokens_consulta.intersection(tokens_descripcion - tokens_nombre):
                linea += f" | descripción: {descripcion[:160]}"
            lineas_productos.append(linea)
        lineas.append(
            etiqueta_prod + ": "
            + "; ".join(lineas_productos)
        )
    else:
        lineas.append("Productos candidatos actuales accesibles que coinciden con el mensaje: ninguno")
    if estado is not None:
        if getattr(estado, "producto_id", None):
            lineas.append(f"Producto en foco: {estado.producto_codigo} {estado.producto_nombre}")
        coleccion = getattr(estado, "coleccion_productos", None) or []
        foco_id = getattr(estado, "producto_id", None)
        foco_codigo = normalizar_codigo_producto(getattr(estado, "producto_codigo", None) or "")
        coleccion_unica_es_foco = bool(
            len(coleccion) == 1
            and (
                (foco_id is not None and coleccion[0].get("id") == foco_id)
                or (
                    foco_codigo
                    and normalizar_codigo_producto(coleccion[0].get("codigo") or "") == foco_codigo
                )
            )
        )
        if coleccion and not coleccion_unica_es_foco:
            listado = "; ".join(f"{c.get('codigo')} {c.get('nombre')}" for c in coleccion[:6])
            total = max(len(coleccion), int(getattr(estado, "coleccion_total_original", len(coleccion)) or len(coleccion)))
            lineas.append(f"Resultados previos: total completo={total}; muestra contextual parcial={listado}")
    return "\n".join(lineas)


_STOPWORDS_UBICACION = {"de", "del", "la", "las", "el", "los", "en", "y", "un", "una", "unos", "unas", "al"}


def _claves_ubicacion_por_nombre(ubicaciones: List[str]) -> Dict[str, str]:
    """Recupera claves canónicas actuales para resolver variaciones de formato."""
    if not ubicaciones:
        return {}
    try:
        from inventario.models import Ubicacion
        filas = Ubicacion.objects.filter(
            nombre__in=ubicaciones,
            activo=True,
            eliminado=False,
            almacen__activo=True,
            almacen__eliminado=False,
        ).values_list("nombre", "clave_canonica")
        return {nombre: clave for nombre, clave in filas}
    except Exception:
        return {}


def _codigos_alfanumericos_ubicacion(texto: str) -> set[str]:
    """Normaliza códigos alfanuméricos de etiquetas sin conocer sus valores."""
    encontrados = set()
    for match in re.finditer(
        r"(?<![a-z0-9])([a-z])\s*[-_ ]?\s*(\d+)(?![a-z0-9])",
        normalizar(texto or ""),
    ):
        encontrados.add(f"{match.group(1)}{int(match.group(2))}")
    return encontrados


def _resolver_menciones_ubicacion(texto: str, ubicaciones: List[str]) -> List[str]:
    """Reconoce ubicaciones literales, códigos o coincidencias con cobertura alta."""
    exactas = _menciones_catalogo(texto, ubicaciones or [])
    if exactas:
        return exactas

    codigos_mencionados = _codigos_alfanumericos_ubicacion(texto)
    if codigos_mencionados:
        claves = _claves_ubicacion_por_nombre(ubicaciones or [])
        coincidencias_codigo = []
        for nombre in sorted(set(ubicaciones or [])):
            codigos_catalogo = (
                _codigos_alfanumericos_ubicacion(nombre)
                | _codigos_alfanumericos_ubicacion(claves.get(nombre, ""))
            )
            if codigos_mencionados & codigos_catalogo:
                coincidencias_codigo.append(nombre)
        if coincidencias_codigo:
            return coincidencias_codigo

    tokens_texto = {token for token in normalizar(texto).split() if token not in _STOPWORDS_UBICACION and len(token) > 2}
    if len(tokens_texto) < 2:
        return []
    candidatas = []
    for nombre in sorted(set(ubicaciones or [])):
        tokens_nombre = {token for token in normalizar(nombre).split() if token not in _STOPWORDS_UBICACION and len(token) > 2}
        comunes = tokens_texto & tokens_nombre
        # Coincidencias parciales pueden ser una categoría o parte del nombre
        # de un producto. Acepta la variante solo si también aparece un tipo
        # físico de ubicación (vitrina, estante, cajón, etc.) en el turno.
        ancla_espacial = bool(comunes & _TERMINOS_UBICACION_GENERICOS)
        if (
            len(comunes) >= 2
            and len(comunes) / max(1, len(tokens_nombre)) >= 0.65
            and ancla_espacial
        ):
            candidatas.append((len(comunes) / len(tokens_nombre), len(comunes), nombre))
    if not candidatas:
        return []
    candidatas.sort(reverse=True)
    mejor = candidatas[0]
    if len(candidatas) > 1 and candidatas[1][:2] == mejor[:2]:
        return []
    return [mejor[2]]


_TERMINOS_UBICACION_GENERICOS = {
    "almacen", "almacenes", "bodega", "bodegas", "estante", "estantes", "vitrina", "vitrinas",
    "cajon", "cajones", "repisa", "repisas", "ubicacion", "ubicaciones", "estanteria", "estanterias",
}


def _es_filtro_ubicacion_generico(valor: Optional[str]) -> bool:
    """Reconoce un tipo genérico de lugar, no un producto ni una ubicación concreta."""
    if not valor:
        return False
    tokens = [normalizar(t) for t in re.findall(r"[\wáéíóúñü-]+", valor, flags=re.IGNORECASE)]
    tokens = [t for t in tokens if t not in {"el", "la", "los", "las", "en", "de", "del", "un", "una"}]
    return bool(tokens) and all(t in _TERMINOS_UBICACION_GENERICOS for t in tokens)


def _es_consulta_ubicacion_eliptica(texto: str) -> bool:
    """Detecta consultas de ubicación que omiten el producto ya activo en contexto.

    La regla es sintáctica y no depende de nombres del catálogo. Si hay un
    sustantivo explícito después del marco de ubicación, la consulta no es
    elíptica y debe conservar su filtro.
    """
    texto_norm = normalizar(re.sub(r"[¿?¡!.,;:]", " ", texto or ""))
    # En conversación es normal enlazar una pregunta con «y», «además» o
    # «entonces». Se quita solo el conector inicial; cualquier sustantivo o
    # entidad que siga impide que la consulta se trate como elíptica.
    while re.match(r"^(?:y|tambien|ademas|entonces|ahora|pues)\s+", texto_norm):
        texto_norm = re.sub(r"^(?:y|tambien|ademas|entonces|ahora|pues)\s+", "", texto_norm, count=1)
    marcos = (
        r"^(?:en\s+)?donde\s+(?:(?:lo|la|los|las)\s+)?"
        r"(?:(?:se\s+)?(?:encuentra|encuentran|guarda|guardan|ubica|ubican|almacena|almacenan|queda|quedan|esta|estan))"
        r"(?:\s+(?:guardado|guardada|guardados|guardadas|ubicado|ubicada|ubicados|ubicadas|almacenado|almacenada|almacenados|almacenadas|situado|situada|situados|situadas|metido|metida|metidos|metidas))?"
        r"(?:\s+(?:ese|esa|esos|esas|este|esta|estos|estas|eso|ese\s+producto|este\s+producto|el\s+mismo|la\s+misma))?$",
        r"^(?:en\s+que|en\s+cual)\s+ubicacion\s+(?:(?:se\s+)?(?:encuentra|encuentran|guarda|guardan|ubica|ubican|almacena|almacenan|queda|quedan|esta|estan))"
        r"(?:\s+(?:guardado|guardada|guardados|guardadas|ubicado|ubicada|ubicados|ubicadas|almacenado|almacenada|almacenados|almacenadas|situado|situada|situados|situadas|metido|metida|metidos|metidas))?"
        r"(?:\s+(?:ese|esa|esos|esas|este|esta|estos|estas|eso|ese\s+producto|este\s+producto|el\s+mismo|la\s+misma))?$",
        r"^cual\s+es\s+su\s+ubicacion$",
        r"^(?:y\s+)?en\s+(?:que|cual)\s+(?:almacen|bodega)\s+(?:(?:se\s+)?(?:encuentra|queda|esta|ubica|guarda)(?:n)?)"
        r"(?:\s+(?:ese|esa|esos|esas|este|esta|estos|estas|eso))?$",
    )
    return any(re.fullmatch(marco, texto_norm) for marco in marcos)


def _mensaje_requiere_referencia_contextual(texto: str) -> bool:
    """Conserva en el esquema la referencia solo cuando el turno puede depender del diálogo."""
    if not texto:
        return True
    texto_norm = normalizar(preprocesar_texto_inventario(texto))
    texto_sin_temporal = re.sub(
        r"\b(?:(?:esta|este|ultima|ultimo)\s+(?:semana|mes|a[ñn]o|jornada|turno)|"
        r"(?:semana|mes|a[ñn]o)\s+anterior(?:es)?|"
        r"ultimos?\s+\w+\s+dias)\b",
        " ",
        texto_norm,
    )
    marcas_referenciales = bool(re.search(
        r"\b(?:su|sus|este|esta|estos|estas|ese|esa|esos|esas|aquel|aquella|aquellos|aquellas|"
        r"dicho|dicha|mismo|misma|ellos|ellas|anteriores?|primero|primera|segundo|segunda|"
        r"ultimo|ultima|cuantos?\s+tiene|cuantas?\s+tiene|cuanto\s+queda|cuantas?\s+quedan)\b",
        texto_sin_temporal,
    ))
    return bool(
        marcas_referenciales
        or tiene_referencia_resultados_previos(texto)
        or tiene_referencia_contextual_operacion(texto)
        or _es_consulta_ubicacion_eliptica(texto)
        or re.match(r"^(?:y|tambien|ademas)\b", texto_norm)
    )


def tiene_referencia_resultados_previos(texto: str) -> bool:
    """Reconoce referencias a una colección previa sin depender de sus productos."""
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    if re.search(
        r"\b(?:olvida\w*|descarta\w*|ignora\w*|deja(?:\s+atras|\s+de\s+lado)?|"
        r"no\s+uses?|sin\s+usar|aparte\s+de)\s+"
        r"(?:esa|esta|la|ese|este|el|esos|esas|estos|estas|los|las)\s+"
        r"(?:lista|listado|coleccion|busqueda|consulta|respuesta|resultado|resultados|conjunto|grupo|tabla)\b",
        texto_norm,
    ):
        return False
    return bool(re.search(
        r"\b(?:"
        r"(?:esa|esta|ese|este|esos|esas|estos|estas)\s+"
        r"(?:lista|listado|coleccion|busqueda|consulta|respuesta|resultado|resultados|conjunto|grupo|tabla)|"
        r"(?:el|la|los|las)\s+(?:busqueda|consulta|respuesta|resultado|resultados)|"
        r"(?:el|la|los|las)\s+(?:lista|listado|coleccion|conjunto|grupo|tabla)"
        r"(?!\s+(?:de\s+(?!es[oa]s?\b|ell[oa]s\b|antes\b|lo\s+anterior\b)|"
        r"del\s+(?!anterior\b|previo\b)|complet[oa]s?\b|general(?:es)?\b|total(?:es)?\b|actual(?:es)?\b|vigentes?\b))|"
        r"(?:lista|listado|coleccion|busqueda|consulta|resultado|resultados|conjunto|grupo|tabla)\s+"
        r"(?:anterior|previa|previo|anteriores|previos)|"
        r"(?:de|entre|sobre)\s+(?:esos|esas|ellos|ellas)(?:\s+(?:productos?|articulos?|referencias?|elementos?))?|"
        r"(?:esos|esas|ellos|ellas)\s+(?:productos?|articulos?|referencias?|elementos?)|"
        r"cuant[oa]s?\s+(?:(?:productos?|articulos?|referencias?|registros?|elementos?|items?|resultados?)\s+)?"
        r"(?:salieron|aparecieron|resultaron|figuraron)"
        r")\b",
        texto_norm,
    ))


def _consulta_coordina_entidad(
    texto: str,
    codigos_productos_candidatos: Optional[List[str]] = None,
    tiene_foco: bool = False,
) -> bool:
    """Detecta coordinación gramatical asociada a una entidad individual.

    Los códigos se leen del catálogo accesible en esta consulta; no se fija
    ningún producto. La marca de coordinación sirve para elegir una forma de
    salida estructurada, mientras el LLM interpreta qué atributos se piden.
    """
    codigos_mencionados = set(extraer_codigos_de_texto(texto or ""))
    codigos_candidatos = {
        str(codigo).strip().upper()
        for codigo in (codigos_productos_candidatos or [])
        if codigo
    }
    entidad_individual = bool(
        tiene_foco
        or len(codigos_mencionados & codigos_candidatos) == 1
        or (not codigos_mencionados and len(codigos_candidatos) == 1)
    )
    codigos_entidad = codigos_mencionados & codigos_candidatos
    if not codigos_entidad and len(codigos_candidatos) == 1:
        codigos_entidad = set(codigos_candidatos)
    texto_para_coordinacion = normalizar(
        preprocesar_texto_inventario(texto or "")
    )
    if entidad_individual and codigos_entidad:
        # Un conector que forma parte del nombre real de un producto o de su
        # categoría no expresa una segunda petición. Retira esas entidades
        # dinámicas antes de analizar la coordinación; los valores proceden
        # del catálogo actual y no de una lista de nombres en el código.
        try:
            from inventario.models import Producto

            etiquetas_catalogo = Producto.objects.filter(
                codigo_interno__in=codigos_entidad,
            ).values_list("nombre", "categoria__nombre")
            nombres_catalogo = {
                normalizar(preprocesar_texto_inventario(valor or ""))
                for fila in etiquetas_catalogo
                for valor in fila
                if valor
            }
        except Exception:
            nombres_catalogo = set()
        for nombre in sorted(nombres_catalogo, key=len, reverse=True):
            if not nombre:
                continue
            texto_para_coordinacion = re.sub(
                rf"(?<!\w){re.escape(nombre)}(?!\w)",
                " ",
                texto_para_coordinacion,
            )
    texto_sin_saludo_y = re.sub(
        r"^(?:(?:hola|oye|buenas|buenos\s+dias|buenas\s+tardes|buenas\s+noches|si|aja|ok|vale|hacu|haku)\s+)*y\b",
        " ",
        texto_para_coordinacion.strip(),
    ).strip()
    hay_coordinacion = bool(re.search(
        r"\b(?:y|e|ni|ademas|tambien|junto\s+con|asi\s+como)\b",
        texto_sin_saludo_y,
    ))
    return entidad_individual and hay_coordinacion


def _construir_prompt_sistema(
    texto: str,
    codigos_productos_candidatos: Optional[List[str]] = None,
    tiene_foco: bool = False,
) -> str:
    """Adapta el énfasis a la estructura y a las entidades de la consulta."""
    prompt = PROMPT_SISTEMA
    consulta_compuesta = _consulta_coordina_entidad(
        texto,
        codigos_productos_candidatos=codigos_productos_candidatos,
        tiene_foco=tiene_foco,
    )
    if re.search(r"[,;]", texto or "") or consulta_compuesta:
        prompt = prompt.replace(
            EJEMPLO_STOCK_INDIVIDUAL,
            INSTRUCCION_CAMPOS_COMPUESTOS,
            1,
        )
    if not _mensaje_requiere_referencia_contextual(texto):
        prompt = re.sub(r',"referencia":"[^"]+"', "", prompt, count=1)
    return prompt


def _destinatario_anclado_en_mensaje(texto: str, destinatario: str) -> bool:
    """Acepta un destinatario solo si aparece tras una construcción de destino explícita."""
    if not texto or not destinatario:
        return False
    patron = re.compile(
        r"\b(?:a\s+nombre\s+de|a\s+favor\s+de|entregar\s+a|entregado\s+a|para|a)\s+"
        r"([a-záéíóúñü0-9][a-záéíóúñü0-9\s'.-]*?)"
        r"(?=\s+(?:por|para|con|en|motivo)\b|[?!,;:]|$)",
        re.IGNORECASE,
    )
    objetivo = normalizar(destinatario)
    if not objetivo:
        return False
    pronombres_y_referencias = {
        "el", "la", "los", "las", "lo", "le", "les", "ellos", "ellas",
        "ese", "esa", "eso", "esos", "esas", "este", "esta", "estos", "estas",
        "aquel", "aquella", "aquello", "aquellos", "aquellas", "uno", "una",
        "alguien", "nadie", "cualquiera", "mismo", "misma",
    }
    objetivo_tokens = set(objetivo.split())
    if objetivo_tokens and objetivo_tokens.issubset(pronombres_y_referencias):
        return False
    return any(
        objetivo in normalizar(coincidencia.group(1))
        or set(objetivo.split()).issubset(set(normalizar(coincidencia.group(1)).split()))
        for coincidencia in patron.finditer(texto)
    )


def _extraer_objeto_consulta_sin_marco(texto: str) -> Optional[str]:
    """Retira pregunta, verbos de consulta y ubicaciones genéricas del texto original."""
    # El filtro conserva la escritura del usuario. Las fracciones y unidades se
    # normalizan después, solo al comparar la consulta con el catálogo.
    t = preprocesar_texto_inventario(
        texto or "", normalizar_medidas=False,
    ).lstrip("¿¡ \t\r\n")
    try:
        from ..servicios_temporales import limpiar_expresiones_temporales
        t = limpiar_expresiones_temporales(t)
    except Exception:
        pass
    # Quita el marco antes del objeto para no convertir palabras de la
    # pregunta ("almacén está", "bodega se encuentra") en términos de búsqueda.
    t = re.sub(
        r"^\s*(?:en\s+)?(?:qu[eé]|cu[aá]l)\s+(?:almac[eé]n|bodega)"
        r"(?:\s+y\s+(?:la\s+)?ubicaci[oó]n)?\s+"
        r"(?:(?:se)\s+)?(?:encuentra|encuentran|est[aá]|est[aá]n|guarda|guardan|ubica|ubican|almacena|almacenan)\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\b(?:en|dentro\s+de)\s+(?:(?:el|la|los|las)\s+)?"
        r"(?:estantes?|vitrinas?|cajones?|repisas?|ubicaciones?|estanter[ií]as?)\b(?:\s+[a-z]\s*-?\s*\d+)?",
        " ", t, flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\bpara\s+(?:sacar|retirar|entregar|usar|prestar|llevar|despachar)\b",
        " ", t, flags=re.IGNORECASE,
    )
    palabras_vacias = {
        "oye", "oiga", "disculpa", "disculpe", "favor", "hola", "buenas", "hacu", "haku",
        "cuadro", "compae", "compadre", "compa", "llave", "ombe", "eche", "aja", "ve", "aca",
        "quiero", "querer", "quisiera", "necesito", "necesitar", "deseo", "desear", "puedo", "puedes",
        "podria", "podrias", "saber", "conocer", "decir", "decirme", "dime", "digame", "mostrar", "mostrarme", "mostrame",
        "busca", "buscar", "buscame", "encuentra", "encontrar", "encontrarme", "localiza", "localizar", "ubicar", "hallar", "consulta", "consultar", "ver", "revisa", "revisar",
        "que", "cual", "cuales", "como", "donde", "cuanto", "cuanta", "cuantos", "cuantas", "quien", "quienes",
        "stock", "existencia", "existencias", "disponibilidad", "unidad", "unidades", "queda", "quedan", "hay", "catalogo",
        "tenemos", "tienen", "tiene", "existe", "existen", "disponible", "disponibles", "actual",
        "escaso", "escasa", "escasos", "escasas", "bajo", "debajo",
        "sigue", "siguen", "prestada", "prestadas", "prestado", "prestados", "prestamo", "prestamos",
        "aun", "todavia", "actualmente", "continua", "continuan", "ahora",
        "actuales", "minimo", "minima", "codigo", "ubicacion", "almacen", "movimiento", "movimientos",
        "historial", "historico", "historica", "ultimos", "ultimas", "ultimo", "ultima", "reciente", "recientes",
        "en", "de", "del", "el", "la", "los", "las", "un", "una", "unos", "unas",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella",
        "aquellos", "aquellas", "mismo", "misma", "mismos", "mismas", "dicho", "dicha",
        "dichos", "dichas", "todo", "toda", "todos", "todas", "completo", "completa",
        "producto", "productos", "articulo", "articulos", "referencia", "referencias", "item", "items",
        "indicame", "indicarme", "detalle", "detalles", "ficha", "informacion", "datos",
        "algo", "algun", "alguna", "alguno", "algunos", "algunas", "nada",
        "vaina", "vainas", "cosa", "cosas", "muestra", "muestrame", "muestreme", "enseña", "enseñame", "ensename", "lista", "listado", "listame", "listar", "dame",
        "prestable", "prestables", "prestar", "prestan", "retornable", "retornables",
        "consumible", "consumibles", "devolutivo", "devolutivos", "devolutiva", "devolutivas",
        "elegible", "elegibles", "puede", "pueden", "pueda", "puedan", "admite", "admiten",
        "permite", "permiten", "se", "me", "te", "le", "les", "nos", "lo", "no", "si", "para", "por", "con", "sin",
        "tuvo", "tuvieron", "hubo", "han", "ha", "habido", "registraron", "registro", "realizaron", "realizo",
        "hicieron", "hizo", "ocurrieron", "ocurrio", "efectuaron", "efectuo", "sucedieron", "sucedio",
        "movio", "movieron", "movido", "presto", "prestaron", "devolvio", "devolvieron",
        "entro", "entraron", "entra", "entran", "ingreso", "ingresaron", "ingresa", "ingresan",
        "llego", "llegaron", "llega", "llegan", "salio", "salieron", "sale", "salen", "salga", "salgan",
        "despacho", "despacharon", "traslado", "trasladaron", "van", "estuvieron", "fueron",
        "vigente", "vigentes", "pendiente", "pendientes", "activo", "activos", "activa", "activas",
        "abierto", "abiertos", "abierta", "abiertas", "dotacion", "dotaciones", "ajuste", "ajustes",
        "semana", "semanas", "mes", "meses", "dia", "dias", "anio", "ano", "jornada", "turno", "periodo",
    }
    # El extractor quita el marco genérico del dominio y los determinantes;
    # los nombres y descripciones de productos se siguen resolviendo desde el
    # catálogo actual, sin enumerarlos aquí.
    palabras_vacias.update(_PALABRAS_DOMINIO_INVENTARIO)
    palabras_vacias.update({"algun", "alguna", "algunos", "algunas", "cualquier", "cualesquiera", "es", "son", "esta", "estan"})
    palabras_vacias = {normalizar(palabra) for palabra in palabras_vacias}
    tokens = re.findall(r"[\wáéíóúñü]+(?:[-/][\wáéíóúñü]+)*", t, flags=re.IGNORECASE)
    tokens_objeto = [token for token in tokens if normalizar(token) not in palabras_vacias]
    while (
        len(tokens_objeto) > 1
        and len(normalizar(tokens_objeto[-1])) >= 6
        and normalizar(tokens_objeto[-1]).endswith(("an", "en"))
    ):
        tokens_objeto.pop()
    while tokens_objeto and normalizar(tokens_objeto[-1]) in {"y", "o", "pero", "aunque"}:
        tokens_objeto.pop()
    objeto = " ".join(tokens_objeto)
    objeto = re.sub(r"\s+", " ", objeto).strip(" ?¿.,!¡;:")
    return objeto or None


_PALABRAS_DOMINIO_INVENTARIO = {
    "inventario", "almacen", "almacenes", "bodega", "bodegas", "stock", "existencia", "existencias",
    "producto", "productos", "articulo", "articulos", "referencia", "referencias", "ficha",
    "movimiento", "movimientos", "prestamo", "prestamos", "devolucion", "devoluciones",
    "entrada", "entradas", "salida", "salidas", "traslado", "traslados", "categoria", "categorias",
    "catalogo", "ubicacion", "estante", "disponible", "disponibles", "agotado", "agotada", "agotados",
    "agotadas", "minimo", "unidades", "unidad", "codigo", "codigos", "barra", "barras", "empaque",
    "prestable", "prestables", "retornable", "retornables", "consumible", "consumibles",
    "reponer", "reposicion",
}


def _es_pregunta_conceptual_fuera_dominio(texto_pre: str, txt_norm: str) -> bool:
    """Detecta preguntas conceptuales, enciclopédicas o de funcionamiento general ajenas al inventario."""
    if not txt_norm:
        return False
    palabras = txt_norm.split()
    if any(p in _PALABRAS_DOMINIO_INVENTARIO for p in palabras):
        return False
    # Solo valida códigos contra BD para preguntas que aún podrían ser conceptuales.
    # En consultas explícitas de inventario, el chequeo era redundante y podía hacer
    # consultas adicionales antes de llegar al fast-path o al agente.
    if extraer_codigos_de_texto(texto_pre):
        return False
    pregunta_conceptual = re.match(
            r"^(?:(?:explicame|dime|sabes|me\s+explicas|puedes\s+explicarme|cuentame|define)\s+)?"
            r"(?:que\s+(?:es|son|significa[n]?)|como\s+funciona[n]?)\s+"
            r"(?:la\s+palabra\s+|el\s+termino\s+|el\s+|la\s+|los\s+|las\s+|un\s+|una\s+|unos\s+|unas\s+)?"
            r"[a-z0-9]+(?:\s+[a-z0-9]+){0,4}$",
            txt_norm,
        )
    pregunta_externa_reconocible = re.search(
        r"(?:^|\b)(?:quien\s+(?:es|fue)\b|cual\s+es\s+la\s+capital\b|"
        r"como\s+(?:esta|va)\s+el\s+clima\b|que\s+hora\s+es\b|"
        r"(?:cuentame|dime|echate)\s+un\s+chiste\b|"
        r"(?:una\s+)?receta\s+de\b|a\s+que\s+hora\s+juega\b)",
        txt_norm,
    )
    return bool(pregunta_conceptual or pregunta_externa_reconocible)


def _planificar_consulta_ubicacion_explicita(texto: str) -> Optional[PlanAgente]:
    """Extrae consultas inequívocas de ubicación con el nombre del producto en literal.

    El patrón describe la forma de la pregunta, nunca nombres del inventario. La búsqueda
    posterior y la autorización siguen en HerramientasInventario contra la base vigente.
    """
    texto_pre = preprocesar_texto_inventario(
        texto or "", normalizar_medidas=False,
    ).strip().lstrip("¿¡ \t\r\n")
    texto_pre = texto_pre.rstrip("?!.¡¿ \t\r\n")
    # Las formas de ubicación proceden del mismo vocabulario de campo que usa
    # el resto del planificador. Así una pregunta por «en qué estante...» se
    # mapea al atributo canónico `ubicacion` aunque no diga literalmente esa
    # etiqueta; el objeto consultado sigue siendo el texto del usuario.
    lugares = "|".join(
        re.escape(termino)
        for termino in sorted(_TERMINOS_UBICACION_GENERICOS, key=len, reverse=True)
    )
    patron_lugar = re.compile(
        rf"^en\s+(?:que|cual)\s+(?:(?:el|la|los|las)\s+)?(?:{lugares})\s+"
        r"(?:(?:se\s+)?(?:encuentra|esta|estan|ubica|guarda|almacena|queda|quedan))\s+(.+)$"
    )
    texto_normalizado = normalizar(texto_pre)
    coincidencia_lugar = patron_lugar.match(texto_normalizado)
    marco_almacen_explicito = bool(re.match(
        r"^en\s+(?:que|cual)\s+(?:(?:el|la|los|las)\s+)?(?:almacen|bodega)\b",
        texto_normalizado,
    ))
    if coincidencia_lugar and not marco_almacen_explicito:
        objeto = coincidencia_lugar.group(1).strip()
        objeto = re.sub(r"^(?:el|la|los|las|un|una|unos|unas)\s+", "", objeto)
        objeto = re.sub(r"^(?:producto|articulo|referencia|elemento)\s+", "", objeto).strip()
        if objeto and normalizar(objeto) not in _TERMINOS_UBICACION_GENERICOS:
            return PlanAgente(
                tipo="consulta",
                operacion="detalle",
                filtros=FiltrosConsulta(texto=objeto),
                referencia="ninguna",
                campos=["ubicacion"],
                fuente_plan="fast_path_ubicacion_generica",
            )
    patrones = (
        (
            re.compile(
                r"^en\s+(?:qu[eé]|cu[aá]l)\s+(?:(?:el\s+)?(?:almac[eé]n|bodega))"
                r"\s+y\s+(?:la\s+)?ubicaci[oó]n\s+"
                r"(?:(?:se\s+)?(?:encuentra|est[aá]n?|ubica|guarda|almacena))\s+(.+)$",
                re.IGNORECASE,
            ),
            ["almacen", "ubicacion"],
        ),
        (
            re.compile(
                r"^en\s+(?:qu[eé]|cu[aá]l)\s+(?:(?:el\s+)?(?:almac[eé]n|bodega))\s+"
                r"(?:(?:se\s+)?(?:encuentra|est[aá]n?|ubica|guarda|almacena))\s+(.+)$",
                re.IGNORECASE,
            ),
            ["almacen"],
        ),
        (
            re.compile(
                r"^en\s+(?:qu[eé]|cu[aá]l)\s+(?:la\s+)?ubicaci[oó]n\s+"
                r"(?:(?:se\s+)?(?:encuentra|est[aá]n?|ubica|guarda))\s+(.+)$",
                re.IGNORECASE,
            ),
            ["ubicacion"],
        ),
        (
            re.compile(
                r"^d[oó]nde\s+(?:(?:se\s+)?(?:encuentra|est[aá]n?|ubica|guarda))\s+(.+)$",
                re.IGNORECASE,
            ),
            ["ubicacion", "almacen"],
        ),
    )

    for patron, campos in patrones:
        coincidencia = patron.match(texto_pre)
        if not coincidencia:
            continue
        objeto = coincidencia.group(1).strip().strip("?!.¡¿ ")
        objeto = re.sub(r"^(?:el|la|los|las|un|una|unos|unas)\s+", "", objeto, flags=re.IGNORECASE)
        objeto = re.sub(
            r"^(?:producto|art[ií]culo|referencia|elemento)\s+",
            "",
            objeto,
            flags=re.IGNORECASE,
        ).strip()
        normalizado = normalizar(objeto)
        if not objeto or normalizado in {"producto", "articulo", "referencia", "inventario", "almacen", "bodega"}:
            return None
        # Una ubicación de filtro adicional requiere interpretar alcance/contexto; deja ese caso al modelo.
        if re.search(r"\b(?:en|dentro\s+de)\s+(?:(?:el|la)\s+)?(?:almac[eé]n|bodega|ubicaci[oó]n)\b", objeto, re.IGNORECASE):
            return None
        return PlanAgente(
            tipo="consulta",
            operacion="detalle",
            filtros=FiltrosConsulta(texto=objeto),
            referencia="ninguna",
            campos=campos,
            fuente_plan="fast_path_ubicacion_explicita",
        )
    return None


def _planificar_consulta_stock_explicita(texto: str) -> Optional[PlanAgente]:
    """Extrae preguntas simples de existencias cuando el producto está nombrado literalmente."""
    texto_pre = preprocesar_texto_inventario(
        texto or "", normalizar_medidas=False,
    ).strip().lstrip("¿¡ \t\r\n")
    texto_pre = texto_pre.rstrip("?!.¡¿ \t\r\n")
    if re.search(
        r"\b(?:pr[eé]st\w*|devolv\w*|devoluci\w*|devuelt\w*|regres\w*|pendient\w*|"
        r"terceros|usuarios|custodia|fuera|movimient\w*|historial|m[ií]nim\w*|"
        r"original\w*|inicial\w*|ayer|semana|mes|a[ñn]o)\b",
        texto_pre,
        re.IGNORECASE,
    ):
        return None
    patrones = (
        r"^cu[aá]nt[oa]\s+(?:stock|existencias?)\s+(?:(?:disponible|actual)\s+)?(?:hay|queda|quedan|tenemos|tiene)\s+(?:de|del|para)\s+(.+)$",
        r"^cu[aá]nt[oa]\s+(?:stock|existencias?)\s+(?:tiene|queda|hay)\s+(.+)$",
        r"^cu[aá]nt[oa]s?\s+unidades?\s+(?:(?:disponibles?|en\s+stock)\s+)?(?:hay|queda|quedan|tenemos|tienen)\s+(?:de|del|para)\s+(.+)$",
        r"^cu[aá]nt[oa]s?\s+unidades?\s+(?:(?:disponibles?|en\s+stock)\s+)?(?:de|del)\s+(.+?)\s+(?:hay|queda|quedan|tenemos|tienen)$",
        r"^cu[aá]nt[oa]\s+hay\s+(?:de|del)\s+(.+)$",
        r"^cu[aá]nt[oa]\s+(?:queda|quedan)\s+(?:de|del)\s+(.+)$",
        r"^(?:y\s+)?(?:de|del)\s+(.+?)[,\s]+cu[aá]nt[oa]s?\s+(?:(?:stock|existencias?|unidades?(?:\s+disponibles?)?)\s+)?(?:hay|queda|quedan|tenemos|tiene)$",
    )
    for patron in patrones:
        coincidencia = re.match(patron, texto_pre, flags=re.IGNORECASE)
        if not coincidencia:
            continue
        objeto = coincidencia.group(1).strip().strip("?!.¡¿ ")
        objeto = re.sub(r"^(?:el|la|los|las|un|una|unos|unas)\s+", "", objeto, flags=re.IGNORECASE)
        objeto = re.sub(r"^(?:producto|art[ií]culo|referencia|elemento)\s+", "", objeto, flags=re.IGNORECASE).strip()
        if not objeto or normalizar(objeto) in {"producto", "articulo", "referencia", "inventario", "stock", "existencia", "existencias"}:
            return None
        # Si se especifica un almacén como alcance, el modelo debe interpretar ese filtro.
        if re.search(r"\b(?:en|dentro\s+de)\s+(?:(?:el|la)\s+)?(?:almac[eé]n|bodega)\b", objeto, re.IGNORECASE):
            return None
        return PlanAgente(
            tipo="consulta",
            operacion="detalle",
            filtros=FiltrosConsulta(texto=objeto),
            referencia="ninguna",
            campos=["stock"],
            medida_solicitada="disponible",
            fuente_plan="fast_path_stock_explicito",
        )
    m_plural = re.match(
        r"^cu[aá]nt[oa]s\s+(.+?)\s+(?:hay|queda|quedan|tenemos|tienen)(?:\s+(?:disponibles?|en\s+stock|en\s+total))?$",
        texto_pre,
        flags=re.IGNORECASE,
    )
    if m_plural:
        obj_pl = m_plural.group(1).strip().strip("?!.¡¿ ")
        obj_pl_norm = normalizar(obj_pl)
        palabras_pl = obj_pl_norm.split()
        genericos_pl = {
            "producto", "productos", "articulo", "articulos", "referencia", "referencias",
            "item", "items", "unidad", "unidades", "equipo", "equipos", "insumo", "insumos",
            "material", "materiales", "categoria", "categorias", "almacen", "almacenes",
            "bodega", "bodegas", "ubicacion", "ubicaciones", "estante", "estantes",
            "prestable", "prestables", "consumible", "consumibles", "retornable", "retornables",
            "agotado", "agotados", "planeta", "planetas", "pais", "paises", "dia", "dias",
        }
        if (
            obj_pl
            and 1 <= len(palabras_pl) <= 4
            and not any(w in genericos_pl for w in palabras_pl)
            and not re.search(r"\b(?:almac[eé]n|bodega|categor[ií]a|estante|ubicaci[oó]n|sistema\s+solar)\b", obj_pl, re.IGNORECASE)
            and re.search(r"\b(?:disponibles?|en\s+stock|en\s+total|quedan)\b", texto_pre, re.IGNORECASE)
        ):
            return PlanAgente(
                tipo="consulta",
                operacion="sumar_unidades",
                filtros=FiltrosConsulta(texto=obj_pl),
                referencia="ninguna",
                campos=["stock"],
                medida_solicitada="disponible",
                alcance_entidad="coleccion",
                fuente_plan="fast_path_stock_explicito",
            )
    return None


def requiere_catalogo_categorias_condicion_stock(texto: str) -> bool:
    """Indica cuándo un filtro explícito de stock o eventos también necesita resolver una categoría viva."""
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    condicion_stock = bool(
        _condiciones_stock_mencionadas(texto_norm)
        or re.search(
            r"\bstock\s+minimo\b|\bagotad\w*\b|\bsin\s+(?:stock|existencias|unidades)\b|"
            r"\b(?:movimientos?|prestamos?|devoluciones?|entradas?|salidas?|dotaciones?|traslados?)\b",
            texto_norm,
        )
    )
    marco_categoria = bool(
        re.search(
            r"\bcategor[ií]a\b|"
            r"\b(?:productos?|articulos?|referencias?)(?:\s+(?:bajo|en|cerca\s+de(?:l)?)\s+(?:(?:el|su)\s+)?(?:stock\s+)?minimo)?\s+(?:de|del)\s+\w+",
            texto_norm,
        )
    )
    return condicion_stock and marco_categoria


def requiere_catalogo_almacenes_condicion_stock(texto: str) -> bool:
    """Indica cuándo la consulta nombra un almacén/bodega y necesita el catálogo de almacenes accesibles."""
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    return bool(re.search(r"\b(?:almacen|almacenes|bodega|bodegas)\b", texto_norm))


_RE_NEGACION_PREVIA_CONDICION = re.compile(
    r"(?:^|\b)(?:no|ni|tampoco|sin\s+estar)(?:\s+(?:est[ae]n?|est[ae]|tengan?|tienen?|sean?|son|queden?|anden?)){0,2}\s*$"
)


def _condiciones_stock_mencionadas(texto: str) -> set[str]:
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    encontradas: set[str] = set()
    for condicion, patron in _PATRONES_CONDICION_STOCK.items():
        for coincidencia in re.finditer(patron, texto_norm):
            ventana_previa = texto_norm[max(0, coincidencia.start() - 36):coincidencia.start()]
            if _RE_NEGACION_PREVIA_CONDICION.search(ventana_previa):
                continue
            encontradas.add(condicion)
            break
    return encontradas


def _condicion_bajo_minimo_gobernada_por_infinitivo(texto: str) -> bool:
    """Detecta si la relación con el umbral se expresa como trayectoria futura.

    Se basa en la estructura temporal y el infinitivo, no en verbos o productos
    enumerados. Un estado buscado antes de alcanzar un umbral no es un déficit
    presente; las afirmaciones independientes de estar ya por debajo se dejan
    al clasificador semántico.
    """
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    patron = _PATRONES_CONDICION_STOCK["bajo_minimo"]
    relaciones = list(re.finditer(patron, texto_norm))
    if relaciones:
        if len(relaciones) != 1:
            return False
        inicio = relaciones[0].start()
        ventana_previa = texto_norm[max(0, inicio - 90):inicio]
        return bool(re.search(
            r"\b(?:de|para|al)\s+(?:\w+\s+){0,3}[a-záéíóúñ]+(?:ar|er|ir)\s*$",
            ventana_previa,
        ))

    # También se expresa proximidad como una pequeña holgura antes de llegar
    # al límite, sin nombrar un estado “bajo” ni usar el sustantivo “stock”.
    # El patrón captura la relación temporal y la forma infinitiva; el término
    # del umbral se valida aparte para no afectar frases sin relación con stock.
    marcador_temporal = re.compile(
        r"\b(?:antes\s+de|previo\s+a|previa\s+a)\s+"
        r"(?:(?:\w+)\s+){0,2}[a-záéíóúñ]+(?:ar|er|ir)\b"
    )
    for coincidencia in marcador_temporal.finditer(texto_norm):
        ventana_umbral = texto_norm[coincidencia.end():coincidencia.end() + 56]
        if re.search(r"\b(?:minimo|umbral|limite|nivel)\b", ventana_umbral):
            return True
    return False


def _producto_es_entidad_concreta(valor: Optional[str]) -> bool:
    """Ignora marcadores de ausencia de entidad y placeholders del planificador."""
    valor_normalizado = normalizar(valor or "")
    return bool(valor_normalizado) and valor_normalizado not in {
        "indefinido", "desconocido", "sin especificar", "ninguno", "ninguna",
        "no identificado", "no especificado",
    }


def _recuperar_descriptor_producto_grounded(texto: str, plan: PlanAgente) -> PlanAgente:
    """Restaura un descriptor omitido solo cuando el mensaje lo respalda.

    El modelo decide si la tarea es una consulta. Esta reparación conserva el
    sintagma de producto que el propio plan extrajo o, cuando el usuario nombró
    explícitamente el inventario, recupera el objeto del mensaje original.
    La herramienta resuelve después el descriptor contra el catálogo vivo.
    """
    if (
        plan.tipo != "consulta"
        or plan.operacion not in {"listar", "contar", "sumar_unidades", "detalle"}
        or plan.referencia != "ninguna"
        or _es_consulta_global_inventario(texto)
        or any((
            plan.filtros.codigos,
            plan.filtros.categoria,
            plan.filtros.categoria_explicita,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
            plan.filtros.condicion not in (None, "con_stock"),
            plan.filtros.es_prestable is not None,
        ))
    ):
        return plan

    tokens_mensaje = set(tokens_significativos(texto or ""))
    if plan.filtros.texto:
        tokens_filtro_existente = set(tokens_significativos(plan.filtros.texto))
        plan.descriptor_producto_grounded = bool(
            tokens_filtro_existente
            and tokens_filtro_existente.issubset(tokens_mensaje)
            and not tokens_filtro_existente.issubset(_PALABRAS_DOMINIO_INVENTARIO)
            and not _es_filtro_marco_global(plan.filtros.texto)
        )
        return plan

    descriptor = plan.producto if _producto_es_entidad_concreta(plan.producto) else None
    tokens_descriptor = set(tokens_significativos(descriptor or ""))
    if tokens_descriptor and tokens_descriptor.issubset(tokens_mensaje):
        # No convertir etiquetas de campo o alcance en filtros de producto.
        if not tokens_descriptor.issubset(_PALABRAS_DOMINIO_INVENTARIO):
            plan.filtros.texto = str(descriptor).strip()
            plan.descriptor_producto_grounded = True
            return plan

    # Si el modelo omitió el nombre, solo reconstruye desde el mensaje cuando
    # la persona ancló explícitamente la consulta al dominio del inventario.
    # Esto conserva consultas negativas (p. ej. un tipo de artículo inexistente)
    # sin reinterpretar preguntas generales como búsquedas de stock.
    tokens_mensaje_crudos = set(normalizar(preprocesar_texto_inventario(texto or "")).split())
    consulta_cuantitativa_entidad = bool(
        plan.operacion in {"sumar_unidades", "detalle"}
        and re.search(r"\bcuant[oa]s?\b", normalizar(texto or ""))
        and re.search(r"\b(?:queda[n]?|hay|tenemos|tiene[n]?|libre[s]?)\b", normalizar(texto or ""))
    )
    if not (tokens_mensaje_crudos & _PALABRAS_DOMINIO_INVENTARIO) and not consulta_cuantitativa_entidad:
        return plan
    objeto = _extraer_objeto_consulta_sin_marco(texto)
    tokens_objeto = set(tokens_significativos(objeto or ""))
    if objeto and tokens_objeto and not tokens_objeto.issubset(_PALABRAS_DOMINIO_INVENTARIO):
        plan.filtros.texto = objeto
        plan.descriptor_producto_grounded = True
        if plan.filtros.condicion == "con_stock":
            plan.filtros.condicion = None
    return plan


def _relacion_stock_explicita_sin_ambiguedad(
    texto: str,
    plan: PlanAgente,
) -> Optional[str]:
    """Devuelve la relación tipada solo para condiciones literales inequívocas.

    Este camino evita una segunda inferencia cuando el plan ya conserva una
    condición estructural reconocida. Las expresiones subordinadas o futuras
    siguen pasando por el clasificador semántico para no confundir trayectoria
    con estado presente.
    """
    condiciones = _condiciones_stock_mencionadas(texto)
    if len(condiciones) != 1:
        return None
    condicion = next(iter(condiciones))
    if plan.filtros.condicion != condicion:
        return None
    if (plan.medida_solicitada or "").strip().casefold() in {
        "atributo", "disponible", "ambos", "pendiente_devolucion",
    } or plan.alcance in {"evento", "ambos"}:
        return None

    texto_normalizado = normalizar(preprocesar_texto_inventario(texto or ""))
    tramos_condicion = [
        coincidencia.span()
        for nombre, patron in _PATRONES_CONDICION_STOCK.items()
        if nombre in condiciones
        for coincidencia in re.finditer(patron, texto_normalizado)
    ]
    texto_fuera_de_condicion = list(texto_normalizado)
    for inicio, fin in tramos_condicion:
        texto_fuera_de_condicion[inicio:fin] = " " * (fin - inicio)
    texto_fuera_de_condicion = "".join(texto_fuera_de_condicion)
    for campo in set(plan.campos or []) & {"stock", "stock_minimo"}:
        etiqueta = normalizar(campo.replace("_", " "))
        if etiqueta and re.search(rf"(?<!\w){re.escape(etiqueta)}(?!\w)", texto_fuera_de_condicion):
            return None

    subordinacion_temporal_o_condicional = re.search(
        r"\b(?:antes(?:\s+de)?|despu[eé]s(?:\s+de)?|si|cuando|mientras|"
        r"hasta(?:\s+que)?|a\s+medida\s+que|podr[ií]a[n]?|tendencia)\b",
        texto_normalizado,
    )
    condicion_inicio = min((inicio for inicio, _ in tramos_condicion), default=len(texto_normalizado))
    ventana_previa = texto_normalizado[max(0, condicion_inicio - 90):condicion_inicio]
    verbo_no_finito_previo = re.search(
        r"\b(?:de|para|al)\s+(?:\w+\s+){0,3}[a-záéíóúñ]+(?:ar|er|ir)\s*$",
        ventana_previa,
    )
    if subordinacion_temporal_o_condicional or verbo_no_finito_previo:
        return None

    return next(
        (
            relacion
            for relacion, condicion_mapeada in _RELACION_A_CONDICION_STOCK.items()
            if condicion_mapeada == condicion
        ),
        None,
    )


def _limpiar_filtro_repetido_condicion_stock(
    texto: str,
    plan: PlanAgente,
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> PlanAgente:
    """Quita del filtro de producto los términos que solo repiten una condición explícita.

    En preguntas colectivas, el LLM puede copiar al filtro una palabra del
    umbral que también aparece en el nombre de un candidato del contexto. El
    filtro se conserva si aporta algún término distinto, o si hay una
    referencia individual/código explícito.
    """
    if (
        plan.tipo != "consulta"
        or not plan.filtros.condicion
        or not plan.filtros.texto
        or plan.filtros.codigos
        or plan.referencia in {"foco", "resultados_previos"}
        or not (
            plan.alcance_entidad == "coleccion"
            or _es_consulta_global_inventario(texto)
        )
    ):
        return plan

    if (
        plan.alcance_entidad in {None, "coleccion", "global", "indefinido"}
        and plan.filtros.condicion in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA
        and _RELACION_A_CONDICION_STOCK.get(plan.relacion_stock_verificada)
        == plan.filtros.condicion
        and codigos_productos_candidatos is not None
        and not any((
            plan.filtros.categoria,
            plan.filtros.categoria_explicita,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
        ))
    ):
        # Una relación de stock ya validada por el clasificador no debe quedar
        # restringida por palabras residuales de la pregunta. Conserva el
        # filtro solo si se puede anclar a una referencia o nombre vigente del
        # catálogo; de ese modo, los nombres reales siguen siendo consultables
        # y una paráfrasis no puede convertirse en un producto inexistente.
        candidatos = _obtener_productos_candidatos_para_verificacion(
            codigos_productos_candidatos,
        )
        if not _filtro_texto_anclado_a_candidato(
            plan.filtros.texto,
            candidatos,
        ):
            plan.filtros.texto = None
            if not plan.filtros.codigos:
                plan.producto = None
            plan.alcance_entidad = "coleccion"
            return plan

    texto_normalizado = normalizar(preprocesar_texto_inventario(texto or ""))
    # El identificador validado del plan es otra descripción canónica del
    # mismo filtro; permite limpiar eco léxico incluso si la paráfrasis no
    # coincide con una expresión local de la condición.
    tokens_condicion = set(tokens_significativos(plan.filtros.condicion.replace("_", " ")))
    for patron in _PATRONES_CONDICION_STOCK.values():
        for coincidencia in re.finditer(patron, texto_normalizado):
            tokens_condicion.update(tokens_significativos(coincidencia.group(0)))
    tokens_filtro = set(tokens_significativos(plan.filtros.texto.replace("_", " ")))
    if not tokens_filtro or not tokens_filtro.issubset(tokens_condicion):
        return plan

    plan.filtros.texto = None
    codigos_usuario = set(extraer_codigos_de_texto(texto))
    codigos_en_producto = set(extraer_codigos_de_texto(plan.producto or ""))
    if not (codigos_usuario & codigos_en_producto):
        plan.producto = None
    plan.alcance_entidad = "coleccion"
    return plan


def _condicion_validada_propuesta_por_modelo(plan: PlanAgente) -> bool:
    """Indica si el plan estructurado conserva una condición que venía del LLM.

    Las reglas léxicas son útiles cuando el LLM omite el filtro, pero no deben
    invalidar una clasificación semántica válida solo porque la paráfrasis no
    coincide con los patrones locales.
    """
    relacion_verificada = getattr(plan, "relacion_stock_verificada", None)
    if (
        relacion_verificada in _RELACION_A_CONDICION_STOCK
        and _RELACION_A_CONDICION_STOCK[relacion_verificada] == plan.filtros.condicion
    ):
        return True
    if not isinstance(plan.crudo, dict):
        return False
    filtros_crudos = plan.crudo.get("filtros")
    if not isinstance(filtros_crudos, dict):
        return False
    condicion_cruda = _str(filtros_crudos.get("condicion"))
    return bool(condicion_cruda and condicion_cruda == plan.filtros.condicion)


def _normalizar_union_condiciones_stock(texto: str, plan: PlanAgente) -> PlanAgente:
    """Conserva una disyunción de condiciones cuando el esquema ofrece su unión canónica."""
    if (
        plan.tipo != "consulta"
        or plan.operacion not in {"listar", "contar"}
        or plan.alcance in {"evento", "ambos"}
    ):
        return plan

    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    menciones = sorted(
        (match.start(), match.end(), condicion)
        for condicion, patron in _PATRONES_CONDICION_STOCK.items()
        for match in re.finditer(patron, texto_norm)
    )
    for indice, (_, fin, condicion_a) in enumerate(menciones):
        for inicio, _, condicion_b in menciones[indice + 1:]:
            if {condicion_a, condicion_b} != {"agotado", "bajo_minimo"}:
                continue
            if re.search(r"\b(?:o|u|or)\b", texto_norm[fin:inicio]):
                plan.filtros.condicion = "requiere_reposicion"
                return plan
    return plan


def _normalizar_listado_colectivo_condicionado(plan: PlanAgente) -> PlanAgente:
    """Resolve a complete collection predicate when no measure was requested.

    A model may mark a typed stock or catalog-property filter as `aclaracion`
    because it mistakes the absence of a requested quantity for ambiguity.
    The filter, collection scope, and empty measure already describe a complete
    read-only listing request.
    """
    if (
        plan.tipo == "consulta"
        and plan.operacion == "sumar_unidades"
        and plan.filtros.condicion in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA
        and not plan.campos
        and not plan.filtros.codigos
        and not plan.filtros.texto
    ):
        plan.operacion = "listar"
        plan.alcance_entidad = "coleccion"
        plan.medida_solicitada = "ninguna"
        return plan
    if not (
        plan.tipo == "aclaracion"
        and (
            plan.filtros.condicion in _CONDICIONES_STOCK_PARA_CLASIFICAR
            or plan.filtros.es_prestable is not None
        )
        and plan.alcance_entidad in {"coleccion", "global"}
        and plan.medida_solicitada in {None, "", "ninguna", "producto"}
        and not plan.campos
        and not plan.accion
    ):
        return plan

    plan.tipo = "consulta"
    plan.operacion = "listar"
    plan.campos = []
    plan.medida_solicitada = "ninguna"
    plan.alcance = "actual"
    plan.referencia = "ninguna"
    plan.pregunta = None
    return plan


def _normalizar_condicion_stock_explicita(
    texto: str,
    plan: PlanAgente,
    categorias_accesibles: Optional[List[str]] = None,
    almacenes_accesibles: Optional[List[str]] = None,
) -> PlanAgente:
    """Reaplica al plan validado los filtros de inventario expresos del mensaje.

    El modelo decide la intención general; esta capa conserva los valores
    tipados del dominio si la generación omite una condición literal.
    """
    if plan.tipo != "consulta":
        return plan

    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    generica = bool(re.search(r"\b(?:lo|los|las)\s+que\b", texto_norm))
    condiciones = _condiciones_stock_mencionadas(texto)
    if not condiciones and _PATRON_REPOSICION_EXPLICITA.search(texto_norm):
        condiciones = {"requiere_reposicion"}
    if not condiciones:
        return plan

    # Las referencias a la conversación o a un código identifican un producto
    # concreto y no deben convertirse en un listado global.
    if plan.referencia in {"foco", "resultados_previos"} or plan.filtros.codigos:
        return plan

    if plan.operacion not in {"listar", "contar", "sumar_unidades"}:
        if not generica and (
            plan.alcance_entidad == "producto"
            or _producto_es_entidad_concreta(plan.producto)
            or (plan.operacion == "detalle" and bool(plan.filtros.texto))
        ):
            if (
                "stock_minimo" in (plan.atributos_stock_verificados or [])
                and "stock_minimo" not in (plan.campos or [])
            ):
                plan.operacion = "detalle"
                plan.campos = list(dict.fromkeys([*(plan.campos or []), "stock_minimo"]))
                plan.medida_solicitada = "atributo"
                plan.alcance_entidad = "producto"
                plan.alcance = "actual"
            return plan
        plan.operacion = "listar"
        plan.referencia = "ninguna"
        if generica:
            plan.campos = ["stock"]
            plan.filtros.texto = None
        plan.filtros.codigos = []

    if len(condiciones) == 1 and not _condicion_bajo_minimo_gobernada_por_infinitivo(texto):
        condicion_unica = next(iter(condiciones))
        if (
            condicion_unica == "minimo_cero"
            and not generica
            and "stock_minimo" in (plan.atributos_stock_verificados or [])
            and (
                bool(plan.filtros.texto)
                or bool(plan.filtros.codigos)
                or _producto_es_entidad_concreta(plan.producto)
            )
        ):
            plan.operacion = "detalle"
            plan.campos = list(dict.fromkeys([*(plan.campos or []), "stock_minimo"]))
            plan.medida_solicitada = "atributo"
            plan.alcance_entidad = "producto"
            plan.alcance = "actual"
            plan.filtros.condicion = None
            return plan
        if (
            not _condicion_validada_propuesta_por_modelo(plan)
            or plan.filtros.condicion in {None, "con_stock"}
            or (condicion_unica == "bajo_minimo" and plan.filtros.condicion == "agotado")
        ):
            # Completa filtros omitidos por el modelo (o una positividad genérica
            # con_stock, o un agotado erróneo cuando el mensaje expresa déficit
            # positivo), sin contradecir una coincidencia subordinada a infinitivo.
            plan.filtros.condicion = condicion_unica
    if (
        generica
        and len(condiciones) == 1
        and not re.search(
            r"\b(?:cu[aá]nt\w*|cantidad(?:es)?|unidades?|total(?:izar|es)?)\b",
            texto_norm,
        )
        and plan.alcance_entidad != "producto"
    ):
        # Un pedido de ver los registros que cumplen una condición explícita
        # conserva forma de listado aunque el plan preliminar lo haya reducido
        # a una suma o haya inventado la medida de préstamos.
        plan.operacion = "listar"
        plan.campos = []
        plan.medida_solicitada = "ninguna"
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
    if generica:
        plan.filtros.texto = None
    return plan


def _obtener_productos_candidatos_para_verificacion(
    codigos_productos_candidatos: Optional[List[str]],
) -> List[Tuple[str, str]]:
    """Carga etiquetas vivas del catálogo para validar un filtro preliminar."""
    codigos = list(dict.fromkeys(
        str(codigo).strip()
        for codigo in (codigos_productos_candidatos or [])
        if str(codigo or "").strip()
    ))
    if not codigos:
        return []
    try:
        from inventario.models import Producto
        return list(
            Producto.objects.filter(
                codigo_interno__in=codigos,
                activo=True,
                archivado=False,
            ).values_list("codigo_interno", "nombre")
        )
    except Exception:
        logger.debug("No se pudo cargar el catálogo candidato para validar el filtro", exc_info=True)
        return []


def _filtro_texto_anclado_a_candidato(
    filtro_texto: Optional[str],
    productos_candidatos: List[Tuple[str, str]],
) -> bool:
    """Comprueba si el filtro preliminar tiene un anclaje actual a una entidad candidata."""
    filtro_norm = normalizar(filtro_texto or "")
    if not filtro_norm:
        return False
    codigos_filtro = {
        normalizar_codigo_producto(codigo)
        for codigo in extraer_codigos_de_texto(filtro_texto)
    }
    tokens_filtro = set(tokens_significativos(filtro_norm))
    compacto_filtro = re.sub(r"[^a-z0-9]", "", filtro_norm)
    for codigo, nombre in productos_candidatos:
        codigo_norm = normalizar_codigo_producto(codigo)
        if codigo_norm and codigo_norm in codigos_filtro:
            return True
        compacto_codigo = re.sub(r"[^a-z0-9]", "", normalizar(codigo))
        if compacto_codigo and compacto_codigo == compacto_filtro:
            return True
        nombre_norm = normalizar(nombre)
        if nombre_norm and (filtro_norm in nombre_norm or nombre_norm in filtro_norm):
            return True
        tokens_candidato = set(tokens_significativos(f"{codigo} {nombre}"))
        if tokens_filtro and tokens_filtro.issubset(tokens_candidato):
            return True
    return False


def _anclar_producto_propuesto_vigente(
    texto: str,
    plan: PlanAgente,
    codigos_productos_candidatos: Optional[List[str]],
    telemetria: Optional[Dict[str, Any]] = None,
) -> PlanAgente:
    """Grounds the primary model's proposed code in the live catalog candidates."""
    if (
        plan.tipo != "consulta"
        or plan.operacion not in {"listar", "contar", "sumar_unidades", "detalle"}
        or not plan.producto
        or not codigos_productos_candidatos
        or _es_consulta_global_inventario(texto or "")
    ):
        return plan

    codigo_propuesto = normalizar_codigo_producto(plan.producto)
    if not codigo_propuesto:
        return plan
    candidatos = _obtener_productos_candidatos_para_verificacion(codigos_productos_candidatos)
    seleccionado = next(
        (
            (codigo, nombre)
            for codigo, nombre in candidatos
            if normalizar_codigo_producto(codigo) == codigo_propuesto
        ),
        None,
    )
    if not seleccionado:
        return plan

    codigo, nombre = seleccionado
    codigos_mencionados = {
        normalizar_codigo_producto(valor)
        for valor in extraer_codigos_de_texto(texto or "")
    }
    texto_normalizado = normalizar(preprocesar_texto_inventario(texto or ""))
    nombres_mencionados = {
        codigo_candidato
        for codigo_candidato, nombre_candidato in candidatos
        if normalizar(nombre_candidato) and normalizar(nombre_candidato) in texto_normalizado
    }
    if codigos_mencionados and codigo_propuesto not in codigos_mencionados:
        return plan
    if nombres_mencionados and codigo not in nombres_mencionados:
        return plan
    if plan.filtros.codigos and codigo not in plan.filtros.codigos:
        return plan
    if (
        plan.filtros.texto
        and _filtro_texto_anclado_a_candidato(plan.filtros.texto, candidatos)
        and not _filtro_texto_anclado_a_candidato(plan.filtros.texto, [(codigo, nombre)])
    ):
        return plan

    if codigo not in codigos_mencionados and codigo not in nombres_mencionados:
        descriptor_mencionado = _extraer_objeto_consulta_sin_marco(texto or "")
        coincidencias_descriptor = [
            codigo_candidato
            for codigo_candidato, nombre_candidato in candidatos
            if _filtro_texto_anclado_a_candidato(
                descriptor_mencionado,
                [(codigo_candidato, nombre_candidato)],
            )
        ]
        if len(set(coincidencias_descriptor)) > 1:
            # El mensaje solo respalda un descriptor compartido por varios
            # productos vivos. La elección del modelo no convierte esa
            # mención en una referencia única; el resolutor posterior debe
            # conservar los candidatos y pedir precisión.
            return plan

    tokens_mensaje = set(tokens_significativos(texto_normalizado))
    tokens_producto = set(tokens_significativos(f"{codigo} {nombre}"))
    if not tokens_mensaje.intersection(tokens_producto):
        return plan

    plan.filtros.codigos = [codigo]
    plan.filtros.texto = None
    plan.producto = nombre
    plan.alcance_entidad = "producto"
    plan.codigo_catalogo_verificado = True
    if telemetria is not None:
        telemetria["resolucion_entidad_catalogo"] = {
            "codigo": codigo,
            "fuente": "codigo_propuesto_por_llm_validado_en_catalogo_vigente",
        }
    return plan


def _relacion_stock_de_condicion_plan_primario(
    plan: PlanAgente,
    codigos_productos_candidatos: Optional[List[str]],
    texto: Optional[str] = None,
) -> Optional[str]:
    """Reutiliza una condición primaria solo si el mensaje la expresa sin ambigüedad.

    El plan primario puede confundir una paráfrasis de déficit presente con
    riesgo futuro. La condición propuesta por ese plan no constituye evidencia
    lingüística; las expresiones no literales pasan al clasificador semántico.
    """
    if (
        plan.tipo != "consulta"
        or plan.operacion not in {"listar", "contar", "sumar_unidades"}
        or plan.alcance_entidad not in {"coleccion", "global"}
        or plan.referencia in {"foco", "resultados_previos"}
        or plan.filtros.codigos
        or normalizar(plan.producto or "") not in {"", "ninguno", "none", "null"}
    ):
        return None

    relacion_explicita = (
        _relacion_stock_explicita_sin_ambiguedad(texto, plan)
        if texto is not None
        else None
    )

    relacion_tipada = getattr(plan, "relacion_stock_verificada", None)
    if relacion_tipada in RELACIONES_STOCK_VALIDAS:
        # El esquema nuevo obliga al LLM a declarar su relación semántica. Si
        # esa declaración contradice el filtro, no permitimos que la condición
        # cruda reemplace silenciosamente el dato contradictorio y evite la
        # clasificación de respaldo.
        if (
            _RELACION_A_CONDICION_STOCK.get(relacion_tipada) != plan.filtros.condicion
            or plan.atributos_stock_verificados != []
            or (texto is not None and relacion_explicita != relacion_tipada)
        ):
            return None
        if plan.filtros.texto:
            candidatos = _obtener_productos_candidatos_para_verificacion(
                codigos_productos_candidatos,
            )
            if (
                plan.filtro_texto_es_entidad_verificado is not False
                and _filtro_texto_anclado_a_candidato(plan.filtros.texto, candidatos)
            ):
                return None
            if not plan.filtro_texto_es_entidad_verificado:
                plan.filtro_texto_es_entidad_verificado = False
        return relacion_tipada

    filtros_crudos = plan.crudo.get("filtros") if isinstance(plan.crudo, dict) else None
    condicion_cruda = _str(filtros_crudos.get("condicion")) if isinstance(filtros_crudos, dict) else ""
    if not condicion_cruda or condicion_cruda != plan.filtros.condicion:
        return None

    relacion_cruda = next(
        (
            relacion
            for relacion, condicion in _RELACION_A_CONDICION_STOCK.items()
            if condicion == condicion_cruda
        ),
        None,
    )
    if texto is not None and relacion_explicita != relacion_cruda:
        return None

    if plan.filtros.texto:
        candidatos = _obtener_productos_candidatos_para_verificacion(
            codigos_productos_candidatos,
        )
        if _filtro_texto_anclado_a_candidato(plan.filtros.texto, candidatos):
            # Una mención aterrizada en el catálogo requiere comprobar si la
            # condición se aplica a un producto/clase concretos.
            return None
        # El plan ya delimitó la consulta como colectiva y no hay referencia
        # vigente que respalde este texto. Lo tratamos como residuo del marco
        # lingüístico; el limpiador final vuelve a verificar el anclaje.
        plan.filtro_texto_es_entidad_verificado = False

    return relacion_cruda


def _verificar_condicion_stock_semantica(
    texto: str,
    cliente_llm: Any,
    plan: PlanAgente,
    timeout: float,
    telemetria: Dict[str, Any],
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> PlanAgente:
    """Clasifica el umbral de stock y recupera filtros omitidos en consultas extremales."""
    plan = _anclar_producto_propuesto_vigente(
        texto,
        plan,
        codigos_productos_candidatos,
        telemetria,
    )

    relacion_plan_primario = _relacion_stock_de_condicion_plan_primario(
        plan,
        codigos_productos_candidatos,
        texto,
    )
    if relacion_plan_primario:
        plan.relacion_stock_verificada = relacion_plan_primario
        plan.atributos_stock_verificados = []
        if plan.filtros.texto:
            plan.filtro_texto_es_entidad_verificado = False
        telemetria["clasificacion_condicion"] = {
            "relacion": relacion_plan_primario,
            "condicion_inicial": plan.filtros.condicion,
            "condicion_validada": plan.filtros.condicion,
            "filtro_texto_es_entidad": False if plan.filtros.texto else None,
            "atributos_stock_solicitados": [],
            "fuente": "plan_primario_colectivo_sin_entidad_candidata",
            "omitida_por_plan_primario": True,
        }
        return plan

    condicion_inicial = plan.filtros.condicion
    campos_plan_iniciales = list(plan.campos or [])
    medida_plan_inicial = plan.medida_solicitada
    campos_stock_plan = set(campos_plan_iniciales) & {"stock", "stock_minimo"}
    consulta_coordina_entidad_con_stock = _consulta_coordina_entidad(
        texto,
        codigos_productos_candidatos=(
            codigos_productos_candidatos or list(plan.filtros.codigos or [])
        ),
        tiene_foco=(
            plan.referencia == "foco"
            or plan.alcance_entidad == "producto"
            or bool(plan.filtros.codigos)
            or bool(plan.filtros.texto)
            or bool(plan.filtros.categoria_explicita)
            or _producto_es_entidad_concreta(plan.producto)
        ),
    )
    consulta_cuantitativa_compuesta_con_stock = bool(
        plan.tipo == "consulta"
        and plan.operacion in {"detalle", "listar", "contar", "sumar_unidades"}
        and bool(campos_stock_plan)
        and (
            len(set(campos_plan_iniciales)) > 1
            or consulta_coordina_entidad_con_stock
        )
        and plan.alcance not in {"evento", "ambos"}
    )
    menciona_umbral = bool(re.search(
        r"\b(?:minim\w*|umbral\w*|reposicion\w*)\b",
        normalizar(preprocesar_texto_inventario(texto or "")),
    ))
    operacion_colectiva = plan.operacion in {"listar", "contar", "sumar_unidades"}
    comparacion_de_coleccion = bool(
        plan.tipo == "consulta"
        and plan.operacion == "comparar"
        and len(plan.ubicaciones_comparadas or []) < 2
        and not plan.filtros.codigos
        and not plan.filtros.categoria
        and not plan.filtros.almacen
        and not plan.filtros.ubicacion
        and not _producto_es_entidad_concreta(plan.producto)
    )
    detalle_colectivo_con_umbral = bool(
        plan.tipo == "consulta"
        and plan.operacion == "detalle"
        and menciona_umbral
        and not plan.filtros.codigos
        and not _producto_es_entidad_concreta(plan.producto)
        and plan.referencia not in {"foco", "resultados_previos"}
        and plan.alcance_entidad != "producto"
    )
    consulta_colectiva_con_umbral = bool(
        plan.tipo == "consulta"
        and (operacion_colectiva or comparacion_de_coleccion or detalle_colectivo_con_umbral)
        and menciona_umbral
    )
    consulta_extremal_con_umbral = bool(
        plan.tipo == "consulta"
        and plan.operacion in {"mayor_stock", "menor_stock"}
        and menciona_umbral
    )
    tiene_filtro_colectivo_no_anclado = bool(
        plan.tipo == "consulta"
        and plan.operacion in {"listar", "contar", "sumar_unidades"}
        and plan.filtros.texto
        and not plan.filtros.codigos
        and not _producto_es_entidad_concreta(plan.producto)
        and plan.referencia not in {"foco", "resultados_previos"}
        and plan.alcance_entidad != "producto"
        and not any((
            plan.filtros.categoria,
            plan.filtros.categoria_explicita,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
        ))
        and codigos_productos_candidatos is not None
    )
    productos_candidatos = (
        _obtener_productos_candidatos_para_verificacion(codigos_productos_candidatos)
        if tiene_filtro_colectivo_no_anclado
        or (plan.tipo == "consulta" and (plan.producto or plan.filtros.codigos))
        else []
    )
    productos_candidatos_clasificador = productos_candidatos
    if plan.alcance_entidad == "producto" and len(plan.filtros.codigos or []) == 1:
        # Reduce el contexto al producto solo si el mensaje lo ancla por código
        # o por nombre completo del catálogo. Con un descriptor parcial se
        # conservan todos los candidatos para mantener la detección de empate.
        codigo_plan = normalizar_codigo_producto(plan.filtros.codigos[0])
        codigos_dichos = {
            normalizar_codigo_producto(codigo)
            for codigo in extraer_codigos_de_texto(texto or "")
        }
        texto_norm_entidad = normalizar(preprocesar_texto_inventario(texto or ""))
        candidato_plan = next((
            (codigo, nombre)
            for codigo, nombre in productos_candidatos
            if normalizar_codigo_producto(codigo) == codigo_plan
        ), None)
        nombre_completo_mencionado = bool(
            candidato_plan
            and normalizar(preprocesar_texto_inventario(candidato_plan[1]))
            and normalizar(preprocesar_texto_inventario(candidato_plan[1])) in texto_norm_entidad
        )
        if candidato_plan and (
            codigo_plan in codigos_dichos or nombre_completo_mencionado
        ):
            productos_candidatos_clasificador = [candidato_plan]
    consulta_colectiva_con_filtro_no_anclado = bool(
        tiene_filtro_colectivo_no_anclado
        and not _filtro_texto_anclado_a_candidato(plan.filtros.texto, productos_candidatos)
    )
    consulta_colectiva_sin_descriptor = bool(
        plan.tipo == "consulta"
        and plan.operacion in {"listar", "contar", "sumar_unidades"}
        and plan.referencia == "ninguna"
        and plan.alcance_entidad in {None, "coleccion", "global", "indefinido"}
        and not plan.alcance_global_confirmado
        and not any((
            plan.filtros.texto,
            plan.filtros.codigos,
            plan.filtros.categoria,
            plan.filtros.categoria_explicita,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
            plan.producto if _producto_es_entidad_concreta(plan.producto) else None,
        ))
    )
    consulta_colectiva_sin_entidad = bool(
        plan.tipo == "consulta"
        and plan.operacion in {"listar", "contar", "sumar_unidades"}
        and plan.alcance_entidad in {None, "coleccion", "global", "indefinido"}
        and plan.referencia not in {"foco", "resultados_previos"}
        and not plan.filtros.texto
        and not plan.filtros.codigos
        and not _producto_es_entidad_concreta(plan.producto)
    )
    if (
        plan.tipo != "consulta"
        or (
            condicion_inicial == "con_prestamos"
            and not menciona_umbral
            and not re.search(
                r"\b(?:stock|existencias?|disponibles?|agotad\w*|escas\w*|sin\s+unidades)\b",
                normalizar(preprocesar_texto_inventario(texto or "")),
            )
        )
        or not (
            (
                (operacion_colectiva or comparacion_de_coleccion or detalle_colectivo_con_umbral)
                and (
                    condicion_inicial in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA
                    or consulta_colectiva_con_umbral
                )
            )
            or consulta_extremal_con_umbral
            or consulta_colectiva_con_filtro_no_anclado
            or consulta_colectiva_sin_descriptor
            or consulta_cuantitativa_compuesta_con_stock
        )
    ):
        return plan

    relacion_estructural = _relacion_stock_explicita_sin_ambiguedad(texto, plan)
    if relacion_estructural:
        # La comprobación estructural confirma que stock/minimo aparecen
        # únicamente como condición; si el valor se pidiera como salida, la
        # comprobación habría delegado al clasificador semántico.
        plan.campos = [
            campo for campo in (plan.campos or [])
            if campo not in {"stock", "stock_minimo"}
        ]
        if plan.medida_solicitada == "atributo" and not plan.campos:
            plan.medida_solicitada = "ninguna"
        plan.relacion_stock_verificada = relacion_estructural
        telemetria["clasificacion_condicion"] = {
            "relacion": relacion_estructural,
            "condicion_inicial": condicion_inicial,
            "condicion_validada": plan.filtros.condicion,
            "fuente": "condicion_explicita_sin_ambiguedad",
            "filtro_texto_es_entidad": None,
            "atributos_stock_solicitados": [],
        }
        return plan

    texto_usuario = preprocesar_texto_inventario(texto or "", normalizar_medidas=False)[:400]
    clasificador_colectivo_compacto = consulta_colectiva_sin_entidad
    if clasificador_colectivo_compacto:
        mensajes = [
            {"role": "system", "content": PROMPT_CLASIFICADOR_RELACION_STOCK_COLECTIVO},
            {"role": "user", "content": f"Mensaje actual: {texto_usuario}"},
        ]
        esquema_relacion = copy.deepcopy(_ESQUEMA_RELACION_STOCK_COLECTIVO_JSON)
        nombre_esquema_relacion = "haku_collective_stock_relation"
        max_tokens_relacion = 64
    else:
        mensajes = [
            {"role": "system", "content": PROMPT_CLASIFICADOR_RELACION_STOCK},
            {
                "role": "user",
                "content": (
                    f"[Mensaje actual]\n{texto_usuario}\n\n"
                    f"[Filtro de texto propuesto]\n{plan.filtros.texto or '(ninguno)'}\n\n"
                    f"[Entidad identificada por el plan]\n{plan.producto or ', '.join(plan.filtros.codigos) or plan.filtros.texto or '(ninguna)'}\n\n"
                    f"[Candidatos del catálogo vigente]\n{json.dumps(productos_candidatos_clasificador, ensure_ascii=False) if productos_candidatos_clasificador else '(sin coincidencia literal recuperada)'}"
                ),
            },
        ]
        esquema_relacion = copy.deepcopy(_ESQUEMA_RELACION_STOCK_JSON)
        codigos_candidatos_vigentes = list(dict.fromkeys(
            codigo for codigo, _ in productos_candidatos_clasificador if codigo
        ))
        esquema_relacion["properties"]["c"] = {
            "type": "string",
            "enum": ["ninguno", *codigos_candidatos_vigentes],
        }
        esquema_relacion["required"] = [*esquema_relacion["required"], "c"]
        nombre_esquema_relacion = "haku_stock_relation_entity"
        max_tokens_relacion = 80
    chars_clasificador = sum(len(str(m.get("content") or "")) for m in mensajes)
    inicio = time.perf_counter()
    telemetria["llamadas_fisicas"] = int(telemetria.get("llamadas_fisicas", 0) or 0) + 1
    try:
        respuesta = cliente_llm.generar(
            mensajes=mensajes,
            formato_json=True,
            temperatura=0.0,
            max_tokens=max_tokens_relacion,
            timeout=timeout,
            json_schema=esquema_relacion,
            json_schema_name=nombre_esquema_relacion,
        )
    except Exception as exc:
        duracion_ms = (time.perf_counter() - inicio) * 1000.0
        plan.llm_ms += duracion_ms
        telemetria["duracion_ms"] = float(telemetria.get("duracion_ms", 0.0) or 0.0) + duracion_ms
        telemetria["clasificacion_condicion"] = "error"
        logger.warning(
            "[HakuAgente] No se pudo verificar la relación de stock: %s",
            type(exc).__name__,
        )
        return plan

    duracion_ms = (time.perf_counter() - inicio) * 1000.0
    plan.llm_ms += float(getattr(respuesta, "duracion_ms", 0.0) or duracion_ms)
    plan.llm_cola_ms += float(getattr(respuesta, "espera_cola_ms", 0.0) or 0.0)
    plan.chars_prompt += chars_clasificador
    plan.tokens_estimados_prompt += max(1, chars_clasificador // 4)
    tokens = getattr(respuesta, "tokens_completados", 0)
    if isinstance(tokens, (int, float)):
        plan.tokens_completados += int(tokens)
    telemetria["duracion_ms"] = plan.llm_ms

    datos = getattr(respuesta, "datos_json", None)
    relacion_raw = (
        datos.get("r", datos.get("relacion"))
        if isinstance(datos, dict) else None
    )
    relacion = (
        _RELACION_STOCK_CORTA_A_CANONICA.get(
            str(relacion_raw).strip().casefold(), relacion_raw,
        )
        if relacion_raw is not None else None
    )
    codigo_entidad_raw = (
        datos.get("c") if isinstance(datos, dict) else None
    )
    codigo_entidad = str(codigo_entidad_raw or "").strip().casefold()
    candidato_seleccionado = next(
        (
            (codigo, nombre)
            for codigo, nombre in productos_candidatos
            if codigo.casefold() == codigo_entidad
        ),
        None,
    )
    if consulta_colectiva_con_filtro_no_anclado and candidato_seleccionado:
        codigo, nombre = candidato_seleccionado
        plan.filtros.codigos = [codigo]
        plan.filtros.texto = None
        plan.producto = nombre
        plan.alcance_entidad = "producto"
        plan.filtro_texto_es_entidad_verificado = True
        plan.codigo_catalogo_verificado = True
        telemetria["resolucion_entidad_catalogo"] = {
            "codigo": codigo,
            "fuente": "clasificador_semantico_candidato_vigente",
        }
    margen_raw = (
        datos.get("m", datos.get("margen_cualitativo", ""))
        if isinstance(datos, dict) else ""
    )
    disponibilidad_raw = (
        datos.get("x", "") if isinstance(datos, dict) else ""
    )
    margen_cualitativo = (
        _MARGEN_STOCK_CORTO_A_CANONICO.get(
            str(margen_raw).strip().casefold(),
            str(margen_raw).strip().casefold(),
        )
    )
    disponibilidad_explicita = str(disponibilidad_raw).strip().casefold()
    if _condicion_bajo_minimo_gobernada_por_infinitivo(texto):
        relacion = "aproximacion"
    if (
        relacion in {None, "sin_condicion", "ambigua", "con_stock"}
        and disponibilidad_explicita == "cero"
    ):
        # La disponibilidad declarada ayuda a completar una clasificación
        # ausente, pero no sustituye el predicado solicitado (p. ej. igualdad
        # con un umbral configurado en cero o una petición amplia de reposición).
        relacion = "agotamiento"
    filtro_texto_es_entidad = (
        datos.get("e", datos.get("filtro_texto_es_entidad"))
        if isinstance(datos, dict)
        and isinstance(datos.get("e", datos.get("filtro_texto_es_entidad")), bool)
        else None
    )
    if (
        filtro_texto_es_entidad is True
        and tiene_filtro_colectivo_no_anclado
        and not _filtro_texto_anclado_a_candidato(plan.filtros.texto, productos_candidatos)
    ):
        # A semantic label cannot turn a fragment of a condition into a
        # product: the live catalogue must independently anchor that filter.
        filtro_texto_es_entidad = False
    atributos_stock_raw = (
        datos.get("a", datos.get("atributos_stock_solicitados"))
        if isinstance(datos, dict) else None
    )
    atributos_stock_verificacion_valida = isinstance(atributos_stock_raw, list)
    atributos_stock_solicitados = sorted({
        str(campo).strip().casefold()
        for campo in (atributos_stock_raw or [])
        if str(campo).strip().casefold() in {"stock", "stock_minimo"}
        and (str(campo).strip().casefold() != "stock_minimo" or _texto_menciona_stock_minimo(texto))
    })
    # Este resultado diferencia una decisión semántica explícita del
    # clasificador de un campo candidato que solo venía en el plan inicial.
    # Si el clasificador falla, el valor None permite conservar el candidato
    # y evita borrar una paráfrasis válida por no coincidir con una etiqueta.
    plan.atributos_stock_verificados = (
        list(atributos_stock_solicitados)
        if atributos_stock_verificacion_valida
        else None
    )
    texto_cantidad_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    pide_cantidad_de_stock = bool(re.search(
        r"\b(?:cu[aá]nt[oa]s?|unidad(?:es)?|cantidad)\b",
        texto_cantidad_norm,
    ))
    if (
        pide_cantidad_de_stock
        and "stock" in campos_plan_iniciales
        and "stock_minimo" in atributos_stock_solicitados
        and "stock" not in atributos_stock_solicitados
    ):
        # Una consulta compuesta por unidades actuales y umbral necesita ambas
        # columnas. El clasificador puede confirmar el umbral aunque omita la
        # dimensión física ya expresada por el plan cuantitativo.
        atributos_stock_solicitados.append("stock")
        atributos_stock_solicitados.sort()
    atributos_stock_clasificados = list(atributos_stock_solicitados)
    plan.relacion_stock_verificada = relacion if isinstance(relacion, str) else None
    # Solo se valida la propuesta concreta de `filtros.texto`. Si el plan
    # identifica por separado una entidad singular (p. ej. nombre/código en
    # `producto`), el resultado sobre un filtro vacío no debe borrarla ni
    # convertir la consulta en colectiva.
    plan.filtro_texto_es_entidad_verificado = (
        filtro_texto_es_entidad if plan.filtros.texto else None
    )
    condicion_validada: Optional[str] = None
    if isinstance(relacion, str):
        condicion_validada = _RELACION_A_CONDICION_STOCK.get(relacion.strip().casefold())
    elif isinstance(relacion, list):
        relaciones = {str(valor).strip().casefold() for valor in relacion}
        if relaciones == {"inferior_actual", "agotamiento"}:
            condicion_validada = "requiere_reposicion"
    entidad_individual_explicita = bool(
        plan.filtros.codigos
        or _producto_es_entidad_concreta(plan.producto)
        or plan.referencia == "foco"
        or plan.alcance_entidad == "producto"
        or (
            plan.filtros.texto
            and not _es_filtro_marco_global(plan.filtros.texto)
            and _filtro_texto_anclado_a_candidato(plan.filtros.texto, productos_candidatos)
        )
    )
    if (
        condicion_validada == "agotado"
        and _condiciones_stock_mencionadas(texto) == {"bajo_minimo"}
    ):
        condicion_validada = "bajo_minimo"
        relacion = "inferior_actual"
        plan.relacion_stock_verificada = relacion

    if (
        condicion_validada in CONDICIONES_VALIDAS
        and not entidad_individual_explicita
        and plan.operacion in {"detalle", "comparar"}
    ):
        # A stock relation selects a set of records. When the model used a
        # single-record/comparison operation without identifying any record,
        # normalize it to the collection selected by that predicate.
        plan.operacion = "listar"
        plan.referencia = "ninguna"
        plan.alcance_entidad = "coleccion"
        plan.ubicaciones_comparadas = []
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
        operacion_colectiva = True
        comparacion_de_coleccion = False
        detalle_colectivo_con_umbral = False

    # El modelo puede reconocer el campo configurado y el valor cero, pero
    # confundir su función de filtro con una columna solicitada. En una
    # selección colectiva, ambos datos describen el criterio de pertenencia.
    if (
        relacion == "sin_condicion"
        and (
            "stock_minimo" in atributos_stock_solicitados
            or "stock_minimo" in campos_plan_iniciales
        )
        and not entidad_individual_explicita
        and plan.operacion in {"listar", "contar"}
        and re.search(r"(?<!\d)0+(?!\d)|\bcero\b", normalizar(preprocesar_texto_inventario(texto or "")))
    ):
        relacion = "minimo_cero"
        plan.relacion_stock_verificada = relacion
        condicion_validada = "minimo_cero"
        atributos_stock_solicitados = [
            campo for campo in atributos_stock_solicitados if campo != "stock_minimo"
        ]

    # En un listado condicionado, una métrica necesaria para decidir la
    # pertenencia no se vuelve automáticamente una columna de salida. Conserva
    # un campo stock solo si el plan primario también lo identificó como dato
    # solicitado; una propuesta preliminar con muchos campos no cuenta como
    # confirmación.
    consulta_lista_filtrada = bool(
        (operacion_colectiva or comparacion_de_coleccion or detalle_colectivo_con_umbral)
        and not entidad_individual_explicita
        and plan.alcance_entidad in {"coleccion", "global", None}
        and (
            condicion_validada in CONDICIONES_VALIDAS
            or condicion_inicial in _CONDICIONES_STOCK_PARA_CLASIFICAR
            or plan.filtros.condicion in _CONDICIONES_STOCK_PARA_CLASIFICAR
        )
    )
    if consulta_lista_filtrada and atributos_stock_solicitados:
        propuesta_precisa = (
            len(campos_plan_iniciales) <= 3
            and medida_plan_inicial == "atributo"
        )
        if not propuesta_precisa:
            atributos_stock_solicitados = []

    # El valor del umbral como atributo desplaza una relación contradictoria
    # únicamente en una consulta de entidad individual. En una selección de
    # colección, una relación clasificada por el LLM sigue siendo un filtro.
    menciona_cero = bool(re.search(
        r"(?<!\d)0+(?!\d)|\bcero\b",
        normalizar(preprocesar_texto_inventario(texto or "")),
    ))
    if condicion_validada in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA and not _condicion_stock_tiene_indicio_textual(texto, condicion_validada):
        if condicion_validada == "suficiente" and _condicion_stock_tiene_indicio_textual(texto, "con_stock"):
            relacion = "con_stock"
            plan.relacion_stock_verificada = relacion
            condicion_validada = "con_stock"
        else:
            relacion = "sin_condicion"
            plan.relacion_stock_verificada = relacion
            condicion_validada = None
            if plan.filtros.condicion in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA:
                plan.filtros.condicion = None
    if condicion_validada == "minimo_cero" and not menciona_cero:
        relacion = "sin_condicion"
        plan.relacion_stock_verificada = relacion
        condicion_validada = None
        if plan.filtros.condicion == "minimo_cero":
            plan.filtros.condicion = None
    if (
        "stock_minimo" in atributos_stock_solicitados
        and entidad_individual_explicita
        and (
            not _condiciones_stock_mencionadas(texto)
            or _condiciones_stock_mencionadas(texto) == {"minimo_cero"}
        )
    ):
        relacion = "sin_condicion"
        plan.relacion_stock_verificada = relacion
        condicion_validada = None
        plan.filtros.condicion = None
    if atributos_stock_solicitados:
        campos_ajenos_stock = [
            campo for campo in (plan.campos or [])
            if campo not in {"stock", "stock_minimo"}
        ]
        plan.campos = list(dict.fromkeys(campos_ajenos_stock + atributos_stock_solicitados))
        plan.medida_solicitada = "atributo"
        if "stock_minimo" in atributos_stock_solicitados:
            if entidad_individual_explicita:
                plan.operacion = "detalle"
                plan.alcance_entidad = "producto"
                plan.alcance = "actual"
                plan.filtros.condicion = None
    elif condicion_validada in CONDICIONES_VALIDAS:
        # El primer plan puede confundir el umbral usado para seleccionar
        # registros con un atributo pedido. Si el clasificador semántico no
        # confirma esos datos, descarta solo las columnas de stock inferidas.
        plan.campos = [
            campo for campo in (plan.campos or [])
            if campo not in {"stock", "stock_minimo"}
        ]
        if plan.medida_solicitada == "atributo":
            plan.medida_solicitada = "ninguna"

    telemetria["clasificacion_condicion"] = {
        "relacion": relacion,
        "margen_cualitativo": margen_cualitativo or None,
        "disponibilidad_explicita": disponibilidad_explicita or None,
        "condicion_inicial": condicion_inicial,
        "condicion_validada": condicion_validada,
        "fuente": (
            "condicion_semantica_verificada"
            if clasificador_colectivo_compacto
            else "clasificador_relacion_entidad"
        ),
        "filtro_texto_es_entidad": filtro_texto_es_entidad,
        "atributos_stock_clasificados": atributos_stock_clasificados,
        "atributos_stock_solicitados": atributos_stock_solicitados,
        "respuesta_json_valida": isinstance(datos, dict),
        "tokens_completados": int(tokens) if isinstance(tokens, (int, float)) else None,
        "finish_reason": getattr(respuesta, "finish_reason", None),
    }
    if isinstance(relacion, str) and relacion in {"sin_condicion", "ambigua"}:
        condiciones_explicitas = _condiciones_stock_mencionadas(texto)
        if plan.filtros.condicion in condiciones_explicitas:
            # El parser semántico puede omitir la relación aunque el plan ya
            # conserve un filtro inequívoco derivado del mensaje. No dejes que
            # un `sin_condicion` genérico lo borre (p. ej. stock mínimo igual
            # a cero). Tampoco conviertas el fragmento del filtro en producto.
            condicion_explicita = plan.filtros.condicion
            plan.relacion_stock_verificada = next(
                (
                    nombre
                    for nombre, condicion
                    in _RELACION_A_CONDICION_STOCK.items()
                    if condicion == condicion_explicita
                ),
                relacion,
            )
            if (
                filtro_texto_es_entidad is False
                and plan.referencia not in {"foco", "resultados_previos"}
                and not plan.filtros.codigos
            ):
                plan.filtros.texto = None
                if plan.operacion in {"listar", "contar", "sumar_unidades"}:
                    plan.alcance_entidad = "coleccion"
            return plan
        plan.filtros.condicion = None
        plan.filtros.umbral = None
        return plan
    if condicion_validada in CONDICIONES_VALIDAS:
        plan.filtros.condicion = condicion_validada
        if (operacion_colectiva or comparacion_de_coleccion) and plan.tipo == "consulta":
            if plan.referencia not in {"foco", "resultados_previos"} and not plan.filtros.codigos:
                plan.alcance_entidad = "coleccion"
            if (
                filtro_texto_es_entidad is False
                and plan.referencia not in {"foco", "resultados_previos"}
                and not plan.filtros.codigos
            ):
                plan.filtros.texto = None
    return plan


def _reverificar_atributos_stock_compuestos_si_falta(
    texto: str,
    cliente_llm: Any,
    plan: PlanAgente,
    timeout: float,
    telemetria: Dict[str, Any],
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> PlanAgente:
    """Reintenta atributos múltiples después de resolver la entidad del catálogo.

    El plan inicial puede posponer la resolución de un nombre a código; en ese
    punto una coordinación entre atributos todavía no parece referirse a un
    producto individual. Tras anclar la entidad con datos vivos, se valida de
    nuevo solo si el plan conserva únicamente stock y aún no confirma el mínimo.
    """
    campos_stock = set(plan.campos or []) & {"stock", "stock_minimo"}
    if (
        plan.tipo != "consulta"
        or plan.operacion not in {"detalle", "sumar_unidades"}
        or campos_stock != {"stock"}
        or plan.alcance in {"evento", "ambos"}
        or "stock_minimo" in (plan.atributos_stock_verificados or [])
    ):
        return plan

    codigos_entidad = list(plan.filtros.codigos or [])
    candidatos_vigentes = _obtener_productos_candidatos_para_verificacion(
        codigos_productos_candidatos,
    )
    if not codigos_entidad and plan.filtros.texto:
        coincidencias_nombre = [
            codigo for codigo, nombre in candidatos_vigentes
            if _filtro_texto_anclado_a_candidato(plan.filtros.texto, [(codigo, nombre)])
        ]
        if len(set(coincidencias_nombre)) == 1:
            codigos_entidad = list(dict.fromkeys(coincidencias_nombre))
    entidad_individual_resuelta = bool(
        codigos_entidad
        or plan.alcance_entidad == "producto"
        or _producto_es_entidad_concreta(plan.producto)
        or plan.referencia == "foco"
    )
    if not entidad_individual_resuelta and (
        plan.alcance_entidad in {"coleccion", "global"}
        or not plan.filtros.texto
    ):
        return plan
    codigos_para_deteccion = codigos_entidad or list(codigos_productos_candidatos or [])
    consulta_compuesta = _consulta_coordina_entidad(
        texto,
        codigos_productos_candidatos=codigos_para_deteccion,
        tiene_foco=True,
    )
    if not consulta_compuesta:
        return plan
    return _verificar_condicion_stock_semantica(
        texto,
        cliente_llm,
        plan,
        timeout,
        telemetria,
        codigos_productos_candidatos,
    )


def _quitar_con_stock_de_producto_individual(plan: PlanAgente) -> PlanAgente:
    """Evita que un filtro de colección oculte el saldo pedido de una referencia.

    La entidad debe estar resuelta por el contrato a un producto individual o
    describir un producto concreto; los identificadores y atributos provienen
    del catálogo activo.
    """
    entidad_producto_individual = bool(
        plan.tipo == "consulta"
        and (
            (
                plan.alcance_entidad == "producto"
                and (
                    plan.filtros.codigos
                    or plan.filtros.texto
                    or _producto_es_entidad_concreta(plan.producto)
                    or plan.referencia in {"foco", "resultados_previos"}
                )
            )
            or (
                bool(plan.filtros.codigos)
                and plan.operacion in {"detalle", "sumar_unidades"}
            )
            or (
                bool(plan.filtros.texto)
                and not plan.filtros.categoria
                and not plan.filtros.categoria_explicita
                and plan.operacion in {"detalle", "sumar_unidades", "listar"}
            )
        )
    )
    if plan.filtros.condicion == "con_stock" and entidad_producto_individual:
        # «Disponible» o «¿quedan X?» identifica el saldo a consultar; no debe
        # ocultar un producto agotado devolviendo un falso "no encontré".
        plan.filtros.condicion = None
        plan.filtros.umbral = None
        plan.relacion_stock_verificada = "sin_condicion"
    return plan


def _revalidar_condicion_stock_final(
    texto: str,
    cliente_llm: Any,
    plan: PlanAgente,
    timeout: float,
    telemetria: Dict[str, Any],
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> PlanAgente:
    """Valida los filtros después de normalizar la operación y la entidad.

    El plan primario puede clasificar una pregunta cuantitativa como `detalle`
    antes de que la capa de intención la convierta en `sumar_unidades`. Las
    verificaciones previas no siempre alcanzan entonces la rama de filtros.
    Esta última pasada exige evidencia textual o una relación ya verificada;
    `plan.crudo` por sí solo no acredita que el usuario pidiera el filtro.
    """
    condicion = plan.filtros.condicion
    plan = _quitar_con_stock_de_producto_individual(plan)
    if condicion != plan.filtros.condicion:
        return plan

    if condicion not in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA:
        return plan

    relacion_verificada = getattr(plan, "relacion_stock_verificada", None)
    condicion_verificada = _RELACION_A_CONDICION_STOCK.get(relacion_verificada)
    condiciones_explicitas = _condiciones_stock_mencionadas(texto)
    if condicion_verificada == condicion and condicion not in condiciones_explicitas:
        return plan

    plan = _verificar_condicion_stock_semantica(
        texto,
        cliente_llm,
        plan,
        timeout,
        telemetria,
        codigos_productos_candidatos,
    )
    condicion_antes_union = plan.filtros.condicion
    plan = _normalizar_union_condiciones_stock(texto, plan)
    if plan.filtros.condicion == "requiere_reposicion" and condicion_antes_union != "requiere_reposicion":
        plan.relacion_stock_verificada = "union_reposicion"
    condicion_final = plan.filtros.condicion
    relacion_final = getattr(plan, "relacion_stock_verificada", None)
    condicion_final_verificada = _RELACION_A_CONDICION_STOCK.get(relacion_final)
    if (
        condicion_final in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA
        and condicion_final not in condiciones_explicitas
        and condicion_final_verificada != condicion_final
    ):
        # Si el clasificador no pudo verificarla (incluidos errores/timeout),
        # no restrinjas la consulta a partir de una hipótesis del plan inicial.
        plan.filtros.condicion = None
        plan.filtros.umbral = None
    return plan


def _verificar_intencion_listado_global(
    texto: str,
    cliente_llm: Any,
    plan: PlanAgente,
    timeout: float,
    telemetria: Dict[str, Any],
    codigos_productos_candidatos: Optional[List[str]] = None,
    contexto_catalogo: Optional[str] = None,
    categorias_accesibles: Optional[List[str]] = None,
    almacenes_accesibles: Optional[List[str]] = None,
    ubicaciones_accesibles: Optional[List[str]] = None,
) -> PlanAgente:
    """Evita que una pregunta sobre las funciones termine mostrando todo el catálogo."""
    # Esta verificación se ejecuta antes de completar entidades del catálogo y
    #, si hace falta, de nuevo al final tras normalizar el plan. Evita una
    # segunda llamada al LLM cuando la primera ya clasificó una intención.
    if "clasificacion_listado_global" in telemetria:
        return plan

    clasificacion_condicion = telemetria.get("clasificacion_condicion") or {}
    condicion_verificada = clasificacion_condicion.get("condicion_validada")
    fuente_condicion_verificada = clasificacion_condicion.get("fuente")
    relacion_verificada = getattr(plan, "relacion_stock_verificada", None)
    relacion_explicita_actual = _relacion_stock_explicita_sin_ambiguedad(texto, plan)
    if (
        plan.tipo == "consulta"
        and plan.operacion in {"listar", "contar", "sumar_unidades"}
        and plan.alcance_entidad in {"coleccion", "global", None}
        and plan.filtros.es_prestable is None
        and plan.filtros.condicion is not None
        and (
            (
                condicion_verificada == plan.filtros.condicion
                and _RELACION_A_CONDICION_STOCK.get(relacion_verificada) == plan.filtros.condicion
                and fuente_condicion_verificada in {
                    "plan_primario_colectivo_sin_entidad_candidata",
                    "condicion_explicita_sin_ambiguedad",
                    "condicion_semantica_verificada",
                }
                and relacion_explicita_actual == relacion_verificada
            )
            or (
                not clasificacion_condicion
                and relacion_verificada is None
                and _RELACION_A_CONDICION_STOCK.get(relacion_explicita_actual) == plan.filtros.condicion
            )
        )
    ):
        # Solo se omite la clasificación global cuando el mensaje afirma de
        # forma explícita el mismo estado. Una condición inferida no descarta
        # que se haya pedido filtrar por otra propiedad (por ejemplo,
        # elegibilidad para préstamo).
        telemetria["clasificacion_listado_global"] = {
            "intencion": "consulta_inventario",
            "condicion_stock": plan.filtros.condicion,
            "omitida_por_condicion_stock_verificada": True,
        }
        return plan

    seguimiento_anaforico_con_atributo = bool(
        telemetria.get("seguimiento_foco_anaforico")
        and plan.tipo in {"consulta", "aclaracion"}
        and plan.campos
        and plan.operacion in {"detalle", "listar", "contar", "sumar_unidades"}
        and not any((
            plan.filtros.texto,
            plan.filtros.codigos,
            plan.filtros.categoria,
            plan.filtros.categoria_explicita,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
        ))
        and plan.referencia == "ninguna"
    )
    if seguimiento_anaforico_con_atributo:
        # El agente confirmó antes de inferir que hay un único producto en foco
        # y el mensaje contiene un clítico singular. El atributo estructurado
        # ya permite desambiguar el seguimiento, por lo que no hace falta otra
        # llamada de intención global.
        telemetria["clasificacion_listado_global"] = {
            "intencion": "consulta_entidad",
            "condicion_stock": "sin_condicion",
            "omitida_por_seguimiento_anaforico": True,
        }
        return plan

    # Si la inferencia principal ya propuso una operación de datos para una
    # entidad viva y accesible del catálogo, evita una segunda inferencia que
    # vuelva a tratar la frase sin contexto y la reclasifique como tema externo.
    # Que un descriptor aparezca literalmente en el mensaje no demuestra que
    # sea una entidad: puede ser la forma en que la persona expresa una
    # propiedad del catálogo (por ejemplo, elegibilidad o estado de préstamo).
    # La evidencia se deriva del catálogo vigente de esta cuenta, no de alias
    # ni nombres mantenidos en el código.
    productos_candidatos = _obtener_productos_candidatos_para_verificacion(
        codigos_productos_candidatos or [],
    )
    codigo_producto_respaldado = bool(
        plan.filtros.codigos
        and set(plan.filtros.codigos).issubset({codigo for codigo, _ in productos_candidatos})
    )
    texto_producto_respaldado = bool(
        plan.filtros.texto
        and _filtro_texto_anclado_a_candidato(plan.filtros.texto, productos_candidatos)
    )
    categoria_del_plan_respaldada = bool(
        plan.filtros.categoria
        and _menciones_catalogo(plan.filtros.categoria, categorias_accesibles or [])
    )
    consulta_de_entidad_viva = bool(
        cliente_llm is not None
        and plan.tipo == "consulta"
        and plan.operacion in {"listar", "contar", "sumar_unidades", "detalle"}
        and plan.referencia == "ninguna"
        and not plan.ubicaciones_comparadas
        and (
            codigo_producto_respaldado
            or (texto_producto_respaldado and plan.alcance_entidad == "producto")
            or categoria_del_plan_respaldada
        )
    )
    if consulta_de_entidad_viva:
        telemetria["clasificacion_listado_global"] = {
            "intencion": "consulta_entidad",
            "condicion_stock": "con_stock" if plan.filtros.condicion == "con_stock" else "sin_condicion",
            "omitida_por_entidad_catalogo_vigente": True,
        }
        return plan

    solo_con_stock = bool(
        plan.filtros.condicion == "con_stock"
        and not any((
            plan.filtros.texto,
            plan.filtros.codigos,
            plan.filtros.categoria,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
            plan.filtros.categoria_generica,
        ))
    )
    filtro_es_marco_global = bool(
        plan.filtros.texto and _es_filtro_marco_global(plan.filtros.texto)
    )
    palabras_mensaje = set(normalizar(preprocesar_texto_inventario(texto or "")).split())
    detalle_con_marco_global = bool(
        plan.operacion == "detalle" and filtro_es_marco_global
    )
    sin_entidad_de_catalogo = bool(
        not plan.filtros.codigos
        and not plan.filtros.categoria
        and not plan.filtros.categoria_explicita
        and not plan.filtros.almacen
        and not plan.filtros.ubicacion
        and not plan.filtros.categoria_generica
        and (not plan.filtros.texto or filtro_es_marco_global)
    )
    detalle_sin_entidad_ni_dominio = bool(
        plan.operacion == "detalle"
        and sin_entidad_de_catalogo
        and not (palabras_mensaje & _PALABRAS_DOMINIO_INVENTARIO)
        and not plan.campos
        and not plan.filtros.condicion
    )
    filtro_de_producto_sin_anclar = bool(
        plan.tipo in {"consulta", "aclaracion"}
        and plan.filtros.texto
        and not filtro_es_marco_global
        and not plan.filtros.codigos
        and (
            codigos_productos_candidatos is not None
            or plan.alcance_entidad in {"coleccion", "global"}
        )
        and not any((
            plan.filtros.categoria,
            plan.filtros.categoria_explicita,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
            plan.filtros.categoria_generica,
        ))
        and (
            codigos_productos_candidatos is None
            or not _filtro_texto_anclado_a_candidato(
                plan.filtros.texto, productos_candidatos,
            )
        )
    )
    if (
        cliente_llm is None
        or plan.tipo not in {"consulta", "aclaracion"}
        or (
            plan.tipo == "consulta"
            and plan.alcance == "evento"
            and "movimientos" in (plan.campos or [])
            and (plan.fecha_desde or plan.fecha_hasta)
        )
        or (
            plan.tipo == "consulta"
            and plan.operacion == "listar"
            and plan.filtros.condicion == "con_prestamos"
            and not re.search(
                r"\b(?:cuant\w*|cantidad(?:es)?|unidades?|total(?:es)?|saldos?|"
                r"prestar|presta(?:me|le|lo|la|los|las|les|s)?|preste\w*|"
                r"registrar|registra\w*|hacer|haz|quiero|necesito|deseo)\b",
                normalizar(preprocesar_texto_inventario(texto or "")),
            )
            and bool(re.search(
                r"\b(?:muestr\w*|mostr\w*|enseñ\w*|ensen\w*|ver|dime|dame|list\w*|que|cual(?:es)?|quien(?:es)?)\b",
                normalizar(preprocesar_texto_inventario(texto or "")),
            ))
        )
        or plan.operacion not in {None, "listar", "explicar_estado", "contar", "sumar_unidades", "detalle"}
        or (
            plan.operacion == "detalle"
            and not (
                detalle_con_marco_global
                or filtro_de_producto_sin_anclar
                or detalle_sin_entidad_ni_dominio
            )
        )
        or not (
            sin_entidad_de_catalogo
            or filtro_de_producto_sin_anclar
            or (
                texto_producto_respaldado
                and plan.alcance_entidad in {"coleccion", "global"}
            )
        )
        or plan.referencia != "ninguna"
        or plan.ubicaciones_comparadas
    ):
        return plan

    prefijos_catalogo_utiles = (
        "Categorías del catálogo mencionadas:",
        "Categorías candidatas actuales:",
        "Coincidencia única de producto actual accesible:",
        "Candidatos actuales de producto accesibles",
        "Productos candidatos actuales accesibles que coinciden con el mensaje:",
    )
    lineas_catalogo = [
        linea.strip()
        for linea in (contexto_catalogo or "").splitlines()
        if linea.strip().startswith(prefijos_catalogo_utiles)
    ]
    evidencia_catalogo = "\n".join(lineas_catalogo)[:1600]
    contenido_usuario_clasificador = (
        f"[Mensaje actual]\n{preprocesar_texto_inventario(texto or '', normalizar_medidas=False)[:400]}"
        "\n\n[Plan preliminar; puede contener errores, pero su intención estructurada "
        "y sus entidades deben contrastarse con la tarea y el catálogo]\n"
        f"tipo={plan.tipo}; operacion={plan.operacion or 'ninguna'}; "
        f"producto={plan.filtros.texto or 'ninguno'}; "
        f"codigos={', '.join(plan.filtros.codigos or []) or 'ninguno'}; "
        f"categoria={plan.filtros.categoria or 'ninguna'}; "
        f"alcance={plan.alcance_entidad or 'indefinido'}; "
        f"medida={plan.medida_solicitada or 'ninguna'}; "
        f"es_prestable={plan.filtros.es_prestable}; "
        f"condicion={plan.filtros.condicion or 'ninguna'}; "
        f"campos={', '.join(plan.campos or []) or 'ninguno'}"
    )
    if evidencia_catalogo:
        contenido_usuario_clasificador += (
            "\n\n[Referencias actuales del catálogo accesible; sirven para reconocer "
            "entidades, no para cambiar la tarea solicitada]\n"
            + evidencia_catalogo
        )
    mensajes = [
        {"role": "system", "content": PROMPT_CLASIFICADOR_LISTADO_GLOBAL},
        {"role": "user", "content": contenido_usuario_clasificador},
    ]
    chars_clasificador = sum(len(str(m.get("content") or "")) for m in mensajes)
    inicio = time.perf_counter()
    condicion_stock = "con_stock" if solo_con_stock else "sin_condicion"
    relacion_stock = "sin_condicion"
    telemetria["llamadas_fisicas"] = int(telemetria.get("llamadas_fisicas", 0) or 0) + 1
    try:
        respuesta = cliente_llm.generar(
            mensajes=mensajes,
            formato_json=True,
            temperatura=0.0,
            max_tokens=128,
            timeout=timeout,
            json_schema=_ESQUEMA_INTENCION_LISTADO_GLOBAL_JSON,
            json_schema_name="haku_global_intent",
        )
    except Exception as exc:
        duracion_ms = (time.perf_counter() - inicio) * 1000.0
        plan.llm_ms += duracion_ms
        telemetria["duracion_ms"] = float(telemetria.get("duracion_ms", 0.0) or 0.0) + duracion_ms
        telemetria["clasificacion_listado_global"] = {"intencion": "ambigua", "error": type(exc).__name__}
        logger.warning("[HakuAgente] No se pudo verificar un listado global: %s", type(exc).__name__)
        intencion = "ambigua"
    else:
        duracion_ms = (time.perf_counter() - inicio) * 1000.0
        plan.llm_ms += float(getattr(respuesta, "duracion_ms", 0.0) or duracion_ms)
        plan.llm_cola_ms += float(getattr(respuesta, "espera_cola_ms", 0.0) or 0.0)
        plan.chars_prompt += chars_clasificador
        plan.tokens_estimados_prompt += max(1, chars_clasificador // 4)
        tokens = getattr(respuesta, "tokens_completados", 0)
        if isinstance(tokens, (int, float)):
            plan.tokens_completados += int(tokens)
        datos = getattr(respuesta, "datos_json", None)
        intencion = datos.get("intencion") if isinstance(datos, dict) else None
        accion_solicitada = datos.get("accion_solicitada") if isinstance(datos, dict) else None
        if accion_solicitada not in ACCIONES_VALIDAS:
            accion_solicitada = None
        clasificacion_prestamo = (
            datos.get("clasificacion_prestamo")
            if isinstance(datos, dict)
            else None
        )
        if clasificacion_prestamo not in {
            "operacion", "elegibilidad", "saldo_pendiente",
            "prestamos_activos", "no_aplica",
        }:
            clasificacion_prestamo = "no_aplica"
        if clasificacion_prestamo == "operacion":
            accion_solicitada = "prestamo"
            intencion = "operacion_inventario"
        elif accion_solicitada == "prestamo":
            # A specific loan-task classification outranks a contradictory
            # action slot, preventing an eligibility query from becoming a
            # proposed loan operation.
            accion_solicitada = None
        elif accion_solicitada:
            # The action slot is the explicit semantic decision for this
            # classifier. If its broad scope label contradicts that decision,
            # preserve the specific operation and avoid turning it into a query.
            intencion = "operacion_inventario"
        relacion_raw = datos.get("relacion_stock") if isinstance(datos, dict) else None
        propiedad_prestable_solicitada = (
            datos.get("propiedad_prestable_solicitada") is True
            if isinstance(datos, dict)
            else False
        )
        es_prestable_raw = datos.get("es_prestable") if isinstance(datos, dict) else None
        if not propiedad_prestable_solicitada or not isinstance(es_prestable_raw, bool):
            es_prestable_raw = None
        prestamos_activos_raw = datos.get("prestamos_activos") if isinstance(datos, dict) else None
        prestamos_activos = prestamos_activos_raw is True
        pide_saldo_pendiente = (
            datos.get("pide_saldo_pendiente") is True
            if isinstance(datos, dict)
            else False
        )
        filtro_texto_es_entidad = (
            datos.get("filtro_texto_es_entidad")
            if isinstance(datos, dict)
            and isinstance(datos.get("filtro_texto_es_entidad"), bool)
            else None
        )
        relacion_stock = str(relacion_raw or "sin_condicion").strip().casefold()
        if intencion not in {
            "capacidades", "social", "fuera_dominio", "consulta_inventario",
            "listar_catalogo", "listar_disponibles", "consulta_entidad",
            "operacion_inventario",
        }:
            intencion = "ambigua"
        if relacion_stock not in {*_RELACION_A_CONDICION_STOCK, "sin_condicion", "ambigua"}:
            relacion_stock = "sin_condicion"
        if intencion == "listar_disponibles" and relacion_stock == "sin_condicion":
            relacion_stock = "con_stock"
        condicion_validada = _RELACION_A_CONDICION_STOCK.get(relacion_stock)
        if condicion_validada and not _condicion_stock_tiene_indicio_textual(texto, condicion_validada):
            if condicion_validada == "suficiente" and _condicion_stock_tiene_indicio_textual(texto, "con_stock"):
                relacion_stock = "con_stock"
                condicion_validada = "con_stock"
            else:
                relacion_stock = "sin_condicion"
                condicion_validada = None
        if propiedad_prestable_solicitada and not _texto_menciona_elegibilidad_prestamo(texto):
            propiedad_prestable_solicitada = False
            es_prestable_raw = None
        if (prestamos_activos or pide_saldo_pendiente) and not _texto_menciona_prestamo_o_custodia(texto):
            prestamos_activos = False
            pide_saldo_pendiente = False
        sin_entidad_concreta = not any((
            plan.filtros.codigos,
            plan.filtros.texto,
            plan.filtros.categoria,
            plan.filtros.categoria_explicita,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
            plan.producto,
        ))
        alcance_de_coleccion = bool(
            plan.alcance_entidad in {None, "coleccion", "global", "indefinido"}
            or sin_entidad_concreta
        )
        clasificacion_inventario = intencion in {
            "consulta_inventario", "listar_catalogo", "listar_disponibles",
            "consulta_entidad", "ambigua",
        }
        if (
            clasificacion_inventario
            and alcance_de_coleccion
            and not propiedad_prestable_solicitada
        ):
            # La propiedad del producto no se deduce de una consulta de stock.
            # Si el plan primario la añadió por error, quitar solo ese filtro y
            # conservar los predicados independientes (p. ej. reposición).
            plan.filtros.es_prestable = None
            plan.campos = [campo for campo in plan.campos if campo != "es_prestable"]
        propiedad_catalogo_aplicable = bool(
            es_prestable_raw is not None
            and clasificacion_inventario
            and alcance_de_coleccion
        )
        consulta_prestamos_activos_aplicable = bool(
            prestamos_activos
            and clasificacion_inventario
            and alcance_de_coleccion
        )
        consulta_saldo_pendiente_aplicable = bool(
            pide_saldo_pendiente
            and clasificacion_inventario
            and alcance_de_coleccion
        )
        if (
            propiedad_catalogo_aplicable
            or consulta_prestamos_activos_aplicable
            or consulta_saldo_pendiente_aplicable
        ):
            # The classifier resolves whether the request selects a stored
            # catalog property. Keep a real item/category filter only when the
            # semantic check confirms it; task wording is not a product name.
            plan.tipo = "consulta"
            if propiedad_catalogo_aplicable:
                texto_eleg_norm = normalizar(preprocesar_texto_inventario(texto or ""))
                if re.search(
                    r"\b(?:consumibles?|no\s+(?:prestables?|retornables?|devolutiv[oa]s?|se\s+pueden?\s+prestar))\b",
                    texto_eleg_norm,
                ):
                    es_prestable_raw = False
                plan.filtros.es_prestable = es_prestable_raw
                # The property used to select the collection is already true
                # for every result; echoing it as a separate output column is
                # redundant unless the user asks for another attribute.
                plan.campos = [campo for campo in plan.campos if campo != "es_prestable"]
                if not plan.campos and plan.medida_solicitada == "atributo":
                    plan.medida_solicitada = "ninguna"
                if (
                    relacion_stock == "con_stock"
                    and not re.search(
                        r"\b(?:disponibles?|con\s+(?:stock|existencias?)|ahora(?:\s+mismo)?|en\s+stock|que\s+queden|que\s+haya)\b",
                        texto_eleg_norm,
                    )
                ):
                    relacion_stock = "sin_condicion"
                    condicion_validada = None
                    plan.filtros.condicion = None
                    if intencion == "listar_disponibles":
                        intencion = "consulta_inventario"
            else:
                plan.filtros.es_prestable = None
            if filtro_texto_es_entidad is False:
                plan.filtro_texto_es_entidad_verificado = False
                plan.filtros.texto = None
                if not plan.filtros.codigos:
                    plan.producto = None
            if consulta_saldo_pendiente_aplicable:
                telemetria["_saldo_pendiente_global_plan_previo"] = {
                    "tipo": plan.tipo,
                    "operacion": plan.operacion,
                    "medida_solicitada": plan.medida_solicitada,
                    "campos": list(plan.campos or []),
                    "alcance": plan.alcance,
                    "tipo_movimiento": plan.tipo_movimiento,
                    "pregunta": plan.pregunta,
                    "condicion": plan.filtros.condicion,
                    "umbral": plan.filtros.umbral,
                }
                plan.filtros.condicion = "con_prestamos"
                condicion_validada = None
                plan.operacion = "sumar_unidades"
                plan.campos = ["prestamos"]
                plan.medida_solicitada = "pendiente_devolucion"
                plan.alcance = "actual"
                plan.tipo_movimiento = None
            elif consulta_prestamos_activos_aplicable:
                plan.filtros.condicion = "con_prestamos"
                condicion_validada = None
                if not pide_saldo_pendiente and plan.operacion != "contar":
                    plan.operacion = "listar"
                    plan.campos = []
                    plan.medida_solicitada = "ninguna"
            elif relacion_stock == "sin_condicion":
                plan.filtros.condicion = None
                condicion_validada = None
            plan.alcance_entidad = "coleccion"
            plan.alcance = "actual"
            plan.alcance_global_confirmado = True
            plan.referencia = "ninguna"
            plan.pregunta = None
            if plan.operacion not in {"contar", "sumar_unidades"}:
                plan.operacion = "listar"
            plan = _recortar_campos_sobregenerados_de_coleccion_stock(texto, plan)
        bloquea_con_stock_entidad_sec = bool(
            condicion_validada == "con_stock"
            and (
                bool(plan.filtros.codigos)
                or plan.alcance_entidad == "producto"
                or plan.referencia == "foco"
                or (bool(plan.filtros.texto) and not filtro_de_producto_sin_anclar and intencion != "listar_disponibles")
            )
        )
        if (
            condicion_validada in CONDICIONES_VALIDAS
            and not bloquea_con_stock_entidad_sec
            and plan.alcance != "evento"
            and intencion in {
                "consulta_inventario", "listar_catalogo", "listar_disponibles",
                "consulta_entidad", "ambigua",
            }
        ):
            if intencion == "ambigua":
                intencion = "consulta_inventario"
            plan.filtros.condicion = condicion_validada
            plan.filtros.umbral = None
            if filtro_de_producto_sin_anclar:
                plan.filtros.texto = None
            plan.alcance_entidad = "coleccion"
            plan.alcance = "actual"
            if plan.operacion in {"detalle", "comparar"}:
                plan.operacion = "listar"
                plan.campos = []
        condicion_stock = (
            condicion_validada or
            ("con_stock" if intencion == "listar_disponibles" else "sin_condicion")
        )
        telemetria["clasificacion_listado_global"] = {
            "intencion": intencion,
            "accion_solicitada": accion_solicitada,
            "clasificacion_prestamo": clasificacion_prestamo,
            "condicion_stock": condicion_stock,
            "relacion_stock": relacion_stock,
            "propiedad_prestable_solicitada": propiedad_prestable_solicitada,
            "es_prestable": es_prestable_raw,
            "filtro_texto_es_entidad": filtro_texto_es_entidad,
            "prestamos_activos": prestamos_activos,
            "pide_saldo_pendiente": pide_saldo_pendiente,
        }
        telemetria["duracion_ms"] = float(telemetria.get("duracion_ms", 0.0) or 0.0) + float(
            getattr(respuesta, "duracion_ms", 0.0) or duracion_ms
        )

    if intencion == "operacion_inventario" and not _texto_menciona_verbo_mutacion(texto):
        telemetria.setdefault("clasificacion_listado_global", {})["operacion_descartada_sin_verbo"] = True
        intencion = "consulta_entidad" if not sin_entidad_concreta else "consulta_inventario"

    if intencion == "operacion_inventario":
        if accion_solicitada is None:
            plan.tipo = "aclaracion"
            plan.operacion = None
            plan.filtros = FiltrosConsulta()
            plan.campos = []
            plan.referencia = "ninguna"
            plan.alcance = None
            plan.tipo_movimiento = None
            plan.pregunta = "¿Qué operación de inventario desea realizar?"
            return plan
        plan.tipo = "operacion"
        plan.operacion = None
        plan.accion = accion_solicitada
        texto_entidad = (
            plan.filtros.texto
            if filtro_texto_es_entidad is True
            else None
        )
        codigos_mencionados = set(extraer_codigos_de_texto(texto))
        categorias_mencionadas = _menciones_catalogo(
            texto, categorias_accesibles or [],
        )
        almacenes_mencionados = _menciones_almacenes_catalogo(
            texto, almacenes_accesibles or [],
        )
        ubicaciones_mencionadas = _resolver_menciones_ubicacion(
            texto, ubicaciones_accesibles or [],
        )
        plan.filtros = FiltrosConsulta(
            texto=texto_entidad,
            codigos=[
                codigo for codigo in (plan.filtros.codigos or [])
                if codigo in codigos_mencionados
            ],
            categoria=categorias_mencionadas[0] if len(categorias_mencionadas) == 1 else None,
            almacen=almacenes_mencionados[0] if len(almacenes_mencionados) == 1 else None,
            ubicacion=ubicaciones_mencionadas[0] if len(ubicaciones_mencionadas) == 1 else None,
            categoria_explicita=len(categorias_mencionadas) == 1,
        )
        if texto_entidad is None:
            plan.producto = None
        elif not plan.producto:
            plan.producto = texto_entidad
        plan.campos = []
        plan.referencia = "ninguna"
        plan.alcance = None
        plan.alcance_entidad = None
        plan.alcance_global_confirmado = False
        plan.pregunta = None
        return plan

    if intencion == "capacidades":
        plan.tipo = "capacidades"
        plan.operacion = None
        plan.filtros = FiltrosConsulta()
        plan.campos = []
        plan.referencia = "ninguna"
        plan.alcance = None
        plan.tipo_movimiento = None
        plan.ubicaciones_comparadas = []
        return plan

    if intencion == "social":
        plan.tipo = "social"
        plan.operacion = None
        plan.filtros = FiltrosConsulta()
        plan.campos = []
        plan.referencia = "ninguna"
        plan.alcance = None
        plan.tipo_movimiento = None
        plan.ubicaciones_comparadas = []
        return plan

    if intencion == "fuera_dominio":
        plan.tipo = "fuera_dominio"
        plan.operacion = None
        plan.filtros = FiltrosConsulta()
        plan.campos = []
        plan.referencia = "ninguna"
        plan.alcance = None
        plan.tipo_movimiento = None
        plan.ubicaciones_comparadas = []
        return plan

    if intencion == "consulta_inventario":
        return plan

    if intencion == "consulta_entidad":
        return plan

    if intencion == "listar_catalogo":
        plan.alcance_global_confirmado = True
        if solo_con_stock:
            plan.filtros.condicion = None
        if filtro_es_marco_global:
            plan.filtros.texto = None
        if detalle_con_marco_global:
            plan.operacion = "listar"
            plan.campos = []
            plan.referencia = "ninguna"
        plan.alcance_entidad = "global"
        plan = _recortar_campos_sobregenerados_de_coleccion_stock(texto, plan)
    elif intencion == "listar_disponibles":
        plan.alcance_global_confirmado = True
        plan.filtros.condicion = "con_stock"
        if filtro_es_marco_global:
            plan.filtros.texto = None
        if detalle_con_marco_global:
            plan.operacion = "listar"
            plan.campos = []
            plan.referencia = "ninguna"
        plan.alcance_entidad = "global"
        plan = _recortar_campos_sobregenerados_de_coleccion_stock(texto, plan)

    if intencion == "ambigua":
        plan.tipo = "aclaracion"
        plan.operacion = None
        plan.filtros = FiltrosConsulta()
        plan.campos = []
        plan.referencia = "ninguna"
        plan.alcance = None
        plan.tipo_movimiento = None
        plan.ubicaciones_comparadas = []
        if re.search(r"\b(hacu|haku|ayuda|funcion|funciones|sirve|puedes|hacer|productos?|inventario|stock|almacen)\b", normalizar(texto or "")):
            plan.pregunta = "¿Desea conocer las funciones de Hacu Inventario o consultar los productos actuales?"
        else:
            plan.pregunta = "No logré comprender bien su solicitud. ¿Podría repetirla o escribirla, por favor?"
    return plan


def _es_peticion_listado(texto: str, plan: PlanAgente) -> bool:
    """Detecta una petición de registros a partir de la forma de la consulta."""
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    if re.search(
        r"^\s*(?:lista(?:r|do)?|enumera(?:r)?|muestra(?:r|me)?|dame\s+(?:un\s+)?listado)\b",
        texto_norm,
    ):
        return True

    pregunta_por_elementos = bool(
        re.search(r"\b(?:que|cual(?:es)?)\b", texto_norm)
        and re.search(
            r"\b(?:hay|estan|tienen|aparecen|figuran|existen|manejamos|contamos)\b",
            texto_norm,
        )
        and re.search(r"\b(?:catalogo|inventario|vitrina|estante|almacen|bodega)\b", texto_norm)
    )
    alcance_catalogo = bool(
        plan.filtros.codigos
        or plan.filtros.texto
        or plan.filtros.categoria
        or plan.filtros.almacen
        or plan.filtros.ubicacion
        or plan.filtros.condicion
        or plan.alcance_entidad in {"producto", "coleccion", "global"}
    )
    pregunta_cuales = bool(
        re.search(r"\b(?:que|cual(?:es)?)\b", texto_norm)
        and re.search(r"\b(?:productos?|articulos?|referencias?|elementos?|existencias?)\b", texto_norm)
        and re.search(r"\b(?:hay|estan|tienen|aparecen|figuran|existen|manejamos|contamos)\b", texto_norm)
    )
    pregunta_por_registros_de_coleccion = bool(
        plan.alcance_entidad == "coleccion"
        and re.search(r"\b(?:que|cual(?:es)?)\b", texto_norm)
        and re.search(
            r"\b(?:hay|estan|tienen|aparecen|figuran|existen|manejamos|contamos|tenemos)\b",
            texto_norm,
        )
        and not re.search(
            r"\b(?:cuant\w*|cantidad|numero|cifra|total|suma|saldo)\b",
            texto_norm,
        )
    )
    return bool(
        (alcance_catalogo and (pregunta_por_elementos or pregunta_cuales))
        or pregunta_por_registros_de_coleccion
    )


def _pide_recuento_filas(texto_norm: str) -> bool:
    """Reconoce que se cuentan referencias, no el saldo físico de una clase."""
    return bool(
        re.search(
            r"\b(?:cu[aá]nt[oa]s?|n[uú]mero\s+de|cantidad\s+de)\s+"
            r"(?:los?\s+|las?\s+|de\s+)?"
            r"(?:productos?|art[ií]culos?|referencias?|registros?|modelos?|tipos?|categor[ií]as?|"
            r"almacenes?|items?|sku)\b",
            texto_norm,
        )
        or re.search(
            r"\b(?:productos?|referencias?|registros?|modelos?)\b.{0,32}\b"
            r"(?:distintos?|diferentes?|registrados?)\b",
            texto_norm,
        )
    )


def _normalizar_intencion_conteo(texto: str, plan: PlanAgente) -> PlanAgente:
    """Alinea el formato de salida con una petición inequívoca de cardinalidad."""
    if plan.tipo != "consulta":
        return plan
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    pide_unidades = bool(re.search(r"\bunidad(?:es)?\b", texto_norm))
    pide_recuento_filas = not pide_unidades and _pide_recuento_filas(texto_norm)
    menciona_historial_explicito = bool(re.search(
        r"\b(?:movimientos?|historial(?:es)?|transacciones?|devoluciones?)\b",
        texto_norm,
    ))
    if plan.medida_solicitada in {"pendiente_devolucion", "evento"} and (
        not pide_recuento_filas
        or (plan.medida_solicitada == "evento" and menciona_historial_explicito)
    ):
        # Conserva una dimensión temporal cuando la pregunta realmente pide
        # movimientos, historial o devoluciones. Un plan preliminar etiquetado
        # como temporal no debe vencer una petición explícita de contar
        # productos; «productos» cuenta referencias, no unidades prestadas.
        return plan
    if not pide_unidades and _es_peticion_listado(texto, plan):
        # La forma de catálogo prevalece sobre una medida física propuesta por
        # el modelo. Conserva los filtros dinámicos y los atributos explícitos
        # que una normalización anterior ya reconoció para las filas listadas.
        plan.operacion = "listar"
        campos_solicitados = [
            campo for campo in (plan.campos or []) if campo in CAMPOS_VALIDOS
        ]
        if plan.medida_solicitada == "atributo" and campos_solicitados:
            plan.campos = list(dict.fromkeys(campos_solicitados))
        else:
            plan.campos = []
            plan.medida_solicitada = "ninguna"
        plan.alcance = "actual"
        plan.alcance_entidad = "coleccion"
        plan.pregunta = None
        return plan

    pide_solo_cifra = bool(
        re.search(r"\b(?:solo|unicamente)\b.{0,28}\b(?:numero|cifra)\s+(?:de|total\s+de)\b", texto_norm)
    )
    pide_cantidad_registros = bool(
        re.search(r"\bcuant[oa]s?\b", texto_norm)
        or re.search(r"\b(?:numero|cifra|cantidad)\s+de\b", texto_norm)
    )
    pide_lista_explicita = bool(
        re.search(r"^\s*(?:lista(?:r)?|enumera|muestra|muestrame)\b", texto_norm)
        or re.search(r"\b(?:cuales|nombres?|codigos?)\b", texto_norm)
    )

    if (
        tiene_referencia_resultados_previos(texto)
        and plan.operacion not in {"mayor_stock", "menor_stock"}
        and not any((
            plan.filtros.categoria,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
            plan.filtros.condicion,
            plan.filtros.es_prestable is not None,
        ))
    ):
        plan.referencia = "resultados_previos"

    cuenta_coleccion_previa = bool(
        plan.referencia == "resultados_previos"
        and plan.operacion in {"detalle", "listar", "sumar_unidades", "contar", None}
        and pide_cantidad_registros
        and not pide_unidades
        and not pide_lista_explicita
        and not re.search(r"\b(?:movimientos?|historial|transacciones?)\b", texto_norm)
    )
    if cuenta_coleccion_previa:
        plan.operacion = "contar"
        plan.campos = []
        plan.medida_solicitada = "producto"
        plan.alcance = None
        plan.tipo_movimiento = None
        return plan
    if (
        plan.referencia == "resultados_previos"
        and plan.operacion == "contar"
        and pide_cantidad_registros
        and not pide_unidades
        and not pide_lista_explicita
    ):
        # Un conteo de la colección conversacional previa no se convierte en
        # saldo físico por aplicar reglas de cantidad de una entidad nueva.
        plan.campos = []
        plan.alcance = None
        return plan

    if pide_recuento_filas and plan.operacion in {"detalle", "listar", "contar", "sumar_unidades"}:
        # La cardinalidad se decide por lo que se pidió en el mensaje, no por
        # una dimensión que el modelo pudo haber asignado por error.
        plan.operacion = "contar"
        plan.campos = []
        plan.medida_solicitada = "producto"
        plan.alcance_entidad = "coleccion"
        plan.alcance = None
        plan.tipo_movimiento = None
        plan.pregunta = None
        return plan
    pide_elementos_de_categoria = bool(
        plan.filtros.categoria
        and plan.operacion in {"contar", "sumar_unidades", "detalle", "listar"}
        and plan.referencia not in {"foco", "resultados_previos"}
        and re.match(r"^\s*que\b", texto_norm)
        and re.search(r"\b(?:hay|existen|tenemos|tienen|manejamos|contamos)\b", texto_norm)
        and not re.search(
            r"\b(?:cu[aá]nt\w*|cantidad|n[uú]mero|total|suma|stock|existencias?|unidades?|disponibles?)\b",
            texto_norm,
        )
    )
    if pide_elementos_de_categoria:
        # «Qué hay/tenemos de esta categoría» pide registros identificables.
        # El nombre de la categoría viene del catálogo vigente y se conserva
        # como filtro; no se transforma la colección en una suma de stock.
        plan.operacion = "listar"
        plan.campos = []
        plan.medida_solicitada = "ninguna"
        plan.alcance = "actual"
        plan.alcance_entidad = "coleccion"
        plan.pregunta = None
        return plan

    pide_cantidad_de_entidad = bool(re.search(r"\bcu[aá]nt[oa]s?\b", texto_norm))
    if (pide_cantidad_de_entidad or bool(set(plan.campos or []) - {"stock", "prestamos"})) and _preservar_atributos_compuestos(plan):
        # La pregunta puede pedir una cantidad o stock junto con otros atributos de la
        # misma entidad (p. ej. ubicación o stock mínimo). En ese caso, el stock es
        # una columna más y no debe borrar los demás atributos solicitados.
        return plan

    pide_stock_de_entidad_singular = bool(
        plan.filtros.texto
        and plan.operacion in {"contar", "detalle"}
        and not pide_recuento_filas
        and plan.alcance_entidad != "coleccion"
        and re.search(r"\b(?:stock|existencias?|disponibilidad)\b", texto_norm)
        and (
            plan.alcance_entidad == "producto"
            or re.search(r"\b(?:el|la|un|una|este|esta|ese|esa)\s+\S+", texto_norm)
        )
    )
    if pide_stock_de_entidad_singular:
        # Una consulta de stock para una entidad en singular debe pasar por la
        # resolución de ambigüedad del catálogo. Sumar de inmediato puede
        # combinar referencias distintas que solo comparten parte del nombre.
        plan.operacion = "detalle"
        plan.campos = ["stock"]
        plan.medida_solicitada = "atributo"
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        return plan

    pide_dato_temporal = bool(
        re.search(r"\b(?:pr[eé]stw*|movimientw*|historial|devoluci[oó]n|devueltw*)\b", texto_norm)
        or plan.alcance in {"evento", "ambos"}
        or "movimientos" in (plan.campos or [])
    )
    tiene_entidad_o_conjunto = bool(
        plan.filtros.codigos
        or plan.filtros.texto
        or plan.filtros.categoria
        or plan.filtros.almacen
        or plan.filtros.ubicacion
        or plan.alcance_entidad in {"producto", "coleccion"}
    )
    if (
        plan.operacion in {"contar", "listar", "detalle"}
        and pide_cantidad_de_entidad
        and not pide_recuento_filas
        and not pide_dato_temporal
        and not plan.filtros.condicion
        and (tiene_entidad_o_conjunto or plan.medida_solicitada == "disponible")
    ):
        # Una cantidad de una entidad de inventario es saldo físico. El conteo
        # de filas exige que el mensaje delimite explícitamente referencias o
        # registros; no se infiere por la morfología plural del producto.
        plan.tipo = "consulta"
        plan.operacion = "sumar_unidades"
        plan.campos = ["stock"]
        plan.medida_solicitada = "disponible"
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
        return plan

    if plan.operacion != "listar":
        return plan

    if pide_solo_cifra or (pide_cantidad_registros and not pide_unidades and not pide_lista_explicita):
        plan.operacion = "contar"
        plan.campos = []
    return plan


def _normalizar_intencion_suma_unidades(
    texto: str,
    plan: PlanAgente,
    categorias_accesibles: Optional[List[str]] = None,
) -> PlanAgente:
    """Separa una suma física de unidades de un conteo de referencias o un detalle."""
    if plan.tipo not in {"consulta", "aclaracion"} or (
        plan.tipo == "consulta"
        and plan.operacion not in {"listar", "contar", "sumar_unidades", "detalle"}
    ):
        return plan
    if plan.campos == ["categoria"] and plan.filtros.vacio():
        return plan
    entidad_individual = bool(
        plan.alcance_entidad == "producto"
        and (
            plan.filtros.codigos
            or plan.filtros.texto
            or _producto_es_entidad_concreta(plan.producto)
            or plan.referencia in {"foco", "resultados_previos"}
        )
    )
    campos_compuestos = set(plan.campos or []) - {"stock"}
    if (
        plan.tipo == "consulta"
        and plan.operacion in {"detalle", "sumar_unidades"}
        and plan.medida_solicitada == "disponible"
        and entidad_individual
        and not campos_compuestos
        and plan.alcance not in {"evento", "ambos"}
    ):
        # El contrato tipado ya expresa que se consulta la disponibilidad de
        # un producto. No conviertas esa cantidad en una ficha completa solo
        # porque el mensaje no use una forma interrogativa de «cuánto».
        plan.operacion = "sumar_unidades"
        plan.alcance = "actual"
        plan.campos = ["stock"]
        plan.pregunta = None
        return plan
    if plan.medida_solicitada in {"pendiente_devolucion", "evento"}:
        # La suma puede ser de préstamos activos o de otra dimensión tipada;
        # la normalización de cantidades físicas no debe sustituirla.
        return plan

    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    if not re.search(r"\bunidad(?:es)?\b", texto_norm) and _es_peticion_listado(texto, plan):
        plan.operacion = "listar"
        campos_solicitados = [
            campo for campo in (plan.campos or []) if campo in CAMPOS_VALIDOS
        ]
        if plan.medida_solicitada != "atributo" or not campos_solicitados:
            plan.campos = []
            plan.medida_solicitada = "ninguna"
        else:
            # Un listado de productos también puede pedir columnas. No borres
            # esos atributos solo porque la persona no pidió una cantidad.
            plan.campos = campos_solicitados
        plan.alcance = "actual"
        if plan.alcance_entidad != "global":
            plan.alcance_entidad = "coleccion"
        plan.pregunta = None
        return plan
    if (
        plan.operacion == "detalle"
        and plan.medida_solicitada == "atributo"
        and plan.filtros.texto
        and re.search(r"\b(?:stock|existencias?|disponibilidad)\b", texto_norm)
        and re.search(r"\b(?:el|la|un|una|este|esta|ese|esa)\s+\S+", texto_norm)
    ):
        # Mantener el nombre singular como una entidad que puede ser ambigua.
        # El detalle permite devolver el saldo si es único y pedir precisión si
        # el catálogo contiene varias referencias compatibles.
        return plan

    pide_stock_directo = bool(
        re.search(r"\b(?:stock|existencias?|disponibilidad)\s+(?:actual(?:es)?\s+)?(?:de|del)\b", texto_norm)
    )
    pide_cantidad = bool(re.search(r"\b(?:cu[aá]nt[oa]s?|cantidad(?:es)?)\b", texto_norm))
    pide_recuento = _pide_recuento_filas(texto_norm) and not re.search(
        r"\bunidad(?:es)?\b", texto_norm,
    )
    pide_unidades = bool(
        re.search(r"\bcuant[oa]s?\s+(?:(?:las?|del)\s+)?unidades?\b", texto_norm)
        or pide_stock_directo
        or (
            plan.medida_solicitada == "ambos"
            and re.search(r"\bunidad(?:es)?\b", texto_norm)
        )
        or (
            pide_cantidad
            and not pide_recuento
            and plan.operacion in {"contar", "listar", "detalle", "sumar_unidades"}
        )
    )
    if (
        plan.referencia == "resultados_previos"
        and plan.operacion == "contar"
        and pide_cantidad
        and not pide_stock_directo
        and not re.search(r"\bunidad(?:es)?\b", texto_norm)
    ):
        return plan
    if not pide_unidades or (
        plan.alcance in {"evento", "ambos"}
        and plan.medida_solicitada != "ambos"
    ):
        return plan

    # Un filtro de categoría también puede aparecer copiado en campos por el
    # modelo. No es una columna pedida cuando la solicitud cuantifica el saldo
    # de esa categoría. Solo se conserva si pregunta explícitamente por ella.
    pide_categoria_como_atributo = bool(re.search(
        r"\b(?:que\s+categor[ií]a|cu[aá]l\s+(?:es\s+(?:su|la)\s+)?categor[ií]a)\b",
        texto_norm,
    ))
    if (
        plan.filtros.categoria
        and "categoria" in (plan.campos or [])
        and not pide_categoria_como_atributo
    ):
        plan.campos = [campo for campo in plan.campos if campo != "categoria"]
        if (
            plan.medida_solicitada == "atributo"
            and set(plan.campos) == {"stock"}
        ):
            plan.medida_solicitada = "disponible"

    # "ambos" es una dimensión de saldo, no una instrucción para enumerar.
    # Cuando el plan confirma ambas dimensiones (o no trae campos), una
    # pregunta de cantidad se normaliza directamente al agregado tipado.
    dimensiones = set(plan.campos or []) & {"stock", "prestamos"}
    campos_no_saldo = set(plan.campos or []) - {"stock", "prestamos"}
    if (
        plan.medida_solicitada == "ambos"
        and not campos_no_saldo
        and dimensiones in (set(), {"stock", "prestamos"})
    ):
        plan.tipo = "consulta"
        plan.operacion = "sumar_unidades"
        plan.campos = ["stock", "prestamos"]
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
        return plan

    if pide_stock_directo:
        # El nombre de un campo extra generado por el modelo no vuelve la
        # solicitud compuesta. Conserva campos expresamente pedidos por la
        # persona: la coincidencia literal cubre atributos y contexto, y la
        # clasificación semántica cubre paráfrasis del atributo de stock.
        campos_explicitos = [
            campo for campo in plan.campos
            if normalizar(campo.replace("_", " "))
            and re.search(
                rf"(?<!\w){re.escape(normalizar(campo.replace('_', ' ')))}(?!\w)",
                texto_norm,
            )
        ]
        if plan.atributos_stock_verificados is not None:
            campos_explicitos.extend(
                campo
                for campo in plan.atributos_stock_verificados
                if campo in {"stock", "stock_minimo"}
            )
        if "stock" not in campos_explicitos:
            campos_explicitos.append("stock")
        plan.campos = list(dict.fromkeys(campos_explicitos))

    campos_plan = set(plan.campos or [])
    campos_adicionales = campos_plan - {"stock", "prestamos"}
    if (
        plan.tipo == "consulta"
        and plan.operacion in {"detalle", "sumar_unidades"}
        and pide_unidades
        and (
            plan.medida_solicitada in {None, "atributo", "ninguna", "disponible"}
            or plan.atributos_stock_verificados is not None
        )
        and "stock" in campos_plan
        and campos_adicionales
        and plan.alcance not in {"evento", "ambos"}
    ):
        # El plan del modelo a veces agrega el almacén como contexto de la
        # entidad al responder una cantidad. Contrasta esos campos con los
        # atributos que se pueden recuperar del mensaje y del esquema actual,
        # incluyendo los marcos naturales de ubicación del normalizador.
        plan_solicitudes_textuales = copy.deepcopy(plan)
        plan_solicitudes_textuales.campos = []
        plan_solicitudes_textuales.medida_solicitada = "atributo"
        plan_solicitudes_textuales = _completar_atributos_canonicos_solicitados(
            texto,
            plan_solicitudes_textuales,
        )
        plan_solicitudes_textuales = _normalizar_peticion_atributo(
            texto,
            plan_solicitudes_textuales,
        )
        campos_adicionales_explicitos = (
            set(plan_solicitudes_textuales.campos or []) - {"stock", "prestamos"}
        )
        atributos_stock_verificados = plan.atributos_stock_verificados
        if atributos_stock_verificados is None:
            # Sin una verificación semántica disponible, no descartes una
            # interpretación tipada del plan para un umbral expresado con una
            # paráfrasis. En cambio, los atributos de contexto (como almacén)
            # siguen requiriendo una señal independiente en el mensaje.
            if "stock_minimo" in campos_adicionales:
                campos_adicionales_explicitos.add("stock_minimo")
        else:
            campos_adicionales_explicitos.update(
                set(atributos_stock_verificados) & campos_adicionales
            )
        campos_adicionales_no_pedidos = campos_adicionales - campos_adicionales_explicitos
        if campos_adicionales_no_pedidos:
            plan.campos = [
                campo for campo in (plan.campos or [])
                if campo not in campos_adicionales_no_pedidos
            ]
            if set(plan.campos or []) == {"stock"}:
                plan.tipo = "consulta"
                plan.operacion = "sumar_unidades"
                plan.campos = ["stock"]
                plan.medida_solicitada = "disponible"
                plan.alcance = "actual"
                plan.tipo_movimiento = None
                plan.pregunta = None
                return plan

    # La estructura tipada ya expresa que además de la cantidad se pidieron
    # otros atributos. No convertir esa consulta compuesta en una suma de stock.
    if _preservar_atributos_compuestos(plan):
        # Si la consulta compuesta también pregunta cuántas unidades tiene la
        # entidad, el stock es uno de sus atributos de salida. La respuesta no
        # debe conservar el mínimo y la ubicación dejando fuera la cantidad.
        campos_temporales = {"prestamos", "movimientos"}
        if (
            pide_unidades
            and not campos_temporales.intersection(plan.campos or [])
            and plan.alcance not in {"evento", "ambos"}
            and "stock" not in (plan.campos or [])
        ):
            plan.campos.append("stock")
        return plan
    if re.search(r"\b(?:prest\w*|movimient\w*|devoluci\w*|originalmente|historico|historial)\b", texto_norm):
        return plan

    # Si el modelo puso en texto una categoría accesible, reubícala al filtro
    # de categoría. La lista procede del inventario activo, no del código.
    objeto = _extraer_objeto_consulta_sin_marco(
        re.sub(r"\b(?:en\s+)?total(?:es)?\b", " ", texto or "", flags=re.IGNORECASE)
    )
    categoria_objeto = next(
        (cat for cat in categorias_accesibles or [] if objeto and normalizar(cat) == normalizar(objeto)),
        None,
    )
    if categoria_objeto:
        plan.filtros.categoria = categoria_objeto
        plan.filtros.categoria_explicita = True
        plan.filtros.texto = None
    elif plan.filtros.categoria:
        plan.filtros.texto = None

    plan.operacion = "sumar_unidades"
    plan.alcance = "actual"
    plan.campos = ["stock"]
    if plan.medida_solicitada in {"atributo", "ninguna"}:
        plan.medida_solicitada = "disponible"
    return plan


def _normalizar_peticion_atributo(texto: str, plan: PlanAgente) -> PlanAgente:
    """Da prioridad a una ficha o propiedad pedida sobre la medida sugerida."""
    if plan.tipo != "consulta" or plan.operacion not in {
        "detalle", "listar", "contar", "sumar_unidades",
    }:
        return plan
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    medida_original = (plan.medida_solicitada or "").strip().casefold()
    if medida_original in {"pendiente_devolucion", "evento", "ambos"}:
        # Una medida combinada puede coexistir con una ubicación pedida como
        # salida; solo se deja intacta si el mensaje no pide esa propiedad.
        if not re.search(
            r"\b(?:donde|ubicacion|guardad\w*|situad\w*|almacen|bodega|deposito)\b",
            texto_norm,
        ):
            return plan
    filtro_entidad = bool(
        plan.filtros.codigos or plan.filtros.texto or plan.producto
        or plan.referencia == "foco"
    )
    pide_ubicacion_explicita = bool(re.search(
        r"\bubicacion\b|"
        r"\b(?:en\s+)?(?:que|cual)\s+(?:(?:el|la|los|las)\s+)?(?:estante|estanteria)\b",
        texto_norm,
    ))
    pide_almacen = bool(
        re.search(r"\b(?:almacen|bodega|deposito)\b", texto_norm)
        and not re.search(
            r"\bfuera\s+de(?:l)?\s+(?:(?:el|la)\s+)?(?:almacen|bodega|deposito)\b",
            texto_norm,
        )
    )
    pide_donde_guardado = bool(re.search(
        r"\b(?:donde|guardad\w*|situad\w*)\b",
        texto_norm,
    ))
    pide_ubicacion = bool(
        pide_ubicacion_explicita
        or pide_donde_guardado
        or re.search(
            r"\b(?:en\s+)?(?:que|cual)\s+(?:(?:el|la|los|las)\s+)?"
            r"(?:estante|estanteria)\b",
            texto_norm,
        )
    )
    campos_lugar_solicitados = []
    if pide_ubicacion:
        campos_lugar_solicitados.append("ubicacion")
    # An explicit request for the physical location means the internal
    # position only. A general "where is it stored?" request needs the
    # containing warehouse as context; an explicit warehouse request asks for
    # that field directly.
    if pide_almacen or (pide_donde_guardado and not pide_ubicacion_explicita):
        campos_lugar_solicitados.append("almacen")
    pide_ficha = bool(re.search(
        r"\b(?:informacion|ficha|detalles?|caracteristicas|datos)\b",
        texto_norm,
    ))
    consulta_sobre_entidad = bool(re.search(
        r"\b(?:consulta|pregunta)\s+(?:acerca|sobre)\b",
        texto_norm,
    ))
    coleccion_filtrada = bool(
        plan.operacion == "listar"
        and plan.filtros.condicion in _CONDICIONES_STOCK_PARA_CLASIFICAR
    )
    if not filtro_entidad and not ((pide_ubicacion or pide_almacen) and coleccion_filtrada):
        return plan
    if not (pide_ubicacion or pide_almacen or pide_ficha or consulta_sobre_entidad):
        return plan

    if coleccion_filtrada and not filtro_entidad:
        plan.campos = list(dict.fromkeys([
            *(campo for campo in (plan.campos or []) if campo in CAMPOS_VALIDOS),
            *campos_lugar_solicitados,
        ]))
        plan.medida_solicitada = "atributo"
        return plan

    campos = [campo for campo in (plan.campos or []) if campo in CAMPOS_VALIDOS]
    # Cuando el usuario nombra una de las dos dimensiones espaciales de forma
    # explícita, no arrastres la otra aunque el LLM la haya incluido como dato
    # complementario. La pregunta genérica «dónde está guardado» sí puede pedir
    # ambas; «en qué almacén» y «en qué ubicación» son preguntas distintas.
    if pide_almacen and not pide_ubicacion and not pide_donde_guardado:
        campos = [campo for campo in campos if campo != "ubicacion"]
    elif pide_ubicacion_explicita and not pide_almacen and not pide_donde_guardado:
        campos = [campo for campo in campos if campo != "almacen"]
    pide_cantidad = bool(re.search(
        r"\b(?:cu[aá]nt[oa]s?|unidad(?:es)?|cantidad)\b",
        texto_norm,
    ))
    stock_explicito = bool(re.search(r"\bstock\b(?!\s+minimo\b)", texto_norm))
    if (
        medida_original == "disponible"
        and not pide_cantidad
        and not stock_explicito
    ):
        campos = [campo for campo in campos if campo != "stock"]
    campos.extend(campos_lugar_solicitados)

    plan.operacion = "detalle"
    plan.medida_solicitada = "atributo"
    plan.alcance = "actual"
    plan.tipo_movimiento = None
    plan.pregunta = None
    plan.campos = list(dict.fromkeys(campos))
    if plan.filtros.codigos or plan.producto:
        plan.alcance_entidad = "producto"
    elif re.search(r"\b(?:el|la|un|una|este|esta|ese|esa|del)\s+\w+", texto_norm):
        plan.alcance_entidad = "indefinido"
    elif plan.alcance_entidad not in {"producto", "coleccion"}:
        # El ejecutor contrastará el descriptor con el catálogo y pedirá
        # precisión si apunta a más de una referencia.
        plan.alcance_entidad = "indefinido"
    return plan


def _fechas_intervalo_iso(intervalo: Any) -> Tuple[Optional[str], Optional[str]]:
    """Convierte un intervalo local [inicio, fin) a fechas ISO inclusivas."""
    inicio = getattr(intervalo, "inicio", None)
    fin = getattr(intervalo, "fin", None)
    if inicio is None or fin is None:
        return None, None
    fecha_desde = inicio.date().isoformat()
    fecha_hasta_obj = fin.date()
    if getattr(fin, "time", lambda: None)() == datetime.time.min:
        fecha_hasta_obj -= datetime.timedelta(days=1)
    return fecha_desde, fecha_hasta_obj.isoformat()


def _planificar_historial_ordinal(texto: str, estado: Any = None) -> Optional[PlanAgente]:
    """Construye el plan inequívoco para el primer/último movimiento de un producto.

    Las etiquetas de tipo de movimiento se toman de los choices vivos del
    modelo; la base de datos de productos no participa en esta clasificación.
    """
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    es_ordinal = bool(re.search(
        r"\b(?:primer\w*|primera\s+vez|inicialmente|originalmente|ultim\w*|mas\s+reciente)\b",
        texto_norm,
    ))
    tipo_movimiento = _inferir_tipo_movimiento_desde_etiqueta(texto, "evento", None)
    # El ordinal debe modificar al evento, no a un período como «última semana».
    vinculo_ordinal_evento = bool(re.search(
        r"\b(?:primer\w*|ultim\w*|mas\s+reciente)\s+(?:movimient\w*|prestam\w*|entrad\w*|salid\w*|devoluci\w*|traslad\w*)\b|"
        r"\b(?:movimient\w*|prestam\w*|entrad\w*|salid\w*|devoluci\w*|traslad\w*)\s+(?:primer\w*|ultim\w*|mas\s+reciente)\b",
        texto_norm,
    ))
    if not es_ordinal or not vinculo_ordinal_evento:
        return None

    codigos = list(dict.fromkeys(extraer_codigos_de_texto(texto or "")))
    descriptor = _extraer_objeto_consulta_sin_marco(texto or "")
    tokens_descriptor = set(tokens_significativos(descriptor or "")) - {"tenia", "ant", "antes", "habia", "cuanto", "cuanta"}
    codigo_foco = getattr(estado, "producto_codigo", None) if estado else None
    if not codigo_foco and isinstance(estado, str):
        foco_match = re.search(r"(?m)^Producto en foco:\s*(\S+)", estado)
        codigo_foco = foco_match.group(1) if foco_match else None
    if codigos:
        filtros = FiltrosConsulta(codigos=codigos)
        referencia = "ninguna"
    elif codigo_foco and not tokens_descriptor:
        filtros = FiltrosConsulta()
        referencia = "foco"
    else:
        return None

    es_primero = bool(re.search(
        r"\b(?:primer\w*|primera\s+vez|inicialmente|originalmente)\b", texto_norm,
    ))
    pregunta_saldo_previo = bool(
        re.search(r"\bantes\b", texto_norm)
        and re.search(r"\b(?:cuant\w*|stock|existenc\w*|disponible\w*|ten[ií]a|hab[ií]a)\b", texto_norm)
    )
    try:
        from ..servicios_temporales import resolver_intervalo_temporal
        intervalo = resolver_intervalo_temporal(texto)
    except Exception:
        intervalo = None
    fecha_desde, fecha_hasta = _fechas_intervalo_iso(intervalo) if intervalo else (None, None)
    return PlanAgente(
        tipo="consulta",
        operacion="detalle",
        filtros=filtros,
        referencia=referencia,
        alcance="evento",
        alcance_entidad="producto",
        tipo_movimiento=tipo_movimiento,
        campos=["stock"] if pregunta_saldo_previo else ["movimientos"],
        orden_movimiento="asc" if es_primero else "desc",
        medida_solicitada="evento",
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
        fuente_plan="fast_path_historial_ordinal",
    )


def _normalizar_historial_temporal(texto: str, plan: PlanAgente) -> PlanAgente:
    """Aplica intervalos temporales y evita confundir ordinals con fechas."""
    if plan.tipo != "consulta":
        return plan
    texto_original = texto or ""
    texto_norm = normalizar(preprocesar_texto_inventario(texto_original))
    try:
        from ..servicios_temporales import resolver_intervalo_temporal
        intervalo = resolver_intervalo_temporal(texto_original)
    except Exception:
        intervalo = None

    if intervalo is not None:
        menciona_eventos_en_periodo = bool(
            "movimientos" in (plan.campos or [])
            or "prestamos" in (plan.campos or [])
            or plan.tipo_movimiento
            or re.search(
                r"\b(?:movimientos?|pr[eé]stamos?|entradas?|salidas?|traslados?|devoluciones?|dotaciones?|ajustes?|"
                r"movi(?:o|eron)|ha\s+movido|prest(?:o|aron)|devolv(?:io|ieron)|"
                r"entr(?:o|aron)|ingres(?:o|aron)|sali(?:o|eron)|despach(?:o|aron)|traslad(?:o|aron))\b",
                texto_norm,
            )
        )
        pide_conteo_eventos = bool(
            re.search(r"\b(?:cuant\w*|numero\s+de|total\s+de|cantidad\s+de)\b", texto_norm)
            and re.search(
                r"\b(?:registr\w*|realiz\w*|ocurr\w*|hicier\w*|hizo|hubo|suced\w*|"
                r"tuvo|tuvieron|present\w*|efectu\w*|salieron|entraron|prestaron|devolvieron)\b",
                texto_norm,
            )
            and menciona_eventos_en_periodo
        )
        if pide_conteo_eventos:
            if (intervalo.inicio.time() != datetime.time.min
                    or intervalo.fin.time() != datetime.time.min):
                # El contrato actual filtra días completos. Convertir un
                # intervalo parcial a fechas incluiría horas que no se pidieron.
                plan.tipo = "aclaracion"
                plan.operacion = None
                plan.campos = []
                plan.fecha_desde = plan.fecha_hasta = None
                plan.pregunta = (
                    "Para dar un conteo exacto, indique las fechas completas "
                    "de inicio y fin del período que desea consultar."
                )
                return plan
            fecha_desde, fecha_hasta = _fechas_intervalo_iso(intervalo)
            entidad_individual_conteo = bool(
                plan.filtros.codigos
                or plan.filtros.texto
                or plan.referencia == "foco"
                or re.search(r"\b(?:ese|esa|este|esta|su|sus)\s+(?:producto|articulo|referencia)\b", texto_norm)
            )
            plan.tipo = "consulta"
            plan.operacion = "detalle" if entidad_individual_conteo else "contar"
            if not entidad_individual_conteo:
                plan.alcance_entidad = "global"
                plan.alcance_global_confirmado = True
            plan.alcance = "evento"
            plan.campos = ["movimientos"]
            plan.medida_solicitada = "evento"
            plan.tipo_movimiento = _inferir_tipo_movimiento_desde_etiqueta(
                texto_original, "evento", plan.tipo_movimiento,
            )
            plan.orden_movimiento = None
            plan.fecha_desde = fecha_desde
            plan.fecha_hasta = fecha_hasta
            if plan.referencia == "resultados_previos" and re.search(
                r"\b(?:ese|esa|este|esta|su|sus)\s+(?:producto|articulo|referencia)\b",
                texto_norm,
            ):
                plan.referencia = "foco"
        elif (
            plan.alcance in {"evento", "ambos"}
            or "movimientos" in (plan.campos or [])
            or (
                menciona_eventos_en_periodo
                and re.search(
                    r"\b(?:movimientos?|registr\w*|realiz\w*|ocurr\w*|hicier\w*|hubo|han\s+habido|"
                    r"tuvo|tuvieron|historial|muestr\w*|enseñ\w*|ensen\w*|ver|dime|dame|list\w*|"
                    r"consult\w*|qu[eé]|cu[aá]les|c[oó]mo\s+van)\b",
                    texto_norm,
                )
            )
        ):
            plan.fecha_desde, plan.fecha_hasta = _fechas_intervalo_iso(intervalo)
            if plan.alcance not in {"evento", "ambos"}:
                plan.alcance = "evento"
            if "movimientos" not in (plan.campos or []) and "ficha" not in (plan.campos or []):
                plan.campos = [c for c in (plan.campos or []) if c not in {"stock", "prestamos"}] + ["movimientos"]
            plan.tipo_movimiento = _inferir_tipo_movimiento_desde_etiqueta(
                texto_original, "evento", plan.tipo_movimiento,
            )
            if (
                not plan.filtros.codigos
                and not plan.filtros.texto
                and plan.referencia == "ninguna"
                and not _extraer_objeto_consulta_sin_marco(texto_original)
            ):
                plan.alcance_entidad = "global"
                plan.alcance_global_confirmado = True
                if plan.operacion in {None, "detalle"}:
                    plan.operacion = "listar"
    else:
        # Conserva fechas absolutas expresadas de forma explícita, pero quita
        # fechas inferidas solo por «último movimiento» o «primer préstamo».
        expresa_fecha_calendario = bool(re.search(
            r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            texto_original,
        ))
        if not expresa_fecha_calendario:
            plan.fecha_desde = None
            plan.fecha_hasta = None

    # «Antes del último movimiento» pide el saldo que quedó antes del evento,
    # no la cantidad movida ni una fecha del calendario.
    if (
        plan.alcance == "evento"
        and re.search(r"\bantes\b", texto_norm)
        and re.search(r"\b(?:ultim\w*|primer\w*|mas\s+reciente)\b", texto_norm)
        and re.search(r"\bmovimient\w*\b", texto_norm)
        and re.search(r"\b(?:cuant\w*|stock|existenc\w*|disponible\w*|ten[ií]a|hab[ií]a)\b", texto_norm)
    ):
        plan.operacion = "detalle"
        plan.campos = ["stock"]
        plan.medida_solicitada = "evento"
        plan.tipo_movimiento = None
        plan.orden_movimiento = "asc" if re.search(r"\b(?:primer\w*|inicialmente|originalmente)\b", texto_norm) else "desc"
        plan.fecha_desde = None
        plan.fecha_hasta = None
        plan.alcance = "evento"
        if plan.referencia == "resultados_previos" and re.search(
            r"\b(?:ese|esa|este|esta|su|sus)\s+(?:producto|articulo|referencia)\b",
            texto_norm,
        ):
            plan.referencia = "foco"
    return plan


def _preservar_atributos_compuestos(plan: PlanAgente) -> bool:
    """Conserva campos tipados múltiples en lugar de colapsarlos a una medida."""
    campos = list(dict.fromkeys(
        campo for campo in (plan.campos or []) if campo in CAMPOS_VALIDOS
    ))
    if plan.tipo != "consulta" or len(campos) < 2:
        return False

    plan.campos = campos
    plan.medida_solicitada = "atributo"
    if plan.operacion in {None, "contar", "sumar_unidades"}:
        entidad_individual = bool(
            plan.alcance_entidad == "producto"
            or plan.filtros.codigos
            or plan.referencia == "foco"
        )
        plan.operacion = "detalle" if entidad_individual else "listar"
    if plan.operacion in {"detalle", "listar"}:
        plan.alcance = "actual"
    return True


def _plan_tiene_referencia_individual_sin_codigo(
    plan: PlanAgente,
    texto: Optional[str] = None,
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> bool:
    """Conserva ambigüedad singular salvo que un nombre vivo ancle un único producto."""
    if plan.filtros.codigos or plan.filtros.categoria or plan.filtros.categoria_explicita:
        return False
    crudo = plan.crudo if isinstance(plan.crudo, dict) else {}
    producto_crudo = str(crudo.get("producto") or "").strip()
    tiene_descriptor = bool(plan.filtros.texto or plan.producto or producto_crudo)
    senal_individual = bool(
        plan.alcance_entidad in {"producto", "indefinido"}
        or plan.producto
        or producto_crudo
    )
    if not tiene_descriptor or not senal_individual:
        return False

    # A nombre completo vigente puede desambiguar la referencia aunque el LLM
    # haya conservado un filtro textual. Los candidatos se leen del catálogo
    # activo en esta petición; los descriptores parciales siguen protegidos.
    mensaje = normalizar(preprocesar_texto_inventario(texto or ""))
    if mensaje and codigos_productos_candidatos:
        nombres_mencionados = {
            codigo
            for codigo, nombre in _obtener_productos_candidatos_para_verificacion(
                codigos_productos_candidatos,
            )
            if normalizar(nombre) and normalizar(nombre) in mensaje
        }
        if len(nombres_mencionados) == 1:
            return False
    return True


def _consulta_saldo_individual_anclada_a_catalogo(
    plan: PlanAgente,
    texto: str,
    codigos_productos_candidatos: Optional[List[str]],
) -> bool:
    """Indica si una suma de saldo actual trata de un producto vivo y único.

    El anclaje se obtiene del catálogo accesible de esta petición (código o
    nombre completo). Esto permite verificar la dimensión del saldo aunque el
    clasificador principal haya producido un plan internamente consistente.
    """
    if (
        plan.operacion != "sumar_unidades"
        or plan.alcance in {"evento", "ambos"}
        or plan.tipo_movimiento
        or not codigos_productos_candidatos
    ):
        return False
    dimensiones = set(plan.campos or []) & {"stock", "prestamos"}
    if not dimensiones or set(plan.campos or []) - {"stock", "prestamos"}:
        return False

    candidatos = _obtener_productos_candidatos_para_verificacion(
        codigos_productos_candidatos,
    )
    codigos_vigentes = {
        normalizar_codigo_producto(codigo)
        for codigo, _ in candidatos
        if normalizar_codigo_producto(codigo)
    }
    codigos_anclados = {
        normalizar_codigo_producto(codigo)
        for codigo in (plan.filtros.codigos or [])
        if normalizar_codigo_producto(codigo) in codigos_vigentes
    }
    codigos_anclados.update(
        normalizar_codigo_producto(codigo)
        for codigo in extraer_codigos_de_texto(texto or "")
        if normalizar_codigo_producto(codigo) in codigos_vigentes
    )
    if len(codigos_anclados) == 1:
        return True

    mensaje = normalizar(preprocesar_texto_inventario(texto or ""))
    nombres_anclados = {
        normalizar(nombre)
        for _, nombre in candidatos
        if normalizar(nombre) and normalizar(nombre) in mensaje
    }
    return len(nombres_anclados) == 1


def _es_pregunta_quien_tiene_prestamo(texto: str) -> bool:
    """Detecta preguntas directas por la persona/destinatario que posee un artículo prestado."""
    texto_norm = normalizar(preprocesar_texto_inventario(texto or "", normalizar_medidas=False))
    if not texto_norm:
        return False
    if re.search(r"\b(?:cuant[oa]s?|cantidad(?:es)?|unidades?|total(?:es)?|saldos?|numero\s+de)\b", texto_norm):
        return False
    return bool(
        re.search(
            r"\b(?:quien(?:es)?\s+(?:(?:lo|la|los|las)\s+)?tiene(?:n)?\b|"
            r"a\s+quien(?:es)?\s+(?:se\s+(?:lo|la|los|las)\s+)?(?:presto|prestaron|esta|estan|fue|fueron)\b|"
            r"a\s+quien(?:es)?\s+le(?:s)?\s+(?:presto|prestaron|entregaron)\b)",
            texto_norm,
        )
    )


def _normalizar_intencion_prestamo_actual(
    texto: str,
    plan: PlanAgente,
    categorias_accesibles: Optional[List[str]] = None,
    foco_disponible: bool = False,
) -> PlanAgente:
    """Aplica la dimensión de saldo que el LLM ya tipó en el plan semántico."""
    if plan.tipo not in {"consulta", "aclaracion"} or (
        plan.tipo == "consulta"
        and plan.operacion not in {"listar", "contar", "sumar_unidades", "detalle"}
    ):
        return plan
    if plan.campos == ["categoria"] and plan.filtros.vacio():
        return plan
    if _es_pregunta_quien_tiene_prestamo(texto):
        es_prod = bool(
            plan.alcance_entidad == "producto"
            or plan.filtros.codigos
            or (plan.referencia == "foco" and foco_disponible)
            or (plan.filtros.texto and plan.alcance_entidad != "coleccion")
        )
        plan.tipo = "consulta"
        plan.pregunta = None
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.campos = ["prestamos"]
        plan.medida_solicitada = "ninguna"
        if plan.operacion == "sumar_unidades" or (es_prod and plan.operacion != "detalle"):
            plan.operacion = "detalle" if es_prod else "listar"
        if not es_prod:
            plan.filtros.condicion = "con_prestamos"
        return plan
    medida = (plan.medida_solicitada or "").strip().casefold()
    if medida == "pendiente_devolucion":
        pass
    elif medida:
        return plan
    elif not (
        plan.operacion == "sumar_unidades"
        and "prestamos" in (plan.campos or [])
        and plan.alcance == "actual"
    ):
        return plan
    es_producto_especifico = bool(
        plan.alcance_entidad == "producto"
        or plan.filtros.codigos
        or (plan.referencia == "foco" and foco_disponible)
        or (plan.filtros.texto and plan.alcance_entidad != "coleccion")
    )
    # La medida proviene del contrato del LLM. El alcance y los filtros
    # distinguen una referencia concreta de una consulta sobre una colección.
    plan.tipo = "consulta"
    plan.pregunta = None
    plan.operacion = "sumar_unidades"
    plan.alcance = "actual"
    plan.tipo_movimiento = None
    plan.campos = ["prestamos"]
    plan.filtros.condicion = None if es_producto_especifico else "con_prestamos"
    if not es_producto_especifico:
        plan.referencia = "ninguna"
    if not es_producto_especifico and not plan.filtros.categoria:
        plan.filtros.texto = None
    return plan


def _verificar_forma_respuesta_prestamos(
    texto: str,
    plan: PlanAgente,
    cliente_llm: Any,
    timeout: Optional[float],
    telemetria: Dict[str, Any],
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> PlanAgente:
    """Desambigua saldo agregado frente a detalle usando el mensaje completo.

    Se activa cuando el plan eligió préstamos como atributo, o cuando el
    mensaje nombra el campo canónico del esquema pero el plan lo omitió. El
    clasificador decide la forma de respuesta sin una lista de paráfrasis.
    """
    campos = set(plan.campos or [])
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    pide_cantidad = bool(re.search(r"\b(?:cuant\w*|cantidad(?:es)?|unidades?)\b", texto_norm))
    campos_canonicos_mencionados = {
        campo
        for campo in CAMPOS_VALIDOS
        if re.search(
            rf"(?<!\w){re.escape(normalizar(campo.replace('_', ' ')))}(?!\w)",
            texto_norm,
        )
    }
    menciona_campo_prestamos = "prestamos" in campos_canonicos_mencionados
    campos_saldo_y_contexto = campos <= {"prestamos", "almacen"} and "prestamos" in campos
    referencia_resuelta = not _plan_tiene_referencia_individual_sin_codigo(
        plan, texto, codigos_productos_candidatos,
    )
    if (
        _es_pregunta_quien_tiene_prestamo(texto)
        or plan.tipo != "consulta"
        or plan.operacion not in {"detalle", "listar"}
        or (
            plan.operacion == "listar"
            and plan.filtros.condicion == "con_prestamos"
            and not re.search(r"\b(?:cuant\w*|cantidad(?:es)?|unidades?|total(?:es)?|saldos?)\b", texto_norm)
        )
        or not (
            campos == {"prestamos"}
            or (campos_saldo_y_contexto and pide_cantidad and referencia_resuelta)
            or (not campos and menciona_campo_prestamos and referencia_resuelta)
        )
        or plan.alcance in {"evento", "ambos"}
        or plan.medida_solicitada not in {None, "atributo", "ninguna"}
    ):
        return plan

    def solicitar_aclaracion() -> None:
        plan.tipo = "aclaracion"
        plan.operacion = None
        plan.campos = []
        plan.medida_solicitada = "ninguna"
        plan.pregunta = (
            "¿Desea el saldo total pendiente de devolución o el detalle de quién tiene cada préstamo activo?"
        )

    texto_usuario = preprocesar_texto_inventario(texto or "", normalizar_medidas=False)[:400]
    mensajes = [
        {"role": "system", "content": PROMPT_CLASIFICADOR_SALDO_PRESTAMOS},
        {"role": "user", "content": f"[Mensaje original]\n{texto_usuario}"},
    ]
    chars_prompt = sum(len(str(m.get("content") or "")) for m in mensajes)
    inicio = time.perf_counter()
    telemetria["llamadas_fisicas"] = int(telemetria.get("llamadas_fisicas", 0) or 0) + 1
    try:
        respuesta = cliente_llm.generar(
            mensajes=mensajes,
            formato_json=True,
            temperatura=0.0,
            max_tokens=16,
            timeout=timeout,
            json_schema=_ESQUEMA_SALDO_PRESTAMOS_JSON,
            json_schema_name="haku_loan_answer_shape",
        )
    except Exception as exc:
        duracion_ms = (time.perf_counter() - inicio) * 1000.0
        plan.llm_ms += duracion_ms
        telemetria["duracion_ms"] = float(telemetria.get("duracion_ms", 0.0) or 0.0) + duracion_ms
        telemetria["clasificacion_saldo_prestamos"] = {
            "modo": "aclarar_por_error",
            "error": type(exc).__name__,
        }
        logger.warning("[HakuAgente] No se pudo desambiguar la salida de préstamos: %s", type(exc).__name__)
        solicitar_aclaracion()
        return plan

    duracion_ms = (time.perf_counter() - inicio) * 1000.0
    duracion_modelo_ms = float(getattr(respuesta, "duracion_ms", 0.0) or duracion_ms)
    plan.llm_ms += duracion_modelo_ms
    plan.llm_cola_ms += float(getattr(respuesta, "espera_cola_ms", 0.0) or 0.0)
    plan.chars_prompt += chars_prompt
    plan.tokens_estimados_prompt += max(1, chars_prompt // 4)
    tokens = getattr(respuesta, "tokens_completados", 0)
    if isinstance(tokens, (int, float)):
        plan.tokens_completados += int(tokens)
    telemetria["duracion_ms"] = float(telemetria.get("duracion_ms", 0.0) or 0.0) + duracion_modelo_ms

    datos = getattr(respuesta, "datos_json", None)
    modo = datos.get("modo") if isinstance(datos, dict) else None
    if modo not in {"saldo_total", "detalle_individual", "otra_consulta", "ambigua"}:
        modo = "ambigua"
    telemetria["clasificacion_saldo_prestamos"] = {"modo": modo}

    if modo == "saldo_total":
        plan.tipo = "consulta"
        plan.operacion = "sumar_unidades"
        plan.campos = ["prestamos"]
        plan.medida_solicitada = "pendiente_devolucion"
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
    elif modo == "detalle_individual":
        plan.tipo = "consulta"
        plan.campos = ["prestamos"]
        plan.medida_solicitada = "atributo"
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
    elif modo == "ambigua":
        solicitar_aclaracion()
    elif modo == "otra_consulta" and menciona_campo_prestamos:
        # La mención del campo canónico del esquema se conserva como atributo
        # explícito; nunca derive una ficha completa al perder ese campo.
        plan.tipo = "consulta"
        plan.operacion = "detalle"
        plan.campos = ["prestamos"]
        plan.medida_solicitada = "atributo"
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
    return plan


def _aplicar_medida_solicitada(plan: PlanAgente) -> PlanAgente:
    """Alinea operación/campos con la dimensión semántica seleccionada por el LLM."""
    medida = (plan.medida_solicitada or "").strip().casefold()
    if (
        plan.tipo != "consulta"
        or plan.operacion in {"listar", "comparar", "mayor_stock", "menor_stock"}
        or plan.alcance == "evento"
    ):
        return plan

    campos_por_medida = {
        "disponible": {"stock"},
        "pendiente_devolucion": {"prestamos"},
        "evento": {"movimientos"},
        "ambos": {"stock", "prestamos"},
    }
    campos_medida = campos_por_medida.get(medida)
    campos_plan = set(plan.campos or [])
    if (
        campos_medida
        and len(campos_plan) > 1
        and not campos_plan.issubset(campos_medida)
        and _preservar_atributos_compuestos(plan)
    ):
        return plan

    # El planificador devuelve dos señales para la dimensión cuantitativa:
    # medida_solicitada y campos. Si marca "ambos" pero solo aporta uno de los
    # dos campos canónicos, no amplíes silenciosamente la consulta: conserva la
    # dimensión que sí tipó en campos. La medida agregada solo es consistente
    # cuando ambos campos aparecen, o cuando el modelo no envió ninguno.
    if medida == "ambos" and campos_plan:
        dimensiones_presentes = campos_plan & {"stock", "prestamos"}
        if dimensiones_presentes == {"stock"}:
            medida = "disponible"
            plan.medida_solicitada = medida
        elif dimensiones_presentes == {"prestamos"}:
            medida = "pendiente_devolucion"
            plan.medida_solicitada = medida

    if medida == "pendiente_devolucion":
        plan.tipo = "consulta"
        plan.operacion = "sumar_unidades"
        plan.campos = ["prestamos"]
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
        plan.filtros.condicion = None
    elif medida == "disponible":
        plan.tipo = "consulta"
        plan.operacion = "sumar_unidades"
        plan.campos = ["stock"]
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
    elif medida == "ambos":
        plan.tipo = "consulta"
        plan.operacion = "sumar_unidades"
        plan.campos = ["stock", "prestamos"]
        plan.alcance = "actual"
        plan.tipo_movimiento = None
        plan.pregunta = None
    elif medida == "evento":
        plan.tipo = "consulta"
        plan.operacion = "detalle"
        plan.campos = ["movimientos"]
        plan.alcance = "evento"
        plan.pregunta = None
    return plan


def _normalizar_saldo_cuantitativo_final(plan: PlanAgente) -> PlanAgente:
    """Hace que una suma de saldo exponga solo la dimensión tipada.

    Un plan agregado puede incluir columnas accesorias que no se pidieron.
    Las solicitudes compuestas pasan por el flujo de atributos y conservan
    `medida_solicitada=atributo`; los saldos tipados deben seguir el contrato
    exacto de stock, préstamos o ambos.
    """
    medidas = {
        "disponible": ["stock"],
        "pendiente_devolucion": ["prestamos"],
        "ambos": ["stock", "prestamos"],
    }
    campos_esperados = medidas.get((plan.medida_solicitada or "").casefold())
    if (
        plan.tipo != "consulta"
        or plan.operacion not in {"detalle", "sumar_unidades"}
        or plan.alcance in {"evento", "ambos"}
        or plan.tipo_movimiento
        or not campos_esperados
        or not (set(plan.campos or []) & {"stock", "prestamos"})
        or bool(set(plan.campos or []) - {"stock", "prestamos"})
    ):
        return plan

    plan.operacion = "sumar_unidades"
    plan.campos = list(campos_esperados)
    plan.alcance = "actual"
    plan.tipo_movimiento = None
    plan.pregunta = None
    return plan


def _verificar_medida_cantidad_semantica(
    texto: str,
    plan: PlanAgente,
    cliente_llm: Any,
    timeout: Optional[float],
    telemetria: Dict[str, Any],
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> PlanAgente:
    """Valida la dimensión de los saldos cuantitativos con el mensaje completo.

    Llama al Cerebro cuando `medida_solicitada` contradice los campos, cuando
    varias preguntas cuantitativas quedaron reducidas a una dimensión o cuando
    una cantidad de una referencia concreta depende de elegir entre stock y
    custodia fuera del inventario. No requiere que el plan primario haya
    detectado la palabra "préstamo" para validar esa dimensión.
    """
    if plan.tipo != "consulta":
        return plan
    medida = (plan.medida_solicitada or "").strip().casefold()
    medidas_cuantitativas = {"disponible", "pendiente_devolucion", "ambos"}
    if medida not in medidas_cuantitativas | {"atributo", "ninguna", ""}:
        return plan

    campos = set(plan.campos or [])
    if bool(campos - {"stock", "prestamos"}):
        return plan
    dimensiones = campos & {"stock", "prestamos"}
    clasificacion_prestamos = telemetria.get("clasificacion_saldo_prestamos")
    if (
        plan.operacion == "sumar_unidades"
        and medida == "pendiente_devolucion"
        and dimensiones == {"prestamos"}
        and isinstance(clasificacion_prestamos, dict)
        and clasificacion_prestamos.get("modo") == "saldo_total"
    ):
        # El clasificador especializado ya resolvió el saldo de préstamos
        # como agregado. Volver a preguntarle al clasificador general puede
        # tratar «saldo de préstamos» como ambiguo y degradar una decisión
        # semántica válida a aclaración (que luego puede acabar en ficha).
        return plan
    entidad_singular_con_texto = _plan_tiene_referencia_individual_sin_codigo(
        plan, texto, codigos_productos_candidatos,
    )
    saldo_individual_catalogo = _consulta_saldo_individual_anclada_a_catalogo(
        plan, texto, codigos_productos_candidatos,
    )
    texto_cuantitativo_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    marcadores_cuantitativos = re.findall(
        r"\b(?:cu[aá]nt\w*|cantidad(?:es)?|total(?:es)?)\b",
        texto_cuantitativo_norm,
    )
    pide_magnitud = bool(
        marcadores_cuantitativos
        or re.search(r"\bunidad(?:es)?\b", texto_cuantitativo_norm)
    )
    dimensiones_por_medida = {
        "disponible": {"stock"},
        "pendiente_devolucion": {"prestamos"},
        "ambos": {"stock", "prestamos"},
    }
    contradice_campos = bool(dimensiones) and medida in medidas_cuantitativas and (
        dimensiones != dimensiones_por_medida[medida]
    )
    varias_cantidades = len(marcadores_cuantitativos) >= 2
    codigos_candidatos_unicos = {
        normalizar_codigo_producto(codigo)
        for codigo in (codigos_productos_candidatos or [])
        if normalizar_codigo_producto(codigo)
    }
    plan_crudo_medida = plan.crudo if isinstance(plan.crudo, dict) else {}
    texto_entidad_plan = " ".join(str(valor or "") for valor in (
        plan.producto,
        plan.filtros.texto,
        plan_crudo_medida.get("producto"),
        *(plan.filtros.codigos or []),
    ))
    codigos_entidad_plan = {
        normalizar_codigo_producto(codigo)
        for codigo in extraer_codigos_de_texto(texto_entidad_plan)
        if normalizar_codigo_producto(codigo)
    }
    codigo_unico_propuesto_por_plan = bool(
        len(codigos_candidatos_unicos) == 1
        and codigos_candidatos_unicos.issubset(codigos_entidad_plan)
    )
    referencia_individual_catalogada = bool(
        len(plan.filtros.codigos or []) == 1
        or codigo_unico_propuesto_por_plan
        or (
            plan.alcance_entidad == "producto"
            and _producto_es_entidad_concreta(
                plan.producto or plan.filtros.texto,
            )
            and not _plan_tiene_referencia_individual_sin_codigo(
                plan,
                texto,
                codigos_productos_candidatos,
            )
        )
    )
    consulta_cuantitativa_individual = bool(
        plan.operacion in {"detalle", "sumar_unidades"}
        and dimensiones
        and pide_magnitud
        and referencia_individual_catalogada
    )
    cantidad_sin_medida = bool(
        medida in {"atributo", "ninguna", ""}
        and dimensiones
        and not (campos - {"stock", "prestamos"})
        and pide_magnitud
    )
    if (
        entidad_singular_con_texto
        and not saldo_individual_catalogo
        and not consulta_cuantitativa_individual
    ):
        return plan
    if (
        not contradice_campos
        and not varias_cantidades
        and not cantidad_sin_medida
        and not saldo_individual_catalogo
        and not consulta_cuantitativa_individual
        and not telemetria.get("_saldo_pendiente_global_plan_previo")
    ):
        return plan
    if campos - {"stock", "prestamos"}:
        # Una cantidad acompañada de otros atributos es una respuesta
        # compuesta; su clasificación no se puede reducir a un único saldo.
        return plan

    def aclarar() -> None:
        plan.tipo = "aclaracion"
        plan.operacion = None
        plan.campos = []
        plan.medida_solicitada = "ninguna"
        plan.alcance = None
        plan.pregunta = (
            "¿Desea conocer las unidades disponibles, las pendientes de devolución o ambas cantidades?"
        )

    if cliente_llm is None:
        telemetria["verificacion_medida_cantidad"] = {"estado": "sin_modelo"}
        aclarar()
        return plan

    texto_usuario = preprocesar_texto_inventario(texto or "", normalizar_medidas=False)[:500]
    mensajes = [
        {"role": "system", "content": PROMPT_CLASIFICADOR_MEDIDA_CANTIDAD},
        {"role": "user", "content": f"[Mensaje original]\n{texto_usuario}"},
    ]
    chars_prompt = sum(len(str(m.get("content") or "")) for m in mensajes)
    inicio = time.perf_counter()
    telemetria["llamadas_fisicas"] = int(telemetria.get("llamadas_fisicas", 0) or 0) + 1
    try:
        respuesta = cliente_llm.generar(
            mensajes=mensajes,
            formato_json=True,
            temperatura=0.0,
            max_tokens=12,
            timeout=timeout,
            json_schema=_ESQUEMA_MEDIDA_CANTIDAD_JSON,
            json_schema_name="haku_inventory_quantity_measure",
        )
    except Exception as exc:
        duracion_ms = (time.perf_counter() - inicio) * 1000.0
        plan.llm_ms += duracion_ms
        telemetria["duracion_ms"] = float(telemetria.get("duracion_ms", 0.0) or 0.0) + duracion_ms
        telemetria["verificacion_medida_cantidad"] = {
            "estado": "aclarar_por_error",
            "error": type(exc).__name__,
        }
        logger.warning(
            "[HakuAgente] No se pudo validar la medida cuantitativa: %s",
            type(exc).__name__,
        )
        aclarar()
        return plan

    duracion_ms = (time.perf_counter() - inicio) * 1000.0
    duracion_modelo_ms = float(getattr(respuesta, "duracion_ms", 0.0) or duracion_ms)
    plan.llm_ms += duracion_modelo_ms
    plan.llm_cola_ms += float(getattr(respuesta, "espera_cola_ms", 0.0) or 0.0)
    plan.chars_prompt += chars_prompt
    plan.tokens_estimados_prompt += max(1, chars_prompt // 4)
    tokens = getattr(respuesta, "tokens_completados", 0)
    if isinstance(tokens, (int, float)):
        plan.tokens_completados += int(tokens)
    telemetria["duracion_ms"] = float(telemetria.get("duracion_ms", 0.0) or 0.0) + duracion_modelo_ms

    datos = getattr(respuesta, "datos_json", None)
    medida_validada = datos.get("medida") if isinstance(datos, dict) else None
    if medida_validada not in {"disponible", "pendiente_devolucion", "ambos"}:
        plan_previo_saldo_global = telemetria.pop(
            "_saldo_pendiente_global_plan_previo", None,
        )
        if isinstance(plan_previo_saldo_global, dict):
            medida_previa = plan_previo_saldo_global.get("medida_solicitada")
            if medida_previa in {"disponible", "pendiente_devolucion", "ambos"}:
                plan.tipo = plan_previo_saldo_global.get("tipo") or "consulta"
                plan.operacion = plan_previo_saldo_global.get("operacion") or "sumar_unidades"
                plan.medida_solicitada = medida_previa
                plan.campos = list(plan_previo_saldo_global.get("campos") or [])
                plan.alcance = plan_previo_saldo_global.get("alcance") or "actual"
                plan.tipo_movimiento = plan_previo_saldo_global.get("tipo_movimiento")
                plan.pregunta = plan_previo_saldo_global.get("pregunta")
                plan.filtros.condicion = plan_previo_saldo_global.get("condicion")
                plan.filtros.umbral = plan_previo_saldo_global.get("umbral")
                telemetria["verificacion_medida_cantidad"] = {
                    "estado": "ambigua_se_conserva_medida_previa",
                    "medida": medida_previa,
                }
                return plan
            telemetria["verificacion_medida_cantidad"] = {
                "estado": "ambigua_se_pide_aclaracion_a_saldo_global",
            }
            aclarar()
            return plan
        dimensiones_por_medida = {
            "disponible": {"stock"},
            "pendiente_devolucion": {"prestamos"},
            "ambos": {"stock", "prestamos"},
        }
        dimension_tipificada = dimensiones_por_medida.get(medida)
        if (
            medida in medidas_cuantitativas
            and dimension_tipificada == dimensiones
            and len(marcadores_cuantitativos) < 2
        ):
            # Una respuesta auxiliar ambigua no debe borrar una magnitud que
            # el plan principal ya tipó coherentemente. El contrato aplica el
            # valor por defecto de disponibilidad a una cantidad sin dimensión.
            telemetria["verificacion_medida_cantidad"] = {
                "estado": "ambigua_se_conserva_dimension_tipificada",
                "medida": medida,
            }
            return plan
        telemetria["verificacion_medida_cantidad"] = {"estado": "ambigua"}
        texto_norm_med = normalizar(texto_usuario)
        if (
            re.search(r"\ben\s+total\b", texto_norm_med)
            and not re.search(
                r"\b(?:prest\w*|devol\w*|devuelt\w*|afuera|custodia|terceros|usuarios|ambos|ambas)\b",
                texto_norm_med,
            )
            and plan.operacion == "sumar_unidades"
        ):
            plan.tipo = "consulta"
            plan.operacion = "sumar_unidades"
            plan.medida_solicitada = "disponible"
            plan.campos = ["stock"]
            plan.alcance = "actual"
            plan.tipo_movimiento = None
            plan.pregunta = None
            telemetria["verificacion_medida_cantidad"] = {
                "estado": "ambigua_total_simple_default_disponible",
                "medida": "disponible",
            }
            return plan
        aclarar()
        return plan

    if medida_validada in {"pendiente_devolucion", "ambos"} and not _texto_menciona_prestamo_o_custodia(texto_usuario):
        if medida_validada == "ambos":
            medida_validada = "disponible"
        else:
            telemetria["verificacion_medida_cantidad"] = {
                "estado": "descartada_sin_indicio_prestamo",
                "medida_modelo": medida_validada,
            }
            return plan

    telemetria["verificacion_medida_cantidad"] = {
        "estado": "corregida",
        "medida": medida_validada,
    }
    plan.tipo = "consulta"
    plan.operacion = "sumar_unidades"
    plan.medida_solicitada = medida_validada
    plan.campos = sorted(dimensiones_por_medida[medida_validada], key=lambda campo: (campo != "stock", campo))
    plan.alcance = "actual"
    plan.tipo_movimiento = None
    plan.pregunta = None
    return plan


def _completar_atributos_canonicos_solicitados(
    texto: str,
    plan: PlanAgente,
) -> PlanAgente:
    """Recupera atributos que el modelo omitió cuando se nombra su campo del esquema.

    Las etiquetas se derivan de ``CAMPOS_VALIDOS``; no se mantiene un catálogo
    paralelo de expresiones de usuario. La gramática y las paráfrasis continúan
    a cargo del modelo, mientras que una mención literal del campo canónico
    puede reparar una lista de campos vacía o inconsistente.
    """
    if plan.tipo != "consulta":
        return plan

    texto_normalizado = normalizar(preprocesar_texto_inventario(texto or ""))
    etiquetas = {
        campo: normalizar(campo.replace("_", " "))
        for campo in CAMPOS_VALIDOS
    }
    coincidencias = []
    for campo, etiqueta in etiquetas.items():
        if not etiqueta:
            continue
        patron = re.compile(rf"(?<!\w){re.escape(etiqueta)}(?!\w)")
        for coincidencia in patron.finditer(texto_normalizado):
            if campo == "almacen":
                prefijo_alm = texto_normalizado[max(0, coincidencia.start() - 18):coincidencia.start()]
                if re.search(r"\b(?:fuera\s+de(?:l)?|dentro\s+de(?:l)?)\s*(?:el\s+)?$", prefijo_alm):
                    continue
            coincidencias.append((coincidencia.start(), coincidencia.end(), campo))
    # Reconoce menciones del atributo stock_minimo como «el mínimo», «su mínimo»
    # o «mínimo» coordinado con otros atributos, siempre que no forme parte de
    # un filtro relacional (bajo/cerca/en mínimo, mínimo cero) ni de «menor stock».
    for coincidencia_min in re.finditer(
        r"\b(?:(?:el|su)\s+minimo(?:\s+configurado)?|minimo\s+configurado|"
        r"umbral(?:\s+de\s+reposicion|\s+minimo)?(?:\s+configurado)?|"
        r"minimo(?=\s+(?:y|e|o|de|del|tiene|es)\b|\s*$))\b",
        texto_normalizado,
    ):
        prefijo_min = texto_normalizado[max(0, coincidencia_min.start() - 28):coincidencia_min.start()]
        sufijo_min = texto_normalizado[coincidencia_min.end():coincidencia_min.end() + 24]
        if re.search(r"\b(?:bajo|debajo\s+de|debajo\s+del|cerca\s+de|cerca\s+del|proxim\w*\s+a|justo\s+en|exactamente\s+en|inferior\w*\s+a|menor\w*\s+a|menor\w*\s+que|alcanza\w*|llega\w*|supera\w*)\s*$", prefijo_min):
            continue
        if re.search(r"^\s*(?:en\s+|igual\s+a\s+|es\s+)?(?:0|cero)\b", sufijo_min):
            continue
        coincidencias.append((coincidencia_min.start(), coincidencia_min.end(), "stock_minimo"))

    # Reconoce coordinación explícita de «disponible(s)» y «prestada(s) / en préstamo / unidades pendientes»
    tiene_coordinacion_atributos = bool(
        re.search(r"[,;:]|\b(?:y|e|separa|tanto)\b", texto or "", flags=re.IGNORECASE)
    )
    if tiene_coordinacion_atributos:
        for coincidencia_disp in re.finditer(
            r"\b(?:(?:el|su|lo)\s+disponible|unidades?\s+disponibles?|"
            r"disponibles?(?=\s+(?:y|e)\s+(?:prestad[oa]s?|en\s+prestamo|pendientes?))|"
            r"stock\s+actual)\b",
            texto_normalizado,
        ):
            coincidencias.append((coincidencia_disp.start(), coincidencia_disp.end(), "stock"))
        for coincidencia_prest in re.finditer(
            r"\b(?:en\s+prestamos?|unidades?\s+(?:prestad[oa]s?|pendientes?(?:\s+de\s+(?:devolucion|devolver|regresar))?)|"
            r"pendientes?\s+de\s+(?:devolucion|devolver|regresar)|"
            r"(?<=disponibles y )prestad[oa]s?|(?<=disponible y )prestad[oa]s?)\b",
            texto_normalizado,
        ):
            coincidencias.append((coincidencia_prest.start(), coincidencia_prest.end(), "prestamos"))

    # ``stock, mínimo`` (o ``stock y mínimo``) coordina dos atributos; al
    # normalizar puntuación, la búsqueda más larga puede confundirlo con la
    # etiqueta única ``stock mínimo``. Conserva ambos solo cuando la forma
    # original muestra esa coordinación explícita.
    coordinacion_stock_minimo = re.search(
        r"\bstock\s*(?:[,;:]|\by\b)\s*(?:(?:el|su)\s+)?m[ií]nimo\b",
        texto or "",
        flags=re.IGNORECASE,
    )
    if not coincidencias:
        return plan

    # Si una etiqueta contiene a otra (stock_minimo y stock), conserva la más
    # específica. Las etiquetas no solapadas permiten pedir varios atributos.
    campos_encontrados = []
    tramos_usados = []
    for inicio, fin, campo in sorted(
        coincidencias,
        key=lambda elemento: (-(elemento[1] - elemento[0]), elemento[0], elemento[2]),
    ):
        if any(inicio < fin_usado and fin > inicio_usado for inicio_usado, fin_usado in tramos_usados):
            continue
        tramos_usados.append((inicio, fin))
        campos_encontrados.append((inicio, campo))

    if not campos_encontrados:
        return plan

    # Una entidad que ya ocupa el filtro del plan no es también una columna
    # pedida. Los nombres se toman de los filtros resueltos en el catálogo
    # vigente, por lo que esta comprobación se adapta a almacenes, ubicaciones
    # y categorías nuevas sin enumerar sus valores en el código.
    tramos_filtros = {"almacen": [], "ubicacion": [], "categoria": []}
    valores_filtro = {
        "almacen": plan.filtros.almacen,
        "ubicacion": plan.filtros.ubicacion,
        "categoria": plan.filtros.categoria,
    }
    for campo, valor in valores_filtro.items():
        for entidad in str(valor or "").split("|"):
            entidad_norm = normalizar(entidad.strip())
            if not entidad_norm:
                continue
            patron_entidad = re.compile(rf"(?<!\w){re.escape(entidad_norm)}(?!\w)")
            tramos_filtros[campo].extend(
                coincidencia.span()
                for coincidencia in patron_entidad.finditer(texto_normalizado)
            )
    campos_encontrados = [
        (inicio, campo)
        for inicio, campo in campos_encontrados
        if not any(
            inicio >= inicio_filtro
            and inicio + len(etiquetas[campo]) <= fin_filtro
            for inicio_filtro, fin_filtro in tramos_filtros.get(campo, [])
        )
    ]
    # La etiqueta «categoría» introduce a menudo el filtro de catálogo
    # («stock para la categoría X»). No la conviertas también en columna si el
    # plan ya resolvió esa categoría como filtro y la pregunta no pide su valor.
    pregunta_por_valor_categoria = bool(re.search(
        r"\b(?:qu[eé]\s+categor[ií]a|cu[aá]l\s+(?:es\s+(?:su|la)\s+)?categor[ií]a)\b",
        texto_normalizado,
    ))
    if plan.filtros.categoria and not pregunta_por_valor_categoria:
        plan.campos = [campo for campo in plan.campos if campo != "categoria"]
        campos_encontrados = [
            (inicio, campo) for inicio, campo in campos_encontrados
            if campo != "categoria"
        ]
    if not campos_encontrados:
        return plan

    # Un campo usado como predicado no es automáticamente una columna de
    # salida. Se descartan solo coincidencias dentro del tramo relacional; los
    # demás atributos explícitos (p. ej. el almacén solicitado en un listado)
    # se conservan aunque la misma pregunta tenga un filtro de stock.
    tramos_condicion = [
        coincidencia.span()
        for patron in _PATRONES_CONDICION_STOCK.values()
        for coincidencia in re.finditer(patron, texto_normalizado)
    ]
    campos_explicitos = [
        (inicio, campo)
        for inicio, campo in campos_encontrados
        if campo not in {"stock", "stock_minimo"}
        or not any(
            inicio < fin_condicion and inicio + len(normalizar(campo.replace("_", " "))) > inicio_condicion
            for inicio_condicion, fin_condicion in tramos_condicion
        )
    ]
    if coordinacion_stock_minimo and not any(campo == "stock" for _, campo in campos_explicitos):
        campos_explicitos.append((-1, "stock"))
    if not campos_explicitos:
        return plan

    medida = (plan.medida_solicitada or "").strip().casefold()
    medida_campo = {"disponible": "stock"}.get(medida)
    pide_cantidad = bool(re.search(
        r"\b(?:cu[aá]nt[oa]s?|unidad(?:es)?|cantidad)\b",
        texto_normalizado,
    ))
    if (
        medida_campo
        and medida_campo in (plan.campos or [])
        and medida_campo not in {campo for _, campo in campos_explicitos}
        and not pide_cantidad
    ):
        plan.campos = [campo for campo in plan.campos if campo != medida_campo]

    stock_explicito = bool(re.search(r"\bstock\b(?!\s+minimo\b)", texto_normalizado))
    minimo_explicito = "stock_minimo" in {campo for _, campo in campos_explicitos}
    if (
        "stock" in (plan.campos or [])
        and minimo_explicito
        and not stock_explicito
        and not pide_cantidad
    ):
        # Una solicitud del mínimo no implica pedir también las existencias.
        # Las cantidades explícitas conservan ambos campos en consultas compuestas.
        plan.campos = [campo for campo in plan.campos if campo != "stock"]

    plan.campos = list(dict.fromkeys([
        *(campo for campo in (plan.campos or []) if campo in CAMPOS_VALIDOS),
        *(campo for _, campo in sorted(campos_explicitos)),
    ]))
    campos_puntuales_explicitos = {campo for _, campo in campos_explicitos if campo != "ficha"}
    if (
        "ficha" in plan.campos
        and campos_puntuales_explicitos
        and re.search(r"\b(?:solo|solamente|unicamente)\b", texto_normalizado)
    ):
        plan.campos = [campo for campo in plan.campos if campo != "ficha"]

    # Una etiqueta de dimensión incluida en una suma cuantificada es el
    # resultado solicitado, no una ficha de atributos. Conserva el agregado
    # cuando el usuario pidió solo una o ambas cantidades canónicas.
    dimensiones_solicitadas = set(plan.campos) & {"stock", "prestamos"}
    atributos_adicionales = set(plan.campos) - {"stock", "prestamos"}
    if (
        plan.tipo == "consulta"
        and plan.operacion == "sumar_unidades"
        and pide_cantidad
        and not _plan_tiene_referencia_individual_sin_codigo(plan)
        and not atributos_adicionales
        and dimensiones_solicitadas
    ):
        if dimensiones_solicitadas == {"stock", "prestamos"}:
            plan.medida_solicitada = "ambos"
        elif plan.medida_solicitada in {None, "atributo", "ninguna"}:
            if dimensiones_solicitadas == {"stock"}:
                plan.medida_solicitada = "disponible"
            elif dimensiones_solicitadas == {"prestamos"}:
                plan.medida_solicitada = "pendiente_devolucion"
        return plan

    plan.medida_solicitada = "atributo"
    entidades_individuales = bool(
        plan.filtros.codigos
        or plan.filtros.texto
        or plan.producto
        or plan.referencia == "foco"
        or plan.alcance_entidad == "producto"
    )
    if entidades_individuales:
        plan.operacion = "detalle"
        plan.alcance_entidad = "producto"
        plan.alcance = "actual"
        plan.tipo_movimiento = None
    return plan


def _es_pregunta_capacidades_agente(texto: str) -> bool:
    """Reconoce preguntas sobre funciones por su estructura, no por el catálogo.

    Se limita a preguntas sobre el agente o las clases de información que puede
    consultar. Una referencia de producto expresada como código mantiene la
    petición en el flujo de inventario.
    """
    original = texto or ""
    texto_norm = normalizar(preprocesar_texto_inventario(original))
    if not texto_norm or re.search(r"\b[a-z]{2,6}[- ]?\d{1,6}\b", original, re.IGNORECASE):
        return False

    es_pregunta = bool(re.search(
        r"\b(?:que|cual|cuales|como|indique|indiqueme|explique|expliqueme|dime)\b",
        texto_norm,
    ))
    menciona_agente = bool(re.search(
        r"\b(?:hacu|haku|agente|asistente|tus|tu|sus|usted)\b", texto_norm,
    ))
    menciona_capacidad = bool(re.search(
        r"\b(?:funcion\w*|capacidad\w*|alcance|consultas?|informacion)\b",
        texto_norm,
    ))
    describe_poder_hacer = bool(re.search(
        r"\b(?:puede|puedes|pueden|podria|podrias|sabe|sabes)\b"
        r".{0,80}\b[a-zñ]+(?:ar|er|ir)\b",
        texto_norm,
    ))
    pregunta_de_alcance = bool(re.search(
        r"\b(?:tipo[s]?\s+de\s+consulta|alcance\s+de\s+(?:las?\s+)?funcion)\w*\b",
        texto_norm,
    ))

    if not es_pregunta:
        return False
    if pregunta_de_alcance:
        return True
    if menciona_agente and (menciona_capacidad or describe_poder_hacer):
        return True
    # En una pregunta dirigida al asistente, consultar información del
    # inventario describe una capacidad; el código de producto ya se descartó.
    return bool(
        menciona_capacidad
        and describe_poder_hacer
        and re.search(r"\b(?:inventario|stock|productos?|referencias?)\b", texto_norm)
    )


def _normalizar_detalle_de_coleccion(texto: str, plan: PlanAgente) -> PlanAgente:
    """Evita pedir un producto único cuando la morfología ya define una colección."""
    if (
        plan.tipo != "consulta"
        or plan.operacion != "detalle"
        or plan.alcance == "evento"
        or plan.filtros.codigos
        or not plan.filtros.texto
        or plan.referencia in {"foco", "resultados_previos"}
    ):
        return plan
    plan = _normalizar_alcance_entidad_por_morfologia(texto, plan)
    if plan.alcance_entidad != "coleccion":
        return plan

    plan.operacion = "listar"
    plan.referencia = "ninguna"
    plan.alcance = "actual"
    plan.tipo_movimiento = None
    plan.pregunta = None
    if plan.medida_solicitada == "disponible" and "stock" not in (plan.campos or []):
        plan.campos = list(dict.fromkeys([*(plan.campos or []), "stock"]))
    return plan


def _normalizar_intencion_extremal(texto: str, plan: PlanAgente) -> PlanAgente:
    """Conserva una selección de máximo/mínimo aunque un paso previo la detalle."""
    if plan.tipo != "consulta":
        return plan
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    metrica = bool(re.search(r"\b(?:stock|existencias?|unidades?|cantidad)\b", texto_norm))
    if not metrica:
        return plan
    patrones = (
        ("mayor_stock", r"\b(?:cual|quien)\b.{0,100}\b(?:tiene|tenemos|manejamos|hay|registra|mantiene|dispone)\b.{0,60}\b(?:mas|mayor)\b"),
        ("menor_stock", r"\b(?:cual|quien)\b.{0,100}\b(?:tiene|tenemos|manejamos|hay|registra|mantiene|dispone)\b.{0,60}\b(?:menos|menor)\b"),
        ("mayor_stock", r"\b(?:cual|que\s+producto|el\s+producto)\b.{0,60}\bcon\s+(?:mas|mayor)\s+(?:stock|existencias?|unidades?|cantidad)\b"),
        ("menor_stock", r"\b(?:cual|que\s+producto|el\s+producto)\b.{0,60}\bcon\s+(?:menos|menor)\s+(?:stock|existencias?|unidades?|cantidad)\b"),
    )
    for operacion, patron in patrones:
        if re.search(patron, texto_norm):
            plan.operacion = operacion
            if any((
                plan.filtros.texto,
                plan.filtros.codigos,
                plan.filtros.categoria,
                plan.filtros.almacen,
                plan.filtros.ubicacion,
                plan.filtros.condicion,
            )):
                plan.referencia = "ninguna"
                plan.alcance_entidad = "coleccion"
            elif re.search(r"\b(?:de esos|de ellos|de ellas|entre esos|entre ellos|entre ellas)\b", texto_norm):
                plan.referencia = "resultados_previos"
            else:
                plan.referencia = "ninguna"
                plan.alcance_entidad = "global"
                plan.alcance_global_confirmado = True
            return plan
    return plan


def _recortar_campos_sobregenerados_de_coleccion_stock(texto: str, plan: PlanAgente) -> PlanAgente:
    """Descarta columnas de catálogo que el modelo añade sin pedirlas.

    Se aplica solo a una colección filtrada por estado de stock. Los nombres
    de campo salen del contrato vigente y se conservan cuando la petición los
    respalda; así, cambios en productos o almacenes no afectan la regla.
    """
    if (
        plan.tipo != "consulta"
        or plan.operacion not in {"listar", "contar", "sumar_unidades"}
        or plan.alcance_entidad not in {"coleccion", "global"}
        or (
            plan.filtros.condicion not in _CONDICIONES_STOCK_PARA_CLASIFICAR
            and not (plan.alcance_entidad == "global" and plan.filtros.condicion is None)
        )
    ):
        return plan

    texto_normalizado = normalizar(preprocesar_texto_inventario(texto or ""))
    tokens_mensaje = set(tokens_significativos(texto_normalizado))
    solicita_detalle_completo = bool(
        re.search(r"\b(?:todo|toda|todos|todas|complet[oa]s?)\b", texto_normalizado)
        and re.search(r"\b(?:datos?|detalles?|informaci[oó]n|atributos?|campos?|ficha)\b", texto_normalizado)
    )
    if solicita_detalle_completo:
        plan.campos = ["ficha"]
        return plan

    if plan.medida_solicitada == "atributo" and len(plan.campos or []) <= 3:
        # Para una petición explícita de atributos y una selección acotada,
        # conserva la interpretación semántica del modelo.
        return plan

    atributos_explicitos = set()
    for campo in plan.campos:
        tokens_campo = set(tokens_significativos(campo.replace("_", " ")))
        if tokens_campo and tokens_campo.issubset(tokens_mensaje):
            atributos_explicitos.add(campo)

    # Conserva las formas comunes de ubicación si el modelo ya identificó el
    # campo, aunque la persona no haya usado su etiqueta de base de datos.
    if "ubicacion" in plan.campos and re.search(
        r"\b(?:donde|ubicaci[oó]n|estante|estanter[ií]a|guardad[oa]s?|situad[oa]s?)\b",
        texto_normalizado,
    ):
        atributos_explicitos.add("ubicacion")
    if "almacen" in plan.campos and re.search(r"\b(?:almacen|bodega)\b", texto_normalizado):
        atributos_explicitos.add("almacen")
    if "prestamos" in plan.campos and plan.medida_solicitada == "pendiente_devolucion":
        atributos_explicitos.add("prestamos")
    if "movimientos" in plan.campos and (
        plan.medida_solicitada == "evento" or plan.fecha_desde or plan.fecha_hasta
    ):
        atributos_explicitos.add("movimientos")
    if "stock" in plan.campos and plan.alcance == "evento":
        atributos_explicitos.add("stock")
    if plan.medida_solicitada == "ambos":
        atributos_explicitos.update({"stock", "prestamos"})
    if "ficha" in atributos_explicitos:
        # Una ficha completa ya incluye los demás campos y no se convierte en
        # una lista interminable de columnas repetidas.
        plan.campos = ["ficha"]
    else:
        plan.campos = [campo for campo in plan.campos if campo in atributos_explicitos]
    return plan


def _normalizar_intencion_evento_temporal(
    texto: str,
    plan: PlanAgente,
    foco_disponible: bool = False,
) -> PlanAgente:
    """Separa consultas sobre movimientos pasados de órdenes para crear operaciones."""
    operacion_modelo = plan.tipo == "operacion"
    if not operacion_modelo and plan.tipo != "consulta":
        return plan

    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    pide_cantidad = bool(re.search(r"\b(?:cuant\w*|cantidad(?:\s+de)?|n[uú]mero\s+de)\b", texto_norm))
    marcador_primero = bool(
        re.search(
            r"\b(?:primer[oa]?|primera\s+vez|inicial(?:mente)?|original(?:mente)?|"
            r"al\s+principio|al\s+inicio|al\s+comienzo|comenz\w*|empez\w*|inici\w*|cuando\s+se\s+cre[oó])\b",
            texto_norm,
        )
    )
    marcador_ultimo = bool(re.search(r"\b(?:ultim[oa]?|mas\s+reciente)\b", texto_norm))
    describe_transferencia_pasada = bool(
        re.search(
            r"\b(?:se|le|les|me|te|nos)\s+(?:(?:me|te|le|les|nos)\s+)?"
            r"(?:(?:ha|han|habia|habian|fue|fueron)\s+)?"
            r"(?:pas\w*|entreg\w*|prest\w*|ced\w*|asign\w*|transfir\w*|registr\w*|"
            r"recib\w*|devolv\w*|retir\w*|sac\w*|ingres\w*|mov\w*|dieron|dio)\b",
            texto_norm,
        )
        or re.search(
            r"\b(?:sali(?:o|eron)|entreg(?:o|aron)|prest(?:o|aron)|comenz[oó]|empez[oó]|inici[oó]|cre[oó])\b.{0,40}\bprestamos?\b",
            texto_norm,
        )
        or re.search(
            r"\b(?:prestamos?\s+(?:original|inicial)|cantidad\s+(?:original|inicial)\s+del?\s+prestamo|evento\s+de\s+prestamo)\b",
            texto_norm,
        )
    )
    explicito_evento = plan.alcance == "evento" or "movimientos" in (plan.campos or [])
    transferencia_cuantificada = (pide_cantidad and describe_transferencia_pasada) or bool(
        re.search(
            r"\b(?:cantidad\s+(?:original|inicial)\s+del?\s+prestamo|prestamos?\s+(?:original|inicial)|evento\s+de\s+prestamo)\b",
            texto_norm,
        )
    )
    if not (transferencia_cuantificada or (explicito_evento and (marcador_primero or marcador_ultimo))):
        return plan
    texto_sin_saldo_excluido = re.sub(
        r"\b(?:sin\s+(?:usar|contar)\s+(?:el\s+)?|no\s+(?:el\s+)?)(?:saldo\s+)?(?:pendiente|actual)(?:\s+actual)?\b",
        " ",
        texto_norm,
    )
    expresa_saldo_actual = bool(
        re.search(r"\b(?:actual(?:mente)?|disponibles?|quedan?|siguen?|todav[ií]a|a[uú]n|pendientes?)\b", texto_sin_saldo_excluido)
    )
    try:
        from ..servicios_temporales import resolver_intervalo_temporal
        tiene_periodo_temporal = resolver_intervalo_temporal(texto or "") is not None
    except Exception:
        tiene_periodo_temporal = False
    if (
        transferencia_cuantificada
        and re.search(r"\bse\s+prest(?:o|aron)\b", texto_norm)
        and not marcador_primero
        and not marcador_ultimo
        and not expresa_saldo_actual
        and not tiene_periodo_temporal
        and not plan.fecha_desde
        and not plan.fecha_hasta
        and not plan.destinatario
    ):
        plan.tipo = "aclaracion"
        plan.operacion = None
        plan.campos = []
        plan.alcance = None
        plan.tipo_movimiento = None
        plan.medida_solicitada = "ninguna"
        plan.pregunta = (
            "¿Desea consultar la cantidad prestada originalmente en el movimiento "
            "o las unidades que siguen pendientes de devolución actualmente?"
        )
        return plan
    evento_ordinal_cuantificado = bool(
        transferencia_cuantificada and (marcador_primero or marcador_ultimo)
    )
    consulta_compuesta = evento_ordinal_cuantificado and expresa_saldo_actual
    if evento_ordinal_cuantificado:
        # Una cantidad asociada a un evento histórico ordinal es una consulta,
        # aunque el modelo la haya clasificado como una nueva mutación. Si pide
        # además el saldo vigente, se leen ambos datos; «compare» tampoco debe
        # anclar la entidad a una lista anterior. Catálogo y permisos siguen
        # resolviéndose contra la base de datos actual.
        codigos_explicitos = extraer_codigos_de_texto(texto)
        if codigos_explicitos:
            plan.filtros.codigos = list(dict.fromkeys(codigos_explicitos))
            plan.filtros.texto = None
            plan.referencia = "ninguna"
        elif plan.producto and not plan.filtros.texto:
            plan.filtros.texto = plan.producto

        tiene_objeto_unico = bool(
            len(plan.filtros.codigos) == 1
            or plan.filtros.texto
            or plan.referencia == "foco"
        )
        plan.filtros.condicion = None
        plan.accion = None
        plan.cantidad = None
        plan.producto = None
        plan.destino = None
        plan.valor = None
        if not tiene_objeto_unico and foco_disponible and not _es_consulta_global_inventario(texto):
            plan.referencia = "foco"
            tiene_objeto_unico = True
        if not tiene_objeto_unico:
            plan.tipo = "aclaracion"
            plan.operacion = None
            plan.pregunta = "¿De qué producto desea comparar el historial con el saldo actual?"
            plan.alcance = None
            plan.tipo_movimiento = None
            plan.campos = []
            return plan

        plan.tipo = "consulta"
        plan.operacion = "detalle"
        plan.pregunta = None
        plan.alcance = "ambos" if consulta_compuesta else "evento"
        plan.medida_solicitada = "ambos" if consulta_compuesta else "evento"
        tipo_mencionado = (
            plan.tipo_movimiento
            if _tipo_movimiento_mencionado(texto, plan.tipo_movimiento)
            else None
        )
        plan.tipo_movimiento = _inferir_tipo_movimiento_desde_etiqueta(
            texto, "evento", tipo_mencionado,
        )
        plan.campos = ["movimientos", "prestamos"] if consulta_compuesta else ["movimientos"]
        plan.orden_movimiento = "asc" if marcador_primero else "desc"
        if not consulta_compuesta and (marcador_primero or marcador_ultimo):
            corte = re.search(
                r"\b(?:primer[oa]?|primera\s+vez|inicialmente|originalmente|"
                r"ultim[oa]?|mas\s+reciente)\b",
                texto_norm,
            )
            prefijo = texto_norm[:corte.start()] if corte else ""
            match_destino = re.search(
                r"\b(?:a|al|para)\s+([a-z0-9áéíóúñü][a-z0-9áéíóúñü\s'.-]*)$",
                prefijo,
            )
            if match_destino and not plan.destinatario:
                destino = re.sub(r"\s+(?:por|con|en)\s*$", "", match_destino.group(1)).strip(" .,!?:;")
                destino = re.sub(r"\s+(?:el|la|los|las)$", "", destino)
                destino = re.sub(r"^(?:el|la|los|las)\s+", "", destino)
                if destino:
                    plan.destinatario = destino
        return plan

    if operacion_modelo:
        return plan
    if transferencia_cuantificada and expresa_saldo_actual:
        return plan

    if transferencia_cuantificada:
        plan.alcance = "evento"
        plan.medida_solicitada = "evento"
        plan.campos = [campo for campo in (plan.campos or []) if campo not in {"stock", "prestamos"}]
        if "movimientos" not in plan.campos:
            plan.campos.append("movimientos")
    if marcador_primero:
        plan.orden_movimiento = "asc"
    elif marcador_ultimo:
        plan.orden_movimiento = "desc"

    if transferencia_cuantificada:
        plan.operacion = "detalle"
        tipo_mencionado = (
            plan.tipo_movimiento
            if _tipo_movimiento_mencionado(texto, plan.tipo_movimiento)
            else None
        )
        plan.tipo_movimiento = _inferir_tipo_movimiento_desde_etiqueta(
            texto, "evento", tipo_mencionado,
        )
        # Cuando el destinatario aparece antes del ordinal temporal, se conserva
        # solo la frase que el usuario dijo; la herramienta la coteja con los
        # destinatarios reales del producto y del almacén accesible.
        if marcador_primero or marcador_ultimo:
            corte = re.search(
                r"\b(?:primer[oa]?|primera\s+vez|inicialmente|originalmente|"
                r"ultim[oa]?|mas\s+reciente)\b",
                texto_norm,
            )
            prefijo = texto_norm[:corte.start()] if corte else ""
            match_destino = re.search(r"\b(?:a|al|para)\s+([a-z0-9áéíóúñü][a-z0-9áéíóúñü\s'.-]*)$", prefijo)
            if match_destino:
                destino = re.sub(r"\s+(?:por|con|en)\s*$", "", match_destino.group(1)).strip(" .,!?:;")
                destino = re.sub(r"\s+(?:el|la|los|las)$", "", destino)
                destino = re.sub(r"^(?:el|la|los|las)\s+", "", destino)
                if destino:
                    plan.destinatario = destino
    return plan


def _pregunta_pide_conjunto(texto: str) -> bool:
    """Reconoce una pregunta por los elementos de un conjunto, no por su cardinalidad."""
    return bool(re.search(
        r"\b(?:que|cuales)\b.*\b(?:tenemos|hay|quedan|existen)\b",
        normalizar(texto or ""),
    ))


def _normalizar_aclaracion_de_conjunto(
    texto: str,
    plan: PlanAgente,
    categorias_accesibles: Optional[List[str]] = None,
    ubicaciones_accesibles: Optional[List[str]] = None,
    almacenes_accesibles: Optional[List[str]] = None,
) -> PlanAgente:
    """Convierte una duda/detalle espurio en búsqueda si la pregunta pide un conjunto."""
    if plan.tipo not in {"aclaracion", "consulta"} or plan.referencia in {"foco", "resultados_previos"}:
        return plan
    ubicaciones_mencionadas = _resolver_menciones_ubicacion(
        texto or "", _nombres_ubicaciones_registradas(),
    )
    if plan.tipo == "aclaracion" and len(ubicaciones_mencionadas) >= 2:
        # No reescribas como búsqueda una ambigüedad de varias entidades físicas.
        return plan
    if plan.tipo == "consulta" and plan.operacion != "detalle":
        return plan
    pregunta_pide_conjunto = _pregunta_pide_conjunto(texto)
    if plan.filtros.codigos or not pregunta_pide_conjunto:
        return plan
    if plan.filtros.es_prestable is not None or plan.filtros.condicion is not None:
        plan.tipo = "consulta"
        plan.operacion = "listar"
        plan.referencia = "ninguna"
        plan.pregunta = None
        plan.campos = [c for c in (plan.campos or []) if c != "es_prestable"]
        return plan

    # Si el modelo ya aisló el objeto, consérvelo. Reextraerlo del mensaje
    # completo puede reintroducir cortesía, métricas o el nombre del almacén
    # dentro del filtro de producto (p. ej. "disponibilidad de cajas de grapas").
    objeto = (plan.filtros.texto or "").strip() or _extraer_objeto_consulta_sin_marco(texto)
    ubicacion_explicita = (
        ubicaciones_mencionadas[0]
        if plan.tipo == "aclaracion" and len(ubicaciones_mencionadas) == 1
        else None
    )
    if not objeto and not ubicacion_explicita:
        return plan
    categoria = next(
        (cat for cat in categorias_accesibles or [] if normalizar(cat) == normalizar(objeto)),
        None,
    ) if objeto else None
    plan.tipo = "consulta"
    plan.operacion = "listar"
    plan.referencia = "ninguna"
    plan.pregunta = None
    plan.campos = []
    plan.filtros.codigos = []
    if objeto:
        plan.filtros.categoria = categoria
        plan.filtros.categoria_explicita = bool(categoria)
        plan.filtros.texto = None if categoria else objeto
    elif not plan.filtros.categoria_explicita:
        plan.filtros.categoria = None
        plan.filtros.texto = None
    if ubicacion_explicita:
        plan.filtros.ubicacion = ubicacion_explicita
        plan = _completar_ubicacion_explicita(
            plan,
            texto,
            ubicaciones_accesibles or [],
            almacenes_accesibles or [],
            categorias_accesibles or [],
        )
    return plan


def _recuperar_consulta_catalogo_almacen(
    texto: str,
    plan: PlanAgente,
) -> PlanAgente:
    """No confunde con fuera de dominio una petición explícita de catálogo por almacén.

    La entidad se resuelve con nombres actuales de la base. La herramienta de
    lectura vuelve a comprobar el permiso del usuario antes de exponer datos.
    """
    if plan.tipo != "fuera_dominio":
        return plan

    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    menciona_catalogo = bool(re.search(r"\b(?:catalogo|inventario)\b", texto_norm))
    menciona_almacen = bool(re.search(r"\b(?:almacen|bodega)\b", texto_norm))
    if not (menciona_catalogo and menciona_almacen):
        return plan
    if re.search(r"\b(?:no|nunca|jamas)\b.{0,32}\b(?:catalogo|inventario)\b", texto_norm):
        return plan

    nombres_actuales = _nombres_almacenes_registrados()
    coincidencias = _menciones_catalogo(texto, nombres_actuales)
    if not coincidencias:
        plan.tipo = "aclaracion"
        plan.operacion = None
        plan.pregunta = "No pude identificar ese almacén en el catálogo actual. ¿Podría verificar el nombre?"
        plan.filtros = FiltrosConsulta()
        plan.campos = []
        return plan

    plan.tipo = "consulta"
    plan.operacion = "listar"
    plan.filtros = FiltrosConsulta(almacen="|".join(coincidencias))
    plan.referencia = "ninguna"
    plan.alcance = "actual"
    plan.tipo_movimiento = None
    plan.campos = []
    plan.pregunta = None
    return plan


def _nombres_almacenes_registrados() -> List[str]:
    """Lee etiquetas vivas de almacén para normalizar menciones explícitas."""
    try:
        from inventario.models import Almacen
        return list(
            Almacen.objects.filter(eliminado=False).order_by("nombre").values_list("nombre", flat=True)
        )
    except Exception:
        return []


def _completar_almacen_consulta_explicito(texto: str, plan: PlanAgente) -> PlanAgente:
    """Conserva almacenes nombrados aunque el LLM omita el filtro o lo sustituya."""
    if plan.tipo != "consulta" or not re.search(r"\b(?:almacen|bodega)\b", normalizar(texto or "")):
        return plan
    coincidencias = _menciones_almacenes_catalogo(texto, _nombres_almacenes_registrados())
    if coincidencias:
        plan.filtros.almacen = "|".join(coincidencias)
    return plan


def _validar_referencia_foco_con_contexto(
    plan: PlanAgente,
    contexto_txt: str,
) -> PlanAgente:
    """No permite que una etiqueta generada por el modelo invente un foco."""
    hay_producto_en_foco = bool(
        re.search(r"(?m)^Producto en foco:\s*\S", contexto_txt or "")
    )
    if plan.referencia == "foco" and not hay_producto_en_foco:
        plan.referencia = "ninguna"
    return plan


def _aclarar_consulta_sin_entidad(
    texto: str,
    plan: PlanAgente,
    contexto_txt: str,
) -> PlanAgente:
    """Evita convertir una referencia sin objeto/foco en un listado global."""
    if plan.tipo != "consulta" or plan.operacion not in {
        "detalle", "listar", "contar", "sumar_unidades", "explicar_estado",
    }:
        return plan
    if (
        plan.filtros.condicion in _PATRONES_CONDICION_STOCK
        and plan.filtros.condicion not in _condiciones_stock_mencionadas(texto)
        and not _condicion_validada_propuesta_por_modelo(plan)
    ):
        # No conviertas una palabra elíptica como «queda» en un filtro global
        # de agotados si el usuario no solicitó esa condición.
        plan.filtros.condicion = None
    if plan.filtros.texto and not _extraer_objeto_consulta_sin_marco(plan.filtros.texto):
        # El modelo a veces entrega un pronombre como si fuera un nombre de
        # producto. No lo consultes literalmente: al no contener un objeto,
        # deja que el contexto resuelva el referente único o solicita precisión.
        plan.filtros.texto = None
    if not plan.filtros.vacio() or plan.referencia == "resultados_previos":
        return plan
    texto_norm_acl = normalizar(texto or "")
    es_atributo_categoria_singular = bool(
        re.search(r"\b(?:su|sus|ese|esa|este|esta|dicho|mismo|misma|tiene|pertenece)\b", texto_norm_acl)
        or re.search(r"\b(?:que\s+categoria|cual\s+(?:es\s+(?:su|la)\s+)?categoria)\b", texto_norm_acl)
    )
    if (
        plan.operacion in {"listar", "contar"}
        and plan.campos == ["categoria"]
        and not es_atributo_categoria_singular
    ):
        plan.alcance_entidad = "global"
        plan.alcance_global_confirmado = True
        return plan
    if (
        plan.alcance == "evento"
        and "movimientos" in (plan.campos or [])
        and (plan.fecha_desde or plan.fecha_hasta or plan.alcance_entidad == "global")
    ):
        plan.alcance_entidad = "global"
        return plan
    if plan.alcance_global_confirmado:
        # La clasificación semántica independiente confirmó que se pidió el
        # catálogo completo. No vuelvas a rechazarlo por no coincidir con una
        # plantilla de palabras de alcance.
        plan.alcance_entidad = "global"
        return plan
    # Una operación sobre el catálogo completo requiere una indicación de
    # alcance explícita. El plan del LLM por sí solo no convierte «¿qué queda?»
    # en una consulta global.
    if _es_consulta_global_inventario(texto):
        plan.alcance_entidad = "global"
        return plan

    hay_foco = bool(re.search(r"(?m)^Producto en foco:\s*\S", contexto_txt or ""))
    if plan.referencia == "foco" and not hay_foco:
        # Deja que el gestor conversacional de referencias pida el producto
        # faltante; una aclaración genérica aquí pierde el motivo del seguimiento.
        return plan
    tokens_filtro = set(tokens_significativos((plan.filtros.texto or "").replace("_", " ")))
    tokens_atributos = {
        token
        for campo in (plan.campos or [])
        for token in tokens_significativos(str(campo).replace("_", " "))
    }
    if (
        hay_foco
        and tokens_filtro
        and tokens_atributos
        and tokens_filtro.issubset(tokens_atributos)
    ):
        # El plan a veces duplica el atributo pedido dentro de filtros.texto
        # (p. ej. texto="stock mínimo", campos=["stock_minimo"]). El propio
        # esquema identifica ese texto como atributo, no como otro producto.
        plan.filtros.texto = None

    if (
        hay_foco
        and plan.filtros.vacio()
        and (plan.campos or plan.medida_solicitada)
    ):
        # Una medida/atributo sin entidad nueva es un seguimiento del producto
        # en foco. La decisión usa el plan semántico y el contexto, no una frase
        # fija ni el nombre de un producto.
        plan.referencia = "foco"
        if plan.alcance_entidad in {None, "indefinido"}:
            plan.alcance_entidad = "producto"
        return plan

    if _extraer_objeto_consulta_sin_marco(texto):
        return plan
    if hay_foco:
        plan.referencia = "foco"
        return plan
    plan.tipo = "aclaracion"
    plan.operacion = None
    plan.referencia = "ninguna"
    plan.pregunta = "¿A qué producto se refiere?"
    plan.campos = []
    return plan


def aclarar_agregado_sin_alcance_explicito(texto: str, plan: PlanAgente) -> PlanAgente:
    """Impide que una agregación sin filtros se amplíe implícitamente a todo el catálogo.

    El planificador puede detectar correctamente una operación de suma o conteo,
    pero omitir el producto nombrado. El alcance `coleccion` por sí solo no es
    una autorización para agregar todo: debe haber filtros, referencia previa o
    una petición global explícita en el mensaje original.
    """
    if (
        plan.tipo != "consulta"
        or plan.operacion not in {"listar", "contar", "sumar_unidades"}
        or plan.referencia != "ninguna"
        or plan.ubicaciones_comparadas
    ):
        return plan

    filtros = plan.filtros
    tiene_filtros = any((
        filtros.texto,
        filtros.codigos,
        filtros.categoria,
        filtros.almacen,
        filtros.ubicacion,
        filtros.condicion,
        filtros.umbral,
        filtros.categoria_explicita,
        filtros.categoria_generica,
        filtros.es_prestable is not None,
    ))
    if tiene_filtros:
        return plan

    if (
        plan.alcance == "evento"
        and "movimientos" in (plan.campos or [])
        and (plan.fecha_desde or plan.fecha_hasta or plan.tipo_movimiento)
    ):
        plan.alcance_entidad = "global"
        return plan

    if plan.alcance_global_confirmado:
        plan.alcance_entidad = "global"
        return plan

    if _es_consulta_global_inventario(texto):
        plan.alcance_entidad = "global"
        return plan

    producto = str(plan.producto or "").strip()
    if _producto_es_entidad_concreta(producto):
        tokens_mensaje = set(tokens_significativos(texto))
        tokens_producto = set(tokens_significativos(producto))
        codigos_mensaje = set(extraer_codigos_de_texto(texto))
        codigos_producto = set(extraer_codigos_de_texto(producto))
        producto_respaldado_por_mensaje = bool(
            (tokens_producto and tokens_producto.issubset(tokens_mensaje))
            or (codigos_producto and codigos_producto.issubset(codigos_mensaje))
        )
        if producto_respaldado_por_mensaje:
            # El modelo ya extrajo un referente del mensaje, pero dejó vacío
            # el filtro que ejecuta la consulta. Conservamos ese referente
            # para resolverlo dinámicamente contra el catálogo, aunque aún no
            # exista una coincidencia; nunca se convierte en una suma global.
            filtros.texto = producto
            plan.alcance_entidad = "producto"
            return plan

    plan.tipo = "aclaracion"
    plan.operacion = None
    plan.filtros = FiltrosConsulta()
    plan.campos = []
    plan.medida_solicitada = None
    plan.alcance_entidad = None
    plan.pregunta = (
        "No pude identificar el producto o alcance de esa consulta. "
        "¿Puede indicar el código o nombre del producto, o confirmar que desea el total de todo el inventario?"
    )
    plan.fuente_plan = "aclaracion_alcance_agregado"
    return plan


def _normalizar_alcance_segun_campos(plan: PlanAgente, texto: str) -> PlanAgente:
    """Mantiene coherentes el alcance temporal y los campos del plan validado."""
    if plan.tipo != "consulta":
        return plan

    texto_norm = normalizar(texto or "")
    campos = set(plan.campos or [])
    if plan.operacion == "sumar_unidades" and plan.alcance not in {"actual", "ambos"}:
        tipo_evento = _inferir_tipo_movimiento_desde_etiqueta(texto, "evento", plan.tipo_movimiento)
        if tipo_evento:
            plan.operacion = "detalle"
            plan.alcance = "evento"
            plan.tipo_movimiento = tipo_evento
            plan.campos = [campo for campo in plan.campos if campo not in {"stock", "prestamos"}]
            if "movimientos" not in plan.campos:
                plan.campos.append("movimientos")
            return plan

    menciona_movimientos = bool(re.search(r"\bmovimientos?\b", texto_norm))
    tipo_mencionado = _tipo_movimiento_mencionado(texto, plan.tipo_movimiento)
    alcance_eventos_explicito = plan.alcance == "evento"
    pide_ficha = "ficha" in campos

    # No amplíes una consulta de stock a eventos solo porque el modelo agregó
    # campos de historial sin que la solicitud los mencione.
    if not (menciona_movimientos or alcance_eventos_explicito or pide_ficha or tipo_mencionado):
        plan.campos = [campo for campo in plan.campos if campo != "movimientos"]
        plan.tipo_movimiento = None
        if plan.alcance == "ambos":
            plan.alcance = "actual" if campos & {"stock", "prestamos"} else None
        return plan

    solicita_saldo_actual = bool(
        plan.medida_solicitada in {"disponible", "pendiente_devolucion", "ambos"}
        or re.search(
            r"\b(?:stock|existencias?|disponibles?|quedan?|cantidad|unidades?)\b",
            texto_norm,
        )
    )
    solicita_ficha = bool(re.search(r"\bficha\b", texto_norm))
    if menciona_movimientos and not solicita_saldo_actual and not solicita_ficha:
        # Campos añadidos por el modelo no amplían una consulta de historial:
        # el usuario debe pedir explícitamente el saldo actual para mezclarlo.
        plan.campos = [campo for campo in plan.campos if campo not in {"stock", "prestamos", "ficha"}]
        campos = set(plan.campos or [])
        plan.alcance = "evento"
        if (
            plan.operacion == "listar"
            and plan.filtros.texto
            and not plan.filtros.categoria
            and not plan.filtros.condicion
            and not _es_consulta_global_inventario(texto)
        ):
            tokens_obj = [t for t in normalizar(plan.filtros.texto).split() if t not in _PALABRAS_VACIAS]
            if tokens_obj and _raiz(tokens_obj[0]) == tokens_obj[0]:
                plan.operacion = "detalle"
                plan.alcance_entidad = "producto"
                if "movimientos" not in plan.campos:
                    plan.campos.append("movimientos")

    consulta_saldo = bool(
        campos & {"stock", "prestamos", "ficha"}
        and (solicita_saldo_actual or solicita_ficha)
    )
    if plan.alcance == "evento":
        pass
    elif plan.alcance != "ambos":
        plan.alcance = "ambos" if consulta_saldo else "evento"
    if "movimientos" not in campos and "ficha" not in campos:
        plan.campos.append("movimientos")
    plan.tipo_movimiento = _inferir_tipo_movimiento_desde_etiqueta(
        texto, plan.alcance, plan.tipo_movimiento,
    )
    if plan.alcance == "evento" and plan.operacion == "sumar_unidades":
        plan.operacion = "detalle"
        plan.campos = [campo for campo in plan.campos if campo not in {"stock", "prestamos"}]
        if "movimientos" not in plan.campos:
            plan.campos.append("movimientos")
    consulta_eventos_de_producto = (
        plan.alcance == "evento"
        and plan.operacion == "listar"
        and (
            bool(plan.filtros.codigos)
            or bool((plan.filtros.texto or "").strip() and not plan.filtros.categoria)
            or plan.referencia == "foco"
        )
    )
    if consulta_eventos_de_producto:
        plan.operacion = "detalle"
        plan.campos = [campo for campo in plan.campos if campo not in {"stock", "prestamos"}]
        if "movimientos" not in plan.campos:
            plan.campos.append("movimientos")
    return plan


def _normalizar_filtro_evento_desde_mensaje(texto: str, plan: PlanAgente) -> PlanAgente:
    """Retira del filtro las palabras del marco temporal y conserva la entidad citada.

    La extracción trabaja sobre el mensaje original y elimina campos/alcances
    estructurales; los nombres siguen resolviéndose contra el catálogo vigente.
    """
    if (
        plan.tipo != "consulta"
        or plan.alcance not in {"evento", "ambos"}
        or "movimientos" not in (plan.campos or [])
        or plan.filtros.codigos
        or plan.filtros.categoria
        or plan.referencia in {"foco", "resultados_previos"}
    ):
        return plan

    objeto = _extraer_objeto_consulta_sin_marco(texto or "")
    if objeto:
        plan.filtros.texto = objeto
    return plan


def _planificar_condicion_stock_explicita(
    texto: str,
    categorias_accesibles: Optional[List[str]] = None,
    almacenes_accesibles: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Resuelve listados/conteos globales con una condición de stock inequívoca."""
    texto_norm = normalizar(preprocesar_texto_inventario(texto or "").strip())
    if not texto_norm:
        return None

    # Esta ruta rápida solo representa una condición. Si el usuario combina
    # condiciones distintas, el planificador semántico debe resolver su
    # relación en vez de aplicar silenciosamente la primera coincidencia.
    condiciones_mencionadas = _condiciones_stock_mencionadas(texto_norm)
    if len(condiciones_mencionadas) > 1:
        return None

    if not condiciones_mencionadas:
        neg_bajo_minimo = bool(re.search(
            r"\b(?:no|ni|tampoco)\s+(?:est[ae]n?|queden?|son|sean)\s+(?:por\s+debajo\s+de(?:l)?|bajo|inferior(?:es)?\s+a(?:l)?)\s+(?:(?:su|el)\s+)?(?:stock\s+)?minimo\b|"
            r"\b(?:no|ni|tampoco)\s+(?:tienen?|tengan?)\s+(?:bajo\s+stock|stock\s+bajo|poco\s+stock)\b|"
            r"\b(?:con|tienen?)\s+stock\s+suficiente\b|"
            r"\bpor\s+encima\s+de(?:l)?\s+(?:(?:su|el)\s+)?(?:stock\s+)?minimo\b",
            texto_norm,
        ))
        neg_agotado = bool(re.search(
            r"\b(?:no|ni|tampoco)\s+(?:est[ae]n?|queden?|son|sean|andan?)\s+agotad[oa]s?\b|"
            r"\b(?:no|sin\s+estar)\s+agotad[oa]s?\b|"
            r"\b(?:que\s+)?(?:aun|todavia|si)\s+tienen?\s+(?:stock|existencias|unidades)\b|"
            r"\bcon\s+(?:stock|existencias|unidades)\s+disponibles?\b",
            texto_norm,
        ))
        if neg_bajo_minimo:
            condicion = "suficiente"
        elif neg_agotado:
            condicion = "con_stock"
        else:
            return None
    else:
        condicion_detectada = next(iter(condiciones_mencionadas))
        if condicion_detectada == "bajo_minimo" and _condicion_bajo_minimo_gobernada_por_infinitivo(texto_norm):
            return None
        condicion = condicion_detectada

    # Reconocer una categoría solo si coincide literalmente con un nombre activo
    # del catálogo. Los nombres nunca se fijan en el código y el resultado sigue
    # pasando por las reglas de permisos de HerramientasInventario.
    categorias_encontradas = []
    for categoria in categorias_accesibles or []:
        categoria_norm = normalizar(categoria)
        if not categoria_norm:
            continue
        patron_categoria = re.compile(
            rf"(?<!\w)(?:de\s+(?:(?:la|el)\s+)?(?:categoria\s+)?|categoria\s+)"
            rf"(?:categoria\s+)?{re.escape(categoria_norm)}(?!\w)"
        )
        for coincidencia in patron_categoria.finditer(texto_norm):
            categorias_encontradas.append((categoria, coincidencia.start(), coincidencia.end()))

    categoria_explicita = None
    spans_alcance: List[Tuple[int, int]] = []
    if len(categorias_encontradas) == 1:
        categoria_explicita, inicio, fin = categorias_encontradas[0]
        spans_alcance.append((inicio, fin))
    elif categorias_encontradas:
        return None

    # Resolver almacenes solo contra el catálogo accesible vigente. Se requiere
    # un conector de alcance explícito y una coincidencia única; una mención
    # desconocida o ambigua vuelve al planificador semántico.
    almacenes_encontrados: List[Tuple[str, int, int]] = []
    for almacen in sorted(set(almacenes_accesibles or []), key=lambda valor: len(normalizar(valor)), reverse=True):
        almacen_norm = normalizar(almacen)
        if not almacen_norm:
            continue
        for coincidencia in re.finditer(rf"(?<!\w){re.escape(almacen_norm)}(?!\w)", texto_norm):
            prefijo = texto_norm[:coincidencia.start()]
            conector = re.search(r"\b(?:en|dentro\s+de|del|desde|hacia|al|a)\s+(?:(?:el|la)\s+)?$", prefijo)
            if not conector:
                continue
            almacenes_encontrados.append((almacen, conector.start(), coincidencia.end()))

    nombres_almacenes = {normalizar(item[0]) for item in almacenes_encontrados}
    if len(nombres_almacenes) > 1:
        return None
    almacen_explicito = almacenes_encontrados[0][0] if almacenes_encontrados else None
    spans_alcance.extend((inicio, fin) for _, inicio, fin in almacenes_encontrados)

    # Elimina del chequeo de alcance las categorías y almacenes ya resueltos.
    # Cualquier otro código, ubicación o almacén sigue requiriendo al LLM.
    texto_sin_alcance = texto_norm
    for inicio, fin in sorted(spans_alcance, reverse=True):
        texto_sin_alcance = f"{texto_sin_alcance[:inicio]} {texto_sin_alcance[fin:]}"

    # Sin coincidencia exacta, cualquier alcance adicional conserva el plan
    # semántico; nunca se convierte una pregunta acotada en una búsqueda global.
    tiene_conjunto = bool(
        re.search(
            r"\b(?:productos?|articulos?|referencias?|inventario|cosas?|elementos?|insumos?|materiales?|equipos?|herramientas?)\b",
            texto_norm,
        )
    )
    tiene_colectivo_generico = bool(re.search(r"\b(?:lo|los|las)\s+que\b", texto_norm))
    if not (tiene_conjunto or tiene_colectivo_generico):
        return None
    if not almacen_explicito:
        # Permitir referencia genérica final «en el almacén» / «en la bodega» sin nombre propio de almacén.
        texto_sin_alcance = re.sub(
            r"\b(?:en|dentro\s+de)\s+(?:(?:el|la)\s+)?(?:almacen|bodega)\s*$",
            " ",
            texto_sin_alcance,
        ).strip()
    if re.search(
        r"\b(?:almacen|bodega|ubicacion|codigo)\b|"
        r"\b[a-z]{2,5}[-_]\d{1,5}\b|\breferencia\s+[a-z0-9]+\b",
        texto_sin_alcance,
    ):
        return None
    if not categoria_explicita and (
        re.search(r"\bcategoria\b", texto_norm)
        or re.search(r"\b(?:productos?|articulos?|referencias?)\s+(?:de|del)\s+\w+", texto_norm)
    ):
        return None

    es_conteo = bool(re.search(r"\b(?:cuantos?|cuantas?|cantidad\s+de)\s+(?:productos?|articulos?|referencias?)\b", texto_norm))
    return PlanAgente(
        tipo="consulta",
        operacion="contar" if es_conteo else "listar",
        filtros=FiltrosConsulta(
            condicion=condicion,
            categoria=categoria_explicita,
            almacen=almacen_explicito,
            categoria_explicita=bool(categoria_explicita),
        ),
        referencia="ninguna",
        campos=["stock"],
        fuente_plan=(
            "fast_path_condicion_stock_categoria_almacen"
            if categoria_explicita and almacen_explicito
            else "fast_path_condicion_stock_categoria"
            if categoria_explicita
            else "fast_path_condicion_stock_almacen"
            if almacen_explicito
            else "fast_path_condicion_stock"
        ),
    )


def _planificar_eventos_globales_explicitos(
    texto: str,
    categorias_accesibles: Optional[List[str]] = None,
    almacenes_accesibles: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Resuelve consultas globales inequívocas de movimientos o eventos por período.

    Solo se activa cuando la frase tiene marco de consulta explícito (no una orden de
    mutación), un período temporal de días completos, un sustantivo de evento del
    dominio (movimientos, préstamos, devoluciones, entradas, salidas, dotaciones o
    traslados) y ningún producto, código, persona ni referencia contextual.
    """
    texto_pre = preprocesar_texto_inventario(texto or "").strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm:
        return None
    if re.match(
        r"^(?:(?:por\s+favor|quiero|necesito|deseo|voy\s+a)\s+)*"
        r"(?:prestar|prestame|prestale|prestales|prestanos|presta|prestando|"
        r"devolver|devuelv\w*|registrar|registra\w*|hacer|haz|"
        r"ingresar|ingresa\w*|sacar|saca\w*|retirar|retira\w*|despachar|despacha\w*|"
        r"trasladar|traslada\w*|crear|crea\w*)\b",
        texto_norm,
    ):
        return None
    tiene_marco_consulta = bool(
        re.search(
            r"\b(?:muestr\w*|mostr\w*|enseñ\w*|ensen\w*|dime|digame|decime|decirme|dame|ver|saber|conocer|"
            r"list\w*|consult\w*|que|cual(?:es)?|cuant[oa]s?|numero\s+de|total\s+de|cantidad\s+de|"
            r"como\s+(?:van|estuvieron|fueron)|hubo|han\s+habido|se\s+(?:registraron|hicieron|realizaron|efectuaron|movio|movieron))\b",
            texto_norm,
        )
        or re.match(
            r"^(?:(?:los|las|el)\s+)?(?:movimientos?|historial|prestamos?|devoluciones?|entradas?|salidas?|dotaciones?|traslados?)\s+(?:de|del|en|durante)\b",
            texto_norm,
        )
    )
    menciona_evento = bool(
        re.search(
            r"\b(?:movimientos?|historial|prestamos?|devoluciones?|entradas?|salidas?|dotaciones?|traslados?|ajustes?|"
            r"se\s+movi(?:o|eron)|ha\s+movido|se\s+prest(?:o|aron|a|an)|prestaron|se\s+devolv(?:io|ieron)|devolvieron|"
            r"entr(?:o|aron|a|an)|ingres(?:o|aron|a|an)|sal(?:io|ieron|e|en|ga|gan)|se\s+despach(?:o|aron|a|an)|despacharon|"
            r"se\s+traslad(?:o|aron)|trasladaron)\b",
            texto_norm,
        )
    )
    if not (tiene_marco_consulta and menciona_evento):
        return None
    if extraer_codigos_de_texto(texto_pre):
        return None
    if _mensaje_requiere_referencia_contextual(texto_pre) or tiene_referencia_resultados_previos(texto_pre):
        return None
    if re.search(r"\b(?:quien|quienes|a\s+quien|responsables?|activos?|pendientes?|abiertos?|vigentes?)\b", texto_norm):
        return None

    try:
        from ..servicios_temporales import resolver_intervalo_temporal
        intervalo = resolver_intervalo_temporal(texto_pre)
    except Exception:
        intervalo = None
    if intervalo is not None:
        if intervalo.inicio.time() != datetime.time.min or intervalo.fin.time() != datetime.time.min:
            return None
    else:
        # Sin período explícito: permitir consultas globales de movimientos/historial
        # (excluyendo préstamos sin fecha, que corresponden a préstamos activos).
        if re.search(r"\b(?:prestamos?|prest(?:o|aron|a|an))\b", texto_norm):
            return None
        if not re.search(
            r"\b(?:movimientos?|historial|devoluciones?|entradas?|salidas?|dotaciones?|traslados?)\b",
            texto_norm,
        ):
            return None

    # Resolver almacén o categoría explícitos solo si coinciden con el catálogo accesible.
    # Si mencionan un almacén/categoría no resuelto, se delega al planificador completo
    # para conservar la validación RBAC de almacén no autorizado o inexistente.
    texto_sin_alcance = texto_pre
    almacen_explicito = None
    if re.search(r"\b(?:almacen|bodega)\b", texto_norm):
        almacenes_encontrados = [
            alm for alm in sorted(set(almacenes_accesibles or []), key=lambda v: len(normalizar(v)), reverse=True)
            if normalizar(alm) and re.search(rf"(?<!\w){re.escape(normalizar(alm))}(?!\w)", texto_norm)
        ]
        if len(almacenes_encontrados) == 1:
            almacen_explicito = almacenes_encontrados[0]
            texto_sin_alcance = re.sub(
                rf"(?<!\w){re.escape(almacen_explicito)}(?!\w)",
                " ",
                texto_sin_alcance,
                flags=re.IGNORECASE,
            )
        elif not almacenes_encontrados and not re.search(
            r"\b(?:almacen|bodega)\s+(?!hoy\b|ayer\b|anteayer\b|esta\b|este\b|el\s+mes\b|la\s+semana\b|en\b|de\b|del\b)[a-z0-9]",
            texto_norm,
        ):
            # Referencia genérica («en el almacén» / «del almacén») sin calificador específico
            almacen_explicito = None
        else:
            return None
    categoria_explicita = None
    if re.search(r"\bcategoria\b", texto_norm):
        cats_encontradas = [
            cat for cat in sorted(set(categorias_accesibles or []), key=lambda v: len(normalizar(v)), reverse=True)
            if normalizar(cat) and re.search(rf"(?<!\w){re.escape(normalizar(cat))}(?!\w)", texto_norm)
        ]
        if len(cats_encontradas) != 1:
            return None
        categoria_explicita = cats_encontradas[0]
        texto_sin_alcance = re.sub(
            rf"(?<!\w){re.escape(categoria_explicita)}(?!\w)",
            " ",
            texto_sin_alcance,
            flags=re.IGNORECASE,
        )

    # Si queda algún descriptor de producto o persona tras quitar marco, período, cortesía,
    # participios de evento y almacén/categoría, no es una consulta global de eventos.
    texto_sin_alcance = re.sub(
        r"\b(?:registrad[oa]s?|hech[oa]s?|realizad[oa]s?|efectuad[oa]s?|ocurrid[oa]s?|pasad[oa]s?|"
        r"recientes?|[uú]ltim[oa]s?|generales?|todos?|todas?|"
        r"favor|colabora|revisando|viendo|reg[aá]leme|h[aá]game|haceme|decime|mostrame|por\s+favor|"
        r"bodega|almac[eé]n|inventario)\b",
        " ",
        texto_sin_alcance,
        flags=re.IGNORECASE,
    )
    if _extraer_objeto_consulta_sin_marco(texto_sin_alcance):
        return None

    fecha_desde, fecha_hasta = _fechas_intervalo_iso(intervalo) if intervalo is not None else (None, None)
    tipo_mov = _inferir_tipo_movimiento_desde_etiqueta(texto_pre, "evento", None)
    if tipo_mov is None and re.search(r"\btraslados?\b", texto_norm):
        tipo_mov = "TRASLADO"

    es_conteo = bool(
        re.search(r"\b(?:cuant[oa]s?|numero\s+de|total\s+de|cantidad\s+de)\b", texto_norm)
        and not re.search(r"\b(?:muestr\w*|list\w*|ver|detalle|cuales)\b", texto_norm)
    )
    return PlanAgente(
        tipo="consulta",
        operacion="contar" if es_conteo else "listar",
        filtros=FiltrosConsulta(
            almacen=almacen_explicito,
            categoria=categoria_explicita,
            categoria_explicita=bool(categoria_explicita),
        ),
        referencia="ninguna",
        alcance="evento",
        tipo_movimiento=tipo_mov,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
        campos=["movimientos"],
        medida_solicitada="evento",
        alcance_entidad="global",
        alcance_global_confirmado=True,
        fuente_plan="fast_path_eventos_globales",
    )


def _planificar_prestamos_activos_globales(
    texto: str,
    categorias_accesibles: Optional[List[str]] = None,
    almacenes_accesibles: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Resuelve consultas globales de préstamos activos/vigentes sin contaminar con foco previo."""
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm:
        return None
    if extraer_codigos_de_texto(texto_pre):
        return None
    # Si hay una operación de escritura, consulta de elegibilidad o consulta cuantitativa, delegar.
    # Nota: «sin devolver» / «no han devuelto» / «pendientes de devolución» describen préstamos activos, no una devolución.
    texto_norm_sin_estado_dev = re.sub(
        r"\b(?:sin\s+devolver|no\s+(?:han?\s+|lo\s+han?\s+)?devuelto|pendientes?\s+de\s+devolucion)\b",
        " ",
        texto_norm,
    )
    if re.search(
        r"\b(?:presta(?:r|le|les|me|nos)?|prestando|devolv\w*|devuelve\w*|registra\w*|"
        r"prestable(?:s)?|consumible(?:s)?|retornable(?:s)?|pueden?\s+prestar|"
        r"cuant[oa]s?|cantidad(?:es)?|unidades?|total(?:es)?|numero\s+de|saldos?)\b",
        texto_norm_sin_estado_dev,
    ):
        return None
    # Si tiene anáfora real hacia un producto en foco («sus préstamos», «¿tiene préstamos?», «de ese»), delegar.
    if (
        _mensaje_requiere_referencia_contextual(texto_pre)
        or tiene_referencia_resultados_previos(texto_pre)
        or re.search(r"\b(?:su|sus|ese|esa|esos|esas|este|esta|estos|estas|dicho|mismo|misma|tiene\s+prestamos?)\b", texto_norm)
    ):
        return None
    # Si especifica un período temporal («préstamos de hoy/ayer»), pertenece a eventos fechados.
    try:
        from ..servicios_temporales import resolver_intervalo_temporal
        if resolver_intervalo_temporal(texto_pre) is not None:
            return None
    except Exception:
        pass

    es_listado_prestamos_global = bool(
        re.search(
            r"\b(?:muestr\w*|mostr\w*|enseñ\w*|ensen\w*|dime|digame|decime|decirme|dame|regal\w*|ver|saber|list\w*|consult\w*|cuales(?:\s+son)?|que)\s+"
            r"(?:(?:los|las|la|el)\s+)?(?:lista\s+de\s+|listado\s+de\s+)?(?:(?:los|las)\s+)?prestamos(?:\s+(?:que\s+estan\s+)?(?:activos?|vigentes?|pendientes?(?:\s+de\s+devolucion)?|abiertos?|actuales?|actualmente|ahora(?:\s+mismo)?))*\b|"
            r"^(?:(?:los|la|el)\s+)?(?:lista\s+de\s+|listado\s+de\s+)?prestamos(?:\s+(?:activos?|vigentes?|pendientes?|abiertos?|actuales?|actualmente))+\b|"
            r"^(?:(?:los|la|el)\s+)?(?:lista\s+de\s+|listado\s+de\s+)?prestamos(?:\s+(?:de|del|en)\s+(?:(?:el|la)\s+)?(?:almacen|bodega|categoria|inventario))\b|"
            r"\b(?:quien(?:es)?\s+tiene(?:n)?|a\s+quien(?:es)?\s+le(?:s)?\s+prestaron|"
            r"que\s+(?:productos?|articulos?|cosas?|elementos?|equipos?|materiales?|herramientas?)?\s*(?:estan?|esta|tenemos|hay))\s+"
            r"(?:(?:cosas|equipos?|materiales?|articulos?|productos?|elementos?|herramientas?)\s+)?"
            r"(?:afuera\s+)?(?:en\s+prestamo|prestado[sa]?s?)(?:\s+(?:sin\s+devolver|que\s+(?:todavia|aun)\s+no\s+(?:han\s+)?devuelto|todavia|aun|ahora(?:\s+mismo)?|actualmente))?\b|"
            r"\ba\s+quien(?:es)?\s+le(?:s)?\s+prestaron\s+(?:(?:los|las|unos|unas)\s+)?(?:equipos?|materiales?|articulos?|productos?|cosas?|elementos?|herramientas?)\b",
            texto_norm,
        )
    )
    if not es_listado_prestamos_global:
        return None

    texto_sin_alcance = texto_pre
    almacen_explicito = None
    if re.search(r"\b(?:almacen|bodega)\b", texto_norm):
        almacenes_encontrados = [
            alm for alm in sorted(set(almacenes_accesibles or []), key=lambda v: len(normalizar(v)), reverse=True)
            if normalizar(alm) and re.search(rf"(?<!\w){re.escape(normalizar(alm))}(?!\w)", texto_norm)
        ]
        if len(almacenes_encontrados) == 1:
            almacen_explicito = almacenes_encontrados[0]
            texto_sin_alcance = re.sub(
                rf"(?<!\w){re.escape(almacen_explicito)}(?!\w)",
                " ",
                texto_sin_alcance,
                flags=re.IGNORECASE,
            )
        elif not almacenes_encontrados and not re.search(
            r"\b(?:almacen|bodega)\s+(?!hoy\b|ayer\b|anteayer\b|esta\b|este\b|el\s+mes\b|la\s+semana\b|en\b|de\b|del\b)[a-z0-9]",
            texto_norm,
        ):
            # Referencia genérica («en el almacén» / «del almacén») sin calificador específico
            almacen_explicito = None
        else:
            return None
    categoria_explicita = None
    if re.search(r"\bcategoria\b", texto_norm):
        cats_encontradas = [
            cat for cat in sorted(set(categorias_accesibles or []), key=lambda v: len(normalizar(v)), reverse=True)
            if normalizar(cat) and re.search(rf"(?<!\w){re.escape(normalizar(cat))}(?!\w)", texto_norm)
        ]
        if len(cats_encontradas) != 1:
            return None
        categoria_explicita = cats_encontradas[0]
        texto_sin_alcance = re.sub(
            rf"(?<!\w){re.escape(categoria_explicita)}(?!\w)",
            " ",
            texto_sin_alcance,
            flags=re.IGNORECASE,
        )

    texto_sin_alcance = re.sub(
        r"\b(?:activos?|vigentes?|pendientes?|devoluci[oó]n|devolver|devuelto|abiertos?|actuales?|actualmente|"
        r"ahora|mismo|momento|todav[ií]a|a[uú]n|afuera|prestados?|prestadas?|prestaron|pr[eé]stamos?|"
        r"cosas?|equipos?|materiales?|elementos?|art[ií]culos?|productos?|herramientas?|"
        r"a|le|les|qui[eé]n(?:es)?|tiene(?:n)?|tenemos|est[aá]n?|han|no|sin|me|colabora|viendo|revisando|reg[aá]leme|por\s+favor)\b",
        " ",
        texto_sin_alcance,
        flags=re.IGNORECASE,
    )
    if _extraer_objeto_consulta_sin_marco(texto_sin_alcance):
        return None

    return PlanAgente(
        tipo="consulta",
        operacion="listar",
        filtros=FiltrosConsulta(
            condicion="con_prestamos",
            almacen=almacen_explicito,
            categoria=categoria_explicita,
            categoria_explicita=bool(categoria_explicita),
        ),
        referencia="ninguna",
        alcance="actual",
        campos=["prestamos"],
        medida_solicitada="ninguna",
        alcance_entidad="coleccion",
        alcance_global_confirmado=True,
        fuente_plan="fast_path_prestamos_activos_globales",
    )


def _planificar_elegibilidad_codigo_explicito(texto: str) -> Optional[PlanAgente]:
    """Resuelve consultas de elegibilidad (prestable/consumible/retornable) sobre un código explícito."""
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm:
        return None
    codigos = extraer_codigos_de_texto(texto_pre)
    if len(codigos) != 1:
        return None
    # Descartar si es una solicitud operativa de préstamo/salida (cantidades o destinatario)
    es_pregunta_se_puede_devolver = bool(
        re.search(r"\b(?:se\s+puede|podemos|se\s+pueden)\s+devolver\b", texto_norm)
    )
    if re.search(
        r"\b(?:prestame|prestarle|presta(?:le|les|me|nos)|registrar|necesito\s+prestar|quiero\s+prestar|"
        r"sacar|retirar|ingresar|trasladar|unidades?|cuant[oa]s?|donde|ubicacion|stock)\b",
        texto_norm,
    ) or (not es_pregunta_se_puede_devolver and re.search(r"\bdevolver\b", texto_norm)):
        return None
    es_elegibilidad = bool(
        es_pregunta_se_puede_devolver
        or re.search(
            r"\b(?:prestable(?:s)?|consumible(?:s)?|retornable(?:s)?|devolutiv[oa]s?|"
            r"se\s+puede\s+prestar|podemos\s+prestar|se\s+presta(?:\s+o\s+se\s+gasta)?|"
            r"solo\s+se\s+entrega\s+definitiv[oa]|se\s+gasta)\b",
            texto_norm,
        )
    )
    if not es_elegibilidad:
        return None
    return PlanAgente(
        tipo="consulta",
        operacion="detalle",
        filtros=FiltrosConsulta(codigos=codigos),
        referencia="ninguna",
        campos=["es_prestable"],
        alcance_entidad="producto",
        fuente_plan="fast_path_elegibilidad_codigo",
    )


def _planificar_categorias_globales(texto: str) -> Optional[PlanAgente]:
    """Resuelve consultas globales inequívocas de listado o conteo de categorías."""
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm:
        return None
    if extraer_codigos_de_texto(texto_pre):
        return None
    if (
        _mensaje_requiere_referencia_contextual(texto_pre)
        or tiene_referencia_resultados_previos(texto_pre)
        or re.search(
            r"\b(?:su|sus|ese|esa|esos|esas|este|esta|estos|estas|dicho|mismo|misma|"
            r"productos?|articulos?|referencias?|almacen|almacenes|bodega|bodegas|"
            r"crear|agregar|editar|modificar|eliminar|borrar|registrar)\b",
            texto_norm,
        )
    ):
        return None
    es_conteo = bool(
        re.search(r"\bcuantas\s+categorias\b", texto_norm)
        or re.fullmatch(r"(?:hay|existen?|tenemos) (?:alguna )?categorias?(?: registradas?)?", texto_norm)
    )
    es_listado = bool(
        re.search(
            r"\b(?:que\s+categorias?\s+(?:hay|existen?|tenemos|manejamos|registradas?|creadas?|estan\s+registradas?)|"
            r"cuales\s+(?:son\s+(?:todas\s+)?(?:las\s+)?)?categorias?(?:\s+(?:hay|existen?|tenemos|manejamos|registradas?|del\s+inventario))?|"
            r"(?:lista(?:r|me)?|muestr\w*|mostr\w*|ensen\w*|dime|digame|decime|dame|regal\w*|ver)\s+"
            r"(?:(?:la\s+)?(?:lista|listado)\s+de\s+)?(?:(?:todas\s+)?las\s+)?categorias?)\b",
            texto_norm,
        )
    )
    if not (es_conteo or es_listado):
        return None
    return PlanAgente(
        tipo="consulta",
        operacion="contar" if es_conteo else "listar",
        referencia="ninguna",
        campos=["categoria"],
        alcance_entidad="global",
        alcance_global_confirmado=True,
        fuente_plan="fast_path_categorias_globales",
    )


def _planificar_listado_global_explicito(texto: str) -> Optional[PlanAgente]:
    """Resuelve solicitudes inequívocas de ver el listado/catálogo completo de productos."""
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm:
        return None
    if extraer_codigos_de_texto(texto_pre):
        return None
    # La existencia de referencias en el catálogo es independiente de su saldo.
    # Exigir la frase completa conserva los filtros, nombres y ubicaciones explícitos.
    nombres = r"(?:productos?|articulos?|referencias?|materiales?|insumos?|items?)"
    marco = r"(?:\s+registrad[oa]s?)?(?:\s+(?:en|del?)\s+(?:(?:el|nuestro)\s+)?(?:inventario|catalogo))?"
    peticion = re.sub(
        r"^(?:(?:hola|ha[ck]u|por favor|dime si|quiero saber si|quisiera saber si)\s+)+",
        "", texto_norm,
    ).removesuffix(" por favor")
    cuantificador = r"(?:(?:algun(?:os?|as?)?|un[oa]?)\s+)?"
    existencia = rf"(?:hay|tenemos|tienen|existen?|el inventario tiene)\s+{cuantificador}{nombres}"
    cantidad = rf"cuant[oa]s\s+{nombres}(?:\s+(?:hay|tenemos|existen))?"
    busqueda = rf"(?:busco|busca(?:r|ndo)?|muestrame|ver|lista(?:r)?|que)\s+{nombres}(?:\s+(?:hay|tenemos))?"
    inventario = r"(?:en (?:el |nuestro )?(?:inventario|catalogo))"
    if re.fullmatch(
        rf"cuantas unidades(?: disponibles)? (?:hay|tenemos)(?: disponibles)? (?:en total(?: {inventario})?|{inventario})"
        rf"|total de unidades disponibles(?: {inventario})?", peticion,
    ):
        return PlanAgente(
            tipo="consulta", operacion="sumar_unidades", campos=["stock"],
            medida_solicitada="disponible", referencia="ninguna", alcance_entidad="global",
            alcance_global_confirmado=True, fuente_plan="fast_path_unidades_globales",
        )
    ambito_catalogo = rf"(?:en (?:(?:el|los|las|nuestro|nuestros) )?(?:inventario|catalogo|{nombres}))"
    busqueda = rf"(?:{busqueda}|que (?:hay|tenemos) {ambito_catalogo}|{nombres} registrad[oa]s?)"
    existencia = rf"(?:{existencia}|(?:hay|tenemos|existen?) algo(?: registrado)? {ambito_catalogo}|se ha registrado {cuantificador}{nombres})"
    if re.fullmatch(
        rf"(?:hay|tenemos|existen?) (?:stock|existencias)(?: disponibles?)?(?: {inventario})?",
        peticion,
    ):
        return PlanAgente(
            tipo="consulta",
            operacion="listar",
            filtros=FiltrosConsulta(condicion="con_stock"),
            referencia="ninguna",
            alcance_entidad="global",
            alcance_global_confirmado=True,
            fuente_plan="fast_path_estado_global",
        )
    estados = {
        "disponible": "con_stock", "disponibles": "con_stock", "con stock": "con_stock",
        "en stock": "con_stock", "con existencias": "con_stock", "en prestamo": "con_prestamos",
    }
    estado = "|".join(estados)
    consulta = re.fullmatch(
        rf"(?:(?P<listado>{busqueda})|{existencia}|{cantidad})(?: (?P<estado>{estado}))?{marco}",
        peticion,
    )
    if consulta:
        condicion = estados.get(consulta["estado"])
        return PlanAgente(
            tipo="consulta",
            operacion="listar" if consulta["listado"] else "contar",
            filtros=FiltrosConsulta(condicion=condicion),
            referencia="ninguna",
            alcance_entidad="global",
            alcance_global_confirmado=True,
            fuente_plan="fast_path_estado_global" if condicion else "fast_path_listado_global",
        )
    if _condiciones_stock_mencionadas(texto_norm):
        return None
    if re.match(
        r"^(?:(?:hola(?:\s+ha[ck]u)?|ha[ck]u|por\s+favor|quiero\s+ver|me\s+colabora)\s*)*"
        r"(?:(?:muestr\w*|mostr\w*|ensen\w*|desplieg\w*|pasame|regal\w*|dame|ver|lista(?:r|me)?|consultar?)\s+(?:por\s+favor\s+)?(?:a\s+)?)?"
        r"(?:(?:la\s+|el\s+)?(?:lista|listado)(?:\s+(?:complet[oa]|general|total))?\s+de\s+"
        r"(?:(?:tod[oa]s?\s+)?(?:los|las)\s+)?(?:productos?|articulos?|referencias?)|"
        r"(?:(?:tod[oa]s?\s+)?los\s+(?:productos?|articulos?|referencias?))|"
        r"(?:(?:todo\s+)?(?:el\s+)?catalogo(?:\s+(?:completo|general|de\s+productos?))?))"
        r"(?:\s+(?:que\s+(?:hay|tenemos|existen)(?:\s+registrad[oa]s?)?|registrad[oa]s?|del\s+inventario|en\s+(?:el\s+)?inventario|actual(?:es)?))*"
        r"(?:\s+por\s+favor)?$",
        texto_norm,
    ):
        return PlanAgente(
            tipo="consulta",
            operacion="listar",
            referencia="ninguna",
            campos=[],
            alcance_entidad="global",
            alcance_global_confirmado=True,
            fuente_plan="fast_path_listado_global",
        )
    return None


def _planificar_crear_producto_sin_nombre(texto: str) -> Optional[PlanAgente]:
    """Resuelve solicitudes directas de alta de producto sin datos aún, sin heredar foco previo."""
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm or extraer_codigos_de_texto(texto_pre):
        return None
    if re.match(
        r"^(?:(?:hola(?:\s+ha[ck]u)?|ha[ck]u|por\s+favor)\s*)*"
        r"(?:(?:quiero|necesito|deseo|voy\s+a|me\s+gustaria|puedes|podrias)\s+)?"
        r"(?:(?:crear|registrar|agregar|anadir|dar\s+de\s+alta)\s+(?:un\s+)?(?:nuevo\s+)?(?:producto|articulo)(?:\s+nuevo)?|"
        r"(?:ingresar|meter)\s+(?:un\s+)?(?:nuevo\s+(?:producto|articulo)|(?:producto|articulo)\s+nuevo))"
        r"(?:\s+(?:al|en\s+el)\s+(?:inventario|catalogo))?"
        r"(?:\s+por\s+favor)?$",
        texto_norm,
    ):
        return PlanAgente(
            tipo="operacion",
            accion="crear_producto",
            producto=None,
            referencia="ninguna",
            fuente_plan="fast_path_crear_producto_formulario",
        )
    return None


def _planificar_consulta_almacen_global(
    texto: str,
    almacenes_accesibles: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Resuelve conteos o listados globales de productos en un almacén accesible sin heredar foco previo."""
    if not almacenes_accesibles:
        return None
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm:
        return None
    if extraer_codigos_de_texto(texto_pre) or _condiciones_stock_mencionadas(texto_norm):
        return None
    if (
        _mensaje_requiere_referencia_contextual(texto_pre)
        or tiene_referencia_resultados_previos(texto_pre)
        or re.search(
            r"\b(?:su|sus|ese|esa|esos|esas|este|esta|estos|estas|dicho|mismo|misma|"
            r"movimientos?|entradas?|salidas?|prestamos?|devoluciones?|traslados?|"
            r"ubicacion(?:es)?|estantes?|categorias?|unidades?|prestar|sacar|ingresar|retirar|crear|registrar)\b",
            texto_norm,
        )
    ):
        return None
    try:
        from ..servicios_temporales import resolver_intervalo_temporal
        if resolver_intervalo_temporal(texto_pre) is not None:
            return None
    except Exception:
        pass

    es_conteo = bool(
        re.search(
            r"\b(?:cuant[oa]s?|numero\s+de|total\s+de|cantidad\s+de)\s+"
            r"(?:productos?|articulos?|referencias?|items?)\b",
            texto_norm,
        )
    )
    es_listado = bool(
        re.search(
            r"\b(?:que|cuales|muestr\w*|mostr\w*|list\w*|ver|dime|dame|regal\w*)\s+"
            r"(?:(?:los|las|todos\s+los)\s+)?(?:productos?|articulos?|referencias?|catalogo|inventario)\b",
            texto_norm,
        )
        or re.search(
            r"\b(?:(?:hay|tenemos|existen?)\s+(?:algo|algun(?:os?|as?)?\s+(?:productos?|articulos?|referencias?))|"
            r"que\s+(?:hay|tenemos))\s+en\b",
            texto_norm,
        )
    )
    if not (es_conteo or es_listado):
        return None

    almacenes_encontrados = _menciones_almacenes_catalogo(texto_pre, list(almacenes_accesibles))
    if len(almacenes_encontrados) != 1:
        return None
    almacen_explicito = almacenes_encontrados[0]
    alm_norm_exp = normalizar(almacen_explicito)
    patrones_elim_alm = [re.escape(almacen_explicito), re.escape(alm_norm_exp)]
    if alm_norm_exp.startswith("almacen "):
        patrones_elim_alm.append(re.escape("bodega " + alm_norm_exp[len("almacen "):]))
    elif alm_norm_exp.startswith("bodega "):
        patrones_elim_alm.append(re.escape("almacen " + alm_norm_exp[len("bodega "):]))
    texto_sin_alcance = texto_pre
    for pat_alm in patrones_elim_alm:
        texto_sin_alcance = re.sub(
            rf"(?<!\w){pat_alm}(?!\w)",
            " ",
            texto_sin_alcance,
            flags=re.IGNORECASE,
        )
    texto_sin_alcance = re.sub(
        r"\b(?:registrad[oa]s?|activos?|totales?|en\s+total|almac[eé]n|bodega|inventario)\b",
        " ",
        texto_sin_alcance,
        flags=re.IGNORECASE,
    )
    if _extraer_objeto_consulta_sin_marco(texto_sin_alcance):
        return None

    return PlanAgente(
        tipo="consulta",
        operacion="contar" if es_conteo else "listar",
        filtros=FiltrosConsulta(almacen=almacen_explicito),
        referencia="ninguna",
        campos=[],
        alcance_entidad="coleccion",
        alcance_global_confirmado=True,
        fuente_plan="fast_path_consulta_almacen_global",
    )


def _planificar_atributo_foco_eliptico(texto: str, tiene_foco_unico: bool) -> Optional[PlanAgente]:
    """Resuelve preguntas elípticas inequívocas por un atributo único del producto en foco."""
    if not tiene_foco_unico:
        return None
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm or extraer_codigos_de_texto(texto_pre):
        return None
    texto_norm = re.sub(
        r"^(?:(?:si|aja|ok|bueno|vale|claro|perfecto|entendido|bien)[,\s]+)*"
        r"(?:y\s+)?(?:(?:me\s+(?:dice|indica|confirma)|dime|decime|digame)\s+)?",
        "",
        texto_norm,
    ).strip()

    campo_objetivo: Optional[str] = None
    if re.match(
        r"^(?:cual\s+es\s+(?:su|la)\s+categoria|que\s+categoria\s+(?:tiene|es)|a\s+que\s+categoria\s+pertenece|su\s+categoria)$",
        texto_norm,
    ):
        campo_objetivo = "categoria"
    elif re.match(
        r"^(?:es\s+prestable(?:\s+o\s+(?:consumible|no))?|se\s+puede\s+prestar(?:\s+o\s+es\s+consumible)?|es\s+consumible(?:\s+o\s+prestable)?|es\s+retornable)$",
        texto_norm,
    ):
        campo_objetivo = "es_prestable"
    elif re.match(
        r"^(?:quien(?:es)?\s+(?:lo|la|los|las)\s+tiene(?:n)?(?:\s+(?:prestado[sa]?s?|en\s+prestamo|ahora(?:\s+mismo)?|actualmente))?|"
        r"a\s+quien(?:es)?\s+(?:esta|estan|se\s+(?:le|les)\s+presto|se\s+(?:lo|la|los|las)\s+prestaron|se\s+presto)(?:\s+prestado[sa]?s?)?|"
        r"quien\s+tiene\s+(?:el\s+)?prestamo)$",
        texto_norm,
    ):
        campo_objetivo = "prestamos"

    if not campo_objetivo:
        return None
    return PlanAgente(
        tipo="consulta",
        operacion="detalle",
        referencia="foco",
        campos=[campo_objetivo],
        medida_solicitada="ninguna",
        alcance_entidad="producto",
        fuente_plan="fast_path_atributo_foco",
    )


def _planificar_prioridad_reposicion(texto: str) -> Optional[PlanAgente]:
    """Lista candidatos de reposición para preguntas globales de riesgo de faltantes."""
    texto_norm = normalizar(preprocesar_texto_inventario(texto or ""))
    pide_prioridad = bool(re.search(r"\b(?:prioriz\w*|repon\w*|abastec\w*)\b", texto_norm))
    expresa_riesgo = bool(
        re.search(r"\b(?:riesgo|faltantes?|escasez|quedar(?:me|nos|se)?\s+sin|sin\s+(?:stock|existencias))\b", texto_norm)
    )
    menciona_conjunto = bool(
        re.search(r"\b(?:productos?|articulos?|materiales?|insumos?|inventario|existencias?)\b", texto_norm)
    )
    if not (pide_prioridad and expresa_riesgo and menciona_conjunto):
        return None

    # Una petición acotada a ubicación, categoría o código conserva el plan
    # semántico habitual para no ensanchar silenciosamente su alcance.
    if re.search(
        r"\b(?:almacen|bodega|ubicacion|categoria|codigo)\b|\b[a-z]{2,5}[-_]\d{1,5}\b|"
        r"\b(?:el|la|un|una)\s+(?:producto|articulo|material|insumo)\b",
        texto_norm,
    ):
        return None

    return PlanAgente(
        tipo="consulta",
        operacion="listar",
        filtros=FiltrosConsulta(condicion="bajo_minimo"),
        referencia="ninguna",
        campos=["stock", "stock_minimo"],
        fuente_plan="fast_path_prioridad_reposicion",
    )


def _planificar_cambio_stock_minimo_explicito(texto: str) -> Optional[PlanAgente]:
    """Prepara una propuesta solo si acción, código y nuevo valor son explícitos."""
    texto_norm = normalizar(preprocesar_texto_inventario(texto))
    if not re.search(
        r"\b(?:cambia(?:r|do)?|modifica(?:r|do)?|ajusta(?:r|do)?|actualiza(?:r|do)?)\b",
        texto_norm,
    ) or not re.search(r"\bstock\s+minimo\b", texto_norm):
        return None

    codigos = extraer_codigos_de_texto(texto)
    if len(codigos) != 1:
        return None

    texto_sin_codigos = texto_norm
    for codigo in codigos:
        codigo_norm = normalizar(codigo)
        texto_sin_codigos = re.sub(rf"(?<!\w){re.escape(codigo_norm)}(?!\w)", " ", texto_sin_codigos)
    valores = re.findall(r"\b(?:a|en|por|hasta)\s+(\d+)\b", texto_sin_codigos)
    if len(valores) != 1:
        return None

    return PlanAgente(
        tipo="operacion",
        filtros=FiltrosConsulta(codigos=codigos),
        accion="cambiar_stock_minimo",
        valor=int(valores[0]),
        referencia="ninguna",
        fuente_plan="fast_path_cambio_stock_minimo_explicito",
    )


def _planificar_movimiento_explicito_codigo(texto: str) -> Optional[PlanAgente]:
    """Resuelve órdenes inequívocas de entrada/salida cuando verbo, cantidad y código son explícitos.

    Solo prepara la propuesta de confirmación obligatoria (nunca ejecuta escritura directa).
    Delega al planificador completo si hay almacén, destinatario, motivo o varios códigos.
    """
    from ..lenguaje.observaciones import separar_observacion_operacion
    texto_sin_obs, _ = separar_observacion_operacion(texto or "")
    texto_pre = preprocesar_texto_inventario(texto_sin_obs or texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm or "?" in (texto or "") or "¿" in (texto or ""):
        return None
    # Nunca interceptar consultas, elegibilidad, eventos fechados ni operaciones con almacén/destinatario
    if re.search(
        r"\b(?:que|cuant[oa]s?|cuales?|hubo|hoy|ayer|anteayer|esta\s+semana|este\s+mes|"
        r"prestable|consumible|retornable|stock|disponible|almacen|bodega|ubicacion|"
        r"prestamo|prestar|devolucion|devolver|traslado|trasladar)\b",
        texto_norm,
    ):
        return None

    es_entrada = bool(
        re.search(
            r"\b(?:(?:registr[aeo]\w*|gener[aeo]\w*|cre[aeo]\w*|hac(?:er|e)|da(?:r|me))\s+(?:una?\s+)?entrada|"
            r"ingres[aeo]\w*|met[aeo]\w*)\b",
            texto_norm,
        )
        and not re.search(r"\b(?:salidas?|sacar|sac[aeo]\w*|retir[aeo]\w*|despach[aeo]\w*)\b", texto_norm)
    )
    es_salida = bool(
        re.search(
            r"\b(?:(?:registr[aeo]\w*|gener[aeo]\w*|cre[aeo]\w*|hac(?:er|e)|da(?:r|me))\s+(?:una?\s+)?salida|"
            r"sacar|sac[aeo]\w*|retir[aeo]\w*|despach[aeo]\w*)\b",
            texto_norm,
        )
        and not re.search(r"\b(?:entradas?|ingres[aeo]\w*|met[aeo]\w*)\b", texto_norm)
    )
    if not (es_entrada ^ es_salida):
        return None

    from .herramientas import normalizar_codigos_hablados_en_texto
    texto_cod_norm = normalizar_codigos_hablados_en_texto(texto_pre)
    codigos = extraer_codigos_de_texto(texto_cod_norm)
    if len(codigos) != 1:
        return None

    texto_sin_cod = normalizar(texto_cod_norm)
    for cod in codigos:
        cod_norm = normalizar(cod)
        texto_sin_cod = re.sub(rf"(?<!\w){re.escape(cod_norm)}(?!\w)", " ", texto_sin_cod)
    # También limpiar el código original y la cláusula de destinatario («para Práctica 1») antes de extraer cantidad
    texto_sin_cod = re.sub(r"\b[a-z]{2,5}[-_]\d{1,5}\b", " ", texto_sin_cod)
    texto_para_cant = re.sub(
        r"\b(?:para|a\s+nombre\s+de|entregar\s+a|destino)\s+.+$",
        " ",
        texto_sin_cod,
    )

    from ..lenguaje.numeros import extraer_cantidad_operacion_texto

    cant_val = extraer_cantidad_operacion_texto(texto_para_cant, permitir_cero=False)
    if not cant_val or cant_val <= 0:
        return None

    return PlanAgente(
        tipo="operacion",
        accion="entrada" if es_entrada else "salida",
        producto=codigos[0],
        filtros=FiltrosConsulta(codigos=codigos),
        cantidad=cant_val,
        referencia="ninguna",
        fuente_plan="fast_path_movimiento_explicito_codigo",
    )


def _planificar_devolucion_determinista(texto: str) -> Optional[PlanAgente]:
    """Resuelve órdenes y reportes naturales de devolución de préstamo de forma general y determinista.

    Cubre cualquier conjugación de devolver/regresar/retornar/reintegrar y giros naturales
    ("ya trajo el...", "entregó de vuelta...", "cerrar el préstamo de..."), excluyendo
    estrictamente preguntas, reportes históricos y consultas de estado/elegibilidad.
    """
    from ..lenguaje.observaciones import separar_observacion_operacion
    texto_sin_obs, _ = separar_observacion_operacion(texto or "")
    texto_pre = preprocesar_texto_inventario(texto_sin_obs or texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm or "?" in (texto or "") or "¿" in (texto or ""):
        return None
    # Exclusiones estrictas: preguntas, consultas históricas, reportes o estados de préstamo
    if re.search(
        r"^(?:que|cuant[oa]s?|cuales?|quien(?:es)?|donde|como|muestr\w*|mostr\w*|list\w*|dime|digame|consult\w*|ver|dame\s+(?:los|las)\s+devoluciones)\b|"
        r"\b(?:hubo|movimientos?\s+de\s+devolucion|historial|se\s+puede\s+devolver|es\s+retornable|retornables?|"
        r"sin\s+devolver|no\s+(?:han?\s+|lo\s+han?\s+|me\s+han?\s+)?devuelto|pendientes?\s+de\s+devolucion|falta(?:n)?\s+(?:por\s+)?devolver|por\s+devolver|"
        r"ayer|anteayer|esta\s+semana|este\s+mes|mes\s+pasado|semana\s+pasada)\b",
        texto_norm,
    ):
        return None

    # Familia verbal general de devolución (cualquier conjugación) y sinónimos transitivos de retorno
    es_orden_devolucion = bool(
        re.search(
            r"\b(?:devolv(?:er(?:me|nos|lo|la|los|las)?|io|ieron|i|imos|iendo|amos|emos)|"
            r"devuelv(?:e|en|o|a|an|eme|elo|ela)|devuelto)|"
            r"\b(?:retorn(?:ar|o|aron|e|an|ando)|reintegr(?:ar|o|aron|e|an|ando)|"
            r"regres(?:ar|o|aron|a|an|ando)\s+(?:el|la|los|las|un|una|uno|dos|tres|cuatro|cinco|\d+|su|mi|lo|esta|este))|"
            r"\b(?:devolucion\s+(?:de|del)|(?:registr\w*|hac(?:er|e)|haz|gener\w*|anot\w*|apunt\w*|proces\w*|carg\w*|ingres\w*|tom\w*|pon\w*)\s+(?:una?\s+|la\s+)?devolucion)\b|"
            r"\b(?:trajo|trajeron|trae|traen|traje|trajimos)\s+(?:de\s+vuelta|otra\s+vez|el|la|los|las|un|una|\d+)|"
            r"\b(?:ya\s+volvio\s+(?:el|la)|entreg(?:o|aron|a|an)\s+(?:de\s+vuelta|el\s+prestamo|la\s+devolucion)|"
            r"(?:cerrar|cierra|cierren|saldar|liquidar)\s+(?:el\s+)?prestamo)\b",
            texto_norm,
        )
    )
    if not es_orden_devolucion:
        return None

    from .herramientas import normalizar_codigos_hablados_en_texto
    from ..lenguaje.numeros import extraer_numero_espanol

    texto_cod_norm = normalizar_codigos_hablados_en_texto(texto_pre)
    codigos = extraer_codigos_de_texto(texto_cod_norm)

    # Extraer cantidad explícita solo si acompaña claramente al verbo de devolución o a "unidades/piezas"
    texto_sin_cod = normalizar(texto_cod_norm)
    for cod in codigos:
        texto_sin_cod = re.sub(rf"(?<!\w){re.escape(normalizar(cod))}(?!\w)", " ", texto_sin_cod)
    texto_sin_cod = re.sub(r"\b[a-z]{2,5}[-_]\d{1,5}\b", " ", texto_sin_cod)
    cantidad = None
    m_c_dig = re.search(
        r"\b(?:devolv\w*|devuelv\w*|devolucion\s+de|regres\w*|retorn\w*|reintegr\w*|trajo|trajeron|trae|entreg\w*)\s+(?:(?:de\s+vuelta|otra\s+vez)\s+)?(\d+)\b|"
        r"\b(\d+)\s+(?:unidades?|piezas?)\b",
        texto_sin_cod,
    )
    if m_c_dig:
        val_c = int(m_c_dig.group(1) or m_c_dig.group(2))
        if val_c > 0:
            cantidad = val_c
    else:
        m_c_pal = re.search(
            r"\b(?:devolv\w*|devuelv\w*|devolucion\s+de|regres\w*|retorn\w*|reintegr\w*|trajo|trajeron|trae)\s+(?:(?:de\s+vuelta|otra\s+vez)\s+)?"
            r"([a-z]+(?:\s+y\s+[a-z]+)?)\b",
            texto_sin_cod,
        )
        if m_c_pal:
            val_p = extraer_numero_espanol(m_c_pal.group(1), permitir_cero=False, solo_numero_exacto=True)
            if val_p and val_p > 0:
                cantidad = val_p

    return PlanAgente(
        tipo="operacion",
        accion="devolucion",
        producto=codigos[0] if len(codigos) == 1 else None,
        filtros=FiltrosConsulta(codigos=codigos[:1]) if len(codigos) == 1 else FiltrosConsulta(),
        cantidad=cantidad,
        referencia="ninguna",
        fuente_plan="fast_path_operacion_devolucion",
    )


def _planificar_responsable_prestamo_codigo(texto: str) -> Optional[PlanAgente]:
    """Resuelve consultas directas por el responsable de préstamos activos de un código explícito."""
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm:
        return None
    codigos = extraer_codigos_de_texto(texto_pre)
    if len(codigos) != 1:
        return None
    if re.search(r"\b(?:ultimo|ultima|historial|movimientos?|saco|retiro|devolvio)\b", texto_norm):
        return None
    es_pregunta_responsable = bool(
        re.search(
            r"\b(?:quien(?:es)?\s+(?:es\s+el\s+responsable|son\s+los\s+responsables|tiene(?:n)?(?:\s+prestado)?|lleva)|"
            r"a\s+quien(?:es)?\s+(?:se\s+(?:le\s+)?prest[oa]|le\s+prestaron|esta\s+prestado))\b",
            texto_norm,
        )
        and re.search(r"\b(?:prestam\w*|prestad\w*|presto|prestaron|tiene(?:n)?)\b", texto_norm)
    )
    if not es_pregunta_responsable:
        return None
    return PlanAgente(
        tipo="consulta",
        operacion="detalle",
        filtros=FiltrosConsulta(codigos=codigos),
        referencia="ninguna",
        alcance="actual",
        campos=["prestamos"],
        alcance_entidad="producto",
        fuente_plan="fast_path_responsable_prestamo_codigo",
    )


def _planificar_responsable_prestamo_producto(texto: str) -> Optional[PlanAgente]:
    """Resuelve consultas por el responsable de préstamos activos de un producto nombrado sin código."""
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip().strip("¿?¡!.,;: ")
    texto_norm = normalizar(texto_pre)
    if not texto_norm or extraer_codigos_de_texto(texto_pre):
        return None
    if _mensaje_requiere_referencia_contextual(texto_pre) or tiene_referencia_resultados_previos(texto_pre):
        return None
    if re.search(r"\b(?:ultimo|ultima|historial|movimientos?|saco|retiro|devolvio|ayer|hoy|semana|mes|anio|ano|almacen|bodega|categoria|cuant[oa]s?)\b", texto_norm):
        return None
    m = re.match(
        r"^(?:qui[eé]n(?:es)?\s+(?:tiene(?:n)?(?:\s+prestado[sa]?s?)?|es\s+el\s+responsable\s+(?:del\s+)?pr[eé]stamo\s+(?:de|del)|son\s+los\s+responsables\s+(?:del\s+)?pr[eé]stamo\s+(?:de|del))|"
        r"a\s+qui[eé]n(?:es)?\s+(?:se\s+(?:le\s+)?prest[oó]|le(?:s)?\s+prestaron|est[aá]\s+prestad[oa]))\s+(.+)$",
        texto_pre,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    objeto = m.group(1).strip().strip("?!.¡¿,; ")
    objeto = re.sub(r"^(?:el|la|los|las|un|una|unos|unas|de|del)\s+", "", objeto, flags=re.IGNORECASE).strip()
    objeto = re.sub(r"\s+(?:en\s+pr[eé]stamo|prestad[oa]s?|ahora\s+mismo|actualmente|en\s+este\s+momento)$", "", objeto, flags=re.IGNORECASE).strip()
    obj_norm = normalizar(objeto)
    genericos = {
        "producto", "productos", "articulo", "articulos", "equipo", "equipos", "cosa", "cosas",
        "herramienta", "herramientas", "material", "materiales", "elemento", "elementos",
        "prestamo", "prestamos", "lo", "la", "los", "las", "ese", "esa", "este", "esta", "algo", "alguien",
    }
    if not obj_norm or obj_norm in genericos or len(obj_norm.split()) > 6:
        return None
    return PlanAgente(
        tipo="consulta",
        operacion="detalle",
        filtros=FiltrosConsulta(texto=objeto),
        referencia="ninguna",
        alcance="actual",
        campos=["prestamos"],
        alcance_entidad="producto",
        fuente_plan="fast_path_responsable_prestamo_producto",
    )


def _planificar_comparacion_ubicaciones_explicita(texto: str) -> Optional[PlanAgente]:
    """Resuelve comparaciones directas entre dos ubicaciones registradas en el catálogo vivo."""
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    texto_norm = normalizar(texto_pre)
    if not texto_norm or not re.search(r"\bcompar(?:a|ar|ame|en|emos)\b", texto_norm):
        return None
    if extraer_codigos_de_texto(texto_pre):
        return None
    nombres = _nombres_ubicaciones_registradas()
    if not nombres:
        return None
    ubicaciones = _resolver_menciones_ubicacion(texto_pre, nombres)
    if len(ubicaciones) != 2:
        return None
    ubicaciones.sort(
        key=lambda nom: texto_norm.find(normalizar(nom)) if normalizar(nom) in texto_norm else len(texto_norm)
    )
    restante = texto_pre
    for u in ubicaciones:
        restante = re.sub(re.escape(u), " ", restante, flags=re.IGNORECASE)
    restante = re.sub(
        r"\b(?:compar(?:a|ar|ame|en|emos)|entre|con|vs|versus|y|el|la|los|las|ubicaci[oó]n(?:es)?)\b",
        " ",
        restante,
        flags=re.IGNORECASE,
    )
    if _extraer_objeto_consulta_sin_marco(restante):
        return None
    return PlanAgente(
        tipo="consulta",
        operacion="comparar",
        ubicaciones_comparadas=ubicaciones,
        referencia="ninguna",
        campos=[],
        fuente_plan="fast_path_comparacion_ubicaciones",
    )


def _planificar_operacion_anaforica_foco(texto: str, estado: Any) -> Optional[PlanAgente]:
    """Resuelve operaciones explícitas con anáfora clara sobre un foco único («Ingresa 5 de ese», «Saca 3 de esos para Juan»)."""
    if estado is None or not texto:
        return None
    if "?" in texto or "¿" in texto:
        return None
    codigo_foco = getattr(estado, "producto_codigo", None)
    if not codigo_foco:
        coleccion = getattr(estado, "coleccion_productos", None) or []
        if len(coleccion) == 1:
            codigo_foco = coleccion[0].get("codigo")
    if not codigo_foco:
        return None
    from ..lenguaje.observaciones import separar_observacion_operacion
    texto_sin_obs_af, _ = separar_observacion_operacion(texto or "")
    texto_pre = preprocesar_texto_inventario(texto_sin_obs_af or texto or "", normalizar_medidas=False).strip()
    if extraer_codigos_de_texto(texto_pre) or not tiene_referencia_contextual_operacion(texto_pre):
        return None
    texto_norm = normalizar(texto_pre)
    if re.search(
        r"\b(?:cuant[oa]s?|donde|ubicacion|stock|existencias?|disponibles?|movimientos?|historial|prestamos?\s+activos?)\b",
        texto_norm,
    ):
        return None

    es_entrada = bool(re.search(r"\b(?:ingres[aeo]\w*|met[aeo]\w*|entrada\s+de)\b", texto_norm))
    es_salida = bool(re.search(r"\b(?:sacar|sac[aeo]\w*|retir[aeo]\w*|despach[aeo]\w*|salida\s+de|dotacion|entrega\s+definitiva)\b", texto_norm))
    es_prestamo = bool(re.search(r"\b(?:prestar|prest[aeo]\w*|prestamo\s+de)\b", texto_norm))
    if sum((es_entrada, es_salida, es_prestamo)) != 1:
        return None

    destinatario = None
    m_dest = re.search(
        r"\b(?:para|a\s+nombre\s+de|a)\s+([A-Za-zÁ-ÿñÑ][A-Za-zÁ-ÿñÑ\s'-]{1,40}?)(?:[\.,;!?]|$)",
        texto_pre,
        flags=re.IGNORECASE,
    )
    if m_dest and not es_entrada:
        cand_d = m_dest.group(1).strip()
        if cand_d and not re.search(r"\b(?:almac[eé]n|bodega|inventario|unidades?|piezas?|ese|esa|esos|esas|mismo|misma)\b", cand_d, re.IGNORECASE):
            destinatario = cand_d

    texto_cant = re.sub(r"\b(?:para|a\s+nombre\s+de|a)\s+.+$", " ", texto_norm) if destinatario else texto_norm
    from ..lenguaje.numeros import extraer_cantidad_operacion_texto
    cantidad = extraer_cantidad_operacion_texto(texto_cant)

    accion = "entrada" if es_entrada else ("salida" if es_salida else "prestamo")
    return PlanAgente(
        tipo="operacion",
        accion=accion,
        producto=codigo_foco,
        filtros=FiltrosConsulta(codigos=[codigo_foco]),
        cantidad=cantidad,
        destinatario=destinatario,
        referencia="foco",
        fuente_plan="fast_path_operacion_anafora_foco",
    )


def _planificar_operacion_sin_producto_ni_anafora(texto: str) -> Optional[PlanAgente]:
    """Enruta deterministamente aperturas operativas sin producto ni anáfora para evitar fuga de foco."""
    if not texto or "?" in texto or "¿" in texto:
        return None
    texto_pre = preprocesar_texto_inventario(texto or "", normalizar_medidas=False).strip()
    if extraer_codigos_de_texto(texto_pre) or tiene_referencia_contextual_operacion(texto_pre):
        return None
    texto_norm = normalizar(texto_pre).strip(" .,:;!¡?¿")
    m_op = re.fullmatch(
        r"(?:(?:quiero|necesito|deseo|voy\s+a|quisiera|por\s+favor)\s+)*"
        r"(?:hacer|registrar|generar|crear|realizar|anotar|dar)?\s*"
        r"(?:un|una|el|la)?\s*"
        r"(prestamo|devolucion|entrada|salida|traslado)(?:\s+(?:de\s+stock|de\s+inventario|al\s+inventario|del\s+inventario))?",
        texto_norm,
    )
    if not m_op:
        return None
    return PlanAgente(
        tipo="operacion",
        accion=m_op.group(1),
        referencia="ninguna",
        fuente_plan="fast_path_operacion_sin_producto",
    )


def planificar_determinista(
    texto: str,
    estado: Any = None,
    categorias_accesibles: Optional[List[str]] = None,
    almacenes_accesibles: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Resuelve sin consumir inferencia del Cerebro Central los casos 100% inequívocos:
    1. Capacidades o funciones de HAKU («¿Qué puedes hacer?», «ayuda»).
    2. Preguntas de conocimiento general ajenas al inventario («¿Qué es el sol?»).
    3. Código de producto directo o solicitud explícita de ficha de código (copia el código literal que menciona el usuario).
    4. Preguntas explícitas por la ubicación física o almacén de un producto nombrado literalmente.
    5. Preguntas explícitas por stock o disponibilidad de un producto nombrado literalmente.
    6. Condiciones globales explícitas de stock (agotados, bajo mínimo, mínimo en cero).
    7. Selección ordinal sobre candidatos o colección activa («el primero», «la segunda», «opción 1», «el último»).
    8. Cambio explícito de stock mínimo con código y valor presentes en el mensaje.
    9. Consulta del stock mínimo sobre un foco conversacional único.
    """
    if not texto or not texto.strip():
        return None
    texto_pre = preprocesar_texto_inventario(texto).strip()
    txt_norm = normalizar(texto_pre)

    # 1. Capacidades o funciones generales del agente
    if _es_pregunta_capacidades_agente(texto_pre) or re.search(
        r"^(?:qu[eé]\s+(?:puedes\s+hacer|haces|sabes\s+hacer|funciones\s+tienes)|"
        r"en\s+qu[eé]\s+(?:me\s+)?puedes\s+ayudar|cu[aá]les\s+son\s+tus\s+funciones|"
        r"para\s+qu[eé]\s+sirves|ayuda(?:\s+por\s+favor)?|qu[eé]\s+es\s+ha[ck]u)\b",
        txt_norm,
    ):
        return PlanAgente(
            tipo="capacidades",
            referencia="ninguna",
            fuente_plan="fast_path_capacidades",
        )

    # 2. Fuera de dominio / conocimiento general no inventariable
    if _es_pregunta_conceptual_fuera_dominio(texto_pre, txt_norm) or (
        not any(p in _PALABRAS_DOMINIO_INVENTARIO for p in txt_norm.split())
        and not extraer_codigos_de_texto(texto_pre)
        and re.match(
            r"^(?:qui[eé]n\s+(?:es|fue)\b|cu[aá]l\s+es\s+la\s+capital\b|c[oó]mo\s+est[aá]\s+el\s+clima\b|qu[eé]\s+hora\s+es\b)",
            txt_norm,
        )
    ):
        return PlanAgente(
            tipo="fuera_dominio",
            referencia="ninguna",
            fuente_plan="fast_path_fuera_dominio",
        )

    # 3. Código directo aislado o «ficha/detalle de COD-123»
    cod_directo = normalizar_codigo_producto(texto_pre)
    if not cod_directo:
        m_ficha_cod = re.match(
            r"^(?:ficha|detalle|datos|consultar?)\s+(?:de(?:l)?\s+)?(?:producto\s+)?([a-z]{2,5}[\s\-_]?\d{1,5})$",
            txt_norm,
        )
        if m_ficha_cod:
            cod_directo = normalizar_codigo_producto(m_ficha_cod.group(1))
    if cod_directo:
        return PlanAgente(
            tipo="consulta",
            operacion="detalle",
            filtros=FiltrosConsulta(codigos=[cod_directo]),
            referencia="ninguna",
            campos=["ficha"],
            fuente_plan="fast_path_codigo",
        )

    plan_cambio_minimo = _planificar_cambio_stock_minimo_explicito(texto_pre)
    if plan_cambio_minimo is not None:
        return plan_cambio_minimo
    plan_historial_ordinal = _planificar_historial_ordinal(texto_pre, estado)
    if plan_historial_ordinal is not None:
        return plan_historial_ordinal
    coleccion_foco = getattr(estado, "coleccion_productos", None) or []
    if coleccion_foco and re.fullmatch(
        r"(?:y )?cuantos(?: productos| articulos)? (?:son|habia en (?:esa|la) lista|hay en (?:esa|la) lista)",
        txt_norm,
    ):
        return PlanAgente(
            tipo="consulta", operacion="contar", referencia="resultados_previos",
            fuente_plan="fast_path_conteo_anterior",
        )
    tiene_foco_unico = bool(getattr(estado, "producto_codigo", None)) or (
        len(coleccion_foco) == 1 and bool(coleccion_foco[0].get("codigo"))
    )
    ubicacion_eliptica_completa = bool(
        re.search(r"\b(?:guardad[oa]s?|ubicad[oa]s?|almacenad[oa]s?|situad[oa]s?)\b", txt_norm)
    )
    if _es_consulta_ubicacion_eliptica(texto_pre) and (tiene_foco_unico or ubicacion_eliptica_completa):
        plan_lugar = PlanAgente(
            tipo="consulta",
            operacion="detalle",
            referencia="foco",
            fuente_plan="fast_path_ubicacion_foco",
        )
        plan_lugar = _normalizar_peticion_atributo(texto_pre, plan_lugar)
        if not plan_lugar.campos:
            plan_lugar.campos = ["ubicacion", "almacen"]
        return plan_lugar

    plan_atributo_foco = _planificar_atributo_foco_eliptico(texto_pre, tiene_foco_unico)
    if plan_atributo_foco is not None:
        return plan_atributo_foco

    # 4. Consulta de ubicación con producto literal. Evita gastar una inferencia para
    # extraer una entidad que ya se puede enviar como filtro de texto a la BD.
    plan_ubicacion = _planificar_consulta_ubicacion_explicita(texto)
    if plan_ubicacion is not None:
        return plan_ubicacion

    # 5. Cantidad/stock explícitos para un producto literal, sin inferir nombres del catálogo.
    plan_stock = _planificar_consulta_stock_explicita(texto)
    if plan_stock is not None:
        return plan_stock

    # 6. Priorización global de reposición por faltantes/riesgo.
    plan_prioridad = _planificar_prioridad_reposicion(texto_pre)
    if plan_prioridad is not None:
        return plan_prioridad

    # 7. Condiciones globales de stock con vocabulario explícito.
    plan_condicion_stock = _planificar_condicion_stock_explicita(
        texto_pre,
        categorias_accesibles=categorias_accesibles,
        almacenes_accesibles=almacenes_accesibles,
    )
    if plan_condicion_stock is not None:
        return plan_condicion_stock

    # 7b. Consultas globales explícitas de movimientos/eventos por período.
    plan_eventos_globales = _planificar_eventos_globales_explicitos(
        texto_pre,
        categorias_accesibles=categorias_accesibles,
        almacenes_accesibles=almacenes_accesibles,
    )
    if plan_eventos_globales is not None:
        return plan_eventos_globales

    # 7c. Consultas globales explícitas de préstamos activos/vigentes (sin fuga de foco).
    plan_prestamos_globales = _planificar_prestamos_activos_globales(
        texto_pre,
        categorias_accesibles=categorias_accesibles,
        almacenes_accesibles=almacenes_accesibles,
    )
    if plan_prestamos_globales is not None:
        return plan_prestamos_globales

    # 7d. Elegibilidad y responsables sobre código explícito, comparaciones de ubicaciones, categorías, listado general y alta sin nombre.
    plan_elegibilidad_cod = _planificar_elegibilidad_codigo_explicito(texto_pre)
    if plan_elegibilidad_cod is not None:
        return plan_elegibilidad_cod

    plan_resp_prestamo = _planificar_responsable_prestamo_codigo(texto_pre)
    if plan_resp_prestamo is not None:
        return plan_resp_prestamo

    plan_resp_prestamo_prod = _planificar_responsable_prestamo_producto(texto_pre)
    if plan_resp_prestamo_prod is not None:
        return plan_resp_prestamo_prod

    plan_comp_ubic = _planificar_comparacion_ubicaciones_explicita(texto_pre)
    if plan_comp_ubic is not None:
        return plan_comp_ubic

    plan_categorias_globales = _planificar_categorias_globales(texto_pre)
    if plan_categorias_globales is not None:
        return plan_categorias_globales

    plan_listado_global = _planificar_listado_global_explicito(texto_pre)
    if plan_listado_global is not None:
        return plan_listado_global

    plan_almacen_global = _planificar_consulta_almacen_global(
        texto_pre,
        almacenes_accesibles=almacenes_accesibles,
    )
    if plan_almacen_global is not None:
        return plan_almacen_global

    plan_devolucion = _planificar_devolucion_determinista(texto_pre)
    if plan_devolucion is not None:
        return plan_devolucion

    plan_mov_codigo = _planificar_movimiento_explicito_codigo(texto_pre)
    if plan_mov_codigo is not None:
        return plan_mov_codigo

    plan_op_anafora = _planificar_operacion_anaforica_foco(texto_pre, estado)
    if plan_op_anafora is not None:
        return plan_op_anafora

    plan_op_sin_prod = _planificar_operacion_sin_producto_ni_anafora(texto_pre)
    if plan_op_sin_prod is not None:
        return plan_op_sin_prod

    plan_crear_sin_nombre = _planificar_crear_producto_sin_nombre(texto_pre)
    if plan_crear_sin_nombre is not None:
        return plan_crear_sin_nombre

    # 7. Frases incompletas o truncadas que terminan en preposición o puntos suspensivos
    # Ej. «Haku, por favor, búscame en el almacén de…», «ahora en el almacén de...»
    if re.search(
        r"(?:(?:\ben\s+(?:el\s+)?almac[eé]n\s+de|almac[eé]n\s+de|en\s+la\s+bodega\s+de)\s*[.…]*$|"
        r"\b(?:en|de|del|al|para|por|con|sobre)\s*[.…]+$)",
        texto_pre,
        flags=re.IGNORECASE,
    ):
        return PlanAgente(
            tipo="aclaracion",
            referencia="ninguna",
            pregunta="Por favor indique en qué almacén desea consultar.",
            fuente_plan="fast_path_frase_incompleta",
        )

    # Solicitud explícita de cambiar de producto (no debe reutilizar el foco anterior)
    if re.match(
        r"^(?:busca[r]?|mu[eé]stra(?:me)?|quiero\s+(?:ver|buscar|consultar))\s+otro\s+producto\b",
        txt_norm,
    ):
        return PlanAgente(
            tipo="aclaracion",
            referencia="ninguna",
            pregunta="¿Qué producto desea buscar?",
            fuente_plan="fast_path_otro_producto",
        )

    # Referencia anafórica con almacén ambiguo («Ese mismo, en otro almacén»)
    if re.match(
        r"^(?:ese\s+mismo|este\s+mismo)(?:,\s*|\s+)en\s+otro\s+almac[eé]n\b",
        txt_norm,
    ):
        return PlanAgente(
            tipo="aclaracion",
            referencia="foco",
            pregunta="¿En cuál otro almacén desea consultar?",
            fuente_plan="fast_path_otro_almacen_ambiguo",
        )

    # Pregunta elíptica con almacén específico sobre foco previo («¿Y en el almacén B?», «¿en almacén B?»)
    m_alm_eliptico = re.match(
        r"^(?:[¿?]?(?:y\s+)?(?:ahora\s+)?en\s+(?:el\s+)?(?:almac[eé]n\s+|bodega\s+)?([a-zA-Z0-9]+)[?]?)$",
        txt_norm,
    )
    if m_alm_eliptico:
        nom_alm = m_alm_eliptico.group(1).upper()
        alm_can = f"Almacén {nom_alm}" if len(nom_alm) == 1 else nom_alm
        return PlanAgente(
            tipo="consulta",
            operacion="detalle",
            filtros=FiltrosConsulta(almacen=alm_can),
            referencia="foco",
            campos=["stock"],
            fuente_plan="fast_path_almacen_eliptico",
        )

    # Selección ordinal sobre candidatos o colección previa en contexto
    if estado is not None:
        opciones = getattr(estado, "candidatos", None) or getattr(estado, "coleccion_productos", None) or []
        if opciones:
            idx: Optional[int] = None
            if (
                re.match(r"^(?:el\s+|la\s+)?(?:primer[oa](?:\s+opci[oó]n)?|opci[oó]n\s+1|1)$", txt_norm)
                or re.search(r"\b(?:cual|que)\b.{0,36}\bprimer[oa]\b", txt_norm)
            ):
                idx = 0
            elif (
                re.match(r"^(?:el\s+|la\s+)?(?:segund[oa](?:\s+opci[oó]n)?|opci[oó]n\s+2|2)$", txt_norm)
                or re.search(r"\b(?:cual|que)\b.{0,36}\bsegund[oa]\b", txt_norm)
            ) and len(opciones) >= 2:
                idx = 1
            elif (
                re.match(r"^(?:el\s+|la\s+)?(?:tercer[oa](?:\s+opci[oó]n)?|opci[oó]n\s+3|3)$", txt_norm)
                or re.search(r"\b(?:cual|que)\b.{0,36}\btercer[oa]\b", txt_norm)
                or (
                    tiene_referencia_resultados_previos(texto_pre)
                    and re.search(r"\btercer[oa]\b", txt_norm)
                )
            ) and len(opciones) >= 3:
                idx = 2
            elif (
                re.match(r"^(?:el\s+|la\s+)?(?:[uú]ltim[oa](?:\s+opci[oó]n)?)$", txt_norm)
                or re.search(r"\b(?:cual|que)\b.{0,36}\b[uú]ltim[oa]\b", txt_norm)
            ):
                idx = len(opciones) - 1
            if idx is not None and 0 <= idx < len(opciones):
                elegido = opciones[idx]
                cod_elegido = elegido.get("codigo")
                if cod_elegido:
                    datos_pendientes = getattr(estado, "datos_intencion_pendiente", None) or {}
                    plan_pendiente = (
                        datos_pendientes.get("plan_consulta", {})
                        if isinstance(datos_pendientes, dict)
                        else {}
                    )
                    if not isinstance(plan_pendiente, dict):
                        plan_pendiente = {}
                    operacion_pendiente = plan_pendiente.get("operacion")
                    if operacion_pendiente not in OPERACIONES_VALIDAS:
                        operacion_pendiente = "detalle"
                    campos_pendientes = plan_pendiente.get("campos")
                    if not isinstance(campos_pendientes, list):
                        campos_pendientes = ["ficha"] if operacion_pendiente == "detalle" else []
                    campos_pendientes = [campo for campo in campos_pendientes if campo in CAMPOS_VALIDOS]
                    medida_pendiente = plan_pendiente.get("medida_solicitada")
                    if medida_pendiente not in MEDIDAS_SOLICITADAS_VALIDAS:
                        medida_pendiente = None
                    referencia = "resultados_previos" if getattr(estado, "coleccion_productos", None) and not getattr(estado, "candidatos", None) else "ninguna"
                    return PlanAgente(
                        tipo="consulta",
                        operacion=operacion_pendiente,
                        filtros=FiltrosConsulta(
                            codigos=[cod_elegido],
                            categoria=plan_pendiente.get("categoria"),
                            almacen=plan_pendiente.get("almacen"),
                            ubicacion=plan_pendiente.get("ubicacion"),
                            condicion=plan_pendiente.get("condicion"),
                            umbral=plan_pendiente.get("umbral"),
                        ),
                        referencia=referencia,
                        alcance=plan_pendiente.get("alcance"),
                        tipo_movimiento=plan_pendiente.get("tipo_movimiento"),
                        campos=campos_pendientes,
                        medida_solicitada=medida_pendiente,
                        alcance_entidad="producto",
                        fuente_plan="fast_path_ordinal",
                    )
    estado_tiene_contexto = bool(
        getattr(estado, "producto_codigo", None)
        or getattr(estado, "candidatos", None)
        or getattr(estado, "coleccion_productos", None)
        or getattr(estado, "operacion_pendiente", None)
        or getattr(estado, "propuesta_activa", None)
    )
    if not estado_tiene_contexto and re.fullmatch(
        r"(?:(?:hacu|jacu|oye|por\s+favor)\s+)?(?:que|cuales)\s+(?:(?:productos?|articulos?|referencias?|items?|equipos?|herramientas?)\s+)?no\s+(?:son\s+prestables?|son\s+retornables?|se\s+pueden?\s+prestar)",
        txt_norm,
    ):
        return PlanAgente(
            tipo="consulta",
            operacion="listar",
            filtros=FiltrosConsulta(es_prestable=False),
            referencia="ninguna",
            campos=["es_prestable"],
            alcance_entidad="coleccion",
            alcance_global_confirmado=True,
            fuente_plan="fast_path_elegibilidad_negada",
        )
    if not estado_tiene_contexto and re.fullmatch(
        r"(?:(?:hacu|jacu|oye|por\s+favor)\s+)?(?:que|cuales)\s+(?:(?:productos?|articulos?|referencias?|items?|equipos?|herramientas?)\s+)?(?:son\s+prestables?|son\s+retornables?|se\s+pueden?\s+prestar)",
        txt_norm,
    ):
        return PlanAgente(
            tipo="consulta",
            operacion="listar",
            filtros=FiltrosConsulta(es_prestable=True),
            referencia="ninguna",
            campos=["es_prestable"],
            alcance_entidad="coleccion",
            alcance_global_confirmado=True,
            fuente_plan="fast_path_elegibilidad_positiva",
        )
    tokens_mensaje = txt_norm.split()
    if 1 <= len(tokens_mensaje) <= 5:
        nucleo = re.sub(
            r"^(?:(?:hacu|jacu|oye|por\s+favor)\s+)?(?:(?:ver|mostrar|muestrame|dame|dime|consultar|revisar|listar|listame|pasame|buscar|buscame|ensename|cuales\s+son)\s+)?(?:(?:todo\s+el|todos\s+los|todas\s+las|el|la|los|las|un|una|unos|unas)\s+)?",
            "",
            txt_norm,
        ).strip()
        fuente_corto = "fast_path_comando_unico_dominio" if len(tokens_mensaje) == 1 else "fast_path_comando_corto_dominio"
        if nucleo in ("agotado", "agotados", "agotada", "agotadas", "sin stock", "sin existencias", "sin unidades", "algo agotado", "productos agotados"):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                filtros=FiltrosConsulta(condicion="agotado"),
                referencia="ninguna",
                campos=["stock"],
                alcance_entidad="coleccion",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("bajo stock", "stock bajo", "poco stock", "bajo minimo", "bajo el minimo", "productos bajo stock", "productos con bajo stock"):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                filtros=FiltrosConsulta(condicion="bajo_minimo"),
                referencia="ninguna",
                campos=["stock", "stock_minimo"],
                alcance_entidad="coleccion",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("alerta", "alertas", "alertas de stock", "faltante", "faltantes", "reposicion", "reabastecimiento", "que falta", "que hace falta", "que falta por pedir", "que hace falta por pedir"):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                filtros=FiltrosConsulta(condicion="requiere_reposicion"),
                referencia="ninguna",
                campos=["stock", "stock_minimo"],
                alcance_entidad="coleccion",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("categoria", "categorias"):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                referencia="ninguna",
                campos=["categoria"],
                alcance_entidad="global",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("prestamo", "prestamos", "prestado", "prestados", "prestamos activos"):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                filtros=FiltrosConsulta(condicion="con_prestamos"),
                referencia="ninguna",
                alcance="actual",
                campos=["prestamos"],
                medida_solicitada="ninguna",
                alcance_entidad="coleccion",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("prestable", "prestables", "retornable", "retornables", "productos prestables"):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                filtros=FiltrosConsulta(es_prestable=True),
                referencia="ninguna",
                campos=["es_prestable"],
                alcance_entidad="coleccion",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("consumible", "consumibles", "productos consumibles", "no prestable", "no prestables", "productos no prestables", "no retornable", "no retornables"):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                filtros=FiltrosConsulta(es_prestable=False),
                referencia="ninguna",
                campos=["es_prestable"],
                alcance_entidad="coleccion",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("almacen", "almacenes", "bodega", "bodegas"):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                referencia="ninguna",
                campos=["almacen"],
                alcance_entidad="global",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("ubicaciones", "estantes") or (nucleo in ("ubicacion", "estante") and not estado_tiene_contexto):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                referencia="ninguna",
                campos=["ubicacion"],
                alcance_entidad="global",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        _mapa_mov_corto = {
            "movimiento": None, "movimientos": None, "historial": None, "actividad": None,
            "entrada": "ENTRADA", "entradas": "ENTRADA",
            "salida": "SALIDA", "salidas": "SALIDA",
            "traslado": "TRASLADO", "traslados": "TRASLADO",
            "devolucion": "DEVOLUCION", "devoluciones": "DEVOLUCION",
        }
        if nucleo in _mapa_mov_corto:
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                referencia="ninguna",
                alcance="evento",
                tipo_movimiento=_mapa_mov_corto[nucleo],
                campos=["movimientos"],
                alcance_entidad="global",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("stock disponible", "existencia", "existencias", "disponible", "disponibles", "productos disponibles", "productos en stock", "productos con stock", "que stock hay") or (nucleo == "stock" and (txt_norm != "el stock" or not estado_tiene_contexto)):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                filtros=FiltrosConsulta(condicion="con_stock"),
                referencia="ninguna",
                campos=["stock"],
                alcance_entidad="coleccion",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if nucleo in ("producto", "productos", "inventario", "catalogo", "articulo", "articulos", "referencia", "referencias", "item", "items", "resumen", "resumen del inventario", "estado del inventario") or (not estado_tiene_contexto and txt_norm in ("que hay", "que tenemos", "hay algo", "que existe")):
            return PlanAgente(
                tipo="consulta",
                operacion="listar",
                referencia="ninguna",
                campos=[],
                alcance_entidad="global",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if not estado_tiene_contexto and txt_norm in ("cuantos hay", "cuantos productos", "cuantos articulos"):
            return PlanAgente(
                tipo="consulta",
                operacion="contar",
                referencia="ninguna",
                campos=[],
                alcance_entidad="global",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
        if not estado_tiene_contexto and txt_norm in ("cuanto stock hay", "cuantas unidades hay"):
            return PlanAgente(
                tipo="consulta",
                operacion="sumar_unidades",
                referencia="ninguna",
                campos=["stock"],
                medida_solicitada="disponible",
                alcance_entidad="global",
                alcance_global_confirmado=True,
                fuente_plan=fuente_corto,
            )
    if len(tokens_mensaje) == 1 and not estado_tiene_contexto:
        return PlanAgente(
            tipo="aclaracion",
            referencia="ninguna",
            pregunta="Disculpe, no entendí la solicitud. ¿Podría formularla de nuevo e indicar qué consulta u operación de inventario necesita?",
            fuente_plan="fast_path_solicitud_incompleta",
        )
    return None


# Palabras de frontera sintáctica que delimitan el final de un sintagma nominal de producto en una consulta.
_FRONTERAS_SINTAGMA = {
    "tenemos", "tienen", "tiene", "tienes", "hay", "queda", "quedan", "quedaron", "estan", "esta",
    "son", "es", "habra", "existen", "existe", "manejamos", "manejan", "contamos", "usamos", "vienen",
    "salieron", "entraron", "prestaron", "presto", "llevaron", "llevo", "faltan", "falta", "acabaron",
    "acabo", "agotaron", "agoto", "sirve", "sirven", "sirva", "sirvan", "usa", "usan",
    "sacar", "retirar", "entregar", "despachar", "prestar", "usar", "llevar", "devolver", "ingresar", "trasladar",
    "busca", "buscar", "buscame", "encuentra", "localiza", "consulta", "consultar", "ver", "revisa", "revisar", "muestra", "muestrame", "dame",
    "disponible", "disponibles", "agotado", "agotada", "agotados", "agotadas", "escaso", "escasa",
    "escasos", "escasas", "registrado", "registrada", "registrados", "registradas", "activo", "activos",
    "activa", "activas", "inactivo", "inactivos", "prestable", "prestables", "prestado", "prestada",
    "prestados", "prestadas", "almacen", "almacenes", "bodega", "bodegas", "estante", "estanteria",
    "ubicacion", "inventario", "catalogo", "sistema", "stock", "existencia", "existencias", "disponibilidad", "minimo",
    "maximo", "total", "actualmente", "ahora", "hoy", "ayer", "que", "cual", "cuales", "donde",
    "cuando", "como", "cuanto", "cuantos", "cuantas", "quien", "quienes", "porque", "pero", "sino",
    "o", "u", "y", "por", "a", "al", "esos", "ellos", "ellas", "este", "esta", "estos", "estas",
    "ese", "esa", "aquel", "aquella", "cero", "unidades", "unidad", "cajas", "caja", "paquetes", "paquete",
    "ubicado", "ubicada", "ubicados", "ubicadas", "guardado", "guardada", "guardados", "guardadas",
    "nos", "me", "te", "se", "le", "les", "lo", "mismo",
}
_ARTICULOS_SINTAGMA = {"el", "la", "los", "las", "un", "una", "unos", "unas", "mi", "tu", "su", "mis", "tus", "sus", "nuestro", "nuestra"}


def _preservar_sintagma_nominal(texto_filtro: str, texto_pre: str, categoria_raw: Optional[str] = None) -> str:
    """Preserva de forma general los modificadores adjetivales, alfanuméricos o preposicionales
    (sustantivo + adjetivo / sustantivo + de/en/para/con/sin + sustantivo) que siguen inmediatamente
    al sustantivo núcleo en el mensaje del usuario cuando el LLM los recorta."""
    if not texto_filtro or not texto_pre:
        return texto_filtro
    txt_norm = normalizar(texto_pre)
    # Extracción sintáctica general de modificadores contiguos al sustantivo núcleo en el mensaje.
    # No se enumeran adjetivos ni nombres de productos: se preserva la estructura del sintagma.
    palabras_filtro_originales = re.findall(r"[\w]+(?:[-/][\w]+)*", str(texto_filtro), flags=re.UNICODE)
    palabras_usr_originales = re.findall(r"[\w]+(?:[-/][\w]+)*", str(texto_pre), flags=re.UNICODE)
    palabras_filtro = normalizar(texto_filtro).split()
    palabras_usr = txt_norm.split()
    if not palabras_filtro or not palabras_usr:
        return texto_filtro

    fronteras = set(_FRONTERAS_SINTAGMA)
    if categoria_raw:
        for cw in normalizar(categoria_raw).split():
            if len(cw) >= 3:
                fronteras.add(cw)

    def coincide(token_filtro: str, token_mensaje: str) -> bool:
        raiz_filtro = _raiz(token_filtro)
        raiz_mensaje = _raiz(token_mensaje)
        if raiz_filtro == raiz_mensaje:
            return True
        sinonimos = {_raiz(valor) for valor in _sinonimos_de(token_filtro)}
        return raiz_mensaje in sinonimos

    # Alinear los tokens del filtro con el sintagma que realmente dijo la
    # persona. Conservar el intervalo entero recupera calificadores omitidos
    # por el modelo incluso cuando aparecen antes del último token coincidente
    # (por ejemplo, una resistencia de cierto valor o un cable de cierta
    # longitud). Los nombres y valores siguen procediendo del mensaje y del
    # catálogo activo.
    posiciones: List[int] = []
    cursor = 0
    for token in palabras_filtro:
        indice = next(
            (i for i in range(cursor, len(palabras_usr)) if coincide(token, palabras_usr[i])),
            None,
        )
        if indice is None:
            posiciones = []
            break
        posiciones.append(indice)
        cursor = indice + 1

    if posiciones:
        inicio, fin = posiciones[0], posiciones[-1]
        # Recupera el núcleo nominal que el plan pudo recortar al principio,
        # pero detente en el marco de la pregunta o en un límite sintáctico.
        while inicio > 0:
            previo = palabras_usr[inicio - 1]
            if previo in fronteras or previo in _ARTICULOS_SINTAGMA or previo in {"de", "del", "al"}:
                break
            if re.match(r"^[a-z]{2,5}-\d+$", previo):
                break
            inicio -= 1

        # Extiende el sintagma a modificadores contiguos posteriores. Las
        # preposiciones con artículo o un sustantivo de marco delimitan la
        # entidad; «a + nombre» se conserva para construcciones nominales como
        # conectores con dos extremos.
        while fin + 1 < len(palabras_usr):
            siguiente = palabras_usr[fin + 1]
            if siguiente in fronteras:
                if (
                    siguiente == "a"
                    and fin + 2 < len(palabras_usr)
                    and palabras_usr[fin + 2] not in fronteras | _ARTICULOS_SINTAGMA
                ):
                    fin += 2
                    continue
                break
            if siguiente in _ARTICULOS_SINTAGMA or re.match(r"^[a-z]{2,5}-\d+$", siguiente):
                break
            if siguiente in {"de", "del", "en", "para", "con", "sin"}:
                if fin + 2 >= len(palabras_usr):
                    break
                despues = palabras_usr[fin + 2]
                if despues in _ARTICULOS_SINTAGMA or despues in fronteras:
                    break
            fin += 1
        segmento = palabras_usr_originales[inicio : fin + 1]
        for posicion, token_original in zip(posiciones, palabras_filtro_originales):
            segmento[posicion - inicio] = token_original
        segmento = [
            token for token in segmento
            if normalizar(token) not in _PALABRAS_VACIAS
        ]
        return " ".join(segmento)

    # Compatibilidad con paráfrasis que el modelo haya expresado mediante un
    # sinónimo no presente literalmente en el turno: conserva el camino previo
    # y añade solo los modificadores pospuestos al núcleo reconocido.
    ult_raiz = _raiz(palabras_filtro[-1])
    sinonimos_ult = {_raiz(valor) for valor in _sinonimos_de(palabras_filtro[-1])}
    idx_nucleo = next(
        (i for i, palabra in enumerate(palabras_usr)
         if _raiz(palabra) == ult_raiz or _raiz(palabra) in sinonimos_ult),
        -1,
    )
    if idx_nucleo < 0:
        return texto_filtro
    existentes = set(normalizar(texto_filtro).split())
    agregados: List[str] = []
    j = idx_nucleo + 1
    while j < len(palabras_usr) and len(agregados) < 4:
        w = palabras_usr[j]
        if w in fronteras or re.match(r"^[a-z]{2,5}-\d+$", w):
            break
        if w in ("de", "del", "en", "para", "con", "sin"):
            if j + 1 >= len(palabras_usr):
                break
            sig = palabras_usr[j + 1]
            if sig in _ARTICULOS_SINTAGMA or sig in fronteras or re.match(r"^[a-z]{2,5}-\d+$", sig):
                break
            agregados.extend([w, sig])
            j += 2
            continue
        if w in _ARTICULOS_SINTAGMA:
            break
        if w not in existentes:
            agregados.append(w)
            existentes.add(w)
        j += 1
    return f"{texto_filtro} {' '.join(agregados)}".strip() if agregados else texto_filtro


_PATRON_REF_CONTEXTUAL_OPERACION = re.compile(
    r"\b("
    r"de\s+es[eoa]s?(?!\s+(?:semana|mes|a[ñn]o|d[ií]a|jornada|turno|ma[ñn]ana|tarde|noche)\b)|"
    r"de\s+est[eoa]s?(?!\s+(?:semana|mes|a[ñn]o|d[ií]a|jornada|turno|ma[ñn]ana|tarde|noche)\b)|"
    r"de\s+aquell?[oa]s?(?!\s+(?:semana|mes|a[ñn]o|d[ií]a|jornada|turno|ma[ñn]ana|tarde|noche)\b)|"
    r"es[eoa]\s+mism[oa]|est[eoa]\s+mism[oa]|el\s+mismo|la\s+misma|del\s+mismo|de\s+la\s+misma|"
    r"es[eoa]\s+(?:producto|art[ií]culo|referencia|equipo|herramienta|vaina)|"
    r"est[eoa]\s+(?:producto|art[ií]culo|referencia|equipo|herramienta|vaina)|"
    r"dicho\s+producto|el\s+producto\s+anterior|"
    r"del\s+que\s+(?:acabamos\s+de\s+(?:consultar|ver|buscar|hablar)|consult[eé]|pregunt[eé]|hablamos|vimos)|"
    r"el\s+que\s+(?:acabamos\s+de\s+(?:consultar|ver|buscar|hablar)|consult[eé]|pregunt[eé]|hablamos|vimos)|"
    r"s[aá]ca(?:lo|la|los|las)|ret[ií]ra(?:lo|la|los|las)|desp[aá]cha(?:lo|la|los|las)|"
    r"pr[eé]sta(?:lo|la|los|las|selo|sela|selos|selas)|"
    r"devu[eé]lve(?:lo|la|los|las)|regr[eé]sa(?:lo|la|los|las)|"
    r"ingr[eé]sa(?:lo|la|los|las)|m[eé]te(?:lo|la|los|las)|"
    r"trasl[aá]da(?:lo|la|los|las)|p[aá]sa(?:lo|la|los|las)|mu[eé]ve(?:lo|la|los|las)|"
    r"inhabil[ií]ta(?:lo|la)|inhabilitar(?:lo|la)|desact[ií]va(?:lo|la)|desactivar(?:lo|la)|arch[ií]va(?:lo|la)|archivar(?:lo|la)|"
    r"habil[ií]ta(?:lo|la)|habilitar(?:lo|la)|act[ií]va(?:lo|la)|activar(?:lo|la)|react[ií]va(?:lo|la)|reactivar(?:lo|la)|"
    r"(?:lo|la|los|las)\s+(?:quiero|voy\s+a|deseo|necesito)\s+(?:sacar|retirar|prestar|devolver|ingresar|trasladar|mover|pasar|inhabilitar|desactivar|habilitar|activar|archivar)"
    r")\b",
    re.IGNORECASE,
)


def tiene_referencia_contextual_operacion(texto: str) -> bool:
    """Determina si un mensaje de operación contiene una referencia contextual explícita al producto en foco.

    Garantiza que expresiones genéricas como 'Registra una salida de 5', 'Haz un préstamo',
    'Necesito sacar tres unidades' o 'Quiero hacer un traslado' NO hereden implícitamente
    el último producto consultado.
    """
    if not texto:
        return False
    return bool(_PATRON_REF_CONTEXTUAL_OPERACION.search(texto.strip()))


def _es_consulta_global_inventario(texto: str) -> bool:
    """Reconoce consultas cuyo objeto es el conjunto visible, no una entidad."""
    texto_pre = preprocesar_texto_inventario(texto or "")
    txt_low = (texto_pre or "").lower()
    return bool(
        re.search(
            r"\b(?:qu[eé]\s+(?:productos?|art[ií]culos?|referencias?|materiales?|insumos?|items?)|"
            r"cu[aá]ntos\s+(?:productos?|art[ií]culos?|referencias?)|"
            r"cu[aá]ntas\s+unidades\s+(?:disponibles?\s+)?(?:tenemos|hay)\s+en\s+"
            r"(?:total|el\s+inventario|todo\s+(?:el\s+)?inventario|el\s+almac[eé]n|nuestro\s+inventario)|"
            r"total\s+de\s+unidades|unidades\s+en\s+total|suma\s+de\s+(?:unidades|existencias)|"
            r"qu[eé]\s+(?:tenemos|hay|queda)\s+(?:en|del|de)\s+(?:todo\s+)?(?:(?:el|los|las)\s+)?(?:inventario|cat[aá]logo|almac[eé]n|bodega|productos?)|"
            r"(?:hay|tenemos)\s+algo\s+en\s+(?:(?:el|los|las)\s+)?(?:inventario|cat[aá]logo|productos?)|"
            r"cu[aá]nto\s+(?:queda|hay|tenemos)\s+(?:en\s+(?:todo\s+)?(?:el\s+)?inventario|de\s+todo)|"
            r"(?:lista|listado|cat[aá]logo)\s+(?:(?:complet[oa]|general|total|actual)\s+)?"
            r"(?:de\s+(?:tod[oa]s?\s+(?:los|las)\s+)?(?:productos?|art[ií]culos?|referencias?|materiales?|insumos?|items?)|"
            r"del?\s+(?:inventario|almac[eé]n|bodega))|"
            r"(?:mu[eé]stra(?:me)?|mu[eé]streme|mostrame|mostrar|ens[eé]ña(?:me)?|ver|dame|dime|lista(?:r|me)?|consultar?)\s+(?:todo\s+)?el\s+(?:inventario|cat[aá]logo)(?:\s+completo)?|"
            r"todo\s+(?:el\s+)?(?:inventario|cat[aá]logo)|"
            r"todos\s+los\s+productos|lista(r)?\s+todo)\b",
            txt_low,
        )
        or re.search(r"\b(?:podemos\s+)?(?:despachar|entregar|enviar)\s*(?:hoy|ahora)?\b", txt_low)
        or re.search(r"\b(?:aparecen?|registrados?)\s+en\s+total\b", txt_low)
    )


def _es_filtro_marco_global(texto: Optional[str]) -> bool:
    """Identifica filtros compuestos solo por vocabulario de alcance, no entidades."""
    terminos = set(normalizar(texto or "").split())
    return bool(terminos) and terminos.issubset(_PALABRAS_NO_FILTRO_GLOBAL)


def _plan_es_consulta_global_sin_entidad(plan: PlanAgente, texto: str) -> bool:
    return bool(
        plan.tipo == "consulta"
        and plan.operacion in {"listar", "contar", "sumar_unidades"}
        and plan.referencia == "ninguna"
        and not plan.filtros.codigos
        and not plan.filtros.categoria
        and not plan.filtros.categoria_explicita
        and _es_consulta_global_inventario(texto)
    )


def _normalizar_alcance_entidad_por_morfologia(texto: str, plan: PlanAgente) -> PlanAgente:
    """Resuelve señales gramaticales de conjunto sin conocer nombres del catálogo.

    El modelo sigue interpretando la intención completa. Esta capa pequeña corrige
    únicamente evidencias estructurales inequívocas: un cuantificador sobre un
    filtro acotado, pluralidad del descriptor o una construcción plural que rige
    «de + descriptor». No convierte el plural de un complemento («caja de grapas»)
    en una petición colectiva.
    """
    if not texto:
        return plan

    # A validated unique code is stronger evidence of a single reference than
    # incidental plurals in its name, model description, or measurements.
    if len(plan.filtros.codigos or []) == 1:
        plan.alcance_entidad = "producto"
        return plan
    if (
        plan.alcance_entidad in {"global", "coleccion"}
        or plan.referencia in ("foco", "resultados_previos")
    ):
        return plan

    # Un conteo sin un descriptor de producto sí representa una colección.
    # Si hay texto de producto, conserva el alcance semántico del modelo y deja
    # que la morfología del descriptor distinga «cajas de grapas» (colección)
    # de «caja de grapas» (un tipo de producto, potencialmente ambiguo).
    if (
        plan.tipo == "consulta"
        and plan.operacion == "contar"
        and not plan.filtros.texto
        and not plan.filtros.codigos
    ):
        plan.alcance_entidad = "coleccion"
        return plan
    filtros_de_conjunto = any((
        plan.filtros.categoria,
        plan.filtros.categoria_explicita,
        plan.filtros.almacen,
        plan.filtros.ubicacion,
        plan.filtros.condicion,
        plan.filtros.es_prestable is not None,
    ))
    if (
        plan.tipo == "consulta"
        and plan.operacion in {"listar", "sumar_unidades"}
        and filtros_de_conjunto
        and not plan.filtros.codigos
        and not plan.filtros.texto
    ):
        plan.alcance_entidad = "coleccion"
        return plan

    descriptor = plan.filtros.texto or plan.filtros.categoria
    if not descriptor:
        return plan
    raices_descriptor = set(tokens_significativos(descriptor))
    if not raices_descriptor:
        return plan

    tokens = normalizar(preprocesar_texto_inventario(texto)).split()
    if re.search(r"\b(?:todo|toda|todos|todas|cada)\b", " ".join(tokens)):
        plan.alcance_entidad = "coleccion"
        return plan

    for indice, token in enumerate(tokens):
        raiz = _raiz(token)
        if raiz == token or raiz not in raices_descriptor:
            continue

        # En «caja de grapas», la pluralidad pertenece al contenido, no al
        # referente del inventario. No aplicar esta excepción a «stock de cables».
        if indice >= 2 and tokens[indice - 1] == "de":
            nucleo_previo = tokens[indice - 2]
            if (
                nucleo_previo not in _PALABRAS_NO_FILTRO_GLOBAL
                and _raiz(nucleo_previo) == nucleo_previo
            ):
                continue

        plan.alcance_entidad = "coleccion"
        return plan

    # Una construcción plural que gobierna «de + descriptor» también marca un
    # conjunto aunque el descriptor posterior esté en singular. La decisión se
    # basa en morfología y estructura, no en un vocabulario de productos.
    for indice, token in enumerate(tokens[:-1]):
        if _raiz(token) != token and tokens[indice + 1] in {"de", "del"}:
            if any(_raiz(siguiente) in raices_descriptor for siguiente in tokens[indice + 2 :]):
                plan.alcance_entidad = "coleccion"
                break

    return plan


def validar_plan(
    datos: Dict[str, Any],
    texto_usuario: str,
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Valida el JSON del LLM contra el contrato. Descarta lo que no esté en el esquema."""
    if not isinstance(datos, dict):
        return None

    medida_raw = datos.get("medida_solicitada")
    filtros_raw_medida = datos.get("filtros")
    if not _str(medida_raw) and isinstance(filtros_raw_medida, dict):
        medida_raw = filtros_raw_medida.get("medida_solicitada")
    medida_solicitada = _str(medida_raw)
    medida_solicitada = (
        medida_solicitada.casefold()
        if medida_solicitada and medida_solicitada.casefold() in MEDIDAS_SOLICITADAS_VALIDAS
        else None
    )
    alcance_entidad_raw = _str(datos.get("alcance_entidad"))
    alcance_entidad = (
        alcance_entidad_raw.casefold()
        if alcance_entidad_raw and alcance_entidad_raw.casefold() in ALCANCES_ENTIDAD_VALIDOS
        else None
    )

    # Fase II-C.1: Si los datos provienen del contrato canónico PlanHaku (con clave 'familia')
    if "familia" in datos and "tipo" not in datos:
        from .contrato import adaptar_plan_haku_a_plan_agente, validar_plan_haku
        plan_h, _ = validar_plan_haku(datos, texto_usuario)
        if plan_h is not None:
            adapted = adaptar_plan_haku_a_plan_agente(plan_h)
            datos = dict(datos)
            datos["tipo"] = adapted.tipo
            if adapted.operacion:
                datos["operacion"] = adapted.operacion
            if adapted.accion:
                datos["accion"] = adapted.accion
            if adapted.filtros:
                datos["filtros"] = {
                    "texto": adapted.filtros.texto,
                    "codigos": list(adapted.filtros.codigos),
                    "categoria": adapted.filtros.categoria,
                    "almacen": adapted.filtros.almacen,
                    "ubicacion": adapted.filtros.ubicacion,
                    "condicion": adapted.filtros.condicion,
                    "umbral": adapted.filtros.umbral,
                }
            if adapted.campos:
                datos["campos"] = list(adapted.campos)
            if adapted.referencia:
                datos["referencia"] = adapted.referencia
            if adapted.producto:
                datos["producto"] = adapted.producto
            if adapted.cantidad is not None:
                datos["cantidad"] = adapted.cantidad
            if adapted.destinatario:
                datos["destinatario"] = adapted.destinatario
            if adapted.destino:
                datos["destino"] = adapted.destino
            if adapted.pregunta:
                datos["pregunta"] = adapted.pregunta
            if adapted.alcance_entidad:
                datos["alcance_entidad"] = adapted.alcance_entidad
            alcance_entidad = adapted.alcance_entidad

    tipo = _str(datos.get("tipo"))
    tipo = tipo.lower() if tipo else None
    if tipo not in TIPOS_VALIDOS:
        return None

    alcance_temporal = _str(datos.get("alcance"))
    alcance_temporal = alcance_temporal.lower() if alcance_temporal else None
    if alcance_temporal not in {"actual", "evento", "ambos", "indefinido"}:
        alcance_temporal = None
    fecha_desde = _fecha_iso_validada(datos.get("fecha_desde"))
    fecha_hasta = _fecha_iso_validada(datos.get("fecha_hasta"))
    if fecha_desde and fecha_hasta and fecha_desde > fecha_hasta:
        fecha_desde = fecha_hasta = None
    tipo_movimiento = _normalizar_tipo_movimiento(datos.get("tipo_movimiento"))
    if alcance_temporal == "actual":
        tipo_movimiento = None
    elif tipo_movimiento and alcance_temporal != "ambos":
        alcance_temporal = "evento"
    tipo_movimiento = _inferir_tipo_movimiento_desde_etiqueta(
        texto_usuario, alcance_temporal, tipo_movimiento,
    )

    texto_pre = preprocesar_texto_inventario(texto_usuario, normalizar_medidas=False)
    txt_low = (texto_pre or "").lower()
    txt_norm = normalizar(texto_pre)

    # Protección contra inyección SQL / comandos destructivos de sistema fuera del dominio
    if re.search(
        r"\b(?:drop\s+table|delete\s+from|truncate\s+table|insert\s+into|select\s+\*\s+from|alter\s+table|rm\s+-rf|format\s+c:)\b",
        txt_low,
    ):
        return PlanAgente(tipo="fuera_dominio", crudo=datos)

    if _es_pregunta_capacidades_agente(texto_usuario):
        return PlanAgente(
            tipo="capacidades",
            referencia="ninguna",
            fuente_plan="normalizador_capacidades_agente",
            crudo=datos,
        )

    # Detección de capacidades por lenguaje natural
    pide_capacidades_o_ayuda = bool(
        re.search(
            r"\b(?:qu[eé]\s+(?:puedes\s+hacer|haces|sabes\s+hacer|funciones\s+tienes)|"
            r"en\s+qu[eé]\s+(?:me\s+)?puedes\s+ayudar|cu[aá]les\s+son\s+tus\s+funciones|"
            r"para\s+qu[eé]\s+sirves|qu[eé]\s+es\s+haku|ay[uú]dame|necesito\s+(?:que\s+me\s+ayudes|ayuda))\b",
            txt_low,
        )
    )
    if pide_capacidades_o_ayuda and not re.search(r"\b(?:cu[aá]nt[oa]s?|d[oó]nde|qu[ií]en|busca|muestra|lista|prest|devuelv|sac|ingres|traslad)\b", txt_low):
        tipo = "capacidades"
    elif tipo == "capacidades" and not re.search(
        r"\b(ayud[a-záéíóú]*|puedes|puedo|sabes|haces|hacemos|funciones|capacidades|sirves|haku|asistente)\b",
        txt_low,
    ):
        tipo = "consulta"

    # Detección de fuera de dominio por pregunta conceptual o de definición general
    if _es_pregunta_conceptual_fuera_dominio(texto_pre, txt_norm):
        return PlanAgente(tipo="fuera_dominio", referencia="ninguna", crudo=datos)

    if tipo == "fuera_dominio" and pide_capacidades_o_ayuda and any(
        k in txt_norm for k in ("inventario", "stock", "producto", "almacen", "ubicacion", "referencia", "prestamo", "movimiento")
    ):
        tipo = "capacidades"

    # Detección de consultas de factibilidad de stock («alcanza para...», «se pueden sacar...»)
    es_pregunta_factibilidad = bool(
        re.search(r"\b(?:alcanza|se\s+puede[n]?\s+(?:sacar|retirar|despachar|entregar)|tenemos\s+suficiente|hay\s+suficiente)\b", txt_low)
    )
    if es_pregunta_factibilidad:
        tipo = "consulta"
    # Detección de intención operativa de cambio de stock mínimo
    elif re.search(
        r"\b(?:solicitar|quiero|deseo|voy\s+a\s+|favor|por\s+favor)?\s*(?:un\s+)?cambi(?:o|ar|e)\s+(?:de\s+)?stock\s+m[ií]nimo\b",
        txt_low,
    ):
        tipo = "operacion"
        datos["accion"] = "cambiar_stock_minimo"

    # Corrige una clase equivocada del LLM solo cuando el mensaje empieza con
    # una orden inequívoca. La operación sigue pasando por propuesta y confirmación.
    acciones_explicitas = (
        ("entrada", r"^(?:(?:por\s+favor|quiero|necesito)\s+)*(?:ingresar|ingresa|ingrese|registrar\s+(?:una\s+)?entrada|dar\s+entrada|meter)\b"),
        ("salida", r"^(?:(?:por\s+favor|quiero|necesito)\s+)*(?:registrar\s+(?:una\s+)?salida|sacar|retirar|despachar)\b"),
        ("prestamo", r"^(?:(?:por\s+favor|quiero|necesito)\s+)*(?:prestar|pr[eé]sta(?:r|s|me|le|lo|la|los|las|les)?|pr[eé]ste(?:n|me|le|lo|la|los|las|les)?|pr[eé]sten|pr[eé]stamo)\b"),
        ("devolucion", r"^(?:(?:por\s+favor|quiero|necesito)\s+)*(?:devolver|registrar\s+(?:una\s+)?devoluci[oó]n)\b"),
        ("traslado", r"^(?:(?:por\s+favor|quiero|necesito)\s+)*(?:trasladar|hacer\s+(?:un\s+)?traslado)\b"),
        ("crear_producto", r"^(?:(?:por\s+favor|quiero|necesito)\s+)*(?:crear|registrar)\s+(?:un\s+)?(?:nuevo\s+)?(?:producto|art[ií]culo)\b"),
    )
    for accion_explicita, patron_explicito in acciones_explicitas:
        if re.search(patron_explicito, txt_low):
            tipo = "operacion"
            datos["accion"] = accion_explicita
            break

    # Anti-alucinación de operaciones (mutaciones): requiere verbo o sustantivo de acción explícito en el texto
    if tipo == "operacion":
        mut_verb = re.search(
            r"\b(pr[eé]st|devolv|devuelv|devolu|ingres|entrad|sac[aáeio]|salid|retir|traslad|cambi|modifi|actuali|inhabilit|deshabilit|habilit|crea|registr|alta|nuev)",
            txt_low,
        )
        if not mut_verb:
            tipo = "aclaracion"
    tokens_usuario_base = set(tokens_significativos(texto_pre)) | set(txt_norm.split())
    tokens_usuario: set[str] = set(tokens_usuario_base)
    for u in list(tokens_usuario_base):
        tokens_usuario |= _sinonimos_de(u)
    copio_foco = False

    f = datos.get("filtros") if isinstance(datos.get("filtros"), dict) else {}
    codigos_raw = f.get("codigos") or []
    if isinstance(codigos_raw, str):
        codigos_raw = [codigos_raw]
    codigos_llm: List[str] = []
    for x in codigos_raw:
        c_str = _str(x)
        if not c_str:
            continue
        c_norm = normalizar_codigo_producto(c_str)
        if c_norm:
            codigos_llm.append(c_norm)

    # Los candidatos del catálogo son contexto de búsqueda, no evidencia de que
    # la persona haya elegido uno. Solo un código mencionado en el turno puede
    # estrechar una consulta a una referencia; los nombres completos vigentes
    # se resuelven después mediante el índice literal del catálogo.
    codigos_en_mensaje = extraer_codigos_de_texto(texto_pre)
    codigos = [c for c in codigos_llm if c in codigos_en_mensaje]
    producto_modelo = _str(f.get("producto")) or (_str(datos.get("producto")) if tipo == "consulta" else None)
    codigo_producto_modelo = normalizar_codigo_producto(producto_modelo) if producto_modelo else None
    codigo_producto_modelo_resuelto = bool(
        codigo_producto_modelo
        and codigo_producto_modelo in codigos_en_mensaje
    )
    if codigo_producto_modelo_resuelto:
        if codigo_producto_modelo not in codigos:
            codigos.append(codigo_producto_modelo)
        producto_modelo = None
    elif codigo_producto_modelo:
        # Si el modelo copió un código que solo vio como sugerencia contextual,
        # descártalo. La extracción posterior volverá al descriptor original.
        producto_modelo = None
    copio_foco = bool(codigos_llm and not codigos)
    if codigos_llm and not codigos and not copio_foco:
        copio_foco = True
    if not codigos and codigos_en_mensaje:
        codigos = list(codigos_en_mensaje)
    if codigo_producto_modelo_resuelto:
        copio_foco = False
    if tipo == "aclaracion" and codigos and not re.search(
        r"\b(pr[eé]st|devolv|devuelv|devolu|ingres|entrad|sac[aáeio]|salid|retir|traslad|cambi|modifi|actuali|inhabilit|deshabilit|habilit)",
        txt_low,
    ):
        tipo = "consulta"

    # Frase incompleta o truncada: nunca ejecutar como consulta completa ni reutilizar foco silenciosamente
    es_frase_incompleta = bool(
        re.search(
            r"(?:(?:\ben\s+(?:el\s+)?almac[eé]n\s+de|almac[eé]n\s+de|en\s+la\s+bodega\s+de)\s*[.…]*$|"
            r"\b(?:en|de|del|al|para|por|con|sobre)\s*[.…]+$)",
            texto_pre,
            flags=re.IGNORECASE,
        )
    )
    if es_frase_incompleta:
        tipo = "aclaracion"
        referencia = "ninguna"
        copio_foco = False
        codigos = []

    # Solicitud explícita de cambiar de producto
    if re.match(r"^(?:busca[r]?|mu[eé]stra(?:me)?|quiero\s+(?:ver|buscar|consultar))\s+otro\s+producto\b", txt_norm):
        tipo = "aclaracion"
        referencia = "ninguna"
        copio_foco = False
        codigos = []

    # Referencia al foco con almacén ambiguo no especificado
    if re.match(r"^(?:ese\s+mismo|este\s+mismo)(?:,\s*|\s+)en\s+otro\s+almac[eé]n\b", txt_norm):
        tipo = "aclaracion"
        referencia = "foco"

    campos_raw = datos.get("campos") or []
    if isinstance(campos_raw, str):
        campos_raw = [campos_raw]
    campos = [c for c in (_str(x) for x in campos_raw) if c in CAMPOS_VALIDOS]

    # El alcance temporal es una dimensión semántica del dato, no una lista
    # de expresiones que se detecta en el código. Un evento histórico se enruta
    # al historial; el saldo y la disponibilidad actuales usan sus propios campos.
    if tipo == "consulta" and alcance_temporal == "evento":
        campos = [c for c in campos if c not in ("stock", "prestamos")]
        if "movimientos" not in campos:
            campos.append("movimientos")

    if re.search(r"\b(?:c[oó]digo\s+de\s+barras|c[oó]digo\s+barra|barcode)\b", txt_low) and "codigo_barras" not in campos:
        campos.append("codigo_barras")

    es_consulta_elegibilidad_colectiva = bool(
        not codigos
        and alcance_entidad != "producto"
        and (
            re.search(
                r"\b(?:qu[eé]|cu[aá]les|cu[aá]nt[oa]s?|lista(?:r|me)?|mu[eé]stra(?:me)?|dime|ind[ií]came)\b",
                txt_low,
            )
            and re.search(
                r"\b(?:pr[eé]stables?|retornables?|consumibles?|devolutiv[oa]s?|se\s+pueden\s+prestar|admiten\s+pr[eé]stamos?|elegibles?\s+para\s+pr[eé]stamos?)\b",
                txt_low,
            )
            and not re.search(r"\binsumos\s+y\s+consumibles\b", txt_low)
        )
    )
    es_elegibilidad_negativa = bool(
        re.search(
            r"\b(?:consumibles?|no\s+(?:son\s+|es\s+)?(?:pr[eé]stables?|retornables?|devolutiv[oa]s?|elegibles?\s+para\s+pr[eé]stamos?)|no\s+(?:se\s+)?(?:puede[n]?|admite[n]?)\s+(?:prestar|pr[eé]stamos?))\b",
            txt_low,
        )
        and not re.search(r"\binsumos\s+y\s+consumibles\b", txt_low)
    )

    pregunta_es_prestable = bool(
        not es_consulta_elegibilidad_colectiva
        and re.search(
            r"\b(es\s+pr[eé]stable|son\s+pr[eé]stables|se\s+puede[n]?\s+prestar|admite[n]?\s+pr[eé]stamos?|es\s+retornable|es\s+consumible)\b",
            txt_low,
        )
    )
    if pregunta_es_prestable:
        if "es_prestable" not in campos:
            campos.append("es_prestable")
        if "prestamos" in campos and not re.search(r"\b(qui[eé]n|activos?|pendientes?)\b", txt_low):
            campos.remove("prestamos")
    elif es_consulta_elegibilidad_colectiva:
        campos = [c for c in campos if c != "es_prestable"]
        if "prestamos" in campos and not re.search(r"\b(qui[eé]n|activos?|pendientes?)\b", txt_low):
            campos.remove("prestamos")

    # Detección de atributos explícitos cuando el LLM los omite en preguntas directas
    if re.search(r"\b(c[oó]mo\s+se\s+llama|cu[aá]l\s+es\s+(su|el)\s+nombre)\b", txt_low) and "nombre" not in campos:
        campos.append("nombre")
    if (
        re.search(r"\b(?:cu[aá]l\s+es\s+(?:su|el)\s+c[oó]digo|qu[eé]\s+c[oó]digo\s+tiene|c[oó]digo\s+interno\s+de)\b", txt_low)
        and not re.search(r"\bc[oó]digo\s+(?:de\s+)?barras?\b", txt_low)
        and "codigo" not in campos
    ):
        campos.append("codigo")
    if re.search(r"\b(d[oó]nde\s+(?:est[aá]|se\s+encuentra|lo\s+guardan?|los\s+guardan?)|en\s+qu[eé]\s+ubicaci[oó]n|cu[aá]l\s+es\s+(?:su|la)\s+ubicaci[oó]n|ubicaci[oó]n\s+de|(?:y|e)\s+(?:(?:su|la)\s+)?ubicaci[oó]n)\b", txt_low) and "ubicacion" not in campos:
        campos.append("ubicacion")
    pregunta_almacen = bool(re.search(
        r"\b(?:en\s+(?:que|cual)\s+(?:(?:el|la)\s+)?(?:almacen|bodega|deposito)|"
        r"cual\s+es\s+(?:su\s+)?(?:almacen|bodega|deposito)|"
        r"(?:almacen|bodega|deposito)\s+(?:donde|en\s+el\s+que))\b",
        txt_norm,
    ))
    if pregunta_almacen and "almacen" not in campos:
        campos.append("almacen")
    if (
        not pregunta_es_prestable
        and not es_consulta_elegibilidad_colectiva
        and re.search(
            r"\b(qui[eé]n\s+(?:lo|la|los|las|se|nos)?\s*tiene(?:\s+prestad[oa]s?)?|a\s+qui[eé]n\s+se\s+(?:le\s+)?prest[oó]|pr[eé]stamos?\s+activos?|est[aá]\s+prestad[oa])\b",
            txt_low,
        )
        and "prestamos" not in campos
        and not re.search(r"\bqu[eé]\s+(productos|referencias)\b", txt_low)
    ):
        campos.append("prestamos")

    tiene_periodo_prestamo = False
    try:
        from ..servicios_temporales import resolver_intervalo_temporal
        tiene_periodo_prestamo = resolver_intervalo_temporal(texto_usuario) is not None
    except Exception:
        tiene_periodo_prestamo = False
    pregunta_prestamos_globales = bool(
        re.search(
            r"\b(pr[eé]stamos?\s+(?:activos?|pendientes?|vigentes?|abiertos?)|"
            r"qu[eé]\s+(?:(?:productos?|art[ií]culos?|referencias?|cosas?|material(?:es)?|equipos?|herramientas?)\s+)?(?:est[aá]n?|tenemos|hay|sigue[n]?)\s+(?:actualmente\s+)?(?:prestad[oa]s?|en\s+pr[eé]stamos?)(?:\s+ahora(?:\s+mismo)?)?|"
            r"productos?\s+prestados?)\b",
            txt_low,
        )
        or (
            not tiene_periodo_prestamo
            and not _extraer_objeto_consulta_sin_marco(texto_usuario)
            and re.search(
                r"\b(?:(?:mu[eé]stra(?:me)?|mu[eé]streme|mostrame|mostrar|ver|dime|dame|lista(?:r|me)?|qu[eé]|cu[aá]les\s+son)\s+"
                r"(?:(?:en\s+)?(?:la|el)\s+(?:lista|listado)\s+de\s+)?(?:los\s+)?pr[eé]stamos(?:\s+(?:que\s+hay|hay|vigentes?|abiertos?))?|"
                r"qui[eé]n(?:es)?\s+tiene[n]?\s+(?:(?:cosas?|productos?|art[ií]culos?|material|algo)\s+)?(?:prestad[oa]s?|en\s+pr[eé]stamos?))\b",
                txt_low,
            )
        )
    )

    condicion = _str(f.get("condicion"))
    condicion = condicion if condicion in CONDICIONES_VALIDAS else None
    filtro_es_prestable = f.get("es_prestable")
    if es_consulta_elegibilidad_colectiva and not isinstance(filtro_es_prestable, bool):
        filtro_es_prestable = False if es_elegibilidad_negativa else True
    if (
        not isinstance(filtro_es_prestable, bool)
        or codigos
        or alcance_entidad == "producto"
    ):
        filtro_es_prestable = None
    elif not es_consulta_elegibilidad_colectiva and _str(f.get("texto")) and not re.search(
        r"\b(?:pr[eé]stables?|retornables?|consumibles?|devolutiv[oa]s?|prestar|pr[eé]stamos?)\b",
        txt_low,
    ):
        filtro_es_prestable = None

    # Si pregunta quién tiene, si es prestable o si tiene préstamos de un producto específico, es consulta de atributo (no condición de filtro)
    if (codigos or copio_foco) and ("prestamos" in campos or "es_prestable" in campos):
        condicion = None
    elif pregunta_prestamos_globales and not codigos:
        if not re.match(
            r"^(?:(?:por\s+favor|quiero|necesito|deseo|voy\s+a)\s+)*"
            r"(?:prestar|presta(?:me|le|lo|la|los|las|les|s)?|preste\w*|registrar|hacer|haz)\b",
            txt_norm,
        ):
            tipo = "consulta"
        condicion = "con_prestamos"
        if "prestamos" not in campos:
            campos.append("prestamos")
        if not re.search(r"\b(?:cuant\w*|cantidad(?:es)?|unidades?|total(?:es)?|saldos?)\b", txt_norm):
            medida_solicitada = "atributo"

    # Si pregunta cuál es el valor del stock mínimo de un producto, es consulta de atributo (no condición de filtro)
    condiciones_stock_explicitas = _condiciones_stock_mencionadas(texto_usuario)
    entidad_minimo_individual = bool(codigos_en_mensaje or copio_foco or alcance_entidad == "producto")
    operacion_minimo_individual = _str(datos.get("operacion")) == "detalle"
    pide_atributo_minimo_individual = bool(
        entidad_minimo_individual
        and (operacion_minimo_individual or medida_solicitada == "atributo")
    )
    pide_atributo_minimo_estructurado = bool(
        "stock_minimo" in campos
        and "stock" not in campos
        # Una condición inferida se conserva salvo que el propio plan sitúe la
        # consulta en un producto y en su detalle/atributo individual. Esto
        # evita que stock_minimo, como campo de colección, borre el filtro.
        and (condicion is None or pide_atributo_minimo_individual)
        and "minimo" in tokens_significativos(texto_usuario)
        and "minimo_cero" not in condiciones_stock_explicitas
    )
    pregunta_valor_minimo = bool(
        re.search(r"\b(cu[aá]l\s+es\s+(su|el)\s+stock\s+m[ií]nimo|qu[eé]\s+stock\s+m[ií]nimo\s+tiene)\b", txt_low)
        or pide_atributo_minimo_estructurado
    )
    pide_pregunta_cuantos_articulo = bool(
        (
            re.search(r"\bcu[aá]nt[oa]s?\s+(?:unidades\s+(?:de\s+)?)?.+?\s+(?:tenemos|hay|quedan?|existen?)\b", txt_low)
            or re.search(r"\bcu[aá]nt[oa]s?\s+(?:tenemos|hay|quedan?|existen?|tenemos\s+de|hay\s+de)\b", txt_low)
            or (
                not re.search(r"\b(?:qu[eé]|cu[aá]l(?:es)?|d[oó]nde|qui[eé]n(?:es)?)\b", txt_low)
                and re.search(
                    r"\b(?:hay|existe[n]?)\s+(?!(?:productos?|art[ií]culos?|referencias?|categor[ií]as?|unidades?|stock|existencias?|en|dentro|fuera|aqu[ií]|all[ií]|ah[ií]|hoy|ahora|actualmente|para|por|con|sin)\b)[a-záéíóúñ0-9].+",
                    txt_low,
                )
            )
        )
        and not re.search(r"\bcu[aá]nt[oa]s?\s+(?:productos?|art[ií]culos?|referencias?|categor[ií]as?|almacenes?|items?|tipos?|variedades?)\b", txt_low)
        and not es_consulta_elegibilidad_colectiva
    )

    pide_atributo_stock = (
        not pregunta_valor_minimo
        and not es_consulta_elegibilidad_colectiva
        and bool(
            re.search(
                r"\b(cu[aá]nto\s+(?:stock|queda)|cu[aá]ntas?\s+(?:unidades|existencias?|quedan)|qu[eé]\s+(?:stock|existencias?)\s+tiene|existencias?\s+(?:actuales?\s+)?de|stock\s+(?:actual\s+)?de)\b",
                txt_low,
            )
            or pide_pregunta_cuantos_articulo
            or es_pregunta_factibilidad
        )
    )
    # Cuando el modelo ha identificado una consulta de préstamos o de historial,
    # la cantidad pedida pertenece a ese dato temporal, no al stock disponible.
    if "prestamos" in campos or "movimientos" in campos:
        pide_atributo_stock = False
        pide_pregunta_cuantos_articulo = False
    if pide_atributo_stock and "stock" not in campos:
        campos.append("stock")

    if pregunta_valor_minimo:
        condicion = None
        if "stock_minimo" not in campos:
            campos.append("stock_minimo")
    elif tipo == "consulta" and "minim" not in txt_norm and re.search(
        r"(?<!\bno\s)(?<!\bni\s)(?<!\bsin\sestar\s)\b(?:con\s+(?:0|cero)\s+unidades|sin\s+(?:stock|existencias|unidades)|no\s+tienen?\s+(?:nada\s+de\s+)?(?:stock|existencias|unidades)|agotad[oa]s?|en\s+cero|cero\s+existencias?|stock\s+(?:en\s+)?(?:0|cero|nulo)|existencias?\s+nulas?)\b",
        txt_low,
    ):
        # La condición declarada literalmente por el usuario prevalece sobre
        # una condición opuesta o genérica emitida por el LLM.
        condicion = "agotado"
    elif condicion == "minimo_cero" and "minim" not in txt_norm:
        condicion = "agotado" if _condicion_stock_tiene_indicio_textual(texto_usuario, "agotado") else None
    elif tipo == "consulta":
        if re.search(r"\b(por\s+debajo\s+de(l)?\s+(?:su\s+)?(?:stock\s+)?m[ií]nimo|bajo\s+(?:su\s+)?(?:stock\s+)?m[ií]nimo)\b", txt_low):
            condicion = "bajo_minimo"
        elif re.search(r"\b(cerca\s+de(l)?\s+(?:su\s+)?(?:stock\s+)?m[ií]nimo|pr[oó]xim[oa]s?\s+a(l)?\s+(?:su\s+)?(?:stock\s+)?m[ií]nimo|a\s+punto\s+de\s+(?:llegar|alcanzar)\s+(?:a(l)?\s+)?(?:su\s+)?(?:stock\s+)?m[ií]nimo)\b", txt_low):
            condicion = "cerca_minimo"
        elif condicion is None:
            if re.search(r"(?<!\bno\s)(?<!\bni\s)(?<!\bsin\sestar\s)\b(con\s+(0|cero)\s+unidades|sin\s+(stock|existencias|unidades)|no\s+tienen?\s+(?:nada\s+de\s+)?(?:stock|existencias|unidades)|agotad[oa]s?|en\s+cero|cero\s+existencias?|stock\s+(?:en\s+)?(?:0|cero|nulo)|existencias?\s+(?:nulas?|(?:en\s+)?(?:0|cero)))\b", txt_low) and "minim" not in txt_norm:
                condicion = "agotado"
            elif (
                re.search(r"\b(stock\s+m[ií]nimo\s+(en\s+)?(0|cero)|m[ií]nimo\s+(en\s+)?(0|cero)|sin\s+stock\s+m[ií]nimo)\b", txt_low)
                and not (
                    isinstance(datos.get("atributos_stock_solicitados"), list)
                    and "stock_minimo" in datos.get("atributos_stock_solicitados")
                    and _str(datos.get("relacion_stock")) == "sin_condicion"
                    and (
                        _producto_es_entidad_concreta(_str(datos.get("producto")))
                        or bool(f.get("codigos"))
                        or bool(_str(f.get("texto")))
                    )
                )
            ):
                condicion = "minimo_cero"

    # Limpiar valores pipe-separated que el LLM a veces genera
    categoria_raw = _str(f.get("categoria"))
    if categoria_raw and "|" in categoria_raw:
        categoria_raw = None  # todas las categorías → sin filtro
    almacen_raw = _str(f.get("almacen"))
    if almacen_raw and "|" in almacen_raw:
        almacen_raw = None  # todos los almacenes → sin filtro

    pregunta_atributo_categoria = bool(
        re.search(r"\b(qu[eé]\s+categor[ií]a|cu[aá]l\s+(es\s+(su|la)\s+)?categor[ií]a)\b", txt_low)
    )
    pregunta_listar_categorias = bool(
        re.search(r"\b(qu[eé]\s+categor[ií]as?\s+(hay|existen?|tenemos|manejamos|registradas?)|cu[aá]les\s+(son\s+las\s+)?categor[ií]as?|lista(r)?\s+(las\s+)?categor[ií]as?)\b", txt_low)
    )
    pregunta_contar_categorias = bool(re.search(r"\bcu[aá]ntas\s+categor[ií]as\b", txt_low))
    if pregunta_listar_categorias or pregunta_contar_categorias:
        tipo = "consulta"
        categoria_raw = None
        campos = ["categoria"]
    elif pregunta_atributo_categoria:
        categoria_raw = None
        if "categoria" not in campos:
            campos.append("categoria")

    m_cat_explicita = (
        None
        if (pregunta_atributo_categoria or pregunta_listar_categorias or pregunta_contar_categorias)
        else re.search(
            r"\b(?:de\s+la\s+|en\s+la\s+|la\s+|por\s+)?categor[ií]as?[\?\.\,\:\;]*\s+(?:de\s+)?"
            r"(?!pertenece|est[aá]\b|es\b|tiene\b|son\b|hay\b|existen?\b|tenemos\b|manejamos\b|registradas?\b|"
            r"qu[eé]\b|cu[aá]l(?:es)?\b|cu[aá]nt[oa]s?\b|d[oó]nde\b|c[oó]mo\b|y\s+cu[aá]l\b)"
            r"([a-záéíóúñ0-9][a-záéíóúñ0-9\s]*?)(?:[\?\.\,!]|$|\s+(?:en|del|con|que|de\s+almac[eé]n|en\s+almac[eé]n)\b)",
            txt_low,
        )
    )
    categoria_explicita = bool(m_cat_explicita)
    if _es_filtro_ubicacion_generico(categoria_raw) and not categoria_explicita:
        categoria_raw = None

    CATEGORIAS_GENERICAS = {
        "que exista", "que existan", "existente", "existentes", "alguna", "algun", "cualquiera",
        "todas", "todos", "cualquier", "varias",
    }
    categoria_generica = False

    # Anti-alucinación: categoría solo si el usuario la menciona en su mensaje
    if categoria_raw:
        cat_norm = normalizar(categoria_raw)
        if cat_norm in CATEGORIAS_GENERICAS or cat_norm.endswith("que exista"):
            categoria_raw = None
            categoria_generica = True
        else:
            palabras_cat = [w for w in cat_norm.split() if len(w) >= 3 and w not in ("las", "los", "del", "para", "con")]
            if not any(w in txt_norm for w in palabras_cat):
                categoria_raw = None

    if categoria_explicita and not categoria_raw and m_cat_explicita:
        cand_cat = m_cat_explicita.group(1).strip()
        cand_norm = normalizar(cand_cat)
        if cand_norm in CATEGORIAS_GENERICAS or cand_norm.endswith("que exista"):
            categoria_raw = None
            categoria_explicita = False
            categoria_generica = True
        else:
            categoria_raw = cand_cat

    if (
        categoria_explicita
        and categoria_raw
        and not pregunta_atributo_categoria
        and not pregunta_listar_categorias
        and not pregunta_contar_categorias
        and "categoria" in campos
    ):
        campos = [c for c in campos if c != "categoria"]

    # Anti-alucinación: almacén solo si el usuario menciona ese almacén en su mensaje
    if almacen_raw:
        alm_norm = normalizar(almacen_raw)
        sin_pref = alm_norm.replace("almacen", "").replace("bodega", "").strip()
        palabras_alm = [w for w in sin_pref.split() if w not in ("las", "los", "del", "para", "con", "el", "la")]
        if not palabras_alm or not all(w in txt_norm.split() for w in palabras_alm):
            almacen_raw = None

    producto_atributo_explicito = False
    texto_filtro = _str(f.get("texto")) or producto_modelo
    if pregunta_listar_categorias or pregunta_contar_categorias:
        texto_filtro = None
    elif texto_filtro:
        texto_filtro = preprocesar_texto_inventario(texto_filtro, normalizar_medidas=False)
        # Si texto_filtro es solo un código ya capturado, limpiarlo
        cod_txt = normalizar_codigo_producto(texto_filtro)
        if cod_txt and cod_txt in codigos:
            texto_filtro = None
        elif categoria_explicita and categoria_raw and (
            normalizar(texto_filtro) in (
                normalizar(categoria_raw),
                normalizar(f"categoria {categoria_raw}"),
                normalizar(f"de la categoria {categoria_raw}"),
            )
            or (
                {t for t in tokens_significativos(texto_filtro) if normalizar(t) not in ("categoria", "categorias")}
                <= set(tokens_significativos(categoria_raw))
            )
        ):
            texto_filtro = None
        else:
            # Anti-alucinación: verificar que al menos un token del filtro provenga del mensaje del usuario o sus sinónimos
            toks_f = tokens_significativos(texto_filtro)
            if not toks_f or not any(
                t in tokens_usuario or any(u.startswith(t[:3]) for u in tokens_usuario if len(u) >= 3 and len(t) >= 3)
                for t in toks_f
            ):
                texto_filtro = None
                copio_foco = True
            elif texto_filtro:
                texto_filtro = _preservar_sintagma_nominal(texto_filtro, texto_pre, categoria_raw)

    # En preguntas de atributo, el modificador del atributo no es parte del
    # nombre del producto: «ubicación del sensor DHT22» busca el sensor.
    m_producto_ubicacion = re.search(
        r"\bubicaci[oó]n\s+(?:de\s+las\s+|de\s+los\s+|de\s+la\s+|del\s+|de\s+)"
        r"(.+?)(?:\s+en\s+(?:el\s+)?(?:almac[eé]n|bodega)\b|[?.!,]|$)",
        txt_low,
    )
    if tipo in ("consulta", "fuera_dominio") and m_producto_ubicacion:
        # Algunas respuestas del modelo etiquetan esta consulta como fuera
        # de dominio y omiten el atributo. Una ubicación pedida para un
        # producto es una consulta de inventario inequívoca.
        tipo = "consulta"
        texto_filtro = re.sub(
            r"^(?:el|la|los|las|un|una|producto|art[ií]culo)\s+", "",
            m_producto_ubicacion.group(1).strip(), flags=re.IGNORECASE,
        )
        producto_atributo_explicito = bool(texto_filtro)
        if "ubicacion" not in campos:
            campos.append("ubicacion")
        copio_foco = False

    # Mantiene el nombre del producto al pedir un atributo escrito antes del
    # objeto («código de barras del multímetro»), aunque el modelo lo omita.
    m_producto_codigo_barras = re.search(
        r"\b(?:c[oó]digo\s+de\s+barras|c[oó]digo\s+barra|barcode)\s+"
        r"(?:de\s+las\s+|de\s+los\s+|de\s+la\s+|del\s+|de\s+)"
        r"(.+?)(?:\s+en\s+(?:el\s+)?(?:almac[eé]n|bodega)\b|[?.!,]|$)",
        txt_low,
    )
    if tipo in ("consulta", "fuera_dominio") and m_producto_codigo_barras:
        tipo = "consulta"
        texto_filtro = re.sub(
            r"^(?:el|la|los|las|un|una|producto|art[ií]culo)\s+", "",
            m_producto_codigo_barras.group(1).strip(), flags=re.IGNORECASE,
        )
        producto_atributo_explicito = bool(texto_filtro)
        if "codigo_barras" not in campos:
            campos.append("codigo_barras")
        copio_foco = False

    if _es_filtro_ubicacion_generico(texto_filtro):
        objeto = _extraer_objeto_consulta_sin_marco(texto_pre)
        texto_filtro = objeto if objeto and not _es_filtro_ubicacion_generico(objeto) else None
        if texto_filtro:
            copio_foco = False

    # Si es una consulta funcional («¿qué material sirve para conectar?», «¿qué se usa para pegar?») y el LLM omitió el verbo funcional
    m_func = re.search(
        r"\b(?:sirve[n]?|sirva[n]?|se\s+usa[n]?|utiliza[n]?)\s+para\s+([a-záéíóúñ0-9]+(?:\s+[a-záéíóúñ0-9]+)?)",
        txt_low,
    )
    if m_func and tipo in ("consulta", "capacidades", "fuera_dominio"):
        tipo = "consulta"
        obj_func = m_func.group(1).strip()
        prim_tok = normalizar(obj_func).split()[0]
        sins_func = _sinonimos_de(prim_tok)
        if not texto_filtro:
            texto_filtro = f"para {obj_func}"
        elif not any(_raiz(w) in sins_func for w in normalizar(texto_filtro).split()):
            texto_filtro = f"{texto_filtro} para {obj_func}".strip()

    ubicacion_raw = _str(f.get("ubicacion"))
    if _es_filtro_ubicacion_generico(ubicacion_raw):
        ubicacion_raw = None
    if ubicacion_raw:
        ubi_norm = normalizar(ubicacion_raw)
        if (
            not ubi_norm
            or ubi_norm in ("bodega", "bodegas", "bodera", "almacen", "almacenes", "inventario")
            or ubi_norm not in txt_norm.split()
            or "ubicacion" in campos
        ):
            ubicacion_raw = None

    # Detección estructurada de intención para consultas de inventario
    es_consulta_global_inventario = _es_consulta_global_inventario(texto_pre)

    if tipo == "consulta" and not texto_filtro and not codigos and not categoria_raw and not categoria_generica and not (pregunta_listar_categorias or pregunta_contar_categorias):
        m_stock_de = re.search(
            r"\b(?:stock|existencias?|unidades?|cantidad|disponibilidad)\s+(?:que\s+)?(?:queda[n]?|hay|tenemos)?\s*(?:de|del)?\s+([a-záéíóúñ0-9][a-záéíóúñ0-9\s]*?)(?:[\?\.\,!]|$)",
            txt_low,
        )
        if m_stock_de:
            cand_txt = m_stock_de.group(1).strip()
            cand_txt = re.sub(r"^(?:de|del|para|sobre|en|con)\s+", "", cand_txt.strip())
            if tokens_significativos(cand_txt):
                texto_filtro = cand_txt
        elif not es_consulta_global_inventario and condicion in (None, "con_stock") and not ubicacion_raw and not pregunta_prestamos_globales:
            m_especifico = re.search(
                r"\b(?:tenemos|hay|queda|est[aá]|existe|tienen)\s+(?:disponible\s+)?(?:el|la|los|las|un|una)?\s*([a-záéíóúñ0-9][a-záéíóúñ0-9\s]*?)(?:\s+disponibles?|\s+en\s+stock|\?|$)",
                txt_low,
            )
            if m_especifico:
                cand_esp = m_especifico.group(1).strip()
                cand_esp = re.sub(
                    r"\s*\ben\s+(?:(?:el|la|los|las|nuestro|nuestra)\s+)?(?:almac[eé]n|bodega|ubicaci[oó]n|categor[ií]a|inventario|cat[aá]logo|productos?)\b.*$",
                    "",
                    cand_esp,
                    flags=re.IGNORECASE,
                ).strip()
                cand_esp = re.sub(r"^(?:de|del|para|sobre|en|con)\s+", "", cand_esp.strip())
                palabras_cand = [
                    w for w in cand_esp.split()
                    if normalizar(w) not in _FRONTERAS_SINTAGMA
                    and normalizar(w) not in _ARTICULOS_SINTAGMA
                    and normalizar(w) not in ("de", "del", "para", "sobre", "en", "con")
                ]
                if palabras_cand and not all(normalizar(p) in _PALABRAS_NO_FILTRO_GLOBAL for p in palabras_cand):
                    texto_filtro = " ".join(palabras_cand)

    if filtro_es_prestable is not None and texto_filtro:
        limpio_elig = re.sub(
            r"\b(?:productos?|art[ií]culos?|referencias?|items?|elementos?|materiales?|insumos?|herramientas?|"
            r"pr[eé]stables?|retornables?|consumibles?|devolutiv[oa]s?|elegibles?|"
            r"se\s+pueden?\s+prestar|no\s+se\s+pueden?\s+prestar|admiten?\s+pr[eé]stamos?|"
            r"para\s+pr[eé]stamos?|no|son|es|que|en|del|de|el|la|los|las)\b",
            " ",
            texto_filtro,
            flags=re.IGNORECASE,
        )
        limpio_elig = re.sub(r"\s+", " ", limpio_elig).strip(" ?¿.,!¡;:")
        texto_filtro = limpio_elig or None
    if filtro_es_prestable is not None and categoria_raw and not categoria_explicita:
        cat_tokens = [w for w in normalizar(categoria_raw).split() if len(w) >= 3 and w not in ("las", "los", "del", "para", "con")]
        if len(cat_tokens) >= 2 and not all(w in txt_norm.split() for w in cat_tokens):
            categoria_raw = None
    if texto_filtro:
        toks_tf = set(normalizar(texto_filtro).split())
        if toks_tf and toks_tf.issubset(_PALABRAS_NO_FILTRO_GLOBAL):
            texto_filtro = None

    es_patron_colectivo_plural = bool(
        es_consulta_global_inventario
        or categoria_explicita
        or categoria_generica
        or re.search(
            r"\b(?:qu[eé]\s+(?:productos?|art[ií]culos?|referencias?|materiales?|insumos?|items?|tipos?|variedades?|modelos?|herramientas?|[a-záéíóúñ]+s)|cu[aá]les\b|todos?\s+los\b|todas?\s+las\b|cat[aá]logo\b|(?:mu[eé]stra(?:me)?|lista(?:r)?|ver|dame)\s+(?:todos?\s+|todas?\s+)?(?:los|las)\s+(?:productos?|art[ií]culos?|referencias?|materiales?|insumos?|items?)|que\s+(?:todav[ií]a\s+|a[uú]n\s+)?(?:tengan?|est[eé]n|queden|haya))\b",
            txt_low,
        )
    )
    tf_norm = normalizar(texto_filtro) if texto_filtro else ""
    palabras_tf = tf_norm.split()
    es_sustantivo_grupo_plural = bool(
        len(palabras_tf) == 1
        and palabras_tf[0].endswith("s")
        and not re.search(r"\b(?:el|la|un|una|este|esta|ese|esa)\s+" + re.escape(palabras_tf[0]) + r"\b", txt_norm)
    )
    solicitud_detalle_singular = bool(
        re.search(r"\b(?:informaci[oó]n|detalles?|ficha|datos)\s+(?:de|del|sobre)\s+(?:el\s+|la\s+|un\s+|una\s+)?[a-záéíóúñ][a-záéíóúñ0-9 -]*", txt_low)
        or re.search(r"\bconsulta\s+sobre\s+(?:el\s+|la\s+|un\s+|una\s+)?[a-záéíóúñ][a-záéíóúñ0-9 -]*", txt_low)
        or re.search(r"\b(?:busca|buscar|b[uú]scame)\s+(?:el\s+|la\s+|un\s+|una\s+)?[a-záéíóúñ][a-záéíóúñ0-9 -]*", txt_low)
    ) and not re.search(r"\b(?:todos?|todas?|productos?|referencias?|categor[ií]as?|lista|cat[aá]logo)\b", txt_low)
    es_consulta_especifica_producto = bool(
        solicitud_detalle_singular
        or len(codigos) == 1
        or (
            texto_filtro
            and not es_patron_colectivo_plural
            and not es_sustantivo_grupo_plural
            and (
                pide_atributo_stock
                or re.search(
                    r"\b(?:tenemos|hay|queda[n]?|est[aá][n]?|existe[n]?|tienen|tiene|busca|b[uú]scame|ver|consultar?)\s+(?:disponibles?\s+)?(?:de\s+|del\s+)?(?:el\s+|la\s+|los\s+|las\s+|un\s+|una\s+)?"
                    + re.escape(tf_norm)
                    + r"\b",
                    txt_norm,
                )
                or re.search(
                    r"\b(?:stock|existencias?|unidades?|cantidad)\s+(?:disponibles?\s+)?(?:actual(?:es)?\s+)?(?:que\s+)?(?:queda[n]?|hay|tenemos|tiene[n]?)?\s*(?:de\s+|del\s+)?(?:el\s+|la\s+|los\s+|las\s+|un\s+|una\s+)?"
                    + re.escape(tf_norm)
                    + r"\b",
                    txt_norm,
                )
                or re.search(r"\b(?:el|la|un|una)\s+" + re.escape(tf_norm) + r"\s+(?:est[aá]\s+disponible|tiene\s+stock)\b", txt_norm)
                or re.search(r"^(?:b[uú]sca(?:me)?|consulta|ver)\s+(?:el\s+|la\s+|los\s+|las\s+|un\s+|una\s+)?" + re.escape(tf_norm) + r"$", txt_norm)
            )
        )
    )

    es_consulta_colectiva_disponibles = bool(
        not es_consulta_especifica_producto
        and not codigos
        and (
            re.search(r"\b(?:disponibles|con\s+stock|con\s+existencias|en\s+stock)\b", txt_low)
            or (
                re.search(r"\bdisponible\b", txt_low)
                and (
                    es_consulta_global_inventario
                    or es_patron_colectivo_plural
                    or re.search(r"\b(?:qu[eé]\s+(?:productos?|art[ií]culos?|referencias?|materiales?)|cu[aá]les|listar?|mu[eé]strame|ver)\s+(?:las?|los?)\b", txt_low)
                )
            )
            or re.search(r"\b(?:podemos\s+)?(?:despachar|entregar)\b", txt_low)
        )
    )

    if es_consulta_especifica_producto or codigos or (texto_filtro and not es_consulta_colectiva_disponibles):
        if condicion == "con_stock":
            condicion = None
    elif es_consulta_colectiva_disponibles and condicion not in ("agotado", "bajo_minimo", "en_minimo", "cerca_minimo", "suficiente", "minimo_cero"):
        condicion = "con_stock"

    filtros = FiltrosConsulta(
        texto=texto_filtro,
        codigos=codigos,
        categoria=categoria_raw,
        almacen=almacen_raw,
        ubicacion=ubicacion_raw,
        condicion=condicion,
        umbral=_int(f.get("umbral")) if condicion == "cerca_minimo" else None,
        categoria_explicita=categoria_explicita,
        categoria_generica=categoria_generica,
        es_prestable=filtro_es_prestable,
    )
    # El umbral solo es válido si el usuario dio un número
    if filtros.umbral is not None and not re.search(r"\d|uno|dos|tres|cuatro|cinco|diez", txt_low):
        filtros.umbral = None

    # Los filtros de existencias describen el estado presente. Una consulta de
    # eventos pasados solo usa uno si también solicita el alcance actual.
    if alcance_temporal == "evento":
        filtros.condicion = None

    operacion = _str(datos.get("operacion"))
    operacion = operacion if operacion in OPERACIONES_VALIDAS else None
    if tipo == "consulta":
        # El LLM ya expresa los atributos en las claves del contrato. Usar esas
        # claves evita que una paráfrasis («su mínimo», «dónde se guarda») se
        # reduzca a stock solo porque no repite literalmente una etiqueta del
        # esquema. Se conserva un campo no-stock si el mensaje lo respalda, si
        # el plan lo clasifica como atributo o si la sintaxis pide varias
        # propiedades en cláusulas coordinadas.
        atributos_no_stock = set(campos) - {"stock"}
        atributos_respaldados = set()
        for campo in atributos_no_stock:
            etiqueta = normalizar(campo.replace("_", " "))
            if etiqueta and re.search(rf"(?<!\w){re.escape(etiqueta)}(?!\w)", txt_norm):
                atributos_respaldados.add(campo)
        if "ubicacion" in atributos_no_stock and re.search(
            r"\b(?:donde|estantes?|vitrinas?|cajones?|repisas?|estanterias?|ubicaciones?)\b",
            txt_norm,
        ):
            atributos_respaldados.add("ubicacion")
        if "almacen" in atributos_no_stock and re.search(r"\b(?:almacen|bodega)\b", txt_norm):
            atributos_respaldados.add("almacen")
        if "prestamos" in atributos_no_stock and re.search(r"\b(?:prest\w*|devoluci\w*)\b", txt_norm):
            atributos_respaldados.add("prestamos")
        if "movimientos" in atributos_no_stock and re.search(
            r"\b(?:movimient\w*|historial|historico|entrada|salida|traslado)\b", txt_norm,
        ):
            atributos_respaldados.add("movimientos")
        marcadores_interrogativos = set(re.findall(
            r"\b(?:que|cual|cuales|como|donde|cuanto|cuanta|cuantos|cuantas|quien|quienes)\b",
            txt_norm,
        ))
        pregunta_compuesta = len(marcadores_interrogativos) >= 2
        otros_atributos_explicitos = bool(atributos_no_stock) and (
            medida_solicitada == "atributo"
            or bool(atributos_respaldados)
            or pregunta_compuesta
        )
        if re.search(r"\b(cu[aá]l\s+(de\s+(?:esos|ellos|ellas|las\s+anteriores|los\s+anteriores)\s+)?tiene\s+(?:m[aá]s|mayor)|cu[aá]l\s+es\s+el\s+que\s+tiene\s+(?:m[aá]s|mayor)|con\s+(?:m[aá]s|mayor)\s+(?:unidades|existencias?|stock|cantidad)|mayor\s+(?:stock|existencias?))\b", txt_low):
            # Choose the extremum from the requested result set before the
            # generic stock-quantity rule. The latter otherwise turns
            # "which one has the most available?" into a sum of all stock.
            operacion = "mayor_stock"
        elif re.search(r"\b(cu[aá]l\s+(de\s+(?:esos|ellos|ellas|las\s+anteriores|los\s+anteriores)\s+)?tiene\s+(?:menos|menor)|cu[aá]l\s+es\s+el\s+que\s+tiene\s+(?:menos|menor)|con\s+(?:menos|menor)\s+(?:unidades|existencias?|stock|cantidad)|menor\s+(?:stock|existencias?))\b", txt_low):
            operacion = "menor_stock"
        elif pide_atributo_stock and not otros_atributos_explicitos and condicion != "agotado":
            # Cuando se pide stock como único atributo, la respuesta y el
            # contrato estructurado deben representar cantidades, no una ficha.
            # La suma también conserva una respuesta nominal cuando solo hay un
            # producto; detalle queda para atributos distintos de la existencia.
            # Si el nombre es ambiguo, la resolución posterior pide cuál y no
            # suma entidades diferentes.
            pregunta_atributo_individual = bool(re.search(
                r"\b(?:cu[aá]nto\s+(?:stock|existencias?|unidades?)\s+(?:tiene|hay|queda)|"
                r"qu[eé]\s+(?:stock|existencias?)\s+tiene|cu[aá]nto\s+queda)\b",
                txt_low,
            ))
            operacion = (
                "detalle"
                if codigos or copio_foco or pregunta_atributo_individual
                else "sumar_unidades"
            )
            campos = ["stock"]
        elif pregunta_contar_categorias:
            operacion = "contar"
            campos = ["categoria"]
        elif pregunta_listar_categorias:
            operacion = "listar"
            campos = ["categoria"]
        elif not es_consulta_especifica_producto and re.search(r"\b(cu[aá]ntas\s+unidades|total\s+de\s+unidades|unidades\s+en\s+total|suma\s+de\s+(unidades|existencias)|unidades\s+disponibles\s+en\s+total|cu[aá]nto\s+(?!stock\b|tiempo\b)[a-záéíóúñ]+\s+(?:queda|hay|tenemos))\b", txt_low):
            operacion = "sumar_unidades"
        elif not es_consulta_especifica_producto and re.search(r"\b(cu[aá]ntos\s+(productos|art[ií]culos|tipos)|cu[aá]ntas\s+referencias|cantidad\s+de\s+(productos|referencias))\b", txt_low):
            operacion = "contar"
        elif re.search(r"\b(qu[eé]\s+(?:referencias|productos|art[ií]culos|tipos|variedades|modelos|material|materiales|insumos|herramientas)|cu[aá]les(?!\s+de\s+(?:esos|ellos)))\b", txt_low):
            operacion = "listar"
        elif operacion == "sumar_unidades" and condicion == "agotado":
            operacion = "listar"
        elif operacion == "explicar_estado" and not re.search(r"\b(por\s+qu[eé]|explica|raz[oó]n|motivo|estado)\b", txt_low):
            operacion = "detalle" if (codigos or copio_foco or es_consulta_especifica_producto) else "listar"
        elif pregunta_valor_minimo or pregunta_es_prestable or (campos and operacion in ("listar", "contar") and (codigos or copio_foco)):
            operacion = "detalle"
        elif es_consulta_especifica_producto and not es_consulta_colectiva_disponibles:
            if pide_atributo_stock and not (set(campos) - {"stock"}):
                operacion = "sumar_unidades"
            elif pide_pregunta_cuantos_articulo and not (set(campos) - {"stock"}):
                operacion = "sumar_unidades"
            elif operacion in ("listar", "contar", "sumar_unidades", None):
                operacion = "detalle"
            if not campos and (
                pide_atributo_stock
                or re.search(r"\b(tenemos|hay|queda[n]?|existe[n]?|tienen|tiene|disponibles?|stock|existencias?|unidades?|cu[aá]nt[oa]s?)\b", txt_low)
            ):
                campos = ["stock"]
        elif operacion == "contar" and pide_atributo_stock and not re.search(r"\b(cu[aá]ntos\s+(?:productos?|art[ií]culos?|tipos?|items?)|cu[aá]ntas\s+(?:referencias?|categor[ií]as?)|cantidad\s+de\s+(?:productos?|referencias?))\b", txt_low):
            operacion = "detalle" if (codigos or copio_foco or texto_filtro) else "sumar_unidades"
            if operacion == "detalle" and "stock" not in campos:
                campos.append("stock")
        elif pregunta_prestamos_globales and not codigos:
            operacion = "listar"

    valor_referencia = datos.get("referencia")
    referencia_cruda = (
        str(valor_referencia).strip().lower()
        if isinstance(valor_referencia, str) and valor_referencia.strip()
        else None
    )
    referencia_modelo_explicita = referencia_cruda in REFERENCIAS_VALIDAS
    referencia = referencia_cruda or "ninguna"
    referencia = referencia if referencia in REFERENCIAS_VALIDAS else "ninguna"
    if codigo_producto_modelo_resuelto:
        referencia = "ninguna"
    if producto_atributo_explicito:
        referencia = "ninguna"
        operacion = "detalle"

    es_consulta_general = bool(
        re.search(
            r"\b(qu[eé]\s+(productos?|referencias?|art[ií]culos?)|cu[aá]ntos\s+productos|cu[aá]ntas\s+referencias|cu[aá]ntas\s+unidades\s+(?:disponibles?\s+)?(?:tenemos|hay)\s+en\s+(?:total|el\s+inventario)|total\s+de\s+unidades|unidades\s+en\s+total|suma\s+de\s+(unidades|existencias)|(?:lista|listado|cat[aá]logo)\s+(?:(?:complet[oa]|general|total|actual)\s+)?(?:de\s+(?:tod[oa]s?\s+(?:los|las)\s+)?(?:productos?|art[ií]culos?|referencias?|materiales?|insumos?|items?)|del?\s+(?:inventario|almac[eé]n|bodega))|(?:mu[eé]stra(?:me)?|mu[eé]streme|mostrame|mostrar|ens[eé]ña(?:me)?|ver|dame|dime|lista(?:r|me)?|consultar?)\s+(?:todo\s+)?el\s+(?:inventario|cat[aá]logo)(?:\s+completo)?|todo\s+el\s+(inventario|cat[aá]logo)|todos\s+los\s+productos|lista(r)?\s+todo)\b",
            txt_low,
        )
    )
    pide_suma_global_de_unidades = bool(
        _es_consulta_global_inventario(texto_usuario)
        and re.search(r"\b(?:cu[aá]nt\w*|total|suma)\b", txt_low)
        and re.search(r"\b(?:unidad\w*|stock|existencia\w*)\b", txt_low)
        and not re.search(
            r"\b(?:cu[aá]nt[oa]s?\s+(?:productos?|art[ií]culos?|referencias?|categor[ií]as?)|"
            r"cantidad\s+de\s+(?:productos?|referencias?))\b",
            txt_low,
        )
    )
    if pide_suma_global_de_unidades:
        operacion = "sumar_unidades"
        campos = ["stock"]
        filtros.texto = None
        codigos = []
        filtros.codigos = []
        copio_foco = False
        referencia = "ninguna"
    txt_sin_distributivo = re.sub(
        r"\b(?:bajo|debajo\s+de|cerca\s+de|pr[oó]xim[oa]s?\s+a|en|encima\s+de)\s+su\s+(?:stock\s+)?m[ií]nimo\b",
        "",
        txt_low,
    )
    es_anaforico_foco = bool(
        re.search(
            r"\b(su|sus|este|ese|aquel|dicho|del mismo|d[oó]nde\s+(?:se\s+)?(?:encuentra[n]?|est[aá][n]?|guarda[n]?|ubica[n]?)|cu[aá]nto\s+(?:stock\s+)?tiene|cu[aá]ntas?\s+(?:unidades\s+)?tiene|cu[aá]nto\s+queda|cu[aá]ntas\s+quedan|qui[eé]n\s+(?:lo|la|se|le)?\s*tiene|qui[eé]n\s+se\s+lo\s+llev[oó]|a\s+qui[eé]n\s+se\s+(?:le\s+)?prest[oó]|cu[aá]l\s+es\s+su)\b",
            txt_sin_distributivo if es_consulta_general or re.search(r"\b(qu[eé]|cu[aá]les|cu[aá]nt[oa]s?)\b", txt_low) else txt_low,
        )
    )
    filtro_extremal_actual = bool(
        filtros.texto or codigos or filtros.categoria or filtros.almacen
        or filtros.ubicacion
        # Positive stock is the ranked metric, not a current-turn set filter.
        or (filtros.condicion and filtros.condicion != "con_stock")
    )
    if operacion in {"mayor_stock", "menor_stock"}:
        # Resolve the scope structurally: when this turn supplies the set's
        # filters, a pronoun inside the same sentence refers to that set. Only
        # an extremum with no current-turn set may consume conversational
        # results (or a model-selected reference supported by the utterance).
        referencia = (
            "ninguna" if filtro_extremal_actual
            else "resultados_previos" if tiene_referencia_resultados_previos(texto_usuario)
            else "ninguna"
        )
    elif (es_consulta_general or pregunta_listar_categorias or pregunta_contar_categorias or (pregunta_prestamos_globales and not codigos)) and not tiene_referencia_resultados_previos(texto_usuario):
        referencia = "ninguna"
    elif (
        filtros.condicion and not codigos and not filtros.texto
        and not es_anaforico_foco and referencia != "foco"
        and operacion not in {"mayor_stock", "menor_stock"}
    ):
        referencia = "ninguna"
        if tipo == "consulta":
            operacion = "listar"
    elif operacion in ("mayor_stock", "menor_stock"):
        referencia = "resultados_previos" if re.search(r"\b(de\s+(?:esos|ellos|ellas|los\s+anteriores|las\s+anteriores)|entre\s+(?:ellos|esos|ellas))\b", txt_low) else "ninguna"
    elif not codigos and not filtros.texto and not filtros.categoria:
        if re.search(r"\b(cu[aá]l\s+(de\s+(esos|ellos)\s+)?tiene\s+(m[aá]s|menos|mayor|menor)|cu[aá]l\s+es\s+el\s+que\s+tiene\s+(m[aá]s|menos)|cu[aá]l\s+es\s+la\s+diferencia|comp[aá]ralos|entre\s+(ellos|esos))\b", txt_low):
            referencia = "resultados_previos"
            if tipo == "consulta":
                operacion = "comparar"
        elif re.search(r"\b(cu[aá]les\s+son|cu[aá]les\s+de\s+(ellos|esos)|de\s+(esos|ellos)|mu[eé]stramelos)\b", txt_low):
            referencia = "resultados_previos"
        elif (
            referencia == "ninguna"
            and not referencia_modelo_explicita
            and not es_patron_colectivo_plural
            and not filtros.almacen
            and not filtros.ubicacion
        ):
            if (
                copio_foco
                or bool(campos)
                or operacion in ("detalle", "explicar_estado")
                or es_anaforico_foco
                or re.search(r"\b(su|sus|este|ese|aquel|dicho|del mismo|d[oó]nde\s+(?:se\s+)?(?:encuentra[n]?|est[aá][n]?|guarda[n]?|ubica[n]?)|c[oó]mo\s+se\s+llama|cu[aá]nto\s+queda|cu[aá]ntas\s+quedan|cu[aá]nto\s+stock\s+tiene|cu[aá]ntas?\s+tiene|qui[eé]n\s+lo\s+tiene|a\s+qui[eé]n\s+se\s+le\s+prest[oó]|tiene\s+stock)\b", txt_low)
            ):
                referencia = "foco"
                if tipo in ("consulta", "fuera_dominio"):
                    tipo = "consulta"
                    operacion = "detalle"
                    if filtros.condicion in ("con_stock", "minimo_cero"):
                        filtros.condicion = None

    if (
        referencia == "resultados_previos"
        and not tiene_referencia_resultados_previos(texto_usuario)
        and re.search(
            r"\b(?:el|la|los|las)\s+(?:lista|listado|catalogo|coleccion|tabla)\s+(?:de|del)\b",
            txt_norm,
        )
    ):
        referencia = "ninguna"

    conjunto_definido_en_turno = filtro_extremal_actual and operacion in {"mayor_stock", "menor_stock"}
    if (
        tipo == "consulta"
        and tiene_referencia_resultados_previos(texto_usuario)
        and not conjunto_definido_en_turno
    ):
        referencia = "resultados_previos"

    if referencia == "foco" and tipo == "consulta":
        if operacion in ("listar", "contar", "sumar_unidades", None):
            operacion = "detalle"
        if filtros.condicion in ("con_stock", "minimo_cero"):
            filtros.condicion = None

    if tipo == "consulta" and alcance_temporal == "evento":
        entidad_unica = bool(codigos or filtros.texto or referencia == "foco")
        if entidad_unica and operacion in ("listar", "contar", "sumar_unidades", "comparar", None):
            operacion = "detalle"

    if tipo == "fuera_dominio":
        if any(k in txt_norm for k in ("inventario", "stock", "producto", "almacen", "ubicacion", "referencia", "prestamo", "movimiento", "categoria", "cuanto", "queda", "quedan")):
            if campos or codigos or filtros.texto or filtros.categoria or filtros.condicion or referencia != "ninguna":
                tipo = "consulta"
                if not operacion:
                    operacion = "detalle" if (codigos or referencia == "foco") else "listar"
            elif pide_capacidades_o_ayuda:
                tipo = "capacidades"
        elif campos or codigos or referencia in ("foco", "resultados_previos"):
            tipo = "consulta"
            if not operacion:
                operacion = "detalle"

    # Podado estricto de campos no solicitados para evitar respuestas sobredimensionadas
    es_restrictivo = bool(re.search(r"\b(solo|solamente|[uú]nicamente|nada\s+m[aá]s|no\s+me\s+des\s+m[aá]s)\b", txt_low))
    no_quiere_info = bool(re.search(r"\b(no\s+(?:me\s+)?des\s+(?:m[aá]s\s+)?informaci[oó]n|no\s+m[aá]s\s+datos)\b", txt_low))
    pide_ficha_o_info = bool(
        re.search(
            r"\b(ficha|detalle|detalles|todo|completo|completa|informaci[oó]n|datos|h[aá]blame|cu[eé]ntame|descr[ií]beme|qu[eé]\s+tienes\s+(?:registrado\s+)?de)\b",
            txt_low,
        )
    ) and not no_quiere_info
    pide_movimientos = bool(
        re.search(
            r"\b(movimientos?|historial|transacciones|entradas?|salidas?|pr[eé]stamos?|devoluciones?|traslados?|dotaciones?|ajustes?|pas[oó]|"
            r"movi[oó]|movieron|prest[oó]|prestaron|devolvi[oó]|devolvieron|entr[oó]|entraron|ingres[oó]|ingresaron|sali[oó]|salieron|traslad[oó]|trasladaron)\b",
            txt_low,
        )
        or "movimientos" in campos
        or alcance_temporal in ("evento", "ambos")
    )
    if fecha_desde or fecha_hasta:
        if pide_movimientos:
            alcance_temporal = "evento"
            if "movimientos" not in campos:
                campos.append("movimientos")
        else:
            # A date filter without a history request is unsafe: discard it
            # instead of silently changing a current-stock question.
            fecha_desde = fecha_hasta = None

    if es_restrictivo or no_quiere_info:
        campos_restringidos = []
        if re.search(r"\b(?:c[oó]digo\s+de\s+barras|c[oó]digo\s+barra|barcode)\b", txt_low):
            campos_restringidos.append("codigo_barras")
        if re.search(r"\bc[oó]digos?\b(?!\s+(?:de\s+)?barras?)", txt_low):
            campos_restringidos.append("codigo")
        if re.search(r"\bnombres?\b", txt_low):
            campos_restringidos.append("nombre")
        if re.search(r"\bcategor[ií]as?\b", txt_low) and not categoria_explicita:
            campos_restringidos.append("categoria")
        if pregunta_valor_minimo or re.search(r"\b(?:m[ií]nim[oa]s?|umbral)\b", txt_low):
            campos_restringidos.append("stock_minimo")
        txt_sin_stock_minimo = re.sub(r"\bstock\s+m[ií]nim[oa]s?\b", " ", txt_low)
        if re.search(r"\b(?:stock|cu[aá]nt[oa]s?|quedan?|existencias?|unidades?|disponibles?)\b", txt_sin_stock_minimo):
            campos_restringidos.append("stock")
        if any(k in txt_norm for k in ("donde", "ubicacion", "estante", "lugar")):
            campos_restringidos.append("ubicacion")
        if re.search(r"\b(?:almac[eé]n|bodega)\b", txt_low) and not almacen_raw:
            campos_restringidos.append("almacen")
        if re.search(r"\b(?:pr[eé]stamos?|prestad[oa]s?|pendientes?)\b", txt_low):
            campos_restringidos.append("prestamos")
        if campos_restringidos:
            campos = list(dict.fromkeys(campos_restringidos))
        else:
            campos = [c for c in campos if c != "ficha"]
    elif operacion in ("listar", "contar", "sumar_unidades", "mayor_stock", "menor_stock") and not (pregunta_listar_categorias or pregunta_contar_categorias):
        campos = [
            c for c in campos
            if c in CAMPOS_VALIDOS and (c != "stock_minimo" or _texto_menciona_stock_minimo(texto_usuario))
        ]
    else:
        # Si hay al menos un atributo puntual solicitado y el usuario no pidió la ficha completa ni movimientos, podarlos
        if "stock_minimo" in campos and not _texto_menciona_stock_minimo(texto_usuario):
            campos = [c for c in campos if c != "stock_minimo"]
        campos_puntuales = [c for c in campos if c not in ("ficha", "movimientos")]
        if campos_puntuales:
            if not pide_ficha_o_info and "ficha" in campos:
                campos = [c for c in campos if c != "ficha"]
            if not pide_movimientos and "movimientos" in campos:
                campos = [c for c in campos if c != "movimientos"]
        # No volver a inferir los atributos mediante subcadenas del mensaje.
        # El modelo ya los expresa con claves del esquema y la allowlist los valida.

    accion = (_str(datos.get("accion")) or _str(datos.get("operacion"))) if tipo == "operacion" else None
    accion = accion if accion in ACCIONES_VALIDAS else None
    if tipo == "operacion":
        if re.search(r"\b(?:sac[aáeio]|sacar|saque[ns]?|retir[aáeio]|retirar|salida)\b", txt_low) and not re.search(r"\b(ingres|entrad|devolv|devuelv|pr[eé]st)\b", txt_low):
            accion = "salida"
        elif re.search(r"\b(?:ingres[aáeio]|ingresar|entrad[aá]|meter)\b", txt_low) and not re.search(r"\b(sac[aáeio]|retir|salida)\b", txt_low):
            accion = "entrada"
        elif re.search(r"\b(?:pr[eé]st[aáeio]|prestar|pr[eé]stamo)\b", txt_low) and not re.search(r"\b(devolv|devuelv|devoluci)\b", txt_low):
            accion = "prestamo"
        elif re.search(r"\b(?:devolv[aáeio]|devolver|devuelv[aáeio]|devoluci[oó]n|regres[aáeio])\b", txt_low):
            accion = "devolucion"
        elif re.search(r"\b(?:traslad[aáeio]|trasladar|traslado|moviliz[aáeio]|reubic[aáeio])\b", txt_low):
            accion = "traslado"
        elif re.search(r"\b(?:crear|registrar|alta\s+de)\s+(?:un\s+)?(?:producto|art[ií]culo)\b", txt_low) or re.search(r"\b(?:nuevo\s+producto|alta\s+producto)\b", txt_low):
            accion = "crear_producto"
        elif re.search(r"\bstock\s+m[ií]nimo\b", txt_low):
            accion = "cambiar_stock_minimo"

    cantidad = _int(datos.get("cantidad"))
    if tipo == "operacion" and accion in ("prestamo", "devolucion", "entrada", "salida", "traslado"):
        from ..lenguaje.numeros import extraer_cantidad_operacion_texto
        cant_texto = extraer_cantidad_operacion_texto(texto_usuario)
        if cant_texto is not None and cant_texto > 0:
            cantidad = cant_texto
        elif cantidad is not None:
            # Si el LLM propuso una cantidad que no está anclada en el mensaje (ej. sacó 2 de TAL-002 o 20 de 20V),
            # descartarla para preguntar explícitamente cuántas unidades desea operar.
            cantidad = None

    valor = _int(datos.get("valor"))
    if accion == "cambiar_stock_minimo":
        m_val = re.search(r"\b(?:a|en)\s+(\d+)\b", txt_low)
        if m_val:
            valor = int(m_val.group(1))
        elif valor is None:
            valor = _int(f.get("umbral")) or _int(f.get("stock_minimo")) or _int(datos.get("stock_minimo"))
    elif accion == "crear_producto":
        m_val = re.search(r"\bstock\s+m[ií]nimo\s*(?:de|en|:)?\s*(-?\d+)\b", txt_low)
        if m_val:
            valor = int(m_val.group(1))
        elif valor is None:
            valor = _int(f.get("stock_minimo")) or _int(datos.get("stock_minimo"))
        # En altas, una cantidad mencionada explícitamente con esta etiqueta
        # prevalece sobre cualquier número que el modelo haya puesto en cantidad.
        m_stk = re.search(r"\b(?:stock\s+inicial|stock|cantidad\s+inicial)\s*(?:de|en|:)?\s*(-?\d+)\b", txt_low)
        if m_stk:
            cantidad = int(m_stk.group(1))
        else:
            # Ignora "cantidad: 1" de ejemplos o valores por defecto del LLM.
            # Sin etiqueta o campo stock_inicial explícito, la alta debe preguntar.
            cantidad = _int(datos.get("stock_inicial"))

    destinatario = _str(datos.get("destinatario"))
    if tipo == "operacion" and accion in ("prestamo", "salida") and destinatario:
        if not _destinatario_anclado_en_mensaje(texto_usuario, destinatario):
            destinatario = None
    if not destinatario and tipo == "operacion":
        m_dest = re.search(
            r"\b(?:a\s+nombre\s+de|a\s+favor\s+de|entregar\s+a|entregado\s+a|para|a)\s+([a-zA-Z0-9ñÑáéíóúÁÉÍÓÚ\s'.-]+?)(?=\s+(?:por|para|con|en|motivo)\b|[?!,;:]|$)",
            texto_usuario,
            re.IGNORECASE,
        )
        if m_dest:
            dest_cand = m_dest.group(1).strip(" .,!?:;")
            dest_cand = re.sub(r"^(?:el|la|los|las|don|doña)\s+", "", dest_cand, flags=re.IGNORECASE).strip()
            if (
                dest_cand.lower() not in ("mi", "ti", "el", "la", "los", "las", "almacen", "bodega", "mantenimiento", "inventario", "uso", "consumo")
                and _destinatario_anclado_en_mensaje(texto_usuario, dest_cand)
            ):
                destinatario = dest_cand

    pregunta_limpia = _str(datos.get("pregunta"))
    if tipo == "aclaracion" and pregunta_limpia and "?" not in pregunta_limpia and "¿" not in pregunta_limpia:
        pregunta_limpia = None

    producto_op = _str(datos.get("producto"))
    if tipo == "operacion":
        tiene_ref_op = tiene_referencia_contextual_operacion(texto_pre)
        if producto_op:
            cod_op = normalizar_codigo_producto(producto_op)
            if cod_op:
                if cod_op not in codigos_en_mensaje:
                    producto_op = None
            else:
                toks_p = tokens_significativos(producto_op)
                if not toks_p or not any(
                    t in tokens_usuario or any(u.startswith(t[:3]) for u in tokens_usuario if len(u) >= 3 and len(t) >= 3)
                    for t in toks_p
                ):
                    producto_op = None
        if accion in ("entrada", "salida", "prestamo", "devolucion", "traslado"):
            literal = re.sub(
                r"^(?:(?:por\s+favor|quiero|necesito)\s+)*(?:ingresar|ingresa|ingrese|meter|registrar\s+(?:una\s+)?entrada|dar\s+entrada|registrar\s+(?:una\s+)?salida|sacar|retirar|despachar|prestar|pr[eé]stamo|devolver|registrar\s+(?:una\s+)?devoluci[oó]n|trasladar|hacer\s+(?:un\s+)?traslado)\s*",
                "", texto_pre, flags=re.IGNORECASE,
            ).strip()
            if literal != texto_pre:
                literal = re.sub(r"^(?:de|del|el|la|un|una)\s+", "", literal, flags=re.IGNORECASE)
                literal = re.sub(r"^\d+\s*(?:unidades?\s*)?(?:de\s+)?", "", literal, flags=re.IGNORECASE)
                literal = re.sub(r"^(?:un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s+(?:unidades?\s*)?(?:de\s+)?", "", literal, flags=re.IGNORECASE)
                if destinatario:
                    literal = re.sub(rf"\s+para\s+{re.escape(destinatario)}\s*$", "", literal, flags=re.IGNORECASE)
                incluye_receptor = bool(
                    destinatario and producto_op
                    and re.search(rf"\s+para\s+{re.escape(destinatario)}\s*$", producto_op, re.IGNORECASE)
                )
                cantidad_al_final = bool(
                    cantidad is not None and producto_op
                    and re.search(rf"\s+{cantidad}\s*$", producto_op)
                )
                if literal and (not producto_op or incluye_receptor or cantidad_al_final):
                    producto_op = literal.strip()
        elif accion in ("habilitar", "inhabilitar", "editar_producto", "cambiar_stock_minimo"):
            literal = re.sub(
                r"^(?:(?:por\s+favor|quiero|necesito|puedes?)\s+)*(?:habilitar|habilita|reactivar|reactiva|"
                r"inhabilitar|inhabilita|desactivar|desactiva|editar|edita|modificar|modifica|actualizar|actualiza|"
                r"cambiar|cambia)\s+(?:(?:el|la|un|una)\s+)?(?:(?:producto|art[ií]culo)\s+)?",
                "", texto_pre, flags=re.IGNORECASE,
            ).strip()
            literal = re.sub(r"\s+(?:por\s+favor|en\s+el\s+inventario)$", "", literal, flags=re.IGNORECASE)
            if accion == "cambiar_stock_minimo":
                m_producto_cambio = re.search(
                    r"\b(?:cambio\s+de|cambiar|cambia)\s+stock\s+m[ií]nimo\s+(?:a|para|de|del)\s+"
                    r"(?:(?:el|la|los|las|un|una)\s+)?(.+)$",
                    texto_pre,
                    re.IGNORECASE,
                )
                if m_producto_cambio:
                    candidato = m_producto_cambio.group(1).strip(" .,!?:;")
                    if candidato and not re.fullmatch(r"\d+", candidato):
                        literal = candidato
            if literal and literal != texto_pre and (not producto_op or normalizar(producto_op) not in normalizar(texto_usuario)):
                producto_op = literal
        if not codigos and not filtros.texto and not producto_op:
            referencia = "foco" if tiene_ref_op else "ninguna"

    # Los candidatos del catálogo ayudan a identificar menciones, pero no son
    # filtros implícitos para una consulta que pide un conjunto completo.
    es_consulta_de_conjunto = bool(
        tipo == "consulta"
        and operacion in ("listar", "contar", "sumar_unidades")
        and referencia not in ("foco", "resultados_previos")
        and (
            filtros.categoria
            or filtros.condicion
            or filtros.almacen
            or filtros.ubicacion
            or filtros.categoria_generica
            or filtros.es_prestable is not None
        )
        and not filtros.texto
        and not codigos_en_mensaje
    )
    if es_consulta_de_conjunto:
        codigos = []
        filtros.codigos = []
        copio_foco = False

    # Un filtro de colección no debe ocultar una entidad explícita durante un
    # detalle; sus atributos permiten responder también si cumple el estado.
    if (
        tipo == "consulta"
        and operacion in ("detalle", "sumar_unidades")
        and (codigos or filtros.codigos or referencia == "foco")
        and filtros.condicion
    ):
        filtros.condicion = None

    # Una pregunta de cantidad sobre una colección condicionada pide el número
    # de registros, aunque el modelo la haya clasificado como enumeración.
    if (
        tipo == "consulta"
        and operacion == "listar"
        and (filtros.condicion or filtros.es_prestable is not None)
        and not campos
        and re.search(r"\bcu[aá]nt[oa]s?\b", txt_low)
    ):
        operacion = "contar"

    plan = PlanAgente(
        tipo=tipo,
        operacion=operacion,
        filtros=filtros,
        referencia=referencia,
        alcance=alcance_temporal,
        tipo_movimiento=tipo_movimiento,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
        campos=campos,
        medida_solicitada=medida_solicitada,
        alcance_entidad=alcance_entidad,
        accion=accion,
        producto=producto_op,
        cantidad=cantidad,
        destinatario=destinatario,
        valor=valor,
        destino=_str(datos.get("destino")),
        pregunta=pregunta_limpia,
        crudo=datos,
    )
    relacion_stock_raw = _str(datos.get("relacion_stock"))
    relacion_stock_plan = (
        relacion_stock_raw.casefold()
        if relacion_stock_raw and relacion_stock_raw.casefold() in RELACIONES_STOCK_VALIDAS
        else None
    )
    condicion_stock_plan_prelim = _RELACION_A_CONDICION_STOCK.get(relacion_stock_plan)
    if condicion_stock_plan_prelim and not _condicion_stock_tiene_indicio_textual(texto_usuario, condicion_stock_plan_prelim):
        if condicion_stock_plan_prelim == "suficiente" and _condicion_stock_tiene_indicio_textual(texto_usuario, "con_stock"):
            relacion_stock_plan = "con_stock"
        else:
            relacion_stock_plan = "sin_condicion"
            if filtros.condicion == condicion_stock_plan_prelim:
                filtros.condicion = None
    # No marcar como "verificada" una condición de umbral inferida solo por el
    # JSON primario del LLM sin evidencia explícita: permite que la fase de
    # verificación semántica y _revalidar_condicion_stock_final la auditen.
    cond_prelim_post = _RELACION_A_CONDICION_STOCK.get(relacion_stock_plan)
    if (
        cond_prelim_post in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA
        and cond_prelim_post not in condiciones_stock_explicitas
    ):
        plan.relacion_stock_verificada = None
    else:
        plan.relacion_stock_verificada = relacion_stock_plan
    filtro_texto_es_entidad = datos.get("filtro_texto_es_entidad")
    if isinstance(filtro_texto_es_entidad, bool) and _str(f.get("texto")):
        if filtro_texto_es_entidad is False and codigos_productos_candidatos:
            candidatos_val = _obtener_productos_candidatos_para_verificacion(
                codigos_productos_candidatos,
            )
            if (
                not plan.filtros.condicion
                and plan.filtros.es_prestable is None
                and _filtro_texto_anclado_a_candidato(plan.filtros.texto, candidatos_val)
            ):
                filtro_texto_es_entidad = True
        plan.filtro_texto_es_entidad_verificado = filtro_texto_es_entidad
    atributos_stock_raw = datos.get("atributos_stock_solicitados")
    if isinstance(atributos_stock_raw, list):
        plan.atributos_stock_verificados = sorted({
            str(campo).strip().casefold()
            for campo in atributos_stock_raw
            if str(campo).strip().casefold() in {"stock", "stock_minimo"}
            and (str(campo).strip().casefold() != "stock_minimo" or _texto_menciona_stock_minimo(texto_usuario))
        })
    condicion_stock_plan = _RELACION_A_CONDICION_STOCK.get(relacion_stock_plan)
    if (
        tipo == "consulta"
        and not condicion_stock_plan
        and not filtros.condicion
        and plan.atributos_stock_verificados
    ):
        if (
            "stock_minimo" in plan.atributos_stock_verificados
            and "stock_minimo" not in plan.campos
            and re.search(r"\b(?:m[ií]nim[oa]s?|umbral)\b", txt_low)
        ):
            plan.campos.append("stock_minimo")
        txt_sin_min = re.sub(r"\bstock\s+m[ií]nim[oa]s?\b", " ", txt_low)
        if (
            "stock" in plan.atributos_stock_verificados
            and "stock" not in plan.campos
            and re.search(r"\b(?:stock|existencias?|disponibles?|unidades?|cu[aá]nt[oa]s?)\b", txt_sin_min)
        ):
            plan.campos.append("stock")
    bloquea_con_stock_sobre_entidad = bool(
        condicion_stock_plan == "con_stock"
        and (
            es_consulta_especifica_producto
            or bool(filtros.codigos)
            or referencia == "foco"
            or alcance_entidad == "producto"
            or (bool(filtros.texto) and not es_consulta_colectiva_disponibles)
        )
    )
    if (
        tipo == "consulta"
        and condicion_stock_plan
        and alcance_temporal != "evento"
        and not bloquea_con_stock_sobre_entidad
        and not (pregunta_prestamos_globales and filtros.condicion == "con_prestamos")
    ):
        filtros.condicion = condicion_stock_plan
        entidad_individual_plan = bool(
            filtros.codigos
            or _producto_es_entidad_concreta(producto_op)
            or referencia in {"foco", "resultados_previos"}
            or alcance_entidad == "producto"
        )
        if not entidad_individual_plan:
            alcance_entidad = "coleccion"
            atributos_confirmados = plan.atributos_stock_verificados
            if atributos_confirmados is not None:
                campos_ajenos_stock = [
                    campo for campo in plan.campos
                    if campo not in {"stock", "stock_minimo"}
                ]
                plan.campos = list(dict.fromkeys(
                    campos_ajenos_stock + atributos_confirmados
                ))
            if not atributos_confirmados and operacion in {
                "detalle", "sumar_unidades", "comparar",
            }:
                operacion = "listar"
                plan.operacion = "listar"
                plan.medida_solicitada = "ninguna"
                plan.campos = [
                    campo for campo in plan.campos
                    if campo not in {"stock", "stock_minimo"}
                ]
            plan.alcance_entidad = "coleccion"
    plan.alcance_entidad = alcance_entidad
    ubicaciones_raw = datos.get("ubicaciones_comparadas")
    if isinstance(ubicaciones_raw, str):
        ubicaciones_raw = [ubicaciones_raw]
    if isinstance(ubicaciones_raw, list):
        plan.ubicaciones_comparadas = [
            valor.strip() for valor in ubicaciones_raw
            if isinstance(valor, str) and valor.strip()
        ][:4]
    if (
        plan.tipo in {"social", "fuera_dominio"}
        and codigos_productos_candidatos
        and re.search(r"\b(?:d[oó]nde|ubicaci[oó]n)\b", txt_low)
    ):
        productos_candidatos = _obtener_productos_candidatos_para_verificacion(
            codigos_productos_candidatos,
        )
        tokens_mensaje = set(tokens_significativos(texto_pre))
        entidad_respaldada = any(
            tokens_mensaje & set(tokens_significativos(nombre))
            for _, nombre in productos_candidatos
        )
        if entidad_respaldada:
            # Una petición de ubicación anclada a un producto visible sigue
            # siendo consulta aunque el tono regional haga que el LLM la lea
            # como charla o mensaje externo.
            plan.tipo = "consulta"
            plan.operacion = "detalle"
            plan.referencia = "ninguna"
            if not plan.filtros.codigos and not plan.filtros.texto:
                plan.filtros.texto = _extraer_objeto_consulta_sin_marco(texto_pre)
            plan = _normalizar_peticion_atributo(texto_pre, plan)
            plan.alcance_entidad = "producto"
    if plan.tipo == "consulta" and plan.operacion is None:
        plan.operacion = "detalle" if (plan.filtros.codigos or plan.referencia == "foco") else "listar"
    if plan.tipo == "operacion" and plan.accion is None:
        return None
    ubicaciones_raw = datos.get("ubicaciones_comparadas")
    if isinstance(ubicaciones_raw, str):
        ubicaciones_raw = [ubicaciones_raw]
    if isinstance(ubicaciones_raw, list):
        plan.ubicaciones_comparadas = [
            valor.strip() for valor in ubicaciones_raw
            if isinstance(valor, str) and valor.strip()
        ][:4]
    plan = _normalizar_alcance_entidad_por_morfologia(texto_usuario, plan)
    if (
        plan.tipo == "consulta"
        and plan.operacion not in {"mayor_stock", "menor_stock"}
        and pide_atributo_stock
        and set(plan.campos or []).issubset({"stock"})
    ):
        if plan.alcance_entidad == "producto":
            pregunta_atributo_individual = bool(re.search(
                r"\b(?:cu[aá]nto\s+(?:stock|existencias?|unidades?)\s+(?:tiene|hay|queda)|"
                r"qu[eé]\s+(?:stock|existencias?)\s+tiene|cu[aá]nto\s+queda)\b",
                txt_low,
            ))
            plan.operacion = (
                "detalle"
                if plan.filtros.codigos or plan.referencia == "foco" or pregunta_atributo_individual
                else "sumar_unidades"
            )
        elif plan.alcance_entidad == "coleccion" and not plan.filtros.codigos:
            plan.operacion = "sumar_unidades"
    return plan


def obtener_plan_cache(
    texto: str,
    contexto_txt: str,
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Recupera del micro-caché un plan previamente validado si no ha expirado."""
    clave = (normalizar(preprocesar_texto_inventario(texto)), contexto_txt)
    entrada = _CACHE_PLANES.get(clave)
    if not entrada:
        return None
    ts, datos_crudos = entrada
    if (time.monotonic() - ts) > _CACHE_TTL_SEGUNDOS:
        _CACHE_PLANES.pop(clave, None)
        return None
    if (
        isinstance(datos_crudos, dict)
        and datos_crudos.get("_haku_cache_final_version") == 1
        and isinstance(datos_crudos.get("plan"), dict)
    ):
        datos_plan = copy.deepcopy(datos_crudos["plan"])
        filtros_raw = datos_plan.get("filtros")
        if not isinstance(filtros_raw, dict):
            return None
        campos_plan = set(PlanAgente.__dataclass_fields__)
        datos_plan = {
            nombre: valor
            for nombre, valor in datos_plan.items()
            if nombre in campos_plan and nombre != "filtros"
        }
        campos_filtros = set(FiltrosConsulta.__dataclass_fields__)
        filtros = {
            nombre: valor
            for nombre, valor in filtros_raw.items()
            if nombre in campos_filtros
        }
        try:
            plan_final = PlanAgente(
                **datos_plan,
                filtros=FiltrosConsulta(**filtros),
            )
        except (TypeError, ValueError):
            _CACHE_PLANES.pop(clave, None)
            return None
        if plan_final.tipo != "consulta":
            _CACHE_PLANES.pop(clave, None)
            return None
        plan_final.fuente_plan = "cache_plan"
        plan_final.llm_ms = 0.0
        plan_final.llm_cola_ms = 0.0
        plan_final.chars_prompt = 0
        plan_final.tokens_estimados_prompt = 0
        plan_final.tokens_completados = 0
        plan_final._cache_final_verificado = True
        return plan_final
    plan = validar_plan(copy.deepcopy(datos_crudos), texto, codigos_productos_candidatos)
    if plan is not None:
        plan.fuente_plan = "cache_plan"
        plan.llm_ms = 0.0
    return plan


def limpiar_cache_planes() -> None:
    """Limpia el micro-caché de planes validados cuando cambia el catálogo o estructura."""
    _CACHE_PLANES.clear()


def guardar_plan_cache(texto: str, contexto_txt: str, plan_final: PlanAgente) -> None:
    """Almacena solo consultas ya resueltas para repetirlas sin reclasificar."""
    if not isinstance(plan_final, PlanAgente) or plan_final.tipo != "consulta":
        return
    datos_cache = {
        "_haku_cache_final_version": 1,
        "plan": asdict(plan_final),
    }
    ahora = time.monotonic()
    if len(_CACHE_PLANES) >= _CACHE_MAX_ENTRADAS:
        expiradas = [k for k, (ts, _) in _CACHE_PLANES.items() if (ahora - ts) > _CACHE_TTL_SEGUNDOS]
        for k in expiradas:
            _CACHE_PLANES.pop(k, None)
        if len(_CACHE_PLANES) >= _CACHE_MAX_ENTRADAS:
            mas_antigua = min(_CACHE_PLANES.items(), key=lambda kv: kv[1][0])[0]
            _CACHE_PLANES.pop(mas_antigua, None)
    clave = (normalizar(preprocesar_texto_inventario(texto)), contexto_txt)
    _CACHE_PLANES[clave] = (ahora, datos_cache)


def _completar_almacenes_explicitos(
    plan: PlanAgente,
    texto: str,
    almacenes_accesibles: List[str],
) -> PlanAgente:
    """Resuelve origen/destino de traslados usando nombres reales y accesibles."""
    if plan.tipo != "operacion" or plan.accion != "traslado" or not almacenes_accesibles:
        return plan

    texto_norm = normalizar(texto)
    menciones = []
    for nombre in sorted(set(almacenes_accesibles), key=lambda valor: len(normalizar(valor)), reverse=True):
        nombre_norm = normalizar(nombre)
        variantes = [nombre_norm]
        alias = re.sub(r"^(?:almacen|bodega)\s+", "", nombre_norm).strip()
        # Alias textuales solo si tienen longitud suficiente para evitar que una
        # letra de almacén se confunda con una preposición o artículo.
        if len(alias) >= 3 and alias not in variantes:
            variantes.append(alias)
        for variante in variantes:
            for coincidencia in re.finditer(rf"(?<!\w){re.escape(variante)}(?!\w)", texto_norm):
                prefijo = texto_norm[max(0, coincidencia.start() - 24):coincidencia.start()]
                enlace = re.search(r"\b(desde|del|de|hacia|al|a|para)\s+(?:(?:el|la)\s+)?$", prefijo)
                if not enlace:
                    continue
                rol = "origen" if enlace.group(1) in ("desde", "del", "de") else "destino"
                item = (coincidencia.start(), coincidencia.end(), nombre, rol)
                if item not in menciones:
                    menciones.append(item)

    # Conserva la coincidencia más larga cuando un almacén comparte nombre con
    # otro, y evita asignar un único almacén como ambos extremos del traslado.
    menciones.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    seleccionadas = []
    for item in menciones:
        if any(item[0] < previo[1] and previo[0] < item[1] for previo in seleccionadas):
            continue
        seleccionadas.append(item)

    origenes = [item[2] for item in seleccionadas if item[3] == "origen"]
    destinos = [item[2] for item in seleccionadas if item[3] == "destino"]
    if origenes:
        plan.filtros.almacen = origenes[0]
    if destinos:
        plan.destino = destinos[-1]
    return plan


def _completar_ubicacion_explicita(
    plan: PlanAgente,
    texto: str,
    ubicaciones_accesibles: List[str],
    almacenes_accesibles: Optional[List[str]] = None,
    categorias_accesibles: Optional[List[str]] = None,
) -> PlanAgente:
    """Vincula menciones de ubicación con el catálogo actual, sin nombres fijos."""
    if plan.tipo != "consulta":
        return plan

    # Una etiqueta de almacén alucinada por el modelo no puede restringir o
    # falsear la consulta. Se compara con las entidades actuales mencionadas
    # por el usuario antes de llegar a la herramienta de lectura.
    if plan.filtros.almacen and almacenes_accesibles:
        texto_norm = f" {normalizar(texto)} "
        almacenes_mencionados = set(_menciones_almacenes_catalogo(texto, almacenes_accesibles))
        for almacen in almacenes_accesibles:
            alias = re.sub(r"^(?:almacen|bodega)\s+", "", normalizar(almacen)).strip()
            if len(alias) >= 3 and f" {alias} " in texto_norm:
                almacenes_mencionados.add(almacen)

        menciones_norm = {normalizar(nombre) for nombre in almacenes_mencionados}
        alias_mencionados = {
            re.sub(r"^(?:almacen|bodega)\s+", "", normalizar(nombre)).strip()
            for nombre in almacenes_mencionados
        }
        partes_filtro = [parte.strip() for parte in plan.filtros.almacen.split("|") if parte.strip()]
        coincide_con_mencion = bool(partes_filtro) and all(
            normalizar(parte) in menciones_norm or normalizar(parte) in alias_mencionados
            for parte in partes_filtro
        )
        menciona_tipo_almacen = bool(re.search(r"\b(?:almacen|bodega|deposito|sucursal)\b", texto_norm))
        nombre_filtro_en_texto = bool(partes_filtro) and all(
            re.search(rf"(?<!\w){re.escape(normalizar(parte))}(?!\w)", texto_norm)
            for parte in partes_filtro
        )

        if not coincide_con_mencion:
            if menciona_tipo_almacen and not nombre_filtro_en_texto:
                plan.tipo = "aclaracion"
                plan.operacion = None
                plan.filtros.almacen = None
                plan.pregunta = "¿En qué almacén desea consultar la información?"
                plan.referencia = "ninguna"
                plan.campos = []
                return plan
            if not menciona_tipo_almacen:
                plan.filtros.almacen = None

    if not ubicaciones_accesibles:
        menciones = []
    else:
        menciones = _resolver_menciones_ubicacion(texto, ubicaciones_accesibles)

    # Si el usuario nombra una ubicación registrada fuera de su ámbito, no se
    # puede dejar que el modelo la omita y convierta la consulta en una lista
    # de otro almacén. Se conserva el almacén propietario para que la misma
    # comprobación de permisos de la herramienta rechace el acceso sin leer
    # sus productos.
    if not menciones:
        ubicaciones_registradas = _nombres_ubicaciones_registradas()
        ubicaciones_mencionadas = _resolver_menciones_ubicacion(texto, ubicaciones_registradas)
        accesibles_norm = {normalizar(nombre) for nombre in ubicaciones_accesibles or []}
        if len(ubicaciones_mencionadas) == 1 and normalizar(ubicaciones_mencionadas[0]) not in accesibles_norm:
            propietarios = _almacenes_propietarios_de_ubicacion(ubicaciones_mencionadas[0])
            if len(propietarios) == 1:
                menciones = ubicaciones_mencionadas
                plan.filtros.almacen = propietarios[0]
            elif len(propietarios) > 1:
                plan.tipo = "aclaracion"
                plan.operacion = None
                plan.filtros = FiltrosConsulta()
                plan.campos = []
                plan.referencia = "ninguna"
                plan.pregunta = "¿En cuál almacén se encuentra esa ubicación?"
                return plan

    if plan.filtros.ubicacion:
        ubicacion_en_mensaje = bool(
            _resolver_menciones_ubicacion(texto, [plan.filtros.ubicacion])
        )
        referencia_de_contexto = bool(
            plan.referencia in {"foco", "resultados_previos"}
            and _mensaje_requiere_referencia_contextual(texto)
        )
        if not ubicacion_en_mensaje and not referencia_de_contexto:
            # Un estante citado solo en el catálogo/contexto no limita una
            # consulta por nombre de producto. El filtro espacial debe venir
            # del mensaje o de un seguimiento contextual explícito.
            plan.filtros.ubicacion = None

    if len(menciones) != 1:
        return plan
    plan.filtros.ubicacion = menciones[0]
    # El modelo puede repetir la etiqueta de ubicación en el filtro de
    # producto (por ejemplo, texto="A1" y ubicación="A1"). Si todo el filtro
    # ya queda explicado por la entidad dinámica resuelta, no debe estrechar
    # la búsqueda de productos a un nombre literal "A1".
    if plan.filtros.texto and not plan.filtros.codigos:
        filtro_es_ubicacion = bool(
            _resolver_menciones_ubicacion(plan.filtros.texto, [menciones[0]])
        )
        terminos_ubicacion = set(tokens_significativos(menciones[0]))
        terminos_modelo = set(tokens_significativos(plan.filtros.texto))
        terminos_marco_ubicacion = set(_TERMINOS_UBICACION_GENERICOS)
        if filtro_es_ubicacion or (
            terminos_modelo
            and terminos_modelo.issubset(terminos_ubicacion | terminos_marco_ubicacion)
        ):
            plan.filtros.texto = None

    pregunta_contenido_ubicacion = bool(re.search(
        r"\b(?:qu[eé]\s+(?:(?:productos?|art[ií]culos?|referencias?|vainas?|cosas?)\s+)?(?:hay|contiene|tiene)|"
        r"muestra(?:me)?|lista(?:me|r)?|ver)\b",
        texto,
        re.IGNORECASE,
    ))
    if pregunta_contenido_ubicacion and not plan.filtros.codigos:
        terminos_ubicacion = set(tokens_significativos(menciones[0]))
        terminos_modelo = set(tokens_significativos(plan.filtros.texto)) if plan.filtros.texto else set()
        terminos_genericos = {
            "que", "hay", "producto", "productos", "articulo", "articulos", "referencia", "referencias",
            "vaina", "vainas", "cosa", "cosas", "muestra", "muestrame", "lista", "listame", "listar",
            "ver", "dame", "contiene", "tiene", "tienen", "inventario",
        }
        if not terminos_modelo or terminos_modelo.issubset(terminos_ubicacion | terminos_genericos):
            plan.filtros.texto = None
            if not plan.filtros.categoria_explicita:
                plan.filtros.categoria = None
            plan.referencia = "ninguna"
            plan.operacion = "listar"
    plan = _normalizar_categoria_con_catalogo(plan, categorias_accesibles or [], texto)
    return plan


def _nombres_ubicaciones_registradas() -> List[str]:
    """Obtiene nombres vigentes de ubicación, también para resolver permisos."""
    try:
        from inventario.models import Ubicacion
        return list(
            Ubicacion.objects.filter(
                activo=True,
                eliminado=False,
                almacen__activo=True,
                almacen__eliminado=False,
            ).order_by("nombre").values_list("nombre", flat=True).distinct()
        )
    except Exception:
        return []


def _almacenes_propietarios_de_ubicacion(nombre: str) -> List[str]:
    """Resuelve el almacén de una ubicación vigente sin enumerar sus productos."""
    try:
        from inventario.models import Ubicacion
        return list(
            Ubicacion.objects.filter(
                nombre__iexact=nombre,
                activo=True,
                eliminado=False,
                almacen__activo=True,
                almacen__eliminado=False,
            ).order_by("almacen__nombre").values_list("almacen__nombre", flat=True).distinct()
        )
    except Exception:
        return []


def _normalizar_comparacion_ubicaciones(texto: str, plan: PlanAgente) -> PlanAgente:
    """Resuelve las dos ubicaciones de una comparación contra el catálogo vivo."""
    if plan.tipo != "consulta":
        return plan

    nombres_ubicaciones = _nombres_ubicaciones_registradas()
    menciones = _resolver_menciones_ubicacion(texto, nombres_ubicaciones)
    declara_comparacion = (
        plan.operacion == "comparar" or len(plan.ubicaciones_comparadas or []) == 2
    )
    if len(menciones) == 2 and not declara_comparacion:
        plan.tipo = "aclaracion"
        plan.operacion = None
        plan.filtros = FiltrosConsulta()
        plan.ubicaciones_comparadas = []
        plan.campos = []
        plan.alcance = None
        plan.tipo_movimiento = None
        plan.referencia = "ninguna"
        plan.pregunta = (
            "Identifiqué dos ubicaciones, pero no pude interpretar cómo desea consultarlas. "
            "¿Quiere comparar los productos asociados a ambas o buscar un producto concreto?"
        )
        return plan
    if not declara_comparacion:
        # Los modelos pueden devolver una ubicación aislada en el campo de
        # comparación incluso cuando la pregunta no compara ubicaciones. Ese
        # campo solo es válido para una comparación explícita de dos lugares.
        plan.ubicaciones_comparadas = []
        return plan

    texto_norm = normalizar(texto or "")
    menciones.sort(
        key=lambda nombre: texto_norm.find(normalizar(nombre))
        if normalizar(nombre) in texto_norm else len(texto_norm)
    )
    if len(menciones) != 2 or normalizar(menciones[0]) == normalizar(menciones[1]):
        plan.tipo = "aclaracion"
        plan.operacion = None
        plan.ubicaciones_comparadas = []
        plan.filtros = FiltrosConsulta()
        plan.campos = []
        plan.referencia = "ninguna"
        plan.pregunta = "¿Cuáles son las dos ubicaciones que desea comparar?"
        return plan

    plan.ubicaciones_comparadas = menciones
    plan.filtros.ubicacion = None

    # Una ubicación en el campo de producto contradice el plan estructurado.
    # No se interpreta por vocabulario: se valida contra las etiquetas vigentes
    # y se pide aclaración antes de buscar con un filtro corrupto o ampliarlo.
    ubicaciones_en_filtro = _resolver_menciones_ubicacion(
        plan.filtros.texto or "", nombres_ubicaciones,
    )
    if ubicaciones_en_filtro:
        plan.tipo = "aclaracion"
        plan.operacion = None
        plan.filtros = FiltrosConsulta()
        plan.ubicaciones_comparadas = []
        plan.campos = []
        plan.alcance = None
        plan.tipo_movimiento = None
        plan.referencia = "ninguna"
        plan.pregunta = (
            "El filtro del producto se mezcló con las ubicaciones. "
            "¿Desea comparar todos los productos asociados a ambas o limitar la comparación a un producto?"
        )
        return plan

    # La comparación la define la entidad estructurada reconocida por el modelo;
    # no se decide mediante palabras sueltas de la frase.
    plan.operacion = "comparar"

    # Una comparación no hereda un evento temporal si el plan no pidió consultar
    # movimientos. Los campos solicitados por el modelo se conservan.
    if "movimientos" not in (plan.campos or []):
        plan.alcance = None
        plan.tipo_movimiento = None
    return plan


def _normalizar_comparacion_sin_objetivos(
    texto: str,
    plan: PlanAgente,
    ubicaciones_accesibles: Optional[List[str]] = None,
) -> PlanAgente:
    """Repara un `comparar` que en realidad filtra una colección por estado actual.

    Se guía por campos tipados del plan y por entidades actuales del catálogo,
    no por una frase, producto o ubicación enumerados en el código.
    """
    if (
        plan.tipo != "consulta"
        or plan.operacion != "comparar"
        or plan.filtros.condicion not in CONDICIONES_VALIDAS
        or plan.alcance_entidad not in {"global", "coleccion"}
        or plan.referencia in {"foco", "resultados_previos"}
        or plan.filtros.texto
        or plan.filtros.codigos
        or plan.filtros.categoria
        or plan.filtros.almacen
        or plan.filtros.ubicacion
    ):
        return plan

    objetivos = list(dict.fromkeys(
        list(ubicaciones_accesibles or []) + _nombres_ubicaciones_registradas()
    ))
    menciones = _resolver_menciones_ubicacion(texto or "", objetivos or [])
    if len(menciones) >= 2:
        return plan

    # Una condición de stock define qué filas pertenecen al conjunto pedido.
    # Sin dos ubicaciones declaradas, el operador de comparación carece de
    # objetivos y debe normalizarse al listado filtrado.
    plan.operacion = "listar"
    plan.alcance_entidad = "coleccion"
    plan.referencia = "ninguna"
    plan.ubicaciones_comparadas = []
    plan.alcance = "actual"
    plan.tipo_movimiento = None
    return plan


def _normalizar_categoria_con_catalogo(
    plan: PlanAgente,
    categorias_accesibles: List[str],
    texto: Optional[str] = None,
) -> PlanAgente:
    """Resuelve categorías mencionadas contra los nombres vigentes del catálogo."""
    if plan.tipo != "consulta" or not categorias_accesibles:
        return plan

    categoria = plan.filtros.categoria
    if texto:
        menciones = _menciones_catalogo(texto, categorias_accesibles)
        texto_norm_cat = normalizar(preprocesar_texto_inventario(texto))
        menciona_palabra_categoria = bool(re.search(r"\bcategorias?\b", texto_norm_cat))
        if len(menciones) == 1:
            cat_menc = menciones[0]
            # Si el usuario no dijo "categoría" y hay tokens distintivos de producto
            # (p. ej. "baterías AA" frente a la categoría "Baterías"), no sustituyas
            # el descriptor del producto por la categoría completa.
            palabras_texto_resto = []
            if not menciona_palabra_categoria:
                cat_norm_m = normalizar(cat_menc)
                tras_cat = re.search(rf"(?<!\w){re.escape(cat_norm_m)}\s+([a-z0-9]{{1,12}})\b", texto_norm_cat)
                palabras_funcion = {
                    "hay", "quedan", "queda", "tenemos", "tienen", "estan", "esta", "son",
                    "en", "de", "del", "con", "sin", "disponibles", "disponible", "agotadas",
                    "agotados", "bajo", "cerca", "por", "para", "almacen", "bodega", "inventario",
                    "prestables", "consumibles", "registradas", "registrados", "actualmente",
                }
                if tras_cat and tras_cat.group(1) not in palabras_funcion:
                    palabras_texto_resto.append(tras_cat.group(1))
            if menciona_palabra_categoria or not palabras_texto_resto:
                categoria = cat_menc
                plan.filtros.categoria = categoria
                plan.filtros.categoria_explicita = True
            elif not plan.filtros.texto and not plan.filtros.codigos:
                # Preservar el descriptor específico ("baterías AA") para que el
                # catálogo resuelva el producto exacto en lugar de toda la categoría.
                plan.filtros.texto = f"{cat_menc} {' '.join(palabras_texto_resto)}".strip()
                if plan.filtros.categoria and normalizar(plan.filtros.categoria) == normalizar(cat_menc):
                    plan.filtros.categoria = None
                    plan.filtros.categoria_explicita = False
                    categoria = None
        elif len(menciones) > 1 and not categoria:
            # Una sola casilla de categoría no puede representar una unión.
            # El planificador conserva su intención; no elijamos una categoría
            # arbitrariamente por su posición en el catálogo.
            return plan
        elif not menciones and not categoria:
            # Resuelve menciones abreviadas de una categoría viva (p. ej. prefijo
            # sin un complemento coordinado) únicamente cuando sus raíces
            # significativas (>= 2) identifican de forma única a una sola
            # categoría accesible del catálogo vigente.
            texto_cat_norm = normalizar(preprocesar_texto_inventario(texto))
            match_cat = re.search(
                r"\b(?:(?:productos?|articulos?|referencias?|elementos?|items?)\s+de\s+(?:(?:la|el)\s+)?(?:categoria\s+)?|"
                r"categoria\s+(?:de\s+)?)"
                r"([a-z0-9][a-z0-9\s]{2,80}?)"
                r"(?=\s+(?:hay|estan?|tienen?|en|dentro\s+de|registrad\w*|disponibles?|agotad\w*|bajo|cerca|con|sin|que|cual\w*)\b|$)",
                texto_cat_norm,
            )
            if match_cat:
                raices_frase = {
                    _raiz(token) for token in tokens_significativos(match_cat.group(1))
                    if token
                }
                if len(raices_frase) >= 2:
                    candidatas_unicas = [
                        nombre for nombre in categorias_accesibles
                        if raices_frase.issubset(
                            {_raiz(token) for token in tokens_significativos(nombre) if token}
                        )
                    ]
                    if len(candidatas_unicas) == 1:
                        categoria = candidatas_unicas[0]
                        plan.filtros.categoria = categoria
                        plan.filtros.categoria_explicita = True
                        if plan.filtros.texto:
                            restantes = {
                                _raiz(token)
                                for token in tokens_significativos(plan.filtros.texto)
                                if normalizar(token) not in ("categoria", "categorias")
                            } - {
                                _raiz(token) for token in tokens_significativos(categoria)
                            }
                            if not restantes:
                                plan.filtros.texto = None

    if not categoria:
        return plan

    categoria_norm = normalizar(categoria)
    coincidencias = [
        nombre for nombre in categorias_accesibles
        if normalizar(nombre) == categoria_norm
    ]
    if not coincidencias:
        raices = {_raiz(token) for token in categoria_norm.split() if token}
        if raices:
            coincidencias = [
                nombre for nombre in categorias_accesibles
                if raices.issubset({_raiz(token) for token in normalizar(nombre).split() if token})
            ]

    if len(coincidencias) == 1:
        plan.filtros.categoria = coincidencias[0]
        if texto and _menciones_catalogo(texto, [coincidencias[0]]):
            plan.filtros.categoria_explicita = True
            if plan.filtros.texto:
                restantes = {
                    _raiz(token)
                    for token in tokens_significativos(plan.filtros.texto)
                    if normalizar(token) not in ("categoria", "categorias")
                } - {
                    _raiz(token) for token in tokens_significativos(coincidencias[0])
                }
                if not restantes:
                    plan.filtros.texto = None
        return plan
    if plan.filtros.categoria_explicita:
        return plan

    termino = categoria.strip()
    plan.filtros.categoria = None
    plan.filtros.categoria_generica = False
    if not plan.filtros.texto:
        plan.filtros.texto = termino
    elif normalizar(termino) not in normalizar(plan.filtros.texto):
        plan.filtros.texto = f"{plan.filtros.texto} {termino}".strip()
    return plan


def _completar_nombre_producto_alta(
    plan: PlanAgente,
    texto: str,
    categorias: Optional[List[str]] = None,
    almacenes: Optional[List[str]] = None,
    ubicaciones: Optional[List[str]] = None,
) -> PlanAgente:
    """Conserva el nombre completo dictado al crear un producto.

    Los límites de categoría, almacén y ubicación se resuelven con sus valores
    vigentes; solo se usan nombres que el usuario mencionó en este turno.
    """
    if plan.tipo != "operacion" or plan.accion != "crear_producto" or not texto:
        return plan

    inicio = re.search(
        r"\b(?:crear|crea|cree|registrar|registra|registre|dar\s+de\s+alta)\s+"
        r"(?:(?:un|el)\s+)?(?:nuevo\s+)?(?:producto|art[ií]culo)\b\s*[:\-]?\s*",
        texto,
        flags=re.IGNORECASE,
    )
    if not inicio:
        return plan

    inicio_nombre = inicio.end()
    marcador_nombre = re.search(
        r"\b(?:llamad[oa]s?|denominad[oa]s?|de\s+nombre|con\s+nombre|"
        r"cuyo\s+nombre\s+sea|que\s+se\s+llame)\s+",
        texto[inicio_nombre:],
        flags=re.IGNORECASE,
    )
    if marcador_nombre:
        inicio_nombre += marcador_nombre.end()
    limites = []
    patrones_campos = (
        r"\b(?:en\s+la\s+categor[ií]a|categor[ií]a|categor[ií]as)\b",
        r"\b(?:c[oó]digo\s+de\s+barras|c[oó]digo\s+interno|stock\s+(?:inicial|m[ií]nimo)|"
        r"cantidad\s+inicial|existencias?\s+iniciales?|descripci[oó]n)\b",
        r"\bcon\s+(?:un\s+)?stock\s+(?:inicial|m[ií]nimo)\b",
    )
    for patron in patrones_campos:
        limites.extend(m.start() for m in re.finditer(patron, texto, flags=re.IGNORECASE) if m.start() >= inicio_nombre)

    entidades = [
        *((nombre, "categoria") for nombre in categorias or []),
        *((nombre, "almacen") for nombre in almacenes or []),
        *((nombre, "ubicacion") for nombre in ubicaciones or []),
    ]
    for nombre, tipo_entidad in sorted(
        entidades, key=lambda item: len(normalizar(item[0])), reverse=True,
    ):
        if not nombre:
            continue
        for coincidencia in re.finditer(rf"(?<!\w){re.escape(nombre)}(?!\w)", texto, flags=re.IGNORECASE):
            if coincidencia.start() < inicio_nombre:
                continue
            prefijo = texto[inicio_nombre:coincidencia.start()]
            enlace = re.search(
                r"(?:\b(?:en|dentro\s+de|de|del|desde|hacia|a|al)\s+|"
                r"\b(?:la\s+)?categor[ií]a\s+(?:de\s+)?)$",
                prefijo,
                flags=re.IGNORECASE,
            )
            if enlace:
                limites.append(inicio_nombre + enlace.start())
                break

    fin_nombre = min((pos for pos in limites if pos >= inicio_nombre), default=None)
    # Sin un límite de campo ni un marcador que señale un nombre, solo se
    # expande el sufijo completo cuando el modelo ya confirmó una parte de ese
    # mismo nombre (evita recortar nombres compuestos a su último código/sufijo
    # sin inventar nombres en órdenes genéricas como «crear un producto nuevo»).
    if fin_nombre is None and not marcador_nombre:
        sufijo_crudo = texto[inicio_nombre:].strip(" \t\r\n:;,.-—")
        producto_actual_norm = normalizar(plan.producto or "")
        sufijo_norm = normalizar(sufijo_crudo)
        if not (
            _producto_es_entidad_concreta(plan.producto)
            and producto_actual_norm
            and sufijo_norm
            and producto_actual_norm in sufijo_norm
            and len(sufijo_norm) > len(producto_actual_norm)
        ):
            return plan
    fin_nombre = fin_nombre if fin_nombre is not None else len(texto)
    nombre = texto[inicio_nombre:fin_nombre].strip(" \t\r\n:;,.-—")
    nombre = re.sub(r"^(?:llamado|llamada|denominado|denominada|con\s+nombre)\s+", "", nombre, flags=re.IGNORECASE)
    nombre = re.sub(r"\s+", " ", nombre).strip()
    if not nombre:
        return plan

    plan.producto = nombre
    if isinstance(plan.crudo, dict):
        plan.crudo["producto"] = nombre
        plan.crudo["nombre"] = nombre
    return plan


def _completar_filtro_producto_explicito(
    plan: PlanAgente,
    texto: str,
    almacenes_accesibles: List[str],
    ubicaciones_accesibles: List[str],
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> PlanAgente:
    """Recupera el nombre de producto del mensaje cuando el plan lo omitió.

    La extracción usa el texto original y nombres de almacén/ubicación vigentes;
    no contiene referencias de productos ni ubicaciones fijas.
    """
    # Los planes de capacidades/aclaración no representan una búsqueda de
    # producto; no se deben convertir en filtros a partir de palabras del turno.
    if plan.tipo != "consulta":
        return plan

    # Si el modelo convirtió una descripción en un código que la persona no
    # pronunció, no dejes que la búsqueda lo trate como identificador literal.
    # El código no se puede deducir de un nombre; vuelve al descriptor del turno
    # y deja que el catálogo actual resuelva la entidad.
    if plan.filtros.texto and not plan.filtros.codigos:
        codigo_inferido = normalizar_codigo_producto(plan.filtros.texto)
        codigos_mencionados = set(extraer_codigos_de_texto(texto))
        if codigo_inferido and codigo_inferido not in codigos_mencionados:
            objeto_mencionado = _extraer_objeto_consulta_sin_marco(texto)
            plan.filtros.texto = (
                objeto_mencionado
                if objeto_mencionado and not normalizar_codigo_producto(objeto_mencionado)
                else None
            )

    if plan.filtro_texto_es_entidad_verificado is False:
        candidatos_anclaje = _obtener_productos_candidatos_para_verificacion(
            codigos_productos_candidatos,
        )
        texto_anclado_a_producto = bool(
            plan.filtros.texto
            and not plan.filtros.condicion
            and plan.filtros.es_prestable is None
            and not _es_consulta_global_inventario(texto)
            and _filtro_texto_anclado_a_candidato(plan.filtros.texto, candidatos_anclaje)
        )
        if not texto_anclado_a_producto:
            # El clasificador de relación tuvo evidencia suficiente para concluir
            # que el filtro preliminar era lenguaje funcional, no una entidad. No
            # reconstruyas ese mismo fragmento como producto más adelante.
            plan.filtros.texto = None
            if plan.filtros.condicion and plan.operacion in {"listar", "contar", "sumar_unidades"}:
                plan.alcance_entidad = "coleccion"
            return plan
        plan.filtro_texto_es_entidad_verificado = True

    # El foco solo se reemplaza por una entidad nominal explícita. Las
    # referencias a una colección anterior siempre conservan su alcance.
    if plan.referencia == "resultados_previos":
        return plan
    if plan.referencia == "foco":
        entidad_turno = _extraer_objeto_consulta_sin_marco(texto)
        if _mensaje_requiere_referencia_contextual(texto) or not entidad_turno:
            return plan
        tokens_entidad = normalizar(entidad_turno).split()
        if tokens_entidad and tokens_entidad[-1] in _PALABRAS_VACIAS:
            # The extractor may return an incomplete complement from a
            # correction (for example, the clause before the requested field).
            # A trailing grammatical connector is not evidence of a new item.
            # Keep the model's contextual reference; complete product names
            # and descriptors still go through the live catalogue matcher.
            return plan
        plan.referencia = "ninguna"

    if (
        _plan_es_consulta_global_sin_entidad(plan, texto)
        and _es_filtro_marco_global(plan.filtros.texto)
    ):
        plan.filtros.texto = None
        return plan

    # La validación del LLM ocurre antes de resolver nombres de producto a
    # códigos. Al completar esa resolución, un filtro de colección no debe
    # ocultar el detalle solicitado para la referencia individual.
    if (
        plan.tipo == "consulta"
        and plan.operacion == "detalle"
        and plan.filtros.codigos
        and plan.filtros.condicion
    ):
        plan.filtros.condicion = None

    # Si el modelo repite un código de los candidatos contextuales pero deja
    # vacío el filtro, no aceptes esa referencia singular como respuesta a una
    # pregunta colectiva. Reconstruye el descriptor solo con términos del
    # mensaje respaldados por los nombres vigentes del catálogo. La búsqueda
    # dinámica volverá a resolver el conjunto completo, incluso si supera los
    # candidatos mostrados al modelo.
    filtros_crudos = plan.crudo.get("filtros") if isinstance(plan.crudo, dict) else None
    producto_crudo = filtros_crudos.get("producto") if isinstance(filtros_crudos, dict) else None
    if not producto_crudo and isinstance(plan.crudo, dict):
        producto_crudo = plan.crudo.get("producto")
    if not producto_crudo and isinstance(filtros_crudos, dict):
        codigos_crudos = filtros_crudos.get("codigos")
        if isinstance(codigos_crudos, (list, tuple)):
            producto_crudo = " ".join(str(codigo) for codigo in codigos_crudos if codigo)
        elif codigos_crudos:
            producto_crudo = str(codigos_crudos)
    if (
        plan.tipo == "consulta"
        and plan.referencia == "ninguna"
        and not plan.filtros.codigos
        and not plan.filtros.texto
        and not plan.filtros.categoria
        and not plan.filtros.categoria_explicita
        and not _es_consulta_global_inventario(texto)
        and producto_crudo
    ):
        filtro_reconstruido = _reconstruir_filtro_eco_candidato(
            texto,
            str(producto_crudo),
            codigos_productos_candidatos or [],
        )
        if filtro_reconstruido:
            plan.filtros.texto = filtro_reconstruido
            plan.filtros.codigos = []
            plan.referencia = "ninguna"
            if plan.operacion in {"detalle", "contar"}:
                plan.operacion = "listar"
                plan.campos = []
                plan = _normalizar_intencion_conteo(texto, plan)

    if plan.filtros.texto:
        entidades_filtro = []
        almacenes_mencionados = _menciones_almacenes_catalogo(texto, almacenes_accesibles or [])
        filtros_almacen = {
            normalizar(parte)
            for parte in (plan.filtros.almacen or "").split("|")
            if parte.strip()
        }
        entidades_filtro.extend(
            nombre for nombre in almacenes_mencionados
            if normalizar(nombre) in filtros_almacen
        )
        ubicaciones_mencionadas = _resolver_menciones_ubicacion(
            texto, ubicaciones_accesibles or [],
        )
        filtro_ubicaciones = {
            normalizar(nombre) for nombre in (ubicaciones_mencionadas or [])
        }
        entidades_filtro.extend(
            nombre for nombre in ubicaciones_mencionadas
            if normalizar(nombre) in filtro_ubicaciones
            and normalizar(nombre) == normalizar(plan.filtros.ubicacion or "")
        )
        filtro_limpio = _quitar_entidades_estructuradas_del_filtro(
            plan.filtros.texto, entidades_filtro,
        )
        if filtro_limpio != normalizar(plan.filtros.texto):
            plan.filtros.texto = filtro_limpio or None

    # El modelo a veces copia la etiqueta completa de un candidato contextual
    # (código, nombre y categoría) al filtro. Si el usuario no mencionó ese
    # código, reconstruimos el descriptor con términos del mensaje que estén
    # respaldados por los nombres actuales de varios candidatos.
    filtro_reconstruido = _reconstruir_filtro_eco_candidato(
        texto,
        plan.filtros.texto,
        codigos_productos_candidatos or [],
    )
    if filtro_reconstruido:
        plan.filtros.texto = filtro_reconstruido
        plan.filtros.codigos = []
        if not plan.filtros.categoria_explicita:
            plan.filtros.categoria = None
            plan.filtros.categoria_generica = False
        plan.referencia = "ninguna"

    # El modelo puede conservar solo una parte de una descripción y perder el
    # sustantivo o el valor que la distingue. Reconstruye el sintagma desde el
    # mensaje cuando todos los términos reconocidos aparecen allí; el buscador
    # coteja después el descriptor completo con el catálogo vigente.
    if (
        plan.tipo == "consulta"
        and plan.operacion in {"detalle", "listar", "sumar_unidades"}
        and plan.filtros.texto
        and not plan.filtros.codigos
        and not plan.filtros.categoria
        and not plan.filtros.categoria_explicita
    ):
        objeto_mencionado = _extraer_objeto_consulta_sin_marco(texto)
        if objeto_mencionado:
            filtro_completo = _preservar_sintagma_nominal(
                plan.filtros.texto,
                preprocesar_texto_inventario(objeto_mencionado, normalizar_medidas=False),
            )
            raices_plan = {_raiz(token) for token in tokens_significativos(plan.filtros.texto)}
            raices_recuperadas = {_raiz(token) for token in tokens_significativos(filtro_completo)}
            if (
                raices_plan
                and raices_plan.issubset(raices_recuperadas)
                and len(raices_recuperadas) > len(raices_plan)
            ):
                plan.filtros.texto = filtro_completo

    filtros_crudos = plan.crudo.get("filtros") if isinstance(plan.crudo, dict) else None
    codigos_crudos = filtros_crudos.get("codigos") if isinstance(filtros_crudos, dict) else []
    if isinstance(codigos_crudos, str):
        codigos_crudos = [codigos_crudos]
    productos_crudos = [
        filtros_crudos.get("producto") if isinstance(filtros_crudos, dict) else None,
        plan.crudo.get("producto") if isinstance(plan.crudo, dict) else None,
    ]
    candidatos_norm = {
        codigo
        for valor in codigos_productos_candidatos or []
        if (codigo := normalizar_codigo_producto(str(valor)))
    }
    codigos_usuario = set(extraer_codigos_de_texto(texto))
    eco_codigo_candidato_descartado = any(
        (codigo := normalizar_codigo_producto(str(valor)))
        and codigo in candidatos_norm
        and codigo not in codigos_usuario
        for valor in [*(codigos_crudos or []), *productos_crudos]
    )

    if _es_consulta_ubicacion_eliptica(texto):
        plan.tipo = "consulta"
        plan.operacion = "detalle"
        plan.filtros = FiltrosConsulta()
        plan.campos = ["ubicacion"]
        plan.referencia = "foco"
        return plan

    filtro_modelo_con_marco_almacen = bool(re.search(
        r"\b(?:almacen|bodega)(?:\s+y\s+(?:la\s+)?ubicacion)?\s+"
        r"(?:(?:se)\s+)?(?:encuentra|encuentran|esta|estan|guarda|guardan|ubica|ubican|almacena|almacenan)\b",
        normalizar(plan.filtros.texto or ""),
    ))

    if (
        plan.tipo != "consulta"
        or plan.operacion not in ("detalle", "listar", "sumar_unidades")
        or (plan.filtros.texto and not filtro_modelo_con_marco_almacen)
        or plan.filtros.codigos
        or plan.filtros.categoria
        or plan.filtros.categoria_explicita
        or (plan.filtros.condicion and not eco_codigo_candidato_descartado)
        or (plan.filtros.es_prestable is not None and not eco_codigo_candidato_descartado)
    ):
        return plan

    campos_producto = {
        "stock", "stock_minimo", "ubicacion", "almacen", "codigo_barras",
        "descripcion", "es_prestable", "ficha", "movimientos",
    }
    pide_atributo = bool(set(plan.campos or []) & campos_producto) or bool(re.search(
        r"\b(?:stock|existencias?|unidades?|disponibles?|ubicaci[oó]n|d[oó]nde|cu[aá]nt[oa]s?)\b",
        texto,
        re.IGNORECASE,
    )) or bool(
        plan.filtros.vacio()
        and re.match(
            r"^\s*(?:busca(?:me|r)?|encuentra|localiza|consulta(?:r)?|muestra(?:me)?|ver|revisa(?:r)?)\s+\S",
            normalizar(preprocesar_texto_inventario(texto or "")),
        )
    )
    texto_sin_entidades_de_contexto = texto or ""
    entidades = sorted(
        set(almacenes_accesibles or []) | set(ubicaciones_accesibles or []),
        key=lambda valor: len(normalizar(valor)),
        reverse=True,
    )
    for nombre in entidades:
        if not nombre:
            continue
        texto_sin_entidades_de_contexto = re.sub(
            rf"(?<!\w){re.escape(nombre)}(?!\w)",
            " ",
            texto_sin_entidades_de_contexto,
            flags=re.IGNORECASE,
        )

    objeto = _extraer_objeto_consulta_sin_marco(texto_sin_entidades_de_contexto)
    if not objeto or _es_filtro_ubicacion_generico(objeto):
        return plan
    if (
        _plan_es_consulta_global_sin_entidad(plan, texto)
        and _es_filtro_marco_global(objeto)
    ):
        return plan

    tokens_objeto = [
        token for token in normalizar(objeto).split()
        if token not in _PALABRAS_VACIAS
    ]
    nucleo_objeto = tokens_objeto[0] if tokens_objeto else ""
    nucleo_plural = bool(nucleo_objeto and _raiz(nucleo_objeto) != nucleo_objeto)
    if not pide_atributo and nucleo_plural:
        # Un sustantivo cabeza plural expresa una colección. Recupera el
        # sintagma nominal desde el mensaje sin depender de que el LLM haya
        # completado `filtros.texto` o clasificado la tarea como listado.
        plan.filtros.texto = objeto
        plan.operacion = "listar"
        plan.alcance = "actual"
        plan.alcance_entidad = "coleccion"
        plan.campos = []
        plan.referencia = "ninguna"
        return plan
    if not pide_atributo:
        return plan

    # La extracción del núcleo puede descartar modificadores que el LLM omitió
    # (o que otra normalización ya retiró). Reconstruye el sintagma completo a
    # partir del mensaje original antes de ejecutar la búsqueda de catálogo.
    texto_para_sintagma = re.sub(
        r"\b(?:en|dentro\s+de)\s*$",
        " ",
        texto_sin_entidades_de_contexto,
        flags=re.IGNORECASE,
    )
    texto_normalizado = preprocesar_texto_inventario(texto_para_sintagma, normalizar_medidas=False)
    plan.filtros.texto = _preservar_sintagma_nominal(objeto, texto_normalizado)
    plan.referencia = "ninguna"
    return plan


def _reconstruir_filtro_eco_candidato(
    texto_usuario: str,
    filtro_modelo: Optional[str],
    codigos_productos_candidatos: List[str],
) -> Optional[str]:
    """Deshace una copia de candidatos del contexto y conserva el lenguaje del usuario."""
    if not filtro_modelo or not codigos_productos_candidatos:
        return None

    codigos_candidatos = {
        str(codigo).strip().upper()
        for codigo in codigos_productos_candidatos
        if codigo
    }
    codigos_filtro = set(extraer_codigos_de_texto(filtro_modelo))
    codigos_usuario = set(extraer_codigos_de_texto(texto_usuario))

    try:
        from inventario.models import Producto
        nombres = list(
            Producto.objects.filter(
                codigo_interno__in=codigos_candidatos,
                activo=True,
                archivado=False,
            ).values_list("nombre", flat=True)
        )
    except Exception:
        return None
    if not nombres:
        return None

    filtro_norm = normalizar(filtro_modelo)
    mensaje_norm = normalizar(texto_usuario)
    codigo_copiado = bool((codigos_filtro & codigos_candidatos) - codigos_usuario)
    nombre_copiado = any(
        normalizar(nombre)
        and normalizar(nombre) in filtro_norm
        and normalizar(nombre) not in mensaje_norm
        for nombre in nombres
    )
    if not codigo_copiado and not nombre_copiado:
        return None

    # Los candidatos se recuperan a partir del propio mensaje del usuario. Si el
    # modelo devuelve su etiqueta como si fuera el filtro literal, se pierde la
    # expresión original (por ejemplo, un sinónimo o una perífrasis). Devuelva
    # esa expresión al buscador dinámico: él vuelve a resolverla contra el
    # catálogo e índice vigentes, sin convertir una sugerencia contextual en
    # una selección irrevocable de producto.
    frecuencias: Dict[str, int] = {}
    for nombre in nombres:
        for token in set(tokens_significativos(nombre)):
            frecuencias[token] = frecuencias.get(token, 0) + 1
    umbral = len(nombres) // 2 + 1
    terminos_catalogo = {
        token for token, frecuencia in frecuencias.items()
        if frecuencia >= umbral
    }
    if not terminos_catalogo:
        return _extraer_objeto_consulta_sin_marco(texto_usuario) if codigo_copiado else None

    terminos_usuario = tokens_significativos(texto_usuario)
    terminos = list(dict.fromkeys(
        token for token in terminos_usuario if token in terminos_catalogo
    ))
    reconstruido = " ".join(terminos) or None
    if reconstruido:
        # La intersección del catálogo identifica el núcleo compartido, pero
        # puede quitar un modificador válido de una variante (tamaño, material,
        # formato, etc.). Recupéralo sintácticamente desde el mensaje original.
        reconstruido = _preservar_sintagma_nominal(
            reconstruido,
            preprocesar_texto_inventario(texto_usuario, normalizar_medidas=False),
        )
    if not reconstruido and codigo_copiado:
        return _extraer_objeto_consulta_sin_marco(texto_usuario)
    return reconstruido


def _quitar_entidades_estructuradas_del_filtro(
    filtro_modelo: str,
    entidades_actuales: List[str],
) -> str:
    """Retira del filtro los almacenes o ubicaciones ya representados en campos propios."""
    limpio = normalizar(filtro_modelo)
    for nombre in sorted(set(entidades_actuales or []), key=lambda valor: len(normalizar(valor)), reverse=True):
        entidad = normalizar(nombre)
        if not entidad:
            continue
        patron = (
            r"(?<!\w)(?:(?:en|dentro\s+de)\s+)?"
            r"(?:(?:el|la|los|las)\s+)?"
            + re.escape(entidad)
            + r"(?!\w)"
        )
        limpio = re.sub(patron, " ", limpio)
    limpio = re.sub(r"\b(?:en|dentro\s+de)\s+(?:el|la|los|las)\b", " ", limpio)
    limpio = re.sub(r"\s+", " ", limpio).strip()
    if _es_filtro_ubicacion_generico(limpio):
        return ""
    return limpio


def _plan_comparacion_ubicaciones_inconsistente(texto: str, plan: PlanAgente) -> bool:
    """Detecta conflictos entre menciones de ubicaciones vivas y campos del plan."""
    if plan.tipo != "consulta":
        return False
    nombres = _nombres_ubicaciones_registradas()
    ubicaciones_mencionadas = _resolver_menciones_ubicacion(texto or "", nombres)
    if len(ubicaciones_mencionadas) != 2:
        return False
    compara = plan.operacion == "comparar" or len(plan.ubicaciones_comparadas or []) == 2
    ubicaciones_en_filtro = _resolver_menciones_ubicacion(plan.filtros.texto or "", nombres)
    return not compara or bool(ubicaciones_en_filtro)


def _reparar_plan_comparacion_ubicaciones(
    texto: str,
    contexto_txt: str,
    cliente_llm: Any,
    plan: PlanAgente,
    timeout: float,
    telemetria: Dict[str, Any],
    codigos_productos_candidatos: Optional[List[str]],
    categorias_accesibles: Optional[List[str]] = None,
    duracion_inicial_ms: float = 0.0,
) -> PlanAgente:
    """Pide al mismo LLM resolver solo los campos contradictorios de una consulta por ubicaciones."""
    if not _plan_comparacion_ubicaciones_inconsistente(texto, plan) or cliente_llm is None:
        return plan

    prompt_reparacion = (
        "Interpreta la intención y el filtro de producto del mensaje usando el contexto. "
        "Las ubicaciones enumeradas son entidades físicas actuales. Devuelve solo "
        "{\"intencion\":\"comparar|aclaracion\",\"filtro_producto\":null|\"nombre o código explícito\","
        "\"campos\":[]}. Para comparar, el filtro_producto es null salvo que se pida además un producto concreto. "
        "Nunca uses una ubicación ni palabras de relación como filtro de producto. No inventes entidades ni respondas."
    )
    contenido = (
        f"[Contexto actual]\n{contexto_txt}\n\n"
        f"[Mensaje original]\n{preprocesar_texto_inventario(texto, normalizar_medidas=False)[:400]}\n\n"
        f"[Plan preliminar]\n{json.dumps(plan.resumen(), ensure_ascii=False)}"
    )
    duracion_reparacion = 0.0
    telemetria["llamadas_fisicas"] = int(telemetria.get("llamadas_fisicas", 0) or 0) + 1
    telemetria["reparacion_plan"] = "iniciada"
    try:
        t0 = time.perf_counter()
        respuesta = cliente_llm.generar(
            mensajes=[
                {"role": "system", "content": prompt_reparacion},
                {"role": "user", "content": contenido},
            ],
            formato_json=True,
            temperatura=0.0,
            max_tokens=64,
            timeout=timeout,
        )
        duracion_reparacion = (time.perf_counter() - t0) * 1000.0
        datos_corregidos = getattr(respuesta, "datos_json", None) if respuesta is not None else None
        nombres_ubicaciones = _nombres_ubicaciones_registradas()
        ubicaciones_mencionadas = _resolver_menciones_ubicacion(texto or "", nombres_ubicaciones)
        texto_norm = normalizar(texto or "")
        ubicaciones_mencionadas.sort(
            key=lambda nombre: texto_norm.find(normalizar(nombre))
            if normalizar(nombre) in texto_norm else len(texto_norm)
        )
        intencion = _str(datos_corregidos.get("intencion")) if isinstance(datos_corregidos, dict) else None
        if intencion == "comparar" and len(ubicaciones_mencionadas) == 2:
            filtro_producto = _str(datos_corregidos.get("filtro_producto"))
            ubicaciones_en_filtro = _resolver_menciones_ubicacion(
                filtro_producto or "", nombres_ubicaciones,
            )
            campos_crudos = datos_corregidos.get("campos")
            campos_corregidos = [
                campo for campo in (_str(valor) for valor in campos_crudos or [])
                if campo in CAMPOS_VALIDOS
            ] if isinstance(campos_crudos, list) else []
            if not ubicaciones_en_filtro and "movimientos" not in campos_corregidos:
                plan.tipo = "consulta"
                plan.operacion = "comparar"
                plan.ubicaciones_comparadas = ubicaciones_mencionadas
                plan.filtros.ubicacion = None
                plan.filtros.texto = filtro_producto
                plan.campos = campos_corregidos
                plan.alcance = None
                plan.tipo_movimiento = None
                plan.referencia = "ninguna"
                plan.pregunta = None
                plan.llm_ms = duracion_inicial_ms + float(
                    getattr(respuesta, "duracion_ms", 0.0) or duracion_reparacion
                )
                plan.llm_cola_ms = float(getattr(respuesta, "espera_cola_ms", 0.0) or 0.0)
                plan.chars_prompt = len(prompt_reparacion) + len(contenido)
                plan.tokens_estimados_prompt = max(1, plan.chars_prompt // 4)
                plan.tokens_completados = int(getattr(respuesta, "tokens_completados", 0) or 0)
                plan.fuente_plan = "llm_reparado"
                plan.crudo = {"reparacion_semantica": datos_corregidos}
                telemetria["duracion_reparacion_ms"] = duracion_reparacion
                telemetria["duracion_ms"] = plan.llm_ms
                if filtro_producto:
                    categoria = next(
                        (nombre for nombre in categorias_accesibles or []
                         if normalizar(nombre) == normalizar(filtro_producto)),
                        None,
                    )
                    if categoria:
                        plan.filtros.categoria = categoria
                        plan.filtros.categoria_explicita = True
                        plan.filtros.texto = None
                codigos_explicitos = extraer_codigos_de_texto(filtro_producto or "")
                if codigos_explicitos:
                    codigos_validos = set(codigos_productos_candidatos or [])
                    if not codigos_validos or set(codigos_explicitos).issubset(codigos_validos):
                        plan.filtros.codigos = codigos_explicitos
                        plan.filtros.texto = None
            telemetria["reparacion_plan"] = "exitosa"
            telemetria["estado"] = "LLM_PLAN_REPARADO"
            telemetria["codigo_http"] = 200
            return plan
        if intencion == "aclaracion":
            plan.tipo = "aclaracion"
            plan.operacion = None
            plan.filtros = FiltrosConsulta()
            plan.ubicaciones_comparadas = []
            plan.campos = []
            plan.pregunta = "¿Desea comparar las dos ubicaciones o consultar el contenido de ambas?"
            plan.fuente_plan = "llm_reparado"
            plan.llm_ms = duracion_inicial_ms + float(
                getattr(respuesta, "duracion_ms", 0.0) or duracion_reparacion
            )
            telemetria["duracion_reparacion_ms"] = duracion_reparacion
            telemetria["duracion_ms"] = plan.llm_ms
            telemetria["reparacion_plan"] = "aclaracion"
            telemetria["estado"] = "LLM_PLAN_REPARADO"
            return plan
        telemetria["reparacion_plan"] = "respuesta_inconsistente"
    except Exception as exc:
        telemetria["reparacion_plan"] = "fallida"
        telemetria["error_reparacion_plan"] = str(exc)
        logger.warning("[HakuAgente] No se pudo reparar el plan de ubicaciones: %s", type(exc).__name__)

    plan.fuente_plan = "llm_reparacion_fallida"
    plan.llm_ms = duracion_inicial_ms + duracion_reparacion
    telemetria["duracion_ms"] = plan.llm_ms
    telemetria["estado"] = "LLM_PLAN_REPARACION_FALLIDA"
    return plan


def verificar_intencion_operativa_sin_entidad(
    texto: str,
    cliente_llm: Any,
    timeout: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Contrasta la tarea cuando el primer plan pudo perder una intención operativa.

    Se usa tanto si faltó resolver una entidad como si el producto está identificado:
    encontrar un código/nombre no demuestra que la petición sea una consulta. Una
    respuesta de consulta conserva el plan original; no sustituye la entidad validada.
    """
    if timeout is None:
        timeout = float(os.environ.get("HAKU_LLM_REQUEST_TIMEOUT_SECONDS", "45"))
    contenido = f"[Mensaje original de la persona]\n{preprocesar_texto_inventario(texto, normalizar_medidas=False)[:500]}"
    chars_prompt = len(PROMPT_VERIFICADOR_INTENCION_OPERATIVA) + len(contenido)
    started = time.perf_counter()
    respuesta = cliente_llm.generar(
        mensajes=[
            {"role": "system", "content": PROMPT_VERIFICADOR_INTENCION_OPERATIVA},
            {"role": "user", "content": contenido},
        ],
        formato_json=True,
        temperatura=0.0,
        max_tokens=128,
        timeout=timeout,
        json_schema=ESQUEMA_VERIFICADOR_INTENCION_OPERATIVA,
        json_schema_name="haku_operation_intent_check",
    )
    duration_ms = float(getattr(respuesta, "duracion_ms", 0.0) or ((time.perf_counter() - started) * 1000.0))
    datos = getattr(respuesta, "datos_json", None) if respuesta is not None else None
    if not isinstance(datos, dict) or datos.get("clase") not in {"operacion", "formulario", "consulta", "conversacion"}:
        return None, {
            "duracion_ms": duration_ms,
            "cola_ms": float(getattr(respuesta, "espera_cola_ms", 0.0) or 0.0),
            "chars_prompt": chars_prompt,
            "tokens_prompt": max(1, chars_prompt // 4),
            "tokens_completados": int(getattr(respuesta, "tokens_completados", 0) or 0),
        }
    return datos, {
        "duracion_ms": duration_ms,
        "cola_ms": float(getattr(respuesta, "espera_cola_ms", 0.0) or 0.0),
        "chars_prompt": chars_prompt,
        "tokens_prompt": max(1, chars_prompt // 4),
        "tokens_completados": int(getattr(respuesta, "tokens_completados", 0) or 0),
    }


def _aplicar_guardas_post_plan(
    texto: str,
    plan: PlanAgente,
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> PlanAgente:
    """Guardas finales compartidas entre CACHE_HIT y LLM en vivo para evitar divergencias."""
    if "stock_minimo" in (plan.campos or []) and not _texto_menciona_stock_minimo(texto):
        plan.campos = [c for c in plan.campos if c != "stock_minimo"]
    if (
        plan.tipo == "consulta"
        and plan.operacion == "sumar_unidades"
        and plan.filtros.condicion in _CONDICIONES_STOCK_PARA_VERIFICACION_SEMANTICA
        and set(plan.campos or []) <= {"stock"}
        and not plan.filtros.codigos
        and not plan.filtros.texto
        and not re.search(r"\b(?:cuant\w*|total|suma|unidades?)\b", normalizar(texto or ""))
    ):
        plan.operacion = "listar"
        plan.campos = []
        plan.medida_solicitada = "ninguna"
        plan.alcance_entidad = "coleccion"
    plan = _limpiar_filtro_repetido_condicion_stock(texto, plan, codigos_productos_candidatos)
    return plan


def planificar(
    texto: str,
    cliente_llm: Any,
    contexto_txt: str,
    timeout: Optional[float] = None,
    usar_cache: bool = False,
    telemetria: Optional[Dict[str, Any]] = None,
    almacenes_accesibles: Optional[List[str]] = None,
    ubicaciones_accesibles: Optional[List[str]] = None,
    categorias_accesibles: Optional[List[str]] = None,
    codigos_productos_candidatos: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Interpreta el mensaje y repara con el mismo LLM solo planes de ubicaciones inconsistentes."""
    if telemetria is None:
        telemetria = {}
    if timeout is None:
        timeout = float(os.environ.get("HAKU_LLM_REQUEST_TIMEOUT_SECONDS", "45"))
    telemetria.setdefault("estado", "NO_INTENTADA")
    telemetria.setdefault("llamadas_fisicas", 0)
    # Estas dos formas expresan contratos semánticos inequívocos. Resuélvelas
    # antes de la inferencia aunque el cliente haya desactivado el fast path
    # general: no dependen del catálogo de nombres ni escriben en el inventario.
    plan_estructural = _planificar_historial_ordinal(texto, contexto_txt)
    if plan_estructural is not None:
        telemetria.update({
            "estado": "FAST_PATH_SEMANTICO",
            "llamadas_fisicas": 0,
            "codigo_http": 200,
        })
        return plan_estructural
    plan_cache_previo = (
        obtener_plan_cache(texto, contexto_txt, codigos_productos_candidatos)
        if usar_cache else None
    )

    # El modelo puede interpretar un seguimiento directamente con un esquema
    # reducido. Si no identifica inequívocamente una consulta del foco, se
    # usa el planificador completo y se contabiliza también este intento.
    revision_breve = None
    foco_match = re.search(r"(?m)^Producto en foco:\s*(\S+)", contexto_txt or "")
    candidatos = {str(c).upper() for c in (codigos_productos_candidatos or []) if c}
    if (not getattr(plan_cache_previo, "_cache_final_verificado", False)
            and foco_match and "Resultados previos:" not in (contexto_txt or "")
            and (
                candidatos == {foco_match.group(1).upper()}
                or (
                    not candidatos
                    and (
                        _mensaje_requiere_referencia_contextual(texto)
                        or not _extraer_objeto_consulta_sin_marco(texto)
                    )
                )
            )):
        revision_breve = _revisar_social_con_foco(
            texto, contexto_txt, cliente_llm, PlanAgente(tipo="social"), timeout, telemetria,
            almacenes_accesibles=almacenes_accesibles,
        )
        if revision_breve.tipo == "operacion":
            revision_breve.fuente_plan = "llm_contextual_operacion"
            telemetria["estado"] = "LLM_CONTEXTO_OPERACION"
            telemetria["codigo_http"] = 200
            telemetria["duracion_ms"] = revision_breve.llm_ms
            return revision_breve
        if _consulta_contextual_breve_ejecutable(revision_breve):
            revision_breve.fuente_plan = "llm_contextual"
            telemetria["estado"] = "LLM_CONTEXTO"
            telemetria["codigo_http"] = 200
            telemetria["duracion_ms"] = revision_breve.llm_ms
            return revision_breve
        if revision_breve.tipo == "fuera_dominio":
            revision_breve.fuente_plan = "llm_contextual"
            telemetria["estado"] = "LLM_CONTEXTO"
            telemetria["codigo_http"] = 200
            telemetria["duracion_ms"] = revision_breve.llm_ms
            return revision_breve
        telemetria.pop("revision_contexto", None)

    if usar_cache:
        en_cache = plan_cache_previo
        if en_cache is not None:
            en_cache = _validar_referencia_foco_con_contexto(en_cache, contexto_txt)
            telemetria["estado"] = "CACHE_HIT"
            if revision_breve is not None:
                en_cache.llm_ms += revision_breve.llm_ms
                en_cache.llm_cola_ms += revision_breve.llm_cola_ms
                en_cache.chars_prompt += revision_breve.chars_prompt
                en_cache.tokens_estimados_prompt += revision_breve.tokens_estimados_prompt
                en_cache.tokens_completados += revision_breve.tokens_completados
            else:
                telemetria["llamadas_fisicas"] = 0
            telemetria["codigo_http"] = 200
            en_cache = _revisar_social_con_foco(texto, contexto_txt, cliente_llm, en_cache, timeout, telemetria)
            if getattr(en_cache, "_cache_final_verificado", False):
                en_cache = _normalizar_intencion_evento_temporal(texto, en_cache)
                en_cache = _normalizar_historial_temporal(texto, en_cache)
                en_cache = _normalizar_peticion_atributo(texto, en_cache)
                return en_cache
            en_cache = _reparar_plan_comparacion_ubicaciones(
                texto,
                contexto_txt,
                cliente_llm,
                en_cache,
                timeout,
                telemetria,
                codigos_productos_candidatos,
                categorias_accesibles or [],
            )
            en_cache = _normalizar_condicion_stock_explicita(
                texto, en_cache, categorias_accesibles or [], almacenes_accesibles or [],
            )
            en_cache = _normalizar_comparacion_sin_objetivos(
                texto, en_cache, ubicaciones_accesibles,
            )
            en_cache = _verificar_condicion_stock_semantica(
                texto, cliente_llm, en_cache, timeout, telemetria,
                codigos_productos_candidatos,
            )
            en_cache = _normalizar_comparacion_sin_objetivos(
                texto, en_cache, ubicaciones_accesibles,
            )
            en_cache = _completar_ubicacion_explicita(
                en_cache, texto, ubicaciones_accesibles or [], almacenes_accesibles or [], categorias_accesibles or [],
            )
            en_cache = _recuperar_descriptor_producto_grounded(texto, en_cache)
            en_cache = _verificar_intencion_listado_global(
                texto, cliente_llm, en_cache, timeout, telemetria,
                codigos_productos_candidatos=codigos_productos_candidatos,
                contexto_catalogo=contexto_txt,
                categorias_accesibles=categorias_accesibles,
                almacenes_accesibles=almacenes_accesibles,
                ubicaciones_accesibles=ubicaciones_accesibles,
            )
            if en_cache.tipo in {"capacidades", "aclaracion"}:
                return en_cache
            en_cache = _aclarar_consulta_sin_entidad(texto, en_cache, contexto_txt)
            en_cache = _normalizar_union_condiciones_stock(texto, en_cache)
            en_cache = _normalizar_categoria_con_catalogo(
                en_cache, categorias_accesibles or [], texto,
            )
            en_cache = _normalizar_intencion_conteo(texto, en_cache)
            en_cache = _normalizar_alcance_segun_campos(en_cache, texto)
            en_cache = _normalizar_filtro_evento_desde_mensaje(texto, en_cache)
            en_cache = _completar_almacenes_explicitos(en_cache, texto, almacenes_accesibles or [])
            en_cache = _completar_almacen_consulta_explicito(texto, en_cache)
            en_cache = _normalizar_categoria_con_catalogo(
                en_cache, categorias_accesibles or [], texto,
            )
            en_cache = _completar_filtro_producto_explicito(
                en_cache,
                texto,
                almacenes_accesibles or [],
                ubicaciones_accesibles or [],
                codigos_productos_candidatos or [],
            )
            en_cache = _reverificar_atributos_stock_compuestos_si_falta(
                texto,
                cliente_llm,
                en_cache,
                timeout,
                telemetria,
                codigos_productos_candidatos,
            )
            en_cache = _limpiar_filtro_repetido_condicion_stock(
                texto, en_cache, codigos_productos_candidatos,
            )
            en_cache = _normalizar_comparacion_ubicaciones(texto, en_cache)
            en_cache = _normalizar_intencion_suma_unidades(texto, en_cache, categorias_accesibles or [])
            foco_disponible = bool(re.search(r"(?m)^Producto en foco:\s*\S", contexto_txt or ""))
            en_cache = _normalizar_intencion_prestamo_actual(
                texto, en_cache, categorias_accesibles or [], foco_disponible=foco_disponible,
            )
            en_cache = _normalizar_intencion_evento_temporal(texto, en_cache)
            en_cache = _normalizar_aclaracion_de_conjunto(
                texto,
                en_cache,
                categorias_accesibles or [],
                ubicaciones_accesibles or [],
                almacenes_accesibles or [],
            )
            en_cache = _normalizar_alcance_entidad_por_morfologia(texto, en_cache)
            en_cache = _normalizar_intencion_conteo(texto, en_cache)
            en_cache = _aplicar_medida_solicitada(en_cache)
            en_cache = _completar_atributos_canonicos_solicitados(texto, en_cache)
            en_cache = _recortar_campos_sobregenerados_de_coleccion_stock(texto, en_cache)
            en_cache = _verificar_intencion_listado_global(
                texto, cliente_llm, en_cache, timeout, telemetria,
                codigos_productos_candidatos=codigos_productos_candidatos,
                contexto_catalogo=contexto_txt,
                categorias_accesibles=categorias_accesibles,
                almacenes_accesibles=almacenes_accesibles,
                ubicaciones_accesibles=ubicaciones_accesibles,
            )
            en_cache = _normalizar_intencion_extremal(texto, en_cache)
            en_cache = _normalizar_detalle_de_coleccion(texto, en_cache)
            en_cache = _verificar_forma_respuesta_prestamos(
                texto, en_cache, cliente_llm, timeout, telemetria,
                codigos_productos_candidatos,
            )
            en_cache = _revalidar_condicion_stock_final(
                texto,
                cliente_llm,
                en_cache,
                timeout,
                telemetria,
                codigos_productos_candidatos,
            )
            en_cache = _verificar_medida_cantidad_semantica(
                texto, en_cache, cliente_llm, timeout, telemetria,
                codigos_productos_candidatos,
            )
            en_cache = _normalizar_saldo_cuantitativo_final(en_cache)
            en_cache = _normalizar_listado_colectivo_condicionado(en_cache)
            en_cache = _aplicar_guardas_post_plan(texto, en_cache, codigos_productos_candidatos)
            return _completar_nombre_producto_alta(
                en_cache,
                texto,
                categorias_accesibles or [],
                almacenes_accesibles or [],
                ubicaciones_accesibles or [],
            )

    texto_pre = preprocesar_texto_inventario(texto, normalizar_medidas=False)[:400]
    tipos_movimiento = _contexto_tipos_movimiento()
    contexto_eventos = f"\nTipos de movimiento válidos: {tipos_movimiento}" if tipos_movimiento else ""
    contenido_usuario = (
        f"[Contexto]\n{contexto_txt}{contexto_eventos}\n"
        f"[Fecha local de referencia para interpretar periodos históricos: {_fecha_referencia_local()}]\n\n"
        f"[Mensaje del usuario]\n{texto_pre}"
    )
    usar_canonico = bool(getattr(cliente_llm, "usar_prompt_canonico", None) is True)
    tiene_foco = bool(re.search(r"(?m)^Producto en foco:\s*\S", contexto_txt or ""))
    consulta_campos_compuestos = _consulta_coordina_entidad(
        texto,
        codigos_productos_candidatos=codigos_productos_candidatos,
        tiene_foco=tiene_foco,
    )
    if usar_canonico:
        from .contrato import PROMPT_SISTEMA_CANONICO
        prompt_sistema_efectivo = PROMPT_SISTEMA_CANONICO
    else:
        prompt_sistema_efectivo = _construir_prompt_sistema(
            texto,
            codigos_productos_candidatos=codigos_productos_candidatos,
            tiene_foco=tiene_foco,
        )

    chars_prompt = len(prompt_sistema_efectivo) + len(contenido_usuario)
    tokens_est_prompt = max(1, chars_prompt // 4)

    mensajes = [
        {"role": "system", "content": prompt_sistema_efectivo},
        {"role": "user", "content": contenido_usuario},
    ]

    telemetria["estado"] = "LLM_LLAMADA_INICIADA"
    telemetria["llamadas_fisicas"] = int(telemetria.get("llamadas_fisicas", 0) or 0) + 1
    t0 = time.perf_counter()
    limite_tokens_plan = _limite_tokens_planificador()

    try:
        resp = cliente_llm.generar(
            mensajes=mensajes,
            formato_json=True,
            temperatura=0.0,
            max_tokens=limite_tokens_plan,
            timeout=timeout,
            json_schema=(
                None if usar_canonico else
                _ESQUEMA_PLANIFICADOR_CAMPOS_COMPUESTOS
                if consulta_campos_compuestos else _ESQUEMA_PLANIFICADOR_JSON
            ),
            json_schema_name="haku_inventory_plan",
        )
    except Exception as exc_gen:
        duracion_ms = (time.perf_counter() - t0) * 1000.0
        exc_str = str(exc_gen).lower()
        exc_nombre = type(exc_gen).__name__.lower()
        if "timeout" in exc_str or "timeout" in exc_nombre or isinstance(exc_gen, TimeoutError):
            estado_err = "LLM_ERROR_TIMEOUT"
            codigo_http = 504
            msg_usuario = "Tiempo de espera agotado al consultar el Cerebro Central. Por favor intenta de nuevo."
        else:
            estado_err = "LLM_ERROR_TRANSPORTE"
            codigo_http = 502
            msg_usuario = "Error de comunicación con el servicio central de procesamiento lingüístico."
        telemetria["estado"] = estado_err
        telemetria["codigo_http"] = codigo_http
        telemetria["duracion_ms"] = duracion_ms
        telemetria["error"] = str(exc_gen)
        raise ErrorInferenciaLLM(
            estado=estado_err,
            mensaje=msg_usuario,
            codigo_http=codigo_http,
            error_original=exc_gen,
            duracion_ms=duracion_ms,
        ) from exc_gen

    duracion_ms = (time.perf_counter() - t0) * 1000.0
    telemetria["duracion_ms"] = duracion_ms
    resumen_respuesta = _resumir_respuesta_modelo(resp, limite_tokens_plan)
    telemetria["respuesta_modelo"] = resumen_respuesta

    datos = getattr(resp, "datos_json", None) if resp is not None else None
    if datos is None:
        telemetria["estado"] = "LLM_ERROR_RESPUESTA"
        telemetria["codigo_http"] = 502
        logger.warning(
            "[HakuAgente] Salida del Cerebro Central sin JSON parseable: "
            "finish_reason=%s tokens=%s chars=%s posible_corte=%s duracion_ms=%.1f",
            resumen_respuesta["finish_reason"],
            resumen_respuesta["tokens_completados"],
            resumen_respuesta["caracteres_salida"],
            resumen_respuesta["probable_corte_por_limite"],
            duracion_ms,
        )
        raise ErrorInferenciaLLM(
            estado="LLM_ERROR_RESPUESTA",
            mensaje="El Cerebro Central no devolvió una estructura JSON válida.",
            codigo_http=502,
            duracion_ms=duracion_ms,
        )

    plan = validar_plan(datos, texto, codigos_productos_candidatos) if datos else None
    if plan is None:
        telemetria["estado"] = "LLM_ERROR_PLAN"
        telemetria["codigo_http"] = 502
        logger.warning(
            "[HakuAgente] JSON del Cerebro Central incompatible con el plan: "
            "finish_reason=%s tokens=%s chars=%s posible_corte=%s duracion_ms=%.1f",
            resumen_respuesta["finish_reason"],
            resumen_respuesta["tokens_completados"],
            resumen_respuesta["caracteres_salida"],
            resumen_respuesta["probable_corte_por_limite"],
            duracion_ms,
        )
        raise ErrorInferenciaLLM(
            estado="LLM_ERROR_PLAN",
            mensaje="La respuesta del modelo no cumple con el esquema requerido para la operación.",
            codigo_http=502,
            duracion_ms=duracion_ms,
        )

    plan = _revisar_social_con_foco(texto, contexto_txt, cliente_llm, plan, timeout, telemetria)
    plan = _validar_referencia_foco_con_contexto(plan, contexto_txt)

    plan = _reparar_plan_comparacion_ubicaciones(
        texto,
        contexto_txt,
        cliente_llm,
        plan,
        timeout,
        telemetria,
        codigos_productos_candidatos,
        categorias_accesibles or [],
        duracion_inicial_ms=float(getattr(resp, "duracion_ms", 0.0) or duracion_ms),
    )

    plan = _recuperar_consulta_catalogo_almacen(texto, plan)
    plan = _normalizar_condicion_stock_explicita(
        texto, plan, categorias_accesibles or [], almacenes_accesibles or [],
    )
    plan = _normalizar_comparacion_sin_objetivos(
        texto, plan, ubicaciones_accesibles,
    )
    plan = _verificar_condicion_stock_semantica(
        texto, cliente_llm, plan, timeout, telemetria,
        codigos_productos_candidatos,
    )
    plan = _normalizar_comparacion_sin_objetivos(
        texto, plan, ubicaciones_accesibles,
    )
    plan = _completar_ubicacion_explicita(
        plan, texto, ubicaciones_accesibles or [], almacenes_accesibles or [], categorias_accesibles or [],
    )
    plan = _recuperar_descriptor_producto_grounded(texto, plan)
    plan = _verificar_intencion_listado_global(
        texto, cliente_llm, plan, timeout, telemetria,
        codigos_productos_candidatos=codigos_productos_candidatos,
        contexto_catalogo=contexto_txt,
        categorias_accesibles=categorias_accesibles,
        almacenes_accesibles=almacenes_accesibles,
        ubicaciones_accesibles=ubicaciones_accesibles,
    )
    # Varias correcciones pueden completar el producto después de la primera
    # normalización temporal.
    if "revision_contexto" not in telemetria:
        tipo_previo = plan.tipo
        plan = _revisar_social_con_foco(texto, contexto_txt, cliente_llm, plan, timeout, telemetria)
        if tipo_previo in {"social", "fuera_dominio", "aclaracion"} and plan.tipo == "consulta":
            plan = _validar_referencia_foco_con_contexto(plan, contexto_txt)
            plan = _normalizar_peticion_atributo(texto, plan)
            plan = _aplicar_medida_solicitada(plan)
    # Sincronizar el pipeline determinista de post-normalización con la ruta caché
    # para que las consultas nuevas en vivo también limpien filtros, atributos y campos sobregenerados.
    if plan.tipo not in {"capacidades", "fuera_dominio", "social"}:
        plan = _normalizar_intencion_extremal(texto, plan)
        if plan.operacion not in {"mayor_stock", "menor_stock"}:
            plan = _aclarar_consulta_sin_entidad(texto, plan, contexto_txt)
        plan = _normalizar_union_condiciones_stock(texto, plan)
        plan = _normalizar_categoria_con_catalogo(plan, categorias_accesibles or [], texto)
        plan = _normalizar_intencion_conteo(texto, plan)
        plan = _normalizar_alcance_segun_campos(plan, texto)
        plan = _normalizar_filtro_evento_desde_mensaje(texto, plan)
        plan = _completar_almacenes_explicitos(plan, texto, almacenes_accesibles or [])
        plan = _completar_almacen_consulta_explicito(texto, plan)
        plan = _normalizar_categoria_con_catalogo(plan, categorias_accesibles or [], texto)
        plan = _completar_filtro_producto_explicito(
            plan,
            texto,
            almacenes_accesibles or [],
            ubicaciones_accesibles or [],
            codigos_productos_candidatos or [],
        )
        plan = _limpiar_filtro_repetido_condicion_stock(texto, plan, codigos_productos_candidatos)
        plan = _normalizar_comparacion_ubicaciones(texto, plan)
        plan = _normalizar_intencion_suma_unidades(texto, plan, categorias_accesibles or [])
        foco_disp_live = bool(re.search(r"(?m)^Producto en foco:\s*\S", contexto_txt or ""))
        plan = _normalizar_intencion_prestamo_actual(
            texto, plan, categorias_accesibles or [], foco_disponible=foco_disp_live,
        )
        plan = _normalizar_intencion_evento_temporal(texto, plan)
        plan = _normalizar_aclaracion_de_conjunto(
            texto,
            plan,
            categorias_accesibles or [],
            ubicaciones_accesibles or [],
            almacenes_accesibles or [],
        )
        plan = _normalizar_alcance_entidad_por_morfologia(texto, plan)
        plan = _aplicar_medida_solicitada(plan)
        plan = _completar_atributos_canonicos_solicitados(texto, plan)
        plan = _recortar_campos_sobregenerados_de_coleccion_stock(texto, plan)
        if "stock_minimo" in (plan.campos or []) and not _texto_menciona_stock_minimo(texto):
            plan.campos = [c for c in plan.campos if c != "stock_minimo"]

    # Estos ajustes se aplican después de completar el descriptor del producto:
    # una mención plural y una selección extrema del mismo turno deben conservar
    # el conjunto definido por el mensaje incluso si una normalización previa
    # lo redujo a detalle.
    plan = _normalizar_intencion_extremal(texto, plan)
    plan = _normalizar_detalle_de_coleccion(texto, plan)
    plan = _verificar_forma_respuesta_prestamos(
        texto, plan, cliente_llm, timeout, telemetria,
        codigos_productos_candidatos,
    )
    plan = _revalidar_condicion_stock_final(
        texto,
        cliente_llm,
        plan,
        timeout,
        telemetria,
        codigos_productos_candidatos,
    )
    # Los verificadores finales y la reconstrucción de atributos pueden
    # completar o reclasificar campos después de la primera normalización.
    # Vuelve a imponer al final una cardinalidad explícita de referencias.
    plan = _normalizar_intencion_conteo(texto, plan)
    plan = _verificar_medida_cantidad_semantica(
        texto, plan, cliente_llm, timeout, telemetria,
        codigos_productos_candidatos,
    )
    plan = _normalizar_saldo_cuantitativo_final(plan)
    plan = _normalizar_listado_colectivo_condicionado(plan)
    plan = _aplicar_guardas_post_plan(texto, plan, codigos_productos_candidatos)
    plan = _normalizar_historial_temporal(texto, plan)

    if plan.fuente_plan not in {"llm_reparado", "llm_reparacion_fallida"}:
        plan.llm_ms += float(getattr(resp, "duracion_ms", 0.0) or duracion_ms)
        plan.llm_cola_ms += float(getattr(resp, "espera_cola_ms", 0.0) or 0.0)
        plan.chars_prompt += chars_prompt
        plan.tokens_estimados_prompt += tokens_est_prompt
        toks_comp = getattr(resp, "tokens_completados", 0)
        if isinstance(toks_comp, (int, float)):
            plan.tokens_completados += int(toks_comp)
        plan.fuente_plan = "llm"
    if revision_breve is not None:
        plan.llm_ms += revision_breve.llm_ms
        plan.llm_cola_ms += revision_breve.llm_cola_ms
        plan.chars_prompt += revision_breve.chars_prompt
        plan.tokens_estimados_prompt += revision_breve.tokens_estimados_prompt
        plan.tokens_completados += revision_breve.tokens_completados
    telemetria["duracion_ms"] = plan.llm_ms
    telemetria["estado"] = (
        "LLM_PLAN_REPARADO" if plan.fuente_plan == "llm_reparado"
        else "LLM_PLAN_REPARACION_FALLIDA" if plan.fuente_plan == "llm_reparacion_fallida"
        else "LLM_RESPUESTA_VALIDA"
    )
    telemetria["codigo_http"] = 200

    return plan


def planificar_canonico(
    texto: str,
    cliente_llm: Any,
    contexto_txt: str,
    timeout: Optional[float] = None,
    usar_cache: bool = False,
) -> Tuple[Optional[Any], Optional[PlanAgente]]:
    """Inferencia directa utilizando el prompt de sistema canónico PlanHaku (Fase II-C.1).

    Retorna una tupla (plan_haku, plan_agente_adaptado).
    Garantiza compatibilidad total con HerramientasInventario y redactor.
    """
    from .contrato import PROMPT_SISTEMA_CANONICO, adaptar_plan_haku_a_plan_agente, validar_plan_haku

    if timeout is None:
        timeout = float(os.environ.get("HAKU_LLM_REQUEST_TIMEOUT_SECONDS", "45"))

    texto_pre = preprocesar_texto_inventario(texto, normalizar_medidas=False)[:400]
    tipos_movimiento = _contexto_tipos_movimiento()
    contexto_eventos = f"\nTipos de movimiento válidos: {tipos_movimiento}" if tipos_movimiento else ""
    contenido_usuario = f"[Contexto]\n{contexto_txt}{contexto_eventos}\n\n[Mensaje del usuario]\n{texto_pre}"
    chars_prompt = len(PROMPT_SISTEMA_CANONICO) + len(contenido_usuario)
    tokens_est_prompt = max(1, chars_prompt // 4)

    mensajes = [
        {"role": "system", "content": PROMPT_SISTEMA_CANONICO},
        {"role": "user", "content": contenido_usuario},
    ]
    resp = cliente_llm.generar(
        mensajes=mensajes,
        formato_json=True,
        temperatura=0.0,
        max_tokens=180,
        timeout=timeout,
    )
    datos = getattr(resp, "datos_json", None) if resp is not None else None
    if not datos:
        return None, None

    plan_haku, errores_haku = validar_plan_haku(datos, texto)
    plan_agente = validar_plan(datos, texto)

    if plan_agente is not None:
        plan_agente.llm_ms = float(getattr(resp, "duracion_ms", 0.0) or 0.0)
        plan_agente.llm_cola_ms = float(getattr(resp, "espera_cola_ms", 0.0) or 0.0)
        plan_agente.chars_prompt = chars_prompt
        plan_agente.tokens_estimados_prompt = tokens_est_prompt
        toks_comp = getattr(resp, "tokens_completados", 0)
        plan_agente.tokens_completados = int(toks_comp) if isinstance(toks_comp, (int, float)) else 0
        plan_agente.fuente_plan = "llm_canonico"

    if plan_haku is not None:
        plan_haku.datos_extra["llm_ms"] = float(getattr(resp, "duracion_ms", 0.0) or 0.0)
        plan_haku.datos_extra["chars_prompt"] = chars_prompt
        plan_haku.datos_extra["tokens_estimados_prompt"] = tokens_est_prompt
        plan_haku.datos_extra["tokens_completados"] = int(getattr(resp, "tokens_completados", 0) or 0)
        plan_haku.fuente = "llm_canonico"

    return plan_haku, plan_agente
