"""Agente HAKU Inventario basado en el Cerebro Central compartido.

Ciclo por mensaje:
    1. Planificar  — una inferencia al Cerebro Central (planificador.planificar).
    2. Resolver    — referencias contextuales (foco / resultados previos) de forma explícita.
    3. Consultar   — herramientas de lectura con alcance por almacén.
    4. Redactar    — respuesta a partir de resultados reales y del atributo pedido.
    5. Actualizar  — contexto mínimo (foco o colección) según el resultado.

Las MODIFICACIONES nunca se ejecutan aquí: se traducen al contrato existente
(ExtraccionSemantica / IntencionResultado) para que pasen por el flujo de
validación y confirmación ya establecido.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .herramientas import (
    FiltrosConsulta,
    HerramientasInventario,
    _raiz,
    extraer_codigos_de_texto,
    tokens_significativos,
)
from .planificador import (
    ErrorInferenciaLLM,
    ACCIONES_VALIDAS,
    PlanAgente,
    construir_contexto,
    normalizar,
    planificar,
    planificar_determinista,
    guardar_plan_cache,
    _normalizar_intencion_extremal,
    _normalizar_intencion_suma_unidades,
    _normalizar_saldo_cuantitativo_final,
    _quitar_con_stock_de_producto_individual,
    requiere_catalogo_almacenes_condicion_stock,
    requiere_catalogo_categorias_condicion_stock,
    _es_consulta_global_inventario,
    _es_negacion_operacion,
    _es_pregunta_como_operar,
    _es_pregunta_permiso_capacidad,
    _es_pregunta_quien_tiene_prestamo,
    _texto_respalda_accion_operativa,
    aclarar_agregado_sin_alcance_explicito,
    verificar_intencion_operativa_sin_entidad,
    normalizar_decision_intencion_operativa,
    _str,
)
from .redactor import redactar, redactar_movimientos_globales, seleccionar_ganadores_extremos

logger = logging.getLogger("integraciones.haku.agente")

# Instrumentación verificable por pruebas: número de inferencias, fast-paths, caché y consumo de tokens.
METRICAS_AGENTE: Dict[str, Any] = {
    "inferencias": 0,
    "resueltas_sin_llm": 0,
    "cache_hits": 0,
    "planes_validos": 0,
    "planes_invalidos": 0,
    "chars_prompt_total": 0,
    "tokens_completados_total": 0,
    "ultima": None,
}


@dataclass
class SalidaAgente:
    """Resultado del agente.

    - respuesta: respuesta final lista para devolver al usuario.
    - extraccion: ExtraccionSemantica para el flujo transaccional existente.
    - intencion: IntencionResultado para flujos existentes (p. ej. EDITAR_PRODUCTO).
    - inferencia_realizada: True si ya se consultó al Cerebro Central (evita una segunda inferencia).
    - error_terminal: True si ocurrió un fallo irrecuperable de disponibilidad o inferencia LLM.
    - estado_terminal: Código unívoco de la taxonomía C-03.
    - codigo_http: Código de respuesta HTTP asociado (200, 502, 503, 504).
    - intento_inferencia_fisica: True si se ejecutó llamada física a cliente_llm.generar().
    """

    respuesta: Optional[Dict[str, Any]] = None
    extraccion: Any = None
    intencion: Any = None
    inferencia_realizada: bool = False
    plan: Optional[PlanAgente] = None
    plan_haku: Optional[Any] = None
    traza: Dict[str, Any] = field(default_factory=dict)
    error_terminal: bool = False
    estado_terminal: Optional[str] = None
    codigo_http: int = 200
    intento_inferencia_fisica: bool = False


def _obtener_cliente():
    from ..lenguaje.interprete_semantico import _obtener_cliente_cerebro

    return _obtener_cliente_cerebro()


def _es_comparacion_stock_explicita(texto: str) -> bool:
    """Reconoce comparaciones literales de existencias sin otros atributos."""
    texto_norm = normalizar(texto)
    return bool(
        re.search(r"\bcompar(?:a|ar|ame|en|emos)\b", texto_norm)
        and re.search(r"\b(?:stock|existencias?|unidades?|disponibles?)\b", texto_norm)
        and not re.search(
            r"\b(?:ubicacion(?:es)?|almacenes?|bodegas?|categorias?|prestamos?|codigo|nombre|descripcion|"
            r"minimo|movimientos?|ficha|empaque)\b",
            texto_norm,
        )
    )


def _ordenar_por_urgencia_de_reposicion(items: List[Dict[str, Any]]) -> None:
    """Ordena filas visibles por agotamiento y cobertura relativa del mínimo."""
    def clave(item: Dict[str, Any]):
        disponible = max(0, int(item.get("disponible", 0) or 0))
        minimo = max(0, int(item.get("stock_minimo", 0) or 0))
        cobertura = disponible / minimo if minimo else 1.0
        faltante = max(0, minimo - disponible)
        return (
            0 if disponible == 0 else 1,
            cobertura,
            -faltante,
            disponible,
            str(item.get("nombre", "")).casefold(),
            str(item.get("codigo", "")).casefold(),
        )

    items.sort(key=clave)


def _operacion_comparacion_stock(texto: str) -> str:
    texto_norm = normalizar(texto)
    if re.search(r"\b(?:cual|el que)\b.*\b(?:mas|mayor)\b", texto_norm):
        return "mayor_stock"
    if re.search(r"\b(?:cual|el que)\b.*\b(?:menos|menor)\b", texto_norm):
        return "menor_stock"
    return "comparar"


def _codigos_comparacion_stock(texto: str, herramientas: HerramientasInventario) -> List[str]:
    if not _es_comparacion_stock_explicita(texto):
        return []
    from .herramientas import extraer_codigos_de_texto

    vistos: List[str] = []
    for codigo in list(extraer_codigos_de_texto(texto)) + list(herramientas.codigos_productos_mencionados(texto)):
        if codigo and codigo not in vistos:
            vistos.append(codigo)
    return vistos


def _crear_plan_comparacion_stock(codigos: List[str], texto: str) -> Optional[PlanAgente]:
    if len(codigos) < 2:
        return None
    return PlanAgente(
        tipo="consulta",
        operacion=_operacion_comparacion_stock(texto),
        filtros=FiltrosConsulta(codigos=codigos),
        referencia="ninguna",
        campos=["stock"],
        fuente_plan="fast_path_comparacion_productos",
    )


def _corregir_comparacion_stock_literal(
    plan: Optional[PlanAgente],
    texto: str,
    herramientas: HerramientasInventario,
) -> Optional[PlanAgente]:
    """Revalida contra el catálogo todos los productos nombrados en una comparación."""
    if plan is None or plan.tipo != "consulta":
        return plan
    codigos = _crear_plan_comparacion_stock(_codigos_comparacion_stock(texto, herramientas), texto)
    if codigos is None:
        return plan
    if plan.fuente_plan == "fast_path_comparacion_productos":
        return plan
    plan.operacion = codigos.operacion
    plan.filtros.codigos = codigos.filtros.codigos
    plan.filtros.texto = None
    if not plan.filtros.categoria_explicita:
        plan.filtros.categoria = None
        plan.filtros.categoria_generica = False
    plan.referencia = "ninguna"
    plan.campos = ["stock"]
    return plan


def _usar_codigo_para_producto_literal(
    plan: Optional[PlanAgente],
    texto: str,
    herramientas: HerramientasInventario,
) -> Optional[PlanAgente]:
    """Cambia texto por código exacto cuando un fast-path nombra un producto del catálogo."""
    if (
        plan is None
        or plan.fuente_plan not in ("fast_path_stock_explicito", "fast_path_ubicacion_explicita")
        or not plan.filtros.texto
        or plan.filtros.codigos
    ):
        return plan
    codigos = herramientas.codigos_productos_mencionados(texto)
    menciones = []
    if not codigos:
        from .herramientas import extraer_codigos_de_texto, FiltrosConsulta

        menciones = extraer_codigos_de_texto(texto)
        if len(menciones) == 1:
            resultado_codigo = herramientas.buscar_productos(FiltrosConsulta(codigos=menciones))
            if not resultado_codigo.error and resultado_codigo.total == 1:
                codigos = [resultado_codigo.items[0]["codigo"]]
    if len(codigos) == 1:
        plan.filtros.codigos = codigos
        plan.filtros.texto = None
    return plan


def _reconciliar_producto_literal_catalogo(
    plan: Optional[PlanAgente],
    texto: str,
    herramientas: HerramientasInventario,
    codigos_producto_literal: Optional[List[str]] = None,
) -> Optional[PlanAgente]:
    """Ancla una consulta al producto vigente cuando el mensaje cita su nombre completo.

    La extracción del LLM puede añadir palabras de la petición (por ejemplo,
    "separado por estado") al nombre del producto. Si el catálogo encuentra una
    única mención literal y visible para el usuario, el código actual es una
    referencia más precisa que el texto interpretado por el modelo.
    """
    if plan is None or plan.tipo not in ("consulta", "aclaracion") or plan.accion or plan.operacion == "comparar":
        return plan

    codigos = (
        list(codigos_producto_literal)
        if codigos_producto_literal is not None
        else herramientas.codigos_productos_mencionados(texto)
    )
    if not codigos:
        from .herramientas import extraer_codigos_de_texto, FiltrosConsulta

        menciones = extraer_codigos_de_texto(texto)
        if len(menciones) == 1:
            resultado_codigo = herramientas.buscar_productos(FiltrosConsulta(codigos=menciones))
            if not resultado_codigo.error and resultado_codigo.total == 1:
                codigos = [resultado_codigo.items[0]["codigo"]]
    plan.entidad_producto_mencionada = bool(codigos or menciones)
    if len(codigos) != 1:
        return plan

    # La mención literal del catálogo elimina la necesidad de pedir que se
    # repita el producto y prevalece sobre un foco anterior mal conservado.
    plan.entidad_producto_mencionada = True
    alcance_modelo = (plan.crudo or {}).get("alcance_entidad") if isinstance(plan.crudo, dict) else None
    plan.tipo = "consulta"
    if not plan.operacion or plan.operacion in ("listar", "contar"):
        plan.operacion = "detalle"
    plan.filtros.codigos = codigos
    plan.filtros.texto = None
    plan.referencia = "ninguna"
    if (
        not _es_consulta_global_inventario(texto)
        and alcance_modelo not in {"coleccion", "global"}
    ):
        # One exact live catalogue match identifies a single product when the
        # model omitted or could not determine its scope. Respect an explicit
        # collection/global scope and inventory-wide requests.
        plan.alcance_entidad = "producto"
    if not plan.filtros.categoria_explicita:
        plan.filtros.categoria = None
        plan.filtros.categoria_generica = False
    return plan


def _anclar_producto_literal_operacion(
    plan: Optional[PlanAgente],
    texto: str,
    herramientas: HerramientasInventario,
) -> Optional[PlanAgente]:
    """Ancla una mutación al código vigente si el usuario nombró un producto único.

    El modelo puede mezclar el nombre con cantidades, motivos o marcos de una
    operación. Un nombre completo hallado en el catálogo autorizado prevalece
    sobre ese filtro textual; si no es único, se conserva el plan para pedir
    aclaración sin elegir un producto por aproximación.
    """
    if plan is None or plan.tipo != "operacion" or not plan.accion:
        return plan

    codigos = herramientas.codigos_productos_mencionados(texto)
    if not codigos:
        from .herramientas import extraer_codigos_de_texto, FiltrosConsulta

        menciones = extraer_codigos_de_texto(texto)
        if len(menciones) == 1:
            resultado = herramientas.buscar_productos(FiltrosConsulta(codigos=menciones))
            if not resultado.error and resultado.total == 1:
                codigos = [resultado.items[0]["codigo"]]

    if len(codigos) == 1:
        plan.filtros.codigos = codigos
        plan.filtros.texto = None
        plan.referencia = "ninguna"
        plan.alcance_entidad = "producto"
        return plan

    # Si el LLM devolvió un código real del catálogo en crudo["producto"] para un sustantivo
    # (ej. "Presta 100 osciloscopios a Carlos." -> crudo["producto"] = "OSC-001"), validar anclaje
    # léxico real con el mensaje para recuperarlo sin aceptar alucinaciones sin sustento.
    if not plan.producto and not plan.filtros.codigos and isinstance(plan.crudo, dict):
        from .herramientas import FiltrosConsulta, normalizar_codigo_producto

        crudo_prod = str(plan.crudo.get("producto") or "").strip()
        cod_crudo = normalizar_codigo_producto(crudo_prod) if re.fullmatch(r"[A-Za-z]{2,5}[-_ ]?\d{1,5}", crudo_prod) else None
        if cod_crudo:
            res_cod = herramientas.buscar_productos(FiltrosConsulta(codigos=[cod_crudo]))
            if not res_cod.error and res_cod.total == 1:
                nom_cat = str(res_cod.items[0].get("nombre") or "")
                raices_excluidas = {
                    "producto", "product", "articul", "unidad", "inventari", "almacen",
                    "equip", "prestam", "devolucion", "entrad", "salid", "traslad",
                    "stock", "todo", "complet", "digital", "manual", "kit",
                }
                raices_cat = {r for r in tokens_significativos(nom_cat) if len(r) >= 4 and r not in raices_excluidas}
                palabras_msg = re.findall(r"[A-Za-zÁ-ÿñÑ0-9]+", texto or "")
                tokens_anclados = [
                    p for p in palabras_msg
                    if len(normalizar(p)) >= 4 and _raiz(normalizar(p)) in raices_cat
                ]
                if tokens_anclados:
                    busq_anclaje = herramientas.buscar_productos(FiltrosConsulta(texto=tokens_anclados[0]))
                    if not busq_anclaje.error and busq_anclaje.total == 1 and busq_anclaje.items[0].get("codigo") == cod_crudo:
                        plan.filtros.codigos = [cod_crudo]
                        plan.producto = cod_crudo
                        plan.referencia = "ninguna"
                        plan.alcance_entidad = "producto"
                    else:
                        plan.producto = " ".join(tokens_anclados[:3])

    # Si tras verificar la intención operativa el modelo dejó el producto vacío o contaminado
    # con el verbo/cantidad de la orden (ej. «saca 5 guant» o «saca 5 tornillos de media pulgada»),
    # extraer el sintagma nominal de forma puramente lingüística y contrastarlo contra el
    # catálogo autorizado vigente (con normalización de medidas, tallas y sinónimos).
    if (
        not plan.filtros.codigos
        and plan.accion in {"entrada", "salida", "prestamo", "devolucion", "traslado", "cambiar_stock_minimo", "editar_producto"}
        and texto
    ):
        from .herramientas import FiltrosConsulta
        from ..lenguaje.observaciones import separar_observacion_operacion

        necesita_extraccion_op = not plan.producto and not plan.filtros.texto
        if not necesita_extraccion_op:
            cand_actual = (plan.producto or plan.filtros.texto or "").strip()
            if cand_actual:
                contamina_verbo_op = bool(
                    re.match(
                        r"^(?:(?:oye|hacu|jacu|por\s+favor)\s+)*"
                        r"(?:sac(?:ar|a|ame|ale|e)|retir(?:ar|a|ame|e)|despach(?:ar|a|ame|e)|"
                        r"ingres(?:ar|a|ame|e)|met(?:er|e|ame)|prest(?:ar|a|ame|eme|e)|"
                        r"devolv(?:er|e|eme)|devuelv(?:e|eme)|traslad(?:ar|a|ame|e)|"
                        r"registr(?:ar|a)|hacer|haz|\d+\s+)",
                        cand_actual,
                        flags=re.IGNORECASE,
                    )
                )
                if contamina_verbo_op:
                    necesita_extraccion_op = True
                else:
                    res_actual = herramientas.buscar_productos(FiltrosConsulta(texto=cand_actual))
                    if not res_actual.error and res_actual.total == 0:
                        necesita_extraccion_op = True

        if necesita_extraccion_op:
            base_op, _ = separar_observacion_operacion(texto)
            cand_op = (base_op or texto).strip().strip("¿?¡!.,;: ")
            # Retirar encabezados de cortesía / verbo operativo / cantidad inicial
            cand_op = re.sub(
                r"^(?:(?:oye|hacu|jacu|por\s+favor)\s+)*"
                r"(?:(?:quiero|necesito|deseo|voy\s+a|quisiera|puedes|podrias)\s+)*"
                r"(?:(?:registrar?|hacer|haz|generar?|crear?|anotar?|dar(?:me)?)\s+(?:una?\s+|el\s+|la\s+)?"
                r"(?:entrada|salida|pr[eé]stamo|devoluci[oó]n|traslado|retiro|ingreso|dotaci[oó]n)\s+(?:de\s+)?|"
                r"(?:sac(?:ar|a|ame|ale|arles?)|retir(?:ar|a|ame|e)|despach(?:ar|a|ame|e)|"
                r"ingres(?:ar|a|ame|e)|met(?:er|e|ame)|prest(?:ar|a|ame|eme|e)|"
                r"devolv(?:er|e|eme)|devuelv(?:e|eme)|traslad(?:ar|a|ame|e)|"
                r"entreg(?:ar|a|ame|e)|dame|p[aá]same|tr[aá]eme)\s+)",
                "",
                cand_op,
                flags=re.IGNORECASE,
            ).strip()
            # Retirar cantidad operativa al principio del sintagma ("5 ", "media docena de ", "una ", "tres unidades de ")
            cand_op = re.sub(
                r"^(?:media\s+docena\s+de\s+|(?:una|dos|tres|cuatro|cinco)\s+docenas?\s+de\s+|un\s+par\s+de\s+|"
                r"(?:\d+|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|quince|veinte|treinta|cincuenta|cien)\s+"
                r"(?:(?:unidades?|piezas?|art[ií]culos?)\s+de\s+)?)",
                "",
                cand_op,
                flags=re.IGNORECASE,
            ).strip()
            # Retirar artículos iniciales y cláusulas finales de destinatario o almacén
            cand_op = re.sub(r"^(?:de\s+)?(?:el|la|los|las|un|una|unos|unas)\s+", "", cand_op, flags=re.IGNORECASE).strip()
            cand_op = re.sub(
                r"\s+(?:para|a\s+nombre\s+de|entregar\s+a|al\s+almac[eé]n|en\s+(?:el\s+)?(?:almac[eé]n|bodega)|"
                r"del?\s+(?:almac[eé]n|bodega)|desde\s+(?:el\s+)?(?:almac[eé]n|bodega)|hacia\s+(?:el\s+)?(?:almac[eé]n|bodega))\b.*$",
                "",
                cand_op,
                flags=re.IGNORECASE,
            ).strip(" ,.;:?!¿¡")
            _pronombres_y_vacios_op = {
                "ese", "esa", "esos", "esas", "eso", "este", "esta", "estos", "estas", "esto",
                "mismo", "misma", "mismos", "mismas", "unidad", "unidades", "pieza", "piezas",
                "producto", "productos", "articulo", "articulos", "stock", "inventario", "almacen", "bodega",
                "algo", "todo", "nada", "aqui", "alla", "ahi",
            }
            toks_cand = [t for t in tokens_significativos(cand_op) if t not in _pronombres_y_vacios_op]
            if cand_op and toks_cand:
                res_op = herramientas.buscar_productos(FiltrosConsulta(texto=cand_op))
                if not res_op.error and res_op.total == 1:
                    cod_unico = res_op.items[0]["codigo"]
                    plan.filtros.codigos = [cod_unico]
                    plan.filtros.texto = None
                    plan.producto = cod_unico
                    plan.referencia = "ninguna"
                    plan.alcance_entidad = "producto"
                elif not res_op.error and res_op.total > 1:
                    plan.producto = cand_op
                    plan.filtros.texto = None
    return plan


def _normalizar_extremos_traslado_catalogo(
    plan: Optional[PlanAgente],
    texto: str,
    herramientas: HerramientasInventario,
) -> Optional[PlanAgente]:
    """Usa los almacenes vivos mencionados y su orden para fijar origen y destino.

    El modelo puede reconocer correctamente un traslado y aun intercambiar sus
    dos almacenes. Los nombres se obtienen del catálogo accesible en tiempo de
    ejecución; solo se corrige cuando el mensaje contiene exactamente dos
    menciones identificables.
    """
    if plan is None or plan.tipo != "operacion" or plan.accion != "traslado" or not texto:
        return plan

    texto_norm = normalizar(texto)
    menciones = []
    for nombre in herramientas.nombres_almacenes():
        nombre_norm = normalizar(nombre)
        if not nombre_norm:
            continue
        inicio = 0
        while True:
            posicion = texto_norm.find(nombre_norm, inicio)
            if posicion < 0:
                break
            menciones.append((posicion, posicion + len(nombre_norm), nombre))
            inicio = posicion + len(nombre_norm)

    menciones.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    no_superpuestas = []
    for mencion in menciones:
        if any(mencion[0] < fin and mencion[1] > inicio for inicio, fin, _ in no_superpuestas):
            continue
        no_superpuestas.append(mencion)
    no_superpuestas.sort(key=lambda item: item[0])

    if len(no_superpuestas) == 2:
        plan.filtros.almacen = no_superpuestas[0][2]
        plan.destino = no_superpuestas[1][2]
    return plan


def _tokens_de_catalogo_mencionados(
    texto: str,
    productos_candidatos: List[Dict[str, str]],
) -> List[str]:
    """Conserva, en orden, los términos del mensaje que existen en nombres actuales."""
    vocabulario_catalogo = set().union(*(
        set(tokens_significativos(item.get("nombre") or ""))
        for item in productos_candidatos
        if item.get("codigo") and item.get("nombre")
    )) if productos_candidatos else set()
    return list(dict.fromkeys(
        token for token in tokens_significativos(texto)
        if token in vocabulario_catalogo
    ))


def _reconciliar_producto_por_descriptor_catalogo(
    plan: Optional[PlanAgente],
    texto: str,
    productos_candidatos: List[Dict[str, str]],
) -> Optional[PlanAgente]:
    """Corrige una selección singular solo si el mensaje distingue otro candidato vivo.

    Usa términos que ya aparecen en los nombres del catálogo actual. Por ejemplo,
    una dimensión escrita en el mensaje distingue dos referencias de la misma
    familia aunque el LLM haya elegido la variante vecina. Si el mensaje solo
    identifica una familia con varias referencias vivas, quita la selección
    arbitraria para que la herramienta solicite una aclaración con opciones.
    """
    if (
        plan is None
        or plan.tipo != "consulta"
        or plan.alcance_entidad in {"coleccion", "global"}
        or plan.entidad_producto_mencionada
        or len(plan.filtros.codigos or []) != 1
        or not productos_candidatos
    ):
        return plan

    codigo_actual = str(plan.filtros.codigos[0] or "").strip().upper()
    tokens_consulta = set(tokens_significativos(texto))
    tokens_por_codigo = {
        str(item.get("codigo") or "").strip().upper(): set(
            tokens_significativos(item.get("nombre") or "")
        )
        for item in productos_candidatos
        if item.get("codigo") and item.get("nombre")
    }
    if codigo_actual not in tokens_por_codigo or len(tokens_por_codigo) < 2:
        return plan

    puntuaciones = {
        codigo: len(tokens_consulta & tokens)
        for codigo, tokens in tokens_por_codigo.items()
    }
    mejor_puntuacion = max(puntuaciones.values(), default=0)
    mejores = [codigo for codigo, puntos in puntuaciones.items() if puntos == mejor_puntuacion]
    if len(mejores) == 1:
        codigo_mejor = mejores[0]
        if codigo_mejor == codigo_actual or puntuaciones[codigo_mejor] <= puntuaciones[codigo_actual]:
            return plan

        tokens_unicos_mejor = tokens_por_codigo[codigo_mejor] - set.union(
            *(tokens for codigo, tokens in tokens_por_codigo.items() if codigo != codigo_mejor)
        )
        if not (tokens_unicos_mejor & tokens_consulta):
            return plan

        elegido = next(
            item for item in productos_candidatos
            if str(item.get("codigo") or "").strip().upper() == codigo_mejor
        )
        plan.filtros.codigos = [str(elegido["codigo"]).strip()]
        plan.filtros.texto = None
        plan.producto = str(elegido["nombre"]).strip()
        plan.alcance_entidad = "producto"
        plan.codigo_catalogo_verificado = True
        return plan

    # Varios candidatos obtuvieron el mismo mejor ajuste. Busca con las
    # palabras compartidas que realmente aparecen en el turno y deja que la
    # herramienta devuelva las opciones actuales para aclarar la referencia.
    candidatos_coincidentes = sum(1 for puntos in puntuaciones.values() if puntos == mejor_puntuacion and puntos > 0)
    descriptor = " ".join(_tokens_de_catalogo_mencionados(texto, productos_candidatos))
    if (
        plan.alcance_entidad == "producto"
        and candidatos_coincidentes > 1
        and descriptor
    ):
        plan.filtros.codigos = []
        plan.filtros.texto = descriptor
        plan.producto = None
        plan.codigo_catalogo_verificado = False
    return plan


def _quitar_codigo_inferido_de_consulta_colectiva(
    plan: Optional[PlanAgente],
    texto: str,
    productos_candidatos: List[Dict[str, str]],
) -> Optional[PlanAgente]:
    """Elimina un producto elegido por el modelo si el mensaje pide una colección."""
    if (
        plan is None
        or plan.tipo != "consulta"
        or plan.alcance_entidad not in {"coleccion", "global"}
        or plan.entidad_producto_mencionada
        or not plan.filtros.codigos
    ):
        return plan

    descriptor = " ".join(_tokens_de_catalogo_mencionados(texto, productos_candidatos))
    plan.filtros.codigos = []
    plan.producto = None
    plan.codigo_catalogo_verificado = False
    if descriptor:
        plan.filtros.texto = descriptor
    return plan


def _reconciliar_codigos_del_plan(
    plan: Optional[PlanAgente],
    texto: str,
    herramientas: HerramientasInventario,
) -> Optional[PlanAgente]:
    """No permite que un código inferido por el LLM suplante la entidad hablada."""
    if plan is None or plan.tipo != "consulta":
        return plan

    # Los códigos de una selección ordinal ya fueron resueltos contra los
    # resultados que el propio agente presentó. No son una inferencia del LLM
    # y no tienen por qué repetirse literalmente en el turno de seguimiento.
    if plan.fuente_plan == "fast_path_ordinal":
        return plan

    # These codes were checked against the current visible catalog candidates
    # by the semantic planner; they are grounded resolutions, not unsupported
    # code guesses that need to be repeated literally by the user.
    if plan.filtros.codigos and plan.codigo_catalogo_verificado:
        return plan

    from .herramientas import extraer_codigos_de_texto, normalizar_codigo_producto
    from .planificador import _extraer_objeto_consulta_sin_marco

    codigos_plan = list(plan.filtros.codigos or [])
    texto_filtro = str(plan.filtros.texto or "").strip()
    producto_filtro = str(plan.producto or "").strip()

    # Proteger contra fuga de formato del prompt de candidatos ("COD-001 | Nombre del producto")
    # cuando el usuario no escribió "|" ni dicho código.
    if "|" not in (texto or "") and ("|" in texto_filtro or "|" in producto_filtro):
        if "|" in texto_filtro:
            plan.filtros.texto = None
            texto_filtro = ""
        if "|" in producto_filtro:
            plan.producto = None
            producto_filtro = ""
        if not plan.filtros.codigos and not plan.filtros.texto:
            objeto_limpio = _extraer_objeto_consulta_sin_marco(texto or "")
            if objeto_limpio and not normalizar_codigo_producto(objeto_limpio):
                plan.filtros.texto = objeto_limpio
                texto_filtro = objeto_limpio

    codigo_texto_formado = bool(re.fullmatch(r"[A-Z]{2,5}[-_ ]?\d{1,5}", texto_filtro, re.IGNORECASE))
    codigo_producto_formado = bool(re.fullmatch(r"[A-Z]{2,5}[-_ ]?\d{1,5}", producto_filtro, re.IGNORECASE))
    if not (codigos_plan or codigo_texto_formado or codigo_producto_formado):
        return plan

    codigos_explicitos = {
        normalizar_codigo_producto(codigo)
        for codigo in extraer_codigos_de_texto(texto)
    }
    codigos_literales = set(herramientas.codigos_productos_mencionados(texto))
    codigos_autorizados = {codigo for codigo in codigos_explicitos | codigos_literales if codigo}

    codigo_texto = normalizar_codigo_producto(texto_filtro) if codigo_texto_formado else None
    codigo_producto = normalizar_codigo_producto(producto_filtro) if codigo_producto_formado else None

    codigos_invalidos = [codigo for codigo in codigos_plan if codigo not in codigos_autorizados]
    texto_es_codigo_inferido = bool(codigo_texto_formado and codigo_texto not in codigos_autorizados)
    producto_es_codigo_inferido = bool(codigo_producto_formado and codigo_producto not in codigos_autorizados)
    if not (codigos_invalidos or texto_es_codigo_inferido or producto_es_codigo_inferido):
        return plan

    plan.filtros.codigos = [codigo for codigo in codigos_plan if codigo in codigos_autorizados]
    if texto_es_codigo_inferido:
        plan.filtros.texto = None
    if producto_es_codigo_inferido:
        # En operaciones donde el usuario nombra un producto por sustantivo (ej. "100 osciloscopios")
        # y el LLM infiere el código real del catálogo (ej. "OSC-001"), validar que haya anclaje léxico
        # real en el mensaje para evitar borrar un producto legítimo o aceptar una alucinación sin relación.
        reemplazo_producto = None
        if plan.tipo == "operacion" and codigo_producto:
            from .herramientas import FiltrosConsulta
            res_cod_inf = herramientas.buscar_productos(FiltrosConsulta(codigos=[codigo_producto]))
            if not res_cod_inf.error and res_cod_inf.total == 1:
                nom_cat = str(res_cod_inf.items[0].get("nombre") or "")
                raices_excluidas = {
                    "producto", "product", "articul", "unidad", "inventari", "almacen",
                    "equip", "prestam", "devolucion", "entrad", "salid", "traslad",
                    "stock", "todo", "complet", "digital", "manual", "kit",
                }
                raices_cat = {r for r in tokens_significativos(nom_cat) if len(r) >= 4 and r not in raices_excluidas}
                palabras_msg = re.findall(r"[A-Za-zÁ-ÿñÑ0-9]+", texto or "")
                tokens_anclados = [
                    p for p in palabras_msg
                    if len(normalizar(p)) >= 4 and _raiz(normalizar(p)) in raices_cat
                ]
                if tokens_anclados:
                    busq_anclaje = herramientas.buscar_productos(FiltrosConsulta(texto=tokens_anclados[0]))
                    if not busq_anclaje.error and busq_anclaje.total == 1 and busq_anclaje.items[0].get("codigo") == codigo_producto:
                        reemplazo_producto = codigo_producto
                    else:
                        reemplazo_producto = " ".join(tokens_anclados[:3])
        plan.producto = reemplazo_producto
    if not plan.filtros.codigos and not plan.filtros.texto:
        objeto = _extraer_objeto_consulta_sin_marco(texto or "")
        if objeto and not normalizar_codigo_producto(objeto):
            plan.filtros.texto = objeto
    return plan


class AgenteInventarioHaku:
    @staticmethod
    def _aclarar_seguimiento_sin_sujeto(plan, texto, estado_operativo):
        """Resuelve menciones anafóricas y completa seguimientos con el antecedente único.

        La referencia se confirma con señales gramaticales y el solapamiento
        entre el descriptor del mensaje y el nombre del producto enfocado. Así,
        «ese multímetro» puede referir al MUL-001 ya consultado aunque existan
        otros multímetros. No reemplaza una colección explícita ni un filtro
        incompatible con el nombre del foco.
        """
        es_global_previa = (
            _es_consulta_global_inventario(texto or "")
            or bool(getattr(plan, "alcance_global_confirmado", False))
        )
        referencia_anforica_aplicada = False
        if (
            estado_operativo is not None
            and getattr(estado_operativo, "producto_codigo", None)
            and plan.tipo in {"consulta", "aclaracion"}
            and plan.referencia == "ninguna"
            and plan.alcance_entidad in {None, "producto", "coleccion", "indefinido"}
            and not es_global_previa
            and not plan.filtros.codigos
            and not plan.filtros.categoria
            and not plan.filtros.ubicacion
        ):
            texto_normalizado = normalizar(texto or "")
            texto_low_tildes = (texto or "").strip().lower()
            palabras = texto_normalizado.split()
            demostrativos_singulares = {"ese", "esa", "este", "esta", "aquel", "aquella"}
            modificadores_referenciales = {"mismo", "misma", "anterior"}
            nombre_foco = " ".join(filter(None, (
                getattr(estado_operativo, "producto_codigo", None),
                getattr(estado_operativo, "producto_nombre", None),
            )))
            raices_foco = set(tokens_significativos(nombre_foco))
            descriptor = plan.producto or plan.filtros.texto
            raices_descriptor = set(tokens_significativos(descriptor)) if descriptor else set()
            raices_descriptor -= demostrativos_singulares | modificadores_referenciales

            demostrativo_vinculado = any(
                palabra in demostrativos_singulares
                and (
                    (indice + 1 < len(palabras) and _raiz(palabras[indice + 1]) in raices_foco)
                    or (
                        indice + 2 < len(palabras)
                        and palabras[indice + 1] in modificadores_referenciales
                        and _raiz(palabras[indice + 2]) in raices_foco
                    )
                )
                for indice, palabra in enumerate(palabras)
            ) or any(
                palabra in modificadores_referenciales
                and indice + 1 < len(palabras)
                and _raiz(palabras[indice + 1]) in raices_foco
                for indice, palabra in enumerate(palabras)
            )

            # Un posesivo puede retomar el foco; los clíticos se reconocen por
            # su posición antes de una forma verbal, excluyendo sustantivos
            # plurales/acentuados («los préstamos») y artículos precedidos por
            # preposición o verbo transitivo («muéstrame los ...», «en la ...»).
            clitico_seguido_de_verbo = bool(re.search(
                r"(?<!\bde\s)(?<!\ben\s)(?<!\ba\s)(?<!\bcon\s)(?<!\bpor\s)(?<!\bpara\s)"
                r"(?<!\bver\s)(?<!\bdame\s)(?<!\bdime\s)(?<!\bmu[eé]strame\s)(?<!\bmu[eé]streme\s)"
                r"(?<!\bmostrame\s)(?<!\bens[eé]ñame\s)(?<!\bl[ií]stame\s)(?<!\bb[uú]scame\s)"
                r"(?<!\bson\s)(?<!\bhay\s)(?<!\btenemos\s)"
                r"\b(?:lo|la|los|las)\s+(?!pr[eé]stamos?\b|lista\b|entrada\b|salida\b|categor[ií]a\b|bodega\b|ficha\b)"
                r"[a-záéíóúñ]+(?:amos|emos|imos|aron|ieron|aban|ían|ió|ó|an|en|a|e|o)\b",
                texto_low_tildes,
            ))
            pronombre_vinculado = bool(
                set(palabras) & {"su", "sus"}
            ) or clitico_seguido_de_verbo
            descriptor_coincide_con_foco = bool(
                raices_descriptor
                and raices_foco
                and raices_descriptor.issubset(raices_foco)
            )
            pronombre_refiere_foco = bool(
                pronombre_vinculado
                and (plan.campos or plan.medida_solicitada)
                and (not plan.producto or descriptor_coincide_con_foco)
            )
            if (demostrativo_vinculado and descriptor_coincide_con_foco) or pronombre_refiere_foco:
                plan.referencia = "foco"
                plan.filtros.texto = None
                if plan.producto and descriptor_coincide_con_foco:
                    plan.producto = None
                plan.alcance_entidad = "producto"
                referencia_anforica_aplicada = True

        medida = plan.medida_solicitada
        # El modelo puede expresar la dimensión solicitada mediante el campo
        # estructurado y omitir la etiqueta equivalente. Recuperarla de ese
        # campo evita tratar una pregunta elíptica como una suma global.
        if not medida or medida == "atributo":
            campos = set(plan.campos or [])
            if campos == {"stock"}:
                medida = "disponible"
            elif campos == {"prestamos"}:
                medida = "pendiente_devolucion"

        es_global_explicita = (
            _es_consulta_global_inventario(texto or "")
            or bool(getattr(plan, "alcance_global_confirmado", False))
        )
        texto_norm_seg = normalizar(texto or "")
        if (
            not medida
            and estado_operativo is not None
            and getattr(estado_operativo, "producto_codigo", None)
            and not es_global_explicita
            and not any((plan.filtros.codigos, plan.filtros.texto, plan.filtros.categoria,
                         plan.filtros.almacen, plan.filtros.ubicacion))
            and not re.search(
                r"\b(?:que|cuales|cuantos|cuantas)\s+(?:productos?|articulos?|referencias?|herramientas?|materiales?|insumos?|equipos?)\b",
                texto_norm_seg,
            )
        ):
            if re.search(
                r"\b(?:prestad[oa]s?|pendientes?|devolver|devolucion|regres\w*|sin\s+devolver|"
                r"afuera|fuera\s+del?\s+(?:almacen|bodega|inventario))\b",
                texto_norm_seg,
            ):
                medida = "pendiente_devolucion"
            elif re.search(r"\b(?:disponibles?|libres?)\b", texto_norm_seg):
                medida = "disponible"

        _tiene_sustantivo_colectivo = bool(re.search(
            r"\b(?:productos?|articulos?|referencias?|items?|unidades?|existencias?|"
            r"stock|inventario|equipos?|herramientas?|materiales?|insumos?|bienes?|total|todo|todos|todas)\b",
            texto_norm_seg,
        ))
        _es_elipsis_cantidad_en_almacen = bool(
            plan.filtros.almacen
            and not any((plan.filtros.codigos, plan.filtros.texto, plan.filtros.categoria, plan.filtros.ubicacion))
            and not _tiene_sustantivo_colectivo
            and re.search(
                r"\b(?:cuantos|cuantas|cuanto)\s+(?:hay|quedan|tenemos|disponibles?|prestados?|prestadas?)\b",
                texto_norm_seg,
            )
        )
        if not medida and _es_elipsis_cantidad_en_almacen:
            if re.search(r"\b(?:prestad[oa]s?|pendientes?|devolver)\b", texto_norm_seg):
                medida = "pendiente_devolucion"
            else:
                medida = "disponible"

        if (
            estado_operativo is not None
            and getattr(estado_operativo, "producto_codigo", None)
            and plan.tipo in {"consulta", "aclaracion"}
            and plan.operacion in {None, "sumar_unidades", "contar", "listar", "detalle"}
            and medida in {"disponible", "pendiente_devolucion"}
            and (plan.referencia == "ninguna" or referencia_anforica_aplicada)
            and plan.filtros.condicion in (None, "con_stock", "con_prestamos")
            # A nonempty text/category/location filter is the model's explicit
            # target (including a requested inventory-wide collection). Keep it.
            # Allow an explicit warehouse filter ONLY for bare elliptical focus
            # follow-ups («¿Cuántos hay en el Almacén A?») without collective nouns.
            and not any((plan.filtros.codigos, plan.filtros.texto, plan.filtros.categoria,
                         plan.filtros.ubicacion))
            and (not plan.filtros.almacen or _es_elipsis_cantidad_en_almacen)
            # The model sometimes translates a requested measure into a generic
            # positivity condition. With a unique focus, an unfiltered measure
            # continues that product; an explicit collection keeps its filter.
            and not es_global_explicita
        ):
            plan.tipo = "consulta"
            plan.referencia = "foco"
            plan.alcance_entidad = "producto"
            plan.medida_solicitada = medida
            plan.pregunta = None
            if plan.operacion in {None, "listar", "contar"} and medida in {"disponible", "pendiente_devolucion"}:
                plan.operacion = "sumar_unidades"
            if medida == "disponible":
                plan.campos = ["stock"]
                plan.alcance = "actual"
                plan.filtros.condicion = None
            elif medida == "pendiente_devolucion":
                plan.campos = ["prestamos"]
                plan.alcance = "actual"
                plan.filtros.condicion = None
        return plan

    @classmethod
    def _obtener_mensaje_error_pantalla(cls, estado: str, detalle: str = "") -> str:
        if estado == "LLM_ERROR_DISPONIBILIDAD":
            return (
                "⚠️ **Servicio no disponible**\n\n"
                "El servicio central de procesamiento lingüístico no está disponible en este momento. "
                "Por favor, intenta nuevamente más tarde o contacta al administrador del sistema."
            )
        elif estado == "LLM_ERROR_TIMEOUT":
            return (
                "⚠️ **Tiempo de espera agotado**\n\n"
                "El servicio central de procesamiento lingüístico excedió el tiempo límite de espera. "
                "Por favor, formula una solicitud más corta o intenta nuevamente."
            )
        elif estado == "LLM_ERROR_TRANSPORTE":
            return (
                "⚠️ **Error de comunicación**\n\n"
                "Ocurrió un error al comunicarse con el servicio central de procesamiento lingüístico. "
                "Por favor, intenta nuevamente en unos momentos."
            )
        elif estado == "LLM_ERROR_RESPUESTA":
            return (
                "⚠️ **Error en la respuesta del modelo**\n\n"
                "El servicio central devolvió una respuesta no estructurada o corrupta. "
                "Por favor, intenta formular la solicitud con otras palabras."
            )
        elif estado == "LLM_ERROR_PLAN":
            return (
                "⚠️ **Plan no interpretable**\n\n"
                "El modelo central no generó un plan operativo compatible con las funciones de inventario. "
                "Por favor, reformula tu solicitud."
            )
        return (
            "⚠️ **Error de procesamiento**\n\n"
            "No fue posible procesar la solicitud debido a un problema con el servicio lingüístico central."
        )

    @classmethod
    def _obtener_mensaje_error_verbal(cls, estado: str) -> str:
        if estado == "LLM_ERROR_DISPONIBILIDAD":
            return "El servicio central de procesamiento lingüístico no está disponible en este momento. Por favor intenta más tarde."
        elif estado == "LLM_ERROR_TIMEOUT":
            return "El servicio central de procesamiento lingüístico excedió el tiempo límite de espera. Por favor intenta de nuevo."
        elif estado == "LLM_ERROR_TRANSPORTE":
            return "Ocurrió un error al comunicarse con el servicio central de procesamiento lingüístico. Por favor intenta de nuevo."
        elif estado == "LLM_ERROR_RESPUESTA":
            return "El servicio central devolvió una respuesta no estructurada. Por favor intenta con otras palabras."
        elif estado == "LLM_ERROR_PLAN":
            return "El modelo no generó un plan operativo compatible. Por favor reformula tu solicitud."
        return "No fue posible procesar la solicitud con el servicio central. Por favor intenta más tarde."

    @classmethod
    def _construir_respuesta_error_terminal(
        cls,
        estado: str,
        codigo_http: int,
        mensaje_pantalla: str,
        mensaje_verbal: str,
        llamadas_fisicas: int,
    ) -> Dict[str, Any]:
        return {
            "tipo": "error",
            "mensaje_pantalla": mensaje_pantalla,
            "mensaje_verbal": mensaje_verbal,
            "datos": {
                "error": estado,
                "codigo": estado,
                "codigo_http": codigo_http,
                "error_terminal": True,
                "llamadas_fisicas": llamadas_fisicas,
            },
        }

    @classmethod
    def procesar(
        cls,
        texto: str,
        usuario: Any,
        estado_operativo: Any = None,
        sesion: Optional[dict] = None,
        interaction_id: Optional[str] = None,
        cliente_llm: Any = None,
    ) -> SalidaAgente:
        correlacion = interaction_id or uuid.uuid4().hex[:12]
        salida = SalidaAgente()

        # Fast-path social determinista: 0 inferencias si es un saludo puro
        from ..lenguaje.interprete_semantico import InterpreteSemanticoHaku
        saludo_fast = InterpreteSemanticoHaku._evaluar_fast_path_social(texto)
        if saludo_fast is not None:
            METRICAS_AGENTE["resueltas_sin_llm"] += 1
            salida.estado_terminal = "FAST_PATH_PURO"
            salida.inferencia_realizada = False
            salida.intento_inferencia_fisica = False
            salida.codigo_http = 200
            from django.utils import timezone as _tz_saludo
            _txt_sal_l = (texto or "").lower()
            if re.search(r"\bbuenos?\s+d[ií]as?\b", _txt_sal_l):
                _pref_sal = "Buenos días."
            elif re.search(r"\bbuenas?\s+tardes?\b", _txt_sal_l):
                _pref_sal = "Buenas tardes."
            elif re.search(r"\bbuenas?\s+noches?\b", _txt_sal_l):
                _pref_sal = "Buenas noches."
            else:
                _hora_loc = _tz_saludo.localtime().hour
                _pref_sal = "Buenos días." if _hora_loc < 12 else ("Buenas tardes." if _hora_loc < 19 else "Buenas noches.")
            msg = (
                "Con gusto. Si necesita consultar algo más del inventario, estamos a su disposición."
                if getattr(saludo_fast, "intencion", None) == "agradecimiento"
                else f"{_pref_sal} Hacu Inventario está listo para atenderle. ¿Qué consulta u operación desea realizar?"
            )
            from ..services import normalizar_para_tts as _norm_tts_sal
            salida.respuesta = {
                "tipo": "conversacion",
                "mensaje_pantalla": msg,
                "mensaje_verbal": _norm_tts_sal(msg),
                "datos": {"agente": {"fuente": "fast_path_social", "inferencias": 0, "estado": "FAST_PATH_PURO"}},
            }
            return salida

        # Aislamiento estricto de contexto entre usuarios distintos
        if estado_operativo is not None and getattr(estado_operativo, "usuario_id", None) not in (
            None,
            getattr(usuario, "id", None),
        ):
            estado_operativo = None

        t0 = time.perf_counter()
        herramientas = HerramientasInventario(usuario)

        # Fast-path determinista (0 inferencias) para códigos directos aislados y selección ordinal sobre candidatos
        permitir_fast_path = (
            cliente_llm is None
            or getattr(cliente_llm, "permitir_fast_path", False) is True
        )
        plan: Optional[PlanAgente] = None
        cliente = cliente_llm
        usar_cache = False
        productos_candidatos: List[Dict[str, str]] = []
        codigos_producto_literal: Optional[List[str]] = None
        categorias_fast_path: Optional[List[str]] = None
        almacenes_fast_path: Optional[List[str]] = None
        if permitir_fast_path:
            plan = _crear_plan_comparacion_stock(_codigos_comparacion_stock(texto, herramientas), texto)
            if plan is None:
                if requiere_catalogo_categorias_condicion_stock(texto):
                    categorias_fast_path = herramientas.nombres_categorias()
                if requiere_catalogo_almacenes_condicion_stock(texto):
                    almacenes_fast_path = herramientas.nombres_almacenes()
                plan = planificar_determinista(
                    texto,
                    estado_operativo,
                    categorias_accesibles=categorias_fast_path,
                    almacenes_accesibles=almacenes_fast_path,
                )
            plan = _usar_codigo_para_producto_literal(plan, texto, herramientas)
            if (
                plan is not None
                and plan.fuente_plan == "fast_path_stock_explicito"
                and plan.filtros.texto
                and not plan.filtros.codigos
            ):
                # This parser only recognizes a question shape; it cannot prove
                # that an indirect description names a live product. Delegate
                # that case to the model instead of returning a false zero from
                # a literal text search.
                plan = None

        categorias_para_normalizar = categorias_fast_path or []
        if plan is not None:
            METRICAS_AGENTE["resueltas_sin_llm"] += 1
            METRICAS_AGENTE["planes_validos"] += 1
            salida.inferencia_realizada = False
            salida.intento_inferencia_fisica = False
            salida.estado_terminal = "FAST_PATH_PURO"
            salida.codigo_http = 200
        else:
            cliente = cliente_llm if cliente_llm is not None else _obtener_cliente()
            disponible = False
            if cliente is not None:
                try:
                    salud = cliente.comprobar_salud()
                    disponible = bool(salud.get("disponible") if isinstance(salud, dict) else salud)
                except Exception as e_salud:
                    logger.warning("[HakuAgente] corr=%s error al comprobar salud del Cerebro Central: %s", correlacion, e_salud)
                    disponible = False

            if not disponible:
                salida.error_terminal = True
                salida.estado_terminal = "LLM_ERROR_DISPONIBILIDAD"
                salida.codigo_http = 503
                salida.inferencia_realizada = False
                salida.intento_inferencia_fisica = False
                salida.traza = {"correlacion": correlacion, "error": "LLM_ERROR_DISPONIBILIDAD"}
                salida.respuesta = cls._construir_respuesta_error_terminal(
                    estado="LLM_ERROR_DISPONIBILIDAD",
                    codigo_http=503,
                    mensaje_pantalla=cls._obtener_mensaje_error_pantalla("LLM_ERROR_DISPONIBILIDAD"),
                    mensaje_verbal=cls._obtener_mensaje_error_verbal("LLM_ERROR_DISPONIBILIDAD"),
                    llamadas_fisicas=0,
                )
                return salida

            almacenes_accesibles = (
                almacenes_fast_path
                if almacenes_fast_path is not None
                else herramientas.nombres_almacenes()
            )
            ubicaciones_accesibles = herramientas.nombres_ubicaciones()
            categorias_accesibles = (
                categorias_fast_path
                if categorias_fast_path is not None
                else herramientas.nombres_categorias()
            )
            categorias_para_normalizar = categorias_accesibles
            productos_candidatos = herramientas.candidatos_producto_para_mensaje(texto)
            codigos_producto_literal = herramientas.codigos_productos_mencionados(texto)
            # A búsqueda semántica puede no incluir el producto cuyo nombre
            # completo escribió la persona. El índice vivo sí puede resolver
            # menciones literales con el mismo alcance RBAC; pásalas como señal
            # de entidad para que el planificador aplique desde la primera
            # inferencia las reglas de varios atributos del mismo producto.
            codigos_candidatos_semanticos = [
                str(producto.get("codigo") or "").strip()
                for producto in productos_candidatos
                if producto.get("codigo")
            ]
            # Una mención literal y única es evidencia más fuerte que los
            # candidatos aproximados del índice: conserva los demás para
            # descripciones indirectas o ambiguas, pero no hace que una
            # referencia exacta parezca una colección de productos similares.
            codigos_producto_contexto = list(dict.fromkeys(
                codigos_producto_literal or codigos_candidatos_semanticos
            ))
            contexto_txt = construir_contexto(
                estado_operativo,
                categorias_accesibles,
                almacenes_accesibles,
                texto_usuario=texto,
                ubicaciones=ubicaciones_accesibles,
                productos_candidatos=productos_candidatos,
            )
            usar_cache = (cliente_llm is None) or bool(getattr(cliente_llm, "usar_cache_plan", False))
            telemetria: Dict[str, Any] = {
                "estado": "NO_INTENTADA",
                "llamadas_fisicas": 0,
                "correlacion": correlacion,
            }
            seguimiento_con_foco = bool(
                estado_operativo is not None
                and getattr(estado_operativo, "producto_codigo", None)
                and re.search(
                    r"\b(?:lo|la)\s+[a-zñ]{4,}(?:amos|emos|imos|aron|ieron|aban|ian|io|an|en)\b",
                    normalizar(texto or ""),
                )
            )
            if seguimiento_con_foco:
                telemetria["seguimiento_foco_anaforico"] = True
            try:
                plan = planificar(
                    texto=texto,
                    cliente_llm=cliente,
                    contexto_txt=contexto_txt,
                    usar_cache=usar_cache,
                    telemetria=telemetria,
                    almacenes_accesibles=almacenes_accesibles,
                    ubicaciones_accesibles=ubicaciones_accesibles,
                    categorias_accesibles=categorias_accesibles,
                    codigos_productos_candidatos=codigos_producto_contexto,
                )
            except ErrorInferenciaLLM as e_inf:
                logger.warning("[HakuAgente] corr=%s fallo de inferencia [%s]: %s", correlacion, e_inf.estado, e_inf.mensaje)
                METRICAS_AGENTE["inferencias"] += telemetria.get("llamadas_fisicas", 0)
                METRICAS_AGENTE["planes_invalidos"] += 1
                salida.error_terminal = True
                salida.estado_terminal = e_inf.estado
                salida.codigo_http = e_inf.codigo_http
                salida.inferencia_realizada = bool(telemetria.get("llamadas_fisicas", 0) > 0)
                salida.intento_inferencia_fisica = salida.inferencia_realizada
                salida.traza = {"correlacion": correlacion, "error": e_inf.estado, "detalle": str(e_inf)}
                salida.respuesta = cls._construir_respuesta_error_terminal(
                    estado=e_inf.estado,
                    codigo_http=e_inf.codigo_http,
                    mensaje_pantalla=cls._obtener_mensaje_error_pantalla(e_inf.estado, e_inf.mensaje),
                    mensaje_verbal=cls._obtener_mensaje_error_verbal(e_inf.estado),
                    llamadas_fisicas=telemetria.get("llamadas_fisicas", 0),
                )
                return salida
            except Exception as e_inesperado:
                exc_str = str(e_inesperado).lower()
                if "timeout" in exc_str or isinstance(e_inesperado, TimeoutError):
                    estado_err = "LLM_ERROR_TIMEOUT"
                    cod_http = 504
                else:
                    estado_err = "LLM_ERROR_TRANSPORTE"
                    cod_http = 502
                logger.warning("[HakuAgente] corr=%s error inesperado: %s", correlacion, type(e_inesperado).__name__, exc_info=True)
                METRICAS_AGENTE["inferencias"] += 1
                METRICAS_AGENTE["planes_invalidos"] += 1
                salida.error_terminal = True
                salida.estado_terminal = estado_err
                salida.codigo_http = cod_http
                salida.inferencia_realizada = True
                salida.intento_inferencia_fisica = True
                salida.traza = {"correlacion": correlacion, "error": estado_err}
                salida.respuesta = cls._construir_respuesta_error_terminal(
                    estado=estado_err,
                    codigo_http=cod_http,
                    mensaje_pantalla=cls._obtener_mensaje_error_pantalla(estado_err),
                    mensaje_verbal=cls._obtener_mensaje_error_verbal(estado_err),
                    llamadas_fisicas=1,
                )
                return salida

            if plan is None:
                salida.error_terminal = True
                salida.estado_terminal = "LLM_ERROR_PLAN"
                salida.codigo_http = 502
                salida.inferencia_realizada = True
                salida.intento_inferencia_fisica = True
                salida.respuesta = cls._construir_respuesta_error_terminal(
                    estado="LLM_ERROR_PLAN",
                    codigo_http=502,
                    mensaje_pantalla=cls._obtener_mensaje_error_pantalla("LLM_ERROR_PLAN"),
                    mensaje_verbal=cls._obtener_mensaje_error_verbal("LLM_ERROR_PLAN"),
                    llamadas_fisicas=1,
                )
                return salida

            llamadas_plan_fisicas = int(telemetria.get("llamadas_fisicas", 0) or 0)
            if plan.fuente_plan == "cache_plan" and llamadas_plan_fisicas == 0:
                METRICAS_AGENTE["cache_hits"] += 1
                METRICAS_AGENTE["resueltas_sin_llm"] += 1
                salida.estado_terminal = "CACHE_HIT"
                salida.inferencia_realizada = False
                salida.intento_inferencia_fisica = False
                salida.codigo_http = 200
            else:
                METRICAS_AGENTE["inferencias"] += max(1, llamadas_plan_fisicas)
                METRICAS_AGENTE["chars_prompt_total"] += plan.chars_prompt
                METRICAS_AGENTE["tokens_completados_total"] += plan.tokens_completados
                salida.estado_terminal = "LLM_RESPUESTA_VALIDA"
                salida.inferencia_realizada = True
                salida.intento_inferencia_fisica = True
                salida.codigo_http = 200
            METRICAS_AGENTE["planes_validos"] += 1

        if plan.fuente_plan != "fast_path_comparacion_productos":
            plan = _reconciliar_producto_literal_catalogo(
                plan,
                texto,
                herramientas,
                codigos_producto_literal=codigos_producto_literal,
            )
            plan = _anclar_producto_literal_operacion(plan, texto, herramientas)
            plan = _reconciliar_producto_por_descriptor_catalogo(
                plan, texto, productos_candidatos,
            )
            plan = _quitar_codigo_inferido_de_consulta_colectiva(
                plan, texto, productos_candidatos,
            )
            plan = _reconciliar_codigos_del_plan(plan, texto, herramientas)
            plan = _corregir_comparacion_stock_literal(plan, texto, herramientas)

        # Si el LLM duplicó en `filtros.texto` únicamente palabras estructurales o tokens
        # ya capturados por `filtros.almacen` / `filtros.ubicacion` / `filtros.categoria`
        # (ej. filtros.texto="almacén" junto a filtros.almacen="Almacén A"), limpiar `texto`
        # para evitar buscar productos llamados literalmente «almacén» y devolver 0 falso.
        if plan.tipo == "consulta" and plan.filtros.texto and not plan.filtros.codigos:
            _toks_filtro_dup = [w for w in normalizar(plan.filtros.texto).split() if w]
            _toks_estructurales = {"el", "la", "los", "las", "de", "del", "en"}
            if plan.filtros.almacen:
                _toks_estructurales.update(normalizar(plan.filtros.almacen).split())
                _toks_estructurales.update({"almacen", "almacenes", "bodega", "bodegas", "deposito", "depositos"})
            if plan.filtros.ubicacion:
                _toks_estructurales.update(normalizar(plan.filtros.ubicacion).split())
                _toks_estructurales.update({"ubicacion", "ubicaciones"})
            if plan.filtros.categoria:
                _toks_estructurales.update({"categoria", "categorias"})
            if _toks_filtro_dup and all(w in _toks_estructurales for w in _toks_filtro_dup):
                plan.filtros.texto = None

        # Si el planificador (determinista o LLM) extrajo solo adverbios/participios de estado
        # («ubicado actualmente», «guardado», «antes», «ahora») como texto de producto mientras
        # existe un producto único en foco, resolver como seguimiento elíptico sobre ese foco.
        if (
            plan.tipo == "consulta"
            and plan.filtros.texto
            and not plan.filtros.codigos
            and estado_operativo is not None
            and getattr(estado_operativo, "producto_codigo", None)
        ):
            _palabras_elipticas = {
                "antes", "ahora", "ahorita", "actualmente", "hoy", "entonces", "luego", "despues",
                "ubicado", "ubicada", "ubicados", "ubicadas", "guardado", "guardada", "guardados", "guardadas",
                "almacenado", "almacenada", "disponible", "disponibles", "actual", "actuales",
                "aqui", "aca", "alli", "alla", "ahi", "este", "esta", "esto", "ese", "esa", "eso",
                "mismo", "misma", "favor", "dime", "diga", "digame",
            }
            _toks_f = [w for w in normalizar(plan.filtros.texto).split() if w]
            if _toks_f and all(w in _palabras_elipticas for w in _toks_f):
                plan.filtros.texto = None
                plan.filtros.codigos = [estado_operativo.producto_codigo]
                plan.referencia = "foco"
                plan.alcance_entidad = "producto"

        plan = cls._aclarar_seguimiento_sin_sujeto(plan, texto, estado_operativo)
        plan = aclarar_agregado_sin_alcance_explicito(texto, plan)
        # La resolución literal puede completar aquí el código y el alcance
        # individual que el planificador no pudo confirmar antes. Normaliza la
        # cantidad solicitada y retira solo el filtro colectivo que ocultaría
        # un resultado válido con cero unidades.
        plan = _normalizar_intencion_suma_unidades(
            texto, plan, categorias_para_normalizar,
        )
        plan = _quitar_con_stock_de_producto_individual(plan)
        # Reaplica la selección de máximo/mínimo después de reconciliar las
        # entidades actuales del catálogo. Esa reconciliación puede ajustar el
        # alcance, pero no debe reducir una consulta extremal a una ficha.
        plan = _normalizar_intencion_extremal(texto, plan)
        # Las reparaciones de alcance a nivel del agente pueden volver a
        # adjuntar columnas accesorias; impón de nuevo el saldo tipado antes
        # de ejecutar la consulta.
        plan = _normalizar_saldo_cuantitativo_final(plan)

        # Rescate determinista: si el LLM clasificó una orden explícita de mutación
        # (ej. «Quiero hacer un préstamo del taladro», «Crea un producto», «Registra una salida...»)
        # como consulta o aclaración, convertirla al plan de operación correspondiente
        # preservando el producto/código/cantidad/destinatario/almacén detectados.
        if (
            plan.tipo in {"consulta", "aclaracion"}
            and not plan.fuente_plan.startswith("fast_path_")
            and texto
            and "?" not in texto
            and "¿" not in texto
            and not _es_pregunta_como_operar(texto)
            and not _es_pregunta_permiso_capacidad(texto)
            and not _es_negacion_operacion(texto)
            and not re.search(
                r"\b(?:cu[aá]nt[oa]s?|cu[aá]l(?:es)?|qui[eé]n(?:es)?|d[oó]nde|c[oó]mo|"
                r"qu[eé]\s+(?:hay|tenemos|productos|pr[eé]stamos|movimientos|almacenes|bodegas|ubicaciones|categor[ií]as)|"
                r"mu[eé]str\w*|ense[nñ]\w*|dime|ver\b|list[aeo]\w*|historial|reporte|consulta\w*|revisar|buscar)\b",
                texto,
                re.IGNORECASE,
            )
        ):
            _rescate_accion = None
            _pref_imp = r"^\s*(?:(?:oye|oiga|hacu|jacu|por\s+favor)\s+)*"
            if re.search(
                r"\b(?:(?:quiero|necesito|deseo|voy\s+a|vamos\s+a)\s+(?:hacer|registrar|solicitar|crear|sacar)?\s*(?:un\s+)?pr[eé]stamo|"
                r"(?:quiero|necesito|deseo|voy\s+a)\s+prestar|pr[eé]sta(?:me|le)|"
                r"(?:haz(?:me)?|haga(?:me)?|registra(?:r)?)\s+(?:un\s+)?pr[eé]stamo)\b|"
                + _pref_imp + r"presta(?:me|le|nos)?\b",
                texto,
                re.IGNORECASE,
            ):
                _rescate_accion = "prestamo"
            elif re.search(
                r"\b(?:(?:quiero|necesito|deseo|voy\s+a)\s+(?:hacer|registrar)?\s*(?:una\s+)?devoluci[oó]n|"
                r"(?:quiero|necesito|deseo|voy\s+a)\s+devolver|"
                r"(?:haz(?:me)?|haga(?:me)?|registra(?:r)?)\s+(?:una\s+)?devoluci[oó]n)\b|"
                + _pref_imp + r"devuelv(?:e|eme|a|ame)\b",
                texto,
                re.IGNORECASE,
            ):
                _rescate_accion = "devolucion"
            elif re.search(
                r"\b(?:(?:quiero|necesito|deseo|voy\s+a)\s+(?:hacer|registrar)?\s*(?:una\s+)?entrada|"
                r"(?:quiero|necesito|deseo|voy\s+a)\s+ingresar|"
                r"(?:haz(?:me)?|haga(?:me)?|registra(?:r)?)\s+(?:una\s+)?entrada)\b|"
                + _pref_imp + r"(?:ingres(?:a|ame|e)|met(?:e|eme|a))\b",
                texto,
                re.IGNORECASE,
            ):
                _rescate_accion = "entrada"
            elif re.search(
                r"\b(?:(?:quiero|necesito|deseo|voy\s+a)\s+(?:hacer|registrar)?\s*(?:una\s+)?(?:salida|dotaci[oó]n)|"
                r"(?:quiero|necesito|deseo|voy\s+a)\s+(?:sacar|retirar|despachar)|"
                r"(?:haz(?:me)?|haga(?:me)?|registra(?:r)?)\s+(?:una\s+)?(?:salida|dotaci[oó]n))\b|"
                + _pref_imp + r"(?:sac(?:a|ame|ale|e)|retir(?:a|ame|e)|despach(?:a|ame|e))\b",
                texto,
                re.IGNORECASE,
            ):
                _rescate_accion = "salida"
            elif re.search(
                r"\b(?:(?:quiero|necesito|deseo|voy\s+a)\s+(?:hacer|registrar)?\s*(?:un\s+)?traslado|"
                r"(?:quiero|necesito|deseo|voy\s+a)\s+trasladar|"
                r"(?:haz(?:me)?|haga(?:me)?|registra(?:r)?)\s+(?:un\s+)?traslado)\b|"
                + _pref_imp + r"traslad(?:a|ame|e)\b",
                texto,
                re.IGNORECASE,
            ):
                _rescate_accion = "traslado"
            elif re.search(
                r"\b(?:(?:quiero|necesito|deseo|voy\s+a)\s+(?:crear|registrar|agregar|a[nñ]adir|dar\s+de\s+alta)|"
                r"crea(?:r)?|registra(?:r)?|agrega(?:r)?|a[nñ]ade|dar\s+de\s+alta|da\s+de\s+alta)\s+"
                r"(?:un\s+|el\s+)?(?:nuevo\s+)?(?:producto|art[ií]culo)\b",
                texto,
                re.IGNORECASE,
            ):
                _rescate_accion = "crear_producto"
            elif re.search(
                r"\b(?:(?:quiero|necesito|deseo|voy\s+a)\s+(?:inhabilitar|desactivar|deshabilitar|archivar)|"
                r"inhabil[ií]t(?:a|e|alo|ala|ar)|desact[ií]v(?:a|e|alo|ala|ar)|deshabil[ií]t(?:a|e|alo|ala|ar)|arch[ií]v(?:a|e|alo|ala|ar))\b",
                texto,
                re.IGNORECASE,
            ):
                _rescate_accion = "inhabilitar"
            elif re.search(
                r"\b(?:(?:quiero|necesito|deseo|voy\s+a)\s+(?:habilitar|reactivar)|"
                r"(?<!in)(?<!des)habil[ií]t(?:a|e|alo|ala|ar)|react[ií]v(?:a|e|alo|ala|ar))\b",
                texto,
                re.IGNORECASE,
            ):
                _rescate_accion = "habilitar"

            if _rescate_accion:
                from ..lenguaje.numeros import extraer_cantidad_operacion_texto
                plan.tipo = "operacion"
                plan.operacion = None
                plan.accion = _rescate_accion
                if _rescate_accion == "crear_producto":
                    plan.producto = None
                    plan.filtros.codigos = []
                    plan.filtros.texto = None
                    plan.referencia = "ninguna"
                else:
                    plan.producto = (
                        (plan.filtros.codigos[0] if plan.filtros.codigos else None)
                        or plan.producto
                        or plan.filtros.texto
                    )
                    if plan.cantidad is None and _rescate_accion not in ("inhabilitar", "habilitar"):
                        plan.cantidad = extraer_cantidad_operacion_texto(texto)
                plan.campos = []
                plan.fuente_plan = "rescate_operacion_explicita"
                if _rescate_accion != "crear_producto":
                    plan = _anclar_producto_literal_operacion(plan, texto, herramientas)

        resultado_consulta_prevalidado = None
        filtros_plan = plan.filtros
        verificacion_intencion: Optional[Dict[str, Any]] = None
        filtros_sin_entidad = bool(
            not filtros_plan.codigos
            and not filtros_plan.categoria
            and not filtros_plan.almacen
            and not filtros_plan.ubicacion
            and not filtros_plan.condicion
            and filtros_plan.es_prestable is None
            and plan.referencia == "ninguna"
        )
        categoria_inferida = bool(
            plan.tipo == "consulta"
            and filtros_plan.categoria
            and not filtros_plan.categoria_explicita
        )
        tiene_campos_operativos = bool(
            plan.tipo == "consulta"
            and (
                plan.cantidad is not None
                or plan.valor is not None
                or plan.destinatario
                or plan.destino
                or plan.tipo_movimiento
                or plan.orden_movimiento
            )
        )
        # A product code/name is only the entity, not proof the task is read-only.
        # The primary plan can mistake an edit/enable/disable request for a product
        # detail query, so let the semantic intent verifier check grounded details too.
        consulta_detalle_de_entidad = bool(
            plan.tipo == "consulta"
            and plan.operacion == "detalle"
            and plan.referencia == "ninguna"
            and (filtros_plan.codigos or filtros_plan.texto)
        )
        es_consulta_eventos_global_plan = bool(
            plan.tipo == "consulta"
            and plan.alcance == "evento"
            and "movimientos" in (plan.campos or [])
            and not filtros_plan.codigos
            and not filtros_plan.texto
            and plan.referencia == "ninguna"
            and (plan.fecha_desde or plan.fecha_hasta or plan.alcance_entidad == "global")
        )
        requiere_segunda_clasificacion = bool(
            plan.tipo in {"consulta", "aclaracion"}
            # Los planes deterministas ya clasificaron la tarea a partir de una
            # forma estructurada; no intentes validarlos con un cliente que no
            # participó en el fast-path ni vuelvas a llamar al modelo.
            and not plan.fuente_plan.startswith("fast_path_")
            and not getattr(plan, "_cache_final_verificado", False)
            and not es_consulta_eventos_global_plan
            and (
                (
                    (filtros_sin_entidad or categoria_inferida or tiene_campos_operativos)
                    and not _es_consulta_global_inventario(texto)
                    and (
                        plan.tipo == "aclaracion"
                        or bool(filtros_plan.texto)
                        or categoria_inferida
                        or tiene_campos_operativos
                        or (plan.tipo == "consulta" and plan.operacion in {"detalle", "explicar_estado"})
                    )
                )
                or consulta_detalle_de_entidad
            )
        )
        if requiere_segunda_clasificacion and filtros_plan.texto:
            resultado_consulta_prevalidado = herramientas.buscar_productos(filtros_plan)
            if resultado_consulta_prevalidado.error or (
                resultado_consulta_prevalidado.total > 0
                and not tiene_campos_operativos
                and not categoria_inferida
                and not consulta_detalle_de_entidad
            ):
                requiere_segunda_clasificacion = False

        if requiere_segunda_clasificacion:
            decision = None
            metricas_verificacion: Dict[str, Any] = {}
            try:
                decision, metricas_verificacion = verificar_intencion_operativa_sin_entidad(
                    texto, cliente, timeout=None,
                )
            except Exception as exc_verificacion:
                logger.warning(
                    "[HakuAgente] corr=%s no se pudo validar la intención sin entidad: %s",
                    correlacion, type(exc_verificacion).__name__,
                )

            if isinstance(decision, dict):
                decision = normalizar_decision_intencion_operativa(
                    decision,
                    categorias_catalogo=herramientas.nombres_categorias(),
                    almacenes_catalogo=herramientas.nombres_almacenes(),
                    ubicaciones_catalogo=herramientas.nombres_ubicaciones(),
                    texto_usuario=texto,
                )

            if metricas_verificacion:
                plan.llm_ms += float(metricas_verificacion.get("duracion_ms", 0.0) or 0.0)
                plan.llm_cola_ms += float(metricas_verificacion.get("cola_ms", 0.0) or 0.0)
                plan.chars_prompt += int(metricas_verificacion.get("chars_prompt", 0) or 0)
                plan.tokens_estimados_prompt += int(metricas_verificacion.get("tokens_prompt", 0) or 0)
                plan.tokens_completados += int(metricas_verificacion.get("tokens_completados", 0) or 0)
                salida.inferencia_realizada = True
                salida.intento_inferencia_fisica = True
                METRICAS_AGENTE["inferencias"] += 1
                METRICAS_AGENTE["chars_prompt_total"] += int(metricas_verificacion.get("chars_prompt", 0) or 0)
                METRICAS_AGENTE["tokens_completados_total"] += int(metricas_verificacion.get("tokens_completados", 0) or 0)

            clase = decision.get("clase") if isinstance(decision, dict) else None
            accion = _str(decision.get("accion")) if isinstance(decision, dict) else None
            verificacion_intencion = {
                "clase": clase,
                "accion": accion,
                "latencia_ms": round(float(metricas_verificacion.get("duracion_ms", 0.0) or 0.0), 1),
            }
            accion_valida = accion in ACCIONES_VALIDAS and _texto_respalda_accion_operativa(texto, accion)
            if _es_pregunta_como_operar(texto) or _es_pregunta_permiso_capacidad(texto):
                plan.tipo = "capacidades"
                plan.operacion = None
                plan.accion = None
                plan.filtros = FiltrosConsulta()
                plan.campos = []
                plan.referencia = "ninguna"
                plan.fuente_plan = "guarda_capacidades_procedimiento"
            elif _es_negacion_operacion(texto):
                plan.tipo = "aclaracion"
                plan.operacion = None
                plan.accion = None
                plan.filtros = FiltrosConsulta()
                plan.campos = []
                plan.referencia = "ninguna"
                plan.pregunta = (
                    "Entendido, no se realizará ninguna operación en el inventario. "
                    "¿Desea consultar algún dato?"
                )
                plan.fuente_plan = "guarda_negacion_operacion"
            elif decision and clase in {"operacion", "formulario"} and accion_valida:
                def _entero_opcional(valor: Any, permitir_cero: bool = False) -> Optional[int]:
                    if isinstance(valor, bool) or not isinstance(valor, int):
                        return None
                    return valor if (valor >= 0 if permitir_cero else valor > 0) else None

                plan.tipo = "operacion"
                plan.operacion = None
                plan.accion = accion
                plan.producto = _str(decision.get("producto"))
                plan.cantidad = _entero_opcional(
                    decision.get("stock_inicial") if accion == "crear_producto" else decision.get("cantidad"),
                    permitir_cero=(accion == "crear_producto"),
                )
                plan.destinatario = _str(decision.get("destinatario"))
                plan.destino = _str(decision.get("destino"))
                plan.valor = _entero_opcional(decision.get("stock_minimo"), permitir_cero=True)
                plan.filtros = FiltrosConsulta(
                    almacen=_str(decision.get("almacen")),
                    categoria=_str(decision.get("categoria")),
                    ubicacion=_str(decision.get("ubicacion")),
                )
                plan.campos = []
                plan.referencia = "ninguna"
                plan.crudo = dict(decision)
                plan.fuente_plan = "llm_intencion_operativa_verificada"
                plan = _anclar_producto_literal_operacion(plan, texto, herramientas)
            elif (
                resultado_consulta_prevalidado is not None
                and not resultado_consulta_prevalidado.error
                and resultado_consulta_prevalidado.total == 0
                and not (isinstance(decision, dict) and clase == "consulta")
            ):
                # Si la segunda clasificación confirma que la persona pidió una
                # consulta, un resultado vacío es información válida: el registro
                # descrito no está en el catálogo. Aclara solo si la tarea sigue
                # sin confirmarse, para no ocultar un "no encontrado" correcto.
                plan.tipo = "aclaracion"
                plan.operacion = None
                plan.filtros = FiltrosConsulta()
                plan.referencia = "ninguna"
                plan.alcance_entidad = None
                plan.campos = []
                plan.pregunta = (
                    "No logré relacionar la frase con un producto del inventario ni confirmar una operación. "
                    "¿Qué producto desea consultar o qué operación desea realizar?"
                )
                plan.fuente_plan = "llm_aclaracion_sin_entidad"
            # Cuando la categoría fue inferida a partir de una descripción
            # general, una decisión de consulta deja vigente el plan original;
            # la verificación solo reemplaza ese plan si confirmó una acción.

        if plan.tipo == "operacion" and plan.accion == "traslado":
            plan = _normalizar_extremos_traslado_catalogo(plan, texto, herramientas)

        if plan.tipo == "fuera_dominio":
            # Una conversación fuera del dominio no debe conservar campos operativos
            # que el LLM haya rellenado accidentalmente.
            plan.operacion = None
            plan.filtros = FiltrosConsulta()
            plan.referencia = "ninguna"
            plan.alcance = None
            plan.tipo_movimiento = None
            plan.campos = []
            plan.medida_solicitada = None
            plan.alcance_entidad = None
            plan.accion = None
            plan.producto = None
            plan.cantidad = None
            plan.destinatario = None
            plan.valor = None
            plan.destino = None
            plan.ubicaciones_comparadas = []
            plan.orden_movimiento = None
            plan.pregunta = None

        if usar_cache and plan.tipo == "consulta" and plan.fuente_plan != "cache_plan":
            guardar_plan_cache(texto, contexto_txt, plan)

        salida.plan = plan
        salida.plan_haku = plan.a_plan_haku() if hasattr(plan, "a_plan_haku") else None

        traza = {
            "correlacion": correlacion,
            "plan": plan.resumen(),
            "plan_haku": salida.plan_haku.resumen() if salida.plan_haku else None,
        }
        if verificacion_intencion is not None:
            traza["verificacion_intencion"] = verificacion_intencion

        if plan.tipo == "consulta":
            salida.respuesta = cls._ejecutar_consulta(
                plan, herramientas, estado_operativo, sesion, traza,
                resultado_prevalidado=resultado_consulta_prevalidado,
            )
        elif plan.tipo == "operacion":
            cls._traducir_operacion(plan, herramientas, estado_operativo, salida, traza, texto=texto)
        elif plan.tipo == "capacidades":
            msg_pantalla = (
                "**Puedo ayudarle con las siguientes funciones de inventario:**\n\n"
                "- **Consultar existencias y disponibilidad:** stock actual, productos agotados o bajo mínimo.\n"
                "- **Ubicar productos:** almacén, estante o posición de cualquier referencia.\n"
                "- **Fichas y detalles:** características, empaque y datos técnicos.\n"
                "- **Préstamos y movimientos:** historial de entradas, salidas y préstamos activos.\n"
                "- **Operaciones guiadas:** registrar préstamos, devoluciones, traslados o solicitar cambio de stock mínimo."
            )
            msg_verbal = "Puedo ayudarle a consultar existencias, ubicaciones, fichas de producto, préstamos, movimientos y registrar operaciones guiadas de inventario."
            salida.respuesta = {"tipo": "conversacion", "mensaje_pantalla": msg_pantalla, "mensaje_verbal": msg_verbal}
        elif plan.tipo == "aclaracion":
            msg = plan.pregunta or "¿Podría precisar qué producto o conjunto de productos desea consultar?"
            tipo_respuesta = (
                "no_comprendido"
                if plan.fuente_plan == "fast_path_solicitud_incompleta"
                else "aclaracion"
            )
            salida.respuesta = {"tipo": tipo_respuesta, "mensaje_pantalla": msg, "mensaje_verbal": msg}
        elif plan.tipo == "fuera_dominio":
            msg = "Solo puedo asistirle con el inventario: existencias, ubicaciones, productos, préstamos y movimientos."
            salida.respuesta = {"tipo": "fuera_de_dominio", "mensaje_pantalla": msg, "mensaje_verbal": msg}
        elif plan.tipo == "social":
            txt_l = (texto or "").lower()
            if re.search(r"\b(gracias|agradezco|listo|vale|ok|adi[oó]s|chao|hasta\s+luego|eso\s+era\s+todo)\b", txt_l):
                msg = "Con gusto. Si necesita consultar algo más del inventario, estamos a su disposición."
                salida.respuesta = {"tipo": "conversacion", "mensaje_pantalla": msg, "mensaje_verbal": msg}
            elif re.search(
                r"\b(hola|buenos?\s+d[ií]as?|buenas?\s+tardes?|buenas?\s+noches?|buenas|saludos?|"
                r"qu[eé]\s+tal|c[oó]mo\s+est[aá]s?|hey|ey|aj[aá])\b",
                txt_l,
            ):
                from django.utils import timezone as _tz_sal
                if re.search(r"\bbuenos?\s+d[ií]as?\b", txt_l):
                    _pref_s2 = "Buenos días."
                elif re.search(r"\bbuenas?\s+tardes?\b", txt_l):
                    _pref_s2 = "Buenas tardes."
                elif re.search(r"\bbuenas?\s+noches?\b", txt_l):
                    _pref_s2 = "Buenas noches."
                else:
                    _h2 = _tz_sal.localtime().hour
                    _pref_s2 = "Buenos días." if _h2 < 12 else ("Buenas tardes." if _h2 < 19 else "Buenas noches.")
                msg = f"{_pref_s2} Hacu Inventario está listo para atenderle. ¿Qué consulta u operación desea realizar?"
                salida.respuesta = {"tipo": "conversacion", "mensaje_pantalla": msg, "mensaje_verbal": msg}
            else:
                msg = "No logré comprender bien su solicitud. ¿Podría repetirla o escribirla, por favor?"
                salida.respuesta = {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        traza["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        salida.traza = traza
        METRICAS_AGENTE["ultima"] = traza
        logger.info("[HakuAgente] %s", json.dumps(traza, ensure_ascii=False, default=str))
        if salida.respuesta is not None:
            salida.respuesta.setdefault("datos", {})
            salida.respuesta["datos"]["agente"] = {
                "fuente": "cerebro_central",
                "correlacion": correlacion,
                "plan": plan.resumen(),
                "plan_haku": salida.plan_haku.resumen() if salida.plan_haku else None,
                "herramientas": traza.get("herramientas", []),
                "total_resultados": traza.get("total_resultados"),
            }
        return salida

    # ── Resolución de referencias contextuales ─────────────────────────────
    @staticmethod
    def _aplicar_referencia(plan: PlanAgente, estado: Any) -> Optional[str]:
        """Devuelve un mensaje de aclaración si la referencia no puede resolverse."""
        f = plan.filtros
        if plan.referencia == "foco":
            codigo = getattr(estado, "producto_codigo", None) if estado else None
            if not codigo:
                coleccion = (getattr(estado, "coleccion_productos", None) or []) if estado else []
                if len(coleccion) == 1 and coleccion[0].get("codigo"):
                    codigo = coleccion[0]["codigo"]
                    if estado:
                        estado.asignar_producto(coleccion[0].get("id"), codigo, coleccion[0].get("nombre"))
                else:
                    return "¿A qué producto se refiere?"
            f.codigos = [codigo]
            f.texto = None
            if plan.campos and f.condicion in ("minimo_cero", "con_stock"):
                f.condicion = None
            if plan.operacion in (None, "listar"):
                plan.operacion = "detalle"
        elif plan.referencia == "resultados_previos":
            coleccion = (getattr(estado, "coleccion_productos", None) or []) if estado else []
            if not coleccion and getattr(estado, "producto_codigo", None):
                coleccion = [{"codigo": estado.producto_codigo, "nombre": estado.producto_nombre, "id": estado.producto_id}]
            if not coleccion:
                return "No tengo un listado previo al que referirme. ¿Qué productos desea consultar?"
            if not f.codigos:
                f.codigos = [c["codigo"] for c in coleccion if c.get("codigo")]
                f.texto = None
        return None

    @classmethod
    def _ejecutar_consulta(
        cls, plan, herramientas, estado, sesion, traza, resultado_prevalidado=None,
    ) -> Dict[str, Any]:
        # Un conteo conversacional debe usar el total original del resultado,
        # no la cantidad de filas de vista previa guardadas en la sesión.
        # La intención de contar y la referencia las decide el planificador.
        if plan.operacion == "contar" and plan.referencia == "resultados_previos":
            coleccion = (getattr(estado, "coleccion_productos", None) or []) if estado else []
            total = int(getattr(estado, "coleccion_total_original", 0) or len(coleccion))
            traza["total_resultados"] = total
            mensaje = f"La lista anterior tenía **{total} productos**."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": mensaje,
                "mensaje_verbal": f"La lista anterior tenía {total} productos.",
                "datos": {"total": total, "operacion_consulta": "contar"},
            }

        aclaracion = cls._aplicar_referencia(plan, estado)
        if aclaracion:
            traza["aclaracion"] = "referencia_no_resuelta"
            return {"tipo": "aclaracion", "mensaje_pantalla": aclaracion, "mensaje_verbal": aclaracion}

        es_consulta_eventos_global = bool(
            plan.alcance == "evento"
            and "movimientos" in (plan.campos or [])
            and not plan.filtros.codigos
            and not plan.filtros.texto
            and not plan.filtros.ubicacion
            and not plan.filtros.condicion
            and plan.filtros.es_prestable is None
            and plan.referencia == "ninguna"
            and (
                plan.fecha_desde
                or plan.fecha_hasta
                or plan.alcance_entidad == "global"
            )
        )
        if es_consulta_eventos_global:
            t_mov = time.perf_counter()
            movimientos_globales = herramientas.buscar_movimientos(
                almacen=plan.filtros.almacen,
                categoria=plan.filtros.categoria,
                tipo=plan.tipo_movimiento,
                destinatario=plan.destinatario,
                orden=plan.orden_movimiento,
                fecha_desde=plan.fecha_desde,
                fecha_hasta=plan.fecha_hasta,
            )
            traza["herramientas"] = [{
                "herramienta": "buscar_movimientos",
                "ms": round((time.perf_counter() - t_mov) * 1000.0, 1),
                "total": getattr(movimientos_globales, "total_eventos", len(movimientos_globales)),
                "error": getattr(movimientos_globales, "error", None),
            }]
            traza["total_resultados"] = getattr(movimientos_globales, "total_eventos", len(movimientos_globales))
            return redactar_movimientos_globales(plan, movimientos_globales)

        if plan.ubicaciones_comparadas:
            from .planificador import _almacenes_propietarios_de_ubicacion

            resultados_ubicacion = []
            for nombre_ubicacion in plan.ubicaciones_comparadas:
                propietarios = _almacenes_propietarios_de_ubicacion(nombre_ubicacion)
                if len(propietarios) != 1:
                    mensaje = f"¿En qué almacén se encuentra la ubicación {nombre_ubicacion}?"
                    return {"tipo": "aclaracion", "mensaje_pantalla": mensaje, "mensaje_verbal": mensaje}
                almacen = propietarios[0]
                if plan.filtros.almacen and normalizar(plan.filtros.almacen) != normalizar(almacen):
                    mensaje = (
                        f"La ubicación {nombre_ubicacion} no corresponde al almacén indicado. "
                        "¿Podría confirmar las ubicaciones que desea comparar?"
                    )
                    return {"tipo": "aclaracion", "mensaje_pantalla": mensaje, "mensaje_verbal": mensaje}

                filtros_ubicacion = FiltrosConsulta(
                    texto=plan.filtros.texto,
                    codigos=list(plan.filtros.codigos or []),
                    categoria=plan.filtros.categoria,
                    almacen=almacen,
                    ubicacion=nombre_ubicacion,
                    condicion=plan.filtros.condicion,
                    umbral=plan.filtros.umbral,
                    categoria_explicita=plan.filtros.categoria_explicita,
                    categoria_generica=plan.filtros.categoria_generica,
                )
                t_ubicacion = time.perf_counter()
                resultado_ubicacion = herramientas.buscar_productos(filtros_ubicacion)
                traza.setdefault("herramientas", []).append({
                    "herramienta": "buscar_productos_por_ubicacion",
                    "ubicacion": nombre_ubicacion,
                    "ms": round((time.perf_counter() - t_ubicacion) * 1000.0, 1),
                    "total": resultado_ubicacion.total,
                    "error": resultado_ubicacion.error,
                })
                if resultado_ubicacion.error:
                    return redactar(plan, resultado_ubicacion, {}, {})
                resultados_ubicacion.append({
                    "ubicacion": nombre_ubicacion,
                    "total_productos": resultado_ubicacion.total,
                })

            diferencia = abs(
                resultados_ubicacion[0]["total_productos"]
                - resultados_ubicacion[1]["total_productos"]
            )
            pantalla = "; ".join(
                f"En **{item['ubicacion']}** hay **{item['total_productos']}** "
                f"{'producto' if item['total_productos'] == 1 else 'productos'} asociados"
                for item in resultados_ubicacion
            ) + f". La diferencia es de **{diferencia}** productos."
            verbal = "; ".join(
                f"En {item['ubicacion']} hay {item['total_productos']} "
                f"{'producto' if item['total_productos'] == 1 else 'productos'} asociados"
                for item in resultados_ubicacion
            ) + f". La diferencia es de {diferencia} productos."
            solicita_stock_por_ubicacion = "stock" in (plan.campos or [])
            if solicita_stock_por_ubicacion:
                pantalla += (
                    " El inventario registra existencias por almacén, no por ubicación física; "
                    "por eso no puedo atribuir unidades a cada estante."
                )
                verbal += (
                    " El stock se registra por almacén, no por ubicación física, "
                    "así que no puedo atribuir unidades a cada estante."
                )
            traza["total_resultados"] = sum(item["total_productos"] for item in resultados_ubicacion)
            datos_comparacion = {
                "comparacion_ubicaciones": resultados_ubicacion,
                "diferencia_productos": diferencia,
                "metrica": "productos_asociados",
            }
            if solicita_stock_por_ubicacion:
                datos_comparacion.update({
                    "stock_por_ubicacion": False,
                    "granularidad_existencias": "almacen",
                })
            return {
                "tipo": "consulta",
                "mensaje_pantalla": pantalla,
                "mensaje_verbal": verbal,
                "datos": datos_comparacion,
            }

        # El esquema registra las existencias por almacén, no por ubicación
        # física. No se deben sumar existencias del almacén y presentarlas
        # como si estuvieran todas en el estante consultado.
        if (
            plan.operacion == "sumar_unidades"
            and plan.filtros.ubicacion
            and "stock" in (plan.campos or [])
        ):
            ubicacion = plan.filtros.ubicacion
            pantalla = (
                f"No es posible determinar cuántas unidades hay en **{ubicacion}**: "
                "las existencias se registran por almacén, no por ubicación física. "
                "Puedo mostrarle los productos asociados a esa ubicación y el stock registrado por almacén."
            )
            verbal = (
                f"No es posible determinar las unidades de {ubicacion}; "
                "el stock se registra por almacén, no por ubicación."
            )
            traza["limitacion"] = "stock_no_registrado_por_ubicacion"
            traza["total_resultados"] = None
            return {
                "tipo": "consulta",
                "mensaje_pantalla": pantalla,
                "mensaje_verbal": verbal,
                "datos": {
                    "ubicacion": ubicacion,
                    "stock_por_ubicacion": False,
                    "granularidad_existencias": "almacen",
                },
            }

        if plan.filtros.vacio() and plan.operacion in ("detalle", "comparar", "explicar_estado"):
            msg = "¿De qué producto desea esa información?"
            return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}

        # Conteo o listado de categorías registradas cuando la consulta es general de categorías
        if plan.operacion in ("listar", "contar") and plan.campos == ["categoria"] and plan.filtros.vacio():
            if not herramientas.tiene_permiso("catalogo.ver", "infraestructura.ver"):
                msg_p = "No cuentas con el permiso requerido (`catalogo.ver`) para consultar las categorías del inventario."
                msg_v = "No cuenta con permisos suficientes para consultar las categorías del inventario."
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": msg_p,
                    "mensaje_verbal": msg_v,
                    "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
                }
            cats = sorted(herramientas.nombres_categorias())
            if cats:
                traza["total_resultados"] = len(cats)
                if plan.operacion == "contar":
                    pantalla = f"En el inventario manejamos **{len(cats)} categorías**:\n\n" + "\n".join(f"- **{c}**" for c in cats)
                    verbal = f"En el inventario manejamos {len(cats)} categorías: " + ", ".join(cats) + "."
                else:
                    pantalla = "Las categorías registradas en el inventario son:\n\n" + "\n".join(f"- **{c}**" for c in cats)
                    verbal = "Las categorías registradas son: " + ", ".join(cats) + "."
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": pantalla,
                    "mensaje_verbal": verbal,
                    "datos": {"categorias": cats, "total": len(cats)},
                }

        # Conteo o listado de almacenes accesibles cuando la consulta es general de almacenes
        if plan.operacion in ("listar", "contar") and plan.campos == ["almacen"] and plan.filtros.vacio():
            if not herramientas.tiene_permiso("catalogo.ver", "infraestructura.ver"):
                msg_p = "No cuentas con el permiso requerido (`catalogo.ver`) para consultar los almacenes del inventario."
                msg_v = "No cuenta con permisos suficientes para consultar los almacenes del inventario."
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": msg_p,
                    "mensaje_verbal": msg_v,
                    "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
                }
            alms = sorted(herramientas.nombres_almacenes())
            traza["total_resultados"] = len(alms)
            if alms:
                if plan.operacion == "contar":
                    pantalla = f"Hay **{len(alms)} {'almacén disponible' if len(alms) == 1 else 'almacenes disponibles'}**:\n\n" + "\n".join(f"- **{a}**" for a in alms)
                    verbal = f"Hay {len(alms)} {'almacén disponible' if len(alms) == 1 else 'almacenes disponibles'}: " + ", ".join(alms) + "."
                else:
                    pantalla = "Los almacenes disponibles son:\n\n" + "\n".join(f"- **{a}**" for a in alms)
                    verbal = "Los almacenes disponibles son: " + ", ".join(alms) + "."
            else:
                pantalla = "No hay almacenes disponibles para su cuenta."
                verbal = pantalla
            return {
                "tipo": "consulta",
                "mensaje_pantalla": pantalla,
                "mensaje_verbal": verbal,
                "datos": {"almacenes": alms, "total": len(alms)},
            }

        # Conteo o listado de ubicaciones accesibles cuando la consulta es general de ubicaciones
        if (
            plan.operacion in ("listar", "contar")
            and plan.campos == ["ubicacion"]
            and (
                plan.filtros.vacio()
                or (
                    bool(plan.filtros.almacen)
                    and not plan.filtros.codigos
                    and not plan.filtros.texto
                    and not plan.filtros.categoria
                    and not plan.filtros.ubicacion
                    and not plan.filtros.condicion
                    and plan.filtros.es_prestable is None
                )
            )
        ):
            if not herramientas.tiene_permiso("catalogo.ver", "infraestructura.ver"):
                msg_p = "No cuentas con el permiso requerido (`catalogo.ver`) para consultar las ubicaciones del inventario."
                msg_v = "No cuenta con permisos suficientes para consultar las ubicaciones del inventario."
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": msg_p,
                    "mensaje_verbal": msg_v,
                    "datos": {"error_permiso": True, "permiso_requerido": "catalogo.ver"},
                }
            from inventario.models import Ubicacion
            ubis_qs = (
                Ubicacion.objects.filter(
                    almacen_id__in=herramientas._ids_almacenes,
                    activo=True,
                    eliminado=False,
                )
                .select_related("almacen")
                .order_by("almacen__nombre", "nombre")
            )
            almacen_filtro_ubi = plan.filtros.almacen
            if almacen_filtro_ubi:
                ubis_qs = ubis_qs.filter(almacen__nombre__iexact=almacen_filtro_ubi)
            ubis = [(u.nombre, u.almacen.nombre if u.almacen_id else "") for u in ubis_qs]
            traza["total_resultados"] = len(ubis)
            sufijo_alm_p = f" en **{almacen_filtro_ubi}**" if almacen_filtro_ubi else " en los almacenes disponibles"
            sufijo_alm_v = f" en {almacen_filtro_ubi}" if almacen_filtro_ubi else ""
            if ubis:
                lineas_p = [f"- **{nom}** ({alm})" if alm else f"- **{nom}**" for nom, alm in ubis]
                lineas_v = [f"{nom} en {alm}" if alm else nom for nom, alm in ubis]
                if plan.operacion == "contar":
                    pantalla = f"Hay **{len(ubis)} {'ubicación registrada' if len(ubis) == 1 else 'ubicaciones registradas'}**{sufijo_alm_p}:\n\n" + "\n".join(lineas_p)
                    verbal = f"Hay {len(ubis)} {'ubicación registrada' if len(ubis) == 1 else 'ubicaciones registradas'}{sufijo_alm_v}: " + ", ".join(lineas_v[:8]) + "."
                else:
                    pantalla = f"Las ubicaciones registradas{sufijo_alm_p} son:\n\n" + "\n".join(lineas_p)
                    verbal = f"Las ubicaciones registradas{sufijo_alm_v} son: " + ", ".join(lineas_v[:8]) + "."
            else:
                pantalla = f"No hay ubicaciones registradas{sufijo_alm_p}."
                verbal = f"No hay ubicaciones registradas{sufijo_alm_v or ' en los almacenes disponibles'}."
            return {
                "tipo": "consulta",
                "mensaje_pantalla": pantalla,
                "mensaje_verbal": verbal,
                "datos": {"ubicaciones": [nom for nom, _ in ubis], "total": len(ubis)},
            }

        if getattr(plan.filtros, "categoria_generica", False) and not plan.filtros.categoria:
            cats = herramientas.nombres_categorias()
            if cats:
                plan.filtros.categoria = cats[0]
                plan.filtros.categoria_explicita = True

        t = time.perf_counter()
        resultado = (
            resultado_prevalidado
            if resultado_prevalidado is not None
            else herramientas.buscar_productos(plan.filtros)
        )
        herramientas_usadas = [{"herramienta": "buscar_productos", "ms": round((time.perf_counter() - t) * 1000.0, 1), "total": resultado.total, "error": resultado.error}]
        if plan.fuente_plan == "fast_path_prioridad_reposicion" and not resultado.error:
            _ordenar_por_urgencia_de_reposicion(resultado.items)

        requiere_producto_unico = (
            plan.alcance_entidad in {"producto", "indefinido"}
            or (plan.alcance_entidad is None and plan.operacion == "detalle")
        )
        # Cuando la consulta trata sobre préstamos activos ("¿Quién tiene el taladro?",
        # "¿A quién le prestaron el multímetro?"), acotar coincidencias múltiples únicamente
        # a los productos que realmente tienen préstamos activos vigentes.
        if (
            not resultado.error
            and resultado.total > 1
            and requiere_producto_unico
            and plan.referencia != "resultados_previos"
            and plan.alcance != "evento"
            and plan.tipo_movimiento is None
            and "prestamos" in (plan.campos or [])
            and (
                set(plan.campos or []) == {"prestamos"}
                or plan.medida_solicitada == "pendiente_devolucion"
                or (
                    set(plan.campos or []) <= {"prestamos", "stock"}
                    and _es_pregunta_quien_tiene_prestamo(texto or "")
                )
            )
        ):
            con_prestamos = [
                item for item in resultado.items
                if int(item.get("en_prestamo", 0) or 0) > 0
            ]
            if len(con_prestamos) == 1:
                resultado.items = con_prestamos
                resultado.total = 1
            elif len(con_prestamos) > 1:
                resultado.items = con_prestamos
                resultado.total = len(con_prestamos)
                traza["aclaracion_solo_prestamos"] = True
            else:
                alguno_prestable = any(bool(item.get("es_prestable", False)) for item in resultado.items)
                if not alguno_prestable:
                    msg_sin_p = "Los productos que coinciden con esa consulta son artículos consumibles y no admiten préstamos."
                else:
                    msg_sin_p = "Ninguno de los productos que coinciden tiene préstamos activos en este momento."
                traza["total_resultados"] = resultado.total
                return {
                    "tipo": "consulta",
                    "mensaje_pantalla": msg_sin_p,
                    "mensaje_verbal": msg_sin_p,
                    "datos": {"prestamos": [], "total": 0},
                }

        if (
            not resultado.error
            and resultado.total > 1
            and requiere_producto_unico
            and plan.referencia != "resultados_previos"
        ):
            # No agregues existencias ni atributos de objetos distintos cuando
            # el plan expresa que la consulta busca un solo producto o no puede
            # determinar si la mención es individual o colectiva.
            opciones = resultado.items[:3]
            opciones_txt = "; ".join(
                f"{item['nombre']} ({item['codigo']})" for item in opciones
            )
            if traza.get("aclaracion_solo_prestamos"):
                msg = (
                    f"Encontré varios productos con préstamos activos que coinciden: {opciones_txt}. "
                    "¿Cuál desea consultar? Puede indicar el código o el nombre del producto."
                )
            else:
                msg = (
                    f"Encontré varios productos que podrían coincidir: {opciones_txt}. "
                    "¿Cuál desea consultar? Puede indicar el código o el nombre del producto."
                )
            traza["aclaracion"] = "alcance_entidad_ambiguo"
            traza["total_resultados"] = resultado.total
            if estado is not None:
                # Guarda los candidatos y la consulta incompleta como estructura,
                # para que una selección posterior continúe la misma operación.
                estado.asignar_candidatos(
                    opciones,
                    intencion_original="CONSULTA_AGENTE",
                    slot_pendiente="producto",
                    datos_extra={
                        "plan_consulta": {
                            "operacion": plan.operacion,
                            "campos": list(plan.campos or []),
                            "alcance": plan.alcance,
                            "tipo_movimiento": plan.tipo_movimiento,
                            "medida_solicitada": plan.medida_solicitada,
                            "categoria": plan.filtros.categoria,
                            "almacen": plan.filtros.almacen,
                            "ubicacion": plan.filtros.ubicacion,
                            "condicion": plan.filtros.condicion,
                            "umbral": plan.filtros.umbral,
                        },
                    },
                )
                estado.producto_id = None
                estado.producto_codigo = None
                estado.producto_nombre = None
                estado.coleccion_productos = []
                estado.coleccion_total_original = 0
                try:
                    from ..contexto import HakuOperationalContext
                    HakuOperationalContext.guardar(estado)
                except Exception:
                    logger.debug("[HakuAgente] No se pudo persistir la aclaración de producto", exc_info=True)
            return {
                "tipo": "aclaracion",
                "mensaje_pantalla": msg,
                "mensaje_verbal": msg,
                "datos": {
                    "candidatos": opciones,
                    "total_candidatos": resultado.total,
                    "requiere_aclaracion": True,
                },
            }

        movimientos: Dict[int, List[Dict[str, Any]]] = {}
        quiere_movs = (
            "ficha" in plan.campos
            or "movimientos" in plan.campos
            or plan.alcance in ("evento", "ambos")
            or bool(plan.tipo_movimiento)
            or (plan.operacion == "detalle" and not plan.campos)
        )
        if quiere_movs and 0 < resultado.total <= 3:
            t = time.perf_counter()
            for it in resultado.items:
                movimientos[it["id"]] = herramientas.movimientos_producto(
                    it["id"], almacen=plan.filtros.almacen, tipo=plan.tipo_movimiento,
                    destinatario=plan.destinatario, orden=plan.orden_movimiento,
                    fecha_desde=plan.fecha_desde, fecha_hasta=plan.fecha_hasta,
                )
            herramientas_usadas.append({"herramienta": "movimientos_producto", "ms": round((time.perf_counter() - t) * 1000.0, 1)})

        prestamos: Dict[int, List[Dict[str, Any]]] = {}
        consulta_solo_saldo_prestamos = (
            plan.operacion == "sumar_unidades"
            and "prestamos" in plan.campos
            and "stock" not in plan.campos
        )
        quiere_prest = (
            "ficha" in plan.campos
            or ("prestamos" in plan.campos and not consulta_solo_saldo_prestamos)
            or plan.filtros.condicion == "con_prestamos"
            or plan.operacion == "explicar_estado"
            or (plan.operacion == "detalle" and not plan.campos)
        )
        if quiere_prest and 0 < resultado.total <= 10:
            t = time.perf_counter()
            for it in resultado.items:
                prestamos[it["id"]] = herramientas.prestamos_activos(it["id"], almacen=plan.filtros.almacen)
            herramientas_usadas.append({"herramienta": "prestamos_activos", "ms": round((time.perf_counter() - t) * 1000.0, 1)})

        saldo_prestamos_pendientes = None
        if consulta_solo_saldo_prestamos and resultado.items:
            t = time.perf_counter()
            saldo_prestamos_pendientes, error_saldo = herramientas.sumar_unidades_pendientes_devolucion(
                [item["id"] for item in resultado.items],
                almacen=plan.filtros.almacen,
            )
            herramientas_usadas.append({
                "herramienta": "sumar_unidades_pendientes_devolucion",
                "ms": round((time.perf_counter() - t) * 1000.0, 1),
                "error": error_saldo,
            })
            if error_saldo == "permiso_denegado":
                traza["herramientas"] = herramientas_usadas
                traza["total_resultados"] = resultado.total
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": "No cuentas con el permiso requerido (`prestamos.ver`) para consultar los préstamos activos.",
                    "mensaje_verbal": "No cuenta con permisos suficientes para consultar los préstamos activos de inventario.",
                    "datos": {"error_permiso": True, "permiso_requerido": "prestamos.ver"},
                }
            if error_saldo:
                traza["herramientas"] = herramientas_usadas
                traza["total_resultados"] = resultado.total
                return {
                    "tipo": "error",
                    "mensaje_pantalla": "No pude validar el alcance del saldo de préstamos para esa consulta.",
                    "mensaje_verbal": "No pude validar el alcance de los préstamos para esa consulta.",
                    "datos": {"error": error_saldo},
                }

        traza["herramientas"] = herramientas_usadas
        traza["total_resultados"] = resultado.total
        respuesta = redactar(
            plan,
            resultado,
            movimientos,
            prestamos,
            saldo_prestamos_pendientes=saldo_prestamos_pendientes,
        )
        respuesta.setdefault("datos", {})
        respuesta["datos"].update({
            "productos": resultado.items[:25],
            "total": resultado.total,
            "mostrados": min(resultado.total, 25),
            "resultados_completos": resultado.total <= 25,
            "operacion_consulta": plan.operacion,
        })
        if plan.operacion == "sumar_unidades":
            if "prestamos" in plan.campos and "stock" not in plan.campos:
                respuesta["datos"]["unidades_en_prestamo"] = (
                    saldo_prestamos_pendientes
                    if saldo_prestamos_pendientes is not None
                    else sum(item["en_prestamo"] for item in resultado.items)
                )
            else:
                respuesta["datos"]["unidades_disponibles"] = sum(item["disponible"] for item in resultado.items)
                respuesta["datos"]["unidades_en_prestamo"] = sum(item["en_prestamo"] for item in resultado.items)
        cls._actualizar_contexto(plan, resultado, estado, sesion)
        return respuesta

    @staticmethod
    def _actualizar_contexto(plan, resultado, estado, sesion) -> None:
        if estado is None or resultado.error:
            return
        from ..contexto import HakuOperationalContext

        estado.candidatos = []
        if plan.operacion in ("mayor_stock", "menor_stock") and resultado.items:
            _, ganadores = seleccionar_ganadores_extremos(resultado.items, plan.operacion)
            if len(ganadores) == 1:
                elegido = ganadores[0]
                estado.asignar_producto(elegido["id"], elegido["codigo"], elegido["nombre"])
                estado.coleccion_productos = [elegido]
                estado.coleccion_total_original = 1
                if sesion is not None:
                    sesion["haku_ultimo_producto_id"] = elegido["id"]
            else:
                # Keep every tied winner in context; silently choosing one
                # makes the next pronoun resolve to an arbitrary product.
                estado.asignar_coleccion(ganadores, total_original=len(ganadores))
                estado.producto_id = None
                estado.producto_codigo = None
                estado.producto_nombre = None
                if sesion is not None:
                    sesion["haku_ultimo_producto_id"] = None
        elif resultado.total == 1 and (
            plan.operacion not in ("contar", "sumar_unidades")
            or bool(plan.filtros.codigos)
            or plan.referencia == "foco"
        ):
            it = resultado.items[0]
            estado.asignar_producto(it["id"], it["codigo"], it["nombre"])
            # Una selección de la lista previa establece un foco individual,
            # pero no debe destruir la lista que permite seguir preguntando
            # por su cantidad, posición u otros elementos.
            if plan.referencia != "resultados_previos":
                estado.coleccion_productos = [it]
                estado.coleccion_total_original = 1
            if sesion is not None:
                sesion["haku_ultimo_producto_id"] = it["id"]
        elif resultado.total >= 1 and (
            plan.operacion in ("contar", "sumar_unidades")
            or (resultado.total > 1 and plan.referencia != "resultados_previos")
        ):
            # Un conjunto o conteo reemplaza el foco individual para evitar herencias accidentales
            estado.asignar_coleccion(resultado.items, total_original=resultado.total)
            if 2 <= resultado.total <= 5 and plan.operacion == "detalle":
                estado.asignar_candidatos(resultado.items[:5], intencion_original="CONSULTA_DETALLE")
            estado.producto_id = None
            estado.producto_codigo = None
            estado.producto_nombre = None
            if sesion is not None:
                sesion["haku_ultimo_producto_id"] = None
        elif resultado.total == 0 and (plan.referencia == "ninguna" or (not plan.filtros.almacen and not plan.filtros.ubicacion)):
            # Si el usuario cambió de tema, buscó otro producto inexistente o perdió visibilidad sobre el foco, purgar contexto
            estado.producto_id = None
            estado.producto_codigo = None
            estado.producto_nombre = None
            estado.coleccion_productos = []
            estado.coleccion_total_original = 0
            if getattr(resultado, "relacionados", None):
                estado.asignar_candidatos(resultado.relacionados[:5], intencion_original="CONSULTA_RELACIONADA")
            if sesion is not None:
                sesion["haku_ultimo_producto_id"] = None
        estado.ultima_intencion = f"agente:{plan.operacion}"
        try:
            HakuOperationalContext.guardar(estado)
        except Exception:
            logger.debug("[HakuAgente] No se pudo persistir el contexto", exc_info=True)

    # ── Traducción de modificaciones al flujo seguro existente ─────────────
    @classmethod
    def _traducir_operacion(cls, plan, herramientas, estado, salida: SalidaAgente, traza, texto: str = "") -> None:
        from ..lenguaje.interprete_semantico import ExtraccionSemantica
        from .planificador import tiene_referencia_contextual_operacion

        # Un código validado y expresado por el usuario identifica el producto
        # con más precisión que un nombre/filtro que el modelo haya mezclado
        # con el marco de la operación (p. ej. «stock mínimo de PAP-030 a 4»).
        if plan.accion in ("crear_producto", "crear"):
            # Una alta requiere el valor del campo producto/nombre. Un filtro de
            # texto puede ser solo el marco de la petición, así que no se recicla
            # como nombre de un nuevo registro ni se hereda del producto en foco.
            producto_txt = (
                _str(plan.producto)
                or _str(plan.crudo.get("nombre"))
                or _str(plan.crudo.get("producto"))
            )
            if producto_txt and estado is not None and texto:
                foco_cod = _str(getattr(estado, "producto_codigo", None))
                foco_nom = _str(getattr(estado, "producto_nombre", None))
                prod_norm_op = normalizar(producto_txt)
                cods_en_texto = {c.casefold() for c in extraer_codigos_de_texto(texto)}
                es_fuga_foco = False
                if foco_cod and normalizar(foco_cod) in prod_norm_op and foco_cod.casefold() not in cods_en_texto:
                    es_fuga_foco = True
                elif foco_nom and normalizar(foco_nom) and (
                    prod_norm_op == normalizar(foco_nom) or normalizar(foco_nom) in prod_norm_op
                ):
                    toks_foco = set(tokens_significativos(foco_nom))
                    toks_txt = set(tokens_significativos(texto))
                    if toks_foco and not (toks_foco & toks_txt):
                        es_fuga_foco = True
                if es_fuga_foco:
                    producto_txt = None
        else:
            producto_txt = (plan.filtros.codigos[0] if plan.filtros.codigos else None) or plan.producto or plan.filtros.texto
        tiene_ref_ctx = tiene_referencia_contextual_operacion(texto) or getattr(plan, "referencia", "ninguna") in ("foco", "resultados_previos")
        if (
            producto_txt
            and tiene_ref_ctx
            and estado is not None
            and getattr(estado, "producto_codigo", None)
            and plan.accion not in ("crear_producto", "crear")
            and not (plan.filtros.codigos and getattr(plan, "referencia", "ninguna") == "ninguna")
            and normalizar(producto_txt) != normalizar(estado.producto_codigo)
        ):
            # Cuando el usuario expresa una referencia anafórica explícita («uno de esos»),
            # solo acepta otro producto si el mensaje contiene al menos un token distintivo
            # (no numeral ni pronombre) de ese producto; así un cuantificador como «uno»
            # no puede reemplazar el foco por un producto cuyo nombre contiene la palabra «Uno».
            _numerales_y_vacios = {
                "uno", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
                "de", "del", "el", "la", "los", "las", "un", "en", "a", "al", "por", "para",
                "ese", "esa", "esos", "esas", "este", "esta", "estos", "estas", "mismo", "misma",
            }
            tokens_prod = {
                t for t in tokens_significativos(producto_txt)
                if t not in _numerales_y_vacios and not t.isdigit()
            }
            tokens_msg = {
                t for t in tokens_significativos(texto or "")
                if t not in _numerales_y_vacios and not t.isdigit()
            }
            from .herramientas import normalizar_codigos_hablados_en_texto
            cod_mencionados = {
                c.casefold()
                for c in extraer_codigos_de_texto(normalizar_codigos_hablados_en_texto(texto or ""))
            }
            if not (tokens_prod & tokens_msg) and producto_txt.strip().casefold() not in cod_mencionados:
                producto_txt = None
        if not producto_txt and estado is not None and tiene_ref_ctx and plan.accion not in ("crear_producto", "crear"):
            # Solo heredar producto en foco si es único y sin ambigüedad
            if getattr(estado, "producto_codigo", None) and not getattr(estado, "candidatos", None) and len(getattr(estado, "coleccion_productos", []) or []) <= 1:
                producto_txt = estado.producto_codigo
            elif getattr(estado, "coleccion_productos", None) and len(estado.coleccion_productos) == 1:
                producto_txt = estado.coleccion_productos[0].get("codigo") or estado.coleccion_productos[0].get("nombre")
        traza["operacion"] = plan.accion

        if plan.accion in ("cambiar_stock_minimo", "editar_producto"):
            if not herramientas.tiene_permiso("catalogo.editar"):
                from ..lenguaje.normalizador import IntencionResultado, IntencionTipo

                salida.intencion = IntencionResultado(
                    tipo=IntencionTipo.EDITAR_PRODUCTO,
                    requiere_confirmacion=True,
                )
                return
            if not producto_txt:
                msg = "¿A qué producto desea cambiarle " + ("el stock mínimo?" if plan.accion == "cambiar_stock_minimo" else "los datos?")
                salida.respuesta = {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}
                return
            from .herramientas import FiltrosConsulta

            es_codigo = bool(plan.filtros.codigos) or producto_txt == getattr(estado, "producto_codigo", None)
            res = herramientas.buscar_productos(
                FiltrosConsulta(codigos=[producto_txt]) if es_codigo else FiltrosConsulta(texto=producto_txt)
            )
            if res.total != 1:
                if res.total == 0:
                    msg = f"No encontré el producto «{producto_txt}» en los almacenes autorizados para su cuenta. ¿Qué producto desea modificar?"
                else:
                    opciones = ", ".join(f"{i['nombre']} ({i['codigo']})" for i in res.items[:5])
                    msg = f"Hay varios productos que coinciden: {opciones}. ¿Cuál desea modificar?"
                    if estado is not None:
                        from ..contexto import HakuOperationalContext
                        estado.asignar_candidatos(res.items[:3], intencion_original="EDITAR_PRODUCTO")
                        HakuOperationalContext.guardar(estado)
                salida.respuesta = {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}
                return
            from inventario.models import Producto
            from ..lenguaje.normalizador import IntencionResultado, IntencionTipo

            prod = Producto.objects.filter(id=res.items[0]["id"]).first()
            datos = {"origen": "agente"}
            nuevo_minimo = plan.valor if plan.valor is not None else plan.cantidad
            valores_prop = {}
            if plan.accion == "cambiar_stock_minimo":
                if nuevo_minimo is not None:
                    valores_prop["stock_minimo"] = nuevo_minimo
                else:
                    msg = f"¿A qué valor desea cambiar el stock mínimo de {prod.nombre} ({prod.codigo_interno})?"
                    salida.respuesta = {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}
                    if estado is not None:
                        from ..contexto import HakuOperationalContext
                        estado.asignar_producto(prod.id, prod.codigo_interno, prod.nombre)
                        HakuOperationalContext.guardar(estado)
                    return
            elif plan.accion == "editar_producto":
                if getattr(plan, "valores_propuestos", None):
                    valores_prop.update(plan.valores_propuestos)
                if nuevo_minimo is not None and "stock_minimo" not in valores_prop:
                    valores_prop["stock_minimo"] = nuevo_minimo

            datos["valores_propuestos"] = valores_prop
            salida.intencion = IntencionResultado(
                tipo=IntencionTipo.EDITAR_PRODUCTO,
                producto=prod,
                codigo_producto=prod.codigo_interno if prod else None,
                datos_extra=datos,
                requiere_confirmacion=True,
            )
            return

        if plan.accion in ("inhabilitar", "habilitar"):
            salida.extraccion = ExtraccionSemantica(
                intencion=f"{plan.accion}_producto", producto=producto_txt, confianza=0.9, fuente="llm",
                datos_extra={
                    "mecanismo": "agente",
                    "llm_ms": plan.llm_ms,
                    "llm_cola_ms": plan.llm_cola_ms,
                },
            )
            return

        if plan.accion in ("crear_producto", "crear"):
            if not herramientas.tiene_permiso("catalogo.crear"):
                from ..lenguaje.normalizador import IntencionResultado, IntencionTipo
                salida.intencion = IntencionResultado(
                    tipo=IntencionTipo.CREAR_PRODUCTO,
                    requiere_confirmacion=True,
                )
                return

            de_crear = {
                "mecanismo": "agente",
                "llm_ms": plan.llm_ms,
                "llm_cola_ms": plan.llm_cola_ms,
                "categoria": plan.filtros.categoria or plan.crudo.get("categoria"),
                "almacen": plan.filtros.almacen or plan.crudo.get("almacen"),
                "ubicacion": plan.filtros.ubicacion or plan.crudo.get("ubicacion"),
                "stock_minimo": plan.valor if plan.valor is not None else plan.crudo.get("stock_minimo"),
                "stock_inicial": plan.cantidad if plan.cantidad is not None else plan.crudo.get("stock_inicial"),
                "codigo_barras": plan.crudo.get("codigo_barras", ""),
                "descripcion": plan.crudo.get("descripcion", ""),
            }
            salida.extraccion = ExtraccionSemantica(
                intencion="crear_producto",
                tipo_operacion="CREAR_PRODUCTO",
                producto=producto_txt,
                cantidad=plan.cantidad,
                almacen=plan.filtros.almacen or plan.crudo.get("almacen"),
                confianza=0.9,
                fuente="llm",
                datos_extra=de_crear,
            )
            return

        almacen = plan.filtros.almacen
        datos_extra = {
            "mecanismo": "agente",
            "llm_ms": plan.llm_ms,
            "llm_cola_ms": plan.llm_cola_ms,
        }
        accion_op = plan.accion
        if texto and re.search(r"\b(?:dotaci[oó]n|entrega\s+definitiva)\b", texto, re.IGNORECASE):
            if accion_op in ("entrada", "salida", "prestamo"):
                accion_op = "salida"
            datos_extra["motivo"] = "Dotación"
        if texto and "motivo" not in datos_extra:
            m_mot = re.search(r"\bpor\s+([a-zA-ZñÑáéíóúÁÉÍÓÚ\s'-]+?)(?:$|\.|\,)", texto, re.IGNORECASE)
            if m_mot:
                cand_mot = m_mot.group(1).strip()
                if cand_mot and not re.fullmatch(
                    r"(?:favor|si\s+acaso|ahora|el\s+momento|hoy|ayer|ma[ñn]ana|"
                    r"(?:el\s+)?almac[eé]n\s+\w+|(?:la\s+)?bodega\s+\w+)",
                    cand_mot,
                    flags=re.IGNORECASE,
                ):
                    datos_extra["motivo"] = cand_mot

        salida.extraccion = ExtraccionSemantica(
            intencion="proponer_operacion",
            tipo_operacion=accion_op,
            producto=producto_txt,
            cantidad=plan.cantidad,
            destinatario=plan.destinatario,
            almacen=almacen,
            origen=almacen if accion_op == "traslado" else None,
            destino=plan.destino,
            confianza=0.9,
            fuente="llm",
            datos_extra=datos_extra,
        )
