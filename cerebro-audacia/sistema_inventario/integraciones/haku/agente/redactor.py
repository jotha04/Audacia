"""Redactor de respuestas del agente HAKU Inventario.

Construye la respuesta EXCLUSIVAMENTE a partir de los resultados reales de las
herramientas y de los atributos solicitados en el plan. No hay plantillas por
pregunta: la forma depende de (operación, campos, número de resultados).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    MAX_PANTALLA = max(1, min(50, int(os.environ.get("HAKU_MAX_ITEMS_CHAT", "5"))))
except (TypeError, ValueError):
    MAX_PANTALLA = 5
MAX_VERBAL = 5


def _u(n: int) -> str:
    return "unidad" if n == 1 else "unidades"


def _stock_cantidad(n: int) -> str:
    """Forma concordante para presentar unidades disponibles."""
    return f"{n} {_u(n)} {'disponible' if n == 1 else 'disponibles'}"


def _p(n: int) -> str:
    return "producto" if n == 1 else "productos"


def _almacenes_txt(item: Dict[str, Any]) -> str:
    nombres = [a["nombre"] for a in item["almacenes"]]
    return ", ".join(nombres) if nombres else "sin almacén asignado"


def _ubicaciones_txt(item: Dict[str, Any]) -> str:
    if not item["ubicaciones"]:
        return "sin ubicación registrada"
    return ", ".join(
        f"{u['nombre']} ({u['almacen']})" if u.get("almacen") else u["nombre"] for u in item["ubicaciones"]
    )


def _almacenes_no_cubiertos_por_ubicacion(item: Dict[str, Any]) -> List[str]:
    almacenes_item = [a["nombre"] for a in item.get("almacenes", []) if a.get("nombre")]
    almacenes_en_ubic = {u.get("almacen") for u in item.get("ubicaciones", []) if u.get("almacen")}
    return [nombre for nombre in almacenes_item if nombre not in almacenes_en_ubic]


def _campos_efectivos_item(item: Dict[str, Any], campos_base: List[str]) -> List[str]:
    if (
        "ubicacion" in campos_base
        and "almacen" in campos_base
        and item.get("ubicaciones")
        and not _almacenes_no_cubiertos_por_ubicacion(item)
    ):
        return [c for c in campos_base if c != "almacen"]
    return list(campos_base)


def _stock_txt(item: Dict[str, Any], con_almacen: bool) -> str:
    if item["disponible"] == 0:
        base = "0 unidades disponibles (agotado)"
    else:
        base = _stock_cantidad(item["disponible"])
    if item["en_prestamo"]:
        base += f" y {item['en_prestamo']} en préstamo"
    if con_almacen and len(item["almacenes"]) > 1:
        base += " (" + "; ".join(f"{a['nombre']}: {a['disponible']}" for a in item["almacenes"]) + ")"
    return base


def _valor_campo(
    item: Dict[str, Any],
    campo: str,
    prestamos_activos: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    if campo == "almacen":
        return _almacenes_txt(item)
    if campo == "ubicacion":
        return _ubicaciones_txt(item)
    if campo == "stock":
        return _stock_txt(item, con_almacen=True)
    if campo == "stock_minimo":
        return str(item["stock_minimo"])
    if campo == "categoria":
        return item["categoria"] or "sin categoría"
    if campo == "nombre":
        return item["nombre"]
    if campo == "codigo":
        return item["codigo"]
    if campo == "codigo_barras":
        return item["codigo_barras"] if item.get("codigo_barras") else "sin código de barras registrado"
    if campo == "prestamos":
        if getattr(prestamos_activos, "permiso_denegado", False):
            return "sin permiso para consultar préstamos activos"
        if prestamos_activos:
            return ", ".join(
                f"{p['cantidad']} {_u(p['cantidad'])} a {p['destinatario']}" for p in prestamos_activos
            )
        if item["en_prestamo"]:
            return f"{item['en_prestamo']} {_u(item['en_prestamo'])}"
        return "ninguna actualmente"
    if campo == "empaque":
        return item["empaque"] or "sin empaque registrado"
    if campo == "es_prestable":
        return "sí (retornable)" if item.get("es_prestable") else "no (consumible)"
    return None


def _redactar_movimiento_tipado(
    item: Dict[str, Any],
    movimientos: List[Dict[str, Any]],
    tipo_movimiento: Optional[str] = None,
) -> Dict[str, Any]:
    """Responde desde el agregado acotado y los eventos visibles, sin asumir un único evento."""
    if getattr(movimientos, "permiso_denegado", False):
        return {
            "tipo": "error_permiso",
            "mensaje_pantalla": "No cuentas con el permiso requerido (`movimientos.ver`) para consultar el historial.",
            "mensaje_verbal": "No cuenta con permisos suficientes para consultar el historial de movimientos.",
            "datos": {"error_permiso": True, "permiso_requerido": "movimientos.ver"},
        }
    total = getattr(movimientos, "total_eventos", None)
    unidades = getattr(movimientos, "unidades_evento", None)
    periodo = getattr(movimientos, "periodo_descripcion", None)
    tipo = movimientos[0]["tipo"] if movimientos else None
    if not tipo and tipo_movimiento:
        try:
            from inventario.models import Movimiento
            tipo = dict(Movimiento.Tipo.choices).get(tipo_movimiento)
        except Exception:
            tipo = None
    tipo = str(tipo or tipo_movimiento or "")
    frase_tipo = f"movimientos de {tipo.casefold()}" if tipo else "movimientos"
    ident = _ident(item)
    if total == 0:
        periodo_txt = f" {periodo}" if periodo else ""
        pantalla = f"No hay {frase_tipo} registrados para **{ident}**{periodo_txt}."
        verbal = f"No hay {frase_tipo} registrados para {item['nombre']}{periodo_txt}."
    elif total == 1 and movimientos:
        mov = movimientos[0]
        detalle = f" ({mov['fecha']})" if mov.get("fecha") else ""
        destinatario = f" para {mov['destinatario']}" if mov.get("destinatario") else ""
        est_p, est_v = _sufijo_estado_prestamo(mov)
        periodo_txt = f" {periodo}" if periodo else ""
        etiqueta_tipo = f" de **{tipo.casefold()}**" if tipo else ""
        etiqueta_verbal = f" de {tipo.casefold()}" if tipo else ""
        pantalla = f"Se registró un movimiento{etiqueta_tipo} para **{ident}**: **{mov['cantidad']} {_u(mov['cantidad'])}**{destinatario}{detalle}{est_p}{periodo_txt}."
        verbal = f"Se registró un movimiento{etiqueta_verbal} para {item['nombre']}: {mov['cantidad']} {_u(mov['cantidad'])}{destinatario}{detalle}{est_v}{periodo_txt}."
    else:
        suma = unidades if unidades is not None else sum(m.get("cantidad", 0) for m in movimientos)
        ultimos_p = "; ".join(
            f"{m['cantidad']} {_u(m['cantidad'])} ({m['fecha']})"
            + (f" para {m['destinatario']}" if m.get("destinatario") else "")
            + _sufijo_estado_prestamo(m)[0]
            for m in movimientos[:5]
        )
        ultimos_v = "; ".join(
            f"{m['cantidad']} {_u(m['cantidad'])} ({m['fecha']})"
            + (f" para {m['destinatario']}" if m.get("destinatario") else "")
            + _sufijo_estado_prestamo(m)[1]
            for m in movimientos[:5]
        )
        pantalla = (
            f"Se registran **{total} {frase_tipo}** para **{ident}**"
            f"{f' {periodo}' if periodo else ''}, "
            f"con **{suma} {_u(suma)} acumuladas**. Eventos más recientes: {ultimos_p or 'sin detalle disponible'}."
        )
        verbal = (
            f"Se registran {total} {frase_tipo} para {item['nombre']}"
            f"{f' {periodo}' if periodo else ''}, "
            f"con {suma} {_u(suma)} acumuladas. Los más recientes son: {ultimos_v or 'sin detalle disponible'}."
        )
    return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}


def _sufijo_estado_prestamo(mov: Dict[str, Any]) -> Tuple[str, str]:
    """Devuelve (etiqueta_pantalla, etiqueta_verbal) del estado actual de devolución de un préstamo."""
    estado = str(mov.get("estado_prestamo") or "").upper()
    if not estado:
        return "", ""
    if estado == "CERRADO":
        f_dev = mov.get("fecha_devolucion")
        pant = f" · **Devuelto** ({f_dev})" if f_dev else " · **Devuelto**"
        return pant, ", ya devuelto"
    if estado == "DEVUELTO_PARCIAL":
        pend = int(mov.get("pendiente_devolucion") or 0)
        palabra_pend = "pendiente" if pend == 1 else "pendientes"
        return (
            f" · **Devuelto parcial** ({pend} {_u(pend)} {palabra_pend})",
            f", devuelto parcialmente, {pend} {_u(pend)} {palabra_pend}",
        )
    if estado == "ABIERTO":
        return " · **Pendiente de devolución**", ", pendiente de devolución"
    return "", ""


_ETIQUETA_CAMPO = {
    "almacen": "Almacén", "ubicacion": "Ubicación", "stock": "Existencias", "stock_minimo": "Stock mínimo",
    "categoria": "Categoría", "codigo": "Código", "nombre": "Nombre", "prestamos": "En préstamo", "empaque": "Empaque",
    "codigo_barras": "Código de barras", "es_prestable": "Prestable",
}


def _ident(item: Dict[str, Any]) -> str:
    return f"{item['nombre']} ({item['codigo']})"


def seleccionar_ganadores_extremos(
    items: List[Dict[str, Any]], operacion: str,
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """Devuelve todos los productos empatados en el máximo o mínimo disponible."""
    if not items or operacion not in {"mayor_stock", "menor_stock"}:
        return None, []
    valores = [int(item.get("disponible", 0) or 0) for item in items]
    valor = max(valores) if operacion == "mayor_stock" else min(valores)
    ganadores = [item for item, actual in zip(items, valores) if actual == valor]
    ganadores.sort(key=lambda item: str(item.get("codigo") or ""))
    return valor, ganadores


def _ficha(
    item: Dict[str, Any],
    movimientos: List[Dict[str, Any]],
    prestamos: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    lineas = [
        f"### {item['nombre']} (`{item['codigo']}`)",
        f"- **Categoría:** {item['categoria'] or '—'}",
        f"- **Existencias:** {_stock_txt(item, con_almacen=True)}",
        f"- **Stock mínimo:** {item['stock_minimo']}",
        f"- **Almacén:** {_almacenes_txt(item)}",
        f"- **Ubicación:** {_ubicaciones_txt(item)}",
        f"- **Prestable:** {'sí' if item['es_prestable'] else 'no'}",
    ]
    if getattr(prestamos, "permiso_denegado", False):
        lineas.append("- **Préstamos activos:** _restringido (requiere permiso `prestamos.ver`)_")
    elif prestamos:
        det_prest = ", ".join(f"{p['cantidad']} {_u(p['cantidad'])} a {p['destinatario']}" for p in prestamos)
        lineas.append(f"- **Préstamos activos:** {det_prest}")
    if item.get("descripcion"):
        lineas.append(f"- **Descripción:** {item['descripcion']}")
    if item["empaque"]:
        lineas.append(f"- **Empaque:** {item['empaque']}")
    if item["codigo_barras"]:
        lineas.append(f"- **Código de barras:** {item['codigo_barras']}")
    if not item["activo"]:
        lineas.append("- **Estado:** inactivo o archivado")
    if getattr(movimientos, "permiso_denegado", False):
        lineas.append("\n_Historial de movimientos restringido (requiere permiso `movimientos.ver`)._")
    elif movimientos:
        lineas.append("\n**Últimos movimientos:**")
        for m in movimientos:
            dest = f" → {m['destinatario']}" if m.get("destinatario") else ""
            est_p, _ = _sufijo_estado_prestamo(m)
            lineas.append(f"- {m['fecha']} · {m['tipo']} · {m['cantidad']}{dest}{est_p}")
    else:
        lineas.append("\n_Sin movimientos registrados._")
    verbal = (
        f"{item['nombre']}, código {item['codigo']}, categoría {item['categoria'] or 'sin categoría'}. "
        f"Tiene {_stock_txt(item, con_almacen=False)}, stock mínimo {item['stock_minimo']}, "
        f"en {_almacenes_txt(item)}, ubicación {_ubicaciones_txt(item)}."
    )
    if prestamos:
        det_prest = ", ".join(f"{p['cantidad']} {_u(p['cantidad'])} a {p['destinatario']}" for p in prestamos)
        verbal += f" Préstamos activos: {det_prest}."
    if getattr(movimientos, "permiso_denegado", False):
        pass
    elif movimientos:
        verbal += f" Registra {len(movimientos)} movimientos recientes; los ves en pantalla."
    else:
        verbal += " No tiene movimientos registrados."
    return "\n".join(lineas), verbal


_RE_TOKEN_MEDIDA_INTERNO = re.compile(r"\bq(\d+)d(\d+)\b", flags=re.IGNORECASE)
_RE_TOKEN_TALLA_INTERNO = re.compile(r"\btalla(xxxl|xxl|xs|xl|[sml])0\b", flags=re.IGNORECASE)
_RE_VERBO_BUSQUEDA_INICIAL = re.compile(
    r"^(?:encontrar(?:me)?|buscar(?:me)?|localizar|ubicar|hallar)\s+",
    flags=re.IGNORECASE,
)


def _humanizar_filtro_texto(texto: str) -> str:
    """Garantiza que jamás se muestren ni se pronuncien tokens internos de normalización."""
    if not texto:
        return ""

    def _fmt_fraccion(m: re.Match) -> str:
        num, den = int(m.group(1)), int(m.group(2))
        if den <= 1:
            return str(num)
        if num > den:
            ent, resto = divmod(num, den)
            return f"{ent} {resto}/{den}" if resto else str(ent)
        return f"{num}/{den}"

    limpio = _RE_TOKEN_MEDIDA_INTERNO.sub(_fmt_fraccion, str(texto).strip())
    limpio = _RE_TOKEN_TALLA_INTERNO.sub(lambda m: f"talla {m.group(1).upper()}", limpio)
    limpio = _RE_VERBO_BUSQUEDA_INICIAL.sub("", limpio).strip()
    return limpio or str(texto).strip()


def _describir_filtro(plan, singular: bool = False) -> str:
    f = plan.filtros
    partes = []
    if f.texto:
        partes.append(f"«{_humanizar_filtro_texto(f.texto)}»")
    if f.categoria:
        if getattr(f, "categoria_generica", False):
            partes.append(f"de la categoría {f.categoria} (categoría existente)")
        else:
            partes.append(f"de la categoría {f.categoria}")
    if f.ubicacion:
        partes.append(f"en la ubicación {f.ubicacion}")
    if f.almacen:
        partes.append(f"en {f.almacen}")
    cond = {
        "agotado": (
            "agotado (sin unidades disponibles)"
            if singular else "agotados (sin unidades disponibles)"
        ),
        "bajo_minimo": "por debajo de su stock mínimo",
        "requiere_reposicion": (
            "que requiere reposición (agotado o por debajo del mínimo)"
            if singular else "que requieren reposición (agotados o por debajo del mínimo)"
        ),
        "en_minimo": (
            "que alcanza exactamente su stock mínimo"
            if singular else "que alcanzan exactamente su stock mínimo"
        ),
        "cerca_minimo": f"cerca del stock mínimo (hasta {f.umbral or 2} unidades por encima)",
        "suficiente": "con stock suficiente por encima del mínimo",
        "con_stock": "con existencias disponibles",
        "minimo_cero": "con stock mínimo configurado en cero",
        "con_prestamos": "con unidades en préstamo",
    }.get(f.condicion or "")
    if cond:
        partes.append(cond)
    es_prestable = getattr(f, "es_prestable", None)
    if es_prestable is not None:
        if singular:
            partes.append(
                "retornable o elegible para préstamo"
                if es_prestable
                else "consumible o no elegible para préstamo"
            )
        else:
            partes.append(
                "retornables o elegibles para préstamo"
                if es_prestable
                else "consumibles o no elegibles para préstamo"
            )
    return " ".join(partes) if partes else "del inventario visible"


def _redactar_respuesta(
    plan,
    resultado,
    movimientos_por_id: Dict[int, List[Dict[str, Any]]],
    prestamos_por_id: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    saldo_prestamos_pendientes: Optional[int] = None,
) -> Dict[str, Any]:
    """Devuelve dict con mensaje_pantalla, mensaje_verbal y datos estructurados."""
    items = resultado.items
    total = resultado.total
    desc = _describir_filtro(plan, singular=(total == 1))
    desc_plural = _describir_filtro(plan, singular=False)
    campos = [c for c in plan.campos if c not in ("ficha", "movimientos")]
    op = plan.operacion
    prestamos_map = prestamos_por_id or {}
    consulta_producto_unico = bool(
        total == 1
        and items
        and (
            plan.filtros.codigos
            or plan.referencia in ("foco", "resultados_previos")
            or plan.filtros.texto
        )
    )
    consulta_de_coleccion = getattr(plan, "alcance_entidad", None) in {"coleccion", "global"}

    # ── Errores de herramienta / permisos (sin inventar) ───────────────────
    if resultado.error == "permiso_denegado":
        perm = resultado.detalle_filtros.get("permiso", "catalogo.ver")
        msg_p = f"No cuentas con el permiso requerido (`{perm}`) para consultar el catálogo o existencias."
        msg_v = "No cuenta con permisos suficientes para consultar el catálogo de inventario."
        return {
            "tipo": "error_permiso",
            "mensaje_pantalla": msg_p,
            "mensaje_verbal": msg_v,
            "datos": {"error_permiso": True, "permiso_requerido": perm},
        }
    if resultado.error == "almacen_no_autorizado":
        almacen_denegado = resultado.detalle_filtros.get("almacen", "ese almacén")
        if isinstance(almacen_denegado, (list, tuple, set)):
            nombres_denegados = [str(nombre).strip() for nombre in almacen_denegado if str(nombre).strip()]
            etiqueta_almacen = nombres_denegados[0] if len(nombres_denegados) == 1 else ", ".join(nombres_denegados)
            valor_almacen_denegado = nombres_denegados[0] if len(nombres_denegados) == 1 else nombres_denegados
        else:
            etiqueta_almacen = str(almacen_denegado)
            valor_almacen_denegado = almacen_denegado
        msg = f"No cuenta con autorización para consultar {etiqueta_almacen}; su cuenta no tiene acceso a ese almacén."
        return {
            "tipo": "error_permiso",
            "mensaje_pantalla": msg,
            "mensaje_verbal": msg,
            "datos": {
                "error_permiso": True,
                "error_permisos": True,
                "permiso_requerido": "almacen.asignado",
                "almacen_denegado": valor_almacen_denegado,
            },
        }
    if resultado.error == "almacen_inexistente":
        msg = f"No encontré un almacén llamado «{resultado.detalle_filtros.get('almacen')}». ¿Cuál almacén desea consultar?"
        return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}
    if resultado.error == "categoria_inexistente":
        cat_nom = resultado.detalle_filtros.get("categoria_sin_coincidencia") or plan.filtros.categoria
        msg = f"No existe la categoría «{cat_nom}» en el inventario."
        return {"tipo": "consulta", "mensaje_pantalla": msg, "mensaje_verbal": msg}

    if op == "contar" and plan.fuente_plan == "fast_path_listado_global":
        if total == 0:
            mensaje = "No hay productos registrados en el inventario que puede consultar."
        elif total == 1:
            mensaje = "Hay un producto registrado en el inventario que puede consultar."
        else:
            mensaje = f"Hay {total} productos registrados en el inventario que puede consultar."
        return {"tipo": "consulta", "mensaje_pantalla": mensaje, "mensaje_verbal": mensaje}

    # ── Sin resultados ─────────────────────────────────────────────────────
    if total == 0:
        if plan.filtros.codigos:
            cods = ", ".join(f"`{c}`" for c in plan.filtros.codigos)
            cods_v = ", ".join(plan.filtros.codigos)
            msg_p = f"No encontré productos con el código {cods} en el inventario disponible para su cuenta."
            msg_v = f"No encontré productos con el código {cods_v} en el inventario disponible para su cuenta."
            return {"tipo": "consulta", "mensaje_pantalla": msg_p, "mensaje_verbal": msg_v}
        relacionados = getattr(resultado, "relacionados", None) or []
        if relacionados:
            rels_p = ", ".join(
                f"**{r['nombre']}** (`{r['codigo']}`, {_stock_cantidad(r['disponible'])})"
                for r in relacionados[:5]
            )
            rels_v = ", ".join(
                f"{r['nombre']} ({_stock_cantidad(r['disponible'])})"
                for r in relacionados[:5]
            )
            texto_no_encontrado = (
                "No encontré productos que coincidan con esa descripción en el inventario al que tiene acceso."
                if plan.filtros.texto
                else f"No encontré productos {desc} en los almacenes disponibles para su cuenta."
            )
            msg_p = f"{texto_no_encontrado} Como referencia relacionada sí tenemos: {rels_p}."
            msg_v = f"{texto_no_encontrado} Como referencia relacionada sí tenemos: {rels_v}."
            return {"tipo": "consulta", "mensaje_pantalla": msg_p, "mensaje_verbal": msg_v}
        if plan.filtros.texto:
            if op == "contar":
                mensaje = (
                    "El conteo es 0: no encontré productos que coincidan con esa descripción "
                    "en el inventario disponible para su cuenta."
                )
            elif op == "sumar_unidades" and "stock" in (plan.campos or []):
                mensaje = (
                    "No encontré productos que coincidan con esa descripción en el inventario "
                    "disponible para su cuenta; el total es 0 unidades disponibles."
                )
            else:
                mensaje = (
                    "No encontré productos que coincidan con esa descripción "
                    "en el inventario disponible para su cuenta."
                )
            return {"tipo": "consulta", "mensaje_pantalla": mensaje, "mensaje_verbal": mensaje}
        if op == "contar":
            msg = f"No hay productos {desc}."
        elif op == "sumar_unidades":
            if "prestamos" in plan.campos and "stock" not in plan.campos:
                msg = f"No hay unidades actualmente en préstamo {desc} (0 unidades)."
            else:
                msg = f"No hay existencias registradas de productos {desc} (0 unidades)."
        else:
            msg = f"No encontré productos {desc} en los almacenes disponibles para su cuenta."
        return {"tipo": "consulta", "mensaje_pantalla": msg, "mensaje_verbal": msg}

    nota_aprox = ""
    if resultado.coincidencia == "aproximada":
        nota_aprox = " (coincidencia aproximada por nombre)"
    elif resultado.coincidencia == "sinonimo":
        nota_aprox = " (coincidencia por sinónimo)"
    elif resultado.coincidencia == "funcional":
        nota_aprox = " (coincidencia funcional)"
    elif resultado.coincidencia == "categoria":
        nota_aprox = " (coincidencia por categoría)"
    elif resultado.coincidencia == "descriptiva":
        nota_aprox = " (coincidencia por descripción)"

    if (
        total == 1
        and items
        and getattr(plan, "alcance", None) == "evento"
        and "movimientos" in (plan.campos or [])
        and not getattr(plan, "orden_movimiento", None)
        and (getattr(plan, "tipo_movimiento", None) or getattr(plan, "fecha_desde", None) or getattr(plan, "fecha_hasta", None))
    ):
        return _redactar_movimiento_tipado(
            items[0], movimientos_por_id.get(items[0]["id"], []), plan.tipo_movimiento,
        )

    # ── Conteo de productos distintos ──────────────────────────────────────
    if op == "contar":
        if total == 1 and items and any((
            plan.filtros.texto,
            plan.filtros.codigos,
            plan.filtros.categoria,
            plan.filtros.almacen,
            plan.filtros.ubicacion,
        )):
            # Un conteo filtrado que arroja una sola referencia debe identificar
            # qué encontró. El dato de stock viene de la herramienta, no del LLM.
            item = items[0]
            pantalla = (
                f"Encontré 1 producto que coincide con {desc}: "
                f"**{_ident(item)}**, con **{_stock_cantidad(item['disponible'])}**{nota_aprox}."
            )
            verbal = (
                f"Encontré un producto que coincide con {desc}: {_ident(item)}, "
                f"con {_stock_cantidad(item['disponible'])}{nota_aprox}."
            )
            return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
        reg_txt = "registrado" if total == 1 else "registrados"
        msg_v = f"Hay {total} {_p(total)} {reg_txt} {desc}{nota_aprox}."
        pantalla = f"**Hay {total} {_p(total)}** {reg_txt} {desc}{nota_aprox}."
        return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": msg_v}

    # ── Suma de unidades físicas ───────────────────────────────────────────
    if op == "sumar_unidades":
        if "prestamos" in plan.campos and "stock" not in plan.campos:
            prest = (
                max(0, int(saldo_prestamos_pendientes))
                if saldo_prestamos_pendientes is not None
                else sum(i["en_prestamo"] for i in items)
            )
            if consulta_producto_unico:
                item = items[0]
                msg_v = f"{_ident(item)} tiene {prest} {_u(prest)} pendientes de devolución."
                pantalla = f"**{_ident(item)}** tiene **{prest} {_u(prest)} pendientes de devolución**."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": msg_v}
            msg_v = f"Hay {prest} {_u(prest)} pendientes de devolución en {total} {_p(total)} {desc}."
            pantalla = f"**{prest} {_u(prest)} pendientes de devolución** en {total} {_p(total)} {desc}."
            return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": msg_v}
        disp = sum(i["disponible"] for i in items)
        prest = sum(i["en_prestamo"] for i in items)
        if getattr(plan, "medida_solicitada", None) == "disponible":
            prest = 0
        if consulta_producto_unico:
            item = items[0]
            msg_v = f"{_ident(item)} tiene {_stock_cantidad(disp)}"
            msg_v += f" y {prest} {_u(prest)} en préstamo." if prest else "."
            pantalla = f"**{_ident(item)}** tiene **{_stock_cantidad(disp)}**"
            pantalla += f" y **{prest} {_u(prest)} en préstamo**." if prest else "."
            return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": msg_v}
        msg_v = f"Hay {_stock_cantidad(disp)} en {total} {_p(total)} {desc}{nota_aprox}"
        msg_v += f", más {prest} en préstamo." if prest else "."
        pantalla = f"**{_stock_cantidad(disp)}**" + (f" y **{prest} en préstamo**" if prest else "")
        pantalla += f" en {total} {_p(total)} {desc}{nota_aprox}."
        return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": msg_v}

    # ── Operaciones extremales (mayor/menor stock) ─────────────────────────
    if op in ("mayor_stock", "menor_stock") and items:
        valor_extremo, ganadores = seleccionar_ganadores_extremos(items, op)
        cantidad_ganadores = len(ganadores)
        mayor = op == "mayor_stock"
        comparativo = "más" if mayor else "menos"
        prefijo = "mayor" if mayor else "menor"
        if cantidad_ganadores == 1:
            elegido = ganadores[0]
            pantalla = (
                f"El producto con {comparativo} unidades disponibles {desc_plural} es "
                f"**{_ident(elegido)}** con **{valor_extremo} {_u(valor_extremo)}**."
            )
            verbal = (
                f"El que tiene {comparativo} existencias es {elegido['nombre']} "
                f"con {valor_extremo} {_u(valor_extremo)}."
            )
        else:
            mostrados = ganadores[:MAX_PANTALLA]
            nombres = "; ".join(f"**{_ident(item)}**" for item in mostrados)
            adicionales = cantidad_ganadores - len(mostrados)
            pantalla = (
                f"Hay un empate: **{cantidad_ganadores} productos** comparten el "
                f"{prefijo} stock {desc_plural}, con **{valor_extremo} {_u(valor_extremo)}** cada uno: "
                f"{nombres}"
            )
            if adicionales:
                pantalla += f" y **{adicionales} más**"
            pantalla += "."
            if cantidad_ganadores <= MAX_VERBAL:
                verbal = (
                    f"Hay un empate entre {cantidad_ganadores} productos. "
                    f"Todos tienen {valor_extremo} {_u(valor_extremo)}: "
                    + "; ".join(item["nombre"] for item in ganadores) + "."
                )
            else:
                verbal = (
                    f"Hay un empate entre {cantidad_ganadores} productos; "
                    f"todos tienen {valor_extremo} {_u(valor_extremo)} disponibles. "
                    "Puede consultar los códigos en pantalla."
                )
        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": verbal,
            "datos": {
                "valor_extremo_disponible": valor_extremo,
                "cantidad_ganadores": cantidad_ganadores,
                "productos_extremo": [
                    {"codigo": item.get("codigo"), "nombre": item.get("nombre"), "disponible": item.get("disponible")}
                    for item in ganadores
                ],
            },
        }

    # ── Explicación de estado (escaso/agotado/en mínimo/cerca/suficiente) ──
    if op == "explicar_estado":
        partes_p, partes_v = [], []
        for i in items[:MAX_VERBAL]:
            d, m = i["disponible"], i["stock_minimo"]
            est = i.get("estado_stock")
            if d == 0 or est == "agotado":
                razon = f"está agotado: tiene 0 unidades disponibles" + (f" y su mínimo es {m}" if m else " (su stock mínimo es 0)")
            elif est == "bajo_minimo" or (m > 0 and d < m):
                razon = f"está por debajo de su mínimo: tiene {_stock_cantidad(d)} y su stock mínimo es {m}, le faltan {m - d}"
            elif est == "en_minimo" or (m > 0 and d == m):
                razon = f"está justo en su mínimo: {_stock_cantidad(d)} con un stock mínimo de {m}"
            elif est == "cerca_minimo":
                razon = f"está próximo a su mínimo (por agotarse): tiene {_stock_cantidad(d)} y su stock mínimo es {m} (solo {d - m} por encima)"
            else:
                razon = f"no está escaso (stock suficiente): tiene {_stock_cantidad(d)} y su stock mínimo es {m}"
            if i["en_prestamo"]:
                razon += f"; además hay {i['en_prestamo']} en préstamo que no cuentan como disponibles"
                activos = prestamos_map.get(i["id"], [])
                if activos:
                    razon += " (" + ", ".join(f"{p['cantidad']} a {p['destinatario']}" for p in activos) + ")"
            partes_p.append(f"- **{_ident(i)}** {razon}.")
            partes_v.append(f"{i['nombre']} {razon}.")
        return {"tipo": "consulta", "mensaje_pantalla": "\n".join(partes_p), "mensaje_verbal": " ".join(partes_v)}

    # ── Comparación entre productos ────────────────────────────────────────
    if op == "comparar" and total >= 2:
        atributos = [
            ("Existencias", lambda i: str(i["disponible"])),
            ("Stock mínimo", lambda i: str(i["stock_minimo"])),
            ("Categoría", lambda i: i["categoria"] or "—"),
            ("Almacén", _almacenes_txt),
            ("Ubicación", _ubicaciones_txt),
            ("Empaque", lambda i: i["empaque"] or "—"),
            ("En préstamo", lambda i: str(i["en_prestamo"])),
        ]
        sub = items[:8]
        difieren = [n for n, fn in atributos if len({fn(i) for i in sub}) > 1]
        comunes = [(n, fn(sub[0])) for n, fn in atributos if len({fn(i) for i in sub}) == 1]
        cab = "| Código | Nombre | " + " | ".join(n for n, _ in atributos) + " |"
        sep = "|" + "---|" * (len(atributos) + 2)
        filas = [f"| {i['codigo']} | {i['nombre']} | " + " | ".join(fn(i) for _, fn in atributos) + " |" for i in sub]
        pantalla = f"Comparación de {len(sub)} {_p(len(sub))} {desc}{nota_aprox}:\n\n" + "\n".join([cab, sep] + filas)
        if total > len(sub):
            pantalla += f"\n\n_Mostrando {len(sub)} de {total}._"
        verbal = f"Comparé {len(sub)} {_p(len(sub))}: " + ", ".join(f"{i['nombre']} ({i['disponible']} {_u(i['disponible'])})" for i in sub) + ". "
        if len({i["disponible"] for i in sub}) > 1:
            mayor = max(sub, key=lambda x: x["disponible"])
            menor = min(sub, key=lambda x: x["disponible"])
            resumen_extremos = (
                f"El que tiene más existencias es {mayor['nombre']} con {mayor['disponible']} {_u(mayor['disponible'])}, "
                f"y el que tiene menos es {menor['nombre']} con {menor['disponible']} {_u(menor['disponible'])}."
            )
            pantalla += f"\n\n{resumen_extremos}"
            verbal += resumen_extremos + " "
        verbal += ("Se diferencian en: " + ", ".join(d.lower() for d in difieren) + ". ") if difieren else "Solo se diferencian en el nombre. "
        if comunes:
            verbal += "Comparten " + ", ".join(f"{n.lower()} {v}" for n, v in comunes[:3]) + "."
        return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}

    # ── Detalle de un producto ─────────────────────────────────────────────
    if total == 1 and op in ("detalle", "listar", "comparar") and not consulta_de_coleccion:
        i = items[0]
        orden_movimiento = getattr(plan, "orden_movimiento", None)
        if orden_movimiento in {"asc", "desc"}:
            movs = movimientos_por_id.get(i["id"], [])
            incluir_saldo_actual = (
                getattr(plan, "alcance", None) == "ambos"
                and "prestamos" in (plan.campos or [])
            )
            prestamos_actuales = prestamos_map.get(i["id"], [])
            if getattr(movs, "permiso_denegado", False):
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": "No cuentas con el permiso requerido (`movimientos.ver`) para consultar el historial.",
                    "mensaje_verbal": "No cuenta con permisos suficientes para consultar el historial de movimientos.",
                    "datos": {"error_permiso": True, "permiso_requerido": "movimientos.ver"},
                }
            if incluir_saldo_actual and getattr(prestamos_actuales, "permiso_denegado", False):
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": "No cuenta con permisos suficientes para consultar los préstamos activos de inventario.",
                    "mensaje_verbal": "No cuenta con permisos suficientes para consultar los préstamos activos de inventario.",
                    "datos": {"error_permiso": True, "permiso_requerido": "prestamos.ver"},
                }
            ordinal = "primer" if orden_movimiento == "asc" else "último"
            mov = movs[0] if movs else None
            if not mov:
                destinatario = f" para {plan.destinatario}" if getattr(plan, "destinatario", None) else ""
                pantalla = f"No hay movimientos registrados{destinatario} para **{_ident(i)}**."
                verbal = f"No hay movimientos registrados{destinatario} para {i['nombre']}."
            else:
                tipo_mov = str(mov.get("tipo") or "movimiento").casefold()
                unidades = _u(mov.get("cantidad", 0))
                destino = f" para {mov['destinatario']}" if mov.get("destinatario") else ""
                fecha = f" ({mov['fecha']})" if mov.get("fecha") else ""
                if getattr(plan, "alcance", None) == "evento" and "stock" in (plan.campos or []):
                    saldo_anterior = mov.get("stock_disponible_anterior")
                    if saldo_anterior is None:
                        pantalla = f"El {ordinal} movimiento de **{_ident(i)}** no tiene registrado el saldo previo."
                        verbal = f"El {ordinal} movimiento de {i['nombre']} no tiene registrado el saldo previo."
                    else:
                        pantalla = (
                            f"Antes del {ordinal} movimiento de **{_ident(i)}** había **{saldo_anterior} "
                            f"{_u(saldo_anterior)} disponibles**{fecha}."
                        )
                        verbal = (
                            f"Antes del {ordinal} movimiento de {i['nombre']} había {saldo_anterior} "
                            f"{_u(saldo_anterior)} disponibles{fecha}."
                        )
                else:
                    pantalla = (
                        f"El {ordinal} movimiento registrado para **{_ident(i)}** fue de **{tipo_mov}**: "
                        f"**{mov.get('cantidad', 0)} {unidades}**{destino}{fecha}."
                    )
                    verbal = (
                        f"El {ordinal} movimiento registrado para {i['nombre']} fue de {tipo_mov}: "
                        f"{mov.get('cantidad', 0)} {unidades}{destino}{fecha}."
                    )
            if incluir_saldo_actual:
                saldo_pendiente = sum(
                    int(prestamo.get("cantidad", 0) or 0)
                    for prestamo in (prestamos_actuales or [])
                )
                if saldo_pendiente:
                    saldo_pantalla = f" Actualmente mantiene **{saldo_pendiente} {_u(saldo_pendiente)} en préstamo**."
                    saldo_verbal = f" Actualmente mantiene {saldo_pendiente} {_u(saldo_pendiente)} en préstamo."
                else:
                    saldo_pantalla = " Actualmente no mantiene unidades en préstamo."
                    saldo_verbal = " Actualmente no mantiene unidades en préstamo."
                pantalla += saldo_pantalla
                verbal += saldo_verbal
            return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
        if getattr(plan, "tipo_movimiento", None):
            return _redactar_movimiento_tipado(
                i,
                movimientos_por_id.get(i["id"], []),
                plan.tipo_movimiento,
            )
        if getattr(plan, "alcance", None) == "evento" and (getattr(plan, "fecha_desde", None) or getattr(plan, "fecha_hasta", None)):
            return _redactar_movimiento_tipado(
                i,
                movimientos_por_id.get(i["id"], []),
                plan.tipo_movimiento,
            )
        if "movimientos" in plan.campos and not campos and "ficha" not in plan.campos:
            movs = movimientos_por_id.get(i["id"], [])
            if getattr(movs, "permiso_denegado", False):
                msg_p = "No cuentas con el permiso requerido (`movimientos.ver`) para consultar el historial de movimientos."
                msg_v = "No cuenta con permisos suficientes para consultar el historial de movimientos de inventario."
                return {
                    "tipo": "error_permiso",
                    "mensaje_pantalla": msg_p,
                    "mensaje_verbal": msg_v,
                    "datos": {"error_permiso": True, "permiso_requerido": "movimientos.ver"},
                }
            pantalla = f"### Movimientos recientes de **{_ident(i)}**\n"
            if movs:
                pantalla += "\n".join(
                    f"- {m['fecha']} · {m['tipo']} · {m['cantidad']}"
                    + (f" → {m['destinatario']}" if m.get("destinatario") else "")
                    + _sufijo_estado_prestamo(m)[0]
                    for m in movs
                )
                resumen = "; ".join(
                    f"{m['tipo'].casefold()} de {m['cantidad']}"
                    + (f" para {m['destinatario']}" if m.get("destinatario") else "")
                    + _sufijo_estado_prestamo(m)[1]
                    for m in movs
                )
                verbal = f"El historial reciente de {i['nombre']} registra: {resumen}."
            else:
                pantalla += "_Sin movimientos registrados._"
                verbal = f"{i['nombre']} no tiene movimientos registrados."
            return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}

        if "ficha" in plan.campos or not campos:
            p_txt, v_txt = _ficha(i, movimientos_por_id.get(i["id"], []), prestamos_map.get(i["id"], []))
            if nota_aprox:
                p_txt = p_txt.replace(f"### {i['nombre']} (`{i['codigo']}`)", f"### {i['nombre']} (`{i['codigo']}`){nota_aprox}")
            return {"tipo": "consulta", "mensaje_pantalla": p_txt, "mensaje_verbal": v_txt}

        # Respuesta directa y ajustada cuando se pregunta por un solo atributo
        if len(campos) == 1:
            c = campos[0]
            if c == "codigo":
                pantalla = f"El código de **{i['nombre']}** es **{i['codigo']}**."
                verbal = f"El código de {i['nombre']} es {i['codigo']}."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "codigo_barras":
                codigo_barras = i.get("codigo_barras")
                pantalla = (
                    f"El código de barras de **{i['nombre']}** es **{codigo_barras}**."
                    if codigo_barras else f"**{i['nombre']}** no tiene un código de barras registrado."
                )
                verbal = (
                    f"El código de barras de {i['nombre']} es {codigo_barras}."
                    if codigo_barras else f"{i['nombre']} no tiene un código de barras registrado."
                )
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "nombre":
                pantalla = f"El producto con código **{i['codigo']}** es **{i['nombre']}**."
                verbal = f"El producto con código {i['codigo']} es {i['nombre']}."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "stock":
                pantalla = f"**{_ident(i)}** tiene **{_stock_txt(i, con_almacen=True)}**."
                verbal = f"{i['nombre']} tiene {_stock_txt(i, con_almacen=False)}."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "stock_minimo":
                pantalla = f"El stock mínimo de **{_ident(i)}** es de **{i['stock_minimo']}** unidades."
                verbal = f"El stock mínimo de {i['nombre']} es de {i['stock_minimo']} unidades."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "ubicacion":
                pantalla = f"**{_ident(i)}** está en **{_ubicaciones_txt(i)}**."
                verbal = f"{i['nombre']} está en {_ubicaciones_txt(i)}."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "almacen":
                pantalla = f"**{_ident(i)}** está registrado en **{_almacenes_txt(i)}**."
                verbal = f"{i['nombre']} está en {_almacenes_txt(i)}."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "categoria":
                pantalla = f"**{_ident(i)}** pertenece a la categoría **{i['categoria'] or 'sin categoría'}**."
                verbal = f"{i['nombre']} pertenece a la categoría {i['categoria'] or 'sin categoría'}."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "empaque":
                if i.get("empaque"):
                    pantalla = f"El empaque de **{_ident(i)}** es **{i['empaque']}**."
                    verbal = f"El empaque de {i['nombre']} es {i['empaque']}."
                else:
                    pantalla = f"**{_ident(i)}** no tiene empaque registrado (se maneja por unidad individual)."
                    verbal = f"{i['nombre']} no tiene empaque registrado."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "prestamos":
                activos = prestamos_map.get(i["id"], [])
                if getattr(activos, "permiso_denegado", False):
                    msg_p = "No cuentas con el permiso requerido (`prestamos.ver`) para consultar los préstamos activos."
                    msg_v = "No cuenta con permisos suficientes para consultar los préstamos activos de inventario."
                    return {
                        "tipo": "error_permiso",
                        "mensaje_pantalla": msg_p,
                        "mensaje_verbal": msg_v,
                        "datos": {"error_permiso": True, "permiso_requerido": "prestamos.ver"},
                    }
                if activos:
                    detalles = ", ".join(f"{p['cantidad']} {_u(p['cantidad'])} a {p['destinatario']}" for p in activos)
                    pantalla = f"**{_ident(i)}** tiene préstamos activos: **{detalles}**."
                    verbal = f"{i['nombre']} tiene préstamos activos: {detalles}."
                elif i["en_prestamo"]:
                    pantalla = f"**{_ident(i)}** tiene **{i['en_prestamo']} {_u(i['en_prestamo'])}** en préstamo."
                    verbal = f"{i['nombre']} tiene {i['en_prestamo']} {_u(i['en_prestamo'])} en préstamo."
                else:
                    pantalla = f"**{_ident(i)}** no tiene unidades en préstamo actualmente."
                    verbal = f"{i['nombre']} no tiene unidades en préstamo actualmente."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}
            if c == "es_prestable":
                if i.get("es_prestable"):
                    pantalla = f"Sí, **{_ident(i)}** es un artículo **prestable** (retornable)."
                    verbal = f"Sí, {i['nombre']} es prestable."
                else:
                    pantalla = f"No, **{_ident(i)}** **no es prestable** (es un insumo consumible)."
                    verbal = f"No, {i['nombre']} no es prestable; es un artículo consumible."
                return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}

        if set(campos) == {"ubicacion", "almacen"} and "movimientos" not in plan.campos:
            if i.get("ubicaciones"):
                extra_alms = _almacenes_no_cubiertos_por_ubicacion(i)
                if not extra_alms:
                    pantalla = f"**{_ident(i)}** está en **{_ubicaciones_txt(i)}**."
                    verbal = f"{i['nombre']} está en {_ubicaciones_txt(i)}."
                else:
                    extra_txt = ", ".join(extra_alms)
                    pantalla = f"**{_ident(i)}** está en **{_ubicaciones_txt(i)}** (también en **{extra_txt}**)."
                    verbal = f"{i['nombre']} está en {_ubicaciones_txt(i)}, y también en {extra_txt}."
            else:
                pantalla = f"**{_ident(i)}** está en **{_almacenes_txt(i)}** (sin ubicación específica registrada)."
                verbal = f"{i['nombre']} está en {_almacenes_txt(i)}, sin ubicación específica registrada."
            return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}

        campos_item = _campos_efectivos_item(i, campos)
        valores = [(c, _valor_campo(i, c, prestamos_map.get(i["id"], []))) for c in campos_item]
        verbal = f"{i['nombre']}, código {i['codigo']}: " + "; ".join(
            f"{_ETIQUETA_CAMPO.get(c, c)}: {v}" for c, v in valores if v
        ) + "."
        pantalla = f"**{_ident(i)}**\n" + "\n".join(f"- **{_ETIQUETA_CAMPO.get(c, c)}:** {v}" for c, v in valores if v)
        if "movimientos" in plan.campos:
            movs = movimientos_por_id.get(i["id"], [])
            pantalla += "\n\n**Últimos movimientos:**\n" + (
                "\n".join(
                    f"- {m['fecha']} · {m['tipo']} · {m['cantidad']}"
                    + (f" → {m['destinatario']}" if m.get("destinatario") else "")
                    + _sufijo_estado_prestamo(m)[0]
                    for m in movs
                )
                if movs
                else "_Sin movimientos registrados._"
            )
            verbal += f" Tiene {len(movs)} movimientos recientes." if movs else " No tiene movimientos registrados."
        return {"tipo": "consulta", "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}

    # ── Listado (o detalle de varios) ──────────────────────────────────────
    campos_lista = list(campos or [])
    if plan.filtros.condicion == "con_prestamos" and "prestamos" not in campos_lista:
        campos_lista.append("prestamos")

    def linea(i):
        campos_i = _campos_efectivos_item(i, campos_lista)
        extra = "; ".join(
            f"{_ETIQUETA_CAMPO.get(c, c)}: {valor}"
            for c in campos_i
            if c not in ("nombre", "codigo") and (valor := _valor_campo(i, c, prestamos_map.get(i["id"], [])))
        )
        if extra:
            base = f"- **{i['codigo']}** {i['nombre']} — {extra}"
        elif plan.filtros.condicion == "agotado":
            base = f"- **{i['codigo']}** {i['nombre']} (agotado)"
        else:
            base = f"- **{i['codigo']}** {i['nombre']} ({i['disponible']} {_u(i['disponible'])})"
        movs = movimientos_por_id.get(i["id"], [])
        if movs and ("movimientos" in plan.campos or "ficha" in plan.campos):
            sub_movs = "\n  " + "\n  ".join(f"· {m['fecha']} {m['tipo']} {m['cantidad']}" for m in movs[:2])
            base += sub_movs
        return base

    mostrados = items[:MAX_PANTALLA]
    pantalla = f"Encontré **{total} {_p(total)}** {desc}{nota_aprox}:\n\n" + "\n".join(linea(i) for i in mostrados)
    if getattr(plan, "fuente_plan", None) == "fast_path_prioridad_reposicion":
        pantalla += (
            "\n\n_Orden: agotados primero y después menor cobertura respecto del stock mínimo. "
            "El inventario no registra una prioridad comercial o crítica._"
        )
    if total > len(mostrados):
        pantalla += f"\n\n_Mostrando {len(mostrados)} de {total} resultados._"
    verbal_items = []
    for i in items[:MAX_VERBAL]:
        campos_i = _campos_efectivos_item(i, campos_lista)
        extra = "; ".join(
            f"{_ETIQUETA_CAMPO.get(c, c)}: {valor}"
            for c in campos_i
            if c not in ("nombre", "codigo") and (valor := _valor_campo(i, c, prestamos_map.get(i["id"], [])))
        )
        if extra:
            verbal_items.append(f"{i['nombre']}, {extra}")
        elif plan.filtros.condicion == "agotado":
            verbal_items.append(f"{i['nombre']} (agotado)")
        else:
            verbal_items.append(f"{i['nombre']} ({i['disponible']} disponibles)")
    if total <= MAX_VERBAL:
        verbal = f"Encontré {total} {_p(total)} {desc}{nota_aprox}: " + "; ".join(verbal_items) + "."
    else:
        verbal = f"Encontré {total} {_p(total)} {desc}{nota_aprox}. Puede consultar los resultados en pantalla."
    if getattr(plan, "fuente_plan", None) == "fast_path_prioridad_reposicion":
        verbal = (
            "No hay una prioridad comercial registrada; ordené los resultados con los datos de stock, "
            "agotados primero y luego menor cobertura del mínimo. " + verbal
        )
    tipo_respuesta = "consulta"
    if op == "detalle" and 2 <= total <= 5:
        pantalla += "\n\n_¿A cuál de estas referencias se refiere específicamente?_"
        verbal += " ¿A cuál de estas referencias se refiere específicamente?"
        tipo_respuesta = "aclaracion"
    return {"tipo": tipo_respuesta, "mensaje_pantalla": pantalla, "mensaje_verbal": verbal}


def _resolver_navegacion_consulta(plan) -> Tuple[str, str, str]:
    """Determina la sección web y URL de «Mostrar todos» según los filtros activos."""
    from urllib.parse import quote

    filtros = getattr(plan, "filtros", None)
    condicion = getattr(filtros, "condicion", None) if filtros else None
    campos = getattr(plan, "campos", None) or []

    almacen_id = None
    categoria_id = None
    ubicacion_id = None
    try:
        from inventario.models import Almacen, Categoria, Ubicacion
        alm_nom = getattr(filtros, "almacen", None) if filtros else None
        if alm_nom:
            almacen_id = (
                Almacen.objects.filter(nombre__iexact=str(alm_nom).strip(), activo=True, eliminado=False)
                .values_list("id", flat=True)
                .first()
            )
        cat_nom = getattr(filtros, "categoria", None) if filtros else None
        if not cat_nom and filtros and getattr(filtros, "texto", None):
            cat_match_txt = (
                Categoria.objects.filter(nombre__iexact=str(filtros.texto).strip(), activo=True, eliminado=False)
                .values_list("id", "nombre")
                .first()
            )
            if cat_match_txt:
                categoria_id, cat_nom = cat_match_txt[0], cat_match_txt[1]
        if cat_nom and not categoria_id:
            cat_qs = Categoria.objects.filter(nombre__iexact=str(cat_nom).strip(), activo=True, eliminado=False)
            if almacen_id:
                cat_qs = cat_qs.filter(almacen_principal_id=almacen_id)
            categoria_id = cat_qs.values_list("id", flat=True).first()
        ubi_nom = getattr(filtros, "ubicacion", None) if filtros else None
        if ubi_nom:
            ubi_qs = Ubicacion.objects.filter(nombre__iexact=str(ubi_nom).strip(), activo=True, eliminado=False)
            if almacen_id:
                ubi_qs = ubi_qs.filter(almacen_id=almacen_id)
            ubicacion_id = ubi_qs.values_list("id", flat=True).first()
    except Exception:
        pass

    es_contexto_prestamos = bool(condicion == "con_prestamos" or "prestamos" in campos)
    es_contexto_movimientos = bool(
        "movimientos" in campos
        or getattr(plan, "alcance", None) == "evento"
        or getattr(plan, "tipo_movimiento", None)
    )

    if es_contexto_prestamos:
        params = ["vista=PRESTAMOS"]
        if almacen_id:
            params.append(f"almacen_id={almacen_id}")
        return "prestamos", "/inventario/movimientos/?" + "&".join(params), "Mostrar todos los préstamos"

    if es_contexto_movimientos:
        params = ["vista=TODOS"]
        if almacen_id:
            params.append(f"almacen_id={almacen_id}")
        url = "/inventario/movimientos/?" + "&".join(params)
        return "movimientos", url, "Mostrar todos los movimientos"

    if condicion == "agotado":
        params = ["nivel=CRITICA"]
        if almacen_id:
            params.append(f"almacen_id={almacen_id}")
        return "alertas_critica", "/inventario/alertas/?" + "&".join(params), "Mostrar todos los agotados"

    if condicion == "bajo_minimo":
        params = ["nivel=BAJA"]
        if almacen_id:
            params.append(f"almacen_id={almacen_id}")
        return "alertas_baja", "/inventario/alertas/?" + "&".join(params), "Mostrar todos con stock bajo"

    if condicion in ("requiere_reposicion", "en_minimo", "cerca_minimo"):
        params = [f"almacen_id={almacen_id}"] if almacen_id else []
        url = "/inventario/alertas/" + (("?" + "&".join(params)) if params else "")
        return "alertas", url, "Mostrar todas las alertas de stock"

    params = []
    q_filtro = getattr(filtros, "texto", None) if filtros else None
    cat_nom_efectivo = (getattr(filtros, "categoria", None) if filtros else None) or (
        str(q_filtro).strip() if (categoria_id and q_filtro) else None
    )
    if q_filtro and not (categoria_id and not (getattr(filtros, "categoria", None) if filtros else None)):
        params.append(f"q={quote(str(q_filtro))}")
    if almacen_id:
        params.append(f"almacen_id={almacen_id}")
    if categoria_id:
        params.append(f"categoria_id={categoria_id}")
    if ubicacion_id:
        params.append(f"ubicacion_id={ubicacion_id}")
    url = "/inventario/productos/" + (("?" + "&".join(params)) if params else "")
    if cat_nom_efectivo and categoria_id:
        etiqueta = f"Mostrar todos en {cat_nom_efectivo}"
    else:
        etiqueta = "Mostrar todos los productos"
    return "productos", url, etiqueta


def redactar(
    plan,
    resultado,
    movimientos_por_id: Dict[int, List[Dict[str, Any]]],
    prestamos_por_id: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    saldo_prestamos_pendientes: Optional[int] = None,
) -> Dict[str, Any]:
    """Redacta la respuesta y conserva un resumen estructurado del mismo resultado.

    La parte estructurada permite que clientes y verificadores contrasten conteos,
    saldos e identificadores sin inferirlos de la prosa. Solo contiene filas que la
    herramienta ya filtró según los permisos del usuario.
    """
    respuesta = _redactar_respuesta(
        plan,
        resultado,
        movimientos_por_id,
        prestamos_por_id,
        saldo_prestamos_pendientes,
    )
    if not isinstance(respuesta, dict):
        return respuesta

    items = list(getattr(resultado, "items", []) or [])
    contexto_nav, url_todos, etiqueta_todos = _resolver_navegacion_consulta(plan)

    if respuesta.get("tipo") == "aclaracion":
        # Ambiguous searches still need machine-readable candidates so a later
        # ordinal reply can be verified and resolved against the same results.
        if items:
            candidatos = [
                {"id": item.get("id"), "codigo": item.get("codigo"), "nombre": item.get("nombre")}
                for item in items[:5]
            ]
            respuesta.setdefault("datos", {}).update({
                "productos": items[:200],
                "candidatos": candidatos,
                "total": int(getattr(resultado, "total", len(items)) or 0),
                "resultados_completos": len(items) <= 200,
                "unidades_disponibles": sum(int(i.get("disponible", 0)) for i in items),
                "unidades_en_prestamo": sum(int(i.get("en_prestamo", 0)) for i in items),
                "operacion_consulta": getattr(plan, "operacion", None),
                "contexto_navegacion": contexto_nav,
                "url_ver_todos": url_todos,
                "etiqueta_ver_todos": etiqueta_todos,
            })
        return respuesta
    if respuesta.get("tipo") not in ("consulta", "exito"):
        return respuesta

    datos = respuesta.setdefault("datos", {})
    if not isinstance(datos, dict):
        datos = respuesta["datos"] = {}
    limite_datos = 200
    datos.update({
        "productos": items[:limite_datos],
        "total": int(getattr(resultado, "total", len(items)) or 0),
        "resultados_completos": len(items) <= limite_datos,
        "unidades_disponibles": sum(int(i.get("disponible", 0)) for i in items),
        "unidades_en_prestamo": sum(int(i.get("en_prestamo", 0)) for i in items),
        "operacion_consulta": getattr(plan, "operacion", None),
        "contexto_navegacion": contexto_nav,
        "url_ver_todos": url_todos,
        "etiqueta_ver_todos": etiqueta_todos,
    })
    historial = {}
    for producto_id, movimientos in (movimientos_por_id or {}).items():
        if getattr(movimientos, "permiso_denegado", False):
            historial[str(producto_id)] = {"permiso_denegado": True}
        else:
            historial[str(producto_id)] = [dict(m) for m in movimientos]
    if historial:
        datos["movimientos"] = historial
    return respuesta


def redactar_movimientos_globales(
    plan,
    movimientos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Redacta listados y conteos globales de movimientos filtrados por tipo, período y almacén."""
    error = getattr(movimientos, "error", None)
    if error == "permiso_denegado":
        return {
            "tipo": "error_permiso",
            "mensaje_pantalla": "No cuentas con el permiso requerido (`movimientos.ver`) para consultar el historial de movimientos.",
            "mensaje_verbal": "No cuenta con permisos suficientes para consultar el historial de movimientos de inventario.",
            "datos": {"error_permiso": True, "permiso_requerido": "movimientos.ver"},
        }
    if error == "almacen_no_autorizado":
        alm = getattr(movimientos, "almacen_filtrado", None) or plan.filtros.almacen or "ese almacén"
        msg = f"No cuenta con autorización para consultar {alm}; su cuenta no tiene acceso a ese almacén."
        return {
            "tipo": "error_permiso",
            "mensaje_pantalla": msg,
            "mensaje_verbal": msg,
            "datos": {
                "error_permiso": True,
                "error_permisos": True,
                "permiso_requerido": "almacen.asignado",
                "almacen_denegado": alm,
            },
        }
    if error == "almacen_inexistente":
        alm = getattr(movimientos, "almacen_filtrado", None) or plan.filtros.almacen
        msg = f"No encontré un almacén llamado «{alm}». ¿Cuál almacén desea consultar?"
        return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}
    if error == "categoria_inexistente":
        cat = getattr(movimientos, "categoria_sin_coincidencia", None) or plan.filtros.categoria
        msg = f"No existe la categoría «{cat}» en el inventario."
        return {"tipo": "consulta", "mensaje_pantalla": msg, "mensaje_verbal": msg}
    if error == "rango_temporal_invalido":
        msg = "El período de fechas indicado no es válido. Por favor indique una fecha de inicio anterior o igual a la fecha final."
        return {"tipo": "aclaracion", "mensaje_pantalla": msg, "mensaje_verbal": msg}

    total = getattr(movimientos, "total_eventos", None)
    if total is None:
        total = len(movimientos)
    unidades = getattr(movimientos, "unidades_evento", None)
    if unidades is None:
        unidades = sum(int(m.get("cantidad", 0) or 0) for m in movimientos)
    periodo = getattr(movimientos, "periodo_descripcion", None)
    almacen_filtrado = getattr(movimientos, "almacen_filtrado", None)

    tipo_codigo = getattr(plan, "tipo_movimiento", None)
    etiqueta_tipo = None
    if tipo_codigo == "TRASLADO":
        etiqueta_tipo = "traslado"
    elif tipo_codigo:
        try:
            from inventario.models import Movimiento
            etiqueta_tipo = dict(Movimiento.Tipo.choices).get(tipo_codigo)
        except Exception:
            etiqueta_tipo = None
        etiqueta_tipo = str(etiqueta_tipo or tipo_codigo).casefold()

    frase_tipo_plural = f"movimientos de {etiqueta_tipo}" if etiqueta_tipo else "movimientos"
    frase_tipo_singular = f"movimiento de {etiqueta_tipo}" if etiqueta_tipo else "movimiento"
    almacen_txt = f" en {almacen_filtrado}" if almacen_filtrado else ""
    periodo_txt = f" {periodo}" if periodo else ""
    destinatario_txt = f" para {plan.destinatario}" if getattr(plan, "destinatario", None) else ""
    contexto_frase = f"{destinatario_txt}{almacen_txt}{periodo_txt}"

    from urllib.parse import quote
    if tipo_codigo == "PRESTAMO":
        vista_url = "PRESTAMOS"
        etiqueta_todos = "Mostrar todos los préstamos"
        contexto_nav = "prestamos"
    elif tipo_codigo == "ENTRADA":
        vista_url = "ENTRADA"
        etiqueta_todos = "Mostrar todas las entradas"
        contexto_nav = "movimientos"
    elif tipo_codigo in ("TRASLADO", "TRASLADO_INTERNO", "TRASLADO_ENTRE"):
        vista_url = "TRASLADO"
        etiqueta_todos = "Mostrar todos los traslados"
        contexto_nav = "movimientos"
    elif tipo_codigo in ("SALIDA", "DEVOLUCION", "BAJA_PRESTAMO"):
        vista_url = "SALIDA"
        etiqueta_todos = "Mostrar todos los movimientos"
        contexto_nav = "movimientos"
    else:
        vista_url = "TODOS"
        etiqueta_todos = "Mostrar todos los movimientos"
        contexto_nav = "movimientos"

    params_url = []
    if vista_url:
        params_url.append(f"vista={vista_url}")
    if getattr(plan, "fecha_desde", None):
        params_url.append(f"fecha_desde={plan.fecha_desde}")
    if getattr(plan, "fecha_hasta", None):
        params_url.append(f"fecha_hasta={plan.fecha_hasta}")
    if getattr(plan, "destinatario", None):
        params_url.append(f"q={quote(str(plan.destinatario))}")
    url_todos = "/inventario/movimientos/" + (("?" + "&".join(params_url)) if params_url else "")

    datos_resumen = {
        "movimientos": [dict(m) for m in movimientos],
        "total": total,
        "total_eventos": total,
        "unidades_evento": unidades,
        "operacion_consulta": getattr(plan, "operacion", None) or "listar",
        "contexto_navegacion": contexto_nav,
        "url_ver_todos": url_todos,
        "etiqueta_ver_todos": etiqueta_todos,
    }

    if total == 0:
        msg = f"No se registraron {frase_tipo_plural}{contexto_frase} en los almacenes disponibles para su cuenta."
        return {
            "tipo": "consulta",
            "mensaje_pantalla": msg,
            "mensaje_verbal": msg,
            "datos": datos_resumen,
        }

    if getattr(plan, "operacion", None) == "contar":
        etiqueta_n = frase_tipo_singular if total == 1 else frase_tipo_plural
        pantalla = (
            f"Se {'registró' if total == 1 else 'registraron'} **{total} {etiqueta_n}**"
            f"{contexto_frase}, por un total de **{unidades} {_u(unidades)}**."
        )
        verbal = (
            f"Se {'registró' if total == 1 else 'registraron'} {total} {etiqueta_n}"
            f"{contexto_frase}, con un total de {unidades} {_u(unidades)}."
        )
        return {
            "tipo": "consulta",
            "mensaje_pantalla": pantalla,
            "mensaje_verbal": verbal,
            "datos": datos_resumen,
        }

    limite_pantalla = max(MAX_PANTALLA, 10)
    mostrados = list(movimientos)[:limite_pantalla]
    etiqueta_n = frase_tipo_singular if total == 1 else frase_tipo_plural
    lineas = []
    for m in mostrados:
        dest = f" → {m['destinatario']}" if m.get("destinatario") else ""
        alm = f" ({m['almacen']})" if m.get("almacen") else ""
        cod = f"**{m['producto_codigo']}** " if m.get("producto_codigo") else ""
        est_p, _ = _sufijo_estado_prestamo(m)
        lineas.append(
            f"- {m['fecha']} · {cod}{m.get('producto_nombre', '')} · "
            f"{m['tipo']} · **{m['cantidad']} {_u(m['cantidad'])}**{dest}{alm}{est_p}"
        )
    pantalla = (
        f"Se {'registró' if total == 1 else 'registraron'} **{total} {etiqueta_n}**"
        f"{contexto_frase} (**{unidades} {_u(unidades)}** en total):\n\n"
        + "\n".join(lineas)
    )
    if total > len(mostrados):
        pantalla += f"\n\n_Mostrando {len(mostrados)} de {total} movimientos._"

    if total <= MAX_VERBAL:
        partes_v = []
        for m in mostrados[:MAX_VERBAL]:
            dest_v = f" para {m['destinatario']}" if m.get("destinatario") else ""
            _, est_v = _sufijo_estado_prestamo(m)
            partes_v.append(
                f"{m['tipo'].casefold()} de {m['cantidad']} {_u(m['cantidad'])} de {m.get('producto_nombre', '')}{dest_v}{est_v}"
            )
        verbal = (
            f"Se {'registró' if total == 1 else 'registraron'} {total} {etiqueta_n}"
            f"{contexto_frase}: " + "; ".join(partes_v) + "."
        )
    else:
        verbal = (
            f"Se registraron {total} {etiqueta_n}{contexto_frase}, "
            f"por un total de {unidades} {_u(unidades)}. Puede consultar el detalle en pantalla."
        )
    return {
        "tipo": "consulta",
        "mensaje_pantalla": pantalla,
        "mensaje_verbal": verbal,
        "datos": datos_resumen,
    }
