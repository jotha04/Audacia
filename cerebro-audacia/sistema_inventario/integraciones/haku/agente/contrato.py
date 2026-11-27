"""Contrato Canónico de Comprensión para HAKU Inventario — PlanHaku.

Fase II-C.1: Unificación progresiva del contrato de comprensión del agente.

Define:
1. La estructura canónica 'PlanHaku' que unifica los conceptos de PlanAgente,
   ExtraccionSemantica e IntencionResultado en 4 familias fundamentales:
   - CONSULTA (detalle, catalogo, stock, movimientos, prestamos, contar, sumar_unidades, etc.)
   - MUTACION_PROPUESTA (entrada, salida, prestamo, devolucion, traslado, cambiar_stock_minimo, inhabilitar, habilitar)
   - FORMULARIO_ASISTIDO (crear_producto, editar_producto)
   - CONVERSACION (saludo, agradecimiento, ayuda, capacidades, fuera_dominio, aclaracion)
2. Validaciones deterministas de estructura, tipos, límites y combinaciones válidas.
3. Adaptadores bidireccionales y de compatibilidad con los contratos existentes:
   - PlanHaku -> PlanAgente
   - PlanHaku -> ExtraccionSemantica
   - PlanHaku -> IntencionResultado
   - PlanAgente -> PlanHaku
4. El prompt de sistema canónico con Few-Shot Grounding optimizado para el Cerebro Central
   (Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf) con longitud acotada a un máximo de 800 tokens.

GARANTÍAS DE SEGURIDAD:
- Este módulo es ESTRICTAMENTE DECLARATIVO Y DE LECTURA / TRANSFORMACIÓN EN MEMORIA.
- NUNCA ejecuta mutaciones ni escrituras en la base de datos.
- Las mutaciones se representan únicamente como propuestas para el carril transaccional de services.py.
"""

from __future__ import annotations

import json
import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("integraciones.haku.contrato")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENUMS Y CONSTANTES DEL CONTRATO CANÓNICO
# ─────────────────────────────────────────────────────────────────────────────

class FamiliaIntencion(str, Enum):
    """Familias fundamentales de intención reconocidas por PlanHaku."""
    CONSULTA = "consulta"
    MUTACION_PROPUESTA = "mutacion_propuesta"
    FORMULARIO_ASISTIDO = "formulario_asistido"
    CONVERSACION = "conversacion"


class OperacionConsulta(str, Enum):
    """Operaciones de consulta sobre el inventario."""
    DETALLE = "detalle"
    CATALOGO = "catalogo"
    STOCK = "stock"
    MOVIMIENTOS = "movimientos"
    PRESTAMOS = "prestamos"
    CONTAR = "contar"                  # Conteo de referencias / productos diferentes
    SUMAR_UNIDADES = "sumar_unidades"  # Agrega las unidades de la medida y el alcance solicitados
    COMPARAR = "comparar"
    MAYOR_STOCK = "mayor_stock"
    MENOR_STOCK = "menor_stock"
    EXPLICAR_ESTADO = "explicar_estado"


class AccionMutacion(str, Enum):
    """Acciones operativas que requieren confirmación transaccional posterior."""
    ENTRADA = "entrada"
    SALIDA = "salida"
    PRESTAMO = "prestamo"
    DEVOLUCION = "devolucion"
    TRASLADO = "traslado"
    CAMBIAR_STOCK_MINIMO = "cambiar_stock_minimo"
    INHABILITAR = "inhabilitar"
    HABILITAR = "habilitar"


class AccionFormulario(str, Enum):
    """Acciones orientadas a abrir o prellenar formularios interactivos."""
    CREAR_PRODUCTO = "crear_producto"
    EDITAR_PRODUCTO = "editar_producto"


class SubtipoConversacion(str, Enum):
    """Subtipos para interacciones conversacionales, sociales y de ayuda."""
    SALUDO = "saludo"
    AGRADECIMIENTO = "agradecimiento"
    AYUDA = "ayuda"
    CAPACIDADES = "capacidades"
    FUERA_DOMINIO = "fuera_dominio"
    ACLARACION = "aclaracion"


class CondicionFiltro(str, Enum):
    """Condiciones de filtrado sobre el catálogo."""
    AGOTADO = "agotado"
    BAJO_MINIMO = "bajo_minimo"
    REQUIERE_REPOSICION = "requiere_reposicion"
    EN_MINIMO = "en_minimo"
    CERCA_MINIMO = "cerca_minimo"
    SUFICIENTE = "suficiente"
    CON_STOCK = "con_stock"
    MINIMO_CERO = "minimo_cero"
    CON_PRESTAMOS = "con_prestamos"


class ReferenciaContextual(str, Enum):
    """Ámbito de referencia contextual en la conversación."""
    NINGUNA = "ninguna"
    FOCO = "foco"
    RESULTADOS_PREVIOS = "resultados_previos"


FAMILIAS_VALIDAS = {f.value for f in FamiliaIntencion}
OPERACIONES_VALIDAS = {o.value for o in OperacionConsulta}
ACCIONES_MUTACION_VALIDAS = {a.value for a in AccionMutacion}
ACCIONES_FORMULARIO_VALIDAS = {a.value for a in AccionFormulario}
SUBTIPOS_CONVERSACION_VALIDOS = {s.value for s in SubtipoConversacion}
CONDICIONES_VALIDAS = {c.value for c in CondicionFiltro}
REFERENCIAS_VALIDAS = {r.value for r in ReferenciaContextual}
ALCANCES_ENTIDAD_VALIDOS = {"producto", "coleccion", "global", "indefinido"}
MEDIDAS_SOLICITADAS_VALIDAS = {
    "disponible", "pendiente_devolucion", "evento", "ambos", "producto", "atributo", "ninguna",
}

CAMPOS_VALIDOS = {
    "stock",
    "stock_minimo",
    "ubicacion",
    "categoria",
    "codigo",
    "nombre",
    "prestamos",
    "movimientos",
    "ficha",
    "empaque",
    "es_prestable",
    "historial",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTRUCTURAS DE DATOS TIPADAS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FiltrosConsultaHaku:
    """Filtros canónicos para consultas de lectura en el catálogo."""
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

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.texto:
            d["texto"] = self.texto
        if self.codigos:
            d["codigos"] = list(self.codigos)
        if self.categoria:
            d["categoria"] = self.categoria
        if self.almacen:
            d["almacen"] = self.almacen
        if self.ubicacion:
            d["ubicacion"] = self.ubicacion
        if self.condicion:
            d["condicion"] = self.condicion
        if self.umbral is not None:
            d["umbral"] = self.umbral
        if self.es_prestable is not None:
            d["es_prestable"] = self.es_prestable
        if self.categoria_explicita:
            d["categoria_explicita"] = True
        if self.categoria_generica:
            d["categoria_generica"] = True
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> FiltrosConsultaHaku:
        if not data or not isinstance(data, dict):
            return cls()
        cods = data.get("codigos") or []
        if isinstance(cods, str):
            cods = [cods]
        return cls(
            texto=_str_limpio(data.get("texto")),
            codigos=[str(c).strip().upper() for c in cods if _str_limpio(c)],
            categoria=_str_limpio(data.get("categoria")),
            almacen=_str_limpio(data.get("almacen")),
            ubicacion=_str_limpio(data.get("ubicacion")),
            condicion=_str_limpio(data.get("condicion")),
            umbral=_int_limpio(data.get("umbral")),
            categoria_explicita=bool(data.get("categoria_explicita", False)),
            categoria_generica=bool(data.get("categoria_generica", False)),
            es_prestable=(
                data.get("es_prestable")
                if isinstance(data.get("es_prestable"), bool)
                else None
            ),
        )


@dataclass
class DatosMutacionHaku:
    """Parámetros de una mutación propuesta para validación en el backend."""
    producto: Optional[str] = None
    codigo: Optional[str] = None
    cantidad: Optional[int] = None
    destinatario: Optional[str] = None
    almacen: Optional[str] = None
    origen: Optional[str] = None
    destino: Optional[str] = None
    motivo: Optional[str] = None
    nuevo_valor: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.producto:
            d["producto"] = self.producto
        if self.codigo:
            d["codigo"] = self.codigo
        if self.cantidad is not None:
            d["cantidad"] = self.cantidad
        if self.destinatario:
            d["destinatario"] = self.destinatario
        if self.almacen:
            d["almacen"] = self.almacen
        if self.origen:
            d["origen"] = self.origen
        if self.destino:
            d["destino"] = self.destino
        if self.motivo:
            d["motivo"] = self.motivo
        if self.nuevo_valor is not None:
            d["nuevo_valor"] = self.nuevo_valor
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> DatosMutacionHaku:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            producto=_str_limpio(data.get("producto")),
            codigo=_str_limpio(data.get("codigo")),
            cantidad=_int_limpio(data.get("cantidad")),
            destinatario=_str_limpio(data.get("destinatario")),
            almacen=_str_limpio(data.get("almacen")),
            origen=_str_limpio(data.get("origen")),
            destino=_str_limpio(data.get("destino")),
            motivo=_str_limpio(data.get("motivo")),
            nuevo_valor=data.get("nuevo_valor"),
        )


@dataclass
class DatosFormularioHaku:
    """Parámetros para abrir o prellenar un formulario asistido (alta o edición)."""
    nombre: Optional[str] = None
    codigo: Optional[str] = None
    categoria: Optional[str] = None
    stock_minimo: Optional[int] = None
    almacen: Optional[str] = None
    ubicacion: Optional[str] = None
    stock_inicial: Optional[int] = None
    stock_actual: Optional[int] = None
    incertidumbre_cantidades: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.nombre:
            d["nombre"] = self.nombre
        if self.codigo:
            d["codigo"] = self.codigo
        if self.categoria:
            d["categoria"] = self.categoria
        if self.stock_minimo is not None:
            d["stock_minimo"] = self.stock_minimo
        if self.almacen:
            d["almacen"] = self.almacen
        if self.ubicacion:
            d["ubicacion"] = self.ubicacion
        if self.stock_inicial is not None:
            d["stock_inicial"] = self.stock_inicial
        if self.stock_actual is not None:
            d["stock_actual"] = self.stock_actual
        if self.incertidumbre_cantidades:
            d["incertidumbre_cantidades"] = True
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> DatosFormularioHaku:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            nombre=_str_limpio(data.get("nombre")),
            codigo=_str_limpio(data.get("codigo")),
            categoria=_str_limpio(data.get("categoria")),
            stock_minimo=_int_limpio(data.get("stock_minimo")),
            almacen=_str_limpio(data.get("almacen")),
            ubicacion=_str_limpio(data.get("ubicacion")),
            stock_inicial=_int_limpio(data.get("stock_inicial")),
            stock_actual=_int_limpio(data.get("stock_actual")),
            incertidumbre_cantidades=bool(data.get("incertidumbre_cantidades", False)),
        )


@dataclass
class PlanHaku:
    """Contrato Canónico de Comprensión para HAKU Inventario.

    Representa la intención del usuario y sus ranuras estructuradas tras el proceso
    de interpretación en el Cerebro Central o por enrutamiento determinista O(1).
    """
    familia: str
    operacion: Optional[str] = None
    accion: Optional[str] = None
    subtipo: Optional[str] = None
    filtros: FiltrosConsultaHaku = field(default_factory=FiltrosConsultaHaku)
    campos: List[str] = field(default_factory=list)
    medida_solicitada: Optional[str] = None
    referencia: str = "ninguna"
    alcance_entidad: Optional[str] = None
    alcance: Optional[str] = None
    tipo_movimiento: Optional[str] = None
    fecha_desde: Optional[str] = None
    fecha_hasta: Optional[str] = None
    mutacion: Optional[DatosMutacionHaku] = None
    formulario: Optional[DatosFormularioHaku] = None
    pregunta_aclaracion: Optional[str] = None
    parametros_faltantes: List[str] = field(default_factory=list)
    nivel_ambiguedad: str = "ninguno"  # "ninguno" | "parcial" | "alto"
    confianza: float = 1.0
    fuente: str = "llm"  # "llm" | "fast_path" | "reglas" | "adaptador"
    crudo: Dict[str, Any] = field(default_factory=dict)
    datos_extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa PlanHaku a un diccionario limpio (omitiendo valores nulos)."""
        d: Dict[str, Any] = {
            "familia": self.familia,
            "referencia": self.referencia,
        }
        if self.operacion:
            d["operacion"] = self.operacion
        if self.alcance:
            d["alcance"] = self.alcance
        if self.alcance_entidad:
            d["alcance_entidad"] = self.alcance_entidad
        if self.tipo_movimiento:
            d["tipo_movimiento"] = self.tipo_movimiento
        if self.fecha_desde:
            d["fecha_desde"] = self.fecha_desde
        if self.fecha_hasta:
            d["fecha_hasta"] = self.fecha_hasta
        if self.accion:
            d["accion"] = self.accion
        if self.subtipo:
            d["subtipo"] = self.subtipo
        if not self.filtros.vacio():
            d["filtros"] = self.filtros.to_dict()
        if self.campos:
            d["campos"] = list(self.campos)
        if self.medida_solicitada:
            d["medida_solicitada"] = self.medida_solicitada
        if self.mutacion:
            m_dict = self.mutacion.to_dict()
            if m_dict:
                d["mutacion"] = m_dict
        if self.formulario:
            f_dict = self.formulario.to_dict()
            if f_dict:
                d["formulario"] = f_dict
        if self.pregunta_aclaracion:
            d["pregunta_aclaracion"] = self.pregunta_aclaracion
        if self.parametros_faltantes:
            d["parametros_faltantes"] = list(self.parametros_faltantes)
        if self.nivel_ambiguedad != "ninguno":
            d["nivel_ambiguedad"] = self.nivel_ambiguedad
        if self.confianza < 1.0:
            d["confianza"] = round(self.confianza, 2)
        if self.fuente != "llm":
            d["fuente"] = self.fuente
        if self.datos_extra:
            d["datos_extra"] = dict(self.datos_extra)
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanHaku:
        """Construye un PlanHaku a partir de un diccionario."""
        if not isinstance(data, dict):
            return cls(familia=FamiliaIntencion.CONVERSACION.value, subtipo=SubtipoConversacion.ACLARACION.value)

        familia_raw = _str_limpio(data.get("familia")) or FamiliaIntencion.CONVERSACION.value
        operacion_raw = _str_limpio(data.get("operacion"))
        accion_raw = _str_limpio(data.get("accion"))
        subtipo_raw = _str_limpio(data.get("subtipo") or data.get("subtipo_conversacion"))
        referencia_raw = _str_limpio(data.get("referencia")) or ReferenciaContextual.NINGUNA.value

        campos_raw = data.get("campos") or []
        if isinstance(campos_raw, str):
            campos_raw = [campos_raw]
        campos_limpios = [str(c).strip().lower() for c in campos_raw if _str_limpio(c) in CAMPOS_VALIDOS]

        filtros_obj = FiltrosConsultaHaku.from_dict(data.get("filtros"))

        mutacion_obj = None
        if data.get("mutacion") or accion_raw in ACCIONES_MUTACION_VALIDAS:
            m_data = data.get("mutacion") or {}
            # Si campos de mutación vinieron en la raíz del dict
            for k in ("producto", "codigo", "cantidad", "destinatario", "almacen", "origen", "destino", "motivo", "nuevo_valor"):
                if k in data and k not in m_data:
                    m_data[k] = data[k]
            mutacion_obj = DatosMutacionHaku.from_dict(m_data)

        formulario_obj = None
        if data.get("formulario") or accion_raw in ACCIONES_FORMULARIO_VALIDAS:
            f_data = data.get("formulario") or {}
            for k in ("nombre", "codigo", "categoria", "stock_minimo", "almacen", "ubicacion", "stock_inicial", "stock_actual"):
                if k in data and k not in f_data:
                    f_data[k] = data[k]
            formulario_obj = DatosFormularioHaku.from_dict(f_data)

        return cls(
            familia=familia_raw,
            operacion=operacion_raw,
            accion=accion_raw,
            subtipo=subtipo_raw,
            filtros=filtros_obj,
            campos=campos_limpios,
            medida_solicitada=_str_limpio(data.get("medida_solicitada")),
            referencia=referencia_raw,
            alcance_entidad=_str_limpio(data.get("alcance_entidad")),
            alcance=_str_limpio(data.get("alcance")),
            tipo_movimiento=_str_limpio(data.get("tipo_movimiento")),
            fecha_desde=_str_limpio(data.get("fecha_desde")),
            fecha_hasta=_str_limpio(data.get("fecha_hasta")),
            mutacion=mutacion_obj,
            formulario=formulario_obj,
            pregunta_aclaracion=_str_limpio(data.get("pregunta_aclaracion") or data.get("pregunta")),
            parametros_faltantes=data.get("parametros_faltantes") or [],
            nivel_ambiguedad=_str_limpio(data.get("nivel_ambiguedad")) or "ninguno",
            confianza=float(data.get("confianza", 1.0)),
            fuente=_str_limpio(data.get("fuente")) or "llm",
            crudo=dict(data),
            datos_extra=dict(data.get("datos_extra") or {}),
        )

    def resumen(self) -> Dict[str, Any]:
        """Representación concisa para logs y telemetría sin datos personales extensos."""
        return {
            "familia": self.familia,
            "operacion": self.operacion,
            "accion": self.accion,
            "subtipo": self.subtipo,
            "filtros": self.filtros.to_dict() if not self.filtros.vacio() else {},
            "campos": self.campos,
            "medida_solicitada": self.medida_solicitada,
            "referencia": self.referencia,
            "alcance_entidad": self.alcance_entidad,
            "alcance": self.alcance,
            "tipo_movimiento": self.tipo_movimiento,
            "fecha_desde": self.fecha_desde,
            "fecha_hasta": self.fecha_hasta,
            "fuente": self.fuente,
            "confianza": round(self.confianza, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDADOR DETERMINISTA DE PLAN HAKU
# ─────────────────────────────────────────────────────────────────────────────

def validar_plan_haku(datos: Dict[str, Any], texto_usuario: Optional[str] = None) -> Tuple[Optional[PlanHaku], List[str]]:
    """Valida y normaliza un diccionario JSON contra el contrato PlanHaku.

    Reglas deterministas:
    1. La familia debe ser una de las 4 permitidas: consulta, mutacion_propuesta,
       formulario_asistido, conversacion.
    2. En 'consulta', valida operaciones y campos. Distingue 'stock_minimo' como atributo
       frente a 'minimo_cero' como filtro.
    3. En 'mutacion_propuesta', requiere acción válida y valida que las cantidades sean enteros > 0.
    4. En 'formulario_asistido', clasifica en 'crear_producto' o 'editar_producto'.
    5. No inventa entidades ausentes en el mensaje del usuario.
    6. Identifica parámetros faltantes e incertidumbres cuando corresponda.
    """
    errores: List[str] = []
    if not isinstance(datos, dict):
        return None, ["Los datos proporcionados no son un diccionario JSON válido."]

    # 1. Normalizar familia
    familia_raw = _str_limpio(datos.get("familia"))
    if not familia_raw:
        # Intento de inferencia defensiva por claves presentes
        if datos.get("formulario") or datos.get("accion") in ACCIONES_FORMULARIO_VALIDAS:
            familia_raw = FamiliaIntencion.FORMULARIO_ASISTIDO.value
        elif datos.get("mutacion") or datos.get("accion") in ACCIONES_MUTACION_VALIDAS:
            familia_raw = FamiliaIntencion.MUTACION_PROPUESTA.value
        elif datos.get("operacion") or datos.get("campos") or datos.get("filtros"):
            familia_raw = FamiliaIntencion.CONSULTA.value
        elif datos.get("subtipo") or datos.get("tipo") in SUBTIPOS_CONVERSACION_VALIDOS:
            familia_raw = FamiliaIntencion.CONVERSACION.value
        else:
            errores.append("Campo 'familia' no especificado ni inferible.")
            return None, errores

    familia = familia_raw.lower().strip()
    if familia not in FAMILIAS_VALIDAS:
        errores.append(f"Familia '{familia}' no reconocida. Válidas: {sorted(FAMILIAS_VALIDAS)}")
        return None, errores

    plan = PlanHaku.from_dict(datos)
    plan.familia = familia
    if plan.medida_solicitada not in MEDIDAS_SOLICITADAS_VALIDAS:
        plan.medida_solicitada = None

    # 2. Validación de referencia
    if plan.referencia not in REFERENCIAS_VALIDAS:
        plan.referencia = ReferenciaContextual.NINGUNA.value

    if plan.alcance_entidad not in ALCANCES_ENTIDAD_VALIDOS:
        plan.alcance_entidad = None

    if plan.alcance not in {"actual", "evento", "ambos", "indefinido"}:
        plan.alcance = None
    for campo in ("fecha_desde", "fecha_hasta"):
        valor = getattr(plan, campo)
        try:
            fecha = datetime.date.fromisoformat(valor) if valor else None
            if valor and (fecha is None or fecha.isoformat() != valor):
                fecha = None
        except (TypeError, ValueError):
            fecha = None
        setattr(plan, campo, fecha.isoformat() if fecha else None)
    if plan.fecha_desde and plan.fecha_hasta and plan.fecha_desde > plan.fecha_hasta:
        plan.fecha_desde = None
        plan.fecha_hasta = None
    if plan.fecha_desde or plan.fecha_hasta:
        plan.alcance = "evento"
    if plan.tipo_movimiento:
        try:
            from inventario.models import Movimiento
            tipos_movimiento = {codigo for codigo, _ in Movimiento.Tipo.choices}
        except Exception:
            tipos_movimiento = set()
        if plan.tipo_movimiento not in tipos_movimiento:
            plan.tipo_movimiento = None
        else:
            plan.alcance = "evento"
            plan.campos = [c for c in plan.campos if c not in ("stock", "prestamos")]
            if "movimientos" not in plan.campos:
                plan.campos.append("movimientos")

    # 3. Validaciones por familia
    if familia == FamiliaIntencion.CONSULTA.value:
        if plan.operacion and plan.operacion not in OPERACIONES_VALIDAS:
            errores.append(f"Operación de consulta '{plan.operacion}' no reconocida. Válidas: {sorted(OPERACIONES_VALIDAS)}")
            plan.operacion = OperacionConsulta.DETALLE.value

        if not plan.operacion:
            # Si hay filtros de catálogo o colección sin producto específico
            if plan.filtros.condicion in (
                CondicionFiltro.AGOTADO.value,
                CondicionFiltro.BAJO_MINIMO.value,
                CondicionFiltro.REQUIERE_REPOSICION.value,
            ):
                plan.operacion = OperacionConsulta.CATALOGO.value
            elif plan.filtros.codigos or plan.filtros.texto or plan.referencia == ReferenciaContextual.FOCO.value:
                plan.operacion = OperacionConsulta.DETALLE.value
            else:
                plan.operacion = OperacionConsulta.CATALOGO.value

        # Separación canónica: Si el usuario solicita el atributo 'stock_minimo',
        # NUNCA debe marcarse la condición 'minimo_cero'
        if "stock_minimo" in plan.campos and plan.filtros.condicion == CondicionFiltro.MINIMO_CERO.value:
            plan.filtros.condicion = None

        # Si pide explícitamente contar productos frente a sumar unidades
        if plan.operacion == OperacionConsulta.CONTAR.value and "stock" in plan.campos:
            # Conteo de referencias
            pass

    elif familia == FamiliaIntencion.MUTACION_PROPUESTA.value:
        if not plan.accion or plan.accion not in ACCIONES_MUTACION_VALIDAS:
            errores.append(f"Acción de mutación '{plan.accion}' no reconocida. Válidas: {sorted(ACCIONES_MUTACION_VALIDAS)}")

        if not plan.mutacion:
            plan.mutacion = DatosMutacionHaku()

        # Validación de cantidad
        if plan.mutacion.cantidad is not None:
            if plan.mutacion.cantidad <= 0:
                errores.append(f"La cantidad de una mutación debe ser mayor a 0 (recibido: {plan.mutacion.cantidad}).")
                plan.mutacion.cantidad = None

        # Validación de parámetros obligatorios según acción
        if plan.accion == AccionMutacion.PRESTAMO.value:
            if not plan.mutacion.destinatario:
                plan.parametros_faltantes.append("destinatario")
            if not plan.mutacion.producto and not plan.mutacion.codigo and plan.referencia == ReferenciaContextual.NINGUNA.value:
                plan.parametros_faltantes.append("producto")

        elif plan.accion == AccionMutacion.TRASLADO.value:
            if not plan.mutacion.destino:
                plan.parametros_faltantes.append("destino")

        if plan.parametros_faltantes:
            plan.nivel_ambiguedad = "parcial"

    elif familia == FamiliaIntencion.FORMULARIO_ASISTIDO.value:
        if not plan.accion or plan.accion not in ACCIONES_FORMULARIO_VALIDAS:
            errores.append(f"Acción de formulario '{plan.accion}' no reconocida. Válidas: {sorted(ACCIONES_FORMULARIO_VALIDAS)}")
            plan.accion = AccionFormulario.CREAR_PRODUCTO.value

        if not plan.formulario:
            plan.formulario = DatosFormularioHaku()

        # Distinción de stock_inicial vs stock_actual
        if plan.formulario.stock_inicial is not None and plan.formulario.stock_inicial < 0:
            plan.formulario.stock_inicial = None
        if plan.formulario.stock_actual is not None and plan.formulario.stock_actual < 0:
            plan.formulario.stock_actual = None

        # Si se mencionan dos cantidades distintas sin especificar claramente cuál es el saldo de apertura
        if (
            plan.formulario.stock_inicial is not None
            and plan.formulario.stock_actual is not None
            and plan.formulario.stock_inicial != plan.formulario.stock_actual
        ):
            plan.formulario.incertidumbre_cantidades = True

    elif familia == FamiliaIntencion.CONVERSACION.value:
        if not plan.subtipo or plan.subtipo not in SUBTIPOS_CONVERSACION_VALIDOS:
            plan.subtipo = SubtipoConversacion.ACLARACION.value

    return plan, errores


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADAPTADORES HACIA CONTRATOS EXISTENTES
# ─────────────────────────────────────────────────────────────────────────────

def adaptar_plan_haku_a_plan_agente(plan: PlanHaku) -> Any:
    """Convierte un PlanHaku canónico al contrato PlanAgente existente."""
    from .herramientas import FiltrosConsulta
    from .planificador import PlanAgente

    f_orig = FiltrosConsulta(
        texto=plan.filtros.texto,
        codigos=list(plan.filtros.codigos),
        categoria=plan.filtros.categoria,
        almacen=plan.filtros.almacen,
        ubicacion=plan.filtros.ubicacion,
        condicion=plan.filtros.condicion,
        umbral=plan.filtros.umbral,
        categoria_explicita=plan.filtros.categoria_explicita,
        categoria_generica=plan.filtros.categoria_generica,
        es_prestable=plan.filtros.es_prestable,
    )

    if plan.familia == FamiliaIntencion.CONSULTA.value:
        return PlanAgente(
            tipo="consulta",
            operacion=plan.operacion or "detalle",
            filtros=f_orig,
            referencia=plan.referencia,
            alcance_entidad=plan.alcance_entidad,
            alcance=plan.alcance,
            tipo_movimiento=plan.tipo_movimiento,
            fecha_desde=plan.fecha_desde,
            fecha_hasta=plan.fecha_hasta,
            campos=list(plan.campos),
            medida_solicitada=plan.medida_solicitada,
            fuente_plan=plan.fuente,
            crudo=plan.to_dict(),
        )

    elif plan.familia == FamiliaIntencion.MUTACION_PROPUESTA.value:
        mut = plan.mutacion or DatosMutacionHaku()
        if mut.almacen and not f_orig.almacen:
            f_orig.almacen = mut.almacen
        return PlanAgente(
            tipo="operacion",
            accion=plan.accion,
            producto=mut.producto,
            cantidad=mut.cantidad,
            destinatario=mut.destinatario,
            destino=mut.destino,
            valor=mut.nuevo_valor if isinstance(mut.nuevo_valor, int) else None,
            filtros=f_orig,
            referencia=plan.referencia,
            fuente_plan=plan.fuente,
            crudo=plan.to_dict(),
        )

    elif plan.familia == FamiliaIntencion.FORMULARIO_ASISTIDO.value:
        form = plan.formulario or DatosFormularioHaku()
        # En el agente existente, mapear a operación asistida conservando metadatos
        return PlanAgente(
            tipo="operacion",
            accion=plan.accion or "crear_producto",
            producto=form.nombre,
            cantidad=form.stock_inicial or form.stock_actual,
            filtros=f_orig,
            referencia=plan.referencia,
            fuente_plan=plan.fuente,
            crudo=plan.to_dict(),
        )

    else:  # CONVERSACION
        subtipo = plan.subtipo or "aclaracion"
        tipo_map = {
            "saludo": "social",
            "agradecimiento": "social",
            "capacidades": "capacidades",
            "ayuda": "capacidades",
            "fuera_dominio": "fuera_dominio",
            "aclaracion": "aclaracion",
        }
        return PlanAgente(
            tipo=tipo_map.get(subtipo, "aclaracion"),
            referencia=plan.referencia,
            pregunta=plan.pregunta_aclaracion,
            fuente_plan=plan.fuente,
            crudo=plan.to_dict(),
        )


def adaptar_plan_haku_a_extraccion_semantica(plan: PlanHaku) -> Any:
    """Convierte un PlanHaku canónico al contrato ExtraccionSemantica de interprete_semantico.py."""
    from ..lenguaje.interprete_semantico import ExtraccionSemantica

    if plan.familia == FamiliaIntencion.CONSULTA.value:
        # Atributos específicos
        if "ubicacion" in plan.campos:
            return ExtraccionSemantica(
                intencion="consultar_atributo",
                producto=plan.filtros.texto or (plan.filtros.codigos[0] if plan.filtros.codigos else None),
                atributo="ubicacion",
                almacen=plan.filtros.almacen,
                confianza=plan.confianza,
                fuente="llm",
                datos_extra=plan.to_dict(),
            )
        if "stock_minimo" in plan.campos:
            return ExtraccionSemantica(
                intencion="consultar_atributo",
                producto=plan.filtros.texto or (plan.filtros.codigos[0] if plan.filtros.codigos else None),
                atributo="stock_minimo",
                almacen=plan.filtros.almacen,
                confianza=plan.confianza,
                fuente="llm",
                datos_extra=plan.to_dict(),
            )
        # Reposición
        if plan.filtros.condicion == CondicionFiltro.AGOTADO.value:
            return ExtraccionSemantica(
                intencion="consultar_reposicion",
                filtro_reposicion="agotados",
                almacen=plan.filtros.almacen,
                confianza=plan.confianza,
                fuente="llm",
                datos_extra=plan.to_dict(),
            )
        if plan.filtros.condicion == CondicionFiltro.BAJO_MINIMO.value:
            return ExtraccionSemantica(
                intencion="consultar_reposicion",
                filtro_reposicion="bajo_stock",
                almacen=plan.filtros.almacen,
                confianza=plan.confianza,
                fuente="llm",
                datos_extra=plan.to_dict(),
            )
        # Catálogo general vs almacén
        if plan.operacion == OperacionConsulta.CATALOGO.value:
            int_cat = "consultar_catalogo_almacen" if plan.filtros.almacen else "consultar_catalogo"
            return ExtraccionSemantica(
                intencion=int_cat,
                almacen=plan.filtros.almacen,
                confianza=plan.confianza,
                fuente="llm",
                datos_extra=plan.to_dict(),
            )
        # Stock general
        return ExtraccionSemantica(
            intencion="consultar_stock",
            producto=plan.filtros.texto or (plan.filtros.codigos[0] if plan.filtros.codigos else None),
            almacen=plan.filtros.almacen,
            confianza=plan.confianza,
            fuente="llm",
            datos_extra=plan.to_dict(),
        )

    elif plan.familia == FamiliaIntencion.MUTACION_PROPUESTA.value:
        mut = plan.mutacion or DatosMutacionHaku()
        extra = dict(plan.datos_extra)
        if mut.motivo:
            extra["motivo"] = mut.motivo
        return ExtraccionSemantica(
            intencion="proponer_operacion",
            tipo_operacion=plan.accion,
            producto=mut.producto,
            cantidad=mut.cantidad,
            destinatario=mut.destinatario,
            almacen=mut.almacen,
            origen=mut.origen or mut.almacen,
            destino=mut.destino,
            confianza=plan.confianza,
            fuente="llm",
            datos_extra=extra,
        )

    elif plan.familia == FamiliaIntencion.FORMULARIO_ASISTIDO.value:
        form = plan.formulario or DatosFormularioHaku()
        extra = form.to_dict()
        extra.update(plan.datos_extra)
        return ExtraccionSemantica(
            intencion=plan.accion or "crear_producto",
            producto=form.nombre,
            almacen=form.almacen,
            cantidad=form.stock_inicial or form.stock_actual,
            confianza=plan.confianza,
            fuente="llm",
            datos_extra=extra,
        )

    else:  # CONVERSACION
        subtipo = plan.subtipo or "desconocido"
        return ExtraccionSemantica(
            intencion=subtipo,
            confianza=plan.confianza,
            fuente="llm",
            datos_extra=plan.to_dict(),
        )


def adaptar_plan_haku_a_intencion_resultado(plan: PlanHaku, usuario: Any = None) -> Any:
    """Convierte un PlanHaku canónico a IntencionResultado de normalizador.py."""
    from ..lenguaje.normalizador import IntencionResultado, IntencionTipo

    if plan.familia == FamiliaIntencion.CONSULTA.value:
        if "ubicacion" in plan.campos:
            return IntencionResultado(
                tipo=IntencionTipo.CONSULTAR_UBICACION,
                codigo_producto=plan.filtros.codigos[0] if plan.filtros.codigos else None,
                datos_extra=plan.to_dict(),
            )
        if plan.filtros.condicion == CondicionFiltro.AGOTADO.value:
            return IntencionResultado(
                tipo=IntencionTipo.PRODUCTOS_SIN_STOCK,
                datos_extra=plan.to_dict(),
            )
        if plan.filtros.condicion == CondicionFiltro.BAJO_MINIMO.value:
            return IntencionResultado(
                tipo=IntencionTipo.CONSULTAR_FALTANTES_REPOSICION,
                datos_extra=plan.to_dict(),
            )
        if plan.operacion == OperacionConsulta.CATALOGO.value:
            return IntencionResultado(
                tipo=IntencionTipo.LISTAR_CATALOGO,
                datos_extra=plan.to_dict(),
            )
        return IntencionResultado(
            tipo=IntencionTipo.CONSULTAR_STOCK,
            codigo_producto=plan.filtros.codigos[0] if plan.filtros.codigos else None,
            datos_extra=plan.to_dict(),
        )

    elif plan.familia == FamiliaIntencion.MUTACION_PROPUESTA.value:
        mut = plan.mutacion or DatosMutacionHaku()
        mapa_mut = {
            AccionMutacion.PRESTAMO.value: IntencionTipo.PRESTAMO,
            AccionMutacion.DEVOLUCION.value: IntencionTipo.DEVOLUCION,
            AccionMutacion.ENTRADA.value: IntencionTipo.REGISTRAR_ENTRADA,
            AccionMutacion.SALIDA.value: IntencionTipo.REGISTRAR_SALIDA,
            AccionMutacion.TRASLADO.value: IntencionTipo.TRASLADAR_PRODUCTO,
            AccionMutacion.INHABILITAR.value: IntencionTipo.INHABILITAR_PRODUCTO,
            AccionMutacion.HABILITAR.value: IntencionTipo.HABILITAR_PRODUCTO,
            AccionMutacion.CAMBIAR_STOCK_MINIMO.value: IntencionTipo.EDITAR_PRODUCTO,
        }
        tipo_int = mapa_mut.get(plan.accion)
        if tipo_int is None:
            return IntencionResultado(
                tipo=IntencionTipo.SOLICITAR_ACLARACION,
                mensaje_aclaracion="No pude identificar con precisión qué operación desea realizar. ¿Podría indicarla nuevamente?",
                datos_extra=plan.to_dict(),
            )
        return IntencionResultado(
            tipo=tipo_int,
            codigo_producto=mut.codigo,
            cantidad=mut.cantidad,
            destinatario=mut.destinatario,
            motivo=mut.motivo,
            requiere_confirmacion=True,
            datos_extra=plan.to_dict(),
        )

    elif plan.familia == FamiliaIntencion.FORMULARIO_ASISTIDO.value:
        tipo_form = IntencionTipo.CREAR_PRODUCTO if plan.accion == AccionFormulario.CREAR_PRODUCTO.value else IntencionTipo.EDITAR_PRODUCTO
        return IntencionResultado(
            tipo=tipo_form,
            datos_extra=plan.to_dict(),
        )

    else:  # CONVERSACION
        mapa_conv = {
            SubtipoConversacion.SALUDO.value: IntencionTipo.SALUDO,
            SubtipoConversacion.AGRADECIMIENTO.value: IntencionTipo.AGRADECIMIENTO,
            SubtipoConversacion.FUERA_DOMINIO.value: IntencionTipo.NO_INTENCION_INVENTARIO,
            SubtipoConversacion.ACLARACION.value: IntencionTipo.SOLICITAR_ACLARACION,
        }
        tipo_c = mapa_conv.get(plan.subtipo, IntencionTipo.SOLICITUD_NO_COMPRENDIDA)
        return IntencionResultado(
            tipo=tipo_c,
            mensaje_aclaracion=plan.pregunta_aclaracion,
            datos_extra=plan.to_dict(),
        )


def adaptar_plan_agente_a_plan_haku(plan_agente: Any) -> PlanHaku:
    """Convierte un PlanAgente existente al contrato PlanHaku canónico."""
    tipo = getattr(plan_agente, "tipo", "consulta")
    f_orig = getattr(plan_agente, "filtros", None)

    filtros_haku = FiltrosConsultaHaku(
        texto=getattr(f_orig, "texto", None),
        codigos=list(getattr(f_orig, "codigos", [])),
        categoria=getattr(f_orig, "categoria", None),
        almacen=getattr(f_orig, "almacen", None),
        ubicacion=getattr(f_orig, "ubicacion", None),
        condicion=getattr(f_orig, "condicion", None),
        umbral=getattr(f_orig, "umbral", None),
        categoria_explicita=getattr(f_orig, "categoria_explicita", False),
        categoria_generica=getattr(f_orig, "categoria_generica", False),
        es_prestable=getattr(f_orig, "es_prestable", None),
    ) if f_orig else FiltrosConsultaHaku()

    if tipo == "consulta":
        return PlanHaku(
            familia=FamiliaIntencion.CONSULTA.value,
            operacion=getattr(plan_agente, "operacion", "detalle"),
            filtros=filtros_haku,
            campos=list(getattr(plan_agente, "campos", [])),
            referencia=getattr(plan_agente, "referencia", "ninguna"),
            alcance_entidad=getattr(plan_agente, "alcance_entidad", None),
            alcance=getattr(plan_agente, "alcance", None),
            tipo_movimiento=getattr(plan_agente, "tipo_movimiento", None),
            fecha_desde=getattr(plan_agente, "fecha_desde", None),
            fecha_hasta=getattr(plan_agente, "fecha_hasta", None),
            medida_solicitada=getattr(plan_agente, "medida_solicitada", None),
            fuente=getattr(plan_agente, "fuente_plan", "llm"),
            crudo=getattr(plan_agente, "crudo", {}),
        )

    elif tipo == "operacion":
        accion = getattr(plan_agente, "accion", None)
        if accion in ACCIONES_FORMULARIO_VALIDAS:
            return PlanHaku(
                familia=FamiliaIntencion.FORMULARIO_ASISTIDO.value,
                accion=accion,
                formulario=DatosFormularioHaku(
                    nombre=getattr(plan_agente, "producto", None),
                    stock_inicial=getattr(plan_agente, "cantidad", None),
                    almacen=filtros_haku.almacen,
                ),
                filtros=filtros_haku,
                referencia=getattr(plan_agente, "referencia", "ninguna"),
                fuente=getattr(plan_agente, "fuente_plan", "llm"),
                crudo=getattr(plan_agente, "crudo", {}),
            )
        else:
            return PlanHaku(
                familia=FamiliaIntencion.MUTACION_PROPUESTA.value,
                accion=accion,
                mutacion=DatosMutacionHaku(
                    producto=getattr(plan_agente, "producto", None),
                    cantidad=getattr(plan_agente, "cantidad", None),
                    destinatario=getattr(plan_agente, "destinatario", None),
                    destino=getattr(plan_agente, "destino", None),
                    almacen=filtros_haku.almacen,
                    nuevo_valor=getattr(plan_agente, "valor", None),
                ),
                filtros=filtros_haku,
                referencia=getattr(plan_agente, "referencia", "ninguna"),
                fuente=getattr(plan_agente, "fuente_plan", "llm"),
                crudo=getattr(plan_agente, "crudo", {}),
            )

    else:
        mapa_sub = {
            "social": SubtipoConversacion.SALUDO.value,
            "capacidades": SubtipoConversacion.CAPACIDADES.value,
            "fuera_dominio": SubtipoConversacion.FUERA_DOMINIO.value,
            "aclaracion": SubtipoConversacion.ACLARACION.value,
        }
        return PlanHaku(
            familia=FamiliaIntencion.CONVERSACION.value,
            subtipo=mapa_sub.get(tipo, SubtipoConversacion.ACLARACION.value),
            referencia=getattr(plan_agente, "referencia", "ninguna"),
            pregunta_aclaracion=getattr(plan_agente, "pregunta", None),
            fuente=getattr(plan_agente, "fuente_plan", "llm"),
            crudo=getattr(plan_agente, "crudo", {}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. PROMPT CANÓNICO CON FEW-SHOT GROUNDING (MÁXIMO 800 TOKENS)
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_SISTEMA_CANONICO = """Planificador HAKU: interpreta mensaje+contexto; clasifica por tarea antes de entidad; devuelve solo JSON sin responder ni inventar datos; omite claves vacías.
Contrato: familia=consulta|mutacion_propuesta|formulario_asistido|conversacion; operacion=detalle|catalogo|contar|sumar_unidades|comparar|mayor_stock|menor_stock; medida_solicitada=disponible|pendiente_devolucion|evento|ambos|producto|atributo|ninguna; accion=entrada|salida|prestamo|devolucion|traslado|cambiar_stock_minimo|inhabilitar|habilitar|crear_producto|editar_producto; Filtros: texto,codigos,categoria,almacen,ubicacion,condicion,umbral,es_prestable; "campos":[],"referencia":"ninguna|foco|resultados_previos"; alcance_entidad=producto|coleccion|global|indefinido; alcance=actual|evento|ambos|indefinido; fecha_desde/fecha_hasta=ISO inclusivas o null.
Consulta: contar=referencias; sumar_unidades=unidades; catalogo=registros; detalle=atributos o prestamos individuales. «Cuantos de una clase» pide unidades salvo que solicite referencias. Una ubicación mencionada como contexto no es campo de salida.
Custodia: decide por estado físico. disponible=libres dentro; pendiente_devolucion=unidades prestadas a terceros cuyo retorno aún se debe, bajo custodia temporal fuera del inventario aunque no diga préstamo/devolución; ambos solo si pide ambos. Sin dimensión, usa disponible; stock=disponible, prestamos=saldo, movimientos=eventos.
Atributos: pedir mínimo configurado devuelve detalle con stock y stock_minimo (salida, no filtro). Elegibilidad filtra es_prestable=true (retornable) o false (consumible); singular=atributo. «Se puede prestar»=elegibilidad; «está prestado»/«falta devolver»=saldo/evento.
Historial: convierte periodos relativos o fechas a fecha_desde/fecha_hasta ISO inclusivas usando fecha local; sin periodo, null. Si pregunta por stock antes de un movimiento, usa campos=["stock"], alcance=evento y orden pedido.
Alcance: producto=referencia; coleccion=clase; global=todo; indefinido=ambiguo. Aclara singular ambigua. Resuelve categoria/almacen en catalogo vivo; candidatos no limitan listados. Conserva foco hasta entidad nueva; resultados_previos exige coleccion. filtros.texto=descriptor, no marco.
Stock: agotado=0; bajo_minimo=0<disp<min; en_minimo=igual; cerca_minimo=sobre mínimo hasta margen; suficiente=resto positivo; requiere_reposicion=agotados+bajo_minimo. No intercambies estados. Reposicion es listado; no aclares entre stock y prestamos. Un filtro selecciona; solo es salida si se pide esa cantidad.
Distingue consulta actual de encargo externo concreto: crear/editar productos son operaciones de inventario (formulario_asistido/mutacion_propuesta aunque falten datos), no encargos externos; contenido ajeno=fuera_dominio. Cortesía=social; capacidades=funciones de Hacu. Comparar exige dos entidades. Propón cambios, nunca ejecutes. Stock inicial no es entrada. Solo JSON."""


def obtener_metricas_prompt_canonico() -> Dict[str, Any]:
    """Calcula las métricas de caracteres, palabras y tokens estimados del prompt canónico."""
    chars = len(PROMPT_SISTEMA_CANONICO)
    words = len(PROMPT_SISTEMA_CANONICO.split())
    # Estimación para Llama-3 BPE (promedio 3.8 a 4.0 chars/token en español + JSON)
    tokens_est_38 = round(chars / 3.8, 1)
    tokens_est_40 = round(chars / 4.0, 1)
    return {
        "caracteres": chars,
        "palabras": words,
        "tokens_estimados_bpe_38": tokens_est_38,
        "tokens_estimados_bpe_40": tokens_est_40,
        "limite_objetivo": 800,
        "cumple_limite": tokens_est_40 <= 800 and tokens_est_38 <= 800,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. HELPERS DE SANITIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def _str_limpio(val: Any) -> Optional[str]:
    if val is None or isinstance(val, (bool, list, dict)):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("null", "none", "n/a", "ninguno", "ninguna", "undefined") else None


def _int_limpio(val: Any) -> Optional[int]:
    if val is None or isinstance(val, (bool, list, dict)):
        return None
    try:
        return int(float(str(val).strip()))
    except (TypeError, ValueError):
        return None
