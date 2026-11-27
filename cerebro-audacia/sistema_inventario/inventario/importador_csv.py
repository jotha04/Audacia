"""
Servicio desacoplado para análisis, normalización, validación, previsualización
y persistencia atómica e idempotente del catálogo CSV en AudacIA.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction

logger = logging.getLogger("inventario.importador")

# Diccionario curado de correcciones ortográficas seguras y de alta confianza
CORRECCIONES_ORTOGRAFICAS_SEGURAS = {
    "Cinta transprente Gruesa": "Cinta transparente gruesa",
    "Cinta doblefast gruesa": "Cinta doble faz gruesa",
    "Nilon": "Nailon",
    "Recma de papel natural": "Resma de papel natural",
    "Recma de papel blanco": "Resma de papel blanco",
    "Paquete temperas": "Paquete témperas",
    "Cinta doblefast delgada": "Cinta doble faz delgada",
    "CajaGrapas pequeñas": "Caja grapas pequeñas",
    "Caja Grapas gransdes": "Caja grapas grandes",
    "Banderas separadoras (paquete?": "Banderas separadoras (paquete)",
    "Post-it peque": "Post-it pequeños",
    "Tisas": "Tizas",
    "Marcado sharpie normal": "Marcador Sharpie normal",
    "liquid paper liquido": "Corrector líquido (Liquid Paper)",
    "Resina epoxica": "Resina epóxica",
    "Goma liquida": "Goma líquida",
}

# Homologación propuesta de los valores de la columna Ubicación.
# Dado que la columna contiene etiquetas del tipo 'Almacén A'...'Almacén E' que no describen
# espacios físicos validados y el modelo Producto permite ubicaciones opcionales,
# no se crean ubicaciones fijas ficticias a menos que se suministren expresamente.
MAPEO_UBICACIONES_PROPUESTAS: Dict[str, str] = {}


def calcular_stock_minimo_regla(stock_actual: int, referencia: int = 5) -> int:
    """
    Regla aprobada para el piloto de Almacén A (referencia 5):
    - Si stock_actual > 5: mínimo = 5.
    - Si 1 <= stock_actual <= 5: mínimo = stock_actual - 1.
    - Si stock_actual == 0: mínimo = 0.
    """
    if stock_actual > referencia:
        return referencia
    elif stock_actual >= 1:
        return stock_actual - 1
    else:
        return 0



@dataclass
class FilaProcesada:
    numero_fila: int
    codigo: str
    nombre_original: str
    nombre_normalizado: str
    categoria_nombre: str
    ubicacion_original: str
    ubicacion_propuesta: str
    stock_inicial_csv: int
    entradas_csv: int
    salidas_csv: int
    stock_actual_csv: int
    estado_csv: str
    stock_minimo_sugerido: Optional[int] = None
    sugerencia_detalle: str = ""
    saldo_apertura_seleccionado: int = 0
    diferencia_apertura_explicacion: str = ""
    es_inconsistente_stock: bool = False
    inconsistencia_detalle: str = ""
    advertencias: List[str] = field(default_factory=list)
    errores: List[str] = field(default_factory=list)


@dataclass
class ReporteImportacion:
    total_filas: int = 0
    filas_validas: int = 0
    filas_con_advertencia: int = 0
    filas_con_error: int = 0
    categorias_requeridas: Set[str] = field(default_factory=set)
    ubicaciones_requeridas: Set[str] = field(default_factory=set)
    filas: List[FilaProcesada] = field(default_factory=list)
    discrepancias_origen: List[Dict[str, Any]] = field(default_factory=list)
    productos_creados: int = 0
    existencias_creadas: int = 0
    movimientos_creados: int = 0
    auditorias_creadas: int = 0
    es_idempotente_omitido: bool = False


import hashlib


class ImportadorInventarioCSV:
    """Procesador por fases para el archivo CSV de inventario."""

    def __init__(self, ruta_csv: str, almacen_destino_nombre: str = "A"):
        self.ruta_csv = ruta_csv
        self.almacen_destino_nombre = almacen_destino_nombre
        self._huella_contenido = None

    def calcular_huella_contenido(self) -> str:
        """Calcula el hash SHA-256 del contenido real del archivo CSV para idempotencia determinista."""
        if self._huella_contenido is None:
            if not os.path.exists(self.ruta_csv):
                raise FileNotFoundError(f"El archivo CSV no existe en la ruta: {self.ruta_csv}")
            hasher = hashlib.sha256()
            with open(self.ruta_csv, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            self._huella_contenido = hasher.hexdigest()
        return self._huella_contenido

    def analizar_y_validar(self) -> ReporteImportacion:
        """
        Fases 1 a 4: Análisis, normalización, validación y previsualización.
        ESTRICTAMENTE DE SOLO LECTURA. No realiza ninguna escritura en base de datos.
        """
        from inventario.models import Almacen, Categoria, Producto, Ubicacion

        if not os.path.exists(self.ruta_csv):
            raise FileNotFoundError(f"El archivo CSV no existe en la ruta: {self.ruta_csv}")

        reporte = ReporteImportacion()

        with open(self.ruta_csv, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValidationError("El archivo CSV está completamente vacío.")

            filas_raw = list(reader)

        reporte.total_filas = len(filas_raw)
        codigos_vistos = set()
        codigos_bd = set(Producto.objects.values_list("codigo_interno", flat=True))

        for idx, row in enumerate(filas_raw, start=2):
            if len(row) < 11:
                fp = FilaProcesada(
                    numero_fila=idx,
                    codigo="DESCONOCIDO",
                    nombre_original="",
                    nombre_normalizado="",
                    categoria_nombre="",
                    ubicacion_original="",
                    ubicacion_propuesta="",
                    stock_inicial_csv=0,
                    entradas_csv=0,
                    salidas_csv=0,
                    stock_actual_csv=0,
                    estado_csv="",
                    stock_minimo_sugerido=0,
                    errores=[f"Fila incompleta: esperaba 11 columnas, recibió {len(row)}."],
                )
                reporte.filas.append(fp)
                reporte.filas_con_error += 1
                continue

            alm_csv, cod_csv, nom_csv, cat_csv, tipo_csv, ubi_csv, s_ini_csv, ent_csv, sal_csv, s_act_csv, est_csv = [
                c.strip() for c in row[:11]
            ]

            errores_fila = []
            advertencias_fila = []

            # 1. Validación de código y detección de conflictos
            if not cod_csv:
                errores_fila.append("Código interno vacío.")
            elif cod_csv in codigos_vistos:
                errores_fila.append(f"Código duplicado dentro del mismo CSV: '{cod_csv}'.")
            elif cod_csv in codigos_bd:
                # Comprobar si es exactamente el mismo producto o si tiene datos divergentes
                prod_bd = Producto.objects.filter(codigo_interno=cod_csv).first()
                if prod_bd:
                    diferencias = []
                    nombre_esperado = CORRECCIONES_ORTOGRAFICAS_SEGURAS.get(nom_csv, nom_csv).strip()
                    if prod_bd.nombre.strip() != nombre_esperado and prod_bd.nombre.strip() != nom_csv:
                        diferencias.append(f"Nombre en BD '{prod_bd.nombre}' != en CSV '{nom_csv}'")
                    if prod_bd.categoria.nombre != cat_csv:
                        diferencias.append(f"Categoría en BD '{prod_bd.categoria.nombre}' != en CSV '{cat_csv}'")

                    if diferencias:
                        errores_fila.append(
                            f"CONFLICTO DE PRODUCTO: El código '{cod_csv}' ya existe con datos diferentes: "
                            + "; ".join(diferencias)
                        )
                    else:
                        advertencias_fila.append(
                            f"Producto idéntico ya registrado previamente en BD: '{cod_csv}'."
                        )
            codigos_vistos.add(cod_csv)

            # 2. Normalización de nombre
            nombre_limpio = CORRECCIONES_ORTOGRAFICAS_SEGURAS.get(nom_csv, nom_csv)
            nombre_limpio = re.sub(r"\s+", " ", nombre_limpio).strip()
            if nombre_limpio != nom_csv:
                advertencias_fila.append(f"Nombre normalizado: '{nom_csv}' -> '{nombre_limpio}'.")

            # 3. Categoría
            if not cat_csv:
                errores_fila.append("Categoría vacía.")
            else:
                reporte.categorias_requeridas.add(cat_csv)

            # 4. Ubicación
            ubi_propuesta = MAPEO_UBICACIONES_PROPUESTAS.get(ubi_csv, "")
            if ubi_propuesta:
                reporte.ubicaciones_requeridas.add(ubi_propuesta)
            elif ubi_csv:
                advertencias_fila.append(
                    f"Ubicación en CSV '{ubi_csv}' no resuelta a espacio físico verificado; "
                    "se importará sin asignación de ubicación física (campo opcional en Producto)."
                )

            # 5. Cantidades y verificación matemática
            try:
                s_ini = int(s_ini_csv)
                ent = int(ent_csv)
                sal = int(sal_csv)
                s_act = int(s_act_csv)
            except ValueError:
                errores_fila.append("Una o más columnas de stock no son enteros válidos.")
                s_ini = ent = sal = s_act = 0

            es_inconsistente = False
            detalle_inconsistencia = ""
            if s_ini + ent - sal != s_act:
                es_inconsistente = True
                detalle_inconsistencia = (
                    f"Discrepancia de origen: Stock Inicial ({s_ini}) + Entradas ({ent}) "
                    f"- Salidas ({sal}) = {s_ini + ent - sal} != Stock Actual declarado ({s_act}). "
                    f"Diferencia de {s_act - (s_ini + ent - sal)} unidades pendiente de justificación física."
                )
                advertencias_fila.append(detalle_inconsistencia)
                reporte.discrepancias_origen.append({
                    "fila": idx,
                    "codigo": cod_csv,
                    "nombre": nom_csv,
                    "stock_inicial": s_ini,
                    "entradas": ent,
                    "salidas": sal,
                    "calculado": s_ini + ent - sal,
                    "stock_actual": s_act,
                    "diferencia": s_act - (s_ini + ent - sal),
                    "estado_verificacion": "PENDIENTE_CONFIRMACION_ADMINISTRADOR",
                })

            # 6. Evaluación de stock mínimo según la regla aprobada del piloto (referencia 5)
            stock_min_sug = calcular_stock_minimo_regla(s_act, 5)
            sug_detalle = (
                f"Regla piloto Almacén A (referencia 5): stock actual {s_act} -> stock mínimo {stock_min_sug}."
            )

            fila_proc = FilaProcesada(
                numero_fila=idx,
                codigo=cod_csv,
                nombre_original=nom_csv,
                nombre_normalizado=nombre_limpio,
                categoria_nombre=cat_csv,
                ubicacion_original=ubi_csv,
                ubicacion_propuesta=ubi_propuesta,
                stock_inicial_csv=s_ini,
                entradas_csv=ent,
                salidas_csv=sal,
                stock_actual_csv=s_act,
                estado_csv=est_csv,
                stock_minimo_sugerido=stock_min_sug,
                sugerencia_detalle=sug_detalle,
                saldo_apertura_seleccionado=s_act,
                diferencia_apertura_explicacion=detalle_inconsistencia,
                es_inconsistente_stock=es_inconsistente,
                inconsistencia_detalle=detalle_inconsistencia,
                advertencias=advertencias_fila,
                errores=errores_fila,
            )
            reporte.filas.append(fila_proc)
            if errores_fila:
                reporte.filas_con_error += 1
            elif advertencias_fila:
                reporte.filas_con_advertencia += 1
            else:
                reporte.filas_validas += 1

        return reporte

    def calcular_huella_operacion(
        self,
        almacen_id: int,
        usar_stock_actual: bool,
        mapa_cantidades_iniciales_autorizadas: Optional[Dict[str, int]] = None,
        mapa_ubicaciones_fisicas_autorizadas: Optional[Dict[str, str]] = None,
        mapa_stock_minimo_confirmado: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Calcula una huella digital determinista y completa (SHA-256) de la operación de importación,
        combinando:
        1. Huella del contenido físico del archivo CSV.
        2. ID del almacén de destino.
        3. Criterio de saldo (usar_stock_actual).
        4. Mapa de cantidades autorizadas para discrepancias (ordenado).
        5. Mapa de ubicaciones físicas homologadas (ordenado).
        6. Mapa de stock mínimo confirmado (ordenado).
        """
        huella_archivo = self.calcular_huella_contenido()
        payload = [
            f"FILE:{huella_archivo}",
            f"ALM:{almacen_id}",
            f"STOCK_ACTUAL:{usar_stock_actual}",
        ]
        if mapa_cantidades_iniciales_autorizadas:
            items_cant = sorted(f"{k}:{v}" for k, v in mapa_cantidades_iniciales_autorizadas.items())
            payload.append("AUTORIZADAS:" + ",".join(items_cant))
        if mapa_ubicaciones_fisicas_autorizadas:
            items_ubi = sorted(f"{k}:{v}" for k, v in mapa_ubicaciones_fisicas_autorizadas.items())
            payload.append("UBICACIONES:" + ",".join(items_ubi))
        if mapa_stock_minimo_confirmado:
            items_min = sorted(f"{k}:{v}" for k, v in mapa_stock_minimo_confirmado.items())
            payload.append("MINIMOS:" + ",".join(items_min))

        texto_operacion = "|".join(payload)
        return hashlib.sha256(texto_operacion.encode("utf-8")).hexdigest()

    def ejecutar_persistencia(
        self,
        usuario,
        usar_stock_actual: bool = True,
        crear_categorias_si_faltan: bool = True,
        crear_ubicaciones_si_faltan: bool = True,
        mapa_cantidades_iniciales_autorizadas: Optional[Dict[str, int]] = None,
        mapa_ubicaciones_fisicas_autorizadas: Optional[Dict[str, str]] = None,
        mapa_stock_minimo_confirmado: Optional[Dict[str, int]] = None,
        exigir_stock_minimo_completo: bool = False,
        generar_movimientos_iniciales: bool = False,
        ubicacion_provisional_nombre: Optional[str] = "A1",
    ) -> ReporteImportacion:
        """
        Fases 5 a 7: Confirmación atómica y persistencia con trazabilidad e idempotencia fuerte.
        - Idempotente: Si el producto ya fue importado con la misma huella de operación, no se duplica.
        - Saldo de apertura: Aplica decisiones aprobadas para discrepancias (PAP-032=0, QUI-002=23).
        - Stock mínimo: Aplica la regla aprobada del piloto (referencia 5) editable posteriormente.
        - Ubicaciones: Si no se provee mapeo físico validado, se importan sin ubicación asignada (campo opcional).
        - Movimientos: CERO movimientos iniciales artificiales por directriz estricta del piloto.
        - Invalida cachés de HAKU al finalizar.
        """
        from django.db.models import Q
        from inventario.models import Almacen, Auditoria, Categoria, Existencia, Movimiento, Producto, Ubicacion
        from inventario.servicios_ubicacion import UbicacionService

        reporte = self.analizar_y_validar()
        if reporte.filas_con_error > 0:
            raise ValidationError(
                f"No se puede persistir el archivo CSV: se detectaron {reporte.filas_con_error} filas con errores bloqueantes."
            )

        # 1. Validación de discrepancias matemáticas del CSV:
        # Por defecto se adoptan las decisiones expresamente aprobadas para el piloto
        mapa_autorizadas = (
            mapa_cantidades_iniciales_autorizadas
            if mapa_cantidades_iniciales_autorizadas is not None
            else {"PAP-032": 0, "QUI-002": 23}
        )
        filas_discrepantes_no_autorizadas = []
        for d in reporte.discrepancias_origen:
            codigo = d["codigo"]
            if codigo not in mapa_autorizadas:
                filas_discrepantes_no_autorizadas.append(
                    f"{codigo} ({d['nombre']}): Calculado={d['calculado']} vs Declarado={d['stock_actual']}"
                )

        if filas_discrepantes_no_autorizadas:
            raise ValidationError(
                "PERSISTENCIA BLOQUEADA: Existen productos con discrepancias en el CSV que no cuentan con "
                "una cantidad inicial autorizada explícitamente por el administrador: "
                + "; ".join(filas_discrepantes_no_autorizadas)
                + ". Especifique 'mapa_cantidades_iniciales_autorizadas' para continuar."
            )

        # 2. Validación de stock mínimo obligatorio
        if exigir_stock_minimo_completo:
            if not mapa_stock_minimo_confirmado:
                raise ValidationError(
                    "PERSISTENCIA BLOQUEADA: La política exige elección expresa del stock mínimo para cada producto. "
                    "No se proporcionó 'mapa_stock_minimo_confirmado'."
                )
            codigos_faltantes_minimo = [
                f.codigo for f in reporte.filas if f.codigo not in mapa_stock_minimo_confirmado
            ]
            if codigos_faltantes_minimo:
                raise ValidationError(
                    f"PERSISTENCIA BLOQUEADA: Faltan elecciones de stock mínimo confirmadas para {len(codigos_faltantes_minimo)} "
                    f"productos (ej: {', '.join(codigos_faltantes_minimo[:5])})."
                )

        mapa_ubis_conf = mapa_ubicaciones_fisicas_autorizadas or {}

        almacen = Almacen.objects.filter(
            Q(nombre=self.almacen_destino_nombre) | Q(nombre=f"Almacén {self.almacen_destino_nombre}"),
            activo=True,
            eliminado=False,
        ).first()
        if not almacen:
            raise ValidationError(f"El almacén destino '{self.almacen_destino_nombre}' no existe o está inactivo.")

        # Calcular huella determinista de toda la operación
        huella_op = self.calcular_huella_operacion(
            almacen_id=almacen.id,
            usar_stock_actual=usar_stock_actual,
            mapa_cantidades_iniciales_autorizadas=mapa_autorizadas,
            mapa_ubicaciones_fisicas_autorizadas=mapa_ubis_conf,
            mapa_stock_minimo_confirmado=mapa_stock_minimo_confirmado,
        )
        prefijo_huella = huella_op[:16]

        with transaction.atomic():
            # 1. Asegurar categorías requeridas
            mapa_categorias = {}
            for cat_nombre in reporte.categorias_requeridas:
                cat_inst = Categoria.objects.filter(nombre=cat_nombre).first()
                if not cat_inst:
                    if not crear_categorias_si_faltan:
                        raise ValidationError(f"Categoría requerida '{cat_nombre}' no existe y no está autorizado crearla.")
                    cat_inst = Categoria.objects.create(
                        nombre=cat_nombre,
                        almacen_principal=almacen,
                        activo=True,
                        eliminado=False,
                    )
                mapa_categorias[cat_nombre] = cat_inst

            # 2. Asegurar ubicaciones requeridas (solo si se provee mapeo físico validado)
            mapa_ubicaciones = {}
            if mapa_ubis_conf:
                for f in reporte.filas:
                    ubi_nombre_final = mapa_ubis_conf.get(f.ubicacion_original, "")
                    if ubi_nombre_final and ubi_nombre_final not in mapa_ubicaciones:
                        ubi_inst, _ = UbicacionService.obtener_o_crear_ubicacion(
                            almacen=almacen,
                            nombre=ubi_nombre_final,
                            descripcion=f"Ubicación en Almacén {almacen.nombre}",
                        )
                        mapa_ubicaciones[ubi_nombre_final] = ubi_inst

            # 3. Procesar cada fila con verificación de compatibilidad e idempotencia
            for f in reporte.filas:
                clave_idemp = f"apertura-csv-op{prefijo_huella}-alm{almacen.id}-{f.codigo}"[:100]

                # Determinación estricta de la cantidad inicial
                if f.codigo in mapa_autorizadas:
                    saldo_a_cargar = int(mapa_autorizadas[f.codigo])
                else:
                    saldo_a_cargar = f.stock_actual_csv if usar_stock_actual else f.stock_inicial_csv

                # Asignación controlada de stock mínimo: debe provenir de elección expresa o regla aprobada (ref 5)
                if mapa_stock_minimo_confirmado and f.codigo in mapa_stock_minimo_confirmado:
                    stock_min_final = int(mapa_stock_minimo_confirmado[f.codigo])
                else:
                    stock_min_final = calcular_stock_minimo_regla(saldo_a_cargar, 5)

                cat_obj = mapa_categorias[f.categoria_nombre]
                ubi_nombre_final = mapa_ubis_conf.get(f.ubicacion_original, "")
                ubi_obj = mapa_ubicaciones.get(ubi_nombre_final) if ubi_nombre_final else None
                if not ubi_obj and ubicacion_provisional_nombre:
                    ubi_obj, _ = UbicacionService.obtener_o_crear_ubicacion(
                        almacen=almacen,
                        nombre=ubicacion_provisional_nombre,
                        descripcion=f"Ubicación física {ubicacion_provisional_nombre} en {almacen.nombre}",
                    )

                prod_existente = Producto.objects.filter(codigo_interno=f.codigo).first()
                if prod_existente:
                    aud_origen = Auditoria.objects.filter(
                        tipo_entidad="Producto",
                        id_entidad=str(prod_existente.id),
                        valores_nuevos__clave_operacion=clave_idemp,
                    ).first()

                    if aud_origen:
                        # Repetición legítima idéntica: omitir sin duplicar, pero asegurar ubicación
                        if ubi_obj:
                            prod_existente.ubicaciones.add(ubi_obj)
                        continue
                    else:
                        # Si el producto ya existe idéntico en categoría y nombre, asegurar existencia sin duplicar
                        if prod_existente.nombre == f.nombre_normalizado and prod_existente.categoria == cat_obj:
                            existencia, _ = Existencia.objects.get_or_create(
                                producto=prod_existente,
                                almacen=almacen,
                                defaults={"stock_disponible": saldo_a_cargar, "stock_en_prestamo": 0},
                            )
                            if ubi_obj:
                                prod_existente.ubicaciones.add(ubi_obj)
                            continue
                        raise ValidationError(
                            f"CONFLICTO DE IDEMPOTENCIA: El producto '{f.codigo}' ya existe en el sistema "
                            f"pero no corresponde a esta misma operación autorizada (clave esperada: '{clave_idemp}'). "
                            f"Operación bloqueada para impedir corrupción o duplicidad silenciosa."
                        )

                # Crear producto
                producto = Producto.objects.create(
                    codigo_interno=f.codigo,
                    nombre=f.nombre_normalizado,
                    categoria=cat_obj,
                    stock_inicial=saldo_a_cargar,
                    stock_minimo=stock_min_final,
                    activo=True,
                    archivado=False,
                    descripcion=f"Importado de CSV {os.path.basename(self.ruta_csv)}. Estado original: {f.estado_csv}.",
                )
                reporte.productos_creados += 1

                # Asociar ubicación solo si fue resuelta
                if ubi_obj:
                    producto.ubicaciones.add(ubi_obj)

                # Crear existencia en Almacén destino con saldo inicial puro (sin retención en préstamo)
                existencia, _ = Existencia.objects.get_or_create(
                    producto=producto,
                    almacen=almacen,
                    defaults={"stock_disponible": saldo_a_cargar, "stock_en_prestamo": 0},
                )
                reporte.existencias_creadas += 1

                # Generar movimiento inicial trazable SOLO si se solicita explícitamente (por defecto False)
                if generar_movimientos_iniciales and saldo_a_cargar > 0:
                    mov = Movimiento.objects.create(
                        clave_idempotencia=clave_idemp,
                        tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL,
                        producto=producto,
                        cantidad=saldo_a_cargar,
                        almacen_origen=almacen,
                        ubicacion_origen=ubi_obj,
                        usuario=usuario,
                        motivo=f"Establecimiento de saldo de apertura por importación de catálogo CSV ({f.codigo}). Huella completa: {huella_op}.",
                        stock_disponible_anterior=0,
                        stock_disponible_posterior=saldo_a_cargar,
                        via_haku=False,
                    )
                    reporte.movimientos_creados += 1

                # Registrar auditoría obligatoria vinculada a la clave de operación y huella SHA-256 completa
                Auditoria.objects.create(
                    usuario=usuario,
                    accion="APERTURA_INVENTARIO_CSV",
                    tipo_entidad="Producto",
                    id_entidad=str(producto.id),
                    valores_anteriores={"stock_disponible": 0},
                    valores_nuevos={
                        "stock_disponible": saldo_a_cargar,
                        "stock_minimo": stock_min_final,
                        "clave_operacion": clave_idemp,
                        "huella_operacion_completa": huella_op,
                    },
                    motivo=f"Saldo de apertura inicial ({clave_idemp}): {f.nombre_normalizado} ({f.codigo}) con {saldo_a_cargar} unidades (mínimo {stock_min_final}). Cero movimientos históricos.",
                    resultado="OK",
                    via_haku=False,
                )
                reporte.auditorias_creadas += 1

        # 4. Invalidar cachés de HAKU para disponibilizar inmediatamente el catálogo y stock
        try:
            from integraciones.haku.cache import HakuCacheManager
            HakuCacheManager.invalidar_catalogo()
            HakuCacheManager.invalidar_stock()
            HakuCacheManager.invalidar_ubicaciones()
            HakuCacheManager.invalidar_faltantes()
        except Exception as exc:
            logger.warning("No se pudo invalidar caché de Haku automáticamente: %s", exc)
        return reporte
