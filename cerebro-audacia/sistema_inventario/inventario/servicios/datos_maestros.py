"""Servicios de lógica de negocio para datos maestros de inventario.

Gestiona Almacenes, Ubicaciones, Categorías y Productos con validaciones,
autorización por rol/almacén y registro inmutable de auditoría.
"""

from typing import Any, Dict, List, Optional
import logging
import time
from django.core.exceptions import PermissionDenied, ValidationError
from django.db import connection, models, transaction

logger = logging.getLogger(__name__)

from inventario.auditoria import registrar_auditoria
from inventario.models import (
    Almacen,
    Auditoria,
    Categoria,
    ConteoFisico,
    Existencia,
    Movimiento,
    Prestamo,
    Producto,
    TipoCategoria,
    Ubicacion,
)
from inventario.servicios import NecesitaTraslado


# ─────────────────────────────────────────────────────────────────────────────
# 1. ALMACENES
# ─────────────────────────────────────────────────────────────────────────────

def crear_almacen(
    *,
    nombre: str,
    descripcion: str = "",
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Almacen:
    """Crea un nuevo almacén validando permisos de administración y unicidad."""
    if not getattr(usuario, "es_administrador", False):
        raise PermissionDenied("Solo los Administradores o el Superusuario pueden crear almacenes.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre del almacén es obligatorio."})

    alm_existente = Almacen.objects.filter(nombre__iexact=nombre).first()
    if alm_existente:
        if not alm_existente.eliminado:
            raise ValidationError({"nombre": f"Ya existe un almacén activo con el nombre '{nombre}'."})
        else:
            with transaction.atomic():
                alm_existente.eliminado = False
                alm_existente.activo = True
                alm_existente.descripcion = (descripcion or "").strip()
                alm_existente.save()
                registrar_auditoria(
                    usuario=usuario,
                    accion="CREAR_ALMACEN",
                    tipo_entidad="Almacen",
                    id_entidad=str(alm_existente.id),
                    valores_anteriores={"eliminado": True, "activo": False},
                    valores_nuevos={"eliminado": False, "activo": True, "nombre": alm_existente.nombre},
                    motivo=f"Reactivación de almacén '{alm_existente.nombre}' previamente eliminado",
                    resultado="OK",
                    via_haku=False,
                    ip_origen=ip_origen,
                )
            return alm_existente

    with transaction.atomic():
        almacen = Almacen(
            nombre=nombre,
            descripcion=(descripcion or "").strip(),
            activo=True,
            eliminado=False,
        )
        almacen.full_clean()
        almacen.save()

        # §5.2 Creación automática de categoría por defecto para el nuevo almacén
        nombre_cat_defecto = f"General - {almacen.nombre}"
        if not Categoria.objects.filter(almacen_principal=almacen, eliminado=False).exists():
            if not Categoria.objects.filter(nombre=nombre_cat_defecto, eliminado=False).exists():
                cat_defecto = Categoria(
                    nombre=nombre_cat_defecto,
                    almacen_principal=almacen,
                    activo=True,
                    eliminado=False,
                )
                cat_defecto.full_clean()
                cat_defecto.save()

        registrar_auditoria(
            usuario=usuario,
            accion="CREAR_ALMACEN",
            tipo_entidad="Almacen",
            id_entidad=str(almacen.id),
            valores_anteriores=None,
            valores_nuevos={
                "nombre": almacen.nombre,
                "descripcion": almacen.descripcion,
                "activo": True,
            },
            motivo=f"Creación de almacén '{almacen.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return almacen


def actualizar_almacen(
    *,
    almacen: Almacen,
    nombre: str,
    descripcion: str,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Almacen:
    """Actualiza datos básicos de un almacén existente registrando auditoría."""
    if not getattr(usuario, "es_administrador", False):
        raise PermissionDenied("Solo los Administradores o el Superusuario pueden editar almacenes.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre del almacén no puede estar vacío."})

    almacen_db = Almacen.objects.get(pk=almacen.pk)
    valores_anteriores = {
        "nombre": almacen_db.nombre,
        "descripcion": almacen_db.descripcion,
    }

    with transaction.atomic():
        almacen_db.nombre = nombre
        almacen_db.descripcion = (descripcion or "").strip()
        almacen_db.full_clean()
        almacen_db.save()

        registrar_auditoria(
            usuario=usuario,
            accion="EDITAR_ALMACEN",
            tipo_entidad="Almacen",
            id_entidad=str(almacen_db.id),
            valores_anteriores=valores_anteriores,
            valores_nuevos={
                "nombre": almacen_db.nombre,
                "descripcion": almacen_db.descripcion,
            },
            motivo=f"Actualización de almacén '{almacen_db.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return almacen_db


def reactivar_almacen(
    *,
    almacen: Almacen,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Almacen:
    """Reactiva un almacén previamente desactivado."""
    if not getattr(usuario, "es_administrador", False):
        raise PermissionDenied("Solo los Administradores o el Superusuario pueden reactivar almacenes.")

    with transaction.atomic():
        almacen.activo = True
        almacen.save(update_fields=["activo"])

        registrar_auditoria(
            usuario=usuario,
            accion="REACTIVAR_ALMACEN",
            tipo_entidad="Almacen",
            id_entidad=str(almacen.id),
            valores_anteriores={"activo": False},
            valores_nuevos={"activo": True},
            motivo=f"Reactivación de almacén '{almacen.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return almacen


def eliminar_almacen(
    *,
    almacen: Almacen,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> None:
    """Elimina lógicamente un almacén si no tiene stock disponible ni ubicaciones con artículos."""
    if not getattr(usuario, "es_administrador", False):
        raise PermissionDenied("Solo los Administradores o el Superusuario pueden eliminar almacenes.")

    tiene_stock = almacen.existencias.filter(
        models.Q(stock_disponible__gt=0) | models.Q(stock_en_prestamo__gt=0)
    ).exists() if hasattr(almacen, "existencias") else False
    tiene_prestamos = almacen.prestamos.filter(
        estado__in=[Prestamo.Estado.ABIERTO, Prestamo.Estado.DEVUELTO_PARCIAL]
    ).exists() if hasattr(almacen, "prestamos") else False
    if tiene_stock or tiene_prestamos:
        raise NecesitaTraslado(
            f"El almacén '{almacen.nombre}' tiene existencias o préstamos activos. Debe trasladar su inventario antes de eliminarlo.",
            entidad=almacen,
        )

    with transaction.atomic():
        almacen.activo = False
        almacen.eliminado = True
        almacen.save(update_fields=["activo", "eliminado"])

        registrar_auditoria(
            usuario=usuario,
            accion="ELIMINAR_ALMACEN",
            tipo_entidad="Almacen",
            id_entidad=str(almacen.id),
            valores_anteriores={"activo": True, "eliminado": False},
            valores_nuevos={"activo": False, "eliminado": True},
            motivo=f"Eliminación lógica de almacén '{almacen.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. UBICACIONES
# ─────────────────────────────────────────────────────────────────────────────

def crear_ubicacion(
    *,
    nombre: str,
    almacen: Almacen,
    descripcion: str = "",
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Ubicacion:
    """Crea una nueva ubicación dentro de un almacén respetando permisos."""
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para crear ubicaciones.")

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_ubicaciones"):
            raise PermissionDenied("No tienes permisos para registrar ubicaciones.")
        if not usuario.tiene_acceso_almacen(almacen):
            raise PermissionDenied("Solo puedes registrar ubicaciones en tus almacenes asignados.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre de la ubicación es obligatorio."})

    if Ubicacion.objects.filter(almacen=almacen, nombre__iexact=nombre, eliminado=False).exists():
        raise ValidationError({"nombre": f"Ya existe una ubicación activa con el nombre '{nombre}' en este almacén."})

    ubicacion_del = Ubicacion.objects.filter(almacen=almacen, nombre__iexact=nombre, eliminado=True).first()
    if ubicacion_del:
        with transaction.atomic():
            ubicacion_del.eliminado = False
            ubicacion_del.activo = True
            ubicacion_del.descripcion = (descripcion or "").strip()
            ubicacion_del.save(update_fields=["eliminado", "activo", "descripcion"])

            registrar_auditoria(
                usuario=usuario,
                accion="CREAR_UBICACION",
                tipo_entidad="Ubicacion",
                id_entidad=str(ubicacion_del.id),
                valores_anteriores={"eliminado": True, "activo": False},
                valores_nuevos={
                    "nombre": ubicacion_del.nombre,
                    "almacen_id": almacen.id,
                    "almacen_nombre": almacen.nombre,
                    "descripcion": ubicacion_del.descripcion,
                },
                motivo=f"Recreación/reactivación de ubicación '{ubicacion_del.nombre}' en almacén '{almacen.nombre}'",
                resultado="OK",
                via_haku=False,
                ip_origen=ip_origen,
            )
        return ubicacion_del

    with transaction.atomic():
        ubicacion = Ubicacion(
            nombre=nombre,
            almacen=almacen,
            descripcion=(descripcion or "").strip(),
            activo=True,
            eliminado=False,
        )
        ubicacion.full_clean()
        ubicacion.save()

        registrar_auditoria(
            usuario=usuario,
            accion="CREAR_UBICACION",
            tipo_entidad="Ubicacion",
            id_entidad=str(ubicacion.id),
            valores_anteriores=None,
            valores_nuevos={
                "nombre": ubicacion.nombre,
                "almacen_id": almacen.id,
                "almacen_nombre": almacen.nombre,
                "descripcion": ubicacion.descripcion,
            },
            motivo=f"Creación de ubicación '{ubicacion.nombre}' en almacén '{almacen.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return ubicacion


def actualizar_ubicacion(
    *,
    ubicacion: Ubicacion,
    nombre: str,
    descripcion: str,
    almacen: Optional[Almacen] = None,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Ubicacion:
    """Actualiza una ubicación conservando el historial del nombre anterior en auditoría.
    
    Permite cambiar de almacén si la ubicación no tiene productos asignados.
    """
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación.")

    ubicacion_db = Ubicacion.objects.select_related("almacen").get(pk=ubicacion.pk)

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_ubicaciones"):
            raise PermissionDenied("No tienes permisos para modificar ubicaciones.")
        if not usuario.tiene_acceso_almacen(ubicacion_db.almacen):
            raise PermissionDenied("No tienes permisos para modificar ubicaciones de otros almacenes.")
        if almacen and not usuario.tiene_acceso_almacen(almacen):
            raise PermissionDenied("No tienes permisos para asignar la ubicación al almacén seleccionado.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre de la ubicación no puede estar vacío."})

    almacen_efectivo = almacen or ubicacion_db.almacen
    if almacen and almacen.id != ubicacion_db.almacen_id:
        if Producto.objects.filter(ubicacion=ubicacion_db, archivado=False).exists():
            raise NecesitaTraslado(
                f"No se puede cambiar el almacén de la ubicación '{ubicacion_db.nombre}' porque tiene productos asignados. "
                f"Reasigne o traslade los productos primero."
            )

    # Verificar unicidad en el almacén destino/efectivo
    qs_unique = Ubicacion.objects.filter(almacen=almacen_efectivo, nombre__iexact=nombre, eliminado=False).exclude(pk=ubicacion_db.pk)
    if qs_unique.exists():
        raise ValidationError({"nombre": f"Ya existe una ubicación con el nombre '{nombre}' en el almacén '{almacen_efectivo.nombre}'."})

    valores_anteriores = {
        "nombre": ubicacion_db.nombre,
        "descripcion": ubicacion_db.descripcion,
        "almacen_id": ubicacion_db.almacen_id,
        "almacen_nombre": ubicacion_db.almacen.nombre,
    }

    with transaction.atomic():
        ubicacion_db.nombre = nombre
        ubicacion_db.descripcion = (descripcion or "").strip()
        if almacen and almacen.id != ubicacion_db.almacen_id:
            ubicacion_db.almacen = almacen
        ubicacion_db.full_clean()
        ubicacion_db.save()

        registrar_auditoria(
            usuario=usuario,
            accion="EDITAR_UBICACION",
            tipo_entidad="Ubicacion",
            id_entidad=str(ubicacion_db.id),
            valores_anteriores=valores_anteriores,
            valores_nuevos={
                "nombre": ubicacion_db.nombre,
                "descripcion": ubicacion_db.descripcion,
                "almacen_id": ubicacion_db.almacen_id,
                "almacen_nombre": ubicacion_db.almacen.nombre,
            },
            motivo=f"Actualización de ubicación '{ubicacion_db.nombre}' (almacén '{ubicacion_db.almacen.nombre}')",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return ubicacion_db


def reactivar_ubicacion(
    *,
    ubicacion: Ubicacion,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Ubicacion:
    """Reactiva una ubicación física previamente inhabilitada."""
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación.")

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_ubicaciones"):
            raise PermissionDenied("No tienes permisos para reactivar ubicaciones.")
        if not usuario.tiene_acceso_almacen(ubicacion.almacen):
            raise PermissionDenied("No tienes permisos para reactivar ubicaciones de otros almacenes.")

    with transaction.atomic():
        ubicacion.activo = True
        ubicacion.save(update_fields=["activo"])

        registrar_auditoria(
            usuario=usuario,
            accion="REACTIVAR_UBICACION",
            tipo_entidad="Ubicacion",
            id_entidad=str(ubicacion.id),
            valores_anteriores={"activo": False},
            valores_nuevos={"activo": True},
            motivo=f"Reactivación de ubicación '{ubicacion.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )
    return ubicacion


def eliminar_ubicacion(
    *,
    ubicacion: Ubicacion,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> None:
    """Elimina lógicamente una ubicación física si no tiene productos asignados."""
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación.")

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_ubicaciones"):
            raise PermissionDenied("No tienes permisos para eliminar ubicaciones.")
        if not usuario.tiene_acceso_almacen(ubicacion.almacen):
            raise PermissionDenied("No tienes permisos para eliminar ubicaciones de otros almacenes.")

    if Producto.objects.filter(ubicacion=ubicacion, archivado=False).exists():
        raise NecesitaTraslado(
            f"La ubicación '{ubicacion.nombre}' tiene productos asignados. Reasígnelos antes de eliminarla.",
            entidad=ubicacion,
        )

    with transaction.atomic():
        ubicacion.activo = False
        ubicacion.eliminado = True
        ubicacion.save(update_fields=["activo", "eliminado"])

        registrar_auditoria(
            usuario=usuario,
            accion="ELIMINAR_UBICACION",
            tipo_entidad="Ubicacion",
            id_entidad=str(ubicacion.id),
            valores_anteriores={"activo": True, "eliminado": False},
            valores_nuevos={"activo": False, "eliminado": True},
            motivo=f"Eliminación lógica de ubicación '{ubicacion.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. CATEGORÍAS
# ─────────────────────────────────────────────────────────────────────────────

def crear_categoria(
    *,
    nombre: str,
    almacen_principal: Almacen,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Categoria:
    """Crea una nueva categoría asignada a un almacén principal obligatorio.
    
    Valida permisos por rol: usuarios normales solo pueden crear en su almacén.
    """
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para crear categorías.")

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_categorias"):
            raise PermissionDenied("No tienes permisos para registrar categorías.")
        if not usuario.tiene_acceso_almacen(almacen_principal):
            raise PermissionDenied("Solo puedes registrar categorías en tus almacenes asignados.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre de la categoría es obligatorio."})

    if Categoria.objects.filter(nombre__iexact=nombre, eliminado=False).exists():
        raise ValidationError({"nombre": f"Ya existe una categoría activa con el nombre '{nombre}'."})

    categoria_del = Categoria.objects.filter(nombre__iexact=nombre, eliminado=True).first()
    if categoria_del:
        with transaction.atomic():
            categoria_del.eliminado = False
            categoria_del.activo = True
            categoria_del.almacen_principal = almacen_principal
            categoria_del.save(update_fields=["eliminado", "activo", "almacen_principal"])

            registrar_auditoria(
                usuario=usuario,
                accion="CREAR_CATEGORIA",
                tipo_entidad="Categoria",
                id_entidad=str(categoria_del.id),
                valores_anteriores={"eliminado": True, "activo": False},
                valores_nuevos={
                    "nombre": categoria_del.nombre,
                    "almacen_principal_id": almacen_principal.id,
                    "almacen_principal_nombre": almacen_principal.nombre,
                    "prefijo_codigo": categoria_del.prefijo_codigo,
                },
                motivo=f"Recreación/reactivación de categoría '{categoria_del.nombre}' con almacén '{almacen_principal.nombre}'",
                resultado="OK",
                via_haku=False,
                ip_origen=ip_origen,
            )
        return categoria_del

    with transaction.atomic():
        categoria = Categoria(
            nombre=nombre,
            almacen_principal=almacen_principal,
            activo=True,
            eliminado=False,
        )
        categoria.full_clean()
        categoria.save()

        registrar_auditoria(
            usuario=usuario,
            accion="CREAR_CATEGORIA",
            tipo_entidad="Categoria",
            id_entidad=str(categoria.id),
            valores_anteriores=None,
            valores_nuevos={
                "nombre": categoria.nombre,
                "almacen_principal_id": almacen_principal.id,
                "almacen_principal_nombre": almacen_principal.nombre,
                "prefijo_codigo": categoria.prefijo_codigo,
            },
            motivo=f"Creación de categoría '{categoria.nombre}' con almacén principal '{almacen_principal.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return categoria


def actualizar_categoria(
    *,
    categoria: Categoria,
    nombre: str,
    almacen_principal: Almacen,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Categoria:
    """Actualiza una categoría existente.
    
    Si el almacén principal cambia y la categoría ya cuenta con productos,
    lanza NecesitaTraslado para exigir el flujo de traslado estructural (§7.3).
    """
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para modificar categorías.")

    categoria_db = Categoria.objects.select_related("almacen_principal").get(pk=categoria.pk)

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_categorias"):
            raise PermissionDenied("No tienes permisos para modificar categorías.")
        if not usuario.tiene_acceso_almacen(categoria_db.almacen_principal):
            raise PermissionDenied("No tienes permisos para modificar categorías de otros almacenes.")
        if not usuario.tiene_acceso_almacen(almacen_principal):
            raise PermissionDenied("Solo puedes asignar almacenes a los que tienes acceso.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre de la categoría no puede estar vacío."})

    # Verificar si intenta cambiar el almacén principal respecto al guardado en base de datos
    if categoria_db.almacen_principal_id != almacen_principal.id:
        # Si tiene productos asociados no archivados, no se puede cambiar directamente (§7.3)
        if categoria_db.productos.filter(archivado=False).exists():
            raise NecesitaTraslado(
                f"La categoría '{categoria_db.nombre}' tiene productos asociados. "
                f"Cambiar el almacén principal a '{almacen_principal.nombre}' requiere trasladar los productos.",
                entidad=categoria_db,
                destino=almacen_principal,
            )

    valores_anteriores = {
        "nombre": categoria_db.nombre,
        "almacen_principal_id": categoria_db.almacen_principal_id,
        "almacen_principal_nombre": categoria_db.almacen_principal.nombre,
    }

    with transaction.atomic():
        categoria_db.nombre = nombre
        categoria_db.almacen_principal = almacen_principal
        categoria_db.full_clean()
        categoria_db.save()

        registrar_auditoria(
            usuario=usuario,
            accion="EDITAR_CATEGORIA",
            tipo_entidad="Categoria",
            id_entidad=str(categoria_db.id),
            valores_anteriores=valores_anteriores,
            valores_nuevos={
                "nombre": categoria_db.nombre,
                "almacen_principal_id": almacen_principal.id,
                "almacen_principal_nombre": almacen_principal.nombre,
            },
            motivo=f"Actualización de categoría '{categoria_db.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return categoria_db


def crear_tipo_categoria(
    *,
    nombre: str,
    categoria: Categoria,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> TipoCategoria:
    """Crea un tipo opcional ligado a una categoría respetando permisos de almacén."""
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para crear tipos de categoría.")

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_categorias"):
            raise PermissionDenied("No tienes permisos para registrar tipos de categoría.")
        if not usuario.tiene_acceso_almacen(categoria.almacen_principal):
            raise PermissionDenied("Solo puedes registrar tipos en categorías de tus almacenes asignados.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre del tipo es obligatorio."})

    tipo_existente = TipoCategoria.objects.filter(categoria=categoria, nombre__iexact=nombre).first()
    if tipo_existente:
        if tipo_existente.activo:
            raise ValidationError({"nombre": f"Ya existe el tipo activo '{nombre}' en la categoría '{categoria.nombre}'."})
        else:
            with transaction.atomic():
                tipo_existente.activo = True
                tipo_existente.save(update_fields=["activo"])
                registrar_auditoria(
                    usuario=usuario,
                    accion="CREAR_TIPO_CATEGORIA",
                    tipo_entidad="TipoCategoria",
                    id_entidad=str(tipo_existente.id),
                    valores_anteriores={"activo": False},
                    valores_nuevos={"activo": True, "nombre": tipo_existente.nombre},
                    motivo=f"Reactivación de tipo '{tipo_existente.nombre}' en categoría '{categoria.nombre}'",
                    resultado="OK",
                    via_haku=False,
                    ip_origen=ip_origen,
                )
            return tipo_existente

    with transaction.atomic():
        tipo = TipoCategoria(
            nombre=nombre,
            categoria=categoria,
            activo=True,
        )
        tipo.full_clean()
        tipo.save()

        registrar_auditoria(
            usuario=usuario,
            accion="CREAR_TIPO_CATEGORIA",
            tipo_entidad="TipoCategoria",
            id_entidad=str(tipo.id),
            valores_anteriores=None,
            valores_nuevos={
                "nombre": tipo.nombre,
                "categoria_id": categoria.id,
                "categoria_nombre": categoria.nombre,
            },
            motivo=f"Creación de tipo '{tipo.nombre}' en categoría '{categoria.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return tipo


def actualizar_tipo_categoria(
    *,
    tipo: TipoCategoria,
    nombre: str,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> TipoCategoria:
    """Actualiza el nombre de un tipo de categoría conservando los productos ligados."""
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para editar tipos de categoría.")

    tipo_db = TipoCategoria.objects.select_related("categoria", "categoria__almacen_principal").get(pk=tipo.pk)

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_categorias"):
            raise PermissionDenied("No tienes permisos para modificar tipos de categoría.")
        if not usuario.tiene_acceso_almacen(tipo_db.categoria.almacen_principal):
            raise PermissionDenied("No tienes permisos para modificar tipos de otros almacenes.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre del tipo no puede estar vacío."})

    if TipoCategoria.objects.filter(categoria=tipo_db.categoria, nombre__iexact=nombre).exclude(pk=tipo_db.pk).exists():
        raise ValidationError({"nombre": f"Ya existe otro tipo con el nombre '{nombre}' en esta categoría."})

    valores_anteriores = {
        "nombre": tipo_db.nombre,
        "categoria_id": tipo_db.categoria_id,
    }

    with transaction.atomic():
        tipo_db.nombre = nombre
        tipo_db.full_clean()
        tipo_db.save()

        registrar_auditoria(
            usuario=usuario,
            accion="EDITAR_TIPO_CATEGORIA",
            tipo_entidad="TipoCategoria",
            id_entidad=str(tipo_db.id),
            valores_anteriores=valores_anteriores,
            valores_nuevos={
                "nombre": tipo_db.nombre,
                "categoria_id": tipo_db.categoria_id,
            },
            motivo=f"Actualización de tipo '{tipo_db.nombre}' (categoría '{tipo_db.categoria.nombre}')",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return tipo_db


def eliminar_tipo_categoria(
    *,
    tipo: TipoCategoria,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> None:
    """Elimina un tipo de categoría (los productos ligados quedan con tipo=None)."""
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para eliminar tipos de categoría.")

    tipo_db = TipoCategoria.objects.select_related("categoria").get(pk=tipo.pk)

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_categorias"):
            raise PermissionDenied("No tienes permisos para eliminar tipos de categoría.")
        if not usuario.tiene_acceso_almacen(tipo_db.categoria.almacen_principal):
            raise PermissionDenied("No tienes permisos para eliminar tipos de otros almacenes.")

    valores_anteriores = {
        "nombre": tipo_db.nombre,
        "categoria_id": tipo_db.categoria_id,
        "categoria_nombre": tipo_db.categoria.nombre,
    }
    tipo_id = str(tipo_db.id)
    motivo = f"Eliminación de tipo '{tipo_db.nombre}' en categoría '{tipo_db.categoria.nombre}'"

    with transaction.atomic():
        tipo_db.delete()

        registrar_auditoria(
            usuario=usuario,
            accion="ELIMINAR_TIPO_CATEGORIA",
            tipo_entidad="TipoCategoria",
            id_entidad=tipo_id,
            valores_anteriores=valores_anteriores,
            valores_nuevos=None,
            motivo=motivo,
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )


def sincronizar_tipos_categoria(
    *,
    categoria: Categoria,
    nombres_tipos: List[str],
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> List[TipoCategoria]:
    """Crea o mantiene los tipos indicados para una categoría (sin borrar existentes si no se pide)."""
    creados = []
    for raw in nombres_tipos:
        limpio = (raw or "").strip()
        if not limpio:
            continue
        existente = TipoCategoria.objects.filter(categoria=categoria, nombre__iexact=limpio).first()
        if existente:
            if not existente.activo:
                existente.activo = True
                existente.save(update_fields=["activo"])
            creados.append(existente)
        else:
            creados.append(
                crear_tipo_categoria(
                    nombre=limpio,
                    categoria=categoria,
                    usuario=usuario,
                    ip_origen=ip_origen,
                )
            )
    return creados


def reactivar_categoria(
    *,
    categoria: Categoria,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Categoria:
    """Reactiva una categoría previamente inhabilitada."""
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación.")

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_categorias"):
            raise PermissionDenied("No tienes permisos para reactivar categorías.")
        if not usuario.tiene_acceso_almacen(categoria.almacen_principal):
            raise PermissionDenied("No tienes permisos para reactivar categorías de otros almacenes.")

    with transaction.atomic():
        categoria.activo = True
        categoria.save(update_fields=["activo"])

        registrar_auditoria(
            usuario=usuario,
            accion="REACTIVAR_CATEGORIA",
            tipo_entidad="Categoria",
            id_entidad=str(categoria.id),
            valores_anteriores={"activo": False},
            valores_nuevos={"activo": True},
            motivo=f"Reactivación de categoría '{categoria.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )
    return categoria


def eliminar_categoria(
    *,
    categoria: Categoria,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> None:
    """Elimina lógicamente una categoría si no tiene productos activos."""
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación.")

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("infraestructura.gestionar_categorias"):
            raise PermissionDenied("No tienes permisos para eliminar categorías.")
        if not usuario.tiene_acceso_almacen(categoria.almacen_principal):
            raise PermissionDenied("No tienes permisos para eliminar categorías de otros almacenes.")

    if categoria.productos.filter(archivado=False).exists():
        raise NecesitaTraslado(
            f"La categoría '{categoria.nombre}' tiene productos asociados. Debe trasladarlos a otra categoría antes de eliminarla.",
            entidad=categoria,
        )

    with transaction.atomic():
        categoria.activo = False
        categoria.eliminado = True
        categoria.save(update_fields=["activo", "eliminado"])

        registrar_auditoria(
            usuario=usuario,
            accion="ELIMINAR_CATEGORIA",
            tipo_entidad="Categoria",
            id_entidad=str(categoria.id),
            valores_anteriores={"activo": True, "eliminado": False},
            valores_nuevos={"activo": False, "eliminado": True},
            motivo=f"Eliminación lógica de categoría '{categoria.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRODUCTOS
# ─────────────────────────────────────────────────────────────────────────────

def crear_producto(
    *,
    nombre: str,
    categoria: Categoria,
    tipo: Optional[TipoCategoria] = None,
    stock_inicial: int = 0,
    stock_minimo: int = 0,
    es_prestable: bool = False,
    codigo_barras: str = "",
    nombre_empaque: str = "",
    unidades_por_empaque: Optional[int] = None,
    descripcion: str = "",
    ubicaciones: Optional[List[int]] = None,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Producto:
    """Crea un nuevo producto en el catálogo.
    
    Genera automáticamente el código interno inmutable (XXX-0001).
    Si stock_inicial > 0, inicializa existencias y genera el movimiento inicial.
    """
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para crear productos.")

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("catalogo.crear"):
            raise PermissionDenied("No tienes permisos para registrar nuevos productos en el catálogo.")
        if not usuario.tiene_acceso_almacen(categoria.almacen_principal):
            raise PermissionDenied("Solo puedes registrar productos en categorías de tus almacenes asignados.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre del producto es obligatorio."})

    if tipo is not None and tipo.categoria_id != categoria.id:
        raise ValidationError({"tipo": "El tipo seleccionado no pertenece a la categoría del producto."})

    codigo_barras = (codigo_barras or "").strip() or None
    if codigo_barras:
        prod_cb = Producto.objects.filter(codigo_barras=codigo_barras).first()
        if prod_cb:
            if not prod_cb.archivado:
                raise ValidationError({"codigo_barras": "Ya existe un producto activo registrado con este código de barras."})
            else:
                prod_cb.codigo_barras = None
                prod_cb.save(update_fields=["codigo_barras"])

    with transaction.atomic():
        producto = Producto(
            nombre=nombre,
            categoria=categoria,
            tipo=tipo,
            stock_inicial=max(0, int(stock_inicial or 0)),
            stock_minimo=max(0, int(stock_minimo or 0)),
            es_prestable=bool(es_prestable),
            codigo_barras=codigo_barras,
            nombre_empaque=(nombre_empaque or "").strip(),
            unidades_por_empaque=unidades_por_empaque if unidades_por_empaque and unidades_por_empaque > 0 else None,
            descripcion=(descripcion or "").strip(),
            activo=True,
            archivado=False,
        )
        producto.full_clean()
        producto.save()

        if ubicaciones:
            if len(ubicaciones) > 1:
                raise ValidationError({"ubicaciones": "Un producto solo puede estar asignado a una sola ubicación física."})
            uid = ubicaciones[0]
            ub = Ubicacion.objects.filter(id=uid, activo=True, eliminado=False).first()
            if not ub:
                raise ValidationError({"ubicaciones": "La ubicación física seleccionada no existe o está inactiva."})
            if ub.almacen_id != categoria.almacen_principal_id:
                raise ValidationError({
                    "ubicaciones": f"La ubicación '{ub.nombre}' pertenece a '{ub.almacen.nombre}', pero la categoría del producto pertenece a '{categoria.almacen_principal.nombre}'."
                })
            producto.ubicacion = ub
            producto.save(update_fields=["ubicacion"])

        # Crear existencia en almacén principal por defecto
        Existencia.objects.get_or_create(
            producto=producto,
            almacen=categoria.almacen_principal,
            defaults={"stock_disponible": 0, "stock_en_prestamo": 0},
        )

        if producto.stock_inicial > 0:
            from inventario.servicios.stock import inicializar_stock
            inicializar_stock(
                producto=producto,
                almacen=categoria.almacen_principal,
                cantidad=producto.stock_inicial,
                usuario=usuario,
                ip_origen=ip_origen,
            )

        registrar_auditoria(
            usuario=usuario,
            accion="CREAR_PRODUCTO",
            tipo_entidad="Producto",
            id_entidad=str(producto.id),
            valores_anteriores=None,
            valores_nuevos={
                "codigo_interno": producto.codigo_interno,
                "nombre": producto.nombre,
                "categoria_id": categoria.id,
                "tipo_id": tipo.id if tipo else None,
                "tipo_nombre": tipo.nombre if tipo else None,
                "stock_inicial": producto.stock_inicial,
                "stock_minimo": producto.stock_minimo,
                "es_prestable": producto.es_prestable,
                "codigo_barras": producto.codigo_barras,
            },
            motivo=f"Creación de producto '{producto.codigo_interno} — {producto.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return producto


def actualizar_producto(
    *,
    producto: Producto,
    nombre: str,
    stock_minimo: int,
    es_prestable: Optional[bool] = None,
    categoria: Optional[Categoria] = None,
    tipo: Optional[TipoCategoria] = None,
    actualizar_tipo: bool = False,
    codigo_barras: str = "",
    nombre_empaque: str = "",
    unidades_por_empaque: Optional[int] = None,
    descripcion: str = "",
    ubicaciones: Optional[List[int]] = None,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Producto:
    """Actualiza datos operativos y descriptivos de un producto.
    
    Nota: El almacén, ubicación física, categoría y stock son inmutables en este formulario;
    únicamente se modifican mediante movimientos operativos explícitos (Kardex/Traslados).
    """
    if not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para modificar productos.")

    producto_db = Producto.objects.select_related("categoria").get(pk=producto.pk)

    if getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_permiso("catalogo.editar"):
            raise PermissionDenied("No tienes permisos para modificar la información de productos.")
        if not usuario.tiene_acceso_almacen(producto_db.categoria.almacen_principal):
            raise PermissionDenied("No tienes permisos para modificar productos de otros almacenes.")

    nombre = (nombre or "").strip()
    if not nombre:
        raise ValidationError({"nombre": "El nombre del producto no puede estar vacío."})

    codigo_barras = (codigo_barras or "").strip() or None
    if codigo_barras:
        prod_existente = Producto.objects.filter(codigo_barras=codigo_barras).exclude(id=producto_db.id).first()
        if prod_existente:
            if not prod_existente.archivado:
                raise ValidationError({"codigo_barras": "Ya existe otro producto activo con este código de barras."})
            else:
                prod_existente.codigo_barras = None
                prod_existente.save(update_fields=["codigo_barras"])

    valores_anteriores = {
        "nombre": producto_db.nombre,
        "stock_minimo": producto_db.stock_minimo,
        "es_prestable": producto_db.es_prestable,
        "codigo_barras": producto_db.codigo_barras,
        "nombre_empaque": producto_db.nombre_empaque,
        "unidades_por_empaque": producto_db.unidades_por_empaque,
        "descripcion": producto_db.descripcion,
    }

    with transaction.atomic():
        producto_db.nombre = nombre
        producto_db.stock_minimo = max(0, int(stock_minimo or 0))
        if es_prestable is not None:
            producto_db.es_prestable = bool(es_prestable)
        producto_db.codigo_barras = codigo_barras
        producto_db.nombre_empaque = (nombre_empaque or "").strip()
        producto_db.unidades_por_empaque = unidades_por_empaque if unidades_por_empaque and unidades_por_empaque > 0 else None
        producto_db.descripcion = (descripcion or "").strip()
        producto_db.full_clean()
        producto_db.save()

        registrar_auditoria(
            usuario=usuario,
            accion="EDITAR_PRODUCTO",
            tipo_entidad="Producto",
            id_entidad=str(producto_db.id),
            valores_anteriores=valores_anteriores,
            valores_nuevos={
                "nombre": producto_db.nombre,
                "stock_minimo": producto_db.stock_minimo,
                "es_prestable": producto_db.es_prestable,
                "codigo_barras": producto_db.codigo_barras,
                "nombre_empaque": producto_db.nombre_empaque,
                "unidades_por_empaque": producto_db.unidades_por_empaque,
                "descripcion": producto_db.descripcion,
            },
            motivo=f"Actualización de producto '{producto_db.codigo_interno} — {producto_db.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return producto_db


def validar_integridad_dn01_producto(producto: Producto) -> None:
    """Valida la integridad estructural de DN-01 previo a cambios de estado de catálogo.

    Fail-Closed: Rechaza productos sin categoría, sin ubicación, con discrepancia
    de almacén o con estado híbrido inválido (activo=True y archivado=True).
    """
    if not producto.categoria_id:
        raise ValidationError("Infracción DN-01: El producto carece de categoría asignada.")

    if not producto.ubicacion_id:
        raise ValidationError("Infracción DN-01: El producto carece de ubicación física asignada.")

    categoria = producto.categoria
    ubicacion = producto.ubicacion

    if not ubicacion.almacen_id:
        raise ValidationError("Infracción DN-01: La ubicación física asignada no pertenece a un almacén válido.")

    if categoria.almacen_principal_id != ubicacion.almacen_id:
        alm_cat = categoria.almacen_principal.nombre if categoria.almacen_principal else str(categoria.almacen_principal_id)
        alm_ub = ubicacion.almacen.nombre if ubicacion.almacen else str(ubicacion.almacen_id)
        raise ValidationError(
            f"Infracción DN-01: El almacén de la categoría ('{alm_cat}') "
            f"no coincide con el almacén de la ubicación ('{alm_ub}')."
        )

    if getattr(ubicacion, "eliminado", False) or not getattr(ubicacion, "activo", True):
        raise ValidationError(f"Infracción DN-01: La ubicación física '{ubicacion.nombre}' asignada al producto está inactiva o eliminada.")

    if producto.activo and producto.archivado:
        raise ValidationError("Infracción DN-01: Estado híbrido inválido (el producto no puede estar activo y archivado simultáneamente).")


def _validar_permisos_catalogo(
    usuario: Any,
    permiso_o_permisos: Any,
    almacen: Optional[Almacen] = None,
) -> None:
    """Valida permisos granulares RBAC y alcance por almacén."""
    if not usuario or not getattr(usuario, "is_authenticated", False):
        raise PermissionDenied("Se requiere autenticación para modificar productos.")
    if not getattr(usuario, "is_active", True) or getattr(usuario, "eliminado_logicamente", False):
        raise PermissionDenied("Usuario inactivo o no autorizado.")
    if getattr(usuario, "es_administrador", False) or getattr(usuario, "is_superuser", False):
        return

    if isinstance(permiso_o_permisos, (list, tuple, set)):
        codigos = list(permiso_o_permisos)
    else:
        codigos = [permiso_o_permisos]

    tiene_alguno = any(
        getattr(usuario, "tiene_permiso", lambda p: False)(c)
        for c in codigos
    )
    if not tiene_alguno:
        perms_str = " o ".join(codigos)
        raise PermissionDenied(f"No cuentas con permisos suficientes ({perms_str}) para realizar esta operación.")

    if almacen and getattr(usuario, "es_usuario_normal", False):
        if not usuario.tiene_acceso_almacen(almacen):
            raise PermissionDenied("No tienes permisos para modificar productos de otros almacenes.")


CLAVES_AUXILIARES_AUDITORIA_CATALOGO = frozenset({
    "ip_origen",
    "session_token",
    "dispositivo",
    "canal",
    "origen_solicitud",
})


def _validar_formato_factor_haku(factor: str, valor: Any) -> bool:
    """Valida la integridad y formato canónico de los factores HAKU."""
    if not isinstance(valor, str) or not valor.strip():
        return False
    if factor in ("propuesta_id", "idempotency_key"):
        return 3 <= len(valor) <= 128
    if factor == "payload_firma":
        return len(valor) == 67 and valor.startswith(("v1:", "v2:")) and all(c in "0123456789abcdefABCDEF" for c in valor[3:])
    if factor == "payload_hash":
        return len(valor) == 64 and all(c in "0123456789abcdefABCDEF" for c in valor)
    return False


def _construir_valores_nuevos_auditoria(
    *,
    producto_id: int,
    activo: bool,
    archivado: bool,
    via_haku: bool = False,
    metadatos_auditoria: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construye el diccionario de valores_nuevos aplicando whitelist estricta.

    Garantías de seguridad:
    1. 'activo', 'archivado' y 'producto_id' se fijan únicamente desde los parámetros reales del dominio.
    2. Los factores de HAKU ('propuesta_id', 'idempotency_key', 'payload_firma', 'payload_hash')
       solo se incorporan si provienen de via_haku=True y cumplen el formato criptográfico canónico.
    3. Solo se aceptan claves auxiliares expresamente presentes en CLAVES_AUXILIARES_AUDITORIA_CATALOGO.
    4. Cualquier clave desconocida o campo reservado inyectado es ignorado y descartado de forma documentada.
    """
    val_nuevos: Dict[str, Any] = {
        "activo": activo,
        "archivado": archivado,
        "producto_id": str(producto_id),
    }

    if not metadatos_auditoria or not isinstance(metadatos_auditoria, dict):
        return val_nuevos

    # Factores HAKU validados
    if via_haku:
        for factor in ("propuesta_id", "idempotency_key", "payload_firma", "payload_hash"):
            if factor in metadatos_auditoria:
                val = metadatos_auditoria[factor]
                if _validar_formato_factor_haku(factor, val):
                    val_nuevos[factor] = val
                else:
                    logger.warning("[AUDITORIA] Factor HAKU '%s' con formato inválido descartado: %r", factor, val)

    # Claves auxiliares expresamente autorizadas por whitelist
    for k in CLAVES_AUXILIARES_AUDITORIA_CATALOGO:
        if k in metadatos_auditoria:
            v = metadatos_auditoria[k]
            if isinstance(v, (str, int, float, bool)):
                val_nuevos[k] = v

    return val_nuevos


def habilitar_producto(
    *,
    producto: Producto,
    usuario: Any,
    ip_origen: Optional[str] = None,
    via_haku: bool = False,
    metadatos_auditoria: Optional[Dict[str, Any]] = None,
) -> Producto:
    """Habilita y reactiva un producto de forma absoluta (activo=True, archivado=False).

    Aplica validación de permisos ('catalogo.editar'), alcance por almacén,
    cumplimiento estricto de DN-01 (Fail-Closed) y concurrencia OCC atómica
    con reintentos en transacciones autónomas ante contención SQLite.
    """
    from inventario.db.transacciones import (
        _hay_transaccion_externa_activa,
        es_error_transitorio_bd,
    )
    from django.db import DatabaseError, OperationalError

    hay_tx_ext = _hay_transaccion_externa_activa(connection)
    max_intentos = 1 if hay_tx_ext else 3
    tiempos_espera = (0.05, 0.15, 0.40)

    ultimo_error = None
    for intento in range(max_intentos):
        try:
            with transaction.atomic():
                producto_db = Producto.objects.select_related(
                    "categoria__almacen_principal", "ubicacion__almacen"
                ).get(pk=producto.pk)
                _validar_permisos_catalogo(
                    usuario,
                    "catalogo.editar",
                    producto_db.categoria.almacen_principal if producto_db.categoria else None,
                )
                validar_integridad_dn01_producto(producto_db)

                if producto_db.activo and not producto_db.archivado:
                    raise ValidationError(
                        f"El producto '{producto_db.codigo_interno} — {producto_db.nombre}' ya se encuentra habilitado."
                    )

                val_ant = {"activo": producto_db.activo, "archivado": producto_db.archivado}

                filas = Producto.objects.filter(
                    id=producto_db.id,
                    activo=producto_db.activo,
                    archivado=producto_db.archivado,
                ).update(activo=True, archivado=False)

                if filas == 0:
                    raise ValidationError(
                        f"Conflicto de concurrencia: el producto '{producto_db.codigo_interno}' fue modificado concurrentemente."
                    )

                val_nuevos = _construir_valores_nuevos_auditoria(
                    producto_id=producto_db.id,
                    activo=True,
                    archivado=False,
                    via_haku=via_haku,
                    metadatos_auditoria=metadatos_auditoria,
                )

                motivo = f"Habilitación/Reactivación del producto '{producto_db.codigo_interno} — {producto_db.nombre}'"
                if metadatos_auditoria and "motivo_extra" in metadatos_auditoria:
                    motivo = f"{motivo} {metadatos_auditoria['motivo_extra']}".strip()

                registrar_auditoria(
                    usuario=usuario,
                    accion="HABILITAR_PRODUCTO",
                    tipo_entidad="Producto",
                    id_entidad=str(producto_db.id),
                    valores_anteriores=val_ant,
                    valores_nuevos=val_nuevos,
                    motivo=motivo,
                    resultado="OK",
                    via_haku=via_haku,
                    ip_origen=ip_origen,
                )

                producto_db.activo = True
                producto_db.archivado = False

                if via_haku:
                    from integraciones.haku.cache import HakuCacheManager
                    transaction.on_commit(lambda: HakuCacheManager.invalidar_catalogo())
                    transaction.on_commit(lambda p_id=producto_db.id: HakuCacheManager.invalidar_stock(p_id))
                    transaction.on_commit(lambda p_id=producto_db.id: HakuCacheManager.invalidar_ubicaciones(p_id))

            producto.activo = True
            producto.archivado = False
            return producto_db
        except (OperationalError, DatabaseError) as exc:
            ultimo_error = exc
            if not hay_tx_ext and es_error_transitorio_bd(exc) and intento < max_intentos - 1:
                time.sleep(tiempos_espera[intento])
                continue
            raise

    if ultimo_error:
        raise ultimo_error
    return producto


def inhabilitar_producto(
    *,
    producto: Producto,
    usuario: Any,
    ip_origen: Optional[str] = None,
    via_haku: bool = False,
    metadatos_auditoria: Optional[Dict[str, Any]] = None,
) -> Producto:
    """Inhabilita y archiva un producto de forma absoluta (activo=False, archivado=True).

    Aplica validación de permisos ('catalogo.eliminar'), alcance por almacén,
    cumplimiento estricto de DN-01 (Fail-Closed) y concurrencia OCC atómica
    con reintentos en transacciones autónomas ante contención SQLite.
    """
    from inventario.db.transacciones import (
        _hay_transaccion_externa_activa,
        es_error_transitorio_bd,
    )
    from django.db import DatabaseError, OperationalError

    hay_tx_ext = _hay_transaccion_externa_activa(connection)
    max_intentos = 1 if hay_tx_ext else 3
    tiempos_espera = (0.05, 0.15, 0.40)

    ultimo_error = None
    for intento in range(max_intentos):
        try:
            with transaction.atomic():
                producto_db = Producto.objects.select_related(
                    "categoria__almacen_principal", "ubicacion__almacen"
                ).get(pk=producto.pk)
                _validar_permisos_catalogo(
                    usuario,
                    "catalogo.eliminar",
                    producto_db.categoria.almacen_principal if producto_db.categoria else None,
                )
                validar_integridad_dn01_producto(producto_db)

                if not producto_db.activo and producto_db.archivado:
                    raise ValidationError(
                        f"El producto '{producto_db.codigo_interno} — {producto_db.nombre}' ya se encuentra inhabilitado y archivado."
                    )

                val_ant = {"activo": producto_db.activo, "archivado": producto_db.archivado}

                filas = Producto.objects.filter(
                    id=producto_db.id,
                    activo=producto_db.activo,
                    archivado=producto_db.archivado,
                ).update(activo=False, archivado=True)

                if filas == 0:
                    raise ValidationError(
                        f"Conflicto de concurrencia: el producto '{producto_db.codigo_interno}' fue modificado concurrentemente."
                    )

                val_nuevos = _construir_valores_nuevos_auditoria(
                    producto_id=producto_db.id,
                    activo=False,
                    archivado=True,
                    via_haku=via_haku,
                    metadatos_auditoria=metadatos_auditoria,
                )

                motivo = f"Inhabilitación/Archivado del producto '{producto_db.codigo_interno} — {producto_db.nombre}'"
                if metadatos_auditoria and "motivo_extra" in metadatos_auditoria:
                    motivo = f"{motivo} {metadatos_auditoria['motivo_extra']}".strip()

                registrar_auditoria(
                    usuario=usuario,
                    accion="INHABILITAR_PRODUCTO",
                    tipo_entidad="Producto",
                    id_entidad=str(producto_db.id),
                    valores_anteriores=val_ant,
                    valores_nuevos=val_nuevos,
                    motivo=motivo,
                    resultado="OK",
                    via_haku=via_haku,
                    ip_origen=ip_origen,
                )

                producto_db.activo = False
                producto_db.archivado = True

                if via_haku:
                    from integraciones.haku.cache import HakuCacheManager
                    transaction.on_commit(lambda: HakuCacheManager.invalidar_catalogo())
                    transaction.on_commit(lambda p_id=producto_db.id: HakuCacheManager.invalidar_stock(p_id))
                    transaction.on_commit(lambda p_id=producto_db.id: HakuCacheManager.invalidar_ubicaciones(p_id))

            producto.activo = False
            producto.archivado = True
            return producto_db
        except (OperationalError, DatabaseError) as exc:
            ultimo_error = exc
            if not hay_tx_ext and es_error_transitorio_bd(exc) and intento < max_intentos - 1:
                time.sleep(tiempos_espera[intento])
                continue
            raise

    if ultimo_error:
        raise ultimo_error
    return producto


class ErrorConflictoConcurrenteToggle(Exception):
    """Excepción interna ante colisión concurrente (filas == 0) en toggle_activo_producto."""
    pass


def _intentar_toggle_transaccional(
    producto_id: int,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Producto:
    """Intento atómico único de toggle protegido por transacción autónoma."""
    from inventario.db.transacciones import transaccion_con_reintento

    @transaccion_con_reintento(max_intentos=1)
    def _ejecutar():
        producto_db = Producto.objects.select_related("categoria__almacen_principal", "ubicacion__almacen").get(pk=producto_id)
        _validar_permisos_catalogo(usuario, "catalogo.editar", producto_db.categoria.almacen_principal if producto_db.categoria else None)
        validar_integridad_dn01_producto(producto_db)

        ant_activo = producto_db.activo
        ant_archivado = producto_db.archivado

        if ant_activo:
            nuevo_activo = False
            nuevo_archivado = True
        else:
            nuevo_activo = True
            nuevo_archivado = False

        filas = Producto.objects.filter(
            id=producto_db.id,
            activo=ant_activo,
            archivado=ant_archivado,
        ).update(activo=nuevo_activo, archivado=nuevo_archivado)

        if filas == 0:
            raise ErrorConflictoConcurrenteToggle("Conflicto concurrente al actualizar estado del producto.")

        registrar_auditoria(
            usuario=usuario,
            accion="TOGGLE_ACTIVO_PRODUCTO",
            tipo_entidad="Producto",
            id_entidad=str(producto_db.id),
            valores_anteriores={"activo": ant_activo, "archivado": ant_archivado},
            valores_nuevos={"activo": nuevo_activo, "archivado": nuevo_archivado},
            motivo=f"{'Reactivación' if nuevo_activo else 'Inhabilitación'} del producto '{producto_db.codigo_interno} — {producto_db.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

        producto_db.activo = nuevo_activo
        producto_db.archivado = nuevo_archivado
        return producto_db

    return _ejecutar()


def toggle_activo_producto(
    *,
    producto: Producto,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Producto:
    """Inhabilita o reactiva un producto sin estados híbridos, rechazando transacciones externas.

    Garantiza compatibilidad web absoluta retornando un Producto con .activo,
    .codigo_interno y .nombre actualizados.
    """
    from inventario.db.transacciones import _hay_transaccion_externa_activa
    from django.db import OperationalError

    if _hay_transaccion_externa_activa(connection):
        raise RuntimeError("toggle_activo_producto no admite transacciones externas activas.")

    MAX_INTENTOS_TOGGLE = 3
    ultimo_error = None
    for intento in range(MAX_INTENTOS_TOGGLE):
        try:
            prod_actualizado = _intentar_toggle_transaccional(producto.pk, usuario, ip_origen)
            producto.activo = prod_actualizado.activo
            producto.archivado = prod_actualizado.archivado
            return prod_actualizado
        except (ErrorConflictoConcurrenteToggle, OperationalError) as exc:
            ultimo_error = exc
            if intento < MAX_INTENTOS_TOGGLE - 1:
                time.sleep(0.05 * (2 ** intento))
                continue
            raise

    if ultimo_error:
        raise ultimo_error
    return producto


def archivar_producto(
    *,
    producto: Producto,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> None:
    """Archiva un producto ocultándolo de listas operativas pero conservando historial.
    
    Solo Administradores y Superusuarios (§8.1).
    """
    if not getattr(usuario, "es_administrador", False) and not getattr(usuario, "tiene_permiso", lambda p: False)("catalogo.eliminar"):
        raise PermissionDenied("No tienes permisos para archivar o eliminar productos.")

    with transaction.atomic():
        producto.archivado = True
        producto.activo = False
        producto.save(update_fields=["archivado", "activo"])

        registrar_auditoria(
            usuario=usuario,
            accion="ARCHIVAR_PRODUCTO",
            tipo_entidad="Producto",
            id_entidad=str(producto.id),
            valores_anteriores={"archivado": False, "activo": True},
            valores_nuevos={"archivado": True, "activo": False},
            motivo=f"Archivo del producto '{producto.codigo_interno} — {producto.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )


def fusionar_productos(
    *,
    producto_principal: Producto,
    producto_duplicado: Producto,
    usuario: Any,
    ip_origen: Optional[str] = None,
) -> Producto:
    """Fusiona un producto duplicado en el producto principal (§8.3).
    
    1. Suma existencias por almacén.
    2. Reasigna todos los movimientos y préstamos.
    3. Unifica ubicaciones físicas.
    4. Archiva el producto duplicado y deja su existencia en 0.
    5. Registra auditoría exhaustiva.
    """
    if not getattr(usuario, "es_administrador", False):
        raise PermissionDenied("Solo los Administradores o el Superusuario pueden fusionar productos duplicados.")

    if producto_principal.id == producto_duplicado.id:
        raise ValidationError("No se puede fusionar un producto consigo mismo.")

    with transaction.atomic():
        prod_pri = Producto.objects.select_for_update().get(pk=producto_principal.pk)
        prod_dup = Producto.objects.select_for_update().get(pk=producto_duplicado.pk)

        # 1. Sumar existencias al almacén del producto principal
        alm_pri = prod_pri.categoria.almacen_principal
        ex_pri, _ = Existencia.objects.select_for_update().get_or_create(
            producto=prod_pri,
            almacen=alm_pri,
            defaults={"stock_disponible": 0, "stock_en_prestamo": 0},
        )
        for ex_dup in list(prod_dup.existencias.select_for_update()):
            ex_pri.stock_disponible += ex_dup.stock_disponible
            ex_pri.stock_en_prestamo += ex_dup.stock_en_prestamo
            ex_dup.stock_disponible = 0
            ex_dup.stock_en_prestamo = 0
            ex_dup.save()
            if ex_dup.almacen_id != alm_pri.id:
                ex_dup.delete()
        ex_pri.save()

        # 2. Reasignar movimientos
        Movimiento.objects.filter(producto=prod_dup).update(producto=prod_pri)

        # 3. Reasignar préstamos (conservando consistencia DN-01 con el almacén del producto principal)
        Prestamo.objects.filter(producto=prod_dup).update(producto=prod_pri, almacen=alm_pri)

        # 4. Reasignar conteos físicos respetando la restricción única de 1 conteo PENDIENTE por producto
        tiene_pendiente_pri = ConteoFisico.objects.filter(
            producto=prod_pri, estado=ConteoFisico.Estado.PENDIENTE
        ).exists()
        if tiene_pendiente_pri:
            from django.utils import timezone as _tz
            ConteoFisico.objects.filter(
                producto=prod_dup, estado=ConteoFisico.Estado.PENDIENTE
            ).update(
                estado=ConteoFisico.Estado.RECHAZADO,
                motivo_rechazo=f"Rechazado automáticamente al fusionar '{prod_dup.codigo_interno}' en '{prod_pri.codigo_interno}' (el producto principal ya tenía un conteo pendiente).",
                usuario_confirmacion=usuario,
                fecha_confirmacion=_tz.now(),
            )
        else:
            ConteoFisico.objects.filter(
                producto=prod_dup, estado=ConteoFisico.Estado.PENDIENTE
            ).update(almacen=alm_pri)
        ConteoFisico.objects.filter(producto=prod_dup).update(producto=prod_pri)

        # 5. Unificar ubicación física única
        if not prod_pri.ubicacion and prod_dup.ubicacion and prod_dup.ubicacion.almacen_id == alm_pri.id:
            prod_pri.ubicacion = prod_dup.ubicacion
            prod_pri.save(update_fields=["ubicacion"])
        prod_dup.ubicacion = None
        prod_dup.save(update_fields=["ubicacion"])

        # 6. Archivar producto duplicado
        prod_dup.archivado = True
        prod_dup.activo = False
        prod_dup.save(update_fields=["archivado", "activo"])

        # 7. Auditoría
        registrar_auditoria(
            usuario=usuario,
            accion="FUSIONAR_PRODUCTOS",
            tipo_entidad="Producto",
            id_entidad=str(prod_pri.id),
            valores_anteriores={
                "duplicado_id": prod_dup.id,
                "duplicado_codigo": prod_dup.codigo_interno,
                "duplicado_nombre": prod_dup.nombre,
            },
            valores_nuevos={
                "principal_id": prod_pri.id,
                "principal_codigo": prod_pri.codigo_interno,
            },
            motivo=f"Fusión de producto duplicado '{prod_dup.codigo_interno} — {prod_dup.nombre}' en '{prod_pri.codigo_interno} — {prod_pri.nombre}'",
            resultado="OK",
            via_haku=False,
            ip_origen=ip_origen,
        )

    return prod_pri
