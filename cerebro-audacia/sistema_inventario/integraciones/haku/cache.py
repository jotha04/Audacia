"""Gestor de caché en RAM para optimización de lecturas frecuentes en Haku Inventario.

Cachés independientes (LocMemCache):
1. haku_cache_catalogo: Nombres, códigos y categorías (TTL: 30 minutos).
   Invalidado ante creación, edición o archivado de Producto.
2. haku_cache_ubicaciones: Ubicación física formateada por producto (TTL: 15 minutos).
   Invalidado cuando cambia la ubicación del producto.
3. haku_cache_stock: Stock orientativo disponible/en préstamo (TTL: 15 segundos máximo).
   Invalidado INMEDIATAMENTE tras cualquier movimiento, entrada, salida, préstamo o devolución.

REGLAS DE SEGURIDAD INNEGOCIABLES:
- NUNCA se cachean operaciones críticas ni validaciones de saldo para ejecutar transacciones.
- Toda mutación consulta la base de datos con select_for_update() en transacción atómica.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from django.core.cache import caches
from django.db.models.signals import m2m_changed, post_delete, post_save

logger = logging.getLogger("integraciones.haku.cache")

ALIAS_CATALOGO = "haku_catalogo"
ALIAS_UBICACIONES = "haku_ubicaciones"
ALIAS_STOCK = "haku_stock"

TTL_CATALOGO = 1800     # 30 minutos
TTL_UBICACIONES = 900   # 15 minutos
TTL_STOCK = 15          # 15 segundos máximo
TTL_FALTANTES = 60      # 60 segundos para consultas agregadas de faltantes/reposición


class HakuCacheManager:
    """Gestor centralizado de cachés en RAM y métricas de rendimiento interno."""

    _lock = threading.Lock()
    _catalog_hits: int = 0
    _catalog_misses: int = 0
    _location_hits: int = 0
    _location_misses: int = 0
    _stock_hits: int = 0
    _stock_misses: int = 0
    _faltantes_hits: int = 0
    _faltantes_misses: int = 0
    _intenciones_detectadas: Dict[str, int] = {}

    @classmethod
    def _cache(cls, alias: str):
        try:
            return caches[alias]
        except Exception:
            return caches["default"]

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CACHÉ DE CATÁLOGO (haku_cache_catalogo)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def obtener_catalogo(cls, almacen_id: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """Obtiene la lista mínima de productos para reconocimiento del lenguaje."""
        cache = cls._cache(ALIAS_CATALOGO)
        key = f"catalogo:{almacen_id or 'all'}"
        data = cache.get(key)
        with cls._lock:
            if data is not None:
                cls._catalog_hits += 1
            else:
                cls._catalog_misses += 1
        return data

    @classmethod
    def guardar_catalogo(cls, data: List[Dict[str, Any]], almacen_id: Optional[int] = None) -> None:
        cache = cls._cache(ALIAS_CATALOGO)
        key = f"catalogo:{almacen_id or 'all'}"
        cache.set(key, data, timeout=TTL_CATALOGO)

    @classmethod
    def obtener_o_cargar_catalogo(cls, almacen_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Obtiene el catálogo desde caché o lo precarga de DB si no existe."""
        cached = cls.obtener_catalogo(almacen_id)
        if cached is not None:
            return cached
        from inventario.models import Producto
        from django.db.models import Q
        qs = Producto.objects.filter(activo=True, archivado=False)
        if almacen_id:
            qs = qs.filter(Q(categoria__almacen_principal_id=almacen_id) | Q(existencias__almacen_id=almacen_id)).distinct()
        data = list(qs.values("id", "nombre", "codigo_interno", "codigo_barras", "categoria__nombre"))
        cls.guardar_catalogo(data, almacen_id)
        return data

    @classmethod
    def invalidar_catalogo(cls) -> None:
        """Invalida el catálogo completo en RAM y el índice de búsqueda en memoria."""
        cache = cls._cache(ALIAS_CATALOGO)
        cache.clear()
        cls.invalidar_faltantes()
        try:
            from .buscador import HakuSearchIndex
            HakuSearchIndex.invalidar()
        except Exception:
            pass
        try:
            from .audio.transcriber import HakuTranscriber
            HakuTranscriber.invalidar_cache_vocabulario()
        except Exception:
            pass
        try:
            from .agente.planificador import limpiar_cache_planes
            limpiar_cache_planes()
        except Exception:
            pass
        logger.debug("Caché de catálogo, índice de búsqueda, vocabulario STT y caché de planes invalidados")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CACHÉ DE UBICACIONES (haku_cache_ubicaciones)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _ver_ubicaciones(cls, producto_id: int) -> int:
        cache = cls._cache(ALIAS_UBICACIONES)
        return cache.get(f"ubicacion_ver:{producto_id}") or 1

    @classmethod
    def _key_ubicacion(cls, producto_id: int, almacen_id: Optional[int] = None, scope_id: Optional[str] = None) -> str:
        ver = cls._ver_ubicaciones(producto_id)
        if almacen_id is not None:
            return f"ubicacion:{producto_id}:v{ver}:alm_{almacen_id}"
        return f"ubicacion:{producto_id}:v{ver}:scope_{scope_id or 'default'}"

    @classmethod
    def obtener_ubicacion(
        cls,
        producto_id: int,
        almacen_id: Optional[int] = None,
        scope_id: Optional[str] = None,
    ) -> Optional[str]:
        cache = cls._cache(ALIAS_UBICACIONES)
        key = cls._key_ubicacion(producto_id, almacen_id, scope_id)
        val = cache.get(key)
        if val is None and almacen_id is None and (scope_id is None or scope_id == "default"):
            # Compatibilidad con claves de pruebas previas
            val = cache.get(f"ubicacion:{producto_id}")
        with cls._lock:
            if val is not None:
                cls._location_hits += 1
            else:
                cls._location_misses += 1
        return val

    @classmethod
    def guardar_ubicacion(
        cls,
        producto_id: int,
        ubicacion_str: str,
        almacen_id: Optional[int] = None,
        scope_id: Optional[str] = None,
    ) -> None:
        cache = cls._cache(ALIAS_UBICACIONES)
        key = cls._key_ubicacion(producto_id, almacen_id, scope_id)
        cache.set(key, ubicacion_str, timeout=TTL_UBICACIONES)
        if almacen_id is None and (scope_id is None or scope_id == "default"):
            cache.set(f"ubicacion:{producto_id}", ubicacion_str, timeout=TTL_UBICACIONES)

    @classmethod
    def obtener_o_cargar_ubicacion(
        cls,
        producto: Any,
        almacen_id: Optional[int] = None,
        usuario: Any | None = None,
    ) -> str:
        """Obtiene la ubicación formateada de caché o de DB respetando almacén y permisos efectivos."""
        if almacen_id is not None:
            scope_id = f"alm_{almacen_id}"
        elif usuario is not None:
            if getattr(usuario, "es_administrador", False) or getattr(usuario, "acceso_todos_almacenes", False) or getattr(usuario, "is_superuser", False):
                scope_id = "global"
            elif getattr(usuario, "es_usuario_normal", False):
                alms = list(usuario.obtener_almacenes_accesibles().values_list("id", flat=True)) if hasattr(usuario, "obtener_almacenes_accesibles") else []
                scope_id = f"user_{usuario.id}_alms_{'-'.join(map(str, sorted(alms)))}" if alms else "none"
            else:
                scope_id = "default"
        else:
            scope_id = "default"

        cached = cls.obtener_ubicacion(producto.id, almacen_id=almacen_id, scope_id=scope_id)
        if cached is not None:
            return cached

        # Carga de base de datos con aislamiento estricto por almacén y permisos
        if almacen_id is not None:
            qs = producto.ubicaciones.filter(almacen_id=almacen_id, almacen__activo=True, almacen__eliminado=False)
        elif usuario is not None:
            if getattr(usuario, "es_administrador", False) or getattr(usuario, "acceso_todos_almacenes", False) or getattr(usuario, "is_superuser", False):
                qs = producto.ubicaciones.filter(almacen__activo=True, almacen__eliminado=False)
            elif getattr(usuario, "es_usuario_normal", False):
                alms_acc = usuario.obtener_almacenes_accesibles() if hasattr(usuario, "obtener_almacenes_accesibles") else []
                qs = producto.ubicaciones.filter(almacen__in=alms_acc, almacen__activo=True, almacen__eliminado=False)
            else:
                qs = producto.ubicaciones.filter(almacen__activo=True, almacen__eliminado=False)
        else:
            # Política segura sin contexto de usuario: ubicaciones del almacén principal de la categoría
            cat_alm = getattr(getattr(producto, "categoria", None), "almacen_principal", None)
            if cat_alm and getattr(cat_alm, "activo", True) and not getattr(cat_alm, "eliminado", False):
                qs = producto.ubicaciones.filter(almacen=cat_alm)
            else:
                qs = producto.ubicaciones.filter(almacen__activo=True, almacen__eliminado=False)

        ubicaciones = ", ".join(u.nombre for u in qs) or "Sin ubicación asignada"
        cls.guardar_ubicacion(producto.id, ubicaciones, almacen_id=almacen_id, scope_id=scope_id)
        return ubicaciones

    @classmethod
    def invalidar_ubicaciones(cls, producto_id: Optional[int] = None) -> None:
        cache = cls._cache(ALIAS_UBICACIONES)
        if producto_id:
            ver = (cache.get(f"ubicacion_ver:{producto_id}") or 1) + 1
            cache.set(f"ubicacion_ver:{producto_id}", ver, timeout=TTL_UBICACIONES * 10)
            cache.delete(f"ubicacion:{producto_id}")
        else:
            cache.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # 3. CACHÉ DE STOCK (haku_cache_stock - TTL 15s)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _calcular_scope_usuario(cls, usuario: Any | None) -> str:
        if usuario is None:
            return "global"
        if getattr(usuario, "es_administrador", False) or getattr(usuario, "acceso_todos_almacenes", False) or getattr(usuario, "is_superuser", False):
            return f"global_rol_{getattr(usuario, 'rol', 'ADMIN')}"
        if hasattr(usuario, "obtener_almacenes_accesibles"):
            alms_ids = sorted(list(usuario.obtener_almacenes_accesibles().values_list("id", flat=True)))
            return f"user_{usuario.id}_alms_{'-'.join(str(i) for i in alms_ids)}"
        return f"user_{getattr(usuario, 'id', 'anon')}"

    @classmethod
    def _ver_stock_global(cls, producto_id: int) -> int:
        cache = cls._cache(ALIAS_STOCK)
        return cache.get(f"stock_ver_global:{producto_id}") or 1

    @classmethod
    def _key_stock(cls, producto_id: int, almacen_id: Optional[int] = None, scope_id: Optional[str] = None) -> str:
        if almacen_id is not None:
            return f"stock:{producto_id}:{almacen_id}"
        ver = cls._ver_stock_global(producto_id)
        return f"stock:{producto_id}:all:v{ver}:{scope_id or 'global'}"

    @classmethod
    def obtener_stock(
        cls,
        producto_id: int,
        almacen_id: Optional[int] = None,
        scope_id: Optional[str] = None,
    ) -> Optional[Dict[str, int]]:
        """Consulta stock orientativo de respuesta rápida."""
        cache = cls._cache(ALIAS_STOCK)
        key = cls._key_stock(producto_id, almacen_id, scope_id)
        val = cache.get(key)
        if val is None and almacen_id is None and (scope_id is None or scope_id == "global"):
            val = cache.get(f"stock:{producto_id}:all")
        with cls._lock:
            if val is not None:
                cls._stock_hits += 1
            else:
                cls._stock_misses += 1
        return val

    @classmethod
    def guardar_stock(
        cls,
        producto_id: int,
        almacen_id: Optional[int],
        data: Dict[str, int],
        scope_id: Optional[str] = None,
    ) -> None:
        cache = cls._cache(ALIAS_STOCK)
        key = cls._key_stock(producto_id, almacen_id, scope_id)
        cache.set(key, data, timeout=TTL_STOCK)

    @classmethod
    def obtener_o_cargar_stock(
        cls,
        producto_id: int,
        almacen_id: Optional[int] = None,
        usuario: Any | None = None,
        scope_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Obtiene stock orientativo desde caché o lo consulta de DB respetando permisos."""
        if almacen_id is None and scope_id is None and usuario is not None:
            scope_id = cls._calcular_scope_usuario(usuario)

        cached = cls.obtener_stock(producto_id, almacen_id, scope_id=scope_id)
        if cached is not None:
            return cached

        from inventario.models import Existencia
        if almacen_id is not None:
            existencia = Existencia.objects.filter(producto_id=producto_id, almacen_id=almacen_id).first()
            disponible = existencia.stock_disponible if existencia else 0
            prestado = existencia.stock_en_prestamo if existencia else 0
            data = {"disponible": disponible, "prestado": prestado}
        else:
            from django.db.models import Sum
            qs = Existencia.objects.filter(
                producto_id=producto_id,
                almacen__activo=True,
                almacen__eliminado=False,
            )
            if usuario is not None:
                if getattr(usuario, "es_usuario_normal", False) and not getattr(usuario, "acceso_todos_almacenes", False):
                    alms = usuario.obtener_almacenes_accesibles() if hasattr(usuario, "obtener_almacenes_accesibles") else []
                    qs = qs.filter(almacen__in=alms)
            totales = qs.aggregate(
                total_disp=Sum("stock_disponible"),
                total_prest=Sum("stock_en_prestamo"),
            )
            data = {
                "disponible": int(totales["total_disp"] or 0),
                "prestado": int(totales["total_prest"] or 0),
            }

        cls.guardar_stock(producto_id, almacen_id, data, scope_id=scope_id)
        return data

    @classmethod
    def invalidar_stock(cls, producto_id: Optional[int] = None, almacen_id: Optional[int] = None) -> None:
        """Invalida inmediatamente el stock tras cualquier mutación confirmada."""
        cache = cls._cache(ALIAS_STOCK)
        if producto_id:
            if almacen_id:
                cache.delete(f"stock:{producto_id}:{almacen_id}")
            cache.delete(f"stock:{producto_id}:all")
            cache.delete(f"stock:{producto_id}:all:global")
            ver = (cache.get(f"stock_ver_global:{producto_id}") or 1) + 1
            cache.set(f"stock_ver_global:{producto_id}", ver, timeout=TTL_STOCK * 10)
        else:
            cache.clear()
        cls.invalidar_faltantes(almacen_id)
        logger.debug("Caché de stock invalidado (producto=%s, almacen=%s)", producto_id, almacen_id)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CACHÉ DE FALTANTES Y REPOSICIÓN (TTL 60s)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalizar_filtro_faltantes(filtro: Optional[str]) -> str:
        if not filtro:
            return "all"
        import hashlib
        norm = filtro.strip().lower()
        slug = "".join(c if c.isalnum() else "_" for c in norm)
        if len(slug) <= 24:
            return f"f_{slug}"
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
        return f"h_{h}"

    @classmethod
    def _ver_faltantes(cls, almacen_id: Optional[int] = None) -> int:
        cache = cls._cache(ALIAS_STOCK)
        ver_global = cache.get("faltantes:ver:global") or 1
        if almacen_id is not None:
            ver_alm = cache.get(f"faltantes:ver:{almacen_id}") or 1
            return ver_global * 10000 + ver_alm
        return ver_global

    @classmethod
    def _key_faltantes(
        cls,
        almacen_id: Optional[int] = None,
        categoria_id: Optional[int] = None,
        scope_id: Optional[str] = None,
        filtro_nombre: Optional[str] = None,
    ) -> str:
        ver = cls._ver_faltantes(almacen_id)
        filtro_part = cls._normalizar_filtro_faltantes(filtro_nombre)
        return f"faltantes:{almacen_id or 'all'}:v{ver}:{categoria_id or 'all'}:{filtro_part}:{scope_id or 'global'}"

    @classmethod
    def obtener_faltantes(
        cls,
        almacen_id: Optional[int] = None,
        categoria_id: Optional[int] = None,
        scope_id: Optional[str] = None,
        filtro_nombre: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        cache = cls._cache(ALIAS_STOCK)
        key = cls._key_faltantes(almacen_id, categoria_id, scope_id, filtro_nombre)
        data = cache.get(key)
        if data is None and not filtro_nombre:
            data = cache.get(f"faltantes:{almacen_id or 'all'}:{categoria_id or 'all'}:{scope_id or 'global'}")
        with cls._lock:
            if data is not None:
                cls._faltantes_hits += 1
            else:
                cls._faltantes_misses += 1
        return data

    @classmethod
    def guardar_faltantes(
        cls,
        data: Dict[str, Any],
        almacen_id: Optional[int] = None,
        categoria_id: Optional[int] = None,
        scope_id: Optional[str] = None,
        filtro_nombre: Optional[str] = None,
    ) -> None:
        cache = cls._cache(ALIAS_STOCK)
        key = cls._key_faltantes(almacen_id, categoria_id, scope_id, filtro_nombre)
        cache.set(key, data, timeout=TTL_FALTANTES)
        if not filtro_nombre:
            cache.set(f"faltantes:{almacen_id or 'all'}:{categoria_id or 'all'}:{scope_id or 'global'}", data, timeout=TTL_FALTANTES)

    @classmethod
    def invalidar_faltantes(cls, almacen_id: Optional[int] = None) -> None:
        """Invalida granularmente los faltantes mediante versionado sin borrar el stock en RAM."""
        cache = cls._cache(ALIAS_STOCK)
        ver_global = (cache.get("faltantes:ver:global") or 1) + 1
        cache.set("faltantes:ver:global", ver_global, timeout=TTL_FALTANTES * 10)
        if almacen_id is not None:
            ver_alm = (cache.get(f"faltantes:ver:{almacen_id}") or 1) + 1
            cache.set(f"faltantes:ver:{almacen_id}", ver_alm, timeout=TTL_FALTANTES * 10)
        cache.delete("faltantes:all:all:global")
        if almacen_id:
            cache.delete(f"faltantes:{almacen_id}:all:global")
        logger.debug("Caché de faltantes invalidado granularmente (almacen=%s, ver_global=%s)", almacen_id, ver_global)

    # ─────────────────────────────────────────────────────────────────────────
    # GESTIÓN Y MÉTRICAS
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def invalidar_todo(cls) -> None:
        """Limpia todos los cachés de lectura."""
        cls.invalidar_catalogo()
        cls.invalidar_ubicaciones()
        cls.invalidar_stock()
        cls.invalidar_faltantes()
        try:
            from .servicios_personas import HakuPersonasService
            HakuPersonasService.invalidar_cache_personas()
        except Exception:
            pass

    @classmethod
    def obtener_metricas(cls) -> Dict[str, int]:
        """Retorna las métricas acumuladas de aciertos y fallos de caché."""
        with cls._lock:
            return {
                "catalog_hits": cls._catalog_hits,
                "catalog_misses": cls._catalog_misses,
                "location_hits": cls._location_hits,
                "location_misses": cls._location_misses,
                "stock_hits": cls._stock_hits,
                "stock_misses": cls._stock_misses,
                "faltantes_hits": cls._faltantes_hits,
                "faltantes_misses": cls._faltantes_misses,
            }

    @classmethod
    def registrar_intencion_detectada(cls, tipo_intencion: str) -> None:
        """Incrementa el contador interno agregado de intenciones detectadas (sin guardar frases ni audios)."""
        if not tipo_intencion:
            return
        with cls._lock:
            cls._intenciones_detectadas[tipo_intencion] = cls._intenciones_detectadas.get(tipo_intencion, 0) + 1

    @classmethod
    def obtener_metricas_intenciones(cls) -> Dict[str, int]:
        """Retorna copia de los contadores agregados de intenciones detectadas."""
        with cls._lock:
            return dict(cls._intenciones_detectadas)

    @classmethod
    def reset_metricas(cls) -> None:
        with cls._lock:
            cls._catalog_hits = 0
            cls._catalog_misses = 0
            cls._location_hits = 0
            cls._location_misses = 0
            cls._stock_hits = 0
            cls._stock_misses = 0
            cls._faltantes_hits = 0
            cls._faltantes_misses = 0
            cls._intenciones_detectadas = {}


# ─────────────────────────────────────────────────────────────────────────────
# SEÑALES DE INVALIDACIÓN AUTOMÁTICA (POST-COMMIT ESTRICTO)
# ─────────────────────────────────────────────────────────────────────────────

def _on_producto_modificado(sender, instance, **kwargs):
    """Invalida catálogo y ubicaciones cuando un producto se crea, edita o archiva."""
    from django.db import transaction

    def _invalidar():
        HakuCacheManager.invalidar_catalogo()
        HakuCacheManager.invalidar_ubicaciones(instance.id)
        HakuCacheManager.invalidar_stock(instance.id)

    try:
        transaction.on_commit(_invalidar)
    except Exception:
        _invalidar()


def _on_estructura_catalogo_modificada(sender, instance, **kwargs):
    """Actualiza índices derivados cuando cambian categorías o almacenes."""
    from django.db import transaction

    def _invalidar():
        HakuCacheManager.invalidar_catalogo()

    try:
        transaction.on_commit(_invalidar)
    except Exception:
        _invalidar()


def _on_existencia_modificada(sender, instance, created=False, **kwargs):
    """Invalida stock y, si es una nueva existencia (nuevo almacén), invalida catálogo/índice."""
    from django.db import transaction
    prod_id = getattr(instance, "producto_id", None)
    almacen_id = getattr(instance, "almacen_id", None)

    def _invalidar():
        HakuCacheManager.invalidar_stock(prod_id, almacen_id)
        if created:
            HakuCacheManager.invalidar_catalogo()

    try:
        transaction.on_commit(_invalidar)
    except Exception:
        _invalidar()


def _on_existencia_eliminada(sender, instance, **kwargs):
    """Invalida catálogo/índice y stock cuando se elimina una asignación producto-almacén."""
    from django.db import transaction
    prod_id = getattr(instance, "producto_id", None)
    almacen_id = getattr(instance, "almacen_id", None)

    def _invalidar():
        HakuCacheManager.invalidar_stock(prod_id, almacen_id)
        HakuCacheManager.invalidar_catalogo()

    try:
        transaction.on_commit(_invalidar)
    except Exception:
        _invalidar()


def _on_movimiento_modificado(sender, instance, **kwargs):
    """Invalida stock tras movimientos o préstamos (sin reconstruir catálogo)."""
    from django.db import transaction
    prod_id = getattr(instance, "producto_id", None)
    almacen_id = getattr(instance, "almacen_id", None) or getattr(instance, "almacen_origen_id", None)
    almacen_dest_id = getattr(instance, "almacen_destino_id", None)

    def _invalidar():
        HakuCacheManager.invalidar_stock(prod_id, almacen_id)
        if almacen_dest_id:
            HakuCacheManager.invalidar_stock(prod_id, almacen_dest_id)

    try:
        transaction.on_commit(_invalidar)
    except Exception:
        _invalidar()


def _on_existencia_o_movimiento_modificado(sender, instance, **kwargs):
    """Compatibilidad: despacha a la señal adecuada según el sender."""
    from inventario.models import Existencia
    if sender == Existencia:
        _on_existencia_modificada(sender, instance, **kwargs)
    else:
        _on_movimiento_modificado(sender, instance, **kwargs)


def _on_usuario_modificado(sender, instance, **kwargs):
    """Invalida caché de faltantes y personas cuando cambian permisos, roles o estado de un usuario."""
    # Django actualiza ``last_login`` en cada autenticación mediante
    # ``save(update_fields=["last_login"])``. Eso no cambia permisos ni datos
    # de inventario, por lo que no debe vaciar los índices de lectura de Haku:
    # reconstruirlos en la primera consulta puede añadir cientos de ms.
    update_fields = kwargs.get("update_fields")
    if (
        not kwargs.get("created", False)
        and update_fields is not None
        and set(update_fields).issubset({"last_login"})
    ):
        return

    from django.db import transaction

    def _invalidar():
        HakuCacheManager.invalidar_faltantes()
        HakuCacheManager.invalidar_todo()

    try:
        transaction.on_commit(_invalidar)
    except Exception:
        _invalidar()


def _on_usuario_almacenes_changed(sender, instance, action, **kwargs):
    """Invalida efectivamente la caché de permisos y faltantes ante adición, remoción o limpieza de almacenes.
    
    Las operaciones ManyToMany (almacenes_asignados.add, .remove, .clear) disparan m2m_changed,
    NUNCA post_save de UsuarioAudacIA. Este receptor asegura que ninguna consulta posterior entregue
    información de almacenes revocados antes de vencer el TTL.
    """
    if action in ("post_add", "post_remove", "post_clear"):
        from django.db import transaction

        def _invalidar():
            HakuCacheManager.invalidar_faltantes()
            HakuCacheManager.invalidar_todo()

        try:
            transaction.on_commit(_invalidar)
        except Exception:
            _invalidar()


def _on_producto_ubicacion_modificado(sender, instance, **kwargs):
    """Invalida la caché de ubicaciones cuando se modifica o elimina una asignación física."""
    from django.db import transaction
    prod_id = getattr(instance, "producto_id", None)

    def _invalidar():
        HakuCacheManager.invalidar_ubicaciones(prod_id)

    try:
        transaction.on_commit(_invalidar)
    except Exception:
        _invalidar()


def _on_producto_ubicaciones_m2m_changed(sender, instance, action, **kwargs):
    """Invalida la caché de ubicaciones ante cambios ManyToMany en producto.ubicaciones."""
    if action in ("post_add", "post_remove", "post_clear"):
        from django.db import transaction
        prod_id = getattr(instance, "id", None) if hasattr(instance, "id") else None

        def _invalidar():
            HakuCacheManager.invalidar_ubicaciones(prod_id)

        try:
            transaction.on_commit(_invalidar)
        except Exception:
            _invalidar()


_senales_conectadas = False

def conectar_senales_cache():
    """Conecta los listeners de invalidación a los modelos de inventario y usuarios."""
    global _senales_conectadas
    if _senales_conectadas:
        return

    try:
        from inventario.models import Almacen, Categoria, Existencia, Movimiento, Prestamo, Producto, Ubicacion
        from cuentas.models import UsuarioAudacIA

        post_save.connect(_on_producto_modificado, sender=Producto, dispatch_uid="haku_prod_save")
        post_delete.connect(_on_producto_modificado, sender=Producto, dispatch_uid="haku_prod_del")

        # El índice semántico guarda la categoría junto con cada producto y el
        # alcance del catálogo depende de los almacenes activos. Sin señales,
        # esos datos pueden quedar obsoletos hasta vencer el TTL del índice.
        for modelo, clave in ((Categoria, "categoria"), (Almacen, "almacen")):
            post_save.connect(
                _on_estructura_catalogo_modificada,
                sender=modelo,
                dispatch_uid=f"haku_{clave}_save",
            )
            post_delete.connect(
                _on_estructura_catalogo_modificada,
                sender=modelo,
                dispatch_uid=f"haku_{clave}_del",
            )

        post_save.connect(_on_producto_ubicacion_modificado, sender=Ubicacion, dispatch_uid="haku_ubi_save")
        post_delete.connect(_on_producto_ubicacion_modificado, sender=Ubicacion, dispatch_uid="haku_ubi_del")

        # Conectar M2M si el modelo implementara relación ManyToMany en ubicaciones
        m2m_field = getattr(Producto, "ubicaciones", None)
        if hasattr(m2m_field, "through"):
            m2m_changed.connect(
                _on_producto_ubicaciones_m2m_changed,
                sender=m2m_field.through,
                dispatch_uid="haku_prod_ubi_m2m_changed",
            )

        post_save.connect(_on_existencia_modificada, sender=Existencia, dispatch_uid="haku_exist_save")
        post_delete.connect(_on_existencia_eliminada, sender=Existencia, dispatch_uid="haku_exist_del")
        post_save.connect(_on_movimiento_modificado, sender=Movimiento, dispatch_uid="haku_mov_save")
        post_save.connect(_on_movimiento_modificado, sender=Prestamo, dispatch_uid="haku_prest_save")

        post_save.connect(_on_usuario_modificado, sender=UsuarioAudacIA, dispatch_uid="haku_user_save")
        if hasattr(UsuarioAudacIA, "almacenes_asignados") and hasattr(UsuarioAudacIA.almacenes_asignados, "through"):
            m2m_changed.connect(
                _on_usuario_almacenes_changed,
                sender=UsuarioAudacIA.almacenes_asignados.through,
                dispatch_uid="haku_user_m2m_changed",
            )
        _senales_conectadas = True
        logger.info("Señales de invalidación de caché Haku conectadas con éxito")
    except Exception as exc:
        logger.warning("No se pudieron conectar las señales de caché Haku (%s)", exc)


