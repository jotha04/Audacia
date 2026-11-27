from uuid import uuid4

from django.contrib.auth import get_user_model
from django.test import TestCase

from cuentas.models import PermisoAudacIA
from inventario.models import Almacen, Categoria, Existencia, Producto
from .agente.planificador import _planificar_listado_global_explicito, planificar_determinista


class ConsultasCatalogoTests(TestCase):
    def test_variantes_y_consultas_especificas(self):
        for frase, operacion in (
            ("¿Hay algún producto?", "contar"),
            ("Hacu, ¿tenemos artículos registrados en el inventario?", "contar"),
            ("¿Existe alguna referencia?", "contar"),
            ("¿Cuántos productos hay?", "contar"),
            ("buscando productos", "listar"),
            ("¿Qué productos tenemos?", "listar"),
            ("¿Qué hay en el inventario?", "listar"),
            ("¿Hay algo registrado en el catálogo?", "contar"),
            ("¿Cuántas unidades hay en total?", "sumar_unidades"),
        ):
            with self.subTest(frase=frase):
                plan = planificar_determinista(frase)
                self.assertEqual(plan.operacion, operacion)
                self.assertIsNone(plan.filtros.condicion)
                self.assertEqual(plan.alcance_entidad, "global")
        for frase in (
            "¿Hay algún producto bajo el mínimo?", "¿Hay productos disponibles en la sala norte?",
            "¿Hay productos en la bodega norte?", "¿Hay algún producto llamado cable?",
            "¿Hay algún producto en préstamo a Juan?", "¿Hay algún producto con código CAB-123?",
        ):
            with self.subTest(frase=frase):
                self.assertIsNone(_planificar_listado_global_explicito(frase))

    def test_existencia_segun_catalogo_visible_incluso_con_saldo_cero(self):
        visible = Almacen.objects.create(nombre="Almacén asignado")
        otro = Almacen.objects.create(nombre="Otro almacén")
        usuario = get_user_model().objects.create_user(
            username="consulta", nombre_completo="Consulta", almacen_asignado=visible,
        )
        usuario.permisos_personalizados.add(PermisoAudacIA.objects.get(codigo="catalogo.ver"))
        self.client.force_login(usuario)
        categoria = Categoria.objects.create(nombre="Materiales", almacen_principal=otro)
        producto = Producto.objects.create(nombre="Cable", categoria=categoria)
        Existencia.objects.create(almacen=otro, producto=producto, stock_disponible=5)

        conversacion = str(uuid4())

        def consultar(frase):
            resp = self.client.post("/api/haku/mensaje/", {"mensaje": frase, "conversacion_id": conversacion}, content_type="application/json")
            self.assertEqual(resp.status_code, 200)
            return resp.json()

        vacio = consultar("hay algun producto")
        self.assertIn("No hay productos registrados", vacio["mensaje_verbal"])
        categoria_visible = Categoria.objects.create(nombre="Herramientas", almacen_principal=visible)
        martillo = Producto.objects.create(nombre="Martillo", categoria=categoria_visible)
        Existencia.objects.create(almacen=visible, producto=martillo, stock_disponible=0)
        consultar("¿Hay algún producto bajo el mínimo?")
        respuesta = consultar("¿Hay algún producto?")
        for campo in ("mensaje_pantalla", "mensaje_verbal"):
            self.assertIn("Hay un producto registrado", respuesta[campo])
            self.assertNotIn("mínimo", respuesta[campo])

        pinza = Producto.objects.create(nombre="Pinza", categoria=categoria_visible)
        Existencia.objects.create(almacen=visible, producto=pinza, stock_disponible=9)
        self.assertIn("Pinza", consultar("¿Qué hay en el inventario?")["mensaje_pantalla"])
        self.assertIn("2 productos", consultar("¿Y cuántos son?")["mensaje_verbal"])
        self.assertIn("9", consultar("¿Cuántas unidades hay en total?")["mensaje_verbal"])
        self.assertIn("1", consultar("¿Cuántos productos hay en stock?")["mensaje_pantalla"])
        self.assertIn("2 productos", consultar("¿Hay productos registrados?")["mensaje_verbal"])
        self.assertIn("préstamo", consultar("¿Hay algún producto en préstamo?")["mensaje_verbal"])
        self.assertIn("categoría", consultar("¿Hay categorías registradas?")["mensaje_verbal"])
