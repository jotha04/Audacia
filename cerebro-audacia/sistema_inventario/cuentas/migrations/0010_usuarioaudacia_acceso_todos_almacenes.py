# Generated for Haku integration: acceso_todos_almacenes
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cuentas', '0009_agregar_permiso_movimientos_revertir'),
    ]

    operations = [
        migrations.AddField(
            model_name='usuarioaudacia',
            name='acceso_todos_almacenes',
            field=models.BooleanField(
                default=False,
                help_text='Permite operar o consultar en todos los almacenes sin conceder privilegios de administrador. Solo puede ser concedido o revocado por Administradores o Superusuarios.',
                verbose_name='acceso a todos los almacenes',
            ),
        ),
    ]

