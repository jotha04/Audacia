# Generated for Haku integration: clave_canonica, es_prestable, huella_payload
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inventario', '0011_eliminar_productoubicacion'),
    ]

    operations = [
        migrations.AddField(
            model_name='ubicacion',
            name='clave_canonica',
            field=models.CharField(
                blank=True,
                db_index=True,
                default='',
                help_text='Clave normalizada para garantizar unicidad por almacén insensible a mayúsculas, tildes, números cardinales y espacios.',
                max_length=150,
                verbose_name='clave canónica',
            ),
        ),
        migrations.AddField(
            model_name='producto',
            name='es_prestable',
            field=models.BooleanField(
                default=False,
                help_text='Indica si el producto es retornable y admite préstamos. False para consumibles de uso definitivo (ej. tornillos, resmas de papel).',
                verbose_name='es prestable',
            ),
        ),
        migrations.AddField(
            model_name='movimiento',
            name='huella_payload',
            field=models.CharField(
                blank=True,
                default='',
                help_text='Huella SHA-256 del contenido semántico de la propuesta para validar idempotencia y detectar discrepancias de carga útil.',
                max_length=64,
                verbose_name='huella de payload',
            ),
        ),
    ]

