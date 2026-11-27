"""
Módulo de resolución y análisis de expresiones temporales en lenguaje natural
para el sistema de inventario AudacIA (HAKU Inventario).

Soporta expresiones temporales relativas y absolutas en español:
- "hoy"
- "ayer"
- "esta semana"
- "semana pasada"
- "semana pasada hasta hoy"
- "últimos N días" (ej. "últimos 7 días")
- "este mes"
- "mes pasado"
- Meses específicos (ej. "octubre", "noviembre", "enero")
- Intervalos explícitos (ej. "entre el lunes y el viernes", "desde el 1 de octubre hasta hoy")

Reglas semánticas:
- Zona horaria de referencia: America/Bogota (Colombia, UTC-5).
- Límites coherentes: inicio inclusivo (start_dt) y final exclusivo (end_dt).
"""

from __future__ import annotations

import calendar
import datetime
import re
from dataclasses import dataclass
from typing import Optional, Tuple
import zoneinfo

from django.utils import timezone

ZONA_BOGOTA = zoneinfo.ZoneInfo("America/Bogota")

DIAS_SEMANA_MAP = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "miércoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "sábado": 5,
    "domingo": 6,
}

MESES_MAP = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


@dataclass(frozen=True)
class IntervaloTemporal:
    """Intervalo de tiempo resuelto con límite inferior inclusivo y superior exclusivo."""
    etiqueta: str
    inicio: datetime.datetime
    fin: datetime.datetime
    descripcion_legible: str


def obtener_ahora_local(ref_dt: Optional[datetime.datetime] = None) -> datetime.datetime:
    """Retorna el datetime actual en la zona horaria de Bogotá."""
    if ref_dt is None:
        ref_dt = timezone.now()
    if timezone.is_aware(ref_dt):
        return ref_dt.astimezone(ZONA_BOGOTA)
    return ref_dt.replace(tzinfo=ZONA_BOGOTA)


def resolver_intervalo_temporal(
    texto: str,
    ref_dt: Optional[datetime.datetime] = None,
) -> Optional[IntervaloTemporal]:
    """Interpreta expresiones temporales en español y devuelve un IntervaloTemporal.
    
    Si no detecta ninguna expresión temporal, retorna None.
    """
    if not texto:
        return None

    ahora = obtener_ahora_local(ref_dt)
    hoy_date = ahora.date()
    texto_norm = texto.lower()

    # 1. "semana pasada hasta hoy" / "desde la semana pasada hasta hoy"
    if re.search(r"\b(?:desde\s+)?(?:la\s+)?semana\s+pasada\s+hasta\s+(?:hoy|la\s+fecha|ahora)\b", texto_norm):
        lunes_esta_semana = hoy_date - datetime.timedelta(days=hoy_date.weekday())
        lunes_semana_pasada = lunes_esta_semana - datetime.timedelta(days=7)
        inicio = datetime.datetime.combine(lunes_semana_pasada, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(hoy_date + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="SEMANA_PASADA_HASTA_HOY",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"desde la semana pasada ({lunes_semana_pasada.strftime('%d/%m/%Y')}) hasta hoy ({hoy_date.strftime('%d/%m/%Y')})",
        )

    # 2. "semana pasada" / "semana anterior" (semana calendario anterior completa: lunes a domingo)
    if re.search(r"\b(?:la\s+)?semana\s+(?:pasada|anterior)\b", texto_norm):
        lunes_esta_semana = hoy_date - datetime.timedelta(days=hoy_date.weekday())
        lunes_semana_pasada = lunes_esta_semana - datetime.timedelta(days=7)
        domingo_semana_pasada = lunes_semana_pasada + datetime.timedelta(days=6)
        inicio = datetime.datetime.combine(lunes_semana_pasada, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(domingo_semana_pasada + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="SEMANA_PASADA",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"la semana pasada ({lunes_semana_pasada.strftime('%d/%m/%Y')} al {domingo_semana_pasada.strftime('%d/%m/%Y')})",
        )

    # «La última semana» se refiere a los siete días recientes; «semana pasada»
    # arriba conserva el significado de semana calendario anterior.
    if re.search(r"\b(?:la\s+)?[uú]ltima\s+semana\b", texto_norm):
        inicio = ahora - datetime.timedelta(days=7)
        fin = ahora
        return IntervaloTemporal(
            etiqueta="ULTIMA_SEMANA",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"los últimos 7 días (desde {inicio.strftime('%d/%m/%Y')} hasta hoy)",
        )

    # 3. "esta semana" / "de la semana" (desde el lunes de la semana actual hasta el cierre del día actual)
    if re.search(
        r"\b(?:esta\s+semana|(?:de|en|durante)\s+la\s+semana(?!\s+(?:pasada|anterior))|semana\s+actual|semana\s+en\s+curso)\b",
        texto_norm,
    ):
        lunes_esta_semana = hoy_date - datetime.timedelta(days=hoy_date.weekday())
        inicio = datetime.datetime.combine(lunes_esta_semana, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(hoy_date + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="ESTA_SEMANA",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"esta semana (desde el lunes {lunes_esta_semana.strftime('%d/%m/%Y')} hasta hoy)",
        )

    # 4. "últimos N días" (ej. "últimos 7 días", "ultimos 15 dias", "últimos siete días")
    num_map = {
        "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
        "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
        "catorce": 14, "quince": 15, "veinte": 20, "treinta": 30, "sesenta": 60, "noventa": 90,
    }
    patron_numeros = r"\d+|" + "|".join(sorted(num_map.keys(), key=len, reverse=True))
    match_ultimos = re.search(rf"\b[uú]ltimos?\s+({patron_numeros})\s+d[ií]as\b", texto_norm)
    if match_ultimos:
        val_str = match_ultimos.group(1)
        num_dias = int(val_str) if val_str.isdigit() else num_map.get(val_str, 7)
        inicio = ahora - datetime.timedelta(days=num_dias)
        fin = ahora
        return IntervaloTemporal(
            etiqueta="ULTIMOS_DIAS",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"los últimos {num_dias} días (desde {inicio.strftime('%d/%m/%Y')} hasta hoy)",
        )

    # 4b. "hace N días" (día calendario exacto hace N días)
    match_hace_dias = re.search(rf"\bhace\s+({patron_numeros})\s+d[ií]as\b", texto_norm)
    if match_hace_dias:
        val_str = match_hace_dias.group(1)
        num_dias = int(val_str) if val_str.isdigit() else num_map.get(val_str, 1)
        fecha_obj = hoy_date - datetime.timedelta(days=num_dias)
        inicio = datetime.datetime.combine(fecha_obj, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(fecha_obj + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="HACE_DIAS",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"hace {num_dias} días ({fecha_obj.strftime('%d/%m/%Y')})",
        )

    # 5a. "anteayer" / "antier"
    if re.search(r"\b(?:anteayer|antier)\b", texto_norm):
        anteayer_date = hoy_date - datetime.timedelta(days=2)
        inicio = datetime.datetime.combine(anteayer_date, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(anteayer_date + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="ANTEAYER",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"anteayer ({anteayer_date.strftime('%d/%m/%Y')})",
        )

    # 5. "ayer"
    if re.search(r"\bayer\b", texto_norm):
        ayer_date = hoy_date - datetime.timedelta(days=1)
        inicio = datetime.datetime.combine(ayer_date, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(hoy_date, datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="AYER",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"ayer ({ayer_date.strftime('%d/%m/%Y')})",
        )

    # 6. "mes pasado" / "mes anterior" (mes calendario anterior completo)
    if re.search(r"\b(?:el\s+)?mes\s+(?:pasado|anterior)\b", texto_norm):
        primer_dia_este_mes = hoy_date.replace(day=1)
        ultimo_dia_mes_pasado = primer_dia_este_mes - datetime.timedelta(days=1)
        primer_dia_mes_pasado = ultimo_dia_mes_pasado.replace(day=1)
        inicio = datetime.datetime.combine(primer_dia_mes_pasado, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(primer_dia_este_mes, datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="MES_PASADO",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"el mes pasado ({primer_dia_mes_pasado.strftime('%B %Y')})",
        )

    # 7. "este mes" / "mes actual" / "del mes"
    if re.search(
        r"\b(?:este\s+mes|mes\s+actual|(?:del|en\s+el|durante\s+el)\s+mes(?!\s+(?:pasado|anterior|de\b))|mes\s+en\s+curso)\b",
        texto_norm,
    ):
        primer_dia_este_mes = hoy_date.replace(day=1)
        inicio = datetime.datetime.combine(primer_dia_este_mes, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(hoy_date + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="ESTE_MES",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"este mes (desde el 01/{primer_dia_este_mes.strftime('%m/%Y')} hasta hoy)",
        )

    # 7b. "este año" / "año actual" / "del año" y "año pasado" / "año anterior"
    if re.search(
        r"\b(?:este\s+a[ñn]o|a[ñn]o\s+actual|(?:del|en\s+el|durante\s+el)\s+a[ñn]o(?!\s+(?:pasado|anterior|\d{4}))|a[ñn]o\s+en\s+curso)\b",
        texto_norm,
    ):
        inicio = datetime.datetime(hoy_date.year, 1, 1, 0, 0, 0, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(hoy_date + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="ESTE_ANIO",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"este año ({hoy_date.year})",
        )
    if re.search(r"\b(?:el\s+)?a[ñn]o\s+(?:pasado|anterior)\b", texto_norm):
        anio_ant = hoy_date.year - 1
        inicio = datetime.datetime(anio_ant, 1, 1, 0, 0, 0, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime(hoy_date.year, 1, 1, 0, 0, 0, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="ANIO_PASADO",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"el año pasado ({anio_ant})",
        )

    # 7c. Rangos de fechas explícitos y fechas puntuales (antes de evaluar el mes completo)
    nombres_meses_regex = "|".join(MESES_MAP.keys())

    # Rango ISO: "2026-11-01 a 2026-11-15", "entre 2026-11-01 y 2026-11-15", "desde 2026-11-01 hasta 2026-11-15"
    match_rango_iso = re.search(
        r"\b(?:desde\s+(?:el\s+)?|entre\s+(?:el\s+)?|del\s+)?(\d{4}-\d{2}-\d{2})\s+(?:hasta(?:\s+el)?|al|a|y(?:\s+el)?)\s+(\d{4}-\d{2}-\d{2})\b",
        texto_norm,
    )
    if match_rango_iso:
        try:
            f1 = datetime.date.fromisoformat(match_rango_iso.group(1))
            f2 = datetime.date.fromisoformat(match_rango_iso.group(2))
            if f1 > f2:
                f1, f2 = f2, f1
            inicio = datetime.datetime.combine(f1, datetime.time.min, tzinfo=ZONA_BOGOTA)
            fin = datetime.datetime.combine(f2 + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
            return IntervaloTemporal(
                etiqueta="RANGO_FECHAS",
                inicio=inicio,
                fin=fin,
                descripcion_legible=f"del {f1.strftime('%d/%m/%Y')} al {f2.strftime('%d/%m/%Y')}",
            )
        except ValueError:
            pass

    # "desde el <día> de <mes> (de <año>)? hasta hoy"
    match_desde_hasta_hoy = re.search(
        rf"\bdesde\s+(?:el\s+)?(\d{{1,2}})\s+de\s+({nombres_meses_regex})(?:\s+(?:de|del)\s+(\d{{4}}))?\s+hasta\s+(?:hoy|la\s+fecha|ahora)\b",
        texto_norm,
    )
    if match_desde_hasta_hoy:
        dia_ini = int(match_desde_hasta_hoy.group(1))
        mes_ini = MESES_MAP[match_desde_hasta_hoy.group(2)]
        anio_ini = int(match_desde_hasta_hoy.group(3)) if match_desde_hasta_hoy.group(3) else hoy_date.year
        try:
            f_ini = datetime.date(anio_ini, mes_ini, dia_ini)
            if f_ini <= hoy_date:
                inicio = datetime.datetime.combine(f_ini, datetime.time.min, tzinfo=ZONA_BOGOTA)
                fin = datetime.datetime.combine(hoy_date + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
                return IntervaloTemporal(
                    etiqueta="DESDE_FECHA_HASTA_HOY",
                    inicio=inicio,
                    fin=fin,
                    descripcion_legible=f"desde el {f_ini.strftime('%d/%m/%Y')} hasta hoy ({hoy_date.strftime('%d/%m/%Y')})",
                )
        except ValueError:
            pass

    # "entre el <d1> de <mes1> y el <d2> de <mes2>" / "desde el <d1> de <mes1> hasta el <d2> de <mes2>"
    match_rango_meses = re.search(
        rf"\b(?:desde|entre|del)\s+(?:el\s+)?(\d{{1,2}})\s+de\s+({nombres_meses_regex})(?:\s+(?:de|del)\s+(\d{{4}}))?\s+"
        rf"(?:hasta|y|al|a)\s+(?:el\s+)?(\d{{1,2}})\s+de\s+({nombres_meses_regex})(?:\s+(?:de|del)\s+(\d{{4}}))?\b",
        texto_norm,
    )
    if match_rango_meses:
        d1, m1_str, a1_str, d2, m2_str, a2_str = match_rango_meses.groups()
        anio2 = int(a2_str) if a2_str else (int(a1_str) if a1_str else hoy_date.year)
        anio1 = int(a1_str) if a1_str else anio2
        try:
            f1 = datetime.date(anio1, MESES_MAP[m1_str], int(d1))
            f2 = datetime.date(anio2, MESES_MAP[m2_str], int(d2))
            if f1 > f2:
                f1, f2 = f2, f1
            inicio = datetime.datetime.combine(f1, datetime.time.min, tzinfo=ZONA_BOGOTA)
            fin = datetime.datetime.combine(f2 + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
            return IntervaloTemporal(
                etiqueta="RANGO_FECHAS",
                inicio=inicio,
                fin=fin,
                descripcion_legible=f"del {f1.strftime('%d/%m/%Y')} al {f2.strftime('%d/%m/%Y')}",
            )
        except ValueError:
            pass

    # "del <d1> al <d2> de <mes> (de <año>)?" / "entre el <d1> y el <d2> de <mes>"
    match_rango_mismo_mes = re.search(
        rf"\b(?:del|entre\s+el|desde\s+el)\s+(\d{{1,2}})\s+(?:al|y\s+el|hasta\s+el)\s+(\d{{1,2}})\s+de\s+({nombres_meses_regex})(?:\s+(?:de|del)\s+(\d{{4}}))?\b",
        texto_norm,
    )
    if match_rango_mismo_mes:
        d1, d2, m_str, a_str = match_rango_mismo_mes.groups()
        anio = int(a_str) if a_str else hoy_date.year
        try:
            f1 = datetime.date(anio, MESES_MAP[m_str], int(d1))
            f2 = datetime.date(anio, MESES_MAP[m_str], int(d2))
            if f1 > f2:
                f1, f2 = f2, f1
            inicio = datetime.datetime.combine(f1, datetime.time.min, tzinfo=ZONA_BOGOTA)
            fin = datetime.datetime.combine(f2 + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
            return IntervaloTemporal(
                etiqueta="RANGO_FECHAS",
                inicio=inicio,
                fin=fin,
                descripcion_legible=f"del {f1.strftime('%d/%m/%Y')} al {f2.strftime('%d/%m/%Y')}",
            )
        except ValueError:
            pass

    # Fecha puntual: "el <día> de <mes> (de <año>)?" / "del <día> de <mes>"
    match_fecha_puntual = re.search(
        rf"\b(?:(?:el|del)(?:\s+d[ií]a)?|d[ií]a)\s+(\d{{1,2}})\s+de\s+({nombres_meses_regex})(?:\s+(?:de|del)\s+(\d{{4}}))?\b",
        texto_norm,
    )
    if match_fecha_puntual:
        dia_p, mes_p_str, anio_p_str = match_fecha_puntual.groups()
        anio_p = int(anio_p_str) if anio_p_str else hoy_date.year
        try:
            f_puntual = datetime.date(anio_p, MESES_MAP[mes_p_str], int(dia_p))
            inicio = datetime.datetime.combine(f_puntual, datetime.time.min, tzinfo=ZONA_BOGOTA)
            fin = datetime.datetime.combine(f_puntual + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
            return IntervaloTemporal(
                etiqueta="FECHA_ESPECIFICA",
                inicio=inicio,
                fin=fin,
                descripcion_legible=f"el {f_puntual.strftime('%d/%m/%Y')}",
            )
        except ValueError:
            pass

    # 8. Mes específico por nombre (ej. "octubre", "en octubre", "de octubre", "en octubre de 2026")
    for nombre_mes, num_mes in MESES_MAP.items():
        patron_mes = rf"\b(?:en|de|del\s+mes\s+de|durante|para|mes\s+de)?\s*{nombre_mes}(?:\s+(?:de|del)?\s*(\d{{4}}))?\b"
        match_mes = re.search(patron_mes, texto_norm)
        if match_mes:
            anio_match = match_mes.group(1)
            anio = int(anio_match) if anio_match else hoy_date.year
            inicio = datetime.datetime(anio, num_mes, 1, 0, 0, 0, tzinfo=ZONA_BOGOTA)
            if anio == hoy_date.year and num_mes == hoy_date.month:
                fin = datetime.datetime.combine(hoy_date + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
            else:
                mes_sig = 1 if num_mes == 12 else num_mes + 1
                anio_sig = anio + 1 if num_mes == 12 else anio
                fin = datetime.datetime(anio_sig, mes_sig, 1, 0, 0, 0, tzinfo=ZONA_BOGOTA)
            return IntervaloTemporal(
                etiqueta="MES_ESPECIFICO",
                inicio=inicio,
                fin=fin,
                descripcion_legible=f"{nombre_mes.capitalize()} de {anio}",
            )

    # 9. Intervalo explícito entre días de la semana: "entre el <dia1> y el <dia2>" / "del <dia1> al <dia2>"
    match_dias = re.search(
        r"\b(?:entre\s+(?:el\s+)?([a-záéíóú]+)\s+y\s+(?:el\s+)?([a-záéíóú]+)|del\s+([a-záéíóú]+)\s+al\s+([a-záéíóú]+))\b",
        texto_norm,
    )
    if match_dias:
        d1_str = match_dias.group(1) or match_dias.group(3)
        d2_str = match_dias.group(2) or match_dias.group(4)
        if d1_str in DIAS_SEMANA_MAP and d2_str in DIAS_SEMANA_MAP:
            idx1 = DIAS_SEMANA_MAP[d1_str]
            idx2 = DIAS_SEMANA_MAP[d2_str]
            lunes_actual = hoy_date - datetime.timedelta(days=hoy_date.weekday())
            fecha_d1 = lunes_actual + datetime.timedelta(days=idx1)
            fecha_d2 = lunes_actual + datetime.timedelta(days=idx2)
            if fecha_d1 > fecha_d2:
                fecha_d1, fecha_d2 = fecha_d2, fecha_d1
            inicio = datetime.datetime.combine(fecha_d1, datetime.time.min, tzinfo=ZONA_BOGOTA)
            fin = datetime.datetime.combine(fecha_d2 + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
            return IntervaloTemporal(
                etiqueta="ENTRE_DIAS",
                inicio=inicio,
                fin=fin,
                descripcion_legible=f"entre {d1_str} ({fecha_d1.strftime('%d/%m')}) y {d2_str} ({fecha_d2.strftime('%d/%m')})",
            )

    # 10. "hoy", "del día", "de hoy", "en la jornada"
    if re.search(r"\b(?:hoy|del?\s+d[ií]a|en\s+el\s+d[ií]a|esta\s+jornada|en\s+la\s+jornada|este\s+turno)\b", texto_norm):
        inicio = datetime.datetime.combine(hoy_date, datetime.time.min, tzinfo=ZONA_BOGOTA)
        fin = datetime.datetime.combine(hoy_date + datetime.timedelta(days=1), datetime.time.min, tzinfo=ZONA_BOGOTA)
        return IntervaloTemporal(
            etiqueta="HOY",
            inicio=inicio,
            fin=fin,
            descripcion_legible=f"hoy ({hoy_date.strftime('%d/%m/%Y')})",
        )

    return None


def limpiar_expresiones_temporales(texto: str) -> str:
    """Retira expresiones de período temporal para evitar que contaminen el filtro de producto."""
    if not texto:
        return ""
    nombres_meses_regex = "|".join(MESES_MAP.keys())
    dias_semana_regex = "|".join(DIAS_SEMANA_MAP.keys())
    patrones = (
        r"\b(?:durante|en|de|desde)?\s*(?:la\s+)?semana\s+pasada(?:\s+hasta\s+(?:hoy|la\s+fecha|ahora))?\b",
        r"\b(?:durante|en|de|desde)?\s*(?:la\s+)?semana\s+anterior\b",
        r"\b(?:durante|en|de)?\s*(?:la\s+)?[uú]ltima\s+semana\b",
        r"\b(?:durante|en|de)?\s*(?:esta\s+semana|(?:de|en|durante)\s+la\s+semana(?!\s+(?:pasada|anterior))|semana\s+actual|semana\s+en\s+curso)\b",
        r"\b(?:durante|en)?\s*(?:los\s+)?[uú]ltimos?\s+(?:\d+|un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|veinte|treinta|sesenta|noventa)\s+d[ií]as\b",
        r"\bhace\s+(?:\d+|un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|quince|treinta)\s+d[ií]as\b",
        r"\b(?:anteayer|antier|ayer|hoy|(?:del?|en\s+el)\s+d[ií]a(?!\s+\d)|esta\s+jornada|en\s+la\s+jornada|este\s+turno)\b",
        r"\b(?:durante|en|de|del)?\s*(?:el\s+)?mes\s+(?:pasado|anterior|actual|en\s+curso)\b",
        r"\b(?:(?:durante|en|de)\s+este\s+mes|este\s+mes|(?:del|en\s+el|durante\s+el)\s+mes(?!\s+(?:pasado|anterior|de\b)))\b",
        r"\b(?:durante|en|de|del)?\s*(?:el\s+|este\s+)?a[ñn]o(?:\s+(?:pasado|anterior|actual|en\s+curso))?\b",
        rf"\bdesde\s+(?:el\s+)?\d{{1,2}}\s+de\s+(?:{nombres_meses_regex})(?:\s+(?:de|del)\s+\d{{4}})?\s+hasta\s+(?:hoy|la\s+fecha|ahora)\b",
        rf"\b(?:desde|entre|del)\s+(?:el\s+)?\d{{1,2}}\s+de\s+(?:{nombres_meses_regex})(?:\s+(?:de|del)\s+\d{{4}})?\s+(?:hasta|y|al|a)\s+(?:el\s+)?\d{{1,2}}\s+de\s+(?:{nombres_meses_regex})(?:\s+(?:de|del)\s+\d{{4}})?\b",
        rf"\b(?:del|entre\s+el|desde\s+el)\s+\d{{1,2}}\s+(?:al|y\s+el|hasta\s+el)\s+\d{{1,2}}\s+de\s+(?:{nombres_meses_regex})(?:\s+(?:de|del)\s+\d{{4}})?\b",
        rf"\b(?:(?:el|del)(?:\s+d[ií]a)?|d[ií]a)\s+\d{{1,2}}\s+de\s+(?:{nombres_meses_regex})(?:\s+(?:de|del)\s+\d{{4}})?\b",
        rf"\b(?:en|de|del\s+mes\s+de|durante|para|mes\s+de)\s+(?:{nombres_meses_regex})(?:\s+(?:de|del)\s+\d{{4}})?\b",
        rf"\b(?:entre\s+(?:el\s+)?(?:{dias_semana_regex})\s+y\s+(?:el\s+)?(?:{dias_semana_regex})|del\s+(?:{dias_semana_regex})\s+al\s+(?:{dias_semana_regex}))\b",
    )
    resultado = texto
    for patron in patrones:
        resultado = re.sub(patron, " ", resultado, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", resultado).strip()

