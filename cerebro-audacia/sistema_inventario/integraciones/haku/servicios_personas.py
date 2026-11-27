"""Servicio de resolución, normalización y validación canónica de personas y destinatarios en HAKU Inventario.

REGLAS INNEGOCIABLES:
1. Jamás registrar automáticamente nombres mal reconocidos o ambiguos por voz.
2. Si no hay certidumbre suficiente o el nombre está ausente, solicitar expresamente que el usuario lo escriba.
3. Preservar caracteres canónicos (respetar la 'ñ', tildes y mayúsculas oficiales) en el almacenamiento final.
4. Comparación tolerante (Unicode NFD, minúsculas, sin tildes para match) sin alterar el valor guardado.
5. Resolución contra personas reales: usuarios del sistema (UsuarioAudacIA) y destinatarios canónicos históricos.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from django.core.cache import caches

logger = logging.getLogger("integraciones.haku.personas")

CACHE_ALIAS = "haku_catalogo"
KEY_CACHE_PERSONAS = "personas:canonicas:lista"
TTL_PERSONAS = 900  # 15 minutos


def normalizar_para_comparacion(texto: str) -> str:
    """Limpia y estandariza un nombre exclusivamente para fines de búsqueda y comparación.
    
    - Convierte a minúsculas.
    - Descompone caracteres Unicode (NFD) eliminando marcas de tilde diacríticas,
      PERO PRESERVA la letra 'ñ' y 'Ñ' para no confundir identidades (ej. Peña vs Pena).
    - Remueve signos de puntuación y espacios redundantes.
    """
    if not texto:
        return ""
    
    t = texto.strip().lower()
    
    # Preservar la ñ reemplazándola temporalmente si se fuera a degradar
    t = t.replace("ñ", "___enie___")
    # Descomponer y quitar marcas diacríticas (tildes)
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    # Restaurar la ñ
    t = t.replace("___enie___", "ñ")
    
    # Quitar caracteres que no sean letras, espacios o números
    t = re.sub(r"[^\w\sñ]", " ", t)
    # Reducir espacios múltiples
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clave_fonetica_espanol(texto: str) -> str:
    """Genera una clave fonética española sin espacios para tolerar variantes STT de nombres propios.

    Ejemplos:
    - "Giancarlos Hernández" <-> "Yan Carlos Hernández" -> "yancarlosernandes"
    - "Jhon" <-> "John", "Cristian" <-> "Christian", "Vanesa" <-> "Vanessa"
    Preserva la 'ñ' para no mezclar identidades distintas (Peña vs Pena).
    """
    norm = normalizar_para_comparacion(texto)
    if not norm:
        return ""
    s = norm.replace(" ", "")
    # "chr" (Christian -> Cristian) antes de proteger "ch"
    s = s.replace("chr", "cr")
    s = s.replace("ch", "___ch___")
    s = s.replace("nh", "___nh___")
    s = s.replace("ph", "f")
    s = s.replace("qu", "c")
    s = s.replace("k", "c")
    s = s.replace("ll", "y")
    # Anglicismos fonéticos comunes en nombres hispanos (Jhan -> Yan, Yeison -> Jason)
    s = re.sub(r"jh(?=[aei])", "y", s)
    s = re.sub(r"(?:yei|jei)", "ja", s)
    s = re.sub(r"gi(?=[aou])", "y", s)
    s = re.sub(r"ge(?=[aou])", "y", s)
    s = re.sub(r"g(?=[ei])", "j", s)
    s = re.sub(r"j(?=[ei])", "j", s)
    s = re.sub(r"c(?=[ei])", "s", s)
    s = s.replace("z", "s")
    s = s.replace("v", "b")
    s = s.replace("h", "")
    s = s.replace("___ch___", "ch")
    s = s.replace("___nh___", "nh")
    # Colapsar letras repetidas consecutivas (ss -> s, nn -> n, rr -> r)
    s = re.sub(r"([a-zñ])\1+", r"\1", s)
    return s


def _tiene_conflicto_enie_tokens(toks_a: List[str], toks_b: List[str]) -> bool:
    """Detecta si dos listas de tokens difieren precisamente en 'ñ' vs 'n' (ej. Pena vs Peña)."""
    for ta in toks_a:
        if len(ta) < 3:
            continue
        for tb in toks_b:
            if len(tb) < 3 or ("ñ" in ta) == ("ñ" in tb):
                continue
            ta_n = ta.replace("ñ", "n")
            tb_n = tb.replace("ñ", "n")
            if ta_n == tb_n or SequenceMatcher(None, ta_n, tb_n).ratio() >= 0.85:
                return True
    return False


def emparejar_nombre_contra_candidatos(
    texto: str,
    candidatos: List[str],
    umbral: float = 0.84,
    margen_unico: float = 0.12,
) -> Optional[str]:
    """Empareja una frase o nombre transcrito contra una lista de candidatos activos (ej. préstamos).

    Usa coincidencia normalizada y similitud fonética sobre ventanas de palabras para soportar
    variantes STT como "Giancarlos Hernández" -> "Yan Carlos Hernández" sin hardcoding.
    """
    if not texto or not candidatos:
        return None
    norm_txt = normalizar_para_comparacion(texto)
    if len(norm_txt) < 2:
        return None

    # 1. Coincidencia exacta o por inclusión de nombre completo
    exactos = [c for c in candidatos if normalizar_para_comparacion(c) and normalizar_para_comparacion(c) in norm_txt]
    if len(exactos) == 1:
        return exactos[0]

    # 1b. Coincidencia única por subsecuencia de 2+ tokens (normalizada o fonética)
    # Ej. "Yan Carlos quiere devolver..." -> "Yan Carlos Hernández"
    #     "Giancarlos quiere devolver..." -> "Yan Carlos Hernández" (fonética "yancarlos")
    #     "de Carlos Mendoza" -> "Profesor Carlos Mendoza"
    titulos_comunes = {
        "profesor", "profesora", "profe", "prof", "docente",
        "ingeniero", "ingeniera", "ing", "doctor", "doctora", "doc", "dr", "dra",
        "licenciado", "licenciada", "lic", "sr", "sra", "don", "dona", "doña",
    }
    palabras_txt = norm_txt.split()
    ventanas_1_2: List[Tuple[str, str]] = []
    for w_len in (1, 2, 3):
        for i in range(0, len(palabras_txt) - w_len + 1):
            frag = " ".join(palabras_txt[i : i + w_len])
            if len(frag) >= 4:
                ventanas_1_2.append((frag, clave_fonetica_espanol(frag)))

    cands_sub_2: List[str] = []
    for cand in candidatos:
        cand_norm = normalizar_para_comparacion(cand)
        if _tiene_conflicto_enie_tokens(palabras_txt, cand_norm.split()):
            continue
        toks_cand = [t for t in cand_norm.split() if len(t) >= 2 and t not in titulos_comunes]
        if len(toks_cand) >= 2:
            subfrases = [" ".join(toks_cand[i : i + 2]) for i in range(len(toks_cand) - 1)]
            if len(toks_cand) >= 3:
                subfrases.append(f"{toks_cand[0]} {toks_cand[-1]}")
            coincide = False
            for sf in subfrases:
                if sf in norm_txt:
                    coincide = True
                    break
                sf_fon = clave_fonetica_espanol(sf)
                if len(sf_fon) >= 6:
                    for v_norm, v_fon in ventanas_1_2:
                        if ("ñ" in v_norm) != ("ñ" in sf):
                            continue
                        if v_fon == sf_fon or SequenceMatcher(None, v_fon, sf_fon).ratio() >= 0.90:
                            coincide = True
                            break
                if coincide:
                    break
            if coincide:
                cands_sub_2.append(cand)
    if len(cands_sub_2) == 1:
        return cands_sub_2[0]

    # 1c. Coincidencia única por token distintivo de nombre/apellido (ej. "Yan quiere devolver..." -> "Yan Carlos Hernández")
    # Solo se activa si la entrada NO contradice al candidato con otros nombres propios (ej. "Yan Carlos" NO debe empatar "Profesor Carlos Mendoza").
    stopwords_esp = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en", "con", "sin",
        "por", "para", "que", "quiere", "quiero", "va", "viene", "necesita", "desea", "acaba", "ya", "hoy",
        "devolver", "devolvio", "devolvieron", "devolucion", "devuelve", "devuelto",
        "regreso", "regresar", "retorno", "retornar", "reintegrar", "trajo", "trajeron", "trae", "entrego",
        "prestar", "prestamo", "producto", "articulo", "equipo", "todo", "inventario", "completo",
        "llevarse", "cerrar", "registra", "registrar", "unidades", "unidad", "observacion", "nota", "comentario",
    }
    toks_txt_sig = [t for t in palabras_txt if len(t) >= 3 and t not in titulos_comunes and t not in stopwords_esp]
    if toks_txt_sig:
        cands_tok_unico: List[str] = []
        for cand in candidatos:
            cand_norm_1c = normalizar_para_comparacion(cand)
            if _tiene_conflicto_enie_tokens(palabras_txt, cand_norm_1c.split()):
                continue
            toks_c = [t for t in cand_norm_1c.split() if len(t) >= 3 and t not in titulos_comunes]
            toks_c_set = set(toks_c)
            toks_c_fon = {clave_fonetica_espanol(tc): tc for tc in toks_c if len(clave_fonetica_espanol(tc)) >= 3}

            def _token_coincide_cand(tok: str) -> bool:
                if tok in toks_c_set:
                    return True
                t_fon = clave_fonetica_espanol(tok)
                if t_fon in toks_c_fon:
                    tc_orig = toks_c_fon[t_fon]
                    if ("ñ" in tok) == ("ñ" in tc_orig):
                        return True
                return False

            posiciones_match = [
                idx for idx, t in enumerate(palabras_txt)
                if len(t) >= 3 and t not in titulos_comunes and t not in stopwords_esp and _token_coincide_cand(t)
            ]
            if not posiciones_match:
                continue
            # Rechazar si un token coincidente está pegado a otro nombre propio distinto que contradice al candidato
            # (ej. "Yan Carlos" NO debe empatar con "Profesor Carlos Mendoza" solo por "Carlos").
            contradicho = False
            for idx_m in posiciones_match:
                for adj_idx in (idx_m - 1, idx_m + 1):
                    if 0 <= adj_idx < len(palabras_txt):
                        adj_t = palabras_txt[adj_idx]
                        if (
                            len(adj_t) >= 3
                            and not _token_coincide_cand(adj_t)
                            and adj_t not in titulos_comunes
                            and adj_t not in stopwords_esp
                        ):
                            contradicho = True
                            break
                if contradicho:
                    break
            if contradicho:
                continue
            cands_tok_unico.append(cand)
        if len(cands_tok_unico) == 1:
            return cands_tok_unico[0]
        if len(cands_tok_unico) > 1:
            # Empate real por nombre/apellido compartido entre múltiples candidatos activos
            return None

    # 2. Similitud fonética contra ventanas deslizantes de palabras en el texto
    palabras = norm_txt.split()
    ventanas: List[str] = [norm_txt]
    max_w = min(len(palabras), 5)
    for w_len in range(1, max_w + 1):
        for i in range(0, len(palabras) - w_len + 1):
            ventanas.append(" ".join(palabras[i : i + w_len]))
    ventanas_fon = [(v, clave_fonetica_espanol(v)) for v in set(ventanas) if len(v) >= 3]

    puntajes: List[Tuple[float, str]] = []
    for cand in candidatos:
        cand_norm = normalizar_para_comparacion(cand)
        if _tiene_conflicto_enie_tokens(palabras, cand_norm.split()):
            continue
        cand_fon = clave_fonetica_espanol(cand)
        if not cand_fon:
            continue
        mejor_cand = 0.0
        for v_norm, v_fon in ventanas_fon:
            if not v_fon:
                continue
            # Proteger distinción ñ vs n
            if ("ñ" in v_norm) != ("ñ" in cand_norm):
                if SequenceMatcher(None, v_norm.replace("ñ", "n"), cand_norm.replace("ñ", "n")).ratio() > 0.85:
                    continue
            ratio_fon = SequenceMatcher(None, v_fon, cand_fon).ratio()
            ratio_str = SequenceMatcher(None, v_norm, cand_norm).ratio()
            score = max(ratio_fon, ratio_str)
            if score > mejor_cand:
                mejor_cand = score
        puntajes.append((mejor_cand, cand))

    puntajes.sort(key=lambda x: x[0], reverse=True)
    if not puntajes:
        return None
    top_score, top_cand = puntajes[0]
    if top_score < umbral:
        return None
    if len(puntajes) > 1 and (top_score - puntajes[1][0]) < margen_unico:
        return None
    return top_cand


def capitalizar_nombre_propio(texto: str) -> str:
    """Formatea estéticamente un nombre limpio respetando nombres compuestos,
    partículas ("de", "del", "de la", "de los"), guiones, apóstrofes y tildes.
    """
    if not texto:
        return ""
    palabras = texto.strip().split()
    particulas_menores = {"de", "del", "la", "las", "los", "y", "e"}
    res = []
    for i, p in enumerate(palabras):
        p_lower = p.lower()
        if i > 0 and p_lower in particulas_menores:
            res.append(p_lower)
        else:
            if "-" in p:
                res.append("-".join(sub.capitalize() for sub in p.split("-")))
            elif "'" in p:
                res.append("'".join(sub.capitalize() for sub in p.split("'")))
            else:
                res.append(p.capitalize())
    return " ".join(res)


class HakuPersonasService:
    """Servicio para resolución y validación de personas destinatarias."""

    @classmethod
    def obtener_personas_canonicas(cls) -> List[Dict[str, Any]]:
        """Obtiene la lista consolidada de destinatarios canónicos conocidos en el sistema.
        
        Fuentes:
        1. Usuarios del sistema (UsuarioAudacIA activos).
        2. Destinatarios registrados previamente en préstamos cerrados o activos.
        """
        try:
            cache = caches[CACHE_ALIAS]
            cached = cache.get(KEY_CACHE_PERSONAS)
            if cached is not None:
                return cached
        except Exception:
            cache = None

        personas_map: Dict[str, Dict[str, Any]] = {}

        # 1. Usuarios del sistema
        try:
            from cuentas.models import UsuarioAudacIA
            for u in UsuarioAudacIA.objects.filter(is_active=True, eliminado_logicamente=False):
                nombre_oficial = getattr(u, "nombre_completo", "") or u.get_full_name() or u.username
                norm = normalizar_para_comparacion(nombre_oficial)
                if norm and norm not in personas_map:
                    personas_map[norm] = {
                        "nombre_canonico": nombre_oficial.strip(),
                        "norm": norm,
                        "fon": clave_fonetica_espanol(nombre_oficial),
                        "tipo": "USUARIO_SISTEMA",
                        "username": u.username,
                    }
        except Exception as exc:
            logger.warning("Error leyendo usuarios para personas canónicas: %s", exc)

        # 2. Préstamos históricos (incluye personas sin cuenta de usuario registrada)
        try:
            from inventario.models import Prestamo
            destinatarios_db = (
                Prestamo.objects.exclude(destinatario__isnull=True)
                .exclude(destinatario="")
                .values_list("destinatario", flat=True)
                .distinct()[:200]
            )
            for dest in destinatarios_db:
                norm = normalizar_para_comparacion(dest)
                if norm and norm not in personas_map:
                    canon = capitalizar_nombre_propio(dest)
                    personas_map[norm] = {
                        "nombre_canonico": canon,
                        "norm": norm,
                        "fon": clave_fonetica_espanol(canon),
                        "tipo": "DESTINATARIO_HISTORICO",
                    }
        except Exception as exc:
            logger.warning("Error leyendo destinatarios históricos de préstamos: %s", exc)

        # 3. Destinatarios históricos de movimientos/salidas (personas externas sin cuenta)
        try:
            from inventario.models import Movimiento
            destinatarios_mov = (
                Movimiento.objects.exclude(destinatario__isnull=True)
                .exclude(destinatario="")
                .order_by("-fecha_hora")
                .values_list("destinatario", flat=True)[:250]
            )
            for dest in destinatarios_mov:
                norm = normalizar_para_comparacion(dest)
                if norm and norm not in personas_map:
                    canon = capitalizar_nombre_propio(dest)
                    personas_map[norm] = {
                        "nombre_canonico": canon,
                        "norm": norm,
                        "fon": clave_fonetica_espanol(canon),
                        "tipo": "DESTINATARIO_MOVIMIENTO",
                    }
        except Exception as exc:
            logger.warning("Error leyendo destinatarios históricos de movimientos: %s", exc)

        lista = list(personas_map.values())
        if cache:
            try:
                cache.set(KEY_CACHE_PERSONAS, lista, timeout=TTL_PERSONAS)
            except Exception:
                pass
        return lista

    @classmethod
    def invalidar_cache_personas(cls) -> None:
        try:
            cache = caches[CACHE_ALIAS]
            cache.delete(KEY_CACHE_PERSONAS)
        except Exception:
            pass

    @classmethod
    def resolver_destinatario(
        cls,
        texto_destinatario: str,
        origen_voz: bool = False
    ) -> Tuple[Optional[str], float, List[str], bool]:
        """Resuelve un destinatario a partir del texto ingresado o transcrito.
        
        Retorna:
            (nombre_canonico_elegido, confianza, lista_candidatos_similares, requiere_escritura)
            
        Reglas:
        - Si texto_destinatario está vacío: requiere_escritura = True.
        - Coincidencia exacta normalizada: certidumbre 1.0 -> nombre canónico.
        - Coincidencia aproximada en origen de voz:
            - Si hay múltiples candidatos cercanos: requiere_escritura = True (evita sustitución errónea).
            - Si la similitud es baja (< 0.85): requiere_escritura = True.
            - Si similitud >= 0.88 y un único candidato claro: se sugiere el canónico.
        - Si no coincide con nadie en catálogo pero fue escrito por texto: se valida longitud y limpieza.
        """
        if not texto_destinatario or not texto_destinatario.strip():
            return (None, 0.0, [], True)

        # Limpiar preposiciones dativas habladas al inicio («A Miguel», «Para Juan», «A nombre de Ana»)
        texto_limpio_dest = re.sub(
            r"^(?:(?:a\s+nombre\s+de|a\s+favor\s+de|para|a)\s+)+",
            "",
            texto_destinatario.strip(),
            flags=re.IGNORECASE,
        ).strip()
        if not texto_limpio_dest:
            return (None, 0.0, [], True)

        norm_in = normalizar_para_comparacion(texto_limpio_dest)
        if len(norm_in) < 2:
            return (None, 0.0, [], True)

        personas = cls.obtener_personas_canonicas()

        # 1. Coincidencia exacta completa
        for p in personas:
            if p["norm"] == norm_in:
                return (p["nombre_canonico"], 1.0, [p["nombre_canonico"]], False)

        # 1b. Si STT pegó la preposición dativa 'a' al inicio de un nombre conocido (ej. «Amiguel» -> «Miguel», «Acarlos» -> «Carlos»)
        if " " not in norm_in and norm_in.startswith("a") and len(norm_in) >= 5:
            sin_a = norm_in[1:]
            cands_sin_a = [
                p for p in personas
                if p["norm"] == sin_a or sin_a in p["norm"].split()
            ]
            if len(cands_sin_a) == 1:
                return (cands_sin_a[0]["nombre_canonico"], 0.94, [cands_sin_a[0]["nombre_canonico"]], False)
            elif len(cands_sin_a) > 1:
                return (None, 0.5, [c["nombre_canonico"] for c in cands_sin_a[:5]], True)

        # 2. Coincidencia por tokens/palabras exactas (ej: "Carlos" -> "Carlos Andrés Rodríguez").
        # Solo expandir cuando la entrada es subconjunto del candidato; nunca truncar un nombre
        # más específico con apellido (ej: "Miguel Torres") a un registro histórico de 1 sola palabra ("Miguel").
        palabras_in = set(norm_in.split())
        candidatos_sub: List[Dict[str, Any]] = []
        for p in personas:
            palabras_p = set(p["norm"].split())
            if palabras_in.issubset(palabras_p):
                candidatos_sub.append(p)

        if len(candidatos_sub) == 1:
            return (candidatos_sub[0]["nombre_canonico"], 0.95, [candidatos_sub[0]["nombre_canonico"]], False)
        elif len(candidatos_sub) > 1:
            # Varios candidatos (ej: Carlos Gomez vs Carlos Rodriguez)
            nombres = [c["nombre_canonico"] for c in candidatos_sub[:5]]
            return (None, 0.5, nombres, True)

        # 3. Comparación difusa (similitud textual + fonética española preservando ñ vs n)
        fon_in = clave_fonetica_espanol(texto_limpio_dest)
        toks_in_list = norm_in.split()
        mejores: List[Tuple[float, Dict[str, Any], bool]] = []
        for p in personas:
            p_toks = p["norm"].split()
            ratio_str = SequenceMatcher(None, norm_in, p["norm"]).ratio()
            p_fon = p.get("fon") or clave_fonetica_espanol(p["nombre_canonico"])
            ratio_fon = SequenceMatcher(None, fon_in, p_fon).ratio() if (fon_in and p_fon) else 0.0
            fon_exacto = bool(fon_in and p_fon and fon_in == p_fon)
            # Evitar que un nombre compuesto de 2+ palabras (ej. "Miguel Torres" o "Miguel QA")
            # sea sobrescrito por una sola palabra histórica distinta (ej. "Amiguel"),
            # salvo equivalencia fonética exacta (ej. "Yan Carlos" <-> "Giancarlos").
            if len(toks_in_list) >= 2 and len(p_toks) == 1 and not fon_exacto:
                continue
            ratio = max(ratio_str, ratio_fon)
            if ratio >= 0.75:
                mejores.append((ratio, p, fon_exacto))

        mejores.sort(key=lambda x: x[0], reverse=True)

        if mejores:
            top_score, top_p, top_fon_exacto = mejores[0]
            # Si hay ambigüedad (dos con score muy cercano)
            if len(mejores) > 1 and (mejores[0][0] - mejores[1][0] < 0.10):
                candidatos_ambiguos = [m[1]["nombre_canonico"] for m in mejores[:4]]
                return (None, top_score, candidatos_ambiguos, True)

            # Si la diferencia entre la entrada y el candidato involucra 'ñ' vs 'n',
            # no asumir equivalencia silenciosa: una diferencia de 'ñ' frente a 'n' distingue identidades
            # distintas en español (ej. Peña frente a Pena, Castaño frente a Castano).
            # En caso de discrepancia 'ñ' frente a 'n', se exige aclaración o selección explícita.
            tiene_conflicto_enie = False
            if ("ñ" in norm_in and "ñ" not in top_p["norm"]) or ("ñ" not in norm_in and "ñ" in top_p["norm"]):
                norm_in_sin_enie = norm_in.replace("ñ", "n")
                top_p_sin_enie = top_p["norm"].replace("ñ", "n")
                if norm_in_sin_enie == top_p_sin_enie or SequenceMatcher(None, norm_in_sin_enie, top_p_sin_enie).ratio() > 0.85:
                    tiene_conflicto_enie = True

            if tiene_conflicto_enie:
                return (None, top_score, [top_p["nombre_canonico"]], True)

            # Si es por voz y el score no es contundente (>= 0.88), no arriesgar
            if origen_voz:
                if top_score >= 0.88:
                    return (top_p["nombre_canonico"], top_score, [top_p["nombre_canonico"]], False)
                else:
                    return (None, top_score, [top_p["nombre_canonico"]], True)
            else:
                # Si fue escrito manualmente por teclado, respetar el nombre digitado y solo
                # normalizar al canónico ante coincidencia fonética exacta o casi idéntica (>= 0.92)
                # (evita sobrescribir "Ana Silva" por "Ana Sofía" o "Pedro Ruiz" por "Pedro Cruz").
                if top_fon_exacto or top_score >= 0.92:
                    return (top_p["nombre_canonico"], top_score, [top_p["nombre_canonico"]], False)

        # 4. No coincide con nadie registrado
        # Si vino por voz y no coincide con nadie conocido, es peligroso aceptarlo a ciegas
        if origen_voz:
            return (None, 0.0, [], True)

        # Si vino por teclado y tiene un formato razonable de nombre (al menos 3 letras)
        nombre_limpio = capitalizar_nombre_propio(texto_limpio_dest)
        return (nombre_limpio, 0.75, [], False)
