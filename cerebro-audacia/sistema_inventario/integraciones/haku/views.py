"""Vistas y endpoints de la API interna autenticada para Haku Inventario.

Todas las peticiones validan:
- Autenticación activa de Django (sesión de usuario).
- Estado de cuenta: cuentas inactivas o eliminadas lógicamente son rechazadas inmediatamente.
- Permisos por rol y almacén según especificación (Superusuario, Administrador, Usuario).
- Manejo de pre-confirmación en dos pasos para mutaciones.
"""

import copy
import hmac
import json
import logging
import re
import time
import uuid

from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied, ValidationError
from django.http import HttpResponse, JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import TemplateView
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib.auth.password_validation import validate_password
from django.db import OperationalError, transaction
from cuentas.models import UsuarioAudacIA
from inventario.models import Almacen, Auditoria, Existencia, Movimiento, Prestamo, Producto

from .serializers import (
    ValidadorComando,
    ValidadorConfirmacion,
    ValidadorCancelacion,
    ValidadorActualizarDestinatario,
    ValidadorConsultaStock,
    ValidadorSintesisVoz,
)
from .audio.captura_qa import actualizar_metadatos_audio_qa, guardar_audio_qa
from .services import (
    HakuOcupadoError,
    IdempotencyConflictError,
    ejecutar_confirmacion_propuesta,
    normalizar_para_tts,
    procesar_audio_y_transcribir,
    procesar_comando_inventario,
    sintetizar_audio_bytes,
    transcribir_audio_bytes,
)
from .telemetria import adjuntar_headers_haku, obtener_ids_haku, registrar_timing

logger = logging.getLogger("integraciones.haku")



class HakuApiBaseView(LoginRequiredMixin, View):
    """Clase base para todos los endpoints de Haku con validación de seguridad."""

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "No autenticado. Inicie sesión."}, status=401)

        if not request.user.is_active or getattr(request.user, "eliminado_logicamente", False):
            return JsonResponse({"error": "Cuenta desactivada o eliminada.", "sesion_activa": False}, status=403)

        return super().dispatch(request, *args, **kwargs)

    @staticmethod
    def _obtener_json(request) -> dict:
        """Parsea el cuerpo de la petición como JSON."""
        if not request.body:
            return {}
        try:
            return json.loads(request.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}


class MensajeHakuView(HakuApiBaseView):
    """Endpoint principal de interacción conversacional (texto o voz transcrita).

    POST /api/haku/mensaje/
    Payload: {"mensaje": "..."}
    """

    def post(self, request):
        t0_http = time.perf_counter()
        int_id, req_id = obtener_ids_haku(request)
        datos = self._obtener_json(request)
        try:
            validados = ValidadorComando.validar(datos)
        except ValidationError as err:
            dt_err = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "mensaje_total", dt_err, request_id=req_id, status="validation_error")
            resp = JsonResponse({"error": err.message_dict if hasattr(err, "message_dict") else str(err)}, status=400)
            return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_err)

        mensaje = validados["mensaje"]
        session_key = request.session.session_key
        if not session_key:
            request.session.save()
            session_key = request.session.session_key

        cid_cliente = validados.get("conversacion_id")
        from .contexto import HakuOperationalContext
        from .servicios_sesion import (
            capturar_estado_sesion_occ,
            evaluar_vigencia_propuesta,
            obtener_o_crear_session_token,
            persistir_propuesta_sesion_occ,
            resolver_o_crear_conversacion_turno,
            es_propuesta_contrato_anterior,
            invalidar_propuesta_pre_despliegue_en_transaccion,
            proteger_sesion_occ,
            reconciliar_y_guardar_metadatos_occ,
            neutralizar_sesion_en_memoria,
        )

        # Captura temprana de snapshot OCC previo a la inferencia LLM
        estado_occ_inicial = capturar_estado_sesion_occ(session_key, request)
        if not estado_occ_inicial:
            request.session.save()
            session_key = request.session.session_key
            estado_occ_inicial = capturar_estado_sesion_occ(session_key, request)

        if estado_occ_inicial:
            request._haku_raw_original_inicial = estado_occ_inicial.raw_original

        proteger_sesion_occ(request.session, request)

        token_sesion = request.session.get("haku_session_token")
        if not token_sesion:
            token_sesion = obtener_o_crear_session_token(request.session)

        request.session["haku_ultimo_acceso"] = round(time.time(), 4)

        cid_turno, fue_creada, err_conv = resolver_o_crear_conversacion_turno(
            request.session,
            request.user,
            conversacion_id_cliente=cid_cliente,
        )
        if err_conv:
            return JsonResponse({"error": err_conv, "mensaje": "No se pudo inicializar la conversación."}, status=409)

        if estado_occ_inicial:
            estado_occ_inicial.datos_sesion["haku_session_token"] = token_sesion
            estado_occ_inicial.datos_sesion["haku_conversacion_activa_id"] = request.session.get("haku_conversacion_activa_id")
            estado_occ_inicial.datos_sesion["haku_conversaciones"] = request.session.get("haku_conversaciones", {})
            if "haku_historial_propuestas_terminales" in request.session:
                estado_occ_inicial.datos_sesion["haku_historial_propuestas_terminales"] = request.session["haku_historial_propuestas_terminales"]
            if "haku_bloqueo_modificacion" in request.session:
                estado_occ_inicial.datos_sesion["haku_bloqueo_modificacion"] = request.session["haku_bloqueo_modificacion"]

        almacen_id = getattr(request.user, "almacen_asignado_id", None)
        estado_op = HakuOperationalContext.obtener_o_crear(request.user.id, session_key, almacen_id)
        cid_prev_op = str(request.session.get("haku_ultimo_cid_operativo") or "").strip()
        if cid_turno and cid_prev_op and cid_prev_op != str(cid_turno):
            if estado_op:
                estado_op.limpiar_foco_conversacional()
                HakuOperationalContext.guardar(estado_op)
        if cid_turno:
            request.session["haku_ultimo_cid_operativo"] = str(cid_turno)
            if estado_occ_inicial:
                estado_occ_inicial.datos_sesion["haku_ultimo_cid_operativo"] = str(cid_turno)

        # Fuente canónica exclusiva: request.session (validación obligatoria de usuario y token)
        propuesta_activa = request.session.get("haku_propuesta_activa")
        if propuesta_activa:
            if es_propuesta_contrato_anterior(propuesta_activa):
                invalidar_propuesta_pre_despliegue_en_transaccion(session_key, request, propuesta_activa.get("id"))
                propuesta_activa = None
            elif cid_turno and propuesta_activa.get("conversacion_id") and str(propuesta_activa.get("conversacion_id")) != cid_turno:
                propuesta_activa = None
            else:
                token_prop = propuesta_activa.get("session_token")
                token_valido = (
                    token_sesion
                    and token_prop
                    and isinstance(token_sesion, str)
                    and isinstance(token_prop, str)
                    and hmac.compare_digest(str(token_prop).strip(), str(token_sesion).strip())
                )
                usuario_valido = (
                    propuesta_activa.get("usuario_id")
                    and str(propuesta_activa.get("usuario_id")) == str(request.user.id)
                )
                if not token_valido or not usuario_valido:
                    propuesta_activa = None
                    if estado_op:
                        estado_op.propuesta_activa = None
                        HakuOperationalContext.guardar(estado_op)
                else:
                    # Si la propuesta ya caducó (>30 min) y el usuario envía una consulta nueva
                    # (no un intento explícito de confirmar, cancelar ni modificarla), no bloquear
                    # su nueva solicitud con un mensaje fantasma de caducidad.
                    es_vig_ok, esta_cad_msg = evaluar_vigencia_propuesta(propuesta_activa.get("expira_en"))
                    if es_vig_ok and esta_cad_msg:
                        msg_low_cad = (mensaje or "").strip().lower()
                        es_accion_sobre_propuesta = bool(
                            re.search(
                                r"^(?:s[ií]\b|confirm\w*|acept\w*|dale\b|ok\b|cancel\w*|descart\w*|olv[ií]d\w*|"
                                r"no\s+quiero|cambi\w*|modific\w*|corrig\w*|pon(?:lo|le)?\b|"
                                r"que\s+sean?\s+(?:para\b|\d+)|mejor\s+(?:\d+|para\b|a\s+nombre))",
                                msg_low_cad,
                            )
                        )
                        if not es_accion_sobre_propuesta:
                            propuesta_activa = None
                            if estado_op:
                                estado_op.propuesta_activa = None
                                HakuOperationalContext.guardar(estado_op)
                # Sublote 2.4-B: Prohibido expresamente purgar la propuesta legítima en sesión ante petición ajena o no autorizada

        try:
            resultado = procesar_comando_inventario(
                usuario=request.user,
                mensaje=mensaje,
                propuesta_activa=propuesta_activa,
                sesion=request.session,
                estado_operativo=estado_op,
                interaction_id=int_id,
                conversacion_id=cid_turno,
            )

            tipo = resultado.get("tipo")
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            sem_metricas = getattr(estado_op, "metricas_tiempo", {}) or {}
            llm_ms = sem_metricas.get("llm_ms")
            llm_cola_ms = sem_metricas.get("llm_cola_ms")
            tiempo_modelo_ms = (llm_ms or 0.0) + (llm_cola_ms or 0.0)
            agente_ms = max(0.0, dt_total - tiempo_modelo_ms) if llm_ms is not None else dt_total

            registrar_timing(int_id, "agente_haku", agente_ms, request_id=req_id)
            registrar_timing(int_id, "mensaje_total", dt_total, request_id=req_id)

            metricas_resp = {
                "llm_ms": llm_ms,
                "llm_cola_ms": llm_cola_ms,
                "semantica_ms": sem_metricas.get("semantica_ms"),
                "agente_ms": round(agente_ms, 2),
                "mensaje_total_ms": round(dt_total, 2),
            }
            if llm_cola_ms:
                registrar_timing(int_id, "llm_cola", llm_cola_ms, request_id=req_id)

            # 1. El usuario confirmó por voz la propuesta activa (H-03: con reintentos transaccionales)
            if tipo == "accion_confirmar" and propuesta_activa:
                from django.db import connection, transaction
                from .servicios_sesion import (
                    bloquear_y_leer_sesion,
                    guardar_sesion_transaccional,
                    buscar_ejecucion_existente_ledger,
                    reconciliar_sesion_propuesta_ejecutada,
                    es_error_bloqueo_sqlite,
                    ErrorConflictoOCC,
                )

                prop_id = str(propuesta_activa.get("id") or "").strip()
                res_existente = buscar_ejecucion_existente_ledger(prop_id, request.user)
                if res_existente:
                    reconciliar_sesion_propuesta_ejecutada(session_key, request, prop_id, res_existente)
                    resp = JsonResponse({
                        "tipo": "confirmacion_ejecutada",
                        "estado": "EJECUTADA",
                        "mensaje_pantalla": res_existente["mensaje_pantalla"],
                        "mensaje_verbal": res_existente["mensaje_verbal"],
                        "movimiento_id": res_existente.get("movimiento_id"),
                        "stock_nuevo": res_existente.get("stock_nuevo"),
                        "propuesta_activa": False,
                        "metricas_tiempo": metricas_resp,
                    })
                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                MAX_INTENTOS = 5
                exito_confirmacion = False
                res_ejecucion = None

                try:
                    for intento in range(MAX_INTENTOS):
                        try:
                            with transaction.atomic(using="default"):
                                session_obj, datos_s, raw_orig = bloquear_y_leer_sesion(session_key, request)
                                if not session_obj:
                                    res_existente = buscar_ejecucion_existente_ledger(prop_id, request.user)
                                    if res_existente:
                                        reconciliar_sesion_propuesta_ejecutada(session_key, request, prop_id, res_existente)
                                        resp = JsonResponse({
                                            "tipo": "confirmacion_ejecutada",
                                            "estado": "EJECUTADA",
                                            "mensaje_pantalla": res_existente["mensaje_pantalla"],
                                            "mensaje_verbal": res_existente["mensaje_verbal"],
                                            "movimiento_id": res_existente.get("movimiento_id"),
                                            "stock_nuevo": res_existente.get("stock_nuevo"),
                                            "propuesta_activa": False,
                                            "metricas_tiempo": metricas_resp,
                                        })
                                        return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)
                                    resp = JsonResponse({
                                        "error": "No hay ninguna propuesta pendiente de confirmación o la sesión ha expirado.",
                                        "mensaje_pantalla": "No encontré ninguna propuesta pendiente de confirmación.",
                                        "mensaje_verbal": "No hay ninguna propuesta pendiente de confirmación.",
                                        "propuesta_activa": False,
                                        "metricas_tiempo": metricas_resp,
                                    }, status=404)
                                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                                res_existente = buscar_ejecucion_existente_ledger(prop_id, request.user)
                                if res_existente:
                                    reconciliar_sesion_propuesta_ejecutada(session_key, request, prop_id, res_existente)
                                    resp = JsonResponse({
                                        "tipo": "confirmacion_ejecutada",
                                        "estado": "EJECUTADA",
                                        "mensaje_pantalla": res_existente["mensaje_pantalla"],
                                        "mensaje_verbal": res_existente["mensaje_verbal"],
                                        "movimiento_id": res_existente.get("movimiento_id"),
                                        "stock_nuevo": res_existente.get("stock_nuevo"),
                                        "propuesta_activa": False,
                                        "metricas_tiempo": metricas_resp,
                                    })
                                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                                prop_fresca = datos_s.get("haku_propuesta_activa")
                                if not prop_fresca or str(prop_fresca.get("id") or "") != prop_id:
                                    resp = JsonResponse({
                                        "error": "propuesta_no_activa",
                                        "mensaje_pantalla": "La propuesta ya no se encuentra activa o fue modificada.",
                                        "mensaje_verbal": "La propuesta ya no está disponible para confirmación.",
                                        "propuesta_activa": False,
                                        "metricas_tiempo": metricas_resp,
                                    }, status=409)
                                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                                obs_confirm_voz = resultado.get("observacion_usuario")
                                obs_confirm_ui = (
                                    str(datos.get("observacion") or "").strip()[:500]
                                    if isinstance(datos, dict) and "observacion" in datos
                                    else None
                                )
                                obs_confirm_final = obs_confirm_voz if obs_confirm_voz is not None else obs_confirm_ui
                                res_ejecucion = ejecutar_confirmacion_propuesta(
                                    usuario=request.user,
                                    propuesta=prop_fresca,
                                    ip_origen=request.META.get("REMOTE_ADDR", "127.0.0.1"),
                                    observacion_usuario=obs_confirm_final,
                                )
                                mov_id_raw = res_ejecucion.get("movimiento_id")
                                recibo = {
                                    "tipo_registro": "RECIBO_AUXILIAR",
                                    "propuesta_id": str(prop_id),
                                    "movimiento_id": str(mov_id_raw) if mov_id_raw else None,
                                    "ejecutado_en": round(time.time(), 4),
                                }
                                if prop_fresca.get("conversacion_id") and prop_fresca.get("session_token_hash"):
                                    try:
                                        from .servicios_sesion import registrar_evidencia_terminal_sesion
                                        registrar_evidencia_terminal_sesion(datos_s, prop_fresca, "EJECUTADA")
                                    except Exception:
                                        pass
                                bloque_conf_v = datos_s.get("haku_bloqueo_modificacion")
                                if bloque_conf_v and str(bloque_conf_v.get("propuesta_id_origen") or "") == str(prop_id):
                                    datos_s["haku_bloqueo_modificacion"] = None
                                datos_s["haku_propuesta_activa"] = None
                                datos_s["haku_ultimo_evento"] = {
                                    "propuesta_id": str(prop_id),
                                    "estado": "EJECUTADA",
                                    "fecha": round(time.time(), 4),
                                }
                                datos_s["haku_ultima_ejecucion"] = recibo
                                guardar_sesion_transaccional(session_obj, datos_s, raw_orig, request)
                                if estado_op:
                                    estado_op.propuesta_activa = None
                                    HakuOperationalContext.guardar(estado_op)
                                exito_confirmacion = True
                                break
                        except (OperationalError, ErrorConflictoOCC) as err:
                            es_occ = isinstance(err, ErrorConflictoOCC)
                            es_bloqueo = isinstance(err, OperationalError) and es_error_bloqueo_sqlite(err)
                            if es_occ or es_bloqueo:
                                if intento < MAX_INTENTOS - 1:
                                    time.sleep(0.05 * (2 ** intento))
                                    continue
                                codigo_http = 409 if es_occ else 503
                                neutralizar_sesion_en_memoria(request)
                                resp = JsonResponse({
                                    "error": "conflicto_concurrencia" if es_occ else "base_datos_ocupada",
                                    "mensaje_pantalla": "El sistema está ocupado procesando otra transacción concurrente. Por favor intenta confirmar nuevamente.",
                                    "mensaje_verbal": "El sistema se encuentra ocupado procesando otra operación. Intenta confirmar nuevamente en unos segundos.",
                                    "propuesta_activa": True,
                                    "metricas_tiempo": metricas_resp,
                                }, status=codigo_http)
                                if not es_occ:
                                    resp["Retry-After"] = "1"
                                return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)
                            raise

                except PermissionDenied as err:
                    try:
                        from inventario.models import Auditoria
                        from .servicios_sesion import registrar_auditoria_rechazo_seguridad, persistir_propuesta_sesion_occ

                        es_propia = (
                            propuesta_activa.get("usuario_id")
                            and str(propuesta_activa.get("usuario_id")) == str(request.user.id)
                        )
                        aud_prev = Auditoria.objects.filter(
                            tipo_entidad="PropuestaHaku",
                            accion="PROPUESTA_RECHAZADA",
                            id_entidad__in=[prop_id, f"no-verificada:{prop_id}"],
                            usuario=request.user,
                            resultado="RECHAZADA_PERMISO",
                        ).exists()
                        if not aud_prev:
                            registrar_auditoria_rechazo_seguridad(
                                usuario=request.user,
                                propuesta_id=prop_id,
                                codigo_resultado="RECHAZADA_PERMISO",
                                motivo=str(err),
                                operacion_declarada=propuesta_activa.get("tipo_operacion"),
                                producto_id_declarado=propuesta_activa.get("producto_id"),
                                direccion_peer=request.META.get("REMOTE_ADDR"),
                                propuesta_verificada=es_propia,
                            )
                        if es_propia:
                            persistir_propuesta_sesion_occ(
                                request,
                                nueva_propuesta=None,
                                accion="CONFIRMAR_PURGA",
                                propuesta_id_esperada=prop_id,
                                motivo_evento="RECHAZADA",
                            )
                    except Exception as exc_aud:
                        logger.warning("Fallo no fatal registrando auditoría de rechazo por permisos en MensajeHakuView: %s", exc_aud)

                    dt_total = (time.perf_counter() - t0_http) * 1000.0
                    registrar_timing(int_id, "mensaje_total", dt_total, request_id=req_id, status="permission_denied")
                    resp = JsonResponse({"error": str(err)}, status=403)
                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                except ValidationError as err:
                    msg_val = err.messages[0] if getattr(err, "messages", None) else str(err)
                    err_msg = msg_val.lower()
                    es_error_negocio_recuperable = any(
                        tok in err_msg
                        for tok in ("stock", "insuficiente", "almacén único", "almacen unico", "totalidad", "pendiente")
                    )
                    es_error_negocio = es_error_negocio_recuperable or any(
                        tok in err_msg
                        for tok in ("cerrado", "consumible", "no es prestable", "no admite préstamos")
                    )
                    if not es_error_negocio:
                        try:
                            from inventario.models import Auditoria
                            from .servicios_sesion import registrar_auditoria_rechazo_seguridad, persistir_propuesta_sesion_occ

                            es_propia = (
                                propuesta_activa.get("usuario_id")
                                and str(propuesta_activa.get("usuario_id")) == str(request.user.id)
                            )
                            aud_prev = Auditoria.objects.filter(
                                tipo_entidad="PropuestaHaku",
                                accion="PROPUESTA_RECHAZADA",
                                id_entidad__in=[prop_id, f"no-verificada:{prop_id}"],
                                usuario=request.user,
                                resultado="RECHAZADA_INTEGRIDAD",
                            ).exists()
                            if not aud_prev:
                                registrar_auditoria_rechazo_seguridad(
                                    usuario=request.user,
                                    propuesta_id=prop_id,
                                    codigo_resultado="RECHAZADA_INTEGRIDAD",
                                    motivo=msg_val,
                                    operacion_declarada=propuesta_activa.get("tipo_operacion"),
                                    producto_id_declarado=propuesta_activa.get("producto_id"),
                                    direccion_peer=request.META.get("REMOTE_ADDR"),
                                    propuesta_verificada=False,
                                )
                            if es_propia:
                                persistir_propuesta_sesion_occ(
                                    request,
                                    nueva_propuesta=None,
                                    accion="CONFIRMAR_PURGA",
                                    propuesta_id_esperada=prop_id,
                                    motivo_evento="RECHAZADA",
                                )
                                if estado_op:
                                    estado_op.propuesta_activa = None
                                    HakuOperationalContext.guardar(estado_op)
                        except Exception as exc_aud:
                            logger.warning("Fallo no fatal registrando auditoría de rechazo por integridad en MensajeHakuView: %s", exc_aud)
                    elif not es_error_negocio_recuperable:
                        try:
                            from .servicios_sesion import persistir_propuesta_sesion_occ
                            persistir_propuesta_sesion_occ(
                                request,
                                nueva_propuesta=None,
                                accion="CONFIRMAR_PURGA",
                                propuesta_id_esperada=prop_id,
                                motivo_evento="RECHAZADA",
                            )
                            if estado_op:
                                estado_op.propuesta_activa = None
                                HakuOperationalContext.guardar(estado_op)
                        except Exception:
                            pass

                    dt_total = (time.perf_counter() - t0_http) * 1000.0
                    registrar_timing(int_id, "mensaje_total", dt_total, request_id=req_id, status="validation_error")
                    from .services import normalizar_para_tts
                    resp = JsonResponse({
                        "error": msg_val,
                        "mensaje": msg_val,
                        "mensaje_pantalla": msg_val,
                        "mensaje_verbal": normalizar_para_tts(msg_val),
                        "propuesta_activa": es_error_negocio_recuperable,
                    }, status=400)
                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                if exito_confirmacion and res_ejecucion:
                    resp = JsonResponse({
                        "tipo": "confirmacion_ejecutada",
                        "estado": "EJECUTADA",
                        "mensaje_pantalla": res_ejecucion["mensaje_pantalla"],
                        "mensaje_verbal": res_ejecucion["mensaje_verbal"],
                        "movimiento_id": str(res_ejecucion.get("movimiento_id")) if res_ejecucion.get("movimiento_id") else None,
                        "stock_nuevo": res_ejecucion.get("stock_nuevo"),
                        "propuesta_activa": False,
                        "metricas_tiempo": metricas_resp,
                    })
                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

            # 2. El usuario canceló por voz la propuesta activa (H-03: con reintentos transaccionales)
            if tipo == "accion_cancelar" and propuesta_activa:
                from django.db import transaction
                from .servicios_sesion import (
                    bloquear_y_leer_sesion,
                    guardar_sesion_transaccional,
                    buscar_ejecucion_existente_ledger,
                    es_error_bloqueo_sqlite,
                    ErrorConflictoOCC,
                )

                prop_id = str(propuesta_activa.get("id") or "").strip()
                res_existente = buscar_ejecucion_existente_ledger(prop_id, request.user)
                if res_existente:
                    resp = JsonResponse({
                        "error": "propuesta_ya_ejecutada",
                        "mensaje_pantalla": "Esta propuesta ya fue ejecutada y no puede ser cancelada.",
                        "mensaje_verbal": "Esta operación ya fue ejecutada previamente.",
                        "propuesta_activa": False,
                        "metricas_tiempo": metricas_resp,
                    }, status=400)
                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                MAX_INTENTOS = 4
                exito_cancelacion = False

                for intento in range(MAX_INTENTOS):
                    try:
                        with transaction.atomic(using="default"):
                            session_obj, datos_s, raw_orig = bloquear_y_leer_sesion(session_key, request)
                            if not session_obj:
                                exito_cancelacion = True
                                break

                            res_existente = buscar_ejecucion_existente_ledger(prop_id, request.user)
                            if res_existente:
                                resp = JsonResponse({
                                    "error": "propuesta_ya_ejecutada",
                                    "mensaje_pantalla": "Esta propuesta ya fue ejecutada y no puede ser cancelada.",
                                    "mensaje_verbal": "Esta operación ya fue ejecutada previamente.",
                                    "propuesta_activa": False,
                                    "metricas_tiempo": metricas_resp,
                                }, status=400)
                                return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                            prop_fresca = datos_s.get("haku_propuesta_activa")
                            if not prop_fresca or str(prop_fresca.get("id") or "") != prop_id:
                                exito_cancelacion = True
                                break

                            from .servicios_sesion import registrar_auditoria_cancelacion, registrar_evidencia_terminal_sesion
                            if prop_fresca.get("conversacion_id") and prop_fresca.get("session_token_hash"):
                                try:
                                    registrar_evidencia_terminal_sesion(datos_s, prop_fresca, "CANCELADA")
                                except Exception:
                                    pass
                            bloq_canc_v = datos_s.get("haku_bloqueo_modificacion")
                            if bloq_canc_v and str(bloq_canc_v.get("propuesta_id_origen") or "") == str(prop_id):
                                datos_s["haku_bloqueo_modificacion"] = None
                            datos_s["haku_propuesta_activa"] = None
                            datos_s["haku_ultimo_evento"] = {
                                "propuesta_id": str(prop_id),
                                "usuario_id": str(request.user.id),
                                "estado": "CANCELADA",
                                "fecha": round(time.time(), 4),
                                "timestamp": round(time.time(), 4),
                            }
                            guardar_sesion_transaccional(session_obj, datos_s, raw_orig, request)
                            if estado_op:
                                estado_op.propuesta_activa = None
                                estado_op.operacion_pendiente = None
                                HakuOperationalContext.guardar(estado_op)
                            registrar_auditoria_cancelacion(request.user, str(prop_id), prop_fresca)
                            exito_cancelacion = True
                            break
                    except (OperationalError, ErrorConflictoOCC) as err:
                        es_occ = isinstance(err, ErrorConflictoOCC)
                        es_bloqueo = isinstance(err, OperationalError) and es_error_bloqueo_sqlite(err)
                        if es_occ or es_bloqueo:
                            if intento < MAX_INTENTOS - 1:
                                time.sleep(0.05 * (2 ** intento))
                                continue
                            codigo_http = 409 if es_occ else 503
                            neutralizar_sesion_en_memoria(request)
                            resp = JsonResponse({
                                "error": "conflicto_concurrencia" if es_occ else "base_datos_ocupada",
                                "mensaje_pantalla": "El sistema está ocupado procesando otra transacción concurrente. Por favor intenta cancelar nuevamente.",
                                "mensaje_verbal": "El sistema se encuentra ocupado. Intenta cancelar nuevamente en unos segundos.",
                                "propuesta_activa": True,
                                "metricas_tiempo": metricas_resp,
                            }, status=codigo_http)
                            if not es_occ:
                                resp["Retry-After"] = "1"
                            return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)
                        raise

                if exito_cancelacion:
                    resp = JsonResponse({
                        "tipo": "propuesta_cancelada",
                        "estado": "CANCELADA",
                        "mensaje_pantalla": resultado["mensaje_pantalla"],
                        "mensaje_verbal": resultado["mensaje_verbal"],
                        "propuesta_activa": False,
                        "metricas_tiempo": metricas_resp,
                    })
                    return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

            # 3. Se generó una nueva propuesta que requiere confirmación
            if tipo == "propuesta" and resultado.get("requiere_confirmacion"):
                exito_occ, prop_guardada, motivo_occ = persistir_propuesta_sesion_occ(
                    request,
                    nueva_propuesta=resultado["propuesta"],
                    accion="CREAR",
                    estado_occ_previo=estado_occ_inicial,
                )
                if not exito_occ:
                    neutralizar_sesion_en_memoria(request)
                    if motivo_occ == "conflicto_occ":
                        logger.warning("Conflicto OCC en inferencia LLM: sesión modificada durante el procesamiento")
                        resp = JsonResponse({
                            "error": "conflicto_concurrencia",
                            "mensaje": "La sesión fue modificada por otra solicitud mientras se procesaba el comando. La propuesta anterior se mantiene sin cambios.",
                            "mensaje_pantalla": "Se detectó otra solicitud simultánea en tu sesión. Por favor revisa el estado actual.",
                            "mensaje_verbal": "Se detectó otra solicitud en tu sesión. Por favor verifica el estado actual.",
                            "propuesta_activa": True,
                            "metricas_tiempo": metricas_resp,
                        }, status=409)
                        return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)
                    else:
                        resp = JsonResponse({"error": f"Error al guardar propuesta: {motivo_occ}"}, status=400)
                        return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

                prop_final = prop_guardada or resultado["propuesta"]
                datos_res = resultado.get("datos") or {}
                if datos_res.get("propuesta_actualizada") and datos_res.get("propuesta_anterior_id"):
                    try:
                        from .servicios_sesion import registrar_auditoria_modificacion
                        registrar_auditoria_modificacion(
                            usuario=request.user,
                            propuesta_anterior_id=str(datos_res["propuesta_anterior_id"]),
                            propuesta_nueva=prop_final,
                            campos_modificados=[str(datos_res.get("campo_modificado") or "propuesta")],
                        )
                    except Exception as exc_mod:
                        logger.warning("Fallo no fatal registrando auditoría de modificación de propuesta: %s", exc_mod)
                resp = JsonResponse({
                    "tipo": "propuesta",
                    "requiere_confirmacion": True,
                    "estado": "PENDIENTE",
                    "propuesta": prop_final,
                    "mensaje_pantalla": resultado["mensaje_pantalla"],
                    "mensaje_verbal": resultado["mensaje_verbal"],
                    "propuesta_activa": True,
                    "metricas_tiempo": metricas_resp,
                })
                return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

            # 3-bis. Si una corrección sobre la propuesta activa quedó pendiente de aclaración, persistir la marca en sesión
            datos_res = resultado.get("datos") or {}
            if isinstance(datos_res, dict) and datos_res.get("actualizar_propuesta_sesion") and propuesta_activa:
                persistir_propuesta_sesion_occ(
                    request,
                    nueva_propuesta=propuesta_activa,
                    accion="CREAR",
                    estado_occ_previo=estado_occ_inicial,
                )

            # 4. Consulta o diálogo general
            raw_orig = getattr(request, "_haku_raw_original_inicial", None)
            reconciliar_y_guardar_metadatos_occ(
                session_key=session_key,
                raw_original=raw_orig,
                nuevos_datos=getattr(request.session, "_session_cache", {}),
                request=request,
            )
            propuesta_fresca = request.session.get("haku_propuesta_activa")
            codigo_http_resp = (
                resultado.get("datos", {}).get("codigo_http", 200)
                if resultado.get("tipo") == "error" and resultado.get("datos", {}).get("error_terminal")
                else 200
            )
            datos_payload_resp = dict(resultado.get("datos") or {})
            for _clave_ui in ("esperando", "requiere_escritura", "candidatos"):
                if _clave_ui in resultado and _clave_ui not in datos_payload_resp:
                    datos_payload_resp[_clave_ui] = resultado[_clave_ui]
            if (
                tipo == "aclaracion"
                and "esperando" not in resultado
                and "esperando" not in datos_payload_resp
                and not datos_payload_resp.get("operacion_pendiente_preservada")
                and not datos_payload_resp.get("requiere_aclaracion")
                and estado_op
                and isinstance(estado_op.operacion_pendiente, dict)
                and estado_op.operacion_pendiente.get("esperando")
            ):
                datos_payload_resp["esperando"] = estado_op.operacion_pendiente["esperando"]
            payload_resp_general = {
                "tipo": tipo,
                "mensaje_pantalla": resultado.get("mensaje_pantalla", ""),
                "mensaje_verbal": resultado.get("mensaje_verbal", ""),
                "datos": datos_payload_resp,
                "propuesta_activa": propuesta_fresca is not None,
                "metricas_tiempo": metricas_resp,
            }
            for _clave_ui in ("esperando", "requiere_escritura", "candidatos"):
                if _clave_ui in resultado:
                    payload_resp_general[_clave_ui] = resultado[_clave_ui]
                elif _clave_ui == "esperando" and _clave_ui in datos_payload_resp:
                    payload_resp_general[_clave_ui] = datos_payload_resp[_clave_ui]
            resp = JsonResponse(payload_resp_general, status=codigo_http_resp)
            return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total, llm=llm_ms, agente_haku=agente_ms)

        except HakuOcupadoError:
            neutralizar_sesion_en_memoria(request)
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "mensaje_total", dt_total, request_id=req_id, status="haku_ocupado")
            resp = JsonResponse({
                "error": "El servicio de asistencia se encuentra ocupado. Por favor intente nuevamente.",
                "mensaje_pantalla": "Hacu Inventario está atendiendo otra solicitud. Por favor, inténtelo nuevamente en unos instantes.",
                "mensaje_verbal": "Jacu Inventario está atendiendo otra solicitud. Por favor, inténtelo nuevamente en unos instantes.",
            }, status=503)
            resp["Retry-After"] = "5"
            return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total)
        except PermissionDenied as err:
            neutralizar_sesion_en_memoria(request)
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "mensaje_total", dt_total, request_id=req_id, status="permission_denied")
            resp = JsonResponse({"error": str(err)}, status=403)
            return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total)
        except ValidationError as err:
            neutralizar_sesion_en_memoria(request)
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "mensaje_total", dt_total, request_id=req_id, status="validation_error")
            resp = JsonResponse({"error": str(err)}, status=400)
            return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total)
        except OperationalError as err:
            neutralizar_sesion_en_memoria(request)
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "mensaje_total", dt_total, request_id=req_id, status="db_locked")
            err_msg = str(err).lower()
            if "locked" in err_msg or "busy" in err_msg:
                logger.warning("Bloqueo de base de datos SQLite en comando Haku: %s", err)
                resp = JsonResponse({
                    "error": "La base de datos se encuentra temporalmente ocupada por otra transacción. Por favor reintente en unos segundos.",
                    "mensaje_pantalla": "El sistema se encuentra ocupado procesando otra transacción concurrente. Por favor intenta de nuevo.",
                    "mensaje_verbal": "El sistema está ocupado. Por favor intenta de nuevo.",
                }, status=503)
                resp["Retry-After"] = "1"
                return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total)
            logger.exception("Error de base de datos en Haku: %s", err)
            resp = JsonResponse({"error": "Error interno al procesar comando. Intente nuevamente más tarde."}, status=500)
            return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total)
        except Exception as exc:
            neutralizar_sesion_en_memoria(request)
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "mensaje_total", dt_total, request_id=req_id, status="error")
            logger.exception("Error procesando mensaje en Haku: %s", exc)
            resp = JsonResponse({"error": "Error interno al procesar comando. Intente nuevamente más tarde."}, status=500)
            return adjuntar_headers_haku(resp, int_id, req_id, mensaje_total=dt_total)



class ConfirmarPropuestaView(HakuApiBaseView):
    """Confirma y ejecuta atómicamente la propuesta pendiente activa bajo protocolo transaccional único.

    POST /api/haku/confirmar/
    Payload: {"propuesta_id": "...", "idempotency_key": "...", "conversacion_id": "..." (opcional)}
    """

    def post(self, request):
        # Paso 1: Obtener JSON
        datos = self._obtener_json(request)

        # Paso 2 y 3: Validar con ValidadorConfirmacion
        try:
            validados = ValidadorConfirmacion.validar(datos)
        except ValidationError as err:
            err_dict = err.message_dict if hasattr(err, "message_dict") else {}
            cid_err = err_dict.get("conversacion_id", [])
            if any("conversacion_id_obligatorio" in str(m) for m in cid_err):
                return JsonResponse({"ok": False, "error": "conversacion_id_obligatorio", "mensaje": "Se requiere el parámetro 'conversacion_id'."}, status=400)
            return JsonResponse({"error": err_dict if err_dict else str(err)}, status=400)

        propuesta_id = str(validados["propuesta_id"]).strip()
        idempotency_key = validados["idempotency_key"]
        conversacion_id = validados["conversacion_id"]
        observacion_usuario = (
            str(datos.get("observacion") or "").strip()[:500]
            if isinstance(datos, dict) and "observacion" in datos
            else None
        )

        try:
            from .servicios_sesion import (
                ErrorConflictoOCC,
                bloquear_y_leer_sesion,
                guardar_sesion_transaccional,
                buscar_ejecucion_existente_ledger,
                reconciliar_sesion_propuesta_ejecutada,
                es_error_bloqueo_sqlite,
                consultar_estado_propuesta_persistente,
                evaluar_vigencia_propuesta,
                evaluar_bloqueo_modificacion_activo,
                registrar_auditoria_caducidad,
                registrar_auditoria_rechazo_seguridad,
                registrar_evidencia_terminal_sesion,
                obtener_evidencia_verificable_propuesta_terminal,
                es_propuesta_contrato_anterior,
                invalidar_propuesta_pre_despliegue_en_transaccion,
                EstadoPropuestaHaku,
                ErrorRegistroEvidenciaTerminal,
                neutralizar_sesion_en_memoria,
            )
            from .services import verificar_firma_propuesta, ejecutar_confirmacion_propuesta

            session_key = getattr(request.session, "session_key", "") or ""
            if not session_key:
                request.session.save()
                session_key = request.session.session_key

            # Paso 4: Pre-check en estado persistente de propuesta y ledger inmutable
            res_ledger_pre = buscar_ejecucion_existente_ledger(propuesta_id, request.user, conversacion_id_esperada=conversacion_id)
            if res_ledger_pre:
                reconciliada = reconciliar_sesion_propuesta_ejecutada(session_key, request, propuesta_id, res_ledger_pre)
                resp_data = dict(res_ledger_pre)
                if not reconciliada:
                    resp_data["advertencia"] = "sesion_no_reconciliada"
                return JsonResponse(resp_data, status=200)

            estado_pers, datos_pers = consultar_estado_propuesta_persistente(propuesta_id, request.user)
            if estado_pers in ("CONFLICTO_INTEGRIDAD", "CONFLICTO_INTEGRIDAD_LINAJE"):
                return JsonResponse(datos_pers or {}, status=409)
            if estado_pers == EstadoPropuestaHaku.EJECUTADA:
                reconciliada = reconciliar_sesion_propuesta_ejecutada(session_key, request, propuesta_id, datos_pers or {})
                resp_data = dict(datos_pers or {})
                if not reconciliada:
                    resp_data["advertencia"] = "sesion_no_reconciliada"
                return JsonResponse(resp_data, status=200)
            if estado_pers == EstadoPropuestaHaku.CANCELADA:
                return JsonResponse({
                    "error": "propuesta_cancelada",
                    "mensaje": "Esta propuesta fue cancelada y no puede ser confirmada.",
                }, status=409)
            if estado_pers == EstadoPropuestaHaku.SUSTITUIDA:
                return JsonResponse({
                    "error": "propuesta_sustituida",
                    "mensaje": "Esta propuesta fue cancelada porque generaste una solicitud más reciente en otra pestaña. Por favor revisa la propuesta activa actual.",
                    "mensaje_pantalla": "Esta propuesta fue cancelada porque generaste una solicitud más reciente en otra pestaña. Por favor revisa la propuesta activa actual.",
                    "mensaje_verbal": "Esta propuesta fue reemplazada por una solicitud más reciente.",
                }, status=409)
            if estado_pers == EstadoPropuestaHaku.CADUCADA:
                return JsonResponse({
                    "error": "propuesta_caducada",
                    "mensaje": "La propuesta ha caducado por superar el límite de 30 minutos. Por favor genera una nueva solicitud.",
                }, status=400)
            if estado_pers == EstadoPropuestaHaku.RECHAZADA:
                return JsonResponse({
                    "error": "propuesta_rechazada",
                    "mensaje": "Esta propuesta fue rechazada previamente y no puede ser confirmada.",
                }, status=409)

            MAX_INTENTOS = 5
            for intento in range(MAX_INTENTOS):
                try:
                    with transaction.atomic(using="default"):
                        # Paso 5: Bloquear y leer sesión dentro de transacción atómica
                        session_obj, datos_sesion, raw_orig = bloquear_y_leer_sesion(session_key, request)
                        if not session_obj:
                            res_ledger = buscar_ejecucion_existente_ledger(propuesta_id, request.user, conversacion_id_esperada=conversacion_id)
                            if res_ledger:
                                reconciliada = reconciliar_sesion_propuesta_ejecutada(session_key, request, propuesta_id, res_ledger)
                                resp_data = dict(res_ledger)
                                if not reconciliada:
                                    resp_data["advertencia"] = "sesion_no_reconciliada"
                                return JsonResponse(resp_data, status=200)
                            return JsonResponse({"error": "No hay ninguna propuesta pendiente de confirmación o la sesión ha expirado."}, status=404)

                        res_ledger = buscar_ejecucion_existente_ledger(propuesta_id, request.user, conversacion_id_esperada=conversacion_id)
                        if res_ledger:
                            reconciliada = reconciliar_sesion_propuesta_ejecutada(session_key, request, propuesta_id, res_ledger)
                            resp_data = dict(res_ledger)
                            if not reconciliada:
                                resp_data["advertencia"] = "sesion_no_reconciliada"
                            return JsonResponse(resp_data, status=200)

                        # Paso 6: Comprobar presencia de propuesta activa
                        propuesta_activa = datos_sesion.get("haku_propuesta_activa")
                        if not propuesta_activa:
                            res_ledger = buscar_ejecucion_existente_ledger(propuesta_id, request.user, conversacion_id_esperada=conversacion_id)
                            if res_ledger:
                                reconciliada = reconciliar_sesion_propuesta_ejecutada(session_key, request, propuesta_id, res_ledger)
                                resp_data = dict(res_ledger)
                                if not reconciliada:
                                    resp_data["advertencia"] = "sesion_no_reconciliada"
                                return JsonResponse(resp_data, status=200)

                            est_p, dat_p = consultar_estado_propuesta_persistente(propuesta_id, request.user, datos_sesion)
                            if est_p in ("CONFLICTO_INTEGRIDAD", "CONFLICTO_INTEGRIDAD_LINAJE"):
                                return JsonResponse(dat_p or {}, status=409)
                            if est_p == EstadoPropuestaHaku.EJECUTADA:
                                reconciliada = reconciliar_sesion_propuesta_ejecutada(session_key, request, propuesta_id, dat_p or {})
                                resp_data = dict(dat_p or {})
                                if not reconciliada:
                                    resp_data["advertencia"] = "sesion_no_reconciliada"
                                return JsonResponse(resp_data, status=200)
                            if est_p == EstadoPropuestaHaku.CANCELADA:
                                return JsonResponse({
                                    "error": "propuesta_cancelada",
                                    "mensaje": "Esta propuesta fue cancelada y no puede ser confirmada.",
                                }, status=409)
                            if est_p == EstadoPropuestaHaku.SUSTITUIDA:
                                return JsonResponse({
                                    "error": "propuesta_sustituida",
                                    "mensaje": "Esta propuesta fue cancelada porque se generó una solicitud posterior en otra pestaña.",
                                    "mensaje_pantalla": "Esta propuesta fue cancelada porque generaste una solicitud más reciente en otra pestaña. Por favor revisa la propuesta activa actual.",
                                    "mensaje_verbal": "Esta propuesta fue reemplazada por una solicitud más reciente.",
                                }, status=409)
                            if est_p == EstadoPropuestaHaku.CADUCADA:
                                return JsonResponse({
                                    "error": "propuesta_caducada",
                                    "mensaje": "La propuesta ha caducado por superar el límite de tiempo.",
                                }, status=400)
                            return JsonResponse({"error": "No hay ninguna propuesta pendiente de confirmación.", "codigo": "propuesta_no_activa"}, status=404)

                        # Paso 7: Validar propiedad del usuario
                        if not propuesta_activa.get("usuario_id"):
                            registrar_auditoria_rechazo_seguridad(
                                usuario=request.user,
                                propuesta_id=propuesta_id,
                                codigo_resultado="RECHAZADA_PROPIEDAD",
                                motivo="La propuesta no cuenta con identificador de usuario válido",
                                operacion_declarada=propuesta_activa.get("tipo_operacion"),
                                producto_id_declarado=propuesta_activa.get("producto_id"),
                                direccion_peer=request.META.get("REMOTE_ADDR"),
                                propuesta_verificada=False,
                            )
                            return JsonResponse({"error": "La propuesta no cuenta con un identificador de usuario válido."}, status=403)

                        if str(propuesta_activa.get("usuario_id")) != str(request.user.id):
                            registrar_auditoria_rechazo_seguridad(
                                usuario=request.user,
                                propuesta_id=propuesta_id,
                                codigo_resultado="RECHAZADA_PROPIEDAD",
                                motivo="Propuesta pertenece a otro usuario",
                                operacion_declarada=propuesta_activa.get("tipo_operacion"),
                                producto_id_declarado=propuesta_activa.get("producto_id"),
                                direccion_peer=request.META.get("REMOTE_ADDR"),
                                propuesta_verificada=False,
                            )
                            return JsonResponse({"error": "Esta propuesta pertenece a otro usuario y no puede ser confirmada."}, status=403)

                        token_sesion = datos_sesion.get("haku_session_token")
                        token_prop = propuesta_activa.get("session_token")
                        if (
                            not token_sesion
                            or not token_prop
                            or not isinstance(token_sesion, str)
                            or not isinstance(token_prop, str)
                            or not hmac.compare_digest(str(token_prop).strip(), str(token_sesion).strip())
                        ):
                            registrar_auditoria_rechazo_seguridad(
                                usuario=request.user,
                                propuesta_id=propuesta_id,
                                codigo_resultado="RECHAZADA_TOKEN",
                                motivo="Token de sesión no coincide o es inválido",
                                operacion_declarada=propuesta_activa.get("tipo_operacion"),
                                producto_id_declarado=propuesta_activa.get("producto_id"),
                                direccion_peer=request.META.get("REMOTE_ADDR"),
                                propuesta_verificada=False,
                            )
                            return JsonResponse({"error": "La propuesta no corresponde a la sesión activa o carece de token de sesión válido."}, status=403)

                        if str(propuesta_activa.get("id")) != propuesta_id:
                            sustituida_id = datos_sesion.get("haku_propuesta_sustituida_id")
                            if sustituida_id and str(sustituida_id) == propuesta_id:
                                return JsonResponse({
                                    "error": "propuesta_sustituida",
                                    "mensaje": "Esta propuesta fue cancelada porque se generó una solicitud posterior en otra pestaña.",
                                    "mensaje_pantalla": "Esta propuesta fue cancelada porque generaste una solicitud más reciente en otra pestaña. Por favor revisa la propuesta activa actual.",
                                    "mensaje_verbal": "Esta propuesta fue reemplazada por una solicitud más reciente.",
                                }, status=409)
                            return JsonResponse({"error": "El ID de la propuesta no coincide con la propuesta activa."}, status=400)

                        # Paso 8: Validar correspondencia de conversación e identidad conversacional (H-02)
                        cid_prop = propuesta_activa.get("conversacion_id")
                        if not cid_prop or not str(cid_prop).strip():
                            registrar_auditoria_rechazo_seguridad(
                                usuario=request.user,
                                propuesta_id=propuesta_id,
                                codigo_resultado="RECHAZADA_INTEGRIDAD",
                                motivo="Propuesta carece de identidad conversacional válida",
                                operacion_declarada=propuesta_activa.get("tipo_operacion"),
                                producto_id_declarado=propuesta_activa.get("producto_id"),
                                direccion_peer=request.META.get("REMOTE_ADDR"),
                                propuesta_verificada=False,
                            )
                            return JsonResponse({
                                "ok": False,
                                "error": "identidad_conversacion_invalida",
                                "mensaje": "La propuesta carece de una identidad conversacional válida.",
                            }, status=409)

                        try:
                            cid_prop_u = uuid.UUID(str(cid_prop).strip(), version=4)
                            cid_prop_str = str(cid_prop_u)
                        except (ValueError, TypeError, AttributeError):
                            return JsonResponse({
                                "ok": False,
                                "error": "identidad_conversacion_invalida",
                                "mensaje": "La propuesta contiene una identidad conversacional malformada.",
                            }, status=409)

                        if cid_prop_str != conversacion_id:
                            return JsonResponse({
                                "ok": False,
                                "error": "identidad_conversacion_invalida",
                                "mensaje": "La propuesta pertenece a otra pestaña o conversación.",
                            }, status=409)

                        # Paso 9: Validar contrato pre-despliegue (R-06)
                        if es_propuesta_contrato_anterior(propuesta_activa):
                            invalidar_propuesta_pre_despliegue_en_transaccion(session_key, request, propuesta_id)
                            return JsonResponse({
                                "error": "propuesta_contrato_anterior",
                                "mensaje": "La propuesta pertenece a una versión anterior del contrato criptográfico y fue invalidada de manera segura. Por favor genere una nueva propuesta.",
                            }, status=409)

                        # Paso 10: Validación criptográfica estricta HMAC v2
                        if not verificar_firma_propuesta(propuesta_activa):
                            registrar_auditoria_rechazo_seguridad(
                                usuario=request.user,
                                propuesta_id=propuesta_id,
                                codigo_resultado="RECHAZADA_FIRMA",
                                motivo="Firma HMAC inválida o alterada",
                                operacion_declarada=propuesta_activa.get("tipo_operacion"),
                                producto_id_declarado=propuesta_activa.get("producto_id"),
                                direccion_peer=request.META.get("REMOTE_ADDR"),
                                propuesta_verificada=False,
                            )
                            registrar_evidencia_terminal_sesion(datos_sesion, propuesta_activa, "RECHAZADA")
                            datos_sesion["haku_propuesta_activa"] = None
                            datos_sesion["haku_ultimo_evento"] = {
                                "propuesta_id": propuesta_id,
                                "usuario_id": str(request.user.id),
                                "estado": "RECHAZADA",
                                "fecha": round(time.time(), 4),
                                "timestamp": round(time.time(), 4),
                            }
                            guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                            return JsonResponse({
                                "error": "firma_invalida",
                                "mensaje": "Firma de propuesta inválida o alterada. La propuesta fue descartada por seguridad.",
                            }, status=400)

                        # Paso 11: Validar bloqueo de modificación
                        bloqueado, motivo_bloq, _ = evaluar_bloqueo_modificacion_activo(datos_sesion, request.user, token_sesion)
                        if bloqueado:
                            return JsonResponse({
                                "error": "modificacion_sin_resolver",
                                "mensaje": "Existe una modificación de propuesta pendiente de aclaración. Responda o aclare primero la modificación.",
                                "detalle": motivo_bloq,
                            }, status=409)

                        # Paso 12: Validación de vigencia temporal bajo política Fail-Closed
                        es_valida_temporal, esta_caducada = evaluar_vigencia_propuesta(propuesta_activa.get("expira_en"))
                        if not es_valida_temporal:
                            registrar_auditoria_rechazo_seguridad(
                                usuario=request.user,
                                propuesta_id=propuesta_id,
                                codigo_resultado="RECHAZADA_INTEGRIDAD",
                                motivo="Integridad temporal inválida (expira_en corrupto, ausente o fuera de rango)",
                                operacion_declarada=propuesta_activa.get("tipo_operacion"),
                                producto_id_declarado=propuesta_activa.get("producto_id"),
                                direccion_peer=request.META.get("REMOTE_ADDR"),
                                propuesta_verificada=False,
                            )
                            registrar_evidencia_terminal_sesion(datos_sesion, propuesta_activa, "RECHAZADA")
                            datos_sesion["haku_propuesta_activa"] = None
                            datos_sesion["haku_ultimo_evento"] = {
                                "propuesta_id": propuesta_id,
                                "usuario_id": str(request.user.id),
                                "estado": "RECHAZADA",
                                "fecha": round(time.time(), 4),
                                "timestamp": round(time.time(), 4),
                            }
                            guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                            return JsonResponse({
                                "error": "estructura_invalida",
                                "mensaje": "La propuesta contiene datos temporales no válidos. Operación rechazada.",
                            }, status=400)

                        if esta_caducada:
                            registrar_auditoria_caducidad(request.user, propuesta_id, propuesta_activa)
                            registrar_evidencia_terminal_sesion(datos_sesion, propuesta_activa, "CADUCADA")
                            datos_sesion["haku_propuesta_activa"] = None
                            datos_sesion["haku_ultimo_evento"] = {
                                "propuesta_id": propuesta_id,
                                "usuario_id": str(request.user.id),
                                "estado": "CADUCADA",
                                "fecha": round(time.time(), 4),
                                "timestamp": round(time.time(), 4),
                            }
                            guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                            return JsonResponse({
                                "error": "propuesta_caducada",
                                "mensaje": "La propuesta ha caducado por superar el límite de 30 minutos. Por favor genera una nueva solicitud.",
                            }, status=400)

                        # Paso 13: Ejecución canónica de la confirmación
                        resultado = ejecutar_confirmacion_propuesta(
                            usuario=request.user,
                            propuesta=propuesta_activa,
                            idempotency_key=idempotency_key,
                            ip_origen=request.META.get("REMOTE_ADDR", "127.0.0.1"),
                            observacion_usuario=observacion_usuario,
                        )

                        mov_id_raw = resultado.get("movimiento_id")
                        recibo = {
                            "tipo_registro": "RECIBO_AUXILIAR",
                            "propuesta_id": propuesta_id,
                            "movimiento_id": str(mov_id_raw) if mov_id_raw else None,
                            "ejecutado_en": round(time.time(), 4),
                        }

                        # Paso 14: Registrar evidencia terminal EJECUTADA y purgar atómicamente la propuesta
                        registrar_evidencia_terminal_sesion(datos_sesion, propuesta_activa, "EJECUTADA")
                        bloqueo_actual = datos_sesion.get("haku_bloqueo_modificacion")
                        if bloqueo_actual and str(bloqueo_actual.get("propuesta_id_origen") or "") == propuesta_id:
                            datos_sesion["haku_bloqueo_modificacion"] = None

                        datos_sesion["haku_propuesta_activa"] = None
                        datos_sesion["haku_ultimo_evento"] = {
                            "propuesta_id": propuesta_id,
                            "estado": "EJECUTADA",
                            "fecha": round(time.time(), 4),
                        }
                        datos_sesion["haku_ultima_ejecucion"] = recibo
                        guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                        try:
                            from .contexto import HakuOperationalContext
                            estado_op_conf = HakuOperationalContext.obtener(request.user.id, session_key)
                            if estado_op_conf:
                                estado_op_conf.propuesta_activa = None
                                HakuOperationalContext.guardar(estado_op_conf)
                        except Exception:
                            pass

                        resp_data = {
                            "exito": True,
                            "estado": "EJECUTADA",
                            "mensaje_pantalla": resultado["mensaje_pantalla"],
                            "mensaje_verbal": resultado["mensaje_verbal"],
                            "movimiento_id": str(resultado.get("movimiento_id")) if resultado.get("movimiento_id") else None,
                            "stock_nuevo": resultado.get("stock_nuevo"),
                        }
                        if resultado.get("producto_id"):
                            resp_data["producto_id"] = str(resultado["producto_id"])
                        if resultado.get("producto_codigo"):
                            resp_data["producto_codigo"] = str(resultado["producto_codigo"])
                        if resultado.get("idempotente"):
                            resp_data["idempotente"] = True
                        return JsonResponse(resp_data)

                except ErrorRegistroEvidenciaTerminal as err:
                    logger.exception("Fallo crítico al registrar evidencia terminal en confirmación: %s", err)
                    neutralizar_sesion_en_memoria(request)
                    return JsonResponse({
                        "ok": False,
                        "error": "error_registro_evidencia_terminal",
                        "mensaje": "Fallo crítico al registrar la evidencia terminal de la propuesta.",
                    }, status=500)
                except (OperationalError, ErrorConflictoOCC) as err:
                    es_occ = isinstance(err, ErrorConflictoOCC)
                    es_bloqueo = isinstance(err, OperationalError) and es_error_bloqueo_sqlite(err)
                    if es_occ or es_bloqueo:
                        if intento < MAX_INTENTOS - 1:
                            time.sleep(0.05 * (2 ** intento))
                            continue
                        if es_occ:
                            return JsonResponse({"error": "Conflicto de concurrencia al confirmar la propuesta. Intenta nuevamente."}, status=409)
                        else:
                            resp = JsonResponse({
                                "error": "La base de datos se encuentra temporalmente ocupada por otra operación concurrente. Por favor reintente en unos segundos.",
                                "mensaje_pantalla": "El sistema se encuentra ocupado procesando otra transacción concurrente. Por favor intenta confirmar nuevamente.",
                                "mensaje_verbal": "El sistema se encuentra ocupado. Por favor intenta confirmar nuevamente.",
                            }, status=503)
                            resp["Retry-After"] = "1"
                            return resp
                    raise

        except IdempotencyConflictError as err:
            return JsonResponse({
                "error": "conflicto_idempotencia",
                "mensaje": str(err),
                "mensaje_pantalla": "Esta clave de operación ya fue utilizada con una propuesta diferente. La operación fue rechazada por seguridad.",
                "mensaje_verbal": "Conflicto de operación. La clave ya fue usada con otros datos.",
            }, status=409)
        except PermissionDenied as err:
            try:
                from inventario.models import Auditoria
                from .servicios_sesion import persistir_propuesta_sesion_occ, registrar_auditoria_rechazo_seguridad
                aud_prev = Auditoria.objects.filter(
                    tipo_entidad="PropuestaHaku",
                    accion="PROPUESTA_RECHAZADA",
                    id_entidad__in=[propuesta_id, f"no-verificada:{propuesta_id}"],
                    usuario=request.user,
                    resultado="RECHAZADA_PERMISO",
                ).exists()
                if not aud_prev:
                    registrar_auditoria_rechazo_seguridad(
                        usuario=request.user,
                        propuesta_id=propuesta_id,
                        codigo_resultado="RECHAZADA_PERMISO",
                        motivo=str(err),
                        propuesta_verificada=True,
                    )
                persistir_propuesta_sesion_occ(
                    request,
                    nueva_propuesta=None,
                    accion="CONFIRMAR_PURGA",
                    propuesta_id_esperada=propuesta_id,
                    motivo_evento="RECHAZADA",
                )
            except Exception:
                pass
            return JsonResponse({"error": str(err)}, status=403)
        except ValidationError as err:
            msg_val = err.messages[0] if getattr(err, "messages", None) else str(err)
            err_msg = msg_val.lower()
            es_error_negocio_recuperable = any(
                tok in err_msg
                for tok in ("stock", "insuficiente", "almacén único", "almacen unico", "totalidad", "pendiente")
            )
            es_error_negocio = es_error_negocio_recuperable or any(
                tok in err_msg
                for tok in ("cerrado", "consumible", "no es prestable", "no admite préstamos")
            )
            if not es_error_negocio:
                try:
                    from inventario.models import Auditoria
                    from .servicios_sesion import persistir_propuesta_sesion_occ, registrar_auditoria_rechazo_seguridad
                    aud_prev = Auditoria.objects.filter(
                        tipo_entidad="PropuestaHaku",
                        accion="PROPUESTA_RECHAZADA",
                        id_entidad__in=[propuesta_id, f"no-verificada:{propuesta_id}"],
                        usuario=request.user,
                        resultado="RECHAZADA_INTEGRIDAD",
                    ).exists()
                    if not aud_prev:
                        registrar_auditoria_rechazo_seguridad(
                            usuario=request.user,
                            propuesta_id=propuesta_id,
                            codigo_resultado="RECHAZADA_INTEGRIDAD",
                            motivo=msg_val,
                            propuesta_verificada=False,
                        )
                    persistir_propuesta_sesion_occ(
                        request,
                        nueva_propuesta=None,
                        accion="CONFIRMAR_PURGA",
                        propuesta_id_esperada=propuesta_id,
                        motivo_evento="RECHAZADA",
                    )
                    try:
                        from .contexto import HakuOperationalContext
                        estado_op_err = HakuOperationalContext.obtener(request.user.id, request.session.session_key)
                        if estado_op_err:
                            estado_op_err.propuesta_activa = None
                            HakuOperationalContext.guardar(estado_op_err)
                    except Exception:
                        pass
                except Exception:
                    pass
            elif not es_error_negocio_recuperable:
                try:
                    from .servicios_sesion import persistir_propuesta_sesion_occ
                    persistir_propuesta_sesion_occ(
                        request,
                        nueva_propuesta=None,
                        accion="CONFIRMAR_PURGA",
                        propuesta_id_esperada=propuesta_id,
                        motivo_evento="RECHAZADA",
                    )
                    from .contexto import HakuOperationalContext
                    estado_op_err = HakuOperationalContext.obtener(request.user.id, request.session.session_key)
                    if estado_op_err:
                        estado_op_err.propuesta_activa = None
                        HakuOperationalContext.guardar(estado_op_err)
                except Exception:
                    pass
            from .services import normalizar_para_tts
            return JsonResponse({
                "error": msg_val,
                "mensaje": msg_val,
                "mensaje_pantalla": msg_val,
                "mensaje_verbal": normalizar_para_tts(msg_val),
                "propuesta_activa": es_error_negocio_recuperable,
            }, status=400)
        except OperationalError as err:
            err_msg = str(err).lower()
            if "locked" in err_msg or "busy" in err_msg:
                logger.warning("Bloqueo de base de datos SQLite al confirmar propuesta: %s", err)
                resp = JsonResponse({
                    "error": "La base de datos se encuentra temporalmente ocupada por otra operación concurrente. Por favor reintente en unos segundos.",
                    "mensaje_pantalla": "El sistema se encuentra ocupado procesando otra transacción concurrente. Por favor intenta confirmar nuevamente.",
                    "mensaje_verbal": "El sistema se encuentra ocupado. Por favor intenta confirmar nuevamente.",
                }, status=503)
                resp["Retry-After"] = "1"
                return resp
            logger.exception("Error de base de datos al confirmar propuesta: %s", err)
            return JsonResponse({"error": "Error al ejecutar transacción. Intente nuevamente más tarde."}, status=500)
        except ErrorRegistroEvidenciaTerminal as err:
            logger.exception("Fallo crítico al registrar evidencia terminal en confirmación: %s", err)
            neutralizar_sesion_en_memoria(request)
            return JsonResponse({
                "ok": False,
                "error": "error_registro_evidencia_terminal",
                "mensaje": "Fallo crítico al registrar la evidencia terminal de la propuesta.",
            }, status=500)
        except Exception as exc:
            logger.exception("Error al confirmar propuesta: %s", exc)
            return JsonResponse({"error": "Error al ejecutar transacción. Intente nuevamente más tarde."}, status=500)


class CancelarPropuestaView(HakuApiBaseView):
    """Descarta y cancela la propuesta pendiente activa con protección transaccional OCC.

    POST /api/haku/cancelar/
    Payload: {"propuesta_id": "...", "conversacion_id": "..." (opcional)}
    """

    def post(self, request):
        datos = self._obtener_json(request)
        try:
            validados = ValidadorCancelacion.validar(datos)
        except ValidationError as err:
            err_dict = err.message_dict if hasattr(err, "message_dict") else {}
            cid_err = err_dict.get("conversacion_id", [])
            if any("conversacion_id_obligatorio" in str(m) for m in cid_err):
                return JsonResponse({"ok": False, "error": "conversacion_id_obligatorio", "mensaje": "Se requiere el parámetro 'conversacion_id'."}, status=400)
            return JsonResponse({"error": err_dict if err_dict else str(err)}, status=400)

        propuesta_id = validados["propuesta_id"]
        conversacion_id = validados["conversacion_id"]

        from .servicios_sesion import (
            ErrorConflictoOCC,
            bloquear_y_leer_sesion,
            guardar_sesion_transaccional,
            buscar_ejecucion_existente_ledger,
            es_error_bloqueo_sqlite,
            registrar_auditoria_cancelacion,
            registrar_auditoria_rechazo_seguridad,
            registrar_evidencia_terminal_sesion,
            obtener_evidencia_verificable_propuesta_terminal,
            ErrorRegistroEvidenciaTerminal,
            neutralizar_sesion_en_memoria,
        )
        from inventario.models import Auditoria

        res_ledger = buscar_ejecucion_existente_ledger(propuesta_id, request.user, conversacion_id_esperada=conversacion_id)
        if res_ledger:
            return JsonResponse({
                "error": "propuesta_ya_ejecutada",
                "mensaje": "Esta propuesta ya fue confirmada y ejecutada en inventario. No puede ser cancelada.",
            }, status=400)

        aud_cancel = Auditoria.objects.filter(
            tipo_entidad="PropuestaHaku",
            accion="PROPUESTA_CANCELADA",
            id_entidad=propuesta_id,
            usuario_id=getattr(request.user, "id", None),
        ).first()
        if aud_cancel:
            return JsonResponse({
                "exito": True,
                "idempotente": True,
                "estado": "CANCELADA",
                "mensaje_pantalla": "Propuesta cancelada con éxito.",
                "mensaje_verbal": "Operación cancelada. No se realizaron cambios.",
            })

        session_key = getattr(request.session, "session_key", "") or ""
        if not session_key:
            registrar_auditoria_cancelacion(request.user, propuesta_id)
            return JsonResponse({
                "exito": True,
                "mensaje_pantalla": "Propuesta cancelada con éxito.",
                "mensaje_verbal": "Operación cancelada. No se realizaron cambios.",
            })

        MAX_INTENTOS = 4
        for intento in range(MAX_INTENTOS):
            try:
                with transaction.atomic(using="default"):
                    session_obj, datos_sesion, raw_orig = bloquear_y_leer_sesion(session_key, request)
                    if not session_obj:
                        registrar_auditoria_cancelacion(request.user, propuesta_id)
                        return JsonResponse({
                            "exito": True,
                            "mensaje_pantalla": "Propuesta cancelada.",
                            "mensaje_verbal": "Operación cancelada.",
                        })

                    res_ledger = buscar_ejecucion_existente_ledger(propuesta_id, request.user, conversacion_id_esperada=conversacion_id)
                    if res_ledger:
                        return JsonResponse({
                            "error": "propuesta_ya_ejecutada",
                            "mensaje": "Esta propuesta ya fue confirmada y ejecutada en inventario. No puede ser cancelada.",
                        }, status=400)

                    propuesta_activa = datos_sesion.get("haku_propuesta_activa")
                    if not propuesta_activa:
                        aud_c = Auditoria.objects.filter(
                            tipo_entidad="PropuestaHaku",
                            accion="PROPUESTA_CANCELADA",
                            id_entidad=propuesta_id,
                            usuario_id=getattr(request.user, "id", None),
                        ).first()
                        if aud_c:
                            return JsonResponse({
                                "exito": True,
                                "idempotente": True,
                                "estado": "CANCELADA",
                                "mensaje_pantalla": "Esta propuesta ya había sido cancelada.",
                                "mensaje_verbal": "Operación cancelada.",
                            })

                        ultima_ejec = datos_sesion.get("haku_ultima_ejecucion") or {}
                        if str(ultima_ejec.get("propuesta_id") or "") == propuesta_id:
                            return JsonResponse({
                                "error": "propuesta_ya_ejecutada",
                                "mensaje": "Esta propuesta ya fue confirmada y ejecutada en inventario. No puede ser cancelada.",
                            }, status=400)
                        sustituida_id = datos_sesion.get("haku_propuesta_sustituida_id")
                        if sustituida_id and str(sustituida_id) == propuesta_id:
                            return JsonResponse({
                                "exito": True,
                                "estado": "SUSTITUIDA",
                                "mensaje_pantalla": "Esta propuesta ya había sido sustituida o cancelada.",
                                "mensaje_verbal": "Operación cancelada.",
                            })
                        ultimo_evento = datos_sesion.get("haku_ultimo_evento") or {}
                        if str(ultimo_evento.get("propuesta_id") or "") == propuesta_id:
                            return JsonResponse({
                                "exito": True,
                                "estado": ultimo_evento.get("estado", "CANCELADA"),
                                "mensaje_pantalla": "Esta propuesta ya había sido cancelada.",
                                "mensaje_verbal": "Operación cancelada.",
                            })
                        registrar_auditoria_cancelacion(request.user, propuesta_id)
                        return JsonResponse({
                            "exito": True,
                            "mensaje_pantalla": "No hay propuesta activa.",
                            "mensaje_verbal": "No hay operación pendiente.",
                        })

                    if str(propuesta_activa.get("id")) != propuesta_id:
                        sustituida_id = datos_sesion.get("haku_propuesta_sustituida_id")
                        if sustituida_id and str(sustituida_id) == propuesta_id:
                            return JsonResponse({
                                "error": "propuesta_sustituida",
                                "mensaje": "Esta propuesta ya fue cancelada porque se generó una solicitud más reciente. La propuesta actual se mantiene activa.",
                                "mensaje_pantalla": "Esta propuesta fue cancelada porque se generó una solicitud posterior en otra pestaña. La propuesta actual se mantiene activa.",
                                "mensaje_verbal": "Esta propuesta ya fue reemplazada por una solicitud más reciente.",
                            }, status=409)
                        return JsonResponse({"error": "El ID no coincide con la propuesta activa."}, status=400)

                    if str(propuesta_activa.get("usuario_id") or "") != str(request.user.id):
                        registrar_auditoria_rechazo_seguridad(
                            usuario=request.user,
                            propuesta_id=propuesta_id,
                            codigo_resultado="RECHAZADA_PROPIEDAD",
                            motivo="Propuesta pertenece a otro usuario al cancelar",
                            operacion_declarada=propuesta_activa.get("tipo_operacion"),
                            producto_id_declarado=propuesta_activa.get("producto_id"),
                            direccion_peer=request.META.get("REMOTE_ADDR"),
                            propuesta_verificada=False,
                        )
                        return JsonResponse({"error": "Esta propuesta pertenece a otro usuario y no puede ser cancelada."}, status=403)

                    token_sesion = datos_sesion.get("haku_session_token")
                    token_prop = propuesta_activa.get("session_token")
                    if (
                        not token_sesion
                        or not token_prop
                        or not isinstance(token_sesion, str)
                        or not isinstance(token_prop, str)
                        or not hmac.compare_digest(str(token_prop).strip(), str(token_sesion).strip())
                    ):
                        registrar_auditoria_rechazo_seguridad(
                            usuario=request.user,
                            propuesta_id=propuesta_id,
                            codigo_resultado="RECHAZADA_TOKEN",
                            motivo="Token de sesión inválido al cancelar",
                            operacion_declarada=propuesta_activa.get("tipo_operacion"),
                            producto_id_declarado=propuesta_activa.get("producto_id"),
                            direccion_peer=request.META.get("REMOTE_ADDR"),
                            propuesta_verificada=False,
                        )
                        return JsonResponse({"error": "La propuesta no corresponde a la sesión activa o carece de token de sesión válido."}, status=403)

                    # Validar correspondencia de conversación e identidad conversacional (H-02)
                    cid_prop = propuesta_activa.get("conversacion_id")
                    if not cid_prop or not str(cid_prop).strip():
                        registrar_auditoria_rechazo_seguridad(
                            usuario=request.user,
                            propuesta_id=propuesta_id,
                            codigo_resultado="RECHAZADA_INTEGRIDAD",
                            motivo="Propuesta carece de identidad conversacional válida al cancelar",
                            operacion_declarada=propuesta_activa.get("tipo_operacion"),
                            producto_id_declarado=propuesta_activa.get("producto_id"),
                            direccion_peer=request.META.get("REMOTE_ADDR"),
                            propuesta_verificada=False,
                        )
                        return JsonResponse({
                            "ok": False,
                            "error": "identidad_conversacion_invalida",
                            "mensaje": "La propuesta carece de una identidad conversacional válida.",
                        }, status=409)

                    try:
                        cid_prop_u = uuid.UUID(str(cid_prop).strip(), version=4)
                        cid_prop_str = str(cid_prop_u)
                    except (ValueError, TypeError, AttributeError):
                        return JsonResponse({
                            "ok": False,
                            "error": "identidad_conversacion_invalida",
                            "mensaje": "La propuesta contiene una identidad conversacional malformada.",
                        }, status=409)

                    if cid_prop_str != conversacion_id:
                        return JsonResponse({
                            "ok": False,
                            "error": "identidad_conversacion_invalida",
                            "mensaje": "La propuesta pertenece a otra pestaña o conversación.",
                        }, status=409)

                    # Resolución de bloqueo de modificación bajo los 4 casos estrictos
                    bloqueo = datos_sesion.get("haku_bloqueo_modificacion")
                    if bloqueo:
                        prop_bloq = str(bloqueo.get("propuesta_id_origen") or "").strip()
                        cid_bloq = str(bloqueo.get("conversacion_id") or "").strip()
                        uid_bloq = str(bloqueo.get("usuario_id") or "").strip()
                        cid_actual = str(cid_prop or "").strip()

                        # Caso 3: Bloqueo ajeno a la conversación o usuario
                        if (uid_bloq and uid_bloq != str(request.user.id)) or (cid_bloq and cid_actual and cid_bloq != cid_actual):
                            # Conservar intacto el bloqueo ajeno
                            pass
                        elif prop_bloq == propuesta_id:
                            # Caso 1: Bloqueo ligado a la propuesta que se cancela
                            datos_sesion["haku_bloqueo_modificacion"] = None
                        else:
                            # Bloqueo ligado a P0 distinta en la misma conversación/usuario
                            ev_term = obtener_evidencia_verificable_propuesta_terminal(datos_sesion, prop_bloq)
                            if ev_term:
                                # Caso 2: P0 demostradamente terminal
                                datos_sesion["haku_bloqueo_modificacion"] = None
                            else:
                                # Caso 4: P0 no demostrada terminal -> fail-closed
                                return JsonResponse({
                                    "error": "conflicto_integridad_bloqueo",
                                    "mensaje": "No se puede cancelar la propuesta porque existe un bloqueo de modificación sin evidencia terminal verificable.",
                                }, status=409)

                    registrar_evidencia_terminal_sesion(datos_sesion, propuesta_activa, "CANCELADA")
                    datos_sesion["haku_propuesta_activa"] = None
                    datos_sesion["haku_ultimo_evento"] = {
                        "propuesta_id": propuesta_id,
                        "usuario_id": str(request.user.id),
                        "estado": "CANCELADA",
                        "fecha": round(time.time(), 4),
                        "timestamp": round(time.time(), 4),
                    }
                    guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                    try:
                        from .contexto import HakuOperationalContext
                        estado_op_canc = HakuOperationalContext.obtener(request.user.id, session_key)
                        if estado_op_canc:
                            estado_op_canc.propuesta_activa = None
                            estado_op_canc.operacion_pendiente = None
                            HakuOperationalContext.guardar(estado_op_canc)
                    except Exception:
                        pass
                    registrar_auditoria_cancelacion(request.user, propuesta_id, propuesta_activa)

                    return JsonResponse({
                        "exito": True,
                        "estado": "CANCELADA",
                        "mensaje_pantalla": "Propuesta cancelada con éxito.",
                        "mensaje_verbal": "Operación cancelada. No se realizaron cambios.",
                    })
            except ErrorRegistroEvidenciaTerminal as err:
                logger.exception("Fallo crítico al registrar evidencia terminal en cancelación: %s", err)
                neutralizar_sesion_en_memoria(request)
                return JsonResponse({
                    "ok": False,
                    "error": "error_registro_evidencia_terminal",
                    "mensaje": "Fallo crítico al registrar la evidencia terminal de la propuesta.",
                }, status=500)
            except (OperationalError, ErrorConflictoOCC) as err:
                es_occ = isinstance(err, ErrorConflictoOCC)
                es_bloqueo = isinstance(err, OperationalError) and es_error_bloqueo_sqlite(err)
                if es_occ or es_bloqueo:
                    if intento < MAX_INTENTOS - 1:
                        time.sleep(0.05 * (2 ** intento))
                        continue
                    return JsonResponse({"error": "Conflicto de concurrencia al cancelar la propuesta. Intenta nuevamente."}, status=409)
                raise


class ActualizarDestinatarioPropuestaView(HakuApiBaseView):
    """Actualiza directamente el destinatario de la propuesta activa
    sin enviarlo al enrutador general de PLN de HAKU bajo protocolo transaccional único.

    POST /api/haku/propuesta/actualizar-destinatario/
    Payload: {"propuesta_id": "...", "destinatario": "...", "conversacion_id": "..." (opcional)}
    """

    def post(self, request):
        datos = self._obtener_json(request)
        try:
            validados = ValidadorActualizarDestinatario.validar(datos)
        except ValidationError as err:
            err_dict = err.message_dict if hasattr(err, "message_dict") else {}
            cid_err = err_dict.get("conversacion_id", [])
            if any("conversacion_id_obligatorio" in str(m) for m in cid_err):
                return JsonResponse({"ok": False, "error": "conversacion_id_obligatorio", "mensaje": "Se requiere el parámetro 'conversacion_id'."}, status=400)
            return JsonResponse({"error": err_dict if err_dict else str(err)}, status=400)

        propuesta_id = validados["propuesta_id"]
        destinatario = validados["destinatario"]
        conversacion_id = validados["conversacion_id"]

        session_key = getattr(request.session, "session_key", "") or ""
        from .servicios_sesion import (
            ErrorConflictoOCC,
            bloquear_y_leer_sesion,
            guardar_sesion_transaccional,
            buscar_ejecucion_existente_ledger,
            es_error_bloqueo_sqlite,
            registrar_auditoria_modificacion,
            registrar_auditoria_caducidad,
            registrar_auditoria_rechazo_seguridad,
            registrar_evidencia_terminal_sesion,
            evaluar_vigencia_propuesta,
            es_propuesta_contrato_anterior,
            invalidar_propuesta_pre_despliegue_en_transaccion,
            consultar_estado_propuesta_persistente,
            ErrorRegistroEvidenciaTerminal,
            neutralizar_sesion_en_memoria,
        )
        from .servicios_personas import capitalizar_nombre_propio
        from .services import HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, calcular_huella_propuesta, calcular_firma_propuesta, calcular_hash_token_sesion
        from inventario.models import Auditoria, Movimiento

        res_ledger = buscar_ejecucion_existente_ledger(propuesta_id, request.user, conversacion_id_esperada=conversacion_id)
        if res_ledger:
            return JsonResponse({
                "error": "propuesta_ya_ejecutada",
                "mensaje": "Esta propuesta ya fue confirmada y ejecutada en inventario. No puede ser modificada.",
            }, status=400)

        cands = list(Auditoria.objects.filter(
            tipo_entidad="PropuestaHaku",
            accion="PROPUESTA_MODIFICADA",
            usuario_id=getattr(request.user, "id", None),
            valores_nuevos__sustituye=propuesta_id,
        )[:2])
        if len(cands) == 1:
            return JsonResponse({
                "error": "propuesta_sustituida",
                "mensaje": "Esta propuesta fue cancelada porque se generó una solicitud posterior en otra pestaña.",
                "mensaje_pantalla": "Esta propuesta fue cancelada porque generaste una solicitud más reciente en otra pestaña. Por favor revisa la propuesta activa actual.",
                "mensaje_verbal": "Esta propuesta ya fue reemplazada por una solicitud más reciente.",
            }, status=409)
        if len(cands) >= 2:
            return JsonResponse({
                "error": "conflicto_integridad_linaje",
                "mensaje": "Se detectó una discrepancia en el historial de modificaciones de la propuesta.",
            }, status=409)

        MAX_INTENTOS = 4
        for intento in range(MAX_INTENTOS):
            try:
                with transaction.atomic(using="default"):
                    session_obj, datos_sesion, raw_orig = bloquear_y_leer_sesion(session_key, request)
                    if not session_obj:
                        return JsonResponse({"error": "No hay ninguna propuesta activa pendiente de confirmación o la sesión ha expirado."}, status=404)

                    res_ledger = buscar_ejecucion_existente_ledger(propuesta_id, request.user, conversacion_id_esperada=conversacion_id)
                    if res_ledger:
                        return JsonResponse({
                            "error": "propuesta_ya_ejecutada",
                            "mensaje": "Esta propuesta ya fue confirmada y ejecutada en inventario. No puede ser modificada.",
                        }, status=400)

                    propuesta_activa = datos_sesion.get("haku_propuesta_activa")
                    if not propuesta_activa:
                        ultimo_evento = datos_sesion.get("haku_ultimo_evento") or {}
                        if str(ultimo_evento.get("propuesta_id") or "") == propuesta_id:
                            estado_ev = ultimo_evento.get("estado")
                            if estado_ev == "CANCELADA":
                                return JsonResponse({
                                    "error": "propuesta_cancelada",
                                    "mensaje": "Esta propuesta fue cancelada y no puede ser modificada.",
                                }, status=409)
                            elif estado_ev == "SUSTITUIDA":
                                return JsonResponse({
                                    "error": "propuesta_sustituida",
                                    "mensaje": "Esta propuesta fue sustituida por una solicitud más reciente.",
                                }, status=409)
                        ultima_ejec = datos_sesion.get("haku_ultima_ejecucion") or {}
                        if str(ultima_ejec.get("propuesta_id") or "") == propuesta_id:
                            return JsonResponse({
                                "error": "propuesta_ya_ejecutada",
                                "mensaje": "Esta propuesta ya fue confirmada y ejecutada en inventario. No puede ser modificada.",
                            }, status=400)
                        return JsonResponse({"error": "No hay ninguna propuesta activa pendiente de confirmación o la sesión ha expirado."}, status=404)

                    if not propuesta_activa.get("usuario_id") or str(propuesta_activa.get("usuario_id")) != str(request.user.id):
                        registrar_auditoria_rechazo_seguridad(
                            usuario=request.user,
                            propuesta_id=propuesta_id,
                            codigo_resultado="RECHAZADA_PROPIEDAD",
                            motivo="Propuesta pertenece a otro usuario al actualizar destinatario",
                            operacion_declarada=propuesta_activa.get("tipo_operacion"),
                            producto_id_declarado=propuesta_activa.get("producto_id"),
                            direccion_peer=request.META.get("REMOTE_ADDR"),
                            propuesta_verificada=False,
                        )
                        return JsonResponse({"error": "Esta propuesta pertenece a otro usuario y no puede ser modificada."}, status=403)

                    token_sesion = datos_sesion.get("haku_session_token")
                    token_prop = propuesta_activa.get("session_token")
                    if (
                        not token_sesion
                        or not token_prop
                        or not isinstance(token_sesion, str)
                        or not isinstance(token_prop, str)
                        or not hmac.compare_digest(str(token_prop).strip(), str(token_sesion).strip())
                    ):
                        registrar_auditoria_rechazo_seguridad(
                            usuario=request.user,
                            propuesta_id=propuesta_id,
                            codigo_resultado="RECHAZADA_TOKEN",
                            motivo="Token de sesión no coincide al actualizar destinatario",
                            operacion_declarada=propuesta_activa.get("tipo_operacion"),
                            producto_id_declarado=propuesta_activa.get("producto_id"),
                            direccion_peer=request.META.get("REMOTE_ADDR"),
                            propuesta_verificada=False,
                        )
                        return JsonResponse({"error": "La propuesta no corresponde a la sesión activa o carece de token de sesión válido."}, status=403)

                    if str(propuesta_activa.get("id")) != propuesta_id:
                        sustituida_id = datos_sesion.get("haku_propuesta_sustituida_id")
                        if sustituida_id and str(sustituida_id) == propuesta_id:
                            return JsonResponse({
                                "error": "propuesta_sustituida",
                                "mensaje": "Esta propuesta fue cancelada porque se generó una solicitud posterior en otra pestaña.",
                                "mensaje_pantalla": "Esta propuesta fue cancelada porque generaste una solicitud más reciente en otra pestaña. Por favor revisa la propuesta activa actual.",
                                "mensaje_verbal": "Esta propuesta fue reemplazada por una solicitud más reciente.",
                            }, status=409)
                        return JsonResponse({"error": "El ID no coincide con la propuesta activa."}, status=400)

                    # Validar correspondencia de conversación e identidad conversacional (H-02)
                    cid_prop = propuesta_activa.get("conversacion_id")
                    if not cid_prop or not str(cid_prop).strip():
                        registrar_auditoria_rechazo_seguridad(
                            usuario=request.user,
                            propuesta_id=propuesta_id,
                            codigo_resultado="RECHAZADA_INTEGRIDAD",
                            motivo="Propuesta carece de identidad conversacional válida al actualizar destinatario",
                            operacion_declarada=propuesta_activa.get("tipo_operacion"),
                            producto_id_declarado=propuesta_activa.get("producto_id"),
                            direccion_peer=request.META.get("REMOTE_ADDR"),
                            propuesta_verificada=False,
                        )
                        return JsonResponse({
                            "ok": False,
                            "error": "identidad_conversacion_invalida",
                            "mensaje": "La propuesta carece de una identidad conversacional válida.",
                        }, status=409)

                    try:
                        cid_prop_u = uuid.UUID(str(cid_prop).strip(), version=4)
                        cid_prop_str = str(cid_prop_u)
                    except (ValueError, TypeError, AttributeError):
                        return JsonResponse({
                            "ok": False,
                            "error": "identidad_conversacion_invalida",
                            "mensaje": "La propuesta contiene una identidad conversacional malformada.",
                        }, status=409)

                    if cid_prop_str != conversacion_id:
                        return JsonResponse({
                            "ok": False,
                            "error": "identidad_conversacion_invalida",
                            "mensaje": "La propuesta pertenece a otra pestaña o conversación.",
                        }, status=409)

                    # Validar contrato pre-despliegue (R-06)
                    if es_propuesta_contrato_anterior(propuesta_activa):
                        invalidar_propuesta_pre_despliegue_en_transaccion(session_key, request, propuesta_id)
                        return JsonResponse({
                            "error": "propuesta_contrato_anterior",
                            "mensaje": "La propuesta pertenece a una versión anterior del contrato criptográfico y fue invalidada de manera segura. Por favor genere una nueva propuesta.",
                        }, status=409)

                    es_valida_temporal, esta_caducada = evaluar_vigencia_propuesta(propuesta_activa.get("expira_en"))
                    if not es_valida_temporal:
                        registrar_auditoria_rechazo_seguridad(
                            usuario=request.user,
                            propuesta_id=propuesta_id,
                            codigo_resultado="RECHAZADA_INTEGRIDAD",
                            motivo="Integridad temporal inválida al actualizar destinatario",
                            operacion_declarada=propuesta_activa.get("tipo_operacion"),
                            producto_id_declarado=propuesta_activa.get("producto_id"),
                            direccion_peer=request.META.get("REMOTE_ADDR"),
                            propuesta_verificada=False,
                        )
                        registrar_evidencia_terminal_sesion(datos_sesion, propuesta_activa, "RECHAZADA")
                        datos_sesion["haku_propuesta_activa"] = None
                        datos_sesion["haku_ultimo_evento"] = {
                            "propuesta_id": propuesta_id,
                            "usuario_id": str(request.user.id),
                            "estado": "RECHAZADA",
                            "fecha": round(time.time(), 4),
                            "timestamp": round(time.time(), 4),
                        }
                        guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                        return JsonResponse({"error": "La propuesta contiene datos temporales no válidos."}, status=400)

                    if esta_caducada:
                        registrar_auditoria_caducidad(request.user, propuesta_id, propuesta_activa)
                        registrar_evidencia_terminal_sesion(datos_sesion, propuesta_activa, "CADUCADA")
                        datos_sesion["haku_propuesta_activa"] = None
                        datos_sesion["haku_ultimo_evento"] = {
                            "propuesta_id": propuesta_id,
                            "usuario_id": str(request.user.id),
                            "estado": "CADUCADA",
                            "fecha": round(time.time(), 4),
                            "timestamp": round(time.time(), 4),
                        }
                        guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                        return JsonResponse({"error": "La propuesta ha caducado por superar el límite de tiempo. Por favor genera una nueva solicitud."}, status=400)

                    if propuesta_activa.get("confirmada") or propuesta_activa.get("estado") == "confirmada":
                        return JsonResponse({"error": "Esta propuesta ya fue confirmada y no puede ser modificada."}, status=400)

                    tipo_op_dest = str(propuesta_activa.get("tipo_operacion") or "").upper()
                    if tipo_op_dest in (Movimiento.Tipo.DEVOLUCION, "DEVOLUCION"):
                        return JsonResponse(
                            {"error": "No se puede cambiar el responsable de una devolución porque pertenece a un préstamo existente. Cancele y solicite la devolución del préstamo correcto."},
                            status=400,
                        )
                    if tipo_op_dest not in (Movimiento.Tipo.PRESTAMO, "PRESTAMO", Movimiento.Tipo.SALIDA, "SALIDA"):
                        return JsonResponse(
                            {"error": "Esta operación no admite actualización de destinatario o responsable."},
                            status=400,
                        )

                    dest_limpio = capitalizar_nombre_propio(destinatario)

                    # H-05: Capturar copia independiente y profunda del contenido original de P1
                    propuesta_p1_original = copy.deepcopy(propuesta_activa)
                    propuesta_previa_id = propuesta_p1_original["id"]
                    if not propuesta_p1_original.get("session_token_hash"):
                        propuesta_p1_original["session_token_hash"] = calcular_hash_token_sesion(token_sesion)
                    if not propuesta_p1_original.get("usuario_id"):
                        propuesta_p1_original["usuario_id"] = str(request.user.id)

                    # Construir P2 como nueva propuesta independiente
                    nueva_propuesta = copy.deepcopy(propuesta_p1_original)
                    nueva_id = str(uuid.uuid4())
                    nueva_propuesta["id"] = nueva_id
                    nueva_propuesta["sustituye"] = propuesta_previa_id
                    dest_prev_upd = str(propuesta_p1_original.get("destinatario") or "")
                    nueva_propuesta["destinatario"] = dest_limpio
                    motivo_prev_upd = str(nueva_propuesta.get("motivo") or "")
                    if tipo_op_dest in (Movimiento.Tipo.PRESTAMO, "PRESTAMO"):
                        nueva_propuesta["motivo"] = f"Préstamo registrado por Haku a {dest_limpio}"
                    elif tipo_op_dest in (Movimiento.Tipo.SALIDA, "SALIDA"):
                        if dest_prev_upd and dest_prev_upd in motivo_prev_upd:
                            nueva_propuesta["motivo"] = motivo_prev_upd.replace(dest_prev_upd, dest_limpio)
                        elif not motivo_prev_upd or motivo_prev_upd.startswith("Salida registrada por Haku"):
                            nueva_propuesta["motivo"] = f"Salida registrada por Haku para {dest_limpio}"
                    nueva_propuesta["expira_en"] = round(time.time() + HAKU_PROPUESTA_TIMEOUT_SEGUNDOS, 4)
                    nueva_propuesta["creado_en"] = round(time.time(), 4)
                    nueva_propuesta.pop("timestamp", None)
                    nueva_propuesta["conversacion_id"] = conversacion_id
                    nueva_propuesta["session_token_hash"] = calcular_hash_token_sesion(token_sesion)
                    nueva_propuesta["huella_payload"] = calcular_huella_propuesta(nueva_propuesta)
                    nueva_propuesta["firma_hmac"] = calcular_firma_propuesta(nueva_propuesta)

                    # H-05: Registrar evidencia terminal para P1 como SUSTITUIDA con los datos exactos y firma original de P1
                    registrar_evidencia_terminal_sesion(
                        datos_sesion,
                        propuesta_p1_original,
                        "SUSTITUIDA",
                        sustituida_por_id=nueva_propuesta["id"],
                    )

                    # Liberar bloqueo si estaba ligado a P1
                    bloqueo = datos_sesion.get("haku_bloqueo_modificacion")
                    if bloqueo and str(bloqueo.get("propuesta_id_origen") or "") == propuesta_previa_id:
                        datos_sesion["haku_bloqueo_modificacion"] = None

                    datos_sesion["haku_propuesta_activa"] = nueva_propuesta
                    datos_sesion["haku_propuesta_sustituida_id"] = propuesta_previa_id
                    datos_sesion["haku_propuesta_sustituida_timestamp"] = round(time.time(), 4)
                    datos_sesion["haku_ultimo_evento"] = {
                        "propuesta_id": propuesta_previa_id,
                        "usuario_id": str(request.user.id),
                        "estado": "SUSTITUIDA",
                        "fecha": round(time.time(), 4),
                        "timestamp": round(time.time(), 4),
                    }
                    guardar_sesion_transaccional(session_obj, datos_sesion, raw_orig, request)
                    try:
                        from .contexto import HakuOperationalContext
                        estado_op_upd = HakuOperationalContext.obtener(request.user.id, session_key)
                        if estado_op_upd:
                            estado_op_upd.propuesta_activa = nueva_propuesta
                            HakuOperationalContext.guardar(estado_op_upd)
                    except Exception:
                        pass
                    registrar_auditoria_modificacion(
                        usuario=request.user,
                        propuesta_anterior_id=propuesta_previa_id,
                        propuesta_nueva=nueva_propuesta,
                        campos_modificados=["destinatario"],
                    )

                    return JsonResponse({
                        "exito": True,
                        "estado": "PENDIENTE",
                        "propuesta": nueva_propuesta,
                        "mensaje": f"Destinatario actualizado a {dest_limpio}.",
                    })
            except ErrorRegistroEvidenciaTerminal as err:
                logger.exception("Fallo crítico al registrar evidencia terminal en actualización: %s", err)
                neutralizar_sesion_en_memoria(request)
                return JsonResponse({
                    "ok": False,
                    "error": "error_registro_evidencia_terminal",
                    "mensaje": "Fallo crítico al registrar la evidencia terminal de la propuesta.",
                }, status=500)
            except (OperationalError, ErrorConflictoOCC) as err:
                es_occ = isinstance(err, ErrorConflictoOCC)
                es_bloqueo = isinstance(err, OperationalError) and es_error_bloqueo_sqlite(err)
                if es_occ or es_bloqueo:
                    if intento < MAX_INTENTOS - 1:
                        time.sleep(0.05 * (2 ** intento))
                        continue
                    return JsonResponse({"error": "Conflicto de concurrencia al actualizar la propuesta. Intenta nuevamente."}, status=409)
                raise




class ConsultarStockView(HakuApiBaseView):
    """Consulta directa de existencias por término de búsqueda.

    GET /api/haku/consultar-stock/?q=cable
    """

    def get(self, request):
        if not (request.user.es_administrador or request.user.tiene_permiso("catalogo.ver")):
            raise PermissionDenied("No cuentas con el permiso requerido (catalogo.ver) para consultar el catálogo o existencias.")

        termino = request.GET.get("q", "").strip()
        qs = Existencia.objects.filter(
            producto__activo=True,
            producto__archivado=False,
        ).select_related("producto", "producto__categoria", "almacen")

        if not (request.user.es_administrador or getattr(request.user, "acceso_todos_almacenes", False)):
            alms_acc = request.user.obtener_almacenes_accesibles() if hasattr(request.user, "obtener_almacenes_accesibles") else []
            qs = qs.filter(almacen__in=alms_acc)

        if termino:
            from django.db.models import Q as _Q
            qs = qs.filter(
                _Q(producto__nombre__icontains=termino)
                | _Q(producto__codigo_interno__icontains=termino)
            )

        productos = []
        for e in qs[:20]:
            productos.append({
                "id": str(e.producto.id),
                "codigo": e.producto.codigo_interno,
                "codigo_barras": e.producto.codigo_barras,
                "nombre": e.producto.nombre,
                "categoria": e.producto.categoria.nombre if e.producto.categoria else "",
                "almacen": e.almacen.nombre,
                "stock_disponible": e.stock_disponible,
                "stock_en_prestamo": e.stock_en_prestamo,
                "stock_minimo": e.producto.stock_minimo,
                "estado_alerta": "SIN_STOCK" if e.sin_stock else ("BAJO_STOCK" if e.stock_bajo else "NORMAL"),
            })

        return JsonResponse({"productos": productos, "total": len(productos)})


class AlertasStockView(HakuApiBaseView):
    """Consulta de productos con alerta de stock bajo o sin stock.

    GET /api/haku/alertas/
    """

    def get(self, request):
        if not (request.user.es_administrador or request.user.tiene_permiso("analitica.ver_alertas")):
            raise PermissionDenied("No cuentas con el permiso requerido (analitica.ver_alertas) para consultar las alertas de stock.")

        qs = Existencia.objects.filter(
            producto__activo=True,
            producto__archivado=False,
        ).select_related("producto", "almacen")
        if not (request.user.es_administrador or getattr(request.user, "acceso_todos_almacenes", False)):
            alms_acc = request.user.obtener_almacenes_accesibles() if hasattr(request.user, "obtener_almacenes_accesibles") else []
            qs = qs.filter(almacen__in=alms_acc)

        sin_stock = []
        bajo_stock = []

        for e in qs:
            if e.sin_stock:
                sin_stock.append({"id": str(e.producto.id), "codigo": e.producto.codigo_interno, "nombre": e.producto.nombre, "almacen": e.almacen.nombre})
            elif e.stock_bajo:
                bajo_stock.append({
                    "id": str(e.producto.id),
                    "codigo": e.producto.codigo_interno,
                    "nombre": e.producto.nombre,
                    "almacen": e.almacen.nombre,
                    "disponible": e.stock_disponible,
                    "minimo": e.producto.stock_minimo,
                })

        return JsonResponse({
            "sin_stock": sin_stock,
            "bajo_stock": bajo_stock,
            "total_alertas": len(sin_stock) + len(bajo_stock),
        })


class FaltantesReposicionView(HakuApiBaseView):
    """Consulta estructurada de reposición de inventario («¿Qué hace falta por pedir?»).
    
    Distingue:
    - Productos agotados sin existencias disponibles (con stock prestado documentado).
    - Productos con bajo stock (por debajo del mínimo).
    - Recomendación prudente de compra hacia el stock mínimo (sin inventar algoritmos ficticios).
    """

    def get(self, request, *args, **kwargs):
        if not (request.user.es_administrador or request.user.tiene_permiso("analitica.ver_alertas")):
            return JsonResponse({"error": "No cuentas con el permiso requerido (analitica.ver_alertas) para consultar la reposición de existencias."}, status=403)

        from .services import procesar_comando_inventario

        almacen_id = request.GET.get("almacen_id")
        almacen_obj = None
        if almacen_id:
            try:
                almacen_obj = Almacen.objects.get(id=almacen_id, activo=True)
            except (Almacen.DoesNotExist, ValueError):
                return JsonResponse({"error": "Almacén no encontrado o inactivo."}, status=404)
            if not request.user.tiene_acceso_almacen(almacen_obj):
                return JsonResponse({"error": "No tienes autorización para consultar este almacén."}, status=403)

        if almacen_obj:
            mensaje = f"¿Qué hace falta por pedir en el Almacén {almacen_obj.nombre}?"
        else:
            mensaje = "¿Qué hace falta por pedir?"

        res = procesar_comando_inventario(
            usuario=request.user,
            mensaje=mensaje,
            sesion=getattr(request, "session", None),
        )
        return JsonResponse(res)


class PrestamosActivosView(HakuApiBaseView):
    """Consulta de préstamos activos.

    GET /api/haku/prestamos-activos/
    """

    def get(self, request):
        if not (request.user.es_administrador or request.user.tiene_permiso("movimientos.ver")):
            raise PermissionDenied("No cuentas con el permiso requerido (movimientos.ver) para consultar los préstamos activos.")

        qs = Prestamo.objects.filter(
            estado__in=[Prestamo.Estado.ABIERTO, Prestamo.Estado.DEVUELTO_PARCIAL]
        ).select_related("producto", "almacen", "usuario")

        if not (request.user.es_administrador or getattr(request.user, "acceso_todos_almacenes", False)):
            alms_acc = request.user.obtener_almacenes_accesibles() if hasattr(request.user, "obtener_almacenes_accesibles") else []
            qs = qs.filter(almacen__in=alms_acc)

        prestamos = []
        for p in qs:
            prestamos.append({
                "id": p.id,
                "producto": p.producto.nombre,
                "codigo": p.producto.codigo_interno,
                "almacen": p.almacen.nombre,
                "destinatario": p.destinatario,
                "cantidad_prestada": p.cantidad_original,
                "cantidad_pendiente": p.cantidad_pendiente_devolucion,
                "fecha_prestamo": p.fecha_creacion.isoformat() if p.fecha_creacion else "",
            })

        return JsonResponse({"prestamos": prestamos, "total": len(prestamos)})


class TranscribirVozView(HakuApiBaseView):
    """Endpoint de Speech-to-Text: recibe audio desde el navegador y transcribe.

    POST /api/haku/voz/transcribir/
    Multipart form con archivo 'audio', o cuerpo binario crudo.
    """

    def post(self, request):
        t0_http = time.perf_counter()
        int_id, req_id = obtener_ids_haku(request)

        audio_file = request.FILES.get("audio")
        if audio_file:
            audio_bytes = audio_file.read()
        else:
            audio_bytes = request.body

        if not audio_bytes:
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "stt_http_total", dt_total, request_id=req_id, status="no_audio")
            resp = JsonResponse({"error": "No se recibió archivo de audio."}, status=400)
            return adjuntar_headers_haku(resp, int_id, req_id, stt_http_total=dt_total)

        fname = getattr(audio_file, "name", "raw") if audio_file else "raw"
        content_type = getattr(audio_file, "content_type", "unknown") if audio_file else request.content_type
        captura_qa = guardar_audio_qa(
            audio_bytes,
            interaction_id=int_id,
            request_id=req_id,
            content_type=content_type,
        )
        logger.debug(
            "[HakuSTT] Petición recibida: int_id=%s, archivo='%s', MIME='%s', bytes=%d",
            int_id, fname, content_type, len(audio_bytes)
        )

        try:
            res_audio = procesar_audio_y_transcribir(audio_bytes, interaction_id=int_id)
            metricas = res_audio.get("metricas", {})
            actualizar_metadatos_audio_qa(
                captura_qa,
                status=str(res_audio.get("estado") or "desconocido"),
                transcript=str(res_audio.get("texto") or ""),
                metrics=metricas if isinstance(metricas, dict) else {},
            )
            orig = metricas.get("metricas_originales", {})

            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "stt_http_total", dt_total, request_id=req_id, audio_bytes=len(audio_bytes))

            tiempos_stt = metricas.get("tiempos", {})
            inference_ms = tiempos_stt.get("inference_ms") or tiempos_stt.get("stt_inference")

            dur_orig = orig.get("duracion_seg", 0.0)
            dur_vad = metricas.get("duracion_seg", 0.0)
            estado = res_audio.get("estado")
            motivo = res_audio.get("motivo")
            texto_transcrito = res_audio.get("texto", "")

            # Trazabilidad estructurada y segura (sin registrar transcripciones completas de usuarios por defecto)
            log_debug_texto = getattr(settings, "HAKU_STT_DEBUG_TEXTO", False)
            detalle_texto = (
                f", texto='{texto_transcrito}'"
                if log_debug_texto
                else f", chars={len(texto_transcrito)}, palabras={len(texto_transcrito.split())}"
            )

            logger.info(
                "[HakuSTT] interaction=%s estado=%s motivo=%s dur_orig=%.3fs dur_vad=%.3fs stt_infer_ms=%.1f%s",
                int_id,
                estado,
                motivo,
                dur_orig,
                dur_vad,
                inference_ms if inference_ms is not None else 0.0,
                detalle_texto,
            )

            if estado == "SIN_VOZ" and motivo == "ERROR_DECODIFICACION":
                resp = JsonResponse({
                    "status": "error_audio",
                    "estado": "ERROR_AUDIO",
                    "motivo": motivo,
                    "texto": "",
                    "mensaje_pantalla": "No pude leer el formato de audio recibido. Intenta grabar otra vez.",
                    "mensaje_verbal": "No pude leer el audio. Intenta grabar otra vez.",
                }, status=415)
                return adjuntar_headers_haku(resp, int_id, req_id, stt_http_total=dt_total, stt_inference=inference_ms)

            if estado == "SIN_VOZ":
                resp = JsonResponse({
                    "status": "silencio",
                    "estado": "SIN_VOZ",
                    "motivo": motivo or "AUDIO_SILENCIOSO",
                    "texto": "",
                    "mensaje_pantalla": "No se detectó voz audible en el micrófono.",
                    "mensaje_verbal": "No escuché ningún comando. Por favor intenta hablar de nuevo.",
                    "metricas": metricas,
                })
                return adjuntar_headers_haku(resp, int_id, req_id, stt_http_total=dt_total, stt_inference=inference_ms)

            if estado == "MOTOR_CARGANDO":
                resp = JsonResponse({
                    "status": "motor_cargando",
                    "estado": "MOTOR_CARGANDO",
                    "motivo": motivo or "MODELO_CARGANDO",
                    "texto": "",
                    "mensaje_pantalla": "El motor de voz aún está terminando de iniciar. Espera unos segundos e intenta hablar de nuevo.",
                    "mensaje_verbal": "El motor de voz está terminando de iniciar. Espera unos segundos e intenta de nuevo.",
                    "error": res_audio.get("error", "Motor STT inicializando"),
                    "metricas": metricas,
                }, status=503)
                resp["Retry-After"] = "5"
                return adjuntar_headers_haku(resp, int_id, req_id, stt_http_total=dt_total, stt_inference=inference_ms)

            if estado == "ERROR_TRANSCRIPCION":
                resp = JsonResponse({
                    "status": "error_transcripcion",
                    "estado": "ERROR_TRANSCRIPCION",
                    "motivo": motivo or "ERROR_STT",
                    "texto": "",
                    "mensaje_pantalla": "Hubo un problema al transcribir el audio en el servidor.",
                    "mensaje_verbal": "Hubo un error de transcripción. Por favor intenta de nuevo.",
                    "error": res_audio.get("error", "Error en motor STT"),
                    "metricas": metricas,
                })
                return adjuntar_headers_haku(resp, int_id, req_id, stt_http_total=dt_total, stt_inference=inference_ms)

            resp = JsonResponse({
                "status": "ok",
                "estado": "VOZ_DETECTADA",
                "texto": texto_transcrito,
                "longitud_bytes": len(audio_bytes),
                "metricas": metricas,
            })
            return adjuntar_headers_haku(resp, int_id, req_id, stt_http_total=dt_total, stt_inference=inference_ms)
        except HakuOcupadoError:
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "stt_http_total", dt_total, request_id=req_id, status="haku_ocupado")
            logger.warning("[HakuSTT] interaction=%s estado=HAKU_OCUPADO", int_id)
            resp = JsonResponse({
                "estado": "HAKU_OCUPADO",
                "mensaje_pantalla": "Hacu Inventario está atendiendo otra solicitud. Por favor, inténtelo nuevamente en unos instantes.",
                "mensaje_verbal": "Jacu Inventario está atendiendo otra solicitud. Por favor, inténtelo nuevamente en unos instantes.",
                "error": "El servicio de audio se encuentra ocupado. Por favor intente nuevamente.",
            }, status=503)
            resp["Retry-After"] = "5"
            return adjuntar_headers_haku(resp, int_id, req_id, stt_http_total=dt_total)
        except Exception as exc:
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "stt_http_total", dt_total, request_id=req_id, status="error")
            logger.exception("[HakuSTT] interaction=%s error_inesperado=%s", int_id, exc)
            resp = JsonResponse({"error": "Error interno en transcripción de audio."}, status=500)
            return adjuntar_headers_haku(resp, int_id, req_id, stt_http_total=dt_total)


class SintetizarVozView(HakuApiBaseView):
    """Endpoint de Text-to-Speech: sintetiza audio WAV para el navegador cliente.

    POST /api/haku/voz/sintetizar/
    Payload: {"texto": "..."}
    Retorna: audio/wav stream binario.

    CRÍTICO: Este audio suena exclusivamente en el dispositivo del cliente.
    """

    def post(self, request):
        t0_http = time.perf_counter()
        int_id, req_id = obtener_ids_haku(request)
        datos = self._obtener_json(request)
        try:
            validados = ValidadorSintesisVoz.validar(datos)
        except ValidationError as err:
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "tts_http_total", dt_total, request_id=req_id, status="validation_error")
            resp = JsonResponse({"error": str(err)}, status=400)
            return adjuntar_headers_haku(resp, int_id, req_id, tts_http_total=dt_total)

        texto = validados["texto"]
        try:
            from .audio.synthesizer import HakuSynthesizerError
            wav_bytes = sintetizar_audio_bytes(texto, interaction_id=int_id)
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "tts_http_total", dt_total, request_id=req_id, wav_bytes=len(wav_bytes))

            response = HttpResponse(wav_bytes, content_type="audio/wav")
            response["Content-Disposition"] = 'inline; filename="haku_voz.wav"'
            response["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response["Content-Length"] = str(len(wav_bytes))
            adjuntar_headers_haku(response, int_id, req_id, tts_http_total=dt_total)
            response["X-Haku-TTS-Bytes"] = str(len(wav_bytes))
            return response
        except HakuSynthesizerError as exc:
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "tts_http_total", dt_total, request_id=req_id, status="tts_error")
            logger.error("Error en síntesis TTS de Haku: %s", exc)
            resp = JsonResponse({
                "status": "error_tts",
                "mensaje": "La síntesis de voz no está disponible actualmente.",
                "error": "Servicio de síntesis de voz temporalmente no disponible.",
            }, status=503)
            resp["Retry-After"] = "5"
            return adjuntar_headers_haku(resp, int_id, req_id, tts_http_total=dt_total)
        except Exception as exc:
            dt_total = (time.perf_counter() - t0_http) * 1000.0
            registrar_timing(int_id, "tts_http_total", dt_total, request_id=req_id, status="error")
            logger.exception("Error inesperado en síntesis de voz: %s", exc)
            resp = JsonResponse({"error": "Error interno en síntesis de voz."}, status=500)
            return adjuntar_headers_haku(resp, int_id, req_id, tts_http_total=dt_total)

class ProductoCrearHakuView(HakuApiBaseView):
    """Crea un nuevo producto en el inventario validando campos y permisos existentes.

    POST /api/haku/producto/crear/
    """

    def post(self, request):
        if not (request.user.es_administrador or request.user.tiene_permiso("catalogo.crear")):
            raise PermissionDenied("No cuentas con el permiso requerido (catalogo.crear) para registrar productos.")

        datos = self._obtener_json(request)
        nombre = str(datos.get("nombre", "")).strip()
        categoria_id = datos.get("categoria_id")
        codigo_barras = str(datos.get("codigo_barras", "")).strip() or None
        stock_minimo_raw = datos.get("stock_minimo")
        es_prestable_raw = datos.get("es_prestable", False)
        if isinstance(es_prestable_raw, bool):
            es_prestable = es_prestable_raw
        else:
            es_prestable = str(es_prestable_raw).strip().lower() in ("true", "1", "si", "sí", "yes", "on")
        nombre_empaque = str(datos.get("nombre_empaque", "")).strip()
        unidades_por_empaque = datos.get("unidades_por_empaque")
        descripcion = str(datos.get("descripcion", "")).strip()

        if not nombre:
            return JsonResponse({"error": "El nombre del producto es obligatorio."}, status=400)
        if not categoria_id:
            return JsonResponse({"error": "La categoría es obligatoria."}, status=400)
        try:
            categoria_id_int = int(categoria_id)
        except (ValueError, TypeError):
            return JsonResponse({"error": "El identificador de la categoría no es válido."}, status=400)

        from inventario.models import Categoria, Producto, Auditoria, Almacen, Existencia, Movimiento, Ubicacion
        categoria = Categoria.objects.select_related("almacen_principal").filter(
            id=categoria_id_int, activo=True, eliminado=False
        ).first()
        if not categoria:
            return JsonResponse({"error": "La categoría seleccionada no existe o está inactiva."}, status=400)

        # RBAC: verificar acceso del usuario al almacén principal de la categoría
        if not request.user.tiene_acceso_almacen(categoria.almacen_principal):
            raise PermissionDenied("No tienes permisos para registrar productos en categorías de otros almacenes.")

        if codigo_barras:
            if Producto.objects.filter(codigo_barras=codigo_barras, archivado=False).exists():
                return JsonResponse({"error": f"El código de barras '{codigo_barras}' ya está registrado en otro producto."}, status=400)

        if stock_minimo_raw is None or stock_minimo_raw == "":
            return JsonResponse({"error": "El stock mínimo es obligatorio. Debe seleccionar un valor explícito (0 o mayor)."}, status=400)

        try:
            stock_minimo = int(stock_minimo_raw)
            if stock_minimo < 0:
                return JsonResponse({"error": "El stock mínimo no puede ser negativo."}, status=400)
        except (ValueError, TypeError):
            return JsonResponse({"error": "El stock mínimo debe ser un número entero válido (0 o mayor)."}, status=400)

        stock_inicial_raw = datos.get("stock_inicial", 0)
        if stock_inicial_raw is None or stock_inicial_raw == "":
            stock_inicial = 0
        else:
            try:
                stock_inicial = int(stock_inicial_raw)
                if stock_inicial < 0:
                    return JsonResponse({"error": "El stock inicial no puede ser negativo."}, status=400)
            except (ValueError, TypeError):
                return JsonResponse({"error": "El stock inicial debe ser un número entero válido (0 o mayor)."}, status=400)

        if unidades_por_empaque:
            try:
                unidades_por_empaque = int(unidades_por_empaque)
                if unidades_por_empaque <= 0:
                    return JsonResponse({"error": "Las unidades por empaque deben ser mayores a cero."}, status=400)
            except (ValueError, TypeError):
                return JsonResponse({"error": "Las unidades por empaque deben ser un número válido."}, status=400)
        else:
            unidades_por_empaque = None

        almacen_id_raw = datos.get("almacen_id")
        if almacen_id_raw is not None and almacen_id_raw != "":
            try:
                almacen_id_int = int(almacen_id_raw)
            except (ValueError, TypeError):
                return JsonResponse({"error": "El identificador del almacén no es válido."}, status=400)
            if almacen_id_int != categoria.almacen_principal_id:
                return JsonResponse({
                    "error": (
                        f"La categoría '{categoria.nombre}' pertenece exclusivamente al almacén "
                        f"'{categoria.almacen_principal.nombre}'. No puede asignarse a otro almacén."
                    )
                }, status=400)

        ubicacion_obj = None
        ubicacion_id = datos.get("ubicacion_id")
        if ubicacion_id is not None and ubicacion_id != "":
            try:
                ubicacion_id_int = int(ubicacion_id)
            except (ValueError, TypeError):
                return JsonResponse({"error": "El identificador de la ubicación no es válido."}, status=400)
            ubicacion_obj = Ubicacion.objects.filter(id=ubicacion_id_int, activo=True, eliminado=False).first()
            if not ubicacion_obj:
                return JsonResponse({"error": "La ubicación seleccionada no existe o está inactiva."}, status=400)
            if ubicacion_obj.almacen_id != categoria.almacen_principal_id:
                return JsonResponse({
                    "error": (
                        f"La ubicación '{ubicacion_obj.nombre}' no pertenece al almacén "
                        f"'{categoria.almacen_principal.nombre}' de la categoría seleccionada."
                    )
                }, status=400)

        from django.db import transaction
        try:
            with transaction.atomic():
                if codigo_barras:
                    Producto.objects.filter(codigo_barras=codigo_barras, archivado=True).update(codigo_barras=None)

                producto = Producto(
                    nombre=nombre,
                    categoria=categoria,
                    ubicacion=ubicacion_obj,
                    codigo_barras=codigo_barras,
                    stock_inicial=stock_inicial,
                    stock_minimo=stock_minimo,
                    es_prestable=es_prestable,
                    nombre_empaque=nombre_empaque,
                    unidades_por_empaque=unidades_por_empaque,
                    descripcion=descripcion,
                    activo=True,
                    archivado=False,
                )
                producto.full_clean()
                producto.save()

                almacen = categoria.almacen_principal
                if almacen and almacen.activo and not almacen.eliminado:
                    existencia, _ = Existencia.objects.get_or_create(
                        producto=producto,
                        almacen=almacen,
                        defaults={"stock_disponible": stock_inicial, "stock_en_prestamo": 0},
                    )
                    if existencia.stock_disponible != stock_inicial:
                        existencia.stock_disponible = stock_inicial
                        existencia.save(update_fields=["stock_disponible"])

                    if stock_inicial > 0:
                        Movimiento.objects.create(
                            clave_idempotencia=f"init-stock-{producto.id}-{almacen.id}",
                            tipo=Movimiento.Tipo.MOVIMIENTO_INICIAL,
                            producto=producto,
                            cantidad=stock_inicial,
                            almacen_destino=almacen,
                            usuario=request.user,
                            motivo="Stock inicial al registrar producto vía Haku",
                            stock_disponible_anterior=0,
                            stock_disponible_posterior=stock_inicial,
                            via_haku=True,
                        )

                Auditoria.objects.create(
                    usuario=request.user,
                    accion="CREAR_PRODUCTO",
                    tipo_entidad="Producto",
                    id_entidad=str(producto.id),
                    valores_anteriores={},
                    valores_nuevos={
                        "codigo_interno": producto.codigo_interno,
                        "nombre": producto.nombre,
                        "categoria": categoria.nombre,
                        "stock_inicial": stock_inicial,
                        "stock_minimo": stock_minimo,
                        "es_prestable": producto.es_prestable,
                    },
                    motivo="Producto creado mediante formulario interactivo de Haku",
                    resultado="OK",
                    via_haku=True,
                )
        except ValidationError as exc_val:
            msg_val = exc_val.messages[0] if getattr(exc_val, "messages", None) else str(exc_val)
            return JsonResponse({"error": msg_val}, status=400)

        from .cache import HakuCacheManager
        from .contexto import HakuOperationalContext
        HakuCacheManager.invalidar_catalogo()
        session_key = getattr(getattr(request, "session", None), "session_key", None)
        if session_key and getattr(request.user, "id", None):
            estado_op = HakuOperationalContext.obtener_o_crear(
                request.user.id,
                str(session_key),
                getattr(request.user, "almacen_asignado_id", None),
            )
            if estado_op and estado_op.operacion_pendiente and estado_op.operacion_pendiente.get("tipo") == "CREAR_PRODUCTO":
                estado_op.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_op)

        msg_exito = f"Producto {producto.nombre} creado exitosamente con código interno {producto.codigo_interno}."
        return JsonResponse({
            "exito": True,
            "producto": {
                "id": producto.id,
                "codigo_interno": producto.codigo_interno,
                "nombre": producto.nombre,
                "categoria": categoria.nombre,
                "es_prestable": producto.es_prestable,
            },
            "mensaje_pantalla": msg_exito,
            "mensaje_verbal": normalizar_para_tts(f"Producto {producto.nombre} creado con código {producto.codigo_interno}."),
        })


class ProductoEditarHakuView(HakuApiBaseView):
    """Edita un producto existente validando campos permitidos y restricciones de unicidad.

    POST /api/haku/producto/editar/
    """

    def post(self, request):
        if not (request.user.es_administrador or request.user.tiene_permiso("catalogo.editar")):
            raise PermissionDenied("No cuentas con el permiso requerido (catalogo.editar) para modificar productos.")

        datos = self._obtener_json(request)
        producto_id = datos.get("producto_id")
        if not producto_id:
            return JsonResponse({"error": "El ID del producto es obligatorio."}, status=400)
        try:
            producto_id_int = int(producto_id)
        except (ValueError, TypeError):
            return JsonResponse({"error": "El ID del producto no es válido."}, status=400)

        from inventario.models import Categoria, Producto, Auditoria
        producto = Producto.objects.select_related(
            "categoria", "categoria__almacen_principal", "tipo"
        ).filter(id=producto_id_int, archivado=False).first()
        if not producto:
            return JsonResponse({"error": "El producto especificado no existe o fue archivado."}, status=404)

        # RBAC: verificar acceso del usuario al almacén del producto
        if producto.categoria and producto.categoria.almacen_principal:
            if not request.user.tiene_acceso_almacen(producto.categoria.almacen_principal):
                raise PermissionDenied("No tienes permisos para modificar productos de otros almacenes.")

        nombre = str(datos.get("nombre", producto.nombre)).strip()
        if not nombre:
            return JsonResponse({"error": "El nombre del producto no puede estar vacío."}, status=400)

        categoria_id = datos.get("categoria_id", producto.categoria_id)
        try:
            categoria_id_int = int(categoria_id)
        except (ValueError, TypeError):
            return JsonResponse({"error": "El identificador de la categoría no es válido."}, status=400)
        categoria = Categoria.objects.select_related("almacen_principal").filter(
            id=categoria_id_int, activo=True, eliminado=False
        ).first()
        if not categoria:
            return JsonResponse({"error": "La categoría seleccionada no existe o está inactiva."}, status=400)

        # DN-01 y RBAC: no cambiar categoría hacia otro almacén sin traslado formal
        if producto.categoria and categoria.almacen_principal_id != producto.categoria.almacen_principal_id:
            return JsonResponse({
                "error": (
                    f"No se puede asignar la categoría '{categoria.nombre}' porque pertenece a "
                    f"'{categoria.almacen_principal.nombre}', mientras el producto reside en "
                    f"'{producto.categoria.almacen_principal.nombre}'. Para moverlo de almacén debe realizar un traslado."
                )
            }, status=400)
        if not request.user.tiene_acceso_almacen(categoria.almacen_principal):
            raise PermissionDenied("No tienes permisos para asignar categorías de otros almacenes.")

        if "codigo_barras" in datos:
            codigo_barras_in = datos.get("codigo_barras")
            codigo_barras = str(codigo_barras_in).strip() if codigo_barras_in is not None else ""
            codigo_barras = codigo_barras or None
            if codigo_barras and Producto.objects.filter(codigo_barras=codigo_barras, archivado=False).exclude(id=producto.id).exists():
                return JsonResponse({"error": f"El código de barras '{codigo_barras}' ya pertenece a otro producto."}, status=400)
        else:
            codigo_barras = producto.codigo_barras

        stock_minimo_raw = datos.get("stock_minimo")
        if stock_minimo_raw is not None and stock_minimo_raw != "":
            try:
                stock_minimo = int(stock_minimo_raw)
                if stock_minimo < 0:
                    return JsonResponse({"error": "El stock mínimo no puede ser negativo."}, status=400)
            except (ValueError, TypeError):
                return JsonResponse({"error": "El stock mínimo debe ser un número entero válido (0 o mayor)."}, status=400)
        else:
            stock_minimo = producto.stock_minimo

        unidades_por_empaque = datos.get("unidades_por_empaque", producto.unidades_por_empaque)
        if unidades_por_empaque:
            try:
                unidades_por_empaque = int(unidades_por_empaque)
                if unidades_por_empaque <= 0:
                    return JsonResponse({"error": "Las unidades por empaque deben ser mayores a cero."}, status=400)
            except (ValueError, TypeError):
                return JsonResponse({"error": "Las unidades por empaque deben ser un número válido."}, status=400)
        else:
            unidades_por_empaque = None

        nombre_empaque = str(datos.get("nombre_empaque", producto.nombre_empaque)).strip()
        descripcion = str(datos.get("descripcion", producto.descripcion)).strip()
        if "es_prestable" in datos:
            es_prestable_raw = datos.get("es_prestable")
            if isinstance(es_prestable_raw, bool):
                es_prestable = es_prestable_raw
            else:
                es_prestable = str(es_prestable_raw).strip().lower() in ("true", "1", "si", "sí", "yes", "on")
        else:
            es_prestable = producto.es_prestable

        valores_anteriores = {
            "nombre": producto.nombre,
            "categoria_id": producto.categoria_id,
            "codigo_barras": producto.codigo_barras,
            "stock_minimo": producto.stock_minimo,
            "es_prestable": producto.es_prestable,
        }

        from django.db import transaction
        try:
            with transaction.atomic():
                if codigo_barras:
                    Producto.objects.filter(codigo_barras=codigo_barras, archivado=True).exclude(id=producto.id).update(codigo_barras=None)
                producto.nombre = nombre
                producto.categoria = categoria
                if producto.tipo_id and producto.tipo and producto.tipo.categoria_id != categoria.id:
                    producto.tipo = None
                producto.codigo_barras = codigo_barras
                producto.stock_minimo = stock_minimo
                producto.es_prestable = es_prestable
                producto.nombre_empaque = nombre_empaque
                producto.unidades_por_empaque = unidades_por_empaque
                producto.descripcion = descripcion
                producto.full_clean()
                producto.save()

                Auditoria.objects.create(
                    usuario=request.user,
                    accion="EDITAR_PRODUCTO",
                    tipo_entidad="Producto",
                    id_entidad=str(producto.id),
                    valores_anteriores=valores_anteriores,
                    valores_nuevos={
                        "nombre": producto.nombre,
                        "categoria_id": producto.categoria_id,
                        "codigo_barras": producto.codigo_barras,
                        "stock_minimo": producto.stock_minimo,
                        "es_prestable": producto.es_prestable,
                    },
                    motivo="Producto actualizado mediante formulario interactivo de Haku",
                    resultado="OK",
                    via_haku=True,
                )
        except ValidationError as exc_val:
            msg_val = exc_val.messages[0] if getattr(exc_val, "messages", None) else str(exc_val)
            return JsonResponse({"error": msg_val}, status=400)

        from .cache import HakuCacheManager
        from .contexto import HakuOperationalContext
        HakuCacheManager.invalidar_catalogo()
        session_key = getattr(getattr(request, "session", None), "session_key", None)
        if session_key and getattr(request.user, "id", None):
            estado_op = HakuOperationalContext.obtener_o_crear(
                request.user.id,
                str(session_key),
                getattr(request.user, "almacen_asignado_id", None),
            )
            if estado_op and estado_op.operacion_pendiente and estado_op.operacion_pendiente.get("tipo") in ("CREAR_PRODUCTO", "EDITAR_PRODUCTO"):
                estado_op.operacion_pendiente = None
                HakuOperationalContext.guardar(estado_op)
        HakuCacheManager.invalidar_stock(producto.id)
        HakuCacheManager.invalidar_ubicaciones(producto.id)

        msg_exito = f"El producto {producto.nombre} ({producto.codigo_interno}) ha sido actualizado con éxito."
        return JsonResponse({
            "exito": True,
            "producto": {
                "id": producto.id,
                "codigo_interno": producto.codigo_interno,
                "nombre": producto.nombre,
                "categoria": categoria.nombre,
                "stock_minimo": producto.stock_minimo,
                "es_prestable": producto.es_prestable,
            },
            "mensaje_pantalla": msg_exito,
            "mensaje_verbal": normalizar_para_tts(f"Producto {producto.nombre} actualizado correctamente."),
        })


class UsuarioCrearHakuView(HakuApiBaseView):
    """Crea un usuario desde el formulario integrado de HAKU.

    POST /api/haku/usuario/crear/
    """

    def post(self, request):
        usuario = request.user
        if not usuario.is_authenticated or not usuario.is_active or getattr(usuario, "eliminado_logicamente", False):
            return JsonResponse({"error": "No autenticado o cuenta inactiva."}, status=401)

        # Regla 1: Usuario normal no tiene acceso a la gestión de cuentas
        if usuario.es_usuario_normal:
            return JsonResponse({
                "error": "No tienes autorización para crear usuarios.",
                "mensaje_verbal": "No tienes autorización para crear usuarios en el sistema."
            }, status=403)

        payload = self._obtener_json(request)
        if not isinstance(payload, dict):
            return JsonResponse({"error": "Payload inválido."}, status=400)

        username = (payload.get("username") or "").strip().lower()
        nombre_completo = (payload.get("nombre_completo") or "").strip()
        password = payload.get("password") or ""
        rol = (payload.get("rol") or "").strip().upper()
        almacen_id = payload.get("almacen_asignado_id")
        almacenes_ids = payload.get("almacenes_asignados_ids") or []
        acceso_todos = bool(payload.get("acceso_todos_almacenes", False))

        errores = {}
        if not username:
            errores["username"] = "El nombre de usuario es obligatorio."
        elif UsuarioAudacIA.objects.filter(username__iexact=username).exists():
            errores["username"] = "Ya existe un usuario con este nombre de usuario."

        if not nombre_completo:
            errores["nombre_completo"] = "El nombre completo es obligatorio."

        if not rol:
            errores["rol"] = "El rol es obligatorio."
        elif rol not in [UsuarioAudacIA.Rol.SUPERUSUARIO, UsuarioAudacIA.Rol.ADMINISTRADOR, UsuarioAudacIA.Rol.USUARIO]:
            errores["rol"] = "El rol especificado no es válido."

        # Regla 2: Un administrador no puede crear un superusuario ni otro administrador
        if not usuario.es_superusuario_inventario and rol in (UsuarioAudacIA.Rol.SUPERUSUARIO, UsuarioAudacIA.Rol.ADMINISTRADOR):
            return JsonResponse({
                "error": "No tienes autorización para crear una cuenta con ese rol.",
                "mensaje_verbal": "No tienes autorización para crear una cuenta con ese rol."
            }, status=403)

        # Regla 3: Almacén asignado obligatorio para rol USUARIO (Opción A vs Opción B)
        almacen = None
        almacenes_multi_objs = []
        if rol == UsuarioAudacIA.Rol.USUARIO:
            tiene_especificos = bool(almacen_id or (almacenes_ids and len(almacenes_ids) > 0))
            if acceso_todos and tiene_especificos:
                errores["acceso_todos_almacenes"] = "No puedes seleccionar simultáneamente 'Todos los almacenes' y almacenes específicos."
            elif not acceso_todos and not tiene_especificos:
                errores["almacen_asignado_id"] = "Debes asignar al menos un almacén o marcar acceso a todos los almacenes."
            elif acceso_todos:
                almacen = None
                almacenes_multi_objs = []
            else:
                if almacen_id:
                    try:
                        almacen = Almacen.objects.get(id=int(almacen_id), activo=True, eliminado=False)
                    except (Almacen.DoesNotExist, ValueError, TypeError):
                        errores["almacen_asignado_id"] = "El almacén seleccionado no existe o no está activo."
                if almacenes_ids:
                    try:
                        alms_ids_int = [int(x) for x in almacenes_ids]
                        almacenes_multi_objs = list(Almacen.objects.filter(id__in=alms_ids_int, activo=True, eliminado=False))
                    except (ValueError, TypeError):
                        errores["almacenes_asignados_ids"] = "Lista de almacenes inválida."
                    if not almacen and almacenes_multi_objs:
                        almacen = almacenes_multi_objs[0]
        else:
            acceso_todos = False
            almacen = None
            almacenes_multi_objs = []

        # Regla 4: Validar contraseña según Django
        if not password:
            errores["password"] = "La contraseña es obligatoria."
        else:
            try:
                validate_password(password)
            except ValidationError as e:
                errores["password"] = " ".join(e.messages)

        if errores:
            return JsonResponse({"error": "Errores de validación.", "detalles": errores}, status=400)

        # Persistir con atomicidad y registro de auditoría
        with transaction.atomic():
            nuevo_usuario = UsuarioAudacIA(
                username=username,
                nombre_completo=nombre_completo,
                rol=rol,
                almacen_asignado=almacen,
                acceso_todos_almacenes=acceso_todos,
                creado_por=usuario,
                is_active=True,
                eliminado_logicamente=False,
            )
            nuevo_usuario.set_password(password)
            nuevo_usuario.save()

            if almacenes_multi_objs:
                nuevo_usuario.almacenes_asignados.set(almacenes_multi_objs)
            elif almacen:
                nuevo_usuario.almacenes_asignados.add(almacen)

            # Auditoría segura sin contraseñas
            Auditoria.objects.create(
                usuario=usuario,
                accion="CREAR_USUARIO",
                tipo_entidad="Usuario",
                id_entidad=str(nuevo_usuario.pk),
                valores_nuevos={
                    "username": nuevo_usuario.username,
                    "nombre_completo": nuevo_usuario.nombre_completo,
                    "rol": nuevo_usuario.rol,
                    "almacen": nuevo_usuario.almacen_asignado.nombre if nuevo_usuario.almacen_asignado else None,
                    "acceso_todos_almacenes": nuevo_usuario.acceso_todos_almacenes,
                    "is_active": nuevo_usuario.is_active,
                },
                motivo="Creación de usuario mediante HACU Inventario",
                resultado="OK",
                via_haku=True,
            )

        msg_exito = f"Usuario {nuevo_usuario.username} ({nuevo_usuario.nombre_completo}) creado exitosamente con rol {nuevo_usuario.rol}."
        return JsonResponse({
            "status": "ok",
            "usuario": {
                "id": nuevo_usuario.id,
                "username": nuevo_usuario.username,
                "nombre_completo": nuevo_usuario.nombre_completo,
                "rol": nuevo_usuario.rol,
                "almacen": nuevo_usuario.almacen_asignado.nombre if nuevo_usuario.almacen_asignado else None,
                "acceso_todos_almacenes": nuevo_usuario.acceso_todos_almacenes,
            },
            "mensaje_pantalla": msg_exito,
            "mensaje_verbal": normalizar_para_tts(f"Usuario {nuevo_usuario.username} creado correctamente."),
        })


class UsuarioEditarHakuView(HakuApiBaseView):
    """Edita un usuario desde el formulario integrado de HAKU.

    POST /api/haku/usuario/editar/
    """

    def post(self, request):
        usuario = request.user
        if not usuario.is_authenticated or not usuario.is_active or getattr(usuario, "eliminado_logicamente", False):
            return JsonResponse({"error": "No autenticado o cuenta inactiva."}, status=401)

        # Regla 1: Usuario normal no tiene acceso a la gestión de cuentas
        if usuario.es_usuario_normal:
            return JsonResponse({
                "error": "No tienes autorización para editar usuarios.",
                "mensaje_verbal": "No tienes autorización para gestionar cuentas en el sistema."
            }, status=403)

        payload = self._obtener_json(request)
        if not isinstance(payload, dict):
            return JsonResponse({"error": "Payload inválido."}, status=400)

        usuario_id = payload.get("id") or payload.get("usuario_id")
        if not usuario_id:
            return JsonResponse({"error": "Se requiere el ID del usuario a editar."}, status=400)
        try:
            usuario_id_int = int(usuario_id)
        except (ValueError, TypeError):
            return JsonResponse({"error": "El ID del usuario no es válido."}, status=400)

        try:
            target_user = UsuarioAudacIA.objects.get(id=usuario_id_int)
        except UsuarioAudacIA.DoesNotExist:
            return JsonResponse({"error": "El usuario especificado no existe."}, status=404)

        # Regla 2: Un administrador no puede editar, inactivar, eliminar ni modificar una cuenta SUPERUSUARIO ni otro ADMINISTRADOR
        if not usuario.es_superusuario_inventario:
            if target_user.es_superusuario_inventario:
                return JsonResponse({
                    "error": "No tienes autorización para modificar una cuenta de superusuario.",
                    "mensaje_verbal": "No tienes autorización para modificar cuentas con rol superusuario."
                }, status=403)
            if target_user.rol == UsuarioAudacIA.Rol.ADMINISTRADOR and target_user.id != usuario.id:
                return JsonResponse({
                    "error": "No tienes autorización para modificar cuentas de otros administradores.",
                    "mensaje_verbal": "No tienes autorización para modificar cuentas de otros administradores."
                }, status=403)

        nuevo_rol = (payload.get("rol") or target_user.rol).strip().upper()
        # Regla 3: Un administrador no puede convertir una cuenta en SUPERUSUARIO ni elevar privilegios
        if not usuario.es_superusuario_inventario and nuevo_rol == UsuarioAudacIA.Rol.SUPERUSUARIO:
            return JsonResponse({
                "error": "No tienes autorización para asignar el rol de superusuario.",
                "mensaje_verbal": "No puedes conceder privilegios de superusuario."
            }, status=403)

        def _parse_bool_json(val, default):
            if val is None:
                return default
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("true", "1", "si", "sí", "yes", "on")

        nombre_completo = (payload.get("nombre_completo") or target_user.nombre_completo).strip()
        almacen_id = payload.get("almacen_asignado_id")
        almacenes_ids = payload.get("almacenes_asignados_ids")
        acceso_todos = payload.get("acceso_todos_almacenes")
        if acceso_todos is None:
            acceso_todos = target_user.acceso_todos_almacenes
        else:
            acceso_todos = _parse_bool_json(acceso_todos, target_user.acceso_todos_almacenes)

        is_active = _parse_bool_json(payload.get("is_active"), target_user.is_active) if "is_active" in payload else target_user.is_active
        eliminado_logicamente = _parse_bool_json(payload.get("eliminado_logicamente"), target_user.eliminado_logicamente) if "eliminado_logicamente" in payload else target_user.eliminado_logicamente
        if eliminado_logicamente:
            is_active = False

        # Regla 4: Protección del último superusuario
        if target_user.es_superusuario_inventario:
            democion = nuevo_rol != UsuarioAudacIA.Rol.SUPERUSUARIO
            inactivacion = not is_active or eliminado_logicamente
            if democion or inactivacion:
                otros_superusuarios = UsuarioAudacIA.objects.filter(
                    rol=UsuarioAudacIA.Rol.SUPERUSUARIO,
                    is_active=True,
                    eliminado_logicamente=False
                ).exclude(id=target_user.id).count()

                if otros_superusuarios == 0:
                    return JsonResponse({
                        "error": "No se puede despojar de privilegios, inactivar o eliminar al único superusuario activo del sistema.",
                        "mensaje_verbal": "Operación rechazada. No se puede deshabilitar al único superusuario activo."
                    }, status=400)

        errores = {}
        if not nombre_completo:
            errores["nombre_completo"] = "El nombre completo no puede estar vacío."

        if nuevo_rol not in [UsuarioAudacIA.Rol.SUPERUSUARIO, UsuarioAudacIA.Rol.ADMINISTRADOR, UsuarioAudacIA.Rol.USUARIO]:
            errores["rol"] = "Rol inválido."

        almacen = target_user.almacen_asignado
        almacenes_multi_objs = None
        if almacenes_ids is not None:
            try:
                alms_ids_edit = [int(x) for x in almacenes_ids]
                almacenes_multi_objs = list(Almacen.objects.filter(id__in=alms_ids_edit, activo=True, eliminado=False))
            except (ValueError, TypeError):
                errores["almacenes_asignados_ids"] = "Lista de almacenes inválida."

        # Cambio de rol o edición de rol USUARIO
        if nuevo_rol == UsuarioAudacIA.Rol.USUARIO:
            estaba_como_admin = target_user.es_administrador
            if "acceso_todos_almacenes" in payload:
                acceso_todos_solicitado = bool(acceso_todos)
            else:
                acceso_todos_solicitado = target_user.acceso_todos_almacenes if not estaba_como_admin else False

            # Resolver almacenes específicos
            if "almacenes_asignados_ids" in payload:
                alms_list = almacenes_multi_objs or []
            else:
                alms_list = list(target_user.almacenes_asignados.filter(activo=True, eliminado=False)) if not estaba_como_admin else []

            if "almacen_asignado_id" in payload:
                if almacen_id:
                    try:
                        almacen = Almacen.objects.get(id=int(almacen_id), activo=True, eliminado=False)
                    except (Almacen.DoesNotExist, ValueError, TypeError):
                        errores["almacen_asignado_id"] = "El almacén seleccionado no existe o no está activo."
                else:
                    almacen = None
            elif estaba_como_admin:
                almacen = None

            if not almacen and alms_list:
                almacen = alms_list[0]

            tiene_especificos = bool(almacen or len(alms_list) > 0)

            # Si era administrador y ahora pasa a usuario normal, exigir decisión explícita
            if estaba_como_admin and not acceso_todos_solicitado and not tiene_especificos:
                errores["almacen_asignado_id"] = "Al cambiar de Administrador a Usuario normal debes definir explícitamente el alcance de almacenes."
            elif acceso_todos_solicitado and tiene_especificos:
                errores["acceso_todos_almacenes"] = "No puedes seleccionar simultáneamente 'Todos los almacenes' y almacenes específicos."
            elif not acceso_todos_solicitado and not tiene_especificos:
                errores["almacen_asignado_id"] = "Debes asignar al menos un almacén o marcar acceso a todos los almacenes."
            elif acceso_todos_solicitado:
                acceso_todos = True
                almacen = None
                almacenes_multi_objs = []
            else:
                acceso_todos = False
                almacenes_multi_objs = alms_list
        else:
            almacen = None
            almacenes_multi_objs = []
            acceso_todos = False

        # Contraseña opcional en edición
        nueva_password = payload.get("password")
        if nueva_password:
            try:
                validate_password(nueva_password, user=target_user)
            except ValidationError as e:
                errores["password"] = " ".join(e.messages)

        if errores:
            return JsonResponse({"error": "Errores de validación.", "detalles": errores}, status=400)

        valores_anteriores = {
            "nombre_completo": target_user.nombre_completo,
            "rol": target_user.rol,
            "almacen": target_user.almacen_asignado.nombre if target_user.almacen_asignado else None,
            "acceso_todos_almacenes": target_user.acceso_todos_almacenes,
            "is_active": target_user.is_active,
            "eliminado_logicamente": target_user.eliminado_logicamente,
        }

        with transaction.atomic():
            target_user = UsuarioAudacIA.objects.select_for_update().get(id=usuario_id)
            target_user.nombre_completo = nombre_completo
            target_user.rol = nuevo_rol
            target_user.almacen_asignado = almacen
            target_user.acceso_todos_almacenes = acceso_todos
            target_user.is_active = is_active
            target_user.eliminado_logicamente = eliminado_logicamente
            if nueva_password:
                target_user.set_password(nueva_password)
            target_user.save()

            if almacenes_multi_objs is not None:
                target_user.almacenes_asignados.set(almacenes_multi_objs)

            valores_nuevos = {
                "nombre_completo": target_user.nombre_completo,
                "rol": target_user.rol,
                "almacen": target_user.almacen_asignado.nombre if target_user.almacen_asignado else None,
                "acceso_todos_almacenes": target_user.acceso_todos_almacenes,
                "is_active": target_user.is_active,
                "eliminado_logicamente": target_user.eliminado_logicamente,
            }

            Auditoria.objects.create(
                usuario=usuario,
                accion="EDITAR_USUARIO",
                tipo_entidad="Usuario",
                id_entidad=str(target_user.pk),
                valores_anteriores=valores_anteriores,
                valores_nuevos=valores_nuevos,
                motivo="Modificación de usuario mediante HACU Inventario",
                resultado="OK",
                via_haku=True,
            )

        msg_exito = f"Usuario {target_user.username} actualizado correctamente."
        return JsonResponse({
            "status": "ok",
            "usuario": {
                "id": target_user.id,
                "username": target_user.username,
                "nombre_completo": target_user.nombre_completo,
                "rol": target_user.rol,
                "almacen": target_user.almacen_asignado.nombre if target_user.almacen_asignado else None,
                "acceso_todos_almacenes": target_user.acceso_todos_almacenes,
                "is_active": target_user.is_active,
            },
            "mensaje_pantalla": msg_exito,
            "mensaje_verbal": normalizar_para_tts(f"Usuario {target_user.username} actualizado correctamente."),
        })


class ConsultarCodigoBarrasView(HakuApiBaseView):
    """Valida un código de barras y consulta duplicados en BD o sugerencias en catálogo.

    GET /api/haku/producto/consultar-codigo-barras/?codigo=...
    """

    def get(self, request):
        if not (request.user.es_administrador or request.user.tiene_permiso("catalogo.ver")):
            raise PermissionDenied("No cuentas con el permiso requerido (catalogo.ver) para consultar códigos de barras.")

        codigo = request.GET.get("codigo", "")
        if not codigo:
            return JsonResponse({"error": "Parámetro 'codigo' requerido."}, status=400)

        from .servicios_catalogo import inspeccionar_codigo_barras
        resultado = inspeccionar_codigo_barras(codigo)
        return JsonResponse(resultado)


def _describir_alcance_almacenes(usuario) -> tuple[str, str]:
    """Construye (etiqueta_corta, detalle_tooltip) usando el RBAC real del usuario."""
    from inventario.models import Almacen

    alms_qs = (
        usuario.obtener_almacenes_accesibles()
        if hasattr(usuario, "obtener_almacenes_accesibles")
        else Almacen.objects.filter(activo=True, eliminado=False)
    )
    nombres = list(alms_qs.order_by("nombre").values_list("nombre", flat=True))
    tiene_todos = bool(
        getattr(usuario, "es_administrador", False)
        or getattr(usuario, "acceso_todos_almacenes", False)
    )
    if tiene_todos:
        detalle = (
            f"Acceso a todos los almacenes ({', '.join(nombres)})"
            if nombres
            else "Acceso a todos los almacenes"
        )
        return "Todos los almacenes", detalle
    if not nombres:
        return "Sin almacén asignado", "No tiene almacenes asignados actualmente"
    if len(nombres) <= 3:
        etiqueta = ", ".join(nombres)
        return etiqueta, f"Almacenes autorizados: {etiqueta}"
    return f"{len(nombres)} almacenes", f"Almacenes autorizados: {', '.join(nombres)}"


class SaludHakuView(View):
    """Endpoint liviano de disponibilidad y estado de precarga de modelos de Haku.

    GET /api/haku/salud/
    No expone datos de inventario ni de usuarios; permite al lanzador del servidor y al panel
    esperar hasta que el calentamiento de modelos de voz haya terminado por completo.
    """

    def get(self, request):
        from .apps import HakuConfig

        estado = HakuConfig.obtener_estado_warmup()
        stt_listo = bool(estado.get("stt") or estado.get("stt_modelo_listo"))
        warmup_iniciado = bool(estado.get("iniciado"))
        warmup_completado = bool(estado.get("completado")) or (not warmup_iniciado and stt_listo)
        return JsonResponse({
            "servidor_web": True,
            "warmup_iniciado": warmup_iniciado,
            "warmup_completado": warmup_completado,
            "indice_listo": bool(estado.get("indice_busqueda")),
            "vad_listo": bool(estado.get("vad")),
            "stt_listo": stt_listo,
            "tts_listo": bool(estado.get("tts")),
            "stt_dispositivo": estado.get("stt_dispositivo") or "",
            "listo_para_voz": bool(warmup_completado and stt_listo),
            "errores_count": len(estado.get("errores") or []),
        })


class EstadoHakuView(HakuApiBaseView):
    """Consulta el estado de la sesión conversacional de Haku.
    Estrictamente de solo lectura (no muta ni guarda la sesión salvo invalidación de contrato anterior).

    GET /api/haku/estado/
    """

    def get(self, request):
        session_key = getattr(request.session, "session_key", "") or ""
        from .contexto import HakuOperationalContext
        from .servicios_sesion import (
            obtener_estado_propuesta_sesion,
            es_propuesta_contrato_anterior,
            evaluar_vigencia_propuesta,
            invalidar_propuesta_pre_despliegue_en_transaccion,
        )

        estado_op = HakuOperationalContext.obtener(request.user.id, session_key)
        # Fuente canónica única: request.session (sin fallback a RAM que cause resurrección zombi)
        propuesta_activa = request.session.get("haku_propuesta_activa")
        if propuesta_activa and str(propuesta_activa.get("usuario_id") or "") != str(request.user.id):
            propuesta_activa = None

        if propuesta_activa and es_propuesta_contrato_anterior(propuesta_activa):
            invalidar_propuesta_pre_despliegue_en_transaccion(session_key, request, propuesta_activa.get("id"))
            propuesta_activa = None

        propuesta_caducada = False
        if propuesta_activa:
            es_vig_valida, esta_caducada = evaluar_vigencia_propuesta(propuesta_activa.get("expira_en"))
            if not es_vig_valida or esta_caducada:
                propuesta_caducada = bool(esta_caducada)
                propuesta_activa = None

        cid_req = request.GET.get("conversacion_id")
        if cid_req and propuesta_activa:
            cid_prop = propuesta_activa.get("conversacion_id")
            if cid_prop and str(cid_prop).strip() != str(cid_req).strip():
                propuesta_activa = None

        contexto_expirado = estado_op is None or estado_op.producto_id is None

        estado_nombre, _ = obtener_estado_propuesta_sesion(request.session, request.user)
        if not propuesta_activa and estado_nombre == "PENDIENTE":
            estado_nombre = "CADUCADA" if propuesta_caducada else "SIN_PROPUESTA"

        almacen_nombre, _almacen_detalle = _describir_alcance_almacenes(request.user)
        return JsonResponse({
            "usuario": request.user.username,
            "nombre_completo": request.user.nombre_completo or request.user.username,
            "rol": request.user.rol,
            "almacen": almacen_nombre,
            "tiene_propuesta_activa": propuesta_activa is not None,
            "estado_propuesta": estado_nombre,
            "propuesta_activa": propuesta_activa,
            "contexto_expirado": contexto_expirado,
        })


@method_decorator(ensure_csrf_cookie, name="dispatch")
class PanelHakuView(LoginRequiredMixin, TemplateView):
    """Vista web completa para la interfaz de voz y panel de Haku Inventario."""

    template_name = "haku/panel_voz.html"

    def get_context_data(self, **kwargs):
        from inventario.models import Almacen, Categoria
        from .services import usuario_tiene_permiso_operativo

        context = super().get_context_data(**kwargs)
        usuario = self.request.user
        context["usuario"] = usuario
        almacen_etiqueta, almacen_detalle = _describir_alcance_almacenes(usuario)
        context["almacen"] = almacen_etiqueta
        context["almacen_detalle"] = almacen_detalle
        puede_crear = bool(usuario_tiene_permiso_operativo(usuario, "catalogo.crear"))
        context["puede_crear_producto"] = puede_crear
        if puede_crear:
            alms_qs = (
                usuario.obtener_almacenes_accesibles()
                if hasattr(usuario, "obtener_almacenes_accesibles")
                else Almacen.objects.filter(activo=True, eliminado=False)
            )
            almacenes = list(alms_qs.order_by("nombre").values("id", "nombre"))
            alm_ids = [a["id"] for a in almacenes]
            categorias = [
                {
                    "id": c.id,
                    "nombre": c.nombre,
                    "prefijo_codigo": c.prefijo_codigo,
                    "almacen_principal_id": c.almacen_principal_id,
                }
                for c in Categoria.objects.filter(
                    activo=True,
                    eliminado=False,
                    almacen_principal_id__in=alm_ids,
                ).order_by("nombre")
            ]
            from inventario.models import Ubicacion
            ubicaciones = list(
                Ubicacion.objects.filter(
                    activo=True,
                    eliminado=False,
                    almacen_id__in=alm_ids,
                )
                .order_by("nombre")
                .values("id", "nombre", "almacen_id")
            )
            context["crear_producto_datos"] = {
                "categorias": categorias,
                "almacenes": almacenes,
                "ubicaciones": ubicaciones,
            }
        else:
            context["crear_producto_datos"] = {}
        return context
