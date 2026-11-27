"""Configuración de la app integraciones.haku."""

import logging
import os
import threading
import time

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger("integraciones.haku.audio")


class HakuConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "integraciones.haku"
    verbose_name = "HACU Inventario y Voz"

    _estado_warmup_lock = threading.Lock()
    _estado_warmup = {
        "iniciado": False,
        "completado": False,
        "indice_busqueda": False,
        "vad": False,
        "stt": False,
        "tts": False,
        "llm": False,
        "haku": False,
        "errores": [],
    }

    @classmethod
    def obtener_estado_warmup(cls) -> dict:
        """Retorna una instantánea segura del progreso de precarga de modelos."""
        with cls._estado_warmup_lock:
            snapshot = dict(cls._estado_warmup)
            snapshot["errores"] = list(cls._estado_warmup.get("errores") or [])
        stt_modelo_listo = False
        stt_dispositivo = ""
        try:
            from .audio.transcriber import HakuTranscriber

            inst = HakuTranscriber._instancia
            if inst is not None:
                stt_modelo_listo = bool(inst.modelo_listo())
                stt_dispositivo = str(inst.dispositivo_efectivo or "")
        except Exception:
            pass
        snapshot["stt_modelo_listo"] = stt_modelo_listo
        snapshot["stt_dispositivo"] = stt_dispositivo
        return snapshot

    @classmethod
    def _actualizar_estado_warmup(cls, clave: str, valor) -> None:
        with cls._estado_warmup_lock:
            cls._estado_warmup[clave] = valor

    @classmethod
    def _registrar_error_warmup(cls, detalle: str) -> None:
        with cls._estado_warmup_lock:
            cls._estado_warmup["errores"].append(detalle)

    def ready(self):
        """Inicializaciones al arrancar Django."""
        try:
            from .cache import conectar_senales_cache
            from .contexto import conectar_senales_contexto
            conectar_senales_cache()
            conectar_senales_contexto()
        except Exception:
            pass

        # Precalentamiento de búsqueda, STT y TTS en hilo de fondo para eliminar
        # latencia de índice/modelos en la primera petición del usuario.
        # Solo se ejecuta en el proceso servidor principal (no en pruebas, migraciones ni autoreloader).
        if self._es_proceso_servidor():
            if getattr(settings, "HAKU_HUMAN_QA_SAVE_AUDIO", False):
                try:
                    from .audio.captura_qa import limpiar_capturas_expiradas

                    limpiar_capturas_expiradas()
                except Exception:
                    logger.warning("No se pudieron limpiar audios QA vencidos al iniciar", exc_info=True)
            self._iniciar_warmup_modelos()

    @staticmethod
    def _es_proceso_servidor() -> bool:
        """Determina si el proceso actual es un servidor web real y no una prueba o comando de gestión."""
        import sys
        comandos_excluidos = {"test", "migrate", "makemigrations", "collectstatic", "check", "shell", "inspectdb"}
        if any(cmd in sys.argv for cmd in comandos_excluidos):
            return False
        return os.environ.get("RUN_MAIN") == "true" or not os.environ.get("DJANGO_DEV")

    @classmethod
    def _iniciar_warmup_modelos(cls):
        """Lanza la carga de los modelos Faster-Whisper y Piper TTS en un hilo daemon de fondo.

        Al cargar los modelos proactivamente al arrancar el servidor, la primera petición
        real de voz del usuario ya encuentra ambos motores en memoria listos para operar,
        eliminando ~45s de cold-start en STT y ~5s en síntesis TTS.
        """

        with cls._estado_warmup_lock:
            cls._estado_warmup = {
                "iniciado": True,
                "completado": False,
                "indice_busqueda": False,
                "vad": False,
                "stt": False,
                "tts": False,
                "llm": False,
                "haku": False,
                "errores": [],
            }

        def _warmup():
            try:
                # 0. Warmup del índice del catálogo para que la primera búsqueda por
                # nombre use el índice en memoria, no pague su construcción.
                if getattr(settings, "HAKU_WARMUP_SEARCH_INDEX_ON_STARTUP", True):
                    try:
                        from .buscador import HakuSearchIndex

                        t0_indice = time.perf_counter()
                        indice = HakuSearchIndex.obtener()
                        cls._actualizar_estado_warmup("indice_busqueda", True)
                        # Calienta también la ruta exacta usada por las consultas
                        # literales; la entidad se obtiene del catálogo activo.
                        from cuentas.models import UsuarioAudacIA

                        usuario = UsuarioAudacIA.objects.filter(is_active=True).order_by("id").first()
                        producto = next(iter(indice._productos.values()), None)
                        if usuario is not None and producto is not None:
                            indice.mencionar_productos_literales(
                                producto.get("nombre", ""),
                                usuario=usuario,
                            )
                        logger.info(
                            "[HakuSearch-Warmup] Índice listo (%d productos, %.0f ms).",
                            len(indice._productos),
                            (time.perf_counter() - t0_indice) * 1000.0,
                        )
                    except Exception as exc_indice:
                        logger.warning(
                            "[HakuSearch-Warmup] No se pudo precargar el índice: %s. "
                            "La primera búsqueda lo construirá de forma perezosa.",
                            exc_indice,
                        )

                # 1. Warmup STT (Faster-Whisper)
                try:
                    import numpy as np
                    from .audio.transcriber import HakuTranscriber
                    from .services import adquirir_lock_haku

                    logger.info(
                        "[HakuSTT-Warmup] Iniciando precarga del modelo STT en segundo plano..."
                    )
                    transcriptor = HakuTranscriber.obtener_instancia()
                    transcriptor._asegurar_modelo(interaction_id="warmup_startup")
                    dispositivo = transcriptor.dispositivo_efectivo or "desconocido"
                    carga_ms = transcriptor.tiempo_carga_modelo_ms or 0.0
                    logger.info(
                        "[HakuSTT-Warmup] Modelo cargado (device=%s, carga=%.0f ms). "
                        "Iniciando inferencia de calentamiento CuDNN...",
                        dispositivo,
                        carga_ms,
                    )

                    sr = 16000
                    t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
                    warmup_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
                    from .audio.pipeline import AudioPipeline

                    t0_vad = time.perf_counter()
                    AudioPipeline.desde_settings().vad.procesar(warmup_audio)
                    cls._actualizar_estado_warmup("vad", True)
                    logger.info(
                        "[HakuVAD-Warmup] Detector Silero listo (inferencia de calentamiento=%.0f ms).",
                        (time.perf_counter() - t0_vad) * 1000.0,
                    )
                    t0_warm = time.perf_counter()
                    with adquirir_lock_haku(timeout=30.0):
                        transcriptor.transcribir(
                            warmup_audio, idioma="es", beam_size=1, interaction_id="warmup_cudnn"
                        )
                    dt_warm = (time.perf_counter() - t0_warm) * 1000.0
                    cls._actualizar_estado_warmup("stt", True)
                    logger.info(
                        "[HakuSTT-Warmup] Calentamiento CuDNN completado en %.0f ms. "
                        "Sistema STT listo para peticiones inmediatas.",
                        dt_warm,
                    )
                except Exception as exc:
                    cls._registrar_error_warmup(f"STT: {type(exc).__name__}: {exc}")
                    logger.warning(
                        "[HakuSTT-Warmup] No se pudo precargar el modelo STT: %s. "
                        "Se cargará de forma perezosa en la primera petición.",
                        exc,
                    )

                # 2. Warmup TTS (Piper ONNX)
                try:
                    from .audio.synthesizer import HakuSynthesizer
                    synth = HakuSynthesizer.obtener_instancia()
                    synth.cargar_modelo(interaction_id="warmup_tts")
                    t0_synth = time.perf_counter()
                    audio_prueba = synth.sintetizar(
                        "Hacu Inventario está listo. ¿En qué puedo ayudarle con el inventario?",
                        interaction_id="warmup_tts_inference",
                    )
                    if not audio_prueba or len(audio_prueba) <= 44:
                        raise RuntimeError("El calentamiento Piper no produjo audio WAV válido")
                    cls._actualizar_estado_warmup("tts", True)
                    logger.info(
                        "[HakuTTS-Warmup] Voz Piper cargada y calentada (carga=%.0f ms, primera síntesis=%.0f ms).",
                        synth.tiempo_carga_modelo_ms or 0.0,
                        (time.perf_counter() - t0_synth) * 1000.0,
                    )
                except Exception as exc_tts:
                    cls._registrar_error_warmup(f"TTS: {type(exc_tts).__name__}: {exc_tts}")
                    logger.warning(
                        "[HakuTTS-Warmup] No se pudo precargar el modelo TTS: %s. "
                        "Se cargará de forma perezosa en la primera petición.",
                        exc_tts,
                    )

                # Calentar el Cerebro Central remoto con el mismo plan JSON que usa
                # Haku antes de que el servidor acepte usuarios. Un token de texto
                # no ejercita el prompt, la gramática JSON ni la generación real.
                if getattr(settings, "HAKU_WARMUP_LLM_ON_STARTUP", False):
                    try:
                        from .lenguaje.interprete_semantico import _obtener_cliente_cerebro
                        from .agente.planificador import planificar as planificar_plan_json

                        cliente_llm = _obtener_cliente_cerebro()
                        if cliente_llm is None:
                            raise RuntimeError("No se pudo crear el cliente oficial del Cerebro Central")
                        salud_llm = cliente_llm.comprobar_salud(forzar_sondeo=True)
                        if not salud_llm.get("disponible"):
                            raise RuntimeError("El endpoint del Cerebro Central no está disponible")
                        t0_llm = time.perf_counter()
                        plan_calentamiento = planificar_plan_json(
                            texto="¿Qué productos están agotados?",
                            cliente_llm=cliente_llm,
                            contexto_txt=(
                                "Categorías del catálogo mencionadas: ninguna\n"
                                "Almacenes accesibles mencionados: ninguno\n"
                                "Ubicaciones accesibles mencionadas: ninguna"
                            ),
                            usar_cache=False,
                        )
                        latencia_llm_ms = (time.perf_counter() - t0_llm) * 1000.0
                        if plan_calentamiento is None or plan_calentamiento.tipo != "consulta":
                            raise RuntimeError("El Cerebro Central no produjo un plan JSON válido durante el calentamiento")
                        cls._actualizar_estado_warmup("llm", True)
                        logger.info(
                            "[HakuLLM-Warmup] Plan JSON del Cerebro Central listo (%.0f ms, %d tokens).",
                            latencia_llm_ms,
                            plan_calentamiento.tokens_completados,
                        )

                        if getattr(settings, "HAKU_WARMUP_HAKU_QUERY_ON_STARTUP", False):
                            from cuentas.models import UsuarioAudacIA
                            from inventario.models import Producto
                            from .agente import AgenteInventarioHaku

                            usuario_qa = UsuarioAudacIA.objects.filter(is_active=True).order_by("id").first()
                            producto_qa = (
                                Producto.objects.filter(activo=True, archivado=False)
                                .order_by("id")
                                .first()
                            )
                            if usuario_qa is None or producto_qa is None:
                                raise RuntimeError("Faltan una cuenta activa o productos para calentar el flujo Haku")
                            t0_haku = time.perf_counter()
                            salida_haku = AgenteInventarioHaku.procesar(
                                texto=f"¿Cuánto stock disponible hay del {producto_qa.nombre}?",
                                usuario=usuario_qa,
                                estado_operativo=None,
                                sesion=None,
                                interaction_id="warmup_haku_query",
                                cliente_llm=cliente_llm,
                            )
                            if salida_haku.error_terminal or salida_haku.plan is None or salida_haku.respuesta is None:
                                raise RuntimeError("Haku no completó la consulta sintética de calentamiento")
                            cls._actualizar_estado_warmup("haku", True)
                            logger.info(
                                "[Haku-Warmup] Flujo completo de consulta listo (primera consulta de descarte=%.0f ms).",
                                (time.perf_counter() - t0_haku) * 1000.0,
                            )
                    except Exception as exc_llm:
                        cls._registrar_error_warmup(f"LLM: {type(exc_llm).__name__}: {exc_llm}")
                        logger.warning("[HakuLLM-Warmup] No se pudo precalentar el Cerebro Central: %s", exc_llm)
            finally:
                cls._actualizar_estado_warmup("completado", True)

        hilo = threading.Thread(target=_warmup, name="haku-warmup-modelos", daemon=True)
        hilo.start()
        if getattr(settings, "HAKU_BLOCK_SERVER_UNTIL_MODELS_READY", False):
            timeout = float(getattr(settings, "HAKU_MODEL_WARMUP_TIMEOUT_SECONDS", 240))
            hilo.join(timeout=max(1.0, timeout))
            if hilo.is_alive():
                raise RuntimeError(
                    f"Haku no inicia: el calentamiento de modelos excedió {timeout:.0f} segundos"
                )
            estado_final = cls.obtener_estado_warmup()
            componentes = ["vad", "stt", "tts"]
            if getattr(settings, "HAKU_WARMUP_LLM_ON_STARTUP", False):
                componentes.append("llm")
            if getattr(settings, "HAKU_WARMUP_HAKU_QUERY_ON_STARTUP", False):
                componentes.append("haku")
            no_listos = [nombre for nombre in componentes if not estado_final[nombre]]
            if no_listos:
                detalle = "; ".join(estado_final["errores"])
                raise RuntimeError(
                    "Haku no inicia porque los modelos no quedaron listos "
                    f"({', '.join(no_listos)}). Detalle: {detalle}"
                )

