"""Motor de síntesis de voz (Text-to-Speech) propio y desacoplado para Haku Inventario.

Características:
- Independiente: NO depende de hacu/ (Haku Visitante) ni de sus configuraciones.
- Voz neuronal local: Usa Piper TTS (ONNX) con voz en español neutro (es_MX-claude-high).
- Sin dependencias externas de red o APIs cloud.
- Carga diferida (lazy loading) del modelo en memoria RAM.
- Generación en memoria de WAV 16-bit PCM Mono a 16.000 Hz calibrado para el navegador.
- Cero audio en altavoces del servidor físico (reproducción exclusiva en el cliente).
- Manejo estricto de errores mediante HakuSynthesizerError (sin fallbacks sinusoidales silenciosos).
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from django.conf import settings as django_settings

from ..telemetria import registrar_timing

logger = logging.getLogger("integraciones.haku.audio.synthesizer")

try:
    import soundfile as _soundfile_mod  # type: ignore
except ImportError:
    _soundfile_mod = None


class HakuSynthesizerError(Exception):
    """Lanzada cuando el sintetizador de voz falla o el modelo no está disponible."""
    pass


class HakuSynthesizer:
    """Sintetizador neuronal Text-to-Speech determinista y desacoplado para Haku Inventario."""

    _instancia: Optional["HakuSynthesizer"] = None
    _instancia_lock = threading.Lock()

    # Directorio de modelos y caché propio de Haku Inventario
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    CACHE_DIR = BASE_DIR / "cache"

    NOMBRE_VOZ_POR_DEFECTO = "es_MX-claude-high"
    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        modelo_path: Optional[Path | str] = None,
        config_path: Optional[Path | str] = None,
    ) -> None:
        if modelo_path is not None:
            self.modelo_path = Path(modelo_path)
        else:
            configured_model_path = getattr(django_settings, "HAKU_TTS_MODEL_PATH", None)
            self.modelo_path = (
                Path(configured_model_path)
                if configured_model_path
                else self.MODELS_DIR / f"{self.NOMBRE_VOZ_POR_DEFECTO}.onnx"
            )

        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            configured_config_path = getattr(django_settings, "HAKU_TTS_CONFIG_PATH", None)
            self.config_path = (
                Path(configured_config_path)
                if configured_config_path
                else self.MODELS_DIR / f"{self.NOMBRE_VOZ_POR_DEFECTO}.onnx.json"
            )

        self._voice = None
        self._lock = threading.Lock()
        self._mem_cache: dict[str, bytes] = {}
        self._max_cache_items = 64
        self.tiempo_carga_modelo_ms: Optional[float] = None

        # Garantizar que existan las carpetas models y cache
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def obtener_instancia(cls) -> "HakuSynthesizer":
        """Retorna la instancia Singleton compartida de HakuSynthesizer."""
        if cls._instancia is None:
            with cls._instancia_lock:
                if cls._instancia is None:
                    cls._instancia = cls()
        return cls._instancia

    @classmethod
    def reiniciar(cls) -> None:
        """Reinicia la instancia (útil para pruebas unitarias)."""
        with cls._instancia_lock:
            cls._instancia = None

    @property
    def modelo_cargado(self) -> bool:
        """Indica si el modelo ONNX está cargado en memoria."""
        return self._voice is not None

    def cargar_modelo(self, interaction_id: Optional[str] = None) -> None:
        """Carga el modelo ONNX de Piper en memoria de forma diferida y thread-safe."""
        if self._voice is not None:
            return

        with self._lock:
            if self._voice is not None:
                return

            if not self.modelo_path.exists():
                logger.error("Archivo de modelo Piper no encontrado en: %s", self.modelo_path)
                raise HakuSynthesizerError(
                    f"Modelo TTS no encontrado en '{self.modelo_path}'. "
                    f"Asegúrate de que {self.NOMBRE_VOZ_POR_DEFECTO}.onnx esté instalado."
                )

            t0_load = time.perf_counter()
            try:
                from piper import PiperVoice  # type: ignore

                cfg = str(self.config_path) if self.config_path.exists() else None
                logger.info("Cargando modelo Piper en CPU: %s", self.modelo_path.name)
                self._voice = PiperVoice.load(str(self.modelo_path), config_path=cfg)
                self.tiempo_carga_modelo_ms = (time.perf_counter() - t0_load) * 1000.0
                registrar_timing(interaction_id, "tts_model_load", self.tiempo_carga_modelo_ms)
                logger.info(
                    "Modelo Piper cargado con éxito en %.2f ms. Sample rate nativo: %s Hz",
                    self.tiempo_carga_modelo_ms,
                    getattr(self._voice.config, "sample_rate", "desconocido"),
                )
            except Exception as exc:
                logger.exception("Error al inicializar PiperVoice: %s", exc)
                raise HakuSynthesizerError(f"Error al inicializar motor TTS Piper: {exc}") from exc

    def sintetizar(self, texto: str, interaction_id: Optional[str] = None) -> bytes:
        """Sintetiza un texto en voz neutra profesional y retorna bytes WAV (16kHz PCM Mono).

        Parámetros:
            texto: Texto formal normalizado a sintetizar.
            interaction_id: Identificador de interacción para telemetría estructurada.

        Retorna:
            bytes: Buffer WAV válido con encabezado RIFF, 16000 Hz, 1 canal, 16 bits PCM.

        Lanza:
            HakuSynthesizerError: Si el texto está vacío, el modelo falla o hay un error de E/S.
        """
        if not texto or not texto.strip():
            raise HakuSynthesizerError("Texto de entrada vacío para síntesis de voz.")

        texto_limpio = re.sub(r"\b(?:hacu|haku)\b", "Jacu", texto.strip(), flags=re.IGNORECASE)

        # Verificar caché en memoria de forma thread-safe para respuestas repetitivas de almacén
        with self._lock:
            cached_wav = self._mem_cache.get(texto_limpio)
        if cached_wav is not None:
            registrar_timing(interaction_id, "tts_cache_hit", 0.0)
            return cached_wav

        # Asegurar carga del modelo (fuera del lock de síntesis para evitar bloqueo reentrante)
        self.cargar_modelo(interaction_id=interaction_id)

        with self._lock:
            cached_wav = self._mem_cache.get(texto_limpio)
            if cached_wav is not None:
                registrar_timing(interaction_id, "tts_cache_hit", 0.0)
                return cached_wav
            try:
                # Síntesis cruda en memoria usando PiperVoice
                t0_synth = time.perf_counter()
                raw_buffer = io.BytesIO()
                with wave.open(raw_buffer, "wb") as wav_out:
                    self._voice.synthesize_wav(texto_limpio, wav_out)

                raw_bytes = raw_buffer.getvalue()
                if not raw_bytes or len(raw_bytes) < 44:
                    raise HakuSynthesizerError("Piper generó un búfer de audio vacío o truncado.")
                dt_synth = (time.perf_counter() - t0_synth) * 1000.0
                registrar_timing(interaction_id, "tts_synthesis", dt_synth)

                # Resamplear y calibrar a 16.000 Hz Mono 16-bit PCM estándar
                t0_resample = time.perf_counter()
                wav_16k_bytes = self._calibrar_y_resamplear(raw_bytes)
                dt_resample = (time.perf_counter() - t0_resample) * 1000.0
                registrar_timing(interaction_id, "tts_resample", dt_resample)

                dt_total_compute = dt_synth + dt_resample
                registrar_timing(interaction_id, "tts_total_compute", dt_total_compute, bytes=len(wav_16k_bytes))

                # Almacenar en caché LRU básica
                if len(self._mem_cache) >= self._max_cache_items:
                    # Descartar una entrada antigua
                    self._mem_cache.pop(next(iter(self._mem_cache)))
                self._mem_cache[texto_limpio] = wav_16k_bytes

                return wav_16k_bytes

            except HakuSynthesizerError:
                raise
            except Exception as exc:
                logger.exception("Fallo durante la síntesis TTS: %s", exc)
                raise HakuSynthesizerError(f"Error al sintetizar audio TTS: {exc}") from exc

    def _calibrar_y_resamplear(self, raw_wav_bytes: bytes) -> bytes:
        """Convierte cualquier WAV generado por Piper a 16 kHz Mono 16-bit PCM."""
        if _soundfile_mod is not None:
            try:
                with io.BytesIO(raw_wav_bytes) as bio:
                    data, sr = _soundfile_mod.read(bio, dtype="float32")

                # Aplanar a mono si tuviera más de 1 canal
                if data.ndim > 1:
                    data = data.mean(axis=1)

                # Resampleo a 16000 Hz si difiere
                if sr != self.TARGET_SAMPLE_RATE and len(data) > 0:
                    factor = self.TARGET_SAMPLE_RATE / sr
                    nuevas_muestras = int(round(len(data) * factor))
                    indices_orig = np.linspace(0, len(data) - 1, nuevas_muestras)
                    data = np.interp(indices_orig, np.arange(len(data)), data).astype(np.float32)

                # Escribir WAV 16-bit PCM limpio
                out_bio = io.BytesIO()
                _soundfile_mod.write(out_bio, data, self.TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
                return out_bio.getvalue()
            except Exception as exc_sf:
                logger.debug("soundfile falló en calibración (%s); usando parser wave nativo", exc_sf)

        # Ruta nativa sin dependencias externas usando módulo wave estándar
        with io.BytesIO(raw_wav_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                sr = wf.getframerate()
                canales = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw_data = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32) / 128.0

        if canales > 1:
            samples = samples.reshape(-1, canales).mean(axis=1)

        if sr != self.TARGET_SAMPLE_RATE and len(samples) > 0:
            factor = self.TARGET_SAMPLE_RATE / sr
            nuevas_muestras = int(round(len(samples) * factor))
            indices_orig = np.linspace(0, len(samples) - 1, nuevas_muestras)
            samples = np.interp(indices_orig, np.arange(len(samples)), samples).astype(np.float32)

        # Clamp y conversión a int16
        samples = np.clip(samples, -1.0, 1.0)
        int16_samples = (samples * 32767.0).astype(np.int16)

        out_bio = io.BytesIO()
        with wave.open(out_bio, "wb") as wf_out:
            wf_out.setnchannels(1)
            wf_out.setsampwidth(2)
            wf_out.setframerate(self.TARGET_SAMPLE_RATE)
            wf_out.writeframes(int16_samples.tobytes())

        return out_bio.getvalue()
