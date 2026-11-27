"""Decodificación de audio en memoria sin crear archivos temporales en disco.

Soporta:
- WAV PCM (16-bit, 24-bit, 32-bit float, mono/stereo a cualquier sample rate).
- WebM / OGG / FLAC mediante soundfile o PyAV en memoria, sin depender de ffmpeg en PATH.
- Resampleo controlado a 16.000 Hz Mono Float32.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import wave
from typing import Tuple

import numpy as np

logger = logging.getLogger("integraciones.haku.audio")


class AudioDecodeError(Exception):
    """Lanzada cuando el buffer de audio recibido está corrupto o es incompatible."""
    pass


class AudioDecoder:
    """Decodificador de bytes de audio a arrays numpy float32 calibrados a 16kHz mono."""

    TARGET_SAMPLE_RATE = 16000

    @staticmethod
    def _es_contenedor_o_no_pcm(audio_bytes: bytes) -> bool:
        """Detecta cabeceras de contenedor multimedia, texto o longitudes inválidas para PCM16 crudo."""
        if not audio_bytes or len(audio_bytes) < 320 or (len(audio_bytes) % 2 != 0):
            return True
        head = audio_bytes[:16]
        if head.startswith((b"RIFF", b"OggS", b"\x1aE\xdf\xa3", b"ID3", b"fLaC", b"MThd", b"FORM", b"#!AMR")):
            return True
        if b"ftyp" in audio_bytes[:16] or b"moov" in audio_bytes[:32]:
            return True
        if head[:2] in (b"\xff\xf1", b"\xff\xf9", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return True
        stripped = audio_bytes[:32].lstrip()
        if stripped.startswith((b"<", b"{", b"[")):
            return True
        return False

    @classmethod
    def _decodificar_wav_bytes(cls, wav_bytes: bytes) -> Tuple[np.ndarray, float]:
        """Decodifica un buffer WAV estándar usando únicamente la librería estándar wave + numpy."""
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                sr = wf.getframerate()
                canales = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                num_frames = wf.getnframes()
                raw_data = wf.readframes(num_frames)

                if sampwidth == 2:  # 16-bit PCM
                    data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 1:  # 8-bit unsigned
                    data = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                elif sampwidth == 4:  # 32-bit int
                    data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise AudioDecodeError(f"Ancho de muestra no soportado: {sampwidth} bytes")

                if canales > 1:
                    data = data.reshape(-1, canales)

                return cls._normalizar_formato(data, sr)

    @classmethod
    def decodificar_a_16k_mono(cls, audio_bytes: bytes) -> Tuple[np.ndarray, float]:
        """Decodifica un buffer de bytes y retorna (samples_float32, duracion_segundos).

        Garantiza:
        - 1 canal (mono)
        - 16.000 Hz
        - Dtype np.float32 en rango normalizado [-1.0, 1.0]
        """
        if not audio_bytes:
            raise AudioDecodeError("Buffer de audio vacío.")

        err_sf_msg = None
        err_wave_msg = None
        err_av_msg = None
        err_ffmpeg_msg = None
        err_pcm_msg = None

        # 1. Intentar con soundfile sobre io.BytesIO (soporta WAV, WebM, OGG, FLAC si está instalado)
        try:
            import soundfile as sf
            with io.BytesIO(audio_bytes) as bio:
                data, sr = sf.read(bio, dtype="float32")
                return cls._normalizar_formato(data, sr)
        except Exception as exc:
            err_sf_msg = str(exc)
            logger.debug("soundfile en memoria no pudo decodificar (%s); intentando parser nativo WAV", exc)

        # 2. Fallback nativo: intentar con módulo wave estándar de Python
        try:
            return cls._decodificar_wav_bytes(audio_bytes)
        except Exception as exc:
            err_wave_msg = str(exc)
            logger.debug("Parser nativo wave falló: %s", exc)

        # 3. PyAV (incluido por faster-whisper) decodifica WebM/Opus, OGG y MP4/AAC
        # de navegadores de escritorio y móviles (Android/iOS) en memoria.
        try:
            import av

            with av.open(io.BytesIO(audio_bytes), mode="r") as container:
                flujo_audio = next(iter(container.streams.audio), None)
                if flujo_audio is None:
                    raise AudioDecodeError("El archivo no contiene una pista de audio.")

                resampler = av.AudioResampler(
                    format="flt",
                    layout="mono",
                    rate=cls.TARGET_SAMPLE_RATE,
                )
                bloques = []
                # Pasar el objeto AudioStream directamente (no audio=flujo_audio.index)
                # evita IndexError cuando la pista de audio no es el índice 0 (p. ej. MP4 móvil con pista de metadatos/video).
                for frame in container.decode(flujo_audio):
                    bloques.extend(
                        convertido.to_ndarray().reshape(-1)
                        for convertido in resampler.resample(frame)
                    )
                bloques.extend(
                    convertido.to_ndarray().reshape(-1)
                    for convertido in resampler.resample(None)
                )
                if not bloques:
                    raise AudioDecodeError("La pista de audio no contiene muestras decodificables.")
                data = np.concatenate(bloques).astype(np.float32, copy=False)
                return data, len(data) / cls.TARGET_SAMPLE_RATE
        except Exception as exc:
            err_av_msg = str(exc)
            logger.debug("Decodificador PyAV no pudo leer el audio (%s); intentando ffmpeg externo", exc)

        # 4. Fallback: ffmpeg externo (pipe en memoria; y archivo temporal efímero si el contenedor
        # requiere seek, p. ej. MP4/M4A/3GP con átomo moov al final).
        try:
            import subprocess
            cmd_pipe = [
                "ffmpeg", "-loglevel", "error", "-nostdin", "-threads", "1",
                "-i", "pipe:0", "-vn", "-f", "wav", "-acodec", "pcm_s16le",
                "-ar", str(cls.TARGET_SAMPLE_RATE), "-ac", "1", "pipe:1",
            ]
            proc = subprocess.Popen(cmd_pipe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_data, stderr_data = proc.communicate(input=audio_bytes, timeout=8)
            if proc.returncode == 0 and stdout_data:
                return cls._decodificar_wav_bytes(stdout_data)

            err_ffmpeg_msg = stderr_data.decode("utf-8", errors="replace") if stderr_data else f"code {proc.returncode}"

            # Reintento con archivo temporal efímero para contenedores MP4/M4A/3GP no seekeables por pipe
            if b"ftyp" in audio_bytes[:32] or "moov" in err_ffmpeg_msg.lower() or "seek" in err_ffmpeg_msg.lower():
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
                        tf.write(audio_bytes)
                        tmp_path = tf.name
                    cmd_file = [
                        "ffmpeg", "-loglevel", "error", "-nostdin", "-threads", "1",
                        "-i", tmp_path, "-vn", "-f", "wav", "-acodec", "pcm_s16le",
                        "-ar", str(cls.TARGET_SAMPLE_RATE), "-ac", "1", "pipe:1",
                    ]
                    proc2 = subprocess.Popen(cmd_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout2, stderr2 = proc2.communicate(timeout=8)
                    if proc2.returncode == 0 and stdout2:
                        return cls._decodificar_wav_bytes(stdout2)
                    err_ffmpeg_msg = stderr2.decode("utf-8", errors="replace") if stderr2 else f"file code {proc2.returncode}"
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
        except Exception as exc:
            err_ffmpeg_msg = str(exc)
            logger.debug("Decodificador ffmpeg falló o no disponible: %s", exc)

        # 5. Fallback PCM 16-bit crudo a 16kHz SOLO para buffers sin firma de contenedor.
        # Nunca reinterpretar contenedores dañados/truncados (WebM/MP4/OGG/WAV) como ruido blanco PCM.
        if cls._es_contenedor_o_no_pcm(audio_bytes):
            raise AudioDecodeError(
                f"No se pudo decodificar el formato de audio: sf={err_sf_msg}, wave={err_wave_msg}, av={err_av_msg}, ffmpeg={err_ffmpeg_msg}"
            )

        try:
            data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return cls._normalizar_formato(data, cls.TARGET_SAMPLE_RATE)
        except Exception as exc:
            err_pcm_msg = str(exc)
            raise AudioDecodeError(
                f"No se pudo decodificar el formato de audio: sf={err_sf_msg}, wave={err_wave_msg}, av={err_av_msg}, ffmpeg={err_ffmpeg_msg}, pcm={err_pcm_msg}"
            ) from exc

    @classmethod
    def _normalizar_formato(cls, data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, float]:
        """Convierte canales a mono y resamplea a 16.000 Hz si difiere."""
        # A mono si es estéreo o multicanal
        if data.ndim > 1:
            data = data.mean(axis=1)

        data = data.astype(np.float32)

        # Resampleo controlado si no está a 16 kHz
        if sample_rate != cls.TARGET_SAMPLE_RATE and len(data) > 0:
            data = cls._resamplear_lineal(data, sample_rate, cls.TARGET_SAMPLE_RATE)
            sample_rate = cls.TARGET_SAMPLE_RATE

        duracion = len(data) / sample_rate if sample_rate > 0 else 0.0
        return data, duracion

    @staticmethod
    def _resamplear_lineal(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resampleador lineal vectorial de alto rendimiento para arrays float32."""
        if src_rate == dst_rate:
            return data
        factor = dst_rate / src_rate
        num_nuevas_muestras = int(round(len(data) * factor))
        indices_orig = np.linspace(0, len(data) - 1, num_nuevas_muestras)
        resampled = np.interp(indices_orig, np.arange(len(data)), data)
        return resampled.astype(np.float32)
