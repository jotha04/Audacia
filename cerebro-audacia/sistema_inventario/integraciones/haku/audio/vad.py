"""Voice Activity Detection (VAD) para detección de habla y descarte de silencios/ruido."""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("integraciones.haku.audio")


@dataclass(frozen=True)
class VADResult:
    """Resultado del análisis de actividad vocal."""
    tiene_voz: bool
    motivo: str  # "VOZ_DETECTADA" | "AUDIO_SILENCIOSO" | "SOLO_RUIDO" | "AUDIO_VACIO"
    muestras_activas: np.ndarray
    duracion_original_seg: float
    duracion_voz_seg: float
    ratio_voz: float
    mascara_activa: Optional[np.ndarray] = None


class VoiceActivityDetector:
    """Detector de actividad de voz por energía adaptativa y tasa de cruce por cero (ZCR).

    Diseñado para operar en tiempo real (<2ms en CPU) sin modelos pesados ni GPU.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        umbral_silencio_dbfs: float = -46.0,
        min_segundos_voz: float = 0.25,
        margen_silencio_ms: int = 350,
        umbral_probabilidad_voz: float = 0.5,
        silencio_minimo_ms: int = 250,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_len = int((frame_duration_ms / 1000.0) * sample_rate)
        self.umbral_silencio_dbfs = umbral_silencio_dbfs
        self.min_segundos_voz = min_segundos_voz
        self.margen_frames = int((margen_silencio_ms / 1000.0) * (sample_rate / self.frame_len))
        self.margen_silencio_ms = max(0, int(margen_silencio_ms))
        self.umbral_probabilidad_voz = min(0.95, max(0.05, float(umbral_probabilidad_voz)))
        self.silencio_minimo_ms = max(100, int(silencio_minimo_ms))

    def procesar(self, samples: np.ndarray) -> VADResult:
        """Detecta habla con Silero y conserva un fallback ligero si no está instalado."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        total_len = len(samples)
        if total_len == 0:
            return VADResult(
                tiene_voz=False,
                motivo="AUDIO_VACIO",
                muestras_activas=np.zeros(0, dtype=np.float32),
                duracion_original_seg=0.0,
                duracion_voz_seg=0.0,
                ratio_voz=0.0,
            )

        duracion_original = total_len / self.sample_rate

        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            spans = get_speech_timestamps(
                samples,
                VadOptions(
                    threshold=self.umbral_probabilidad_voz,
                    min_speech_duration_ms=max(100, min(150, int(self.min_segundos_voz * 1000))),
                    min_silence_duration_ms=self.silencio_minimo_ms,
                    speech_pad_ms=self.margen_silencio_ms,
                ),
                sampling_rate=self.sample_rate,
            )
        except ImportError:
            logger.warning("Silero VAD no está instalado; usando el detector de energía compatible.")
            return self._procesar_por_energia(samples)
        except Exception as exc:
            logger.warning("Silero VAD falló; usando el detector de energía compatible: %s", exc)
            return self._procesar_por_energia(samples)

        mascara_activa = np.zeros(total_len, dtype=bool)
        limites = []
        for span in spans:
            inicio = max(0, min(total_len, int(span.get("start", 0))))
            fin = max(inicio, min(total_len, int(span.get("end", 0))))
            if fin > inicio:
                mascara_activa[inicio:fin] = True
                limites.append((inicio, fin))

        if not limites:
            # En locuciones muy breves desde celular (monosílabos como "sí", "no", "dos", "stock")
            # con cancelación agresiva de ruido, reintentar Silero con umbral más sensible solo si
            # existe un pico acústico claro que descarte silencio real.
            if 0.15 <= duracion_original <= 1.5 and self.umbral_probabilidad_voz > 0.22:
                global_rms = float(np.sqrt(np.mean(samples ** 2)))
                global_rms_dbfs = 20.0 * math.log10(max(global_rms, 1e-6))
                global_peak = float(np.max(np.abs(samples)))
                global_peak_dbfs = 20.0 * math.log10(max(global_peak, 1e-6))
                if global_peak_dbfs >= -26.0 and global_rms_dbfs >= -42.0:
                    try:
                        spans_corto = get_speech_timestamps(
                            samples,
                            VadOptions(
                                threshold=0.22,
                                min_speech_duration_ms=80,
                                min_silence_duration_ms=min(self.silencio_minimo_ms, 250),
                                speech_pad_ms=self.margen_silencio_ms,
                            ),
                            sampling_rate=self.sample_rate,
                        )
                        for span in spans_corto:
                            inicio = max(0, min(total_len, int(span.get("start", 0))))
                            fin = max(inicio, min(total_len, int(span.get("end", 0))))
                            if fin > inicio:
                                mascara_activa[inicio:fin] = True
                                limites.append((inicio, fin))
                    except Exception:
                        pass
            if not limites:
                return self._resultado_sin_habla(samples, duracion_original)

        inicio = limites[0][0]
        fin = limites[-1][1]
        duracion_voz = sum(fin_span - inicio_span for inicio_span, fin_span in limites) / self.sample_rate
        return VADResult(
            tiene_voz=True,
            motivo="VOZ_DETECTADA",
            muestras_activas=samples[inicio:fin],
            duracion_original_seg=duracion_original,
            duracion_voz_seg=duracion_voz,
            ratio_voz=min(1.0, duracion_voz / max(duracion_original, 1e-9)),
            mascara_activa=mascara_activa[inicio:fin],
        )

    def _resultado_sin_habla(self, samples: np.ndarray, duracion_original: float) -> VADResult:
        """Distingue silencio de señal sin habla usando energía y pico globales."""
        global_rms = float(np.sqrt(np.mean(samples ** 2)))
        global_rms_dbfs = 20.0 * math.log10(max(global_rms, 1e-6))
        global_peak = float(np.max(np.abs(samples)))
        global_peak_dbfs = 20.0 * math.log10(max(global_peak, 1e-6))
        motivo = (
            "AUDIO_SILENCIOSO"
            if global_rms_dbfs < self.umbral_silencio_dbfs
            and global_peak_dbfs < self.umbral_silencio_dbfs + 10.0
            else "SOLO_RUIDO"
        )
        return VADResult(
            tiene_voz=False,
            motivo=motivo,
            muestras_activas=np.zeros(0, dtype=np.float32),
            duracion_original_seg=duracion_original,
            duracion_voz_seg=0.0,
            ratio_voz=0.0,
        )

    def _procesar_por_energia(self, samples: np.ndarray) -> VADResult:
        """Detector anterior como fallback para instalaciones sin el paquete VAD."""
        total_len = len(samples)
        if total_len == 0:
            return self._resultado_sin_habla(samples, 0.0)

        duracion_original = total_len / self.sample_rate

        # 1. Chequeo rápido de energía global y pico
        global_rms = float(np.sqrt(np.mean(samples ** 2)))
        global_rms_dbfs = 20.0 * math.log10(max(global_rms, 1e-6))
        global_peak = float(np.max(np.abs(samples))) if total_len > 0 else 0.0
        global_peak_dbfs = 20.0 * math.log10(max(global_peak, 1e-6))

        # Solo descartar como silencio puro si TANTO el RMS promedio como el pico máximo están por debajo del umbral.
        # Esto previene que locuciones cortas (ej. "sí", "dos", "no") rodeadas de pausas sean descartadas por promedio diluido.
        if global_rms_dbfs < self.umbral_silencio_dbfs and global_peak_dbfs < (self.umbral_silencio_dbfs + 10.0):
            return self._resultado_sin_habla(samples, duracion_original)

        # 2. Análisis trama a trama (30ms por frame)
        num_frames = total_len // self.frame_len
        if num_frames == 0:
            return VADResult(
                tiene_voz=False,
                motivo="SOLO_RUIDO",
                muestras_activas=np.zeros(0, dtype=np.float32),
                duracion_original_seg=duracion_original,
                duracion_voz_seg=0.0,
                ratio_voz=0.0,
            )

        frames_voz = np.zeros(num_frames, dtype=bool)

        for i in range(num_frames):
            frame = samples[i * self.frame_len : (i + 1) * self.frame_len]
            f_rms = np.sqrt(np.mean(frame ** 2))
            f_rms_dbfs = 20.0 * math.log10(max(f_rms, 1e-6))

            # Cruce por cero para diferenciar sibilantes de zumbido de baja frecuencia
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2.0
            zcr = zero_crossings / len(frame)

            # Criterio de voz: energía sobre el umbral y ZCR dentro del rango del habla humana (0.015 - 0.65)
            if f_rms_dbfs > self.umbral_silencio_dbfs and 0.015 < zcr < 0.65:
                frames_voz[i] = True

        total_frames_voz = int(np.sum(frames_voz))
        ratio_voz = total_frames_voz / num_frames
        duracion_voz = total_frames_voz * (self.frame_len / self.sample_rate)

        # Si no supera el umbral mínimo de habla (calibrado a 150 ms para locuciones breves y monosílabos)
        umbral_min_habla = min(self.min_segundos_voz, 0.15)
        if duracion_voz < umbral_min_habla:
            return VADResult(
                tiene_voz=False,
                motivo="SOLO_RUIDO",
                muestras_activas=np.zeros(0, dtype=np.float32),
                duracion_original_seg=duracion_original,
                duracion_voz_seg=duracion_voz,
                ratio_voz=ratio_voz,
            )

        # 3. Recorte de silencios en los extremos (con margen de seguridad para no cortar sílabas)
        indices_voz = np.where(frames_voz)[0]
        start_frame = max(0, indices_voz[0] - self.margen_frames)
        end_frame = min(num_frames, indices_voz[-1] + self.margen_frames + 1)

        start_idx = start_frame * self.frame_len
        end_idx = min(total_len, end_frame * self.frame_len)

        muestras_recortadas = samples[start_idx:end_idx]

        # Máscara booleana muestra a muestra de las tramas activas dentro de muestras_recortadas
        sub_frames = frames_voz[start_frame:end_frame]
        mascara_muestras = np.repeat(sub_frames, self.frame_len)
        if len(mascara_muestras) > len(muestras_recortadas):
            mascara_muestras = mascara_muestras[:len(muestras_recortadas)]
        elif len(mascara_muestras) < len(muestras_recortadas):
            pad_len = len(muestras_recortadas) - len(mascara_muestras)
            mascara_muestras = np.pad(mascara_muestras, (0, pad_len), "constant")

        return VADResult(
            tiene_voz=True,
            motivo="VOZ_DETECTADA",
            muestras_activas=muestras_recortadas,
            duracion_original_seg=duracion_original,
            duracion_voz_seg=len(muestras_recortadas) / self.sample_rate,
            ratio_voz=ratio_voz,
            mascara_activa=mascara_muestras,
        )
