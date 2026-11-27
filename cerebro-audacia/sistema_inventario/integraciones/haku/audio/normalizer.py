"""Normalización dinámica de volumen con limitador suave para Whisper."""

from __future__ import annotations

import math
import numpy as np


class AudioNormalizer:
    """Normaliza el nivel RMS a un umbral acústico óptimo (-18 dBFS) sin distorsión."""

    def __init__(
        self,
        target_rms_dbfs: float = -18.0,
        max_boost_db: float = 24.0,
        max_attenuation_db: float = 30.0,
        peak_limit: float = 0.92,
    ) -> None:
        self.target_rms_dbfs = target_rms_dbfs
        self.max_boost_factor = 10.0 ** (max_boost_db / 20.0)
        self.min_attenuation_factor = 10.0 ** (-max_attenuation_db / 20.0)
        self.peak_limit = peak_limit

    def normalizar(
        self,
        samples: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Ajusta el volumen RMS y limita picos para evitar recorte digital.

        Si se proporciona 'active_mask' (array booleano alineado con samples),
        el nivel RMS de referencia se calcula sobre las muestras activas de voz para evitar
        que silencios o pausas largas diluyan el promedio e inflen indebidamente el piso de ruido.
        """
        if len(samples) == 0:
            return samples

        # Seleccionar muestras de referencia para estimar la energía de la voz
        samples_ref = samples
        if active_mask is not None:
            try:
                # Validar que la máscara coincida en longitud y contenga suficientes muestras (>= 160 = 10ms a 16kHz)
                if (
                    isinstance(active_mask, np.ndarray)
                    and len(active_mask) == len(samples)
                    and np.issubdtype(active_mask.dtype, np.bool_)
                ):
                    num_activas = int(np.sum(active_mask))
                    if num_activas >= 160:
                        samples_ref = samples[active_mask]
            except Exception:
                samples_ref = samples

        current_rms = float(np.sqrt(np.mean(samples_ref ** 2)))
        if current_rms < 1e-6:
            return samples

        current_rms_dbfs = 20.0 * math.log10(current_rms)

        # Ganancia requerida
        delta_db = self.target_rms_dbfs - current_rms_dbfs
        factor = 10.0 ** (delta_db / 20.0)

        # Limitar factor de amplificación para no inflar el piso de ruido
        factor = max(self.min_attenuation_factor, min(self.max_boost_factor, factor))

        scaled = samples * factor

        # Limitador transparente por codo alto (Soft-Knee Limiter):
        # mantiene 100% lineal la señal vocal normal (|x| <= knee) evitando
        # distorsión armónica en consonantes rápidas, y comprime suavemente
        # solo los transitorios por encima de knee hasta peak_limit.
        max_peak = float(np.max(np.abs(scaled)))
        if max_peak > self.peak_limit:
            knee = min(0.80, self.peak_limit * 0.87)
            headroom = max(1e-4, self.peak_limit - knee)
            abs_scaled = np.abs(scaled)
            over = abs_scaled > knee
            if np.any(over):
                limited = scaled.copy()
                excess = abs_scaled[over] - knee
                limited[over] = np.sign(scaled[over]) * (knee + headroom * np.tanh(excess / headroom))
                scaled = limited

        return scaled.astype(np.float32)
