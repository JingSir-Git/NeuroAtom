"""Resampler: sampling rate normalization for EEG signals.

Uses scipy.signal.resample_poly for efficient polyphase resampling,
which provides anti-aliasing filtering automatically.
"""

import logging
import math
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


class Resampler:
    """Resample EEG signals to a target sampling rate.

    Usage:
        resampler = Resampler(target_rate=128.0)
        resampled = resampler.apply(signal, source_rate=256.0)
    """

    def __init__(self, target_rate: float):
        self._target_rate = target_rate

    def apply(
        self,
        signal: np.ndarray,
        source_rate: float,
    ) -> np.ndarray:
        """Resample signal from source_rate to target_rate.

        Args:
            signal: Array of shape (n_channels, n_samples).
            source_rate: Source sampling rate in Hz.

        Returns:
            Resampled signal of shape (n_channels, new_n_samples).
        """
        if abs(source_rate - self._target_rate) < 0.01:
            return signal.astype(np.float32)

        # Find rational fraction up/down
        gcd = math.gcd(int(self._target_rate), int(source_rate))
        up = int(self._target_rate) // gcd
        down = int(source_rate) // gcd

        logger.debug(
            "Resampling: %.1f Hz → %.1f Hz (up=%d, down=%d)",
            source_rate, self._target_rate, up, down,
        )

        resampled = resample_poly(signal, up, down, axis=1)
        return resampled.astype(np.float32)

    @property
    def target_rate(self) -> float:
        return self._target_rate
