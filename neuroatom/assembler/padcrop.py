"""PadCrop: pad or crop signals to a target duration.

Generates a time_mask indicating which samples are real signal (1)
versus padding (0). This mask is critical for attention-based models
that need to ignore padded regions.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PadCrop:
    """Pad or crop EEG signals to a fixed target length.

    Usage:
        padcrop = PadCrop(target_samples=1024, pad_value=0.0)
        signal_out, time_mask = padcrop.apply(signal)
    """

    def __init__(
        self,
        target_samples: int,
        pad_value: float = 0.0,
        pad_side: str = "right",
        crop_side: str = "right",
    ):
        """
        Args:
            target_samples: Target number of time samples.
            pad_value: Value used for padding (typically 0.0).
            pad_side: Where to add padding: 'right', 'left', 'both'.
            crop_side: Where to crop excess: 'right', 'left', 'center'.
        """
        self._target = target_samples
        self._pad_value = pad_value
        self._pad_side = pad_side
        self._crop_side = crop_side

    def apply(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad or crop signal to target length.

        Args:
            signal: Array of shape (n_channels, n_samples).

        Returns:
            Tuple of:
                - Processed signal: shape (n_channels, target_samples)
                - Time mask: shape (target_samples,), 1.0=real, 0.0=padded
        """
        n_channels, n_samples = signal.shape

        if n_samples == self._target:
            return signal.astype(np.float32), np.ones(self._target, dtype=np.float32)

        elif n_samples > self._target:
            return self._crop(signal, n_channels, n_samples)

        else:
            return self._pad(signal, n_channels, n_samples)

    def _crop(
        self, signal: np.ndarray, n_channels: int, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop signal to target length."""
        excess = n_samples - self._target

        if self._crop_side == "right":
            cropped = signal[:, :self._target]
        elif self._crop_side == "left":
            cropped = signal[:, excess:]
        elif self._crop_side == "center":
            start = excess // 2
            cropped = signal[:, start:start + self._target]
        else:
            cropped = signal[:, :self._target]

        mask = np.ones(self._target, dtype=np.float32)
        return cropped.astype(np.float32), mask

    def _pad(
        self, signal: np.ndarray, n_channels: int, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad signal to target length."""
        deficit = self._target - n_samples
        result = np.full(
            (n_channels, self._target), self._pad_value, dtype=np.float32
        )
        mask = np.zeros(self._target, dtype=np.float32)

        if self._pad_side == "right":
            result[:, :n_samples] = signal
            mask[:n_samples] = 1.0
        elif self._pad_side == "left":
            result[:, deficit:] = signal
            mask[deficit:] = 1.0
        elif self._pad_side == "both":
            left_pad = deficit // 2
            result[:, left_pad:left_pad + n_samples] = signal
            mask[left_pad:left_pad + n_samples] = 1.0
        else:
            result[:, :n_samples] = signal
            mask[:n_samples] = 1.0

        return result, mask

    @staticmethod
    def compute_target_samples(target_duration: float, sampling_rate: float) -> int:
        """Compute target sample count from duration and rate."""
        return int(round(target_duration * sampling_rate))
