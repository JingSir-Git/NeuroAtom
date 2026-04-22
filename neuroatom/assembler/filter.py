"""Signal filtering: bandpass and notch filters for EEG preprocessing.

Uses scipy.signal for zero-phase (filtfilt) filtering to avoid
phase distortion, which is critical for EEG analysis.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

logger = logging.getLogger(__name__)


class SignalFilter:
    """Bandpass and notch filtering for EEG signals.

    Usage:
        filt = SignalFilter(
            filter_band=(0.5, 40.0),
            notch_freq=50.0,
            sampling_rate=256.0,
        )
        filtered = filt.apply(signal)
    """

    def __init__(
        self,
        sampling_rate: float,
        filter_band: Optional[Tuple[float, float]] = None,
        notch_freq: Optional[float] = None,
        filter_order: int = 5,
        notch_quality: float = 30.0,
    ):
        self._srate = sampling_rate
        self._band = filter_band
        self._notch_freq = notch_freq
        self._filter_order = filter_order
        self._notch_quality = notch_quality
        self._nyq = sampling_rate / 2.0

        # Pre-compute filter coefficients
        self._bp_b = None
        self._bp_a = None
        self._notch_b = None
        self._notch_a = None

        if filter_band is not None:
            low, high = filter_band
            if low is not None and high is not None:
                self._bp_b, self._bp_a = butter(
                    filter_order,
                    [low / self._nyq, high / self._nyq],
                    btype="band",
                )
            elif low is not None:
                self._bp_b, self._bp_a = butter(
                    filter_order, low / self._nyq, btype="high"
                )
            elif high is not None:
                self._bp_b, self._bp_a = butter(
                    filter_order, high / self._nyq, btype="low"
                )

        if notch_freq is not None:
            self._notch_b, self._notch_a = iirnotch(
                notch_freq, notch_quality, sampling_rate
            )

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass and/or notch filter to signal.

        Args:
            signal: Array of shape (n_channels, n_samples).

        Returns:
            Filtered signal (same shape, float32).
        """
        signal = signal.astype(np.float64)  # filtfilt needs float64

        # Apply bandpass
        if self._bp_b is not None:
            # Minimum signal length for filtfilt
            min_len = 3 * max(len(self._bp_b), len(self._bp_a))
            if signal.shape[1] > min_len:
                signal = filtfilt(self._bp_b, self._bp_a, signal, axis=1)
            else:
                logger.warning(
                    "Signal too short (%d samples) for bandpass filter (need > %d). Skipping.",
                    signal.shape[1], min_len,
                )

        # Apply notch
        if self._notch_b is not None:
            min_len = 3 * max(len(self._notch_b), len(self._notch_a))
            if signal.shape[1] > min_len:
                signal = filtfilt(self._notch_b, self._notch_a, signal, axis=1)
            else:
                logger.warning(
                    "Signal too short for notch filter. Skipping.",
                )

        return signal.astype(np.float32)
