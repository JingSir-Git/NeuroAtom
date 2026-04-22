"""Import-time signal validation.

Checks for common data quality issues during import:
- All-zero channels
- All-NaN channels
- Extreme amplitude values
- Flat-line detection
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def validate_signal(
    signal: np.ndarray,
    atom_id: str,
    config: Dict[str, Any],
) -> List[str]:
    """Validate a signal array and return a list of warning messages.

    Args:
        signal: Array of shape (n_channels, n_samples).
        atom_id: Atom identifier for logging.
        config: Import config dict with validation thresholds.

    Returns:
        List of warning strings (empty if signal passes all checks).
    """
    warnings: List[str] = []

    if signal.size == 0:
        warnings.append(f"[{atom_id}] Empty signal array.")
        return warnings

    skip_all_zero = config.get("skip_all_zero", True)
    skip_all_nan = config.get("skip_all_nan", True)
    amp_range = config.get("amplitude_range_uv", None)

    # Check for all-NaN
    if skip_all_nan and np.all(np.isnan(signal)):
        warnings.append(f"[{atom_id}] All values are NaN.")
        return warnings

    # Check for any NaN
    nan_count = np.count_nonzero(np.isnan(signal))
    if nan_count > 0:
        nan_pct = 100.0 * nan_count / signal.size
        warnings.append(
            f"[{atom_id}] Contains {nan_count} NaN values ({nan_pct:.1f}%)."
        )

    # Check for all-zero (per channel)
    if skip_all_zero:
        zero_channels = []
        for ch_idx in range(signal.shape[0]):
            if np.all(signal[ch_idx] == 0):
                zero_channels.append(ch_idx)
        if zero_channels:
            if len(zero_channels) == signal.shape[0]:
                warnings.append(f"[{atom_id}] All channels are all-zero.")
            else:
                warnings.append(
                    f"[{atom_id}] {len(zero_channels)} all-zero channel(s): {zero_channels[:5]}..."
                )

    # Check amplitude range (in µV, assumes signal has been unit-standardized or is in expected unit)
    if amp_range is not None:
        low, high = amp_range
        signal_clean = signal[~np.isnan(signal)] if nan_count > 0 else signal
        if signal_clean.size > 0:
            sig_min = float(np.min(signal_clean))
            sig_max = float(np.max(signal_clean))
            if sig_min < low or sig_max > high:
                warnings.append(
                    f"[{atom_id}] Amplitude out of range [{low}, {high}]: "
                    f"min={sig_min:.1f}, max={sig_max:.1f}."
                )

    # Flat-line detection: std < threshold per channel
    flatline_threshold = config.get("flatline_std_threshold", 0.01)
    flat_channels = []
    for ch_idx in range(signal.shape[0]):
        ch_data = signal[ch_idx]
        ch_clean = ch_data[~np.isnan(ch_data)] if nan_count > 0 else ch_data
        if ch_clean.size > 1 and np.std(ch_clean) < flatline_threshold:
            flat_channels.append(ch_idx)
    if flat_channels:
        warnings.append(
            f"[{atom_id}] {len(flat_channels)} flat-line channel(s) "
            f"(std < {flatline_threshold}): {flat_channels[:5]}..."
        )

    if warnings:
        for w in warnings:
            logger.warning(w)

    return warnings
