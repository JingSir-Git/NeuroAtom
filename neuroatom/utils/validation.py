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


def _unit_scale_to_uv(signal_unit: str) -> float:
    """Return the multiplier that converts ``signal_unit`` → µV.

    Examples:
        V  → 1e6   (1 V  = 1 000 000 µV)
        mV → 1e3   (1 mV = 1 000 µV)
        uV → 1     (identity)
    """
    _SCALE = {
        "V":  1e6,
        "mV": 1e3,
        "uV": 1.0,
        "µV": 1.0,
    }
    return _SCALE.get(signal_unit, 1.0)


def validate_signal(
    signal: np.ndarray,
    atom_id: str,
    config: Dict[str, Any],
    *,
    signal_unit: str = "",
) -> List[str]:
    """Validate a signal array and return a list of warning messages.

    Args:
        signal: Array of shape (n_channels, n_samples).
        atom_id: Atom identifier for logging.
        config: Import config dict with validation thresholds.
        signal_unit: Explicit signal unit (``"V"``, ``"mV"``, ``"uV"``).
            Overrides ``config["signal_unit"]``.  All amplitude /
            flat-line thresholds are automatically scaled to match.

    Returns:
        List of warning strings (empty if signal passes all checks).
    """
    warnings: List[str] = []

    if signal.size == 0:
        warnings.append(f"[{atom_id}] Empty signal array.")
        return warnings

    # Resolve effective unit: explicit kwarg > config > "uV" default
    effective_unit = signal_unit or config.get("signal_unit", "uV")
    scale = _unit_scale_to_uv(effective_unit)  # signal × scale = µV

    skip_all_zero = config.get("skip_all_zero", True)
    skip_all_nan = config.get("skip_all_nan", True)
    amp_range_uv = config.get("amplitude_range_uv", None)

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

    # Check for any Inf
    inf_count = np.count_nonzero(np.isinf(signal))
    if inf_count > 0:
        inf_pct = 100.0 * inf_count / signal.size
        warnings.append(
            f"[{atom_id}] Contains {inf_count} Inf values ({inf_pct:.1f}%)."
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

    # Amplitude range — thresholds are defined in µV, scale to signal unit
    if amp_range_uv is not None:
        low_native = amp_range_uv[0] / scale   # µV → signal unit
        high_native = amp_range_uv[1] / scale
        signal_clean = signal[~np.isnan(signal)] if nan_count > 0 else signal
        if signal_clean.size > 0:
            sig_min = float(np.min(signal_clean))
            sig_max = float(np.max(signal_clean))
            if sig_min < low_native or sig_max > high_native:
                # Report in µV for consistent logging
                warnings.append(
                    f"[{atom_id}] Amplitude out of range "
                    f"[{amp_range_uv[0]}, {amp_range_uv[1]}] µV: "
                    f"min={sig_min * scale:.1f} µV, max={sig_max * scale:.1f} µV."
                )

    # Flat-line detection: std < threshold per channel
    # Base threshold is 0.01 µV; scale to signal unit
    flatline_threshold = config.get("flatline_std_threshold", None)
    if flatline_threshold is None:
        flatline_threshold = 0.01 / scale  # 0.01 µV → signal unit
    flat_channels = []
    for ch_idx in range(signal.shape[0]):
        ch_data = signal[ch_idx]
        ch_clean = ch_data[~np.isnan(ch_data)] if nan_count > 0 else ch_data
        if ch_clean.size > 1 and np.std(ch_clean) < flatline_threshold:
            flat_channels.append(ch_idx)
    if flat_channels:
        warnings.append(
            f"[{atom_id}] {len(flat_channels)} flat-line channel(s) "
            f"(std < {flatline_threshold:.2e} {effective_unit}): {flat_channels[:5]}..."
        )

    if warnings:
        for w in warnings:
            logger.warning(w)

    return warnings


def validate_sampling_rate(srate: float, context: str = "") -> None:
    """Validate that sampling rate is positive and reasonable.

    Raises:
        ValueError: If sampling rate is non-positive or unreasonably high.
    """
    prefix = f"[{context}] " if context else ""
    if not np.isfinite(srate):
        raise ValueError(f"{prefix}Sampling rate is not finite: {srate}")
    if srate <= 0:
        raise ValueError(f"{prefix}Sampling rate must be positive, got {srate}")
    if srate > 1_000_000:
        logger.warning(
            "%sSampling rate %.0f Hz is unusually high (>1 MHz). "
            "Check if this is correct.",
            prefix, srate,
        )
