"""Import-time unit conversion: standardize signal to pool storage unit.

Called by every importer right before writing signal to HDF5.
The pool convention is µV — this module provides the conversion
and records provenance (original_unit → storage_unit).

Usage in an importer::

    from neuroatom.utils.unit_convert import convert_to_storage_unit

    signal_uv, storage_unit, original_unit = convert_to_storage_unit(
        signal, source_unit="V", pool_config=pool.config,
    )
    # signal_uv is now in µV, write to HDF5
    # set atom.signal_unit = storage_unit, atom.original_unit = original_unit
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Conversion factors: source_unit → µV
_TO_UV = {
    "V":  1e6,
    "v":  1e6,
    "mV": 1e3,
    "mv": 1e3,
    "uV": 1.0,
    "uv": 1.0,
    "µV": 1.0,
    "nV": 1e-3,
    "nv": 1e-3,
}


def convert_to_storage_unit(
    signal: np.ndarray,
    source_unit: str,
    pool_config: Optional[dict] = None,
) -> Tuple[np.ndarray, str, Optional[str]]:
    """Convert signal to the pool's storage unit (default µV).

    Args:
        signal: Signal array of any shape (typically channels × samples).
        source_unit: Physical unit of the input signal ('V', 'mV', 'uV', etc.).
        pool_config: Pool configuration dict. If provided, reads the target
            unit from ``pool_config["storage_conventions"]["signal_unit"]``.

    Returns:
        Tuple of:
        - **signal_out**: Converted signal as float32.
        - **storage_unit**: The unit the signal is now in (e.g. ``"uV"``).
        - **original_unit**: The original unit before conversion, or ``None``
          if no conversion was needed.
    """
    # Determine target unit from pool config (default µV)
    target_unit = "uV"
    if pool_config:
        conventions = pool_config.get("storage_conventions", {})
        target_unit = conventions.get("signal_unit", "uV")

    # Normalize source and target to canonical forms
    src_factor = _TO_UV.get(source_unit)
    tgt_factor = _TO_UV.get(target_unit)

    if src_factor is None:
        logger.warning(
            "Unknown source unit '%s', storing signal as-is. "
            "Set source_unit explicitly in your importer.",
            source_unit,
        )
        return signal.astype(np.float32), target_unit, source_unit

    if tgt_factor is None:
        logger.warning(
            "Unknown target unit '%s' in pool config, defaulting to µV.",
            target_unit,
        )
        tgt_factor = 1.0
        target_unit = "uV"

    # Compute scale factor: source → µV → target
    if src_factor == tgt_factor:
        # No conversion needed
        return signal.astype(np.float32), target_unit, None

    scale = src_factor / tgt_factor
    converted = (signal * scale).astype(np.float32)

    logger.debug(
        "Unit conversion: %s → %s (scale=%.6g)", source_unit, target_unit, scale,
    )
    return converted, target_unit, source_unit
