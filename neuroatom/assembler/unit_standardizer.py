"""UnitStandardizer: convert signal units to a target unit (default µV).

Must be the FIRST step in the assembly pipeline — before any arithmetic
(re-referencing, filtering, normalization).

Supported conversions: V → µV, mV → µV, nV → µV, µV → µV (no-op).
Unknown units are handled per error_handling policy.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Conversion factors to µV
_TO_UV: dict = {
    "V": 1e6,
    "v": 1e6,
    "mV": 1e3,
    "mv": 1e3,
    "uV": 1.0,
    "uv": 1.0,
    "µV": 1.0,
    "nV": 1e-3,
    "nv": 1e-3,
}


class UnitStandardizer:
    """Convert signal amplitude to a target unit.

    Usage:
        standardizer = UnitStandardizer(target_unit="uV")
        signal_uv = standardizer.convert(signal, source_unit="V")
    """

    def __init__(self, target_unit: str = "uV"):
        self._target_unit = target_unit
        if target_unit not in _TO_UV:
            raise ValueError(f"Unsupported target unit: {target_unit}")
        self._target_factor = _TO_UV[target_unit]

    def convert(
        self,
        signal: np.ndarray,
        source_unit: str,
        error_handling: str = "skip",
    ) -> np.ndarray:
        """Convert signal from source_unit to target_unit.

        Args:
            signal: Signal array of any shape.
            source_unit: Source unit string (e.g., 'V', 'mV', 'uV').
            error_handling: What to do if source_unit is unknown:
                'raise': raise ValueError
                'skip': return signal unchanged with warning
                'substitute': assume µV with warning

        Returns:
            Converted signal array (float32).
        """
        if source_unit in _TO_UV:
            source_factor = _TO_UV[source_unit]
            if source_factor == self._target_factor:
                return signal.astype(np.float32)
            scale = source_factor / self._target_factor
            return (signal * scale).astype(np.float32)

        # Unknown unit
        if error_handling == "raise":
            raise ValueError(f"Unknown signal unit: '{source_unit}'")
        elif error_handling == "substitute":
            logger.warning(
                "Unknown unit '%s', assuming µV (error_handling=substitute).",
                source_unit,
            )
            return signal.astype(np.float32)
        else:  # skip
            logger.warning(
                "Unknown unit '%s', returning signal unchanged (error_handling=skip).",
                source_unit,
            )
            return signal.astype(np.float32)

    @property
    def target_unit(self) -> str:
        return self._target_unit
