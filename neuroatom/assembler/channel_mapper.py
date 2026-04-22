"""ChannelMapper: select, reorder, interpolate, or zero-fill channels.

Operates AFTER re-referencing. Maps the source channel set to the target
channel set specified in AssemblyRecipe.target_channels.

Strategies:
- exact match: select channels by standard_name
- zero-fill: missing channels filled with zeros + channel_mask=0
- interpolate: spherical spline interpolation (requires electrode positions)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ChannelMapper:
    """Map source channels to a target channel layout.

    Usage:
        mapper = ChannelMapper(
            target_channels=["C3", "Cz", "C4"],
            missing_strategy="zero_fill",
        )
        mapped_signal, channel_mask = mapper.apply(signal, source_channel_map)
    """

    def __init__(
        self,
        target_channels: List[str],
        missing_strategy: str = "zero_fill",
    ):
        """
        Args:
            target_channels: Ordered list of target standard_name channels.
            missing_strategy: How to handle missing channels:
                'zero_fill': fill with zeros (default)
                'drop': skip atom entirely (return None)
                'interpolate': spherical spline (requires positions, not yet implemented)
        """
        self._target_channels = target_channels
        self._missing_strategy = missing_strategy

    def apply(
        self,
        signal: np.ndarray,
        source_channel_map: Dict[str, int],
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Map source signal to target channel layout.

        Args:
            signal: Source signal, shape (n_source_channels, n_samples).
            source_channel_map: Mapping from standard_name → row index in signal.

        Returns:
            Tuple of:
                - Mapped signal: shape (n_target_channels, n_samples), or None if dropped
                - Channel mask: binary array (n_target_channels,), 1=real, 0=zero-filled
        """
        n_samples = signal.shape[1]
        n_target = len(self._target_channels)

        mapped = np.zeros((n_target, n_samples), dtype=np.float32)
        mask = np.zeros(n_target, dtype=np.float32)

        missing_channels = []

        for target_idx, ch_name in enumerate(self._target_channels):
            if ch_name in source_channel_map:
                src_idx = source_channel_map[ch_name]
                mapped[target_idx] = signal[src_idx]
                mask[target_idx] = 1.0
            else:
                missing_channels.append(ch_name)

        if missing_channels:
            if self._missing_strategy == "drop":
                logger.debug(
                    "Missing channels %s, dropping atom (strategy=drop).",
                    missing_channels,
                )
                return None, mask
            elif self._missing_strategy == "interpolate":
                logger.warning(
                    "Channel interpolation not yet implemented. "
                    "Falling back to zero-fill for: %s",
                    missing_channels,
                )
            # zero_fill: already zeros from initialization
            logger.debug(
                "%d/%d channels zero-filled: %s",
                len(missing_channels), n_target, missing_channels[:5],
            )

        return mapped, mask

    @property
    def target_channels(self) -> List[str]:
        return self._target_channels
