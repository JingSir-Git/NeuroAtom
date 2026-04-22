"""Rereferencer: apply EEG re-referencing on the FULL channel set.

Must operate on ALL channels (minus bad channels) BEFORE channel selection,
because average reference needs the full electrode coverage.

Supported schemes:
- average: subtract mean of all (non-bad) channels
- linked_ears: subtract mean of A1 and A2 (or TP9/TP10)
- Cz: subtract Cz channel

REST (Reference Electrode Standardization Technique) requires a leadfield
matrix and is planned for a future version.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Rereferencer:
    """Apply re-referencing to EEG signal data.

    Usage:
        reref = Rereferencer(
            target_reference="average",
            exclude_channels=["ch_bad1", "ch_bad2"],
        )
        signal_reref = reref.apply(signal, channel_ids)
    """

    def __init__(
        self,
        target_reference: str = "average",
        exclude_channels: Optional[List[str]] = None,
        reference_channels: Optional[Dict[str, List[str]]] = None,
        # Reserved for future REST support
        reference_matrix_cache: Optional[str] = None,
    ):
        """
        Args:
            target_reference: Reference scheme to apply.
            exclude_channels: Channels to exclude from reference computation
                (typically bad channels from QualityInfo.bad_channels).
            reference_channels: For 'linked_ears', mapping of ref scheme to
                channel IDs, e.g. {'linked_ears': ['A1', 'A2']}.
            reference_matrix_cache: Reserved for future REST support.
                Will point to a cached leadfield matrix file.
        """
        self._target = target_reference
        self._exclude = set(exclude_channels or [])
        self._ref_channels = reference_channels or {}
        self._matrix_cache = reference_matrix_cache

    def apply(
        self,
        signal: np.ndarray,
        channel_ids: List[str],
    ) -> np.ndarray:
        """Apply re-referencing to a signal array.

        Args:
            signal: Array of shape (n_channels, n_samples).
            channel_ids: Ordered list of channel IDs matching signal rows.

        Returns:
            Re-referenced signal array (same shape).
        """
        signal = signal.copy().astype(np.float32)
        n_channels, n_samples = signal.shape

        if self._target == "average":
            return self._apply_average(signal, channel_ids)
        elif self._target == "linked_ears":
            return self._apply_linked_ears(signal, channel_ids)
        elif self._target in ("Cz", "cz", "CZ"):
            return self._apply_single_channel(signal, channel_ids, "Cz")
        elif self._target == "REST":
            raise NotImplementedError(
                "REST reference requires a leadfield matrix. "
                "Use reference_matrix_cache parameter (not yet implemented)."
            )
        else:
            logger.warning("Unknown reference scheme '%s', returning unchanged.", self._target)
            return signal

    def _apply_average(
        self, signal: np.ndarray, channel_ids: List[str]
    ) -> np.ndarray:
        """Average reference: subtract mean of all non-excluded channels."""
        # Build mask of included channels
        include_mask = np.array(
            [ch_id not in self._exclude for ch_id in channel_ids],
            dtype=bool,
        )
        n_included = include_mask.sum()
        if n_included == 0:
            logger.warning("All channels excluded from average reference. Returning unchanged.")
            return signal

        # Compute reference as mean of included channels
        reference = signal[include_mask].mean(axis=0)

        # Subtract from ALL channels (including excluded ones)
        signal -= reference[np.newaxis, :]
        return signal

    def _apply_linked_ears(
        self, signal: np.ndarray, channel_ids: List[str]
    ) -> np.ndarray:
        """Linked ears reference: subtract mean of A1/A2 or TP9/TP10."""
        ref_ids = self._ref_channels.get("linked_ears", [])
        if not ref_ids:
            # Try standard names
            for pair in [("A1", "A2"), ("TP9", "TP10"), ("M1", "M2")]:
                if pair[0] in channel_ids and pair[1] in channel_ids:
                    ref_ids = list(pair)
                    break

        if not ref_ids:
            logger.warning("Could not find ear reference channels. Returning unchanged.")
            return signal

        ref_indices = [channel_ids.index(ch) for ch in ref_ids if ch in channel_ids]
        if not ref_indices:
            logger.warning("Reference channels %s not found in channel_ids.", ref_ids)
            return signal

        reference = signal[ref_indices].mean(axis=0)
        signal -= reference[np.newaxis, :]
        return signal

    def _apply_single_channel(
        self, signal: np.ndarray, channel_ids: List[str], ref_name: str
    ) -> np.ndarray:
        """Single channel reference (e.g., Cz)."""
        if ref_name not in channel_ids:
            logger.warning("Reference channel '%s' not found. Returning unchanged.", ref_name)
            return signal

        ref_idx = channel_ids.index(ref_name)
        reference = signal[ref_idx]
        signal -= reference[np.newaxis, :]
        return signal
