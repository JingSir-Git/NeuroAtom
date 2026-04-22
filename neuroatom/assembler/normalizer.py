"""Normalizer: amplitude normalization with method × scope.

Methods: zscore, robust, minmax
Scopes: per_atom (single-pass), per_channel, per_subject, global (two-pass)

Two-pass logic:
    Scope 'global' or 'per_subject' requires pre-scanning all atoms to
    compute statistics BEFORE applying normalization. The DatasetAssembler
    orchestrates the two passes; this module handles the computation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.core.enums import NormalizationMethod, NormalizationScope

logger = logging.getLogger(__name__)


class NormalizationStats:
    """Container for precomputed normalization statistics.

    Used for global/per_subject normalization (two-pass).
    Serializable to JSON for cache provenance.
    """

    def __init__(self):
        self._stats: Dict[str, Dict[str, np.ndarray]] = {}
        # Key structure depends on scope:
        # global: {"__global__": {"mean": ..., "std": ..., ...}}
        # per_subject: {"sub-01": {"mean": ..., "std": ...}, "sub-02": ...}
        # per_channel: {"__channels__": {"mean": ..., "std": ...}}

    def set_stats(
        self,
        scope_key: str,
        mean: np.ndarray,
        std: np.ndarray,
        median: Optional[np.ndarray] = None,
        iqr: Optional[np.ndarray] = None,
        min_val: Optional[np.ndarray] = None,
        max_val: Optional[np.ndarray] = None,
    ) -> None:
        self._stats[scope_key] = {
            "mean": mean,
            "std": std,
        }
        if median is not None:
            self._stats[scope_key]["median"] = median
        if iqr is not None:
            self._stats[scope_key]["iqr"] = iqr
        if min_val is not None:
            self._stats[scope_key]["min"] = min_val
        if max_val is not None:
            self._stats[scope_key]["max"] = max_val

    def get_stats(self, scope_key: str) -> Optional[Dict[str, np.ndarray]]:
        return self._stats.get(scope_key)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        result = {}
        for key, stats in self._stats.items():
            result[key] = {k: v.tolist() for k, v in stats.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationStats":
        obj = cls()
        for key, stats in data.items():
            obj._stats[key] = {k: np.array(v) for k, v in stats.items()}
        return obj


class Normalizer:
    """Apply amplitude normalization to EEG signals.

    Usage (per_atom, single-pass):
        normalizer = Normalizer(method="zscore", scope="per_atom")
        normalized = normalizer.apply(signal)

    Usage (global, two-pass):
        # Pass 1: compute stats
        stats_collector = StatsCollector(method="zscore", n_channels=64)
        for signal in all_signals:
            stats_collector.update(signal)
        stats = stats_collector.finalize()

        # Pass 2: apply
        normalizer = Normalizer(method="zscore", scope="global", precomputed_stats=stats)
        for signal in all_signals:
            normalized = normalizer.apply(signal, scope_key="__global__")
    """

    def __init__(
        self,
        method: NormalizationMethod,
        scope: NormalizationScope = NormalizationScope.PER_ATOM,
        precomputed_stats: Optional[NormalizationStats] = None,
        eps: float = 1e-8,
    ):
        self._method = method
        self._scope = scope
        self._stats = precomputed_stats
        self._eps = eps

    def apply(
        self,
        signal: np.ndarray,
        scope_key: Optional[str] = None,
    ) -> np.ndarray:
        """Normalize a signal array.

        Args:
            signal: Shape (n_channels, n_samples).
            scope_key: Key for looking up precomputed stats
                (required for global/per_subject scopes).

        Returns:
            Normalized signal (same shape, float32).
        """
        signal = signal.astype(np.float32)

        if self._scope == NormalizationScope.PER_ATOM:
            return self._normalize_per_atom(signal)
        elif self._scope == NormalizationScope.PER_CHANNEL:
            return self._normalize_per_channel(signal)
        else:
            # global or per_subject: need precomputed stats
            if self._stats is None or scope_key is None:
                raise ValueError(
                    f"Scope '{self._scope.value}' requires precomputed_stats and scope_key."
                )
            stats = self._stats.get_stats(scope_key)
            if stats is None:
                raise ValueError(f"No stats found for scope_key '{scope_key}'.")
            return self._normalize_with_stats(signal, stats)

    def _normalize_per_atom(self, signal: np.ndarray) -> np.ndarray:
        """Single-pass normalization using atom's own statistics."""
        if self._method == NormalizationMethod.ZSCORE:
            mean = signal.mean()
            std = signal.std()
            return ((signal - mean) / (std + self._eps)).astype(np.float32)

        elif self._method == NormalizationMethod.ROBUST:
            median = np.median(signal)
            q75, q25 = np.percentile(signal, [75, 25])
            iqr = q75 - q25
            return ((signal - median) / (iqr + self._eps)).astype(np.float32)

        elif self._method == NormalizationMethod.MINMAX:
            min_val = signal.min()
            max_val = signal.max()
            range_val = max_val - min_val
            return ((signal - min_val) / (range_val + self._eps)).astype(np.float32)

        raise ValueError(f"Unknown method: {self._method}")

    def _normalize_per_channel(self, signal: np.ndarray) -> np.ndarray:
        """Per-channel normalization (each channel normalized independently)."""
        result = np.empty_like(signal)

        for ch in range(signal.shape[0]):
            ch_data = signal[ch]
            if self._method == NormalizationMethod.ZSCORE:
                mean = ch_data.mean()
                std = ch_data.std()
                result[ch] = (ch_data - mean) / (std + self._eps)
            elif self._method == NormalizationMethod.ROBUST:
                median = np.median(ch_data)
                q75, q25 = np.percentile(ch_data, [75, 25])
                iqr = q75 - q25
                result[ch] = (ch_data - median) / (iqr + self._eps)
            elif self._method == NormalizationMethod.MINMAX:
                min_val = ch_data.min()
                max_val = ch_data.max()
                range_val = max_val - min_val
                result[ch] = (ch_data - min_val) / (range_val + self._eps)

        return result.astype(np.float32)

    def _normalize_with_stats(
        self, signal: np.ndarray, stats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Normalize using precomputed statistics (for global/per_subject)."""
        if self._method == NormalizationMethod.ZSCORE:
            mean = stats["mean"]
            std = stats["std"]
            # Stats shape: (n_channels,) — broadcast over time
            if mean.ndim == 1:
                mean = mean[:, np.newaxis]
                std = std[:, np.newaxis]
            return ((signal - mean) / (std + self._eps)).astype(np.float32)

        elif self._method == NormalizationMethod.ROBUST:
            median = stats["median"]
            iqr = stats["iqr"]
            if median.ndim == 1:
                median = median[:, np.newaxis]
                iqr = iqr[:, np.newaxis]
            return ((signal - median) / (iqr + self._eps)).astype(np.float32)

        elif self._method == NormalizationMethod.MINMAX:
            min_val = stats["min"]
            max_val = stats["max"]
            if min_val.ndim == 1:
                min_val = min_val[:, np.newaxis]
                max_val = max_val[:, np.newaxis]
            range_val = max_val - min_val
            return ((signal - min_val) / (range_val + self._eps)).astype(np.float32)

        raise ValueError(f"Unknown method: {self._method}")


class StatsCollector:
    """Incrementally collect statistics for two-pass normalization.

    Pass 1: call update(signal) for each atom
    Pass 2: call finalize() to get NormalizationStats
    """

    def __init__(
        self,
        method: NormalizationMethod,
        n_channels: int,
        scope: NormalizationScope,
    ):
        self._method = method
        self._n_channels = n_channels
        self._scope = scope

        # Running statistics (Welford's algorithm for mean/var)
        self._counts: Dict[str, int] = {}           # scope_key → sample count
        self._means: Dict[str, np.ndarray] = {}
        self._m2s: Dict[str, np.ndarray] = {}       # for variance
        self._mins: Dict[str, np.ndarray] = {}
        self._maxs: Dict[str, np.ndarray] = {}
        # For robust: accumulate all values (memory-intensive but correct)
        self._all_values: Dict[str, List[np.ndarray]] = {}

    def update(self, signal: np.ndarray, scope_key: str = "__global__") -> None:
        """Update running statistics with a new signal.

        Args:
            signal: Shape (n_channels, n_samples).
            scope_key: Grouping key (e.g., "__global__", subject_id).
        """
        if scope_key not in self._counts:
            self._counts[scope_key] = 0
            self._means[scope_key] = np.zeros(self._n_channels, dtype=np.float64)
            self._m2s[scope_key] = np.zeros(self._n_channels, dtype=np.float64)
            self._mins[scope_key] = np.full(self._n_channels, np.inf)
            self._maxs[scope_key] = np.full(self._n_channels, -np.inf)
            self._all_values[scope_key] = []

        n_samples = signal.shape[1]
        self._counts[scope_key] += n_samples

        # Per-channel statistics
        ch_mean = signal.mean(axis=1)
        ch_var = signal.var(axis=1)

        # Welford update (batch)
        old_mean = self._means[scope_key].copy()
        self._means[scope_key] += (ch_mean - old_mean) * n_samples / self._counts[scope_key]
        self._m2s[scope_key] += ch_var * n_samples + (
            (ch_mean - old_mean) ** 2
        ) * n_samples * (self._counts[scope_key] - n_samples) / self._counts[scope_key]

        # Min/max
        ch_min = signal.min(axis=1)
        ch_max = signal.max(axis=1)
        self._mins[scope_key] = np.minimum(self._mins[scope_key], ch_min)
        self._maxs[scope_key] = np.maximum(self._maxs[scope_key], ch_max)

        # For robust: store per-channel data (flatten per channel)
        if self._method == NormalizationMethod.ROBUST:
            self._all_values[scope_key].append(signal.astype(np.float32))

    def finalize(self) -> NormalizationStats:
        """Compute final statistics from accumulated data."""
        stats = NormalizationStats()

        for scope_key in self._counts:
            count = self._counts[scope_key]
            mean = self._means[scope_key].astype(np.float32)
            var = (self._m2s[scope_key] / max(count, 1)).astype(np.float64)
            std = np.sqrt(np.maximum(var, 0)).astype(np.float32)

            median = None
            iqr = None

            if self._method == NormalizationMethod.ROBUST and self._all_values[scope_key]:
                # Concatenate all signals for this scope
                all_data = np.concatenate(self._all_values[scope_key], axis=1)
                median = np.median(all_data, axis=1).astype(np.float32)
                q75, q25 = np.percentile(all_data, [75, 25], axis=1)
                iqr = (q75 - q25).astype(np.float32)

            stats.set_stats(
                scope_key=scope_key,
                mean=mean,
                std=std,
                median=median,
                iqr=iqr,
                min_val=self._mins[scope_key].astype(np.float32),
                max_val=self._maxs[scope_key].astype(np.float32),
            )

        return stats
