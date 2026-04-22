"""MATLAB .mat Importer: handles .mat format EEG datasets.

Critical for BCI Competition datasets (IIa, IIb, III, etc.) which are
published in MATLAB format. Supports both legacy .mat v5 and HDF5-based
v7.3 files.

Key design points:
- signal_unit from task_config is REQUIRED because .mat files have no
  standardized unit metadata (unlike EDF/GDF).
- Channel names may be embedded in the .mat file or specified in the
  task_config.
- Events may be stored as a separate field or embedded in annotations.

Supported .mat structures:
1. BCI Competition style: 's' (signal), 'HDR.SampleRate' (sfreq),
   'HDR.EVENT' (events), 'HDR.Label' (channel names)
2. Generic: 'data'/'signal'/'X' (signal), 'fs'/'srate' (sfreq),
   'events'/'markers'/'y' (events/labels)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.core.channel import ChannelInfo
from neuroatom.core.enums import ChannelType
from neuroatom.importers.base import BaseImporter, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name

logger = logging.getLogger(__name__)

# Common .mat field names for signal data, sampling rate, and events
_SIGNAL_KEYS = ["s", "data", "signal", "X", "eeg", "EEG", "x", "raw_signal"]
_SRATE_KEYS = ["fs", "srate", "Fs", "sampling_rate", "SampleRate"]
_EVENT_KEYS = ["events", "markers", "y", "labels", "Y", "event"]
_CHANNEL_KEYS = ["ch_names", "channels", "channel_names", "Label", "ch_labels"]


class _MatData:
    """Lightweight wrapper around .mat data, mimicking MNE Raw API for
    compatibility with the atomizer extraction interface."""

    def __init__(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: List[str],
        events: Optional[np.ndarray] = None,
    ):
        self._data = data  # shape (n_channels, n_samples)
        self._sfreq = sfreq
        self._ch_names = ch_names
        self._events = events
        self._n_channels, self._n_samples = data.shape

        # MNE-like info dict (minimal, for compatibility)
        self.info = {
            "sfreq": sfreq,
            "ch_names": ch_names,
            "nchan": self._n_channels,
        }

    def get_data(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        """Return signal data slice, compatible with MNE Raw.get_data()."""
        if stop is None:
            stop = self._n_samples
        return self._data[:, start:stop].copy()

    @property
    def n_times(self) -> int:
        return self._n_samples

    @property
    def filenames(self) -> List[str]:
        return ["<mat_data>"]


def _load_mat_file(path: Path) -> Dict[str, Any]:
    """Load .mat file, handling both v5 and v7.3 formats."""
    import scipy.io as sio

    try:
        # Try scipy first (handles v5 and older v7)
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return mat
    except NotImplementedError:
        # v7.3 (HDF5-based) — use h5py
        pass

    import h5py
    mat = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            try:
                mat[key] = np.array(f[key])
            except Exception:
                mat[key] = f[key]
    return mat


def _find_key(mat: Dict, candidates: List[str]) -> Optional[str]:
    """Find the first matching key from a list of candidates."""
    for key in candidates:
        if key in mat:
            return key
    return None


def _extract_nested(mat: Dict, dotted_key: str) -> Optional[Any]:
    """Extract a value from a possibly nested struct via dotted key path.

    Example: 'HDR.SampleRate' navigates mat['HDR'].SampleRate
    """
    parts = dotted_key.split(".")
    obj = mat
    for part in parts:
        if isinstance(obj, dict):
            if part not in obj:
                return None
            obj = obj[part]
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


def _extract_signal(
    mat: Dict, task_config: TaskConfig
) -> Tuple[np.ndarray, str]:
    """Extract signal matrix from .mat data.

    Returns:
        (data, key_used) where data has shape (n_channels, n_samples).
    """
    # Try task_config hint first
    hint_key = task_config.data.get("mat_signal_key")
    if hint_key:
        val = _extract_nested(mat, hint_key)
        if val is not None:
            data = np.array(val, dtype=np.float64)
            # Ensure (n_channels, n_samples) — most .mat files store (n_samples, n_channels)
            if data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T
            return data, hint_key

    # Auto-detect from common keys
    key = _find_key(mat, _SIGNAL_KEYS)
    if key is None:
        raise ValueError(
            f"Could not find signal data in .mat file. "
            f"Tried keys: {_SIGNAL_KEYS}. "
            f"Available keys: {[k for k in mat.keys() if not k.startswith('__')]}. "
            f"Set 'mat_signal_key' in task config to specify explicitly."
        )

    data = np.array(mat[key], dtype=np.float64)
    if data.ndim == 2 and data.shape[0] > data.shape[1]:
        data = data.T
    return data, key


def _extract_srate(mat: Dict, task_config: TaskConfig) -> float:
    """Extract sampling rate from .mat data or task_config."""
    # Task config override
    srate = task_config.data.get("sampling_rate")
    if srate is not None:
        return float(srate)

    # Nested struct: HDR.SampleRate
    nested_keys = ["HDR.SampleRate", "HDR.Fs", "HDR.fs"]
    for nk in nested_keys:
        val = _extract_nested(mat, nk)
        if val is not None:
            return float(np.squeeze(val))

    # Flat keys
    key = _find_key(mat, _SRATE_KEYS)
    if key is not None:
        return float(np.squeeze(mat[key]))

    raise ValueError(
        "Could not find sampling rate in .mat file. "
        "Set 'sampling_rate' in task config."
    )


def _extract_events_from_mat(
    mat: Dict, task_config: TaskConfig, n_samples: int, sfreq: float
) -> Optional[np.ndarray]:
    """Extract events from .mat file.

    Returns:
        (n_events, 3) array [sample, 0, event_id] or None.
    """
    # Task config hint
    hint_event_key = task_config.data.get("mat_event_key")
    hint_label_key = task_config.data.get("mat_label_key")

    # Try BCI Competition HDR.EVENT structure
    hdr_event = _extract_nested(mat, "HDR.EVENT")
    if hdr_event is not None:
        try:
            typ = np.array(getattr(hdr_event, "TYP", getattr(hdr_event, "typ", None)))
            pos = np.array(getattr(hdr_event, "POS", getattr(hdr_event, "pos", None)))
            if typ is not None and pos is not None:
                typ = typ.flatten().astype(int)
                pos = pos.flatten().astype(int)
                events = np.zeros((len(typ), 3), dtype=int)
                events[:, 0] = pos
                events[:, 2] = typ
                return events
        except Exception as e:
            logger.warning("Could not parse HDR.EVENT: %s", e)

    # Try flat event/label arrays
    if hint_event_key:
        events_raw = _extract_nested(mat, hint_event_key)
        if events_raw is not None:
            return _parse_events_array(events_raw, n_samples)

    if hint_label_key:
        labels_raw = _extract_nested(mat, hint_label_key)
        if labels_raw is not None:
            return _labels_to_events(labels_raw, n_samples, sfreq, task_config)

    # Auto-detect
    key = _find_key(mat, _EVENT_KEYS)
    if key is not None:
        raw_data = mat[key]
        arr = np.array(raw_data).flatten()
        # If it's per-trial labels (length << n_samples), convert
        if len(arr) < n_samples // 10:
            return _labels_to_events(arr, n_samples, sfreq, task_config)
        else:
            return _parse_events_array(arr, n_samples)

    return None


def _parse_events_array(raw: Any, n_samples: int) -> np.ndarray:
    """Parse raw event array into (n, 3) format."""
    arr = np.array(raw)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3].astype(int)
    if arr.ndim == 2 and arr.shape[1] == 2:
        events = np.zeros((arr.shape[0], 3), dtype=int)
        events[:, 0] = arr[:, 0].astype(int)
        events[:, 2] = arr[:, 1].astype(int)
        return events
    # 1D: treat as per-sample event codes
    arr = arr.flatten().astype(int)
    changes = np.where(np.diff(arr) != 0)[0] + 1
    if len(changes) == 0:
        return np.zeros((0, 3), dtype=int)
    events = np.zeros((len(changes), 3), dtype=int)
    events[:, 0] = changes
    events[:, 2] = arr[changes]
    return events[events[:, 2] > 0]


def _labels_to_events(
    labels: np.ndarray, n_samples: int, sfreq: float, task_config: TaskConfig
) -> np.ndarray:
    """Convert per-trial label array into event array.

    Uses trial_definition from task_config to compute event sample positions.
    """
    labels = np.array(labels).flatten().astype(int)
    n_trials = len(labels)

    tdef = task_config.trial_definition
    trial_dur = tdef.get("tmax", 4.0) - tdef.get("tmin", 0.0)
    trial_samples = int(trial_dur * sfreq)

    # Estimate event positions: evenly spaced
    events = np.zeros((n_trials, 3), dtype=int)
    for i in range(n_trials):
        events[i, 0] = int(i * trial_samples)
        events[i, 2] = labels[i]

    return events


def _extract_ch_names(
    mat: Dict, task_config: TaskConfig, n_channels: int
) -> List[str]:
    """Extract or generate channel names."""
    # Task config override
    ch_names = task_config.data.get("channel_names")
    if ch_names and len(ch_names) == n_channels:
        return list(ch_names)

    # HDR.Label
    hdr_label = _extract_nested(mat, "HDR.Label")
    if hdr_label is not None:
        try:
            if hasattr(hdr_label, "__iter__"):
                names = [str(lbl).strip() for lbl in hdr_label]
                if len(names) == n_channels:
                    return names
        except Exception:
            pass

    # Flat keys
    key = _find_key(mat, _CHANNEL_KEYS)
    if key is not None:
        try:
            raw_names = mat[key]
            if hasattr(raw_names, "__iter__"):
                names = [str(n).strip() for n in raw_names]
                if len(names) == n_channels:
                    return names
        except Exception:
            pass

    # Fallback: generate generic names
    logger.warning(
        "Could not extract channel names from .mat file. Using generic Ch_N naming."
    )
    return [f"Ch_{i + 1}" for i in range(n_channels)]


class MATImporter(BaseImporter):
    """Importer for MATLAB .mat EEG files.

    Handles BCI Competition datasets and generic .mat formats.
    Requires signal_unit to be declared in task_config since .mat
    files have no standardized unit metadata.
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        return path.is_file() and path.suffix.lower() == ".mat"

    def load_raw(self, path: Path) -> Tuple[_MatData, Dict[str, Any]]:
        """Load .mat file and construct a _MatData wrapper."""
        path = Path(path)
        logger.info("Loading .mat file: %s", path)

        mat = _load_mat_file(path)

        # Extract signal
        data, signal_key = _extract_signal(mat, self.task_config)
        n_channels, n_samples = data.shape

        # Extract sampling rate
        sfreq = _extract_srate(mat, self.task_config)

        # Extract channel names
        ch_names = _extract_ch_names(mat, self.task_config, n_channels)

        # Extract events
        events = _extract_events_from_mat(mat, self.task_config, n_samples, sfreq)

        # Build wrapper
        raw = _MatData(
            data=data.astype(np.float64),
            sfreq=sfreq,
            ch_names=ch_names,
            events=events,
        )

        # Signal unit (REQUIRED for .mat)
        declared_unit = self.task_config.signal_unit
        if declared_unit is None:
            logger.warning(
                "No signal_unit declared in task config for .mat import. "
                "Assuming µV. Set 'signal_unit' in your task config YAML."
            )
            declared_unit = "uV"

        extra_meta = {
            "declared_unit": declared_unit,
            "signal_key": signal_key,
            "original_shape": data.shape,
            "mat_keys": [k for k in mat.keys() if not k.startswith("__")],
        }

        return raw, extra_meta

    def extract_channel_infos(self, raw: _MatData) -> List[ChannelInfo]:
        """Extract channel info from _MatData wrapper."""
        ch_infos = []
        type_overrides = self.task_config.channel_type_overrides
        exclude_set = set(self.task_config.exclude_channels)
        unit = self.task_config.signal_unit or "uV"

        for idx, ch_name in enumerate(raw.info["ch_names"]):
            if ch_name in exclude_set:
                continue

            if ch_name in type_overrides:
                ch_type = ChannelType(type_overrides[ch_name])
            else:
                ch_type = ChannelType.EEG  # Default for .mat

            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{idx:03d}",
                index=idx,
                name=ch_name,
                standard_name=standardize_channel_name(ch_name),
                type=ch_type,
                unit=unit,
                sampling_rate=raw.info["sfreq"],
            ))

        return ch_infos

    def extract_events(self, raw: _MatData) -> Optional[np.ndarray]:
        """Return events stored during load_raw."""
        return raw._events


# Auto-register
register_importer("mat", MATImporter)
