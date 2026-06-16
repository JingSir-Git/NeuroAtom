"""Generic importer: import EEG from numpy arrays or CSV files.

Provides a low-barrier entry for datasets without a dedicated importer.
Accepts pre-loaded numpy arrays or CSV/NPY/NPZ files with a simple
directory convention.

Supported layouts::

    data_dir/
    ├── config.yaml           # minimal task config (or use CLI args)
    ├── S01/
    │   ├── signal.npy        # (n_channels, n_samples) float32
    │   ├── labels.csv        # optional: onset_sample, label
    │   └── channels.txt      # optional: one channel name per line
    ├── S02/
    │   └── ...

Or flat CSV::

    data_dir/
    ├── config.yaml
    ├── S01.csv               # rows=samples, cols=channels (header row = names)
    ├── S01_labels.csv
    └── ...

Usage via API::

    from neuroatom.importers.generic import GenericImporter
    importer = GenericImporter(pool, task_config)
    results = importer.import_dataset(Path("data_dir"))

Or via CLI::

    neuroatom import-generic ./my_pool ./data_dir --dataset-id my_eeg
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.core.annotation import CategoricalAnnotation
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import AtomType, ChannelType
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.metadata_store import AtomJSONLWriter
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.storage import paths as P
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)


class GenericImporter(BaseImporter):
    """Import EEG from numpy/CSV files with minimal configuration.

    This importer is designed for quick onboarding of new datasets
    that don't yet have a dedicated importer. It handles:

    - ``.npy`` files: (n_channels, n_samples) or (n_samples, n_channels)
    - ``.npz`` files: key 'signal' or 'data' or first array
    - ``.csv`` files: rows=samples, columns=channels (header row)
    """

    @staticmethod
    def detect(path: Path) -> bool:
        """Detect generic importable directory."""
        if path.is_dir():
            for ext in ("*.npy", "*.npz", "*.csv"):
                if list(path.rglob(ext)):
                    return True
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        """Load signal from a single file.

        Returns (signal_array, metadata_dict).
        """
        signal, channels = _load_signal_file(path)
        declared_unit = self._task_config.signal_unit or "uV"
        return signal, {
            "declared_unit": declared_unit,
            "channel_names": channels,
        }

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        """Not used directly — see import_dataset."""
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        """Not used directly — see import_dataset."""
        return None

    def import_dataset(
        self,
        data_dir: Path,
        *,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        sampling_rate: Optional[float] = None,
        signal_unit: Optional[str] = None,
        epoch_seconds: Optional[float] = None,
    ) -> List[ImportResult]:
        """Import all subjects from a generic data directory.

        Args:
            data_dir: Root directory containing subject data.
            dataset_id: Override dataset ID from task config.
            dataset_name: Override dataset name.
            subjects: Only import these subjects. None = all.
            sampling_rate: Sampling rate in Hz (override config).
            signal_unit: Source signal unit (override config).
            epoch_seconds: If set, split continuous data into epochs.

        Returns:
            List of ImportResult, one per subject.
        """
        data_dir = Path(data_dir)
        ds_id = dataset_id or self._task_config.dataset_id
        ds_name = dataset_name or self._task_config.dataset_name
        sr = sampling_rate or self._task_config.data.get("custom", {}).get("sampling_rate", 256.0)
        src_unit = signal_unit or self._task_config.signal_unit or "uV"

        # Register dataset
        self._pool.register_dataset(DatasetMeta(
            dataset_id=ds_id, name=ds_name,
        ))

        results = []

        # Discover subjects
        subject_dirs = _discover_subjects(data_dir)
        if subjects:
            subject_dirs = {k: v for k, v in subject_dirs.items() if k in subjects}

        for subj_id, subj_source in sorted(subject_dirs.items()):
            try:
                result = self._import_subject(
                    ds_id, subj_id, subj_source, sr, src_unit, epoch_seconds,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to import subject %s: %s", subj_id, e)

        logger.info(
            "Generic import complete: %s — %d subjects, %d total atoms",
            ds_id, len(results), sum(r.n_atoms for r in results),
        )

        # Post-import: assess quality (best-effort)
        try:
            tier = self._pool.assess_quality(ds_id)
            if tier:
                logger.info("Quality tier for %s: %s", ds_id, tier)
        except Exception:
            pass

        return results

    def _import_subject(
        self,
        dataset_id: str,
        subject_id: str,
        source: Path,
        sampling_rate: float,
        signal_unit: str,
        epoch_seconds: Optional[float],
    ) -> ImportResult:
        """Import a single subject's data."""
        self._pool.register_subject(SubjectMeta(
            subject_id=subject_id, dataset_id=dataset_id,
        ))
        session_id = "ses-01"
        run_id = "run-01"
        self._pool.ensure_session(dataset_id, subject_id, session_id,
                                  sampling_rate=sampling_rate)
        self._pool.ensure_run(dataset_id, subject_id, session_id, run_id)

        # Load signal
        signal, channel_names = _load_subject_data(source)
        n_channels = signal.shape[0]

        if not channel_names:
            channel_names = [f"Ch{i}" for i in range(n_channels)]

        # Build channel infos
        channel_infos = [
            ChannelInfo(
                channel_id=ch,
                index=i,
                name=ch,
                standard_name=standardize_channel_name(ch),
                type=ChannelType.EEG,
                sampling_rate=sampling_rate,
            )
            for i, ch in enumerate(channel_names)
        ]

        # Load labels (optional)
        labels = _load_labels(source)

        # Convert to storage unit
        signal, storage_unit, orig_unit = convert_to_storage_unit(
            signal, source_unit=signal_unit,
            pool_config=self._pool.config,
        )

        # Create atoms
        atoms = []
        if epoch_seconds and not labels:
            # Split continuous into fixed-length epochs
            epoch_samples = int(epoch_seconds * sampling_rate)
            n_epochs = signal.shape[1] // epoch_samples
            for i in range(n_epochs):
                onset = i * epoch_samples
                seg = signal[:, onset:onset + epoch_samples]
                atom = _make_atom(
                    dataset_id, subject_id, session_id, run_id,
                    channel_names, n_channels, sampling_rate,
                    onset, epoch_samples, storage_unit, orig_unit,
                )
                atoms.append((atom, seg))
        elif labels:
            # One atom per labeled epoch
            for lbl in labels:
                onset = lbl["onset_sample"]
                dur = lbl.get("duration_samples", int(sampling_rate))
                seg = signal[:, onset:onset + dur]
                if seg.shape[1] < dur:
                    continue  # skip truncated
                atom = _make_atom(
                    dataset_id, subject_id, session_id, run_id,
                    channel_names, n_channels, sampling_rate,
                    onset, dur, storage_unit, orig_unit,
                    label=lbl.get("label"),
                )
                atoms.append((atom, seg))
        else:
            # Whole recording as one atom
            atom = _make_atom(
                dataset_id, subject_id, session_id, run_id,
                channel_names, n_channels, sampling_rate,
                0, signal.shape[1], storage_unit, orig_unit,
            )
            atoms.append((atom, signal))

        # Write atoms
        warnings = []
        max_shard_mb = self._pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)

        with ShardManager(
            pool_root=self._pool.root, dataset_id=dataset_id,
            subject_id=subject_id, session_id=session_id, run_id=run_id,
            max_shard_size_mb=max_shard_mb,
        ) as mgr:
            jsonl_path = P.atoms_jsonl_path(
                self._pool.root, dataset_id, subject_id, session_id, run_id,
            )
            stored_atoms = []
            with AtomJSONLWriter(jsonl_path) as writer:
                for atom, seg in atoms:
                    w = validate_signal(
                        seg, atom.atom_id,
                        config=self._pool.config.get("import", {}),
                        signal_unit=storage_unit,
                    )
                    warnings.extend(w)
                    ref = mgr.write_atom_signal(atom.atom_id, seg)
                    atom.signal_ref = ref
                    writer.write_atom(atom)
                    stored_atoms.append(atom)

        self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)

        from neuroatom.core.run import RunMeta
        run_meta = RunMeta(
            run_id=run_id, session_id=session_id,
            subject_id=subject_id, dataset_id=dataset_id,
            task_type=self._task_config.task_type,
            n_trials=len(stored_atoms),
        )

        logger.info(
            "Imported subject %s/%s: %d atoms", dataset_id, subject_id, len(stored_atoms),
        )

        return ImportResult(
            atoms=stored_atoms, run_meta=run_meta,
            channel_infos=channel_infos, warnings=warnings,
        )


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _load_signal_file(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load signal from a single file. Returns (channels, samples) array + names."""
    channels = []
    if path.suffix == ".npy":
        arr = np.load(str(path))
        arr = _ensure_channels_first(arr)
        return arr.astype(np.float32), channels

    elif path.suffix == ".npz":
        npz = np.load(str(path))
        for key in ("signal", "data", "eeg"):
            if key in npz:
                arr = _ensure_channels_first(npz[key])
                return arr.astype(np.float32), channels
        # Fallback: first array
        first_key = list(npz.keys())[0]
        arr = _ensure_channels_first(npz[first_key])
        return arr.astype(np.float32), channels

    elif path.suffix == ".csv":
        return _load_csv_signal(path)

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_csv_signal(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load signal from CSV: rows=samples, cols=channels."""
    import pandas as pd
    df = pd.read_csv(path)
    channels = list(df.columns)
    arr = df.values.T.astype(np.float32)  # (channels, samples)
    return arr, channels


def _ensure_channels_first(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (n_channels, n_samples).

    Heuristic: if dim 0 > dim 1, transpose (assumes more samples than channels).
    """
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def _discover_subjects(data_dir: Path) -> Dict[str, Path]:
    """Discover subjects from directory structure or files."""
    subjects = {}

    # Pattern 1: subdirectories
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            # Check for signal files inside
            has_signal = any(d.glob("signal.*")) or any(d.glob("*.npy")) or any(d.glob("*.csv"))
            if has_signal:
                subjects[d.name] = d

    # Pattern 2: flat files named like S01.npy, S01.csv
    if not subjects:
        for f in sorted(data_dir.iterdir()):
            if f.is_file() and f.suffix in (".npy", ".npz", ".csv"):
                if not f.stem.endswith("_labels"):
                    subjects[f.stem] = f

    return subjects


def _load_subject_data(source: Path) -> Tuple[np.ndarray, List[str]]:
    """Load signal + channel names for a subject."""
    if source.is_file():
        return _load_signal_file(source)

    # Directory: look for signal files
    for name in ("signal.npy", "signal.npz", "data.npy", "data.csv"):
        candidate = source / name
        if candidate.exists():
            signal, channels = _load_signal_file(candidate)
            # Check for channels.txt
            ch_file = source / "channels.txt"
            if ch_file.exists():
                channels = ch_file.read_text().strip().split("\n")
                channels = [c.strip() for c in channels if c.strip()]
            return signal, channels

    # Try any npy/csv
    for f in sorted(source.iterdir()):
        if f.suffix in (".npy", ".npz", ".csv") and "label" not in f.stem.lower():
            signal, channels = _load_signal_file(f)
            ch_file = source / "channels.txt"
            if ch_file.exists():
                channels = ch_file.read_text().strip().split("\n")
                channels = [c.strip() for c in channels if c.strip()]
            return signal, channels

    raise FileNotFoundError(f"No signal file found in {source}")


def _load_labels(source: Path) -> List[dict]:
    """Load optional labels file. Returns list of {onset_sample, label, ...}."""
    if source.is_file():
        label_path = source.parent / f"{source.stem}_labels.csv"
    else:
        label_path = source / "labels.csv"

    if not label_path.exists():
        return []

    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {}
            if "onset_sample" in row:
                entry["onset_sample"] = int(row["onset_sample"])
            elif "onset" in row:
                entry["onset_sample"] = int(row["onset"])
            else:
                continue
            if "duration_samples" in row:
                entry["duration_samples"] = int(row["duration_samples"])
            if "label" in row:
                entry["label"] = row["label"]
            labels.append(entry)

    return labels


def _make_atom(
    dataset_id: str,
    subject_id: str,
    session_id: str,
    run_id: str,
    channel_names: List[str],
    n_channels: int,
    sampling_rate: float,
    onset_sample: int,
    duration_samples: int,
    signal_unit: str,
    original_unit: Optional[str],
    label: Optional[str] = None,
) -> Atom:
    """Create an Atom with minimal metadata."""
    atom_id = compute_atom_id(
        dataset_id, subject_id, session_id, run_id, onset_sample,
    )

    annotations = []
    if label:
        annotations.append(CategoricalAnnotation(
            annotation_id=f"{atom_id}_label",
            name="class_label",
            value=label,
        ))

    return Atom(
        atom_id=atom_id,
        dataset_id=dataset_id,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        atom_type=AtomType.TRIAL if label else AtomType.CONTINUOUS_SEGMENT,
        signal_ref=SignalRef(
            file_path="", internal_path="", shape=(n_channels, duration_samples),
        ),
        temporal=TemporalInfo(
            onset_sample=onset_sample,
            duration_samples=duration_samples,
            onset_seconds=onset_sample / sampling_rate,
            duration_seconds=duration_samples / sampling_rate,
        ),
        channel_ids=channel_names[:n_channels],
        n_channels=n_channels,
        sampling_rate=sampling_rate,
        signal_unit=signal_unit,
        original_unit=original_unit,
        annotations=annotations,
    )


register_importer("generic", GenericImporter)
