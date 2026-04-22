"""PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb) Importer.

Handles the PhysioNet eegmmidb dataset: 109 subjects, 14 EDF runs each,
64 channels @ 160 Hz, with run-dependent event semantics for T1/T2 codes.

Key features:
    - Resolves T1/T2 annotation codes to correct class labels per run number
    - Supports selective import by task type (imagery, execution, or both)
    - Cleans channel names (strips trailing dots from EDF labels)
    - Handles baseline runs (R01 eyes-open, R02 eyes-closed) as rest epochs
    - Validates cross-subject consistency

Data layout expected:
    <dataset_root>/
        S001/
            S001R01.edf ... S001R14.edf
        S002/
            ...
        S109/
            S109R01.edf ... S109R14.edf
"""

import importlib.resources
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the PhysioNet MI importer")

from neuroatom.core.annotation import CategoricalAnnotation, NumericAnnotation
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import AtomType, ChannelStatus, ChannelType
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.quality import QualityInfo
from neuroatom.core.signal_ref import SignalRef
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run-to-task mapping
# ---------------------------------------------------------------------------

# Maps run number → (task_name, T1_label, T2_label, paradigm)
_RUN_TASK_MAP: Dict[int, Tuple[str, str, str, str]] = {
    3:  ("task1_execution", "left_fist",  "right_fist",  "execution"),
    7:  ("task1_execution", "left_fist",  "right_fist",  "execution"),
    11: ("task1_execution", "left_fist",  "right_fist",  "execution"),
    4:  ("task2_imagery",   "left_hand",  "right_hand",  "imagery"),
    8:  ("task2_imagery",   "left_hand",  "right_hand",  "imagery"),
    12: ("task2_imagery",   "left_hand",  "right_hand",  "imagery"),
    5:  ("task3_execution", "both_fists", "both_feet",   "execution"),
    9:  ("task3_execution", "both_fists", "both_feet",   "execution"),
    13: ("task3_execution", "both_fists", "both_feet",   "execution"),
    6:  ("task4_imagery",   "both_hands", "both_feet",   "imagery"),
    10: ("task4_imagery",   "both_hands", "both_feet",   "imagery"),
    14: ("task4_imagery",   "both_hands", "both_feet",   "imagery"),
}

_BASELINE_RUNS: Dict[int, str] = {
    1: "eyes_open",
    2: "eyes_closed",
}


def _clean_channel_name(raw_name: str) -> str:
    """Strip trailing dots from PhysioNet EDF channel names.

    Examples: 'Fc5.' → 'Fc5', 'C3..' → 'C3', 'T10.' → 'T10'
    """
    return raw_name.rstrip(".")


def _load_standard_1020_coords() -> Dict[str, Dict[str, float]]:
    """Load standard 10-20 electrode positions from package config."""
    try:
        ref = importlib.resources.files("neuroatom.configs").joinpath("standard_1020.json")
        with importlib.resources.as_file(ref) as p:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        return data.get("electrodes", {})
    except Exception:
        return {}

_STD_COORDS: Optional[Dict[str, Dict[str, float]]] = None

def _get_standard_coords() -> Dict[str, Dict[str, float]]:
    """Lazy-load standard electrode coordinates."""
    global _STD_COORDS
    if _STD_COORDS is None:
        _STD_COORDS = _load_standard_1020_coords()
    return _STD_COORDS


def _build_channel_infos(
    raw_ch_names: List[str],
    srate: float,
) -> List[ChannelInfo]:
    """Build ChannelInfo list from MNE raw channel names."""
    std_coords = _get_standard_coords()
    ch_infos = []
    for idx, raw_name in enumerate(raw_ch_names):
        clean_name = _clean_channel_name(raw_name)
        std_name = standardize_channel_name(clean_name)

        # Look up electrode coordinates (case-insensitive)
        location = None
        lookup_key = clean_name.capitalize() if clean_name[0].isupper() else clean_name
        # Try exact match first, then capitalized variants
        for try_name in [clean_name, clean_name.capitalize(),
                         clean_name.upper(), std_name or ""]:
            if try_name in std_coords:
                pos = std_coords[try_name]
                location = ElectrodeLocation(
                    x=pos["x"], y=pos["y"], z=pos["z"],
                    coordinate_system="MNI",
                    coordinate_units="m",
                )
                break

        ch_infos.append(ChannelInfo(
            channel_id=f"eeg_{idx:03d}",
            index=idx,
            name=clean_name,
            standard_name=std_name,
            type=ChannelType.EEG,
            unit="V",  # MNE reads EDF in Volts
            sampling_rate=srate,
            status=ChannelStatus.GOOD,
            location=location,
        ))
    return ch_infos


# ---------------------------------------------------------------------------
# PhysioNet MI Importer
# ---------------------------------------------------------------------------

class PhysioNetMIImporter(BaseImporter):
    """Importer for the PhysioNet EEG Motor Movement/Imagery Dataset.

    Supports:
        - Selective import by paradigm ('imagery', 'execution', or 'all')
        - Run-dependent T1/T2 → class label resolution
        - Baseline run import as rest epochs
        - Per-subject directory auto-discovery
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if path is a PhysioNet eegmmidb dataset."""
        path = Path(path)
        if not path.is_dir():
            return False

        # Look for S001/S001R01.edf pattern
        sub_dirs = [d for d in path.iterdir() if d.is_dir() and re.match(r"S\d{3}$", d.name)]
        if not sub_dirs:
            return False

        sample_sub = sub_dirs[0]
        edfs = list(sample_sub.glob(f"{sample_sub.name}R*.edf"))
        return len(edfs) >= 14

    def load_raw(self, path):
        raise NotImplementedError(
            "PhysioNetMIImporter.load_raw() is not used directly. "
            "Use import_subject() for per-subject import."
        )

    def extract_channel_infos(self, raw):
        raise NotImplementedError("Use _build_channel_infos() instead.")

    def extract_events(self, raw):
        return None

    # ------------------------------------------------------------------
    # Import one run
    # ------------------------------------------------------------------

    def _import_run(
        self,
        edf_path: Path,
        run_num: int,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        paradigm_filter: Optional[str] = None,
        include_rest: bool = False,
    ) -> Tuple[List[Atom], List[ChannelInfo], List[str]]:
        """Import a single EDF run, extracting task epochs as atoms.

        Args:
            edf_path: Path to the EDF file.
            run_num: 1-based run number (1-14).
            dataset_id: Dataset identifier.
            subject_id: Subject identifier.
            session_id: Session identifier.
            paradigm_filter: 'imagery', 'execution', or None (import all).
            include_rest: Whether to include T0 (rest) epochs.

        Returns:
            (atoms, channel_infos, warnings)
        """
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P

        # Baseline runs (R01=eyes_open, R02=eyes_closed): import as continuous segments
        if run_num in _BASELINE_RUNS:
            baseline_type = _BASELINE_RUNS[run_num]
            run_id = f"run-{run_num:02d}"

            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            srate = raw.info["sfreq"]
            ch_infos = _build_channel_infos(raw.ch_names, srate)
            channel_ids = [ch.channel_id for ch in ch_infos]
            data = raw.get_data()  # (n_channels, n_samples)

            self.pool.ensure_session(dataset_id, subject_id, session_id, sampling_rate=srate)
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            atom_id = compute_atom_id(
                dataset_id=dataset_id, subject_id=subject_id,
                session_id=session_id, run_id=run_id, onset_sample=0,
            )
            n_samples = data.shape[1]
            atom = Atom(
                atom_id=atom_id,
                atom_type=AtomType.CONTINUOUS_SEGMENT,
                dataset_id=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                trial_index=None,
                signal_ref=SignalRef(
                    file_path="__placeholder__",
                    internal_path=f"/atoms/{atom_id}/signal",
                    shape=(len(channel_ids), n_samples),
                ),
                temporal=TemporalInfo(
                    onset_sample=0, onset_seconds=0.0,
                    duration_samples=n_samples,
                    duration_seconds=n_samples / srate,
                ),
                channel_ids=channel_ids,
                n_channels=len(channel_ids),
                sampling_rate=srate,
                annotations=[
                    CategoricalAnnotation(
                        annotation_id=f"ann_baseline_{run_id}",
                        name="baseline_type",
                        value=baseline_type,
                    ),
                    CategoricalAnnotation(
                        annotation_id=f"ann_paradigm_{run_id}",
                        name="paradigm",
                        value="rest",
                    ),
                ],
                quality=QualityInfo(overall_status="good"),
                processing_history=ProcessingHistory(
                    steps=[ProcessingStep(
                        operation="raw_import",
                        parameters={
                            "format": "physionet_edf",
                            "source_file": edf_path.name,
                            "run_num": run_num,
                            "task": "baseline",
                            "baseline_type": baseline_type,
                            "signal_unit": "V",
                        },
                    )],
                    is_raw=True, version_tag="raw",
                ),
                custom_fields={
                    "run_num": run_num,
                    "paradigm": "rest",
                    "baseline_type": baseline_type,
                },
            )

            max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
            compression = self.pool.config.get("storage", {}).get("compression", "gzip")

            with ShardManager(
                pool_root=self.pool.root, dataset_id=dataset_id,
                subject_id=subject_id, session_id=session_id, run_id=run_id,
                max_shard_size_mb=max_shard_mb, compression=compression,
            ) as shard_mgr:
                signal_ref = shard_mgr.write_atom_signal(atom_id, data)
                atom.signal_ref = signal_ref

            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, dataset_id, subject_id, session_id, run_id
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                writer.write_atom(atom)

            logger.info(
                "Imported %s R%02d (baseline/%s): %d ch × %d samples (%.1fs)",
                subject_id, run_num, baseline_type,
                len(channel_ids), n_samples, n_samples / srate,
            )
            warnings = validate_signal(
                signal=data.astype(np.float32),
                atom_id=atom_id,
                config=self.pool.config.get("import", {}),
            )
            return [atom], ch_infos, warnings

        # Check if this run matches the paradigm filter
        if run_num not in _RUN_TASK_MAP:
            logger.warning("Unknown run number %d, skipping.", run_num)
            return [], [], []

        task_name, t1_label, t2_label, paradigm = _RUN_TASK_MAP[run_num]

        if paradigm_filter and paradigm != paradigm_filter:
            logger.debug("Skipping run R%02d (paradigm=%s, filter=%s)", run_num, paradigm, paradigm_filter)
            return [], [], []

        run_id = f"run-{run_num:02d}"

        # Load EDF
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        srate = raw.info["sfreq"]
        n_channels = len(raw.ch_names)

        # Build channel info
        ch_infos = _build_channel_infos(raw.ch_names, srate)
        channel_ids = [ch.channel_id for ch in ch_infos]

        # Extract events from annotations
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Determine epoch window
        tmin = self.task_config.data.get("epoch_tmin", 0.0)
        tmax = self.task_config.data.get("epoch_tmax", 4.0)
        n_epoch_samples = int((tmax - tmin) * srate) + 1

        # Build label map for this run
        # event_id maps annotation string → integer code
        label_map = {}
        if "T1" in event_id:
            label_map[event_id["T1"]] = t1_label
        if "T2" in event_id:
            label_map[event_id["T2"]] = t2_label
        if include_rest and "T0" in event_id:
            label_map[event_id["T0"]] = "rest"

        # Filter events to only task-relevant ones
        task_events = events[np.isin(events[:, 2], list(label_map.keys()))]

        if len(task_events) == 0:
            logger.warning("No task events in run R%02d", run_num)
            return [], [], []

        # Get full signal data (channels × samples)
        data = raw.get_data()  # (n_channels, n_samples)

        # Ensure pool hierarchy
        self.pool.ensure_session(dataset_id, subject_id, session_id, sampling_rate=srate)
        self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

        quality = QualityInfo(overall_status="good")
        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        atoms = []
        all_warnings = []

        with ShardManager(
            pool_root=self.pool.root,
            dataset_id=dataset_id,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            max_shard_size_mb=max_shard_mb,
            compression=compression,
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, dataset_id, subject_id, session_id, run_id
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                for epoch_idx, event in enumerate(task_events):
                    onset_sample = event[0]
                    event_code = event[2]
                    class_label = label_map[event_code]

                    # Extract epoch
                    end_sample = onset_sample + n_epoch_samples
                    if end_sample > data.shape[1]:
                        logger.warning(
                            "Epoch %d extends beyond data (onset=%d, need=%d, have=%d). Skipping.",
                            epoch_idx, onset_sample, end_sample, data.shape[1],
                        )
                        continue

                    signal = data[:, onset_sample:end_sample]  # (n_channels, n_epoch_samples)

                    # Annotations
                    annotations = [
                        CategoricalAnnotation(
                            annotation_id=f"ann_class_{run_id}_{epoch_idx:04d}",
                            name="mi_class" if paradigm == "imagery" else "motor_class",
                            value=class_label,
                        ),
                        CategoricalAnnotation(
                            annotation_id=f"ann_paradigm_{run_id}_{epoch_idx:04d}",
                            name="paradigm",
                            value=paradigm,
                        ),
                        CategoricalAnnotation(
                            annotation_id=f"ann_task_{run_id}_{epoch_idx:04d}",
                            name="task",
                            value=task_name,
                        ),
                    ]

                    atom_id = compute_atom_id(
                        dataset_id=dataset_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        run_id=run_id,
                        onset_sample=onset_sample,
                    )

                    temporal = TemporalInfo(
                        onset_sample=onset_sample,
                        onset_seconds=onset_sample / srate,
                        duration_samples=n_epoch_samples,
                        duration_seconds=n_epoch_samples / srate,
                    )

                    atom = Atom(
                        atom_id=atom_id,
                        atom_type=AtomType.EVENT_EPOCH,
                        dataset_id=dataset_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        run_id=run_id,
                        trial_index=epoch_idx,
                        signal_ref=SignalRef(
                            file_path="__placeholder__",
                            internal_path=f"/atoms/{atom_id}/signal",
                            shape=(n_channels, n_epoch_samples),
                        ),
                        temporal=temporal,
                        channel_ids=channel_ids,
                        n_channels=n_channels,
                        sampling_rate=srate,
                        annotations=annotations,
                        quality=quality,
                        processing_history=ProcessingHistory(
                            steps=[
                                ProcessingStep(
                                    operation="raw_import",
                                    parameters={
                                        "format": "physionet_edf",
                                        "source_file": edf_path.name,
                                        "run_num": run_num,
                                        "task": task_name,
                                        "paradigm": paradigm,
                                        "class_label": class_label,
                                        "epoch_tmin": tmin,
                                        "epoch_tmax": tmax,
                                        "signal_unit": "V",
                                    },
                                ),
                            ],
                            is_raw=True,
                            version_tag="raw",
                        ),
                        custom_fields={
                            "run_num": run_num,
                            "paradigm": paradigm,
                            "task": task_name,
                        },
                    )

                    # Validate
                    warnings = validate_signal(
                        signal=signal.astype(np.float32),
                        atom_id=atom_id,
                        config=self.pool.config.get("import", {}),
                    )
                    all_warnings.extend(warnings)

                    # Write signal
                    signal_ref = shard_mgr.write_atom_signal(atom_id, signal)
                    atom.signal_ref = signal_ref

                    writer.write_atom(atom)
                    atoms.append(atom)

        logger.info(
            "Imported %s R%02d (%s/%s): %d epochs × %d ch @ %.0f Hz "
            "(T1=%s, T2=%s)",
            subject_id, run_num, paradigm, task_name,
            len(atoms), n_channels, srate,
            t1_label, t2_label,
        )

        return atoms, ch_infos, all_warnings

    # ------------------------------------------------------------------
    # Main entry: import one subject
    # ------------------------------------------------------------------

    def import_subject(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str = "ses-01",
        paradigm: Optional[str] = None,
        max_runs: Optional[int] = None,
        include_rest: bool = False,
    ) -> List[ImportResult]:
        """Import all task runs for a subject.

        Args:
            subject_dir: Path to subject directory (e.g., <dataset_root>/S001)
            subject_id: Subject identifier (e.g., 'S001')
            session_id: Session identifier
            paradigm: 'imagery', 'execution', or None for all
            max_runs: Maximum number of runs to import (for testing)
            include_rest: Include T0 (rest) epochs

        Returns:
            List of ImportResult, one per imported run
        """
        subject_dir = Path(subject_dir)
        dataset_id = self.task_config.dataset_id

        self.pool.ensure_dataset(dataset_id, name=self.task_config.dataset_name)
        self.pool.ensure_subject(dataset_id, subject_id)

        # Discover EDF files
        edfs = sorted(subject_dir.glob(f"{subject_dir.name}R*.edf"))
        if not edfs:
            # Also try case-insensitive
            edfs = sorted(subject_dir.glob("*R*.edf"))

        if not edfs:
            logger.error("No EDF files found in %s", subject_dir)
            return []

        # Extract run numbers
        run_files: List[Tuple[int, Path]] = []
        for edf in edfs:
            m = re.search(r"R(\d+)\.edf$", edf.name, re.IGNORECASE)
            if m:
                run_files.append((int(m.group(1)), edf))

        run_files.sort()

        if max_runs is not None:
            # Keep baseline runs separate, limit only task runs
            baseline_files = [(rn, fp) for rn, fp in run_files if rn in _BASELINE_RUNS]
            task_filtered = []
            for rn, fp in run_files:
                if rn in _BASELINE_RUNS:
                    continue
                if rn not in _RUN_TASK_MAP:
                    continue
                _, _, _, p = _RUN_TASK_MAP[rn]
                if paradigm and p != paradigm:
                    continue
                task_filtered.append((rn, fp))
            run_files = baseline_files + task_filtered[:max_runs]

        results = []
        for run_num, edf_path in run_files:
            atoms, ch_infos, warnings = self._import_run(
                edf_path=edf_path,
                run_num=run_num,
                dataset_id=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                paradigm_filter=paradigm,
                include_rest=include_rest,
            )

            if atoms:
                run_meta = self._make_run_meta(
                    run_id=f"run-{run_num:02d}",
                    session_id=session_id,
                    subject_id=subject_id,
                    dataset_id=dataset_id,
                    run_num=run_num,
                    n_atoms=len(atoms),
                )
                self.pool.register_run(run_meta)
                results.append(ImportResult(
                    atoms=atoms,
                    run_meta=run_meta,
                    channel_infos=ch_infos,
                    warnings=warnings,
                ))

        total_atoms = sum(len(r.atoms) for r in results)
        logger.info(
            "Subject %s: imported %d runs, %d total atoms (paradigm=%s).",
            subject_id, len(results), total_atoms, paradigm or "all",
        )

        return results

    def _make_run_meta(self, run_id, session_id, subject_id, dataset_id, run_num, n_atoms):
        from neuroatom.core.run import RunMeta
        return RunMeta(
            run_id=run_id,
            session_id=session_id,
            subject_id=subject_id,
            dataset_id=dataset_id,
            run_index=run_num,
            task_type=self.task_config.task_type,
            n_trials=n_atoms,
        )

    # ------------------------------------------------------------------
    # Multi-subject import
    # ------------------------------------------------------------------

    def import_dataset(
        self,
        dataset_dir: Path,
        paradigm: Optional[str] = None,
        max_subjects: Optional[int] = None,
        max_runs_per_subject: Optional[int] = None,
        include_rest: bool = False,
    ) -> List[ImportResult]:
        """Import multiple subjects from the dataset directory.

        Args:
            dataset_dir: Root directory containing S001/ ... S109/ subdirectories
            paradigm: 'imagery', 'execution', or None for all
            max_subjects: Maximum number of subjects to import
            max_runs_per_subject: Maximum runs per subject
            include_rest: Include T0 rest epochs

        Returns:
            Aggregated list of ImportResult across all subjects
        """
        dataset_dir = Path(dataset_dir)
        sub_dirs = sorted([
            d for d in dataset_dir.iterdir()
            if d.is_dir() and re.match(r"S\d{3}$", d.name)
        ])

        if max_subjects is not None:
            sub_dirs = sub_dirs[:max_subjects]

        all_results = []
        for sub_dir in sub_dirs:
            subject_id = sub_dir.name
            results = self.import_subject(
                subject_dir=sub_dir,
                subject_id=subject_id,
                paradigm=paradigm,
                max_runs=max_runs_per_subject,
                include_rest=include_rest,
            )
            all_results.extend(results)

        logger.info(
            "Dataset: imported %d subjects, %d runs, %d total atoms.",
            len(sub_dirs), len(all_results),
            sum(len(r.atoms) for r in all_results),
        )

        return all_results


# Auto-register
register_importer("physionet_mi", PhysioNetMIImporter)
