"""OpenBMI Importer: Motor Imagery, ERP (P300), and SSVEP paradigms.

Lee et al., "EEG dataset and OpenBMI toolbox for three BCI paradigms:
an investigation into BCI illiteracy", GigaScience 2019.

54 subjects × 2 sessions, 62 EEG channels @ 1000 Hz, actiCAP (Brain Products).
Data provided as pre-segmented MATLAB v5 .mat files (scipy-readable).

Supported paradigms:
    MI    — 2-class motor imagery (right_hand / left_hand), 4s epochs, 100 trials
    ERP   — P300 oddball (target / nontarget), 0.8s epochs, ~1980 trials
    SSVEP — 4-class SSVEP (up / left / right / down), 4s epochs, 100 trials

.mat structure (same layout for all paradigms):
    EEG_{PARADIGM}_{split}: struct with
        smt:    (epoch_samples, n_trials, 62) float64 — pre-segmented trials
        x:      (N, 62) float64                       — continuous EEG
        t:      (1, n_trials) int32                   — trial onset indices in x
        fs:     uint16                                 — sampling rate (1000)
        y_dec:  (1, n_trials) uint8                   — numeric class labels
        y_class: (1, n_trials) object                  — string class labels
        class:  (n_classes, 2) object                  — class mapping table
        chan:   (1, 62) object                          — channel name strings
        ival:   (1, epoch_samples) uint16              — sample indices in epoch
        EMG:    (N, 4) float64                         — EMG channels
        pre_rest:  (60000, 62)                         — 60s pre-experiment rest
        post_rest: (60000, 62)                         — 60s post-experiment rest

Directory layout:
    OpenBMI/
    ├── MI/     sess{01,02}_subj{01-54}_EEG_MI.mat
    ├── ERP/    sess{01,02}_subj{01-54}_EEG_ERP.mat
    ├── SSVEP/  sess{01,02}_subj{01-54}_EEG_SSVEP.mat
    └── Artifact/  (not imported)
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

from neuroatom.core.annotation import CategoricalAnnotation, NumericAnnotation
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import (
    AtomType,
    ChannelStatus,
    ChannelType,
    QualityStatus,
)
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.quality import QualityInfo
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

# ── Standard montage for electrode coordinates ─────────────────────────
_STD_1005_POS: Optional[Dict[str, Tuple[float, float, float]]] = None


def _get_standard_1005_positions() -> Dict[str, Tuple[float, float, float]]:
    """Lazy-load standard 10-05 electrode positions from MNE.

    Returns dict mapping channel name → (x, y, z) in MNI head coords (metres).
    """
    global _STD_1005_POS
    if _STD_1005_POS is None:
        try:
            import mne
            montage = mne.channels.make_standard_montage("standard_1005")
            ch_pos = montage.get_positions()["ch_pos"]
            _STD_1005_POS = {
                name: (float(pos[0]), float(pos[1]), float(pos[2]))
                for name, pos in ch_pos.items()
            }
        except Exception:
            logger.debug("Could not load standard_1005 montage from MNE.")
            _STD_1005_POS = {}
    return _STD_1005_POS


# Paradigm key → .mat variable name prefixes
_PARADIGM_KEYS = {
    "MI": ("EEG_MI_train", "EEG_MI_test"),
    "ERP": ("EEG_ERP_train", "EEG_ERP_test"),
    "SSVEP": ("EEG_SSVEP_train", "EEG_SSVEP_test"),
}

# Filename pattern: sess{NN}_subj{NN}_EEG_{PARADIGM}.mat
_FILENAME_RE = re.compile(
    r"sess(\d{2})_subj(\d{2})_EEG_(MI|ERP|SSVEP)\.mat",
    re.IGNORECASE,
)


def _parse_questionnaire(csv_path: Path) -> Dict[int, Dict[str, Any]]:
    """Parse OpenBMI Questionnaire_results.csv into per-subject demographics.

    The CSV is transposed: rows are questionnaire items, columns are subjects.
    Returns ``{subject_num: {age, sex, bci_experience, ...}}``.
    """
    if not csv_path.exists():
        return {}

    result: Dict[int, Dict[str, Any]] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        logger.warning("Failed to read questionnaire: %s", e)
        return {}

    # Discover subject columns
    sub_cols = [c for c in rows[0].keys() if c.startswith("subject")]
    for col in sub_cols:
        m = re.match(r"subject(\d+)", col)
        if not m:
            continue
        subj_num = int(m.group(1))
        info: Dict[str, Any] = {}

        for row in rows:
            q = row.get("Questionaire", "").strip()
            val = row.get(col, "").strip()
            if not q or not val:
                continue

            q_lower = q.lower()
            if "age" in q_lower and "number" in q_lower:
                try:
                    info["age"] = float(val)
                except ValueError:
                    pass
            elif "sex" in q_lower:
                # 0=male, 1=female
                info["sex"] = "F" if val == "1" else "M"
            elif "bci experience" in q_lower:
                try:
                    info["bci_experience"] = int(float(val))
                except ValueError:
                    pass
            elif "how long did you sleep" in q_lower:
                try:
                    info["sleep_level"] = int(val)
                except ValueError:
                    pass
            elif "drink coffee" in q_lower:
                try:
                    info["coffee_hours_before"] = float(val)
                except ValueError:
                    pass
            elif "drink alcohol" in q_lower:
                try:
                    info["alcohol_hours_before"] = float(val)
                except ValueError:
                    pass
            elif "smoke" in q_lower:
                try:
                    info["smoke_hours_before"] = float(val)
                except ValueError:
                    pass
            elif "time slot" in q_lower:
                slot_map = {"1": "09:00", "2": "12:00", "3": "15:00", "4": "18:00"}
                info["time_slot"] = slot_map.get(val, val)

        result[subj_num] = info

    return result


def _detect_openbmi(path: Path) -> bool:
    """Detect if a path contains OpenBMI .mat files."""
    path = Path(path)

    if path.is_file():
        return bool(_FILENAME_RE.match(path.name))

    if path.is_dir():
        # Check for paradigm subdirectories or direct .mat files
        for paradigm_dir in ("MI", "ERP", "SSVEP"):
            sub_path = path / paradigm_dir
            if sub_path.is_dir():
                mats = list(sub_path.glob("sess*_subj*_EEG_*.mat"))
                if mats:
                    return True
        # Or direct .mat files in directory
        mats = list(path.glob("sess*_subj*_EEG_*.mat"))
        return len(mats) > 0

    return False


def _parse_filename(name: str) -> Optional[Tuple[int, int, str]]:
    """Parse OpenBMI filename → (session_num, subject_num, paradigm).

    Returns None if filename doesn't match expected pattern.
    """
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3).upper()


class OpenBMIImporter(BaseImporter):
    """Importer for OpenBMI dataset (Lee et al., GigaScience 2019).

    Handles all three paradigms (MI, ERP, SSVEP) with a unified interface.
    Each .mat file contains pre-segmented epochs in the 'smt' field,
    along with class labels and channel information.

    Usage:
        # Import one subject, one paradigm
        importer.import_subject(
            mat_path=Path("MI/sess01_subj01_EEG_MI.mat"),
            subject_id="S01",
        )

        # Import all subjects for a paradigm
        importer.import_paradigm(
            data_dir=Path("OpenBMI"),
            paradigm="MI",
            max_subjects=5,
        )
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Detect OpenBMI dataset files or directory."""
        return _detect_openbmi(Path(path))

    def load_raw(self, path):
        raise NotImplementedError(
            "OpenBMIImporter.load_raw() is not used directly. "
            "Use import_subject() or import_paradigm() instead."
        )

    def extract_channel_infos(self, raw):
        raise NotImplementedError("Use _build_channel_infos() instead.")

    def extract_events(self, raw):
        return None

    # ------------------------------------------------------------------
    # Channel info builder
    # ------------------------------------------------------------------

    def _build_channel_infos(
        self,
        chan_names: List[str],
        srate: float,
    ) -> List[ChannelInfo]:
        """Build ChannelInfo list from .mat channel names.

        Unlike BCI IV 2a, OpenBMI .mat files store channel names directly.
        """
        unit = self.task_config.signal_unit or "uV"
        type_overrides = self.task_config.channel_type_overrides
        exclude_set = set(self.task_config.exclude_channels)

        ch_infos = []
        for idx, ch_name in enumerate(chan_names):
            if ch_name in exclude_set:
                continue

            if ch_name in type_overrides:
                ch_type = ChannelType(type_overrides[ch_name])
            elif ch_name.upper().startswith("EOG"):
                ch_type = ChannelType.EOG
            elif ch_name.upper().startswith("EMG"):
                ch_type = ChannelType.EMG
            else:
                ch_type = ChannelType.EEG

            std_name = (
                standardize_channel_name(ch_name)
                if ch_type == ChannelType.EEG
                else None
            )

            # Look up electrode coordinates from standard 10-05 montage
            location = None
            if ch_type == ChannelType.EEG:
                std_pos = _get_standard_1005_positions()
                # Try exact name, then case-insensitive lookup
                for try_name in [ch_name, std_name or ""]:
                    if try_name in std_pos:
                        x, y, z = std_pos[try_name]
                        location = ElectrodeLocation(
                            x=x, y=y, z=z,
                            coordinate_system="MNI",
                            coordinate_units="m",
                        )
                        break

            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{idx:03d}",
                index=idx,
                name=ch_name,
                standard_name=std_name,
                type=ch_type,
                unit=unit,
                sampling_rate=srate,
                location=location,
                reference="FCz",
                status=ChannelStatus.GOOD,
                custom_fields=(
                    {"coordinate_source": "standard_1005_montage"}
                    if location is not None else {}
                ),
            ))

        return ch_infos

    # ------------------------------------------------------------------
    # Core: extract epochs from one split (train or test)
    # ------------------------------------------------------------------

    def _extract_split(
        self,
        struct: Any,
        paradigm: str,
        split_name: str,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
        class_labels: Dict[int, str],
        mat_filename: str,
        max_trials: Optional[int] = None,
    ) -> Tuple[List[Atom], List[ChannelInfo], List[str]]:
        """Extract all epochs from one train/test split struct.

        Args:
            struct: The MATLAB struct (e.g., mat['EEG_MI_train'][0,0]).
            paradigm: 'MI', 'ERP', or 'SSVEP'.
            split_name: 'train' or 'test'.
            Other args: pool hierarchy identifiers.

        Returns:
            (atoms, channel_infos, warnings)
        """
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P

        # ── Parse struct fields ──────────────────────────────────────────
        from neuroatom.utils.validation import validate_sampling_rate
        smt = struct["smt"]       # (epoch_samples, n_trials, n_channels)
        fs = int(struct["fs"].ravel()[0])
        validate_sampling_rate(float(fs), f"OpenBMI {paradigm} {split_name}")
        y_dec = struct["y_dec"].ravel().astype(int)
        n_epoch_samples, n_trials, n_channels = smt.shape

        # Channel names from .mat — handle both (1, N) and (N,) shapes
        chan_raw = struct["chan"]
        if chan_raw.ndim == 2:
            chan_names = [
                str(chan_raw[0, i][0]) for i in range(chan_raw.shape[1])
            ]
        elif chan_raw.ndim == 1:
            chan_names = [str(chan_raw[i]) for i in range(chan_raw.shape[0])]
        else:
            chan_names = [str(chan_raw.flat[i]) for i in range(chan_raw.size)]

        # Trial onsets in continuous recording (if available)
        t_onsets = struct["t"].ravel().astype(int) if "t" in struct.dtype.names else None

        if max_trials is not None:
            n_trials = min(n_trials, max_trials)

        epoch_duration_s = n_epoch_samples / fs

        # Build channel infos
        ch_infos = self._build_channel_infos(chan_names, float(fs))
        channel_ids = [ch.channel_id for ch in ch_infos]

        # Annotation name varies by paradigm
        annotation_name = {
            "MI": "mi_class",
            "ERP": "erp_class",
            "SSVEP": "ssvep_class",
        }.get(paradigm, "class")

        label_prefix = {
            "MI": "mi",
            "ERP": "erp",
            "SSVEP": "ssvep",
        }.get(paradigm, "class")

        # ── Ensure pool hierarchy ────────────────────────────────────────
        self.pool.ensure_dataset(dataset_id, name=self.task_config.dataset_name)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id,
                                  sampling_rate=float(fs))
        self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

        max_shard_mb = self.pool.config.get("storage", {}).get(
            "max_shard_size_mb", 200.0
        )
        compression = self.pool.config.get("storage", {}).get(
            "compression", "gzip"
        )

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
                for trial_idx in range(n_trials):
                    # Extract epoch: smt is (samples, trials, channels)
                    # → transpose to (channels, samples)
                    epoch = smt[:, trial_idx, :].T.copy()  # (n_channels, n_samples)

                    label_int = int(y_dec[trial_idx])
                    label_name = class_labels.get(
                        label_int, f"class_{label_int}"
                    )

                    # Onset in continuous recording
                    onset_sample = (
                        int(t_onsets[trial_idx]) if t_onsets is not None
                        else trial_idx * n_epoch_samples
                    )

                    # ── Annotations ──────────────────────────────────────
                    annotations = [
                        CategoricalAnnotation(
                            annotation_id=(
                                f"ann_{label_prefix}_class_{run_id}_{trial_idx:04d}"
                            ),
                            name=annotation_name,
                            value=label_name,
                        ),
                        NumericAnnotation(
                            annotation_id=(
                                f"ann_{label_prefix}_label_{run_id}_{trial_idx:04d}"
                            ),
                            name=f"{label_prefix}_label",
                            numeric_value=float(label_int),
                        ),
                        CategoricalAnnotation(
                            annotation_id=(
                                f"ann_split_{run_id}_{trial_idx:04d}"
                            ),
                            name="split",
                            value=split_name,
                        ),
                    ]

                    # SSVEP: add stimulus frequency annotation
                    if paradigm == "SSVEP":
                        stim_freqs = self.task_config.data.get(
                            "stimulus_frequencies", {}
                        )
                        freq = stim_freqs.get(label_int)
                        if freq is not None:
                            annotations.append(NumericAnnotation(
                                annotation_id=(
                                    f"ann_ssvep_freq_{run_id}_{trial_idx:04d}"
                                ),
                                name="stimulus_frequency_hz",
                                numeric_value=float(freq),
                            ))

                    # ── Atom ID ──────────────────────────────────────────
                    atom_id = compute_atom_id(
                        dataset_id=dataset_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        run_id=run_id,
                        onset_sample=onset_sample,
                    )

                    temporal = TemporalInfo(
                        onset_sample=onset_sample,
                        onset_seconds=onset_sample / fs,
                        duration_samples=n_epoch_samples,
                        duration_seconds=epoch_duration_s,
                    )

                    atom = Atom(
                        atom_id=atom_id,
                        atom_type=AtomType.EVENT_EPOCH,
                        dataset_id=dataset_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        run_id=run_id,
                        trial_index=trial_idx,
                        signal_ref=SignalRef(
                            file_path="__placeholder__",
                            internal_path=f"/atoms/{atom_id}/signal",
                            shape=(len(channel_ids), n_epoch_samples),
                        ),
                        temporal=temporal,
                        channel_ids=channel_ids,
                        n_channels=len(channel_ids),
                        sampling_rate=float(fs),
                        annotations=annotations,
                        quality=QualityInfo(
                            overall_status=QualityStatus.GOOD,
                        ),
                        processing_history=ProcessingHistory(
                            steps=[
                                ProcessingStep(
                                    operation="raw_import",
                                    parameters={
                                        "format": "openbmi",
                                        "paradigm": paradigm,
                                        "source_file": mat_filename,
                                        "split": split_name,
                                        "trial_index": trial_idx,
                                        "label": label_int,
                                        "label_name": label_name,
                                        "onset_sample": onset_sample,
                                        "epoch_samples": n_epoch_samples,
                                        "unit": "uV",
                                    },
                                ),
                            ],
                            is_raw=True,
                            version_tag="raw",
                        ),
                        custom_fields={
                            "paradigm": paradigm.lower(),
                            "split": split_name,
                        },
                    )

                    # Convert to pool storage unit (µV → µV, identity)
                    epoch_conv, storage_unit, orig_unit = convert_to_storage_unit(
                        epoch.astype(np.float32), source_unit="uV",
                        pool_config=self.pool.config,
                    )
                    atom.signal_unit = storage_unit
                    atom.original_unit = orig_unit

                    # Validate signal
                    warnings = validate_signal(
                        signal=epoch_conv,
                        atom_id=atom_id,
                        config=self.pool.config.get("import", {}),
                        signal_unit=storage_unit,
                    )
                    all_warnings.extend(warnings)

                    # Write signal to HDF5
                    signal_ref = shard_mgr.write_atom_signal(atom_id, epoch_conv)
                    atom.signal_ref = signal_ref

                    # Write metadata
                    writer.write_atom(atom)
                    atoms.append(atom)

        return atoms, ch_infos, all_warnings

    # ------------------------------------------------------------------
    # Public API: import one subject file
    # ------------------------------------------------------------------

    def import_subject(
        self,
        mat_path: Path,
        subject_id: str,
        session_id: Optional[str] = None,
        max_trials: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import one OpenBMI .mat file (one subject, one session, one paradigm).

        Args:
            mat_path: Path to .mat file (e.g., MI/sess01_subj01_EEG_MI.mat).
            subject_id: Subject identifier (e.g., 'S01').
            session_id: Override session_id. If None, auto-detected from filename.
            max_trials: Limit trials per split (for testing).

        Returns:
            List of ImportResult (one per split: train + test).
        """
        mat_path = Path(mat_path)
        parsed = _parse_filename(mat_path.name)
        if parsed is None:
            raise ValueError(
                f"Cannot parse OpenBMI filename: {mat_path.name}. "
                f"Expected pattern: sess{{NN}}_subj{{NN}}_EEG_{{MI|ERP|SSVEP}}.mat"
            )

        sess_num, subj_num, paradigm = parsed
        if session_id is None:
            session_id = f"ses-{sess_num:02d}"

        dataset_id = self.task_config.dataset_id
        class_labels = self.task_config.data.get("class_labels", {})
        class_labels = {int(k): str(v) for k, v in class_labels.items()}

        logger.info(
            "Loading OpenBMI %s: %s (subject: %s, session: %s)",
            paradigm, mat_path.name, subject_id, session_id,
        )

        from neuroatom.utils.mat_compat import require_mat_v5
        require_mat_v5(mat_path, "OpenBMIImporter")
        mat = sio.loadmat(str(mat_path))

        train_key, test_key = _PARADIGM_KEYS[paradigm]

        results = []
        total_atoms = 0

        for split_name, mat_key in [("train", train_key), ("test", test_key)]:
            if mat_key not in mat:
                logger.warning(
                    "Key '%s' not found in %s, skipping %s split.",
                    mat_key, mat_path.name, split_name,
                )
                continue

            struct = mat[mat_key][0, 0]
            run_id = f"run-{split_name}"

            atoms, ch_infos, warnings = self._extract_split(
                struct=struct,
                paradigm=paradigm,
                split_name=split_name,
                dataset_id=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                class_labels=class_labels,
                mat_filename=mat_path.name,
                max_trials=max_trials,
            )

            n_per_class = {}
            for a in atoms:
                for ann in a.annotations:
                    if hasattr(ann, "value") and ann.name.endswith("_class"):
                        n_per_class[ann.value] = n_per_class.get(ann.value, 0) + 1

            smt_shape = struct["smt"].shape
            logger.info(
                "Imported %s/%s/%s/%s [%s]: %d epochs (%s) "
                "× %d ch × %d samples (%.0f Hz)",
                dataset_id, subject_id, session_id, run_id, split_name,
                len(atoms),
                ", ".join(f"{k}={v}" for k, v in sorted(n_per_class.items())),
                smt_shape[2], smt_shape[0],
                int(struct["fs"].ravel()[0]),
            )

            run_meta = RunMeta(
                run_id=run_id,
                session_id=session_id,
                subject_id=subject_id,
                dataset_id=dataset_id,
                run_index=0 if split_name == "train" else 1,
                task_type=self.task_config.task_type,
                n_trials=len(atoms),
                paradigm_details={
                    "type": paradigm.lower(),
                    "split": split_name,
                    "n_trials": len(atoms),
                    "n_classes": len(n_per_class),
                    "class_distribution": n_per_class,
                    "epoch_samples": smt_shape[0],
                },
            )
            self.pool.register_run(run_meta)
            self._write_channels_json(
                dataset_id, subject_id, session_id, ch_infos
            )

            results.append(ImportResult(
                atoms=atoms,
                run_meta=run_meta,
                channel_infos=ch_infos,
                warnings=warnings,
            ))
            total_atoms += len(atoms)

        logger.info(
            "Subject %s [%s]: imported %d splits, %d total atoms.",
            subject_id, paradigm, len(results), total_atoms,
        )
        return results

    # ------------------------------------------------------------------
    # Convenience: import all subjects for a paradigm
    # ------------------------------------------------------------------

    def import_paradigm(
        self,
        data_dir: Path,
        paradigm: str = "MI",
        max_subjects: Optional[int] = None,
        max_trials: Optional[int] = None,
        sessions: Optional[List[int]] = None,
    ) -> List[ImportResult]:
        """Import all subjects for a given paradigm.

        Args:
            data_dir: Root OpenBMI directory (containing MI/, ERP/, SSVEP/).
            paradigm: 'MI', 'ERP', or 'SSVEP'.
            max_subjects: Limit number of subjects (for testing).
            max_trials: Limit trials per split per subject.
            sessions: List of session numbers to import (default: [1, 2]).

        Returns:
            Flat list of all ImportResult across all subjects.
        """
        data_dir = Path(data_dir)
        paradigm = paradigm.upper()

        if paradigm not in _PARADIGM_KEYS:
            raise ValueError(
                f"Unknown paradigm '{paradigm}'. "
                f"Must be one of: {list(_PARADIGM_KEYS.keys())}"
            )

        # Find paradigm directory
        paradigm_dir = data_dir / paradigm
        if not paradigm_dir.is_dir():
            # Try flat layout
            paradigm_dir = data_dir

        mat_files = sorted(paradigm_dir.glob(f"sess*_subj*_EEG_{paradigm}.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No {paradigm} .mat files found in {paradigm_dir}"
            )

        # Filter by session if specified
        if sessions is not None:
            sess_set = set(sessions)
            mat_files = [
                f for f in mat_files
                if _parse_filename(f.name) and _parse_filename(f.name)[0] in sess_set
            ]

        # Group by subject, limit if requested
        subjects_seen = set()
        filtered_files = []
        for f in mat_files:
            parsed = _parse_filename(f.name)
            if parsed is None:
                continue
            _, subj_num, _ = parsed
            if max_subjects is not None and subj_num not in subjects_seen:
                if len(subjects_seen) >= max_subjects:
                    continue
            subjects_seen.add(subj_num)
            filtered_files.append(f)

        logger.info(
            "OpenBMI %s: found %d files (%d subjects) in %s",
            paradigm, len(filtered_files), len(subjects_seen), paradigm_dir,
        )

        # ── Load questionnaire for subject demographics ──
        questionnaire = _parse_questionnaire(data_dir / "Questionnaire_results.csv")
        if questionnaire:
            logger.info(
                "Loaded questionnaire data for %d subjects", len(questionnaire),
            )

        # Register subjects with demographics before importing
        dataset_id = self.task_config.dataset_id
        registered_subjects: set = set()
        for subj_num in sorted(subjects_seen):
            subject_id = f"S{subj_num:02d}"
            if subject_id in registered_subjects:
                continue
            registered_subjects.add(subject_id)

            q_info = dict(questionnaire.get(subj_num, {}))
            age = q_info.pop("age", None)
            sex = q_info.pop("sex", None)

            self.pool.register_subject(SubjectMeta(
                subject_id=subject_id,
                dataset_id=dataset_id,
                age=age,
                sex=sex,
                custom_fields=q_info if q_info else {},
            ))

        all_results = []
        for mat_file in filtered_files:
            parsed = _parse_filename(mat_file.name)
            _, subj_num, _ = parsed
            subject_id = f"S{subj_num:02d}"

            results = self.import_subject(
                mat_path=mat_file,
                subject_id=subject_id,
                max_trials=max_trials,
            )
            all_results.extend(results)

        total_atoms = sum(len(r.atoms) for r in all_results)
        logger.info(
            "OpenBMI %s: imported %d subjects, %d total atoms.",
            paradigm, len(subjects_seen), total_atoms,
        )
        return all_results


# Auto-register for all three paradigms
register_importer("openbmi_mi", OpenBMIImporter)
register_importer("openbmi_erp", OpenBMIImporter)
register_importer("openbmi_ssvep", OpenBMIImporter)
