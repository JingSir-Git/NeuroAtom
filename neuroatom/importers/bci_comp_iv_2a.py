"""BCI Competition IV Dataset 2a Importer: 4-class Motor Imagery.

Handles the MATLAB-processed version of the BCI Competition IV 2a dataset
(Brunner et al., 2008; Tangermann et al., 2012). The .mat files contain
preprocessed EEG data organized as a struct-array of runs, with trial onsets,
class labels, and artifact flags pre-extracted.

Data structure per subject .mat file:
    data: ndarray(N_runs,) of structs, each containing:
        X:         (n_samples, 25) float64 — continuous EEG in µV
        y:         (n_trials,) uint8 — class labels 1-4 (empty for cal runs)
        trial:     (n_trials,) int32 — trial onset sample indices
        artifacts: (n_trials,) uint8 — artifact flags (0=clean, 1=artifact)
        fs:        int — sampling rate (250 Hz)
        classes:   (4,) object — ['left hand','right hand','feet','tongue']
        age:       int
        gender:    str

Channel layout (25 channels, NOT stored in .mat — from official docs):
    EEG (22):  Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6,
               CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
    EOG (3):   EOG-left, EOG-central, EOG-right

Paradigm:
    Fixation cross (2s) → Cue arrow (1.25s) → Motor imagery → Rest
    4 classes: left hand, right hand, both feet, tongue
    48 trials per run, 6 labelled runs per subject = 288 trials total
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

from neuroatom.core.annotation import CategoricalAnnotation, NumericAnnotation
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
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
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)


# Official BCI Competition IV 2a channel order (from dataset description PDF)
BCI_IV_2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
    "EOG-left", "EOG-central", "EOG-right",
]

# Default class label mapping (1-indexed)
DEFAULT_CLASS_LABELS = {
    1: "left_hand",
    2: "right_hand",
    3: "both_feet",
    4: "tongue",
}


def _detect_bci_iv_2a_mat(path: Path) -> bool:
    """Detect if a .mat file is BCI Competition IV 2a format.

    Checks for: top-level 'data' key → struct array where each struct
    has 'X', 'fs', 'classes' attributes.
    """
    if not path.suffix.lower() == ".mat":
        return False

    try:
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False,
                          variable_names=["data"])
    except Exception:
        return False

    if "data" not in mat:
        return False

    data = mat["data"]
    if not isinstance(data, np.ndarray) or data.dtype != object:
        return False

    if data.size == 0:
        return False

    # Check first element has expected attributes
    first = data.flat[0]
    return (hasattr(first, "X") and hasattr(first, "fs") and
            hasattr(first, "classes"))


class BCICompIV2aImporter(BaseImporter):
    """Importer for BCI Competition IV Dataset 2a (.mat format).

    Each labelled run (runs 3-8) contains 48 trials. Each trial is extracted
    as one Atom with EVENT_EPOCH type. Unlabelled runs (0-2) are skipped by
    default.

    Trial epoching uses the `trial` onset array from the .mat file, with
    a configurable window duration (default 6s from task_config).
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Detect BCI Competition IV 2a .mat files."""
        path = Path(path)
        if path.is_dir():
            mats = list(path.glob("A0?T.mat")) + list(path.glob("A0?E.mat"))
            return len(mats) > 0 and _detect_bci_iv_2a_mat(mats[0])
        return _detect_bci_iv_2a_mat(path)

    def load_raw(self, path):
        raise NotImplementedError(
            "BCICompIV2aImporter.load_raw() is not used directly. "
            "Use import_subject() for multi-run BCI Competition .mat files."
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
        n_channels: int,
        srate: float,
    ) -> List[ChannelInfo]:
        """Build ChannelInfo list from official channel layout.

        Channel names are NOT stored in .mat files — we use the official
        25-channel layout from the BCI Competition IV 2a dataset description.
        """
        # Get channel names from task config or use defaults
        ch_names = self.task_config.data.get("channel_names", BCI_IV_2A_CHANNELS)

        if len(ch_names) != n_channels:
            logger.warning(
                "Task config has %d channel names but data has %d channels. "
                "Falling back to official layout or generating generic names.",
                len(ch_names), n_channels,
            )
            if n_channels == len(BCI_IV_2A_CHANNELS):
                ch_names = BCI_IV_2A_CHANNELS
            else:
                ch_names = [f"Ch_{i+1}" for i in range(n_channels)]

        unit = self.task_config.signal_unit or "uV"
        type_overrides = self.task_config.channel_type_overrides
        exclude_set = set(self.task_config.exclude_channels)

        ch_infos = []
        for idx, ch_name in enumerate(ch_names):
            if ch_name in exclude_set:
                continue

            # Determine channel type
            if ch_name in type_overrides:
                ch_type = ChannelType(type_overrides[ch_name])
            elif ch_name.upper().startswith("EOG"):
                ch_type = ChannelType.EOG
            else:
                ch_type = ChannelType.EEG

            std_name = standardize_channel_name(ch_name) if ch_type == ChannelType.EEG else None

            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{idx:03d}",
                index=idx,
                name=ch_name,
                standard_name=std_name,
                type=ch_type,
                unit=unit,
                sampling_rate=srate,
                status=ChannelStatus.GOOD,
            ))

        return ch_infos

    # ------------------------------------------------------------------
    # Main entry: import one subject file
    # ------------------------------------------------------------------

    def import_subject(
        self,
        mat_path: Path,
        subject_id: str,
        session_id: str = "ses-01",
        include_unlabelled: bool = False,
        max_runs: Optional[int] = None,
        max_trials: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import all labelled runs from a single subject .mat file.

        Args:
            mat_path: Path to subject .mat file (e.g. A01T.mat)
            subject_id: Subject identifier (e.g. 'A01')
            session_id: Session identifier (default 'ses-01'; use 'ses-T' for
                        training, 'ses-E' for evaluation to maintain separation)
            include_unlabelled: If True, also import calibration runs (0-2)
                               as continuous_segment atoms without labels.
            max_runs: Max number of labelled runs to import (for testing).
            max_trials: Max trials per run to import (for testing).

        Returns:
            List of ImportResult, one per imported run.
        """
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P

        mat_path = Path(mat_path)
        logger.info("Loading BCI IV 2a: %s (subject: %s)", mat_path, subject_id)

        # Auto-detect session type from filename (A0xT.mat → Training, A0xE.mat → Evaluation)
        session_type = "unknown"
        fname_upper = mat_path.stem.upper()
        if fname_upper.endswith("T"):
            session_type = "training"
        elif fname_upper.endswith("E"):
            session_type = "evaluation"

        mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        data = mat["data"]

        dataset_id = self.task_config.dataset_id
        class_labels = self.task_config.data.get("class_labels", DEFAULT_CLASS_LABELS)
        # YAML loads dict keys as int sometimes, ensure str keys work
        class_labels = {int(k): str(v) for k, v in class_labels.items()}

        trial_duration_s = self.task_config.trial_definition.get(
            "mat_trial_duration_s", 6.0
        )

        # Ensure pool hierarchy
        self.pool.ensure_dataset(dataset_id, name=self.task_config.dataset_name)
        self.pool.ensure_subject(dataset_id, subject_id)

        results = []
        labelled_run_count = 0

        for run_idx in range(len(data)):
            run = data[run_idx]
            has_labels = hasattr(run, "y") and run.y.size > 0

            # Skip unlabelled runs unless requested
            if not has_labels and not include_unlabelled:
                logger.debug("Skipping unlabelled run %d", run_idx)
                continue

            if has_labels:
                labelled_run_count += 1
                if max_runs is not None and labelled_run_count > max_runs:
                    break

            signal = run.X  # (n_samples, n_channels)
            srate = float(run.fs)
            n_samples_total, n_channels = signal.shape

            # Channel info (built once, same for all runs)
            ch_infos = self._build_channel_infos(n_channels, srate)
            channel_ids = [ch.channel_id for ch in ch_infos]

            run_id = f"run-{run_idx:02d}"

            # Ensure session and run
            self.pool.ensure_session(dataset_id, subject_id, session_id,
                                     sampling_rate=srate)
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            max_shard_mb = self.pool.config.get("storage", {}).get(
                "max_shard_size_mb", 200.0
            )
            compression = self.pool.config.get("storage", {}).get(
                "compression", "gzip"
            )

            atoms = []
            all_warnings = []

            if has_labels:
                # ---- Labelled MI runs: extract trial epochs ----
                trial_onsets = run.trial.astype(int)
                labels = run.y.astype(int)
                artifacts = run.artifacts.astype(int) if run.artifacts.size > 0 else np.zeros_like(labels)

                n_trials = len(trial_onsets)
                trial_samples = int(trial_duration_s * srate)

                if max_trials is not None:
                    n_trials = min(n_trials, max_trials)

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
                            onset = trial_onsets[trial_idx]
                            label = int(labels[trial_idx])
                            is_artifact = bool(artifacts[trial_idx])

                            # Epoch boundaries
                            end = min(onset + trial_samples, n_samples_total)
                            actual_samples = end - onset

                            if actual_samples < trial_samples * 0.5:
                                logger.warning(
                                    "Trial %d truncated: %d/%d samples. Skipping.",
                                    trial_idx, actual_samples, trial_samples,
                                )
                                continue

                            # Extract epoch: (n_channels, n_samples)
                            epoch = signal[onset:end, :].T.copy()

                            # Pad if slightly short (end of recording)
                            if epoch.shape[1] < trial_samples:
                                pad_width = trial_samples - epoch.shape[1]
                                epoch = np.pad(
                                    epoch,
                                    ((0, 0), (0, pad_width)),
                                    mode="constant",
                                    constant_values=0,
                                )

                            # Build annotations
                            label_name = class_labels.get(label, f"class_{label}")
                            annotations = [
                                CategoricalAnnotation(
                                    annotation_id=f"ann_mi_class_{run_idx:02d}_{trial_idx:03d}",
                                    name="mi_class",
                                    value=label_name,
                                ),
                                NumericAnnotation(
                                    annotation_id=f"ann_mi_label_{run_idx:02d}_{trial_idx:03d}",
                                    name="mi_label",
                                    numeric_value=float(label),
                                ),
                            ]

                            if is_artifact:
                                annotations.append(CategoricalAnnotation(
                                    annotation_id=f"ann_artifact_{run_idx:02d}_{trial_idx:03d}",
                                    name="artifact",
                                    value="rejected",
                                ))

                            if session_type != "unknown":
                                annotations.append(CategoricalAnnotation(
                                    annotation_id=f"ann_sesstype_{run_idx:02d}_{trial_idx:03d}",
                                    name="session_type",
                                    value=session_type,
                                ))

                            # Quality
                            quality = QualityInfo(
                                overall_status=QualityStatus.REJECTED if is_artifact
                                else QualityStatus.GOOD,
                            )

                            # Atom ID
                            atom_id = compute_atom_id(
                                dataset_id=dataset_id,
                                subject_id=subject_id,
                                session_id=session_id,
                                run_id=run_id,
                                onset_sample=onset,
                            )

                            temporal = TemporalInfo(
                                onset_sample=onset,
                                onset_seconds=onset / srate,
                                duration_samples=trial_samples,
                                duration_seconds=trial_duration_s,
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
                                    shape=(len(channel_ids), trial_samples),
                                ),
                                temporal=temporal,
                                channel_ids=channel_ids,
                                n_channels=len(channel_ids),
                                sampling_rate=srate,
                                annotations=annotations,
                                quality=quality,
                                processing_history=ProcessingHistory(
                                    steps=[
                                        ProcessingStep(
                                            operation="raw_import",
                                            parameters={
                                                "format": "bci_comp_iv_2a_mat",
                                                "source_file": mat_path.name,
                                                "run_index": run_idx,
                                                "trial_index": trial_idx,
                                                "onset_sample": int(onset),
                                                "trial_duration_s": trial_duration_s,
                                                "label": label,
                                                "artifact": is_artifact,
                                                "unit": "uV",
                                            },
                                        ),
                                    ],
                                    is_raw=True,
                                    version_tag="raw",
                                ),
                                custom_fields={
                                    "age": int(run.age) if hasattr(run, "age") else None,
                                    "gender": str(run.gender) if hasattr(run, "gender") else None,
                                },
                            )

                            # Validate signal
                            warnings = validate_signal(
                                signal=epoch.astype(np.float32),
                                atom_id=atom_id,
                                config=self.pool.config.get("import", {}),
                            )
                            all_warnings.extend(warnings)

                            # Write signal
                            signal_ref = shard_mgr.write_atom_signal(
                                atom_id, epoch
                            )
                            atom.signal_ref = signal_ref

                            # Write JSONL
                            writer.write_atom(atom)
                            atoms.append(atom)

                logger.info(
                    "Imported %s/%s/%s/%s: %d MI trials (%d artifacts) "
                    "× %d ch × %d samples (%.1f Hz)",
                    dataset_id, subject_id, session_id, run_id,
                    len(atoms),
                    sum(1 for a in atoms
                        if a.quality and a.quality.overall_status == QualityStatus.REJECTED),
                    len(channel_ids), trial_samples, srate,
                )

            else:
                # ---- Unlabelled calibration runs: import as continuous ----
                # Store the entire recording as a single CONTINUOUS_SEGMENT atom
                continuous = signal.T.copy()  # (n_channels, n_samples)

                atom_id = compute_atom_id(
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    session_id=session_id,
                    run_id=run_id,
                    onset_sample=0,
                )

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
                        shape=(len(channel_ids), n_samples_total),
                    ),
                    temporal=TemporalInfo(
                        onset_sample=0,
                        onset_seconds=0.0,
                        duration_samples=n_samples_total,
                        duration_seconds=n_samples_total / srate,
                    ),
                    channel_ids=channel_ids,
                    n_channels=len(channel_ids),
                    sampling_rate=srate,
                    annotations=[],
                    processing_history=ProcessingHistory(
                        steps=[
                            ProcessingStep(
                                operation="raw_import",
                                parameters={
                                    "format": "bci_comp_iv_2a_mat",
                                    "source_file": mat_path.name,
                                    "run_index": run_idx,
                                    "type": "calibration",
                                    "unit": "uV",
                                },
                            ),
                        ],
                        is_raw=True,
                        version_tag="raw",
                    ),
                    custom_fields={
                        "age": int(run.age) if hasattr(run, "age") else None,
                        "gender": str(run.gender) if hasattr(run, "gender") else None,
                    },
                )

                warnings = validate_signal(
                    signal=continuous.astype(np.float32),
                    atom_id=atom_id,
                    config=self.pool.config.get("import", {}),
                )
                all_warnings.extend(warnings)

                with ShardManager(
                    pool_root=self.pool.root,
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    session_id=session_id,
                    run_id=run_id,
                    max_shard_size_mb=max_shard_mb,
                    compression=compression,
                ) as shard_mgr:
                    signal_ref = shard_mgr.write_atom_signal(atom_id, continuous)
                    atom.signal_ref = signal_ref

                jsonl_path = P.atoms_jsonl_path(
                    self.pool.root, dataset_id, subject_id, session_id, run_id
                )
                with AtomJSONLWriter(jsonl_path) as writer:
                    writer.write_atom(atom)

                atoms.append(atom)
                logger.info(
                    "Imported %s/%s/%s/%s: calibration segment %d ch × %d samples (%.1fs)",
                    dataset_id, subject_id, session_id, run_id,
                    len(channel_ids), n_samples_total, n_samples_total / srate,
                )

            run_meta = RunMeta(
                run_id=run_id,
                session_id=session_id,
                subject_id=subject_id,
                dataset_id=dataset_id,
                run_index=run_idx,
                task_type=self.task_config.task_type,
                n_trials=len(atoms),
                paradigm_details={
                    "type": "motor_imagery" if has_labels else "calibration",
                    "n_trials": len(atoms),
                    "trial_duration_s": trial_duration_s if has_labels else None,
                },
            )
            self.pool.register_run(run_meta)

            results.append(ImportResult(
                atoms=atoms,
                run_meta=run_meta,
                channel_infos=ch_infos,
                warnings=all_warnings,
            ))

        total_atoms = sum(len(r.atoms) for r in results)
        logger.info(
            "Subject %s: imported %d runs, %d total atoms.",
            subject_id, len(results), total_atoms,
        )

        return results


# Auto-register
register_importer("bci_comp_iv_2a", BCICompIV2aImporter)
