"""SNHL-AAD Importer: Selective Attention in Normal-Hearing / Hearing-Impaired.

Fuglsang et al. (2020) dataset. 44 subjects, 512 Hz BDF, BIDS format.
Special feature: 19 subjects have ear-EEG channels alongside scalp EEG.

Data layout (BIDS):
    ds-eeg-snhl/
        participants.tsv
        dataset_description.json
        sub-001/ ... sub-044/
            eeg/
                sub-XXX_task-selectiveattention_eeg.bdf
                sub-XXX_task-selectiveattention_channels.tsv
                sub-XXX_task-selectiveattention_events.tsv
                sub-XXX_task-tonestimuli_eeg.bdf
                sub-XXX_task-rest_eeg.bdf
                ...

Key design decisions:
    - Each trial (targetonset → trialend) becomes one Atom
    - EarEEG channels preserved as ChannelType.EAR_EEG if available,
      mapped to ChannelType.OTHER when enum unavailable
    - Attention direction (attend_left/attend_right) stored as annotation
    - Condition (single-talker/two-talker) stored as annotation
    - Hearing status (NH/HI) stored as subject metadata
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import re

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the SNHL-AAD importer")

from neuroatom.core.annotation import (
    CategoricalAnnotation,
    ContinuousAnnotation,
    NumericAnnotation,
    TextAnnotation,
)
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import AtomType, ChannelStatus, ChannelType
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.core.session import SessionMeta
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

# Trigger values for target onset conditions
_TARGET_ONSET_VALUES = {224, 240, 248, 252, 254, 255, 135, 133}
_MASKER_ONSET_VALUE = 137
_TRIAL_END_VALUE = 131

# Channel type based on description in channels.tsv
_DESC_TO_TYPE = {
    "scalp": ChannelType.EEG,
    "left_eog_1": ChannelType.EOG,
    "left_eog_2": ChannelType.EOG,
    "left_eog_3": ChannelType.EOG,
    "right_eog_1": ChannelType.EOG,
    "right_eog_2": ChannelType.EOG,
    "right_eog_3": ChannelType.EOG,
    "left_mastoid": ChannelType.REF,
    "right_mastoid": ChannelType.REF,
    "right_eareeg_1": ChannelType.OTHER,  # ear-EEG
    "right_eareeg_2": ChannelType.OTHER,
    "right_eareeg_3": ChannelType.OTHER,
    "left_eareeg_1": ChannelType.OTHER,
    "left_eareeg_2": ChannelType.OTHER,
    "left_eareeg_3": ChannelType.OTHER,
}


def _load_stimulus_envelope(mat_path: Path) -> Optional[Dict[str, Any]]:
    """Load audio stimulus envelope from derivatives .mat file.

    Each .mat contains ``dat`` struct with fields:
    - ``feat``: (N, 1) float64 envelope signal
    - ``fs``: scalar sampling rate (typically 512)
    - ``t``: (N, 1) time vector in seconds

    Returns dict with keys ``envelope``, ``fs``, ``t`` or None on failure.
    """
    try:
        import scipy.io as sio
        mat = sio.loadmat(str(mat_path))
        dat = mat["dat"][0, 0]
        envelope = dat["feat"].ravel().astype(np.float32)
        fs = float(dat["fs"].flat[0])
        t = dat["t"].ravel().astype(np.float64)
        return {"envelope": envelope, "fs": fs, "t": t}
    except Exception as e:
        logger.debug("Could not load stimulus %s: %s", mat_path.name, e)
        return None


def _resolve_stimulus_paths(
    bids_root: Path,
    stim_file: str,
    use_woa: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve target and masker stimulus .mat paths from events stim_file.

    Args:
        bids_root: Dataset root (contains ``derivatives/stimuli/``).
        stim_file: e.g. ``"sub001/target/t004.wav"``.
        use_woa: If True, prefer ``*woa.mat`` (without onset artifact).

    Returns:
        (target_path, masker_path) — masker_path is None for single-talker.
    """
    stim_dir = bids_root / "derivatives" / "stimuli"
    m = re.match(r"(sub\d+)/target/t(\d+)", stim_file)
    if not m:
        return None, None

    sub_stim = m.group(1)
    trial_num = m.group(2)
    suffix = "woa" if use_woa else ""

    target_name = f"t{trial_num}{suffix}.mat"
    target_path = stim_dir / sub_stim / "target" / target_name
    if not target_path.exists():
        # Fall back to non-woa
        target_path = stim_dir / sub_stim / "target" / f"t{trial_num}.mat"
        if not target_path.exists():
            target_path = None

    masker_name = f"m{trial_num}{suffix}.mat"
    masker_path = stim_dir / sub_stim / "masker" / masker_name
    if not masker_path.exists():
        masker_path = stim_dir / sub_stim / "masker" / f"m{trial_num}.mat"
        if not masker_path.exists():
            masker_path = None  # single-talker trials have no masker

    return target_path, masker_path


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a BIDS-style TSV file."""
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def _parse_channels_tsv(
    channels_path: Path,
    sfreq: float,
) -> List[ChannelInfo]:
    """Parse BIDS channels.tsv with ear-EEG awareness."""
    rows = _read_tsv(channels_path)
    ch_infos = []

    for idx, row in enumerate(rows):
        name = row.get("name", f"Ch_{idx}")
        bids_type = row.get("type", "eeg").lower()
        desc = row.get("description", "").lower().strip()
        unit = row.get("units", "uV")

        # Skip Status/trigger channel
        if name == "Status" or bids_type == "unknown":
            continue

        # Determine type from description (most accurate for this dataset)
        if desc in _DESC_TO_TYPE:
            ch_type = _DESC_TO_TYPE[desc]
        elif bids_type == "eareeg":
            ch_type = ChannelType.OTHER  # ear-EEG mapped to OTHER
        elif bids_type == "eeg":
            ch_type = ChannelType.EEG
        elif bids_type == "eog":
            ch_type = ChannelType.EOG
        else:
            ch_type = ChannelType.OTHER

        is_eareeg = "eareeg" in desc or bids_type == "eareeg"

        ch_infos.append(ChannelInfo(
            channel_id=f"ch_{idx:03d}",
            index=idx,
            name=name,
            standard_name=standardize_channel_name(name),
            type=ch_type,
            unit=unit,
            sampling_rate=sfreq,
            status=ChannelStatus.UNKNOWN,
            custom_fields={"is_eareeg": is_eareeg} if is_eareeg else {},
        ))

    return ch_infos


def _parse_selectiveattention_events(
    events_path: Path,
) -> List[Dict[str, Any]]:
    """Parse selective attention events.tsv into trial descriptors.

    Each trial is delimited by a targetonset event and the following
    trialend event. Returns list of dicts with onset/end samples and metadata.
    """
    rows = _read_tsv(events_path)
    if not rows:
        return []

    trials = []
    current_trial = None

    for row in rows:
        trigger_type = row.get("trigger_type", "n/a")
        value = row.get("value", "")

        try:
            sample = int(float(row.get("sample", 0)))
        except (ValueError, TypeError):
            continue

        if trigger_type == "targetonset":
            current_trial = {
                "onset_sample": sample,
                "attend_lr": row.get("attend_left_right", ""),
                "condition": row.get("single_talker_two_talker", ""),
                "attend_speaker": row.get("attend_male_female", ""),
                "stim_file": row.get("stim_file", ""),
                "masker_sample": None,
                "end_sample": None,
                "difficulty": None,
                "accuracy": None,
            }
        elif trigger_type == "maskeronset" and current_trial is not None:
            current_trial["masker_sample"] = sample
        elif trigger_type == "trialend" and current_trial is not None:
            current_trial["end_sample"] = sample
            # Parse behavioural data
            diff = row.get("diffulty_ratings", "n/a")
            acc = row.get("questionnaire_scores", "n/a")
            if diff and diff != "n/a":
                try:
                    current_trial["difficulty"] = float(diff)
                except ValueError:
                    pass
            if acc and acc != "n/a":
                try:
                    current_trial["accuracy"] = float(acc)
                except ValueError:
                    pass
            trials.append(current_trial)
            current_trial = None

    return trials


class SNHLAADImporter(BaseImporter):
    """Importer for the ds-eeg-snhl BIDS dataset.

    Imports the selective attention task as the primary AAD data.
    Optionally also imports tone ERP and resting-state tasks.
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Detect ds-eeg-snhl by dataset_description.json name."""
        path = Path(path)
        desc = path / "dataset_description.json"
        if not desc.exists():
            return False
        try:
            with open(desc, encoding="utf-8") as f:
                d = json.load(f)
            return "selective auditory attention" in d.get("Name", "").lower()
        except Exception:
            return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="WARNING")
        return raw, {"declared_unit": "uV"}

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        """Not used — channels parsed from TSV."""
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        """Not used — events parsed from TSV."""
        return None

    # ------------------------------------------------------------------
    # Main entry: import dataset
    # ------------------------------------------------------------------

    def import_dataset(
        self,
        bids_root: Path,
        tasks: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import ds-eeg-snhl dataset.

        Args:
            bids_root: Path to ds-eeg-snhl root.
            tasks: Which tasks to import. Default: ["selectiveattention"].
            subjects: Subset of subjects. Default: all.
            max_subjects: Limit number of subjects.

        Returns:
            List of ImportResult per subject-task.
        """
        bids_root = Path(bids_root)
        dataset_id = self._task_config.dataset_id
        tasks = tasks or ["selectiveattention"]

        # Read participants.tsv
        participants = _read_tsv(bids_root / "participants.tsv")

        # Register dataset
        desc_path = bids_root / "dataset_description.json"
        ds_name = self._task_config.dataset_name
        if desc_path.exists():
            with open(desc_path, encoding="utf-8") as f:
                desc = json.load(f)
            ds_name = desc.get("Name", ds_name)

        self._pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id,
            name=ds_name,
            task_types=["auditory_attention_decoding"],
            n_subjects=len(participants),
            original_format="bdf",
            license="ODbL",
        ))

        # Discover subjects
        sub_dirs = sorted(bids_root.glob("sub-*"))
        if subjects:
            sub_dirs = [d for d in sub_dirs if d.name in subjects]
        if max_subjects:
            sub_dirs = sub_dirs[:max_subjects]

        results = []
        for sub_dir in sub_dirs:
            sub_id = sub_dir.name
            eeg_dir = sub_dir / "eeg"
            if not eeg_dir.exists():
                continue

            # Find participant info
            p_info = {}
            for p in participants:
                if p.get("participant_id") == sub_id:
                    p_info = p
                    break

            # Register subject
            hearing = p_info.get("hearing_status", "unknown")
            has_eareeg = p_info.get("ear_eeg", "No") == "Yes"
            age_str = p_info.get("age", "")

            self._pool.register_subject(SubjectMeta(
                subject_id=sub_id,
                dataset_id=dataset_id,
                age=int(age_str) if age_str else None,
                sex=p_info.get("gender", "").upper() if p_info.get("gender", "").upper() in ("M", "F", "O") else None,
                custom_fields={
                    "hearing_status": hearing,
                    "has_eareeg": has_eareeg,
                    "handedness": p_info.get("handedness", ""),
                },
            ))

            for task in tasks:
                try:
                    result = self._import_subject_task(
                        bids_root, sub_id, eeg_dir, task, has_eareeg,
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(
                        "Failed to import %s/%s: %s", sub_id, task, e,
                    )

        # Post-import quality assessment
        try:
            tier = self._pool.assess_quality(dataset_id)
            if tier:
                logger.info("Quality tier for %s: %s", dataset_id, tier)
        except Exception:
            pass

        logger.info(
            "SNHL-AAD import complete: %d results from %d subjects",
            len(results), len(sub_dirs),
        )
        return results

    def _import_subject_task(
        self,
        bids_root: Path,
        sub_id: str,
        eeg_dir: Path,
        task: str,
        has_eareeg: bool,
    ) -> Optional[ImportResult]:
        """Import one task for one subject."""
        dataset_id = self._task_config.dataset_id

        # Find BDF file
        bdf_pattern = f"{sub_id}_task-{task}_eeg.bdf"
        bdf_path = eeg_dir / bdf_pattern
        if not bdf_path.exists():
            logger.warning("BDF not found: %s", bdf_path)
            return None

        # Load raw data
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose="WARNING")
        sfreq = raw.info["sfreq"]
        data = raw.get_data()  # (n_channels, n_samples) in Volts
        # BDF data from MNE is in Volts → convert to µV
        data = data * 1e6

        # Parse channels.tsv
        channels_tsv = eeg_dir / f"{sub_id}_task-{task}_channels.tsv"
        channel_infos = _parse_channels_tsv(channels_tsv, sfreq)

        session_id = f"ses-{task}"
        run_id = "run-01"

        # Register session
        self._pool.ensure_session(dataset_id, sub_id, session_id, sfreq)
        self._pool.ensure_run(dataset_id, sub_id, session_id, run_id)

        # Write channel metadata
        self._write_channels_json(dataset_id, sub_id, session_id, channel_infos)

        # Filter data to matching channels only (exclude Status)
        ch_names_in_data = raw.info["ch_names"]
        ch_indices = []
        for ci in channel_infos:
            if ci.name in ch_names_in_data:
                ch_indices.append(ch_names_in_data.index(ci.name))
        signal = data[ch_indices, :]  # (n_ch, n_samples)

        # Convert to storage unit
        signal, storage_unit, _ = convert_to_storage_unit(
            signal, source_unit="uV", pool_config=self._pool.config,
        )

        # Parse events for selective attention
        events_tsv = eeg_dir / f"{sub_id}_task-{task}_events.tsv"

        if task == "selectiveattention":
            trials = _parse_selectiveattention_events(events_tsv)
            atoms = self._create_trial_atoms(
                signal, channel_infos, trials, sfreq,
                dataset_id, sub_id, session_id, run_id, has_eareeg,
                bids_root=bids_root,
            )
        else:
            # For rest/tonestimuli: one continuous atom
            atoms = [self._create_continuous_atom(
                signal, channel_infos, sfreq,
                dataset_id, sub_id, session_id, run_id, task,
            )]

        # Write atoms
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P

        max_shard_mb = self._pool.config.get("storage", {}).get(
            "max_shard_size_mb", 200.0,
        )
        compression = self._pool.config.get("storage", {}).get(
            "compression", "gzip",
        )

        stored_atoms = []
        with ShardManager(
            pool_root=self._pool.root, dataset_id=dataset_id,
            subject_id=sub_id, session_id=session_id, run_id=run_id,
            max_shard_size_mb=max_shard_mb, compression=compression,
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                self._pool.root, dataset_id, sub_id, session_id, run_id,
            )
            with AtomJSONLWriter(jsonl_path) as aw:
                for item in atoms:
                    if len(item) == 3:
                        atom, atom_signal, ann_arrays = item
                    else:
                        atom, atom_signal = item
                        ann_arrays = None
                    sig_ref = shard_mgr.write_atom_signal(
                        atom.atom_id, atom_signal.astype(np.float32),
                        ann_arrays,
                    )
                    atom.signal_ref = sig_ref
                    aw.write_atom(atom)
                    stored_atoms.append(atom)

        run_meta = RunMeta(
            run_id=run_id, session_id=session_id,
            subject_id=sub_id, dataset_id=dataset_id,
            task_type=self._task_config.task_type,
            n_trials=len(stored_atoms),
        )
        self._pool.register_run(run_meta)

        logger.info(
            "Imported %s/%s/%s: %d atoms",
            sub_id, task, run_id, len(stored_atoms),
        )
        return ImportResult(
            atoms=stored_atoms, run_meta=run_meta,
            channel_infos=channel_infos, warnings=[],
        )

    def _create_trial_atoms(
        self,
        signal: np.ndarray,
        channel_infos: List[ChannelInfo],
        trials: List[Dict],
        sfreq: float,
        dataset_id: str,
        sub_id: str,
        session_id: str,
        run_id: str,
        has_eareeg: bool,
        bids_root: Optional[Path] = None,
    ) -> List[Tuple[Atom, np.ndarray, Optional[Dict[str, np.ndarray]]]]:
        """Create one Atom per trial from selective attention events.

        When *bids_root* is given, loads target/masker audio envelope
        stimuli from ``derivatives/stimuli/`` and stores them as
        ``ContinuousAnnotation`` with HDF5 companion arrays.
        """
        atoms = []
        ch_names = [ci.name for ci in channel_infos]
        n_ch = len(channel_infos)

        # Pre-check if stimuli directory exists
        has_stimuli = (
            bids_root is not None
            and (bids_root / "derivatives" / "stimuli").is_dir()
        )
        n_stim_loaded = 0

        for trial_idx, trial in enumerate(trials):
            onset = max(trial["onset_sample"] - 1, 0)  # BIDS uses 1-based
            end = trial["end_sample"] - 1 if trial["end_sample"] else signal.shape[1]
            end = min(end, signal.shape[1])
            duration = end - onset
            if duration <= 0:
                continue

            trial_signal = signal[:n_ch, onset:end].copy()

            # Validate signal (best-effort)
            try:
                val_warnings = validate_signal(
                    trial_signal, f"trial_{trial_idx}",
                    self._pool.config.get("import", {}),
                    signal_unit="uV",
                )
                if val_warnings:
                    logger.warning("Trial %d: %s", trial_idx, val_warnings)
            except Exception:
                pass

            # Annotations
            ann_pfx = f"{run_id}_{trial_idx:04d}"
            annotations = []
            ann_arrays: Dict[str, np.ndarray] = {}

            if trial["attend_lr"]:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_dir_{ann_pfx}",
                    name="attend_direction",
                    value=trial["attend_lr"],
                ))
            if trial["condition"]:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_cond_{ann_pfx}",
                    name="condition",
                    value=trial["condition"],
                ))
            if trial["attend_speaker"]:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_spk_{ann_pfx}",
                    name="attend_speaker",
                    value=trial["attend_speaker"],
                ))
            if trial["stim_file"]:
                annotations.append(TextAnnotation(
                    annotation_id=f"ann_stim_{ann_pfx}",
                    name="stim_file",
                    text_value=trial["stim_file"],
                ))
            if has_eareeg:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_ear_{ann_pfx}",
                    name="has_eareeg",
                    value="true",
                ))
            if trial.get("difficulty") is not None:
                annotations.append(NumericAnnotation(
                    annotation_id=f"ann_diff_{ann_pfx}",
                    name="difficulty_rating",
                    numeric_value=trial["difficulty"],
                ))
            if trial.get("accuracy") is not None:
                annotations.append(NumericAnnotation(
                    annotation_id=f"ann_acc_{ann_pfx}",
                    name="questionnaire_accuracy",
                    numeric_value=trial["accuracy"],
                ))

            # ── Load audio stimulus envelopes ──
            if has_stimuli and trial.get("stim_file"):
                target_path, masker_path = _resolve_stimulus_paths(
                    bids_root, trial["stim_file"],
                )

                if target_path is not None:
                    target_data = _load_stimulus_envelope(target_path)
                    if target_data is not None:
                        env = target_data["envelope"]
                        stim_fs = target_data["fs"]
                        ann_arrays["target_envelope"] = env
                        annotations.append(ContinuousAnnotation(
                            annotation_id=f"ann_tenv_{ann_pfx}",
                            name="target_envelope",
                            domain="stimulus",
                            scope="timepoint",
                            data_ref=SignalRef(
                                file_path="__stim_placeholder__",
                                internal_path=f"__placeholder__/annotations/target_envelope",
                                shape=(int(env.shape[0]),),
                            ),
                            data_sampling_rate=stim_fs,
                            alignment_method="trigger_locked",
                            custom_fields={
                                "stimulus_type": "target",
                                "source_file": target_path.name,
                            },
                        ))
                        n_stim_loaded += 1

                if masker_path is not None:
                    masker_data = _load_stimulus_envelope(masker_path)
                    if masker_data is not None:
                        env = masker_data["envelope"]
                        stim_fs = masker_data["fs"]
                        ann_arrays["masker_envelope"] = env
                        annotations.append(ContinuousAnnotation(
                            annotation_id=f"ann_menv_{ann_pfx}",
                            name="masker_envelope",
                            domain="stimulus",
                            scope="timepoint",
                            data_ref=SignalRef(
                                file_path="__stim_placeholder__",
                                internal_path=f"__placeholder__/annotations/masker_envelope",
                                shape=(int(env.shape[0]),),
                            ),
                            data_sampling_rate=stim_fs,
                            alignment_method="trigger_locked",
                            custom_fields={
                                "stimulus_type": "masker",
                                "source_file": masker_path.name,
                            },
                        ))

            atom_id = compute_atom_id(
                dataset_id, sub_id, session_id, run_id, trial_idx,
            )

            atom = Atom(
                atom_id=atom_id,
                dataset_id=dataset_id,
                subject_id=sub_id,
                session_id=session_id,
                run_id=run_id,
                atom_type=AtomType.TRIAL,
                signal_ref=SignalRef(
                    file_path="", internal_path="",
                    shape=(n_ch, duration),
                ),
                temporal=TemporalInfo(
                    onset_sample=onset,
                    duration_samples=duration,
                    onset_seconds=onset / sfreq,
                    duration_seconds=duration / sfreq,
                ),
                channel_ids=ch_names,
                n_channels=n_ch,
                sampling_rate=sfreq,
                signal_unit="uV",
                annotations=annotations,
            )
            atoms.append((atom, trial_signal, ann_arrays or None))

        if has_stimuli:
            logger.info(
                "Loaded %d stimulus envelopes for %s",
                n_stim_loaded, sub_id,
            )

        return atoms

    def _create_continuous_atom(
        self,
        signal: np.ndarray,
        channel_infos: List[ChannelInfo],
        sfreq: float,
        dataset_id: str,
        sub_id: str,
        session_id: str,
        run_id: str,
        task: str,
    ) -> Tuple[Atom, np.ndarray]:
        """Create one continuous Atom for rest/tonestimuli."""
        ch_names = [ci.name for ci in channel_infos]
        n_ch = len(channel_infos)
        n_samples = signal.shape[1]

        atom_id = compute_atom_id(dataset_id, sub_id, session_id, run_id, 0)

        atom = Atom(
            atom_id=atom_id,
            dataset_id=dataset_id,
            subject_id=sub_id,
            session_id=session_id,
            run_id=run_id,
            atom_type=AtomType.CONTINUOUS_SEGMENT,
            signal_ref=SignalRef(
                file_path="", internal_path="",
                shape=(n_ch, n_samples),
            ),
            temporal=TemporalInfo(
                onset_sample=0,
                duration_samples=n_samples,
                onset_seconds=0.0,
                duration_seconds=n_samples / sfreq,
            ),
            channel_ids=ch_names,
            n_channels=n_ch,
            sampling_rate=sfreq,
            signal_unit="uV",
            annotations=[
                CategoricalAnnotation(
                    annotation_id=f"ann_task_{run_id}_0000",
                    name="task", value=task,
                ),
            ],
        )
        return atom, signal[:n_ch, :]


# Auto-register
register_importer("snhl_aad", SNHLAADImporter)
