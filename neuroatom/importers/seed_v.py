"""SEED-V Emotion Recognition Dataset Importer.

Handles the SEED-V dataset: 16 subjects × 3 sessions, Neuroscan .cnt format,
62 EEG channels @ 1000 Hz, 5-class emotion labels (Disgust/Fear/Sad/Neutral/Happy).

Key features:
    - Segments continuous CNT recordings by trial timestamps
    - Maps per-session emotion label order to each trial
    - Supports both raw CNT and pre-extracted DE features
    - Lazy loading for large CNT files (800MB–1GB each)
    - Extracts only EEG channels (excludes VEO, HEO, mastoid refs)

Data layout expected:
    SEED-V-O/
        EEG_raw/
            1_1_20180804.cnt  (subject_session_date.cnt)
            1_2_20180810.cnt
            ...
        EEG_DE_features/
            1_123.npz  (per-subject, all 3 sessions)
        Channel Order.xlsx
        emotion_label_and_stimuli_order.xlsx
        trial_start_end_timestamp.txt
"""

import importlib.resources
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the SEED-V importer")

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


# Emotion label mapping
EMOTION_LABELS = {0: "disgust", 1: "fear", 2: "sad", 3: "neutral", 4: "happy"}

# Channels that are auxiliary (not EEG scalp)
AUX_CHANNELS = {
    "VEO": ChannelType.EOG,
    "HEO": ChannelType.EOG,
    "M1": ChannelType.REF,
    "M2": ChannelType.REF,
}

# Default exclude channels (only mastoid references)
DEFAULT_EXCLUDE = {"M1", "M2"}


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


def _parse_cnt_filename(name: str) -> Optional[Tuple[int, int, str]]:
    """Parse SEED-V CNT filename.

    Format: {subject}_{session}_{date}.cnt → (subject_num, session_num, date)
    """
    m = re.match(r"(\d+)_(\d+)_(\d+)\.cnt$", name)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None


class SEEDVImporter(BaseImporter):
    """Importer for the SEED-V Emotion Recognition Dataset.

    Supports:
        - Raw CNT import with trial segmentation by timestamps
        - Per-session emotion label mapping
        - EEG channel selection (62 EEG from 66 total)
        - Lazy loading for large CNT files
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

        # Load trial timestamps and emotion order from task config
        self._trial_ts = task_config.data.get("trial_timestamps", {})
        self._emotion_order = task_config.data.get("emotion_order", {})
        self._exclude = set(task_config.exclude_channels) or DEFAULT_EXCLUDE

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if path is a SEED-V dataset."""
        path = Path(path)
        if not path.is_dir():
            return False
        # Look for EEG_raw/ dir with .cnt files
        raw_dir = path / "EEG_raw"
        if raw_dir.exists():
            cnts = list(raw_dir.glob("*.cnt"))
            return len(cnts) > 0
        return False

    def load_raw(self, path):
        raise NotImplementedError("Use import_subject() instead.")

    def extract_channel_infos(self, raw):
        raise NotImplementedError("Use _build_channel_infos() instead.")

    def extract_events(self, raw):
        return None

    # ------------------------------------------------------------------
    # Channel info builder
    # ------------------------------------------------------------------

    def _build_channel_infos(
        self,
        raw_ch_names: List[str],
        srate: float,
    ) -> Tuple[List[ChannelInfo], List[int]]:
        """Build ChannelInfo for EEG + auxiliary channels.

        VEO/HEO are included as EOG channels. M1/M2 (mastoid refs) are excluded.
        Standard 10-20 electrode coordinates are added for EEG channels.

        Returns:
            (channel_infos, selected_indices) — list of ChannelInfo and
            indices of selected channels in the raw data array.
        """
        std_coords = _get_standard_coords()
        ch_infos = []
        selected_indices = []

        for idx, raw_name in enumerate(raw_ch_names):
            if raw_name in self._exclude:
                continue

            # Determine channel type
            if raw_name in AUX_CHANNELS:
                ch_type = AUX_CHANNELS[raw_name]
            else:
                ch_type = ChannelType.EEG

            std_name = standardize_channel_name(raw_name) if ch_type == ChannelType.EEG else None

            # Look up electrode coordinates for EEG channels
            location = None
            if ch_type == ChannelType.EEG:
                for try_name in [raw_name, raw_name.capitalize(),
                                 raw_name.upper(), std_name or ""]:
                    if try_name in std_coords:
                        pos = std_coords[try_name]
                        location = ElectrodeLocation(
                            x=pos["x"], y=pos["y"], z=pos["z"],
                            coordinate_system="MNI",
                            coordinate_units="m",
                        )
                        break

            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{len(ch_infos):03d}",
                index=len(ch_infos),
                name=raw_name,
                standard_name=std_name,
                type=ch_type,
                unit="V",
                sampling_rate=srate,
                status=ChannelStatus.GOOD,
                location=location,
            ))
            selected_indices.append(idx)

        return ch_infos, selected_indices

    # ------------------------------------------------------------------
    # Get trial timestamps for a session
    # ------------------------------------------------------------------

    def _get_session_info(self, session_num: int) -> Tuple[List[int], List[int], List[int]]:
        """Get trial start/end timestamps and emotion labels for a session.

        Returns:
            (starts, ends, emotion_codes)
        """
        sess_key = f"session_{session_num}"
        ts = self._trial_ts.get(sess_key, {})
        starts = ts.get("start", [])
        ends = ts.get("end", [])

        emo_key = f"session_{session_num}"
        emotions = self._emotion_order.get(emo_key, [])

        if not starts or not ends or not emotions:
            raise ValueError(
                f"Missing timestamps or emotion order for session {session_num}. "
                f"Check task config."
            )

        if len(starts) != len(ends) or len(starts) != len(emotions):
            raise ValueError(
                f"Session {session_num}: mismatched lengths: "
                f"starts={len(starts)}, ends={len(ends)}, emotions={len(emotions)}"
            )

        return starts, ends, emotions

    # ------------------------------------------------------------------
    # Import one session (one CNT file)
    # ------------------------------------------------------------------

    def _import_session(
        self,
        cnt_path: Path,
        session_num: int,
        dataset_id: str,
        subject_id: str,
        max_trials: Optional[int] = None,
    ) -> Tuple[List[Atom], List[ChannelInfo], List[str]]:
        """Import one session's trials from a CNT file.

        Returns:
            (atoms, channel_infos, warnings)
        """
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P

        session_id = f"ses-{session_num:02d}"
        run_id = f"run-{session_num:02d}"

        # Get trial info
        starts, ends, emotions = self._get_session_info(session_num)

        if max_trials is not None:
            starts = starts[:max_trials]
            ends = ends[:max_trials]
            emotions = emotions[:max_trials]

        # Load CNT lazily (header only first)
        raw = mne.io.read_raw_cnt(str(cnt_path), preload=False, verbose=False)
        srate = raw.info["sfreq"]
        total_samples = len(raw.times)

        # Build channel info (EEG only)
        ch_infos, eeg_indices = self._build_channel_infos(raw.ch_names, srate)
        channel_ids = [ch.channel_id for ch in ch_infos]
        n_channels = len(ch_infos)

        logger.info(
            "Loading session %d from %s: %d EEG ch (of %d total), %.0f Hz, %.0fs",
            session_num, cnt_path.name, n_channels, len(raw.ch_names),
            srate, total_samples / srate,
        )

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
                for trial_idx, (start_s, end_s, emo_code) in enumerate(
                    zip(starts, ends, emotions)
                ):
                    emo_code = int(emo_code)
                    emotion = EMOTION_LABELS.get(emo_code, f"unknown_{emo_code}")

                    # Convert seconds to samples
                    start_sample = int(start_s * srate)
                    end_sample = int(end_s * srate)

                    if end_sample > total_samples:
                        logger.warning(
                            "Trial %d: end_sample %d > total %d, clipping.",
                            trial_idx, end_sample, total_samples,
                        )
                        end_sample = total_samples

                    n_trial_samples = end_sample - start_sample

                    # Load only this segment
                    raw_segment = raw.get_data(
                        start=start_sample,
                        stop=end_sample,
                    )
                    # Select EEG channels
                    signal = raw_segment[eeg_indices, :]  # (n_eeg, n_samples)

                    # Annotations
                    annotations = [
                        CategoricalAnnotation(
                            annotation_id=f"ann_emotion_{run_id}_{trial_idx:04d}",
                            name="emotion",
                            value=emotion,
                        ),
                        NumericAnnotation(
                            annotation_id=f"ann_emo_code_{run_id}_{trial_idx:04d}",
                            name="emotion_code",
                            numeric_value=float(emo_code),
                        ),
                        CategoricalAnnotation(
                            annotation_id=f"ann_session_{run_id}_{trial_idx:04d}",
                            name="session",
                            value=f"session_{session_num}",
                        ),
                    ]

                    atom_id = compute_atom_id(
                        dataset_id=dataset_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        run_id=run_id,
                        onset_sample=start_sample,
                    )

                    temporal = TemporalInfo(
                        onset_sample=start_sample,
                        onset_seconds=start_s,
                        duration_samples=n_trial_samples,
                        duration_seconds=float(end_s - start_s),
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
                            shape=(n_channels, n_trial_samples),
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
                                        "format": "seed_v_cnt",
                                        "source_file": cnt_path.name,
                                        "session_num": session_num,
                                        "trial_index": trial_idx,
                                        "emotion": emotion,
                                        "emotion_code": emo_code,
                                        "start_second": start_s,
                                        "end_second": end_s,
                                        "signal_unit": "V",
                                    },
                                ),
                            ],
                            is_raw=True,
                            version_tag="raw",
                        ),
                        custom_fields={
                            "session_num": session_num,
                            "emotion": emotion,
                            "emotion_code": emo_code,
                            "trial_duration_s": end_s - start_s,
                        },
                    )

                    # Validate (use first 10s to keep fast)
                    val_signal = signal[:, :min(10000, signal.shape[1])].astype(np.float32)
                    warnings = validate_signal(
                        signal=val_signal,
                        atom_id=atom_id,
                        config=self.pool.config.get("import", {}),
                    )
                    all_warnings.extend(warnings)

                    # Write signal
                    signal_ref = shard_mgr.write_atom_signal(atom_id, signal)
                    atom.signal_ref = signal_ref

                    writer.write_atom(atom)
                    atoms.append(atom)

                    logger.debug(
                        "  Trial %d: %s (%ds, %d samples)",
                        trial_idx, emotion, end_s - start_s, n_trial_samples,
                    )

        logger.info(
            "Imported %s session %d: %d trials × %d ch @ %.0f Hz",
            subject_id, session_num, len(atoms), n_channels, srate,
        )

        return atoms, ch_infos, all_warnings

    # ------------------------------------------------------------------
    # Main entry: import one subject (all or selected sessions)
    # ------------------------------------------------------------------

    def import_subject(
        self,
        dataset_dir: Path,
        subject_num: int,
        sessions: Optional[List[int]] = None,
        max_trials: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import sessions for a subject from the SEED-V dataset.

        Args:
            dataset_dir: Root directory of SEED-V (contains EEG_raw/)
            subject_num: Subject number (1-16)
            sessions: List of session numbers to import (default: all [1,2,3])
            max_trials: Maximum trials per session (for testing)

        Returns:
            List of ImportResult, one per session
        """
        dataset_dir = Path(dataset_dir)
        raw_dir = dataset_dir / "EEG_raw"
        dataset_id = self.task_config.dataset_id
        subject_id = f"sub-{subject_num:02d}"

        self.pool.ensure_dataset(dataset_id, name=self.task_config.dataset_name)
        self.pool.ensure_subject(dataset_id, subject_id)

        if sessions is None:
            sessions = [1, 2, 3]

        # Find CNT files for this subject
        cnt_files: Dict[int, Path] = {}
        for cnt in raw_dir.glob(f"{subject_num}_*.cnt"):
            parsed = _parse_cnt_filename(cnt.name)
            if parsed and parsed[0] == subject_num:
                cnt_files[parsed[1]] = cnt

        results = []
        for sess_num in sessions:
            if sess_num not in cnt_files:
                logger.warning("No CNT file for subject %d session %d", subject_num, sess_num)
                continue

            atoms, ch_infos, warnings = self._import_session(
                cnt_path=cnt_files[sess_num],
                session_num=sess_num,
                dataset_id=dataset_id,
                subject_id=subject_id,
                max_trials=max_trials,
            )

            if atoms:
                from neuroatom.core.run import RunMeta
                run_meta = RunMeta(
                    run_id=f"run-{sess_num:02d}",
                    session_id=f"ses-{sess_num:02d}",
                    subject_id=subject_id,
                    dataset_id=dataset_id,
                    run_index=sess_num,
                    task_type=self.task_config.task_type,
                    n_trials=len(atoms),
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
            "Subject %s: imported %d sessions, %d total trials.",
            subject_id, len(results), total_atoms,
        )

        return results


# Auto-register
register_importer("seed_v", SEEDVImporter)
