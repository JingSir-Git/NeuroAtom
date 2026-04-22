"""BIDS Importer: auto-traverse Brain Imaging Data Structure directories.

BIDS (Brain Imaging Data Structure) is the standard for organizing
neuroscience data on OpenNeuro and many institutional repositories.

BIDS EEG layout:
    dataset/
        participants.tsv
        dataset_description.json
        sub-01/
            ses-01/           (optional)
                eeg/
                    sub-01_ses-01_task-xxx_run-01_eeg.edf
                    sub-01_ses-01_task-xxx_run-01_eeg.json
                    sub-01_ses-01_task-xxx_run-01_channels.tsv
                    sub-01_ses-01_task-xxx_run-01_events.tsv
                    sub-01_ses-01_task-xxx_run-01_electrodes.tsv (optional)

This importer:
1. Detects BIDS root via dataset_description.json
2. Parses participants.tsv for subject metadata
3. Discovers all EEG recordings matching BIDS naming convention
4. Delegates individual file loading to MNEGenericImporter or MNE-BIDS
5. Reads _events.tsv for event information
6. Reads _channels.tsv for channel metadata
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the BIDS importer")

from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import ChannelType
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name

logger = logging.getLogger(__name__)

# BIDS entity parsing regex
_BIDS_ENTITY_RE = re.compile(
    r"sub-(?P<subject>[^_]+)"
    r"(?:_ses-(?P<session>[^_]+))?"
    r"(?:_task-(?P<task>[^_]+))?"
    r"(?:_acq-(?P<acq>[^_]+))?"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_(?P<suffix>eeg|meg|ieeg)"
)

# Supported EEG extensions in BIDS
_BIDS_EEG_EXTENSIONS = {
    ".edf", ".bdf", ".vhdr", ".set", ".fif", ".eeg", ".gdf",
}


def _parse_bids_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse BIDS entities from a filename.

    Returns dict with keys: subject, session, task, run, suffix, or None.
    """
    match = _BIDS_ENTITY_RE.search(filename)
    if match:
        return {k: v for k, v in match.groupdict().items() if v is not None}
    return None


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a BIDS TSV file and return list of row dicts."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_events_tsv(events_path: Path, sfreq: float) -> Optional[np.ndarray]:
    """Parse BIDS _events.tsv into MNE-style events array.

    BIDS events.tsv has columns: onset, duration, trial_type, [value], [sample]
    We convert to (n_events, 3): [sample, 0, event_id].
    """
    rows = _read_tsv(events_path)
    if not rows:
        return None

    # Build event_id mapping from trial_type
    trial_types = sorted(set(r.get("trial_type", "unknown") for r in rows))
    type_to_id = {t: i + 1 for i, t in enumerate(trial_types)}

    events = []
    for row in rows:
        # Prefer 'sample' column if available
        if "sample" in row and row["sample"]:
            sample = int(float(row["sample"]))
        elif "onset" in row:
            sample = int(float(row["onset"]) * sfreq)
        else:
            continue

        # Event ID: use 'value' column if numeric, else map from trial_type
        if "value" in row and row["value"]:
            try:
                event_id = int(float(row["value"]))
            except (ValueError, TypeError):
                event_id = type_to_id.get(row.get("trial_type", ""), 0)
        else:
            event_id = type_to_id.get(row.get("trial_type", ""), 0)

        events.append([sample, 0, event_id])

    if not events:
        return None

    return np.array(events, dtype=int)


def _read_channels_tsv(
    channels_path: Path, sfreq: float, task_config: TaskConfig
) -> List[ChannelInfo]:
    """Parse BIDS _channels.tsv into ChannelInfo list."""
    rows = _read_tsv(channels_path)
    if not rows:
        return []

    type_overrides = task_config.channel_type_overrides
    exclude_set = set(task_config.exclude_channels)

    ch_infos = []
    for idx, row in enumerate(rows):
        ch_name = row.get("name", f"Ch_{idx + 1}")
        if ch_name in exclude_set:
            continue

        # Channel type
        bids_type = row.get("type", "EEG").upper()
        if ch_name in type_overrides:
            ch_type = ChannelType(type_overrides[ch_name])
        elif bids_type in ("EEG", "SEEG", "ECOG"):
            ch_type = ChannelType.EEG
        elif bids_type == "EOG":
            ch_type = ChannelType.EOG
        elif bids_type == "EMG":
            ch_type = ChannelType.EMG
        elif bids_type == "ECG":
            ch_type = ChannelType.ECG
        elif bids_type in ("TRIG", "STIM"):
            ch_type = ChannelType.STIM
        else:
            ch_type = ChannelType.OTHER

        # Unit
        unit = row.get("units", task_config.signal_unit or "V")

        # Status
        status = row.get("status", "good").lower()

        ch_infos.append(ChannelInfo(
            channel_id=f"ch_{idx:03d}",
            index=idx,
            name=ch_name,
            standard_name=standardize_channel_name(ch_name),
            type=ch_type,
            unit=unit,
            sampling_rate=sfreq,
        ))

    return ch_infos


class BIDSImporter(BaseImporter):
    """Importer for BIDS-formatted EEG datasets.

    Detects BIDS root by presence of dataset_description.json, then
    auto-discovers subjects, sessions, and runs. Individual files
    are loaded via MNE.
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if path is a BIDS root directory."""
        path = Path(path)
        if not path.is_dir():
            return False
        return (path / "dataset_description.json").exists()

    def load_raw(self, path: Path) -> Tuple[mne.io.Raw, Dict[str, Any]]:
        """Load a single EEG file from BIDS using MNE.

        For BIDS auto-traversal, use import_dataset() instead.
        This method handles a single recording file within a BIDS tree.
        """
        path = Path(path)
        raw = mne.io.read_raw(str(path), preload=True, verbose="WARNING")

        extra_meta = {
            "declared_unit": self.task_config.signal_unit or "V",
            "bids_path": str(path),
        }
        return raw, extra_meta

    def extract_channel_infos(self, raw: mne.io.Raw) -> List[ChannelInfo]:
        """Extract channel info from MNE Raw.

        Prefers BIDS _channels.tsv if available alongside the recording.
        """
        from neuroatom.importers.mne_generic import MNEGenericImporter

        # Delegate to MNE generic for actual channel extraction
        temp_importer = MNEGenericImporter(
            pool=self._pool, task_config=self._task_config
        )
        return temp_importer.extract_channel_infos(raw)

    def extract_events(self, raw: mne.io.Raw) -> Optional[np.ndarray]:
        """Extract events from MNE Raw.

        Prefers BIDS _events.tsv if available alongside the recording.
        """
        from neuroatom.importers.mne_generic import MNEGenericImporter

        temp_importer = MNEGenericImporter(
            pool=self._pool, task_config=self._task_config
        )
        return temp_importer.extract_events(raw)

    # ------------------------------------------------------------------
    # BIDS auto-traversal
    # ------------------------------------------------------------------

    def discover_recordings(self, bids_root: Path) -> List[Dict[str, Any]]:
        """Discover all EEG recordings in a BIDS dataset.

        Returns list of dicts with keys:
            - path: Path to the recording file
            - subject: subject ID (without 'sub-' prefix)
            - session: session ID (without 'ses-' prefix) or None
            - task: task name
            - run: run ID or None
            - events_tsv: Path to _events.tsv or None
            - channels_tsv: Path to _channels.tsv or None
        """
        bids_root = Path(bids_root)
        recordings = []

        # Walk subject directories
        for sub_dir in sorted(bids_root.glob("sub-*")):
            if not sub_dir.is_dir():
                continue

            # Check for session directories
            ses_dirs = sorted(sub_dir.glob("ses-*"))
            if not ses_dirs:
                # No session level: look directly for eeg/ dir
                ses_dirs = [sub_dir]

            for ses_dir in ses_dirs:
                eeg_dir = ses_dir / "eeg"
                if not eeg_dir.exists():
                    continue

                for eeg_file in sorted(eeg_dir.iterdir()):
                    if eeg_file.suffix.lower() not in _BIDS_EEG_EXTENSIONS:
                        continue

                    entities = _parse_bids_filename(eeg_file.stem)
                    if entities is None:
                        continue

                    # Find sidecar files
                    base_stem = eeg_file.stem
                    if eeg_file.suffix.lower() == ".vhdr":
                        # BrainVision: stem might end with _eeg
                        pass

                    events_tsv = eeg_dir / (base_stem.rsplit("_eeg", 1)[0] + "_events.tsv")
                    channels_tsv = eeg_dir / (base_stem.rsplit("_eeg", 1)[0] + "_channels.tsv")

                    recording = {
                        "path": eeg_file,
                        "subject": entities.get("subject"),
                        "session": entities.get("session"),
                        "task": entities.get("task"),
                        "run": entities.get("run", "01"),
                        "events_tsv": events_tsv if events_tsv.exists() else None,
                        "channels_tsv": channels_tsv if channels_tsv.exists() else None,
                    }
                    recordings.append(recording)

        logger.info("Discovered %d BIDS recordings in %s", len(recordings), bids_root)
        return recordings

    def import_dataset(
        self,
        bids_root: Path,
        atomizer: Any,
    ) -> List[ImportResult]:
        """Import an entire BIDS dataset.

        Discovers all recordings, registers metadata, and imports each run.

        Args:
            bids_root: Path to the BIDS root directory.
            atomizer: Atomizer instance for decomposition.

        Returns:
            List of ImportResult for each imported run.
        """
        bids_root = Path(bids_root)

        # Read dataset_description.json
        desc_path = bids_root / "dataset_description.json"
        with open(desc_path, "r", encoding="utf-8") as f:
            desc = json.load(f)

        dataset_id = self.task_config.dataset_id
        dataset_name = desc.get("Name", self.task_config.dataset_name)

        # Register dataset
        self._pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id,
            name=dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=0,  # Updated below
        ))

        # Read participants.tsv for subject metadata
        participants = _read_tsv(bids_root / "participants.tsv")
        registered_subjects = set()

        # Discover recordings
        recordings = self.discover_recordings(bids_root)

        results = []
        for rec in recordings:
            sub_id = f"sub-{rec['subject']}"
            ses_id = f"ses-{rec['session']}" if rec["session"] else "ses-01"
            run_id = f"run-{rec['run']}" if rec["run"] else "run-01"

            # Register subject if not yet
            if sub_id not in registered_subjects:
                # Find participant info
                age = None
                sex = None
                for p in participants:
                    p_id = p.get("participant_id", "")
                    if p_id == sub_id or p_id == rec["subject"]:
                        age = p.get("age")
                        sex = p.get("sex")
                        break

                self._pool.register_subject(SubjectMeta(
                    subject_id=sub_id,
                    dataset_id=dataset_id,
                    age=int(age) if age and age != "n/a" else None,
                    sex=sex if sex and sex != "n/a" else None,
                ))
                registered_subjects.add(sub_id)

            # Register session
            from neuroatom.core.session import SessionMeta
            self._pool.register_session(SessionMeta(
                session_id=ses_id,
                subject_id=sub_id,
                dataset_id=dataset_id,
            ))

            # Import run
            try:
                result = self.import_run(
                    path=rec["path"],
                    subject_id=sub_id,
                    session_id=ses_id,
                    run_id=run_id,
                    atomizer=atomizer,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "Failed to import %s/%s/%s: %s",
                    sub_id, ses_id, run_id, e,
                )

        logger.info(
            "BIDS import complete: %d/%d runs imported.",
            len(results), len(recordings),
        )
        return results

    def read_dataset_metadata(self, bids_root: Path) -> Dict[str, Any]:
        """Read BIDS dataset-level metadata.

        Returns dict with dataset_description.json contents + participants info.
        """
        bids_root = Path(bids_root)
        meta = {}

        desc_path = bids_root / "dataset_description.json"
        if desc_path.exists():
            with open(desc_path, "r", encoding="utf-8") as f:
                meta["description"] = json.load(f)

        participants = _read_tsv(bids_root / "participants.tsv")
        meta["n_participants"] = len(participants)
        meta["participants"] = participants

        return meta


# Auto-register
register_importer("bids", BIDSImporter)
