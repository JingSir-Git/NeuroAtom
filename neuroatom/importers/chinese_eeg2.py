"""ChineseEEG-2 Importer: sentence-level EEG for Chinese language decoding.

ChineseEEG-2 (Chen et al., 2024) is a BIDS-formatted dataset with two tasks:
- **Passive Listening**: 10 subjects listen to Chinese novel audiobooks (1000 Hz)
- **Reading Aloud**: 4 subjects read the same novels aloud (250 Hz)

Both tasks use 128-channel EGI HydroCel-128 with CapTrak coordinates.
Events mark sentence boundaries (ROWS=start, ROWE=end), enabling
sentence-level epoch extraction for neural language decoding.

Run naming convention: ``run-{rep}{chapter}`` where rep=1/2 (repetition)
and chapter=1-13+ (novel chapter). Session = novel name (littleprince/garnettdream).

This importer extends the generic ``BIDSImporter`` with:
1. Sentence-level atomization from ROWS→ROWE event pairs
2. Novel/chapter/sentence metadata in custom_fields
3. Electrode coordinate extraction from CapTrak electrodes.tsv
4. Bad channel info from preprocessed derivatives (optional)

Usage::

    from neuroatom import Pool, Indexer
    from neuroatom.importers.chinese_eeg2 import ChineseEEG2Importer

    pool = Pool.create("my_pool")
    importer = ChineseEEG2Importer(pool, task="listening")
    results = importer.import_dataset("C:/Data/ChineseEEG-2/PassiveListening")
"""

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the ChineseEEG-2 importer")

from neuroatom.core.annotation import (
    CategoricalAnnotation,
    ContinuousAnnotation,
    NumericAnnotation,
    TextAnnotation,
)
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import AtomType, ChannelType
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.session import SessionMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.core.run import RunMeta
from neuroatom.importers.base import ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage import paths as P
from neuroatom.storage.metadata_store import AtomJSONLWriter
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

# Run ID format: run-{rep}{chapter}, e.g. run-11 = rep1 ch1, run-213 = rep2 ch13
_RUN_RE = re.compile(r"run-(\d)(\d+)")


def _parse_run_id(run_str: str) -> Tuple[int, int]:
    """Parse repetition and chapter from run ID like 'run-11' → (1, 1)."""
    m = _RUN_RE.match(run_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1, 0


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a BIDS TSV file (handles UTF-8 BOM)."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def _read_electrodes_tsv(path: Path) -> Dict[str, ElectrodeLocation]:
    """Read electrodes.tsv and return name → ElectrodeLocation mapping."""
    rows = _read_tsv(path)
    electrodes = {}
    for row in rows:
        name = row.get("name", "")
        try:
            x = float(row.get("x", 0))
            y = float(row.get("y", 0))
            z = float(row.get("z", 0))
            electrodes[name] = ElectrodeLocation(
                x=x,
                y=y,
                z=z,
                coordinate_system="CapTrak",
            )
        except (ValueError, TypeError):
            continue
    return electrodes


def _extract_sentence_epochs(
    events_tsv: Path,
    sfreq: float,
    start_marker: str = "ROWS",
    end_marker: str = "ROWE",
) -> List[Dict[str, Any]]:
    """Extract sentence epochs from ROWS→ROWE event pairs.

    Returns list of dicts: {onset_sample, offset_sample, duration_samples,
                            onset_sec, offset_sec, sentence_index}
    """
    rows = _read_tsv(events_tsv)
    if not rows:
        return []

    # Separate start and end markers
    starts = []
    ends = []
    for row in rows:
        trial_type = row.get("trial_type", "")
        if trial_type == start_marker:
            sample = int(float(row.get("sample", float(row.get("onset", 0)) * sfreq)))
            onset = float(row.get("onset", sample / sfreq))
            starts.append({"sample": sample, "onset": onset})
        elif trial_type == end_marker:
            sample = int(float(row.get("sample", float(row.get("onset", 0)) * sfreq)))
            onset = float(row.get("onset", sample / sfreq))
            ends.append({"sample": sample, "onset": onset})

    # Pair: each ROWS is followed by the next ROWE
    epochs = []
    end_idx = 0
    for s_idx, start in enumerate(starts):
        # Find the first ROWE after this ROWS
        while end_idx < len(ends) and ends[end_idx]["sample"] <= start["sample"]:
            end_idx += 1
        if end_idx >= len(ends):
            break

        onset_sample = start["sample"]
        offset_sample = ends[end_idx]["sample"]
        duration_samples = offset_sample - onset_sample

        if duration_samples <= 0:
            continue

        epochs.append({
            "onset_sample": onset_sample,
            "offset_sample": offset_sample,
            "duration_samples": duration_samples,
            "onset_sec": start["onset"],
            "offset_sec": ends[end_idx]["onset"],
            "sentence_index": len(epochs),
        })
        end_idx += 1

    return epochs


# Novel name → audio embedding folder prefix
_NOVEL_TO_AUDIO_PREFIX: Dict[str, str] = {
    "littleprince": "xiaowangzi",
    "garnettdream": "langwangmeng",
}

# ReadingAloud sub IDs → audio embedding suffix
_SUBJECT_TO_AUDIO_SUFFIX: Dict[str, str] = {
    "f1": "female_subject_1",
    "f2": "female_subject_2",
    "m1": "male_subject_1",
    "m2": "male_subject_2",
}


def _load_text_embeddings_novel(
    materials_root: Path, novel: str,
) -> Optional[np.ndarray]:
    """Load global BERT text embeddings for an entire novel.

    Returns array of shape (N_sentences, 768) or None.
    """
    npy_path = materials_root / "text_embedding" / f"text_embeddings_{novel.lower()}.npy"
    if not npy_path.exists():
        return None
    try:
        return np.load(str(npy_path)).astype(np.float32)
    except Exception as exc:
        logger.debug("Could not load text embeddings from %s: %s", npy_path, exc)
        return None


def _load_audio_embedding_chapter(
    materials_root: Path,
    novel: str,
    chapter_idx: int,
    subject_id: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load audio embedding for one chapter (0-based chapter_idx).

    For ReadingAloud, subject_id (e.g. 'f1') selects the speaker folder.
    For PassiveListening, uses the first available folder for the novel.

    Returns flat array of shape (1024,) or None.
    """
    novel_prefix = _NOVEL_TO_AUDIO_PREFIX.get(novel.lower())
    if novel_prefix is None:
        return None

    ae_root = materials_root / "audio_embedding"
    if not ae_root.exists():
        return None

    # Determine suffix: ReadingAloud subject or first available
    if subject_id is not None:
        suffix = _SUBJECT_TO_AUDIO_SUFFIX.get(subject_id)
        if suffix:
            candidate_dirs = [ae_root / f"{novel_prefix}_{suffix}"]
        else:
            candidate_dirs = []
    else:
        # PassiveListening: pick the first matching dir (female_subject_1)
        candidate_dirs = [
            ae_root / f"{novel_prefix}_female_subject_1",
            ae_root / f"{novel_prefix}_male_subject_1",
        ]

    for ae_dir in candidate_dirs:
        npy_path = ae_dir / f"audio_embedding_audio_{chapter_idx}.npy"
        if npy_path.exists():
            try:
                arr = np.load(str(npy_path)).astype(np.float32)
                return arr.ravel()  # flatten (1, 1024) → (1024,)
            except Exception as exc:
                logger.debug("Could not load audio embedding %s: %s", npy_path, exc)
    return None


class ChineseEEG2Importer:
    """Importer for ChineseEEG-2 BIDS dataset.

    Extracts sentence-level EEG epochs from passive listening or reading aloud
    tasks, paired by (session, run) = (novel, chapter).

    Usage::

        pool = Pool.create("ceeg2_pool")
        importer = ChineseEEG2Importer(pool, task="listening")
        results = importer.import_dataset(
            Path("C:/Data/ChineseEEG-2/PassiveListening")
        )
        print(f"Imported {sum(r.n_atoms for r in results)} sentence atoms")

    Args:
        pool: Target Pool instance.
        task: ``"listening"`` or ``"reading"`` — determines task config.
        use_preprocessed: If True, import from derivatives/preprocessed.
        max_shard_mb: Max HDF5 shard size in MB.
        compression: HDF5 compression algorithm.
    """

    def __init__(
        self,
        pool: Pool,
        task: str = "listening",
        use_preprocessed: bool = False,
        max_shard_mb: int = 200,
        compression: str = "gzip",
        materials_root: Optional[Path] = None,
    ):
        self.pool = pool
        self.task = task
        self.use_preprocessed = use_preprocessed
        self.max_shard_mb = max_shard_mb
        self.compression = compression
        self.materials_root = Path(materials_root) if materials_root else None

        if task == "listening":
            self.dataset_id = "chinese_eeg2_listening"
            self.dataset_name = "ChineseEEG-2 Passive Listening"
            self.task_name = "lis"
            self.start_marker = "ROWS"
            self.end_marker = "ROWE"
        elif task == "reading":
            self.dataset_id = "chinese_eeg2_reading"
            self.dataset_name = "ChineseEEG-2 Reading Aloud"
            self.task_name = "reading"
            self.start_marker = "ROWS"
            self.end_marker = "ROWE"
        else:
            raise ValueError(f"task must be 'listening' or 'reading', got '{task}'")

    @staticmethod
    def detect(path: Path) -> bool:
        """Detect if path is a ChineseEEG-2 BIDS directory."""
        path = Path(path)
        desc = path / "dataset_description.json"
        if not desc.exists():
            return False
        try:
            with open(desc, "r", encoding="utf-8") as f:
                data = json.load(f)
            return "Chinese Novel" in data.get("Name", "")
        except Exception:
            return False

    def discover_recordings(self, bids_root: Path) -> List[Dict[str, Any]]:
        """Discover all EEG recordings in the BIDS tree.

        Returns:
            List of recording dicts with path, subject, session, run, and
            sidecar file paths.
        """
        bids_root = Path(bids_root)
        if self.use_preprocessed:
            bids_root = bids_root / "derivatives" / "preprocessed"

        recordings = []
        for sub_dir in sorted(bids_root.glob("sub-*")):
            if not sub_dir.is_dir():
                continue
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                eeg_dir = ses_dir / "eeg"
                if not eeg_dir.exists():
                    continue

                for vhdr_file in sorted(eeg_dir.glob("*.vhdr")):
                    # Extract BIDS entities
                    stem = vhdr_file.stem  # e.g. sub-01_ses-littleprince_task-lis_run-11_eeg
                    entities = self._parse_entities(stem)
                    if not entities:
                        continue

                    # Filter by task
                    if entities.get("task") != self.task_name:
                        continue

                    base = stem.rsplit("_eeg", 1)[0]
                    events_tsv = eeg_dir / f"{base}_events.tsv"
                    channels_tsv = eeg_dir / f"{base}_channels.tsv"
                    electrodes_glob = list(eeg_dir.glob("*_electrodes.tsv"))

                    # Bad channels from preprocessed
                    bad_ch_path = eeg_dir / f"{base}_bad_channels.json"

                    recordings.append({
                        "path": vhdr_file,
                        "subject": entities["subject"],
                        "session": entities.get("session"),
                        "run": entities.get("run", "01"),
                        "task": entities.get("task"),
                        "events_tsv": events_tsv if events_tsv.exists() else None,
                        "channels_tsv": channels_tsv if channels_tsv.exists() else None,
                        "electrodes_tsv": electrodes_glob[0] if electrodes_glob else None,
                        "bad_channels_json": bad_ch_path if bad_ch_path.exists() else None,
                    })

        logger.info(
            "Discovered %d %s recordings in %s",
            len(recordings), self.task, bids_root,
        )
        return recordings

    def _parse_entities(self, stem: str) -> Optional[Dict[str, str]]:
        """Parse BIDS entities from filename stem."""
        parts = stem.split("_")
        entities = {}
        for part in parts:
            if part.startswith("sub-"):
                entities["subject"] = part[4:]
            elif part.startswith("ses-"):
                entities["session"] = part[4:]
            elif part.startswith("task-"):
                entities["task"] = part[5:]
            elif part.startswith("run-"):
                entities["run"] = part[4:]
        return entities if "subject" in entities else None

    def import_dataset(
        self,
        bids_root: Path,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        max_runs: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import the ChineseEEG-2 dataset with sentence-level atomization.

        Args:
            bids_root: Path to PassiveListening/ or ReadingAloud/ directory.
            subjects: Filter by subject IDs (e.g. ["01", "02"]). None = all.
            sessions: Filter by session (e.g. ["littleprince"]). None = all.
            max_runs: Max runs to import (for testing). None = all.

        Returns:
            List of ImportResult, one per run.
        """
        bids_root = Path(bids_root)
        t0 = time.time()

        # Register dataset
        self.pool.register_dataset(DatasetMeta(
            dataset_id=self.dataset_id,
            name=self.dataset_name,
            task_types=["language_comprehension" if self.task == "listening"
                        else "language_production"],
            n_subjects=0,
        ))

        # Read participants
        participants_path = bids_root / "participants.tsv"
        if self.use_preprocessed:
            pp = bids_root / "derivatives" / "preprocessed" / "participants.tsv"
            if pp.exists():
                participants_path = pp
        participants = _read_tsv(participants_path)

        # Discover
        recordings = self.discover_recordings(bids_root)
        if subjects:
            recordings = [r for r in recordings if r["subject"] in subjects]
        if sessions:
            recordings = [r for r in recordings if r.get("session") in sessions]
        if max_runs:
            recordings = recordings[:max_runs]

        # Register subjects
        registered_subjects = set()
        for rec in recordings:
            sub_id = f"sub-{rec['subject']}"
            if sub_id not in registered_subjects:
                p_info = next(
                    (p for p in participants
                     if p.get("participant_id", "") in (sub_id, rec["subject"])),
                    {},
                )
                self.pool.register_subject(SubjectMeta(
                    subject_id=sub_id,
                    dataset_id=self.dataset_id,
                    age=self._safe_int(p_info.get("age")),
                    sex=p_info.get("sex") if p_info.get("sex") != "n/a" else None,
                ))
                registered_subjects.add(sub_id)

        # Locate materials root (text/audio embeddings)
        materials_root = self.materials_root
        if materials_root is None:
            # Try sibling of bids_root: ChineseEEG-2/materials&embeddings
            candidate = bids_root.parent / "materials&embeddings"
            if candidate.exists():
                materials_root = candidate

        # Pre-load text embeddings per novel (cached across runs)
        _text_emb_cache: Dict[str, Optional[np.ndarray]] = {}

        # Compute global sentence offsets per (subject, session, run)
        # Group recordings by (subject, session) to find ordered runs
        ses_run_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
        if materials_root is not None:
            for rec in recordings:
                ses_key = (rec["subject"], rec.get("session", ""))
                if ses_key not in ses_run_counts:
                    ses_run_counts[ses_key] = {}
                n_sents = 0
                if rec.get("events_tsv"):
                    ep = _extract_sentence_epochs(
                        rec["events_tsv"],
                        1000.0,  # approximate; exact sfreq not needed for counting
                        self.start_marker,
                        self.end_marker,
                    )
                    n_sents = len(ep)
                ses_run_counts[ses_key][rec["run"]] = n_sents

        # Build cumulative offsets: sort runs by (repetition, chapter) numerically
        global_offsets: Dict[Tuple[str, str, str], int] = {}
        for (sub, ses), run_counts in ses_run_counts.items():
            sorted_runs = sorted(
                run_counts.keys(),
                key=lambda r: _parse_run_id(f"run-{r}"),
            )
            offset = 0
            for run in sorted_runs:
                global_offsets[(sub, ses, run)] = offset
                offset += run_counts[run]

        # Import each run
        results = []
        for rec in recordings:
            try:
                novel = rec.get("session", "unknown")
                # Get cached text embeddings for this novel
                text_emb = None
                if materials_root is not None and novel != "unknown":
                    if novel not in _text_emb_cache:
                        _text_emb_cache[novel] = _load_text_embeddings_novel(
                            materials_root, novel,
                        )
                    text_emb = _text_emb_cache[novel]

                g_offset = global_offsets.get(
                    (rec["subject"], rec.get("session", ""), rec["run"]), 0
                )

                result = self._import_run(
                    rec, bids_root,
                    materials_root=materials_root,
                    text_embeddings=text_emb,
                    global_sent_offset=g_offset,
                )
                results.append(result)
                logger.info(
                    "Imported %s/%s/%s: %d sentence atoms",
                    f"sub-{rec['subject']}", f"ses-{rec.get('session', '?')}",
                    f"run-{rec['run']}", result.n_atoms,
                )
            except Exception as e:
                logger.error(
                    "Failed run-%s for sub-%s: %s",
                    rec["run"], rec["subject"], e, exc_info=True,
                )

        elapsed = time.time() - t0
        total_atoms = sum(r.n_atoms for r in results)
        logger.info(
            "ChineseEEG-2 %s import: %d runs, %d atoms in %.1fs",
            self.task, len(results), total_atoms, elapsed,
        )
        return results

    def _import_run(
        self,
        rec: Dict[str, Any],
        bids_root: Path,
        materials_root: Optional[Path] = None,
        text_embeddings: Optional[np.ndarray] = None,
        global_sent_offset: int = 0,
    ) -> ImportResult:
        """Import a single run: load EEG, extract sentence epochs, store atoms."""
        sub_id = f"sub-{rec['subject']}"
        ses_id = f"ses-{rec['session']}" if rec.get("session") else "ses-01"
        run_id = f"run-{rec['run']}"
        repetition, chapter = _parse_run_id(run_id)
        novel = rec.get("session", "unknown")
        # chapter is 1-based from run ID; audio embedding uses 0-based index
        audio_chapter_idx = chapter - 1

        # Load audio embedding for this chapter (if materials available)
        audio_emb: Optional[np.ndarray] = None
        if materials_root is not None and novel != "unknown":
            subject_raw = rec["subject"]  # e.g. 'f1', 'm2' for ReadingAloud
            sub_id_for_audio = (
                subject_raw if subject_raw in _SUBJECT_TO_AUDIO_SUFFIX else None
            )
            audio_emb = _load_audio_embedding_chapter(
                materials_root, novel, audio_chapter_idx, sub_id_for_audio,
            )

        # Register session
        sfreq_nominal = 1000.0 if self.task == "listening" else 250.0
        self.pool.register_session(SessionMeta(
            session_id=ses_id,
            subject_id=sub_id,
            dataset_id=self.dataset_id,
            sampling_rate=sfreq_nominal,
            device_manufacturer="EGI",
            device_model="Geodesic EEG 400",
            placement_scheme="GSN-HydroCel-128",
            line_freq=50.0,
            recording_type="continuous",
        ))

        # Load EEG
        raw = mne.io.read_raw_brainvision(
            str(rec["path"]), preload=True, verbose="WARNING"
        )
        sfreq = raw.info["sfreq"]
        n_total = raw.n_times

        # Channel info
        ch_infos, electrodes = self._extract_channels(rec, sfreq)
        eeg_ch_names = [ci.name for ci in ch_infos if ci.type == ChannelType.EEG]
        eeg_picks = mne.pick_channels(raw.ch_names, eeg_ch_names, ordered=True)

        # Bad channels
        bad_channels = []
        if rec.get("bad_channels_json"):
            try:
                with open(rec["bad_channels_json"], "r") as f:
                    bad_data = json.load(f)
                bad_channels = bad_data.get("bad channels", [])
            except Exception:
                pass

        # Extract sentence epochs
        if rec.get("events_tsv") is None:
            logger.warning("No events.tsv for %s/%s/%s", sub_id, ses_id, run_id)
            run_meta = RunMeta(
                run_id=run_id, session_id=ses_id, subject_id=sub_id,
                dataset_id=self.dataset_id, task_type=self.task,
            )
            return ImportResult(
                atoms=[], run_meta=run_meta, channel_infos=ch_infos,
                n_atoms=0, warnings=[],
            )

        epochs = _extract_sentence_epochs(
            rec["events_tsv"], sfreq,
            self.start_marker, self.end_marker,
        )

        if not epochs:
            logger.warning("No sentence epochs found in %s/%s/%s", sub_id, ses_id, run_id)
            run_meta = RunMeta(
                run_id=run_id, session_id=ses_id, subject_id=sub_id,
                dataset_id=self.dataset_id, task_type=self.task,
            )
            return ImportResult(
                atoms=[], run_meta=run_meta, channel_infos=ch_infos,
                n_atoms=0, warnings=["no_sentence_epochs"],
            )

        # Store atoms
        channel_ids = [ci.channel_id for ci in ch_infos if ci.type == ChannelType.EEG]
        n_channels = len(channel_ids)
        all_warnings = []
        atoms = []

        with ShardManager(
            pool_root=self.pool.root,
            dataset_id=self.dataset_id,
            subject_id=sub_id,
            session_id=ses_id,
            run_id=run_id,
            max_shard_size_mb=self.max_shard_mb,
            compression=self.compression,
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, self.dataset_id, sub_id, ses_id, run_id
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                for epoch in epochs:
                    onset = epoch["onset_sample"]
                    offset = min(epoch["offset_sample"], n_total)
                    duration = offset - onset
                    if duration <= 0:
                        continue

                    # Extract signal segment
                    signal = raw.get_data(
                        picks=eeg_picks, start=onset, stop=offset,
                    )  # (n_eeg, n_samples)

                    atom_id = compute_atom_id(
                        dataset_id=self.dataset_id,
                        subject_id=sub_id,
                        session_id=ses_id,
                        run_id=run_id,
                        onset_sample=onset,
                    )

                    sent_idx = epoch["sentence_index"]
                    global_sent_idx = global_sent_offset + sent_idx

                    # Companion arrays for embedding annotations
                    ann_arrays: Dict[str, np.ndarray] = {}

                    # Annotations
                    annotations = [
                        CategoricalAnnotation(
                            annotation_id=f"ann_novel_{run_id}_{sent_idx:04d}",
                            name="novel",
                            value=novel,
                        ),
                        NumericAnnotation(
                            annotation_id=f"ann_chapter_{run_id}_{sent_idx:04d}",
                            name="chapter",
                            numeric_value=float(chapter),
                        ),
                        NumericAnnotation(
                            annotation_id=f"ann_rep_{run_id}_{sent_idx:04d}",
                            name="repetition",
                            numeric_value=float(repetition),
                        ),
                        NumericAnnotation(
                            annotation_id=f"ann_sent_{run_id}_{sent_idx:04d}",
                            name="sentence_index",
                            numeric_value=float(sent_idx),
                        ),
                    ]

                    # BERT text embedding (global sentence index into novel)
                    if (
                        text_embeddings is not None
                        and global_sent_idx < text_embeddings.shape[0]
                    ):
                        emb_vec = text_embeddings[global_sent_idx]  # (768,)
                        ann_arrays["text_embedding"] = emb_vec
                        annotations.append(ContinuousAnnotation(
                            annotation_id=f"ann_textemb_{run_id}_{sent_idx:04d}",
                            name="text_embedding",
                            domain="stimulus",
                            scope="atom",
                            data_ref=SignalRef(
                                file_path="__placeholder__",
                                internal_path="__placeholder__/annotations/text_embedding",
                                shape=(int(emb_vec.shape[0]),),
                            ),
                            data_sampling_rate=1.0,
                            alignment_method="sample_aligned",
                            custom_fields={
                                "embedding_model": "BERT-base-Chinese",
                                "embedding_dim": int(emb_vec.shape[0]),
                                "global_sentence_index": global_sent_idx,
                                "novel": novel,
                            },
                        ))

                    # Audio embedding (one per chapter/run)
                    if audio_emb is not None:
                        ann_arrays["audio_embedding"] = audio_emb
                        annotations.append(ContinuousAnnotation(
                            annotation_id=f"ann_audioemb_{run_id}_{sent_idx:04d}",
                            name="audio_embedding",
                            domain="stimulus",
                            scope="atom",
                            data_ref=SignalRef(
                                file_path="__placeholder__",
                                internal_path="__placeholder__/annotations/audio_embedding",
                                shape=(int(audio_emb.shape[0]),),
                            ),
                            data_sampling_rate=1.0,
                            alignment_method="sample_aligned",
                            custom_fields={
                                "embedding_model": "Wav2Vec2",
                                "embedding_dim": int(audio_emb.shape[0]),
                                "chapter_index": audio_chapter_idx,
                                "novel": novel,
                            },
                        ))

                    atom = Atom(
                        atom_id=atom_id,
                        atom_type=AtomType.EVENT_EPOCH,
                        dataset_id=self.dataset_id,
                        subject_id=sub_id,
                        session_id=ses_id,
                        run_id=run_id,
                        trial_index=sent_idx,
                        signal_ref=SignalRef(
                            file_path="__placeholder__",
                            internal_path=f"/atoms/{atom_id}/signal",
                            shape=(n_channels, duration),
                        ),
                        temporal=TemporalInfo(
                            onset_sample=onset,
                            onset_seconds=epoch["onset_sec"],
                            duration_samples=duration,
                            duration_seconds=duration / sfreq,
                        ),
                        channel_ids=channel_ids,
                        n_channels=n_channels,
                        sampling_rate=sfreq,
                        annotations=annotations,
                        processing_history=ProcessingHistory(
                            steps=[
                                ProcessingStep(
                                    operation="raw_import",
                                    parameters={
                                        "format": "brainvision_bids",
                                        "source": rec["path"].name,
                                        "novel": novel,
                                        "chapter": chapter,
                                        "repetition": repetition,
                                        "task": self.task,
                                        "preprocessed": self.use_preprocessed,
                                    },
                                ),
                            ],
                            is_raw=not self.use_preprocessed,
                            version_tag="preprocessed" if self.use_preprocessed else "raw",
                        ),
                        custom_fields={
                            "novel": novel,
                            "chapter": chapter,
                            "repetition": repetition,
                            "sentence_index": sent_idx,
                            "global_sentence_index": global_sent_idx,
                            "paradigm": "passive_listening" if self.task == "listening" else "reading_aloud",
                        },
                    )

                    # Convert to pool storage unit (V → µV)
                    signal, storage_unit, orig_unit = convert_to_storage_unit(
                        signal, source_unit="V", pool_config=self.pool.config,
                    )
                    atom.signal_unit = storage_unit
                    atom.original_unit = orig_unit

                    # Validate (first 5s)
                    max_val_samples = int(5 * sfreq)
                    val_sig = signal[:, :min(max_val_samples, signal.shape[1])]
                    warnings = validate_signal(
                        signal=val_sig,
                        atom_id=atom_id,
                        config=self.pool.config.get("import", {}),
                        signal_unit=storage_unit,
                    )
                    all_warnings.extend(warnings)

                    # Write signal (+ companion embedding arrays)
                    signal_ref = shard_mgr.write_atom_signal(
                        atom_id, signal,
                        ann_arrays if ann_arrays else None,
                    )
                    atom.signal_ref = signal_ref

                    writer.write_atom(atom)
                    atoms.append(atom)

        run_meta = RunMeta(
            run_id=run_id,
            session_id=ses_id,
            subject_id=sub_id,
            dataset_id=self.dataset_id,
            task_type=self.task,
            n_trials=len(atoms),
            duration_seconds=raw.times[-1] if raw.n_times > 0 else None,
            paradigm_details={
                "novel": novel,
                "chapter": chapter,
                "repetition": repetition,
                "n_sentences": len(atoms),
            },
        )
        self.pool.register_run(run_meta)

        # Persist channel_id → standard_name mapping
        ch_file = P.channels_path(
            self.pool.root, self.dataset_id, sub_id, ses_id
        )
        if not ch_file.exists():
            ch_file.parent.mkdir(parents=True, exist_ok=True)
            records = [
                {
                    "channel_id": ch.channel_id,
                    "name": ch.name,
                    "standard_name": ch.standard_name,
                    "channel_type": ch.type.value if ch.type else None,
                }
                for ch in ch_infos
            ]
            with open(ch_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        return ImportResult(
            atoms=atoms,
            run_meta=run_meta,
            channel_infos=ch_infos,
            n_atoms=len(atoms),
            warnings=all_warnings,
        )

    def _extract_channels(
        self,
        rec: Dict[str, Any],
        sfreq: float,
    ) -> Tuple[List[ChannelInfo], Dict[str, ElectrodeLocation]]:
        """Extract channel info and electrode positions."""
        # Electrodes
        electrodes = {}
        if rec.get("electrodes_tsv"):
            electrodes = _read_electrodes_tsv(rec["electrodes_tsv"])

        # Channels from channels.tsv
        ch_infos = []
        if rec.get("channels_tsv"):
            rows = _read_tsv(rec["channels_tsv"])
            for idx, row in enumerate(rows):
                ch_name = row.get("name", f"E{idx + 1}")
                bids_type = row.get("type", "EEG").upper()

                if bids_type == "EEG":
                    ch_type = ChannelType.EEG
                elif bids_type == "EOG":
                    ch_type = ChannelType.EOG
                elif bids_type in ("TRIG", "STIM"):
                    ch_type = ChannelType.STIM
                else:
                    ch_type = ChannelType.OTHER

                unit = row.get("units", "V")
                location = electrodes.get(ch_name)

                ch_infos.append(ChannelInfo(
                    channel_id=f"ch_{idx:03d}",
                    index=idx,
                    name=ch_name,
                    standard_name=standardize_channel_name(ch_name),
                    type=ch_type,
                    unit=unit,
                    sampling_rate=sfreq,
                    location=location,
                ))

        return ch_infos, electrodes

    @staticmethod
    def _safe_int(val) -> Optional[int]:
        """Safely convert to int, returning None for 'n/a' etc."""
        if val is None or val == "n/a":
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None


register_importer("chinese_eeg2", ChineseEEG2Importer)
