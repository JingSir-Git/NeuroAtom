"""ChineseEEG Importer: sentence-level EEG for Chinese reading comprehension.

ChineseEEG (Mou et al., 2024) is a BIDS-formatted 128-channel EEG dataset
recorded during visual reading of Chinese novels.

- **10 subjects** (sub-04 through sub-15, with gaps)
- **2 sessions** per subject: ses-LittlePrince (7 runs) and ses-GarnettDream (18 runs)
- **128-channel** EGI HydroCel-128 @ 1000 Hz, BrainVision format
- **Sentence-level epochs** from ROWS → ROWE event markers
- **Chapter markers** (CHxx) provide document structure metadata
- **BERT embeddings** available in derivatives/text_embeddings (optional)

This is the **predecessor** to ChineseEEG-2 (Chen et al., 2024); the key
difference is that subjects here *read* (visual presentation line-by-line)
rather than passively listen to audiobook narration.

Run numbering is sequential per session (run-01, run-02, …), each run
corresponds to one chapter of the novel.

Reference::

    Mou, X., He, C., Tan, L. et al. ChineseEEG: A Chinese Linguistic Corpora
    EEG Dataset for Semantic Alignment and Neural Decoding. Sci Data 11, 550
    (2024). https://doi.org/10.1038/s41597-024-03398-7

Usage::

    from neuroatom import Pool
    from neuroatom.importers.chinese_eeg import ChineseEEGImporter

    pool = Pool.create("my_pool")
    importer = ChineseEEGImporter(pool)
    results = importer.import_dataset(
        r"\\\\wsqlab\\ugreen\\Language\\25+ChineseEEG ..."
    )
    print(f"Imported {sum(r.n_atoms for r in results)} sentence atoms")
"""

import csv
import json
import logging
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the ChineseEEG importer")

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
from neuroatom.core.run import RunMeta
from neuroatom.core.session import SessionMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import ImportResult
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

# Regex for chapter markers: CH01, CH02, …, CH18
_CHAPTER_RE = re.compile(r"^CH(\d+)$")

# Dataset constants
DATASET_ID = "chinese_eeg_reading"
DATASET_NAME = "ChineseEEG Reading"
TASK_TYPE = "language_comprehension"
PARADIGM = "visual_reading"
SFREQ_NOMINAL = 1000.0
N_CHANNELS = 128
LINE_FREQ = 50.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a BIDS TSV file (handles UTF-8 BOM)."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def _read_electrodes_tsv(path: Path) -> Dict[str, ElectrodeLocation]:
    """Read electrodes.tsv → {channel_name: ElectrodeLocation}."""
    rows = _read_tsv(path)
    electrodes: Dict[str, ElectrodeLocation] = {}
    for row in rows:
        name = row.get("name", "")
        try:
            x = float(row.get("x", 0))
            y = float(row.get("y", 0))
            z = float(row.get("z", 0))
            electrodes[name] = ElectrodeLocation(
                x=x, y=y, z=z, coordinate_system="CapTrak",
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

    Returns list of dicts with keys:
        onset_sample, offset_sample, duration_samples,
        onset_sec, offset_sec, sentence_index
    """
    rows = _read_tsv(events_tsv)
    if not rows:
        return []

    starts: List[Dict[str, Any]] = []
    ends: List[Dict[str, Any]] = []
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

    # Pair each ROWS with the next ROWE that follows it
    epochs: List[Dict[str, Any]] = []
    end_idx = 0
    for start in starts:
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


def _load_text_embeddings_run(
    bids_root: Path, novel: str, run_num: str,
) -> Optional[np.ndarray]:
    """Load pre-computed BERT text embeddings for one run.

    Returns numpy array of shape (n_sentences, 768), or None if unavailable.
    Looks in ``derivatives/text_embeddings/{novel}_text_embedding/
    text_embedding_run_{run_num}.npy``.
    """
    npy_path = (
        bids_root / "derivatives" / "text_embeddings"
        / f"{novel}_text_embedding"
        / f"text_embedding_run_{int(run_num)}.npy"
    )
    if not npy_path.exists():
        return None
    try:
        return np.load(str(npy_path)).astype(np.float32)
    except Exception as exc:
        logger.debug("Could not load text embeddings from %s: %s", npy_path, exc)
        return None


def _detect_chapter(events_tsv: Path) -> Optional[int]:
    """Detect chapter number from CHxx event in events.tsv."""
    rows = _read_tsv(events_tsv)
    for row in rows:
        trial_type = row.get("trial_type", "")
        m = _CHAPTER_RE.match(trial_type)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Importer class
# ---------------------------------------------------------------------------

class ChineseEEGImporter:
    """Importer for ChineseEEG BIDS dataset.

    Extracts sentence-level EEG epochs from visual reading tasks across two
    novel sessions (LittlePrince, GarnettDream).

    Args:
        pool: Target Pool instance.
        max_shard_mb: Max HDF5 shard size in MB.
        compression: HDF5 compression algorithm.
    """

    def __init__(
        self,
        pool: Pool,
        task_config=None,
        max_shard_mb: int = 200,
        compression: str = "gzip",
    ):
        self.pool = pool
        self.max_shard_mb = max_shard_mb
        self.compression = compression

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect(path: Path) -> bool:
        """Detect if *path* is a ChineseEEG BIDS directory."""
        path = Path(path)
        desc_file = path / "dataset_description.json"
        if not desc_file.exists():
            return False
        try:
            with open(desc_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("Name", "")
            # ChineseEEG has "ChineseEEG:" in its name; distinguish from
            # ChineseEEG-2 which contains "Chinese Novel"
            return "ChineseEEG:" in name or "Linguistic Corpora" in name
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_recordings(self, bids_root: Path) -> List[Dict[str, Any]]:
        """Discover all EEG recordings in the BIDS tree.

        Returns:
            List of recording dicts with keys:
                path, subject, session, run, task,
                events_tsv, channels_tsv, electrodes_tsv
        """
        bids_root = Path(bids_root)
        recordings: List[Dict[str, Any]] = []

        for sub_dir in sorted(bids_root.glob("sub-*")):
            if not sub_dir.is_dir():
                continue
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                eeg_dir = ses_dir / "eeg"
                if not eeg_dir.exists():
                    continue

                for vhdr_file in sorted(eeg_dir.glob("*.vhdr")):
                    entities = self._parse_entities(vhdr_file.stem)
                    if not entities:
                        continue
                    # Filter only task=reading
                    if entities.get("task") != "reading":
                        continue

                    base = vhdr_file.stem.rsplit("_eeg", 1)[0]
                    events_tsv = eeg_dir / f"{base}_events.tsv"
                    channels_tsv = eeg_dir / f"{base}_channels.tsv"
                    electrodes_glob = list(eeg_dir.glob("*_electrodes.tsv"))

                    recordings.append({
                        "path": vhdr_file,
                        "subject": entities["subject"],
                        "session": entities.get("session"),
                        "run": entities.get("run", "01"),
                        "task": entities.get("task"),
                        "events_tsv": events_tsv if events_tsv.exists() else None,
                        "channels_tsv": channels_tsv if channels_tsv.exists() else None,
                        "electrodes_tsv": electrodes_glob[0] if electrodes_glob else None,
                    })

        logger.info(
            "Discovered %d recordings in %s", len(recordings), bids_root,
        )
        return recordings

    # ------------------------------------------------------------------
    # Public API: import full dataset
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sentence_texts(
        bids_root: Path, novel: str, run_num: str,
    ) -> List[str]:
        """Load sentence texts from the dataset's segmented novel xlsx files.

        Looks in ``derivatives/novels/segmented_novel/{novel}/
        segmented_Chinense_novel_run_{run_num}.xlsx``.

        Returns list of sentence strings (0-indexed), or empty list if
        the file is missing or openpyxl is not installed.
        """
        xlsx_path = (
            bids_root / "derivatives" / "novels" / "segmented_novel"
            / novel / f"segmented_Chinense_novel_run_{int(run_num)}.xlsx"
        )
        if not xlsx_path.exists():
            return []
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
            ws = wb.active
            sentences: List[str] = []
            for row in ws.iter_rows(min_row=2, values_only=True):
                val = row[0]
                if val is None:
                    continue
                val = str(val).strip()
                # Skip chapter number rows (pure digits)
                if val and not val.isdigit():
                    sentences.append(val)
            wb.close()
            return sentences
        except Exception as exc:
            logger.debug("Could not load sentence texts from %s: %s", xlsx_path, exc)
            return []

    def import_dataset(
        self,
        bids_root: Path,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        max_runs: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import the ChineseEEG dataset with sentence-level atomization.

        Args:
            bids_root: Path to the ChineseEEG BIDS root directory.
            subjects: Filter by subject IDs (e.g. ``["04", "05"]``). None = all.
            sessions: Filter by session names (e.g. ``["LittlePrince"]``). None = all.
            max_runs: Max runs to import (for testing). None = all.

        Returns:
            List of ImportResult, one per run.
        """
        bids_root = Path(bids_root)
        t0 = time.time()

        # Register dataset
        self.pool.register_dataset(DatasetMeta(
            dataset_id=DATASET_ID,
            name=DATASET_NAME,
            task_types=[TASK_TYPE],
            n_subjects=0,
        ))

        # Read participants
        participants = _read_tsv(bids_root / "participants.tsv")

        # Discover recordings
        recordings = self.discover_recordings(bids_root)
        if subjects:
            recordings = [r for r in recordings if r["subject"] in subjects]
        if sessions:
            recordings = [r for r in recordings if r.get("session") in sessions]
        if max_runs is not None:
            recordings = recordings[:max_runs]

        # Register unique subjects
        registered_subjects: set = set()
        for rec in recordings:
            sub_id = f"sub-{rec['subject']}"
            if sub_id in registered_subjects:
                continue
            p_info = next(
                (p for p in participants
                 if p.get("participant_id", "") in (sub_id, rec["subject"])),
                {},
            )
            self.pool.register_subject(SubjectMeta(
                subject_id=sub_id,
                dataset_id=DATASET_ID,
                sex=p_info.get("sex") if p_info.get("sex") != "n/a" else None,
            ))
            registered_subjects.add(sub_id)

        # Attach bids_root to each rec so _import_run can locate stimuli
        for rec in recordings:
            rec["bids_root"] = bids_root

        # Import each run
        results: List[ImportResult] = []
        for rec in recordings:
            try:
                result = self._import_run(rec)
                results.append(result)
                logger.info(
                    "Imported %s/%s/%s: %d sentence atoms",
                    f"sub-{rec['subject']}", f"ses-{rec.get('session', '?')}",
                    f"run-{rec['run']}", result.n_atoms,
                )
            except Exception as e:
                logger.error(
                    "Failed run-%s for sub-%s ses-%s: %s",
                    rec["run"], rec["subject"], rec.get("session", "?"),
                    e, exc_info=True,
                )

        elapsed = time.time() - t0
        total_atoms = sum(r.n_atoms for r in results)
        logger.info(
            "ChineseEEG import: %d runs, %d sentence atoms in %.1fs",
            len(results), total_atoms, elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Public API: import one subject
    # ------------------------------------------------------------------

    def import_subject(
        self,
        bids_root: Path,
        subject_id: str,
        sessions: Optional[List[str]] = None,
        max_runs: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import all runs for a single subject.

        Args:
            bids_root: Path to the ChineseEEG BIDS root directory.
            subject_id: Subject identifier (e.g. ``"04"`` or ``"sub-04"``).
            sessions: Optional filter for sessions.
            max_runs: Max runs to import. None = all.

        Returns:
            List of ImportResult, one per run.
        """
        # Normalize subject_id
        if subject_id.startswith("sub-"):
            subject_id = subject_id[4:]
        return self.import_dataset(
            bids_root,
            subjects=[subject_id],
            sessions=sessions,
            max_runs=max_runs,
        )

    # ------------------------------------------------------------------
    # Internal: import a single run
    # ------------------------------------------------------------------

    def _import_run(self, rec: Dict[str, Any]) -> ImportResult:
        """Import a single run: load EEG → extract sentence epochs → store atoms."""
        sub_id = f"sub-{rec['subject']}"
        ses_id = f"ses-{rec['session']}" if rec.get("session") else "ses-01"
        run_id = f"run-{rec['run']}"
        novel = rec.get("session", "unknown")

        # Detect chapter number from events
        chapter = None
        if rec.get("events_tsv"):
            chapter = _detect_chapter(rec["events_tsv"])

        # Load sentence texts from derivatives (if available)
        sentence_texts: List[str] = []
        if rec.get("bids_root") and novel != "unknown":
            sentence_texts = self._load_sentence_texts(
                rec["bids_root"], novel, rec["run"],
            )
            if sentence_texts:
                logger.debug(
                    "Loaded %d sentence texts for %s/%s/%s",
                    len(sentence_texts), sub_id, ses_id, run_id,
                )

        # Load BERT text embeddings for this run (if available)
        text_embeddings: Optional[np.ndarray] = None
        if rec.get("bids_root") and novel != "unknown":
            text_embeddings = _load_text_embeddings_run(
                rec["bids_root"], novel, rec["run"],
            )
            if text_embeddings is not None:
                logger.debug(
                    "Loaded text embeddings %s for %s/%s/%s",
                    text_embeddings.shape, sub_id, ses_id, run_id,
                )

        # Register session
        self.pool.register_session(SessionMeta(
            session_id=ses_id,
            subject_id=sub_id,
            dataset_id=DATASET_ID,
            sampling_rate=SFREQ_NOMINAL,
            device_manufacturer="EGI",
            device_model="Geodesic EEG 400",
            placement_scheme="GSN-HydroCel-128",
            line_freq=LINE_FREQ,
            recording_type="continuous",
        ))

        # Load raw EEG (fix vhdr internal references if needed)
        vhdr_path = self._fix_vhdr_references(rec["path"])
        try:
            raw = mne.io.read_raw_brainvision(
                str(vhdr_path), preload=True, verbose="WARNING",
            )
        finally:
            # Clean up patched temp file
            if vhdr_path != rec["path"]:
                try:
                    vhdr_path.unlink(missing_ok=True)
                except OSError:
                    pass
        sfreq = raw.info["sfreq"]
        n_total = raw.n_times

        # Channel info + electrode positions
        ch_infos, electrodes = self._extract_channels(rec, sfreq)
        eeg_ch_names = [ci.name for ci in ch_infos if ci.type == ChannelType.EEG]
        eeg_picks = mne.pick_channels(raw.ch_names, eeg_ch_names, ordered=True)

        if len(eeg_picks) != len(eeg_ch_names):
            logger.warning(
                "%s/%s/%s: channels.tsv lists %d EEG channels but only %d "
                "found in raw data. Using intersection.",
                sub_id, ses_id, run_id, len(eeg_ch_names), len(eeg_picks),
            )
            matched_names = [raw.ch_names[i] for i in eeg_picks]
            ch_infos = [ci for ci in ch_infos
                        if ci.type != ChannelType.EEG or ci.name in matched_names]

        if len(eeg_picks) == 0:
            logger.warning("No EEG channels found in %s/%s/%s", sub_id, ses_id, run_id)
            return self._empty_result(sub_id, ses_id, run_id, ch_infos)

        # Early return if no events.tsv
        if rec.get("events_tsv") is None:
            logger.warning("No events.tsv for %s/%s/%s", sub_id, ses_id, run_id)
            return self._empty_result(sub_id, ses_id, run_id, ch_infos)

        # Extract sentence epochs from ROWS→ROWE pairs
        epochs = _extract_sentence_epochs(
            rec["events_tsv"], sfreq, "ROWS", "ROWE",
        )
        if not epochs:
            logger.warning("No sentence epochs in %s/%s/%s", sub_id, ses_id, run_id)
            return self._empty_result(
                sub_id, ses_id, run_id, ch_infos,
                warnings=["no_sentence_epochs"],
            )

        # Store atoms
        channel_ids = [ci.channel_id for ci in ch_infos if ci.type == ChannelType.EEG]
        n_channels = len(channel_ids)
        all_warnings: List[str] = []
        atoms: List[Atom] = []

        with ShardManager(
            pool_root=self.pool.root,
            dataset_id=DATASET_ID,
            subject_id=sub_id,
            session_id=ses_id,
            run_id=run_id,
            max_shard_size_mb=self.max_shard_mb,
            compression=self.compression,
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, DATASET_ID, sub_id, ses_id, run_id,
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                for epoch in epochs:
                    onset = epoch["onset_sample"]
                    offset = min(epoch["offset_sample"], n_total)
                    duration = offset - onset
                    if duration <= 0:
                        continue

                    # Extract EEG signal segment
                    signal = raw.get_data(
                        picks=eeg_picks, start=onset, stop=offset,
                    )  # shape: (n_eeg_channels, n_samples)

                    atom_id = compute_atom_id(
                        dataset_id=DATASET_ID,
                        subject_id=sub_id,
                        session_id=ses_id,
                        run_id=run_id,
                        onset_sample=onset,
                    )

                    # Build annotations
                    sent_idx = epoch["sentence_index"]
                    annotations = [
                        CategoricalAnnotation(
                            annotation_id=f"ann_novel_{run_id}_{sent_idx:04d}",
                            name="novel",
                            value=novel,
                        ),
                        NumericAnnotation(
                            annotation_id=f"ann_sent_{run_id}_{sent_idx:04d}",
                            name="sentence_index",
                            numeric_value=float(sent_idx),
                        ),
                    ]
                    if chapter is not None:
                        annotations.append(NumericAnnotation(
                            annotation_id=f"ann_chapter_{run_id}_{sent_idx:04d}",
                            name="chapter",
                            numeric_value=float(chapter),
                        ))

                    # Sentence text from derivatives xlsx (if available)
                    if sent_idx < len(sentence_texts):
                        annotations.append(TextAnnotation(
                            annotation_id=f"ann_text_{run_id}_{sent_idx:04d}",
                            name="sentence_text",
                            text_value=sentence_texts[sent_idx],
                            custom_fields={
                                "source": "dataset_file",
                                "file": f"derivatives/novels/segmented_novel/{novel}/segmented_Chinense_novel_run_{int(rec['run'])}.xlsx",
                            },
                        ))

                    # BERT text embedding companion array (if available)
                    ann_arrays: Dict[str, np.ndarray] = {}
                    if (
                        text_embeddings is not None
                        and sent_idx < text_embeddings.shape[0]
                    ):
                        emb_vec = text_embeddings[sent_idx]  # (768,)
                        ann_arrays["text_embedding"] = emb_vec
                        annotations.append(ContinuousAnnotation(
                            annotation_id=f"ann_textemb_{run_id}_{sent_idx:04d}",
                            name="text_embedding",
                            domain="stimulus",
                            scope="atom",
                            data_ref=SignalRef(
                                file_path="__placeholder__",
                                internal_path=f"__placeholder__/annotations/text_embedding",
                                shape=(int(emb_vec.shape[0]),),
                            ),
                            data_sampling_rate=1.0,
                            alignment_method="sample_aligned",
                            custom_fields={
                                "embedding_model": "BERT",
                                "embedding_dim": int(emb_vec.shape[0]),
                                "novel": novel,
                                "run": rec["run"],
                            },
                        ))

                    atom = Atom(
                        atom_id=atom_id,
                        atom_type=AtomType.EVENT_EPOCH,
                        dataset_id=DATASET_ID,
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
                                        "task": "reading",
                                    },
                                ),
                            ],
                            is_raw=True,
                            version_tag="raw",
                        ),
                        custom_fields={
                            "novel": novel,
                            "chapter": chapter,
                            "sentence_index": sent_idx,
                            "paradigm": PARADIGM,
                        },
                    )

                    # Convert to pool storage unit (V → µV)
                    signal, storage_unit, orig_unit = convert_to_storage_unit(
                        signal, source_unit="V", pool_config=self.pool.config,
                    )
                    atom.signal_unit = storage_unit
                    atom.original_unit = orig_unit

                    # Validate (first 5 s)
                    max_val_samples = int(5 * sfreq)
                    val_sig = signal[
                        :, :min(max_val_samples, signal.shape[1])
                    ]
                    warnings = validate_signal(
                        signal=val_sig, atom_id=atom_id,
                        config=self.pool.config.get("import", {}),
                        signal_unit=storage_unit,
                    )
                    all_warnings.extend(warnings)

                    # Write signal (+ companion embedding array) to shard
                    signal_ref = shard_mgr.write_atom_signal(
                        atom_id, signal,
                        ann_arrays if ann_arrays else None,
                    )
                    atom.signal_ref = signal_ref

                    writer.write_atom(atom)
                    atoms.append(atom)

        # Register run metadata
        run_meta = RunMeta(
            run_id=run_id,
            session_id=ses_id,
            subject_id=sub_id,
            dataset_id=DATASET_ID,
            task_type=TASK_TYPE,
            n_trials=len(atoms),
            duration_seconds=raw.times[-1] if raw.n_times > 0 else None,
            paradigm_details={
                "novel": novel,
                "chapter": chapter,
                "n_sentences": len(atoms),
                "paradigm": PARADIGM,
            },
        )
        self.pool.register_run(run_meta)

        # Persist channel_id → standard_name mapping
        self._write_channels_json(sub_id, ses_id, ch_infos)

        return ImportResult(
            atoms=atoms,
            run_meta=run_meta,
            channel_infos=ch_infos,
            n_atoms=len(atoms),
            warnings=all_warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_result(
        self,
        sub_id: str,
        ses_id: str,
        run_id: str,
        ch_infos: List[ChannelInfo],
        warnings: Optional[List[str]] = None,
    ) -> ImportResult:
        """Return an empty ImportResult for runs with no usable epochs."""
        run_meta = RunMeta(
            run_id=run_id,
            session_id=ses_id,
            subject_id=sub_id,
            dataset_id=DATASET_ID,
            task_type=TASK_TYPE,
        )
        return ImportResult(
            atoms=[],
            run_meta=run_meta,
            channel_infos=ch_infos,
            n_atoms=0,
            warnings=warnings or [],
        )

    def _extract_channels(
        self,
        rec: Dict[str, Any],
        sfreq: float,
    ) -> Tuple[List[ChannelInfo], Dict[str, ElectrodeLocation]]:
        """Extract channel info and electrode positions from BIDS sidecars."""
        # Electrode positions
        electrodes: Dict[str, ElectrodeLocation] = {}
        if rec.get("electrodes_tsv"):
            electrodes = _read_electrodes_tsv(rec["electrodes_tsv"])

        # Channel info from channels.tsv
        ch_infos: List[ChannelInfo] = []
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

    def _write_channels_json(
        self,
        sub_id: str,
        ses_id: str,
        ch_infos: List[ChannelInfo],
    ) -> None:
        """Persist channel_id → standard_name mapping to channels.json."""
        ch_file = P.channels_path(self.pool.root, DATASET_ID, sub_id, ses_id)
        if ch_file.exists():
            return
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

    @staticmethod
    def _fix_vhdr_references(vhdr_path: Path) -> Path:
        """Patch .vhdr if its DataFile/MarkerFile references don't exist on disk.

        The ChineseEEG dataset has a known typo: some .vhdr files reference
        ``GranettDream`` while the actual filenames use ``GarnettDream``.
        If the referenced files are missing, this creates a patched copy of
        the .vhdr in a temp directory with corrected references.

        Returns the (possibly patched) path to use for MNE loading.
        """
        vhdr_path = Path(vhdr_path)
        eeg_dir = vhdr_path.parent

        with open(vhdr_path, "r", encoding="utf-8") as f:
            content = f.read()

        needs_patch = False
        patched = content
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("DataFile=") or stripped.startswith("MarkerFile="):
                ref_name = stripped.split("=", 1)[1]
                ref_full = eeg_dir / ref_name
                if not ref_full.exists():
                    # Try replacing GranettDream ↔ GarnettDream
                    if "GranettDream" in ref_name:
                        fixed = ref_name.replace("GranettDream", "GarnettDream")
                    elif "GarnettDream" in ref_name:
                        fixed = ref_name.replace("GarnettDream", "GranettDream")
                    else:
                        continue
                    if (eeg_dir / fixed).exists():
                        patched = patched.replace(ref_name, fixed)
                        needs_patch = True
                        logger.debug(
                            "Patching vhdr reference: %s → %s", ref_name, fixed,
                        )

        if not needs_patch:
            return vhdr_path

        # MNE resolves DataFile/MarkerFile relative to the .vhdr location.
        # Strategy 1: write patched .vhdr next to the original (same dir).
        temp_vhdr = eeg_dir / (vhdr_path.stem + "_patched.vhdr")
        try:
            with open(temp_vhdr, "w", encoding="utf-8") as f:
                f.write(patched)
            return temp_vhdr
        except OSError:
            pass

        # Strategy 2: read-only share → write to a temp dir with absolute paths
        # so MNE can resolve the referenced files.
        abs_patched = patched
        for line in patched.splitlines():
            stripped = line.strip()
            if stripped.startswith("DataFile=") or stripped.startswith("MarkerFile="):
                ref = stripped.split("=", 1)[1]
                abs_patched = abs_patched.replace(
                    ref, str(eeg_dir / ref),
                )
        tmp_dir = Path(tempfile.mkdtemp(prefix="ceeg_vhdr_"))
        patched_vhdr = tmp_dir / vhdr_path.name
        with open(patched_vhdr, "w", encoding="utf-8") as f:
            f.write(abs_patched)
        return patched_vhdr

    @staticmethod
    def _parse_entities(stem: str) -> Optional[Dict[str, str]]:
        """Parse BIDS entities from a filename stem.

        Example: ``sub-04_ses-LittlePrince_task-reading_run-01_eeg``
        → ``{"subject": "04", "session": "LittlePrince", "task": "reading", "run": "01"}``
        """
        parts = stem.split("_")
        entities: Dict[str, str] = {}
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


register_importer("chinese_eeg", ChineseEEGImporter)
