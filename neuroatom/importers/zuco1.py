"""ZuCo 1.0 Importer: multi-task natural reading EEG + eye-tracking.

Hollenstein et al. (2018). 12 subjects, 3 tasks (SR, NR, TSR),
105 EGI channels @ 500 Hz, HDF5 (MATLAB v7.3) EEGLAB format.

Shares the identical HDF5 structure with ZuCo 2.0. The core parsing
helpers from zuco2.py are reused; this module adds multi-task
discovery and the slightly different file naming conventions.

Data layout expected:
    ZuCo_1.0_Full/
        task1- SR/
            Preprocessed/
                ZAB/
                    gip_ZAB_SR1_EEG.mat ... gip_ZAB_SR8_EEG.mat
                    oip_ZAB_SR4_EEG.mat ...
                    ZAB_SR1_corrected_ET.mat ...
                    wordbounds_SNR1_ZAB.mat ...
                sentencesSR.mat
        task2 - NR/
            Preprocessed/
                ZAB/
                    gip_ZAB_NR1_EEG.mat ... gip_ZAB_NR6_EEG.mat
                    wordbounds_NR_ZAB.mat
                sentencesNR.mat
        task3 - TSR/
            Preprocessed/
                ZAB/
                    gip_ZAB_TSR1_EEG.mat ... gip_ZAB_TSR7_EEG.mat
                sentencesTSR.mat
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from neuroatom.core.annotation import CategoricalAnnotation, NumericAnnotation, TextAnnotation
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import AtomType, ChannelStatus, ChannelType
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.quality import QualityInfo
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.importers.zuco2 import (
    _extract_automagic_quality,
    _extract_channel_infos,
    _extract_events,
    _h5_read_string,
    _load_wordbounds,
    _sentence_epochs,
)
from neuroatom.importers.zuco_features import (
    build_et_annotations,
    build_word_annotations,
    find_et_file,
    find_results_file,
    load_corrected_et_v5,
    load_zuco_results_v5,
    segment_et_by_sentences,
)
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

# Task directory names in ZuCo 1.0
_TASK_DIRS = {
    "sr": "task1- SR",
    "nr": "task2 - NR",
    "tsr": "task3 - TSR",
}

# Text ID regex per task
_TEXT_ID_RE = {
    "sr": re.compile(r"((?:SR|SNR)\d+)"),
    "nr": re.compile(r"(NR\d+)"),
    "tsr": re.compile(r"(TSR\d+)"),
}


class Zuco1Importer(BaseImporter):
    """Importer for ZuCo 1.0 (multi-task natural reading).

    Reuses ZuCo 2.0 HDF5 parsing but handles:
        - 3 tasks with different directory layouts
        - SR/SNR/NR/TSR text prefixes
        - Per-subject wordbounds naming (wordbounds_X_SUBJ.mat)
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)
        self._file_prefix = task_config.data.get("file_prefix", "gip")

    @staticmethod
    def detect(path: Path) -> bool:
        """Detect ZuCo 1.0 by its directory structure."""
        path = Path(path)
        if not path.is_dir():
            return False
        return (path / "task1- SR").exists() and (path / "task3 - TSR").exists()

    def load_raw(self, path):
        raise NotImplementedError("Use import_dataset() instead.")

    def extract_channel_infos(self, raw):
        raise NotImplementedError("Use import_dataset() instead.")

    def extract_events(self, raw):
        return None

    # ------------------------------------------------------------------
    # Main entry: import full dataset
    # ------------------------------------------------------------------

    def import_dataset(
        self,
        root: Path,
        tasks: Optional[List[str]] = None,
        subjects: Optional[List[str]] = None,
        max_texts: Optional[int] = None,
        max_sentences: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import ZuCo 1.0 dataset.

        Args:
            root: Path to ZuCo_1.0_Full directory.
            tasks: Subset of ["sr", "nr", "tsr"]. Default: all.
            subjects: Subset of subject codes. Default: all discovered.
            max_texts: Limit texts per subject per task (for testing).
            max_sentences: Limit sentences per text (for testing).

        Returns:
            List of ImportResult, one per subject-text.
        """
        root = Path(root)
        dataset_id = self._task_config.dataset_id
        tasks = tasks or list(_TASK_DIRS.keys())

        # Discover subjects from first available task
        all_subjects = self._discover_subjects(root)
        if subjects:
            all_subjects = [s for s in all_subjects if s in subjects]

        # Register dataset
        self._pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id,
            name=self._task_config.dataset_name,
            task_types=["reading"],
            n_subjects=len(all_subjects),
            original_format="hdf5_eeglab",
        ))

        # Register subjects
        for sub_id in all_subjects:
            self._pool.register_subject(SubjectMeta(
                subject_id=sub_id,
                dataset_id=dataset_id,
            ))

        results = []
        for task in tasks:
            task_dir = root / _TASK_DIRS.get(task, "")
            if not task_dir.exists():
                logger.warning("Task dir not found: %s", task_dir)
                continue

            prep_dir = task_dir / "Preprocessed"
            if not prep_dir.exists():
                continue

            for sub_id in all_subjects:
                sub_dir = prep_dir / sub_id
                if not sub_dir.exists():
                    continue

                try:
                    sub_results = self._import_subject_task(
                        prep_dir, sub_dir, sub_id, task, dataset_id,
                        max_texts, max_sentences,
                    )
                    results.extend(sub_results)
                except Exception as e:
                    logger.error(
                        "Failed %s/%s: %s", sub_id, task, e,
                    )

        # Post-import quality assessment
        try:
            tier = self._pool.assess_quality(dataset_id)
            if tier:
                logger.info("Quality tier for %s: %s", dataset_id, tier)
        except Exception:
            pass

        total_atoms = sum(len(r.atoms) for r in results)
        logger.info(
            "ZuCo 1.0 import complete: %d subjects × %d tasks → "
            "%d results, %d atoms total.",
            len(all_subjects), len(tasks), len(results), total_atoms,
        )
        return results

    def _discover_subjects(self, root: Path) -> List[str]:
        """Find all subject codes across task directories."""
        subjects = set()
        for task_key, task_dir_name in _TASK_DIRS.items():
            prep_dir = root / task_dir_name / "Preprocessed"
            if prep_dir.exists():
                for d in prep_dir.iterdir():
                    if d.is_dir() and d.name[0].isupper() and len(d.name) == 3:
                        subjects.add(d.name)
        return sorted(subjects)

    def _import_subject_task(
        self,
        prep_dir: Path,
        sub_dir: Path,
        subject_id: str,
        task: str,
        dataset_id: str,
        max_texts: Optional[int],
        max_sentences: Optional[int],
    ) -> List[ImportResult]:
        """Import all texts for one subject in one task."""
        prefix = self._file_prefix
        text_re = _TEXT_ID_RE.get(task, re.compile(r"([A-Z]+\d+)"))

        # Discover available text files
        available: Dict[str, Path] = {}
        # Try gip first, then oip as fallback
        for fp in (prefix, "oip"):
            for mat_path in sub_dir.glob(f"{fp}_{subject_id}_*_EEG.mat"):
                m = text_re.search(mat_path.name)
                if m:
                    text_id = m.group(1)
                    if text_id not in available:
                        available[text_id] = mat_path

        if not available:
            logger.warning("No EEG files found for %s/%s", subject_id, task)
            return []

        texts = sorted(available.keys())
        if max_texts:
            texts = texts[:max_texts]

        # ── Load results file for word-level features (best-effort) ──
        root = prep_dir.parent  # task dir
        results_root = root.parent  # dataset root
        results_sentences: Optional[List[Dict[str, Any]]] = None
        results_path = find_results_file(results_root, subject_id, task)
        if results_path is not None:
            try:
                results_sentences = load_zuco_results_v5(results_path)
                logger.info(
                    "Loaded %d results sentences for %s/%s from %s",
                    len(results_sentences), subject_id, task, results_path.name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load results file %s: %s", results_path.name, e,
                )

        sentence_offset = 0
        # Only apply results for texts matching the task prefix
        # (e.g., SR* for "sr", not SNR*; NR* for "nr")
        task_prefix = task.upper()
        results = []
        for text_id in texts:
            mat_path = available[text_id]
            # SNR texts are control stimuli with different sentences;
            # only pass results for texts matching the exact task prefix
            text_matches = text_id.upper().startswith(task_prefix) and not (
                task_prefix == "SR" and text_id.upper().startswith("SNR")
            )
            try:
                atoms, ch_infos, warnings = self._import_text(
                    mat_path=mat_path,
                    text_id=text_id,
                    task=task,
                    dataset_id=dataset_id,
                    subject_id=subject_id,
                    prep_dir=prep_dir,
                    max_sentences=max_sentences,
                    results_sentences=results_sentences if text_matches else None,
                    results_offset=sentence_offset,
                )
                if atoms:
                    session_id = f"ses-{task}-{text_id.lower()}"
                    run_id = "run-01"
                    run_meta = RunMeta(
                        run_id=run_id,
                        session_id=session_id,
                        subject_id=subject_id,
                        dataset_id=dataset_id,
                        task_type="reading",
                        n_trials=len(atoms),
                    )
                    self._pool.register_run(run_meta)
                    self._write_channels_json(
                        dataset_id, subject_id, session_id, ch_infos,
                    )
                    results.append(ImportResult(
                        atoms=atoms,
                        run_meta=run_meta,
                        channel_infos=ch_infos,
                        warnings=warnings,
                    ))
                    # Advance offset only for texts using results data
                    if text_matches:
                        sentence_offset += len(atoms)
            except Exception as e:
                logger.error(
                    "Failed %s/%s/%s: %s", subject_id, task, text_id, e,
                )

        return results

    def _import_text(
        self,
        mat_path: Path,
        text_id: str,
        task: str,
        dataset_id: str,
        subject_id: str,
        prep_dir: Path,
        max_sentences: Optional[int] = None,
        results_sentences: Optional[List[Dict[str, Any]]] = None,
        results_offset: int = 0,
    ) -> Tuple[List[Atom], List[ChannelInfo], List[str]]:
        """Import sentence-level epochs from one text's EEG file.

        Nearly identical to Zuco2Importer._import_text(), adapted for
        multi-task layout and ZuCo 1.0 naming.

        If *results_sentences* is provided (from the results .mat file),
        each atom is enriched with:
        - TextAnnotation: sentence content
        - EventSequenceAnnotation: per-word eye-tracking + band-power features
        - NumericAnnotation: omission rate
        - HDF5 companion arrays: sentence fixation data

        If a corrected ET file is found, raw gaze and pupil time series
        are stored as ContinuousAnnotation + HDF5 arrays.
        """
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P

        session_id = f"ses-{task}-{text_id.lower()}"
        run_id = "run-01"

        # Load word boundaries (ZuCo 1.0 uses per-subject naming)
        wordbounds = self._load_wordbounds_v1(prep_dir, text_id, subject_id, task)

        # ── Load corrected eye-tracking file (best-effort) ──
        et_segments: List[Optional[Dict[str, np.ndarray]]] = []
        et_srate = 500.0  # EyeLink default
        et_path = find_et_file(prep_dir, subject_id, text_id)
        if et_path is not None:
            et_data = load_corrected_et_v5(et_path)
            if et_data is not None:
                sentence_event = self._task_config.data.get("events", {}).get(
                    "sentence_onset", "10"
                )
                et_segments = segment_et_by_sentences(
                    et_data, sentence_event_code=int(sentence_event),
                )
                # Estimate ET sample rate from timestamps
                ts = et_data["data"][:, 0]
                if len(ts) > 1:
                    median_dt = np.median(np.diff(ts))
                    if median_dt > 0:
                        et_srate = 1000.0 / median_dt  # timestamps in ms
                logger.info(
                    "Loaded ET for %s/%s: %d sentence segments @ ~%.0f Hz",
                    subject_id, text_id, len(et_segments), et_srate,
                )

        from neuroatom.utils.mat_compat import require_mat_v73
        require_mat_v73(mat_path, "Zuco1Importer")

        with h5py.File(str(mat_path), "r") as f:
            eeg = f["EEG"]

            srate = float(eeg["srate"][()].flat[0])
            data_ds = eeg["data"]
            total_samples, n_channels = data_ds.shape

            automagic_info = _extract_automagic_quality(f)
            ch_infos = _extract_channel_infos(f, eeg, srate)
            events = _extract_events(f, eeg)

            sentence_event = self._task_config.data.get("events", {}).get(
                "sentence_onset", "10"
            )
            epochs = _sentence_epochs(events, total_samples, sentence_event)

            if max_sentences is not None:
                epochs = epochs[:max_sentences]

            logger.info(
                "Loading %s/%s/%s: %d sentences, %d ch @ %.0f Hz",
                subject_id, task, text_id, len(epochs), n_channels, srate,
            )

            if not epochs:
                return [], ch_infos, []

            # Ensure pool hierarchy
            self._pool.ensure_session(
                dataset_id, subject_id, session_id, sampling_rate=srate,
            )
            self._pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            bad_ch_names = automagic_info.get("bad_channels", [])
            quality = QualityInfo(
                overall_status="good",
                bad_channels=bad_ch_names,
                auto_qc_passed=automagic_info.get("automagic_rate", "") != "bad"
                if "automagic_rate" in automagic_info else None,
            )
            max_shard_mb = self._pool.config.get("storage", {}).get(
                "max_shard_size_mb", 200.0,
            )
            compression = self._pool.config.get("storage", {}).get(
                "compression", "gzip",
            )
            channel_ids = [ch.channel_id for ch in ch_infos]

            atoms: List[Atom] = []
            all_warnings: List[str] = []

            with ShardManager(
                pool_root=self._pool.root,
                dataset_id=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                max_shard_size_mb=max_shard_mb,
                compression=compression,
            ) as shard_mgr:
                jsonl_path = P.atoms_jsonl_path(
                    self._pool.root, dataset_id, subject_id, session_id, run_id,
                )
                with AtomJSONLWriter(jsonl_path) as writer:
                    for sent_idx, (start_samp, end_samp) in enumerate(epochs):
                        n_sent = end_samp - start_samp
                        raw_seg = data_ds[start_samp:end_samp, :]
                        signal = raw_seg.T.astype(np.float32)
                        dur_s = n_sent / srate

                        annotations = [
                            CategoricalAnnotation(
                                annotation_id=f"ann_task_{run_id}_{sent_idx:04d}",
                                name="task",
                                value=task,
                            ),
                            CategoricalAnnotation(
                                annotation_id=f"ann_text_{run_id}_{sent_idx:04d}",
                                name="text_id",
                                value=text_id,
                            ),
                            NumericAnnotation(
                                annotation_id=f"ann_sent_{run_id}_{sent_idx:04d}",
                                name="sentence_index",
                                numeric_value=float(sent_idx),
                            ),
                        ]

                        # Word count from wordbounds
                        if wordbounds is not None:
                            try:
                                wb = wordbounds[0, sent_idx] if sent_idx < wordbounds.shape[1] else None
                                if wb is not None and hasattr(wb, "shape"):
                                    annotations.append(NumericAnnotation(
                                        annotation_id=f"ann_nw_{run_id}_{sent_idx:04d}",
                                        name="n_words",
                                        numeric_value=float(wb.shape[0]),
                                    ))
                            except Exception:
                                pass

                        # ── Word-level features from results file ──
                        ann_arrays: Dict[str, np.ndarray] = {}
                        res_idx = results_offset + sent_idx
                        if (
                            results_sentences is not None
                            and res_idx < len(results_sentences)
                        ):
                            word_anns, word_arrays = build_word_annotations(
                                results_sentences[res_idx],
                                run_id, sent_idx, srate,
                            )
                            annotations.extend(word_anns)
                            ann_arrays.update(word_arrays)

                        # ── Raw eye-tracking time series ──
                        et_seg = (
                            et_segments[sent_idx]
                            if sent_idx < len(et_segments)
                            else None
                        )
                        if et_seg is not None:
                            et_anns, et_arrays = build_et_annotations(
                                et_seg, run_id, sent_idx, et_srate,
                            )
                            annotations.extend(et_anns)
                            ann_arrays.update(et_arrays)

                        atom_id = compute_atom_id(
                            dataset_id, subject_id, session_id, run_id, start_samp,
                        )

                        atom = Atom(
                            atom_id=atom_id,
                            atom_type=AtomType.EVENT_EPOCH,
                            dataset_id=dataset_id,
                            subject_id=subject_id,
                            session_id=session_id,
                            run_id=run_id,
                            trial_index=sent_idx,
                            signal_ref=SignalRef(
                                file_path="__placeholder__",
                                internal_path=f"/atoms/{atom_id}/signal",
                                shape=(n_channels, n_sent),
                            ),
                            temporal=TemporalInfo(
                                onset_sample=start_samp,
                                onset_seconds=start_samp / srate,
                                duration_samples=n_sent,
                                duration_seconds=dur_s,
                            ),
                            channel_ids=channel_ids,
                            n_channels=n_channels,
                            sampling_rate=srate,
                            annotations=annotations,
                            quality=quality,
                            processing_history=ProcessingHistory(
                                steps=[ProcessingStep(
                                    operation="raw_import",
                                    parameters={
                                        "format": "zuco1_hdf5_eeglab",
                                        "source_file": mat_path.name,
                                        "task": task,
                                        "text_id": text_id,
                                        "sentence_index": sent_idx,
                                        "has_word_features": bool(
                                            results_sentences
                                            and res_idx < len(results_sentences)
                                        ),
                                        "has_eye_tracking": et_seg is not None,
                                    },
                                )],
                                is_raw=False,
                                version_tag="preprocessed",
                            ),
                            custom_fields={
                                "text_id": text_id,
                                "task": task,
                                "sentence_index": sent_idx,
                                "reading_duration_s": round(dur_s, 3),
                            },
                        )

                        signal, su, ou = convert_to_storage_unit(
                            signal, source_unit="uV",
                            pool_config=self._pool.config,
                        )
                        atom.signal_unit = su
                        atom.original_unit = ou

                        val_sig = signal[:, :min(5000, signal.shape[1])]
                        ws = validate_signal(
                            signal=val_sig,
                            atom_id=atom_id,
                            config=self._pool.config.get("import", {}),
                            signal_unit=su,
                        )
                        all_warnings.extend(ws)

                        sig_ref = shard_mgr.write_atom_signal(
                            atom_id, signal, ann_arrays or None,
                        )
                        atom.signal_ref = sig_ref
                        writer.write_atom(atom)
                        atoms.append(atom)

        logger.info(
            "Imported %s/%s/%s: %d sentences × %d ch @ %.0f Hz",
            subject_id, task, text_id, len(atoms), n_channels, srate,
        )
        return atoms, ch_infos, all_warnings

    @staticmethod
    def _load_wordbounds_v1(
        prep_dir: Path,
        text_id: str,
        subject_id: str,
        task: str,
    ) -> Optional[np.ndarray]:
        """Load ZuCo 1.0 wordbounds (per-subject naming).

        ZuCo 1.0 uses: wordbounds_X_SUBJ.mat (e.g. wordbounds_NR_ZAB.mat)
        or wordbounds_SNR1_ZAB.mat for individual SR texts.
        """
        import scipy.io as sio

        # Try per-text naming first
        p1 = prep_dir / subject_id / f"wordbounds_{text_id}_{subject_id}.mat"
        if p1.exists():
            try:
                mat = sio.loadmat(str(p1))
                return mat.get("wordbounds")
            except Exception:
                pass

        # Try task-level naming (NR has single file)
        task_prefix = text_id.rstrip("0123456789")  # "SR" / "NR" / "TSR"
        p2 = prep_dir / subject_id / f"wordbounds_{task_prefix}_{subject_id}.mat"
        if p2.exists():
            try:
                mat = sio.loadmat(str(p2))
                return mat.get("wordbounds")
            except Exception:
                pass

        # Try global wordbounds
        p3 = prep_dir / f"wordbounds_{text_id}.mat"
        if p3.exists():
            try:
                mat = sio.loadmat(str(p3))
                return mat.get("wordbounds")
            except Exception:
                pass

        return None


# Auto-register
register_importer("zuco1", Zuco1Importer)
