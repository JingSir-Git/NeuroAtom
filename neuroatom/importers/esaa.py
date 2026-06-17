"""ESAA Importer: SCUT spatial auditory attention decoding (HRTF).

Spatial AAD dataset (South China University of Technology). Subjects attend one
of two competing speech streams rendered at different spatial directions (the
``clean/`` and ``hrtf/`` wav folders hold the anechoic and HRTF-spatialised
stimuli). 32 trials/subject, continuous recording at 1000 Hz, 64-channel EEG
(10-10) + ECG.

Each subject is one continuous ``.mat`` (BrainVision-style export) with the EEG
stored as one array per channel plus a ``Markers`` struct array. Trials are
delimited by ``trail{k}D`` / ``trail{k}S`` / ``trail{k}E`` markers; the trial
window is taken as 55 s forward from the ``S`` (start) marker, exactly matching
the dataset's own ``preprocess.py``.

Attention labels follow the SCUT protocol (``db/SCUT.py``): a fixed 32-trial
``[direction, speaker]`` table, with a documented reorder + label reversal for
subjects > 8 (``scut_order``). We replicate that mapping verbatim so labels match
the dataset's official preprocessing. ``direction`` / ``speaker`` are the two
binary AAD targets. A handful of trials flagged for poor behavioural answers are
imported but annotated ``excluded=true``.

Data layout::

    ESAA/
        clean/Trail*.wav   hrtf/Trail*.wav    # stimuli (not imported)
        preprocess/db/SCUT.py                  # label protocol (source of truth)
        S1/S1/S1.mat ... S20/S20/S20.mat
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
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import AtomType, ChannelStatus, ChannelType
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.mat_compat import require_mat_v5
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

_TRIAL_SECONDS = 55.0

# --- SCUT protocol (verbatim from preprocess/db/SCUT.py) ---------------------
# Per-trial [direction, speaker] labels, trial order 1..32.
_SCUT_LABEL: List[List[int]] = [
    [0, 0], [1, 0], [0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [1, 1],
    [1, 0], [0, 0], [1, 0], [0, 0], [1, 1], [0, 1], [1, 1], [0, 1],
    [0, 0], [1, 0], [0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [1, 1],
    [1, 0], [0, 0], [1, 0], [0, 0], [1, 1], [0, 1], [1, 1], [0, 1],
]
# Subjects > 8: trial/label order differs and labels are reversed.
_SCUT_SUF_ORDER = [
    n - 1 for n in [24, 8, 32, 31, 7, 15, 29, 6, 12, 26, 9, 1, 5, 28, 20, 11,
                    30, 21, 2, 4, 10, 13, 17, 25, 14, 22, 23, 19, 16, 27, 18, 3]
]
# Trials (1-indexed) flagged for poor behavioural answers.
_SCUT_REMOVE: Dict[str, set] = {
    "1": {1, 2, 3, 9, 14, 24, 31}, "2": {9, 29}, "3": {1, 9, 19}, "4": {31},
    "5": {14}, "6": {7, 9, 10, 13, 15, 29}, "7": {9, 25},
    "8": {25, 26, 27, 28, 29, 30, 31, 32}, "9": {15}, "10": {12, 15, 16},
    "11": {12, 17, 30}, "13": {27, 29}, "14": {9, 14, 26, 27}, "16": {1},
}
# Canonical EEG channel order (64).
_CH_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
    "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz", "FC1", "FC2", "CP1", "CP2",
    "FC5", "FC6", "CP5", "CP6", "TP9", "TP10", "POz", "F1", "F2", "C1", "C2",
    "P1", "P2", "AF3", "AF4", "FC3", "FC4", "CP3", "CP4", "PO3", "PO4", "F5",
    "F6", "C5", "C6", "P5", "P6", "AF7", "AF8", "FT7", "FT8", "TP7", "TP8",
    "PO7", "PO8", "FT9", "FT10", "Fpz", "CPz", "FCz",
]

_SUBJECT_RE = re.compile(r"S(\d+)", re.IGNORECASE)


def _scut_labels(sub_num: int) -> List[List[int]]:
    """Per-trial [direction, speaker] labels for a subject (SCUT scut_order)."""
    label = [list(row) for row in _SCUT_LABEL]
    if sub_num > 8:
        label = [label[i] for i in _SCUT_SUF_ORDER]      # _adjust_order
        label = [[1 - v for v in row] for row in label]   # _reverse_label
    return label


def _marker_position(mk: Any) -> Optional[int]:
    pos = getattr(mk, "Position", None)
    if pos is None:
        return None
    arr = np.ravel(pos)
    return int(arr[0]) if arr.size else None


def _trial_segments(markers: np.ndarray) -> List[Tuple[int, int]]:
    """Return [(global_trial_num, onset_sample), ...] from ``trail{N}S`` markers.

    Marker descriptions carry the GLOBAL trial number (e.g. ``trail22S``), so this
    works whether a subject is one file or split across files — S4_1 holds
    trail1-21 and S4_2 holds trail22-32, each tagged with its true trial number.
    The ``S`` (start) marker is the trial onset (matching preprocess.py's 3k+2).
    """
    out: List[Tuple[int, int]] = []
    for mk in np.atleast_1d(markers):
        desc = str(getattr(mk, "Description", ""))
        m = re.match(r"trail(\d+)S$", desc, re.IGNORECASE)
        if not m:
            continue
        pos = _marker_position(mk)
        if pos is not None:
            out.append((int(m.group(1)), pos))
    return out


def _find_subject_mats(subject_dir: Path) -> List[Path]:
    """All data .mat files for a subject, across ESAA's layout variants.

    Handles ``Sx/Sx/Sx.mat`` (single), ``Sx/Sx/Sx_1.mat`` + ``Sx_2.mat`` (split
    recordings, e.g. S4/S9/S12/S13/S14/S17), and ``Sx/Sx.mat`` (S8). Excludes
    ``__MACOSX`` resource-fork junk (``._*`` files).
    """
    found = list(subject_dir.glob("*.mat")) + list(subject_dir.glob("*/*.mat"))
    mats = [
        p for p in found
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    ]
    return sorted(set(mats), key=lambda p: p.name)


class ESAAImporter(BaseImporter):
    """Importer for the ESAA / SCUT spatial AAD dataset."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return (
                path.suffix.lower() == ".mat"
                and not path.name.startswith("._")
                and "__MACOSX" not in path.parts
                and bool(re.match(r"S\d+", path.stem))
            )
        if path.is_dir():
            return any(
                sub.is_dir() and _SUBJECT_RE.match(sub.name) and _find_subject_mats(sub)
                for sub in path.glob("S*")
            )
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("ESAA uses import_subject().")

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------

    def _build_channel_infos(self, names: List[str], sfreq: float) -> List[ChannelInfo]:
        ch_infos: List[ChannelInfo] = []
        for idx, name in enumerate(names):
            if name in self.task_config.exclude_channels:
                continue
            ch_type = ChannelType.OTHER if name.upper() == "ECG" else ChannelType.EEG
            override = self.task_config.channel_type_overrides.get(name)
            if override:
                ch_type = ChannelType(override)
            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{len(ch_infos):03d}",
                index=len(ch_infos),
                name=name,
                standard_name=standardize_channel_name(name),
                type=ch_type,
                unit=self.task_config.signal_unit or "uV",
                sampling_rate=sfreq,
                status=ChannelStatus.UNKNOWN,
            ))
        return ch_infos

    def import_subject(
        self,
        mat_path: Path,
        subject_id: Optional[str] = None,
        session_id: str = "ses-01",
        max_trials: Optional[int] = None,
    ) -> ImportResult:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        mat_path = Path(mat_path)
        require_mat_v5(mat_path, "ESAAImporter")
        if subject_id is None:
            mm = re.match(r"(S\d+)", mat_path.stem)
            subject_id = mm.group(1) if mm else mat_path.stem
        sub_num = int(re.sub(r"\D", "", subject_id) or 0)

        ch_order = [c for c in (_CH_NAMES + ["ECG"])]
        logger.info("Loading ESAA .mat: %s", mat_path)
        m = sio.loadmat(
            str(mat_path), squeeze_me=True, struct_as_record=False,
            variable_names=ch_order + ["Markers", "SampleRate"],
        )
        sfreq = float(np.ravel(m.get("SampleRate", 1000))[0])
        # Keep only channels actually present in the file.
        ch_order = [c for c in ch_order if c in m and m[c] is not None]
        markers = np.atleast_1d(m["Markers"])
        segments = _trial_segments(markers)
        labels = _scut_labels(sub_num)
        win = int(_TRIAL_SECONDS * sfreq)
        n_total = int(np.ravel(m[ch_order[0]]).shape[0])

        if max_trials is not None:
            segments = segments[:max_trials]

        channel_infos = self._build_channel_infos(ch_order, sfreq)
        keep_names = [ci.name for ci in channel_infos]
        channel_ids = [ci.channel_id for ci in channel_infos]

        dataset_id = self.task_config.dataset_id
        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id, sfreq)

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")
        poor = _SCUT_REMOVE.get(str(sub_num), set())

        stored_atoms: List[Atom] = []
        all_warnings: List[str] = []

        for trial_num, onset in segments:
            end = min(onset + win, n_total)
            if end <= onset:
                continue
            seg = np.stack([np.asarray(m[name])[onset:end] for name in keep_names]).astype(np.float32)
            seg = np.ascontiguousarray(seg)
            n_samples = seg.shape[1]

            signal, storage_unit, orig_unit = convert_to_storage_unit(
                seg, source_unit="uV", pool_config=self.pool.config,
            )

            run_id = f"trial_{trial_num:02d}"
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            idx = trial_num - 1
            direction, speaker = (labels[idx] if 0 <= idx < len(labels) else ("unknown", "unknown"))
            annotations: List[Any] = [
                CategoricalAnnotation(
                    annotation_id=f"ann_dir_{run_id}",
                    name="attended_direction", value=str(direction),
                ),
                CategoricalAnnotation(
                    annotation_id=f"ann_spk_{run_id}",
                    name="attended_speaker", value=str(speaker),
                ),
                NumericAnnotation(
                    annotation_id=f"ann_trial_{run_id}",
                    name="trial_number", numeric_value=float(trial_num),
                ),
            ]
            if trial_num in poor:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_excl_{run_id}",
                    name="excluded", value="true",
                ))

            atom_id = compute_atom_id(
                dataset_id=dataset_id, subject_id=subject_id,
                session_id=session_id, run_id=run_id, onset_sample=0,
            )
            atom = Atom(
                atom_id=atom_id, atom_type=AtomType.TRIAL,
                dataset_id=dataset_id, subject_id=subject_id,
                session_id=session_id, run_id=run_id, trial_index=trial_num,
                signal_ref=SignalRef(
                    file_path="", internal_path="",
                    shape=(len(channel_ids), n_samples),
                ),
                temporal=TemporalInfo(
                    onset_sample=0, onset_seconds=0.0,
                    duration_samples=n_samples, duration_seconds=n_samples / sfreq,
                ),
                channel_ids=channel_ids, n_channels=len(channel_ids),
                sampling_rate=sfreq, signal_unit=storage_unit, original_unit=orig_unit,
                annotations=annotations,
                custom_fields={
                    "attended_direction": str(direction),
                    "attended_speaker": str(speaker),
                    "trial_number": trial_num,
                },
            )

            all_warnings.extend(validate_signal(
                signal=signal, atom_id=atom.atom_id,
                config=self.pool.config.get("import", {}), signal_unit=storage_unit,
            ))

            with ShardManager(
                pool_root=self.pool.root, dataset_id=dataset_id,
                subject_id=subject_id, session_id=session_id, run_id=run_id,
                max_shard_size_mb=max_shard_mb, compression=compression,
            ) as shard_mgr:
                atom.signal_ref = shard_mgr.write_atom_signal(atom.atom_id, signal, None)

            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, dataset_id, subject_id, session_id, run_id,
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                writer.write_atom(atom)

            self.pool.register_run(RunMeta(
                run_id=run_id, session_id=session_id, subject_id=subject_id,
                dataset_id=dataset_id, run_index=trial_num,
                task_type=self.task_config.task_type, n_events=0, n_trials=1,
                paradigm_details={
                    "attended_direction": str(direction),
                    "attended_speaker": str(speaker),
                },
            ))
            self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
            stored_atoms.append(atom)

        logger.info(
            "Imported ESAA subject %s: %d trials", subject_id, len(stored_atoms),
        )
        run_meta = RunMeta(
            run_id=stored_atoms[0].run_id if stored_atoms else "trial_01",
            session_id=session_id, subject_id=subject_id, dataset_id=dataset_id,
            task_type=self.task_config.task_type, n_trials=len(stored_atoms),
        )
        return ImportResult(
            atoms=stored_atoms, run_meta=run_meta,
            channel_infos=channel_infos, warnings=all_warnings,
        )

    def import_dataset(
        self,
        root: Path,
        subjects: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
        max_trials: Optional[int] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        dataset_id = self.task_config.dataset_id

        # One entry per subject dir, paired with all its .mat files
        # (single Sx/Sx/Sx.mat, split Sx_1+Sx_2, or S8/S8.mat).
        subject_dirs = sorted(
            (d for d in root.glob("S*") if d.is_dir() and _SUBJECT_RE.match(d.name)),
            key=lambda d: int(re.sub(r"\D", "", d.name) or 0),
        )
        if subjects:
            subject_dirs = [d for d in subject_dirs if d.name in set(subjects)]
        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]

        subject_files = [(d.name, _find_subject_mats(d)) for d in subject_dirs]
        subject_files = [(s, fs) for s, fs in subject_files if fs]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len(subject_files), original_format="mat",
        ))

        results: List[ImportResult] = []
        for sub_id, files in subject_files:
            self.pool.register_subject(SubjectMeta(
                subject_id=sub_id, dataset_id=dataset_id,
            ))
            # Merge a subject's (possibly split) recordings into one result.
            atoms: List[Atom] = []
            warnings_acc: List[str] = []
            chans: List[ChannelInfo] = []
            for mat_file in files:
                try:
                    r = self.import_subject(
                        mat_file, subject_id=sub_id, max_trials=max_trials,
                    )
                    atoms.extend(r.atoms)
                    warnings_acc.extend(r.warnings)
                    chans = r.channel_infos or chans
                except Exception as e:
                    logger.error("Failed to import %s: %s", mat_file.name, e)
            if atoms:
                results.append(ImportResult(
                    atoms=atoms,
                    run_meta=RunMeta(
                        run_id=atoms[0].run_id, session_id="ses-01",
                        subject_id=sub_id, dataset_id=dataset_id,
                        task_type=self.task_config.task_type, n_trials=len(atoms),
                    ),
                    channel_infos=chans, warnings=warnings_acc,
                ))

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("ESAA import complete: %d subjects", len(results))
        return results


# Auto-register
register_importer("esaa", ESAAImporter)
