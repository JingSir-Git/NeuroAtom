"""AV-GC-AAD Importer: Audio-Visual Gaze-Controlled Auditory Attention Decoding.

KU Leuven (2024) dataset. Subjects attend one of two competing Flemish speech
streams (left vs right) under three visual conditions:

    - NoVisuals:   audio only
    - FixedVideo:  a static talking-face video
    - MovingVideo: a moving talking-face video (gaze-controlled)

Distinctive feature — **dynamic attention**: within each ~600 s trial the
attended side switches at ``switch_times`` (typically once, at 300 s). Unlike
the static KUL/DTU paradigms, a single trial therefore carries *two* attention
labels over time. The dataset ships precomputed attended/unattended speech
envelopes that already account for the switch, alongside the raw left/right
envelopes — all four are stored as ``ContinuousAnnotation`` companion arrays.

Data layout (flat .mat per subject)::

    AV-GC-AAD/
        2024-AV-GC-AAD-sub01_preprocessed.mat
        2024-AV-GC-AAD-sub03_preprocessed.mat
        ...

Each subject .mat (MATLAB v5) contains:
    data            (n_trials,) object — each (n_samples, n_channels) float32, µV
    fs              scalar — sampling rate (128 Hz, preprocessed)
    metadata        (n_trials,) struct — FileHeader (channel labels) + RawData
    conditionID     (n_trials,) str — visual condition (MovingVideo1, NoVisuals1, ...)
    initAttention   (n_trials,) str — initially attended side ('left'/'right')
    randomization   (n_trials,) struct — left/right wav, first_attended_side,
                    switch_times, comprehension_question, subject_answer
    stimulus        struct — attendedEnvelopes / unattendedEnvelopes /
                    leftEnvelopes / rightEnvelopes, each (n_trials,) object

Each trial becomes one TRIAL atom flagged ``dynamic_attention=true``.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

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
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.mat_compat import require_mat_v5
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

_SUBJECT_RE = re.compile(r"(sub\d+)", re.IGNORECASE)

# Visual condition (conditionID has a trailing repetition digit, e.g. "MovingVideo2")
_VISUAL_TYPE = {
    "movingvideo": "moving_video",
    "fixedvideo": "fixed_video",
    "novisuals": "no_visuals",
}

# The four precomputed envelopes shipped per trial.
_ENVELOPE_FIELDS = {
    "attendedEnvelopes": "attended_envelope",
    "unattendedEnvelopes": "unattended_envelope",
    "leftEnvelopes": "left_envelope",
    "rightEnvelopes": "right_envelope",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _scalar_str(value: Any) -> str:
    """Coerce a squeezed scipy value to a plain stripped string."""
    if value is None:
        return ""
    arr = np.ravel(value)
    if arr.size == 0:
        return ""
    return str(arr[0]).strip()


def _as_1d_f32(value: Any) -> Optional[np.ndarray]:
    """Coerce an envelope value to a 1-D float32 array, or None if empty."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).ravel()
    return arr if arr.size else None


def _switch_times(rand: Any) -> List[float]:
    """Extract the within-trial attention switch times (seconds)."""
    sw = getattr(rand, "switch_times", None)
    if sw is None:
        return []
    arr = np.atleast_1d(np.asarray(sw, dtype=float)).ravel()
    return [float(x) for x in arr if np.isfinite(x)]


def _extract_channels(meta_trial: Any) -> Tuple[List[str], str]:
    """Read channel labels + physical unit from a trial's FileHeader struct."""
    fh = getattr(meta_trial, "FileHeader", None)
    chans = getattr(fh, "Channels", None) if fh is not None else None
    names: List[str] = []
    unit = "uV"
    if chans is None:
        return names, unit
    for ch in np.atleast_1d(chans):
        label = getattr(ch, "Label", None)
        if label is None:
            label = getattr(ch, "label", None)
        names.append(str(label) if label is not None else f"Ch_{len(names) + 1}")
    first = np.atleast_1d(chans)[0]
    dim = getattr(first, "PhysicalDimension", None)
    if dim:
        unit = str(dim).strip()
    return names, unit


def _channel_type(name: str) -> ChannelType:
    """EXG external electrodes → EOG (KU Leuven BioSemi convention); else EEG."""
    if name.upper().startswith("EXG"):
        return ChannelType.EOG
    return ChannelType.EEG


def _parse_subject_mat(mat_path: Path) -> Dict[str, Any]:
    """Parse one AV-GC-AAD subject .mat into a flat dict of per-trial records."""
    require_mat_v5(mat_path, "AVGCAADImporter")
    logger.info("Loading AV-GC-AAD .mat: %s", mat_path)
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    fs = float(np.ravel(mat["fs"])[0])
    subj_id_full = _scalar_str(mat.get("subjID", ""))

    data = np.atleast_1d(mat["data"])
    n_trials = len(data)
    condition_ids = np.atleast_1d(mat["conditionID"])
    init_attention = np.atleast_1d(mat["initAttention"])
    metadata = np.atleast_1d(mat["metadata"])
    randomization = np.atleast_1d(mat["randomization"])
    stimulus = mat["stimulus"]

    # Channel labels are identical across trials — read from the first.
    ch_names, unit = _extract_channels(metadata.flat[0])

    trials: List[Dict[str, Any]] = []
    for i in range(n_trials):
        eeg = np.asarray(data.flat[i])  # (n_samples, n_channels)
        rand = randomization.flat[i]

        envelopes: Dict[str, np.ndarray] = {}
        for mat_field, ann_name in _ENVELOPE_FIELDS.items():
            field = getattr(stimulus, mat_field, None)
            if field is None:
                continue
            env = _as_1d_f32(np.atleast_1d(field).flat[i])
            if env is not None:
                envelopes[ann_name] = env

        cond_raw = _scalar_str(condition_ids.flat[i])
        visual_type = _VISUAL_TYPE.get(re.sub(r"\d+$", "", cond_raw).lower(), "")

        trials.append({
            "eeg": eeg,
            "ch_names": ch_names,
            "unit": unit,
            "srate": fs,
            "trial_index": i + 1,
            "visual_condition": cond_raw,
            "visual_type": visual_type,
            "init_attention": _scalar_str(init_attention.flat[i]).lower(),
            "first_attended_side": _scalar_str(getattr(rand, "first_attended_side", "")),
            "switch_times": _switch_times(rand),
            "left_audio": _scalar_str(getattr(rand, "left", "")),
            "right_audio": _scalar_str(getattr(rand, "right", "")),
            "comprehension_question": _scalar_str(getattr(rand, "comprehension_question", "")),
            "subject_answer": _scalar_str(getattr(rand, "subject_answer", "")),
            "envelopes": envelopes,
        })

    return {
        "fs": fs,
        "subj_id_full": subj_id_full,
        "ch_names": ch_names,
        "unit": unit,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

class AVGCAADImporter(BaseImporter):
    """Importer for the AV-GC-AAD audio-visual auditory attention dataset."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_dir():
            return any(path.glob("*AV-GC-AAD*.mat"))
        if path.suffix.lower() != ".mat":
            return False
        return "av-gc-aad" in path.name.lower()

    # AV-GC uses the multi-trial import_subject entry point, not load_raw.
    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError(
            "AVGCAADImporter uses import_subject() for multi-trial .mat files."
        )

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------

    def _build_channel_infos(
        self, ch_names: List[str], sfreq: float,
    ) -> List[ChannelInfo]:
        ch_infos: List[ChannelInfo] = []
        for idx, name in enumerate(ch_names):
            if name in self.task_config.exclude_channels:
                continue
            ch_type = _channel_type(name)
            override = self.task_config.channel_type_overrides.get(name)
            if override:
                ch_type = ChannelType(override)
            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{idx:03d}",
                index=idx,
                name=name,
                standard_name=standardize_channel_name(name),
                type=ch_type,
                unit=self.task_config.signal_unit or "uV",
                sampling_rate=sfreq,
                status=ChannelStatus.UNKNOWN,
            ))
        return ch_infos

    def _build_trial_atom(
        self,
        trial: Dict[str, Any],
        channel_infos: List[ChannelInfo],
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
    ) -> Tuple[Atom, np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Build (atom, signal, companion_arrays) for one trial."""
        eeg = np.asarray(trial["eeg"], dtype=np.float32)
        # Orient to (n_channels, n_samples).
        if eeg.ndim == 2 and eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        n_channels, n_samples = eeg.shape
        srate = trial["srate"]

        channel_ids = [ci.channel_id for ci in channel_infos]
        # Keep signal channels aligned with the (non-excluded) channel_infos.
        keep_idx = [ci.index for ci in channel_infos if ci.index < n_channels]
        signal = np.ascontiguousarray(eeg[keep_idx, :])

        signal, storage_unit, orig_unit = convert_to_storage_unit(
            signal, source_unit=trial["unit"], pool_config=self.pool.config,
        )

        prefix = f"{run_id}"
        annotations: List[Any] = []
        ann_arrays: Dict[str, np.ndarray] = {}

        annotations.append(CategoricalAnnotation(
            annotation_id=f"ann_initatt_{prefix}",
            name="init_attention",
            value=trial["init_attention"] or "unknown",
        ))
        if trial["first_attended_side"]:
            annotations.append(CategoricalAnnotation(
                annotation_id=f"ann_firstside_{prefix}",
                name="first_attended_side",
                value=trial["first_attended_side"],
            ))
        if trial["visual_condition"]:
            annotations.append(CategoricalAnnotation(
                annotation_id=f"ann_viscond_{prefix}",
                name="visual_condition",
                value=trial["visual_condition"],
            ))
        if trial["visual_type"]:
            annotations.append(CategoricalAnnotation(
                annotation_id=f"ann_vistype_{prefix}",
                name="visual_type",
                value=trial["visual_type"],
            ))
        # Dynamic-attention flag + switch time(s): this is what sets AV-GC apart
        # from the static KUL/DTU paradigms.
        annotations.append(CategoricalAnnotation(
            annotation_id=f"ann_dynamic_{prefix}",
            name="dynamic_attention",
            value="true" if trial["switch_times"] else "false",
        ))
        if trial["switch_times"]:
            annotations.append(NumericAnnotation(
                annotation_id=f"ann_switch_{prefix}",
                name="switch_time_seconds",
                numeric_value=trial["switch_times"][0],
            ))
        if trial["left_audio"]:
            annotations.append(TextAnnotation(
                annotation_id=f"ann_laudio_{prefix}",
                name="left_audio",
                text_value=trial["left_audio"],
            ))
        if trial["right_audio"]:
            annotations.append(TextAnnotation(
                annotation_id=f"ann_raudio_{prefix}",
                name="right_audio",
                text_value=trial["right_audio"],
            ))
        if trial["comprehension_question"]:
            annotations.append(TextAnnotation(
                annotation_id=f"ann_question_{prefix}",
                name="comprehension_question",
                text_value=trial["comprehension_question"],
            ))
        if trial["subject_answer"]:
            annotations.append(TextAnnotation(
                annotation_id=f"ann_answer_{prefix}",
                name="subject_answer",
                text_value=trial["subject_answer"],
            ))

        # Speech envelopes → ContinuousAnnotation + HDF5 companion arrays.
        for ann_name, env in trial["envelopes"].items():
            ann_arrays[ann_name] = env
            annotations.append(ContinuousAnnotation(
                annotation_id=f"ann_{ann_name}_{prefix}",
                name=ann_name,
                domain="stimulus",
                scope="timepoint",
                data_ref=SignalRef(
                    file_path="__stim_placeholder__",
                    internal_path=f"__placeholder__/annotations/{ann_name}",
                    shape=(int(env.shape[0]),),
                ),
                data_sampling_rate=srate,
                alignment_method="trigger_locked",
                custom_fields={"stimulus_type": ann_name.replace("_envelope", "")},
            ))

        atom_id = compute_atom_id(
            dataset_id=dataset_id,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            onset_sample=0,
        )
        atom = Atom(
            atom_id=atom_id,
            atom_type=AtomType.TRIAL,
            dataset_id=dataset_id,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            trial_index=trial["trial_index"],
            signal_ref=SignalRef(
                file_path="", internal_path="",
                shape=(len(channel_ids), n_samples),
            ),
            temporal=TemporalInfo(
                onset_sample=0,
                onset_seconds=0.0,
                duration_samples=n_samples,
                duration_seconds=n_samples / srate,
            ),
            channel_ids=channel_ids,
            n_channels=len(channel_ids),
            sampling_rate=srate,
            signal_unit=storage_unit,
            original_unit=orig_unit,
            annotations=annotations,
            custom_fields={
                "visual_type": trial["visual_type"],
                "switch_times": trial["switch_times"],
            },
        )
        return atom, signal.astype(np.float32), (ann_arrays or None)

    def import_subject(
        self,
        mat_path: Path,
        subject_id: Optional[str] = None,
        session_id: str = "ses-01",
        max_trials: Optional[int] = None,
    ) -> ImportResult:
        """Import all trials from a single subject .mat file."""
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        mat_path = Path(mat_path)
        parsed = _parse_subject_mat(mat_path)

        if subject_id is None:
            m = _SUBJECT_RE.search(mat_path.name)
            subject_id = m.group(1).lower() if m else mat_path.stem

        dataset_id = self.task_config.dataset_id
        trials = parsed["trials"]
        if max_trials is not None:
            trials = trials[:max_trials]

        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id, parsed["fs"])

        channel_infos = self._build_channel_infos(parsed["ch_names"], parsed["fs"])

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        stored_atoms: List[Atom] = []
        all_warnings: List[str] = []

        for trial in trials:
            run_id = f"trial_{trial['trial_index']:02d}"
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            atom, signal, ann_arrays = self._build_trial_atom(
                trial, channel_infos, dataset_id, subject_id, session_id, run_id,
            )

            warnings = validate_signal(
                signal=signal,
                atom_id=atom.atom_id,
                config=self.pool.config.get("import", {}),
                signal_unit=atom.signal_unit,
            )
            all_warnings.extend(warnings)

            with ShardManager(
                pool_root=self.pool.root, dataset_id=dataset_id,
                subject_id=subject_id, session_id=session_id, run_id=run_id,
                max_shard_size_mb=max_shard_mb, compression=compression,
            ) as shard_mgr:
                sig_ref = shard_mgr.write_atom_signal(atom.atom_id, signal, ann_arrays)
                atom.signal_ref = sig_ref

            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, dataset_id, subject_id, session_id, run_id,
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                writer.write_atom(atom)

            self.pool.register_run(RunMeta(
                run_id=run_id, session_id=session_id, subject_id=subject_id,
                dataset_id=dataset_id, run_index=trial["trial_index"],
                task_type=self.task_config.task_type, n_events=0, n_trials=1,
                paradigm_details={
                    "visual_condition": trial["visual_condition"],
                    "init_attention": trial["init_attention"],
                    "switch_times": trial["switch_times"],
                },
            ))
            self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
            stored_atoms.append(atom)

        logger.info(
            "Imported AV-GC-AAD subject %s: %d trials", subject_id, len(stored_atoms),
        )
        run_meta = RunMeta(
            run_id=f"trial_{trials[0]['trial_index']:02d}" if trials else "trial_01",
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
        """Import every subject .mat under *root*."""
        root = Path(root)
        dataset_id = self.task_config.dataset_id

        mat_files = sorted(root.glob("*AV-GC-AAD*.mat"))
        if subjects:
            wanted = {s.lower() for s in subjects}
            mat_files = [
                f for f in mat_files
                if (m := _SUBJECT_RE.search(f.name)) and m.group(1).lower() in wanted
            ]
        if max_subjects:
            mat_files = mat_files[:max_subjects]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id,
            name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len(mat_files),
            original_format="mat",
        ))

        results: List[ImportResult] = []
        for mat_file in mat_files:
            m = _SUBJECT_RE.search(mat_file.name)
            subject_id = m.group(1).lower() if m else mat_file.stem
            self.pool.register_subject(SubjectMeta(
                subject_id=subject_id, dataset_id=dataset_id,
            ))
            try:
                results.append(self.import_subject(
                    mat_file, subject_id=subject_id, max_trials=max_trials,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", mat_file.name, e)

        try:
            tier = self.pool.assess_quality(dataset_id)
            if tier:
                logger.info("Quality tier for %s: %s", dataset_id, tier)
        except Exception:
            pass

        logger.info(
            "AV-GC-AAD import complete: %d subjects", len(results),
        )
        return results


# Auto-register
register_importer("avgc_aad", AVGCAADImporter)
