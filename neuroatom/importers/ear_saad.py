"""Ear-SAAD Importer: ear-EEG Spatial Auditory Attention Decoding.

Preprocessed audio-visual AAD dataset combining scalp EEG with in-ear
(cEEGrid-style) electrodes. Each subject attends one of two competing speech
streams; half the trials add a talking-face video (``videoCondition``).

Data is heavily downsampled (20 Hz) for envelope-tracking analysis and ships
the two speaker speech envelopes plus acoustic-edge (onset) features aligned
sample-for-sample with the EEG. Preprocessing leaves NaN gaps where segments
were rejected — these are imported faithfully and surfaced as validation
warnings rather than silently patched.

Data layout::

    Ear-SAAD/
        bids_dataset/              # raw BIDS (not imported here)
        preprocessedData/
            dataSubject1.mat ... dataSubject15.mat

Each ``dataSubjectN.mat`` (MATLAB v5) contains:
    eegTrials       (n_trials,) object — each (n_samples, n_channels) float64
    channels        (n_channels,) str — labels (scalp + M1/M2 + cER* ear-EEG)
    fs              scalar — sampling rate (20 Hz)
    attendedEar     (n_trials,) str — attended ear ('1'/'2')
    attSpeaker      (n_trials,) str — attended speaker index ('1'/'2')
    videoCondition  (n_trials,) str — '0' (audio only) / '1' (audio-visual)
    envelopes       (n_trials,) object — each (n_samples, 2): speaker1/speaker2
    acousticEdges   (n_trials,) object — each (n_samples, 2): onset features

Each trial → one TRIAL atom with static attention labels and four companion
arrays (envelope_speaker1/2, acoustic_edges_speaker1/2).
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
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.mat_compat import require_mat_v5
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

_SUBJECT_RE = re.compile(r"dataSubject(\d+)", re.IGNORECASE)
_MASTOIDS = {"M1", "M2"}
_EOG = {"HEO", "VEO", "HEOG", "VEOG", "EOG"}


def _channel_type(name: str) -> Tuple[ChannelType, bool]:
    """Map a channel label to (type, is_eareeg). cER* are in-ear electrodes."""
    upper = name.upper()
    if upper.startswith("CER"):
        return ChannelType.OTHER, True  # in-ear EEG
    if name in _MASTOIDS:
        return ChannelType.REF, False
    if upper in _EOG:
        return ChannelType.EOG, False
    return ChannelType.EEG, False


def _scalar_str(value: Any) -> str:
    arr = np.ravel(value)
    return str(arr[0]).strip() if arr.size else ""


def _column(arr: Any, idx: int) -> Optional[np.ndarray]:
    """Return column *idx* of a 2-D companion array as 1-D float32."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2 or idx >= a.shape[1]:
        return None
    col = np.ascontiguousarray(a[:, idx])
    return col if col.size else None


def _parse_subject_mat(mat_path: Path) -> Dict[str, Any]:
    """Parse one Ear-SAAD subject .mat into per-trial records."""
    require_mat_v5(mat_path, "EarSAADImporter")
    logger.info("Loading Ear-SAAD .mat: %s", mat_path)
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    fs = float(np.ravel(mat["fs"])[0])
    ch_names = [str(c) for c in np.atleast_1d(mat["channels"])]

    eeg_trials = np.atleast_1d(mat["eegTrials"])
    envelopes = np.atleast_1d(mat["envelopes"])
    edges = np.atleast_1d(mat["acousticEdges"])
    att_ear = np.atleast_1d(mat["attendedEar"])
    att_spk = np.atleast_1d(mat["attSpeaker"])
    video = np.atleast_1d(mat["videoCondition"])

    n_trials = len(eeg_trials)
    trials: List[Dict[str, Any]] = []
    for i in range(n_trials):
        eeg = np.asarray(eeg_trials.flat[i])  # (n_samples, n_channels)
        env = envelopes.flat[i] if i < len(envelopes) else None
        edge = edges.flat[i] if i < len(edges) else None

        companions: Dict[str, np.ndarray] = {}
        for col, name in ((0, "envelope_speaker1"), (1, "envelope_speaker2")):
            c = _column(env, col)
            if c is not None:
                companions[name] = c
        for col, name in ((0, "acoustic_edges_speaker1"), (1, "acoustic_edges_speaker2")):
            c = _column(edge, col)
            if c is not None:
                companions[name] = c

        vid = _scalar_str(video.flat[i]) if i < len(video) else ""
        trials.append({
            "eeg": eeg,
            "srate": fs,
            "trial_index": i + 1,
            "attended_ear": _scalar_str(att_ear.flat[i]) if i < len(att_ear) else "",
            "attended_speaker": _scalar_str(att_spk.flat[i]) if i < len(att_spk) else "",
            "video_condition": "video" if vid == "1" else "no_video",
            "companions": companions,
        })

    return {"fs": fs, "ch_names": ch_names, "trials": trials}


class EarSAADImporter(BaseImporter):
    """Importer for the Ear-SAAD preprocessed ear-EEG AAD dataset."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_dir():
            if any(path.glob("dataSubject*.mat")):
                return True
            # Also accept the dataset root that holds preprocessedData/.
            return any((path / "preprocessedData").glob("dataSubject*.mat")) \
                if (path / "preprocessedData").is_dir() else False
        if path.suffix.lower() != ".mat":
            return False
        return bool(_SUBJECT_RE.search(path.name))

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError(
            "EarSAADImporter uses import_subject() for multi-trial .mat files."
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
            ch_type, is_eareeg = _channel_type(name)
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
                custom_fields={"is_eareeg": True} if is_eareeg else {},
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
        eeg = np.asarray(trial["eeg"], dtype=np.float32)
        if eeg.ndim == 2 and eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T  # → (n_channels, n_samples)
        n_channels, n_samples = eeg.shape
        srate = trial["srate"]

        channel_ids = [ci.channel_id for ci in channel_infos]
        keep_idx = [ci.index for ci in channel_infos if ci.index < n_channels]
        signal = np.ascontiguousarray(eeg[keep_idx, :])

        signal, storage_unit, orig_unit = convert_to_storage_unit(
            signal, source_unit="uV", pool_config=self.pool.config,
        )

        prefix = run_id
        annotations: List[Any] = []
        ann_arrays: Dict[str, np.ndarray] = {}

        if trial["attended_ear"]:
            annotations.append(CategoricalAnnotation(
                annotation_id=f"ann_ear_{prefix}",
                name="attended_ear", value=trial["attended_ear"],
            ))
        if trial["attended_speaker"]:
            annotations.append(CategoricalAnnotation(
                annotation_id=f"ann_spk_{prefix}",
                name="attended_speaker", value=trial["attended_speaker"],
            ))
        annotations.append(CategoricalAnnotation(
            annotation_id=f"ann_video_{prefix}",
            name="video_condition", value=trial["video_condition"],
        ))

        for name, arr in trial["companions"].items():
            ann_arrays[name] = arr
            domain = "stimulus" if name.startswith("envelope") else "derived"
            annotations.append(ContinuousAnnotation(
                annotation_id=f"ann_{name}_{prefix}",
                name=name, domain=domain, scope="timepoint",
                data_ref=SignalRef(
                    file_path="__stim_placeholder__",
                    internal_path=f"__placeholder__/annotations/{name}",
                    shape=(int(arr.shape[0]),),
                ),
                data_sampling_rate=srate,
                alignment_method="trigger_locked",
            ))

        atom_id = compute_atom_id(
            dataset_id=dataset_id, subject_id=subject_id,
            session_id=session_id, run_id=run_id, onset_sample=0,
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
                onset_sample=0, onset_seconds=0.0,
                duration_samples=n_samples, duration_seconds=n_samples / srate,
            ),
            channel_ids=channel_ids,
            n_channels=len(channel_ids),
            sampling_rate=srate,
            signal_unit=storage_unit,
            original_unit=orig_unit,
            annotations=annotations,
            custom_fields={"video_condition": trial["video_condition"]},
        )
        return atom, signal.astype(np.float32), (ann_arrays or None)

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
        parsed = _parse_subject_mat(mat_path)

        if subject_id is None:
            m = _SUBJECT_RE.search(mat_path.name)
            subject_id = f"sub-{int(m.group(1)):02d}" if m else mat_path.stem

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

            all_warnings.extend(validate_signal(
                signal=signal, atom_id=atom.atom_id,
                config=self.pool.config.get("import", {}),
                signal_unit=atom.signal_unit,
            ))

            with ShardManager(
                pool_root=self.pool.root, dataset_id=dataset_id,
                subject_id=subject_id, session_id=session_id, run_id=run_id,
                max_shard_size_mb=max_shard_mb, compression=compression,
            ) as shard_mgr:
                atom.signal_ref = shard_mgr.write_atom_signal(
                    atom.atom_id, signal, ann_arrays,
                )

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
                    "attended_ear": trial["attended_ear"],
                    "attended_speaker": trial["attended_speaker"],
                    "video_condition": trial["video_condition"],
                },
            ))
            self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
            stored_atoms.append(atom)

        logger.info(
            "Imported Ear-SAAD subject %s: %d trials", subject_id, len(stored_atoms),
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
        root = Path(root)
        # Accept either preprocessedData/ or the dataset root.
        if not any(root.glob("dataSubject*.mat")) and (root / "preprocessedData").is_dir():
            root = root / "preprocessedData"
        dataset_id = self.task_config.dataset_id

        mat_files = sorted(
            root.glob("dataSubject*.mat"),
            key=lambda p: int(_SUBJECT_RE.search(p.name).group(1)),
        )
        if subjects:
            wanted = {s.lower() for s in subjects}
            mat_files = [
                f for f in mat_files
                if (m := _SUBJECT_RE.search(f.name))
                and f"sub-{int(m.group(1)):02d}" in wanted
            ]
        if max_subjects:
            mat_files = mat_files[:max_subjects]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len(mat_files), original_format="mat",
        ))

        results: List[ImportResult] = []
        for mat_file in mat_files:
            m = _SUBJECT_RE.search(mat_file.name)
            subject_id = f"sub-{int(m.group(1)):02d}"
            self.pool.register_subject(SubjectMeta(
                subject_id=subject_id, dataset_id=dataset_id,
                custom_fields={"has_eareeg": True},
            ))
            try:
                results.append(self.import_subject(
                    mat_file, subject_id=subject_id, max_trials=max_trials,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", mat_file.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("Ear-SAAD import complete: %d subjects", len(results))
        return results


# Auto-register
register_importer("ear_saad", EarSAADImporter)
