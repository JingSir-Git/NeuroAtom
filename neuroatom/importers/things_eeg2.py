"""THINGS-EEG2 Importer: rapid visual object-recognition EEG (Gifford et al. 2022).

10 subjects viewed THINGS object images in an RSVP stream. The preprocessed
release stores, per subject, a training and a test array::

    preprocessed_eeg_data : (n_image_conditions, n_repetitions, n_channels, n_times)
    ch_names              : channel labels
    times                 : epoch time vector (s, relative to image onset)

Two channel variants are shipped: a 17-channel occipito-parietal subset
(``sub-XX/``) and the full 63-channel montage (``sub-XX__63_channels/``).
Each epoch (one image presentation) becomes one EVENT_EPOCH atom labelled with
its image-condition index, repetition, and split (training/test).

IMPORTANT: the published data is multivariate-noise-normalised (whitened), not
in µV. It is imported as-is with ``signal_unit="au"`` rather than mislabelled.

The full set is ~82k epochs/subject, so import is scoped with
``max_conditions`` / ``max_reps`` / ``splits``; defaults import everything.

Data layout::

    things_eeg2/preprocessed/
        sub-01/                 preprocessed_eeg_training.npy  preprocessed_eeg_test.npy
        sub-01__63_channels/    (full montage variant)
        ...
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

_SUBJECT_RE = re.compile(r"(sub-\d+)", re.IGNORECASE)
_UNIT = "au"  # whitened / normalised, not µV
_SPLITS = ("training", "test")


def _subject_dirs(preproc_root: Path, use_63: bool) -> List[Path]:
    suffix = "__63_channels" if use_63 else ""
    out = []
    for d in sorted(preproc_root.glob("sub-*")):
        if not d.is_dir():
            continue
        is_63 = d.name.endswith("__63_channels")
        if use_63 == is_63:
            out.append(d)
    return out


class ThingsEEG2Importer(BaseImporter):
    """Importer for the THINGS-EEG2 visual object-recognition dataset."""

    @staticmethod
    def _preproc_root(root: Path) -> Path:
        cand = root / "preprocessed"
        return cand if cand.is_dir() else root

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.name.startswith("preprocessed_eeg_") and path.suffix == ".npy"
        if path.is_dir():
            pr = ThingsEEG2Importer._preproc_root(path)
            return any(pr.glob("sub-*/preprocessed_eeg_*.npy"))
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("THINGS-EEG2 uses import_subject().")

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------

    def _build_channel_infos(self, names: List[str], sfreq: float) -> List[ChannelInfo]:
        ch_infos: List[ChannelInfo] = []
        for name in names:
            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{len(ch_infos):03d}",
                index=len(ch_infos),
                name=str(name),
                standard_name=standardize_channel_name(str(name)),
                type=ChannelType.EEG,
                unit=_UNIT,
                sampling_rate=sfreq,
                status=ChannelStatus.UNKNOWN,
            ))
        return ch_infos

    def import_subject(
        self,
        subject_dir: Path,
        subject_id: Optional[str] = None,
        splits: Optional[List[str]] = None,
        max_conditions: Optional[int] = None,
        max_reps: Optional[int] = None,
    ) -> ImportResult:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        subject_dir = Path(subject_dir)
        if subject_id is None:
            m = _SUBJECT_RE.search(subject_dir.name)
            subject_id = m.group(1) if m else subject_dir.name
        session_id = "ses-01"
        splits = splits or list(_SPLITS)

        dataset_id = self.task_config.dataset_id
        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        stored_atoms: List[Atom] = []
        all_warnings: List[str] = []
        channel_infos: List[ChannelInfo] = []

        for split in splits:
            npy = subject_dir / f"preprocessed_eeg_{split}.npy"
            if not npy.exists():
                continue
            data = np.load(npy, allow_pickle=True).item()
            eeg = data["preprocessed_eeg_data"]  # (cond, rep, ch, time)
            ch_names = [str(c) for c in data.get("ch_names", [])]
            times = np.asarray(data.get("times", []), dtype=float)
            n_cond, n_rep, n_ch, n_time = eeg.shape
            sfreq = float(round(1.0 / (times[1] - times[0]))) if times.size > 1 else 100.0
            onset_s = float(times[0]) if times.size else 0.0

            if not channel_infos:
                channel_infos = self._build_channel_infos(ch_names or [f"E{i}" for i in range(n_ch)], sfreq)
            channel_ids = [ci.channel_id for ci in channel_infos]

            n_cond_use = min(n_cond, max_conditions) if max_conditions else n_cond
            n_rep_use = min(n_rep, max_reps) if max_reps else n_rep

            run_id = split
            self.pool.ensure_session(dataset_id, subject_id, session_id, sfreq)
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, dataset_id, subject_id, session_id, run_id,
            )
            with ShardManager(
                pool_root=self.pool.root, dataset_id=dataset_id,
                subject_id=subject_id, session_id=session_id, run_id=run_id,
                max_shard_size_mb=max_shard_mb, compression=compression,
            ) as shard_mgr, AtomJSONLWriter(jsonl_path) as writer:
                for c in range(n_cond_use):
                    for r in range(n_rep_use):
                        idx = c * n_rep + r
                        signal = np.ascontiguousarray(eeg[c, r], dtype=np.float32)
                        atom_id = compute_atom_id(
                            dataset_id=dataset_id, subject_id=subject_id,
                            session_id=session_id, run_id=run_id, onset_sample=idx,
                        )
                        atom = Atom(
                            atom_id=atom_id, atom_type=AtomType.EVENT_EPOCH,
                            dataset_id=dataset_id, subject_id=subject_id,
                            session_id=session_id, run_id=run_id, trial_index=idx,
                            signal_ref=SignalRef(
                                file_path="", internal_path="",
                                shape=(n_ch, n_time),
                            ),
                            temporal=TemporalInfo(
                                onset_sample=0, onset_seconds=0.0,
                                duration_samples=n_time, duration_seconds=n_time / sfreq,
                            ),
                            channel_ids=channel_ids, n_channels=n_ch,
                            sampling_rate=sfreq, signal_unit=_UNIT, original_unit=_UNIT,
                            annotations=[
                                NumericAnnotation(
                                    annotation_id=f"ann_cond_{split}_{idx}",
                                    name="image_condition", numeric_value=float(c),
                                ),
                                NumericAnnotation(
                                    annotation_id=f"ann_rep_{split}_{idx}",
                                    name="repetition", numeric_value=float(r),
                                ),
                                CategoricalAnnotation(
                                    annotation_id=f"ann_split_{split}_{idx}",
                                    name="split", value=split,
                                ),
                            ],
                            custom_fields={
                                "split": split, "image_condition": c,
                                "epoch_tmin_s": onset_s,  # epoch start relative to image onset
                            },
                        )
                        if idx < 1:  # validate a sample, not every epoch (perf)
                            all_warnings.extend(validate_signal(
                                signal=signal, atom_id=atom.atom_id,
                                config=self.pool.config.get("import", {}),
                                signal_unit=_UNIT,
                            ))
                        atom.signal_ref = shard_mgr.write_atom_signal(atom.atom_id, signal, None)
                        writer.write_atom(atom)
                        stored_atoms.append(atom)

            self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
            self.pool.register_run(RunMeta(
                run_id=run_id, session_id=session_id, subject_id=subject_id,
                dataset_id=dataset_id, task_type=self.task_config.task_type,
                n_trials=n_cond_use * n_rep_use,
                paradigm_details={"split": split, "n_conditions": n_cond_use, "n_reps": n_rep_use},
            ))
            logger.info(
                "Imported THINGS-EEG2 %s/%s: %d epochs (%d cond x %d rep)",
                subject_id, split, n_cond_use * n_rep_use, n_cond_use, n_rep_use,
            )

        run_meta = RunMeta(
            run_id=stored_atoms[0].run_id if stored_atoms else "training",
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
        use_63_channels: bool = False,
        splits: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
        max_conditions: Optional[int] = None,
        max_reps: Optional[int] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        preproc = self._preproc_root(root)
        dataset_id = self.task_config.dataset_id

        sub_dirs = _subject_dirs(preproc, use_63_channels)
        if subjects:
            want = set(subjects)
            sub_dirs = [d for d in sub_dirs if _SUBJECT_RE.search(d.name).group(1) in want]
        if max_subjects:
            sub_dirs = sub_dirs[:max_subjects]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len(sub_dirs), original_format="npy",
        ))

        results: List[ImportResult] = []
        for sub_dir in sub_dirs:
            sub_id = _SUBJECT_RE.search(sub_dir.name).group(1)
            self.pool.register_subject(SubjectMeta(subject_id=sub_id, dataset_id=dataset_id))
            try:
                results.append(self.import_subject(
                    sub_dir, subject_id=sub_id, splits=splits,
                    max_conditions=max_conditions, max_reps=max_reps,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", sub_dir.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("THINGS-EEG2 import complete: %d subjects", len(results))
        return results


# Auto-register
register_importer("things_eeg2", ThingsEEG2Importer)
