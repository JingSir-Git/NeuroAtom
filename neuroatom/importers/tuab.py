"""TUAB Importer: TUH Abnormal EEG Corpus (normal vs abnormal classification).

Temple University Hospital Abnormal EEG Corpus. Clinical scalp-EEG recordings
labelled normal / abnormal, pre-split into train / eval. Files use the TUH
``01_tcp_ar`` averaged-reference montage and the ``EEG <name>-REF`` naming
convention; sampling rate and channel count vary across recordings.

Each EDF recording becomes one CONTINUOUS_SEGMENT atom labelled with its
diagnosis (``label`` = normal/abnormal) and ``split`` (train/eval). The corpus is
large (~3000 recordings, ~20 min each), so ``max_files`` / ``max_seconds`` scope
the import; defaults import full recordings.

Data layout::

    tuab/edf/
        train/abnormal/01_tcp_ar/<subj>_s###_t###.edf
        train/normal/01_tcp_ar/...
        eval/abnormal/01_tcp_ar/...
        eval/normal/01_tcp_ar/...
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the TUAB importer")

from neuroatom.core.annotation import CategoricalAnnotation
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
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

_STEM_RE = re.compile(r"(?P<subj>.+?)_(?P<ses>s\d+)_(?P<tok>t\d+)$", re.IGNORECASE)
_SPLITS = ("train", "eval")
_LABELS = ("normal", "abnormal")


def _ch_type(name: str) -> ChannelType:
    u = name.upper()
    if "EKG" in u or "ECG" in u:
        return ChannelType.ECG
    if "ROC" in u or "LOC" in u or "EOG" in u:
        return ChannelType.EOG
    if "A1-" in u or "A2-" in u:
        return ChannelType.REF
    return ChannelType.EEG


class TUABImporter(BaseImporter):
    """Importer for the TUH Abnormal EEG Corpus."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            parts = {p.lower() for p in path.parts}
            return path.suffix.lower() == ".edf" and bool(parts & {"normal", "abnormal"})
        if path.is_dir():
            edf = path / "edf"
            if edf.is_dir() and any((edf / s).is_dir() for s in _SPLITS):
                return True
            return any(path.glob("*/normal/*/*.edf")) or any(path.glob("*/abnormal/*/*.edf"))
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        return raw, {"declared_unit": "V"}

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------

    def _build_channel_infos(self, names: List[str], sfreq: float) -> List[ChannelInfo]:
        ch_infos: List[ChannelInfo] = []
        for name in names:
            if name in self.task_config.exclude_channels:
                continue
            ch_type = _ch_type(name)
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

    def import_recording(
        self,
        edf_path: Path,
        label: str,
        split: str,
        max_seconds: Optional[float] = None,
    ) -> Optional[ImportResult]:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        edf_path = Path(edf_path)
        m = _STEM_RE.match(edf_path.stem)
        subject_id = m.group("subj") if m else edf_path.stem
        session_id = m.group("ses") if m else "ses-01"
        run_id = m.group("tok") if m else "t000"

        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        n_keep = int(min(max_seconds, raw.n_times / sfreq) * sfreq) if max_seconds else raw.n_times
        # Some TUH EDFs mix sampling rates (e.g. EKG); MNE warns about edge
        # artifacts on a ranged preload=False read. The effect is <=1 sample on a
        # non-EEG channel at the read boundary — negligible for normal/abnormal
        # EEG classification — and the ranged read keeps this ~3000-file corpus
        # memory-light, so the warning is suppressed rather than force-preloading.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*mixed sampling frequencies.*",
                category=RuntimeWarning,
            )
            signal = raw.get_data(stop=n_keep).astype(np.float32)  # (n_ch, n) Volts
        signal, storage_unit, orig_unit = convert_to_storage_unit(
            signal, source_unit="V", pool_config=self.pool.config,
        )
        n_samples = signal.shape[1]

        channel_infos = self._build_channel_infos(raw.ch_names, sfreq)
        channel_ids = [ci.channel_id for ci in channel_infos]

        dataset_id = self.task_config.dataset_id
        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id, sfreq)
        self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

        atom_id = compute_atom_id(
            dataset_id=dataset_id, subject_id=subject_id,
            session_id=session_id, run_id=run_id, onset_sample=0,
        )
        atom = Atom(
            atom_id=atom_id, atom_type=AtomType.CONTINUOUS_SEGMENT,
            dataset_id=dataset_id, subject_id=subject_id,
            session_id=session_id, run_id=run_id,
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
            annotations=[
                CategoricalAnnotation(
                    annotation_id=f"ann_label_{run_id}",
                    name="label", value=label,
                ),
                CategoricalAnnotation(
                    annotation_id=f"ann_split_{run_id}",
                    name="split", value=split,
                ),
            ],
            custom_fields={"label": label, "split": split, "recording": edf_path.name},
        )
        warnings_list = validate_signal(
            signal=signal, atom_id=atom.atom_id,
            config=self.pool.config.get("import", {}), signal_unit=storage_unit,
        )

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")
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

        self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
        run_meta = RunMeta(
            run_id=run_id, session_id=session_id, subject_id=subject_id,
            dataset_id=dataset_id, task_type=self.task_config.task_type, n_trials=1,
            paradigm_details={"label": label, "split": split},
        )
        self.pool.register_run(run_meta)
        return ImportResult(
            atoms=[atom], run_meta=run_meta,
            channel_infos=channel_infos, warnings=warnings_list,
        )

    def import_dataset(
        self,
        root: Path,
        splits: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        edf_root = root / "edf" if (root / "edf").is_dir() else root
        dataset_id = self.task_config.dataset_id
        splits = splits or list(_SPLITS)
        labels = labels or list(_LABELS)

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=0, original_format="edf",
        ))

        results: List[ImportResult] = []
        subjects_seen: set = set()
        for split in splits:
            for label in labels:
                label_dir = edf_root / split / label
                if not label_dir.is_dir():
                    continue
                edfs = sorted(label_dir.rglob("*.edf"))
                if max_files is not None:
                    edfs = edfs[:max_files]
                for edf in edfs:
                    m = _STEM_RE.match(edf.stem)
                    sub = m.group("subj") if m else edf.stem
                    if sub not in subjects_seen:
                        self.pool.register_subject(SubjectMeta(subject_id=sub, dataset_id=dataset_id))
                        subjects_seen.add(sub)
                    try:
                        r = self.import_recording(edf, label, split, max_seconds=max_seconds)
                        if r:
                            results.append(r)
                    except Exception as e:
                        logger.error("Failed to import %s: %s", edf.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info(
            "TUAB import complete: %d recordings from %d subjects",
            len(results), len(subjects_seen),
        )
        return results


# Auto-register
register_importer("tuab", TUABImporter)
