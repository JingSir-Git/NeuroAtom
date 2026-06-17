"""LEMON Importer: MPI-Leipzig Mind-Brain-Body resting-state EEG.

Large resting-state dataset (Babayan et al., 2019). Per subject, two conditions:
eyes-closed (EC) and eyes-open (EO), recorded on a 62-channel BrainAmp cap and
distributed here as ICA-cleaned EEGLAB files at 250 Hz (~8 min each).

Resting state has no trial structure, so each ``sub-XXXXXX_{EC,EO}.set`` becomes
one CONTINUOUS_SEGMENT atom labelled with its resting condition.

Data layout::

    lemon/
        preprocessed/EEG_Preprocessed/
            sub-010002_EC.set  sub-010002_EC.fdt
            sub-010002_EO.set  sub-010002_EO.fdt
            ...
        raw/   (BrainVision raw — not imported here)
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the LEMON importer")

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

_FILE_RE = re.compile(r"(sub-\w+?)_(E[CO0])$", re.IGNORECASE)
# "E0" (zero) is a dataset filename typo for "EO" (eyes-open) — handle it so the
# file isn't silently skipped.
_CONDITION = {"EC": "eyes_closed", "EO": "eyes_open", "E0": "eyes_open"}
_EOG = {"VEOG", "HEOG", "EOG", "VEO", "HEO"}


def _preproc_dir(root: Path) -> Path:
    """Resolve the directory that holds the *_E[CO].set files."""
    cand = root / "preprocessed" / "EEG_Preprocessed"
    if cand.is_dir():
        return cand
    return root


def _ch_type(name: str) -> ChannelType:
    u = name.upper()
    if u in _EOG:
        return ChannelType.EOG
    if u == "ECG":
        return ChannelType.ECG
    return ChannelType.EEG


class LEMONImporter(BaseImporter):
    """Importer for the LEMON resting-state EEG dataset (EEGLAB .set)."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.suffix.lower() == ".set" and bool(_FILE_RE.match(path.stem))
        if path.is_dir():
            return any(_preproc_dir(path).glob("*_E[CO].set")) or \
                any(_preproc_dir(path).glob("*_e[co].set"))
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose="ERROR")
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
        set_path: Path,
        subject_id: Optional[str] = None,
        session_id: str = "ses-01",
        max_seconds: Optional[float] = None,
    ) -> ImportResult:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        set_path = Path(set_path)
        m = _FILE_RE.match(set_path.stem)
        sub_auto = m.group(1) if m else set_path.stem
        cond_code = m.group(2).upper() if m else "EC"
        subject_id = subject_id or sub_auto
        condition = _CONDITION.get(cond_code, "rest")
        run_id = cond_code.lower()

        raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        n_keep = int(min(max_seconds, raw.n_times / sfreq) * sfreq) if max_seconds else raw.n_times
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
                    annotation_id=f"ann_cond_{run_id}",
                    name="condition", value=condition,
                ),
                CategoricalAnnotation(
                    annotation_id=f"ann_task_{run_id}",
                    name="task", value="resting_state",
                ),
            ],
            custom_fields={"condition": condition},
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
            paradigm_details={"condition": condition},
        )
        self.pool.register_run(run_meta)
        logger.info("Imported LEMON %s (%s)", subject_id, condition)
        return ImportResult(
            atoms=[atom], run_meta=run_meta,
            channel_infos=channel_infos, warnings=warnings_list,
        )

    def import_dataset(
        self,
        root: Path,
        subjects: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        ppdir = _preproc_dir(root)
        dataset_id = self.task_config.dataset_id

        set_files = sorted(
            f for f in ppdir.glob("*.set") if _FILE_RE.match(f.stem)
        )
        if conditions:
            want = {c.upper() for c in conditions}
            set_files = [f for f in set_files if _FILE_RE.match(f.stem).group(2).upper() in want]
        if subjects:
            subs = set(subjects)
            set_files = [f for f in set_files if _FILE_RE.match(f.stem).group(1) in subs]
        if max_subjects:
            seen, picked = set(), []
            for f in set_files:
                s = _FILE_RE.match(f.stem).group(1)
                if s not in seen and len(seen) >= max_subjects:
                    continue
                seen.add(s)
                picked.append(f)
            set_files = picked

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len({_FILE_RE.match(f.stem).group(1) for f in set_files}),
            original_format="eeglab",
        ))

        results: List[ImportResult] = []
        registered: set = set()
        for set_file in set_files:
            sub = _FILE_RE.match(set_file.stem).group(1)
            if sub not in registered:
                self.pool.register_subject(SubjectMeta(subject_id=sub, dataset_id=dataset_id))
                registered.add(sub)
            try:
                results.append(self.import_recording(
                    set_file, subject_id=sub, max_seconds=max_seconds,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", set_file.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("LEMON import complete: %d recordings", len(results))
        return results


# Auto-register
register_importer("lemon", LEMONImporter)
