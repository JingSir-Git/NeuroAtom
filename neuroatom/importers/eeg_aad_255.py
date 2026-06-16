"""EEG-AAD-255 Importer: 256-channel high-density auditory attention decoding.

High-density (255 scalp + 2 extra) AAD dataset recorded with a Neuroscan
Curry system at 1000 Hz. 32 subjects, 4 recordings each. The attended side is
encoded directly in the filename — ``S0_AAD_1L`` (block 1, attend Left) /
``S0_AAD_1R`` (attend Right) — so labels are unambiguous.

Data layout (Neuroscan Curry 7: .dat + .dap + .rs3 + .ceo)::

    EEG-AAD-255/
        misc/misc/eeg255ch_locs.csv        # electrode layout (not imported)
        S0/S0/
            S0_AAD_1L.dat  S0_AAD_1R.dat  S0_AAD_2L.dat  S0_AAD_2R.dat
        S1/S1/ ...

Each .dat → one TRIAL atom. 256 channels: 253 scalp EEG + XtraL/XtraR (external,
typed OTHER) + Trigger (excluded). MNE returns SI volts; converted to the pool
storage unit (µV) on import. The recording is continuous single-side attention,
so ``attended_direction`` (left/right) is a static per-trial label.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the EEG-AAD-255 importer")

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
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

_FILE_RE = re.compile(r"_AAD_(\d+)([LR])", re.IGNORECASE)
_EXTERNAL = {"XtraL", "XtraR"}


def _channel_type(name: str, mne_type: str) -> ChannelType:
    if name == "Trigger":
        return ChannelType.STIM
    if name in _EXTERNAL:
        return ChannelType.OTHER
    if mne_type == "eeg":
        return ChannelType.EEG
    return ChannelType.OTHER


def _find_subject_dir(subject_root: Path) -> Path:
    """EEG-AAD-255 nests data one level: S0/S0/. Return the dir with the .dat."""
    if any(subject_root.glob("*_AAD_*.dat")):
        return subject_root
    inner = subject_root / subject_root.name
    if inner.is_dir() and any(inner.glob("*_AAD_*.dat")):
        return inner
    return subject_root


class EEGAAD255Importer(BaseImporter):
    """Importer for the EEG-AAD-255 high-density Curry AAD dataset."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.suffix.lower() == ".dat" and bool(_FILE_RE.search(path.name))
        if path.is_dir():
            for pat in ("*_AAD_*.dat", "*/*_AAD_*.dat", "S*/S*/*_AAD_*.dat"):
                if any(path.glob(pat)):
                    return True
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raw = mne.io.read_raw_curry(str(path), preload=False, verbose="ERROR")
        return raw, {"declared_unit": "V"}

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------

    def _build_channel_infos(
        self, names: List[str], mne_types: List[str], sfreq: float,
    ) -> Tuple[List[ChannelInfo], List[int]]:
        """Return (channel_infos, picks) for the kept (non-excluded) channels."""
        ch_infos: List[ChannelInfo] = []
        picks: List[int] = []
        for raw_idx, (name, mtype) in enumerate(zip(names, mne_types)):
            if name in self.task_config.exclude_channels:
                continue
            ch_type = _channel_type(name, mtype)
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
            picks.append(raw_idx)
        return ch_infos, picks

    def import_subject(
        self,
        subject_root: Path,
        subject_id: Optional[str] = None,
        session_id: str = "ses-01",
        max_trials: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> ImportResult:
        """Import all .dat recordings for one subject.

        Args:
            subject_root: Subject dir (``S0`` or ``S0/S0``) or a single .dat file.
            max_seconds: Crop each recording to this many seconds (keeps tests light).
        """
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        subject_root = Path(subject_root)
        if subject_root.is_file():
            dat_files = [subject_root]
            data_dir = subject_root.parent
        else:
            data_dir = _find_subject_dir(subject_root)
            dat_files = sorted(data_dir.glob("*_AAD_*.dat"))

        if subject_id is None:
            # .../S0/S0/S0_AAD_1L.dat → "S0"
            m = re.match(r"(S\d+)", dat_files[0].name) if dat_files else None
            subject_id = m.group(1) if m else data_dir.parent.name

        dataset_id = self.task_config.dataset_id
        if max_trials is not None:
            dat_files = dat_files[:max_trials]

        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id)

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        stored_atoms: List[Atom] = []
        all_warnings: List[str] = []
        channel_infos: List[ChannelInfo] = []

        for dat_path in dat_files:
            m = _FILE_RE.search(dat_path.name)
            block = m.group(1) if m else "0"
            side = m.group(2).upper() if m else ""
            attended = {"L": "left", "R": "right"}.get(side, "unknown")
            run_id = f"block{block}_{side}".lower()

            raw = mne.io.read_raw_curry(str(dat_path), preload=False, verbose="ERROR")
            if max_seconds is not None:
                raw.crop(tmax=min(max_seconds, raw.n_times / raw.info["sfreq"] - 1.0 / raw.info["sfreq"]))
            sfreq = float(raw.info["sfreq"])
            n_triggers = len(raw.annotations)

            channel_infos, picks = self._build_channel_infos(
                raw.ch_names, raw.get_channel_types(), sfreq,
            )
            channel_ids = [ci.channel_id for ci in channel_infos]

            signal = raw.get_data(picks=picks).astype(np.float32)  # (n_ch, n_samples), V
            signal, storage_unit, orig_unit = convert_to_storage_unit(
                signal, source_unit="V", pool_config=self.pool.config,
            )
            n_samples = signal.shape[1]

            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            annotations = [
                CategoricalAnnotation(
                    annotation_id=f"ann_dir_{run_id}",
                    name="attended_direction", value=attended,
                ),
                CategoricalAnnotation(
                    annotation_id=f"ann_block_{run_id}",
                    name="block", value=block,
                ),
            ]
            if n_triggers:
                annotations.append(NumericAnnotation(
                    annotation_id=f"ann_ntrig_{run_id}",
                    name="n_triggers", numeric_value=float(n_triggers),
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
                trial_index=int(block),
                signal_ref=SignalRef(
                    file_path="", internal_path="",
                    shape=(len(channel_ids), n_samples),
                ),
                temporal=TemporalInfo(
                    onset_sample=0, onset_seconds=0.0,
                    duration_samples=n_samples, duration_seconds=n_samples / sfreq,
                ),
                channel_ids=channel_ids,
                n_channels=len(channel_ids),
                sampling_rate=sfreq,
                signal_unit=storage_unit,
                original_unit=orig_unit,
                annotations=annotations,
                custom_fields={"attended_direction": attended, "block": block},
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
                dataset_id=dataset_id, run_index=int(block),
                task_type=self.task_config.task_type, n_events=n_triggers, n_trials=1,
                paradigm_details={"attended_direction": attended, "block": block},
            ))
            self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
            stored_atoms.append(atom)

        logger.info(
            "Imported EEG-AAD-255 subject %s: %d recordings", subject_id, len(stored_atoms),
        )
        run_meta = RunMeta(
            run_id=stored_atoms[0].run_id if stored_atoms else "block1_l",
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
        max_seconds: Optional[float] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        dataset_id = self.task_config.dataset_id

        subject_dirs = sorted(
            (d for d in root.glob("S*") if d.is_dir()),
            key=lambda d: int(re.match(r"S(\d+)", d.name).group(1))
            if re.match(r"S(\d+)", d.name) else 0,
        )
        if subjects:
            wanted = {s for s in subjects}
            subject_dirs = [d for d in subject_dirs if d.name in wanted]
        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len(subject_dirs), original_format="curry",
        ))

        results: List[ImportResult] = []
        for sub_dir in subject_dirs:
            self.pool.register_subject(SubjectMeta(
                subject_id=sub_dir.name, dataset_id=dataset_id,
            ))
            try:
                results.append(self.import_subject(
                    sub_dir, subject_id=sub_dir.name,
                    max_trials=max_trials, max_seconds=max_seconds,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", sub_dir.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("EEG-AAD-255 import complete: %d subjects", len(results))
        return results


# Auto-register
register_importer("eeg_aad_255", EEGAAD255Importer)
