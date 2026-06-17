"""ASA Importer: auditory spatial attention EEG (SCUT analysis set, per-trial FIF).

20 subjects (S001–S020), one experiment block (E1) of 20 trials each, stored as
per-trial MNE FIF: ``S001_E1_Trial1_raw.fif`` … 64-channel 10-20 EEG @ 500 Hz.
Each file carries two annotations marking the attended-listening window:
``trail{N}S`` (start) and ``trail{N}E`` (end).

Two competing talkers are placed at one of five spatial separations (±90°, ±60°,
±45°, ±30°, ±5°); the listener attends one side. Labels come from the dataset's
own ``main.py``: a fixed per-trial binary attended-direction list (uniform across
all 20 subjects, no reorder) and a separation angle implied by the trial group
(trials 1-4 → 90°, 5-8 → 60°, 9-12 → 45°, 13-16 → 30°, 17-20 → 5°). The unrelated
``db/SCUT.py`` (a 32-trial table borrowed from another dataset) does NOT apply
here. Each trial atom carries ``attended_direction`` (binary) +
``spatial_separation_deg``.

Data layout::

    ASA/
        db/SCUT.py            # 32-trial label table (does NOT match the 20 FIF trials)
        S001/E1/S001_E1_Trial1_raw.fif … _Trial20_raw.fif
        S002/E1/ ...
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the ASA importer")

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

_TRIAL_RE = re.compile(r"_Trial(\d+)_raw$", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"^S\d+$", re.IGNORECASE)

# Per-trial labels from the dataset's own main.py (uniform across all subjects).
# Binary attended direction for trials 1..20:
_DIRECTION_LABELS = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
# Speaker spatial separation (degrees) by trial group:
_SEPARATION_DEG = [90] * 4 + [60] * 4 + [45] * 4 + [30] * 4 + [5] * 4


def _trial_label(trial_num: int) -> Tuple[Optional[str], Optional[int]]:
    """(attended_direction, separation_deg) for a 1-indexed trial, per main.py."""
    i = trial_num - 1
    if 0 <= i < len(_DIRECTION_LABELS):
        return str(_DIRECTION_LABELS[i]), _SEPARATION_DEG[i]
    return None, None


def _segment_bounds(raw: Any, sfreq: float) -> Tuple[int, int]:
    """Attended window [start, stop] samples from trail*S / trail*E annotations.

    FIF annotation onsets are relative to the recording origin, so subtract
    ``raw.first_time`` to get sample indices into this raw, then clamp into
    ``[0, n_times]`` and fall back to the whole file if the markers don't yield a
    valid sub-window.
    """
    n = raw.n_times
    first = raw.first_time
    start, stop = 0, n
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        s = int(round((onset - first) * sfreq))
        d = str(desc).lower()
        if d.endswith("s"):
            start = s
        elif d.endswith("e"):
            stop = s
    start = max(0, min(start, n))
    stop = max(0, min(stop, n))
    if stop <= start:
        start, stop = 0, n
    return start, stop


class ASAImporter(BaseImporter):
    """Importer for the ASA per-trial FIF auditory-attention recordings."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.suffix.lower() == ".fif" and bool(_TRIAL_RE.search(path.stem))
        if path.is_dir():
            return any(path.glob("S*/E*/S*_Trial*_raw.fif")) or \
                any(path.glob("*_Trial*_raw.fif")) or \
                any(path.glob("E*/S*_Trial*_raw.fif"))
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
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
            ch_type = ChannelType.EEG
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
        subject_dir: Path,
        subject_id: Optional[str] = None,
        max_trials: Optional[int] = None,
    ) -> ImportResult:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        subject_dir = Path(subject_dir)
        subject_id = subject_id or subject_dir.name
        # Trials live under a block subdir (E1); fall back to the dir itself.
        fif_files = sorted(
            subject_dir.glob("E*/S*_Trial*_raw.fif"),
            key=lambda p: int(_TRIAL_RE.search(p.stem).group(1)),
        ) or sorted(
            subject_dir.glob("*_Trial*_raw.fif"),
            key=lambda p: int(_TRIAL_RE.search(p.stem).group(1)),
        )
        if max_trials is not None:
            fif_files = fif_files[:max_trials]

        dataset_id = self.task_config.dataset_id
        session_id = "ses-01"
        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id)

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        stored_atoms: List[Atom] = []
        all_warnings: List[str] = []
        channel_infos: List[ChannelInfo] = []

        for fif_path in fif_files:
            trial_num = int(_TRIAL_RE.search(fif_path.stem).group(1))
            raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            start, stop = _segment_bounds(raw, sfreq)

            signal = raw.get_data(start=start, stop=stop).astype(np.float32)  # Volts
            signal, storage_unit, orig_unit = convert_to_storage_unit(
                signal, source_unit="V", pool_config=self.pool.config,
            )
            n_samples = signal.shape[1]

            channel_infos = self._build_channel_infos(raw.ch_names, sfreq)
            channel_ids = [ci.channel_id for ci in channel_infos]

            run_id = f"trial_{trial_num:02d}"
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            direction, separation = _trial_label(trial_num)

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
                annotations=[
                    CategoricalAnnotation(
                        annotation_id=f"ann_dir_{run_id}",
                        name="attended_direction", value=direction or "unknown",
                    ),
                    NumericAnnotation(
                        annotation_id=f"ann_sep_{run_id}",
                        name="spatial_separation_deg",
                        numeric_value=float(separation) if separation is not None else -1.0,
                    ),
                    NumericAnnotation(
                        annotation_id=f"ann_trial_{run_id}",
                        name="trial_number", numeric_value=float(trial_num),
                    ),
                    CategoricalAnnotation(
                        annotation_id=f"ann_prov_{run_id}",
                        name="label_provenance", value="asa_main_py",
                    ),
                ],
                custom_fields={
                    "trial_number": trial_num,
                    "attended_direction": direction,
                    "spatial_separation_deg": separation,
                    "label_provenance": "asa_main_py",
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
                task_type=self.task_config.task_type, n_trials=1,
                paradigm_details={
                    "trial_number": trial_num,
                    "attended_direction": direction,
                    "spatial_separation_deg": separation,
                },
            ))
            self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
            stored_atoms.append(atom)

        logger.info("Imported ASA %s: %d trials", subject_id, len(stored_atoms))
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

        subject_dirs = sorted(
            (d for d in root.glob("S*") if d.is_dir() and _SUBJECT_RE.match(d.name)),
            key=lambda d: int(re.sub(r"\D", "", d.name) or 0),
        )
        if subjects:
            subject_dirs = [d for d in subject_dirs if d.name in set(subjects)]
        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len(subject_dirs), original_format="fif",
        ))

        results: List[ImportResult] = []
        for sub_dir in subject_dirs:
            self.pool.register_subject(SubjectMeta(subject_id=sub_dir.name, dataset_id=dataset_id))
            try:
                results.append(self.import_subject(
                    sub_dir, subject_id=sub_dir.name, max_trials=max_trials,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", sub_dir.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("ASA import complete: %d subjects", len(results))
        return results


# Auto-register
register_importer("asa", ASAImporter)
