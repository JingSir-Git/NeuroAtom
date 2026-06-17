"""Tsinghua SSVEP Importer: Benchmark SSVEP-BCI dataset (Wang et al., 2017).

35 subjects fixating on 40 characters that flicker at distinct frequencies
(8–15.8 Hz, 0.2 Hz steps) — the canonical SSVEP speller benchmark. Recorded on
a Neuroscan Synamps2 at 1000 Hz, downsampled to 250 Hz, 64-channel 10-20 EEG.

Each subject is one ``S{n}.mat`` holding a 4-D ``data`` matrix of shape
``[64, 1500, 40, 6]`` = (electrode, time, target, block): 40 targets × 6 blocks
= 240 stimulus epochs, each 6 s (0.5 s pre-stimulus + 5.5 s post-onset = 1500
samples @ 250 Hz). Each ``data[:, :, t, b]`` becomes one TRIAL atom labelled with
the target's stimulation frequency + phase (from ``Freq_Phase.mat``).

Data layout::

    tsinghua_ssvep/
        S1.mat ... S35.mat        # 4-D `data` per subject
        Freq_Phase.mat            # freqs[40], phases[40] per target
        64-channels.loc           # electrode labels (col 4)
        Sub_info.txt              # demographics + experienced/naive group
        Readme.txt
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

_SUBJECT_RE = re.compile(r"^S(\d+)$", re.IGNORECASE)
_PRESTIM_SECONDS = 0.5  # cue/pre-stimulus before flicker onset
_MASTOIDS = {"M1", "M2"}

# Fallback 64-channel order (matches 64-channels.loc), used if the .loc is absent.
_CH_NAMES_FALLBACK = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5",
    "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3",
    "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6",
    "PO8", "CB1", "O1", "Oz", "O2", "CB2",
]


def _parse_loc(loc_path: Path) -> List[str]:
    """Channel labels from a 64-channels.loc file (4th whitespace column)."""
    names: List[str] = []
    try:
        for line in loc_path.read_text(encoding="latin-1").splitlines():
            parts = line.split()
            if len(parts) >= 4:
                names.append(parts[3])
    except OSError:
        return []
    return names


def _load_freq_phase(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load per-target frequencies + phases from Freq_Phase.mat (defensive keys)."""
    if not path.exists():
        return None, None
    fp = sio.loadmat(str(path))
    freqs = phases = None
    for k, v in fp.items():
        if k.startswith("__"):
            continue
        arr = np.ravel(np.asarray(v, dtype=float))
        if "freq" in k.lower():
            freqs = arr
        elif "phase" in k.lower():
            phases = arr
    return freqs, phases


def _parse_sub_info(path: Path) -> Dict[str, Dict[str, str]]:
    """Parse Sub_info.txt → {subject_id: {gender, age, handedness, group}}."""
    out: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = line.split()
        if len(parts) >= 5 and re.match(r"^S\d+$", parts[0], re.IGNORECASE):
            out[parts[0].upper()] = {
                "gender": parts[1], "age": parts[2],
                "handedness": parts[3], "group": parts[4],
            }
    return out


class TsinghuaSSVEPImporter(BaseImporter):
    """Importer for the Tsinghua Benchmark SSVEP dataset."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return (
                path.suffix.lower() == ".mat"
                and bool(_SUBJECT_RE.match(path.stem))
                and (path.parent / "Freq_Phase.mat").exists()
            )
        if path.is_dir():
            return (path / "Freq_Phase.mat").exists() and any(
                _SUBJECT_RE.match(p.stem) for p in path.glob("S*.mat")
            )
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("Tsinghua SSVEP uses import_subject().")

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------

    def _channel_infos(self, names: List[str], sfreq: float) -> List[ChannelInfo]:
        ch_infos: List[ChannelInfo] = []
        for idx, name in enumerate(names):
            if name in self.task_config.exclude_channels:
                continue
            ch_type = ChannelType.REF if name in _MASTOIDS else ChannelType.EEG
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

    def import_subject(
        self,
        mat_path: Path,
        subject_id: Optional[str] = None,
        session_id: str = "ses-01",
        ch_names: Optional[List[str]] = None,
        freqs: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
        sfreq: float = 250.0,
        max_targets: Optional[int] = None,
        max_blocks: Optional[int] = None,
        max_trials: Optional[int] = None,
    ) -> ImportResult:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        mat_path = Path(mat_path)
        require_mat_v5(mat_path, "TsinghuaSSVEPImporter")
        if subject_id is None:
            m = _SUBJECT_RE.match(mat_path.stem)
            subject_id = f"S{int(m.group(1)):02d}" if m else mat_path.stem

        if ch_names is None:
            ch_names = _parse_loc(mat_path.parent / "64-channels.loc") or _CH_NAMES_FALLBACK
        if freqs is None:
            freqs, phases = _load_freq_phase(mat_path.parent / "Freq_Phase.mat")

        logger.info("Loading Tsinghua SSVEP .mat: %s", mat_path)
        data = sio.loadmat(str(mat_path))["data"]  # [64, 1500, 40, 6]
        n_ch, n_samples, n_targets, n_blocks = data.shape
        if len(ch_names) != n_ch:
            ch_names = (ch_names + [f"Ch_{i}" for i in range(n_ch)])[:n_ch]

        n_targets_use = min(n_targets, max_targets) if max_targets else n_targets
        n_blocks_use = min(n_blocks, max_blocks) if max_blocks else n_blocks

        channel_infos = self._channel_infos(ch_names, sfreq)
        channel_ids = [ci.channel_id for ci in channel_infos]
        prestim = int(_PRESTIM_SECONDS * sfreq)

        dataset_id = self.task_config.dataset_id
        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id, sfreq)

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        stored_atoms: List[Atom] = []
        all_warnings: List[str] = []
        count = 0

        for t in range(n_targets_use):
            if max_trials is not None and count >= max_trials:
                break
            freq = float(freqs[t]) if freqs is not None and t < len(freqs) else None
            phase = float(phases[t]) if phases is not None and t < len(phases) else None
            for b in range(n_blocks_use):
                if max_trials is not None and count >= max_trials:
                    break
                seg = np.ascontiguousarray(
                    np.asarray(data[:, :, t, b], dtype=np.float32)
                )  # (n_ch, n_samples)
                signal, storage_unit, orig_unit = convert_to_storage_unit(
                    seg, source_unit="uV", pool_config=self.pool.config,
                )

                run_id = f"t{t:02d}_b{b}"
                self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

                annotations: List[Any] = [
                    CategoricalAnnotation(
                        annotation_id=f"ann_target_{run_id}",
                        name="target_index", value=str(t),
                    ),
                    NumericAnnotation(
                        annotation_id=f"ann_block_{run_id}",
                        name="block", numeric_value=float(b),
                    ),
                ]
                if freq is not None:
                    annotations.append(NumericAnnotation(
                        annotation_id=f"ann_freq_{run_id}",
                        name="ssvep_frequency", numeric_value=freq, unit="Hz",
                    ))
                if phase is not None:
                    annotations.append(NumericAnnotation(
                        annotation_id=f"ann_phase_{run_id}",
                        name="ssvep_phase", numeric_value=phase, unit="rad",
                    ))

                atom_id = compute_atom_id(
                    dataset_id=dataset_id, subject_id=subject_id,
                    session_id=session_id, run_id=run_id, onset_sample=0,
                )
                atom = Atom(
                    atom_id=atom_id, atom_type=AtomType.TRIAL,
                    dataset_id=dataset_id, subject_id=subject_id,
                    session_id=session_id, run_id=run_id, trial_index=t * n_blocks + b,
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
                        "target_index": t, "block": b,
                        "ssvep_frequency": freq,
                        "stimulus_onset_seconds": _PRESTIM_SECONDS,
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
                    dataset_id=dataset_id, run_index=t * n_blocks + b,
                    task_type=self.task_config.task_type, n_events=0, n_trials=1,
                    paradigm_details={"target_index": t, "block": b, "ssvep_frequency": freq},
                ))
                self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
                stored_atoms.append(atom)
                count += 1

        logger.info(
            "Imported Tsinghua SSVEP subject %s: %d trials", subject_id, len(stored_atoms),
        )
        run_meta = RunMeta(
            run_id=stored_atoms[0].run_id if stored_atoms else "t00_b0",
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
        max_targets: Optional[int] = None,
        max_blocks: Optional[int] = None,
        max_trials: Optional[int] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        dataset_id = self.task_config.dataset_id

        ch_names = _parse_loc(root / "64-channels.loc") or _CH_NAMES_FALLBACK
        freqs, phases = _load_freq_phase(root / "Freq_Phase.mat")
        demographics = _parse_sub_info(root / "Sub_info.txt")

        mat_files = sorted(
            (p for p in root.glob("S*.mat") if _SUBJECT_RE.match(p.stem)),
            key=lambda p: int(_SUBJECT_RE.match(p.stem).group(1)),
        )
        if subjects:
            wanted = {s.upper() for s in subjects}
            mat_files = [
                f for f in mat_files
                if f.stem.upper() in wanted
                or f"S{int(_SUBJECT_RE.match(f.stem).group(1)):02d}" in wanted
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
            n = int(_SUBJECT_RE.match(mat_file.stem).group(1))
            subject_id = f"S{n:02d}"
            info = demographics.get(subject_id, {})
            age = info.get("age", "")
            self.pool.register_subject(SubjectMeta(
                subject_id=subject_id, dataset_id=dataset_id,
                age=int(age) if age.isdigit() else None,
                sex=info.get("gender", "")[:1].upper() if info.get("gender") else None,
                custom_fields={k: v for k, v in info.items() if k in ("handedness", "group")},
            ))
            try:
                results.append(self.import_subject(
                    mat_file, subject_id=subject_id, ch_names=ch_names,
                    freqs=freqs, phases=phases,
                    max_targets=max_targets, max_blocks=max_blocks, max_trials=max_trials,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", mat_file.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("Tsinghua SSVEP import complete: %d subjects", len(results))
        return results


# Auto-register
register_importer("tsinghua_ssvep", TsinghuaSSVEPImporter)
