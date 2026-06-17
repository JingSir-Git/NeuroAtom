"""Sleep-EDF Expanded Importer: polysomnography sleep staging (PhysioNet).

The canonical sleep-staging benchmark. Two studies:
    - sleep-cassette (SC): healthy ambulatory recordings, ~20 h each
    - sleep-telemetry (ST): temazepam study, ~9 h each

Each recording is a pair of EDF files matched by ID prefix::

    SC4001E0-PSG.edf         # polysomnography signals
    SC4001EC-Hypnogram.edf   # expert sleep-stage annotations (R&K)

The PSG holds EEG Fpz-Cz / EEG Pz-Oz / EOG horizontal at 100 Hz plus
respiration / EMG / temperature / event-marker channels at 1 Hz. We keep the
four scoreable channels (2 EEG + EOG + EMG); the 1 Hz PSG extras and the event
marker are dropped.

The hypnogram annotates variable-duration stage bouts (``Sleep stage W/1/2/3/4/R``,
plus ``?`` / ``Movement time``). Following standard practice we emit one 30 s
WINDOW atom per epoch, labelled with its sleep stage, and trim the long
ambulatory wake periods to ``crop_wake_mins`` (default 30) around sleep —
configurable, set to ``None`` to keep the full recording.

Data layout::

    Sleep_EDF_Expanded/
        sleep-cassette/   SC4001E0-PSG.edf  SC4001EC-Hypnogram.edf ...
        sleep-telemetry/  ST7011J0-PSG.edf  ST7011JP-Hypnogram.edf ...
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the Sleep-EDF importer")

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

_EPOCH_SECONDS = 30.0

# Hypnogram description → canonical sleep stage.
_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N4",
    "Sleep stage R": "REM",
    "Sleep stage ?": "unknown",
    "Movement time": "movement",
}
_UNSCORED = {"unknown", "movement"}

# Scoreable channels kept (others dropped: Resp oro-nasal, Temp rectal, Event marker).
_KEEP_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental")


def _ch_type(name: str) -> ChannelType:
    n = name.lower()
    if "eeg" in n:
        return ChannelType.EEG
    if "eog" in n:
        return ChannelType.EOG
    if "emg" in n:
        return ChannelType.EMG
    return ChannelType.MISC


def _pair_recordings(folder: Path) -> List[Tuple[Path, Path]]:
    """Match *-PSG.edf with *-Hypnogram.edf by their shared 7-char ID prefix."""
    psgs = sorted(folder.glob("*-PSG.edf"))
    hyps = {h.name[:7]: h for h in folder.glob("*-Hypnogram.edf")}
    pairs = []
    for psg in psgs:
        hyp = hyps.get(psg.name[:7])
        if hyp is not None:
            pairs.append((psg, hyp))
        else:
            logger.warning("No hypnogram for %s", psg.name)
    return pairs


def _load_stage_bouts(hyp_path: Path) -> List[Tuple[float, float, str]]:
    """Return [(onset_s, end_s, stage), ...] from a hypnogram EDF."""
    ann = mne.read_annotations(str(hyp_path))
    bouts = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        stage = _STAGE_MAP.get(str(desc), "unknown")
        bouts.append((float(onset), float(onset) + float(dur), stage))
    return bouts


def _stage_at(bouts: List[Tuple[float, float, str]], t: float) -> str:
    for onset, end, stage in bouts:
        if onset <= t < end:
            return stage
    return "unknown"


def _parse_recording_id(psg_path: Path) -> Tuple[str, str, str]:
    """(subject_id, session_id, base) from a PSG filename, e.g. SC4001E0-PSG.edf."""
    base = psg_path.name.split("-PSG")[0]          # SC4001E0
    study = base[:2]                                # SC / ST
    subj = base[3:5] if len(base) >= 6 else "00"    # 2-digit subject
    night = base[5] if len(base) >= 6 else "1"      # night digit
    return f"{study}{subj}", f"night{night}", base


class SleepEDFImporter(BaseImporter):
    """Importer for the Sleep-EDF Expanded polysomnography dataset."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.name.endswith("-PSG.edf")
        if path.is_dir():
            for pat in ("*-PSG.edf", "sleep-cassette/*-PSG.edf", "sleep-telemetry/*-PSG.edf"):
                if any(path.glob(pat)):
                    return True
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
        psg_path: Path,
        hyp_path: Path,
        subject_id: Optional[str] = None,
        session_id: Optional[str] = None,
        crop_wake_mins: Optional[float] = 30.0,
        drop_unscored: bool = True,
        max_epochs: Optional[int] = None,
    ) -> ImportResult:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        psg_path, hyp_path = Path(psg_path), Path(hyp_path)
        sub_auto, ses_auto, base = _parse_recording_id(psg_path)
        subject_id = subject_id or sub_auto
        session_id = session_id or ses_auto
        run_id = "run-01"

        raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        keep = [c for c in _KEEP_CHANNELS if c in raw.ch_names]
        bouts = _load_stage_bouts(hyp_path)

        # Determine import window (trim long ambulatory wake around sleep).
        rec_end = raw.n_times / sfreq
        win_start, win_end = 0.0, rec_end
        if crop_wake_mins is not None:
            sleep = [b for b in bouts if b[2] not in _UNSCORED and b[2] != "W"]
            if sleep:
                pad = crop_wake_mins * 60.0
                win_start = max(0.0, sleep[0][0] - pad)
                win_end = min(rec_end, sleep[-1][1] + pad)
        if max_epochs is not None:
            win_end = min(win_end, win_start + max_epochs * _EPOCH_SECONDS)

        epoch_len = int(_EPOCH_SECONDS * sfreq)
        # Crop to the import window then preload before reading. Sleep-EDF mixes
        # sampling rates (EMG submental @ 1 Hz, EEG/EOG @ 100 Hz); a ranged
        # get_data() with preload=False would introduce edge artifacts, so we
        # materialise the cropped window first.
        tmax = max(win_start, min(win_end, rec_end) - 1.0 / sfreq)
        # EMG submental is 1 Hz vs 100 Hz EEG/EOG; MNE warns about mixed-rate
        # upsampling edge effects. After crop+preload the interior is exact and
        # any boundary effect (<=1 s of EMG) falls in the wake-padding, never in
        # a scored sleep epoch — so this warning is benign here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*mixed sampling frequencies.*",
                category=RuntimeWarning,
            )
            raw.crop(tmin=win_start, tmax=tmax).pick(keep).load_data()
        seg = raw.get_data()  # (n_ch, n) in Volts, channels in `keep` order

        channel_infos = self._build_channel_infos(keep, sfreq)
        channel_ids = [ci.channel_id for ci in channel_infos]

        dataset_id = self.task_config.dataset_id
        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id, sfreq)
        self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        n_epochs = seg.shape[1] // epoch_len
        stored_atoms: List[Atom] = []
        all_warnings: List[str] = []

        with ShardManager(
            pool_root=self.pool.root, dataset_id=dataset_id,
            subject_id=subject_id, session_id=session_id, run_id=run_id,
            max_shard_size_mb=max_shard_mb, compression=compression,
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, dataset_id, subject_id, session_id, run_id,
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                for e in range(n_epochs):
                    t_abs = win_start + e * _EPOCH_SECONDS
                    stage = _stage_at(bouts, t_abs)
                    if drop_unscored and stage in _UNSCORED:
                        continue

                    epoch = np.ascontiguousarray(
                        seg[:, e * epoch_len:(e + 1) * epoch_len], dtype=np.float32,
                    )
                    signal, storage_unit, orig_unit = convert_to_storage_unit(
                        epoch, source_unit="V", pool_config=self.pool.config,
                    )
                    onset_sample = int(t_abs * sfreq)
                    atom_id = compute_atom_id(
                        dataset_id=dataset_id, subject_id=subject_id,
                        session_id=session_id, run_id=run_id, onset_sample=onset_sample,
                    )
                    atom = Atom(
                        atom_id=atom_id, atom_type=AtomType.WINDOW,
                        dataset_id=dataset_id, subject_id=subject_id,
                        session_id=session_id, run_id=run_id, trial_index=e,
                        signal_ref=SignalRef(
                            file_path="", internal_path="",
                            shape=(len(channel_ids), epoch_len),
                        ),
                        temporal=TemporalInfo(
                            onset_sample=onset_sample, onset_seconds=t_abs,
                            duration_samples=epoch_len, duration_seconds=_EPOCH_SECONDS,
                        ),
                        channel_ids=channel_ids, n_channels=len(channel_ids),
                        sampling_rate=sfreq, signal_unit=storage_unit, original_unit=orig_unit,
                        annotations=[
                            CategoricalAnnotation(
                                annotation_id=f"ann_stage_{onset_sample}",
                                name="sleep_stage", value=stage,
                            ),
                            NumericAnnotation(
                                annotation_id=f"ann_epoch_{onset_sample}",
                                name="epoch_index", numeric_value=float(e),
                            ),
                        ],
                        custom_fields={"study": base[:2], "sleep_stage": stage},
                    )
                    all_warnings.extend(validate_signal(
                        signal=signal, atom_id=atom.atom_id,
                        config=self.pool.config.get("import", {}), signal_unit=storage_unit,
                    ))
                    atom.signal_ref = shard_mgr.write_atom_signal(atom.atom_id, signal, None)
                    writer.write_atom(atom)
                    stored_atoms.append(atom)

        self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
        self.pool.register_run(RunMeta(
            run_id=run_id, session_id=session_id, subject_id=subject_id,
            dataset_id=dataset_id, task_type=self.task_config.task_type,
            n_trials=len(stored_atoms),
            paradigm_details={"recording": base, "n_epochs": len(stored_atoms)},
        ))

        logger.info(
            "Imported Sleep-EDF %s (%s/%s): %d epochs",
            base, subject_id, session_id, len(stored_atoms),
        )
        run_meta = RunMeta(
            run_id=run_id, session_id=session_id, subject_id=subject_id,
            dataset_id=dataset_id, task_type=self.task_config.task_type,
            n_trials=len(stored_atoms),
        )
        return ImportResult(
            atoms=stored_atoms, run_meta=run_meta,
            channel_infos=channel_infos, warnings=all_warnings,
        )

    def import_dataset(
        self,
        root: Path,
        studies: Optional[List[str]] = None,
        max_recordings: Optional[int] = None,
        crop_wake_mins: Optional[float] = 30.0,
        max_epochs: Optional[int] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        dataset_id = self.task_config.dataset_id

        folders = []
        for sub in ("sleep-cassette", "sleep-telemetry"):
            if (root / sub).is_dir():
                folders.append(root / sub)
        if not folders and any(root.glob("*-PSG.edf")):
            folders = [root]
        if studies:
            folders = [f for f in folders if f.name in studies]

        pairs: List[Tuple[Path, Path]] = []
        for folder in folders:
            pairs.extend(_pair_recordings(folder))
        if max_recordings:
            pairs = pairs[:max_recordings]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len({_parse_recording_id(p)[0] for p, _ in pairs}),
            original_format="edf",
        ))

        results: List[ImportResult] = []
        for psg, hyp in pairs:
            sub_id, _, _ = _parse_recording_id(psg)
            self.pool.register_subject(SubjectMeta(subject_id=sub_id, dataset_id=dataset_id))
            try:
                results.append(self.import_recording(
                    psg, hyp, crop_wake_mins=crop_wake_mins, max_epochs=max_epochs,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", psg.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("Sleep-EDF import complete: %d recordings", len(results))
        return results


# Auto-register
register_importer("sleep_edf", SleepEDFImporter)
