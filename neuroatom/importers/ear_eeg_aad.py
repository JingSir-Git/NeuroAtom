"""ear-EEG-AAD Importer: ear + scalp EEG auditory attention (TMSi SAGA Poly5).

16 subjects, each one continuous TMSi SAGA recording (~66 min @ 500 Hz) stored
as a ``.Poly5`` (TMS32) file. MNE cannot read these, so this module ships a
small, self-contained Poly5 reader validated against the file layout.

Poly5 / TMS32 layout (version 2.03), as implemented by FieldTrip's
``read_tmsi_poly5`` and cross-checked here byte-for-byte:
    - 217-byte header: sample_rate (int16 @114), NumberOfSignals NS (int16 @119),
      NumberOfSamplePeriods (int32 @121).
    - NS x 136-byte signal descriptions. SAGA stores each physical channel as a
      ``(Lo)``/``(Hi)`` *pair*, so the true channel count is ``NS // 2`` and the
      channel name comes from every other ("(Lo) …") description.
    - Data: ``num_blocks`` blocks, each an 86-byte block header followed by
      ``samples_per_block x (NS//2)`` float32 samples (already in µV), stored
      sample-interleaved. Block/sample counts are derived from the file size and
      NumberOfSamplePeriods and asserted to match the file exactly.

Channels named ``UNI …`` / ``BIP …`` are the EEG; ``TRIGGERS`` / ``STATUS`` /
``Counter`` / ``PLETH`` / ``HRate`` / ``Stat`` are device/auxiliary and dropped.

LABELS: there is no attention-label source in the dataset (only ``Record.xses``
session XML per subject), so each subject is imported as one CONTINUOUS_SEGMENT
atom with ``label_provenance="unresolved"`` — the ear-EEG signal is preserved
faithfully and attention labels can be attached later if a source surfaces.

Data layout::

    ear_eeg_aad/
        sub1/sub1.Poly5  sub1/Record.xses
        sub2/ ... sub16/
"""

import logging
import os
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

_POLY5_MAGIC = b"POLY SAMPLE FIL"
_NON_EEG = {"TRIGGERS", "STATUS", "Counter 2power24", "PLETH", "HRate", "Stat"}
_LO_PREFIX = re.compile(r"^\(Lo\)\s*", re.IGNORECASE)


def _read_poly5(path: Path, max_seconds: Optional[float] = None) -> Tuple[float, List[str], np.ndarray]:
    """Read a TMSi SAGA Poly5 file → (sample_rate, channel_names, data[n_ch, n]).

    Data are float32 already in µV. Channel count is NumberOfSignals//2 (the
    SAGA (Lo)/(Hi) pairing). The block layout is derived from the file size and
    asserted to match exactly, so a malformed/unexpected file raises rather than
    silently returning corrupt signal.
    """
    path = Path(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(217)
        if head[:15] != _POLY5_MAGIC:
            raise ValueError(f"{path.name}: not a Poly5 file")
        sample_rate = struct.unpack("<h", head[114:116])[0]
        num_signals = struct.unpack("<H", head[119:121])[0]
        total_periods = struct.unpack("<i", head[121:125])[0]
        n_ch = num_signals // 2

        data_region = size - 217 - num_signals * 136
        num_blocks = (data_region - total_periods * n_ch * 4) // 86
        if num_blocks <= 0:
            raise ValueError(f"{path.name}: cannot derive Poly5 block layout")
        samples_per_block = total_periods // num_blocks
        sd = samples_per_block * n_ch * 4  # data bytes per block
        if 217 + num_signals * 136 + num_blocks * (86 + sd) != size:
            raise ValueError(f"{path.name}: Poly5 layout does not match file size")

        # Channel names: every other ("(Lo) …") description, prefix stripped.
        names: List[str] = []
        for i in range(0, num_signals, 2):
            f.seek(217 + i * 136)
            d = f.read(136)
            ln = d[0]
            nm = d[1:1 + ln].decode("latin-1", "replace").strip()
            names.append(_LO_PREFIX.sub("", nm))

        n_blocks_read = num_blocks
        if max_seconds is not None:
            n_blocks_read = min(
                num_blocks, max(1, int(np.ceil(max_seconds * sample_rate / samples_per_block)))
            )

        out = np.empty((n_blocks_read * samples_per_block, n_ch), dtype=np.float32)
        base = 217 + num_signals * 136
        for g in range(n_blocks_read):
            f.seek(base + g * (86 + sd) + 86)
            block = np.frombuffer(f.read(sd), dtype="<f4").reshape(samples_per_block, n_ch)
            out[g * samples_per_block:(g + 1) * samples_per_block] = block

    return float(sample_rate), names, out.T  # (n_ch, n_samples)


def _is_eeg(name: str) -> bool:
    u = name.upper()
    return (u.startswith("UNI") or u.startswith("BIP")) and name not in _NON_EEG


class EarEEGAADImporter(BaseImporter):
    """Importer for the ear-EEG-AAD TMSi SAGA Poly5 recordings."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.suffix.lower() == ".poly5"
        if path.is_dir():
            return any(path.glob("*.Poly5")) or any(path.glob("sub*/*.Poly5"))
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("ear-EEG-AAD uses import_subject().")

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        return []

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------

    def _build_channel_infos(
        self, names: List[str], keep_idx: List[int], sfreq: float,
    ) -> List[ChannelInfo]:
        ch_infos: List[ChannelInfo] = []
        for new_i, raw_i in enumerate(keep_idx):
            name = names[raw_i]
            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{new_i:03d}",
                index=new_i,
                name=name,
                standard_name=standardize_channel_name(name),
                type=ChannelType.EEG,
                unit=self.task_config.signal_unit or "uV",
                sampling_rate=sfreq,
                status=ChannelStatus.UNKNOWN,
            ))
        return ch_infos

    def import_subject(
        self,
        poly5_path: Path,
        subject_id: Optional[str] = None,
        session_id: str = "ses-01",
        max_seconds: Optional[float] = None,
    ) -> ImportResult:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        poly5_path = Path(poly5_path)
        subject_id = subject_id or poly5_path.parent.name

        sfreq, names, data = _read_poly5(poly5_path, max_seconds=max_seconds)
        keep_idx = [i for i, n in enumerate(names) if _is_eeg(n)]
        if not keep_idx:
            raise ValueError(f"{poly5_path.name}: no EEG (UNI/BIP) channels found")
        signal = np.ascontiguousarray(data[keep_idx, :])
        signal, storage_unit, orig_unit = convert_to_storage_unit(
            signal, source_unit="uV", pool_config=self.pool.config,
        )
        n_samples = signal.shape[1]

        channel_infos = self._build_channel_infos(names, keep_idx, sfreq)
        channel_ids = [ci.channel_id for ci in channel_infos]

        dataset_id = self.task_config.dataset_id
        run_id = "run-01"
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
                    annotation_id=f"ann_modality_{run_id}",
                    name="modality", value="ear_eeg",
                ),
                CategoricalAnnotation(
                    annotation_id=f"ann_prov_{run_id}",
                    name="label_provenance", value="unresolved",
                ),
            ],
            custom_fields={"modality": "ear_eeg", "label_provenance": "unresolved"},
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
            paradigm_details={"modality": "ear_eeg"},
        )
        self.pool.register_run(run_meta)
        logger.info(
            "Imported ear-EEG-AAD %s: %d EEG ch x %d samples (no attention labels)",
            subject_id, len(channel_ids), n_samples,
        )
        return ImportResult(
            atoms=[atom], run_meta=run_meta,
            channel_infos=channel_infos, warnings=warnings_list,
        )

    def import_dataset(
        self,
        root: Path,
        subjects: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        dataset_id = self.task_config.dataset_id

        poly5_files = sorted(
            root.glob("sub*/*.Poly5"),
            key=lambda p: int(re.sub(r"\D", "", p.parent.name) or 0),
        ) or sorted(root.glob("*.Poly5"))
        if subjects:
            poly5_files = [p for p in poly5_files if p.parent.name in set(subjects)]
        if max_subjects:
            poly5_files = poly5_files[:max_subjects]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len(poly5_files), original_format="poly5",
        ))

        results: List[ImportResult] = []
        for poly5 in poly5_files:
            sub = poly5.parent.name
            self.pool.register_subject(SubjectMeta(subject_id=sub, dataset_id=dataset_id))
            try:
                results.append(self.import_subject(
                    poly5, subject_id=sub, max_seconds=max_seconds,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", poly5.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("ear-EEG-AAD import complete: %d subjects", len(results))
        return results


# Auto-register
register_importer("ear_eeg_aad", EarEEGAADImporter)
