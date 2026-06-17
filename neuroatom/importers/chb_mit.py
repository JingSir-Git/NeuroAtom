"""CHB-MIT Importer: pediatric scalp-EEG seizure dataset (PhysioNet).

23 pediatric epilepsy subjects (cases chb01–chb24), continuous scalp EEG in a
bipolar longitudinal montage, 256 Hz. Each case is a directory of ~1 h EDF files
plus a ``chbXX-summary.txt`` listing the channel montage and, per file, the
number of seizures with their start/end times (seconds).

Each EDF file becomes one CONTINUOUS_SEGMENT atom carrying the seizure intervals
as annotations (``has_seizure`` flag, ``n_seizures`` count, and the
``seizure_intervals`` list in custom_fields). Downstream code can window each
recording into ictal / interictal segments via the assembler — we import the
recording faithfully rather than baking in a windowing/sampling choice.

Data layout::

    CHB-MIT/
        chb01/
            chb01-summary.txt
            chb01_01.edf ... chb01_46.edf
            chb01_03.edf.seizures      # binary marker (ignored; times come from summary)
        chb02/ ...
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the CHB-MIT importer")

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

_CASE_RE = re.compile(r"chb\d+", re.IGNORECASE)


def _parse_summary(path: Path) -> Dict[str, Any]:
    """Parse a chbXX-summary.txt → sampling rate, channels, per-file seizures."""
    text = path.read_text(encoding="latin-1", errors="replace")
    sr = re.search(r"Data Sampling Rate:\s*([\d.]+)", text)
    srate = float(sr.group(1)) if sr else 256.0
    # Channel list appears once near the top.
    channels = re.findall(r"Channel \d+:\s*(.+)", text)

    files: Dict[str, List[Tuple[float, float]]] = {}
    for blk in re.split(r"File Name:\s*", text)[1:]:
        fname = blk.splitlines()[0].strip()
        starts = [float(x) for x in re.findall(
            r"Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)", blk)]
        ends = [float(x) for x in re.findall(
            r"Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)", blk)]
        files[fname] = list(zip(starts, ends))
    return {"srate": srate, "channels": channels, "files": files}


class CHBMITImporter(BaseImporter):
    """Importer for the CHB-MIT scalp-EEG seizure dataset."""

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_file():
            return path.suffix.lower() == ".edf" and bool(
                _CASE_RE.search(path.stem)
            ) and any(path.parent.glob("*-summary.txt"))
        if path.is_dir():
            if any(path.glob("*-summary.txt")) and any(path.glob("*.edf")):
                return True
            return any(path.glob("chb*/*-summary.txt"))
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
            ch_type = ChannelType.EEG  # bipolar EEG derivations
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

    def import_case(
        self,
        case_dir: Path,
        subject_id: Optional[str] = None,
        session_id: str = "ses-01",
        max_files: Optional[int] = None,
        max_seconds: Optional[float] = None,
        only_files: Optional[List[str]] = None,
    ) -> ImportResult:
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.signal_store import ShardManager

        case_dir = Path(case_dir)
        subject_id = subject_id or case_dir.name
        summary_path = next(iter(case_dir.glob("*-summary.txt")), None)
        summary = _parse_summary(summary_path) if summary_path else {"files": {}}
        seizure_map = summary.get("files", {})

        # NB: glob "{case}*.edf" (not "{case}_*.edf") — session-split cases store
        # files as chb17a_03.edf / chb17b_*.edf / chb17c_*.edf (no "chb17_" prefix),
        # which a "{case}_*" glob would silently drop (the whole subject).
        edf_files = sorted(case_dir.glob(f"{case_dir.name}*.edf"))
        if only_files:
            wanted = set(only_files)
            edf_files = [f for f in edf_files if f.name in wanted]
        if max_files is not None:
            edf_files = edf_files[:max_files]

        dataset_id = self.task_config.dataset_id
        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)
        self.pool.ensure_session(dataset_id, subject_id, session_id)

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        stored_atoms: List[Atom] = []
        all_warnings: List[str] = []
        channel_infos: List[ChannelInfo] = []

        for edf_path in edf_files:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            rec_secs = raw.n_times / sfreq
            keep_secs = min(rec_secs, max_seconds) if max_seconds else rec_secs
            n_keep = int(keep_secs * sfreq)

            signal = raw.get_data(stop=n_keep).astype(np.float32)  # (n_ch, n) Volts
            signal, storage_unit, orig_unit = convert_to_storage_unit(
                signal, source_unit="V", pool_config=self.pool.config,
            )
            n_ch, n_samples = signal.shape

            channel_infos = self._build_channel_infos(raw.ch_names, sfreq)
            channel_ids = [ci.channel_id for ci in channel_infos]

            # Seizure intervals, clipped to the imported window.
            raw_seizures = seizure_map.get(edf_path.name, [])
            seizures = []
            for s, e in raw_seizures:
                s2, e2 = max(0.0, s), min(e, keep_secs)
                if e2 > s2:
                    seizures.append([round(s2, 3), round(e2, 3)])

            run_id = edf_path.stem  # e.g. chb01_03
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            annotations: List[Any] = [
                CategoricalAnnotation(
                    annotation_id=f"ann_seiz_{run_id}",
                    name="has_seizure", value="true" if seizures else "false",
                ),
                NumericAnnotation(
                    annotation_id=f"ann_nseiz_{run_id}",
                    name="n_seizures", numeric_value=float(len(seizures)),
                ),
            ]
            for i, (s, e) in enumerate(seizures):
                annotations.append(NumericAnnotation(
                    annotation_id=f"ann_seizonset_{run_id}_{i}",
                    name="seizure_onset_seconds", numeric_value=float(s),
                ))

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
                annotations=annotations,
                custom_fields={
                    "has_seizure": bool(seizures),
                    "seizure_intervals": seizures,
                    "recording": edf_path.name,
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
                dataset_id=dataset_id, task_type=self.task_config.task_type,
                n_trials=1,
                paradigm_details={"recording": edf_path.name, "n_seizures": len(seizures)},
            ))
            self._write_channels_json(dataset_id, subject_id, session_id, channel_infos)
            stored_atoms.append(atom)

        logger.info(
            "Imported CHB-MIT %s: %d recordings (%d with seizures)",
            subject_id, len(stored_atoms),
            sum(1 for a in stored_atoms if a.custom_fields.get("has_seizure")),
        )
        run_meta = RunMeta(
            run_id=stored_atoms[0].run_id if stored_atoms else "run-01",
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
        cases: Optional[List[str]] = None,
        max_cases: Optional[int] = None,
        max_files: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> List[ImportResult]:
        root = Path(root)
        dataset_id = self.task_config.dataset_id

        case_dirs = sorted(
            (d for d in root.glob("chb*") if d.is_dir()),
            key=lambda d: int(re.sub(r"\D", "", d.name) or 0),
        )
        if cases:
            case_dirs = [d for d in case_dirs if d.name in set(cases)]
        if max_cases:
            case_dirs = case_dirs[:max_cases]

        self.pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id, name=self.task_config.dataset_name,
            task_types=[self.task_config.task_type],
            n_subjects=len(case_dirs), original_format="edf",
        ))

        results: List[ImportResult] = []
        for case_dir in case_dirs:
            self.pool.register_subject(SubjectMeta(
                subject_id=case_dir.name, dataset_id=dataset_id,
            ))
            try:
                results.append(self.import_case(
                    case_dir, max_files=max_files, max_seconds=max_seconds,
                ))
            except Exception as e:
                logger.error("Failed to import %s: %s", case_dir.name, e)

        try:
            self.pool.assess_quality(dataset_id)
        except Exception:
            pass

        logger.info("CHB-MIT import complete: %d cases", len(results))
        return results


# Auto-register
register_importer("chb_mit", CHBMITImporter)
