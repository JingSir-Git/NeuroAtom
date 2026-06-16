"""EEG-iEEG Verbal Working Memory Importer.

Dimakopoulos et al. (2022) dataset. 15 epilepsy patients with
simultaneous scalp EEG (19 ch @ 200 Hz) + iEEG (32-80 ch @ 2000 Hz).

BIDS format, EDF files. Both modalities use the same trial events
(modified Sternberg task). iEEG electrodes have MNI coordinates
and anatomical labels.

Special features:
    - Dual-modality: each trial produces a paired (EEG, iEEG) atom set
    - iEEG may be SEEG, ECoG, or mixed (sub-10 through sub-13)
    - AtomRelation links EEG ↔ iEEG atoms for the same trial
    - Electrode coordinates preserved in custom_fields
    - Working memory parameters (SetSize, Match, Correct) as annotations

Data layout (BIDS):
    original/
        sub-01/ ... sub-15/
            ses-01/ ... ses-0N/
                eeg/
                    sub-XX_ses-XX_task-verbalWM_run-01_eeg.edf
                    sub-XX_ses-XX_task-verbalWM_run-01_channels.tsv
                    sub-XX_ses-XX_task-verbalWM_run-01_events.tsv
                ieeg/
                    sub-XX_ses-XX_task-verbalWM_run-01_ieeg.edf
                    sub-XX_ses-XX_task-verbalWM_run-01_channels.tsv
                    sub-XX_ses-XX_task-verbalWM_run-01_electrodes.tsv
                    sub-XX_ses-XX_task-verbalWM_run-01_events.tsv
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the EEG-iEEG WM importer")

from neuroatom.core.annotation import CategoricalAnnotation, NumericAnnotation
from neuroatom.core.atom import Atom, AtomRelation, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import AtomType, ChannelStatus, ChannelType
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.core.session import SessionMeta
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.unit_convert import convert_to_storage_unit
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)

# iEEG channel type mapping from BIDS
_IEEG_TYPE_MAP = {
    "SEEG": ChannelType.EEG,   # depth electrodes → treat as EEG in NeuroAtom
    "ECOG": ChannelType.EEG,   # grid electrodes → treat as EEG
    "EEG": ChannelType.EEG,
    "EOG": ChannelType.EOG,
    "ECG": ChannelType.ECG,
    "EMG": ChannelType.EMG,
}


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a BIDS TSV file."""
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def _parse_channels_tsv(
    channels_path: Path,
    sfreq: float,
) -> List[ChannelInfo]:
    """Parse a BIDS channels.tsv for this dataset."""
    rows = _read_tsv(channels_path)
    ch_infos = []

    for idx, row in enumerate(rows):
        name = row.get("name", f"Ch_{idx}")
        bids_type = row.get("type", "EEG").upper()
        unit = row.get("units", "μV").replace("μV", "uV")

        ch_type = _IEEG_TYPE_MAP.get(bids_type, ChannelType.OTHER)

        ch_infos.append(ChannelInfo(
            channel_id=f"ch_{idx:03d}",
            index=idx,
            name=name,
            standard_name=standardize_channel_name(name),
            type=ch_type,
            unit=unit,
            sampling_rate=sfreq,
            status=ChannelStatus.UNKNOWN,
            custom_fields={"bids_type": bids_type},
        ))

    return ch_infos


def _parse_electrodes_tsv(
    electrodes_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Parse iEEG electrodes.tsv into a dict of electrode info.

    Returns: {name: {x, y, z, size, anatomical_location}}
    """
    rows = _read_tsv(electrodes_path)
    electrodes = {}
    for row in rows:
        name = row.get("name", "")
        try:
            x = float(row.get("x", "nan"))
            y = float(row.get("y", "nan"))
            z = float(row.get("z", "nan"))
        except ValueError:
            x = y = z = float("nan")

        electrodes[name] = {
            "x": x, "y": y, "z": z,
            "size": row.get("size", ""),
            "anatomical_location": row.get("AnatomicalLocation", ""),
        }
    return electrodes


def _parse_events_tsv(events_path: Path) -> List[Dict[str, Any]]:
    """Parse events.tsv for the Sternberg WM task.

    Returns list of trial dicts with: onset, duration, nTrial,
    begSample, endSample, SetSize, ProbeLetter, Match, Correct,
    ResponseTime, Artifact.
    """
    rows = _read_tsv(events_path)
    trials = []
    for row in rows:
        try:
            trial = {
                "onset": float(row.get("onset", 0)),
                "duration": float(row.get("duration", 8)),
                "n_trial": int(float(row.get("nTrial", 0))),
                "beg_sample": int(float(row.get("begSample", 0))),
                "end_sample": int(float(row.get("endSample", 0))),
                "set_size": int(float(row.get("SetSize", 0))),
                "probe_letter": row.get("ProbeLetter", ""),
                "match": row.get("Match", ""),
                "correct": int(float(row.get("Correct", 0))),
                "response_time": float(row.get("ResponseTime", 0)),
                "artifact": int(float(row.get("Artifact", 0))),
            }
            trials.append(trial)
        except (ValueError, TypeError) as e:
            logger.warning("Skipping invalid event row: %s", e)
    return trials


class EEGiEEGWMImporter(BaseImporter):
    """Importer for the EEG-iEEG Verbal Working Memory dataset.

    Imports simultaneous scalp EEG and intracranial EEG, creating
    linked atom pairs for each trial.
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Detect by BIDS description containing 'intracranial EEG'."""
        path = Path(path)
        desc = path / "dataset_description.json"
        if not desc.exists():
            return False
        try:
            with open(desc, encoding="utf-8") as f:
                d = json.load(f)
            name = d.get("Name", "").lower()
            return "intracranial" in name and "verbal" in name
        except Exception:
            return False

    def load_raw(self, path):
        raise NotImplementedError("Use import_dataset() instead.")

    def extract_channel_infos(self, raw):
        return []

    def extract_events(self, raw):
        return None

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def import_dataset(
        self,
        bids_root: Path,
        subjects: Optional[List[str]] = None,
        max_sessions: Optional[int] = None,
        modalities: Optional[List[str]] = None,
    ) -> List[ImportResult]:
        """Import the EEG-iEEG WM dataset.

        Args:
            bids_root: Path to BIDS root.
            subjects: Subset of subjects (e.g. ["sub-01", "sub-02"]).
            max_sessions: Max sessions per subject.
            modalities: ["eeg", "ieeg"] or subset. Default: both.

        Returns:
            List of ImportResult per subject-session-modality.
        """
        bids_root = Path(bids_root)
        dataset_id = self._task_config.dataset_id
        modalities = modalities or ["eeg", "ieeg"]

        # Read participants
        participants = _read_tsv(bids_root / "participants.tsv")

        # Register dataset
        desc_path = bids_root / "dataset_description.json"
        ds_name = self._task_config.dataset_name
        if desc_path.exists():
            with open(desc_path, encoding="utf-8") as f:
                desc = json.load(f)
            ds_name = desc.get("Name", ds_name)

        self._pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id,
            name=ds_name,
            task_types=["working_memory"],
            n_subjects=len(participants),
            original_format="edf",
            license="CC0",
        ))

        # Discover subjects
        sub_dirs = sorted(bids_root.glob("sub-*"))
        sub_dirs = [d for d in sub_dirs if d.is_dir() and d.name.startswith("sub-")]
        if subjects:
            sub_dirs = [d for d in sub_dirs if d.name in subjects]

        results = []
        for sub_dir in sub_dirs:
            sub_id = sub_dir.name

            # Find participant info
            p_info = {}
            for p in participants:
                if p.get("participant_id") == sub_id:
                    p_info = p
                    break

            age_str = p_info.get("age", "")
            sex = p_info.get("sex", "").upper()
            self._pool.register_subject(SubjectMeta(
                subject_id=sub_id,
                dataset_id=dataset_id,
                age=int(age_str) if age_str else None,
                sex=sex if sex in ("M", "F", "O") else None,
                custom_fields={
                    "pathology": p_info.get("pathology", ""),
                },
            ))

            # Discover sessions
            ses_dirs = sorted(sub_dir.glob("ses-*"))
            if max_sessions:
                ses_dirs = ses_dirs[:max_sessions]

            for ses_dir in ses_dirs:
                ses_id = ses_dir.name

                # Register session (use EEG sfreq as default)
                self._pool.register_session(SessionMeta(
                    session_id=ses_id,
                    subject_id=sub_id,
                    dataset_id=dataset_id,
                    sampling_rate=200.0,
                ))

                try:
                    ses_results = self._import_session(
                        bids_root, sub_id, ses_dir, ses_id,
                        dataset_id, modalities,
                    )
                    results.extend(ses_results)
                except Exception as e:
                    logger.error(
                        "Failed %s/%s: %s", sub_id, ses_id, e,
                    )

        # Post-import quality assessment
        try:
            tier = self._pool.assess_quality(dataset_id)
            if tier:
                logger.info("Quality tier for %s: %s", dataset_id, tier)
        except Exception:
            pass

        total_atoms = sum(len(r.atoms) for r in results)
        logger.info(
            "EEG-iEEG WM import complete: %d results, %d atoms",
            len(results), total_atoms,
        )
        return results

    def _import_session(
        self,
        bids_root: Path,
        sub_id: str,
        ses_dir: Path,
        ses_id: str,
        dataset_id: str,
        modalities: List[str],
    ) -> List[ImportResult]:
        """Import one session: load EEG and iEEG, create trial atoms."""
        results = []

        # Parse events (shared between EEG and iEEG)
        # Use iEEG events as reference (both have the same trial structure)
        events = None
        for mod in ["eeg", "ieeg"]:
            mod_dir = ses_dir / mod
            if mod_dir.exists():
                events_path = list(mod_dir.glob("*_events.tsv"))
                if events_path:
                    events = _parse_events_tsv(events_path[0])
                    break

        if not events:
            logger.warning("No events found for %s/%s", sub_id, ses_id)
            return []

        # Track atom IDs for cross-modality linking
        eeg_atom_ids: Dict[int, str] = {}
        ieeg_atom_ids: Dict[int, str] = {}

        for modality in modalities:
            mod_dir = ses_dir / modality
            if not mod_dir.exists():
                continue

            # Find EDF file
            suffix = "eeg" if modality == "eeg" else "ieeg"
            edf_files = list(mod_dir.glob(f"*_{suffix}.edf"))
            if not edf_files:
                continue
            edf_path = edf_files[0]

            # Parse channels
            ch_files = list(mod_dir.glob("*_channels.tsv"))
            if not ch_files:
                continue
            ch_tsv = ch_files[0]

            # Parse electrodes (iEEG only)
            electrodes = {}
            if modality == "ieeg":
                elec_files = list(mod_dir.glob("*_electrodes.tsv"))
                if elec_files:
                    electrodes = _parse_electrodes_tsv(elec_files[0])

            # Parse sidecar JSON for sampling frequency
            json_files = list(mod_dir.glob(f"*_{suffix}.json"))
            sfreq = 200.0
            if json_files:
                with open(json_files[0], encoding="utf-8") as f:
                    sidecar = json.load(f)
                sfreq = sidecar.get("SamplingFrequency", sfreq)

            # Parse channels
            channel_infos = _parse_channels_tsv(ch_tsv, sfreq)

            # Enrich iEEG channels with electrode coordinates
            if electrodes:
                for ci in channel_infos:
                    elec = electrodes.get(ci.name)
                    if elec and not (np.isnan(elec["x"]) or np.isnan(elec["y"]) or np.isnan(elec["z"])):
                        ci.location = ElectrodeLocation(
                            x=elec["x"], y=elec["y"], z=elec["z"],
                            coordinate_system="MNI",
                            coordinate_units="mm",
                        )
                        ci.custom_fields["anatomical_location"] = elec.get(
                            "anatomical_location", ""
                        )

            # Load EDF data
            raw = mne.io.read_raw_edf(
                str(edf_path), preload=True, verbose="WARNING",
            )
            data = raw.get_data() * 1e6  # V → µV
            actual_sfreq = raw.info["sfreq"]

            run_id = f"run-{modality}"

            self._pool.ensure_session(dataset_id, sub_id, ses_id, actual_sfreq)
            self._pool.ensure_run(dataset_id, sub_id, ses_id, run_id)
            self._write_channels_json(dataset_id, sub_id, ses_id, channel_infos)

            # Create trial atoms
            from neuroatom.storage.signal_store import ShardManager
            from neuroatom.storage.metadata_store import AtomJSONLWriter
            from neuroatom.storage import paths as P

            ch_names = [ci.name for ci in channel_infos]
            n_ch = len(channel_infos)
            stored_atoms = []

            max_shard_mb = self._pool.config.get("storage", {}).get(
                "max_shard_size_mb", 200.0,
            )
            compression = self._pool.config.get("storage", {}).get(
                "compression", "gzip",
            )

            with ShardManager(
                pool_root=self._pool.root, dataset_id=dataset_id,
                subject_id=sub_id, session_id=ses_id, run_id=run_id,
                max_shard_size_mb=max_shard_mb, compression=compression,
            ) as shard_mgr:
                jsonl_path = P.atoms_jsonl_path(
                    self._pool.root, dataset_id, sub_id, ses_id, run_id,
                )
                with AtomJSONLWriter(jsonl_path) as aw:
                    for trial in events:
                        trial_idx = trial["n_trial"]

                        # Sample range (1-based in events.tsv)
                        beg = max(trial["beg_sample"] - 1, 0)
                        end = trial["end_sample"]
                        end = min(end, data.shape[1])
                        duration = end - beg
                        if duration <= 0:
                            continue

                        trial_signal = data[:n_ch, beg:end].copy().astype(np.float32)

                        # Annotations
                        apfx = f"{run_id}_{trial_idx:04d}"
                        annotations = [
                            NumericAnnotation(
                                annotation_id=f"ann_ss_{apfx}",
                                name="set_size",
                                numeric_value=float(trial["set_size"]),
                            ),
                            CategoricalAnnotation(
                                annotation_id=f"ann_match_{apfx}",
                                name="match",
                                value=trial["match"],
                            ),
                            CategoricalAnnotation(
                                annotation_id=f"ann_corr_{apfx}",
                                name="correct",
                                value=str(trial["correct"]),
                            ),
                            NumericAnnotation(
                                annotation_id=f"ann_rt_{apfx}",
                                name="response_time",
                                numeric_value=trial["response_time"],
                            ),
                            CategoricalAnnotation(
                                annotation_id=f"ann_mod_{apfx}",
                                name="modality",
                                value=modality,
                            ),
                        ]

                        if trial["artifact"]:
                            annotations.append(CategoricalAnnotation(
                                annotation_id=f"ann_art_{apfx}",
                                name="artifact",
                                value="true",
                            ))

                        # Determine iEEG subtypes present
                        if modality == "ieeg":
                            seeg_count = sum(
                                1 for ci in channel_infos
                                if ci.custom_fields.get("bids_type") == "SEEG"
                            )
                            ecog_count = sum(
                                1 for ci in channel_infos
                                if ci.custom_fields.get("bids_type") == "ECOG"
                            )
                            if seeg_count > 0:
                                annotations.append(CategoricalAnnotation(
                                    annotation_id=f"ann_seeg_{apfx}",
                                    name="has_seeg", value="true",
                                ))
                            if ecog_count > 0:
                                annotations.append(CategoricalAnnotation(
                                    annotation_id=f"ann_ecog_{apfx}",
                                    name="has_ecog", value="true",
                                ))

                        atom_id = compute_atom_id(
                            dataset_id, sub_id, ses_id, run_id, trial_idx,
                        )

                        # Store for cross-linking
                        if modality == "eeg":
                            eeg_atom_ids[trial_idx] = atom_id
                        else:
                            ieeg_atom_ids[trial_idx] = atom_id

                        # Cross-modality relation
                        relations = []
                        if modality == "eeg" and trial_idx in ieeg_atom_ids:
                            relations.append(AtomRelation(
                                target_atom_id=ieeg_atom_ids[trial_idx],
                                relation_type="simultaneous_recording",
                                description="Paired iEEG atom for same trial",
                            ))
                        elif modality == "ieeg" and trial_idx in eeg_atom_ids:
                            relations.append(AtomRelation(
                                target_atom_id=eeg_atom_ids[trial_idx],
                                relation_type="simultaneous_recording",
                                description="Paired EEG atom for same trial",
                            ))

                        atom = Atom(
                            atom_id=atom_id,
                            dataset_id=dataset_id,
                            subject_id=sub_id,
                            session_id=ses_id,
                            run_id=run_id,
                            atom_type=AtomType.TRIAL,
                            signal_ref=SignalRef(
                                file_path="", internal_path="",
                                shape=(n_ch, duration),
                            ),
                            temporal=TemporalInfo(
                                onset_sample=beg,
                                duration_samples=duration,
                                onset_seconds=beg / actual_sfreq,
                                duration_seconds=duration / actual_sfreq,
                            ),
                            channel_ids=ch_names,
                            n_channels=n_ch,
                            sampling_rate=actual_sfreq,
                            signal_unit="uV",
                            annotations=annotations,
                            relations=relations,
                            custom_fields={
                                "modality": modality,
                                "set_size": trial["set_size"],
                                "match": trial["match"],
                                "correct": trial["correct"],
                            },
                        )

                        sig_ref = shard_mgr.write_atom_signal(
                            atom_id, trial_signal,
                        )
                        atom.signal_ref = sig_ref
                        aw.write_atom(atom)
                        stored_atoms.append(atom)

            run_meta = RunMeta(
                run_id=run_id, session_id=ses_id,
                subject_id=sub_id, dataset_id=dataset_id,
                task_type="working_memory",
                n_trials=len(stored_atoms),
            )
            self._pool.register_run(run_meta)

            logger.info(
                "Imported %s/%s/%s/%s: %d atoms (%d ch @ %.0f Hz)",
                sub_id, ses_id, modality, run_id,
                len(stored_atoms), n_ch, actual_sfreq,
            )

            results.append(ImportResult(
                atoms=stored_atoms,
                run_meta=run_meta,
                channel_infos=channel_infos,
                warnings=[],
            ))

        return results


# Auto-register
register_importer("eeg_ieeg_wm", EEGiEEGWMImporter)
