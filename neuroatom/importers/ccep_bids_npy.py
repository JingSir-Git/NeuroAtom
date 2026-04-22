"""CCEP BIDS NPY Importer: simultaneous EEG + sEEG (iEEG) epoch data.

Handles the CCEP-COREG dataset (Parmigiani et al.) and similar BIDS-derivative
datasets where preprocessed epochs are stored as NumPy .npy arrays with
BIDS-style sidecar TSVs for channel info and epoch metadata.

Key features:
    - Imports both scalp EEG and iEEG (sEEG) epochs as separate atoms
    - Links EEG ↔ iEEG atoms within the same run via AtomRelation
    - Parses stimulation parameters from trial_type strings
    - Preserves channel quality (good/bad) from channels.tsv
    - Stores electrode coordinates as atom custom_fields

Data structure expected:
    derivatives/epochs/sub-XX/
        eeg/
            sub-XX_task-*_run-NN_epochs.npy      (epochs × channels × samples)
            sub-XX_task-*_run-NN_epochs.json      (baseline info)
            sub-XX_task-*_run-NN_epochs.tsv       (per-epoch trial_type)
            sub-XX_task-*_run-NN_channels.tsv     (name, type, units, status)
            sub-XX_task-*_electrodes.tsv           (electrode coords)
        ieeg/
            sub-XX_task-*_run-NN_epochs.npy
            sub-XX_task-*_run-NN_epochs.json
            sub-XX_task-*_run-NN_epochs.tsv
            sub-XX_task-*_run-NN_channels.tsv
            sub-XX_task-*_space-*_electrodes.tsv
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.core.annotation import CategoricalAnnotation, NumericAnnotation, TextAnnotation
from neuroatom.core.atom import Atom, AtomRelation, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import AtomType, ChannelStatus, ChannelType
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.quality import QualityInfo
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BIDS TSV / JSON helpers
# ---------------------------------------------------------------------------

def _read_tsv(path: Path) -> List[Dict[str, str]]:
    """Read a BIDS-style TSV file into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_stim_description(trial_type: str) -> Dict[str, str]:
    """Parse CCEP stimulation description.

    Example: "R'6-7 5ma 0.5ms 0.5hz parallel wh_wh"
    → {stim_contact: "R'6-7", stim_intensity_ma: "5", stim_duration_ms: "0.5",
       stim_frequency_hz: "0.5", stim_angle: "parallel", stim_tissue: "wh_wh"}
    """
    parts = trial_type.strip().split()
    result: Dict[str, str] = {"stim_description": trial_type}

    if len(parts) >= 1:
        result["stim_contact"] = parts[0]
    if len(parts) >= 2:
        m = re.match(r"([\d.]+)ma", parts[1], re.IGNORECASE)
        if m:
            result["stim_intensity_ma"] = m.group(1)
    if len(parts) >= 3:
        m = re.match(r"([\d.]+)ms", parts[2], re.IGNORECASE)
        if m:
            result["stim_duration_ms"] = m.group(1)
    if len(parts) >= 4:
        m = re.match(r"([\d.]+)hz", parts[3], re.IGNORECASE)
        if m:
            result["stim_frequency_hz"] = m.group(1)
    if len(parts) >= 5:
        result["stim_angle"] = parts[4]
    if len(parts) >= 6:
        result["stim_tissue"] = parts[5]

    return result


def _read_electrodes(
    electrodes_tsv: Path,
    coordsystem_json: Optional[Path] = None,
) -> Tuple[Dict[str, ElectrodeLocation], Dict[str, Dict[str, Any]]]:
    """Read electrode coordinates from a BIDS electrodes.tsv file.

    Returns:
        (locations, electrode_meta)
        - locations: mapping channel_name → ElectrodeLocation
        - electrode_meta: mapping channel_name → {material, manufacturer, size, ...}
    """
    rows = _read_tsv(electrodes_tsv)

    # Determine coordinate system and units from coordsystem JSON
    coord_system = "unknown"
    coord_units = "m"  # BIDS default
    coord_system_info: Dict[str, Any] = {}

    if coordsystem_json and coordsystem_json.exists():
        cs = _read_json(coordsystem_json)
        coord_system_info = cs
        # BIDS uses EEGCoordinateSystem or iEEGCoordinateSystem
        coord_system = (
            cs.get("EEGCoordinateSystem")
            or cs.get("iEEGCoordinateSystem")
            or "unknown"
        )
        coord_units = (
            cs.get("EEGCoordinateUnits")
            or cs.get("iEEGCoordinateUnits")
            or "m"
        )

    locations: Dict[str, ElectrodeLocation] = {}
    electrode_meta: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        name = row["name"]
        try:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
        except (ValueError, KeyError):
            continue

        locations[name] = ElectrodeLocation(
            x=x, y=y, z=z,
            coordinate_system=coord_system,
            coordinate_units=coord_units,
        )

        meta: Dict[str, Any] = {}
        if "material" in row:
            meta["material"] = row["material"]
        if "manufacturer" in row:
            meta["manufacturer"] = row["manufacturer"]
        if "size" in row:
            try:
                meta["size_mm2"] = float(row["size"])
            except ValueError:
                meta["size"] = row["size"]
        electrode_meta[name] = meta

    return locations, electrode_meta


def _find_electrodes_tsv(mod_dir: Path, modality: str) -> Optional[Path]:
    """Find the electrode TSV file for a modality directory.

    Handles both EEG (simple name) and iEEG (with space- prefix) naming.
    """
    candidates = sorted(mod_dir.glob("*electrodes*.tsv"))
    if not candidates:
        return None

    # Prefer MNI space for iEEG if available
    if modality == "ieeg":
        mni = [c for c in candidates if "MNI" in c.name]
        if mni:
            return mni[0]
    return candidates[0]


def _find_coordsystem_json(mod_dir: Path, electrodes_tsv: Optional[Path]) -> Optional[Path]:
    """Find the matching coordsystem JSON for an electrode file."""
    candidates = sorted(mod_dir.glob("*coordsystem*.json"))
    if not candidates:
        return None

    # If electrode TSV has a space- entity, match it
    if electrodes_tsv:
        space_match = re.search(r"space-(\S+?)_", electrodes_tsv.name)
        if space_match:
            space = space_match.group(1)
            matching = [c for c in candidates if space in c.name]
            if matching:
                return matching[0]

    return candidates[0]


def _channel_type_from_bids(bids_type: str) -> ChannelType:
    """Map BIDS channel type string to NeuroAtom ChannelType."""
    mapping = {
        "eeg": ChannelType.EEG,
        "ieeg": ChannelType.SEEG,
        "seeg": ChannelType.SEEG,
        "ecog": ChannelType.ECOG,
        "eog": ChannelType.EOG,
        "emg": ChannelType.EMG,
        "ecg": ChannelType.ECG,
        "misc": ChannelType.MISC,
        "stim": ChannelType.STIM,
        "trig": ChannelType.TRIGGER,
        "ref": ChannelType.REF,
    }
    return mapping.get(bids_type.lower(), ChannelType.OTHER)


# ---------------------------------------------------------------------------
# CCEP BIDS NPY Importer
# ---------------------------------------------------------------------------

class CCEPImporter(BaseImporter):
    """Importer for simultaneous EEG + sEEG (CCEP) BIDS-derivative datasets.

    Each run produces two sets of atoms: scalp EEG epochs and iEEG epochs,
    linked via AtomRelation with relation_type='cross_modal_paired_run'.
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if path is a CCEP BIDS-derivative dataset."""
        path = Path(path)
        if path.is_dir():
            # Look for derivatives/epochs/ structure with both eeg/ and ieeg/
            epochs_dir = path / "derivatives" / "epochs"
            if not epochs_dir.exists():
                epochs_dir = path  # Maybe we're pointing directly to epochs
            subs = [d for d in epochs_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
            for sub in subs[:1]:  # Check first subject
                if (sub / "eeg").exists() and (sub / "ieeg").exists():
                    return True
        return False

    def load_raw(self, path):
        raise NotImplementedError(
            "CCEPImporter.load_raw() is not used directly. "
            "Use import_subject() for multi-modal CCEP datasets."
        )

    def extract_channel_infos(self, raw):
        raise NotImplementedError("Use _build_channel_infos() instead.")

    def extract_events(self, raw):
        return None

    # ------------------------------------------------------------------
    # Channel info builder from BIDS channels.tsv
    # ------------------------------------------------------------------

    def _build_channel_infos(
        self,
        channels_tsv: Path,
        modality: str,
        electrode_locations: Optional[Dict[str, ElectrodeLocation]] = None,
        electrode_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[List[ChannelInfo], List[str]]:
        """Build ChannelInfo list from a BIDS channels.tsv file.

        Enriches each channel with:
        - Electrode location (3D coordinates from electrodes.tsv)
        - Reference type (from channels.tsv 'reference' column)
        - Filter settings (low_cutoff, high_cutoff in custom_fields)
        - Electrode material/manufacturer/size (in custom_fields)

        Returns:
            (channel_infos, bad_channel_names)
        """
        rows = _read_tsv(channels_tsv)
        ch_infos = []
        bad_channels = []
        electrode_locations = electrode_locations or {}
        electrode_meta = electrode_meta or {}

        for idx, row in enumerate(rows):
            ch_name = row["name"]
            bids_type = row.get("type", modality)
            ch_type = _channel_type_from_bids(bids_type)

            # Task config overrides
            override = self.task_config.channel_type_overrides.get(ch_name)
            if override:
                ch_type = ChannelType(override)

            # Skip excluded channels
            if ch_name in self.task_config.exclude_channels:
                continue

            status = ChannelStatus.GOOD if row.get("status", "good") == "good" else ChannelStatus.BAD
            if status == ChannelStatus.BAD:
                bad_channels.append(ch_name)

            srate = float(row.get("sampling_frequency", 1000))
            unit = row.get("units", "V")
            std_name = standardize_channel_name(ch_name) if ch_type == ChannelType.EEG else None

            # Reference from channels.tsv
            reference = row.get("reference")

            # Electrode location
            location = electrode_locations.get(ch_name)

            # Custom fields: filter settings + electrode metadata
            custom: Dict[str, Any] = {}
            if row.get("low_cutoff"):
                try:
                    custom["low_cutoff_hz"] = float(row["low_cutoff"])
                except ValueError:
                    pass
            if row.get("high_cutoff"):
                try:
                    custom["high_cutoff_hz"] = float(row["high_cutoff"])
                except ValueError:
                    pass

            # Electrode material/manufacturer/size from electrodes.tsv
            emeta = electrode_meta.get(ch_name, {})
            if emeta:
                custom.update(emeta)

            ch_infos.append(ChannelInfo(
                channel_id=f"{modality}_{idx:03d}",
                index=idx,
                name=ch_name,
                standard_name=std_name,
                type=ch_type,
                unit=unit,
                sampling_rate=srate,
                status=status,
                reference=reference,
                location=location,
                custom_fields=custom if custom else {},
            ))

        return ch_infos, bad_channels

    # ------------------------------------------------------------------
    # Discover runs for a subject
    # ------------------------------------------------------------------

    def _discover_runs(self, sub_dir: Path, modality: str) -> List[Dict[str, Any]]:
        """Find all runs for a modality (eeg or ieeg) under a subject directory.

        Returns list of dicts with keys: run_id, run_num, epochs_npy,
        epochs_json, epochs_tsv, channels_tsv, electrodes_tsv,
        coordsystem_json, electrode_locations, electrode_meta
        """
        mod_dir = sub_dir / modality
        if not mod_dir.exists():
            return []

        # Electrode coords are per-subject (not per-run) — load once
        electrodes_tsv = _find_electrodes_tsv(mod_dir, modality)
        coordsystem_json = _find_coordsystem_json(mod_dir, electrodes_tsv)

        electrode_locations: Dict[str, ElectrodeLocation] = {}
        electrode_meta: Dict[str, Dict[str, Any]] = {}
        coord_system_info: Dict[str, Any] = {}

        if electrodes_tsv and electrodes_tsv.exists():
            electrode_locations, electrode_meta = _read_electrodes(
                electrodes_tsv, coordsystem_json
            )
            logger.info(
                "%s: loaded %d electrode positions from %s (coord: %s)",
                modality, len(electrode_locations), electrodes_tsv.name,
                electrode_locations[next(iter(electrode_locations))].coordinate_system
                if electrode_locations else "none",
            )
        else:
            logger.warning("No electrodes.tsv found for %s in %s", modality, mod_dir)

        if coordsystem_json and coordsystem_json.exists():
            coord_system_info = _read_json(coordsystem_json)

        runs = []
        for npy_path in sorted(mod_dir.glob("*_epochs.npy")):
            # Extract run ID from filename: sub-XX_task-*_run-NN_epochs.npy
            m = re.search(r"run-(\d+)", npy_path.stem)
            if not m:
                continue
            run_num = m.group(1)
            run_id = f"run-{run_num}"

            # Corresponding files (same prefix)
            json_path = npy_path.with_suffix(".json")
            tsv_path = npy_path.with_suffix(".tsv")  # epochs.tsv

            # channels.tsv naming: sub-XX_task-*_run-NN_channels.tsv
            ch_candidates = list(mod_dir.glob(f"*_run-{run_num}_channels.tsv"))
            ch_path = ch_candidates[0] if ch_candidates else None

            if ch_path is None or not ch_path.exists():
                logger.warning("No channels.tsv for %s %s, skipping.", modality, run_id)
                continue

            runs.append({
                "run_id": run_id,
                "run_num": int(run_num),
                "epochs_npy": npy_path,
                "epochs_json": json_path if json_path.exists() else None,
                "epochs_tsv": tsv_path if tsv_path.exists() else None,
                "channels_tsv": ch_path,
                "electrodes_tsv": electrodes_tsv,
                "coordsystem_json": coordsystem_json,
                "electrode_locations": electrode_locations,
                "electrode_meta": electrode_meta,
                "coord_system_info": coord_system_info,
            })

        return runs

    # ------------------------------------------------------------------
    # Import a single modality run → list of epoch atoms
    # ------------------------------------------------------------------

    def _import_modality_run(
        self,
        run_info: Dict[str, Any],
        modality: str,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        max_epochs: Optional[int] = None,
    ) -> Tuple[List[Atom], List[ChannelInfo], List[str]]:
        """Import all epochs from one modality of one run.

        Returns:
            (atoms, channel_infos, warnings)
        """
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P

        run_id = f"{run_info['run_id']}_{modality}"

        # Load epoch data
        epochs_data = np.load(str(run_info["epochs_npy"]))
        n_epochs, n_channels, n_samples = epochs_data.shape
        logger.info(
            "Loading %s %s: %d epochs × %d ch × %d samples",
            modality, run_info["run_id"], n_epochs, n_channels, n_samples,
        )

        if max_epochs is not None:
            epochs_data = epochs_data[:max_epochs]
            n_epochs = epochs_data.shape[0]

        # Channel info — enriched with electrode locations and metadata
        ch_infos, bad_channels = self._build_channel_infos(
            run_info["channels_tsv"],
            modality,
            electrode_locations=run_info.get("electrode_locations"),
            electrode_meta=run_info.get("electrode_meta"),
        )
        channel_ids = [ch.channel_id for ch in ch_infos]
        srate = ch_infos[0].sampling_rate if ch_infos else 1000.0

        # Coordinate system info for atom-level metadata
        coord_system_info = run_info.get("coord_system_info", {})
        n_electrodes_with_coords = sum(
            1 for ch in ch_infos if ch.location is not None
        )

        # Verify channel count
        if len(ch_infos) != n_channels:
            logger.warning(
                "%s %s: channels.tsv has %d entries but .npy has %d channels. Using min.",
                modality, run_info["run_id"], len(ch_infos), n_channels,
            )
            n_use = min(len(ch_infos), n_channels)
            ch_infos = ch_infos[:n_use]
            channel_ids = channel_ids[:n_use]
            epochs_data = epochs_data[:, :n_use, :]

        # Epoch metadata (trial_type etc.)
        epoch_meta = []
        if run_info.get("epochs_tsv") and run_info["epochs_tsv"].exists():
            epoch_meta = _read_tsv(run_info["epochs_tsv"])

        # Baseline info from JSON
        baseline_info = {}
        if run_info.get("epochs_json") and run_info["epochs_json"].exists():
            baseline_info = _read_json(run_info["epochs_json"])

        baseline_samples = None
        if baseline_info.get("BaselinePeriod"):
            bp = baseline_info["BaselinePeriod"]  # e.g. [-0.3, 0]
            zero_time = float(epoch_meta[0].get("zero_time", 0.3)) if epoch_meta else 0.3
            b_start = int((bp[0] + zero_time) * srate)
            b_end = int((bp[1] + zero_time) * srate)
            baseline_samples = (max(0, b_start), max(0, b_end))

        # Ensure pool hierarchy
        self.pool.ensure_session(dataset_id, subject_id, session_id, sampling_rate=srate)
        self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

        # Quality info
        quality = QualityInfo(
            overall_status="good" if len(bad_channels) < n_channels * 0.5 else "suspect",
            bad_channels=bad_channels,
        )

        max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self.pool.config.get("storage", {}).get("compression", "gzip")

        atoms = []
        all_warnings = []

        with ShardManager(
            pool_root=self.pool.root,
            dataset_id=dataset_id,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            max_shard_size_mb=max_shard_mb,
            compression=compression,
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, dataset_id, subject_id, session_id, run_id
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                for epoch_idx in range(n_epochs):
                    signal = epochs_data[epoch_idx]  # (n_channels, n_samples)

                    # Build annotations
                    annotations = []

                    # Parse trial_type for stimulation parameters
                    if epoch_idx < len(epoch_meta):
                        trial_type = epoch_meta[epoch_idx].get("trial_type", "")
                        stim_params = _parse_stim_description(trial_type)

                        annotations.append(TextAnnotation(
                            annotation_id=f"ann_trial_type_{modality}_{epoch_idx:04d}",
                            name="trial_type",
                            text_value=trial_type,
                        ))

                        if "stim_contact" in stim_params:
                            annotations.append(CategoricalAnnotation(
                                annotation_id=f"ann_stim_contact_{modality}_{epoch_idx:04d}",
                                name="stim_contact",
                                value=stim_params["stim_contact"],
                            ))

                        if "stim_intensity_ma" in stim_params:
                            annotations.append(NumericAnnotation(
                                annotation_id=f"ann_stim_intensity_{modality}_{epoch_idx:04d}",
                                name="stim_intensity_ma",
                                numeric_value=float(stim_params["stim_intensity_ma"]),
                            ))

                        if "stim_tissue" in stim_params:
                            annotations.append(CategoricalAnnotation(
                                annotation_id=f"ann_stim_tissue_{modality}_{epoch_idx:04d}",
                                name="stim_tissue",
                                value=stim_params["stim_tissue"],
                            ))

                    # Modality annotation
                    annotations.append(CategoricalAnnotation(
                        annotation_id=f"ann_modality_{modality}_{epoch_idx:04d}",
                        name="modality",
                        value=modality,
                    ))

                    # Compute atom ID
                    atom_id = compute_atom_id(
                        dataset_id=dataset_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        run_id=run_id,
                        onset_sample=epoch_idx * n_samples,
                    )

                    # Temporal info (epoch-relative)
                    zero_time = float(epoch_meta[epoch_idx].get("zero_time", 0.3)) if epoch_idx < len(epoch_meta) else 0.3
                    onset_sample = epoch_idx * n_samples

                    temporal = TemporalInfo(
                        onset_sample=onset_sample,
                        onset_seconds=onset_sample / srate,
                        duration_samples=n_samples,
                        duration_seconds=n_samples / srate,
                    )
                    if baseline_samples:
                        temporal.baseline_start_sample = baseline_samples[0]
                        temporal.baseline_end_sample = baseline_samples[1]

                    # Build custom_fields with coordinate system info
                    atom_custom: Dict[str, Any] = {}
                    if coord_system_info:
                        cs_name = (
                            coord_system_info.get("EEGCoordinateSystem")
                            or coord_system_info.get("iEEGCoordinateSystem")
                        )
                        cs_units = (
                            coord_system_info.get("EEGCoordinateUnits")
                            or coord_system_info.get("iEEGCoordinateUnits")
                        )
                        if cs_name:
                            atom_custom["coordinate_system"] = cs_name
                        if cs_units:
                            atom_custom["coordinate_units"] = cs_units
                        landmarks = coord_system_info.get("AnatomicalLandmarkCoordinates")
                        if landmarks:
                            atom_custom["anatomical_landmarks"] = landmarks
                    atom_custom["n_electrodes_with_coords"] = n_electrodes_with_coords

                    # Reference type from first channel (uniform within modality)
                    ref_type = ch_infos[0].reference if ch_infos else None
                    if ref_type:
                        atom_custom["reference"] = ref_type

                    # Filter settings from first channel (uniform within modality)
                    if ch_infos and ch_infos[0].custom_fields.get("low_cutoff_hz"):
                        atom_custom["low_cutoff_hz"] = ch_infos[0].custom_fields["low_cutoff_hz"]
                    if ch_infos and ch_infos[0].custom_fields.get("high_cutoff_hz"):
                        atom_custom["high_cutoff_hz"] = ch_infos[0].custom_fields["high_cutoff_hz"]

                    atom = Atom(
                        atom_id=atom_id,
                        atom_type=AtomType.EVENT_EPOCH,
                        dataset_id=dataset_id,
                        subject_id=subject_id,
                        session_id=session_id,
                        run_id=run_id,
                        modality=modality,
                        trial_index=epoch_idx,
                        signal_ref=SignalRef(
                            file_path="__placeholder__",
                            internal_path=f"/atoms/{atom_id}/signal",
                            shape=(len(channel_ids), n_samples),
                        ),
                        temporal=temporal,
                        channel_ids=channel_ids,
                        n_channels=len(channel_ids),
                        sampling_rate=srate,
                        annotations=annotations,
                        quality=quality,
                        processing_history=ProcessingHistory(
                            steps=[
                                ProcessingStep(
                                    operation="raw_import",
                                    parameters={
                                        "format": "ccep_bids_npy",
                                        "modality": modality,
                                        "source_file": run_info["epochs_npy"].name,
                                        "epoch_index": epoch_idx,
                                        "baseline_corrected": baseline_info.get("BaselineCorrection", False),
                                        "baseline_method": baseline_info.get("BaselineCorrectionMethod", ""),
                                        "signal_unit": ch_infos[0].unit if ch_infos else "V",
                                    },
                                ),
                            ],
                            is_raw=True,
                            version_tag="raw",
                        ),
                        custom_fields=atom_custom,
                    )

                    # Validate signal
                    warnings = validate_signal(
                        signal=signal.astype(np.float32),
                        atom_id=atom_id,
                        config=self.pool.config.get("import", {}),
                    )
                    all_warnings.extend(warnings)

                    # Write signal to HDF5
                    signal_ref = shard_mgr.write_atom_signal(atom_id, signal)
                    atom.signal_ref = signal_ref

                    # Write JSONL
                    writer.write_atom(atom)
                    atoms.append(atom)

        logger.info(
            "Imported %s %s %s: %d epochs × %d ch × %d samples "
            "(%.1f Hz, %d bad ch, %d with coords, unit=%s, ref=%s)",
            subject_id, run_id, modality,
            n_epochs, len(channel_ids), n_samples, srate, len(bad_channels),
            n_electrodes_with_coords,
            ch_infos[0].unit if ch_infos else "?",
            ch_infos[0].reference if ch_infos else "?",
        )

        return atoms, ch_infos, all_warnings

    # ------------------------------------------------------------------
    # Main entry: import one subject, one or more runs
    # ------------------------------------------------------------------

    def import_subject(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str = "ses-01",
        max_runs: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import all runs for a subject, both EEG and iEEG modalities.

        Each run produces two ImportResult objects (one per modality).
        Cross-modal AtomRelation links are added between matched runs.

        Args:
            subject_dir: Path to the subject directory under derivatives/epochs/
            subject_id: Subject identifier (e.g. 'sub-01')
            session_id: Session identifier
            max_runs: Max number of runs to import (for testing)
            max_epochs: Max epochs per modality per run (for testing)

        Returns:
            List of ImportResult, two per run (eeg + ieeg)
        """
        subject_dir = Path(subject_dir)
        dataset_id = self.task_config.dataset_id

        # Ensure hierarchy
        self.pool.ensure_dataset(dataset_id, name=self.task_config.dataset_name)
        self.pool.ensure_subject(dataset_id, subject_id)

        # Discover runs for both modalities
        eeg_runs = self._discover_runs(subject_dir, "eeg")
        ieeg_runs = self._discover_runs(subject_dir, "ieeg")

        if max_runs is not None:
            eeg_runs = eeg_runs[:max_runs]
            ieeg_runs = ieeg_runs[:max_runs]

        # Build run number → run info maps for cross-modal matching
        eeg_by_num = {r["run_num"]: r for r in eeg_runs}
        ieeg_by_num = {r["run_num"]: r for r in ieeg_runs}
        all_run_nums = sorted(set(eeg_by_num.keys()) | set(ieeg_by_num.keys()))

        results = []

        for run_num in all_run_nums:
            eeg_atoms = []
            ieeg_atoms = []
            eeg_ch_infos = []
            ieeg_ch_infos = []

            # Import EEG epochs for this run
            if run_num in eeg_by_num:
                eeg_atoms, eeg_ch_infos, eeg_warns = self._import_modality_run(
                    eeg_by_num[run_num], "eeg",
                    dataset_id, subject_id, session_id, max_epochs,
                )
                run_meta_eeg = RunMeta(
                    run_id=f"run-{run_num:02d}_eeg",
                    session_id=session_id,
                    subject_id=subject_id,
                    dataset_id=dataset_id,
                    run_index=run_num,
                    task_type=self.task_config.task_type,
                    n_trials=len(eeg_atoms),
                )
                self.pool.register_run(run_meta_eeg)
                results.append(ImportResult(
                    atoms=eeg_atoms,
                    run_meta=run_meta_eeg,
                    channel_infos=eeg_ch_infos,
                    warnings=eeg_warns,
                ))

            # Import iEEG epochs for this run
            if run_num in ieeg_by_num:
                ieeg_atoms, ieeg_ch_infos, ieeg_warns = self._import_modality_run(
                    ieeg_by_num[run_num], "ieeg",
                    dataset_id, subject_id, session_id, max_epochs,
                )
                run_meta_ieeg = RunMeta(
                    run_id=f"run-{run_num:02d}_ieeg",
                    session_id=session_id,
                    subject_id=subject_id,
                    dataset_id=dataset_id,
                    run_index=run_num,
                    task_type=self.task_config.task_type,
                    n_trials=len(ieeg_atoms),
                )
                self.pool.register_run(run_meta_ieeg)
                results.append(ImportResult(
                    atoms=ieeg_atoms,
                    run_meta=run_meta_ieeg,
                    channel_infos=ieeg_ch_infos,
                    warnings=ieeg_warns,
                ))

            # Cross-modal linking: each EEG atom gets a relation to the iEEG run
            if eeg_atoms and ieeg_atoms:
                # Link representative atoms (first in each set)
                stim_desc = ""
                if eeg_atoms[0].annotations:
                    for ann in eeg_atoms[0].annotations:
                        if ann.name == "trial_type":
                            stim_desc = ann.text_value
                            break

                for eeg_atom in eeg_atoms:
                    eeg_atom.relations.append(AtomRelation(
                        target_atom_id=ieeg_atoms[0].atom_id,
                        relation_type="cross_modal_paired_run",
                        metadata={
                            "target_modality": "ieeg",
                            "run_num": run_num,
                            "stim_description": stim_desc,
                            "n_target_epochs": len(ieeg_atoms),
                        },
                    ))
                for ieeg_atom in ieeg_atoms:
                    ieeg_atom.relations.append(AtomRelation(
                        target_atom_id=eeg_atoms[0].atom_id,
                        relation_type="cross_modal_paired_run",
                        metadata={
                            "target_modality": "eeg",
                            "run_num": run_num,
                            "stim_description": stim_desc,
                            "n_target_epochs": len(eeg_atoms),
                        },
                    ))

                logger.info(
                    "Linked run-%02d: %d EEG epochs ↔ %d iEEG epochs (stim: %s)",
                    run_num, len(eeg_atoms), len(ieeg_atoms),
                    stim_desc[:50] if stim_desc else "?",
                )

        logger.info(
            "Subject %s: imported %d runs total (%d results, %d total atoms).",
            subject_id, len(all_run_nums), len(results),
            sum(len(r.atoms) for r in results),
        )

        return results


# Auto-register
register_importer("ccep_bids_npy", CCEPImporter)
