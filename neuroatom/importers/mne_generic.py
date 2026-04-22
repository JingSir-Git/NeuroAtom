"""MNE Generic Importer: handles any format supported by MNE-Python.

Covers: .edf, .bdf, .gdf, .fif, .vhdr (BrainVision), .cnt, and more.
Uses mne.io.read_raw() as a universal loader.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the MNE generic importer")

from neuroatom.core.channel import ChannelInfo
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import ChannelStatus, ChannelType
from neuroatom.importers.base import BaseImporter, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name

logger = logging.getLogger(__name__)

# MNE channel type mapping
_MNE_TYPE_MAP: Dict[str, ChannelType] = {
    "eeg": ChannelType.EEG,
    "eog": ChannelType.EOG,
    "emg": ChannelType.EMG,
    "ecg": ChannelType.ECG,
    "stim": ChannelType.STIM,
    "misc": ChannelType.MISC,
    "ref_meg": ChannelType.REF,
}

# MNE-supported file extensions
_MNE_EXTENSIONS = {
    ".edf", ".bdf", ".gdf", ".fif", ".fif.gz",
    ".vhdr", ".vmrk", ".eeg",  # BrainVision
    ".cnt",  # Neuroscan
    ".set",  # EEGLAB
    ".mff",  # EGI
    ".nxe",
}


class MNEGenericImporter(BaseImporter):
    """Generic importer using MNE-Python as the loading backend."""

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if the file has an MNE-supported extension."""
        path = Path(path)
        if path.is_dir():
            # Check for .mff directories
            return path.suffix.lower() == ".mff"
        suffix = path.suffix.lower()
        # Handle compound extensions like .fif.gz
        if suffix == ".gz" and path.stem.endswith(".fif"):
            return True
        return suffix in _MNE_EXTENSIONS

    def load_raw(self, path: Path) -> Tuple[mne.io.Raw, Dict[str, Any]]:
        """Load raw data using MNE's universal reader.

        MNE internally converts all data to SI units (Volts for EEG).
        """
        path = Path(path)
        logger.info("Loading via MNE: %s", path)

        raw = mne.io.read_raw(path, preload=True, verbose="WARNING")

        # Extract extra metadata from MNE info
        info = raw.info
        extra_meta: Dict[str, Any] = {
            "mne_info": {
                "sfreq": info["sfreq"],
                "highpass": info.get("highpass", 0.0),
                "lowpass": info.get("lowpass", 0.0),
                "meas_date": str(info.get("meas_date", "")),
                "n_channels": len(info["ch_names"]),
            },
        }

        # Apply task config overrides: signal unit
        if self.task_config.signal_unit:
            extra_meta["declared_unit"] = self.task_config.signal_unit
        else:
            # MNE loads data in SI (Volts)
            extra_meta["declared_unit"] = "V"

        return raw, extra_meta

    def extract_channel_infos(self, raw: mne.io.Raw) -> List[ChannelInfo]:
        """Extract per-channel metadata from MNE Raw."""
        info = raw.info
        channel_infos = []

        # Get montage positions if available
        montage = raw.get_montage()
        positions = {}
        if montage is not None:
            try:
                pos_dict = montage.get_positions()
                ch_pos = pos_dict.get("ch_pos", {})
                coord_frame = pos_dict.get("coord_frame", "unknown")
                for ch_name, pos in ch_pos.items():
                    positions[ch_name] = ElectrodeLocation(
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=float(pos[2]),
                        coordinate_system=coord_frame,
                        coordinate_units="m",
                    )
            except Exception as e:
                logger.warning("Could not extract montage positions: %s", e)

        # Determine signal unit
        declared_unit = "V"  # MNE default
        if self.task_config.signal_unit:
            declared_unit = self.task_config.signal_unit

        # Channels excluded by task config
        exclude_set = set(self.task_config.exclude_channels)
        type_overrides = self.task_config.channel_type_overrides

        for idx, ch_name in enumerate(info["ch_names"]):
            if ch_name in exclude_set:
                continue

            # MNE channel type
            mne_type = mne.channel_type(info, idx)
            if ch_name in type_overrides:
                ch_type = ChannelType(type_overrides[ch_name])
            else:
                ch_type = _MNE_TYPE_MAP.get(mne_type, ChannelType.OTHER)

            # Standardize name
            std_name = standardize_channel_name(ch_name)

            # Determine unit
            unit = declared_unit
            if ch_type in (ChannelType.STIM, ChannelType.TRIGGER):
                unit = "n/a"

            ch_info = ChannelInfo(
                channel_id=f"ch_{idx:03d}",
                index=idx,
                name=ch_name,
                standard_name=std_name,
                type=ch_type,
                unit=unit,
                sampling_rate=info["sfreq"],
                location=positions.get(ch_name),
                status=ChannelStatus.UNKNOWN,
            )
            channel_infos.append(ch_info)

        return channel_infos

    def extract_events(self, raw: mne.io.Raw) -> Optional[np.ndarray]:
        """Extract events from MNE Raw.

        Tries stim channel first, then MNE annotations.
        """
        # Strategy 1: stim channel events
        try:
            events = mne.find_events(raw, verbose="WARNING")
            if events is not None and len(events) > 0:
                logger.info("Found %d events from stim channel.", len(events))
                return events
        except (ValueError, RuntimeError):
            pass

        # Strategy 2: MNE annotations → events
        if raw.annotations and len(raw.annotations) > 0:
            try:
                events, event_id = mne.events_from_annotations(
                    raw, verbose="WARNING"
                )
                if len(events) > 0:
                    logger.info(
                        "Found %d events from annotations. Event IDs: %s",
                        len(events), event_id,
                    )
                    return events
            except Exception as e:
                logger.warning("Could not extract events from annotations: %s", e)

        logger.info("No events found in %s", raw.filenames)
        return None


# Auto-register
register_importer("mne_generic", MNEGenericImporter)
