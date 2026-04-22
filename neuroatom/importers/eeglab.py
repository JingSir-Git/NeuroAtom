"""EEGLAB Importer: handles EEGLAB .set/.fdt format.

EEGLAB (.set) files contain a MATLAB struct with:
- EEG.data: signal matrix or reference to .fdt file
- EEG.srate: sampling rate
- EEG.nbchan: number of channels
- EEG.pnts: number of time points
- EEG.event: event structure
- EEG.chanlocs: channel location structure

MNE can read .set files natively via mne.io.read_raw_eeglab().
This importer wraps MNE's EEGLAB reader with proper event and
channel metadata extraction.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the EEGLAB importer")

from neuroatom.core.channel import ChannelInfo
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import ChannelStatus, ChannelType
from neuroatom.importers.base import BaseImporter, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name

logger = logging.getLogger(__name__)


class EEGLABImporter(BaseImporter):
    """Importer for EEGLAB .set/.fdt format.

    Uses MNE's read_raw_eeglab() for reliable loading, then extracts
    EEGLAB-specific metadata (channel locations, event types) from the
    underlying MATLAB structure.
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if the file is an EEGLAB .set file."""
        path = Path(path)
        return path.is_file() and path.suffix.lower() == ".set"

    def load_raw(self, path: Path) -> Tuple[mne.io.Raw, Dict[str, Any]]:
        """Load .set file via MNE's EEGLAB reader.

        Returns:
            (raw, extra_meta) where raw is mne.io.Raw.
        """
        path = Path(path)
        logger.info("Loading EEGLAB .set file: %s", path)

        raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="WARNING")

        # Try to extract EEGLAB-specific metadata from the .set file
        eeglab_meta = self._extract_eeglab_meta(path)

        extra_meta = {
            "declared_unit": self.task_config.signal_unit or "uV",
            "eeglab_meta": eeglab_meta,
        }

        return raw, extra_meta

    def extract_channel_infos(self, raw: mne.io.Raw) -> List[ChannelInfo]:
        """Extract per-channel metadata from MNE Raw (EEGLAB source)."""
        info = raw.info
        ch_infos = []

        type_overrides = self.task_config.channel_type_overrides
        exclude_set = set(self.task_config.exclude_channels)
        declared_unit = self.task_config.signal_unit or "uV"

        # Get montage if available
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
                logger.warning("Could not extract EEGLAB montage: %s", e)

        for idx, ch_name in enumerate(info["ch_names"]):
            if ch_name in exclude_set:
                continue

            # MNE channel type
            mne_type = mne.channel_type(info, idx)
            if ch_name in type_overrides:
                ch_type = ChannelType(type_overrides[ch_name])
            elif mne_type in ("eeg", "seeg", "ecog"):
                ch_type = ChannelType.EEG
            elif mne_type == "eog":
                ch_type = ChannelType.EOG
            elif mne_type == "emg":
                ch_type = ChannelType.EMG
            elif mne_type == "ecg":
                ch_type = ChannelType.ECG
            elif mne_type == "stim":
                ch_type = ChannelType.STIM
            else:
                ch_type = ChannelType.OTHER

            unit = declared_unit
            if ch_type in (ChannelType.STIM, ChannelType.TRIGGER):
                unit = "n/a"

            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{idx:03d}",
                index=idx,
                name=ch_name,
                standard_name=standardize_channel_name(ch_name),
                type=ch_type,
                unit=unit,
                sampling_rate=info["sfreq"],
                location=positions.get(ch_name),
                status=ChannelStatus.UNKNOWN,
            ))

        return ch_infos

    def extract_events(self, raw: mne.io.Raw) -> Optional[np.ndarray]:
        """Extract events from EEGLAB file via MNE.

        EEGLAB stores events in EEG.event structure which MNE reads
        as annotations. We convert annotations back to events array.
        """
        # MNE converts EEGLAB events to annotations during loading
        if raw.annotations and len(raw.annotations) > 0:
            try:
                events, event_id = mne.events_from_annotations(
                    raw, verbose="WARNING"
                )
                if len(events) > 0:
                    logger.info(
                        "Found %d events from EEGLAB annotations. Types: %s",
                        len(events),
                        list(event_id.keys())[:10],
                    )
                    return events
            except Exception as e:
                logger.warning("Could not extract EEGLAB events: %s", e)

        logger.info("No events found in EEGLAB file.")
        return None

    def _extract_eeglab_meta(self, path: Path) -> Dict[str, Any]:
        """Try to extract extra EEGLAB metadata from the .set file directly.

        This is best-effort; if it fails, we fall back to MNE-only metadata.
        """
        meta = {}
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
            eeg = mat.get("EEG")
            if eeg is not None:
                meta["setname"] = getattr(eeg, "setname", "")
                meta["nbchan"] = int(getattr(eeg, "nbchan", 0))
                meta["pnts"] = int(getattr(eeg, "pnts", 0))
                meta["srate"] = float(getattr(eeg, "srate", 0))
                meta["xmin"] = float(getattr(eeg, "xmin", 0))
                meta["xmax"] = float(getattr(eeg, "xmax", 0))
                ref = getattr(eeg, "ref", None)
                if ref is not None:
                    meta["reference"] = str(ref)
        except Exception as e:
            logger.debug("Could not extract EEGLAB meta: %s", e)

        return meta


# Auto-register
register_importer("eeglab", EEGLABImporter)
