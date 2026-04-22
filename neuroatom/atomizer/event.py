"""EventAtomizer: decompose recordings into event-locked epochs.

Similar to TrialAtomizer but supports multiple event types simultaneously,
variable epoch durations, and more flexible annotation assignment.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from neuroatom.atomizer.base import BaseAtomizer
from neuroatom.core.annotation import CategoricalAnnotation
from neuroatom.core.atom import Atom, AtomRelation, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.enums import AtomType
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.importers.base import TaskConfig
from neuroatom.utils.hashing import compute_atom_id

logger = logging.getLogger(__name__)


class EventAtomizer(BaseAtomizer):
    """Decomposes a recording into event-locked epoch atoms.

    Unlike TrialAtomizer, EventAtomizer:
    - Can create atoms for ALL event types (not just anchor events)
    - Supports per-event-type epoch windows
    - Assigns event_id as an annotation

    Trial definition from task config:
        trial_definition:
            mode: "event_epoch"
            event_windows:  # Per event-type windows
                769:
                    tmin: -0.2
                    tmax: 0.8
                    label: "target"
                770:
                    tmin: -0.2
                    tmax: 0.8
                    label: "non_target"
            default_tmin: -0.2
            default_tmax: 0.8
    """

    def atomize(
        self,
        raw: Any,
        events: Optional[np.ndarray],
        task_config: TaskConfig,
        run_meta: RunMeta,
        channel_infos: List[ChannelInfo],
    ) -> List[Atom]:
        if events is None or len(events) == 0:
            logger.warning("No events provided for EventAtomizer.")
            return []

        trial_def = task_config.trial_definition
        event_windows = trial_def.get("event_windows", {})
        default_tmin = trial_def.get("default_tmin", -0.2)
        default_tmax = trial_def.get("default_tmax", 0.8)

        sfreq = raw.info["sfreq"]
        n_total_samples = raw.n_times
        channel_ids = [ch.channel_id for ch in channel_infos]

        atoms: List[Atom] = []

        for event_idx, event in enumerate(events):
            event_sample = int(event[0])
            event_id = int(event[2])

            # Get event-specific or default window
            ew = event_windows.get(str(event_id), event_windows.get(event_id, {}))
            tmin = ew.get("tmin", default_tmin)
            tmax = ew.get("tmax", default_tmax)
            label = ew.get("label", str(event_id))

            # If this event_id is not in event_windows and no default_tmin/tmax,
            # skip unknown events
            if not ew and event_id not in event_windows:
                # Use defaults
                pass

            # Compute sample window
            onset_sample = int(event_sample + tmin * sfreq)
            offset_sample = int(event_sample + tmax * sfreq)

            # Boundary clipping
            onset_sample = max(0, onset_sample)
            offset_sample = min(n_total_samples, offset_sample)
            duration_samples = offset_sample - onset_sample
            if duration_samples <= 0:
                continue

            # Annotation
            ann = CategoricalAnnotation(
                annotation_id=f"ann_event_{event_idx:04d}",
                name="event_label",
                value=label,
            )

            atom_id = compute_atom_id(
                dataset_id=run_meta.dataset_id,
                subject_id=run_meta.subject_id,
                session_id=run_meta.session_id,
                run_id=run_meta.run_id,
                onset_sample=onset_sample,
            )

            atom = Atom(
                atom_id=atom_id,
                atom_type=AtomType.EVENT_EPOCH,
                dataset_id=run_meta.dataset_id,
                subject_id=run_meta.subject_id,
                session_id=run_meta.session_id,
                run_id=run_meta.run_id,
                trial_index=event_idx,
                signal_ref=SignalRef(
                    file_path="__placeholder__",
                    internal_path=f"/atoms/{atom_id}/signal",
                    shape=(len(channel_ids), duration_samples),
                ),
                temporal=TemporalInfo(
                    onset_sample=onset_sample,
                    onset_seconds=onset_sample / sfreq,
                    duration_samples=duration_samples,
                    duration_seconds=duration_samples / sfreq,
                ),
                channel_ids=channel_ids,
                n_channels=len(channel_ids),
                sampling_rate=sfreq,
                annotations=[ann],
                processing_history=ProcessingHistory(
                    steps=[
                        ProcessingStep(
                            operation="raw_import",
                            parameters={
                                "format": "mne",
                                "tmin": tmin,
                                "tmax": tmax,
                                "event_id": event_id,
                            },
                        ),
                    ],
                    is_raw=True,
                    version_tag="raw",
                ),
            )
            atoms.append(atom)

        logger.info(
            "EventAtomizer produced %d atoms from %d events.",
            len(atoms), len(events),
        )
        return atoms
