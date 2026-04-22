"""TrialAtomizer: decompose recordings into trial-based atoms.

For task paradigms where each event marker defines the onset of a trial
(e.g., MI, P300, SSVEP). The trial window is defined by tmin/tmax
relative to the event onset.
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
from neuroatom.core.quality import QualityInfo
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.importers.base import TaskConfig
from neuroatom.utils.hashing import compute_atom_id

logger = logging.getLogger(__name__)


class TrialAtomizer(BaseAtomizer):
    """Decomposes a recording into trial-based atoms using event markers.

    Trial definition from task config:
        trial_definition:
            mode: "trial"
            anchor_events: [769, 770, 771, 772]
            tmin: 0.0        # seconds before event (usually 0 or negative)
            tmax: 4.0        # seconds after event
            baseline_tmin: -0.5  # optional baseline window
            baseline_tmax: 0.0
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
            logger.warning("No events provided for TrialAtomizer.")
            return []

        trial_def = task_config.trial_definition
        anchor_events = set(trial_def.get("anchor_events", []))
        tmin = trial_def.get("tmin", 0.0)
        tmax = trial_def.get("tmax", 4.0)
        baseline_tmin = trial_def.get("baseline_tmin", None)
        baseline_tmax = trial_def.get("baseline_tmax", None)

        # Event ID → label name mapping
        event_mapping = task_config.event_mapping  # e.g., {769: "left_hand", 770: "right_hand"}
        # Convert string keys to int if needed
        event_label_map = {}
        for k, v in event_mapping.items():
            event_label_map[int(k)] = str(v)

        sfreq = raw.info["sfreq"]
        n_total_samples = raw.n_times
        channel_ids = [ch.channel_id for ch in channel_infos]

        atoms: List[Atom] = []
        trial_index = 0

        for event_idx, event in enumerate(events):
            event_sample = int(event[0])
            event_id = int(event[2])

            # Filter by anchor events
            if anchor_events and event_id not in anchor_events:
                continue

            # Compute sample window
            onset_sample = int(event_sample + tmin * sfreq)
            offset_sample = int(event_sample + tmax * sfreq)

            # Boundary check
            if onset_sample < 0:
                logger.debug(
                    "Trial %d onset before recording start, clipping.", trial_index
                )
                onset_sample = 0
            if offset_sample > n_total_samples:
                logger.debug(
                    "Trial %d offset after recording end, clipping.", trial_index
                )
                offset_sample = n_total_samples

            duration_samples = offset_sample - onset_sample
            if duration_samples <= 0:
                logger.warning("Trial %d has non-positive duration, skipping.", trial_index)
                continue

            # Baseline samples (relative to atom start)
            baseline_start = None
            baseline_end = None
            if baseline_tmin is not None and baseline_tmax is not None:
                baseline_start = int((baseline_tmin - tmin) * sfreq)
                baseline_end = int((baseline_tmax - tmin) * sfreq)
                baseline_start = max(0, baseline_start)
                baseline_end = max(0, baseline_end)

            # Build annotation
            annotations = []
            label_name = event_label_map.get(event_id)
            if label_name is not None:
                ann = CategoricalAnnotation(
                    annotation_id=f"ann_trial_{trial_index:04d}",
                    name="trial_label",
                    value=label_name,
                )
                annotations.append(ann)

            # Compute atom ID
            atom_id = compute_atom_id(
                dataset_id=run_meta.dataset_id,
                subject_id=run_meta.subject_id,
                session_id=run_meta.session_id,
                run_id=run_meta.run_id,
                onset_sample=onset_sample,
            )

            # Placeholder SignalRef (will be overwritten by importer)
            placeholder_ref = SignalRef(
                file_path="__placeholder__",
                internal_path=f"/atoms/{atom_id}/signal",
                shape=(len(channel_ids), duration_samples),
            )

            atom = Atom(
                atom_id=atom_id,
                atom_type=AtomType.TRIAL,
                dataset_id=run_meta.dataset_id,
                subject_id=run_meta.subject_id,
                session_id=run_meta.session_id,
                run_id=run_meta.run_id,
                trial_index=trial_index,
                signal_ref=placeholder_ref,
                temporal=TemporalInfo(
                    onset_sample=onset_sample,
                    onset_seconds=onset_sample / sfreq,
                    duration_samples=duration_samples,
                    duration_seconds=duration_samples / sfreq,
                    baseline_start_sample=baseline_start,
                    baseline_end_sample=baseline_end,
                ),
                channel_ids=channel_ids,
                n_channels=len(channel_ids),
                sampling_rate=sfreq,
                annotations=annotations,
                processing_history=ProcessingHistory(
                    steps=[
                        ProcessingStep(
                            operation="raw_import",
                            parameters={"format": "mne", "tmin": tmin, "tmax": tmax},
                        ),
                    ],
                    is_raw=True,
                    version_tag="raw",
                ),
            )

            # Add sequential relations
            if atoms:
                prev_atom = atoms[-1]
                atom.relations.append(
                    AtomRelation(
                        target_atom_id=prev_atom.atom_id,
                        relation_type="sequential_prev",
                        metadata={"order_index": trial_index - 1},
                    )
                )
                prev_atom.relations.append(
                    AtomRelation(
                        target_atom_id=atom.atom_id,
                        relation_type="sequential_next",
                        metadata={"order_index": trial_index},
                    )
                )

            atoms.append(atom)
            trial_index += 1

        logger.info(
            "TrialAtomizer produced %d atoms from %d events.",
            len(atoms), len(events),
        )
        return atoms
