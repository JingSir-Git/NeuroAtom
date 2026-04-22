"""WindowAtomizer: sliding window decomposition for continuous paradigms.

For tasks without discrete trial boundaries (AAD, language decoding,
sleep staging, seizure detection). Decomposes a continuous recording
into overlapping fixed-length windows.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from neuroatom.atomizer.base import BaseAtomizer
from neuroatom.core.annotation import CategoricalAnnotation, EventItem, EventSequenceAnnotation
from neuroatom.core.atom import Atom, AtomRelation, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.enums import AtomType
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.run import RunMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.importers.base import TaskConfig
from neuroatom.utils.hashing import compute_atom_id

logger = logging.getLogger(__name__)


class WindowAtomizer(BaseAtomizer):
    """Sliding window atomizer for continuous paradigms.

    Trial definition from task config:
        trial_definition:
            mode: "window"
            window_seconds: 5.0
            step_seconds: 2.5              # 50% overlap
            annotation_boundary: "include_if_onset"

    Annotation boundary strategies:
        - "include_if_onset": discrete event included if onset falls within window
        - "include_if_complete": event included only if fully within window
        - "proportional": include if overlap_ratio > threshold (default 0.5)
    """

    def atomize(
        self,
        raw: Any,
        events: Optional[np.ndarray],
        task_config: TaskConfig,
        run_meta: RunMeta,
        channel_infos: List[ChannelInfo],
    ) -> List[Atom]:
        trial_def = task_config.trial_definition
        window_seconds = trial_def.get("window_seconds", 5.0)
        step_seconds = trial_def.get("step_seconds", window_seconds)
        annotation_boundary = trial_def.get("annotation_boundary", "include_if_onset")

        sfreq = raw.info["sfreq"]
        n_total_samples = raw.n_times
        window_samples = int(window_seconds * sfreq)
        step_samples = int(step_seconds * sfreq)

        if window_samples <= 0 or step_samples <= 0:
            logger.error("Invalid window/step: %d/%d samples", window_samples, step_samples)
            return []

        channel_ids = [ch.channel_id for ch in channel_infos]

        # Event mapping for discrete annotations
        event_label_map = {}
        for k, v in task_config.event_mapping.items():
            event_label_map[int(k)] = str(v)

        # Generate windows
        atoms: List[Atom] = []
        window_idx = 0
        onset_sample = 0

        while onset_sample + window_samples <= n_total_samples:
            offset_sample = onset_sample + window_samples

            # Collect events within this window
            window_annotations = []
            if events is not None and len(events) > 0:
                window_events = self._filter_events_in_window(
                    events=events,
                    onset_sample=onset_sample,
                    offset_sample=offset_sample,
                    strategy=annotation_boundary,
                    sfreq=sfreq,
                    event_label_map=event_label_map,
                )
                if window_events:
                    # Store as EventSequenceAnnotation
                    event_items = [
                        EventItem(
                            onset=(ev_sample - onset_sample) / sfreq,
                            value=ev_label,
                        )
                        for ev_sample, ev_label in window_events
                    ]
                    ann = EventSequenceAnnotation(
                        annotation_id=f"ann_window_events_{window_idx:06d}",
                        name="window_events",
                        events=event_items,
                    )
                    window_annotations.append(ann)

            # Compute atom ID
            atom_id = compute_atom_id(
                dataset_id=run_meta.dataset_id,
                subject_id=run_meta.subject_id,
                session_id=run_meta.session_id,
                run_id=run_meta.run_id,
                onset_sample=onset_sample,
            )

            atom = Atom(
                atom_id=atom_id,
                atom_type=AtomType.WINDOW,
                dataset_id=run_meta.dataset_id,
                subject_id=run_meta.subject_id,
                session_id=run_meta.session_id,
                run_id=run_meta.run_id,
                trial_index=None,
                signal_ref=SignalRef(
                    file_path="__placeholder__",
                    internal_path=f"/atoms/{atom_id}/signal",
                    shape=(len(channel_ids), window_samples),
                ),
                temporal=TemporalInfo(
                    onset_sample=onset_sample,
                    onset_seconds=onset_sample / sfreq,
                    duration_samples=window_samples,
                    duration_seconds=window_seconds,
                ),
                channel_ids=channel_ids,
                n_channels=len(channel_ids),
                sampling_rate=sfreq,
                annotations=window_annotations,
                processing_history=ProcessingHistory(
                    steps=[
                        ProcessingStep(
                            operation="raw_import",
                            parameters={
                                "format": "mne",
                                "window_seconds": window_seconds,
                                "step_seconds": step_seconds,
                            },
                        ),
                    ],
                    is_raw=True,
                    version_tag="raw",
                ),
            )

            # Overlap relations with previous window
            if atoms and step_samples < window_samples:
                prev_atom = atoms[-1]
                overlap_samples = window_samples - step_samples
                overlap_ratio = overlap_samples / window_samples
                overlap_seconds = overlap_samples / sfreq

                atom.relations.append(
                    AtomRelation(
                        target_atom_id=prev_atom.atom_id,
                        relation_type="overlapping",
                        metadata={
                            "overlap_samples": overlap_samples,
                            "overlap_ratio": round(overlap_ratio, 4),
                            "overlap_seconds": round(overlap_seconds, 4),
                        },
                    )
                )
                prev_atom.relations.append(
                    AtomRelation(
                        target_atom_id=atom.atom_id,
                        relation_type="overlapping",
                        metadata={
                            "overlap_samples": overlap_samples,
                            "overlap_ratio": round(overlap_ratio, 4),
                            "overlap_seconds": round(overlap_seconds, 4),
                        },
                    )
                )

            # Sequential relations
            if atoms:
                prev_atom = atoms[-1]
                atom.relations.append(
                    AtomRelation(
                        target_atom_id=prev_atom.atom_id,
                        relation_type="sequential_prev",
                    )
                )
                prev_atom.relations.append(
                    AtomRelation(
                        target_atom_id=atom.atom_id,
                        relation_type="sequential_next",
                    )
                )

            atoms.append(atom)
            window_idx += 1
            onset_sample += step_samples

        logger.info(
            "WindowAtomizer produced %d atoms (window=%.1fs, step=%.1fs, overlap=%.1f%%).",
            len(atoms),
            window_seconds,
            step_seconds,
            100.0 * (1.0 - step_seconds / window_seconds) if window_seconds > 0 else 0,
        )
        return atoms

    # ------------------------------------------------------------------
    # Event filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_events_in_window(
        events: np.ndarray,
        onset_sample: int,
        offset_sample: int,
        strategy: str,
        sfreq: float,
        event_label_map: Dict[int, str],
    ) -> List[tuple]:
        """Filter events that belong to this window, returning (sample, label) pairs."""
        results = []

        for event in events:
            ev_sample = int(event[0])
            ev_id = int(event[2])
            label = event_label_map.get(ev_id, str(ev_id))

            if strategy == "include_if_onset":
                if onset_sample <= ev_sample < offset_sample:
                    results.append((ev_sample, label))

            elif strategy == "include_if_complete":
                # For instantaneous events, same as include_if_onset
                if onset_sample <= ev_sample < offset_sample:
                    results.append((ev_sample, label))

            elif strategy == "proportional":
                # For instantaneous events, check if within window
                if onset_sample <= ev_sample < offset_sample:
                    results.append((ev_sample, label))

        return results
