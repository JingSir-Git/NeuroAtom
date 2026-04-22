"""Atom: the fundamental information-complete unit of EEG data.

An Atom is the smallest data unit that preserves temporal-spatial-label
coherence. For trial-based tasks, one atom = one trial across all channels.
For event-related tasks, one atom = one event-locked epoch. For continuous
tasks, one atom = one sliding window segment.

Each atom is self-contained: it carries its own signal reference, temporal
context, channel snapshot, annotations (labels), quality info, processing
provenance, and inter-atom relationships.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from neuroatom.core.annotation import AnnotationUnion
from neuroatom.core.enums import AtomType
from neuroatom.core.provenance import ProcessingHistory
from neuroatom.core.quality import QualityInfo
from neuroatom.core.signal_ref import SignalRef


class TemporalInfo(BaseModel):
    """Temporal context of an atom within its source recording.

    All sample indices are absolute within the original recording (run).
    Seconds are relative to the recording start. Baseline fields are
    only populated for epoched data where a baseline period is included.
    """

    onset_sample: int = Field(
        ...,
        ge=0,
        description="Absolute sample index of atom start in the original recording.",
    )
    onset_seconds: float = Field(
        ...,
        ge=0,
        description="Onset time in seconds from recording start.",
    )
    duration_samples: int = Field(
        ...,
        gt=0,
        description="Number of samples in this atom.",
    )
    duration_seconds: float = Field(
        ...,
        gt=0,
        description="Duration in seconds.",
    )
    baseline_start_sample: Optional[int] = Field(
        default=None,
        ge=0,
        description="Baseline period start sample (inclusive), relative to atom start. None if no baseline.",
    )
    baseline_end_sample: Optional[int] = Field(
        default=None,
        ge=0,
        description="Baseline period end sample (exclusive), relative to atom start.",
    )
    global_timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of atom onset, if available from the recording.",
    )


class AtomRelation(BaseModel):
    """Relationship between two atoms.

    Encodes sequential order (trial ordering within a run), parent-child
    (a window derived from a continuous segment), pairing (EEG atom paired
    with its audio stimulus atom), and overlap (adjacent sliding windows).
    """

    target_atom_id: str = Field(
        ...,
        description="ID of the related atom.",
    )
    relation_type: str = Field(
        ...,
        description=(
            "Relationship type. Standard values: "
            "'sequential_next', 'sequential_prev', "
            "'parent', 'child', "
            "'paired', 'overlapping'."
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Relation-specific metadata. "
            "For 'overlapping': {'overlap_samples': int, 'overlap_ratio': float, 'overlap_seconds': float}. "
            "For 'paired': {'role': 'attended_audio'}. "
            "For 'sequential_next'/'sequential_prev': {'order_index': int}."
        ),
    )


class Atom(BaseModel):
    """The fundamental information-complete unit of EEG data.

    An atom is self-contained: given a pool_root path, all information
    needed to load, interpret, and use the signal is embedded in or
    referenced from this model.
    """

    atom_id: str = Field(
        ...,
        description=(
            "Content-addressable unique ID: SHA-256 of "
            "(dataset_id + subject_id + session_id + run_id + onset_sample + processing_hash)."
        ),
    )
    atom_type: AtomType = Field(
        ...,
        description="Type of atomic unit.",
    )

    # ---- Provenance: where this atom came from ----
    dataset_id: str = Field(
        ...,
        description="ID of the source dataset.",
    )
    subject_id: str = Field(
        ...,
        description="ID of the source subject.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID of the recording session, if applicable.",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="ID of the recording run within the session.",
    )
    modality: Optional[str] = Field(
        default=None,
        description=(
            "Recording modality for multi-modal datasets: "
            "'eeg', 'ieeg', 'seeg', 'ecog', 'meg', 'fnirs', etc. "
            "None for single-modality datasets."
        ),
    )
    trial_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Sequential trial index within the run (for ordering). None for non-trial atoms.",
    )

    # ---- Signal reference ----
    signal_ref: SignalRef = Field(
        ...,
        description="Pointer to the signal data in an HDF5 shard.",
    )

    # ---- Temporal context ----
    temporal: TemporalInfo = Field(
        ...,
        description="Temporal position and duration within the source recording.",
    )

    # ---- Channel snapshot ----
    channel_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of channel IDs present in this atom's signal array.",
    )
    n_channels: int = Field(
        ...,
        gt=0,
        description="Number of channels (must equal len(channel_ids)).",
    )
    sampling_rate: float = Field(
        ...,
        gt=0,
        description="Sampling rate of the signal in Hz.",
    )

    # ---- Annotations (discriminated union) ----
    annotations: List[AnnotationUnion] = Field(
        default_factory=list,
        description="All labels, events, stimuli references, and quality markers for this atom.",
    )

    # ---- Quality ----
    quality: Optional[QualityInfo] = Field(
        default=None,
        description="Quality assessment of this atom's signal.",
    )

    # ---- Processing provenance ----
    processing_history: ProcessingHistory = Field(
        default_factory=ProcessingHistory,
        description="Ordered chain of all processing steps applied to produce this atom's signal.",
    )

    # ---- Inter-atom relations ----
    relations: List[AtomRelation] = Field(
        default_factory=list,
        description="Relationships to other atoms (sequential, paired, overlapping, etc.).",
    )

    # ---- Extensible ----
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs for dataset-specific atom metadata.",
    )
