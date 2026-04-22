"""Annotation system: discriminated union of 7 annotation subtypes.

This is the most critical design element of NeuroAtom. Instead of fixed
label fields, we use a Pydantic v2 discriminated union that can represent:

- Simple class labels (MI: "left_hand")
- Continuous time-aligned signals (audio envelope for AAD)
- Event sequences (word onsets for language decoding)
- Stimulus references (which audio/image was presented)
- Numeric scores (valence=0.7)
- Free text (spoken sentence)
- Binary masks (artifact regions)

Each subtype carries only its own required fields — no wasted Optional fields.
Serialization/deserialization auto-dispatches on ``annotation_type``.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Discriminator, Field, Tag

from neuroatom.core.signal_ref import SignalRef


# ---------------------------------------------------------------------------
# Event item (used within EventSequenceAnnotation)
# ---------------------------------------------------------------------------

class EventItem(BaseModel):
    """A single discrete event within an event sequence annotation.

    Used for word onsets, phoneme boundaries, stimulus markers, etc.
    Temporal values are in seconds relative to the atom start.
    """

    onset: float = Field(
        ...,
        description="Onset time in seconds relative to atom start. Sub-sample precision (float).",
    )
    duration: Optional[float] = Field(
        default=None,
        ge=0,
        description="Duration in seconds. None for instantaneous events.",
    )
    value: str = Field(
        ...,
        description="Event label or content (e.g. 'the', 'left_hand', 'target').",
    )
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary features attached to this event (e.g. word embedding, frequency).",
    )


# ---------------------------------------------------------------------------
# Base annotation (shared fields)
# ---------------------------------------------------------------------------

class _BaseAnnotation(BaseModel):
    """Shared fields for all annotation subtypes."""

    annotation_id: str = Field(
        ...,
        description="Unique identifier for this annotation instance.",
    )
    name: str = Field(
        ...,
        description=(
            "Semantic name of the annotation. Examples: 'mi_class', "
            "'attended_speaker', 'word_onsets', 'artifact_mask', 'valence'."
        ),
    )
    domain: str = Field(
        default="task_label",
        description=(
            "Semantic domain. Standard values: 'task_label', 'stimulus', "
            "'quality', 'physiological', 'demographic'."
        ),
    )
    scope: Literal["atom", "timepoint"] = Field(
        default="atom",
        description=(
            "'atom' = annotation applies to the entire atom. "
            "'timepoint' = annotation is per-sample aligned."
        ),
    )
    onset: Optional[float] = Field(
        default=None,
        description="Onset in seconds relative to atom start. Sub-sample precision.",
    )
    duration: Optional[float] = Field(
        default=None,
        ge=0,
        description="Duration in seconds.",
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs.",
    )


# ---------------------------------------------------------------------------
# 7 annotation subtypes
# ---------------------------------------------------------------------------

class CategoricalAnnotation(_BaseAnnotation):
    """Single categorical label for the whole atom (or a time segment).

    Examples: MI class ("left_hand"), attended ear ("left"), emotion ("happy").
    """

    annotation_type: Literal["categorical"] = "categorical"
    value: str = Field(
        ...,
        description="Categorical label value.",
    )
    encoding: Optional[Dict[str, int]] = Field(
        default=None,
        description="Label-to-integer encoding map, e.g. {'left_hand': 0, 'right_hand': 1}.",
    )


class NumericAnnotation(_BaseAnnotation):
    """Single numeric value for the whole atom (or a time segment).

    Examples: valence=0.7, arousal=0.3, reaction_time=1.23.
    """

    annotation_type: Literal["numeric"] = "numeric"
    numeric_value: float = Field(
        ...,
        description="Numeric label value.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of the numeric value, e.g. 'score', 'seconds', 'Hz'.",
    )


class TextAnnotation(_BaseAnnotation):
    """Free text annotation.

    Examples: spoken sentence, task instruction, clinical note.
    """

    annotation_type: Literal["text"] = "text"
    text_value: str = Field(
        ...,
        description="Text content.",
    )


class ContinuousAnnotation(_BaseAnnotation):
    """Time-aligned continuous signal stored alongside the EEG.

    Examples: audio envelope, EMG trace, eye-tracking signal, mel spectrogram.
    The data is stored in the same HDF5 shard as the atom signal, referenced
    by ``data_ref``.
    """

    annotation_type: Literal["continuous"] = "continuous"
    data_ref: SignalRef = Field(
        ...,
        description="Reference to the continuous data array in HDF5.",
    )
    data_sampling_rate: float = Field(
        ...,
        gt=0,
        description="Sampling rate of the continuous annotation data in Hz.",
    )
    alignment_method: Optional[str] = Field(
        default=None,
        description=(
            "How this signal was aligned to the EEG. "
            "Values: 'sample_aligned', 'interpolated', 'trigger_locked'."
        ),
    )


class EventSequenceAnnotation(_BaseAnnotation):
    """Ordered sequence of discrete timed events.

    Examples: word onsets with text, phoneme boundaries, stimulus event markers.
    """

    annotation_type: Literal["event_sequence"] = "event_sequence"
    events: List[EventItem] = Field(
        default_factory=list,
        description="List of events, ordered by onset time.",
    )


class StimulusRefAnnotation(_BaseAnnotation):
    """Reference to an external stimulus resource.

    Examples: which audio file was played, which image was shown.
    """

    annotation_type: Literal["stimulus_ref"] = "stimulus_ref"
    stimulus_id: str = Field(
        ...,
        description="ID of the StimulusResource in the pool's stimuli/ directory.",
    )
    stimulus_onset: Optional[float] = Field(
        default=None,
        description="Stimulus onset time relative to atom start, in seconds.",
    )
    stimulus_offset: Optional[float] = Field(
        default=None,
        description="Stimulus offset time relative to atom start, in seconds.",
    )


class BinaryMaskAnnotation(_BaseAnnotation):
    """Per-sample binary mask stored in HDF5.

    Examples: artifact mask, speech activity mask, attention mask.
    """

    annotation_type: Literal["binary_mask"] = "binary_mask"
    mask_ref: SignalRef = Field(
        ...,
        description="Reference to the binary mask array in HDF5 (dtype typically uint8 or bool).",
    )


# ---------------------------------------------------------------------------
# Discriminated union type
# ---------------------------------------------------------------------------

AnnotationUnion = Annotated[
    Union[
        Annotated[CategoricalAnnotation, Tag("categorical")],
        Annotated[NumericAnnotation, Tag("numeric")],
        Annotated[TextAnnotation, Tag("text")],
        Annotated[ContinuousAnnotation, Tag("continuous")],
        Annotated[EventSequenceAnnotation, Tag("event_sequence")],
        Annotated[StimulusRefAnnotation, Tag("stimulus_ref")],
        Annotated[BinaryMaskAnnotation, Tag("binary_mask")],
    ],
    Discriminator("annotation_type"),
]
"""Union type for all annotation subtypes, discriminated on ``annotation_type``."""
