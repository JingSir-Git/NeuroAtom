"""StimulusResource: external stimulus referenced by annotations."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from neuroatom.core.annotation import EventItem


class StimulusResource(BaseModel):
    """An external stimulus resource (audio, image, text, video) stored in the pool.

    Stimuli are stored once in ``pool_root/stimuli/{stimulus_id}/`` and
    referenced by ``StimulusRefAnnotation`` from individual atoms.
    Deduplication across datasets uses ``content_hash``.
    """

    stimulus_id: str = Field(
        ...,
        description="Unique stimulus identifier. Typically a content hash of the file.",
    )
    stimulus_type: str = Field(
        ...,
        description="Stimulus modality: 'audio', 'image', 'text', 'video', 'cue'.",
    )
    description: Optional[str] = Field(default=None, description="Human-readable description.")
    file_path: Optional[str] = Field(
        default=None,
        description="Path relative to pool_root/stimuli/ where the resource file is stored.",
    )
    duration: Optional[float] = Field(
        default=None,
        gt=0,
        description="Duration in seconds (for audio/video).",
    )
    sampling_rate: Optional[float] = Field(
        default=None,
        gt=0,
        description="Sampling rate in Hz (for audio/video).",
    )
    transcript: Optional[str] = Field(
        default=None,
        description="Text transcript (for speech stimuli).",
    )
    word_timestamps: Optional[List[EventItem]] = Field(
        default=None,
        description="Word-level timestamps within the stimulus.",
    )
    language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code (e.g. 'en', 'zh').",
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the stimulus file content, for deduplication across datasets.",
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs.",
    )
