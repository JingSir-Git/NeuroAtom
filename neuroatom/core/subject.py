"""SubjectMeta: per-subject metadata."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class SubjectMeta(BaseModel):
    """Metadata for a single experimental subject.

    All fields beyond IDs are nullable — datasets that do not provide
    demographic info leave fields as None.
    """

    subject_id: str = Field(..., description="Unique subject identifier within the dataset.")
    dataset_id: str = Field(..., description="ID of the parent dataset.")
    age: Optional[float] = Field(default=None, ge=0, description="Age in years.")
    sex: Optional[Literal["M", "F", "O"]] = Field(
        default=None,
        description="Biological sex: M=male, F=female, O=other.",
    )
    handedness: Optional[Literal["R", "L", "A"]] = Field(
        default=None,
        description="Handedness: R=right, L=left, A=ambidextrous.",
    )
    group: Optional[str] = Field(
        default=None,
        description="Subject group: 'patient', 'control', 'healthy', etc.",
    )
    diagnosis: Optional[str] = Field(
        default=None,
        description="Clinical diagnosis if applicable: 'ALS', 'stroke', 'depression', etc.",
    )
    medication: Optional[str] = Field(
        default=None,
        description="Current medication, if relevant and available.",
    )
    native_language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code (e.g. 'zh', 'en'). Important for language decoding tasks.",
    )
    hearing_status: Optional[str] = Field(
        default=None,
        description="Hearing status: 'normal', 'impaired'. Important for AAD tasks.",
    )
    visual_acuity: Optional[str] = Field(
        default=None,
        description="Visual acuity: 'normal', 'corrected'. Important for SSVEP/P300 tasks.",
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs for dataset-specific subject metadata.",
    )
