"""QualityInfo: quality assessment metadata for an atom."""

from typing import List, Optional

from pydantic import BaseModel, Field

from neuroatom.core.enums import QualityStatus


class QualityInfo(BaseModel):
    """Quality assessment information for a single atom.

    All fields are nullable — datasets that do not provide quality info
    leave fields as None. Import-time validation populates what it can.
    """

    overall_status: QualityStatus = Field(
        default=QualityStatus.UNKNOWN,
        description="Overall quality verdict for this atom.",
    )
    snr_db: Optional[float] = Field(
        default=None,
        description="Estimated signal-to-noise ratio in decibels.",
    )
    artifact_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of samples flagged as artifact [0, 1].",
    )
    bad_channels: List[str] = Field(
        default_factory=list,
        description="List of channel IDs marked as bad in this atom.",
    )
    max_amplitude_uv: Optional[float] = Field(
        default=None,
        description="Maximum absolute amplitude observed (in µV after unit standardization).",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason for rejection, if applicable.",
    )
    auto_qc_passed: Optional[bool] = Field(
        default=None,
        description="Result of automatic quality check during import.",
    )
