"""ProcessingHistory: full provenance chain for reproducibility."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessingStep(BaseModel):
    """A single processing operation applied to EEG data.

    Records enough information to reproduce the transformation.
    """

    operation: str = Field(
        ...,
        description=(
            "Operation identifier. Standard values: 'raw_import', 'bandpass_filter', "
            "'notch_filter', 'ica', 'rereference', 'resample', 'baseline_correction', "
            "'artifact_rejection', 'epoch_extraction', 'interpolate_channels'."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation parameters, e.g. {'l_freq': 0.5, 'h_freq': 40, 'method': 'fir'}.",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of when this step was executed.",
    )
    software: Optional[str] = Field(
        default=None,
        description="Software and version, e.g. 'mne==1.6.0', 'neuroatom==0.1.0'.",
    )
    input_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of input data before this step.",
    )
    output_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of output data after this step.",
    )


class ProcessingHistory(BaseModel):
    """Ordered sequence of processing steps applied to produce an atom's signal.

    Enables full reproducibility: given the original recording and this history,
    the exact atom signal can be reconstructed.
    """

    steps: List[ProcessingStep] = Field(
        default_factory=list,
        description="Ordered list of processing steps, earliest first.",
    )
    is_raw: bool = Field(
        default=True,
        description="True if no processing beyond import has been applied.",
    )
    version_tag: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable version tag for this processing state. "
            "Examples: 'raw', 'filtered_0.5_40', 'ica_cleaned'. "
            "Used by AssemblyRecipe.source_version to select which version to use."
        ),
    )
