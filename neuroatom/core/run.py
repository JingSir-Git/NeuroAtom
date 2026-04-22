"""RunMeta: per-recording-run metadata."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RunMeta(BaseModel):
    """Metadata for a single recording run within a session.

    A run is a continuous block of recorded data for a specific task.
    One session may contain multiple runs (e.g., different task blocks).
    """

    run_id: str = Field(..., description="Unique run identifier within the session.")
    session_id: str = Field(..., description="Parent session ID.")
    subject_id: str = Field(..., description="Parent subject ID.")
    dataset_id: str = Field(..., description="Parent dataset ID.")
    run_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Sequential order of this run within the session (for inter-run ordering).",
    )
    task_type: str = Field(
        ...,
        description=(
            "Task paradigm type. Standard values: 'motor_imagery', 'p300', 'ssvep', "
            "'auditory_attention_decoding', 'language_decoding', 'emotion', "
            "'sleep_staging', 'seizure_detection', 'resting_state', 'other'."
        ),
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Human-readable task description.",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total run duration in seconds.",
    )
    n_trials: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of trials/epochs in this run.",
    )
    n_events: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of event markers in this run.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Task instructions given to the subject.",
    )
    paradigm_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task-specific paradigm parameters (ISI, SOA, trial structure, etc.).",
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs.",
    )
