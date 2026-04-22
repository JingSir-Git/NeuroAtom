"""SessionMeta: per-recording-session metadata."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SessionMeta(BaseModel):
    """Metadata for a single recording session.

    A session represents one visit / one sitting at the recording device.
    Contains device information, montage, reference scheme, and environmental
    conditions — all of which are important confounding variables for
    cross-dataset training.
    """

    session_id: str = Field(..., description="Unique session identifier within the subject.")
    subject_id: str = Field(..., description="Parent subject ID.")
    dataset_id: str = Field(..., description="Parent dataset ID.")
    date: Optional[str] = Field(
        default=None,
        description="Recording date in ISO 8601 format (e.g. '2024-03-15').",
    )

    # ---- Device ----
    device_manufacturer: Optional[str] = Field(
        default=None,
        description="Device manufacturer: 'BioSemi', 'g.tec', 'Neuroscan', 'OpenBCI', 'Emotiv', etc.",
    )
    device_model: Optional[str] = Field(
        default=None,
        description="Device model: 'ActiveTwo', 'USBamp', 'Cyton', etc.",
    )
    device_serial: Optional[str] = Field(default=None, description="Device serial number.")
    electrode_type: Optional[str] = Field(
        default=None,
        description="Electrode type: 'wet', 'dry', 'saline', 'gel'.",
    )

    # ---- Recording parameters ----
    sampling_rate: float = Field(..., gt=0, description="Nominal sampling rate in Hz.")
    reference_scheme: Optional[str] = Field(
        default=None,
        description=(
            "Reference scheme used during recording: "
            "'average', 'Cz', 'linked_ears', 'REST', 'monopolar', 'custom'."
        ),
    )
    ground_electrode: Optional[str] = Field(
        default=None,
        description="Ground electrode position (e.g. 'AFz', 'right mastoid').",
    )
    montage_id: Optional[str] = Field(
        default=None,
        description="ID of the montage definition (references montages/ directory).",
    )
    placement_scheme: Optional[str] = Field(
        default=None,
        description="Electrode placement scheme: '10-20', '10-10', '10-5', 'custom'.",
    )
    line_freq: Optional[float] = Field(
        default=None,
        description="Power line frequency in Hz (50 or 60).",
    )
    recording_type: Optional[str] = Field(
        default=None,
        description="Recording type: 'continuous', 'epoched', 'discontinuous'.",
    )

    # ---- Physical ----
    head_circumference: Optional[float] = Field(
        default=None,
        gt=0,
        description="Head circumference in cm.",
    )

    # ---- Filters applied during acquisition ----
    hardware_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hardware filters applied during acquisition (e.g. anti-aliasing).",
    )
    software_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Software filters applied during acquisition.",
    )

    # ---- Environment ----
    environment: Optional[str] = Field(
        default=None,
        description="Recording environment: 'shielded_room', 'hospital', 'home', 'lab'.",
    )

    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs.",
    )
