"""ChannelInfo: per-channel metadata with name standardization."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import ChannelStatus, ChannelType


class ChannelInfo(BaseModel):
    """Metadata for a single recording channel.

    Each channel has both an original ``name`` (from the source file) and
    an optional ``standard_name`` (resolved via the channel alias table).
    All cross-dataset channel matching in queries and assembly uses
    ``standard_name``.
    """

    channel_id: str = Field(
        ...,
        description="Unique identifier within a run, typically '{name}_{index}'.",
    )
    index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of this channel in the data array.",
    )
    name: str = Field(
        ...,
        description="Original channel name from the source file (e.g. 'EEG Fp1', 'FP1').",
    )
    standard_name: Optional[str] = Field(
        default=None,
        description=(
            "Standardized channel name after alias resolution (e.g. 'Fp1'). "
            "None if no alias match was found."
        ),
    )
    type: ChannelType = Field(
        default=ChannelType.EEG,
        description="Channel modality type.",
    )
    unit: str = Field(
        default="uV",
        description=(
            "Physical unit of the signal: 'V', 'mV', 'uV', 'nV', or 'unknown'. "
            "Used by UnitStandardizer to convert all signals to a common unit."
        ),
    )
    sampling_rate: float = Field(
        ...,
        gt=0,
        description="Sampling rate of this channel in Hz.",
    )
    reference: Optional[str] = Field(
        default=None,
        description="Reference electrode for this channel, if known (e.g. 'Cz', 'average').",
    )
    location: Optional[ElectrodeLocation] = Field(
        default=None,
        description="3D electrode position, if available.",
    )
    impedance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Electrode impedance in kOhm, if recorded.",
    )
    status: ChannelStatus = Field(
        default=ChannelStatus.UNKNOWN,
        description="Channel quality status.",
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs for dataset-specific channel metadata.",
    )
