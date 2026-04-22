"""MontageInfo: electrode montage definitions."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from neuroatom.core.electrode import ElectrodeLocation


class MontageInfo(BaseModel):
    """Definition of an electrode montage (spatial layout).

    Standard montages (10-20, 10-10, BioSemi64, etc.) are stored as JSON
    files in ``neuroatom/configs/montages/`` and loaded on demand.
    Custom montages from specific datasets can also be registered.
    """

    montage_id: str = Field(
        ...,
        description="Unique montage identifier (e.g. 'standard_1020', 'biosemi64').",
    )
    name: str = Field(
        ...,
        description="Human-readable montage name.",
    )
    description: Optional[str] = Field(default=None, description="Montage description.")
    n_channels: int = Field(
        ...,
        gt=0,
        description="Number of electrode positions in this montage.",
    )
    channel_names: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of standard channel names in this montage.",
    )
    positions: Dict[str, ElectrodeLocation] = Field(
        default_factory=dict,
        description="Mapping from channel name to 3D electrode position.",
    )
    coordinate_system: str = Field(
        default="unknown",
        description="Coordinate system for all positions in this montage.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source of montage definition (e.g. 'mne', 'custom', 'bids').",
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs.",
    )
