"""ElectrodeLocation: 3D spatial position of an EEG electrode."""

from pydantic import BaseModel, Field


class ElectrodeLocation(BaseModel):
    """Three-dimensional position of an electrode on the scalp.

    Used for channel mapping (nearest-neighbor matching, spherical spline
    interpolation) when combining data from different montages.
    """

    x: float = Field(..., description="X coordinate.")
    y: float = Field(..., description="Y coordinate.")
    z: float = Field(..., description="Z coordinate.")
    coordinate_system: str = Field(
        default="unknown",
        description=(
            "Coordinate system name. Standard values: 'CapTrak', 'CTF', 'MNI', "
            "'EEGLAB', 'FreeSurfer', 'unknown'. ChannelMapper will attempt to "
            "transform all coordinates to MNI head space before comparison."
        ),
    )
    coordinate_units: str = Field(
        default="m",
        description="Unit of coordinate values: 'm', 'mm', 'cm'.",
    )
