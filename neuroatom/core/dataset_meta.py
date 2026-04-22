"""DatasetMeta: dataset-level metadata."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class DatasetMeta(BaseModel):
    """Top-level metadata for an imported dataset.

    One DatasetMeta per unique dataset in the pool.
    """

    dataset_id: str = Field(
        ...,
        description="Unique dataset identifier (e.g. 'bci_comp_iv_2a', 'physionet_mi').",
    )
    name: str = Field(
        ...,
        description="Human-readable dataset name.",
    )
    description: Optional[str] = Field(default=None, description="Dataset description.")
    source_url: Optional[str] = Field(
        default=None,
        description="URL where the original dataset can be downloaded.",
    )
    license: Optional[str] = Field(
        default=None,
        description="License identifier (e.g. 'CC0', 'CC BY 4.0').",
    )
    citation: Optional[str] = Field(
        default=None,
        description="Citation string or BibTeX for the dataset.",
    )
    task_types: List[str] = Field(
        default_factory=list,
        description="List of task paradigm types present in this dataset.",
    )
    original_format: Optional[str] = Field(
        default=None,
        description="Original file format: 'edf', 'gdf', 'mat', 'bids', 'set', 'vhdr', 'fif', etc.",
    )
    n_subjects: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of subjects in this dataset.",
    )
    n_channels_range: Optional[Tuple[int, int]] = Field(
        default=None,
        description="(min, max) channel count across all recordings.",
    )
    sampling_rate_range: Optional[Tuple[float, float]] = Field(
        default=None,
        description="(min, max) sampling rate across all recordings.",
    )
    import_timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of when this dataset was imported into the pool.",
    )
    import_config_ref: Optional[str] = Field(
        default=None,
        description="Path to the task config YAML file used for import.",
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible key-value pairs.",
    )
