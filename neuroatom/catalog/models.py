"""Catalog data models: DatasetEntry and CatalogIndex."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from neuroatom.core.enums import QualityTier


class DatasetEntry(BaseModel):
    """A single dataset entry in the catalog.

    Contains enough metadata for discovery, filtering, and display
    without needing to read the full pool.
    """

    dataset_id: str = Field(
        ..., description="Unique dataset identifier.",
    )
    name: str = Field(
        ..., description="Human-readable dataset name.",
    )
    description: Optional[str] = Field(
        default=None, description="Short dataset description.",
    )
    task_types: List[str] = Field(
        default_factory=list,
        description="Paradigm types: motor_imagery, erp, ssvep, aad, etc.",
    )
    n_subjects: Optional[int] = Field(
        default=None, ge=0, description="Number of subjects.",
    )
    n_atoms: Optional[int] = Field(
        default=None, ge=0, description="Total number of atoms.",
    )
    n_channels_range: Optional[Tuple[int, int]] = Field(
        default=None, description="(min, max) channel count.",
    )
    sampling_rate_range: Optional[Tuple[float, float]] = Field(
        default=None, description="(min, max) sampling rate Hz.",
    )
    quality_tier: Optional[QualityTier] = Field(
        default=None, description="Quality tier (silver/gold/platinum).",
    )
    license: Optional[str] = Field(
        default=None, description="License identifier.",
    )
    citation: Optional[str] = Field(
        default=None, description="Citation string.",
    )
    source_url: Optional[str] = Field(
        default=None, description="Original data download URL.",
    )
    pool_url: Optional[str] = Field(
        default=None,
        description="URL to download a pre-built .napool archive.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Searchable tags: eeg, ecog, bci, clinical, etc.",
    )
    import_timestamp: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp of last import.",
    )
    pool_path: Optional[str] = Field(
        default=None, description="Local pool path (for local catalog entries).",
    )

    def matches(
        self,
        query: Optional[str] = None,
        task_type: Optional[str] = None,
        min_subjects: Optional[int] = None,
        min_channels: Optional[int] = None,
        tier: Optional[QualityTier] = None,
        tag: Optional[str] = None,
    ) -> bool:
        """Check if this entry matches the given filters."""
        if query:
            q = query.lower()
            searchable = " ".join([
                self.dataset_id, self.name,
                self.description or "",
                " ".join(self.task_types),
                " ".join(self.tags),
            ]).lower()
            if q not in searchable:
                return False

        if task_type and task_type not in self.task_types:
            return False

        if min_subjects and (self.n_subjects or 0) < min_subjects:
            return False

        if min_channels and self.n_channels_range:
            if self.n_channels_range[1] < min_channels:
                return False

        if tier and self.quality_tier != tier:
            return False

        if tag and tag not in self.tags:
            return False

        return True


class CatalogIndex(BaseModel):
    """A collection of dataset entries, serializable to JSON.

    Can represent a local pool catalog or a remote registry.
    """

    version: str = Field(default="1.0", description="Catalog schema version.")
    name: str = Field(
        default="local", description="Catalog name (e.g. 'local', 'neuroatom-hub').",
    )
    description: Optional[str] = Field(
        default=None, description="Catalog description.",
    )
    url: Optional[str] = Field(
        default=None, description="URL of this catalog (if remote).",
    )
    updated_at: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp of last update.",
    )
    datasets: List[DatasetEntry] = Field(
        default_factory=list, description="Dataset entries.",
    )

    def search(self, **kwargs) -> List[DatasetEntry]:
        """Search/filter datasets. See DatasetEntry.matches() for params."""
        return [d for d in self.datasets if d.matches(**kwargs)]

    def get(self, dataset_id: str) -> Optional[DatasetEntry]:
        """Get a dataset entry by ID."""
        for d in self.datasets:
            if d.dataset_id == dataset_id:
                return d
        return None

    def upsert(self, entry: DatasetEntry) -> None:
        """Add or update a dataset entry."""
        for i, d in enumerate(self.datasets):
            if d.dataset_id == entry.dataset_id:
                self.datasets[i] = entry
                return
        self.datasets.append(entry)

    def remove(self, dataset_id: str) -> bool:
        """Remove a dataset entry. Returns True if found."""
        for i, d in enumerate(self.datasets):
            if d.dataset_id == dataset_id:
                self.datasets.pop(i)
                return True
        return False

    def merge(self, other: "CatalogIndex") -> int:
        """Merge entries from another catalog. Returns count of new/updated."""
        count = 0
        for entry in other.datasets:
            existing = self.get(entry.dataset_id)
            if existing is None or entry != existing:
                self.upsert(entry)
                count += 1
        return count
