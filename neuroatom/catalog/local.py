"""Local catalog: build and maintain a dataset index from pool metadata.

The local catalog lives at ``<pool_root>/catalog.json`` and is automatically
updated when datasets are imported, deleted, or quality-assessed.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from neuroatom.catalog.models import CatalogIndex, DatasetEntry
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import QualityTier
from neuroatom.storage.pool import Pool
from neuroatom.storage import paths as P

logger = logging.getLogger(__name__)

CATALOG_FILENAME = "catalog.json"


def catalog_path(pool: Pool) -> Path:
    """Return the path to the pool's catalog file."""
    return pool.root / CATALOG_FILENAME


def load_catalog(pool: Pool) -> CatalogIndex:
    """Load the catalog from disk, or return an empty one."""
    path = catalog_path(pool)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CatalogIndex.model_validate(data)
        except Exception as e:
            logger.warning("Failed to load catalog: %s — starting fresh", e)
    return CatalogIndex(name="local")


def save_catalog(pool: Pool, catalog: CatalogIndex) -> None:
    """Persist the catalog to disk."""
    catalog.updated_at = datetime.now(timezone.utc).isoformat()
    path = catalog_path(pool)
    path.write_text(
        catalog.model_dump_json(indent=2),
        encoding="utf-8",
    )


def entry_from_dataset_meta(
    meta: DatasetMeta,
    pool: Pool,
    n_atoms: Optional[int] = None,
) -> DatasetEntry:
    """Convert a DatasetMeta + pool stats into a DatasetEntry."""
    return DatasetEntry(
        dataset_id=meta.dataset_id,
        name=meta.name,
        description=meta.description,
        task_types=meta.task_types,
        n_subjects=meta.n_subjects,
        n_atoms=n_atoms,
        n_channels_range=meta.n_channels_range,
        sampling_rate_range=meta.sampling_rate_range,
        quality_tier=meta.quality_tier,
        license=meta.license,
        citation=meta.citation,
        source_url=meta.source_url,
        import_timestamp=meta.import_timestamp,
        pool_path=str(pool.root),
    )


def rebuild_catalog(pool: Pool) -> CatalogIndex:
    """Rebuild the entire catalog from pool metadata.

    Scans all datasets in the pool and creates entries from their
    dataset.json files. This is useful after manual edits or migration.
    """
    catalog = CatalogIndex(name="local")

    for ds_id in pool.list_datasets():
        try:
            meta = pool.get_dataset_meta(ds_id)
            n_atoms = _count_atoms(pool, ds_id)
            entry = entry_from_dataset_meta(meta, pool, n_atoms=n_atoms)
            catalog.upsert(entry)
        except Exception as e:
            logger.warning("Skipping dataset %s: %s", ds_id, e)

    save_catalog(pool, catalog)
    logger.info(
        "Rebuilt catalog: %d datasets", len(catalog.datasets),
    )
    return catalog


def update_catalog_entry(
    pool: Pool,
    dataset_id: str,
    n_atoms: Optional[int] = None,
) -> DatasetEntry:
    """Update a single dataset entry in the catalog."""
    catalog = load_catalog(pool)
    meta = pool.get_dataset_meta(dataset_id)
    if n_atoms is None:
        n_atoms = _count_atoms(pool, dataset_id)
    entry = entry_from_dataset_meta(meta, pool, n_atoms=n_atoms)
    catalog.upsert(entry)
    save_catalog(pool, catalog)
    return entry


def remove_catalog_entry(pool: Pool, dataset_id: str) -> bool:
    """Remove a dataset from the catalog."""
    catalog = load_catalog(pool)
    removed = catalog.remove(dataset_id)
    if removed:
        save_catalog(pool, catalog)
    return removed


def _count_atoms(pool: Pool, dataset_id: str) -> int:
    """Count total atoms in a dataset by scanning JSONL files."""
    ds_dir = P.dataset_dir(pool.root, dataset_id)
    count = 0
    for jsonl_path in ds_dir.rglob("atoms.jsonl"):
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception:
            pass
    return count
