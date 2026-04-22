"""Import provenance logging.

Records every import operation for data traceability and reproducibility.
Each entry captures: when, what importer, which dataset/subject, how many
atoms, any errors, and the import parameters.

Usage from an importer::

    from neuroatom.index.import_log import log_import

    results = importer.import_subject(...)
    log_import(
        indexer=indexer,
        dataset_id="bci_comp_iv_2a",
        importer_name="BCICompIV2aImporter",
        n_atoms=sum(len(r.atoms) for r in results),
        subject_id="A01",
        parameters={"mat_path": str(mat_path)},
        duration_seconds=elapsed,
    )

Query import history::

    history = get_import_history(indexer)
    for entry in history:
        print(f"{entry['timestamp']} {entry['importer_name']} → {entry['n_atoms']} atoms")
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def log_import(
    indexer,
    dataset_id: str,
    importer_name: str,
    n_atoms: int,
    subject_id: Optional[str] = None,
    n_errors: int = 0,
    parameters: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
) -> int:
    """Record an import event in the SQLite index for provenance tracking.

    This is a convenience function that wraps
    ``indexer.backend.insert_import_log()``.

    Args:
        indexer: The Indexer instance (provides access to SQLiteBackend).
        dataset_id: The dataset that was imported.
        importer_name: Name of the importer class.
        n_atoms: Total atoms imported in this operation.
        subject_id: Subject imported (None for whole-dataset).
        n_errors: Number of errors during import.
        parameters: Dict of import parameters (will be JSON-serialized).
        duration_seconds: Wall-clock time of the import.

    Returns:
        The log_id of the recorded entry.
    """
    import neuroatom

    params_json = json.dumps(parameters, default=str) if parameters else None
    version = getattr(neuroatom, "__version__", None)

    log_id = indexer.backend.insert_import_log(
        dataset_id=dataset_id,
        importer_name=importer_name,
        n_atoms=n_atoms,
        subject_id=subject_id,
        importer_version=version,
        n_errors=n_errors,
        parameters=params_json,
        duration_seconds=duration_seconds,
    )

    logger.info(
        "Import logged: %s %s/%s → %d atoms (log_id=%d)",
        importer_name, dataset_id, subject_id or "*", n_atoms, log_id,
    )
    return log_id


def get_import_history(
    indexer,
    dataset_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Retrieve import history from the index.

    Usage::

        history = get_import_history(indexer, dataset_id="bci_comp_iv_2a")
        for entry in history:
            print(f"{entry['timestamp']} {entry['importer_name']} → {entry['n_atoms']} atoms")

    Args:
        indexer: The Indexer instance.
        dataset_id: Filter by dataset. None = all.
        limit: Max entries to return.

    Returns:
        List of dicts (newest first), each containing:

        - ``log_id`` (int): Auto-incremented row ID.
        - ``timestamp`` (str): ISO 8601 UTC timestamp.
        - ``dataset_id`` (str): Dataset that was imported.
        - ``subject_id`` (str or None): Subject imported.
        - ``importer_name`` (str): Importer class name.
        - ``importer_version`` (str or None): Package version.
        - ``n_atoms`` (int): Atoms imported.
        - ``n_errors`` (int): Errors encountered.
        - ``parameters`` (dict or None): Deserialized import parameters.
        - ``duration_seconds`` (float or None): Wall-clock time.
    """
    rows = indexer.backend.get_import_history(dataset_id=dataset_id, limit=limit)

    # Parse parameters JSON back to dict
    for row in rows:
        if row.get("parameters"):
            try:
                row["parameters"] = json.loads(row["parameters"])
            except (json.JSONDecodeError, TypeError):
                pass

    return rows
