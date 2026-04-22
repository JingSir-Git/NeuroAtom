"""Incremental import progress tracker.

Tracks which runs have been successfully imported, enabling resume
after interruption. Progress is stored as a JSON file at pool root.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from neuroatom.storage import paths as P

logger = logging.getLogger(__name__)


class ImportProgress:
    """Tracks import progress for incremental/resumable imports.

    Stores a mapping of (dataset_id, subject_id, session_id, run_id) → status.
    Persisted as JSON at pool_root/import_progress.json.
    """

    def __init__(self, pool_root: Path):
        self._pool_root = Path(pool_root)
        self._path = P.import_progress_path(self._pool_root)
        self._data: Dict[str, Dict] = self._load()

    def _load(self) -> Dict:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp.replace(self._path)

    @staticmethod
    def _key(dataset_id: str, subject_id: str, session_id: str, run_id: str) -> str:
        return f"{dataset_id}|{subject_id}|{session_id}|{run_id}"

    def mark_started(
        self,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
    ) -> None:
        """Mark a run as started (in-progress)."""
        key = self._key(dataset_id, subject_id, session_id, run_id)
        self._data[key] = {"status": "started"}
        self._save()

    def mark_completed(
        self,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
        n_atoms: int = 0,
    ) -> None:
        """Mark a run as successfully completed."""
        key = self._key(dataset_id, subject_id, session_id, run_id)
        self._data[key] = {"status": "completed", "n_atoms": n_atoms}
        self._save()

    def mark_failed(
        self,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
        error: str = "",
    ) -> None:
        """Mark a run as failed."""
        key = self._key(dataset_id, subject_id, session_id, run_id)
        self._data[key] = {"status": "failed", "error": error}
        self._save()

    def is_completed(
        self,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
    ) -> bool:
        """Check if a run has already been successfully imported."""
        key = self._key(dataset_id, subject_id, session_id, run_id)
        entry = self._data.get(key)
        return entry is not None and entry.get("status") == "completed"

    def get_status(
        self,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
    ) -> Optional[str]:
        """Get the import status for a run."""
        key = self._key(dataset_id, subject_id, session_id, run_id)
        entry = self._data.get(key)
        return entry.get("status") if entry else None

    def get_completed_runs(self, dataset_id: str) -> List[str]:
        """Get all completed run keys for a dataset."""
        prefix = f"{dataset_id}|"
        return [
            key for key, entry in self._data.items()
            if key.startswith(prefix) and entry.get("status") == "completed"
        ]

    def reset_dataset(self, dataset_id: str) -> int:
        """Remove all progress entries for a dataset. Returns count removed."""
        prefix = f"{dataset_id}|"
        keys_to_remove = [k for k in self._data if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._data[k]
        self._save()
        return len(keys_to_remove)

    def summary(self) -> Dict[str, int]:
        """Return summary counts by status."""
        counts: Dict[str, int] = {}
        for entry in self._data.values():
            status = entry.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts
