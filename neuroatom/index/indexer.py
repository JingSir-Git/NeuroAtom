"""Indexer: build and maintain the SQLite index from JSONL source of truth.

Supports:
- Full reindex of entire pool or single dataset
- Incremental index (only process changed JSONL files via hash comparison)
- Channel standard_name resolution during indexing
"""

import logging
from pathlib import Path
from typing import List, Optional

from neuroatom.core.channel import ChannelInfo
from neuroatom.index.sqlite_backend import SQLiteBackend
from neuroatom.storage import paths as P
from neuroatom.storage.metadata_store import AtomJSONLReader, compute_jsonl_hash, read_json
from neuroatom.storage.pool import Pool

logger = logging.getLogger(__name__)


class Indexer:
    """Builds and maintains the SQLite atom index from JSONL files."""

    def __init__(self, pool: Pool, backend: Optional[SQLiteBackend] = None):
        self._pool = pool
        self._backend = backend or SQLiteBackend(P.index_db_path(pool.root))
        if not self._backend._conn:
            self._backend.connect()

    @property
    def backend(self) -> SQLiteBackend:
        return self._backend

    # ------------------------------------------------------------------
    # Full reindex
    # ------------------------------------------------------------------

    def reindex_all(self) -> int:
        """Reindex all datasets in the pool. Returns total atom count."""
        total = 0
        for dataset_id in self._pool.list_datasets():
            count = self.reindex_dataset(dataset_id)
            total += count
        logger.info("Full reindex complete: %d atoms indexed.", total)
        return total

    def reindex_dataset(self, dataset_id: str) -> int:
        """Reindex a single dataset. Returns atom count."""
        # Delete old index entries for this dataset
        self._backend.delete_dataset(dataset_id)

        count = 0
        for subject_id in self._pool.list_subjects(dataset_id):
            for session_id in self._pool.list_sessions(dataset_id, subject_id):
                for run_id in self._pool.list_runs(dataset_id, subject_id, session_id):
                    n = self._index_run(dataset_id, subject_id, session_id, run_id)
                    count += n

        logger.info("Reindexed dataset '%s': %d atoms.", dataset_id, count)
        return count

    # ------------------------------------------------------------------
    # Incremental index
    # ------------------------------------------------------------------

    def index_incremental(self) -> int:
        """Incrementally index: only process JSONL files that changed since last index.

        Uses SHA-256 hash comparison to detect changes.
        Returns count of newly indexed atoms.
        """
        total = 0
        for dataset_id in self._pool.list_datasets():
            for subject_id in self._pool.list_subjects(dataset_id):
                for session_id in self._pool.list_sessions(dataset_id, subject_id):
                    for run_id in self._pool.list_runs(dataset_id, subject_id, session_id):
                        run_key = f"{dataset_id}|{subject_id}|{session_id}|{run_id}"
                        jsonl_path = P.atoms_jsonl_path(
                            self._pool.root, dataset_id, subject_id, session_id, run_id
                        )

                        if not jsonl_path.exists():
                            continue

                        current_hash = compute_jsonl_hash(jsonl_path)
                        stored_hash = self._backend.get_jsonl_hash(run_key)

                        if stored_hash == current_hash:
                            logger.debug("Run %s unchanged, skipping.", run_key)
                            continue

                        # Re-index this run
                        self._backend.delete_run(dataset_id, subject_id, session_id, run_id)
                        n = self._index_run(dataset_id, subject_id, session_id, run_id)
                        total += n

        logger.info("Incremental index complete: %d new atoms.", total)
        return total

    # ------------------------------------------------------------------
    # Single run indexing
    # ------------------------------------------------------------------

    def _index_run(
        self,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
    ) -> int:
        """Index all atoms from a single run's JSONL file."""
        jsonl_path = P.atoms_jsonl_path(
            self._pool.root, dataset_id, subject_id, session_id, run_id
        )

        if not jsonl_path.exists():
            logger.warning("JSONL not found for %s/%s/%s/%s", dataset_id, subject_id, session_id, run_id)
            return 0

        reader = AtomJSONLReader(jsonl_path)
        atoms = reader.read_all()

        if not atoms:
            return 0

        # Load channel infos for standard_name resolution
        channel_map = self._load_channel_map(dataset_id, subject_id, session_id)

        # Batch upsert
        self._backend.upsert_atoms(atoms)

        # Update standard names if channel map available
        if channel_map:
            for atom in atoms:
                mapping = {}
                for ch_id in atom.channel_ids:
                    if ch_id in channel_map:
                        mapping[ch_id] = channel_map[ch_id]
                if mapping:
                    self._backend.upsert_channel_standard_names(atom.atom_id, mapping)

        # Store JSONL hash
        run_key = f"{dataset_id}|{subject_id}|{session_id}|{run_id}"
        file_hash = compute_jsonl_hash(jsonl_path)
        self._backend.set_jsonl_hash(run_key, file_hash)

        logger.debug("Indexed %d atoms from %s", len(atoms), jsonl_path)
        return len(atoms)

    def _load_channel_map(
        self,
        dataset_id: str,
        subject_id: str,
        session_id: str,
    ) -> dict:
        """Load channel_id → standard_name mapping from session metadata."""
        channels_file = P.channels_path(
            self._pool.root, dataset_id, subject_id, session_id
        )
        if not channels_file.exists():
            return {}

        try:
            import json
            with open(channels_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Expect list of dicts with channel_id and standard_name
            mapping = {}
            for ch in data:
                ch_id = ch.get("channel_id")
                std_name = ch.get("standard_name")
                if ch_id:
                    mapping[ch_id] = std_name
            return mapping
        except Exception as e:
            logger.warning("Could not load channel map from %s: %s", channels_file, e)
            return {}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return summary statistics of the index."""
        conn = self._backend.conn
        total = self._backend.count_atoms()

        # Per dataset
        rows = conn.execute(
            "SELECT dataset_id, COUNT(*) as cnt FROM atoms GROUP BY dataset_id"
        ).fetchall()
        per_dataset = {r["dataset_id"]: r["cnt"] for r in rows}

        # Per atom type
        rows = conn.execute(
            "SELECT atom_type, COUNT(*) as cnt FROM atoms GROUP BY atom_type"
        ).fetchall()
        per_type = {r["atom_type"]: r["cnt"] for r in rows}

        # Label distribution
        rows = conn.execute(
            "SELECT name, value_text, COUNT(*) as cnt FROM annotations "
            "WHERE annotation_type='categorical' GROUP BY name, value_text"
        ).fetchall()
        label_dist = {}
        for r in rows:
            key = f"{r['name']}:{r['value_text']}"
            label_dist[key] = r["cnt"]

        # Sampling rate distribution
        rows = conn.execute(
            "SELECT DISTINCT sampling_rate FROM atoms ORDER BY sampling_rate"
        ).fetchall()
        srates = [r["sampling_rate"] for r in rows]

        return {
            "total_atoms": total,
            "per_dataset": per_dataset,
            "per_type": per_type,
            "label_distribution": label_dist,
            "sampling_rates": srates,
        }

    def close(self) -> None:
        self._backend.close()
