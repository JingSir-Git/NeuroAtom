"""SQLite WAL backend for the NeuroAtom query accelerator index.

The SQLite index is a read-optimized accelerator — the JSONL files remain
the source of truth. The index can always be rebuilt via ``reindex``.

Design:
- WAL mode for concurrent reads during DataLoader access
- Flat tables optimized for the Query DSL filters
- Hash column for JSONL↔SQLite consistency checking
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neuroatom.core.atom import Atom
from neuroatom.core.enums import AtomType
from neuroatom.storage import paths as P

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# DDL: table definitions
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS atoms (
    atom_id           TEXT PRIMARY KEY,
    atom_type         TEXT NOT NULL,
    dataset_id        TEXT NOT NULL,
    subject_id        TEXT NOT NULL,
    session_id        TEXT,
    run_id            TEXT,
    trial_index       INTEGER,
    n_channels        INTEGER NOT NULL,
    sampling_rate     REAL NOT NULL,
    duration_samples  INTEGER NOT NULL,
    duration_seconds  REAL NOT NULL,
    onset_sample      INTEGER NOT NULL,
    onset_seconds     REAL NOT NULL,
    modality          TEXT,
    quality_status    TEXT,
    source_version    TEXT,
    signal_file_path  TEXT,
    shard_index       INTEGER
);

CREATE TABLE IF NOT EXISTS channels (
    atom_id        TEXT NOT NULL,
    channel_id     TEXT NOT NULL,
    standard_name  TEXT,
    PRIMARY KEY (atom_id, channel_id),
    FOREIGN KEY (atom_id) REFERENCES atoms(atom_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id   TEXT NOT NULL,
    atom_id         TEXT NOT NULL,
    annotation_type TEXT NOT NULL,
    name            TEXT NOT NULL,
    value_text      TEXT,         -- For categorical/text annotations
    value_numeric   REAL,         -- For numeric annotations
    stimulus_id     TEXT,         -- For stimulus_ref annotations
    PRIMARY KEY (annotation_id, atom_id),
    FOREIGN KEY (atom_id) REFERENCES atoms(atom_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jsonl_hashes (
    run_key  TEXT PRIMARY KEY,   -- "dataset_id|subject_id|session_id|run_id"
    hash     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_log (
    log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,       -- ISO 8601 UTC
    dataset_id     TEXT NOT NULL,
    subject_id     TEXT,                -- NULL for whole-dataset imports
    importer_name  TEXT NOT NULL,       -- e.g. 'BCICompIV2aImporter'
    importer_version TEXT,              -- package version at import time
    n_atoms        INTEGER NOT NULL DEFAULT 0,
    n_errors       INTEGER NOT NULL DEFAULT 0,
    parameters     TEXT,                -- JSON-serialized import parameters
    duration_seconds REAL               -- wall-clock time of import
);

CREATE INDEX IF NOT EXISTS idx_import_log_dataset ON import_log(dataset_id);
CREATE INDEX IF NOT EXISTS idx_import_log_time ON import_log(timestamp);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_atoms_dataset ON atoms(dataset_id);
CREATE INDEX IF NOT EXISTS idx_atoms_subject ON atoms(dataset_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(atom_type);
CREATE INDEX IF NOT EXISTS idx_atoms_quality ON atoms(quality_status);
CREATE INDEX IF NOT EXISTS idx_atoms_srate ON atoms(sampling_rate);
CREATE INDEX IF NOT EXISTS idx_atoms_duration ON atoms(duration_seconds);
CREATE INDEX IF NOT EXISTS idx_channels_standard ON channels(standard_name);
CREATE INDEX IF NOT EXISTS idx_annotations_name ON annotations(name);
CREATE INDEX IF NOT EXISTS idx_annotations_value ON annotations(name, value_text);
CREATE INDEX IF NOT EXISTS idx_atoms_modality ON atoms(modality);
"""


class SQLiteBackend:
    """SQLite WAL backend for the atom index.

    Thread-safe for concurrent reads (WAL mode). Writes should be
    serialized by the caller (e.g., via dataset file lock).
    """

    def __init__(self, db_path: Path):
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open the database connection and ensure schema exists."""
        self._conn = sqlite3.connect(
            str(self._path),
            timeout=30,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)

        # Store schema version
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Insert / upsert
    # ------------------------------------------------------------------

    def upsert_atom(self, atom: Atom) -> None:
        """Insert or replace an atom and its channels/annotations in the index."""
        conn = self.conn

        # Atom row
        conn.execute(
            """
            INSERT OR REPLACE INTO atoms (
                atom_id, atom_type, dataset_id, subject_id, session_id, run_id,
                trial_index, n_channels, sampling_rate,
                duration_samples, duration_seconds, onset_sample, onset_seconds,
                modality, quality_status, source_version, signal_file_path, shard_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atom.atom_id,
                atom.atom_type.value,
                atom.dataset_id,
                atom.subject_id,
                atom.session_id,
                atom.run_id,
                atom.trial_index,
                atom.n_channels,
                atom.sampling_rate,
                atom.temporal.duration_samples,
                atom.temporal.duration_seconds,
                atom.temporal.onset_sample,
                atom.temporal.onset_seconds,
                atom.modality,
                atom.quality.overall_status.value if atom.quality else None,
                atom.processing_history.version_tag,
                atom.signal_ref.file_path,
                atom.signal_ref.shard_index,
            ),
        )

        # Channels
        conn.execute("DELETE FROM channels WHERE atom_id = ?", (atom.atom_id,))
        for ch_id in atom.channel_ids:
            # We store channel_id; standard_name lookup is done during indexing
            conn.execute(
                "INSERT INTO channels (atom_id, channel_id, standard_name) VALUES (?, ?, ?)",
                (atom.atom_id, ch_id, None),
            )

        # Annotations
        conn.execute("DELETE FROM annotations WHERE atom_id = ?", (atom.atom_id,))
        for ann in atom.annotations:
            value_text = None
            value_numeric = None
            stimulus_id = None

            if ann.annotation_type == "categorical":
                value_text = ann.value
            elif ann.annotation_type == "numeric":
                value_numeric = ann.numeric_value
            elif ann.annotation_type == "text":
                value_text = ann.text_value
            elif ann.annotation_type == "stimulus_ref":
                stimulus_id = ann.stimulus_id
            elif ann.annotation_type == "event_sequence":
                # Store event count as text for basic filtering
                value_text = f"n_events={len(ann.events)}"

            conn.execute(
                """
                INSERT INTO annotations (
                    annotation_id, atom_id, annotation_type, name,
                    value_text, value_numeric, stimulus_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ann.annotation_id,
                    atom.atom_id,
                    ann.annotation_type,
                    ann.name,
                    value_text,
                    value_numeric,
                    stimulus_id,
                ),
            )

    def upsert_atoms(self, atoms: List[Atom]) -> None:
        """Batch insert/replace atoms."""
        for atom in atoms:
            self.upsert_atom(atom)
        self.conn.commit()

    def upsert_channel_standard_names(
        self, atom_id: str, channel_mapping: Dict[str, Optional[str]]
    ) -> None:
        """Update standard_name for channels of an atom."""
        conn = self.conn
        for ch_id, std_name in channel_mapping.items():
            conn.execute(
                "UPDATE channels SET standard_name = ? WHERE atom_id = ? AND channel_id = ?",
                (std_name, atom_id, ch_id),
            )
        conn.commit()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_atom(self, atom_id: str) -> None:
        """Remove an atom and its related data from the index."""
        conn = self.conn
        conn.execute("DELETE FROM atoms WHERE atom_id = ?", (atom_id,))
        conn.commit()

    def delete_run(self, dataset_id: str, subject_id: str, session_id: str, run_id: str) -> int:
        """Delete all atoms for a specific run. Returns count deleted."""
        conn = self.conn
        cursor = conn.execute(
            "DELETE FROM atoms WHERE dataset_id=? AND subject_id=? AND session_id=? AND run_id=?",
            (dataset_id, subject_id, session_id, run_id),
        )
        conn.commit()
        return cursor.rowcount

    def delete_dataset(self, dataset_id: str) -> int:
        """Delete all atoms for a dataset. Returns count deleted."""
        conn = self.conn
        cursor = conn.execute(
            "DELETE FROM atoms WHERE dataset_id = ?", (dataset_id,)
        )
        conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # JSONL hash tracking
    # ------------------------------------------------------------------

    def set_jsonl_hash(self, run_key: str, hash_val: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO jsonl_hashes (run_key, hash) VALUES (?, ?)",
            (run_key, hash_val),
        )
        self.conn.commit()

    def get_jsonl_hash(self, run_key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT hash FROM jsonl_hashes WHERE run_key = ?", (run_key,)
        ).fetchone()
        return row["hash"] if row else None

    # ------------------------------------------------------------------
    # Import provenance log
    # ------------------------------------------------------------------

    def insert_import_log(
        self,
        dataset_id: str,
        importer_name: str,
        n_atoms: int,
        subject_id: Optional[str] = None,
        importer_version: Optional[str] = None,
        n_errors: int = 0,
        parameters: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> int:
        """Record an import operation for provenance tracking.

        Args:
            dataset_id: Dataset that was imported.
            importer_name: Class name of the importer (e.g. 'BCICompIV2aImporter').
            n_atoms: Number of atoms successfully imported.
            subject_id: Subject imported (None for whole-dataset).
            importer_version: Package version at import time.
            n_errors: Number of errors encountered.
            parameters: JSON string of import parameters.
            duration_seconds: Wall-clock time of the import.

        Returns:
            The log_id of the inserted row.
        """
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            """INSERT INTO import_log
               (timestamp, dataset_id, subject_id, importer_name,
                importer_version, n_atoms, n_errors, parameters, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, dataset_id, subject_id, importer_name,
             importer_version, n_atoms, n_errors, parameters, duration_seconds),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_import_history(
        self, dataset_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve import history, newest first.

        Args:
            dataset_id: Filter by dataset. None = all datasets.
            limit: Max rows to return.

        Returns:
            List of dicts with import log fields.
        """
        if dataset_id:
            rows = self.conn.execute(
                "SELECT * FROM import_log WHERE dataset_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (dataset_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM import_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Query helpers (raw SQL)
    # ------------------------------------------------------------------

    def count_atoms(self, dataset_id: Optional[str] = None) -> int:
        if dataset_id:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM atoms WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) as cnt FROM atoms").fetchone()
        return row["cnt"]

    def execute_query(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute a parameterized SQL query and return rows."""
        return self.conn.execute(sql, params).fetchall()

    def get_atom_ids(self, dataset_id: Optional[str] = None) -> List[str]:
        if dataset_id:
            rows = self.conn.execute(
                "SELECT atom_id FROM atoms WHERE dataset_id = ?", (dataset_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT atom_id FROM atoms").fetchall()
        return [r["atom_id"] for r in rows]
