"""Metadata I/O: JSONL for atom metadata, JSON for structured metadata.

JSONL (one JSON object per line) is the source of truth for atom metadata.
JSON files store dataset, subject, session, and run metadata.
All Pydantic models are serialized/deserialized via model_dump/model_validate.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from neuroatom.core.atom import Atom

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# JSON: single-document metadata files
# ---------------------------------------------------------------------------

def write_json(model: BaseModel, path: Path) -> None:
    """Write a Pydantic model to a JSON file.

    Creates parent directories if they don't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.model_dump(mode="json")
    # Atomic write: write to .tmp then rename
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def read_json(path: Path, model_class: Type[T]) -> T:
    """Read a JSON file and validate it as a Pydantic model."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model_class.model_validate(data)


def read_json_raw(path: Path) -> Dict[str, Any]:
    """Read a JSON file as a plain dict (no validation)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# JSONL: atom metadata (one line per atom, source of truth)
# ---------------------------------------------------------------------------

class AtomJSONLWriter:
    """Append-only writer for atom JSONL files.

    Each line is a self-contained JSON object representing one Atom.
    """

    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def write_atom(self, atom: Atom) -> None:
        """Append one atom as a JSON line."""
        line = atom.model_dump_json()
        self._file.write(line + "\n")

    def write_atoms(self, atoms: List[Atom]) -> None:
        """Append multiple atoms."""
        for atom in atoms:
            self.write_atom(atom)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class AtomJSONLReader:
    """Reader for atom JSONL files."""

    def __init__(self, path: Path):
        self._path = path

    def read_all(self) -> List[Atom]:
        """Read all atoms from the JSONL file."""
        atoms = []
        if not self._path.exists():
            return atoms
        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    atom = Atom.model_validate(data)
                    atoms.append(atom)
                except Exception as e:
                    logger.warning(
                        "Failed to parse atom at line %d in %s: %s",
                        line_num, self._path, e,
                    )
        return atoms

    def iter_atoms(self):
        """Iterate over atoms lazily (generator)."""
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield Atom.model_validate(data)
                except Exception as e:
                    logger.warning(
                        "Failed to parse atom at line %d in %s: %s",
                        line_num, self._path, e,
                    )

    def count(self) -> int:
        """Count atoms without full deserialization."""
        if not self._path.exists():
            return 0
        count = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def get_atom_ids(self) -> List[str]:
        """Extract only atom_ids without full deserialization."""
        ids = []
        if not self._path.exists():
            return ids
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ids.append(data["atom_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return ids


# ---------------------------------------------------------------------------
# JSONL hash for consistency checking
# ---------------------------------------------------------------------------

def compute_jsonl_hash(path: Path) -> str:
    """Compute SHA-256 hash of a JSONL file for consistency checking.

    Used to detect if the JSONL source of truth has been modified
    since the SQLite index was last built.
    """
    import hashlib

    h = hashlib.sha256()
    if not path.exists():
        return h.hexdigest()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
