"""Content-addressable hashing utilities for atom IDs and deduplication."""

import hashlib
from typing import Optional


def compute_atom_id(
    dataset_id: str,
    subject_id: str,
    session_id: Optional[str],
    run_id: Optional[str],
    onset_sample: int,
    processing_hash: str = "raw",
) -> str:
    """Compute a deterministic, content-addressable atom ID.

    The atom ID is a SHA-256 hash of the concatenation of all identifying
    fields. This ensures:
    - Same data always produces the same ID (deduplication)
    - Different processing versions produce different IDs
    - IDs are globally unique across all pools

    Returns:
        Hex-encoded SHA-256 hash (64 characters).
    """
    parts = [
        dataset_id,
        subject_id,
        session_id or "",
        run_id or "",
        str(onset_sample),
        processing_hash,
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_processing_hash(steps_repr: str) -> str:
    """Compute a hash of the processing history for versioning.

    Args:
        steps_repr: JSON-serialized representation of the processing steps.

    Returns:
        First 16 hex chars of SHA-256 hash (sufficient for versioning).
    """
    return hashlib.sha256(steps_repr.encode("utf-8")).hexdigest()[:16]


def compute_content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of arbitrary binary data.

    Used for stimulus deduplication and JSONL consistency checking.
    """
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(filepath: str, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file without loading it fully into memory."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
