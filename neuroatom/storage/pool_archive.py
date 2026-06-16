"""Pool archive: export / import NeuroAtom pools as `.napool` packages.

A `.napool` file is a tar.gz archive containing:

    manifest.json          ← content manifest + SHA-256 checksums
    pool.yaml              ← pool configuration snapshot
    datasets/              ← all dataset directories (HDF5 + JSONL + JSON metadata)
    stimuli/               ← (optional) shared stimulus files
    montages/              ← (optional) electrode montage definitions

Usage::

    # Full export
    export_pool(pool, Path("kul_aad_v1.napool"))

    # Selective export (only certain datasets)
    export_pool(pool, Path("kul_only.napool"), dataset_ids=["kul_aad"])

    # Incremental export (only datasets imported after a date)
    export_pool(pool, Path("incremental.napool"), since="2025-01-01")

    # Import into an existing or new pool
    import_pool(Path("kul_aad_v1.napool"), target_root=Path("./shared_pool"))
"""

import datetime
import hashlib
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from neuroatom.storage import paths as P
from neuroatom.storage.migration import CURRENT_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Archive format version (independent of pool schema version)
ARCHIVE_FORMAT_VERSION = "1.0.0"

# File extension
NAPOOL_EXTENSION = ".napool"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(
    pool_root: Path,
    files: List[Path],
    dataset_ids: List[str],
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the manifest.json content.

    Args:
        pool_root: Absolute pool root path.
        files: List of absolute file paths to include.
        dataset_ids: IDs of datasets included in this archive.
        description: Optional human-readable description.

    Returns:
        Manifest dictionary.
    """
    file_entries = []
    total_size = 0
    for f in files:
        rel = f.relative_to(pool_root).as_posix()
        size = f.stat().st_size
        sha = _sha256_file(f)
        file_entries.append({
            "path": rel,
            "size": size,
            "sha256": sha,
        })
        total_size += size

    # Pool config for metadata
    config_path = P.pool_config_path(pool_root)
    pool_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            pool_config = yaml.safe_load(fh) or {}

    manifest = {
        "archive_format_version": ARCHIVE_FORMAT_VERSION,
        "schema_version": pool_config.get("schema_version", CURRENT_SCHEMA_VERSION),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "description": description,
        "datasets": dataset_ids,
        "n_files": len(file_entries),
        "total_size_bytes": total_size,
        "files": file_entries,
    }
    return manifest


def _verify_manifest(
    extracted_root: Path,
    manifest: Dict[str, Any],
    strict: bool = True,
) -> List[str]:
    """Verify file integrity against manifest checksums.

    Returns:
        List of error messages (empty if all OK).
    """
    errors = []
    for entry in manifest.get("files", []):
        rel_path = entry["path"]
        expected_sha = entry["sha256"]
        expected_size = entry["size"]

        file_path = extracted_root / rel_path
        if not file_path.exists():
            errors.append(f"Missing file: {rel_path}")
            continue

        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            errors.append(
                f"Size mismatch: {rel_path} "
                f"(expected {expected_size}, got {actual_size})"
            )

        if strict:
            actual_sha = _sha256_file(file_path)
            if actual_sha != expected_sha:
                errors.append(
                    f"SHA-256 mismatch: {rel_path} "
                    f"(expected {expected_sha[:16]}..., got {actual_sha[:16]}...)"
                )

    return errors


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _collect_pool_files(
    pool_root: Path,
    dataset_ids: Optional[List[str]] = None,
    subject_ids: Optional[List[str]] = None,
    since: Optional[str] = None,
) -> List[Path]:
    """Collect all files that should go into the archive.

    Args:
        pool_root: Pool root directory.
        dataset_ids: If specified, only include these datasets.
        subject_ids: If specified, only include these subjects (within selected
            datasets). Format: bare ID like ``"S01"`` or qualified
            ``"dataset_id/S01"``.
        since: ISO date string; only include datasets imported after this date.

    Returns:
        List of absolute file paths.
    """
    files = []

    # Always include pool.yaml
    config_path = P.pool_config_path(pool_root)
    if config_path.exists():
        files.append(config_path)

    # Determine which datasets to include
    datasets_root = P.datasets_dir(pool_root)
    if not datasets_root.exists():
        return files

    available = sorted(d.name for d in datasets_root.iterdir() if d.is_dir())
    if dataset_ids:
        target_ds = [d for d in available if d in dataset_ids]
    else:
        target_ds = available

    # Filter by import date if requested
    if since:
        since_dt = datetime.datetime.fromisoformat(since)
        filtered = []
        for ds_id in target_ds:
            meta_path = P.dataset_meta_path(pool_root, ds_id)
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                ts = meta.get("import_timestamp")
                if ts:
                    try:
                        import_dt = datetime.datetime.fromisoformat(ts)
                        if import_dt < since_dt:
                            continue
                    except (ValueError, TypeError):
                        pass
            filtered.append(ds_id)
        target_ds = filtered

    # Parse subject filter: accept "S01" or "dataset_id/S01"
    _subject_filter: Optional[Dict[str, set]] = None
    if subject_ids:
        _subject_filter = {}  # ds_id → {subject_id, ...}
        for sid in subject_ids:
            if "/" in sid:
                ds_part, sub_part = sid.split("/", 1)
                _subject_filter.setdefault(ds_part, set()).add(sub_part)
            else:
                # Apply to all target datasets
                for ds_id in target_ds:
                    _subject_filter.setdefault(ds_id, set()).add(sid)

    # Collect files under each dataset
    for ds_id in target_ds:
        ds_dir = P.dataset_dir(pool_root, ds_id)

        # Always include dataset-level metadata (dataset.json etc.)
        for fname in os.listdir(ds_dir):
            fpath = ds_dir / fname
            if fpath.is_file() and not fname.startswith(".") and not fname.endswith(".lock"):
                files.append(fpath)

        # Determine which subjects to include
        subjects_base = ds_dir / "subjects"
        if not subjects_base.exists():
            continue

        allowed_subjects = None
        if _subject_filter and ds_id in _subject_filter:
            allowed_subjects = _subject_filter[ds_id]

        for sub_dir in sorted(subjects_base.iterdir()):
            if not sub_dir.is_dir():
                continue
            if allowed_subjects and sub_dir.name not in allowed_subjects:
                continue
            for root_dir, _dirs, filenames in os.walk(sub_dir):
                for fname in filenames:
                    fpath = Path(root_dir) / fname
                    if fname.startswith(".") or fname.endswith(".lock"):
                        continue
                    files.append(fpath)

    # Include stimuli and montages if present and not doing selective export
    if not dataset_ids:
        for subdir in [P.stimuli_dir(pool_root), P.montages_dir(pool_root)]:
            if subdir.exists():
                for root_dir, _dirs, filenames in os.walk(subdir):
                    for fname in filenames:
                        if not fname.startswith("."):
                            files.append(Path(root_dir) / fname)

    return files


def export_pool(
    pool_root: Path,
    output_path: Path,
    dataset_ids: Optional[List[str]] = None,
    subject_ids: Optional[List[str]] = None,
    since: Optional[str] = None,
    description: Optional[str] = None,
    compression: str = "gz",
) -> Dict[str, Any]:
    """Export a pool (or subset) as a `.napool` archive.

    Args:
        pool_root: Path to the pool directory.
        output_path: Destination path for the `.napool` file.
        dataset_ids: Export only these datasets (None = all).
        subject_ids: Export only these subjects. Accepts bare IDs like
            ``"S01"`` (applied to all selected datasets) or qualified
            ``"dataset_id/S01"``.
        since: Only include datasets imported after this ISO date.
        description: Human-readable archive description.
        compression: Compression type ('gz', 'bz2', 'xz', or '' for none).

    Returns:
        The manifest dictionary that was written into the archive.
    """
    pool_root = Path(pool_root).resolve()
    output_path = Path(output_path)

    # Ensure .napool extension
    if output_path.suffix != NAPOOL_EXTENSION:
        output_path = output_path.with_suffix(NAPOOL_EXTENSION)

    logger.info("Collecting files for export from %s ...", pool_root)
    files = _collect_pool_files(
        pool_root, dataset_ids=dataset_ids,
        subject_ids=subject_ids, since=since,
    )

    if not files:
        raise ValueError("No files to export. Check dataset_ids or since filter.")

    # Determine included dataset IDs
    datasets_root = P.datasets_dir(pool_root)
    included_ds = sorted(set(
        f.relative_to(pool_root).parts[1]
        for f in files
        if len(f.relative_to(pool_root).parts) > 1
        and f.relative_to(pool_root).parts[0] == "datasets"
    ))

    logger.info(
        "Building manifest: %d files, %d datasets",
        len(files), len(included_ds),
    )
    manifest = _build_manifest(pool_root, files, included_ds, description)

    # Write manifest to a temp file
    mode = f"w:{compression}" if compression else "w"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(manifest, tmp, indent=2, ensure_ascii=False)
        manifest_tmp = tmp.name

    try:
        logger.info("Writing archive to %s ...", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(str(output_path), mode) as tar:
            # Add manifest first
            tar.add(manifest_tmp, arcname="manifest.json")

            # Add pool files with relative paths
            for fpath in files:
                arcname = fpath.relative_to(pool_root).as_posix()
                tar.add(str(fpath), arcname=arcname)

        archive_size = output_path.stat().st_size
        logger.info(
            "Export complete: %s (%.1f MB, %d files, %d datasets)",
            output_path.name,
            archive_size / (1024 * 1024),
            len(files),
            len(included_ds),
        )
    finally:
        os.unlink(manifest_tmp)

    return manifest


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_pool(
    archive_path: Path,
    target_root: Path,
    verify: bool = True,
    merge: bool = True,
) -> Dict[str, Any]:
    """Import a `.napool` archive into a pool directory.

    Args:
        archive_path: Path to the `.napool` file.
        target_root: Target pool directory. Created if it doesn't exist.
        verify: If True, verify SHA-256 checksums after extraction.
        merge: If True, merge into existing pool. If False, require empty dir.

    Returns:
        The manifest from the archive.

    Raises:
        FileNotFoundError: If archive doesn't exist.
        ValueError: On integrity check failure.
    """
    archive_path = Path(archive_path).resolve()
    target_root = Path(target_root).resolve()

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Detect compression from file
    if str(archive_path).endswith(".napool"):
        mode = "r:*"  # auto-detect compression
    else:
        mode = "r:*"

    logger.info("Extracting %s → %s ...", archive_path.name, target_root)

    target_root.mkdir(parents=True, exist_ok=True)

    # Check if target is an existing pool
    existing_config = P.pool_config_path(target_root)
    is_existing_pool = existing_config.exists()

    if is_existing_pool and not merge:
        raise ValueError(
            f"Target {target_root} is an existing pool and merge=False. "
            "Use merge=True to merge archive contents into the existing pool."
        )

    # Extract to a temp directory first, then move files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        with tarfile.open(str(archive_path), mode) as tar:
            tar.extractall(tmp_path, filter="data")

        # Read manifest
        manifest_path = tmp_path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                "Invalid .napool archive: manifest.json not found."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        logger.info(
            "Archive: format v%s, schema v%s, %d files, %d datasets",
            manifest.get("archive_format_version", "?"),
            manifest.get("schema_version", "?"),
            manifest.get("n_files", 0),
            len(manifest.get("datasets", [])),
        )

        # Verify integrity
        if verify:
            logger.info("Verifying file integrity ...")
            errors = _verify_manifest(tmp_path, manifest, strict=True)
            if errors:
                for e in errors[:10]:
                    logger.error("  %s", e)
                raise ValueError(
                    f"Integrity check failed: {len(errors)} error(s). "
                    "Use verify=False to skip."
                )
            logger.info("Integrity check passed: all %d files OK.", len(manifest["files"]))

        # Move files to target
        # If merging into existing pool, skip pool.yaml if it already exists
        for entry in manifest["files"]:
            rel_path = entry["path"]
            src = tmp_path / rel_path
            dst = target_root / rel_path

            if is_existing_pool and rel_path == "pool.yaml":
                logger.debug("Skipping pool.yaml (existing pool)")
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)

            # For HDF5/binary files, use binary copy; for text, text copy
            import shutil
            if dst.exists():
                logger.debug("Overwriting: %s", rel_path)
            shutil.copy2(str(src), str(dst))

        # If new pool, also copy pool.yaml
        if not is_existing_pool:
            src_config = tmp_path / "pool.yaml"
            if src_config.exists():
                import shutil
                shutil.copy2(str(src_config), str(existing_config))

        # Create standard directories if they don't exist
        P.datasets_dir(target_root).mkdir(exist_ok=True)
        P.stimuli_dir(target_root).mkdir(exist_ok=True)
        P.montages_dir(target_root).mkdir(exist_ok=True)

    logger.info(
        "Import complete: %d datasets merged into %s",
        len(manifest.get("datasets", [])), target_root,
    )
    return manifest
