"""Schema Migration: versioned migrations for the NeuroAtom pool format.

Design:
- Each pool has a schema version in pool.yaml
- Migrations are registered functions that upgrade from version N to N+1
- ``migrate()`` applies all pending migrations in sequence
- Migrations must be idempotent (safe to re-run)

Version history:
    0.1.0 — Initial release
    0.2.0 — Indexer records ``jsonl_byte_offset`` for O(1) atom lookups.
            Migration auto-reindexes pools created with 0.1.0.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Current schema version
CURRENT_SCHEMA_VERSION = "0.2.0"

# Registry: (from_version, to_version) → migration function
_MIGRATIONS: Dict[Tuple[str, str], Callable[[Path], None]] = {}


def register_migration(
    from_version: str, to_version: str
) -> Callable:
    """Decorator to register a migration function.

    Usage:
        @register_migration("0.1.0", "0.2.0")
        def migrate_0_1_to_0_2(pool_root: Path) -> None:
            # ... perform migration ...
    """
    def decorator(func: Callable[[Path], None]) -> Callable:
        _MIGRATIONS[(from_version, to_version)] = func
        logger.debug("Registered migration: %s → %s", from_version, to_version)
        return func
    return decorator


def get_pool_version(pool_root: Path) -> str:
    """Read the schema version from pool.yaml.

    Returns CURRENT_SCHEMA_VERSION if no version is recorded.
    """
    config_path = pool_root / "pool.yaml"
    if not config_path.exists():
        return CURRENT_SCHEMA_VERSION

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config.get("schema_version", "0.1.0")


def set_pool_version(pool_root: Path, version: str) -> None:
    """Write the schema version to pool.yaml."""
    config_path = pool_root / "pool.yaml"

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    config["schema_version"] = version

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def _build_migration_chain(
    from_version: str, to_version: str
) -> List[Tuple[str, str]]:
    """Build the ordered chain of migrations from from_version to to_version.

    Uses BFS over the migration graph to find the shortest path.
    """
    if from_version == to_version:
        return []

    # Build adjacency list
    graph: Dict[str, List[str]] = {}
    for (fv, tv) in _MIGRATIONS:
        if fv not in graph:
            graph[fv] = []
        graph[fv].append(tv)

    # BFS
    from collections import deque
    queue = deque([(from_version, [])])
    visited = {from_version}

    while queue:
        current, path = queue.popleft()
        if current == to_version:
            return path

        for next_ver in graph.get(current, []):
            if next_ver not in visited:
                visited.add(next_ver)
                queue.append((next_ver, path + [(current, next_ver)]))

    return []


def needs_migration(pool_root: Path) -> bool:
    """Check if the pool needs a schema migration."""
    current = get_pool_version(pool_root)
    return current != CURRENT_SCHEMA_VERSION


def migrate(
    pool_root: Path,
    target_version: Optional[str] = None,
    dry_run: bool = False,
) -> List[str]:
    """Apply all pending migrations to bring pool to target version.

    Args:
        pool_root: Path to pool root directory.
        target_version: Target schema version. None = CURRENT_SCHEMA_VERSION.
        dry_run: If True, report migrations but don't apply them.

    Returns:
        List of migration descriptions that were (or would be) applied.
    """
    if target_version is None:
        target_version = CURRENT_SCHEMA_VERSION

    current = get_pool_version(pool_root)
    if current == target_version:
        logger.info("Pool is already at version %s. No migration needed.", current)
        return []

    chain = _build_migration_chain(current, target_version)
    if not chain:
        if current != target_version:
            logger.warning(
                "No migration path found from %s to %s.", current, target_version
            )
        return []

    applied = []
    for from_ver, to_ver in chain:
        desc = f"{from_ver} → {to_ver}"
        migration_func = _MIGRATIONS.get((from_ver, to_ver))

        if migration_func is None:
            logger.error("Migration function for %s not found!", desc)
            break

        if dry_run:
            logger.info("[DRY RUN] Would apply migration: %s", desc)
            applied.append(f"[dry-run] {desc}")
        else:
            logger.info("Applying migration: %s", desc)
            try:
                migration_func(pool_root)
                set_pool_version(pool_root, to_ver)
                applied.append(desc)
                logger.info("Migration %s complete.", desc)
            except Exception as e:
                logger.error("Migration %s FAILED: %s", desc, e)
                raise RuntimeError(
                    f"Migration {desc} failed. Pool may be in an inconsistent state. "
                    f"Error: {e}"
                ) from e

    return applied


def list_available_migrations() -> List[Tuple[str, str]]:
    """List all registered migration steps."""
    return sorted(_MIGRATIONS.keys())


# ---------------------------------------------------------------------------
# Registered migrations
# ---------------------------------------------------------------------------


@register_migration("0.1.0", "0.2.0")
def migrate_0_1_to_0_2(pool_root: Path) -> None:
    """Populate the ``jsonl_byte_offset`` column for atoms indexed under 0.1.0.

    The 0.2.0 indexer captures each atom's byte offset in its run's JSONL,
    enabling O(1) random reads in the assembler. Pools indexed under 0.1.0
    have NULL offsets; the runtime falls back to linear scans, but that's
    slow. This migration re-runs the indexer so future reads are fast.

    Idempotent: re-running on an already-migrated pool is harmless — the
    indexer's incremental path detects unchanged JSONL hashes and skips.
    """
    # Avoid circular import at module load time.
    from neuroatom.index.indexer import Indexer
    from neuroatom.storage.pool import Pool

    if not (pool_root / "pool.yaml").exists():
        logger.info(
            "Pool at %s has no pool.yaml — skipping byte-offset reindex.",
            pool_root,
        )
        return

    pool = Pool.open(pool_root)
    indexer = Indexer(pool)
    try:
        n = indexer.reindex_all()
        logger.info(
            "0.1.0→0.2.0: re-indexed %d atoms to populate byte offsets.", n,
        )
    finally:
        indexer.close()
