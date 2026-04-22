"""Cross-pool federation: query and assemble across multiple pools.

Enables transparent cross-dataset training (e.g., BCI IV 2a + PhysioNet MI
for transfer learning) without requiring all data in a single pool.

The federation layer is conceptually simple: queries are dispatched to each
pool's index independently, and results are merged. The assembler loads
atoms from whichever pool owns them.

Usage::

    pool_a = Pool(Path("/data/pool_a"))
    pool_b = Pool(Path("/data/pool_b"))
    idx_a = Indexer(pool_a)
    idx_b = Indexer(pool_b)

    fed = FederatedPool([pool_a, pool_b], [idx_a, idx_b])
    qb = FederatedQueryBuilder(fed)

    # Query across all pools transparently
    ids = qb.query_atom_ids({"annotations": [{"name": "mi_class"}]})

    # Assemble across pools
    assembler = FederatedAssembler(fed)
    result = assembler.assemble(recipe)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from neuroatom.core.atom import Atom
from neuroatom.index.indexer import Indexer
from neuroatom.index.query import QueryBuilder
from neuroatom.storage.metadata_store import AtomJSONLReader
from neuroatom.storage.pool import Pool
from neuroatom.storage import paths as P

logger = logging.getLogger(__name__)


class PoolHandle:
    """A (Pool, Indexer) pair with a unique tag for provenance tracking."""

    def __init__(self, pool: Pool, indexer: Indexer, tag: Optional[str] = None):
        self.pool = pool
        self.indexer = indexer
        self.tag = tag or str(pool.root)

    def __repr__(self):
        return f"PoolHandle(tag='{self.tag}', root={self.pool.root})"


class FederatedPool:
    """Manages multiple (Pool, Indexer) pairs for cross-pool operations.

    Each pool retains its own index; the federation dispatches queries to
    all pools and merges results. Atom provenance is tracked so the correct
    pool is used for signal loading.

    Usage::

        from neuroatom import Pool, Indexer, FederatedPool

        pool_bci = Pool(Path("/data/bci_pool"))
        pool_pn  = Pool(Path("/data/physionet_pool"))
        idx_bci  = Indexer(pool_bci)
        idx_pn   = Indexer(pool_pn)

        fed = FederatedPool(
            [pool_bci, pool_pn],
            [idx_bci, idx_pn],
            tags=["bci", "physionet"],  # optional human-readable tags
        )

        # Check how many atoms each pool has
        print(fed.count_atoms())  # {"bci": 2592, "physionet": 12180}

    Note:
        Atom IDs are content-addressable SHA-256 hashes. Different datasets
        always produce different IDs, so there is **no collision risk** when
        federating pools with distinct datasets.
    """

    def __init__(
        self,
        pools: List[Pool],
        indexers: List[Indexer],
        tags: Optional[List[str]] = None,
    ):
        if len(pools) != len(indexers):
            raise ValueError(
                f"Number of pools ({len(pools)}) must equal number of indexers "
                f"({len(indexers)})."
            )
        if len(pools) == 0:
            raise ValueError("FederatedPool requires at least one pool.")

        _tags = tags or [str(p.root) for p in pools]
        if len(set(_tags)) != len(_tags):
            raise ValueError(
                f"Pool tags must be unique, got duplicates in: {_tags}"
            )

        self._handles: List[PoolHandle] = [
            PoolHandle(p, i, t) for p, i, t in zip(pools, indexers, _tags)
        ]
        # Map atom_id → PoolHandle (built lazily during queries)
        self._atom_pool_map: Dict[str, PoolHandle] = {}

        logger.info(
            "FederatedPool created with %d pools: %s",
            len(self._handles),
            [h.tag for h in self._handles],
        )

    @property
    def handles(self) -> List[PoolHandle]:
        return self._handles

    def resolve_pool(self, atom_id: str) -> Optional[PoolHandle]:
        """Look up which pool owns a given atom_id."""
        return self._atom_pool_map.get(atom_id)

    def register_atom(self, atom_id: str, handle: PoolHandle) -> None:
        """Record atom → pool mapping (called during query)."""
        self._atom_pool_map[atom_id] = handle

    def count_atoms(self) -> Dict[str, int]:
        """Count atoms per pool."""
        return {h.tag: h.indexer.backend.count_atoms() for h in self._handles}


class FederatedQueryBuilder:
    """Query across multiple pools with UNION semantics.

    Dispatches the same query to each pool's ``QueryBuilder``, collects and
    merges results. Atom provenance is tracked via the ``FederatedPool`` so
    that downstream assemblers load from the correct pool.

    **Atom ID safety:** NeuroAtom atom_ids are SHA-256 hashes of
    ``(dataset_id, subject_id, session_id, run_id, onset_sample,
    processing_hash)``. Different datasets *always* produce different
    atom_ids. If the same dataset is imported into multiple pools, the
    atom_ids will match — this is by design (content-addressable). In this
    case, dedup keeps the first pool's copy and logs a warning.

    Usage::

        fed = FederatedPool([pool_a, pool_b], [idx_a, idx_b])
        fqb = FederatedQueryBuilder(fed)

        # Query across both pools
        atom_ids = fqb.query_atom_ids({"dataset_id": "bci_comp_iv_2a"})

        # See per-pool breakdown
        per_pool = fqb.query_per_pool({"dataset_id": "bci_comp_iv_2a"})
    """

    def __init__(self, federation: FederatedPool):
        self._fed = federation

    def query_atom_ids(self, query: Dict[str, Any]) -> List[str]:
        """Execute a query across all pools and return merged atom IDs.

        Returns:
            Deduplicated list of atom_ids. If the same atom_id appears in
            multiple pools (same dataset imported twice), only the first
            pool's copy is kept and a warning is logged.
        """
        all_ids: List[str] = []
        per_pool_counts: Dict[str, int] = {}

        for handle in self._fed.handles:
            qb = QueryBuilder(handle.indexer.backend)
            ids = qb.query_atom_ids(query)
            per_pool_counts[handle.tag] = len(ids)

            for aid in ids:
                self._fed.register_atom(aid, handle)

            all_ids.extend(ids)

        # Deduplicate — detect overlapping atom_ids across pools
        seen: Dict[str, str] = {}  # atom_id → first pool tag
        unique_ids: List[str] = []
        n_overlaps = 0

        for aid in all_ids:
            if aid not in seen:
                handle = self._fed.resolve_pool(aid)
                seen[aid] = handle.tag if handle else "unknown"
                unique_ids.append(aid)
            else:
                n_overlaps += 1

        if n_overlaps > 0:
            logger.warning(
                "%d atom_ids appear in multiple pools (same data imported twice). "
                "Keeping first pool's copy for each. This is expected if the same "
                "dataset was imported into multiple pools. Per-pool counts: %s",
                n_overlaps, per_pool_counts,
            )

        logger.info(
            "Federated query: %d unique atoms from %d pools. Per-pool: %s",
            len(unique_ids), len(self._fed.handles), per_pool_counts,
        )
        return unique_ids

    def query_count(self, query: Dict[str, Any]) -> int:
        """Count matching atoms across all pools."""
        total = 0
        for handle in self._fed.handles:
            qb = QueryBuilder(handle.indexer.backend)
            total += qb.query_count(query)
        return total

    def query_per_pool(
        self, query: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Query each pool separately and return {tag: [atom_ids]}."""
        result = {}
        for handle in self._fed.handles:
            qb = QueryBuilder(handle.indexer.backend)
            ids = qb.query_atom_ids(query)
            result[handle.tag] = ids
            for aid in ids:
                self._fed.register_atom(aid, handle)
        return result


def load_federated_atoms(
    federation: FederatedPool,
    atom_ids: List[str],
) -> List[Atom]:
    """Load atoms from multiple pools, routing each atom to its owning pool.

    Args:
        federation: The FederatedPool with atom → pool mapping.
        atom_ids: List of atom IDs to load.

    Returns:
        List of loaded Atom objects.
    """
    # Group atom_ids by pool
    pool_groups: Dict[str, Tuple[PoolHandle, List[str]]] = {}
    unmapped: List[str] = []

    for aid in atom_ids:
        handle = federation.resolve_pool(aid)
        if handle is None:
            unmapped.append(aid)
            continue
        if handle.tag not in pool_groups:
            pool_groups[handle.tag] = (handle, [])
        pool_groups[handle.tag][1].append(aid)

    if unmapped:
        logger.warning(
            "%d atom_ids have no known pool. Run a federated query first to "
            "populate the atom→pool mapping. Unmapped: %s",
            len(unmapped), unmapped[:5],
        )

    all_atoms: List[Atom] = []

    for tag, (handle, ids) in pool_groups.items():
        conn = handle.indexer.backend.conn
        rows = conn.execute(
            "SELECT atom_id, dataset_id, subject_id, session_id, run_id FROM atoms "
            f"WHERE atom_id IN ({','.join('?' * len(ids))})",
            tuple(ids),
        ).fetchall()

        run_atoms: Dict[str, List[str]] = {}
        for r in rows:
            rk = f"{r['dataset_id']}|{r['subject_id']}|{r['session_id']}|{r['run_id']}"
            run_atoms.setdefault(rk, []).append(r["atom_id"])

        target_set = set(ids)

        for rk, rk_ids in run_atoms.items():
            parts = rk.split("|")
            ds_id, sub_id, ses_id, run_id = parts
            jsonl_path = P.atoms_jsonl_path(
                handle.pool.root, ds_id, sub_id, ses_id, run_id
            )
            reader = AtomJSONLReader(jsonl_path)
            for atom in reader.iter_atoms():
                if atom.atom_id in target_set:
                    all_atoms.append(atom)

    logger.info(
        "Loaded %d federated atoms from %d pools.",
        len(all_atoms), len(pool_groups),
    )
    return all_atoms
