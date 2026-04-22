"""NeuroAtom Performance Benchmark: BCI IV 2a (9 subjects) + PhysioNet (N subjects).

Measures:
    - Import throughput (atoms/s, MB/s)
    - Indexing speed (atoms/s)
    - Query latency (ms)
    - Assembly throughput (atoms/s)
    - DataLoader iteration speed (batches/s)

Usage:
    python examples/benchmark_scale.py \\
        --bci-dir /path/to/bci_data \\
        --physionet-dir /path/to/physionet \\
        --physionet-subjects 10

    # Or set environment variables:
    NEUROATOM_BCI_IV_2A_DIR=/path/to/bci
    NEUROATOM_PHYSIONET_DIR=/path/to/physionet

Prerequisites:
    - BCI IV 2a .mat files (A01T.mat ... A09T.mat)
    - PhysioNet eegmmidb (S001/ ... S109/)
    - PyTorch installed:  pip install neuroatom[torch]
"""

import argparse
import gc
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# NeuroAtom imports
from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
from neuroatom.importers.physionet_mi import PhysioNetMIImporter
from neuroatom.importers.base import TaskConfig
from neuroatom.storage.pool import Pool
from neuroatom.index.indexer import Indexer
from neuroatom.index.query import QueryBuilder
from neuroatom.assembler.dataset_assembler import DatasetAssembler
from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
from neuroatom.core.enums import (
    NormalizationMethod,
    NormalizationScope,
    SplitStrategy,
)
from neuroatom.loader.torch_dataset import AtomDataset

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)


# ─── Task config paths (resolved relative to package) ─────────────────────
TC_BCI = Path(__file__).resolve().parent.parent / "neuroatom" / "importers" / "task_configs" / "bci_comp_iv_2a.yaml"
TC_PHYSIONET = Path(__file__).resolve().parent.parent / "neuroatom" / "importers" / "task_configs" / "physionet_mi.yaml"


def format_rate(n: int, elapsed: float) -> str:
    """Format as 'N items in X.Xs (Y.Y items/s)'."""
    rate = n / elapsed if elapsed > 0 else float("inf")
    return f"{n} in {elapsed:.2f}s ({rate:.1f}/s)"


def benchmark_bci_import(pool: Pool) -> Tuple[int, float]:
    """Import all 9 BCI IV 2a subjects (6 labelled runs each)."""
    tc = TaskConfig.from_yaml(TC_BCI)
    importer = BCICompIV2aImporter(pool, tc)

    total_atoms = 0
    t0 = time.time()

    for i in range(1, 10):
        subject_id = f"A{i:02d}"
        mat_path = BCI_DIR / f"{subject_id}T.mat"
        if not mat_path.exists():
            continue
        results = importer.import_subject(mat_path=mat_path, subject_id=subject_id)
        n = sum(len(r.atoms) for r in results)
        total_atoms += n

    elapsed = time.time() - t0
    return total_atoms, elapsed


def benchmark_physionet_import(pool: Pool, n_subjects: int, max_runs: int) -> Tuple[int, float]:
    """Import N PhysioNet subjects with up to max_runs each."""
    tc = TaskConfig.from_yaml(TC_PHYSIONET)
    importer = PhysioNetMIImporter(pool, tc)

    total_atoms = 0
    t0 = time.time()

    for i in range(1, n_subjects + 1):
        subject_id = f"S{i:03d}"
        subject_dir = PHYSIONET_DIR / subject_id
        if not subject_dir.exists():
            continue
        results = importer.import_subject(
            subject_dir=subject_dir,
            subject_id=subject_id,
            paradigm="imagery",
            max_runs=max_runs,
        )
        n = sum(len(r.atoms) for r in results)
        total_atoms += n

    elapsed = time.time() - t0
    return total_atoms, elapsed


def benchmark_index(pool: Pool) -> Tuple[int, float]:
    """Build index over all atoms."""
    indexer = Indexer(pool)
    t0 = time.time()
    n = indexer.reindex_all()
    elapsed = time.time() - t0
    return n, elapsed, indexer


def benchmark_queries(indexer: Indexer) -> Dict[str, float]:
    """Run various queries and measure latency."""
    qb = QueryBuilder(indexer.backend)
    timings = {}

    # Query 1: All BCI atoms
    t0 = time.time()
    ids = qb.query_atom_ids({"dataset_id": "bci_comp_iv_2a"})
    timings["bci_all"] = (time.time() - t0) * 1000
    timings["bci_all_count"] = len(ids)

    # Query 2: BCI left_hand only
    t0 = time.time()
    ids = qb.query_atom_ids({
        "dataset_id": "bci_comp_iv_2a",
        "annotations": [{"name": "mi_class", "value_in": ["left_hand"]}],
    })
    timings["bci_left_hand"] = (time.time() - t0) * 1000
    timings["bci_left_hand_count"] = len(ids)

    # Query 3: All PhysioNet atoms
    t0 = time.time()
    ids = qb.query_atom_ids({"dataset_id": "physionet_mi"})
    timings["physionet_all"] = (time.time() - t0) * 1000
    timings["physionet_all_count"] = len(ids)

    # Query 4: PhysioNet by subject
    t0 = time.time()
    ids = qb.query_atom_ids({"dataset_id": "physionet_mi", "subject_id": "S001"})
    timings["physionet_s001"] = (time.time() - t0) * 1000
    timings["physionet_s001_count"] = len(ids)

    # Query 5: Cross-dataset MI class
    t0 = time.time()
    ids = qb.query_atom_ids({
        "annotations": [{"name": "mi_class", "value_in": ["left_hand"]}],
    })
    timings["cross_dataset_left_hand"] = (time.time() - t0) * 1000
    timings["cross_dataset_left_hand_count"] = len(ids)

    return timings


def benchmark_assembly(pool: Pool, indexer: Indexer, dataset_id: str) -> Tuple[int, float, object, Dict]:
    """Assemble a dataset and measure per-stage throughput.

    Returns:
        (n_atoms, total_elapsed, result, stage_timings)
    """
    recipe = AssemblyRecipe(
        recipe_id=f"bench_{dataset_id}",
        description=f"Benchmark assembly for {dataset_id}",
        query={
            "dataset_id": dataset_id,
            "annotations": [{"name": "mi_class"}],
        },
        target_sampling_rate=250.0,
        target_duration=4.0,
        target_unit="uV",
        normalization_method=NormalizationMethod.ZSCORE,
        normalization_scope=NormalizationScope.PER_ATOM,
        label_fields=[
            LabelSpec(annotation_name="mi_class", output_key="mi_class", encoding="ordinal"),
        ],
        split_strategy=SplitStrategy.SUBJECT,
        split_config={"val_ratio": 0.1, "test_ratio": 0.2, "seed": 42},
    )

    stage_timings: Dict[str, float] = {}

    # Stage 1: Query
    from neuroatom.index.query import QueryBuilder as _QB
    qb = _QB(indexer.backend)
    t0 = time.time()
    atom_ids = qb.query_atom_ids(recipe.query)
    stage_timings["query"] = time.time() - t0

    # Full assembly (includes query again, but measures end-to-end)
    assembler = DatasetAssembler(pool, indexer)
    t_full_start = time.time()
    result = assembler.assemble(recipe)
    elapsed = time.time() - t_full_start

    n = len(result.train_samples) + len(result.val_samples) + len(result.test_samples)

    # Extract per-stage timings from assembly_log
    stage_timings["total"] = elapsed
    stage_timings["n_queried"] = result.assembly_log.get("n_queried", 0)
    stage_timings["n_processed"] = result.assembly_log.get("n_processed", 0)

    return n, elapsed, result, stage_timings


def benchmark_dataloader(result, batch_size: int = 32, n_batches: int = 50) -> Dict[str, float]:
    """Iterate through DataLoader and measure throughput.

    Reports:
        - cold_start_ms: Time for the very first batch (includes Dataset init overhead)
        - batches_per_s: Steady-state throughput after warm-up
    """
    import torch
    from torch.utils.data import DataLoader

    ds = AtomDataset(result.train_samples + result.test_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    cold_start_ms = None
    t0 = time.time()
    count = 0
    for batch in loader:
        _ = batch["signal"]
        count += 1
        if count == 1:
            cold_start_ms = (time.time() - t0) * 1000
        if count >= n_batches:
            break
    elapsed = time.time() - t0
    rate = count / elapsed if elapsed > 0 else float("inf")

    return {
        "batches_per_s": rate,
        "cold_start_ms": cold_start_ms or 0.0,
        "n_batches": count,
        "elapsed_s": elapsed,
    }


def main():
    import os

    parser = argparse.ArgumentParser(description="NeuroAtom Scale Benchmark")
    parser.add_argument("--bci-dir", type=str,
                        default=os.environ.get("NEUROATOM_BCI_IV_2A_DIR", ""),
                        help="Path to BCI IV 2a .mat files (or NEUROATOM_BCI_IV_2A_DIR)")
    parser.add_argument("--physionet-dir", type=str,
                        default=os.environ.get("NEUROATOM_PHYSIONET_DIR", ""),
                        help="Path to PhysioNet eegmmidb (or NEUROATOM_PHYSIONET_DIR)")
    parser.add_argument("--physionet-subjects", type=int, default=10,
                        help="Number of PhysioNet subjects to import (max 109)")
    parser.add_argument("--physionet-max-runs", type=int, default=3,
                        help="Max MI runs per PhysioNet subject")
    args = parser.parse_args()

    global BCI_DIR, PHYSIONET_DIR
    BCI_DIR = Path(args.bci_dir) if args.bci_dir else None
    PHYSIONET_DIR = Path(args.physionet_dir) if args.physionet_dir else None

    if not BCI_DIR and not PHYSIONET_DIR:
        print(
            "ERROR: No data directories specified.\n"
            "Usage: python examples/benchmark_scale.py --bci-dir /path --physionet-dir /path\n"
            "   Or: set NEUROATOM_BCI_IV_2A_DIR / NEUROATOM_PHYSIONET_DIR env vars."
        )
        sys.exit(1)

    pool_dir = Path(tempfile.mkdtemp(prefix="neuroatom_bench_"))
    logger.info("Benchmark pool: %s", pool_dir)
    pool = Pool.create(pool_dir)

    results = {}

    # ── BCI IV 2a Import ─────────────────────────────────────────────
    if BCI_DIR and BCI_DIR.exists():
        logger.info("═══ BCI IV 2a: Import (9 subjects × 6 runs) ═══")
        n, t = benchmark_bci_import(pool)
        results["bci_import"] = {"atoms": n, "time_s": t}
        logger.info("  %s", format_rate(n, t))
    else:
        logger.warning("BCI data not found at %s", BCI_DIR)

    # ── PhysioNet Import ─────────────────────────────────────────────
    if PHYSIONET_DIR and PHYSIONET_DIR.exists():
        logger.info("═══ PhysioNet MI: Import (%d subjects) ═══", args.physionet_subjects)
        n, t = benchmark_physionet_import(pool, args.physionet_subjects, args.physionet_max_runs)
        results["physionet_import"] = {"atoms": n, "time_s": t}
        logger.info("  %s", format_rate(n, t))
    else:
        logger.warning("PhysioNet data not found at %s", PHYSIONET_DIR)

    # ── Index ────────────────────────────────────────────────────────────
    logger.info("═══ Index ═══")
    n, t, indexer = benchmark_index(pool)
    results["index"] = {"atoms": n, "time_s": t}
    logger.info("  %s", format_rate(n, t))

    # ── Queries ──────────────────────────────────────────────────────────
    logger.info("═══ Queries ═══")
    query_results = benchmark_queries(indexer)
    results["queries"] = query_results
    for k, v in query_results.items():
        if k.endswith("_count"):
            continue
        count_key = f"{k}_count"
        count = query_results.get(count_key, "?")
        logger.info("  %-30s %6.1fms  (%s results)", k, v, count)

    # ── Assembly (BCI) ───────────────────────────────────────────────────
    if "bci_import" in results:
        logger.info("═══ Assembly: BCI IV 2a ═══")
        n, t, asm_result_bci, stage_t = benchmark_assembly(pool, indexer, "bci_comp_iv_2a")
        results["bci_assembly"] = {"atoms": n, "time_s": t, "stages": stage_t}
        logger.info("  %s", format_rate(n, t))
        logger.info("    query: %.1fms  (%d atoms)", stage_t["query"] * 1000, stage_t["n_queried"])

        # DataLoader
        logger.info("═══ DataLoader: BCI IV 2a ═══")
        dl_stats = benchmark_dataloader(asm_result_bci)
        results["bci_dataloader"] = dl_stats
        logger.info("  %.1f batches/s  (cold start: %.1fms)", dl_stats["batches_per_s"], dl_stats["cold_start_ms"])

    # ── Assembly (PhysioNet) ─────────────────────────────────────────────
    if "physionet_import" in results and results["physionet_import"]["atoms"] > 0:
        logger.info("═══ Assembly: PhysioNet MI ═══")
        n, t, asm_result_phys, stage_t = benchmark_assembly(pool, indexer, "physionet_mi")
        results["physionet_assembly"] = {"atoms": n, "time_s": t, "stages": stage_t}
        logger.info("  %s", format_rate(n, t))
        logger.info("    query: %.1fms  (%d atoms)", stage_t["query"] * 1000, stage_t["n_queried"])

        logger.info("═══ DataLoader: PhysioNet MI ═══")
        dl_stats = benchmark_dataloader(asm_result_phys)
        results["physionet_dataloader"] = dl_stats
        logger.info("  %.1f batches/s  (cold start: %.1fms)", dl_stats["batches_per_s"], dl_stats["cold_start_ms"])

    # ── Final Report ─────────────────────────────────────────────────────
    indexer.close()

    print("\n" + "=" * 70)
    print("NeuroAtom Scale Benchmark — Results")
    print("=" * 70)

    total_atoms = results.get("bci_import", {}).get("atoms", 0) + \
                  results.get("physionet_import", {}).get("atoms", 0)
    total_import_time = results.get("bci_import", {}).get("time_s", 0) + \
                        results.get("physionet_import", {}).get("time_s", 0)

    print(f"\n  Total atoms imported:  {total_atoms}")
    print(f"  Total import time:     {total_import_time:.1f}s")
    if total_import_time > 0:
        print(f"  Import throughput:     {total_atoms / total_import_time:.1f} atoms/s")

    idx = results.get("index", {})
    if idx:
        print(f"\n  Indexing:              {format_rate(idx['atoms'], idx['time_s'])}")

    for label, key in [("BCI IV 2a", "bci_assembly"), ("PhysioNet MI", "physionet_assembly")]:
        asm = results.get(key, {})
        if asm:
            print(f"\n  Assembly ({label}):  {format_rate(asm['atoms'], asm['time_s'])}")

    for label, key in [("BCI IV 2a", "bci_dataloader"), ("PhysioNet MI", "physionet_dataloader")]:
        dl = results.get(key, {})
        if dl:
            print(f"  DataLoader ({label}): {dl['batches_per_s']:.1f} batches/s")
            print(f"    Cold-start first batch: {dl['cold_start_ms']:.1f}ms")

    print(f"\n  Pool path: {pool_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
