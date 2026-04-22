"""BCI Competition IV 2a: Full Import → Query → Assemble → DataLoader Pipeline.

Demonstrates the complete NeuroAtom workflow end-to-end:
    1. Create a pool
    2. Import BCI IV 2a data (2 subjects × 2 runs each)
    3. Index the pool
    4. Query atoms by dataset and MI class
    5. Assemble into ML-ready format with normalization, padding, label encoding
    6. Wrap in a PyTorch DataLoader and print batch shapes

Usage:
    python examples/bci_iv_2a_full_pipeline.py [--data-dir /path/to/bci_data]

    # Or set via environment variable:
    NEUROATOM_BCI_IV_2A_DIR=/path/to/bci_data python examples/bci_iv_2a_full_pipeline.py

Prerequisites:
    - BCI IV 2a .mat files (A01T.mat ... A09T.mat) in the data directory
    - PyTorch installed:  pip install neuroatom[torch]
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# NeuroAtom imports
from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("example.bci_iv_2a")

# ─── Configuration ───────────────────────────────────────────────────────
TASK_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "neuroatom" / "importers" / "task_configs" / "bci_comp_iv_2a.yaml"
)
SUBJECTS = ["A01", "A02"]       # Import 2 subjects for demo
MAX_RUNS = 2                    # 2 labelled runs per subject
BATCH_SIZE = 16
TARGET_DURATION_S = 4.0         # Pad/crop all trials to 4s
TARGET_SRATE = 250.0            # Keep original 250 Hz
FILTER_BAND = (0.5, 40.0)      # Standard MI band
# ─────────────────────────────────────────────────────────────────────────


def _resolve_data_dir() -> Path:
    """Resolve BCI IV 2a data directory from CLI args or environment."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="BCI IV 2a full pipeline demo")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("NEUROATOM_BCI_IV_2A_DIR", ""),
        help="Path to BCI IV 2a .mat files (or set NEUROATOM_BCI_IV_2A_DIR)",
    )
    args = parser.parse_args()

    if not args.data_dir:
        print(
            "ERROR: Data directory not specified.\n"
            "Usage: python examples/bci_iv_2a_full_pipeline.py --data-dir /path/to/bci_data\n"
            "   Or: set NEUROATOM_BCI_IV_2A_DIR environment variable."
        )
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory does not exist: {data_dir}")
        sys.exit(1)

    return data_dir


def main():
    DATA_DIR = _resolve_data_dir()
    total_start = time.time()

    # ── Step 1: Create pool ──────────────────────────────────────────────
    pool_dir = Path(tempfile.mkdtemp(prefix="neuroatom_bci_"))
    logger.info("Creating pool at %s", pool_dir)
    pool = Pool.create(pool_dir)

    # ── Step 2: Import subjects ──────────────────────────────────────────
    task_config = TaskConfig.from_yaml(TASK_CONFIG_PATH)
    importer = BCICompIV2aImporter(pool, task_config)

    t0 = time.time()
    total_atoms = 0
    for subject_id in SUBJECTS:
        mat_path = DATA_DIR / f"{subject_id}T.mat"
        if not mat_path.exists():
            logger.warning("File not found: %s — skipping", mat_path)
            continue

        results = importer.import_subject(
            mat_path=mat_path,
            subject_id=subject_id,
            max_runs=MAX_RUNS,
        )
        n = sum(len(r.atoms) for r in results)
        total_atoms += n
        logger.info("  %s: %d runs, %d atoms imported", subject_id, len(results), n)

    import_time = time.time() - t0
    logger.info("Import complete: %d atoms in %.1fs", total_atoms, import_time)

    if total_atoms == 0:
        logger.error("No atoms imported. Check DATA_DIR: %s", DATA_DIR)
        sys.exit(1)

    # ── Step 3: Index ────────────────────────────────────────────────────
    t0 = time.time()
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    index_time = time.time() - t0
    logger.info("Indexed %d atoms in %.2fs", n_indexed, index_time)

    # ── Step 4: Query ────────────────────────────────────────────────────
    qb = QueryBuilder(indexer.backend)

    # Count per class
    for cls in ["left_hand", "right_hand", "both_feet", "tongue"]:
        ids = qb.query_atom_ids({
            "dataset_id": "bci_comp_iv_2a",
            "annotations": [{"name": "mi_class", "value_in": [cls]}],
        })
        logger.info("  Class '%s': %d atoms", cls, len(ids))

    # ── Step 5: Assemble ─────────────────────────────────────────────────
    recipe = AssemblyRecipe(
        recipe_id="bci_iv_2a_demo",
        description="BCI IV 2a 4-class MI classification demo",
        query={
            "dataset_id": "bci_comp_iv_2a",
            "annotations": [{"name": "mi_class"}],  # Only labelled trials
        },
        target_sampling_rate=TARGET_SRATE,
        target_duration=TARGET_DURATION_S,
        filter_band=FILTER_BAND,
        target_unit="uV",
        normalization_method=NormalizationMethod.ZSCORE,
        normalization_scope=NormalizationScope.PER_ATOM,
        label_fields=[
            LabelSpec(
                annotation_name="mi_class",
                output_key="mi_class",
                encoding="ordinal",
            ),
        ],
        split_strategy=SplitStrategy.SUBJECT,
        split_config={
            "val_ratio": 0.0,
            "test_ratio": 0.5,
            "seed": 42,
        },
    )

    t0 = time.time()
    assembler = DatasetAssembler(pool, indexer)
    result = assembler.assemble(recipe, cache_dir=pool_dir / "cache")
    assemble_time = time.time() - t0

    logger.info(
        "Assembly: %d train, %d val, %d test (%.2fs)",
        len(result.train_samples), len(result.val_samples),
        len(result.test_samples), assemble_time,
    )
    logger.info("Assembly log: %s", result.assembly_log)

    # ── Step 6: DataLoader ───────────────────────────────────────────────
    import torch
    from torch.utils.data import DataLoader

    train_ds = AtomDataset(result.train_samples)
    test_ds = AtomDataset(result.test_samples)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Train DataLoader: %d batches", len(train_loader))
    logger.info("Test DataLoader:  %d batches", len(test_loader))

    # Print first batch shape
    for batch in train_loader:
        signal = batch["signal"]
        labels = batch["labels"]
        logger.info("─── First Training Batch ───")
        logger.info("  signal.shape:  %s  dtype=%s", signal.shape, signal.dtype)
        for k, v in labels.items():
            logger.info("  labels[%s].shape: %s  dtype=%s", k, v.shape, v.dtype)
        if "channel_mask" in batch:
            logger.info("  channel_mask.shape: %s", batch["channel_mask"].shape)
        if "time_mask" in batch:
            logger.info("  time_mask.shape: %s", batch["time_mask"].shape)

        # Signal statistics
        logger.info("  signal mean: %.3f  std: %.3f", signal.mean().item(), signal.std().item())
        logger.info("  signal range: [%.3f, %.3f]", signal.min().item(), signal.max().item())
        break

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("BCI IV 2a Full Pipeline Demo — Summary")
    print("=" * 60)
    print(f"  Pool:          {pool_dir}")
    print(f"  Subjects:      {SUBJECTS}")
    print(f"  Total atoms:   {total_atoms}")
    print(f"  Train samples: {len(result.train_samples)}")
    print(f"  Test samples:  {len(result.test_samples)}")
    print(f"  Batch shape:   ({BATCH_SIZE}, 25, {int(TARGET_DURATION_S * TARGET_SRATE)})")
    print(f"  Import time:   {import_time:.1f}s")
    print(f"  Index time:    {index_time:.2f}s")
    print(f"  Assembly time: {assemble_time:.2f}s")
    print(f"  Total time:    {total_time:.1f}s")
    print("=" * 60)

    indexer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
