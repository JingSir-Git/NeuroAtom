"""Example: Cross-dataset federated query and assembly.

Demonstrates querying across two pools (BCI IV 2a + PhysioNet MI)
for cross-dataset transfer learning experiments.

Usage:
    python examples/federated_cross_dataset.py

Prerequisites:
    - Two pools with BCI and PhysioNet data already imported.
    - Set POOL_BCI and POOL_PHYSIONET paths below (or via env vars).
"""

import os
from pathlib import Path

from neuroatom import (
    AssemblyRecipe,
    FederatedAssembler,
    FederatedPool,
    FederatedQueryBuilder,
    Indexer,
    LabelSpec,
    Pool,
)
from neuroatom.index.import_log import get_import_history

# ── Configure pool paths ─────────────────────────────────────────────
POOL_BCI = Path(os.environ.get("NEUROATOM_POOL_BCI", "data/pool_bci"))
POOL_PHYSIONET = Path(os.environ.get("NEUROATOM_POOL_PHYSIONET", "data/pool_physionet"))


def main():
    # Open both pools
    pool_bci = Pool(POOL_BCI)
    pool_pn = Pool(POOL_PHYSIONET)
    idx_bci = Indexer(pool_bci)
    idx_pn = Indexer(pool_pn)

    # Create federated view
    fed = FederatedPool(
        [pool_bci, pool_pn],
        [idx_bci, idx_pn],
        tags=["bci_iv_2a", "physionet_mi"],
    )

    # Check atom counts
    counts = fed.count_atoms()
    print(f"Pool atom counts: {counts}")

    # ── Federated query ──────────────────────────────────────────────
    fqb = FederatedQueryBuilder(fed)

    # Query all motor imagery atoms across both pools
    all_mi_ids = fqb.query_atom_ids({
        "annotations": [{"name": "mi_class"}],
    })
    print(f"Total MI atoms across both pools: {len(all_mi_ids)}")

    # Per-pool breakdown
    per_pool = fqb.query_per_pool({
        "annotations": [{"name": "mi_class"}],
    })
    for tag, ids in per_pool.items():
        print(f"  {tag}: {len(ids)} atoms")

    # ── Federated assembly ───────────────────────────────────────────
    recipe = AssemblyRecipe(
        recipe_id="cross_dataset_mi",
        description="Cross-dataset motor imagery (BCI + PhysioNet)",
        query={"annotations": [{"name": "mi_class"}]},
        target_sampling_rate=160.0,  # resample to PhysioNet's rate
        target_duration=4.0,
        target_unit="uV",
        filter_band=(0.5, 40.0),
        label_fields=[
            LabelSpec(annotation_name="mi_class", output_key="mi_class"),
        ],
    )

    assembler = FederatedAssembler(fed)
    result = assembler.assemble(recipe)

    print(f"\nAssembly result:")
    print(f"  Train: {len(result.train_samples)}")
    print(f"  Val:   {len(result.val_samples)}")
    print(f"  Test:  {len(result.test_samples)}")

    # Check which pool each sample came from
    pool_tags = [s.get("pool_tag", "?") for s in result.train_samples]
    from collections import Counter
    print(f"  Train pool distribution: {dict(Counter(pool_tags))}")

    # ── Provenance ───────────────────────────────────────────────────
    print("\nImport history (BCI pool):")
    for entry in get_import_history(idx_bci, limit=3):
        print(f"  {entry['timestamp']}: {entry['importer_name']} → {entry['n_atoms']} atoms")

    idx_bci.close()
    idx_pn.close()


if __name__ == "__main__":
    main()
