"""Example: Load assembly pipeline from YAML config.

Demonstrates reproducible experiment configuration using YAML files
that can be version-controlled alongside your experiment code.

Usage:
    python examples/yaml_pipeline.py

Prerequisites:
    - A pool with BCI IV 2a data imported and indexed.
    - Set NEUROATOM_POOL_DIR env var or adjust POOL_DIR below.
"""

import os
from pathlib import Path

from neuroatom import AssemblyRecipe, DatasetAssembler, Indexer, Pool

# ── Configure ─────────────────────────────────────────────────────────
POOL_DIR = Path(os.environ.get("NEUROATOM_POOL_DIR", "data/pool"))
RECIPE_PATH = Path("examples/configs/mi_4class_bci.yaml")


def main():
    # Load recipe from YAML — one line, fully reproducible
    recipe = AssemblyRecipe.from_yaml(RECIPE_PATH)
    print(f"Loaded recipe: {recipe.recipe_id}")
    print(f"  Query: {recipe.query}")
    print(f"  Sampling rate: {recipe.target_sampling_rate} Hz")
    print(f"  Filter: {recipe.filter_band} Hz")
    print(f"  Labels: {[l.output_key for l in recipe.label_fields]}")

    # Assemble
    pool = Pool(POOL_DIR)
    indexer = Indexer(pool)
    assembler = DatasetAssembler(pool, indexer)
    result = assembler.assemble(recipe)

    print(f"\nAssembly result:")
    print(f"  Train: {len(result.train_samples)}")
    print(f"  Val:   {len(result.val_samples)}")
    print(f"  Test:  {len(result.test_samples)}")

    # Save the recipe used (for experiment tracking)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    recipe.to_yaml(output_dir / "recipe_used.yaml")
    print(f"\nRecipe saved to {output_dir / 'recipe_used.yaml'}")

    indexer.close()


if __name__ == "__main__":
    main()
