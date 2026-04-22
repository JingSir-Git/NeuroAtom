# NeuroAtom

**Universal EEG Resource Pool** — decompose heterogeneous EEG/iEEG datasets into standardized atomic units for unified ML pipeline integration.

NeuroAtom solves a fundamental problem in neural signal research: every dataset uses a different format, different channel layouts, different event coding, different preprocessing. Training ML models across datasets requires weeks of glue code. NeuroAtom replaces that with a unified data model, declarative import, and reproducible assembly into ML-ready tensors.

## Installation

```bash
pip install ".[all]"       # Everything (MNE + PyTorch + tools)
```

<details>
<summary>Install only what you need</summary>

```bash
pip install .              # Core only (pool, index, assembly)
pip install ".[mne]"       # + MNE-Python (EDF, BDF, GDF, BrainVision, CNT)
pip install ".[torch]"     # + PyTorch DataLoader integration
pip install ".[dev]"       # + pytest for development
```
</details>

## Quick Start

5 lines from raw data to a PyTorch DataLoader:

```python
import neuroatom as na

loader = na.quickload(
    "bci_comp_iv_2a",              # dataset name (auto-resolves config)
    data_path="data/A01T.mat",     # your data file
    subject="A01",
    batch_size=32,
    band=(0.5, 40.0),             # optional bandpass filter
)

for batch in loader:
    signals = batch["signal"]      # (32, 25, 1500) float32
    labels = batch["labels"]       # {"mi_class": tensor}
    break
```

Need train/test split? Add one argument:

```python
train_loader, test_loader = na.quickload(
    "bci_comp_iv_2a",
    data_path="data/A01T.mat",
    subject="A01",
    batch_size=32,
    split_test_ratio=0.2,
)
```

See [`examples/`](examples/) for complete scripts including a full [EEGNet training loop](examples/train_simple_cnn.py).

<details>
<summary><strong>Advanced Usage</strong> — full control over Pool, Indexer, Assembler</summary>

The `quickload` API is a thin wrapper around NeuroAtom's modular pipeline. When you need fine-grained control — custom queries, multi-dataset federation, channel mapping, per-subject normalization — use the lower-level API:

```python
from neuroatom import Pool, Indexer, QueryBuilder, DatasetAssembler
from neuroatom import AssemblyRecipe, LabelSpec, TaskConfig

# 1. Create pool and import
pool = Pool.create("./my_pool")
config = TaskConfig.builtin("bci_comp_iv_2a")  # built-in YAML config

from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
importer = BCICompIV2aImporter(pool, config)
importer.import_subject(mat_path="data/A01T.mat", subject_id="A01")

# 2. Index and query
indexer = Indexer(pool)
indexer.reindex_all()

qb = QueryBuilder(indexer.backend)
left_ids = qb.query_atom_ids({
    "dataset_id": "bci_comp_iv_2a",
    "annotations": [{"name": "mi_class", "value_in": ["left_hand"]}],
})

# 3. Assemble with full pipeline control
recipe = AssemblyRecipe(
    recipe_id="my_experiment",
    query={"dataset_id": "bci_comp_iv_2a"},
    target_sampling_rate=250.0,
    target_duration=4.0,
    filter_band=(0.5, 40.0),
    target_unit="uV",
    label_fields=[LabelSpec(annotation_name="mi_class", output_key="mi_class")],
)
result = DatasetAssembler(pool, indexer).assemble(recipe)

# 4. Feed to PyTorch
from neuroatom.loader.torch_dataset import AtomDataset
from torch.utils.data import DataLoader

loader = DataLoader(AtomDataset(result.train_samples), batch_size=32, shuffle=True)
```

</details>

## Supported Datasets & Importers

| Dataset | Importer | Scale | Notes |
|---------|----------|-------|-------|
| BCI IV 2a | `bci_comp_iv_2a` | 9 subj, 4-class MI | .mat, 25 ch @ 250 Hz |
| PhysioNet MI | `physionet_mi` | 109 subj | EDF, 64 ch @ 160 Hz |
| SEED-V | `seed_v` | 16 subj, 5-emotion | .cnt, 62 ch @ 1000 Hz |
| Zuco 2.0 | `zuco2` | 18 subj, reading | EEGLAB HDF5, 105 ch @ 500 Hz |
| CCEP-COREG | `ccep_bids_npy` | 36 subj, EEG+sEEG | BIDS .npy, multi-modal |
| ChineseEEG-2 | `chinese_eeg2` | 10 subj, listen+read | BIDS BrainVision, 128 ch |
| KUL / DTU AAD | `aad_mat` | 16+18 subj | .mat, auditory attention |
| Any MNE format | `mne_generic` | — | EDF, BDF, GDF, FIF, CNT, MFF |
| Any BIDS | `bids` | — | Auto-traversal |
| Any EEGLAB | `eeglab` | — | .set/.fdt |
| MOABB bridge | `moabb_bridge` | 30+ datasets | Via MOABB library |

Each dataset has a built-in YAML config: `TaskConfig.builtin("bci_comp_iv_2a")`.

## CLI

The CLI is for batch processing and scripting. The Python API is for interactive exploration and model training. Both operate on the same Pool and can be mixed freely.

```bash
neuroatom init ./my_pool                      # Create pool
neuroatom import ./my_pool data/ config.yaml  # Import dataset
neuroatom index ./my_pool                     # Build/update SQLite index
neuroatom query ./my_pool query.yaml          # Query atoms
neuroatom assemble ./my_pool recipe.yaml      # Assemble ML dataset
neuroatom info ./my_pool                      # Pool overview
neuroatom stats ./my_pool                     # Detailed statistics
neuroatom export ./my_pool query.yaml -o out  # Export to CSV/NumPy
```

## Requirements

- **Python** >= 3.10
- **Core**: numpy, scipy, h5py, pydantic (v2), pyyaml, filelock, click
- **Optional**: mne (format support), torch (DataLoader), moabb (dataset bridge)

## License

MIT
