# NeuroAtom

**Universal EEG Resource Pool** — decompose heterogeneous EEG/iEEG datasets into standardized atomic units for unified ML pipeline integration.

NeuroAtom solves a fundamental problem in neural signal research: every dataset uses a different format, different channel layouts, different event coding, different preprocessing. Training ML models across datasets requires weeks of glue code. NeuroAtom replaces that with a unified data model, declarative import, and reproducible assembly into ML-ready tensors.

## Installation

```bash
# Core (pool, index, assembly — no heavy dependencies)
pip install .

# With MNE-Python support (EDF, BDF, GDF, BrainVision, CNT, BIDS)
pip install ".[mne]"

# With PyTorch DataLoader integration
pip install ".[torch]"

# Everything
pip install ".[all]"

# Development (adds pytest)
pip install ".[dev]"
```

## Quick Start

```python
from pathlib import Path
from neuroatom import Pool, Indexer, QueryBuilder, DatasetAssembler, AssemblyRecipe, LabelSpec
from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
from neuroatom.importers.base import TaskConfig

# 1. Create pool and import
pool = Pool.create("./my_pool")
config = TaskConfig.from_yaml("neuroatom/importers/task_configs/bci_comp_iv_2a.yaml")
importer = BCICompIV2aImporter(pool, config)
importer.import_subject(mat_path="data/A01T.mat", subject_id="A01")

# 2. Index and query
indexer = Indexer(pool)
indexer.reindex_all()
qb = QueryBuilder(indexer.backend)
ids = qb.query_atom_ids({"dataset_id": "bci_comp_iv_2a"})  # 288 atoms

# 3. Assemble into ML-ready dataset
recipe = AssemblyRecipe(
    recipe_id="demo",
    query={"dataset_id": "bci_comp_iv_2a"},
    target_sampling_rate=250.0,
    filter_band=(0.5, 40.0),
    target_unit="uV",
    label_fields=[LabelSpec(annotation_name="mi_class", output_key="mi_class")],
)
result = DatasetAssembler(pool, indexer).assemble(recipe)

# 4. Feed to PyTorch
from neuroatom.loader.torch_dataset import AtomDataset
from torch.utils.data import DataLoader

loader = DataLoader(AtomDataset(result.train_samples), batch_size=32, shuffle=True)
for batch in loader:
    signals = batch["signal"]   # (B, C, T) float32
    labels = batch["labels"]    # dict of label tensors
    break
```

See [`examples/`](examples/) for complete working scripts including a full training loop with EEGNet.

## Supported Formats & Importers

| Format | Importer | Datasets | Notes |
|--------|----------|----------|-------|
| MNE-native | `mne_generic` | Any | EDF, BDF, GDF, FIF, BrainVision, CNT, MFF |
| MATLAB .mat | `mat` | Any | v5 + v7.3 (HDF5), auto-detect keys |
| BCI IV 2a | `bci_comp_iv_2a` | 9 subj, 4-class MI | .mat struct-of-runs |
| PhysioNet MI | `physionet_mi` | 109 subj, motor imagery/execution | EDF, 64 ch @ 160 Hz |
| SEED-V | `seed_v` | 16 subj, 5-emotion | Neuroscan .cnt, 62 ch @ 1000 Hz |
| Zuco 2.0 | `zuco2` | 18 subj, reading EEG | EEGLAB HDF5, 105 ch @ 500 Hz |
| CCEP-COREG | `ccep_bids_npy` | 36 subj, EEG+sEEG | BIDS .npy, multi-modal |
| ChineseEEG-2 | `chinese_eeg2` | 10 subj, listening+reading | BIDS BrainVision, 128 ch, sentence-level |
| KUL / DTU AAD | `aad_mat` | 16+18 subj, auditory attention | .mat struct-of-trials |
| BIDS generic | `bids` | Any BIDS dataset | Auto-traversal |
| EEGLAB | `eeglab` | Any .set/.fdt | Via MNE |
| MOABB | `moabb_bridge` | 30+ public BCI datasets | Via MOABB library |

Each importer has a matching YAML task config in `neuroatom/importers/task_configs/`.

## CLI

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
