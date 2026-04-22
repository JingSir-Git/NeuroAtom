# Changelog

All notable changes to NeuroAtom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2025-04-22

Initial public release.

### Added

#### Core Data Model
- `Atom` — self-contained EEG data unit with signal reference, temporal context,
  channel snapshot, annotations, quality info, and processing provenance
- Discriminated-union annotation system (7 subtypes: Categorical, Numeric, Text,
  Temporal, Spatial, Binary, Composite)
- `AtomRelation` for inter-atom relationships (sequential, paired, cross-modal)
- `ProcessingHistory` with full provenance chain
- `QualityInfo` with per-channel status and auto-QC flags

#### Storage
- HDF5 shard manager with per-run smart sharding (auto-split at 200 MB)
- JSONL per-run metadata (source of truth) + SQLite WAL query accelerator
- `Pool` manager for creating, opening, and managing atom pools
- Thread-safe file locking via `filelock`
- Schema migration framework with BFS chain builder

#### Importers (12 format adapters, 9 real-data validated)
- **BCI Competition IV 2a** — 9 subjects, 4-class motor imagery (.mat)
- **PhysioNet MI** — 109 subjects, motor movement/imagery (EDF, 64 ch @ 160 Hz)
- **KUL AAD** — 16 subjects, auditory attention detection (.mat, 64 ch @ 128 Hz)
- **DTU AAD** — 18 subjects, auditory attention preprocessed (.mat, 66 ch @ 64 Hz)
- **SEED-V** — 16 subjects, 5-emotion recognition (Neuroscan .cnt, 62 ch @ 1000 Hz)
- **Zuco 2.0** — 18 subjects, task-specific reading (EEGLAB HDF5, 105 ch @ 500 Hz)
- **CCEP-COREG** — 36 subjects, cortico-cortical evoked potentials (BIDS .npy, EEG+sEEG)
- **ChineseEEG-2** — 10 subjects, passive listening + reading aloud (BIDS BrainVision, 128 ch, sentence-level epochs)
- **MNE Generic** — universal fallback for any MNE-readable format
- **BIDS Generic** — auto-traversal of any BIDS-compliant dataset
- **EEGLAB** — .set/.fdt via MNE
- **MOABB Bridge** — access to 30+ public BCI datasets via MOABB
- 15 pre-built YAML task configurations

#### Atomizers
- `TrialAtomizer` — extract trials from event markers
- `EventAtomizer` — event-locked epoch extraction
- `WindowAtomizer` — sliding/tumbling window decomposition

#### Assembly Pipeline
- Declarative `AssemblyRecipe` for reproducible dataset construction
- Full processing chain: Unit Standardize → Re-reference → Channel Map →
  Bandpass Filter → Resample → Baseline Correct → Normalize → Pad/Crop
- Two-pass normalization (global / per-subject scope)
- Multi-label support with `LabelSpec`
- Subject-based, temporal, stratified, and predefined split strategies
- Cache provenance: `recipe.yaml` + `assembly_log.json` + `stats.json`
- Extra-field validation (`extra='forbid'`) on recipe models for catching YAML typos

#### Query Engine
- SQLite-backed indexer with per-field indices
- `QueryBuilder` with fluent API for dataset, subject, session, run,
  modality, channel, annotation, and quality filters
- Modality column with dedicated index for multi-modal queries

#### Cross-Pool Federation
- `FederatedPool` — unified query across multiple independent pools
- `FederatedQueryBuilder` — cross-pool query with atom ID deduplication
- `FederatedAssembler` — assemble from federated queries

#### DataLoader (PyTorch)
- `AtomDataset` — in-memory PyTorch Dataset from assembled samples
- `HDF5AtomDataset` — lazy-loading with worker-safe HDF5 access
- `PairedAtomDataset` — multi-modal paired samples
- `skip_none_collate`, `neuroatom_collate`, `dynamic_pad_collate`
- 6 augmentation transforms matching Pydantic AugmentationUnion types
- `worker_init_fn` for multi-worker DataLoader compatibility

#### Multi-Modal Support
- `MultiModalAssembler` — paired run-level assembly for EEG + sEEG
- `MultiModalRecipe` with per-modality pipeline configs
- Run-level pairing via `AtomRelation(cross_modal_paired_run)`

#### Import Provenance
- `log_import()` / `get_import_history()` for tracking import operations

#### CLI
- 9 commands: `init`, `info`, `stats`, `index`, `reindex`, `query`, `import`,
  `export` (atom_ids/jsonl/numpy/csv), `assemble`, `migrate`

#### Utilities
- Channel name standardization via alias table (`standard_1020.json`)
- `optional_deps.require()` for helpful import error messages
- Configs bundled inside package via `importlib.resources`

### Infrastructure
- `pyproject.toml` with `setuptools.build_meta`, optional dependency extras
  (`mne`, `bids`, `eeglab`, `torch`, `moabb`, `all`, `dev`)
- All data paths resolved from environment variables (no hardcoded paths)
- 225 tests (unit + E2E integration), 1 skipped, 0 failures
- Examples: `bci_iv_2a_full_pipeline.py`, `train_simple_cnn.py`,
  `federated_cross_dataset.py`, `yaml_pipeline.py`
- Benchmark: `benchmark_scale.py` — import/index/query/assembly/DataLoader throughput
