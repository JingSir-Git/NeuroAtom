# NeuroAtom Architecture

A 10-minute tour of how raw EEG becomes a `DataLoader`.

## End-to-end flow

```
   ┌────────────────┐
   │ Raw EEG files  │  .mat / .edf / .cnt / .set / BIDS / ...
   └────────┬───────┘
            │  Importer.load_raw()            (format-specific)
            ▼
   ┌────────────────┐
   │ Raw + metadata │  channels, events, sampling rate
   └────────┬───────┘
            │  Atomizer.atomize()             (TrialAtomizer / EventAtomizer / WindowAtomizer)
            ▼
   ┌────────────────┐
   │     Atoms      │  Pydantic model — temporal + spatial + labels + provenance
   └────────┬───────┘
            │  ShardManager.write_atom_signal()  + AtomJSONLWriter
            ▼
   ┌──────────────────────────────────────────────────────┐
   │ Pool on disk                                         │
   │ ├── pool.yaml                       (config)         │
   │ ├── index.db                        (SQLite WAL)     │
   │ └── datasets/<ds_id>/subjects/<s>/sessions/<ses>/    │
   │       runs/<run>/                                    │
   │         ├── atoms.jsonl   (source of truth, append) │
   │         └── shard_*.h5    (signal arrays, ≤200 MB)   │
   └────────┬─────────────────────────────────────────────┘
            │  Indexer.reindex_all()   (records byte offsets!)
            ▼
   ┌────────────────┐
   │  SQLite index  │  per-field indices for QueryBuilder
   └────────┬───────┘
            │  QueryBuilder.query_atom_ids(filter dict)
            ▼
   ┌────────────────┐
   │    AtomIds     │
   └────────┬───────┘
            │  DatasetAssembler.assemble(AssemblyRecipe)
            │   Pipeline: unit-std → re-ref → ch-map → filter → resample
            │             → baseline → normalize → pad/crop → label-encode → split
            ▼
   ┌────────────────┐
   │  Train / Val   │  list[{"signal": ndarray, "labels": dict, "atom_id": ...}]
   │  / Test split  │
   └────────┬───────┘
            │  AtomDataset (in-memory) or HDF5AtomDataset (lazy)
            ▼
   ┌────────────────┐
   │  DataLoader    │  PyTorch — multi-worker safe via worker_init_fn
   └────────────────┘
```

## Module responsibilities

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **Models** | `core/` | Pydantic models. `Atom` is the smallest information-complete unit. Discriminated-union `AnnotationUnion` covers 7 annotation kinds. |
| **Storage** | `storage/pool.py` | Top-level CRUD for datasets/subjects/sessions/runs. Dataset-level FileLock. |
| | `storage/signal_store.py` | HDF5 shard manager. Per-run sharding @ 200 MB. Thread-local LRU read cache. |
| | `storage/metadata_store.py` | JSONL writer + reader (linear + random-access by byte offset). |
| | `storage/migration.py` | Versioned schema migrations (`0.1.0 → 0.2.0` registered). |
| **Index** | `index/sqlite_backend.py` | SQLite WAL. Atom + channel + annotation tables. `jsonl_byte_offset` enables O(1) atom lookup. |
| | `index/indexer.py` | Builds/updates the SQLite index from JSONL source of truth. Incremental via SHA-256 hashing. |
| | `index/query.py` | `QueryBuilder` — fluent filters (dataset, subject, modality, channels, annotations). |
| | `index/federation.py` | `FederatedPool` — multi-pool unified query. |
| **Importers** | `importers/base.py` | `BaseImporter` template-method pipeline. `TaskConfig` reads YAML (including `quickload:` metadata). Auto-acquires `dataset_lock`. |
| | `importers/<name>.py` | Format-specific subclass per dataset (BCI IV 2a, PhysioNet, SEED-V, OpenBMI, ChineseEEG-2, CCEP, Zuco 2.0, KUL/DTU AAD, …). |
| | `importers/registry.py` | Lazy importer discovery by dataset name. |
| | `importers/task_configs/` | One YAML per dataset; declares channels, events, label_field, subject_pattern. |
| **Atomizers** | `atomizer/` | `TrialAtomizer`, `EventAtomizer`, `WindowAtomizer`. Slice raw into atoms by anchors. |
| **Assembler** | `assembler/dataset_assembler.py` | Recipe-driven pipeline. Two-pass normalization for `global`/`per_subject` scopes. |
| | `assembler/{filter,resampler,…}.py` | One step per file. Composable. |
| | `assembler/federated_assembler.py` | Cross-pool assembly. |
| | `assembler/multimodal_assembler.py` | Paired EEG + sEEG/iEEG. |
| **Loader** | `loader/torch_dataset.py` | `AtomDataset` (in-memory, with size warning) / `HDF5AtomDataset` (lazy, worker-safe). |
| | `loader/collate.py` | `skip_none_collate`, `neuroatom_collate`, `dynamic_pad_collate`. |
| | `loader/transforms.py` | 6 augmentations matching `AugmentationUnion`. |
| **Quick API** | `quick.py` | One-call `quickload()` — import → index → assemble → DataLoader. Reads `TaskConfig.quickload_meta`. |
| **CLI** | `cli/main.py` | 9 commands (`init`, `import`, `index`, `query`, `assemble`, …). Same backend as Python API. |

## Three layers, three units of work

| Layer | Smallest unit | Persists as |
|-------|---------------|-------------|
| **Storage** | one Atom's JSON line + HDF5 group | `atoms.jsonl` + `shard_NNN.h5` |
| **Query**   | one atom row | SQLite row |
| **Train**   | one tensor dict | `{"signal": (C, T), "labels": {...}, ...}` |

The recipe layer is **purely declarative** — the same `AssemblyRecipe` YAML produces the same training tensors across runs and machines. Cached provenance (`recipe.yaml` + `assembly_log.json` + `stats.json`) lets you trace any model back to its exact data pipeline.

## Concurrency model

| Operation | Concurrency safety | Mechanism |
|-----------|--------------------|-----------|
| Read signals (assembly, training) | Lock-free | HDF5 read-only handles + SQLite WAL |
| DataLoader workers | Safe | `worker_init_fn` resets inherited handles |
| Write same dataset from N processes | Serialized | `pool.dataset_lock(dataset_id)` FileLock |
| Write different datasets from N processes | Parallel | Per-dataset FileLock |
| Concurrent import + train on same dataset | Not supported | Importer holds the dataset lock |

## Performance notes

- **Hot path read**: `ShardManager.static_read()` uses a thread-local LRU handle cache. Reading 1000 atoms from the same shard costs ~1 open + 1000 reads, not 1000 opens.
- **Atom lookup by ID**: `DatasetAssembler._load_atoms_by_ids` uses `jsonl_byte_offset` from SQLite for O(1) seek-and-read per atom. Pre-0.2.0 pools fall back to linear scan (warning is emitted; recommended to `migrate()`).
- **In-memory dataset cap**: `AtomDataset` warns above ~100k samples / 4 GiB. For larger pools, use `HDF5AtomDataset`.

## Adding a new dataset

The whole thing is one new file:

```
neuroatom/importers/my_dataset.py        # subclass BaseImporter
neuroatom/importers/task_configs/my_dataset.yaml   # declare paradigm + quickload meta
```

See [`writing_an_importer.md`](writing_an_importer.md) for the full guide.

## Where to look next

- `tests/test_concurrency_and_perf.py` — read-fast-path, dataset_lock, multi-worker DataLoader assertions.
- `tests/test_regression_v0_2_fixes.py` — the bug fixes that motivated 0.2.0.
- `examples/pretrain_eeg_mae.py` — end-to-end multi-dataset MAE pre-training.
- `examples/train_simple_cnn.py` — minimal EEGNet training loop.
