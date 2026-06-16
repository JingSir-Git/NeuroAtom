"""End-to-end cross-subject AAD pipeline test.

Validates the COMPLETE pipeline for KUL + DTU Auditory Attention Detection:

  Import (all subjects, 1 trial each)
    → Index (SQLite)
      → Query (cross-subject)
        → Assemble (channel mapping + resampling + normalization)
          → Split (subject-level, no leakage)
            → DataLoader (PyTorch tensors)

Requires:
    C:\\Data\\KUL\\S1.mat ... S16.mat
    C:\\Data\\DTU\\DATA_preproc\\S1_data_preproc.mat ... S18_data_preproc.mat
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════
# Data discovery
# ═══════════════════════════════════════════════════════════════════

KUL_DIR = Path(os.environ.get("NEUROATOM_KUL_DIR", r"C:\Data\KUL"))
DTU_DIR = Path(os.environ.get("NEUROATOM_DTU_DIR", r"C:\Data\DTU"))
DTU_PREPROC_DIR = DTU_DIR / "DATA_preproc"

_kul_mats = sorted(KUL_DIR.glob("S*.mat")) if KUL_DIR.exists() else []
_dtu_mats = sorted(DTU_PREPROC_DIR.glob("S*_data_preproc.mat")) if DTU_PREPROC_DIR.exists() else []

pytestmark = pytest.mark.skipif(
    len(_kul_mats) < 16 or len(_dtu_mats) < 18,
    reason=f"Need 16 KUL + 18 DTU .mat files "
           f"(found {len(_kul_mats)} KUL, {len(_dtu_mats)} DTU)",
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def pool_dir():
    """Shared temp pool for the entire module (import is expensive)."""
    d = tempfile.mkdtemp(prefix="neuroatom_aad_xsubj_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def imported_pool(pool_dir):
    """Import 1 trial per subject for all KUL (16) and DTU (18) subjects.

    This fixture runs once per module — all tests share the same pool.
    """
    from neuroatom.importers.aad_mat import AADImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool

    logging.basicConfig(level=logging.INFO, force=True)

    pool = Pool.create(pool_dir)

    # ── KUL: 16 subjects ──────────────────────────────────────────
    kul_config = TaskConfig.builtin("kul_aad")
    kul_importer = AADImporter(pool, kul_config)

    for mat_path in _kul_mats:
        subject_id = mat_path.stem  # S1, S2, ...
        kul_importer.import_subject(
            mat_path=mat_path,
            subject_id=subject_id,
            session_id="ses-01",
            format_hint="kul",
            max_trials=1,
        )

    # ── DTU: 18 subjects ──────────────────────────────────────────
    dtu_config = TaskConfig.builtin("dtu_aad")
    dtu_importer = AADImporter(pool, dtu_config)

    for mat_path in _dtu_mats:
        # S1_data_preproc.mat → S1
        subject_id = mat_path.stem.split("_")[0]
        dtu_importer.import_subject(
            mat_path=mat_path,
            subject_id=subject_id,
            session_id="ses-01",
            format_hint="dtu_preproc",
            max_trials=1,
        )

    return pool


@pytest.fixture(scope="module")
def indexed_pool(imported_pool):
    """Index all atoms in the shared pool."""
    from neuroatom.index.indexer import Indexer

    indexer = Indexer(imported_pool)
    n = indexer.reindex_all()
    return imported_pool, indexer, n


# ═══════════════════════════════════════════════════════════════════
# 1. Import correctness
# ═══════════════════════════════════════════════════════════════════

class TestImport:
    """Verify that importing all subjects produces correct atoms."""

    def test_pool_has_two_datasets(self, imported_pool):
        """Pool should contain both kul_aad and dtu_aad datasets."""
        ds_dir = imported_pool.root / "datasets"
        datasets = [d.name for d in ds_dir.iterdir() if d.is_dir()]
        assert "kul_aad" in datasets, f"Missing kul_aad, found: {datasets}"
        assert "dtu_aad" in datasets, f"Missing dtu_aad, found: {datasets}"

    def test_kul_subject_count(self, imported_pool):
        """KUL should have 16 subjects."""
        sub_dir = imported_pool.root / "datasets" / "kul_aad" / "subjects"
        subjects = sorted(d.name for d in sub_dir.iterdir() if d.is_dir())
        assert len(subjects) == 16, f"Expected 16 KUL subjects, got {len(subjects)}: {subjects}"

    def test_dtu_subject_count(self, imported_pool):
        """DTU should have 18 subjects."""
        sub_dir = imported_pool.root / "datasets" / "dtu_aad" / "subjects"
        subjects = sorted(d.name for d in sub_dir.iterdir() if d.is_dir())
        assert len(subjects) == 18, f"Expected 18 DTU subjects, got {len(subjects)}: {subjects}"

    def test_kul_atom_structure(self, imported_pool):
        """Each KUL atom should have correct metadata."""
        from neuroatom.storage.metadata_store import AtomJSONLReader
        from neuroatom.storage import paths as P

        jsonl = P.atoms_jsonl_path(imported_pool.root, "kul_aad", "S1", "ses-01", "trial_001")
        atoms = list(AtomJSONLReader(jsonl).iter_atoms())
        assert len(atoms) == 1

        atom = atoms[0]
        assert atom.dataset_id == "kul_aad"
        assert atom.subject_id == "S1"
        assert atom.sampling_rate == 128.0
        assert atom.n_channels == 64

        ann_names = {a.name for a in atom.annotations}
        assert "attended_ear" in ann_names, f"Missing attended_ear, found: {ann_names}"

    def test_dtu_atom_structure(self, imported_pool):
        """Each DTU atom should have correct metadata."""
        from neuroatom.storage.metadata_store import AtomJSONLReader
        from neuroatom.storage import paths as P

        jsonl = P.atoms_jsonl_path(imported_pool.root, "dtu_aad", "S1", "ses-01", "trial_001")
        atoms = list(AtomJSONLReader(jsonl).iter_atoms())
        assert len(atoms) == 1

        atom = atoms[0]
        assert atom.dataset_id == "dtu_aad"
        assert atom.subject_id == "S1"
        assert atom.sampling_rate == 64.0
        # DTU has 66 channels (64 EEG + EXG1 + EXG2) or 64 depending on preprocessing
        assert atom.n_channels >= 64

    def test_signal_readback_kul(self, imported_pool):
        """KUL signal can be read back from HDF5 with correct shape and dtype."""
        from neuroatom.storage.metadata_store import AtomJSONLReader
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage import paths as P

        jsonl = P.atoms_jsonl_path(imported_pool.root, "kul_aad", "S1", "ses-01", "trial_001")
        atom = list(AtomJSONLReader(jsonl).iter_atoms())[0]

        signal = ShardManager.static_read(imported_pool.root, atom.signal_ref)
        assert signal.dtype == np.float32
        assert signal.shape[0] == 64  # 64 channels
        assert signal.shape[1] > 10000  # reasonable length
        assert np.isfinite(signal).all(), "Signal contains NaN/Inf"

    def test_signal_readback_dtu(self, imported_pool):
        """DTU signal can be read back from HDF5 with correct shape."""
        from neuroatom.storage.metadata_store import AtomJSONLReader
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage import paths as P

        jsonl = P.atoms_jsonl_path(imported_pool.root, "dtu_aad", "S1", "ses-01", "trial_001")
        atom = list(AtomJSONLReader(jsonl).iter_atoms())[0]

        signal = ShardManager.static_read(imported_pool.root, atom.signal_ref)
        assert signal.dtype == np.float32
        assert signal.shape[0] >= 64
        assert signal.shape[1] > 1000
        assert np.isfinite(signal).all(), "Signal contains NaN/Inf"


# ═══════════════════════════════════════════════════════════════════
# 2. Indexing
# ═══════════════════════════════════════════════════════════════════

class TestIndex:
    """Verify that all atoms are indexed correctly."""

    def test_total_indexed(self, indexed_pool):
        """Should have 34 atoms total (16 KUL + 18 DTU, 1 trial each)."""
        _, _, n_indexed = indexed_pool
        assert n_indexed == 34, f"Expected 34 indexed atoms, got {n_indexed}"

    def test_query_by_dataset(self, indexed_pool):
        """Can query atoms by dataset_id."""
        _, indexer, _ = indexed_pool
        from neuroatom.index.query import QueryBuilder

        qb = QueryBuilder(indexer.backend)

        kul_ids = qb.query_atom_ids({"dataset_id": "kul_aad"})
        assert len(kul_ids) == 16, f"Expected 16 KUL atoms, got {len(kul_ids)}"

        dtu_ids = qb.query_atom_ids({"dataset_id": "dtu_aad"})
        assert len(dtu_ids) == 18, f"Expected 18 DTU atoms, got {len(dtu_ids)}"

    def test_query_by_subject(self, indexed_pool):
        """Can query atoms by subject_id."""
        _, indexer, _ = indexed_pool
        from neuroatom.index.query import QueryBuilder

        qb = QueryBuilder(indexer.backend)
        s1_kul = qb.query_atom_ids({"dataset_id": "kul_aad", "subject_id": "S1"})
        assert len(s1_kul) == 1

    def test_query_cross_dataset(self, indexed_pool):
        """Can query across both datasets in one query."""
        _, indexer, _ = indexed_pool
        from neuroatom.index.query import QueryBuilder

        qb = QueryBuilder(indexer.backend)
        all_ids = qb.query_atom_ids({"dataset_id": ["kul_aad", "dtu_aad"]})
        assert len(all_ids) == 34

    def test_query_by_annotation(self, indexed_pool):
        """Can query by annotation name."""
        _, indexer, _ = indexed_pool
        from neuroatom.index.query import QueryBuilder

        qb = QueryBuilder(indexer.backend)
        # KUL atoms have 'attended_ear'
        ear_ids = qb.query_atom_ids({
            "dataset_id": "kul_aad",
            "annotation": "attended_ear",
        })
        assert len(ear_ids) == 16


# ═══════════════════════════════════════════════════════════════════
# 3. Assembly: cross-subject split, channel mapping, resampling
# ═══════════════════════════════════════════════════════════════════

class TestAssembly:
    """Test the full assembly pipeline with cross-subject splitting."""

    def _assemble(self, indexed_pool, **recipe_overrides):
        """Helper: assemble with defaults, overridable."""
        pool, indexer, _ = indexed_pool
        from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
        from neuroatom.core.enums import (
            NormalizationMethod, NormalizationScope, SplitStrategy,
        )
        from neuroatom.assembler.dataset_assembler import DatasetAssembler

        defaults = dict(
            recipe_id="aad_cross_subject_test",
            query={"dataset_id": ["kul_aad", "dtu_aad"]},
            target_sampling_rate=64.0,
            target_duration=10.0,
            target_unit="uV",
            label_fields=[
                LabelSpec(annotation_name="attended_ear", output_key="label"),
            ],
            split_strategy=SplitStrategy.SUBJECT,
            split_config={
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "seed": 42,
            },
            normalization_method=NormalizationMethod.ZSCORE,
            normalization_scope=NormalizationScope.PER_ATOM,
        )
        defaults.update(recipe_overrides)
        recipe = AssemblyRecipe(**defaults)

        assembler = DatasetAssembler(pool, indexer)
        return assembler.assemble(recipe)

    def test_assembly_produces_samples(self, indexed_pool):
        """Assembly should produce train/val/test samples."""
        result = self._assemble(indexed_pool)

        n_total = len(result.train_samples) + len(result.val_samples) + len(result.test_samples)
        # Some atoms may be skipped if they lack 'attended_ear' (DTU uses 'attended_speaker')
        # But KUL's 16 atoms all have it
        assert n_total >= 16, f"Expected ≥16 samples, got {n_total}"

        log = result.assembly_log
        print(f"\n  Assembly: {log['n_train']} train, {log['n_val']} val, "
              f"{log['n_test']} test ({log['elapsed_seconds']:.1f}s)")

    def test_signal_shape_uniform(self, indexed_pool):
        """All assembled signals should have uniform shape after resampling + pad/crop."""
        result = self._assemble(indexed_pool)

        all_samples = result.train_samples + result.val_samples + result.test_samples
        assert len(all_samples) > 0

        # target_duration=10.0, target_srate=64.0 → 640 samples
        expected_samples = int(10.0 * 64.0)

        for s in all_samples:
            sig = s["signal"]
            assert sig.shape[1] == expected_samples, (
                f"Atom {s['atom_id']}: expected {expected_samples} samples, "
                f"got {sig.shape[1]}"
            )
            assert np.isfinite(sig).all(), f"NaN/Inf in {s['atom_id']}"

    def test_subject_split_no_leakage(self, indexed_pool):
        """No subject should appear in both train and test."""
        result = self._assemble(indexed_pool)

        train_subs = {(s["dataset_id"], s["subject_id"]) for s in result.train_samples}
        val_subs = {(s["dataset_id"], s["subject_id"]) for s in result.val_samples}
        test_subs = {(s["dataset_id"], s["subject_id"]) for s in result.test_samples}

        assert train_subs.isdisjoint(test_subs), (
            f"Subject leakage! Train ∩ Test = {train_subs & test_subs}"
        )
        assert train_subs.isdisjoint(val_subs), (
            f"Subject leakage! Train ∩ Val = {train_subs & val_subs}"
        )
        assert val_subs.isdisjoint(test_subs), (
            f"Subject leakage! Val ∩ Test = {val_subs & test_subs}"
        )

        print(f"\n  No leakage: train={len(train_subs)} subs, "
              f"val={len(val_subs)} subs, test={len(test_subs)} subs")

    def test_labels_encoded(self, indexed_pool):
        """Labels should be encoded as integers."""
        result = self._assemble(indexed_pool)

        for s in result.train_samples:
            assert "label" in s["labels"], f"Missing 'label' key in {s['atom_id']}"
            val = s["labels"]["label"]
            assert isinstance(val, (int, np.integer)), (
                f"Label should be int, got {type(val)}: {val}"
            )

    def test_normalization_applied(self, indexed_pool):
        """Z-score normalized signals should have ~0 mean, ~1 std."""
        result = self._assemble(indexed_pool)

        for s in result.train_samples[:5]:
            sig = s["signal"]
            # Per-atom z-score: each atom's mean ≈ 0, std ≈ 1
            # (across all channels and time)
            mean = np.abs(sig.mean())
            std = sig.std()
            assert mean < 0.5, f"Mean {mean:.2f} too far from 0 after z-score"
            assert 0.1 < std < 5.0, f"Std {std:.2f} unexpected after z-score"


# ═══════════════════════════════════════════════════════════════════
# 4. Cross-dataset assembly (KUL + DTU unified label)
# ═══════════════════════════════════════════════════════════════════

class TestCrossDatasetAssembly:
    """Test assembling KUL and DTU together with a unified label scheme."""

    def test_kul_only_assembly(self, indexed_pool):
        """KUL-only assembly should work with attended_ear label."""
        pool, indexer, _ = indexed_pool
        from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
        from neuroatom.core.enums import SplitStrategy
        from neuroatom.assembler.dataset_assembler import DatasetAssembler

        recipe = AssemblyRecipe(
            recipe_id="kul_only",
            query={"dataset_id": "kul_aad"},
            target_sampling_rate=64.0,
            target_duration=10.0,
            label_fields=[
                LabelSpec(annotation_name="attended_ear", output_key="ear"),
            ],
            split_strategy=SplitStrategy.SUBJECT,
            split_config={"val_ratio": 0.15, "test_ratio": 0.15, "seed": 42},
        )

        result = DatasetAssembler(pool, indexer).assemble(recipe)
        n = len(result.train_samples) + len(result.val_samples) + len(result.test_samples)
        assert n == 16, f"Expected 16 KUL samples, got {n}"

    def test_dtu_only_assembly(self, indexed_pool):
        """DTU-only assembly should work with attended_speaker label."""
        pool, indexer, _ = indexed_pool
        from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
        from neuroatom.core.enums import SplitStrategy
        from neuroatom.assembler.dataset_assembler import DatasetAssembler

        recipe = AssemblyRecipe(
            recipe_id="dtu_only",
            query={"dataset_id": "dtu_aad"},
            target_sampling_rate=64.0,
            target_duration=10.0,
            label_fields=[
                LabelSpec(annotation_name="attended_speaker", output_key="speaker"),
            ],
            split_strategy=SplitStrategy.SUBJECT,
            split_config={"val_ratio": 0.15, "test_ratio": 0.15, "seed": 42},
        )

        result = DatasetAssembler(pool, indexer).assemble(recipe)
        n = len(result.train_samples) + len(result.val_samples) + len(result.test_samples)
        assert n == 18, f"Expected 18 DTU samples, got {n}"


# ═══════════════════════════════════════════════════════════════════
# 5. DataLoader (PyTorch integration)
# ═══════════════════════════════════════════════════════════════════

class TestDataLoader:
    """Test PyTorch DataLoader integration."""

    def _get_result(self, indexed_pool):
        pool, indexer, _ = indexed_pool
        from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
        from neuroatom.core.enums import (
            NormalizationMethod, NormalizationScope, SplitStrategy,
        )
        from neuroatom.assembler.dataset_assembler import DatasetAssembler

        recipe = AssemblyRecipe(
            recipe_id="aad_dataloader_test",
            query={"dataset_id": "kul_aad"},
            target_sampling_rate=64.0,
            target_duration=10.0,
            target_unit="uV",
            label_fields=[
                LabelSpec(annotation_name="attended_ear", output_key="ear"),
            ],
            split_strategy=SplitStrategy.SUBJECT,
            split_config={
                "test_subjects": ["kul_aad|S16"],
                "val_subjects": ["kul_aad|S15"],
            },
            normalization_method=NormalizationMethod.ZSCORE,
            normalization_scope=NormalizationScope.PER_ATOM,
        )

        return DatasetAssembler(pool, indexer).assemble(recipe)

    def test_torch_dataset(self, indexed_pool):
        """AtomDataset wraps assembled samples correctly."""
        pytest.importorskip("torch")
        from neuroatom.loader.torch_dataset import AtomDataset

        result = self._get_result(indexed_pool)
        ds = AtomDataset(result.train_samples)
        assert len(ds) == 14  # 16 - 1 test - 1 val

        sample = ds[0]
        assert "signal" in sample
        assert "labels" in sample
        assert sample["signal"].ndim == 2

    def test_dataloader_batching(self, indexed_pool):
        """DataLoader produces correct batch shapes."""
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader
        from neuroatom.loader.torch_dataset import AtomDataset

        result = self._get_result(indexed_pool)
        ds = AtomDataset(result.train_samples)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        sig = batch["signal"]

        # (batch, channels, time)
        assert sig.ndim == 3
        assert sig.shape[0] <= 4
        assert sig.shape[2] == int(10.0 * 64.0)  # 640 samples
        assert sig.dtype == torch.float32

        print(f"\n  Batch shape: {sig.shape} (B×C×T)")

    def test_dataloader_labels(self, indexed_pool):
        """Labels in batch should be integer tensors."""
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader
        from neuroatom.loader.torch_dataset import AtomDataset

        result = self._get_result(indexed_pool)
        ds = AtomDataset(result.train_samples)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        labels = batch["labels"]["ear"]
        assert labels.dtype == torch.int64
        assert labels.ndim == 1

    def test_dataloader_no_nan(self, indexed_pool):
        """Full iteration through DataLoader should produce no NaN/Inf."""
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader
        from neuroatom.loader.torch_dataset import AtomDataset

        result = self._get_result(indexed_pool)
        ds = AtomDataset(result.train_samples)
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        for batch in loader:
            sig = batch["signal"]
            assert torch.isfinite(sig).all(), "NaN/Inf found in training data"

        print(f"\n  ✓ {len(ds)} samples, all finite")


# ═══════════════════════════════════════════════════════════════════
# 6. multiload() convenience API
# ═══════════════════════════════════════════════════════════════════

class TestMultiloadAAD:
    """Test the high-level multiload() API with KUL data."""

    def test_multiload_kul_all_subjects(self):
        """multiload() imports all KUL subjects and produces DataLoaders."""
        pytest.importorskip("torch")
        from neuroatom.quick import multiload

        sources = []
        for mat in _kul_mats:
            sources.append({
                "dataset": "kul_aad",
                "path": str(mat),
                "subjects": [mat.stem],
                "import_kwargs": {"max_trials": 1},
            })

        train, val, test = multiload(
            sources=sources,
            target_srate=64.0,
            target_duration=10.0,
            label_field="attended_ear",
            split_strategy="subject",
            split_config={"val_ratio": 0.15, "test_ratio": 0.15, "seed": 42},
        )

        # At least train should exist
        assert train is not None, "No training data produced"
        batch = next(iter(train))
        sig = batch["signal"]
        assert sig.ndim == 3
        print(f"\n  ✓ multiload KUL: train batch shape = {sig.shape}")
