"""Regression tests for v0.2 bug fixes and API improvements.

Covers:
- Pool.open / Pool.exists / Pool.open_or_create (Bug 2 from review)
- quickload pool re-open path (Bug 1: pool.json vs pool.yaml mismatch)
- HDF5 read handle cache safety after thread/worker reset
- HDF5AtomDataset.safe_collate_fn auto-selection (skip mode footgun)
- AtomDataset large-dataset memory warning
- TaskConfig quickload metadata (label_field, aliases, subject_pattern)
- quickload subject inference using YAML-declared regex
"""

import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────
# Pool.open / open_or_create / exists
# ──────────────────────────────────────────────────────────────────────────

class TestPoolOpenAPI:
    """Verify the symmetric create/open/exists/open_or_create API."""

    def test_open_existing_pool(self, tmp_path):
        from neuroatom.storage.pool import Pool

        created = Pool.create(tmp_path / "pool1")
        assert created.root.exists()

        # The bug we fixed: Pool.open existed in callers but not on the class.
        reopened = Pool.open(tmp_path / "pool1")
        assert reopened.root == created.root

    def test_open_missing_pool_raises(self, tmp_path):
        from neuroatom.storage.pool import Pool

        with pytest.raises(FileNotFoundError):
            Pool.open(tmp_path / "nope")

    def test_exists_returns_true_after_create(self, tmp_path):
        from neuroatom.storage.pool import Pool

        assert Pool.exists(tmp_path / "p") is False
        Pool.create(tmp_path / "p")
        assert Pool.exists(tmp_path / "p") is True

    def test_open_or_create_idempotent(self, tmp_path):
        """open_or_create must not warn-overwrite an existing pool."""
        from neuroatom.storage.pool import Pool

        p1 = Pool.open_or_create(tmp_path / "p")
        # Mutate config so we can tell whether it was overwritten
        config_path = p1.root / "pool.yaml"
        original = config_path.read_text(encoding="utf-8")
        p2 = Pool.open_or_create(tmp_path / "p")
        assert config_path.read_text(encoding="utf-8") == original
        assert p2.root == p1.root


# ──────────────────────────────────────────────────────────────────────────
# quickload pool persistence regression
# ──────────────────────────────────────────────────────────────────────────

class TestQuickloadPoolPersistence:
    """Bug 1: quickload's `pool.json` check never matched (config is pool.yaml).

    Verify quickload now opens the existing pool instead of recreating.
    """

    def test_quickload_pool_check_uses_yaml(self, tmp_path):
        """The fixed quickload must use Pool.exists (which checks pool.yaml)."""
        from neuroatom.storage.pool import Pool

        # Create a pool first
        Pool.create(tmp_path / "pool_a")
        # The check inside quickload is now Pool.exists / open_or_create
        # — Pool.exists must return True at this point.
        assert Pool.exists(tmp_path / "pool_a") is True
        # The previous buggy check `(pool_dir / "pool.json").exists()` would
        # have returned False even after Pool.create succeeded:
        assert not (tmp_path / "pool_a" / "pool.json").exists()


# ──────────────────────────────────────────────────────────────────────────
# HDF5 read handle cache
# ──────────────────────────────────────────────────────────────────────────

class TestStaticReadHandleCache:
    """Verify the thread-local LRU read handle cache works and is resettable."""

    def _make_shard_with_signal(self, tmp_path):
        """Create a minimal HDF5 shard with one atom signal, return (pool_root, signal_ref)."""
        from neuroatom.core.signal_ref import SignalRef
        from neuroatom.storage.signal_store import ShardManager

        mgr = ShardManager(
            pool_root=tmp_path,
            dataset_id="ds",
            subject_id="s01",
            session_id="ses1",
            run_id="r1",
        )
        signal = np.random.randn(4, 100).astype(np.float32)
        ref = mgr.write_atom_signal("a1", signal)
        mgr.close()
        return tmp_path, ref, signal

    def test_repeated_static_read_returns_same_data(self, tmp_path):
        from neuroatom.storage.signal_store import ShardManager

        pool_root, ref, expected = self._make_shard_with_signal(tmp_path)
        # Read many times — exercises the cache hit path.
        for _ in range(5):
            data = ShardManager.static_read(pool_root, ref)
            np.testing.assert_array_equal(data, expected)

    def test_close_read_handles_releases_cache(self, tmp_path):
        from neuroatom.storage.signal_store import (
            ShardManager,
            _get_read_handle_cache,
            close_read_handles,
        )

        pool_root, ref, _ = self._make_shard_with_signal(tmp_path)
        ShardManager.static_read(pool_root, ref)
        cache = _get_read_handle_cache()
        assert len(cache) >= 1

        close_read_handles()
        cache_after = _get_read_handle_cache()
        assert len(cache_after) == 0


# ──────────────────────────────────────────────────────────────────────────
# HDF5AtomDataset auto-collate
# ──────────────────────────────────────────────────────────────────────────

class TestHDF5DatasetSafeCollate:
    """`error_handling='skip'` returning None used to crash default_collate."""

    def test_safe_collate_fn_picks_skip_none_for_skip_mode(self, tmp_path):
        torch = pytest.importorskip("torch")
        from neuroatom.loader.torch_dataset import (
            HDF5AtomDataset,
            skip_none_collate,
        )

        ds = HDF5AtomDataset(atoms=[], pool_root=tmp_path, error_handling="skip")
        assert ds.safe_collate_fn() is skip_none_collate
        assert ds.error_handling == "skip"

    def test_safe_collate_fn_picks_default_for_raise_mode(self, tmp_path):
        torch = pytest.importorskip("torch")
        from neuroatom.loader.torch_dataset import HDF5AtomDataset

        ds = HDF5AtomDataset(atoms=[], pool_root=tmp_path, error_handling="raise")
        fn = ds.safe_collate_fn()
        # default_collate is the standard torch one
        assert fn is torch.utils.data.dataloader.default_collate

    def test_skip_none_collate_filters_none(self):
        torch = pytest.importorskip("torch")
        from neuroatom.loader.torch_dataset import skip_none_collate

        batch = [None, {"x": torch.tensor([1.0])}, None, {"x": torch.tensor([2.0])}]
        out = skip_none_collate(batch)
        assert out is not None
        assert out["x"].shape == (2, 1)

    def test_skip_none_collate_returns_none_when_all_none(self):
        pytest.importorskip("torch")
        from neuroatom.loader.torch_dataset import skip_none_collate

        assert skip_none_collate([None, None]) is None

    def test_invalid_error_handling_rejected(self, tmp_path):
        pytest.importorskip("torch")
        from neuroatom.loader.torch_dataset import HDF5AtomDataset

        with pytest.raises(ValueError, match="error_handling"):
            HDF5AtomDataset(atoms=[], pool_root=tmp_path, error_handling="bogus")


# ──────────────────────────────────────────────────────────────────────────
# AtomDataset memory warning
# ──────────────────────────────────────────────────────────────────────────

class TestAtomDatasetMemoryWarning:
    def test_warn_on_many_samples(self, caplog):
        pytest.importorskip("torch")
        from neuroatom.loader.torch_dataset import AtomDataset

        # 11 samples with low warn_threshold → should warn
        samples = [
            {
                "signal": np.zeros((1, 1), dtype=np.float32),
                "labels": {},
                "atom_id": str(i),
                "subject_id": "s",
                "dataset_id": "d",
            }
            for i in range(11)
        ]
        with caplog.at_level(logging.WARNING, logger="neuroatom.loader.torch_dataset"):
            AtomDataset(samples, warn_threshold=10)
        assert any("in memory" in r.message for r in caplog.records)

    def test_no_warn_on_small_dataset(self, caplog):
        pytest.importorskip("torch")
        from neuroatom.loader.torch_dataset import AtomDataset

        samples = [
            {
                "signal": np.zeros((1, 1), dtype=np.float32),
                "labels": {},
                "atom_id": str(i),
                "subject_id": "s",
                "dataset_id": "d",
            }
            for i in range(3)
        ]
        with caplog.at_level(logging.WARNING, logger="neuroatom.loader.torch_dataset"):
            AtomDataset(samples)
        assert not any("memory" in r.message for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────
# TaskConfig quickload metadata
# ──────────────────────────────────────────────────────────────────────────

class TestTaskConfigQuickloadMeta:
    def test_quickload_meta_fields_round_trip(self, tmp_path):
        """A YAML with a quickload section must surface through TaskConfig."""
        import yaml

        from neuroatom.importers.base import TaskConfig

        cfg_path = tmp_path / "x.yaml"
        cfg_path.write_text(
            yaml.safe_dump({
                "dataset_id": "x",
                "quickload": {
                    "label_field": "my_label",
                    "aliases": ["x_alias", "x2"],
                    "data_path_kwarg": "mat_path",
                    "entry_method": "import_subject",
                    "subject_pattern": r"^(A\d+)",
                },
            }),
            encoding="utf-8",
        )
        cfg = TaskConfig.from_yaml(cfg_path)
        assert cfg.label_field == "my_label"
        assert cfg.quickload_aliases == ["x_alias", "x2"]
        assert cfg.quickload_data_path_kwarg == "mat_path"
        assert cfg.quickload_entry_method == "import_subject"
        assert cfg.quickload_subject_pattern == r"^(A\d+)"

    def test_quickload_meta_absent_returns_empty(self, tmp_path):
        import yaml

        from neuroatom.importers.base import TaskConfig

        cfg_path = tmp_path / "x.yaml"
        cfg_path.write_text(yaml.safe_dump({"dataset_id": "x"}), encoding="utf-8")
        cfg = TaskConfig.from_yaml(cfg_path)
        assert cfg.label_field is None
        assert cfg.quickload_aliases == []
        assert cfg.quickload_data_path_kwarg is None
        # Sensible default for entry_method, never None
        assert cfg.quickload_entry_method == "import_subject"

    def test_builtin_bci_has_quickload_meta(self):
        """The two YAMLs we populated should expose their quickload section."""
        from neuroatom.importers.base import TaskConfig

        cfg = TaskConfig.builtin("bci_comp_iv_2a")
        assert cfg.label_field == "mi_class"
        assert cfg.quickload_data_path_kwarg == "mat_path"
        assert cfg.quickload_subject_pattern is not None

        cfg2 = TaskConfig.builtin("physionet_mi")
        assert cfg2.label_field == "mi_class"
        assert cfg2.quickload_data_path_kwarg == "subject_dir"


class TestQuickloadSubjectInference:
    """quickload._infer_subject should honor YAML-declared regex first."""

    def test_yaml_regex_wins_over_heuristic(self, tmp_path):
        from neuroatom.importers.base import TaskConfig
        from neuroatom.quick import _infer_subject
        import yaml

        cfg_path = tmp_path / "z.yaml"
        cfg_path.write_text(
            yaml.safe_dump({
                "dataset_id": "z",
                "quickload": {"subject_pattern": r"sub-(\w+)"},
            }),
            encoding="utf-8",
        )
        cfg = TaskConfig.from_yaml(cfg_path)
        result = _infer_subject(Path("/data/sub-AB12.mat"), "z", config=cfg)
        assert result == "AB12"

    def test_fallback_when_no_regex(self):
        from neuroatom.quick import _infer_subject

        # BCI heuristic: first 3 chars of stem
        result = _infer_subject(Path("/x/A07T.mat"), "bci_comp_iv_2a", config=None)
        assert result == "A07"
