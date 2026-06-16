"""Tests for data catalog: models, local index, search, remote protocol."""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

from neuroatom.catalog.models import CatalogIndex, DatasetEntry
from neuroatom.catalog.local import (
    load_catalog, save_catalog, rebuild_catalog,
    update_catalog_entry, remove_catalog_entry, catalog_path,
)
from neuroatom.catalog.remote import fetch_remote_catalog, merge_remote
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import QualityTier
from neuroatom.importers.base import TaskConfig
from neuroatom.importers.generic import GenericImporter
from neuroatom.storage.pool import Pool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pool(tmp_path):
    return Pool.create(tmp_path / "pool")


def _make_entry(ds_id="ds1", name="Dataset One", task_types=None,
                n_subjects=10, quality_tier=None, tags=None):
    return DatasetEntry(
        dataset_id=ds_id,
        name=name,
        task_types=task_types or ["motor_imagery"],
        n_subjects=n_subjects,
        n_atoms=100,
        quality_tier=quality_tier,
        tags=tags or ["eeg", "bci"],
    )


def _import_dummy(pool, ds_id="test_ds"):
    """Import a tiny dataset for testing."""
    data_dir = pool.root.parent / "data" / "S01"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    np.save(data_dir / "signal.npy", rng.standard_normal((4, 128)).astype(np.float32))

    config = TaskConfig({
        "dataset_id": ds_id, "dataset_name": f"Test {ds_id}",
        "signal_unit": "uV", "custom": {"sampling_rate": 128.0},
    })
    importer = GenericImporter(pool=pool, task_config=config)
    importer.import_dataset(
        data_dir.parent, dataset_id=ds_id, sampling_rate=128.0,
    )


# ---------------------------------------------------------------------------
# CatalogIndex model tests
# ---------------------------------------------------------------------------

class TestCatalogModels:

    def test_entry_matches_query(self):
        e = _make_entry()
        assert e.matches(query="dataset")
        assert e.matches(query="motor")
        assert not e.matches(query="nonexistent")

    def test_entry_matches_task_type(self):
        e = _make_entry()
        assert e.matches(task_type="motor_imagery")
        assert not e.matches(task_type="erp")

    def test_entry_matches_tier(self):
        e = _make_entry(quality_tier=QualityTier.GOLD)
        assert e.matches(tier=QualityTier.GOLD)
        assert not e.matches(tier=QualityTier.PLATINUM)

    def test_entry_matches_tag(self):
        e = _make_entry(tags=["eeg", "bci"])
        assert e.matches(tag="bci")
        assert not e.matches(tag="clinical")

    def test_entry_matches_min_subjects(self):
        e = _make_entry(n_subjects=10)
        assert e.matches(min_subjects=5)
        assert not e.matches(min_subjects=20)

    def test_catalog_search(self):
        cat = CatalogIndex(datasets=[
            _make_entry("ds1", task_types=["motor_imagery"]),
            _make_entry("ds2", task_types=["erp"]),
            _make_entry("ds3", task_types=["motor_imagery", "erp"]),
        ])
        assert len(cat.search(task_type="motor_imagery")) == 2
        assert len(cat.search(task_type="erp")) == 2
        assert len(cat.search(query="ds1")) == 1

    def test_catalog_upsert(self):
        cat = CatalogIndex()
        e1 = _make_entry("ds1")
        cat.upsert(e1)
        assert len(cat.datasets) == 1
        # Update
        e2 = _make_entry("ds1", name="Updated")
        cat.upsert(e2)
        assert len(cat.datasets) == 1
        assert cat.get("ds1").name == "Updated"

    def test_catalog_remove(self):
        cat = CatalogIndex(datasets=[_make_entry("ds1"), _make_entry("ds2")])
        assert cat.remove("ds1")
        assert len(cat.datasets) == 1
        assert not cat.remove("nonexistent")

    def test_catalog_merge(self):
        local = CatalogIndex(datasets=[_make_entry("ds1")])
        remote = CatalogIndex(datasets=[
            _make_entry("ds1", name="Updated"),
            _make_entry("ds2"),
        ])
        count = local.merge(remote)
        assert count == 2  # ds1 updated + ds2 new
        assert len(local.datasets) == 2

    def test_catalog_serialization(self):
        cat = CatalogIndex(datasets=[_make_entry("ds1")])
        json_str = cat.model_dump_json()
        parsed = CatalogIndex.model_validate_json(json_str)
        assert len(parsed.datasets) == 1
        assert parsed.datasets[0].dataset_id == "ds1"


# ---------------------------------------------------------------------------
# Local catalog tests
# ---------------------------------------------------------------------------

class TestLocalCatalog:

    def test_load_empty(self, pool):
        cat = load_catalog(pool)
        assert len(cat.datasets) == 0

    def test_save_and_load(self, pool):
        cat = CatalogIndex(datasets=[_make_entry("ds1")])
        save_catalog(pool, cat)
        assert catalog_path(pool).exists()

        loaded = load_catalog(pool)
        assert len(loaded.datasets) == 1
        assert loaded.datasets[0].dataset_id == "ds1"
        assert loaded.updated_at is not None

    def test_rebuild_from_pool(self, pool):
        _import_dummy(pool, "test_ds")
        catalog = rebuild_catalog(pool)
        assert len(catalog.datasets) == 1
        assert catalog.datasets[0].dataset_id == "test_ds"
        assert catalog.datasets[0].n_atoms > 0

    def test_update_single_entry(self, pool):
        _import_dummy(pool, "test_ds")
        entry = update_catalog_entry(pool, "test_ds")
        assert entry.dataset_id == "test_ds"

        cat = load_catalog(pool)
        assert cat.get("test_ds") is not None

    def test_remove_entry(self, pool):
        _import_dummy(pool, "test_ds")
        rebuild_catalog(pool)
        assert remove_catalog_entry(pool, "test_ds")
        cat = load_catalog(pool)
        assert cat.get("test_ds") is None


# ---------------------------------------------------------------------------
# Remote catalog tests
# ---------------------------------------------------------------------------

class _CatalogHandler(SimpleHTTPRequestHandler):
    """Serves a catalog.json from a temp directory."""
    def log_message(self, *args):
        pass  # suppress log output


class TestRemoteCatalog:

    @pytest.fixture
    def remote_server(self, tmp_path):
        """Start an HTTP server serving a catalog."""
        catalog = CatalogIndex(
            name="test-remote",
            datasets=[
                _make_entry("remote_ds1", name="Remote Dataset 1"),
                _make_entry("remote_ds2", name="Remote Dataset 2",
                            task_types=["erp"]),
            ],
        )
        catalog_file = tmp_path / "catalog.json"
        catalog_file.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")

        import functools
        handler = functools.partial(_CatalogHandler, directory=str(tmp_path))
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{port}"
        server.shutdown()

    def test_fetch_remote(self, remote_server):
        url = f"{remote_server}/catalog.json"
        cat = fetch_remote_catalog(url)
        assert len(cat.datasets) == 2
        assert cat.datasets[0].dataset_id == "remote_ds1"

    def test_merge_remote_into_local(self, pool, remote_server):
        url = f"{remote_server}/catalog.json"
        count = merge_remote(pool, url)
        assert count == 2

        cat = load_catalog(pool)
        assert len(cat.datasets) == 2

    def test_merge_is_idempotent(self, pool, remote_server):
        url = f"{remote_server}/catalog.json"
        merge_remote(pool, url)
        count2 = merge_remote(pool, url)
        assert count2 == 0  # no new entries

    def test_fetch_bad_url(self):
        with pytest.raises(ConnectionError):
            fetch_remote_catalog("http://127.0.0.1:1/nonexistent.json", timeout=1)
