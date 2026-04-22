"""End-to-end tests for Zuco 2.0 Task-Specific Reading Dataset.

Tests cover:
    1. Import → index → query → signal verification
    2. Sentence-level epoch extraction
    3. Channel info with 3D coordinates
    4. Variable-length reading epochs
    5. Dataset auto-detection
"""

import os
from pathlib import Path

import numpy as np
import pytest

# Zuco 2.0 — skip if unavailable
_zuco_env = os.environ.get("NEUROATOM_ZUCO2_DIR", "")
ZUCO_DIR = Path(_zuco_env) if _zuco_env else Path("__nonexistent__")
SKIP = not ZUCO_DIR.exists()
pytestmark = pytest.mark.skipif(SKIP, reason="NEUROATOM_ZUCO2_DIR not set or path not found")


@pytest.fixture
def pool_dir(tmp_path):
    return tmp_path / "pool_zuco2"


# ------------------------------------------------------------------
# Test 1: Core import + index + query + signal verification
# ------------------------------------------------------------------

def test_zuco2_import_and_index(pool_dir):
    """Import 3 sentences from 1 text → index → query → read signal."""
    from neuroatom.importers.zuco2 import Zuco2Importer
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.signal_store import ShardManager

    pool = Pool.create(pool_dir)
    tc_path = (
        Path(__file__).parent.parent
        / "neuroatom" / "importers" / "task_configs" / "zuco2_tsr.yaml"
    )
    task_config = TaskConfig.from_yaml(tc_path)

    importer = Zuco2Importer(pool, task_config)

    # Import 3 sentences from YAC, TSR1
    results = importer.import_subject(
        dataset_dir=ZUCO_DIR,
        subject_id="YAC",
        texts=["TSR1"],
        max_sentences=3,
    )

    assert len(results) == 1
    result = results[0]
    assert len(result.atoms) == 3

    # Check channel count
    assert len(result.channel_infos) == 105

    # Check channel names (EGI HydroCel)
    ch_names = [ch.name for ch in result.channel_infos]
    assert "E2" in ch_names
    assert "Cz" in ch_names

    # Check sampling rate and units
    assert result.channel_infos[0].sampling_rate == 500.0
    assert result.channel_infos[0].unit == "uV"

    # Check 3D electrode coordinates
    n_with_coords = sum(1 for ch in result.channel_infos if ch.location is not None)
    assert n_with_coords > 90, f"Expected >90 channels with coords, got {n_with_coords}"

    # Check annotations
    atom0 = result.atoms[0]
    text_ann = [a for a in atom0.annotations if a.name == "text_id"][0]
    assert text_ann.value == "TSR1"

    sent_ann = [a for a in atom0.annotations if a.name == "sentence_index"][0]
    assert sent_ann.numeric_value == 0.0

    task_ann = [a for a in atom0.annotations if a.name == "task"][0]
    assert task_ann.value == "reading"

    # Check variable-length epochs
    durations = [a.temporal.duration_seconds for a in result.atoms]
    assert all(d > 1.0 for d in durations), f"Durations too short: {durations}"
    assert all(d < 60.0 for d in durations), f"Durations too long: {durations}"
    # Durations should differ (variable reading times)
    assert len(set(durations)) > 1, "All durations identical — suspicious"

    # Index and query
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    assert n_indexed == 3

    qb = QueryBuilder(indexer.backend)
    all_ids = qb.query_atom_ids({"dataset_id": "zuco2_tsr"})
    assert len(all_ids) == 3

    # Query by text_id annotation
    tsr1_ids = qb.query_atom_ids({
        "annotations": [{"name": "text_id", "value_in": ["TSR1"]}],
    })
    assert len(tsr1_ids) == 3

    # Read back signal and verify
    signal = ShardManager.static_read(pool.root, atom0.signal_ref)
    assert signal.shape[0] == 105  # channels
    assert signal.shape[1] > 100   # samples (variable)
    assert signal.dtype == np.float32

    # Signal in µV — preprocessed EEG
    max_abs = np.abs(signal).max()
    assert max_abs > 0.01, f"Signal too small for µV: {max_abs}"
    assert max_abs < 1000, f"Signal too large for µV: {max_abs}"

    indexer.close()
    print(f"\n✓ Zuco 2.0: 3 sentences imported, 105 ch @ 500 Hz, "
          f"durations={[f'{d:.1f}s' for d in durations]}")


# ------------------------------------------------------------------
# Test 2: Dataset detection
# ------------------------------------------------------------------

def test_zuco2_detect():
    """Verify detect() identifies the dataset."""
    from neuroatom.importers.zuco2 import Zuco2Importer

    assert Zuco2Importer.detect(ZUCO_DIR) is True
    assert Zuco2Importer.detect(Path("C:\\nonexistent")) is False

    print("\n✓ Zuco 2.0 detection verified.")


# ------------------------------------------------------------------
# Test 3: Electrode coordinates
# ------------------------------------------------------------------

def test_zuco2_electrode_coordinates(pool_dir):
    """Verify 3D electrode coordinates are correctly extracted."""
    from neuroatom.importers.zuco2 import Zuco2Importer
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)
    tc_path = (
        Path(__file__).parent.parent
        / "neuroatom" / "importers" / "task_configs" / "zuco2_tsr.yaml"
    )
    task_config = TaskConfig.from_yaml(tc_path)

    importer = Zuco2Importer(pool, task_config)
    results = importer.import_subject(
        dataset_dir=ZUCO_DIR,
        subject_id="YAC",
        texts=["TSR1"],
        max_sentences=1,
    )

    ch_infos = results[0].channel_infos

    # Check that most channels have 3D coordinates
    with_coords = [ch for ch in ch_infos if ch.location is not None]
    assert len(with_coords) >= 100, f"Expected ≥100 channels with coords, got {len(with_coords)}"

    # Verify coordinate values are reasonable
    for ch in with_coords[:5]:
        loc = ch.location
        assert loc.coordinate_system == "EGI_cart"
        assert abs(loc.x) < 20  # cm
        assert abs(loc.y) < 20
        assert abs(loc.z) < 20

    print(f"\n✓ Electrode coordinates: {len(with_coords)}/{len(ch_infos)} channels with 3D positions")
