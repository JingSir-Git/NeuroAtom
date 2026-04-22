"""End-to-end tests for SEED-V Emotion Recognition Dataset.

Tests cover:
    1. Import → index → query → signal verification
    2. Emotion label mapping per session
    3. Channel selection (64 channels: 62 EEG + 2 EOG from 66 total)
    4. Variable-length trial handling
    5. Dataset auto-detection
"""

import os
from pathlib import Path

import numpy as np
import pytest

# SEED-V — skip if unavailable
_seedv_env = os.environ.get("NEUROATOM_SEEDV_DIR", "")
SEEDV_DIR = Path(_seedv_env) if _seedv_env else Path("__nonexistent__")
SKIP = not SEEDV_DIR.exists()
pytestmark = pytest.mark.skipif(SKIP, reason="NEUROATOM_SEEDV_DIR not set or path not found")


@pytest.fixture
def pool_dir(tmp_path):
    return tmp_path / "pool_seedv"


# ------------------------------------------------------------------
# Test 1: Core import + index + query + signal verification
# ------------------------------------------------------------------

def test_seedv_import_and_index(pool_dir):
    """Import 2 trials from 1 session → index → query by emotion → read signal."""
    from neuroatom.importers.seed_v import SEEDVImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.signal_store import ShardManager

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "seed_v.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = SEEDVImporter(pool, task_config)

    # Import 2 trials from subject 1, session 1
    results = importer.import_subject(
        dataset_dir=SEEDV_DIR,
        subject_num=1,
        sessions=[1],
        max_trials=2,
    )

    assert len(results) == 1
    result = results[0]

    # Should have 2 trials
    assert len(result.atoms) == 2

    # Check channel count — 64 channels: 62 EEG + 2 EOG (VEO, HEO), excluding M1/M2
    assert len(result.channel_infos) == 64
    ch_names = [ch.name for ch in result.channel_infos]
    assert "FP1" in ch_names
    assert "OZ" in ch_names
    assert "VEO" in ch_names  # Now included as EOG
    assert "HEO" in ch_names  # Now included as EOG
    assert "M1" not in ch_names  # Mastoid refs excluded
    assert "M2" not in ch_names

    # Verify EOG channel types
    eog_chs = [ch for ch in result.channel_infos if ch.type.value == "eog"]
    assert len(eog_chs) == 2
    eeg_chs = [ch for ch in result.channel_infos if ch.type.value == "eeg"]
    assert len(eeg_chs) == 62

    # Verify electrode coordinates on EEG channels
    eeg_with_coords = [ch for ch in eeg_chs if ch.location is not None]
    assert len(eeg_with_coords) >= 40, f"Expected ≥40 EEG with coords, got {len(eeg_with_coords)}"

    # Check sampling rate and units
    assert result.channel_infos[0].sampling_rate == 1000.0
    assert result.channel_infos[0].unit == "V"

    # Check emotion annotations
    # Session 1, trials 1-2: Happy(4), Fear(1)
    atom0 = result.atoms[0]
    emo0 = [a for a in atom0.annotations if a.name == "emotion"][0]
    assert emo0.value == "happy"

    atom1 = result.atoms[1]
    emo1 = [a for a in atom1.annotations if a.name == "emotion"][0]
    assert emo1.value == "fear"

    # Check variable-length trials
    # Trial 1: 30-102s = 72s = 72000 samples
    assert atom0.temporal.duration_seconds == 72.0
    assert atom0.signal_ref.shape == (64, 72000)

    # Trial 2: 132-228s = 96s = 96000 samples
    assert atom1.temporal.duration_seconds == 96.0
    assert atom1.signal_ref.shape == (64, 96000)

    # Index and query
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    assert n_indexed == 2

    qb = QueryBuilder(indexer.backend)
    happy_ids = qb.query_atom_ids({
        "annotations": [{"name": "emotion", "value_in": ["happy"]}],
    })
    assert len(happy_ids) == 1

    fear_ids = qb.query_atom_ids({
        "annotations": [{"name": "emotion", "value_in": ["fear"]}],
    })
    assert len(fear_ids) == 1

    # Read back signal and verify
    signal = ShardManager.static_read(pool.root, atom0.signal_ref)
    assert signal.shape == (64, 72000)
    assert signal.dtype == np.float32

    # Signal in V — typical EEG range
    max_abs = np.abs(signal).max()
    assert max_abs > 1e-6, f"Signal too small: {max_abs}"
    assert max_abs < 0.1, f"Signal too large for V: {max_abs}"

    indexer.close()
    print(f"\n✓ SEED-V: 2 trials imported (happy 72s, fear 96s), "
          f"64 ch (62 EEG + 2 EOG) @ 1000 Hz, indexed + queried.")


# ------------------------------------------------------------------
# Test 2: Dataset auto-detection
# ------------------------------------------------------------------

def test_seedv_detect():
    """Verify detect() correctly identifies the dataset."""
    from neuroatom.importers.seed_v import SEEDVImporter

    assert SEEDVImporter.detect(SEEDV_DIR) is True
    assert SEEDVImporter.detect(Path("C:\\nonexistent")) is False
    assert SEEDVImporter.detect(Path(__file__).parent) is False

    print("\n✓ SEED-V detection verified.")


# ------------------------------------------------------------------
# Test 3: Emotion label consistency across sessions
# ------------------------------------------------------------------

def test_seedv_emotion_labels(pool_dir):
    """Verify emotion labels differ between sessions (different order)."""
    from neuroatom.importers.seed_v import SEEDVImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "seed_v.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = SEEDVImporter(pool, task_config)

    # Import first trial from session 1 and session 2
    results_s1 = importer.import_subject(
        dataset_dir=SEEDV_DIR, subject_num=1, sessions=[1], max_trials=1,
    )
    results_s2 = importer.import_subject(
        dataset_dir=SEEDV_DIR, subject_num=1, sessions=[2], max_trials=1,
    )

    # Session 1, trial 1: Happy(4)
    emo_s1 = [a for a in results_s1[0].atoms[0].annotations if a.name == "emotion"][0]
    assert emo_s1.value == "happy"

    # Session 2, trial 1: Sad(2)
    emo_s2 = [a for a in results_s2[0].atoms[0].annotations if a.name == "emotion"][0]
    assert emo_s2.value == "sad"

    print(f"\n✓ Session-specific emotion order: S1T1={emo_s1.value}, S2T1={emo_s2.value}")
