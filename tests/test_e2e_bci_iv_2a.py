"""End-to-end real data test: BCI Competition IV Dataset 2a (4-class MI).

Imports 1 labelled run from A01T, indexes, queries by MI class,
verifies signal shape/range, and validates class distribution.

Requires: NEUROATOM_BCI_IV_2A_DIR env var pointing to the data directory.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

_bci_dir = os.environ.get("NEUROATOM_BCI_IV_2A_DIR", "")
BCI_A01T = Path(_bci_dir) / "A01T.mat" if _bci_dir else Path(r"__nonexistent__")
pytestmark = pytest.mark.skipif(
    not BCI_A01T.exists(),
    reason="NEUROATOM_BCI_IV_2A_DIR not set or A01T.mat not found",
)


@pytest.fixture
def pool_dir():
    d = tempfile.mkdtemp(prefix="neuroatom_bci_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_bci_iv_2a_import_and_index(pool_dir):
    """Import 1 MI run, index, query by class label."""
    from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.pool import Pool
    from neuroatom.storage.signal_store import ShardManager

    logging.basicConfig(level=logging.INFO)

    pool = Pool.create(pool_dir)
    tc_path = (
        Path(__file__).parent.parent
        / "neuroatom" / "importers" / "task_configs" / "bci_comp_iv_2a.yaml"
    )
    task_config = TaskConfig.from_yaml(tc_path)

    importer = BCICompIV2aImporter(pool, task_config)
    results = importer.import_subject(
        mat_path=BCI_A01T,
        subject_id="A01",
        session_id="ses-T",
        max_runs=1,  # Only first labelled run (run-03)
    )

    # Should have 1 result for 1 labelled run
    assert len(results) == 1
    result = results[0]

    # 48 trials per run
    assert len(result.atoms) == 48

    # Verify atom properties
    atom = result.atoms[0]
    assert atom.dataset_id == "bci_comp_iv_2a"
    assert atom.subject_id == "A01"
    assert atom.atom_type.value == "event_epoch"
    assert atom.sampling_rate == 250.0
    assert atom.n_channels == 25  # 22 EEG + 3 EOG

    # Verify channel names from official layout
    ch_infos = result.channel_infos
    ch_names = [ch.name for ch in ch_infos]
    assert "Fz" in ch_names
    assert "C3" in ch_names
    assert "Cz" in ch_names
    assert "C4" in ch_names
    assert "EOG-left" in ch_names
    assert "EOG-central" in ch_names

    # Verify standard names
    eeg_chs = [ch for ch in ch_infos if ch.type == ChannelType.EEG]
    eog_chs = [ch for ch in ch_infos if ch.type == ChannelType.EOG]
    assert len(eeg_chs) == 22
    assert len(eog_chs) == 3
    # Standard name for C3 should be "C3"
    c3 = [ch for ch in ch_infos if ch.name == "C3"][0]
    assert c3.standard_name == "C3"

    # Verify annotations: mi_class label
    ann_names = {a.name for a in atom.annotations}
    assert "mi_class" in ann_names
    assert "mi_label" in ann_names

    mi_class_ann = [a for a in atom.annotations if a.name == "mi_class"][0]
    assert mi_class_ann.value in ("left_hand", "right_hand", "both_feet", "tongue")

    # Verify class distribution across 48 trials
    class_counts = {}
    artifact_count = 0
    for a in result.atoms:
        for ann in a.annotations:
            if ann.name == "mi_class":
                class_counts[ann.value] = class_counts.get(ann.value, 0) + 1
            if ann.name == "artifact":
                artifact_count += 1

    # 48 trials, roughly 12 per class
    assert sum(class_counts.values()) == 48
    assert len(class_counts) == 4  # All 4 classes present
    for cls, count in class_counts.items():
        assert count == 12, f"Expected 12 trials for {cls}, got {count}"
    print(f"  Class distribution: {class_counts}")
    print(f"  Artifacts: {artifact_count}")

    # Verify artifact marking
    rejected = [a for a in result.atoms
                if a.quality and a.quality.overall_status.value == "rejected"]
    good = [a for a in result.atoms
            if a.quality and a.quality.overall_status.value == "good"]
    assert len(rejected) + len(good) == 48
    assert len(rejected) == artifact_count

    # Index
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    assert n_indexed == 48

    # Query by annotation (MI class)
    qb = QueryBuilder(indexer.backend)

    left_ids = qb.query_atom_ids({
        "annotations": [{"name": "mi_class", "value_in": ["left_hand"]}],
    })
    assert len(left_ids) == 12

    right_ids = qb.query_atom_ids({
        "annotations": [{"name": "mi_class", "value_in": ["right_hand"]}],
    })
    assert len(right_ids) == 12

    # Query all
    all_ids = qb.query_atom_ids({"dataset_id": "bci_comp_iv_2a"})
    assert len(all_ids) == 48

    # Read back signal and verify shape/range
    signal = ShardManager.static_read(pool.root, atom.signal_ref)
    assert signal.shape == (25, 1500)  # 25 ch × 6s at 250 Hz
    assert signal.dtype == np.float32

    # Data is in µV — typical range ~[-400, 700] for raw, most within [-100, 100]
    assert np.abs(signal[:22]).max() < 500  # EEG channels
    assert not np.all(signal == 0)
    assert not np.any(np.isnan(signal))

    print(f"\n✓ BCI IV 2a E2E: 1 run, 48 trials, 4×12 balanced, "
          f"indexed, queried by MI class.")


def test_bci_iv_2a_signal_characteristics(pool_dir):
    """Verify EEG signal properties match expected BCI IV 2a characteristics."""
    from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool
    from neuroatom.storage.signal_store import ShardManager

    pool = Pool.create(pool_dir)
    tc_path = (
        Path(__file__).parent.parent
        / "neuroatom" / "importers" / "task_configs" / "bci_comp_iv_2a.yaml"
    )
    task_config = TaskConfig.from_yaml(tc_path)

    importer = BCICompIV2aImporter(pool, task_config)
    results = importer.import_subject(
        mat_path=BCI_A01T,
        subject_id="A01",
        session_id="ses-T",
        max_runs=1,
        max_trials=5,
    )

    # Read first clean trial
    result = results[0]
    clean_atoms = [a for a in result.atoms
                   if a.quality and a.quality.overall_status.value == "good"]
    assert len(clean_atoms) > 0

    atom = clean_atoms[0]
    signal = ShardManager.static_read(pool.root, atom.signal_ref)

    # Shape: 25 channels × 1500 samples (6s at 250 Hz)
    assert signal.shape == (25, 1500)

    # EEG channels (0-21): should have typical MI characteristics
    eeg = signal[:22, :]
    eog = signal[22:, :]

    # EEG should be ~zero-mean (bandpass filtered)
    assert abs(eeg.mean()) < 5.0  # µV, close to zero

    # EEG std should be in ~5-30 µV range (typical for filtered MI-EEG)
    eeg_std = eeg.std()
    assert 1.0 < eeg_std < 100.0, f"EEG std={eeg_std:.2f} µV outside expected range"

    # EOG channels can have larger amplitude
    # No strict assertion — just verify non-zero
    assert eog.std() > 0.1

    # Verify temporal resolution: 250 Hz → 4ms per sample
    dt = 1.0 / atom.sampling_rate
    assert abs(dt - 0.004) < 1e-6

    print(f"  EEG mean={eeg.mean():.2f} µV, std={eeg_std:.2f} µV")
    print(f"  EOG mean={eog.mean():.2f} µV, std={eog.std():.2f} µV")
    print(f"\n✓ Signal characteristics verified: µV range, zero-mean, "
          f"proper amplitude for filtered MI-EEG.")


def test_bci_iv_2a_detect(pool_dir):
    """Verify format detection works."""
    from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter, _detect_bci_iv_2a_mat

    # File-level detection
    assert _detect_bci_iv_2a_mat(BCI_A01T) is True

    # Directory-level detection
    assert BCICompIV2aImporter.detect(BCI_A01T.parent) is True

    # Non-existent file
    assert _detect_bci_iv_2a_mat(Path("nonexistent.mat")) is False

    print(f"\n✓ Format detection verified.")


def test_bci_iv_2a_multi_subject(pool_dir):
    """Import 2 subjects (1 run each), verify pool structure and cross-subject query."""
    from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)
    tc_path = (
        Path(__file__).parent.parent
        / "neuroatom" / "importers" / "task_configs" / "bci_comp_iv_2a.yaml"
    )
    task_config = TaskConfig.from_yaml(tc_path)
    importer = BCICompIV2aImporter(pool, task_config)

    # Import A01 and A02
    a02_path = BCI_A01T.parent / "A02T.mat"
    if not a02_path.exists():
        pytest.skip("A02T.mat not found")

    results_a01 = importer.import_subject(
        BCI_A01T, subject_id="A01", max_runs=1, max_trials=12,
    )
    results_a02 = importer.import_subject(
        a02_path, subject_id="A02", max_runs=1, max_trials=12,
    )

    assert len(results_a01) == 1
    assert len(results_a02) == 1

    # Index
    indexer = Indexer(pool)
    n = indexer.reindex_all()
    assert n == 24  # 12 + 12

    # Query by subject
    qb = QueryBuilder(indexer.backend)
    a01_ids = qb.query_atom_ids({"subject_id": "A01"})
    a02_ids = qb.query_atom_ids({"subject_id": "A02"})
    assert len(a01_ids) == 12
    assert len(a02_ids) == 12

    # No overlap
    assert set(a01_ids).isdisjoint(set(a02_ids))

    # Cross-subject query (all left_hand)
    left_ids = qb.query_atom_ids({
        "annotations": [{"name": "mi_class", "value_in": ["left_hand"]}],
    })
    # First 12 trials of each run: not guaranteed exactly 3 per class
    # but should be > 0 and <= 12 for each subject
    assert len(left_ids) > 0
    assert len(left_ids) <= 24

    print(f"  Left hand trials across 2 subjects: {len(left_ids)}")
    print(f"\n✓ Multi-subject: A01 + A02, 12 trials each, cross-subject query works.")


# Import ChannelType for the first test
from neuroatom.core.enums import ChannelType
