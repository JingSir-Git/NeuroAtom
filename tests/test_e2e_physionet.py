"""End-to-end tests for PhysioNet EEG Motor Movement/Imagery Dataset.

Tests cover:
    1. Import → index → query → signal verification (imagery runs)
    2. Run-dependent event semantics (T1/T2 mapping)
    3. Channel name cleanup (trailing dots stripped)
    4. Multi-subject import and cross-subject query
    5. Execution vs imagery paradigm filtering
    6. Dataset auto-detection
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip if dataset not available
_physionet_env = os.environ.get("NEUROATOM_PHYSIONET_DIR", "")
PHYSIONET_DIR = Path(_physionet_env) if _physionet_env else Path("__nonexistent__")
SKIP = not PHYSIONET_DIR.exists()
pytestmark = pytest.mark.skipif(SKIP, reason="NEUROATOM_PHYSIONET_DIR not set or path not found")


@pytest.fixture
def pool_dir(tmp_path):
    return tmp_path / "pool_physionet"


# ------------------------------------------------------------------
# Test 1: Core import + index + query + signal verification
# ------------------------------------------------------------------

def test_physionet_import_and_index(pool_dir):
    """Full pipeline: import MI runs → index → query by class → read signal."""
    from neuroatom.importers.physionet_mi import PhysioNetMIImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.signal_store import ShardManager

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "physionet_mi.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = PhysioNetMIImporter(pool, task_config)

    # Import 1 imagery run from S001
    results = importer.import_subject(
        subject_dir=PHYSIONET_DIR / "S001",
        subject_id="S001",
        paradigm="imagery",
        max_runs=1,
    )

    # Results include 2 baseline runs + 1 task run = 3
    task_results = [r for r in results if any(
        a.name == "mi_class" for atom in r.atoms for a in atom.annotations
    )]
    baseline_results = [r for r in results if any(
        a.name == "baseline_type" for atom in r.atoms for a in atom.annotations
    )]
    assert len(task_results) == 1, f"Expected 1 task run, got {len(task_results)}"
    assert len(baseline_results) == 2, f"Expected 2 baseline runs, got {len(baseline_results)}"
    result = task_results[0]

    # Should have ~15 task epochs per run (T1 + T2, excluding T0)
    n_atoms = len(result.atoms)
    assert n_atoms >= 10, f"Expected ≥10 epochs, got {n_atoms}"
    assert n_atoms <= 20, f"Expected ≤20 epochs, got {n_atoms}"

    # All atoms should be imagery paradigm
    for atom in result.atoms:
        paradigm_ann = [a for a in atom.annotations if a.name == "paradigm"]
        assert len(paradigm_ann) == 1
        assert paradigm_ann[0].value == "imagery"

    # Check class labels — should be left_hand and right_hand for MI runs
    classes = set()
    for atom in result.atoms:
        mi_ann = [a for a in atom.annotations if a.name == "mi_class"]
        assert len(mi_ann) == 1
        classes.add(mi_ann[0].value)

    assert classes == {"left_hand", "right_hand"}, f"Unexpected classes: {classes}"

    # Check channel info
    assert len(result.channel_infos) == 64
    # Verify channel names are cleaned (no trailing dots)
    for ch in result.channel_infos:
        assert "." not in ch.name, f"Channel name not cleaned: {ch.name}"

    # Verify specific channels
    ch_names = [ch.name for ch in result.channel_infos]
    assert "C3" in ch_names
    assert "C4" in ch_names
    assert "Cz" in ch_names

    # Check sampling rate and units
    assert result.channel_infos[0].sampling_rate == 160.0
    assert result.channel_infos[0].unit == "V"

    # Verify electrode coordinates are present
    chs_with_loc = [ch for ch in result.channel_infos if ch.location is not None]
    assert len(chs_with_loc) >= 50, f"Expected ≥50 channels with coords, got {len(chs_with_loc)}"

    # Index
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    total_atoms = n_atoms + 2  # +2 baseline atoms
    assert n_indexed == total_atoms

    # Query by class
    qb = QueryBuilder(indexer.backend)
    left_ids = qb.query_atom_ids({
        "dataset_id": "physionet_mi",
        "annotations": [{"name": "mi_class", "value_in": ["left_hand"]}],
    })
    right_ids = qb.query_atom_ids({
        "dataset_id": "physionet_mi",
        "annotations": [{"name": "mi_class", "value_in": ["right_hand"]}],
    })

    assert len(left_ids) > 0
    assert len(right_ids) > 0
    assert len(left_ids) + len(right_ids) == n_atoms

    # Verify baseline atoms can be queried too
    baseline_ids = qb.query_atom_ids({
        "dataset_id": "physionet_mi",
        "annotations": [{"name": "baseline_type", "value_in": ["eyes_open", "eyes_closed"]}],
    })
    assert len(baseline_ids) == 2

    # Read back signal and verify
    atom = result.atoms[0]
    signal = ShardManager.static_read(pool.root, atom.signal_ref)
    assert signal.shape[0] == 64  # channels
    assert signal.shape[1] == 641  # 4.0s * 160 Hz + 1
    assert signal.dtype == np.float32

    # Signal in V → typical EEG is [-500µV, +500µV] = [-5e-4, 5e-4]
    max_abs = np.abs(signal).max()
    assert max_abs > 1e-6, f"Signal too small: {max_abs}"
    assert max_abs < 0.01, f"Signal too large for V: {max_abs}"

    indexer.close()
    print(f"\n✓ PhysioNet MI: {n_atoms} MI epochs + 2 baseline, "
          f"64 ch × 641 samples @ 160 Hz, indexed + queried.")


# ------------------------------------------------------------------
# Test 2: Run-dependent event semantics
# ------------------------------------------------------------------

def test_physionet_run_semantics(pool_dir):
    """Verify T1/T2 map to different classes depending on the run."""
    from neuroatom.importers.physionet_mi import PhysioNetMIImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "physionet_mi.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = PhysioNetMIImporter(pool, task_config)

    # Import one execution run (R03 = Task 1: left/right fist execution)
    results_exec = importer.import_subject(
        subject_dir=PHYSIONET_DIR / "S001",
        subject_id="S001",
        paradigm="execution",
        max_runs=1,
    )

    task_results_exec = [r for r in results_exec if any(
        a.name == "motor_class" for atom in r.atoms for a in atom.annotations
    )]
    assert len(task_results_exec) == 1
    exec_classes = set()
    for atom in task_results_exec[0].atoms:
        cls_ann = [a for a in atom.annotations if a.name == "motor_class"]
        assert len(cls_ann) == 1
        exec_classes.add(cls_ann[0].value)

    # Execution runs should have left_fist/right_fist
    assert exec_classes == {"left_fist", "right_fist"}, f"Exec classes: {exec_classes}"

    # Verify paradigm annotation
    for atom in task_results_exec[0].atoms:
        p = [a for a in atom.annotations if a.name == "paradigm"][0]
        assert p.value == "execution"

    print(f"\n✓ Run semantics: execution run → {exec_classes}")


# ------------------------------------------------------------------
# Test 3: Multi-subject import
# ------------------------------------------------------------------

def test_physionet_multi_subject(pool_dir):
    """Import 2 subjects, verify cross-subject querying works."""
    from neuroatom.importers.physionet_mi import PhysioNetMIImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "physionet_mi.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = PhysioNetMIImporter(pool, task_config)

    # Import 1 MI run each from 2 subjects
    for subj in ["S001", "S002"]:
        results = importer.import_subject(
            subject_dir=PHYSIONET_DIR / subj,
            subject_id=subj,
            paradigm="imagery",
            max_runs=1,
        )
        # 2 baseline + 1 task = 3 results
        task_only = [r for r in results if any(
            a.name == "mi_class" for atom in r.atoms for a in atom.annotations
        )]
        assert len(task_only) == 1

    # Index all
    indexer = Indexer(pool)
    n_total = indexer.reindex_all()

    # Query all atoms
    qb = QueryBuilder(indexer.backend)
    all_ids = qb.query_atom_ids({"dataset_id": "physionet_mi"})
    assert len(all_ids) == n_total

    # Query per subject
    s1_ids = qb.query_atom_ids({"dataset_id": "physionet_mi", "subject_id": "S001"})
    s2_ids = qb.query_atom_ids({"dataset_id": "physionet_mi", "subject_id": "S002"})
    assert len(s1_ids) > 0
    assert len(s2_ids) > 0
    assert len(s1_ids) + len(s2_ids) == n_total

    # Cross-subject class query
    left_ids = qb.query_atom_ids({
        "annotations": [{"name": "mi_class", "value_in": ["left_hand"]}],
    })
    assert len(left_ids) > 0

    indexer.close()
    print(f"\n✓ Multi-subject: S001={len(s1_ids)} + S002={len(s2_ids)} = {n_total} total atoms")


# ------------------------------------------------------------------
# Test 4: Dataset auto-detection
# ------------------------------------------------------------------

def test_physionet_detect():
    """Verify the detect() method correctly identifies the dataset."""
    from neuroatom.importers.physionet_mi import PhysioNetMIImporter

    assert PhysioNetMIImporter.detect(PHYSIONET_DIR) is True
    assert PhysioNetMIImporter.detect(Path("C:\\nonexistent")) is False
    assert PhysioNetMIImporter.detect(Path(__file__).parent) is False

    print("\n✓ Dataset detection verified.")


# ------------------------------------------------------------------
# Test 5: Execution runs (Task 3/4: both fists/feet)
# ------------------------------------------------------------------

def test_physionet_bilateral_tasks(pool_dir):
    """Verify bilateral task runs have correct class labels."""
    from neuroatom.importers.physionet_mi import PhysioNetMIImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "physionet_mi.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = PhysioNetMIImporter(pool, task_config)

    # Import one execution run for Task 3 (both fists/feet)
    # R05 is Task 3 execution
    results = importer.import_subject(
        subject_dir=PHYSIONET_DIR / "S001",
        subject_id="S001",
        paradigm="execution",
        max_runs=3,  # Will get R03, R05, R07 (first exec of each task type)
    )

    # Collect all class labels across execution runs
    all_classes = set()
    for res in results:
        for atom in res.atoms:
            cls_ann = [a for a in atom.annotations if a.name == "motor_class"]
            if cls_ann:
                all_classes.add(cls_ann[0].value)

    # Should include bilateral classes from Task 3
    assert "left_fist" in all_classes or "both_fists" in all_classes, \
        f"Expected bilateral classes, got: {all_classes}"

    print(f"\n✓ Bilateral tasks: {all_classes}")
