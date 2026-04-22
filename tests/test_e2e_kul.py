"""End-to-end real data test: KUL AAD dataset.

Imports 2 trials from KUL S1, indexes, queries, and verifies the full pipeline.
Requires: C:\\Data\\KUL\\S1.mat to exist.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip if KUL data not available
_kul_env = os.environ.get("NEUROATOM_KUL_DIR", "")
KUL_S1 = Path(_kul_env) / "S1.mat" if _kul_env else Path("__nonexistent__")
pytestmark = pytest.mark.skipif(
    not KUL_S1.exists(),
    reason="NEUROATOM_KUL_DIR not set or S1.mat not found",
)


@pytest.fixture
def pool_dir():
    """Create a temporary pool directory for testing."""
    d = tempfile.mkdtemp(prefix="neuroatom_kul_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_kul_import_and_index(pool_dir):
    """Import 2 KUL trials, index, query, and verify signals."""
    from neuroatom.importers.aad_mat import AADImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.pool import Pool
    from neuroatom.storage.signal_store import ShardManager

    logging.basicConfig(level=logging.INFO)

    # 1. Create pool
    pool = Pool.create(pool_dir)

    # 2. Load task config
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "kul_aad.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    # 3. Import 2 trials
    importer = AADImporter(pool, task_config)
    results = importer.import_subject(
        mat_path=KUL_S1,
        subject_id="S1",
        session_id="ses-01",
        format_hint="kul",
        max_trials=2,
    )

    assert len(results) == 2
    for r in results:
        assert len(r.atoms) == 1
        atom = r.atoms[0]
        assert atom.dataset_id == "kul_aad"
        assert atom.subject_id == "S1"
        assert atom.sampling_rate == 128.0
        assert atom.n_channels == 64

    # Check annotations
    atom0 = results[0].atoms[0]
    ann_names = {a.name for a in atom0.annotations}
    assert "attended_ear" in ann_names
    assert "condition" in ann_names

    # 4. Index
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    assert n_indexed == 2

    # 5. Query: all atoms
    qb = QueryBuilder(indexer.backend)
    all_ids = qb.query_atom_ids({"dataset_id": "kul_aad"})
    assert len(all_ids) == 2

    # 6. Query by subject
    s1_ids = qb.query_atom_ids({"subject_id": "S1"})
    assert len(s1_ids) == 2

    # 7. Read back signal from HDF5
    atom0 = results[0].atoms[0]
    signal = ShardManager.static_read(pool.root, atom0.signal_ref)
    assert signal.shape[0] == 64  # 64 channels
    assert signal.shape[1] > 40000  # ~50k samples for trial 1

    # Verify signal is in V (MNE convention: data stored as V, original was µV)
    assert signal.dtype == np.float32
    # µV → V conversion: values should be tiny
    assert np.abs(signal).max() < 1.0  # Should be in V range, not µV

    # 8. Verify JSONL round-trip
    from neuroatom.storage.metadata_store import AtomJSONLReader
    from neuroatom.storage import paths as P

    jsonl0 = P.atoms_jsonl_path(pool.root, "kul_aad", "S1", "ses-01", "trial_001")
    reader = AtomJSONLReader(jsonl0)
    loaded_atoms = list(reader.iter_atoms())
    assert len(loaded_atoms) == 1
    loaded = loaded_atoms[0]
    assert loaded.atom_id == atom0.atom_id
    assert loaded.n_channels == 64
    assert loaded.sampling_rate == 128.0

    print(f"\n✓ KUL E2E test passed: 2 trials imported, indexed, queried, signals verified.")


def test_kul_pool_structure(pool_dir):
    """Verify the pool directory structure after import."""
    from neuroatom.importers.aad_mat import AADImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "kul_aad.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = AADImporter(pool, task_config)
    importer.import_subject(
        mat_path=KUL_S1,
        subject_id="S1",
        session_id="ses-01",
        format_hint="kul",
        max_trials=1,
    )

    # Verify directory structure
    ds_dir = pool_dir / "datasets" / "kul_aad"
    assert ds_dir.exists()
    assert (ds_dir / "dataset.json").exists()

    sub_dir = ds_dir / "subjects" / "S1"
    assert sub_dir.exists()
    assert (sub_dir / "subject.json").exists()

    ses_dir = sub_dir / "sessions" / "ses-01"
    assert ses_dir.exists()

    run_dir = ses_dir / "runs" / "trial_001"
    assert run_dir.exists()
    assert (run_dir / "run.json").exists()

    # Check for atoms.jsonl and HDF5 shards
    jsonl_files = list(run_dir.glob("*.jsonl"))
    assert len(jsonl_files) >= 1

    h5_files = list(run_dir.glob("*.h5"))
    assert len(h5_files) >= 1

    print(f"\n✓ KUL pool structure verified.")
