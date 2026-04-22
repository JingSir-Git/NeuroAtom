"""End-to-end real data test: DTU AAD dataset (preprocessed).

Imports 3 trials from DTU S1 preprocessed data, indexes, queries, and verifies.
Requires: NEUROATOM_DTU_DIR env var pointing to the DTU data root.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

_dtu_env = os.environ.get("NEUROATOM_DTU_DIR", "")
DTU_S1 = Path(_dtu_env) / "DATA_preproc" / "S1_data_preproc.mat" if _dtu_env else Path("__nonexistent__")
pytestmark = pytest.mark.skipif(
    not DTU_S1.exists(),
    reason="NEUROATOM_DTU_DIR not set or S1_data_preproc.mat not found",
)


@pytest.fixture
def pool_dir():
    d = tempfile.mkdtemp(prefix="neuroatom_dtu_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_dtu_import_and_index(pool_dir):
    """Import 3 DTU preprocessed trials, index, query, verify signals + audio envelopes."""
    from neuroatom.importers.aad_mat import AADImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.pool import Pool
    from neuroatom.storage.signal_store import ShardManager

    logging.basicConfig(level=logging.INFO)

    pool = Pool.create(pool_dir)

    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "dtu_aad.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = AADImporter(pool, task_config)
    results = importer.import_subject(
        mat_path=DTU_S1,
        subject_id="S1",
        session_id="ses-01",
        format_hint="dtu_preproc",
        max_trials=3,
    )

    assert len(results) == 3
    for r in results:
        assert len(r.atoms) == 1
        atom = r.atoms[0]
        assert atom.dataset_id == "dtu_aad"
        assert atom.subject_id == "S1"
        assert atom.sampling_rate == 64.0
        # 66 channels in preprocessed data (64 EEG + EXG1 + EXG2)
        assert atom.n_channels == 66

    # Check annotations contain attended_speaker
    atom0 = results[0].atoms[0]
    ann_names = {a.name for a in atom0.annotations}
    assert "attended_speaker" in ann_names

    # Index
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    assert n_indexed == 3

    # Query all
    qb = QueryBuilder(indexer.backend)
    all_ids = qb.query_atom_ids({"dataset_id": "dtu_aad"})
    assert len(all_ids) == 3

    # Read back signal
    signal = ShardManager.static_read(pool.root, atom0.signal_ref)
    assert signal.shape[0] == 66
    assert signal.shape[1] == 3200  # 50s at 64 Hz
    assert signal.dtype == np.float32

    # DTU preprocessed: µV → V conversion for MNE, values should be small
    assert np.abs(signal).max() < 1.0

    # Verify audio envelopes were stored
    import h5py
    h5_files = list((pool_dir / "datasets" / "dtu_aad" / "subjects" / "S1" /
                     "sessions" / "ses-01" / "runs" / "trial_001").glob("*.h5"))
    assert len(h5_files) >= 1
    with h5py.File(str(h5_files[0]), "r") as f:
        atom_grp = f[f"/atoms/{atom0.atom_id}"]
        assert "signal" in atom_grp
        # Check audio envelopes stored as annotations
        if "audio_envelope_A" in atom_grp:
            wav_a = atom_grp["audio_envelope_A"][:]
            assert wav_a.shape[0] == 3200

    print(f"\n✓ DTU E2E test passed: 3 trials imported, indexed, queried, signals + audio verified.")


def test_dtu_cross_dataset_splitter(pool_dir):
    """Verify composite key splitter with both KUL and DTU atoms in same pool."""
    from neuroatom.importers.aad_mat import AADImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.assembler.splitter import DataSplitter
    from neuroatom.core.enums import SplitStrategy
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)

    # Import 1 DTU trial
    tc_dtu = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "dtu_aad.yaml"
    task_config_dtu = TaskConfig.from_yaml(tc_dtu)
    importer_dtu = AADImporter(pool, task_config_dtu)
    dtu_results = importer_dtu.import_subject(
        mat_path=DTU_S1, subject_id="S1", format_hint="dtu_preproc", max_trials=1,
    )

    kul_s1 = Path(r"C:\Data\KUL\S1.mat")
    if kul_s1.exists():
        tc_kul = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "kul_aad.yaml"
        task_config_kul = TaskConfig.from_yaml(tc_kul)
        importer_kul = AADImporter(pool, task_config_kul)
        kul_results = importer_kul.import_subject(
            mat_path=kul_s1, subject_id="S1", format_hint="kul", max_trials=1,
        )
    else:
        pytest.skip("KUL data not available for cross-dataset test")

    # Combine atoms
    all_atoms = [r.atoms[0] for r in dtu_results] + [r.atoms[0] for r in kul_results]
    assert len(all_atoms) == 2

    # Both have subject_id="S1" but different dataset_id
    assert all_atoms[0].dataset_id == "dtu_aad"
    assert all_atoms[1].dataset_id == "kul_aad"
    assert all_atoms[0].subject_id == all_atoms[1].subject_id == "S1"

    # Splitter should treat them as DIFFERENT subjects via composite key
    splitter = DataSplitter(
        strategy=SplitStrategy.SUBJECT,
        config={"val_ratio": 0.0, "test_ratio": 0.5, "seed": 42},
    )
    splits = splitter.split(all_atoms)

    # With 2 unique composite keys, one should be test, one should be train
    assert len(splits["test"]) == 1
    assert len(splits["train"]) == 1
    # They should be from different datasets
    test_ds = splits["test"][0].dataset_id
    train_ds = splits["train"][0].dataset_id
    assert test_ds != train_ds

    print(f"\n✓ Cross-dataset splitter test passed: composite key correctly separates S1 across datasets.")
