"""Integration tests: import → index → query round-trip.

Uses synthetic data to validate the full pipeline from atom creation
through SQLite indexing to query execution via the Query DSL.
"""

import numpy as np
import pytest

import mne

from neuroatom.atomizer.trial import TrialAtomizer
from neuroatom.atomizer.window import WindowAtomizer
from neuroatom.core.annotation import CategoricalAnnotation, NumericAnnotation
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import AtomType, QualityStatus
from neuroatom.core.quality import QualityInfo
from neuroatom.core.run import RunMeta
from neuroatom.core.session import SessionMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import TaskConfig
from neuroatom.index.indexer import Indexer
from neuroatom.index.query import QueryBuilder
from neuroatom.index.sqlite_backend import SQLiteBackend
from neuroatom.storage import paths as P
from neuroatom.storage.metadata_store import AtomJSONLWriter
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _setup_pool_with_data(tmp_path):
    """Create a pool with 2 datasets, multiple subjects, and diverse atoms."""
    pool = Pool.create(tmp_path / "pool")
    pool_root = pool.root

    # Dataset 1: motor_imagery, 2 subjects, 3 trials each
    ds1_id = "mi_dataset"
    pool.register_dataset(DatasetMeta(
        dataset_id=ds1_id, name="MI Dataset",
        task_types=["motor_imagery"], n_subjects=2,
    ))

    for sub_idx in range(2):
        sub_id = f"sub-{sub_idx:02d}"
        ses_id = "ses-01"
        run_id = "run-01"

        pool.register_subject(SubjectMeta(subject_id=sub_id, dataset_id=ds1_id))
        pool.register_session(SessionMeta(
            session_id=ses_id, subject_id=sub_id, dataset_id=ds1_id,
            sampling_rate=256.0,
        ))
        pool.register_run(RunMeta(
            run_id=run_id, session_id=ses_id, subject_id=sub_id,
            dataset_id=ds1_id, task_type="motor_imagery", run_index=0,
        ))

        # Create atoms
        atoms = []
        for trial_idx in range(3):
            label = "left_hand" if trial_idx % 2 == 0 else "right_hand"
            quality = QualityStatus.GOOD if trial_idx < 2 else QualityStatus.SUSPECT

            atom_id = f"{ds1_id}_{sub_id}_{trial_idx:03d}"
            atom = Atom(
                atom_id=atom_id,
                atom_type=AtomType.TRIAL,
                dataset_id=ds1_id,
                subject_id=sub_id,
                session_id=ses_id,
                run_id=run_id,
                trial_index=trial_idx,
                signal_ref=SignalRef(
                    file_path="__placeholder__",
                    internal_path=f"/atoms/{atom_id}/signal",
                    shape=(8, 512),
                ),
                temporal=TemporalInfo(
                    onset_sample=trial_idx * 1024,
                    onset_seconds=trial_idx * 4.0,
                    duration_samples=512,
                    duration_seconds=2.0,
                ),
                channel_ids=[f"ch_{i:03d}" for i in range(8)],
                n_channels=8,
                sampling_rate=256.0,
                annotations=[
                    CategoricalAnnotation(
                        annotation_id=f"ann_{atom_id}",
                        name="mi_class",
                        value=label,
                    ),
                ],
                quality=QualityInfo(overall_status=quality),
            )
            atoms.append(atom)

        # Write signals + JSONL
        with ShardManager(pool_root, ds1_id, sub_id, ses_id, run_id) as mgr:
            jsonl_path = P.atoms_jsonl_path(pool_root, ds1_id, sub_id, ses_id, run_id)
            with AtomJSONLWriter(jsonl_path) as writer:
                for atom in atoms:
                    signal = np.random.randn(8, 512).astype(np.float32)
                    ref = mgr.write_atom_signal(atom.atom_id, signal)
                    atom.signal_ref = ref
                    writer.write_atom(atom)

    # Dataset 2: p300, 1 subject, high srate
    ds2_id = "p300_dataset"
    pool.register_dataset(DatasetMeta(
        dataset_id=ds2_id, name="P300 Dataset",
        task_types=["p300"], n_subjects=1,
    ))
    sub_id = "sub-00"
    ses_id = "ses-01"
    run_id = "run-01"
    pool.register_subject(SubjectMeta(subject_id=sub_id, dataset_id=ds2_id))
    pool.register_session(SessionMeta(
        session_id=ses_id, subject_id=sub_id, dataset_id=ds2_id,
        sampling_rate=512.0,
    ))
    pool.register_run(RunMeta(
        run_id=run_id, session_id=ses_id, subject_id=sub_id,
        dataset_id=ds2_id, task_type="p300", run_index=0,
    ))

    atoms_p300 = []
    for trial_idx in range(4):
        label = "target" if trial_idx == 0 else "non_target"
        atom_id = f"{ds2_id}_{sub_id}_{trial_idx:03d}"
        atom = Atom(
            atom_id=atom_id,
            atom_type=AtomType.EVENT_EPOCH,
            dataset_id=ds2_id,
            subject_id=sub_id,
            session_id=ses_id,
            run_id=run_id,
            trial_index=trial_idx,
            signal_ref=SignalRef(
                file_path="__placeholder__",
                internal_path=f"/atoms/{atom_id}/signal",
                shape=(32, 512),
            ),
            temporal=TemporalInfo(
                onset_sample=trial_idx * 512,
                onset_seconds=trial_idx * 1.0,
                duration_samples=512,
                duration_seconds=1.0,
            ),
            channel_ids=[f"ch_{i:03d}" for i in range(32)],
            n_channels=32,
            sampling_rate=512.0,
            annotations=[
                CategoricalAnnotation(
                    annotation_id=f"ann_{atom_id}",
                    name="p300_class",
                    value=label,
                ),
                NumericAnnotation(
                    annotation_id=f"ann_rt_{atom_id}",
                    name="reaction_time",
                    numeric_value=0.3 + trial_idx * 0.1,
                ),
            ],
            quality=QualityInfo(overall_status=QualityStatus.GOOD),
        )
        atoms_p300.append(atom)

    with ShardManager(pool_root, ds2_id, sub_id, ses_id, run_id) as mgr:
        jsonl_path = P.atoms_jsonl_path(pool_root, ds2_id, sub_id, ses_id, run_id)
        with AtomJSONLWriter(jsonl_path) as writer:
            for atom in atoms_p300:
                signal = np.random.randn(32, 512).astype(np.float32)
                ref = mgr.write_atom_signal(atom.atom_id, signal)
                atom.signal_ref = ref
                writer.write_atom(atom)

    return pool  # Total: 6 MI + 4 P300 = 10 atoms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_full_reindex(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        total = indexer.reindex_all()
        assert total == 10

        stats = indexer.get_stats()
        assert stats["total_atoms"] == 10
        assert stats["per_dataset"]["mi_dataset"] == 6
        assert stats["per_dataset"]["p300_dataset"] == 4
        indexer.close()

    def test_incremental_index(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)

        # First index
        total1 = indexer.index_incremental()
        assert total1 == 10

        # Second index (no changes) → 0 new
        total2 = indexer.index_incremental()
        assert total2 == 0

        indexer.close()


class TestQueryDSL:
    def test_query_by_dataset(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({"dataset_id": ["mi_dataset"]})
        assert len(ids) == 6

        ids2 = qb.query_atom_ids({"dataset_id": ["p300_dataset"]})
        assert len(ids2) == 4

        indexer.close()

    def test_query_by_subject(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({
            "dataset_id": ["mi_dataset"],
            "subject_id": ["sub-00"],
        })
        assert len(ids) == 3

        indexer.close()

    def test_query_by_annotation(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)

        # Query by annotation name + value (existential)
        ids = qb.query_atom_ids({
            "dataset_id": ["mi_dataset"],
            "annotations": [
                {"name": "mi_class", "value_in": ["left_hand"]},
            ],
        })
        # 2 subjects × 2 left_hand trials each = 4
        assert len(ids) == 4

        indexer.close()

    def test_query_by_quality(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({
            "quality": {"overall_status": ["good"]},
        })
        # MI: 2 subjects × 2 good = 4; P300: 4 good
        assert len(ids) == 8

        indexer.close()

    def test_query_by_sampling_rate(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({"sampling_rate_min": 500.0})
        assert len(ids) == 4  # Only p300 at 512 Hz

        indexer.close()

    def test_query_by_duration(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({"duration_seconds_min": 1.5})
        assert len(ids) == 6  # MI atoms: 2s duration

        indexer.close()

    def test_query_by_channels_min(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({"channels_min": 16})
        assert len(ids) == 4  # Only P300 with 32 channels

        indexer.close()

    def test_query_by_atom_type(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({"atom_type": ["event_epoch"]})
        assert len(ids) == 4

        indexer.close()

    def test_query_count(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        count = qb.query_count({"dataset_id": ["mi_dataset"]})
        assert count == 6

        indexer.close()

    def test_compound_query(self, tmp_path):
        """Test multiple filters combined (AND logic)."""
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({
            "dataset_id": ["mi_dataset"],
            "quality": {"overall_status": ["good"]},
            "annotations": [
                {"name": "mi_class", "value_in": ["left_hand"]},
            ],
        })
        # MI: 2 subjects × left_hand trials that are also good
        # sub-00: trial 0 (left, good), trial 2 (left, suspect)
        # sub-01: trial 0 (left, good), trial 2 (left, suspect)
        # → 2 atoms match
        assert len(ids) == 2

        indexer.close()

    def test_empty_query_returns_all(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({})
        assert len(ids) == 10

        indexer.close()

    def test_annotation_value_not_in(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        qb = QueryBuilder(indexer.backend)
        ids = qb.query_atom_ids({
            "dataset_id": ["mi_dataset"],
            "annotations": [
                {"name": "mi_class", "value_not_in": ["right_hand"]},
            ],
        })
        # Only left_hand atoms: 2 subjects × 2 = 4
        assert len(ids) == 4

        indexer.close()


class TestIndexerStats:
    def test_stats(self, tmp_path):
        pool = _setup_pool_with_data(tmp_path)
        indexer = Indexer(pool)
        indexer.reindex_all()

        stats = indexer.get_stats()
        assert stats["total_atoms"] == 10
        assert "mi_class:left_hand" in stats["label_distribution"]
        assert "p300_class:target" in stats["label_distribution"]
        assert 256.0 in stats["sampling_rates"]
        assert 512.0 in stats["sampling_rates"]

        indexer.close()
