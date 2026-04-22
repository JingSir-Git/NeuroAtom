"""Unit tests for NeuroAtom storage layer: ShardManager, metadata I/O, Pool."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import AtomType
from neuroatom.core.run import RunMeta
from neuroatom.core.session import SessionMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.storage.metadata_store import (
    AtomJSONLReader,
    AtomJSONLWriter,
    compute_jsonl_hash,
    read_json,
    write_json,
)
from neuroatom.storage.paths import (
    atoms_jsonl_path,
    dataset_dir,
    run_dir,
    shard_filename,
    shard_relative_path,
)
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_ref(shard_index: int = 0, atom_id: str = "a1") -> SignalRef:
    return SignalRef(
        file_path=shard_relative_path("ds", "s01", "ses-01", "run-01", shard_index),
        internal_path=f"/atoms/{atom_id}/signal",
        shape=(8, 256),
    )


def _make_atom(atom_id: str, shard_index: int = 0) -> Atom:
    return Atom(
        atom_id=atom_id,
        atom_type=AtomType.TRIAL,
        dataset_id="ds",
        subject_id="s01",
        session_id="ses-01",
        run_id="run-01",
        trial_index=0,
        signal_ref=_make_signal_ref(shard_index=shard_index, atom_id=atom_id),
        temporal=TemporalInfo(
            onset_sample=0,
            onset_seconds=0.0,
            duration_samples=256,
            duration_seconds=1.0,
        ),
        channel_ids=[f"ch_{i}" for i in range(8)],
        n_channels=8,
        sampling_rate=256.0,
    )


# ---------------------------------------------------------------------------
# ShardManager
# ---------------------------------------------------------------------------

class TestShardManager:
    def test_write_and_read_single_atom(self, tmp_path):
        signal = np.random.randn(8, 256).astype(np.float32)
        with ShardManager(
            pool_root=tmp_path,
            dataset_id="ds",
            subject_id="s01",
            session_id="ses-01",
            run_id="run-01",
        ) as mgr:
            ref = mgr.write_atom_signal("atom_001", signal)

        assert ref.shard_index == 0
        assert ref.shape == (8, 256)
        assert "signals_000.h5" in ref.file_path

        # Read back
        loaded = ShardManager.static_read(tmp_path, ref)
        np.testing.assert_allclose(loaded, signal, atol=1e-6)

    def test_write_with_annotations(self, tmp_path):
        signal = np.random.randn(8, 256).astype(np.float32)
        envelope = np.random.randn(256).astype(np.float32)

        with ShardManager(
            tmp_path, "ds", "s01", "ses-01", "run-01"
        ) as mgr:
            ref = mgr.write_atom_signal(
                "atom_002", signal, annotations={"audio_env": envelope}
            )
            ann = mgr.read_annotation(ref, "audio_env")

        np.testing.assert_allclose(ann, envelope, atol=1e-6)

    def test_multiple_atoms(self, tmp_path):
        refs = []
        with ShardManager(
            tmp_path, "ds", "s01", "ses-01", "run-01"
        ) as mgr:
            for i in range(5):
                signal = np.random.randn(8, 256).astype(np.float32)
                ref = mgr.write_atom_signal(f"atom_{i:03d}", signal)
                refs.append(ref)

        # All in same shard (total ~40KB << 200MB)
        assert all(r.shard_index == 0 for r in refs)

        # All readable
        for ref in refs:
            data = ShardManager.static_read(tmp_path, ref)
            assert data.shape == (8, 256)

    def test_smart_sharding_splits(self, tmp_path):
        """With very small max_shard_size, shards should split."""
        refs = []
        with ShardManager(
            tmp_path, "ds", "s01", "ses-01", "run-01",
            max_shard_size_mb=0.01,  # 10KB threshold → forces splitting
        ) as mgr:
            for i in range(10):
                signal = np.random.randn(8, 256).astype(np.float32)
                ref = mgr.write_atom_signal(f"atom_{i:03d}", signal)
                refs.append(ref)

        # Should have multiple shards
        shard_indices = set(r.shard_index for r in refs)
        assert len(shard_indices) > 1, f"Expected multiple shards, got {shard_indices}"

        # All still readable
        for ref in refs:
            data = ShardManager.static_read(tmp_path, ref)
            assert data.shape == (8, 256)

    def test_single_atom_exceeds_threshold(self, tmp_path):
        """A single atom larger than threshold should still be accepted."""
        large_signal = np.random.randn(128, 10000).astype(np.float32)  # ~5MB

        with ShardManager(
            tmp_path, "ds", "s01", "ses-01", "run-01",
            max_shard_size_mb=0.001,  # 1KB threshold
        ) as mgr:
            ref = mgr.write_atom_signal("big_atom", large_signal)

        loaded = ShardManager.static_read(tmp_path, ref)
        np.testing.assert_allclose(loaded, large_signal, atol=1e-6)


# ---------------------------------------------------------------------------
# Metadata Store (JSONL)
# ---------------------------------------------------------------------------

class TestAtomJSONL:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "atoms.jsonl"
        atom1 = _make_atom("a1")
        atom2 = _make_atom("a2")

        with AtomJSONLWriter(path) as writer:
            writer.write_atoms([atom1, atom2])

        reader = AtomJSONLReader(path)
        atoms = reader.read_all()
        assert len(atoms) == 2
        assert atoms[0].atom_id == "a1"
        assert atoms[1].atom_id == "a2"

    def test_iter_atoms(self, tmp_path):
        path = tmp_path / "atoms.jsonl"
        with AtomJSONLWriter(path) as writer:
            for i in range(5):
                writer.write_atom(_make_atom(f"a{i}"))

        reader = AtomJSONLReader(path)
        ids = [a.atom_id for a in reader.iter_atoms()]
        assert ids == [f"a{i}" for i in range(5)]

    def test_count(self, tmp_path):
        path = tmp_path / "atoms.jsonl"
        with AtomJSONLWriter(path) as writer:
            for i in range(3):
                writer.write_atom(_make_atom(f"a{i}"))

        reader = AtomJSONLReader(path)
        assert reader.count() == 3

    def test_get_atom_ids(self, tmp_path):
        path = tmp_path / "atoms.jsonl"
        with AtomJSONLWriter(path) as writer:
            writer.write_atoms([_make_atom("x"), _make_atom("y")])

        reader = AtomJSONLReader(path)
        assert reader.get_atom_ids() == ["x", "y"]

    def test_hash_consistency(self, tmp_path):
        path = tmp_path / "atoms.jsonl"
        with AtomJSONLWriter(path) as writer:
            writer.write_atom(_make_atom("a1"))

        h1 = compute_jsonl_hash(path)
        h2 = compute_jsonl_hash(path)
        assert h1 == h2

        # Modify file → hash changes
        with open(path, "a") as f:
            f.write('{"extra": true}\n')
        h3 = compute_jsonl_hash(path)
        assert h3 != h1


# ---------------------------------------------------------------------------
# Metadata Store (JSON)
# ---------------------------------------------------------------------------

class TestJSONMetadata:
    def test_write_and_read_dataset(self, tmp_path):
        meta = DatasetMeta(
            dataset_id="test", name="Test Dataset",
            task_types=["motor_imagery"], n_subjects=2,
        )
        path = tmp_path / "dataset.json"
        write_json(meta, path)
        loaded = read_json(path, DatasetMeta)
        assert loaded.dataset_id == "test"
        assert loaded.n_subjects == 2

    def test_write_and_read_subject(self, tmp_path):
        meta = SubjectMeta(
            subject_id="sub-01", dataset_id="test", age=25.0, sex="M",
        )
        path = tmp_path / "subject.json"
        write_json(meta, path)
        loaded = read_json(path, SubjectMeta)
        assert loaded.age == 25.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

class TestPaths:
    def test_shard_filename(self):
        assert shard_filename(0) == "signals_000.h5"
        assert shard_filename(42) == "signals_042.h5"

    def test_shard_relative_path(self):
        rel = shard_relative_path("ds", "s01", "ses-01", "run-01", 0)
        assert rel == "datasets/ds/subjects/s01/sessions/ses-01/runs/run-01/signals_000.h5"

    def test_run_dir(self, tmp_path):
        rd = run_dir(tmp_path, "ds", "s01", "ses-01", "run-01")
        assert rd.name == "run-01"
        assert "sessions" in str(rd)


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class TestPool:
    def test_create_and_open(self, tmp_path):
        pool_root = tmp_path / "my_pool"
        pool = Pool.create(pool_root)
        assert pool.root == pool_root.resolve()
        assert (pool_root / "pool.yaml").exists()
        assert (pool_root / "datasets").is_dir()
        assert (pool_root / "stimuli").is_dir()

        # Re-open
        pool2 = Pool(pool_root)
        assert pool2.root == pool.root

    def test_dataset_lifecycle(self, tmp_path):
        pool = Pool.create(tmp_path / "pool")
        meta = DatasetMeta(
            dataset_id="bci4", name="BCI IV",
            task_types=["motor_imagery"],
        )
        pool.register_dataset(meta)
        assert "bci4" in pool.list_datasets()

        loaded = pool.get_dataset_meta("bci4")
        assert loaded.name == "BCI IV"

        pool.delete_dataset("bci4")
        assert "bci4" not in pool.list_datasets()

    def test_subject_session_run_lifecycle(self, tmp_path):
        pool = Pool.create(tmp_path / "pool")

        # Register hierarchy
        pool.register_dataset(DatasetMeta(dataset_id="ds", name="D"))
        pool.register_subject(SubjectMeta(subject_id="s01", dataset_id="ds"))
        pool.register_session(SessionMeta(
            session_id="ses-01", subject_id="s01", dataset_id="ds",
            sampling_rate=256.0,
        ))
        pool.register_run(RunMeta(
            run_id="run-01", session_id="ses-01", subject_id="s01",
            dataset_id="ds", task_type="motor_imagery", run_index=0,
        ))

        assert pool.list_subjects("ds") == ["s01"]
        assert pool.list_sessions("ds", "s01") == ["ses-01"]
        assert pool.list_runs("ds", "s01", "ses-01") == ["run-01"]

        run = pool.get_run_meta("ds", "s01", "ses-01", "run-01")
        assert run.run_index == 0
        assert run.task_type == "motor_imagery"

    def test_dataset_lock(self, tmp_path):
        pool = Pool.create(tmp_path / "pool")
        pool.register_dataset(DatasetMeta(dataset_id="ds", name="D"))
        lock = pool.dataset_lock("ds")
        with lock:
            # Lock acquired — would block other writers
            pass
