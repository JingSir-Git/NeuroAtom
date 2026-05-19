"""Concurrency and performance regression tests for Sprint 5.

Covers:
- byte_offset fast-path in DatasetAssembler._load_atoms_by_ids
- Fallback to linear scan when offsets are NULL (legacy index)
- Multi-worker DataLoader with worker_init_fn
- dataset_lock serializes concurrent imports to the same dataset
- Migration 0.1.0 → 0.2.0 (byte_offset reindex)
"""

import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────
# Helpers: build a tiny synthetic pool without going through real importers
# ──────────────────────────────────────────────────────────────────────────


def _build_mini_pool(tmp_path, n_atoms: int = 10):
    """Create a minimal pool with `n_atoms` synthetic atoms under one run.

    Returns (pool, indexer, atom_ids).
    """
    from neuroatom.core.atom import Atom, TemporalInfo
    from neuroatom.core.enums import AtomType
    from neuroatom.core.run import RunMeta
    from neuroatom.core.signal_ref import SignalRef
    from neuroatom.index.indexer import Indexer
    from neuroatom.storage import paths as P
    from neuroatom.storage.metadata_store import AtomJSONLWriter
    from neuroatom.storage.pool import Pool
    from neuroatom.storage.signal_store import ShardManager

    pool = Pool.create(tmp_path / "pool")
    pool.ensure_dataset("ds")
    pool.ensure_subject("ds", "s01")
    pool.ensure_session("ds", "s01", "ses1")
    # register_run writes run.json — required for Pool.list_runs to find it.
    pool.register_run(RunMeta(
        run_id="r1",
        session_id="ses1",
        subject_id="s01",
        dataset_id="ds",
        task_type="other",
    ))

    shard_mgr = ShardManager(
        pool_root=pool.root,
        dataset_id="ds",
        subject_id="s01",
        session_id="ses1",
        run_id="r1",
    )
    jsonl_path = P.atoms_jsonl_path(pool.root, "ds", "s01", "ses1", "r1")
    atom_ids = []

    with AtomJSONLWriter(jsonl_path) as writer:
        for i in range(n_atoms):
            signal = (
                np.arange(4 * 100, dtype=np.float32).reshape(4, 100) + i
            )
            atom_id = f"a{i:03d}"
            ref = shard_mgr.write_atom_signal(atom_id, signal)
            atom = Atom(
                atom_id=atom_id,
                atom_type=AtomType.TRIAL,
                dataset_id="ds",
                subject_id="s01",
                session_id="ses1",
                run_id="r1",
                signal_ref=ref,
                temporal=TemporalInfo(
                    onset_sample=i * 100,
                    onset_seconds=float(i),
                    duration_samples=100,
                    duration_seconds=1.0,
                ),
                channel_ids=["c1", "c2", "c3", "c4"],
                n_channels=4,
                sampling_rate=100.0,
            )
            writer.write_atom(atom)
            atom_ids.append(atom_id)
    shard_mgr.close()

    indexer = Indexer(pool)
    indexer.reindex_all()
    return pool, indexer, atom_ids


# ──────────────────────────────────────────────────────────────────────────
# byte_offset fast path
# ──────────────────────────────────────────────────────────────────────────


class TestByteOffsetFastPath:
    def test_offsets_populated_by_indexer(self, tmp_path):
        pool, indexer, atom_ids = _build_mini_pool(tmp_path, n_atoms=5)
        rows = indexer.backend.conn.execute(
            "SELECT atom_id, jsonl_byte_offset FROM atoms ORDER BY atom_id"
        ).fetchall()
        # Every atom must have a non-null offset after reindex_all
        offsets = [r["jsonl_byte_offset"] for r in rows]
        assert all(o is not None for o in offsets), offsets
        # Offsets are strictly increasing within a run (line-by-line writes)
        assert offsets == sorted(offsets)
        # First offset is 0 (top of file)
        assert offsets[0] == 0
        indexer.close()

    def test_load_atoms_by_ids_uses_offsets(self, tmp_path):
        from neuroatom.assembler.dataset_assembler import DatasetAssembler

        pool, indexer, atom_ids = _build_mini_pool(tmp_path, n_atoms=10)
        assembler = DatasetAssembler(pool, indexer)

        # Load 3 specific atoms — fast path
        target = [atom_ids[1], atom_ids[5], atom_ids[8]]
        loaded = assembler._load_atoms_by_ids(target)
        assert {a.atom_id for a in loaded} == set(target)
        indexer.close()

    def test_fallback_to_scan_when_offset_null(self, tmp_path, caplog):
        """Force NULL offsets in SQLite and verify the slow path still works."""
        import logging

        from neuroatom.assembler.dataset_assembler import DatasetAssembler

        pool, indexer, atom_ids = _build_mini_pool(tmp_path, n_atoms=5)
        # Simulate a legacy index where offsets were never recorded.
        indexer.backend.conn.execute(
            "UPDATE atoms SET jsonl_byte_offset = NULL"
        )
        indexer.backend.conn.commit()

        assembler = DatasetAssembler(pool, indexer)
        with caplog.at_level(logging.WARNING, logger="neuroatom.assembler.dataset_assembler"):
            loaded = assembler._load_atoms_by_ids([atom_ids[0], atom_ids[3]])
        assert {a.atom_id for a in loaded} == {atom_ids[0], atom_ids[3]}
        # Warning about legacy path should have fired
        assert any("legacy" in r.message.lower() for r in caplog.records)
        indexer.close()

    def test_mixed_offsets_use_both_paths(self, tmp_path):
        """Some atoms with offset, some without — both paths combine."""
        from neuroatom.assembler.dataset_assembler import DatasetAssembler

        pool, indexer, atom_ids = _build_mini_pool(tmp_path, n_atoms=6)
        # Null out half the offsets
        indexer.backend.conn.execute(
            "UPDATE atoms SET jsonl_byte_offset = NULL "
            "WHERE atom_id IN (?, ?, ?)",
            (atom_ids[0], atom_ids[2], atom_ids[4]),
        )
        indexer.backend.conn.commit()

        assembler = DatasetAssembler(pool, indexer)
        loaded = assembler._load_atoms_by_ids(atom_ids)
        assert {a.atom_id for a in loaded} == set(atom_ids)
        indexer.close()


# ──────────────────────────────────────────────────────────────────────────
# Random-access JSONL reader
# ──────────────────────────────────────────────────────────────────────────


class TestAtomJSONLRandomAccess:
    def test_iter_with_offset_round_trip(self, tmp_path):
        """For every (atom, offset) pair, read_at_offset must return the same atom."""
        pool, indexer, atom_ids = _build_mini_pool(tmp_path, n_atoms=4)
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLReader

        jsonl_path = P.atoms_jsonl_path(pool.root, "ds", "s01", "ses1", "r1")
        reader = AtomJSONLReader(jsonl_path)

        pairs = list(reader.iter_atoms_with_offset())
        assert len(pairs) == 4
        for atom, offset in pairs:
            round_trip = reader.read_at_offset(offset)
            assert round_trip is not None
            assert round_trip.atom_id == atom.atom_id
        indexer.close()

    def test_read_at_offset_eof_returns_none(self, tmp_path):
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLReader

        pool, indexer, _ = _build_mini_pool(tmp_path, n_atoms=2)
        jsonl_path = P.atoms_jsonl_path(pool.root, "ds", "s01", "ses1", "r1")
        reader = AtomJSONLReader(jsonl_path)

        # Way past EOF
        result = reader.read_at_offset(10_000_000)
        assert result is None
        indexer.close()


# ──────────────────────────────────────────────────────────────────────────
# dataset_lock serialization
# ──────────────────────────────────────────────────────────────────────────


class TestDatasetLock:
    def test_concurrent_acquisition_serializes(self, tmp_path):
        """Two threads acquiring the same dataset_lock must not overlap."""
        from neuroatom.storage.pool import Pool

        pool = Pool.create(tmp_path / "p")
        pool.ensure_dataset("ds")

        overlap_observed = []

        def hold_lock(idx: int):
            with pool.dataset_lock("ds"):
                # If a peer is already inside, the marker would show
                if overlap_observed:
                    pass  # impossible if serialization works
                overlap_observed.append(f"enter-{idx}")
                time.sleep(0.05)
                overlap_observed.append(f"exit-{idx}")

        with ThreadPoolExecutor(max_workers=2) as pool_ex:
            list(pool_ex.map(hold_lock, [1, 2]))

        # Must see enter/exit interleaved per thread, never enter-1 + enter-2 adjacent
        # at the start (means both inside simultaneously).
        for i in range(len(overlap_observed) - 1):
            cur = overlap_observed[i]
            nxt = overlap_observed[i + 1]
            if cur.startswith("enter-") and nxt.startswith("enter-"):
                pytest.fail(
                    f"Lock did not serialize: {overlap_observed!r}"
                )

    def test_different_datasets_lock_independently(self, tmp_path):
        from neuroatom.storage.pool import Pool

        pool = Pool.create(tmp_path / "p")
        pool.ensure_dataset("ds1")
        pool.ensure_dataset("ds2")

        # Two different dataset_locks should be distinct FileLock instances
        l1 = pool.dataset_lock("ds1")
        l2 = pool.dataset_lock("ds2")
        assert str(l1.lock_file) != str(l2.lock_file)


# ──────────────────────────────────────────────────────────────────────────
# Multi-worker DataLoader
# ──────────────────────────────────────────────────────────────────────────


class TestMultiWorkerDataLoader:
    """Validates that worker_init_fn does the right thing.

    Note: On Windows, DataLoader workers use 'spawn' (not 'fork'), so
    inherited HDF5 handles aren't actually a risk on Windows. Test verifies
    correctness on whatever start method the platform uses.
    """

    def test_num_workers_2_yields_all_samples(self, tmp_path):
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        from neuroatom.loader.torch_dataset import (
            AtomDataset,
            worker_init_fn,
        )

        # Build synthetic in-memory samples (don't need HDF5 here — AtomDataset
        # is in-memory; we're testing worker mechanics, not lazy HDF5 reads).
        samples = [
            {
                "signal": np.ones((2, 16), dtype=np.float32) * i,
                "labels": {"label": i % 3},
                "atom_id": f"a{i}",
                "subject_id": "s",
                "dataset_id": "d",
            }
            for i in range(16)
        ]
        ds = AtomDataset(samples)

        # num_workers=0 on Windows-without-fork to keep CI deterministic;
        # bump to 2 only if available.
        num_workers = 2 if sys.platform != "win32" else 0
        loader = DataLoader(
            ds,
            batch_size=4,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )
        seen_atoms = []
        for batch in loader:
            seen_atoms.extend(batch["atom_id"])
        assert sorted(seen_atoms) == sorted(s["atom_id"] for s in samples)

    def test_hdf5_dataset_multi_worker(self, tmp_path):
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        from neuroatom.loader.torch_dataset import (
            HDF5AtomDataset,
            worker_init_fn,
        )

        pool, indexer, atom_ids = _build_mini_pool(tmp_path, n_atoms=8)
        # Build atom info dicts that HDF5AtomDataset expects
        atoms_info = []
        for a in indexer.backend.conn.execute(
            "SELECT atom_id, signal_file_path FROM atoms"
        ).fetchall():
            atoms_info.append({
                "atom_id": a["atom_id"],
                "signal_file_path": a["signal_file_path"],
                "signal_internal_path": f"/atoms/{a['atom_id']}/signal",
                "labels": {},
                "subject_id": "s01",
                "dataset_id": "ds",
            })

        ds = HDF5AtomDataset(
            atoms=atoms_info,
            pool_root=pool.root,
            error_handling="skip",
        )
        num_workers = 2 if sys.platform != "win32" else 0
        loader = DataLoader(
            ds,
            batch_size=2,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=ds.safe_collate_fn(),
        )
        n_samples = 0
        for batch in loader:
            if batch is None:
                continue
            n_samples += len(batch["atom_id"])
        assert n_samples == 8
        indexer.close()


# ──────────────────────────────────────────────────────────────────────────
# Migration 0.1.0 → 0.2.0
# ──────────────────────────────────────────────────────────────────────────


class TestMigration_0_1_to_0_2:
    def test_migration_registered(self):
        from neuroatom.storage.migration import list_available_migrations

        assert ("0.1.0", "0.2.0") in list_available_migrations()

    def test_migration_repopulates_offsets(self, tmp_path):
        """Simulate an old pool (offsets NULL), run migration, verify offsets fill."""
        from neuroatom.storage.migration import migrate, set_pool_version

        pool, indexer, atom_ids = _build_mini_pool(tmp_path, n_atoms=4)
        pool_root = pool.root
        # Null out all offsets and pretend pool is at 0.1.0
        indexer.backend.conn.execute(
            "UPDATE atoms SET jsonl_byte_offset = NULL"
        )
        indexer.backend.conn.commit()
        indexer.close()
        set_pool_version(pool_root, "0.1.0")

        applied = migrate(pool_root, target_version="0.2.0")
        assert any("0.1.0" in step for step in applied)

        # Re-open and check offsets are now populated
        from neuroatom.index.indexer import Indexer
        from neuroatom.storage.pool import Pool

        pool2 = Pool.open(pool_root)
        indexer2 = Indexer(pool2)
        rows = indexer2.backend.conn.execute(
            "SELECT jsonl_byte_offset FROM atoms"
        ).fetchall()
        assert all(r["jsonl_byte_offset"] is not None for r in rows)
        indexer2.close()
