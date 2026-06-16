"""Tests for pool export/import (.napool archive format).

Tests cover:
- Full export → import round-trip
- Manifest integrity verification
- Selective dataset export
- Merge into existing pool
- Atom data survives round-trip
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import AtomType
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.subject import SubjectMeta
from neuroatom.storage.metadata_store import AtomJSONLWriter, AtomJSONLReader
from neuroatom.storage.pool import Pool
from neuroatom.storage.pool_archive import (
    export_pool,
    import_pool,
    _sha256_file,
    _verify_manifest,
    NAPOOL_EXTENSION,
)
from neuroatom.storage.signal_store import ShardManager
from neuroatom.storage import paths as P


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pool(tmp_path):
    """Create a minimal pool with one dataset, one subject, one atom."""
    pool = Pool.create(tmp_path / "source_pool")

    dataset_id = "test_ds"
    subject_id = "S01"
    session_id = "ses-01"
    run_id = "run-01"

    # Register hierarchy
    pool.register_dataset(DatasetMeta(
        dataset_id=dataset_id,
        name="Test Dataset",
        task_types=["test_task"],
        import_timestamp="2025-06-01T00:00:00+00:00",
    ))
    pool.register_subject(SubjectMeta(
        subject_id=subject_id, dataset_id=dataset_id,
    ))
    pool.ensure_session(dataset_id, subject_id, session_id, sampling_rate=128.0)
    pool.ensure_run(dataset_id, subject_id, session_id, run_id)

    # Create a signal + atom
    rng = np.random.default_rng(42)
    signal = rng.standard_normal((4, 256)).astype(np.float32) * 50.0  # µV range

    atom = Atom(
        atom_id="atom_001",
        dataset_id=dataset_id,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        atom_type=AtomType.TRIAL,
        signal_ref=SignalRef(
            file_path="", internal_path="", shape=(4, 256),
        ),
        temporal=TemporalInfo(
            onset_sample=0, duration_samples=256,
            onset_seconds=0.0, duration_seconds=2.0,
        ),
        channel_ids=["Fp1", "Fp2", "C3", "C4"],
        n_channels=4,
        sampling_rate=128.0,
        signal_unit="uV",
        original_unit="V",
    )

    # Write signal to HDF5 shard
    with ShardManager(
        pool_root=pool.root,
        dataset_id=dataset_id,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        max_shard_size_mb=10.0,
    ) as mgr:
        ref = mgr.write_atom_signal(atom.atom_id, signal)
        atom.signal_ref = ref

    # Write atom JSONL
    jsonl_path = P.atoms_jsonl_path(
        pool.root, dataset_id, subject_id, session_id, run_id,
    )
    with AtomJSONLWriter(jsonl_path) as w:
        w.write_atom(atom)

    return pool, atom, signal


# ── Tests ─────────────────────────────────────────────────────────────────

class TestPoolArchive:

    def test_full_export_import_roundtrip(self, sample_pool, tmp_path):
        """Export → import → verify atoms and signals match."""
        pool, orig_atom, orig_signal = sample_pool

        archive_path = tmp_path / "test.napool"
        target_root = tmp_path / "imported_pool"

        # Export
        manifest = export_pool(pool.root, archive_path)
        assert archive_path.exists()
        assert manifest["n_files"] > 0
        assert "test_ds" in manifest["datasets"]

        # Import
        imported_pool = Pool.import_from(archive_path, target_root)
        assert imported_pool.root == target_root.resolve()

        # Verify dataset exists
        datasets = imported_pool.list_datasets()
        assert "test_ds" in datasets

        # Verify atom data
        reader = AtomJSONLReader(P.atoms_jsonl_path(
            imported_pool.root, "test_ds", "S01", "ses-01", "run-01",
        ))
        atoms = list(reader.iter_atoms())
        assert len(atoms) == 1
        atom = atoms[0]
        assert atom.atom_id == "atom_001"
        assert atom.signal_unit == "uV"
        assert atom.original_unit == "V"

        # Verify signal data
        sig = ShardManager.static_read(imported_pool.root, atom.signal_ref)
        assert sig.shape == orig_signal.shape
        np.testing.assert_allclose(sig, orig_signal, rtol=1e-6)

    def test_manifest_integrity(self, sample_pool, tmp_path):
        """Manifest SHA-256 checksums are correct."""
        pool, _, _ = sample_pool

        archive_path = tmp_path / "integrity.napool"
        manifest = export_pool(pool.root, archive_path)

        # All file entries have sha256
        for entry in manifest["files"]:
            assert "sha256" in entry
            assert len(entry["sha256"]) == 64  # hex SHA-256

        # Verify against source
        errors = _verify_manifest(pool.root, manifest)
        assert errors == []

    def test_selective_dataset_export(self, tmp_path):
        """Export only specific datasets."""
        pool = Pool.create(tmp_path / "multi_pool")

        for ds_id in ["ds_a", "ds_b", "ds_c"]:
            pool.register_dataset(DatasetMeta(
                dataset_id=ds_id, name=ds_id,
            ))

        archive_path = tmp_path / "selective.napool"
        manifest = export_pool(
            pool.root, archive_path, dataset_ids=["ds_a", "ds_c"],
        )

        assert "ds_a" in manifest["datasets"]
        assert "ds_c" in manifest["datasets"]
        assert "ds_b" not in manifest["datasets"]

    def test_merge_into_existing_pool(self, sample_pool, tmp_path):
        """Import merges new dataset into existing pool."""
        pool, _, _ = sample_pool

        # Create another pool
        pool2 = Pool.create(tmp_path / "pool2")
        pool2.register_dataset(DatasetMeta(
            dataset_id="existing_ds", name="Existing",
        ))

        # Export first pool
        archive_path = tmp_path / "merge.napool"
        export_pool(pool.root, archive_path)

        # Import into second pool (merge)
        merged = Pool.import_from(archive_path, pool2.root, merge=True)
        datasets = merged.list_datasets()
        assert "existing_ds" in datasets
        assert "test_ds" in datasets

    def test_import_no_merge_fails_on_existing(self, sample_pool, tmp_path):
        """Import with merge=False raises on existing pool."""
        pool, _, _ = sample_pool
        archive_path = tmp_path / "nomerge.napool"
        export_pool(pool.root, archive_path)

        pool2 = Pool.create(tmp_path / "pool2")

        with pytest.raises(ValueError, match="merge=False"):
            import_pool(archive_path, pool2.root, merge=False)

    def test_sha256_file(self, tmp_path):
        """SHA-256 helper produces correct hex digest."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        sha = _sha256_file(test_file)
        assert len(sha) == 64
        assert sha == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_export_creates_napool_extension(self, sample_pool, tmp_path):
        """Output path auto-gets .napool extension."""
        pool, _, _ = sample_pool
        archive_path = tmp_path / "noext"
        export_pool(pool.root, archive_path)
        assert (tmp_path / "noext.napool").exists()

    def test_subject_level_export(self, tmp_path):
        """Export only specific subjects from a dataset."""
        pool = Pool.create(tmp_path / "subj_pool")
        ds_id = "test_ds"
        pool.register_dataset(DatasetMeta(dataset_id=ds_id, name="Test"))

        # Create 3 subjects with minimal data
        for sid in ["S01", "S02", "S03"]:
            pool.register_subject(SubjectMeta(
                subject_id=sid, dataset_id=ds_id,
            ))
            pool.ensure_session(ds_id, sid, "ses-01", sampling_rate=128.0)
            pool.ensure_run(ds_id, sid, "ses-01", "run-01")

            rng = np.random.default_rng(42)
            signal = rng.standard_normal((4, 128)).astype(np.float32)
            atom = Atom(
                atom_id=f"atom_{sid}",
                dataset_id=ds_id, subject_id=sid,
                session_id="ses-01", run_id="run-01",
                atom_type=AtomType.TRIAL,
                signal_ref=SignalRef(
                    file_path="", internal_path="", shape=(4, 128),
                ),
                temporal=TemporalInfo(
                    onset_sample=0, duration_samples=128,
                    onset_seconds=0.0, duration_seconds=1.0,
                ),
                channel_ids=["Fp1", "Fp2", "C3", "C4"],
                n_channels=4, sampling_rate=128.0,
                signal_unit="uV",
            )
            with ShardManager(
                pool_root=pool.root, dataset_id=ds_id,
                subject_id=sid, session_id="ses-01", run_id="run-01",
                max_shard_size_mb=10.0,
            ) as mgr:
                ref = mgr.write_atom_signal(atom.atom_id, signal)
                atom.signal_ref = ref
            jsonl = P.atoms_jsonl_path(pool.root, ds_id, sid, "ses-01", "run-01")
            with AtomJSONLWriter(jsonl) as w:
                w.write_atom(atom)

        # Export only S01 and S03
        archive = tmp_path / "subj.napool"
        manifest = export_pool(
            pool.root, archive, subject_ids=["S01", "S03"],
        )

        # Import and verify only S01, S03 present
        target = tmp_path / "subj_imported"
        imported = Pool.import_from(archive, target)
        subjects = imported.list_subjects(ds_id)
        assert "S01" in subjects
        assert "S03" in subjects
        assert "S02" not in subjects

    def test_qualified_subject_export(self, tmp_path):
        """Export subject with qualified 'dataset_id/subject_id' format."""
        pool = Pool.create(tmp_path / "qual_pool")
        for ds_id in ["ds_a", "ds_b"]:
            pool.register_dataset(DatasetMeta(dataset_id=ds_id, name=ds_id))
            for sid in ["S01", "S02"]:
                pool.register_subject(SubjectMeta(
                    subject_id=sid, dataset_id=ds_id,
                ))

        archive = tmp_path / "qual.napool"
        manifest = export_pool(
            pool.root, archive, subject_ids=["ds_a/S01", "ds_b/S02"],
        )

        target = tmp_path / "qual_imported"
        imported = Pool.import_from(archive, target)
        assert "S01" in imported.list_subjects("ds_a")
        assert "S02" not in imported.list_subjects("ds_a")
        assert "S02" in imported.list_subjects("ds_b")
        assert "S01" not in imported.list_subjects("ds_b")

    def test_pool_convenience_methods(self, sample_pool, tmp_path):
        """Pool.export() and Pool.import_from() work."""
        pool, _, orig_signal = sample_pool

        archive_path = tmp_path / "via_pool.napool"
        manifest = pool.export(archive_path, description="Test export")
        assert manifest["description"] == "Test export"

        target = tmp_path / "via_import"
        imported = Pool.import_from(archive_path, target)
        assert imported.list_datasets() == ["test_ds"]
