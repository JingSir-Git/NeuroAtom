"""End-to-end tests for ChineseEEG-2 importer.

Tests import of PassiveListening BIDS data with sentence-level atomization.
Requires: C:\\Data\\ChineseEEG-2\\PassiveListening
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuroatom.importers.chinese_eeg2 import (
    ChineseEEG2Importer,
    _extract_sentence_epochs,
    _load_audio_embedding_chapter,
    _load_text_embeddings_novel,
    _parse_run_id,
    _read_electrodes_tsv,
    _read_tsv,
)
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.index.indexer import Indexer

PL_ROOT = Path(r"D:\Data\ChineseEEG-2\PassiveListening")
RA_ROOT = Path(r"D:\Data\ChineseEEG-2\ReadingAloud")
MATERIALS_ROOT = Path(r"D:\Data\ChineseEEG-2\materials&embeddings")

SKIP_MSG = "ChineseEEG-2 data not found at expected path"
HAS_PL = PL_ROOT.exists() and (PL_ROOT / "dataset_description.json").exists()
HAS_RA = RA_ROOT.exists() and (RA_ROOT / "dataset_description.json").exists()
HAS_MATERIALS = MATERIALS_ROOT.exists()
skip_no_pl = pytest.mark.skipif(not HAS_PL, reason=SKIP_MSG)
skip_no_materials = pytest.mark.skipif(
    not (HAS_PL and HAS_MATERIALS), reason="ChineseEEG-2 data/materials not found"
)


class TestRunIdParsing:
    """Unit tests for run ID parsing (no data needed)."""

    def test_run_11(self):
        rep, ch = _parse_run_id("run-11")
        assert rep == 1 and ch == 1

    def test_run_213(self):
        rep, ch = _parse_run_id("run-213")
        assert rep == 2 and ch == 13

    def test_run_15(self):
        rep, ch = _parse_run_id("run-15")
        assert rep == 1 and ch == 5

    def test_fallback(self):
        rep, ch = _parse_run_id("run-x")
        assert rep == 1 and ch == 0


@pytest.mark.skipif(not HAS_PL, reason=SKIP_MSG)
class TestDetection:
    """Test dataset detection."""

    def test_detect_passive_listening(self):
        assert ChineseEEG2Importer.detect(PL_ROOT) is True

    @pytest.mark.skipif(not HAS_RA, reason="ReadingAloud not found")
    def test_detect_reading_aloud(self):
        assert ChineseEEG2Importer.detect(RA_ROOT) is True

    def test_detect_random_dir(self, tmp_path):
        assert ChineseEEG2Importer.detect(tmp_path) is False


@pytest.mark.skipif(not HAS_PL, reason=SKIP_MSG)
class TestSentenceEpochExtraction:
    """Test ROWS→ROWE sentence epoch extraction."""

    def test_extract_epochs_run11(self):
        events_path = (
            PL_ROOT / "sub-01" / "ses-littleprince" / "eeg"
            / "sub-01_ses-littleprince_task-lis_run-11_events.tsv"
        )
        epochs = _extract_sentence_epochs(events_path, 1000.0)
        assert len(epochs) > 0, "Should find sentence epochs"

        for ep in epochs:
            assert ep["duration_samples"] > 0
            assert ep["offset_sample"] > ep["onset_sample"]
            assert ep["onset_sec"] >= 0

    def test_epoch_ordering(self):
        events_path = (
            PL_ROOT / "sub-01" / "ses-littleprince" / "eeg"
            / "sub-01_ses-littleprince_task-lis_run-11_events.tsv"
        )
        epochs = _extract_sentence_epochs(events_path, 1000.0)
        onsets = [e["onset_sample"] for e in epochs]
        assert onsets == sorted(onsets), "Epochs should be in temporal order"

    def test_sentence_indices_sequential(self):
        events_path = (
            PL_ROOT / "sub-01" / "ses-littleprince" / "eeg"
            / "sub-01_ses-littleprince_task-lis_run-11_events.tsv"
        )
        epochs = _extract_sentence_epochs(events_path, 1000.0)
        indices = [e["sentence_index"] for e in epochs]
        assert indices == list(range(len(epochs)))


@pytest.mark.skipif(not HAS_PL, reason=SKIP_MSG)
class TestElectrodes:
    """Test electrode coordinate extraction."""

    def test_electrode_count(self):
        elec_path = (
            PL_ROOT / "sub-01" / "ses-littleprince" / "eeg"
            / "sub-01_ses-littleprince_space-CapTrak_electrodes.tsv"
        )
        elecs = _read_electrodes_tsv(elec_path)
        assert len(elecs) == 128, f"Expected 128 electrodes, got {len(elecs)}"

    def test_electrode_has_coordinates(self):
        elec_path = (
            PL_ROOT / "sub-01" / "ses-littleprince" / "eeg"
            / "sub-01_ses-littleprince_space-CapTrak_electrodes.tsv"
        )
        elecs = _read_electrodes_tsv(elec_path)
        e1 = elecs["E1"]
        assert e1.coordinate_system == "CapTrak"
        assert e1.x != 0 or e1.y != 0 or e1.z != 0


@pytest.mark.skipif(not HAS_PL, reason=SKIP_MSG)
class TestDiscovery:
    """Test BIDS recording discovery."""

    def test_discover_all_subjects(self):
        pool = Pool.create(tempfile.mkdtemp())
        imp = ChineseEEG2Importer(pool, task="listening")
        recs = imp.discover_recordings(PL_ROOT)
        assert len(recs) > 100, f"Expected >100 recordings, got {len(recs)}"

        subjects = sorted(set(r["subject"] for r in recs))
        assert len(subjects) >= 8, f"Expected >=8 subjects, got {len(subjects)}"

    def test_all_have_events(self):
        pool = Pool.create(tempfile.mkdtemp())
        imp = ChineseEEG2Importer(pool, task="listening")
        recs = imp.discover_recordings(PL_ROOT)
        with_events = sum(1 for r in recs if r["events_tsv"] is not None)
        assert with_events == len(recs), "All recordings should have events.tsv"

    def test_session_names(self):
        pool = Pool.create(tempfile.mkdtemp())
        imp = ChineseEEG2Importer(pool, task="listening")
        recs = imp.discover_recordings(PL_ROOT)
        sessions = sorted(set(r.get("session") for r in recs))
        assert "littleprince" in sessions
        assert "garnettdream" in sessions


@pytest.mark.skipif(not HAS_PL, reason=SKIP_MSG)
class TestImportAndIndex:
    """Full import + index + query cycle."""

    @pytest.fixture(scope="class")
    def imported(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEG2Importer(pool, task="listening")
        results = imp.import_dataset(
            PL_ROOT,
            subjects=["01"],
            sessions=["littleprince"],
            max_runs=1,
        )
        assert len(results) == 1
        return pool, results[0]

    def test_atom_count(self, imported):
        _, result = imported
        assert result.n_atoms > 50, f"Expected >50 atoms, got {result.n_atoms}"

    def test_atom_metadata(self, imported):
        _, result = imported
        a0 = result.atoms[0]
        assert a0.n_channels == 128
        assert a0.sampling_rate == 1000.0
        assert a0.dataset_id == "chinese_eeg2_listening"
        assert a0.subject_id == "sub-01"
        assert "ses-littleprince" == a0.session_id

    def test_temporal_info(self, imported):
        _, result = imported
        a0 = result.atoms[0]
        assert a0.temporal.onset_sample >= 0
        assert a0.temporal.duration_samples > 0
        assert a0.temporal.duration_seconds > 0

    def test_annotations(self, imported):
        _, result = imported
        a0 = result.atoms[0]
        ann_names = {a.name for a in a0.annotations}
        assert "novel" in ann_names
        assert "chapter" in ann_names
        assert "repetition" in ann_names
        assert "sentence_index" in ann_names

    def test_signal_readback(self, imported):
        pool, result = imported
        a0 = result.atoms[0]
        mgr = ShardManager(
            pool_root=pool.root,
            dataset_id=a0.dataset_id,
            subject_id=a0.subject_id,
            session_id=a0.session_id,
            run_id=a0.run_id,
        )
        sig = mgr.read_atom_signal(a0.signal_ref)
        mgr.close()

        assert sig.shape[0] == 128
        assert sig.shape[1] == a0.temporal.duration_samples
        assert np.isfinite(sig).all()

    def test_index_and_query(self, imported):
        pool, result = imported
        indexer = Indexer(pool)
        indexer.reindex_all()

        from neuroatom.index.query import QueryBuilder
        qb = QueryBuilder(indexer.backend)
        atom_ids = qb.query_atom_ids({"dataset_id": "chinese_eeg2_listening"})
        assert len(atom_ids) == result.n_atoms

    def test_processing_history(self, imported):
        _, result = imported
        a0 = result.atoms[0]
        assert a0.processing_history.is_raw is True
        assert a0.processing_history.version_tag == "raw"
        assert len(a0.processing_history.steps) == 1
        assert a0.processing_history.steps[0].operation == "raw_import"

    def test_custom_fields(self, imported):
        _, result = imported
        a0 = result.atoms[0]
        assert a0.custom_fields["paradigm"] == "passive_listening"
        assert a0.custom_fields["novel"] == "littleprince"
        assert isinstance(a0.custom_fields["chapter"], int)


@pytest.mark.skipif(not HAS_PL, reason=SKIP_MSG)
class TestSignalCharacteristics:
    """Verify signal data quality."""

    @pytest.fixture(scope="class")
    def signal_data(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEG2Importer(pool, task="listening")
        results = imp.import_dataset(
            PL_ROOT,
            subjects=["01"],
            sessions=["littleprince"],
            max_runs=1,
        )
        a0 = results[0].atoms[0]
        mgr = ShardManager(
            pool_root=pool.root,
            dataset_id=a0.dataset_id,
            subject_id=a0.subject_id,
            session_id=a0.session_id,
            run_id=a0.run_id,
        )
        sig = mgr.read_atom_signal(a0.signal_ref)
        mgr.close()
        return sig

    def test_units_microvolts(self, signal_data):
        """Signals are stored in µV — typical EEG range."""
        sig = signal_data
        max_abs = np.abs(sig).max()
        assert max_abs < 1e6, f"Signal in µV should have max < 1e6, got {max_abs}"
        assert max_abs > 0.01, f"Signal seems too small for µV, got {max_abs}"

    def test_no_nans(self, signal_data):
        assert np.isfinite(signal_data).all()

    def test_not_all_zero(self, signal_data):
        assert not np.allclose(signal_data, 0)


# ======================================================================
# P4: Text and audio embedding tests
# ======================================================================


@pytest.mark.skipif(not HAS_MATERIALS, reason="ChineseEEG-2 materials not found")
class TestEmbeddingHelpers:
    """Unit tests for embedding loader functions (no EEG import needed)."""

    def test_load_text_embeddings_littleprince(self):
        emb = _load_text_embeddings_novel(MATERIALS_ROOT, "littleprince")
        assert emb is not None
        assert emb.ndim == 2
        assert emb.shape[1] == 768
        assert emb.dtype == np.float32

    def test_load_text_embeddings_garnettdream(self):
        emb = _load_text_embeddings_novel(MATERIALS_ROOT, "garnettdream")
        assert emb is not None
        assert emb.shape[1] == 768

    def test_load_text_embeddings_missing_novel(self):
        emb = _load_text_embeddings_novel(MATERIALS_ROOT, "nonexistent_novel")
        assert emb is None

    def test_load_audio_embedding_chapter0(self):
        emb = _load_audio_embedding_chapter(MATERIALS_ROOT, "littleprince", 0)
        assert emb is not None
        assert emb.ndim == 1
        assert emb.shape[0] == 1024
        assert emb.dtype == np.float32

    def test_load_audio_embedding_out_of_range(self):
        emb = _load_audio_embedding_chapter(MATERIALS_ROOT, "littleprince", 9999)
        assert emb is None

    def test_load_audio_embedding_unknown_novel(self):
        emb = _load_audio_embedding_chapter(MATERIALS_ROOT, "unknown_novel", 0)
        assert emb is None


@pytest.mark.skipif(
    not (HAS_PL and HAS_MATERIALS),
    reason="ChineseEEG-2 PassiveListening + materials not found",
)
class TestEmbeddingAnnotations:
    """Verify embedding ContinuousAnnotations are present after import."""

    @pytest.fixture(scope="class")
    def imported(self, tmp_path_factory):
        td = tmp_path_factory.mktemp("ceeg2_emb")
        pool = Pool.create(td / "pool")
        imp = ChineseEEG2Importer(
            pool, task="listening",
            materials_root=MATERIALS_ROOT,
        )
        results = imp.import_dataset(
            PL_ROOT, subjects=["01"], sessions=["littleprince"], max_runs=1,
        )
        return pool, results

    def test_text_embedding_annotation_present(self, imported):
        pool, results = imported
        assert results[0].n_atoms > 0
        atom = results[0].atoms[0]
        ann_names = [a.name for a in atom.annotations]
        assert "text_embedding" in ann_names, f"Got: {ann_names}"

    def test_audio_embedding_annotation_present(self, imported):
        pool, results = imported
        atom = results[0].atoms[0]
        ann_names = [a.name for a in atom.annotations]
        assert "audio_embedding" in ann_names, f"Got: {ann_names}"

    def test_text_embedding_shape(self, imported):
        pool, results = imported
        from neuroatom.core.annotation import ContinuousAnnotation
        atom = results[0].atoms[0]
        ann = next(
            (a for a in atom.annotations
             if isinstance(a, ContinuousAnnotation) and a.name == "text_embedding"),
            None,
        )
        assert ann is not None
        assert ann.data_ref.shape == (768,)

    def test_audio_embedding_shape(self, imported):
        pool, results = imported
        from neuroatom.core.annotation import ContinuousAnnotation
        atom = results[0].atoms[0]
        ann = next(
            (a for a in atom.annotations
             if isinstance(a, ContinuousAnnotation) and a.name == "audio_embedding"),
            None,
        )
        assert ann is not None
        assert ann.data_ref.shape == (1024,)

    def test_text_embedding_in_hdf5(self, imported):
        pool, results = imported
        import h5py
        atom = results[0].atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            path = f"/atoms/{atom.atom_id}/annotations/text_embedding"
            assert path in f, f"text_embedding missing in HDF5"
            arr = f[path][:]
        assert arr.shape == (768,)

    def test_audio_embedding_in_hdf5(self, imported):
        pool, results = imported
        import h5py
        atom = results[0].atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            path = f"/atoms/{atom.atom_id}/annotations/audio_embedding"
            assert path in f, f"audio_embedding missing in HDF5"
            arr = f[path][:]
        assert arr.shape == (1024,)

    def test_global_sentence_index_in_custom_fields(self, imported):
        pool, results = imported
        for atom in results[0].atoms:
            assert "global_sentence_index" in atom.custom_fields
