"""End-to-end tests for ChineseEEG importer.

Tests import of BIDS reading-task data with sentence-level atomization.
Requires the dataset at the path specified by NEUROATOM_CHINESE_EEG_DIR
or the default network location.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuroatom.importers.chinese_eeg import (
    ChineseEEGImporter,
    _detect_chapter,
    _extract_sentence_epochs,
    _load_text_embeddings_run,
    _read_electrodes_tsv,
    _read_tsv,
)
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.index.indexer import Indexer

# ---------------------------------------------------------------------------
# Dataset root — configurable via env var
# ---------------------------------------------------------------------------
_DEFAULT_ROOT = (
    r"\\wsqlab\ugreen\Language"
    r"\25+ChineseEEG A Chinese Linguistic Corpora EEG Dataset"
    r" for Semantic Alignment and Neural Decoding"
)
BIDS_ROOT = Path(os.environ.get("NEUROATOM_CHINESE_EEG_DIR", _DEFAULT_ROOT))

SKIP_MSG = "ChineseEEG data not found"
HAS_DATA = BIDS_ROOT.exists() and (BIDS_ROOT / "dataset_description.json").exists()

# A known subject that has both sessions
TEST_SUBJECT = "04"
TEST_SESSION = "LittlePrince"

# Paths used by unit-style tests (no full import needed)
_EEG_DIR = (
    BIDS_ROOT / f"sub-{TEST_SUBJECT}" / f"ses-{TEST_SESSION}" / "eeg"
)
_EVENTS_PATH = (
    _EEG_DIR
    / f"sub-{TEST_SUBJECT}_ses-{TEST_SESSION}_task-reading_run-01_events.tsv"
)
_CHANNELS_PATH = (
    _EEG_DIR
    / f"sub-{TEST_SUBJECT}_ses-{TEST_SESSION}_task-reading_run-01_channels.tsv"
)
_ELECTRODES_PATH = (
    _EEG_DIR
    / f"sub-{TEST_SUBJECT}_ses-{TEST_SESSION}_space-CapTrak_electrodes.tsv"
)


# ======================================================================
# Unit-level tests (no heavy import)
# ======================================================================


@pytest.mark.skipif(not HAS_DATA, reason=SKIP_MSG)
class TestDetection:
    """Test dataset auto-detection."""

    def test_detect_positive(self):
        assert ChineseEEGImporter.detect(BIDS_ROOT) is True

    def test_detect_random_dir(self, tmp_path):
        assert ChineseEEGImporter.detect(tmp_path) is False


@pytest.mark.skipif(not _EVENTS_PATH.exists(), reason=SKIP_MSG)
class TestSentenceEpochExtraction:
    """Test ROWS→ROWE sentence epoch extraction from events.tsv."""

    def test_extract_epochs_count(self):
        epochs = _extract_sentence_epochs(_EVENTS_PATH, 1000.0)
        assert len(epochs) > 100, f"Expected >100 sentence epochs, got {len(epochs)}"

    def test_epoch_fields(self):
        epochs = _extract_sentence_epochs(_EVENTS_PATH, 1000.0)
        ep = epochs[0]
        assert ep["onset_sample"] >= 0
        assert ep["offset_sample"] > ep["onset_sample"]
        assert ep["duration_samples"] > 0
        assert ep["onset_sec"] >= 0
        assert ep["sentence_index"] == 0

    def test_epoch_ordering(self):
        epochs = _extract_sentence_epochs(_EVENTS_PATH, 1000.0)
        onsets = [e["onset_sample"] for e in epochs]
        assert onsets == sorted(onsets), "Epochs should be temporally ordered"

    def test_sentence_indices_sequential(self):
        epochs = _extract_sentence_epochs(_EVENTS_PATH, 1000.0)
        indices = [e["sentence_index"] for e in epochs]
        assert indices == list(range(len(epochs)))


@pytest.mark.skipif(not _EVENTS_PATH.exists(), reason=SKIP_MSG)
class TestChapterDetection:
    """Test chapter marker extraction."""

    def test_chapter_from_run01(self):
        chapter = _detect_chapter(_EVENTS_PATH)
        # run-01 of LittlePrince should have CH01
        assert chapter == 1, f"Expected chapter 1, got {chapter}"

    def test_chapter_from_garnett_run10(self):
        gd_events = (
            BIDS_ROOT / f"sub-{TEST_SUBJECT}" / "ses-GarnettDream" / "eeg"
            / f"sub-{TEST_SUBJECT}_ses-GarnettDream_task-reading_run-10_events.tsv"
        )
        if not gd_events.exists():
            pytest.skip("GarnettDream run-10 events not found")
        chapter = _detect_chapter(gd_events)
        assert chapter == 10, f"Expected chapter 10, got {chapter}"


@pytest.mark.skipif(not _ELECTRODES_PATH.exists(), reason=SKIP_MSG)
class TestElectrodes:
    """Test electrode coordinate extraction."""

    def test_electrode_count(self):
        elecs = _read_electrodes_tsv(_ELECTRODES_PATH)
        assert len(elecs) == 128, f"Expected 128 electrodes, got {len(elecs)}"

    def test_electrode_coordinates(self):
        elecs = _read_electrodes_tsv(_ELECTRODES_PATH)
        e1 = elecs["E1"]
        assert e1.coordinate_system == "CapTrak"
        assert not (e1.x == 0 and e1.y == 0 and e1.z == 0)


@pytest.mark.skipif(not _CHANNELS_PATH.exists(), reason=SKIP_MSG)
class TestChannelsTSV:
    """Test channels.tsv parsing."""

    def test_channel_count(self):
        rows = _read_tsv(_CHANNELS_PATH)
        assert len(rows) == 128

    def test_channel_names(self):
        rows = _read_tsv(_CHANNELS_PATH)
        names = [r["name"] for r in rows]
        assert names[0] == "E1"
        assert names[-1] == "E128"

    def test_all_eeg_type(self):
        rows = _read_tsv(_CHANNELS_PATH)
        types = set(r["type"] for r in rows)
        assert types == {"EEG"}, f"Expected only EEG, got {types}"


# ======================================================================
# Discovery tests
# ======================================================================


@pytest.mark.skipif(not HAS_DATA, reason=SKIP_MSG)
class TestDiscovery:
    """Test BIDS recording discovery."""

    def test_discover_all(self):
        pool = Pool.create(tempfile.mkdtemp())
        imp = ChineseEEGImporter(pool)
        recs = imp.discover_recordings(BIDS_ROOT)
        # 10 subjects × (7 LP + 18 GD) = 250 max, minus missing ~4
        assert len(recs) > 200, f"Expected >200 recordings, got {len(recs)}"

    def test_discover_sessions(self):
        pool = Pool.create(tempfile.mkdtemp())
        imp = ChineseEEGImporter(pool)
        recs = imp.discover_recordings(BIDS_ROOT)
        sessions = sorted(set(r.get("session") for r in recs))
        assert "LittlePrince" in sessions
        assert "GarnettDream" in sessions

    def test_discover_subjects(self):
        pool = Pool.create(tempfile.mkdtemp())
        imp = ChineseEEGImporter(pool)
        recs = imp.discover_recordings(BIDS_ROOT)
        subjects = sorted(set(r["subject"] for r in recs))
        assert len(subjects) == 10, f"Expected 10 subjects, got {subjects}"

    def test_all_have_events(self):
        pool = Pool.create(tempfile.mkdtemp())
        imp = ChineseEEGImporter(pool)
        recs = imp.discover_recordings(BIDS_ROOT)
        with_events = sum(1 for r in recs if r["events_tsv"] is not None)
        assert with_events == len(recs), "All recordings should have events.tsv"


# ======================================================================
# Full import + index tests (single subject, single session, 1 run)
# ======================================================================


@pytest.mark.skipif(not HAS_DATA, reason=SKIP_MSG)
class TestImportSingleRun:
    """Full import + index + query cycle on a single run."""

    @pytest.fixture(scope="class")
    def imported(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEGImporter(pool)
        results = imp.import_dataset(
            BIDS_ROOT,
            subjects=[TEST_SUBJECT],
            sessions=[TEST_SESSION],
            max_runs=1,
        )
        assert len(results) == 1
        return pool, results[0]

    def test_atom_count(self, imported):
        _, result = imported
        assert result.n_atoms > 100, (
            f"Expected >100 sentence atoms, got {result.n_atoms}"
        )

    def test_atom_metadata(self, imported):
        _, result = imported
        a0 = result.atoms[0]
        assert a0.n_channels == 128
        assert a0.sampling_rate == 1000.0
        assert a0.dataset_id == "chinese_eeg_reading"
        assert a0.subject_id == f"sub-{TEST_SUBJECT}"
        assert a0.session_id == f"ses-{TEST_SESSION}"

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
        assert "sentence_index" in ann_names
        assert "chapter" in ann_names

    def test_novel_annotation_value(self, imported):
        _, result = imported
        a0 = result.atoms[0]
        novel_ann = next(a for a in a0.annotations if a.name == "novel")
        assert novel_ann.value == TEST_SESSION

    def test_signal_readback(self, imported):
        pool, result = imported
        a0 = result.atoms[0]
        sig = ShardManager.static_read(pool.root, a0.signal_ref)
        assert sig.shape[0] == 128
        assert sig.shape[1] == a0.temporal.duration_samples
        assert np.isfinite(sig).all()

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
        assert a0.custom_fields["paradigm"] == "visual_reading"
        assert a0.custom_fields["novel"] == TEST_SESSION
        assert a0.custom_fields["chapter"] is not None

    def test_index_and_query(self, imported):
        pool, result = imported
        indexer = Indexer(pool)
        indexer.reindex_all()

        from neuroatom.index.query import QueryBuilder

        qb = QueryBuilder(indexer.backend)
        atom_ids = qb.query_atom_ids({"dataset_id": "chinese_eeg_reading"})
        assert len(atom_ids) == result.n_atoms

    def test_channels_json_written(self, imported):
        pool, result = imported
        from neuroatom.storage import paths as P

        ch_file = P.channels_path(
            pool.root, "chinese_eeg_reading",
            f"sub-{TEST_SUBJECT}", f"ses-{TEST_SESSION}",
        )
        assert ch_file.exists(), "channels.json should be written"
        import json

        with open(ch_file) as f:
            data = json.load(f)
        assert len(data) == 128
        assert data[0]["channel_id"] == "ch_000"
        assert data[0]["name"] == "E1"
        # EGI E1-E128 names don't map to standard 10-20; may be None
        assert "standard_name" in data[0]


# ======================================================================
# Signal characteristics
# ======================================================================


@pytest.mark.skipif(not HAS_DATA, reason=SKIP_MSG)
class TestSignalCharacteristics:
    """Verify signal data quality from raw BrainVision import."""

    @pytest.fixture(scope="class")
    def signal_data(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEGImporter(pool)
        results = imp.import_dataset(
            BIDS_ROOT,
            subjects=[TEST_SUBJECT],
            sessions=[TEST_SESSION],
            max_runs=1,
        )
        a0 = results[0].atoms[0]
        sig = ShardManager.static_read(pool.root, a0.signal_ref)
        return sig

    def test_units_microvolts(self, signal_data):
        """Signals are stored in µV — typical EEG range."""
        max_abs = np.abs(signal_data).max()
        assert max_abs < 1e6, f"Signal in µV should have max < 1e6, got {max_abs}"
        assert max_abs > 0.01, f"Signal seems too small for µV, got {max_abs}"

    def test_no_nans(self, signal_data):
        assert np.isfinite(signal_data).all()

    def test_not_all_zero(self, signal_data):
        assert not np.allclose(signal_data, 0)

    def test_channel_variance(self, signal_data):
        """Most channels should have non-zero variance."""
        per_ch_std = np.std(signal_data, axis=1)
        zero_channels = int(np.sum(per_ch_std == 0))
        # Allow up to 10 flat channels (short epoch artifacts)
        assert zero_channels <= 10, (
            f"{zero_channels}/128 channels have zero variance"
        )


# ======================================================================
# Multi-session import
# ======================================================================


@pytest.mark.skipif(not HAS_DATA, reason=SKIP_MSG)
class TestMultiSession:
    """Import a single subject across both sessions (1 run each)."""

    @pytest.fixture(scope="class")
    def imported(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEGImporter(pool)
        # Import 1 run from each session to cover both novels
        results_lp = imp.import_dataset(
            BIDS_ROOT,
            subjects=[TEST_SUBJECT],
            sessions=["LittlePrince"],
            max_runs=1,
        )
        results_gd = imp.import_dataset(
            BIDS_ROOT,
            subjects=[TEST_SUBJECT],
            sessions=["GarnettDream"],
            max_runs=1,
        )
        return pool, results_lp + results_gd

    def test_two_sessions(self, imported):
        _, results = imported
        sessions = set(r.run_meta.session_id for r in results if r.n_atoms > 0)
        assert len(sessions) == 2, f"Expected 2 sessions, got {sessions}"

    def test_both_novels(self, imported):
        _, results = imported
        novels = set()
        for r in results:
            if r.atoms:
                novels.add(r.atoms[0].custom_fields["novel"])
        assert "LittlePrince" in novels
        assert "GarnettDream" in novels


# ======================================================================
# import_subject convenience API
# ======================================================================


@pytest.mark.skipif(not HAS_DATA, reason=SKIP_MSG)
class TestImportSubjectAPI:
    """Test the import_subject convenience method."""

    def test_import_single_subject(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEGImporter(pool)
        results = imp.import_subject(
            BIDS_ROOT,
            subject_id=TEST_SUBJECT,
            sessions=[TEST_SESSION],
            max_runs=1,
        )
        assert len(results) == 1
        assert results[0].n_atoms > 0

    def test_import_subject_with_prefix(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEGImporter(pool)
        results = imp.import_subject(
            BIDS_ROOT,
            subject_id=f"sub-{TEST_SUBJECT}",
            sessions=[TEST_SESSION],
            max_runs=1,
        )
        assert len(results) == 1
        assert results[0].atoms[0].subject_id == f"sub-{TEST_SUBJECT}"


# ======================================================================
# P4: Text embedding tests
# ======================================================================


@pytest.mark.skipif(not HAS_DATA, reason=SKIP_MSG)
class TestTextEmbeddings:
    """Tests for BERT text embedding import in ChineseEEG."""

    def test_load_text_embeddings_run(self):
        emb = _load_text_embeddings_run(BIDS_ROOT, TEST_SESSION, "1")
        assert emb is not None, "Text embedding file should exist for run-01"
        assert emb.ndim == 2
        assert emb.shape[1] == 768
        assert emb.dtype == np.float32

    def test_load_text_embeddings_missing_run(self):
        emb = _load_text_embeddings_run(BIDS_ROOT, TEST_SESSION, "999")
        assert emb is None

    def test_text_embedding_annotation_present(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEGImporter(pool)
        results = imp.import_subject(
            BIDS_ROOT,
            subject_id=TEST_SUBJECT,
            sessions=[TEST_SESSION],
            max_runs=1,
        )
        assert results[0].n_atoms > 0
        atom = results[0].atoms[0]
        ann_names = [a.name for a in atom.annotations]
        assert "text_embedding" in ann_names, (
            f"Expected 'text_embedding' annotation, got: {ann_names}"
        )

    def test_text_embedding_annotation_shape(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEGImporter(pool)
        results = imp.import_subject(
            BIDS_ROOT,
            subject_id=TEST_SUBJECT,
            sessions=[TEST_SESSION],
            max_runs=1,
        )
        from neuroatom.core.annotation import ContinuousAnnotation
        atom = results[0].atoms[0]
        emb_ann = next(
            (a for a in atom.annotations
             if isinstance(a, ContinuousAnnotation) and a.name == "text_embedding"),
            None,
        )
        assert emb_ann is not None
        assert emb_ann.data_ref.shape == (768,)
        assert emb_ann.data_sampling_rate == 1.0

    def test_text_embedding_stored_in_hdf5(self):
        td = tempfile.mkdtemp()
        pool = Pool.create(td)
        imp = ChineseEEGImporter(pool)
        results = imp.import_subject(
            BIDS_ROOT,
            subject_id=TEST_SUBJECT,
            sessions=[TEST_SESSION],
            max_runs=1,
        )
        import h5py
        from neuroatom.storage import paths as P
        atom = results[0].atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        assert shard_path.exists()
        with h5py.File(str(shard_path), "r") as f:
            ann_path = f"/atoms/{atom.atom_id}/annotations/text_embedding"
            assert ann_path in f, f"Expected {ann_path} in HDF5"
            arr = f[ann_path][:]
        assert arr.shape == (768,)
        assert arr.dtype == np.float32
