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
    _parse_run_id,
    _read_electrodes_tsv,
    _read_tsv,
)
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.index.indexer import Indexer

PL_ROOT = Path(r"C:\Data\ChineseEEG-2\PassiveListening")
RA_ROOT = Path(r"C:\Data\ChineseEEG-2\ReadingAloud")

SKIP_MSG = "ChineseEEG-2 data not found at expected path"
HAS_PL = PL_ROOT.exists() and (PL_ROOT / "dataset_description.json").exists()
HAS_RA = RA_ROOT.exists() and (RA_ROOT / "dataset_description.json").exists()


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

    def test_units_volts(self, signal_data):
        """Raw BrainVision data in V — values should be small."""
        sig = signal_data
        assert np.abs(sig).max() < 1.0, "Signal in V should have max < 1.0"

    def test_no_nans(self, signal_data):
        assert np.isfinite(signal_data).all()

    def test_not_all_zero(self, signal_data):
        assert not np.allclose(signal_data, 0)
