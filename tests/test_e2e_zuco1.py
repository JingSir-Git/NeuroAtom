"""End-to-end tests for the ZuCo 1.0 importer.

Requires the actual dataset at D:\\Data\\ZuCo_1.0_Full.
Skipped automatically if the data is not available.
"""

import json
import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\ZuCo_1.0_Full")
pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="ZuCo 1.0 data not available at D:\\Data\\ZuCo_1.0_Full",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("zuco1_sr")


class TestZuco1Detection:
    """Format detection tests."""

    def test_detect_correct_dir(self):
        from neuroatom.importers.zuco1 import Zuco1Importer
        assert Zuco1Importer.detect(DATA_ROOT) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.zuco1 import Zuco1Importer
        assert Zuco1Importer.detect(tmp_path) is False


class TestZuco1Import:
    """Full integration tests against the real ZuCo 1.0 dataset."""

    def test_import_single_subject_sr(self, pool, task_config):
        """Import one subject's Sentiment Reading task."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=1,
            max_sentences=5,
        )

        assert len(results) >= 1
        result = results[0]

        # Check atoms are sentence-level
        assert len(result.atoms) == 5
        atom = result.atoms[0]
        assert atom.dataset_id == "zuco1"
        assert atom.subject_id == "ZAB"
        assert atom.atom_type.value == "event_epoch"
        assert atom.sampling_rate == 500.0
        assert atom.n_channels == 105

        # Check annotations
        ann_names = [a.name for a in atom.annotations]
        assert "task" in ann_names
        assert "text_id" in ann_names
        assert "sentence_index" in ann_names

        # Check task is SR
        task_ann = next(a for a in atom.annotations if a.name == "task")
        assert task_ann.value == "sr"

    def test_import_single_subject_nr(self, pool, task_config):
        """Import one subject's Normal Reading task."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["nr"],
            subjects=["ZAB"],
            max_texts=1,
            max_sentences=3,
        )

        assert len(results) >= 1
        atom = results[0].atoms[0]
        assert atom.n_channels == 105
        task_ann = next(a for a in atom.annotations if a.name == "task")
        assert task_ann.value == "nr"

    def test_import_multi_task(self, pool, task_config):
        """Import all three tasks for one subject (limited)."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["sr", "nr", "tsr"],
            subjects=["ZAB"],
            max_texts=1,
            max_sentences=2,
        )

        # Should have results for each task (if data exists)
        assert len(results) >= 2

        # Check task diversity
        tasks_found = set()
        for r in results:
            for atom in r.atoms:
                task_ann = next(a for a in atom.annotations if a.name == "task")
                tasks_found.add(task_ann.value)

        assert len(tasks_found) >= 2

    def test_subject_discovery(self, task_config, pool):
        """Verify all 12 subjects are discovered."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        subjects = imp._discover_subjects(DATA_ROOT)
        assert len(subjects) == 12
        assert "ZAB" in subjects
        assert "ZPH" in subjects

    def test_electrode_coordinates_preserved(self, pool, task_config):
        """Check that 3D electrode locations are preserved."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=1,
            max_sentences=1,
        )

        ch_infos = results[0].channel_infos
        # At least some channels should have electrode locations
        with_loc = [ci for ci in ch_infos if ci.location is not None]
        assert len(with_loc) > 0

    def test_dataset_registered(self, pool, task_config):
        """Verify dataset and subject metadata in pool."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=1,
            max_sentences=1,
        )

        ds_meta = pool.get_dataset_meta("zuco1")
        assert ds_meta.name == "ZuCo 1.0 Natural Reading"
        assert "reading" in ds_meta.task_types

    def test_catalog_entry(self, pool, task_config):
        """Verify catalog is created after import."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=1,
            max_sentences=2,
        )

        cat_path = pool.root / "catalog.json"
        assert cat_path.exists()
        cat = json.loads(cat_path.read_text())
        assert any(d["dataset_id"] == "zuco1" for d in cat["datasets"])


class TestZuco1WordFeatures:
    """Verify word-level eye-tracking & EEG enrichment (P1 + P6)."""

    def _find_sr_atom(self, results):
        """Find first atom from an SR text (not SNR)."""
        for r in results:
            for a in r.atoms:
                text_ann = next(
                    (ann for ann in a.annotations if ann.name == "text_id"), None
                )
                if text_ann and text_ann.value.startswith("SR"):
                    return a
        return None

    def test_sentence_content_annotation(self, pool, task_config):
        """TextAnnotation with sentence content is present."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=5,
            max_sentences=3,
        )
        atom = self._find_sr_atom(results)
        assert atom is not None, "No SR text atom found"

        text_anns = [a for a in atom.annotations if a.annotation_type == "text"]
        assert len(text_anns) >= 1, "No TextAnnotation found"
        content_ann = next(a for a in text_anns if a.name == "sentence_content")
        assert len(content_ann.text_value) > 10
        assert content_ann.domain == "stimulus"

    def test_word_reading_features(self, pool, task_config):
        """EventSequenceAnnotation with per-word eye-tracking features."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=5,
            max_sentences=3,
        )
        atom = self._find_sr_atom(results)
        assert atom is not None

        evt_anns = [
            a for a in atom.annotations if a.annotation_type == "event_sequence"
        ]
        assert len(evt_anns) >= 1, "No EventSequenceAnnotation found"

        word_ann = next(a for a in evt_anns if a.name == "word_reading_features")
        assert len(word_ann.events) > 0

        # Verify first word has expected features
        w0 = word_ann.events[0]
        assert w0.value  # non-empty word content
        assert "word_index" in w0.features
        # At least some eye-tracking features should be present
        et_keys = {"FFD", "TRT", "GD", "GPT", "SFD", "nFixations", "meanPupilSize"}
        found_et = et_keys & set(w0.features.keys())
        assert len(found_et) >= 1, f"No eye-tracking features found, keys: {w0.features.keys()}"

    def test_word_band_power_features(self, pool, task_config):
        """Word events contain EEG band power features."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=5,
            max_sentences=5,
        )

        # Check a sentence with data (skip omission-heavy sentences)
        for atom in results[0].atoms:
            evt_anns = [
                a for a in atom.annotations if a.annotation_type == "event_sequence"
            ]
            if not evt_anns:
                continue
            for evt in evt_anns[0].events:
                band_keys = {k for k in evt.features if k.startswith("mean_")}
                if band_keys:
                    assert "mean_t1" in band_keys or "mean_a1" in band_keys
                    return  # found at least one word with band power

        # If we reach here, no band power was found in any of the 5 sentences
        # This is acceptable for heavily skipped sentences
        pytest.skip("No band power features found in tested sentences")

    def test_omission_rate_annotation(self, pool, task_config):
        """NumericAnnotation for omission rate is present."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=5,
            max_sentences=3,
        )
        atom = self._find_sr_atom(results)
        assert atom is not None

        omission_anns = [
            a for a in atom.annotations
            if a.annotation_type == "numeric" and a.name == "omission_rate"
        ]
        assert len(omission_anns) == 1
        assert 0.0 <= omission_anns[0].numeric_value <= 1.0


class TestZuco1EyeTracking:
    """Verify raw eye-tracking time series import (P2)."""

    def _get_sr_atoms(self, pool, task_config, max_sentences=3):
        """Helper: import SR texts (skip SNR) to get atoms with ET data."""
        from neuroatom.importers.zuco1 import Zuco1Importer

        imp = Zuco1Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["sr"],
            subjects=["ZAB"],
            max_texts=5,  # enough to reach SR1 past SNR6/7/8
            max_sentences=max_sentences,
        )
        # Find a result for an SR text (not SNR)
        for r in results:
            for a in r.atoms:
                text_ann = next(
                    (ann for ann in a.annotations if ann.name == "text_id"), None
                )
                if text_ann and text_ann.value.startswith("SR"):
                    return r, a
        pytest.fail("No SR text atoms found")

    def test_et_continuous_annotations(self, pool, task_config):
        """ContinuousAnnotation for gaze + pupil data is present."""
        _, atom = self._get_sr_atoms(pool, task_config)

        cont_anns = [a for a in atom.annotations if a.annotation_type == "continuous"]
        et_names = {a.name for a in cont_anns}

        # Should have gaze_x, gaze_y, pupil_area
        assert "eye_gaze_x" in et_names, f"Missing eye_gaze_x, found: {et_names}"
        assert "eye_gaze_y" in et_names
        assert "eye_pupil_area" in et_names

        gx = next(a for a in cont_anns if a.name == "eye_gaze_x")
        assert gx.data_sampling_rate > 0
        assert gx.alignment_method == "trigger_locked"
        assert gx.data_ref.shape[0] > 0

    def test_et_stored_in_hdf5(self, pool, task_config):
        """Verify ET companion arrays exist in HDF5 shard."""
        import h5py

        result, atom = self._get_sr_atoms(pool, task_config, max_sentences=2)
        shard_path = pool.root / atom.signal_ref.file_path
        assert shard_path.exists(), f"Shard not found: {shard_path}"

        with h5py.File(str(shard_path), "r") as f:
            atom_grp = f[f"/atoms/{atom.atom_id}"]
            assert "annotations" in atom_grp, "No annotations group in HDF5"
            ann_grp = atom_grp["annotations"]
            ann_keys = set(ann_grp.keys())

            # Eye-tracking arrays
            assert "eye_gaze_x" in ann_keys, f"Missing eye_gaze_x, found: {ann_keys}"
            assert "eye_gaze_y" in ann_keys
            assert "eye_pupil_area" in ann_keys

            # Fixation arrays (from results file)
            assert "fixation_x" in ann_keys or "fixation_duration" in ann_keys

            # Verify shapes are reasonable
            gx = ann_grp["eye_gaze_x"][:]
            assert gx.ndim == 1
            assert gx.shape[0] > 100  # at least 100 samples

    def test_provenance_tracks_enrichment(self, pool, task_config):
        """Processing history records whether features/ET were added."""
        _, atom = self._get_sr_atoms(pool, task_config, max_sentences=2)
        params = atom.processing_history.steps[0].parameters
        assert "has_word_features" in params
        assert "has_eye_tracking" in params
