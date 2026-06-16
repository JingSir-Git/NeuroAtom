"""End-to-end tests for the EEG-iEEG Verbal Working Memory importer.

Requires the actual dataset at D:\\Data\\original.
Skipped automatically if the data is not available.
"""

import json
import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\original")
pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="EEG-iEEG WM data not available at D:\\Data\\original",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("eeg_ieeg_wm")


class TestEEGiEEGDetection:

    def test_detect_correct_dir(self):
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter
        assert EEGiEEGWMImporter.detect(DATA_ROOT) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter
        assert EEGiEEGWMImporter.detect(tmp_path) is False


class TestEEGiEEGImport:
    """Integration tests against the real EEG-iEEG dataset."""

    def test_import_single_subject_both_modalities(self, pool, task_config):
        """Import one subject, one session, both EEG and iEEG."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-01"],
            max_sessions=1,
        )

        # Should have 2 results: one for EEG, one for iEEG
        assert len(results) == 2

        modalities = {r.run_meta.run_id for r in results}
        assert "run-eeg" in modalities
        assert "run-ieeg" in modalities

    def test_eeg_atoms_correct(self, pool, task_config):
        """Check EEG atom properties."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-01"],
            max_sessions=1,
            modalities=["eeg"],
        )

        assert len(results) == 1
        r = results[0]
        assert r.run_meta.run_id == "run-eeg"
        assert len(r.atoms) > 0

        atom = r.atoms[0]
        assert atom.dataset_id == "eeg_ieeg_wm"
        assert atom.subject_id == "sub-01"
        assert atom.atom_type.value == "trial"
        # EEG: 19 channels @ 200 Hz
        assert atom.n_channels == 19
        assert atom.sampling_rate == 200.0

        # Check WM annotations
        ann_names = [a.name for a in atom.annotations]
        assert "set_size" in ann_names
        assert "match" in ann_names
        assert "correct" in ann_names
        assert "response_time" in ann_names
        assert "modality" in ann_names

    def test_ieeg_atoms_correct(self, pool, task_config):
        """Check iEEG atom properties."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-01"],
            max_sessions=1,
            modalities=["ieeg"],
        )

        assert len(results) == 1
        r = results[0]
        assert r.run_meta.run_id == "run-ieeg"

        atom = r.atoms[0]
        # sub-01 has 48 SEEG channels @ 2000 Hz
        assert atom.n_channels == 48
        assert atom.sampling_rate == 2000.0
        assert atom.custom_fields["modality"] == "ieeg"

    def test_seeg_ecog_mixed_subject(self, pool, task_config):
        """sub-10 has both SEEG + ECoG channels."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-10"],
            max_sessions=1,
            modalities=["ieeg"],
        )

        assert len(results) == 1
        r = results[0]
        atom = r.atoms[0]
        # sub-10 has 80 channels (SEEG + ECOG)
        assert atom.n_channels == 80

        # Check iEEG type annotations
        ann_names = [a.name for a in atom.annotations]
        assert "has_seeg" in ann_names
        assert "has_ecog" in ann_names

    def test_electrode_coordinates(self, pool, task_config):
        """Verify MNI electrode coordinates for iEEG."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-01"],
            max_sessions=1,
            modalities=["ieeg"],
        )

        ch_infos = results[0].channel_infos
        with_loc = [ci for ci in ch_infos if ci.location is not None]
        assert len(with_loc) > 0

        # Check MNI coordinate system
        loc = with_loc[0].location
        assert loc.coordinate_system == "MNI"
        assert loc.coordinate_units == "mm"

        # Check anatomical labels
        with_anat = [ci for ci in ch_infos if ci.custom_fields.get("anatomical_location")]
        assert len(with_anat) > 0

    def test_cross_modality_linking(self, pool, task_config):
        """EEG and iEEG atoms for same trial should be linked."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-01"],
            max_sessions=1,
        )

        # Find EEG and iEEG results
        eeg_result = next(r for r in results if r.run_meta.run_id == "run-eeg")
        ieeg_result = next(r for r in results if r.run_meta.run_id == "run-ieeg")

        # Both should have same number of trials
        assert len(eeg_result.atoms) == len(ieeg_result.atoms)

        # The second modality imported should have cross-links
        # (ieeg is imported second, so it links to eeg)
        linked_ieeg = [a for a in ieeg_result.atoms if a.relations]
        assert len(linked_ieeg) > 0
        assert linked_ieeg[0].relations[0].relation_type == "simultaneous_recording"

    def test_events_parsing(self):
        """Verify events.tsv parsing."""
        from neuroatom.importers.eeg_ieeg_wm import _parse_events_tsv

        events_path = DATA_ROOT / "sub-01" / "ses-01" / "eeg" / "sub-01_ses-01_task-verbalWM_run-01_events.tsv"
        trials = _parse_events_tsv(events_path)

        assert len(trials) > 0
        trial = trials[0]
        assert trial["set_size"] in (4, 6, 8)
        assert trial["match"] in ("IN", "OUT")
        assert trial["correct"] in (0, 1)
        assert trial["duration"] == 8.0

    def test_channels_parsing(self):
        """Verify channels.tsv parsing for both modalities."""
        from neuroatom.importers.eeg_ieeg_wm import _parse_channels_tsv

        # EEG
        eeg_ch = DATA_ROOT / "sub-01" / "ses-01" / "eeg" / "sub-01_ses-01_task-verbalWM_run-01_channels.tsv"
        eeg_infos = _parse_channels_tsv(eeg_ch, 200.0)
        assert len(eeg_infos) == 19

        # iEEG
        ieeg_ch = DATA_ROOT / "sub-01" / "ses-01" / "ieeg" / "sub-01_ses-01_task-verbalWM_run-01_channels.tsv"
        ieeg_infos = _parse_channels_tsv(ieeg_ch, 2000.0)
        assert len(ieeg_infos) == 48
        # All should be SEEG for sub-01
        seeg = [ci for ci in ieeg_infos if ci.custom_fields.get("bids_type") == "SEEG"]
        assert len(seeg) == 48

    def test_subject_metadata(self, pool, task_config):
        """Verify patient metadata stored."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-01"],
            max_sessions=1,
            modalities=["eeg"],
        )

        meta = pool.get_subject_meta("eeg_ieeg_wm", "sub-01")
        assert meta.age == 24
        assert meta.sex == "F"
        assert "pathology" in meta.custom_fields

    def test_multiple_sessions(self, pool, task_config):
        """Sub-01 has 4 sessions. Import 2 sessions."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-01"],
            max_sessions=2,
            modalities=["eeg"],
        )

        # Should have 2 results (1 per session)
        assert len(results) == 2

    def test_catalog_created(self, pool, task_config):
        """Verify catalog.json after import."""
        from neuroatom.importers.eeg_ieeg_wm import EEGiEEGWMImporter

        imp = EEGiEEGWMImporter(pool=pool, task_config=task_config)
        imp.import_dataset(
            DATA_ROOT,
            subjects=["sub-01"],
            max_sessions=1,
            modalities=["eeg"],
        )

        cat_path = pool.root / "catalog.json"
        assert cat_path.exists()
        cat = json.loads(cat_path.read_text())
        assert any(d["dataset_id"] == "eeg_ieeg_wm" for d in cat["datasets"])
