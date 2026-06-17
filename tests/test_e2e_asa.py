"""End-to-end tests for the ASA importer (per-trial FIF, labels unresolved).

Requires the dataset at D:\\Data\\ASA and the optional `mne` dependency.
Skipped automatically otherwise.
"""

import pytest
from pathlib import Path

mne = pytest.importorskip("mne")

DATA_ROOT = Path(r"D:\Data\ASA")
S001 = DATA_ROOT / "S001"
TRIAL1 = S001 / "E1" / "S001_E1_Trial1_raw.fif"

pytestmark = pytest.mark.skipif(
    not TRIAL1.exists(),
    reason="ASA data not available at D:\\Data\\ASA",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("asa")


class TestASAImporter:
    def test_detect_root(self):
        from neuroatom.importers.asa import ASAImporter
        assert ASAImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.asa import ASAImporter
        assert ASAImporter.detect(TRIAL1) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.asa import ASAImporter
        assert ASAImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(TRIAL1) == "asa"

    def test_import_subject(self, pool, task_config):
        from neuroatom.importers.asa import ASAImporter

        imp = ASAImporter(pool=pool, task_config=task_config)
        result = imp.import_subject(S001, max_trials=3)

        assert len(result.atoms) == 3
        atom = result.atoms[0]
        assert atom.dataset_id == "asa"
        assert atom.subject_id == "S001"
        assert atom.atom_type.value == "trial"
        assert atom.sampling_rate == 500.0
        assert atom.n_channels == 64

        ann = {a.name: a for a in atom.annotations}
        assert ann["trial_number"].numeric_value == 1.0
        # labels are deliberately not assigned
        assert ann["label_provenance"].value == "unresolved"
        assert "attended_direction" not in ann
        assert atom.custom_fields["label_provenance"] == "unresolved"

    def test_segment_cropped_to_attended_window(self, pool, task_config):
        """The atom is cropped to trailS->trailE, shorter than the full ~61 s file."""
        from neuroatom.importers.asa import ASAImporter
        imp = ASAImporter(pool=pool, task_config=task_config)
        result = imp.import_subject(S001, max_trials=1)
        atom = result.atoms[0]
        raw = mne.io.read_raw_fif(str(TRIAL1), preload=False, verbose="ERROR")
        assert atom.temporal.duration_samples < raw.n_times

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.asa import ASAImporter
        imp = ASAImporter(pool=pool, task_config=task_config)
        result = imp.import_subject(S001, max_trials=1)
        atom = result.atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape[0] == 64
            assert sig.shape[1] > 1000
