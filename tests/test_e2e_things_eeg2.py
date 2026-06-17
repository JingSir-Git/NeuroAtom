"""End-to-end tests for the THINGS-EEG2 importer.

Requires the dataset at D:\\Data\\things_eeg2. Skipped automatically otherwise.
The .npy arrays are large (200+ MB), so the import test loads the test split
once and scopes to a few conditions/reps.
"""

import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\things_eeg2")
PP = DATA_ROOT / "preprocessed"
SUB01 = PP / "sub-01"
TEST_NPY = SUB01 / "preprocessed_eeg_test.npy"

pytestmark = pytest.mark.skipif(
    not TEST_NPY.exists(),
    reason="THINGS-EEG2 data not available at D:\\Data\\things_eeg2",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("things_eeg2")


class TestThingsEEG2Detect:
    def test_detect_root(self):
        from neuroatom.importers.things_eeg2 import ThingsEEG2Importer
        assert ThingsEEG2Importer.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.things_eeg2 import ThingsEEG2Importer
        assert ThingsEEG2Importer.detect(TEST_NPY) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.things_eeg2 import ThingsEEG2Importer
        assert ThingsEEG2Importer.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(TEST_NPY) == "things_eeg2"


class TestThingsEEG2Importer:
    def test_import_subject_test_split(self, pool, task_config):
        from neuroatom.importers.things_eeg2 import ThingsEEG2Importer

        imp = ThingsEEG2Importer(pool=pool, task_config=task_config)
        result = imp.import_subject(
            SUB01, splits=["test"], max_conditions=2, max_reps=3,
        )
        assert len(result.atoms) == 6  # 2 conditions x 3 reps

        atom = result.atoms[0]
        assert atom.dataset_id == "things_eeg2"
        assert atom.subject_id == "sub-01"
        assert atom.atom_type.value == "event_epoch"
        assert atom.sampling_rate == 100.0
        assert atom.n_channels == 17
        assert atom.signal_unit == "au"
        assert atom.temporal.duration_samples == 100

        ann = {a.name for a in atom.annotations}
        assert {"image_condition", "repetition", "split"} <= ann
        split = next(a.value for a in atom.annotations if a.name == "split")
        assert split == "test"

        # condition / rep indices span the requested scope
        conds = {
            int(next(x.numeric_value for x in a.annotations if x.name == "image_condition"))
            for a in result.atoms
        }
        assert conds == {0, 1}

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.things_eeg2 import ThingsEEG2Importer

        imp = ThingsEEG2Importer(pool=pool, task_config=task_config)
        result = imp.import_subject(
            SUB01, splits=["test"], max_conditions=1, max_reps=2,
        )
        atom = result.atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape == (17, 100)
