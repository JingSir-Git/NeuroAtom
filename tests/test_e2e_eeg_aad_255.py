"""End-to-end tests for the EEG-AAD-255 importer.

Requires the actual dataset at D:\\Data\\EEG-AAD-255 and the optional `mne`
dependency (Curry reader). Skipped automatically otherwise.

Recordings are ~440 MB each, so tests crop to a few seconds (max_seconds).
"""

import pytest
from pathlib import Path

mne = pytest.importorskip("mne")

DATA_ROOT = Path(r"D:\Data\EEG-AAD-255")
SUB0 = DATA_ROOT / "S0" / "S0"
SUB0_1L = SUB0 / "S0_AAD_1L.dat"

pytestmark = pytest.mark.skipif(
    not SUB0_1L.exists(),
    reason="EEG-AAD-255 data not available at D:\\Data\\EEG-AAD-255",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("eeg_aad_255")


class TestEEGAAD255Detect:
    def test_detect_root(self):
        from neuroatom.importers.eeg_aad_255 import EEGAAD255Importer
        assert EEGAAD255Importer.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.eeg_aad_255 import EEGAAD255Importer
        assert EEGAAD255Importer.detect(SUB0_1L) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.eeg_aad_255 import EEGAAD255Importer
        assert EEGAAD255Importer.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(SUB0_1L) == "eeg_aad_255"


class TestEEGAAD255Importer:
    def test_import_single_subject(self, pool, task_config):
        from neuroatom.importers.eeg_aad_255 import EEGAAD255Importer

        imp = EEGAAD255Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["S0"], max_trials=2, max_seconds=15,
        )

        assert len(results) == 1
        result = results[0]
        assert len(result.atoms) == 2

        atom = result.atoms[0]
        assert atom.dataset_id == "eeg_aad_255"
        assert atom.subject_id == "S0"
        assert atom.atom_type.value == "trial"
        assert atom.sampling_rate == 1000.0
        assert atom.signal_unit == "uV"
        # 253 scalp EEG + XtraL + XtraR, Trigger excluded
        assert atom.n_channels == 255

        ann_names = {a.name for a in atom.annotations}
        assert "attended_direction" in ann_names
        assert "block" in ann_names

    def test_attended_direction_from_filename(self, pool, task_config):
        """S0_AAD_1L → left, S0_AAD_1R → right."""
        from neuroatom.importers.eeg_aad_255 import EEGAAD255Importer

        imp = EEGAAD255Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["S0"], max_trials=2, max_seconds=10,
        )
        atoms = results[0].atoms
        directions = {
            a.run_id: next(x.value for x in a.annotations if x.name == "attended_direction")
            for a in atoms
        }
        assert directions.get("block1_l") == "left"
        assert directions.get("block1_r") == "right"

    def test_trigger_excluded_externals_kept(self, pool, task_config):
        from neuroatom.importers.eeg_aad_255 import EEGAAD255Importer

        imp = EEGAAD255Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["S0"], max_trials=1, max_seconds=10,
        )
        ch_infos = results[0].channel_infos
        names = {c.name for c in ch_infos}
        assert "Trigger" not in names
        assert "XtraL" in names and "XtraR" in names
        externals = [c for c in ch_infos if c.name in ("XtraL", "XtraR")]
        assert all(c.type.value == "other" for c in externals)

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.eeg_aad_255 import EEGAAD255Importer

        imp = EEGAAD255Importer(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["S0"], max_trials=1, max_seconds=10,
        )
        atom = results[0].atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        assert shard_path.exists()
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape[0] == 255
            assert sig.shape[1] > 1000  # ~10 s @ 1000 Hz
