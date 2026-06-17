"""End-to-end tests for the Sleep-EDF Expanded importer.

Requires the dataset at D:\\Data\\Sleep_EDF_Expanded and the optional `mne`
dependency. Skipped automatically otherwise.
"""

import pytest
from pathlib import Path

mne = pytest.importorskip("mne")

DATA_ROOT = Path(r"D:\Data\Sleep_EDF_Expanded")
SC = DATA_ROOT / "sleep-cassette"
PSG = SC / "SC4001E0-PSG.edf"
HYP = SC / "SC4001EC-Hypnogram.edf"

pytestmark = pytest.mark.skipif(
    not PSG.exists(),
    reason="Sleep-EDF data not available at D:\\Data\\Sleep_EDF_Expanded",
)

_STAGES = {"W", "N1", "N2", "N3", "N4", "REM"}


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("sleep_edf")


class TestSleepEDFParsing:
    def test_parse_recording_id(self):
        from neuroatom.importers.sleep_edf import _parse_recording_id
        sub, ses, base = _parse_recording_id(PSG)
        assert sub == "SC00"
        assert ses == "night1"
        assert base == "SC4001E0"

    def test_pair_recordings(self):
        from neuroatom.importers.sleep_edf import _pair_recordings
        pairs = _pair_recordings(SC)
        assert len(pairs) > 0
        assert all(p.name[:7] == h.name[:7] for p, h in pairs)
        assert any(p.name == "SC4001E0-PSG.edf" for p, _ in pairs)

    def test_load_stage_bouts(self):
        from neuroatom.importers.sleep_edf import _load_stage_bouts, _stage_at
        bouts = _load_stage_bouts(HYP)
        assert len(bouts) > 100
        onset, end, stage = bouts[0]
        assert onset == 0.0
        assert stage == "W"
        # mid first bout resolves to W
        assert _stage_at(bouts, 100.0) == "W"


class TestSleepEDFImporter:
    def test_detect_root(self):
        from neuroatom.importers.sleep_edf import SleepEDFImporter
        assert SleepEDFImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.sleep_edf import SleepEDFImporter
        assert SleepEDFImporter.detect(PSG) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.sleep_edf import SleepEDFImporter
        assert SleepEDFImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(PSG) == "sleep_edf"

    def test_import_recording_structure(self, pool, task_config):
        from neuroatom.importers.sleep_edf import SleepEDFImporter

        imp = SleepEDFImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(PSG, HYP, max_epochs=8)

        assert len(result.atoms) == 8
        atom = result.atoms[0]
        assert atom.dataset_id == "sleep_edf"
        assert atom.subject_id == "SC00"
        assert atom.session_id == "night1"
        assert atom.atom_type.value == "window"
        assert atom.sampling_rate == 100.0
        assert atom.n_channels == 4  # 2 EEG + EOG + EMG
        assert atom.temporal.duration_samples == 3000  # 30 s @ 100 Hz

        for a in result.atoms:
            stage = next(x.value for x in a.annotations if x.name == "sleep_stage")
            assert stage in _STAGES

    def test_channel_types(self, pool, task_config):
        from neuroatom.importers.sleep_edf import SleepEDFImporter
        imp = SleepEDFImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(PSG, HYP, max_epochs=2)
        types = {c.name: c.type.value for c in result.channel_infos}
        assert types["EEG Fpz-Cz"] == "eeg"
        assert types["EOG horizontal"] == "eog"
        assert types["EMG submental"] == "emg"

    def test_multiple_stages_across_sleep(self, pool, task_config):
        """Spanning sleep onset yields more than one stage."""
        from neuroatom.importers.sleep_edf import SleepEDFImporter
        imp = SleepEDFImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(PSG, HYP, max_epochs=120)
        stages = {
            next(x.value for x in a.annotations if x.name == "sleep_stage")
            for a in result.atoms
        }
        assert len(stages) >= 2
        assert stages <= _STAGES

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.sleep_edf import SleepEDFImporter
        imp = SleepEDFImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(PSG, HYP, max_epochs=3)
        atom = result.atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape == (4, 3000)
