"""End-to-end tests for the ear-EEG-AAD importer (TMSi SAGA Poly5).

Requires the dataset at D:\\Data\\ear_eeg_aad. Skipped automatically otherwise.
The custom Poly5 reader is exercised against the real files.
"""

import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\ear_eeg_aad")
SUB1 = DATA_ROOT / "sub1"
SUB1_POLY5 = SUB1 / "sub1.Poly5"

pytestmark = pytest.mark.skipif(
    not SUB1_POLY5.exists(),
    reason="ear-EEG-AAD data not available at D:\\Data\\ear_eeg_aad",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("ear_eeg_aad")


class TestPoly5Reader:
    def test_read_header_and_data(self):
        from neuroatom.importers.ear_eeg_aad import _read_poly5
        sfreq, names, data = _read_poly5(SUB1_POLY5, max_seconds=10)
        assert sfreq == 500.0
        assert len(names) == 52          # 104 signals / 2 (Lo/Hi pairing)
        assert data.shape[0] == 52
        assert data.shape[1] == pytest.approx(10 * 500, abs=500)  # ~10 s, block-rounded
        # device/aux channels present in the name list
        assert any(n.startswith("UNI") for n in names)
        assert "TRIGGERS" in names and "STATUS" in names

    def test_eeg_channels_have_uv_amplitude(self):
        import numpy as np
        from neuroatom.importers.ear_eeg_aad import _read_poly5, _is_eeg
        sfreq, names, data = _read_poly5(SUB1_POLY5, max_seconds=10)
        eeg = data[[i for i, n in enumerate(names) if _is_eeg(n)], :]
        # EEG channels should sit in a sane µV range, not device-counter values
        assert np.median(eeg.std(axis=1)) < 500


class TestEarEEGAADImporter:
    def test_detect_root(self):
        from neuroatom.importers.ear_eeg_aad import EarEEGAADImporter
        assert EarEEGAADImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.ear_eeg_aad import EarEEGAADImporter
        assert EarEEGAADImporter.detect(SUB1_POLY5) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.ear_eeg_aad import EarEEGAADImporter
        assert EarEEGAADImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(SUB1_POLY5) == "ear_eeg_aad"

    def test_import_subject(self, pool, task_config):
        from neuroatom.importers.ear_eeg_aad import EarEEGAADImporter

        imp = EarEEGAADImporter(pool=pool, task_config=task_config)
        result = imp.import_subject(SUB1_POLY5, max_seconds=15)

        assert len(result.atoms) == 1
        atom = result.atoms[0]
        assert atom.dataset_id == "ear_eeg_aad"
        assert atom.subject_id == "sub1"
        assert atom.atom_type.value == "continuous_segment"
        assert atom.sampling_rate == 500.0
        assert atom.n_channels >= 32  # UNI/BIP EEG channels kept

        names = {c.name for c in result.channel_infos}
        assert "TRIGGERS" not in names and "STATUS" not in names
        assert "Counter 2power24" not in names

        ann = {a.name: a.value for a in atom.annotations}
        assert ann["modality"] == "ear_eeg"
        assert ann["label_provenance"] == "unresolved"

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.ear_eeg_aad import EarEEGAADImporter
        imp = EarEEGAADImporter(pool=pool, task_config=task_config)
        result = imp.import_subject(SUB1_POLY5, max_seconds=10)
        atom = result.atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape[0] == atom.n_channels
            assert sig.shape[1] > 1000
