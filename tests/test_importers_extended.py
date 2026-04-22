"""Tests for extended importers: mat, bids, eeglab, moabb_bridge."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neuroatom.core.channel import ChannelInfo
from neuroatom.core.enums import ChannelType
from neuroatom.importers.base import TaskConfig


# ---------------------------------------------------------------------------
# MATImporter
# ---------------------------------------------------------------------------

class TestMATImporter:
    """Test the .mat format importer."""

    def _make_task_config(self, **overrides) -> TaskConfig:
        base = {
            "dataset_id": "test_mat",
            "dataset_name": "Test MAT Dataset",
            "task_type": "motor_imagery",
            "signal_unit": "uV",
            "trial_definition": {"tmin": 0.0, "tmax": 4.0},
            "event_mapping": {1: "left", 2: "right"},
        }
        base.update(overrides)
        return TaskConfig(base)

    def test_detect_mat_file(self, tmp_path):
        from neuroatom.importers.mat import MATImporter

        mat_file = tmp_path / "test.mat"
        mat_file.touch()
        assert MATImporter.detect(mat_file)

        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        assert not MATImporter.detect(txt_file)

    def test_extract_signal_auto_detect(self):
        from neuroatom.importers.mat import _extract_signal

        tc = self._make_task_config()
        # Signal stored as (n_samples, n_channels) — should be transposed
        mat = {"s": np.random.randn(1000, 22)}
        data, key = _extract_signal(mat, tc)
        assert data.shape == (22, 1000)
        assert key == "s"

    def test_extract_signal_with_hint(self):
        from neuroatom.importers.mat import _extract_signal

        tc = self._make_task_config(mat_signal_key="my_data")
        mat = {"my_data": np.random.randn(8, 500)}
        data, key = _extract_signal(mat, tc)
        assert data.shape == (8, 500)
        assert key == "my_data"

    def test_extract_signal_missing_key(self):
        from neuroatom.importers.mat import _extract_signal

        tc = self._make_task_config()
        mat = {"__header__": b"test", "__version__": "1.0", "unrelated": 42}
        with pytest.raises(ValueError, match="Could not find signal data"):
            _extract_signal(mat, tc)

    def test_extract_srate_flat_key(self):
        from neuroatom.importers.mat import _extract_srate

        tc = self._make_task_config()
        mat = {"fs": np.array(256.0)}
        assert _extract_srate(mat, tc) == 256.0

    def test_extract_srate_from_config(self):
        from neuroatom.importers.mat import _extract_srate

        tc = self._make_task_config(sampling_rate=512.0)
        mat = {}
        assert _extract_srate(mat, tc) == 512.0

    def test_extract_ch_names_auto(self):
        from neuroatom.importers.mat import _extract_ch_names

        tc = self._make_task_config()
        mat = {"ch_names": np.array(["C3", "Cz", "C4"])}
        names = _extract_ch_names(mat, tc, 3)
        assert names == ["C3", "Cz", "C4"]

    def test_extract_ch_names_fallback(self):
        from neuroatom.importers.mat import _extract_ch_names

        tc = self._make_task_config()
        mat = {}
        names = _extract_ch_names(mat, tc, 4)
        assert names == ["Ch_1", "Ch_2", "Ch_3", "Ch_4"]

    def test_labels_to_events(self):
        from neuroatom.importers.mat import _labels_to_events

        tc = self._make_task_config()
        labels = np.array([1, 2, 1, 2])
        events = _labels_to_events(labels, 4096, 256.0, tc)
        assert events.shape == (4, 3)
        assert events[0, 2] == 1
        assert events[1, 2] == 2

    def test_parse_events_2d(self):
        from neuroatom.importers.mat import _parse_events_array

        raw = np.array([[100, 0, 1], [200, 0, 2], [300, 0, 1]])
        events = _parse_events_array(raw, 1000)
        assert events.shape == (3, 3)
        assert events[0, 0] == 100

    def test_parse_events_1d_codes(self):
        from neuroatom.importers.mat import _parse_events_array

        # Simulated per-sample event codes with transitions
        codes = np.zeros(100, dtype=int)
        codes[20:30] = 1
        codes[50:60] = 2
        events = _parse_events_array(codes, 100)
        # Should detect transitions at sample 20 and 50
        assert len(events) >= 2

    def test_load_raw_integration(self, tmp_path):
        """Integration test: create a real .mat file and load it."""
        import scipy.io as sio
        from neuroatom.importers.mat import MATImporter
        from neuroatom.storage.pool import Pool

        # Create synthetic .mat
        n_ch, n_samples = 8, 2048
        mat_data = {
            "s": np.random.randn(n_samples, n_ch).astype(np.float64),
            "fs": np.array(256.0),
            "y": np.array([1, 2, 1, 2]),
        }
        mat_path = tmp_path / "test_data.mat"
        sio.savemat(str(mat_path), mat_data)

        # Create pool and importer
        pool = Pool.create(tmp_path / "pool")
        tc = self._make_task_config()
        importer = MATImporter(pool, tc)

        raw, extra = importer.load_raw(mat_path)

        assert raw.info["sfreq"] == 256.0
        assert raw.info["nchan"] == n_ch
        data = raw.get_data()
        assert data.shape == (n_ch, n_samples)
        assert extra["declared_unit"] == "uV"


# ---------------------------------------------------------------------------
# BIDSImporter
# ---------------------------------------------------------------------------

class TestBIDSImporter:
    """Test BIDS format detection and parsing utilities."""

    def test_detect_bids_root(self, tmp_path):
        from neuroatom.importers.bids import BIDSImporter

        # Not BIDS: no dataset_description.json
        assert not BIDSImporter.detect(tmp_path)

        # Make it BIDS
        (tmp_path / "dataset_description.json").write_text(
            json.dumps({"Name": "Test", "BIDSVersion": "1.7.0"})
        )
        assert BIDSImporter.detect(tmp_path)

    def test_parse_bids_filename(self):
        from neuroatom.importers.bids import _parse_bids_filename

        result = _parse_bids_filename("sub-01_ses-02_task-mi_run-03_eeg")
        assert result["subject"] == "01"
        assert result["session"] == "02"
        assert result["task"] == "mi"
        assert result["run"] == "03"

    def test_parse_bids_filename_no_session(self):
        from neuroatom.importers.bids import _parse_bids_filename

        result = _parse_bids_filename("sub-P01_task-rest_eeg")
        assert result["subject"] == "P01"
        assert result["task"] == "rest"
        assert "session" not in result

    def test_parse_bids_filename_invalid(self):
        from neuroatom.importers.bids import _parse_bids_filename

        assert _parse_bids_filename("random_file.edf") is None

    def test_read_events_tsv(self, tmp_path):
        from neuroatom.importers.bids import _read_events_tsv

        tsv_path = tmp_path / "events.tsv"
        tsv_path.write_text(
            "onset\tduration\ttrial_type\tvalue\n"
            "1.0\t0.5\ttarget\t1\n"
            "3.0\t0.5\tnon_target\t2\n"
            "5.5\t0.5\ttarget\t1\n"
        )

        events = _read_events_tsv(tsv_path, sfreq=256.0)
        assert events is not None
        assert events.shape[0] == 3
        assert events[0, 0] == 256  # 1.0 * 256
        assert events[0, 2] == 1
        assert events[1, 0] == 768  # 3.0 * 256

    def test_read_channels_tsv(self, tmp_path):
        from neuroatom.importers.bids import _read_channels_tsv

        tsv_path = tmp_path / "channels.tsv"
        tsv_path.write_text(
            "name\ttype\tunits\tstatus\n"
            "C3\tEEG\tuV\tgood\n"
            "Cz\tEEG\tuV\tgood\n"
            "EOG1\tEOG\tuV\tgood\n"
        )

        tc = TaskConfig({"dataset_id": "test", "signal_unit": "uV"})
        ch_infos = _read_channels_tsv(tsv_path, 256.0, tc)
        assert len(ch_infos) == 3
        assert ch_infos[0].name == "C3"
        assert ch_infos[0].type == ChannelType.EEG
        assert ch_infos[2].type == ChannelType.EOG

    def test_discover_recordings(self, tmp_path):
        from neuroatom.importers.bids import BIDSImporter
        from neuroatom.storage.pool import Pool

        # Create BIDS structure
        (tmp_path / "dataset_description.json").write_text(
            json.dumps({"Name": "Test"})
        )

        # sub-01/ses-01/eeg/
        eeg_dir = tmp_path / "sub-01" / "ses-01" / "eeg"
        eeg_dir.mkdir(parents=True)
        (eeg_dir / "sub-01_ses-01_task-mi_run-01_eeg.edf").touch()
        (eeg_dir / "sub-01_ses-01_task-mi_run-01_events.tsv").write_text(
            "onset\tduration\ttrial_type\n1.0\t0.5\ttarget\n"
        )

        # sub-02/eeg/ (no session level)
        eeg_dir2 = tmp_path / "sub-02" / "eeg"
        eeg_dir2.mkdir(parents=True)
        (eeg_dir2 / "sub-02_task-mi_run-01_eeg.edf").touch()

        pool = Pool.create(tmp_path / "pool")
        tc = TaskConfig({"dataset_id": "test_bids", "task_type": "mi"})
        importer = BIDSImporter(pool, tc)

        recordings = importer.discover_recordings(tmp_path)
        assert len(recordings) >= 1
        assert recordings[0]["subject"] == "01"


# ---------------------------------------------------------------------------
# EEGLABImporter
# ---------------------------------------------------------------------------

class TestEEGLABImporter:
    def test_detect_set_file(self, tmp_path):
        from neuroatom.importers.eeglab import EEGLABImporter

        set_file = tmp_path / "test.set"
        set_file.touch()
        assert EEGLABImporter.detect(set_file)

        edf_file = tmp_path / "test.edf"
        edf_file.touch()
        assert not EEGLABImporter.detect(edf_file)


# ---------------------------------------------------------------------------
# MOABBBridgeImporter
# ---------------------------------------------------------------------------

class TestMOABBBridge:
    def test_detect_always_false(self, tmp_path):
        """MOABB bridge doesn't detect files."""
        try:
            from neuroatom.importers.moabb_bridge import MOABBBridgeImporter
            assert not MOABBBridgeImporter.detect(tmp_path)
        except ImportError:
            pytest.skip("MOABB not installed")

    def test_load_raw_raises(self, tmp_path):
        """load_raw should raise NotImplementedError."""
        try:
            from neuroatom.importers.moabb_bridge import MOABBBridgeImporter
            from neuroatom.storage.pool import Pool

            pool = Pool.create(tmp_path / "pool")
            tc = TaskConfig({"dataset_id": "test", "task_type": "mi"})
            importer = MOABBBridgeImporter(pool, tc)
            with pytest.raises(NotImplementedError):
                importer.load_raw(tmp_path / "fake.fif")
        except ImportError:
            pytest.skip("MOABB not installed")


# ---------------------------------------------------------------------------
# Task Config files
# ---------------------------------------------------------------------------

class TestTaskConfigs:
    """Test that all task config YAML files are parseable."""

    @pytest.fixture
    def configs_dir(self):
        return Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs"

    def test_bci_comp_iv_2a(self, configs_dir):
        tc = TaskConfig.from_yaml(configs_dir / "bci_comp_iv_2a.yaml")
        assert tc.dataset_id == "bci_comp_iv_2a"
        assert tc.task_type == "motor_imagery"
        assert tc.signal_unit == "uV"  # .mat data is in µV
        assert tc.event_mapping

    def test_p300_speller(self, configs_dir):
        tc = TaskConfig.from_yaml(configs_dir / "p300_speller.yaml")
        assert tc.dataset_id == "p300_speller"
        assert tc.task_type == "p300"
        assert tc.event_mapping

    def test_ssvep_benchmark(self, configs_dir):
        tc = TaskConfig.from_yaml(configs_dir / "ssvep_benchmark.yaml")
        assert tc.dataset_id == "ssvep_benchmark"
        assert tc.task_type == "ssvep"
        assert len(tc.event_mapping) == 40  # 40 frequencies

    def test_inner_speech(self, configs_dir):
        tc = TaskConfig.from_yaml(configs_dir / "inner_speech.yaml")
        assert tc.dataset_id == "inner_speech"
        assert tc.task_type == "inner_speech"
        assert "EXG1" in tc.channel_type_overrides

    def test_lee2019_mi(self, configs_dir):
        tc = TaskConfig.from_yaml(configs_dir / "lee2019_mi.yaml")
        assert tc.dataset_id == "lee2019_mi"
        assert tc.task_type == "motor_imagery"
        assert tc.event_mapping[1] == "left_hand"
