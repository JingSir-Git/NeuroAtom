"""Tests for contributor toolchain: scaffold, generic importer, import validator."""

import csv
from pathlib import Path

import numpy as np
import pytest

from neuroatom.contrib.scaffold import scaffold_importer
from neuroatom.contrib.validate_import import validate_import
from neuroatom.importers.generic import GenericImporter
from neuroatom.importers.base import TaskConfig
from neuroatom.storage.pool import Pool


# ---------------------------------------------------------------------------
# Scaffold tests
# ---------------------------------------------------------------------------

class TestScaffold:

    def test_scaffold_creates_three_files(self, tmp_path):
        result = scaffold_importer("my_dataset", tmp_path)
        assert "importer" in result
        assert "task_config" in result
        assert "test" in result
        assert (tmp_path / "neuroatom" / "importers" / "my_dataset.py").exists()
        assert (tmp_path / "neuroatom" / "importers" / "task_configs" / "my_dataset.yaml").exists()
        assert (tmp_path / "tests" / "test_e2e_my_dataset.py").exists()

    def test_scaffold_does_not_overwrite(self, tmp_path):
        scaffold_importer("my_dataset", tmp_path)
        # Read the importer file content
        importer_path = tmp_path / "neuroatom" / "importers" / "my_dataset.py"
        original = importer_path.read_text()

        # Scaffold again — should not overwrite
        scaffold_importer("my_dataset", tmp_path)
        assert importer_path.read_text() == original

    def test_scaffold_importer_has_class(self, tmp_path):
        scaffold_importer("test_eeg", tmp_path)
        content = (tmp_path / "neuroatom" / "importers" / "test_eeg.py").read_text()
        assert "class TestEegImporter" in content
        assert "register_importer" in content

    def test_scaffold_config_has_dataset_id(self, tmp_path):
        scaffold_importer("test_eeg", tmp_path)
        content = (tmp_path / "neuroatom" / "importers" / "task_configs" / "test_eeg.yaml").read_text()
        assert 'dataset_id: "test_eeg"' in content


# ---------------------------------------------------------------------------
# Generic importer tests
# ---------------------------------------------------------------------------

class TestGenericImporter:

    @pytest.fixture
    def pool(self, tmp_path):
        return Pool.create(tmp_path / "pool")

    @pytest.fixture
    def npy_data_dir(self, tmp_path):
        """Create a data directory with .npy files for 2 subjects."""
        data_dir = tmp_path / "data"
        rng = np.random.default_rng(42)

        for sid in ["S01", "S02"]:
            subj_dir = data_dir / sid
            subj_dir.mkdir(parents=True)
            signal = rng.standard_normal((4, 512)).astype(np.float32)
            np.save(subj_dir / "signal.npy", signal)

            # Channel names
            (subj_dir / "channels.txt").write_text("Fp1\nFp2\nC3\nC4\n")

        return data_dir

    @pytest.fixture
    def csv_data_dir(self, tmp_path):
        """Create a data directory with CSV files."""
        data_dir = tmp_path / "csv_data"
        data_dir.mkdir()
        rng = np.random.default_rng(42)

        for sid in ["S01", "S02"]:
            signal = rng.standard_normal((256, 4)).astype(np.float32)
            csv_path = data_dir / f"{sid}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Fp1", "Fp2", "C3", "C4"])
                for row in signal:
                    writer.writerow(row.tolist())

        return data_dir

    @pytest.fixture
    def labeled_data_dir(self, tmp_path):
        """Create data with labels."""
        data_dir = tmp_path / "labeled"
        subj_dir = data_dir / "S01"
        subj_dir.mkdir(parents=True)

        rng = np.random.default_rng(42)
        signal = rng.standard_normal((4, 1024)).astype(np.float32)
        np.save(subj_dir / "signal.npy", signal)

        with open(subj_dir / "labels.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["onset_sample", "duration_samples", "label"])
            writer.writeheader()
            writer.writerow({"onset_sample": 0, "duration_samples": 128, "label": "left"})
            writer.writerow({"onset_sample": 256, "duration_samples": 128, "label": "right"})
            writer.writerow({"onset_sample": 512, "duration_samples": 128, "label": "left"})

        return data_dir

    def test_import_npy(self, pool, npy_data_dir):
        config = TaskConfig({
            "dataset_id": "test_npy", "signal_unit": "uV",
            "custom": {"sampling_rate": 128.0},
        })
        importer = GenericImporter(pool=pool, task_config=config)
        results = importer.import_dataset(
            npy_data_dir, dataset_id="test_npy", sampling_rate=128.0,
        )
        assert len(results) == 2
        assert all(r.n_atoms > 0 for r in results)

    def test_import_csv(self, pool, csv_data_dir):
        config = TaskConfig({
            "dataset_id": "test_csv", "signal_unit": "uV",
            "custom": {"sampling_rate": 128.0},
        })
        importer = GenericImporter(pool=pool, task_config=config)
        results = importer.import_dataset(
            csv_data_dir, dataset_id="test_csv", sampling_rate=128.0,
        )
        assert len(results) == 2

    def test_import_with_labels(self, pool, labeled_data_dir):
        config = TaskConfig({
            "dataset_id": "test_labeled", "signal_unit": "uV",
            "custom": {"sampling_rate": 128.0},
        })
        importer = GenericImporter(pool=pool, task_config=config)
        results = importer.import_dataset(
            labeled_data_dir, dataset_id="test_labeled", sampling_rate=128.0,
        )
        assert len(results) == 1
        assert results[0].n_atoms == 3  # 3 labeled epochs

    def test_import_with_epoch_split(self, pool, npy_data_dir):
        config = TaskConfig({
            "dataset_id": "test_epoch", "signal_unit": "uV",
            "custom": {"sampling_rate": 128.0},
        })
        importer = GenericImporter(pool=pool, task_config=config)
        results = importer.import_dataset(
            npy_data_dir, dataset_id="test_epoch",
            sampling_rate=128.0, epoch_seconds=1.0,
        )
        # 512 samples / 128 Hz / 1s = 4 epochs per subject
        assert len(results) == 2
        assert results[0].n_atoms == 4

    def test_signal_unit_is_uv(self, pool, npy_data_dir):
        config = TaskConfig({
            "dataset_id": "test_unit", "signal_unit": "uV",
            "custom": {"sampling_rate": 128.0},
        })
        importer = GenericImporter(pool=pool, task_config=config)
        results = importer.import_dataset(
            npy_data_dir, dataset_id="test_unit", sampling_rate=128.0,
        )
        for r in results:
            for atom in r.atoms:
                assert atom.signal_unit == "uV"

    def test_channel_names_from_file(self, pool, npy_data_dir):
        config = TaskConfig({
            "dataset_id": "test_ch", "signal_unit": "uV",
            "custom": {"sampling_rate": 128.0},
        })
        importer = GenericImporter(pool=pool, task_config=config)
        results = importer.import_dataset(
            npy_data_dir, dataset_id="test_ch", sampling_rate=128.0,
        )
        atom = results[0].atoms[0]
        assert atom.channel_ids == ["Fp1", "Fp2", "C3", "C4"]


# ---------------------------------------------------------------------------
# Import validator tests
# ---------------------------------------------------------------------------

class TestValidateImport:

    def test_valid_import_passes(self, tmp_path):
        pool = Pool.create(tmp_path / "pool")
        data_dir = tmp_path / "data" / "S01"
        data_dir.mkdir(parents=True)
        rng = np.random.default_rng(42)
        np.save(data_dir / "signal.npy", rng.standard_normal((4, 256)).astype(np.float32))

        config = TaskConfig({
            "dataset_id": "valid_ds", "signal_unit": "uV",
            "custom": {"sampling_rate": 128.0},
        })
        importer = GenericImporter(pool=pool, task_config=config)
        importer.import_dataset(
            tmp_path / "data", dataset_id="valid_ds", sampling_rate=128.0,
        )

        report = validate_import(pool, "valid_ds")
        assert report.is_valid
        assert report.n_atoms_checked > 0

    def test_missing_dataset_fails(self, tmp_path):
        pool = Pool.create(tmp_path / "pool")
        report = validate_import(pool, "nonexistent")
        assert not report.is_valid

    def test_validate_with_signals(self, tmp_path):
        pool = Pool.create(tmp_path / "pool")
        data_dir = tmp_path / "data" / "S01"
        data_dir.mkdir(parents=True)
        rng = np.random.default_rng(42)
        np.save(data_dir / "signal.npy", rng.standard_normal((4, 256)).astype(np.float32))

        config = TaskConfig({
            "dataset_id": "sig_check", "signal_unit": "uV",
            "custom": {"sampling_rate": 128.0},
        })
        importer = GenericImporter(pool=pool, task_config=config)
        importer.import_dataset(
            tmp_path / "data", dataset_id="sig_check", sampling_rate=128.0,
        )

        report = validate_import(pool, "sig_check", check_signals=True)
        assert report.is_valid
