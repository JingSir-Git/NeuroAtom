"""End-to-end tests for the CHB-MIT seizure importer.

Requires the dataset at D:\\Data\\CHB-MIT and the optional `mne` dependency.
Skipped automatically otherwise.
"""

import pytest
from pathlib import Path

mne = pytest.importorskip("mne")

DATA_ROOT = Path(r"D:\Data\CHB-MIT")
CHB01 = DATA_ROOT / "chb01"
SUMMARY = CHB01 / "chb01-summary.txt"

pytestmark = pytest.mark.skipif(
    not SUMMARY.exists(),
    reason="CHB-MIT data not available at D:\\Data\\CHB-MIT",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("chb_mit")


class TestCHBMITSummary:
    def test_parse_summary(self):
        from neuroatom.importers.chb_mit import _parse_summary
        s = _parse_summary(SUMMARY)
        assert s["srate"] == 256.0
        assert len(s["channels"]) >= 23
        # chb01_03 has one seizure 2996-3036; chb01_01 has none
        assert s["files"]["chb01_03.edf"] == [(2996.0, 3036.0)]
        assert s["files"]["chb01_01.edf"] == []


class TestCHBMITImporter:
    def test_detect_root(self):
        from neuroatom.importers.chb_mit import CHBMITImporter
        assert CHBMITImporter.detect(DATA_ROOT) is True

    def test_detect_case_dir(self):
        from neuroatom.importers.chb_mit import CHBMITImporter
        assert CHBMITImporter.detect(CHB01) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.chb_mit import CHBMITImporter
        assert CHBMITImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(CHB01 / "chb01_03.edf") == "chb_mit"

    def test_import_seizure_file(self, pool, task_config):
        """chb01_21 has a seizure at 327-420 s; crop to 450 s keeps it cheap."""
        from neuroatom.importers.chb_mit import CHBMITImporter

        imp = CHBMITImporter(pool=pool, task_config=task_config)
        result = imp.import_case(
            CHB01, only_files=["chb01_21.edf"], max_seconds=450,
        )
        assert len(result.atoms) == 1
        atom = result.atoms[0]
        assert atom.dataset_id == "chb_mit"
        assert atom.subject_id == "chb01"
        assert atom.atom_type.value == "continuous_segment"
        assert atom.sampling_rate == 256.0
        assert atom.n_channels == 23

        ann = {a.name: a for a in atom.annotations}
        assert ann["has_seizure"].value == "true"
        assert ann["n_seizures"].numeric_value == 1.0
        assert atom.custom_fields["seizure_intervals"] == [[327.0, 420.0]]

    def test_import_nonseizure_file(self, pool, task_config):
        from neuroatom.importers.chb_mit import CHBMITImporter

        imp = CHBMITImporter(pool=pool, task_config=task_config)
        result = imp.import_case(
            CHB01, only_files=["chb01_01.edf"], max_seconds=60,
        )
        atom = result.atoms[0]
        ann = {a.name: a.value for a in atom.annotations if hasattr(a, "value")}
        assert ann["has_seizure"] == "false"
        assert atom.custom_fields["seizure_intervals"] == []

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.chb_mit import CHBMITImporter

        imp = CHBMITImporter(pool=pool, task_config=task_config)
        result = imp.import_case(
            CHB01, only_files=["chb01_01.edf"], max_seconds=30,
        )
        atom = result.atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape[0] == 23
            assert sig.shape[1] == 30 * 256  # 30 s @ 256 Hz
