"""End-to-end tests for the LEMON resting-state importer.

Requires the dataset at D:\\Data\\lemon and the optional `mne` dependency.
Skipped automatically otherwise.
"""

import pytest
from pathlib import Path

mne = pytest.importorskip("mne")

DATA_ROOT = Path(r"D:\Data\lemon")
PPDIR = DATA_ROOT / "preprocessed" / "EEG_Preprocessed"
EC = PPDIR / "sub-010002_EC.set"
EO = PPDIR / "sub-010002_EO.set"

pytestmark = pytest.mark.skipif(
    not EC.exists(),
    reason="LEMON data not available at D:\\Data\\lemon",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("lemon")


class TestLEMONImporter:
    def test_detect_root(self):
        from neuroatom.importers.lemon import LEMONImporter
        assert LEMONImporter.detect(DATA_ROOT) is True

    def test_detect_preproc_dir(self):
        from neuroatom.importers.lemon import LEMONImporter
        assert LEMONImporter.detect(PPDIR) is True

    def test_detect_file(self):
        from neuroatom.importers.lemon import LEMONImporter
        assert LEMONImporter.detect(EC) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.lemon import LEMONImporter
        assert LEMONImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(EC) == "lemon"

    def test_import_eyes_closed(self, pool, task_config):
        from neuroatom.importers.lemon import LEMONImporter

        imp = LEMONImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(EC, max_seconds=20)
        assert len(result.atoms) == 1
        atom = result.atoms[0]
        assert atom.dataset_id == "lemon"
        assert atom.subject_id == "sub-010002"
        assert atom.run_id == "ec"
        assert atom.atom_type.value == "continuous_segment"
        assert atom.sampling_rate == 250.0
        assert atom.n_channels >= 59
        ann = {a.name: a.value for a in atom.annotations}
        assert ann["condition"] == "eyes_closed"
        assert ann["task"] == "resting_state"

    def test_import_eyes_open(self, pool, task_config):
        from neuroatom.importers.lemon import LEMONImporter
        imp = LEMONImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(EO, max_seconds=20)
        ann = {a.name: a.value for a in result.atoms[0].annotations}
        assert ann["condition"] == "eyes_open"

    def test_import_dataset_one_subject(self, pool, task_config):
        from neuroatom.importers.lemon import LEMONImporter
        imp = LEMONImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["sub-010002"], max_seconds=15,
        )
        # EC + EO for the one subject
        conds = {
            next(x.value for x in r.atoms[0].annotations if x.name == "condition")
            for r in results
        }
        assert conds == {"eyes_closed", "eyes_open"}

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.lemon import LEMONImporter
        imp = LEMONImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(EC, max_seconds=10)
        atom = result.atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape[1] == 10 * 250  # 10 s @ 250 Hz

    def test_e0_filename_typo_handled(self, pool, task_config):
        """sub-010229_E0.set (zero, a typo for _EO) must import as eyes_open, not be skipped."""
        e0 = PPDIR / "sub-010229_E0.set"
        if not e0.exists():
            pytest.skip("E0-typo file not present")
        from neuroatom.importers.lemon import LEMONImporter
        imp = LEMONImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(e0, max_seconds=10)
        ann = {a.name: a.value for a in result.atoms[0].annotations}
        assert ann["condition"] == "eyes_open"
