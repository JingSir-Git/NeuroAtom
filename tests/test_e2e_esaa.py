"""End-to-end tests for the ESAA (SCUT spatial AAD) importer.

Requires the actual dataset at D:\\Data\\ESAA. Skipped automatically otherwise.
The per-subject .mat is large (~526 MB), so the import test uses few trials.
"""

import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\ESAA")
S1_MAT = DATA_ROOT / "S1" / "S1" / "S1.mat"

pytestmark = pytest.mark.skipif(
    not S1_MAT.exists(),
    reason="ESAA data not available at D:\\Data\\ESAA",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("esaa")


class TestESAALabels:
    """Pure label-protocol logic (no data load)."""

    def test_labels_le8_identity(self):
        from neuroatom.importers.esaa import _scut_labels, _SCUT_LABEL
        assert _scut_labels(1) == [list(r) for r in _SCUT_LABEL]
        assert _scut_labels(8) == [list(r) for r in _SCUT_LABEL]

    def test_labels_gt8_reorder_and_reverse(self):
        from neuroatom.importers.esaa import _scut_labels, _SCUT_LABEL, _SCUT_SUF_ORDER
        out = _scut_labels(11)
        # reordered then bit-flipped relative to the base table
        expected = [[1 - v for v in _SCUT_LABEL[i]] for i in _SCUT_SUF_ORDER]
        assert out == expected
        assert out != [list(r) for r in _SCUT_LABEL]


class TestESAAMarkers:
    def test_trial_onsets(self):
        import numpy as np
        import scipy.io as sio
        from neuroatom.importers.esaa import _trial_onsets
        m = sio.loadmat(str(S1_MAT), squeeze_me=True, struct_as_record=False,
                        variable_names=["Markers"])
        onsets = _trial_onsets(np.atleast_1d(m["Markers"]))
        assert len(onsets) == 32
        assert onsets[0] == 2541          # trail1S position
        assert onsets[1] == 135521        # trail2S position
        assert onsets == sorted(onsets)


class TestESAAImporter:
    def test_detect_root(self):
        from neuroatom.importers.esaa import ESAAImporter
        assert ESAAImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.esaa import ESAAImporter
        assert ESAAImporter.detect(S1_MAT) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.esaa import ESAAImporter
        assert ESAAImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(S1_MAT) == "esaa"

    def test_import_single_subject(self, pool, task_config):
        from neuroatom.importers.esaa import ESAAImporter

        imp = ESAAImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["S1"], max_trials=2)

        assert len(results) == 1
        result = results[0]
        assert len(result.atoms) == 2

        atom = result.atoms[0]
        assert atom.dataset_id == "esaa"
        assert atom.subject_id == "S1"
        assert atom.atom_type.value == "trial"
        assert atom.sampling_rate == 1000.0
        assert atom.n_channels == 65  # 64 EEG + ECG
        # 55 s window @ 1000 Hz
        assert atom.temporal.duration_samples == 55000

        ann = {a.name: a for a in atom.annotations}
        assert ann["attended_direction"].value == "0"  # S1 trial 1 = [0, 0]
        assert ann["attended_speaker"].value == "0"
        # S1 trial 1 is a flagged poor-answer trial
        assert "excluded" in ann

    def test_second_trial_label(self, pool, task_config):
        from neuroatom.importers.esaa import ESAAImporter

        imp = ESAAImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["S1"], max_trials=2)
        atom2 = results[0].atoms[1]
        ann = {a.name: a.value for a in atom2.annotations if hasattr(a, "value")}
        assert ann["attended_direction"] == "1"  # S1 trial 2 = [1, 0]
        assert ann["attended_speaker"] == "0"

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.esaa import ESAAImporter

        imp = ESAAImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["S1"], max_trials=1)
        atom = results[0].atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape == (65, 55000)
