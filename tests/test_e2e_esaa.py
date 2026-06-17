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
    def test_trial_segments_single_file(self):
        import numpy as np
        import scipy.io as sio
        from neuroatom.importers.esaa import _trial_segments
        m = sio.loadmat(str(S1_MAT), squeeze_me=True, struct_as_record=False,
                        variable_names=["Markers"])
        segs = _trial_segments(np.atleast_1d(m["Markers"]))
        assert len(segs) == 32
        assert segs[0] == (1, 2541)        # trail1S → global trial 1
        assert segs[1] == (2, 135521)      # trail2S → global trial 2
        assert [n for n, _ in segs] == list(range(1, 33))

    def test_find_subject_mats_layouts(self):
        from neuroatom.importers.esaa import _find_subject_mats
        assert len(_find_subject_mats(DATA_ROOT / "S1")) == 1   # single file
        assert len(_find_subject_mats(DATA_ROOT / "S4")) == 2   # split S4_1 + S4_2
        assert len(_find_subject_mats(DATA_ROOT / "S8")) == 1   # S8/S8.mat (one level up)
        assert all("__MACOSX" not in p.parts for p in _find_subject_mats(DATA_ROOT / "S1"))

    def test_split_file_global_numbering(self):
        """S4_2 trials must keep their GLOBAL numbers 22..32, not restart at 1."""
        import numpy as np
        import scipy.io as sio
        from neuroatom.importers.esaa import _trial_segments, _find_subject_mats
        files = _find_subject_mats(DATA_ROOT / "S4")
        m2 = sio.loadmat(str(files[1]), squeeze_me=True, struct_as_record=False,
                         variable_names=["Markers"])
        nums = [n for n, _ in _trial_segments(np.atleast_1d(m2["Markers"]))]
        assert nums[0] == 22
        assert max(nums) == 32


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

    def test_split_subject_imported(self, pool, task_config):
        """S4 is split across S4_1/S4_2; both files import with global trial numbers."""
        from neuroatom.importers.esaa import ESAAImporter, _scut_labels

        imp = ESAAImporter(pool=pool, task_config=task_config)
        # max_trials applies per file → trials 1,2 (S4_1) + 22,23 (S4_2)
        results = imp.import_dataset(DATA_ROOT, subjects=["S4"], max_trials=2)
        assert len(results) == 1
        trial_nums = sorted(a.trial_index for a in results[0].atoms)
        assert 22 in trial_nums  # a trial from the SECOND file, globally numbered
        atom22 = next(a for a in results[0].atoms if a.trial_index == 22)
        dir22 = next(x.value for x in atom22.annotations if x.name == "attended_direction")
        assert dir22 == str(_scut_labels(4)[21][0])  # scut_order(S4)[trial 22 - 1]

    def test_s8_alt_nesting_imported(self, pool, task_config):
        """S8 is stored as S8/S8.mat (one level up) and must still import."""
        from neuroatom.importers.esaa import ESAAImporter
        imp = ESAAImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["S8"], max_trials=1)
        assert len(results) == 1
        assert len(results[0].atoms) == 1
        assert results[0].atoms[0].subject_id == "S8"
