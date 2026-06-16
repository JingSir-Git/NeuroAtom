"""End-to-end tests for the BASE-AAD-USTC importer.

Requires the actual dataset at D:\\Data\\BASE-AAD-USTC and `openpyxl`.
Skipped automatically otherwise. Trials are 120 s @ 1000 Hz, so tests crop
to a few seconds (max_seconds) and import only a couple of trials.
"""

import pytest
from pathlib import Path

pytest.importorskip("openpyxl")

DATA_ROOT = Path(r"D:\Data\BASE-AAD-USTC")
S1_CDT = DATA_ROOT / "EEG" / "s1.cdt"

pytestmark = pytest.mark.skipif(
    not S1_CDT.exists(),
    reason="BASE-AAD-USTC data not available at D:\\Data\\BASE-AAD-USTC",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("base_aad_ustc")


class TestBASEParsing:
    def test_parse_dpa(self):
        from neuroatom.importers.base_aad_ustc import _parse_dpa
        dpa = _parse_dpa(Path(str(S1_CDT) + ".dpa"))
        assert dpa["n_channels"] == 69
        assert dpa["sfreq"] == 1000.0
        assert dpa["unit"].lower() == "uv"
        assert len(dpa["labels"]) == 69
        assert "FP1" in dpa["labels"]
        assert "Trigger" in dpa["labels"]

    def test_parse_group_xlsx(self):
        from neuroatom.importers.base_aad_ustc import _parse_group_xlsx
        m = _parse_group_xlsx(DATA_ROOT / "group1.xlsx")
        assert "s1" in m
        assert len(m["s1"]) == 20
        assert m["s1"][1]["attended"] == "L"
        assert m["s1"][2]["attended"] == "R"

    def test_find_trial_onsets(self):
        import numpy as np
        from neuroatom.importers.base_aad_ustc import _parse_dpa, _read_cdt, _find_trial_onsets
        dpa = _parse_dpa(Path(str(S1_CDT) + ".dpa"))
        data = _read_cdt(S1_CDT, dpa["n_channels"])
        trig_idx = dpa["labels"].index("Trigger")
        onsets = _find_trial_onsets(np.asarray(data[:, trig_idx]))
        assert len(onsets) == 20
        assert set(onsets) == set(range(1, 21))  # trial numbers 1..20
        # onsets are increasing in chronological trial order
        assert onsets[1] < onsets[2] < onsets[20]
        # a 120 s trial fits within the recording
        assert onsets[1] + 120 * dpa["sfreq"] <= data.shape[0]


class TestBASEImporter:
    def test_detect_root(self):
        from neuroatom.importers.base_aad_ustc import BASEAADUSTCImporter
        assert BASEAADUSTCImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.base_aad_ustc import BASEAADUSTCImporter
        assert BASEAADUSTCImporter.detect(S1_CDT) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.base_aad_ustc import BASEAADUSTCImporter
        assert BASEAADUSTCImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(S1_CDT) == "base_aad_ustc"

    def test_import_single_subject(self, pool, task_config):
        from neuroatom.importers.base_aad_ustc import BASEAADUSTCImporter

        imp = BASEAADUSTCImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["s1"], max_trials=2, max_seconds=10,
        )
        assert len(results) == 1
        result = results[0]
        assert len(result.atoms) == 2

        atom = result.atoms[0]
        assert atom.dataset_id == "base_aad_ustc"
        assert atom.subject_id == "s1"
        assert atom.atom_type.value == "trial"
        assert atom.sampling_rate == 1000.0
        assert atom.signal_unit == "uV"
        # 64 EEG + HEO/VEO/EKG/EMG, Trigger excluded
        assert atom.n_channels == 68

        ann = {a.name: a for a in atom.annotations}
        assert "attended_ear" in ann
        assert "trial_number" in ann
        assert "stimulus_file" in ann

    def test_attended_ear_from_xlsx(self, pool, task_config):
        """s1 trial 1 attended L → left; trial 2 → right."""
        from neuroatom.importers.base_aad_ustc import BASEAADUSTCImporter

        imp = BASEAADUSTCImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["s1"], max_trials=2, max_seconds=8,
        )
        by_trial = {
            int(next(x.numeric_value for x in a.annotations if x.name == "trial_number")):
            next(x.value for x in a.annotations if x.name == "attended_ear")
            for a in results[0].atoms
        }
        assert by_trial[1] == "left"
        assert by_trial[2] == "right"

    def test_trigger_excluded(self, pool, task_config):
        from neuroatom.importers.base_aad_ustc import BASEAADUSTCImporter

        imp = BASEAADUSTCImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["s1"], max_trials=1, max_seconds=8,
        )
        names = {c.name for c in results[0].channel_infos}
        assert "Trigger" not in names
        assert "HEO" in names and "VEO" in names

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.base_aad_ustc import BASEAADUSTCImporter

        imp = BASEAADUSTCImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["s1"], max_trials=1, max_seconds=8,
        )
        atom = results[0].atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape[0] == 68
            assert sig.shape[1] > 1000
