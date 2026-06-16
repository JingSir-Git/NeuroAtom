"""End-to-end tests for the Ear-SAAD importer.

Requires the actual dataset at D:\\Data\\Ear-SAAD.
Skipped automatically if the data is not available.
"""

import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\Ear-SAAD")
PREPROC = DATA_ROOT / "preprocessedData"
SUB1 = PREPROC / "dataSubject1.mat"

pytestmark = pytest.mark.skipif(
    not SUB1.exists(),
    reason="Ear-SAAD data not available at D:\\Data\\Ear-SAAD",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("ear_saad")


class TestEarSAADParsing:
    def test_parse_subject_mat(self):
        from neuroatom.importers.ear_saad import _parse_subject_mat

        parsed = _parse_subject_mat(SUB1)
        assert parsed["fs"] == 20.0
        assert len(parsed["ch_names"]) == 61
        # in-ear electrodes present
        assert any(n.upper().startswith("CER") for n in parsed["ch_names"])

        trials = parsed["trials"]
        assert len(trials) == 6
        t0 = trials[0]
        assert t0["attended_ear"] in ("1", "2")
        assert t0["video_condition"] in ("video", "no_video")
        assert "envelope_speaker1" in t0["companions"]
        assert "envelope_speaker2" in t0["companions"]
        assert "acoustic_edges_speaker1" in t0["companions"]


class TestEarSAADImporter:
    def test_detect_dir(self):
        from neuroatom.importers.ear_saad import EarSAADImporter
        assert EarSAADImporter.detect(PREPROC) is True
        # also accept the dataset root (holds preprocessedData/)
        assert EarSAADImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.ear_saad import EarSAADImporter
        assert EarSAADImporter.detect(SUB1) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.ear_saad import EarSAADImporter
        assert EarSAADImporter.detect(tmp_path) is False

    def test_registry_detects_ear_saad(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(SUB1) == "ear_saad"

    def test_import_single_subject(self, pool, task_config):
        from neuroatom.importers.ear_saad import EarSAADImporter

        imp = EarSAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["sub-01"], max_trials=3)

        assert len(results) == 1
        result = results[0]
        assert len(result.atoms) == 3

        atom = result.atoms[0]
        assert atom.dataset_id == "ear_saad"
        assert atom.subject_id == "sub-01"
        assert atom.atom_type.value == "trial"
        assert atom.sampling_rate == 20.0
        assert atom.n_channels == 61

        ann_names = {a.name for a in atom.annotations}
        assert "attended_ear" in ann_names
        assert "attended_speaker" in ann_names
        assert "video_condition" in ann_names

    def test_eareeg_channels_typed(self, pool, task_config):
        from neuroatom.importers.ear_saad import EarSAADImporter

        imp = EarSAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["sub-01"], max_trials=1)
        ch_infos = results[0].channel_infos

        eareeg = [c for c in ch_infos if c.custom_fields.get("is_eareeg")]
        assert len(eareeg) >= 6  # cER5-10
        assert all(c.type.value == "other" for c in eareeg)

    def test_envelope_companions(self, pool, task_config):
        from neuroatom.importers.ear_saad import EarSAADImporter

        imp = EarSAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["sub-01"], max_trials=1)
        atom = results[0].atoms[0]

        cont = {a.name for a in atom.annotations if a.annotation_type == "continuous"}
        assert "envelope_speaker1" in cont
        assert "envelope_speaker2" in cont
        env = next(a for a in atom.annotations if a.name == "envelope_speaker1")
        assert env.data_sampling_rate == 20.0
        assert env.data_ref.shape[0] > 100

    def test_envelope_stored_in_hdf5(self, pool, task_config):
        import h5py
        from neuroatom.importers.ear_saad import EarSAADImporter

        imp = EarSAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["sub-01"], max_trials=1)
        atom = results[0].atoms[0]

        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            keys = set(f[f"/atoms/{atom.atom_id}/annotations"].keys())
            assert "envelope_speaker1" in keys
            assert "envelope_speaker2" in keys
