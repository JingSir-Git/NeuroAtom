"""End-to-end tests for the AV-GC-AAD importer.

Requires the actual dataset at D:\\Data\\AV-GC-AAD.
Skipped automatically if the data is not available.
"""

import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\AV-GC-AAD")
SUB01 = DATA_ROOT / "2024-AV-GC-AAD-sub01_preprocessed.mat"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="AV-GC-AAD data not available at D:\\Data\\AV-GC-AAD",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("avgc_aad")


class TestAVGCAADParsing:
    """Cheap parse-level tests (load the .mat, no pool writes)."""

    def test_parse_subject_mat(self):
        from neuroatom.importers.avgc_aad import _parse_subject_mat

        parsed = _parse_subject_mat(SUB01)
        assert parsed["fs"] == 128.0
        assert len(parsed["ch_names"]) == 68
        # 64 scalp + 4 external EXG
        assert "Fp1" in parsed["ch_names"]
        assert any(n.upper().startswith("EXG") for n in parsed["ch_names"])

        trials = parsed["trials"]
        assert len(trials) == 6

        t0 = trials[0]
        assert t0["init_attention"] in ("left", "right")
        assert t0["visual_type"] in ("no_visuals", "fixed_video", "moving_video")
        # Mid-trial attention switch is the defining feature.
        assert len(t0["switch_times"]) >= 1
        # All four precomputed envelopes present.
        assert set(t0["envelopes"]) == {
            "attended_envelope", "unattended_envelope",
            "left_envelope", "right_envelope",
        }


class TestAVGCAADImporter:
    """Full integration tests against the real dataset."""

    def test_detect_dir(self):
        from neuroatom.importers.avgc_aad import AVGCAADImporter
        assert AVGCAADImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.avgc_aad import AVGCAADImporter
        assert AVGCAADImporter.detect(SUB01) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.avgc_aad import AVGCAADImporter
        assert AVGCAADImporter.detect(tmp_path) is False

    def test_registry_detects_avgc(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(SUB01) == "avgc_aad"

    def test_import_single_subject(self, pool, task_config):
        from neuroatom.importers.avgc_aad import AVGCAADImporter

        imp = AVGCAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["sub01"], max_trials=2,
        )

        assert len(results) == 1
        result = results[0]
        assert len(result.atoms) == 2

        atom = result.atoms[0]
        assert atom.dataset_id == "avgc_aad"
        assert atom.subject_id == "sub01"
        assert atom.atom_type.value == "trial"
        assert atom.sampling_rate == 128.0
        assert atom.n_channels == 68
        assert atom.signal_unit == "uV"

        ann_names = {a.name for a in atom.annotations}
        assert "init_attention" in ann_names
        assert "visual_condition" in ann_names
        assert "visual_type" in ann_names
        assert "dynamic_attention" in ann_names

    def test_dynamic_attention_switch(self, pool, task_config):
        """The within-trial attention switch is captured."""
        from neuroatom.importers.avgc_aad import AVGCAADImporter

        imp = AVGCAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["sub01"], max_trials=1)
        atom = results[0].atoms[0]

        dyn = next(a for a in atom.annotations if a.name == "dynamic_attention")
        assert dyn.value == "true"

        switch = next(
            (a for a in atom.annotations if a.name == "switch_time_seconds"), None,
        )
        assert switch is not None
        assert switch.numeric_value > 0

    def test_envelope_annotations(self, pool, task_config):
        """All four speech envelopes are present as ContinuousAnnotation."""
        from neuroatom.importers.avgc_aad import AVGCAADImporter

        imp = AVGCAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["sub01"], max_trials=1)
        atom = results[0].atoms[0]

        cont = {a.name for a in atom.annotations if a.annotation_type == "continuous"}
        assert cont == {
            "attended_envelope", "unattended_envelope",
            "left_envelope", "right_envelope",
        }
        env = next(a for a in atom.annotations if a.name == "attended_envelope")
        assert env.domain == "stimulus"
        assert env.data_sampling_rate == 128.0
        assert env.data_ref.shape[0] > 1000

    def test_envelope_stored_in_hdf5(self, pool, task_config):
        import h5py
        from neuroatom.importers.avgc_aad import AVGCAADImporter

        imp = AVGCAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["sub01"], max_trials=1)
        atom = results[0].atoms[0]

        shard_path = pool.root / atom.signal_ref.file_path
        assert shard_path.exists()
        with h5py.File(str(shard_path), "r") as f:
            ann_grp = f[f"/atoms/{atom.atom_id}/annotations"]
            keys = set(ann_grp.keys())
            assert "attended_envelope" in keys
            assert ann_grp["attended_envelope"][:].ndim == 1

    def test_exg_channels_typed_eog(self, pool, task_config):
        from neuroatom.importers.avgc_aad import AVGCAADImporter

        imp = AVGCAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(DATA_ROOT, subjects=["sub01"], max_trials=1)
        ch_infos = results[0].channel_infos

        exg = [c for c in ch_infos if c.name.upper().startswith("EXG")]
        assert len(exg) == 4
        assert all(c.type.value == "eog" for c in exg)
