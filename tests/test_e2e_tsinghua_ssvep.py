"""End-to-end tests for the Tsinghua Benchmark SSVEP importer.

Requires the actual dataset at D:\\Data\\tsinghua_ssvep. Skipped otherwise.
Per-subject .mat is ~368 MB, so the import test uses few targets/blocks.
"""

import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\tsinghua_ssvep")
S1_MAT = DATA_ROOT / "S1.mat"

pytestmark = pytest.mark.skipif(
    not S1_MAT.exists(),
    reason="Tsinghua SSVEP data not available at D:\\Data\\tsinghua_ssvep",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("tsinghua_ssvep")


class TestTsinghuaParsing:
    def test_parse_loc(self):
        from neuroatom.importers.tsinghua_ssvep import _parse_loc
        names = _parse_loc(DATA_ROOT / "64-channels.loc")
        assert len(names) == 64
        assert names[0] == "FP1"
        assert names[-1] == "CB2"

    def test_load_freq_phase(self):
        from neuroatom.importers.tsinghua_ssvep import _load_freq_phase
        freqs, phases = _load_freq_phase(DATA_ROOT / "Freq_Phase.mat")
        assert freqs is not None and len(freqs) == 40
        assert phases is not None and len(phases) == 40
        # frequencies span the documented 8–15.8 Hz range
        assert 8.0 <= min(freqs) <= max(freqs) <= 15.8

    def test_parse_sub_info(self):
        from neuroatom.importers.tsinghua_ssvep import _parse_sub_info
        info = _parse_sub_info(DATA_ROOT / "Sub_info.txt")
        assert info["S01"]["group"] == "experienced"
        assert info["S01"]["age"] == "31"
        assert info["S09"]["group"] == "naive"


class TestTsinghuaImporter:
    def test_detect_root(self):
        from neuroatom.importers.tsinghua_ssvep import TsinghuaSSVEPImporter
        assert TsinghuaSSVEPImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.tsinghua_ssvep import TsinghuaSSVEPImporter
        assert TsinghuaSSVEPImporter.detect(S1_MAT) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.tsinghua_ssvep import TsinghuaSSVEPImporter
        assert TsinghuaSSVEPImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        assert detect_format(S1_MAT) == "tsinghua_ssvep"

    def test_import_single_subject(self, pool, task_config):
        from neuroatom.importers.tsinghua_ssvep import TsinghuaSSVEPImporter

        imp = TsinghuaSSVEPImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["S01"], max_targets=2, max_blocks=2,
        )
        assert len(results) == 1
        result = results[0]
        assert len(result.atoms) == 4  # 2 targets x 2 blocks

        atom = result.atoms[0]
        assert atom.dataset_id == "tsinghua_ssvep"
        assert atom.subject_id == "S01"
        assert atom.atom_type.value == "trial"
        assert atom.sampling_rate == 250.0
        assert atom.n_channels == 64
        assert atom.temporal.duration_samples == 1500

        ann = {a.name for a in atom.annotations}
        assert "target_index" in ann
        assert "ssvep_frequency" in ann
        assert "block" in ann

    def test_frequency_matches_target(self, pool, task_config):
        """Each trial's ssvep_frequency matches Freq_Phase for its target index."""
        from neuroatom.importers.tsinghua_ssvep import (
            TsinghuaSSVEPImporter, _load_freq_phase,
        )
        freqs, _ = _load_freq_phase(DATA_ROOT / "Freq_Phase.mat")

        imp = TsinghuaSSVEPImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["S01"], max_targets=2, max_blocks=1,
        )
        for atom in results[0].atoms:
            t = int(next(a.value for a in atom.annotations if a.name == "target_index"))
            freq = next(a.numeric_value for a in atom.annotations if a.name == "ssvep_frequency")
            assert abs(freq - float(freqs[t])) < 1e-6

    def test_subject_demographics(self, pool, task_config):
        from neuroatom.importers.tsinghua_ssvep import TsinghuaSSVEPImporter

        imp = TsinghuaSSVEPImporter(pool=pool, task_config=task_config)
        imp.import_dataset(DATA_ROOT, subjects=["S01"], max_targets=1, max_blocks=1)
        meta = pool.get_subject_meta("tsinghua_ssvep", "S01")
        assert meta.age == 31
        assert meta.custom_fields.get("group") == "experienced"

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.tsinghua_ssvep import TsinghuaSSVEPImporter

        imp = TsinghuaSSVEPImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, subjects=["S01"], max_targets=1, max_blocks=1,
        )
        atom = results[0].atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape == (64, 1500)
