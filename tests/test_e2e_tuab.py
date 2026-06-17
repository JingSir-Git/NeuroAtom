"""End-to-end tests for the TUAB (TUH Abnormal EEG) importer.

Requires the dataset at D:\\Data\\tuab and the optional `mne` dependency.
Skipped automatically otherwise.
"""

import pytest
from pathlib import Path

mne = pytest.importorskip("mne")

DATA_ROOT = Path(r"D:\Data\tuab")
EVAL_ABN = DATA_ROOT / "edf" / "eval" / "abnormal" / "01_tcp_ar"
EVAL_NORM = DATA_ROOT / "edf" / "eval" / "normal" / "01_tcp_ar"

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "edf").is_dir(),
    reason="TUAB data not available at D:\\Data\\tuab",
)


def _first_edf(d: Path):
    files = sorted(d.glob("*.edf")) if d.is_dir() else []
    return files[0] if files else None


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("tuab")


class TestTUABDetect:
    def test_detect_root(self):
        from neuroatom.importers.tuab import TUABImporter
        assert TUABImporter.detect(DATA_ROOT) is True

    def test_detect_file(self):
        from neuroatom.importers.tuab import TUABImporter
        edf = _first_edf(EVAL_ABN)
        assert edf is not None
        assert TUABImporter.detect(edf) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.tuab import TUABImporter
        assert TUABImporter.detect(tmp_path) is False

    def test_registry_detects(self):
        from neuroatom.importers.registry import detect_format
        edf = _first_edf(EVAL_ABN)
        assert detect_format(edf) == "tuab"


class TestTUABImporter:
    def test_import_abnormal(self, pool, task_config):
        from neuroatom.importers.tuab import TUABImporter
        edf = _first_edf(EVAL_ABN)
        imp = TUABImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(edf, label="abnormal", split="eval", max_seconds=30)
        atom = result.atoms[0]
        assert atom.dataset_id == "tuab"
        assert atom.atom_type.value == "continuous_segment"
        assert atom.n_channels > 0
        ann = {a.name: a.value for a in atom.annotations}
        assert ann["label"] == "abnormal"
        assert ann["split"] == "eval"

    def test_import_dataset_eval_both_labels(self, pool, task_config):
        from neuroatom.importers.tuab import TUABImporter
        imp = TUABImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT, splits=["eval"], max_files=1, max_seconds=20,
        )
        labels = {
            next(x.value for x in r.atoms[0].annotations if x.name == "label")
            for r in results
        }
        assert labels == {"normal", "abnormal"}

    def test_channel_typing(self, pool, task_config):
        from neuroatom.importers.tuab import TUABImporter
        edf = _first_edf(EVAL_ABN)
        imp = TUABImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(edf, label="abnormal", split="eval", max_seconds=10)
        types = {c.name: c.type.value for c in result.channel_infos}
        # EKG channel → ecg if present
        ekg = [t for n, t in types.items() if "EKG" in n.upper()]
        assert all(t == "ecg" for t in ekg)

    def test_signal_written(self, pool, task_config):
        import h5py
        from neuroatom.importers.tuab import TUABImporter
        edf = _first_edf(EVAL_NORM)
        imp = TUABImporter(pool=pool, task_config=task_config)
        result = imp.import_recording(edf, label="normal", split="eval", max_seconds=10)
        atom = result.atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        with h5py.File(str(shard_path), "r") as f:
            sig = f[f"/atoms/{atom.atom_id}/signal"][:]
            assert sig.shape[0] == atom.n_channels
            assert sig.shape[1] > 0
