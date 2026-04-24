"""End-to-end tests for the OpenBMI importer (Lee et al., GigaScience 2019).

Requires the OpenBMI dataset at r'\\wsqlab\\share\\JCH\\OpenBMI'
with MI/, ERP/, and SSVEP/ subdirectories.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuroatom.importers.base import TaskConfig
from neuroatom.importers.openbmi import (
    OpenBMIImporter,
    _detect_openbmi,
    _parse_filename,
)
from neuroatom.storage.pool import Pool

# ── Data paths ────────────────────────────────────────────────────────────────
OPENBMI_ROOT = Path(r"\\wsqlab\share\JCH\OpenBMI")
MI_DIR = OPENBMI_ROOT / "MI"
ERP_DIR = OPENBMI_ROOT / "ERP"
SSVEP_DIR = OPENBMI_ROOT / "SSVEP"

_HAS_DATA = OPENBMI_ROOT.exists() and MI_DIR.exists()
skip_no_data = pytest.mark.skipif(
    not _HAS_DATA, reason="OpenBMI data not available"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests (no data needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilenameParser:
    """Test OpenBMI filename parsing."""

    def test_mi_sess01(self):
        assert _parse_filename("sess01_subj01_EEG_MI.mat") == (1, 1, "MI")

    def test_ssvep_sess02(self):
        assert _parse_filename("sess02_subj54_EEG_SSVEP.mat") == (2, 54, "SSVEP")

    def test_erp_middle(self):
        assert _parse_filename("sess01_subj27_EEG_ERP.mat") == (1, 27, "ERP")

    def test_invalid(self):
        assert _parse_filename("A01T.mat") is None

    def test_wrong_ext(self):
        assert _parse_filename("sess01_subj01_EEG_MI.csv") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Detection tests
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_data
class TestDetection:
    """Test OpenBMI format detection."""

    def test_detect_root_dir(self):
        assert _detect_openbmi(OPENBMI_ROOT) is True

    def test_detect_paradigm_dir(self):
        assert _detect_openbmi(MI_DIR) is True

    def test_detect_single_file(self):
        f = next(MI_DIR.glob("sess01_subj01_EEG_MI.mat"))
        assert _detect_openbmi(f) is True

    def test_detect_class_method(self):
        assert OpenBMIImporter.detect(OPENBMI_ROOT) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Import tests
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_data
class TestMIImport:
    """Test Motor Imagery import."""

    @pytest.fixture()
    def mi_results(self, tmp_path):
        pool = Pool.create(tmp_path / "pool_mi")
        config = TaskConfig.builtin("openbmi_mi")
        imp = OpenBMIImporter(pool, config)
        return imp.import_subject(
            mat_path=MI_DIR / "sess01_subj01_EEG_MI.mat",
            subject_id="S01",
            max_trials=10,
        )

    def test_two_splits(self, mi_results):
        """Train + test = 2 ImportResults."""
        assert len(mi_results) == 2

    def test_atom_count(self, mi_results):
        """10 trials × 2 splits = 20 atoms."""
        total = sum(len(r.atoms) for r in mi_results)
        assert total == 20

    def test_signal_shape(self, mi_results):
        """Each atom: (62 channels, 4000 samples)."""
        atom = mi_results[0].atoms[0]
        assert atom.signal_ref.shape == (62, 4000)

    def test_sampling_rate(self, mi_results):
        atom = mi_results[0].atoms[0]
        assert atom.sampling_rate == 1000.0

    def test_class_labels(self, mi_results):
        """Labels should be right_hand or left_hand."""
        labels = set()
        for r in mi_results:
            for a in r.atoms:
                for ann in a.annotations:
                    if ann.name == "mi_class":
                        labels.add(ann.value)
        assert labels == {"right_hand", "left_hand"}

    def test_split_annotation(self, mi_results):
        """First split is 'train', second is 'test'."""
        splits = []
        for r in mi_results:
            for ann in r.atoms[0].annotations:
                if ann.name == "split":
                    splits.append(ann.value)
                    break
        assert splits == ["train", "test"]

    def test_channel_infos(self, mi_results):
        ch_infos = mi_results[0].channel_infos
        assert len(ch_infos) == 62
        names = [ch.name for ch in ch_infos]
        assert "C3" in names
        assert "Cz" in names
        assert "Fz" in names


@skip_no_data
class TestERPImport:
    """Test ERP (P300) import."""

    @pytest.fixture()
    def erp_results(self, tmp_path):
        pool = Pool.create(tmp_path / "pool_erp")
        config = TaskConfig.builtin("openbmi_erp")
        imp = OpenBMIImporter(pool, config)
        return imp.import_subject(
            mat_path=ERP_DIR / "sess01_subj01_EEG_ERP.mat",
            subject_id="S01",
            max_trials=10,
        )

    def test_erp_shape(self, erp_results):
        """ERP epochs: (62, 800) = 0.8s @ 1000 Hz."""
        atom = erp_results[0].atoms[0]
        assert atom.signal_ref.shape == (62, 800)

    def test_erp_classes(self, erp_results):
        labels = set()
        for r in erp_results:
            for a in r.atoms:
                for ann in a.annotations:
                    if ann.name == "erp_class":
                        labels.add(ann.value)
        assert labels <= {"target", "nontarget"}

    def test_erp_annotation_name(self, erp_results):
        ann_names = {ann.name for ann in erp_results[0].atoms[0].annotations}
        assert "erp_class" in ann_names
        assert "erp_label" in ann_names


@skip_no_data
class TestSSVEPImport:
    """Test SSVEP import."""

    @pytest.fixture()
    def ssvep_results(self, tmp_path):
        pool = Pool.create(tmp_path / "pool_ssvep")
        config = TaskConfig.builtin("openbmi_ssvep")
        imp = OpenBMIImporter(pool, config)
        return imp.import_subject(
            mat_path=SSVEP_DIR / "sess01_subj01_EEG_SSVEP.mat",
            subject_id="S01",
            max_trials=10,
        )

    def test_ssvep_shape(self, ssvep_results):
        """SSVEP epochs: (62, 4000) = 4s @ 1000 Hz."""
        atom = ssvep_results[0].atoms[0]
        assert atom.signal_ref.shape == (62, 4000)

    def test_ssvep_four_classes(self, ssvep_results):
        labels = set()
        for r in ssvep_results:
            for a in r.atoms:
                for ann in a.annotations:
                    if ann.name == "ssvep_class":
                        labels.add(ann.value)
        assert labels <= {"up", "left", "right", "down"}

    def test_stimulus_frequency(self, ssvep_results):
        """SSVEP atoms should have stimulus_frequency_hz annotation."""
        freqs = set()
        for r in ssvep_results:
            for a in r.atoms:
                for ann in a.annotations:
                    if ann.name == "stimulus_frequency_hz":
                        freqs.add(ann.numeric_value)
        assert freqs <= {5.45, 6.67, 8.57, 12.0}
        assert len(freqs) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Signal quality tests
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_data
class TestSignalCharacteristics:
    """Verify imported signal properties match known OpenBMI specs."""

    @pytest.fixture()
    def mi_atom(self, tmp_path):
        pool = Pool.create(tmp_path / "pool_sig")
        config = TaskConfig.builtin("openbmi_mi")
        imp = OpenBMIImporter(pool, config)
        results = imp.import_subject(
            mat_path=MI_DIR / "sess01_subj01_EEG_MI.mat",
            subject_id="S01",
            max_trials=3,
        )
        atom = results[0].atoms[0]
        # Read signal back
        from neuroatom.storage.signal_store import ShardManager
        signal = ShardManager.static_read(pool.root, atom.signal_ref)
        return signal

    def test_signal_dtype(self, mi_atom):
        """Signal stored as float64 (or float32 after validation)."""
        assert mi_atom.dtype in (np.float32, np.float64)

    def test_signal_not_flat(self, mi_atom):
        """Signal should have meaningful variance."""
        assert mi_atom.std() > 1.0  # µV-scale EEG

    def test_signal_amplitude_range(self, mi_atom):
        """Typical EEG in µV should be within [-2000, 2000]."""
        assert mi_atom.min() > -5000
        assert mi_atom.max() < 5000


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-subject / paradigm import
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_data
class TestMultiSubjectImport:
    """Test import_paradigm for multiple subjects."""

    def test_import_paradigm_mi(self, tmp_path):
        pool = Pool.create(tmp_path / "pool_multi")
        config = TaskConfig.builtin("openbmi_mi")
        imp = OpenBMIImporter(pool, config)
        results = imp.import_paradigm(
            data_dir=OPENBMI_ROOT,
            paradigm="MI",
            max_subjects=2,
            max_trials=5,
            sessions=[1],
        )
        # 2 subjects × 2 splits × 5 trials = 20 atoms
        total = sum(len(r.atoms) for r in results)
        assert total == 20
        # Should have 4 ImportResults (2 subjects × 2 splits)
        assert len(results) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# Index + Query integration
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_data
class TestIndexAndQuery:
    """Test import → index → query pipeline."""

    def test_import_index_query(self, tmp_path):
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.query import QueryBuilder

        pool = Pool.create(tmp_path / "pool_iq")
        config = TaskConfig.builtin("openbmi_mi")
        imp = OpenBMIImporter(pool, config)
        imp.import_subject(
            mat_path=MI_DIR / "sess01_subj01_EEG_MI.mat",
            subject_id="S01",
            max_trials=10,
        )

        # Index
        indexer = Indexer(pool)
        n_indexed = indexer.reindex_all()
        assert n_indexed == 20  # 10 train + 10 test

        # Query: get only right_hand trials
        qb = QueryBuilder(indexer.backend)
        right_ids = qb.query_atom_ids({
            "dataset_id": "openbmi_mi",
            "annotations": [{"name": "mi_class", "value_in": ["right_hand"]}],
        })
        assert len(right_ids) > 0

    def test_query_by_paradigm(self, tmp_path):
        """Import MI and SSVEP, query by dataset_id."""
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.query import QueryBuilder

        pool = Pool.create(tmp_path / "pool_paradigm")

        # Import MI
        mi_config = TaskConfig.builtin("openbmi_mi")
        mi_imp = OpenBMIImporter(pool, mi_config)
        mi_imp.import_subject(
            mat_path=MI_DIR / "sess01_subj01_EEG_MI.mat",
            subject_id="S01",
            max_trials=5,
        )

        # Import SSVEP
        ssvep_config = TaskConfig.builtin("openbmi_ssvep")
        ssvep_imp = OpenBMIImporter(pool, ssvep_config)
        ssvep_imp.import_subject(
            mat_path=SSVEP_DIR / "sess01_subj01_EEG_SSVEP.mat",
            subject_id="S01",
            max_trials=5,
        )

        # Index
        indexer = Indexer(pool)
        n_indexed = indexer.reindex_all()
        assert n_indexed == 20  # (5+5)×2 = 20

        # Query MI only
        qb = QueryBuilder(indexer.backend)
        mi_ids = qb.query_atom_ids({"dataset_id": "openbmi_mi"})
        assert len(mi_ids) == 10

        # Query SSVEP only
        ssvep_ids = qb.query_atom_ids({"dataset_id": "openbmi_ssvep"})
        assert len(ssvep_ids) == 10
