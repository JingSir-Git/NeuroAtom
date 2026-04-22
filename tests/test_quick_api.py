"""Tests for the high-level convenience API (quickload, TaskConfig.builtin, etc.)."""

import pytest
from pathlib import Path

from neuroatom.importers.base import TaskConfig
from neuroatom.importers.registry import get_importer_class, list_formats


# ═══════════════════════════════════════════════════════════════════════════
# TaskConfig.builtin
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskConfigBuiltin:
    def test_builtin_bci(self):
        cfg = TaskConfig.builtin("bci_comp_iv_2a")
        assert cfg.dataset_id == "bci_comp_iv_2a"
        assert cfg.task_type == "motor_imagery"

    def test_builtin_physionet(self):
        cfg = TaskConfig.builtin("physionet_mi")
        assert cfg.dataset_id == "physionet_mi"

    def test_builtin_seed_v(self):
        cfg = TaskConfig.builtin("seed_v")
        assert cfg.dataset_id == "seed_v"

    def test_builtin_zuco2(self):
        cfg = TaskConfig.builtin("zuco2_tsr")
        assert "dataset_id" in cfg.data

    def test_builtin_kul_aad(self):
        cfg = TaskConfig.builtin("kul_aad")
        assert "dataset_id" in cfg.data

    def test_builtin_chinese_eeg2(self):
        cfg = TaskConfig.builtin("chinese_eeg2_listening")
        assert cfg.dataset_id == "chinese_eeg2_listening"

    def test_builtin_not_found(self):
        with pytest.raises(FileNotFoundError, match="No built-in task config"):
            TaskConfig.builtin("nonexistent_dataset_xyz")

    def test_builtin_error_lists_available(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            TaskConfig.builtin("nope")
        msg = str(exc_info.value)
        assert "bci_comp_iv_2a" in msg
        assert "physionet_mi" in msg


# ═══════════════════════════════════════════════════════════════════════════
# Importer registry
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_list_formats_contains_core(self):
        fmts = list_formats()
        assert "bci_comp_iv_2a" in fmts
        assert "physionet_mi" in fmts
        assert "mne_generic" in fmts

    def test_get_importer_class_bci(self):
        cls = get_importer_class("bci_comp_iv_2a")
        assert cls.__name__ == "BCICompIV2aImporter"

    def test_get_importer_class_unknown(self):
        with pytest.raises(ValueError, match="Unknown format"):
            get_importer_class("this_does_not_exist")

    def test_lazy_load_all_importers(self):
        fmts = list_formats()
        assert len(fmts) >= 10


# ═══════════════════════════════════════════════════════════════════════════
# quickload (unit-level: no data needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestQuickloadUnit:
    def test_import_quickload(self):
        from neuroatom.quick import quickload
        assert callable(quickload)

    def test_quickload_via_na(self):
        import neuroatom as na
        assert hasattr(na, "quickload")
        assert callable(na.quickload)

    def test_infer_label_field(self):
        from neuroatom.quick import _infer_label_field
        cfg = TaskConfig.builtin("bci_comp_iv_2a")
        lf = _infer_label_field(cfg, "bci_comp_iv_2a")
        assert lf == "mi_class"

    def test_infer_subject_bci(self):
        from neuroatom.quick import _infer_subject
        assert _infer_subject(Path("data/A01T.mat"), "bci_comp_iv_2a") == "A01"

    def test_infer_subject_aad(self):
        from neuroatom.quick import _infer_subject
        assert _infer_subject(Path("data/S5.mat"), "kul_aad") == "S5"

    def test_resolve_config_name(self):
        from neuroatom.quick import _resolve_config_name
        assert _resolve_config_name("zuco2") == "zuco2_tsr"
        assert _resolve_config_name("bci_comp_iv_2a") == "bci_comp_iv_2a"


# ═══════════════════════════════════════════════════════════════════════════
# quickload E2E (requires real data — skipped if not available)
# ═══════════════════════════════════════════════════════════════════════════

BCI_MAT = Path(r"C:\Data\BCI_Competition\A01T.mat")


@pytest.mark.skipif(not BCI_MAT.exists(), reason="BCI IV 2a data not available")
class TestQuickloadE2E:
    def test_quickload_default(self):
        import neuroatom as na
        loader = na.quickload(
            "bci_comp_iv_2a",
            data_path=BCI_MAT,
            subject="A01",
            batch_size=8,
        )
        batch = next(iter(loader))
        assert batch["signal"].shape[0] == 8         # batch size
        assert batch["signal"].shape[1] == 25        # 25 channels
        assert "mi_class" in batch["labels"]

    def test_quickload_with_split(self):
        import neuroatom as na
        train_l, test_l = na.quickload(
            "bci_comp_iv_2a",
            data_path=BCI_MAT,
            subject="A01",
            batch_size=8,
            split_test_ratio=0.2,
        )
        assert len(train_l) > 0
        assert len(test_l) > 0

    def test_quickload_with_band(self):
        import neuroatom as na
        loader = na.quickload(
            "bci_comp_iv_2a",
            data_path=BCI_MAT,
            subject="A01",
            batch_size=8,
            band=(8.0, 30.0),
        )
        batch = next(iter(loader))
        assert batch["signal"].shape[0] == 8
