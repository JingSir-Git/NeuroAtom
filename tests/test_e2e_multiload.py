"""End-to-end tests for multiload cross-dataset convenience API.

Tests cross-subject, cross-dataset, and cross-task assembly via
``neuroatom.multiload()``.

Requires:
- NEUROATOM_BCI_IV_2A_DIR env var pointing to BCI IV 2a data directory
- NEUROATOM_OPENBMI_DIR env var or default path for OpenBMI data
"""

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

logging.basicConfig(level=logging.INFO)

# ── Data paths ──────────────────────────────────────────────────────────
_bci_dir = os.environ.get("NEUROATOM_BCI_IV_2A_DIR", "")
BCI_A01T = Path(_bci_dir) / "A01T.mat" if _bci_dir else Path("__nonexistent__")
BCI_A02T = Path(_bci_dir) / "A02T.mat" if _bci_dir else Path("__nonexistent__")

_openbmi_dir = os.environ.get(
    "NEUROATOM_OPENBMI_DIR",
    r"\\wsqlab\share\JCH\OpenBMI",
)
OPENBMI_MI_S01 = (
    Path(_openbmi_dir) / "MI" / "sess01_subj01_EEG_MI.mat"
)

# Skip entire module if data unavailable
_has_bci = BCI_A01T.exists()
_has_openbmi = OPENBMI_MI_S01.exists()

COMMON_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


@pytest.fixture
def pool_dir(tmp_path):
    d = tmp_path / "multiload_pool"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. Cross-dataset: OpenBMI MI + BCI IV 2a
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (_has_bci and _has_openbmi),
    reason="Requires BCI IV 2a + OpenBMI data",
)
class TestCrossDataset:
    """Fuse OpenBMI MI (62ch/1000Hz) + BCI IV 2a (25ch/250Hz)."""

    def test_basic_cross_dataset(self, pool_dir):
        """Import 1 subject from each dataset, unify channels & srate."""
        from neuroatom.quick import multiload

        train, val, test = multiload(
            sources=[
                {
                    "dataset": "openbmi_mi",
                    "path": str(OPENBMI_MI_S01),
                    "subjects": ["S01"],
                },
                {
                    "dataset": "bci_comp_iv_2a",
                    "path": str(BCI_A01T),
                    "subjects": ["A01"],
                },
            ],
            pool_dir=pool_dir,
            target_channels=COMMON_22,
            target_srate=250,
            target_duration=4.0,
            label_field="mi_class",
            split_strategy="stratified",
            split_config={"val_ratio": 0.0, "test_ratio": 0.2, "seed": 42},
        )

        # At least one loader should be non-empty
        all_loaders = [l for l in (train, val, test) if l is not None]
        assert len(all_loaders) >= 1

        # Total samples = 40 (OpenBMI, 20 train + 20 test) + 288 (BCI IV 2a)
        total = sum(len(l.dataset) for l in all_loaders)
        assert total >= 300, f"Expected ≥300 total atoms, got {total}"

        # Check signal shape in first batch
        batch = next(iter(all_loaders[0]))
        sig = batch["signal"]
        assert sig.shape[1] == len(COMMON_22), (
            f"Channel count: {sig.shape[1]} vs {len(COMMON_22)}"
        )
        assert sig.shape[2] == 1000, (
            f"Samples: {sig.shape[2]} vs 1000 (4s × 250Hz)"
        )

        # Verify both datasets appear in the data
        all_ds_ids = set()
        for loader in all_loaders:
            for batch in loader:
                all_ds_ids.update(batch["dataset_id"])
                if len(all_ds_ids) == 2:
                    break
            if len(all_ds_ids) == 2:
                break

        assert "openbmi_mi" in all_ds_ids
        assert "bci_comp_iv_2a" in all_ds_ids

        print(f"\n✓ Cross-dataset: {total} atoms, shape {sig.shape}, "
              f"datasets={all_ds_ids}")

    def test_cross_dataset_bandpass(self, pool_dir):
        """Cross-dataset with bandpass filter applied."""
        from neuroatom.quick import multiload

        pool_dir2 = pool_dir / "bandpass"
        pool_dir2.mkdir()

        train, val, test = multiload(
            sources=[
                {
                    "dataset": "openbmi_mi",
                    "path": str(OPENBMI_MI_S01),
                    "subjects": ["S01"],
                },
                {
                    "dataset": "bci_comp_iv_2a",
                    "path": str(BCI_A01T),
                    "subjects": ["A01"],
                },
            ],
            pool_dir=pool_dir2,
            target_channels=COMMON_22,
            target_srate=250,
            target_duration=4.0,
            band=(8.0, 30.0),  # mu + beta rhythm
            label_field="mi_class",
            split_strategy="stratified",
            split_config={"val_ratio": 0.0, "test_ratio": 0.2, "seed": 42},
        )

        loader = train or val or test
        assert loader is not None
        batch = next(iter(loader))
        sig = batch["signal"].numpy()

        # After bandpass filter, signal should be reasonable
        assert np.isfinite(sig).all(), "Filtered signal contains NaN/Inf"
        assert sig.std() > 0, "Signal is flat after filtering"

        print(f"\n✓ Cross-dataset with 8-30Hz bandpass: shape {sig.shape}")


# ══════════════════════════════════════════════════════════════════════════
# 2. Cross-subject within single dataset
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not _has_bci or not (Path(_bci_dir) / "A02T.mat").exists(),
    reason="Requires A01T.mat and A02T.mat in BCI IV 2a directory",
)
class TestCrossSubject:
    """Multiple subjects from same dataset → subject-level split."""

    def test_multi_subject_bci(self, pool_dir):
        """2 BCI IV 2a subjects → subject split (1 train, 1 test)."""
        from neuroatom.quick import multiload

        train, val, test = multiload(
            sources=[
                {
                    "dataset": "bci_comp_iv_2a",
                    "path": str(BCI_A01T),
                    "subjects": ["A01"],
                },
                {
                    "dataset": "bci_comp_iv_2a",
                    "path": str(BCI_A02T),
                    "subjects": ["A02"],
                },
            ],
            pool_dir=pool_dir,
            target_channels=COMMON_22,
            target_srate=250,
            target_duration=4.0,
            label_field="mi_class",
            # 2 subjects → val+test take 1 each, train = 0.
            # Instead use test_subjects to control.
            split_strategy="subject",
            split_config={
                "test_subjects": ["bci_comp_iv_2a|A02"],
                "val_ratio": 0.0,
            },
        )

        # Train should have A01's 288 atoms, test should have A02's 288
        assert train is not None
        assert test is not None
        n_train = len(train.dataset)
        n_test = len(test.dataset)

        assert n_train == 288, f"Expected 288 train atoms (A01), got {n_train}"
        assert n_test == 288, f"Expected 288 test atoms (A02), got {n_test}"

        # Verify no subject leakage
        train_subs = set()
        for batch in train:
            train_subs.update(batch["subject_id"])
        assert train_subs == {"A01"}, f"Train subjects: {train_subs}"

        test_subs = set()
        for batch in test:
            test_subs.update(batch["subject_id"])
        assert test_subs == {"A02"}, f"Test subjects: {test_subs}"

        print(f"\n✓ Cross-subject: {n_train} train (A01) + {n_test} test (A02), "
              f"no subject leakage")


# ══════════════════════════════════════════════════════════════════════════
# 3. Error handling & edge cases
# ══════════════════════════════════════════════════════════════════════════

class TestMultiloadEdgeCases:
    """Edge cases for multiload()."""

    def test_empty_sources(self):
        """multiload with no sources should raise."""
        from neuroatom.quick import multiload
        with pytest.raises(ValueError, match="at least one source"):
            multiload(sources=[])

    def test_multiload_importable(self):
        """multiload should be importable from top-level."""
        import neuroatom as na
        assert hasattr(na, "multiload")
        assert callable(na.multiload)

    @pytest.mark.skipif(not _has_bci, reason="Requires BCI IV 2a data")
    def test_single_source(self, pool_dir):
        """multiload with single source works like quickload."""
        from neuroatom.quick import multiload

        train, val, test = multiload(
            sources=[
                {
                    "dataset": "bci_comp_iv_2a",
                    "path": str(BCI_A01T),
                    "subjects": ["A01"],
                },
            ],
            pool_dir=pool_dir,
            target_channels=COMMON_22,
            target_srate=250,
            target_duration=4.0,
            label_field="mi_class",
            split_strategy="stratified",
            split_config={"val_ratio": 0.0, "test_ratio": 0.2, "seed": 42},
        )

        total = sum(len(l.dataset) for l in (train, val, test) if l)
        assert total == 288, f"Expected 288 atoms from single A01, got {total}"
        print(f"\n✓ Single-source multiload: {total} atoms")
