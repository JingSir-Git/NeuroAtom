"""Tests for quality gate and admission policy.

Tests cover:
- Default policy structure
- Tier assessment with controlled atoms
- Silver-level pass/fail
- Gold-level requirements
- Quality report formatting
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.annotation import CategoricalAnnotation
from neuroatom.core.enums import AtomType, QualityTier
from neuroatom.core.quality import QualityInfo
from neuroatom.core.signal_ref import SignalRef
from neuroatom.quality.admission import AdmissionPolicy, TierCriteria, default_policy
from neuroatom.quality.gate import QualityGate, QualityReport


def _make_atom(
    atom_id="a1",
    subject_id="S01",
    n_channels=32,
    has_labels=True,
    bad_channels=None,
):
    """Helper to create a minimal Atom for testing."""
    ch_ids = [f"Ch{i}" for i in range(n_channels)]
    annotations = []
    if has_labels:
        annotations = [CategoricalAnnotation(
            annotation_id="ann1", name="label", value="left_hand",
        )]

    quality = None
    if bad_channels:
        quality = QualityInfo(bad_channels=bad_channels)

    return Atom(
        atom_id=atom_id,
        dataset_id="test_ds",
        subject_id=subject_id,
        session_id="ses-01",
        run_id="run-01",
        atom_type=AtomType.TRIAL,
        signal_ref=SignalRef(
            file_path="fake.h5", internal_path="/atoms/a1/signal",
            shape=(n_channels, 128),
        ),
        temporal=TemporalInfo(
            onset_sample=0, duration_samples=128,
            onset_seconds=0.0, duration_seconds=1.0,
        ),
        channel_ids=ch_ids,
        n_channels=n_channels,
        sampling_rate=128.0,
        signal_unit="uV",
        annotations=annotations,
        quality=quality,
    )


class TestAdmissionPolicy:

    def test_default_policy_has_three_tiers(self):
        policy = default_policy()
        assert QualityTier.SILVER in policy.tiers
        assert QualityTier.GOLD in policy.tiers
        assert QualityTier.PLATINUM in policy.tiers

    def test_silver_is_least_strict(self):
        policy = default_policy()
        silver = policy.tiers[QualityTier.SILVER]
        gold = policy.tiers[QualityTier.GOLD]
        assert silver.min_channels <= gold.min_channels
        assert silver.min_atoms <= gold.min_atoms

    def test_from_pool_config_default(self):
        policy = AdmissionPolicy.from_pool_config({})
        assert len(policy.tiers) == 3


class TestQualityGate:

    def test_silver_pass_with_labeled_atoms(self, tmp_path):
        """Atoms with labels should pass Silver."""
        from neuroatom.storage.pool import Pool
        pool = Pool.create(tmp_path / "pool")
        gate = QualityGate(pool)

        atoms = [_make_atom(atom_id=f"a{i}", has_labels=True) for i in range(5)]
        report = gate.assess_atoms(atoms)

        assert report.tier == QualityTier.SILVER
        assert report.n_atoms == 5

    def test_no_labels_fails_silver(self, tmp_path):
        """Atoms without labels should fail Silver (require_labels=True)."""
        from neuroatom.storage.pool import Pool
        pool = Pool.create(tmp_path / "pool")
        gate = QualityGate(pool)

        atoms = [_make_atom(atom_id=f"a{i}", has_labels=False) for i in range(5)]
        report = gate.assess_atoms(atoms)

        assert report.tier is None

    def test_gold_needs_more_atoms_and_channels(self, tmp_path):
        """Gold requires 10+ atoms and 8+ channels."""
        from neuroatom.storage.pool import Pool
        pool = Pool.create(tmp_path / "pool")

        # Custom policy: relax electrode/channel-name requirements
        policy = AdmissionPolicy(
            tiers={
                QualityTier.SILVER: TierCriteria(
                    min_channels=1, require_labels=True,
                ),
                QualityTier.GOLD: TierCriteria(
                    min_channels=8, require_labels=True,
                    require_standard_channels=False,
                    require_electrode_locations=False,
                    min_atoms=10,
                ),
            }
        )
        gate = QualityGate(pool, policy=policy)

        # Only 5 atoms → Silver, not Gold
        atoms = [
            _make_atom(atom_id=f"a{i}", n_channels=32, has_labels=True)
            for i in range(5)
        ]
        report = gate.assess_atoms(atoms)
        assert report.tier == QualityTier.SILVER

    def test_gold_pass_with_enough_atoms(self, tmp_path):
        """Gold passes with 10+ atoms, 8+ channels, labels."""
        from neuroatom.storage.pool import Pool
        pool = Pool.create(tmp_path / "pool")

        policy = AdmissionPolicy(
            tiers={
                QualityTier.SILVER: TierCriteria(
                    min_channels=1, require_labels=True,
                ),
                QualityTier.GOLD: TierCriteria(
                    min_channels=8, require_labels=True,
                    require_standard_channels=False,
                    require_electrode_locations=False,
                    min_atoms=10,
                ),
            }
        )
        gate = QualityGate(pool, policy=policy)

        atoms = [
            _make_atom(atom_id=f"a{i}", n_channels=32, has_labels=True)
            for i in range(10)
        ]
        report = gate.assess_atoms(atoms)
        assert report.tier == QualityTier.GOLD

    def test_bad_channels_affect_tier(self, tmp_path):
        """Too many bad channels should block Gold."""
        from neuroatom.storage.pool import Pool
        pool = Pool.create(tmp_path / "pool")
        gate = QualityGate(pool)

        # 60% bad channels → exceeds Gold max_bad_channel_ratio (0.2)
        atoms = [
            _make_atom(
                atom_id=f"a{i}", n_channels=10, has_labels=True,
                bad_channels=[f"Ch{j}" for j in range(6)],
            )
            for i in range(10)
        ]
        report = gate.assess_atoms(atoms)
        # Should not pass Gold (max_bad_channel_ratio=0.2)
        assert report.tier != QualityTier.GOLD

    def test_report_summary(self, tmp_path):
        from neuroatom.storage.pool import Pool
        pool = Pool.create(tmp_path / "pool")
        gate = QualityGate(pool)

        atoms = [_make_atom(atom_id=f"a{i}") for i in range(3)]
        report = gate.assess_atoms(atoms)
        assert "test_ds" in report.summary
        assert "3 atoms" in report.summary

    def test_report_to_dict(self, tmp_path):
        from neuroatom.storage.pool import Pool
        pool = Pool.create(tmp_path / "pool")
        gate = QualityGate(pool)

        atoms = [_make_atom(atom_id=f"a{i}") for i in range(3)]
        report = gate.assess_atoms(atoms)
        d = report.to_dict()
        assert "tier" in d
        assert "stats" in d
        assert "tier_results" in d

    def test_format_report_text(self, tmp_path):
        from neuroatom.storage.pool import Pool
        from neuroatom.quality.report import format_report
        pool = Pool.create(tmp_path / "pool")
        gate = QualityGate(pool)

        atoms = [_make_atom(atom_id=f"a{i}") for i in range(3)]
        report = gate.assess_atoms(atoms)
        text = format_report(report)
        assert "Quality Report" in text
        assert "SILVER" in text

    def test_empty_dataset(self, tmp_path):
        from neuroatom.storage.pool import Pool
        pool = Pool.create(tmp_path / "pool")
        gate = QualityGate(pool)

        report = gate.assess_atoms([])
        assert report.tier is None
        assert report.n_atoms == 0
