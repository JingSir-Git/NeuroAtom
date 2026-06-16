"""Quality gate: evaluate a dataset's atoms and assign a quality tier.

The gate examines atoms from a completed import and checks them
against the admission policy criteria. It returns a detailed report
and the highest tier achieved.

Usage::

    from neuroatom.quality.gate import QualityGate

    gate = QualityGate(pool)
    report = gate.assess_dataset("physionet_mi")
    print(report.tier, report.summary)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from neuroatom.core.atom import Atom
from neuroatom.core.enums import QualityTier
from neuroatom.quality.admission import AdmissionPolicy, TierCriteria, default_policy
from neuroatom.storage.metadata_store import AtomJSONLReader
from neuroatom.storage.pool import Pool
from neuroatom.storage import paths as P
from neuroatom.utils.channel_names import standardize_channel_name

logger = logging.getLogger(__name__)


@dataclass
class TierCheckResult:
    """Result of checking one tier's criteria."""
    tier: QualityTier
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Full quality assessment report for a dataset."""
    dataset_id: str
    tier: Optional[QualityTier]
    n_atoms: int
    n_subjects: int
    tier_results: List[TierCheckResult] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One-line human-readable summary."""
        tier_str = self.tier.value.upper() if self.tier else "UNRATED"
        return (
            f"[{self.dataset_id}] {tier_str} — "
            f"{self.n_atoms} atoms, {self.n_subjects} subjects"
        )

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "dataset_id": self.dataset_id,
            "tier": self.tier.value if self.tier else None,
            "n_atoms": self.n_atoms,
            "n_subjects": self.n_subjects,
            "stats": self.stats,
            "tier_results": [
                {
                    "tier": r.tier.value,
                    "passed": r.passed,
                    "checks": r.checks,
                    "details": r.details,
                }
                for r in self.tier_results
            ],
            "warnings": self.warnings,
        }


class QualityGate:
    """Evaluate atoms against admission policy and assign a quality tier."""

    def __init__(
        self,
        pool: Pool,
        policy: Optional[AdmissionPolicy] = None,
    ):
        self._pool = pool
        self._policy = policy or AdmissionPolicy.from_pool_config(pool.config)

    def assess_dataset(self, dataset_id: str) -> QualityReport:
        """Evaluate all atoms in a dataset and produce a quality report.

        Args:
            dataset_id: The dataset to assess.

        Returns:
            QualityReport with tier assignment and detailed checks.
        """
        atoms = self._load_all_atoms(dataset_id)

        if not atoms:
            logger.warning("No atoms found for dataset %s", dataset_id)
            return QualityReport(
                dataset_id=dataset_id,
                tier=None, n_atoms=0, n_subjects=0,
                warnings=["No atoms found in dataset."],
            )

        # Collect stats across all atoms
        stats = self._compute_stats(atoms)

        # Evaluate each tier
        tier_order = [QualityTier.PLATINUM, QualityTier.GOLD, QualityTier.SILVER]
        tier_results = []
        achieved_tier = None

        for tier in tier_order:
            criteria = self._policy.tiers.get(tier)
            if criteria is None:
                continue
            result = self._check_tier(tier, criteria, atoms, stats)
            tier_results.append(result)
            if result.passed and achieved_tier is None:
                achieved_tier = tier

        # Tier results in ascending order for display
        tier_results.reverse()

        # Warnings
        warnings = []
        if achieved_tier is None:
            warnings.append("Dataset does not meet Silver tier criteria.")
        if stats.get("nan_atom_count", 0) > 0:
            warnings.append(
                f"{stats['nan_atom_count']} atoms contain NaN values."
            )
        if stats.get("flatline_atom_count", 0) > 0:
            warnings.append(
                f"{stats['flatline_atom_count']} atoms have flatline channels."
            )

        report = QualityReport(
            dataset_id=dataset_id,
            tier=achieved_tier,
            n_atoms=stats["n_atoms"],
            n_subjects=stats["n_subjects"],
            tier_results=tier_results,
            stats=stats,
            warnings=warnings,
        )

        logger.info("Quality assessment: %s", report.summary)
        return report

    def assess_atoms(self, atoms: List[Atom]) -> QualityReport:
        """Evaluate a list of atoms (e.g. from a single import) in memory.

        Useful for calling right after import without re-reading from disk.
        """
        if not atoms:
            return QualityReport(
                dataset_id="unknown", tier=None, n_atoms=0, n_subjects=0,
                warnings=["No atoms provided."],
            )

        dataset_id = atoms[0].dataset_id
        stats = self._compute_stats(atoms)

        tier_order = [QualityTier.PLATINUM, QualityTier.GOLD, QualityTier.SILVER]
        tier_results = []
        achieved_tier = None

        for tier in tier_order:
            criteria = self._policy.tiers.get(tier)
            if criteria is None:
                continue
            result = self._check_tier(tier, criteria, atoms, stats)
            tier_results.append(result)
            if result.passed and achieved_tier is None:
                achieved_tier = tier

        tier_results.reverse()

        return QualityReport(
            dataset_id=dataset_id,
            tier=achieved_tier,
            n_atoms=stats["n_atoms"],
            n_subjects=stats["n_subjects"],
            tier_results=tier_results,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_all_atoms(self, dataset_id: str) -> List[Atom]:
        """Load all atoms for a dataset by scanning JSONL files."""
        atoms = []
        ds_dir = P.dataset_dir(self._pool.root, dataset_id)
        if not ds_dir.exists():
            return atoms

        # Walk all atoms.jsonl files
        for jsonl_path in sorted(ds_dir.rglob("atoms.jsonl")):
            try:
                reader = AtomJSONLReader(jsonl_path)
                for atom in reader.iter_atoms():
                    atoms.append(atom)
            except Exception as e:
                logger.warning("Failed to read %s: %s", jsonl_path, e)

        return atoms

    def _compute_stats(self, atoms: List[Atom]) -> Dict:
        """Compute aggregate statistics across atoms."""
        subjects: Set[str] = set()
        channel_counts: List[int] = []
        has_labels: int = 0
        has_standard_channels: int = 0
        has_electrode_locations: int = 0
        has_processing_history: int = 0
        has_stimulus_ref: int = 0
        bad_channel_ratios: List[float] = []
        nan_atom_count: int = 0
        flatline_atom_count: int = 0

        for atom in atoms:
            subjects.add(atom.subject_id)
            channel_counts.append(atom.n_channels)

            # Labels
            if atom.annotations:
                has_labels += 1

            # Standard channel names
            all_standard = True
            for ch_id in atom.channel_ids:
                std = standardize_channel_name(ch_id)
                if std == ch_id.upper() and std not in _KNOWN_STANDARD:
                    all_standard = False
                    break
            if all_standard:
                has_standard_channels += 1

            # Electrode locations: check session-level via pool
            # (skipped here — checked at dataset level separately)

            # Processing history
            if atom.processing_history and atom.processing_history.steps:
                has_processing_history += 1

            # Stimulus reference: check if any annotation is a stimulus ref
            for ann in atom.annotations:
                ann_type = getattr(ann, "annotation_type", "")
                if ann_type == "stimulus_ref":
                    has_stimulus_ref += 1
                    break

            # Bad channels
            n_bad = 0
            if atom.quality and atom.quality.bad_channels:
                n_bad = len(atom.quality.bad_channels)
            ratio = n_bad / max(atom.n_channels, 1)
            bad_channel_ratios.append(ratio)

            # Quality flags
            if atom.quality:
                if atom.quality.rejection_reason and "nan" in atom.quality.rejection_reason.lower():
                    nan_atom_count += 1
                if atom.quality.rejection_reason and "flat" in atom.quality.rejection_reason.lower():
                    flatline_atom_count += 1

        # Check electrode locations at session level
        # Count atoms whose session has an electrodes.json
        sessions_with_electrodes: set = set()
        try:
            ds_dir = P.dataset_dir(self._pool.root, atoms[0].dataset_id)
            for elec_path in ds_dir.rglob("electrodes.json"):
                # Extract session key from path
                parts = elec_path.relative_to(ds_dir).parts
                # subjects/{sub}/sessions/{ses}/electrodes.json
                if len(parts) >= 4:
                    sub, ses = parts[1], parts[3]
                    sessions_with_electrodes.add((sub, ses))
        except Exception:
            pass

        for atom in atoms:
            key = (atom.subject_id, atom.session_id or "")
            if key in sessions_with_electrodes:
                has_electrode_locations += 1

        n_atoms = len(atoms)
        return {
            "n_atoms": n_atoms,
            "n_subjects": len(subjects),
            "subjects": sorted(subjects),
            "min_channels": min(channel_counts) if channel_counts else 0,
            "max_channels": max(channel_counts) if channel_counts else 0,
            "avg_channels": round(sum(channel_counts) / n_atoms, 1) if n_atoms else 0,
            "has_labels_ratio": has_labels / n_atoms if n_atoms else 0,
            "has_standard_channels_ratio": has_standard_channels / n_atoms if n_atoms else 0,
            "has_electrode_locations_ratio": has_electrode_locations / n_atoms if n_atoms else 0,
            "has_processing_history_ratio": has_processing_history / n_atoms if n_atoms else 0,
            "has_stimulus_ref_ratio": has_stimulus_ref / n_atoms if n_atoms else 0,
            "avg_bad_channel_ratio": round(np.mean(bad_channel_ratios), 4) if bad_channel_ratios else 0,
            "max_bad_channel_ratio": round(max(bad_channel_ratios), 4) if bad_channel_ratios else 0,
            "nan_atom_count": nan_atom_count,
            "flatline_atom_count": flatline_atom_count,
        }

    def _check_tier(
        self,
        tier: QualityTier,
        criteria: TierCriteria,
        atoms: List[Atom],
        stats: Dict,
    ) -> TierCheckResult:
        """Check if atoms meet a specific tier's criteria."""
        checks = {}
        details = {}
        n = stats["n_atoms"]

        # min_channels
        ok = stats["min_channels"] >= criteria.min_channels
        checks["min_channels"] = ok
        if not ok:
            details["min_channels"] = (
                f"Need {criteria.min_channels}, got {stats['min_channels']}"
            )

        # max_bad_channel_ratio
        ok = stats["max_bad_channel_ratio"] <= criteria.max_bad_channel_ratio
        checks["max_bad_channel_ratio"] = ok
        if not ok:
            details["max_bad_channel_ratio"] = (
                f"Max {criteria.max_bad_channel_ratio}, got {stats['max_bad_channel_ratio']}"
            )

        # max_nan_ratio
        nan_ratio = stats["nan_atom_count"] / n if n else 0
        ok = nan_ratio <= criteria.max_nan_ratio
        checks["max_nan_ratio"] = ok
        if not ok:
            details["max_nan_ratio"] = (
                f"Max {criteria.max_nan_ratio}, got {nan_ratio:.3f}"
            )

        # max_flatline_ratio
        flat_ratio = stats["flatline_atom_count"] / n if n else 0
        ok = flat_ratio <= criteria.max_flatline_ratio
        checks["max_flatline_ratio"] = ok
        if not ok:
            details["max_flatline_ratio"] = (
                f"Max {criteria.max_flatline_ratio}, got {flat_ratio:.3f}"
            )

        # require_labels
        if criteria.require_labels:
            ok = stats["has_labels_ratio"] >= 0.9  # 90% atoms need labels
            checks["require_labels"] = ok
            if not ok:
                details["require_labels"] = (
                    f"Need ≥90% labeled, got {stats['has_labels_ratio']:.1%}"
                )
        else:
            checks["require_labels"] = True

        # require_standard_channels
        if criteria.require_standard_channels:
            ok = stats["has_standard_channels_ratio"] >= 0.9
            checks["require_standard_channels"] = ok
            if not ok:
                details["require_standard_channels"] = (
                    f"Need ≥90% standard, got {stats['has_standard_channels_ratio']:.1%}"
                )
        else:
            checks["require_standard_channels"] = True

        # require_electrode_locations
        if criteria.require_electrode_locations:
            ok = stats["has_electrode_locations_ratio"] >= 0.8
            checks["require_electrode_locations"] = ok
            if not ok:
                details["require_electrode_locations"] = (
                    f"Need ≥80% with coords, got {stats['has_electrode_locations_ratio']:.1%}"
                )
        else:
            checks["require_electrode_locations"] = True

        # require_processing_history
        if criteria.require_processing_history:
            ok = stats["has_processing_history_ratio"] >= 0.9
            checks["require_processing_history"] = ok
            if not ok:
                details["require_processing_history"] = (
                    f"Need ≥90%, got {stats['has_processing_history_ratio']:.1%}"
                )
        else:
            checks["require_processing_history"] = True

        # require_stimulus_ref
        if criteria.require_stimulus_ref:
            ok = stats["has_stimulus_ref_ratio"] >= 0.5
            checks["require_stimulus_ref"] = ok
            if not ok:
                details["require_stimulus_ref"] = (
                    f"Need ≥50%, got {stats['has_stimulus_ref_ratio']:.1%}"
                )
        else:
            checks["require_stimulus_ref"] = True

        # min_atoms
        ok = n >= criteria.min_atoms
        checks["min_atoms"] = ok
        if not ok:
            details["min_atoms"] = f"Need {criteria.min_atoms}, got {n}"

        # min_subjects
        ok = stats["n_subjects"] >= criteria.min_subjects
        checks["min_subjects"] = ok
        if not ok:
            details["min_subjects"] = (
                f"Need {criteria.min_subjects}, got {stats['n_subjects']}"
            )

        passed = all(checks.values())
        return TierCheckResult(tier=tier, passed=passed, checks=checks, details=details)


# Known standard 10-20 / 10-10 channel names for quick lookup
_KNOWN_STANDARD = {
    "FP1", "FP2", "FPZ", "F7", "F3", "FZ", "F4", "F8",
    "FT7", "FC3", "FC1", "FCZ", "FC2", "FC4", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
    "AF3", "AF4", "AF7", "AF8", "AFZ",
    "T3", "T4", "T5", "T6",  # legacy 10-20
    "A1", "A2",  # ear references
    "F9", "F10", "P9", "P10",  # extended
    "IZ", "NZ",
    "FC5", "FC6", "CP3", "CP4",
    "TP9", "TP10",
}
