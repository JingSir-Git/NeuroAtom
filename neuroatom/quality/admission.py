"""Admission policy: defines tiered quality criteria for dataset import.

Each tier (Silver / Gold / Platinum) specifies minimum requirements that
a dataset's atoms must collectively satisfy. The quality gate evaluates
atoms against these criteria and assigns the highest tier met.

Usage::

    from neuroatom.quality.admission import AdmissionPolicy, default_policy

    policy = default_policy()
    # Or load from pool config:
    policy = AdmissionPolicy.from_pool_config(pool.config)
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from neuroatom.core.enums import QualityTier


class TierCriteria(BaseModel):
    """Criteria for a single quality tier.

    All criteria are thresholds: a dataset must meet *all* criteria
    for a tier to qualify. Criteria are evaluated across all atoms
    in the dataset.
    """

    # Signal quality
    min_channels: int = Field(
        default=1,
        ge=1,
        description="Minimum number of EEG channels per atom.",
    )
    max_bad_channel_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of channels marked as bad.",
    )
    max_nan_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of atoms with NaN values.",
    )
    max_flatline_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of atoms with flatline channels.",
    )

    # Metadata completeness
    require_labels: bool = Field(
        default=False,
        description="Require at least one annotation/label per atom.",
    )
    require_standard_channels: bool = Field(
        default=False,
        description="Require all channel names to be 10-20 standard.",
    )
    require_electrode_locations: bool = Field(
        default=False,
        description="Require electrode (x, y, z) coordinates.",
    )
    require_sampling_rate: bool = Field(
        default=True,
        description="Require explicit sampling rate.",
    )

    # Provenance
    require_processing_history: bool = Field(
        default=False,
        description="Require non-empty processing provenance chain.",
    )
    require_stimulus_ref: bool = Field(
        default=False,
        description="Require stimulus file reference.",
    )

    # Scale
    min_atoms: int = Field(
        default=1,
        ge=1,
        description="Minimum number of atoms in the dataset.",
    )
    min_subjects: int = Field(
        default=1,
        ge=1,
        description="Minimum number of distinct subjects.",
    )


class AdmissionPolicy(BaseModel):
    """Tiered admission policy defining Silver / Gold / Platinum criteria.

    The quality gate evaluates a dataset and assigns the highest tier
    whose criteria are fully met. If no tier is met, the dataset is
    still imported but marked as un-assessed (quality_tier = None).
    """

    tiers: Dict[QualityTier, TierCriteria] = Field(
        default_factory=dict,
        description="Criteria per quality tier.",
    )

    reject_below_silver: bool = Field(
        default=False,
        description=(
            "If True, reject imports that do not meet Silver criteria. "
            "If False, import anyway but leave quality_tier as None."
        ),
    )

    def tier_names(self) -> List[str]:
        """Return tier names in ascending order."""
        order = [QualityTier.SILVER, QualityTier.GOLD, QualityTier.PLATINUM]
        return [t.value for t in order if t in self.tiers]

    @classmethod
    def from_pool_config(cls, pool_config: dict) -> "AdmissionPolicy":
        """Build policy from pool config (if present), else use defaults."""
        admission = pool_config.get("admission_policy")
        if admission:
            return cls.model_validate(admission)
        return default_policy()


def default_policy() -> AdmissionPolicy:
    """Return the default three-tier admission policy.

    Silver: minimum viable data
    Gold: well-curated
    Platinum: publication-grade
    """
    return AdmissionPolicy(
        tiers={
            QualityTier.SILVER: TierCriteria(
                min_channels=1,
                max_bad_channel_ratio=0.5,
                max_nan_ratio=0.1,
                require_labels=True,
                require_standard_channels=False,
                require_electrode_locations=False,
                require_processing_history=False,
                require_stimulus_ref=False,
                min_atoms=1,
                min_subjects=1,
            ),
            QualityTier.GOLD: TierCriteria(
                min_channels=8,
                max_bad_channel_ratio=0.2,
                max_nan_ratio=0.0,
                max_flatline_ratio=0.1,
                require_labels=True,
                require_standard_channels=True,
                require_electrode_locations=True,
                require_processing_history=False,
                require_stimulus_ref=False,
                min_atoms=10,
                min_subjects=1,
            ),
            QualityTier.PLATINUM: TierCriteria(
                min_channels=16,
                max_bad_channel_ratio=0.1,
                max_nan_ratio=0.0,
                max_flatline_ratio=0.05,
                require_labels=True,
                require_standard_channels=True,
                require_electrode_locations=True,
                require_processing_history=True,
                require_stimulus_ref=True,
                min_atoms=50,
                min_subjects=2,
            ),
        },
        reject_below_silver=False,
    )
