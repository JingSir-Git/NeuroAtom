"""MultiModalRecipe: declarative configuration for paired multi-modal assembly.

Extends the single-modality AssemblyRecipe to support paired assembly of
multiple recording modalities (e.g., EEG + sEEG from CCEP-COREG, EEG + fMRI,
EEG + eye-tracking) where atoms from different modalities are linked by
user-specified pairing keys.

Pairing is fully configurable via ``pairing_keys``:

- **Run-level** (default): ``["subject_id", "session_id", "run_id"]``
  For CCEP-COREG where EEG and sEEG share the same run.
- **Trial-level**: ``["subject_id", "session_id", "run_id", "trial_index"]``
  For EEG + eye-tracking where each trial has a 1:1 correspondence.
- **Session-level**: ``["subject_id", "session_id"]``
  For EEG-fMRI where modalities share a session but not specific runs.
- **Custom**: Any combination of Atom fields or ``custom_fields`` keys.

Each modality gets its own processing pipeline (different sampling rates,
channel maps, filter bands), and samples are emitted as paired dicts::

    {"eeg": tensor_eeg, "ieeg": tensor_ieeg, "labels": labels_dict}
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from neuroatom.core.enums import (
    ErrorHandling,
    NormalizationMethod,
    NormalizationScope,
    SplitStrategy,
)
from neuroatom.core.recipe import AugmentationUnion, LabelSpec


class ModalityPipelineConfig(BaseModel):
    """Processing pipeline configuration for a single modality."""

    model_config = ConfigDict(extra="forbid")

    query: Dict[str, Any] = Field(
        ...,
        description="Query dict for selecting atoms of this modality.",
    )
    target_channels: Optional[List[str]] = Field(
        default=None,
        description="Target channel names for channel mapping.",
    )
    target_sampling_rate: Optional[float] = Field(
        default=None, gt=0,
        description="Target sampling rate in Hz.",
    )
    target_reference: Optional[str] = Field(
        default=None,
        description="Reference scheme for this modality.",
    )
    target_duration: Optional[float] = Field(
        default=None, gt=0,
        description="Target duration in seconds for pad/crop.",
    )
    target_unit: str = Field(
        default="uV",
        description="Target signal unit.",
    )
    filter_band: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Bandpass filter (low_hz, high_hz).",
    )
    notch_freq: Optional[float] = Field(
        default=None, gt=0,
        description="Notch filter frequency.",
    )
    normalization_method: Optional[NormalizationMethod] = Field(default=None)
    normalization_scope: NormalizationScope = Field(
        default=NormalizationScope.PER_ATOM,
    )
    baseline_correction: Optional[str] = Field(default=None)
    baseline_before_normalize: bool = Field(default=True)


class MultiModalRecipe(BaseModel):
    """Recipe for paired multi-modal assembly.

    Each modality is processed through its own pipeline, then samples are
    paired at the run level using ``AtomRelation`` cross-modal links.

    Unknown fields are rejected (``extra='forbid'``) to catch typos early.

    Example YAML::

        recipe_id: "ccep_paired"
        modalities:
          eeg:
            query: {dataset_id: "ccepcoreg", modality: "eeg"}
            target_sampling_rate: 1000
            filter_band: [0.5, 45.0]
            target_reference: "average"
          ieeg:
            query: {dataset_id: "ccepcoreg", modality: "ieeg"}
            target_sampling_rate: 1000
            filter_band: [0.5, 300.0]
        pairing_keys: ["subject_id", "session_id", "run_id"]
        label_fields:
          - annotation_name: "stim_type"
            output_key: "stim_type"

    Trial-level pairing (EEG + eye-tracking)::

        pairing_keys: ["subject_id", "session_id", "run_id", "trial_index"]

    Session-level pairing (EEG-fMRI)::

        pairing_keys: ["subject_id", "session_id"]
    """

    model_config = ConfigDict(extra="forbid")

    recipe_id: str = Field(..., description="Unique recipe identifier.")
    description: Optional[str] = Field(default=None)

    modalities: Dict[str, ModalityPipelineConfig] = Field(
        ...,
        min_length=2,
        description="Per-modality pipeline configs. Keys are modality names (e.g. 'eeg', 'ieeg').",
    )

    pairing_keys: List[str] = Field(
        default=["subject_id", "session_id", "run_id"],
        description=(
            "Atom fields used to build the pairing key. Atoms from different "
            "modalities that share the same pairing key are grouped together. "
            "Default: ['subject_id', 'session_id', 'run_id'] (run-level pairing). "
            "Examples: ['subject_id', 'session_id'] for session-level, "
            "['subject_id', 'session_id', 'run_id', 'trial_index'] for trial-level. "
            "Any Atom field name or custom_fields key is valid."
        ),
    )

    label_fields: List[LabelSpec] = Field(
        ..., min_length=1,
        description="Label specs applied to the primary modality's atoms.",
    )
    primary_modality: Optional[str] = Field(
        default=None,
        description=(
            "Which modality's labels to use. If None, uses the first modality "
            "in the dict. Labels are extracted from the primary modality's atoms."
        ),
    )

    split_strategy: SplitStrategy = Field(default=SplitStrategy.SUBJECT)
    split_config: Dict[str, Any] = Field(default_factory=dict)

    augmentations: List[AugmentationUnion] = Field(default_factory=list)
    error_handling: ErrorHandling = Field(default=ErrorHandling.SKIP)

    include_channel_mask: bool = Field(default=True)
    include_time_mask: bool = Field(default=True)

    # ------------------------------------------------------------------
    # Serialization: YAML ↔ Recipe for reproducible experiments
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path) -> "MultiModalRecipe":
        """Load a multi-modal recipe from a YAML file.

        Usage::

            recipe = MultiModalRecipe.from_yaml("configs/ccep_paired.yaml")
            result = MultiModalAssembler(pool, indexer).assemble(recipe)

        Minimal YAML::

            recipe_id: my_paired
            modalities:
              eeg:
                query: {dataset_id: ccepcoreg, modality: eeg}
              ieeg:
                query: {dataset_id: ccepcoreg, modality: ieeg}
            label_fields:
              - annotation_name: stim_type
                output_key: stim_type

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML is empty, not a dict, or fails validation.
        """
        import yaml
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"MultiModalRecipe YAML not found: {path}. "
                f"Create one with recipe.to_yaml('{path}')."
            )

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Failed to parse YAML file {path}: {e}\n"
                    f"Check for indentation errors, missing colons, or invalid syntax."
                ) from e

        if data is None:
            raise ValueError(
                f"YAML file is empty: {path}. "
                f"A multi-modal recipe needs at minimum: recipe_id, modalities (2+), "
                f"and label_fields."
            )
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a YAML mapping (dict) in {path}, got {type(data).__name__}."
            )

        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"MultiModalRecipe validation failed for {path}:\n{e}\n\n"
                f"Required: recipe_id, modalities (2+ entries), label_fields.\n"
                f"Run `recipe.to_yaml('example.yaml')` to see a valid example."
            ) from e

    def to_yaml(self, path) -> None:
        """Save this recipe to a YAML file for reproducibility."""
        import yaml
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    @classmethod
    def from_dict(cls, data: dict) -> "MultiModalRecipe":
        """Create a recipe from a plain dictionary."""
        return cls.model_validate(data)
