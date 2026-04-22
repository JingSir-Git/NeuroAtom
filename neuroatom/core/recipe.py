"""AssemblyRecipe: declarative, serializable configuration for data assembly.

A Recipe fully specifies how to query, process, and output atoms from the
pool. Serialized to YAML for experiment reproducibility — every assembled
dataset can be traced back to its exact Recipe.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from neuroatom.core.enums import (
    ErrorHandling,
    NormalizationMethod,
    NormalizationScope,
    OutputFormat,
    SplitStrategy,
)


# ---------------------------------------------------------------------------
# Label specification (multi-task support)
# ---------------------------------------------------------------------------

class LabelSpec(BaseModel):
    """Specification for extracting one label from atom annotations.

    Multiple LabelSpecs enable multi-task learning (e.g., MI class +
    subject identity simultaneously).
    """

    annotation_name: str = Field(
        ...,
        description="Name of the annotation to extract (matched against annotation.name).",
    )
    output_key: str = Field(
        ...,
        description="Key in the output dictionary for this label (e.g. 'mi_class', 'subject_id').",
    )
    encoding: str = Field(
        default="auto",
        description="Encoding strategy: 'auto' (infer), 'ordinal', 'onehot', 'raw' (no encoding).",
    )
    label_mapping: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description=(
            "Unified label name -> list of aliases. "
            "Example: {'left_hand': ['left', 'L', 'class_1']}."
        ),
    )


# ---------------------------------------------------------------------------
# Typed augmentations (discriminated union)
# ---------------------------------------------------------------------------

class TemporalShiftAug(BaseModel):
    """Randomly shift the signal along the time axis."""
    type: Literal["temporal_shift"] = "temporal_shift"
    max_shift_seconds: float = Field(default=0.1, gt=0)

class ChannelDropoutAug(BaseModel):
    """Randomly zero out entire channels."""
    type: Literal["channel_dropout"] = "channel_dropout"
    drop_prob: float = Field(default=0.1, ge=0, le=1)

class GaussianNoiseAug(BaseModel):
    """Add Gaussian noise to the signal."""
    type: Literal["gaussian_noise"] = "gaussian_noise"
    std_uv: float = Field(default=1.0, gt=0, description="Noise std in µV.")

class SignalScaleAug(BaseModel):
    """Randomly scale the signal amplitude."""
    type: Literal["signal_scale"] = "signal_scale"
    scale_range: Tuple[float, float] = Field(default=(0.8, 1.2))

class TimeReversalAug(BaseModel):
    """Reverse the signal along the time axis with given probability."""
    type: Literal["time_reversal"] = "time_reversal"
    prob: float = Field(default=0.5, ge=0, le=1)

class FrequencyShiftAug(BaseModel):
    """Shift frequency content of the signal."""
    type: Literal["frequency_shift"] = "frequency_shift"
    max_shift_hz: float = Field(default=1.0, gt=0)


AugmentationUnion = Annotated[
    Union[
        Annotated[TemporalShiftAug, Tag("temporal_shift")],
        Annotated[ChannelDropoutAug, Tag("channel_dropout")],
        Annotated[GaussianNoiseAug, Tag("gaussian_noise")],
        Annotated[SignalScaleAug, Tag("signal_scale")],
        Annotated[TimeReversalAug, Tag("time_reversal")],
        Annotated[FrequencyShiftAug, Tag("frequency_shift")],
    ],
    Discriminator("type"),
]
"""Union type for all augmentation configs, discriminated on ``type``."""


# ---------------------------------------------------------------------------
# Assembly Recipe
# ---------------------------------------------------------------------------

class AssemblyRecipe(BaseModel):
    """Declarative configuration for assembling ML-ready datasets from the pool.

    Fully serializable to YAML. Determines which atoms to include, how to
    process them, how to extract labels, how to split, and what augmentations
    to apply.
    """

    model_config = ConfigDict(extra="forbid")

    recipe_id: str = Field(..., description="Unique recipe identifier.")
    description: Optional[str] = Field(default=None, description="Human-readable description.")

    # ---- Source selection ----
    query: Dict[str, Any] = Field(
        ...,
        description=(
            "Query specification for selecting atoms. See Query DSL documentation. "
            "All channel name matches use standard_name (post-alias-resolution). "
            "Annotation filters use existential semantics (at least one annotation matches)."
        ),
    )
    source_version: Optional[str] = Field(
        default=None,
        description=(
            "Match ProcessingHistory.version_tag. None = accept any version. "
            "Example: 'raw', 'ica_cleaned'."
        ),
    )

    # ---- Signal processing pipeline ----
    target_channels: Optional[List[str]] = Field(
        default=None,
        description="Target standard channel names for channel mapping. None = keep all.",
    )
    target_sampling_rate: Optional[float] = Field(
        default=None,
        gt=0,
        description="Target sampling rate in Hz. None = keep original.",
    )
    target_reference: Optional[str] = Field(
        default=None,
        description="Target reference scheme: 'average', 'linked_ears', 'Cz', 'REST'. None = keep original.",
    )
    target_duration: Optional[float] = Field(
        default=None,
        gt=0,
        description="Target duration in seconds. Atoms are padded or cropped to this length.",
    )
    target_unit: str = Field(
        default="uV",
        description="Target signal unit: 'V', 'mV', 'uV'.",
    )
    filter_band: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Bandpass filter cutoff frequencies (low_hz, high_hz).",
    )
    notch_freq: Optional[float] = Field(
        default=None,
        gt=0,
        description="Notch filter frequency in Hz (e.g. 50 or 60).",
    )
    normalization_method: Optional[NormalizationMethod] = Field(
        default=None,
        description="Normalization method: 'zscore', 'robust', 'minmax'. None = no normalization.",
    )
    normalization_scope: NormalizationScope = Field(
        default=NormalizationScope.PER_ATOM,
        description=(
            "Scope for normalization statistics: 'per_atom', 'per_subject', 'per_channel', 'global'. "
            "scope='global' or 'per_subject' triggers two-pass assembly."
        ),
    )
    baseline_correction: Optional[str] = Field(
        default=None,
        description="Baseline correction method: 'mean', 'median'. None = no baseline correction.",
    )
    baseline_before_normalize: bool = Field(
        default=True,
        description=(
            "If True (default, domain convention), baseline correction is applied before "
            "normalization. If False, normalization is applied first."
        ),
    )

    # ---- Labels (multi-task support) ----
    label_fields: List[LabelSpec] = Field(
        ...,
        min_length=1,
        description="Label extraction specifications. At least one required.",
    )

    # ---- Split ----
    split_strategy: SplitStrategy = Field(
        default=SplitStrategy.SUBJECT,
        description="Train/val/test split strategy.",
    )
    split_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters (e.g. test_subjects, val_ratio, seed).",
    )

    # ---- Augmentations ----
    augmentations: List[AugmentationUnion] = Field(
        default_factory=list,
        description="List of typed augmentation configs, applied in order during training.",
    )

    # ---- Output ----
    output_format: OutputFormat = Field(
        default=OutputFormat.CHANNELS_TIME,
        description="Tensor layout for ML framework.",
    )
    include_channel_mask: bool = Field(
        default=True,
        description="Include a binary mask indicating which channels are real vs zero-filled.",
    )
    include_time_mask: bool = Field(
        default=True,
        description="Include a binary mask indicating which time samples are real vs padded.",
    )

    # ---- Error handling ----
    error_handling: ErrorHandling = Field(
        default=ErrorHandling.SKIP,
        description=(
            "How to handle errors during assembly (atom loading, pipeline steps). "
            "DataLoader inherits this setting by default unless explicitly overridden."
        ),
    )

    # ------------------------------------------------------------------
    # Serialization: YAML ↔ Recipe for reproducible experiments
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path) -> "AssemblyRecipe":
        """Load a recipe from a YAML file.

        Usage::

            recipe = AssemblyRecipe.from_yaml("configs/mi_4class.yaml")
            result = assembler.assemble(recipe)

        Minimal YAML::

            recipe_id: my_experiment
            query:
              dataset_id: bci_comp_iv_2a
            label_fields:
              - annotation_name: mi_class
                output_key: mi_class

        The YAML file can be checked into version control for reproducibility.
        All optional fields have sensible defaults; only ``recipe_id``,
        ``query``, and ``label_fields`` are required.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML is empty, not a dict, or fails validation.
        """
        import yaml
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Recipe YAML not found: {path}. "
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
                f"A recipe needs at minimum: recipe_id, query, and label_fields."
            )
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a YAML mapping (dict) in {path}, got {type(data).__name__}. "
                f"Make sure the file starts with key-value pairs, not a list or scalar."
            )

        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"Recipe validation failed for {path}:\n{e}\n\n"
                f"Required fields: recipe_id, query, label_fields.\n"
                f"Run `recipe.to_yaml('example.yaml')` to see a valid example."
            ) from e

    def to_yaml(self, path) -> None:
        """Save this recipe to a YAML file for reproducibility.

        The output file includes all fields with their values, making the
        full pipeline configuration explicit and version-controllable.
        """
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
    def from_dict(cls, data: dict) -> "AssemblyRecipe":
        """Create a recipe from a plain dictionary."""
        return cls.model_validate(data)
