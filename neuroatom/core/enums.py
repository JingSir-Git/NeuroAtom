"""Shared enumerations for NeuroAtom data models."""

from enum import Enum


class AtomType(str, Enum):
    """Type of atomic data unit."""
    TRIAL = "trial"
    EVENT_EPOCH = "event_epoch"
    WINDOW = "window"
    CONTINUOUS_SEGMENT = "continuous_segment"


class ChannelType(str, Enum):
    """Type of recording channel."""
    EEG = "eeg"
    SEEG = "seeg"
    ECOG = "ecog"
    EOG = "eog"
    EMG = "emg"
    ECG = "ecg"
    MISC = "misc"
    TRIGGER = "trigger"
    STIM = "stim"
    REF = "ref"
    OTHER = "other"


class AnnotationType(str, Enum):
    """Discriminator for annotation subtypes."""
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    TEXT = "text"
    CONTINUOUS = "continuous"
    EVENT_SEQUENCE = "event_sequence"
    STIMULUS_REF = "stimulus_ref"
    BINARY_MASK = "binary_mask"


class QualityStatus(str, Enum):
    """Overall quality status of an atom."""
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    UNKNOWN = "unknown"
    REJECTED = "rejected"


class ChannelStatus(str, Enum):
    """Status of a single channel."""
    GOOD = "good"
    BAD = "bad"
    UNKNOWN = "unknown"


class NormalizationMethod(str, Enum):
    """Normalization method for assembly."""
    ZSCORE = "zscore"
    ROBUST = "robust"
    MINMAX = "minmax"


class NormalizationScope(str, Enum):
    """Scope at which normalization statistics are computed."""
    PER_ATOM = "per_atom"
    PER_SUBJECT = "per_subject"
    PER_CHANNEL = "per_channel"
    GLOBAL = "global"


class SplitStrategy(str, Enum):
    """Train/val/test split strategy."""
    SUBJECT = "subject"
    DATASET = "dataset"
    TEMPORAL = "temporal"
    PREDEFINED = "predefined"
    STRATIFIED = "stratified"


class OutputFormat(str, Enum):
    """Tensor output format for ML frameworks."""
    CHANNELS_TIME = "channels_time"      # (B, C, T) for CNN
    TIME_CHANNELS = "time_channels"      # (B, T, C) for Transformer
    PATCHES = "patches"                  # (B, N_patches, patch_dim) for ViT-style


class ErrorHandling(str, Enum):
    """Error handling strategy."""
    RAISE = "raise"
    SKIP = "skip"
    SUBSTITUTE = "substitute"


class ReferenceScheme(str, Enum):
    """EEG reference scheme."""
    AVERAGE = "average"
    LINKED_EARS = "linked_ears"
    CZ = "Cz"
    REST = "REST"
    MONOPOLAR = "monopolar"
    CUSTOM = "custom"
