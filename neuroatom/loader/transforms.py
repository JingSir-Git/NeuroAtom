"""Runtime augmentation transforms for EEG data.

Each transform class corresponds to one AugmentationUnion type from
core/recipe.py. Transforms operate on sample dicts (as returned by
AtomDataset.__getitem__) with key 'signal' being an np.ndarray of
shape (n_channels, n_samples).

All transforms are:
- Stochastic (probability-gated and/or randomized)
- Signal-preserving (no NaN/inf introduction)
- Channel-mask-aware (zero-filled channels are excluded from augmentation)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from neuroatom.core.recipe import (
    AugmentationUnion,
    ChannelDropoutAug,
    FrequencyShiftAug,
    GaussianNoiseAug,
    SignalScaleAug,
    TemporalShiftAug,
    TimeReversalAug,
)

logger = logging.getLogger(__name__)


class TemporalShiftTransform:
    """Randomly shift signal along the time axis with circular wrapping.

    Preserves signal energy by using np.roll instead of zero-padding.
    """

    def __init__(self, config: TemporalShiftAug, sampling_rate: float):
        self._max_shift_samples = int(config.max_shift_seconds * sampling_rate)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        signal = sample["signal"]
        shift = np.random.randint(-self._max_shift_samples, self._max_shift_samples + 1)
        if shift != 0:
            sample["signal"] = np.roll(signal, shift, axis=1)
            # Update time mask if present (shifted samples at boundary are padded)
            if sample.get("time_mask") is not None:
                sample["time_mask"] = np.roll(sample["time_mask"], shift)
        return sample


class ChannelDropoutTransform:
    """Randomly zero out entire channels.

    Respects channel_mask: only drops channels that are real (mask=1).
    Updates channel_mask to reflect dropped channels.
    """

    def __init__(self, config: ChannelDropoutAug):
        self._drop_prob = config.drop_prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        signal = sample["signal"]
        n_channels = signal.shape[0]

        # Only drop channels that are real
        channel_mask = sample.get("channel_mask")
        if channel_mask is not None:
            active = np.where(channel_mask > 0)[0]
        else:
            active = np.arange(n_channels)

        if len(active) == 0:
            return sample

        # Generate drop mask
        drop_mask = np.random.random(len(active)) < self._drop_prob
        drop_indices = active[drop_mask]

        if len(drop_indices) > 0:
            signal = signal.copy()
            signal[drop_indices] = 0.0
            sample["signal"] = signal

            if channel_mask is not None:
                channel_mask = channel_mask.copy()
                channel_mask[drop_indices] = 0.0
                sample["channel_mask"] = channel_mask

        return sample


class GaussianNoiseTransform:
    """Add Gaussian noise to the signal.

    Noise is added only to real channels (channel_mask=1).
    """

    def __init__(self, config: GaussianNoiseAug):
        self._std = config.std_uv

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        signal = sample["signal"]
        noise = np.random.randn(*signal.shape).astype(signal.dtype) * self._std

        # Mask noise for zero-filled channels
        channel_mask = sample.get("channel_mask")
        if channel_mask is not None:
            noise *= channel_mask[:, np.newaxis]

        # Mask noise for padded time samples
        time_mask = sample.get("time_mask")
        if time_mask is not None:
            noise *= time_mask[np.newaxis, :]

        sample["signal"] = signal + noise
        return sample


class SignalScaleTransform:
    """Randomly scale signal amplitude.

    Scale factor is drawn uniformly from [scale_min, scale_max].
    """

    def __init__(self, config: SignalScaleAug):
        self._min_scale, self._max_scale = config.scale_range

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        scale = np.random.uniform(self._min_scale, self._max_scale)
        sample["signal"] = sample["signal"] * scale
        return sample


class TimeReversalTransform:
    """Reverse signal along the time axis with given probability.

    Time reversal is a simple but effective EEG augmentation that
    preserves frequency content while disrupting temporal structure.
    """

    def __init__(self, config: TimeReversalAug):
        self._prob = config.prob

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() < self._prob:
            sample["signal"] = np.ascontiguousarray(sample["signal"][:, ::-1])
            if sample.get("time_mask") is not None:
                sample["time_mask"] = np.ascontiguousarray(
                    sample["time_mask"][::-1]
                )
        return sample


class FrequencyShiftTransform:
    """Shift frequency content of the signal.

    Implements frequency shifting via modulation with a complex exponential
    (SSB modulation), keeping only the real part. This shifts all spectral
    components by max_shift_hz (randomly chosen per sample).
    """

    def __init__(self, config: FrequencyShiftAug, sampling_rate: float):
        self._max_shift = config.max_shift_hz
        self._srate = sampling_rate

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        signal = sample["signal"]
        n_channels, n_samples = signal.shape

        shift_hz = np.random.uniform(-self._max_shift, self._max_shift)
        t = np.arange(n_samples) / self._srate
        modulator = np.cos(2 * np.pi * shift_hz * t).astype(signal.dtype)

        sample["signal"] = signal * modulator[np.newaxis, :]
        return sample


# ---------------------------------------------------------------------------
# Factory: build transforms from AugmentationUnion configs
# ---------------------------------------------------------------------------

def build_transforms(
    aug_configs: List[AugmentationUnion],
    sampling_rate: float,
) -> List:
    """Build a list of callable transform objects from recipe augmentation configs.

    Args:
        aug_configs: List of AugmentationUnion from AssemblyRecipe.augmentations.
        sampling_rate: Sampling rate of the assembled data (after resampling).

    Returns:
        List of callable transforms.
    """
    transforms = []
    for config in aug_configs:
        if isinstance(config, TemporalShiftAug):
            transforms.append(TemporalShiftTransform(config, sampling_rate))
        elif isinstance(config, ChannelDropoutAug):
            transforms.append(ChannelDropoutTransform(config))
        elif isinstance(config, GaussianNoiseAug):
            transforms.append(GaussianNoiseTransform(config))
        elif isinstance(config, SignalScaleAug):
            transforms.append(SignalScaleTransform(config))
        elif isinstance(config, TimeReversalAug):
            transforms.append(TimeReversalTransform(config))
        elif isinstance(config, FrequencyShiftAug):
            transforms.append(FrequencyShiftTransform(config, sampling_rate))
        else:
            logger.warning("Unknown augmentation type: %s", type(config).__name__)

    logger.info("Built %d transforms.", len(transforms))
    return transforms


class ComposeTransforms:
    """Compose multiple transforms into a single callable.

    Usage:
        transform = ComposeTransforms(build_transforms(aug_configs, srate))
        sample = transform(sample)
    """

    def __init__(self, transforms: List):
        self._transforms = transforms

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self._transforms:
            sample = transform(sample)
        return sample

    def __len__(self):
        return len(self._transforms)
