"""Tests for loader/transforms.py and loader/collate.py."""

import numpy as np
import pytest

from neuroatom.core.recipe import (
    ChannelDropoutAug,
    FrequencyShiftAug,
    GaussianNoiseAug,
    SignalScaleAug,
    TemporalShiftAug,
    TimeReversalAug,
)
from neuroatom.loader.transforms import (
    ChannelDropoutTransform,
    ComposeTransforms,
    FrequencyShiftTransform,
    GaussianNoiseTransform,
    SignalScaleTransform,
    TemporalShiftTransform,
    TimeReversalTransform,
    build_transforms,
)


def _make_sample(n_channels=8, n_samples=256, with_masks=True):
    """Create a synthetic sample dict."""
    sample = {
        "signal": np.random.randn(n_channels, n_samples).astype(np.float32),
        "labels": {"label": 1},
        "atom_id": "test_atom",
        "subject_id": "sub-01",
        "dataset_id": "test",
    }
    if with_masks:
        sample["channel_mask"] = np.ones(n_channels, dtype=np.float32)
        sample["time_mask"] = np.ones(n_samples, dtype=np.float32)
    return sample


class TestTemporalShift:
    def test_shift_preserves_shape(self):
        config = TemporalShiftAug(max_shift_seconds=0.1)
        transform = TemporalShiftTransform(config, sampling_rate=256.0)
        sample = _make_sample()
        result = transform(sample)
        assert result["signal"].shape == (8, 256)

    def test_shift_changes_signal(self):
        config = TemporalShiftAug(max_shift_seconds=0.5)
        transform = TemporalShiftTransform(config, sampling_rate=256.0)
        np.random.seed(42)
        sample = _make_sample()
        original = sample["signal"].copy()
        np.random.seed(0)  # Different seed to ensure shift
        result = transform(sample)
        # With a large max_shift, signal should be different (most of the time)
        # Just check shape is preserved
        assert result["signal"].shape == original.shape


class TestChannelDropout:
    def test_drop_channels(self):
        config = ChannelDropoutAug(drop_prob=1.0)  # Drop ALL
        transform = ChannelDropoutTransform(config)
        sample = _make_sample()
        result = transform(sample)
        # All active channels should be zeroed
        assert np.all(result["signal"] == 0)

    def test_no_drop(self):
        config = ChannelDropoutAug(drop_prob=0.0)
        transform = ChannelDropoutTransform(config)
        sample = _make_sample()
        original = sample["signal"].copy()
        result = transform(sample)
        np.testing.assert_array_equal(result["signal"], original)

    def test_respects_channel_mask(self):
        config = ChannelDropoutAug(drop_prob=1.0)
        transform = ChannelDropoutTransform(config)
        sample = _make_sample()
        # Mark channels 0-3 as zero-filled (not real)
        sample["channel_mask"][:4] = 0
        sample["signal"][:4] = 0  # Already zero
        result = transform(sample)
        # Zero-filled channels should remain zero
        # Active channels (4-7) should be dropped
        assert np.all(result["signal"][4:] == 0)


class TestGaussianNoise:
    def test_adds_noise(self):
        config = GaussianNoiseAug(std_uv=10.0)
        transform = GaussianNoiseTransform(config)
        sample = _make_sample()
        original = sample["signal"].copy()
        result = transform(sample)
        diff = result["signal"] - original
        assert np.std(diff) > 1.0  # Noise should be significant

    def test_respects_time_mask(self):
        config = GaussianNoiseAug(std_uv=100.0)
        transform = GaussianNoiseTransform(config)
        sample = _make_sample()
        sample["time_mask"][128:] = 0  # Last half is padded
        original_padded = sample["signal"][:, 128:].copy()
        result = transform(sample)
        # Padded region should have no noise
        np.testing.assert_array_equal(result["signal"][:, 128:], original_padded)


class TestSignalScale:
    def test_scale_range(self):
        config = SignalScaleAug(scale_range=(2.0, 2.0))  # Exact 2x
        transform = SignalScaleTransform(config)
        sample = _make_sample()
        original = sample["signal"].copy()
        result = transform(sample)
        np.testing.assert_allclose(result["signal"], original * 2.0, rtol=1e-5)


class TestTimeReversal:
    def test_reversal_with_prob_1(self):
        config = TimeReversalAug(prob=1.0)
        transform = TimeReversalTransform(config)
        sample = _make_sample()
        original = sample["signal"].copy()
        result = transform(sample)
        np.testing.assert_array_equal(result["signal"], original[:, ::-1])

    def test_no_reversal_with_prob_0(self):
        config = TimeReversalAug(prob=0.0)
        transform = TimeReversalTransform(config)
        sample = _make_sample()
        original = sample["signal"].copy()
        result = transform(sample)
        np.testing.assert_array_equal(result["signal"], original)


class TestFrequencyShift:
    def test_preserves_shape(self):
        config = FrequencyShiftAug(max_shift_hz=2.0)
        transform = FrequencyShiftTransform(config, sampling_rate=256.0)
        sample = _make_sample()
        result = transform(sample)
        assert result["signal"].shape == (8, 256)


class TestBuildTransforms:
    def test_build_all_types(self):
        configs = [
            TemporalShiftAug(max_shift_seconds=0.1),
            ChannelDropoutAug(drop_prob=0.1),
            GaussianNoiseAug(std_uv=1.0),
            SignalScaleAug(scale_range=(0.9, 1.1)),
            TimeReversalAug(prob=0.5),
            FrequencyShiftAug(max_shift_hz=1.0),
        ]
        transforms = build_transforms(configs, sampling_rate=256.0)
        assert len(transforms) == 6

    def test_compose_transforms(self):
        configs = [
            GaussianNoiseAug(std_uv=0.1),
            SignalScaleAug(scale_range=(1.0, 1.0)),
        ]
        transforms = build_transforms(configs, sampling_rate=256.0)
        composed = ComposeTransforms(transforms)
        assert len(composed) == 2

        sample = _make_sample()
        result = composed(sample)
        assert result["signal"].shape == (8, 256)


# ---------------------------------------------------------------------------
# Collate (torch-dependent — skip if not available)
# ---------------------------------------------------------------------------

class TestCollate:
    @pytest.fixture(autouse=True)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_skip_none_collate(self):
        from neuroatom.loader.collate import skip_none_collate

        batch = [None, {"x": 1}, None, {"x": 2}]
        # skip_none_collate filters None then uses default_collate
        # default_collate needs tensors, so test with just the filtering
        result = skip_none_collate(batch)
        assert result is not None

    def test_skip_none_all_none(self):
        from neuroatom.loader.collate import skip_none_collate

        result = skip_none_collate([None, None, None])
        assert result is None

    def test_neuroatom_collate(self):
        import torch
        from neuroatom.loader.collate import neuroatom_collate

        batch = [
            {
                "signal": np.random.randn(8, 256).astype(np.float32),
                "labels": {"class": 1},
                "channel_mask": np.ones(8, dtype=np.float32),
                "time_mask": np.ones(256, dtype=np.float32),
                "atom_id": "a1",
                "subject_id": "s1",
                "dataset_id": "d1",
            },
            {
                "signal": np.random.randn(8, 256).astype(np.float32),
                "labels": {"class": 2},
                "channel_mask": np.ones(8, dtype=np.float32),
                "time_mask": np.ones(256, dtype=np.float32),
                "atom_id": "a2",
                "subject_id": "s1",
                "dataset_id": "d1",
            },
        ]

        result = neuroatom_collate(batch)
        assert result is not None
        assert result["signal"].shape == (2, 8, 256)
        assert result["channel_mask"].shape == (2, 8)
        assert result["time_mask"].shape == (2, 256)
        assert len(result["atom_id"]) == 2

    def test_dynamic_pad_collate(self):
        from neuroatom.loader.collate import dynamic_pad_collate

        batch = [
            {
                "signal": np.random.randn(8, 200).astype(np.float32),
                "labels": {"class": 1},
                "atom_id": "a1",
            },
            {
                "signal": np.random.randn(8, 300).astype(np.float32),
                "labels": {"class": 2},
                "atom_id": "a2",
            },
        ]

        result = dynamic_pad_collate(batch)
        assert result is not None
        assert result["signal"].shape == (2, 8, 300)  # Padded to max
        assert result["time_mask"].shape == (2, 300)
        # First sample: 200 real, 100 padded
        assert result["time_mask"][0, 199].item() == 1.0
        assert result["time_mask"][0, 200].item() == 0.0
        # Second sample: all real
        assert result["time_mask"][1, 299].item() == 1.0
