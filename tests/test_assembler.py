"""Tests for assembler pipeline modules + full assembly E2E."""

import numpy as np
import pytest

from neuroatom.assembler.channel_mapper import ChannelMapper
from neuroatom.assembler.filter import SignalFilter
from neuroatom.assembler.normalizer import (
    Normalizer,
    NormalizationStats,
    StatsCollector,
)
from neuroatom.assembler.padcrop import PadCrop
from neuroatom.assembler.resampler import Resampler
from neuroatom.assembler.rereferencer import Rereferencer
from neuroatom.assembler.unit_standardizer import UnitStandardizer
from neuroatom.core.enums import NormalizationMethod, NormalizationScope


# ---------------------------------------------------------------------------
# UnitStandardizer
# ---------------------------------------------------------------------------

class TestUnitStandardizer:
    def test_v_to_uv(self):
        std = UnitStandardizer(target_unit="uV")
        signal = np.ones((4, 100), dtype=np.float32) * 50e-6  # 50 µV in Volts
        result = std.convert(signal, "V")
        np.testing.assert_allclose(result, 50.0, atol=0.1)

    def test_mv_to_uv(self):
        std = UnitStandardizer(target_unit="uV")
        signal = np.ones((4, 100), dtype=np.float32) * 0.05  # 50 µV in mV
        result = std.convert(signal, "mV")
        np.testing.assert_allclose(result, 50.0, atol=0.1)

    def test_uv_noop(self):
        std = UnitStandardizer(target_unit="uV")
        signal = np.random.randn(4, 100).astype(np.float32)
        result = std.convert(signal, "uV")
        np.testing.assert_array_equal(result, signal)

    def test_unknown_unit_skip(self):
        std = UnitStandardizer(target_unit="uV")
        signal = np.ones((4, 100), dtype=np.float32)
        result = std.convert(signal, "weird_unit", error_handling="skip")
        np.testing.assert_array_equal(result, signal)

    def test_unknown_unit_raise(self):
        std = UnitStandardizer(target_unit="uV")
        signal = np.ones((4, 100), dtype=np.float32)
        with pytest.raises(ValueError):
            std.convert(signal, "weird_unit", error_handling="raise")


# ---------------------------------------------------------------------------
# Rereferencer
# ---------------------------------------------------------------------------

class TestRereferencer:
    def test_average_reference(self):
        rng = np.random.RandomState(42)
        signal = rng.randn(4, 100).astype(np.float32)
        ch_ids = ["C3", "Cz", "C4", "Pz"]

        reref = Rereferencer(target_reference="average")
        result = reref.apply(signal, ch_ids)

        # After average reference, mean across channels should be ~0
        assert abs(result.mean(axis=0).mean()) < 0.01

    def test_average_excludes_bad(self):
        signal = np.array([
            [100, 100, 100],  # "bad" channel with high values
            [10, 10, 10],
            [20, 20, 20],
        ], dtype=np.float32)

        reref = Rereferencer(target_reference="average", exclude_channels=["ch0"])
        result = reref.apply(signal, ["ch0", "ch1", "ch2"])

        # Reference = mean of ch1 and ch2 = 15
        np.testing.assert_allclose(result[1], [-5, -5, -5], atol=0.01)
        np.testing.assert_allclose(result[2], [5, 5, 5], atol=0.01)

    def test_single_channel_reference(self):
        signal = np.array([
            [10, 20, 30],
            [5, 10, 15],
            [0, 0, 0],
        ], dtype=np.float32)

        reref = Rereferencer(target_reference="Cz")
        result = reref.apply(signal, ["C3", "Cz", "C4"])

        # Cz row = [5, 10, 15], subtracted from all
        np.testing.assert_allclose(result[0], [5, 10, 15], atol=0.01)
        np.testing.assert_allclose(result[1], [0, 0, 0], atol=0.01)
        np.testing.assert_allclose(result[2], [-5, -10, -15], atol=0.01)


# ---------------------------------------------------------------------------
# ChannelMapper
# ---------------------------------------------------------------------------

class TestChannelMapper:
    def test_exact_mapping(self):
        mapper = ChannelMapper(target_channels=["C3", "Cz", "C4"])
        signal = np.random.randn(5, 100).astype(np.float32)
        source_map = {"C3": 0, "Cz": 1, "C4": 2, "Pz": 3, "Oz": 4}

        mapped, mask = mapper.apply(signal, source_map)
        assert mapped.shape == (3, 100)
        np.testing.assert_array_equal(mask, [1, 1, 1])
        np.testing.assert_array_equal(mapped[0], signal[0])

    def test_zero_fill_missing(self):
        mapper = ChannelMapper(target_channels=["C3", "Cz", "C4", "P3"])
        signal = np.random.randn(3, 100).astype(np.float32)
        source_map = {"C3": 0, "Cz": 1, "C4": 2}

        mapped, mask = mapper.apply(signal, source_map)
        assert mapped.shape == (4, 100)
        assert mask[3] == 0.0  # P3 is zero-filled
        np.testing.assert_array_equal(mapped[3], 0.0)

    def test_drop_strategy(self):
        mapper = ChannelMapper(
            target_channels=["C3", "MISSING"],
            missing_strategy="drop",
        )
        signal = np.random.randn(3, 100).astype(np.float32)
        source_map = {"C3": 0, "Cz": 1, "C4": 2}

        mapped, mask = mapper.apply(signal, source_map)
        assert mapped is None


# ---------------------------------------------------------------------------
# SignalFilter
# ---------------------------------------------------------------------------

class TestSignalFilter:
    def test_bandpass(self):
        rng = np.random.RandomState(42)
        srate = 256.0
        signal = rng.randn(4, 512).astype(np.float32)

        filt = SignalFilter(sampling_rate=srate, filter_band=(0.5, 40.0))
        result = filt.apply(signal)
        assert result.shape == signal.shape
        assert result.dtype == np.float32

    def test_notch(self):
        srate = 256.0
        t = np.arange(512) / srate
        noise_50hz = np.sin(2 * np.pi * 50 * t)
        signal = np.stack([noise_50hz] * 4).astype(np.float32)

        filt = SignalFilter(sampling_rate=srate, notch_freq=50.0)
        result = filt.apply(signal)
        # 50 Hz component should be attenuated
        assert np.std(result) < np.std(signal) * 0.5


# ---------------------------------------------------------------------------
# Resampler
# ---------------------------------------------------------------------------

class TestResampler:
    def test_downsample(self):
        resampler = Resampler(target_rate=128.0)
        signal = np.random.randn(4, 512).astype(np.float32)  # 256 Hz, 2 sec
        result = resampler.apply(signal, source_rate=256.0)
        assert result.shape == (4, 256)  # 128 Hz, 2 sec

    def test_upsample(self):
        resampler = Resampler(target_rate=512.0)
        signal = np.random.randn(4, 256).astype(np.float32)  # 256 Hz, 1 sec
        result = resampler.apply(signal, source_rate=256.0)
        assert result.shape == (4, 512)

    def test_same_rate_noop(self):
        resampler = Resampler(target_rate=256.0)
        signal = np.random.randn(4, 256).astype(np.float32)
        result = resampler.apply(signal, source_rate=256.0)
        np.testing.assert_array_equal(result, signal)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_zscore_per_atom(self):
        normalizer = Normalizer(
            method=NormalizationMethod.ZSCORE,
            scope=NormalizationScope.PER_ATOM,
        )
        signal = np.random.randn(4, 256).astype(np.float32) * 50 + 100
        result = normalizer.apply(signal)
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_robust_per_atom(self):
        normalizer = Normalizer(
            method=NormalizationMethod.ROBUST,
            scope=NormalizationScope.PER_ATOM,
        )
        signal = np.random.randn(4, 256).astype(np.float32) * 50 + 100
        result = normalizer.apply(signal)
        assert abs(np.median(result)) < 1.0

    def test_minmax_per_atom(self):
        normalizer = Normalizer(
            method=NormalizationMethod.MINMAX,
            scope=NormalizationScope.PER_ATOM,
        )
        signal = np.random.randn(4, 256).astype(np.float32) * 50 + 100
        result = normalizer.apply(signal)
        assert result.min() >= -0.01
        assert result.max() <= 1.01

    def test_per_channel(self):
        normalizer = Normalizer(
            method=NormalizationMethod.ZSCORE,
            scope=NormalizationScope.PER_CHANNEL,
        )
        signal = np.random.randn(4, 256).astype(np.float32) * 50 + 100
        result = normalizer.apply(signal)
        # Each channel should be approximately zero mean
        for ch in range(4):
            assert abs(result[ch].mean()) < 0.1

    def test_two_pass_global(self):
        rng = np.random.RandomState(42)
        n_channels = 4
        signals = [rng.randn(n_channels, 256).astype(np.float32) * 50 + 100 for _ in range(10)]

        # Pass 1: collect stats
        collector = StatsCollector(
            method=NormalizationMethod.ZSCORE,
            n_channels=n_channels,
            scope=NormalizationScope.GLOBAL,
        )
        for sig in signals:
            collector.update(sig, "__global__")
        stats = collector.finalize()

        # Pass 2: normalize
        normalizer = Normalizer(
            method=NormalizationMethod.ZSCORE,
            scope=NormalizationScope.GLOBAL,
            precomputed_stats=stats,
        )
        results = [normalizer.apply(sig, scope_key="__global__") for sig in signals]
        all_data = np.concatenate(results, axis=1)
        # Global mean should be ~0 per channel
        for ch in range(n_channels):
            assert abs(all_data[ch].mean()) < 0.5

    def test_stats_serialization(self):
        stats = NormalizationStats()
        stats.set_stats(
            "__global__",
            mean=np.array([1.0, 2.0]),
            std=np.array([0.5, 0.3]),
        )
        d = stats.to_dict()
        restored = NormalizationStats.from_dict(d)
        np.testing.assert_array_equal(
            restored.get_stats("__global__")["mean"],
            np.array([1.0, 2.0]),
        )


# ---------------------------------------------------------------------------
# PadCrop
# ---------------------------------------------------------------------------

class TestPadCrop:
    def test_pad_right(self):
        padcrop = PadCrop(target_samples=512)
        signal = np.random.randn(4, 256).astype(np.float32)
        result, mask = padcrop.apply(signal)
        assert result.shape == (4, 512)
        assert mask[:256].all()
        assert not mask[256:].any()

    def test_pad_left(self):
        padcrop = PadCrop(target_samples=512, pad_side="left")
        signal = np.random.randn(4, 256).astype(np.float32)
        result, mask = padcrop.apply(signal)
        assert result.shape == (4, 512)
        assert not mask[:256].any()
        assert mask[256:].all()

    def test_crop_right(self):
        padcrop = PadCrop(target_samples=128)
        signal = np.random.randn(4, 256).astype(np.float32)
        result, mask = padcrop.apply(signal)
        assert result.shape == (4, 128)
        np.testing.assert_array_equal(result, signal[:, :128])
        assert mask.all()

    def test_exact_length_noop(self):
        padcrop = PadCrop(target_samples=256)
        signal = np.random.randn(4, 256).astype(np.float32)
        result, mask = padcrop.apply(signal)
        np.testing.assert_array_equal(result, signal)
        assert mask.all()

    def test_compute_target_samples(self):
        assert PadCrop.compute_target_samples(2.0, 256.0) == 512
        assert PadCrop.compute_target_samples(0.5, 128.0) == 64


# ---------------------------------------------------------------------------
# Full assembly E2E
# ---------------------------------------------------------------------------

class TestFullAssemblyE2E:
    def test_assemble_mi_dataset(self, tmp_path):
        """Full assembly: import → index → assemble with recipe."""
        from neuroatom.assembler.dataset_assembler import DatasetAssembler
        from neuroatom.core.annotation import CategoricalAnnotation
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.dataset_meta import DatasetMeta
        from neuroatom.core.enums import AtomType, SplitStrategy
        from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
        from neuroatom.core.run import RunMeta
        from neuroatom.core.session import SessionMeta
        from neuroatom.core.signal_ref import SignalRef
        from neuroatom.core.subject import SubjectMeta
        from neuroatom.index.indexer import Indexer
        from neuroatom.storage import paths as P
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage.pool import Pool
        from neuroatom.storage.signal_store import ShardManager

        pool = Pool.create(tmp_path / "pool")
        pool_root = pool.root

        # Create 3 subjects with 4 trials each (2 left, 2 right)
        ds_id = "synth_mi"
        pool.register_dataset(DatasetMeta(
            dataset_id=ds_id, name="Synthetic MI",
            task_types=["motor_imagery"], n_subjects=3,
        ))

        rng = np.random.RandomState(42)
        for sub_idx in range(3):
            sub_id = f"sub-{sub_idx:02d}"
            ses_id = "ses-01"
            run_id = "run-01"

            pool.register_subject(SubjectMeta(subject_id=sub_id, dataset_id=ds_id))
            pool.register_session(SessionMeta(
                session_id=ses_id, subject_id=sub_id, dataset_id=ds_id,
                sampling_rate=256.0,
            ))
            pool.register_run(RunMeta(
                run_id=run_id, session_id=ses_id, subject_id=sub_id,
                dataset_id=ds_id, task_type="motor_imagery", run_index=0,
            ))

            with ShardManager(pool_root, ds_id, sub_id, ses_id, run_id) as mgr:
                jsonl_path = P.atoms_jsonl_path(pool_root, ds_id, sub_id, ses_id, run_id)
                with AtomJSONLWriter(jsonl_path) as writer:
                    for trial_idx in range(4):
                        label = "left_hand" if trial_idx % 2 == 0 else "right_hand"
                        atom_id = f"{ds_id}_{sub_id}_{trial_idx:03d}"
                        signal = rng.randn(8, 512).astype(np.float32) * 20e-6

                        ref = mgr.write_atom_signal(atom_id, signal)
                        atom = Atom(
                            atom_id=atom_id,
                            atom_type=AtomType.TRIAL,
                            dataset_id=ds_id,
                            subject_id=sub_id,
                            session_id=ses_id,
                            run_id=run_id,
                            trial_index=trial_idx,
                            signal_ref=ref,
                            temporal=TemporalInfo(
                                onset_sample=trial_idx * 1024,
                                onset_seconds=trial_idx * 4.0,
                                duration_samples=512,
                                duration_seconds=2.0,
                            ),
                            channel_ids=[f"ch_{i:03d}" for i in range(8)],
                            n_channels=8,
                            sampling_rate=256.0,
                            annotations=[
                                CategoricalAnnotation(
                                    annotation_id=f"ann_{atom_id}",
                                    name="mi_class",
                                    value=label,
                                ),
                            ],
                        )
                        writer.write_atom(atom)

        # Index
        indexer = Indexer(pool)
        indexer.reindex_all()

        # Assemble
        recipe = AssemblyRecipe(
            recipe_id="test_mi_assembly",
            query={"dataset_id": [ds_id]},
            target_unit="uV",
            normalization_method=NormalizationMethod.ZSCORE,
            normalization_scope=NormalizationScope.PER_ATOM,
            label_fields=[
                LabelSpec(annotation_name="mi_class", output_key="label"),
            ],
            split_strategy=SplitStrategy.SUBJECT,
            split_config={"seed": 42, "val_ratio": 0.33, "test_ratio": 0.33},
        )

        assembler = DatasetAssembler(pool, indexer)
        result = assembler.assemble(recipe, cache_dir=tmp_path / "cache")

        total = len(result.train_samples) + len(result.val_samples) + len(result.test_samples)
        assert total == 12  # 3 subjects × 4 trials

        # Check sample structure
        sample = result.train_samples[0] if result.train_samples else result.val_samples[0]
        assert "signal" in sample
        assert "labels" in sample
        assert sample["signal"].shape[0] == 8  # channels
        assert "label" in sample["labels"]

        # Check cache provenance
        assert (tmp_path / "cache" / "recipe.yaml").exists()
        assert (tmp_path / "cache" / "assembly_log.json").exists()

        indexer.close()
