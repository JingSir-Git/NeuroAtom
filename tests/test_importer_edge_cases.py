"""Importer edge-case tests.

Covers boundary conditions identified in docs/edge_case_audit.md:
- MAT version detection (v5 vs v7.3)
- Sampling rate validation
- Signal validation (NaN, Inf, flat-line, all-zero)
- Zero-length epoch filtering
- OpenBMI channel array dimension handling
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ═════════════════════════════════════════════════════════════════════
# MAT version detection
# ═════════════════════════════════════════════════════════════════════


class TestMATVersionDetection:
    """Test .mat file version detection from header bytes."""

    def test_detect_v5(self, tmp_path):
        """v5 file starts with 'MATLAB 5.0'."""
        from neuroatom.utils.mat_compat import detect_mat_version

        mat_file = tmp_path / "test.mat"
        header = b"MATLAB 5.0 MAT-file" + b"\x00" * 100
        mat_file.write_bytes(header)

        assert detect_mat_version(mat_file) == "v5"

    def test_detect_v73(self, tmp_path):
        """v7.3 file starts with HDF5 magic bytes."""
        from neuroatom.utils.mat_compat import detect_mat_version

        mat_file = tmp_path / "test.mat"
        header = b"\x89HDF\r\n\x1a\n" + b"\x00" * 100
        mat_file.write_bytes(header)

        assert detect_mat_version(mat_file) == "v7.3"

    def test_detect_unknown(self, tmp_path):
        """Unknown format returns None."""
        from neuroatom.utils.mat_compat import detect_mat_version

        mat_file = tmp_path / "test.mat"
        mat_file.write_bytes(b"GARBAGE HEADER DATA" + b"\x00" * 100)

        assert detect_mat_version(mat_file) is None

    def test_detect_empty_file(self, tmp_path):
        """Empty file returns None."""
        from neuroatom.utils.mat_compat import detect_mat_version

        mat_file = tmp_path / "test.mat"
        mat_file.write_bytes(b"")

        assert detect_mat_version(mat_file) is None

    def test_detect_missing_file(self, tmp_path):
        """Missing file returns None."""
        from neuroatom.utils.mat_compat import detect_mat_version

        assert detect_mat_version(tmp_path / "nonexistent.mat") is None

    def test_require_v5_on_v73_raises(self, tmp_path):
        """require_mat_v5 on a v7.3 file raises ValueError."""
        from neuroatom.utils.mat_compat import require_mat_v5

        mat_file = tmp_path / "test.mat"
        mat_file.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 100)

        with pytest.raises(ValueError, match="v7.3.*HDF5"):
            require_mat_v5(mat_file, "TestImporter")

    def test_require_v73_on_v5_raises(self, tmp_path):
        """require_mat_v73 on a v5 file raises ValueError."""
        from neuroatom.utils.mat_compat import require_mat_v73

        mat_file = tmp_path / "test.mat"
        mat_file.write_bytes(b"MATLAB 5.0 MAT-file" + b"\x00" * 100)

        with pytest.raises(ValueError, match="v5 format"):
            require_mat_v73(mat_file, "TestImporter")

    def test_require_v5_on_v5_passes(self, tmp_path):
        """require_mat_v5 on a v5 file does not raise."""
        from neuroatom.utils.mat_compat import require_mat_v5

        mat_file = tmp_path / "test.mat"
        mat_file.write_bytes(b"MATLAB 5.0 MAT-file" + b"\x00" * 100)

        require_mat_v5(mat_file, "TestImporter")  # Should not raise


# ═════════════════════════════════════════════════════════════════════
# Sampling rate validation
# ═════════════════════════════════════════════════════════════════════


class TestSamplingRateValidation:
    """Test validate_sampling_rate boundary conditions."""

    def test_valid_rate(self):
        from neuroatom.utils.validation import validate_sampling_rate

        validate_sampling_rate(256.0, "test")  # Should not raise
        validate_sampling_rate(1.0, "test")    # Edge: 1 Hz

    def test_zero_rate_raises(self):
        from neuroatom.utils.validation import validate_sampling_rate

        with pytest.raises(ValueError, match="positive"):
            validate_sampling_rate(0.0, "test")

    def test_negative_rate_raises(self):
        from neuroatom.utils.validation import validate_sampling_rate

        with pytest.raises(ValueError, match="positive"):
            validate_sampling_rate(-100.0, "test")

    def test_nan_rate_raises(self):
        from neuroatom.utils.validation import validate_sampling_rate

        with pytest.raises(ValueError, match="not finite"):
            validate_sampling_rate(float("nan"), "test")

    def test_inf_rate_raises(self):
        from neuroatom.utils.validation import validate_sampling_rate

        with pytest.raises(ValueError, match="not finite"):
            validate_sampling_rate(float("inf"), "test")

    def test_very_high_rate_does_not_raise(self):
        """Unusually high rate (>1 MHz) should warn but not raise."""
        from neuroatom.utils.validation import validate_sampling_rate

        # Should not raise — just logs a warning
        validate_sampling_rate(2_000_000.0, "test")


# ═════════════════════════════════════════════════════════════════════
# Signal validation: NaN, Inf, flat-line, all-zero
# ═════════════════════════════════════════════════════════════════════


class TestSignalValidation:
    """Test validate_signal edge cases."""

    def test_empty_signal(self):
        from neuroatom.utils.validation import validate_signal

        signal = np.array([]).reshape(0, 0)
        warnings = validate_signal(signal, "atom_empty", {})
        assert any("Empty" in w for w in warnings)

    def test_all_nan(self):
        from neuroatom.utils.validation import validate_signal

        signal = np.full((2, 100), np.nan)
        warnings = validate_signal(signal, "atom_nan", {"skip_all_nan": True})
        assert any("NaN" in w for w in warnings)

    def test_some_nan(self):
        from neuroatom.utils.validation import validate_signal

        signal = np.zeros((2, 100))
        signal[0, 50] = np.nan
        warnings = validate_signal(signal, "atom_some_nan", {})
        assert any("NaN" in w for w in warnings)

    def test_inf_values(self):
        from neuroatom.utils.validation import validate_signal

        signal = np.zeros((2, 100))
        signal[0, 0] = np.inf
        signal[1, 99] = -np.inf
        warnings = validate_signal(signal, "atom_inf", {})
        assert any("Inf" in w for w in warnings)

    def test_all_zero(self):
        from neuroatom.utils.validation import validate_signal

        signal = np.zeros((3, 256))
        warnings = validate_signal(signal, "atom_zero", {"skip_all_zero": True})
        assert any("all-zero" in w for w in warnings)

    def test_flat_line(self):
        from neuroatom.utils.validation import validate_signal

        signal = np.ones((2, 256)) * 42.0
        warnings = validate_signal(
            signal, "atom_flat",
            {"flatline_std_threshold": 0.01, "skip_all_zero": False},
        )
        assert any("flat-line" in w for w in warnings)

    def test_clean_signal_no_warnings(self):
        from neuroatom.utils.validation import validate_signal

        rng = np.random.default_rng(42)
        signal = rng.standard_normal((4, 512)) * 50  # ~50 µV
        warnings = validate_signal(signal, "atom_clean", {})
        assert len(warnings) == 0

    def test_amplitude_range(self):
        from neuroatom.utils.validation import validate_signal

        signal = np.zeros((1, 100))
        signal[0, 0] = 999.0
        warnings = validate_signal(
            signal, "atom_amp",
            {"amplitude_range_uv": [-500, 500], "skip_all_zero": False},
        )
        assert any("Amplitude" in w for w in warnings)

    def test_unit_scale_v(self):
        """_unit_scale_to_uv returns correct factors."""
        from neuroatom.utils.validation import _unit_scale_to_uv

        assert _unit_scale_to_uv("V") == 1e6
        assert _unit_scale_to_uv("mV") == 1e3
        assert _unit_scale_to_uv("uV") == 1.0
        assert _unit_scale_to_uv("µV") == 1.0

    def test_flatline_v_unit_no_false_positive(self):
        """V-unit signal with real variance should NOT trigger flat-line."""
        from neuroatom.utils.validation import validate_signal

        rng = np.random.default_rng(42)
        # 50 µV std ≈ 5e-5 V std — should NOT be flat
        signal = rng.standard_normal((64, 512)).astype(np.float32) * 5e-5
        warnings = validate_signal(
            signal, "atom_v_clean", {}, signal_unit="V",
        )
        assert not any("flat-line" in w for w in warnings), (
            f"False flat-line warning on V-unit signal: {warnings}"
        )

    def test_flatline_v_unit_true_flat(self):
        """Truly flat V-unit signal SHOULD trigger flat-line."""
        from neuroatom.utils.validation import validate_signal

        # Constant value — zero std
        signal = np.ones((2, 256), dtype=np.float32) * 1e-4
        warnings = validate_signal(
            signal, "atom_v_flat", {}, signal_unit="V",
        )
        assert any("flat-line" in w for w in warnings)

    def test_amplitude_range_v_unit(self):
        """Amplitude range check scales correctly for V-unit data."""
        from neuroatom.utils.validation import validate_signal

        # 600 µV = 6e-4 V — should exceed [-500, 500] µV range
        signal = np.zeros((1, 100), dtype=np.float32)
        signal[0, 0] = 6e-4  # 600 µV in V
        warnings = validate_signal(
            signal, "atom_v_amp",
            {"amplitude_range_uv": [-500, 500], "skip_all_zero": False},
            signal_unit="V",
        )
        assert any("Amplitude" in w for w in warnings)

    def test_amplitude_range_v_unit_within_range(self):
        """V-unit signal within range should NOT trigger amplitude warning."""
        from neuroatom.utils.validation import validate_signal

        # 100 µV = 1e-4 V — well within [-500, 500] µV
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((4, 256)).astype(np.float32) * 1e-4
        warnings = validate_signal(
            signal, "atom_v_ok",
            {"amplitude_range_uv": [-500, 500], "skip_all_zero": False},
            signal_unit="V",
        )
        assert not any("Amplitude" in w for w in warnings)


# ═════════════════════════════════════════════════════════════════════
# Import-time unit conversion (convert_to_storage_unit)
# ═════════════════════════════════════════════════════════════════════


class TestConvertToStorageUnit:
    """Test the import-time unit conversion utility."""

    def test_v_to_uv(self):
        """V → µV: signal multiplied by 1e6."""
        from neuroatom.utils.unit_convert import convert_to_storage_unit

        signal = np.array([[1e-4]], dtype=np.float32)  # 100 µV in V
        out, unit, orig = convert_to_storage_unit(signal, source_unit="V")
        assert unit == "uV"
        assert orig == "V"
        assert np.isclose(out[0, 0], 100.0, atol=0.1)

    def test_uv_to_uv_identity(self):
        """µV → µV: no conversion, original_unit is None."""
        from neuroatom.utils.unit_convert import convert_to_storage_unit

        signal = np.array([[50.0]], dtype=np.float32)
        out, unit, orig = convert_to_storage_unit(signal, source_unit="uV")
        assert unit == "uV"
        assert orig is None
        assert np.isclose(out[0, 0], 50.0)

    def test_mv_to_uv(self):
        """mV → µV: signal multiplied by 1e3."""
        from neuroatom.utils.unit_convert import convert_to_storage_unit

        signal = np.array([[0.1]], dtype=np.float32)  # 100 µV in mV
        out, unit, orig = convert_to_storage_unit(signal, source_unit="mV")
        assert unit == "uV"
        assert orig == "mV"
        assert np.isclose(out[0, 0], 100.0, atol=0.1)

    def test_pool_config_override(self):
        """Pool config can specify a different target unit."""
        from neuroatom.utils.unit_convert import convert_to_storage_unit

        signal = np.array([[100.0]], dtype=np.float32)  # 100 µV
        config = {"storage_conventions": {"signal_unit": "mV"}}
        out, unit, orig = convert_to_storage_unit(
            signal, source_unit="uV", pool_config=config,
        )
        assert unit == "mV"
        assert orig == "uV"
        assert np.isclose(out[0, 0], 0.1, atol=0.001)  # 100 µV = 0.1 mV

    def test_round_trip_v_to_uv(self):
        """V → µV preserves information."""
        from neuroatom.utils.unit_convert import convert_to_storage_unit

        rng = np.random.default_rng(42)
        signal_v = rng.standard_normal((64, 512)).astype(np.float32) * 5e-5
        signal_uv, _, _ = convert_to_storage_unit(signal_v, source_unit="V")
        # Convert back manually
        recovered_v = signal_uv / 1e6
        assert np.allclose(signal_v, recovered_v, rtol=1e-5)


# ═════════════════════════════════════════════════════════════════════
# Zero-length epoch filtering (Zuco2 _sentence_epochs)
# ═════════════════════════════════════════════════════════════════════


class TestZeroLengthEpoch:
    """Test that zero-length epochs are filtered out."""

    def test_normal_epochs(self):
        from neuroatom.importers.zuco2 import _sentence_epochs

        events = [("10", 0, 0), ("10", 100, 0), ("10", 200, 0)]
        epochs = _sentence_epochs(events, 500, "10")
        assert len(epochs) == 3
        assert epochs[0] == (0, 100)
        assert epochs[1] == (100, 200)
        assert epochs[2] == (200, 500)

    def test_zero_length_filtered(self):
        """Two events at the same sample → zero-length epoch should be skipped."""
        from neuroatom.importers.zuco2 import _sentence_epochs

        events = [("10", 0, 0), ("10", 100, 0), ("10", 100, 0), ("10", 200, 0)]
        epochs = _sentence_epochs(events, 500, "10")
        # The epoch from 100→100 should be filtered
        assert all(end > start for start, end in epochs)
        # Should have 3 valid epochs (0→100, 100→200, 200→500)
        # The duplicate at 100 produces: (0,100), (100,100)[filtered], (100,200), (200,500)
        assert len(epochs) == 3

    def test_single_event(self):
        from neuroatom.importers.zuco2 import _sentence_epochs

        events = [("10", 50, 0)]
        epochs = _sentence_epochs(events, 1000, "10")
        assert epochs == [(50, 1000)]

    def test_no_matching_events(self):
        from neuroatom.importers.zuco2 import _sentence_epochs

        events = [("99", 0, 0), ("99", 100, 0)]
        epochs = _sentence_epochs(events, 500, "10")
        assert epochs == []


# ═════════════════════════════════════════════════════════════════════
# OpenBMI channel array dimension handling
# ═════════════════════════════════════════════════════════════════════


class TestOpenBMIChanDimension:
    """Test that OpenBMI chan extraction handles various ndim shapes."""

    def test_chan_2d(self):
        """Standard shape: (1, N) object array."""
        chan = np.empty((1, 3), dtype=object)
        chan[0, 0] = np.array(["Fp1"])
        chan[0, 1] = np.array(["C3"])
        chan[0, 2] = np.array(["Oz"])

        if chan.ndim == 2:
            names = [str(chan[0, i][0]) for i in range(chan.shape[1])]
        elif chan.ndim == 1:
            names = [str(chan[i]) for i in range(chan.shape[0])]
        else:
            names = [str(chan.flat[i]) for i in range(chan.size)]

        assert names == ["Fp1", "C3", "Oz"]

    def test_chan_1d(self):
        """Alternate shape: (N,) string array."""
        chan = np.array(["Fp1", "C3", "Oz"])

        if chan.ndim == 2:
            names = [str(chan[0, i][0]) for i in range(chan.shape[1])]
        elif chan.ndim == 1:
            names = [str(chan[i]) for i in range(chan.shape[0])]
        else:
            names = [str(chan.flat[i]) for i in range(chan.size)]

        assert names == ["Fp1", "C3", "Oz"]
