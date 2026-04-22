"""Unit tests for NeuroAtom utilities: hashing and channel name standardization."""

import pytest

from neuroatom.utils.hashing import (
    compute_atom_id,
    compute_content_hash,
    compute_processing_hash,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

class TestHashing:
    def test_atom_id_deterministic(self):
        id1 = compute_atom_id("ds", "s01", "ses-01", "run-01", 0, "raw")
        id2 = compute_atom_id("ds", "s01", "ses-01", "run-01", 0, "raw")
        assert id1 == id2
        assert len(id1) == 64  # SHA-256 hex

    def test_atom_id_differs_by_onset(self):
        id1 = compute_atom_id("ds", "s01", "ses-01", "run-01", 0, "raw")
        id2 = compute_atom_id("ds", "s01", "ses-01", "run-01", 256, "raw")
        assert id1 != id2

    def test_atom_id_differs_by_processing(self):
        id1 = compute_atom_id("ds", "s01", "ses-01", "run-01", 0, "raw")
        id2 = compute_atom_id("ds", "s01", "ses-01", "run-01", 0, "filtered")
        assert id1 != id2

    def test_atom_id_handles_none_session(self):
        id1 = compute_atom_id("ds", "s01", None, "run-01", 0)
        assert len(id1) == 64

    def test_processing_hash(self):
        h = compute_processing_hash('[{"op": "bandpass", "l_freq": 0.5}]')
        assert len(h) == 16  # truncated hash

    def test_content_hash(self):
        h1 = compute_content_hash(b"hello")
        h2 = compute_content_hash(b"hello")
        h3 = compute_content_hash(b"world")
        assert h1 == h2
        assert h1 != h3


# ---------------------------------------------------------------------------
# Channel Names
# ---------------------------------------------------------------------------

class TestChannelNames:
    def test_exact_match(self):
        from neuroatom.utils.channel_names import standardize_channel_name
        assert standardize_channel_name("Fp1") == "Fp1"
        assert standardize_channel_name("C3") == "C3"

    def test_case_insensitive(self):
        from neuroatom.utils.channel_names import standardize_channel_name
        assert standardize_channel_name("fp1") == "Fp1"
        assert standardize_channel_name("FP1") == "Fp1"
        assert standardize_channel_name("cz") == "Cz"
        assert standardize_channel_name("CZ") == "Cz"

    def test_strip_eeg_prefix(self):
        from neuroatom.utils.channel_names import standardize_channel_name
        assert standardize_channel_name("EEG Fp1") == "Fp1"
        assert standardize_channel_name("EEG-C3") == "C3"
        assert standardize_channel_name("EEG_Cz") == "Cz"

    def test_unknown_channel(self):
        from neuroatom.utils.channel_names import standardize_channel_name
        result = standardize_channel_name("RANDOM_CHANNEL_XYZ")
        assert result is None

    def test_batch_standardize(self):
        from neuroatom.utils.channel_names import standardize_channel_names
        raw = ["EEG Fp1", "C3", "cz", "RANDOM"]
        result = standardize_channel_names(raw)
        assert result["EEG Fp1"] == "Fp1"
        assert result["C3"] == "C3"
        assert result["cz"] == "Cz"
        assert result["RANDOM"] is None

    def test_get_standard_list(self):
        from neuroatom.utils.channel_names import get_standard_channel_list
        channels = get_standard_channel_list()
        assert "Fp1" in channels
        assert "Cz" in channels
        assert "O2" in channels
        assert len(channels) > 50  # Should have many channels
