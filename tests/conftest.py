"""Shared test configuration.

Data directory resolution:
    All E2E tests resolve dataset paths from environment variables.
    If the env var is not set, tests that require real data are skipped.

    Set the following before running E2E tests:
        NEUROATOM_BCI_IV_2A_DIR   — BCI IV 2a .mat files (A01T.mat ... A09T.mat)
        NEUROATOM_PHYSIONET_DIR   — PhysioNet eegmmidb (S001/ ... S109/)
        NEUROATOM_KUL_DIR         — KUL AAD data (S1.mat ... S16.mat)
        NEUROATOM_DTU_DIR         — DTU AAD data (S1.mat ... S18.mat)
        NEUROATOM_SEEDV_DIR       — SEED-V data (root with EEG_raw_data/)
        NEUROATOM_ZUCO2_DIR       — Zuco 2.0 TSR data (Preprocessed/ dir)
        NEUROATOM_CCEP_DIR        — CCEP-COREG BIDS derivatives
"""

import os
from pathlib import Path

import pytest


def _data_dir(env_var: str) -> Path:
    """Resolve a data directory from an environment variable, or return None."""
    val = os.environ.get(env_var, "")
    if val:
        p = Path(val)
        if p.exists():
            return p
    return None


@pytest.fixture
def bci_data_dir():
    d = _data_dir("NEUROATOM_BCI_IV_2A_DIR")
    if d is None:
        pytest.skip("NEUROATOM_BCI_IV_2A_DIR not set or path does not exist")
    return d


@pytest.fixture
def physionet_data_dir():
    d = _data_dir("NEUROATOM_PHYSIONET_DIR")
    if d is None:
        pytest.skip("NEUROATOM_PHYSIONET_DIR not set or path does not exist")
    return d


@pytest.fixture
def kul_data_dir():
    d = _data_dir("NEUROATOM_KUL_DIR")
    if d is None:
        pytest.skip("NEUROATOM_KUL_DIR not set or path does not exist")
    return d


@pytest.fixture
def dtu_data_dir():
    d = _data_dir("NEUROATOM_DTU_DIR")
    if d is None:
        pytest.skip("NEUROATOM_DTU_DIR not set or path does not exist")
    return d


@pytest.fixture
def seedv_data_dir():
    d = _data_dir("NEUROATOM_SEEDV_DIR")
    if d is None:
        pytest.skip("NEUROATOM_SEEDV_DIR not set or path does not exist")
    return d


@pytest.fixture
def zuco2_data_dir():
    d = _data_dir("NEUROATOM_ZUCO2_DIR")
    if d is None:
        pytest.skip("NEUROATOM_ZUCO2_DIR not set or path does not exist")
    return d


@pytest.fixture
def ccep_data_dir():
    d = _data_dir("NEUROATOM_CCEP_DIR")
    if d is None:
        pytest.skip("NEUROATOM_CCEP_DIR not set or path does not exist")
    return d
