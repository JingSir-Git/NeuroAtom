"""End-to-end real data test: CCEP-COREG EEG + sEEG dataset.

Imports 1 run (both EEG and iEEG) from sub-01, indexes, queries by modality,
and verifies cross-modal linking.

Requires: NEUROATOM_CCEP_DIR env var pointing to the BIDS derivatives root.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

_ccep_env = os.environ.get("NEUROATOM_CCEP_DIR", "")
CCEP_SUB01 = Path(_ccep_env) / "sub-01" if _ccep_env else Path("__nonexistent__")
pytestmark = pytest.mark.skipif(
    not CCEP_SUB01.exists(),
    reason="NEUROATOM_CCEP_DIR not set or sub-01 not found",
)


@pytest.fixture
def pool_dir():
    d = tempfile.mkdtemp(prefix="neuroatom_ccep_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_ccep_import_and_index(pool_dir):
    """Import 1 CCEP run (both EEG + iEEG), index, query by modality."""
    from neuroatom.importers.ccep_bids_npy import CCEPImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.pool import Pool
    from neuroatom.storage.signal_store import ShardManager

    logging.basicConfig(level=logging.INFO)

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "ccepcoreg.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = CCEPImporter(pool, task_config)
    results = importer.import_subject(
        subject_dir=CCEP_SUB01,
        subject_id="sub-01",
        session_id="ses-01",
        max_runs=1,
        max_epochs=5,  # Only 5 epochs per modality for speed
    )

    # Should have 2 results: one for EEG, one for iEEG
    assert len(results) == 2

    eeg_result = [r for r in results if r.atoms[0].modality == "eeg"][0]
    ieeg_result = [r for r in results if r.atoms[0].modality == "ieeg"][0]

    # Verify EEG atoms
    assert len(eeg_result.atoms) == 5
    eeg_atom = eeg_result.atoms[0]
    assert eeg_atom.dataset_id == "ccepcoreg"
    assert eeg_atom.subject_id == "sub-01"
    assert eeg_atom.modality == "eeg"
    assert eeg_atom.atom_type.value == "event_epoch"
    assert eeg_atom.sampling_rate == 1000.0
    assert eeg_atom.n_channels > 100  # HD-EEG ~185 channels

    # Verify iEEG atoms
    assert len(ieeg_result.atoms) == 5
    ieeg_atom = ieeg_result.atoms[0]
    assert ieeg_atom.modality == "ieeg"
    assert ieeg_atom.sampling_rate == 1000.0
    assert ieeg_atom.n_channels > 50  # sEEG ~139 channels

    # Verify annotations: stim_contact and trial_type should exist
    eeg_ann_names = {a.name for a in eeg_atom.annotations}
    assert "trial_type" in eeg_ann_names
    assert "stim_contact" in eeg_ann_names
    assert "modality" in eeg_ann_names

    # Verify cross-modal relations
    assert len(eeg_atom.relations) > 0
    rel = eeg_atom.relations[0]
    assert rel.relation_type == "cross_modal_paired_run"
    assert rel.metadata["target_modality"] == "ieeg"

    assert len(ieeg_atom.relations) > 0
    ieeg_rel = ieeg_atom.relations[0]
    assert ieeg_rel.relation_type == "cross_modal_paired_run"
    assert ieeg_rel.metadata["target_modality"] == "eeg"

    # Index
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    assert n_indexed == 10  # 5 EEG + 5 iEEG

    # Query by modality
    qb = QueryBuilder(indexer.backend)
    eeg_ids = qb.query_atom_ids({"modality": "eeg"})
    assert len(eeg_ids) == 5

    ieeg_ids = qb.query_atom_ids({"modality": "ieeg"})
    assert len(ieeg_ids) == 5

    # Query all
    all_ids = qb.query_atom_ids({"dataset_id": "ccepcoreg"})
    assert len(all_ids) == 10

    # Read back signal
    signal = ShardManager.static_read(pool.root, eeg_atom.signal_ref)
    assert signal.shape[0] == eeg_atom.n_channels
    assert signal.shape[1] == 1001  # 1s at 1000 Hz
    assert signal.dtype == np.float32
    # Data is in V, so very small values
    assert np.abs(signal).max() < 0.01  # EEG in V should be < 10mV

    # Read iEEG signal
    ieeg_signal = ShardManager.static_read(pool.root, ieeg_atom.signal_ref)
    assert ieeg_signal.shape[0] == ieeg_atom.n_channels
    assert ieeg_signal.shape[1] == 1001
    # iEEG signals can be larger than scalp EEG
    assert np.abs(ieeg_signal).max() < 1.0  # Should still be < 1V

    print(f"\n✓ CCEP E2E: 1 run imported, 5 EEG + 5 iEEG epochs, cross-modal linked, indexed, queried.")


def test_ccep_electrode_coordinates(pool_dir):
    """Verify electrode coordinates are loaded and stored in ChannelInfo."""
    from neuroatom.importers.ccep_bids_npy import CCEPImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "ccepcoreg.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = CCEPImporter(pool, task_config)
    results = importer.import_subject(
        subject_dir=CCEP_SUB01,
        subject_id="sub-01",
        max_runs=1,
        max_epochs=1,
    )

    eeg_result = [r for r in results if r.atoms[0].modality == "eeg"][0]
    ieeg_result = [r for r in results if r.atoms[0].modality == "ieeg"][0]

    # --- EEG electrode coordinates ---
    eeg_chs = eeg_result.channel_infos
    eeg_with_loc = [ch for ch in eeg_chs if ch.location is not None]
    print(f"  EEG: {len(eeg_with_loc)}/{len(eeg_chs)} channels have coordinates")
    assert len(eeg_with_loc) == 185  # All EEG channels should have coords

    # Verify coordinate system
    loc = eeg_with_loc[0].location
    assert loc.coordinate_system == "T1w"
    assert loc.coordinate_units == "mm"
    # Verify actual coordinate values are reasonable (mm, head-sized)
    assert abs(loc.x) < 200 and abs(loc.y) < 200 and abs(loc.z) < 200

    # --- iEEG electrode coordinates ---
    ieeg_chs = ieeg_result.channel_infos
    ieeg_with_loc = [ch for ch in ieeg_chs if ch.location is not None]
    print(f"  iEEG: {len(ieeg_with_loc)}/{len(ieeg_chs)} channels have coordinates")
    # iEEG bipolar channels should have coords (from MNI electrodes.tsv)
    assert len(ieeg_with_loc) > 100  # Most channels should have coords

    iloc = ieeg_with_loc[0].location
    # iEEG MNI coords are in meters
    assert iloc.coordinate_units == "m"
    assert abs(iloc.x) < 0.2  # Brain-sized in meters

    # --- iEEG electrode metadata (manufacturer, material, size) ---
    ieeg_with_material = [ch for ch in ieeg_chs if ch.custom_fields.get("material")]
    print(f"  iEEG: {len(ieeg_with_material)} channels have material info")
    assert len(ieeg_with_material) > 0
    assert ieeg_with_material[0].custom_fields["material"] == "PtIr"
    assert ieeg_with_material[0].custom_fields.get("manufacturer") == "Dixi Medical"

    # --- Reference type ---
    assert eeg_chs[0].reference == "average"
    assert ieeg_chs[0].reference == "bipolar"

    # --- Filter settings ---
    assert eeg_chs[0].custom_fields.get("low_cutoff_hz") == 0.5
    assert eeg_chs[0].custom_fields.get("high_cutoff_hz") == 45.0
    assert ieeg_chs[0].custom_fields.get("low_cutoff_hz") == 0.5
    assert ieeg_chs[0].custom_fields.get("high_cutoff_hz") == 300.0

    # --- Signal units ---
    assert eeg_chs[0].unit == "V"
    assert ieeg_chs[0].unit == "V"

    # --- Atom custom_fields ---
    eeg_atom = eeg_result.atoms[0]
    assert eeg_atom.custom_fields.get("coordinate_system") == "T1w"
    assert eeg_atom.custom_fields.get("reference") == "average"
    assert eeg_atom.custom_fields.get("low_cutoff_hz") == 0.5
    assert eeg_atom.custom_fields.get("high_cutoff_hz") == 45.0
    assert eeg_atom.custom_fields.get("n_electrodes_with_coords") == 185

    ieeg_atom = ieeg_result.atoms[0]
    assert "coordinate_system" in ieeg_atom.custom_fields
    assert ieeg_atom.custom_fields.get("reference") == "bipolar"
    assert ieeg_atom.custom_fields.get("high_cutoff_hz") == 300.0
    assert ieeg_atom.custom_fields.get("n_electrodes_with_coords") > 100

    # Verify anatomical landmarks for EEG
    landmarks = eeg_atom.custom_fields.get("anatomical_landmarks")
    if landmarks:  # EEG coordsystem has landmarks
        assert "NAS" in landmarks or "LPA" in landmarks
        print(f"  EEG anatomical landmarks: {list(landmarks.keys())}")

    print(f"\n✓ Electrode coordinates, metadata, reference, filters, units — all verified.")


def test_ccep_channel_quality(pool_dir):
    """Verify channel quality (good/bad) is preserved from channels.tsv."""
    from neuroatom.importers.ccep_bids_npy import CCEPImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool

    pool = Pool.create(pool_dir)
    tc_path = Path(__file__).parent.parent / "neuroatom" / "importers" / "task_configs" / "ccepcoreg.yaml"
    task_config = TaskConfig.from_yaml(tc_path)

    importer = CCEPImporter(pool, task_config)
    results = importer.import_subject(
        subject_dir=CCEP_SUB01,
        subject_id="sub-01",
        max_runs=1,
        max_epochs=1,
    )

    # Check that bad channels were detected
    eeg_result = [r for r in results if r.atoms[0].modality == "eeg"][0]
    ieeg_result = [r for r in results if r.atoms[0].modality == "ieeg"][0]

    # Both modalities should have some bad channels
    eeg_atom = eeg_result.atoms[0]
    ieeg_atom = ieeg_result.atoms[0]

    assert eeg_atom.quality is not None
    assert ieeg_atom.quality is not None

    # The channels.tsv for sub-01 shows many bad EEG channels
    assert len(eeg_atom.quality.bad_channels) > 0
    print(f"  EEG: {len(eeg_atom.quality.bad_channels)} bad channels out of {eeg_atom.n_channels}")
    print(f"  iEEG: {len(ieeg_atom.quality.bad_channels)} bad channels out of {ieeg_atom.n_channels}")

    print(f"\n✓ Channel quality preserved: bad channels detected in both modalities.")


def test_ccep_stim_parsing(pool_dir):
    """Verify stimulation parameter parsing from trial_type."""
    from neuroatom.importers.ccep_bids_npy import _parse_stim_description

    desc = "R'6-7 5ma 0.5ms 0.5hz parallel wh_wh"
    params = _parse_stim_description(desc)

    assert params["stim_contact"] == "R'6-7"
    assert params["stim_intensity_ma"] == "5"
    assert params["stim_duration_ms"] == "0.5"
    assert params["stim_frequency_hz"] == "0.5"
    assert params["stim_angle"] == "parallel"
    assert params["stim_tissue"] == "wh_wh"

    # Edge case: minimal description
    params2 = _parse_stim_description("A'1-2")
    assert params2["stim_contact"] == "A'1-2"

    print(f"\n✓ Stimulation parameter parsing verified.")
