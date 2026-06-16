"""End-to-end tests for the SNHL-AAD importer.

Requires the actual dataset at D:\\Data\\ds-eeg-snhl.
Skipped automatically if the data is not available.
"""

import json
import pytest
from pathlib import Path

DATA_ROOT = Path(r"D:\Data\ds-eeg-snhl")
pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="SNHL-AAD data not available at D:\\Data\\ds-eeg-snhl",
)


@pytest.fixture
def pool(tmp_path):
    from neuroatom.storage.pool import Pool
    return Pool.create(tmp_path / "pool")


@pytest.fixture
def task_config():
    from neuroatom.importers.base import TaskConfig
    return TaskConfig.builtin("snhl_aad")


class TestSNHLAADImporter:
    """Full integration tests against the real ds-eeg-snhl dataset."""

    def test_detect(self):
        from neuroatom.importers.snhl_aad import SNHLAADImporter
        assert SNHLAADImporter.detect(DATA_ROOT) is True

    def test_detect_wrong_dir(self, tmp_path):
        from neuroatom.importers.snhl_aad import SNHLAADImporter
        assert SNHLAADImporter.detect(tmp_path) is False

    def test_import_single_subject_selectiveattention(self, pool, task_config):
        """Import one subject's selective attention task."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-001"],
        )

        assert len(results) == 1
        result = results[0]

        # sub-001 has earEEG, should have ~48 trials
        assert result.run_meta.subject_id == "sub-001"
        assert len(result.atoms) > 0

        # Check atom structure
        atom = result.atoms[0]
        assert atom.dataset_id == "snhl_aad"
        assert atom.subject_id == "sub-001"
        assert atom.atom_type.value == "trial"
        assert atom.sampling_rate == 512.0

        # Check channels include both EEG and EXG
        assert atom.n_channels > 60  # 64 EEG + some EXG

        # Check annotations
        ann_names = [a.name for a in atom.annotations]
        assert "attend_direction" in ann_names
        assert "condition" in ann_names

        # sub-001 has earEEG
        assert "has_eareeg" in ann_names

    def test_import_non_eareeg_subject(self, pool, task_config):
        """Import subject without ear-EEG to verify different channel layout."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-010"],  # No earEEG
        )

        assert len(results) == 1
        atom = results[0].atoms[0]

        # Check no ear-EEG annotation
        ann_names = [a.name for a in atom.annotations]
        assert "has_eareeg" not in ann_names

    def test_import_rest_task(self, pool, task_config):
        """Import resting-state data as continuous segment."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["rest"],
            subjects=["sub-001"],
        )

        assert len(results) == 1
        atom = results[0].atoms[0]
        assert atom.atom_type.value == "continuous_segment"

    def test_channels_tsv_parsing(self):
        """Verify channels.tsv is correctly parsed for earEEG vs non-earEEG."""
        from neuroatom.importers.snhl_aad import _parse_channels_tsv

        # earEEG subject
        ch_path = DATA_ROOT / "sub-001" / "eeg" / "sub-001_task-selectiveattention_channels.tsv"
        ch_infos = _parse_channels_tsv(ch_path, 512.0)

        # Should have 64 EEG + EXG channels (Status excluded)
        assert len(ch_infos) >= 64
        eareeg = [c for c in ch_infos if c.custom_fields.get("is_eareeg")]
        assert len(eareeg) == 6  # 3 right + 3 left

        # non-earEEG subject
        ch_path2 = DATA_ROOT / "sub-010" / "eeg" / "sub-010_task-selectiveattention_channels.tsv"
        ch_infos2 = _parse_channels_tsv(ch_path2, 512.0)
        eareeg2 = [c for c in ch_infos2 if c.custom_fields.get("is_eareeg")]
        assert len(eareeg2) == 0

    def test_events_parsing(self):
        """Verify selective attention events are correctly parsed."""
        from neuroatom.importers.snhl_aad import _parse_selectiveattention_events

        events_path = DATA_ROOT / "sub-001" / "eeg" / "sub-001_task-selectiveattention_events.tsv"
        trials = _parse_selectiveattention_events(events_path)

        assert len(trials) > 0
        trial = trials[0]
        assert "onset_sample" in trial
        assert "end_sample" in trial
        assert trial["attend_lr"] in ("attendleft", "attendright")
        assert trial["condition"] in ("singletalker", "twotalker")

    def test_subject_metadata(self, pool, task_config):
        """Verify subject hearing status and earEEG flag are stored."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-001"],
        )

        meta = pool.get_subject_meta("snhl_aad", "sub-001")
        assert meta.custom_fields["hearing_status"] == "hi"
        assert meta.custom_fields["has_eareeg"] is True

    def test_catalog_auto_created(self, pool, task_config):
        """Verify catalog.json is created after import."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-001"],
        )

        cat_path = pool.root / "catalog.json"
        assert cat_path.exists()
        cat = json.loads(cat_path.read_text())
        assert len(cat["datasets"]) > 0
        assert cat["datasets"][0]["dataset_id"] == "snhl_aad"


class TestSNHLAudioStimuli:
    """Verify audio stimulus envelope import (P3)."""

    def test_target_envelope_annotation(self, pool, task_config):
        """ContinuousAnnotation for target envelope is present."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-001"],
        )
        assert len(results) >= 1
        atoms = results[0].atoms
        assert len(atoms) > 0

        # All atoms should have target_envelope
        atom = atoms[0]
        cont_anns = [a for a in atom.annotations if a.annotation_type == "continuous"]
        target_anns = [a for a in cont_anns if a.name == "target_envelope"]
        assert len(target_anns) == 1, f"Expected target_envelope, got {[a.name for a in cont_anns]}"

        tenv = target_anns[0]
        assert tenv.domain == "stimulus"
        assert tenv.data_sampling_rate == 512.0
        assert tenv.data_ref.shape[0] > 1000  # ~50s @ 512 Hz = ~25600

    def test_masker_envelope_for_twotalker(self, pool, task_config):
        """Two-talker trials have masker_envelope annotation."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-001"],
        )
        atoms = results[0].atoms

        # Find a two-talker trial
        twotalker = None
        for a in atoms:
            cond_ann = next(
                (ann for ann in a.annotations if ann.name == "condition"), None
            )
            if cond_ann and cond_ann.value == "twotalker":
                twotalker = a
                break

        assert twotalker is not None, "No two-talker trial found"
        cont_anns = [a for a in twotalker.annotations if a.annotation_type == "continuous"]
        masker_anns = [a for a in cont_anns if a.name == "masker_envelope"]
        assert len(masker_anns) == 1

    def test_singletalker_no_masker(self, pool, task_config):
        """Single-talker trials should not have masker_envelope."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-001"],
        )
        atoms = results[0].atoms

        for a in atoms:
            cond_ann = next(
                (ann for ann in a.annotations if ann.name == "condition"), None
            )
            if cond_ann and cond_ann.value == "singletalker":
                cont_anns = [ann for ann in a.annotations if ann.annotation_type == "continuous"]
                masker_anns = [ann for ann in cont_anns if ann.name == "masker_envelope"]
                assert len(masker_anns) == 0, "Single-talker trial has masker_envelope"
                return

        pytest.skip("No single-talker trial found")

    def test_envelope_stored_in_hdf5(self, pool, task_config):
        """Verify envelope companion arrays exist in HDF5 shard."""
        import h5py
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-001"],
        )
        atom = results[0].atoms[0]
        shard_path = pool.root / atom.signal_ref.file_path
        assert shard_path.exists()

        with h5py.File(str(shard_path), "r") as f:
            atom_grp = f[f"/atoms/{atom.atom_id}"]
            assert "annotations" in atom_grp
            ann_keys = set(atom_grp["annotations"].keys())
            assert "target_envelope" in ann_keys

            # Verify envelope shape matches annotation
            env = atom_grp["annotations"]["target_envelope"][:]
            assert env.ndim == 1
            assert env.shape[0] > 1000  # ~25600 samples

    def test_difficulty_and_accuracy_annotations(self, pool, task_config):
        """Verify behavioural annotations (difficulty, accuracy) are stored."""
        from neuroatom.importers.snhl_aad import SNHLAADImporter

        imp = SNHLAADImporter(pool=pool, task_config=task_config)
        results = imp.import_dataset(
            DATA_ROOT,
            tasks=["selectiveattention"],
            subjects=["sub-001"],
        )
        atoms = results[0].atoms

        # At least some trials should have difficulty ratings
        has_diff = False
        has_acc = False
        for a in atoms:
            for ann in a.annotations:
                if ann.annotation_type == "numeric":
                    if ann.name == "difficulty_rating":
                        has_diff = True
                    if ann.name == "questionnaire_accuracy":
                        has_acc = True
        assert has_diff, "No difficulty_rating annotations found"
        assert has_acc, "No questionnaire_accuracy annotations found"
