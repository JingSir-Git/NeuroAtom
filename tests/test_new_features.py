"""Tests for v0.2 features: pairing_keys, federation, YAML recipe, import provenance."""

import json
import tempfile
from pathlib import Path

import pytest

from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
from neuroatom.core.multimodal_recipe import MultiModalRecipe, ModalityPipelineConfig
from neuroatom.core.enums import NormalizationMethod, SplitStrategy


# ── YAML serialization round-trip ────────────────────────────────────


class TestAssemblyRecipeYAML:
    """Test AssemblyRecipe.from_yaml / to_yaml round-trip."""

    def test_round_trip(self, tmp_path):
        recipe = AssemblyRecipe(
            recipe_id="test_yaml_rt",
            description="YAML round-trip test",
            query={"dataset_id": "bci_comp_iv_2a", "annotations": [{"name": "mi_class"}]},
            target_sampling_rate=250.0,
            target_duration=4.0,
            target_unit="uV",
            filter_band=(0.5, 40.0),
            normalization_method=NormalizationMethod.ZSCORE,
            label_fields=[LabelSpec(annotation_name="mi_class", output_key="mi_class")],
            split_strategy=SplitStrategy.SUBJECT,
            split_config={"val_ratio": 0.1, "test_ratio": 0.2, "seed": 42},
        )

        yaml_path = tmp_path / "recipe.yaml"
        recipe.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = AssemblyRecipe.from_yaml(yaml_path)
        assert loaded.recipe_id == "test_yaml_rt"
        assert loaded.target_sampling_rate == 250.0
        assert loaded.filter_band == (0.5, 40.0)
        assert loaded.normalization_method == NormalizationMethod.ZSCORE
        assert loaded.split_config["seed"] == 42

    def test_from_yaml_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Recipe YAML not found"):
            AssemblyRecipe.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_dict(self):
        data = {
            "recipe_id": "from_dict",
            "query": {"dataset_id": "x"},
            "label_fields": [{"annotation_name": "c", "output_key": "d"}],
        }
        r = AssemblyRecipe.from_dict(data)
        assert r.recipe_id == "from_dict"


class TestMultiModalRecipeYAML:
    """Test MultiModalRecipe.from_yaml / to_yaml round-trip."""

    def test_round_trip(self, tmp_path):
        recipe = MultiModalRecipe(
            recipe_id="ccep_test",
            modalities={
                "eeg": ModalityPipelineConfig(
                    query={"dataset_id": "ccepcoreg", "modality": "eeg"},
                    target_sampling_rate=1000.0,
                    filter_band=(0.5, 45.0),
                ),
                "ieeg": ModalityPipelineConfig(
                    query={"dataset_id": "ccepcoreg", "modality": "ieeg"},
                    target_sampling_rate=1000.0,
                ),
            },
            pairing_keys=["subject_id", "session_id", "run_id"],
            label_fields=[LabelSpec(annotation_name="stim_type", output_key="stim_type")],
        )

        yaml_path = tmp_path / "mm_recipe.yaml"
        recipe.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = MultiModalRecipe.from_yaml(yaml_path)
        assert loaded.recipe_id == "ccep_test"
        assert set(loaded.modalities.keys()) == {"eeg", "ieeg"}
        assert loaded.pairing_keys == ["subject_id", "session_id", "run_id"]
        assert loaded.modalities["eeg"].target_sampling_rate == 1000.0


# ── Pairing keys ─────────────────────────────────────────────────────


class TestPairingKeys:
    """Test configurable pairing_keys in MultiModalRecipe."""

    def test_default_pairing_keys(self):
        recipe = MultiModalRecipe(
            recipe_id="test_default",
            modalities={
                "a": ModalityPipelineConfig(query={"dataset_id": "x"}),
                "b": ModalityPipelineConfig(query={"dataset_id": "y"}),
            },
            label_fields=[LabelSpec(annotation_name="c", output_key="d")],
        )
        assert recipe.pairing_keys == ["subject_id", "session_id", "run_id"]

    def test_custom_pairing_keys(self):
        recipe = MultiModalRecipe(
            recipe_id="test_custom",
            modalities={
                "a": ModalityPipelineConfig(query={"dataset_id": "x"}),
                "b": ModalityPipelineConfig(query={"dataset_id": "y"}),
            },
            pairing_keys=["subject_id", "session_id"],
            label_fields=[LabelSpec(annotation_name="c", output_key="d")],
        )
        assert recipe.pairing_keys == ["subject_id", "session_id"]

    def test_trial_level_pairing_keys(self):
        recipe = MultiModalRecipe(
            recipe_id="test_trial",
            modalities={
                "a": ModalityPipelineConfig(query={"dataset_id": "x"}),
                "b": ModalityPipelineConfig(query={"dataset_id": "y"}),
            },
            pairing_keys=["subject_id", "session_id", "run_id", "trial_index"],
            label_fields=[LabelSpec(annotation_name="c", output_key="d")],
        )
        assert len(recipe.pairing_keys) == 4
        assert "trial_index" in recipe.pairing_keys


# ── Build pairing key ────────────────────────────────────────────────


class TestBuildPairingKey:
    """Test MultiModalAssembler._build_pairing_key."""

    def test_standard_fields(self):
        from neuroatom.assembler.multimodal_assembler import MultiModalAssembler
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.signal_ref import SignalRef

        atom = Atom(
            atom_id="test-001",
            atom_type="trial",
            dataset_id="ds",
            subject_id="S01",
            session_id="ses-01",
            run_id="run-01",
            trial_index=5,
            channel_ids=["Cz"],
            n_channels=1,
            sampling_rate=256.0,
            temporal=TemporalInfo(
                onset_sample=0, duration_samples=256,
                onset_seconds=0.0, duration_seconds=1.0,
            ),
            signal_ref=SignalRef(file_path="x.h5", internal_path="/atoms/test-001/signal", shape=(1, 256)),
        )

        key = MultiModalAssembler._build_pairing_key(
            atom, ["subject_id", "session_id", "run_id"]
        )
        assert key == "S01|ses-01|run-01"

    def test_trial_index(self):
        from neuroatom.assembler.multimodal_assembler import MultiModalAssembler
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.signal_ref import SignalRef

        atom = Atom(
            atom_id="test-002",
            atom_type="trial",
            dataset_id="ds",
            subject_id="S01",
            session_id="ses-01",
            run_id="run-01",
            trial_index=7,
            channel_ids=["Cz"],
            n_channels=1,
            sampling_rate=256.0,
            temporal=TemporalInfo(
                onset_sample=0, duration_samples=256,
                onset_seconds=0.0, duration_seconds=1.0,
            ),
            signal_ref=SignalRef(file_path="x.h5", internal_path="/atoms/test-002/signal", shape=(1, 256)),
        )

        key = MultiModalAssembler._build_pairing_key(
            atom, ["subject_id", "session_id", "run_id", "trial_index"]
        )
        assert key == "S01|ses-01|run-01|7"

    def test_custom_fields(self):
        from neuroatom.assembler.multimodal_assembler import MultiModalAssembler
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.signal_ref import SignalRef

        atom = Atom(
            atom_id="test-003",
            atom_type="trial",
            dataset_id="ds",
            subject_id="S01",
            session_id="ses-01",
            run_id="run-01",
            channel_ids=["Cz"],
            n_channels=1,
            sampling_rate=256.0,
            temporal=TemporalInfo(
                onset_sample=0, duration_samples=256,
                onset_seconds=0.0, duration_seconds=1.0,
            ),
            signal_ref=SignalRef(file_path="x.h5", internal_path="/atoms/test-003/signal", shape=(1, 256)),
            custom_fields={"block_id": "B3"},
        )

        key = MultiModalAssembler._build_pairing_key(
            atom, ["subject_id", "block_id"]
        )
        assert key == "S01|B3"


# ── Federation ───────────────────────────────────────────────────────


class TestFederation:
    """Test FederatedPool, FederatedQueryBuilder basic construction."""

    def test_create_federated_pool(self, tmp_path):
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.federation import FederatedPool

        pool_a = Pool.create(tmp_path / "pool_a")
        pool_b = Pool.create(tmp_path / "pool_b")
        idx_a = Indexer(pool_a)
        idx_b = Indexer(pool_b)

        fed = FederatedPool(
            [pool_a, pool_b], [idx_a, idx_b], tags=["bci", "physionet"]
        )
        assert len(fed.handles) == 2
        assert fed.handles[0].tag == "bci"
        assert fed.handles[1].tag == "physionet"

        # Count should be 0 for empty pools
        counts = fed.count_atoms()
        assert counts["bci"] == 0
        assert counts["physionet"] == 0

        idx_a.close()
        idx_b.close()

    def test_federated_pool_validation(self, tmp_path):
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.federation import FederatedPool

        pool = Pool.create(tmp_path / "pool")
        idx = Indexer(pool)

        with pytest.raises(ValueError, match="at least one pool"):
            FederatedPool([], [])

        with pytest.raises(ValueError, match="must equal"):
            FederatedPool([pool], [idx, idx])

        with pytest.raises(ValueError, match="unique"):
            FederatedPool([pool, pool], [idx, idx], tags=["same", "same"])

        idx.close()

    def test_federated_query_empty(self, tmp_path):
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.federation import FederatedPool, FederatedQueryBuilder

        pool = Pool.create(tmp_path / "pool")
        idx = Indexer(pool)
        fed = FederatedPool([pool], [idx])
        fqb = FederatedQueryBuilder(fed)

        ids = fqb.query_atom_ids({"dataset_id": "nonexistent"})
        assert ids == []

        count = fqb.query_count({"dataset_id": "nonexistent"})
        assert count == 0

        idx.close()


# ── Import provenance ────────────────────────────────────────────────


class TestImportProvenance:
    """Test import_log table and provenance logging."""

    def test_insert_and_query(self, tmp_path):
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.import_log import log_import, get_import_history

        pool = Pool.create(tmp_path / "pool")
        indexer = Indexer(pool)

        # Log an import
        log_id = log_import(
            indexer=indexer,
            dataset_id="bci_comp_iv_2a",
            importer_name="BCICompIV2aImporter",
            n_atoms=288,
            subject_id="A01",
            parameters={"mat_path": "/data/A01T.mat", "labelled_runs": [3, 4, 5, 6, 7, 8]},
            duration_seconds=5.2,
        )
        assert log_id > 0

        # Log another
        log_import(
            indexer=indexer,
            dataset_id="physionet_mi",
            importer_name="PhysioNetMIImporter",
            n_atoms=90,
            subject_id="S001",
            n_errors=2,
        )

        # Query all
        history = get_import_history(indexer)
        assert len(history) == 2
        assert history[0]["importer_name"] == "PhysioNetMIImporter"  # newest first
        assert history[1]["importer_name"] == "BCICompIV2aImporter"

        # Query by dataset
        bci_history = get_import_history(indexer, dataset_id="bci_comp_iv_2a")
        assert len(bci_history) == 1
        assert bci_history[0]["n_atoms"] == 288
        assert bci_history[0]["subject_id"] == "A01"

        # Parameters are deserialized back to dict
        params = bci_history[0]["parameters"]
        assert isinstance(params, dict)
        assert params["mat_path"] == "/data/A01T.mat"

        indexer.close()

    def test_import_log_version_tracking(self, tmp_path):
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.import_log import log_import, get_import_history

        pool = Pool.create(tmp_path / "pool")
        indexer = Indexer(pool)

        log_import(
            indexer=indexer,
            dataset_id="test",
            importer_name="TestImporter",
            n_atoms=10,
        )

        history = get_import_history(indexer)
        assert history[0]["importer_version"] == "0.1.0"
        assert history[0]["timestamp"] is not None

        indexer.close()
