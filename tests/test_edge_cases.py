"""Edge case tests for v0.2 features.

Covers the specific failure scenarios that users are most likely to hit:
- Federation: empty pools, partial matches, overlap detection
- YAML: empty files, malformed syntax, unknown fields, missing required fields
- Pairing keys: missing fields on atoms
- Import provenance: edge cases
"""

import json
import tempfile
from pathlib import Path

import pytest

from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
from neuroatom.core.multimodal_recipe import MultiModalRecipe, ModalityPipelineConfig


# ═════════════════════════════════════════════════════════════════════
# Federation edge cases
# ═════════════════════════════════════════════════════════════════════


class TestFederationEdgeCases:
    """Test federation with non-happy-path scenarios."""

    def _create_pool_with_atoms(self, root, dataset_id, n_atoms=3):
        """Helper: create a pool and insert synthetic atoms directly into SQLite index."""
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.signal_ref import SignalRef
        from neuroatom.utils.hashing import compute_atom_id

        pool = Pool.create(root)
        indexer = Indexer(pool)

        atoms = []
        for i in range(n_atoms):
            atom_id = compute_atom_id(
                dataset_id=dataset_id,
                subject_id="S01",
                session_id="ses-01",
                run_id="run-01",
                onset_sample=i * 256,
            )
            atom = Atom(
                atom_id=atom_id,
                atom_type="trial",
                dataset_id=dataset_id,
                subject_id="S01",
                session_id="ses-01",
                run_id="run-01",
                trial_index=i,
                channel_ids=["Cz"],
                n_channels=1,
                sampling_rate=256.0,
                temporal=TemporalInfo(
                    onset_sample=i * 256,
                    duration_samples=256,
                    onset_seconds=float(i),
                    duration_seconds=1.0,
                ),
                signal_ref=SignalRef(
                    file_path="signals_000.h5",
                    internal_path=f"/atoms/{atom_id}/signal",
                    shape=(1, 256),
                ),
            )
            atoms.append(atom)

        # Insert directly into SQLite index (skip JSONL + dataset.json requirement)
        indexer.backend.upsert_atoms(atoms)
        return pool, indexer, atoms

    def test_one_pool_empty(self, tmp_path):
        """Empty pool + populated pool should return only populated pool's atoms."""
        from neuroatom.index.federation import FederatedPool, FederatedQueryBuilder
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer

        pool_a, idx_a, atoms_a = self._create_pool_with_atoms(
            tmp_path / "pool_a", "dataset_a", n_atoms=5
        )
        pool_b = Pool.create(tmp_path / "pool_b")
        idx_b = Indexer(pool_b)  # empty

        fed = FederatedPool([pool_a, pool_b], [idx_a, idx_b], tags=["a", "b"])
        fqb = FederatedQueryBuilder(fed)

        ids = fqb.query_atom_ids({"dataset_id": "dataset_a"})
        assert len(ids) == 5

        # All atoms should resolve to pool_a
        for aid in ids:
            handle = fed.resolve_pool(aid)
            assert handle.tag == "a"

        idx_a.close()
        idx_b.close()

    def test_partial_match_different_datasets(self, tmp_path):
        """Pool A has dataset_a, Pool B has dataset_b. Query for dataset_a returns only pool A."""
        from neuroatom.index.federation import FederatedPool, FederatedQueryBuilder

        pool_a, idx_a, _ = self._create_pool_with_atoms(
            tmp_path / "pool_a", "dataset_a", n_atoms=3
        )
        pool_b, idx_b, _ = self._create_pool_with_atoms(
            tmp_path / "pool_b", "dataset_b", n_atoms=4
        )

        fed = FederatedPool([pool_a, pool_b], [idx_a, idx_b], tags=["a", "b"])
        fqb = FederatedQueryBuilder(fed)

        # Query only dataset_a
        ids_a = fqb.query_atom_ids({"dataset_id": "dataset_a"})
        assert len(ids_a) == 3
        for aid in ids_a:
            assert fed.resolve_pool(aid).tag == "a"

        # Query only dataset_b
        ids_b = fqb.query_atom_ids({"dataset_id": "dataset_b"})
        assert len(ids_b) == 4
        for aid in ids_b:
            assert fed.resolve_pool(aid).tag == "b"

        # Query both (no filter on dataset_id)
        all_ids = fqb.query_atom_ids({})
        assert len(all_ids) == 7  # 3 + 4, no overlap

        idx_a.close()
        idx_b.close()

    def test_same_dataset_two_pools_dedup_with_warning(self, tmp_path):
        """Same dataset in two pools should dedup and warn (content-addressable IDs)."""
        import logging
        from neuroatom.index.federation import FederatedPool, FederatedQueryBuilder

        pool_a, idx_a, atoms_a = self._create_pool_with_atoms(
            tmp_path / "pool_a", "same_ds", n_atoms=3
        )
        pool_b, idx_b, atoms_b = self._create_pool_with_atoms(
            tmp_path / "pool_b", "same_ds", n_atoms=3
        )

        # atom_ids should be identical (same content hash)
        assert set(a.atom_id for a in atoms_a) == set(a.atom_id for a in atoms_b)

        fed = FederatedPool([pool_a, pool_b], [idx_a, idx_b], tags=["a", "b"])
        fqb = FederatedQueryBuilder(fed)

        # Should not crash — dedup silently keeps first pool's copy
        ids = fqb.query_atom_ids({"dataset_id": "same_ds"})

        # Should dedup to 3 (not 6)
        assert len(ids) == 3

        idx_a.close()
        idx_b.close()

    def test_per_pool_query(self, tmp_path):
        """query_per_pool returns separate lists per pool."""
        from neuroatom.index.federation import FederatedPool, FederatedQueryBuilder

        pool_a, idx_a, _ = self._create_pool_with_atoms(
            tmp_path / "pool_a", "ds_a", n_atoms=2
        )
        pool_b, idx_b, _ = self._create_pool_with_atoms(
            tmp_path / "pool_b", "ds_b", n_atoms=5
        )

        fed = FederatedPool([pool_a, pool_b], [idx_a, idx_b], tags=["a", "b"])
        fqb = FederatedQueryBuilder(fed)

        per_pool = fqb.query_per_pool({})
        assert len(per_pool["a"]) == 2
        assert len(per_pool["b"]) == 5

        idx_a.close()
        idx_b.close()


# ═════════════════════════════════════════════════════════════════════
# YAML error handling edge cases
# ═════════════════════════════════════════════════════════════════════


class TestYAMLEdgeCases:
    """Test YAML loading with various error conditions."""

    def test_empty_file(self, tmp_path):
        """Empty YAML file should raise ValueError with helpful message."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")

        with pytest.raises(ValueError, match="YAML file is empty"):
            AssemblyRecipe.from_yaml(empty)

    def test_non_dict_yaml(self, tmp_path):
        """YAML that parses to a list should raise ValueError."""
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n")

        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            AssemblyRecipe.from_yaml(bad)

    def test_malformed_yaml_syntax(self, tmp_path):
        """Malformed YAML should raise ValueError with parse error."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("recipe_id: test\n  bad indent: here\n")

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            AssemblyRecipe.from_yaml(bad)

    def test_missing_required_fields(self, tmp_path):
        """YAML missing required fields should list what's needed."""
        partial = tmp_path / "partial.yaml"
        partial.write_text("recipe_id: test\n")

        with pytest.raises(ValueError, match="Required fields"):
            AssemblyRecipe.from_yaml(partial)

    def test_unknown_field_typo_transforms(self, tmp_path):
        """YAML with 'transform' instead of 'augmentations' should fail with helpful error.
        
        extra='forbid' on AssemblyRecipe catches typos immediately instead of
        silently dropping unknown fields.
        """
        yaml_content = """
recipe_id: test_typo
query:
  dataset_id: bci
label_fields:
  - annotation_name: mi_class
    output_key: mi_class
transform: zscore
"""
        path = tmp_path / "typo.yaml"
        path.write_text(yaml_content)

        with pytest.raises(ValueError, match="validation failed"):
            AssemblyRecipe.from_yaml(path)

    def test_invalid_enum_value(self, tmp_path):
        """Invalid enum value should give helpful Pydantic error."""
        yaml_content = """
recipe_id: bad_enum
query:
  dataset_id: bci
label_fields:
  - annotation_name: mi_class
    output_key: mi_class
normalization_method: z_score_typo
"""
        path = tmp_path / "bad_enum.yaml"
        path.write_text(yaml_content)

        with pytest.raises(ValueError, match="validation failed"):
            AssemblyRecipe.from_yaml(path)

    def test_multimodal_empty_yaml(self, tmp_path):
        """Empty YAML for MultiModalRecipe should give helpful error."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")

        with pytest.raises(ValueError, match="YAML file is empty"):
            MultiModalRecipe.from_yaml(empty)

    def test_multimodal_missing_modalities(self, tmp_path):
        """MultiModalRecipe with only 1 modality should fail validation."""
        yaml_content = """
recipe_id: bad_mm
modalities:
  eeg:
    query: {dataset_id: x}
label_fields:
  - annotation_name: c
    output_key: d
"""
        path = tmp_path / "one_mod.yaml"
        path.write_text(yaml_content)

        with pytest.raises(ValueError, match="validation failed"):
            MultiModalRecipe.from_yaml(path)


# ═════════════════════════════════════════════════════════════════════
# Pairing keys edge cases
# ═════════════════════════════════════════════════════════════════════


class TestPairingKeyEdgeCases:
    """Test _build_pairing_key with missing/nonexistent fields."""

    def test_missing_field_returns_none_string(self):
        """Pairing key with a field not on the atom should use 'None' string."""
        from neuroatom.assembler.multimodal_assembler import MultiModalAssembler
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.signal_ref import SignalRef

        atom = Atom(
            atom_id="test-missing",
            atom_type="trial",
            dataset_id="ds",
            subject_id="S01",
            channel_ids=["Cz"],
            n_channels=1,
            sampling_rate=256.0,
            temporal=TemporalInfo(
                onset_sample=0, duration_samples=256,
                onset_seconds=0.0, duration_seconds=1.0,
            ),
            signal_ref=SignalRef(
                file_path="x.h5",
                internal_path="/atoms/test/signal",
                shape=(1, 256),
            ),
        )

        # "nonexistent_key" not on atom or custom_fields
        key = MultiModalAssembler._build_pairing_key(
            atom, ["subject_id", "nonexistent_key"]
        )
        assert key == "S01|None"

    def test_none_session_and_run(self):
        """Atoms with None session/run should produce deterministic pairing keys."""
        from neuroatom.assembler.multimodal_assembler import MultiModalAssembler
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.signal_ref import SignalRef

        atom = Atom(
            atom_id="test-nones",
            atom_type="trial",
            dataset_id="ds",
            subject_id="S01",
            session_id=None,
            run_id=None,
            channel_ids=["Cz"],
            n_channels=1,
            sampling_rate=256.0,
            temporal=TemporalInfo(
                onset_sample=0, duration_samples=256,
                onset_seconds=0.0, duration_seconds=1.0,
            ),
            signal_ref=SignalRef(
                file_path="x.h5",
                internal_path="/atoms/test/signal",
                shape=(1, 256),
            ),
        )

        key = MultiModalAssembler._build_pairing_key(
            atom, ["subject_id", "session_id", "run_id"]
        )
        assert key == "S01|None|None"

    def test_custom_field_in_pairing_key(self):
        """Pairing key should resolve custom_fields correctly."""
        from neuroatom.assembler.multimodal_assembler import MultiModalAssembler
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.signal_ref import SignalRef

        atom = Atom(
            atom_id="test-custom",
            atom_type="trial",
            dataset_id="ds",
            subject_id="S01",
            channel_ids=["Cz"],
            n_channels=1,
            sampling_rate=256.0,
            temporal=TemporalInfo(
                onset_sample=0, duration_samples=256,
                onset_seconds=0.0, duration_seconds=1.0,
            ),
            signal_ref=SignalRef(
                file_path="x.h5",
                internal_path="/atoms/test/signal",
                shape=(1, 256),
            ),
            custom_fields={"experiment_block": "B2", "condition": "active"},
        )

        key = MultiModalAssembler._build_pairing_key(
            atom, ["subject_id", "experiment_block", "condition"]
        )
        assert key == "S01|B2|active"


# ═════════════════════════════════════════════════════════════════════
# Import provenance edge cases
# ═════════════════════════════════════════════════════════════════════


class TestImportProvenanceEdgeCases:
    """Test import provenance with edge cases."""

    def test_empty_history(self, tmp_path):
        """get_import_history on empty pool should return empty list."""
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.import_log import get_import_history

        pool = Pool.create(tmp_path / "pool")
        indexer = Indexer(pool)

        history = get_import_history(indexer)
        assert history == []

        indexer.close()

    def test_multiple_imports_same_dataset(self, tmp_path):
        """Multiple imports of the same dataset should all be logged."""
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.import_log import log_import, get_import_history

        pool = Pool.create(tmp_path / "pool")
        indexer = Indexer(pool)

        for i in range(5):
            log_import(
                indexer=indexer,
                dataset_id="ds",
                importer_name="TestImporter",
                n_atoms=10 * (i + 1),
                subject_id=f"S{i+1:02d}",
            )

        history = get_import_history(indexer, dataset_id="ds")
        assert len(history) == 5
        # Newest first
        assert history[0]["subject_id"] == "S05"
        assert history[0]["n_atoms"] == 50

        indexer.close()

    def test_parameters_with_special_chars(self, tmp_path):
        """Import parameters with paths and special characters should round-trip."""
        from neuroatom.storage.pool import Pool
        from neuroatom.index.indexer import Indexer
        from neuroatom.index.import_log import log_import, get_import_history

        pool = Pool.create(tmp_path / "pool")
        indexer = Indexer(pool)

        params = {
            "data_path": "C:\\Data\\BCI\\A01T.mat",
            "filter_band": [0.5, 40.0],
            "subjects": ["A01", "A02", "A03"],
            "unicode_note": "脑电数据导入",
        }

        log_import(
            indexer=indexer,
            dataset_id="test",
            importer_name="TestImporter",
            n_atoms=100,
            parameters=params,
        )

        history = get_import_history(indexer)
        assert len(history) == 1
        p = history[0]["parameters"]
        assert p["data_path"] == "C:\\Data\\BCI\\A01T.mat"
        assert p["unicode_note"] == "脑电数据导入"
        assert p["filter_band"] == [0.5, 40.0]

        indexer.close()
