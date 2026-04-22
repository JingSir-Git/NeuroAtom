"""Unit tests for NeuroAtom core Pydantic models.

Tests model creation, serialization/deserialization round-trips,
validation constraints, and discriminated union dispatch.
"""

import json

import pytest

from neuroatom.core.annotation import (
    AnnotationUnion,
    BinaryMaskAnnotation,
    CategoricalAnnotation,
    ContinuousAnnotation,
    EventItem,
    EventSequenceAnnotation,
    NumericAnnotation,
    StimulusRefAnnotation,
    TextAnnotation,
)
from neuroatom.core.atom import Atom, AtomRelation, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import (
    AnnotationType,
    AtomType,
    ChannelType,
    ErrorHandling,
    NormalizationMethod,
    NormalizationScope,
    QualityStatus,
    SplitStrategy,
)
from neuroatom.core.montage import MontageInfo
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.quality import QualityInfo
from neuroatom.core.recipe import (
    AssemblyRecipe,
    AugmentationUnion,
    ChannelDropoutAug,
    GaussianNoiseAug,
    LabelSpec,
    TemporalShiftAug,
)
from neuroatom.core.run import RunMeta
from neuroatom.core.session import SessionMeta
from neuroatom.core.signal_ref import SignalRef
from neuroatom.core.stimulus import StimulusResource
from neuroatom.core.subject import SubjectMeta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_signal_ref(**overrides):
    defaults = {
        "file_path": "datasets/test/subjects/s01/sessions/ses-01/runs/run-01/signals_000.h5",
        "internal_path": "/atoms/abc123/signal",
        "shape": (64, 512),
    }
    defaults.update(overrides)
    return SignalRef(**defaults)


def _make_temporal_info(**overrides):
    defaults = {
        "onset_sample": 0,
        "onset_seconds": 0.0,
        "duration_samples": 512,
        "duration_seconds": 2.0,
    }
    defaults.update(overrides)
    return TemporalInfo(**defaults)


def _make_atom(**overrides):
    defaults = {
        "atom_id": "abc123def456",
        "atom_type": AtomType.TRIAL,
        "dataset_id": "test_dataset",
        "subject_id": "sub-01",
        "session_id": "ses-01",
        "run_id": "run-01",
        "trial_index": 0,
        "signal_ref": _make_signal_ref(),
        "temporal": _make_temporal_info(),
        "channel_ids": [f"ch_{i}" for i in range(64)],
        "n_channels": 64,
        "sampling_rate": 256.0,
    }
    defaults.update(overrides)
    return Atom(**defaults)


# ---------------------------------------------------------------------------
# SignalRef
# ---------------------------------------------------------------------------

class TestSignalRef:
    def test_creation(self):
        ref = _make_signal_ref()
        assert ref.file_path.endswith("signals_000.h5")
        assert ref.shard_index == 0
        assert ref.dtype == "float32"

    def test_round_trip(self):
        ref = _make_signal_ref(compression="gzip")
        data = ref.model_dump(mode="json")
        ref2 = SignalRef.model_validate(data)
        assert ref == ref2


# ---------------------------------------------------------------------------
# QualityInfo
# ---------------------------------------------------------------------------

class TestQualityInfo:
    def test_defaults(self):
        q = QualityInfo()
        assert q.overall_status == QualityStatus.UNKNOWN
        assert q.bad_channels == []
        assert q.snr_db is None

    def test_with_bad_channels(self):
        q = QualityInfo(
            overall_status=QualityStatus.SUSPECT,
            bad_channels=["Fp1", "F8"],
            artifact_ratio=0.15,
        )
        assert len(q.bad_channels) == 2
        assert q.artifact_ratio == 0.15


# ---------------------------------------------------------------------------
# ProcessingHistory
# ---------------------------------------------------------------------------

class TestProcessingHistory:
    def test_raw_default(self):
        h = ProcessingHistory()
        assert h.is_raw is True
        assert h.steps == []

    def test_with_steps(self):
        steps = [
            ProcessingStep(operation="raw_import", parameters={"format": "edf"}),
            ProcessingStep(
                operation="bandpass_filter",
                parameters={"l_freq": 0.5, "h_freq": 40.0, "method": "fir"},
            ),
        ]
        h = ProcessingHistory(steps=steps, is_raw=False, version_tag="filtered_0.5_40")
        assert len(h.steps) == 2
        assert h.version_tag == "filtered_0.5_40"


# ---------------------------------------------------------------------------
# Annotations (discriminated union)
# ---------------------------------------------------------------------------

class TestAnnotations:
    def test_categorical(self):
        ann = CategoricalAnnotation(
            annotation_id="ann_001",
            name="mi_class",
            annotation_type="categorical",
            value="left_hand",
        )
        assert ann.annotation_type == "categorical"
        assert ann.value == "left_hand"

    def test_numeric(self):
        ann = NumericAnnotation(
            annotation_id="ann_002",
            name="valence",
            annotation_type="numeric",
            numeric_value=0.7,
            unit="score",
        )
        assert ann.numeric_value == 0.7

    def test_text(self):
        ann = TextAnnotation(
            annotation_id="ann_003",
            name="spoken_sentence",
            annotation_type="text",
            text_value="The cat sat on the mat.",
        )
        assert "cat" in ann.text_value

    def test_continuous(self):
        ann = ContinuousAnnotation(
            annotation_id="ann_004",
            name="audio_envelope",
            annotation_type="continuous",
            scope="timepoint",
            data_ref=_make_signal_ref(
                internal_path="/atoms/abc123/annotations/audio_envelope"
            ),
            data_sampling_rate=64.0,
        )
        assert ann.data_sampling_rate == 64.0

    def test_event_sequence(self):
        events = [
            EventItem(onset=0.5, value="the"),
            EventItem(onset=0.8, value="cat", duration=0.2),
        ]
        ann = EventSequenceAnnotation(
            annotation_id="ann_005",
            name="word_onsets",
            annotation_type="event_sequence",
            events=events,
        )
        assert len(ann.events) == 2

    def test_stimulus_ref(self):
        ann = StimulusRefAnnotation(
            annotation_id="ann_006",
            name="presented_audio",
            annotation_type="stimulus_ref",
            stimulus_id="stim_audio_001",
            stimulus_onset=0.0,
            stimulus_offset=10.0,
        )
        assert ann.stimulus_id == "stim_audio_001"

    def test_binary_mask(self):
        ann = BinaryMaskAnnotation(
            annotation_id="ann_007",
            name="artifact_mask",
            annotation_type="binary_mask",
            scope="timepoint",
            mask_ref=_make_signal_ref(
                internal_path="/atoms/abc123/annotations/artifact_mask",
                dtype="uint8",
            ),
        )
        assert ann.mask_ref.dtype == "uint8"

    def test_discriminated_union_serialization(self):
        """Test that the discriminated union serializes and deserializes correctly."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(AnnotationUnion)

        # Create a categorical annotation
        cat = CategoricalAnnotation(
            annotation_id="ann_001",
            name="mi_class",
            annotation_type="categorical",
            value="left_hand",
        )

        # Serialize
        data = adapter.dump_python(cat, mode="json")
        assert data["annotation_type"] == "categorical"

        # Deserialize: should auto-dispatch to CategoricalAnnotation
        restored = adapter.validate_python(data)
        assert isinstance(restored, CategoricalAnnotation)
        assert restored.value == "left_hand"

    def test_union_dispatch_all_types(self):
        """Verify all 7 annotation types can round-trip through the union."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(AnnotationUnion)

        annotations = [
            CategoricalAnnotation(annotation_id="a1", name="c", value="v"),
            NumericAnnotation(annotation_id="a2", name="n", numeric_value=1.0),
            TextAnnotation(annotation_id="a3", name="t", text_value="hello"),
            ContinuousAnnotation(
                annotation_id="a4", name="cont",
                data_ref=_make_signal_ref(internal_path="/atoms/x/annotations/y"),
                data_sampling_rate=64.0,
            ),
            EventSequenceAnnotation(
                annotation_id="a5", name="ev",
                events=[EventItem(onset=0.0, value="start")],
            ),
            StimulusRefAnnotation(
                annotation_id="a6", name="stim", stimulus_id="s1",
            ),
            BinaryMaskAnnotation(
                annotation_id="a7", name="mask",
                mask_ref=_make_signal_ref(internal_path="/atoms/x/annotations/m"),
            ),
        ]

        for ann in annotations:
            data = adapter.dump_python(ann, mode="json")
            restored = adapter.validate_python(data)
            assert type(restored) == type(ann), f"Failed for {ann.annotation_type}"


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------

class TestAtom:
    def test_creation(self):
        atom = _make_atom()
        assert atom.atom_type == AtomType.TRIAL
        assert atom.n_channels == 64

    def test_with_annotations(self):
        atom = _make_atom(
            annotations=[
                CategoricalAnnotation(
                    annotation_id="a1", name="mi_class", value="left_hand"
                ),
                NumericAnnotation(
                    annotation_id="a2", name="confidence", numeric_value=0.95
                ),
            ]
        )
        assert len(atom.annotations) == 2
        assert isinstance(atom.annotations[0], CategoricalAnnotation)
        assert isinstance(atom.annotations[1], NumericAnnotation)

    def test_with_relations(self):
        atom = _make_atom(
            relations=[
                AtomRelation(
                    target_atom_id="next_atom",
                    relation_type="sequential_next",
                    metadata={"order_index": 1},
                ),
                AtomRelation(
                    target_atom_id="prev_window",
                    relation_type="overlapping",
                    metadata={
                        "overlap_samples": 128,
                        "overlap_ratio": 0.5,
                        "overlap_seconds": 0.5,
                    },
                ),
            ]
        )
        assert len(atom.relations) == 2
        assert atom.relations[1].metadata["overlap_ratio"] == 0.5

    def test_full_serialization_round_trip(self):
        """Full Atom with annotations, quality, provenance → JSON → Atom."""
        atom = _make_atom(
            annotations=[
                CategoricalAnnotation(
                    annotation_id="a1", name="mi_class", value="left_hand"
                ),
            ],
            quality=QualityInfo(
                overall_status=QualityStatus.GOOD,
                bad_channels=["Fp1"],
            ),
            processing_history=ProcessingHistory(
                steps=[
                    ProcessingStep(operation="raw_import", parameters={"format": "gdf"}),
                ],
                is_raw=True,
            ),
        )

        json_str = atom.model_dump_json()
        restored = Atom.model_validate_json(json_str)
        assert restored.atom_id == atom.atom_id
        assert isinstance(restored.annotations[0], CategoricalAnnotation)
        assert restored.quality.bad_channels == ["Fp1"]

    def test_window_atom(self):
        atom = _make_atom(
            atom_type=AtomType.WINDOW,
            trial_index=None,
        )
        assert atom.atom_type == AtomType.WINDOW
        assert atom.trial_index is None


# ---------------------------------------------------------------------------
# Metadata models
# ---------------------------------------------------------------------------

class TestMetadataModels:
    def test_subject_meta(self):
        s = SubjectMeta(
            subject_id="sub-01", dataset_id="test",
            age=25.0, sex="M", handedness="R",
        )
        assert s.age == 25.0

    def test_session_meta(self):
        ses = SessionMeta(
            session_id="ses-01", subject_id="sub-01", dataset_id="test",
            sampling_rate=256.0,
            device_manufacturer="BioSemi",
            reference_scheme="average",
        )
        assert ses.sampling_rate == 256.0

    def test_run_meta(self):
        run = RunMeta(
            run_id="run-01", session_id="ses-01",
            subject_id="sub-01", dataset_id="test",
            run_index=0,
            task_type="motor_imagery",
        )
        assert run.run_index == 0

    def test_dataset_meta(self):
        ds = DatasetMeta(
            dataset_id="bci_comp_iv_2a",
            name="BCI Competition IV 2a",
            task_types=["motor_imagery"],
            n_subjects=9,
        )
        assert ds.n_subjects == 9


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

class TestRecipe:
    def test_minimal_recipe(self):
        recipe = AssemblyRecipe(
            recipe_id="test_recipe",
            query={"dataset_id": ["bci_comp_iv_2a"]},
            label_fields=[
                LabelSpec(annotation_name="mi_class", output_key="label"),
            ],
        )
        assert recipe.normalization_method is None
        assert recipe.normalization_scope == NormalizationScope.PER_ATOM
        assert recipe.baseline_before_normalize is True

    def test_full_recipe(self):
        recipe = AssemblyRecipe(
            recipe_id="full_recipe",
            description="Full MI pipeline",
            query={
                "dataset_id": ["bci_comp_iv_2a"],
                "annotations": [{"name": "mi_class", "value_in": ["left_hand", "right_hand"]}],
            },
            source_version="raw",
            target_channels=["C3", "Cz", "C4"],
            target_sampling_rate=128.0,
            target_reference="average",
            target_duration=4.0,
            filter_band=(0.5, 40.0),
            normalization_method=NormalizationMethod.ZSCORE,
            normalization_scope=NormalizationScope.GLOBAL,
            baseline_correction="mean",
            label_fields=[
                LabelSpec(annotation_name="mi_class", output_key="label"),
                LabelSpec(annotation_name="subject_id", output_key="domain", encoding="ordinal"),
            ],
            split_strategy=SplitStrategy.SUBJECT,
            augmentations=[
                TemporalShiftAug(max_shift_seconds=0.2),
                ChannelDropoutAug(drop_prob=0.15),
                GaussianNoiseAug(std_uv=2.0),
            ],
            error_handling=ErrorHandling.SKIP,
        )
        assert recipe.normalization_scope == NormalizationScope.GLOBAL
        assert len(recipe.augmentations) == 3
        assert len(recipe.label_fields) == 2

    def test_augmentation_union_round_trip(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(AugmentationUnion)
        aug = TemporalShiftAug(max_shift_seconds=0.3)
        data = adapter.dump_python(aug, mode="json")
        restored = adapter.validate_python(data)
        assert isinstance(restored, TemporalShiftAug)
        assert restored.max_shift_seconds == 0.3

    def test_recipe_yaml_round_trip(self):
        """Recipe → dict → YAML string → dict → Recipe."""
        import yaml

        recipe = AssemblyRecipe(
            recipe_id="yaml_test",
            query={"dataset_id": ["test"]},
            label_fields=[
                LabelSpec(annotation_name="label", output_key="y"),
            ],
            augmentations=[ChannelDropoutAug(drop_prob=0.1)],
        )
        yaml_str = yaml.safe_dump(
            recipe.model_dump(mode="json"),
            default_flow_style=False,
        )
        data = yaml.safe_load(yaml_str)
        restored = AssemblyRecipe.model_validate(data)
        assert restored.recipe_id == "yaml_test"
        assert len(restored.augmentations) == 1


# ---------------------------------------------------------------------------
# Electrode & Channel
# ---------------------------------------------------------------------------

class TestElectrodeAndChannel:
    def test_electrode(self):
        e = ElectrodeLocation(x=0.0, y=0.0, z=1.0, coordinate_system="MNI")
        assert e.coordinate_system == "MNI"

    def test_channel_info(self):
        ch = ChannelInfo(
            channel_id="ch_0",
            index=0,
            name="EEG Fp1",
            standard_name="Fp1",
            type=ChannelType.EEG,
            unit="uV",
            sampling_rate=256.0,
            location=ElectrodeLocation(x=0.1, y=0.9, z=0.5),
        )
        assert ch.standard_name == "Fp1"
        assert ch.location.x == 0.1

    def test_montage(self):
        m = MontageInfo(
            montage_id="standard_1020",
            name="Standard 10-20",
            n_channels=21,
            channel_names=["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                           "T3", "C3", "Cz", "C4", "T4",
                           "T5", "P3", "Pz", "P4", "T6",
                           "O1", "Oz", "O2", "A1"],
        )
        assert m.n_channels == 21
