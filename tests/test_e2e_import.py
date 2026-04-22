"""End-to-end test: create synthetic EEG → import → atomize → store → verify.

This test uses MNE's RawArray to create synthetic data, avoiding any
external file dependencies. It validates the full import pipeline:
Pool creation → Importer → Atomizer → ShardManager → JSONL → read-back.
"""

import numpy as np
import pytest

import mne

from neuroatom.atomizer.event import EventAtomizer
from neuroatom.atomizer.trial import TrialAtomizer
from neuroatom.atomizer.window import WindowAtomizer
from neuroatom.core.annotation import (
    CategoricalAnnotation,
    EventSequenceAnnotation,
)
from neuroatom.core.atom import Atom
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import AtomType
from neuroatom.core.session import SessionMeta
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import TaskConfig
from neuroatom.importers.mne_generic import MNEGenericImporter
from neuroatom.storage.metadata_store import AtomJSONLReader
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.storage import paths as P


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def create_synthetic_raw(
    n_channels: int = 8,
    sfreq: float = 256.0,
    duration_seconds: float = 30.0,
    n_trials: int = 6,
    seed: int = 42,
):
    """Create a synthetic MNE Raw object with embedded events.

    Returns:
        (raw, events) where events is (n_trials, 3) array.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_seconds * sfreq)

    # Channel names in standard 10-20 format
    ch_names = ["Fp1", "Fp2", "C3", "Cz", "C4", "P3", "Pz", "P4"][:n_channels]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Generate random EEG-like data (µV scale, but MNE expects V)
    data = rng.randn(n_channels, n_samples) * 20e-6  # ~20 µV amplitude

    raw = mne.io.RawArray(data, info, verbose="WARNING")

    # Create events: alternating between event IDs 1 and 2
    event_spacing = int(duration_seconds / (n_trials + 1) * sfreq)
    events = np.zeros((n_trials, 3), dtype=int)
    for i in range(n_trials):
        events[i, 0] = (i + 1) * event_spacing  # sample
        events[i, 2] = (i % 2) + 1               # event_id: 1 or 2

    return raw, events


# ---------------------------------------------------------------------------
# Test: TrialAtomizer E2E
# ---------------------------------------------------------------------------

class TestTrialAtomizerE2E:
    def test_full_pipeline_via_monkey_patch(self, tmp_path):
        """Full import pipeline using monkey-patched load_raw to inject synthetic data."""
        pool = Pool.create(tmp_path / "pool")
        pool.register_dataset(DatasetMeta(
            dataset_id="synth", name="Synthetic",
            task_types=["motor_imagery"], n_subjects=1,
        ))
        pool.register_subject(SubjectMeta(
            subject_id="sub-01", dataset_id="synth",
        ))
        pool.register_session(SessionMeta(
            session_id="ses-01", subject_id="sub-01", dataset_id="synth",
            sampling_rate=256.0,
        ))

        raw, events = create_synthetic_raw(n_channels=8, n_trials=6)

        task_config = TaskConfig({
            "dataset_id": "synth",
            "dataset_name": "Synthetic",
            "task_type": "motor_imagery",
            "trial_definition": {
                "mode": "trial",
                "anchor_events": [1, 2],
                "tmin": 0.0,
                "tmax": 2.0,
            },
            "event_mapping": {1: "left_hand", 2: "right_hand"},
        })

        importer = MNEGenericImporter(pool=pool, task_config=task_config)

        # Monkey-patch load_raw and extract_events to inject synthetic data
        importer.load_raw = lambda path: (raw, {"declared_unit": "V"})
        importer.extract_events = lambda r: events

        atomizer = TrialAtomizer()
        result = importer.import_run(
            path=tmp_path / "fake.edf",
            subject_id="sub-01",
            session_id="ses-01",
            run_id="run-01",
            atomizer=atomizer,
            run_index=0,
        )

        assert result.n_atoms == 6
        assert len(result.atoms) == 6
        assert len(result.errors) == 0

    def test_atomizer_and_storage(self, tmp_path):
        """Test atomizer + ShardManager + JSONL directly with synthetic data."""
        pool = Pool.create(tmp_path / "pool")
        pool_root = pool.root

        raw, events = create_synthetic_raw(n_channels=8, n_trials=6)

        task_config = TaskConfig({
            "dataset_id": "synth",
            "task_type": "motor_imagery",
            "trial_definition": {
                "mode": "trial",
                "anchor_events": [1, 2],
                "tmin": 0.0,
                "tmax": 2.0,
            },
            "event_mapping": {1: "left_hand", 2: "right_hand"},
        })

        from neuroatom.core.channel import ChannelInfo
        from neuroatom.core.run import RunMeta

        # Build channel infos
        channel_infos = []
        for i, ch_name in enumerate(raw.info["ch_names"]):
            channel_infos.append(ChannelInfo(
                channel_id=f"ch_{i:03d}",
                index=i,
                name=ch_name,
                standard_name=ch_name,
                unit="V",
                sampling_rate=raw.info["sfreq"],
            ))

        run_meta = RunMeta(
            run_id="run-01", session_id="ses-01",
            subject_id="sub-01", dataset_id="synth",
            task_type="motor_imagery", run_index=0,
        )

        # Atomize
        atomizer = TrialAtomizer()
        atoms = atomizer.atomize(raw, events, task_config, run_meta, channel_infos)
        assert len(atoms) == 6, f"Expected 6 atoms, got {len(atoms)}"

        # Check atom types and labels
        for atom in atoms:
            assert atom.atom_type == AtomType.TRIAL
            assert len(atom.annotations) == 1
            ann = atom.annotations[0]
            assert isinstance(ann, CategoricalAnnotation)
            assert ann.value in ("left_hand", "right_hand")

        # Check sequential relations
        assert len(atoms[1].relations) >= 1  # Has at least prev relation

        # Store signals + metadata
        dataset_id = "synth"
        subject_id = "sub-01"
        session_id = "ses-01"
        run_id = "run-01"

        from neuroatom.storage.metadata_store import AtomJSONLWriter

        with ShardManager(
            pool_root, dataset_id, subject_id, session_id, run_id,
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                pool_root, dataset_id, subject_id, session_id, run_id,
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                for atom in atoms:
                    # Extract signal
                    start = atom.temporal.onset_sample
                    stop = start + atom.temporal.duration_samples
                    signal = raw.get_data(start=start, stop=stop).astype(np.float32)

                    # Write to HDF5
                    signal_ref = shard_mgr.write_atom_signal(atom.atom_id, signal)
                    atom.signal_ref = signal_ref

                    # Write to JSONL
                    writer.write_atom(atom)

        # Verify: read back from JSONL
        reader = AtomJSONLReader(jsonl_path)
        loaded_atoms = reader.read_all()
        assert len(loaded_atoms) == 6

        # Verify: read back signals from HDF5
        for loaded_atom in loaded_atoms:
            signal = ShardManager.static_read(pool_root, loaded_atom.signal_ref)
            assert signal.shape[0] == 8  # n_channels
            assert signal.shape[1] == int(2.0 * 256.0)  # 2 seconds at 256 Hz
            assert not np.all(signal == 0)

        # Verify: atom IDs are unique
        atom_ids = [a.atom_id for a in loaded_atoms]
        assert len(set(atom_ids)) == len(atom_ids)


# ---------------------------------------------------------------------------
# Test: WindowAtomizer E2E
# ---------------------------------------------------------------------------

class TestWindowAtomizerE2E:
    def test_sliding_window(self, tmp_path):
        """Test WindowAtomizer with overlapping windows."""
        pool = Pool.create(tmp_path / "pool")
        pool_root = pool.root

        raw, events = create_synthetic_raw(
            n_channels=4, duration_seconds=20.0, n_trials=3,
        )

        task_config = TaskConfig({
            "dataset_id": "synth_cont",
            "task_type": "auditory_attention_decoding",
            "trial_definition": {
                "mode": "window",
                "window_seconds": 5.0,
                "step_seconds": 2.5,
                "annotation_boundary": "include_if_onset",
            },
            "event_mapping": {1: "attend_left", 2: "attend_right"},
        })

        from neuroatom.core.channel import ChannelInfo
        from neuroatom.core.run import RunMeta

        channel_infos = [
            ChannelInfo(
                channel_id=f"ch_{i:03d}", index=i,
                name=name, standard_name=name,
                unit="V", sampling_rate=256.0,
            )
            for i, name in enumerate(raw.info["ch_names"])
        ]

        run_meta = RunMeta(
            run_id="run-01", session_id="ses-01",
            subject_id="sub-01", dataset_id="synth_cont",
            task_type="auditory_attention_decoding", run_index=0,
        )

        # Atomize
        atomizer = WindowAtomizer()
        atoms = atomizer.atomize(raw, events, task_config, run_meta, channel_infos)

        # With 20s recording, 5s window, 2.5s step: expect ~7 windows
        expected = int((20.0 - 5.0) / 2.5) + 1  # = 7
        assert len(atoms) == expected, f"Expected {expected}, got {len(atoms)}"

        # Check window type
        for atom in atoms:
            assert atom.atom_type == AtomType.WINDOW
            assert atom.trial_index is None
            assert atom.temporal.duration_seconds == 5.0

        # Check overlap relations
        for i in range(1, len(atoms)):
            overlap_rels = [
                r for r in atoms[i].relations if r.relation_type == "overlapping"
            ]
            assert len(overlap_rels) >= 1
            metadata = overlap_rels[0].metadata
            assert metadata["overlap_ratio"] == pytest.approx(0.5, abs=0.01)

        # Store and read back
        from neuroatom.storage.metadata_store import AtomJSONLWriter

        with ShardManager(
            pool_root, "synth_cont", "sub-01", "ses-01", "run-01",
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                pool_root, "synth_cont", "sub-01", "ses-01", "run-01",
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                for atom in atoms:
                    start = atom.temporal.onset_sample
                    stop = start + atom.temporal.duration_samples
                    signal = raw.get_data(start=start, stop=stop).astype(np.float32)
                    ref = shard_mgr.write_atom_signal(atom.atom_id, signal)
                    atom.signal_ref = ref
                    writer.write_atom(atom)

        # Read back
        reader = AtomJSONLReader(jsonl_path)
        loaded = reader.read_all()
        assert len(loaded) == expected

        for atom in loaded:
            signal = ShardManager.static_read(pool_root, atom.signal_ref)
            assert signal.shape == (4, int(5.0 * 256.0))


# ---------------------------------------------------------------------------
# Test: EventAtomizer E2E
# ---------------------------------------------------------------------------

class TestEventAtomizerE2E:
    def test_event_epochs(self, tmp_path):
        """Test EventAtomizer with per-event-type windows."""
        pool = Pool.create(tmp_path / "pool")
        pool_root = pool.root

        raw, events = create_synthetic_raw(n_channels=4, n_trials=6)

        task_config = TaskConfig({
            "dataset_id": "synth_event",
            "task_type": "p300",
            "trial_definition": {
                "mode": "event_epoch",
                "event_windows": {
                    "1": {"tmin": -0.2, "tmax": 0.8, "label": "target"},
                    "2": {"tmin": -0.2, "tmax": 0.8, "label": "non_target"},
                },
                "default_tmin": -0.2,
                "default_tmax": 0.8,
            },
            "event_mapping": {1: "target", 2: "non_target"},
        })

        from neuroatom.core.channel import ChannelInfo
        from neuroatom.core.run import RunMeta

        channel_infos = [
            ChannelInfo(
                channel_id=f"ch_{i:03d}", index=i,
                name=name, standard_name=name,
                unit="V", sampling_rate=256.0,
            )
            for i, name in enumerate(raw.info["ch_names"])
        ]

        run_meta = RunMeta(
            run_id="run-01", session_id="ses-01",
            subject_id="sub-01", dataset_id="synth_event",
            task_type="p300", run_index=0,
        )

        atomizer = EventAtomizer()
        atoms = atomizer.atomize(raw, events, task_config, run_meta, channel_infos)

        assert len(atoms) == 6

        for atom in atoms:
            assert atom.atom_type == AtomType.EVENT_EPOCH
            assert len(atom.annotations) == 1
            ann = atom.annotations[0]
            assert isinstance(ann, CategoricalAnnotation)
            assert ann.value in ("target", "non_target")

        # Store and verify
        from neuroatom.storage.metadata_store import AtomJSONLWriter

        with ShardManager(
            pool_root, "synth_event", "sub-01", "ses-01", "run-01",
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                pool_root, "synth_event", "sub-01", "ses-01", "run-01",
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                for atom in atoms:
                    start = atom.temporal.onset_sample
                    stop = start + atom.temporal.duration_samples
                    signal = raw.get_data(start=start, stop=stop).astype(np.float32)
                    ref = shard_mgr.write_atom_signal(atom.atom_id, signal)
                    atom.signal_ref = ref
                    writer.write_atom(atom)

        loaded = AtomJSONLReader(jsonl_path).read_all()
        assert len(loaded) == 6

        for atom in loaded:
            signal = ShardManager.static_read(pool_root, atom.signal_ref)
            assert signal.shape[0] == 4
            expected_samples = int(1.0 * 256.0)  # 0.2 + 0.8 = 1.0 seconds
            assert signal.shape[1] == expected_samples


# ---------------------------------------------------------------------------
# Test: Validation during import
# ---------------------------------------------------------------------------

class TestValidation:
    def test_all_zero_warning(self):
        from neuroatom.utils.validation import validate_signal
        signal = np.zeros((4, 256), dtype=np.float32)
        warnings = validate_signal(signal, "test_atom", {"skip_all_zero": True})
        assert any("all-zero" in w.lower() for w in warnings)

    def test_nan_warning(self):
        from neuroatom.utils.validation import validate_signal
        signal = np.full((4, 256), np.nan, dtype=np.float32)
        warnings = validate_signal(signal, "test_atom", {"skip_all_nan": True})
        assert any("nan" in w.lower() for w in warnings)

    def test_amplitude_warning(self):
        from neuroatom.utils.validation import validate_signal
        signal = np.ones((4, 256), dtype=np.float32) * 1000
        warnings = validate_signal(
            signal, "test_atom",
            {"amplitude_range_uv": [-500, 500]},
        )
        assert any("amplitude" in w.lower() for w in warnings)

    def test_clean_signal_no_warnings(self):
        from neuroatom.utils.validation import validate_signal
        rng = np.random.RandomState(42)
        signal = rng.randn(4, 256).astype(np.float32) * 20
        warnings = validate_signal(
            signal, "test_atom",
            {"skip_all_zero": True, "skip_all_nan": True, "amplitude_range_uv": [-500, 500]},
        )
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Test: Import progress tracker
# ---------------------------------------------------------------------------

class TestImportProgress:
    def test_progress_lifecycle(self, tmp_path):
        from neuroatom.importers.progress import ImportProgress

        progress = ImportProgress(tmp_path)

        # Mark started
        progress.mark_started("ds", "s01", "ses-01", "run-01")
        assert progress.get_status("ds", "s01", "ses-01", "run-01") == "started"
        assert not progress.is_completed("ds", "s01", "ses-01", "run-01")

        # Mark completed
        progress.mark_completed("ds", "s01", "ses-01", "run-01", n_atoms=10)
        assert progress.is_completed("ds", "s01", "ses-01", "run-01")

        # Mark another as failed
        progress.mark_failed("ds", "s01", "ses-01", "run-02", error="corrupt file")
        assert progress.get_status("ds", "s01", "ses-01", "run-02") == "failed"

        # Summary
        summary = progress.summary()
        assert summary["completed"] == 1
        assert summary["failed"] == 1

        # Persistence: reload from disk
        progress2 = ImportProgress(tmp_path)
        assert progress2.is_completed("ds", "s01", "ses-01", "run-01")

        # Reset
        removed = progress2.reset_dataset("ds")
        assert removed == 2
