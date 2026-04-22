"""MOABB Bridge: access 30+ EEG datasets via the MOABB library.

MOABB (Mother of All BCI Benchmarks) provides a unified interface to
download and load many popular BCI datasets. This bridge converts MOABB's
paradigm-based API into NeuroAtom's import pipeline.

Supported MOABB paradigms:
- MotorImagery: BCI Competition datasets, PhysioNet, etc.
- P300: BNCI, EPFL, etc.
- SSVEP: Nakanishi, etc.

Requires: pip install neuroatom[moabb]
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import ChannelType
from neuroatom.core.run import RunMeta
from neuroatom.core.session import SessionMeta
from neuroatom.core.subject import SubjectMeta
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name

logger = logging.getLogger(__name__)

try:
    import moabb
    from moabb.datasets.base import BaseDataset as MOABBBaseDataset
    from moabb.paradigms.base import BaseParadigm as MOABBBaseParadigm

    HAS_MOABB = True
except ImportError:
    HAS_MOABB = False
    MOABBBaseDataset = None
    MOABBBaseParadigm = None


def _check_moabb():
    if not HAS_MOABB:
        raise ImportError(
            "MOABB is required for the moabb_bridge importer. "
            "Install it with: pip install neuroatom[moabb]"
        )


class _MOABBRawWrapper:
    """Wrapper around MOABB raw data to provide MNE-compatible interface."""

    def __init__(self, raw, events=None):
        self._raw = raw
        self._events = events
        self.info = raw.info

    def get_data(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        return self._raw.get_data(start=start, stop=stop)

    @property
    def n_times(self):
        return self._raw.n_times

    @property
    def filenames(self):
        return getattr(self._raw, "filenames", ["<moabb>"])


class MOABBBridgeImporter(BaseImporter):
    """Bridge between MOABB library and NeuroAtom import pipeline.

    Usage:
        from moabb.datasets import BNCI2014_001  # BCI Competition IV 2a

        importer = MOABBBridgeImporter(pool, task_config)
        results = importer.import_moabb_dataset(
            dataset=BNCI2014_001(),
            atomizer=TrialAtomizer(),
        )
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)
        _check_moabb()

    @staticmethod
    def detect(path: Path) -> bool:
        """MOABB bridge doesn't detect files — it uses programmatic access."""
        return False

    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        """Not used directly. Use import_moabb_dataset() instead."""
        raise NotImplementedError(
            "MOABBBridgeImporter does not load from file paths. "
            "Use import_moabb_dataset() instead."
        )

    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        """Extract channel info from MNE Raw (MOABB provides MNE objects)."""
        if isinstance(raw, _MOABBRawWrapper):
            raw_obj = raw._raw
        else:
            raw_obj = raw

        info = raw_obj.info
        ch_infos = []
        type_overrides = self.task_config.channel_type_overrides
        exclude_set = set(self.task_config.exclude_channels)

        import mne as _mne

        for idx, ch_name in enumerate(info["ch_names"]):
            if ch_name in exclude_set:
                continue

            mne_type = _mne.channel_type(info, idx)
            if ch_name in type_overrides:
                ch_type = ChannelType(type_overrides[ch_name])
            elif mne_type == "eeg":
                ch_type = ChannelType.EEG
            elif mne_type == "eog":
                ch_type = ChannelType.EOG
            elif mne_type == "stim":
                ch_type = ChannelType.STIM
            else:
                ch_type = ChannelType.OTHER

            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{idx:03d}",
                index=idx,
                name=ch_name,
                standard_name=standardize_channel_name(ch_name),
                type=ch_type,
                unit=self.task_config.signal_unit or "V",
                sampling_rate=info["sfreq"],
            ))

        return ch_infos

    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        """Extract events from MOABB raw data."""
        if isinstance(raw, _MOABBRawWrapper) and raw._events is not None:
            return raw._events

        if hasattr(raw, "_raw"):
            raw = raw._raw

        import mne as _mne

        try:
            events = _mne.find_events(raw, verbose="WARNING")
            if events is not None and len(events) > 0:
                return events
        except (ValueError, RuntimeError):
            pass

        if raw.annotations and len(raw.annotations) > 0:
            try:
                events, event_id = _mne.events_from_annotations(
                    raw, verbose="WARNING"
                )
                if len(events) > 0:
                    return events
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # MOABB-specific import
    # ------------------------------------------------------------------

    def import_moabb_dataset(
        self,
        dataset: Any,
        atomizer: Any,
        subjects: Optional[List[int]] = None,
    ) -> List[ImportResult]:
        """Import an entire MOABB dataset into the pool.

        Args:
            dataset: MOABB BaseDataset instance.
            atomizer: Atomizer for decomposition.
            subjects: Optional list of subject IDs to import. None = all.

        Returns:
            List of ImportResult for each imported run.
        """
        _check_moabb()

        # Register dataset
        dataset_id = self.task_config.dataset_id
        dataset_name = getattr(dataset, "dataset_name", dataset.__class__.__name__)
        paradigm = getattr(dataset, "paradigm", self.task_config.task_type)

        if subjects is None:
            subjects = dataset.subject_list

        self._pool.register_dataset(DatasetMeta(
            dataset_id=dataset_id,
            name=dataset_name,
            task_types=[paradigm] if isinstance(paradigm, str) else [self.task_config.task_type],
            n_subjects=len(subjects),
        ))

        results = []

        for sub_id_int in subjects:
            sub_id = f"sub-{sub_id_int:03d}"

            self._pool.register_subject(SubjectMeta(
                subject_id=sub_id,
                dataset_id=dataset_id,
            ))

            try:
                # MOABB returns dict: {session_name: {run_name: raw}}
                data = dataset.get_data(subjects=[sub_id_int])
            except Exception as e:
                logger.error("Failed to load MOABB data for subject %s: %s", sub_id_int, e)
                continue

            for ses_name, runs in data.get(sub_id_int, {}).items():
                ses_id = f"ses-{ses_name}" if not str(ses_name).startswith("ses-") else str(ses_name)

                self._pool.register_session(SessionMeta(
                    session_id=ses_id,
                    subject_id=sub_id,
                    dataset_id=dataset_id,
                ))

                run_index = 0
                for run_name, raw in runs.items():
                    run_id = f"run-{run_name}" if not str(run_name).startswith("run-") else str(run_name)

                    # Extract events from raw
                    import mne as _mne
                    events = None
                    try:
                        events = _mne.find_events(raw, verbose="WARNING")
                    except Exception:
                        try:
                            events, _ = _mne.events_from_annotations(
                                raw, verbose="WARNING"
                            )
                        except Exception:
                            pass

                    # Wrap raw for compatibility
                    wrapped = _MOABBRawWrapper(raw, events)

                    # Build channel infos
                    channel_infos = self.extract_channel_infos(wrapped)

                    # Build run meta
                    run_meta = RunMeta(
                        run_id=run_id,
                        session_id=ses_id,
                        subject_id=sub_id,
                        dataset_id=dataset_id,
                        task_type=self.task_config.task_type,
                        run_index=run_index,
                    )

                    # Atomize
                    atoms = atomizer.atomize(
                        raw=raw,
                        events=events,
                        task_config=self.task_config,
                        run_meta=run_meta,
                        channel_infos=channel_infos,
                    )

                    # Store
                    from neuroatom.storage.metadata_store import AtomJSONLWriter
                    from neuroatom.storage.signal_store import ShardManager
                    from neuroatom.storage import paths as P

                    with ShardManager(
                        self._pool.root, dataset_id, sub_id, ses_id, run_id,
                    ) as shard_mgr:
                        jsonl_path = P.atoms_jsonl_path(
                            self._pool.root, dataset_id, sub_id, ses_id, run_id,
                        )
                        with AtomJSONLWriter(jsonl_path) as writer:
                            for atom in atoms:
                                start = atom.temporal.onset_sample
                                stop = start + atom.temporal.duration_samples
                                signal = raw.get_data(start=start, stop=stop).astype(np.float32)
                                ref = shard_mgr.write_atom_signal(atom.atom_id, signal)
                                atom.signal_ref = ref
                                writer.write_atom(atom)

                    self._pool.register_run(run_meta)

                    result = ImportResult(
                        atoms=atoms,
                        run_meta=run_meta,
                        channel_infos=channel_infos,
                    )
                    results.append(result)
                    run_index += 1

                    logger.info(
                        "Imported MOABB %s/%s/%s: %d atoms",
                        sub_id, ses_id, run_id, len(atoms),
                    )

        logger.info(
            "MOABB import complete: %d runs from %s.",
            len(results), dataset_name,
        )
        return results


# Auto-register (only if moabb is available)
if HAS_MOABB:
    register_importer("moabb", MOABBBridgeImporter)
