"""BaseImporter: abstract base for all format-specific importers.

Each importer is responsible for:
1. Loading a raw EEG recording from disk
2. Extracting metadata (subject, session, run, channels, events)
3. Delegating to an Atomizer for decomposition into atoms
4. Writing atoms (signals + metadata) to the pool via ShardManager

The base class provides the framework; subclasses implement format-specific
loading logic.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from neuroatom.core.atom import Atom
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.enums import ChannelType
from neuroatom.core.run import RunMeta
from neuroatom.core.session import SessionMeta
from neuroatom.core.subject import SubjectMeta
from neuroatom.storage.metadata_store import AtomJSONLWriter
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.utils.channel_names import standardize_channel_name

logger = logging.getLogger(__name__)


class TaskConfig:
    """Parsed task configuration loaded from a YAML file.

    Provides dataset-specific import parameters: event-to-label mappings,
    trial definitions, channel overrides, signal unit declarations, etc.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self._data = config_dict

    @classmethod
    def from_yaml(cls, path: Path) -> "TaskConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    @classmethod
    def builtin(cls, name: str) -> "TaskConfig":
        """Load a built-in task config by name (e.g. ``"bci_comp_iv_2a"``).

        Resolves the YAML from the installed package data so it works
        regardless of the current working directory::

            config = TaskConfig.builtin("bci_comp_iv_2a")

        Available names correspond to filenames (without ``.yaml``) in
        ``neuroatom/importers/task_configs/``.
        """
        import importlib.resources as _res

        pkg = "neuroatom.importers.task_configs"
        fname = f"{name}.yaml"
        try:
            ref = _res.files(pkg).joinpath(fname)
            text = ref.read_text(encoding="utf-8")
        except (FileNotFoundError, TypeError, ModuleNotFoundError):
            # List available configs for a helpful error message
            avail = sorted(
                r.name.removesuffix(".yaml")
                for r in _res.files(pkg).iterdir()
                if r.name.endswith(".yaml") and not r.name.startswith("_")
            )
            raise FileNotFoundError(
                f"No built-in task config '{name}'. "
                f"Available: {avail}"
            ) from None
        data = yaml.safe_load(text) or {}
        return cls(data)

    @property
    def dataset_id(self) -> str:
        return self._data.get("dataset_id", "unknown")

    @property
    def dataset_name(self) -> str:
        return self._data.get("dataset_name", self.dataset_id)

    @property
    def task_type(self) -> str:
        return self._data.get("task_type", "other")

    @property
    def trial_definition(self) -> Dict[str, Any]:
        return self._data.get("trial_definition", {})

    @property
    def event_mapping(self) -> Dict[str, Any]:
        return self._data.get("event_mapping", {})

    @property
    def signal_unit(self) -> Optional[str]:
        return self._data.get("signal_unit", None)

    @property
    def channel_type_overrides(self) -> Dict[str, str]:
        return self._data.get("channel_type_overrides", {})

    @property
    def exclude_channels(self) -> List[str]:
        return self._data.get("exclude_channels", [])

    @property
    def data(self) -> Dict[str, Any]:
        return self._data


class ImportResult:
    """Result of importing a single run."""

    def __init__(
        self,
        atoms: List[Atom],
        run_meta: RunMeta,
        channel_infos: List[ChannelInfo],
        n_atoms: int = 0,
        warnings: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
    ):
        self.atoms = atoms
        self.run_meta = run_meta
        self.channel_infos = channel_infos
        self.n_atoms = n_atoms or len(atoms)
        self.warnings = warnings or []
        self.errors = errors or []


class BaseImporter(ABC):
    """Abstract base class for format-specific EEG data importers.

    Subclasses must implement:
    - ``detect(path)`` → bool: can this importer handle the given file?
    - ``load_raw(path, task_config)`` → raw data + metadata
    - ``extract_channel_infos(raw)`` → list of ChannelInfo
    - ``extract_events(raw, task_config)`` → events array
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        self._pool = pool
        self._task_config = task_config

    @property
    def pool(self) -> Pool:
        return self._pool

    @property
    def task_config(self) -> TaskConfig:
        return self._task_config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def detect(path: Path) -> bool:
        """Return True if this importer can handle the given file/directory."""
        ...

    @abstractmethod
    def load_raw(
        self, path: Path
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load raw data from the given path.

        Returns:
            Tuple of (raw_data, extra_metadata_dict).
            raw_data: framework-specific raw object (e.g., mne.io.Raw).
            extra_metadata_dict: any additional metadata extracted during loading.
        """
        ...

    @abstractmethod
    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        """Extract ChannelInfo models from the raw data object."""
        ...

    @abstractmethod
    def extract_events(
        self, raw: Any
    ) -> Optional[np.ndarray]:
        """Extract event markers from the raw data.

        Returns:
            NumPy array of shape (n_events, 3) with columns [sample, prev_id, event_id],
            or None if no events.
        """
        ...

    # ------------------------------------------------------------------
    # Template method: full import pipeline
    # ------------------------------------------------------------------

    def import_run(
        self,
        path: Path,
        subject_id: str,
        session_id: str,
        run_id: str,
        atomizer: Any,
        run_index: Optional[int] = None,
    ) -> ImportResult:
        """Import a single run: load → extract metadata → atomize → store.

        This is the main entry point for importing data.
        """
        from neuroatom.utils.validation import validate_signal

        dataset_id = self._task_config.dataset_id

        # 1. Load raw data
        logger.info("Loading %s for %s/%s/%s...", path, subject_id, session_id, run_id)
        raw, extra_meta = self.load_raw(path)

        # 2. Extract channel info
        channel_infos = self.extract_channel_infos(raw)

        # Standardize channel names
        for ch in channel_infos:
            if ch.standard_name is None:
                ch.standard_name = standardize_channel_name(ch.name)

        # 3. Extract events
        events = self.extract_events(raw)

        # 4. Build RunMeta
        run_meta = RunMeta(
            run_id=run_id,
            session_id=session_id,
            subject_id=subject_id,
            dataset_id=dataset_id,
            run_index=run_index,
            task_type=self._task_config.task_type,
            n_events=len(events) if events is not None else 0,
            paradigm_details=extra_meta.get("paradigm_details"),
        )

        # 5. Atomize
        logger.info("Atomizing %d events...", len(events) if events is not None else 0)
        atoms = atomizer.atomize(
            raw=raw,
            events=events,
            task_config=self._task_config,
            run_meta=run_meta,
            channel_infos=channel_infos,
        )
        run_meta.n_trials = len(atoms)

        # 6. Store: signals to HDF5 shards, metadata to JSONL
        warnings = []
        max_shard_mb = self._pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
        compression = self._pool.config.get("storage", {}).get("compression", "gzip")

        from neuroatom.storage import paths as P

        with ShardManager(
            pool_root=self._pool.root,
            dataset_id=dataset_id,
            subject_id=subject_id,
            session_id=session_id,
            run_id=run_id,
            max_shard_size_mb=max_shard_mb,
            compression=compression,
        ) as shard_mgr:
            jsonl_path = P.atoms_jsonl_path(
                self._pool.root, dataset_id, subject_id, session_id, run_id
            )
            with AtomJSONLWriter(jsonl_path) as jsonl_writer:
                for atom in atoms:
                    # Extract signal data from raw
                    signal = self._extract_atom_signal(raw, atom, channel_infos)

                    # Validate signal
                    validation_warnings = validate_signal(
                        signal=signal,
                        atom_id=atom.atom_id,
                        config=self._pool.config.get("import", {}),
                    )
                    warnings.extend(validation_warnings)

                    # Extract annotation arrays
                    ann_arrays = self._extract_annotation_arrays(raw, atom)

                    # Write to shard
                    signal_ref = shard_mgr.write_atom_signal(
                        atom.atom_id, signal, ann_arrays
                    )

                    # Update atom's signal_ref with actual storage location
                    atom.signal_ref = signal_ref

                    # Write metadata
                    jsonl_writer.write_atom(atom)

                jsonl_writer.flush()

        # 7. Register metadata in pool
        self._pool.register_run(run_meta)

        logger.info(
            "Imported run %s/%s/%s/%s: %d atoms",
            dataset_id, subject_id, session_id, run_id, len(atoms),
        )

        return ImportResult(
            atoms=atoms,
            run_meta=run_meta,
            channel_infos=channel_infos,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Helpers (overridable)
    # ------------------------------------------------------------------

    def _extract_atom_signal(
        self, raw: Any, atom: Atom, channel_infos: List[ChannelInfo]
    ) -> np.ndarray:
        """Extract the signal array for a single atom from the raw data.

        Default implementation uses MNE Raw API. Override for non-MNE formats.
        """
        start = atom.temporal.onset_sample
        stop = start + atom.temporal.duration_samples
        data = raw.get_data(start=start, stop=stop)
        return data.astype(np.float32)

    def _extract_annotation_arrays(
        self, raw: Any, atom: Atom
    ) -> Dict[str, np.ndarray]:
        """Extract annotation data arrays for a single atom.

        Override to extract continuous annotations (audio envelopes, etc.).
        Default returns empty dict.
        """
        return {}
