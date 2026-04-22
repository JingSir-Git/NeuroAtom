"""Pool: top-level manager for the NeuroAtom resource pool.

Handles pool initialization, dataset registration, metadata CRUD,
and dataset-level file locking for concurrent import safety.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from filelock import FileLock

from neuroatom.core.dataset_meta import DatasetMeta
from neuroatom.core.run import RunMeta
from neuroatom.core.session import SessionMeta
from neuroatom.core.subject import SubjectMeta
from neuroatom.storage import paths as P
from neuroatom.storage.metadata_store import read_json, write_json

logger = logging.getLogger(__name__)


class Pool:
    """Top-level manager for a NeuroAtom resource pool.

    A pool is a directory tree containing:
    - pool.yaml: configuration
    - index.db: SQLite query accelerator
    - datasets/: per-dataset data
    - stimuli/: shared stimulus resources
    - montages/: electrode montage definitions

    Concurrency model:
    - Dataset-level file locks (.lock per dataset dir)
    - Different datasets can be imported in parallel
    - Same dataset: serialized writes
    - Reads: always lock-free (SQLite WAL handles concurrent reads)
    """

    def __init__(self, pool_root: Path):
        self._root = Path(pool_root).resolve()
        if not self._root.exists():
            raise FileNotFoundError(f"Pool root does not exist: {self._root}")
        config_path = P.pool_config_path(self._root)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Pool config not found at {config_path}. "
                "Use Pool.create() to initialize a new pool."
            )
        self._config = self._load_config()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def config(self) -> Dict:
        return self._config

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, pool_root: Path, config_overrides: Optional[Dict] = None) -> "Pool":
        """Initialize a new empty pool at the given directory.

        Creates the directory structure and default pool.yaml.
        """
        pool_root = Path(pool_root).resolve()

        # Warn if pool already exists (create is idempotent but may be unintentional)
        config_path = P.pool_config_path(pool_root)
        if config_path.exists():
            logger.warning(
                "Pool already exists at %s (pool.yaml found). "
                "Config will be overwritten. Use Pool(pool_root) to open an existing pool.",
                pool_root,
            )

        pool_root.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        P.datasets_dir(pool_root).mkdir(exist_ok=True)
        P.stimuli_dir(pool_root).mkdir(exist_ok=True)
        P.montages_dir(pool_root).mkdir(exist_ok=True)

        # Write default config
        default_config = cls._default_config()
        if config_overrides:
            cls._deep_merge(default_config, config_overrides)

        config_path = P.pool_config_path(pool_root)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(default_config, f, default_flow_style=False, allow_unicode=True)

        logger.info("Created new pool at %s", pool_root)
        return cls(pool_root)

    # ------------------------------------------------------------------
    # Dataset CRUD
    # ------------------------------------------------------------------

    def register_dataset(self, meta: DatasetMeta) -> None:
        """Register a new dataset in the pool."""
        ds_dir = P.dataset_dir(self._root, meta.dataset_id)
        ds_dir.mkdir(parents=True, exist_ok=True)
        write_json(meta, P.dataset_meta_path(self._root, meta.dataset_id))
        logger.info("Registered dataset: %s", meta.dataset_id)

    def get_dataset_meta(self, dataset_id: str) -> DatasetMeta:
        """Load dataset metadata."""
        path = P.dataset_meta_path(self._root, dataset_id)
        return read_json(path, DatasetMeta)

    def list_datasets(self) -> List[str]:
        """List all dataset IDs in the pool."""
        ds_dir = P.datasets_dir(self._root)
        if not ds_dir.exists():
            return []
        return sorted(
            d.name
            for d in ds_dir.iterdir()
            if d.is_dir() and (d / "dataset.json").exists()
        )

    def delete_dataset(self, dataset_id: str) -> None:
        """Remove a dataset and all its data from the pool."""
        ds_dir = P.dataset_dir(self._root, dataset_id)
        if ds_dir.exists():
            shutil.rmtree(ds_dir)
            logger.info("Deleted dataset: %s", dataset_id)
        else:
            logger.warning("Dataset not found for deletion: %s", dataset_id)

    # ------------------------------------------------------------------
    # Subject CRUD
    # ------------------------------------------------------------------

    def register_subject(self, meta: SubjectMeta) -> None:
        """Register a subject within a dataset."""
        sub_dir = P.subject_dir(self._root, meta.dataset_id, meta.subject_id)
        sub_dir.mkdir(parents=True, exist_ok=True)
        write_json(meta, P.subject_meta_path(self._root, meta.dataset_id, meta.subject_id))

    def get_subject_meta(self, dataset_id: str, subject_id: str) -> SubjectMeta:
        path = P.subject_meta_path(self._root, dataset_id, subject_id)
        return read_json(path, SubjectMeta)

    def list_subjects(self, dataset_id: str) -> List[str]:
        sub_base = P.dataset_dir(self._root, dataset_id) / "subjects"
        if not sub_base.exists():
            return []
        return sorted(
            d.name
            for d in sub_base.iterdir()
            if d.is_dir() and (d / "subject.json").exists()
        )

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def register_session(self, meta: SessionMeta) -> None:
        ses_dir = P.session_dir(
            self._root, meta.dataset_id, meta.subject_id, meta.session_id
        )
        ses_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            meta,
            P.session_meta_path(
                self._root, meta.dataset_id, meta.subject_id, meta.session_id
            ),
        )

    def get_session_meta(
        self, dataset_id: str, subject_id: str, session_id: str
    ) -> SessionMeta:
        path = P.session_meta_path(self._root, dataset_id, subject_id, session_id)
        return read_json(path, SessionMeta)

    def list_sessions(self, dataset_id: str, subject_id: str) -> List[str]:
        ses_base = P.subject_dir(self._root, dataset_id, subject_id) / "sessions"
        if not ses_base.exists():
            return []
        return sorted(
            d.name
            for d in ses_base.iterdir()
            if d.is_dir() and (d / "session.json").exists()
        )

    # ------------------------------------------------------------------
    # Run CRUD
    # ------------------------------------------------------------------

    def register_run(self, meta: RunMeta) -> None:
        run_d = P.run_dir(
            self._root, meta.dataset_id, meta.subject_id, meta.session_id, meta.run_id
        )
        run_d.mkdir(parents=True, exist_ok=True)
        write_json(
            meta,
            P.run_meta_path(
                self._root, meta.dataset_id, meta.subject_id,
                meta.session_id, meta.run_id,
            ),
        )

    def get_run_meta(
        self, dataset_id: str, subject_id: str, session_id: str, run_id: str
    ) -> RunMeta:
        path = P.run_meta_path(
            self._root, dataset_id, subject_id, session_id, run_id
        )
        return read_json(path, RunMeta)

    def list_runs(
        self, dataset_id: str, subject_id: str, session_id: str
    ) -> List[str]:
        run_base = (
            P.session_dir(self._root, dataset_id, subject_id, session_id) / "runs"
        )
        if not run_base.exists():
            return []
        return sorted(
            d.name
            for d in run_base.iterdir()
            if d.is_dir() and (d / "run.json").exists()
        )

    # ------------------------------------------------------------------
    # Ensure-if-not-exists helpers (idempotent create-or-skip)
    # ------------------------------------------------------------------

    def ensure_dataset(self, dataset_id: str, name: Optional[str] = None) -> None:
        """Create dataset directory and minimal metadata if it doesn't exist."""
        ds_dir = P.dataset_dir(self._root, dataset_id)
        meta_path = P.dataset_meta_path(self._root, dataset_id)
        if meta_path.exists():
            return
        ds_dir.mkdir(parents=True, exist_ok=True)
        meta = DatasetMeta(dataset_id=dataset_id, name=name or dataset_id)
        write_json(meta, meta_path)
        logger.debug("Ensured dataset: %s", dataset_id)

    def ensure_subject(self, dataset_id: str, subject_id: str) -> None:
        """Create subject directory and minimal metadata if it doesn't exist."""
        meta_path = P.subject_meta_path(self._root, dataset_id, subject_id)
        if meta_path.exists():
            return
        sub_dir = P.subject_dir(self._root, dataset_id, subject_id)
        sub_dir.mkdir(parents=True, exist_ok=True)
        meta = SubjectMeta(subject_id=subject_id, dataset_id=dataset_id)
        write_json(meta, meta_path)
        logger.debug("Ensured subject: %s/%s", dataset_id, subject_id)

    def ensure_session(
        self, dataset_id: str, subject_id: str, session_id: str,
        sampling_rate: float = 128.0,
    ) -> None:
        """Create session directory and minimal metadata if it doesn't exist."""
        meta_path = P.session_meta_path(self._root, dataset_id, subject_id, session_id)
        if meta_path.exists():
            return
        ses_dir = P.session_dir(self._root, dataset_id, subject_id, session_id)
        ses_dir.mkdir(parents=True, exist_ok=True)
        meta = SessionMeta(
            session_id=session_id,
            subject_id=subject_id,
            dataset_id=dataset_id,
            sampling_rate=sampling_rate,
        )
        write_json(meta, meta_path)
        logger.debug("Ensured session: %s/%s/%s", dataset_id, subject_id, session_id)

    def ensure_run(
        self, dataset_id: str, subject_id: str, session_id: str, run_id: str
    ) -> None:
        """Create run directory if it doesn't exist (metadata written by register_run)."""
        run_d = P.run_dir(self._root, dataset_id, subject_id, session_id, run_id)
        run_d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset-level locking
    # ------------------------------------------------------------------

    def dataset_lock(self, dataset_id: str) -> FileLock:
        """Return a FileLock for the given dataset.

        Usage:
            with pool.dataset_lock(dataset_id):
                # safe to write to this dataset
                ...
        """
        lock_path = P.dataset_lock_path(self._root, dataset_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(lock_path), timeout=300)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_config() -> Dict:
        """Load the default pool config from package data."""
        import importlib.resources as pkg_resources

        config_ref = pkg_resources.files("neuroatom.configs").joinpath("default_pool.yaml")
        with pkg_resources.as_file(config_ref) as config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    def _load_config(self) -> Dict:
        config_path = P.pool_config_path(self._root)
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _deep_merge(base: Dict, overrides: Dict) -> None:
        """Recursively merge overrides into base dict (in-place)."""
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Pool._deep_merge(base[key], value)
            else:
                base[key] = value
