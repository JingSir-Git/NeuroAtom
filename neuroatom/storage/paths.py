"""Path conventions for the NeuroAtom resource pool directory structure.

All paths returned are relative to pool_root unless explicitly stated.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Pool-level paths
# ---------------------------------------------------------------------------

POOL_CONFIG_FILENAME = "pool.yaml"
INDEX_DB_FILENAME = "index.db"
IMPORT_PROGRESS_FILENAME = "import_progress.json"
RELATIONS_FILENAME = "relations.jsonl"
STIMULI_DIR = "stimuli"
MONTAGES_DIR = "montages"
DATASETS_DIR = "datasets"


def pool_config_path(pool_root: Path) -> Path:
    return pool_root / POOL_CONFIG_FILENAME


def index_db_path(pool_root: Path) -> Path:
    return pool_root / INDEX_DB_FILENAME


def import_progress_path(pool_root: Path) -> Path:
    return pool_root / IMPORT_PROGRESS_FILENAME


def stimuli_dir(pool_root: Path) -> Path:
    return pool_root / STIMULI_DIR


def montages_dir(pool_root: Path) -> Path:
    return pool_root / MONTAGES_DIR


def datasets_dir(pool_root: Path) -> Path:
    return pool_root / DATASETS_DIR


# ---------------------------------------------------------------------------
# Dataset-level paths
# ---------------------------------------------------------------------------

def dataset_dir(pool_root: Path, dataset_id: str) -> Path:
    return datasets_dir(pool_root) / dataset_id


def dataset_meta_path(pool_root: Path, dataset_id: str) -> Path:
    return dataset_dir(pool_root, dataset_id) / "dataset.json"


def dataset_lock_path(pool_root: Path, dataset_id: str) -> Path:
    return dataset_dir(pool_root, dataset_id) / ".lock"


# ---------------------------------------------------------------------------
# Subject-level paths
# ---------------------------------------------------------------------------

def subject_dir(pool_root: Path, dataset_id: str, subject_id: str) -> Path:
    return dataset_dir(pool_root, dataset_id) / "subjects" / subject_id


def subject_meta_path(pool_root: Path, dataset_id: str, subject_id: str) -> Path:
    return subject_dir(pool_root, dataset_id, subject_id) / "subject.json"


# ---------------------------------------------------------------------------
# Session-level paths
# ---------------------------------------------------------------------------

def session_dir(
    pool_root: Path, dataset_id: str, subject_id: str, session_id: str
) -> Path:
    return subject_dir(pool_root, dataset_id, subject_id) / "sessions" / session_id


def session_meta_path(
    pool_root: Path, dataset_id: str, subject_id: str, session_id: str
) -> Path:
    return session_dir(pool_root, dataset_id, subject_id, session_id) / "session.json"


def channels_path(
    pool_root: Path, dataset_id: str, subject_id: str, session_id: str
) -> Path:
    return session_dir(pool_root, dataset_id, subject_id, session_id) / "channels.json"


def electrodes_path(
    pool_root: Path, dataset_id: str, subject_id: str, session_id: str
) -> Path:
    return session_dir(pool_root, dataset_id, subject_id, session_id) / "electrodes.json"


# ---------------------------------------------------------------------------
# Run-level paths
# ---------------------------------------------------------------------------

def run_dir(
    pool_root: Path,
    dataset_id: str,
    subject_id: str,
    session_id: str,
    run_id: str,
) -> Path:
    return (
        session_dir(pool_root, dataset_id, subject_id, session_id) / "runs" / run_id
    )


def run_meta_path(
    pool_root: Path,
    dataset_id: str,
    subject_id: str,
    session_id: str,
    run_id: str,
) -> Path:
    return run_dir(pool_root, dataset_id, subject_id, session_id, run_id) / "run.json"


def events_path(
    pool_root: Path,
    dataset_id: str,
    subject_id: str,
    session_id: str,
    run_id: str,
) -> Path:
    return run_dir(pool_root, dataset_id, subject_id, session_id, run_id) / "events.json"


def atoms_jsonl_path(
    pool_root: Path,
    dataset_id: str,
    subject_id: str,
    session_id: str,
    run_id: str,
) -> Path:
    return run_dir(pool_root, dataset_id, subject_id, session_id, run_id) / "atoms.jsonl"


def shard_filename(shard_index: int) -> str:
    """Generate shard filename: signals_000.h5, signals_001.h5, etc."""
    return f"signals_{shard_index:03d}.h5"


def shard_path(
    pool_root: Path,
    dataset_id: str,
    subject_id: str,
    session_id: str,
    run_id: str,
    shard_index: int,
) -> Path:
    return (
        run_dir(pool_root, dataset_id, subject_id, session_id, run_id)
        / shard_filename(shard_index)
    )


def shard_relative_path(
    dataset_id: str,
    subject_id: str,
    session_id: str,
    run_id: str,
    shard_index: int,
) -> str:
    """Return the shard path relative to pool_root (for SignalRef.file_path)."""
    parts = [
        DATASETS_DIR,
        dataset_id,
        "subjects",
        subject_id,
        "sessions",
        session_id,
        "runs",
        run_id,
        shard_filename(shard_index),
    ]
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Stimulus-level paths
# ---------------------------------------------------------------------------

def stimulus_dir(pool_root: Path, stimulus_id: str) -> Path:
    return stimuli_dir(pool_root) / stimulus_id


def stimulus_meta_path(pool_root: Path, stimulus_id: str) -> Path:
    return stimulus_dir(pool_root, stimulus_id) / "stimulus.json"


def stimulus_data_dir(pool_root: Path, stimulus_id: str) -> Path:
    return stimulus_dir(pool_root, stimulus_id) / "data"
