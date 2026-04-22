"""PyTorch Dataset and DataLoader for NeuroAtom assembled data.

Features:
- Worker-safe HDF5 access (per-worker file handles via worker_init_fn)
- Configurable error handling (raise/skip/substitute)
- Support for channel masks and time masks
- Optional augmentations at load time
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Provide stubs so the module can be imported without torch
    Dataset = object
    DataLoader = None


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for the loader module. "
            "Install it with: pip install neuroatom[torch]"
        )


class AtomDataset(Dataset if HAS_TORCH else object):
    """PyTorch Dataset wrapping assembled atom samples.

    Each sample is a dict with:
        - 'signal': np.ndarray (n_channels, n_samples)
        - 'labels': dict of label_key → encoded value
        - 'channel_mask': np.ndarray or None
        - 'time_mask': np.ndarray or None
        - 'atom_id': str
        - 'subject_id': str
        - 'dataset_id': str
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        transforms: Optional[List[Callable]] = None,
    ):
        _check_torch()
        self._samples = samples
        self._transforms = transforms or []

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        sample = self._samples[idx].copy()

        # Apply transforms
        for transform in self._transforms:
            sample = transform(sample)

        # Convert to tensors
        signal = torch.from_numpy(sample["signal"]).float()

        labels_dict = {}
        for key, value in sample["labels"].items():
            if isinstance(value, np.ndarray):
                labels_dict[key] = torch.from_numpy(value).float()
            elif isinstance(value, (int, float)):
                labels_dict[key] = torch.tensor(value).long()
            else:
                labels_dict[key] = value

        result = {
            "signal": signal,
            "labels": labels_dict,
            "atom_id": sample["atom_id"],
            "subject_id": sample["subject_id"],
            "dataset_id": sample["dataset_id"],
        }

        if sample.get("channel_mask") is not None:
            result["channel_mask"] = torch.from_numpy(sample["channel_mask"]).float()
        if sample.get("time_mask") is not None:
            result["time_mask"] = torch.from_numpy(sample["time_mask"]).float()

        return result


class HDF5AtomDataset(Dataset if HAS_TORCH else object):
    """Lazy-loading PyTorch Dataset that reads signals from HDF5 on demand.

    Worker-safe: each DataLoader worker opens its own HDF5 file handles
    via worker_init_fn, preventing concurrent access issues.

    Concurrency model:
    - Each worker gets independent h5py.File handles (opened read-only)
    - No file handle sharing across worker processes
    - worker_init_fn initializes handles; cleanup_fn closes them
    - NEVER use this during active imports to the same shards
    """

    def __init__(
        self,
        atoms: List[Dict[str, Any]],
        pool_root: Path,
        transforms: Optional[List[Callable]] = None,
        error_handling: str = "skip",
    ):
        """
        Args:
            atoms: List of dicts with at least 'atom_id', 'signal_ref', 'labels'.
            pool_root: Absolute path to the pool root.
            transforms: Optional list of transform functions.
            error_handling: 'raise', 'skip', or 'substitute'.
        """
        _check_torch()
        self._atoms = atoms
        self._pool_root = Path(pool_root)
        self._transforms = transforms or []
        self._error_handling = error_handling

        # Per-worker HDF5 file handles (set by worker_init_fn)
        self._h5_handles: Dict[str, Any] = {}

    def __len__(self):
        return len(self._atoms)

    def __getitem__(self, idx):
        atom_info = self._atoms[idx]

        try:
            signal = self._read_signal(atom_info)
        except Exception as e:
            if self._error_handling == "raise":
                raise
            elif self._error_handling == "substitute":
                logger.warning("Error reading atom %s, substituting zeros: %s",
                               atom_info.get("atom_id"), e)
                shape = atom_info.get("shape", (8, 256))
                signal = np.zeros(shape, dtype=np.float32)
            else:  # skip — return None, filtered by collate
                logger.warning("Error reading atom %s, skipping: %s",
                               atom_info.get("atom_id"), e)
                return None

        sample = {
            "signal": signal,
            "labels": atom_info.get("labels", {}),
            "atom_id": atom_info.get("atom_id", ""),
            "subject_id": atom_info.get("subject_id", ""),
            "dataset_id": atom_info.get("dataset_id", ""),
        }

        for transform in self._transforms:
            sample = transform(sample)

        result = {
            "signal": torch.from_numpy(sample["signal"]).float(),
            "atom_id": sample["atom_id"],
        }

        for key, value in sample["labels"].items():
            if isinstance(value, np.ndarray):
                result[f"label_{key}"] = torch.from_numpy(value).float()
            elif isinstance(value, (int, float)):
                result[f"label_{key}"] = torch.tensor(value).long()

        return result

    def _read_signal(self, atom_info: Dict) -> np.ndarray:
        """Read signal from HDF5 using per-worker handle."""
        import h5py

        file_path = atom_info["signal_file_path"]
        internal_path = atom_info["signal_internal_path"]

        abs_path = str(self._pool_root / file_path)

        # Use cached handle if available
        if abs_path not in self._h5_handles:
            self._h5_handles[abs_path] = h5py.File(abs_path, "r")

        h5f = self._h5_handles[abs_path]
        return h5f[internal_path][:]

    def open_handles(self) -> None:
        """Open HDF5 file handles (call in worker_init_fn)."""
        pass  # Handles are opened lazily in _read_signal

    def close_handles(self) -> None:
        """Close all HDF5 file handles (call in cleanup or worker exit)."""
        for h5f in self._h5_handles.values():
            try:
                h5f.close()
            except Exception:
                pass
        self._h5_handles.clear()

    def __del__(self):
        self.close_handles()


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn: ensure each worker has fresh HDF5 handles.

    Usage:
        loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, HDF5AtomDataset):
            # Close any inherited handles (from fork)
            dataset.close_handles()
            # Handles will be re-opened lazily per worker
            logger.debug("Worker %d: HDF5 handles reset.", worker_id)


def cleanup_fn(dataset: Any) -> None:
    """Cleanup function to close HDF5 handles after DataLoader use."""
    if isinstance(dataset, HDF5AtomDataset):
        dataset.close_handles()


def skip_none_collate(batch: List) -> Optional[Dict]:
    """Custom collate function that filters out None samples (from skip errors).

    Usage:
        loader = DataLoader(dataset, collate_fn=skip_none_collate)
    """
    _check_torch()
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
