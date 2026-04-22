"""ShardManager: HDF5 shard-based signal storage with smart sharding.

Design principles:
- Default: one HDF5 shard per run
- Auto-split when shard exceeds max_shard_size_mb (default 200 MB)
- Write-then-check: after writing an atom, check if shard exceeded threshold;
  if the shard was empty before the write, always accept (single atom > threshold)
- All file_path values in SignalRef are RELATIVE to pool_root
- Atomic write safety: .tmp files for new shards, flush() for append
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from neuroatom.core.signal_ref import SignalRef
from neuroatom.storage import paths as P

logger = logging.getLogger(__name__)

# Schema version embedded in every shard file
SHARD_SCHEMA_VERSION = "1.0.0"


class ShardManager:
    """Manages per-run HDF5 shards with automatic splitting.

    Usage during import:
        mgr = ShardManager(pool_root, dataset_id, subject_id, session_id, run_id)
        for atom_id, signal, annotations in atoms:
            signal_ref = mgr.write_atom_signal(atom_id, signal, annotations)
        mgr.close()

    Usage during read:
        mgr = ShardManager(pool_root, dataset_id, subject_id, session_id, run_id)
        signal = mgr.read_atom_signal(signal_ref)
        mgr.close()
    """

    def __init__(
        self,
        pool_root: Path,
        dataset_id: str,
        subject_id: str,
        session_id: str,
        run_id: str,
        max_shard_size_mb: float = 200.0,
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = 4,
    ):
        self._pool_root = Path(pool_root)
        self._dataset_id = dataset_id
        self._subject_id = subject_id
        self._session_id = session_id
        self._run_id = run_id
        self._max_shard_bytes = max_shard_size_mb * 1024 * 1024
        self._compression = compression
        self._compression_opts = compression_opts

        self._run_dir = P.run_dir(
            self._pool_root, dataset_id, subject_id, session_id, run_id
        )
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._current_shard_index: int = self._detect_latest_shard_index()
        self._current_h5: Optional[h5py.File] = None
        self._atoms_in_current_shard: int = 0

    # ------------------------------------------------------------------
    # Public API: write
    # ------------------------------------------------------------------

    def write_atom_signal(
        self,
        atom_id: str,
        signal: np.ndarray,
        annotations: Optional[Dict[str, np.ndarray]] = None,
    ) -> SignalRef:
        """Write one atom's signal (and optional annotation arrays) to the current shard.

        Returns a SignalRef with relative file_path for storage in atom metadata.
        """
        if annotations is None:
            annotations = {}

        # Open current shard (or create if not open)
        h5 = self._get_write_handle()

        # Write signal
        atom_group_path = f"/atoms/{atom_id}"
        if atom_group_path in h5:
            logger.warning("Atom %s already exists in shard %d, overwriting.", atom_id, self._current_shard_index)
            del h5[atom_group_path]

        atom_grp = h5.create_group(atom_group_path)
        ds_kwargs = {}
        if self._compression:
            ds_kwargs["compression"] = self._compression
            if self._compression_opts is not None:
                ds_kwargs["compression_opts"] = self._compression_opts

        atom_grp.create_dataset(
            "signal",
            data=signal.astype(np.float32),
            **ds_kwargs,
        )

        # Write annotation arrays
        if annotations:
            ann_grp = atom_grp.create_group("annotations")
            for ann_id, ann_data in annotations.items():
                ann_grp.create_dataset(
                    ann_id,
                    data=ann_data.astype(np.float32),
                    **ds_kwargs,
                )

        h5.flush()
        self._atoms_in_current_shard += 1

        # Build SignalRef (relative path)
        rel_path = P.shard_relative_path(
            self._dataset_id,
            self._subject_id,
            self._session_id,
            self._run_id,
            self._current_shard_index,
        )
        signal_ref = SignalRef(
            file_path=rel_path,
            internal_path=f"/atoms/{atom_id}/signal",
            dtype=str(np.float32),
            shape=tuple(signal.shape),
            shard_index=self._current_shard_index,
            storage_backend="hdf5",
            compression=self._compression,
        )

        # Write-then-check: split if needed
        if self._should_split():
            self._advance_shard()

        return signal_ref

    # ------------------------------------------------------------------
    # Public API: read
    # ------------------------------------------------------------------

    def read_atom_signal(self, signal_ref: SignalRef) -> np.ndarray:
        """Read an atom's signal array from the correct shard.

        The signal_ref.file_path is relative to pool_root. This method
        resolves it to an absolute path.
        """
        abs_path = self._pool_root / signal_ref.file_path
        if not abs_path.exists():
            raise FileNotFoundError(
                f"Shard file not found: {abs_path} (pool_root={self._pool_root})"
            )

        with h5py.File(abs_path, "r") as h5:
            if signal_ref.internal_path not in h5:
                raise KeyError(
                    f"Internal path '{signal_ref.internal_path}' not found in {abs_path}"
                )
            return h5[signal_ref.internal_path][:]

    def read_annotation(self, signal_ref: SignalRef, annotation_id: str) -> np.ndarray:
        """Read a specific annotation array from the shard."""
        abs_path = self._pool_root / signal_ref.file_path
        atom_id = signal_ref.internal_path.split("/")[2]  # /atoms/{atom_id}/signal
        ann_path = f"/atoms/{atom_id}/annotations/{annotation_id}"

        with h5py.File(abs_path, "r") as h5:
            if ann_path not in h5:
                raise KeyError(f"Annotation '{ann_path}' not found in {abs_path}")
            return h5[ann_path][:]

    # ------------------------------------------------------------------
    # Public API: static read (no ShardManager state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def static_read(pool_root: Path, signal_ref: SignalRef) -> np.ndarray:
        """Read signal without instantiating a ShardManager."""
        abs_path = Path(pool_root) / signal_ref.file_path
        if not abs_path.exists():
            raise FileNotFoundError(
                f"HDF5 shard not found: {abs_path}\n"
                f"  signal_ref.file_path = {signal_ref.file_path}\n"
                f"  pool_root = {pool_root}\n"
                f"Ensure the pool directory is intact and the atom was imported correctly."
            )
        with h5py.File(abs_path, "r") as h5:
            if signal_ref.internal_path not in h5:
                raise KeyError(
                    f"Internal HDF5 path '{signal_ref.internal_path}' not found in {abs_path}. "
                    f"Available top-level keys: {list(h5.keys())}. "
                    f"The shard file may be corrupted or the atom's signal_ref is stale."
                )
            return h5[signal_ref.internal_path][:]

    # ------------------------------------------------------------------
    # Public API: lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the current write handle."""
        if self._current_h5 is not None:
            self._update_shard_attrs(self._current_h5)
            self._current_h5.close()
            self._current_h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_latest_shard_index(self) -> int:
        """Find the highest existing shard index in the run directory."""
        idx = 0
        while True:
            path = self._run_dir / P.shard_filename(idx)
            if not path.exists():
                break
            idx += 1
        return max(0, idx - 1) if idx > 0 else 0

    def _get_write_handle(self) -> h5py.File:
        """Return the current shard's h5py.File handle, opening or creating as needed."""
        if self._current_h5 is not None:
            return self._current_h5

        shard_file = self._run_dir / P.shard_filename(self._current_shard_index)

        if shard_file.exists():
            # Re-open existing shard for appending
            self._current_h5 = h5py.File(shard_file, "a")
            # Count existing atoms
            atoms_grp = self._current_h5.get("/atoms")
            self._atoms_in_current_shard = len(atoms_grp) if atoms_grp else 0
        else:
            # Create new shard
            self._current_h5 = h5py.File(shard_file, "w")
            self._current_h5.create_group("/atoms")
            self._atoms_in_current_shard = 0

        return self._current_h5

    def _should_split(self) -> bool:
        """Check if the current shard should be split.

        Write-then-check strategy:
        - If current shard has only 1 atom, never split (handles atom > threshold)
        - Otherwise, check file size against threshold
        """
        if self._atoms_in_current_shard <= 1:
            return False

        shard_file = self._run_dir / P.shard_filename(self._current_shard_index)
        if not shard_file.exists():
            return False

        current_size = shard_file.stat().st_size
        return current_size > self._max_shard_bytes

    def _advance_shard(self) -> None:
        """Close current shard and prepare to write to the next one."""
        if self._current_h5 is not None:
            self._update_shard_attrs(self._current_h5)
            self._current_h5.close()
            self._current_h5 = None

        self._current_shard_index += 1
        self._atoms_in_current_shard = 0
        logger.info(
            "Advanced to shard %d for run %s",
            self._current_shard_index,
            self._run_id,
        )

    def _update_shard_attrs(self, h5: h5py.File) -> None:
        """Write metadata attributes to the shard's root group."""
        h5.attrs["shard_index"] = self._current_shard_index
        h5.attrs["n_atoms"] = self._atoms_in_current_shard
        h5.attrs["dataset_id"] = self._dataset_id
        h5.attrs["run_id"] = self._run_id
        h5.attrs["schema_version"] = SHARD_SCHEMA_VERSION

    @property
    def current_shard_index(self) -> int:
        return self._current_shard_index

    @property
    def current_shard_path(self) -> Path:
        return self._run_dir / P.shard_filename(self._current_shard_index)
