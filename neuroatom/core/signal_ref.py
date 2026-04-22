"""SignalRef: pointer from metadata to actual signal data in HDF5 shards."""

from typing import Optional, Tuple

from pydantic import BaseModel, Field


class SignalRef(BaseModel):
    """Reference to a signal array stored in an HDF5 shard file.

    Attributes:
        file_path: ALWAYS relative to pool_root.
            Example: "datasets/bci4_2a/subjects/A01/sessions/ses-01/runs/run-01/signals_000.h5"
            ShardManager.read_atom_signal() accepts pool_root and joins at runtime.
            JSONL and SQLite also store relative paths for pool portability.
        internal_path: HDF5 internal dataset path.
            Example: "/atoms/a1b2c3d4/signal"
        dtype: NumPy dtype string for the stored array.
        shape: Shape of the stored array, typically (n_channels, n_samples).
        shard_index: Index of the shard file this data resides in.
        storage_backend: Storage backend identifier.
        compression: Compression algorithm used, if any.
    """

    file_path: str = Field(
        ...,
        description="Path relative to pool_root pointing to the HDF5 shard file.",
    )
    internal_path: str = Field(
        ...,
        description="HDF5 internal dataset path, e.g. '/atoms/{atom_id}/signal'.",
    )
    dtype: str = Field(
        default="float32",
        description="NumPy dtype string for the stored array.",
    )
    shape: Tuple[int, ...] = Field(
        ...,
        description="Shape of the stored array, typically (n_channels, n_samples).",
    )
    shard_index: int = Field(
        default=0,
        ge=0,
        description="Index of the HDF5 shard file within the run directory.",
    )
    storage_backend: str = Field(
        default="hdf5",
        description="Storage backend identifier. Currently only 'hdf5' is supported.",
    )
    compression: Optional[str] = Field(
        default=None,
        description="Compression algorithm: 'gzip', 'lzf', or None.",
    )
