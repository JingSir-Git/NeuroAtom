"""MATLAB .mat file format detection and compatibility helpers.

MATLAB .mat files come in two major formats:
- v5 (MATLAB 5-7.2): Standard format read by scipy.io.loadmat
- v7.3 (MATLAB 7.3+): HDF5-based format read by h5py

Using the wrong reader on the wrong format gives cryptic errors.
This module provides clear detection and error messages.
"""

import logging
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


def detect_mat_version(path: Path) -> Optional[Literal["v5", "v7.3"]]:
    """Detect .mat file format version by inspecting file header bytes.

    Returns:
        "v5" for traditional MATLAB 5.0 format (scipy.io compatible).
        "v7.3" for HDF5-based MATLAB 7.3 format (h5py compatible).
        None if file is not a recognizable .mat file.
    """
    path = Path(path)
    try:
        with open(path, "rb") as f:
            header = f.read(16)
    except OSError as exc:
        logger.debug("Cannot read header of %s: %s", path, exc)
        return None

    if len(header) < 8:
        return None

    # HDF5 magic number: \x89HDF\r\n\x1a\n
    if header[:8] == b"\x89HDF\r\n\x1a\n":
        return "v7.3"

    # MATLAB v5: first 116 bytes are a text header starting with "MATLAB 5.0"
    if header[:10] == b"MATLAB 5.0":
        return "v5"

    return None


def require_mat_v5(path: Path, importer_name: str) -> None:
    """Assert that a .mat file is v5 format (scipy.io compatible).

    Raises ValueError with a helpful message if the file is v7.3 / HDF5.
    """
    version = detect_mat_version(path)
    if version == "v7.3":
        raise ValueError(
            f"{importer_name}: {path.name} is MATLAB v7.3 (HDF5) format, "
            f"but this importer requires v5 format. "
            f"Please resave the file in MATLAB with: "
            f"save('{path.stem}.mat', '-v7')"
        )


def require_mat_v73(path: Path, importer_name: str) -> None:
    """Assert that a .mat file is v7.3 / HDF5 format (h5py compatible).

    Raises ValueError with a helpful message if the file is v5.
    """
    version = detect_mat_version(path)
    if version == "v5":
        raise ValueError(
            f"{importer_name}: {path.name} is MATLAB v5 format, "
            f"but this importer requires v7.3 (HDF5) format. "
            f"Please resave the file in MATLAB with: "
            f"save('{path.stem}.mat', '-v7.3')"
        )
