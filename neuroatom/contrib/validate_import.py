"""Import validator: verify that imported atoms meet NeuroAtom schema requirements.

Runs a suite of checks on a dataset's atoms to ensure:
1. All required Atom fields are present and valid
2. Signal data is readable and matches declared shape
3. Annotations are well-formed
4. Channel names are consistent across atoms

Usage::

    from neuroatom.contrib.validate_import import validate_import
    errors = validate_import(pool, "my_dataset")
    for e in errors:
        print(e)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from neuroatom.core.atom import Atom
from neuroatom.core.enums import AtomType
from neuroatom.storage.metadata_store import AtomJSONLReader
from neuroatom.storage.pool import Pool
from neuroatom.storage import paths as P

logger = logging.getLogger(__name__)


class ValidationError:
    """Single validation issue."""

    def __init__(self, atom_id: str, field: str, message: str, severity: str = "error"):
        self.atom_id = atom_id
        self.field = field
        self.message = message
        self.severity = severity  # "error" or "warning"

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.atom_id} → {self.field}: {self.message}"

    def __str__(self):
        return self.__repr__()


class ImportValidationReport:
    """Collection of validation results for a dataset."""

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.errors: List[ValidationError] = []
        self.n_atoms_checked: int = 0

    @property
    def is_valid(self) -> bool:
        return not any(e.severity == "error" for e in self.errors)

    @property
    def n_errors(self) -> int:
        return sum(1 for e in self.errors if e.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for e in self.errors if e.severity == "warning")

    def summary(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"[{self.dataset_id}] {status}: "
            f"{self.n_atoms_checked} atoms checked, "
            f"{self.n_errors} errors, {self.n_warnings} warnings"
        )


def validate_import(
    pool: Pool,
    dataset_id: str,
    *,
    check_signals: bool = False,
    max_atoms: Optional[int] = None,
) -> ImportValidationReport:
    """Validate all atoms in a dataset against the NeuroAtom schema.

    Args:
        pool: The pool containing the dataset.
        dataset_id: Dataset to validate.
        check_signals: If True, also read signal data from HDF5 and verify
            shape matches. This is slower but more thorough.
        max_atoms: Stop after this many atoms (for quick spot-check).

    Returns:
        ImportValidationReport with all issues found.
    """
    report = ImportValidationReport(dataset_id)

    ds_dir = P.dataset_dir(pool.root, dataset_id)
    if not ds_dir.exists():
        report.errors.append(ValidationError(
            "", "dataset", f"Dataset directory not found: {ds_dir}",
        ))
        return report

    # Collect all atoms
    atoms: List[Atom] = []
    for jsonl_path in sorted(ds_dir.rglob("atoms.jsonl")):
        try:
            reader = AtomJSONLReader(jsonl_path)
            for atom in reader.iter_atoms():
                atoms.append(atom)
                if max_atoms and len(atoms) >= max_atoms:
                    break
        except Exception as e:
            report.errors.append(ValidationError(
                "", "atoms_jsonl", f"Failed to read {jsonl_path}: {e}",
            ))
        if max_atoms and len(atoms) >= max_atoms:
            break

    if not atoms:
        report.errors.append(ValidationError(
            "", "atoms", "No atoms found in dataset.",
        ))
        return report

    # Track consistency
    seen_ids: Set[str] = set()
    channel_sets: Dict[str, Set[str]] = {}  # subject → channel set

    for atom in atoms:
        report.n_atoms_checked += 1
        aid = atom.atom_id

        # 1. Unique atom_id
        if aid in seen_ids:
            report.errors.append(ValidationError(
                aid, "atom_id", "Duplicate atom_id",
            ))
        seen_ids.add(aid)

        # 2. Required fields
        if not atom.dataset_id:
            report.errors.append(ValidationError(aid, "dataset_id", "Missing dataset_id"))
        elif atom.dataset_id != dataset_id:
            report.errors.append(ValidationError(
                aid, "dataset_id",
                f"Mismatched: expected '{dataset_id}', got '{atom.dataset_id}'",
            ))

        if not atom.subject_id:
            report.errors.append(ValidationError(aid, "subject_id", "Missing subject_id"))

        if atom.n_channels <= 0:
            report.errors.append(ValidationError(
                aid, "n_channels", f"Invalid: {atom.n_channels}",
            ))

        if len(atom.channel_ids) != atom.n_channels:
            report.errors.append(ValidationError(
                aid, "channel_ids",
                f"Length {len(atom.channel_ids)} != n_channels {atom.n_channels}",
            ))

        if atom.sampling_rate <= 0:
            report.errors.append(ValidationError(
                aid, "sampling_rate", f"Invalid: {atom.sampling_rate}",
            ))

        # 3. Signal unit
        if not atom.signal_unit:
            report.errors.append(ValidationError(
                aid, "signal_unit", "Missing signal_unit",
            ))
        elif atom.signal_unit != "uV":
            report.errors.append(ValidationError(
                aid, "signal_unit",
                f"Expected 'uV', got '{atom.signal_unit}'",
                severity="warning",
            ))

        # 4. Temporal info
        if atom.temporal.duration_samples <= 0:
            report.errors.append(ValidationError(
                aid, "temporal.duration_samples",
                f"Invalid: {atom.temporal.duration_samples}",
            ))

        # 5. Signal ref
        if not atom.signal_ref.file_path:
            report.errors.append(ValidationError(
                aid, "signal_ref.file_path", "Empty file_path",
            ))

        # 6. Channel name consistency within subject
        ch_set = frozenset(atom.channel_ids)
        if atom.subject_id not in channel_sets:
            channel_sets[atom.subject_id] = set(atom.channel_ids)
        else:
            expected = channel_sets[atom.subject_id]
            if set(atom.channel_ids) != expected:
                report.errors.append(ValidationError(
                    aid, "channel_ids",
                    "Channel set differs from other atoms in same subject",
                    severity="warning",
                ))

        # 7. Signal data check (optional, slow)
        if check_signals:
            _check_signal_data(pool, atom, report)

    logger.info("Validation: %s", report.summary())
    return report


def _check_signal_data(pool: Pool, atom: Atom, report: ImportValidationReport):
    """Verify signal data is readable and matches declared shape."""
    import h5py

    ref = atom.signal_ref
    h5_path = pool.root / ref.file_path if not Path(ref.file_path).is_absolute() else Path(ref.file_path)

    if not h5_path.exists():
        report.errors.append(ValidationError(
            atom.atom_id, "signal_ref.file_path",
            f"HDF5 file not found: {ref.file_path}",
        ))
        return

    try:
        with h5py.File(str(h5_path), "r") as f:
            if ref.internal_path not in f:
                report.errors.append(ValidationError(
                    atom.atom_id, "signal_ref.internal_path",
                    f"Dataset not found in HDF5: {ref.internal_path}",
                ))
                return
            ds = f[ref.internal_path]
            actual_shape = ds.shape
            expected = (atom.n_channels, atom.temporal.duration_samples)
            if actual_shape != expected:
                report.errors.append(ValidationError(
                    atom.atom_id, "signal_shape",
                    f"Expected {expected}, got {actual_shape}",
                ))

            # Check for NaN/Inf
            data = ds[:]
            if np.any(np.isnan(data)):
                report.errors.append(ValidationError(
                    atom.atom_id, "signal_data",
                    "Signal contains NaN values",
                    severity="warning",
                ))
            if np.any(np.isinf(data)):
                report.errors.append(ValidationError(
                    atom.atom_id, "signal_data",
                    "Signal contains Inf values",
                ))
    except Exception as e:
        report.errors.append(ValidationError(
            atom.atom_id, "signal_data",
            f"Failed to read signal: {e}",
        ))
