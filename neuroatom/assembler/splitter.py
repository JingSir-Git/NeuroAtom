"""Splitter: train/val/test split strategies.

Supports 5 strategies:
- subject: group by subject (no subject leakage)
- dataset: group by dataset
- temporal: split by time (earlier → train, later → test)
- predefined: use explicit lists from split_config
- stratified: stratified sampling maintaining label distribution
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from neuroatom.core.atom import Atom
from neuroatom.core.enums import SplitStrategy

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split atom lists into train/val/test sets.

    Usage:
        splitter = DataSplitter(
            strategy=SplitStrategy.SUBJECT,
            config={"val_ratio": 0.15, "test_ratio": 0.15, "seed": 42},
        )
        splits = splitter.split(atoms)
        # → {"train": [...], "val": [...], "test": [...]}
    """

    def __init__(
        self,
        strategy: SplitStrategy,
        config: Dict[str, Any],
    ):
        self._strategy = strategy
        self._config = config

    def split(self, atoms: List[Atom]) -> Dict[str, List[Atom]]:
        """Split atoms into train/val/test sets."""
        if self._strategy == SplitStrategy.SUBJECT:
            return self._split_by_subject(atoms)
        elif self._strategy == SplitStrategy.DATASET:
            return self._split_by_dataset(atoms)
        elif self._strategy == SplitStrategy.TEMPORAL:
            return self._split_temporal(atoms)
        elif self._strategy == SplitStrategy.PREDEFINED:
            return self._split_predefined(atoms)
        elif self._strategy == SplitStrategy.STRATIFIED:
            return self._split_stratified(atoms)
        else:
            raise ValueError(f"Unknown split strategy: {self._strategy}")

    def _split_by_subject(self, atoms: List[Atom]) -> Dict[str, List[Atom]]:
        """Group atoms by subject, then assign groups to splits.

        Uses composite key (dataset_id, subject_id) to prevent cross-dataset
        subject collision — e.g. two datasets both having "S01" are treated
        as different subjects.
        """
        config = self._config
        val_ratio = config.get("val_ratio", 0.15)
        test_ratio = config.get("test_ratio", 0.15)
        seed = config.get("seed", 42)
        test_subjects = config.get("test_subjects", None)
        val_subjects = config.get("val_subjects", None)

        # Group by composite key (dataset_id|subject_id) to avoid collision
        subject_atoms: Dict[str, List[Atom]] = defaultdict(list)
        for atom in atoms:
            composite_key = f"{atom.dataset_id}|{atom.subject_id}"
            subject_atoms[composite_key].append(atom)

        subjects = sorted(subject_atoms.keys())
        rng = np.random.RandomState(seed)

        if test_subjects and val_subjects:
            # Explicit assignment
            train_subs = [s for s in subjects if s not in test_subjects and s not in val_subjects]
        elif test_subjects:
            remaining = [s for s in subjects if s not in test_subjects]
            rng.shuffle(remaining)
            n_val = max(1, int(len(remaining) * val_ratio / (1 - test_ratio)))
            val_subs = remaining[:n_val]
            train_subs = remaining[n_val:]
            val_subjects = val_subs
        else:
            rng.shuffle(subjects)
            n_test = max(1, int(len(subjects) * test_ratio)) if test_ratio > 0 else 0
            n_val = max(1, int(len(subjects) * val_ratio)) if val_ratio > 0 else 0
            test_subjects = subjects[:n_test]
            val_subjects = subjects[n_test:n_test + n_val]
            train_subs = subjects[n_test + n_val:]

        test_set = set(test_subjects)
        val_set = set(val_subjects)

        result: Dict[str, List[Atom]] = {"train": [], "val": [], "test": []}
        for sub, sub_atoms in subject_atoms.items():
            if sub in test_set:
                result["test"].extend(sub_atoms)
            elif sub in val_set:
                result["val"].extend(sub_atoms)
            else:
                result["train"].extend(sub_atoms)

        logger.info(
            "Subject split: train=%d (%d subs), val=%d, test=%d",
            len(result["train"]),
            len(subjects) - len(test_set) - len(val_set),
            len(result["val"]),
            len(result["test"]),
        )
        return result

    def _split_by_dataset(self, atoms: List[Atom]) -> Dict[str, List[Atom]]:
        """Group by dataset, assign to splits."""
        config = self._config
        test_datasets = set(config.get("test_datasets", []))
        val_datasets = set(config.get("val_datasets", []))

        result: Dict[str, List[Atom]] = {"train": [], "val": [], "test": []}
        for atom in atoms:
            if atom.dataset_id in test_datasets:
                result["test"].append(atom)
            elif atom.dataset_id in val_datasets:
                result["val"].append(atom)
            else:
                result["train"].append(atom)

        return result

    def _split_temporal(self, atoms: List[Atom]) -> Dict[str, List[Atom]]:
        """Split by temporal order (earlier trials → train)."""
        config = self._config
        val_ratio = config.get("val_ratio", 0.15)
        test_ratio = config.get("test_ratio", 0.15)

        # Sort by onset
        sorted_atoms = sorted(atoms, key=lambda a: a.temporal.onset_seconds)
        n = len(sorted_atoms)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        return {
            "train": sorted_atoms[:n - n_test - n_val],
            "val": sorted_atoms[n - n_test - n_val:n - n_test],
            "test": sorted_atoms[n - n_test:],
        }

    def _split_predefined(self, atoms: List[Atom]) -> Dict[str, List[Atom]]:
        """Use explicit atom_id or subject_id lists from config."""
        config = self._config
        test_ids = set(config.get("test_atom_ids", []))
        val_ids = set(config.get("val_atom_ids", []))

        result: Dict[str, List[Atom]] = {"train": [], "val": [], "test": []}
        for atom in atoms:
            if atom.atom_id in test_ids:
                result["test"].append(atom)
            elif atom.atom_id in val_ids:
                result["val"].append(atom)
            else:
                result["train"].append(atom)

        return result

    def _split_stratified(self, atoms: List[Atom]) -> Dict[str, List[Atom]]:
        """Stratified random split maintaining label distribution."""
        config = self._config
        val_ratio = config.get("val_ratio", 0.15)
        test_ratio = config.get("test_ratio", 0.15)
        seed = config.get("seed", 42)
        stratify_by = config.get("stratify_by", "trial_label")

        rng = np.random.RandomState(seed)

        # Group by label
        label_atoms: Dict[str, List[Atom]] = defaultdict(list)
        for atom in atoms:
            label = "unknown"
            for ann in atom.annotations:
                if hasattr(ann, "value") and ann.name == stratify_by:
                    label = ann.value
                    break
            label_atoms[label].append(atom)

        result: Dict[str, List[Atom]] = {"train": [], "val": [], "test": []}
        for label, group in label_atoms.items():
            rng.shuffle(group)
            n = len(group)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))

            result["test"].extend(group[:n_test])
            result["val"].extend(group[n_test:n_test + n_val])
            result["train"].extend(group[n_test + n_val:])

        return result
