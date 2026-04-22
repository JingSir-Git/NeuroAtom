"""Importer registry: auto-detect format and dispatch to the correct importer."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from neuroatom.importers.base import BaseImporter, TaskConfig
from neuroatom.storage.pool import Pool

logger = logging.getLogger(__name__)

# Global registry of importers (format_name → importer_class)
_REGISTRY: Dict[str, Type[BaseImporter]] = {}
_ALL_LOADED = False


def register_importer(format_name: str, importer_class: Type[BaseImporter]) -> None:
    """Register an importer class for a given format name."""
    _REGISTRY[format_name] = importer_class
    logger.debug("Registered importer '%s': %s", format_name, importer_class.__name__)


def _ensure_all_registered() -> None:
    """Lazily import every importer module so their ``register_importer``
    calls run.  Safe to call repeatedly (no-ops after the first time)."""
    global _ALL_LOADED
    if _ALL_LOADED:
        return
    # Each module has a top-level ``register_importer(...)`` call.
    import importlib
    _modules = [
        "neuroatom.importers.bci_comp_iv_2a",
        "neuroatom.importers.physionet_mi",
        "neuroatom.importers.seed_v",
        "neuroatom.importers.zuco2",
        "neuroatom.importers.ccep_bids_npy",
        "neuroatom.importers.chinese_eeg2",
        "neuroatom.importers.aad_mat",
        "neuroatom.importers.bids",
        "neuroatom.importers.eeglab",
        "neuroatom.importers.mat",
        "neuroatom.importers.mne_generic",
        "neuroatom.importers.moabb_bridge",
    ]
    for mod in _modules:
        try:
            importlib.import_module(mod)
        except Exception:
            logger.debug("Could not load importer module %s", mod, exc_info=True)
    _ALL_LOADED = True


def get_importer_class(format_name: str) -> Type[BaseImporter]:
    """Return the importer **class** by format name (no instantiation).

    Ensures all built-in importers are loaded first.
    """
    _ensure_all_registered()
    if format_name not in _REGISTRY:
        raise ValueError(
            f"Unknown format '{format_name}'. "
            f"Registered formats: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[format_name]


def get_importer(
    format_name: str,
    pool: Pool,
    task_config: TaskConfig,
) -> BaseImporter:
    """Get an importer instance by format name."""
    cls = get_importer_class(format_name)
    return cls(pool=pool, task_config=task_config)


def detect_format(path: Path) -> Optional[str]:
    """Auto-detect the format of a data file/directory.

    Iterates through registered importers and returns the first match.
    """
    _ensure_all_registered()
    for format_name, importer_class in _REGISTRY.items():
        try:
            if importer_class.detect(path):
                logger.info("Detected format '%s' for %s", format_name, path)
                return format_name
        except Exception as e:
            logger.debug(
                "Error during detection for '%s' on %s: %s",
                format_name, path, e,
            )
    return None


def list_formats() -> List[str]:
    """Return all registered format names."""
    _ensure_all_registered()
    return sorted(_REGISTRY.keys())
