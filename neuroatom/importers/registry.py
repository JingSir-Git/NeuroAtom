"""Importer registry: auto-detect format and dispatch to the correct importer."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from neuroatom.importers.base import BaseImporter, TaskConfig
from neuroatom.storage.pool import Pool

logger = logging.getLogger(__name__)

# Global registry of importers (format_name → importer_class)
_REGISTRY: Dict[str, Type[BaseImporter]] = {}


def register_importer(format_name: str, importer_class: Type[BaseImporter]) -> None:
    """Register an importer class for a given format name."""
    _REGISTRY[format_name] = importer_class
    logger.debug("Registered importer '%s': %s", format_name, importer_class.__name__)


def get_importer(
    format_name: str,
    pool: Pool,
    task_config: TaskConfig,
) -> BaseImporter:
    """Get an importer instance by format name."""
    if format_name not in _REGISTRY:
        raise ValueError(
            f"Unknown format '{format_name}'. "
            f"Registered formats: {list(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[format_name]
    return cls(pool=pool, task_config=task_config)


def detect_format(path: Path) -> Optional[str]:
    """Auto-detect the format of a data file/directory.

    Iterates through registered importers and returns the first match.
    """
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
    return sorted(_REGISTRY.keys())
