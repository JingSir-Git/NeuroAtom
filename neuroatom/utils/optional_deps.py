"""Lazy import helpers for optional dependencies.

Provides clear error messages when an optional dependency is missing,
guiding the user to the correct ``pip install neuroatom[extra]`` command.
"""

from typing import Any

_EXTRAS_MAP = {
    "mne": "mne",
    "mne_bids": "bids",
    "torch": "torch",
    "moabb": "moabb",
    "pandas": "all",
    "matplotlib": "all",
    "sklearn": "all",
}


def require(module_name: str, purpose: str = "") -> Any:
    """Import *module_name* or raise a helpful ``ImportError``.

    Args:
        module_name: Top-level module to import (e.g. ``"mne"``).
        purpose: Human-readable reason (e.g. ``"the PhysioNet MI importer"``).

    Returns:
        The imported module.

    Raises:
        ImportError: With a message that includes the correct pip extra.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError:
        extra = _EXTRAS_MAP.get(module_name, module_name)
        msg = (
            f"'{module_name}' is required"
            + (f" for {purpose}" if purpose else "")
            + f" but is not installed.\n"
            f"Install it with:  pip install neuroatom[{extra}]"
        )
        raise ImportError(msg) from None
