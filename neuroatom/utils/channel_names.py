"""Channel name standardization using an alias lookup table.

Different datasets use different names for the same electrode:
  - 'EEG Fp1', 'EEG FP1', 'FP1', 'Fp1' → standard 'Fp1'
  - 'EEG C3', 'C3', 'c3' → standard 'C3'

The alias table is loaded from ``neuroatom/configs/channel_aliases.yaml``
and maps each standard name to a list of known aliases.

All cross-dataset channel matching in queries and assembly uses
the resolved ``standard_name`` — never raw channel names.
"""

import importlib.resources as pkg_resources
import logging
import re
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Module-level cache
_ALIAS_TABLE: Optional[Dict[str, List[str]]] = None
_REVERSE_MAP: Optional[Dict[str, str]] = None


def _load_alias_table() -> Dict[str, List[str]]:
    """Load channel alias table from package config."""
    global _ALIAS_TABLE
    if _ALIAS_TABLE is not None:
        return _ALIAS_TABLE

    config_ref = pkg_resources.files("neuroatom.configs").joinpath("channel_aliases.yaml")
    with pkg_resources.as_file(config_ref) as config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            _ALIAS_TABLE = yaml.safe_load(f) or {}

    logger.debug("Loaded channel alias table with %d standard names.", len(_ALIAS_TABLE))
    return _ALIAS_TABLE


def _build_reverse_map() -> Dict[str, str]:
    """Build reverse lookup: alias (lowercased) → standard_name."""
    global _REVERSE_MAP
    if _REVERSE_MAP is not None:
        return _REVERSE_MAP

    table = _load_alias_table()
    _REVERSE_MAP = {}
    for standard_name, aliases in table.items():
        # The standard name itself is also an alias
        _REVERSE_MAP[standard_name.lower()] = standard_name
        for alias in aliases:
            key = alias.lower()
            if key in _REVERSE_MAP and _REVERSE_MAP[key] != standard_name:
                logger.warning(
                    "Duplicate alias '%s' maps to both '%s' and '%s'. Using first.",
                    alias, _REVERSE_MAP[key], standard_name,
                )
                continue
            _REVERSE_MAP[key] = standard_name

    return _REVERSE_MAP


def standardize_channel_name(raw_name: str) -> Optional[str]:
    """Resolve a raw channel name to its standard name.

    Attempts multiple strategies:
    1. Exact case-insensitive match against alias table
    2. Strip common prefixes ('EEG ', 'EEG-', etc.) and retry
    3. Return None if no match found

    Args:
        raw_name: The raw channel name from the source file.

    Returns:
        Standard channel name (e.g. 'Fp1', 'C3', 'Oz'), or None if unknown.
    """
    reverse = _build_reverse_map()

    # Strategy 1: direct lookup
    key = raw_name.strip().lower()
    if key in reverse:
        return reverse[key]

    # Strategy 2: strip common prefixes
    prefixes_to_strip = [
        r"^eeg[\s\-_]+",  # 'EEG Fp1', 'EEG-Fp1'
        r"^eog[\s\-_]+",
        r"^emg[\s\-_]+",
        r"^ecg[\s\-_]+",
        r"^ref[\s\-_]+",
    ]
    for prefix_pattern in prefixes_to_strip:
        stripped = re.sub(prefix_pattern, "", raw_name.strip(), flags=re.IGNORECASE)
        if stripped.lower() != key:
            stripped_key = stripped.lower()
            if stripped_key in reverse:
                return reverse[stripped_key]

    # Strategy 3: no match
    return None


def standardize_channel_names(raw_names: List[str]) -> Dict[str, Optional[str]]:
    """Batch standardize a list of channel names.

    Returns:
        Dict mapping raw_name → standard_name (or None if unknown).
    """
    return {name: standardize_channel_name(name) for name in raw_names}


def get_standard_channel_list() -> List[str]:
    """Return all known standard channel names."""
    table = _load_alias_table()
    return sorted(table.keys())


def reload_alias_table() -> None:
    """Force reload the alias table (useful for testing)."""
    global _ALIAS_TABLE, _REVERSE_MAP
    _ALIAS_TABLE = None
    _REVERSE_MAP = None
    _load_alias_table()
    _build_reverse_map()
