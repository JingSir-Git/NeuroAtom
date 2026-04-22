"""Custom collation functions for NeuroAtom DataLoaders.

Handles:
- Variable-length signals via dynamic padding + mask generation
- None-sample filtering (from skip error handling)
- Multi-label dict merging into stacked tensors
- Channel mask stacking
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for collation functions. "
            "Install it with: pip install neuroatom[torch]"
        )


def skip_none_collate(batch: List) -> Optional[Dict]:
    """Filter out None samples, then collate remaining.

    Use this when error_handling='skip' in HDF5AtomDataset.
    Returns None if all samples are None.
    """
    _check_torch()
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def neuroatom_collate(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Full-featured collation for NeuroAtom samples.

    Handles:
    - Filtering None samples
    - Stacking signals into (B, C, T) tensor
    - Stacking channel_mask into (B, C) tensor
    - Stacking time_mask into (B, T) tensor
    - Merging label dicts into stacked tensors per key
    - Preserving string metadata (atom_id, subject_id, dataset_id)
    """
    _check_torch()

    # Filter None
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    result = {}

    # Stack signals: all should be (C, T) with same shape after pad/crop
    signals = [b["signal"] for b in batch]
    if isinstance(signals[0], torch.Tensor):
        result["signal"] = torch.stack(signals, dim=0)
    elif isinstance(signals[0], np.ndarray):
        result["signal"] = torch.from_numpy(np.stack(signals, axis=0)).float()
    else:
        result["signal"] = signals

    # Channel masks: (B, C)
    if batch[0].get("channel_mask") is not None:
        masks = [b["channel_mask"] for b in batch]
        if isinstance(masks[0], torch.Tensor):
            result["channel_mask"] = torch.stack(masks, dim=0)
        elif isinstance(masks[0], np.ndarray):
            result["channel_mask"] = torch.from_numpy(
                np.stack(masks, axis=0)
            ).float()

    # Time masks: (B, T)
    if batch[0].get("time_mask") is not None:
        masks = [b["time_mask"] for b in batch]
        if isinstance(masks[0], torch.Tensor):
            result["time_mask"] = torch.stack(masks, dim=0)
        elif isinstance(masks[0], np.ndarray):
            result["time_mask"] = torch.from_numpy(
                np.stack(masks, axis=0)
            ).float()

    # Labels: merge dicts
    if "labels" in batch[0] and isinstance(batch[0]["labels"], dict):
        label_keys = batch[0]["labels"].keys()
        for key in label_keys:
            values = [b["labels"][key] for b in batch]
            if isinstance(values[0], (int, float, np.integer, np.floating)):
                result[f"label_{key}"] = torch.tensor(values).long()
            elif isinstance(values[0], np.ndarray):
                result[f"label_{key}"] = torch.from_numpy(
                    np.stack(values, axis=0)
                ).float()
            elif isinstance(values[0], torch.Tensor):
                result[f"label_{key}"] = torch.stack(values, dim=0)
            else:
                result[f"label_{key}"] = values

    # String metadata (not tensorized)
    for key in ["atom_id", "subject_id", "dataset_id"]:
        if key in batch[0]:
            result[key] = [b[key] for b in batch]

    return result


def dynamic_pad_collate(
    batch: List[Dict[str, Any]],
    pad_value: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Collate variable-length signals by padding to max length in batch.

    Generates time_mask dynamically based on actual signal lengths.
    Use this when target_duration is not set in the recipe.

    Args:
        batch: List of sample dicts with 'signal' key.
        pad_value: Value for padding shorter signals.

    Returns:
        Collated batch dict with padded signals and generated time_masks.
    """
    _check_torch()

    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    signals = [b["signal"] for b in batch]

    # Convert to numpy if tensor
    if isinstance(signals[0], torch.Tensor):
        signals_np = [s.numpy() for s in signals]
    else:
        signals_np = signals

    # Find max time dimension
    n_channels = signals_np[0].shape[0]
    max_time = max(s.shape[1] for s in signals_np)

    # Pad all signals and generate masks
    padded_signals = np.full(
        (len(batch), n_channels, max_time), pad_value, dtype=np.float32
    )
    time_masks = np.zeros((len(batch), max_time), dtype=np.float32)

    for i, sig in enumerate(signals_np):
        actual_len = sig.shape[1]
        padded_signals[i, :, :actual_len] = sig
        time_masks[i, :actual_len] = 1.0

    result = {
        "signal": torch.from_numpy(padded_signals).float(),
        "time_mask": torch.from_numpy(time_masks).float(),
    }

    # Channel masks
    if batch[0].get("channel_mask") is not None:
        ch_masks = [b["channel_mask"] for b in batch]
        if isinstance(ch_masks[0], torch.Tensor):
            result["channel_mask"] = torch.stack(ch_masks, dim=0)
        elif isinstance(ch_masks[0], np.ndarray):
            result["channel_mask"] = torch.from_numpy(
                np.stack(ch_masks, axis=0)
            ).float()

    # Labels
    if "labels" in batch[0] and isinstance(batch[0]["labels"], dict):
        for key in batch[0]["labels"]:
            values = [b["labels"][key] for b in batch]
            if isinstance(values[0], (int, float, np.integer, np.floating)):
                result[f"label_{key}"] = torch.tensor(values).long()
            elif isinstance(values[0], np.ndarray):
                result[f"label_{key}"] = torch.from_numpy(
                    np.stack(values, axis=0)
                ).float()

    # Metadata
    for key in ["atom_id", "subject_id", "dataset_id"]:
        if key in batch[0]:
            result[key] = [b[key] for b in batch]

    return result
