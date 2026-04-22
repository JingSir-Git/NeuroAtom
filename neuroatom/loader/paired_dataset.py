"""PairedAtomDataset: PyTorch Dataset for multi-modal paired samples.

Wraps the output of ``MultiModalAssembler`` into a PyTorch-compatible
``Dataset`` that yields paired tensors for each modality::

    dataset = PairedAtomDataset(paired_samples, modality_keys=["eeg", "ieeg"])
    sample = dataset[0]
    # sample = {"eeg": Tensor(C1, T1), "ieeg": Tensor(C2, T2), "labels": {...}}

**No temporal alignment is performed.** Each modality tensor retains its
native shape (n_channels, n_samples) after per-modality pipeline processing.
When modalities have different time lengths (e.g., EEG 1000 samples,
sEEG 2000 samples), the tensors are returned as-is in a dict. This is by
design — different downstream models require different alignment strategies:

- CNNs may need zero-pad to a common length
- Transformers can handle variable-length with attention masks
- Cross-modal fusion models may use separate encoders per modality

Users who need alignment should implement it as a ``transform`` callable
passed to the dataset constructor, or handle it in their model's forward
method or a custom ``collate_fn``.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for PairedAtomDataset. "
            "Install it with: pip install neuroatom[torch]"
        )


class PairedAtomDataset(Dataset if HAS_TORCH else object):
    """PyTorch Dataset for multi-modal paired samples.

    Each sample is a dict with:
        - One key per modality containing a float32 tensor ``(C, T)``
          where C and T may differ between modalities (no forced alignment).
        - ``'labels'``: dict of label_key → tensor
        - ``'atom_id'``: str (from primary modality)
        - ``'subject_id'``: str
        - ``'dataset_id'``: str
        - ``'pairing_key'``: str (pipe-delimited pairing key)

    Note:
        Modality tensors may have **different temporal lengths**. The default
        PyTorch collate_fn will fail if tensors have mismatched shapes within
        a batch. Use a custom ``collate_fn`` to handle this, for example by
        padding to the max length in the batch or by using nested tensors.

    Args:
        paired_samples: List of dicts from ``MultiModalAssemblyResult``.
        modality_keys: Modality names to include in each sample
            (e.g., ``["eeg", "ieeg"]``).
        transforms: Optional list of per-sample transform callables.
            Each receives and returns the full sample dict.
    """

    def __init__(
        self,
        paired_samples: List[Dict[str, Any]],
        modality_keys: List[str],
        transforms: Optional[List[Callable]] = None,
    ):
        _check_torch()
        self._samples = paired_samples
        self._modality_keys = modality_keys
        self._transforms = transforms or []

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        sample = self._samples[idx]

        result = {}

        # Convert each modality signal to tensor
        for mod_key in self._modality_keys:
            signal = sample.get(mod_key)
            if signal is not None:
                if isinstance(signal, np.ndarray):
                    result[mod_key] = torch.from_numpy(signal).float()
                else:
                    result[mod_key] = signal
            else:
                result[mod_key] = None

        # Convert labels
        labels_dict = {}
        raw_labels = sample.get("labels", {})
        for key, value in raw_labels.items():
            if isinstance(value, np.ndarray):
                labels_dict[key] = torch.from_numpy(value).float()
            elif isinstance(value, (int, float)):
                labels_dict[key] = torch.tensor(value).long()
            else:
                labels_dict[key] = value
        result["labels"] = labels_dict

        # Metadata
        result["atom_id"] = sample.get("atom_id", "")
        result["subject_id"] = sample.get("subject_id", "")
        result["dataset_id"] = sample.get("dataset_id", "")
        result["pairing_key"] = sample.get("pairing_key", "")

        # Apply transforms
        for transform in self._transforms:
            result = transform(result)

        return result
