"""High-level convenience API for NeuroAtom.

Get from raw data to a PyTorch DataLoader in ~5 lines::

    import neuroatom as na

    loader = na.quickload(
        "bci_comp_iv_2a",
        data_path="data/A01T.mat",
        subject="A01",
        batch_size=32,
    )

    for batch in loader:
        signals = batch["signal"]   # (B, C, T)
        labels  = batch["labels"]   # dict of tensors
        break

For full control, use the lower-level :class:`Pool`, :class:`Indexer`,
:class:`DatasetAssembler` APIs directly.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def quickload(
    dataset: str,
    data_path: Union[str, Path],
    *,
    subject: Optional[str] = None,
    pool_dir: Union[str, Path, None] = None,
    batch_size: int = 32,
    band: Optional[Tuple[float, float]] = None,
    target_unit: str = "uV",
    label_field: Optional[str] = None,
    shuffle_train: bool = True,
    split_test_ratio: float = 0.0,
    extra_import_kwargs: Optional[Dict[str, Any]] = None,
    extra_recipe_kwargs: Optional[Dict[str, Any]] = None,
):
    """One-call import → index → assemble → DataLoader.

    Args:
        dataset: Built-in dataset name (e.g. ``"bci_comp_iv_2a"``,
            ``"physionet_mi"``, ``"seed_v"``). Must match a registered
            importer **and** a built-in YAML task config.
        data_path: Path to the data file or directory.
        subject: Subject identifier. If *None*, inferred from filename
            when possible (e.g. ``"A01"`` from ``A01T.mat``).
        pool_dir: Where to store the atom pool. Defaults to a temporary
            directory (cleaned up by the OS eventually).
        batch_size: Batch size for the returned DataLoader.
        band: Optional ``(low_hz, high_hz)`` bandpass filter tuple.
        target_unit: Signal unit after assembly (default ``"uV"``).
        label_field: Annotation field to use as label. If *None*,
            auto-detected from the task config.
        shuffle_train: Whether to shuffle the training DataLoader.
        split_test_ratio: Fraction of data held out for test split.
            ``0.0`` means everything goes into the train set.
        extra_import_kwargs: Forwarded to the importer's
            ``import_subject`` / ``import_run`` call.
        extra_recipe_kwargs: Forwarded to :class:`AssemblyRecipe`.

    Returns:
        A :class:`torch.utils.data.DataLoader` (if ``split_test_ratio == 0``)
        or a ``(train_loader, test_loader)`` tuple.
    """
    data_path = Path(data_path)
    extra_import_kwargs = extra_import_kwargs or {}
    extra_recipe_kwargs = extra_recipe_kwargs or {}

    # ── 1. Resolve pool ──────────────────────────────────────────────────
    if pool_dir is None:
        pool_dir = Path(tempfile.mkdtemp(prefix="neuroatom_quick_"))
    else:
        pool_dir = Path(pool_dir)

    from neuroatom.storage.pool import Pool
    if (pool_dir / "pool.json").exists():
        pool = Pool.open(pool_dir)
    else:
        pool = Pool.create(pool_dir)

    # ── 2. Load task config ──────────────────────────────────────────────
    from neuroatom.importers.base import TaskConfig

    config = TaskConfig.builtin(_resolve_config_name(dataset))

    # ── 3. Instantiate importer and import data ──────────────────────────
    from neuroatom.importers.registry import get_importer

    importer = get_importer(dataset, pool, config)
    subject = subject or _infer_subject(data_path, dataset)

    _do_import(importer, dataset, data_path, subject, extra_import_kwargs)

    # ── 4. Index ─────────────────────────────────────────────────────────
    from neuroatom.index.indexer import Indexer
    indexer = Indexer(pool)
    n = indexer.reindex_all()
    if n == 0:
        raise RuntimeError(
            f"Import produced 0 atoms. Check data_path={data_path} and subject={subject}."
        )
    logger.info("Indexed %d atoms.", n)

    # ── 5. Assemble ──────────────────────────────────────────────────────
    label_field = label_field or _infer_label_field(config, dataset)

    from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
    from neuroatom.core.enums import (
        NormalizationMethod, NormalizationScope, SplitStrategy,
    )

    recipe_kw: Dict[str, Any] = {
        "recipe_id": f"quick_{dataset}",
        "query": {"dataset_id": config.dataset_id},
        "target_unit": target_unit,
        "normalization_method": NormalizationMethod.ZSCORE,
        "normalization_scope": NormalizationScope.PER_ATOM,
    }
    if band is not None:
        recipe_kw["filter_band"] = band
    if label_field:
        recipe_kw["query"]["annotations"] = [{"name": label_field}]
        recipe_kw["label_fields"] = [
            LabelSpec(annotation_name=label_field, output_key=label_field),
        ]
    else:
        logger.warning(
            "Could not infer label field for '%s'. "
            "Using atom_type as a placeholder label.",
            dataset,
        )
        recipe_kw["label_fields"] = [
            LabelSpec(annotation_name="atom_type", output_key="atom_type"),
        ]

    if split_test_ratio > 0:
        recipe_kw["split_strategy"] = SplitStrategy.STRATIFIED
        recipe_kw["split_config"] = {
            "val_ratio": 0.0,
            "test_ratio": split_test_ratio,
            "seed": 42,
        }

    recipe_kw.update(extra_recipe_kwargs)
    recipe = AssemblyRecipe(**recipe_kw)

    from neuroatom.assembler.dataset_assembler import DatasetAssembler
    result = DatasetAssembler(pool, indexer).assemble(recipe)
    indexer.close()

    # ── 6. DataLoader ────────────────────────────────────────────────────
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader
        from neuroatom.loader.torch_dataset import AtomDataset
    except ImportError:
        raise ImportError(
            "PyTorch is required for quickload. Install with: pip install neuroatom[torch]"
        ) from None

    all_samples = result.train_samples + result.val_samples + result.test_samples

    if split_test_ratio > 0 and result.test_samples:
        train_loader = DataLoader(
            AtomDataset(result.train_samples),
            batch_size=batch_size,
            shuffle=shuffle_train,
        )
        test_loader = DataLoader(
            AtomDataset(result.test_samples),
            batch_size=batch_size,
            shuffle=False,
        )
        logger.info(
            "quickload: %d train + %d test samples → DataLoaders ready.",
            len(result.train_samples), len(result.test_samples),
        )
        return train_loader, test_loader
    else:
        loader = DataLoader(
            AtomDataset(all_samples),
            batch_size=batch_size,
            shuffle=shuffle_train,
        )
        logger.info("quickload: %d samples → DataLoader ready.", len(all_samples))
        return loader


# ══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════

# Map dataset names to their built-in config names
# (most are identical; exceptions listed here)
_CONFIG_ALIASES: Dict[str, str] = {
    "zuco2": "zuco2_tsr",
    "zuco2_tsr": "zuco2_tsr",
    "kul_aad": "kul_aad",
    "dtu_aad": "dtu_aad",
    "aad_mat": "kul_aad",
    "chinese_eeg2": "chinese_eeg2_listening",
}


def _resolve_config_name(dataset: str) -> str:
    return _CONFIG_ALIASES.get(dataset, dataset)


def _infer_subject(data_path: Path, dataset: str) -> str:
    """Best-effort subject ID inference from the file path."""
    stem = data_path.stem
    if dataset == "bci_comp_iv_2a":
        # A01T.mat → A01
        return stem[:3] if len(stem) >= 3 else stem
    if dataset in ("kul_aad", "dtu_aad", "aad_mat"):
        # S1.mat → S1
        return stem
    if dataset == "physionet_mi":
        # S001 directory → S001
        return data_path.name
    # Fallback: use the filename stem
    return stem


_KNOWN_LABEL_FIELDS: Dict[str, str] = {
    "bci_comp_iv_2a": "mi_class",
    "physionet_mi": "mi_class",
    "seed_v": "emotion",
    "zuco2_tsr": "sentence_id",
    "zuco2": "sentence_id",
    "ccep_bids_npy": "stim_pair",
    "chinese_eeg2": "sentence_index",
    "chinese_eeg2_listening": "sentence_index",
    "chinese_eeg2_reading": "sentence_index",
    "kul_aad": "attended_ear",
    "dtu_aad": "attended_speaker",
    "aad_mat": "attended_ear",
    "openbmi_mi": "mi_class",
    "openbmi_erp": "erp_class",
    "openbmi_ssvep": "ssvep_class",
    "lee2019_mi": "mi_class",
    "p300_speller": "target",
    "ssvep_benchmark": "frequency",
    "inner_speech": "inner_speech_class",
}


def _infer_label_field(config, dataset: str = "") -> Optional[str]:
    """Infer the primary label annotation name from the dataset or config."""
    # 1. Check known lookup table
    if dataset in _KNOWN_LABEL_FIELDS:
        return _KNOWN_LABEL_FIELDS[dataset]
    # 2. Explicit field in config
    mapping = config.data
    for key in ("label_field", "annotation_name"):
        if key in mapping:
            return mapping[key]
    em = config.event_mapping
    if isinstance(em, dict):
        if "label_field" in em:
            return em["label_field"]
    return None


def _do_import(importer, dataset: str, data_path: Path, subject: str, kwargs: dict):
    """Dispatch to the correct importer's entry method."""

    # ── BCI Competition IV 2a ────────────────────────────────────────────
    if dataset == "bci_comp_iv_2a":
        importer.import_subject(
            mat_path=data_path,
            subject_id=subject,
            **kwargs,
        )
        return

    # ── PhysioNet MI ─────────────────────────────────────────────────────
    if dataset == "physionet_mi":
        importer.import_subject(
            subject_dir=data_path,
            subject_id=subject,
            **kwargs,
        )
        return

    # ── SEED-V ───────────────────────────────────────────────────────────
    if dataset == "seed_v":
        importer.import_subject(
            subject_dir=data_path,
            subject_id=subject,
            **kwargs,
        )
        return

    # ── Zuco 2.0 ─────────────────────────────────────────────────────────
    if dataset in ("zuco2", "zuco2_tsr"):
        importer.import_subject(
            subject_dir=data_path,
            subject_id=subject,
            **kwargs,
        )
        return

    # ── CCEP-COREG ───────────────────────────────────────────────────────
    if dataset == "ccep_bids_npy":
        importer.import_subject(
            subject_dir=data_path,
            subject_id=subject,
            **kwargs,
        )
        return

    # ── ChineseEEG-2 ────────────────────────────────────────────────────
    if dataset in ("chinese_eeg2", "chinese_eeg2_listening", "chinese_eeg2_reading"):
        importer.import_dataset(
            bids_root=data_path,
            subjects=[subject] if subject != data_path.stem else None,
            **kwargs,
        )
        return

    # ── OpenBMI (MI / ERP / SSVEP) ─────────────────────────────────────
    if dataset in ("openbmi_mi", "openbmi_erp", "openbmi_ssvep"):
        importer.import_subject(
            mat_path=data_path,
            subject_id=subject,
            **kwargs,
        )
        return

    # ── AAD (KUL / DTU) ─────────────────────────────────────────────────
    if dataset in ("aad_mat", "kul_aad", "dtu_aad"):
        importer.import_subject(
            mat_path=data_path,
            subject_id=subject,
            **kwargs,
        )
        return

    # ── Fallback: try common signatures ──────────────────────────────────
    if hasattr(importer, "import_subject"):
        importer.import_subject(data_path, subject_id=subject, **kwargs)
    elif hasattr(importer, "import_dataset"):
        importer.import_dataset(data_path, **kwargs)
    else:
        raise NotImplementedError(
            f"Don't know how to call importer for dataset '{dataset}'. "
            "Pass data manually or use the low-level API."
        )
