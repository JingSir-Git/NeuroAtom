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
    pool = Pool.open_or_create(pool_dir)

    # ── 2. Load task config ──────────────────────────────────────────────
    from neuroatom.importers.base import TaskConfig

    config = TaskConfig.builtin(_resolve_config_name(dataset))

    # ── 3. Instantiate importer and import data ──────────────────────────
    from neuroatom.importers.registry import get_importer

    importer = get_importer(dataset, pool, config)
    subject = subject or _infer_subject(data_path, dataset, config=config)

    _do_import(importer, dataset, data_path, subject, extra_import_kwargs, config=config)

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
    label_field = label_field or _infer_label_field(config, dataset=dataset)

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
#
# Source-of-truth precedence for per-dataset metadata:
#   1. TaskConfig.quickload_meta from the YAML config (preferred)
#   2. Fallback hardcoded defaults below (legacy compatibility)
#
# To add a new dataset: drop a YAML with a ``quickload:`` section into
# ``importers/task_configs/``. No edits to this file required.
# ══════════════════════════════════════════════════════════════════════════

# Fallback aliases for datasets that don't yet ship a YAML ``quickload.aliases``.
_FALLBACK_CONFIG_ALIASES: Dict[str, str] = {
    "zuco2": "zuco2_tsr",
    "zuco2_tsr": "zuco2_tsr",
    "kul_aad": "kul_aad",
    "dtu_aad": "dtu_aad",
    "aad_mat": "kul_aad",
    "chinese_eeg2": "chinese_eeg2_listening",
}

# Fallback label fields. Datasets with a ``quickload.label_field`` in their
# YAML override this.
_FALLBACK_LABEL_FIELDS: Dict[str, str] = {
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

# Fallback (data_path_kwarg, entry_method) per dataset, for YAMLs that
# don't yet declare ``quickload``.
_FALLBACK_IMPORT_SIGNATURES: Dict[str, Tuple[str, str]] = {
    "bci_comp_iv_2a": ("mat_path", "import_subject"),
    "physionet_mi": ("subject_dir", "import_subject"),
    "seed_v": ("subject_dir", "import_subject"),
    "zuco2": ("subject_dir", "import_subject"),
    "zuco2_tsr": ("subject_dir", "import_subject"),
    "ccep_bids_npy": ("subject_dir", "import_subject"),
    "openbmi_mi": ("mat_path", "import_subject"),
    "openbmi_erp": ("mat_path", "import_subject"),
    "openbmi_ssvep": ("mat_path", "import_subject"),
    "aad_mat": ("mat_path", "import_subject"),
    "kul_aad": ("mat_path", "import_subject"),
    "dtu_aad": ("mat_path", "import_subject"),
    # ChineseEEG-2 takes a BIDS root and a subject list; handled specially.
    "chinese_eeg2": ("bids_root", "import_dataset"),
    "chinese_eeg2_listening": ("bids_root", "import_dataset"),
    "chinese_eeg2_reading": ("bids_root", "import_dataset"),
}


def _resolve_config_name(dataset: str) -> str:
    """Resolve a user-facing dataset name to the actual config file basename.

    Reads ``quickload.aliases`` from each built-in YAML config to build a
    forward map; falls back to ``_FALLBACK_CONFIG_ALIASES`` for unconfigured
    datasets.
    """
    # Try built-in YAMLs' aliases first.
    try:
        import importlib.resources as _res

        pkg = "neuroatom.importers.task_configs"
        for ref in _res.files(pkg).iterdir():
            if not ref.name.endswith(".yaml") or ref.name.startswith("_"):
                continue
            config_name = ref.name.removesuffix(".yaml")
            if config_name == dataset:
                return config_name  # exact match
            try:
                from neuroatom.importers.base import TaskConfig
                cfg = TaskConfig.builtin(config_name)
                if dataset in cfg.quickload_aliases:
                    return config_name
            except Exception:
                continue
    except Exception:
        pass

    return _FALLBACK_CONFIG_ALIASES.get(dataset, dataset)


def _infer_subject(data_path: Path, dataset: str, config=None) -> str:
    """Best-effort subject ID inference from the file path.

    Precedence:
        1. ``quickload.subject_pattern`` regex from TaskConfig (if matches).
        2. Per-dataset hardcoded heuristic below.
    """
    stem = data_path.stem

    # YAML-declared regex wins.
    if config is not None:
        pat = config.quickload_subject_pattern
        if pat:
            import re
            m = re.search(pat, stem)
            if m:
                return m.group(1) if m.groups() else m.group(0)

    # Fallback heuristics.
    if dataset == "bci_comp_iv_2a":
        # A01T.mat → A01
        return stem[:3] if len(stem) >= 3 else stem
    if dataset in ("kul_aad", "dtu_aad", "aad_mat"):
        # S1.mat → S1
        return stem
    if dataset == "physionet_mi":
        # S001 directory → S001
        return data_path.name
    return stem


def _infer_label_field(config, dataset: str = "") -> Optional[str]:
    """Infer the primary label annotation name.

    Precedence:
        1. ``quickload.label_field`` in TaskConfig (YAML-declared).
        2. Top-level ``label_field`` / ``annotation_name`` in config data.
        3. Fallback hardcoded ``_FALLBACK_LABEL_FIELDS`` lookup.
        4. ``event_mapping.label_field`` if dict-shaped.
    """
    # 1. quickload.label_field (preferred)
    if config is not None and config.label_field:
        return config.label_field

    # 2. top-level keys
    if config is not None:
        mapping = config.data
        for key in ("label_field", "annotation_name"):
            if key in mapping:
                return mapping[key]

    # 3. legacy hardcoded table
    if dataset in _FALLBACK_LABEL_FIELDS:
        return _FALLBACK_LABEL_FIELDS[dataset]

    # 4. nested event_mapping fallback
    if config is not None:
        em = config.event_mapping
        if isinstance(em, dict) and "label_field" in em:
            return em["label_field"]
    return None


def _do_import(importer, dataset: str, data_path: Path, subject: str, kwargs: dict, config=None):
    """Dispatch to the correct importer's entry method.

    Reads ``quickload.data_path_kwarg`` and ``quickload.entry_method`` from
    TaskConfig (YAML-declared) when available; falls back to the legacy
    hardcoded signatures for datasets without ``quickload`` metadata.
    """
    # ── 1. YAML-declared signature (preferred) ────────────────────────────
    data_kwarg = config.quickload_data_path_kwarg if config else None
    entry_method = config.quickload_entry_method if config else "import_subject"

    # ── 2. Fallback to hardcoded signatures ───────────────────────────────
    if data_kwarg is None:
        sig = _FALLBACK_IMPORT_SIGNATURES.get(dataset)
        if sig:
            data_kwarg, entry_method = sig

    # ── 3. ChineseEEG-2 special case (multi-subject BIDS) ─────────────────
    if dataset.startswith("chinese_eeg2") and entry_method == "import_dataset":
        importer.import_dataset(
            bids_root=data_path,
            subjects=[subject] if subject != data_path.stem else None,
            **kwargs,
        )
        return

    # ── 4. Standard single-subject path ───────────────────────────────────
    if data_kwarg:
        method = getattr(importer, entry_method, None)
        if method is None:
            raise NotImplementedError(
                f"Importer {type(importer).__name__} has no method '{entry_method}'."
            )
        method(**{data_kwarg: data_path}, subject_id=subject, **kwargs)
        return

    # ── 5. Last-resort heuristics ─────────────────────────────────────────
    if hasattr(importer, "import_subject"):
        importer.import_subject(data_path, subject_id=subject, **kwargs)
    elif hasattr(importer, "import_dataset"):
        importer.import_dataset(data_path, **kwargs)
    else:
        raise NotImplementedError(
            f"Don't know how to call importer for dataset '{dataset}'. "
            "Add a `quickload:` section to its YAML config or use the low-level API."
        )
