"""High-level convenience API for NeuroAtom.

Single subject (5 lines)::

    import neuroatom as na
    loader = na.quickload("bci_comp_iv_2a", "data/A01T.mat")

Cross-subject / cross-dataset / cross-task (~10 lines)::

    loader = na.multiload(
        sources=[
            {"dataset": "openbmi_mi",     "path": "OpenBMI/MI",  "subjects": ["S01", "S02"]},
            {"dataset": "bci_comp_iv_2a", "path": "data/A01T.mat"},
            {"dataset": "physionet_mi",   "path": "Physionet/S001"},
        ],
        target_channels=["C3", "Cz", "C4", ...],  # unify channel layout
        target_srate=250,
        label_field="mi_class",
    )

For full control, use the lower-level :class:`Pool`, :class:`Indexer`,
:class:`DatasetAssembler`, or :class:`FederatedAssembler` APIs directly.
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
# multiload: cross-subject / cross-dataset / cross-task convenience API
# ══════════════════════════════════════════════════════════════════════════

def multiload(
    sources: List[Dict[str, Any]],
    *,
    pool_dir: Union[str, Path, None] = None,
    target_channels: Optional[List[str]] = None,
    target_srate: Optional[float] = None,
    target_duration: Optional[float] = None,
    band: Optional[Tuple[float, float]] = None,
    target_unit: str = "uV",
    label_field: Optional[str] = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    split_strategy: str = "subject",
    split_config: Optional[Dict[str, Any]] = None,
    extra_recipe_kwargs: Optional[Dict[str, Any]] = None,
    normalization: str = "zscore",
    normalization_scope: str = "per_atom",
):
    """Import multiple data sources → unified assembly → DataLoader(s).

    Designed for the core NeuroAtom use case: **cross-subject, cross-dataset,
    and cross-task** training on heterogeneous EEG data.

    Each source is a dict with:
        - ``dataset`` (str): registered dataset name (e.g. ``"openbmi_mi"``).
        - ``path`` (str|Path): file or directory containing data.
        - ``subjects`` (list[str], optional): specific subjects to import.
          If omitted, auto-inferred from path (single file) or import all
          (directory with ``import_paradigm`` / ``import_dataset``).
        - ``import_kwargs`` (dict, optional): extra kwargs for the importer.

    The function creates a **single shared Pool**, imports all sources,
    builds a unified SQLite index, assembles with ChannelMapper +
    Resampler + Filter + Normalizer, and returns PyTorch DataLoaders.

    Args:
        sources: List of source dicts (see above).
        pool_dir: Pool directory. Defaults to a temp dir.
        target_channels: Common channel layout (e.g. 22 standard MI channels).
            If None, keeps all channels from each atom (heterogeneous output).
        target_srate: Target sampling rate in Hz.
        target_duration: Target epoch duration in seconds (pad/crop).
        band: Bandpass filter ``(low_hz, high_hz)``.
        target_unit: Signal unit after assembly ("uV", "mV", "V").
        label_field: Primary annotation name for labels. Auto-inferred if
            all sources share the same paradigm.
        batch_size: Batch size for DataLoaders.
        shuffle_train: Shuffle training DataLoader.
        split_strategy: ``"subject"`` (default), ``"stratified"``, ``"dataset"``.
        split_config: Strategy-specific params (e.g. ``{"test_ratio": 0.2}``).
        extra_recipe_kwargs: Additional ``AssemblyRecipe`` fields.
        normalization: ``"zscore"`` (default), ``"robust"``, ``"minmax"``, or ``None``.
        normalization_scope: ``"per_atom"`` (default), ``"global"``, ``"per_subject"``.

    Returns:
        ``(train_loader, val_loader, test_loader)`` — any may be ``None`` if the
        split produces no samples for that partition.

    Example — cross-dataset MI pre-training::

        import neuroatom as na

        train, val, test = na.multiload(
            sources=[
                {"dataset": "openbmi_mi", "path": r"\\\\server\\OpenBMI\\MI",
                 "subjects": ["S01", "S02", "S03"]},
                {"dataset": "bci_comp_iv_2a", "path": "data/A01T.mat"},
                {"dataset": "bci_comp_iv_2a", "path": "data/A02T.mat"},
            ],
            target_channels=["C3", "C1", "Cz", "C2", "C4"],  # subset
            target_srate=250,
            target_duration=4.0,
            label_field="mi_class",
            split_strategy="subject",
        )

        for batch in train:
            x = batch["signal"]   # (B, 5, 1000) — unified!
            y = batch["labels"]   # {"mi_class": tensor}
            break
    """
    from neuroatom.storage.pool import Pool
    from neuroatom.importers.base import TaskConfig
    from neuroatom.importers.registry import get_importer
    from neuroatom.index.indexer import Indexer
    from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
    from neuroatom.core.enums import (
        NormalizationMethod, NormalizationScope, SplitStrategy,
    )
    from neuroatom.assembler.dataset_assembler import DatasetAssembler

    if not sources:
        raise ValueError("multiload() requires at least one source.")

    extra_recipe_kwargs = extra_recipe_kwargs or {}
    split_config = split_config or {}

    # ── 1. Create or open pool ────────────────────────────────────────
    if pool_dir is None:
        pool_dir = Path(tempfile.mkdtemp(prefix="neuroatom_multi_"))
    else:
        pool_dir = Path(pool_dir)

    if (pool_dir / "pool.json").exists():
        pool = Pool.open(pool_dir)
    else:
        pool = Pool.create(pool_dir)

    logger.info(
        "multiload: %d sources → pool at %s", len(sources), pool_dir
    )

    # ── 2. Import each source ─────────────────────────────────────────
    datasets_imported = set()

    for src_idx, src in enumerate(sources):
        ds_name = src["dataset"]
        data_path = Path(src["path"])
        subjects = src.get("subjects")
        import_kwargs = src.get("import_kwargs", {})

        config_name = _resolve_config_name(ds_name)
        config = TaskConfig.builtin(config_name)
        importer = get_importer(ds_name, pool, config)

        if subjects:
            # Import specific subjects
            for subj in subjects:
                logger.info(
                    "  [%d/%d] %s / %s from %s",
                    src_idx + 1, len(sources), ds_name, subj, data_path,
                )
                _do_import(importer, ds_name, data_path, subj, import_kwargs)
        else:
            # Try import_paradigm (OpenBMI), import_dataset (ChineseEEG2),
            # or single-file import
            if hasattr(importer, "import_paradigm") and data_path.is_dir():
                # OpenBMI-style: extract paradigm from dataset name
                paradigm_map = {
                    "openbmi_mi": "MI",
                    "openbmi_erp": "ERP",
                    "openbmi_ssvep": "SSVEP",
                }
                paradigm = paradigm_map.get(ds_name)
                if paradigm:
                    logger.info(
                        "  [%d/%d] %s (all subjects) from %s",
                        src_idx + 1, len(sources), ds_name, data_path,
                    )
                    importer.import_paradigm(
                        data_dir=data_path,
                        paradigm=paradigm,
                        **import_kwargs,
                    )
                else:
                    # Generic directory: infer subject
                    subj = _infer_subject(data_path, ds_name)
                    _do_import(importer, ds_name, data_path, subj, import_kwargs)
            elif hasattr(importer, "import_dataset") and data_path.is_dir():
                logger.info(
                    "  [%d/%d] %s (full dataset) from %s",
                    src_idx + 1, len(sources), ds_name, data_path,
                )
                importer.import_dataset(bids_root=data_path, **import_kwargs)
            else:
                subj = _infer_subject(data_path, ds_name)
                logger.info(
                    "  [%d/%d] %s / %s from %s",
                    src_idx + 1, len(sources), ds_name, subj, data_path,
                )
                _do_import(importer, ds_name, data_path, subj, import_kwargs)

        datasets_imported.add(ds_name)

    # ── 3. Index ──────────────────────────────────────────────────────
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    logger.info("multiload: indexed %d atoms.", n_indexed)

    if n_indexed == 0:
        raise RuntimeError(
            f"Import produced 0 atoms across {len(sources)} sources. "
            "Check data paths and subject identifiers."
        )

    # ── 4. Resolve label field ────────────────────────────────────────
    if label_field is None:
        # Try to auto-infer from the first dataset
        for ds_name in datasets_imported:
            lf = _KNOWN_LABEL_FIELDS.get(ds_name)
            if lf:
                label_field = lf
                break

    if label_field is None:
        logger.warning(
            "Could not auto-detect label_field for datasets %s. "
            "Using 'atom_type' as placeholder.",
            datasets_imported,
        )
        label_field = "atom_type"

    # ── 5. Build recipe ───────────────────────────────────────────────
    # Query: all atoms from imported datasets
    query: Dict[str, Any] = {}
    if len(datasets_imported) > 0:
        # Use all dataset_ids from imported sources
        all_ds_ids = set()
        for src in sources:
            cfg_name = _resolve_config_name(src["dataset"])
            try:
                cfg = TaskConfig.builtin(cfg_name)
                all_ds_ids.add(cfg.dataset_id)
            except Exception:
                all_ds_ids.add(src["dataset"])
        query["dataset_id"] = sorted(all_ds_ids)

    # Map string split_strategy to enum
    strategy_map = {
        "subject": SplitStrategy.SUBJECT,
        "stratified": SplitStrategy.STRATIFIED,
        "dataset": SplitStrategy.DATASET,
        "temporal": SplitStrategy.TEMPORAL,
        "predefined": SplitStrategy.PREDEFINED,
    }
    split_strat = strategy_map.get(
        split_strategy, SplitStrategy.SUBJECT
    )

    # Map normalization strings
    norm_method = None
    if normalization:
        norm_map = {
            "zscore": NormalizationMethod.ZSCORE,
            "robust": NormalizationMethod.ROBUST,
            "minmax": NormalizationMethod.MINMAX,
        }
        norm_method = norm_map.get(normalization)

    scope_map = {
        "per_atom": NormalizationScope.PER_ATOM,
        "global": NormalizationScope.GLOBAL,
        "per_subject": NormalizationScope.PER_SUBJECT,
        "per_channel": NormalizationScope.PER_CHANNEL,
    }
    norm_scope = scope_map.get(
        normalization_scope, NormalizationScope.PER_ATOM
    )

    recipe_kw: Dict[str, Any] = {
        "recipe_id": f"multiload_{'_'.join(sorted(datasets_imported))}",
        "query": query,
        "target_unit": target_unit,
        "label_fields": [
            LabelSpec(annotation_name=label_field, output_key=label_field),
        ],
        "split_strategy": split_strat,
        "split_config": split_config,
    }
    if target_channels:
        recipe_kw["target_channels"] = target_channels
    if target_srate:
        recipe_kw["target_sampling_rate"] = target_srate
    if target_duration:
        recipe_kw["target_duration"] = target_duration
    if band:
        recipe_kw["filter_band"] = band
    if norm_method:
        recipe_kw["normalization_method"] = norm_method
        recipe_kw["normalization_scope"] = norm_scope

    recipe_kw.update(extra_recipe_kwargs)
    recipe = AssemblyRecipe(**recipe_kw)

    # ── 6. Assemble ───────────────────────────────────────────────────
    result = DatasetAssembler(pool, indexer).assemble(recipe)
    indexer.close()

    logger.info(
        "multiload assembly: %d train, %d val, %d test from %d sources.",
        len(result.train_samples), len(result.val_samples),
        len(result.test_samples), len(sources),
    )

    # ── 7. DataLoaders ────────────────────────────────────────────────
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader
        from neuroatom.loader.torch_dataset import AtomDataset
    except ImportError:
        raise ImportError(
            "PyTorch is required for multiload. "
            "Install with: pip install neuroatom[torch]"
        ) from None

    def _make_loader(samples, shuffle):
        if not samples:
            return None
        return DataLoader(
            AtomDataset(samples),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader = _make_loader(result.train_samples, shuffle_train)
    val_loader = _make_loader(result.val_samples, False)
    test_loader = _make_loader(result.test_samples, False)

    return train_loader, val_loader, test_loader


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
    "chinese_eeg_reading": "sentence_index",
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

    # ── ChineseEEG (reading) ────────────────────────────────────────────
    if dataset == "chinese_eeg_reading":
        importer.import_subject(
            bids_root=data_path,
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
