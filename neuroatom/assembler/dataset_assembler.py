"""DatasetAssembler: Recipe-driven assembly of ML-ready datasets.

Orchestrates the full assembly pipeline:
    Query → Load atoms → Unit Std → Re-ref → Ch Map → Filter → Resample
    → Baseline → Normalize → Pad/Crop → Label Encode → Split → Cache

Supports two-pass normalization for global/per_subject scopes:
    Pass 1: stream all atoms → compute normalization stats
    Pass 2: stream again → apply normalization + write cache

Cache provenance files:
    cache_dir/recipe.yaml      — full Recipe serialization
    cache_dir/assembly_log.json — execution log with timestamps
    cache_dir/stats.json       — normalization stats (if computed)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from neuroatom.assembler.channel_mapper import ChannelMapper
from neuroatom.assembler.filter import SignalFilter
from neuroatom.assembler.label_encoder import LabelEncoder
from neuroatom.assembler.normalizer import (
    Normalizer,
    NormalizationStats,
    StatsCollector,
)
from neuroatom.assembler.padcrop import PadCrop
from neuroatom.assembler.resampler import Resampler
from neuroatom.assembler.rereferencer import Rereferencer
from neuroatom.assembler.splitter import DataSplitter
from neuroatom.assembler.unit_standardizer import UnitStandardizer
from neuroatom.core.atom import Atom
from neuroatom.core.enums import (
    ErrorHandling,
    NormalizationMethod,
    NormalizationScope,
)
from neuroatom.core.recipe import AssemblyRecipe
from neuroatom.index.indexer import Indexer
from neuroatom.index.query import QueryBuilder
from neuroatom.storage.metadata_store import AtomJSONLReader
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.storage import paths as P

logger = logging.getLogger(__name__)


class RecipeValidationError(ValueError):
    """Raised when a recipe contains logically inconsistent parameters."""
    pass


def validate_recipe(recipe: AssemblyRecipe) -> List[str]:
    """Pre-flight validation of an AssemblyRecipe.

    Catches logical contradictions BEFORE the pipeline runs, converting
    cryptic numpy/scipy errors into actionable user-facing messages.

    Returns list of warning messages (non-fatal). Raises RecipeValidationError
    for fatal contradictions.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Nyquist check: bandpass upper frequency vs target_sampling_rate
    if recipe.filter_band and recipe.target_sampling_rate:
        nyquist = recipe.target_sampling_rate / 2.0
        low_hz, high_hz = recipe.filter_band
        if high_hz >= nyquist:
            errors.append(
                f"Bandpass upper frequency ({high_hz} Hz) ≥ Nyquist frequency "
                f"({nyquist} Hz) for target sampling rate {recipe.target_sampling_rate} Hz. "
                f"Either lower the bandpass upper limit or increase target_sampling_rate."
            )
        if low_hz >= high_hz:
            errors.append(
                f"Bandpass lower frequency ({low_hz} Hz) ≥ upper frequency ({high_hz} Hz)."
            )
        if low_hz < 0:
            errors.append(f"Bandpass lower frequency ({low_hz} Hz) must be ≥ 0.")

    # 2. Notch frequency vs Nyquist
    if recipe.notch_freq and recipe.target_sampling_rate:
        nyquist = recipe.target_sampling_rate / 2.0
        if recipe.notch_freq >= nyquist:
            errors.append(
                f"Notch frequency ({recipe.notch_freq} Hz) ≥ Nyquist frequency "
                f"({nyquist} Hz) for target sampling rate {recipe.target_sampling_rate} Hz."
            )

    # 3. Normalization scope warnings
    if recipe.normalization_method and recipe.normalization_scope == NormalizationScope.PER_SUBJECT:
        warnings.append(
            "normalization_scope='per_subject' requires all atoms to have non-null subject_id. "
            "Atoms with null subject_id will fall back to a single 'unknown' group."
        )

    # 4. Channel mapping: warn if target_channels is very large without explicit data knowledge
    if recipe.target_channels and len(recipe.target_channels) > 128:
        warnings.append(
            f"target_channels has {len(recipe.target_channels)} channels. "
            f"Atoms with fewer channels will have heavily zero-filled output."
        )

    # 5. Duration vs sampling rate: check that target_duration produces reasonable sample count
    if recipe.target_duration and recipe.target_sampling_rate:
        n_samples = int(recipe.target_duration * recipe.target_sampling_rate)
        if n_samples < 10:
            errors.append(
                f"target_duration={recipe.target_duration}s at {recipe.target_sampling_rate}Hz "
                f"produces only {n_samples} samples. This is too short for meaningful analysis."
            )
        if n_samples > 1_000_000:
            warnings.append(
                f"target_duration={recipe.target_duration}s at {recipe.target_sampling_rate}Hz "
                f"produces {n_samples:,} samples per atom. Consider reducing duration or sampling rate."
            )

    # 6. Augmentation sanity checks
    for aug in recipe.augmentations:
        if hasattr(aug, "max_shift_seconds") and recipe.target_duration:
            if aug.max_shift_seconds > recipe.target_duration * 0.5:
                warnings.append(
                    f"TemporalShift max_shift_seconds ({aug.max_shift_seconds}s) is > 50% "
                    f"of target_duration ({recipe.target_duration}s). This may degrade signal quality."
                )

    # 7. Scale range sanity
    for aug in recipe.augmentations:
        if hasattr(aug, "scale_range"):
            lo, hi = aug.scale_range
            if lo <= 0:
                errors.append(f"SignalScale scale_range lower bound ({lo}) must be > 0.")
            if lo > hi:
                errors.append(f"SignalScale scale_range ({lo}, {hi}) is inverted.")

    if errors:
        raise RecipeValidationError(
            "Recipe validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return warnings


class AssemblyResult:
    """Result of dataset assembly."""

    def __init__(self):
        self.train_samples: List[Dict[str, Any]] = []
        self.val_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []
        self.label_encoder: Optional[LabelEncoder] = None
        self.norm_stats: Optional[NormalizationStats] = None
        self.assembly_log: Dict[str, Any] = {}
        self.n_errors: int = 0
        self.n_skipped: int = 0


class DatasetAssembler:
    """Recipe-driven assembly of ML-ready datasets from the atom pool.

    Usage:
        assembler = DatasetAssembler(pool, indexer)
        result = assembler.assemble(recipe)
        # result.train_samples = [{"signal": ..., "labels": ..., "masks": ...}, ...]
    """

    def __init__(self, pool: Pool, indexer: Indexer):
        self._pool = pool
        self._indexer = indexer

    def assemble(
        self,
        recipe: AssemblyRecipe,
        cache_dir: Optional[Path] = None,
    ) -> AssemblyResult:
        """Execute the full assembly pipeline.

        Args:
            recipe: Assembly recipe specification.
            cache_dir: Optional directory to cache provenance files.

        Returns:
            AssemblyResult containing processed samples split into train/val/test.
        """
        start_time = time.time()
        result = AssemblyResult()

        # ---- Step 0: Validate recipe ----
        logger.info("Step 0: Validating recipe...")
        validation_warnings = validate_recipe(recipe)
        for w in validation_warnings:
            logger.warning("Recipe warning: %s", w)

        # ---- Step 1: Query atoms ----
        logger.info("Step 1: Querying atoms...")
        qb = QueryBuilder(self._indexer.backend)
        atom_ids = qb.query_atom_ids(recipe.query)
        logger.info("Found %d matching atoms.", len(atom_ids))

        if not atom_ids:
            logger.warning(
                "No atoms matched the query: %s. "
                "Check that: (1) atoms have been imported and indexed, "
                "(2) query field values match actual data (dataset_id, subject_id, etc.), "
                "(3) annotation filters reference existing annotation names.",
                recipe.query,
            )
            return result

        # ---- Step 2: Load atom metadata ----
        logger.info("Step 2: Loading atom metadata...")
        atoms = self._load_atoms_by_ids(atom_ids)
        logger.info("Loaded %d atoms.", len(atoms))

        # ---- Step 3: Build pipeline components ----
        unit_std = UnitStandardizer(target_unit=recipe.target_unit)
        reref = None
        if recipe.target_reference:
            reref = Rereferencer(target_reference=recipe.target_reference)

        ch_mapper = None
        if recipe.target_channels:
            ch_mapper = ChannelMapper(target_channels=recipe.target_channels)

        sig_filter = None
        if recipe.filter_band or recipe.notch_freq:
            # Determine the effective sampling rate for filter design
            effective_srate = recipe.target_sampling_rate or self._guess_srate(atoms)
            sig_filter = SignalFilter(
                sampling_rate=effective_srate,
                filter_band=recipe.filter_band,
                notch_freq=recipe.notch_freq,
            )

        resampler = None
        if recipe.target_sampling_rate:
            resampler = Resampler(target_rate=recipe.target_sampling_rate)

        normalizer = None
        norm_stats = None
        needs_two_pass = (
            recipe.normalization_method is not None
            and recipe.normalization_scope in (
                NormalizationScope.GLOBAL,
                NormalizationScope.PER_SUBJECT,
            )
        )

        padcrop = None
        if recipe.target_duration and recipe.target_sampling_rate:
            target_samples = PadCrop.compute_target_samples(
                recipe.target_duration, recipe.target_sampling_rate
            )
            padcrop = PadCrop(target_samples=target_samples)
        elif recipe.target_duration:
            srate = self._guess_srate(atoms)
            target_samples = PadCrop.compute_target_samples(recipe.target_duration, srate)
            padcrop = PadCrop(target_samples=target_samples)

        # ---- Step 4: Label encoder (fit pass) ----
        logger.info("Step 4: Fitting label encoder...")
        label_encoder = LabelEncoder(recipe.label_fields)
        for atom in atoms:
            label_encoder.fit_atom(atom.annotations)
        label_encoder.finalize()
        result.label_encoder = label_encoder

        # ---- Step 5: Two-pass normalization (if needed) ----
        if needs_two_pass:
            logger.info("Step 5: Pass 1 — Computing normalization statistics...")
            n_ch = len(recipe.target_channels) if recipe.target_channels else atoms[0].n_channels
            stats_collector = StatsCollector(
                method=recipe.normalization_method,
                n_channels=n_ch,
                scope=recipe.normalization_scope,
            )
            for atom in atoms:
                try:
                    signal = self._load_and_preprocess_signal(
                        atom, unit_std, reref, ch_mapper, sig_filter, resampler, recipe
                    )
                    if signal is None:
                        continue
                    scope_key = self._get_scope_key(atom, recipe.normalization_scope)
                    stats_collector.update(signal, scope_key)
                except Exception as e:
                    self._handle_error(e, atom, recipe.error_handling, result)

            norm_stats = stats_collector.finalize()
            result.norm_stats = norm_stats
            normalizer = Normalizer(
                method=recipe.normalization_method,
                scope=recipe.normalization_scope,
                precomputed_stats=norm_stats,
            )
        elif recipe.normalization_method is not None:
            normalizer = Normalizer(
                method=recipe.normalization_method,
                scope=recipe.normalization_scope,
            )

        # ---- Step 6: Process all atoms (Pass 2 for two-pass, or single pass) ----
        logger.info("Step 6: Processing atoms...")
        processed_atoms: List[Tuple[Atom, Dict[str, Any]]] = []

        for atom in atoms:
            try:
                signal = self._load_and_preprocess_signal(
                    atom, unit_std, reref, ch_mapper, sig_filter, resampler, recipe
                )
                if signal is None:
                    result.n_skipped += 1
                    continue

                # Baseline correction
                if recipe.baseline_correction and recipe.baseline_before_normalize:
                    signal = self._apply_baseline(signal, atom, recipe.baseline_correction)

                # Normalize
                if normalizer:
                    scope_key = self._get_scope_key(atom, recipe.normalization_scope)
                    signal = normalizer.apply(signal, scope_key=scope_key)

                # Baseline after normalize (if configured)
                if recipe.baseline_correction and not recipe.baseline_before_normalize:
                    signal = self._apply_baseline(signal, atom, recipe.baseline_correction)

                # Pad/crop
                time_mask = None
                if padcrop:
                    signal, time_mask = padcrop.apply(signal)

                # Channel mask
                channel_mask = np.ones(signal.shape[0], dtype=np.float32)

                # Encode labels
                labels = label_encoder.encode(
                    atom.annotations, subject_id=atom.subject_id
                )

                sample = {
                    "atom_id": atom.atom_id,
                    "signal": signal,
                    "labels": labels,
                    "channel_mask": channel_mask if recipe.include_channel_mask else None,
                    "time_mask": time_mask if recipe.include_time_mask else None,
                    "subject_id": atom.subject_id,
                    "dataset_id": atom.dataset_id,
                }
                processed_atoms.append((atom, sample))

            except Exception as e:
                self._handle_error(e, atom, recipe.error_handling, result)

        # ---- Step 7: Split ----
        logger.info("Step 7: Splitting data...")
        atom_list = [a for a, _ in processed_atoms]
        sample_map = {a.atom_id: s for a, s in processed_atoms}

        splitter = DataSplitter(
            strategy=recipe.split_strategy,
            config=recipe.split_config,
        )
        splits = splitter.split(atom_list)

        result.train_samples = [sample_map[a.atom_id] for a in splits["train"]]
        result.val_samples = [sample_map[a.atom_id] for a in splits["val"]]
        result.test_samples = [sample_map[a.atom_id] for a in splits["test"]]

        # ---- Step 8: Cache provenance ----
        elapsed = time.time() - start_time
        result.assembly_log = {
            "recipe_id": recipe.recipe_id,
            "n_queried": len(atom_ids),
            "n_processed": len(processed_atoms),
            "n_skipped": result.n_skipped,
            "n_errors": result.n_errors,
            "n_train": len(result.train_samples),
            "n_val": len(result.val_samples),
            "n_test": len(result.test_samples),
            "elapsed_seconds": round(elapsed, 2),
        }

        if cache_dir:
            self._write_cache_provenance(cache_dir, recipe, result)

        logger.info(
            "Assembly complete: %d train, %d val, %d test (%.1fs)",
            len(result.train_samples),
            len(result.val_samples),
            len(result.test_samples),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: signal preprocessing
    # ------------------------------------------------------------------

    def _load_and_preprocess_signal(
        self,
        atom: Atom,
        unit_std: UnitStandardizer,
        reref: Optional[Rereferencer],
        ch_mapper: Optional[ChannelMapper],
        sig_filter: Optional[SignalFilter],
        resampler: Optional[Resampler],
        recipe: AssemblyRecipe,
    ) -> Optional[np.ndarray]:
        """Load and apply early pipeline steps (before baseline/normalize)."""
        # Load signal
        signal = ShardManager.static_read(self._pool.root, atom.signal_ref)

        # Unit standardize
        # Determine source unit (from channel info or default V)
        source_unit = "V"  # MNE default
        signal = unit_std.convert(signal, source_unit, recipe.error_handling.value)

        # Re-reference
        if reref:
            exclude = []
            if atom.quality:
                exclude = atom.quality.bad_channels
            reref_instance = Rereferencer(
                target_reference=recipe.target_reference,
                exclude_channels=exclude,
            )
            signal = reref_instance.apply(signal, atom.channel_ids)

        # Channel map
        if ch_mapper:
            # Build standard_name → index map
            # For now, use channel_ids as standard names (proper mapping via session metadata)
            source_map = {ch_id: idx for idx, ch_id in enumerate(atom.channel_ids)}
            mapped, _ = ch_mapper.apply(signal, source_map)
            if mapped is None:
                return None
            signal = mapped

        # Filter (before resample for proper cutoff)
        if sig_filter:
            signal = sig_filter.apply(signal)

        # Resample
        if resampler:
            signal = resampler.apply(signal, atom.sampling_rate)

        return signal

    def _apply_baseline(
        self, signal: np.ndarray, atom: Atom, method: str
    ) -> np.ndarray:
        """Apply baseline correction."""
        bl_start = atom.temporal.baseline_start_sample
        bl_end = atom.temporal.baseline_end_sample

        if bl_start is None or bl_end is None:
            return signal

        baseline = signal[:, bl_start:bl_end]
        if baseline.size == 0:
            return signal

        if method == "mean":
            correction = baseline.mean(axis=1, keepdims=True)
        elif method == "median":
            correction = np.median(baseline, axis=1, keepdims=True)
        else:
            return signal

        return (signal - correction).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _load_atoms_by_ids(self, atom_ids: List[str]) -> List[Atom]:
        """Load full Atom objects from JSONL files by their IDs."""
        # Build index: atom_id → run location
        conn = self._indexer.backend.conn
        rows = conn.execute(
            "SELECT atom_id, dataset_id, subject_id, session_id, run_id FROM atoms "
            f"WHERE atom_id IN ({','.join('?' * len(atom_ids))})",
            tuple(atom_ids),
        ).fetchall()

        # Group by run
        run_atoms: Dict[str, List[str]] = {}
        atom_locations: Dict[str, Tuple[str, str, str, str]] = {}
        for r in rows:
            run_key = f"{r['dataset_id']}|{r['subject_id']}|{r['session_id']}|{r['run_id']}"
            if run_key not in run_atoms:
                run_atoms[run_key] = []
            run_atoms[run_key].append(r["atom_id"])
            atom_locations[r["atom_id"]] = (
                r["dataset_id"], r["subject_id"], r["session_id"], r["run_id"]
            )

        # Load atoms from JSONL
        target_ids = set(atom_ids)
        result = []

        for run_key, ids in run_atoms.items():
            parts = run_key.split("|")
            ds_id, sub_id, ses_id, run_id = parts
            jsonl_path = P.atoms_jsonl_path(
                self._pool.root, ds_id, sub_id, ses_id, run_id
            )
            reader = AtomJSONLReader(jsonl_path)
            for atom in reader.iter_atoms():
                if atom.atom_id in target_ids:
                    result.append(atom)

        return result

    def _get_scope_key(self, atom: Atom, scope: NormalizationScope) -> Optional[str]:
        """Get the normalization scope key for an atom."""
        if scope == NormalizationScope.GLOBAL:
            return "__global__"
        elif scope == NormalizationScope.PER_SUBJECT:
            return atom.subject_id
        elif scope == NormalizationScope.PER_ATOM:
            return None
        elif scope == NormalizationScope.PER_CHANNEL:
            return None
        return None

    def _guess_srate(self, atoms: List[Atom]) -> float:
        """Guess sampling rate from first atom."""
        if atoms:
            return atoms[0].sampling_rate
        return 256.0

    def _handle_error(
        self,
        error: Exception,
        atom: Atom,
        error_handling: ErrorHandling,
        result: AssemblyResult,
    ) -> None:
        """Handle errors during assembly per error_handling policy."""
        result.n_errors += 1
        if error_handling == ErrorHandling.RAISE:
            raise error
        elif error_handling == ErrorHandling.SKIP:
            logger.warning("Skipping atom %s: %s", atom.atom_id, error)
            result.n_skipped += 1
        elif error_handling == ErrorHandling.SUBSTITUTE:
            logger.warning("Error on atom %s (substitute): %s", atom.atom_id, error)

    def _write_cache_provenance(
        self,
        cache_dir: Path,
        recipe: AssemblyRecipe,
        result: AssemblyResult,
    ) -> None:
        """Write cache provenance files for reproducibility."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # recipe.yaml
        recipe_path = cache_dir / "recipe.yaml"
        with open(recipe_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                recipe.model_dump(mode="json"),
                f,
                default_flow_style=False,
                allow_unicode=True,
            )

        # assembly_log.json
        log_path = cache_dir / "assembly_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result.assembly_log, f, indent=2, ensure_ascii=False)

        # stats.json (if normalization stats exist)
        if result.norm_stats:
            stats_path = cache_dir / "stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(result.norm_stats.to_dict(), f, indent=2)

        logger.info("Cache provenance written to %s", cache_dir)
