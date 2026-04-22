"""FederatedAssembler: assembly across multiple pools.

Transparently queries and loads atoms from multiple pools using the
``FederatedPool`` abstraction, then runs the same processing pipeline
as ``DatasetAssembler``.

Usage::

    from neuroatom.index.federation import FederatedPool
    from neuroatom.assembler.federated_assembler import FederatedAssembler

    fed = FederatedPool([pool_a, pool_b], [idx_a, idx_b])
    assembler = FederatedAssembler(fed)
    result = assembler.assemble(recipe)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.assembler.channel_mapper import ChannelMapper
from neuroatom.assembler.dataset_assembler import (
    AssemblyResult,
    DatasetAssembler,
    validate_recipe,
)
from neuroatom.assembler.filter import SignalFilter
from neuroatom.assembler.label_encoder import LabelEncoder
from neuroatom.assembler.normalizer import Normalizer, StatsCollector
from neuroatom.assembler.padcrop import PadCrop
from neuroatom.assembler.resampler import Resampler
from neuroatom.assembler.rereferencer import Rereferencer
from neuroatom.assembler.splitter import DataSplitter
from neuroatom.assembler.unit_standardizer import UnitStandardizer
from neuroatom.core.atom import Atom
from neuroatom.core.enums import ErrorHandling, NormalizationScope
from neuroatom.core.recipe import AssemblyRecipe
from neuroatom.index.federation import (
    FederatedPool,
    FederatedQueryBuilder,
    load_federated_atoms,
)
from neuroatom.storage.signal_store import ShardManager

logger = logging.getLogger(__name__)


def _apply_baseline(signal: np.ndarray, atom: Atom, method: str) -> np.ndarray:
    """Apply baseline correction (standalone version)."""
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


class FederatedAssembler:
    """Assembler that transparently loads from multiple pools.

    Follows the same pipeline as ``DatasetAssembler`` but:
    - Uses ``FederatedQueryBuilder`` for cross-pool queries
    - Routes signal loading to the correct pool per atom
    - Tracks provenance (which pool each atom came from)
    """

    def __init__(self, federation: FederatedPool):
        self._fed = federation

    def assemble(self, recipe: AssemblyRecipe) -> AssemblyResult:
        """Execute the assembly pipeline across federated pools.

        The pipeline stages are identical to DatasetAssembler.
        """
        start_time = time.time()
        result = AssemblyResult()

        # Validate recipe
        errors, warnings = validate_recipe(recipe)
        for e in errors:
            raise ValueError(f"Recipe validation error: {e}")
        for w in warnings:
            logger.warning("Recipe warning: %s", w)

        # ── Step 1: Federated query ──────────────────────────────────
        logger.info("Step 1: Federated query across %d pools...", len(self._fed.handles))
        fqb = FederatedQueryBuilder(self._fed)
        atom_ids = fqb.query_atom_ids(recipe.query)
        logger.info("Found %d matching atoms across all pools.", len(atom_ids))

        if not atom_ids:
            logger.warning(
                "No atoms matched the federated query: %s. "
                "Check that atoms have been imported and indexed in at least one pool.",
                recipe.query,
            )
            return result

        # ── Step 2: Load atoms from correct pools ────────────────────
        logger.info("Step 2: Loading atoms from federated pools...")
        atoms = load_federated_atoms(self._fed, atom_ids)
        logger.info("Loaded %d atoms.", len(atoms))

        # Build a map: atom_id → pool_root for signal loading
        atom_pool_root = {}
        for atom in atoms:
            handle = self._fed.resolve_pool(atom.atom_id)
            if handle:
                atom_pool_root[atom.atom_id] = handle.pool.root

        # ── Step 3: Build pipeline ───────────────────────────────────
        unit_std = UnitStandardizer(target_unit=recipe.target_unit)

        reref = None
        if recipe.target_reference:
            reref = Rereferencer(target_reference=recipe.target_reference)

        ch_mapper = None
        if recipe.target_channels:
            ch_mapper = ChannelMapper(target_channels=recipe.target_channels)

        sig_filter = None
        if recipe.filter_band or recipe.notch_freq:
            effective_srate = recipe.target_sampling_rate or (
                atoms[0].sampling_rate if atoms else 256.0
            )
            sig_filter = SignalFilter(
                sampling_rate=effective_srate,
                filter_band=recipe.filter_band,
                notch_freq=recipe.notch_freq,
            )

        resampler = None
        if recipe.target_sampling_rate:
            resampler = Resampler(target_rate=recipe.target_sampling_rate)

        padcrop = None
        if recipe.target_duration and recipe.target_sampling_rate:
            target_samples = PadCrop.compute_target_samples(
                recipe.target_duration, recipe.target_sampling_rate
            )
            padcrop = PadCrop(target_samples=target_samples)

        normalizer = None
        needs_two_pass = (
            recipe.normalization_method is not None
            and recipe.normalization_scope
            in (NormalizationScope.GLOBAL, NormalizationScope.PER_SUBJECT)
        )

        # ── Step 4: Label encoder ────────────────────────────────────
        logger.info("Step 4: Fitting label encoder...")
        label_encoder = LabelEncoder(recipe.label_fields)
        for atom in atoms:
            label_encoder.fit_atom(atom.annotations)
        label_encoder.finalize()
        result.label_encoder = label_encoder

        # ── Step 5: Two-pass normalization (if needed) ───────────────
        if needs_two_pass:
            logger.info("Step 5: Pass 1 — global/per-subject normalization stats...")
            n_ch = len(recipe.target_channels) if recipe.target_channels else atoms[0].n_channels
            stats_collector = StatsCollector(
                method=recipe.normalization_method,
                n_channels=n_ch,
                scope=recipe.normalization_scope,
            )
            for atom in atoms:
                try:
                    pool_root = atom_pool_root.get(atom.atom_id)
                    if pool_root is None:
                        continue
                    signal = self._load_and_preprocess(
                        atom, pool_root, unit_std, reref, ch_mapper,
                        sig_filter, resampler, recipe,
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

        # ── Step 6: Process all atoms ────────────────────────────────
        logger.info("Step 6: Processing atoms...")
        processed_atoms: List[Tuple[Atom, Dict[str, Any]]] = []

        for atom in atoms:
            try:
                pool_root = atom_pool_root.get(atom.atom_id)
                if pool_root is None:
                    result.n_skipped += 1
                    continue

                signal = self._load_and_preprocess(
                    atom, pool_root, unit_std, reref, ch_mapper,
                    sig_filter, resampler, recipe,
                )
                if signal is None:
                    result.n_skipped += 1
                    continue

                # Baseline
                if recipe.baseline_correction and recipe.baseline_before_normalize:
                    signal = _apply_baseline(
                        signal, atom, recipe.baseline_correction
                    )

                # Normalize
                if normalizer:
                    scope_key = self._get_scope_key(atom, recipe.normalization_scope)
                    signal = normalizer.apply(signal, scope_key=scope_key)

                # Pad/crop
                time_mask = None
                if padcrop:
                    signal, time_mask = padcrop.apply(signal)

                channel_mask = np.ones(signal.shape[0], dtype=np.float32)
                labels = label_encoder.encode(
                    atom.annotations, subject_id=atom.subject_id
                )

                # Track which pool this came from
                handle = self._fed.resolve_pool(atom.atom_id)
                pool_tag = handle.tag if handle else "unknown"

                sample = {
                    "atom_id": atom.atom_id,
                    "signal": signal,
                    "labels": labels,
                    "channel_mask": channel_mask if recipe.include_channel_mask else None,
                    "time_mask": time_mask if recipe.include_time_mask else None,
                    "subject_id": atom.subject_id,
                    "dataset_id": atom.dataset_id,
                    "pool_tag": pool_tag,
                }
                processed_atoms.append((atom, sample))

            except Exception as e:
                self._handle_error(e, atom, recipe.error_handling, result)

        # ── Step 7: Split ────────────────────────────────────────────
        logger.info("Step 7: Splitting...")
        splitter = DataSplitter(
            strategy=recipe.split_strategy,
            config=recipe.split_config,
        )
        atom_list = [a for a, _ in processed_atoms]
        splits = splitter.split(atom_list)

        sample_map = {a.atom_id: s for a, s in processed_atoms}
        result.train_samples = [
            sample_map[a.atom_id] for a in splits["train"] if a.atom_id in sample_map
        ]
        result.val_samples = [
            sample_map[a.atom_id] for a in splits["val"] if a.atom_id in sample_map
        ]
        result.test_samples = [
            sample_map[a.atom_id] for a in splits["test"] if a.atom_id in sample_map
        ]

        elapsed = time.time() - start_time
        result.assembly_log = {
            "recipe_id": recipe.recipe_id,
            "n_pools": len(self._fed.handles),
            "pool_tags": [h.tag for h in self._fed.handles],
            "n_queried": len(atom_ids),
            "n_loaded": len(atoms),
            "n_processed": len(processed_atoms),
            "n_train": len(result.train_samples),
            "n_val": len(result.val_samples),
            "n_test": len(result.test_samples),
            "n_errors": result.n_errors,
            "n_skipped": result.n_skipped,
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(
            "Federated assembly complete: %d processed (%d train, %d val, %d test) "
            "from %d pools in %.1fs",
            len(processed_atoms),
            len(result.train_samples),
            len(result.val_samples),
            len(result.test_samples),
            len(self._fed.handles),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_preprocess(
        atom, pool_root, unit_std, reref, ch_mapper, sig_filter, resampler, recipe,
    ):
        """Load signal from the correct pool and run pipeline steps."""
        signal = ShardManager.static_read(pool_root, atom.signal_ref)

        # Unit standardize
        source_unit = "V"
        signal = unit_std.convert(signal, source_unit, recipe.error_handling.value)

        # Re-reference
        if reref:
            exclude = atom.quality.bad_channels if atom.quality else []
            reref_inst = Rereferencer(
                target_reference=recipe.target_reference,
                exclude_channels=exclude,
            )
            signal = reref_inst.apply(signal, atom.channel_ids)

        # Channel map
        if ch_mapper:
            source_map = {ch_id: idx for idx, ch_id in enumerate(atom.channel_ids)}
            mapped, _ = ch_mapper.apply(signal, source_map)
            if mapped is None:
                return None
            signal = mapped

        # Filter
        if sig_filter:
            signal = sig_filter.apply(signal)

        # Resample
        if resampler:
            signal = resampler.apply(signal, atom.sampling_rate)

        return signal

    @staticmethod
    def _get_scope_key(atom: Atom, scope: NormalizationScope) -> str:
        if scope == NormalizationScope.PER_SUBJECT:
            return f"{atom.dataset_id}|{atom.subject_id}"
        return "global"

    @staticmethod
    def _handle_error(error, atom, error_handling, result):
        result.n_errors += 1
        if error_handling == ErrorHandling.RAISE:
            raise error
        elif error_handling == ErrorHandling.SKIP:
            logger.warning("Skipping atom %s: %s", atom.atom_id, error)
            result.n_skipped += 1
        elif error_handling == ErrorHandling.SUBSTITUTE:
            logger.warning("Error on atom %s (substitute): %s", atom.atom_id, error)
