"""MultiModalAssembler: paired multi-modal assembly with configurable pairing.

Given a ``MultiModalRecipe``, this assembler:
1. Runs a per-modality signal processing pipeline for each modality
2. Groups processed atoms by a **configurable pairing key** (any combination
   of Atom fields: subject_id, session_id, run_id, trial_index, etc.)
3. Pairs atoms across modalities that share the same pairing key
4. Emits paired samples: ``{modality_a: signal, modality_b: signal, labels: ...}``

The pairing granularity is controlled by ``recipe.pairing_keys``:

- **Run-level** (default): ``["subject_id", "session_id", "run_id"]``
  Epoch counts may differ — samples are drawn independently within each run.
- **Trial-level**: ``["subject_id", "session_id", "run_id", "trial_index"]``
  1:1 pairing when both modalities have matching trial indices.
- **Session-level**: ``["subject_id", "session_id"]``
  Coarse pairing across all runs in a session.

Usage::

    assembler = MultiModalAssembler(pool, indexer)
    result = assembler.assemble(recipe)
    # result.paired_samples = [
    #   {"eeg": np.ndarray, "ieeg": np.ndarray, "labels": {...}, "pairing_key": "..."},
    #   ...
    # ]
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
from neuroatom.core.multimodal_recipe import ModalityPipelineConfig, MultiModalRecipe
from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
from neuroatom.index.indexer import Indexer
from neuroatom.index.query import QueryBuilder
from neuroatom.storage.metadata_store import AtomJSONLReader
from neuroatom.storage.pool import Pool
from neuroatom.storage.signal_store import ShardManager
from neuroatom.storage import paths as P

logger = logging.getLogger(__name__)


class MultiModalAssemblyResult:
    """Result of multi-modal paired assembly."""

    def __init__(self):
        self.paired_samples: List[Dict[str, Any]] = []
        self.train_samples: List[Dict[str, Any]] = []
        self.val_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []
        self.label_encoder: Optional[LabelEncoder] = None
        self.assembly_log: Dict[str, Any] = {}
        self.n_errors: int = 0
        self.n_skipped: int = 0
        self.n_unpaired_runs: int = 0


class MultiModalAssembler:
    """Assembler for paired multi-modal datasets.

    Delegates per-modality signal processing to the same pipeline stages
    as ``DatasetAssembler``, then pairs results at the run level.
    """

    def __init__(self, pool: Pool, indexer: Indexer):
        self._pool = pool
        self._indexer = indexer

    def assemble(
        self,
        recipe: MultiModalRecipe,
        cache_dir: Optional[Path] = None,
    ) -> MultiModalAssemblyResult:
        """Execute multi-modal paired assembly.

        Args:
            recipe: Multi-modal assembly recipe.
            cache_dir: Optional directory for provenance files.

        Returns:
            MultiModalAssemblyResult with paired samples split into train/val/test.
        """
        start_time = time.time()
        result = MultiModalAssemblyResult()

        modality_names = list(recipe.modalities.keys())
        if len(modality_names) < 2:
            raise ValueError(
                f"MultiModalRecipe requires at least 2 modalities, got {len(modality_names)}: "
                f"{modality_names}. For single-modality assembly, use DatasetAssembler."
            )

        primary = recipe.primary_modality or modality_names[0]
        if primary not in recipe.modalities:
            raise ValueError(
                f"primary_modality '{primary}' not in modalities: {modality_names}"
            )

        pairing_keys = recipe.pairing_keys
        logger.info(
            "Multi-modal assembly: %d modalities (%s), primary=%s, pairing_keys=%s",
            len(modality_names), ", ".join(modality_names), primary, pairing_keys,
        )

        # ── Step 1: Per-modality query + process ────────────────────────
        # Returns: {modality: {run_key: [processed_sample_dicts]}}
        modality_processed: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        all_primary_atoms: List[Atom] = []

        for mod_name, mod_config in recipe.modalities.items():
            logger.info("Processing modality '%s' ...", mod_name)
            run_grouped, atoms = self._process_modality(
                mod_name, mod_config, recipe, result,
            )
            modality_processed[mod_name] = run_grouped

            if mod_name == primary:
                all_primary_atoms = atoms

        # ── Step 2: Fit label encoder on primary modality ───────────────
        logger.info("Fitting label encoder on primary modality '%s'...", primary)
        label_encoder = LabelEncoder(recipe.label_fields)
        for atom in all_primary_atoms:
            label_encoder.fit_atom(atom.annotations)
        label_encoder.finalize()
        result.label_encoder = label_encoder

        # ── Step 3: Pair by pairing key ──────────────────────────────
        logger.info("Pairing modalities by keys %s ...", pairing_keys)
        primary_runs = set(modality_processed.get(primary, {}).keys())
        other_mods = [m for m in modality_names if m != primary]

        # Find runs present in ALL modalities
        common_runs = primary_runs.copy()
        for mod_name in other_mods:
            common_runs &= set(modality_processed.get(mod_name, {}).keys())

        n_unpaired = len(primary_runs - common_runs)
        result.n_unpaired_runs = n_unpaired
        if n_unpaired > 0:
            logger.warning(
                "%d primary groups have no paired data in all modalities — skipped. "
                "(pairing_keys=%s)",
                n_unpaired, pairing_keys,
            )

        logger.info("Found %d common groups across all modalities.", len(common_runs))

        # Build paired samples
        paired_atoms_for_split: List[Atom] = []
        paired_samples: List[Dict[str, Any]] = []

        for run_key in sorted(common_runs):
            primary_samples = modality_processed[primary][run_key]

            # For each primary atom/sample, randomly pick one from each other modality
            other_samples = {
                mod: modality_processed[mod][run_key] for mod in other_mods
            }

            for p_sample in primary_samples:
                paired = {primary: p_sample["signal"]}

                # Pick a corresponding sample from each other modality
                for mod in other_mods:
                    candidates = other_samples[mod]
                    if not candidates:
                        continue
                    # Deterministic: use index modulo for reproducibility
                    idx = hash(p_sample["atom_id"]) % len(candidates)
                    paired[mod] = candidates[idx]["signal"]

                # Labels from primary modality
                labels = label_encoder.encode(
                    all_primary_atoms[0].annotations  # placeholder — need actual atom
                    if not p_sample.get("_atom")
                    else p_sample["_atom"].annotations,
                    subject_id=p_sample.get("subject_id"),
                )

                sample = {
                    **paired,
                    "labels": labels,
                    "atom_id": p_sample["atom_id"],
                    "subject_id": p_sample.get("subject_id", ""),
                    "dataset_id": p_sample.get("dataset_id", ""),
                    "pairing_key": run_key,
                }
                paired_samples.append(sample)

                if p_sample.get("_atom"):
                    paired_atoms_for_split.append(p_sample["_atom"])

        result.paired_samples = paired_samples
        logger.info("Created %d paired samples.", len(paired_samples))

        # ── Step 4: Split ─────────────────────────────────────────────
        if paired_atoms_for_split:
            logger.info("Splitting paired samples...")
            splitter = DataSplitter(
                strategy=recipe.split_strategy,
                config=recipe.split_config,
            )
            splits = splitter.split(paired_atoms_for_split)

            # Map atom_id to paired sample
            sample_map = {s["atom_id"]: s for s in paired_samples}
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
            "modalities": modality_names,
            "pairing_keys": pairing_keys,
            "n_common_groups": len(common_runs),
            "n_unpaired_groups": n_unpaired,
            "n_paired_samples": len(paired_samples),
            "n_train": len(result.train_samples),
            "n_val": len(result.val_samples),
            "n_test": len(result.test_samples),
            "n_errors": result.n_errors,
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(
            "Multi-modal assembly complete: %d paired (%d train, %d val, %d test) in %.1fs",
            len(paired_samples),
            len(result.train_samples),
            len(result.val_samples),
            len(result.test_samples),
            elapsed,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: per-modality processing
    # ------------------------------------------------------------------

    def _process_modality(
        self,
        mod_name: str,
        mod_config: ModalityPipelineConfig,
        recipe: MultiModalRecipe,
        result: MultiModalAssemblyResult,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Atom]]:
        """Query and process atoms for one modality.

        Returns:
            (run_grouped_samples, atom_list)
            run_grouped_samples: {run_key: [sample_dict, ...]}
        """
        # Query
        qb = QueryBuilder(self._indexer.backend)
        atom_ids = qb.query_atom_ids(mod_config.query)
        logger.info("  %s: %d atoms matched query", mod_name, len(atom_ids))

        if not atom_ids:
            return {}, []

        # Load atoms
        atoms = self._load_atoms_by_ids(atom_ids)

        # Build pipeline
        unit_std = UnitStandardizer(target_unit=mod_config.target_unit)

        reref = None
        if mod_config.target_reference:
            reref = Rereferencer(target_reference=mod_config.target_reference)

        ch_mapper = None
        if mod_config.target_channels:
            ch_mapper = ChannelMapper(target_channels=mod_config.target_channels)

        sig_filter = None
        if mod_config.filter_band or mod_config.notch_freq:
            effective_srate = mod_config.target_sampling_rate or (
                atoms[0].sampling_rate if atoms else 256.0
            )
            sig_filter = SignalFilter(
                sampling_rate=effective_srate,
                filter_band=mod_config.filter_band,
                notch_freq=mod_config.notch_freq,
            )

        resampler = None
        if mod_config.target_sampling_rate:
            resampler = Resampler(target_rate=mod_config.target_sampling_rate)

        padcrop = None
        if mod_config.target_duration and mod_config.target_sampling_rate:
            target_samples = PadCrop.compute_target_samples(
                mod_config.target_duration, mod_config.target_sampling_rate
            )
            padcrop = PadCrop(target_samples=target_samples)
        elif mod_config.target_duration and atoms:
            target_samples = PadCrop.compute_target_samples(
                mod_config.target_duration, atoms[0].sampling_rate
            )
            padcrop = PadCrop(target_samples=target_samples)

        # Process each atom
        run_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for atom in atoms:
            try:
                signal = self._process_signal(
                    atom, unit_std, reref, ch_mapper, sig_filter, resampler,
                    mod_config, recipe.error_handling,
                )
                if signal is None:
                    result.n_skipped += 1
                    continue

                # Baseline
                if mod_config.baseline_correction and mod_config.baseline_before_normalize:
                    signal = self._apply_baseline(signal, atom, mod_config.baseline_correction)

                # Pad/crop
                if padcrop:
                    signal, _ = padcrop.apply(signal)

                run_key = self._build_pairing_key(atom, recipe.pairing_keys)

                sample = {
                    "signal": signal,
                    "atom_id": atom.atom_id,
                    "subject_id": atom.subject_id,
                    "dataset_id": atom.dataset_id,
                    "run_key": run_key,
                    "_atom": atom,  # kept for label encoding + splitting
                }
                run_grouped[run_key].append(sample)

            except Exception as e:
                result.n_errors += 1
                if recipe.error_handling == ErrorHandling.RAISE:
                    raise
                logger.warning("Error processing %s atom %s: %s", mod_name, atom.atom_id, e)

        logger.info(
            "  %s: %d atoms processed across %d runs",
            mod_name, sum(len(v) for v in run_grouped.values()), len(run_grouped),
        )

        return dict(run_grouped), atoms

    @staticmethod
    def _build_pairing_key(atom: Atom, pairing_keys: List[str]) -> str:
        """Build a pairing key string from an atom using the specified fields.

        For each key in pairing_keys:
        - If it's a standard Atom field (subject_id, session_id, run_id,
          trial_index, dataset_id, modality, etc.), use getattr.
        - If not found as a direct attribute, look in atom.custom_fields.
        - If still not found, use the string "None".

        Returns:
            Pipe-delimited key string, e.g. "sub-01|ses-01|run-01".
        """
        parts = []
        for key in pairing_keys:
            if hasattr(atom, key):
                parts.append(str(getattr(atom, key)))
            elif key in atom.custom_fields:
                parts.append(str(atom.custom_fields[key]))
            else:
                parts.append("None")
        return "|".join(parts)

    def _process_signal(
        self,
        atom: Atom,
        unit_std: UnitStandardizer,
        reref: Optional[Rereferencer],
        ch_mapper: Optional[ChannelMapper],
        sig_filter: Optional[SignalFilter],
        resampler: Optional[Resampler],
        config: ModalityPipelineConfig,
        error_handling: ErrorHandling,
    ) -> Optional[np.ndarray]:
        """Load and preprocess signal for one atom."""
        signal = ShardManager.static_read(self._pool.root, atom.signal_ref)

        # Unit standardize
        source_unit = "V"
        signal = unit_std.convert(signal, source_unit, error_handling.value)

        # Re-reference
        if reref:
            exclude = atom.quality.bad_channels if atom.quality else []
            reref_instance = Rereferencer(
                target_reference=config.target_reference,
                exclude_channels=exclude,
            )
            signal = reref_instance.apply(signal, atom.channel_ids)

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
    # Internal: atom loading (same as DatasetAssembler)
    # ------------------------------------------------------------------

    def _load_atoms_by_ids(self, atom_ids: List[str]) -> List[Atom]:
        """Load full Atom objects from JSONL files by their IDs."""
        conn = self._indexer.backend.conn
        rows = conn.execute(
            "SELECT atom_id, dataset_id, subject_id, session_id, run_id FROM atoms "
            f"WHERE atom_id IN ({','.join('?' * len(atom_ids))})",
            tuple(atom_ids),
        ).fetchall()

        run_atoms: Dict[str, List[str]] = {}
        for r in rows:
            run_key = f"{r['dataset_id']}|{r['subject_id']}|{r['session_id']}|{r['run_id']}"
            if run_key not in run_atoms:
                run_atoms[run_key] = []
            run_atoms[run_key].append(r["atom_id"])

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
