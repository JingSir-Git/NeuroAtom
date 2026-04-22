#!/usr/bin/env python3
"""Self-supervised pre-training with a Masked Autoencoder (MAE) Transformer.

===========================================================================
  NeuroAtom × EEG-MAE: Heterogeneous Data → Unified Self-supervised Learning
===========================================================================

This example demonstrates NeuroAtom's core value proposition:

    Multiple EEG datasets with DIFFERENT formats, channel counts,
    and sampling rates  →  unified atomic tensors  →  single MAE
    pre-training loop  →  learned representations transferable
    across datasets and tasks.

Specifically, it shows two modes:

1. **Single-dataset mode** (quickload — 5 lines):
       python pretrain_eeg_mae.py --data-dir C:/Data/BCI_Competition --dataset bci_comp_iv_2a

2. **Multi-dataset mode** (heterogeneous pre-training):
       python pretrain_eeg_mae.py --multi \
           --bci-dir C:/Data/BCI_Competition \
           --physionet-dir C:/Data/Physionet

   In multi-dataset mode, NeuroAtom's assembly pipeline:
   - Maps BCI IV 2a (25 ch @ 250 Hz) and PhysioNet MI (64 ch @ 160 Hz)
     to a COMMON 22-channel × 128 Hz representation
   - Applies identical bandpass filtering and normalization
   - Outputs uniform (B, 22, 512) tensors regardless of source

Architecture: EEG-MAE
---------------------
Inspired by He et al., "Masked Autoencoders Are Scalable Vision Learners"
(CVPR 2022), adapted for multi-channel EEG time series:

1. **Patchify**: Divide (C, T) into temporal patches → (N_patches, C·P)
2. **Mask**: Randomly mask 75% of patches (aggressive masking, as in MAE)
3. **Encode**: Transformer encoder on VISIBLE patches only (efficient)
4. **Decode**: Lightweight decoder reconstructs MASKED patches from
   encoded visible patches + mask tokens + positional embeddings
5. **Loss**: MSE between reconstructed and original patches

The encoder learns dense representations from sparse input, capturing
temporal and spatial dependencies in EEG without labels.

Prerequisites:
    pip install neuroatom[torch,mne]

Author: NeuroAtom Project
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eeg_mae")


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — EEG-MAE Transformer Model
# ═══════════════════════════════════════════════════════════════════════════


def build_eeg_mae(
    n_channels: int,
    n_samples: int,
    patch_size: int = 32,
    embed_dim: int = 256,
    encoder_depth: int = 6,
    encoder_heads: int = 8,
    decoder_embed_dim: int = 128,
    decoder_depth: int = 2,
    decoder_heads: int = 4,
    mask_ratio: float = 0.75,
    dropout: float = 0.1,
):
    """Build an EEG Masked Autoencoder.

    Args:
        n_channels: Number of EEG channels (e.g. 22).
        n_samples: Number of time samples per trial (e.g. 512).
        patch_size: Temporal patch size in samples (e.g. 32 → 16 patches).
        embed_dim: Encoder embedding dimension.
        encoder_depth: Number of encoder Transformer blocks.
        encoder_heads: Number of attention heads in encoder.
        decoder_embed_dim: Decoder embedding dimension (lighter).
        decoder_depth: Number of decoder Transformer blocks.
        decoder_heads: Number of attention heads in decoder.
        mask_ratio: Fraction of patches to mask (default 0.75).
        dropout: Dropout rate.

    Returns:
        EEGMAE model instance.
    """
    import torch
    import torch.nn as nn

    assert n_samples % patch_size == 0, (
        f"n_samples ({n_samples}) must be divisible by patch_size ({patch_size})"
    )
    n_patches = n_samples // patch_size
    patch_dim = n_channels * patch_size  # each patch is all channels × P samples

    # ------------------------------------------------------------------
    # Transformer building blocks
    # ------------------------------------------------------------------

    class MultiHeadSelfAttention(nn.Module):
        """Standard multi-head self-attention."""

        def __init__(self, dim, n_heads, drop=0.0):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.attn_drop = nn.Dropout(drop)
            self.proj_drop = nn.Dropout(drop)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj_drop(self.proj(out))

    class TransformerBlock(nn.Module):
        """Pre-norm Transformer block: LN → MHSA → LN → FFN."""

        def __init__(self, dim, n_heads, mlp_ratio=4.0, drop=0.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = MultiHeadSelfAttention(dim, n_heads, drop)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(int(dim * mlp_ratio), dim),
                nn.Dropout(drop),
            )

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    # ------------------------------------------------------------------
    # EEG-MAE Model
    # ------------------------------------------------------------------

    class EEGMAE(nn.Module):
        """Masked Autoencoder for multi-channel EEG signals.

        Workflow:
            1. Patchify (C, T) → (N_patches, C·P)
            2. Project to embed_dim → add positional embeddings
            3. Random masking: keep (1 - mask_ratio) visible patches
            4. Encoder: Transformer on visible patches only
            5. Insert mask tokens at masked positions
            6. Decoder: lightweight Transformer → reconstruct all patches
            7. Loss: MSE on MASKED patches only (reconstruction target)
        """

        def __init__(self):
            super().__init__()

            self.n_channels = n_channels
            self.n_samples = n_samples
            self.patch_size = patch_size
            self.n_patches = n_patches
            self.patch_dim = patch_dim
            self.mask_ratio = mask_ratio

            # ---- Encoder ----
            self.patch_embed = nn.Linear(patch_dim, embed_dim)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, n_patches, embed_dim)
            )
            self.encoder_blocks = nn.ModuleList([
                TransformerBlock(embed_dim, encoder_heads, drop=dropout)
                for _ in range(encoder_depth)
            ])
            self.encoder_norm = nn.LayerNorm(embed_dim)

            # ---- Decoder ----
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, n_patches, decoder_embed_dim)
            )
            self.decoder_blocks = nn.ModuleList([
                TransformerBlock(decoder_embed_dim, decoder_heads, drop=dropout)
                for _ in range(decoder_depth)
            ])
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)

            self._init_weights()

        def _init_weights(self):
            # Sinusoidal positional encoding for initialization
            pos_enc = _sinusoidal_position_encoding(n_patches, embed_dim)
            self.pos_embed.data.copy_(torch.from_numpy(pos_enc).unsqueeze(0))

            dec_pos_enc = _sinusoidal_position_encoding(n_patches, decoder_embed_dim)
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(dec_pos_enc).unsqueeze(0)
            )

            nn.init.normal_(self.mask_token, std=0.02)

            # Initialize linear layers
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def patchify(self, x: torch.Tensor) -> torch.Tensor:
            """(B, C, T) → (B, N_patches, C·P)"""
            B, C, T = x.shape
            assert T == self.n_samples and C == self.n_channels
            # Reshape: (B, C, N, P) → (B, N, C, P) → (B, N, C·P)
            x = x.reshape(B, C, self.n_patches, self.patch_size)
            x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, self.patch_dim)
            return x

        def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
            """(B, N_patches, C·P) → (B, C, T)"""
            B = x.shape[0]
            # (B, N, C·P) → (B, N, C, P) → (B, C, N, P) → (B, C, T)
            x = x.reshape(B, self.n_patches, self.n_channels, self.patch_size)
            x = x.permute(0, 2, 1, 3).reshape(B, self.n_channels, self.n_samples)
            return x

        def random_masking(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Apply random masking.

            Args:
                x: (B, N, D) embedded patches.

            Returns:
                x_visible: (B, N_visible, D) — visible patches only.
                mask: (B, N) — binary mask: 1 = MASKED, 0 = visible.
                ids_restore: (B, N) — indices to unshuffle.
            """
            B, N, D = x.shape
            n_keep = int(N * (1 - self.mask_ratio))

            # Random permutation per sample
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Keep first n_keep (visible)
            ids_keep = ids_shuffle[:, :n_keep]
            x_visible = torch.gather(
                x, dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
            )

            # Binary mask: 1 = masked
            mask = torch.ones(B, N, device=x.device)
            mask[:, :n_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return x_visible, mask, ids_restore

        def encode(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Encode: patchify → embed → mask → Transformer.

            Returns:
                latent: (B, N_visible, embed_dim).
                mask: (B, N_patches) — 1=masked, 0=visible.
                ids_restore: (B, N_patches).
            """
            # Patchify + embed
            patches = self.patchify(x)                    # (B, N, patch_dim)
            tokens = self.patch_embed(patches)            # (B, N, embed_dim)
            tokens = tokens + self.pos_embed              # add positional encoding

            # Mask
            visible, mask, ids_restore = self.random_masking(tokens)

            # Encode visible tokens only (the key efficiency trick of MAE)
            for blk in self.encoder_blocks:
                visible = blk(visible)
            latent = self.encoder_norm(visible)

            return latent, mask, ids_restore

        def decode(
            self,
            latent: torch.Tensor,
            ids_restore: torch.Tensor,
        ) -> torch.Tensor:
            """Decode: project → insert mask tokens → Transformer → predict.

            Returns:
                pred: (B, N_patches, patch_dim) — reconstructed patches.
            """
            # Project encoder output to decoder dimension
            x = self.decoder_embed(latent)                # (B, N_vis, dec_dim)

            # Append mask tokens
            B, N_vis, D = x.shape
            n_mask = self.n_patches - N_vis
            mask_tokens = self.mask_token.expand(B, n_mask, -1)
            x = torch.cat([x, mask_tokens], dim=1)       # (B, N, dec_dim)

            # Unshuffle to original order
            x = torch.gather(
                x, dim=1,
                index=ids_restore.unsqueeze(-1).expand(-1, -1, D),
            )

            # Add decoder positional encoding
            x = x + self.decoder_pos_embed

            # Decode
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # Predict patch pixels
            pred = self.decoder_pred(x)                   # (B, N, patch_dim)
            return pred

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Full forward pass: encode → decode → loss.

            Args:
                x: (B, C, T) input EEG signals.

            Returns:
                loss: scalar MSE loss on masked patches.
                pred: (B, N, patch_dim) predictions.
                mask: (B, N) binary mask (1=masked).
            """
            latent, mask, ids_restore = self.encode(x)
            pred = self.decode(latent, ids_restore)

            # Compute loss on masked patches only
            target = self.patchify(x)
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)           # per-patch MSE: (B, N)
            loss = (loss * mask).sum() / mask.sum()  # mean over masked patches

            return loss, pred, mask

        def encode_features(self, x: torch.Tensor) -> torch.Tensor:
            """Extract features for downstream tasks (no masking).

            Args:
                x: (B, C, T) input EEG signals.

            Returns:
                features: (B, N_patches, embed_dim) — full sequence encoded.
            """
            patches = self.patchify(x)
            tokens = self.patch_embed(patches) + self.pos_embed
            for blk in self.encoder_blocks:
                tokens = blk(tokens)
            return self.encoder_norm(tokens)

    return EEGMAE()


def _sinusoidal_position_encoding(n_pos: int, d_model: int) -> np.ndarray:
    """Generate sinusoidal positional encoding (Vaswani et al., 2017)."""
    pe = np.zeros((n_pos, d_model), dtype=np.float32)
    position = np.arange(n_pos)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[: d_model // 2])
    return pe


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Data Loading: Single-Dataset (quickload) vs. Multi-Dataset
# ═══════════════════════════════════════════════════════════════════════════

# Standard 10-20 motor cortex channels shared by BCI IV 2a and PhysioNet MI.
# NeuroAtom's ChannelMapper maps each dataset's native layout to this set,
# zero-filling any channels that don't exist in a particular dataset.
COMMON_CHANNELS_22 = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
    "FC1", "FC2", "CP1",
]


def load_single_dataset(
    dataset: str,
    data_path: str,
    subject: str,
    batch_size: int = 32,
    n_samples: int = 512,
    sampling_rate: float = 128.0,
):
    """Load a single dataset using NeuroAtom's quickload API.

    This is the simplest path: 5 lines to a DataLoader.
    """
    import neuroatom as na

    logger.info("=" * 60)
    logger.info("Loading single dataset: %s", dataset)
    logger.info("=" * 60)

    loader = na.quickload(
        dataset,
        data_path=data_path,
        subject=subject,
        batch_size=batch_size,
        band=(0.5, 40.0),
    )
    return loader


def load_multi_dataset(
    bci_dir: str,
    physionet_dir: str,
    pool_dir: Optional[str] = None,
    batch_size: int = 32,
    n_channels: int = 22,
    n_samples: int = 512,
    sampling_rate: float = 128.0,
):
    """Load heterogeneous datasets into a single training DataLoader.

    Demonstrates NeuroAtom's core value: BCI IV 2a (25 ch @ 250 Hz .mat)
    and PhysioNet MI (64 ch @ 160 Hz .edf) → unified (B, 22, 512) tensors.
    """
    from neuroatom import (
        Pool, Indexer, TaskConfig,
        DatasetAssembler, AssemblyRecipe, LabelSpec,
    )
    from neuroatom.loader.torch_dataset import AtomDataset
    import torch
    from torch.utils.data import DataLoader

    logger.info("=" * 60)
    logger.info("Multi-dataset heterogeneous pre-training")
    logger.info("  BCI IV 2a  :  25 ch @ 250 Hz  (.mat)")
    logger.info("  PhysioNet  :  64 ch @ 160 Hz  (.edf)")
    logger.info("  Output     :  %d ch @ %d Hz   (unified)", n_channels, int(sampling_rate))
    logger.info("=" * 60)

    # ── 1. Create a shared pool ──────────────────────────────────────────
    if pool_dir is None:
        pool_dir = tempfile.mkdtemp(prefix="neuroatom_mae_multi_")
    pool_path = Path(pool_dir)

    if (pool_path / "pool.json").exists():
        pool = Pool.open(pool_path)
        logger.info("Opened existing pool at %s", pool_path)
    else:
        pool = Pool.create(pool_path)
        logger.info("Created new pool at %s", pool_path)

    indexer = Indexer(pool)

    # ── 2. Import BCI IV 2a subjects ─────────────────────────────────────
    bci_path = Path(bci_dir)
    bci_files = sorted(bci_path.glob("A0*T.mat"))
    if not bci_files:
        logger.warning("No BCI IV 2a .mat files found in %s", bci_dir)
    else:
        bci_config = TaskConfig.builtin("bci_comp_iv_2a")
        from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
        bci_importer = BCICompIV2aImporter(pool, bci_config)

        for mat_file in bci_files[:3]:  # First 3 subjects for demo
            subject_id = mat_file.stem[:3]  # A01, A02, A03
            logger.info("Importing BCI IV 2a: %s (%s)", mat_file.name, subject_id)
            bci_importer.import_subject(
                mat_path=mat_file,
                subject_id=subject_id,
            )

    # ── 3. Import PhysioNet MI subjects ──────────────────────────────────
    phys_path = Path(physionet_dir)
    phys_subjects = sorted(phys_path.glob("S*"))
    phys_subjects = [s for s in phys_subjects if s.is_dir()]
    if not phys_subjects:
        logger.warning("No PhysioNet subject directories found in %s", physionet_dir)
    else:
        phys_config = TaskConfig.builtin("physionet_mi")
        from neuroatom.importers.physionet_mi import PhysioNetMIImporter
        phys_importer = PhysioNetMIImporter(pool, phys_config)

        for subj_dir in phys_subjects[:3]:  # First 3 subjects for demo
            subject_id = subj_dir.name
            logger.info("Importing PhysioNet MI: %s", subject_id)
            phys_importer.import_subject(
                subject_dir=subj_dir,
                subject_id=subject_id,
            )

    # ── 4. Index everything ──────────────────────────────────────────────
    n_atoms = indexer.reindex_all()
    logger.info("Indexed %d atoms from heterogeneous sources.", n_atoms)

    # ── 5. Assemble with COMMON channel layout ──────────────────────────
    #
    # This is where the magic happens:
    # - ChannelMapper maps both datasets to the 22 common channels
    # - Resampler brings both to 128 Hz
    # - PadCrop ensures uniform temporal length
    # - Unit standardizer converts to µV
    # - Bandpass filter (0.5–40 Hz) applied to all
    #
    target_duration = n_samples / sampling_rate

    recipe = AssemblyRecipe(
        recipe_id="mae_pretrain_multi",
        query={},  # all atoms in pool
        target_channels=COMMON_CHANNELS_22,
        target_sampling_rate=sampling_rate,
        target_duration=target_duration,
        filter_band=(0.5, 40.0),
        target_unit="uV",
        normalization_method="zscore",
        normalization_scope="per_atom",
        label_fields=[
            LabelSpec(annotation_name="mi_class", output_key="mi_class"),
        ],
    )

    logger.info("Assembling unified dataset...")
    result = DatasetAssembler(pool, indexer).assemble(recipe)
    indexer.close()

    all_samples = result.train_samples + result.val_samples + result.test_samples
    logger.info(
        "Assembly complete: %d samples → (%d, %d) per atom",
        len(all_samples), n_channels, n_samples,
    )

    # Log source dataset breakdown
    datasets = {}
    for s in all_samples:
        ds = s.get("dataset_id", "unknown")
        datasets[ds] = datasets.get(ds, 0) + 1
    for ds, count in sorted(datasets.items()):
        logger.info("  %s: %d atoms", ds, count)

    loader = DataLoader(
        AtomDataset(all_samples),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return loader


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Training Loop
# ═══════════════════════════════════════════════════════════════════════════

class TrainingConfig:
    """Training hyperparameters."""

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        batch_size: int = 32,
        patch_size: int = 32,
        mask_ratio: float = 0.75,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        dropout: float = 0.1,
        save_every: int = 10,
        log_every: int = 5,
        checkpoint_dir: str = "checkpoints/eeg_mae",
    ):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_heads = decoder_heads
        self.dropout = dropout
        self.save_every = save_every
        self.log_every = log_every
        self.checkpoint_dir = checkpoint_dir


def cosine_scheduler(
    base_lr: float,
    min_lr: float,
    epochs: int,
    warmup_epochs: int,
) -> List[float]:
    """Cosine annealing schedule with linear warmup."""
    schedule = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        schedule.append(lr)
    return schedule


def train_mae(
    loader,
    n_channels: int,
    n_samples: int,
    config: TrainingConfig,
    device: str = "auto",
):
    """Train the EEG-MAE model.

    Args:
        loader: PyTorch DataLoader yielding {"signal": (B, C, T), ...}.
        n_channels: Number of channels (after channel mapping).
        n_samples: Number of time samples (after pad/crop).
        config: TrainingConfig with hyperparameters.
        device: "auto", "cuda", "cpu", or "mps".
    """
    import torch

    # ── Device selection ─────────────────────────────────────────────────
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    logger.info("Using device: %s", device)

    # ── Build model ──────────────────────────────────────────────────────
    model = build_eeg_mae(
        n_channels=n_channels,
        n_samples=n_samples,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        encoder_depth=config.encoder_depth,
        encoder_heads=config.encoder_heads,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_heads=config.decoder_heads,
        mask_ratio=config.mask_ratio,
        dropout=config.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(
        p.numel() for n, p in model.named_parameters() if "decoder" not in n
    )
    logger.info(
        "EEG-MAE: %s params total (%s encoder, %s decoder)",
        f"{n_params:,}", f"{n_encoder:,}", f"{n_params - n_encoder:,}",
    )

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    lr_schedule = cosine_scheduler(
        config.lr, config.min_lr, config.epochs, config.warmup_epochs,
    )

    # ── Checkpoint directory ─────────────────────────────────────────────
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training log ─────────────────────────────────────────────────────
    history = {
        "epoch": [],
        "loss": [],
        "lr": [],
        "time": [],
    }

    # ── Training loop ────────────────────────────────────────────────────
    logger.info("Starting MAE pre-training for %d epochs...", config.epochs)
    logger.info(
        "  Patch size: %d samples, Mask ratio: %.0f%%, Patches/trial: %d",
        config.patch_size,
        config.mask_ratio * 100,
        n_samples // config.patch_size,
    )

    best_loss = float("inf")
    total_start = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()
        model.train()

        # Set learning rate
        lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            signal = batch["signal"].to(device)  # (B, C, T)

            # Ensure correct shape (skip malformed batches)
            if signal.shape[1] != n_channels or signal.shape[2] != n_samples:
                logger.warning(
                    "Skipping batch with shape %s (expected [*, %d, %d])",
                    signal.shape, n_channels, n_samples,
                )
                continue

            loss, pred, mask = model(signal)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            logger.warning("Epoch %d: no valid batches!", epoch + 1)
            continue

        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["lr"].append(lr)
        history["time"].append(epoch_time)

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Logging
        if (epoch + 1) % config.log_every == 0 or epoch == 0:
            logger.info(
                "Epoch %3d/%d  |  loss: %.6f  |  best: %.6f  |  "
                "lr: %.2e  |  %.1fs",
                epoch + 1, config.epochs, avg_loss, best_loss, lr, epoch_time,
            )

        # Checkpointing
        if (epoch + 1) % config.save_every == 0:
            ckpt_path = ckpt_dir / f"eeg_mae_epoch_{epoch + 1:04d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": {
                    "n_channels": n_channels,
                    "n_samples": n_samples,
                    "patch_size": config.patch_size,
                    "embed_dim": config.embed_dim,
                    "encoder_depth": config.encoder_depth,
                    "encoder_heads": config.encoder_heads,
                    "decoder_embed_dim": config.decoder_embed_dim,
                    "decoder_depth": config.decoder_depth,
                    "decoder_heads": config.decoder_heads,
                    "mask_ratio": config.mask_ratio,
                },
            }, ckpt_path)
            logger.info("  → Saved checkpoint: %s", ckpt_path.name)

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(
        "Training complete! %d epochs in %.1f min  |  best loss: %.6f",
        config.epochs, total_time / 60, best_loss,
    )

    # Save final model (encoder only — for downstream transfer)
    final_path = ckpt_dir / "eeg_mae_encoder_final.pt"
    encoder_state = {
        k: v for k, v in model.state_dict().items()
        if "decoder" not in k
    }
    torch.save({
        "encoder_state_dict": encoder_state,
        "config": {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "patch_size": config.patch_size,
            "embed_dim": config.embed_dim,
            "encoder_depth": config.encoder_depth,
            "encoder_heads": config.encoder_heads,
        },
    }, final_path)
    logger.info("Saved encoder weights: %s", final_path)

    # Save training history
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history: %s", history_path)

    # ── Reconstruction quality summary ───────────────────────────────────
    model.eval()
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            if signal.shape[1] == n_channels and signal.shape[2] == n_samples:
                loss, pred, mask = model(signal)
                target = model.patchify(signal)

                # Compute per-patch reconstruction metrics
                masked_pred = pred[mask.bool()]
                masked_target = target[mask.bool()]
                mse = ((masked_pred - masked_target) ** 2).mean().item()
                # Correlation
                pred_flat = masked_pred.reshape(-1).cpu().numpy()
                target_flat = masked_target.reshape(-1).cpu().numpy()
                if np.std(pred_flat) > 1e-10 and np.std(target_flat) > 1e-10:
                    corr = np.corrcoef(pred_flat, target_flat)[0, 1]
                else:
                    corr = 0.0  # degenerate case (constant signal)

                logger.info("Final reconstruction quality:")
                logger.info("  MSE  (masked patches): %.6f", mse)
                logger.info("  Corr (masked patches): %.4f", corr)
                break

    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG-MAE: Self-supervised pre-training with NeuroAtom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset (quickload):
  python pretrain_eeg_mae.py --data-dir C:/Data/BCI_Competition

  # Multi-dataset heterogeneous:
  python pretrain_eeg_mae.py --multi \\
      --bci-dir C:/Data/BCI_Competition \\
      --physionet-dir C:/Data/Physionet

  # Quick test run (3 epochs):
  python pretrain_eeg_mae.py --data-dir C:/Data/BCI_Competition --epochs 3
        """,
    )

    # Mode selection
    parser.add_argument(
        "--multi", action="store_true",
        help="Enable multi-dataset heterogeneous pre-training mode.",
    )

    # Single-dataset options
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory for single-dataset mode (default: BCI IV 2a).",
    )
    parser.add_argument(
        "--dataset", type=str, default="bci_comp_iv_2a",
        help="Dataset name for single-dataset mode.",
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Subject ID for single-dataset mode.",
    )

    # Multi-dataset options
    parser.add_argument(
        "--bci-dir", type=str, default=None,
        help="BCI IV 2a data directory (multi-dataset mode).",
    )
    parser.add_argument(
        "--physionet-dir", type=str, default=None,
        help="PhysioNet MI data directory (multi-dataset mode).",
    )
    parser.add_argument(
        "--pool-dir", type=str, default=None,
        help="Pool directory for multi-dataset mode.",
    )

    # Model architecture
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/eeg_mae")

    # Signal processing
    parser.add_argument(
        "--n-channels", type=int, default=22,
        help="Number of channels (multi-dataset mode uses COMMON_CHANNELS_22).",
    )
    parser.add_argument(
        "--sampling-rate", type=float, default=128.0,
        help="Target sampling rate in Hz.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=512,
        help="Target number of time samples per trial (= duration × sampling_rate).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate args
    if args.multi:
        if not args.bci_dir and not args.physionet_dir:
            logger.error("--multi requires at least one of --bci-dir or --physionet-dir")
            sys.exit(1)
    else:
        if not args.data_dir:
            # Try environment variable
            args.data_dir = os.environ.get("NEUROATOM_DATA_DIR")
            if not args.data_dir:
                logger.error(
                    "Provide --data-dir or set NEUROATOM_DATA_DIR. "
                    "Use --multi for heterogeneous pre-training."
                )
                sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────
    n_channels = args.n_channels
    n_samples = args.n_samples

    if args.multi:
        loader = load_multi_dataset(
            bci_dir=args.bci_dir or "",
            physionet_dir=args.physionet_dir or "",
            pool_dir=args.pool_dir,
            batch_size=args.batch_size,
            n_channels=n_channels,
            n_samples=n_samples,
            sampling_rate=args.sampling_rate,
        )
    else:
        # Single dataset — use quickload
        # Detect first .mat file if subject not specified
        data_path = Path(args.data_dir)
        if data_path.is_dir():
            mat_files = sorted(data_path.glob("A0*T.mat"))
            if mat_files:
                data_path = mat_files[0]
                subject = args.subject or data_path.stem[:3]
            else:
                # Generic fallback: first file
                all_files = sorted(data_path.iterdir())
                if all_files:
                    data_path = all_files[0]
                    subject = args.subject or data_path.stem
                else:
                    logger.error("No data files found in %s", args.data_dir)
                    sys.exit(1)
        else:
            subject = args.subject or data_path.stem[:3]

        loader = load_single_dataset(
            dataset=args.dataset,
            data_path=str(data_path),
            subject=subject,
            batch_size=args.batch_size,
            n_samples=n_samples,
            sampling_rate=args.sampling_rate,
        )

        # Infer n_channels from first batch
        first_batch = next(iter(loader))
        n_channels = first_batch["signal"].shape[1]
        n_samples = first_batch["signal"].shape[2]
        logger.info(
            "Inferred from data: %d channels × %d samples",
            n_channels, n_samples,
        )

    # ── Validate patch size ──────────────────────────────────────────────
    if n_samples % args.patch_size != 0:
        old_ps = args.patch_size
        # Find nearest valid patch size
        for ps in [32, 16, 64, 25, 50, 8]:
            if n_samples % ps == 0:
                args.patch_size = ps
                break
        logger.warning(
            "Adjusted patch_size %d → %d (n_samples=%d must be divisible)",
            old_ps, args.patch_size, n_samples,
        )

    # ── Train ────────────────────────────────────────────────────────────
    config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
    )

    model, history = train_mae(
        loader=loader,
        n_channels=n_channels,
        n_samples=n_samples,
        config=config,
        device=args.device,
    )

    logger.info("Done! Encoder weights saved to %s/", args.checkpoint_dir)


if __name__ == "__main__":
    main()
