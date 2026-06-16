#!/usr/bin/env python
"""AAD Cross-Subject MLP Baseline — NeuroAtom End-to-End Validation.

Trains a simple MLP on KUL Auditory Attention Detection data with
**leave-subjects-out** cross-subject evaluation, proving the full
NeuroAtom pipeline (Import → Index → Assemble → DataLoader → Train)
produces genuinely usable ML-ready data.

KUL task: binary classification — attended_ear ∈ {L, R}
 - 16 subjects × 20 trials, 64 ch @ 128 Hz, ~60 s/trial
 - We use 5 trials per subject (speed), 2-second windows, resample to 64 Hz
 - Leave 3 subjects for test, 2 for val, 11 for train

Expected output:
  - Training loss curve converges
  - Test accuracy well above chance (50%) → proves data is meaningful
  - No subject leakage between splits

Usage:
    python examples/aad_mlp_cross_subject.py
    python examples/aad_mlp_cross_subject.py --trials 10 --epochs 30
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════
# 0. Args & config
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="AAD cross-subject MLP baseline")
    p.add_argument("--kul-dir", type=str,
                   default=os.environ.get("NEUROATOM_KUL_DIR", r"C:\Data\KUL"),
                   help="KUL data directory containing S1.mat ... S16.mat")
    p.add_argument("--trials", type=int, default=5,
                   help="Max trials per subject to import (default: 5)")
    p.add_argument("--window", type=float, default=2.0,
                   help="Window duration in seconds (default: 2.0)")
    p.add_argument("--srate", type=float, default=64.0,
                   help="Target sampling rate in Hz (default: 64.0)")
    p.add_argument("--epochs", type=int, default=20,
                   help="Training epochs (default: 20)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--hidden", type=int, default=128,
                   help="MLP hidden size (default: 128)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-pool", action="store_true",
                   help="Don't delete temp pool after run")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 1. NeuroAtom pipeline: Import → Index → Assemble → DataLoaders
# ═══════════════════════════════════════════════════════════════════

def build_dataloaders(args):
    """Full NeuroAtom pipeline → (train_loader, val_loader, test_loader, meta)."""
    import torch
    from torch.utils.data import DataLoader

    from neuroatom.importers.aad_mat import AADImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.storage.pool import Pool
    from neuroatom.index.indexer import Indexer
    from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
    from neuroatom.core.enums import (
        NormalizationMethod, NormalizationScope, SplitStrategy,
    )
    from neuroatom.assembler.dataset_assembler import DatasetAssembler
    from neuroatom.loader.torch_dataset import AtomDataset

    kul_dir = Path(args.kul_dir)
    mats = sorted(kul_dir.glob("S*.mat"))
    if len(mats) < 16:
        print(f"ERROR: Found only {len(mats)} .mat files in {kul_dir}. Need 16.")
        sys.exit(1)

    # ── 1a. Create pool ──────────────────────────────────────────
    pool_dir = Path(tempfile.mkdtemp(prefix="neuroatom_aad_mlp_"))
    pool = Pool.create(pool_dir)
    print(f"  Pool: {pool_dir}")

    # ── 1b. Import all 16 subjects ───────────────────────────────
    config = TaskConfig.builtin("kul_aad")
    importer = AADImporter(pool, config)

    t0 = time.time()
    for mat_path in mats:
        subject_id = mat_path.stem
        importer.import_subject(
            mat_path=mat_path,
            subject_id=subject_id,
            session_id="ses-01",
            format_hint="kul",
            max_trials=args.trials,
        )
    import_time = time.time() - t0
    print(f"  Imported 16 subjects × {args.trials} trials in {import_time:.1f}s")

    # ── 1c. Index ────────────────────────────────────────────────
    indexer = Indexer(pool)
    n_indexed = indexer.reindex_all()
    print(f"  Indexed {n_indexed} atoms")

    # ── 1d. Assemble with cross-subject split ────────────────────
    # 3 test subjects, 2 val, 11 train
    test_subjects = ["kul_aad|S14", "kul_aad|S15", "kul_aad|S16"]
    val_subjects = ["kul_aad|S12", "kul_aad|S13"]

    recipe = AssemblyRecipe(
        recipe_id="aad_mlp_cross_subject",
        query={"dataset_id": "kul_aad"},
        target_sampling_rate=args.srate,
        target_duration=args.window,
        target_unit="uV",
        label_fields=[
            LabelSpec(annotation_name="attended_ear", output_key="label"),
        ],
        split_strategy=SplitStrategy.SUBJECT,
        split_config={
            "test_subjects": test_subjects,
            "val_subjects": val_subjects,
        },
        normalization_method=NormalizationMethod.ZSCORE,
        normalization_scope=NormalizationScope.PER_ATOM,
    )

    t0 = time.time()
    result = DatasetAssembler(pool, indexer).assemble(recipe)
    assemble_time = time.time() - t0
    indexer.close()

    n_train = len(result.train_samples)
    n_val = len(result.val_samples)
    n_test = len(result.test_samples)
    print(f"  Assembly: {n_train} train, {n_val} val, {n_test} test ({assemble_time:.1f}s)")

    # Verify no subject leakage
    train_subs = {s["subject_id"] for s in result.train_samples}
    val_subs = {s["subject_id"] for s in result.val_samples}
    test_subs = {s["subject_id"] for s in result.test_samples}
    assert train_subs.isdisjoint(test_subs), "LEAKAGE: train ∩ test!"
    assert train_subs.isdisjoint(val_subs), "LEAKAGE: train ∩ val!"
    print(f"  ✓ No leakage: train={sorted(train_subs)}")
    print(f"                val  ={sorted(val_subs)}")
    print(f"                test ={sorted(test_subs)}")

    # Show label distribution
    le = result.label_encoder
    print(f"  Label encoding: {le.encodings}")

    train_labels = [s["labels"]["label"] for s in result.train_samples]
    print(f"  Train label dist: {dict(zip(*np.unique(train_labels, return_counts=True)))}")

    # ── 1e. DataLoaders ──────────────────────────────────────────
    train_loader = DataLoader(
        AtomDataset(result.train_samples),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        AtomDataset(result.val_samples),
        batch_size=args.batch_size, shuffle=False,
    ) if result.val_samples else None
    test_loader = DataLoader(
        AtomDataset(result.test_samples),
        batch_size=args.batch_size, shuffle=False,
    ) if result.test_samples else None

    # Get shape info from first sample
    sample = result.train_samples[0]
    n_channels = sample["signal"].shape[0]
    n_times = sample["signal"].shape[1]
    n_classes = len(le.encodings["label"])

    meta = {
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "pool_dir": pool_dir,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }

    return train_loader, val_loader, test_loader, meta


# ═══════════════════════════════════════════════════════════════════
# 2. Simple MLP model
# ═══════════════════════════════════════════════════════════════════

def build_model(n_channels, n_times, n_classes, hidden_size):
    """Simple 2-layer MLP: flatten → FC → ReLU → Dropout → FC → logits."""
    input_dim = n_channels * n_times

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(0.3),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size // 2, n_classes),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MLP: {input_dim} → {hidden_size} → {hidden_size//2} → {n_classes}")
    print(f"  Parameters: {n_params:,}")
    return model


# ═══════════════════════════════════════════════════════════════════
# 3. Training loop
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["labels"]["label"].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_subjects = [], [], []
    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["labels"]["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_subjects.extend(batch["subject_id"])
    return total_loss / total, correct / total, all_preds, all_labels, all_subjects


# ═══════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("=" * 65)
    print("  AAD Cross-Subject MLP Baseline — NeuroAtom Pipeline Validation")
    print("=" * 65)

    # Fix seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Config] device={device}, window={args.window}s, srate={args.srate}Hz, "
          f"trials/subj={args.trials}, epochs={args.epochs}, hidden={args.hidden}")

    # ── Step 1: NeuroAtom pipeline ───────────────────────────────
    print(f"\n[Step 1] NeuroAtom Pipeline: Import → Index → Assemble")
    train_loader, val_loader, test_loader, meta = build_dataloaders(args)

    # ── Step 2: Build MLP ────────────────────────────────────────
    print(f"\n[Step 2] Build MLP")
    model = build_model(
        meta["n_channels"], meta["n_times"],
        meta["n_classes"], args.hidden,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Step 3: Train ────────────────────────────────────────────
    print(f"\n[Step 3] Training ({args.epochs} epochs)")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*7}")

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  {epoch:5d}  {train_loss:10.4f}  {train_acc:8.1%}  {val_loss:8.4f}  {val_acc:7.1%}")

    # ── Step 4: Test ─────────────────────────────────────────────
    print(f"\n[Step 4] Cross-Subject Test Evaluation")

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"  Loaded best model (val acc = {best_val_acc:.1%})")

    if test_loader:
        test_loss, test_acc, preds, labels, subjects = evaluate(
            model, test_loader, criterion, device,
        )

        # Per-subject accuracy
        from collections import defaultdict
        subj_correct = defaultdict(int)
        subj_total = defaultdict(int)
        for p, l, s in zip(preds, labels, subjects):
            subj_total[s] += 1
            subj_correct[s] += int(p == l)

        print(f"\n  {'Subject':>10}  {'Acc':>7}  {'N':>5}")
        print(f"  {'─'*10}  {'─'*7}  {'─'*5}")
        for s in sorted(subj_total.keys()):
            acc = subj_correct[s] / subj_total[s]
            print(f"  {s:>10}  {acc:7.1%}  {subj_total[s]:5d}")

        print(f"\n  ╔═══════════════════════════════════════╗")
        print(f"  ║  Overall Test Accuracy: {test_acc:7.1%}        ║")
        print(f"  ║  Test Loss:             {test_loss:7.4f}        ║")
        print(f"  ║  Chance Level:          {1/meta['n_classes']:7.1%}        ║")
        print(f"  ╚═══════════════════════════════════════╝")

        if test_acc > (1 / meta["n_classes"]):
            print(f"\n  ✓ PASS: Test accuracy ({test_acc:.1%}) > chance ({1/meta['n_classes']:.1%})")
            print(f"    → Pipeline produces meaningful, learnable EEG representations.")
        else:
            print(f"\n  ⚠ Test accuracy at chance — check data or increase training.")
    else:
        print("  No test data available.")

    # ── Cleanup ──────────────────────────────────────────────────
    print(f"\n[Summary]")
    print(f"  Data: 16 subjects × {args.trials} trials, {meta['n_channels']} ch, "
          f"{args.window}s @ {args.srate} Hz")
    print(f"  Split: {meta['n_train']} train / {meta['n_val']} val / {meta['n_test']} test atoms")
    print(f"  Model: MLP ({sum(p.numel() for p in model.parameters()):,} params)")

    pool_dir = meta["pool_dir"]
    if not args.keep_pool:
        shutil.rmtree(pool_dir, ignore_errors=True)
        print(f"  Pool cleaned up.")
    else:
        print(f"  Pool kept at: {pool_dir}")


if __name__ == "__main__":
    main()
