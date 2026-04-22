"""Train a simple EEGNet on BCI Competition IV 2a data using NeuroAtom.

Demonstrates the complete workflow: Import → Index → Query → Assemble → Train.
This is NOT a benchmark — it shows that NeuroAtom output plugs into standard
PyTorch training with zero friction.

Usage:
    python examples/train_simple_cnn.py --data-dir /path/to/bci_data

    # Or set via environment variable:
    NEUROATOM_BCI_IV_2A_DIR=/path/to/data python examples/train_simple_cnn.py

Prerequisites:
    - BCI IV 2a .mat files (A01T.mat, A02T.mat, ...) in the data directory
    - pip install neuroatom[torch,mne]
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_eegnet")


# ═══════════════════════════════════════════════════════════════════════════
# EEGNet — compact CNN for EEG classification (Lawhern et al., 2018)
# ═══════════════════════════════════════════════════════════════════════════

def build_eegnet(n_channels: int, n_samples: int, n_classes: int):
    """Build EEGNet model.

    Architecture:
        Conv2D (temporal) → BatchNorm → DepthwiseConv2D (spatial) →
        BatchNorm → ELU → AvgPool → Dropout →
        SeparableConv2D → BatchNorm → ELU → AvgPool → Dropout →
        Flatten → Dense
    """
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self, C, T, N, F1=8, D=2, F2=16, dropout=0.5):
            super().__init__()
            # Block 1: Temporal convolution + Depthwise spatial
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
                nn.BatchNorm2d(F1),
                # DepthwiseConv2d: groups=F1 for channel-wise spatial filtering
                nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
            )
            # Block 2: Separable convolution
            self.block2 = nn.Sequential(
                # Depthwise
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                          groups=F1 * D, bias=False),
                # Pointwise
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )
            # Classifier
            # Compute flatten size
            with torch.no_grad():
                dummy = torch.zeros(1, 1, C, T)
                out = self.block2(self.block1(dummy))
                flat_size = out.numel()
            self.classifier = nn.Linear(flat_size, N)

        def forward(self, x):
            # x: (B, C, T) → (B, 1, C, T)
            x = x.unsqueeze(1)
            x = self.block1(x)
            x = self.block2(x)
            x = x.flatten(1)
            return self.classifier(x)

    return EEGNet(n_channels, n_samples, n_classes)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGNet with NeuroAtom")
    parser.add_argument(
        "--data-dir", type=str,
        default=os.environ.get("NEUROATOM_BCI_IV_2A_DIR", ""),
        help="Path to BCI IV 2a .mat files (or set NEUROATOM_BCI_IV_2A_DIR)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subjects", nargs="+", default=["A01", "A02", "A03"])
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_dir:
        print("ERROR: --data-dir not specified. Set NEUROATOM_BCI_IV_2A_DIR or use --data-dir.")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist.")
        sys.exit(1)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from neuroatom import (
        Pool, Indexer, DatasetAssembler,
        AssemblyRecipe, LabelSpec,
    )
    from neuroatom.importers.bci_comp_iv_2a import BCICompIV2aImporter
    from neuroatom.importers.base import TaskConfig
    from neuroatom.loader.torch_dataset import AtomDataset
    from neuroatom.core.enums import (
        NormalizationMethod, NormalizationScope, SplitStrategy,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Step 1: Import ────────────────────────────────────────────────────
    pool_dir = Path(tempfile.mkdtemp(prefix="neuroatom_train_"))
    pool = Pool.create(pool_dir)

    task_config_path = (
        Path(__file__).resolve().parent.parent
        / "neuroatom" / "importers" / "task_configs" / "bci_comp_iv_2a.yaml"
    )
    config = TaskConfig.from_yaml(task_config_path)
    importer = BCICompIV2aImporter(pool, config)

    t0 = time.time()
    total_atoms = 0
    for subj in args.subjects:
        mat_path = data_dir / f"{subj}T.mat"
        if not mat_path.exists():
            logger.warning("Skipping %s: %s not found", subj, mat_path)
            continue
        results = importer.import_subject(mat_path=mat_path, subject_id=subj)
        n = sum(len(r.atoms) for r in results)
        total_atoms += n
        logger.info("  %s: %d atoms", subj, n)

    if total_atoms == 0:
        logger.error("No atoms imported — check data directory.")
        sys.exit(1)
    logger.info("Imported %d atoms in %.1fs", total_atoms, time.time() - t0)

    # ── Step 2: Index ─────────────────────────────────────────────────────
    indexer = Indexer(pool)
    indexer.reindex_all()

    # ── Step 3: Assemble ──────────────────────────────────────────────────
    recipe = AssemblyRecipe(
        recipe_id="eegnet_train",
        description="BCI IV 2a 4-class MI for EEGNet training",
        query={
            "dataset_id": "bci_comp_iv_2a",
            "annotations": [{"name": "mi_class"}],
        },
        target_sampling_rate=250.0,
        target_duration=4.0,
        filter_band=(0.5, 40.0),
        target_unit="uV",
        normalization_method=NormalizationMethod.ZSCORE,
        normalization_scope=NormalizationScope.PER_ATOM,
        label_fields=[
            LabelSpec(
                annotation_name="mi_class",
                output_key="mi_class",
                encoding="ordinal",
            ),
        ],
        split_strategy=SplitStrategy.SUBJECT,
        split_config={"val_ratio": 0.0, "test_ratio": 0.5, "seed": 42},
    )

    assembler = DatasetAssembler(pool, indexer)
    result = assembler.assemble(recipe)
    logger.info(
        "Assembled: train=%d, test=%d",
        len(result.train_samples), len(result.test_samples),
    )

    if not result.train_samples or not result.test_samples:
        logger.error("Empty train or test set — need >=2 subjects for split.")
        sys.exit(1)

    # ── Step 4: DataLoaders ───────────────────────────────────────────────
    train_ds = AtomDataset(result.train_samples)
    test_ds = AtomDataset(result.test_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Infer dimensions from first sample
    sample0 = train_ds[0]
    n_channels = sample0["signal"].shape[0]
    n_samples = sample0["signal"].shape[1]
    class_values = sorted(set(
        s["labels"]["mi_class"].item()
        for s in result.train_samples + result.test_samples
        if isinstance(s["labels"]["mi_class"], (int, float, np.integer))
    ))
    n_classes = len(class_values) if class_values else 4
    logger.info("Model input: %d ch x %d samples, %d classes", n_channels, n_samples, n_classes)

    # ── Step 5: Model + Training ──────────────────────────────────────────
    model = build_eegnet(n_channels, n_samples, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    logger.info("Training for %d epochs...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            signals = batch["signal"].to(device)               # (B, C, T)
            labels = batch["labels"]["mi_class"].to(device)    # (B,)

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * signals.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += signals.size(0)

        # ── Eval ──
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                signals = batch["signal"].to(device)
                labels = batch["labels"]["mi_class"].to(device)
                logits = model(signals)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += signals.size(0)

        train_acc = 100.0 * train_correct / max(train_total, 1)
        test_acc = 100.0 * test_correct / max(test_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        logger.info(
            "Epoch %2d/%d  loss=%.4f  train_acc=%.1f%%  test_acc=%.1f%%",
            epoch, args.epochs, avg_loss, train_acc, test_acc,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EEGNet Training with NeuroAtom — Complete")
    print("=" * 60)
    print(f"  Subjects:    {args.subjects}")
    print(f"  Atoms:       {total_atoms}")
    print(f"  Train/Test:  {len(result.train_samples)} / {len(result.test_samples)}")
    print(f"  Model:       EEGNet ({n_channels}ch x {n_samples}t, {n_classes} classes)")
    print(f"  Final test:  {test_acc:.1f}%")
    print(f"  Device:      {device}")
    print("=" * 60)

    indexer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
