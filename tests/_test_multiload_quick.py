"""Quick smoke test for multiload cross-dataset scenario."""
import logging
import sys

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(name)s %(message)s",
)

import neuroatom as na

COMMON_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

train, val, test = na.multiload(
    sources=[
        # OpenBMI MI: 62ch @ 1000Hz, 2-class
        {
            "dataset": "openbmi_mi",
            "path": r"\\wsqlab\share\JCH\OpenBMI\MI\sess01_subj01_EEG_MI.mat",
            "subjects": ["S01"],
            "import_kwargs": {"max_trials": 20},
        },
        # BCI IV 2a: 25ch @ 250Hz, 4-class (only left/right overlap with OpenBMI)
        {
            "dataset": "bci_comp_iv_2a",
            "path": r"C:\Data\BCI_Competition\A01T.mat",
            "subjects": ["A01"],
        },
    ],
    target_channels=COMMON_22,
    target_srate=250,
    target_duration=4.0,
    label_field="mi_class",
    # With only 2 subjects, subject split can't fill 3 partitions.
    # Use stratified to get a proper train/test split at trial level.
    split_strategy="stratified",
    split_config={"val_ratio": 0.0, "test_ratio": 0.2, "seed": 42},
)

n_train = len(train.dataset) if train else 0
n_val = len(val.dataset) if val else 0
n_test = len(test.dataset) if test else 0

print(f"\n{'='*60}")
print("CROSS-DATASET ASSEMBLY RESULT:")
print(f"  Sources:")
print(f"    - OpenBMI MI : 62ch @ 1000Hz → unified 22ch @ 250Hz")
print(f"    - BCI IV 2a  : 25ch @ 250Hz  → unified 22ch @ 250Hz")
print(f"  Train: {n_train} samples")
print(f"  Val:   {n_val} samples")
print(f"  Test:  {n_test} samples")
print(f"  Total: {n_train + n_val + n_test} atoms")

# Grab a batch from whichever loader is non-empty
loader = train or val or test
assert loader is not None, "All loaders are None — something is wrong."

batch = next(iter(loader))
sig = batch["signal"]
print(f"\n  Batch signal shape: {sig.shape}")
print(f"    → (batch={sig.shape[0]}, "
      f"channels={sig.shape[1]} [{COMMON_22[0]}..{COMMON_22[-1]}], "
      f"samples={sig.shape[2]} [{int(250*4)}=4s@250Hz])")
print(f"  Label keys: {list(batch['labels'].keys())}")
print(f"  Dataset IDs in batch: {set(batch['dataset_id'])}")
print(f"  Subject IDs in batch: {set(batch['subject_id'])}")

# Verify shape invariants
assert sig.shape[1] == len(COMMON_22), f"Channel count mismatch: {sig.shape[1]} vs {len(COMMON_22)}"
assert sig.shape[2] == 1000, f"Sample count mismatch: {sig.shape[2]} vs 1000 (4s × 250Hz)"

print(f"{'='*60}")
print("✓ multiload cross-dataset test PASSED!")
