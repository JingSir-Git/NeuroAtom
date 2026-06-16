"""Debug bandpass filtering in cross-dataset assembly."""
import logging, sys, numpy as np
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(name)s %(message)s")

from neuroatom.quick import multiload

COMMON_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

train, val, test = multiload(
    sources=[
        {
            "dataset": "openbmi_mi",
            "path": r"\\wsqlab\share\JCH\OpenBMI\MI\sess01_subj01_EEG_MI.mat",
            "subjects": ["S01"],
        },
        {
            "dataset": "bci_comp_iv_2a",
            "path": r"C:\Data\BCI_Competition\A01T.mat",
            "subjects": ["A01"],
        },
    ],
    target_channels=COMMON_22,
    target_srate=250,
    target_duration=4.0,
    band=(8.0, 30.0),
    label_field="mi_class",
    split_strategy="stratified",
    split_config={"val_ratio": 0.0, "test_ratio": 0.2, "seed": 42},
)

loader = train or val or test
batch = next(iter(loader))
sig = batch["signal"].numpy()
print(f"\nSignal shape: {sig.shape}")
print(f"Signal stats: mean={sig.mean():.6f}, std={sig.std():.6f}, min={sig.min():.2f}, max={sig.max():.2f}")
print(f"Per-channel std: {np.std(sig, axis=2).mean(axis=0)}")
print(f"Any finite: {np.isfinite(sig).all()}")
print(f"Any nonzero channels: {(np.abs(sig).sum(axis=2) > 0).sum(axis=1)}")
