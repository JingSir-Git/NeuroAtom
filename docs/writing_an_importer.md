# Writing a New Importer

Add a new EEG dataset to NeuroAtom in 2 files. No edits to the core library.

## The contract

A NeuroAtom importer subclasses `BaseImporter` and implements **4 abstract methods**:

| Method | Returns | Purpose |
|--------|---------|---------|
| `detect(path)` | `bool` | Recognize files this importer can handle (used by registry). |
| `load_raw(path)` | `(raw, extra_meta)` | Load the recording. `raw` is anything — your code consumes it. |
| `extract_channel_infos(raw)` | `list[ChannelInfo]` | Channel name, type, position, reference. |
| `extract_events(raw)` | `np.ndarray | None` | Shape `(n_events, 3)`: `[sample_idx, prev_id, event_id]`. |

`BaseImporter` provides the **template method** `import_run()` that wires these together: it calls your hooks, asks the atomizer to slice, writes signals to HDF5 shards and metadata to JSONL, registers the run, and acquires the dataset lock. You typically don't override `import_run`.

## Step 1: Write the YAML task config

`neuroatom/importers/task_configs/my_dataset.yaml`

```yaml
# Required
dataset_id: my_dataset
dataset_name: "My Cool BCI Dataset"
task_type: motor_imagery

# Atomizer hints
trial_definition:
  mode: trial          # 'trial' | 'event' | 'window'
  anchor_events: [1, 2, 3, 4]
  tmin: 0.0
  tmax: 4.0
  baseline_tmin: -0.5
  baseline_tmax: 0.0

# Event code → human-readable label
event_mapping:
  1: "left_hand"
  2: "right_hand"
  3: "feet"
  4: "tongue"

# Optional metadata
signal_unit: uV
exclude_channels: []
channel_type_overrides:
  "EOG": eog

# ⚡ quickload metadata: enables neuroatom.quickload() to drive your importer
# without any Python edits in quick.py.
quickload:
  label_field: mi_class          # primary annotation used as the training label
  data_path_kwarg: mat_path      # name of the keyword arg your import_subject() expects
  entry_method: import_subject   # 'import_subject' (single subject) or 'import_dataset' (BIDS-style)
  subject_pattern: '^(S\d{2})'   # regex on filename stem; group(1) → subject_id
  aliases: [my_ds, mydataset]    # alternate names users may pass to quickload()
```

## Step 2: Write the Python importer

`neuroatom/importers/my_dataset.py`

```python
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.atomizer.trial import TrialAtomizer
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.enums import ChannelType
from neuroatom.core.subject import SubjectMeta
from neuroatom.core.session import SessionMeta
from neuroatom.importers.base import BaseImporter


class MyDatasetImporter(BaseImporter):
    """Importer for My Cool BCI Dataset."""

    # ── 1. Detection ─────────────────────────────────────────────────────
    @staticmethod
    def detect(path: Path) -> bool:
        # Return True for files/dirs your importer can handle.
        return path.suffix == ".mat" and path.stem.startswith("S")

    # ── 2. Raw loader ────────────────────────────────────────────────────
    def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        import scipy.io as sio
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)

        # Return whatever your other hooks need. `extra_meta` is folded into
        # RunMeta.paradigm_details.
        return mat, {"paradigm_details": {"source": path.name}}

    # ── 3. Channels ──────────────────────────────────────────────────────
    def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
        names = list(raw["channel_names"])
        return [
            ChannelInfo(
                channel_id=name,
                name=name,
                channel_type=ChannelType.EEG,
                index=idx,
            )
            for idx, name in enumerate(names)
        ]

    # ── 4. Events ────────────────────────────────────────────────────────
    def extract_events(self, raw: Any) -> Optional[np.ndarray]:
        # Build (n_events, 3) array: [sample, prev_id (unused, 0), event_id]
        onsets = raw["onsets"]       # (n_trials,)
        labels = raw["labels"]       # (n_trials,)
        events = np.column_stack([
            onsets.astype(int),
            np.zeros(len(onsets), dtype=int),
            labels.astype(int),
        ])
        return events

    # ── 5. Top-level entry point (called by quickload) ───────────────────
    def import_subject(self, *, mat_path: Path, subject_id: str, **kwargs):
        """Import a single subject's file.

        Arg name `mat_path` must match `quickload.data_path_kwarg` in YAML.
        """
        # Register subject + session (idempotent helpers on Pool)
        self._pool.ensure_subject(self._task_config.dataset_id, subject_id)
        session_id = "ses-01"
        self._pool.ensure_session(
            self._task_config.dataset_id,
            subject_id,
            session_id,
            sampling_rate=250.0,
        )

        # Delegate to BaseImporter.import_run for the heavy lifting.
        return self.import_run(
            path=mat_path,
            subject_id=subject_id,
            session_id=session_id,
            run_id="run-01",
            atomizer=TrialAtomizer(),
            run_index=0,
        )

    # ── 6. (Optional) Override signal extraction ─────────────────────────
    # Default uses MNE Raw.get_data(). Override if your `raw` isn't MNE.
    def _extract_atom_signal(self, raw, atom, channel_infos) -> np.ndarray:
        start = atom.temporal.onset_sample
        stop = start + atom.temporal.duration_samples
        # Your raw is a dict from scipy.io.loadmat
        data = raw["signal"][:, start:stop]
        return data.astype(np.float32)
```

## Step 3: Register the importer

Add one line to `neuroatom/importers/registry.py` to make the importer discoverable via name:

```python
_IMPORTER_REGISTRY: Dict[str, Tuple[str, str]] = {
    # ...
    "my_dataset": ("neuroatom.importers.my_dataset", "MyDatasetImporter"),
}
```

That's it. From a user's perspective:

```python
import neuroatom as na

loader = na.quickload(
    "my_dataset",
    data_path="data/S01.mat",
    batch_size=32,
)
```

The subject is auto-inferred from the YAML's `subject_pattern` regex. The label field comes from `quickload.label_field`. No edits to `quick.py`.

## Atomizer cheat sheet

Pick the right atomizer based on your data's structure:

| Atomizer | Use when |
|----------|----------|
| `TrialAtomizer` | Cued trials with discrete event codes (motor imagery, P300, SSVEP). Reads `trial_definition.anchor_events` + `tmin`/`tmax`. |
| `EventAtomizer` | Event-related potentials with continuous baseline before each event. |
| `WindowAtomizer` | Continuous recordings (resting-state, sleep). Set `mode: window` + `window_seconds` + `stride_seconds`. |

## Common pitfalls

| Symptom | Likely cause |
|---------|--------------|
| `0 atoms indexed` after import | `register_run` not called, or events array is empty. Make sure your importer creates run.json. |
| `KeyError: 'mi_class'` in assembly | Your annotations don't include the `label_field` declared in YAML. Add it inside the atomizer or override `extract_events` to attach annotations. |
| Signals look ~1e-6× too small | Source data was in V but `signal_unit: V` not declared; the unit standardizer treated it as already-µV. Set `signal_unit: V` in YAML. |
| Multibyte channel names corrupt | Don't open JSONL/text files with default encoding on Windows; the framework writes UTF-8 — your import code should too. |
| `RuntimeError: deadlock` on parallel import | You're acquiring `pool.dataset_lock` manually inside `import_run`. Don't — `BaseImporter.import_run` already does it. |

## Tests for your importer

Two recommended test patterns:

```python
# tests/test_e2e_my_dataset.py — gated on real data
def test_my_dataset_import(my_data_dir, tmp_path):
    """E2E: real .mat → pool → query → assembled tensor."""
    pool = Pool.create(tmp_path / "p")
    config = TaskConfig.builtin("my_dataset")
    importer = MyDatasetImporter(pool, config)
    importer.import_subject(mat_path=my_data_dir / "S01.mat", subject_id="S01")
    # ... assertions ...
```

Add the corresponding fixture to `tests/conftest.py`:

```python
@pytest.fixture
def my_data_dir():
    d = _data_dir("NEUROATOM_MY_DATASET_DIR")
    if d is None:
        pytest.skip("NEUROATOM_MY_DATASET_DIR not set")
    return d
```

This pattern keeps CI green for contributors without your data (the fixture skips) while letting maintainers run the real test.

## Where to look for reference

- **Smallest importer** — `neuroatom/importers/bci_comp_iv_2a.py`
- **Multi-modal (EEG + sEEG)** — `neuroatom/importers/ccep_bids_npy.py`
- **MNE generic fallback** — `neuroatom/importers/mne_generic.py`
- **MOABB bridge (wraps another library)** — `neuroatom/importers/moabb_bridge.py`
