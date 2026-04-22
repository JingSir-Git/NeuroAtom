# Multi-Modal Assembly Workflow

## Problem

When datasets contain paired multi-modal recordings (e.g., simultaneous scalp EEG + sEEG from CCEP-COREG), the assembler must decide how to handle cross-modal data. Three distinct use cases exist, each with different requirements.

## Design Decision: Three Assembly Modes

### Mode 1: Single-Modality Assembly (default, already supported)

Use the existing `modality` filter in the query to select one modality at a time. Each modality goes through its own pipeline with modality-appropriate parameters.

```yaml
# EEG-only assembly
query:
  dataset_id: "ccepcoreg"
  modality: "eeg"
pipeline:
  - unit_standardize: {target_unit: "uV"}
  - rereference: {method: "average"}
  - channel_map: {target_montage: "standard_1020", n_channels: 64}
  - bandpass_filter: {l_freq: 0.5, h_freq: 45.0}
```

```yaml
# iEEG-only assembly (separate recipe)
query:
  dataset_id: "ccepcoreg"
  modality: "ieeg"
pipeline:
  - unit_standardize: {target_unit: "uV"}
  # No rereference (already bipolar)
  # No channel_map (patient-specific, skip or use ROI mapping)
  - bandpass_filter: {l_freq: 0.5, h_freq: 300.0}
```

**Status**: Fully supported today. No changes needed.

### Mode 2: Paired Multi-Modal Assembly (future)

For models that consume paired EEG + iEEG simultaneously (e.g., transfer learning, multi-view learning), we need to:

1. Query atoms from both modalities
2. Group them by `AtomRelation(cross_modal_paired_run)`
3. Apply separate pipelines per modality
4. Return paired tuples: `(eeg_tensor, ieeg_tensor, label)`

**Proposed recipe syntax** (not yet implemented):

```yaml
multi_modal:
  modalities:
    eeg:
      query: {modality: "eeg"}
      pipeline:
        - unit_standardize: {target_unit: "uV"}
        - channel_map: {target_montage: "standard_1020"}
        - bandpass_filter: {l_freq: 0.5, h_freq: 45.0}
    ieeg:
      query: {modality: "ieeg"}
      pipeline:
        - unit_standardize: {target_unit: "uV"}
        - bandpass_filter: {l_freq: 0.5, h_freq: 300.0}

  pairing:
    strategy: "run_level"
    # Each sample = (random EEG epoch from run, random iEEG epoch from run)
    # Because epoch counts differ per modality per run
```

**Why run-level pairing, not epoch-level**:
- CCEP data has independently artifact-rejected epochs → different counts per modality
- Most multi-modal datasets don't have strict 1:1 epoch correspondence
- Run-level pairing preserves the experimental context while allowing flexible sampling

**Implementation plan** (Phase 4+):
1. Add `MultiModalAssembler` that delegates to per-modality `DatasetAssembler` instances
2. Add `PairedDataset` that yields `Dict[modality, tensor]` per sample
3. Pairing uses `AtomRelation` to find matching runs, then samples independently

### Mode 3: Cross-Modal Feature Fusion (research frontier)

More advanced: combine features from both modalities into a single representation.
This is model-specific and should not be in the assembly pipeline. Instead:

- Assemble each modality separately (Mode 1)
- Use a custom PyTorch `Dataset` that loads both caches
- Let the model architecture handle fusion

## Current Recommendation

For now (v0.1), use **Mode 1** — assemble each modality separately with its own recipe. This is:
- Already fully functional
- Sufficient for most current research
- Clean separation of concerns

When Mode 2 is needed, the `AtomRelation` infrastructure is already in place to support it. The main engineering work is in the assembler and dataset classes.

## Channel Mapping Challenges for iEEG

Unlike scalp EEG with standardized 10-20 positions, intracranial electrodes are patient-specific. Cross-subject assembly options:

1. **ROI-based mapping**: Group bipolar channels by anatomical ROI (requires atlas + electrode coordinates)
2. **Skip channel mapping**: Keep patient-specific channels, use models that handle variable input
3. **Functional connectivity features**: Extract connectivity metrics instead of raw signals

All three approaches are compatible with the current atom model — electrode coordinates and anatomical metadata are stored in `ChannelInfo.location` and `custom_fields`.

## Summary

| Mode | Status | When to Use |
|------|--------|-------------|
| Single-modality | ✅ Supported | Most use cases: train on EEG or iEEG independently |
| Paired multi-modal | 🔧 Planned | Multi-view models, transfer learning |
| Cross-modal fusion | 📋 Design only | Research-specific, model-dependent |
