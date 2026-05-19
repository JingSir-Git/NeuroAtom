# Importer Metadata Inventory

Per-dataset checklist of which metadata fields each importer extracts and where they end up in the Atom model. **Status as of 0.2.0** — every row marked ✓ has been verified against the current source.

Verification methodology: each ✓ corresponds to a search match in the named file (`grep` confirms the field is parsed and either stored as an `annotation`, surfaced in `custom_fields`, or used to populate a typed `ChannelInfo` / `QualityInfo` / `RunMeta` slot). Items still labelled ⚠ have no source match and are open work.

## 1. KUL AAD ([aad_mat.py](../neuroatom/importers/aad_mat.py))

| Field | Status | Storage |
|-------|--------|---------|
| Attended ear (`attended_ear`) | ✓ | CategoricalAnnotation |
| Experiment number (1–3) | ✓ | CategoricalAnnotation `name="experiment"` |
| Part index | ✓ | CategoricalAnnotation `name="part"` |
| Repetition (0/1) | ✓ | CategoricalAnnotation `name="repetition"` |
| Electrode cap (`BioSemi64`) | ✓ | `custom_fields["electrode_cap"]` + RunMeta paradigm_details |
| Audio envelopes per trial | ✓ | Annotation arrays in HDF5 shard |

## 2. DTU AAD ([aad_mat.py](../neuroatom/importers/aad_mat.py))

| Field | Status | Storage |
|-------|--------|---------|
| Attended speaker | ✓ | CategoricalAnnotation |
| Audio envelopes (left/right speaker) | ✓ | Annotation arrays in HDF5 shard |
| EXG channels (typed as EOG) | ✓ | `ChannelType.EOG` |
| Per-trial Hugo/Olivia speaker IDs | ✓ | NumericAnnotation |

## 3. BCI Competition IV 2a ([bci_comp_iv_2a.py](../neuroatom/importers/bci_comp_iv_2a.py))

| Field | Status | Storage |
|-------|--------|---------|
| 4-class MI labels | ✓ | CategoricalAnnotation `mi_class` |
| Artifact flag per trial | ✓ | CategoricalAnnotation `artifact` + `QualityInfo.overall_status` |
| Age / gender | ✓ | `custom_fields` |
| Session type (Training T / Evaluation E) | ✓ | CategoricalAnnotation `session_type`, auto-detected from filename suffix |
| Calibration runs (eyes-open / closed) | ✓ | `AtomType.CONTINUOUS_SEGMENT` |
| EOG channels (typed) | ✓ | `ChannelType.EOG` |

## 4. PhysioNet MI ([physionet_mi.py](../neuroatom/importers/physionet_mi.py))

| Field | Status | Storage |
|-------|--------|---------|
| Run-dependent T1/T2 semantics | ✓ | Resolved by run number → `mi_class` |
| Paradigm (execution vs imagery) | ✓ | CategoricalAnnotation `paradigm` |
| Channel name cleanup (`Fc5.` → `FC5`) | ✓ | `standardize_channel_name` |
| Baseline runs R01 (eyes open) / R02 (eyes closed) | ✓ | `AtomType.CONTINUOUS_SEGMENT` with rest labels |
| Standard 10-20 electrode coordinates | ✓ | `ChannelInfo.electrode` from `configs/standard_1020.json` |
| EDF event timing | ✓ | `events` array |

## 5. SEED-V ([seed_v.py](../neuroatom/importers/seed_v.py))

| Field | Status | Storage |
|-------|--------|---------|
| 5-emotion labels | ✓ | CategoricalAnnotation `emotion` |
| Per-session segmentation (3 sessions × 15 trials) | ✓ | session_id structure |
| VEO / HEO as EOG channels | ✓ | `ChannelType.EOG` (channel-type override map) |
| Within-session stimulus ordinal | ✓ | NumericAnnotation `stimulus_index` |
| Per-clip stimulus ID (from xlsx) | ✓ | CategoricalAnnotation `stimulus_id` when YAML `stimulus_order` populated |
| Standard 10-20 coords for 62 ch montage | ✓ | `ChannelInfo.location` from `configs/standard_1020.json` (capitalize-fold lookup) |

## 6. Zuco 2.0 ([zuco2.py](../neuroatom/importers/zuco2.py))

| Field | Status | Storage |
|-------|--------|---------|
| 3D electrode coordinates | ✓ | `ChannelInfo.electrode.position_3d` |
| Sentence epoch boundaries | ✓ | TemporalInfo per atom |
| Text/sentence IDs | ✓ | `sentence_id`, `text_id` annotations |
| Word-boundary metadata (`wordbounds_TSRx.mat`) | ✓ | Custom_fields `wordbounds` array |
| Automagic quality scores | ✓ | `QualityInfo` + `automagic_rate` field |
| Automagic bad-channel list | ✓ | `QualityInfo.bad_channels` |
| EEG reference type | ✓ | RunMeta paradigm_details |

## 7. CCEP-COREG ([ccep_bids_npy.py](../neuroatom/importers/ccep_bids_npy.py))

| Field | Status | Storage |
|-------|--------|---------|
| EEG + sEEG paired modality | ✓ | `Atom.modality`, `AtomRelation(cross_modal_paired_run)` |
| Per-electrode coordinates (BIDS) | ✓ | `ChannelInfo.electrode.position_3d` |
| Filter settings (BIDS sidecar) | ✓ | RunMeta paradigm_details |
| Reference type (BIDS sidecar) | ✓ | RunMeta paradigm_details |
| Stim pair labels | ✓ | CategoricalAnnotation `stim_pair` |
| Electrode material | ✓ | `custom_fields["electrode_material"]` |

## 8. ChineseEEG-2 ([chinese_eeg2.py](../neuroatom/importers/chinese_eeg2.py))

| Field | Status | Storage |
|-------|--------|---------|
| Sentence index | ✓ | `sentence_index` annotation |
| Listening vs reading task config | ✓ | TaskConfig + task_type |
| 128-channel BrainVision data | ✓ | MNE BIDS pipeline |
| Bad-channel JSON sidecar | ✓ | `ChannelInfo.status=BAD` (run-level) + `QualityInfo.bad_channels` per atom |

## 9. OpenBMI MI / ERP / SSVEP ([openbmi.py](../neuroatom/importers/openbmi.py))

| Field | Status | Storage |
|-------|--------|---------|
| MI 2-class labels | ✓ | `mi_class` annotation |
| ERP target / non-target | ✓ | `erp_class` annotation |
| SSVEP frequency labels | ✓ | `ssvep_class` annotation |
| Two-session structure | ✓ | session_id |

## 10. Generic adapters (no dataset-specific audit)

| Importer | Purpose |
|----------|---------|
| [mat.py](../neuroatom/importers/mat.py) | Generic MATLAB struct loader |
| [bids.py](../neuroatom/importers/bids.py) | Auto-traverse BIDS root |
| [eeglab.py](../neuroatom/importers/eeglab.py) | `.set`/`.fdt` via MNE |
| [mne_generic.py](../neuroatom/importers/mne_generic.py) | EDF/BDF/GDF/FIF/CNT/MFF fallback |
| [moabb_bridge.py](../neuroatom/importers/moabb_bridge.py) | 30+ datasets via MOABB |

## Open work

No ⚠ items remain. Everything previously flagged in the 0.1.0 audit has been closed; the audit table above is the current ground truth.
