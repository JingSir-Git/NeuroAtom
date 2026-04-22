# Importer Metadata Audit

## Summary of Gaps Found and Fixed

### 1. KUL AAD (`aad_mat.py`)
- **Missing**: `experiment`, `part`, `repetition` annotations (parsed but not stored)
- **Missing**: `ElectrodeCap` info ('BioSemi64') in custom_fields
- **Fix**: Add all four as annotations/custom_fields

### 2. DTU AAD (`aad_mat.py`)
- Audio envelopes already stored as annotation arrays ✓
- EXG channel types already handled ✓
- **Mostly complete**

### 3. BCI IV 2a (`bci_comp_iv_2a.py`)
- Class labels ✓, artifact flags ✓, age/gender ✓, calibration runs ✓
- **Missing**: `session_type` annotation (Training 'T' vs Evaluation 'E')
- **Fix**: Auto-detect T/E from filename, store as annotation

### 4. PhysioNet MI (`physionet_mi.py`)
- Run-dependent semantics ✓, paradigm ✓, channel cleanup ✓
- **Missing**: Baseline runs (R01 eyes-open, R02 eyes-closed) — currently SKIPPED entirely
- **Missing**: Standard 10-20 electrode coordinates (available in `standard_1020.json`)
- **Fix**: Import baseline runs as CONTINUOUS_SEGMENT; add electrode coords from standard montage

### 5. SEED-V (`seed_v.py`)
- Emotion labels ✓, session info ✓, trial segmentation ✓
- **Missing**: VEO/HEO stored with proper ChannelType.EOG (currently excluded entirely)
- **Missing**: Standard 10-20 electrode coordinates
- **Missing**: Video clip / stimuli info per trial
- **Fix**: Include VEO/HEO as EOG; add electrode coords; add video_clip annotation

### 6. Zuco 2.0 (`zuco2.py`)
- 3D electrode coords ✓, sentence epochs ✓, text_id ✓
- **Missing**: Word boundary metadata from `wordbounds_TSRx.mat`
- **Missing**: Automagic quality scores (`automagic.qualityScores`, `automagic.finalBadChans`)
- **Missing**: Reference type from `EEG.ref`
- **Fix**: Load wordbounds, integrate quality scores, store reference

### 7. CCEP (`ccep_bids_npy.py`)
- Already fully audited: electrode coords, filter settings, reference type, material ✓

### 8. Generic importers (mat.py, bids.py, eeglab.py, mne_generic.py, moabb_bridge.py)
- These are format-agnostic utilities. Not dataset-specific. No audit needed.
