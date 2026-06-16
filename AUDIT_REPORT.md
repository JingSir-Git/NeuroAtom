# NeuroAtom Dataset Support Audit Report

> Generated: 2026-04-29 | Scope: All 13 datasets in D:\Data

---

## Executive Summary

After a systematic audit of every dataset in `D:\Data`, I identified **7 major gaps** and **5 framework-level improvements** needed. The most impactful issues are around **multi-modal companion data** (eye-tracking, audio stimuli, text embeddings) that our importers currently discard, and **missing metadata** (questionnaires, comprehension scores, artifact annotations) that could be valuable for downstream ML.

---

## Per-Dataset Audit

### 1. ZuCo 1.0 (`D:\Data\ZuCo_1.0_Full`)

**Current support**: ✅ EEG sentence-level epochs, 105 ch @ 500 Hz
**Gaps found**:

| Gap | Severity | Data Available | Currently Imported |
|-----|----------|---------------|-------------------|
| **Eye-tracking raw data** | 🔴 HIGH | `*_corrected_ET.mat` per subject/text — gaze X/Y, pupil area, saccades, fixations, blinks (time-aligned to EEG) | ❌ Not imported |
| **Word-level eye metrics** | 🔴 HIGH | Per-word: `rawET`, `rawEEG`, `nFixations`, `meanPupilSize`, FFD, TRT, GD, GPT, SFD (standard reading measures) | ❌ Not imported |
| **Word-level band power** | 🟡 MEDIUM | Per-word: `mean_t1/t2/a1/a2/b1/b2/g1/g2` (theta/alpha/beta/gamma band power) | ❌ Not imported |
| **Comprehension answers** | 🟡 MEDIUM | `answers/Fullresults_SR_ZAB_T1.mat` — per-question accuracy | ❌ Not imported |
| **Word boundaries** | 🟢 LOW | `wordbounds` in sentenceData — pixel bounding boxes | Partially (ZuCo2 loads them, ZuCo1 does not) |
| **Sentence content text** | 🟢 LOW | `content` field in sentenceData | ❌ Not stored as TextAnnotation |

### 2. ZuCo 2.0 (`D:\Data\ZuCo_2.0_Full`)

**Current support**: ✅ EEG sentence-level epochs, 105 ch @ 500 Hz
**Same gaps as ZuCo 1.0**, plus:

| Gap | Severity | Notes |
|-----|----------|-------|
| **Eye-tracking files** | 🔴 HIGH | `*_corrected_ET.mat` files present per subject |
| **Word-level metrics** | 🔴 HIGH | Same rich word-level eye+EEG features as ZuCo 1.0 |
| **Only TSR task** | 🟡 MEDIUM | Importer only handles TSR (task2); NR (task1) also available |

### 3. ChineseEEG (`D:\Data\ChineseEEG`)

**Current support**: ✅ EEG sentence-level epochs, 128 ch @ 1000 Hz
**Gaps found**:

| Gap | Severity | Data Available |
|-----|----------|---------------|
| **Eye-tracking data** | 🔴 HIGH | `derivatives/eyetracking_data/sub-XX/ses-*/eyetracking/*.rar` — complete eye-tracking recordings per run |
| **Text embeddings** | 🟡 MEDIUM | `derivatives/text_embeddings/*/text_embedding_run_*.npy` — pre-computed BERT embeddings per run |
| **Novel text** | 🟡 MEDIUM | `derivatives/novels/original_novel/GarnettDream.txt`, `LittlePrince.txt` |
| **Segmented novel** | 🟡 MEDIUM | `derivatives/novels/segmented_novel/*/segmented_*.xlsx` — word/sentence segmentation per run |
| **Preprocessed EEG** | 🟢 LOW | `derivatives/filtered_0.5_30/`, `derivatives/filtered_0.5_80/`, `derivatives/preproc/` — filtered versions |

### 4. ChineseEEG-2 (`D:\Data\ChineseEEG-2`)

**Current support**: ✅ EEG sentence-level epochs (listening + reading), 128 ch
**Gaps found**:

| Gap | Severity | Data Available |
|-----|----------|---------------|
| **Audio embeddings** | 🔴 HIGH | `materials&embeddings/audio_embedding/` — per-speaker audio embeddings (male/female × subject) |
| **Text embeddings** | 🟡 MEDIUM | `materials&embeddings/text_embedding/text_embeddings_*.npy` |
| **Novel text** | 🟡 MEDIUM | `materials&embeddings/original_novel/GarnettDream.txt`, `LittlePrince.txt` |
| **Code/scripts** | 🟢 LOW | `ChineseEEG-2-Code/` — analysis scripts (not for import) |

### 5. ds-eeg-snhl (`D:\Data\ds-eeg-snhl`)

**Current support**: ✅ EEG + EarEEG trials, 72 ch @ 512 Hz
**Gaps found**:

| Gap | Severity | Data Available |
|-----|----------|---------------|
| **Audio stimuli (target/masker)** | 🔴 HIGH | `derivatives/stimuli/sub*/target/*.mat`, `derivatives/stimuli/sub*/masker/*.mat` — per-trial audio envelopes |
| **Tone stimuli task** | 🟡 MEDIUM | `task-tonestimuli` BDF files, channels.tsv, events.tsv — tone audiometry paradigm, not imported |
| **Rest task** | 🟢 LOW | `task-rest` BDF files available — resting-state EEG |

### 6. EEG-iEEG WM (`D:\Data\original`)

**Current support**: ✅ EEG + iEEG paired trials, cross-modal linking
**Gaps found**:

| Gap | Severity | Notes |
|-----|----------|-------|
| **SessionMeta sampling_rate** | 🟡 MEDIUM | Currently hardcoded at 200 Hz; should be read from sidecar JSON per modality (EEG=200, iEEG=2000) |

### 7. KUL AAD (`D:\Data\KUL`)

**Current support**: ✅ EEG trials + audio envelopes via `ContinuousAnnotation`
**Gaps found**:

| Gap | Severity | Data Available |
|-----|----------|---------------|
| **Raw audio WAV files** | 🟡 MEDIUM | `stimuli/part*_track*_{dry,hrtf}.wav` — original audio stimuli (could be stored as StimulusRef) |

### 8. DTU AAD (`D:\Data\DTU`)

**Current support**: ✅ EEG trials + audio envelopes
**Gaps found**:

| Gap | Severity | Data Available |
|-----|----------|---------------|
| **Raw audio WAV files** | 🟡 MEDIUM | `AUDIO/aske_story*_trial_*.wav` — original audio stimuli |

### 9. OpenBMI (`D:\Data\OpenBMI`)

**Current support**: ✅ MI, ERP, SSVEP paradigms
**Gaps found**:

| Gap | Severity | Data Available |
|-----|----------|---------------|
| **Questionnaire data** | 🔴 HIGH | `Questionnaire_results.csv` — age, sex, BCI experience, sleep hours, coffee intake, fatigue ratings per subject |
| **Artifact data** | 🟡 MEDIUM | `Artifact/sess*_subj*_EEG_Artifact.mat` — artifact-flagged epochs |
| **Subject demographics** | 🟡 MEDIUM | Age, sex available in questionnaire but not extracted into SubjectMeta |

### 10. BCI Competition IV 2a (`D:\Data\BCI_Competition` + `D:\Data\BCIC_IV_2a_Raw`)

**Current support**: ✅ 4-class MI from .mat, supports both T and E sets
**Gaps found**:

| Gap | Severity | Data Available |
|-----|----------|---------------|
| **Raw GDF files** | 🟢 LOW | `BCIC_IV_2a_Raw/A0?{T,E}.gdf` — raw continuous recordings. Our importer uses pre-segmented .mat only |

### 11. Physionet MI (`D:\Data\Physionet`)

**Current support**: ✅ 109 subjects, EDF, 64 ch @ 160 Hz
**Gaps**: None significant. Clean import.

### 12. EEG-SEEG / CCEP (`D:\Data\EEG-SEEG`)

**Current support**: ✅ Pre-epoched .npy + cross-modal EEG↔sEEG linking
**Gaps**: None significant. Comprehensive import.

---

## Framework-Level Improvements Needed

### 🔴 GAP 1: No Multi-Modal Companion Data Pipeline

**Problem**: Our `ContinuousAnnotation` type exists but is only used by KUL/DTU AAD (audio envelopes). The ZuCo, ChineseEEG, and SNHL datasets all have rich companion data (eye-tracking, audio stimuli, text embeddings) that we silently discard.

**Impact**: 6 out of 13 datasets have companion data we don't import.

**Proposed fix**:
1. Create a `CompanionDataManager` that mirrors `ShardManager` but writes companion data (eye-tracking, audio, embeddings) into separate HDF5 groups.
2. Extend the `ContinuousAnnotation` pattern to systematically store:
   - Eye-tracking time series (gaze_x, gaze_y, pupil_size) → `ContinuousAnnotation`
   - Audio envelopes/embeddings → `ContinuousAnnotation`
   - Text embeddings → `ContinuousAnnotation` (per-sentence vectors)
3. Use `EventSequenceAnnotation` for fixation/saccade/blink events.
4. Use `StimulusRefAnnotation` to link atoms to external audio WAV files.

### 🔴 GAP 2: Word-Level Features Not Captured (ZuCo)

**Problem**: ZuCo has extremely rich word-level aligned data: per-word EEG, per-word eye-tracking metrics (FFD, TRT, GD, GPT, SFD, nFixations, meanPupilSize), and per-word band power features. These are the core of most ZuCo-based ML papers.

**Impact**: Without these, ZuCo import is only useful for sentence-level tasks — not word-level decoding, which is the main use case.

**Proposed fix**:
1. Add `EventSequenceAnnotation` for word onsets with features dict containing eye-tracking metrics.
2. Store per-word EEG as sub-atoms or as a `ContinuousAnnotation` array.
3. Store per-word band power in `EventItem.features`.

### 🟡 GAP 3: Subject Metadata Underutilized

**Problem**: OpenBMI has rich questionnaire data (age, sex, BCI experience, sleep, coffee, fatigue) that should go into `SubjectMeta.custom_fields`. ChineseEEG has participant demographics. These are never extracted.

**Proposed fix**: Enhance importers to read `participants.tsv`, `Questionnaire_results.csv`, etc. and populate `SubjectMeta` fields.

### 🟡 GAP 4: Derivatives/Preprocessed Versions Not Supported

**Problem**: ChineseEEG has `derivatives/filtered_0.5_30/`, `derivatives/preproc/`. ChineseEEG-2 has `derivatives/`. Users may want to import preprocessed rather than raw data, but our importers only support raw.

**Proposed fix**: Add a `use_preprocessed: bool` parameter to relevant importers (ChineseEEG already has the field but doesn't use it).

### 🟢 GAP 5: Stimulus Resource Management

**Problem**: Our `StimulusRefAnnotation` type exists but no importer actually uses it. KUL/DTU have WAV files, SNHL has `.mat` audio stimuli, ZuCo has text passages — all should be indexed.

**Proposed fix**: Create a `StimulusStore` that copies/links stimulus files into the pool's `stimuli/` directory and generates `StimulusRefAnnotation` entries.

---

## Priority Ranking

| Priority | What | Datasets Affected | Effort |
|----------|------|-------------------|--------|
| 1 | Word-level eye+EEG features for ZuCo 1.0/2.0 | ZuCo 1.0, ZuCo 2.0 | Large |
| 2 | Eye-tracking time series import | ZuCo 1.0/2.0, ChineseEEG | Large |
| 3 | Audio stimuli/envelope import for SNHL | ds-eeg-snhl | Medium |
| 4 | Text/audio embeddings for ChineseEEG/2 | ChineseEEG, ChineseEEG-2 | Medium |
| 5 | OpenBMI questionnaire → SubjectMeta | OpenBMI | Small |
| 6 | Sentence text → TextAnnotation | ZuCo 1.0/2.0 | Small |
| 7 | ZuCo 2.0 NR task support | ZuCo 2.0 | Medium |
| 8 | Preprocessed derivatives support | ChineseEEG | Medium |
| 9 | StimulusRef for audio WAV files | KUL, DTU | Small |
| 10 | Comprehension answers for ZuCo | ZuCo 1.0/2.0 | Small |
