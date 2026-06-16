"""ZuCo Word-Level Feature & Eye-Tracking Helpers.

Parses ZuCo results .mat files (v5 format) to extract:
- Sentence content text
- Per-word eye-tracking reading measures (FFD, TRT, GD, GPT, SFD, nFixations, meanPupilSize)
- Per-word EEG band power features (theta, alpha, beta, gamma)
- Sentence-level fixation data (x, y, duration, pupilsize)
- Raw eye-tracking time series from corrected ET files

Used by both ZuCo 1.0 and ZuCo 2.0 importers to enrich EEG sentence atoms
with multimodal reading annotations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroatom.core.annotation import (
    ContinuousAnnotation,
    EventSequenceAnnotation,
    NumericAnnotation,
    TextAnnotation,
)
from neuroatom.core.annotation import EventItem
from neuroatom.core.signal_ref import SignalRef

logger = logging.getLogger(__name__)

# Standard word-level eye-tracking measures
_WORD_ET_FIELDS = [
    "nFixations",       # Number of fixations on the word
    "meanPupilSize",    # Mean pupil size during word reading
    "FFD",              # First fixation duration (ms)
    "TRT",              # Total reading time (ms)
    "GD",               # Gaze duration (ms)
    "GPT",              # Go-past time (ms)
    "SFD",              # Single fixation duration (ms)
]

# Word-level EEG band power features (8 sub-bands)
_WORD_BAND_FIELDS = [
    "mean_t1", "mean_t2",   # Theta (4-6 Hz, 6.5-8 Hz)
    "mean_a1", "mean_a2",   # Alpha (8.5-10 Hz, 10.5-13 Hz)
    "mean_b1", "mean_b2",   # Beta (13.5-18 Hz, 18.5-30 Hz)
    "mean_g1", "mean_g2",   # Gamma (30.5-40 Hz, 40-49.5 Hz)
]


# ---------------------------------------------------------------------------
# MATLAB struct field helpers
# ---------------------------------------------------------------------------

def _safe_scalar(struct, field_name: str) -> Optional[float]:
    """Safely extract a scalar float from a MATLAB struct field."""
    try:
        val = struct[field_name]
        if val.size == 0:
            return None
        v = float(val.flat[0])
        if np.isnan(v):
            return None
        return v
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _safe_str(struct, field_name: str) -> str:
    """Safely extract a string from a MATLAB struct field."""
    try:
        val = struct[field_name]
        if val.size == 0:
            return ""
        return str(val.flat[0])
    except (KeyError, IndexError, TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Results .mat loading (v5 / scipy format)
# ---------------------------------------------------------------------------

def load_zuco_results_v5(results_path: Path) -> List[Dict[str, Any]]:
    """Load ZuCo results .mat file (v5 / scipy format).

    Args:
        results_path: Path to results*.mat file.

    Returns:
        List of sentence dicts with keys:
            content, omission_rate, words, all_fixations, n_words,
            rawData_shape
    """
    import scipy.io as sio

    mat = sio.loadmat(str(results_path))
    sd = mat.get("sentenceData")
    if sd is None:
        logger.warning("No sentenceData in %s", results_path.name)
        return []

    sentences = []
    for i in range(sd.shape[1]):
        s = sd[0, i]
        content = _safe_str(s, "content")
        omission = _safe_scalar(s, "omissionRate")

        # Raw EEG shape (for alignment verification)
        raw_shape = None
        try:
            rd = s["rawData"]
            if rd.size > 0:
                raw_shape = rd.shape
        except (KeyError, ValueError):
            pass

        words = _extract_word_features(s)
        all_fix = _extract_all_fixations(s)

        sentences.append({
            "content": content,
            "omission_rate": omission,
            "words": words,
            "all_fixations": all_fix,
            "n_words": len(words),
            "rawData_shape": raw_shape,
        })

    logger.info(
        "Loaded %d sentences from %s", len(sentences), results_path.name,
    )
    return sentences


def _extract_word_features(sentence_struct) -> List[Dict[str, Any]]:
    """Extract per-word features from sentenceData[i].word array."""
    try:
        word_arr = sentence_struct["word"]
    except (KeyError, ValueError):
        return []

    if word_arr.size == 0:
        return []

    words = []
    n_words = word_arr.shape[1] if word_arr.ndim >= 2 else word_arr.size
    for j in range(n_words):
        w = word_arr[0, j] if word_arr.ndim >= 2 else word_arr.flat[j]
        word: Dict[str, Any] = {"content": _safe_str(w, "content")}

        # Eye-tracking reading measures
        for field in _WORD_ET_FIELDS:
            word[field] = _safe_scalar(w, field)

        # Band power features
        for field in _WORD_BAND_FIELDS:
            word[field] = _safe_scalar(w, field)

        # Word-level raw data durations (samples)
        for raw_field, key in [("rawEEG", "rawEEG_samples"), ("rawET", "rawET_samples")]:
            try:
                raw = w[raw_field]
                if raw.size > 0 and raw.flat[0] is not None:
                    inner = raw.flat[0]
                    if hasattr(inner, "shape") and inner.size > 0:
                        word[key] = int(
                            inner.shape[1] if inner.ndim == 2 else inner.shape[0]
                        )
            except (KeyError, IndexError, TypeError):
                pass

        words.append(word)

    return words


def _extract_all_fixations(
    sentence_struct,
) -> Optional[Dict[str, np.ndarray]]:
    """Extract sentence-level fixation data (x, y, duration, pupilsize)."""
    try:
        af = sentence_struct["allFixations"]
        if af.size == 0:
            return None
        af = af[0, 0]
        result: Dict[str, np.ndarray] = {}
        for name in ("x", "y", "duration", "pupilsize"):
            if name in af.dtype.names:
                arr = af[name]
                if arr.size > 0:
                    result[name] = np.array(arr).flatten().astype(np.float32)
        return result if result else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Corrected ET file loading
# ---------------------------------------------------------------------------

def load_corrected_et_v5(et_path: Path) -> Optional[Dict[str, Any]]:
    """Load corrected eye-tracking .mat file (v5 / scipy format).

    The corrected ET files contain raw gaze data, trigger events,
    and parsed eye-movement events (saccades, fixations, blinks).

    Returns:
        Dict with keys:
            data: (N, 4) float64 [TIME, GAZE_X, GAZE_Y, PUPIL_AREA]
            events: (M, 2) int32 [timestamp, event_code]
            colheader: list of column name strings
        or None on failure.
    """
    import scipy.io as sio

    if not et_path.exists():
        return None

    try:
        mat = sio.loadmat(str(et_path))
    except Exception as e:
        logger.warning("Failed to load ET file %s: %s", et_path.name, e)
        return None

    data = mat.get("data")
    events = mat.get("event")
    if data is None:
        return None

    colheader: List[str] = []
    if "colheader" in mat:
        ch = mat["colheader"]
        for i in range(ch.shape[1]):
            try:
                colheader.append(str(ch[0, i].flat[0]))
            except Exception:
                colheader.append(f"col_{i}")

    return {
        "data": data.astype(np.float64),
        "events": (
            events.astype(np.int64)
            if events is not None
            else np.zeros((0, 2), dtype=np.int64)
        ),
        "colheader": colheader,
    }


def segment_et_by_sentences(
    et_data: Dict[str, Any],
    sentence_event_code: int = 10,
) -> List[Optional[Dict[str, np.ndarray]]]:
    """Segment eye-tracking data into per-sentence chunks.

    Uses the same trigger code as the EEG (default 10 = sentence onset).
    Each segment spans from one sentence onset to the next.

    Returns:
        List of per-sentence dicts (or None if empty), each with keys:
            gaze_x, gaze_y, pupil_area: 1D float32 arrays
            timestamps: 1D float64 array
    """
    data = et_data["data"]
    events = et_data["events"]

    # Find sentence onset timestamps
    onsets: List[float] = []
    for i in range(events.shape[0]):
        if int(events[i, 1]) == sentence_event_code:
            onsets.append(float(events[i, 0]))

    if not onsets:
        return []

    timestamps = data[:, 0]

    segments: List[Optional[Dict[str, np.ndarray]]] = []
    for i, onset_time in enumerate(onsets):
        end_time = onsets[i + 1] if (i + 1 < len(onsets)) else (timestamps[-1] + 1)
        mask = (timestamps >= onset_time) & (timestamps < end_time)
        seg = data[mask]

        if seg.shape[0] == 0:
            segments.append(None)
            continue

        segments.append({
            "timestamps": seg[:, 0].astype(np.float64),
            "gaze_x": seg[:, 1].astype(np.float32),
            "gaze_y": seg[:, 2].astype(np.float32),
            "pupil_area": seg[:, 3].astype(np.float32),
        })

    return segments


# ---------------------------------------------------------------------------
# Annotation builders
# ---------------------------------------------------------------------------

def build_word_annotations(
    sentence_data: Dict[str, Any],
    run_id: str,
    sent_idx: int,
    srate: float,
) -> Tuple[List, Dict[str, np.ndarray]]:
    """Build annotation objects and HDF5 arrays from word-level features.

    Args:
        sentence_data: Dict from ``load_zuco_results_v5``.
        run_id: Run identifier for annotation IDs.
        sent_idx: Sentence index within the run.
        srate: EEG sampling rate (Hz) for computing word onset times.

    Returns:
        (annotation_list, annotation_arrays)
        - annotation_list: Annotation objects to append to atom.annotations
        - annotation_arrays: ``{name: ndarray}`` for HDF5 companion storage
    """
    annotations: List = []
    ann_arrays: Dict[str, np.ndarray] = {}

    # ── Sentence content ──
    content = sentence_data.get("content", "")
    if content:
        annotations.append(TextAnnotation(
            annotation_id=f"ann_content_{run_id}_{sent_idx:04d}",
            name="sentence_content",
            text_value=content,
            domain="stimulus",
        ))

    # ── Omission rate ──
    omission = sentence_data.get("omission_rate")
    if omission is not None:
        annotations.append(NumericAnnotation(
            annotation_id=f"ann_omission_{run_id}_{sent_idx:04d}",
            name="omission_rate",
            numeric_value=omission,
            domain="quality",
        ))

    # ── Word-level features → EventSequenceAnnotation ──
    words = sentence_data.get("words", [])
    if words:
        event_items: List[EventItem] = []
        cumulative_samples = 0
        for w_idx, w in enumerate(words):
            # Approximate word onset from cumulative rawEEG durations
            onset_sec = cumulative_samples / srate if srate > 0 else 0.0
            eeg_samples = w.get("rawEEG_samples")
            if eeg_samples is not None and eeg_samples > 0:
                dur_sec: Optional[float] = eeg_samples / srate
                cumulative_samples += eeg_samples
            else:
                dur_sec = None

            # Collect non-None features
            features: Dict[str, Any] = {"word_index": w_idx}
            for field in _WORD_ET_FIELDS:
                val = w.get(field)
                if val is not None:
                    features[field] = val
            for field in _WORD_BAND_FIELDS:
                val = w.get(field)
                if val is not None:
                    features[field] = val

            event_items.append(EventItem(
                onset=onset_sec,
                duration=dur_sec,
                value=w.get("content", ""),
                features=features,
            ))

        annotations.append(EventSequenceAnnotation(
            annotation_id=f"ann_words_{run_id}_{sent_idx:04d}",
            name="word_reading_features",
            domain="stimulus",
            events=event_items,
        ))

    # ── Sentence fixation arrays → HDF5 companion data ──
    all_fix = sentence_data.get("all_fixations")
    if all_fix:
        for key in ("x", "y", "duration", "pupilsize"):
            arr = all_fix.get(key)
            if arr is not None and arr.size > 0:
                ann_arrays[f"fixation_{key}"] = arr

    return annotations, ann_arrays


def build_et_annotations(
    et_segment: Optional[Dict[str, np.ndarray]],
    run_id: str,
    sent_idx: int,
    et_srate: float = 500.0,
) -> Tuple[List, Dict[str, np.ndarray]]:
    """Build ContinuousAnnotation objects for raw eye-tracking time series.

    The actual data is stored as HDF5 annotation arrays (via ShardManager),
    and each ContinuousAnnotation holds a ``data_ref`` pointing to the array.

    Args:
        et_segment: Per-sentence ET dict from ``segment_et_by_sentences``.
        run_id: Run identifier.
        sent_idx: Sentence index.
        et_srate: Eye-tracker sampling rate (Hz).

    Returns:
        (annotation_list, annotation_arrays)
    """
    annotations: List = []
    ann_arrays: Dict[str, np.ndarray] = {}

    if et_segment is None:
        return annotations, ann_arrays

    for channel, ann_name in [
        ("gaze_x", "eye_gaze_x"),
        ("gaze_y", "eye_gaze_y"),
        ("pupil_area", "eye_pupil_area"),
    ]:
        arr = et_segment.get(channel)
        if arr is not None and arr.size > 0:
            ann_arrays[ann_name] = arr
            annotations.append(ContinuousAnnotation(
                annotation_id=f"ann_{ann_name}_{run_id}_{sent_idx:04d}",
                name=ann_name,
                domain="physiological",
                scope="timepoint",
                data_ref=SignalRef(
                    file_path="__et_placeholder__",
                    internal_path=f"__placeholder__/annotations/{ann_name}",
                    shape=(int(arr.shape[0]),),
                ),
                data_sampling_rate=et_srate,
                alignment_method="trigger_locked",
            ))

    return annotations, ann_arrays


# ---------------------------------------------------------------------------
# Results file discovery
# ---------------------------------------------------------------------------

def find_results_file(
    root: Path,
    subject_id: str,
    task: str,
) -> Optional[Path]:
    """Find ZuCo results .mat file for a subject+task.

    Searches for: ``{root}/{task_dir}/Matlab files/results{subject}_{TASK}.mat``

    Args:
        root: Dataset root (ZuCo_1.0_Full or ZuCo_2.0_Full).
        subject_id: Subject code (e.g. 'ZAB').
        task: Task key (e.g. 'sr', 'nr', 'tsr').

    Returns:
        Path to results file, or None.
    """
    # ZuCo 1.0 directory mapping
    task_dirs = {
        "sr": "task1- SR",
        "nr": "task2 - NR",
        "tsr": "task3 - TSR",
    }
    task_dir_name = task_dirs.get(task, "")
    matlab_dir = root / task_dir_name / "Matlab files"

    if not matlab_dir.exists():
        # Try ZuCo 2.0 layout
        matlab_dir = root / "Matlab files"

    if not matlab_dir.exists():
        return None

    # Try exact naming: results{SUBJ}_{TASK}.mat
    task_upper = task.upper()
    p = matlab_dir / f"results{subject_id}_{task_upper}.mat"
    if p.exists():
        return p

    # Fallback: glob for matching pattern
    for p in matlab_dir.glob(f"results{subject_id}*{task_upper}*.mat"):
        return p

    return None


def find_et_file(
    prep_dir: Path,
    subject_id: str,
    text_id: str,
) -> Optional[Path]:
    """Find corrected eye-tracking .mat for a subject+text.

    Searches for: ``{prep_dir}/{subject}/{subject}_{text}_corrected_ET.mat``

    Args:
        prep_dir: Preprocessed directory path.
        subject_id: Subject code.
        text_id: Text identifier (e.g. 'SR1', 'TSR1').

    Returns:
        Path to ET file, or None.
    """
    sub_dir = prep_dir / subject_id
    p = sub_dir / f"{subject_id}_{text_id}_corrected_ET.mat"
    if p.exists():
        return p

    # Glob fallback
    for p in sub_dir.glob(f"*{text_id}*corrected_ET*.mat"):
        return p
    for p in sub_dir.glob(f"*{text_id}*ET*.mat"):
        if "corrected" in p.name.lower():
            return p

    return None
