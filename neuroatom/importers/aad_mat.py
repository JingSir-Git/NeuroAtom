"""AAD .mat Importer: Auditory Attention Detection datasets (KUL, DTU, etc.).

Handles the peculiar MATLAB struct-of-trials format used by AAD research labs.
Each subject .mat file contains multiple trials with EEG data, attended-ear
labels, stimulus metadata, and condition information.

Supported datasets:
    - KUL (Biesmans et al., 2016): 16 subjects, 64 EEG, 128 Hz, preprocessed
    - DTU (Fuglsang et al., 2017): 18 subjects, 64+EXG, 512 Hz raw / 64 Hz preproc
    - Generic AAD struct-of-trials format

Data structures differ significantly between labs — this importer uses
format-detection heuristics and a plugin-style adapter pattern.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

from neuroatom.utils.optional_deps import require as _require

mne = _require("mne", "the AAD (KUL/DTU) importer")

from neuroatom.core.annotation import CategoricalAnnotation, TextAnnotation
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.enums import ChannelStatus, ChannelType
from neuroatom.importers.base import BaseImporter, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KUL adapter
# ---------------------------------------------------------------------------

def _parse_kul_trial(trial_struct, subject_id: str) -> Dict[str, Any]:
    """Parse a single KUL trial struct into a flat dict.

    KUL trial struct fields:
        RawData.EegData:  (n_samples, n_channels) float64, already in µV
        FileHeader.SampleRate: int (128)
        FileHeader.Channels:  (n_channels,) struct array with .Label
        attended_ear: 'L' or 'R'
        attended_track: int (1 or 2)
        condition: 'hrtf' or 'dry'
        experiment: int (1-3)
        part: int
        TrialID: int (1-20)
        stimuli: (2,) object array of wav filenames
        subject: str ('S1')
        repetition: int (0 or 1)
    """
    eeg = trial_struct.RawData.EegData  # (samples, channels)
    srate = float(trial_struct.FileHeader.SampleRate)

    # Extract channel names from nested struct array
    ch_structs = trial_struct.FileHeader.Channels
    ch_names = []
    for ch in ch_structs:
        name = getattr(ch, "Label", None)
        if name is None:
            name = getattr(ch, "label", f"Ch_{len(ch_names)+1}")
        ch_names.append(str(name))

    # Get unit from first channel
    unit = "uV"  # KUL default
    if len(ch_structs) > 0:
        dim = getattr(ch_structs[0], "PhysicalDimension", "uV")
        if dim:
            unit = str(dim).strip()

    # Stimuli filenames
    stimuli = []
    if hasattr(trial_struct, "stimuli"):
        stim = trial_struct.stimuli
        if hasattr(stim, "__iter__"):
            stimuli = [str(s) for s in stim]

    # Electrode cap info
    electrode_cap = None
    if hasattr(trial_struct, "FileHeader"):
        fh = trial_struct.FileHeader
        if hasattr(fh, "ElectrodeCap"):
            electrode_cap = str(fh.ElectrodeCap)

    trial_meta = {
        "eeg": eeg,  # (n_samples, n_channels)
        "srate": srate,
        "ch_names": ch_names,
        "unit": unit,
        "trial_id": int(trial_struct.TrialID),
        "attended_ear": str(trial_struct.attended_ear),
        "attended_track": int(trial_struct.attended_track),
        "condition": str(trial_struct.condition),
        "experiment": int(trial_struct.experiment),
        "part": int(trial_struct.part),
        "repetition": int(getattr(trial_struct, "repetition", 0)),
        "subject": str(getattr(trial_struct, "subject", subject_id)),
        "stimuli": stimuli,
        "electrode_cap": electrode_cap,
    }
    return trial_meta


def _parse_kul_mat(mat_path: Path, subject_id: str) -> List[Dict[str, Any]]:
    """Parse all trials from a KUL .mat file."""
    logger.info("Loading KUL .mat: %s", mat_path)
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    trials_raw = mat["trials"]

    trials = []
    for t in trials_raw:
        trial = _parse_kul_trial(t, subject_id)
        trials.append(trial)
    logger.info("Parsed %d KUL trials from %s", len(trials), mat_path.name)
    return trials


# ---------------------------------------------------------------------------
# DTU adapter (preprocessed DATA_preproc format)
# ---------------------------------------------------------------------------

def _parse_dtu_preproc_mat(mat_path: Path, subject_id: str) -> List[Dict[str, Any]]:
    """Parse DTU preprocessed .mat file (DATA_preproc/Sx_data_preproc.mat).

    Structure:
        data.eeg:       object array (n_trials,), each (n_samples, n_channels) float64
        data.fsample.eeg:  int (typically 64 Hz)
        data.dim.chan.eeg:  object array (n_trials,), each (n_channels,) str array
        data.event.eeg:    object array (n_trials,), each struct with .sample, .value
        data.wavA:      object array (n_trials,), each (n_samples,) — audio envelope A
        data.wavB:      object array (n_trials,), each (n_samples,) — audio envelope B
    """
    logger.info("Loading DTU preprocessed .mat: %s", mat_path)
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    data = mat["data"]

    # Sampling rate
    srate = float(data.fsample.eeg)

    # EEG trials
    eeg_trials = data.eeg  # object array
    n_trials = len(eeg_trials)

    # Channel names — per-trial (take from dim.chan.eeg)
    chan_arrays = data.dim.chan.eeg  # object array of per-trial channel arrays

    # Events — per-trial
    event_array = data.event.eeg  # object array of structs

    # Audio envelopes
    wavA = data.wavA if hasattr(data, "wavA") else None
    wavB = data.wavB if hasattr(data, "wavB") else None

    trials = []
    for i in range(n_trials):
        eeg_data = eeg_trials.flat[i]  # (n_samples, n_channels)

        # Channel names for this trial
        ch_names_raw = chan_arrays.flat[i] if chan_arrays.shape[0] > i else chan_arrays.flat[0]
        ch_names = [str(c) for c in ch_names_raw]

        # Event for this trial — attended speaker
        attended_value = "unknown"
        ev = event_array.flat[i]
        if hasattr(ev, "value"):
            v = ev.value
            if hasattr(v, "__len__"):
                attended_value = str(v.flat[0]) if len(v) > 0 else "unknown"
            else:
                attended_value = str(v)

        # Audio envelopes
        wav_a = wavA.flat[i] if wavA is not None and i < len(wavA) else None
        wav_b = wavB.flat[i] if wavB is not None and i < len(wavB) else None

        trial = {
            "eeg": eeg_data,  # (n_samples, n_channels)
            "srate": srate,
            "ch_names": ch_names,
            "unit": "uV",  # Preprocessed DTU data is in µV
            "trial_id": i + 1,
            "attended_speaker": attended_value,  # '1' or '2'
            "attended_ear": "unknown",  # Not available in preproc
            "condition": "dichotic",
            "experiment": 1,
            "part": i + 1,
            "repetition": 0,
            "subject": subject_id,
            "stimuli": [],
            "wav_a": wav_a,
            "wav_b": wav_b,
        }
        trials.append(trial)

    logger.info("Parsed %d DTU preprocessed trials from %s", n_trials, mat_path.name)
    return trials


# ---------------------------------------------------------------------------
# DTU raw EEG format
# ---------------------------------------------------------------------------

def _parse_dtu_raw_mat(mat_path: Path, subject_id: str) -> List[Dict[str, Any]]:
    """Parse DTU raw EEG .mat file (EEG/Sx.mat).

    Structure:
        data.eeg:        (n_total_samples, n_channels) float64 — continuous
        data.fsample.eeg:   int (512 Hz)
        data.dim.chan.eeg:  (n_channels,) str array
        data.event.eeg:     struct with .sample (n_events,), .value (n_events,)

    This is a continuous recording — we return it as a single "trial"
    that the WindowAtomizer can segment.
    """
    logger.info("Loading DTU raw .mat: %s", mat_path)
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    data = mat["data"]

    srate = float(data.fsample.eeg)
    eeg = data.eeg  # (n_samples, n_channels)

    # Channel names
    ch_names = [str(c) for c in data.dim.chan.eeg]

    # Events
    ev = data.event.eeg
    event_samples = ev.sample  # (n_events,) int32
    event_values = ev.value    # (n_events,) object (str)

    # Build MNE-format events array: (n_events, 3) [sample, 0, event_id]
    events_list = []
    for s, v in zip(event_samples, event_values):
        events_list.append([int(s), 0, int(v)])

    trial = {
        "eeg": eeg,
        "srate": srate,
        "ch_names": ch_names,
        "unit": "raw_biosemi",  # Raw BioSemi integers — needs conversion
        "trial_id": 0,
        "attended_ear": "unknown",
        "condition": "continuous",
        "experiment": 1,
        "part": 0,
        "repetition": 0,
        "subject": subject_id,
        "stimuli": [],
        "events_array": np.array(events_list, dtype=np.int64) if events_list else None,
    }

    logger.info(
        "Parsed DTU raw: %d samples × %d channels at %g Hz (%d events)",
        eeg.shape[0], eeg.shape[1], srate, len(events_list),
    )
    return [trial]


# ---------------------------------------------------------------------------
# AAD Importer
# ---------------------------------------------------------------------------

def _detect_aad_format(mat_path: Path) -> Optional[str]:
    """Detect the specific AAD sub-format from a .mat file.

    Returns: 'kul', 'dtu_preproc', 'dtu_raw', or None.
    """
    if not mat_path.suffix.lower() == ".mat":
        return None

    try:
        mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False,
                          variable_names=["trials", "data"])
    except Exception:
        return None

    # KUL: top-level 'trials' key → array of structs
    if "trials" in mat:
        t = mat["trials"]
        if hasattr(t, "__len__") and len(t) > 0:
            first = t.flat[0] if hasattr(t, "flat") else t[0]
            if hasattr(first, "RawData") and hasattr(first, "FileHeader"):
                return "kul"

    # DTU: top-level 'data' struct with .eeg, .fsample, .event, .dim
    if "data" in mat:
        d = mat["data"]
        if hasattr(d, "eeg") and hasattr(d, "fsample") and hasattr(d, "event"):
            eeg = d.eeg
            if isinstance(eeg, np.ndarray) and eeg.dtype == object:
                return "dtu_preproc"
            elif isinstance(eeg, np.ndarray) and eeg.ndim == 2:
                return "dtu_raw"

    return None


class AADImporter(BaseImporter):
    """Importer for Auditory Attention Detection .mat datasets.

    Handles KUL, DTU (raw + preprocessed), and similar struct-of-trials formats.
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)
        self._format_hint: Optional[str] = None

    @staticmethod
    def detect(path: Path) -> bool:
        path = Path(path)
        if path.is_dir():
            # Check if directory contains AAD-style .mat files
            mats = list(path.glob("S*.mat"))
            if mats:
                fmt = _detect_aad_format(mats[0])
                return fmt is not None
            return False

        if path.suffix.lower() != ".mat":
            return False

        return _detect_aad_format(path) is not None

    def load_raw(self, path: Path) -> Tuple[mne.io.Raw, Dict[str, Any]]:
        """Load a SINGLE trial as MNE RawArray.

        For AAD datasets, we don't use load_raw in the standard way.
        Instead, use import_subject() which handles multi-trial files.
        """
        raise NotImplementedError(
            "AADImporter.load_raw() is not used directly. "
            "Use import_subject() for multi-trial AAD .mat files."
        )

    def extract_channel_infos(self, raw: mne.io.Raw) -> List[ChannelInfo]:
        """Extract channel info from an MNE Raw object (used after wrapping)."""
        ch_infos = []
        for idx, ch_name in enumerate(raw.info["ch_names"]):
            ch_type = ChannelType.EEG
            # EXG channels
            if ch_name.upper().startswith("EXG"):
                ch_type = ChannelType.EOG
            elif ch_name.upper() == "STATUS":
                ch_type = ChannelType.STIM

            # Task config overrides
            override = self.task_config.channel_type_overrides.get(ch_name)
            if override:
                ch_type = ChannelType(override)

            # Skip excluded channels
            if ch_name in self.task_config.exclude_channels:
                continue

            std_name = standardize_channel_name(ch_name)
            ch_infos.append(ChannelInfo(
                channel_id=f"ch_{idx:03d}",
                index=idx,
                name=ch_name,
                standard_name=std_name,
                type=ch_type,
                unit=self.task_config.signal_unit or "uV",
                sampling_rate=raw.info["sfreq"],
                status=ChannelStatus.UNKNOWN,
            ))
        return ch_infos

    def extract_events(self, raw: mne.io.Raw) -> Optional[np.ndarray]:
        """Not used — events are embedded in trial metadata."""
        return None

    # ------------------------------------------------------------------
    # Main entry point: import entire subject
    # ------------------------------------------------------------------

    def import_subject(
        self,
        mat_path: Path,
        subject_id: str,
        session_id: str = "ses-01",
        format_hint: Optional[str] = None,
        max_trials: Optional[int] = None,
    ) -> List:
        """Import all trials from a single subject .mat file.

        This is the primary entry point for AAD datasets. Each trial
        becomes a separate run in the pool.

        Args:
            mat_path: Path to subject .mat file.
            subject_id: Subject identifier (e.g. 'S1').
            session_id: Session identifier.
            format_hint: 'kul', 'dtu_preproc', or 'dtu_raw'. Auto-detected if None.
            max_trials: Maximum number of trials to import (for testing).

        Returns:
            List of ImportResult objects, one per trial.
        """
        from neuroatom.atomizer.window import WindowAtomizer
        from neuroatom.importers.base import ImportResult
        from neuroatom.core.run import RunMeta
        from neuroatom.core.signal_ref import SignalRef
        from neuroatom.utils.validation import validate_signal
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P
        from neuroatom.core.annotation import CategoricalAnnotation, TextAnnotation
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.enums import AtomType
        from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
        from neuroatom.utils.hashing import compute_atom_id

        mat_path = Path(mat_path)

        # Detect format
        fmt = format_hint or _detect_aad_format(mat_path)
        if fmt is None:
            raise ValueError(f"Cannot detect AAD format for {mat_path}")

        self._format_hint = fmt
        dataset_id = self.task_config.dataset_id

        # Parse trials
        if fmt == "kul":
            trials = _parse_kul_mat(mat_path, subject_id)
        elif fmt == "dtu_preproc":
            trials = _parse_dtu_preproc_mat(mat_path, subject_id)
        elif fmt == "dtu_raw":
            trials = _parse_dtu_raw_mat(mat_path, subject_id)
        else:
            raise ValueError(f"Unknown AAD format: {fmt}")

        if max_trials is not None:
            trials = trials[:max_trials]

        # Ensure dataset and subject exist in pool
        self.pool.ensure_dataset(dataset_id)
        self.pool.ensure_subject(dataset_id, subject_id)

        results = []

        for trial in trials:
            trial_id = trial["trial_id"]
            run_id = f"trial_{trial_id:03d}"

            # Build MNE RawArray wrapper
            eeg_data = trial["eeg"]  # (n_samples, n_channels) or (n_channels, n_samples)

            # Ensure (n_channels, n_samples) orientation
            if eeg_data.ndim == 2 and eeg_data.shape[0] > eeg_data.shape[1]:
                # (n_samples, n_channels) → transpose
                eeg_data = eeg_data.T

            n_channels, n_samples = eeg_data.shape
            srate = trial["srate"]
            ch_names = trial["ch_names"]

            # Ensure channel count matches
            if len(ch_names) != n_channels:
                logger.warning(
                    "Channel name count (%d) != data channel count (%d) in trial %d. "
                    "Generating generic names.",
                    len(ch_names), n_channels, trial_id,
                )
                ch_names = [f"Ch_{i+1}" for i in range(n_channels)]

            # Create MNE Info and RawArray
            ch_types = []
            for ch_name in ch_names:
                if ch_name.upper().startswith("EXG"):
                    ch_types.append("eog")
                elif ch_name.upper() == "STATUS":
                    ch_types.append("stim")
                else:
                    ch_types.append("eeg")

            info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)

            # Convert units: if already in µV, convert to V for MNE (MNE expects V)
            unit = trial.get("unit", "uV")
            if unit in ("uV", "µV"):
                eeg_data_volts = eeg_data * 1e-6
                declared_unit = "uV"
            elif unit == "V":
                eeg_data_volts = eeg_data
                declared_unit = "V"
            elif unit == "raw_biosemi":
                # Raw BioSemi integers — don't scale, just pass through as-is
                eeg_data_volts = eeg_data
                declared_unit = "raw"
            else:
                eeg_data_volts = eeg_data * 1e-6  # Assume µV
                declared_unit = unit

            raw = mne.io.RawArray(eeg_data_volts.astype(np.float64), info, verbose=False)

            # Extract channel infos
            channel_infos = self.extract_channel_infos(raw)
            channel_ids = [ch.channel_id for ch in channel_infos]

            # Build annotations for this trial
            annotations = []

            # Attended ear / speaker
            if "attended_ear" in trial and trial["attended_ear"] != "unknown":
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_ear_{trial_id:03d}",
                    name="attended_ear",
                    value=trial["attended_ear"],
                ))

            if "attended_speaker" in trial and trial["attended_speaker"] != "unknown":
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_speaker_{trial_id:03d}",
                    name="attended_speaker",
                    value=trial["attended_speaker"],
                ))

            if "attended_track" in trial:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_track_{trial_id:03d}",
                    name="attended_track",
                    value=str(trial["attended_track"]),
                ))

            if "condition" in trial:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_cond_{trial_id:03d}",
                    name="condition",
                    value=trial["condition"],
                ))

            if trial.get("stimuli"):
                annotations.append(TextAnnotation(
                    annotation_id=f"ann_stim_{trial_id:03d}",
                    name="stimuli",
                    text_value="|".join(trial["stimuli"]),
                ))

            if "experiment" in trial and trial["experiment"]:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_exp_{trial_id:03d}",
                    name="experiment",
                    value=str(trial["experiment"]),
                ))

            if "part" in trial:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_part_{trial_id:03d}",
                    name="part",
                    value=str(trial["part"]),
                ))

            if "repetition" in trial:
                annotations.append(CategoricalAnnotation(
                    annotation_id=f"ann_rep_{trial_id:03d}",
                    name="repetition",
                    value=str(trial["repetition"]),
                ))

            # For AAD paradigm, each trial is one continuous segment
            # We create a single TRIAL atom per trial
            atom_id = compute_atom_id(
                dataset_id=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                onset_sample=0,
            )

            atom = Atom(
                atom_id=atom_id,
                atom_type=AtomType.TRIAL,
                dataset_id=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                trial_index=trial_id,
                signal_ref=SignalRef(
                    file_path="__placeholder__",
                    internal_path=f"/atoms/{atom_id}/signal",
                    shape=(len(channel_ids), n_samples),
                ),
                temporal=TemporalInfo(
                    onset_sample=0,
                    onset_seconds=0.0,
                    duration_samples=n_samples,
                    duration_seconds=n_samples / srate,
                ),
                channel_ids=channel_ids,
                n_channels=len(channel_ids),
                sampling_rate=srate,
                annotations=annotations,
                processing_history=ProcessingHistory(
                    steps=[
                        ProcessingStep(
                            operation="raw_import",
                            parameters={
                                "format": f"aad_{fmt}",
                                "source_file": mat_path.name,
                                "trial_id": trial_id,
                                "unit": declared_unit,
                            },
                        ),
                    ],
                    is_raw=True,
                    version_tag="raw",
                ),
                custom_fields={
                    "electrode_cap": trial.get("electrode_cap", None),
                    "experiment": trial.get("experiment", None),
                    "part": trial.get("part", None),
                    "repetition": trial.get("repetition", None),
                },
            )

            # Standardize channel names
            for ch in channel_infos:
                if ch.standard_name is None:
                    ch.standard_name = standardize_channel_name(ch.name)

            # Build run meta
            run_meta = RunMeta(
                run_id=run_id,
                session_id=session_id,
                subject_id=subject_id,
                dataset_id=dataset_id,
                run_index=trial_id,
                task_type=self.task_config.task_type,
                n_events=0,
                n_trials=1,
                paradigm_details={
                    "format": fmt,
                    "trial_id": trial_id,
                    "condition": trial.get("condition", ""),
                    "attended_ear": trial.get("attended_ear", ""),
                    "experiment": trial.get("experiment", ""),
                },
            )

            # Store signals and metadata
            self.pool.ensure_session(dataset_id, subject_id, session_id)
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            max_shard_mb = self.pool.config.get("storage", {}).get("max_shard_size_mb", 200.0)
            compression = self.pool.config.get("storage", {}).get("compression", "gzip")

            # Extract signal data (channels × samples)
            signal = raw.get_data().astype(np.float32)

            # Validate signal
            warnings = validate_signal(
                signal=signal,
                atom_id=atom.atom_id,
                config=self.pool.config.get("import", {}),
            )

            # Write to HDF5 shard
            with ShardManager(
                pool_root=self.pool.root,
                dataset_id=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                max_shard_size_mb=max_shard_mb,
                compression=compression,
            ) as shard_mgr:
                # Prepare annotation arrays (audio envelopes if available)
                ann_arrays = {}
                if trial.get("wav_a") is not None:
                    ann_arrays["audio_envelope_A"] = np.array(trial["wav_a"], dtype=np.float32)
                if trial.get("wav_b") is not None:
                    ann_arrays["audio_envelope_B"] = np.array(trial["wav_b"], dtype=np.float32)

                signal_ref = shard_mgr.write_atom_signal(
                    atom.atom_id, signal, ann_arrays
                )
                atom.signal_ref = signal_ref

            # Write JSONL
            jsonl_path = P.atoms_jsonl_path(
                self.pool.root, dataset_id, subject_id, session_id, run_id
            )
            with AtomJSONLWriter(jsonl_path) as writer:
                writer.write_atom(atom)

            # Register in pool
            self.pool.register_run(run_meta)

            logger.info(
                "Imported %s/%s/%s/%s: trial %d (%d ch × %d samples = %.1fs at %g Hz)",
                dataset_id, subject_id, session_id, run_id,
                trial_id, len(channel_ids), n_samples,
                n_samples / srate, srate,
            )

            results.append(ImportResult(
                atoms=[atom],
                run_meta=run_meta,
                channel_infos=channel_infos,
                warnings=warnings,
            ))

        logger.info(
            "Imported subject %s: %d trials, %d total atoms.",
            subject_id, len(results), sum(len(r.atoms) for r in results),
        )
        return results


# Auto-register
register_importer("aad_mat", AADImporter)
