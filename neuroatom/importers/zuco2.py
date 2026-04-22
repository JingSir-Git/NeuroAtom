"""Zuco 2.0 Task-Specific Reading (TSR) Importer.

Handles the Zuco 2.0 EEG reading dataset: 18 subjects, 7 texts each,
105 EGI channels @ 500 Hz, HDF5 (MATLAB v7.3) EEGLAB format.

Key features:
    - HDF5-based EEGLAB struct parsing with h5py
    - Sentence-level epoch extraction from event markers (type='10')
    - 3D electrode coordinates from chanlocs
    - Word boundary metadata integration
    - Supports both gip (cleaned) and bip (artifact) file prefixes

Data layout expected:
    task2 - TSR/
        Preprocessed/
            wordbounds_TSR1.mat ... wordbounds_TSR7.mat
            YAC/
                gip_YAC_TSR1_EEG.mat ... gip_YAC_TSR7_EEG.mat
                bip_YAC_TSR4_EEG.mat ...
                YAC_TSR1_corrected_ET.mat ...
            YAG/
                ...
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio

from neuroatom.core.annotation import CategoricalAnnotation, NumericAnnotation
from neuroatom.core.atom import Atom, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import AtomType, ChannelStatus, ChannelType
from neuroatom.core.provenance import ProcessingHistory, ProcessingStep
from neuroatom.core.quality import QualityInfo
from neuroatom.core.signal_ref import SignalRef
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
from neuroatom.importers.registry import register_importer
from neuroatom.storage.pool import Pool
from neuroatom.utils.channel_names import standardize_channel_name
from neuroatom.utils.hashing import compute_atom_id
from neuroatom.utils.validation import validate_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HDF5 EEGLAB helpers
# ---------------------------------------------------------------------------

def _h5_read_string(f: h5py.File, ref) -> str:
    """Read a string from an HDF5 object reference."""
    data = f[ref]
    if data.ndim == 2:
        return "".join(chr(int(c)) for c in data[:, 0] if c > 0)
    elif data.ndim == 1:
        return "".join(chr(int(c)) for c in data[:] if c > 0)
    return str(data[()])


def _h5_read_scalar(f: h5py.File, val) -> float:
    """Read a scalar float from an HDF5 dataset or reference."""
    if isinstance(val, h5py.Reference):
        data = f[val]
        if hasattr(data, "shape") and data.size == 1:
            return float(data[()].flat[0])
        return float(data[()])
    if hasattr(val, "shape"):
        return float(val.flat[0])
    return float(val)


def _extract_channel_infos(
    f: h5py.File,
    eeg_group: h5py.Group,
    srate: float,
) -> List[ChannelInfo]:
    """Extract channel names and 3D coordinates from EEGLAB chanlocs."""
    chanlocs = eeg_group["chanlocs"]
    labels_ds = chanlocs["labels"]
    n_ch = labels_ds.shape[0]

    has_coords = "X" in chanlocs and "Y" in chanlocs and "Z" in chanlocs

    ch_infos = []
    for i in range(n_ch):
        name = _h5_read_string(f, labels_ds[i, 0])

        location = None
        if has_coords:
            try:
                x = _h5_read_scalar(f, chanlocs["X"][i, 0])
                y = _h5_read_scalar(f, chanlocs["Y"][i, 0])
                z = _h5_read_scalar(f, chanlocs["Z"][i, 0])
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    location = ElectrodeLocation(
                        x=x, y=y, z=z,
                        coordinate_system="EGI_cart",
                        coordinate_units="cm",
                    )
            except Exception:
                pass

        std_name = standardize_channel_name(name)
        ch_infos.append(ChannelInfo(
            channel_id=f"eeg_{i:03d}",
            index=i,
            name=name,
            standard_name=std_name,
            type=ChannelType.EEG,
            unit="uV",  # Preprocessed data in µV
            sampling_rate=srate,
            status=ChannelStatus.GOOD,
            location=location,
        ))

    return ch_infos


def _extract_events(
    f: h5py.File,
    eeg_group: h5py.Group,
) -> List[Tuple[str, int, float]]:
    """Extract events as (type_str, latency_sample, duration).

    Returns sorted by latency.
    """
    event = eeg_group["event"]
    type_ds = event["type"]
    lat_ds = event["latency"]
    dur_ds = event["duration"]

    n_events = type_ds.shape[0]
    events = []

    for i in range(n_events):
        etype = _h5_read_string(f, type_ds[i, 0]).strip()
        lat = int(_h5_read_scalar(f, lat_ds[i, 0]))
        dur = _h5_read_scalar(f, dur_ds[i, 0])
        events.append((etype, lat, dur))

    events.sort(key=lambda x: x[1])
    return events


def _sentence_epochs(
    events: List[Tuple[str, int, float]],
    total_samples: int,
    sentence_event: str = "10",
) -> List[Tuple[int, int]]:
    """Extract sentence epoch boundaries from events.

    Each sentence starts at a '10' event and ends at the next '10' event
    (or at the end of the recording for the last sentence).

    Returns:
        List of (start_sample, end_sample) tuples.
    """
    onsets = [lat for etype, lat, _ in events if etype == sentence_event]
    epochs = []
    for i, onset in enumerate(onsets):
        if i + 1 < len(onsets):
            end = onsets[i + 1]
        else:
            end = total_samples
        epochs.append((onset, end))
    return epochs


def _load_wordbounds(prep_dir: Path, text_id: str) -> Optional[np.ndarray]:
    """Load word boundaries for a text from wordbounds_TSRx.mat.

    Returns:
        Object array of shape (1, n_sentences) where each element is
        an (n_words, 4) array of pixel bounding boxes, or None.
    """
    wb_path = prep_dir / f"wordbounds_{text_id}.mat"
    if not wb_path.exists():
        return None
    try:
        mat = sio.loadmat(str(wb_path))
        return mat.get("wordbounds", None)
    except Exception:
        return None


def _extract_automagic_quality(
    f: h5py.File,
) -> Dict[str, Any]:
    """Extract quality metrics from automagic group."""
    quality_info: Dict[str, Any] = {}
    if "automagic" not in f:
        return quality_info

    auto = f["automagic"]

    # Quality scores
    if "qualityScores" in auto:
        qs = auto["qualityScores"]
        if hasattr(qs, "keys"):
            for k in qs.keys():
                try:
                    val = float(qs[k][()].flat[0])
                    quality_info[f"qc_{k}"] = val
                except Exception:
                    pass

    # Selected quality score
    if "selectedQualityScore" in auto:
        try:
            quality_info["selected_qc"] = float(auto["selectedQualityScore"][()].flat[0])
        except Exception:
            pass

    # Rating
    if "rate" in auto:
        try:
            r = _h5_read_string(f, auto["rate"][0, 0]) if auto["rate"].dtype == "O" else str(auto["rate"][()].flat[0])
            quality_info["automagic_rate"] = r
        except Exception:
            pass

    # Bad channels
    if "finalBadChans" in auto:
        try:
            bc = auto["finalBadChans"]
            if bc.shape[0] > 0 and bc.dtype.kind == "O":
                bad_chs = []
                for i in range(bc.shape[0]):
                    bad_chs.append(_h5_read_string(f, bc[i, 0]))
                quality_info["bad_channels"] = bad_chs
            elif bc.size > 0:
                quality_info["n_bad_channels"] = int(bc.size)
        except Exception:
            pass

    # Reference type
    if "EEGReference" in auto:
        try:
            ref_data = f[auto["EEGReference"][0, 0]]
            quality_info["reference"] = "".join(chr(int(c)) for c in ref_data[:, 0] if c > 0)
        except Exception:
            pass

    return quality_info


# ---------------------------------------------------------------------------
# Zuco 2.0 Importer
# ---------------------------------------------------------------------------

class Zuco2Importer(BaseImporter):
    """Importer for the Zuco 2.0 Task-Specific Reading Dataset.

    Supports:
        - HDF5 EEGLAB format parsing
        - Sentence-level epoch extraction
        - 3D electrode coordinate extraction
        - Per-text import with sentence-level atoms
    """

    def __init__(self, pool: Pool, task_config: TaskConfig):
        super().__init__(pool, task_config)
        self._file_prefix = task_config.data.get("file_prefix", "gip")

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if path is a Zuco 2.0 dataset."""
        path = Path(path)
        if not path.is_dir():
            return False
        prep = path / "Preprocessed"
        if not prep.exists():
            return False
        # Look for subject dirs with gip_*_EEG.mat files
        for sub_dir in prep.iterdir():
            if sub_dir.is_dir():
                gips = list(sub_dir.glob("gip_*_EEG.mat"))
                if gips:
                    return True
        return False

    def load_raw(self, path):
        raise NotImplementedError("Use import_subject() instead.")

    def extract_channel_infos(self, raw):
        raise NotImplementedError("Use _extract_channel_infos() instead.")

    def extract_events(self, raw):
        return None

    # ------------------------------------------------------------------
    # Import one text (one EEG file)
    # ------------------------------------------------------------------

    def _import_text(
        self,
        mat_path: Path,
        text_id: str,
        dataset_id: str,
        subject_id: str,
        max_sentences: Optional[int] = None,
    ) -> Tuple[List[Atom], List[ChannelInfo], List[str]]:
        """Import sentence-level epochs from one text's EEG file.

        Returns:
            (atoms, channel_infos, warnings)
        """
        from neuroatom.storage.signal_store import ShardManager
        from neuroatom.storage.metadata_store import AtomJSONLWriter
        from neuroatom.storage import paths as P

        session_id = f"ses-{text_id.lower()}"
        run_id = "run-01"

        # Load word boundaries if available
        prep_dir = mat_path.parent.parent  # Preprocessed/ dir
        wordbounds = _load_wordbounds(prep_dir, text_id)

        with h5py.File(str(mat_path), "r") as f:
            eeg = f["EEG"]

            # Basic info
            srate = float(eeg["srate"][()].flat[0])
            data_ds = eeg["data"]
            total_samples, n_channels = data_ds.shape

            # Extract automagic quality metrics
            automagic_info = _extract_automagic_quality(f)

            # Extract reference type from EEG.ref if present
            ref_type = None
            if "ref" in eeg:
                try:
                    ref_ds = eeg["ref"]
                    if ref_ds.dtype.kind == "O":
                        ref_type = _h5_read_string(f, ref_ds[0, 0])
                    elif ref_ds.ndim == 2:
                        ref_type = "".join(chr(int(c)) for c in ref_ds[:, 0] if c > 0)
                    else:
                        ref_type = str(ref_ds[()].flat[0])
                except Exception:
                    pass

            # Channel info
            ch_infos = _extract_channel_infos(f, eeg, srate)

            # Events
            events = _extract_events(f, eeg)
            sentence_event = self.task_config.data.get("events", {}).get(
                "sentence_onset", "10"
            )
            epochs = _sentence_epochs(events, total_samples, sentence_event)

            if max_sentences is not None:
                epochs = epochs[:max_sentences]

            logger.info(
                "Loading %s/%s: %d sentences, %d ch @ %.0f Hz, %.0fs total",
                subject_id, text_id, len(epochs), n_channels, srate,
                total_samples / srate,
            )

            if not epochs:
                logger.warning("No sentence epochs found in %s", mat_path.name)
                return [], ch_infos, []

            # Ensure pool hierarchy
            self.pool.ensure_session(
                dataset_id, subject_id, session_id, sampling_rate=srate
            )
            self.pool.ensure_run(dataset_id, subject_id, session_id, run_id)

            # Build quality from automagic info
            bad_ch_names = automagic_info.get("bad_channels", [])
            quality = QualityInfo(
                overall_status="good",
                bad_channels=bad_ch_names,
                auto_qc_passed=automagic_info.get("automagic_rate", "") != "bad"
                if "automagic_rate" in automagic_info else None,
            )
            max_shard_mb = self.pool.config.get("storage", {}).get(
                "max_shard_size_mb", 200.0
            )
            compression = self.pool.config.get("storage", {}).get(
                "compression", "gzip"
            )
            channel_ids = [ch.channel_id for ch in ch_infos]

            atoms = []
            all_warnings = []

            with ShardManager(
                pool_root=self.pool.root,
                dataset_id=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
                max_shard_size_mb=max_shard_mb,
                compression=compression,
            ) as shard_mgr:
                jsonl_path = P.atoms_jsonl_path(
                    self.pool.root, dataset_id, subject_id, session_id, run_id
                )
                with AtomJSONLWriter(jsonl_path) as writer:
                    for sent_idx, (start_samp, end_samp) in enumerate(epochs):
                        n_sent_samples = end_samp - start_samp

                        # Read this sentence's EEG segment
                        # Data is (samples, channels) in HDF5
                        raw_segment = data_ds[start_samp:end_samp, :]
                        # Transpose to (channels, samples) for NeuroAtom
                        signal = raw_segment.T.astype(np.float32)

                        duration_s = n_sent_samples / srate

                        annotations = [
                            CategoricalAnnotation(
                                annotation_id=f"ann_text_{run_id}_{sent_idx:04d}",
                                name="text_id",
                                value=text_id,
                            ),
                            NumericAnnotation(
                                annotation_id=f"ann_sent_{run_id}_{sent_idx:04d}",
                                name="sentence_index",
                                numeric_value=float(sent_idx),
                            ),
                            CategoricalAnnotation(
                                annotation_id=f"ann_task_{run_id}_{sent_idx:04d}",
                                name="task",
                                value="reading",
                            ),
                        ]

                        # Word count from wordbounds
                        n_words = None
                        if wordbounds is not None:
                            try:
                                wb_arr = wordbounds[0, sent_idx] if sent_idx < wordbounds.shape[1] else None
                                if wb_arr is not None and hasattr(wb_arr, "shape"):
                                    n_words = int(wb_arr.shape[0])
                                    annotations.append(NumericAnnotation(
                                        annotation_id=f"ann_nwords_{run_id}_{sent_idx:04d}",
                                        name="n_words",
                                        numeric_value=float(n_words),
                                    ))
                            except Exception:
                                pass

                        atom_id = compute_atom_id(
                            dataset_id=dataset_id,
                            subject_id=subject_id,
                            session_id=session_id,
                            run_id=run_id,
                            onset_sample=start_samp,
                        )

                        temporal = TemporalInfo(
                            onset_sample=start_samp,
                            onset_seconds=start_samp / srate,
                            duration_samples=n_sent_samples,
                            duration_seconds=duration_s,
                        )

                        atom = Atom(
                            atom_id=atom_id,
                            atom_type=AtomType.EVENT_EPOCH,
                            dataset_id=dataset_id,
                            subject_id=subject_id,
                            session_id=session_id,
                            run_id=run_id,
                            trial_index=sent_idx,
                            signal_ref=SignalRef(
                                file_path="__placeholder__",
                                internal_path=f"/atoms/{atom_id}/signal",
                                shape=(n_channels, n_sent_samples),
                            ),
                            temporal=temporal,
                            channel_ids=channel_ids,
                            n_channels=n_channels,
                            sampling_rate=srate,
                            annotations=annotations,
                            quality=quality,
                            processing_history=ProcessingHistory(
                                steps=[
                                    ProcessingStep(
                                        operation="raw_import",
                                        parameters={
                                            "format": "zuco2_hdf5_eeglab",
                                            "source_file": mat_path.name,
                                            "text_id": text_id,
                                            "sentence_index": sent_idx,
                                            "duration_seconds": round(duration_s, 3),
                                            "signal_unit": "uV",
                                        },
                                    ),
                                ],
                                is_raw=False,  # Preprocessed data
                                version_tag="preprocessed",
                            ),
                            custom_fields={
                                "text_id": text_id,
                                "sentence_index": sent_idx,
                                "reading_duration_s": round(duration_s, 3),
                                "n_words": n_words,
                                "reference": ref_type,
                                **{k: v for k, v in automagic_info.items()
                                   if k not in ("bad_channels",)},
                            },
                        )

                        # Validate
                        val_sig = signal[:, :min(5000, signal.shape[1])]
                        warnings = validate_signal(
                            signal=val_sig,
                            atom_id=atom_id,
                            config=self.pool.config.get("import", {}),
                        )
                        all_warnings.extend(warnings)

                        # Write signal
                        signal_ref = shard_mgr.write_atom_signal(atom_id, signal)
                        atom.signal_ref = signal_ref

                        writer.write_atom(atom)
                        atoms.append(atom)

        logger.info(
            "Imported %s/%s: %d sentence epochs × %d ch @ %.0f Hz",
            subject_id, text_id, len(atoms), n_channels, srate,
        )

        return atoms, ch_infos, all_warnings

    # ------------------------------------------------------------------
    # Main entry: import one subject
    # ------------------------------------------------------------------

    def import_subject(
        self,
        dataset_dir: Path,
        subject_id: str,
        texts: Optional[List[str]] = None,
        max_sentences: Optional[int] = None,
    ) -> List[ImportResult]:
        """Import texts for a subject.

        Args:
            dataset_dir: Root dir (contains Preprocessed/)
            subject_id: Subject code (e.g., 'YAC')
            texts: List of text IDs to import (default: all available)
            max_sentences: Max sentences per text (for testing)

        Returns:
            List of ImportResult, one per text
        """
        dataset_dir = Path(dataset_dir)
        prep_dir = dataset_dir / "Preprocessed"
        sub_dir = prep_dir / subject_id
        dataset_id = self.task_config.dataset_id

        self.pool.ensure_dataset(dataset_id, name=self.task_config.dataset_name)
        self.pool.ensure_subject(dataset_id, subject_id)

        # Find available text files
        prefix = self._file_prefix
        available: Dict[str, Path] = {}
        for mat_path in sub_dir.glob(f"{prefix}_{subject_id}_*_EEG.mat"):
            m = re.search(r"(TSR\d+)", mat_path.name)
            if m:
                available[m.group(1)] = mat_path

        if texts is None:
            texts = sorted(available.keys())
        else:
            texts = [t for t in texts if t in available]

        results = []
        for text_id in texts:
            mat_path = available[text_id]
            atoms, ch_infos, warnings = self._import_text(
                mat_path=mat_path,
                text_id=text_id,
                dataset_id=dataset_id,
                subject_id=subject_id,
                max_sentences=max_sentences,
            )

            if atoms:
                from neuroatom.core.run import RunMeta

                run_meta = RunMeta(
                    run_id="run-01",
                    session_id=f"ses-{text_id.lower()}",
                    subject_id=subject_id,
                    dataset_id=dataset_id,
                    run_index=int(text_id.replace("TSR", "")),
                    task_type=self.task_config.task_type,
                    n_trials=len(atoms),
                )
                self.pool.register_run(run_meta)
                results.append(ImportResult(
                    atoms=atoms,
                    run_meta=run_meta,
                    channel_infos=ch_infos,
                    warnings=warnings,
                ))

        total = sum(len(r.atoms) for r in results)
        logger.info(
            "Subject %s: imported %d texts, %d sentence epochs total.",
            subject_id, len(results), total,
        )

        return results


# Auto-register
register_importer("zuco2_tsr", Zuco2Importer)
