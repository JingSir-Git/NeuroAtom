"""Quick smoke test for ChineseEEG-2 importer (not part of CI suite)."""
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from neuroatom.storage.pool import Pool
from neuroatom.importers.chinese_eeg2 import ChineseEEG2Importer

DATA_ROOT = Path(r"C:\Data\ChineseEEG-2\PassiveListening")

def main():
    td = tempfile.mkdtemp()
    pool = Pool.create(td)
    imp = ChineseEEG2Importer(pool, task="listening")
    results = imp.import_dataset(
        DATA_ROOT,
        subjects=["01"],
        sessions=["littleprince"],
        max_runs=1,
    )
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    r = results[0]
    print(f"Run: {r.run_meta.run_id}, atoms: {r.n_atoms}, warnings: {len(r.warnings)}")

    a0 = r.atoms[0]
    print(f"First atom: {a0.atom_id[:16]}...")
    print(f"  channels={a0.n_channels}, sr={a0.sampling_rate}")
    print(f"  onset={a0.temporal.onset_sample}, dur_samples={a0.temporal.duration_samples}, dur_s={a0.temporal.duration_seconds:.3f}")
    print(f"  annotations: {[(a.name, getattr(a, 'value', getattr(a, 'numeric_value', None))) for a in a0.annotations]}")
    print(f"  signal_ref: {a0.signal_ref.file_path} shape={a0.signal_ref.shape}")
    print(f"  custom: {a0.custom_fields}")
    print(f"  processing: raw={a0.processing_history.is_raw}, tag={a0.processing_history.version_tag}")

    # Verify signal can be read back
    from neuroatom.storage.signal_store import ShardManager
    mgr = ShardManager(
        pool_root=pool.root,
        dataset_id=a0.dataset_id,
        subject_id=a0.subject_id,
        session_id=a0.session_id,
        run_id=a0.run_id,
    )
    sig = mgr.read_atom_signal(a0.signal_ref)
    print(f"  signal_read: shape={sig.shape}, dtype={sig.dtype}")
    print(f"  signal range: [{sig.min():.6f}, {sig.max():.6f}]")
    mgr.close()

    print(f"\nSUCCESS: {r.n_atoms} sentence atoms imported and verified.")


if __name__ == "__main__":
    main()
