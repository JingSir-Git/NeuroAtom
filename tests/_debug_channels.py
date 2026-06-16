"""Debug channel names from both datasets."""
import sys
from pathlib import Path
from neuroatom.storage.pool import Pool
from neuroatom.importers.base import TaskConfig
from neuroatom.importers.registry import get_importer
from neuroatom.index.indexer import Indexer
import tempfile, shutil

pool_dir = Path(tempfile.mkdtemp(prefix="ch_debug_"))
pool = Pool.create(pool_dir)

# Import one atom from each dataset
cfg_bci = TaskConfig.builtin("bci_comp_iv_2a")
imp_bci = get_importer("bci_comp_iv_2a", pool, cfg_bci)
imp_bci.import_subject(mat_path=Path(r"C:\Data\BCI_Competition\A01T.mat"), subject_id="A01")

cfg_omi = TaskConfig.builtin("openbmi_mi")
imp_omi = get_importer("openbmi_mi", pool, cfg_omi)
imp_omi.import_subject(
    mat_path=Path(r"\\wsqlab\share\JCH\OpenBMI\MI\sess01_subj01_EEG_MI.mat"),
    subject_id="S01",
)

indexer = Indexer(pool)
n = indexer.reindex_all()
print(f"\nIndexed {n} atoms")

# Check channel names from first atom of each dataset
from neuroatom.storage.metadata_store import AtomJSONLReader
from neuroatom.storage import paths as P

for ds_id in ("bci_comp_iv_2a", "openbmi_mi"):
    # Find a JSONL file for this dataset
    ds_dir = pool_dir / "data" / ds_id
    jsonl_files = list(ds_dir.rglob("*.jsonl"))
    if jsonl_files:
        reader = AtomJSONLReader(jsonl_files[0])
        atom = next(reader.iter_atoms())
        print(f"\n{ds_id}:")
        print(f"  channel_ids ({len(atom.channel_ids)}): {atom.channel_ids[:10]}...")
        print(f"  channels[0] info: {atom.channels[0] if atom.channels else 'N/A'}")

indexer.close()
shutil.rmtree(pool_dir, ignore_errors=True)
