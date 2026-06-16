"""NeuroAtom: Universal EEG Resource Pool.

Decompose heterogeneous EEG datasets into standardized atomic units
with rich metadata, enabling unified querying and ML pipeline integration.

Quick-start — single subject (5 lines)::

    import neuroatom as na
    loader = na.quickload("bci_comp_iv_2a", "data/A01T.mat")

Cross-subject / cross-dataset / cross-task (~10 lines)::

    train, val, test = na.multiload(
        sources=[
            {"dataset": "openbmi_mi",     "path": "OpenBMI/MI", "subjects": ["S01", "S02"]},
            {"dataset": "bci_comp_iv_2a", "path": "data/A01T.mat"},
        ],
        target_channels=["C3", "Cz", "C4"],
        target_srate=250,
        label_field="mi_class",
    )

For full control, use the lower-level API::

    from neuroatom import Pool, Indexer, QueryBuilder
    from neuroatom import DatasetAssembler, AssemblyRecipe, LabelSpec
"""

__version__ = "0.1.0"

# ── Core data models ─────────────────────────────────────────────────────
from neuroatom.core.atom import Atom, AtomRelation, TemporalInfo
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.electrode import ElectrodeLocation
from neuroatom.core.enums import (
    AtomType,
    ChannelStatus,
    ChannelType,
    ErrorHandling,
    NormalizationMethod,
    NormalizationScope,
    QualityTier,
    SplitStrategy,
)
from neuroatom.core.quality import QualityInfo
from neuroatom.core.recipe import AssemblyRecipe, LabelSpec
from neuroatom.core.signal_ref import SignalRef

# ── Storage ──────────────────────────────────────────────────────────────
from neuroatom.storage.pool import Pool

# ── Indexing & Query ─────────────────────────────────────────────────────
from neuroatom.index.indexer import Indexer
from neuroatom.index.query import QueryBuilder

# ── Assembly ─────────────────────────────────────────────────────────────
from neuroatom.assembler.dataset_assembler import DatasetAssembler

# ── Multi-modal assembly ─────────────────────────────────────────────
from neuroatom.assembler.multimodal_assembler import MultiModalAssembler
from neuroatom.core.multimodal_recipe import (
    ModalityPipelineConfig,
    MultiModalRecipe,
)

# ── Cross-pool federation ─────────────────────────────────────────────
from neuroatom.index.federation import (
    FederatedPool,
    FederatedQueryBuilder,
)
from neuroatom.assembler.federated_assembler import FederatedAssembler

# ── Import provenance ────────────────────────────────────────────────
from neuroatom.index.import_log import log_import, get_import_history

# ── Importer base (for building custom importers) ────────────────────
from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig

# ── High-level convenience API ────────────────────────────────────────
from neuroatom.quick import quickload, multiload
from neuroatom.importers.registry import get_importer_class, list_formats

__all__ = [
    # version
    "__version__",
    # core models
    "Atom",
    "AtomRelation",
    "TemporalInfo",
    "ChannelInfo",
    "ElectrodeLocation",
    "QualityInfo",
    "SignalRef",
    # enums
    "AtomType",
    "ChannelStatus",
    "ChannelType",
    "ErrorHandling",
    "NormalizationMethod",
    "NormalizationScope",
    "QualityTier",
    "SplitStrategy",
    # recipe
    "AssemblyRecipe",
    "LabelSpec",
    # storage
    "Pool",
    # index
    "Indexer",
    "QueryBuilder",
    # assembly
    "DatasetAssembler",
    # multi-modal assembly
    "MultiModalAssembler",
    "MultiModalRecipe",
    "ModalityPipelineConfig",
    # cross-pool federation
    "FederatedPool",
    "FederatedQueryBuilder",
    "FederatedAssembler",
    # import provenance
    "log_import",
    "get_import_history",
    # importer base
    "BaseImporter",
    "ImportResult",
    "TaskConfig",
    # convenience API
    "quickload",
    "multiload",
    "get_importer_class",
    "list_formats",
]
