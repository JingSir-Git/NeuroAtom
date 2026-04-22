"""NeuroAtom: Universal EEG Resource Pool.

Decompose heterogeneous EEG datasets into standardized atomic units
with rich metadata, enabling unified querying and ML pipeline integration.

Quick-start::

    from neuroatom import Pool, Indexer, QueryBuilder
    from neuroatom import DatasetAssembler, AssemblyRecipe, LabelSpec
    from neuroatom import AtomDataset  # requires pip install neuroatom[torch]
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
]
