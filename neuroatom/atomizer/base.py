"""BaseAtomizer: abstract base for data decomposition engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from neuroatom.core.atom import Atom
from neuroatom.core.channel import ChannelInfo
from neuroatom.core.run import RunMeta
from neuroatom.importers.base import TaskConfig


class BaseAtomizer(ABC):
    """Abstract base class for atomizers.

    An atomizer takes a raw recording (or an MNE Raw object) along with
    event markers and task configuration, and decomposes it into a list
    of Atom objects.
    """

    @abstractmethod
    def atomize(
        self,
        raw: Any,
        events: Optional[np.ndarray],
        task_config: TaskConfig,
        run_meta: RunMeta,
        channel_infos: List[ChannelInfo],
    ) -> List[Atom]:
        """Decompose a raw recording into atoms.

        Args:
            raw: The raw data object (e.g., mne.io.Raw).
            events: Event array of shape (n_events, 3) or None.
            task_config: Parsed task configuration.
            run_meta: Metadata for this run.
            channel_infos: Per-channel metadata.

        Returns:
            List of Atom objects (without signal_ref populated yet —
            that is filled in by the importer after writing to HDF5).
        """
        ...
