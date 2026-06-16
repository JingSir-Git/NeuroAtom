"""Scaffold generator: produce importer boilerplate from interactive prompts.

Generates:
  1. ``neuroatom/importers/<name>.py``  — importer class with TODOs
  2. ``neuroatom/importers/task_configs/<name>.yaml`` — task config skeleton
  3. ``tests/test_e2e_<name>.py``  — test skeleton

Usage::

    from neuroatom.contrib.scaffold import scaffold_importer
    scaffold_importer("my_dataset", output_dir=Path("."))
"""

import logging
import textwrap
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def scaffold_importer(
    name: str,
    output_dir: Path,
    dataset_name: Optional[str] = None,
    task_type: str = "other",
    n_channels: int = 64,
    sampling_rate: float = 256.0,
    signal_unit: str = "uV",
    file_format: str = "mat",
) -> dict:
    """Generate importer boilerplate files.

    Args:
        name: Snake_case importer/dataset ID (e.g. "my_dataset").
        output_dir: Project root directory.
        dataset_name: Human-readable name. Defaults to titlecase of name.
        task_type: Task type (e.g. "motor_imagery", "aad", "erp").
        n_channels: Number of EEG channels.
        sampling_rate: Sampling rate in Hz.
        signal_unit: Signal unit of source data.
        file_format: Source file format (mat, edf, csv, npy, etc.).

    Returns:
        Dict with paths of generated files.
    """
    output_dir = Path(output_dir)
    dataset_name = dataset_name or name.replace("_", " ").title()
    class_name = "".join(w.capitalize() for w in name.split("_")) + "Importer"

    # Paths
    importer_path = output_dir / "neuroatom" / "importers" / f"{name}.py"
    config_path = output_dir / "neuroatom" / "importers" / "task_configs" / f"{name}.yaml"
    test_path = output_dir / "tests" / f"test_e2e_{name}.py"

    generated = {}

    # 1. Importer module
    importer_code = _render_importer(name, class_name, dataset_name, file_format)
    _write_if_new(importer_path, importer_code)
    generated["importer"] = str(importer_path)

    # 2. Task config
    config_yaml = _render_task_config(
        name, dataset_name, task_type, n_channels,
        sampling_rate, signal_unit,
    )
    _write_if_new(config_path, config_yaml)
    generated["task_config"] = str(config_path)

    # 3. Test skeleton
    test_code = _render_test(name, class_name, dataset_name)
    _write_if_new(test_path, test_code)
    generated["test"] = str(test_path)

    logger.info("Scaffolded importer '%s': %d files generated", name, len(generated))
    return generated


def _write_if_new(path: Path, content: str) -> None:
    """Write content to path, refusing to overwrite existing files."""
    if path.exists():
        logger.warning("Skipping %s (already exists)", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Created %s", path)


def _render_importer(
    name: str, class_name: str, dataset_name: str, file_format: str,
) -> str:
    """Render the importer .py template."""
    return textwrap.dedent(f'''\
        """{dataset_name} Importer.

        TODO: Add dataset description, citation, and file format notes.
        """

        import logging
        from pathlib import Path
        from typing import Any, Dict, List, Optional, Tuple

        import numpy as np

        from neuroatom.core.annotation import CategoricalAnnotation
        from neuroatom.core.atom import Atom, TemporalInfo
        from neuroatom.core.channel import ChannelInfo
        from neuroatom.core.enums import AtomType, ChannelType
        from neuroatom.core.signal_ref import SignalRef
        from neuroatom.importers.base import BaseImporter, ImportResult, TaskConfig
        from neuroatom.importers.registry import register_importer
        from neuroatom.storage.pool import Pool
        from neuroatom.utils.channel_names import standardize_channel_name
        from neuroatom.utils.hashing import compute_atom_id
        from neuroatom.utils.unit_convert import convert_to_storage_unit
        from neuroatom.utils.validation import validate_signal

        logger = logging.getLogger(__name__)


        class {class_name}(BaseImporter):
            """Importer for {dataset_name}.

            TODO: Document the expected directory layout and file structure.
            """

            @staticmethod
            def detect(path: Path) -> bool:
                """Return True if *path* looks like {dataset_name} data.

                TODO: Implement format detection heuristic.
                """
                # Example: check for characteristic filename pattern
                return False

            def load_raw(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
                """Load raw data from *path*.

                TODO: Implement loading logic for .{file_format} files.

                Returns:
                    (raw_data, extra_meta) where raw_data is your native object
                    and extra_meta carries declared_unit, paradigm_details, etc.
                """
                raise NotImplementedError("TODO: load {file_format} file")

            def extract_channel_infos(self, raw: Any) -> List[ChannelInfo]:
                """Extract channel information from the raw data object.

                TODO: Build ChannelInfo list from raw data headers.
                """
                raise NotImplementedError("TODO: extract channel info")

            def extract_events(self, raw: Any) -> Optional[np.ndarray]:
                """Extract event array (n_events, 3): [sample, prev_id, event_id].

                TODO: Parse event markers from raw data.
                Return None if the data is pre-segmented (events in metadata).
                """
                return None

            def import_dataset(
                self,
                data_dir: Path,
                *,
                subjects: Optional[List[str]] = None,
                sessions: Optional[List[str]] = None,
            ) -> List[ImportResult]:
                """Import the full dataset from *data_dir*.

                TODO: Iterate over subjects/sessions/runs and call self.import_run()
                for each. This is your main entry point.

                Example skeleton::

                    results = []
                    dataset_id = self.task_config.dataset_id
                    self.pool.register_dataset(DatasetMeta(
                        dataset_id=dataset_id,
                        name="{dataset_name}",
                    ))

                    for subj_dir in sorted(data_dir.iterdir()):
                        subject_id = subj_dir.name
                        self.pool.register_subject(SubjectMeta(
                            subject_id=subject_id,
                            dataset_id=dataset_id,
                        ))
                        # ... iterate sessions, runs
                        # result = self.import_run(file, subject_id, session_id, run_id, atomizer)
                        # results.append(result)

                    return results
                """
                raise NotImplementedError("TODO: implement dataset iteration")


        # Register this importer so it can be discovered by format name
        register_importer("{name}", {class_name})
    ''')


def _render_task_config(
    name: str, dataset_name: str, task_type: str,
    n_channels: int, sampling_rate: float, signal_unit: str,
) -> str:
    """Render the task config YAML template."""
    return textwrap.dedent(f"""\
        # {dataset_name}
        # TODO: Add citation and dataset description
        #
        # TODO: Document file structure and data layout

        dataset_id: "{name}"
        dataset_name: "{dataset_name}"
        task_type: "{task_type}"

        trial_definition:
          mode: "event_based"   # or "pre_segmented", "continuous"
          tmin: 0.0             # TODO: set epoch start (seconds relative to event)
          tmax: 4.0             # TODO: set epoch end

        class_labels:
          1: "class_a"          # TODO: map event codes to label names
          2: "class_b"

        signal_unit: "{signal_unit}"

        channel_names: []       # TODO: list channel names or leave empty to read from file

        channel_type_overrides: {{}}
        exclude_channels: []

        custom:
          n_subjects: 0         # TODO: fill in dataset info
          n_channels: {n_channels}
          sampling_rate: {sampling_rate}
          license: "TODO"
          citation: "TODO"
    """)


def _render_test(name: str, class_name: str, dataset_name: str) -> str:
    """Render the test skeleton."""
    return textwrap.dedent(f'''\
        """End-to-end tests for {dataset_name} importer.

        TODO: Implement test fixtures and test cases.
        """

        import pytest
        from pathlib import Path

        from neuroatom.importers.base import TaskConfig
        from neuroatom.storage.pool import Pool


        @pytest.fixture
        def pool(tmp_path):
            """Create a temporary pool for testing."""
            return Pool.create(tmp_path / "test_pool")


        @pytest.fixture
        def task_config():
            """Load the task config for {name}."""
            return TaskConfig.builtin("{name}")


        @pytest.mark.skip(reason="TODO: provide test data or mock")
        class TestImport{class_name.replace("Importer", "")}:

            def test_import_produces_atoms(self, pool, task_config, tmp_path):
                """Verify that import produces non-empty atom list."""
                # TODO: Set up test data, run importer, check results
                pass

            def test_atom_metadata_valid(self, pool, task_config, tmp_path):
                """Verify atom metadata fields are correctly populated."""
                # TODO: Check dataset_id, subject_id, channel_ids, etc.
                pass

            def test_signal_shape(self, pool, task_config, tmp_path):
                """Verify signal array shape matches channel count and duration."""
                # TODO: Load signal via SignalRef and check shape
                pass

            def test_signal_unit_is_uv(self, pool, task_config, tmp_path):
                """Verify signal is stored in µV."""
                # TODO: Check atom.signal_unit == "uV"
                pass
    ''')
