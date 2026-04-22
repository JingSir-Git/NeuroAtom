"""LabelEncoder: extract and encode labels from atom annotations.

Supports multi-task learning via multiple LabelSpec entries in
AssemblyRecipe.label_fields.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from neuroatom.core.annotation import AnnotationUnion, CategoricalAnnotation, NumericAnnotation
from neuroatom.core.recipe import LabelSpec

logger = logging.getLogger(__name__)


class LabelEncoder:
    """Extract and encode labels from atom annotations.

    Usage:
        encoder = LabelEncoder(label_specs=[
            LabelSpec(annotation_name="mi_class", output_key="label"),
            LabelSpec(annotation_name="subject_id", output_key="domain", encoding="ordinal"),
        ])

        # Build encoding from all atoms (pass 1)
        for atom in all_atoms:
            encoder.fit_atom(atom.annotations)
        encoder.finalize()

        # Encode (pass 2 or inline)
        labels = encoder.encode(atom.annotations)
        # → {"label": 0, "domain": 3}
    """

    def __init__(self, label_specs: List[LabelSpec]):
        self._specs = label_specs

        # Per output_key: value → int mapping
        self._encodings: Dict[str, Dict[str, int]] = {}
        # Per output_key: seen values (for auto encoding)
        self._seen_values: Dict[str, set] = {
            spec.output_key: set() for spec in label_specs
        }
        self._finalized = False

    def fit_atom(self, annotations: List[AnnotationUnion]) -> None:
        """Accumulate label values from one atom's annotations."""
        for spec in self._specs:
            value = self._extract_value(annotations, spec)
            if value is not None:
                self._seen_values[spec.output_key].add(str(value))

    def finalize(self) -> None:
        """Build encoding maps from accumulated values."""
        for spec in self._specs:
            key = spec.output_key
            seen = sorted(self._seen_values[key])

            if not seen and spec.encoding != "raw":
                logger.warning(
                    "No values found for label spec '%s' (annotation_name='%s'). "
                    "Check that the annotation name matches an annotation in the queried atoms. "
                    "All labels for this key will be -1 (missing).",
                    key, spec.annotation_name,
                )

            if spec.label_mapping:
                # Use provided mapping
                encoding = {}
                idx = 0
                for unified_label, aliases in spec.label_mapping.items():
                    encoding[unified_label] = idx
                    for alias in aliases:
                        encoding[alias] = idx
                    idx += 1
                self._encodings[key] = encoding
            elif spec.encoding == "raw":
                self._encodings[key] = {}
            else:
                # Auto ordinal encoding
                self._encodings[key] = {v: i for i, v in enumerate(seen)}

        self._finalized = True
        logger.info(
            "LabelEncoder finalized: %s",
            {k: len(v) for k, v in self._encodings.items()},
        )

    def encode(
        self,
        annotations: List[AnnotationUnion],
        subject_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encode labels for one atom.

        Args:
            annotations: Atom's annotation list.
            subject_id: Subject ID (used if annotation_name is "subject_id").

        Returns:
            Dict mapping output_key → encoded value (int or float or str).
        """
        result = {}

        for spec in self._specs:
            # Special case: subject_id as label
            if spec.annotation_name == "subject_id" and subject_id is not None:
                value = subject_id
            else:
                value = self._extract_value(annotations, spec)

            if value is None:
                result[spec.output_key] = -1  # Missing label sentinel
                continue

            value_str = str(value)
            encoding = self._encodings.get(spec.output_key, {})

            if spec.encoding == "raw":
                result[spec.output_key] = value
            elif spec.encoding == "onehot":
                n_classes = len(encoding)
                onehot = np.zeros(n_classes, dtype=np.float32)
                if value_str in encoding:
                    onehot[encoding[value_str]] = 1.0
                result[spec.output_key] = onehot
            else:
                # ordinal or auto
                if value_str in encoding:
                    result[spec.output_key] = encoding[value_str]
                else:
                    logger.warning(
                        "Unknown label value '%s' for '%s'. Using -1.",
                        value_str, spec.output_key,
                    )
                    result[spec.output_key] = -1

        return result

    @property
    def encodings(self) -> Dict[str, Dict[str, int]]:
        return self._encodings

    def _extract_value(
        self, annotations: List[AnnotationUnion], spec: LabelSpec
    ) -> Optional[Any]:
        """Extract the first matching annotation value."""
        for ann in annotations:
            if ann.name == spec.annotation_name:
                if isinstance(ann, CategoricalAnnotation):
                    return ann.value
                elif isinstance(ann, NumericAnnotation):
                    return ann.numeric_value
                elif hasattr(ann, "text_value"):
                    return ann.text_value
        return None
