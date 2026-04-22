"""Query DSL → SQL translator for the NeuroAtom atom index.

Translates a declarative query dict (from AssemblyRecipe.query) into
parameterized SQL against the SQLite index.

Query DSL fields:
    dataset_id: List[str]              → OR
    subject_id: List[str]              → OR
    atom_type: List[str]               → OR
    source_version: str                → exact match
    sampling_rate_min: float           → >=
    sampling_rate_max: float           → <=
    duration_seconds_min: float        → >=
    duration_seconds_max: float        → <=
    channels_include: List[str]        → all channels must be present (standard_name)
    channels_min: int                  → n_channels >= X
    quality: Dict                      → overall_status IN [...]
    annotations: List[Dict]            → existential: atom has at least one annotation matching
        - name: str                    → annotation.name
        - value_in: List[str]          → categorical value IN [...]
        - value_not_in: List[str]      → categorical value NOT IN [...]

All channel name matches use standard_name (post-alias-resolution).
Annotation filters use existential semantics: atom matches if at least
one annotation satisfies the filter. Multiple annotation filters are AND-combined.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from neuroatom.index.sqlite_backend import SQLiteBackend

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Builds parameterized SQL queries from the Query DSL dict."""

    def __init__(self, backend: SQLiteBackend):
        self._backend = backend

    def query_atom_ids(self, query: Dict[str, Any]) -> List[str]:
        """Execute a query and return matching atom IDs."""
        sql, params = self.build_sql(query)
        rows = self._backend.execute_query(sql, params)
        return [r["atom_id"] for r in rows]

    def query_count(self, query: Dict[str, Any]) -> int:
        """Count matching atoms without fetching all IDs."""
        sql, params = self.build_sql(query, count_only=True)
        rows = self._backend.execute_query(sql, params)
        return rows[0]["cnt"] if rows else 0

    # Known query DSL keys
    _KNOWN_KEYS = {
        "dataset_id", "subject_id", "atom_type", "modality",
        "source_version", "sampling_rate_min", "sampling_rate_max",
        "duration_seconds_min", "duration_seconds_max",
        "channels_include", "channels_min", "quality", "annotations",
    }

    def build_sql(
        self, query: Dict[str, Any], count_only: bool = False
    ) -> Tuple[str, Tuple]:
        """Translate query dict into SQL + params.

        Returns:
            (sql_string, param_tuple)
        """
        # Warn about unrecognized keys (likely typos)
        unknown_keys = set(query.keys()) - self._KNOWN_KEYS
        if unknown_keys:
            logger.warning(
                "Unknown query keys ignored: %s. "
                "Valid keys: %s",
                sorted(unknown_keys),
                sorted(self._KNOWN_KEYS),
            )

        conditions: List[str] = []
        params: List[Any] = []
        joins: List[str] = []
        annotation_subquery_idx = 0

        # ---- dataset_id ----
        if "dataset_id" in query:
            ds_list = _ensure_list(query["dataset_id"])
            placeholders = ",".join("?" * len(ds_list))
            conditions.append(f"a.dataset_id IN ({placeholders})")
            params.extend(ds_list)

        # ---- subject_id ----
        if "subject_id" in query:
            sub_list = _ensure_list(query["subject_id"])
            placeholders = ",".join("?" * len(sub_list))
            conditions.append(f"a.subject_id IN ({placeholders})")
            params.extend(sub_list)

        # ---- atom_type ----
        if "atom_type" in query:
            type_list = _ensure_list(query["atom_type"])
            placeholders = ",".join("?" * len(type_list))
            conditions.append(f"a.atom_type IN ({placeholders})")
            params.extend(type_list)

        # ---- modality ----
        if "modality" in query:
            mod_list = _ensure_list(query["modality"])
            placeholders = ",".join("?" * len(mod_list))
            conditions.append(f"a.modality IN ({placeholders})")
            params.extend(mod_list)

        # ---- source_version ----
        if "source_version" in query:
            conditions.append("a.source_version = ?")
            params.append(query["source_version"])

        # ---- sampling_rate ----
        if "sampling_rate_min" in query:
            conditions.append("a.sampling_rate >= ?")
            params.append(query["sampling_rate_min"])
        if "sampling_rate_max" in query:
            conditions.append("a.sampling_rate <= ?")
            params.append(query["sampling_rate_max"])

        # ---- duration ----
        if "duration_seconds_min" in query:
            conditions.append("a.duration_seconds >= ?")
            params.append(query["duration_seconds_min"])
        if "duration_seconds_max" in query:
            conditions.append("a.duration_seconds <= ?")
            params.append(query["duration_seconds_max"])

        # ---- channels_min ----
        if "channels_min" in query:
            conditions.append("a.n_channels >= ?")
            params.append(query["channels_min"])

        # ---- quality ----
        if "quality" in query:
            q = query["quality"]
            if "overall_status" in q:
                status_list = _ensure_list(q["overall_status"])
                placeholders = ",".join("?" * len(status_list))
                conditions.append(f"a.quality_status IN ({placeholders})")
                params.extend(status_list)

        # ---- channels_include (standard_name) ----
        if "channels_include" in query:
            ch_list = _ensure_list(query["channels_include"])
            # All listed channels must be present in the atom
            # Use a subquery that counts matches
            for ch_name in ch_list:
                conditions.append(
                    "EXISTS (SELECT 1 FROM channels c "
                    "WHERE c.atom_id = a.atom_id AND c.standard_name = ?)"
                )
                params.append(ch_name)

        # ---- annotations (existential, AND-combined) ----
        if "annotations" in query:
            ann_filters = _ensure_list(query["annotations"])
            for ann_filter in ann_filters:
                sub_conditions = []
                sub_params = []

                if "name" in ann_filter:
                    sub_conditions.append("ann.name = ?")
                    sub_params.append(ann_filter["name"])

                if "value_in" in ann_filter:
                    val_list = _ensure_list(ann_filter["value_in"])
                    placeholders = ",".join("?" * len(val_list))
                    sub_conditions.append(f"ann.value_text IN ({placeholders})")
                    sub_params.extend(val_list)

                if "value_not_in" in ann_filter:
                    val_list = _ensure_list(ann_filter["value_not_in"])
                    placeholders = ",".join("?" * len(val_list))
                    sub_conditions.append(f"ann.value_text NOT IN ({placeholders})")
                    sub_params.extend(val_list)

                if "annotation_type" in ann_filter:
                    sub_conditions.append("ann.annotation_type = ?")
                    sub_params.append(ann_filter["annotation_type"])

                if sub_conditions:
                    where_clause = " AND ".join(sub_conditions)
                    conditions.append(
                        f"EXISTS (SELECT 1 FROM annotations ann "
                        f"WHERE ann.atom_id = a.atom_id AND {where_clause})"
                    )
                    params.extend(sub_params)

        # ---- Build final SQL ----
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        if count_only:
            sql = f"SELECT COUNT(*) as cnt FROM atoms a WHERE {where_clause}"
        else:
            sql = f"SELECT a.atom_id FROM atoms a WHERE {where_clause} ORDER BY a.atom_id"

        return sql, tuple(params)


def _ensure_list(value: Any) -> list:
    """Ensure a value is a list."""
    if isinstance(value, list):
        return value
    return [value]
