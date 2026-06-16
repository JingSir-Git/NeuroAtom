"""Quality report: generate human-readable and machine-readable quality reports.

Produces both terminal output and JSON export for a dataset's quality
assessment against the admission policy.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from neuroatom.quality.gate import QualityGate, QualityReport
from neuroatom.storage.pool import Pool

logger = logging.getLogger(__name__)


def format_report(report: QualityReport) -> str:
    """Format a quality report as human-readable text."""
    lines = []
    tier_str = report.tier.value.upper() if report.tier else "UNRATED"

    lines.append(f"{'=' * 60}")
    lines.append(f"  Quality Report: {report.dataset_id}")
    lines.append(f"{'=' * 60}")
    lines.append(f"  Tier:      {tier_str}")
    lines.append(f"  Atoms:     {report.n_atoms}")
    lines.append(f"  Subjects:  {report.n_subjects}")
    lines.append("")

    # Stats
    lines.append("  Signal Stats:")
    lines.append(f"    Channels:  {report.stats.get('min_channels', '?')}"
                 f" – {report.stats.get('max_channels', '?')}"
                 f" (avg {report.stats.get('avg_channels', '?')})")
    lines.append(f"    Bad ch ratio (max): {report.stats.get('max_bad_channel_ratio', '?')}")
    lines.append(f"    NaN atoms:     {report.stats.get('nan_atom_count', 0)}")
    lines.append(f"    Flatline atoms: {report.stats.get('flatline_atom_count', 0)}")
    lines.append("")

    # Metadata completeness
    lines.append("  Metadata Completeness:")
    for key in ["has_labels_ratio", "has_standard_channels_ratio",
                "has_electrode_locations_ratio", "has_processing_history_ratio",
                "has_stimulus_ref_ratio"]:
        val = report.stats.get(key, 0)
        label = key.replace("has_", "").replace("_ratio", "").replace("_", " ").title()
        lines.append(f"    {label}: {val:.0%}")
    lines.append("")

    # Tier checks
    for tr in report.tier_results:
        icon = "✅" if tr.passed else "❌"
        lines.append(f"  {icon} {tr.tier.value.upper()}")
        for check_name, passed in tr.checks.items():
            mark = "✓" if passed else "✗"
            detail = tr.details.get(check_name, "")
            suffix = f"  ({detail})" if detail else ""
            lines.append(f"      {mark} {check_name}{suffix}")
        lines.append("")

    # Warnings
    if report.warnings:
        lines.append("  Warnings:")
        for w in report.warnings:
            lines.append(f"    ⚠ {w}")
        lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def generate_quality_report(
    pool: Pool,
    dataset_id: str,
    json_path: Optional[Path] = None,
) -> QualityReport:
    """Generate a quality report for a dataset.

    Args:
        pool: The pool containing the dataset.
        dataset_id: Dataset to assess.
        json_path: If provided, write JSON report to this path.

    Returns:
        The QualityReport object.
    """
    gate = QualityGate(pool)
    report = gate.assess_dataset(dataset_id)

    if json_path:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Quality report written to %s", json_path)

    return report
