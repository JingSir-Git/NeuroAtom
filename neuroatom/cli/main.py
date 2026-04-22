"""NeuroAtom CLI: command-line interface for pool management.

Commands:
- init: Create a new pool
- import: Import a dataset into the pool
- reindex: Build/rebuild the SQLite index (alias: index)
- stats: Show pool statistics (atom count, label distribution, channel coverage, srate histogram)
- query: Query atoms from the pool
- assemble: Assemble a dataset from a recipe
- export: Export query results to various formats
- info: Shorthand summary
- migrate: Check and apply schema migrations
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from neuroatom.storage.pool import Pool

logger = logging.getLogger("neuroatom")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool):
    """NeuroAtom: Universal EEG Resource Pool."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.argument("pool_path", type=click.Path())
@click.option("--config", "-c", type=click.Path(exists=True), help="Custom pool config YAML.")
def init(pool_path: str, config: Optional[str]):
    """Initialize a new NeuroAtom resource pool."""
    overrides = None
    if config:
        with open(config, "r") as f:
            overrides = yaml.safe_load(f)

    pool = Pool.create(Path(pool_path), config_overrides=overrides)
    click.echo(f"Pool created at: {pool.root}")


@cli.command()
@click.argument("pool_path", type=click.Path(exists=True))
def info(pool_path: str):
    """Shorthand: show pool overview."""
    pool = Pool(Path(pool_path))
    datasets = pool.list_datasets()

    from neuroatom.storage.migration import get_pool_version
    version = get_pool_version(pool.root)

    click.echo(f"Pool: {pool.root}")
    click.echo(f"Schema version: {version}")
    click.echo(f"Datasets: {len(datasets)}")

    for ds_id in datasets:
        meta = pool.get_dataset_meta(ds_id)
        subjects = pool.list_subjects(ds_id)
        click.echo(f"  [{ds_id}] {meta.name}")
        click.echo(f"    Subjects: {len(subjects)}")
        click.echo(f"    Tasks: {', '.join(meta.task_types)}")

    # Quick index atom count
    try:
        from neuroatom.index.sqlite_backend import SQLiteBackend
        from neuroatom.storage import paths as P
        db_path = P.index_db_path(pool.root)
        if db_path.exists():
            backend = SQLiteBackend(db_path)
            backend.connect()
            total = backend.count_atoms()
            click.echo(f"\nIndexed atoms: {total}")
            backend.close()
    except Exception:
        pass


@cli.command()
@click.argument("pool_path", type=click.Path(exists=True))
@click.option("--dataset", "-d", help="Restrict stats to this dataset.")
@click.option("--json-out", "-j", type=click.Path(), help="Write stats to JSON file.")
def stats(pool_path: str, dataset: Optional[str], json_out: Optional[str]):
    """Show detailed pool statistics (§6.7): atom count, label distribution, channel coverage, sampling rates."""
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.sqlite_backend import SQLiteBackend
    from neuroatom.storage import paths as P

    pool = Pool(Path(pool_path))

    db_path = P.index_db_path(pool.root)
    if not db_path.exists():
        click.echo("Error: Index not found. Run 'neuroatom reindex' first.")
        return

    backend = SQLiteBackend(db_path)
    backend.connect()
    indexer = Indexer(pool, backend=backend)

    stats_data = indexer.get_stats()

    # Channel coverage: distinct standard_name across all atoms
    conn = backend.conn
    ch_rows = conn.execute(
        "SELECT standard_name, COUNT(DISTINCT atom_id) as cnt FROM channels "
        "WHERE standard_name IS NOT NULL GROUP BY standard_name ORDER BY cnt DESC"
    ).fetchall()
    channel_coverage = {r["standard_name"]: r["cnt"] for r in ch_rows}
    stats_data["channel_coverage"] = channel_coverage

    # Sampling rate histogram
    srate_rows = conn.execute(
        "SELECT sampling_rate, COUNT(*) as cnt FROM atoms GROUP BY sampling_rate ORDER BY sampling_rate"
    ).fetchall()
    srate_hist = {r["sampling_rate"]: r["cnt"] for r in srate_rows}
    stats_data["sampling_rate_histogram"] = srate_hist

    # Duration stats
    dur_row = conn.execute(
        "SELECT MIN(duration_seconds) as dmin, MAX(duration_seconds) as dmax, "
        "AVG(duration_seconds) as davg FROM atoms"
    ).fetchone()
    stats_data["duration_stats"] = {
        "min_seconds": dur_row["dmin"],
        "max_seconds": dur_row["dmax"],
        "avg_seconds": round(dur_row["davg"], 3) if dur_row["davg"] else None,
    }

    # Display
    click.echo(f"Total atoms: {stats_data['total_atoms']}")
    click.echo(f"\nPer dataset:")
    for ds, count in stats_data["per_dataset"].items():
        click.echo(f"  [{ds}] {count} atoms")
    click.echo(f"\nPer atom type:")
    for atype, count in stats_data["per_type"].items():
        click.echo(f"  {atype}: {count}")
    click.echo(f"\nLabel distribution:")
    for label, count in sorted(stats_data["label_distribution"].items()):
        click.echo(f"  {label}: {count}")
    click.echo(f"\nSampling rates:")
    for srate, count in srate_hist.items():
        click.echo(f"  {srate} Hz: {count} atoms")
    click.echo(f"\nDuration: min={dur_row['dmin']:.2f}s, max={dur_row['dmax']:.2f}s, avg={dur_row['davg']:.3f}s")
    click.echo(f"\nChannel coverage ({len(channel_coverage)} unique):")
    for ch, count in list(channel_coverage.items())[:20]:
        click.echo(f"  {ch}: {count} atoms")
    if len(channel_coverage) > 20:
        click.echo(f"  ... and {len(channel_coverage) - 20} more")

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        click.echo(f"\nStats written to {json_out}")

    indexer.close()


def _run_index(pool_path: str, dataset: Optional[str], incremental: bool):
    """Shared logic for index/reindex commands."""
    from neuroatom.index.indexer import Indexer

    pool = Pool(Path(pool_path))
    indexer = Indexer(pool)

    if incremental:
        count = indexer.index_incremental()
        click.echo(f"Incremental index: {count} new atoms.")
    elif dataset:
        count = indexer.reindex_dataset(dataset)
        click.echo(f"Reindexed dataset '{dataset}': {count} atoms.")
    else:
        count = indexer.reindex_all()
        click.echo(f"Full reindex: {count} atoms.")

    indexer.close()


@cli.command("index")
@click.argument("pool_path", type=click.Path(exists=True))
@click.option("--dataset", "-d", help="Index only this dataset.")
@click.option("--incremental", "-i", is_flag=True, help="Only process changed files.")
def index_cmd(pool_path: str, dataset: Optional[str], incremental: bool):
    """Build or rebuild the SQLite index."""
    _run_index(pool_path, dataset, incremental)


@cli.command("reindex")
@click.argument("pool_path", type=click.Path(exists=True))
@click.option("--dataset", "-d", help="Reindex only this dataset.")
@click.option("--incremental", "-i", is_flag=True, help="Only process changed files.")
def reindex_cmd(pool_path: str, dataset: Optional[str], incremental: bool):
    """Alias for 'index': force rebuild of SQLite index from JSONL source of truth."""
    _run_index(pool_path, dataset, incremental)


@cli.command()
@click.argument("pool_path", type=click.Path(exists=True))
@click.argument("query_yaml", type=click.Path(exists=True))
def query(pool_path: str, query_yaml: str):
    """Query atoms from the pool using a YAML query file."""
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder

    pool = Pool(Path(pool_path))
    indexer = Indexer(pool)

    with open(query_yaml, "r") as f:
        query_dict = yaml.safe_load(f)

    qb = QueryBuilder(indexer.backend)
    atom_ids = qb.query_atom_ids(query_dict)
    count = len(atom_ids)

    click.echo(f"Found {count} matching atoms.")
    if count <= 20:
        for aid in atom_ids:
            click.echo(f"  {aid}")

    indexer.close()


@cli.command()
@click.argument("pool_path", type=click.Path(exists=True))
@click.argument("recipe_yaml", type=click.Path(exists=True))
@click.option("--cache-dir", "-o", type=click.Path(), help="Output cache directory.")
def assemble(pool_path: str, recipe_yaml: str, cache_dir: Optional[str]):
    """Assemble a dataset from a recipe YAML file."""
    from neuroatom.assembler.dataset_assembler import DatasetAssembler
    from neuroatom.core.recipe import AssemblyRecipe
    from neuroatom.index.indexer import Indexer

    pool = Pool(Path(pool_path))
    indexer = Indexer(pool)

    with open(recipe_yaml, "r") as f:
        recipe_dict = yaml.safe_load(f)

    recipe = AssemblyRecipe.model_validate(recipe_dict)

    assembler = DatasetAssembler(pool, indexer)
    result = assembler.assemble(
        recipe,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    click.echo(f"Assembly complete:")
    click.echo(f"  Train: {len(result.train_samples)}")
    click.echo(f"  Val:   {len(result.val_samples)}")
    click.echo(f"  Test:  {len(result.test_samples)}")
    click.echo(f"  Errors: {result.n_errors}")
    click.echo(f"  Skipped: {result.n_skipped}")

    indexer.close()


@cli.command("import")
@click.argument("pool_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("task_config_yaml", type=click.Path(exists=True))
@click.option("--subject", "-s", required=True, help="Subject ID.")
@click.option("--session", default="ses-01", help="Session ID.")
@click.option("--run", default="run-01", help="Run ID.")
@click.option("--format", "fmt", default=None, help="Force format (auto-detect if omitted).")
@click.option("--atomizer", default="trial", type=click.Choice(["trial", "event", "window"]), help="Atomizer type.")
def import_cmd(
    pool_path: str, data_path: str, task_config_yaml: str,
    subject: str, session: str, run: str, fmt: Optional[str], atomizer: str,
):
    """Import a single run into the pool."""
    from neuroatom.importers.base import TaskConfig
    from neuroatom.importers.registry import detect_format, get_importer

    # Ensure all importers are registered
    import neuroatom.importers.mne_generic  # noqa: F401
    import neuroatom.importers.mat  # noqa: F401
    import neuroatom.importers.eeglab  # noqa: F401

    pool = Pool(Path(pool_path))
    task_config = TaskConfig.from_yaml(Path(task_config_yaml))

    # Auto-detect or use specified format
    if fmt is None:
        fmt = detect_format(Path(data_path))
        if fmt is None:
            click.echo("Error: Could not auto-detect format. Use --format.")
            return
        click.echo(f"Detected format: {fmt}")

    importer = get_importer(fmt, pool, task_config)

    # Register dataset + subject + session if not exists
    from neuroatom.core.dataset_meta import DatasetMeta
    from neuroatom.core.subject import SubjectMeta
    from neuroatom.core.session import SessionMeta

    try:
        pool.register_dataset(DatasetMeta(
            dataset_id=task_config.dataset_id,
            name=task_config.dataset_name,
            task_types=[task_config.task_type],
            n_subjects=1,
        ))
    except Exception:
        pass  # Already exists

    try:
        pool.register_subject(SubjectMeta(
            subject_id=subject, dataset_id=task_config.dataset_id,
        ))
    except Exception:
        pass

    try:
        pool.register_session(SessionMeta(
            session_id=session, subject_id=subject,
            dataset_id=task_config.dataset_id,
        ))
    except Exception:
        pass

    # Build atomizer
    if atomizer == "trial":
        from neuroatom.atomizer.trial import TrialAtomizer
        atom_obj = TrialAtomizer()
    elif atomizer == "event":
        from neuroatom.atomizer.event import EventAtomizer
        atom_obj = EventAtomizer()
    else:
        from neuroatom.atomizer.window import WindowAtomizer
        atom_obj = WindowAtomizer()

    result = importer.import_run(
        path=Path(data_path),
        subject_id=subject,
        session_id=session,
        run_id=run,
        atomizer=atom_obj,
    )

    click.echo(f"Import complete: {result.n_atoms} atoms.")
    if result.warnings:
        click.echo(f"Warnings: {len(result.warnings)}")
    if result.errors:
        click.echo(f"Errors: {len(result.errors)}")


@cli.command()
@click.argument("pool_path", type=click.Path(exists=True))
@click.argument("query_yaml", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path.")
@click.option(
    "--format", "fmt",
    type=click.Choice(["atom_ids", "jsonl", "numpy", "csv"]),
    default="atom_ids",
    help="Export format.",
)
def export(pool_path: str, query_yaml: str, output: Optional[str], fmt: str):
    """Export query results to various formats."""
    from neuroatom.index.indexer import Indexer
    from neuroatom.index.query import QueryBuilder
    from neuroatom.storage.metadata_store import AtomJSONLReader
    from neuroatom.storage import paths as P
    import numpy as np

    pool = Pool(Path(pool_path))
    indexer = Indexer(pool)

    with open(query_yaml, "r") as f:
        query_dict = yaml.safe_load(f)

    qb = QueryBuilder(indexer.backend)
    atom_ids = qb.query_atom_ids(query_dict)
    click.echo(f"Found {len(atom_ids)} matching atoms.")

    if not atom_ids:
        indexer.close()
        return

    if fmt == "atom_ids":
        if output:
            with open(output, "w") as f:
                for aid in atom_ids:
                    f.write(aid + "\n")
            click.echo(f"Atom IDs written to {output}")
        else:
            for aid in atom_ids:
                click.echo(aid)

    elif fmt == "jsonl":
        # Fetch atom metadata from index, re-export as JSONL
        conn = indexer.backend.conn
        rows = conn.execute(
            "SELECT atom_id, dataset_id, subject_id, session_id, run_id FROM atoms "
            f"WHERE atom_id IN ({','.join('?' * len(atom_ids))})",
            tuple(atom_ids),
        ).fetchall()

        out_path = output or "export.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                line = json.dumps(dict(r), ensure_ascii=False)
                f.write(line + "\n")
        click.echo(f"JSONL written to {out_path}")

    elif fmt == "numpy":
        # Export signal data as a single .npz file
        from neuroatom.storage.signal_store import ShardManager

        conn = indexer.backend.conn
        rows = conn.execute(
            "SELECT atom_id, dataset_id, subject_id, session_id, run_id, "
            "signal_file_path, shard_index FROM atoms "
            f"WHERE atom_id IN ({','.join('?' * len(atom_ids))})",
            tuple(atom_ids),
        ).fetchall()

        # Load all atoms from JSONL to get signal_refs
        from neuroatom.core.signal_ref import SignalRef
        run_groups = {}
        for r in rows:
            rk = f"{r['dataset_id']}|{r['subject_id']}|{r['session_id']}|{r['run_id']}"
            if rk not in run_groups:
                run_groups[rk] = []
            run_groups[rk].append(r["atom_id"])

        signals = {}
        target_set = set(atom_ids)
        for rk, aids in run_groups.items():
            parts = rk.split("|")
            jsonl_path = P.atoms_jsonl_path(pool.root, *parts)
            reader = AtomJSONLReader(jsonl_path)
            for atom in reader.iter_atoms():
                if atom.atom_id in target_set:
                    try:
                        sig = ShardManager.static_read(pool.root, atom.signal_ref)
                        signals[atom.atom_id] = sig
                    except Exception as e:
                        click.echo(f"Warning: could not read {atom.atom_id}: {e}")

        out_path = output or "export.npz"
        np.savez_compressed(out_path, **signals)
        click.echo(f"Numpy archive written to {out_path} ({len(signals)} signals)")

    elif fmt == "csv":
        conn = indexer.backend.conn
        rows = conn.execute(
            "SELECT atom_id, atom_type, dataset_id, subject_id, session_id, run_id, "
            "n_channels, sampling_rate, duration_seconds, quality_status FROM atoms "
            f"WHERE atom_id IN ({','.join('?' * len(atom_ids))})",
            tuple(atom_ids),
        ).fetchall()

        import csv
        out_path = output or "export.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "atom_id", "atom_type", "dataset_id", "subject_id",
                "session_id", "run_id", "n_channels", "sampling_rate",
                "duration_seconds", "quality_status",
            ])
            for r in rows:
                writer.writerow(list(r))
        click.echo(f"CSV written to {out_path}")

    indexer.close()


@cli.command()
@click.argument("pool_path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show pending migrations without applying.")
def migrate(pool_path: str, dry_run: bool):
    """Check and apply schema migrations."""
    from neuroatom.storage.migration import (
        get_pool_version,
        CURRENT_SCHEMA_VERSION,
        migrate as run_migrate,
        needs_migration,
    )

    current = get_pool_version(Path(pool_path))
    click.echo(f"Current schema version: {current}")
    click.echo(f"Target schema version: {CURRENT_SCHEMA_VERSION}")

    if not needs_migration(Path(pool_path)):
        click.echo("No migration needed.")
        return

    applied = run_migrate(Path(pool_path), dry_run=dry_run)
    if applied:
        for desc in applied:
            click.echo(f"  Applied: {desc}")
    else:
        click.echo("No migration path available.")


def main():
    cli()


if __name__ == "__main__":
    main()
