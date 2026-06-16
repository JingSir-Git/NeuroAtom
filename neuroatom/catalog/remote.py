"""Remote registry: fetch, merge, and pull datasets from remote catalogs.

A remote catalog is a ``CatalogIndex`` served as JSON over HTTP(S).
Dataset entries may include a ``pool_url`` field pointing to a downloadable
``.napool`` archive.

Protocol::

    GET <registry_url>/catalog.json  →  CatalogIndex JSON

    # Optional: download a pre-built pool
    GET <entry.pool_url>             →  .napool file (tar.gz)

Built-in registry URLs can be configured in ``pool.yaml``::

    catalog:
      registries:
        - https://example.com/neuroatom/catalog.json
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from neuroatom.catalog.models import CatalogIndex, DatasetEntry
from neuroatom.catalog.local import load_catalog, save_catalog
from neuroatom.storage.pool import Pool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30  # seconds


def fetch_remote_catalog(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> CatalogIndex:
    """Fetch a remote catalog index from a URL.

    Args:
        url: URL to the catalog JSON file.
        timeout: Request timeout in seconds.

    Returns:
        Parsed CatalogIndex.

    Raises:
        ConnectionError: If the fetch fails.
    """
    logger.info("Fetching remote catalog: %s", url)
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        catalog = CatalogIndex.model_validate(data)
        catalog.url = url
        logger.info(
            "Fetched %d datasets from %s", len(catalog.datasets), url,
        )
        return catalog
    except (URLError, json.JSONDecodeError, Exception) as e:
        raise ConnectionError(f"Failed to fetch catalog from {url}: {e}") from e


def merge_remote(
    pool: Pool,
    remote_url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> int:
    """Fetch a remote catalog and merge it into the local catalog.

    Args:
        pool: Local pool.
        remote_url: URL of the remote catalog.
        timeout: Request timeout.

    Returns:
        Number of new/updated entries.
    """
    remote = fetch_remote_catalog(remote_url, timeout=timeout)
    local = load_catalog(pool)
    count = local.merge(remote)
    if count:
        save_catalog(pool, local)
    logger.info("Merged %d entries from %s", count, remote_url)
    return count


def pull_dataset(
    pool: Pool,
    dataset_id: str,
    pool_url: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path:
    """Download a .napool archive and import it into the pool.

    If pool_url is not provided, looks it up in the local catalog.

    Args:
        pool: Target pool to import into.
        dataset_id: Dataset to pull.
        pool_url: Direct URL to the .napool file. If None, looked up
            from the catalog entry.
        timeout: Request timeout.

    Returns:
        Path to the imported dataset directory.

    Raises:
        ValueError: If no pool_url available.
        ConnectionError: If download fails.
    """
    if not pool_url:
        catalog = load_catalog(pool)
        entry = catalog.get(dataset_id)
        if entry and entry.pool_url:
            pool_url = entry.pool_url
        else:
            raise ValueError(
                f"No pool_url for dataset '{dataset_id}'. "
                "Provide --url or add pool_url to catalog entry."
            )

    logger.info("Downloading %s from %s ...", dataset_id, pool_url)

    # Download to temp file
    try:
        req = Request(pool_url)
        with urlopen(req, timeout=timeout) as resp:
            with tempfile.NamedTemporaryFile(
                suffix=".napool", delete=False,
            ) as tmp:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    tmp.write(chunk)
                tmp_path = Path(tmp.name)
    except (URLError, Exception) as e:
        raise ConnectionError(f"Failed to download {pool_url}: {e}") from e

    logger.info("Downloaded to %s, importing...", tmp_path)

    # Import into pool
    try:
        from neuroatom.storage.pool_archive import import_pool
        result = import_pool(tmp_path, pool.root)
        logger.info("Imported dataset '%s' (%d files)", dataset_id, result.get("files_imported", 0))
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass

    from neuroatom.storage import paths as P
    return P.dataset_dir(pool.root, dataset_id)


def list_registries(pool: Pool) -> List[str]:
    """List configured remote registry URLs from pool config."""
    config = pool.config
    return config.get("catalog", {}).get("registries", [])
