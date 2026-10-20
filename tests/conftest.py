"""Shared fixtures: seeded synthetic data and sealed-evidence paths.

Sealed evidence is never part of the repository. Its location is passed at
test time through environment variables:

``QRQ_SEALED_LEDGERS``  evidence root with ``<host>/outputs/<source>/<date>/<tree>.jsonl.gz``
``QRQ_SEALED_TABLES``   directory with ``endpoints.csv.gz`` and ``scale_pairs.csv.gz``
``QRQ_SEALED_INPUTS``   directory with ``geometry_qa.json`` and ``queries/<source>.json``
``QRQ_SEALED_SWEEP``    directory with ``price_grid.csv``, ``price_by_source.csv``, ``price_pairs.csv``
``QRQ_SEALED_AA``       directory with ``aa_endpoints.csv.gz`` and ``aa_by_source.csv``
"""
from __future__ import annotations

import csv
import gzip

import numpy as np
import pytest

import qrq

from evidence import env_dir
from synthetic import synthetic_cloud, write_synthetic_ledgers


CLOUDS = [
    ("uniform", 3000, 101),
    ("clustered", 3000, 202),
    ("lattice", 2000, 303),
    ("uniform", 1, 404),
    ("lattice", 5, 505),
]


@pytest.fixture(scope="session", params=CLOUDS, ids=lambda p: f"{p[0]}-{p[1]}")
def cloud(request) -> np.ndarray:
    kind, n, seed = request.param
    return synthetic_cloud(kind, n, seed)


@pytest.fixture(scope="session")
def indexed_cloud() -> tuple[np.ndarray, dict[str, qrq.SpatioTemporalIndex]]:
    points = synthetic_cloud("clustered", 4000, seed=20260001)
    indexes = {kind: qrq.SpatioTemporalIndex.build(points, kind=kind) for kind in qrq.TREE_KINDS}
    return points, indexes


@pytest.fixture(scope="session")
def synthetic_ledgers(tmp_path_factory):
    """Synthetic per-tree ledgers in the sealed directory layout."""
    return write_synthetic_ledgers(tmp_path_factory.mktemp("ledgers"))


@pytest.fixture(scope="session")
def sealed_reproduction():
    """Full re-pricing of the sealed ledgers, computed once per session."""
    from qrq.ledger import reproduce

    return reproduce(env_dir("QRQ_SEALED_LEDGERS"))


@pytest.fixture(scope="session")
def sealed_amplified():
    """AA-inclusive re-enveloping of the sealed ledgers, computed once per session."""
    from qrq.ledger import reproduce

    return reproduce(env_dir("QRQ_SEALED_LEDGERS"), amplified=True)


@pytest.fixture(scope="session")
def sealed_cities() -> dict[str, str]:
    """Source-to-city labels from the sealed endpoint table (ledgers carry none)."""
    path = env_dir("QRQ_SEALED_TABLES") / "endpoints.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return {row["source_id"]: row["city"] for row in csv.DictReader(handle)}
