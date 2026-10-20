"""Shared fixtures: seeded synthetic point clouds, indexes and ledgers.

The test suite needs no external data. Only the optional backend cross-check
in ``test_reference_backend.py`` reads an environment variable,
``QRQ_REFERENCE_EXTENSION`` (see README.md, section Testing).
"""
from __future__ import annotations

import numpy as np
import pytest

import qrq

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
    """Synthetic per-tree ledgers of several source-days."""
    return write_synthetic_ledgers(tmp_path_factory.mktemp("ledgers"))
