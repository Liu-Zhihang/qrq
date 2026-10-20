"""Seeded synthetic point clouds and query boxes used by the tests."""
from __future__ import annotations

import numpy as np

import qrq


def synthetic_cloud(kind: str, n: int, seed: int) -> np.ndarray:
    """Seeded synthetic POINT arrays with different spatial structure.

    ``uniform``: uniform in a 10 km x 10 km x 1 day cube.
    ``clustered``: Gaussian hot spots with entity-correlated tracks.
    ``lattice``: integer coordinates with many exact ties and duplicates,
    which exercises the closed-bound boundary cases.
    """
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        x = rng.uniform(0, 10000, n)
        y = rng.uniform(0, 10000, n)
        t = rng.uniform(0, 86400, n)
        entity = rng.integers(0, max(1, n // 20), n)
    elif kind == "clustered":
        centres = rng.uniform(1000, 9000, (5, 2))
        which = rng.integers(0, len(centres), n)
        x = centres[which, 0] + rng.normal(0, 300, n)
        y = centres[which, 1] + rng.normal(0, 300, n)
        t = np.sort(rng.uniform(0, 86400, n))
        entity = which * 7 + rng.integers(0, 7, n)
    elif kind == "lattice":
        x = rng.integers(0, 12, n).astype(float) * 100
        y = rng.integers(0, 12, n).astype(float) * 100
        t = rng.integers(0, 24, n).astype(float) * 3600
        entity = rng.integers(0, 9, n)
    else:
        raise ValueError(kind)
    return qrq.make_points(x, y, t, entity.astype(np.int64))


def random_boxes(points: np.ndarray, count: int, seed: int) -> list[qrq.SpaceTimeBox]:
    """Random closed boxes of widely varying extent around the cloud.

    Some boxes snap their bounds to record coordinates (boundary hits), some
    are degenerate on one axis and some lie outside the data entirely.
    """
    rng = np.random.default_rng(seed)
    xyz = qrq.records.coordinates(points)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    boxes = []
    for i in range(count):
        mode = i % 5
        centre = xyz[rng.integers(len(xyz))] if mode in (0, 1) else rng.uniform(lo, hi)
        extent = span * rng.choice([0.0, 0.01, 0.05, 0.2, 0.5, 1.2], 3)
        a, b = centre - extent / 2, centre + extent / 2
        if mode == 1:
            snap = xyz[rng.integers(len(xyz), size=2)]
            a, b = np.minimum(snap[0], snap[1]), np.maximum(snap[0], snap[1])
        if mode == 4:
            a, b = hi + span * 0.1, hi + span * 0.2
        boxes.append(qrq.SpaceTimeBox(a[0], b[0], a[1], b[1], a[2], b[2]))
    return boxes
