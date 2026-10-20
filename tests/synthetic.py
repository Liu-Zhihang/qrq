"""Seeded synthetic point clouds, query boxes and ledgers used by the tests."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

import qrq
import qrq.ledger


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


LEDGER_CENTRE = (20000.0, 20000.0)
LEDGER_GRID = qrq.WindowGrid(widths=(300, 700, 1500, 4000, 9000), durations=(1200, 5400, 28800, 86400),
                             centres=(28800, 50400), anchor_step=6000.0)
"""Synthetic test grid; any grid works, this one keeps the ledgers small."""
LEDGER_ANCHORS = ((1, 1), (0, 2))
LEDGER_DAYS = (("alpha", "2020-01-01", 11), ("alpha", "2020-01-02", 12), ("beta", "2020-01-01", 13))


def ledger_cloud(n: int, seed: int) -> np.ndarray:
    """Hub-and-spoke cloud around :data:`LEDGER_CENTRE` covering the used anchors."""
    rng = np.random.default_rng(seed)
    cx, cy = LEDGER_CENTRE
    step = LEDGER_GRID.anchor_step
    hubs = np.array([[cx, cy], [cx - step, cy + step], [cx + 0.4 * step, cy - 0.3 * step]])
    which = rng.integers(0, len(hubs), n)
    spread = rng.choice([150.0, 600.0, 2500.0], n)
    x = hubs[which, 0] + rng.normal(0, 1, n) * spread
    y = hubs[which, 1] + rng.normal(0, 1, n) * spread
    t = rng.uniform(0, 86400, n)
    return qrq.make_points(x, y, t, rng.integers(0, 40, n).astype(np.int64))


def write_synthetic_ledgers(
    root: Path,
    days: Sequence[tuple[str, str, int]] = LEDGER_DAYS,
    n: int = 2500,
    anchors: Sequence[tuple[int, int]] = LEDGER_ANCHORS,
) -> list[qrq.ledger.LedgerDay]:
    """Write per-tree ledgers for synthetic days and return their ledger days.

    Each ``(source, date, seed)`` day indexes :func:`ledger_cloud` with the
    three tree kinds and records the windows of :data:`LEDGER_GRID` at
    ``anchors`` with :func:`qrq.ledger.ledger_row` (``L1`` costs).
    """
    out = []
    for source, date, seed in days:
        points = ledger_cloud(n, seed)
        indexes = {kind: qrq.SpatioTemporalIndex.build(points, kind=kind) for kind in qrq.TREE_KINDS}
        windows = [w for w in qrq.register_windows(*LEDGER_CENTRE, LEDGER_GRID) if w.anchor_index in anchors]
        paths = {}
        for kind, index in indexes.items():
            rows = (qrq.ledger.ledger_row(w, index, qrq.L1, source_id=source, date=date,
                                          arm="baseline", variant="base") for w in windows)
            paths[kind] = qrq.ledger.write_ledger(root / source / date / f"{kind}.jsonl.gz", rows)
        out.append(qrq.ledger.LedgerDay(source, date, paths))
    return out
