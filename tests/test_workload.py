"""Registered windows on caller-supplied grids, adjacent-scale pairs and crossings."""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

import qrq
from qrq.workload import WindowGrid, adjacent_pairs, classify_pair, register_windows

GRID = WindowGrid(widths=(300, 800, 2000), durations=(1800, 7200, 43200, 86400),
                  centres=(3600, 43200), anchor_step=2500.0)


@pytest.mark.parametrize("cx,cy", [(0.0, 0.0), (448478.20497292466, 4417664.577536389), (-1234.5, 98765.25)])
def test_register_windows_formulas(cx, cy):
    windows = register_windows(cx, cy, GRID)
    assert len(windows) == GRID.size == 9 * 2 * 3 * 4
    assert len({w.query_id for w in windows}) == len(windows)
    expected = itertools.product(range(3), range(3), GRID.centres, GRID.widths, GRID.durations)
    for w, (ix, iy, tc, width, d) in zip(windows, expected):
        x = cx + (ix - 1) * 2500.0
        y = cy + (iy - 1) * 2500.0
        assert w.query_id == f"g{ix}{iy}_c{tc}_w{width}_d{d}"
        assert w.anchor_index == (ix, iy)
        assert (w.width_m, w.duration_s, w.temporal_center_s) == (width, d, tc)
        assert list(w.box.bounds) == [x - width / 2, x + width / 2, y - width / 2, y + width / 2,
                                      tc - d / 2, tc + d / 2]
        unsupported = tc - d / 2 < 0 or tc + d / 2 > 86400
        assert w.status == ("unsupported_index_span" if unsupported else "scheduled")
        assert w.supported is not unsupported
    assert sum(not w.supported for w in windows) == 9 * 3 * 2


@pytest.mark.parametrize("anchors,offsets", [(1, [0.0]), (2, [-0.5, 0.5]), (4, [-1.5, -0.5, 0.5, 1.5])])
def test_anchor_lattice(anchors, offsets):
    grid = WindowGrid(widths=(10,), durations=(60,), centres=(100,), anchor_step=1000.0, anchors=anchors)
    windows = register_windows(500.0, -500.0, grid)
    assert len(windows) == anchors ** 2 == grid.size
    xs = sorted({(w.box.xmin + w.box.xmax) / 2 for w in windows})
    ys = sorted({(w.box.ymin + w.box.ymax) / 2 for w in windows})
    assert xs == [500.0 + o * 1000.0 for o in offsets]
    assert ys == [-500.0 + o * 1000.0 for o in offsets]


def test_grid_validation_and_day_length():
    for bad in (dict(widths=(200, 100)), dict(durations=()), dict(centres=()), dict(anchors=0),
                dict(anchor_step=0.0), dict(widths=(0, 10))):
        spec = dict(widths=(100, 200), durations=(60,), centres=(30,), anchor_step=1000.0)
        spec.update(bad)
        with pytest.raises(ValueError):
            WindowGrid(**spec)
    short_day = WindowGrid(widths=(100,), durations=(60, 7200), centres=(1800,), anchor_step=1.0,
                           anchors=1, day_seconds=3600)
    assert [w.supported for w in register_windows(0, 0, short_day)] == [True, False]
    windows = register_windows(0, 0, WindowGrid(widths=[100, 200], durations=[60], centres=[30],
                                                anchor_step=1000))
    assert len(windows) == 9 * 2 and all(w.supported for w in windows)
    assert windows[0].box.bounds == (-1050.0, -950.0, -1050.0, -950.0, 0.0, 60.0)


def test_adjacent_pairs_on_full_grid():
    windows = register_windows(0.0, 0.0, GRID)
    pairs = adjacent_pairs(windows, GRID)
    per_cell = (len(GRID.widths) - 1) * len(GRID.durations) + len(GRID.widths) * (len(GRID.durations) - 1)
    assert len(pairs) == 9 * len(GRID.centres) * per_cell
    for p in pairs:
        a, b = p.small, p.large
        assert a.anchor_index == b.anchor_index and a.temporal_center_s == b.temporal_center_s
        if p.axis == "space":
            assert a.duration_s == b.duration_s
            assert GRID.widths.index(b.width_m) == GRID.widths.index(a.width_m) + 1
        else:
            assert a.width_m == b.width_m
            assert GRID.durations.index(b.duration_s) == GRID.durations.index(a.duration_s) + 1


def test_adjacent_pairs_do_not_skip_missing_windows():
    grid = WindowGrid(widths=(250, 500, 1000), durations=(900, 3600), centres=(43200,), anchor_step=1.0)

    def item(w, d):
        return SimpleNamespace(anchor_index=[1, 1], temporal_center_s=43200, width_m=w, duration_s=d)

    present = [item(250, 900), item(1000, 900), item(250, 3600), item(500, 900)]
    pairs = adjacent_pairs(present, grid)
    keys = {(p.small.width_m, p.small.duration_s, p.large.width_m, p.large.duration_s, p.axis) for p in pairs}
    assert keys == {(250, 900, 500, 900, "space"), (250, 900, 250, 3600, "time"), (500, 900, 1000, 900, "space")}


@pytest.mark.parametrize(
    "d_small,d_large,expected",
    [
        (1.0, -1.0, "forward"),
        (0.25, -3.0, "forward"),
        (-1.0, 1.0, "reverse"),
        (-0.5, 0.001, "reverse"),
        (1.0, 1.0, "no_cross"),
        (-1.0, -2.0, "no_cross"),
        (0.0, -1.0, "no_cross"),
        (1.0, 0.0, "no_cross"),
        (0.0, 0.0, "no_cross"),
        (-1.0, 0.0, "no_cross"),
    ],
)
def test_classify_pair(d_small, d_large, expected):
    assert classify_pair(d_small, d_large) == expected


def test_window_boxes_are_valid_queries():
    for w in register_windows(5000.0, 5000.0, GRID):
        assert isinstance(w.box, qrq.SpaceTimeBox)
        assert w.box.xmax - w.box.xmin == w.width_m
