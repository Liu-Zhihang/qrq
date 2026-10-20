"""Registered windows, adjacent-scale pairs and crossing classification."""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

import qrq
from qrq.workload import CENTRES, DURATIONS, WIDTHS, adjacent_pairs, classify_pair, register_windows


@pytest.mark.parametrize("cx,cy", [(0.0, 0.0), (448478.20497292466, 4417664.577536389), (-1234.5, 98765.25)])
def test_register_windows_formulas(cx, cy):
    windows = register_windows(cx, cy)
    assert len(windows) == 648
    assert len({w.query_id for w in windows}) == 648
    expected = list(itertools.product(range(3), range(3), [21600, 43200, 64800],
                                      [250, 500, 1000, 2000, 5000, 10000], [900, 3600, 21600, 86400]))
    for w, (ix, iy, tc, width, d) in zip(windows, expected):
        x = cx + (ix - 1) * 20000 / 3
        y = cy + (iy - 1) * 20000 / 3
        assert w.query_id == f"g{ix}{iy}_c{tc}_w{width}_d{d}"
        assert w.anchor_index == (ix, iy)
        assert (w.width_m, w.duration_s, w.temporal_center_s) == (width, d, tc)
        assert list(w.box.bounds) == [x - width / 2, x + width / 2, y - width / 2, y + width / 2,
                                      tc - d / 2, tc + d / 2]
        unsupported = tc - d / 2 < 0 or tc + d / 2 > 86400
        assert w.status == ("unsupported_index_span" if unsupported else "scheduled")
        assert w.supported is not unsupported
    assert sum(not w.supported for w in windows) == 108


def test_register_windows_custom_grid():
    windows = register_windows(0, 0, widths=[100, 200], durations=[60], centres=[30], anchor_step=1000)
    assert len(windows) == 9 * 2
    assert all(w.supported for w in windows)
    assert windows[0].box.bounds == (-1050.0, -950.0, -1050.0, -950.0, 0.0, 60.0)


def test_adjacent_pairs_on_full_grid():
    windows = register_windows(0.0, 0.0)
    pairs = adjacent_pairs(windows)
    per_cell = (len(WIDTHS) - 1) * len(DURATIONS) + len(WIDTHS) * (len(DURATIONS) - 1)
    assert len(pairs) == 9 * len(CENTRES) * per_cell == 1026
    for p in pairs:
        a, b = p.small, p.large
        assert a.anchor_index == b.anchor_index and a.temporal_center_s == b.temporal_center_s
        if p.axis == "space":
            assert a.duration_s == b.duration_s
            assert WIDTHS.index(b.width_m) == WIDTHS.index(a.width_m) + 1
        else:
            assert a.width_m == b.width_m
            assert DURATIONS.index(b.duration_s) == DURATIONS.index(a.duration_s) + 1


def test_adjacent_pairs_do_not_skip_missing_windows():
    def item(w, d):
        return SimpleNamespace(anchor_index=[1, 1], temporal_center_s=43200, width_m=w, duration_s=d)

    present = [item(250, 900), item(1000, 900), item(250, 3600), item(500, 900)]
    pairs = adjacent_pairs(present)
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
    for w in register_windows(5000.0, 5000.0):
        assert isinstance(w.box, qrq.SpaceTimeBox)
        assert w.box.xmax - w.box.xmin == w.width_m
