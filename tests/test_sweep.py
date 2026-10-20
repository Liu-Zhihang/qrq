"""Exact repricing of per-tree endpoints over price grids (synthetic data)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import qrq
from qrq import sweep
from qrq.ledger import audit_cohort, read_ledger

from synthetic import LEDGER_GRID


LOOKUPS = (50, 20, 5, 1)
GEOMETRIES = (5000, 500, 50, 5, 1)
"""Synthetic test grid; any lookup and geometry prices work."""


def test_price_grid_order():
    grid = sweep.price_grid(LOOKUPS, GEOMETRIES)
    assert len(grid) == len(LOOKUPS) * len(GEOMETRIES)
    assert [(p.lookup, p.g_quantum) for p in grid[:5]] == [(50, g) for g in GEOMETRIES]
    assert [p.lookup for p in grid[::5]] == list(LOOKUPS)
    assert all((p.g_classical, p.lam, p.repeats) == (1, 1, 1) for p in grid)
    assert sweep.price_grid([2], [3, 4], base=qrq.UNIT) == [qrq.PriceTable(1, 2, 3, 1, 1), qrq.PriceTable(1, 2, 4, 1, 1)]


def tiny_cohort(**changes) -> sweep.SweepCohort:
    """Three windows on two trees; the last window has tied quantum trees."""
    fields = dict(
        keys=(("s1", "d1", "a"), ("s1", "d1", "b"), ("s2", "d2", "a")),
        group=np.array(["x", "x", "y"]),
        k=np.array([10, 20, 5]),
        width_m=np.array([500, 1000, 250]),
        duration_s=np.array([900, 900, 3600]),
        M=np.array([[20, 40], [20, 25], [5, 5]]),
        G=np.array([[100, 50], [300, 200], [10, 10]]),
        bits=np.array([[4, 4], [4, 4], [3, 3]]),
        P0=np.array([[50.0, 60.0], [70.0, 80.0], [5.0, 5.0]]),
        C=np.array([[1000.0, 900.0], [500.0, 500.0], [300.0, 320.0]]),
        classical_lookups=np.full((3, 2), 2.0),
        small=np.array([0]),
        large=np.array([1]),
        axis=np.array(["space"]),
        trees=("packed_rtree", "kd_tree"),
    )
    fields.update(changes)
    return sweep.SweepCohort(**fields)


@pytest.mark.parametrize("lookup,geometry", [(100, 10000), (10, 1000), (1, 1), (3, 30)])
def test_reprice_closed_form_by_hand(lookup, geometry):
    cohort = tiny_cohort()
    point = sweep.reprice(cohort, dataclasses.replace(qrq.L1, lookup=lookup, g_quantum=geometry))
    for i in range(3):
        k = int(cohort.k[i])
        for t in range(2):
            m, g, b, p0 = (int(cohort.M[i, t]), int(cohort.G[i, t]), int(cohort.bits[i, t]),
                           float(cohort.P0[i, t]))
            q = p0 + m / k * (g + 2 * lookup + geometry + 1 + b) + 2 * lookup - b
            assert point.Q_tree[i, t] == q
            assert point.C_tree[i, t] == cohort.C[i, t] - 2 * (100 - lookup)
        assert point.Q[i] == point.Q_tree[i].min() and point.C[i] == point.C_tree[i].min()
        assert point.quantum_index[i] == int(np.argmin(point.Q_tree[i]))
        qi = point.quantum_index[i]
        assert point.trials[i] == cohort.M[i, qi] / k
        assert point.structure_share[i] == (cohort.M[i, qi] / k * cohort.G[i, qi]) / point.Q[i]
        assert point.prefix_share[i] == cohort.P0[i, qi] / point.Q[i]
    assert point.quantum_index[2] == 0
    np.testing.assert_array_equal(point.D, np.log2(point.Q / point.C))
    d = point.D
    assert point.forward[0] == (d[0] > 0 > d[1]) and point.reverse[0] == (d[0] < 0 < d[1])


def test_tiny_sweep_rows():
    cohort = tiny_cohort()
    subsets = [("short", cohort.duration_s == 900), ("narrow", cohort.width_m <= 1000), ("large", cohort.k >= 20000)]
    tables = sweep.sweep(cohort, [qrq.L1, qrq.UNIT], subsets=subsets)
    assert [r["lookup"] for r in tables.grid] == [100, 1]
    unit = tables.grid[1]
    assert unit["eligible_windows"] == 3 and unit["q_tree_tie_windows"] == 1
    assert unit["short_windows"] == 2 and unit["narrow_windows"] == 3 and unit["large_windows"] == 0
    assert unit["large_ratio_median"] is None and unit["large_lower"] == 0
    assert "forward_retained" not in unit and "forward_retained" not in tables.pairs[0]
    assert list(unit)[5:8] == ["short_windows", "short_lower", "short_ratio_median"]
    plain = sweep.sweep(cohort, [qrq.UNIT]).grid[0]
    assert "short_windows" not in plain and plain["lower_windows"] == unit["lower_windows"]
    with pytest.raises(ValueError):
        sweep.sweep(cohort, [qrq.UNIT], subsets=[("bad", np.ones(2, dtype=bool))])
    assert [r["source_id"] for r in tables.by_source] == ["s1", "s2"] * 2
    assert [r["source_days"] for r in tables.by_source[:2]] == [1, 1]
    assert [(r["source_id"], r["axis"]) for r in tables.pairs[:3]] == [("ALL", "all"), ("ALL", "space"), ("s1", "all")]


def test_cohort_validation():
    with pytest.raises(ValueError):
        tiny_cohort(k=np.array([10, 20]))
    with pytest.raises(ValueError):
        tiny_cohort(M=np.array([[5, 40], [20, 25], [5, 5]]))
    with pytest.raises(ValueError):
        tiny_cohort(classical_lookups=np.array([[2.0, math.nan], [2.0, 2.0], [2.0, 2.0]]))
    with pytest.raises(ValueError):
        tiny_cohort(reference=qrq.PriceTable(1, 100, 10000, 2, 1))
    cohort = tiny_cohort()
    for prices in (qrq.PriceTable(1, 100, 10000, 2, 1), qrq.PriceTable(2, 100, 10000, 1, 1)):
        with pytest.raises(ValueError):
            sweep.reprice(cohort, prices)


@pytest.fixture(scope="module")
def ledger_cohort(synthetic_ledgers):
    r = audit_cohort(synthetic_ledgers, LEDGER_GRID)
    return r, sweep.cohort_from_ledger(r)


def test_cohort_from_ledger(ledger_cohort):
    r, cohort = ledger_cohort
    assert len(cohort.keys) == r.window_count and len(cohort.small) == len(r.pairs)
    assert cohort.reference == qrq.L1 and cohort.trees == qrq.TREE_KINDS
    np.testing.assert_array_equal(cohort.group, cohort.source)
    merged = sweep.cohort_from_ledger(r, group_of={"alpha": "one", "beta": "one"})
    assert set(merged.group) == {"one"}
    rows = sweep.sweep(merged, [qrq.UNIT]).by_source
    assert {row["group"] for row in rows} == {"one"} and "group" in rows[0]


def test_closed_form_equals_route_cost(ledger_cohort, synthetic_ledgers):
    r, cohort = ledger_cohort
    index = {key: i for i, key in enumerate(cohort.keys)}
    tree_pos = {t: j for j, t in enumerate(cohort.trees)}
    grid = [qrq.L1, qrq.UNIT, dataclasses.replace(qrq.L1, lookup=3, g_quantum=300)]
    points = [sweep.reprice(cohort, p) for p in grid]
    checked = 0
    for day in synthetic_ledgers:
        for tree, path in day.paths.items():
            for row in read_ledger(path):
                i = index.get((day.source_id, day.date, row["query_id"]))
                if i is None:
                    continue
                j = tree_pos[tree]
                for prices, point in zip(grid, points):
                    costs = qrq.route_costs(row["routes"], prices)
                    assert math.isclose(point.Q_tree[i, j], costs["QRQ_FW"], rel_tol=1e-12)
                    assert point.C_tree[i, j] == min(v for name, v in costs.items() if name != "QRQ_FW")
                checked += 1
    assert checked == 3 * len(cohort.keys)
    base = points[0]
    assert [bool(f) for f in base.forward] == [p.direction == "forward" for p in r.pairs]
    assert [bool(v) for v in base.reverse] == [p.direction == "reverse" for p in r.pairs]


def test_sweep_row_invariants(ledger_cohort):
    r, cohort = ledger_cohort
    grid = sweep.price_grid(LOOKUPS, GEOMETRIES, base=qrq.L1)
    seen = []
    tables = sweep.sweep(cohort, [qrq.L1] + grid, reference=qrq.L1, on_point=seen.append)
    grid = [qrq.L1] + grid
    unit_ref = sweep.sweep(cohort, grid, reference=qrq.UNIT)
    sources = sorted(set(cohort.source))
    axes = sorted(set(cohort.axis))
    assert len(seen) == len(tables.grid) == len(grid)
    assert len(tables.by_source) == len(grid) * len(sources)
    assert len(tables.pairs) == len(grid) * (len(sources) + 1) * (len(axes) + 1)
    first = tables.grid[0]
    assert first["forward_retained"] == first["forward_pairs"] and first["forward_added"] == first["forward_lost"] == 0
    assert first["lower_windows"] == r.quantum_lower_count
    assert any(row["forward_pairs"] for row in tables.grid)
    unit_forward = unit_ref.grid[-1]["forward_pairs"]
    assert unit_ref.grid[-1]["forward_retained"] == unit_forward
    for g, (row, ref_row) in enumerate(zip(tables.grid, unit_ref.grid)):
        assert ref_row["forward_retained"] + ref_row["forward_lost"] == unit_forward
        assert ref_row["forward_retained"] + ref_row["forward_added"] == row["forward_pairs"]
        assert row["forward_retained"] + row["forward_added"] == row["forward_pairs"]
        per_source = tables.by_source[g * len(sources):(g + 1) * len(sources)]
        assert sum(s["eligible_windows"] for s in per_source) == row["eligible_windows"]
        assert sum(s["lower_windows"] for s in per_source) == row["lower_windows"]
        block = tables.pairs[g * (len(sources) + 1) * (len(axes) + 1):(g + 1) * (len(sources) + 1) * (len(axes) + 1)]
        overall = block[0]
        assert (overall["source_id"], overall["axis"]) == ("ALL", "all")
        for name in ("eligible_pairs", "forward_pairs", "reverse_pairs", "forward_retained",
                     "forward_added", "forward_lost"):
            assert overall[name] == row[name]
            assert sum(b[name] for b in block if b["source_id"] != "ALL" and b["axis"] == "all") == row[name]
        assert row["groups_with_forward"] == row["sources_with_forward"] <= len(sources)
