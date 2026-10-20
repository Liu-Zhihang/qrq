"""Reconstruction-task comparison of crossing and noncrossing pairs (synthetic data)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from qrq import tasks
from qrq.tasks import TaskPair

STRATA = (("space", 500, 2000, 900), ("space", 2000, 5000, 900), ("time", 900, 3600, 500))


def synthetic_pairs(seed: int = 3) -> tuple[list[TaskPair], dict[tuple[str, str], list[str]]]:
    """Random pairs of two families; registry dates are deliberately unsorted."""
    rng = np.random.default_rng(seed)
    pairs: list[TaskPair] = []
    registry: dict[tuple[str, str], list[str]] = {}
    for family, sources in (("f1", ("s_b", "s_a")), ("f2", ("s_c",))):
        for source in sources:
            dates = [f"2021-01-{d:02d}" for d in range(1, 7)]
            registry[(family, source)] = dates[::-1]
            for date in dates[:-1]:
                for axis, small, large, fixed in STRATA:
                    for _ in range(int(rng.integers(0, 6))):
                        supported = bool(rng.random() > 0.1)
                        d_small, d_large = (float(rng.choice([-1.0, 1.0, 1.0, 0.0]) * rng.uniform(0.1, 2))
                                            for _ in range(2))
                        delta = float(rng.choice([-1.0, 0.0, 1.0, 1.0]) * rng.uniform(0.1, 1))
                        pairs.append(TaskPair(family, source, date, axis, small, large, fixed, supported,
                                              d_small, d_large, delta if supported else math.nan))
    return pairs, registry


def reference_estimate(pairs) -> tuple[int, float, int, int]:
    """Plain-Python standardisation: (forward up, expected control up, used, excluded)."""
    cells: dict = {}
    for p in pairs:
        if not p.supported or p.error == "tie":
            continue
        cell = cells.setdefault(p.stratum, [0, 0, 0, 0])
        crossing = tasks.crossing_direction(p.small_D, p.large_D)
        if crossing == "forward":
            cell[0 if p.error == "up" else 1] += 1
        elif crossing == "other":
            cell[2 if p.error == "up" else 3] += 1
    up = used = excluded = 0
    expected = 0.0
    for fu, fd, cu, cd in cells.values():
        f, c = fu + fd, cu + cd
        if f and c:
            up, used, expected = up + fu, used + f, expected + cu / c * f
        elif f:
            excluded += f
    return up, expected, used, excluded


@pytest.mark.parametrize("d_small,d_large,direction,category", [
    (1.0, -1.0, "forward", "forward"),
    (-0.5, 2.0, "reverse", "reverse"),
    (1.0, 3.0, "other", "classical"),
    (-1.0, -3.0, "other", "quantum"),
    (0.0, -1.0, "other", "boundary"),
    (1.0, 0.0, "other", "boundary"),
])
def test_crossing_classification(d_small, d_large, direction, category):
    assert tasks.crossing_direction(d_small, d_large) == direction
    assert tasks.crossing_category(d_small, d_large) == category


def test_error_direction_and_worsening_rate():
    assert [tasks.error_direction(v) for v in (0.2, -1e-300, 0.0, -0.0)] == ["up", "down", "tie", "tie"]
    assert tasks.worsening_rate(3, 1) == 0.75 and math.isnan(tasks.worsening_rate(0, 0))


def endpoint(width, duration, k=5, d=1.0, error=0.3):
    return dict(width_m=width, duration_s=duration, k=k, D=d, absolute_error=error)


def test_pair_from_endpoints():
    p = tasks.pair_from_endpoints("f", "s", "d", "space", endpoint(500, 900), endpoint(2000, 900, d=-1.0), 0.1)
    assert (p.small, p.large, p.fixed, p.supported, p.direction, p.error) == (500, 2000, 900, True, "forward", "up")
    t = tasks.pair_from_endpoints("f", "s", "d", "time", endpoint(500, 900), endpoint(500, 3600), -0.1)
    assert (t.stratum, t.category, t.error) == (("time", 900, 3600, 500), "classical", "down")
    for small in (endpoint(500, 900, k=0), endpoint(500, 900, d=math.nan), endpoint(500, 900, error=None)):
        u = tasks.pair_from_endpoints("f", "s", "d", "space", small, endpoint(2000, 900), math.nan)
        assert not u.supported and u.direction == u.category == u.error == "unsupported"
    with pytest.raises(ValueError):
        tasks.pair_from_endpoints("f", "s", "d", "space", endpoint(2000, 900), endpoint(500, 900), 0.1)
    with pytest.raises(ValueError):
        tasks.pair_from_endpoints("f", "s", "d", "space", endpoint(500, 900), endpoint(2000, 3600), 0.1)
    with pytest.raises(ValueError):
        tasks.pair_from_endpoints("f", "s", "d", "space", endpoint(500, 900), endpoint(2000, 900), math.nan)
    with pytest.raises(ValueError):
        tasks.pair_from_endpoints("f", "s", "d", "depth", endpoint(500, 900), endpoint(2000, 900), 0.1)


def test_date_stratum_counts_rules():
    def pair(date, d_small, d_large, delta, supported=True, stratum=STRATA[0]):
        return TaskPair("f", "s", date, *stratum, supported, d_small, d_large, delta)

    pairs = [pair("a", 1, -1, 0.5), pair("a", 1, -1, -0.5), pair("a", 1, -1, 0.0),
             pair("b", 1, 1, 0.5), pair("b", -1, -1, -0.5), pair("b", 0, 1, 0.0),
             pair("b", -1, 1, 0.5), pair("b", 1, -1, 0.5, supported=False),
             pair("a", 1, 2, 0.1, stratum=STRATA[1])]
    counts = tasks.date_stratum_counts(pairs, ["a", "b"], STRATA[:2])
    np.testing.assert_array_equal(counts[0, 0], [1, 1, 0, 0, 1])
    np.testing.assert_array_equal(counts[1, 0], [0, 0, 1, 1, 0])
    np.testing.assert_array_equal(counts[0, 1], [0, 0, 1, 0, 0])
    assert counts.sum() == 6


def test_standardised_estimates_by_hand():
    counts = np.array([[3, 1, 2, 2, 7], [1, 0, 0, 0, 0], [0, 0, 1, 2, 0]])
    up, expected, used, excluded = tasks.standardised_estimates(counts)
    assert (int(up), float(expected), int(used), int(excluded)) == (3, 2.0, 4, 1)
    batch = np.stack([counts, np.zeros_like(counts)])
    up, expected, used, excluded = tasks.standardised_estimates(batch)
    assert up.tolist() == [3, 0] and expected.tolist() == [2.0, 0.0] and used.tolist() == [4, 0]


@pytest.fixture(scope="module")
def comparison():
    pairs, registry = synthetic_pairs()
    return pairs, registry, tasks.compare_crossings(pairs, registry, ("f1", "f2"), draws=400, seed=11)


def test_point_estimates_match_reference(comparison):
    pairs, registry, result = comparison
    assert [(s.family, s.source) for s in result.sources] == [("f1", "s_a"), ("f1", "s_b"), ("f2", "s_c")]
    for s in result.sources:
        members = [p for p in pairs if (p.family, p.source) == (s.family, s.source)]
        up, expected, used, excluded = reference_estimate(members)
        assert (s.used_forward_n, s.excluded_forward_no_control) == (used, excluded)
        assert s.forward_rate == up / used
        assert math.isclose(s.control_rate, expected / used, rel_tol=1e-12)
        assert s.dates == tuple(sorted(registry[(s.family, s.source)]))
        assert s.registered_n == len(members) and s.counts.shape == (6, len(s.strata), 5)
        row = s.row()
        assert row["forward_n"] == sum(p.direction == "forward" for p in members)
        assert row["valid_bootstrap_draws"] + row["invalid_bootstrap_draws"] == 400
        assert len(s.stratum_rows()) == len(s.strata)


def test_bootstrap_draws_match_reference(comparison):
    pairs, registry, result = comparison
    rng = np.random.default_rng(11)
    for s in result.sources:
        dates = sorted(registry[(s.family, s.source)])
        selected = rng.integers(0, len(dates), size=(400, len(dates)))
        members = [p for p in pairs if (p.family, p.source) == (s.family, s.source)]
        for b in (0, 1, 399):
            drawn = [p for j in selected[b] for p in members if p.date == dates[j]]
            up, expected, used, excluded = reference_estimate(drawn)
            got = s.draws[b]
            assert (got[0], got[2], got[3]) == (up, used, excluded)
            assert math.isclose(got[1], expected, rel_tol=1e-12, abs_tol=1e-12)


def test_family_pooling(comparison):
    pairs, registry, result = comparison
    for family in result.families:
        np.testing.assert_array_equal(family.draws, sum(s.draws for s in family.sources))
        weights = np.array([s.used_forward_n for s in family.sources])
        rates = np.array([s.forward_rate for s in family.sources])
        assert math.isclose(family.forward_rate, np.nansum(rates * weights) / weights.sum(), rel_tol=1e-15)
        row = family.row()
        assert row["forward_n"] == sum(s.forward_n for s in family.sources)
        assert row["source_days"] == sum(len(s.dates) for s in family.sources)
    assert [f.row()["family"] for f in result.families] == ["f1", "f2"]


def test_undefined_draws_stay_undefined():
    pairs = [TaskPair("f", "s", "d1", *STRATA[0], True, 1.0, -1.0, 0.4),
             TaskPair("f", "s", "d1", *STRATA[0], True, 1.0, 1.0, -0.4)]
    pairs += [TaskPair("f", "s", f"d{i}", *STRATA[0], True, 1.0, 2.0, 0.2) for i in range(2, 7)]
    result = tasks.compare_crossings(pairs, {("f", "s"): [f"d{i}" for i in range(1, 7)]}, draws=600, seed=5)
    row = result.sources[0].row()
    assert row["invalid_bootstrap_draws"] > 0
    assert row["valid_bootstrap_draws"] + row["invalid_bootstrap_draws"] == 600
    forward, control, delta = result.sources[0].draw_rates()
    assert np.array_equal(np.isnan(delta), result.sources[0].draws[:, 2] == 0)


def test_random_stream_order(comparison):
    pairs, registry, result = comparison
    again = tasks.compare_crossings(pairs, registry, ("f1", "f2"), draws=400, seed=11)
    swapped = tasks.compare_crossings(pairs, registry, ("f2", "f1"), draws=400, seed=11)
    other_seed = tasks.compare_crossings(pairs, registry, ("f1", "f2"), draws=400, seed=12)
    for a, b, c, d in zip(result.sources, again.sources, sorted(swapped.sources, key=lambda s: s.family),
                          other_seed.sources):
        assert np.array_equal(a.draws, b.draws)
        assert (a.forward_rate, a.control_rate) == (c.forward_rate, c.control_rate)
        assert not np.array_equal(a.draws, d.draws)
    assert not np.array_equal(result.sources[0].draws, swapped.sources[1].draws)
    default_order = tasks.compare_crossings(pairs, registry, draws=400, seed=11)
    assert np.array_equal(default_order.sources[0].draws, result.sources[0].draws)


def test_registry_errors():
    pairs, registry = synthetic_pairs()
    with pytest.raises(ValueError):
        tasks.compare_crossings(pairs, {k: v for k, v in registry.items() if k[1] != "s_c"}, draws=10)
    with pytest.raises(ValueError):
        tasks.compare_crossings(pairs, {**registry, ("f1", "s_a"): ["2021-01-01"]}, draws=10)
    with pytest.raises(ValueError):
        tasks.compare_crossings(pairs, registry, family_order=("f1",), draws=10)
    with pytest.raises(ValueError):
        tasks.compare_crossings([], {("f", "s"): []}, draws=10)


def test_source_without_forward_pairs():
    pairs = [TaskPair("f", "s", "d1", *STRATA[0], True, 1.0, 2.0, 0.2)]
    s = tasks.compare_crossings(pairs, {("f", "s"): ["d1", "d2"]}, draws=50, seed=1).sources[0]
    row = s.row()
    assert math.isnan(row["forward_worse_rate"]) and row["delta_ci_low"] is None
    assert row["operational_label"] == tasks.NO_COMPARABLE_FORWARD and row["invalid_bootstrap_draws"] == 50


@pytest.mark.parametrize("delta,interval,label", [
    (math.nan, (0.1, 0.2), "NO_COMPARABLE_FORWARD"),
    (0.2, (None, None), "NO_COMPARABLE_FORWARD"),
    (0.2, (-0.01, 0.4), "NO_INFORMATION_OPERATIONAL"),
    (0.03, (0.01, 0.05), "NO_INFORMATION_OPERATIONAL"),
    (0.15, (0.01, 0.3), "CROSSING_WORSE"),
    (-0.15, (-0.3, -0.05), "CROSSING_BETTER"),
])
def test_operational_label(delta, interval, label):
    assert tasks.operational_label(delta, interval) == label


def test_percentile_interval():
    assert tasks.percentile_interval(np.array([np.nan, np.nan])) == (None, None)
    values = np.array([np.nan, *range(101)], dtype=float)
    assert tasks.percentile_interval(values) == (2.5, 97.5)


def test_category_summary_and_mean():
    pairs, _ = synthetic_pairs()
    rows = tasks.category_summary(pairs)
    keys = [(r["family"], r["source"], r["category"]) for r in rows]
    assert keys == sorted(keys) and sum(r["n"] for r in rows) == len(pairs)
    for r, key in zip(rows, keys):
        members = [p for p in pairs if (p.family, p.source, p.category) == key]
        deltas = [p.delta_error for p in members if math.isfinite(p.delta_error)]
        assert r["n_up"] + r["n_down"] + r["n_tie"] == len(deltas)
        if deltas:
            assert math.isclose(r["mean_delta_error"], math.fsum(deltas) / len(deltas), rel_tol=1e-12, abs_tol=1e-15)
        else:
            assert math.isnan(r["mean_delta_error"]) and r["category"] == "unsupported"
    assert tasks.compensated_mean([1.0, 2.0, 4.0]) == 7 / 3 and math.isnan(tasks.compensated_mean([]))
