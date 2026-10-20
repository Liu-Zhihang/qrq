"""AA-inclusive envelope against the published sensitivity (skipped by default).

Needs ``QRQ_SEALED_LEDGERS``; the group count also needs ``QRQ_SEALED_TABLES``
and the row-level comparison ``QRQ_SEALED_AA`` (the sealed
``aa_endpoints.csv.gz`` and ``aa_by_source.csv``).
"""
from __future__ import annotations

import collections
import math

from qrq.workload import classify_pair

from evidence import assert_rows_equal, env_dir, read_csv

PUBLISHED_AA = dict(windows=13370, quantum_lower=869, pairs=17592, forward=1211, reverse=13)


def pair_key(p):
    return (p.source_id, p.date, p.axis, p.small_query, p.large_query)


def amplified_only_directions(reproduction):
    """Directions when the AA cost replaces the first-witness cost outright."""
    d = {(w.source_id, w.date, w.query_id): math.log2(w.amplified_cost / w.C) for w in reproduction.windows}
    return {pair_key(p): classify_pair(d[(p.source_id, p.date, p.small_query)],
                                       d[(p.source_id, p.date, p.large_query)])
            for p in reproduction.pairs}


def test_amplified_envelope_counts(sealed_reproduction, sealed_amplified):
    summary = sealed_amplified.summary()
    for key, value in PUBLISHED_AA.items():
        assert summary[key] == value, key
    legacy = {pair_key(p): p.direction for p in sealed_reproduction.pairs}
    joint = {pair_key(p): p for p in sealed_amplified.pairs}
    assert legacy.keys() == joint.keys()
    was_forward = [k for k, d in legacy.items() if d == "forward"]
    assert len(was_forward) == 1060
    assert sum(joint[k].direction == "forward" for k in was_forward) == 973
    assert sum(joint[k].small_D < 0 and joint[k].large_D < 0 for k in was_forward) == 87
    windows = sealed_amplified.windows
    assert sum(w.quantum_lower for w in windows if w.duration_s == 900) == 0
    assert sum(w.quantum_lower for w in windows if w.width_m <= 1000) == 3
    assert sum(w.amplified_cost < w.filter_cost for w in windows) == 11685
    assert sum(w.quantum_route == "QRQ_AA" for w in windows) == 11685
    assert len({p.source_id for p in sealed_amplified.pairs if p.direction == "forward"}) == 12
    alone = collections.Counter(amplified_only_directions(sealed_amplified).values())
    assert (alone["forward"], alone["reverse"]) == (1200, 11)
    for a, b in zip(sealed_reproduction.windows, windows):
        assert (a.quantum_tree, a.C, a.Q) == (b.quantum_tree, b.C, b.filter_cost)


def test_amplified_groups(sealed_amplified, sealed_groups):
    forward = {sealed_groups[p.source_id] for p in sealed_amplified.pairs if p.direction == "forward"}
    assert (len(forward), len(set(sealed_groups.values()))) == (11, 13)


def test_amplified_matches_sealed_tables(sealed_reproduction, sealed_amplified):
    """Per-window AA costs and the per-source table equal the sealed AA tables.

    ``Q_aa`` and its inputs are bit-identical. The sealed ``Q_joint`` and
    ``D_joint`` were computed from first-witness costs parsed with a
    non-round-trip CSV reader, which shifts some values by one unit in the
    last place; they therefore compare within ``4e-16`` relative error. No
    sign, direction or count is affected.
    """
    root = env_dir("QRQ_SEALED_AA")
    ours = {(w.source_id, w.date, w.query_id): w for w in sealed_amplified.windows}
    rows = [r for r in read_csv(root / "aa_endpoints.csv.gz") if r["comparator"] == "envelope"]
    assert len(rows) == len(ours) == PUBLISHED_AA["windows"]
    for r in rows:
        w = ours[(r["source_id"], r["date"], r["query_id"])]
        tree = next(t for t in w.trees if t.tree == w.quantum_tree)
        assert (r["q_tree"], int(r["M"]), int(r["G"]), int(r["address_bits"])) == (
            w.quantum_tree, tree.M, tree.G, tree.bits)
        assert (float(r["P0"]), float(r["Q_aa"]), float(r["C"])) == (tree.P0, w.amplified_cost, w.C)
        assert math.isclose(float(r["Q_joint"]), w.Q, rel_tol=4e-16)
        assert math.isclose(float(r["D_joint"]), w.D, rel_tol=4e-16, abs_tol=1e-15)
        assert (float(r["D_joint"]) < 0) == w.quantum_lower
    legacy = {pair_key(p): p for p in sealed_reproduction.pairs}
    alone = amplified_only_directions(sealed_amplified)
    by_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    days: dict[str, set] = collections.defaultdict(set)
    for p in sealed_amplified.pairs:
        key, old = pair_key(p), legacy[pair_key(p)].direction
        c = by_source[p.source_id]
        days[p.source_id].add(p.date)
        c["pairs"] += 1
        c[f"legacy_{old}"] += 1
        c[f"aa_only_{alone[key]}"] += 1
        c[f"joint_{p.direction}"] += 1
        c["legacy_forward_retained"] += old == "forward" and p.direction == "forward"
        c["legacy_forward_both_now_Q_lower"] += old == "forward" and p.small_D < 0 and p.large_D < 0
    stored = [r for r in read_csv(root / "aa_by_source.csv") if r["comparator"] == "envelope"]
    expected = []
    for row in stored:
        c = by_source[row["source_id"]]
        expected.append({name: (row[name] if name in ("comparator", "source_id") else
                                len(days[row["source_id"]]) if name == "source_days" else int(c[name]))
                         for name in row})
    assert_rows_equal(stored, expected)
