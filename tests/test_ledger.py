"""Ledger row schema, writer, reader and re-pricing audit on synthetic ledgers."""
from __future__ import annotations

import math

import pytest

import qrq
from qrq.engine import select_envelope
from qrq.ledger import (ELIGIBLE_EMPTY, ELIGIBLE_NONEMPTY, LedgerDay, audit_cohort, audit_day, ledger_row,
                        pair_endpoints, read_ledger, write_ledger)
from qrq.routes import rank_bits
from qrq.workload import UNSUPPORTED_INDEX_SPAN, WindowGrid, classify_pair

from synthetic import LEDGER_DAYS, LEDGER_GRID


def test_ledger_row_schema(indexed_cloud, tmp_path):
    points, indexes = indexed_cloud
    grid = WindowGrid(widths=(500, 3000), durations=(3600, 86400), centres=(3600, 43200), anchor_step=2000.0)
    windows = qrq.register_windows(5000.0, 5000.0, grid)
    index = indexes["kd_tree"]
    rows = [ledger_row(w, index, qrq.UNIT, source_id="s", arm="a") for w in windows]
    statuses = {row["result_status"] for row in rows}
    assert {UNSUPPORTED_INDEX_SPAN, ELIGIBLE_NONEMPTY} <= statuses
    for w, row in zip(windows, rows):
        assert (row["query_id"], row["tree"], row["n"], row["source_id"], row["arm"]) == (
            w.query_id, "kd_tree", index.size, "s", "a")
        assert (row["anchor_index"], row["width_m"], row["duration_s"], row["temporal_center_s"]) == (
            list(w.anchor_index), w.width_m, w.duration_s, w.temporal_center_s)
        if not w.supported:
            assert row["result_status"] == UNSUPPORTED_INDEX_SPAN and "routes" not in row
            continue
        k, routes = qrq.evaluate_routes(index.plan(w.box), index.size)
        assert (row["k"], row["routes"]) == (k, routes)
        assert row["costs"] == qrq.route_costs(routes, qrq.UNIT)
        assert row["result_status"] == (ELIGIBLE_NONEMPTY if k else ELIGIBLE_EMPTY)
    path = write_ledger(tmp_path / "nested" / "kd_tree.jsonl.gz", rows)
    assert list(read_ledger(path)) == rows


def test_synthetic_audit(synthetic_ledgers):
    r = audit_cohort(synthetic_ledgers, LEDGER_GRID)
    assert r.days == len(LEDGER_DAYS) and r.ledgers == 3 * len(LEDGER_DAYS)
    assert r.exact_cost_matches == r.priced_routes > 0 and r.max_relative_error == 0.0
    assert r.prices == qrq.L1 and r.window_count > 0 and r.pairs
    for w in r.windows:
        assert math.isclose(w.Q, w.prefix + w.trials * w.trial_cost, rel_tol=1e-12)
        assert w.filter_cost == w.Q and (w.arm, w.variant) == ("baseline", "base")
    for p in r.pairs:
        assert p.direction == classify_pair(p.small_D, p.large_D)
    summary = r.summary()
    assert summary["windows"] == r.window_count and sum(s["windows"] for s in summary["by_source"].values()) == \
        r.window_count


def test_tree_endpoints_match_ledger_rows(synthetic_ledgers):
    r = audit_cohort(synthetic_ledgers, LEDGER_GRID)
    windows = {(w.source_id, w.date, w.query_id): w for w in r.windows}
    checked = 0
    for day in synthetic_ledgers:
        for tree, path in day.paths.items():
            for row in read_ledger(path):
                w = windows.get((day.source_id, day.date, row["query_id"]))
                if w is None:
                    continue
                t = next(t for t in w.trees if t.tree == tree)
                q_row = row["routes"][0]
                c = q_row["classical"]
                assert (t.M, t.G, t.bits) == (q_row["domain"], q_row["preparation"]["gate_total"], rank_bits(row["n"]))
                assert t.P0 == c["nodes"] + c["slots"] + c["geometry"] + c["metadata"]
                assert t.Q == row["costs"]["QRQ_FW"]
                assert t.C == min(v for name, v in row["costs"].items() if name != "QRQ_FW")
                assert t.classical_lookups == 2.0
                checked += 1
    assert checked == 3 * r.window_count
    for w in r.windows:
        assert [t.tree for t in w.trees] == list(qrq.TREE_KINDS)
        env = select_envelope((t.tree, {"QRQ_FW": t.Q, "C_ANY": t.C}) for t in w.trees)
        assert (env.Q, env.C, env.quantum_tree) == (w.Q, w.C, w.quantum_tree)


def test_arm_selection_and_rows_without_metadata(indexed_cloud, tmp_path):
    points, indexes = indexed_cloud
    grid = WindowGrid(widths=(2000, 6000), durations=(21600, 43200), centres=(43200,), anchor_step=3000.0,
                      anchors=1)
    windows = qrq.register_windows(5000.0, 5000.0, grid)
    paths = {}
    for kind, index in indexes.items():
        rows = [ledger_row(w, index, arm=arm, variant="v") for arm in ("first", "second") for w in windows]
        for row in rows:
            row["query_id"] = f"{row['arm']}_{row['query_id']}"
        sentinel = ledger_row(windows[-1], index)
        for name in ("anchor_index", "width_m", "duration_s", "temporal_center_s"):
            del sentinel[name]
        sentinel["query_id"] = "sentinel"
        paths[kind] = write_ledger(tmp_path / f"{kind}.jsonl.gz", rows + [sentinel])
    day = LedgerDay("s", "d", paths)
    every = audit_day(day)
    nonempty = sum(1 for w in windows if w.supported) * 2
    assert len(every.endpoints) <= nonempty and every.rows == 3 * (2 * len(windows) + 1)
    assert "sentinel" not in {e.query_id for e in every.endpoints}
    only_first = audit_day(day, arms=("first",))
    assert {e.arm for e in only_first.endpoints} <= {"first"}
    assert len(only_first.endpoints) * 2 == len(every.endpoints)
    cohort = audit_cohort([day], grid, arm="second", variant="v")
    assert {w.arm for w in cohort.windows} == {"second"} and len(cohort.windows) * 2 == len(every.endpoints)
    assert all(p.small_query.startswith("second_") for p in cohort.pairs)
    assert audit_cohort([day], grid, variant="other").window_count == 0


def test_pair_endpoints_rejects_shrinking_answers(synthetic_ledgers):
    r = audit_cohort(synthetic_ledgers[:1], LEDGER_GRID)
    pair = next(p for p in r.pairs)
    small = next(w for w in r.windows if w.query_id == pair.small_query)
    large = next(w for w in r.windows if w.query_id == pair.large_query)
    broken = qrq.ledger.LedgerEndpoint(**{**large.__dict__, "k": small.k - 1})
    with pytest.raises(AssertionError):
        pair_endpoints([small, broken], LEDGER_GRID)
