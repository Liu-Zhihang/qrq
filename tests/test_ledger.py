"""Ledger re-pricing and per-tree endpoints on synthetic ledgers."""
from __future__ import annotations

import csv
import json
import math

import qrq
from qrq.engine import select_envelope
from qrq.ledger import find_ledger_days, main, read_ledger, reproduce
from qrq.routes import rank_bits

from synthetic import LEDGER_DAYS


def test_synthetic_reproduction(synthetic_ledgers):
    r = reproduce(synthetic_ledgers)
    assert r.days == len(LEDGER_DAYS) and r.ledgers == 3 * len(LEDGER_DAYS)
    assert r.exact_cost_matches == r.priced_routes > 0 and r.max_relative_error == 0.0
    assert r.prices == qrq.L1 and r.window_count > 0 and r.pairs
    for w in r.windows:
        assert math.isclose(w.Q, w.prefix + w.trials * w.trial_cost, rel_tol=1e-12)


def test_tree_endpoints_match_ledger_rows(synthetic_ledgers):
    r = reproduce(synthetic_ledgers)
    windows = {(w.source_id, w.date, w.query_id): w for w in r.windows}
    checked = 0
    for day in find_ledger_days(synthetic_ledgers):
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


def test_command_line(synthetic_ledgers, tmp_path, capsys):
    out = tmp_path / "windows.csv"
    assert main(["--ledgers", str(synthetic_ledgers), "--windows-csv", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    r = reproduce(synthetic_ledgers)
    assert summary["windows"] == r.window_count and summary["quantum_lower"] == r.quantum_lower_count
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == r.window_count
    assert "trees" not in rows[0] and {"query_id", "Q", "C", "D"} <= set(rows[0])
