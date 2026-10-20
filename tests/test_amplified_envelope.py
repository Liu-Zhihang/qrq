"""Amplitude amplification as an optional quantum route of the envelope."""
from __future__ import annotations

import math

import pytest

import qrq
from qrq.amplitude import QRQ_AA, amplified_cost, amplified_cost_from_route
from qrq.engine import Envelope, include_amplified, select_envelope
from qrq.ledger import reproduce
from qrq.workload import classify_pair

from synthetic import random_boxes


def first_witness_envelope() -> Envelope:
    return select_envelope([
        ("packed_rtree", {"QRQ_FW": 10.0, "C_BLOCK": 7.0, "C_ENUM": 8.0}),
        ("kd_tree", {"QRQ_FW": 11.0, "C_BLOCK": 9.0, "C_ENUM": 7.5}),
    ])


def test_select_envelope_records_first_witness_cost():
    env = first_witness_envelope()
    assert (env.quantum_route, env.filter_cost, env.amplified_cost) == ("QRQ_FW", 10.0, None)


def test_include_amplified_takes_strictly_cheaper_route():
    env = first_witness_envelope()
    lower = include_amplified(env, 4.0)
    assert (lower.Q, lower.quantum_route, lower.filter_cost, lower.amplified_cost) == (4.0, QRQ_AA, 10.0, 4.0)
    assert (lower.C, lower.quantum_tree, lower.classical_tree, lower.classical_route) == (
        env.C, env.quantum_tree, env.classical_tree, env.classical_route)
    assert lower.D == math.log2(4.0 / 7.0) and lower.quantum_lower
    for cost in (10.0, 12.5):
        kept = include_amplified(env, cost)
        assert (kept.Q, kept.quantum_route, kept.filter_cost, kept.amplified_cost) == (10.0, "QRQ_FW", 10.0, cost)
    with pytest.raises(ValueError):
        include_amplified(lower, 1.0)


def test_include_amplified_on_manual_envelope():
    env = Envelope(Q=9.0, C=3.0, quantum_tree="kd_tree", classical_tree="kd_tree", classical_route="C_BLOCK")
    assert include_amplified(env, 8.0).filter_cost == 9.0
    assert include_amplified(env, 9.5).Q == 9.0


@pytest.mark.parametrize("prices", [qrq.L1, qrq.UNIT])
def test_evaluate_window_amplified(indexed_cloud, prices):
    points, indexes = indexed_cloud
    switched = 0
    for box in random_boxes(points, 60, seed=17):
        plain = qrq.evaluate_window(indexes, box, prices)
        joint = qrq.evaluate_window(indexes, box, prices, amplified=True)
        assert joint.k == plain.k and joint.costs == plain.costs
        if not plain.k:
            assert (joint.Q, joint.quantum_route, joint.amplified_cost) == (None, None, None)
            continue
        assert (plain.quantum_route, plain.filter_cost, plain.amplified_cost) == ("QRQ_FW", plain.Q, None)
        assert (joint.quantum_tree, joint.C, joint.classical_tree, joint.classical_route) == (
            plain.quantum_tree, plain.C, plain.classical_tree, plain.classical_route)
        q_row = next(r for r in joint.routes[joint.quantum_tree] if r["route"] == "QRQ_FW")
        expected = amplified_cost_from_route(q_row, joint.k, prices).cost
        assert joint.amplified_cost == expected and joint.filter_cost == plain.Q
        assert joint.Q == min(plain.Q, expected)
        assert joint.quantum_route == (QRQ_AA if expected < plain.Q else "QRQ_FW")
        assert joint.D == math.log2(joint.Q / joint.C)
        switched += joint.quantum_route == QRQ_AA
    assert switched


def test_evaluate_window_amplified_requires_unit_weights(indexed_cloud):
    points, indexes = indexed_cloud
    box = qrq.SpaceTimeBox(0, 10000, 0, 10000, 0, 43200)
    with pytest.raises(ValueError):
        qrq.evaluate_window(indexes, box, qrq.PriceTable(1, 100, 10000, 2, 1), amplified=True)


def test_ledger_reproduction_with_amplification(synthetic_ledgers):
    plain = reproduce(synthetic_ledgers)
    joint = reproduce(synthetic_ledgers, amplified=True)
    assert (plain.amplified, joint.amplified) == (False, True)
    assert [w.query_id for w in plain.windows] == [w.query_id for w in joint.windows]
    for a, b in zip(plain.windows, joint.windows):
        assert (a.quantum_route, a.filter_cost, a.amplified_cost) == ("QRQ_FW", a.Q, None)
        tree = next(t for t in b.trees if t.tree == b.quantum_tree)
        expected = amplified_cost(tree.M, b.k, float(tree.G), tree.bits, tree.P0).cost
        assert b.amplified_cost == expected and b.filter_cost == a.Q
        assert (b.Q, b.C, b.quantum_tree) == (min(a.Q, expected), a.C, a.quantum_tree)
        assert b.quantum_route == (QRQ_AA if expected < a.Q else "QRQ_FW")
    assert any(w.quantum_route == QRQ_AA for w in joint.windows)
    d = {(w.source_id, w.date, w.query_id): w.D for w in joint.windows}
    assert len(joint.pairs) == len(plain.pairs) > 0
    for p in joint.pairs:
        assert (p.small_D, p.large_D) == (d[(p.source_id, p.date, p.small_query)],
                                          d[(p.source_id, p.date, p.large_query)])
        assert p.direction == classify_pair(p.small_D, p.large_D)
