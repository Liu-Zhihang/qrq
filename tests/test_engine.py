"""Window envelope selection and evaluation."""
from __future__ import annotations

import math

import pytest

import qrq
from qrq.engine import select_envelope

from synthetic import random_boxes


def test_select_envelope_tie_breaking():
    costs = [
        ("packed_rtree", {"QRQ_FW": 10.0, "C_BLOCK": 7.0, "C_ENUM": 9.0, "C_TABLE_BLOCK": 7.0, "C_TABLE_ENUM": 8.0}),
        ("tb_tree", {"QRQ_FW": 10.0, "C_BLOCK": 7.0, "C_ENUM": 7.0, "C_TABLE_BLOCK": 9.0, "C_TABLE_ENUM": 8.0}),
        ("kd_tree", {"QRQ_FW": 11.0, "C_BLOCK": 7.0, "C_ENUM": 12.0, "C_TABLE_BLOCK": 9.0, "C_TABLE_ENUM": 8.0}),
    ]
    env = select_envelope(costs)
    assert env.Q == 10.0 and env.quantum_tree == "packed_rtree"
    assert env.C == 7.0 and (env.classical_tree, env.classical_route) == ("kd_tree", "C_BLOCK")
    assert env.D == math.log2(10.0 / 7.0) and not env.quantum_lower
    env = select_envelope(reversed(costs))
    assert env.quantum_tree == "tb_tree"


def test_evaluate_window_matches_manual_envelope(indexed_cloud):
    points, indexes = indexed_cloud
    seen_nonempty = seen_empty = 0
    for box in random_boxes(points, 80, seed=5):
        result = qrq.evaluate_window(indexes, box, qrq.L1, verify=True)
        assert result.verified and result.k == box.count(points)
        manual = {}
        for kind, index in indexes.items():
            k, rows = qrq.evaluate_routes(index.plan(box), index.size)
            assert k == result.k
            manual[kind] = {row["route"]: qrq.route_cost(row, qrq.L1) for row in rows}
        assert result.costs == manual
        if not result.k:
            seen_empty += 1
            assert result.Q is None and result.C is None and result.D is None
            assert not result.quantum_lower
            continue
        seen_nonempty += 1
        q = min(c["QRQ_FW"] for c in manual.values())
        c = min(v for costs in manual.values() for name, v in costs.items() if name != "QRQ_FW")
        assert (result.Q, result.C) == (q, c)
        assert result.D == math.log2(q / c)
        assert manual[result.quantum_tree]["QRQ_FW"] == q
        assert manual[result.classical_tree][result.classical_route] == c
    assert seen_nonempty and seen_empty


def test_evaluate_window_rejects_mismatched_indexes(indexed_cloud):
    points, indexes = indexed_cloud
    other = qrq.SpatioTemporalIndex.build(points[:100], kind="kd_tree")
    with pytest.raises(ValueError):
        qrq.evaluate_window({"packed_rtree": indexes["packed_rtree"], "kd_tree": other},
                            qrq.SpaceTimeBox(0, 1, 0, 1, 0, 1))
    with pytest.raises(ValueError):
        qrq.evaluate_window({}, qrq.SpaceTimeBox(0, 1, 0, 1, 0, 1))
