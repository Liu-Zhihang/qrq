"""Index answers on seeded synthetic clouds versus brute force."""
from __future__ import annotations

import numpy as np
import pytest

import qrq
from qrq.routes import C_BLOCK, C_ENUM, QRQ_FW

from synthetic import random_boxes


def rank_set(ranges) -> set[int]:
    return {r for lo, hi in ranges for r in range(lo, hi)}


@pytest.mark.parametrize("kind", qrq.TREE_KINDS)
def test_domains_match_brute_force(cloud, kind):
    index = qrq.SpatioTemporalIndex.build(cloud, kind=kind)
    n = index.size
    assert n == len(cloud)
    layout = index.rank_to_record()
    assert sorted(layout.tolist()) == list(range(n))
    nonempty = 0
    for box in random_boxes(cloud, 150, seed=n + len(kind)):
        truth_mask = box.contains(cloud)
        truth = int(truth_mask.sum())
        plan = index.plan(box)
        k, rows = qrq.evaluate_routes(plan, n)
        domains = {row["route"]: row["domain"] for row in rows}
        assert k == truth == index.count(box)
        assert domains[C_BLOCK] == domains[C_ENUM] == k
        assert domains["C_TABLE_BLOCK"] == domains["C_TABLE_ENUM"] == k
        answer = rank_set(plan.block.ranges)
        assert answer == rank_set(plan.enumeration.ranges)
        assert set(layout[sorted(answer)].tolist()) == set(np.flatnonzero(truth_mask).tolist())
        if k:
            nonempty += 1
            assert domains[QRQ_FW] >= k
            assert rank_set(plan.candidate_ranges) >= answer
            assert plan.depth >= 0 and plan.witness_kind != "none"
        else:
            assert domains[QRQ_FW] == 0
            assert plan.candidate_ranges == ()
            assert plan.depth == -1 and plan.witness_kind == "none"
    if n > 100:
        assert nonempty > 20


def test_tb_tree_uses_entity_ids():
    points = qrq.make_points(np.arange(16.0), np.zeros(16), np.zeros(16), np.repeat([0, 1], 8))
    swapped = qrq.make_points(np.arange(16.0), np.zeros(16), np.zeros(16), np.tile([0, 1], 8))
    a = qrq.SpatioTemporalIndex.build(points, kind="tb_tree")
    b = qrq.SpatioTemporalIndex.build(swapped, kind="tb_tree")
    assert not np.array_equal(a.rank_to_record(), b.rank_to_record())


def test_plain_coordinates_require_ids():
    xyz = np.zeros((3, 3))
    with pytest.raises(ValueError):
        qrq.SpatioTemporalIndex.build(xyz)
    index = qrq.SpatioTemporalIndex.build(xyz, np.arange(3, dtype=np.int64))
    assert index.count(qrq.SpaceTimeBox(0, 0, 0, 0, 0, 0)) == 3


def test_build_rejects_bad_input():
    with pytest.raises(ValueError):
        qrq.SpatioTemporalIndex.build(np.empty(0, dtype=qrq.POINT))
    with pytest.raises(ValueError):
        qrq.SpatioTemporalIndex.build(np.zeros((2, 3)), np.zeros(2, dtype=np.int64), kind="quadtree")
    with pytest.raises(ValueError):
        qrq.SpatioTemporalIndex.build(np.zeros((2, 3)), np.zeros(2, dtype=np.int32))


def test_plan_round_trip(indexed_cloud):
    points, indexes = indexed_cloud
    box = qrq.SpaceTimeBox(0, 10000, 0, 10000, 20000, 60000)
    assert box.count(points) > 0
    for index in indexes.values():
        raw = index.matched_query(box)
        plan = qrq.AccessPlan.from_matched(raw)
        assert plan.to_matched() == raw
        assert qrq.evaluate_routes(plan, index.size) == qrq.evaluate_routes(raw, index.size)
        assert plan.frontier_size >= 1 and plan.height >= 1
