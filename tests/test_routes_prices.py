"""Route rows and prices against the reference definitions."""
from __future__ import annotations

import math

import numpy as np
import pytest

import qrq
from qrq._vendor.qrq_matched import compile_intervals, price, route_resources
from qrq._vendor.table_solver_contract import table_from_classical_solver
from qrq.prices import decompose_quantum_cost

from synthetic import random_boxes


def reference_routes_for(plan, n):
    """Verbatim logic of the reference ``routes_for(plan, n)``."""
    bits = max(1, (n - 1).bit_length())
    prepared = {name: compile_intervals(ranges, bits, False) for name, ranges in
                [('QRQ_FW', plan['candidate_ranges']), ('C_BLOCK', plan['block']['ranges']),
                 ('C_ENUM', plan['enumeration']['ranges'])]}
    assert prepared['C_BLOCK']['ranges'] == prepared['C_ENUM']['ranges']
    k = prepared['C_BLOCK']['domain']
    classical = {'QRQ_FW': dict(node_reads=plan['prefix_nodes'], slots=plan['prefix_slots'],
                                geometry=plan['prefix_geometry']),
                 'C_BLOCK': plan['block'], 'C_ENUM': plan['enumeration']}
    routes = [route_resources(classical[name], prepared[name], k, name, bits) for name in prepared]
    routes.extend([table_from_classical_solver(n, routes[i], 4) for i in (1, 2)])
    for row in routes:
        row['preparation'] = {key: value for key, value in row['preparation'].items()
                              if key not in ('ranges', 'program')}
    return k, routes


def l1_independent(row):
    """Independent L1 price used by the sealed analysis."""
    c = row["classical"]
    q = row["quantum"]
    return (sum(c[k] for k in ("nodes", "slots", "geometry", "metadata")) + q["prepare_gate_total"]
            + 100 * q["lookup_calls"] + 10000 * q["geometry_calls"] + q["measurements"]
            + q["reset_rank_bits"])


def test_evaluate_routes_identical_to_reference(indexed_cloud):
    points, indexes = indexed_cloud
    for kind, index in indexes.items():
        for box in random_boxes(points, 60, seed=len(kind)):
            raw = index.matched_query(box)
            ours = qrq.evaluate_routes(index.plan(box), index.size)
            assert ours == reference_routes_for(raw, index.size)
            assert [row["route"] for row in ours[1]] == list(qrq.ROUTES)


def random_row(rng):
    return dict(
        route="QRQ_FW",
        classical=dict(nodes=int(rng.integers(0, 10**6)), slots=int(rng.integers(0, 10**6)),
                       geometry=int(rng.integers(0, 10**6)), metadata=int(rng.integers(0, 10**6))),
        quantum=dict(prepare_gate_total=float(rng.uniform(0, 1e7)), lookup_calls=float(rng.uniform(0, 1e3)),
                     geometry_calls=float(rng.uniform(0, 1e3)), measurements=float(rng.uniform(0, 1e3)),
                     reset_rank_bits=float(rng.uniform(0, 1e4))),
    )


def test_route_cost_equals_reference_price():
    rng = np.random.default_rng(20260001)
    for _ in range(2000):
        row = random_row(rng)
        prices = qrq.PriceTable(
            g_classical=float(rng.choice([1, 2.5, rng.uniform(0, 50)])),
            lookup=float(rng.choice([1, 100, rng.uniform(0, 1e3)])),
            g_quantum=float(rng.choice([1, 10000, rng.uniform(0, 1e5)])),
            lam=float(rng.choice([1, 0.5, rng.uniform(0, 3)])),
            repeats=int(rng.integers(1, 5)),
        )
        assert qrq.route_cost(row, prices) == price(row, **prices.as_kwargs())
        c, q = row["classical"], row["quantum"]
        explicit = (c["nodes"] + c["slots"] + prices.g_classical * c["geometry"] + c["metadata"]
                    + prices.lam * prices.repeats * (q["prepare_gate_total"] + prices.lookup * q["lookup_calls"]
                                                     + prices.g_quantum * q["geometry_calls"]
                                                     + q["measurements"] + q["reset_rank_bits"]))
        assert math.isclose(qrq.route_cost(row, prices), explicit, rel_tol=1e-12)
        assert qrq.route_cost(row, qrq.L1) == price(row, g_classical=1, lookup=100, g_quantum=10000)
        assert math.isclose(qrq.route_cost(row, qrq.L1), l1_independent(row), rel_tol=1e-13)
        assert qrq.route_cost(row, qrq.UNIT) == price(row)


def test_presets():
    assert qrq.L1 == qrq.PriceTable(1, 100, 10000, 1, 1)
    assert qrq.UNIT == qrq.PriceTable(1, 1, 1, 1, 1)
    assert qrq.L1.as_kwargs() == dict(g_classical=1, lookup=100, g_quantum=10000, lam=1, repeats=1)


@pytest.mark.parametrize("prices", [qrq.L1, qrq.UNIT, qrq.PriceTable(3, 7, 50, 0.5, 2)])
def test_quantum_cost_decomposition(indexed_cloud, prices):
    points, indexes = indexed_cloud
    checked = 0
    for index in indexes.values():
        for box in random_boxes(points, 40, seed=99):
            k, rows = qrq.evaluate_routes(index.plan(box), index.size)
            q_row = rows[0]
            if not k:
                with pytest.raises(ValueError):
                    decompose_quantum_cost(q_row, prices)
                continue
            parts = decompose_quantum_cost(q_row, prices)
            assert parts.trials == q_row["domain"] / k
            assert math.isclose(parts.total, qrq.route_cost(q_row, prices), rel_tol=1e-12)
            checked += 1
    assert checked > 0
