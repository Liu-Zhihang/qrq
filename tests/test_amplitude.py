"""Unknown-K amplitude amplification route."""
from __future__ import annotations

import math

import pytest

import qrq
from qrq import amplitude
from qrq._vendor import aa_challenger


def original_cost(M, K, G, bits, P0):
    """The reference cost expression with its hard-coded L1 prices."""
    e = aa_challenger.expectation(int(M), int(K))
    a = e["attempts"]
    j = e["iterations"]
    marks = a + j
    return (P0 + (a + 2 * j) * G + 100 * (2 * marks + 2) + 10000 * marks
            + j * aa_challenger.reflection_gates(bits) + 4 * j + a + bits * (a - 1) + e["control"])


def test_policy_windows_cap():
    for M in range(1, 200):
        windows = amplitude.policy_windows(M)
        assert windows[-1] == math.ceil(math.sqrt(M))
        assert windows == sorted(windows)


def test_uniform_success_matches_direct_sum():
    for M in (1, 7, 31):
        for K in range(1, M + 1):
            theta = math.asin(math.sqrt(K / M))
            for w in set(amplitude.policy_windows(M)):
                direct = sum(math.sin((2 * j + 1) * theta) ** 2 for j in range(w)) / w
                assert abs(direct - amplitude.uniform_success(K / M, w)) < 2e-13


@pytest.mark.parametrize("M,K,G,bits,P0", [(1, 1, 10.0, 1, 3), (80, 25, 6106.0, 21, 727), (4096, 3, 1.5e4, 12, 99)])
def test_default_prices_reproduce_reference(M, K, G, bits, P0):
    result = amplitude.amplified_cost(M, K, G, bits, P0, qrq.L1)
    assert result.cost == original_cost(M, K, G, bits, P0)
    assert result.cost == aa_challenger.price(M, K, G, bits, P0)[0]
    assert result.expectation["attempts"] >= 1 - 1e-12


def test_prices_are_parameters():
    base = amplitude.amplified_cost(64, 4, 100.0, 6, 10, qrq.L1).cost
    cheaper = amplitude.amplified_cost(64, 4, 100.0, 6, 10, qrq.PriceTable(1, 1, 1, 1, 1)).cost
    assert cheaper < base
    with pytest.raises(ValueError):
        amplitude.amplified_cost(64, 4, 100.0, 6, 10, qrq.PriceTable(1, 100, 10000, 2, 1))
    with pytest.raises(ValueError):
        amplitude.expectation(4, 0)


def test_from_route_row(indexed_cloud):
    points, indexes = indexed_cloud
    box = qrq.SpaceTimeBox(0, 10000, 0, 10000, 0, 43200)
    index = indexes["packed_rtree"]
    k, rows = qrq.evaluate_routes(index.plan(box), index.size)
    assert k > 0
    row = rows[0]
    c = row["classical"]
    p0 = c["nodes"] + c["slots"] + c["geometry"] + c["metadata"]
    expected = original_cost(row["domain"], k, float(row["preparation"]["gate_total"]),
                             row["shared_initial_rank_bits"], p0)
    assert amplitude.amplified_cost_from_route(row, k).cost == expected
    with pytest.raises(ValueError):
        amplitude.amplified_cost_from_route(rows[1], k)
