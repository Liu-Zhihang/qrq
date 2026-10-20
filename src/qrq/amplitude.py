"""Unknown-K randomized amplitude amplification as an alternative quantum route.

Instead of repeating independent first-witness trials (``M/K`` expected),
the candidate domain of size ``M`` can be searched with the exponential
window schedule of unknown-``K`` amplitude amplification. The schedule is
chosen without knowledge of ``K``; ``K`` is only used to evaluate the
expected cost offline.

The cost of the amplified route is

``P0 + G (a + 2j) + L (2 (a + j) + 2) + g_q (a + j) + R j + 4 j + a
+ b (a - 1) + control``

for ``a`` expected attempts, ``j`` expected Grover iterations, preparation
gate count ``G``, zero-reflection gate count ``R``, rank bits ``b`` and the
expected classical schedule control cost. It delegates to the vendored
reference ``price()``; the lookup and coherent geometry prices come from a
:class:`qrq.prices.PriceTable`. The reference model has no quantum weight or
repetition factor, so ``lam`` and ``repeats`` must both equal 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._vendor import aa_challenger as _aa
from .prices import L1, PriceTable

QRQ_AA = "QRQ_AA"
"""Route label of amplitude amplification over the first-witness frontier."""


def policy_windows(M: int) -> list[int]:
    """Iteration windows of the unknown-K schedule for a domain of size ``M``."""
    return list(_aa.policy_windows(M))


def uniform_success(p: float, w: int) -> float:
    """Success probability of one attempt with a uniformly drawn iteration count < ``w``."""
    return _aa.uniform_success(p, w)


def expectation(M: int, K: int) -> dict[str, float]:
    """Expected attempts, iterations and control cost for ``0 < K <= M``."""
    return dict(_aa.expectation(int(M), int(K)))


@dataclass(frozen=True)
class AmplifiedCost:
    """Scalar cost, schedule expectation and resource vector of the route."""

    cost: float
    expectation: dict[str, float]
    resources: dict[str, float]


def _check_prices(prices: PriceTable) -> None:
    if prices.lam != 1 or prices.repeats != 1:
        raise ValueError("the amplified route is defined for lam == repeats == 1 only")


def amplified_cost(
    M: int, K: int, G: float, bits: int, P0: float, prices: PriceTable = L1
) -> AmplifiedCost:
    """Expected cost of amplitude amplification over ``M`` candidates.

    ``G`` is the preparation gate count, ``bits`` the rank register width and
    ``P0`` the classical prefix cost (frontier discovery plus preparation
    metadata).
    """
    _check_prices(prices)
    cost, expected, vector = _aa.price(
        M, K, G, bits, P0, lookup=prices.lookup, g_quantum=prices.g_quantum
    )
    return AmplifiedCost(cost=cost, expectation=dict(expected), resources=dict(vector))


def amplified_cost_from_route(
    row: Mapping[str, Any], k: int, prices: PriceTable = L1
) -> AmplifiedCost:
    """Price amplitude amplification on the frontier of a ``QRQ_FW`` row.

    ``M``, ``G`` and ``bits`` are read from the row; ``P0`` is its classical
    resource cost ``nodes + slots + g_classical * geometry + metadata``.
    """
    if row.get("route") != "QRQ_FW":
        raise ValueError("a QRQ_FW route row is required")
    c = row["classical"]
    p0 = c["nodes"] + c["slots"] + prices.g_classical * c["geometry"] + c["metadata"]
    return amplified_cost(
        int(row["domain"]),
        int(k),
        float(row["preparation"]["gate_total"]),
        int(row["shared_initial_rank_bits"]),
        p0,
        prices,
    )
