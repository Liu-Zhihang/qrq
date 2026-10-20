"""Price tables and the scalar route cost.

A route row carries two resource vectors. The scalar cost is

.. math::

    \\mathrm{cost} = n_{nodes} + n_{slots} + g_c\\,n_{geom} + n_{meta}
        + \\lambda r\\,(G_{prep} + L\\,n_{lookup} + g_q\\,n_{qgeom}
        + n_{meas} + n_{reset})

with classical geometry price ``g_c``, coherent record lookup price ``L``,
coherent geometry predicate price ``g_q``, quantum weight ``lambda`` and
repetition count ``r``. :func:`route_cost` delegates to the vendored
reference ``price()`` so that results are bit-identical.

For the quantum route with ``k > 0`` the cost separates as
``Q = P + (M/K) H`` (see :func:`decompose_quantum_cost`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ._vendor.qrq_matched import price as _reference_price


@dataclass(frozen=True)
class PriceTable:
    """Unit prices of the cost model (dimensionless scenario values)."""

    g_classical: float = 1
    lookup: float = 1
    g_quantum: float = 1
    lam: float = 1
    repeats: float = 1

    def as_kwargs(self) -> dict[str, float]:
        """Keyword arguments for the reference ``price()`` function."""
        return asdict(self)


UNIT = PriceTable(g_classical=1, lookup=1, g_quantum=1, lam=1, repeats=1)
"""Every primitive costs one unit."""

L1 = PriceTable(g_classical=1, lookup=100, g_quantum=10000, lam=1, repeats=1)
"""Main published scenario: expensive coherent record access and geometry."""


def route_cost(row: Mapping[str, Any], prices: PriceTable = L1) -> float:
    """Scalar cost of one route resource row under ``prices``."""
    return _reference_price(row, **prices.as_kwargs())


def route_costs(rows: list[Mapping[str, Any]], prices: PriceTable = L1) -> dict[str, float]:
    """Map route name to scalar cost for a list of resource rows."""
    return {row["route"]: route_cost(row, prices) for row in rows}


@dataclass(frozen=True)
class QuantumCostDecomposition:
    """``Q = P + a * H`` for a quantum route with a non-empty answer.

    ``P`` is the trial-independent prefix (classical frontier discovery,
    preparation metadata and the two fixed lookups, minus the shared initial
    rank register), ``a = M/K`` the expected number of trials and ``H`` the
    cost of one trial (preparation gates, two lookups, one coherent
    geometry test, one measurement and one rank-register reset).
    """

    prefix: float
    trials: float
    trial_cost: float

    @property
    def total(self) -> float:
        return self.prefix + self.trials * self.trial_cost


def decompose_quantum_cost(
    row: Mapping[str, Any], prices: PriceTable = L1
) -> QuantumCostDecomposition:
    """Split a ``QRQ_FW`` row cost into ``P + (M/K) H``.

    Only defined for rows with a non-empty answer (``expected_trials > 0``).
    """
    if row.get("route") != "QRQ_FW":
        raise ValueError("decomposition is defined for the QRQ_FW route only")
    trials = row["expected_trials"]
    if not trials > 0:
        raise ValueError("decomposition requires a non-empty answer")
    c = row["classical"]
    bits = row["shared_initial_rank_bits"]
    gates = row["preparation"]["gate_total"]
    weight = prices.lam * prices.repeats
    classical = c["nodes"] + c["slots"] + prices.g_classical * c["geometry"] + c["metadata"]
    prefix = classical + weight * (2 * prices.lookup - bits)
    trial_cost = weight * (gates + 2 * prices.lookup + prices.g_quantum + 1 + bits)
    return QuantumCostDecomposition(prefix=prefix, trials=trials, trial_cost=trial_cost)
