"""Per-window evaluation across several indexes: the cost envelope.

For one window the engine evaluates every route on every index and forms

* ``Q``: the minimum quantum-filtering (``QRQ_FW``) cost over the indexes;
* ``C``: the minimum over the indexes and the four classical routes
  ``C_BLOCK``, ``C_ENUM``, ``C_TABLE_BLOCK`` and ``C_TABLE_ENUM``;
* ``D = log2(Q / C)``; ``D < 0`` means the quantum route is cheaper.

Tie-breaking reproduces the sealed analysis: the quantum tree is the first
minimal tree in iteration order, and the classical choice minimises the
tuple ``(cost, tree, route)``. Empty windows (``k == 0``) have no envelope.

Optionally the envelope also admits unknown-K amplitude amplification
(``QRQ_AA``, :mod:`qrq.amplitude`) as published in the sensitivity
analysis: the amplified route is priced on the quantum tree already selected
by the first-witness costs and ``Q = min(Q_FW, Q_AA)``
(:func:`include_amplified`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from .amplitude import QRQ_AA, amplified_cost_from_route
from .geometry import SpaceTimeBox
from .index import SpatioTemporalIndex
from .prices import L1, PriceTable, route_costs
from .routes import QRQ_FW, evaluate_routes


@dataclass(frozen=True)
class Envelope:
    """Envelope of one non-empty window.

    ``quantum_route`` is the quantum route that attains ``Q``.
    ``filter_cost`` is the first-witness (``QRQ_FW``) cost on
    ``quantum_tree``; ``amplified_cost`` is the amplitude-amplification cost
    on the same tree, or ``None`` when that route was not evaluated.
    """

    Q: float
    C: float
    quantum_tree: str
    classical_tree: str
    classical_route: str
    quantum_route: str = QRQ_FW
    filter_cost: float | None = None
    amplified_cost: float | None = None

    @property
    def D(self) -> float:
        """``log2(Q / C)``."""
        return math.log2(self.Q / self.C)

    @property
    def quantum_lower(self) -> bool:
        """True when the quantum route is strictly cheaper (``Q < C``)."""
        return self.Q < self.C


def select_envelope(costs_by_tree: Iterable[tuple[str, Mapping[str, float]]]) -> Envelope:
    """Select ``Q`` and ``C`` from ``(tree, {route: cost})`` pairs.

    The iteration order of ``costs_by_tree`` is the quantum tie-break order;
    the sealed analysis used ``packed_rtree, tb_tree, kd_tree``.
    """
    candidates = list(costs_by_tree)
    if not candidates:
        raise ValueError("at least one tree is required")
    q_tree, q_costs = min(candidates, key=lambda item: item[1][QRQ_FW])
    choices = [
        (value, tree, name)
        for tree, costs in candidates
        for name, value in costs.items()
        if name != QRQ_FW
    ]
    c_value, c_tree, c_route = min(choices, key=lambda item: item[:3])
    return Envelope(
        Q=q_costs[QRQ_FW],
        C=c_value,
        quantum_tree=q_tree,
        classical_tree=c_tree,
        classical_route=c_route,
        filter_cost=q_costs[QRQ_FW],
    )


def include_amplified(env: Envelope, amplified_cost: float) -> Envelope:
    """Admit the amplitude-amplification route into a first-witness envelope.

    ``amplified_cost`` must be the ``QRQ_AA`` cost on ``env.quantum_tree``.
    This is the published sensitivity rule: the quantum tree is not
    re-optimised, ``Q = min(Q_FW, Q_AA)`` on that tree, and ``quantum_route``
    becomes ``QRQ_AA`` only when the amplified cost is strictly lower. The
    classical side is unchanged.
    """
    if env.amplified_cost is not None:
        raise ValueError("the envelope already includes the amplified route")
    filter_cost = env.Q if env.filter_cost is None else env.filter_cost
    if amplified_cost < filter_cost:
        return replace(env, Q=amplified_cost, quantum_route=QRQ_AA,
                       filter_cost=filter_cost, amplified_cost=amplified_cost)
    return replace(env, filter_cost=filter_cost, amplified_cost=amplified_cost)


@dataclass(frozen=True)
class WindowResult:
    """Outcome of :func:`evaluate_window`.

    ``Q``, ``C``, ``D`` and the selected trees/routes are ``None`` when the
    window is empty. ``costs`` maps tree kind to ``{route: cost}`` and
    ``routes`` maps tree kind to the full resource rows. ``filter_cost`` and
    ``amplified_cost`` are the ``QRQ_FW`` and ``QRQ_AA`` costs on
    ``quantum_tree`` (the latter only with ``amplified=True``).
    """

    k: int
    Q: float | None
    C: float | None
    D: float | None
    quantum_tree: str | None
    classical_route: str | None
    classical_tree: str | None
    costs: dict[str, dict[str, float]]
    routes: dict[str, list[dict[str, Any]]] = field(repr=False)
    verified: bool = False
    quantum_route: str | None = None
    filter_cost: float | None = None
    amplified_cost: float | None = None

    @property
    def quantum_lower(self) -> bool:
        """True when the window is non-empty and ``Q < C``."""
        return self.Q is not None and self.C is not None and self.Q < self.C


def evaluate_window(
    indexes: Mapping[str, SpatioTemporalIndex],
    box: SpaceTimeBox,
    prices: PriceTable = L1,
    verify: bool = False,
    amplified: bool = False,
) -> WindowResult:
    """Evaluate one window on every index and return its envelope.

    All indexes must cover the same records. With ``verify=True`` the
    answer size is checked against a brute-force count of the records.
    With ``amplified=True`` the quantum side also admits amplitude
    amplification on the selected quantum tree (:func:`include_amplified`);
    this requires ``prices.lam == prices.repeats == 1``.
    """
    if not indexes:
        raise ValueError("at least one index is required")
    sizes = {index.size for index in indexes.values()}
    if len(sizes) != 1:
        raise ValueError("all indexes must cover the same records")
    n = sizes.pop()
    answers: set[int] = set()
    costs: dict[str, dict[str, float]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, index in indexes.items():
        k, route_rows = evaluate_routes(index.plan(box), n)
        answers.add(k)
        rows[name] = route_rows
        costs[name] = route_costs(route_rows, prices)
    if len(answers) != 1:
        raise AssertionError("indexes disagree on the answer size")
    k = answers.pop()
    verified = False
    if verify:
        truth = next(iter(indexes.values())).count(box)
        if truth != k:
            raise AssertionError(f"index answer {k} differs from brute-force count {truth}")
        verified = True
    if not k:
        return WindowResult(k, None, None, None, None, None, None, costs, rows, verified)
    env = select_envelope(costs.items())
    if amplified:
        q_row = next(row for row in rows[env.quantum_tree] if row["route"] == QRQ_FW)
        env = include_amplified(env, amplified_cost_from_route(q_row, k, prices).cost)
    return WindowResult(
        k=k,
        Q=env.Q,
        C=env.C,
        D=env.D,
        quantum_tree=env.quantum_tree,
        classical_route=env.classical_route,
        classical_tree=env.classical_tree,
        costs=costs,
        routes=rows,
        verified=verified,
        quantum_route=env.quantum_route,
        filter_cost=env.filter_cost,
        amplified_cost=env.amplified_cost,
    )
