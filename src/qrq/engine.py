"""Per-window evaluation across several indexes: the cost envelope.

For one window the engine evaluates every route on every index and forms

* ``Q``: the minimum quantum-filtering (``QRQ_FW``) cost over the indexes;
* ``C``: the minimum over the indexes and the four classical routes
  ``C_BLOCK``, ``C_ENUM``, ``C_TABLE_BLOCK`` and ``C_TABLE_ENUM``;
* ``D = log2(Q / C)``; ``D < 0`` means the quantum route is cheaper.

Tie-breaking reproduces the sealed analysis: the quantum tree is the first
minimal tree in iteration order, and the classical choice minimises the
tuple ``(cost, tree, route)``. Empty windows (``k == 0``) have no envelope.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .geometry import SpaceTimeBox
from .index import SpatioTemporalIndex
from .prices import L1, PriceTable, route_costs
from .routes import QRQ_FW, evaluate_routes


@dataclass(frozen=True)
class Envelope:
    """Envelope of one non-empty window."""

    Q: float
    C: float
    quantum_tree: str
    classical_tree: str
    classical_route: str

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
    )


@dataclass(frozen=True)
class WindowResult:
    """Outcome of :func:`evaluate_window`.

    ``Q``, ``C``, ``D`` and the selected tree/route are ``None`` when the
    window is empty. ``costs`` maps tree kind to ``{route: cost}`` and
    ``routes`` maps tree kind to the full resource rows.
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

    @property
    def quantum_lower(self) -> bool:
        """True when the window is non-empty and ``Q < C``."""
        return self.Q is not None and self.C is not None and self.Q < self.C


def evaluate_window(
    indexes: Mapping[str, SpatioTemporalIndex],
    box: SpaceTimeBox,
    prices: PriceTable = L1,
    verify: bool = False,
) -> WindowResult:
    """Evaluate one window on every index and return its envelope.

    All indexes must cover the same records. With ``verify=True`` the
    answer size is checked against a brute-force count of the records.
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
    )
