"""Access routes and their resource rows.

Every window is answered by five matched-output routes over the same static
rank layout:

``QRQ_FW``
    Quantum first-witness filtering: a classical prefix discovers the
    first-witness frontier, then the frontier's candidate rank domain is
    prepared coherently and filtered by repeated trials (``M/K`` expected).
``C_BLOCK``
    Exact classical traversal that stops at fully contained subtrees, then
    prepares the exact answer ranges.
``C_ENUM``
    Exact classical traversal that tests every reachable record, then
    prepares the exact answer ranges.
``C_TABLE_BLOCK`` / ``C_TABLE_ENUM``
    The two classical solvers followed by writing a contiguous query table
    of the ``k`` hits (four paid construction units per hit).

:func:`evaluate_routes` reproduces the reference ``routes_for(plan, n)``
exactly (same compiler calls, same row order, same pruned preparation
fields).
"""
from __future__ import annotations

import operator
from typing import Any, Literal, Mapping, Union

from ._vendor.qrq_matched import compile_intervals, route_resources
from ._vendor.table_solver_contract import table_from_classical_solver
from .index import AccessPlan

AccessRoute = Literal["QRQ_FW", "C_BLOCK", "C_ENUM", "C_TABLE_BLOCK", "C_TABLE_ENUM"]

QRQ_FW: AccessRoute = "QRQ_FW"
C_BLOCK: AccessRoute = "C_BLOCK"
C_ENUM: AccessRoute = "C_ENUM"
C_TABLE_BLOCK: AccessRoute = "C_TABLE_BLOCK"
C_TABLE_ENUM: AccessRoute = "C_TABLE_ENUM"

QUANTUM_ROUTE: AccessRoute = QRQ_FW
CLASSICAL_ROUTES: tuple[AccessRoute, ...] = (C_BLOCK, C_ENUM, C_TABLE_BLOCK, C_TABLE_ENUM)
ROUTES: tuple[AccessRoute, ...] = (QRQ_FW,) + CLASSICAL_ROUTES
"""All routes in resource-row order."""

TABLE_WRITE_UNITS = 4
"""Paid construction operations per hit for the query-table routes."""

RouteRow = dict[str, Any]


def rank_bits(n: int) -> int:
    """Address bits of the rank register for ``n`` records."""
    return max(1, (n - 1).bit_length())


def evaluate_routes(
    plan: Union[AccessPlan, Mapping[str, Any]], n: int
) -> tuple[int, list[RouteRow]]:
    """Return ``(k, rows)`` for one access plan over ``n`` indexed records.

    ``k`` is the exact answer size (domain of the classical block solver).
    ``rows`` holds one resource row per route in :data:`ROUTES` order, each
    with ``classical`` and ``quantum`` resource vectors suitable for
    :func:`qrq.prices.route_cost`. Whole interval arrays and gate programs
    are dropped from the ``preparation`` summaries, as in the sealed ledgers.
    """
    n = operator.index(n)
    raw = plan.to_matched() if isinstance(plan, AccessPlan) else plan
    bits = rank_bits(n)
    prepared = {
        name: compile_intervals(ranges, bits, False)
        for name, ranges in [
            (QRQ_FW, raw["candidate_ranges"]),
            (C_BLOCK, raw["block"]["ranges"]),
            (C_ENUM, raw["enumeration"]["ranges"]),
        ]
    }
    if prepared[C_BLOCK]["ranges"] != prepared[C_ENUM]["ranges"]:
        raise AssertionError("block and enumeration solvers disagree on the answer")
    k = prepared[C_BLOCK]["domain"]
    classical = {
        QRQ_FW: dict(
            node_reads=raw["prefix_nodes"],
            slots=raw["prefix_slots"],
            geometry=raw["prefix_geometry"],
        ),
        C_BLOCK: raw["block"],
        C_ENUM: raw["enumeration"],
    }
    rows = [route_resources(classical[name], prepared[name], k, name, bits) for name in prepared]
    rows.extend([table_from_classical_solver(n, rows[i], TABLE_WRITE_UNITS) for i in (1, 2)])
    for row in rows:
        row["preparation"] = {
            key: value
            for key, value in row["preparation"].items()
            if key not in ("ranges", "program")
        }
    return k, rows
