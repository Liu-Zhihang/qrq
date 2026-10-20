"""Per-tree route ledgers: row schema, writer, reader and re-pricing audit.

A ledger is a gzip JSON-lines file holding one row per registered window for
one tree kind; a :class:`LedgerDay` groups the ledgers of all tree kinds that
cover the same records (for example one source on one day). Row schema:

``query_id``
    window identifier, unique within the ledger.
``result_status``
    ``eligible_nonempty`` or ``eligible_empty`` for costed windows; any other
    value (for example ``unsupported_index_span``) marks an uncosted window.
``anchor_index``, ``width_m``, ``duration_s``, ``temporal_center_s``
    window metadata, required for envelopes and adjacent-scale pairs.
``n``, ``k``, ``routes``, ``costs``
    for costed windows: indexed records, exact answer size, the five route
    resource rows of :func:`qrq.routes.evaluate_routes` and their costs
    ``{route: cost}`` at the ledger prices.
``source_id``, ``date``, ``arm``, ``variant``, ``tree``
    optional labels.

:func:`ledger_row` and :func:`write_ledger` produce ledgers in this schema.
:func:`audit_day` re-prices every stored route from its resource vectors,
checks the stored costs, and recomputes the envelope of every non-empty
window with :func:`qrq.engine.select_envelope` (optionally admitting
amplitude amplification, :func:`qrq.engine.include_amplified`). Every window
keeps its per-tree inputs (:class:`TreeEndpoint`) for exact re-pricing with
:mod:`qrq.sweep`. :func:`audit_cohort` aggregates many ledger days and forms
the adjacent-scale pairs of each day on a caller-supplied window grid.
"""
from __future__ import annotations

import collections
import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Iterable, Iterator, Mapping, Sequence

from .amplitude import amplified_cost_from_route
from .engine import include_amplified, select_envelope
from .index import SpatioTemporalIndex
from .prices import L1, PriceTable, decompose_quantum_cost, route_cost, route_costs
from .routes import QRQ_FW, evaluate_routes
from .workload import UNSUPPORTED_INDEX_SPAN, Window, WindowGrid, adjacent_pairs, classify_pair

ELIGIBLE_NONEMPTY = "eligible_nonempty"
ELIGIBLE_EMPTY = "eligible_empty"
WINDOW_FIELDS = ("anchor_index", "width_m", "duration_s", "temporal_center_s")


def ledger_row(
    window: Window, index: SpatioTemporalIndex, prices: PriceTable = L1, **labels: Any
) -> dict[str, Any]:
    """Ledger row of one registered window on one index.

    Unsupported windows are recorded without routes; every other window is
    costed with :func:`qrq.routes.evaluate_routes` and priced at ``prices``.
    ``labels`` (for example ``source_id``, ``date``, ``arm``) are copied
    into the row.
    """
    row: dict[str, Any] = dict(
        query_id=window.query_id,
        anchor_index=list(window.anchor_index),
        width_m=window.width_m,
        duration_s=window.duration_s,
        temporal_center_s=window.temporal_center_s,
        n=index.size,
        tree=index.kind,
        **labels,
    )
    if not window.supported:
        row["result_status"] = UNSUPPORTED_INDEX_SPAN
        return row
    k, rows = evaluate_routes(index.plan(window.box), index.size)
    row.update(result_status=ELIGIBLE_NONEMPTY if k else ELIGIBLE_EMPTY, k=k, routes=rows,
               costs=route_costs(rows, prices))
    return row


def write_ledger(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write ledger rows as gzip JSON lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return path


def read_ledger(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate over the JSON rows of one gzip ledger."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


@dataclass(frozen=True)
class LedgerDay:
    """The per-tree ledgers of one group of windows (for example a source-day).

    ``paths`` maps tree kind to ledger file; its order is the envelope
    tie-break order.
    """

    source_id: str
    date: str
    paths: Mapping[str, Path]


@dataclass(frozen=True)
class TreeEndpoint:
    """Per-tree inputs of one non-empty window for exact re-pricing.

    ``P0`` is the classical prefix of the ``QRQ_FW`` route (frontier
    discovery plus preparation metadata), ``Q`` its cost and ``C`` the
    cheapest classical route on this tree, both at the audit prices.
    ``classical_lookups`` is the coherent lookup count shared by the four
    classical routes, which then keep their order under any lookup price;
    it is ``nan`` when the routes differ in lookups or use coherent
    geometry.
    """

    tree: str
    M: int
    G: int
    bits: int
    P0: float
    Q: float
    C: float
    classical_lookups: float


def _tree_endpoint(
    tree: str, row: Mapping[str, Any], costs: Mapping[str, float], prices: PriceTable
) -> TreeEndpoint:
    q_route = next(r for r in row["routes"] if r["route"] == QRQ_FW)
    classical = [r for r in row["routes"] if r["route"] != QRQ_FW]
    c = q_route["classical"]
    lookups = {r["quantum"]["lookup_calls"] for r in classical}
    coherent_geometry = {r["quantum"]["geometry_calls"] for r in classical}
    shared = float(lookups.pop()) if len(lookups) == 1 and coherent_geometry == {0} else math.nan
    return TreeEndpoint(
        tree=tree,
        M=int(q_route["domain"]),
        G=int(q_route["preparation"]["gate_total"]),
        bits=int(q_route["shared_initial_rank_bits"]),
        P0=float(c["nodes"] + c["slots"] + prices.g_classical * c["geometry"] + c["metadata"]),
        Q=costs[QRQ_FW],
        C=min(value for name, value in costs.items() if name != QRQ_FW),
        classical_lookups=shared,
    )


@dataclass(frozen=True)
class LedgerEndpoint:
    """Envelope of one non-empty window recomputed from ledger rows.

    ``M``, ``trials``, ``G``, ``prefix`` and ``trial_cost`` describe the
    ``QRQ_FW`` route on ``quantum_tree``. ``quantum_route``,
    ``filter_cost`` and ``amplified_cost`` follow :class:`qrq.engine.Envelope`;
    ``trees`` holds the per-tree inputs in ledger tree order. ``arm`` and
    ``variant`` are the optional row labels (empty when absent).
    """

    source_id: str
    date: str
    query_id: str
    arm: str
    variant: str
    anchor_index: tuple[int, int]
    width_m: int
    duration_s: int
    temporal_center_s: int
    n: int
    k: int
    Q: float
    C: float
    D: float
    quantum_tree: str
    classical_tree: str
    classical_route: str
    M: int
    trials: float
    G: int
    prefix: float
    trial_cost: float
    quantum_route: str = QRQ_FW
    filter_cost: float | None = None
    amplified_cost: float | None = None
    trees: tuple[TreeEndpoint, ...] = field(default=(), repr=False)

    @property
    def quantum_lower(self) -> bool:
        return self.Q < self.C


@dataclass
class DayAudit:
    """Result of re-pricing and re-enveloping one ledger day."""

    day: LedgerDay
    rows: int = 0
    priced_routes: int = 0
    exact_cost_matches: int = 0
    max_relative_error: float = 0.0
    statuses: collections.Counter = field(default_factory=collections.Counter)
    endpoints: list[LedgerEndpoint] = field(default_factory=list)


def _relative_error(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def audit_day(
    day: LedgerDay,
    prices: PriceTable = L1,
    rel_tol: float = 1e-12,
    amplified: bool = False,
    arms: Collection[str] | None = None,
) -> DayAudit:
    """Re-price every route of one ledger day and recompute its envelopes.

    Envelopes are built for every ``eligible_nonempty`` window that carries
    the window metadata and, when ``arms`` is given, whose ``arm`` label is
    one of them. With ``amplified=True`` each envelope also admits amplitude
    amplification on its selected quantum tree. Raises
    :class:`AssertionError` when a stored cost differs from the recomputed
    one beyond ``rel_tol``, when trees disagree on ``k`` or on the window
    status, or when the quantum cost does not decompose as ``P + (M/K) H``.
    """
    audit = DayAudit(day)
    by_query: dict[str, list[tuple[str, dict[str, Any], dict[str, float]]]] = {}
    for tree, path in day.paths.items():
        for row in read_ledger(path):
            audit.rows += 1
            costs: dict[str, float] = {}
            if "routes" in row:
                for route in row["routes"]:
                    value = route_cost(route, prices)
                    stored = row["costs"][route["route"]]
                    err = _relative_error(value, stored)
                    if not math.isclose(value, stored, rel_tol=rel_tol):
                        raise AssertionError(
                            f"{day.source_id} {day.date} {tree} {row['query_id']} "
                            f"{route['route']}: stored {stored} != recomputed {value}"
                        )
                    audit.priced_routes += 1
                    audit.exact_cost_matches += int(value == stored)
                    audit.max_relative_error = max(audit.max_relative_error, err)
                    costs[route["route"]] = value
            by_query.setdefault(row["query_id"], []).append((tree, row, costs))
    for query_id, entries in by_query.items():
        if len(entries) != len(day.paths):
            raise AssertionError(f"{query_id}: not present in every tree ledger")
        statuses = {row["result_status"] for _, row, _ in entries}
        if len(statuses) != 1:
            raise AssertionError(f"{query_id}: trees disagree on the window status")
        first = entries[0][1]
        audit.statuses[(first.get("arm"), first["result_status"])] += 1
        if "routes" in first and len({row["k"] for _, row, _ in entries}) != 1:
            raise AssertionError(f"{query_id}: trees disagree on k")
        if first["result_status"] != ELIGIBLE_NONEMPTY or not all(f in first for f in WINDOW_FIELDS):
            continue
        if arms is not None and first.get("arm") not in arms:
            continue
        env = select_envelope((tree, costs) for tree, _, costs in entries)
        q_row = next(row for tree, row, _ in entries if tree == env.quantum_tree)
        q_route = next(r for r in q_row["routes"] if r["route"] == QRQ_FW)
        parts = decompose_quantum_cost(q_route, prices)
        if not math.isclose(env.Q, parts.total, rel_tol=rel_tol):
            raise AssertionError(f"{query_id}: Q does not decompose as P + (M/K) H")
        if amplified:
            env = include_amplified(env, amplified_cost_from_route(q_route, first["k"], prices).cost)
        audit.endpoints.append(
            LedgerEndpoint(
                source_id=day.source_id,
                date=day.date,
                query_id=query_id,
                arm=first.get("arm", ""),
                variant=first.get("variant", ""),
                anchor_index=tuple(first["anchor_index"]),  # type: ignore[arg-type]
                width_m=first["width_m"],
                duration_s=first["duration_s"],
                temporal_center_s=first["temporal_center_s"],
                n=first["n"],
                k=first["k"],
                Q=env.Q,
                C=env.C,
                D=env.D,
                quantum_tree=env.quantum_tree,
                classical_tree=env.classical_tree,
                classical_route=env.classical_route,
                M=q_route["domain"],
                trials=q_route["expected_trials"],
                G=q_route["preparation"]["gate_total"],
                prefix=parts.prefix,
                trial_cost=parts.trial_cost,
                quantum_route=env.quantum_route,
                filter_cost=env.filter_cost,
                amplified_cost=env.amplified_cost,
                trees=tuple(_tree_endpoint(tree, row, costs, prices) for tree, row, costs in entries),
            )
        )
    return audit


@dataclass(frozen=True)
class PairRecord:
    """One adjacent-scale pair of envelopes."""

    source_id: str
    date: str
    axis: str
    small_query: str
    large_query: str
    small_D: float
    large_D: float
    direction: str


def pair_endpoints(endpoints: Sequence[LedgerEndpoint], grid: WindowGrid) -> list[PairRecord]:
    """Adjacent-scale pairs among the envelopes of one ledger day.

    Raises :class:`AssertionError` when a larger window has fewer answers
    than the nested smaller one.
    """
    pairs = []
    for pair in adjacent_pairs(endpoints, grid):
        small, large = pair.small, pair.large
        if large.k < small.k:
            raise AssertionError("nested windows must not lose answers")
        pairs.append(PairRecord(
            source_id=small.source_id,
            date=small.date,
            axis=pair.axis,
            small_query=small.query_id,
            large_query=large.query_id,
            small_D=small.D,
            large_D=large.D,
            direction=classify_pair(small.D, large.D),
        ))
    return pairs


@dataclass
class LedgerCohort:
    """Envelopes and pairs of many ledger days (see :func:`audit_cohort`).

    ``prices`` are the audit prices of every stored and recomputed cost;
    ``amplified`` tells whether the envelopes admit the ``QRQ_AA`` route.
    """

    days: int = 0
    ledgers: int = 0
    rows: int = 0
    priced_routes: int = 0
    exact_cost_matches: int = 0
    max_relative_error: float = 0.0
    windows: list[LedgerEndpoint] = field(default_factory=list)
    pairs: list[PairRecord] = field(default_factory=list)
    prices: PriceTable = L1
    amplified: bool = False

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def quantum_lower_count(self) -> int:
        return sum(w.quantum_lower for w in self.windows)

    def direction_counts(self) -> collections.Counter:
        return collections.Counter(p.direction for p in self.pairs)

    def by_source(self) -> dict[str, dict[str, int]]:
        """Per-source window, ``Q < C`` and crossing counts."""
        out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for w in self.windows:
            out[w.source_id]["windows"] += 1
            out[w.source_id]["quantum_lower"] += int(w.quantum_lower)
        for p in self.pairs:
            out[p.source_id]["pairs"] += 1
            out[p.source_id][p.direction] += 1
        keys = ("windows", "quantum_lower", "pairs", "forward", "reverse")
        return {s: {k: int(c[k]) for k in keys} for s, c in sorted(out.items())}

    def summary(self) -> dict[str, Any]:
        directions = self.direction_counts()
        return dict(
            days=self.days,
            ledgers=self.ledgers,
            rows=self.rows,
            priced_routes=self.priced_routes,
            exact_cost_matches=self.exact_cost_matches,
            max_relative_error=self.max_relative_error,
            windows=self.window_count,
            quantum_lower=self.quantum_lower_count,
            pairs=len(self.pairs),
            forward=int(directions["forward"]),
            reverse=int(directions["reverse"]),
            by_source=self.by_source(),
        )


def audit_cohort(
    days: Iterable[LedgerDay],
    grid: WindowGrid,
    prices: PriceTable = L1,
    rel_tol: float = 1e-12,
    amplified: bool = False,
    arms: Collection[str] | None = None,
    arm: str | None = None,
    variant: str | None = None,
) -> LedgerCohort:
    """Audit every ledger day and collect its envelopes and adjacent pairs.

    ``arms`` restricts envelope building (see :func:`audit_day`); ``arm`` and
    ``variant``, when given, select the envelopes that enter the cohort.
    Pairs are formed within each day on ``grid``.
    """
    result = LedgerCohort(prices=prices, amplified=amplified)
    for day in days:
        audit = audit_day(day, prices, rel_tol, amplified, arms)
        result.days += 1
        result.ledgers += len(day.paths)
        result.rows += audit.rows
        result.priced_routes += audit.priced_routes
        result.exact_cost_matches += audit.exact_cost_matches
        result.max_relative_error = max(result.max_relative_error, audit.max_relative_error)
        selected = [e for e in audit.endpoints
                    if (arm is None or e.arm == arm) and (variant is None or e.variant == variant)]
        result.windows.extend(selected)
        result.pairs.extend(pair_endpoints(selected, grid))
    return result
