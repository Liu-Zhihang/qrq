"""Sealed per-tree resource ledgers: re-pricing and envelope reproduction.

A sealed evidence directory has the layout::

    <root>/<host>/outputs/<source_id>/<date>/<tree>.jsonl.gz

with one gzip JSON-lines ledger per source-day and tree kind. Each line holds
the window metadata, its ``result_status`` and, for costed windows, the
answer size ``k``, the five route resource rows and their stored costs.

:func:`audit_day` re-prices every stored route from its resource vectors,
checks the stored costs, and recomputes the per-window envelope with
:func:`qrq.engine.select_envelope`. :func:`reproduce` aggregates all days
into the published window and adjacent-pair counts. Every window keeps its
per-tree inputs
(:class:`TreeEndpoint`) for exact re-pricing with :mod:`qrq.sweep`. Run it
from the command line with ``python -m qrq.ledger --ledgers <root>``.

The ledgers themselves are restricted evidence and are not distributed with
this package; see ``docs/reproducing.md``.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .engine import select_envelope
from .index import TREE_KINDS
from .prices import L1, PriceTable, decompose_quantum_cost, route_cost
from .routes import QRQ_FW
from .workload import adjacent_pairs, classify_pair

ELIGIBLE_NONEMPTY = "eligible_nonempty"
ENDPOINT_ARMS = ("baseline", "perturb")


@dataclass(frozen=True)
class LedgerDay:
    """The three per-tree ledgers of one source-day."""

    host: str
    source_id: str
    date: str
    paths: dict[str, Path]


def find_ledger_days(root: str | Path, trees: Sequence[str] = TREE_KINDS) -> list[LedgerDay]:
    """Locate every complete source-day under an evidence root."""
    root = Path(root)
    found: dict[tuple[str, str, str], dict[str, Path]] = collections.defaultdict(dict)
    for path in sorted(root.glob("*/outputs/*/*/*.jsonl.gz")):
        tree = path.name[: -len(".jsonl.gz")]
        if tree not in trees:
            continue
        date_dir = path.parent
        host = date_dir.parent.parent.parent.name
        found[(host, date_dir.parent.name, date_dir.name)][tree] = path
    days = []
    for (host, source, date), paths in sorted(found.items()):
        missing = [t for t in trees if t not in paths]
        if missing:
            raise ValueError(f"{source} {date}: missing ledgers for {missing}")
        days.append(LedgerDay(host, source, date, {t: paths[t] for t in trees}))
    return days


def read_ledger(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate over the JSON rows of one gzip ledger."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


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
    ``QRQ_FW`` route on ``quantum_tree``; ``trees`` holds the per-tree
    inputs in ledger tree order.
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
    trees: tuple[TreeEndpoint, ...] = field(default=(), repr=False)

    @property
    def quantum_lower(self) -> bool:
        return self.Q < self.C


@dataclass
class DayAudit:
    """Result of re-pricing and re-enveloping one source-day."""

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


def audit_day(day: LedgerDay, prices: PriceTable = L1, rel_tol: float = 1e-12) -> DayAudit:
    """Re-price every route of one source-day and recompute its envelopes.

    Raises :class:`AssertionError` when a stored cost differs from the
    recomputed one beyond ``rel_tol``, when trees disagree on ``k`` or on the
    window status, or when the quantum cost does not decompose as
    ``P + (M/K) H``.
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
        if first.get("arm") not in ENDPOINT_ARMS or first["result_status"] != ELIGIBLE_NONEMPTY:
            continue
        env = select_envelope((tree, costs) for tree, _, costs in entries)
        q_row = next(row for tree, row, _ in entries if tree == env.quantum_tree)
        q_route = next(r for r in q_row["routes"] if r["route"] == QRQ_FW)
        parts = decompose_quantum_cost(q_route, prices)
        if not math.isclose(env.Q, parts.total, rel_tol=rel_tol):
            raise AssertionError(f"{query_id}: Q does not decompose as P + (M/K) H")
        audit.endpoints.append(
            LedgerEndpoint(
                source_id=day.source_id,
                date=day.date,
                query_id=query_id,
                arm=first["arm"],
                variant=first["variant"],
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
                trees=tuple(_tree_endpoint(tree, row, costs, prices) for tree, row, costs in entries),
            )
        )
    return audit


@dataclass(frozen=True)
class PairRecord:
    """One adjacent-scale pair of baseline envelopes."""

    source_id: str
    date: str
    axis: str
    small_query: str
    large_query: str
    small_D: float
    large_D: float
    direction: str


@dataclass
class Reproduction:
    """Aggregate of :func:`reproduce` over all source-days.

    ``prices`` are the audit prices of every stored and recomputed cost.
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


def reproduce(
    root: str | Path,
    prices: PriceTable = L1,
    arm: str = "baseline",
    variant: str = "base",
    rel_tol: float = 1e-12,
) -> Reproduction:
    """Re-price all sealed ledgers and rebuild the envelope counts.

    Windows are the non-empty, fully supported envelopes of the selected
    ``arm`` and ``variant``; pairs are adjacent-scale pairs among them
    within each source-day.
    """
    result = Reproduction(prices=prices)
    for day in find_ledger_days(root):
        audit = audit_day(day, prices, rel_tol)
        result.days += 1
        result.ledgers += len(day.paths)
        result.rows += audit.rows
        result.priced_routes += audit.priced_routes
        result.exact_cost_matches += audit.exact_cost_matches
        result.max_relative_error = max(result.max_relative_error, audit.max_relative_error)
        selected = [e for e in audit.endpoints if e.arm == arm and e.variant == variant]
        result.windows.extend(selected)
        for pair in adjacent_pairs(selected):
            small, large = pair.small, pair.large
            if large.k < small.k:
                raise AssertionError("nested windows must not lose answers")
            result.pairs.append(
                PairRecord(
                    source_id=day.source_id,
                    date=day.date,
                    axis=pair.axis,
                    small_query=small.query_id,
                    large_query=large.query_id,
                    small_D=small.D,
                    large_D=large.D,
                    direction=classify_pair(small.D, large.D),
                )
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qrq.ledger",
        description="Re-price sealed ledgers and reproduce the envelope counts.",
    )
    parser.add_argument("--ledgers", required=True, help="evidence root directory")
    parser.add_argument("--windows-csv", help="optional output CSV of recomputed windows")
    args = parser.parse_args(argv)
    result = reproduce(args.ledgers)
    if args.windows_csv:
        import csv

        with open(args.windows_csv, "w", newline="", encoding="utf-8") as handle:
            columns = [f.name for f in fields(LedgerEndpoint) if f.name != "trees"]
            rows = [{name: getattr(w, name) for name in columns} for w in result.windows]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["query_id"])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
