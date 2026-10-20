"""Exact re-pricing of a fixed window cohort over a grid of oracle prices.

The sweep keeps every resource count, gate price, classical price and the
repetition factor fixed and varies only the coherent record-lookup price
``L`` and the coherent geometry price ``g_q``. For every tree of every
non-empty window it re-prices the per-tree endpoint in closed form,

.. math::

    Q' = P_0 + \\frac{M}{K}\\,(G + 2L + g_q + 1 + b) + 2L - b,
    \\qquad
    C' = C - n_L\\,(L_{ref} - L),

where ``P0`` is the classical prefix of the ``QRQ_FW`` route, ``b`` the rank
register width, ``C`` the cheapest classical route at the reference prices
and ``n_L`` the coherent lookups shared by the four classical routes (two
in the published workload; the classical arg-min is therefore unchanged).
Both tree minima are then reselected at every price point with the
envelope tie rule (first minimal tree in tree order), ``D' = log2(Q'/C')``,
and adjacent pairs are re-classified.

The expressions are evaluated in the same order as the sealed sweep, so a
cohort built from the sealed ledgers reproduces the published price tables
bit for bit. The closed form equals the route cost of
:func:`qrq.prices.route_cost` up to floating-point rounding; at the
reference prices the two agree within ``1e-12`` relative error.

Typical use with the sealed ledgers::

    from qrq.ledger import reproduce
    from qrq import sweep

    cohort = sweep.cohort_from_reproduction(reproduce(root), group_of=groups)
    tables = sweep.sweep(cohort, sweep.price_grid())
    tables.grid[0]["lower_windows"]       # 742 at the L1 point
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .index import TREE_KINDS
from .prices import L1, PriceTable

LOOKUP_PRICES: tuple[int, ...] = (100, 30, 10, 3, 1)
"""Coherent lookup prices of the published grid."""

GEOMETRY_PRICES: tuple[int, ...] = (10000, 3000, 1000, 300, 100, 30, 10, 3, 1)
"""Coherent geometry prices of the published grid."""

REPORT_POINTS: tuple[tuple[int, int], ...] = ((100, 10000), (10, 1000), (1, 100), (1, 1))
"""``(lookup, g_quantum)`` points reported in the text."""

FINE_DURATION_S = 900
"""Duration of the 15 min windows."""

FINE_WIDTH_M = 1000
"""Largest width of the ``<= 1 km`` windows."""

LARGE_ANSWER = 20000
"""Smallest answer size of the large-answer subset."""

QUANTILES: tuple[tuple[str, float], ...] = (("q25", 0.25), ("median", 0.5), ("q75", 0.75))

Row = dict[str, Any]


def price_grid(
    lookups: Sequence[float] = LOOKUP_PRICES,
    geometries: Sequence[float] = GEOMETRY_PRICES,
    base: PriceTable = L1,
) -> list[PriceTable]:
    """Price tables of the grid, lookup price outer and geometry price inner.

    All other prices are taken from ``base``. The defaults give the 45
    published points in published order.
    """
    return [dataclasses.replace(base, lookup=l, g_quantum=g) for l in lookups for g in geometries]


@dataclass(frozen=True, eq=False)
class SweepCohort:
    """Per-tree endpoints of a fixed window cohort and its adjacent pairs.

    Window arrays have shape ``(W,)`` and per-tree arrays ``(W, T)`` with
    trees in envelope tie-break order. ``small`` and ``large`` index the two
    windows of each adjacent pair; ``axis`` is the pair's expansion axis.
    ``C`` holds the cheapest classical cost of each tree at ``reference``.
    ``group`` is a user-defined label that groups sources (a region, a
    campaign, a fleet, ...); readouts count distinct groups as well as
    distinct sources.
    """

    keys: tuple[tuple[str, str, str], ...]
    group: np.ndarray
    k: np.ndarray
    width_m: np.ndarray
    duration_s: np.ndarray
    M: np.ndarray
    G: np.ndarray
    bits: np.ndarray
    P0: np.ndarray
    C: np.ndarray
    classical_lookups: np.ndarray
    small: np.ndarray
    large: np.ndarray
    axis: np.ndarray
    reference: PriceTable = L1
    trees: tuple[str, ...] = TREE_KINDS

    def __post_init__(self) -> None:
        w, t = len(self.keys), len(self.trees)
        for name in ("group", "k", "width_m", "duration_s"):
            if getattr(self, name).shape != (w,):
                raise ValueError(f"{name} must have shape ({w},)")
        for name in ("M", "G", "bits", "P0", "C", "classical_lookups"):
            if getattr(self, name).shape != (w, t):
                raise ValueError(f"{name} must have shape ({w}, {t})")
        if not (len(self.small) == len(self.large) == len(self.axis)):
            raise ValueError("pair arrays must have equal length")
        if not np.all(self.k > 0) or not np.all(self.M >= self.k[:, None]):
            raise ValueError("every window needs 0 < K <= M on every tree")
        if not np.all(np.isfinite(self.classical_lookups)):
            raise ValueError("classical routes must share one lookup count and use no coherent geometry")
        _check_prices(self.reference, self.reference)

    @property
    def source(self) -> np.ndarray:
        """Source identifier of every window."""
        return np.array([key[0] for key in self.keys])

    @property
    def pair_source(self) -> np.ndarray:
        """Source identifier of every pair."""
        return self.source[self.small]


def cohort_from_endpoints(
    windows: Iterable[Any],
    pairs: Iterable[Any],
    group_of: Mapping[str, str] | None = None,
    reference: PriceTable = L1,
    trees: Sequence[str] = TREE_KINDS,
) -> SweepCohort:
    """Build a cohort from envelope endpoints with per-tree inputs.

    ``windows`` are objects like :class:`qrq.ledger.LedgerEndpoint` (with
    ``source_id``, ``date``, ``query_id``, ``k``, ``width_m``,
    ``duration_s`` and ``trees``); ``pairs`` are objects like
    :class:`qrq.ledger.PairRecord` (``source_id``, ``date``, ``axis``,
    ``small_query``, ``large_query``). ``group_of`` maps source identifiers
    to group labels; without it every source forms its own group.
    """
    windows = list(windows)
    keys = tuple((w.source_id, w.date, w.query_id) for w in windows)
    if len(set(keys)) != len(keys):
        raise ValueError("window keys must be unique")
    index = {key: i for i, key in enumerate(keys)}
    per_tree = []
    for w in windows:
        by_tree = {t.tree: t for t in w.trees}
        if set(by_tree) != set(trees):
            raise ValueError(f"{w.query_id}: per-tree inputs must cover {tuple(trees)}")
        per_tree.append([by_tree[t] for t in trees])

    def table(name: str, dtype: Any) -> np.ndarray:
        return np.array([[getattr(t, name) for t in row] for row in per_tree], dtype=dtype).reshape(
            len(windows), len(trees))

    pairs = list(pairs)
    small = [index[(p.source_id, p.date, p.small_query)] for p in pairs]
    large = [index[(p.source_id, p.date, p.large_query)] for p in pairs]
    groups = group_of or {}
    return SweepCohort(
        keys=keys,
        group=np.array([groups.get(key[0], key[0]) for key in keys]),
        k=np.array([w.k for w in windows], dtype=np.int64),
        width_m=np.array([w.width_m for w in windows]),
        duration_s=np.array([w.duration_s for w in windows]),
        M=table("M", np.int64),
        G=table("G", np.int64),
        bits=table("bits", np.int64),
        P0=table("P0", np.float64),
        C=table("C", np.float64),
        classical_lookups=table("classical_lookups", np.float64),
        small=np.array(small, dtype=np.int64),
        large=np.array(large, dtype=np.int64),
        axis=np.array([p.axis for p in pairs]),
        reference=reference,
        trees=tuple(trees),
    )


def cohort_from_reproduction(reproduction: Any, group_of: Mapping[str, str] | None = None) -> SweepCohort:
    """Cohort of a :class:`qrq.ledger.Reproduction`, priced at its audit prices."""
    return cohort_from_endpoints(reproduction.windows, reproduction.pairs, group_of, reproduction.prices)


def _check_prices(prices: PriceTable, reference: PriceTable) -> None:
    if prices.lam != 1 or prices.repeats != 1:
        raise ValueError("the closed-form sweep is defined for lam == repeats == 1 only")
    if prices.g_classical != reference.g_classical:
        raise ValueError("the classical geometry price must equal the reference price")


@dataclass(frozen=True, eq=False)
class PricePoint:
    """Re-priced cohort at one price table.

    ``Q_tree``/``C_tree`` are the per-tree costs, ``quantum_index`` and
    ``classical_index`` the selected tree positions, and ``forward`` /
    ``reverse`` the pair directions. The share arrays describe the selected
    quantum tree: ``(M/K) G / Q`` (structure), ``P0 / Q`` (prefix), ``M/K``
    and ``G/K``.
    """

    prices: PriceTable
    Q_tree: np.ndarray
    C_tree: np.ndarray
    quantum_index: np.ndarray
    classical_index: np.ndarray
    Q: np.ndarray
    C: np.ndarray
    forward: np.ndarray
    reverse: np.ndarray
    structure_share: np.ndarray
    prefix_share: np.ndarray
    trials: np.ndarray
    gates_per_answer: np.ndarray

    @property
    def D(self) -> np.ndarray:
        """``log2(Q / C)`` of every window."""
        return np.log2(self.Q / self.C)

    @property
    def lower(self) -> np.ndarray:
        """Windows with ``Q < C``."""
        return self.Q < self.C


def reprice(cohort: SweepCohort, prices: PriceTable) -> PricePoint:
    """Re-price every tree, reselect both minima and re-classify the pairs."""
    _check_prices(prices, cohort.reference)
    lookup, geometry = prices.lookup, prices.g_quantum
    a = cohort.M / cohort.k[:, None]
    q_tree = cohort.P0 + a * (cohort.G + 2 * lookup + geometry + 1 + cohort.bits) + 2 * lookup - cohort.bits
    c_tree = cohort.C - cohort.classical_lookups * (cohort.reference.lookup - lookup)
    rows = np.arange(len(cohort.keys))
    qi = np.argmin(q_tree, axis=1)
    ci = np.argmin(c_tree, axis=1)
    q = q_tree[rows, qi]
    c = c_tree[rows, ci]
    if not (np.all(q > 0) and np.all(c > 0)):
        raise ValueError("re-priced costs must stay positive")
    d = np.log2(q / c)
    small, large = d[cohort.small], d[cohort.large]
    return PricePoint(
        prices=prices,
        Q_tree=q_tree,
        C_tree=c_tree,
        quantum_index=qi,
        classical_index=ci,
        Q=q,
        C=c,
        forward=(small > 0) & (large < 0),
        reverse=(small < 0) & (large > 0),
        structure_share=(a * cohort.G)[rows, qi] / q,
        prefix_share=cohort.P0[rows, qi] / q,
        trials=a[rows, qi],
        gates_per_answer=(cohort.G / cohort.k[:, None])[rows, qi],
    )


def _quantile(values: np.ndarray, q: float) -> float | None:
    return float(np.quantile(values, q)) if len(values) else None


def window_subsets(cohort: SweepCohort) -> list[tuple[str, np.ndarray]]:
    """Named window subsets: 15 min, width <= 1 km and ``K >= 20000``."""
    return [
        ("15min", cohort.duration_s == FINE_DURATION_S),
        ("width_le_1km", cohort.width_m <= FINE_WIDTH_M),
        ("K_ge_20000", cohort.k >= LARGE_ANSWER),
    ]


def window_summary(cohort: SweepCohort, point: PricePoint, mask: np.ndarray) -> Row:
    """Window readouts of one price point over the windows in ``mask``.

    Counts of eligible and ``Q < C`` windows, median ``Q/C`` overall and per
    subset (:func:`window_subsets`), and quartiles of the structure share,
    prefix share, ``M/K``, ``G/K`` and ``C/K`` on the selected trees.
    """
    ratio = point.Q / point.C
    lower = point.lower
    out: Row = {
        "lookup": point.prices.lookup,
        "quantum_geometry": point.prices.g_quantum,
        "eligible_windows": int(mask.sum()),
        "lower_windows": int((lower & mask).sum()),
        "q_over_c_median": _quantile(ratio[mask], 0.5),
    }
    for label, subset in window_subsets(cohort):
        selected = mask & subset
        out[f"{label}_windows"] = int(selected.sum())
        out[f"{label}_lower"] = int((lower & selected).sum())
        out[f"{label}_ratio_median"] = _quantile(ratio[selected], 0.5)
    distributions = (
        ("structure_share", point.structure_share),
        ("prefix_share", point.prefix_share),
        ("M_over_K", point.trials),
        ("G_over_K", point.gates_per_answer),
        ("C_over_K", point.C / cohort.k),
    )
    for label, values in distributions:
        for name, q in QUANTILES:
            out[f"{label}_{name}"] = _quantile(values[mask], q)
    return out


def pair_summary(point: PricePoint, reference_forward: np.ndarray, mask: np.ndarray) -> Row:
    """Pair directions of one price point over the pairs in ``mask``.

    Retained, added and lost forward pairs are relative to
    ``reference_forward``.
    """
    fwd, rev = point.forward, point.reverse
    return {
        "eligible_pairs": int(mask.sum()),
        "forward_pairs": int((fwd & mask).sum()),
        "reverse_pairs": int((rev & mask).sum()),
        "no_cross_pairs": int((~fwd & ~rev & mask).sum()),
        "forward_retained": int((fwd & reference_forward & mask).sum()),
        "forward_added": int((fwd & ~reference_forward & mask).sum()),
        "forward_lost": int((~fwd & reference_forward & mask).sum()),
    }


def _tie_windows(costs: np.ndarray, selected: np.ndarray) -> int:
    return int((np.sum(costs == selected[:, None], axis=1) > 1).sum())


def grid_row(cohort: SweepCohort, point: PricePoint, reference_forward: np.ndarray) -> Row:
    """Pooled readouts of one price point (one row of the price grid table)."""
    out = window_summary(cohort, point, np.ones(len(cohort.keys), dtype=bool))
    pairs = pair_summary(point, reference_forward, np.ones(len(cohort.small), dtype=bool))
    fwd = point.forward
    out.update(
        forward_pairs=pairs["forward_pairs"],
        reverse_pairs=pairs["reverse_pairs"],
        eligible_pairs=pairs["eligible_pairs"],
        sources_with_forward=len(set(cohort.pair_source[fwd])),
        groups_with_forward=len(set(cohort.group[cohort.small[fwd]])),
        forward_retained=pairs["forward_retained"],
        forward_added=pairs["forward_added"],
        forward_lost=pairs["forward_lost"],
        q_tree_tie_windows=_tie_windows(point.Q_tree, point.Q),
        c_tree_tie_windows=_tie_windows(point.C_tree, point.C),
    )
    return out


@dataclass(frozen=True)
class SweepTables:
    """Readouts of a sweep, one list of rows per published table.

    ``grid``: one pooled row per price point. ``by_source``: one row per
    price point and source. ``pairs``: one row per price point, source
    (``ALL`` first) and axis (``all`` first).
    """

    grid: list[Row]
    by_source: list[Row]
    pairs: list[Row]


def sweep(
    cohort: SweepCohort,
    grid: Iterable[PriceTable],
    reference: PriceTable | None = None,
    on_point: Callable[[PricePoint], None] | None = None,
) -> SweepTables:
    """Re-price ``cohort`` at every point of ``grid`` and summarise.

    Retained, added and lost forward pairs are relative to the directions at
    ``reference`` (default: the cohort's reference prices, ``L1`` for the
    sealed ledgers). ``on_point`` is called with every :class:`PricePoint`.
    """
    base = reprice(cohort, cohort.reference if reference is None else reference)
    sources = sorted(set(cohort.source))
    window_source = cohort.source
    pair_source = cohort.pair_source
    axes = ("all",) + tuple(sorted(set(cohort.axis)))
    source_days = {s: len({key[1] for key in cohort.keys if key[0] == s}) for s in sources}
    tables = SweepTables([], [], [])
    for prices in grid:
        point = reprice(cohort, prices)
        if on_point is not None:
            on_point(point)
        tables.grid.append(grid_row(cohort, point, base.forward))
        for source in sources:
            mask = window_source == source
            row = window_summary(cohort, point, mask)
            row.update(source_id=source, group=str(cohort.group[mask][0]), source_days=source_days[source])
            tables.by_source.append(row)
        for source in ["ALL"] + sources:
            in_source = np.ones(len(pair_source), dtype=bool) if source == "ALL" else pair_source == source
            for axis in axes:
                mask = in_source if axis == "all" else in_source & (cohort.axis == axis)
                row = {"lookup": prices.lookup, "quantum_geometry": prices.g_quantum,
                       "source_id": source, "axis": axis}
                row.update(pair_summary(point, base.forward, mask))
                tables.pairs.append(row)
    return tables
