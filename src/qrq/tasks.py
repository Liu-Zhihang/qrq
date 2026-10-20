"""Reconstruction-task comparison of crossing and noncrossing expansions.

A task pair expands one window of a reconstruction task to the next larger
extent along one axis (``space``: width, ``time``: duration) with the other
extent fixed. Both endpoints carry the access-cost envelope
``D = log2(Q / C)`` and the absolute reconstruction error. The comparison
asks whether expansions that cross the cost boundary worsen the
reconstruction more often than comparable expansions that do not.

* **Support.** Both endpoints are non-empty and both ``D`` and both errors
  are finite.
* **Crossing.** ``forward``: ``D_small > 0`` and ``D_large < 0``;
  ``reverse``: the opposite. Every other supported pair is noncrossing and
  is labelled ``classical`` (both ``D > 0``), ``quantum`` (both ``D < 0``)
  or ``boundary`` (an exact zero). Reverse pairs are never controls.
* **Error direction.** The sign of the stored error change (large minus
  small): ``up`` (worse), ``down`` or ``tie``. Ties are excluded from every
  worsening-rate denominator.
* **Standardisation.** A stratum is task source x expansion axis x smaller
  extent x larger extent x fixed extent. The noncrossing worsening rate of
  each stratum is weighted by its non-tied forward count. A stratum enters
  only with at least one non-tied forward and one non-tied noncrossing
  pair; forward pairs of strata without controls are reported as excluded.
* **Families** pool their sources with forward-count weights.
* **Uncertainty.** Source-stratified date-block percentile bootstrap: each
  draw resamples all registered dates of every source with replacement and
  keeps all pairs of a drawn date with its multiplicity; comparable strata
  and weights are recomputed per draw, and a draw without comparable strata
  stays undefined (it is not redrawn). One NumPy PCG64 stream is consumed
  family by family in the given order and source by source in sorted order.

The estimator is generic: it consumes :class:`TaskPair` records and a date
registry and knows nothing about particular sources. With the published
inputs it reproduces the sealed tables exactly (``docs/reproducing.md``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FORWARD = "forward"
REVERSE = "reverse"
OTHER = "other"
UNSUPPORTED = "unsupported"
CLASSICAL = "classical"
QUANTUM = "quantum"
BOUNDARY = "boundary"
UP, DOWN, TIE = "up", "down", "tie"

DRAWS = 5000
SEED = 20260924
MARGIN = 0.05
"""Smallest absolute rate difference that can be labelled informative."""

NO_COMPARABLE_FORWARD = "NO_COMPARABLE_FORWARD"
NO_INFORMATION = "NO_INFORMATION_OPERATIONAL"
CROSSING_WORSE = "CROSSING_WORSE"
CROSSING_BETTER = "CROSSING_BETTER"

COUNT_FIELDS: tuple[str, ...] = ("forward_up", "forward_down", "control_up", "control_down", "forward_tie")
"""Last axis of the date x stratum count arrays."""

Stratum = tuple[str, int, int, int]
Row = dict[str, Any]


def crossing_direction(d_small: float, d_large: float) -> str:
    """``forward``, ``reverse`` or ``other`` from the endpoint ``D`` signs."""
    if d_small > 0 and d_large < 0:
        return FORWARD
    if d_small < 0 and d_large > 0:
        return REVERSE
    return OTHER


def crossing_category(d_small: float, d_large: float) -> str:
    """Crossing direction, with noncrossing pairs split by side."""
    direction = crossing_direction(d_small, d_large)
    if direction != OTHER:
        return direction
    if d_small > 0 and d_large > 0:
        return CLASSICAL
    if d_small < 0 and d_large < 0:
        return QUANTUM
    return BOUNDARY


def error_direction(delta: float) -> str:
    """``up`` (worse), ``down`` (better) or ``tie`` from an error change."""
    return UP if delta > 0 else DOWN if delta < 0 else TIE


def worsening_rate(up: int, down: int) -> float:
    """``up / (up + down)``; ties never enter. ``nan`` without non-tied pairs."""
    return up / (up + down) if up + down else math.nan


@dataclass(frozen=True)
class TaskPair:
    """One registered expansion pair of a reconstruction task.

    ``small`` and ``large`` are the extents along ``axis``; ``fixed`` is the
    extent of the other axis. ``delta_error`` is the large minus the small
    absolute error (``nan`` for unsupported pairs).
    """

    family: str
    source: str
    date: str
    axis: str
    small: int
    large: int
    fixed: int
    supported: bool
    small_D: float
    large_D: float
    delta_error: float

    @property
    def stratum(self) -> Stratum:
        """``(axis, small, large, fixed)``; the source is implicit."""
        return (self.axis, self.small, self.large, self.fixed)

    @property
    def direction(self) -> str:
        return crossing_direction(self.small_D, self.large_D) if self.supported else UNSUPPORTED

    @property
    def category(self) -> str:
        return crossing_category(self.small_D, self.large_D) if self.supported else UNSUPPORTED

    @property
    def error(self) -> str:
        return error_direction(self.delta_error) if self.supported else UNSUPPORTED


def _finite(*values: Any) -> bool:
    return all(isinstance(v, (int, float, np.number)) and math.isfinite(v) for v in values)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def pair_from_endpoints(
    family: str,
    source: str,
    date: str,
    axis: str,
    small: Mapping[str, Any],
    large: Mapping[str, Any],
    delta_error: float,
) -> TaskPair:
    """Normalise one pair from its two endpoint records.

    Endpoint records need ``width_m``, ``duration_s``, ``k``, ``D`` and
    ``absolute_error``. The larger endpoint must extend ``axis`` and keep the
    other extent fixed; ``delta_error`` must be finite for supported pairs.
    """
    if axis not in ("space", "time"):
        raise ValueError(f"unknown axis {axis!r}")
    along, other = ("width_m", "duration_s") if axis == "space" else ("duration_s", "width_m")
    lo, hi, fixed = int(small[along]), int(large[along]), int(small[other])
    if hi <= lo or int(large[other]) != fixed:
        raise ValueError(f"{source} {date}: not an expansion along {axis}")
    supported = (small["k"] > 0 and large["k"] > 0
                 and _finite(small["D"], large["D"], small["absolute_error"], large["absolute_error"]))
    delta = float(delta_error) if supported else math.nan
    if supported and not math.isfinite(delta):
        raise ValueError(f"{source} {date}: supported pair without a finite error change")
    return TaskPair(family, source, str(date), axis, lo, hi, fixed, bool(supported),
                    _as_float(small["D"]), _as_float(large["D"]), delta)


def date_stratum_counts(
    pairs: Iterable[TaskPair], dates: Sequence[str], strata: Sequence[Stratum]
) -> np.ndarray:
    """Counts of shape ``(dates, strata, 5)`` with fields :data:`COUNT_FIELDS`.

    Forward pairs count by error direction (ties separately); noncrossing
    pairs count only when not tied. Reverse and unsupported pairs are
    ignored.
    """
    didx = {d: i for i, d in enumerate(dates)}
    sidx = {s: i for i, s in enumerate(strata)}
    counts = np.zeros((len(dates), len(strata), len(COUNT_FIELDS)), dtype=np.int64)
    for p in pairs:
        if not p.supported:
            continue
        j, k = didx[p.date], sidx[p.stratum]
        direction, error = p.direction, p.error
        if direction == FORWARD:
            counts[j, k, {UP: 0, DOWN: 1, TIE: 4}[error]] += 1
        elif direction == OTHER and error != TIE:
            counts[j, k, 2 if error == UP else 3] += 1
    return counts


def standardised_estimates(counts: np.ndarray) -> tuple[Any, Any, Any, Any]:
    """Standardised sums over the stratum axis of ``(..., strata, 5)`` counts.

    Returns ``(forward_up, expected_control_up, used_forward, excluded)``:
    worsened forward pairs, the forward-weighted noncrossing worsening
    expectation, the non-tied forward count of comparable strata and the
    non-tied forward count of strata without controls. Rates follow by
    dividing the first two by the third.
    """
    f = counts[..., 0] + counts[..., 1]
    c = counts[..., 2] + counts[..., 3]
    usable = (f > 0) & (c > 0)
    denom = np.where(usable, f, 0).sum(axis=-1)
    up = np.where(usable, counts[..., 0], 0).sum(axis=-1)
    control = np.divide(counts[..., 2], c, out=np.zeros_like(c, dtype=float), where=c > 0)
    expected = np.where(usable, control * f, 0).sum(axis=-1)
    omitted = np.where((f > 0) & (c == 0), f, 0).sum(axis=-1)
    return up, expected, denom, omitted


def percentile_interval(values: np.ndarray) -> tuple[float | None, float | None]:
    """2.5 % and 97.5 % linear quantiles of the finite values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (None, None)
    lo, hi = np.quantile(x, [0.025, 0.975])
    return (float(lo), float(hi))


def operational_label(
    delta: float, interval: tuple[float | None, float | None], margin: float = MARGIN
) -> str:
    """Preregistered three-way label of a rate difference and its interval."""
    lo, hi = interval
    if not np.isfinite(delta) or lo is None or hi is None:
        return NO_COMPARABLE_FORWARD
    if lo <= 0 <= hi or abs(delta) < margin:
        return NO_INFORMATION
    return CROSSING_WORSE if lo > 0 and delta >= margin else CROSSING_BETTER


def _rates(up: np.ndarray, expected: np.ndarray, used: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nan = np.full(len(used), np.nan)
    forward = np.divide(up, used, out=nan.copy(), where=used > 0)
    control = np.divide(expected, used, out=nan.copy(), where=used > 0)
    return forward, control


@dataclass(frozen=True, eq=False)
class SourceComparison:
    """Standardised comparison and bootstrap draws of one task source.

    ``counts`` has shape ``(dates, strata, 5)``; ``stratum_cells`` holds per
    stratum the registered, supported and forward pairs and the tied
    noncrossing pairs. ``draws`` has one row per bootstrap draw with the
    worsened forward count, expected control count, used forward count and
    excluded forward count.
    """

    family: str
    source: str
    dates: tuple[str, ...]
    strata: tuple[Stratum, ...]
    counts: np.ndarray
    stratum_cells: np.ndarray
    registered_n: int
    supported_n: int
    forward_n: int
    forward_ties: int
    forward_days: int
    used_forward_n: int
    excluded_forward_no_control: int
    forward_rate: float
    control_rate: float
    draws: np.ndarray
    margin: float = MARGIN

    @property
    def delta(self) -> float:
        """Forward minus standardised noncrossing worsening rate."""
        return self.forward_rate - self.control_rate

    def draw_rates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-draw forward rate, control rate and difference (``nan`` if undefined)."""
        up, expected, used = self.draws[:, 0], self.draws[:, 1], self.draws[:, 2]
        forward, control = _rates(up, expected, used)
        return forward, control, forward - control

    def row(self) -> Row:
        """Readouts of this source (one row of the by-source table)."""
        forward, control, delta = self.draw_rates()
        ci = percentile_interval(delta)
        f_ci, c_ci = percentile_interval(forward), percentile_interval(control)
        excluded = self.draws[:, 3]
        return {
            "family": self.family,
            "source": self.source,
            "registered_days": len(self.dates),
            "dates": ";".join(self.dates),
            "registered_n": self.registered_n,
            "supported_n": self.supported_n,
            "forward_n": self.forward_n,
            "forward_ties": self.forward_ties,
            "used_forward_n": self.used_forward_n,
            "excluded_forward_no_control": self.excluded_forward_no_control,
            "forward_worse_rate": self.forward_rate,
            "standardised_control_worse_rate": self.control_rate,
            "delta": self.delta,
            "forward_ci_low": f_ci[0],
            "forward_ci_high": f_ci[1],
            "control_ci_low": c_ci[0],
            "control_ci_high": c_ci[1],
            "delta_ci_low": ci[0],
            "delta_ci_high": ci[1],
            "valid_bootstrap_draws": int(np.isfinite(delta).sum()),
            "invalid_bootstrap_draws": int((~np.isfinite(delta)).sum()),
            "bootstrap_excluded_forward_min": int(excluded.min()),
            "bootstrap_excluded_forward_max": int(excluded.max()),
            "operational_label": operational_label(self.delta, ci, self.margin),
            "one_date_caveat": len(self.dates) == 1,
            "forward_days": self.forward_days,
        }

    def stratum_rows(self) -> list[Row]:
        """Per-stratum counts over all dates (one row per stratum)."""
        totals = self.counts.sum(axis=0)
        rows = []
        for (axis, small, large, fixed), t, cell in zip(self.strata, totals, self.stratum_cells):
            f, c = t[0] + t[1], t[2] + t[3]
            rows.append({
                "family": self.family, "source": self.source, "axis": axis,
                "small": small, "large": large, "fixed": fixed,
                "registered_n": int(cell[0]), "supported_n": int(cell[1]), "forward_n": int(cell[2]),
                "forward_up": int(t[0]), "forward_down": int(t[1]), "forward_tie": int(t[4]),
                "control_up": int(t[2]), "control_down": int(t[3]), "control_tie": int(cell[3]),
                "comparable": bool(f > 0 and c > 0),
                "excluded_forward_no_control": int(f if c == 0 else 0),
                "forward_worse_rate": float(t[0] / f) if f else math.nan,
                "control_worse_rate": float(t[2] / c) if c else math.nan,
            })
        return rows


@dataclass(frozen=True, eq=False)
class FamilyComparison:
    """Pooled comparison of the sources of one task family.

    The point rates are forward-count weighted means of the source rates;
    ``draws`` sums the per-draw source columns, so each draw pools the
    sources' comparable strata.
    """

    family: str
    sources: tuple[SourceComparison, ...]
    draws: np.ndarray
    forward_rate: float
    control_rate: float
    margin: float = MARGIN

    @property
    def delta(self) -> float:
        return self.forward_rate - self.control_rate

    def draw_rates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-draw pooled forward rate, control rate and difference."""
        up, expected, used = self.draws[:, 0], self.draws[:, 1], self.draws[:, 2]
        forward, control = _rates(up, expected, used)
        return forward, control, forward - control

    def row(self) -> Row:
        """Readouts of this family (one row of the family table)."""
        forward, control, delta = self.draw_rates()
        ci = percentile_interval(delta)
        f_ci, c_ci = percentile_interval(forward), percentile_interval(control)
        return {
            "family": self.family,
            "sources": len(self.sources),
            "source_days": sum(len(s.dates) for s in self.sources),
            "forward_n": sum(s.forward_n for s in self.sources),
            "forward_ties": sum(s.forward_ties for s in self.sources),
            "used_forward_n": sum(s.used_forward_n for s in self.sources),
            "excluded_forward_no_control": sum(s.excluded_forward_no_control for s in self.sources),
            "forward_worse_rate": self.forward_rate,
            "standardised_control_worse_rate": self.control_rate,
            "delta": self.delta,
            "forward_ci_low": f_ci[0],
            "forward_ci_high": f_ci[1],
            "control_ci_low": c_ci[0],
            "control_ci_high": c_ci[1],
            "delta_ci_low": ci[0],
            "delta_ci_high": ci[1],
            "valid_bootstrap_draws": int(np.isfinite(delta).sum()),
            "invalid_bootstrap_draws": int((~np.isfinite(delta)).sum()),
            "operational_label": operational_label(self.delta, ci, self.margin),
        }


@dataclass(frozen=True)
class TaskComparison:
    """Result of :func:`compare_crossings`: sources in processing order and families."""

    sources: tuple[SourceComparison, ...]
    families: tuple[FamilyComparison, ...]
    draws: int
    seed: int

    def source_rows(self) -> list[Row]:
        return [s.row() for s in self.sources]

    def family_rows(self) -> list[Row]:
        return [f.row() for f in self.families]

    def stratum_rows(self) -> list[Row]:
        return [row for s in self.sources for row in s.stratum_rows()]


def _compare_source(
    family: str,
    source: str,
    dates: Sequence[str],
    pairs: Sequence[TaskPair],
    rng: np.random.Generator,
    draws: int,
    margin: float,
) -> SourceComparison:
    if not dates:
        raise ValueError(f"{family}/{source}: at least one registered date is required")
    unknown = {p.date for p in pairs} - set(dates)
    if unknown:
        raise ValueError(f"{family}/{source}: pairs on unregistered dates {sorted(unknown)}")
    strata = tuple(sorted({p.stratum for p in pairs}))
    counts = date_stratum_counts(pairs, dates, strata)
    sidx = {s: i for i, s in enumerate(strata)}
    cells = np.zeros((len(strata), 4), dtype=np.int64)
    for p in pairs:
        k = sidx[p.stratum]
        cells[k, 0] += 1
        cells[k, 1] += p.supported
        cells[k, 2] += p.direction == FORWARD
        cells[k, 3] += p.direction == OTHER and p.error == TIE
    up, expected, used, omitted = standardised_estimates(counts.sum(axis=0))
    nd = len(dates)
    selected = rng.integers(0, nd, size=(draws, nd))
    multiplicities = np.stack([np.bincount(x, minlength=nd) for x in selected])
    sampled = np.einsum("bd,dsk->bsk", multiplicities, counts)
    bu, be, bn, bo = standardised_estimates(sampled)
    forward = [p for p in pairs if p.direction == FORWARD]
    return SourceComparison(
        family=family,
        source=source,
        dates=tuple(dates),
        strata=strata,
        counts=counts,
        stratum_cells=cells,
        registered_n=len(pairs),
        supported_n=sum(p.supported for p in pairs),
        forward_n=len(forward),
        forward_ties=sum(p.error == TIE for p in forward),
        forward_days=len({p.date for p in forward}),
        used_forward_n=int(used),
        excluded_forward_no_control=int(omitted),
        forward_rate=up / used if used else math.nan,
        control_rate=expected / used if used else math.nan,
        draws=np.column_stack([bu, be, bn, bo]),
        margin=margin,
    )


def _pool_family(family: str, sources: Sequence[SourceComparison], margin: float) -> FamilyComparison:
    pooled: Any = 0
    for s in sources:
        pooled = pooled + s.draws
    weight = np.array([s.used_forward_n for s in sources], dtype=np.int64)
    den = weight.sum()
    forward = np.array([s.forward_rate for s in sources], dtype=float)
    control = np.array([s.control_rate for s in sources], dtype=float)
    if den:
        f_rate, c_rate = float(np.nansum(forward * weight) / den), float(np.nansum(control * weight) / den)
    else:
        f_rate = c_rate = math.nan
    return FamilyComparison(family, tuple(sources), np.asarray(pooled, dtype=float).reshape(-1, 4),
                            f_rate, c_rate, margin)


def compare_crossings(
    pairs: Iterable[TaskPair],
    registry: Mapping[tuple[str, str], Sequence[str]],
    family_order: Sequence[str] | None = None,
    draws: int = DRAWS,
    seed: int = SEED,
    margin: float = MARGIN,
) -> TaskComparison:
    """Standardised forward-versus-noncrossing comparison with bootstrap.

    ``registry`` maps ``(family, source)`` to the registered dates of that
    task source (the resampling blocks; dates without pairs still count).
    ``family_order`` fixes the order in which families consume the random
    stream (default: first appearance in ``registry``); sources within a
    family are processed in sorted order and dates are sorted.
    """
    by_source: dict[tuple[str, str], list[TaskPair]] = {key: [] for key in registry}
    for p in pairs:
        key = (p.family, p.source)
        if key not in by_source:
            raise ValueError(f"{key} is not in the date registry")
        by_source[key].append(p)
    order = list(dict.fromkeys(f for f, _ in registry)) if family_order is None else list(family_order)
    missing = {f for f, _ in registry} - set(order)
    if missing:
        raise ValueError(f"family_order misses {sorted(missing)}")
    rng = np.random.default_rng(seed)
    sources: list[SourceComparison] = []
    families: list[FamilyComparison] = []
    for family in order:
        members = []
        for key in sorted(k for k in registry if k[0] == family):
            dates = sorted(registry[key])
            members.append(_compare_source(family, key[1], dates, by_source[key], rng, draws, margin))
        sources.extend(members)
        families.append(_pool_family(family, members, margin))
    return TaskComparison(tuple(sources), tuple(families), draws, seed)


def compensated_mean(values: Sequence[float]) -> float:
    """Mean with Kahan-compensated summation in the given order.

    This is the summation used by grouped means of the sealed analysis, so
    category means agree bit for bit. ``nan`` for no values.
    """
    total = compensation = 0.0
    for value in values:
        y = value - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total / len(values) if len(values) else math.nan


def category_summary(pairs: Iterable[TaskPair]) -> list[Row]:
    """Pairs per family, source and category with error directions.

    ``worse_rate`` excludes ties; ``mean_delta_error`` averages the finite
    error changes in the original units (ties included) in pair order. Rows
    are sorted by family, source and category.
    """
    groups: dict[tuple[str, str, str], list[TaskPair]] = {}
    for p in pairs:
        groups.setdefault((p.family, p.source, p.category), []).append(p)
    rows = []
    for (family, source, category), members in sorted(groups.items()):
        errors = [p.error for p in members]
        up, down = errors.count(UP), errors.count(DOWN)
        deltas = [p.delta_error for p in members if math.isfinite(p.delta_error)]
        rows.append({
            "family": family, "source": source, "category": category, "n": len(members),
            "n_up": up, "n_down": down, "n_tie": errors.count(TIE),
            "mean_delta_error": compensated_mean(deltas),
            "worse_rate": worsening_rate(up, down),
        })
    return rows
