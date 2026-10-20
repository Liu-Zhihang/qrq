"""Registered window workloads and adjacent-scale pairs.

A workload is a fixed grid of closed space-time windows registered before
any cost is computed. The grid is supplied by the caller as a
:class:`WindowGrid`:

* an ``anchors x anchors`` lattice centred on a reference point ``(cx, cy)``
  with spacing ``anchor_step``: anchor ``(ix, iy)`` sits at
  ``x = cx + (ix - (anchors - 1) / 2) * anchor_step`` and likewise for ``y``;
* temporal centres ``centres`` (seconds of the index day);
* square spatial widths ``widths`` and durations ``durations``, each in
  ascending order.

Every combination is one window, registered in the order
``(ix, iy, centre, width, duration)``. A window whose time span leaves
``[0, day_seconds]`` is registered with status ``unsupported_index_span`` and
is never costed.

Adjacent-scale pairs change exactly one extent to the next grid value
(width for the ``space`` axis, duration for the ``time`` axis) while keeping
the anchor, temporal centre and the other extent fixed.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .geometry import SpaceTimeBox
from .records import DAY_SECONDS

SCHEDULED = "scheduled"
UNSUPPORTED_INDEX_SPAN = "unsupported_index_span"

Direction = Literal["forward", "reverse", "no_cross"]
Axis = Literal["space", "time"]


def _ascending(name: str, values: Sequence[float]) -> tuple[float, ...]:
    values = tuple(values)
    if not values or any(v <= 0 for v in values) or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be positive and strictly ascending")
    return values


@dataclass(frozen=True)
class WindowGrid:
    """Caller-supplied specification of a registered window grid.

    ``widths`` (metres) and ``durations`` (seconds) are strictly ascending;
    ``centres`` are temporal centres in seconds of the index day;
    ``anchors`` is the number of anchors per axis and ``anchor_step`` their
    spacing in metres. ``day_seconds`` is the length of the index day
    (the record convention of :mod:`qrq.records` by default).
    """

    widths: tuple[float, ...]
    durations: tuple[float, ...]
    centres: tuple[float, ...]
    anchor_step: float
    anchors: int = 3
    day_seconds: float = DAY_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "widths", _ascending("widths", self.widths))
        object.__setattr__(self, "durations", _ascending("durations", self.durations))
        object.__setattr__(self, "centres", tuple(self.centres))
        if not self.centres:
            raise ValueError("at least one temporal centre is required")
        if self.anchors < 1 or not self.anchor_step > 0:
            raise ValueError("anchors must be >= 1 and anchor_step positive")

    @property
    def size(self) -> int:
        """Number of registered windows per reference point."""
        return self.anchors ** 2 * len(self.centres) * len(self.widths) * len(self.durations)


@dataclass(frozen=True)
class Window:
    """One registered window."""

    query_id: str
    anchor_index: tuple[int, int]
    width_m: float
    duration_s: float
    temporal_center_s: float
    box: SpaceTimeBox
    status: str

    @property
    def supported(self) -> bool:
        """False when the time span leaves the index day."""
        return self.status != UNSUPPORTED_INDEX_SPAN


def window_id(ix: int, iy: int, centre: Any, width: Any, duration: Any) -> str:
    """Registered window identifier ``g{ix}{iy}_c{tc}_w{w}_d{d}``."""
    return f"g{ix}{iy}_c{centre}_w{width}_d{duration}"


def register_windows(cx: float, cy: float, grid: WindowGrid) -> list[Window]:
    """Return the registered windows of ``grid`` around ``(cx, cy)``, in order.

    Bounds are ``[x - w/2, x + w/2, y - w/2, y + w/2, tc - d/2, tc + d/2]``.
    """
    centre_offset = (grid.anchors - 1) / 2
    out: list[Window] = []
    lattice = range(grid.anchors)
    for ix, iy, tc, w, d in itertools.product(lattice, lattice, grid.centres, grid.widths, grid.durations):
        x = cx + (ix - centre_offset) * grid.anchor_step
        y = cy + (iy - centre_offset) * grid.anchor_step
        lo, hi = tc - d / 2, tc + d / 2
        status = UNSUPPORTED_INDEX_SPAN if lo < 0 or hi > grid.day_seconds else SCHEDULED
        box = SpaceTimeBox(x - w / 2, x + w / 2, y - w / 2, y + w / 2, lo, hi)
        out.append(Window(window_id(ix, iy, tc, w, d), (ix, iy), w, d, tc, box, status))
    return out


@dataclass(frozen=True)
class ScalePair:
    """Two windows that differ by one step along one extent."""

    small: Any
    large: Any
    axis: Axis


def _anchor(item: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in item.anchor_index)


def adjacent_pairs(items: Sequence[Any], grid: WindowGrid) -> list[ScalePair]:
    """Return all adjacent-scale pairs among ``items`` on ``grid``.

    ``items`` are objects with ``anchor_index``, ``temporal_center_s``,
    ``width_m`` and ``duration_s`` attributes (for example :class:`Window`
    or :class:`qrq.ledger.LedgerEndpoint`), drawn from one reference point
    and one comparator. A pair is reported only when both windows are
    present, so missing (empty or unsupported) windows are never skipped
    over. Pairs are listed per small window, ``space`` before ``time``.
    """
    index = {
        (_anchor(a), a.temporal_center_s, a.width_m, a.duration_s): a for a in items
    }
    pairs: list[ScalePair] = []
    for a in items:
        for axis, values, field_name in (
            ("space", list(grid.widths), "width_m"),
            ("time", list(grid.durations), "duration_s"),
        ):
            j = values.index(getattr(a, field_name))
            if j == len(values) - 1:
                continue
            w = values[j + 1] if axis == "space" else a.width_m
            d = values[j + 1] if axis == "time" else a.duration_s
            b = index.get((_anchor(a), a.temporal_center_s, w, d))
            if b is not None:
                pairs.append(ScalePair(a, b, axis))  # type: ignore[arg-type]
    return pairs


def classify_pair(d_small: float, d_large: float) -> Direction:
    """Ordering change between the smaller and the larger window.

    ``forward``: the classical route wins the small window (``D > 0``) and
    the quantum route wins the large one (``D < 0``). ``reverse`` is the
    opposite crossing; every other combination (including ``D == 0``) is
    ``no_cross``.
    """
    if d_small > 0 and d_large < 0:
        return "forward"
    if d_small < 0 and d_large > 0:
        return "reverse"
    return "no_cross"
