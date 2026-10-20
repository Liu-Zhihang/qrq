"""The registered window workload and adjacent-scale pairs.

Each source-day is queried with a fixed grid of baseline windows centred on
a 3 x 3 anchor lattice around a city reference point ``(cx, cy)``:

* anchors ``x = cx + (ix - 1) * step``, ``y = cy + (iy - 1) * step`` with
  ``ix, iy in {0, 1, 2}`` and ``step = 20000 / 3`` metres;
* temporal centres 21600, 43200 and 64800 s (06:00, 12:00, 18:00);
* square spatial widths 250, 500, 1000, 2000, 5000 and 10000 m;
* durations 900, 3600, 21600 and 86400 s.

That is 9 * 3 * 6 * 4 = 648 windows per source-day. A window whose time
span leaves ``[0, 86400]`` is registered with status
``unsupported_index_span`` and is never costed.

Adjacent-scale pairs change exactly one extent to the next grid value
(width for the ``space`` axis, duration for the ``time`` axis) while keeping
the anchor, temporal centre and the other extent fixed.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .geometry import SpaceTimeBox

WIDTHS: tuple[int, ...] = (250, 500, 1000, 2000, 5000, 10000)
DURATIONS: tuple[int, ...] = (900, 3600, 21600, 86400)
CENTRES: tuple[int, ...] = (21600, 43200, 64800)
ANCHOR_STEP: float = 20000 / 3
DAY_SECONDS: int = 86400

SCHEDULED = "scheduled"
UNSUPPORTED_INDEX_SPAN = "unsupported_index_span"

Direction = Literal["forward", "reverse", "no_cross"]
Axis = Literal["space", "time"]


@dataclass(frozen=True)
class Window:
    """One registered baseline window."""

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


def register_windows(
    cx: float,
    cy: float,
    widths: Sequence[float] = WIDTHS,
    durations: Sequence[float] = DURATIONS,
    centres: Sequence[float] = CENTRES,
    anchor_step: float = ANCHOR_STEP,
    day_seconds: float = DAY_SECONDS,
) -> list[Window]:
    """Return the baseline windows of one source-day, in registration order.

    With the defaults this yields the 648 windows of the sealed workload,
    ordered by ``(ix, iy, centre, width, duration)``. Bounds are
    ``[x - w/2, x + w/2, y - w/2, y + w/2, tc - d/2, tc + d/2]``.
    """
    out: list[Window] = []
    for ix, iy, tc, w, d in itertools.product(range(3), range(3), centres, widths, durations):
        x = cx + (ix - 1) * anchor_step
        y = cy + (iy - 1) * anchor_step
        lo, hi = tc - d / 2, tc + d / 2
        status = UNSUPPORTED_INDEX_SPAN if lo < 0 or hi > day_seconds else SCHEDULED
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


def adjacent_pairs(
    items: Sequence[Any],
    widths: Sequence[float] = WIDTHS,
    durations: Sequence[float] = DURATIONS,
) -> list[ScalePair]:
    """Return all adjacent-scale pairs among ``items``.

    ``items`` are objects with ``anchor_index``, ``temporal_center_s``,
    ``width_m`` and ``duration_s`` attributes (for example :class:`Window`
    or :class:`qrq.ledger.LedgerEndpoint`), drawn from one source-day and
    one comparator. A pair is reported only when both windows are present,
    so missing (empty or unsupported) windows are never skipped over.
    Pairs are listed per small window, ``space`` before ``time``.
    """
    index = {
        (_anchor(a), a.temporal_center_s, a.width_m, a.duration_s): a for a in items
    }
    pairs: list[ScalePair] = []
    for a in items:
        for axis, values, field_name in (
            ("space", list(widths), "width_m"),
            ("time", list(durations), "duration_s"),
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
