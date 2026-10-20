"""Query predicates over space-time records.

The only predicate used for published results is the closed axis-aligned
space-time box :class:`SpaceTimeBox`. Its brute-force :meth:`~SpaceTimeBox.count`
is the independent membership count every index answer is checked against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from .records import coordinates


@runtime_checkable
class Predicate(Protocol):
    """A record predicate that can be evaluated exhaustively.

    Implementations return a boolean mask with one entry per record. The
    mask is the ground truth used to verify index-based answers.
    """

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask of records satisfying the predicate."""
        ...


@dataclass(frozen=True)
class SpaceTimeBox:
    """Closed box ``[xmin, xmax] x [ymin, ymax] x [tmin, tmax]``.

    All six bounds must be finite and every lower bound must not exceed its
    upper bound (degenerate, zero-extent boxes are allowed). A record is
    inside when every coordinate lies within the closed interval.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    tmin: float
    tmax: float

    def __post_init__(self) -> None:
        bounds = self.bounds
        if not all(math.isfinite(v) for v in bounds):
            raise ValueError("box bounds must be finite")
        if self.xmin > self.xmax or self.ymin > self.ymax or self.tmin > self.tmax:
            raise ValueError("box lower bound exceeds upper bound")

    @classmethod
    def from_bounds(cls, bounds: Iterable[float]) -> "SpaceTimeBox":
        """Build a box from ``[xmin, xmax, ymin, ymax, tmin, tmax]``."""
        values = list(bounds)
        if len(values) != 6:
            raise ValueError("six bounds required")
        return cls(*values)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """The six bounds in backend argument order."""
        return (self.xmin, self.xmax, self.ymin, self.ymax, self.tmin, self.tmax)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Vectorised brute-force membership mask (closed bounds).

        ``points`` is a :data:`qrq.records.POINT` array or an ``(N, 3)``
        ``(x, y, t)`` array.
        """
        xyz = coordinates(points)
        return (
            (xyz[:, 0] >= self.xmin)
            & (xyz[:, 0] <= self.xmax)
            & (xyz[:, 1] >= self.ymin)
            & (xyz[:, 1] <= self.ymax)
            & (xyz[:, 2] >= self.tmin)
            & (xyz[:, 2] <= self.tmax)
        )

    def count(self, points: np.ndarray) -> int:
        """Number of records inside the box (independent of any index)."""
        return int(np.count_nonzero(self.contains(points)))
