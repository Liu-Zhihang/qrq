"""A space-time box extended by one semantic interval (experimental)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..geometry import SpaceTimeBox
from .protocols import SemanticEncoder


@dataclass(frozen=True)
class SemanticBox(SpaceTimeBox):
    """Closed box over ``(x, y, t, s)`` where ``s`` is an encoded semantic axis.

    The current indexes are three-dimensional, so a semantic box can only be
    answered by running :meth:`spacetime` through an index and filtering the
    semantic interval afterwards. A native 4-D index is future work.
    """

    smin: float = -math.inf
    smax: float = math.inf

    def __post_init__(self) -> None:
        super().__post_init__()
        if math.isnan(self.smin) or math.isnan(self.smax) or self.smin > self.smax:
            raise ValueError("invalid semantic interval")

    @classmethod
    def from_labels(
        cls, box: SpaceTimeBox, encoder: SemanticEncoder, labels: Sequence[str]
    ) -> "SemanticBox":
        """Combine a space-time box with the interval of ``labels``."""
        smin, smax = encoder.interval(labels)
        return cls(*box.bounds, smin=smin, smax=smax)

    def spacetime(self) -> SpaceTimeBox:
        """The space-time part, usable with the 3-D indexes."""
        return SpaceTimeBox(*self.bounds)

    def contains(self, points: np.ndarray, semantic: np.ndarray | None = None) -> np.ndarray:
        """Membership mask; ``semantic`` holds one encoded value per record."""
        if semantic is None:
            raise ValueError("encoded semantic values are required")
        values = np.asarray(semantic, dtype=np.float64)
        inside = super().contains(points)
        if values.shape != inside.shape:
            raise ValueError("one semantic value per record is required")
        return inside & (values >= self.smin) & (values <= self.smax)

    def count(self, points: np.ndarray, semantic: np.ndarray | None = None) -> int:
        return int(np.count_nonzero(self.contains(points, semantic)))
