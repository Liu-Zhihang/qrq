"""Interfaces for semantic predicates (experimental)."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class SemanticEncoder(Protocol):
    """Map semantic labels onto one ordered numeric axis.

    A semantic predicate is indexable alongside space and time only if the
    labels it selects form a closed interval on that axis. ``encode`` maps
    labels to axis values; ``interval`` returns the closed axis interval
    covering a set of labels and raises :class:`ValueError` when the labels
    are not contiguous on the axis.
    """

    def encode(self, labels: Sequence[str]) -> np.ndarray:
        """Return float64 axis values, one per label."""
        ...

    def interval(self, labels: Sequence[str]) -> tuple[float, float]:
        """Return the closed axis interval of a contiguous label set."""
        ...
