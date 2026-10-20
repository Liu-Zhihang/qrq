"""A toy semantic encoder over an ordered list of categories (experimental)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class CategoricalIntervalEncoder:
    """Encode categories by their position in a fixed order.

    ``CategoricalIntervalEncoder(["residential", "mixed", "commercial"])``
    maps the labels to 0, 1 and 2. A label set is a valid interval when its
    positions are consecutive, e.g. ``{"mixed", "commercial"}`` -> ``(1, 2)``.
    Choosing an order that makes useful queries contiguous is the actual
    research problem and is outside the scope of this toy implementation.
    """

    def __init__(self, categories: Sequence[str]):
        if len(set(categories)) != len(categories) or not categories:
            raise ValueError("categories must be a non-empty sequence of unique labels")
        self.categories = tuple(categories)
        self._position = {label: float(i) for i, label in enumerate(self.categories)}

    def encode(self, labels: Sequence[str]) -> np.ndarray:
        try:
            return np.array([self._position[label] for label in labels], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"unknown category {exc.args[0]!r}") from None

    def interval(self, labels: Sequence[str]) -> tuple[float, float]:
        values = sorted(set(self.encode(labels).tolist()))
        if not values:
            raise ValueError("at least one label is required")
        if values[-1] - values[0] + 1 != len(values):
            raise ValueError("labels are not contiguous in the encoder order")
        return values[0], values[-1]
