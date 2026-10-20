"""Locate sealed evidence passed through environment variables and compare tables."""
from __future__ import annotations

import csv
import gzip
import math
import os
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import pytest


def env_dir(name: str) -> Path:
    """Return the directory named by ``name`` or skip the calling test."""
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"{name} does not point to a directory")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a plain or gzip CSV file as strings."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matches(stored: str, value: Any, rel_tol: float) -> bool:
    if stored == "":
        return value is None or (isinstance(value, float) and math.isnan(value))
    if isinstance(value, (bool, str)):
        return stored == str(value)
    if isinstance(value, int):
        return int(stored) == value
    if rel_tol:
        return math.isclose(float(stored), float(value), rel_tol=rel_tol, abs_tol=rel_tol)
    return float(stored) == float(value)


def assert_rows_equal(
    stored: Sequence[Mapping[str, str]],
    ours: Sequence[Mapping[str, Any]],
    tolerances: Mapping[str, float] | None = None,
    ignore: Collection[str] = (),
) -> None:
    """Compare recomputed rows with a sealed CSV table, column by column.

    Columns and row order must agree. Numbers compare exactly (CSV floats are
    stored at full precision) unless ``tolerances`` gives a relative
    tolerance for a column; an empty cell matches ``None`` or ``nan``.
    """
    tolerances = tolerances or {}
    assert len(stored) == len(ours)
    for i, (a, b) in enumerate(zip(stored, ours)):
        assert [c for c in a if c not in ignore] == [c for c in b if c not in ignore], i
        for column, text in a.items():
            if column in ignore:
                continue
            assert _matches(text, b[column], tolerances.get(column, 0.0)), (i, column, text, b[column])
