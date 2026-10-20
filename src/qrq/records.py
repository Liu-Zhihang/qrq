"""Space-time point records: the on-disk layout and validation helpers.

A record is one observed position of one entity at one time. Records are
stored as a flat little-endian binary array of the structured dtype
:data:`POINT` (40 bytes per record):

=================  ======  ==================================================
field              dtype   meaning
=================  ======  ==================================================
``x``              <f8     projected easting in metres (metric CRS)
``y``              <f8     projected northing in metres (metric CRS)
``t``              <f8     seconds since the start of the index day
``entity``         <i8     opaque entity proxy (vehicle, device, person, ...)
``source_ordinal`` <u8     strictly increasing row ordinal within the source
=================  ======  ==================================================

The index backend consumes an ``(N, 3)`` float64 coordinate array
``(x, y, t)`` plus an ``(N,)`` int64 entity array; :func:`backend_arrays`
produces both with the exact dtype and contiguity checks the backend expects.
"""
from __future__ import annotations

from os import PathLike
from typing import Union

import numpy as np

POINT = np.dtype(
    [
        ("x", "<f8"),
        ("y", "<f8"),
        ("t", "<f8"),
        ("entity", "<i8"),
        ("source_ordinal", "<u8"),
    ]
)
"""Structured dtype of one point record (40 bytes)."""

RECORD_BYTES = POINT.itemsize
DAY_SECONDS = 86400

StrPath = Union[str, "PathLike[str]"]


def make_points(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    entity: np.ndarray | None = None,
    source_ordinal: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble a :data:`POINT` array from column arrays.

    ``entity`` defaults to zeros and ``source_ordinal`` to ``0..N-1``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    out = np.empty(n, dtype=POINT)
    out["x"] = x
    out["y"] = np.asarray(y, dtype=np.float64)
    out["t"] = np.asarray(t, dtype=np.float64)
    out["entity"] = 0 if entity is None else np.asarray(entity, dtype=np.int64)
    out["source_ordinal"] = (
        np.arange(n, dtype=np.uint64)
        if source_ordinal is None
        else np.asarray(source_ordinal, dtype=np.uint64)
    )
    return out


def load_points(path: StrPath, mmap: bool = True) -> np.ndarray:
    """Load a binary record file written with the :data:`POINT` layout.

    The file size must be an exact multiple of :data:`RECORD_BYTES`. With
    ``mmap=True`` the file is memory-mapped read-only.
    """
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
    if size % RECORD_BYTES:
        raise ValueError(f"file size {size} is not a multiple of {RECORD_BYTES} bytes")
    if size == 0:
        return np.empty(0, dtype=POINT)
    if mmap:
        return np.memmap(path, dtype=POINT, mode="r")
    return np.fromfile(path, dtype=POINT)


def save_points(path: StrPath, points: np.ndarray) -> None:
    """Write a :data:`POINT` array as a flat binary record file."""
    validate_points(points)
    np.ascontiguousarray(points, dtype=POINT).tofile(path)


def validate_points(
    points: np.ndarray,
    day_seconds: float | None = None,
    require_ordered: bool = False,
) -> int:
    """Validate a record array and return its length.

    Checks: structured dtype equals :data:`POINT`, coordinates are finite.
    Optional checks: every ``t`` lies in ``[0, day_seconds)`` and
    ``source_ordinal`` is strictly increasing (the sealed workloads used both).
    Raises :class:`ValueError` on the first violation.
    """
    if not isinstance(points, np.ndarray) or points.dtype != POINT:
        raise ValueError("records must be a numpy array with the qrq POINT dtype")
    if points.ndim != 1:
        raise ValueError("records must be a one-dimensional structured array")
    for name in ("x", "y", "t"):
        if not np.isfinite(points[name]).all():
            raise ValueError(f"non-finite values in field {name!r}")
    if day_seconds is not None:
        t = points["t"]
        if not np.all((t >= 0) & (t < day_seconds)):
            raise ValueError("time values outside the index day")
    if require_ordered and len(points) > 1:
        ordinal = points["source_ordinal"]
        if not np.all(ordinal[1:] > ordinal[:-1]):
            raise ValueError("source_ordinal is not strictly increasing")
    return int(len(points))


def coordinates(points: np.ndarray) -> np.ndarray:
    """Return the C-contiguous ``(N, 3)`` float64 ``(x, y, t)`` array."""
    if isinstance(points, np.ndarray) and points.dtype == POINT:
        return np.ascontiguousarray(np.column_stack([points["x"], points["y"], points["t"]]))
    xyz = np.ascontiguousarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must be a POINT array or an (N, 3) array")
    return xyz


def backend_arrays(
    points: np.ndarray, entity: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz, entity)`` arrays in the exact form the backend expects.

    ``points`` is either a :data:`POINT` array (``entity`` is then taken from
    the record field unless given) or an ``(N, 3)`` array, in which case
    ``entity`` is required. The checks mirror the reference array guard:
    float64 C-contiguous ``(N, 3)`` coordinates, finite values, and an exact
    int64 entity array without implicit casts.
    """
    xyz = coordinates(points)
    if entity is None:
        if not (isinstance(points, np.ndarray) and points.dtype == POINT):
            raise ValueError("entity ids are required for plain coordinate arrays")
        entity = np.array(points["entity"], dtype=np.int64)
    if not isinstance(entity, np.ndarray) or entity.dtype != np.int64:
        raise ValueError("entity ids must be an explicit int64 numpy array")
    entity = np.ascontiguousarray(entity)
    if entity.shape != (len(xyz),):
        raise ValueError("entity ids must have shape (N,)")
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite record coordinates")
    return xyz, entity
