"""Record layout, validation and the brute-force box predicate."""
from __future__ import annotations

import numpy as np
import pytest

import qrq
from qrq.records import RECORD_BYTES, backend_arrays, coordinates


def test_point_dtype_layout():
    assert qrq.POINT.itemsize == RECORD_BYTES == 40
    assert qrq.POINT.names == ("x", "y", "t", "entity", "source_ordinal")
    assert [qrq.POINT.fields[n][0].str for n in qrq.POINT.names] == ["<f8", "<f8", "<f8", "<i8", "<u8"]


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    pts = qrq.make_points(rng.uniform(size=50), rng.uniform(size=50), rng.uniform(0, 86400, 50),
                          rng.integers(-5, 5, 50))
    path = tmp_path / "day.bin"
    qrq.save_points(path, pts)
    assert path.stat().st_size == 50 * 40
    for mmap in (True, False):
        loaded = qrq.load_points(path, mmap=mmap)
        assert np.array_equal(np.asarray(loaded), pts)
    assert qrq.validate_points(np.asarray(qrq.load_points(path)), day_seconds=86400, require_ordered=True) == 50
    (tmp_path / "bad.bin").write_bytes(b"x" * 41)
    with pytest.raises(ValueError):
        qrq.load_points(tmp_path / "bad.bin")


def test_validation_failures():
    pts = qrq.make_points([0.0, 1.0], [0.0, 1.0], [0.0, 90000.0])
    with pytest.raises(ValueError):
        qrq.validate_points(pts, day_seconds=86400)
    pts["source_ordinal"] = [3, 3]
    with pytest.raises(ValueError):
        qrq.validate_points(pts, require_ordered=True)
    pts["x"][0] = np.nan
    with pytest.raises(ValueError):
        qrq.validate_points(pts)
    with pytest.raises(ValueError):
        qrq.validate_points(np.zeros((2, 3)))


def test_backend_arrays():
    pts = qrq.make_points([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7, 8])
    xyz, ids = backend_arrays(pts)
    assert xyz.dtype == np.float64 and xyz.flags.c_contiguous and xyz.shape == (2, 3)
    assert ids.dtype == np.int64 and ids.tolist() == [7, 8]
    with pytest.raises(ValueError):
        backend_arrays(xyz)
    with pytest.raises(ValueError):
        backend_arrays(xyz, np.array([1, 2], dtype=np.uint64))
    with pytest.raises(ValueError):
        backend_arrays(np.array([[np.inf, 0, 0]]), np.array([1], dtype=np.int64))


def test_box_validation_and_closed_bounds():
    with pytest.raises(ValueError):
        qrq.SpaceTimeBox(1, 0, 0, 1, 0, 1)
    with pytest.raises(ValueError):
        qrq.SpaceTimeBox(0, 1, 0, 1, 0, float("nan"))
    with pytest.raises(ValueError):
        qrq.SpaceTimeBox.from_bounds([0, 1])
    box = qrq.SpaceTimeBox.from_bounds([0, 1, 0, 1, 0, 1])
    xyz = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1.0000001], [0.5, -1e-12, 0.5]], dtype=float)
    assert box.contains(xyz).tolist() == [True, True, False, False]
    assert box.count(qrq.make_points(*xyz.T)) == 2
    assert isinstance(box, qrq.Predicate)
    assert np.array_equal(coordinates(qrq.make_points(*xyz.T)), xyz)
