"""Experimental semantic extension point."""
from __future__ import annotations

import numpy as np
import pytest

import qrq
from qrq.semantic import CategoricalIntervalEncoder, SemanticBox, SemanticEncoder


def test_encoder_protocol_and_intervals():
    enc = CategoricalIntervalEncoder(["residential", "mixed", "commercial", "industrial"])
    assert isinstance(enc, SemanticEncoder)
    assert enc.encode(["mixed", "industrial"]).tolist() == [1.0, 3.0]
    assert enc.interval(["commercial", "mixed"]) == (1.0, 2.0)
    with pytest.raises(ValueError):
        enc.interval(["residential", "commercial"])
    with pytest.raises(ValueError):
        enc.encode(["park"])
    with pytest.raises(ValueError):
        CategoricalIntervalEncoder(["a", "a"])


def test_semantic_box_filters_after_spacetime():
    enc = CategoricalIntervalEncoder(["a", "b", "c"])
    pts = qrq.make_points([0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 0, 0])
    labels = enc.encode(["a", "b", "c", "b"])
    base = qrq.SpaceTimeBox(0, 2, 0, 2, 0, 0)
    box = SemanticBox.from_labels(base, enc, ["b", "c"])
    assert box.spacetime() == base
    assert box.contains(pts, labels).tolist() == [False, True, True, False]
    assert box.count(pts, labels) == 2
    with pytest.raises(ValueError):
        box.contains(pts)
    with pytest.raises(ValueError):
        SemanticBox(0, 1, 0, 1, 0, 1, smin=2.0, smax=1.0)
    assert np.isinf(SemanticBox(0, 1, 0, 1, 0, 1).smax)
