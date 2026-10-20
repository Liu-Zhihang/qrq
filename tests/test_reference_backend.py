"""Optional cross-check against a separately compiled reference backend.

Set ``QRQ_REFERENCE_EXTENSION`` to a ``_qrq_matched`` shared library built
from ``cpp/trajectory_trees_3d.cpp`` with the reference command
``g++ -O2 -shared -std=c++17 -fPIC -fvisibility=hidden`` (plus the Python
and pybind11 include flags) for the running interpreter. The test checks
that the packaged extension returns identical ``matched_query`` results.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os

import pytest

import qrq

from synthetic import random_boxes, synthetic_cloud


@pytest.fixture(scope="module")
def reference_module():
    path = os.environ.get("QRQ_REFERENCE_EXTENSION")
    if not path:
        pytest.skip("QRQ_REFERENCE_EXTENSION is not set")
    loader = importlib.machinery.ExtensionFileLoader("_qrq_matched", path)
    spec = importlib.util.spec_from_file_location("_qrq_matched", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.parametrize("cloud_kind", ["uniform", "clustered", "lattice"])
@pytest.mark.parametrize("kind", qrq.TREE_KINDS)
def test_matched_query_identical(reference_module, cloud_kind, kind):
    points = synthetic_cloud(cloud_kind, 2500, seed=77)
    ours = qrq.SpatioTemporalIndex.build(points, kind=kind)
    xyz, ids = qrq.backend_arrays(points)
    theirs = reference_module.QuantumTrajectoryTree3D(xyz, xyz, ids, 4, kind)
    assert (theirs.size, theirs.height, theirs.node_count) == (ours.size, ours.height, ours.node_count)
    assert theirs.matched_layout()["rank_to_id"] == ours.rank_to_record().tolist()
    for box in random_boxes(points, 100, seed=3):
        assert dict(theirs.matched_query(*box.bounds)) == ours.matched_query(box)
