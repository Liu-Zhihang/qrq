"""qrq: quantum range queries over space-time point records.

qrq compares matched classical and quantum access routes for closed
space-time box queries over static record indexes. For every window it
counts the resources of one quantum first-witness filtering route and four
exact classical routes on three tree layouts, prices them with an explicit
:class:`~qrq.prices.PriceTable`, and reports the cost envelope
``D = log2(Q / C)``. It is a resource-accounting model, not a hardware
runtime benchmark.

Typical use::

    import numpy as np
    import qrq

    rng = np.random.default_rng(0)
    pts = qrq.make_points(rng.uniform(0, 1e4, 5000), rng.uniform(0, 1e4, 5000),
                          rng.uniform(0, 86400, 5000), rng.integers(0, 50, 5000))
    indexes = {kind: qrq.SpatioTemporalIndex.build(pts, kind=kind) for kind in qrq.TREE_KINDS}
    result = qrq.evaluate_window(indexes, qrq.SpaceTimeBox(0, 5000, 0, 5000, 0, 43200),
                                 qrq.L1, verify=True)
"""
from .engine import Envelope, WindowResult, evaluate_window, include_amplified, select_envelope
from .geometry import Predicate, SpaceTimeBox
from .index import TREE_KINDS, AccessPlan, ClassicalAccess, SpatioTemporalIndex
from .prices import L1, UNIT, PriceTable, decompose_quantum_cost, route_cost, route_costs
from .records import POINT, backend_arrays, load_points, make_points, save_points, validate_points
from .routes import CLASSICAL_ROUTES, QRQ_FW, ROUTES, AccessRoute, evaluate_routes
from .workload import Window, adjacent_pairs, classify_pair, register_windows

__version__ = "0.2.0"

__all__ = [
    "AccessPlan",
    "AccessRoute",
    "CLASSICAL_ROUTES",
    "ClassicalAccess",
    "Envelope",
    "L1",
    "POINT",
    "PriceTable",
    "Predicate",
    "QRQ_FW",
    "ROUTES",
    "SpaceTimeBox",
    "SpatioTemporalIndex",
    "TREE_KINDS",
    "UNIT",
    "Window",
    "WindowResult",
    "adjacent_pairs",
    "backend_arrays",
    "classify_pair",
    "decompose_quantum_cost",
    "evaluate_routes",
    "evaluate_window",
    "include_amplified",
    "load_points",
    "make_points",
    "register_windows",
    "route_cost",
    "route_costs",
    "save_points",
    "select_envelope",
    "validate_points",
]
