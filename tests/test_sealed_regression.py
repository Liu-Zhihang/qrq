"""Regression against the sealed published evidence (skipped by default).

Set ``QRQ_SEALED_LEDGERS`` to the evidence root holding the 321 per-tree
ledgers, ``QRQ_SEALED_TABLES`` to the directory holding ``endpoints.csv.gz``
and ``scale_pairs.csv.gz``, and optionally ``QRQ_SEALED_INPUTS`` to the
directory holding ``geometry_qa.json`` and ``queries/``. None of these data
are distributed with the repository.
"""
from __future__ import annotations

import collections
import json
import math

import pytest

from qrq.workload import classify_pair, register_windows

from evidence import env_dir

PUBLISHED = dict(days=107, ledgers=321, windows=13370, quantum_lower=742, pairs=17592, forward=1060, reverse=11)

# Published per-source table: non-empty windows, Q < C windows, forward and reverse pairs.
# Keys are the source identifiers of the sealed archive.
PUBLISHED_BY_SOURCE = {
    "tdrive": (3305, 208, 330, 0),
    "geolife": (265, 4, 8, 0),
    "jinan": (2460, 232, 345, 0),
    "dublin": (282, 30, 41, 9),
    "porto": (240, 2, 4, 0),
    "montreal": (1253, 18, 32, 0),
    "barcelona": (85, 0, 0, 0),
    "rio": (817, 27, 47, 0),
    "stellenbosch": (159, 7, 10, 0),
    "nyc_mat": (328, 28, 34, 0),
    "paris_mat": (138, 0, 0, 0),
    "sf_cabspotting": (1698, 93, 101, 0),
    "rome_taxi": (2150, 91, 104, 0),
    "brighton_shl": (190, 2, 4, 2),
}


def test_ledger_costs_and_envelope(sealed_reproduction):
    """Every stored L1 route cost is recomputed from its resource vectors."""
    r = sealed_reproduction
    assert r.days == PUBLISHED["days"] and r.ledgers == PUBLISHED["ledgers"]
    assert r.priced_routes > 0 and r.priced_routes % 5 == 0
    assert r.max_relative_error <= 1e-12
    summary = r.summary()
    for key in ("windows", "quantum_lower", "pairs", "forward", "reverse"):
        assert summary[key] == PUBLISHED[key], key
    for source, (windows, lower, forward, reverse) in PUBLISHED_BY_SOURCE.items():
        got = summary["by_source"][source]
        assert (got["windows"], got["quantum_lower"], got["forward"], got["reverse"]) == (
            windows, lower, forward, reverse), source
    for w in r.windows:
        assert w.D == math.log2(w.Q / w.C)
        assert math.isclose(w.Q, w.prefix + w.trials * w.trial_cost, rel_tol=1e-12)


def _tables():
    pd = pytest.importorskip("pandas")
    root = env_dir("QRQ_SEALED_TABLES")
    keep = dict(dtype={"date": str, "anchor": str, "query_id": str, "small_query": str, "large_query": str})
    ep = pd.read_csv(root / "endpoints.csv.gz", **keep)
    sp = pd.read_csv(root / "scale_pairs.csv.gz", **keep)
    select = lambda df: df[(df.comparator == "envelope") & (df.arm == "baseline") & (df.variant == "base")]
    return select(ep), select(sp)


def test_published_counts_in_tables():
    """The sealed tables state the published counts; directions re-classify."""
    ep, sp = _tables()
    assert len(ep) == PUBLISHED["windows"]
    assert int((ep.Q < ep.C).sum()) == PUBLISHED["quantum_lower"]
    assert len(sp) == PUBLISHED["pairs"]
    directions = [classify_pair(a, b) for a, b in zip(sp.small_D, sp.large_D)]
    assert directions == sp.direction.tolist()
    counts = collections.Counter(directions)
    assert (counts["forward"], counts["reverse"]) == (PUBLISHED["forward"], PUBLISHED["reverse"])


def test_envelope_matches_tables(sealed_reproduction):
    """Recomputed envelopes and pairs equal the sealed tables row by row."""
    ep, sp = _tables()
    table = {(r.source_id, r.date, r.query_id): r for r in ep.itertuples(index=False)}
    assert len(table) == len(ep)
    ours = {(w.source_id, w.date, w.query_id): w for w in sealed_reproduction.windows}
    assert ours.keys() == table.keys()
    for key, w in ours.items():
        t = table[key]
        assert (w.k, w.n, w.M, w.G) == (t.k, t.n, t.M, t.G)
        assert (w.quantum_tree, w.classical_tree, w.classical_route) == (t.q_tree, t.c_tree, t.c_route)
        for name in ("Q", "C", "D", "trials", "prefix", "trial_cost"):
            assert math.isclose(getattr(w, name), getattr(t, name), rel_tol=1e-12, abs_tol=1e-12), (key, name)
        assert w.quantum_lower == bool(t.lower)
    table_pairs = {(r.source_id, r.date, r.axis, r.small_query, r.large_query): r.direction
                   for r in sp.itertuples(index=False)}
    our_pairs = {(p.source_id, p.date, p.axis, p.small_query, p.large_query): p.direction
                 for p in sealed_reproduction.pairs}
    assert our_pairs == table_pairs


def test_registered_windows_match_sealed_inputs():
    """``register_windows`` reproduces every sealed baseline window exactly."""
    root = env_dir("QRQ_SEALED_INPUTS")
    geometry = json.loads((root / "geometry_qa.json").read_text(encoding="utf-8"))
    assert len(geometry) == 14
    for source in geometry:
        ref = source["reference"]
        queries = json.loads((root / "queries" / f"{source['source_id']}.json").read_text(encoding="utf-8"))
        baseline = [q for q in queries if q.get("arm") == "baseline"]
        windows = register_windows(ref["x"], ref["y"])
        assert len(baseline) == len(windows) == 648
        for q, w in zip(baseline, windows):
            assert q["query_id"] == w.query_id
            assert q["bounds"] == list(w.box.bounds)
            assert tuple(q["anchor_index"]) == w.anchor_index
            assert q["status"] == w.status
