"""Normalise the sealed reconstruction-task inputs into ``TaskPair`` records.

The published comparison reads four pair tables with their endpoint tables.
Paths are run labels relative to the sealed archive root passed in
``QRQ_SEALED_TASKS``; nothing here is a file-system location.

Table layouts differ: two tables carry no source column (their source label
is given below), one table carries no axis column (all its pairs expand
space), one table carries no query identifiers (its endpoints are matched by
target, width and duration), and one pair table holds several cost metrics
of which only the main envelope ``D`` is used.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from qrq.tasks import TaskPair, pair_from_endpoints

INPUTS: tuple[tuple[str, str, str, str | None], ...] = (
    ("speed",
     "cities_speed_reconstruction_stage2_20260923_v1/tables/pairs.csv.gz",
     "cities_speed_reconstruction_stage2_20260923_v1/tables/endpoints.csv.gz", None),
    ("monitoring",
     "cities_jinan_e1_20260921_v1r1/tables/scale_pairs.csv",
     "cities_jinan_e1_20260921_v1r1/tables/endpoints.csv", "jinan"),
    ("monitoring",
     "cities_dublin_e2_20260922_v1/tables/scale_pairs.csv",
     "cities_dublin_e2_20260922_v1/tables/endpoints.csv", "dublin"),
    ("monitoring",
     "cities_monitoring_e3e4_20260923_v1/aggregation_r1/tables/task_scale_pairs.csv.gz",
     "cities_monitoring_e3e4_20260923_v1/aggregation_r1/tables/task_endpoints.csv", None),
)
"""``(family, pair table, endpoint table, source label)`` in published order."""

RESULTS = "cities_crossing_information_20260924_v1/tables"
FAMILY_ORDER = ("speed", "monitoring")


def _with_source(frame: Any, label: str | None) -> Any:
    frame = frame.copy()
    if label is not None:
        frame["source"] = label
    elif "source_id" in frame:
        frame = frame.rename(columns={"source_id": "source"})
    return frame


def load_task_inputs(root: Path) -> tuple[list[TaskPair], dict[tuple[str, str], list[str]]]:
    """Return all normalised pairs and the registered dates of every task source."""
    pd = pytest.importorskip("pandas")
    pairs: list[TaskPair] = []
    registry: dict[tuple[str, str], list[str]] = {}
    for family, pair_file, endpoint_file, label in INPUTS:
        table = pd.read_csv(root / pair_file)
        if "metric" in table:
            table = table[table.metric.eq("D")]
        table = _with_source(table, label)
        endpoints = _with_source(pd.read_csv(root / endpoint_file), label)
        records = endpoints.to_dict("records")
        by_query = {(r["source"], r["date"], int(r["target_index"]), r["query_id"]): r for r in records}
        by_extent = {(r["source"], r["date"], int(r["target_index"]), int(r["width_m"]), int(r["duration_s"])): r
                     for r in records}
        assert len(by_query) == len(by_extent) == len(records)
        for source, group in endpoints.groupby("source"):
            registry[(family, source)] = sorted(group.date.unique().tolist())
        for p in table.to_dict("records"):
            source, date, target = p["source"], p["date"], int(p["target_index"])
            axis = p.get("axis", "space")
            if "small_query" in p:
                small = by_query[(source, date, target, p["small_query"])]
                large = by_query[(source, date, target, p["large_query"])]
            else:
                small = by_extent[(source, date, target, int(p["small"]), int(p["duration_s"]))]
                large = by_extent[(source, date, target, int(p["large"]), int(p["duration_s"]))]
            delta = p["delta_error"] if "delta_error" in p else p["delta_absolute_error"]
            pair = pair_from_endpoints(family, source, date, axis, small, large, delta)
            if pair.supported:
                expected = large["absolute_error"] - small["absolute_error"]
                assert math.isclose(pair.delta_error, expected, rel_tol=1e-9, abs_tol=1e-10)
            pairs.append(pair)
    return pairs, registry
