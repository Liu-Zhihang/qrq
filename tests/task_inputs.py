"""Normalise the sealed reconstruction-task inputs into ``TaskPair`` records.

The published comparison run lists, in its sealed ``inputs/freeze.json``,
exactly the pair and endpoint tables it read, in reading order and with
their SHA-256. Those paths are run labels relative to the sealed archive
root passed in ``QRQ_SEALED_TASKS``; nothing here is a file-system location.

Table layouts differ: single-source runs carry no source column (their
source identifier is the second field of the run label, see
:func:`run_source`), one table carries no axis column (all its pairs expand
space), one table carries no query identifiers (its endpoints are matched by
target, width and duration), and one pair table holds several cost metrics
of which only the main envelope ``D`` is used.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from qrq.tasks import TaskPair, pair_from_endpoints

COMPARISON_RUN = "cities_crossing_information_20260924_v1"
RESULTS = f"{COMPARISON_RUN}/tables"
FREEZE = f"{COMPARISON_RUN}/inputs/freeze.json"
FAMILY_ORDER = ("speed", "monitoring")
FAMILY_BY_RUN = {"cities_speed_reconstruction_stage2_20260923_v1": "speed"}
"""Task family of the input runs; every other input run is a monitoring task."""
DEFAULT_FAMILY = "monitoring"


def run_source(root: Path, run: str) -> str:
    """Source identifier of a single-source run.

    The sealed archive labels runs ``<prefix>_<source>_<stage>_<date>_<version>``;
    when the run's ``config.json`` names its source, the two must agree.
    """
    source = run.split("_")[1]
    config = root / run / "config.json"
    if config.is_file():
        stated = json.loads(config.read_text(encoding="utf-8")).get("inputs", {}).get("source_id")
        assert stated in (None, source), (run, stated)
    return source


def input_tables(root: Path) -> list[tuple[str, str, Path, Path]]:
    """``(run, family, pair table, endpoint table)`` in the comparison's reading order."""
    freeze = json.loads((root / FREEZE).read_text(encoding="utf-8"))
    by_directory: dict[str, dict[str, str]] = {}
    for name, meta in freeze["inputs"].items():
        if "/" not in name:
            continue
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == meta["sha256"], name
        kind = "pairs" if "pairs" in Path(name).name else "endpoints"
        by_directory.setdefault(Path(name).parent.as_posix(), {})[kind] = name
    tables = []
    for directory, files in by_directory.items():
        run = directory.split("/")[0]
        tables.append((run, FAMILY_BY_RUN.get(run, DEFAULT_FAMILY), root / files["pairs"], root / files["endpoints"]))
    return tables


def _with_source(frame: Any, label: str | None) -> Any:
    frame = frame.copy()
    if "source_id" in frame:
        frame = frame.rename(columns={"source_id": "source"})
    elif "source" not in frame:
        frame["source"] = label
    return frame


def load_task_inputs(root: Path) -> tuple[list[TaskPair], dict[tuple[str, str], list[str]]]:
    """Return all normalised pairs and the registered dates of every task source."""
    pd = pytest.importorskip("pandas")
    pairs: list[TaskPair] = []
    registry: dict[tuple[str, str], list[str]] = {}
    for run, family, pair_file, endpoint_file in input_tables(root):
        table = pd.read_csv(pair_file)
        endpoints = pd.read_csv(endpoint_file)
        if "metric" in table:
            table = table[table.metric.eq("D")]
        has_source = bool({"source", "source_id"} & (set(table) | set(endpoints)))
        label = None if has_source else run_source(root, run)
        table, endpoints = _with_source(table, label), _with_source(endpoints, label)
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
