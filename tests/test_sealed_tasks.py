"""Reconstruction-task comparison against the sealed tables (skipped by default).

Needs ``QRQ_SEALED_TASKS``: the sealed archive root holding the comparison
run and the input runs listed in its ``inputs/freeze.json``
(``tests/task_inputs.py``).
"""
from __future__ import annotations

import pytest

from qrq import tasks

from evidence import assert_rows_equal, env_dir, read_csv
from task_inputs import FAMILY_ORDER, RESULTS, load_task_inputs


@pytest.fixture(scope="module")
def sealed_tasks():
    root = env_dir("QRQ_SEALED_TASKS")
    pairs, registry = load_task_inputs(root)
    return root, pairs, tasks.compare_crossings(pairs, registry, FAMILY_ORDER)


def test_task_tables_match_sealed(sealed_tasks):
    """Family, source, stratum and category tables are reproduced exactly."""
    root, pairs, result = sealed_tasks
    tables = root / RESULTS
    assert_rows_equal(read_csv(tables / "family_summary.csv"), result.family_rows())
    assert_rows_equal(read_csv(tables / "by_task_source.csv"), result.source_rows())
    assert_rows_equal(read_csv(tables / "strata.csv"), result.stratum_rows())
    assert_rows_equal(read_csv(tables / "category_summary.csv"), tasks.category_summary(pairs))


def test_published_family_readouts(sealed_tasks):
    root, pairs, result = sealed_tasks
    speed, monitoring = (f.row() for f in result.families)
    pct = lambda v: round(100 * v, 2)
    pp = lambda v: round(100 * v, 4)
    assert (speed["sources"], speed["source_days"], speed["forward_n"]) == (9, 65, 1242)
    assert (pct(speed["forward_worse_rate"]), pct(speed["standardised_control_worse_rate"])) == (56.36, 56.35)
    assert (pp(speed["delta"]), round(100 * speed["delta_ci_low"], 3), round(100 * speed["delta_ci_high"], 3)) == (
        0.0099, -4.416, 4.546)
    assert (monitoring["sources"], monitoring["source_days"], monitoring["forward_n"]) == (4, 26, 242)
    assert (pct(monitoring["forward_worse_rate"]), pct(monitoring["standardised_control_worse_rate"])) == (
        55.37, 39.65)
    assert (round(100 * monitoring["delta"], 3), round(100 * monitoring["delta_ci_low"], 3),
            round(100 * monitoring["delta_ci_high"], 3)) == (15.722, 1.308, 28.287)
    assert (speed["operational_label"], monitoring["operational_label"]) == (
        tasks.NO_INFORMATION, tasks.CROSSING_WORSE)
    counts = {family: sum(p.family == family for p in pairs) for family in FAMILY_ORDER}
    assert counts == {"speed": 15432, "monitoring": 3552}
    assert sum(p.supported for p in pairs if p.family == "speed") == 12912
