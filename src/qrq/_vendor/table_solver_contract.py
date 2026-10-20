"""Paid BLOCK/ENUM -> TABLE resource adapter; not a city-scale executor.

This reuses the legacy table accounting but requires k to come from a charged
exact classical solver row. No query result, city record or new oracle is built.
"""
import math

from .qrq_table_baseline import table_resources


def table_from_classical_solver(n, solver_row, write_units=4):
    solver = solver_row.get("route")
    if solver not in ("C_BLOCK", "C_ENUM"):
        raise ValueError("TABLE requires charged exact C_BLOCK or C_ENUM support, not a QRQ candidate domain")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("positive integer n required")
    k = solver_row.get("domain")
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= n:
        raise ValueError("k must be the certified classical solver domain")
    paid = solver_row.get("classical", {})
    for name in ("nodes", "slots", "geometry"):
        value = paid.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("missing/non-finite classical solver work: " + name)
    if not math.isfinite(write_units) or write_units < 4:
        raise ValueError("this contract includes four paid construction operations per hit")
    row = table_resources(n, k, solver_row, write_units=write_units)
    row["route"] = "C_TABLE_" + solver.removeprefix("C_")
    row["solver"] = solver
    row["k_source"] = "charged_exact_classical_solver_domain"
    row["execution_scope"] = "resource_formula_only_not_city_table_materialization"
    return row
