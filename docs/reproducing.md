# Reproducing the published counts

## What is reproduced

The published workload covers 14 sources in 13 cities, 107 source-days and
three indexes per day. Each source-day registers 648 baseline windows
(69,336 in total). The main scenario `L1` gives

| quantity                                                  | value  |
|-----------------------------------------------------------|--------|
| non-empty, fully supported baseline windows               | 13,370 |
| windows with `Q < C`                                      | 742    |
| adjacent-scale pairs of non-empty baseline windows        | 17,592 |
| forward crossings (`D_small > 0`, `D_large < 0`)          | 1,060  |
| reverse crossings (`D_small < 0`, `D_large > 0`)          | 11     |

## Which data are restricted

The record files are derived from licensed or access-controlled mobility
data and stay on the data-owning machines. They are not distributed. The
following sealed artefacts are derived from them and are also not part of
this repository; access to them is governed by the terms of the underlying
sources:

* **per-tree ledgers**, 321 files
  `<root>/<host>/outputs/<source>/<date>/<tree>.jsonl.gz`. Each line holds
  the window metadata (no bounds), `result_status`, `k`, the five route rows
  with their classical and quantum resource vectors, and the stored costs.
  They contain no coordinates of any record.
* **tables**: `endpoints.csv.gz` (one row per window and comparator) and
  `scale_pairs.csv.gz` (one row per adjacent-scale pair).
* **inputs**: `geometry_qa.json` (city reference points in the metric CRS)
  and `queries/<source>.json` (registered windows with bounds).

The public sources the records were drawn from are listed in
`docs/data_access.md`.

## Steps

1. Install the package and the test extras:

   ```bash
   pip install ".[test]"
   ```

2. Run the synthetic suite (no data needed):

   ```bash
   pytest
   ```

3. Point the environment variables to the sealed artefacts and run the
   regression:

   ```bash
   QRQ_SEALED_LEDGERS=/path/to/evidence \
   QRQ_SEALED_TABLES=/path/to/tables \
   QRQ_SEALED_INPUTS=/path/to/inputs \
   pytest tests/test_sealed_regression.py -v
   ```

   * `test_ledger_costs_and_envelope` (needs the ledgers) recomputes the
     `L1` cost of every route on every ledger line from the stored resource
     vectors and compares it with the stored cost (relative tolerance
     `1e-12`; in practice every value is bit-identical). It then rebuilds
     every window envelope and the adjacent-scale pairs and checks the five
     published totals and the per-source table.
   * `test_published_counts_in_tables` (needs the tables) checks that the
     tables state the published counts and that every stored pair direction
     equals `classify_pair(small_D, large_D)`.
   * `test_envelope_matches_tables` (needs both) compares every recomputed
     window (`k`, `M`, `G`, `Q`, `C`, `D`, selected trees and route, the
     `P + (M/K) H` decomposition) and every pair with the tables.
   * `test_registered_windows_match_sealed_inputs` (needs the inputs)
     checks that `register_windows` reproduces all 648 baseline windows of
     every source exactly, including floating-point bounds.

4. Alternatively print the reproduced counts directly:

   ```bash
   python -m qrq.ledger --ledgers /path/to/evidence
   ```

## Optional backend cross-check

`tests/test_reference_backend.py` compares the packaged extension with a
copy of the backend compiled separately using the reference command
`g++ -O2 -shared -std=c++17 -fPIC -fvisibility=hidden -I<python include>
-I<pybind11 include> trajectory_trees_3d.cpp -o _qrq_matched<suffix>`; set
`QRQ_REFERENCE_EXTENSION` to that library.

## Scope

The ledgers were produced by evaluating the indexes on the full record days
and checking every answer size against an independent count. Re-running the
indexes on the records requires the restricted data; with the sealed ledgers
the cost model, the envelope rule and the pairing rule are reproduced
exactly, and the index code is checked on synthetic data.
