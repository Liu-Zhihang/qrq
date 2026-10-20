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

### Price sweep (`qrq.sweep`)

The same 13,370 windows and 17,592 pairs re-priced at 45 points (coherent
lookup `L` in {100, 30, 10, 3, 1} x coherent geometry `g_q` in {10000, 3000,
1000, 300, 100, 30, 10, 3, 1}; classical prices, gate price and repetitions
fixed at 1), with both tree minima reselected at every point:

| `L` / `g_q`   | `Q < C` windows | median `Q/C` | forward / reverse |
|---------------|-----------------|--------------|-------------------|
| 100 / 10000   | 742             | 33.13        | 1,060 / 11        |
| 10 / 1000     | 1,253           | 19.29        | 1,348 / 28        |
| 1 / 100       | 1,514           | 16.89        | 1,368 / 46        |
| 1 / 1         | 1,559           | 16.60        | 1,357 / 49        |

At unit prices 907 of the 1,060 `L1` forward pairs are retained, 450 are
added and 153 are lost; 49 of 2,917 fifteen-minute and 47 of 5,429
`<= 1 km` windows have `Q < C`. All 45 rows of the grid table, the 630
source rows and the 2,025 source x axis pair rows are reproduced bit for
bit.

### Amplitude amplification in the envelope

Admitting unknown-K amplitude amplification on the originally selected
quantum tree (`Q = min(Q_FW, Q_AA)`, classical side unchanged) gives 869
windows with `Q < C`, 1,211 forward and 13 reverse pairs, forward pairs in
11 of 13 cities (12 sources); 973 of the original 1,060 forward pairs stay
forward and 87 become quantum-favourable at both endpoints; no 15 min
window and 3 `<= 1 km` windows become favourable. The AA inputs (`P0`, `G`,
rank bits, `M`, `K`) are read from the `QRQ_FW` row of every ledger line,
so no further sealed table is needed for these counts.

### Reconstruction-task comparison (`qrq.tasks`)

Worsening rates of forward crossings against noncrossing expansions
standardised within source x axis x smaller x larger x fixed extent, with a
5,000-draw source-stratified date-block bootstrap (PCG64, seed 20260924):

| family     | forward | forward worse | standardised control | difference (pp) | 95 % interval (pp) |
|------------|---------|---------------|----------------------|-----------------|--------------------|
| speed      | 1,242   | 56.36 %       | 56.35 %              | +0.0099         | -4.416 to 4.546    |
| monitoring | 242     | 55.37 %       | 39.65 %              | +15.722         | 1.308 to 28.287    |

The family, source (13 rows), stratum (136 rows) and category (46 rows)
tables are reproduced exactly.

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
  `scale_pairs.csv.gz` (one row per adjacent-scale pair). The ledgers carry
  no city label; the sealed tests read the source-to-city mapping from the
  `city` column of `endpoints.csv.gz`.
* **inputs**: `geometry_qa.json` (city reference points in the metric CRS)
  and `queries/<source>.json` (registered windows with bounds).
* **sweep tables**: `price_grid.csv`, `price_by_source.csv` and
  `price_pairs.csv` of the sealed price sweep.
* **AA tables**: `aa_endpoints.csv.gz` (one row per window and comparator
  with `P0`, `G`, rank bits, `Q_aa`, `Q_joint`, `D_joint`) and
  `aa_by_source.csv` of the sealed amplitude-amplification analysis.
* **task archive**: the sealed archive root holding, as run directories,
  the four input runs and the comparison run listed in
  `tests/task_inputs.py`:

  | run label (relative to the archive root)                    | files used                                       |
  |-------------------------------------------------------------|--------------------------------------------------|
  | `cities_speed_reconstruction_stage2_20260923_v1/tables`     | `pairs.csv.gz`, `endpoints.csv.gz`               |
  | `cities_jinan_e1_20260921_v1r1/tables`                      | `scale_pairs.csv`, `endpoints.csv`               |
  | `cities_dublin_e2_20260922_v1/tables`                       | `scale_pairs.csv`, `endpoints.csv`               |
  | `cities_monitoring_e3e4_20260923_v1/aggregation_r1/tables`  | `task_scale_pairs.csv.gz`, `task_endpoints.csv`  |
  | `cities_crossing_information_20260924_v1/tables`            | `family_summary.csv`, `by_task_source.csv`, `strata.csv`, `category_summary.csv` |

  The speed run is one family with nine sources; the other three runs form
  the monitoring family (four task sources). `tests/task_inputs.py`
  normalises their layouts into `qrq.tasks.TaskPair` records: two tables
  have no source column (labels `jinan`, `dublin`), one has no axis column
  (space pairs only), one has no query identifiers (endpoints matched by
  target, width and duration) and one holds several metrics (only `D` is
  used). Registered dates are the dates of each endpoint table.

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

4. Run the sensitivity and task regressions:

   ```bash
   QRQ_SEALED_LEDGERS=/path/to/evidence \
   QRQ_SEALED_TABLES=/path/to/tables \
   QRQ_SEALED_SWEEP=/path/to/sweep/tables \
   QRQ_SEALED_AA=/path/to/aa/tables \
   QRQ_SEALED_TASKS=/path/to/archive \
   pytest tests/test_sealed_sweep.py tests/test_sealed_amplified.py tests/test_sealed_tasks.py -v
   ```

   * `test_sealed_sweep.py`: at `L1` the closed form equals every ledger
     cost within `1e-12` and the published pair directions exactly (ledgers);
     the four reported points and the unit-price retained/added/lost counts
     (ledgers); all three sweep tables bit for bit, including 12 sources and
     11 cities with forward pairs at `L1` (ledgers, tables, sweep tables).
   * `test_sealed_amplified.py`: the AA-inclusive counts above, 11,685
     windows where AA is cheaper than first-witness filtering and the
     AA-only directions 1,200 / 11 (ledgers); 11 of 13 cities (ledgers,
     tables); every window's `P0`, `G`, rank bits and `Q_aa` bit for bit and
     the per-source table exactly (ledgers, AA tables).
   * `test_sealed_tasks.py`: the four comparison tables and the published
     rounded readouts (task archive).

5. Alternatively print the reproduced counts directly:

   ```bash
   python -m qrq.ledger --ledgers /path/to/evidence [--amplified]
   ```

## Numerical notes

* The sweep evaluates `Q'` in closed form in the order of the sealed
  analysis; this is what makes its tables bit-identical. The closed form
  differs from the route cost of `qrq.prices.route_cost` by rounding only
  (at most a few units in the last place at `L1`).
* The sealed AA analysis read the first-witness costs with a non-round-trip
  CSV parser, so its stored `Q_joint` and `D_joint` differ from the exact
  values by one unit in the last place in some rows where the first-witness
  route wins. qrq uses the exact ledger costs; the AA costs and every sign,
  direction and count agree exactly.
* Category means of the task comparison use Kahan-compensated summation in
  pair order, as the grouped means of the sealed analysis did.
* Not ported from the task comparison: within-stratum Spearman correlations
  and the exploratory unstandardised interval of its design notes; neither
  is part of the published readouts.

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
exactly, and the index code is checked on synthetic data. The price sweep
varies only the coherent lookup and geometry prices; the amplified envelope
prices AA on the first-witness quantum tree only (no three-tree AA
re-optimisation); the task comparison is a descriptive date-resampling
association, not a causal or equivalence test.
