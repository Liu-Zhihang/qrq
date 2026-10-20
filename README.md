# qrq: quantum range queries over space-time point records

qrq is a small research package that asks a precise question: for a closed
space-time box query over a static index of point records, when does a
coherent (quantum) filtering route need fewer priced resources than the best
exact classical route on the same index?

The records can come from any domain: vehicle and pedestrian trajectories,
ocean drifters, wildlife tracking collars, mobile sensor campaigns or places
of interest. Sources may be grouped by any user-defined label (a region, a
campaign, a fleet, ...), and qrq reports results per source and per group.

For every window it

1. builds three static bulk-loaded indexes over the same records
   (`packed_rtree`, `tb_tree`, `kd_tree`, fan-out 4) with a compiled C++
   backend;
2. extracts a matched access plan: the first-witness frontier used by the
   quantum route and two exact classical traversals;
3. counts the resources of five matched-output routes (`QRQ_FW`, `C_BLOCK`,
   `C_ENUM`, `C_TABLE_BLOCK`, `C_TABLE_ENUM`);
4. prices them with an explicit price table and reports the envelope
   `Q` (best quantum), `C` (best classical) and `D = log2(Q / C)`.

qrq is a resource-accounting model with explicitly priced oracles. It does not
simulate large-scale quantum circuits, does not measure hardware run time and
does not claim a hardware speed-up.

## Install

Requirements: Python >= 3.10, a C++17 compiler, CMake >= 3.15.

```bash
pip install .            # builds the extension qrq._qrq_matched
pip install ".[test]"    # adds pytest and pandas
```

If the build tools are already present in the environment, `pip install
--no-build-isolation .` avoids downloading them.

## Quick example (synthetic data)

```python
import numpy as np
import qrq

rng = np.random.default_rng(7)
n = 20000
points = qrq.make_points(
    x=rng.normal(5000, 1500, n),
    y=rng.normal(5000, 1500, n),
    t=rng.uniform(0, 86400, n),
    entity=rng.integers(0, 200, n),
)

indexes = {kind: qrq.SpatioTemporalIndex.build(points, kind=kind) for kind in qrq.TREE_KINDS}
grid = qrq.WindowGrid(widths=(500, 1000, 2000), durations=(900, 3600, 21600),
                      centres=(43200,), anchor_step=3000.0)

for window in qrq.register_windows(cx=5000.0, cy=5000.0, grid=grid):
    if window.anchor_index != (1, 1):
        continue
    result = qrq.evaluate_window(indexes, window.box, qrq.L1, verify=True)
    if result.k:
        print(window.query_id, result.k, round(result.D, 3), result.quantum_tree,
              result.classical_tree, result.classical_route)
```

`verify=True` checks the index answer size against a brute-force count.

## Cost model in one paragraph

Each route has a classical resource vector (node reads, slot scans, exact
geometry tests, preparation metadata) and a quantum resource vector
(preparation gates, coherent record lookups, coherent geometry tests,
measurements, rank-register resets). With classical geometry price `g_c`,
lookup price `L`, coherent geometry price `g_q`, weight `lambda` and repeats
`r`, a route costs `classical + lambda * r * quantum`. For the quantum route
on a non-empty window this separates into

    Q = P + (M / K) H

where `M` is the candidate domain prepared at the first-witness frontier,
`K = k` the exact answer size, `M / K` the expected number of trials,
`H = lambda r (G + 2L + g_q + 1 + b)` the cost of one trial (`G` preparation
gates, `b` rank bits) and `P` the trial-independent prefix (frontier
discovery and metadata plus `lambda r (2L - b)`). The main published scenario
`L1` uses `g_c = 1, L = 100, g_q = 10000, lambda = 1, r = 1`.
`Q` is the minimum quantum cost over the three indexes, `C` the minimum over
the three indexes and the four classical routes. `D < 0` means the quantum
route is cheaper. See `docs/interfaces.md` for every interface.

## Sensitivity and task analyses

* **Amplitude amplification in the envelope.** `evaluate_window(...,
  amplified=True)` and `qrq.ledger.reproduce(..., amplified=True)` also admit
  unknown-K amplitude amplification (`QRQ_AA`, `qrq.amplitude`) on the
  quantum tree selected by the first-witness costs, `Q = min(Q_FW, Q_AA)`,
  as in the published sensitivity analysis.
* **Exact price sweep (`qrq.sweep`).** Re-prices the per-tree endpoints of a
  fixed cohort over a grid of coherent lookup and coherent geometry prices,
  reselects both tree minima at every point and summarises lower windows,
  median `Q/C`, forward and reverse pairs and their change relative to `L1`,
  pooled, per source and per group.
* **Reconstruction-task comparison (`qrq.tasks`).** For pairs of
  reconstruction windows, compares how often forward crossings and
  standardised noncrossing expansions worsen the reconstruction error, with
  a source-stratified date-block bootstrap.

```python
from qrq import sweep
from qrq.ledger import reproduce

cohort = sweep.cohort_from_reproduction(reproduce("/path/to/evidence"))
tables = sweep.sweep(cohort, sweep.price_grid())   # 45 points, L1 first
```

## Ecosystem and specifications

qrq is meant to be the shared core for quantum range queries over
spatio-temporal and, later, semantic records from any domain: mobility
traces, ocean drifters, wildlife tracks, places and their descriptions.
Version 0.2.0 is the frozen reference for the accompanying manuscript;
extensions are specified before they are implemented:

* `docs/specs/records.md`: point records (implemented) and place (POI)
  records with per-row licences, sidecars and dataset manifests;
* `docs/specs/semantic.md`: interval, embedding-similarity and circuit
  predicates, and how they enter `Q = P + (M/K) H` and the ledgers;
* `docs/specs/plugins.md`: entry-point groups for datasets, encoders,
  indexes, routes and price tables, with conformance rules.

The release plan is in `docs/roadmap.md`; contribution rules are in
`CONTRIBUTING.md` and changes in `CHANGELOG.md`.

## Reproducibility

* The C++ backend and the core Python resource code are vendored from the
  reference implementation with unchanged algorithms. Every vendored file,
  its original SHA-256 and every edit are listed in
  `provenance/vendored_sources.json`; the test suite checks these hashes.
* The synthetic test suite runs anywhere. The optional sealed regressions
  belong to the reference study of the accompanying manuscript, one
  application of qrq to mobility and environmental monitoring records. They
  re-price all 321 per-tree resource ledgers of its workload (107 source-days
  x 3 indexes) and reproduce the published counts: 13,370 non-empty baseline
  windows, 742 with `Q < C`, and 17,592 adjacent-scale pairs of which 1,060
  cross forward and 11 cross in reverse. Those ledgers derive from restricted
  record data and are not part of this repository; see
  `docs/reference_study/reproduction.md`.
* Further optional sealed tests reproduce the published price sweep (all 45
  points bit for bit; at unit prices 1,559 lower windows and 1,357 / 49
  crossings), the amplitude-amplification sensitivity (869 lower windows,
  1,211 / 13 crossings) and the reconstruction-task comparison (speed
  +0.0099 and monitoring +15.722 percentage points) from the sealed inputs.
* The public data sources of the reference study are listed in
  `docs/reference_study/data_access.md`.

## Repository layout

    cpp/                  C++ backend (vendored, byte-identical)
    src/qrq/              Python package
    src/qrq/_vendor/      vendored reference modules
    src/qrq/semantic/     experimental semantic-predicate interfaces
    provenance/           vendored-source manifest
    docs/                 interfaces and roadmap
    docs/specs/           draft specifications: records, semantic predicates, plugins
    docs/reference_study/ reproduction and data access of the reference study
    tests/                synthetic tests and the optional sealed regressions

## License

MIT, see `LICENSE`. Citation metadata is in `CITATION.cff`.
