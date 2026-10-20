# Standard interfaces

qrq is organised as a chain of small interfaces. Each layer can be replaced
without touching the others, as long as the contract below is kept.

    records -> predicate -> index -> access plan -> routes -> prices -> engine
                                                                  \-> workload
    sealed ledgers -> ledger (per-tree endpoints) -> sweep
    reconstruction-task pair tables -> tasks

## 1. Records (`qrq.records`)

A record is one observed point `(x, y, t)` of one entity.

| field            | dtype | meaning                                      |
|------------------|-------|----------------------------------------------|
| `x`              | `<f8` | projected easting, metres                    |
| `y`              | `<f8` | projected northing, metres                   |
| `t`              | `<f8` | seconds since the start of the index day     |
| `entity`         | `<i8` | opaque entity proxy                          |
| `source_ordinal` | `<u8` | strictly increasing row ordinal in a source  |

`POINT` is this 40-byte structured dtype. Files are flat little-endian
arrays of `POINT` (`load_points`, `save_points`). `validate_points` checks the
dtype and finiteness, and optionally the day clock (`0 <= t < 86400`) and a
strictly increasing `source_ordinal`. `backend_arrays` returns the exact
`(N, 3)` float64 coordinates and `(N,)` int64 entity ids the backend expects.

## 2. Predicates (`qrq.geometry`)

`Predicate` is a protocol with one method, `contains(points) -> bool mask`.
`SpaceTimeBox(xmin, xmax, ymin, ymax, tmin, tmax)` is the closed box used for
all published results. Bounds must be finite and ordered; zero extent is
allowed. `SpaceTimeBox.count(points)` is the independent brute-force answer
size every index result can be checked against.

## 3. Index (`qrq.index`)

`SpatioTemporalIndex.build(points, ids=None, kind="packed_rtree", fanout=4)`
bulk-loads one of `TREE_KINDS = ("packed_rtree", "tb_tree", "kd_tree")` over
the records (points are zero-length segments in the backend). The index is
static; there is no insertion or deletion.

`index.plan(box) -> AccessPlan` mirrors the backend `matched_query` result:

| field                        | meaning                                                         |
|------------------------------|-----------------------------------------------------------------|
| `candidate_ranges`           | rank intervals of the first-witness frontier (quantum domain)   |
| `prefix_nodes/slots/geometry`| classical work spent discovering the frontier                   |
| `frontier_size`, `height`    | frontier node count and node height                             |
| `depth`, `witness_kind`      | frontier depth (`-1` if empty) and `contained_subtree`, `leaf_segment` or `none` |
| `block`                      | exact traversal stopping at contained subtrees (`ClassicalAccess`) |
| `enumeration`                | exact traversal testing every reachable record (`ClassicalAccess`) |

Ranks refer to a static depth-first layout of the records
(`index.rank_to_record()`). `AccessPlan.to_matched()` returns the raw
dictionary form.

## 4. Routes (`qrq.routes`)

`AccessRoute` is one of `QRQ_FW`, `C_BLOCK`, `C_ENUM`, `C_TABLE_BLOCK`,
`C_TABLE_ENUM`. `evaluate_routes(plan, n) -> (k, rows)` returns the exact
answer size and one resource row per route, in that order. Each row holds

* `classical`: `nodes`, `slots`, `geometry`, `metadata`;
* `quantum`: `prepare_gate_total`, `lookup_calls`, `geometry_calls`,
  `measurements`, `reset_rank_bits`;
* `domain` (prepared rank domain; `M` for `QRQ_FW`, `k` otherwise),
  `expected_trials` (`M/k` for `QRQ_FW`), `shared_initial_rank_bits` and a
  `preparation` summary (gate counts, merged ranges, ...).

Invariants: the classical domains equal `k`; the `QRQ_FW` domain is at least
`k` when `k > 0`; all domains are zero when `k == 0`.

## 5. Prices (`qrq.prices`)

`PriceTable(g_classical, lookup, g_quantum, lam, repeats)`; presets `UNIT`
(all ones) and `L1` (`1, 100, 10000, 1, 1`, the main published scenario).
`route_cost(row, prices)` is

    nodes + slots + g_classical * geometry + metadata
      + lam * repeats * (prepare_gate_total + lookup * lookup_calls
                         + g_quantum * geometry_calls + measurements + reset_rank_bits)

`decompose_quantum_cost(row, prices)` returns `P`, `M/K` and `H` with
`Q = P + (M/K) H` for a non-empty `QRQ_FW` row.

## 6. Engine (`qrq.engine`)

`evaluate_window(indexes, box, prices=L1, verify=False, amplified=False) ->
WindowResult` evaluates every route on every index (`indexes` maps kind to
index; all must cover the same records). `WindowResult` carries `k`, `Q`,
`C`, `D = log2(Q / C)`, `quantum_tree`, `classical_tree`,
`classical_route`, `quantum_route`, `filter_cost`, `amplified_cost`, the
per-tree cost table and the full rows. With `verify=True` the answer size is
checked against brute force. Empty windows have `Q = C = D = None`.

`select_envelope(costs_by_tree) -> Envelope` is the selection rule on its
own: the quantum tree is the first minimal tree in iteration order; the
classical choice minimises `(cost, tree, route)`. `Envelope` holds `Q`, `C`,
the selected trees and classical route, `quantum_route` (`QRQ_FW`),
`filter_cost` (the `QRQ_FW` cost on the quantum tree) and `amplified_cost`
(`None`).

`qrq.amplitude` provides unknown-K randomized amplitude amplification as an
alternative quantum route `QRQ_AA` (`amplified_cost`,
`amplified_cost_from_route`); it is priced with the `lookup` and
`g_quantum` entries of a `PriceTable` and requires `lam == repeats == 1`.
Its inputs are the candidate domain `M`, the answer size `K`, the
preparation gate count `G`, the rank bits `b` and the classical prefix `P0`
of the `QRQ_FW` row; all of them are in every route row and ledger line.

`include_amplified(env, amplified_cost) -> Envelope` admits that route into a
first-witness envelope exactly as in the published sensitivity analysis:
`amplified_cost` is the `QRQ_AA` cost on `env.quantum_tree` (the tree is
not re-optimised), `Q = min(Q_FW, Q_AA)`, `quantum_route` becomes `QRQ_AA`
only when it is strictly cheaper, and the classical side is unchanged.
`evaluate_window(..., amplified=True)` applies it to every non-empty
window.

## 7. Workload (`qrq.workload`)

`register_windows(cx, cy, widths, durations, centres, anchor_step)` returns
the registered baseline windows of one source-day (648 with defaults):

* identifier `g{ix}{iy}_c{tc}_w{w}_d{d}`;
* anchor `x = cx + (ix - 1) * 20000/3`, `y = cy + (iy - 1) * 20000/3`;
* bounds `[x - w/2, x + w/2, y - w/2, y + w/2, tc - d/2, tc + d/2]`;
* status `unsupported_index_span` when the time span leaves `[0, 86400]`,
  otherwise `scheduled`.

`adjacent_pairs(items)` lists pairs that change exactly one extent to the
next grid value (`space`: width, `time`: duration) with the anchor, temporal
centre and other extent fixed; missing windows are never skipped over.
`classify_pair(d_small, d_large)` returns `forward` (`D_small > 0` and
`D_large < 0`), `reverse` (`D_small < 0` and `D_large > 0`) or `no_cross`.

## 8. Sealed ledgers (`qrq.ledger`)

`find_ledger_days`, `read_ledger`, `audit_day` and `reproduce` re-price the
sealed per-tree ledgers and rebuild the envelope counts; see
`docs/reproducing.md`.

* `reproduce(root, prices=L1, arm="baseline", variant="base",
  rel_tol=1e-12, amplified=False) -> Reproduction` with `windows`
  (`LedgerEndpoint`), `pairs` (`PairRecord`), `prices` and `amplified`.
  `amplified=True` uses `include_amplified` on every window.
* `LedgerEndpoint` carries the envelope fields above plus `trees`, one
  `TreeEndpoint` per tree in ledger order: `M`, `G`, `bits`, `P0` (classical
  prefix of `QRQ_FW`), `Q` (`QRQ_FW` cost), `C` (cheapest classical route)
  and `classical_lookups` (coherent lookups shared by the four classical
  routes, `nan` if they differ or use coherent geometry).
* `python -m qrq.ledger --ledgers <root> [--amplified] [--windows-csv out]`.

## 9. Price sweep (`qrq.sweep`)

Re-prices a fixed cohort of non-empty windows when only the coherent lookup
price `L` and the coherent geometry price `g_q` change (classical prices,
gate price, `lam` and `repeats` fixed; `lam == repeats == 1`). Per tree,

    Q' = P0 + (M/K) (G + 2L + g_q + 1 + b) + 2L - b
    C' = C - n_L (L_ref - L)

with `n_L = classical_lookups` (2 in the published workload, so the
classical arg-min route is unchanged). Both tree minima are reselected at
every point with the envelope tie rule and pairs are re-classified with
`D' = log2(Q'/C')`. The evaluation order matches the sealed sweep, so its
tables are reproduced bit for bit; against `route_cost` the closed form
agrees within `1e-12` relative error.

* `price_grid(lookups=LOOKUP_PRICES, geometries=GEOMETRY_PRICES, base=L1)`:
  the 5 x 9 published grid, lookup outer; `REPORT_POINTS` lists the four
  reported points.
* `SweepCohort`: window arrays (`keys`, `city`, `k`, `width_m`,
  `duration_s`), per-tree arrays (`M`, `G`, `bits`, `P0`, `C`,
  `classical_lookups`), pair indices `small`, `large` and `axis`, and the
  `reference` prices of `C`. Build it with
  `cohort_from_reproduction(reproduction, city_of=None)` or
  `cohort_from_endpoints(windows, pairs, city_of, reference, trees)`.
  `city_of` maps source to city; without it each source is its own city.
* `reprice(cohort, prices) -> PricePoint`: per-tree and selected `Q`, `C`,
  `D`, selected tree indices, `forward`/`reverse` pair flags and the
  structure share `(M/K) G / Q`, prefix share `P0 / Q`, `M/K` and `G/K` on
  the selected quantum tree.
* `sweep(cohort, grid, reference=None, on_point=None) -> SweepTables` with
  `grid` (one pooled row per point), `by_source` (point x source) and
  `pairs` (point x source incl. `ALL` x axis incl. `all`). Row keys are the
  column names of the published tables: `eligible_windows`,
  `lower_windows`, `q_over_c_median`; `15min_*`, `width_le_1km_*` and
  `K_ge_20000_*` (`_windows`, `_lower`, `_ratio_median`); quartiles
  `*_q25/_median/_q75` of `structure_share`, `prefix_share`, `M_over_K`,
  `G_over_K`, `C_over_K`; `forward_pairs`, `reverse_pairs`,
  `eligible_pairs`, `sources_with_forward`, `cities_with_forward`,
  `forward_retained`, `forward_added`, `forward_lost` (relative to the
  forward pairs at `reference`, default the cohort reference `L1`) and the
  tree tie counts. Quantiles are linear (`numpy.quantile`).

## 10. Reconstruction-task comparison (`qrq.tasks`)

Inputs are generic pair tables: one `TaskPair(family, source, date, axis,
small, large, fixed, supported, small_D, large_D, delta_error)` per
registered expansion, where `small`/`large` are the extents along `axis`
(`space`: width, `time`: duration), `fixed` the other extent and
`delta_error` the large minus small absolute error.
`pair_from_endpoints(family, source, date, axis, small, large, delta)` builds
one from two endpoint records with `width_m`, `duration_s`, `k`, `D` and
`absolute_error` (supported: both `k > 0`, both `D` and errors finite).

* Classification: `crossing_direction` (`forward`: `D_small > 0 > D_large`,
  `reverse`, `other`), `crossing_category` (`forward`, `reverse`,
  `classical`, `quantum`, `boundary`), `error_direction` (`up`, `down`,
  `tie`) and `worsening_rate(up, down)` (ties excluded).
* `date_stratum_counts(pairs, dates, strata)`: `(dates, strata, 5)` counts of
  forward up/down, noncrossing up/down (non-tied only) and forward ties;
  a stratum is `(axis, small, large, fixed)` within one source.
* `standardised_estimates(counts)`: worsened forward pairs, the
  noncrossing worsening rates weighted by non-tied forward counts, the used
  forward count and the forward pairs of strata without controls; only
  strata with both kinds enter.
* `compare_crossings(pairs, registry, family_order=None, draws=5000,
  seed=20260924, margin=0.05) -> TaskComparison`. `registry` maps
  `(family, source)` to its registered dates (the bootstrap blocks).
  Families consume one PCG64 stream in `family_order` (default: registry
  order), sources in sorted order; each draw resamples all registered dates
  of a source with replacement, keeps all pairs of a drawn date with its
  multiplicity and recomputes comparable strata; undefined draws stay
  undefined. `SourceComparison` and `FamilyComparison` give point rates,
  per-draw arrays, percentile intervals (2.5 %, 97.5 %) and `row()`;
  `TaskComparison` has `source_rows()`, `family_rows()` and
  `stratum_rows()`. Family point rates are forward-count weighted means of
  the source rates; family draws pool the source draw columns.
* `operational_label(delta, interval, margin)`: `NO_COMPARABLE_FORWARD`,
  `NO_INFORMATION_OPERATIONAL` (interval contains 0 or `|delta| < margin`),
  `CROSSING_WORSE` or `CROSSING_BETTER`.
* `category_summary(pairs)`: counts, tie-excluded worsening rate and mean
  error change (`compensated_mean`) per family, source and category.

## 11. Semantic extension (`qrq.semantic`, experimental)

Not used for any published result.

* `SemanticEncoder` protocol: `encode(labels) -> float array` and
  `interval(labels) -> (lo, hi)` for a label set that is contiguous on the
  encoded axis.
* `CategoricalIntervalEncoder(categories)`: toy encoder by list position.
* `SemanticBox(SpaceTimeBox)`: adds a closed interval `[smin, smax]` on the
  encoded axis; `spacetime()` returns the 3-D part for the existing indexes
  and `contains(points, semantic)` filters the semantic interval afterwards.

A native 4-D index and a matched cost model for semantic predicates are
future work (`docs/roadmap.md`).
