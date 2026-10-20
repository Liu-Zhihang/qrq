# Standard interfaces

qrq is organised as a chain of small interfaces. Each layer can be replaced
without touching the others, as long as the contract below is kept.

    records -> predicate -> index -> access plan -> routes -> prices -> engine
                                                                  \-> workload

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

`evaluate_window(indexes, box, prices=L1, verify=False) -> WindowResult`
evaluates every route on every index (`indexes` maps kind to index; all must
cover the same records). `WindowResult` carries `k`, `Q`, `C`,
`D = log2(Q / C)`, `quantum_tree`, `classical_tree`, `classical_route`, the
per-tree cost table and the full rows. With `verify=True` the answer size is
checked against brute force. Empty windows have `Q = C = D = None`.

`select_envelope(costs_by_tree)` is the selection rule on its own: the
quantum tree is the first minimal tree in iteration order; the classical
choice minimises `(cost, tree, route)`.

`qrq.amplitude` provides unknown-K randomized amplitude amplification as an
alternative quantum route (`amplified_cost`, `amplified_cost_from_route`);
it is priced with the `lookup` and `g_quantum` entries of a `PriceTable`
and requires `lam == repeats == 1`.

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

## 9. Semantic extension (`qrq.semantic`, experimental)

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
