# Roadmap

## Semantic range queries (future work)

The published results cover closed space-time boxes only. A natural
extension asks for records in a box that also satisfy a semantic condition
(land use, activity type, transport mode, ...). `qrq.semantic` fixes the
interfaces for that work but implements nothing used in any result:

* `SemanticEncoder` maps labels onto one ordered axis so that a label set
  becomes an interval;
* `CategoricalIntervalEncoder` is a toy encoder by list position;
* `SemanticBox` adds one closed semantic interval to a `SpaceTimeBox`.

Open items, roughly in order:

1. **Encoders.** Orders under which the label sets of real tasks are
   contiguous (hierarchical taxonomies, learned one-dimensional embeddings);
   handling of label sets that need several intervals.
2. **4-D index.** A static index over `(x, y, t, s)` with the same matched
   access plan as the 3-D backend (first-witness frontier, block and
   enumeration traversals, depth-first rank layout).
3. **Cost model.** A coherent semantic predicate price next to `g_quantum`,
   and the corresponding classical test price, so that `Q = P + (M/K) H`
   extends with an explicit semantic term.
4. **Workload.** A registered semantic window grid and adjacent pairs along
   the semantic axis, analogous to the space and time axes.

## Other directions

* Trajectory segments (the backend already supports segments; the published
  workload uses points only).
* Sweeps over the gate price, `lam` and `repeats`. `qrq.sweep` varies the
  coherent lookup and geometry prices only; the ledgers hold the full route
  rows, so the other prices could be re-priced per route.
* A three-tree amplitude-amplification envelope. The `amplified` option
  prices `QRQ_AA` on the first-witness quantum tree only, as published;
  choosing the quantum tree by the joint cost is not implemented.
* A command-line entry point for the price sweep and the task comparison.
