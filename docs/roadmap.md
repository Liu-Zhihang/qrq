# Roadmap

qrq is the shared core for quantum range queries over spatio-temporal and
semantic records from any domain (mobility traces, ocean drifters, wildlife
tracks, places): trajectories now, places (POIs) and their text next.
Specifications for the extensions are in `docs/specs/`; they are drafts and
produce no published result until implemented and tested. The reference
study of the accompanying manuscript is one application
(`docs/reference_study/`).

## Releases

**0.2.0 (this release).** Frozen reference for the accompanying manuscript.
Reproduces the reference study's access cohort, 45-point price sweep,
amplitude-amplification sensitivity and reconstruction-task comparison from
sealed evidence. Adds the draft specifications for records, semantic
predicates and plugins, and uses domain-neutral names (generic group
labels). No change to any published number.

**0.3.0, places and plugins.**

* Place records, sidecars and dataset manifests (`specs/records.md`).
* Loaders for Overture Places, Foursquare OS Places and OpenStreetMap,
  with per-row licence handling.
* Plugin discovery and the conformance suite (`specs/plugins.md`).
* Axis encoders for category taxonomies, multi-interval selections and
  validity intervals from opening-hours expressions.
* Command-line entry points for the price sweep and the task comparison.

**0.4.0, text semantics.**

* Embedding encoders for pinned transformer text models, with sidecar
  storage (`specs/semantic.md`, section 2.2).
* Mode II tested predicates: semantic test counts in the ledger and the two
  semantic prices in `PriceTable`.
* Registered semantic workloads with threshold and taxonomy ladders.

**0.5.0, circuit predicates and a 4-D index.**

* An optional adapter that turns compositional sentence circuits from a
  quantum natural-language-processing toolkit into circuit predicates with
  counted resources.
* Mode I: a static `(x, y, t, s)` index with the matched access plan of the
  3-D backend.

## Other directions

* Trajectory segments (the backend already supports segments; the published
  workload uses points only).
* Sweeps over the gate price, `lam` and `repeats`. `qrq.sweep` varies the
  coherent lookup and geometry prices only; the ledgers hold the full route
  rows, so the other prices could be re-priced per route.
* A three-tree amplitude-amplification envelope. The `amplified` option
  prices `QRQ_AA` on the first-witness quantum tree only, as published;
  choosing the quantum tree by the joint cost is not implemented.
