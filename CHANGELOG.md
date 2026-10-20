# Changelog

All notable changes are listed here. Versions follow semantic versioning;
within 0.x the stable core is defined in `docs/specs/plugins.md`, section 6.

## 0.2.0 - 2026-10-20

Reference release for the accompanying manuscript. No published number
changes relative to 0.1.0.

### Added

* `qrq.sweep`: exact repricing of the per-tree endpoints over a price grid;
  reproduces the 45-point coherent-lookup by quantum-geometry sweep.
* `amplified` option in `qrq.engine` and `qrq.ledger`: the envelope with the
  priced amplitude-amplification route, as published.
* `qrq.tasks`: crossing versus standardised noncrossing comparison with the
  source-stratified date-block bootstrap.
* Per-tree endpoints in ledger reproduction.
* Draft specifications in `docs/specs/`: record schema (points and places),
  semantic predicates and costs, plugins.
* `CHANGELOG.md` and `CONTRIBUTING.md`.

### Changed

* A generic `group` label (any user-defined grouping of sources: region,
  campaign, fleet, ...) replaces the domain-specific names of the reference
  study in `qrq.sweep`: `SweepCohort.group`, `group_of`,
  `groups_with_forward` and the per-source row key `group`.
* The reproduction guide and the data-access list of the reference study
  moved to `docs/reference_study/` (`reproduction.md`, `data_access.md`).
* The vendored reversible-gate module is now `qrq._vendor.reversible_access`
  (renamed from its archive file name) and vendored docstrings use
  domain-neutral wording; code bodies are unchanged and fingerprinted in
  `provenance/vendored_sources.json`.
* Keywords: `trajectory data` and `geospatial data` replace `mobility data`.

## 0.1.0 - 2026-10-20

* Records, geometry, index backend, routes, prices, engine and workload
  modules; sealed-ledger reproduction of the access cohort.
* Experimental semantic interfaces (`qrq.semantic`).
