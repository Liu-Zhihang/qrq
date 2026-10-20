# Changelog

All notable changes are listed here. Versions follow semantic versioning;
within 0.x the stable core is defined in `docs/specs/plugins.md`, section 6.

## 0.2.0 - 2026-10-20

Reference release for the accompanying manuscript. No published number
changes relative to 0.1.0.

### Added

* `qrq.sweep`: exact repricing of per-tree endpoints over a caller-supplied
  price grid, with optional caller-supplied group labels.
* `amplified` option in `qrq.engine` and `qrq.ledger`: the envelope with the
  priced amplitude-amplification route.
* Per-tree endpoints in ledger reproduction.
* Caller-supplied window grids: registering windows and forming
  adjacent-scale pairs takes the grid from the caller.
* Draft specifications in `docs/specs/`: record schema (points and places),
  semantic predicates and costs, plugins.
* `CHANGELOG.md` and `CONTRIBUTING.md`.

### Changed

* A generic `group` label (any user-defined grouping of sources: region,
  campaign, fleet, ...) replaces the domain-specific names of the reference
  study in `qrq.sweep`: `SweepCohort.group`, `group_of`,
  `groups_with_forward` and the per-source row key `group`.
* Study-specific reproduction, sealed regressions and the
  reconstruction-task comparison are not part of qrq. A study package
  depends on qrq and supplies its own grid, evidence readers and tests.
* The vendored reversible-gate module is now `qrq._vendor.reversible_access`
  (renamed from its archive file name) and vendored docstrings use
  domain-neutral wording; code bodies are unchanged and fingerprinted in
  `provenance/vendored_sources.json`.
* Keywords: `trajectory data` and `geospatial data` replace `mobility data`.

## 0.1.0 - 2026-10-20

* Records, geometry, index backend, routes, prices, engine and workload
  modules; sealed-ledger reproduction of the access cohort.
* Experimental semantic interfaces (`qrq.semantic`).
