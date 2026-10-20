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

## 0.1.0 - 2026-10-20

* Records, geometry, index backend, routes, prices, engine and workload
  modules; sealed-ledger reproduction of the access cohort.
* Experimental semantic interfaces (`qrq.semantic`).
