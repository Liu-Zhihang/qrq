# Specification: plugins

Status: draft 0.1, released with qrq 0.2.0. Discovery and the conformance
suite are specified for qrq 0.3 and are not implemented in 0.2.0.

The key words MUST, SHOULD and MAY are used as in RFC 2119.

## 1. Purpose

Datasets, encoders, indexes, routes and price tables vary between studies.
Plugins let other packages provide them without forking qrq, while every
result keeps the same output contract, cost model and ledger format, so that
results from different packages remain comparable.

## 2. Entry-point groups

Plugins register Python entry points in one of these groups:

| group | object returned by the factory | contract |
|---|---|---|
| `qrq.datasets` | loader with `load(path) -> (manifest, records)` | `records.md` |
| `qrq.encoders` | axis encoder, embedding encoder or circuit predicate | `semantic.md`, section 2 |
| `qrq.indexes` | index builder returning a matched access plan | `docs/interfaces.md`, section 3 |
| `qrq.routes` | route counter returning ledger rows | `docs/interfaces.md`, sections 4 and 8 |
| `qrq.prices` | named `PriceTable` | `docs/interfaces.md`, section 5 |

A third-party package registers a plugin in its `pyproject.toml`:

```toml
[project.entry-points."qrq.encoders"]
taxonomy-overture-v1 = "qrq_places.encoders:taxonomy_overture_v1"
```

The entry-point name is the plugin identifier. It MUST be unique within its
group, stable across releases of the plugin, and versioned in the name when
its output changes (for example `-v1`, `-v2`).

## 3. Plugin contract

* The module that defines the factory MUST expose `QRQ_PLUGIN_API = "1"`.
  qrq refuses plugins with an unknown API version.
* Factories take keyword arguments only and MUST be side-effect free until
  called.
* Plugins MUST be deterministic for fixed inputs and parameters, and MUST
  record every parameter that affects output in the ledger or manifest they
  write.
* Plugins MUST NOT change the meaning of existing ledger fields. New fields
  are added under a plugin prefix, e.g. `x_<plugin>_<field>`.

## 4. Discovery (planned for 0.3)

    qrq.plugins.available(group) -> list[str]
    qrq.plugins.load(group, name, **params) -> object

Discovery uses `importlib.metadata.entry_points`. Built-in components are
registered through the same groups, so the core and third-party plugins are
loaded in one way.

## 5. Conformance (planned for 0.3)

`qrq.testing` will provide one conformance check per group. A plugin SHOULD
run them in its own test suite:

* datasets: manifest fields, licence columns, strictly increasing
  `source_ordinal`, finite coordinates;
* encoders: determinism, interval soundness against brute force, embedding
  shape and normalisation;
* indexes: exact membership against a brute-force scan on random windows;
* routes: identical output identifiers for every route on the same window,
  and resource counts that price to the stored cost;
* prices: all prices finite and non-negative.

## 6. Naming and versioning

* Third-party distributions SHOULD be named `qrq-<topic>` (for example
  `qrq-places`) and import as `qrq_<topic>`.
* qrq follows semantic versioning. Within 0.x, the stable core is
  `records`, `geometry`, `index`, `routes`, `prices`, `engine`, `workload`,
  `ledger`, `sweep` and `tasks`: breaking changes there happen only in a new
  minor version and are announced one minor version earlier. `qrq.semantic`
  and anything marked experimental may change in any release.
* Results cited in publications SHOULD name the exact qrq version and the
  identifiers and versions of all plugins used.
