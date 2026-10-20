# Contributing

qrq aims to be a common core for quantum range queries over spatio-temporal
and semantic records. Contributions of datasets, encoders, indexes, routes
and price tables are welcome, preferably as plugins (`docs/specs/plugins.md`)
so that they can evolve independently of the core.

## Ground rules

* Published numbers are fixed. A change that alters any sealed regression
  (`tests/test_sealed_*.py`) is a bug unless it comes with a new major or
  minor version and a changelog entry explaining the difference.
* Every result must be reproducible from recorded inputs: pin data releases,
  model revisions and plugin versions, and record all prices.
* Keep the output contract: routes that are compared must prepare the same
  output over the same identifiers.
* Do not commit third-party data. Loaders read user-provided copies and keep
  licence and attribution columns (`docs/specs/records.md`, section 6).
* Do not store direct identifiers of people in any record or ledger.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install --no-build-isolation ".[test]"
pytest
```

Sealed regressions are skipped unless the evidence environment variables are
set; see `docs/reproducing.md`.

## Style

* Python 3.10+, type hints, English docstrings that state the rule a block
  implements.
* Small modules with one responsibility; experimental code lives under
  `qrq.semantic` or in a plugin until it has tests and a specification.
* Every new module comes with synthetic tests that run without sealed data.
