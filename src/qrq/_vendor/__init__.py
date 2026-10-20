"""Vendored reference modules.

The modules in this sub-package are copies of the reference implementation
used for the published resource ledgers. Their algorithms are unchanged; the
only edits are relative imports, removal of import-path manipulation, the
exposure of two hard-coded prices as keyword parameters with identical
defaults, domain-neutral docstring wording and one file name
(``reversible_access``). Every file, its original SHA-256, a fingerprint of
its code bodies and the exact list of edits are recorded in
``provenance/vendored_sources.json`` at the repository root.

Public code should use the wrappers in :mod:`qrq.routes`, :mod:`qrq.prices`
and :mod:`qrq.amplitude` instead of importing from here directly.
"""
