# Specification: semantic predicates and their costs

Status: draft 0.1, released with qrq 0.2.0. Implemented in 0.2.0:
`SemanticEncoder`, `CategoricalIntervalEncoder` and `SemanticBox` (single
interval, filtered after the 3-D index). Everything else in this document is
specified for later releases and has no implementation yet.

The key words MUST, SHOULD and MAY are used as in RFC 2119.

## 1. Query form

A semantic range query is a conjunction

    R = B AND S

of a closed space-time box `B` (`qrq.geometry.SpaceTimeBox`) and a semantic
condition `S` on the same records (points or places, see `records.md`). The
output contract is unchanged: every route prepares the uniform state
`|S_1(I)>` over the identifiers `I` that satisfy `R`, with auxiliary
registers cleaned, so that routes remain comparable.

`S` takes one of three forms.

| form | membership | indexable |
|---|---|---|
| interval set | encoded axis value lies in a union of closed intervals | yes, as a fourth index axis |
| similarity ball | cosine similarity between record and query embeddings is at least `tau` | no; evaluated as a test |
| circuit predicate | acceptance probability of a parameterised circuit is at least `tau` | no; evaluated as a test |

## 2. Encoders

Three encoder protocols cover the three forms. All encoders MUST be
deterministic: fixed parameters, a stable `encoder_id`, and identical output
for identical input on every platform within float64 round-off.

### 2.1 Axis encoders (interval sets)

The existing `SemanticEncoder` protocol (`encode`, `interval`) is the axis
encoder for a single interval. It is extended with

    intervals(labels) -> list[tuple[float, float]]

which returns the minimal list of disjoint closed intervals covering a label
set, so that non-contiguous selections remain expressible. Typical axis
encoders order a category taxonomy depth-first, so that every subtree is one
interval.

### 2.2 Embedding encoders (similarity balls)

    class EmbeddingEncoder(Protocol):
        encoder_id: str
        model_id: str
        model_revision: str
        dim: int
        normalised: bool
        def embed(self, texts: Sequence[str]) -> np.ndarray: ...   # float32 (n, dim)

* Record embeddings are computed once and stored as sidecars
  (`records.md`, section 4); queries are embedded at query time.
* Transformer text models MUST be pinned by model identifier and revision,
  with the text fields, normalisation and truncation rule recorded in the
  sidecar metadata. The licence of the model weights MUST be recorded.
* Membership is `cos(e_record, e_query) >= tau`. Ground truth is computed
  exactly in float64 from the stored float32 embeddings.

### 2.3 Circuit predicates

    class CircuitPredicate(Protocol):
        encoder_id: str
        def resources(self, query) -> Mapping[str, int]: ...     # qubits, depth, gate counts by type
        def accept_probability(self, record_ids, query) -> np.ndarray: ...

* A circuit predicate wraps a parameterised circuit that compares a record
  with the query, for example a compositional sentence circuit built with a
  quantum natural-language-processing toolkit. qrq will ship such an adapter
  as an optional extra; the core package MUST NOT depend on it.
* Membership is defined by a deterministic threshold on the exactly computed
  acceptance probability (classical simulation for ground truth). Both routes
  therefore share one ground truth; sampling error of an execution is a
  property of the route and MUST be priced or reported, never folded into
  membership.
* `resources` provides the counts used to price the coherent test.

## 3. Integration with the access routes

Two modes integrate `S` with the existing matched access plan.

**Mode I, indexed axis (interval sets only).** The axis value becomes a
fourth coordinate of a static `(x, y, t, s)` index with the same matched plan
as the 3-D backend (first-witness frontier, block and enumeration
traversals). Semantic bounds can shrink the candidate frontier, so the
semantic condition changes `M` as well as `K`.

**Mode II, tested predicate (all forms).** The index stays three-dimensional;
`S` enters only the per-candidate test. With `K_S` records satisfying `R`
and a candidate frontier of `M` records selected by `B`,

    Q = P + (M / K_S) (H_geo + H_sem)

where `H_sem` is the priced coherent semantic test per attempt, and the
classical-led cost `C` adds the classical semantic test on every candidate it
inspects. When `K_S = 0` the window is empty and is excluded, as for boxes.

## 4. Prices and ledgers

* `PriceTable` gains two prices: `g_sem_classical`, the cost of one classical
  semantic test (one interval check, one `dim`-length dot product, or one
  classical evaluation of the predicate), and `g_sem_quantum`, the cost of
  one coherent semantic test. Encoders MAY propose defaults from their
  resource counts; every reported result MUST state the prices used and
  SHOULD include a sweep over both.
* Ledger rows gain `n_sem` (classical semantic tests) and `n_qsem` (coherent
  semantic tests) per route. Existing ledgers remain valid with both counts
  equal to zero, so existing ledgers price exactly as before.

## 5. Workloads

Semantic adjacency mirrors the widening and lengthening pairs of the
space-time workload (`qrq.workload`). A registered semantic workload fixes,
before costing,

* the space-time window grid;
* the semantic ladder: taxonomy levels from leaf to root for interval sets,
  or a decreasing threshold ladder `tau_1 > tau_2 > ...` for similarity balls
  and circuit predicates;
* adjacent pairs along the semantic ladder at a fixed space-time window.

A semantic reversal is a pair whose smaller selection has `Q < C` and whose
larger selection has `Q > C` (or the reverse), exactly as for space and time.

## 6. Reporting

Results produced with semantic predicates SHOULD report the items of an
evaluation checklist: task and target, how the semantic selection was
chosen independently of access advantage, local
support and denominators, the output contract and its consumer, the
strongest classical competitor, prices and their sweep, reuse, and negative
results.
