# Structured whole-wave cost model

Status: staged implementation. Optional capture connects future route projection,
actual execution, and receipt-qualified host settlement. It does not enable a new
predictor, import a new profile, or qualify a performance result.

## Problem and choice

The current selected-work model partitions observations by a complete execution
family. Terminal host work can restore the fully ordered family, making changes
in terminal position or host row pattern split otherwise reusable device work
into separate calibration populations. Requiring independent fit and residual
samples for every such combination makes coverage expensive.

The next model will describe device execution and host work separately while
retaining a single whole-wave time target. A shared statistical population is an
explicit empirical hypothesis, not permission to reorder execution. Improving
coverage and controlling underestimation are separate requirements.

We retain the whole-wave target because splitting host and device estimates does
not itself remove sparse combinations or correlated error. The current host
receipts can distinguish an execution envelope and post-return settlement, but
the execution envelope is not pure GPU time. Separate component P99 estimates
cannot simply be added to establish a whole-wave P99 bound.

## Evidence contract

`CanonicalWaveCostBuilder::new_with_structured_statistics` explicitly opts into
the new collection. `finish_with_structure` returns the existing exact and
statistical evidence together with `UnsettledStructuredWaveEvidenceV1`.
The ordinary builder does not collect an additional host-row representation.

The device template comes from the same checked selected-command stream used by
actual execution and future route projection, before host categories are added.
It retains provider and algorithm identity, command order, transfers, participant
layout, output product, readback route, retry count, and execution domain. An
optional grouped template retains only the existing provider-declared
independent attention grouping; it grants no host or whole-wave permutation
authority. The first contract supports the existing eager, Graph-disabled domain.
Graph support requires a later adapter consuming its actual sealed recipe.

Host structure retains physical row position, installed policy, decoder bounds,
pending UTF-8 state, first-token state, prefill boundaries, masks, and decode
output requirements. A final prefill produces output but does not by itself
imply a FullLogits device product. Reaching the length boundary is an expectation,
not proof that terminal cleanup succeeded; a token below that boundary may still
terminate through EOS or a stop condition.

The structure remains bound to the complete exact shape. Existing equality,
serialization, profile interpretation, execution guards, and resource authority
are unchanged. Missing selected evidence or unsupported execution remains
explicitly unavailable.

## Settlement and modeling boundary

The unsettled type is not a training sample. The engine adapter binds it to actual
completion evidence from the same call and request incarnations, generations,
work ranges, and physical rows. Actual host processing order comes
from the completion receipt, not from physical row position. Failed, cancelled,
partial, or unmodeled additional work cannot become successful settlement through
a caller-supplied boolean. `QualifiedStructuredWaveEvidenceV1` has no public
constructor or deserializer. Its binding also detects changes to the enclosing
diagnostic shape, identities, clocks, and host receipts.

Enable capture explicitly in the SLO policy file shared by `run`, `serve`, and
`calibrate-slo`:

```toml
[cost_observation]
structured_capture = "host_settled_v1"
```

The default is `"disabled"`. Enabling capture does not select a predictor or
create observation hooks when the configured execution path does not collect
cost observations. The calibration raw wave record exposes `structured_evidence`
through its explicit diagnostic view. Existing source/profile serialization
omits this field. Queue and export limits account for retained backing capacity;
the export size estimate reserves a conservative fixed structural overhead.

The time target remains:

```text
T = max(executor_returned, all row.settled) - prepare_started
```

Receipt finalization supplies observation age and TTL; it is not a replacement
for this time boundary. Device event durations and host waits must not double
count overlapping work.

The new fit and joint support need sufficient numerical detail for the proposed
sharing. Aggregate kernel work is not per-algorithm work: equal totals can hide
different allocations of work to different algorithms. The first contract does
not claim to resolve that ambiguity. An optional selected-command builder now
records work by algorithm and transfer kind, with a binding to the numeric
assignment in that same command. Its default path does not collect the sparse
table. Complete-wave aggregation retains a bounded sparse table and an ordered
binding to the original command assignments. Missing producers, overflow, or
capacity exhaustion leave the table unavailable; partial tables are not training
inputs. Retention accounts for allocated backing capacity. Complete provider
opt-in remains a separate integration step. Features must be available from both
actual and prospective execution, without using future durations or actual EOS
outcomes as inputs.

The model must retain independent fit, residual calibration, and heldout phases.
Residuals apply to the complete prediction and complete measured wave. Old
profiles or raw records cannot acquire missing structure by changing a version
header. A minimum sample count is an eligibility floor, not proof of statistical
sufficiency or end-to-end SLO compliance.

## Numerical core and initial scope

The separately versioned Rust numerical core implements normalized row-space
identification and QR fitting. It rejects ill-conditioned or unidentified
directions instead of inventing coefficients for unobserved work. The initial
scope is ordinary decode with a fixed row count, an established output history,
and at most one expected length termination. Prefill, first decode, multiple
terminations, pending UTF-8, masks, and EOS/stop cleanup require additional scope
design and qualification. This is not the complete serving predictor.

Per-algorithm work and host work share one whole-wave target. Terminal count and
physical-position moments are numerical features; positions do not each create
a new calibration family. All raw work coordinates still participate in joint
support checks. Each accepted query must be supported by a complete fit sample
and a complete residual sample, with observed lower bounds; coordinate-wise
maxima from different samples cannot manufacture a supported combination.

Fit, residual calibration, and qualification consume complete declared
populations with source/protocol identities, unique call IDs, original clock
boundaries, and a fixed TTL. The initial dense-ordinal interface cannot represent
a filtered multi-domain FIFO; that needs an explicit population adapter. Residual
calibration uses an empirical whole-wave q99 plus a declared margin. Qualification
requires coverage of every declared physical termination position and the
nonterminal case, with no unknown or underestimated heldout point. This finite
challenge does not establish a distribution-free p99 guarantee.

The core does not deserialize qualified receipts, import a profile, install a
production predictor, or authorize execution. The live collector, independent
full-model calibration, drift handling, and serving measurements remain separate
completion requirements.

## Feedback and acceptance

The existing explicit `RetrospectiveFamilyMarginV1` policy can adjust margins for
known families, invalidate prior planning epochs, and revoke model validity under
the declared limits. It does not create missing support,
refit the frozen model, or extend TTL. Its settings and persistence policy must
be declared for an experiment; enabling it does not retroactively qualify a
failed heldout evaluation.

The Metal tiny-model test exercises projected and actual partial/final prefill,
decode, and real terminal settlement. This verifies that capture path, not the
Qwen3.5-9B performance target. The remaining sequence is complete per-algorithm
producer wiring, justified numerical features and model identity, fresh
independent calibration, and service validation. Backend tests must exercise the
real selected routes. Final acceptance remains concurrent TTFT, TPOT, visible-text ITL,
throughput, errors, and memory measurements. See
[SLO configuration](slo-configuration.md) for the current explicit controls.
