# Structured whole-wave cost model

Status: initial scope implementation. Optional capture connects future route
projection, actual execution, and receipt-qualified host settlement. Explicit
profile schema 9 imports a qualified ordinary-decode model into the planner.
This does not establish complete serving coverage or a performance result.

## Problem and choice

The current selected-work model partitions observations by a complete execution
family. Terminal host work can restore the fully ordered family, making changes
in terminal position or host row pattern split otherwise reusable device work
into separate calibration populations. Requiring independent fit and residual
samples for every such combination makes coverage expensive.

The structured model describes device execution and host work separately while
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
inputs. Retention accounts for allocated backing capacity. Metal producer opt-in
covers the existing proven dense, SwiGLU, attention, primitive and core-transfer
routes. Other routes still need explicit producers. Features must be available from both
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
boundaries, and a fixed TTL. Numerical callers explicitly choose dense FIFO
ordinals or pre-execution reserved members; these coordinates cannot be mixed.
Neither numerical input format proves live provenance on its own. Residual
calibration uses an empirical whole-wave q99. At fit freeze, the model also retains
the largest positive error of the complete fit population, evaluated with the
final fitted predictor and its normal upward rounding. Planning adds the larger
of this fit error floor and the independent residual q99, then adds the original
declared static margin once. Qualification
requires coverage of every declared physical termination position and the
nonterminal case, with no unknown or underestimated heldout point. This finite
challenge does not establish a distribution-free p99 guarantee.

The fit floor is a deliberately more conservative rule than a pure residual
quantile; an observed training maximum above a quantile prediction alone is not
a statistical correctness error. A single slow fit outlier can dominate this
floor, reduce admitted concurrency, or prevent qualification under the existing
maximum wave cost. Predictions exceeding that maximum remain Unknown; values
are never clamped to manufacture a usable model. This rule does not promise
better throughput or SLO compliance and must be evaluated on a new independent
calibration and fixed serving workload. The original TTL, static margin,
membership and phase sample requirements are unchanged. Fit-only or unsupported
queries remain Unknown even if the floor would cover their cost.

Predictions and the calibration export report distinguish `fit_error_floor_ns`,
`residual_ns` (only the independent empirical q99), and
`effective_residual_ns` (their maximum, before static margin). The floor is a
finite observed-error safeguard, not a future hard bound, confidence interval,
or additional qualification sample. Qualification cannot update any of these
frozen components. No qualification sample is reused to enlarge the floor.

The algorithm identity is
`structured_whole_wave_pending_envelope_v2_fit_floor_v1`. The source3/profile10
wire schemas remain unchanged, but their model revision, protocol, domain and
parameter bindings use the new rule. Old source, profile and catalog revisions
are rejected explicitly; they cannot be relabeled or replayed into new
qualification. A fresh complete three-phase capture is required. The shared
runtime adapter used by `run` and `serve` consumes the same planning value;
optional runtime feedback remains a separate, bounded addition and cannot
renew the original TTL or grant missing support.

The core does not deserialize qualified receipts or authorize execution. The
separate schema-9 adapter below owns startup import. Independent full-model
calibration, drift handling, and serving measurements remain completion requirements.

## Live calibration collection

`CalibrationSession::begin_structured_cost_calibration` opens an explicit
diagnostic collector after independent scope discovery and resource warmup.
`HostSettledV1` must be configured before engine creation. The caller declares
one ordinary-decode row count and domain, fit/residual/qualification population
sizes, numerical settings, and source/memory limits before collecting samples.
Beginning collection drains the observation FIFO without resetting live request
owners or their clocks.

Each offered wave is recorded before preparation. Successful preparation reserves
an eligible member and binds its capture before execution. A preparation that
cannot produce a wave consumes an offered attempt without consuming a member.
After reservation, failed execution, missing evidence, a domain mismatch, or an
unexpected termination retains the failed member slot. It cannot be removed or
replaced with a later successful sample. Source-record positions, offered
attempts, member positions, and the original accepted FIFO ordinals remain
separate. Reaping a cancelled waiter settles the original reservation.

The caller uses `structured_cost_progress` and
`freeze_structured_cost_phase` to close each complete declared population before
collecting the next phase. A freeze receipt binds the original clock, FIFO and
member cutoffs, source-prefix digest, protocol, and numerical parameters. Missing
members or unaudited FIFO records fail the calibration. Source write or capacity
errors also revoke eligibility while ordinary request execution can continue.

`finish_structured_cost_calibration` returns a diagnostic artifact with a source
digest and an optional in-memory qualified model. Incomplete qualification has a
failed footer and no model; an incomplete file cannot receive a successful source
receipt. This API does not install a serving predictor or load a model from JSON.
Real backend calibration and heldout measurements are still required.

## Explicit schema-9 predictor

Select the predictor and its required future-route evidence together in the
shared policy file:

```toml
cost_profile = "/absolute/path/to/structured-profile.json"

[cost_observation]
predictor = "structured_whole_wave_v1"
structured_capture = "host_settled_v1"

[cost_observation.profile_import]
declared_local_clock_max_error_ns = 1000000
```

The clock-error value must describe the actual host clock; it is not an inferred
accuracy guarantee. Schema 9 limits the combined declared source and local error.
The source freezes its own model settings. Legacy `cost_observation.model`
overrides, legacy `profile_export`, and selected-family feedback are incompatible
with this predictor. Observe calibration may run without an installed artifact;
it does not start a legacy online trainer.

Export and load verify the entire original source, including offered attempts,
reserved members, FIFO order, the three phase cuts and prefix hashes, exact
receipts and algorithm work, and the independently recomputed numerical payload.
They replay fit, residual calibration and qualification at their original freeze
times and compare parameter hashes. JSON produces replay-only numbers, never a
live qualified settlement receipt. The source file must remain available with
the exported envelope. Source schema 2 includes the independent-attention
sidecar needed to reproduce the original host-stage binding.

An immutable imported model retains the original capture epoch and TTL. Every
candidate projects a typed input from its attached selected route once; queries
perform no file IO or fitting. Unsupported scopes, missing evidence, stale
samples and unsupported numerical combinations remain Unknown. There is no
fallback to an older profile. Every host-content alternative must be covered;
the planner uses their largest planning cost and shortest remaining validity,
including the time needed to finish the wave.

`ferrum.engine.structured_cost_queries_total` reports bounded Known/Unknown
reason labels separately from selected-profile counters. The startup receipt
records source and profile hashes, domain, parameters, original phase cutoffs and
clock ages. These are diagnostic provenance, not proof of a feasible plan.

The initial artifact covers one fixed ordinary-decode width/domain. Prefill,
first decode, multiple simultaneous length terminations, other widths and UTF-8
branches need independently qualified scopes before a complete concurrent
horizon can be established. The ordinary PlainTextGreedy scope excludes active
model EOS, explicit stops and structured completion under the installed policy.

## Feedback and acceptance

The existing explicit `RetrospectiveFamilyMarginV1` policy can adjust margins for
known families, invalidate prior planning epochs, and revoke model validity under
the declared limits. It does not create missing support,
refit the frozen model, or extend TTL. Its settings and persistence policy must
be declared for an experiment; enabling it does not retroactively qualify a
failed heldout evaluation.

That feedback protocol belongs to selected profiles 6–8 and is not applied to
schema 9. Schema 9 has no runtime margin policy. The separate V2 policy below
does not refit or refresh the imported model from serving observations.

### Explicit V2 owner feedback

Source3/profile10 supports a separate, default-off
`cost_observation.structured_feedback.kind = "retrospective_owner_margin_v1"`
policy for an actually imported qualified child or complete catalog. It reuses
the bounded selected-feedback state machine and persistence protocol. The
`policy` object has the same explicit limits as `selected_feedback`: window and
consecutive-error counts, trigger/padding, `maximum_family_margin_ns`, consumption
lag, uncomparable/failed/queue-drop limits and state-byte capacity. Here “family”
means one complete V2 owner/domain. `storage` explicitly selects `create_new` or
`resume` with an absolute receipt path. Source-frozen wave/age limits also apply;
legacy model settings do not replace them.

The worker compares original, privately qualified whole-wave settlements with
one immutable pre-drain model at actual consumption time. A correction ends the
current drain; queued observations retain their original receipt times and use
the newly published epoch in the next drain. This is retrospective
feedback, not a pre-submit prediction or a client SLO observation. Unknown,
unsupported and expired baselines cannot acquire a correction or new support.
Margins only increase relative to the original frozen planning value. Excess
over the declared limit, observation loss or persistence failure revokes the
whole serving snapshot. No fit, residual, qualification sample or source changes.

Every correction or revocation closes the old catalog gate before persistence
and publishes a new monotonic epoch. All retained old snapshots and common-plan
witnesses become invalid, including plans for other owners. Unchanged children
share their original immutable models. Off/Observe selection semantics and
CompleteRequests remain unchanged. `valid_until` remains the original model
expiry; feedback cannot renew it or make a failed calibration deployable.

The observation audit exposes `structured_feedback`, its declared owner-domain
inventory, per-owner comparisons/margins, correction/revocation epoch and full
failure counters. Query metrics use `scope="retrospective_actual"` separately
from candidate queries. These diagnostics do not establish q99 coverage. This
policy does not implement background fit/requalification or model renewal.

The Metal tiny-model test exercises projected and actual partial/final prefill,
decode, and real terminal settlement. This verifies that capture path, not the
Qwen3.5-9B performance target. Remaining work includes complete workload scope
coverage, fresh independent calibration, drift handling, and service validation.
Backend tests must exercise the
real selected routes. Final acceptance remains concurrent TTFT, TPOT, visible-text ITL,
throughput, errors, and memory measurements. See
[SLO configuration](slo-configuration.md) for the current explicit controls.
