# SLO learning and scheduling

Both `ferrum run` and `ferrum serve` support optional execution-cost learning.
The feature is off when no SLO configuration is supplied. Its predictions cover
host wall time for an actual model invocation; client TTFT, TPOT and visible SSE
text-event ITL require separate end-to-end measurement.

For a quick start, either entrypoint accepts:

```sh
ferrum run qwen3.5:4b-q4_k_m --scheduler-slo ttft:200,tpot:15,itl:50
ferrum serve --model qwen3.5:4b-q4_k_m --scheduler-slo ttft:200,tpot:15,itl:50
```

These millisecond values are illustrative service targets. The shorthand enables
`learn_and_control`, `auto`, and the joint policy with default quantiles. Supplying
`--scheduler-slo` replaces the SLO configuration from `ferrum.toml`; use the file
when selecting observation mode, model family, quantiles or fallback behavior.

```toml
[scheduler.slo]
mode = "observe"
predictor = "physical"
strategy = "joint"
ttft_ms = 200
tpot_ms = 15
itl_ms = 50

[scheduler.slo.joint]
ttft_quantile = 0.95
tpot_quantile = 0.95
itl_quantile = 0.95
min_completion_rate = 1.0
request_deadline_ms = 120000
completion_cohort_maturity_ms = 30000

[scheduler.slo.unknown_cost]
mode = "static_fallback"

[scheduler.slo.unknown_stopping]
mode = "static_fallback"
```

## Modes and selectors

| Setting | Behavior |
|---|---|
| `mode = "off"` | No learning worker, collection queue or cost source. |
| `mode = "observe"` | Collect, fit and diagnose; the scheduler continues its configured ordinary policy. |
| `mode = "learn_and_control"` | Learn and make supported predictions available to the selected SLO policy. |
| `predictor = "physical"` | Nonnegative, scaled regularized regression plus signed residual calibration. |
| `predictor = "quantile_gbdt"` | Bounded trees with separate squared-error center and pinball-loss risk models, plus residual calibration. |
| `predictor = "auto"` | Reuse the family from compatible, fresh bound state; otherwise use a visible conservative Physical fallback. This is not a measured family-winner claim. |
| `predictor = "reference_diagnostic"` | Exact-work observed window maximum plus an engineering margin; no learned-state persistence or fitted quantile claim. |
| `strategy = "joint"` | Compare request-level TTFT/TPOT/ITL and completion constraints under explicit bounded forecast assumptions. |
| `strategy = "reference_step_diagnostic"` | Preserve the older step-budget policy as a diagnostic comparison independent of the model family. |

The base families use the same pre-execution work and actual invocation labels.
Prefill, Decode and one mixed invocation have distinct domains. Sequential
Prefill-then-Decode is represented by separate invocations. Unknown internal
reuse is preserved as unknown; restored tokens are not automatically subtracted
from physical work.

Defaults bound resource use. They do not specify the experiment's sample size,
acceptable regression, or confidence level. Models publish only after disjoint
training, validation and calibration partitions qualify. Work outside the
measured feature envelope, expired evidence and missing stopping/completion
support remain unknown. Shared residual-rank scenarios are an empirical
dependence assumption, not a coverage certificate.

Between fits, a fresh eligible committed invocation may extend the current
model's envelope to include its actually measured features, using the same
qualification as the disjoint calibration suffix. Its label updates only the
signed residual for the unchanged model revision; coefficients, scaling and
model expiry stay fixed. Prior snapshots and pre-execution Unknown diagnostics
remain unchanged. Qualification does not assert conditional coverage for every
combination inside the envelope; calibration can still use a same-domain global
fallback. Refitting rebuilds support from that fit's evidence and does not carry
forward an old envelope automatically.

The default training history policy is `recent`: each fit uses the latest 128
training, 32 validation and 32 calibration observations from its domain's
512-observation window. Both entrypoints also accept this optional TOML setting:

```toml
[scheduler.slo.training_history]
mode = "width_stratified"
samples_per_width = 2
```

`samples_per_width` must be 1 or 2. This policy retains that many newest eligible
actual observations per batch width and work domain, separately from the ordinary
window. Before a fit, it combines fresh retained observations with window entries
strictly before the unchanged validation suffix, removes duplicate sequence IDs,
and selects the fixed training count round-robin by ascending width, newest first
within each width. The validation and calibration suffixes stay chronological and
disjoint. Selection never depends on labels, residuals or benchmark responses.
The temporary union and per-domain history are included in configured memory
bounds. Widths with no eligible measured evidence gain no support.

Retained observations keep their original timestamps. With `width_stratified`,
model expiry is bounded by the oldest observation used in any fitting or
qualification partition; another fit cannot renew old width evidence. New
revisions recompute residuals from their own calibration suffix and never copy
the previous revision's envelope or residuals. The policy and representative
count must match persisted state exactly; older files without this setting mean
`recent`. Observation histories are not persisted and refill after restart.
This is empirical support retention, not conditional calibration or a guarantee
that every future width or context will be supported. Experiments must freeze
the same declared history policy across compared arms.

Completion estimates select arrivals by original ingress age, from the configured
maturity through the stopping evidence TTL (60 seconds by default), inclusively.
The same bounds apply to successes, failures, active requests and missing facts.
Missing facts with unknown ingress age remain unresolved. Recent successes cannot
enter early, and expired arrivals cannot linger as unknown only because a quiet
service has not yet filled its count-based observation window.

## Unknown and infeasible forecasts

`unknown_cost` and `unknown_stopping` each accept `static_fallback`, `reject`, or
`wait`. A bounded wait uses the original ingress time and requires
`max_wait_ms`; its `on_timeout` is `static_fallback` or `reject`. The default is
`static_fallback`. A fallback is ordinary scheduling and is not reported as SLO
assurance.

Known infeasibility has a separate admission policy under
`[scheduler.slo.joint.known_infeasible_admission]`: `mode = "reject"` by default,
or `mode = "wait"` with `max_wait_ms`. A rejected or timed-out arrival remains in
completion-rate accounting. Already admitted work continues; rejection cannot
repair a latency miss that already happened.

## State and execution binding

Use `--scheduler-slo-state /absolute/path/state.json` or set
`slo_state_path = "/absolute/path/state.json"` under `[scheduler]`. Keep state
outside the repository. Each process should use its own destination; the file
is replaced atomically after a successful worker drain on orderly shutdown.
There is no periodic checkpoint. A run without qualified fresh state skips the
save and preserves any existing file.

Learning starts only after the actual executor, selected memory plan and source
binding are available. Reusable state checks the resolved source, numerical and
runtime configuration, actual device information, executor implementation and
binary/operator fingerprints. Custom or incompletely identified execution uses
an instance-scoped identity: learning still works, while requested persistent
reuse reports `SkippedUnbound` and leaves the file untouched. Unsupported
multi-device identity also uses this explicit fallback.

Loading validates schema, feature semantics, identity, support domains, model
structure, calibration semantics and finite values before publishing anything.
A missing file starts cold. Expired state reports `Stale` and starts cold without
renewing its TTL. Corrupt, incompatible or mismatched files and clock rollback
are errors. An explicit family selector rejects state from another family.
Auto restores a fresh compatible state's family; absent or stale state falls
back to Physical with a visible resolution reason. Reusing a numerically valid
state does not establish comparative performance for that family.

Joint planning also needs executor-owned resource evidence. For a complete
resident cohort, compiled-plan executors can account for held persistent state
and remaining paged growth through each request's admitted ceiling, using
currently allocated physical pools and logical availability. This observation
uses nonblocking locks and supports up to 64 resident sequences. It does not
reserve capacity, credit future pool growth, or authorize waiting admissions.
Incomplete cohorts, contention, unsupported growth and missing admission
evidence remain Unknown. Cohort changes invalidate the observation, and actual
execution retains its normal resource checks. Fact and resource projection time
counts toward the same planning time limit as candidate comparison.

A separate complete-cohort observation can cover sequential waiting admissions.
It binds every request lifetime, execution incarnation and admitted context
ceiling, including the final sampled token's lack of subsequent consumption.
The original owner must prove the worst persistent demand of any supported
simultaneously live subset from actual allocated storage and reusable metadata.
The initial native implementation supports declared sequence-only persistent
pools with page-aligned growth or homogeneous fixed contiguous cells. Foreign,
prepared, retiring, checkpoint or unsupported storage ownership yields Unknown;
configured maxima alone do not prove residency. Adapter loads are outside this
base-plan query. Actual admission and release retain their original authority.

The trajectory uses the explicit `next_wave_after_modeled_completion` slot
release assumption, also exposed as `waiting_slot_release_assumption` in health.
It does not model asynchronous close, recovery or release-notification delay.
The capacity observation establishes neither a release-time bound nor latency
coverage, and equal assumptions across candidates do not imply equal errors.
Client-observed TTFT, TPOT, visible SSE ITL and completion remain the acceptance
criteria. Reprojection invalidates prior cohort observations; exact current
query, queue residency and retained-memory budgets are checked before use.

## Diagnostics

Server health exposes scheduler diagnostics and `scheduler_slo.feedback`; the
feedback object is available in observation mode even without a control policy.
It includes binding and persistence status, load status, collection drops,
training failures, fit time and feedback delay. Feedback delay is the interval
from actual invocation completion to worker consumption, including queue time.

`planning_rounds` includes turns that end during admission, including errors and
cancellation. `planning_rounds_before_selection` identifies that subset; each
Engine turn is recorded once, before device execution. These counters describe
planning activity and do not establish SLO feasibility or actual intervention.

`live_prediction` groups actual invocation counters by model family and by
phase/reuse domain for the current process. The observer freezes one center, risk base and available
calibrated upper prediction before execution. Completion records center signed,
absolute and squared error, and risk-base/calibrated exceedances. The center
error denominator is `base_known`; the calibrated exceedance denominator is
`calibrated_known`. `eligible`, unknown reasons, excluded facts and
`mismatched_or_noncausal` are reported separately. Queue drops still contribute
eligible live diagnostics and are counted as `forwarding_dropped`; they cannot
become learning samples. `unprepared_completions` counts direct producers with
no execution-before-start ticket and never contributes fabricated live errors.

The same domain counters report acknowledged prefix restores, known internal
reuse and unknown-reuse participants. Unknown reuse is not counted as zero
tokens, and restore counts alone do not measure saved physical computation.
`worker_base_prediction_known`/`unknown` refer to worker-before-update calibration
opportunities, not live causal prediction coverage. Preparation time measures
the optional per-invocation prediction hook. All these costs are host-side
diagnostics; they are not client-visible latency measurements.

Health reads sample individual atomics and can observe counters during an update;
drained shutdown totals are stable. Memory budgets bound owned payload, queues,
fit scratch and current publication separately. Allocator overhead, process RSS
and old snapshots retained by external readers require separate measurement.
Each in-flight observed invocation also owns one fixed-size private prediction
ticket, released at completion; it retains no model or snapshot.

Use [performance evaluation](performance-evaluation.md) for held-out request
selection, fixed workload/capacity, latency definitions and reporting. Compare
static, Observe, Physical and GBDT under the same initial state; do not tune the
family on the final evaluation requests.

Compiled-plan comparisons can explicitly select
[`configured-envelope` persistent pool preparation](persistent-pool-preparation.md)
for every arm on that backend. Its default remains `demand-driven`; the option
checks the full configured memory shape and allocates persistent backing before
readiness. The backing remains reclaimable, and the scheduler still checks
actual owners on every capacity observation.
