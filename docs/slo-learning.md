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
| `strategy = "bounded"` | Recompare a finite number of future execution waves and apply only the next wave. Local ranking does not certify full-request SLO feasibility. |
| `strategy = "reference_step_diagnostic"` | Preserve the older step-budget policy as a diagnostic comparison independent of the model family. |

The base families use the same pre-execution work and actual invocation labels.
Prefill, Decode and one mixed invocation have distinct domains. Sequential
Prefill-then-Decode is represented by separate invocations. Unknown internal
reuse is preserved as unknown; restored tokens are not automatically subtracted
from physical work.

## Finite-horizon control

The opt-in `bounded` strategy uses the same cost models, immutable calibrated
snapshots, request clocks and resource owners as `joint`. Both `run` and `serve`
read it from the SLO configuration:

```toml
[scheduler.slo]
mode = "learn_and_control"
predictor = "physical" # or "quantile_gbdt"
strategy = "bounded"
ttft_ms = 200
tpot_ms = 15
itl_ms = 50

[scheduler.slo.planning]
max_lookahead_rounds = 8
max_catchup_rounds = 8
```

The values above are illustrative, not a validated workload preset. Each cost
scenario compares static dispatch and a bounded set of smaller prefill or
decode-only first waves. `max_lookahead_rounds` is the static reference horizon
in complete execution waves. Each candidate must cover that reference's
per-request prefill and output frontiers and reach its elapsed-time boundary.
The candidate may use at most `max_lookahead_rounds + max_catchup_rounds`
complete waves, continuing ordinary scheduling after its proposed first wave.
It cannot pause after reaching the frontiers to manufacture a common boundary;
reaching all declared scenario endpoints is the only exception to the elapsed-time
condition.
If the extra-wave allowance is exhausted first, that candidate remains Unknown
and cannot be selected from its partial prefix. Other candidates may still be
evaluated. The extra allowance defaults to 8 and accepts 0; it is independent of
the empirical rank limit. It does not require replay to complete every request.
Only the first wave is applied. Actual execution updates the request facts
before a later decision; simulated progress never repays a real deferral.
The original whole-comparison query, elapsed-time, participant and memory
budgets still apply to all reference and catch-up work. With a one-wave
reference and zero extra waves, decode-only cannot cover static prefill
progress. A larger configured allowance can demonstrate later progress but
does not guarantee support, selection or completion within the CPU budget.

The ranking uses equally weighted, SLO-normalized local delay/debt proxies for
generated service opportunities, over the same request cohort. A generated
token may produce no text. Bounded trajectories therefore keep the observed
text clock unchanged and advance a separate service-opportunity clock; a
forecast token never increments the observed event count or clears a text
stall. Legal histories with different token and text-event counts, including
generated tokens before any text, can participate. The observation boundary is
the engine output channel, before server protocol projection and transport;
client-visible SSE timing still requires end-to-end measurement.

Candidate-only output beyond the static reference earns no extra TPOT token
credit. Its actual elapsed time, output frontier and service clocks remain in
the comparison. Service ITL pressure counts new bad service intervals plus one
currently open service gap when it strictly exceeds the ITL target; equality
does not count as a violation. Both terms use violation counts, not a mixture
of interval counts and open-gap severity. They do not predict visible SSE gaps.

The comparison retains waiting requests and unfinished work. Observed open
TTFT/text-gap exposure is reported separately at the larger of the static and
candidate durations, without inventing future text updates or adding another
scoring weight. Future visible timing remains unknown even at the token cap.
Historical violations are not erased.
Prefill service-rate extrapolation is a ranking heuristic, not a physical time
bound. When both plans leave a waiter without service or first text, a shorter
candidate observation earns no waiting-time improvement: its exposure score
uses at least the static observation duration. This is a ranking guard, not a
prediction of that waiter's TTFT. The separate catch-up rule above determines
whether the candidate has actually reached the reference boundary in simulation.
A request already past its TTFT target cannot lose predicted prefill
progress relative to static. Decode-only also retains the actual-progress
deferral guard. Ties retain static; this initial strategy selects strict local
delay improvements, not unvalidated throughput-only gains.

Future natural stopping and transport stalls remain unknown. The finite
survival scenario does not call the full-output stopping/completion forecast,
and does not turn a finite horizon into an EOS event or a complete-request
feasibility result. Unknown cost support retains ordinary scheduling. Model
uncertainty or a poor local score does not reject admission; normal resource
probes and capacity checks remain authoritative. Full-request TTFT, TPOT,
client-visible SSE ITL, errors and completion rates must still be measured.
Because this strategy does not implement predictive admission waits or
rejections, it requires `unknown_cost` and `unknown_stopping` to be
`static_fallback` (their defaults). Explicit `reject` or `wait` settings are
configuration errors rather than silently ignored policies. The full-trajectory
`joint` admission verdict settings do not apply to a bounded local ranking.

Health diagnostics report `bounded` comparisons, completed evaluations,
selected and applied actions, retained static actions, unknown comparisons and
invalidated applications separately. `bounded.candidate_horizon_exhaustions`
counts candidate-by-rank traversals that exhausted their configured catch-up
allowance. It can be nonzero in a completed comparison that retained static or
selected another action; it is not a count of unknown whole comparisons and
does not imply global query or time-budget exhaustion. These counts also
propagate to the existing per-iteration planning trace. The scores are local
ranking proxies, not violation probabilities or client SLO verdicts.
Full-trajectory `joint` remains the default during validation of this new strategy.

The shared engine also writes `scheduler.iteration` events when
`--scheduler-trace-jsonl` is enabled. These link the iteration and batch IDs to
the final scheduled request ranges, that planning round's statistics and
prediction time, and the engine process result and host duration. Scheduled
ranges are not proof of successful device work: request failure/close events
and committed-token deltas must be checked separately. These records support
execution diagnosis; client timestamps remain the latency measurement.

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

To retain causal feature extrema as training candidates as well, select:

```toml
[scheduler.slo.training_history]
mode = "width_and_extrema"
samples_per_width = 2
```

This policy keeps at most 18 distinct actual observations for the nine feature
axes' minima and maxima, in addition to the bounded width history. An observation
enters that reservoir only after it leaves the latest validation/calibration
suffix. Before fitting, fresh extrema from the causal candidate union receive
priority within the unchanged training sample count; remaining slots are selected
by width and recency, accounting for the already selected records. If distinct
extrema exceed the training budget, the candidate fit fails and the previous
coherent model/calibrator remains active. This policy does not enlarge training
or calibration windows, change the collection workload, or extrapolate support.

The reservoir retains original timestamps and labels. Labels of old extremal
observations enter training only; they are never copied into the new model's
calibration suffix. Equal extrema can use a newer actual observation, but replay
does not renew evidence. This bounded reservoir is not an exact record of all
observations within the TTL: after a dominant extreme expires, a discarded
runner-up may no longer be available, and support may shrink. It also cannot
guarantee a direct training representative for every width when there are more
widths than training slots. The default `recent` and existing `width_stratified`
selection rules remain unchanged.

Retained observations keep their original timestamps. With either width policy,
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

An observed stopping forecast filters the retained records once, then sweeps
integer risk-set and natural-stop events over the requested positions. Its work
is bounded by the retained record count plus the configured forecast-position
limit. Temporary event storage uses two machine-sized counters per position
(at most 64 KiB with the default 4096 positions on a 64-bit host). Each query
checks the original observation times and rule identity; it does not cache or
renew evidence across query times. Cap endpoints, support gaps and unresolved
probability mass retain their existing meaning.

The joint policy keeps the original static wave and compares smaller prefill
chunks. When that wave contains both prefill and decode work, it can also
consider a decode-only first wave. This action retains every selected decoder
and postpones the selected prefills; it does not change their admission, prompt
position, output limit, or resource ownership. These first-wave actions use the
same continuation after that wave and retain the existing support, planning,
joint-risk, and gain checks.

A request can be postponed this way only once without actual logical prefill
progress. Planning another wave, restoring a cached prefix, or moving queues
does not renew that eligibility. An executed prefill chunk that advances its
logical progress does. A pure prefill wave cannot become an empty batch, and
ordinary scheduling remains responsible for resource waits and subsequent
work. These rules do not establish a measured latency or throughput benefit.
`joint_decode_only_interventions` counts successfully applied decode-only waves
within `joint_interventions`; merely evaluating or rejecting one does not count.

Dispatch also compares positive prefill phase limits derived from half and a
quarter of the selected prefill chunk. A phase binds only the actual selected
resident prefills, through their original prompt endpoints. Every simulated
phase wave must advance those endpoints; after completion it resumes the same
ordinary continuation. The real scheduler installs the selected limits
atomically against the request lifecycle, work frontier and prior phase state.
The two durable candidates preserve a policy budget charge while reducing actual
execution. With a configured prefill chunk, that charge uses the current chunk
before shortening it for the prompt's final tail. It remains bounded by the
current step and active-decode budgets, executor ceiling and any prior cap's
budget basis. Saved execution tokens therefore do not immediately admit more
prefill work into the same wave, including its final short chunk. Both prefill
selection passes share this accounting; cost queries, progress and resource
requirements still describe only actual execution tokens.

Without a configured chunk, the charge keeps the ordinary request-bounded
proposal before the new cap; it does not invent a chunk size. The existing
first-wave-only smaller-chunk actions retain ordinary refill behavior. These
finite alternatives keep the existing candidate and planning budgets. Neither
packing rule guarantees that every future wave is inside the learned support
envelope; a genuine one-token tail, for example, still needs measured support.

An installed limit survives stale or unknown forecasts and execution-capacity
feedback. Further decisions can tighten it; it clears when the original prefill
enters decode or the request ends. New arrivals and reused request IDs do not
inherit it. The policy cap is separate from the executor's capacity ceiling.
`joint_prefill_phase_interventions` counts successful installations within
`joint_interventions`, not completed GPU executions.

Admission can also compare a persistent cap for the eligible waiting and resident
prefills in its original natural first wave. This preview uses the same actual
work and policy budget charges as subsequent scheduling. If selected, the
scheduler validates the entire captured cohort and
atomically installs the caps before reevaluating admission. Installing a cap
does not reserve resources, admit a request, or advance its work frontier; the
original executor still decides resource admission. Waiting requests retain
their caps when they are admitted.

The fresh admission comparison uses the same remaining time and query budget,
without proposing another cap installation. If it becomes unknown, the declared
unknown policy still applies and the installed caps remain. A usable cost model
does not imply known resource capacity or guaranteed admission. The separate
`joint_admission_prefill_phase_commits` counter records successful pre-admission
installations; it is not included in dispatch's `joint_interventions` and does
not assert successful execution or SLO compliance. Positive chunks and the
modeled progress check do not establish general starvation freedom. The shared
scheduler applies these rules to both `run` and `serve`.

## Unknown and infeasible forecasts

`unknown_cost` and `unknown_stopping` each accept `static_fallback`, `reject`, or
`wait`. A bounded wait uses the original ingress time and requires
`max_wait_ms`; its `on_timeout` is `static_fallback` or `reject`. The default is
`static_fallback`. A fallback is ordinary scheduling and is not reported as SLO
assurance.

For a full `ScenarioCompletion` comparison, a known initial scenario distribution
and at least one applied known execution group can establish a static trajectory
prefix. If a later static cost query is outside observed support, the planner
may retain that prefix as a lower bound. Full static completion time remains
unknown, and the target remains all declared endpoints. An alternative can
rescue this case only when the static risk lower bound is already Infeasible,
every candidate rank has fully supported costs and completes every declared
request endpoint with a completion clock, and its joint risk is Feasible. The
candidate need not finish before the static prefix's lower bound: that bound
does not establish when the full static trajectory would finish. Static target
time remains unknown and the window-unproven diagnostic remains observable;
selection does not prove a relative throughput gain. Static-Feasible optimization
still requires its existing paired gain and common-window checks. An unknown
initial distribution, no applied known prefix, finite planning horizons, or
other failures retain their existing Unknown/error behavior.

`rank_partial_static` reports these partial trajectories separately.
`resolved_mass` means endpoints have been assigned to risk categories; it does
not mean every execution cost was covered. An unfinished static tail cannot be
declared Feasible. This rule neither expands model support nor establishes an
observed serving SLO improvement.

A Known prediction for the next execution does not establish support through a
request's stopping endpoint. Decode context grows as output is generated, and
alternative batches can have different aggregate work features. A cold process
that has only completed shorter requests can therefore keep falling back while
longer requests run. Increasing the number of short warmup requests does not
qualify those unseen costs. Retaining a candidate's known prefix also cannot
prove a required 100% completion rate while its remaining completion time is
unknown.

Learning can qualify new work after an eligible execution actually completes;
the snapshot already captured by a comparison remains unchanged. Prior training
or initialization must supply real completed observations, separate fitting
from calibration, and preserve the original evidence timestamps. Cost support,
fresh calibration, stopping-rule support, and product completion evidence are
distinct requirements. More training data or a loaded model alone does not
establish all four, or guarantee the client latency SLOs.

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

`joint_evaluation.admission` and `.dispatch` count entered scenario evaluations,
including failed traversals. Their denominator excludes cohort or capacity checks
that return before the evaluator. `provider_calls` includes seed, failed and
panicking calls to the original cost provider. It supplies the scenario part of
the round's actual query charge; stopping and completion forecasts are charged
separately. `simulation_queries` also includes exact cache hits, which do not
consume another provider call. The older
`joint_cost_snapshot_queries` and cache counters describe completed traversals
only. A completed traversal is not proof of a feasible action or a policy change.
Evaluator elapsed time is measured wall time, not OS CPU time or the charged
planning budget. Rank and query maxima are maxima, not additive counters.

`joint_decision.admission` and `.dispatch` distinguish decisions computed by the
risk accumulator, decisions returned after the final time check, and decisions
consumed by the admission or dispatch policy. A post-finish time rejection is
counted separately. Consumer outcomes describe the selected policy or admission
verdict; only `joint_interventions` records successfully applied dispatch edits.
An evaluator error does not produce a completed decision observation.

The `unresolved` counters describe only final `FeasibilityUnresolved` consumer
outcomes. They separate zero alternatives, the static risk classification, its
effective selection classification, and the candidates' metric classifications
and first reached selection gates. A partial static trajectory can have an
effective `Unresolved` selection status even when its risk bounds are feasible;
the admission policy still evaluates its original risk bounds. Candidate counts
use alternatives as their denominator, while static counts use decisions.
Completion and window flags are nonexclusive observations. A candidate's first
gate is the branch actually reached, not a list of every condition it might fail.
Undefined TPOT or ITL is distinct from feasible or unresolved risk.

Each consumer retains its latest `FeasibilityUnresolved` witness: the static
action and at most two alternatives, their original risk intervals, weighted
TPOT/ITL denominators, target-time support, and window facts. Omitted alternative
details are explicit; all alternatives still enter the counters and policy.
The failure-limit array is ordered TTFT, TPOT, ITL, completion. Product-success
uncertainty is separate from request mass whose modeled completion is unknown.
Partial-static ranks and unresolved endpoint mass do not imply complete cost
coverage. This witness belongs to one aggregate and is not joined to the cost
owner's separately retained rejection. It contains no prompts or request IDs.

`joint_decision.retained_bytes_bound` describes one fixed Rust diagnostics value,
including both consumer slots. It excludes other copies, cold serialization,
allocator overhead and process RSS. The payload does not grow with requests or
health reads. Risk aggregation runs before the existing final comparison time
check. Admission verdict recording and diagnostics merging remain outside that
timer, so it does not measure all observation overhead. These diagnostics make
no claim of negligible runtime cost and issue no additional model or stopping
queries.

A model-support rejection does not spend unissued queries. Later admission or
dispatch work in the same round can use only the remaining original allowance;
this introduces no retry loop and does not relax support requirements. An
evaluation that exhausts its provider or simulation query budget prevents
subsequent forecasts and evaluations in that round after it returns, including
when an unsuccessful alternative
observed the exhaustion. An opaque provider planning-budget rejection also
closes the round conservatively without inventing a call count.
The elapsed-time limit remains cumulative and is not refunded after a failure.

Each population retains its latest first failure and latest terminal failure,
including the rank, candidate, wave, invocation group and consumed query counts.
An unsupported alternative can produce a first failure in an otherwise completed
traversal. These bounded samples do not describe the frequency of every cause.
`joint_cost_unsupported_detail` subdivides the existing unsupported-domain bucket;
do not add it to that parent bucket. Details distinguish unqualified work or
empirical support, invalid numeric predictions and scenario-contract failures
without exposing model-private features. Deadline exhaustion after a provider
call is a planning-budget failure, not evidence of missing model support. These
diagnostics do not change support bounds, deadlines, candidates or fallback policy.

The learned owner's cold `query_rejections` statistics record calls rejected
outside observed feature support. `below_axis_counts` and `above_axis_counts`
can both count several axes for one query; they are not disjoint percentages.
The bounded `last` witness contains that call's exact work, features, model
bounds, revisions and validity remainder. It is explicitly unjoined to scheduler
admission/dispatch samples and describes the last successfully retained call,
not necessarily the latest query clock. It contains no prompt, response or
request identifier. An observed feature box does not establish coverage of
every combination within the box.

All published snapshots share one preallocated recorder. A rejected call copies
at most the configured participant limit using `try_lock`; contention drops the
witness instead of waiting. The counters expose dropped and overwritten samples.
Cold reads can report `last_witness_read_busy`, and concurrent counter and witness
reads are not an atomic snapshot. The fixed recorder and work payload count
toward `max_retained_bytes`; cold returned copies and allocator/RSS overhead do
not. Diagnostics are not persisted with learned models and do not re-evaluate
predictions, alter support, or prove negligible runtime overhead.

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

## Offline qualification inputs

The [offline example](../crates/ferrum-cli/examples/prepare_slo_qualification.rs)
prepares synthetic inputs using a [typed recipe](../crates/ferrum-cli/examples/prepare_slo_qualification/recipe.rs)
and local tokenizer/template files:

```sh
cargo run -p ferrum-cli --example prepare_slo_qualification -- RECIPE.json NEW_OUTPUT_DIRECTORY
```

The recipe declares the template source, output protocol, context/block geometry,
request policy and finite stages. Ferrum's renderer and actual special-token-aware
encoding must reach each requested input length exactly. The example writes client
configurations, rendered-token audits and source/output hashes into a new directory
without overwriting an existing destination. It does not load weights, execute
inference, read generated answers or access a dataset or network.

These outputs establish prepared input geometry only. Bind recipe declarations to
the actual server configuration before use. Prefix audits compare all prior
prepared token sequences and report potential reuse; nonce text does not establish
a cache miss. Finite search/file/payload limits do not impose a hard RSS or time
sandbox on template/tokenizer internals. Actual invocation width, successful
execution, model/calibration qualification and latency SLOs require separate
observations. Synthetic initialization is not the primary ShareGPT benchmark.
Declare stage sequencing, failure stops and measurement boundaries separately;
the tool has no default request matrix.
