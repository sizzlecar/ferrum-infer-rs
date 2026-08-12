# Runtime vNext v0.8.0 Metal host-global swap noise amendment (2026-08-13)

## Status and precedence

- Status: Active, mechanical, fail-closed.
- This amendment applies only to the Metal host-global swap classification and completed-HTTP-cell
  cumulative resume contract of each formal R2 and R3 performance lane. It supersedes the
  exact-zero Metal swap-growth and single-server-epoch wording in
  [`GOAL.md`](GOAL.md), [`MODEL_MATRIX.md`](MODEL_MATRIX.md),
  [`G09_PERFORMANCE.md`](G09_PERFORMANCE.md), [`G10_RELEASE.md`](G10_RELEASE.md), and
  [`PERFORMANCE_ACCEPTANCE_AMENDMENT_2026-08-06.md`](PERFORMANCE_ACCEPTANCE_AMENDMENT_2026-08-06.md).
- This amendment does not change the three models, reference hardware, model files, workload,
  request counts, repeats, concurrency/admission floors, throughput/latency/CV thresholds,
  correctness prerequisites, profile requirements, typed memory budgets, or artifact freshness.
- The same rules apply independently to exact staged-binary R3 evidence. R2 evidence cannot be
  resumed into or substituted for R3; R2 acceptance does not waive or replace the R3 measurement.

## Measurement fact and scope

On macOS, `vm.swapusage` is a host-global counter. It is not attributable to the Ferrum process.
The validator must therefore distinguish one bounded host-global counter step from sustained or
repeated swap pressure without changing any raw observation.

One `(model, metal)` performance lane contains exactly these eight formal measurement scopes:

1. random HTTP c1;
2. random HTTP c4;
3. random HTTP c16;
4. real-chat HTTP c1;
5. real-chat HTTP c16;
6. `ferrum run` sample 1;
7. `ferrum run` sample 2;
8. `ferrum run` sample 3.

Missing, duplicate, unbound, or additional scopes make the lane REJECT. The allowance below is
lane-global across all eight scopes; it is not one allowance per scope or per cell.

For each scope, the validator reads the immutable, hash-bound raw resource observations inside the
formal measurement window, orders them by their recorded monotonic timestamp, and computes every
adjacent positive swap delta as:

```text
positive_delta_bytes = max(0, next_swap_used_bytes - current_swap_used_bytes)
```

Negative deltas never cancel or credit a positive delta. A scope's first sample is its baseline;
only observed adjacent samples inside that scope form steps.

## Mechanical acceptance rule

The normal no-growth path remains unchanged. If all eight scopes contain zero positive swap
steps, the lane uses the pre-existing Metal resource contract, including the `2 GiB`
(`2,147,483,648` bytes) measured physical-headroom floor. This path does not inherit the stricter
`4 GiB` noise-qualification floor.

The host-global-noise path is available only when all of the following are true across the exact
eight-scope set:

1. `positive_step_count == 1`;
2. `max_positive_step_bytes <= 1,048,576`;
3. `total_positive_growth_bytes <= 1,048,576`;
4. the minimum recorded physical headroom across every scope is at least
   `4 GiB` (`4,294,967,296` bytes);
5. thermal throttling, OOM, admission failure, and resource-probe error counts are all `0` in every
   scope;
6. the positive step's containing scope has at least `30` later samples, at least `10.0` seconds
   from that step to the last later sample, and no later positive step.

The unchanged broader Metal lane contracts still require zero thermal throttling, OOM, admission
failure, resource-probe error, and resource leak on the normal no-growth path. The list above is
the additional complete safety envelope required to classify a positive step as host-global noise;
it must not be imposed as a new `4 GiB` floor on an `exact_zero` M1/M2/M3 lane.

The lane classification is machine-derived:

- `exact_zero`: all eight scopes contain zero positive swap steps. Here `exact_zero` means exact
  zero positive growth; the host's absolute swap-used value need not be zero.
- `host_global_swap_noise`: exactly one positive step exists across the lane and every condition
  above passes. The containing scope may expose the intermediate classification
  `host_global_swap_noise_candidate`.
- `swap_pressure`: any other result, including a second positive step, repeated or sustained
  growth, a step or cumulative positive growth above `1 MiB`, insufficient post-step stability,
  noise-path headroom below `4 GiB`, thermal throttling, OOM, admission failure, probe error, or
  incomplete evidence.

`exact_zero` and `host_global_swap_noise` satisfy only the swap portion of the Metal resource gate;
all other performance and correctness assertions must still pass. `swap_pressure` always rejects
the lane. A later decrease, a high final headroom value, or a passing throughput result cannot
convert `swap_pressure` to PASS.

## Completed-cell cumulative resume

The five HTTP cells have the fixed order `random:c1`, `random:c4`, `random:c16`,
`real-chat:c1`, `real-chat:c16`. Each cell independently executes its own required warmup followed
by all three measured repeats and the model/dataset-specific request counts. A completed cell may
be reused across a later clean server epoch of the same R2 or R3 lane so that an interruption does
not restart already accepted work.

Resume is mechanical and fail-closed:

1. Only a strictly contiguous prefix of the fixed five-cell order is eligible. A later completed
   cell cannot be cherry-picked across a missing, partial, or failed earlier cell.
2. Every reused cell must exactly match the lane config fingerprint; source SHA/tree, binary
   SHA256, model revision/file SHA256, hardware identity, dataset/seed/workload, effective typed
   config, benchmark argv and resource-sampling settings must all be unchanged.
3. The validator must deep-validate the cell report, complete warmup counts and quality fields,
   every measured-request record, all three repeat reports, the resource observation stream and
   terminal footer, the post-cell idle proof, every referenced artifact hash, and the prior epoch's
   server/process-group cleanup receipt. Cleanup must prove the old process group is gone and no
   server/model child remains.
4. A cell is reusable only after all required requests and three repeats completed with their
   ordinary correctness, performance-field, and resource evidence intact. A partial cell is
   discarded in full and rerun from its own warmup; partial requests or repeats never contribute
   to the final lane.
5. Each new epoch starts at the first cell after the validated prefix. It must not rerun a reused
   cell or silently skip an incomplete one. A prior epoch may end at a planned boundary or a
   bounded supervisory interruption after the checkpoint, but panic, OOM, product failure, missing
   cleanup, or incomplete evidence makes that checkpoint ineligible.
6. Every final cell record binds its real `session_epoch_id`, server PID/PGID, session timing,
   runtime log and resource evidence. The aggregate records all epochs in order and the actual
   server process count. It must not rewrite multi-epoch evidence as one continuous session or one
   server process.

Every epoch must also bind the actual collector, support module, and resource-sampler Git path/blob
SHA256 used to create it. The old epoch identities are locked by the frozen collection plan; each
new epoch's identities are locked by the current validation commit. Those byte identities need not
be equal across epochs because adding fail-closed resume or derived classification changes the
collector control plane.

For this amendment, a cross-epoch collector identity change may alter only collection orchestration
and derived validation. It may not replace or change the resource sampler: the old and new epochs
must bind the same exact resource-sampler SHA256, argv, interval and output-field contract.
Benchmark argv, workloads, request counts/repeats and immutable raw resource bytes also remain
unchanged. If the support-module identity differs, every affected raw header and provenance field
must bind its actual support identity and the current checked-in validator must deep-validate it.

The current checked-in validator performs that deep validation directly against the frozen plan,
epoch identities and immutable raw artifacts; no new qualification platform or standalone
equivalence receipt is required. A collector's self-reported equivalence is never sufficient.
Any benchmark/workload/resource-sampling semantic change, unbound identity, invalid raw header, or
raw mutation rejects reuse and starts a fresh affected lane.

Resume changes only collection scheduling. The final lane still contains exactly five independently
warmed HTTP cells with three measured repeats each plus three independent `ferrum run` samples. It
does not change any request count, denominator, threshold, ordering, raw byte, or eight-scope swap
aggregation. R2 and R3 maintain separate cumulative prefixes because their binary/source identity
requirements are separate. This is evidence reuse under the active
[`CHANGE_IMPACT_REGRESSION_PLAN_2026-08-12.md`](CHANGE_IMPACT_REGRESSION_PLAN_2026-08-12.md),
not a waiver.

## Provenance and raw-evidence integrity

The validator must recompute the classification from raw observations and save, for every scope:

- raw artifact path, byte size, SHA256, sampler/collector identity, and measurement-window
  derivation;
- first/last sample timestamp and swap-used value, sample count, positive-step count, maximum step,
  cumulative positive growth, and net change;
- every positive step's sample indexes, timestamps, before/after values, and delta;
- minimum physical headroom, post-step sample count and duration, and all thermal/OOM/admission/probe
  error counters;
- the threshold constants, scope classification, and explicit classification reason.

The lane aggregate must bind the ordered eight scope artifacts and their hashes, recompute the
lane-wide counts and maxima, and record the final classification. A summary without the raw
hash-bound observations is not evidence.

The collector and validator may derive a new classification artifact beside an otherwise eligible
existing raw artifact, but they must not edit, truncate, renumber, smooth, subtract from, or replace
any raw sample or old summary. There is no manual allowlist, command-line waiver, model-specific
exception, or post-hoc threshold. Accepting `host_global_swap_noise` is application of this common
machine rule, not a waiver.

This amendment does not authorize reuse across an invalid source, binary, model, hardware, config,
workload, sampler, or unbound collector/support identity. It only prevents a single tiny
host-global counter quantization step, proven under the complete safety envelope above, from being
misclassified as Ferrum memory pressure, and prevents already completed cells from being repeated
after a fully evidenced clean epoch boundary.
