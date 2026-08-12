# Runtime vNext v0.8.0 Metal host-global swap observation amendment (2026-08-13)

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

On macOS, `vm.swapusage` is a host-global counter. It is not attributable to the Ferrum process or
its process group, so neither a positive delta nor a sequence of positive deltas proves that Ferrum
allocated, retained, or swapped those bytes. The validator must preserve this signal transparently
as host-level diagnostic context without using its value, delta count, or growth shape to decide
PASS or REJECT.

This is a mechanism correction based on two bounded observations of the same M1 Metal lane, not a
loosening of a numeric limit. The first observation contained one `786,432`-byte positive step
while measured physical headroom remained above `14 GB`. A later clean-server epoch contained four
positive steps totaling `1,635,779` bytes, with a maximum individual step of `587,202` bytes and a
minimum measured physical headroom of `9,190,129,664` bytes. Both observations had zero thermal,
OOM, admission, and resource-probe failures. Because the counter covers unrelated host activity,
choosing a larger byte, step-count, headroom, or post-step-stability allowance after either
observation would remain a post-hoc threshold rather than process-attributable evidence.

One `(model, metal)` performance lane contains exactly these eight formal measurement scopes:

1. random HTTP c1;
2. random HTTP c4;
3. random HTTP c16;
4. real-chat HTTP c1;
5. real-chat HTTP c16;
6. `ferrum run` sample 1;
7. `ferrum run` sample 2;
8. `ferrum run` sample 3.

Missing, duplicate, unbound, or additional scopes make the lane REJECT as an evidence-completeness
failure. The host-global observation is aggregated across all eight scopes; it is not interpreted
independently as a resource verdict for any scope or cell.

For each scope, the validator reads the immutable, hash-bound raw resource observations inside the
formal measurement window, orders them by their recorded monotonic timestamp, and computes every
adjacent positive swap delta as:

```text
positive_delta_bytes = max(0, next_swap_used_bytes - current_swap_used_bytes)
```

Negative deltas never cancel or credit a positive delta. A scope's first sample is its baseline;
only observed adjacent samples inside that scope form steps.

## Mechanical acceptance rule

`vm.swapusage` is recorded and machine-classified, but there is no Metal resource-gate branch whose
outcome depends on its absolute value, positive-step count, maximum step, cumulative growth, net
change, or later decrease. The diagnostic classification is:

- `exact_zero`: all eight scopes contain zero positive swap steps. Here `exact_zero` means exact
  zero observed positive growth; the host's absolute swap-used value need not be zero.
- `host_global_swap_observed`: one or more positive steps exist across the lane. The aggregate must
  retain every step and all computed totals, without interpreting them as Ferrum memory pressure.

Malformed, missing, unhashed, or unbound raw observations still reject the lane as an evidence-
integrity failure. A host-global swap observation by itself does not. Conversely, `exact_zero`
does not satisfy, weaken, or substitute for a process-attributable resource gate.

The unchanged hard Metal resource gates are evaluated independently and remain fail-closed:

1. peak Ferrum process-group RSS must not exceed the lane's typed memory budget;
2. measured physical headroom must remain at least `2 GiB` (`2,147,483,648` bytes);
3. the thermal state must remain nominal and the power state normal;
4. OOM, admission-failure, and resource-probe-error counts must each remain `0`;
5. every HTTP cell must provide post-cell quiescence evidence, preserve the typed dynamic-pool
   ledger, and prove all transient resources returned to zero;
6. every `ferrum run` sample's process group must exit and leave no product child behind.

Every ordinary correctness, workload, performance, provenance, cleanup, and artifact-integrity
assertion also remains unchanged. Host-global swap can never mask or override failure of any hard
gate above, and passing any hard gate cannot erase or rewrite the recorded host-global observation.
The same mechanism and separation of diagnostic and hard signals applies to R3.

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

The validator must recompute the diagnostic classification from raw observations and save, for
every scope:

- raw artifact path, byte size, SHA256, sampler/collector identity, and measurement-window
  derivation;
- first/last sample timestamp and swap-used value, sample count, positive-step count, maximum step,
  cumulative positive growth, and net change;
- every positive step's sample indexes, timestamps, before/after values, and delta;
- minimum physical headroom and all thermal/OOM/admission/probe error counters;
- the scope classification and explicit classification reason that identifies `vm.swapusage` as a
  host-global, non-attributable diagnostic.

The lane aggregate must bind the ordered eight scope artifacts and their hashes, recompute the
lane-wide counts and maxima, and record the final diagnostic classification separately from the
hard resource-gate verdict. A summary without the raw hash-bound observations is not evidence.

The collector and validator may derive a new classification artifact beside an otherwise eligible
existing raw artifact, but they must not edit, truncate, renumber, smooth, subtract from, or replace
any raw sample or old summary. There is no manual allowlist, command-line waiver, model-specific
exception, or post-hoc threshold. Treating `host_global_swap_observed` as diagnostic-only is the
common machine rule for a non-attributable host counter, not an evidence waiver.

This amendment does not authorize reuse across an invalid source, binary, model, hardware, config,
workload, sampler, or unbound collector/support identity. It prevents an unowned host-global
counter from being misclassified as Ferrum memory pressure while preserving the process-
attributable hard gates, and prevents already completed cells from being repeated after a fully
evidenced clean epoch boundary.
