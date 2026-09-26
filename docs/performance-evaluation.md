# Performance evaluation

Ferrum's project-wide performance objective is:

**Maximize output throughput subject to meeting TTFT, TPOT, and ITL latency SLOs.**

This applies to backend, kernel, scheduling, memory, and serving optimizations.
Correct output and valid protocol behavior remain prerequisites. Microbenchmarks
help explain an implementation; adoption requires end-to-end evidence for the
intended workload and latency limits.

Using the same weight file does not establish identical arithmetic or output
quality. Disclose weight, KV, activation and accumulation policies, and validate
any additional approximation against the declared quality requirements before
including it in a performance claim. A finite argmax agreement test alone does
not establish quality equivalence with another implementation.

Memory is a declared capacity constraint and a required measurement. Within that
constraint, spending more memory for lower latency or higher throughput is a
valid tradeoff; lower memory usage does not compensate for a failed latency or
throughput requirement.

## Qualifying a configuration

Declare the workload and the TTFT, TPOT, and ITL thresholds and percentiles before
comparing candidates. Use P99 for tail-latency evaluation unless the scenario
explicitly declares another percentile. There is no universal latency threshold
for all hardware, models, input lengths, and applications.

First establish that all three latency SLOs are satisfied, then maximize output
throughput among qualifying configurations. A higher-throughput configuration
that violates a required SLO does not qualify. If none qualify, report that no
tested configuration meets the SLO. Do not select a winner on throughput alone.

Missing thresholds, unbounded thresholds used by a reporting tool, missing timing
evidence, or insufficient samples cannot establish compliance. Mark these cases
as **not evaluated** or **insufficient evidence**, rather than passing them.
Report each repetition's status; averaging percentiles must not conceal a failed
repetition. State sample counts and uncertainty, especially for P99 estimates
from small exploratory runs.

The initial optimization reference is llama.cpp on the same hardware, model,
workload, and client concurrency. Until a workload has absolute latency limits,
use its llama.cpp TTFT, TPOT, and ITL P99 values as the relative latency reference:
all three must be no worse before ranking Ferrum's output throughput. Repeat
measurements to distinguish improvements from noise. Label this a relative
comparison; even the baseline may have unacceptable interactive latency.

## Primary dataset

Use ShareGPT as the primary serving workload. The starting dataset is
[`ShareGPT_V3_unfiltered_cleaned_split.json`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/745745adf6cd15b84e4f1c4a5a051fb4304f9342/ShareGPT_V3_unfiltered_cleaned_split.json),
also used in the [vLLM benchmark examples](https://docs.vllm.ai/en/latest/benchmarking/cli/).
Pin its revision and file hash in the measurement evidence and keep the data
outside the repository. ShareGPT names a dataset, not a complete test protocol.

Freeze the sampled conversation IDs, seed, order, tokenizer, and output-length
policy. Use exactly the same samples across implementations and concurrency
cells. Declare whether requests use the first user/assistant pair or multi-turn
history. For first-pair length replay, preserve the user text and derive each
request's output budget from the reference assistant answer's token length;
report actual generated usage separately. Disclose EOS handling. A fixed output
override is a different workload and must be labeled explicitly.

Report input/output length distributions and the counts and reasons for excluded
records. A context-limited subset is a declared length bucket, not the complete
dataset. Do not silently truncate long prompts, repeat samples, or substitute
fixed random input/output lengths. Synthetic workloads remain useful for kernel,
scheduler, and regression diagnosis but do not replace the primary ShareGPT
comparison.

## Required comparison table

Use one row per implementation/configuration and client concurrency:

| Implementation / configuration | Concurrency | TTFT P50 / P99 (ms) | TPOT P50 / P99 (ms/token) | ITL, visible text updates P50 / P99 (ms) | Output throughput (tokens/s) | Peak GPU allocated (GiB) | Peak OS footprint (GiB) | Maximum RSS (GiB) | SLO status |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| Measured configuration | Measured concurrency | Measured values | Measured values | Measured values | Measured value | Measured value and source, or not collected | Measured value and source, or not collected | Measured value and source, or not collected | All three pass / violated metrics / not evaluated |

Attach successful/attempted requests, errors, output validation, sample counts,
warmup policy, repetitions, and the actual thresholds to the table. Keep
exploratory measurements separate from repeated validation. Preserve raw results
outside the repository, including failures and unfavorable runs.

Always show the memory columns, including in status summaries. Name the GPU API
(for example, Metal `currentAllocatedSize`) and mark a sampled maximum as sampled.
An unavailable value must remain **not collected** in its own column; an OS
memory measurement must not silently replace it.

TTFT measures client-observed time from request submission to the first nonempty
visible text update, including network, queueing, and input processing. The
legacy benchmark calculates per-request TPOT as
`(terminal time - TTFT) / (output tokens - 1)`. The versioned SLO sidecar and
`slo-compare` use `(last visible text time - TTFT) / (usage output tokens - 1)`.
Both take percentiles across eligible requests; keep these distinct endpoint
definitions explicit and identical across the comparison arms.

For the primary serving table and SLO, **ITL means client-observed time between
successive non-empty visible text updates** (SSE text events). Pool these observed
intervals within each measurement repeat to calculate P50/P99; disclose request
and interval counts. Role-only, empty-text, usage-only, and finish-only messages
do not count as visible updates. A text event may contain several model tokens:
this metric measures visible output stalls and is not a reconstruction of the
generation time of every token. Do not divide event gaps by usage-token counts.

Retain strict single-token timing eligibility as a separate diagnostic. Disclose
event/usage mismatches and transport coalescing; they must not erase observed
visible-text stalls. Requests with fewer than two visible updates have no such
interval and must be disclosed. Protocol failures remain failures. Older reports
without the visible-text interval statistics are **not collected** for this
metric; do not relabel their strict-token aggregates or zero placeholders as the
new measurement. Both comparison arms must use the same client and definition.

Output throughput is successful output usage tokens divided by the complete
measurement window, including scheduling and stalls. It is an aggregate service
rate, not a single user's generation rate. Do not improve the reported latency by
omitting queueing, failed requests, rejected load, or long gaps. Disclose any
admission/rejection policy and report failures alongside successful latencies.

## Compare frozen evidence

`ferrum slo-compare` reads existing artifacts and produces the same comparison
as JSON and English/Chinese Markdown. Keep these artifacts outside the source
repository. Output parents must already exist, and all three output files must
be new, distinct files:

```text
ferrum slo-compare /path/to/evidence/manifest.json \
  --out /path/to/results/comparison.json \
  --markdown-en /path/to/results/comparison.en.md \
  --markdown-zh /path/to/results/comparison.zh.md
```

For a larger complete matrix plus original pilot, add
`--limits-config /path/to/evidence/limits.json`. This version-1 JSON file can
override individual typed reading limits; omitted fields keep current defaults:

```json
{
  "schema_version": 1,
  "limits": {
    "max_files": 1024,
    "max_total_bytes": 536870912,
    "max_total_visible_gaps": 8000000
  }
}
```

These are resource examples, not workload or performance thresholds. Default
total limits are 256 referenced files, 256 MiB and 4 million visible gaps; hard
ceilings are 4,096 referenced files, 1 GiB and 16 million gaps. Main and pilot
evidence share these cumulative budgets. `max_files` counts distinct referenced
file paths; the outer manifest is separately bounded by `max_manifest_bytes`
and still counts toward `max_total_bytes`. Other fields and defaults are defined by
[`ArtifactLoadLimits`](../crates/ferrum-bench-core/src/slo_comparison/artifact/types.rs),
with hard ceilings checked by its `validate()` method. The configuration must
be a regular JSON file of at most 16 KiB; unknown fields, invalid integers and
values beyond hard ceilings are rejected before output creation. Outputs cannot
overwrite this input either. Capacity settings do not alter the frozen scope,
SLO/ratio thresholds, provenance checks or bootstrap computation ceilings; keep
the complete matrix and pilot when sizing them.

The version-1 manifest references the frozen contract and each declared
concurrency/pair's baseline and candidate artifacts. Each file reference records
its relative path, exact byte size, and SHA-256. Sidecars come directly from
`bench-serve --slo-client-config ... --slo-out ...`; references select a zero-based
nonempty JSONL record and a zero-based repeat, plus its ordered ShareGPT selection
hash. Execution files contain acquisition-time run/process, hardware, model,
configuration and measurement-window declarations. The loader verifies these
bindings and file integrity; hashes do not attest that hardware actually ran.
The typed manifest is defined in
[`artifact/types.rs`](../crates/ferrum-bench-core/src/slo_comparison/artifact/types.rs).

Native Metal memory JSONL is read directly from `--device-memory-jsonl`; peaks,
sample counts and gaps are reconstructed from raw samples and checked against
the final summary. Versioned external device/footprint/RSS observation formats
are accepted, but this command does not implement those external collectors.
Missing or incomplete memory coverage remains `Unknown`, with available raw
observations labeled unverified. Each arm's absolute SLO status is separate from
the comparison result.

Reports are saved before exit **3** (requirements not met) or **4** (unknown,
inconclusive, or descriptive-only evidence). Load/configuration/output errors
use **1**, and CLI usage errors use **2**. Code **0** requires the frozen contract
to pass the raw-pilot eligibility and approximate simultaneous inference path
described below. It is a decision under declared experimental assumptions, not
an unconditional statistical guarantee. Paired means and observed ranges alone
remain descriptive; no fixed repetition count establishes proof. The existence
of this command does not establish Qwen3.5-9B superiority over llama.cpp.

### Paired uncertainty and independent-pilot eligibility

The optional typed method `paired-cluster-percentile-bonferroni-v1` computes
approximate one-sided bootstrap bounds from the validated original paired
measurements. Its estimand is the **arithmetic mean of paired candidate/baseline
ratios**, exactly as in the descriptive table. It is not a ratio of means, mean
log ratio, median, or count of winning repetitions. Construct a declaration with
`FrozenStatisticalMethod::paired_cluster_bootstrap(FrozenPairedBootstrap { ... })`;
the API computes the canonical configuration SHA-256, which comparison validates
again. Existing external method declarations remain unverified.

Freeze the seed, resample count, family error budget, Monte Carlo error budget,
all seven relative precision targets, complete primary cell range, and all
paired repetitions before candidate measurements. `declared_design` records an
earlier independent pilot digest/time, planned pairs, required request/gap
support, alternating AB/BA arm order, and the independent-block restart/warmup
protocol. These fields are **design declarations, not verified sufficiency**.
Choose sample size and repetition allocation from an independent baseline pilot;
do not change them after seeing candidate performance. This distinction follows
the experimental-design concerns in [Kalibera and Jones (2013), author's
repository](https://kar.kent.ac.uk/33611/).

Each resample selects whole paired repetitions with replacement, using the same
indices for all metrics and primary cells. All pairs must replay the same ordered
ShareGPT samples; complete cross-cell paired blocks must follow frozen order,
with nonoverlapping measurement windows and alternating arm order. Timestamps
can reject inconsistent acquisition but cannot prove independence or thermal
reset. Resampling also assumes exchangeable paired blocks: residual arm-order
effects or time drift need pilot assessment, even with alternating AB/BA order.
Requests and pooled visible gaps within a repetition are never counted as
independent repetitions. Resampling follows the empirical-distribution approach
of [Efron (1979)](https://doi.org/10.1214/aos/1176344552).

The family contains seven comparisons for every primary cell. After subtracting
the Monte Carlo budget, divide the remaining alpha by this full family size.
Report upper bounds for six latency ratios and a lower bound for successful
usage-token throughput. Numerical threshold clearance requires upper `< limit`
or lower `> limit`; equality is insufficient. Correlated metrics/cells are
retained, and the family adjustment uses the [Bonferroni
inequality](https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm).
Diagnostic cells remain visible but are outside the predeclared primary family.

The random stream is versioned SHA-256 counter output with unbiased rejection
mapping. Finite simulation uses an outward order-statistic rank rather than
interpolating a sparsely sampled tail. For `B` draws, per-comparison bootstrap
tail `p`, and Monte Carlo failure allowance `delta`, use
`k = floor(B*p - sqrt(2*B*p*ln(1/delta))) - 1`: kth largest for an upper bound,
kth smallest for a lower bound. If `k < 1`, the tail is unresolved and no bound is
reported. This conservative rank follows a binomial lower-tail bound; it controls
simulation error **conditional on the empirical bootstrap distribution**, not
population coverage. See [Chernoff–Hoeffding bounds, Dubhashi and Panconesi
(2009)](https://doi.org/10.1017/CBO9780511581274).

The implementation checks real resource limits before resampling: at most 4,096
pairs, 1,000,000 resamples, 64 primary cells, 4,000,000 stored floating-point
values, and 64,000,000 metric accumulations. These are CPU/memory limits, not
statistical qualification thresholds. Missing/invalid observations are never
imputed or removed to obtain an interval. A single cluster, constant empirical
ratios, inadequate declared P99 support, or unresolved simulation tails cannot
produce a credible narrow interval.

Without original independent pilot evidence, intervals retain
`calibration: unverified`, and numerical clearance remains `Inconclusive`, with
exit 4. Caller-supplied intervals or hashes cannot unlock `ProofPass`. Percentile
bootstrap coverage is approximate and can be poor with few independent clusters;
see [Hall (1988)](https://doi.org/10.1214/aos/1176350933). Many gaps from one short
run do not fix that problem. These intervals concern repeat variability of the
fixed workload's empirical P50/P99 statistics; they do not prove a population
P99 for all ShareGPT prompts.

The optional manifest field `eligibility` references a `FrozenEligibilityPlan`
JSON file and an original A/A `pilot_manifest`, using the same
`{ "path": ..., "sha256": ..., "bytes": ... }` file-reference format. **All**
references in both manifests resolve relative to the outer comparison manifest
directory. The pilot cannot reference another eligibility manifest. Main and
pilot files share the loader's cumulative byte, file and visible-gap limits;
the loader does not reset its budget when entering the pilot.

Freeze the planning document before the pilot. It declares an increasing grid
of prospective paired-run allocations, simulation count/seed, empirical P99 rank
resolution, acquisition diagnostic tolerances, and the independent-block
protocol. The original pilot must use identical baseline binaries/configuration
in both arms, the same hardware, model, precision, ordered ShareGPT workload,
client boundary, absolute SLOs and complete primary family as the main study.
The final bootstrap configuration binds the planning document through
`eligibility_plan_sha256` and the pilot manifest through
`declared_design.pilot_source_sha256`; final freezing follows pilot completion
and precedes candidate acquisition.

The verifier recomputes the whole frozen planning grid from original A/A paired
ratios, with shared cross-cell draws. It requires the final allocation to equal
the first point meeting every declared relative precision target. This is an
empirical precision forecast, not a coverage claim or a promise that the
candidate will have the pilot's variance. The comparison entrypoint checks an
already frozen allocation; it is not an independent planning CLI. Planning
checks the cumulative cost across every grid point before resampling: at most
128 grid points and 128 million metric accumulations, in addition to the
per-allocation bounds above. These remain resource ceilings rather than
statistical minimum sample sizes.

P99 support is reconstructed from raw requests/gaps. Declared empirical rank
steps must resolve the upper one-percent tail and must also be enforced by the
main study's support contract. Rank resolution does not establish an IID
population-quantile interval. Pilot diagnostics report both A/A ratio directions,
arm-order log shifts, time trends and lag-one dependence; they check frozen
tolerances and within-pair idle boundaries. In particular, `E[A/B]` is not
generally 1 merely because A and B are identically distributed. These diagnostics
do not prove independence or absence of all drift. Simulation-based planning is
conditional on its empirical data-generating model, as distinguished from
universal method guarantees in [Morris, White and Crowther
(2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/).

`verify_inference_eligibility` issues a non-deserializable
`VerifiedInferenceEligibility` only after recomputation; serializing its report
does not create a reusable success certificate. `compare_with_eligibility`
recomputes the main experiment, rejects reuse of pilot runs or a changed
contract, and requires every primary simultaneous bound to strictly clear its
limit **and** meet actual precision. Only then can `ProofPass` and exit 0 occur.
The report records `eligible_under_declared_assumptions`: independent,
exchangeable complete paired blocks and approximate bootstrap validity remain
explicit assumptions. Missing, insufficient, degenerate or invalid evidence
cannot pass. The verifier does not claim to prove real-world IID, unconditional
coverage, semantic output quality or performance on other configurations.

## Capacity and memory

Keep the server's capacity and configuration fixed while changing client
concurrency. A different scheduling policy or capacity is a separately identified
configuration with its own sweep. Record the model and precision, input/output
lengths, dataset, server/client versions, commands, cache state, and intentional
differences between implementations. Use the same hardware for comparisons and
retain thermal, memory-pressure, and swapping observations.

On discrete GPUs, report peak VRAM with the measurement source, sampling interval,
and process or device scope. A sampled maximum is not an exact allocation high
water mark. On Apple Silicon, report peak process memory footprint and maximum
RSS separately, with their accounting scope; do not label either as dedicated
VRAM or total model working set. Clean file-backed mappings and shared Metal
buffers can make footprint differ substantially between implementations. A lower
footprint alone does not demonstrate a lower total memory requirement. Collect
Metal allocation peaks separately when available. Metal allocation accounting
and process memory may overlap and must not be added together. Never infer a peak
from an end-of-run snapshot, runtime budget, or model file size.

Specify whether a peak covers load, warmup, and measurement or only measurement.
For a per-process lifetime peak, start a fresh server with the same configuration
for each concurrency cell. On macOS, `/usr/bin/time -l` can report the actual
server child's peak memory footprint and maximum RSS after it exits. Keep its
raw output and ensure the measured child is the server, not a shell wrapper.
Report unavailable memory measurements as **not collected**.

`ferrum slo-compare` can import original macOS `/usr/bin/time -l` output without
pretending its process lifetime peaks are periodic observations. Freeze
`memory.os_footprint` as `{"kind":"macos_time_l_v1"}` and
`memory.maximum_rss_window` as `"process_lifetime"`; the existing sampled policy
(`window`, `interval_ms`, `max_sample_gap_ns`) remains compatible. The generated
English and Chinese tables explicitly label whole-process lifetime peaks,
including load, warmup, measurement and shutdown. Metal sampling remains separate.

For each arm, keep `memory.device` as before, omit `footprint` and `rss`, and add:

```json
"process_lifetime": {
  "format": "macos_time_l_v1",
  "capture": {"path": "candidate/process-memory.json", "sha256": "<SHA-256>", "bytes": 1234}
}
```

The referenced strict JSON capture has `schema_version: 1`, the same `identity`
as the arm's execution artifact (`benchmark_run_id`, `cell_id`, `repeat_index`,
`server_pid`), `subject: "declared_direct_server_child"`, declared
`process_started_unix_ns` / optional `process_ended_unix_ns`, and `time_output`
pointing to the original native text. Optional `exit_status` references the
original decimal shell exit status file; optional `launch_evidence` references
the original command/process-listing evidence. Every reference uses the same
`path` / `sha256` / `bytes` contract, relative to the outer comparison manifest,
and counts toward the shared main-plus-pilot file and byte limits. Example hashes,
sizes and timestamps must be replaced with actual acquisition values.

Capture the native output in a separate file with `LC_ALL=C` and retain the
actual exit result. The importer reads `maximum resident set size` and `peak
memory footprint` as bytes, and accepts no manually supplied peak or completion
flag. Duplicate fields, invalid integers, overflow, explicit/invalid units,
corrupt references and contradictory identities fail loading. Missing metrics,
missing exit/end evidence, or nonzero exit leave the memory result **Unknown**;
any parsed peak remains an explicitly unverified diagnostic. A lifetime footprint
cannot satisfy a frozen sampled-footprint policy. The parser does not establish
that the measured child really was the declared server; the acquisition-time
declaration and original command/process evidence remain auditable provenance,
not hardware attestation. These metadata declarations must be accurate, with
declared lifetime bounds enclosing the benchmark. Importing memory evidence
alone establishes neither latency/TPS success nor statistical eligibility.

These are evaluation requirements, not a claim that every current benchmark
command automatically enforces them. A tool's existing `goodput` field alone
does not establish all three latency SLOs.
