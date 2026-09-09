# Release regression

For the change-based selection policy, cost rules, implementation status and a
worked example, see [发布回归决策方案](release-regression-policy.zh.md).

This is a validation procedure, not a record of completed checks. Use the current
[English](../README.md) and [Chinese](../README_zh.md) README as the product contract.
Every Quick Start model must run on its advertised backend before release.
Sample performance-snapshot models by architecture, quantization and affected
execution path; an exhaustive model-by-backend matrix is not required.

## Generate the regression plan

The [product catalog](release-regression-catalog.md) declares Quick Start profiles
and advertised execution groups. Generate a plan from the **previous formal
release to the complete candidate**, including all merged changes:

```bash
cargo run --locked -p ferrum-bench-core --example regression_plan -- \
  --catalog docs/release-regression-catalog.json \
  --base v0.8.7 --candidate HEAD --stage release \
  --output /path/outside/repository/regression-plan.json \
  --summary /path/outside/repository/regression-plan.md
```

Replace `--base` with the preceding formal release for the candidate being
validated. For `--stage release`, the CLI requires the highest semantic version
among reachable `vMAJOR.MINOR.PATCH` tags, excluding tags on the candidate itself.
Fetch the relevant complete history and tags first. This follows the repository's
formal-tag convention; it does not query the GitHub Release API or prove a tag
was publicly released. `--stage pull_request` uses the PR base instead;
`--stage nightly` plans periodic inventory coverage without the formal-tag check.
Output files must be new. Git endpoints and the catalog digest identify planning
inputs; neither proves correctness.

The CLI compares Cargo inputs from those Git revisions. It narrows only the
manifest/lock changes that exactly match a coordinated version update produced
by the release preparer. Those paths retain build scope and release baseline
checks. Other paths in the same diff keep contributing their impact. Extra
dependency/build settings, unsupported membership or incomplete snapshots retain
the conservative classification, with an explanation. The path-only library API
does not perform this refinement.

The report envelope is now `schema_version: 2`, with `release_base_tag` and
`version_refinement` in provenance. Architecture and protocol obligation scopes
also carry `execution_path`; consumers must preserve that dimension. The
[scope analysis](../crates/ferrum-bench-core/examples/regression_plan/scope.rs) and
[version recognizer](../crates/ferrum-bench-core/src/release_regression/version_change.rs)
implement these rules.

Code PR CI generates this plan after compilation and before the workspace tests
inside its existing CPU job, then uploads it with a job summary. Documentation-only
PRs retain the inexpensive documentation checks; changed README promises also
appear in the next complete release diff. Compilation failures can prevent plan
generation and remain CI failures. Planning does not repeat the workspace suite or
allocate GPUs. Existing CPU, Metal and CUDA checks retain their required outcomes. Code CI also
requires the [real-device numerical lane](backend-numerics.md); the initial
RMSNorm checks do not satisfy the entire kernel or architecture obligation set.

A successful planning command means the input was parsed and the plan generated.
It is **not** a runtime pass or publication permission. Review every reported
coverage gap, selected and omitted profile, and unknown estimate. The current
catalog deliberately leaves costs unknown without measurements. The CLI now adds
built-in bindings for the model runner's implemented behaviors and emits
`model_tasks`; other obligations remain gaps. The model-task gate below consumes
actual runner results for that limited scope. Full-plan execution and verification
at the publication action remain pending; generating a plan does not do either.

The selector reserves each Quick Start profile, then adds representatives that
cover remaining obligations. Equal coverage is ordered by known estimated duration,
then by price when currencies match, then by stable profile identifier. Architecture
sampling retains architecture, protocol and execution path; protocol sampling also
retains backend. Different execution paths cannot represent one another.

Before assigning model obligations, the selector adds an available, scope-compatible
profile with a complete checker binding if the initial selection lacks one and
such a profile exists. It then assigns each obligation once, preferring a complete
binding before the estimate ordering. Missing bindings remain gaps. A shared check
is not assigned again to every compatible selected model. Quick Start obligations
remain bound to their individual profiles and cannot be filled by another profile.
The catalog's `checks: []` supplies no additional bindings. The CLI's built-in
model bindings are executable capabilities, not completed executions; unsupported
behaviors and other evidence layers still retain their gaps.

The selector does not infer currency conversion, cache availability, parallel
critical paths or an optimal monetary schedule. Unknown estimates remain listed.
Reported phase totals are sums of selected profile estimates, not measured
end-to-end release latency; GPU charges use declared billable duration separately
from preparation time. Neither planning nor assignment rents hardware or publishes.

## Prepare the candidate

Complete the workspace and relevant backend checks in [AGENTS.md](../AGENTS.md).
Build the release assets, then run model regressions against the extracted,
staged `ferrum` binaries that will be published. Building the regression runner
does not replace or rebuild the binary selected with `--ferrum-bin`.

Record the binary version and checksum, model repository/revision and weight
checksum, tokenizer/config sources, precision, backend, hardware and commands.
Keep reports, raw responses, logs and weights outside the repository. Reuse Cargo
and model caches; verify cached content against the selected upstream revision.
A cached weight must not hide stale tokenizer or chat-template sidecars. Record
which files were downloaded and which were reused; do not call reuse a fresh
download test.

Check binary architecture, accelerator features, driver/runtime dependencies,
available memory and storage before downloading large weights. CUDA assets target
sm89 and require compatible NVIDIA, CUDA and NCCL runtimes. Start paid hardware
when the candidate and test inputs are ready. Export evidence before releasing
temporary instances; retain paid storage only when its reuse justifies the cost.

## GGUF source inventory

Before choosing a quantized model sample, inspect every tensor in the exact
artifact. The Rust inventory tool reports sorted external names, logical shapes,
quantization block sizes, byte ranges, mixed dtypes, and split metadata:

```sh
cargo run --locked -p ferrum-quantization --example gguf_inventory -- /path/to/model.gguf
```

A downloaded prefix containing the complete GGUF header can be inspected without
downloading the weights. Supply the full artifact length from its pinned source:

```sh
cargo run --locked -p ferrum-quantization --example gguf_inventory -- \
  /path/to/model.header.gguf --file-size 5680522464
```

The example length is illustrative; use the actual selected artifact's length.
The tool rejects incomplete row blocks, overlapping or out-of-bounds tensor
ranges, and inconsistent split metadata. Its JSON distinguishes a caller-declared
length from a local file length and records that payload bytes were not verified
or materialized. It can inventory IQ3_S, IQ4_XS and IQ4_NL descriptors even though the
existing Candle runtime reader does not recognize those encodings.
An inventory proves neither the downloaded payload's hash nor model/backend
support. Record source revision and SHA-256 separately, inspect every shard when
present, and map external tensor names to the prepared family's actual roles.
Keep inventories and downloaded prefixes outside the repository.

## Mandatory Quick Start

The currently advertised paths are:

| Platform | MODEL | Source/format |
|---|---|---|
| Apple Silicon Metal | `qwen3.5:4b-q4_k_m` | `unsloth/Qwen3.5-4B-GGUF`, Q4_K_M; metadata from `Qwen/Qwen3.5-4B` |
| Linux CUDA | `qwen3.5:4b` | `Qwen/Qwen3.5-4B`, official safetensors |

Run the README's `--version`, `--help`, `doctor`, interactive `run`, and `serve`
commands, including `--disable-thinking` and `--served-model-name ferrum` where
shown. Submit a real prompt and the documented HTTP request; check meaningful,
non-empty output. Exercise both non-streaming and streaming responses.

Preserve the default context, memory and concurrency settings for these first-use
checks. A smaller-model substitution or capacity override is diagnostic evidence,
not completion of the advertised Quick Start. Capture startup/download failures,
timeouts, output errors and resource limits instead of silently changing flags.

## Rust model runner

Use the [model_regression development tool](../crates/ferrum-devtools/src/bin/model_regression.rs)
for repeatable checks against an explicit binary. For example, on Metal:

```bash
cargo build --release --locked -p ferrum-devtools --bin model_regression
./target/release/model_regression \
  --ferrum-bin /path/to/staged/ferrum \
  --model qwen3.5:4b-q4_k_m --backend metal \
  --report-dir /path/outside/repository/metal-quickstart \
  --checks basic,stop,structured,tools --disable-thinking --use-default-backend
```

Use `--model qwen3.5:4b --backend cuda` for the CUDA Quick Start and a separate
report directory. Reports must use a new or empty directory. `--checks` defaults
to `basic`; explicitly select the checks relevant to the release:

- `basic`: two `run` turns; server non-streaming and streaming answers plus
  actual assistant-history replay through both HTTP modes.
- `stop`: termination behavior through both entrypoints, including stream text.
- `structured`: server JSON/schema behavior and valid structured responses.
- `tools`: server tool calls and a tool-result continuation.

For a fixed Hugging Face snapshot, `ferrum run`, `ferrum serve`, `ferrum pull`,
and the runner accept `OWNER/REPOSITORY@FULL_40_HEX_COMMIT` as the model.
This selects the exact snapshot through Ferrum's downloader and cache; aliases,
branches and tags cannot be used as pins. The runner verifies the actual
repository, revision and file fingerprints reported by both product entrypoints.

For a GGUF repository, `pull`, `run` and `serve` accept `--gguf-file FILE` to
select one repository-relative artifact, including a file in a subdirectory.
The same selection works for a fresh download or a cache containing several
quantizations. An immutable repository pin applies to the selected file:

```sh
ferrum pull unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523 --gguf-file Qwen3.5-4B-Q4_K_M.gguf
ferrum run unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523 --gguf-file Qwen3.5-4B-Q4_K_M.gguf --disable-thinking
```

Semantic configuration and tokenizer files resolve independently from colocated
metadata or the GGUF's declared source repository. `run` and `serve` also accept
`--semantic-source DIR` and `--tokenizer-source DIR` for explicit metadata roles.
A bare repository reuses one unambiguous cached GGUF; without a cached selection,
choose an exact file instead of downloading every quantization.

The model runner also forwards `--gguf-file FILE` to both entrypoints and requires
an immutable `--model` pin with this option. A prepared `ModelProfile` can declare
the selected artifact and separate metadata expectations:

```json
"gguf": {
  "filename": "Qwen3.5-4B-Q4_K_M.gguf",
  "semantic_source": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
}
```

An optional `tokenizer_source` is another pinned repository; when omitted it uses
the semantic expectation. Local and cloud task runners forward the filename and
reject unpinned expectations before executing tasks. The report verifier checks
the actual weight file, all three repository revisions, and their observed file
fingerprints from `run` and `serve`. Metadata expectations assert what the product
resolved; they do not override source selection. If upstream metadata changes,
the task fails. A standalone probe without a prepared profile records and validates
metadata identities but does not claim that their revisions were fixed in advance.

A stop probe derives an internal boundary from actual reasoning or final text.
Baseline and replay keep the same prompt, mode and budget; checks require the
exact nonempty prefix, no stop leakage and fewer generated tokens. If the default
probe stops inside reasoning, a separate `--disable-thinking` probe must establish
a real final-text boundary. The flag alone does not prove reasoning is absent.
Failure to cover the final channel remains a failure. `--stop-prompt` explicitly
sets the shared workload text and is bound to the prepared task.

For reasoning-history compatibility, add `--reasoning-alias-replay` to a run
including `--checks tools`. Preserve the model's reasoning defaults for that run
by omitting `--disable-thinking`. It must replay actual, non-empty reasoning;
an empty reasoning field or a fabricated trace does not cover the behavior.
Choose a capable model and report a failure to produce that trace explicitly.

The runner complements literal README commands and installation checks. With
`--use-default-backend`, `--backend` is the expected observation: child commands
omit the explicit backend flag and remove inherited `FERRUM_*` overrides. Command
records list removed key names only. Other environment settings, including PATH,
CUDA device visibility and Hugging Face cache/authentication, remain inherited.
Without this option the child receives `--backend` and retains its environment.
Every run-ready event must report the requested model selector and matching actual
backend; server readiness requires `status: healthy` and checks
`auto_config.hardware_capabilities.backend`. HTTP 200 alone is insufficient.
Missing, unknown or mismatched backend observations fail the check.

Sampling uses temperature 0, seed 7 and a controlled output budget (`--max-tokens`,
default 512). Quick Start retains default capacity. Other prepared functional tasks
use `--context-tokens 2048 --max-num-seqs 1`, binding the public KV/context and
concurrency options in both entrypoints and checking their actual configuration.
Explicit-capacity `run` also disables context shifting so an oversized request
cannot silently shrink its output budget or discard history. This is not download,
general answer quality, throughput or numerical reference evidence.
All selected HTTP cases share one server process. `basic` uses one separate run
process; `stop` uses its existing baseline and replay run processes. A Quick Start
binding reuses that profile's basic cases, without another model load.

## Verify selected model tasks

The Rust [model_gate example](../crates/ferrum-bench-core/examples/model_gate.rs)
prepares typed expectations from the plan's assigned model obligations and staged
asset metadata. It groups checks by profile, preserves each Quick Start's own
profile, and rejects unsupported model obligations before hardware allocation.
It does not start runners or rent hardware itself. For example, using the actual
candidate version and adjacent staging metadata:

```bash
cargo run --locked -p ferrum-bench-core --example model_gate -- \
  prepare --plan /path/outside/repository/regression-plan.json \
  --version 0.8.8 \
  --abi /path/to/staged/ferrum-macos-aarch64.tar.gz.abi.json \
  --abi /path/to/staged/ferrum-linux-x86_64-cuda-sm89.tar.gz.abi.json \
  --output-dir /path/outside/repository/model-tasks
```

Supply the staged binary for each selected backend; paths and `0.8.8` above are
illustrative. The output directory must be new. Each `task-N.json` fixes the
profile, model selector, declared target, binary digest, version, checks, thinking
mode, backend-selection mode and sampling inputs. Run the existing example once
per task with `--expected-task /path/to/model-tasks/task-N.json` and its matching
`--model`, `--backend`, `--checks` and other options. The runner supplies the task's
profile ID if omitted, checks configuration before loading, and verifies its
terminal report against the expectation. It does not silently apply other task
options: choose them explicitly from the prepared task. For a bound local model,
prepare the task with its canonical absolute path; the runner canonicalizes the
supplied path before comparison and rejects a mismatch before loading. Public
aliases retain their ordinary selector. A failed binary-version check also stops
before model loading.

Quick Start tasks retain the normal README alias, require automatic backend
selection and `--disable-thinking`, and use the controlled prompts above. They
exercise the default capacity and template configuration in the same basic
run/serve cases; installed `--help`, `doctor`, literal README requests and fresh
download checks remain separate obligations.

Then supply every actual schema-2 runner report:

```bash
cargo run --locked -p ferrum-bench-core --example model_gate -- \
  verify --tasks /path/outside/repository/model-tasks/tasks.json \
  --report /path/outside/repository/metal-model/report.json \
  --report /path/outside/repository/cuda-model/report.json \
  --output /path/outside/repository/model-gate.json
```

Use as many reports as the prepared tasks require; the examples do not define a
fixed matrix. Missing, duplicate, unfinished, failed or mismatched reports fail.
The runner's Rust answer, stop, JSON, tool and SSE assertions determine each case
result. The gate checks those terminal results and their task/identity bindings;
it does not independently replay the raw responses or authenticate arbitrary
hand-edited reports. Counts, PASS text and binary hashes do not replace the
semantic assertions. The target's architecture/precision/execution-path fields
remain declared configuration; observing the backend does not measure every
kernel or establish those fields independently.

The output is explicitly scoped to `model_runtime`, retains remaining plan gaps
and always records `release_approved: false`. Current bindings cover model load,
basic forward/natural completion, Quick Start, history, user stop, structured
validity and tool-result continuation in their implemented entrypoints. Tool
selection/handoff, reasoning-alias coverage, length boundaries, scheduling/KV,
performance and other unimplemented obligations remain unsupported. This gate
cannot approve the complete release, installation, CI or backend-numerical work.

## Representative coverage and cost

Use the Quick Start dense hybrid model as one representative. Add cached MoE
models when scheduling, batching or resource ownership changed; choose an
attention-only or recurrent MoE path according to the affected code. Include
concurrent generation, late prefill, cancellation and a subsequent clean request
when those behaviors changed, using focused Rust tests or bounded model runs.

GPT-OSS is a useful separate representative for Harmony, reasoning channels and
tool handoff. Gemma is useful for its distinct template and compressed-tensors
path. Neither must be added merely to increase a model count. Sample AWQ, GPTQ,
MXFP4 or block-FP8 when the relevant loader/kernel changed; several related FP8
checkpoints do not all need to be downloaded for an unrelated protocol fix.

Prefer existing verified weights and one sufficiently sized GPU used sequentially.
For sharded models, fetch the files referenced by the selected index and required
sidecars; avoid duplicate original-format copies. Record the representatives,
omitted groups and reasoning so the limits of coverage remain visible.

## Interpret results and publish

Compilation proves code builds. Fixture tests prove the exercised protocol and
boundary behavior. Real-model runs additionally test loading, backend execution
and the specific semantic assertions used; HTTP 200 alone is not answer validity.
Keep failures and unexecuted checks visible. Do not substitute repetition totals,
PASS ratios or machine/commit identifiers for assertions about actual behavior.

Performance claims require same-hardware comparisons with recorded workloads,
precision, input/output lengths, concurrency, repetitions and uncertainty. Count
valid outputs and errors alongside timing; SSE text events are not usage tokens.
Correctness on a different GPU does not reproduce a README throughput number.

In bench reports, `actual_input_tokens_per_request` records client-tokenized
prompt content. `server_input_tokens_per_request` separately records
`usage.prompt_tokens`, including the server's chat template. Missing server
usage stays null; older reports omit the field. Release latency comparisons
require observed server input lengths, verify their context budget and totals,
and require the same rendered lengths for baseline and candidate.

Configure release latency limits in
[`policy.limits`](../.github/release-performance.json) before starting a release.
The shared Mac temporarily allows 30% TTFT and 20% time-per-output-token
increases; restore both limits to `0.1` when using a stable dedicated host.
Each limit also bounds both ends of the baseline A/A 95% confidence interval.
Candidate comparisons pass only when the upper confidence bound is within the
configured increase; wider uncertainty remains inconclusive. These relaxed limits
permit regressions up to 30% TTFT and 20% time per output token; they do not establish
performance within the original 10% limits. Correctness checks are unchanged.
A changed policy applies to new measurements; earlier reports retain their original
policy and conclusions and are not reclassified.

For a manual release retry, `reuse_cuda_run_id` can name a completed main-branch
release run. The workflow rechecks its original successful CUDA execution,
artifact digest, product and test-runner bytes, complete task inputs and model
reports before reusing evidence. Any mismatch falls back to normal cloud
regression. Performance is measured again under the newly registered policy.

Resolve failures in mandatory Quick Start paths and selected required regressions
before publishing. Publish the validated staged bytes and verify the public
tarballs and checksums.
Check the advertised Homebrew formulas and crates.io installation command as well
as direct downloads: installed `--version`, `--help` and `doctor` must work with
their stated dependencies. Preserve the mandatory model-runtime evidence for
each backend, and disclose any installation or runtime check still incomplete.
