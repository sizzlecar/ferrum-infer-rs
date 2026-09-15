# Product regression catalog

[release-regression-catalog.json](release-regression-catalog.json) describes
current product execution groups, using the
[English](../README.md) and [Chinese](../README_zh.md) README as the contract.
It is planning input, not a record of passed tests or a benchmark baseline.
The [release procedure](release-regression.md) still requires the final staged
binary and the actual installation and first-use paths.

## Meaning of the fields

- `profiles` are concrete model selectors that can represent execution groups.
  `available: true` means a run can be arranged; it does not mean weights are
  cached, a GPU is allocated, loading has succeeded, or a check has passed.
- `target` distinguishes architecture, output protocol, weight precision or
  format, backend, and the actual production execution path. These are coverage
  dimensions, not GPU model identifiers.
- `precision` describes stored weights. It does not assert that activations,
  accumulation, KV state, and every tensor share that dtype.
- `estimate: null` means execution cost has not been measured for the planned
  check. It is not zero cost. Record download, load, test, and paid resource costs
  separately when preparing the run.
- `quick_start_profile_ids` contains two independent mandatory profiles.
  Both are pinned 4B Q4_K_M GGUF profiles, one per backend. Neither can be
  replaced by another precision, smaller model, or backend. Metal has only
  `release-qwen35-4b-gguf-metal` as its Quick Start profile.
- `release_profile_ids` retains explicit model commitments beyond README examples.
  Release planning requires each enabled named profile; a cheaper model with the
  same architecture cannot replace it. The explicit CUDA and Metal policies
  below determine which release profiles are enabled. An unavailable or missing
  enabled profile is a gap.
  PR and nightly sampling do not automatically repeat these release commitments.
- `release_cuda` separates mandatory pinned local representatives from optional
  extended cloud profiles. Every CUDA profile belongs to exactly one lane.
  The committed `cloud: "disabled"` default can be changed to `required` only by
  explicit release opt-in. Cloud-only model-runtime obligations are disclosed as
  `extended_not_run` when disabled, not reported as passing. This policy cannot
  defer numerical, contract or installation obligations, remove local tasks, or
  change PR/nightly coverage.
- `release_metal` separates three mandatory local profiles from larger extended
  profiles. Every Metal profile belongs to exactly one lane. Extended Metal
  model-runtime obligations remain in `extended_not_run`; there is no flag to
  enable their execution in this release lane. They have not been regressed by
  this lane and are not passing evidence. This policy does not defer numerical,
  safety or installation checks, change PR/nightly coverage, or allow dense
  models to represent MoE model behavior.
- `required_targets` keeps advertised capability groups visible. Selection must
  consider stage and changed behavior; the number of profiles is not a gate.
  Additional precision-specific work belongs to affected loader/kernel changes.
- `checks` is empty: the catalog supplies no extra executable bindings. The
  regression-plan CLI adds the existing model runner's built-in capabilities;
  all other missing bindings remain gaps. Neither declaration nor model
  availability means a check ran or passed.

The current default commitments include pinned Qwen3.5 4B GGUF on CPU/CUDA/Metal
and Qwen3.5 2B SafeTensors on CPU/Metal. Local Metal additionally requires pinned
Llama 3.1 8B Q4_K_M GGUF. Its 9B, 27B, 30B attention-only MoE and 35B hybrid MoE
profiles remain unexecuted extended coverage, not release passes. Local CUDA
continues to require Qwen3.5 0.8B and Llama3.2 1B SafeTensors on the RTX 4050.
CUDA 9B/27B and the other large CUDA profiles remain an explicit cloud extension;
the Metal policy does not change that opt-in.
Each GGUF profile declares its exact file and independent semantic source;
execution must observe those sources. These are planned checks, not completed
support claims. Existing CUDA SafeTensors and other advertised architecture and
encoding representatives remain in the catalog.

Resolve aliases through the production resolver. Record the selected immutable
weight revision and the tokenizer/config/template revisions in runtime evidence.
A revision below explains a declaration; it is not a mandatory commit identifier
for future tests. Upstream changes require the corresponding source checks.

## Scope and assignment semantics

The current CLI report uses `schema_version: 2`. Its architecture scopes contain
`architecture`, `protocol` and `execution_path`; protocol scopes contain
`protocol`, `backend` and `execution_path`. A production plan-runtime target
cannot stand in for a legacy executor just because the architecture or protocol
matches. Architecture baseline sampling may cross backends within the same
execution path; a separate obligation retains actual execution on each backend.
Affected compute/resource changes still use exact target combinations.

Each enabled model obligation is assigned to one profile. Excluded extended
release obligations remain in `extended_not_run`, not in passing local results.
If the initial selection lacks a complete checker binding, the selector first
adds an available, scope-compatible
profile with a complete binding when one exists. It then assigns the obligation
once, preferring a complete binding, followed by declared estimates and a stable
ID. Missing bindings remain gaps. Other compatible profiles do not implicitly
repeat that obligation. Each Quick Start has its own profile scope and cannot be
merged with another Quick Start; the same profile may also satisfy shared obligations.
These are planned assignments, not evidence that any command ran.

Production request metadata and journal/profile/trace sinks contribute an
`observability` contract and one model-runtime representative per backend.
That does not infer kernel-numerical coverage or repeat every architecture and
precision. Unknown modules remain conservative; the crate name or an `examples`
directory alone is not evidence of validation-only use.

For Cargo inputs, the CLI can narrow only a content-proven coordinated version
update to build scope. Other changed paths remain in the same accumulated impact;
additional dependency/build changes or an unsupported snapshot retain the broad
classification. The report records the refinement decision and release tag
baseline in provenance. See the [release procedure](release-regression.md).
The catalog itself remains unchanged by that report schema revision. Its
`checks: []` does not remove the CLI's built-in model bindings, and bindings
outside that supported set still need an executable checker. Planning does not
rent hardware, execute the catalog or publish a release.

## Model task bindings

The [model schedule](../crates/ferrum-bench-core/src/release_regression/model_schedule.rs)
connects only the runner's implemented behaviors: basic model load/forward/natural
completion, Quick Start, history replay, user stop, structured validity and
canonical tool-result continuation. Each binding lists its real entrypoints;
tool selection/handoff, reasoning-alias replay, length limits, scheduling/KV and
performance obligations are not satisfied by those nearby checks.

The CLI emits `model_tasks` alongside the plan. The
[model-task gate](release-regression.md#verify-selected-model-tasks) groups assigned
checks by profile without selecting replacement models. It uses staged metadata
to prepare an expected task before execution, then consumes actual schema-2
runner reports. Quick Start retains its own profile, default backend selection
and disabled thinking with controlled prompts, sharing that profile's basic
cases. Selected HTTP checks share one server; separate run baseline/replay cases
retain their existing processes. There is no extra Quick Start model launch.

The mandatory Metal tasks are `release-qwen35-4b-gguf-metal`,
`release-qwen35-08b-safetensors-metal` and `llama-dense-metal`. The first is the
sole Metal Quick Start: its entire assigned task retains product capacity
defaults. The latter two bind typed `ModelRunCapacity` values of 2048 context
tokens, one sequence and `runtime_memory_budget_bytes: 10737418240` (10 GiB),
forwarded to both `run` and `serve`; the controlled output budget remains 512
tokens. This runtime planning budget is not an RSS hard limit or proof of fit.
The actual staged-candidate CI execution on the local release worker must establish
the tested capacity and model behavior. Timeout, OOM and assertion failures
remain failures. See the [Metal lane procedure](release-regression.md#required-local-metal-model-lane).

Missing or failed execution, wrong observed backend, mismatched task inputs and
unsupported model obligations fail this limited gate. The target remains a
declaration of architecture, precision and execution path; backend observation
does not prove every declared dimension or kernel. Success is ModelRuntime-only,
retains other plan gaps and never approves publication. Installation, numerical
checks, fresh downloads and unimplemented behavior checks remain separate.

## Current representatives

| Profile ID | Declared coverage | Source (see catalog for pinned identities) | Release lane |
|---|---|---|---|
| `release-qwen35-4b-gguf-metal` | Dense hybrid, GGUF Q4_K_M, Metal | `unsloth/Qwen3.5-4B-GGUF` | Mandatory local; sole Metal Quick Start |
| `release-qwen35-08b-safetensors-metal` | Dense hybrid, SafeTensors BF16 + F32, Metal | `Qwen/Qwen3.5-0.8B` | Mandatory local |
| `llama-dense-metal` | Llama dense, GGUF Q4_K_M, legacy Metal path | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | Mandatory local |
| `release-qwen35-9b-gguf-metal` | Dense hybrid, GGUF Q4_K_M, Metal | `unsloth/Qwen3.5-9B-GGUF` | Extended, not run |
| `release-qwen38-27b-gguf-metal` | Dense hybrid, mixed GGUF 4-bit, Metal | `unsloth/Qwen3.8-27B-GGUF` | Extended, not run |
| `hybrid-moe-metal` | Hybrid MoE, GGUF Q4_K_S, Metal | `qwen3.5:35b-a3b-q4_k_s` | Extended, not run |
| `attention-moe-metal` | Attention-only MoE, GGUF Q4_K_M, Metal | `qwen3:30b-a3b-q4_k_m` | Extended, not run |
| `release-qwen35-4b-gguf-cuda` | Dense hybrid, GGUF Q4_K_M, CUDA | `unsloth/Qwen3.5-4B-GGUF` | Mandatory local; CUDA Quick Start |
| `release-qwen35-08b-safetensors-cuda` | Dense hybrid, SafeTensors BF16 + F32, CUDA | `Qwen/Qwen3.5-0.8B` | Mandatory local |
| `release-llama32-1b-safetensors-cuda` | Llama dense, SafeTensors BF16, legacy CUDA path | `unsloth/Llama-3.2-1B-Instruct` | Mandatory local |
| `release-qwen35-08b-gguf-cpu` | Dense hybrid, GGUF Q4_K_M, CPU | `unsloth/Qwen3.5-0.8B-GGUF` | Mandatory CPU |
| `release-qwen35-08b-safetensors-cpu` | Dense hybrid, SafeTensors BF16 + F32, CPU | `Qwen/Qwen3.5-0.8B` | Mandatory CPU |
| `quick-start-cuda` | Dense hybrid, SafeTensors BF16 + F32, CUDA | `qwen3.5:4b` | Cloud extension; not a current Quick Start binding |
| `attention-moe-cuda` | Attention-only MoE, GPTQ INT4, CUDA | `Qwen/Qwen3-30B-A3B-GPTQ-Int4@9b534e4318b7ebc3c961a839f13eb18b1833f441` | Cloud extension |
| `llama-dense-cuda` | Llama dense, SafeTensors BF16, legacy CUDA path | `unsloth/Meta-Llama-3.1-8B-Instruct` | Cloud extension |
| `hybrid-dense-ct-int4-cuda` | Dense hybrid, compressed-tensors INT4, CUDA | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | Cloud extension |
| `hybrid-dense-block-fp8-cuda` | Dense hybrid, block-FP8, CUDA | `Qwen/Qwen3.8-27B-FP8` | Cloud extension |
| `hybrid-moe-block-fp8-cuda` | Hybrid MoE, block-FP8, CUDA | `Qwen/Qwen3.6-35B-A3B-FP8` | Cloud extension |
| `harmony-mxfp4-cuda` | GPT-OSS MoE, MXFP4, Harmony, CUDA | `openai/gpt-oss-20b` | Cloud extension |
| `gemma-ct-w4a16-cuda` | Gemma dense, compressed-tensors W4A16, native thought, CUDA | `google/gemma-4-12B-it-qat-w4a16-ct` | Cloud extension |

This inventory declares potential coverage, not execution results. Extended
Metal rows are explicitly unexecuted; cloud CUDA rows require the existing
opt-in. A small dense-model pass does not qualify either Metal MoE architecture
or any larger model's end-to-end behavior or capacity.

The [alias table](../crates/ferrum-cli/src/source_resolver.rs) supplies the GGUF
filenames and semantic sidecar repositories. The
[family registrations](../crates/ferrum-models/src/vnext/mod.rs) distinguish
production plan-runtime families from explicitly registered legacy families.
The Qwen3 CUDA profile uses the original M3 GPTQ checkpoint and immutable
revision from the v0.8.0 source lock. Its functional checks do not reproduce
the original performance workload.

The two dense block-FP8 snapshot rows, Qwen3.8 27B and Qwen3.6 27B, contribute the
same registered dense-hybrid execution group. Only one representative is listed.
If metadata, shapes, or the changed implementation distinguish their behavior,
add the resulting obligation and an appropriate profile; do not assume the
family name proves interchangeability.

## Source evidence and unresolved detail

- Qwen3.5 4B safetensors metadata at revision
  [851bf6e](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
  reports BF16 weights and a small F32 component. The catalog preserves both.
- The mandatory Llama Metal GGUF pins
  [bartowski revision 4f0c246f](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/tree/4f0c246f125fc7594238ebe7beb1435a8335f519)
  and the exact file `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`. Its independent
  semantic and tokenizer source is
  [Unsloth revision 0f68888b](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/tree/0f68888b9c8c55098bea14b4225237ebab9b1f5b),
  which contains `config.json`, `tokenizer_config.json` and `tokenizer.json`
  for Llama 3.1 8B Instruct. These source identities do not prove a runtime pass.
- The Llama CUDA representative reports BF16 weights at
  [a2856192](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/tree/a2856192dd7c25b842431f39c179a6c2c2f627d1).
  The Qwen3 CUDA representative pins
  [9b534e43](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/tree/9b534e4318b7ebc3c961a839f13eb18b1833f441),
  the historical M3 checkpoint with `Qwen3MoeForCausalLM`, F16 dense weights,
  and symmetric GPTQ INT4 groups of 128. Its config, tokenizer config, and
  quantization config hashes match the historical source lock.
- Qwen3.8's AWQ-labelled model declares `compressed-tensors`, not a generic
  AWQ loader contract; see its checked-in
  [config fixture](../crates/ferrum-models/tests/fixtures/qwen38_awq_int4_config.contract.json).
  The [FP8 fixture](../crates/ferrum-models/tests/fixtures/qwen38_fp8_config.contract.json)
  binds the separate dense block-FP8 recipe.
- The README links immutable sources for the FP8, GPT-OSS, and Gemma snapshots.
  Its first three performance rows do not themselves identify precise weight
  source/revision and quantization. The historical M3 source lock identifies
  the Qwen3 CUDA checkpoint above. Other representatives cover declared
  execution groups; they do not reconstruct those benchmark inputs.
- Reproducing a snapshot therefore needs its original source, workload,
  configuration, hardware, and baseline evidence. Until supplied, performance
  comparison remains an explicit unassigned check, regardless of a semantic
  model run passing.

## Evidence limits

Existing CPU tiny-Llama tests exercise real model arithmetic on the legacy
engine, with mock engine allocation helpers. Server `tiny_stack_wire` uses a
stub engine. Neither supplies production plan-runtime or GPU numerical evidence.
The `op_diff` Marlin and paged-varlen modules are planning stubs; an unavailable
backend is not a successful comparison. Existing Metal numerical tests can
return early when no device exists. Their selected checks must produce actual
execution and comparison evidence rather than relying on Cargo's exit status.
Compilation, protocol assertions, model semantics, numerical comparison, and
performance measurements remain separate obligations.

## Current CI hardware boundary

The [workflow](../.github/workflows/ci.yml) runs CPU tests on standard
`ubuntu-latest`. Its `macos-latest` job enables Metal tests, but that does not
prove GPU execution: device-dependent tests can return early without a device.
The CUDA Ubuntu container type-checks the CUDA CLI and optionally builds PTX;
it does not execute CUDA operators.

The separate `GPU runtime` matrix schedules repository-controlled code on the
self-hosted Metal and CUDA workers. Metal device Quality selects the
`ferrum-metal` pool (the local Mac and the 16 GiB MacBook Pro). Release models
and Homebrew verification select the local Mac's `ferrum-metal-release` role.
Mac mini 2 is unregistered. One service per Mac serializes that host's jobs;
different Macs do not share a Metal concurrency lock. The CUDA jobs retain
their shared physical-host lock. Model tasks run serially, with a 3600-second
per-task timeout and a 240-minute model job limit. Those labels and limits select
and schedule resources; they do not prove model fit or success. CUDA retains
the mandatory local RTX 4050 lane and explicit cloud extension.
The Rust checker requires the selected backend, completed GPU work and finite
outputs within a declared
numerical tolerance. Missing devices, missing reports and failed or skipped GPU
jobs cannot satisfy `CI required` for code changes. See the
[commands and current coverage](backend-numerics.md).

This initial lane checks RMSNorm only, with aligned and tail dimensions. It does
not bind all catalog kernel obligations, validate complete architectures or
replace real-model and final-asset checks. The catalog's unassigned checks remain
unassigned until each actual execution path has an appropriate executable binding.

GitHub's [standard runners are free for public repositories](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
Its [larger runner options](https://docs.github.com/en/actions/reference/runners/larger-runners)
include GPU-accelerated macOS xlarge and NVIDIA Tesla T4 runners; compatibility
with the selected Ferrum binary and workload still needs verification.
[Larger runners remain billable for public repositories](https://docs.github.com/en/billing/reference/actions-runner-pricing).
