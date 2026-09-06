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
  Neither can be replaced by another precision, smaller model, or backend.
- `required_targets` keeps advertised capability groups visible. Selection must
  consider stage and changed behavior; the number of profiles is not a gate.
  Additional precision-specific work belongs to affected loader/kernel changes.
- `checks` is empty: the catalog supplies no extra executable bindings. The
  regression-plan CLI adds the existing model runner's built-in capabilities;
  all other missing bindings remain gaps. Neither declaration nor model
  availability means a check ran or passed.

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

Each model obligation is assigned to one profile. If the initial selection lacks
a complete checker binding, the selector first adds an available, scope-compatible
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
runner reports. Quick Start uses its own normal alias, default backend selection
and disabled thinking with controlled prompts, sharing that profile's basic
cases. Selected HTTP checks share one server; separate run baseline/replay cases
retain their existing processes. There is no extra Quick Start model launch.

Missing or failed execution, wrong observed backend, mismatched task inputs and
unsupported model obligations fail this limited gate. The target remains a
declaration of architecture, precision and execution path; backend observation
does not prove every declared dimension or kernel. Success is ModelRuntime-only,
retains other plan gaps and never approves publication. Installation, numerical
checks, fresh downloads and unimplemented behavior checks remain separate.

## Current representatives

| Profile ID | Coverage contributed | Source |
|---|---|---|
| `quick-start-metal` | Dense hybrid, GGUF Q4_K_M, Metal | `qwen3.5:4b-q4_k_m` |
| `quick-start-cuda` | Dense hybrid, safetensors BF16 + F32, CUDA | `qwen3.5:4b` |
| `hybrid-moe-metal` | Hybrid MoE, GGUF Q4_K_S, Metal | `qwen3.5:35b-a3b-q4_k_s` |
| `attention-moe-metal` | Attention-only MoE, GGUF Q4_K_M, Metal | `qwen3:30b-a3b-q4_k_m` |
| `attention-moe-cuda` | Attention-only MoE, GPTQ INT4, CUDA | `qwen3-coder:30b-gptq` |
| `llama-dense-metal` | Llama dense, GGUF Q4_K_M, legacy Metal path | `llama3.1:8b-q4_k_m` |
| `llama-dense-cuda` | Llama dense, safetensors BF16, legacy CUDA path | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| `hybrid-dense-ct-int4-cuda` | Dense hybrid, compressed-tensors INT4, CUDA | `cyankiwi/Qwen3.8-27B-AWQ-INT4` |
| `hybrid-dense-block-fp8-cuda` | Dense hybrid, block-FP8, CUDA | `Qwen/Qwen3.8-27B-FP8` |
| `hybrid-moe-block-fp8-cuda` | Hybrid MoE, block-FP8, CUDA | `Qwen/Qwen3.6-35B-A3B-FP8` |
| `harmony-mxfp4-cuda` | GPT-OSS MoE, MXFP4, Harmony, CUDA | `openai/gpt-oss-20b` |
| `gemma-ct-w4a16-cuda` | Gemma dense, compressed-tensors W4A16, native thought, CUDA | `google/gemma-4-12B-it-qat-w4a16-ct` |

The [alias table](../crates/ferrum-cli/src/source_resolver.rs) supplies the GGUF
filenames and semantic sidecar repositories. The
[family registrations](../crates/ferrum-models/src/vnext/mod.rs) distinguish
production plan-runtime families from explicitly registered legacy families.
The Qwen3 Coder profile is an existing same-architecture alternative for CUDA;
it does not reproduce the original Qwen3 30B performance row.

The two dense block-FP8 snapshot rows, Qwen3.8 27B and Qwen3.6 27B, contribute the
same registered dense-hybrid execution group. Only one representative is listed.
If metadata, shapes, or the changed implementation distinguish their behavior,
add the resulting obligation and an appropriate profile; do not assume the
family name proves interchangeability.

## Source evidence and unresolved detail

- Qwen3.5 4B safetensors metadata at revision
  [851bf6e](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
  reports BF16 weights and a small F32 component. The catalog preserves both.
- The Llama CUDA representative reports BF16 weights at
  [a2856192](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/tree/a2856192dd7c25b842431f39c179a6c2c2f627d1).
  The Qwen3 Coder GPTQ alias resolves to
  [fd445bf6](https://huggingface.co/jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq/tree/fd445bf690e2f0b34337ec6ce3be95486550296a),
  whose declared architecture is `Qwen3MoeForCausalLM`.
- Qwen3.8's AWQ-labelled model declares `compressed-tensors`, not a generic
  AWQ loader contract; see its checked-in
  [config fixture](../crates/ferrum-models/tests/fixtures/qwen38_awq_int4_config.contract.json).
  The [FP8 fixture](../crates/ferrum-models/tests/fixtures/qwen38_fp8_config.contract.json)
  binds the separate dense block-FP8 recipe.
- The README links immutable sources for the FP8, GPT-OSS, and Gemma snapshots.
  Its first three performance rows do not identify their precise weight
  source/revision and quantization. The representatives above cover declared
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
`ferrum-metal` and `ferrum-cuda` self-hosted workers. Its Rust checker requires the
selected backend, completed GPU work and finite outputs within a declared
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
