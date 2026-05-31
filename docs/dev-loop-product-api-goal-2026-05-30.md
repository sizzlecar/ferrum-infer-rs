# Goal: Shorten Performance Iteration Loop and Harden Product API

**Status:** draft @ 2026-05-30  
**Owner:** ferrum core  
**Source review baseline:** local HEAD `cb95d05`  
**Primary scope:** build iteration speed, profiling/benchmark discipline, runtime config clarity, auto configuration, OpenAI-compatible product API

---

## Objective

Make Ferrum easier to optimize and safer to expose as a product server by fixing structural issues that slow the loop:

1. bottleneck localization is too dependent on ad hoc env vars, hidden switches, and log grep;
2. small kernel edits can still trigger broad CUDA rebuild/link cycles;
3. M3 A/B validation scripts duplicate fragile server/process/artifact logic;
4. API compatibility is bench-oriented rather than full OpenAI-compatible;
5. strict JSON/schema/tool-calling behavior is not represented in the internal request/response model;
6. best-known runtime paths are encoded as shell env bundles instead of a validated auto-selection policy.

This goal is achieved only when a developer can make a narrow kernel/API/scheduler change, validate it with a scoped harness, and get comparable artifacts without manually reconstructing environment state from shell scripts and logs.

## Acceptance Summary

| Area | Current pain | Target |
|---|---|---|
| CUDA edit loop | attention-only edits can take minutes to tens of minutes depending on touched files | attention-only rebuild p95 `<= 90s`; unrelated Marlin/MoE-Marlin objects are not rebuilt |
| Profiler output | scripts grep human log lines | all required profile data emitted as JSONL or chrome trace with schema tests |
| Bench harness | many copied `m3_*_ab.sh` scripts | one reusable runner covers A/B, gates, cleanup, metadata, and summary |
| Runtime config | `FERRUM_*` switches are scattered across scripts, Rust hot paths, and CUDA/C++ code | central env registry plus typed config; hot-path direct env reads are reduced by `>= 85%` and every knob is visible in artifacts |
| Auto configuration | best M3 path is assembled from many manual env switches | vLLM-style typed config builder selects backend, graph, scheduler, and KV settings from model + hardware + workload and emits a decision trace |
| OpenAI API | chat works for smoke/bench, missing tools and several fields | SDK contract tests cover tools, stream options, JSON schema, usage, errors |
| Strict structured output | `json_object` is soft-biased; schema support is partial | unsupported strict schema is rejected; supported strict schema is validated before response |
| Correctness and regression gates | perf experiments can pass throughput while silently breaking behavior or another cell | every default-path change has mandatory correctness gates and no material perf regression |

## Milestone A: Incremental CUDA Build Boundary

### Required Changes

- Split the PTX build path in `crates/ferrum-kernels/build.rs` into per-kernel content-hash artifacts instead of one `bindgen_cuda` batch for every core `.cu`.
- Keep Marlin, vLLM-Marlin, vLLM-MoE-Marlin, vLLM paged-attn, and FA2 source builds behind independently cached artifacts.
- Add a build summary line for every CUDA artifact: `built`, `cache_hit`, `reason`, `elapsed_ms`, `inputs_hash`.
- Add a documented fast path for attention-only iteration that does not compile Marlin or MoE-Marlin unless their inputs changed.

### Quantitative Acceptance

- After a clean release CUDA build, touching only `crates/ferrum-kernels/kernels/paged_varlen_attention_vllm.cu` and rebuilding `ferrum-cli` with the M3 feature set completes in:
  - p50 `<= 75s`
  - p95 `<= 90s`
  - measured across 5 consecutive rebuilds on the restored RTX 4090 pod.
- The same attention-only rebuild logs `cache_hit` for:
  - `marlin`
  - `vllm_marlin`
  - `vllm_moe_marlin`
  - `vllm_paged_attn` unless that exact library is the edited target.
- Touching one vLLM-MoE-Marlin source file rebuilds only the vLLM-MoE-Marlin static library plus final Rust link; it does not rebuild PTX core kernels.
- `cargo check -q -p ferrum-kernels --features cuda,vllm-paged-attn-v2,vllm-moe-marlin,fa2-source` still succeeds on the GPU pod.

## Milestone B: Structured Profiling and Artifact Schema

### Required Changes

- Replace primary grep-only profile lines (`[unified-prof]`, `[bucket-prof]`, `[iter-prof]`, `[graph-prof]`, `[MOE_DUMP:*]`, `[vllm-moe-config]`) with structured JSONL events.
- Keep human log lines only as a convenience layer; scripts must consume JSONL or chrome trace.
- Define a stable schema for profile events in `ferrum-bench-core`, including:
  - `event`
  - `commit_sha`
  - `env_hash`
  - `model`
  - `concurrency`
  - `shape`
  - `stage_us`
  - `graph_enabled`
  - `runtime_flags`
- Emit profile output path in the bench artifact manifest.

### Quantitative Acceptance

- `scripts/m3_route_unified_profile.sh` no longer greps server logs for required fields; it fails only by validating structured events.
- A c32 profile run produces at least these structured event groups:
  - `moe_dump`
  - `vllm_moe_config`
  - `unified_prof`
  - `unified_layer_prof` or `batched_decode_prof`
  - `iter_prof`
  - `bucket_prof` when MoE profiling is enabled
- JSON schema/unit tests reject missing required profile fields.
- Profile parser tests cover at least 3 fixture artifacts: default graph-on, graph-off route dump, and FA2/FA-layout attention A/B.
- Profile overhead remains bounded:
  - graph-on c32 throughput with structured low-intrusion profiling disabled changes by `< 1%` versus current default;
  - graph-off sync-timer profile is explicitly labeled as non-throughput data.

## Milestone C: Unified A/B Bench Runner

### Required Changes

- Replace copied `scripts/m3_*_ab.sh` logic with a single reusable runner that supports:
  - case matrix definitions;
  - env per case;
  - server launch and health wait;
  - process cleanup;
  - Paris gate;
  - multi-turn gate;
  - optional structured profile validation;
  - `bench-serve`;
  - artifact manifest;
  - same-binary A/B summary.
- Keep existing scripts as thin wrappers or delete them after migration.
- Add a preflight check that records GPU process state, git status, binary hash, feature set, and effective runtime config.

### Quantitative Acceptance

- At least 4 existing M3 scripts are migrated to the runner:
  - FA2 direct/source A/B;
  - FA-layout A/B;
  - route/unified profile;
  - one scheduler/admission A/B.
- Every run writes `manifest.json` with:
  - `git_head`
  - `git_status_short`
  - `binary_sha256`
  - `features`
  - `env_hash`
  - `case_env`
  - `model_dir`
  - `server_log`
  - `bench_json`
  - `profile_jsonl`
  - `correctness_gates`
  - `cleanup_status`
- Failed correctness gates stop before throughput measurement.
- Failed server health or interrupted runs leave no live `ferrum`, `cargo`, `nvcc`, `vllm`, or benchmark client process owned by the script.
- A same-binary N=3 c32 A/B can be started with one command and yields a summary table with mean, stddev, CI95 half-width, TTFT p50, TPOT p50, ITL p95, completed, and errored.

## Milestone D: Runtime Config Registry and Snapshot

### Required Changes

- Add a single environment-variable registry for all `FERRUM_*` knobs. Each entry must declare:
  - name;
  - type;
  - default;
  - owner crate/module;
  - scope: `runtime`, `benchmark`, `debug`, `build`, or `test`;
  - stability: `default`, `experimental`, `diagnostic`, or `deprecated`;
  - read phase: `build`, `startup`, `request`, or `test-only`;
  - replacement config key or CLI flag when applicable;
  - sunset condition for experimental and diagnostic knobs.
- Parse performance-affecting runtime env vars once into typed config structs and pass those structs through engine/model/backend boundaries instead of reading `std::env::var` throughout hot code.
- Replace CUDA/C++ `std::getenv` dispatch switches with typed launcher parameters or one-time init state. Remaining `getenv` reads must be diagnostic-only and registry allowlisted.
- Move common M3 benchmark defaults into named presets so scripts set a preset plus only case-specific overrides.
- Preserve explicit debug overrides, but make defaults, sources, and effective values visible.
- Include config snapshot in:
  - `/health` or a new diagnostics endpoint;
  - `bench-serve` reports;
  - structured profile events;
  - artifact manifests.

### Quantitative Acceptance

- Baseline measured on 2026-05-30:
  - `143` unique `FERRUM_*` names across `crates/`, `scripts/`, root `Cargo.toml`, and `ferrum.toml`;
  - `269` direct env reads across the same scope (`std::env::var`, `env::var`, `var_os`, `std::getenv`, `getenv`);
  - hot core paths (`ferrum-engine/src`, `ferrum-models/src`, `ferrum-kernels/src`, `ferrum-kernels/kernels`) contain `116` unique `FERRUM_*` names and `175` direct env reads.
- Registry coverage is `100%` for all `FERRUM_*` names found by static scan.
- CI/static check fails if a new `FERRUM_*` name appears without a registry entry and parser test.
- Direct env reads in hot core paths are reduced by at least `85%` from the `175` baseline, to `<= 26`.
- No direct env read is allowed inside per-token, per-layer, scheduler-decision, kernel-launch, or request-handling hot loops except allowlisted diagnostic probes that are cached before the loop.
- CUDA/C++ source has `<= 5` direct `getenv` reads, all diagnostic or build/probe related and registry allowlisted.
- All default-on behavior for M3 is represented in typed config:
  - `FERRUM_MOE_GRAPH`
  - `FERRUM_VLLM_MOE`
  - `FERRUM_VLLM_MOE_PAIR_IDS`
  - `FERRUM_USE_VLLM_PAGED_ATTN`
  - `FERRUM_VLLM_PAGED_ATTN_V1_SHORT`
  - `FERRUM_GREEDY_ARGMAX`
  - `FERRUM_FA_LAYOUT_VARLEN`
  - `FERRUM_FA2_SOURCE`
  - `FERRUM_FA2_DIRECT_FFI`
- No known no-op or placebo runtime switch remains undocumented. Deprecated aliases must warn once, map to the typed config, and have a removal target.
- M3 A/B scripts use a named runtime preset for common defaults and set `<= 3` case-specific env overrides per case, excluding port/path/log variables.
- Config snapshot is stable sorted JSON and includes for every non-default value:
  - key;
  - effective value;
  - source: default, config file, CLI, env, or script case;
  - whether it affects correctness, performance, memory, or diagnostics.
- Config snapshot diff between two A/B cases is machine-readable and included in the bench summary.
- Unit tests cover env parsing for boolean, integer, path, and tri-state default/forced-off/forced-on cases.

## Milestone E: vLLM-Style Auto Configuration and Backend Selection

### Reference Pattern

vLLM does not make the user discover the fast path by manually composing dozens of env vars. Its normal path is:

- parse CLI/config/env into typed engine config;
- infer model properties from model metadata;
- inspect platform and GPU capabilities;
- choose compatible backends from priority-ordered selectors;
- profile available GPU memory at startup and size the KV cache;
- allow manual overrides, but validate them and log why they changed the automatic choice.

Ferrum should adopt the same control-plane shape while keeping Ferrum-specific kernels, M3 constraints, and correctness gates.

### Required Changes

- Add a `FerrumConfigBuilder` or equivalent startup resolver that combines:
  - CLI args;
  - config file values;
  - registry-backed env overrides;
  - model metadata;
  - hardware capabilities;
  - workload preset.
- Add typed capability structs:
  - `ModelCapabilities`: architecture, quantization, MoE shape, context length, head dim, KV heads, supported dtypes, graph-safety constraints;
  - `HardwareCapabilities`: backend, CUDA runtime, compute capability, VRAM, SM count, supported dtypes, graph support, available compiled kernel features;
  - `WorkloadProfile`: serving mode, target concurrency, prompt/output length class, latency/throughput priority.
- Add selector modules with explicit candidate lists, validation, and priority order for:
  - attention prefill/mixed path;
  - attention decode path;
  - MoE route/GEMM path;
  - CUDA graph policy;
  - scheduler/admission policy;
  - KV cache sizing and max batched tokens.
- Emit an auto-config decision trace at startup and into artifacts. Each selected field must include:
  - selected value;
  - source: default, CLI, config file, env, model metadata, hardware capability, memory profile, or workload preset;
  - candidates considered;
  - rejected candidates with reasons;
  - whether the selection affects correctness, performance, memory, or diagnostics.
- Keep manual overrides, but validate them. Invalid combinations must fail fast or explicitly degrade with a warning and decision-trace reason.
- Treat diagnostic paths separately from product defaults:
  - runtime-loaded vLLM/Torch FA2 direct FFI can remain diagnostic-only;
  - source-built/Ferrum-owned FA2 or native kernels are eligible for auto selection only after correctness and non-regression gates pass.

### Ferrum M3 Target Behavior

- A single named preset, for example `m3_qwen3_30b_a3b_int4`, should replace the common M3 env bundle currently repeated across scripts.
- With the M3 preset and no case-specific overrides, the builder should select the current safe default equivalents for:
  - CUDA backend;
  - device-route MoE;
  - vLLM-Marlin MoE when available;
  - pair-id MoE combine path;
  - greedy GPU argmax;
  - paged attention;
  - graph-clean decode policy;
  - prefix cache off for the current benchmark profile unless explicitly enabled.
- If FA-layout or FA2-source paths are compiled and validated, the selector may choose them for M3 only when:
  - the dependency is Ferrum-owned or source-built, not a runtime vLLM/Torch extension;
  - the selected path has a current same-pod correctness and non-regression artifact;
  - the decision trace names the artifact used to justify default selection.

### Quantitative Acceptance

- Running the M3 server with the named preset requires `<= 2` performance-affecting env vars in scripts, excluding paths, ports, logging, and artifact destinations.
- At least `90%` of the current M3 default/fast-path switches are expressed as typed selector outputs rather than direct env-controlled branches.
- Startup writes `effective_config.json` and `decision_trace.jsonl`; schema tests validate both files.
- Decision trace covers at least these selections:
  - attention prefill/mixed backend;
  - attention decode backend;
  - MoE implementation;
  - MoE graph policy;
  - KV block count or KV memory budget;
  - max sequences;
  - max batched tokens;
  - scheduler chunk/admission policy;
  - sampling/readback path.
- Invalid override tests cover at least 10 combinations, including:
  - graph enabled with a graph-unsafe MoE path;
  - FA2 selected without compiled/source-built support;
  - BF16 selected on unsupported hardware;
  - max batched tokens smaller than max sequences;
  - KV cache budget too small for the requested max model length.
- The M3 named preset without the old env bundle is performance equivalent to the old explicit-env default:
  - c=1,4,16,32 full sweep with `n_repeats >= 3`;
  - no cell regresses by more than `3%`;
  - c32 does not regress by more than `2%`;
  - Paris single-turn and multi-turn gates pass.
- Auto-selection must not hide performance experiments:
  - every experimental candidate remains opt-in until it has a publishable artifact;
  - selector priority changes require the correctness and non-regression gates from Milestone I.

## Milestone F: Product API Compatibility

### Required Changes

- Extend OpenAI chat request/response support to include:
  - `tools`
  - `tool_choice`
  - assistant `tool_calls`
  - tool messages
  - legacy `functions` / `function_call` compatibility if needed by SDKs
  - `stream_options.include_usage`
  - `logprobs` / `top_logprobs` rejection or implementation with correct error semantics
  - tokenizer-based usage accounting
  - `n` either implemented or rejected when `n != 1`
- Replace internal `InferenceRequest { prompt: String }` as the only product boundary with a structured request that can carry messages, tool definitions, tool results, and response constraints.
- Keep prompt rendering in a model-family chat-template layer, not in the HTTP handler.
- Implement `/v1/completions` or remove the claim of compatibility for that endpoint from product docs.

### Quantitative Acceptance

- Add non-ignored API contract tests using a stub or deterministic model for:
  - basic chat;
  - streaming chat;
  - `stream_options.include_usage`;
  - `n=2` rejection or two-choice response;
  - tool call request parsing;
  - assistant tool call response serialization;
  - tool role message parsing;
  - unsupported multimodal content returning 400 instead of silent drop;
  - `logit_bias` rejection or implementation;
  - completions endpoint behavior.
- Add ignored real-model SDK tests for `async-openai` and one Python OpenAI SDK compatibility smoke.
- Usage accounting uses tokenizer counts, not whitespace counts:
  - prompt token count differs from tokenizer result by `0` on test fixtures;
  - completion token count equals generated token count.
- Error responses map correctly:
  - invalid request -> HTTP 400, `type=invalid_request_error`;
  - unsupported feature -> HTTP 400 or 422 with clear `param`;
  - engine unavailable -> HTTP 503;
  - internal generation failure -> HTTP 500.
- OpenAI API compatibility report documents each supported/unsupported field with tests linked.

## Milestone G: Strict JSON and Schema Output

### Required Changes

- Treat `response_format.json_schema.strict=true` as a contract:
  - if the schema subset is supported, enforce it with hard masking and validate returned JSON;
  - if unsupported, reject at request boundary.
- Treat `json_object` as best-effort unless hard JSON grammar masking is implemented; document behavior accurately.
- Add response validation before non-streaming return and final streaming completion when feasible.

### Quantitative Acceptance

- Supported strict schemas pass 100 consecutive deterministic stub-model tests without invalid JSON.
- Unsupported strict schemas reject with HTTP 400 and `param=response_format.json_schema`.
- Real-model smoke for a simple object schema passes 20/20 runs at temperature 0.
- No markdown fence stripping is needed for strict schema success in the deterministic test path.
- Existing `response_format=json_object` tests remain green, but product docs explicitly mark it as best-effort until a full JSON grammar mask is shipped.

## Milestone H: Codebase Shape and Ownership

### Required Changes

- Split oversized modules along operational boundaries:
  - `qwen3_moe.rs`: model state, scratch allocation, MoE config, and profiling counters separated;
  - `qwen3_moe_forward_unified.rs`: attention plan, MoE plan, final sampling/readback, and graph/profile wrappers separated;
  - `continuous_engine.rs`: request state, scheduler loop, batch materialization, streaming, and completion handling separated;
  - `backend/traits.rs`: capability traits moved to smaller files.
- Convert long kernel launch signatures into typed parameter structs where the same group of fields recurs.

### Quantitative Acceptance

- No single Rust source file in these core paths exceeds `1500` lines:
  - `ferrum-engine/src/continuous_engine.rs`
  - `ferrum-models/src/models/qwen3_moe.rs`
  - `ferrum-kernels/src/backend/traits.rs`
- `qwen3_moe_forward_unified.rs` is `<= 700` lines or split by stage.
- No new function in model/backend hot paths has more than `15` parameters unless it is a low-level FFI boundary.
- Existing local gates pass:
  - `cargo fmt --all -- --check`
  - `cargo check -q -p ferrum-cli`
  - `cargo test -q -p ferrum-engine --test continuous_batch_test`
  - `cargo test -q -p ferrum-scheduler`

## Milestone I: Correctness and Performance Non-Regression Gates

### Required Changes

- Add a mandatory validation checklist for every change that touches:
  - CUDA kernels or CUDA build logic;
  - model forward paths;
  - scheduler/admission policy;
  - sampling / structured output;
  - OpenAI server request or response types;
  - runtime default flags.
- Make benchmark artifacts record both correctness gates and performance regression gates, not just throughput.
- Define a stable default-path baseline artifact before changing defaults.
- Keep opt-in experiments clearly separated from default-runtime validation.

### Correctness Acceptance

- Any default-path performance PR must pass all relevant local gates:
  - `cargo fmt --all -- --check`
  - `cargo check -q -p ferrum-cli`
  - `cargo test -q -p ferrum-engine --test continuous_batch_test`
  - `cargo test -q -p ferrum-scheduler`
  - kernel-specific parity tests for changed kernel families.
- Any CUDA kernel change must include a parity gate with documented tolerance:
  - attention kernels: max absolute error `<= 5e-3` and max relative error `<= 1e-2` against the current reference on representative decode, prefill, and mixed shapes;
  - MoE/Marlin kernels: existing quantized parity thresholds must be preserved or tightened, never loosened without a written numerical reason;
  - graph-enabled and graph-disabled paths must both pass at least one smoke when the changed code can run under graph capture.
- Any server/API change must pass non-ignored deterministic contract tests for request parsing, response serialization, streaming SSE shape, error mapping, and usage accounting.
- Any full-model GPU validation used for performance claims must first pass:
  - Paris single-turn gate;
  - Paris multi-turn gate;
  - `bench-serve` completion rate `100%`;
  - error count `0`;
  - no server panic or CUDA error in logs.
- If a correctness gate is skipped, the artifact manifest must mark the run `not_publishable=true` and include the reason.

### Performance Non-Regression Acceptance

- For any default-path change, compare against a same-pod baseline with the same model, binary feature set, GPU lock state, dataset, prompt/output lengths, and runtime config except for the intended change.
- M3 default-path throughput must not regress materially:
  - c=1,4,16,32 full sweep with `n_repeats >= 3`;
  - no cell may regress by more than `3%` versus baseline unless CI95 intervals overlap and the regression is explicitly classified as noise;
  - c32 must not regress by more than `2%` for changes touching attention, MoE, graph, scheduler, or sampling hot paths.
- Latency must not trade away product usability:
  - TTFT p50 may not regress by more than `10%`;
  - ITL p95 may not regress by more than `10%`;
  - TPOT p50 may not regress by more than `5%`;
  - any intentional latency tradeoff must be opt-in, documented, and excluded from default-path completion.
- Build-loop changes must not regress build speed:
  - attention-only rebuild p95 remains `<= 90s`;
  - no unrelated static library changes from `cache_hit` to `built` in the attention-only rebuild gate.
- API-only changes must not regress M3 c32 throughput by more than `1%` on a same-binary smoke, or must prove the touched path is not exercised by the benchmark.
- Opt-in experiments may regress default-path metrics only if the default path is proven unchanged by a forced-off A/B row.

### Publishable Artifact Requirements

Every publishable validation artifact must include:

- baseline and candidate git SHAs;
- git dirty status;
- binary SHA256;
- runtime config snapshot;
- env hash;
- GPU process preflight and cleanup status;
- correctness gate results;
- throughput/latency regression table;
- explicit verdict: `pass`, `fail`, or `diagnostic-only`.

## Final Completion Criteria

This goal is complete when all milestones A through I meet their quantitative acceptance criteria and one full validation packet is committed under `docs/bench/` or `docs/status/` containing:

- build timing table before/after;
- structured profile sample and schema validation result;
- migrated runner example artifact;
- env registry, named preset, and runtime config snapshot example;
- auto-config decision trace and selector validation report;
- OpenAI API compatibility matrix;
- strict JSON/schema validation report;
- code-size and ownership summary;
- correctness and performance non-regression report;
- exact commands used for local and GPU validation.

## Non-Goals

- This goal does not require a new M3 throughput record.
- This goal does not require FA2 source to become default.
- This goal does not require implementing multimodal chat input beyond rejecting unsupported parts correctly.
- This goal does not require full JSON Schema support; unsupported strict schemas may be rejected.
- This goal does not require deleting all env vars; it requires typed resolution, registry ownership, source attribution, and artifact visibility.
