# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Behavioral guidelines (read first)

Reduce common LLM coding mistakes. Merge with project-specific
instructions below as needed.

**Tradeoff:** these guidelines bias toward caution over speed. For
trivial tasks, use judgment.

### 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?"
If yes, simplify.

### 3. Surgical changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's
request.

### 4. Goal-driven execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria
("make it work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in
diffs, fewer rewrites due to overcomplication, and clarifying questions
come before implementation rather than after mistakes.

---

## M3 80% goal operating protocol

When a task mentions M3, Qwen3-MoE, CUDA performance, vLLM ratio,
Vast pods, or benchmark sessions, read these before acting:

1. `docs/bench/m3-80pct-goal-2026-05-25/GOAL.md`
2. `docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-27-iteration3/SESSION-REPORT.md`
3. `docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-28/INCIDENT-REPORT.md`

Current facts to preserve:

- Goal: `Qwen/Qwen3-30B-A3B-GPTQ-Int4` must reach `>= 0.80 × vLLM`
  throughput for `c=1/4/16/32`.
- Best recent c=32 progress: VPA bridge on 2026-05-28 makes
  `FERRUM_USE_VLLM_PAGED_ATTN=1` pass the Paris smoke and lifts a
  same-pod N=3 random 256/128 sweep by about
  `+3.1%/+2.8%/+3.4%/+6.5%` for `c=1/4/16/32`; c=32 reached
  `1127.9 ± 101.4 tok/s` versus no-VPA `1058.9 ± 9.2 tok/s`.
- The post-VPA c=32 active-batch profile still points at MoE, not
  attention: m=32 decode was roughly `~69%` MoE, `~11%` attention,
  `~14%` dense; inside MoE, vLLM-Marlin gate_up/down GEMMs dominate.
- The easy env levers are exhausted. Do not spend another session
  flipping `FERRUM_VLLM_MOE`, `FERRUM_MOE_GRAPH`, or
  `FERRUM_PAGED_FLASH_SPLITS` without a new code-level hypothesis.
- Existing Triton fused-MoE PTX was checked as a minimal small-m MoE
  validation and is negative for this shape: gate_up only measured about
  `453 µs/layer`, slower than the vLLM-Marlin gate_up profile at about
  `150 µs/layer`.
- Skipping redundant vLLM-Marlin MoE workspace zeroing is correct but
  only a small/noisy win: Paris passed, c=32 N=3 measured
  `1153.1 ± 152.7 tok/s` vs forced old zeroing `1130.4 ± 168.7 tok/s`.
- 2026-05-29 Codex handoff fixed two default-path issues and one small
  performance lever: graph capture now requires graph-clean MoE
  (`FERRUM_VLLM_MOE=1`, no forced host route), server c=4 defaults to the
  batched-decode path (`FERRUM_MOE_BATCH_THRESHOLD=4`), and
  `FERRUM_VLLM_MOE_PAIR_IDS=1` defaults on with vLLM MoE. Paris passed;
  measured full sweep before pair-id defaulting was c=1 `155.4 ± 1.0`,
  c=4 `425.6 ± 36.6`, c=16 `965.8 ± 6.4`, c=32 `1205.6 ± 55.1`
  tok/s. Pair-id plus the combine fast path improves c=16 to
  `993.8 ± 26.6` and c=32 to `1264.0 ± 29.4`, still not enough to
  declare the 0.80× goal done.
- The pair-id combine fast path is not residual fusion. It adds
  `weighted_sum_batched_f16` for the vLLM pair-id layout; artifact
  `/workspace/m3-graph-loop/bench_pairids_residual_c16_c32_n3_rerun2/`
  is valid despite the misleading name. Ignore
  `/workspace/m3-graph-loop/bench_pairids_residual_c16_c32_n3/` because
  the server was interrupted by process-group SIGINT.
- Same-pod vLLM 0.20.2 c=16/c=32 N=3 baseline is now available:
  `/workspace/m3-graph-loop/vllm0202_baseline_c16_c32_n3_retry/`
  measured c=16 `1328.7 ± 44.4`, c=32 `1971.8 ± 7.4` tok/s. Current
  Ferrum ratios are about c=16 `0.75×`, c=32 `0.64×`; c=32 remains the
  main blocker and needs about `+25%`.
- A partial vLLM 0.20.2 Marlin-MoE scheduling/tile backport was tested
  and reverted: Paris passed, but c=16 `1000.9 ± 34.7`, c=32
  `1250.4 ± 65.5` tok/s showed no c=32 gain. Do not repeat partial
  thread-config/tile scheduling changes without a new profiler-backed
  hypothesis.
- A full-model `FERRUM_MOE_BLOCK_SIZE=8` validation was tested because
  vLLM 0.20.2 selects block size 8 for small-M MoE. Ferrum now supports
  the override and sizes vLLM-MoE `c_tmp` safely for block8, but the
  same-pod result is negative: c=16 `975.1 ± 3.6`, c=32
  `1209.5 ± 37.2` in
  `/workspace/m3-graph-loop/block8_validation_rerun/`, below the block16
  fast-path baseline. Do not make block8 default or repeat block8-only
  tests; revisit only as part of a full vLLM 0.20.2 Marlin template/source
  parity port.
- Fresh current-default c32 profile after pair-id combine lives at
  `/workspace/m3-graph-loop/profile_current_pairid_combine_c32/`.
  With graph disabled for sync timers, steady m≈30/31 decode is still
  `16–17 ms`; MoE is `~64–66%`, dominated by `gemm1≈6.2–6.6 ms` and
  `gemm3≈3.0–3.2 ms`, while combine is only `~0.25 ms`. Do not spend a
  primary lever on combine.
- New debug-only tools are ready for the next GPU run:
  `FERRUM_MOE_DUMP=1 FERRUM_MOE_DUMP_BATCH_X_TOPK=256` captures real c=32
  decode `active_blocks/unique_experts`, and `FERRUM_UNIFIED_POST_PROF=1`
  splits unified model time from decode post-process/sample/scheduler/stream/
  stop/complete. Run these before making a new Marlin or engine optimization
  claim.
- Use `scripts/m3_route_unified_profile.sh` on the restored pod to collect the
  route dump and `[unified-prof]`/`[iter-prof]`/`[bucket-prof]` snippets in
  one c=32 run. It sets `FERRUM_MOE_GRAPH=0` because route dumping syncs/copies
  GPU buffers. The script fails if route shape or unified timing is missing.
- Vast instance `38237968` stopped during the route-dump build. Restart
  returned `resources_unavailable`; renting a replacement 48GB RTX 4090
  failed with `insufficient_credit`. Treat later GPU work as not run until a
  pod is restored.
- `FERRUM_USE_VLLM_PAGED_ATTN=1` is no longer the c=1 correctness bug
  after the VPA bridge; keep Paris as the minimum gate for future
  attention or MoE routing changes.
- vLLM-ratio numbers remain indicative until an apples-to-apples vLLM
  0.20.2 sweep is rerun with random 256/128 and `n_repeats >= 5`.

### Required execution contract before GPU spend

Before opening or using a paid GPU pod, write a short contract:

```text
Lever:
Expected gain:
Files/paths to inspect or change:
Correctness gate:
Benchmark gate:
Budget cap:
Stop condition:
```

If the task cannot be stated this concretely, do not open the pod yet.
Ask for scope or do local analysis first.

### Anti-drift rules

- One session = one lever. Do not drift from a scoped kernel/graph bug
  into broad sweeps.
- Stop after the first failed correctness gate; do not benchmark known
  garbage output.
- For deltas under 10%, require same-pod `N >= 3`; single-run Vast
  numbers are not evidence.
- During CUDA builds, run parallel local work: source trace, microbench
  design, or report update. Waiting is not a task.
- If 30 minutes of GPU time produces no code change, no new profile, and
  no falsified hypothesis, stop and report.

### Recommended plan to regain goal progress

1. **Lock baseline** — with explicit user budget approval, rerun
   ferrum/vLLM apples-to-apples for `c=1/4/16/32`, random 256/128,
   `n_repeats >= 5`, and update `GOAL.md`.
2. **Attack MoE first** — because post-VPA profile shows MoE dominates,
   validate full vLLM 0.20.2 Marlin-MoE source parity, a small-m fused-MoE
   path, or a concrete vLLM-Marlin gate_up/down reduction before spending
   time on smaller attention wins.
3. **Fix one medium lever** — choose exactly one non-MoE lever only when
   profiling justifies it: block-table shared-memory cache, `moe_align`
   rewrite/port, or graph coverage beyond the layer loop. Ship only if
   Paris and same-pod A/B pass.
4. **Update the source of truth** — after any shipped lever, add the
   exact artifacts and commands to the M3 goal session docs before
   opening another GPU session.

---

## What is this

Ferrum Infer is a Rust-native LLM inference engine. Single binary, no Python — supports Metal (macOS), CUDA (NVIDIA), CPU backends. Targets vLLM-level performance with PagedAttention, continuous batching, custom CUDA kernels.

## Current baselines (RTX 4090, ShareGPT apples-to-apples vs vLLM 0.20.2)

| model | c | ferrum | vLLM | ratio |
|---|---:|---:|---:|---:|
| M1 (Llama-3.1-8B FP16) | 32 | see `docs/bench/cuda-rtx4090-2026-05-07/` | | |
| M2 (Llama-3.1-8B-INT4) | 32 | **881** | 2179 | 40% |
| M3 (Qwen3-30B-A3B-Int4) | 32 | **1023** | 1867 | 55% |

M2 currently primary win target via dense Marlin tile tuning + continuous batching. M3 primary via Phase 3 engine-init scratch budget + chunked prefill (see `docs/continuous-batching-redesign.md`). 80%-of-vLLM is the stretch target.

**Latest progress report**: `docs/progress/2026-05-13-continuous-batching-foundation.md` — current state, what shipped, what's next.

## Build & dev commands

```bash
cargo check --workspace --all-targets   # Mac, no GPU features
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -A warnings

# CLI surface
cargo run -p ferrum-cli -- run qwen3:0.6b                                          # interactive chat
cargo run -p ferrum-cli -- serve --model qwen3:0.6b --port 8000                    # OpenAI API
cargo run -p ferrum-cli -- bench qwen3:4b --concurrency 4 --max-tokens 128         # internal bench
cargo run -p ferrum-cli -- bench-serve --base-url http://127.0.0.1:8000 ...        # HTTP bench, vLLM-parity

# Feature gates
cargo run -p ferrum-cli --features metal -- ...
cargo run -p ferrum-cli --features cuda,vllm-moe-marlin -- ...
cargo run -p ferrum-cli --features cuda,triton-kernels -- ...   # FERRUM_TRITON_INT4=1 to enable
```

Internal env vars (`FERRUM_KV_MAX_BLOCKS` / `FERRUM_PAGED_MAX_SEQS` / `FERRUM_VLLM_MOE` etc.) are set by `gpu_mem_autosize` for `serve` and chat-profile defaults for `run`. Users should not set by hand — autosizer's job.

## Bench reproducibility

Full methodology + env blocks live in `bench/v0.2-cuda/` and `docs/bench/`:
- `bench/v0.2-cuda/apples_all_drive.sh` — single driver, M1/M2/M3 × c=1/4/16/32 sweep (use `SKIP_VLLM=1` to skip vLLM rerun)
- `bench/v0.2-cuda/PERF_TRACKER.md` — running tracker (R0-R2 + Wave 1 + Phase 2B retrospectives)
- `docs/bench/macos-2026-05-02/` — Metal apples-to-apples vs llama.cpp + mistralrs (M1 Max 32GB)

Profile probes (opt-in via env): `FERRUM_NEXT_BATCH_PROF`, `FERRUM_SCHED_NONE_PROF`, `FERRUM_BATCH_PREFILL_PROF`, `FERRUM_RBD_PROF`, `FERRUM_BATCH_DECODE_PROF`, `FERRUM_DECODE_OP_PROFILE`.

## Architecture

**Model-as-Code v2**: per-family Rust struct generic over `Backend<B>` + `K: KvDtypeKind`. `LlamaFamilyModel<B, K>` covers Llama / Qwen2.x / Qwen3 / Mistral (per-family quirks are config toggles). `Qwen3MoeModel<B, K>` covers Qwen3-MoE / 30B-A3B. Adding a backend = implementing the relevant supertraits, not editing models.

**5 polymorphism dimensions**: model architecture / compute precision (`Linear<B>`) / weight format (`WeightLoader<B>` + `MarlinExpertStack<B>`) / device (`Backend` + supertraits) / KV precision (`KvDtypeKind` marker).

**13 crates, bottom-up**:
- Foundation: `ferrum-types`, `ferrum-interfaces`
- Core logic (no GPU): `ferrum-scheduler`, `ferrum-sampler`, `ferrum-tokenizer`, `ferrum-kv`
- Compute (feature-gated): `ferrum-kernels`, `ferrum-quantization`
- Application: `ferrum-engine` (only LLM engine = `ContinuousBatchEngine`), `ferrum-models`, `ferrum-server`, `ferrum-cli`
- Testing: `ferrum-testkit`

**Inference flow** (decode, OpenAI chat):
```
HTTP → ContinuousBatchEngine.run_iteration (under iteration_lock)
     → ContinuousBatchScheduler.next_batch (mixed prefill+decode batch)
     → process_batch_unified → model_executor.unified_decode
     → per-layer: rms_norm → qkv_proj → varlen attn → o_proj → MLP/MoE → residual
     → Sampler → mpsc StreamChunk → SSE
```

The unified path lives in `crates/ferrum-engine/src/continuous_engine.rs::process_batch_unified` (Llama already supports `unified_forward`; Qwen3MoE implementation lands dormant pending Phase 3 scratch budgeting).

**Iteration lock**: `EngineInner::iteration_lock` serializes batches. Mixed prefill+decode in one batch recovers intra-iteration parallelism.

## Code style

- Rust 2021, `rustfmt.toml`: 4-space indent, max width 100
- `snake_case` fns/modules, `CamelCase` types/traits, `SCREAMING_SNAKE_CASE` constants
- Conventional commits: `feat(scope):`, `fix(scope):`, `refactor(scope):`
- `cargo check --workspace` and `cargo test --workspace` must pass on Mac without GPU features
- CUDA code behind `#[cfg(feature = "cuda")]`, Metal behind target OS gates, Triton kernels behind `#[cfg(feature = "triton-kernels")]` (implies cuda)
- Shared types/traits go in `ferrum-types`/`ferrum-interfaces`, never duplicated

## Done criteria — engine / scheduler / sampler / CLI run / HTTP server

Any PR that touches `ferrum-engine/src/continuous_engine.rs`,
`ferrum-engine/src/sampler/`, `ferrum-scheduler/`,
`ferrum-cli/src/commands/run.rs`, `ferrum-cli/src/commands/serve.rs`,
or `ferrum-server/` is in the d67fbbb blast radius (EOS /
stop_sequences / stream / multi-turn KV / chat template / OpenAI wire
format). It is NOT done until both blast-radius suites pass locally:

```bash
# Pre-pull models once
ferrum pull qwen3:0.6b           # + tinyllama, qwen2.5:0.5b for full chat matrix

# CLI lane (REPL / PTY / stress)
cargo test --release -p ferrum-cli --features metal --test chat_smoke   -- --ignored --test-threads=1
cargo test --release -p ferrum-cli --features metal --test chat_pty     -- --ignored --test-threads=1
cargo test --release -p ferrum-cli --features metal --test chat_stress  -- --ignored --test-threads=1

# HTTP server lane (OpenAI /v1/chat/completions)
cargo test --release -p ferrum-cli --features metal --test server_smoke         -- --ignored --test-threads=1
cargo test --release -p ferrum-cli --features metal --test server_openai_compat -- --ignored --test-threads=1
cargo test --release -p ferrum-cli --features metal --test server_stress        -- --ignored --test-threads=1

# Correctness lane — byte-equal baseline of greedy decode outputs
# (vLLM test_fingerprint style). Re-baseline with FERRUM_UPDATE_FIXTURES=1
# only after human-reviewing the diff.
cargo test --release -p ferrum-cli --features metal --test reference_match      -- --ignored --test-threads=1
```

Total ~120 s on M1 Metal (`--release` is required — debug is ~5× slower).
Nightly CI (`.github/workflows/chat-smoke.yml`) runs every suite above on
macos-latest. PR-time CI (`.github/workflows/ci.yml`) only compiles them
via `cargo check --all-targets` — actual runs are nightly because each
test cold-loads a real model.

Known server-side gaps (loose-assertion floor in current tests, tighten
when each lands a fix): see `~/.claude/projects/*/memory/project_http_server_gaps_2026_05_19.md`.

## CUDA backend layout

`crates/ferrum-kernels/src/backend/cuda/` split by supertrait:

| file | trait |
|---|---|
| `mod.rs` | core `Backend` + `CudaState` + `KvFp16` |
| `collective.rs` | `BackendCollective` (TP all-reduce) |
| `graph.rs` | `BackendGraph` + multi-slot graph cache |
| `int8_kv.rs` | `BackendInt8KvOps` + KvCacheQuant ctor |
| `quant.rs` | `BackendQuantMarlin` + `BackendQuantGguf` + vllm marlin moe |
| `paged.rs` | `BackendPagedKv` + SplitKScratch + paged dispatchers |
| `moe.rs` | `BackendMoeFused` (route_topk_softmax, moe_align_block_size, moe_combine) |

Custom CUDA kernels (PTX compiled at build time via `ferrum-kernels/build.rs`): `rms_norm.cu`, `fused_add_rms_norm.cu`, `rope.cu`, `fused_silu_mul.cu`, `decode_attention.cu` / `flash_decode_attention.cu` / `paged_decode_attention.cu` (+ split-K + INT8 variants), `residual_add.cu`, `marlin.cu`, `vllm_marlin_moe/ops.cu` (vendored, under `vllm-moe-marlin` feature).

Triton-rs PTX path (`triton-kernels` feature): `crates/ferrum-kernels/triton_ptx/` (offline-compiled, embedded via `include_str!`, loaded by `cudarc::get_or_load_custom_func`). Triton w4a16 INT4 `FERRUM_TRITON_INT4=1` has a known prefill-NaN bug under ContinuousBatch scheduler — Marlin remains default. See `crates/ferrum-kernels/src/backend/cuda/triton_*.rs` for wired kernels.

## Model support

**LLM** (via `LlamaFamilyModel<B>`): Qwen3 (0.6B–4B), Qwen2.5-Instruct (0.5B–7B), Llama-3.x-Instruct (1B–8B), TinyLlama-1.1B-Chat, Mistral (sliding-window).

**MoE** (via `Qwen3MoeModel<B>`): Qwen3-MoE / Qwen3-30B-A3B.

**ASR**: Whisper (tiny → large-v3-turbo). Files: `crates/ferrum-models/src/multimodal/whisper.rs` + `whisper_decoder.rs`. Recommended: `whisper-turbo`. Quirk: Metal float32 matmul accumulation differs from CPU after ~4 encoder layers (~minor char-level drift; hardware-level, not fixable in software).

**TTS**: Qwen3-TTS-0.6B / 1.7B. **Embeddings**: CLIP, Chinese-CLIP, SigLIP, BERT.

**Quantization**: GPTQ INT4 auto-detected at load. CUDA defaults to Marlin (repacks at load); Triton w4a16 (`FERRUM_TRITON_INT4=1`) runs on-disk GPTQ tile layout directly. CPU path uses dequant+cuBLAS fallback.

Models cached at `~/.cache/huggingface` (shared with HF Python).

## Config

Runtime defaults: `EngineConfig::default()` in `ferrum-types::config`. Build script: `ferrum-kernels/build.rs` compiles CUDA .cu → PTX via `bindgen_cuda`, requires `CUDA_HOME`. Metal shaders embed at compile time via `include_str!`.
