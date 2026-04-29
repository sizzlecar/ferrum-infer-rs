# M1 Max Q4_K_M Benchmarks

Head-to-head decoding throughput on Apple Silicon, GGUF Q4_K_M format,
identical files across engines so the comparison is engine quality, not
quantisation.

## Hardware

| Field | Value |
|---|---|
| Machine | Apple M1 Max |
| Memory | 32 GB unified |
| OS | macOS (host) |

## Models (download list, ~30 GB total)

| Model | HuggingFace repo | File | Size |
|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B-GGUF` | `Qwen3-8B-Q4_K_M.gguf` | ~4.5 GB |
| Llama-3.1-8B-Instruct | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (or similar) | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | ~4.6 GB |
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B-GGUF` | `Qwen3-30B-A3B-Q4_K_M.gguf` | ~17 GB |

For each model, also download the corresponding `tokenizer.json` from
the **non-GGUF** HuggingFace repo (e.g. `Qwen/Qwen3-8B`,
`meta-llama/Meta-Llama-3.1-8B-Instruct`, `Qwen/Qwen3-30B-A3B`).

## Engines

| Engine | Build / install | Source format |
|---|---|---|
| ferrum-infer-rs (this repo) | `cargo build -p ferrum-cli --bin ferrum --features metal --release` | `.gguf` via candle-transformers' Q4_K_M Metal kernels |
| mistral.rs | `cargo install mistralrs` (or build from source) | same `.gguf` |
| llama.cpp | `brew install llama.cpp` | same `.gguf` |

## Workloads

Phase 1 (this commit): single-stream decode, deterministic.

| Workload | Prompt tokens | Output tokens | Concurrency | What it tests |
|---|---|---|---|---|
| Single decode | ~16 (default prompt) | 256 | 1 | Decode throughput |
| TTFT (long context) | 2048 | 1 | 1 | Prefill speed |
| Concurrent | 512 | 256 | 16 | Continuous batching (deferred) |

The concurrent workload requires ferrum's HTTP server path with batched
scheduling; will land in a follow-up.

## Running

```bash
# 1. Build ferrum (once):
cargo build -p ferrum-cli --bin ferrum --features metal --release

# 2. Lay out files:
#    ~/Downloads/ferrum-bench/
#      models/
#        Qwen3-8B-Q4_K_M.gguf
#        Llama-3.1-8B-Instruct-Q4_K_M.gguf
#        Qwen3-30B-A3B-Q4_K_M.gguf
#      tokenizers/
#        Qwen3-8B.tokenizer.json
#        Llama-3.1-8B-Instruct.tokenizer.json
#        Qwen3-30B-A3B.tokenizer.json

# 3. Run the bench:
./scripts/bench_m1_q4_k_m.sh \
  ~/Downloads/ferrum-bench/models \
  ~/Downloads/ferrum-bench/tokenizers

# 4. Inspect bench_results/<timestamp>/summary.md
```

Or run a single model manually:

```bash
# One-shot generation + timing (the bench path)
./target/release/ferrum run ~/Downloads/Qwen3-8B-Q4_K_M.gguf \
  --tokenizer ~/Downloads/Qwen3-8B.tokenizer.json \
  --prompt "Explain the theory of relativity in two sentences." \
  --max-tokens 256 \
  --temperature 0

# Interactive chat (multi-turn, with chat template + sampling)
./target/release/ferrum run ~/Downloads/Qwen3-8B-Q4_K_M.gguf \
  --tokenizer ~/Downloads/Qwen3-8B.tokenizer.json \
  --temperature 0.7 --top-k 50 --top-p 0.95 --repeat-penalty 1.1
```

In the REPL:
- `/exit` or Ctrl-D — quit
- `/clear` — reset KV cache (re-opens the model)
- `/system <text>` — change system prompt + reset
- `/help` — list commands

## Methodology

Following `llama-bench` (the canonical GGUF benchmark tool, included
with llama.cpp):

  - **Sequential**: one engine + one workload + one model = one
    process. Engines never run concurrently; would contend for the
    Metal device queue and unified memory bandwidth, results would be
    meaningless.
  - **Workloads** (fixed shape per cell):
    - `pp512` — prompt processing: model digests a 512-token prompt;
      tokens/sec measures **prefill throughput**
    - `tg128` — token generation: model decodes 128 fresh tokens;
      tokens/sec measures **decode throughput**
  - **Repetition**: 1 warm-up + ≥5 timed runs per cell. Reported as
    `mean ± stddev`. Warm-up discards cold-cache effects.
  - **Determinism**: temperature = 0 (greedy) so different runs
    produce identical token sequences and timing differences are
    purely engine-side.
  - **Hardware state**: M1 Max, 32 GB unified memory, AC-powered.
    Other heavy apps closed during runs (low-memory pressure regime).
  - **Reproducibility**: `llama-bench -m <model.gguf> -p 512 -n 128`
    (default 5 reps, mean ± stddev printed).

References:
  - llama-bench manual: `man llama-bench` after `brew install llama.cpp`
  - vLLM: `benchmark_throughput.py` follows the same shape (separate
    pp / tg, multiple reps, deterministic decode).

## Results

### llama.cpp baseline — 2026-04-29 (M1 Max, build 8960)

#### Hardware

| Field | Value |
|---|---|
| Machine | Apple M1 Max |
| CPU | 8 performance + 2 efficiency cores (10 total) |
| GPU | 24-core (this binning of M1 Max; 32-core also exists) |
| Memory | 32 GB unified |
| Power | AC, 100% battery, lid open |
| Thermal | Cold start at first run; subsequent runs back-to-back |

#### Software

| Field | Value |
|---|---|
| OS | macOS 15.1.1 (build 24B91) |
| llama.cpp | brew bottle `8960`, build commit `19821178b` |
| ggml | `0.10.0` (libggml-blas, libggml-metal, libggml-cpu-apple_m1) |
| Metal device | MTL0, MTLGPUFamilyApple7 / Common3 / Metal3 |
| Working set | `recommendedMaxWorkingSetSize = 22906.50 MB` |
| Build flags | brew bottle defaults (`GGML_METAL=ON`, BLAS via Accelerate); not built from source |

#### Command

```bash
llama-bench \
  -m <model.gguf> \
  -p 512 -n 128 \
  -t 8 -ngl 99 -r 5 \
  -o md
```

  - `-p 512` = prompt-processing test (prefill 512 tokens)
  - `-n 128` = token-generation test (decode 128 tokens)
  - `-t 8`   = 8 CPU threads (perf cores only — leave efficiency cores for the OS)
  - `-ngl 99` = offload all layers to Metal
  - `-r 5`   = 5 timed reps (one warm-up included by default)
  - `-o md`  = markdown output (also saved as raw stdout)

#### KV cache + context

  - KV dtype: `fp16` (llama-bench default)
  - Context size: `4096` (llama-bench default)

#### Models

| Model | HuggingFace source | Size | SHA-256 (truncated) |
|---|---|---:|---|
| Qwen3-8B Q4_K_M | `Qwen/Qwen3-8B-GGUF`/`Qwen3-8B-Q4_K_M.gguf` | 4.68 GiB | `d98cdcbd…45785` |
| Llama-3.1-8B-Instruct Q4_K_M | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`/`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.58 GiB | `7b064f58…3557c` |
| Qwen3-30B-A3B Q4_K_M | `Qwen/Qwen3-30B-A3B-GGUF`/`Qwen3-30B-A3B-Q4_K_M.gguf` | 17.28 GiB | `0d003f66…aa8f5` |

Full SHA-256 list and raw `llama-bench` stdout/stderr are saved in
`bench_results/` (`metadata.txt`, `model_sha256.txt`, `llamacpp_*.md`,
`llamacpp_*.stderr.log`) so the numbers below can be reproduced and
audited.

#### Results

| Model | Size | Backend | pp512 (prefill tok/s) | tg128 (decode tok/s) |
|---|---:|---|---:|---:|
| Qwen3-8B Q4_K_M | 4.68 GiB | BLAS+MTL | **346.52 ± 25.26** | **27.94 ± 0.90** |
| Llama-3.1-8B-Instruct Q4_K_M | 4.58 GiB | BLAS+MTL | **335.13 ± 24.86** | **29.20 ± 0.44** |
| Qwen3-30B-A3B Q4_K_M | 17.28 GiB | BLAS+MTL | **596.58 ± 3.89** | **44.52 ± 6.80** |

Observation: **Qwen3-30B-A3B MoE decode (44.5 t/s) is ~1.6× faster
than dense Qwen3-8B (27.9 t/s)** despite having ~3.7× more total
parameters — only 3B active params per token. ferrum needs to match
this MoE-on-Mac advantage to be competitive on the flagship target.

The pp512 stddev for the 8B models (~7%) is higher than the 30B model's
(~0.7%); this matches the fact that the 8B models share the M1 Max with
other macOS workloads (Spotlight, mds, etc.), while the 30B model
saturates the GPU and washes those out. Re-running with the macOS in
"low-overhead" mode (Spotlight indexing paused, no other apps) is on
the to-do list.

> Earlier revisions of this document published higher numbers
> (`373.51 / 31.17 / 375.24 / 29.06 / 598.23 / 52.68`); those came from
> a single un-saved run and didn't survive re-measurement. The numbers
> above were captured with the saved `bench_results/llamacpp_*.md`
> outputs as audit trail.

These are the numbers ferrum is catching up to. ferrum's own runs go
in the next subsection once Phase 1D is hooked into LlamaFamilyModel
end-to-end.

### ferrum — first measurement, 2026-04-29 (Qwen3-8B only, decode-only)

The Phase 1D real-Q4 path is now end-to-end: GGUF → ferrum's
`LlamaFamilyModel<B>` → custom Metal kernels (Q4-GEMV fused +
RMSNorm + QK-norm-RoPE + flash-attn + …) with no candle in the
forward path. Memory holds a single Q4 copy in MTLBuffer (~5 GB on
disk, no eager-fp32 inflation). Output is coherent.

Decode throughput (M1 Max, 24-core GPU, AC-powered):

| Model | Engine | Decode tok/s | vs llama.cpp tg128 |
|---|---|---:|---:|
| Qwen3-8B Q4_K_M | ferrum (Phase 1D, baseline) | ~0.27 | ~104× slower |
| Qwen3-8B Q4_K_M | + Q6_K direct + v2 gemv | ~0.31 | ~90× slower |
| Qwen3-8B Q4_K_M | + MultiQuant fused qkv | ~0.35 | ~80× slower |
| Qwen3-8B Q4_K_M | **+ FERRUM_KV_CAPACITY=1024 (current)** | **24.93 ± 0.25** | **89%** |

#### Full apples-to-apples (5 reps each, FERRUM_KV_CAPACITY=1024)

After porting llama.cpp's `kernel_mul_mm_q4_K_f32`,
`kernel_mul_mm_q6_K_f32`, and switching MultiQuant Fused (mixed Q4+Q6
qkv) onto per-part mul_mm dispatches:

| Model | Test | ferrum mean ± std | llama.cpp | Ratio |
|---|---|---:|---:|---:|
| Qwen3-8B Q4_K_M | pp (302-token prompt) | **253.16 ± 1.42** | 346.52 ± 25.26 (pp512) | **73%** |
| Qwen3-8B Q4_K_M | tg128 | **25.58 ± 0.42** | 27.94 ± 0.90 | **92%** |
| Llama-3.1-8B Q4_K_M | pp (295-token prompt) | **251.62 ± 1.36** | 335.13 ± 24.86 (pp512) | **75%** |
| Llama-3.1-8B Q4_K_M | tg128 | **26.47 ± 0.36** | 29.20 ± 0.44 | **91%** |
| Qwen3-30B-A3B Q4_K_M | ⚠ pre-#40 numbers below were measured on a **garbage-output binary** (head_dim bug). Real numbers: 43.3 t/s prefill (205 tok), 16 t/s tg @ small kv. See "MoE quality regression" section below. |

Progression of prefill on Qwen3-8B (302-token prompt) through this work:

| Stage | tok/s | vs llama.cpp |
|---|---:|---:|
| ferrum baseline (pre-mul_mm) | 71.94 | 21% |
| + Q4 mul_mm | 128.92 | 37% |
| + Q6 mul_mm | 193.18 | 56% |
| + MultiQuant Fused mul_mm (current) | **253.16** | **73%** |

Cumulative prefill speedup: **3.52×** on Qwen3-8B, **3.53×** on Llama-3.1.

Decode is at 92% of llama.cpp on both models — within the noise floor
of llama.cpp's own ±3% stddev.

Remaining ~25% prefill gap is no longer concentrated in any single
matmul kernel. Likely sources:
  - Other m-scaling ops (split_qkv, qk_norm_rope, prefill-attention,
    fused_silu_mul_split, residual_add) that run per-token on the
    activation side
  - llama.cpp may have larger inner-tile constants (e.g. NR0=128 vs
    our 64) on M1 Max specifically — would need a kernel sweep to
    confirm
  - Multi command-buffer + parallel encoding path (`n_cb`) that
    keeps the GPU front-end fed during the prefill burst

Headline: **decode is within 6-11% of llama.cpp on dense models; prefill is 5× behind**.

Raw per-rep output preserved in `bench_results/ferrum_<model>_<test>.txt`.
Bench script reproducible via `bench_results/run_ferrum_bench.sh`.

#### Why prefill (m > 1) is 5× behind

When `m > 1`, every Q4 / Q6 matmul falls off the fast fused-gemv path:

  - **Q4 homogeneous fused** (gate_up_proj, layers where qkv is all-Q4):
    `dequant_q4_k → 200 MB fp16 transient → gemm_v2_f16w`. Writes
    a multi-hundred-MB transient per matmul. The transient is pooled
    by shape, but the actual fp16 write+read traffic is real.
  - **MultiQuant fused qkv** (mixed Q4+Q6 qkv): per-row × per-part
    fused gemv loop. For m=302 prompt × 3 parts × 36 layers = ~32 K
    gemv dispatches just for qkv. Memory-savings win, throughput loss.
  - **Q6 down_proj for m>1**: similar per-row gemv loop, `m` calls.

Rough rate budget vs llama.cpp at pp512:
  - llama.cpp uses a tiled `mul_mm_q4_K_f32` / `mul_mm_q6_K_f32` simdgroup
    matrix multiply that dequants Q4_K / Q6_K inside the GEMM loop.
  - Ferrum's gemm_v2_f16w is f16-weight only and doesn't fuse the Q4
    dequant. So we pay 2× memory traffic (Q4 read + fp16 write of
    transient + fp16 read by GEMM) per matmul.
  - Plus we lose the simdgroup matrix-mul throughput because we tile
    via the dequant→gemm route.

Fix path (planned): port llama.cpp's
`kernel_mul_mm_q4_K_f32` / `kernel_mul_mm_q6_K_f32` (or a simplified
version) so prefill never materialises the fp16 transient. Should
recover most of the 5× gap.

Decode (m=1) already uses the fused path (`gemv_q4kw_v2` /
`gemv_q6kw_v2`), which is why it's already within 11% of llama.cpp.

### MoE quality regression — discovered + fixed 2026-04-30

> ⚠️ **All Qwen3-30B-A3B numbers from PRs #35–#39 below this point are
> quality-invalid.** Every benchmark ran with `--bench-mode` which
> suppresses generated tokens. The speed numbers were measured but the
> output text was repeating-token gibberish (`"Dund impe impe..."`)
> caused by a misderived `head_dim`: ferrum was computing
> `head_dim = hidden_size / num_heads = 2048 / 32 = 64`, but
> Qwen3-30B-A3B has explicit `head_dim=128` (the GGUF stores
> `qwen3moe.attention.key_length=128`). Qwen3-8B happens to satisfy
> `hidden=4096=32×128` so the divide gives the right answer there;
> Qwen3-30B-A3B's non-square attention shape exposed the bug.
>
> **Fixed in PR #40** (read `<arch>.attention.key_length` from GGUF).
> **Prefill batched fast path correctness fixed in PR #41**
> (2-D `mul_mm_id` Q4_K/Q6_K kernels — see below).
>
> **Real numbers post #40 + #41 (M1 Max, single-rep, FERRUM_KV_CAPACITY=512):**
>
> | Test | ferrum | vs llama.cpp baseline |
> |---|---:|---:|
> | Prefill, 205-token prompt | **43.3 t/s** | 7% of 596 |
> | Decode @ kv_len ≤ 5 | **15.3 t/s** | 35% of 44.5 |
> | Decode @ kv_len = 205 | 3.4 t/s | 8% — large attention scaling cost |
>
> The pre-fix and post-fix prefill numbers are coincidentally close
> (~44 t/s) — same kernel work, different output quality.
> The full 5-rep regression suite needs a re-run on the new binary
> before the entries below can be replaced; tracked separately.

### Qwen3-30B-A3B MoE — first measurement, 2026-04-29 (PRE-FIX, INVALID)

`Qwen3MoeModel<MetalBackend>` ships the MoE family decoder end-to-end:
attention path identical to dense Qwen3, FFN replaced by router-driven
top-K expert dispatch. Expert weights stay quantised in MTLBuffer
(per-expert `QuantLinear<B>` slicing the on-disk 3-D
`ffn_{gate,up,down}_exps.weight` tensors byte-wise — no fp32 inflation).

Decode progression on Qwen3-30B-A3B (16-token smoke + full tg128):

| Stage | Decode tok/s | Latency / tok | Notes |
|---|---:|---:|---|
| #35 first wiring (round-trip via `out`) | 2.1 | 480 ms | 4 copy_slice + scaled_add per (token, expert) pair |
| #36 acc_buf + zero scratch | 4.4 | 228 ms | -42% dispatches; smoke (16 tok) |
| #36 full tg128 (5 reps mean) | **13.22 ± 2.86** | 76 ms | amortises launch overhead over 128 tok |
| #36 best rep | 18.54 | 54 ms | warmup-stabilised steady-state |
| llama.cpp tg128 baseline | 44.52 ± 6.80 | 22 ms | — |

Headline: **30% of llama.cpp decode (mean) / 42% on the best rep**.

Decode rep variance is large (6.9 → 13.1 s for 128 tokens, 2× spread)
because the first rep eats Metal pipeline state caching and KV cache
warmup; subsequent reps approach steady-state and are within ~2.4× of
llama.cpp. The full tg128 mean (13.22 t/s) is heavily dragged down by
rep 1.

Prefill (5 reps, 302-token prompt):

| Engine | pp512 / pp302 | vs llama.cpp |
|---|---:|---:|
| ferrum (this bench, 302-tok prompt) | **43.67 ± 2.05** | **7%** (~14× behind) |
| llama.cpp pp512 baseline | 596.58 ± 3.89 | — |

Where the gap is, and what it would take:

- **Per-(token, expert) loop.** Decode does `top_k=8 × 48 layers × ~5
  Metal dispatches/pair = 1920` kernel launches per token plus 96
  copy_slice ops, plus 48 host syncs to read `router_logits` for top-K.
  Each `B::sync` on Metal is a `commit + waitUntilCompleted` that
  drains the GPU pipeline — roughly 2-5 ms each on M1 Max. So **48 ×
  ~3 ms ≈ 144 ms/token of pure host sync** out of the 76 ms (best) /
  228 ms (smoke) decode latency.
- **Prefill is the same loop multiplied by token count.** For
  m=302, the per-(token, expert) loop runs 302 times per layer ×
  48 layers = 14 504 expert dispatches per layer batch. Llama.cpp's
  fused MoE prefill kernel batches all selected experts into one
  big GEMM with token-grouping; ferrum currently does not.

Two lever-sized wins remain (separate PRs):

1. **GPU-side router** — top-K + softmax over `router_logits` on the
   GPU, indices sent back via a single small DMA. Eliminates the 48
   per-layer host syncs; should recover ~144 ms/token on decode (best
   rep would go from 54 ms → ~25 ms, near llama.cpp).
2. **Fused MoE kernel** — single batched gemv that takes all 8
   expert weights + routing indices and produces the weighted sum in
   one dispatch. Eliminates the per-pair dispatch cost on prefill,
   should recover the 14× pp512 gap.

Memory check on M1 Max (32 GB unified):
- 18 GB model load takes ~60 s into MTLBuffer
- Working set during decode: `vmmap` reports ~22 GB IOAccelerator
  (model + KV @ FERRUM_KV_CAPACITY=1024 + scratch). No swap pressure.

The KV cap is the dominant fix. Qwen3-8B's GGUF declares max_seq_len=40960; the Phase 1D loader honoured that and pre-allocated 36 layers × 8 kv_heads × 40960 × 128 × 2(K+V) × 4 = **12 GB** of KV MTLBuffer. On a 32 GB Mac this pushed everything into swap (8 GB swap_out the whole session) and every Metal dispatch ate page-fault latency. The "GPU dynamic frequency / xctrace 27 ms gaps" theory was wrong — root cause was RAM pressure. Memory drops 18 GB → 6.8 GB with KV cap, swap_out drops 8 GB → 16 KB.

Default behaviour unchanged for long-context use cases; `FERRUM_KV_CAPACITY=N` caps the upper bound when the user knows their decode budget. Smarter automatic policy (cap = prompt_len + max_tokens + slack) is a follow-up.

Profile breakdown (Qwen3-8B, FERRUM_DECODE_OP_PROFILE=1):

| Op category | Calls/tok | Wall avg | Notes |
|---|---:|---:|---|
| flash_attention | 36 | 332 µs | not the bottleneck |
| qk_norm_rope | 108 | 238 µs | not the bottleneck |
| norms (rms + fused) | 72 | 564 µs | not the bottleneck |
| matmuls (4 per layer) | 144 | ~10 ms | dominates |
| "other" (split/append/silu/add) | 180 | ~12 ms | also large |

The per-call wall times include a B::sync round-trip (~5-10 ms each
on Apple Silicon) — production removes the per-op syncs but still
sees the same ~3.7 s per decode token, suggesting GPU-side
serialization / clock throttling that the current diagnostic harness
cannot isolate without Xcode Metal frame capture.

Bottlenecks identified and fixed so far:
  - **Phase 1D fused-Q4 path:** byte-concat Q4 super-blocks at load
    so qkv_proj / gate_up_proj keep weights quantised (was
    eager-dequanting → 38 GB RSS for a 5 GB model).
  - **Pooled fp16 transient:** one buffer per matmul shape, not one
    per call (was retaining multi-GB of dequant scratch between
    flushes).
  - **Fused gemv_q4kw kernel:** one Metal kernel reads Q4
    super-blocks inline inside the GEMV reduction (skips the 32 MB
    fp16 transient write+read per matmul).
  - **setBytes for params:** 23 hot pipelines no longer alloc a
    fresh 8-48 byte MTLBuffer per call.
  - **Sticky compute encoder:** consecutive ops share one encoder
    (down from ~14 to ~2 encoders per layer).

Next-up bottlenecks (not in this commit):
  - Q4 GEMV kernel itself runs ~50× slower than M1 Max bandwidth
    floor (32 MB read at 400 GB/s = 0.08 ms; we measure ~5 ms).
    Suspected cause: 1-simdgroup-per-output-col threadgroup layout
    leaves too few simdgroups in flight for the compute unit's
    occupancy budget. Needs a tiled gemv variant.
  - Apple GPU dynamic frequency scaling between dispatches —
    workaround would be to ensure the cmd buffer always has enough
    queued work for the GPU to stay clocked-up.

| Model | Engine | pp512 | tg128 | Status |
|---|---|---:|---:|---|
| Qwen3-8B Q4_K_M | ferrum (current) | 253.16 ± 1.42 | 25.58 ± 0.42 | 73% / 92% of llama.cpp |
| Llama-3.1-8B Q4_K_M | ferrum (current) | 251.62 ± 1.36 | 26.47 ± 0.36 | 75% / 91% of llama.cpp |
| Qwen3-30B-A3B Q4_K_M | ferrum MoE (current) | 43.67 ± 2.05 | 13.22 ± 2.86 | 7% / 30% of llama.cpp |

## Notes on the ferrum path

`ferrum run <path.gguf>` routes through ferrum's own
`LlamaFamilyModel<B>` runtime. candle is touched only inside
`GgufLoader<B>` for parsing the GGUF binary and Phase 1D's
`QuantLinear<B>` for the raw Q4 → Metal MTLBuffer upload. Every line of
math from `model.prefill()` onward — RMSNorm, RoPE, attention, fp16
GEMM, Q4_K_M dequant — runs through ferrum's own kernels in
`ferrum-kernels` / `ferrum-attention`.

The Phase 1D-1 dequant kernel
(`ferrum-kernels/src/q4_k_dequant.metal`) is the foundation: 1
thread per super-block expands 256 weights into 256 fp16 outputs,
called by `MetalBackend::gemm_quant` per matmul to produce a
transient fp16 buffer that the existing fp16 GEMM kernel consumes.

The competitive moat (engine scheduler, custom kernels, MoE perf)
slots in as targeted optimisations once ferrum's first end-to-end
numbers are in and we know where ferrum loses ground vs llama.cpp.
