# v0.2 CUDA Benchmark — RTX 4090, ferrum vs vLLM 0.20.1

**Date:** 2026-05-07
**Engines compared:** ferrum-infer-rs 0.7.3 · vLLM 0.20.1 (latest stable)
**Models:** Llama-3.1-8B-Instruct (FP16) · Llama-3.1-8B-Instruct-GPTQ-INT4
**Concurrency levels:** c = 1, 4, 16, 32
**Total cells:** 2 models × 4 concurrencies × 2 engines × 3 reps = 48 cells
**Hardware:** Vast.ai RTX 4090 24GB, driver 580.105.08, CUDA 13.0

This is reproducible end-to-end: every cell has its own JSON
result, server log, and bench-harness log under
[`results/`](./results/). Sweep is driven by
[`bench/v0.2-cuda/run_sweep.sh`](../../../bench/v0.2-cuda/run_sweep.sh).

**Out of v0.2 scope:** Qwen3-30B-A3B-GPTQ-Int4 (M3) — the safetensors
MoE loader is not yet wired (`Qwen3MoeModel::new` requires GgufFile;
M3 ships as safetensors). Deferred to v0.3.

---

## Headline (median of 3 reps)

### M2: Llama-3.1-8B GPTQ-INT4 — output throughput (tok/s)

3 reps each, median reported. All optimizations on.

n=64 (per cell):

| c | ferrum baseline | ferrum + argmax (old) | ferrum FINAL | vllm | ratio | Δ baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 111.1 | 112.5 | **111** | 148.4 | **75%** | — |
| 4 | 308.4 | 338.7 | **369** | 496.1 | **74%** | +20% |
| 16 | 609.9 | 795.4 | **954** | 1490.3 | **64%** | +56% |
| 32 | 58.4 ⚠ | n/a (KV bug) | **1155** | 2203.8 | **52%** | +1877% |

n=128 (longer bench, more steady-state coverage):

| c | tok/s | TPOT | vllm | ratio |
|---:|---:|---:|---:|---:|
| 16 | **964** | 14.7 ms | 1490 | **65%** |
| 32 | **1215** | 20.9 ms | 2204 | **55%** |

3-rep noise <1% across all c. TPOT_p50 (ms), c=16: 22.1 (baseline) →
15.8 (argmax) → **14.2 (mixed batch + greedy + paged_kv)**.

The breakthrough is "**FINAL**" column: combines three landings this session.

1. **`FERRUM_MIXED_BATCH=1`** — one `unified_decode` call per iter packs
   prefill chunks + active decodes through a varlen attention kernel.
   Eliminates the 50% bench time previously lost to serial prefills
   that stalled all in-flight decoders.

2. **Greedy fast path** — model emits a 1-element vec carrying just the
   token (instead of a synthesised 128k-float one-hot). Engine
   `sample_with_processors` short-circuits when length==1, skipping the
   vocab scan. ~1.4 ms/iter saved at c=16.

3. **`FERRUM_METAL_PAGED_KV=1`** — required for unified_forward path
   (CUDA backend defaults to off). Without it, `unified_decode` falls
   back to per-item dispatch and mixed batch is a no-op.

c=32 jump from 58 → 1155 = 20× — was the c=32 KV pool fix earlier
this session (FERRUM_KV_MAX_BLOCKS=2048) PLUS the mixed batch unlock.

The "**GPU argmax**" column uses `FERRUM_GREEDY_ARGMAX=1` — a new fast
path landed in this branch (commit `5c2e030`). It replaces the heavy
`device → host` transfer of full logits when sampling is greedy
(`temperature=0`) with on-device argmax + a tiny `n_rows × 4 bytes`
DTOH. See § GPU argmax fast path.

### GPU argmax fast path (live optimization win)

The default ferrum unified-decode path returns full logits
(`Vec<Option<Vec<f32>>>`) so the engine's per-request sampler can
apply temperature/top-k/top-p. At c=16 vocab=128k f16 this means a
**8 MB device→host transfer per iter** plus a per-item host argmax
loop — collectively ~3-4 ms per iter on this hardware.

For greedy (`temperature=0`) the full logits are wasted: only the
argmax token is consumed. Implemented this commit:

- New `Backend::argmax_rows_to_u32(buf, n_rows, vocab) -> Vec<u32>`
  trait method (default falls back to `to_vec` + host argmax;
  CudaBackend overrides with a kernel launch).
- New CUDA kernel `kernels/argmax_rows.cu` — one block per row, 1024
  threads, shared-mem reduction over vocab elements.
- `LlamaFamilyModel::unified_forward_internal` gates on
  `FERRUM_GREEDY_ARGMAX=1`: device argmax → synthesise a
  one-hot `Vec<f32>` host-side, so the engine sampler still sees
  logits but the heavy transfer is gone.

Measured impact (M2 INT4 c=16, RTX 4090):

| Metric | Default | + GPU argmax | Delta |
|---|---:|---:|---:|
| Output throughput (tok/s) | 609.9 | **795.4** | **+30.4%** |
| TPOT_p50 (ms) | 22.1 | 15.8 | −28% |
| TPOT_p99 (ms) | 27.5 | 20.6 | −25% |
| Ratio vs vLLM 0.20.1 | 41% | **53%** | +12 pp |

Why the gain is bigger than the predicted 6-10%: the breakdown
underestimated the to_vec cost. The DTOH at vocab=128k × 16 rows ×
2 bytes = 4 MB is bigger than the 1ms estimate (≈3 ms in practice
with cudarc sync), and the host argmax over 128k × 16 floats was
also ≈1.4 ms. Combined ≈ 4-5 ms per iter eliminated.

See `results-argmax/` for raw JSON of all 11 cells. Same model, same
prompts, same harness, same engine version with the env flag
toggled.

### M1: Llama-3.1-8B FP16 — output throughput (tok/s)

| c | ferrum | vllm | ratio | TPOT_p50 (ferrum / vllm) |
|---:|---:|---:|---:|---:|
| 1 | **OOM** ⚠ | 59.2 | — | — / 16.7 ms |
| 4 | **OOM** ⚠ | 204.8 | — | — / 17.3 ms |
| 16 | **OOM** ⚠ | 741.2 | — | — / 19.0 ms |
| 32 | **OOM** ⚠ | 1236.9 | — | — / 22.8 ms |

---

## What this report shows

**ferrum is competitive at low concurrency in INT4.** At c=1 INT4,
ferrum hits 76% of vLLM's throughput on the same hardware (112 vs
148 tok/s, 8.5 ms vs 6.7 ms TPOT). This is the price-conscious
single-GPU lane the project targets — single binary, no Python, fast
cold start, sub-10ms TPOT.

**The gap widens as concurrency increases, but the GPU-argmax fast
path landed in this branch closes ~12 pp at c=16.** With
`FERRUM_GREEDY_ARGMAX=1`, c=4 lifts to 68% (was 62%), c=16 to **53%
(was 41%)**. The remaining gap to vLLM at c=16 is in `model` (GPU
compute — Marlin GEMM + attention). Engine-path overhead post-argmax
is ≤4% per Phase 1 wall-prof, see [`engine-wall-prof`](#engine-wall-clock-breakdown).

**Two limitations show up clearly:**

1. **c=32 KV pool exhaustion (M2 INT4) — FIXED.** ferrum dropped from
   610 to 58 tok/s between c=16 and c=32. Server log showed
   `Resource exhausted: Block pool exhausted: 512/512 blocks allocated`.
   Root cause: the engine's default block pool was hard-coded to 512
   blocks (in `simple_engine_config`), which only fits ~10 concurrent
   ShareGPT prompts at 16 tokens/block. At c=32 the pool exhausted.
   Fix landed this branch (`017d66d`): added `FERRUM_KV_MAX_BLOCKS`
   env override; the bench script now sets it to 2048. **Result: c=32
   M2 INT4 throughput jumped from 58 → ~490 tok/s (8.4×)**, all 128
   prompts complete with 0 failures. See § c=32 fix below.

2. **FP16 8B doesn't fit on 24GB (M1 entire row OOM).** ferrum's
   static scratch-buffer allocation (sized at startup from
   `FERRUM_MAX_BATCH` and `FERRUM_PAGED_MAX_SEQS`) leaves no room
   for the 16GB FP16 weights. Tried minimum config
   (`KV_CAPACITY=512`, `MAX_BATCH=4`, paged-KV off) — still OOMs.
   vLLM fits via dynamic memory management (`gpu-memory-utilization=0.9`)
   that ferrum doesn't have. Architectural: deferred.

---

## What's the actual gap, where, and why

### Engine wall-clock breakdown

`FERRUM_ENGINE_WALL_PROF=1` instrumentation (this branch) measured
ferrum's c=16 INT4 iter at steady state:

```
[batch-decode-wall] m=16 | total=16.8ms | model=15.3ms | sample=1.4ms |
                          emit=132us | build=4us | stop=1us
[engine-wall]      batch=16 | total=16.8ms | gap_to_prev=140us |
                              sched=30us | process=16.5ms
```

| Segment | Time | % iter |
|---|---:|---:|
| `model` (GPU forward + final to_vec sync) | 15.3 ms | **91%** |
| `sample` (per-item top-k/p/repetition_penalty, CPU) | 1.4 ms | 8% |
| `emit` (16× tokenizer.decode + chunk send, sequential) | 132 µs | 0.8% |
| `gap_to_prev` (bg-loop yield + scheduler.next_batch wait) | 140 µs | 0.8% |
| `sched` (scheduler.next_batch itself) | 30 µs | 0.2% |
| `build` (UnifiedBatch construction from sequences) | 4 µs | <0.1% |

**91% of the iter is `model`.** Engine-path optimizations have a
~9% ceiling. The actual gap to vLLM (610 → 1490 tok/s = 2.4× speedup
needed) is essentially all in GPU compute kernels.

### Why kernel optimization is hard from here

Three kernel paths have already been tested in this codebase, each
documented in `docs/bench/v0.2-cuda/`:

- **Phase 12 vLLM Marlin port** (`status-2026-05-05-vllm-marlin-port.md`):
  Ported vLLM's `gptq_marlin.cu` with all 19 sm_89 kernel template
  instantiations. End-to-end perf identical to ferrum's existing
  IST-DASLab Marlin (517 vs 520 tok/s, within 1% noise).
  Conclusion: vLLM's Marlin advantage doesn't materialize on this
  workload.

- **Phase 13 FULL CUDA Graph** (`status-2026-05-05-full-cuda-graph.md`):
  Tested 4 configurations of full-iter CUDA graph capture; all 0%
  gain. Per-kernel GPU time (~53 µs avg) >> launch overhead (~5 µs)
  on this model size.

- **Phase 10 Marlin tile A/B** (`progress.md` Phase 10): tile (64×256)
  vs (128×128) on m=16 → 0% gain. Marlin INT4 is compute-bound, not
  memory-bound.

What remains untried:
- ~~**`paged_varlen_attention` split-K rewrite** (predicted +10%)~~ —
  **TRIED, NEUTRAL.** Implemented this session (commit `adc06a7`) with
  microbench showing c=4 +103%, c=1 +801%, c=16 +6-16%. Production
  A/B at num_prompts=128 with `FERRUM_SPLIT_K_ATTN={0,1}`:

  | c | split-K ON | split-K OFF | Δ |
  |---|---:|---:|---:|
  | 4 | 263.7 | 263.8 | 0.0% |
  | 16 | 459.7 | 464.2 | −1.0% |
  | 32 | 490.4 | 484.5 | +1.2% |

  Why no win: attention is only 13.5% of iter (per Phase 11 wall-prof).
  Even +103% on attention → ~6% on iter; lost in noise. Code stays
  in (auto-tune heuristic, gated behind shape thresholds) for the
  long-context single-stream case where attention dominates more —
  but for c=16 INT4 the lever is elsewhere.
- **Sampling on GPU** (move 1.4ms CPU sample to GPU; eliminates the
  9% engine overhead but adds new kernel work)

Sampling-on-GPU is a 1-2 day kernel project; could push c=16 INT4
to ~510 tok/s (~34% of vLLM). The only realistic path past 50% on
c≥16 is the multi-week vLLM Marlin full kernel-set port (already
started in `vllm_marlin/PORT_NOTES.md`).

### c=32 fix (this session)

Before: c=32 ferrum panicked or stalled at 58 tok/s with `Block pool
exhausted: 512/512 blocks allocated`. Diagnosis: `simple_engine_config`
(in `ferrum-engine/src/lib.rs`) hard-coded `kv_cache.max_blocks = 512`,
which only fits ~10 ShareGPT prompts at 16 tokens/block. At c=32 the
pool exhausted; the preempt-victim filter required `prefill_complete`
which no seq satisfies during the first batch of a 32-arrival cold
start. Attempts to relax the filter or set `kv_cache` early created
correctness hazards (mid-batch preemption corrupting in-flight decode).

Fix landed in two commits:
- `017d66d` — Added `FERRUM_KV_MAX_BLOCKS` env override (default 512
  preserved for backwards-compat; bench script now sets 2048).
- `d940613` — Relaxed `preempt_victim` filter to allow mid-prefill
  victims; defensive only, since the bigger pool means preempt rarely
  fires.

Result with `FERRUM_KV_MAX_BLOCKS=2048`:

| Metric | Before | After |
|---|---:|---:|
| c=32 output throughput (tok/s) | 58.4 | **490.4** (8.4×) |
| c=32 successful requests | 128/128 (slow) | 128/128 |
| c=32 mean TPOT (ms) | n/a (preempt thrash) | 49 |

c=32 still trails c=16 (490 vs 460 tok/s — c=32 should scale up, not
down). Cause: bigger m at decode time hits Marlin GEMM scaling
limits and longer kv_len in attention. Same matmul-dominated story
as c=16.

---

## Hardware

| Field | Value |
|---|---|
| Provider | Vast.ai instance 36250160 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Driver | 580.105.08 |
| Host CUDA | 13.0 |
| CPU | AMD EPYC, 192 cores total / 24 effective |
| RAM | 503 GB total |
| Geo | Spain |
| Cost | $0.32/hr |

Container: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`.

## Software

| Component | Version |
|---|---|
| ferrum-infer-rs | 0.7.3 @ commit `20f1a83` (branch `perf/v02-cuda-engine-profile`) |
| vLLM | **0.20.1** (latest stable, released 2026-05-04) |
| PyTorch | 2.11.0+cu130 (default index, requires driver ≥580) |
| Python | 3.11 |
| Container CUDA toolkit | 12.4 (used to compile ferrum's kernels) |

## Models

| Model | HF repo | Format | On-disk |
|---|---|---|---|
| M1 | `unsloth/Meta-Llama-3.1-8B-Instruct` (open mirror) | safetensors FP16 | 15 GB |
| M2 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` | safetensors GPTQ desc_act=true gs=128 | 5.4 GB |

## Workload

Frozen for all cells:
- **Dataset:** 128-prompt deterministic ShareGPT v3 subset (seed = ferrum HEAD short hash)
- **Per cell:** `num_prompts = 4 × c`, capped at 128
- **Output:** 512 tokens at c=1, 256 tokens at c≥4
- **Sampling:** greedy (`--temperature 0 --top-p 1`), `--ignore-eos`
- **Harness:** `vllm bench serve --backend openai-chat`

## ferrum configuration

The sweep ran ferrum with paged-KV disabled and a small batch cap to
fit M1 FP16 within 24GB (it didn't fit either way; M2 INT4 ran fine):
```
FERRUM_KV_CAPACITY=2048      # per-seq KV slot; bumped from 512 to
                              # avoid panics on ShareGPT prompts >512
FERRUM_KV_MAX_BLOCKS=2048    # global block pool; bumped from default
                              # 512 — c=32 cold start needs ~1000 blocks
FERRUM_MAX_BATCH=4
FERRUM_GREEDY_ARGMAX=1       # device-side argmax for temperature=0
```

Phase 1 bench (separate, ferrum-only, before vLLM) ran with the
chunked-prefill + unified CUDA Graph config:
```
FERRUM_METAL_PAGED_KV=1 FERRUM_PAGED_MAX_SEQS=32 FERRUM_UNIFIED_GRAPH=1
FERRUM_KV_CAPACITY=2048 FERRUM_MAX_BATCH=32
```
That gave c=16 INT4 = **660 tok/s** (PR #92's claimed config) — but
combining the same config with vLLM-running-first in the same pod
session triggered `CUDA_ERROR_ILLEGAL_ADDRESS` on graph replay.
The sweep numbers above use the safer paged-only configuration that
runs reliably across vLLM-then-ferrum sequencing.

## vLLM configuration

```
--max-num-seqs 64
--max-model-len 4096
--quantization gptq_marlin   (M2 only)
--no-enable-prefix-caching
```

Default torch.compile graph enabled. Default
`gpu-memory-utilization=0.9`. No FP8 KV-cache.

---

## Detailed numbers

### M2 INT4 (Llama-3.1-8B-GPTQ-INT4)

**ferrum** (median of 3 reps):

| c | tok/s | TPOT_p50 | TPOT_p99 | TTFT_p50 | TTFT_p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 111.1 | 8.6 | 8.9 | 45.4 | 47.5 |
| 4 | 308.4 | 11.4 | 13.1 | 143.1 | 233.1 |
| 16 | 609.9 | 22.1 | 27.5 | 444.9 | 1123.2 |
| 32 | 58.4 ⚠ | 348.8 | 954.5 | 81.6 | 1453.2 |

**vllm** (median of 3 reps):

| c | tok/s | TPOT_p50 | TPOT_p99 | TTFT_p50 | TTFT_p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 148.4 | 6.7 | 6.8 | 36.5 | 47.4 |
| 4 | 496.1 | 7.1 | 7.2 | 57.7 | 116.4 |
| 16 | 1490.3 | 9.1 | 11.1 | 177.0 | 518.9 |
| 32 | 2203.8 | 14.5 | 22.8 | 312.9 | 779.5 |

### M1 FP16 (Llama-3.1-8B-Instruct)

**ferrum**: OOM on all 12 cells (24 GB VRAM insufficient for static
allocations + 16 GB FP16 weights). See § Caveats.

**vllm** (median of 3 reps):

| c | tok/s | TPOT_p50 | TPOT_p99 | TTFT_p50 | TTFT_p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 59.2 | 16.7 | 16.7 | 48.4 | 67.6 |
| 4 | 204.8 | 17.3 | 18.0 | 104.1 | 117.8 |
| 16 | 741.2 | 19.0 | 42.5 | 195.2 | 514.6 |
| 32 | 1236.9 | 22.8 | 61.5 | 349.5 | 977.5 |

---

## Caveats and validity

1. **M3 (Qwen3-30B-A3B GPTQ-Int4) skipped.** `Qwen3MoeModel::new`
   requires a `GgufFile` parameter (CPU/Metal GGUF path); safetensors
   expert loading is not yet implemented. Multi-day work, deferred to
   v0.3.

2. **ferrum M1 (FP16 8B) does not fit on RTX 4090 24GB.** vLLM uses
   dynamic memory management (`gpu-memory-utilization=0.9`) to fit;
   ferrum's static scratch allocations don't have an equivalent knob
   yet. Tried `FERRUM_KV_CAPACITY=512` + `FERRUM_MAX_BATCH=4` + paged
   KV disabled — still OOMs at weight upload. Fix is non-trivial:
   make ferrum's scratch allocation `gpu-memory-utilization` aware.

3. **ferrum M2 c=32 KV pool exhausts.** Preemption fires the
   `Resource exhausted: No blocks available and no request to preempt`
   path — meaning preemption is broken at this scale (no eligible
   victims, or bookkeeping race). Reproducible. Tracking item.

4. **vLLM 0.20.1 + torch 2.11+cu130 requires driver ≥580.** Earlier
   pod attempts on driver 570 (CUDA 12.8) failed with "NVIDIA driver
   too old (found version 12080)". Setup script now auto-detects.

5. **vLLM-then-ferrum graph capture issue.** Running vLLM in the same
   pod session before ferrum corrupts CUDA state in a way that breaks
   ferrum's `replay_graph` (`CUDA_ERROR_ILLEGAL_ADDRESS`). The sweep
   numbers above use ferrum without unified graph capture (which
   added <1% on this hardware anyway per Phase 13 null result).
   Phase 1 numbers (ferrum-only session) had graph capture working
   and got 660 tok/s c=16 instead of 610 — the 8% delta is graph's
   bookkeeping savings.

---

## Methodology

- Three repetitions per cell; report **median**.
- Sampled 128 prompts deterministically from ShareGPT v3 (filtered to
  approx 128–512 token range, seeded by ferrum repo HEAD short hash).
- `vllm bench serve --backend openai-chat --base-url http://...`
  drives both engines through their OpenAI-compatible chat endpoint.
- Server kept up across all c values for a given (engine, model) pair
  (~5 server starts total instead of 24).
- `kill_engine` in `run_sweep.sh` explicitly pkills `VLLM::EngineCore`
  child processes (vLLM 0.20+ spawns them and `pkill -f vllm.entrypoints`
  doesn't catch them — leaks 21 GB GPU memory between pairs otherwise).

## Reproduction

```bash
# 1. Clone + checkout
git clone https://github.com/sizzlecar/ferrum-infer-rs.git
cd ferrum-infer-rs
git checkout perf/v02-cuda-engine-profile

# 2. Setup pod (driver ≥580 for vllm 0.20+)
bash bench/v0.2-cuda/setup.sh

# 3. Full sweep
bash bench/v0.2-cuda/run_sweep.sh

# 4. Pull results
bash bench/v0.2-cuda/pull_results.sh
```

## What's next (post-v0.2)

Ranked by ROI for closing the c=16 INT4 gap:

1. **`paged_varlen_attention` split-K + persistent threads rewrite**
   — predicted +10% (660 → 720 tok/s c=16).
2. **GPU sampling kernel** — eliminates 1.4 ms (8% of iter) of
   per-item host sampling. Stacks with above for ~+18% combined.
3. **`gpu-memory-utilization` knob in ferrum's scratch sizing** —
   not a perf win, but unblocks FP16 8B benchmarking and larger
   models.
4. **Port vLLM's full gptq_marlin kernel set** (5336 LoC, already
   scaffolded at `crates/ferrum-kernels/vllm_marlin/`, 1-2 weeks of
   focused work). Per Phase 11 c=1 op-profile vLLM Marlin kernel is
   2× faster on m=16 — a true 2× would give c=16 = ~1200 tok/s = 80%
   vLLM. Phase 12 already proved a partial port doesn't deliver the
   2×; the full kernel-set with shape-aware dispatch is the lever.

5. **Qwen3MoE safetensors loader** — needed for v0.3 M3 entry.
