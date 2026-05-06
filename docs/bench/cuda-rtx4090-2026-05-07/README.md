# v0.2 CUDA Benchmark — RTX 4090, ferrum vs vLLM

**Date:** 2026-05-07
**Engines compared:** ferrum-infer-rs · vLLM 0.20.1
**Models:** Llama-3.1-8B-Instruct (FP16) · Llama-3.1-8B-Instruct-GPTQ-INT4
**Concurrency levels:** c = 1, 4, 16, 32
**Total cells:** 2 models × 4 concurrencies × 2 engines × 3 reps = 48 cells

This report is reproducible end-to-end from this directory: every cell has its
own JSON result and server log. Sweep is driven by
[`bench/v0.2-cuda/run_sweep.sh`](../../../bench/v0.2-cuda/run_sweep.sh).

**Out of v0.2 scope:** Qwen3-30B-A3B-GPTQ-Int4 (M3) — the safetensors MoE
loader is not yet wired (currently GGUF-only via `Qwen3MoeModel::new`); will
land in v0.3.

---

## Headline

(Filled in after sweep completes — see Phase-1 numbers below for what to expect.)

| Model | c | ferrum tok/s | vllm tok/s | ratio |
|---|---|---|---|---|
| Llama-3.1-8B-INT4 (M2) | 1 | _TBD_ | _TBD_ | _TBD_ |
| Llama-3.1-8B-INT4 (M2) | 4 | _TBD_ | _TBD_ | _TBD_ |
| Llama-3.1-8B-INT4 (M2) | 16 | _TBD_ | _TBD_ | _TBD_ |
| Llama-3.1-8B-INT4 (M2) | 32 | _TBD_ | _TBD_ | _TBD_ |
| Llama-3.1-8B-FP16 (M1) | 1 | _TBD_ | _TBD_ | _TBD_ |
| Llama-3.1-8B-FP16 (M1) | 4 | _TBD_ | _TBD_ | _TBD_ |
| Llama-3.1-8B-FP16 (M1) | 16 | _TBD_ | _TBD_ | _TBD_ |
| Llama-3.1-8B-FP16 (M1) | 32 | _TBD_ | _TBD_ | _TBD_ |

(Phase 1 quick check: ferrum 660 / vllm 1425 at c=16 INT4 = 46% — see § Phase 1.)

## Hardware

| Field | Value |
|---|---|
| Provider | Vast.ai instance 36250160 |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Driver | 580.105.08 |
| Host CUDA | 13.0 |
| CPU | (TBD from `_env.txt`) |
| RAM | 503 GB total, 168 GB free |
| Geo | Spain |
| Hourly cost | ~$0.32/hr |

## Software versions

| Component | Version | Source |
|---|---|---|
| ferrum-infer-rs | 0.7.3 @ `0ab5f73` (branch `perf/v02-cuda-engine-profile`) | `cargo build --release --features cuda` |
| vLLM | 0.20.1 (latest stable, released 2026-05-04) | `pip install "vllm[bench]==0.20.1"` |
| PyTorch | 2.11.0+cu130 | `pip install torch==2.11.0` (default index) |
| Rust | stable | rustup default |
| Container CUDA toolkit | 12.4 | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` |

## Models

| Model | HF repo | Format | On-disk |
|---|---|---|---|
| M1 Llama-3.1-8B FP16 | `unsloth/Meta-Llama-3.1-8B-Instruct` (open mirror) | safetensors | 15 GB |
| M2 Llama-3.1-8B GPTQ-INT4 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` | safetensors (desc_act=true, group_size=128) | 5.4 GB |

## Workload

Frozen for all cells:
- **Dataset:** vLLM bench `--dataset-name random` (seeded)
- **Prompt length:** 256 tokens
- **Output length:** 128 tokens
- **Sampling:** greedy (`--temperature 0 --top-p 1`)
- **EOS:** `--ignore-eos` (decode runs to full 128 output tokens)
- **Per cell:** 32 prompts at the configured concurrency, 3 reps

## ferrum configuration

ferrum runs in **chunked-prefill + CUDA Graph capture** mode:
```
FERRUM_KV_CAPACITY=2048
FERRUM_MAX_BATCH=32
FERRUM_METAL_PAGED_KV=1
FERRUM_PAGED_MAX_SEQS=64
FERRUM_UNIFIED_GRAPH=1
```
This is the production-quality decode path landed in PR #92 (commit
`7561af2`), routing both prefill chunks and decode tokens through the
unified `[M_total, hidden]` forward pass via `paged_varlen_attention`.

## vLLM configuration

```
--max-num-seqs 64
--max-model-len 4096
--quantization gptq_marlin (M2 only)
--no-enable-prefix-caching
```
Default torch.compile graph is enabled. Default `gpu-memory-utilization`.
No FP8 KV-cache, no chunked-prefill toggle (vLLM v1 has it default-on).

---

## Phase 1: documented baseline check (sanity)

Before running the full sweep, we verified the reported numbers from
`docs/bench/v0.2-cuda/status-2026-05-05-chunked-prefill.md` reproduce
on this hardware:

| Cell | Config | tok/s (median 3 reps) | TPOT_p50 |
|---|---|---|---|
| A | ferrum, no paged, no graph | 471 | 25.5 ms |
| B | ferrum, paged + unified graph | 660 | 20.3 ms |
| C | B + `FERRUM_ENGINE_WALL_PROF=1` | 660 | 20.0 ms |
| D | vllm 0.20.1 | 1425 | 8.9 ms |

ferrum chunked-prefill gain over baseline: **+40%** (matches PR #92's claimed +39%).
ferrum vs vllm at c=16 INT4: **46%** (better than the 36% the prior pod showed,
but still a 50%+ gap).

## Engine wall-clock breakdown

`FERRUM_ENGINE_WALL_PROF=1` was added in this branch to find where ferrum's
iter time goes. At c=16, M2 INT4, steady-state:

```
[batch-decode-wall] m=16 | total=16.8ms | build=4us | model=15.3ms | sample=1.4ms | emit=132us | stop=1us
[engine-wall]      batch=16 | total=16.8ms | gap_to_prev=140us | sched=30us | process=16.5ms
```

| Segment | Time | % iter |
|---|---|---|
| `model` (unified_decode + GPU forward + to_vec) | 15.3 ms | **91%** |
| `sample` (per-item top-k/p/repetition_penalty) | 1.4 ms | 8% |
| `emit` (per-item tokenizer.decode + chunk send) | 132 µs | 0.8% |
| `gap_to_prev` (bg-loop yield + scheduler.next_batch wait) | 140 µs | 0.8% |
| `sched` (scheduler.next_batch itself) | 30 µs | 0.2% |
| `build` (UnifiedBatch construction from sequences) | 4 µs | <0.1% |

**Implication:** the gap to vLLM is 99% inside `model` (Marlin GEMM dominates,
per [Phase 11 c16-opprofile](../v0.2-cuda/status-2026-05-05-c16-opprofile.md):
matmul = 65% of iter, attn = 13.5%). Engine-path optimization (sample/emit/etc.)
has at most a ~5% headroom on this workload — not the 50%+ delta observed.

This corrects the prior reading of the Phase 11 doc (which said "65% of
wall-clock outside the timed iter"). That measurement was inflated by
`B::sync()` barriers added during profiling; once the syncs are removed,
real engine overhead is ~9%.

## Methodology

(TBD: ShareGPT-vs-random, deterministic seeding, prefix-cache disabled
explicitly, vllm bench serve invocation, max-model-len rationale)

## Reproduction

```bash
# 1. Clone + checkout
git clone https://github.com/sizzlecar/ferrum-infer-rs.git
cd ferrum-infer-rs
git checkout perf/v02-cuda-engine-profile

# 2. Setup pod (needs CUDA 13 host driver ≥580 for vllm 0.20+ torch 2.11+cu130)
bash bench/v0.2-cuda/setup.sh

# 3. Run full sweep
bash bench/v0.2-cuda/run_sweep.sh

# 4. Pull results
bash bench/v0.2-cuda/pull_results.sh
```

## Caveats

1. **M3 (Qwen3-30B-A3B GPTQ-INT4) skipped.** `Qwen3MoeModel::new` requires
   a GgufFile parameter (CPU/Metal GGUF path); safetensors expert loading
   is not yet implemented. Adding it is multi-day work, deferred to v0.3.

2. **vLLM 0.20.1 + torch 2.11+cu130 requires driver ≥580.** Hosts with
   driver 555-575 (cuda_max_good 12.6/12.7/12.8) need either an older vllm
   or the `cu128` torch wheel via `--index-url
   https://download.pytorch.org/whl/cu128`. setup.sh now auto-detects.

3. **ferrum c=1 path is different.** At c=1 the engine routes through the
   legacy `run_decode_step` (not `run_batch_decode` / `unified_decode`);
   chunked-prefill and CUDA Graph capture do not apply at c=1.

4. **Prior pod reported 714 tok/s** (status-2026-05-05-chunked-prefill.md);
   this pod measures 660 tok/s in the same config. RTX 4090 24GB +
   driver 580 vs prior pod's setup is the only differentiator. Numbers are
   pod-specific.

## What's next (post-v0.2)

- **Port vLLM's gptq_marlin kernel completely** — scaffolding at
  `crates/ferrum-kernels/vllm_marlin/` (commit `30aa283`); 5336 LoC, 1-2
  weeks. Phase 12 attempt was correctness-only, did not deliver perf;
  needs `kernel_selector.h` template generation + per-shape dispatch tuning.
- **Qwen3MoE safetensors loader** — required for M3 in v0.3.
- **paged_varlen_attention split-K rewrite** — predicted +10% per
  status-2026-05-05-chunked-prefill.md "Known gaps vs vLLM" section.
- **Sampling on GPU** — predicted +8%; only a saving if `model` time drops.
