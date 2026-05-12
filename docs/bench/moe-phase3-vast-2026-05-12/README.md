# MoE Phase 1.5+2+3 bench — Vast 4090 — 2026-05-12

A/B validation of the routing-on-device + CUDA Graph capture stack
landed on `main` between 2ac0088 (Phase 1.5) and e423636 (Phase 3 +
release fix). Model: Qwen3-30B-A3B-GPTQ-Int4. Hardware: Vast.ai RTX
4090 instance 36555702. Driver 580.126.09, CUDA-max-good 13.0.

Each variant runs `bench/v0.2-cuda/bench_phase3_variants.sh <name>`,
which spawns a fresh `ferrum serve` with the variant's env vars +
sweeps c=1/8/16/32 via `ferrum bench-serve --random-input-len 256
--random-output-len 128 --num-prompts $((c*4))`.

## Variant matrix

| Variant       | Env vars added on top of m3_bench_serve.sh defaults |
|---------------|------------------------------------------------------|
| baseline      | (none — matches PR #173 reference)                  |
| device_route  | `FERRUM_MOE_DEVICE_ROUTE=1`                         |
| graph         | `FERRUM_MOE_DEVICE_ROUTE=1 FERRUM_MOE_GRAPH=1`      |

Always-on for all variants: `FERRUM_VLLM_MOE=1`,
`FERRUM_KV_MAX_BLOCKS=4096`, `FERRUM_PAGED_MAX_SEQS=32`,
`FERRUM_MOE_BUCKETED=1`, `FERRUM_MOE_STREAMS=4`,
`FERRUM_MARLIN_SKIP_WS_ZERO=1`.

## Results

| c  | baseline  tok/s | device_route tok/s | graph tok/s | Δ device_route vs baseline | Δ graph vs device_route |
|----|----------------:|-------------------:|------------:|---------------------------:|------------------------:|
| 1  | 132.4           | 146.0              | 146.6       | **+10.3%**                 | +0.4% (noise)          |
| 8  | 450.9           | 525.3              | 519.2       | **+16.5%**                 | −1.2% (noise)          |
| 16 | 608.1           | 680.0              | 701.7       | **+11.8%**                 | +3.2% (marginal)       |
| 32 | 734.8           | 847.7              | 846.7       | **+15.4%**                 | −0.1% (noise)          |

TPOT (ms):

| c  | baseline | device_route | graph |
|----|---------:|-------------:|------:|
| 1  | 7.24     | 6.56         | 6.52  |
| 8  | 15.85    | 13.55        | 13.71 |
| 16 | 23.27    | 20.72        | 20.00 |
| 32 | 38.08    | 32.77        | 32.77 |

vs vLLM 0.20.1 reference (PR #102): c=32 ~1870 tok/s. We were at
~38% pre-refactor, now **~45%** after device_route. Phase 3 graph
capture adds nothing measurable.

## Findings

### Phase 2 (device_route) — the real win

Skipping `try_gpu_route_topk_into_host` (D2H + cuStreamSynchronize) +
host `vllm_routing` builder + per-layer phase1/3 dispatch list build
saves ~5.3ms/token at c=32 (TPOT 38.08 → 32.77ms). At 48 layers per
forward this is ~110µs per layer of host roundtrip eliminated.

Did NOT expect this big a win — the host path was supposed to be
"redundant but harmless". Reality: per-layer D2H + synchronize stalls
the GPU between layers. With 48 layers × 32 tokens per c=32 round of
~600 decode iters, the savings compound.

### Phase 3 (CUDA Graph) — marginal + has a capture bug

The capture wrapper itself works for ONE iteration then trips a
`cuGraphLaunch: CUDA_ERROR_INVALID_VALUE` on the first PRE-capture
replay (the immediate POST-capture replay always succeeds). After
that, `batched_graph_failed=true` sticks and the server stays eager
for the rest of its lifetime.

Even with the bug, the eager-fallback numbers match `device_route`
within noise — so the bench mostly measures device_route + a small
amount of warmup overhead.

Theoretical max win from graph capture on this MoE: ~3% (Llama
unified gets +5% at c=16 on a smaller model). At c=32 the GPU is
already memory-bandwidth-bound, so eliminating CPU-side launch
overhead doesn't help.

**Known issue, not yet fixed:** the first pre-capture replay fails
with INVALID_VALUE. Capture + post-capture replay both succeed.
Hypothesis: something between post-replay completion and the next
decode entry (ensure_kv? ensure_scratch? embedding_lookup?)
invalidates the cuGraphExec. Llama unified works in the same
engine path — so the issue is likely MoE-specific (some kernel
inside `moe_forward_bucketed` that doesn't compose with graph
capture). Left for a follow-up session.

## Reproduction

```bash
# On a Vast 4090 pod with /workspace/ferrum-infer-rs at e423636+:
cd /workspace/ferrum-infer-rs
cargo build --release --features cuda,vllm-moe-marlin --bin ferrum
bash bench/v0.2-cuda/bench_phase3_variants.sh baseline
bash bench/v0.2-cuda/bench_phase3_variants.sh device_route
bash bench/v0.2-cuda/bench_phase3_variants.sh graph
```

Each variant takes ~6 min (build + serve + bench-serve sweep). Pod
cost: ~$0.27/hour.

## Recommended next steps

1. **Ship Phase 2 default-on**: `FERRUM_MOE_DEVICE_ROUTE` default
   should flip to `1` (or just be removed and become unconditional)
   given the +15% c=32 win + no parity regressions.
2. **Debug the Phase 3 capture bug** (deferred): instrument
   `replay_graph` to log the cuGraphLaunch call site / args; trace
   what happens between post-replay and the next call's pre-replay.
   Likely a kernel inside `moe_forward_bucketed` interacts badly
   with graph capture lifetime. ~half day.
3. **Investigate the c=32 cliff** vs vLLM (~45% gap): expected next
   bottlenecks are (a) MoE phase 1/3 GEMM utilization
   (`gemm_phase_vllm`'s `vllm_moe_c_tmp` reduce overhead), (b) the
   per-layer post-attn + MoE residual pattern (still 6+ kernels per
   layer where vLLM has 2-3). This is multi-day algorithmic work.
