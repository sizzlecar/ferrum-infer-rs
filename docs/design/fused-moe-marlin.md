# Fused MoE Marlin port — design note

**Status**: planning. Stage 9 (paged-KV) is shipped at 267.7 tok/s c=32 (+95%
over Stage 5 baseline, 14.3% of vLLM). Reaching 80% of vLLM (~1500 tok/s)
needs another 5.6× — primarily blocked on this kernel.

## What ships in vLLM

`vllm/csrc/moe/marlin_moe_wna16/`:
- `kernel.h` (47 LoC): kernel signature macro
- `marlin_template.h` (2241 LoC): the kernel itself, CUTLASS-style template
- `ops.cu` (874 LoC): host-side dispatcher, kernel launch, type plumbing
- `generate_kernels.py` (315 LoC): generates all the type×size specializations

Plus `csrc/moe/moe_align_sum_kernels.cu` (758 LoC) for the prep step.

Total: ~4200 LoC, plus implicit dependencies on `csrc/quantization/marlin/`
headers.

## Why we can't just call vLLM's `.so`

The kernel is shipped as `_moe_C.abi3.so` in the vllm Python wheel — but it
links against PyTorch's libtorch and depends on Python ABI. Loading it from
ferrum would re-introduce the Python runtime ferrum is explicitly designed
to avoid, plus pin us to a specific vLLM minor version.

## What we have already

- `crates/ferrum-kernels/kernels/marlin_cuda_kernel.cu` (843 LoC) — IST-DASLab
  Marlin, single-expert. Vendored.
- `crates/ferrum-kernels/src/marlin.rs` (661 LoC) — Rust glue, GPTQ→Marlin
  tile repack, per-expert offset GEMM call.
- Stage 8: GPU-side `route_topk_softmax` produces `expert_ids[B, top_k]`
  on device. Currently D2H'd back to host because the bucket plan + per-
  expert dispatch loop is host-side.
- Stage 9: paged-KV — orthogonal to MoE compute, already shipped.

## Why fused MoE matters

At c=32 with Qwen3-MoE 30B-A3B, we have ~256 (token, expert) pairs across
~100 active experts per layer. Today's path:

```
host: build dispatch list per active expert
host: for each active expert e:
        launch Marlin kernel with m=tokens_per_expert[e]
host: 4-stream pool serializes ~64 calls per stream
```

Per layer: ~200 Marlin launches + cuEvent sync barriers + per-call setup
overhead. At ~5µs/launch this is ~1ms/layer of overhead alone, plus
serialization that prevents the GPU from running all experts in parallel.

Fused path:
```
gpu (prep): build sorted_token_ids[N_padded] + expert_ids[N_padded/M_BLOCK]
gpu (kernel): ONE Marlin launch where each tile reads its expert_id from
              expert_ids[blockIdx.y], its rows from sorted_token_ids[...]
```

vLLM measures ~3–5× speedup on this segment alone.

## Three port strategies

### A. Verbatim port (full vLLM fidelity)

Copy `marlin_template.h` + `ops.cu` into ferrum, adapt to ferrum's GPTQ
tile layout and Rust glue. Dependencies on `csrc/quantization/marlin/`
get vendored too.

- Pro: get vLLM's exact perf
- Pro: future vLLM updates are mechanical to absorb
- Con: 4000+ LoC port
- Con: ferrum's existing GPTQ tile layout differs from vLLM's int4-packed
  format — would need dual storage OR a repack at load time

### B. Modify ferrum's existing Marlin (smallest diff)

Add expert-id indirection to ferrum's existing `marlin_cuda_kernel.cu`:

```cpp
// before:
//   marlin_cuda(input, weight_e, output, m, n, k, ...)
// where caller has already pointer-arithmetic'd weight_e
// to expert e's tile.
//
// after:
//   marlin_cuda_moe(input, all_experts_weight, output,
//                   sorted_token_ids, expert_ids, n_per_expert, ...)
// kernel reads:
//   int e = expert_ids[blockIdx.y];
//   const int4* B = all_experts_weight + e * (n_per_expert * k / 8);
//   int input_row = sorted_token_ids[blockIdx.y * BLOCK_M + lane];
//   const int4* A = input + input_row * k_per_int4;
```

Plus the prep kernel (sorted_token_ids + expert_ids construction).

- Pro: ~300-500 LoC of targeted modifications
- Pro: keeps existing per-expert path as fallback
- Con: requires careful Marlin kernel surgery — the kernel does intricate
  tile loading and adding indirection without breaking shapes is risky
- Con: ferrum's repacked Marlin tile layout puts `[K-tile-row OUTER,
  N-tile MIDDLE, ik, in_]` per-expert-contiguously already (see
  `1b567be: per-expert repack + concat`), so `e * n_per_expert * k / 8`
  pointer arithmetic does map to expert e's contiguous block. Good news
  for this approach.

### C. Lightweight wrapper that batches launches more efficiently

Keep ferrum's per-expert kernel unchanged. Instead of round-robin across
4 streams, launch one MEGA grid where blockIdx.y = expert_idx and the
kernel internally reads its expert assignment.

This is essentially Strategy B but with less aggressive fusion. The
expert-block doesn't pre-sort tokens — it just scatters via expert_offsets.

- Pro: simpler than B (~100 LoC modification)
- Con: doesn't get the SM-utilization win of Strategy A/B (still launches
  one block per expert, not one block per (token-block, n-block))

## Recommended sequence

1. **Stage 10 (~3-4hr): Strategy B prep kernel** —
   `kernels/moe_align_block_size.cu`. Takes the GPU-side expert_ids[B*K]
   from Stage 8 + bucket plan offsets, produces sorted_token_ids[N_padded]
   + per-tile expert_ids[N_padded / BLOCK_M]. Add Backend trait method
   + CUDA impl. No kernel modification yet — useful regardless because
   it eliminates the host bucket plan rebuild and is graph-capturable.

2. **Stage 11 (~6-8hr): Strategy B Marlin kernel modification** —
   Add expert-id indirection to `marlin_cuda_kernel.cu`. New entrypoint
   `marlin_cuda_moe`. The grid covers all (block, expert) pairs.

3. **Stage 12 (~4hr): Wire into `moe_forward_bucketed`**: replace the
   per-expert dispatch loop with one fused call. Bench. Target: 600+
   tok/s at c=32 on RTX 4090 (3× over Stage 9).

4. **Stage 13 (~4hr): CUDA Graph capture of the MoE block** — now that
   all dispatch decisions are on-GPU and there's just one Marlin call per
   phase, the whole 48-layer MoE forward becomes graph-capturable.
   Target: another +30% via launch overhead elimination.

## Out-of-scope for this design note

- Quant types beyond INT4 (vLLM kernel handles Int8, MXFP4, FP8, etc.)
- Bias support (Qwen3-MoE doesn't use bias in expert MLPs)
- LoRA expert maps
- Multi-GPU TP within MoE expert sharding

These can be added incrementally after the basic INT4 path lands.
