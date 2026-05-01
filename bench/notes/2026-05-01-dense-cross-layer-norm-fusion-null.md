# Dense cross-layer norm fusion — null result

**Date**: 2026-05-01 · **Hardware**: M1 Max 32 GB
**Models tested**: Qwen3-8B-Q4_K_M, Llama-3.1-8B-Q4_K_M

## Hypothesis

Port the MoE PR #49 cross-layer norm fusion to dense `LlamaFamilyModel`:
- End of layer N: instead of `add_inplace(residual, mlp_out)`, run
  `fused_add_rms_norm(residual, mlp_out, layers[N+1].input_ln_w, …, norm_out)`
- Start of layer N+1: skip the leading `rms_norm` (norm_out is pre-populated)

Saves 1 dispatch per layer transition (31 dispatches per token at 32 layers).
For Qwen3-MoE this gave ~30% of the PR #49 decode win.

## What was built

`forward_layer_with_fusion` variant added to `LlamaFamilyModel`,
mirroring `qwen3_moe.rs::forward_layer`'s `next_layer_idx` /
`prev_did_norm_fusion` parameter pair. Decode path threads the flag
across layers; prefill / multi-position decode paths kept on the
plain `forward_layer`.

Build clean. Output correct on both models ("The capital of France is
Paris…").

## Result: regression, not improvement

| Model | Before fusion | After fusion | Δ |
|-------|--------------:|-------------:|---:|
| Qwen3-8B tg128 | 30.7 t/s | 29.1 t/s | **-5%** |
| Llama-3.1-8B tg128 | 32.0 t/s | 30.5 t/s | **-5%** |

3 trials each, prompt "Once upon a time", `--max-tokens 128`.

## Why it didn't transfer from MoE

MoE's `weighted_sum_residual_norm_stacked` (PR #49) fuses the
**weighted sum across `top_k` slots** + residual-add + rms-norm into
one kernel. The win came from collapsing many small writes (8 expert
slots × hidden) into a single threadgroup-level reduction. The fused
norm at the layer boundary was a side-effect of an already big
fusion.

Dense's add+rms-norm pair is two simple ops with no per-row reduction
to amortise. Wrapping them into `fused_add_rms_norm`:

- **Doesn't save bandwidth**: residual was already hot in cache after
  `add_inplace`; the NEXT layer's `rms_norm` reads it from L1, not DRAM.
- **Adds work per call**: the fused kernel does add + sumsq + sqrt +
  scale + mul; plain rms_norm only does sumsq + sqrt + scale + mul.
- **Forces a longer dependency chain**: in serial encoder mode (current
  default), the fused kernel can't start until `down_proj` finishes
  writing `mlp_out`; the unfused pair lets `add_inplace` run first
  and cache-warm `residual` before `rms_norm` runs.

The `fused_add_rms_norm` kernel is the *same* one used at post-attn
(step 7), where it works fine — but at that position it's collapsing
two ops that *both* touch DRAM (post-`o_proj` `o_proj_out` is large
and was just written from off-chip). At the layer-transition position,
the input pair is already cache-resident, so fusion has nothing to save.

## Lesson

**Fusion only helps when the un-fused version pays for an extra DRAM
trip**. Same kernel, different position → different cost / benefit.

Quick predictor: if both inputs of the candidate fusion have already
been touched by a kernel that finished < ~1 ms ago, they're probably
in the system-level cache (24 MB on M1 Max) and fusion won't save
bandwidth.

## What this does NOT preclude

- Lower-level kernel rewrites (different threadgroup sizing) on the
  individual `rms_norm` / `add_inplace` kernels.
- A SDPA-style "wide TG" rewrite of `add_inplace` if it's measurably
  bottlenecked by fixed per-call overhead (it isn't — already 256
  threads/TG).
- The MoE-style fusion at the LM-head transition (last layer's
  `add_inplace` + `final_norm`) — different cost profile, untested.

## Status

Branch `perf/dense-cross-layer-norm-fusion` discarded. Code change
reverted from main. This note captures the experiment so the next
person doesn't repeat it without seeing the cost-benefit difference
from MoE's superficially similar pattern.
