# Qwen3-MoE paged-KV mirror — c=16 ferrum **matches llama.cpp** (2026-05-02)

## Summary

Ported `LlamaFamilyModel`'s paged-KV architecture into `Qwen3MoeModel`,
unblocking the existing `paged_decode_attention` Metal kernel for the
batched-decode path. The kernel folds 16 sequential m=1 flash_attn
calls + ~6 dispatches/item of plumbing into a single
`paged_decode_attention(num_seqs=m)` dispatch.

**Result on Qwen3-30B-A3B Q4_K_M / M1 Max:**

| c  | per_token tok/s | batched paged tok/s | Δ      | llama.cpp ref |
|---:|---------------:|--------------------:|-------:|--------------:|
| 4  | **42.1**       | 38.9                | -8%    | (n/a)         |
| 8  | 47.4           | **58.9**            | +24%   | (n/a)         |
| 16 | 47.9           | **79.7**            | +66%   | 78.9          |

| c  | per_token TPOT med | batched paged TPOT med |
|---:|------------------:|-----------------------:|
| 4  | 82 ms             | 90 ms                  |
| 8  | 158 ms            | **121 ms**             |
| 16 | 298 ms            | **164 ms**             |

**At c=16, ferrum 79.7 tok/s matches llama.cpp 78.9** on the same
machine + same model + same bench harness.

The crossover is between c=4 and c=8: paged batched needs at least
c=6-8 to amortize the bigger setup work (per-item paged write +
host-side block_table stack + larger output copy). At c=4 the
4-sequence GPU grid (1, num_heads=32, num_seqs=4) doesn't saturate
M1 Max; per-token mode wins by ~8%. The default for c=4 stays
per-token; paged batched is the c≥8 path.

## Why this works

Three rounds of MoE FFN optimization (#79 offset-aware,
df64ac1 fused-gate-up-silu, e85baeb / 3b68b91 batched MoE GEMV) each
delivered +1-2% — confirming the FFN itself wasn't the bottleneck. GPU
Frame Capture (Xcode) showed the c=16 batched-decode path's
`forward_layer_batched_decode` was spending **55 ms / round (20% of
total)** on a per-item attention loop:

  ```
  for i in 0..m:           # m=16 sequential iterations
      copy_slice q          # 1 dispatch
      copy_slice k          # 1
      copy_slice v          # 1
      qk_norm_rope (q)      # 1 (m=1)
      qk_norm_rope (k)      # 1
      qk_norm_rope (v)      # 1
      kv_cache_append        # 1
      flash_attn (m=1)      # 1
      copy_slice output      # 1
  ```
  9 dispatches × 16 items × 48 layers = **6912 dispatches per round**
  on the attention block alone.

The replacement uses two existing kernels that LlamaFamilyModel was
already exercising for c=16 dense (where it gets +40% over per-token):

  - `split_qkv_norm_rope_into_paged_cache` — fused split + norm + RoPE
    + paged cache write. m dispatches, ONE per item, vs 4 per item.
  - `paged_decode_attention(num_seqs=m)` — single Metal launch covers
    all m sequences' decode attention, reading per-seq K/V from the
    shared paged pool via per-seq block_tables. Replaces the 16 ×
    m=1 flash_attn calls with ONE m=16 dispatch.

Per-layer attention dispatch count drops from 9m to m+1: at m=16 /
48 layers that's **6912 → 816 dispatches per round**.

## Implementation

Mirrors LlamaFamilyModel's Phase 4 paged-KV scaffolding into
`Qwen3MoeModel`, plus a paged branch in `forward_layer_batched_decode`:

  - `Qwen3MoeScratch` gets `paged_batch_q`, `paged_batch_o`,
    `paged_batch_block_tables`, `paged_batch_context_lens`,
    `paged_max_blocks_per_seq` (lazy-init via `enable_paged_batch`).
  - `Qwen3MoeModel` gets `paged_pools: Option<Vec<(B::Buffer, B::Buffer)>>`
    and `paged_block_alloc: Option<Mutex<BlockAllocator>>`.
  - `ensure_kv` on `FERRUM_METAL_PAGED_KV=1` allocates the shared pool
    on first call, hands per-cache_id `block_table` + `context_lens`
    metadata, and pre-allocates `max_blocks_per_seq` blocks from the
    allocator.
  - `release` (was a stub for the contiguous path) returns each
    cache_id's blocks to the allocator on cache_id eviction. This is
    the critical piece that lets a long-running bench cycle
    cache_ids 0..32 through a pool with only `MAX_SEQS` slots
    without exhaustion.
  - `forward_layer` (m=1 path) gets a paged branch that calls
    `split_qkv_norm_rope_into_paged_cache` + `paged_decode_attention(num_seqs=1)`.
  - `forward_layer_batched_decode` (m≥2 path) gets a paged branch that
    runs:
      1. m × `split_qkv_norm_rope_into_paged_cache` (1 dispatch / item)
      2. host-side stack of block_tables + context_lens, upload via
         `B::write_u32`
      3. ONE `paged_decode_attention(num_seqs=m)`
      4. m × `copy_slice` from `paged_batch_o[i]` → `attn_flat[i*q_dim]`

Total: **m × 2 + 2 dispatches/layer for attention** (vs 9m before).
At m=16: 34 vs 144 — a **4.2× dispatch count reduction** AND a much
larger GPU-time saving from the single batched flash vs 16 sequential.

## Tuning

`FERRUM_PAGED_MAX_SEQS` must accommodate **(in-flight + about-to-release)**
sequences. Bench harness creates a fresh cache_id per request; the
release fires after each request, but transient overlap means at
peak we briefly have c+δ sequences with allocated blocks. **Default
to `2 × max_concurrency`** (= 32 for c=16). Combined with
`FERRUM_KV_CAPACITY=512` (32 blocks/seq), the pool is **3.1 GB**
and fits comfortably alongside the 18.6 GB GGUF mmap on a 32 GB Mac.

Reproduce:

```bash
cargo build --release --features metal -p ferrum-cli
./bench/group-a-paged-kv-2026-05-02/run_ab.sh 16
```

## Lesson

Three rounds of MoE FFN optimization were null because the bottleneck
moved earlier than I'd located it. xctrace was the inflection: per-stage
profile data showed MoE FFN was 70% of round time but it was already
near-saturated on the GPU side. **The +66% throughput came from one
architectural change (paged-KV mirror), not any kernel-craft tuning.**
The 200 LOC of mirror code from LlamaFamilyModel — which already
delivered +40% c=16 dense — was sitting unused for Qwen3-MoE because
of a single skipped Phase 4 wiring item.

When kernel-level optimization plateaus at +1-2%, look at the
**architectural bottleneck above the kernel**: dispatch count,
batched-vs-serial structure, pipelining. That's where the 1.5×+ wins
hide, and they don't show up in a microbench.
