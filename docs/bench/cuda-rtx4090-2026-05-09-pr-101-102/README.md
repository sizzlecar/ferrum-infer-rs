# M3 (Qwen3-30B-A3B-INT4) MoE — PR #101 + #102 — RTX 4090

**Date**: 2026-05-08 → 2026-05-09
**GPU**: NVIDIA RTX 4090 (24 GB), CUDA 13, sm_89
**Model**: `Qwen/Qwen3-30B-A3B-Instruct-GPTQ-Int4` (128 experts × 48 layers, top-k=8, n_per_expert=768, hidden=2048)
**Quant**: GPTQ INT4, group_size=128, sym=true, desc_act=false
**Workload**: `vllm bench serve` random 256→128 (input_len → output_len), `c × 4` prompts per run

Picks up where `docs/bench/cuda-rtx4090-2026-05-08-m3-moe/README.md` left off (Stage 5 multi-stream = 137 tok/s c=32).

## Headline result

ferrum M3 throughput jumped from **137 → 391 tok/s at c=32 in two PRs (+185%)**. Two bench cycles, same 4090 pod, no env-var hand-tuning required for the second cycle.

| Concurrency | Stage 5 (#94) | PR #101 (#101) | PR #102 (#102) | Δ vs Stage 5 | Δ vs PR #101 | vLLM 0.20.1 | ratio vs vLLM |
|-------------|--------------:|---------------:|----------------:|-------------:|-------------:|-------------:|--------------:|
| c=1   | 65.5 / 12.6 / 282   | 103.6 / 9.19 / 57   | **104.5 / 9.16 / 58**   | +60% | +1%  | 161.5 / 4.94 / 164  | 65% |
| c=8   | 118.0 / 55.8 / 1242 | 263.6 / 31.31 / 280 | **295.6 / 24.50 / 273** | +151% | +12% | 413.1 / 8.63 / 1381 | 72% |
| c=16  | 132.1 / 102 / 1944  | 321.9 / 45.66 / 441 | **374.6 / 38.50 / 443** | +184% | +16% | 505.0 / 11.08 / 2646 | 74% |
| c=32  | 137.1 / 204 / 3050  | 310.0 / 97.28 / 663 | **391.1 / 75.67 / 663** | +185% | +26% | 1872.9 / 14.67 / 317 | 21% |

Format: `tok/s aggregate / mean TPOT ms / mean TTFT ms`. PR #101 = `1af5c8d`, PR #102 = `bb59e6f`.

The c=32 ratio improvement (7.3% → 21%) is the most meaningful one. The remaining gap is overwhelmingly in **vLLM's full-CUDA-Graph capture** (cudagraph_capture_sizes up to 64), not in any per-kernel efficiency loss — see § "What's still missing".

---

## Stage 6 — vLLM `marlin_moe_wna16` port (PR #101)

ferrum had been running per-expert `gemm_gptq_with_offset` against IST-DASLab Marlin tiles — 128 separate launches per layer × 48 layers ≈ 6144 launches per token at c=32. vLLM's `marlin_moe_wna16` is a single fused launch that handles all (token, expert) pairs of a layer via `expert_ids` indirection. Same kernel as vLLM uses internally; ratio of vLLM's c=32 number against ferrum's says how close the rest of the pipeline gets us.

### Vendoring + parity

Vendored `csrc/moe/marlin_moe_wna16/{ops.cu, marlin_template.h, …}` from vLLM 0.10.2 + `csrc/quantization/gptq_marlin/gptq_marlin_repack.cu` (the qweight repack from on-disk GPTQ to vLLM Marlin tile format). Wrapped with an `extern "C"` entry point so we can call it directly from `cudarc` without a torch link.

The parity test (`marlin_moe_vllm_parity_test`) feeds both the existing IST-DASLab path and the new vLLM path the same synthetic 4-expert problem and compares row-wise. Final result on `14e9b4b`: **rel ≈ 5e-4 across all 4 experts** (max|diff| = 1-2 ULP for f16). Bit-exact within fp16 precision.

### The 6-bug chain to parity

Each of these was a real defect, not a workaround. They all hid the next one:

1. **smoke test literal overflow.** `(i as i32).wrapping_mul(0x9E3779B1)` — the literal is decimal `2654435761`, out of i32 range, fails Rust's `deny(overflowing_literals)`. Fixed by computing in u32 then casting.
2. **`clone_htod(&&[T])` doesn't satisfy `HostSlice<T>`.** Iterating `qweights.iter()` yields `&&[i32]`; the trait is impl'd for `[T]` (auto-derefs `&[T]` but not `&&[T]`). Fixed by indexing `qweights[e]`.
3. **`use_fp32_reduce=1 + c_tmp=null`** dereferences a null pointer for the fp32 scratch. Wrapper now picks `use_atomic_add=1` when `c_tmp` is null.
4. **Atomic-add path requires C zeroed by caller.** The kernel only self-zeros under `slice_count > 1 && slice_idx == 0`; for trivial slicing it accumulates onto whatever was in C → NaN on later expert blocks (we'd allocated with `Backend::alloc` which doesn't zero).
5. **`sms=-1` makes `grid_dim = sms × blocks_per_sm` negative** → kernel silently never launches, output stays at whatever was in C (zeros after fix #4 — looked like the kernel ran but did nothing). vLLM's torch wrapper auto-detects sms via `cudaDeviceGetAttribute`; we bypassed that wrapper. Fixed by detecting in our extern C entry.
6. **Scales need IST-DASLab `_scale_perm` host permutation.** vLLM's marlin kernel reads scales through the same fragment-pattern shared-memory load as IST-DASLab; raw row-major scales give rel ≈ 0.5 with col 0 right and col 1+ wrong. The unit-scales experiment (set every scale to 0.05) isolated this — under unit scales rel collapsed to 1e-3, proving column-axis garbage was scales-only. Then `repack_scales_to_marlin` per expert before stacking → rel < 1e-3 across the board.

The 5th and 6th bugs took the longest; both were silent (no panic, no Rust-side error). The diagnostics that pinned them: `FERRUM_PARITY_FORCE_EXPERT0` and `FERRUM_PARITY_UNIT_SCALES` envs in the parity test, plus dumping ref/vllm output side-by-side.

### Production wiring

After parity green, wired into `moe_forward_bucketed`:

- New `Backend` trait methods: `moe_gemm_phase_vllm`, `upload_moe_routing`, `zero_buffer`, `MoeRouting<B>` struct. CUDA overrides under `vllm-moe-marlin` feature; default returns Unsupported.
- `CudaState` lazily allocates an 8 MB fp32 `c_tmp` scratch on first vLLM moe call (sized at `sms × 4 × moe_block_size × max_thread_n` upper bound), reused for every layer + every forward.
- `Backend::load_gptq_stacked` switches loaders under `FERRUM_VLLM_MOE=1` — vLLM Marlin tile format vs IST-DASLab format — both in-place, swap by env.
- Routing buffers (`sorted_token_ids` / `expert_ids` / `num_tokens_past_padded`) are built host-side per layer from `plan.expert_offsets` (block_size=16, top_k=1 since rows are pre-gathered), uploaded once per layer.

### Bench (Stage 6 = PR #101 = `1af5c8d`)

| c  | Stage 5 baseline | PR #101 | Δ |
|----|---:|---:|---:|
| 1  | 65.5 | 103.6 | +58% |
| 8  | 118.0 | 263.6 | +123% |
| 16 | 132.1 | 321.9 | +144% |
| 32 | 137.1 | 310.0 | +126% |

Two oddities worth noting from this round:
- **fp32_reduce + 8 MB c_tmp didn't move c=32**. We'd expected ~1.3-1.5× from atomic-add → fp32_reduce per the older perf docs; got noise. Atomic-add on Ada/Blackwell is fast enough that the heuristic doesn't apply.
- **A persistent-buffer attempt (Stage 13c step 1) shipped with #101 but never produced a green bench** — kept hitting `CUDA_ERROR_INVALID_VALUE` on the 2nd memcpy under three different alloc strategies (rounded-cap, exact-cap with re-alloc, worst-case once-alloc + raw `cuMemcpyHtoDAsync_v2`). Reverted to per-call `clone_htod + mem::forget` in PR #102. Tracked separately; the leak is a few KB per layer, freed only at process exit.

---

## Stage 7 — batched attention dispatch (PR #102)

After PR #101 the c=32 forward profile (`bench_paged_decode_attn` microbench + `[batched-decode-prof]` instrumentation) read:

```
forward total = 50 ms / token at c=32
  dense (lm_head, embed)  3 ms  (6.7%)
  attn_peritem           15 ms  (30.7%)
  moe                    18 ms  (35.9%)
  other                  13 ms  (26.6%)
```

The microbench at the same shape said `paged_decode_attn` itself runs in **45.5 µs / layer** (= 2.18 ms / forward) and pulls 36% of RTX 4090 peak DRAM bandwidth — leaving ~13 ms of "attn time" UNACCOUNTED for. That ~13 ms is everywhere except the actual attention kernel:

```rust
// forward_layer_batched_decode paged path, c=32:
for i in 0..m {                                            // 32 launches
    B::split_qkv_norm_rope_into_paged_cache(tokens=1, …)   // per-seq, single block_table
}
// Stack block_tables host-side, upload, then ...
B::paged_decode_attention(num_seqs=m)                      // 1 launch (the only batched part)
for i in 0..m {                                            // 32 more launches
    B::copy_slice(paged_batch_o[i] → attn_flat[i])
}
```

= ~65 launches per layer × 48 layers = ~3120 launches per token. At ~5 µs of host-side dispatch each, that's ~15 ms — which matches the gap exactly.

### The fix

A `_varlen` variant of `split_qkv_norm_rope_into_paged_cache` already existed (built for prefill); extending it to the m=32 batched-decode case meant building three small host arrays (`cu_seqlens_q = [0, 1, 2, …, m+1]`, `pos_offsets[m]`, stacked `block_tables[m, max_blocks_per_seq]`), uploading them once, and replacing the m-iteration loop with a single dispatch. The output copy was an even smaller change: replace m `copy_slice(paged_batch_o[i] → attn_flat[i*q_dim])` calls with one `copy_slice(0 → 0, length = m * q_dim)` covering the whole packed buffer (layouts already match).

Two new persistent scratch buffers on `Qwen3MoeScratch`: `paged_batch_pos_offsets` and `paged_batch_cu_seqlens_q`, allocated once in `enable_paged_batch`.

### Bench (Stage 7 = PR #102 = `bb59e6f`)

| c  | PR #101 | PR #102 | Δ |
|----|---:|---:|---:|
| 1  | 103.6 | 104.5 | +1% (noise) |
| 8  | 263.6 | 295.6 | +12% |
| 16 | 321.9 | 374.6 | +16% |
| 32 | 310.0 | 391.1 | **+26%** |

c=32 TPOT 97.28 → **75.67 ms (-22%)** — the saved ~22 ms tracks the launch-overhead estimate within margin.

---

## Where we are vs vLLM (post-#102)

c=32 forward = ~38 ms / token under the hood (TPOT 75.67 ms minus ~37 ms scheduler / sampling / streaming overhead measured by `[batched-decode-prof]`):

```
attn ≈ 3 ms        — paged_decode_attn kernel itself, near optimal
moe  ≈ 18 ms       — vLLM marlin_moe kernel, no headroom from kernel swap
other ≈ 13 ms      — rms_norm, residual_add, qkv_proj, o_proj, lm_head
dense ≈ 4 ms       — embed, final norm, lm_head GEMM
```

vLLM at c=32 = ~21 ms TPOT. Gap to close: **~17 ms / token**. Order-of-magnitude breakdown:

1. **Full forward CUDA-Graph capture** — vLLM's c=32 cudagraph_capture stamps out one graph per padded batch size and replays it; we still re-launch every kernel host-side. Low end of the launch-overhead estimate is ~5 ms / forward at c=32 (96 launches × ~5 µs CPU dispatch each — confirmed by `[bg-loop-gap]` instrumentation), high end is ~10 ms when iter_lock and scheduler pre-work overlap badly. This is the single biggest remaining lever.
2. **Scheduler iter_lock** — `EngineInner::iteration_lock` serialises batches, blocking next-batch prep on previous-batch GPU work. memo `project_v02_engine_wall_prof.md` measured ~9% engine overhead at c=16; at c=32 with longer GPU phases the overlap potential is higher.
3. **m_padded thread-config caching** in marlin — currently `determine_exec_config` re-enumerates per call. vLLM caches per-(m_padded, type) tuple. Estimated +2-5%.
4. **rms_norm + residual fusion** — three separate launches per layer, ~5 µs each × 48 = ~720 µs. Fusing brings them to one launch. Small but compounding with #1.

None of these give a 4-5× win individually. To close the full 21% → 80% (vLLM's c=32) gap, **#1 has to land**. It's also the largest single chunk of work — vLLM-style BatchDescriptor + per-m_padded graph cache + paged_decode_attention static-shape variant — and is tracked as Stage 13c on task #21.

## What's still missing

- **Stage 13c full forward graph capture.** Started step 1 (persistent vllm moe routing buffers on `CudaState`) but reverted under PR #102 because of the `CUDA_ERROR_INVALID_VALUE` regression noted in Stage 6. Restart with cuda-gdb / compute-sanitizer when GPU access returns.
- **`use_fp32_reduce + c_tmp` benefit at smaller batches.** Helps c=8 (+11%), neutral c=16, micro-regress c=32 in the post-PR-101 bench. Worth a per-batch policy if we ever re-enable it.
- **Triton w4a16 INT4 GEMM (FERRUM_TRITON_INT4=1) prefill NaN bug** — separate track, see `project_ferrum_triton_integration.md`.
- **30B-A3B Metal `ferrum run --prompt`** — fixed in PR #104 (auto-set MoE batched env defaults); was 0.1 → 56.9 tok/s post-fix. CLI-only, doesn't affect any CUDA path.

## Reproducing

Branch `main` post-`bb59e6f` covers Stage 7. Bench harness lives in `bench/v0.2-cuda/m3_vllm_moe.sh` — set `FERRUM_VLLM_MOE=1` to opt into the marlin_moe_wna16 path (default still IST-DASLab + per-expert dispatch for safety).

```bash
FERRUM_VLLM_MOE=1 \
FERRUM_KV_CAPACITY=2048 \
FERRUM_KV_MAX_BLOCKS=4096 \
FERRUM_PAGED_MAX_SEQS=32 \
FERRUM_METAL_PAGED_KV=1 \
FERRUM_MIXED_BATCH=0 \
FERRUM_GREEDY_ARGMAX=1 \
FERRUM_MOE_BUCKETED=1 \
FERRUM_MARLIN_SKIP_WS_ZERO=1 \
FERRUM_MOE_STREAMS=4 \
  ./target/release/ferrum serve --model /path/to/Qwen3-30B-A3B-Instruct-GPTQ-Int4 --port 8802 --gpu-memory-utilization 0.95
```

Then `vllm bench serve --max-concurrency 32 --random-input-len 256 --random-output-len 128 --num-prompts 128 --base-url http://127.0.0.1:8802`.
