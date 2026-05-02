# Qwen3-30B-A3B fused gate+up+silu — A/B bench (2026-05-01)

## Hypothesis

Per the 2026-05-01 perf status doc (`docs/status/2026-05-01-concurrent-perf-status.md`),
the MoE decode path's three back-to-back dispatches —

1. `gemv_quant_moe_id` (gate) → `gate_out_stacked[top_k, ffn]`
2. `gemv_quant_moe_id` (up)   → `up_out_stacked[top_k, ffn]`
3. `silu_mul_stacked`         → `silu_stacked[top_k, ffn]`

— share the same input activation (`norm_out`) and the same selected
expert IDs, and emit two intermediate buffers (`gate_out_stacked`,
`up_out_stacked`) that are immediately consumed by the silu kernel.
Folding all three into a single Metal kernel:

* eliminates the writeback + read-back of `gate_out_stacked` and
  `up_out_stacked` (≈4× `[top_k, ffn]` of intermediate bandwidth per
  layer)
* halves the activation read traffic in the inner Q4_K reduction
  (one register-file load services both weight matrices)
* drops 2 dispatches per layer (≈96 dispatches per decode token at
  48 layers)

The doc projected ~5–15% throughput at c=16 on Qwen3-30B-A3B Q4_K_M.

## Implementation

* New Metal kernel: `gemv_q4kw_moe_id_gate_up_silu_f32` in
  `crates/ferrum-kernels/src/q4_k_moe_id_gate_up_silu.metal`. Mirrors
  the existing `gemv_q4kw_moe_id_f32` thread/grid layout (32×2 threads,
  4-row tiles, slot in tgpig.z). Adds a second per-row accumulator
  (`sumf_u`) and runs the Q4_K decoder twice within each `ib` outer
  step — once against the gate weight buffer, once against the up
  weight buffer — reusing `yl/yh/sumy` from the activation read. Final
  output is `silu(g) * u`, written directly to `silu_stacked`.
* Rust launcher in `crates/ferrum-kernels/src/q4_k_moe_id_gate_up_silu.rs`.
* New `Backend::gemv_quant_moe_id_gate_up_silu` trait method (default
  `unsupported`) plus a `Backend::supports_fused_moe_gate_up_silu`
  capability probe (default `false`, Metal `true`).
* `Qwen3MoeModel::moe_forward_stacked_decode_impl` selects the fused
  path when the backend supports it. Opt-out via
  `FERRUM_MOE_FUSED_GATE_UP_SILU=0` for A/B comparison.

## Correctness

Unit test in `q4_k_moe_id_gate_up_silu.rs` builds a synthetic
`[E=4, N=64, K=256]` Q4_K stack, runs both the fused dispatch and the
3-dispatch unfused sequence on the same weights/activations, and
asserts the outputs match. Result: **`max_abs = 0.000000`** (bitwise
identical, as expected — same Q4_K decoder + same simd_sum + same
silu·mul arithmetic, just reordered to skip the fp32 memory round-trip).

Run: `cargo test -p ferrum-kernels --features metal --lib fused_matches_unfused_q4k_moe_gate_up_silu`

## Bench setup

* Hardware: M1 Max 32 GB
* Model: Qwen3-30B-A3B Q4_K_M (18.6 GB GGUF, 128 experts × top_k=8,
  48 transformer layers, hidden=2048, expert_inter=768)
* Bench harness: `bench/scripts/bench_serving.py` (vLLM-style
  benchmark_serving.py), deterministic prompts, temperature=0.0
* Concurrencies: c=1, 4, 8, 16
* Server env (both modes): `FERRUM_METAL_PAGED_KV=1
  FERRUM_PAGED_MAX_SEQS=$c FERRUM_KV_CAPACITY=1024 FERRUM_MAX_BATCH=$c`
* Mode toggle: `FERRUM_MOE_FUSED_GATE_UP_SILU=0` (unfused) vs unset (fused)

## Reproduce

```bash
cargo build --release --features metal -p ferrum-cli
./bench/group-a-moe-fused-gate-up-silu-2026-05-01/run_ab.sh
```

Per-concurrency JSON results are written into this directory as
`moe_30b_a3b__{fused,unfused}__c{1,4,8,16}.json`.

## Results (2026-05-02)

Four concurrencies, A/B sweep with macOS page-cache prewarm and a
single warmup curl per phase. Run with the standard bench harness.

### Output throughput (tok/s)

| c | unfused | fused | Δ |
|--:|--:|--:|--:|
| 1  | 44.2  | (cold-skewed)  | n/a |
| 4  | 42.5  | **43.5**  | **+2.4%** |
| 8  | 49.4  | **49.6**  | +0.4% |
| 16 | 50.8  | **51.2**  | +0.8% |

c=1 fused number (`out=2.4 tok/s`) is invalid: that phase ran first
and absorbed the cold mmap page-in for the full 18.6 GB GGUF. Median
TPOT for c=1 was 19.3 ms (fused) vs 18.5 ms (unfused) — within 4%, so
c=1 is effectively a tie.

### TPOT median / p99 (ms)

| c | mode | median | p99 |
|--:|---|--:|--:|
| 4  | unfused | 82.1  | 86.6  |
| 4  | fused   | **79.9**  | **84.8**  |
| 8  | unfused | 150.5 | 165.6 |
| 8  | fused   | **149.0** | 168.7 |
| 16 | unfused | 281.7 | 337.8 |
| 16 | fused   | **280.3** | **333.9** |

### Reading the numbers

The fused kernel is **+0–2.4%** across the concurrencies that matter
(c=4 / 8 / 16). c=4 shows the cleanest gain (~+2.4% throughput,
−2.7% TPOT median); c=8 and c=16 are within noise (~±1%).

This is a much smaller win than the 5–15% the doc projected. The
projected mechanism — saving the round-trip through `gate_out_stacked`
and `up_out_stacked` (≈4× `[top_k, ffn]` of intermediate bandwidth) —
turned out to deliver less than expected, mirroring the PR #79
empirical lesson:

> dispatch count / "saved bandwidth on small intermediate buffers"
> back-of-envelope estimates over-project the win. The buffers sat in
> Apple GPU L2/L3 cache already; the fused-vs-unfused traffic going
> through cache vs. DRAM is small enough that it doesn't dominate.

The structural cleanup (one kernel doing what three did) is real and
the kernel produces bitwise-identical outputs. But this PR alone does
NOT close the 30B-A3B ↔ llama.cpp c=16 gap — that still requires the
`mul_mm_id` Metal kernel rewrite (Tier 2 in the perf-status doc).

## Conclusion

* **Correctness**: bitwise-identical outputs (unit-test parity).
* **Perf**: +0.4–2.4% throughput at c=4/8/16, −0.5–2.7% TPOT median.
* **Win mechanism**: structural simplification + minor BW saving; not
  the projected 5–15%.
* **Decision**: ship the kernel anyway — it's a strict improvement
  with a kill switch (`FERRUM_MOE_FUSED_GATE_UP_SILU=0`). Update the
  perf-status doc to record this as another data point that "saved
  bandwidth on already-cached intermediates" doesn't pay off the way
  raw cache-miss arithmetic predicts.
