# M3 Quick Regression - 1e3ce42 c32 - 2026-06-01

Purpose: fast GPU regression after the Metal/Qwen3-MoE prefill readback sync fix in commit `1e3ce42`.

Artifact mirror:

- Remote: `/workspace/m3-quick-regress-1e3ce42-c32-20260601`
- Local mirror: `docs/bench/dev-loop-product-api-goal-progress-20260601/m3-quick-regress-1e3ce42-c32-20260601/`

Run shape:

- Model: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`
- Hardware: Vast RTX 4090
- Candidate: `FERRUM_FA_LAYOUT_VARLEN=1,FERRUM_FA2_SOURCE=1`
- Baseline/control: `FERRUM_FA_LAYOUT_VARLEN=1,FERRUM_FA2_DIRECT_FFI=0`
- Concurrency: `32`
- Repeats: `1`
- Prompts: `64`
- Warmup requests: `2`

Correctness gates:

| Case | Paris | multi-turn | three-round recall |
|---|---|---|---|
| `fa2_source` | `The capital of France is **Paris**.` | `Paris` | `Paris/basalt` |
| `fa_layout` | `The capital of France is **Paris**.` | `Paris` | `Paris/basalt` |

Performance quick check:

| Case | Output throughput tok/s | TTFT p50 ms | TPOT p50 ms | ITL p95 ms | completed/errors |
|---|---:|---:|---:|---:|---:|
| `fa2_source` | `1403.98` | `242.87` | `19.43` | `49.34` | `64/0` |
| `fa_layout` | `1230.54` | `309.43` | `21.70` | `65.81` | `64/0` |

Regression gate:

- `fa2_source` throughput delta versus `fa_layout`: `+14.10%`.
- Runner performance gate: `ok=true`.
- This is a quick regression packet, not a replacement for the existing N=3 release-performance packet.

Conclusion:

- Commit `1e3ce42` did not introduce an obvious GPU correctness or c32 performance regression.
- The release-performance source of truth remains `/workspace/m3-fa2-source-current-allcells-n3-20260601`.
