# Session bench results — 2026-05-25 vllm-moe-marlin unblock

> ⚠️ **PRELIMINARY** — `n_repeats=1`, `num_prompts=30`, no nsys / chrome-trace
> captured for the ON path, vLLM ratio column uses a different dataset than
> ferrum's runs (see `../GOAL.md` Update caveats). The OFF→ON Δ% is
> internally consistent; the ratio_OFF / ratio_ON columns are indicative
> only. Don't quote these numbers externally before the next-session
> re-baseline (re-baseline checklist at the bottom of `../GOAL.md`).

GPU: Vast contract 37796550, RTX 4090 (sm_89), CUDA 13.0.48, driver 580.82.09,
locked 2520 MHz / 350 W. ferrum commit `47e8dec`. Dataset: random in=256
out=128, `num_prompts=30`, warmup=5, **`n_repeats=1` (single-shot, no CI95)**.
Same iter-3 env knob set as GOAL.md cell-2026-05-25.

## Numbers

| c  | OFF tput tok/s | ON tput tok/s | Δ%   | TTFT p50 OFF→ON (ms) | TPOT p50 OFF→ON (ms) |
|----|---------------:|--------------:|-----:|---:|---:|
| 1  | 128.0          | **146.5**     | +14.4% | 53.8 → 41.4 | 7.40 → 6.44 |
| 4  | 130.1          | 138.8         | +6.7% | 140.8 → 136.2 | 29.91 → 25.69 |
| 16 | 673.6          | **818.3**     | +21.5% | 309.5 → 288.5 | 19.86 → 12.76 |
| 32 | 871.1          | **1079.4**    | +23.9% | 693.1 → 750.4 | 28.01 → 19.57 |

OFF = `FERRUM_VLLM_MOE` unset.
ON  = `FERRUM_VLLM_MOE=1` (exercises the newly-unblocked
`marlin_moe_wna16::Marlin<...>` path).

c=4 is still anomalously slow as predicted by GOAL.md — n_prompts=30 is
prefill-dominated at small c. Re-run with n_prompts≥128 before quoting
the c=4 ratio.

## vs vLLM (historical baseline)

vLLM 2026-05-13 ShareGPT row from GOAL.md (not re-measured this session
— blocked on a driver≥575 pod, this pod has 580 but vLLM 0.20.2 install
adds 15 min disk and was deferred):

| c | ratio_OFF | ratio_ON | gap to 0.80 |
|--:|---:|---:|---:|
| 1  | 0.62 | **0.71** | 9 pp |
| 4  | 0.26 | 0.27 | (re-measure needed) |
| 16 | 0.54 | **0.65** | 15 pp |
| 32 | 0.46 | **0.57** | 23 pp |

Headline: vllm-moe-marlin delivers +9 to +15 pp ratio gain (lower end of
GOAL.md's 15–30 pp prediction). The full 0.80 target is NOT met by this
lever alone — Phase C (lm_head cutlass) + Phase D (Marlin tile small-m
heuristic) + Phase E (chunked-prefill 8192 tokens) still needed.

## What shipped in this session

- `kernel_instantiations.cu` — vllm-moe-marlin explicit template
  instantiation TU (CUDA 13 hidden-visibility workaround idiom)
- `marlin_template.h` + `build.rs` — `visibility("default")` attribute
  + `-Xcompiler -fvisibility=default` on vllm-marlin
- `marlin.cu` — disable `get_marlin_kernel` dispatch as CUDA-13
  workaround (`#include "kernel_selector.h"` commented out) so the
  build links. M2 (Llama-INT4) loses the vllm-marlin GEMM kernel and
  falls back to IST-DASLab marlin. Track follow-up to re-enable safely.

## Pod cost

- $0.577/h × ~2h10min ≈ $1.25
- Of which: ~7 min × 3 rebuilds = ~21 min nvcc time at full SM89 sweep
  (vllm-marlin sm80/sm89 + vllm-moe-marlin ops.cu + kernel_instantiations.cu)

## Easy-win check: c=32 + FERRUM_GRAPH=0

Per GOAL.md lever #5 prediction of "+3pp at c=32". With vllm-moe-marlin
on:

| config | c=32 tput tok/s (n=1) |
|---|---:|
| GRAPH=1 (current default, ON path) | 1079.4 |
| GRAPH=0 (this test) | 1063.2 |

⚠️ **Inconclusive at n=1.** −1.5% is inside single-shot noise (we've
seen day-to-day c=32 numbers wobble ~3-5% even with locked clocks). Do
**not** read this as "graph-off doesn't help once vllm-moe-marlin is
on" — that needs ≥3 repeats per cell. The most we can say from this
data is "graph-off is not an obvious win at c=32; merits a proper A/B
in the next session" — not a refutation of GOAL.md lever #5.
