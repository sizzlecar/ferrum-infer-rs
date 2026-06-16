# W2 Gemma3 Marlin cache-policy native CUDA probe

- Artifact: `docs/goals/model-coverage-2026-06-12/artifacts/w2_marlin_cache_policy_native_probe_2026-06-16`
- Lane: W2 Gemma3 Marlin cache-policy native probe
- Remote git HEAD: `018ea7bce6494db5539ce32e22f104144fe87eba`
- Probe rc: `0`
- Baseline binary SHA256: `50e4ad67f5d79293da1d524eedcae2cde7edb71d7e6d85387e94b5b37cb0ca41`
- Evict-first binary SHA256: `69655f683cc80daf98737e290946ca69bbcec87d69c818deff5cf2038e8c8e41`
- PASS line: `VERDICT: gemma3 Marlin cache-policy native CUDA probe complete`
- Vast cleanup: instance `40826362` confirmed `stopped/exited`

## Key Rows

Product-style workspace-zero chain timing:

| m | baseline chain_event_us | evict_first chain_event_us | delta | baseline down_us | evict_first down_us |
|---:|---:|---:|---:|---:|---:|
| 1 | 209.590 | 202.707 | -3.3% | 70.825 | 66.969 |
| 10 | 214.771 | 208.636 | -2.9% | 71.008 | 68.232 |
| 16 | 215.580 | 211.690 | -1.8% | 70.549 | 68.800 |
| 23 | 224.924 | 221.983 | -1.3% | 75.153 | 73.939 |
| 32 | 227.722 | 225.173 | -1.1% | 75.659 | 75.339 |

Kernel-only workspace-prezero diagnostic timing:

| m | baseline chain_event_us | evict_first chain_event_us | delta | baseline down_us | evict_first down_us |
|---:|---:|---:|---:|---:|---:|
| 1 | 207.087 | 199.872 | -3.5% | 68.644 | 64.778 |
| 10 | 210.514 | 205.768 | -2.3% | 68.342 | 66.049 |
| 16 | 212.959 | 208.732 | -2.0% | 68.477 | 66.686 |
| 23 | 222.531 | 218.693 | -1.7% | 72.863 | 71.885 |
| 32 | 225.519 | 222.158 | -1.5% | 73.561 | 73.435 |

## Interpretation

The compile-time `FERRUM_MARLIN_CP_ASYNC_EVICT_FIRST=1` variant is accepted by
CUDA 12.4 on Ada (`sm_89`) and improves the product-shaped tail-MLP chain
slightly across all tested batch rows.

This is a real kernel-level lever, unlike the explicit down prefetch branch
that made the down kernel faster while worsening segment wall time. However,
the observed m16/m32 segment gain is only about 1-2%, so this alone is not
enough to close the W2 release-grade gap to the same-hardware vLLM baseline.

Next useful step: either productize the cache policy as a typed/default CUDA
build choice and run `ferrum run` / `ferrum serve` correctness before endpoint
performance, or keep looking for a higher-return dense MLP gate_up/work-reduction
lever. This native probe is diagnostic evidence, not W2 release evidence. The
final W2 validator has not produced `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.
