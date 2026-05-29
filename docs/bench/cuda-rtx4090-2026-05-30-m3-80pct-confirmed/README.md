# M3 80% Throughput Confirmation - RTX 4090 - 2026-05-30

This directory records the same-pod `n_repeats=5` confirmation sweep for the
opt-in Ferrum FA2 direct FFI path:

```text
FERRUM_FA_LAYOUT_VARLEN=1
FERRUM_FA2_DIRECT_FFI_SHIM=/workspace/libferrum_fa2_shim.so
```

As of the follow-up checkpoint, `FERRUM_FA2_DIRECT_FFI_SHIM` alone enables the
direct path; `FERRUM_FA2_DIRECT_FFI=0` forces it off.

Workload: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`, random `256/128`, 128 prompts,
10 warmup requests, closed-loop `c=1/4/16/32`, RTX 4090 locked clocks.
vLLM baseline used vLLM `0.20.2`, `--gpu-memory-utilization 0.85`,
`--max-num-batched-tokens 2048`, `--max-num-seqs 32`,
`--no-enable-prefix-caching`, and `--enable-chunked-prefill`.

Both sweeps completed with zero request errors. Ferrum also passed the Paris
smoke gate before the benchmark. Post-run process audits were empty and
`nvidia-smi` reported `0, 1 MiB, 0 %` for both runs.

| c | Ferrum FA2 direct tok/s | vLLM 0.20.2 tok/s | ratio | 0.80x threshold | margin |
|---:|---:|---:|---:|---:|---:|
| 1 | 160.4 +/- 0.2 | 183.9 +/- 0.2 | 0.872x | 147.1 | +13.3 |
| 4 | 446.3 +/- 7.0 | 512.5 +/- 2.8 | 0.871x | 410.0 | +36.3 |
| 16 | 1185.1 +/- 12.3 | 1331.9 +/- 5.7 | 0.890x | 1065.5 | +119.6 |
| 32 | 1641.9 +/- 4.8 | 1972.9 +/- 18.6 | 0.832x | 1578.3 | +63.5 |

Artifacts:

- `ferrum-fa2-direct/`: Ferrum FA2-direct JSON reports, bench logs, metadata,
  Paris smoke response, and post-run hygiene checks.
- `vllm0202/`: vLLM JSON reports, bench logs, metadata, and post-run hygiene
  checks.

Interpretation:

- The opt-in FA2 direct FFI path clears the 0.80x throughput target for all four
  concurrency cells under the same-pod `n_repeats=5` confirmation sweep.
- This is not yet a default-runtime result. The path depends on the vLLM/Torch
  FA2 extension through a runtime shim and allocates the extra FA-compatible K/V
  pool. Defaulting still needs a dependency and memory decision or a cleaner
  source-built FA2 wrapper.
