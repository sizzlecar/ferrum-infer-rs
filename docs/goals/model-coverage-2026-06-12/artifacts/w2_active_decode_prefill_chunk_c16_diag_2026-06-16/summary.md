# W2 Active-Decode Prefill Chunk c16 Diagnostic

Status: diagnostic complete; no `MODEL_RELEASE_GRADE_W2 PASS` was produced.

## Scope

- Source: `8bc7cf087ae5fe6e7e2e34405ca5781cc8d0acdc`.
- Build: `cargo build --release -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`.
- Hardware: 1x RTX 4090, driver 565.77, `nvidia-smi` CUDA 12.7, `nvcc` 12.4.131.
- Runtime: 2026-06-16T05:43:39Z to 2026-06-16T06:17:03Z UTC, 33.4 minutes, estimated GPU cost $0.237.
- Shape: c=16, ShareGPT dataset, 64 prompts, 1 repeat, `--fail-on-error`, seed 9271.
- Baseline reference: vLLM c16 same-dataset LCB 491.150 tok/s; 80% threshold 392.920 tok/s.

## Correctness

- `ferrum run` rc 0, validation PASS, content `5`.
- default `ferrum serve` smoke PASS, content `5`.
- chunk32 `ferrum serve` smoke PASS, content `5`.
- default/chunk32 `bench-serve` rc 0, completed `[64]`, errored `[0]`, usage token counting.

## Performance

| arm | output tok/s | vs vLLM LCB | gap to 80% | TTFT p95 ms | TPOT p95 ms | ITL p95 ms | E2E p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 320.311 | 65.22% | 72.609 | 852.787 | 49.085 | 86.128 | 4193.402 |
| chunk32 | 312.911 | 63.71% | 80.009 | 697.100 | 46.461 | 67.816 | 4305.749 |

chunk32 vs default: throughput -2.31%, ITL p95 -21.26%, TTFT p95 -18.26%, TPOT p95 -5.35%, E2E p95 2.68%.

## Profile

Remote `rg` was unavailable, so `profile_extract.log` was regenerated locally from `server.log`.

| arm | category | n | p50 us | p95 us | max us |
|---|---|---:|---:|---:|---:|
| default | decode_only | 73 | 33908.0 | 40314.0 | 42826.0 |
| default | mixed | 0 |  |  |  |
| default | prefill_only | 2 | 324712.0 | 497352.7 | 516535.0 |
| chunk32 | decode_only | 67 | 33090.0 | 40967.3 | 45903.0 |
| chunk32 | mixed | 6 | 73604.0 | 84571.5 | 85609.0 |
| chunk32 | prefill_only | 2 | 229262.5 | 313134.8 | 322454.0 |

## Interpretation

chunk32 is not a W2 throughput lever. It reduced ITL p95 from 86.128 ms to 67.816 ms, but output throughput fell from 320.311 tok/s to 312.911 tok/s.

The profile does not support continuing to sweep active-decode prefill chunk as the main fix: default had zero mixed prefill+decode rows in this run, while chunk32 added bounded mixed rows and still remained below the 80% throughput threshold. Decode-only steps remain around 34-41 ms, so the next useful work should return to model-side decode/batched execution cost, with minimal native/product probes before any broad sweep.

Vast cleanup: stop response success `True`, final `actual_status=exited`, `cur_state=stopped`, `intended_status=stopped`.
