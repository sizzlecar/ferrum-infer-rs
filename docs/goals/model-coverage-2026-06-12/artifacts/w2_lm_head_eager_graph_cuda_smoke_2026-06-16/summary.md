# W2 Gemma3 lm-head-eager unified graph diagnostic

- Artifact:
  `docs/goals/model-coverage-2026-06-12/artifacts/w2_lm_head_eager_graph_cuda_smoke_2026-06-16/`.
- Source checkpoint:
  `dded3b7d test(cuda): add lm-head-eager graph scope`.
- Lane:
  W2 Gemma3 CUDA unified graph lm-head-eager minimal diagnostic.
- Hardware:
  Vast instance `41187356`, 1x RTX 4090, driver `580.95.05`, CUDA `12.4`;
  instance cleanup was confirmed with `cur_state=stopped` and
  `actual_status=exited` in `local_vast/instance_stop_check.json`.
- Build:
  dense CUDA diagnostic build with features `cuda,vllm-paged-attn-v2`;
  `remote/build_dense.rc` is `0`; binary SHA256 is saved in
  `remote/meta/ferrum.sha256`.

## Correctness Smoke

- `ferrum run` rc `0`; output JSONL content was `5`, finish reason `stop`,
  `n_tokens=3`.
- `ferrum serve` chat response content was `5`, with usage present
  (`prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`).
- Repeated same-shape serve requests returned `5` and logged
  `scope=lm_head_eager` graph capture/replay entries.
- No illegal address was observed in this diagnostic path.

## Diagnostic Performance

Tiny c16 A/B, diagnostic only (`n_repeats=1`, random 16 input / 8 output,
16 measured requests, 4 warmup):

| Mode | Completed | Errors | Throughput tok/s | TTFT p50 ms | TPOT p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | 16 | 0 | 246.621 | 268.764 | 35.249 |
| lm-head-eager | 16 | 0 | 233.070 | 286.446 | 37.144 |

An earlier c16 diagnostic with `kv-capacity=2048` OOMed after the KV pool
reserved too much memory, leaving only about 94 MiB free before a 128 MiB
allocation. That OOM is a configuration-sizing diagnostic, not a graph
correctness failure.

## Interpretation

`lm-head-eager` narrows the full unified graph illegal-address suspect to the
part excluded from the scope, primarily `lm_head` / dense Marlin graph capture
or captured workspace aliasing. However, it does not improve endpoint
throughput in this diagnostic and should not be treated as the W2 performance
lever.

The next W2 performance work should stay focused on the Gemma3 GPTQ dense tail
MLP / Marlin path and vLLM comparison, not broader graph knob sweeps.

No `MODEL_RELEASE_GRADE_W2 PASS: <out_dir>` was produced.
