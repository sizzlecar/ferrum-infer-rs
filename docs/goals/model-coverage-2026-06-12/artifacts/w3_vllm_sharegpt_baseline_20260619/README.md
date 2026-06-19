# W3 Qwen3.5 ShareGPT vLLM Baseline

Date: 2026-06-19
Ferrum git SHA: `20b8946b781c56601d84e093811859c99fd648ab`
Model: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
Dataset: ASCII ShareGPT 100 prompts
Dataset SHA256: `58d5721d8389d7ed9ec4b8b2dbd8797faa61641c6ba023dd150a1a9d93c0a01e`
Hardware lane: 1x Vast CUDA host reporting RTX 4090, 49140 MiB visible VRAM
vLLM: `0.23.0`
Torch: `2.11.0+cu130`

This artifact records the same-host vLLM baseline used as the next Ferrum 80%
optimization target. It is not a W3 final PASS artifact because Ferrum has not
yet produced the matching full c=1/4/16/32 ShareGPT performance sweep or final
`MODEL_RELEASE_GRADE_W3 PASS` validator output.

## Baseline

| concurrency | vLLM mean output tok/s | vLLM LCB tok/s | Ferrum 80% target tok/s | mean p95 ITL ms | completed | errored | bad output |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | 136.1429 | 134.3690 | 107.4952 | 6.7922 | 100/100/100 | 0/0/0 | 0/0/0 |
| 4 | 405.4203 | 405.0572 | 324.0457 | 9.1199 | 100/100/100 | 0/0/0 | 0/0/0 |
| 16 | 1190.6921 | 1120.2993 | 896.2394 | 11.8902 | 100/100/100 | 0/0/0 | 0/0/0 |
| 32 | 1708.5279 | 1687.3965 | 1349.9172 | 15.7098 | 100/100/100 | 0/0/0 | 0/0/0 |

## Diagnostic Ferrum Gap

The attempted Ferrum ShareGPT sweep was stopped after the first c=1 repeat
because it was already far below the 80% target:

- c=1 repeat 1: `100 completed / 0 errored / 340.3s`
- Diagnostic throughput from that partial repeat: about `37.6` output tok/s
  for `100 * 128` generated tokens.
- This is about `28%` of the vLLM c=1 LCB and about `35%` of the c=1 80%
  target.

This single Ferrum repeat is only a bottleneck signal. Any W3 performance claim
still requires a fresh full Ferrum sweep with `--fail-on-error --require-ci
--seed 9271 --n-repeats 3`, same dataset, same hardware class, and the final W3
validator.

## Files

- `bench_vllm_sharegpt_sweep_100x3.json`: raw vLLM bench-serve report.
- `vllm_baseline_summary.json`: compact baseline and 80% target summary.
- `bench-vllm.command.txt`: exact bench command.
- `vllm_versions.json`: vLLM/Torch/CUDA availability snapshot.
- `dataset.sha256`: dataset SHA used by the run.
