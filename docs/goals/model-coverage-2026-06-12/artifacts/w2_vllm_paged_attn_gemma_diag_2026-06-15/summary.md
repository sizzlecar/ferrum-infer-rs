# W2 vLLM Paged-Attn Gemma Diagnostic

Date: 2026-06-15

Scope: diagnostic-only native CUDA run on Vast instance `40826362`
(1x RTX 4090). This is not release-grade evidence: `n_repeats=1`, no
`--require-ci`, no `model_release_grade_manifest.json`, and no
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## Contract

- lane: W2 Gemma3 CUDA typed vLLM paged-attn ShareGPT diagnostic
- expected runtime/cost: 10-25min, hard cap 35min, about USD 0.425/hr while
  running
- stop condition: start/SSH/CUDA/server readiness first failure, typed
  attention-selection assertion failure, chat smoke failure, c16/c32 ShareGPT
  diagnostic complete and artifacts copied, or 35min hard cap
- correctness gate: `ferrum serve` from artifact-local `ferrum.toml` with
  `runtime.use_vllm_paged_attn=true`, readiness, decision-trace assertion, and
  non-stream chat smoke before `bench-serve`
- performance command: diagnostic-only natural ASCII ShareGPT c16/c32 with
  `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`

## Correctness

- `run.status`: `PASS`
- `bench-serve.rc`: `0`
- attention selection: `PASS prefill=vllm_paged_varlen
  decode=vllm_paged_attn_v1_short`
- effective config source: `attention_prefill_mixed_backend` selected
  `vllm_paged_varlen` from `config_file` key `FERRUM_USE_VLLM_PAGED_ATTN`
- chat smoke content: `5`
- chat smoke usage: `prompt_tokens=23`, `completion_tokens=3`,
  `total_tokens=26`
- c16: `16 completed / 0 errored`, bad output `[0]`
- c32 diagnostic cell: `16 completed / 0 errored`, bad output `[0]`

No new Ferrum product correctness issue was found in this diagnostic.

## Performance

Ferrum typed vLLM paged-attn diagnostic:

| cell | completed | errored | output throughput | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|---:|---:|
| c16 | 16 | 0 | 340.443 tok/s | 890.332 ms | 1453.858 ms |
| c32 diagnostic | 16 | 0 | 341.419 tok/s | 889.279 ms | 1440.689 ms |

Compared with the clean vLLM ShareGPT diagnostic baseline from
`w2_vllm_sharegpt_baseline_probe_2026-06-15`:

| cell | Ferrum VPA | vLLM baseline | ratio | gap to vLLM | points below 80% |
|---|---:|---:|---:|---:|---:|
| c16 | 340.443 tok/s | 518.796 tok/s | 0.656 | 34.4% | 14.4 |
| c32 diagnostic | 341.419 tok/s | 524.128 tok/s | 0.651 | 34.9% | 14.9 |

Compared with the immediately prior Ferrum no-VPA ShareGPT diagnostic:

- c16: `340.003 -> 340.443 tok/s`, `+0.13%`
- c32 diagnostic: `342.284 -> 341.419 tok/s`, `-0.25%`

Compared with the typed prefix-cache diagnostic:

- c16: `340.618 -> 340.443 tok/s`, `-0.05%`
- c32 diagnostic: `342.350 -> 341.419 tok/s`, `-0.27%`

## Interpretation

The typed product config path successfully selects the vLLM paged-attention
bridge for Gemma3, but it does not move W2 ShareGPT throughput in this
diagnostic. This rules out vLLM paged attention as the immediate missing
14-15 percentage points to the 80% line for this cell.

Continue with the already-identified higher-impact Gemma tail/GEMM hotspots,
especially `tail_gate_up` and `tail_down`. Do not repeat full sweeps for this
lever unless a source change makes the VPA path materially different.

## Shutdown

Artifact copied back locally before stopping the instance. Vast shutdown poll 2
recorded `cur_state=stopped`, `actual_status=exited`.
