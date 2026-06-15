# W2 Prefix-Cache ShareGPT Diagnostic

Date: 2026-06-15

Scope: diagnostic-only native CUDA run on Vast instance `40826362`
(1x RTX 4090). This is not release-grade evidence: `n_repeats=1`, no
`--require-ci`, no `model_release_grade_manifest.json`, and no
`MODEL_RELEASE_GRADE_W2 PASS: <out_dir>`.

## Contract

- lane: W2 Gemma3 CUDA typed prefix-cache ShareGPT diagnostic
- expected runtime/cost: 10-25min, hard cap 35min, about USD 0.425/hr while running
- stop condition: start/SSH/CUDA/server readiness first failure, chat smoke failure,
  c16/c32 ShareGPT diagnostic complete and artifacts copied, or 35min hard cap
- correctness gate: `ferrum serve --enable-prefix-cache` readiness plus non-stream
  chat smoke before `bench-serve`
- performance command: diagnostic-only natural ASCII ShareGPT c16/c32 with
  `bench-serve --fail-on-error --seed 9271 --n-repeats 1 --num-prompts 16`

## Correctness

- `run.status`: `PASS`
- `bench-serve.rc`: `0`
- chat smoke content: `5`
- chat smoke usage: `prompt_tokens=23`, `completion_tokens=3`, `total_tokens=26`
- c16: `16 completed / 0 errored`, bad output `[0]`
- c32: `16 completed / 0 errored`, bad output `[0]`

No new Ferrum product correctness issue was found in this diagnostic.

## Prefix Cache

`--enable-prefix-cache` was accepted through the typed product CLI path:

- decision trace: `prefix_cache_policy=prefix_cache_enabled`, source `cli`
- health before: `enabled=true`, `hits=0`, `misses=0`, `saved_prefill_tokens=0`
- health after: `enabled=true`, `hits=0`, `misses=53`,
  `saved_prefill_tokens=0`, `entries=0`

Interpretation: prefix cache was enabled but did not hit this repeated-prompt
ShareGPT scenario.

## Performance

Ferrum prefix-cache diagnostic:

| cell | completed | errored | output throughput | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|---:|---:|
| c16 | 16 | 0 | 340.618 tok/s | 889.469 ms | 1453.788 ms |
| c32 | 16 | 0 | 342.350 tok/s | 887.527 ms | 1438.820 ms |

Compared with the clean vLLM ShareGPT diagnostic baseline from
`w2_vllm_sharegpt_baseline_probe_2026-06-15`:

| cell | Ferrum prefix-cache | vLLM baseline | ratio | gap to vLLM | points below 80% |
|---|---:|---:|---:|---:|---:|
| c16 | 340.618 tok/s | 518.796 tok/s | 0.657 | 34.3% | 14.3 |
| c32 | 342.350 tok/s | 524.128 tok/s | 0.653 | 34.7% | 14.7 |

Compared with the immediately prior Ferrum no-prefix ShareGPT diagnostic:

- c16: `340.003 -> 340.618 tok/s`, `+0.18%`
- c32: `342.284 -> 342.350 tok/s`, `+0.02%`

## Next Step

Do not repeat full sweeps for this lever. Either inspect why the typed
prefix-cache path records zero hits/entries for identical prompts, or return to
the higher-impact Gemma tail/GEMM hotspots (`tail_gate_up`, `tail_down`) already
identified by the decode profile.
