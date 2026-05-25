  → docs/bench/m3-80pct-goal-2026-05-25/bottleneck-c1.md
  → docs/bench/m3-80pct-goal-2026-05-25/bottleneck-c4.md
  → docs/bench/m3-80pct-goal-2026-05-25/bottleneck-c16.md
  → docs/bench/m3-80pct-goal-2026-05-25/bottleneck-c32.md
# Sweep summary — Qwen3-30B-A3B-GPTQ-Int4

_data: `docs/bench/sweep-2026-05-25-1631-qwen3-moe-30b-int4`_

## Throughput vs vLLM

| c | ferrum (tok/s) | vLLM (tok/s) | ratio | gap to 0.80 |
|---:|---:|---:|---:|---:|
| 1 | — | — | — | — |
| 4 | — | — | — | — |
| 16 | — | — | — | — |
| 32 | — | — | — | — |

## Latency vs vLLM (p50)

| c | ferrum TTFT | vLLM TTFT | ratio | ferrum TPOT | vLLM TPOT | ratio |
|---:|---:|---:|---:|---:|---:|---:|

## ferrum chrome-trace category split (Phase 1.5)


### c=1

| category | total µs | % |
|---|---:|---:|
| moe | 9,499,434 | 73.4% |
| attention | 2,471,762 | 19.1% |
| decode_step | 971,404 | 7.5% |

### c=16

| category | total µs | % |
|---|---:|---:|
| moe | 4,391,443 | 73.4% |
| attention | 1,161,389 | 19.4% |
| decode_step | 426,552 | 7.1% |

### c=32

| category | total µs | % |
|---|---:|---:|
| moe | 4,386,612 | 73.4% |
| attention | 1,167,602 | 19.5% |
| decode_step | 423,463 | 7.1% |

## Env metadata

- ferrum commit: `cbe04ea`
- env_hash: `sha256:4ef7f61be78e5c0fac993f4a665d1833b23c31f175107dda30b29aea88b5bfa6`
