# Bottleneck — c=32 (Qwen3-30B-A3B-GPTQ-Int4 / RTX 4090)

## ferrum chrome-trace category split

| category | µs | % |
|---|---:|---:|
| moe | 4,386,612 | 73.4% |
| attention | 1,167,602 | 19.5% |
| decode_step | 423,463 | 7.1% |

_(no nsys profile captured for this cell — see c=32 for kernel-level data)_
