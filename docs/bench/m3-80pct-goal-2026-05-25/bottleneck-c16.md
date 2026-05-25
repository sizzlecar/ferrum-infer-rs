# Bottleneck — c=16 (Qwen3-30B-A3B-GPTQ-Int4 / RTX 4090)

## ferrum chrome-trace category split

| category | µs | % |
|---|---:|---:|
| moe | 4,391,443 | 73.4% |
| attention | 1,161,389 | 19.4% |
| decode_step | 426,552 | 7.1% |

_(no nsys profile captured for this cell — see c=32 for kernel-level data)_
