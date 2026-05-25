# Bottleneck — c=1 (Qwen3-30B-A3B-GPTQ-Int4 / RTX 4090)

## ferrum chrome-trace category split

| category | µs | % |
|---|---:|---:|
| moe | 9,499,434 | 73.4% |
| attention | 2,471,762 | 19.1% |
| decode_step | 971,404 | 7.5% |

_(no nsys profile captured for this cell — see c=32 for kernel-level data)_
