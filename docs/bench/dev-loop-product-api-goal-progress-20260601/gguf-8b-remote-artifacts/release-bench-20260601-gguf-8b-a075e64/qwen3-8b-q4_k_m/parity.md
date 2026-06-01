# Config parity — 2026-06-01T10:02:19Z

Model:    qwen3:8b-q4_k_m (vllm sees `Qwen/Qwen3-8B-GGUF:Q4_K_M`, bench sends `Qwen/Qwen3-8B-GGUF:Q4_K_M`)
Sweep:    1,4,16,32
Repeats:  3
Warmup:   10
Prompts:  128 per cell
SLO:      ttft=500  tpot=50  e2el=30000 (ms)

| knob | ferrum | vllm | parity |
|---|---|---|---|
| gpu_memory_utilization | (computed at load) | 0.85 | ⚠ inspect after run |
| max_seqs | 32 | 32 | ✓ |
| max_num_batched_tokens | (engine constant) | 2048 | ⚠ inspect |
| chunked_prefill | on (Phase 3 path) | true | ✓ |
| prefix_caching | off | off | ✓ |
| dtype | (model config) | fp16 | ⚠ inspect after run |

**Action required**: confirm the ⚠ rows match before trusting any ratio.
