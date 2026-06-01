# Metal Qwen3-8B Q4_K_M smoke - 2026-06-01

- commit: e7c6f42
- model: /Users/chejinxuan/ferrum-bench/models/Qwen3-8B-Q4_K_M.gguf
- tokenizer: /Users/chejinxuan/ferrum-bench/tokenizers/Qwen3-8B.tokenizer.json
- backend: metal
- kv_capacity: 1024
- swap_before: total = 2048.00M  used = 1644.06M  free = 403.94M  (encrypted)

## Correctness

- semantic_paris_answer: PASS

```text
The capital of France is Paris.

```

## Performance

Prompt: `Write a concise technical explanation of why caching improves inference throughput. Continue until the answer is complete.`

```text
run_1: [64 tokens, 23.8 tok/s, 2.7s]
run_2: [64 tokens, 23.5 tok/s, 2.7s]
run_3: [64 tokens, 23.5 tok/s, 2.7s]

```

- decode throughput median: 23.50 tok/s
- decode throughput values: [23.8, 23.5, 23.5]
- generated token counts: [64, 64, 64]
- swap_after: total = 2048.00M  used = 1644.06M  free = 403.94M  (encrypted)
