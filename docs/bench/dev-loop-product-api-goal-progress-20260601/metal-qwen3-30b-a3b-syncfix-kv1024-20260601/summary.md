# Metal Qwen3-30B-A3B regression after prefill sync fix, KV=1024 - 2026-06-01

- base_head: 6016c1d
- code_state: base head plus uncommitted qwen3_moe prefill sync fix
- model: /Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf
- tokenizer: /Users/chejinxuan/ferrum-bench/tokenizers/Qwen3-30B-A3B.tokenizer.json
- kv_capacity: 1024
- local_swap_before: total = 2048.00M  used = 1100.81M  free = 947.19M  (encrypted)

## Correctness

- semantic_paris_answer: PASS

```text
The capital of France is **Paris**.
```

## Performance

### pp512 prefill raw

```text

```

- pp512 prefill parsed median: unavailable; inspect raw lines.

### tg128 decode raw

```text

```

- tg128 throughput parsed median: unavailable; inspect raw lines.

