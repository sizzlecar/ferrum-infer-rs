# Metal Qwen3-30B-A3B regression after prefill sync fix - 2026-06-01

- code_head: 6016c1d

- model: /Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf
- tokenizer: /Users/chejinxuan/ferrum-bench/tokenizers/Qwen3-30B-A3B.tokenizer.json
- kv_capacity: 512
- fix: qwen3_moe prefill sync before logits to_vec


## Correctness

- result: FAIL
- expected_substring: `The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid.`

stdout excerpt:

```text
The capital of France is **Paris**.

```

stderr excerpt:

```text
[auto-size] MAX_BATCHED_TOKENS=2048 (profile=Chat)
Loading /Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf...
Using Metal backend
Loading weights to GPU... (30s+ for >10 GB models)
2026-06-01T11:59:41.181929Z  INFO ferrum_engine::registry: Initializing global component registry
2026-06-01T11:59:41.182005Z  INFO ferrum_engine::registry: Registering default component factories
2026-06-01T11:59:41.182215Z  INFO ferrum_engine::registry: Loading HuggingFace tokenizer from: "/Users/chejinxuan/ferrum-bench/tokenizers/Qwen3-30B-A3B.tokenizer.json"
2026-06-01T11:59:41.355812Z  INFO ferrum_engine::registry: HuggingFace tokenizer loaded successfully
2026-06-01T11:59:41.355909Z  INFO ferrum_engine::registry: Creating multinomial sampler
2026-06-01T11:59:41.355916Z  INFO ferrum_engine::registry: Creating priority scheduler
2026-06-01T11:59:41.355946Z  INFO ferrum_engine::registry: Creating default KV cache manager: device=Metal, block_size=16, max_blocks=2048
2026-06-01T11:59:41.356048Z  INFO ferrum_engine::registry: Loading model from /Users/chejinxuan/ferrum-bench/models/Qwen3-30B-A3B-Q4_K_M.gguf (format: gguf)
Model loaded in 1.4s.
[9 tokens, 4.6 tok/s, 1.9s]

```

## Correctness classification

- legacy_exact_expected_sentence: FAIL
- semantic_paris_answer: PASS
- metal_encoder_assertion: PASS_NOT_REPRODUCED_AFTER_SYNC_FIX
- code_state: base HEAD 6016c1d plus uncommitted qwen3_moe prefill sync fix

