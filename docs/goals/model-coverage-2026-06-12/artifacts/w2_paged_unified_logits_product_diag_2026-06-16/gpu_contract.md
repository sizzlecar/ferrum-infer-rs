# W2 paged-unified product logits diagnostic

- lane: W2 paged-unified product logits diagnostic
- instance: Vast 40826362, 1x RTX 4090, cache-retained CUDA machine
- expected runtime/cost: 10-20 minutes, about USD 0.425/hr while running
- stop condition: start/SSH/CUDA/source sync/diagnostic patch/build/serve/chat first failure, or `[unified-logits]` evidence collected, then copy artifacts and stop instance
- correctness command: `ferrum serve` + one non-stream chat smoke with `max_tokens=1`
- performance command: none; diagnostic only
- diagnostic caveat: remote source is intentionally dirty with a temporary paged-KV guard override. This is not product validation or release evidence.
