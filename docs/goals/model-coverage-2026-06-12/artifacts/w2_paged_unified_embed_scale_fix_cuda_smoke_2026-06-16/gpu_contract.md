# W2 paged-unified embed-scale fix product smoke

- lane: W2 paged-unified embed-scale fix product smoke
- instance: Vast 40826362, 1x RTX 4090, cache-retained CUDA machine
- expected runtime/cost: 10-20 minutes, about USD 0.425/hr while running
- stop condition: start/SSH/source sync/diagnostic guard patch/build/serve/chat first failure, or fixed-path `[unified-logits]` and response evidence collected, then stop instance
- correctness command: `ferrum serve` + one non-stream chat request with `max_tokens=1`
- performance command: none; diagnostic correctness only
- diagnostic caveat: remote source is intentionally dirty with the same temporary paged-KV guard override as LXXXIV. The checked-in default guard remains protected.
