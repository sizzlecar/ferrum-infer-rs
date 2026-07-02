# W2 Dynamic KV C32 Serve Smoke CUDA Contract

- Lane: W2 dynamic paged-KV c32 serve correctness smoke.
- Hardware: exactly 1x RTX 4090 Vast instance, preferring cached instance 41241013.
- Expected runtime/cost: 10-25 minutes if cached build/model are reusable; roughly <$0.20 at the prior $0.471/hour rate.
- Stop condition: after one dynamic-KV `ferrum serve` correctness smoke PASS/FAIL artifact is copied back, stop the instance and verify `actual_status=exited`.
- Correctness gate: build current HEAD with CUDA features, start `ferrum serve gemma3:27b-gptq --backend cuda --max-num-seqs 32` with dynamic paged KV defaults, send streaming `/v1/chat/completions`, require non-empty output, exactly one `data: [DONE]`, usage tokens, effective `FERRUM_PAGED_MAX_SEQS` remaining 32, and no OOM/panic scan hits.
- Performance command: none in this smoke. If correctness passes, a later diagnostic run may use `ferrum bench-serve --fail-on-error --seed 9271`; release evidence still requires the goal gate and repeated same-hardware benchmark artifacts.
