# W2 Gemma3 CUDA unified-tail product correctness smoke

- lane: W2 Gemma3 CUDA unified-tail product correctness smoke
- expected runtime/cost: 10-25 minutes, hard cap 35 minutes, approximately USD 0.425/hr on existing 1x RTX 4090 Vast instance 40826362
- stop condition: instance start/SSH/CUDA/source sync/build first failure, `ferrum run` first failure, `ferrum serve` first failure, malformed/garbage output, missing usage or missing stream DONE, or smoke PASS after artifacts copied back
- correctness gate: CUDA release build, `ferrum run` Paris/multi-turn smoke, `ferrum serve` non-stream and streaming smoke
- performance command: none; this checkpoint is correctness-only and produces no performance claim
