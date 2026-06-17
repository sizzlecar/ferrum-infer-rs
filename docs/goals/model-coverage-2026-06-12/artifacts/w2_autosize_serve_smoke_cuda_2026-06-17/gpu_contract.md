# W2 Autosize Serve Smoke CUDA Contract

- Date: 2026-06-17
- Lane: W2 CUDA serve autosizer correctness smoke
- Hardware target: 1x NVIDIA RTX 4090, preferably by restarting cached Vast instance 41256521
- Expected runtime/cost: 20-45 minutes if the cached build/model environment is usable, estimated below $0.35 at the prior roughly $0.42/hour rate
- Stop condition: stop the instance after a `ferrum serve` streaming smoke PASS/FAIL artifact is copied back locally
- Correctness gate:
  - Build current local `HEAD` with `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
  - Start `ferrum serve gemma3:27b-gptq --backend cuda --kv-capacity 512 --max-num-seqs 32`
  - Send a streaming `/v1/chat/completions` request
  - Require no OOM/panic, exactly one `data: [DONE]`, usage metadata, and non-empty generated content
- Performance command: none for this lane; full same-hardware performance matrix is deferred until correctness passes
