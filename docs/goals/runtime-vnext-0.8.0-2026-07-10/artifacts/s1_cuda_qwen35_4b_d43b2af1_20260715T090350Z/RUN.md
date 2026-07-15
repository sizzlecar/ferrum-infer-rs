# S1 CUDA Qwen3.5-4B Diagnostic

- Lane: `S1 CUDA basic vertical slice`
- Initial source SHA: `d43b2af1`
- Branch: `goal/runtime-vnext-0.8.0`
- Model: `Qwen/Qwen3.5-4B`
- Hardware: `1x NVIDIA RTX 4090 24 GB`
- Vast instance: `44965014`
- Vast offer: `36887014`
- Requested disk: `150 GB`
- Actual total rate: `$0.4166666667/hour`
- Expected runtime: `2-4 hours`
- Expected cost: `$0.83-$1.67`

## Stop Conditions

1. CUDA feature compilation fails: save the exact compiler log and stop before model work.
2. `ferrum run` crashes, emits malformed UTF-8/reserved-token garbage, or produces no token: save the product log and do not broaden the run.
3. `ferrum serve` non-stream or stream violates the basic OpenAI/SSE contract: save the response and server log and do not start performance work.
4. The instance is idle for more than 10 minutes or reaches four paid hours without a new PASS/REJECT artifact: stop it and reassess from source/artifacts.

## Correctness Order

1. CUDA compile/check with `cuda,vllm-moe-marlin,vllm-paged-attn-v2`.
2. Qwen3.5-4B `ferrum run` single-turn smoke with product-visible defaults.
3. Shared `ferrum serve` non-stream and streaming smoke.
4. Capture vNext basic/resource trace and scan crash, mojibake, reserved-token, and SSE blockers.

## Diagnostic Performance Command

Correctness must pass before this command is treated as evidence.

```text
target/release/ferrum bench-serve --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.5-4B --tokenizer /workspace/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/<REVISION> --target-backend cuda --concurrency 1 --dataset random --random-input-len 64 --random-output-len 32 --num-prompts 8 --warmup-requests 1 --n-repeats 1 --fail-on-error --seed 9271 --out <ARTIFACT_DIR>/bench-c1-smoke.json
```

This is diagnostic smoke only and is not release performance evidence.

## Diagnostic Ledger

- Attempt 1, `cefb8a54`: REJECT before allocation. CUDA causal-attention provider
  version `2.0` could not satisfy the standard operation's incorrect provider
  minimum `1.0`. Fixed by `abf89fbe`; the local standard-operation contract
  gate passed `4/4`.
- Attempt 2, `abf89fbe`: REJECT during plan compilation. The real HF
  `linear_attn.conv1d.weight` has physical shape `[8192, 1, 4]`, while the
  semantic operation requires `[8192, 4]`. The typed schema now retains the
  physical shape and exposes a zero-copy strided singleton reshape in
  `94511e4f`; Qwen3.5 semantic/schema tests passed `4/4` and compiler contract
  tests passed `4/4`.
- Attempt 3 prediction: plan compilation must pass
  `node.layer.0.attention`; otherwise reject the reshape hypothesis. Stop at the
  first new initialization or execution failure and preserve its exact class.
