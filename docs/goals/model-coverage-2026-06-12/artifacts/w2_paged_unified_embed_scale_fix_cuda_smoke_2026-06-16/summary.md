# W2 paged-unified embed-scale fix CUDA smoke

## Verdict

Diagnostic correctness smoke passed for the fixed source checkpoint, but this is
not release evidence.

The remote source intentionally included a temporary diagnostic patch that
allowed paged KV for windowed Gemma3 when the CUDA backend supports varlen QKV.
The checked-in default guard was still protected at this checkpoint.

## Inputs

- Local source checkpoint: `fb6789c7f99cc08f05842503846ea42af2be842d`
- Vast instance: `40826362`, 1x RTX 4090
- CUDA driver/runtime visibility: `NVIDIA-SMI 565.77`, CUDA `12.7`
- Build command: CUDA release build for `ferrum-cli` with
  `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
- Build rc: `0`
- Binary SHA256:
  `e131ce885efb3f8aeb6049a9181f646638c4c8f81d0c993cfb33da29a4d7bc65`
- Correctness command: `ferrum serve` plus one non-stream chat request with
  `max_tokens=1`
- Performance command: none

## Request

```json
{"model":"gemma3:27b-gptq","messages":[{"role":"user","content":"What is 2+3? Answer with only the number."}],"max_tokens":1,"temperature":0,"stream":false}
```

## Response

```json
{"id":"c95f5315-ff3e-4a1e-8398-2e052530046f","object":"chat.completion","created":1781558998,"model":"gemma3:27b-gptq","choices":[{"index":0,"message":{"role":"assistant","content":"5"},"finish_reason":"length"}],"usage":{"prompt_tokens":23,"completion_tokens":1,"total_tokens":24}}
```

## Key Logs

```text
[unified-logits] call#0 row=0 orig_idx=0 global=22 finite=262208 nan=0 pos_inf=0 neg_inf=0 top=[236810:42.031250,239374:20.453125,247918:20.453125,239341:20.187500,242323:20.015625]
[unified-decode] call#0 items=1 prefill=1 decode=0 total_q=23 attempted_unified=true fallback=false fallback_reason=none elapsed=141708us
```

## Interpretation

- The pre-fix diagnostic ranked EOS token id `106` (`<end_of_turn>`) as top-1 and
  returned an empty response with `finish_reason=stop`.
- After applying the unified-path `embed_scale` fix, the same minimal smoke shape
  returned the expected first token `5`.
- Sampled logits were finite, had no NaN/Inf, and no longer ranked EOS top-1.
- This confirms the missing Gemma3 embedding scale in `unified_forward_internal`
  was a real correctness bug in the paged-unified product path.

## Shutdown

- Server process was stopped before artifact copyback completed.
- `nvidia_smi_after_stop.txt` showed no running GPU processes.
- Vast shutdown poll verified `cur_state=stopped actual_status=exited`.

## Remaining Work

- Promote the paged-KV guard change into checked-in source only after local unit
  tests are updated.
- Then rerun default product-path CUDA correctness for both `ferrum run` and
  `ferrum serve` without the remote diagnostic patch.
- Do not run c16/c32 performance comparisons until default-path correctness is
  clean.
