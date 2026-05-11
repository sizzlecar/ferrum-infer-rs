# Phase B-2 / C / D Backend trait refactor — bench result

Date: 2026-05-12  
Commit: `81b5c46` (PRs #163-#173)  
Hardware: RTX 4090 24GB on vast.ai pod  
Model: Qwen3-30B-A3B-GPTQ-Int4 (`Qwen/Qwen3-30B-A3B-GPTQ-Int4`)  
Bench: `bench/v0.2-cuda/m3_bench_serve.sh`, release build with `--features cuda,vllm-moe-marlin`  
Workload: random_input_len=256, random_output_len=128, num_prompts = `c * 4`

## Result

| c | output tok/s | mean TPOT | mean TTFT | p99 TPOT | completed |
|---|-------------:|----------:|----------:|---------:|----------:|
| 1  | 127.2 | 7.54 ms  | 49 ms  | —      | 4/4 |
| 8  | 432.8 | 16.63 ms | 243 ms | —      | 32/32 |
| 16 | 582.1 | 24.42 ms | 408 ms | —      | 64/64 |
| 32 | **717.5** | **39.11 ms** | 736 ms | 48.69 ms | 128/128 |

All concurrency levels: `fail=0`.

## Comparison vs pre-refactor baseline

| c | pre (#162) | post (#173) | Δ tok/s | Δ TPOT |
|---|-----------:|------------:|--------:|-------:|
| 1  | 96.2  | 127.2 | **+32%**  | -25% |
| 8  | 241.5 | 432.8 | **+79%**  | -47% |
| 16 | 272.1 | 582.1 | **+114%** | -56% |
| 32 | 318.4 | 717.5 | **+125%** | -59% |

vs vLLM 0.20.1 baseline (~1870 tok/s at c=32):
- pre-refactor: 17%
- post-refactor: ~38%

## Root cause hypothesis (perf wasn't the refactor goal — side effect)

1. **PR #163 `write_u32` stream-ordering fix**: The pre-refactor path used `cuMemcpyHtoD_v2` on the legacy NULL stream — which under CUDA's per-thread default stream model isn't synchronized with `ctx.stream`. The MoE host-topk fallback path was reading stale `cache_lens`. Refactor routes all typed writes through `stream.memcpy_htod` + explicit synchronize on `ctx.stream`, eliminating the race.
2. **Phase C trait-object cleanup**: dispatch through `MarlinExpertStack<B>` (PRs #166-#173) removed one layer of `Backend::moe_gemm_phase_batched` trait method dispatch per expert per layer. At c=32 / 48 layers / ~100 active experts that's ~480k indirect calls/decode collapsed.
3. **`upload_moe_routing` leak fix** (PR #173): the pre-refactor path leaked `~MB/s` of i32 routing buffers (the `mem::forget` after `upgrade_device_ptr`) — typed `CudaBuf::I32` now owns the memory cleanly.

## Reproduce

```bash
# On RTX 4090 pod (or any sm_89+ GPU with ≥20GB VRAM):
cd /workspace/ferrum-infer-rs
CUDA_HOME=/usr/local/cuda \
  cargo build --release -p ferrum-cli --features cuda,vllm-moe-marlin

# Model from HF (17GB):
mkdir -p /workspace/.hf_home/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/manual
cd /workspace/.hf_home/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/manual
for f in config.json generation_config.json merges.txt model.safetensors \
         quantize_config.json tokenizer.json tokenizer_config.json vocab.json; do
    curl -sL -o $f https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/resolve/main/$f
done
cd /workspace/ferrum-infer-rs
WORKSPACE=/workspace HF_HOME=/workspace/.hf_home bash bench/v0.2-cuda/m3_bench_serve.sh
```

## Active perf gap

Now at ~38% of vLLM 0.20.1 at c=32. Remaining headroom:

- **`Qwen3MoeModel` has no CUDA Graph capture** (vs `LlamaFamilyModel` which has full unified-graph + piecewise-graph paths). This is the canonical next perf gap per `CLAUDE.md`. Sprint-scale port.
- **vLLM MoE feature parity**: BatchDescriptor-cached PagedAttn V2, Marlin tile-fixed dispatch, FP32 reduce on every MoE gemm. Some already ported; full parity is ~50% perf ceiling without these.
