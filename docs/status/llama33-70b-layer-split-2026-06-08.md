# Llama 3.3 70B 4bit Layer Split Status

Date: 2026-06-08
Branch: `checkpoint/backend-runtime-preset-gates`
Git SHA: `89daf6e983c50081a411d08c014c61ac00cc0044`

## Summary

Ferrum can run `clowman/Llama-3.3-70B-Instruct-GPTQ-Int4` on two RTX 4090 GPUs through the product `layer_split` path. The validated split is:

```text
stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79
```

This is a correct sequential layer-split implementation, not tensor parallel and not yet overlapped pipeline parallelism.

## Evidence

Final goal PASS:

```text
LLAMA33_70B_4BIT_2X4090 GOAL PASS: docs/release/g0/llama33-2x4090-goal-final-20260608-89daf6e9
```

70B lane PASS:

```text
G0 SOURCE g0_cuda2x4090_llama33_70b_4bit PASS: docs/release/g0/llama33-2x4090-ferrum-only-full-20260608-89daf6e9
```

Binary SHA256:

```text
0f99fc0775d545e5f74c07ca01256a7f8987479dc21916e6320efdeeba2821f3
```

Build features:

```text
cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source
```

The raw release evidence directories are intentionally not committed with this note.

## Performance Snapshot

The full Ferrum-only `bench-serve` gate used:

```text
--fail-on-error --require-ci --seed 9271 --n-repeats 3
--concurrency-sweep 1,4,8,16
```

Each concurrency cell completed `96/96` requests for each repeat with `0` errored requests and `0` bad outputs. Token counts came from OpenAI usage fields.

Mean output throughput:

| Concurrency | Output throughput |
| --- | ---: |
| 1 | 20.85 tok/s |
| 4 | 20.87 tok/s |
| 8 | 20.85 tok/s |
| 16 | 20.80 tok/s |

The flat throughput curve is expected for the current implementation. Higher concurrency increases queueing and per-request latency, but it does not increase total decode throughput.

## Current Execution Model

The current `LlamaFamilyPipelineModel` runs stages sequentially:

```text
stage0 forward on GPU0
sync
copy hidden state to host Vec<f32>
copy hidden state from host to GPU1
stage1 forward on GPU1
sync
copy logits to host
```

Relevant implementation points:

- `crates/ferrum-models/src/models/llama_family_pipeline.rs`
  - `prefill` and `decode` call each stage in order.
- `crates/ferrum-models/src/models/llama_family.rs`
  - stage bridge methods call `B::sync()` and return host `Vec<f32>`.
- `crates/ferrum-models/src/executor/llm_executor.rs`
  - the executor holds one model lock while running model forward.

This path is useful because it makes 70B 4bit fit on two 24 GB GPUs and keeps correctness straightforward. It does not keep both GPUs busy in steady state for decode.

## Pipeline Overlap Direction

Pipeline overlap should remain layer split: every transformer layer still lives entirely on one GPU. The difference is scheduling.

Target steady state:

```text
GPU1: stage1(microbatch i)
GPU0: stage0(microbatch i + 1)
```

Practical implementation order:

1. Add a batch-aware `decode_batch` implementation for `LlamaFamilyPipelineModel`.
   - Run stage0 over an `M`-row batch.
   - Run stage1 over the resulting `M` hidden rows.
   - Keep the host bridge at first to isolate batching correctness.
2. Add a device-resident stage bridge.
   - Replace host `Vec<f32>` transfer with a typed hidden buffer.
   - Add CUDA peer/device copy transport while keeping host fallback for CPU/Metal.
3. Add microbatch pipeline workers.
   - Split a decode batch into microbatches.
   - Drive stage0 and stage1 concurrently with queues and CUDA events.
   - Preserve request order at the collector.

Expected effect:

- Batch-aware layer split should show whether c=4/c=8 throughput can rise before full overlap.
- True overlap can improve steady-state throughput only when stages are reasonably balanced and stage transfer is not dominant.
- Single-request latency is not expected to improve materially; this is a serving-throughput optimization.
