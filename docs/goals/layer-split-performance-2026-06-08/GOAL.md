# Layer Split Performance Goal

## 状态

草案目标文件。

本目标不能因为“代码改完了”或“单次 benchmark 数字上涨了”就宣称完成。只有最终验证器打印下面这一行，才算完成：

```text
LAYER_SPLIT_PERF GOAL PASS: <out_dir>
```

## 目标

提升 Ferrum 当前 Llama 70B-class 2x4090 layer split 服务吞吐，让双卡不只用于装下模型，也能在并发 decode 下产生可证明的吞吐提升。

本目标专注当前架构：

- 每个 transformer layer 仍完整放在一张 GPU 上。
- 不做 tensor parallel。
- 不做 speculative decoding。
- 不通过隐藏环境变量启用产品默认行为。
- 优先优化 `ferrum serve` 的并发总吞吐，同时保证 `ferrum run` 不回归。

最终性能目标：在相同 2xRTX 4090 硬件、同一 Llama 70B-class 4bit 模型、同一产品路径和同一 benchmark 命令下，Ferrum 输出吞吐达到选定公开主流引擎基准的至少 80%。

## 当前基线

当前已验证 Ferrum 能通过产品 `layer_split` 路径运行：

```text
clowman/Llama-3.3-70B-Instruct-GPTQ-Int4
```

已验证 split：

```text
stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79
```

当前完成证据：

```text
LLAMA33_70B_4BIT_2X4090 GOAL PASS: docs/release/g0/llama33-2x4090-goal-final-20260608-89daf6e9
G0 SOURCE g0_cuda2x4090_llama33_70b_4bit PASS: docs/release/g0/llama33-2x4090-ferrum-only-full-20260608-89daf6e9
```

基线 metadata：

- Git SHA: `89daf6e983c50081a411d08c014c61ac00cc0044`
- Binary SHA256: `0f99fc0775d545e5f74c07ca01256a7f8987479dc21916e6320efdeeba2821f3`
- Build features: `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
- Bench flags: `--fail-on-error --require-ci --seed 9271 --n-repeats 3 --concurrency-sweep 1,4,8,16`
- Each cell: `completed=96/96`, `errored=0`, `bad_outputs=0`
- Token count source: OpenAI usage fields

Current mean output throughput:

| Concurrency | Output throughput |
| --- | ---: |
| 1 | 20.85 tok/s |
| 4 | 20.87 tok/s |
| 8 | 20.85 tok/s |
| 16 | 20.80 tok/s |

This flat curve is the performance problem. It shows correctness and fit are solved, but concurrent serving does not yet keep both GPUs busy in steady state.

## External Baseline Policy

External numbers are not Ferrum evidence. They are target-setting inputs only.

A valid external baseline candidate must record:

- model family and model size;
- quantization or dtype;
- hardware, including GPU count and GPU model;
- engine and version when available;
- metric definition, especially single-stream decode vs aggregate output throughput;
- context, prompt length, output length, concurrency, and whether speculative decoding is used;
- source URL and access date.

Baseline selection priority:

1. Same hardware: 2x RTX 4090 over other consumer or datacenter GPUs.
2. Same model class: Llama 70B-class dense over smaller models.
3. Same precision class: 4bit over FP16, FP8, IQ2, or speculative paths.
4. Same serving shape: OpenAI-compatible serving aggregate output throughput over local single prompt demos.
5. Reproducible project or engine source over affiliate hardware guides or community anecdotes.

Do not use these as hard baselines:

- speculative decoding numbers;
- extreme low-bit IQ2/IQ3 numbers when the model quality class differs from 4bit GPTQ/AWQ/GGUF Q4;
- datacenter GPU numbers;
- single 24GB 4090 AWQ paths that rely on a model format or KV dtype Ferrum does not support in the same product path;
- numbers without enough methodology to distinguish prompt throughput from output throughput.

## Public Baseline Snapshot

Access date: 2026-06-08.

The public numbers found are not perfectly apples-to-apples, so this goal uses a tiered target.

| Source | Engine / architecture | Hardware | Model / quant | Metric | Published number | Use in this goal |
| --- | --- | --- | --- | --- | ---: | --- |
| MLC blog, 2023-10-19, `https://blog.mlc.ai/2023/10/19/Scalable-Language-Model-Inference-on-Multiple-NVDIA-AMD-GPUs` | MLC LLM, SPMD/tensor-parallel style | 2x RTX 4090 PCIe | Llama2-70B 4bit | single-batch decode, prefill=8, decode=256 | 34 to 34.5 tok/s | primary reproducible target floor |
| vLLM docs, `https://docs.vllm.ai/en/latest/serving/parallelism_scaling/` | vLLM tensor parallel / pipeline parallel guidance | multi-GPU | general large models | design guidance | use tensor parallel for single-node multi-GPU when model is too large for one GPU | architecture reference, not a numeric baseline |
| llama.cpp CLI docs, `https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md` | layer split / row split / tensor split modes | multi-GPU | GGUF models | split mode behavior | `layer` is pipelined, `row` and `tensor` are parallelized modes | architecture reference |
| LLMHardware dual GPU guide, `https://llmhardware.io/guides/dual-gpu-llm-setup-guide` | llama.cpp/Ollama/vLLM guide | 2x RTX 4090 | Llama 3 70B Q4 | approximate generation throughput | 35-45 tok/s | secondary target range; methodology is approximate |
| WillItRunAI vLLM guide, `https://willitrunai.com/blog/vllm-multi-gpu-setup-guide` | vLLM tensor parallel | 2x RTX 4090 | 70B Q4 | aggregate serving estimate | 25-30 tok/s | lower mainstream serving target range |
| WillItRunAI Ollama guide, `https://willitrunai.com/blog/ollama-multi-gpu-guide` | Ollama layer split | 2x RTX 4090 | Llama 70B Q4 | approximate decode | 25-30 tok/s | layer-split sanity range |
| Local AI Master Ollama/vLLM comparison, `https://localaimaster.com/blog/ollama-multi-gpu-setup` | Ollama pipeline, vLLM/TGI tensor parallel | 2x RTX 4090 | Llama 3.3 70B Q4_K_M | 512 prompt, 256 completion, temp 0 | Ollama 22.6, vLLM 41.2, TGI 38.9 tok/s | secondary comparison only |

Initial target math:

- Primary floor: `0.80 * 34.5 = 27.6 tok/s`.
- Mainstream tensor-parallel stretch: `0.80 * 41.2 = 33.0 tok/s`.
- Approximate guide range floor: `0.80 * 35 = 28.0 tok/s`.

Therefore this goal has two performance thresholds:

- Required: Ferrum reaches at least `27.6 tok/s` aggregate output throughput on the selected same-hardware full gate.
- Stretch: Ferrum reaches at least `33.0 tok/s` aggregate output throughput, or the goal document is updated with a stronger same-hardware vLLM baseline collected by us.

The required target is a floor, not an end state. If we run a same-pod current vLLM baseline before implementation and it is higher than 34.5 tok/s, the final target becomes `0.80 * same_pod_vllm_output_tps`.

## Non-Goals

- Do not implement tensor parallel in this goal.
- Do not target Qwen3 MoE layer split in this goal.
- Do not add a second HTTP benchmark client.
- Do not change official G0 release readiness rules.
- Do not tag, publish, or claim release-ready.
- Do not make performance claims from public web numbers alone.
- Do not hide behavior behind undocumented `FERRUM_*` combinations.
- Do not accept lower correctness quality, missing usage counts, missing `[DONE]`, or bad-output filtering to win throughput.

## Implementation Plan

### Phase 0. Baseline Refresh and Profiling

Before changing runtime behavior, collect a fresh baseline on the target 2x4090 lane:

- git SHA and dirty status;
- binary SHA256;
- build features;
- driver, CUDA runtime, GPU model, PCIe link width;
- model id/path, model file manifest/hash, tokenizer metadata;
- selected runtime preset and effective config;
- selected layer split plan;
- per-GPU memory and utilization snapshots;
- `ferrum run` correctness;
- `ferrum serve` correctness and benchmark artifacts.

Profiling must answer:

- stage0 decode time;
- stage1 decode time;
- hidden-state bridge time;
- host copy vs device copy time;
- model lock wait time;
- scheduler/admission wait time;
- GPU utilization for both devices at c=1, c=4, c=8, c=16.

### Phase 1. Batch-Aware Layer Split

Add batch-aware decode for `LlamaFamilyPipelineModel`.

Current behavior effectively serializes work per request. The first target is not overlap yet; it is to make the layer-split model consume an `M`-row decode batch:

```text
stage0.decode_batch(rows[0..M]) -> hidden[M, hidden_size]
stage1.decode_batch(hidden[M, hidden_size]) -> logits[M, vocab]
```

Keep the existing host bridge initially. This isolates batch correctness from transport and overlap.

Acceptance for this phase:

- c=1 remains within 10% of baseline unless the regression is explained and accepted.
- c=4 or c=8 aggregate throughput rises materially above the flat 20.8 tok/s baseline.
- request ordering and per-sequence KV state are preserved.
- `output_token_count_source == usage` for release-quality benchmark artifacts.

### Phase 2. Device-Resident Stage Bridge

Replace the default host `Vec<f32>` hidden-state transfer with a typed hidden buffer abstraction:

```text
PipelineHidden {
  shape: [batch, hidden_size],
  dtype,
  device,
  layout,
}
```

CUDA path should support device-resident transfer:

- peer copy when peer access is available;
- device-to-device staged copy when direct peer access is unavailable;
- explicit host fallback for unsupported backends or diagnostic mode.

Acceptance for this phase:

- artifacts record bridge mode: `host`, `cuda_peer`, or `cuda_device_staged`.
- if a CUDA run requests device bridge but falls back to host, the artifact is diagnostic only.
- hidden buffer lifetime is explicit and does not rely on dangling backend-local state.
- CPU/Metal compile paths are preserved.

### Phase 3. Microbatch Pipeline Overlap

Add overlapped scheduling while keeping layer ownership unchanged.

Target steady state:

```text
GPU1: stage1(microbatch i)
GPU0: stage0(microbatch i + 1)
```

Implementation shape:

- split a decode batch into bounded microbatches;
- drive stage0 and stage1 with explicit queues;
- use CUDA events or equivalent backend synchronization to order buffer handoff;
- preserve output ordering at the collector;
- expose queue depth, microbatch size, in-flight stage count, and per-stage timing in artifact metadata.

Acceptance for this phase:

- no request output corruption under c=1, c=4, c=8, c=16;
- no duplicate or missing stream `[DONE]`;
- no request cross-talk between concurrent sequences;
- no unbounded queue growth;
- both GPUs show sustained utilization during c=4/c=8/c=16 decode.

### Phase 4. Scheduler and Admission Tuning

Tune serving defaults for this path after the runtime can actually batch or overlap.

Required effective config fields:

- `selected_distributed_strategy = layer_split`
- `selected_pipeline_mode = sequential | batch | overlapped`
- `selected_microbatch_size`
- `selected_stage_bridge`
- `selected_max_sequences`
- `selected_max_batched_tokens`
- `selected_admission_limit`
- `selected_kv_capacity`
- `selected_max_model_len`

Admission must be productized through typed config or documented CLI/config, not hidden env.

## Correctness Gates

Correctness gates must pass before performance measurements count as evidence.

At minimum, every release-quality artifact for this goal must cover:

- `ferrum run` single-turn correctness;
- `ferrum run` multi-turn correctness;
- `ferrum serve` single-turn correctness;
- `ferrum serve` multi-turn correctness;
- OpenAI-compatible streaming with exactly one `data: [DONE]`;
- streaming usage with `stream_options.include_usage=true`;
- tool calling;
- structured output;
- deterministic diagnostic settings where applicable;
- log scans for panic, CUDA error, OOM, `<unk>`, `[PAD]`, mojibake, duplicate `[DONE]`, missing `[DONE]`, malformed SSE JSON, and silent fallback.

If correctness fails, performance artifacts from the same run are diagnostic only.

## Performance Gate

The canonical HTTP benchmark client is still `ferrum bench-serve`.

Required benchmark shape:

```bash
ferrum bench-serve ... \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --n-repeats 3 \
  --concurrency-sweep 1,4,8,16
```

The final performance artifact must include:

- same-hardware Ferrum baseline artifact;
- same-hardware Ferrum candidate artifact;
- optional same-pod vLLM baseline artifact if collected;
- output throughput by concurrency;
- TTFT, TPOT, E2E latency, completed, failed, errored, bad output counts;
- confidence interval data for `n_repeats=3`;
- actual tokenizer-counted input length;
- output token count source;
- effective runtime config;
- binary SHA256;
- git SHA and dirty status.

Required threshold:

```text
max(c4,c8,c16 aggregate output throughput) >= 27.6 tok/s
```

Stretch threshold:

```text
max(c4,c8,c16 aggregate output throughput) >= 33.0 tok/s
```

If a same-pod vLLM baseline is collected, replace the fixed required threshold with:

```text
max(c4,c8,c16 Ferrum output throughput) >= 0.80 * same_pod_vllm_output_tps
```

The final report must state whether the goal passed the fixed public floor, the same-pod vLLM 80% target, or both.

## New Goal Gate

Add a final validator for this goal, for example:

```bash
python3 scripts/release/layer_split_perf_goal_gate.py \
  --out <out_dir> \
  --baseline-artifact <baseline_out> \
  --candidate-artifact <candidate_out> \
  --correctness-artifact <correctness_out> \
  --optional-vllm-artifact <vllm_out>
```

Required final PASS line:

```text
LAYER_SPLIT_PERF GOAL PASS: <out_dir>
```

The validator must reject:

- dirty or missing git metadata;
- missing binary SHA256 when available;
- missing effective config;
- missing selected layer split plan;
- missing correctness artifact;
- missing usage token counts;
- any failed request when `--fail-on-error` is expected;
- `n_repeats < 3` for release-quality performance evidence;
- output throughput below the selected target;
- artifacts marked diagnostic-only.

## Paid GPU Lane

Before starting any paid GPU run for this goal, state the lane, expected runtime/cost, stop condition, correctness gate, and performance command.

Suggested lanes:

- `layer-split-perf-smoke`
  - Expected runtime/cost: 30-60 minutes on a 2x4090 host, target about 1 USD/hour if available.
  - Stop condition: any correctness failure, model load failure, CUDA OOM, or throughput still flat after one candidate run.
  - Correctness gate: product run/serve smoke plus stream usage.
  - Performance command: one `bench-serve` sweep with `--fail-on-error --seed 9271`, diagnostic if `--require-ci` or `n_repeats=3` is skipped.
- `layer-split-perf-full`
  - Expected runtime/cost: 1-3 hours on a 2x4090 host, target about 1 USD/hour if available.
  - Stop condition: final PASS, correctness failure, or target miss with enough profiling to explain the miss.
  - Correctness gate: full goal correctness matrix.
  - Performance command: required `bench-serve` command with `--fail-on-error --require-ci --seed 9271 --n-repeats 3`.

Do not leave the paid host idle after evidence is collected.

## Checkpoints and Commits

Use small commits at checkpoints:

1. Goal doc and external baseline notes.
2. Profiling and artifact schema only.
3. Batch-aware layer split correctness.
4. Device-resident stage bridge.
5. Pipeline overlap.
6. Goal gate and final report.

Do not commit raw benchmark artifact directories into the repo. Store large or temporary records outside the repository, for example:

```text
../ferrum-infer-rs-records-<YYYYMMDD>/
```

## Open Questions

- Do we collect a same-pod current vLLM baseline before implementation, or initially use the MLC 34.5 tok/s public floor?
- Should the first accepted win target c=4 or c=8 specifically, or simply `max(c4,c8,c16)`?
- Can the CUDA backend expose peer-copy capability cleanly enough for a product default, or should device bridge start as an explicit documented experimental mode?
- Is the existing model lock in `llm_executor` compatible with overlapped stage workers, or does it need a narrow model-forward critical section?
