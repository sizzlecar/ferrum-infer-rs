# 层切分性能提升目标

## 状态

草案目标文件。

本目标不能因为“代码改完了”或“单次 benchmark 数字上涨了”就宣称完成。只有最终验证器打印下面这一行，才算完成：

```text
LAYER_SPLIT_PERF GOAL PASS: <out_dir>
```

## 目标

提升 Ferrum 当前 Qwen 72B 级别模型在 2x4090 层切分路径上的服务吞吐，让双卡不只是用于装下模型，也能在并发解码时产生可证明的吞吐提升。

本目标专注当前架构：

- 每个 transformer layer 仍完整放在一张 GPU 上。
- 不做张量并行。
- 不做推测解码。
- 不通过隐藏环境变量启用产品默认行为。
- 优先优化 `ferrum serve` 的并发总吞吐，同时保证 `ferrum run` 不回归。

最终性能目标：在相同 2xRTX 4090 硬件、同一 Qwen 72B 级别 4bit 模型、同一产品路径和同一 benchmark 命令下，Ferrum 输出吞吐达到选定公开主流引擎基准的至少 80%。若采集同 pod vLLM baseline，则以 Qwen 同 pod vLLM 的 80% 为优先目标。

## 当前基线

当前目标模型已改为：

```text
Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4
```

已验证切分计划：

```text
stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79
```

历史 Llama 70B layer-split 证据仅作为参考，不再作为本目标完成依据：

```text
LLAMA33_70B_4BIT_2X4090 GOAL PASS: docs/release/g0/llama33-2x4090-goal-final-20260608-89daf6e9
G0 SOURCE g0_cuda2x4090_llama33_70b_4bit PASS: docs/release/g0/llama33-2x4090-ferrum-only-full-20260608-89daf6e9
```

历史 Llama 基线元数据：

- Git SHA：`89daf6e983c50081a411d08c014c61ac00cc0044`
- 二进制 SHA256：`0f99fc0775d545e5f74c07ca01256a7f8987479dc21916e6320efdeeba2821f3`
- 构建特性：`cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`
- Benchmark 参数：`--fail-on-error --require-ci --seed 9271 --n-repeats 3 --concurrency-sweep 1,4,8,16,32`
- 每个 cell：`completed=96/96`，`errored=0`，`bad_outputs=0`
- Token 计数来源：OpenAI usage 字段

当前平均输出吞吐：

| 并发 | 输出吞吐 |
| --- | ---: |
| 1 | 20.85 tok/s |
| 4 | 20.87 tok/s |
| 8 | 20.85 tok/s |
| 16 | 20.80 tok/s |

这条平坦曲线是原 Llama 目标下暴露的性能问题。目标切换到 Qwen2.5-72B-GPTQ 后，需要重新采集 Qwen baseline 和 candidate；Llama 数字不再作为目标完成证据。

## 外部基准口径策略

外部数字不是 Ferrum 证据，只能用于设定目标。

一个有效的外部基准候选必须记录：

- 模型家族和模型规模；
- 量化格式或 dtype；
- 硬件，包括 GPU 数量和 GPU 型号；
- 引擎和版本，如果来源有记录；
- 指标定义，尤其要区分单流解码和聚合输出吞吐；
- 上下文长度、prompt 长度、输出长度、并发数，以及是否使用推测解码；
- 来源 URL 和访问日期。

基准选择优先级：

1. 同硬件：优先 2x RTX 4090，而不是其他消费卡或数据中心 GPU。
2. 同模型级别：优先 Qwen 72B 或同等 70B 级别 dense 模型，而不是更小模型。
3. 同精度级别：优先 4bit，而不是 FP16、FP8、IQ2 或推测路径。
4. 同服务形态：优先 OpenAI-compatible 服务聚合输出吞吐，而不是本地单 prompt 演示。
5. 优先可复现的项目或引擎来源，而不是硬件导购文章或社区轶事。

以下数字不能作为硬门槛：

- 推测解码数字。
- IQ2/IQ3 等极低 bit 数字，除非其模型质量级别可证明等价于 4bit GPTQ/AWQ/GGUF Q4。
- 数据中心 GPU 数字。
- 单张 24GB 4090 AWQ 路径，除非 Ferrum 在同一产品路径中支持相同模型格式和 KV dtype。
- 无法区分 prompt 吞吐和输出吞吐的方法不明数字。

## 公开基准快照

访问日期：2026-06-08。

查到的公开数字并不完全同口径，且多数不是 Qwen2.5-72B-GPTQ 的同模型数字。因此本目标使用分层目标：先保留 70B 级别公开下限作为临时硬门槛，同时优先采集 Qwen 同 pod vLLM baseline，一旦可用即替代公开下限。

| 来源 | 引擎和架构 | 硬件 | 模型和量化 | 指标 | 公开数字 | 在本目标中的用途 |
| --- | --- | --- | --- | --- | ---: | --- |
| MLC 博客，2023-10-19，`https://blog.mlc.ai/2023/10/19/Scalable-Language-Model-Inference-on-Multiple-NVDIA-AMD-GPUs` | MLC LLM，SPMD / 类张量并行 | 2x RTX 4090 PCIe | Llama2-70B 4bit | 单 batch 解码，prefill=8，decode=256 | 34 到 34.5 tok/s | 主要可复现目标下限 |
| vLLM 文档，`https://docs.vllm.ai/en/latest/serving/parallelism_scaling/` | vLLM 张量并行 / pipeline parallel 指南 | 多 GPU | 通用大模型 | 架构建议 | 单节点多 GPU 且模型无法放入单卡时使用张量并行 | 架构参考，不作为数字基准 |
| llama.cpp CLI 文档，`https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md` | layer / row / tensor split 模式 | 多 GPU | GGUF 模型 | split mode 行为 | `layer` 是 pipelined，`row` 和 `tensor` 是 parallelized | 架构参考 |
| LLMHardware 双卡指南，`https://llmhardware.io/guides/dual-gpu-llm-setup-guide` | llama.cpp / Ollama / vLLM 指南 | 2x RTX 4090 | Llama 3 70B Q4 | 近似生成吞吐 | 35-45 tok/s | 次级目标区间；方法偏近似 |
| WillItRunAI vLLM 指南，`https://willitrunai.com/blog/vllm-multi-gpu-setup-guide` | vLLM 张量并行 | 2x RTX 4090 | 70B Q4 | 聚合服务估算 | 25-30 tok/s | 较低主流服务目标区间 |
| WillItRunAI Ollama 指南，`https://willitrunai.com/blog/ollama-multi-gpu-guide` | Ollama 层切分 | 2x RTX 4090 | Llama 70B Q4 | 近似解码吞吐 | 25-30 tok/s | 层切分 sanity 区间 |
| Local AI Master Ollama / vLLM 对比，`https://localaimaster.com/blog/ollama-multi-gpu-setup` | Ollama pipeline，vLLM / TGI 张量并行 | 2x RTX 4090 | Llama 3.3 70B Q4_K_M | 512 prompt，256 completion，temperature 0 | Ollama 22.6，vLLM 41.2，TGI 38.9 tok/s | 仅作为次级对比 |

初始目标计算：

- 主要下限：`0.80 * 34.5 = 27.6 tok/s`。
- 主流张量并行 stretch：`0.80 * 41.2 = 33.0 tok/s`。
- 近似指南区间下限：`0.80 * 35 = 28.0 tok/s`。

因此本目标有两个性能阈值：

- 必达：Ferrum 在选定同硬件完整 gate 中达到至少 `27.6 tok/s` 聚合输出吞吐。
- Stretch：Ferrum 达到至少 `33.0 tok/s` 聚合输出吞吐，或者先由我们采集更强的同硬件 vLLM baseline，再更新目标文档。

必达目标只是下限，不是终点。如果实现前采集了同 pod 当前 Qwen vLLM baseline，且它高于 34.5 tok/s，则最终目标变为：

```text
0.80 * same_pod_vllm_output_tps
```

## 非目标

- 本目标不实现张量并行。
- 本目标不优化 Qwen3 MoE 层切分。
- 不新增第二套 HTTP benchmark 客户端。
- 不修改官方 G0 release-ready 规则。
- 不打 tag、不发布、不宣称 release-ready。
- 不仅凭公开网页数字做性能宣称。
- 不把行为藏在未文档化的 `FERRUM_*` 组合后面。
- 不用降低正确性质量、缺失 usage 计数、缺失 `[DONE]` 或过滤坏输出来换吞吐。
- 本目标不再以 `clowman/Llama-3.3-70B-Instruct-GPTQ-Int4` 作为完成模型；该模型只保留为历史诊断参考。

## 实现计划

### 阶段 0：刷新基线和性能剖析

改动运行时行为前，先在目标 2x4090 lane 上采集新基线：

- git SHA 和 dirty status；
- 二进制 SHA256；
- 构建特性；
- driver、CUDA runtime、GPU 型号、PCIe link width；
- model id/path、模型文件 manifest/hash、tokenizer metadata；
- 选中的 runtime preset 和 effective config；
- 选中的 layer split plan；
- 每张 GPU 的显存和利用率快照；
- `ferrum run` 正确性；
- `ferrum serve` 正确性和 benchmark artifacts。

性能剖析必须回答：

- stage0 解码耗时；
- stage1 解码耗时；
- hidden-state bridge 耗时；
- host copy 和 device copy 耗时；
- model lock 等待时间；
- scheduler / admission 等待时间；
- c=1、c=4、c=8、c=16、c=32 下两张 GPU 的利用率。

### 阶段 1：批量感知的层切分

为 `LlamaFamilyPipelineModel` 增加批量感知解码。

当前行为实质上按请求串行执行。第一步目标不是 overlap，而是让层切分模型能消费一个 `M` 行 decode batch：

```text
stage0.decode_batch(rows[0..M]) -> hidden[M, hidden_size]
stage1.decode_batch(hidden[M, hidden_size]) -> logits[M, vocab]
```

初期继续保留现有 host bridge。这样可以先隔离 batch 正确性，不同时混入传输和 overlap 变量。

本阶段验收标准：

- c=1 相比基线回退不超过 10%，除非有明确解释并被接受。
- c=4 或 c=8 聚合吞吐明显高于当前平坦的 20.8 tok/s 基线。
- 请求顺序和每个 sequence 的 KV 状态保持正确。
- release-quality benchmark artifacts 中 `output_token_count_source == usage`。

### 阶段 2：设备驻留的 stage bridge

用 typed hidden buffer 抽象替换默认 host `Vec<f32>` hidden-state 传输：

```text
PipelineHidden {
  shape: [batch, hidden_size],
  dtype,
  device,
  layout,
}
```

CUDA 路径应支持设备驻留传输：

- peer access 可用时使用 peer copy；
- direct peer access 不可用时使用 device-to-device staged copy；
- 不支持的 backend 或 diagnostic mode 显式使用 host fallback。

本阶段验收标准：

- artifact 记录 bridge mode：`host`、`cuda_peer` 或 `cuda_device_staged`。
- 如果 CUDA run 请求 device bridge 但 fallback 到 host，该 artifact 只能算 diagnostic。
- hidden buffer 生命周期明确，不依赖悬空的 backend-local 状态。
- CPU / Metal 编译路径不被破坏。

### 阶段 3：微批 pipeline overlap

在保持 layer ownership 不变的前提下增加 overlap 调度。

目标稳定状态：

```text
GPU1: stage1(microbatch i)
GPU0: stage0(microbatch i + 1)
```

实现形态：

- 把 decode batch 拆成有界 microbatch；
- 用显式队列驱动 stage0 和 stage1；
- 使用 CUDA events 或等价 backend 同步机制来约束 buffer handoff 顺序；
- collector 侧保持输出顺序；
- 在 artifact metadata 中暴露 queue depth、microbatch size、in-flight stage count 和每个 stage 的耗时。

本阶段验收标准：

- c=1、c=4、c=8、c=16、c=32 下没有请求输出污染。
- 没有重复或缺失 stream `[DONE]`。
- 并发 sequence 之间没有请求串扰。
- 队列不会无界增长。
- c=4、c=8、c=16 解码期间两张 GPU 都有持续利用率。

### 阶段 4：Scheduler 和 admission 调优

只有当 runtime 真的具备 batch 或 overlap 能力后，才调整该路径的服务默认值。

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

Admission 必须通过 typed config 或已文档化 CLI/config 产品化，不能依赖隐藏 env。

## 正确性 Gate

正确性 gate 必须先通过，性能测量才能算证据。

每个 release-quality artifact 至少必须覆盖：

- `ferrum run` 单轮正确性；
- `ferrum run` 多轮正确性；
- `ferrum serve` 单轮正确性；
- `ferrum serve` 多轮正确性；
- OpenAI-compatible streaming，且正好一个 `data: [DONE]`；
- streaming usage，且请求包含 `stream_options.include_usage=true`；
- tool calling；
- structured output；
- 适用时使用确定性的 diagnostic settings；
- 日志扫描 panic、CUDA error、OOM、`<unk>`、`[PAD]`、mojibake、duplicate `[DONE]`、missing `[DONE]`、malformed SSE JSON 和 silent fallback。

如果正确性失败，同一次 run 的性能 artifact 只能算 diagnostic。

## 性能 Gate

唯一规范 HTTP benchmark 客户端仍然是 `ferrum bench-serve`。

必需 benchmark 形态：

```bash
ferrum bench-serve ... \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --n-repeats 3 \
  --concurrency-sweep 1,4,8,16,32
```

最终性能 artifact 必须包含：

- 同硬件 Ferrum baseline artifact；
- 同硬件 Ferrum candidate artifact；
- 如果采集了同 pod vLLM baseline，也必须包含该 artifact；
- 每个并发的输出吞吐；
- TTFT、TPOT、E2E latency、completed、failed、errored、bad output counts；
- `n_repeats=3` 的置信区间数据；
- tokenizer 实际计数的 input length；
- output token count source；
- effective runtime config；
- 二进制 SHA256；
- git SHA 和 dirty status。

必达阈值：

```text
max(c4,c8,c16,c32 aggregate output throughput) >= 27.6 tok/s
```

Stretch 阈值：

```text
max(c4,c8,c16,c32 aggregate output throughput) >= 33.0 tok/s
```

如果采集了同 pod vLLM baseline，则用下面的目标替代固定必达阈值：

```text
max(c4,c8,c16,c32 Ferrum output throughput) >= 0.80 * same_pod_vllm_output_tps
```

最终报告必须说明本目标通过的是固定公开下限、同 pod vLLM 80% 目标，还是两者都通过。

## 新增目标 Gate

为本目标新增最终验证器，例如：

```bash
python3 scripts/release/layer_split_perf_goal_gate.py \
  --out <out_dir> \
  --baseline-artifact <baseline_out> \
  --candidate-artifact <candidate_out> \
  --correctness-artifact <correctness_out> \
  --optional-vllm-artifact <vllm_out>
```

必需最终 PASS line：

```text
LAYER_SPLIT_PERF GOAL PASS: <out_dir>
```

验证器必须拒绝：

- dirty 或缺失的 git metadata；
- available 时缺失二进制 SHA256；
- 缺失 effective config；
- 缺失 selected layer split plan；
- 缺失 correctness artifact；
- 缺失 usage token counts；
- 期望 `--fail-on-error` 时出现任何 failed request；
- release-quality performance evidence 中 `n_repeats < 3`；
- 输出吞吐低于选定目标；
- artifact 标记为 diagnostic-only。

## 付费 GPU Lane

开始任何付费 GPU run 前，必须先说明 lane、预计 runtime/cost、stop condition、correctness gate 和 performance command。

建议 lane：

- `layer-split-perf-smoke`
  - 预计 runtime/cost：2x4090 host 上 30-60 分钟，优先选择约 1 USD/hour 的机器。
  - Stop condition：任何正确性失败、模型加载失败、CUDA OOM，或一次 candidate run 后吞吐仍然保持平线。
  - Correctness gate：product run/serve smoke 加 stream usage。
  - Performance command：一次 `bench-serve` sweep，带 `--fail-on-error --seed 9271`；如果跳过 `--require-ci` 或 `n_repeats=3`，只能算 diagnostic。
- `layer-split-perf-full`
  - 预计 runtime/cost：2x4090 host 上 1-3 小时，优先选择约 1 USD/hour 的机器。
  - Stop condition：最终 PASS、正确性失败，或目标 miss 但已有足够 profiling 解释 miss。
  - Correctness gate：完整目标正确性矩阵。
  - Performance command：必需 `bench-serve` 命令，包含 `--fail-on-error --require-ci --seed 9271 --n-repeats 3`。

证据采集完成后，不允许让付费 host 闲置。

## Checkpoint 和提交

按 checkpoint 做小提交：

1. Goal 文档和外部基准说明。
2. Profiling 和 artifact schema。
3. 批量感知 layer split 正确性。
4. 设备驻留 stage bridge。
5. Pipeline overlap。
6. Goal gate 和最终报告。

不要把原始 benchmark artifact 目录提交进 repo。大型或临时记录放在仓库外，例如：

```text
../ferrum-infer-rs-records-<YYYYMMDD>/
```

## 待确认问题

- 实现前是否先采集 Qwen2.5-72B-GPTQ 的同 pod 当前 vLLM baseline，还是先使用 MLC 34.5 tok/s 70B 级别公开下限？
- 第一个可接受收益应指定 c=4、c=8、c=16，还是使用 `max(c4,c8,c16,c32)`？
- CUDA backend 是否能干净暴露 peer-copy capability 作为产品默认，还是 device bridge 应先作为显式文档化 experimental mode？
- `llm_executor` 当前 model lock 是否兼容 overlapped stage workers，还是需要缩小 model-forward critical section？
