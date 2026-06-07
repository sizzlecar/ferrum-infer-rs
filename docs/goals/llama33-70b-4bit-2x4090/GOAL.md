# Llama 3.3 70B 4bit 双 4090 高效推理目标

## 状态

草案目标文件。

本目标是 `docs/goals/backend-runtime-preset-fast-iteration/GOAL.md` 完成后的下一阶段目标。

本目标不能在前置目标未完成时宣称完成。只有最终验证器打印下面这一行，才算完成：

```text
LLAMA33_70B_4BIT_2X4090 GOAL PASS: <out_dir>
```

## 目标

在 2 张 RTX 4090 上让 Ferrum 可以高效推理 Llama 3.3 70B 4bit，并把该模型纳入 CUDA 正式回归矩阵。

目标包括：

- 支持 Llama 3.3 70B 4bit 在 2x4090 上通过产品路径加载和推理。
- `ferrum run` 和 `ferrum serve` 都必须可用。
- 正确性、多轮、流式、工具调用、结构化输出、并发质量和性能都必须通过 gate。
- Metal 仍然执行既有完整回归，防止 shared runtime/default/template/admission 改动破坏 Metal。
- CUDA 仍然执行既有全量回归，并新增 Llama 3.3 70B 4bit 双卡 lane。

## 前置条件

必须先完成：

```text
BACKEND RUNTIME PRESET GOAL PASS: <out_dir>
```

也就是前置目标已经完成：

- typed `RuntimePreset` / `BackendPreset`。
- runtime availability records。
- backend boundary audit。
- preset snapshot。
- scenario DSL。
- Metal/CUDA 最终整体验收回归矩阵。

如果前置目标未完成，本目标只能做调研或 diagnostic，不能宣称 release-ready 或 goal-complete。

## 非目标

- 不降低 Metal 或既有 CUDA G0 gate 标准。
- 不用隐藏环境变量作为正式产品默认。
- 不为了 70B 单模型破坏 Llama 8B dense 或 Qwen3-30B-A3B MoE/GPTQ 路径。
- 不新增第二套 HTTP 性能入口；HTTP 性能仍然使用 `ferrum bench-serve`。
- 不接受 silent CPU fallback、silent single-GPU fallback 或 silent non-4bit fallback。
- 不在没有同硬件 A/B artifact 的情况下做性能宣称。

## 模型与硬件

### 必需硬件

- GPU：2 张 NVIDIA RTX 4090。
- 每张显存：24GB。
- CUDA device count：必须等于 2。
- 目标 backend：CUDA。
- 双卡策略：layer split / 层切分。

### 必需模型

目标模型：

```text
Llama-3.3-70B-Instruct 4bit
```

允许的 4bit 格式：

- GPTQ Int4。
- AWQ Int4。
- GGUF Q4_K_M，仅当 Ferrum CUDA 路径明确支持该格式且不会 fallback。

每次 gate 必须记录：

- model id 或本地路径。
- quant format。
- tokenizer path。
- model config hash。
- weight file SHA256 或 manifest hash。
- chat template source。
- stop tokens。
- layer split plan。
- per-GPU memory usage。

## 架构要求

### A1. 双卡 CUDA layer split 不是临时 env 路径

2x4090 推理必须通过 typed preset 选择，不能靠未文档化 env 堆叠。

验收标准：

- `ferrum run` 支持通过产品参数显式指定多个 CUDA GPU，例如 `--gpu-devices 0,1` 或等价 typed config。
- `ferrum serve` 支持通过产品参数显式指定多个 CUDA GPU，例如 `--gpu-devices 0,1` 或等价 typed config。
- 多 GPU 参数必须进入 typed config / effective config，不能只依赖 `CUDA_VISIBLE_DEVICES`。
- effective config 写出：
  - `backend = cuda`
  - `requested_gpu_devices = [0, 1]`
  - `selected_gpu_devices = [0, 1]`
  - `cuda_device_count = 2`
  - `selected_distributed_strategy = layer_split`
  - `selected_layer_split_plan`
  - `selected_weight_placement`
  - `selected_kv_layout`
  - `selected_attention_impl`
  - `selected_graph_mode`
  - `selected_max_sequences`
  - `selected_max_model_len`
  - `selected_kv_capacity`
  - `selected_max_batched_tokens`
  - `model_capabilities`
- 如果请求双卡 layer split 但只使用 1 卡，gate 必须 fail。
- 如果请求的 GPU 列表和最终选择的 GPU 列表不一致，gate 必须 fail，除非 artifact 标记为 diagnostic 且不能作为完成证据。
- 如果任一 GPU 显存利用长期低于另一张 GPU 的 30%，gate 必须 fail，除非 artifact 标记为 diagnostic 且不能作为完成证据。

### A1.1 `run` / `serve` 多 GPU 参数

`ferrum run` 和 `ferrum serve` 必须支持同一套多 GPU 选择语义。

建议参数：

```bash
ferrum run <model> --backend cuda --gpu-devices 0,1
ferrum serve --model <model> --backend cuda --gpu-devices 0,1
```

验收标准：

- `--gpu-devices` 接受逗号分隔 GPU id。
- GPU id 必须是非负整数。
- 重复 GPU id 必须报错。
- 请求不存在的 GPU id 必须报错。
- 只指定 1 个 GPU 时，不允许选择 2 卡 layer split。
- 指定 2 个 GPU 时，70B 4bit lane 必须选择 layer split。
- `run` 与 `serve` 对同一 GPU 列表产出的 selected GPU 列表必须一致。
- effective config 必须记录：
  - raw CLI value。
  - parsed requested GPU list。
  - selected GPU list。
  - rejected GPU list 及原因，若有。
- 日志必须打印 selected GPU list 和 layer split plan。
- release gate 不允许只用 `CUDA_VISIBLE_DEVICES=0,1` 作为多 GPU 产品入口。

### A2. 模型能力来自 metadata/template

Llama 3.3 70B 的 chat template、tool calling、structured output、stop token、system message 行为必须来自模型 metadata、HF tokenizer config 或显式 documented config。

验收标准：

- 不允许通过模型名字符串硬编码 Llama 3.3 行为。
- effective config 记录 `ModelCapabilities` 和来源。
- `run` 与 `serve` 使用同一份 template 渲染逻辑。
- tool calling 和 structured output 的请求 shape 与 OpenAI-compatible server 共用。

### A3. 70B lane 不影响既有单卡 CUDA lane

新增 2x4090 lane 后，既有 1x4090 CUDA gate 仍必须保持原标准。

验收标准：

- `cuda-full` 仍使用 1x4090 Qwen3-30B-A3B MoE/GPTQ。
- `cuda-llama-dense` 仍使用 1x4090 Llama 8B-class dense。
- 新增 70B lane 不能修改既有 lane 的模型、并发 cell、阈值或 PASS line，除非单独 PR 明确说明。

## 新增 Gate

新增 CUDA 2x4090 70B lane：

```bash
scripts/release/g0_source_gate.sh cuda-llama33-70b-4bit-2x4090 <out_root>
```

必需 PASS line：

```text
G0 SOURCE g0_cuda2x4090_llama33_70b_4bit PASS: <out_root>
```

最终目标 gate：

```bash
python3 scripts/release/llama33_70b_4bit_2x4090_goal_gate.py \
  --out <out_dir> \
  --metal-artifact <metal_gate_out> \
  --cuda-full-artifact <cuda_full_out> \
  --cuda-llama-dense-artifact <cuda_llama_dense_out> \
  --cuda-llama33-70b-artifact <cuda_llama33_70b_out>
```

最终必需 PASS line：

```text
LLAMA33_70B_4BIT_2X4090 GOAL PASS: <out_dir>
```

## 最终验收矩阵

本目标完成时，必须同时完成 Metal 回归、既有 CUDA 全量回归和新增 Llama 3.3 70B 4bit 双卡回归。

### M1. Metal 回归

必须运行：

```bash
scripts/release/g0_source_gate.sh metal <out_root>
```

必需 PASS line：

```text
G0 SOURCE metal PASS: <out_root>
```

Metal artifact 必须证明：

- `ferrum run` 正确性。
- `ferrum run` 多轮对话。
- `ferrum serve` 正确性。
- `ferrum serve` 多轮对话。
- streaming，且 `[DONE]` 正好 1 个。
- tool calling。
- structured output。
- context shift / max token / KV capacity 行为。
- 并发质量，至少 c1、c4、c16。
- Metal README 性能 gate。
- Qwen3-30B-A3B MoE。
- Llama 8B-class dense。

Metal 性能阈值：

- 每个 performance cell `completed == prompts`。
- 每个 performance cell `failed == 0`。
- 每个 performance cell `ratio_to_readme >= 0.90`。
- 每个 performance cell `not_regressed_90pct == true`。

### C1. 既有 CUDA 全量回归

必须运行：

```bash
scripts/release/g0_source_gate.sh cuda-full <out_root>
scripts/release/g0_source_gate.sh cuda-llama-dense <out_root>
```

必需 PASS lines：

```text
G0 SOURCE g0_cuda4090_full PASS: <out_root>
G0 SOURCE g0_cuda4090_llama_dense PASS: <out_root>
```

既有 CUDA artifact 必须证明：

- Qwen3-30B-A3B MoE/GPTQ 在 1x4090 上通过 c1、c4、c16、c32。
- Llama 8B-class dense 在 1x4090 上通过 run、serve、streaming 和 bench-serve。
- `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`。
- completed requests > 0。
- errored requests == 0。
- `output_token_count_source == "usage"`。

### C2. Llama 3.3 70B 4bit 双卡 CUDA 回归

新增 lane 必须覆盖以下检查。

#### C2.1 启动与配置

验收标准：

- 2 张 RTX 4090 均被 runtime 发现。
- `run` 和 `serve` 均通过产品参数指定 GPU 列表。
- requested GPU devices 为 `[0, 1]`。
- selected GPU devices 为 `[0, 1]`。
- selected distributed strategy 为 `layer_split`。
- layer split plan 已写入 artifact。
- 两张 GPU 都有权重或 KV/activation 相关显存使用证据。
- health endpoint 返回 200。
- `/v1/models` 返回目标模型。
- effective config 和 decision trace 均写入 artifact。
- 日志中没有：
  - panic
  - CUDA illegal memory access
  - NCCL error
  - OOM
  - KV cache overflow
  - missing tokenizer
  - chat template render failure
  - CPU fallback
  - single-GPU fallback

#### C2.2 `ferrum run` 正确性

必须覆盖：

- 单轮数学/事实正确性。
- 多轮 recall。
- context shift / KV capacity 行为。
- 输出不含 `<unk>`、`[PAD]`、reserved special tokens 或乱码。

量化阈值：

- return code == 0。
- 至少 2 个 assistant turn。
- 第二轮必须精确召回第一轮 secret。
- `finish_reason != "length"`，如果 run 输出包含 finish reason。
- 每轮 assistant 输出非空。
- 孤立 `</think>` 行数为 0。

#### C2.3 `ferrum serve` 非流式正确性

必须覆盖：

- 单轮数学/事实正确性。
- 多轮 recall。
- structured output。
- tool calling。
- tool result follow-up。

量化阈值：

- 每个请求 HTTP status == 200。
- 每个请求 JSON parse 成功。
- 多轮 recall 精确包含 first-turn secret。
- structured output 能被 JSON parse。
- strict schema 请求必须满足 schema。
- required tool call 必须返回 `tool_calls`。
- tool call name 必须等于请求指定工具名。
- tool arguments 必须是合法 JSON。
- forbidden decoded text count == 0。
- HTTP 500 count == 0。

#### C2.4 Streaming 正确性

必须覆盖：

- 普通 streaming chat。
- streaming with `stream_options.include_usage=true`。
- streaming tool call。
- streaming structured output，若当前产品语义支持。

量化阈值：

- 每个 streaming 请求 HTTP status == 200。
- 每个 response 正好 1 个 `data: [DONE]`。
- 至少 1 个非空 content delta，tool-only response 除外。
- usage chunk 正好 1 个，若请求 `include_usage=true`。
- malformed SSE JSON count == 0。
- duplicate `[DONE]` count == 0。
- missing `[DONE]` count == 0。
- stream error count == 0。

#### C2.5 并发质量

必须使用 `run_scenarios.py` 或等价 scenario DSL。

必需并发 cells：

- c1
- c4
- c8
- c16

c32 是 stretch cell。若 c32 因显存或 admission 被拒绝，可以作为 diagnostic，但不能影响 c1/c4/c8/c16 的完成判断。

量化阈值：

- 每个 required cell HTTP 200 数等于请求数。
- `json_ok == requests`。
- exact marker 命中数等于请求数。
- exact checksum 命中数等于请求数。
- `crosstalk == 0`。
- `length_finishes == 0`。
- `forbidden_count == 0`。
- `rejected_requests == 0`，除非该 scenario 明确测试 admission rejection。
- `failed_requests == 0`。

#### C2.6 性能

性能必须使用 `ferrum bench-serve`，并保存同硬件 A/B artifact。

必需命令形态：

```bash
ferrum bench-serve \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --n-repeats 3 \
  --concurrency-sweep 1,4,8,16 \
  ...
```

必须额外保存同一台 2x4090 机器上的 vLLM baseline，使用同模型、同 tokenizer、同量化格式、同输入/输出长度、同并发 cells。

性能量化阈值：

- 每个 required cell 都有 Ferrum report。
- 每个 required cell 都有 vLLM baseline report。
- Ferrum 每个 run completed > 0。
- Ferrum 每个 run errored == 0。
- Ferrum 每个 report `n_repeats >= 3`。
- Ferrum 每个 report `output_token_count_source == "usage"`。
- Ferrum 每个 required cell 平均 output throughput >= 同硬件 vLLM baseline 的 70%。
- Ferrum 每个 required cell 平均 TTFT <= 同硬件 vLLM baseline 的 150%。
- Ferrum 每个 required cell 平均 TPOT <= 同硬件 vLLM baseline 的 150%。
- p95 end-to-end latency 必须为有限正数。
- bad output count == 0。
- malformed stream count == 0。

如果 vLLM 无法在同一模型/量化格式/2x4090 上成功跑完 baseline，则该性能结果只能标记为 diagnostic，不能作为本目标完成证据。

## Artifact 要求

新增 70B lane artifact 必须包含：

- `gate.json`
- `metadata.json`
- `effective_config.json`
- `decision_trace.jsonl`
- `hardware.json`
- `nvidia-smi.before.txt`
- `nvidia-smi.during.txt`
- `nvidia-smi.after.txt`
- `model_manifest.json`
- `run.command.json`
- `run.effective_config.json`
- `run.stdin`
- `run.stdout`
- `run.stderr`
- `serve.command.json`
- `serve.effective_config.json`
- `serve.log`
- `serve.health.json`
- `serve.models.json`
- `serve.correctness.json`
- `serve.multiturn.json`
- `serve.structured_output.json`
- `serve.tool_call.json`
- `serve.streaming.sse`
- `concurrency_quality_regression.json`
- `bench-serve.command.json`
- `bench-serve.json`
- `bench-serve.stdout`
- `bench-serve.stderr`
- `vllm-baseline.command.json`
- `vllm-baseline.json`
- `comparison.json`

`metadata.json` 必须包含：

- git SHA。
- dirty status。
- binary SHA256。
- command line。
- build features。
- CUDA version。
- driver version。
- GPU names。
- GPU UUIDs。
- requested GPU devices。
- selected GPU devices。
- model id/path。
- quant format。
- distributed strategy。
- layer split plan。
- sanitized env。

## GPU 成本规则

开始任何 2x4090 paid GPU 前，必须写明：

- lane。
- 预计运行时长。
- 预计成本。
- stop condition。
- 正确性 gate。
- 性能命令。

失败后不要反复跑 full sweep。必须先保存失败 artifact，定位失败模式，再决定下一次 targeted run。

## 完成标准

本目标完成必须同时满足：

- 前置目标已完成并有 `BACKEND RUNTIME PRESET GOAL PASS`。
- Metal 最终回归通过。
- 既有 CUDA full 回归通过。
- 既有 CUDA Llama dense 回归通过。
- 新增 Llama 3.3 70B 4bit 2x4090 回归通过。
- 所有 artifact 都包含 required metadata。
- 没有 hidden-env-only product behavior。
- 没有 silent fallback。
- 最终验证器打印：

```text
LLAMA33_70B_4BIT_2X4090 GOAL PASS: <out_dir>
```

## 阶段性开发 smoke

这一节不是完成标准。

开发期间可以先跑较小 smoke：

```bash
cargo test --workspace --all-targets
python3 -m py_compile scripts/release/*.py scripts/metal_readme_regression.py
python3 scripts/release/backend_boundary_audit.py --out <out_dir>/backend-boundary
python3 scripts/release/backend_runtime_preset_snapshot.py --out <out_dir>/preset-snapshots
```

如果改动涉及 CUDA 双卡 runtime、layer split、KV、attention、scheduler、serve 或 run，至少还需要一个 2x4090 targeted smoke：

```bash
scripts/release/g0_source_gate.sh cuda-llama33-70b-4bit-2x4090-smoke <out_root>
```

smoke 只允许覆盖 c1 和 c4，不允许作为最终完成证据。
