# 后端运行时预设与快速回归目标

## 状态

草案目标文件。

本目标不能因为“代码改完了”或“单个 smoke 过了”就宣称完成。只有最终验证器打印下面这一行，才算完成：

```text
BACKEND RUNTIME PRESET GOAL PASS: <out_dir>
```

## 目标

提升 Ferrum 的快速迭代能力，避免 Metal、CUDA、run、serve、性能 gate、release 发布状态相互影响且难以定位。

本目标覆盖两类问题：

- 真实线上/本地失败场景必须固化成可执行回归用例。
- Metal 和 CUDA 的差异必须集中到后端实现、运行时 resolver 和 gate 里，不能散落在 shared product path。

## 背景问题

本次会话暴露出以下工程问题，后续必须通过代码和脚本机制解决。

### P1. Release/gate 入口分散

当前 gate 分散在：

- `scripts/release/g0_source_gate.sh`
- `scripts/release/release_binary_gate.py`
- `scripts/release/g0_release_summary.py`
- 各种单独 Python gate
- 旧的 `scripts/release.sh`

其中旧 `scripts/release.sh` 还残留过期 crate 名，不能作为可信发布入口。

需要收敛成统一入口，例如：

```text
scripts/release/run_gate.py
```

输入 lane，输出统一 manifest、artifact dir 和 PASS line。目标是以后不再靠人工记“还差 Cargo、Homebrew、CUDA、Metal、GitHub Release 哪一步”。

### P2. 真实失败场景没有全部变成固定回归

这次暴露的问题包括：

- 多轮截断。
- 孤立或空的 `</think>` 输出。
- 并发错误和请求串扰。
- stream 缺失、重复或格式错误的 `[DONE]`。
- tool calling / structured output 回归。
- context shift / max token / KV capacity 退化。
- Metal c16 性能异常。
- 首 token 等待期间没有提示或提示体验差。

这些必须进入统一 scenario DSL，`ferrum run` 和 `ferrum serve` 共用，不要散落在临时报告里。

目标是：用户或 Claude 测出的问题，本地脚本可以独立复现。

### P3. 运行默认值依赖 env 堆叠

现在仍有不少 `FERRUM_*` 在 `runtime_env.rs`、`axum_server.rs` 等路径影响产品行为。

调试可以使用 env，但产品默认不能依赖隐藏 env。

需要明确的 typed preset：

- `RuntimePreset`
- `BackendPreset`
- `ModelCapabilities`

Metal、CUDA、MoE、Dense、GGUF、HF 路径必须 resolve 出最终 scheduler、KV、batch、cache、graph、FA2、attention、chat template 状态，并写入 artifact。

### P4. 并发/admission 没有产品化

Metal 并发问题说明用户不能手动猜并发。

Server 必须根据 backend、model、KV capacity、max model length 和内存情况自动决定 admission，并暴露运行状态：

- `effective_max_concurrent`
- queue depth
- active prefill
- active decode
- batch size
- rejected count
- failed count

`bench-serve --concurrency` 是压测客户端参数，不应该成为服务端正确性的前置条件。

### P5. Chat template / capability 不能按模型名判断

模型是否支持 thinking、tool calling、structured output、chat template kwargs、stop tokens，必须来自模型 metadata、chat template 或显式配置。

禁止通过模型名字符串硬编码产品行为。

### P6. `bench-serve` 要继续强化为唯一 HTTP 性能入口

`crates/ferrum-cli/src/commands/bench_serve.rs` 已经有：

- `--fail-on-error`
- `--require-ci`
- `--seed`
- usage token 统计

下一步必须把坏输出扫描和质量断言接进去：

- `<unk>`
- `[PAD]`
- 乱码 / mojibake
- duplicate `[DONE]`
- missing `[DONE]`
- malformed SSE JSON
- 0 output tokens
- stream bulk flush
- HTTP 500
- panic

## 非目标

- 不把 Ferrum 拆成 Metal engine 一套、CUDA engine 一套。
- 不新增扁平的 `BackendCapabilities { supports_xxx: bool }` 产品 API。
- 不用隐藏环境变量作为正式产品默认行为。
- 不在没有同硬件 artifact 的情况下做性能宣称。
- 不在本目标里顺手重写 kernel，除非 gate 证明 runtime preset 暴露出 kernel-local 缺陷。

## 架构结论

结合 Ferrum 当前架构和主流推理引擎做法，正确方向不是完全拆分 Metal/CUDA 代码，也不是做一张扁平 bool capability 表。

目标架构分三层。

### L1. Kernel Trait Capability

编译期后端能力继续通过 trait 和具体 impl 表达，例如：

- `BackendPagedKv`
- `BackendGraph`
- `BackendQuantMarlin`
- `BackendQuantGguf`
- `BackendMoeFused`
- `BackendKvDtype<K>`

这一层只回答：某个 backend 类型是否实现某类 kernel API。

这一层不能承载产品策略，例如目标并发、stream usage、release 性能阈值。

### L2. Runtime Availability

运行时可用性回答：在当前 binary、device、model、shape、dtype、build feature、config 下，某条候选路径是否可用。

候选路径必须带状态和原因，不能只是 bool。

必需状态：

- `available`
- `selected`
- `disabled_by_config`
- `unsupported_backend`
- `missing_build_feature`
- `unsupported_model_arch`
- `unsupported_shape`
- `diagnostic_only`

示例：

- FA2 可能由 CUDA 实现，但因为没编译 `fa2-source`、shape 不支持或模型 layout 不匹配而不可用。
- CUDA graph 可能对某个 decode path 可用，但对另一个 path 只能是 diagnostic-only。
- Metal MoE batched decode 可能有 kernel，但没有通过正确性和性能 gate 前不能进入产品默认。

### L3. Resolved Runtime Preset

resolved preset 是产品代码消费的唯一事实来源。

它必须记录最终选中的值：

- backend
- model architecture
- model format
- selected scheduler
- selected KV layout
- selected KV dtype
- selected KV capacity
- selected max model length
- selected max sequences
- selected max batched tokens
- selected attention implementation
- selected graph mode
- selected MoE decode path
- selected prefix/session cache mode
- chat template source
- model capabilities
- runtime overrides
- 所有候选 availability 及其未选中原因

`ferrum run` 和 `ferrum serve` 必须使用同一个 resolver，并输出兼容的 effective config JSON。

## 设计规则

- Shared product path 只读取 resolved preset，不直接通过 `cuda` 或 `metal` 字符串推断产品行为。
- Backend-local 代码可以基于具体 backend 类型特化。
- Registry 代码可以实例化 `MetalBackend`、`CudaBackend` 或 `CpuBackend`。
- Auto-config / resolver 可以读取 hardware、model metadata 和 build feature。
- Release evidence 如果依赖未文档化 env，只能算 diagnostic，不能算 release evidence。
- 请求某条路径时，如果实际 silent fallback 到另一条路径，属于 release blocker。

允许出现 backend-specific 判断的位置：

- `crates/ferrum-kernels/src/backend/` 下的 backend trait impl。
- backend 注册和实例化。
- runtime preset resolver。
- hardware discovery。
- 专门选择 backend lane 的测试和 release 脚本。

其他位置新增 backend 字符串判断时，必须进入 allowlist，并说明理由。

## 交付项与量化验收

### D1. 统一 release/gate runner

新增统一 gate 入口，例如：

```bash
python3 scripts/release/run_gate.py <lane> --out <out_dir>
```

必需 lane：

- `unit`
- `metal`
- `cuda-smoke`
- `cuda-full`
- `cuda-llama-dense`
- `metal-tarball`
- `cuda-tarball`
- `homebrew-metal`
- `homebrew-cuda-fetch`
- `release-summary`
- `release-complete`

验收标准：

- 每个 lane 都输出一个 `gate.manifest.json`。
- 每个 manifest 都包含：
  - lane
  - status
  - command line
  - git SHA
  - dirty status
  - artifact dir
  - started_at
  - finished_at
  - duration_sec
  - binary sha256，若该 lane 使用 binary
  - model path / model id，若该 lane 加载模型
  - sanitized env summary
  - PASS line
- 所有 lane 的 PASS line 都由 `run_gate.py` 统一打印。
- 旧 `scripts/release.sh` 不能再作为 release source of truth；要么删除，要么改成明确失败的 compatibility wrapper。
- `python3 scripts/release/run_gate.py --list-lanes` 能列出所有 lane。

必需 PASS line：

```text
FERRUM GATE <lane> PASS: <out_dir>
```

### D2. 统一真实场景回归 runner

新增 scenario DSL 和 runner，统一覆盖 `run` 与 `serve`。

建议入口：

```bash
python3 scripts/release/run_scenarios.py \
  --manifest scripts/release/scenarios/product_regression.json \
  --out <out_dir>
```

必需场景：

- 并发质量：c1、c4、c16、c32。
- 多轮 recall。
- 多轮截断检测。
- context shift / KV capacity 检测。
- 孤立 `</think>` 检测。
- stream `[DONE]` 检测。
- tool calling。
- structured output。
- first-token UX。

验收标准：

- 每个 scenario 输出独立 JSON artifact。
- runner 输出一个 summary JSON。
- 任一 scenario fail，runner 返回非零。
- 同一份 scenario manifest 至少支持：
  - `ferrum run`
  - `ferrum serve`
  - streaming serve
  - Python/OpenAI SDK 兼容检查，若环境具备 SDK
- 用户或 Claude 发现的上述已知问题，必须能由该 runner 独立测出。

量化阈值：

- 并发 cells 必须包含 `1,4,16,32`。
- 每个并发 cell：
  - HTTP 200 数等于请求数。
  - `json_ok == requests`。
  - exact marker 命中数等于请求数。
  - exact checksum 命中数等于请求数。
  - `crosstalk == 0`。
  - `length_finishes == 0`。
  - `forbidden_count == 0`。
- 多轮 `run`：
  - 至少 2 个 assistant turn。
  - 第二轮必须精确召回第一轮 secret。
  - return code 为 0。
  - 无 forbidden decoded text。
- 多轮 `serve`：
  - HTTP status 200。
  - strip 完整 `<think>...</think>` 后包含 expected secret。
  - `finish_reason != "length"`。
- Stateful loop：
  - 至少 5 个 user turn。
  - 至少 4 个非空 assistant response。
  - `length_finishes == 0`。
  - `repeated_prefixes == 0`。
  - 不允许任意一行只有 `</think>`。
- SSE stream：
  - `data: [DONE]` 正好 1 个。
  - 至少 1 个非空 content delta。
  - `stream_options.include_usage=true` 时 usage chunk 正好 1 个。
- First-token UX：
  - 交互式 TTY 输入 prompt 后，首个 decoded token 前 1000 ms 内必须出现进度提示。
  - 进度提示必须只有 1 行。
  - 除 elapsed time 外，可见提示文本长度不超过 32 个字符。
  - 不允许包含 `waiting for first token`。
  - assistant 文本打印前必须清除提示。

必需 PASS line：

```text
BACKEND REGRESSION SMOKE PASS: <out_dir>
```

### D3. Typed RuntimePreset / BackendPreset

新增或重构一个统一 resolver，产出 typed preset。

验收标准：

- `ferrum serve --effective-config-json <path>` 写出完整 preset schema。
- `ferrum run --effective-config-json <path>` 写出相同 schema。
- 同一 model/backend/config 输入下，`run` 与 `serve` 下列字段必须一致：
  - selected scheduler
  - selected KV layout
  - selected KV dtype
  - selected KV capacity
  - selected max model length
  - selected max sequences
  - selected max batched tokens
  - selected attention implementation
  - selected graph mode
  - selected MoE decode path
  - chat template source
- effective config 必须列出所有影响产品默认的 env override：
  - key
  - sanitized value
  - source
  - 是否改变产品默认
- release gate 不允许依赖未文档化 env。

必需 candidate groups：

- `attention_impls`
- `graph_modes`
- `kv_layouts`
- `kv_dtypes`
- `moe_decode_paths`
- `cache_modes`

量化阈值：

- 每个适用 group 正好有 1 个 `selected`。
- 每个非 selected candidate 必须有非空 reason。
- 如果适用 group 没有 selected candidate，resolver 必须在模型启动前返回结构化错误。

### D4. 并发/admission 产品化

Server 必须自动决定 admission，而不是要求用户猜并发。

验收标准：

- `/health` 或 `/metrics` 暴露：
  - `effective_max_concurrent`
  - `queue_depth`
  - `active_prefill`
  - `active_decode`
  - `current_batch_size`
  - `rejected_requests_total`
  - `failed_requests_total`
  - `completed_requests_total`
- effective config 记录 admission 的选择依据：
  - backend
  - model architecture
  - KV capacity
  - max model length
  - max sequences
  - max batched tokens
  - memory estimate
- 服务端默认 admission 能覆盖 release gate 需要的并发 cell。
- `bench-serve --concurrency` 只作为客户端压测参数，不作为服务端正确性的配置前置。

量化阈值：

- Metal README gate 的每个 c16 cell，serve effective max sequences 必须 `>= 16`。
- CUDA full gate 的最大并发 cell，serve effective max sequences 必须 `>= 32`。
- 并发 quality gate 里 rejected requests 必须为 0，除非 scenario 明确测试 admission rejection。
- 常规 release scenario 中 HTTP 500 必须为 0。

### D5. ModelCapabilities 走模型元数据

新增统一 `ModelCapabilities` 解析。

能力至少包含：

- supports thinking
- supports tool calling
- supports structured output
- supports chat template kwargs
- stop tokens
- template source
- tokenizer special tokens

来源优先级：

1. GGUF metadata / tokenizer metadata。
2. HF `tokenizer_config.json` / `chat_template`。
3. 显式用户配置。
4. documented fallback。

验收标准：

- 不允许通过模型名字符串硬编码 thinking/tool/structured 行为。
- 模型名不含 `qwen3` 时，Qwen3 相关能力仍能从 metadata/template 解析。
- `run` 和 `serve` 使用同一份 chat template 渲染逻辑。
- effective config 写出 capability 来源。
- capability 解析失败时必须写出 fallback reason。

量化阈值：

- 至少覆盖 4 个 snapshot：
  - Metal + Llama 8B-class dense GGUF。
  - Metal + Qwen3-30B-A3B MoE GGUF。
  - CUDA + Llama 8B-class dense。
  - CUDA + Qwen3-30B-A3B MoE/GPTQ。
- 每个 snapshot 必须包含 `ModelCapabilities` 和 chat template source。
- snapshot 变更时测试输出 exact field path 和 old/new value。

### D6. 强化 `bench-serve` 为唯一 HTTP 性能入口

`bench-serve` 必须同时输出性能和质量证据。

验收标准：

- Release 性能测量统一使用 `ferrum bench-serve`。
- 不新增第二套 HTTP throughput release path。
- `bench-serve --fail-on-error` 遇到任意请求错误必须非零退出。
- `bench-serve --require-ci` 要求 `n_repeats >= 3`，否则非零退出。
- streaming 请求必须发送 `stream_options.include_usage=true`。
- streaming 成功定义：
  - 正好 1 个 `[DONE]`。
  - 至少 1 个 output token。
  - 无 stream error。
  - 无 malformed SSE JSON。
- release artifact 必须要求 `output_token_count_source == "usage"`。

新增坏输出扫描：

- `<unk>`
- `[PAD]`
- reserved special token
- invalid UTF-8 / mojibake
- duplicate `[DONE]`
- missing `[DONE]`
- malformed SSE JSON
- 0 output tokens
- stream bulk flush
- HTTP 500
- panic

量化阈值：

- release performance report 中每个 run：
  - completed requests > 0。
  - errored requests == 0。
  - bad output count == 0。
  - malformed stream count == 0。
  - output token source 为 `usage`。
  - `n_repeats >= 3`。

### D7. Metal README 性能 gate

Metal README 性能是 hard gate，不是 diagnostic。

验收标准：

- Metal gate 至少覆盖：
  - Llama 8B-class dense model。
  - Qwen3-30B-A3B MoE model。
  - 所有 README 宣传的 c16 行。
- 每个 Metal performance cell：
  - `completed == prompts`。
  - `failed == 0`。
  - output throughput > 0。
  - `ratio_to_readme >= 0.90`。
  - `not_regressed_90pct == true`。
- 如果 Qwen3-30B-A3B c16 实测低于 README baseline 的 90%，validator 必须 fail。

必需 PASS line：

```text
METAL README GATE PASS: <out_dir>
```

### D8. CUDA 双架构 release 覆盖

CUDA release 不能只测 Qwen3-30B-A3B MoE/GPTQ，也必须测 Llama 8B-class dense。

验收标准：

- CUDA smoke 至少覆盖 c1 和 c32。
- CUDA full 覆盖 c1、c4、c16、c32。
- CUDA dense supplemental gate 覆盖 Llama 8B-class dense，并包含：
  - `ferrum run` 多轮正确性。
  - `ferrum serve` 多轮正确性。
  - stream 正好 1 个 `[DONE]`。
  - `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`。
- 每个 release performance report：
  - `n_repeats >= 3`。
  - completed requests > 0。
  - errored requests == 0。
  - `output_token_count_source == "usage"`。

必需 PASS lines：

```text
G0 SOURCE g0_cuda4090_smoke PASS: <out_root>
G0 SOURCE g0_cuda4090_full PASS: <out_root>
G0 SOURCE g0_cuda4090_llama_dense PASS: <out_root>
```

### D9. 后端边界审计

新增 cheap audit，防止 backend-specific 产品决策扩散到 shared path。

验收标准：

- 审计输出所有不在允许位置的 `cuda` / `metal` 直接判断。
- 新增未批准 backend 判断时审计 fail。
- 有 checked-in allowlist，包含：
  - file path
  - reason
  - owner
  - expiry 或 review condition
- 至少扫描这些模式：
  - `eq_ignore_ascii_case("cuda")`
  - `eq_ignore_ascii_case("metal")`
  - `== "cuda"`
  - `== "metal"`
  - `is_cuda_backend(`
  - `is_metal_backend(`
  - `B::is_cuda_backend(`
  - `B::is_metal_backend(`

必需 PASS line：

```text
BACKEND BOUNDARY AUDIT PASS: <out_dir>
```

### D10. Backend preset snapshot

新增不加载完整权重的 preset snapshot test，用来捕捉 Metal/CUDA 互相影响。

必需 snapshot cases：

- Metal + Llama 8B-class dense GGUF。
- Metal + Qwen3-30B-A3B MoE GGUF。
- CUDA + Llama 8B-class dense。
- CUDA + Qwen3-30B-A3B MoE/GPTQ。

验收标准：

- snapshot 输出包含所有 selected preset 字段和 availability groups。
- Metal-only 预期改动不得改变 CUDA snapshot。
- CUDA-only 预期改动不得改变 Metal snapshot。
- snapshot 变化时，测试输出 exact field path 和 old/new value。

必需 PASS line：

```text
BACKEND PRESET SNAPSHOT PASS: <out_dir>
```

### D11. Release completion manifest

新增 release completion manifest，避免 tag、GitHub Release、Homebrew、Cargo、G0 artifact 任一环节遗漏。

manifest 必须记录：

- git SHA。
- dirty status。
- tag。
- GitHub Release URL。
- release asset 名称和 SHA256。
- Metal source gate artifact。
- CUDA full source gate artifact。
- CUDA dense source gate artifact。
- Metal tarball gate artifact。
- CUDA tarball gate artifact。
- Homebrew Metal gate artifact。
- Homebrew CUDA fetch gate artifact。
- Cargo workspace crates 在 crates.io 上的可见版本。

验收标准：

- 任意 workspace crate 未在 crates.io 可见目标版本，release completion validator 必须 fail。
- GitHub Release 缺 asset 或 SHA256 不匹配时必须 fail。
- Homebrew formula 版本或 SHA256 不匹配时必须 fail。

必需 PASS line：

```text
FERRUM RELEASE COMPLETION PASS: <out_dir>
```

## 最终目标 gate

本目标完成必须运行：

```bash
python3 scripts/release/backend_runtime_preset_goal_gate.py \
  --out <out_dir> \
  --metal-artifact <metal_gate_out> \
  --cuda-smoke-artifact <cuda_smoke_out> \
  --cuda-full-artifact <cuda_full_out> \
  --cuda-dense-artifact <cuda_dense_out>
```

该脚本必须验证：

- unit/source checks。
- unified gate runner selftest。
- scenario regression smoke。
- runtime preset schema。
- backend boundary audit。
- backend preset snapshots。
- Metal 整体回归。
- CUDA 整体回归。
- release completion manifest validator，若提供 release version。

### 最终整体验收回归矩阵

本目标涉及 runtime resolver、默认值、admission、chat template、capability、bench 和 release gate。

因此目标完成时，不能只跑 source/unit 或单后端 smoke。必须完整回归 Metal 和 CUDA。

#### Metal 最终回归

必须运行：

```bash
scripts/release/g0_source_gate.sh metal <out_root>
```

必需 PASS line：

```text
G0 SOURCE metal PASS: <out_root>
```

Metal 回归必须覆盖并在 artifact 中证明：

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

Metal 性能量化阈值：

- 每个 performance cell `completed == prompts`。
- 每个 performance cell `failed == 0`。
- 每个 performance cell `ratio_to_readme >= 0.90`。
- 每个 performance cell `not_regressed_90pct == true`。
- c16 宣传行必须全部存在且通过。

#### CUDA 最终回归

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

CUDA 回归必须覆盖并在 artifact 中证明：

- `ferrum run` 正确性。
- `ferrum run` 多轮对话。
- `ferrum serve` 正确性。
- `ferrum serve` 多轮对话。
- streaming，且 `[DONE]` 正好 1 个。
- tool calling。
- structured output。
- context shift / max token / KV capacity 行为。
- 并发质量，至少 c1、c4、c16、c32。
- `bench-serve --fail-on-error --require-ci --seed 9271 --n-repeats 3`。
- Qwen3-30B-A3B MoE/GPTQ。
- Llama 8B-class dense。

CUDA 性能量化阈值：

- 每个 run `completed > 0`。
- 每个 run `errored == 0`。
- 每个 report `n_repeats >= 3`。
- 每个 report `output_token_count_source == "usage"`。
- 每个 required concurrency cell 都存在。
- 没有 HTTP 500、panic、malformed SSE、missing/duplicate `[DONE]`、0 output tokens、`<unk>`、`[PAD]` 或乱码。

#### 双后端共同验收

最终 goal gate 还必须证明：

- Metal-only 改动没有改变 CUDA preset snapshot，除非目标文件或 PR 明确声明并解释。
- CUDA-only 改动没有改变 Metal preset snapshot，除非目标文件或 PR 明确声明并解释。
- shared runtime/default/admission/template 代码改动后，Metal 和 CUDA 都完成上述整体回归。
- 所有最终回归 artifact 必须保存 command line、git SHA、dirty status、binary SHA256、model path/id、sanitized env、effective config 和 PASS line。

最终必须打印：

```text
BACKEND RUNTIME PRESET GOAL PASS: <out_dir>
```

## 阶段性开发 smoke gate

这一节不是完成标准，也不能用于 release-ready、goal-complete 或 performance-ready 声明。

它只用于大目标拆分开发时快速筛错，避免每个小改动都立刻跑完整 Metal/CUDA 全量回归。只要某个 PR 宣称完成本目标、改变 shared runtime/default/admission/template，或要进入正式发布判断，就必须跑上面的最终整体验收回归矩阵。

阶段性开发 PR 至少需要：

```bash
cargo test --workspace --all-targets
python3 -m py_compile scripts/release/*.py scripts/metal_readme_regression.py
bash -n scripts/release/g0_source_gate.sh
python3 scripts/release/backend_boundary_audit.py --out <out_dir>/backend-boundary
python3 scripts/release/backend_runtime_preset_snapshot.py --out <out_dir>/preset-snapshots
```

必需 PASS lines：

```text
BACKEND BOUNDARY AUDIT PASS: <out_dir>/backend-boundary
BACKEND PRESET SNAPSHOT PASS: <out_dir>/preset-snapshots
```

如果阶段性 PR 涉及用户可见行为，还必须追加最小 scenario smoke：

```bash
python3 scripts/release/run_scenarios.py \
  --manifest scripts/release/scenarios/product_regression_smoke.json \
  --out <out_dir>/scenario-smoke
```

必需 PASS line：

```text
BACKEND REGRESSION SMOKE PASS: <out_dir>/scenario-smoke
```

## 失败处理规则

gate 失败时：

- 保存失败 artifact directory。
- 不要在失败原因未明确前反复跑 expensive CUDA full sweep。
- 先添加或更新最小复现 scenario。
- 再修代码。
- 修完先跑最小失败 scenario。
- 然后跑对应 lane gate。
- 如果要宣称 release ready，再跑完整 release gate。
