# Release Regression Hardening Goal

## 状态

草案目标文件，创建于 2026-06-28。

本目标用于解决四类反复出现的问题：

- 资源所有权不清导致 OOM、slot 泄漏、defer/rollback 不一致。
- CUDA 与 Metal 相互影响，优化一端改坏另一端。
- 问题发现太晚，经常到 release 前由人工手测暴露。
- release 回归像乒乓球，CUDA 修完回归 Metal，Metal 修完又回归 CUDA。

本目标不能因为“代码改完了”“某个 smoke 过了”就宣称完成。只有最终验证器打印下面这一行，才算完成：

```text
RELEASE_REGRESSION_HARDENING GOAL PASS: <out_dir>
```

最终验证器计划落地为：

```text
scripts/release/release_regression_hardening_goal_gate.py
```

在该验证器存在并打印精确 PASS line 之前，本目标保持 open。

## 总目标

把 release 前人工发现问题的模式，改成提交前和 release 候选阶段自动发现问题。

量化目标：

| 指标 | 当前问题 | 目标 |
|---|---|---|
| 资源类故障 | OOM/slot/KV/recurrent 依赖人工日志定位 | 资源事务 invariant gate 覆盖所有 allocate/defer/release/rollback 路径，零 underflow、零 leak、零 silent OOM |
| CUDA/Metal 互相影响 | 靠人工判断是否需要互相回归 | 100% `crates/`、`scripts/release/` 变更被 impact classifier 归类，自动生成 gate plan |
| 晚发现 | release 前人工 smoke 才发现 | full release gate 前必须先过 cheap sentinel，覆盖 `ferrum run` 和 `ferrum serve` |
| 乒乓回归 | gate 状态没有显式 invalidation | release candidate manifest 记录每个 lane 的 SHA、dirty、impact domain、失效原因 |
| 新模型接入质量 | 模型支持靠临场判断和补丁 | 每个新增模型必须有 model onboarding contract，缺合同不得写入支持矩阵 |

## 非目标

- 不在本目标里重写 CUDA/Metal kernel。
- 不把 Ferrum 拆成 CUDA engine 和 Metal engine 两套产品路径。
- 不用隐藏环境变量作为产品验收证据。
- 不把 cheap sentinel 当作 full release performance evidence。
- 不用 live vLLM 跑分作为本目标必需项；vLLM 只作为源码/历史行为参考，除非另立性能目标。
- 不在本目标里扩大付费 GPU 矩阵；需要 CUDA 运行时，优先 cheap smoke，full sweep 仍按现有 G0/release 规则执行。

## 阶段 0: 资源所有权与事务 invariant

### 目标

所有运行时资源必须由统一事务记录 allocate、defer、commit、rollback、release。

覆盖资源类型：

- KV cache blocks。
- recurrent-state slots。
- scheduler/admission capacity。
- model/session cache 引用。
- prefill/decode batch 内的临时 backend workspace。

### 必须交付

1. 一个资源事务记录结构，能够在 artifact 中输出：
   - `request_id`
   - `sequence_id`
   - `resource_kind`
   - `requested`
   - `reserved`
   - `committed`
   - `released`
   - `deferred`
   - `rollback_reason`
   - `owner_component`
2. 一个 invariant checker，至少检查：
   - `released <= committed`
   - `committed <= reserved`
   - `reserved <= capacity`
   - request 结束后该 request 的资源净占用为 0
   - defer 后不能留下 committed 资源
   - rollback 必须释放本事务内所有 reserved/committed 资源
   - OOM/admission reject 必须是显式状态，不能 panic 或 CUDA OOM 后才失败
3. 覆盖以下最小测试场景：
   - 单请求成功完成。
   - `ferrum run` 多轮成功完成。
   - `ferrum serve` streaming 成功完成。
   - admission defer 后再成功。
   - admission reject。
   - prefill 中途失败。
   - decode 中途失败。
   - client disconnect。
   - KV capacity 不足。
   - recurrent-state slot 不足。
   - batch 内部分 request 成功、部分失败。
   - shutdown/cancel 清理。

### 阶段验收

必须有阶段验证器：

```text
scripts/release/resource_invariant_gate.py
```

必需 PASS line：

```text
RESOURCE INVARIANT GATE PASS: <out_dir>
```

阶段 artifact 必须包含：

- `resource_trace.jsonl`
- `invariant_report.json`
- `gate.manifest.json`
- 触发的测试命令、git SHA、dirty status
- 每类资源的 `capacity/reserved/committed/released/leaked` 汇总

量化通过标准：

- 12/12 最小场景通过。
- `leaked_resources == 0`。
- `underflow_count == 0`。
- `silent_oom_count == 0`。
- `panic_count == 0`。
- 所有失败场景都有显式 error kind。

## 阶段 1: Change-impact classifier 与 gate planner

### 目标

每次代码变化都必须自动归类影响域，并生成 gate plan。不能再靠人工猜“这次要不要回归 Metal/CUDA”。

### 影响域

至少支持以下 domain：

| Domain | 例子 | 最小 gate 规则 |
|---|---|---|
| `cuda_backend` | CUDA kernels、CUDA runner、CUDA feature gate | unit + backend boundary + preset snapshot + CUDA smoke |
| `metal_backend` | Metal kernels、Metal runner、Metal feature gate | unit + backend boundary + preset snapshot + Metal smoke |
| `shared_runtime` | engine、scheduler、KV、admission、auto-config | unit + resource invariant + boundary + preset snapshot + Metal smoke + CUDA smoke |
| `model_contract` | registry、model loader、capabilities、chat template | unit + model contract gate + product sentinel |
| `server_api` | OpenAI server、streaming、tool/structured output | unit + product sentinel covering serve |
| `cli_run` | `ferrum run` entrypoint、interactive/multiturn | unit + product sentinel covering run |
| `release_gate` | release scripts、validators、scenario manifests | validator self-test + affected lane dry-run |
| `docs_only` | docs not changing gate behavior | markdown/document review only unless linked scripts/config changed |

### 必须交付

1. 机器可读规则文件：

```text
scripts/release/change_impact_rules.json
```

2. gate planner：

```text
scripts/release/plan_gates.py
```

3. planner 输出：

```text
gate_plan.json
gate_plan.md
```

4. release candidate manifest：

```text
release_candidate_manifest.json
```

manifest 必须记录：

- base SHA
- head SHA
- dirty status
- changed files
- impact domains
- required gates
- satisfied gates
- invalidated gates
- invalidation reason
- artifact paths
- PASS lines

### 阶段验收

必需 PASS line：

```text
CHANGE IMPACT GATE PLAN PASS: <out_dir>
```

量化通过标准：

- `crates/`、`scripts/release/`、`ferrum.toml`、`Cargo.toml` 下的变更 100% 被归类。
- release 候选不允许 `impact_domain = unknown`。
- 至少 16 个 fixture diff 用例通过。
- 每个 domain 至少有 1 个 fixture。
- shared runtime fixture 必须同时要求 Metal smoke 和 CUDA smoke。
- backend-local CUDA fixture 不得无条件要求 Metal full，只要求 Metal cheap sentinel 或 boundary 证明。
- backend-local Metal fixture 不得无条件要求 CUDA full，只要求 CUDA cheap sentinel 或 boundary 证明。
- 改动发生在已 PASS lane 之后时，manifest 必须自动把受影响 lane 标为 invalidated。

## 阶段 2: Shift-left product/backend sentinel

### 目标

在 full release gate 之前，用便宜、固定、可重复的 sentinel 提前发现产品路径问题。

sentinel 不是性能发布证据。它只回答：

- `ferrum run` 是否还能工作。
- `ferrum serve` 是否还能工作。
- streaming、usage、tool、structured output 是否还有基本行为。
- CUDA/Metal runtime preset 是否漂移。
- 是否出现 `<unk>`、`[PAD]`、mojibake、missing/duplicate `[DONE]`。

### 必须覆盖

每个 product sentinel 至少包含：

| 场景 | 最小覆盖 |
|---|---|
| `run_multiturn` | 2 轮以上，多轮输出非空，无坏 token |
| `run_first_token` | 首 token 可观测，失败显式报错 |
| `serve_chat` | OpenAI-compatible `/v1/chat/completions` 非流式 |
| `serve_stream` | exactly one `data: [DONE]`，至少 1 个输出 token |
| `serve_multiturn` | 多轮上下文不串扰 |
| `serve_tool_call` | required tool call 成功 |
| `serve_structured_output` | strict JSON schema 成功 |
| `serve_concurrency_quality` | c=1 和 c=4 均零错误 |

### Backend 矩阵

最小矩阵：

| Tier | Backend | 模型 | 触发条件 |
|---|---|---|---|
| L0 no-weight | all | runtime preset snapshot cases | 每次 shared/runtime/model/release gate 变更 |
| L1 local Metal | Metal | 一个小型 dense/chat 模型 | macOS 本地可跑时，Metal/shared/product 变更 |
| L1 CUDA cheap | CUDA 1x4090 | Qwen3 MoE/GPTQ 或 Llama dense 的最小可用 smoke | CUDA/shared/product 变更，付费前按 GPU policy 声明 |
| L2 full release | Metal/CUDA | 现有 G0 正式矩阵 | 只有 L0/L1 全过后才进入 |

### 阶段验收

必需 PASS line：

```text
PRODUCT BACKEND SENTINEL PASS: <out_dir>
```

量化通过标准：

- L0 snapshot 覆盖至少 4 个组合：
  - Metal + Llama dense
  - Metal + Qwen MoE
  - CUDA + Llama dense
  - CUDA + Qwen MoE/GPTQ
- product scenario 至少 8/8 场景通过。
- `completed_requests == expected_requests`。
- `failed_requests == 0`。
- `bad_text_count == 0`。
- streaming 场景 `done_count == 1`。
- streaming 场景 `output_tokens > 0`。
- `run` 与 `serve` 都必须有 artifact。
- hidden env 只能出现在 diagnostic artifact，不能作为 PASS artifact。
- 本地 L0/L1 sentinel 目标耗时：增量构建后 15 分钟内完成。
- CUDA cheap sentinel 目标耗时：warm pod 30 分钟内完成；超时必须保存日志并停止 full sweep。

## 阶段 3: Model onboarding conformance contract

### 目标

新增模型不能靠模型名硬编码和 release 前补洞。每个新增模型必须先通过合同，再进入 README/support matrix。

### 合同维度

模型合同必须显式记录 Ferrum 已有的五个维度：

- architecture
- compute precision
- weight format
- device/backend
- KV precision/layout

还必须补充：

- tokenizer source
- chat template source
- EOS/BOS/stop token source
- tool calling capability and source
- structured output capability and source
- reasoning/thinking behavior and source
- selected runtime preset
- selected scheduler
- selected attention implementation
- selected KV/recurrent resource budget
- fallback policy

### 必须交付

1. 合同 schema：

```text
scripts/release/model_onboarding_contract.schema.json
```

2. 合同验证器：

```text
scripts/release/model_onboarding_contract_gate.py
```

3. 每个支持模型或模型族的合同文件：

```text
docs/goals/model-contracts/<model-or-family>.json
```

4. README/support matrix 更新前的 gate 要求：

```text
MODEL ONBOARDING CONTRACT PASS: <out_dir>
```

### 阶段验收

量化通过标准：

- 新增 README 支持矩阵行之前，必须存在对应合同。
- 合同 schema 校验 100% 通过。
- chat template golden 至少覆盖 5 类输入：
  - 单轮
  - 多轮
  - system message
  - tool injection
  - reasoning history
- 模板渲染失败必须显式 fail，禁止 silent fallback 到 builtin ChatML。
- EOS/BOS/stop token 必须来自 metadata、generation config 或显式 documented config。
- runtime preset snapshot 必须记录 selected 与 rejected candidates。
- requested feature silent fallback 次数必须为 0。
- 每个新增模型至少有 `ferrum run` 和 `ferrum serve` 的 product smoke artifact。
- 性能 claim 只能在正确性 gate 全过之后出现。

## 最终验收

最终 gate 必须聚合四个阶段：

```bash
python3 scripts/release/release_regression_hardening_goal_gate.py \
  --out <out_dir> \
  --resource-invariant <resource_invariant_out> \
  --change-impact <change_impact_out> \
  --product-sentinel <product_sentinel_out> \
  --model-contract <model_contract_out>
```

必需最终 PASS line：

```text
RELEASE_REGRESSION_HARDENING GOAL PASS: <out_dir>
```

最终 artifact 必须包含：

- `goal_manifest.json`
- 四个阶段的 artifact 路径和 PASS line
- git SHA
- dirty status
- changed files
- binary SHA256，若本目标阶段构建了产品 binary
- sanitized env
- release candidate manifest
- gate plan
- resource invariant summary
- product scenario summary
- model contract summary

最终量化通过标准：

- 4/4 阶段 PASS。
- 0 个 unknown impact domain。
- 0 个 invalidated-but-counted-as-pass lane。
- 0 个 leaked resources。
- 0 个 silent fallback。
- 0 个 product sentinel failed request。
- 0 个 bad output blocker。
- 至少 1 个 CUDA/Metal shared-runtime fixture 证明 gate planner 会同时要求两端 sentinel。
- 至少 1 个 backend-local fixture 证明 gate planner 不会无脑触发另一端 full gate。

## 执行顺序

推荐顺序：

1. 阶段 1：先做 change-impact classifier 和 gate planner，让后续工作不再靠人工选择 gate。
2. 阶段 0：做资源 invariant，解决 OOM/slot/KV/recurrent 的根因可观测性。
3. 阶段 2：把便宜 product/backend sentinel 接到 planner。
4. 阶段 3：把模型接入合同接到 README/support matrix 和 release gate。

阶段 1 可以先落地 dry-run，不阻塞开发；一旦 fixture 覆盖完整，再改成 release 候选硬门禁。

## GPU 与成本规则

本目标默认不启动付费 GPU。需要 CUDA L1 sentinel 时，必须先写明：

- lane
- 预期 runtime/cost
- stop condition
- correctness gate
- performance command，若本次需要性能

CUDA sentinel 失败后：

- 保存 artifact。
- 停止 full sweep。
- 不反复重租机器试错。
- 先根据失败 domain 回到阶段 0/1/2 对应机制修复。

## Release 乒乓防止规则

release candidate 状态只能单调推进：

1. 代码变更进入 manifest。
2. impact classifier 生成 domains。
3. gate planner 生成 required gates。
4. gate 通过后记录 artifact 与 PASS line。
5. 新代码变更自动 invalidates 受影响 gates。
6. full release gate 只能在所有 required cheap sentinel 都有效时启动。

如果 CUDA 修复改动了 shared runtime：

- invalidate Metal sentinel。
- invalidate CUDA sentinel。
- 不直接启动 Metal full。
- 先跑两端 cheap sentinel。

如果 CUDA 修复只改动 CUDA backend-local：

- invalidate CUDA smoke/full。
- 保留 Metal full 结果。
- 只要求 Metal cheap sentinel 或 backend boundary audit 证明没有 shared 漏出。

Metal 方向同理。
