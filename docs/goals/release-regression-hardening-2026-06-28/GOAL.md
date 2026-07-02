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

## 为什么原版不够

原版目标文档只定义了四个方向和最终 PASS line，作为立项合同够用，但不足以指导实现。

本目标必须补齐以下工程细节：

- 每个阶段改哪些代码路径。
- 每个阶段产出哪些文件，字段是什么。
- 哪些 fixture 必须存在，fixture 覆盖什么风险。
- 失败后如何分流，避免继续做 full sweep。
- 哪些 gate 是 dry-run，哪些 gate 是 release hard gate。
- 如何证明 CUDA/Metal 没有互相污染。
- 如何证明新增模型不是靠模型名和临时补丁接入。

没有这些细节，目标仍会退化成“到 release 前人工判断还差什么”。

## 现有代码落点

本目标优先复用现有模块，不先做大规模目录重排。

| 关注点 | 现有落点 | 本目标要求 |
|---|---|---|
| KV cache allocation | `crates/ferrum-kv/src/managers/default.rs`、`crates/ferrum-kv/src/managers/paged.rs` | 增加可选 trace/invariant 事件，不改变正常 hot path 语义 |
| recurrent-state manager | `crates/ferrum-engine/src/builder.rs`、`crates/ferrum-engine/src/recurrent_state*` | 把 slot reserve/release 纳入统一资源事务 |
| scheduler/admission | `crates/ferrum-scheduler/src/implementations/continuous.rs` | defer、reject、reopen、cancel 必须进入资源/容量事件 |
| engine request lifecycle | `crates/ferrum-engine/src/continuous_engine/` | request start/end、prefill/decode failure、client cancel 必须闭合事务 |
| runtime preset | `crates/ferrum-types/src/auto_config.rs`、`crates/ferrum-types/examples/backend_runtime_preset_snapshot.rs` | snapshot 必须成为 change-impact 和 sentinel 的输入 |
| model executor contract | `crates/ferrum-interfaces/src/model_executor.rs` | silent fallback、capability 默认值和 unsupported path 必须可验证 |
| model registry/load | `crates/ferrum-engine/src/registry.rs`、`crates/ferrum-models/` | 新模型必须绑定 onboarding contract |
| CLI run | `crates/ferrum-cli/src/commands/run.rs` | sentinel 必须覆盖多轮和首 token |
| CLI serve/server | `crates/ferrum-cli/src/commands/serve.rs`、`crates/ferrum-server/` | sentinel 必须覆盖 OpenAI-compatible 非流/流式/tool/schema |
| backend boundary | `scripts/release/backend_boundary_audit.py`、`scripts/release/backend_boundary_allowlist.json` | boundary 只能证明隔离，不能代替 product sentinel |
| product scenarios | `scripts/release/run_scenarios.py`、`scripts/release/scenarios/*.json` | 扩展为 planner 可调用的 cheap sentinel |
| unified gate | `scripts/release/run_gate.py`、`scripts/release/g0_source_gate.sh` | 最终由 gate planner 输出 lane 顺序和 invalidation |

## Artifact 目录约定

每个阶段 artifact 目录必须自包含，不能依赖屏幕日志或对话记录。

标准布局：

```text
<out_dir>/
  gate.manifest.json
  command.log
  git_status.txt
  sanitized_env.json
  pass_line.txt
  failures/
  diagnostics/
```

`gate.manifest.json` 必须包含：

- `goal = "release-regression-hardening-2026-06-28"`
- `phase`
- `status`
- `started_at`
- `ended_at`
- `duration_sec`
- `repo_root`
- `git_sha`
- `git_branch`
- `git_dirty`
- `command`
- `artifact_dir`
- `pass_line`
- `inputs`
- `outputs`
- `validation_summary`

如果 `git_dirty == true`，manifest 必须列出 dirty files。dirty artifact 可以用于开发诊断，但不能用于 release-ready claim，除非最终 goal gate 显式接受并记录原因。

## 阶段升级规则

每个阶段先以 dry-run 进入，fixture 和 artifact schema 稳定后再变成 hard gate。

| 等级 | 含义 | 可以做什么 | 不能做什么 |
|---|---|---|---|
| `diagnostic` | 手工/临时验证 | 定位问题、设计 schema | 宣称阶段完成 |
| `dry_run` | 自动运行但不阻塞 | 收集误报/漏报、校准 fixture | 作为 release-ready 证据 |
| `required_for_goal` | goal 内硬门禁 | 计入阶段 PASS | 替代 G0 full release gate |
| `required_for_release` | release 候选硬门禁 | 阻止 full release gate 启动 | 跳过现有 G0 要求 |

每个阶段从 `dry_run` 升级到 `required_for_goal` 前，必须满足：

- validator 有自测。
- artifact schema 有 fixture。
- 至少一个失败 fixture 被验证为 fail。
- 至少一个成功 fixture 被验证为 pass。
- 文档中的 PASS line 被 validator 精确打印。

## 阶段 0: 资源所有权与事务 invariant

### 目标

所有运行时资源必须由统一事务记录 allocate、defer、commit、rollback、release。

覆盖资源类型：

- KV cache blocks。
- recurrent-state slots。
- scheduler/admission capacity。
- model/session cache 引用。
- prefill/decode batch 内的临时 backend workspace。

### 实现落点

优先新增一个小型内部 crate 或模块，而不是把检查逻辑散到各处。

建议路径：

```text
crates/ferrum-types/src/resource_trace.rs
crates/ferrum-engine/src/resource_invariant.rs
scripts/release/resource_invariant_gate.py
```

要求：

- `ferrum-types` 只放可序列化事件类型和 summary 类型。
- `ferrum-engine` 负责 request lifecycle 的事务闭合。
- `ferrum-kv`、`ferrum-scheduler`、recurrent-state manager 只发事件，不各自实现一套 checker。
- checker 必须可在 unit/integration test 内直接调用，也必须可从 `resource_trace.jsonl` 离线验证。
- 默认产品路径不能因为 trace 关闭而改变行为。

### 资源事件模型

事件至少包含以下类型：

| Event | 含义 |
|---|---|
| `request_open` | request 进入 engine |
| `reserve` | 资源被预留，但还未成为用户可见执行状态 |
| `commit` | 资源成为执行状态的一部分 |
| `defer` | request 因容量不足等待 |
| `reject` | request 因容量/配置不满足被拒绝 |
| `release` | 资源释放 |
| `rollback` | 事务失败，释放本事务内资源 |
| `request_close` | request 生命周期结束 |
| `capacity_snapshot` | 当前可用容量快照，用于解释 defer/reopen |

单条 JSONL 事件示例：

```json
{
  "ts": "2026-07-02T00:00:00Z",
  "event": "reserve",
  "request_id": "req-123",
  "sequence_id": "seq-123",
  "resource_kind": "recurrent_state_slot",
  "amount": 1,
  "capacity": 16,
  "owner_component": "continuous_engine.prefill",
  "phase": "prefill",
  "reason": "admitted"
}
```

### 必须捕获的失败类型

| Failure kind | 判定 |
|---|---|
| `resource_leak` | request close 后仍有净占用 |
| `release_underflow` | release 大于 commit/reserve |
| `capacity_overcommit` | reserved 或 committed 超过 capacity |
| `defer_with_committed_resource` | defer 后留下 committed 资源 |
| `rollback_incomplete` | rollback 未释放本事务资源 |
| `silent_cuda_oom` | 没有 admission/reject 事件，最终 CUDA OOM |
| `panic_after_resource_error` | 资源错误导致 panic 而非显式 error |
| `stale_capacity_reopen` | 未达到容量条件就重新 admission |

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

### 阶段 0 fixture 明细

必须新增或改造 fixture：

| Fixture | 类型 | 最小断言 |
|---|---|---|
| `kv_allocate_release_success` | unit | KV reserve/commit/release 闭合 |
| `kv_capacity_reject` | unit | capacity 不足时 reject，无 panic |
| `recurrent_slot_limit_reject` | unit | slot 不足时 reject，净占用为 0 |
| `scheduler_defer_reopen_after_capacity` | unit | 只有 capacity snapshot 满足条件才 reopen |
| `scheduler_cancel_releases_capacity` | unit | cancel 后释放 waiting/running 占用 |
| `engine_prefill_failure_rolls_back` | integration | prefill 失败后所有资源归零 |
| `engine_decode_failure_rolls_back` | integration | decode 失败后所有资源归零 |
| `serve_client_disconnect_cleans_up` | integration | client disconnect 后净占用为 0 |
| `run_multiturn_resource_balance` | product smoke | 多轮后 request/session 资源闭合 |
| `mixed_batch_partial_failure` | integration | batch 内部分失败不污染成功 request |
| `oom_prevented_by_admission` | diagnostic/CUDA | 显式 reject/defer 先于 CUDA OOM |
| `trace_replay_selftest` | validator selftest | JSONL replay 能检测 pass/fail fixture |

阶段 0 不要求真实 4090 才能完成。CUDA OOM fixture 可以先用 synthetic capacity 模拟；真实 CUDA 证据只作为后续 release sentinel 的补充。

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

### 规则文件 schema

`scripts/release/change_impact_rules.json` 必须是可审查、可测试的配置，不允许把主要规则硬编码在 Python if/else 里。

最小 schema：

```json
{
  "version": 1,
  "rules": [
    {
      "id": "shared-runtime-engine",
      "path_globs": ["crates/ferrum-engine/src/**"],
      "domains": ["shared_runtime"],
      "required_gates": [
        "unit",
        "resource_invariant",
        "backend_boundary",
        "preset_snapshot",
        "metal_sentinel",
        "cuda_sentinel"
      ],
      "reason": "engine lifecycle can affect all product backends"
    }
  ]
}
```

规则必须支持：

- `path_globs`
- `domain`
- `required_gates`
- `release_invalidation`
- `exceptions`
- `owner`
- `reason`

`exceptions` 只能缩小 gate，不允许完全跳过 unknown/shared impact。每个 exception 必须有 fixture。

### Planner 输入输出

输入：

```bash
python3 scripts/release/plan_gates.py \
  --base <base_sha> \
  --head <head_sha> \
  --out <out_dir>
```

输出：

```text
<out_dir>/
  gate_plan.json
  gate_plan.md
  changed_files.json
  release_candidate_manifest.json
  planner_selfcheck.json
```

`gate_plan.json` 最小字段：

```json
{
  "base_sha": "...",
  "head_sha": "...",
  "dirty": false,
  "changed_files": [],
  "impact_domains": [],
  "required_gates": [],
  "optional_diagnostic_gates": [],
  "invalidated_previous_gates": [],
  "unknown_files": [],
  "decision_log": []
}
```

### Invalidation 规则

必须显式编码：

| 改动域 | invalidate |
|---|---|
| `shared_runtime` | unit、resource invariant、preset snapshot、Metal sentinel、CUDA sentinel、所有 full release lanes |
| `cuda_backend` | CUDA sentinel、CUDA full/dense lanes；Metal full 保留，但要求 Metal cheap sentinel 或 boundary 证明 |
| `metal_backend` | Metal sentinel、Metal full；CUDA full 保留，但要求 CUDA cheap sentinel 或 boundary 证明 |
| `model_contract` | model contract、product sentinel、涉及该模型的 README/support evidence |
| `server_api` | serve sentinel、bench-serve correctness checks、OpenAI compatibility gates |
| `cli_run` | run sentinel、run product evidence |
| `release_gate` | validator selftests、受影响 lane dry-run |
| `docs_only` | 不 invalidate 已有 gate，除非 docs 引用的 command/config 变化 |

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

### 阶段 1 fixture 明细

至少 16 个 fixture，必须覆盖：

| Fixture | changed files 示例 | 期望 domain/gate |
|---|---|---|
| `engine_shared_runtime` | `crates/ferrum-engine/src/continuous_engine/inner/batch.rs` | shared runtime + 两端 sentinel |
| `scheduler_shared_runtime` | `crates/ferrum-scheduler/src/implementations/continuous.rs` | shared runtime + resource invariant |
| `auto_config_preset` | `crates/ferrum-types/src/auto_config.rs` | preset snapshot + 两端 sentinel |
| `kv_manager` | `crates/ferrum-kv/src/managers/paged.rs` | resource invariant + 两端 sentinel |
| `model_executor_contract` | `crates/ferrum-interfaces/src/model_executor.rs` | model contract + product sentinel |
| `registry_dispatch` | `crates/ferrum-engine/src/registry.rs` | model contract + preset snapshot |
| `cuda_kernel_local` | `crates/ferrum-kernels/src/cuda/**` | CUDA sentinel，不要求 Metal full |
| `metal_kernel_local` | `crates/ferrum-kernels/src/metal/**` | Metal sentinel，不要求 CUDA full |
| `cli_run` | `crates/ferrum-cli/src/commands/run.rs` | run sentinel |
| `serve_api` | `crates/ferrum-cli/src/commands/serve.rs` | serve sentinel |
| `bench_serve` | `crates/ferrum-cli/src/commands/bench_serve.rs` | bench correctness selftests |
| `scenario_manifest` | `scripts/release/scenarios/product_regression_smoke.json` | run_scenarios selftest + sentinel dry-run |
| `backend_boundary` | `scripts/release/backend_boundary_audit.py` | validator selftest |
| `preset_snapshot_script` | `scripts/release/backend_runtime_preset_snapshot.py` | snapshot selftest |
| `docs_goal_only` | `docs/goals/**/GOAL.md` | docs_only |
| `unknown_path` | new path under `crates/` with no rule | fail with unknown domain |

每个 fixture 必须同时验证 machine-readable JSON 和 human-readable markdown，防止 planner 输出只给机器或只给人看。

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

### Sentinel 与现有 gate 的关系

sentinel 必须复用现有工具，不新增第二套产品验证入口。

| Sentinel 部分 | 复用工具 | 本目标新增 |
|---|---|---|
| L0 preset snapshot | `scripts/release/backend_runtime_preset_snapshot.py` | planner 触发和 artifact 聚合 |
| backend boundary | `scripts/release/backend_boundary_audit.py` | planner 触发和 allowlist 审计 |
| product scenarios | `scripts/release/run_scenarios.py` | 更细的 smoke manifest 与 backend matrix |
| Metal quick | 现有 Metal build/scenario | 明确 quick sentinel artifact，不等同 Metal full |
| CUDA quick | 现有 CUDA smoke/scenario | warm pod 30 分钟 stop condition |
| final aggregation | 新 validator | `PRODUCT BACKEND SENTINEL PASS` |

### Product scenario manifest 要求

新增或扩展：

```text
scripts/release/scenarios/product_backend_sentinel.json
```

manifest 必须支持：

- `entrypoint = run | serve`
- `backend = metal | cuda | cpu | external`
- `model_id`
- `scenario_type`
- `timeout_sec`
- `expected_requests`
- `quality_assertions`
- `bad_text_scan`
- `requires_usage`
- `requires_stream_done_once`
- `diagnostic_only`

每个 scenario artifact 必须包含：

- request payload
- response body 或 stream chunks
- token usage 来源
- output token count
- bad-text scan 结果
- server logs
- effective config
- model id/path
- backend
- binary SHA256，若使用本地 binary

### 失败分流

sentinel 失败后不允许直接启动 full release sweep。必须按失败类型分流：

| 失败 | 分流 |
|---|---|
| run 失败、serve 通过 | `cli_run` 或 shared runtime |
| serve 非流失败、run 通过 | `server_api` |
| stream missing/duplicate `[DONE]` | `server_api` / SSE |
| usage 缺失或 token 计数错误 | `bench_serve` / server usage |
| `<unk>`、`[PAD]`、mojibake | tokenizer/logits/sampling/model contract |
| CUDA only 失败 | CUDA backend 或 CUDA runtime preset |
| Metal only 失败 | Metal backend 或 Metal runtime preset |
| 两端都失败 | shared runtime/model/template |
| preset snapshot diff | auto_config/runtime preset |

每个失败 artifact 必须写入 `failure_classification.json`，包括：

- `failure_kind`
- `first_bad_artifact`
- `suspected_domain`
- `next_gate`
- `do_not_run`

`do_not_run` 用于显式阻止无意义的 full sweep，例如：product sentinel 已经 stream 格式错误时，不得继续跑性能 full gate。

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

### 阶段 2 fixture 明细

必须有 runner selftest fixture 覆盖：

| Fixture | 期望 |
|---|---|
| `serve_stream_done_once_pass` | exactly one `[DONE]` pass |
| `serve_stream_missing_done_fail` | missing `[DONE]` fail |
| `serve_stream_duplicate_done_fail` | duplicate `[DONE]` fail |
| `serve_stream_zero_output_fail` | 0 output token fail |
| `serve_malformed_sse_fail` | malformed SSE JSON fail |
| `bad_text_unk_fail` | `<unk>` fail |
| `bad_text_pad_fail` | `[PAD]` fail |
| `mojibake_fail` | mojibake fail |
| `run_multiturn_empty_fail` | multi-turn empty output fail |
| `tool_required_missing_fail` | required tool-call missing fail |
| `structured_invalid_json_fail` | strict schema invalid fail |
| `hidden_env_release_evidence_fail` | hidden env cannot be PASS evidence |

这些 fixture 可以全部用 synthetic server/log fixture 完成，不需要真实模型。

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

### 合同 schema 轮廓

合同必须能区分“模型事实”“Ferrum 选择”“验证证据”。

最小结构：

```json
{
  "schema_version": 1,
  "model": {
    "id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
    "family": "qwen3_moe",
    "architecture": "qwen3_moe",
    "weight_format": "gptq_int4",
    "source": "hf"
  },
  "facts": {
    "tokenizer_source": "tokenizer_config.json",
    "chat_template_source": "tokenizer_config.json",
    "generation_config_source": "generation_config.json",
    "eos_token_ids": [],
    "bos_token_id": null,
    "stop_tokens": []
  },
  "ferrum_selection": {
    "backend": "cuda",
    "runtime_preset": "auto",
    "scheduler": "...",
    "attention_impl": "...",
    "kv_layout": "...",
    "kv_dtype": "...",
    "recurrent_state_max_slots": null
  },
  "capabilities": {
    "tool_calling": {"supported": true, "source": "chat_template"},
    "structured_output": {"supported": true, "source": "server_guided_decode"},
    "reasoning": {"supported": false, "source": "metadata"}
  },
  "fallback_policy": {
    "allow_builtin_template_fallback": false,
    "allow_backend_fallback": false,
    "allow_attention_fallback": false
  },
  "evidence": {
    "template_golden": "...",
    "run_smoke": "...",
    "serve_smoke": "...",
    "preset_snapshot": "..."
  }
}
```

### 支持矩阵写入规则

新增或修改 README/support matrix 行前，必须满足：

- 合同文件存在。
- 合同 gate PASS。
- `ferrum run` artifact 存在。
- `ferrum serve` artifact 存在。
- 如果声明 tool calling，必须有 required tool-call artifact。
- 如果声明 structured output，必须有 strict schema artifact。
- 如果声明 CUDA support，必须有 CUDA artifact。
- 如果声明 Metal support，必须有 Metal artifact。
- 如果声明性能数字，必须有同硬件 benchmark artifact。

README 只允许引用合同中已经通过的能力。合同未覆盖的能力必须写成 unsupported 或 not yet validated。

### 禁止模式

以下模式必须由 gate 拒绝：

- 按模型名字符串硬编码 chat template 行为。
- 模板渲染失败后 silent fallback 到 builtin ChatML。
- 请求 CUDA/Metal 某路径后 silent fallback 到 base path。
- EOS/BOS/stop token 按家族猜测但 artifact 没有来源。
- README 声称支持 tool/schema，但合同缺对应 artifact。
- 只跑 `serve`，没有 `run`。
- 只跑 `run`，没有 `serve`。
- 只跑 CPU/no-weight，声称 Metal/CUDA 支持。
- correctness 未过就写性能数字。

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

### 阶段 3 fixture 明细

至少覆盖：

| Fixture | 期望 |
|---|---|
| `valid_qwen_moe_contract` | pass |
| `valid_llama_dense_contract` | pass |
| `missing_chat_template_source` | fail |
| `builtin_template_fallback_allowed` | fail |
| `missing_run_artifact` | fail |
| `missing_serve_artifact` | fail |
| `tool_claim_without_tool_artifact` | fail |
| `schema_claim_without_schema_artifact` | fail |
| `cuda_claim_without_cuda_artifact` | fail |
| `performance_claim_before_correctness` | fail |
| `unknown_architecture_without_design_doc` | fail |
| `runtime_preset_missing_rejected_candidates` | fail |

阶段 3 完成后，新增模型接入应该先填合同，再写代码和 gate；不再反过来先把模型跑起来，release 前补文档。

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

## Work packages

本目标必须拆成小 PR，避免把 release gate、runtime 行为、模型合同和测试架构混在一个 patch。

### WP1: Planner dry-run

范围：

- `scripts/release/change_impact_rules.json`
- `scripts/release/plan_gates.py`
- planner selftests
- fixture diffs

不允许修改 engine/runtime 行为。

完成标准：

- `CHANGE IMPACT GATE PLAN PASS: <out_dir>`
- 16 个 fixture 全过。
- 当前工作树能生成 `gate_plan.md`，并清楚列出本 PR 只影响 planner。

### WP2: Resource trace schema and offline checker

范围：

- `ferrum-types` trace event/schema。
- offline checker。
- synthetic JSONL pass/fail fixtures。

不允许改 engine admission 策略。

完成标准：

- checker 能发现 leak、underflow、overcommit、defer-with-commit、rollback incomplete。
- `RESOURCE INVARIANT GATE PASS` 可以在 synthetic fixtures 上通过。

### WP3: Resource instrumentation

范围：

- KV manager trace。
- recurrent-state manager trace。
- scheduler/admission trace。
- engine lifecycle trace。

要求先接 unit/integration fixture，再考虑产品 smoke。

完成标准：

- 12/12 阶段 0 场景通过。
- trace 关闭时现有测试行为不变。
- trace 打开时 artifact 能离线 replay。

### WP4: Product/backend sentinel manifest

范围：

- `scripts/release/scenarios/product_backend_sentinel.json`
- `run_scenarios.py` 必要扩展。
- synthetic bad-output/SSE selftests。

不允许新增第二套 HTTP benchmark 客户端。

完成标准：

- 12 个阶段 2 fixture 全过。
- `PRODUCT BACKEND SENTINEL PASS` 在 synthetic/no-weight 层可验证。

### WP5: Sentinel integration with planner

范围：

- planner 输出 required sentinel。
- release candidate manifest invalidation。
- sentinel artifact 聚合。

完成标准：

- shared runtime fixture 会要求 Metal + CUDA sentinel。
- backend-local fixture 不会无脑要求另一端 full gate。
- 修改已 PASS lane 相关文件会自动 invalidate 该 lane。

### WP6: Model contract schema and validator

范围：

- `model_onboarding_contract.schema.json`
- `model_onboarding_contract_gate.py`
- valid/invalid contract fixtures。

不要求一次性补齐所有现有模型合同。

完成标准：

- 阶段 3 fixture 全过。
- validator 能拒绝 silent fallback、缺 run/serve artifact、缺能力证据。

### WP7: First real model contract pilot

范围：

- 选择一个已支持且体积小的模型族做 pilot。
- 产出合同、template golden、run/serve smoke artifact。

完成标准：

- `MODEL ONBOARDING CONTRACT PASS: <out_dir>`。
- README/support matrix 引用合同字段，而不是重复手写能力声明。

### WP8: Final aggregator

范围：

- `scripts/release/release_regression_hardening_goal_gate.py`
- 聚合四阶段 artifacts。
- final selftest。

完成标准：

- `RELEASE_REGRESSION_HARDENING GOAL PASS: <out_dir>`。
- final manifest 能指出每个阶段 artifact 和 PASS line。

## 每个 PR 的 Definition of Done

每个属于本目标的 PR 必须在描述里写清：

- 属于哪个 WP。
- 本 PR 的 impact domains。
- planner 输出的 required gates。
- 实际运行的命令。
- artifact 目录。
- PASS line。
- 未运行 gate 的原因。
- 是否触碰 shared runtime。
- 是否需要 CUDA/Metal cross-backend sentinel。

如果 PR 修改了 `crates/` 但 planner 结果是 `docs_only`，PR 必须 fail。

如果 PR 修改了 shared runtime，但没有 Metal/CUDA sentinel 计划，PR 必须 fail。

如果 PR 修改了模型能力、模板、registry、runtime preset，但没有 model contract 或 product sentinel 计划，PR 必须 fail。

## 成功后的日常工作流

目标完成后，普通开发流程应该变成：

```bash
python3 scripts/release/plan_gates.py --base origin/main --head HEAD --out <out>/plan
cat <out>/plan/gate_plan.md
python3 scripts/release/run_gate.py unit --out <out>/unit
python3 scripts/release/release_regression_hardening_goal_gate.py --out <out>/final ...
```

对于 release candidate：

1. 先生成 gate plan。
2. 先跑 required cheap gates。
3. cheap gates 全部有效后再跑 full release gates。
4. 任意代码变化后重新生成 plan。
5. final summary 只接受未失效 artifact。

这样 release 的状态从“人脑记忆”变成 manifest 驱动。

## 失败处理手册

### 资源 invariant 失败

处理顺序：

1. 找 `first_bad_event_index`。
2. 看 `owner_component`。
3. 分类到 KV、recurrent、scheduler、engine lifecycle。
4. 只修对应 ownership 闭合。
5. 重新跑 resource invariant。
6. 不进入 product sentinel，直到 invariant pass。

### Planner 失败

处理顺序：

1. 如果是 unknown file，先补规则和 fixture。
2. 如果是 gate 过宽，补 exception 和证明 fixture。
3. 如果是 gate 过窄，补 required gate 和 regression fixture。
4. 不用人工绕过 planner，除非 artifact 标记 diagnostic。

### Product sentinel 失败

处理顺序：

1. 看 `failure_classification.json`。
2. run/serve 单边失败先修 entrypoint。
3. 两端失败先查 shared runtime/template/model contract。
4. backend 单边失败查 backend/runtime preset。
5. streaming/usage/tool/schema 失败查 server API。
6. 在 sentinel pass 前，不跑 full performance gate。

### Model contract 失败

处理顺序：

1. 缺 metadata 来源，先补来源或标 unsupported。
2. 缺 artifact，先补 smoke，不补 README。
3. silent fallback，先修产品路径。
4. correctness 未过，不写性能 claim。

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
