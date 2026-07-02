# Release Regression Hardening Goal

## 状态

草案目标文件，创建于 2026-06-28。

本目标用于解决五类反复出现的问题：

- 资源所有权不清导致 OOM、slot 泄漏、defer/rollback 不一致。
- CUDA 与 Metal 相互影响，优化一端改坏另一端。
- 问题发现太晚，经常到 release 前由人工手测暴露。
- release 回归像乒乓球，CUDA 修完回归 Metal，Metal 修完又回归 CUDA。
- 正确性、内存、耗时问题缺少统一 profile/replay artifact，定位常依赖临场日志和人工猜测。

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
| 定位痛苦 | 正确性/内存/耗时问题缺统一 request/resource/memory/latency 证据 | `run` 和 `serve` 都能产出同一 schema 的 observability profile；100% blocking failure 有 request id、phase、first failure event、memory/resource 快照和 replay command |

## 非目标

- 不在本目标里重写 CUDA/Metal kernel。
- 不把 Ferrum 拆成 CUDA engine 和 Metal engine 两套产品路径。
- 不用隐藏环境变量作为产品验收证据。
- 不把 cheap sentinel 当作 full release performance evidence。
- 不用 live vLLM 跑分作为本目标必需项；vLLM 只作为源码/历史行为参考，除非另立性能目标。
- 不在本目标里扩大付费 GPU 矩阵；需要 CUDA 运行时，优先 cheap smoke，full sweep 仍按现有 G0/release 规则执行。

## 为什么原版不够

原版目标文档只定义了四个方向和最终 PASS line，作为立项合同够用，但不足以指导实现。后续补充的第五个方向必须把这些阶段串起来，解决“知道失败但不知道在哪里失败、为什么失败、如何复现”的问题。

本目标必须补齐以下工程细节：

- 每个阶段改哪些代码路径。
- 每个阶段产出哪些文件，字段是什么。
- 哪些 fixture 必须存在，fixture 覆盖什么风险。
- 失败后如何分流，避免继续做 full sweep。
- 哪些 gate 是 dry-run，哪些 gate 是 release hard gate。
- 如何证明 CUDA/Metal 没有互相污染。
- 如何证明新增模型不是靠模型名和临时补丁接入。
- 如何在正确性、内存、耗时故障出现时，用固定 artifact 在 10 分钟内定位到 request、资源、阶段和可 replay 输入。

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
| existing profile JSONL | `crates/ferrum-bench-core/src/profile.rs` | 现有 bench/profile envelope 可保留，但必须升级为 product `run`/`serve` 共用的 request-scoped observability schema |
| chrome trace | `crates/ferrum-bench-core/src/trace.rs` | 继续作为 Perfetto/Nsight 友好的时间线输出，但不能代替 JSONL 诊断证据 |
| server metrics | `crates/ferrum-server/src/traits.rs`、`crates/ferrum-types/src/metrics.rs`、`scripts/release/g1_vllm_migration_gate.py` | Prometheus/health 指标必须和 profile summary 可交叉核对 |
| runtime memory sizing | `crates/ferrum-cli/src/gpu_mem_autosize.rs`、`crates/ferrum-engine/src/continuous_engine/inner.rs` | startup/profile-run/graph/KV/recurrent/serve 运行期内存快照必须落入统一 memory profile |

## 本地 vLLM 代码对照与 Ferrum 设计映射

本节只使用本地源码对照，不要求 live vLLM 运行或跑分。对照版本：

```text
/Users/chejinxuan/py_ws/vllm
git rev-parse HEAD = 0b3ba88f165976e77ca5e6a7a3f5bba4562b80af
```

主要审计文件：

- `vllm/config/observability.py`：typed observability config、KV residency、NVTX、iteration details。
- `vllm/config/profiler.py`、`vllm/profiler/wrapper.py`、`vllm/entrypoints/serve/profile/api_router.py`：typed profiler config、delay/max iteration、start/stop profile。
- `vllm/utils/mem_utils.py`、`vllm/v1/worker/gpu_worker.py`：startup memory snapshot、profile_run memory accounting、KV cache available memory calculation。
- `vllm/v1/core/kv_cache_metrics.py`、`vllm/v1/metrics/stats.py`、`vllm/v1/metrics/loggers.py`：KV residency sampling、scheduler/request stats、Prometheus/logging metrics。
- `vllm/tracing/utils.py`、`vllm/v1/engine/output_processor.py`、`tests/v1/tracing/test_tracing.py`：request-level OTel span fields and tests。

vLLM 的可借鉴点不是某个单点 profiler，而是五层闭环：

| vLLM 本地源码 | 关键做法 | Ferrum 设计映射 | 量化指标 |
|---|---|---|---|
| `vllm/config/observability.py` | `ObservabilityConfig` 统一控制 OTel、detailed traces、KV cache residency、CUDA graph、NVTX、MFU、iteration details；详细 trace 明确标注可能有开销 | 新增 typed `FerrumObservabilityConfig`，由 CLI/config 显式设置，禁止只靠隐藏 env 作为产品验收入口；`run` 和 `serve` 使用同一配置结构 | 100% product observability 功能有 typed flag/config 字段；0 个 release gate 依赖未文档化 env-only 开关 |
| `vllm/config/profiler.py`、`vllm/profiler/wrapper.py`、`vllm/entrypoints/serve/profile/api_router.py` | profiler 类型、输出目录、stack/flops/memory/shapes、delay/max/warmup/active/wait iteration 全部是 typed config；start/stop 幂等；API profiler 明确只用于本地开发 | Ferrum 分成低开销 always-available JSONL profile 与高开销 diagnostic profiler；serve 的 start/stop profile 只能在显式 diagnostic 模式启用，artifact 必须标注不可作为 performance evidence | basic profile c=1/c=4 吞吐开销 <= 1%；debug/full profiler artifact 一律标记 `diagnostic_only=true`；start/stop profiler 自测覆盖重复 start、stop-before-start、delay、max-iteration |
| `vllm/utils/mem_utils.py`、`vllm/v1/worker/gpu_worker.py` | 先 init distributed/NCCL 再取 memory snapshot；把内存分为外部进程、torch/current instance、non-torch/current instance；profile_run 后计算 non-KV、peak activation、weights、CUDA graph estimate，再决定 KV cache 可用内存；若 profiling 期间外部内存变化，显式 fail | Ferrum 启动和 autosize 必须输出分阶段 memory profile：device/backend init、model weights、warmup/profile run、CUDA graph/Metal pipeline cache、KV cache、recurrent state、backend workspace、serve runtime high-water；CUDA 和 Metal 使用各自可用采样源但写入同一 schema | 每次 CUDA/Metal product sentinel 至少 6 个 memory snapshot stage；内存事件 100% 有 before/after/current/high_water bytes；`available_kv_or_state_bytes` 不允许为负；外部显存波动超过 256 MiB 或 2% 必须标为 `external_memory_changed` 并阻止 performance claim |
| `vllm/v1/core/kv_cache_metrics.py`、`vllm/v1/metrics/stats.py` | KV block lifetime/idle/reuse gaps 采样；scheduler stats 记录 running/waiting/skipped/deferred、KV usage、prefix hit、preemptions、corrupted requests；request stats 记录 queued/scheduled/first/last token 时间 | Ferrum 的 KV blocks 和 recurrent-state slots 都要有生命周期采样；scheduler/admission/resource invariant 共享 request id/sequence id/correlation id；defer/reject/reopen 事件必须能和 profile summary 对齐 | KV/recurrent 采样默认 1%，可配置到 100% 用于 fixture；profile summary 输出 p50/p95/p99 lifetime/idle/reuse gap；0 个 request close 后仍有 active sampled slot/block；defer/reject 100% 有容量原因 |
| `vllm/tracing/utils.py`、`vllm/v1/engine/output_processor.py`、`tests/v1/tracing/test_tracing.py` | OTel span 带 request id、prompt/completion tokens、temperature/top_p/max_tokens/n、queue/TTFT/e2e/prefill/decode/inference latency；测试验证 span 属性存在 | Ferrum JSONL profile 先成为 release evidence；OTel 作为可选导出层，字段从同一 typed event 派生，避免 JSONL/OTel 两套语义漂移 | 100% finished request 有 request id、prompt token count、output token count、finish reason、TTFT、ITL、E2E；OTel 开启时 span 与 JSONL request summary 数值误差 <= 1ms 或 1% |
| `vllm/v1/metrics/loggers.py`、`vllm/v1/metrics/prometheus.py`、`vllm/v1/metrics/reader.py` | 人类日志、Prometheus 指标、in-memory snapshot 共用 scheduler/request stats；Prometheus 多进程目录有生命周期约束 | Ferrum `/metrics`、`/health`、profile summary、gate analyzer 必须由同一 runtime stats 源生成或可校验一致；release gate 不能只信日志字符串 | profile summary 与 `/metrics` 中 active/queued/failed/prefix-cache 计数差异必须为 0；多进程/多 worker 指标目录或临时文件必须在 manifest 中记录并由 gate 检查清理 |

结合 Ferrum 当前架构，本目标不照搬 PyTorch/TorchProfiler 细节，而采用以下分层：

1. **Schema 层**：`ferrum-types` 定义 `FerrumProfileEvent`、`FerrumMemorySnapshot`、`FerrumRequestReplayBundle`、`FerrumObservabilityConfig`。所有事件可 JSON 序列化，可由 unit tests 和 release scripts 离线验证。
2. **Runtime emit 层**：`ferrum-engine`、`ferrum-kv`、`ferrum-scheduler`、recurrent-state manager 只发事件，不各自实现 checker；事件必须带 `request_id`、`sequence_id`、`correlation_id`、`backend`、`model_id`、`phase`。
3. **Product wiring 层**：`ferrum run` 和 `ferrum serve` 暴露相同 typed profile/replay flags。`run` 不能只靠 bench-only env，`serve` 不能只靠 HTTP benchmark side-channel。
4. **Backend adapter 层**：CUDA/Metal/CPU 只负责填充 backend-specific memory/workspace/graph/pipeline 字段；公共 analyzer 不允许读取 backend 专用日志才能判断是否 PASS。
5. **Gate/analyzer 层**：release scripts 只消费 artifact，不需要 attach 到 live process；失败后输出 first failure event、最小 replay command、resource/memory/latency 摘要。

本阶段的核心量化目标：

| 指标 | 目标 |
|---|---|
| 失败定位闭环 | 每个 blocking failure 在 artifact 中有 `first_failure_event`、`request_id` 或 `global_failure_id`、phase、error kind、nearest resource event、nearest memory snapshot、replay command |
| 定位时间 | 对已覆盖 fixture，gate summary 必须能在 10 分钟内从 artifact 定位到具体 request/phase/resource，不依赖屏幕日志 |
| profile 完整性 | product sentinel 中 100% `run`/`serve` request 有 lifecycle summary；100% timed event 有 `duration_us`；100% memory event 有 before/after/high-water bytes |
| replay 覆盖 | 100% correctness blocker、OOM/admission blocker、panic/error blocker 产出 replay bundle；synthetic replay gate 0 failure |
| 低开销 | `profile_detail=basic` 在 no-weight/synthetic 或小模型 c=1/c=4 下吞吐开销 <= 1%，事件丢失率 0；`debug/full` 允许更高开销但必须标记 diagnostic-only |
| 跨入口一致性 | `ferrum run` 与 `ferrum serve` 的 event schema 完全相同；同一 synthetic request 的 token counts、finish reason、bad-output classification 一致 |
| 跨后端一致性 | CUDA、Metal 同名阶段字段一致；backend 专有字段只能放在 `backend_detail`，公共 gate 不依赖专有字段 |

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

## 阶段 4: Unified observability profile and replay diagnostics

### 目标

把正确性、内存、耗时问题的定位从“看散落日志、猜哪条路径变了”改成固定 artifact 驱动。

阶段 4 必须覆盖：

- `ferrum run`。
- `ferrum serve` 非流式。
- `ferrum serve` 流式。
- scheduler/admission/defer/reject/reopen。
- KV cache blocks。
- recurrent-state slots。
- CUDA/Metal/CPU 公共阶段。
- 后端专有 memory/workspace/graph/pipeline 细节。
- bad output、OOM/prevented OOM、panic/error、client cancel、missing/duplicate `[DONE]`。

阶段 4 不替代阶段 0/1/2/3，而是为这些阶段提供同一证据底座：

- 阶段 0 的 resource invariant 消费 resource/resource-lifecycle events。
- 阶段 2 的 product sentinel 消费 request lifecycle、bad-output、streaming、latency、memory events。
- 阶段 3 的 model onboarding contract 引用 request replay bundle、token dump、runtime preset snapshot。
- 最终 goal gate 只接受已经通过 analyzer 的 observability profile summary。

### 必须交付

建议新增或扩展以下路径：

```text
crates/ferrum-types/src/observability_profile.rs
crates/ferrum-bench-core/src/profile.rs
crates/ferrum-engine/src/observability/
crates/ferrum-cli/src/commands/run.rs
crates/ferrum-cli/src/commands/serve.rs
scripts/release/analyze_ferrum_profile.py
scripts/release/observability_profile_gate.py
scripts/release/fixtures/observability_profile/
```

`crates/ferrum-types` 是 schema 源头；`ferrum-bench-core` 可以继续承载 bench profile 兼容层，但 product runtime 的 schema 不应只定义在 bench crate 里。

必须新增 typed product flags/config，至少包括：

| Option | 入口 | 默认 | 要求 |
|---|---|---|---|
| `--profile-jsonl <path>` | `run`、`serve` | off | 输出统一 request-scoped JSONL profile |
| `--profile-detail <off|basic|debug|full>` | `run`、`serve` | `off` | `basic` 为低开销 release 诊断；`debug/full` 标记 diagnostic-only |
| `--memory-profile-jsonl <path>` | `run`、`serve` | off | 输出 memory lifecycle snapshot，可与 profile-jsonl 合并但 schema 必须一致 |
| `--scheduler-trace-jsonl <path>` | `run`、`serve` | off | 输出 admission/defer/reject/reopen/batch composition |
| `--request-dump-dir <dir>` | `run`、`serve` | off | 输出 sanitized request/replay bundle |
| `--profile-sample-rate <float>` | `run`、`serve` | `0.01` | 控制 KV/recurrent lifecycle sampling，fixture 可设为 1.0 |

隐藏 env 可以作为开发 shortcut，但 release gate 和产品验收必须使用 typed options 或 documented config preset。

### JSONL event envelope

统一 profile JSONL 至少包含以下顶层字段：

| Field | 必需 | 说明 |
|---|---|---|
| `schema_version` | yes | 例如 `ferrum.observability.v1` |
| `ts_unix_nanos` | yes | 单调排序可验证 |
| `event_id` | yes | 单事件唯一 id |
| `event_kind` | yes | `request`、`scheduler`、`resource`、`memory`、`latency`、`token`、`stream`、`error`、`profile_marker` |
| `phase` | yes | `startup`、`load_model`、`warmup`、`prefill`、`decode`、`stream_send`、`shutdown` 等 |
| `request_id` | conditional | request-scoped event 必须有；global failure 用 `global_failure_id` |
| `sequence_id` | conditional | sequence-scoped event 必须有 |
| `correlation_id` | yes | 同一次 product command 内贯穿 run/serve/server/engine/scheduler |
| `backend` | yes | `cpu`、`cuda`、`metal` |
| `entrypoint` | yes | `run`、`serve`、`bench-serve`、`test` |
| `model_id` | yes | HF id、本地模型 label 或 fixture label |
| `runtime_preset_hash` | yes | 与 runtime preset snapshot 对齐 |
| `duration_us` | conditional | timed event 必须有，且大于等于 0 |
| `shape` | yes | batch size、tokens、seq len、blocks、slots 等 |
| `resource` | conditional | resource event 必须有 owner、kind、capacity、reserved、committed、released |
| `memory` | conditional | memory event 必须有 before/after/current/high_water bytes |
| `token` | conditional | token event 必须有 token id/source；文本可脱敏或省略 |
| `error` | conditional | error event 必须有 error kind、recoverability、first/caused_by |
| `backend_detail` | no | 后端专有字段，只能被 backend-specific analyzer 使用 |

示例：

```json
{
  "schema_version": "ferrum.observability.v1",
  "ts_unix_nanos": 1783000000000000000,
  "event_id": "evt-000123",
  "event_kind": "memory",
  "phase": "profile_run",
  "request_id": null,
  "sequence_id": null,
  "correlation_id": "cmd-20260702-204400",
  "backend": "cuda",
  "entrypoint": "serve",
  "model_id": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
  "runtime_preset_hash": "sha256:...",
  "duration_us": 1182234,
  "shape": {"batch_size": 1, "max_num_seqs": 16},
  "memory": {
    "stage": "after_profile_run",
    "before_bytes": 16911433728,
    "after_bytes": 18791268352,
    "current_bytes": 18791268352,
    "high_water_bytes": 19025346560,
    "free_before_bytes": 7969177600,
    "free_after_bytes": 6089342976,
    "weights_bytes": 14500000000,
    "kv_cache_bytes": 0,
    "recurrent_state_bytes": 0,
    "graph_or_pipeline_bytes": 391000000,
    "backend_workspace_bytes": 294000000,
    "external_delta_bytes": 0
  },
  "backend_detail": {"cuda_device": 0}
}
```

### Memory profile requirements

借鉴 vLLM startup memory profiling，但按 Ferrum 架构扩展到 CUDA/Metal。

必须采集这些阶段：

| Stage | CUDA | Metal | 量化要求 |
|---|---|---|---|
| `process_start` | driver/NVIDIA visible memory when available | process RSS/Metal device info when available | command 启动后第一张快照 |
| `backend_initialized` | CUDA context/NCCL/workspace 初始化后 | Metal device/pipeline/cache 初始化后 | 必须发生在容量 autosize 前 |
| `model_loaded` | weights、quant buffers、backend workspace | weights、Metal buffers、pipeline cache | 记录 weights/current/high-water |
| `profile_run_done` | warmup/profile_run peak activation、non-KV memory、CUDA graph estimate | warmup/profile peak、pipeline/workspace estimate | 用于决定 KV/recurrent 可用容量 |
| `cache_allocated` | KV blocks、recurrent slots、graph pool | KV/recurrent buffers、Metal cache | 记录 capacity 和 reservation |
| `first_request_done` | runtime high-water | runtime high-water | 与 request TTFT/E2E 对齐 |
| `shutdown` | remaining allocations | remaining allocations | resource leak count 必须为 0 |

Memory analyzer 必须输出：

- `memory_high_water_bytes` per backend/device。
- `weights_bytes`。
- `kv_cache_bytes`。
- `recurrent_state_bytes`。
- `graph_or_pipeline_bytes`。
- `backend_workspace_bytes`。
- `peak_activation_bytes`。
- `external_delta_bytes`。
- `available_kv_or_state_bytes`。
- `oom_prevented_count`。
- `oom_after_admission_count`。

量化验收：

- `available_kv_or_state_bytes < 0` 必须 fail。
- 没有 admission/reject/defer 事件而出现 CUDA/Metal OOM，必须计为 `silent_oom` fail。
- 外部显存/进程内存在 profile_run 前后变化超过 256 MiB 或 2%，必须把 artifact 标为 diagnostic-only，不允许作为 performance claim。
- `shutdown` 后 resource leak count 必须为 0。

### KV/recurrent residency sampling

借鉴 vLLM KV cache residency metrics，但 Ferrum 必须同时覆盖 recurrent-state slots。

采样对象：

- KV block。
- paged KV page/block。
- recurrent-state slot。
- scheduler admission capacity token。
- backend temporary workspace allocation，若可追踪。

采样事件：

| Event | 字段 |
|---|---|
| `resource_allocated` | owner、kind、id、capacity、amount、request_id、phase |
| `resource_accessed` | owner、kind、id、request_id、phase |
| `resource_deferred` | owner、kind、needed、available、reason |
| `resource_released` | owner、kind、id、request_id、phase |
| `resource_evicted` | owner、kind、id、lifetime_us、idle_us、reuse_gap_us[]、reason |

量化验收：

- 默认 sample rate 为 1%，fixture 必须可设为 100%。
- summary 必须输出 lifetime/idle/reuse gap 的 p50/p95/p99。
- request close 后仍 active 的 sampled resource 数必须为 0。
- defer/reject 事件 100% 有 `needed`、`available`、`capacity`、`reason`。

### Request dump and replay

每个 correctness blocker 必须产出 sanitized replay bundle。

目录结构：

```text
request_dumps/
  <request_id>/
    request.json
    prompt_token_ids.json
    sampling_params.json
    runtime_effective_config.json
    backend_selection.json
    output_token_ids.json
    output_text.txt
    bad_output_scan.json
    replay.command.json
```

`request.json` 必须脱敏用户原文；release fixtures 可保存完整 synthetic prompt。真实用户文本默认不写入 artifact，除非 gate 明确使用公开 fixture。

Replay 必须支持：

- 不启动 HTTP server 的 engine-level replay。
- `ferrum run` replay。
- `ferrum serve` request replay，至少能重放 request body、headers 中的 trace context、stream/nonstream 模式。

量化验收：

- 100% bad-output blocker 有 request dump。
- 100% OOM/admission blocker 有 capacity/resource/memory dump。
- 100% panic/error blocker 有 first failure event、backtrace/log excerpt、nearest request 或 global failure id。
- replay fixture 成功率 100%；replay 失败必须成为 gate failure，而不是 warning。

### Correctness and stream diagnostics

Profile analyzer 必须分类并计数：

- `<unk>`。
- `[PAD]` 或 tokenizer reserved id 泄漏。
- invalid UTF-8/mojibake。
- bad stop reason。
- missing `[DONE]`。
- duplicate `[DONE]`。
- malformed SSE JSON。
- stream bulk flush。
- strict schema failure。
- required tool failure。
- NaN/Inf logits，若 backend 可检测。
- silent fallback from requested feature to base behavior。

量化验收：

- release candidate 中 `bad_text_count == 0`。
- `stream_missing_done_count == 0`。
- `stream_duplicate_done_count == 0`。
- `malformed_sse_count == 0`。
- `silent_fallback_count == 0`。
- 如果出现 bad output，`first_bad_token_id` 或 `first_bad_text_span` 必须存在。

### Latency diagnostics

Profile analyzer 必须输出 request 和 iteration 两层 latency。

Request latency：

- queued。
- scheduled。
- prefill。
- decode。
- inference。
- time to first token。
- inter-token latency。
- e2e。
- stream send。
- tokenizer/template time，若可观测。

Iteration latency：

- iteration id。
- running request count。
- waiting request count。
- deferred request count。
- prefill token count。
- decode token count。
- batch composition。
- model forward。
- sampler/logits。
- detokenize。
- response serialization/send。

量化验收：

- 100% finished request 有 TTFT/E2E。
- 生成 token 数大于 1 的 request 必须有 ITL summary。
- c=1/c=4 synthetic sentinel 必须输出 TTFT/ITL/E2E p50/p95/p99。
- summary 必须输出 top 5 slow phases，且每个 slow phase 能链接到 event id。

### OTel/NVTX/Chrome trace 关系

OTel、NVTX、Chrome trace 是可选诊断导出层，不是唯一 release evidence。

规则：

- JSONL observability profile 是 hard gate 输入。
- OTel span 字段必须从同一 request lifecycle event 派生。
- NVTX/layerwise trace 只能在 diagnostic/full profile 启用。
- Chrome trace 可用于 Perfetto/Nsight，但 analyzer 不能只靠 Chrome trace 判断 correctness。
- `debug/full` profile 产生的 performance 数字默认 diagnostic-only，除非另有同硬件 A/B 证据和低开销证明。

### 阶段 4 fixture 明细

至少覆盖：

| Fixture | 期望 |
|---|---|
| `run_success_basic_profile` | pass，包含 request lifecycle、TTFT/E2E、token count |
| `serve_nonstream_success_basic_profile` | pass，包含 request lifecycle、finish reason、metrics 对齐 |
| `serve_stream_success_basic_profile` | pass，exactly one `[DONE]`，usage/token count 对齐 |
| `missing_request_id` | fail |
| `timed_event_missing_duration` | fail |
| `memory_event_missing_before_after` | fail |
| `resource_event_missing_owner` | fail |
| `bad_output_without_replay_bundle` | fail |
| `oom_without_admission_event` | fail |
| `panic_without_first_failure_event` | fail |
| `scheduler_defer_without_capacity_reason` | fail |
| `kv_recurrent_lifecycle_leak` | fail |
| `profile_schema_run_serve_mismatch` | fail |
| `debug_profile_used_as_perf_claim` | fail |
| `prometheus_profile_count_mismatch` | fail |

### 阶段验收

必须有阶段验证器：

```bash
python3 scripts/release/observability_profile_gate.py \
  --out <out_dir> \
  --fixtures scripts/release/fixtures/observability_profile
```

必需 PASS line：

```text
OBSERVABILITY PROFILE GATE PASS: <out_dir>
```

阶段 4 通过标准：

- 15 个 fixture 全过。
- `run` 和 `serve` 至少各有一个通过 analyzer 的 artifact。
- `profile_detail=basic` overhead fixture <= 1% 或在 no-weight/synthetic gate 中证明事件写入路径固定成本满足预算。
- `observability_profile_summary.json` 存在，并包含：
  - `request_count`
  - `failed_count`
  - `corrupted_count`
  - `bad_text_count`
  - `oom_prevented_count`
  - `silent_oom_count`
  - `latency_p50_p95_p99`
  - `memory_high_water_bytes`
  - `resource_leak_count`
  - `top_slow_phases`
  - `first_failure_event`
  - `replay_commands`
- `first_failure_event` 对失败 fixture 100% 存在。
- release claim 中 0 个 unclassified failure。

## 最终验收

最终 gate 必须聚合五个阶段：

```bash
python3 scripts/release/release_regression_hardening_goal_gate.py \
  --out <out_dir> \
  --resource-invariant <resource_invariant_out> \
  --change-impact <change_impact_out> \
  --product-sentinel <product_sentinel_out> \
  --model-contract <model_contract_out> \
  --observability-profile <observability_profile_out>
```

必需最终 PASS line：

```text
RELEASE_REGRESSION_HARDENING GOAL PASS: <out_dir>
```

最终 artifact 必须包含：

- `goal_manifest.json`
- 五个阶段的 artifact 路径和 PASS line
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
- observability profile summary
- replay bundle index

最终量化通过标准：

- 5/5 阶段 PASS。
- 0 个 unknown impact domain。
- 0 个 invalidated-but-counted-as-pass lane。
- 0 个 leaked resources。
- 0 个 silent fallback。
- 0 个 product sentinel failed request。
- 0 个 bad output blocker。
- 0 个 missing request/correlation id。
- 0 个 unclassified failure。
- 0 个 replay bundle failure。
- 至少 1 个 CUDA/Metal shared-runtime fixture 证明 gate planner 会同时要求两端 sentinel。
- 至少 1 个 backend-local fixture 证明 gate planner 不会无脑触发另一端 full gate。

## 依赖关系与修改切片

本目标必须先把依赖关系切清楚，避免先做一轮局部重构，后面接 observability、sentinel、model contract 时再推翻。

### 总依赖图

```text
A. shared schema + offline analyzers + fixtures
   ├── B. resource invariant checker schema
   ├── C. product run/serve observability flags + sink wiring
   │     ├── D. engine/scheduler/KV/recurrent instrumentation
   │     │     ├── E. product/backend sentinel consumes profile artifacts
   │     │     └── F. request dump + failure replay
   │     └── G. backend memory adapters: CUDA / Metal / CPU
   ├── H. change-impact planner consumes changed files + artifact domains
   └── I. model onboarding contract consumes sentinel + replay + runtime preset artifacts

J. final goal gate aggregates A-I only after each phase validator has PASS artifacts.
```

关键约束：

- A 是根。没有统一 schema、analyzer fixture 和 PASS/FAIL 样例，不允许先大面积改 engine instrumentation。
- B 和阶段 4 的 resource/memory event envelope 必须在同一 schema 里设计；不能做两套 `resource_trace.jsonl` 与 `profile.jsonl` 语义。
- C 必须同时覆盖 `ferrum run` 和 `ferrum serve`。只接一个入口会制造新的回归盲区。
- D 只能填充 A 定义的事件，不允许每个模块自造字段。
- E 依赖 C/D/F；product sentinel 失败必须能直接链接 profile event 和 replay bundle。
- I 依赖 E/F；模型合同不应该反向驱动 runtime schema 重构，只能引用已稳定 artifact。

### 必须放在一起改

| 修改组 | 必须一起的原因 | 最小交付 |
|---|---|---|
| Schema + analyzer + fixtures | schema 没有 offline validator 就会变成日志格式，后续每接一个模块都可能重改 | `FerrumProfileEvent`、resource/memory/request 子结构、pass/fail JSONL fixtures、`analyze_ferrum_profile.py` 自测 |
| Resource event envelope + phase 0 invariant input | OOM/slot/KV/recurrent 与 observability 共用 request/resource/memory 语义，拆开会出现两个 owner/capacity 定义 | `resource` 字段规范、invariant checker 消费同一 JSONL、leak/underflow/defer fixtures |
| `run` + `serve` typed flags/config propagation | release 问题反复出在单入口通过、另一个入口坏；入口配置必须同源 | `--profile-jsonl`、`--profile-detail`、`--memory-profile-jsonl`、`--scheduler-trace-jsonl`、`--request-dump-dir` 两个入口都可用 |
| Scheduler admission + capacity/defer/reject event semantics | defer/reopen、OOM prevention、resource invariant 是同一状态机，拆开容易出现“等待释放”和“直接 OOM”两套行为 | admission decision event、capacity snapshot、defer/reject/reopen reason、unit fixtures |
| Request dump + bad-output classifier + replay command | 只有 bad-output 分类没有 replay，定位仍会回到人工日志；只有 replay 没有分类，gate 不能自动拦截 | bad text/stream/schema/tool failure 分类、request dump、engine-level replay command、fixture |

### 必须分开改

| 修改组 | 分开的原因 | 前置依赖 |
|---|---|---|
| CUDA memory sampler | 后端细节多，容易把 CUDA 语义漏进公共 schema | A 的 `memory` 公共字段稳定 |
| Metal memory sampler | Metal 可见内存能力与 CUDA 不同，不能被 CUDA profiling 设计绑死 | A 的 `memory` 公共字段稳定 |
| Prometheus/OTel/NVTX export | 它们是导出层，不是 hard gate 输入；先做会分散语义来源 | A/C/D 稳定，JSONL analyzer 已通过 |
| Product sentinel scenario 扩展 | sentinel 应消费已稳定 artifact，而不是边跑场景边定义日志格式 | A/C/D/F |
| Model contract pilot | 模型合同应引用稳定的 run/serve/replay evidence，不应推动 runtime 重构 | E/F 与 runtime preset snapshot 稳定 |
| Final aggregator | 只聚合 PASS artifacts，不承载业务逻辑 | A-I 的阶段 gate 都存在 |

### 防二次重构规则

- 新字段必须先进入 schema fixture，再进入 runtime emitter。
- 一旦 D 开始接 engine instrumentation，A 的 breaking schema change 必须写 migration note，并同步更新 analyzer、fixtures、阶段 0/4 文档。
- backend 专有信息只能进入 `backend_detail`；公共 analyzer 需要的字段必须先抽到公共 schema。
- `run` 和 `serve` 任何一个入口缺 profile wiring，相关 PR 不能声称 product observability 完成。
- product sentinel 不允许解析非结构化日志来弥补 profile schema 缺字段；缺字段必须回到 A 修 schema。
- 模型合同不得要求新增临时 profile 字段；需要新证据时先扩展 A，再更新 E/F，最后更新 I。

## 执行顺序

推荐顺序：

1. A：先定 shared schema、offline analyzer、fixture，锁住事件语义。
2. 阶段 1：做 change-impact classifier 和 gate planner，让后续工作不再靠人工选择 gate。
3. 阶段 0：做资源 invariant，但输入必须复用 A 的 resource event envelope。
4. C/D/F：接 `run`/`serve` observability wiring、engine/scheduler/KV/recurrent instrumentation、request dump/replay。
5. 阶段 2：把便宜 product/backend sentinel 接到 planner，并消费 profile/replay artifact。
6. G：按共享 memory schema 分别补 CUDA/Metal memory sampler。
7. 阶段 3：把模型接入合同接到 README/support matrix 和 release gate，引用稳定 artifact。
8. J：最终 aggregator 只聚合各阶段 PASS artifact。

阶段 1 可以先落地 dry-run，不阻塞开发；一旦 fixture 覆盖完整，再改成 release 候选硬门禁。

## Work packages

本目标必须拆成小 PR，避免把 release gate、runtime 行为、模型合同和测试架构混在一个 patch。

### WP1: Shared observability/resource schema and analyzer fixtures

范围：

- `crates/ferrum-types/src/observability_profile.rs`
- `crates/ferrum-types/src/resource_trace.rs`
- `scripts/release/analyze_ferrum_profile.py`
- `scripts/release/fixtures/observability_profile/`
- synthetic pass/fail JSONL fixtures

要求：

- 先定义公共 envelope，再接任何 runtime emitter。
- resource、memory、request、latency、token、stream、error 子结构必须一次性设计清楚。
- analyzer 必须能在没有模型、没有 GPU 的情况下验证 fixture。

不允许修改 engine/runtime 行为。

完成标准：

- analyzer 自测通过。
- 所有 schema fixture pass/fail 结果符合预期。
- 缺 `request_id`、缺 `duration_us`、缺 memory before/after、缺 resource owner 的 fixture 必须 fail。

### WP2: Planner dry-run

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

### WP3: Resource invariant checker on shared envelope

范围：

- 复用 WP1 的 resource event/schema。
- offline checker。
- synthetic JSONL pass/fail fixtures。

不允许改 engine admission 策略。

完成标准：

- checker 能发现 leak、underflow、overcommit、defer-with-commit、rollback incomplete。
- `RESOURCE INVARIANT GATE PASS` 可以在 synthetic fixtures 上通过。

### WP4: Product observability wiring for both entrypoints

范围：

- `ferrum run` typed profile/replay flags。
- `ferrum serve` typed profile/replay flags。
- shared config propagation into engine/server。
- artifact manifest records profile paths and profile detail level。

必须把 `run` 和 `serve` 放在同一个 WP。只接一个入口不允许合并为完成。

完成标准：

- `run` 和 `serve` 都能写出同一 schema 的 empty/no-weight 或 synthetic profile artifact。
- profile 关闭时产品行为不变。
- release gate 不依赖 hidden env-only 开关。

### WP5: Engine, scheduler, KV, and recurrent instrumentation

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

### WP6: Request dump, bad-output classifier, and replay

范围：

- sanitized request dump。
- prompt/output token id dump。
- runtime effective config dump。
- bad text/stream/schema/tool failure classifier。
- engine-level replay command。
- serve request replay command。

必须和 classifier 一起交付；不能只 dump、不分类，也不能只分类、不能 replay。

完成标准：

- bad-output、OOM/admission、panic/error synthetic fixtures 都产出 replay bundle。
- replay command fixture 100% 通过。
- `OBSERVABILITY PROFILE GATE PASS` 覆盖 replay bundle 校验。

### WP7: Product/backend sentinel manifest consumes profile artifacts

范围：

- `scripts/release/scenarios/product_backend_sentinel.json`
- `run_scenarios.py` 必要扩展。
- synthetic bad-output/SSE selftests。
- profile/replay artifact 引用。

不允许新增第二套 HTTP benchmark 客户端。

完成标准：

- 12 个阶段 2 fixture 全过。
- `PRODUCT BACKEND SENTINEL PASS` 在 synthetic/no-weight 层可验证。
- sentinel failure summary 能链接到 profile event id 和 replay command。

### WP8: Sentinel integration with planner

范围：

- planner 输出 required sentinel。
- release candidate manifest invalidation。
- sentinel artifact 聚合。

完成标准：

- shared runtime fixture 会要求 Metal + CUDA sentinel。
- backend-local fixture 不会无脑要求另一端 full gate。
- 修改已 PASS lane 相关文件会自动 invalidate 该 lane。

### WP9: Backend memory profile adapters

范围：

- common memory snapshot adapter trait/schema。
- CUDA memory sampler。
- Metal memory sampler。
- CPU/no-weight fallback sampler。
- memory profile analyzer summary。

切片规则：

- common trait/schema 必须先做。
- CUDA 与 Metal sampler 可以分 PR，但不能改变公共 schema。
- 后端专有字段只能写入 `backend_detail`。

完成标准：

- `process_start`、`backend_initialized`、`model_loaded`、`profile_run_done`、`cache_allocated`、`first_request_done`、`shutdown` 至少在 synthetic/no-weight fixture 中可验证。
- `available_kv_or_state_bytes < 0` fixture fail。
- external memory delta 超阈值 fixture 标记 diagnostic-only。

### WP10: Model contract schema and validator

范围：

- `model_onboarding_contract.schema.json`
- `model_onboarding_contract_gate.py`
- valid/invalid contract fixtures。

不要求一次性补齐所有现有模型合同。

完成标准：

- 阶段 3 fixture 全过。
- validator 能拒绝 silent fallback、缺 run/serve artifact、缺能力证据。
- contract 引用稳定 profile/replay artifact，不新增临时诊断字段。

### WP11: First real model contract pilot

范围：

- 选择一个已支持且体积小的模型族做 pilot。
- 产出合同、template golden、run/serve smoke artifact。

完成标准：

- `MODEL ONBOARDING CONTRACT PASS: <out_dir>`。
- README/support matrix 引用合同字段，而不是重复手写能力声明。

### WP12: Final aggregator

范围：

- `scripts/release/release_regression_hardening_goal_gate.py`
- 聚合五阶段 artifacts。
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
- 是否新增、修改或消费 observability profile 字段。
- 如果出现 correctness/memory/latency 风险，artifact 是否包含 replay command 或明确说明为什么不需要。

如果 PR 修改了 `crates/` 但 planner 结果是 `docs_only`，PR 必须 fail。

如果 PR 修改了 shared runtime，但没有 Metal/CUDA sentinel 计划，PR 必须 fail。

如果 PR 修改了模型能力、模板、registry、runtime preset，但没有 model contract 或 product sentinel 计划，PR 必须 fail。

如果 PR 修改了 scheduler/admission、KV、recurrent-state、engine request lifecycle、CLI `run`、CLI `serve`、server streaming、memory autosize 或 backend runtime path，但没有 observability profile impact 说明，PR 必须 fail。

如果 PR 新增 profile 字段但没有更新 schema fixture、analyzer 和对应阶段文档，PR 必须 fail。

## 成功后的日常工作流

目标完成后，普通开发流程应该变成：

```bash
python3 scripts/release/plan_gates.py --base origin/main --head HEAD --out <out>/plan
cat <out>/plan/gate_plan.md
python3 scripts/release/run_gate.py unit --out <out>/unit
python3 scripts/release/observability_profile_gate.py --out <out>/observability
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

### Observability/replay 失败

处理顺序：

1. 先看 `observability_profile_summary.json` 的 `first_failure_event`。
2. 如果缺 request/correlation id，先修 schema/emitter，不继续查模型输出。
3. 如果是 memory event 缺字段，先修 memory sampler 或 adapter，不用日志补证据。
4. 如果是 resource lifecycle 不闭合，回到阶段 0 checker。
5. 如果是 bad output 但缺 replay bundle，先补 request dump/replay，再修模型路径。
6. 如果是 debug/full profile 被用于 performance claim，重新跑 basic profile 或正式 performance gate。
7. 在 observability gate pass 前，不把 product sentinel failure 当成已定位。

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
