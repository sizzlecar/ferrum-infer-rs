# Release Regression Hardening Goal

## 状态

草案目标文件，创建于 2026-06-28。

本目标用于解决五类反复出现的问题：

- 资源所有权不清导致 OOM、slot 泄漏、defer/rollback 不一致。
- CUDA 与 Metal 相互影响，优化一端改坏另一端。
- 问题发现太晚，经常到 release 前由人工手测暴露。
- release 回归像乒乓球，CUDA 修完回归 Metal，Metal 修完又回归 CUDA。
- 正确性、内存、耗时问题缺少统一 profile/replay artifact，定位常依赖临场日志和人工猜测。
- 主仓库 vendored 大量 FA2/CUTLASS C++/CUDA 源码，普通开发和 CUDA gate 容易被几十分钟编译拖住，也会让以后接入新算子继续复制同样维护成本。

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
| C++ 编译等待 | `fa2-source` 把大量 FlashAttention/CUTLASS C++ 输入带入 CUDA release build，冷编译可达几十分钟；未来新算子若继续 vendored C++ 会重复该问题 | 主仓库不再保存大体量第三方 C++/CUDA 算子源码；FA2 和未来算子统一走 native operator ABI、artifact manifest、resolver、独立 artifact gate |

## 非目标

- 不在本目标里重写 CUDA/Metal kernel。
- 不把 Ferrum 拆成 CUDA engine 和 Metal engine 两套产品路径。
- 不用隐藏环境变量作为产品验收证据。
- 不把 cheap sentinel 当作 full release performance evidence。
- 不用 live vLLM 跑分作为本目标必需项；vLLM 只作为源码/历史行为参考，除非另立性能目标。
- 不在本目标里扩大付费 GPU 矩阵；需要 CUDA 运行时，优先 cheap smoke，full sweep 仍按现有 G0/release 规则执行。
- 不在没有替代正确性/性能证据的情况下删除 FA2 功能；本目标删除的是主仓库里的大体量第三方 C++/CUDA 源码和默认源码编译路径，并用 Ferrum native operator artifact 机制承接 FA2 与未来算子。

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
| native operator ABI/resolver | 新增 `crates/ferrum-native-ops` 或等价模块、`crates/ferrum-kernels/src/native_ops.rs`、native artifact manifest schema | 主仓库只保留 ABI、resolver、manifest schema、small shim 和 fixtures，不保存大体量第三方 C++/CUDA 算子源码 |
| FA2 source migration | `crates/ferrum-kernels/build.rs`、`crates/ferrum-kernels/kernels/fa2_source/`、`crates/ferrum-kernels/src/backend/cuda/fa2_source.rs` | FA2 必须迁移为 native operator artifact；目标完成时主仓库中的 FlashAttention/CUTLASS bulk source 计数为 0 |
| FA2 runtime selection | `crates/ferrum-types/src/auto_config.rs`、`crates/ferrum-models/src/models/qwen3_moe_runtime.rs`、`crates/ferrum-kernels/src/backend/cuda/paged.rs` | runtime opt-in、native artifact manifest、actual model evidence 必须一致；缺 artifact 时必须显式 fallback/reject，不能 silent fallback |

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
- 阶段实际模型回归要求已满足，且 artifact SHA 不早于本阶段最后一次 backend-visible 代码变更。

## 实际模型回归层级

Synthetic/no-weight gate 只能证明 schema、planner、checker 和失败分类，不证明 Metal/CUDA 实际模型路径可用。每个阶段都必须有实际模型触点，但按层级控制粒度，避免把时间全花在重复验证上。

### 层级定义

| 层级 | 用途 | 模型/后端 | 入口覆盖 | 目标耗时 | 适用时机 |
|---|---|---|---|---|---|
| L0 synthetic/no-weight | schema、analyzer、planner、checker、failure fixture | 无真实模型 | 可用 stub/no-weight `run`/`serve` | warm local <= 5 分钟 | 每个 schema/gate PR |
| L1 actual smoke | 证明当前阶段没有打坏真实模型 product path | repo 维护的轻量实际模型，当前默认 `Qwen/Qwen3-0.6B`；Metal/CUDA 各用已支持格式或该模型合同指定 alias | `ferrum run` + `ferrum serve` nonstream/stream | warm local/已有 pod <= 15 分钟；CUDA paid smoke 目标 <= 30 分钟 | 每个阶段 promotion、shared runtime PR、entrypoint PR |
| L2 representative backend | 证明代表性架构没有被阶段改动打坏 | Metal: Llama 8B-class dense + Qwen3 8B/MoE GGUF 路径；CUDA: Qwen3-30B-A3B-GPTQ-Int4 + Llama 8B-class dense | `run` + `serve` + stream + basic concurrency | Metal local按 `metal_readme_regression.py` 预算；CUDA 优先 smoke，不重复 full sweep | 阶段完成前、backend adapter/resource/scheduler 改动后 |
| L3 release full | 官方 release evidence | 现有 G0/release matrix | 完整 G0 要求 | 按 AGENTS/G0 规则 | release candidate |

L1 不是性能证据；它只证明真实模型路径没有立即回归。L2 可以作为阶段完成证据，但仍不能替代 L3 release full。

### 阶段绑定

| 阶段 | L0 | L1 actual smoke | L2 representative backend | 说明 |
|---|---|---|---|---|
| 阶段 0 resource invariant | 必须 | 必须，带 resource trace 跑实际模型 `run`/`serve` | 如果改 admission/KV/recurrent 语义，必须 | OOM/slot/KV/recurrent 不能只用 synthetic capacity 证明 |
| 阶段 1 planner | 必须 | 必须至少执行 planner 产出的一个实际模型 smoke plan | 阶段完成前必须消费最新 L2 artifact manifest 并验证 invalidation | planner 不直接改 runtime，但必须证明计划能驱动真实模型 gate |
| 阶段 2 product/backend sentinel | 必须 | 必须，每次 sentinel runner 或 scenario schema 改动都要跑 L1 | 必须，阶段完成前覆盖 Metal 和 CUDA 各至少一个代表模型 | sentinel 本身就是提前发现真实模型回归的机制 |
| 阶段 3 model contract | 必须 | 每个 pilot 合同必须有实际模型 `run`/`serve` artifact | 如果合同声明 Metal/CUDA support，必须有对应后端 L2 或目标模型 artifact | 不能只靠合同 JSON 声明模型支持 |
| 阶段 4 observability/replay | 必须 | 必须，实际模型 profile/replay artifact 至少覆盖 Metal 和 CUDA | backend memory adapter 完成前必须各有 L2 memory/profile artifact | profile schema 必须证明在真实模型输出、stream 和 memory path 下可用 |

### 粒度规则

- 一个 PR 不应因为改了一个 analyzer fixture 就跑 L2；L0 足够。
- 一个 PR 如果改了 `run`/`serve`、engine lifecycle、scheduler/admission、KV/recurrent、runtime preset、memory autosize、backend adapter、server streaming，至少需要 L1 actual smoke 或明确说明被同一 PR 的更高层 gate 覆盖。
- 同一 coherent slice 内只跑一次 L1，不按文件数量重复跑。例如 schema+entrypoint wiring 是一个 slice，scheduler+resource invariant 是一个 slice，CUDA memory adapter 是一个 slice。
- L2 只在阶段 promotion、backend-visible 语义变化、或多个 L1 failure 指向同一后端风险时运行。
- L3 只在 release candidate 或明确要求 release evidence 时运行。
- 任一 L1/L2 actual model gate 失败后停止扩大验证，先保存 artifact、分类失败、修对应 slice；不要继续跑 full sweep 证明已知失败。

### stale artifact 规则

实际模型 artifact 必须记录：

- git SHA。
- dirty status。
- backend。
- model id/path。
- entrypoints covered。
- command line。
- profile/detail level，若启用。
- PASS line。
- artifact dir。

如果 artifact 的 git SHA 早于最后一次影响同一 backend/model/entrypoint 的代码变更，该 artifact 自动 stale，不能计入阶段完成。

## Native operator artifact governance

FA2 不能只做短期 build 隔离。本目标要求把大体量 C++/CUDA 算子源码从主仓库移出去，并形成后续所有 native 算子复用的一套模式。

一句话方案：

> Ferrum 主仓库只保留 native operator ABI、Rust resolver、manifest schema、small shim 和 gate fixtures；FA2/CUTLASS/未来第三方 C++/CUDA 算子源码不再 vendored 到 `crates/`，而是通过可校验 native artifact 被链接或加载。

当前仓库负担：

- `crates/ferrum-kernels/kernels/fa2_source/flash_attn/src/` 有 98 个 FlashAttention source/header 输入。
- `crates/ferrum-kernels/kernels/fa2_source/cutlass/include/` 有 763 个 CUTLASS header/input。
- 这些输入合计约 647k 行 C++/CUDA/header 内容。
- `crates/ferrum-kernels/build.rs` 当前只编译 Ferrum FA2 shim 加两个 selected FA2 split-hdim128 fp16 instantiation 到 `libfa2_source.a`，但源码和 header 体量仍然很大。
- `fa2-source` 已出现在 CUDA release/smoke feature set 中，例如 `cuda,vllm-moe-marlin,vllm-paged-attn-v2,fa2-source`，因此可能让不关心 FA2 的 CUDA 开发反复支付冷编译成本。

### vLLM 本地构建对照

本节只借鉴本地 vLLM 的 native 扩展治理原则，不引入 Python packaging 依赖。

vLLM 本地源码观察：

- `setup.py` 支持 `VLLM_USE_PRECOMPILED`、`VLLM_PRECOMPILED_WHEEL_LOCATION`、`VLLM_PRECOMPILED_WHEEL_COMMIT`、`VLLM_PRECOMPILED_WHEEL_VARIANT`。
- vLLM 的 precompiled 模式会从 Python wheel（`.whl`，Python 包格式）中提取已经编好的 native `.so`，例如 `_C.abi3.so`、`_moe_C.abi3.so`、`vllm_flash_attn/_vllm_fa2_C.abi3.so`，然后 `precompiled_build_ext` 跳过本地 C++ extension build。
- 源码构建时，vLLM 使用 CMake/Ninja、`MAX_JOBS`、`NVCC_THREADS`、sccache/ccache、`FETCHCONTENT_BASE_DIR`、per-file CUDA arch、component target 来控制构建成本。
- `cmake/external_projects/vllm_flash_attn.cmake` 把 flash-attn 当作独立 external project，并支持 `VLLM_FLASH_ATTN_SRC_DIR` 指向本地源码；FA2/FA3/FA4 也有独立 component。

Ferrum 不能使用 vLLM 的 Python wheel 机制作为依赖。可借鉴的只有原则：

1. 重 native 扩展默认消费已构建产物，不在普通开发路径冷编译。
2. 源码构建是独立 component/domain，有明确输入 hash、输出 artifact、build summary 和耗时预算。
3. 运行时选择必须和编译产物 manifest 对齐；未被 runtime 选择的路径不应强制进入普通 CUDA smoke build。

### Ferrum native operator 模式

新增或等价实现一层 native operator 系统：

```text
crates/ferrum-native-ops/
  src/abi.rs
  src/manifest.rs
  src/resolver.rs
  src/registry.rs

crates/ferrum-kernels/src/native_ops.rs
scripts/release/native_operator_artifact_gate.py
scripts/release/schemas/native_operator_manifest.schema.json
scripts/release/fixtures/native_operator/
```

主仓库允许保留：

- Rust FFI 类型、operator trait、resolver 和 registry。
- 小型 C ABI header 或生成绑定模板，目标是描述 ABI，不携带第三方 kernel 实现。
- manifest schema、fixture、gate script。
- Ferrum 自己的 thin shim；如果 shim 需要 C/CUDA，必须保持小规模，并且由 gate 统计行数和编译耗时。

主仓库禁止保留：

- FlashAttention、CUTLASS、Triton、第三方 CUDA/C++ kernel 的大体量源码树。
- 新算子为了方便直接放入 `crates/**/kernels/<op>/third_party/**`。
- `.whl`、Python package index、Torch/vLLM runtime、Python import 作为构建或运行依赖。
- 没有 manifest、没有 sha256、没有 ABI version 的 native binary。

### Native artifact contract

每个 native 算子 artifact 必须是 Ferrum 原生格式，不是 Python wheel。

允许的 artifact 形态：

| Artifact | 用途 | 规则 |
|---|---|---|
| `ferrum-native-op-<op>-cuda-sm89-<abi>-<hash>.tar.zst` | 官方或本地 native 算子包 | 包含 native lib、manifest、license、build log digest、source package hash |
| `libferrum_native_<op>.a` | release/static link path | 默认推荐；最终 binary 不依赖额外 runtime loader |
| `libferrum_native_<op>.so` | diagnostic/dynamic path | 只用于开发诊断或热切换验证，release 默认不依赖 |
| `native_operator_manifest.json` | artifact contract | resolver 和 gate 的唯一事实来源 |
| release tarball embedded native lib | 官方 CUDA release asset | 必须通过 CUDA tarball gate 验证无 Python/Torch/vLLM runtime linkage |

manifest 最小字段：

```json
{
  "schema_version": 1,
  "operator": "fa2",
  "operator_abi_version": "1",
  "ferrum_native_abi_version": "1",
  "backend": "cuda",
  "cuda_toolkit": "12.4",
  "cuda_runtime_min": "12.4",
  "compute_capabilities": ["sm_89"],
  "source_package": {
    "kind": "external-archive-or-repo",
    "revision": "...",
    "sha256": "..."
  },
  "inputs_sha256": "...",
  "binary_sha256": "...",
  "linkage": "static",
  "exports": ["ferrum_native_op_init", "ferrum_native_op_descriptor"],
  "license_files": [],
  "build_summary": {
    "builder_sha": "...",
    "elapsed_ms": 0,
    "nvcc_version": "...",
    "host_compiler": "..."
  }
}
```

兼容键必须使用 CUDA version、runtime ABI、`sm_xx` compute capability、operator ABI、source/input hash、binary sha256。不能用 “RTX” 品牌名作为 artifact 兼容键；`1x RTX 4090` 只描述当前 G0 硬件事实，对应 native artifact key 是 `sm_89`。

### 源码归属

FA2 和未来 native 算子的第三方源码只能存在于主仓库之外：

- 独立 artifact builder 仓库。
- release artifact source archive。
- 本地 override path，仅用于 developer 重新构建 artifact。

主仓库中的 manifest 必须记录外部 source package 的 revision 和 sha256。没有 hash 的外部源码不能进入 gate。

如果未来需要引入新的 C++/CUDA 算子，必须先新增 operator manifest 和 artifact gate fixture，不能先把源码目录放进主仓库。

### Build and runtime domains

Native operator 必须从普通 CUDA build domain 中拆出来：

| Domain | Feature/build | 是否编第三方 C++ source | 用途 |
|---|---|---|---|
| `cuda-dev` | `cuda,vllm-moe-marlin,vllm-paged-attn-v2` | no | 日常 CUDA smoke、scheduler/resource/model 开发 |
| `native-op-artifact` | consumes manifest-selected `.a`/`.so` | no | 普通 release candidate fast path |
| `native-op-source-build` | external builder only, outside main repo | yes, outside main repo | 生成或更新 native artifact |
| `cuda-release-full` | G0/release lane | no local third-party source compile | 官方 release evidence；只允许消费已校验 artifact |

Planner 规则：

- 如果 diff 不触碰 native operator ABI、manifest、resolver、runtime selection 或 CUDA attention dispatch，不得要求 native operator artifact gate。
- 如果 diff 只触碰 docs/analyzer/schema，不得触发 native operator source build。
- 如果 diff 触碰 native operator ABI/manifest/resolver/runtime selection，必须触发 `NATIVE OP ARTIFACT PASS` 和 actual model L1/L2 CUDA smoke。
- release config 不允许继续依赖 `fa2-source` 作为默认路径；需要 FA2 时必须通过 native artifact manifest。
- 缺 native artifact 时，runtime 必须显式选择 non-FA2 fallback 或返回清晰 unsupported/error，不能 silent fallback 后继续声称 FA2 evidence。

### Migration plan for FA2

FA2 是第一个 native operator 迁移对象。迁移完成不是“隔离 lane”，而是主仓库删除大体量源码。

步骤：

1. **定义 ABI 和 resolver**：新增 native operator manifest schema、resolver、registry、fixture，不改变现有 runtime 行为。
2. **产出 FA2 native artifact**：在主仓库外部 builder 产生 `libferrum_native_fa2.a` 和 manifest，至少覆盖当前 CUDA G0 `sm_89`。
3. **切换 Ferrum 构建**：`build.rs` 不再直接编译 vendored FlashAttention/CUTLASS；改为验证 manifest 并链接 native artifact。
4. **切换 runtime selection**：FA2 runtime path 只在 manifest 匹配且 actual model smoke 通过时启用；否则显式 fallback/reject。
5. **删除主仓库 bulk source**：删除或移出 `crates/ferrum-kernels/kernels/fa2_source/flash_attn/` 和 `crates/ferrum-kernels/kernels/fa2_source/cutlass/`。实际删除前按目录清理规则运行 `scripts/release/inventory_tree.py` 并提交 inventory。
6. **更新 release configs**：CUDA smoke/full 不再依赖本地 `fa2-source` 源码编译；release artifact 记录 native op manifest。
7. **真实模型验证**：用 Qwen3 MoE/GPTQ 或 Llama dense 覆盖 `ferrum run` 和 `ferrum serve`，证明 artifact path 正确性；性能收益用 same-hardware A/B 决定默认是否启用。

### Native operator artifact gate

新增：

```bash
python3 scripts/release/native_operator_artifact_gate.py \
  --out <out_dir>
```

必需 PASS line：

```text
NATIVE OP ARTIFACT PASS: <out_dir>
```

该 gate 必须验证：

- 主仓库内第三方 FA2/CUTLASS bulk source count = 0。
- `crates/**` 下不存在未登记的大体量第三方 C++/CUDA source tree。
- `native_operator_manifest.json` 通过 schema 校验。
- artifact binary sha256、source package sha256、inputs sha256 全部匹配。
- resolver 在 manifest 缺失、compute capability 不匹配、ABI 不匹配、sha256 不匹配时 fail closed。
- 普通 `cargo check --workspace --all-targets` 不调用 nvcc 编译 native operator source。
- release tarball gate 能证明没有 Python/Torch/vLLM runtime linkage。
- FA2 artifact-selected smoke 覆盖 `ferrum run` 和 `ferrum serve`。
- 新增一个 dummy native operator fixture，证明以后新算子能复用同一 ABI/manifest/resolver/gate，不需要复制 FA2 专用逻辑。

量化目标：

| 指标 | 目标 |
|---|---|
| 主仓库第三方 FA2/CUTLASS bulk source | 0 files under `crates/**/fa2_source/flash_attn` and `crates/**/fa2_source/cutlass` |
| 普通 CUDA dev build | native operator source compile count = 0 |
| 非 native-op PR | native operator artifact gate 不触发，除非 planner 证明 runtime selection 被影响 |
| artifact mismatch | 100% fail before product run starts |
| future operator onboarding | 新算子只新增 manifest、resolver registry entry、artifact fixture、actual model smoke，不新增独立 build system |
| Python dependency | 0 `.whl`、0 Python import、0 Torch/vLLM runtime linkage in release gate |
| actual model correctness | artifact-selected FA2 smoke 必须覆盖 `ferrum run` 和 `ferrum serve` |

### FA2 retention decision

FA2 迁移后仍必须做 retention decision，但决策对象变成“是否默认启用 FA2 native operator”，不是“是否保留主仓库 C++ 源码”。主仓库 C++ bulk source 不因 retention decision 保留。

决策输入：

- 最近一次同硬件 FA2 native artifact vs non-FA2 A/B。
- c=1/c=4/c=16/c=32 至少两个 concurrency cells，除非 goal 明确只关注一个 shape。
- correctness 先过，包括 Paris、多轮、stream、bench completion。
- build cost summary，包括 artifact resolve/link time、source artifact build time、prebuilt reuse。
- runtime selection summary，说明默认是否选择 FA2 native operator。

决策输出：

| Decision | 条件 | 后续动作 |
|---|---|---|
| `keep_native_default` | 正确性全过，收益明确，artifact resolver 稳定 | release full 可默认选择 FA2 native operator，但必须记录 manifest |
| `keep_native_diagnostic_only` | 有局部收益但正确性/适配风险仍高 | release full 不默认启用，保留显式 diagnostic preset |
| `drop_fa2_operator` | 收益不足、维护成本高、或长期 stale | 删除 FA2 runtime selection 和 artifact manifest；仍不恢复 vendored source |

在 retention decision 完成前，不允许把 native operator artifact 失败和普通 CUDA correctness failure 混在一起处理；artifact/manifest/ABI 失败应归类为 native operator gate failure。

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

阶段 0 不要求真实 CUDA G0 硬件才能完成；当前 CUDA G0 硬件事实是 1x RTX 4090。CUDA OOM fixture 可以先用 synthetic capacity 模拟；真实 CUDA 证据只作为后续 release sentinel 的补充。

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
| L1 CUDA cheap | CUDA G0 1x RTX 4090 lane | Qwen3 MoE/GPTQ 或 Llama dense 的最小可用 smoke | CUDA/shared/product 变更，付费前按 GPU policy 声明 |
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

### Schema ownership and compatibility

统一 profile schema 是本目标的基础设施，必须有清晰归属，避免后续每接一个模块都改格式。

字段归属：

| 字段区域 | Owner | 修改规则 |
|---|---|---|
| top-level envelope | `ferrum-types` | breaking change 必须 bump `schema_version`，同步更新 analyzer、fixtures、阶段 0/4 文档 |
| `request`/`token`/`stream` | `ferrum-server`、`ferrum-cli`、`ferrum-engine` 共同消费，schema 归 `ferrum-types` | 不能加入入口专有字段；入口差异放 `entrypoint_detail` |
| `resource` | `ferrum-engine`、`ferrum-kv`、`ferrum-scheduler`、recurrent-state manager 共同消费，schema 归 `ferrum-types` | owner/kind/capacity/reserved/committed/released 字段不可省略 |
| `memory` | schema 归 `ferrum-types`，采样实现归 backend adapter | CUDA/Metal 专有字段只能放 `backend_detail` |
| `backend_detail` | backend adapter | 公共 analyzer 不允许依赖该字段才能 PASS |
| `extensions` | experimental only | 只能出现在 diagnostic artifact；required gate 不能依赖 |

兼容规则：

- `schema_version` 必须是字符串，例如 `ferrum.observability.v1`。
- breaking change 必须同时保留上一版 fixture，直到所有 product emitters 迁移完成。
- analyzer 对 unknown top-level field 默认 fail；实验字段只能进 `extensions`。
- required gate artifact 不允许混用多个 schema version，除非 manifest 显式列出 migration validator PASS。
- schema 变更 PR 必须包含至少一个 pass fixture、一个 fail fixture、一个旧版本兼容或拒绝 fixture。

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

隐私和安全规则：

| 数据 | 默认行为 | 允许例外 |
|---|---|---|
| raw user text | 不写入 artifact | 只允许公开 synthetic fixture |
| prompt token ids | 可写入，但必须记录 tokenizer/model id | 若 token ids 可反推出敏感数据，写 hash 或截断 |
| output text | bad-output fixture 可写入；真实请求默认写分类和 span/hash | 用户明确提供公开样例时可写全文 |
| headers | 只保留 trace context、content type、stream flag | auth/cookie/token 一律剔除 |
| request body | 保存 sanitized body 和 replay body | secret-looking 字段必须 redact |
| environment | 保存 sanitized env | secrets、tokens、keys 一律不得进入 artifact |

request dump validator 必须扫描常见 secret key 名称和 token-looking value。发现疑似 secret 时，fixture 必须 fail。

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

- `OBSERVABILITY VERTICAL SLICE PASS: <out_dir>` artifact 存在并被阶段 4 manifest 引用。
- 15 个 fixture 全过。
- `run` 和 `serve` 至少各有一个通过 analyzer 的 artifact。
- L1 actual smoke profile artifact 至少覆盖 Metal 和 CUDA；若当前机器无法运行某后端，必须引用同 SHA 的远端 artifact，且不允许 stale。
- backend memory adapter 声称完成前，必须有对应后端 L2 actual model memory/profile artifact。
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

最终 gate 必须聚合五个质量阶段和 native operator 迁移 gate：

```bash
python3 scripts/release/release_regression_hardening_goal_gate.py \
  --out <out_dir> \
  --resource-invariant <resource_invariant_out> \
  --change-impact <change_impact_out> \
  --product-sentinel <product_sentinel_out> \
  --model-contract <model_contract_out> \
  --observability-profile <observability_profile_out> \
  --native-operator <native_operator_out>
```

必需最终 PASS line：

```text
RELEASE_REGRESSION_HARDENING GOAL PASS: <out_dir>
```

最终 artifact 必须包含：

- `goal_manifest.json`
- 五个质量阶段和 native operator 迁移 gate 的 artifact 路径和 PASS line
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
- observability vertical slice summary
- actual model regression summary
- native operator artifact summary
- FA2 source removal inventory
- replay bundle index

最终量化通过标准：

- 5/5 阶段 PASS。
- `NATIVE OP ARTIFACT PASS: <out_dir>` 被最终 manifest 引用。
- 每个阶段至少有一个未 stale 的 L1 actual model artifact 或更高层级 artifact。
- Metal 和 CUDA 各至少一个 L2 actual model artifact 被最终 manifest 引用。
- 主仓库内第三方 FA2/CUTLASS bulk source count = 0。
- `fa2-source` 不再是普通 CUDA smoke/release source compile 依赖；FA2 如启用，必须通过 native operator manifest。
- 0 个 unknown impact domain。
- 0 个 invalidated-but-counted-as-pass lane。
- 0 个 stale actual-model artifact 被计入 PASS。
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

### 全局依赖图

```text
M0. Evidence foundation
    ├── shared artifact manifest
    ├── shared profile/resource schema
    ├── offline analyzers
    └── pass/fail fixtures

M1. Planning and native-op foundation
    ├── change-impact planner consumes artifact domains from M0
    └── native operator ABI/manifest/resolver consumes artifact manifest from M0

M2. Product observability vertical slice
    ├── run/serve typed profile wiring consumes M0 schema
    └── synthetic/no-weight vertical slice produces first real product artifacts

M3. Runtime diagnostics and cheap gates
    ├── resource invariant checker consumes M0 resource envelope
    ├── engine/scheduler/KV/recurrent instrumentation consumes M2 wiring
    ├── request dump/replay consumes M2 request identity
    ├── product/backend sentinel consumes M2/M3 artifacts and M1 planner
    └── backend memory adapters consume M0 memory schema and M2 wiring

M4. Native-op FA2 migration
    ├── FA2 artifact package consumes M1 native-op ABI/resolver
    ├── build.rs links artifact instead of compiling vendored source
    ├── runtime selection consumes native-op manifest and sentinel evidence
    └── repository removes FA2/CUTLASS bulk source after inventory

M5. Model onboarding contract
    ├── contract validator consumes M1 planner and M3 sentinel/replay
    └── first real model pilot consumes M3/M4 actual-model artifacts

M6. Final aggregation
    └── final goal gate aggregates M0-M5 only after each validator has PASS artifacts.
```

关键约束：

- M0 是根。没有统一 manifest/schema、analyzer fixture 和 PASS/FAIL 样例，不允许先大面积改 engine instrumentation、FA2 build、model contract。
- M1 必须早做。planner 不知道 native-op domain，后续删除 FA2 源码就无法自动要求正确 gate。
- M2 必须先打通 `run` 和 `serve`。只接一个入口会制造新的回归盲区。
- M3 只能填充 M0/M2 定义的事件，不允许每个模块自造字段。
- M4 依赖 M1 native-op ABI/resolver。不能先删除 FA2/CUTLASS 源码再临时设计 artifact 格式。
- M4 的 runtime selection 依赖 M3 sentinel 或等价 actual-model artifact。不能只证明 artifact 能链接就宣称 FA2 可用。
- M5 依赖 M3/M4 的真实模型 evidence。模型合同不应该反向驱动 runtime schema 重构，只能引用已稳定 artifact。
- M6 只聚合 PASS artifacts，不承载业务逻辑。

### 里程碑与阻塞关系

| 里程碑 | 必须先完成 | 允许并行 | 阻塞的后续任务 | 禁止提前做 |
|---|---|---|---|---|
| M0 Evidence foundation | 无 | schema、fixture、analyzer、manifest schema 可以同 PR | 所有 runtime/profile/native-op/model contract 工作 | 大面积 engine instrumentation、删除 FA2 源码、真实模型 release claim |
| M1 Planner/native-op foundation | M0 的 artifact manifest 初版 | planner dry-run 与 native-op ABI/resolver 可并行 | M4 FA2 迁移、M6 final aggregation | 修改 release config 默认走新 native-op path |
| M2 Product vertical slice | M0 profile schema | `run`/`serve` wiring 必须同 WP，不分开 | M3 instrumentation、M3 sentinel、M5 contract | 只接 `run` 或只接 `serve` 后声称完成 |
| M3 Runtime diagnostics/sentinel | M0 + M2；planner dry-run 可先软依赖 | resource invariant、replay、backend memory adapter 可分 PR | M4 runtime selection、M5 model contract、final L2 promotion | sentinel 解析非结构化日志补 schema 缺口 |
| M4 Native-op FA2 migration | M1 native-op ABI/resolver；M3 至少 L1 actual smoke | FA2 artifact packaging、build.rs link、runtime selection、inventory 可分 PR | CUDA release configs、FA2 retention decision、final native-op PASS | 在主仓库继续新增第三方 C++ bulk source |
| M5 Model contract | M3 sentinel/replay；M4 仅当模型声明依赖 native-op | contract schema 和 first pilot 可分 PR | README/support matrix 自动化、final goal gate | 用模型名特判或临时 profile 字段作为合同证据 |
| M6 Final aggregation | M0-M5 全部 PASS | 无 | release-ready claim | final gate 内重新实现业务判断 |

### Work-package 依赖矩阵

| WP | 主题 | Depends on | Blocks | 可并行对象 |
|---|---|---|---|---|
| WP1 | Shared artifact/profile/resource schema | 无 | WP2-WP15 | 无 |
| WP2 | Planner dry-run and domain rules | WP1 manifest/domain schema | WP8、WP9、WP10、WP15 | WP3 |
| WP3 | Native operator ABI/manifest/resolver skeleton | WP1 manifest schema | WP10、WP15 | WP2、WP4 |
| WP4 | Observability vertical slice | WP1 profile schema | WP5-WP9、WP11-WP15 | WP2、WP3 |
| WP5 | Resource invariant checker | WP1 resource envelope | WP6、WP8、WP15 | WP7 common memory schema |
| WP6 | Product observability wiring | WP4 | WP7、WP8、WP9、WP11-WP15 | WP5 |
| WP7 | Backend memory profile adapters | WP1 memory schema + WP4 wiring | WP8、WP11、WP15 | WP5、WP6 |
| WP8 | Engine/scheduler/KV/recurrent instrumentation | WP5 + WP6 | WP9、WP11、WP12、WP15 | WP7 |
| WP9 | Request dump/classifier/replay | WP6 + request identity | WP11、WP12、WP13、WP15 | WP8 |
| WP10 | FA2 native-op artifact migration and source removal | WP2 + WP3 | WP11 CUDA native-op smoke、WP13、WP15 | WP8、WP9 |
| WP11 | Product/backend sentinel consumes artifacts | WP2 + WP7 + WP8 + WP9；CUDA native-op cases also depend on WP10 | WP12、WP13、WP15 | WP10 non-runtime pieces |
| WP12 | Model contract schema/validator | WP2 + WP9 + WP11 | WP13、WP15 | WP10 retention evidence |
| WP13 | First real model contract pilot | WP11 + WP12；native-op models depend on WP10 | WP15 | WP14 |
| WP14 | L2 representative backend promotion matrix | WP10 + WP11 + affected backend adapters | WP15 | WP13 |
| WP15 | Final aggregator | WP1-WP14 PASS artifacts | goal completion | 无 |

### 必须放在一起改

| 修改组 | 必须一起的原因 | 最小交付 |
|---|---|---|
| Schema + analyzer + fixtures | schema 没有 offline validator 就会变成日志格式，后续每接一个模块都可能重改 | `FerrumProfileEvent`、resource/memory/request 子结构、pass/fail JSONL fixtures、`analyze_ferrum_profile.py` 自测 |
| Resource event envelope + phase 0 invariant input | OOM/slot/KV/recurrent 与 observability 共用 request/resource/memory 语义，拆开会出现两个 owner/capacity 定义 | `resource` 字段规范、invariant checker 消费同一 JSONL、leak/underflow/defer fixtures |
| `run` + `serve` typed flags/config propagation | release 问题反复出在单入口通过、另一个入口坏；入口配置必须同源 | `--profile-jsonl`、`--profile-detail`、`--memory-profile-jsonl`、`--scheduler-trace-jsonl`、`--request-dump-dir` 两个入口都可用 |
| Scheduler admission + capacity/defer/reject event semantics | defer/reopen、OOM prevention、resource invariant 是同一状态机，拆开容易出现“等待释放”和“直接 OOM”两套行为 | admission decision event、capacity snapshot、defer/reject/reopen reason、unit fixtures |
| Request dump + bad-output classifier + replay command | 只有 bad-output 分类没有 replay，定位仍会回到人工日志；只有 replay 没有分类，gate 不能自动拦截 | bad text/stream/schema/tool failure 分类、request dump、engine-level replay command、fixture |
| Native operator ABI + manifest + resolver fixtures | 只有 manifest 没有 resolver 无法落地；只有 resolver 没有 fail-closed fixture 会引入 silent fallback | ABI version、manifest schema、resolver fail-closed、dummy operator fixture |
| FA2 artifact link + runtime selection + source removal inventory | 只改 build 不改 runtime 会产生“编了但没用”；只删源码没有 inventory/gate 会破坏目录治理 | `libferrum_native_fa2.*` manifest、runtime selection summary、`inventory_tree.py` artifact、bulk source count = 0 |

### 必须分开改

| 修改组 | 分开的原因 | 前置依赖 |
|---|---|---|
| CUDA memory sampler | 后端细节多，容易把 CUDA 语义漏进公共 schema | WP1 的 `memory` 公共字段稳定，WP4 wiring 已存在 |
| Metal memory sampler | Metal 可见内存能力与 CUDA 不同，不能被 CUDA profiling 设计绑死 | WP1 的 `memory` 公共字段稳定，WP4 wiring 已存在 |
| Prometheus/OTel/NVTX export | 它们是导出层，不是 hard gate 输入；先做会分散语义来源 | WP1/WP4/WP8 稳定，JSONL analyzer 已通过 |
| Product sentinel scenario 扩展 | sentinel 应消费已稳定 artifact，而不是边跑场景边定义日志格式 | WP2/WP7/WP8/WP9 |
| Model contract pilot | 模型合同应引用稳定的 run/serve/replay evidence，不应推动 runtime 重构 | WP11/WP12 与 runtime preset snapshot 稳定 |
| Native operator source artifact builder | builder 可以在主仓库外独立演进；主仓库只消费 artifact/manifest | native operator ABI/manifest 稳定 |
| Final aggregator | 只聚合 PASS artifacts，不承载业务逻辑 | WP1-WP14 的 gate artifacts 都存在 |

### 防二次重构规则

- 新字段必须先进入 schema fixture，再进入 runtime emitter。
- 一旦 WP8 开始接 engine instrumentation，WP1 的 breaking schema change 必须写 migration note，并同步更新 analyzer、fixtures、阶段 0/4 文档。
- backend 专有信息只能进入 `backend_detail`；公共 analyzer 需要的字段必须先抽到公共 schema。
- `run` 和 `serve` 任何一个入口缺 profile wiring，相关 PR 不能声称 product observability 完成。
- product sentinel 不允许解析非结构化日志来弥补 profile schema 缺字段；缺字段必须回到 A 修 schema。
- 模型合同不得要求新增临时 profile 字段；需要新证据时先扩展 WP1，再更新 WP9/WP11，最后更新 WP12/WP13。

### 最小可用纵切

为避免再次出现长时间重构但没有可用产物，本目标必须先交付一个最小纵切，再扩大 instrumentation。

最小纵切不要求真实 CUDA/Metal，不要求真实大模型，但必须经过产品入口和离线 analyzer：

```bash
python3 scripts/release/observability_vertical_slice_gate.py \
  --out <out_dir>
```

必需 PASS line：

```text
OBSERVABILITY VERTICAL SLICE PASS: <out_dir>
```

最小纵切必须产出：

- 一个 `ferrum run` synthetic/no-weight profile artifact。
- 一个 `ferrum serve` synthetic/no-weight profile artifact。
- 一个 pass fixture JSONL。
- 一个 fail fixture JSONL。
- 一个 sanitized request dump。
- 一个 replay command。
- 一个 `observability_profile_summary.json`。
- analyzer 输出 first failure event。

最小纵切量化标准：

- artifact 从产品入口产生，不允许只手写 JSONL。
- `run` 和 `serve` 使用同一个 schema version。
- analyzer 检测到缺 `request_id`、缺 memory before/after、缺 resource owner 至少三类失败。
- replay command 在 synthetic/no-weight 下可执行并返回 0。
- 从 command 开始到 PASS，warm local 环境目标耗时 <= 5 分钟。

在最小纵切 PASS 前，不允许开始大面积改 engine/scheduler/KV/recurrent instrumentation；只允许补 schema、analyzer、fixture、入口 wiring 所需的最小代码。

最小纵切只是 L0 证据。它不能算阶段 4 完成，也不能作为 Metal/CUDA 实际模型回归证据。最小纵切 PASS 后，必须尽快补 L1 actual smoke，避免 profile schema 只适配 synthetic artifact。

## 执行顺序

推荐顺序：

1. WP1：先定 shared artifact/profile/resource/native-op manifest schema、offline analyzer、fixture，锁住证据格式。
2. WP2 + WP3：并行做 gate planner dry-run 和 native operator ABI/resolver skeleton。这里不改 runtime 默认行为。
3. WP4：做最小可用纵切，用 `run` 和 `serve` 产出同 schema artifact，并让 analyzer/replay 跑通。
4. WP5 + WP6：补 resource invariant checker 和 product observability wiring。`run`/`serve` 继续同源。
5. WP7 + WP8 + WP9：补 backend memory adapter、engine/scheduler/KV/recurrent instrumentation、request dump/classifier/replay。
6. WP10：迁移 FA2 到 native operator artifact，并删除主仓库 FA2/CUTLASS bulk source。实际删除前必须有 inventory artifact。
7. WP11：把 product/backend sentinel 接到 planner，并消费 profile/replay/native-op artifact。
8. WP12 + WP13：做模型接入合同 schema 和第一个真实模型 pilot。
9. WP14：跑 L2 representative backend promotion matrix，避免只在最终 release 发现 Metal/CUDA 单边炸。
10. WP15：final aggregator 只聚合 WP1-WP14 的 PASS artifact。

WP2 可以先落地 dry-run，不阻塞开发；一旦 fixture 覆盖完整，再改成 release 候选硬门禁。WP10 不允许抢跑到 WP3 前面，否则会变成“先删源码、后补 ABI”的返工。

## Work packages

本目标必须拆成小 PR，避免把 release gate、runtime 行为、模型合同和测试架构混在一个 patch。

### WP1: Shared evidence schema and analyzer fixtures

Depends on: none.

范围：

- `crates/ferrum-types/src/observability_profile.rs`
- `crates/ferrum-types/src/resource_trace.rs`
- shared artifact manifest schema
- native operator manifest schema draft
- `scripts/release/analyze_ferrum_profile.py`
- synthetic pass/fail fixtures

不允许修改 engine/runtime 行为。

完成标准：

- analyzer 自测通过。
- profile/resource/memory/request/native-op manifest fixtures 的 pass/fail 结果符合预期。
- 缺 `request_id`、缺 `duration_us`、缺 memory before/after、缺 resource owner、缺 native artifact sha256 的 fixture 必须 fail。

### WP2: Planner dry-run and domain rules

Depends on: WP1.

范围：

- `scripts/release/change_impact_rules.json`
- `scripts/release/plan_gates.py`
- planner selftests
- fixture diffs
- native-op impact domain 和 invalidation rules

不允许修改 engine/runtime 行为。

完成标准：

- `CHANGE IMPACT GATE PLAN PASS: <out_dir>`。
- 至少 16 个 fixture 全过，且包含 native-op ABI、native-op manifest、FA2 runtime selection、docs-only 四类 diff。
- planner 能读取 L1/L2 actual model artifact manifest，并正确把 stale artifact 标为 invalidated。

### WP3: Native operator ABI and resolver skeleton

Depends on: WP1. Can run parallel with WP2/WP4.

范围：

- `crates/ferrum-native-ops/` 或等价模块。
- native operator ABI version。
- manifest parser/validator。
- resolver fail-closed behavior。
- dummy native operator fixture。

不允许迁移 FA2，不允许删除 FA2/CUTLASS source。

完成标准：

- manifest 缺失、sha256 不匹配、ABI 不匹配、compute capability 不匹配时 resolver fail closed。
- dummy operator fixture 证明未来新算子复用同一 ABI/manifest/resolver/gate。
- 0 个 Python/wheel/Torch/vLLM runtime dependency。

### WP4: Observability vertical slice

Depends on: WP1. Can run parallel with WP2/WP3 after schema stabilizes.

范围：

- `scripts/release/observability_vertical_slice_gate.py`
- `run` synthetic/no-weight profile artifact。
- `serve` synthetic/no-weight profile artifact。
- minimal request dump。
- minimal replay command。
- `observability_profile_summary.json`。

要求：

- 只做打通链路所需最小入口 wiring。
- 不开始大面积 engine/scheduler/KV/recurrent instrumentation。
- 不引入 CUDA/Metal 专有 memory sampler。

完成标准：

- `OBSERVABILITY VERTICAL SLICE PASS: <out_dir>`。
- warm local 目标耗时 <= 5 分钟。
- `run` 和 `serve` artifact 使用同一个 schema version。
- 明确标记为 L0，不允许作为实际模型回归证据。

### WP5: Resource invariant checker on shared envelope

Depends on: WP1 and WP4.

范围：

- 复用 WP1 的 resource event/schema。
- offline checker。
- synthetic JSONL pass/fail fixtures。

不允许改 engine admission 策略。

完成标准：

- checker 能发现 leak、underflow、overcommit、defer-with-commit、rollback incomplete。
- `RESOURCE INVARIANT GATE PASS` 可以在 synthetic fixtures 上通过。

### WP6: Product observability wiring for both entrypoints

Depends on: WP4.

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
- 至少一个 L1 actual smoke artifact 通过 analyzer，证明 wiring 对真实模型可用。

### WP7: Backend memory profile adapters

Depends on: WP1 memory schema and WP4 wiring.

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
- 每个 backend adapter 完成前必须跑该 backend 的 L2 actual model memory/profile artifact。

### WP8: Engine, scheduler, KV, and recurrent instrumentation

Depends on: WP5 and WP6. Can run parallel with WP7 after common schema is stable.

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
- shared scheduler/resource/KV/recurrent 语义变化必须有 L1 actual smoke；阶段 promotion 前必须有 L2 representative backend artifact。

### WP9: Request dump, bad-output classifier, and replay

Depends on: WP6. Uses WP8 events when available.

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
- 至少一个实际模型 normal-output replay smoke 通过；若出现真实 bad-output，必须能生成 replay bundle。

### WP10: FA2 native-op migration and source removal

Depends on: WP2 and WP3. Runtime selection evidence depends on WP11 when actual model smoke is required.

范围：

- 外部 builder 或 source archive 产生 FA2 native artifact。
- `libferrum_native_fa2.a` 或等价 artifact。
- `native_operator_manifest.json`。
- `build.rs` 改为验证 manifest 并链接 artifact。
- FA2 runtime selection 使用 native-op resolver。
- 删除或移出主仓库 FA2/CUTLASS bulk source。
- 删除前运行 `scripts/release/inventory_tree.py`。

完成标准：

- `NATIVE OP ARTIFACT PASS: <out_dir>`。
- 主仓库内第三方 FA2/CUTLASS bulk source count = 0。
- 普通 `cargo check --workspace --all-targets` 不调用 nvcc 编译 native operator source。
- artifact mismatch 在产品运行前 fail closed。
- release tarball gate 能证明无 Python/Torch/vLLM runtime linkage。

### WP11: Product/backend sentinel consumes artifacts

Depends on: WP2, WP7, WP8, WP9. CUDA native-op scenarios also depend on WP10.

范围：

- `scripts/release/scenarios/product_backend_sentinel.json`
- `run_scenarios.py` 必要扩展。
- synthetic bad-output/SSE selftests。
- profile/replay/native-op artifact 引用。

不允许新增第二套 HTTP benchmark 客户端。

完成标准：

- 12 个阶段 2 fixture 全过。
- `PRODUCT BACKEND SENTINEL PASS` 在 synthetic/no-weight 层可验证。
- sentinel failure summary 能链接到 profile event id 和 replay command。
- L1 actual smoke 是 sentinel PR 的默认 gate；阶段完成前必须有 Metal 和 CUDA 的 L2 representative artifact。

### WP12: Model contract schema and validator

Depends on: WP2, WP9, WP11.

范围：

- `model_onboarding_contract.schema.json`
- `model_onboarding_contract_gate.py`
- valid/invalid contract fixtures。

不要求一次性补齐所有现有模型合同。

完成标准：

- 阶段 3 fixture 全过。
- validator 能拒绝 silent fallback、缺 run/serve artifact、缺能力证据。
- contract 引用稳定 profile/replay/native-op artifact，不新增临时诊断字段。
- 合同声明的每个 backend support 必须引用未 stale 的实际模型 artifact。

### WP13: First real model contract pilot

Depends on: WP11 and WP12. If the pilot relies on FA2/native-op, also depends on WP10.

范围：

- 选择一个已支持且体积小的模型族做 pilot。
- 产出合同、template golden、run/serve smoke artifact。
- 若该模型声明 native-op acceleration，合同必须引用 native-op manifest。

完成标准：

- `MODEL ONBOARDING CONTRACT PASS: <out_dir>`。
- README/support matrix 引用合同字段，而不是重复手写能力声明。

### WP14: L2 representative backend promotion matrix

Depends on: WP10 and WP11; also depends on affected backend memory adapters from WP7.

范围：

- Metal L2 representative model artifact。
- CUDA L2 representative model artifact。
- FA2 native artifact selected/non-selected summary。
- actual model regression summary。

完成标准：

- Metal 和 CUDA 各至少一个 L2 actual model artifact。
- native-op selected path 至少一个 CUDA artifact；如果默认不选择 FA2，必须有 explicit non-selected summary。
- artifact git SHA 不 stale。

### WP15: Final aggregator

Depends on: WP1-WP14 PASS artifacts.

范围：

- `scripts/release/release_regression_hardening_goal_gate.py`
- 聚合五个质量阶段、native-op gate、L2 promotion matrix。
- final selftest。

完成标准：

- `RELEASE_REGRESSION_HARDENING GOAL PASS: <out_dir>`。
- final manifest 能指出每个阶段 artifact、native-op artifact、L2 artifact 和 PASS line。

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
- 是否新增、修改或消费 native operator ABI/manifest/resolver/artifact。
- 是否新增或删除第三方 C++/CUDA source tree；若删除，必须引用 `inventory_tree.py` artifact。
- 如果出现 correctness/memory/latency 风险，artifact 是否包含 replay command 或明确说明为什么不需要。

如果 PR 修改了 `crates/` 但 planner 结果是 `docs_only`，PR 必须 fail。

如果 PR 修改了 shared runtime，但没有 Metal/CUDA sentinel 计划，PR 必须 fail。

如果 PR 修改了模型能力、模板、registry、runtime preset，但没有 model contract 或 product sentinel 计划，PR 必须 fail。

如果 PR 修改了 scheduler/admission、KV、recurrent-state、engine request lifecycle、CLI `run`、CLI `serve`、server streaming、memory autosize 或 backend runtime path，但没有 observability profile impact 说明，PR 必须 fail。

如果 PR 新增 profile 字段但没有更新 schema fixture、analyzer 和对应阶段文档，PR 必须 fail。

如果 PR 修改 native operator ABI、manifest、resolver、runtime selection 或 FA2/CUDA attention dispatch，但没有 `NATIVE OP ARTIFACT PASS` 计划和 actual model smoke 计划，PR 必须 fail。

如果 PR 在 `crates/` 下新增未登记的大体量第三方 C++/CUDA source tree，PR 必须 fail；新 native 算子必须先走 WP3/WP10 的 manifest/resolver/artifact 模式。

## 成功后的日常工作流

目标完成后，普通开发流程应该变成：

```bash
python3 scripts/release/plan_gates.py --base origin/main --head HEAD --out <out>/plan
cat <out>/plan/gate_plan.md
python3 scripts/release/run_gate.py unit --out <out>/unit
python3 scripts/release/observability_profile_gate.py --out <out>/observability
python3 scripts/release/native_operator_artifact_gate.py --out <out>/native-op
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
