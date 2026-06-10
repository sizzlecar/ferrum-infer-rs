# 测试体系与平台隔离改造目标

## 状态

草案目标文件，起草于 2026-06-10。

本目标不能因为"测试加了很多"或"某次回归全绿了"就宣称完成。只有最终验证器打印下面这一行，才算完成：

```text
TEST_ARCH GOAL PASS: <out_dir>
```

最终验证器：`scripts/release/test_arch_goal_gate.py`（阶段 0 落地，含 `--self-test`），
汇总验收各 Gate 的 artifact 与 PASS line。

## 痛点与根因（基于代码与历史证据）

### 痛点 1：优化 Metal 改坏 CUDA，优化 CUDA 改坏 Metal

根因不是"代码共享"本身，而是三个具体机制：

1. **共享模型代码里有后端能力分支**。`ferrum-models` 的 forward/decode 路径直接写
   `if B::supports_varlen_qkv()` 这类分支（如
   `crates/ferrum-models/src/models/qwen3_moe/decode_batch.rs`），CUDA 加 varlen
   优化会改变共享代码走哪条分支，另一个后端的分支无人执行、悄悄腐烂。
   实证：`fbdfcfe2 fix(qwen3-moe): per-item fallback when supports_varlen_qkv()=false (#159)`。
2. **共享代码直接读 FERRUM_* 环境变量，且平台默认值不同**。全仓 100+ 个 FERRUM_*
   变量；`ferrum-models` 内 12 处、`ferrum-engine` 内 3 处直接 `env::var`；
   `FERRUM_USE_VLLM_PAGED_ATTN`（CUDA 默认开 / Metal 默认关）、`FERRUM_METAL_PAGED_KV`
   等在共享代码里改变行为。给一个平台调默认值会改变另一个平台走的代码。
3. **验证时机不对称**。PR CI 对 CUDA 只有 `cargo check`；全部 15 套行为级集成测试
   （chat/server/reference_match）都是 Metal-only、`#[ignore]`、nightly。CUDA 行为
   回归只能在发版前租 pod 手工发现 → 修 CUDA → Metal 再红 → 乒乓循环。

### 痛点 2：问题靠手动试出来（多轮第 3 轮 panic、重复输出、乱码）

1. 多轮 KV 状态、EOS/stop、模板这些**状态机 bug 本可单元级暴露**，但全栈行为测试
   全部依赖"下载真模型 + Metal + nightly"，没有任何东西在每次提交时跑真实
   `ContinuousBatchEngine` + 真实模型 forward。engine 现有 103 个测试用的是
   MockModelExecutor，mock 绕过了真实 forward / 真实 paged KV 布局。
2. CUDA 专属 bug 类（多轮第 3 轮 `paged_varlen_attn CUDA_ERROR_INVALID_VALUE`、
   vllm-moe-marlin 乱码、CUDA 13 模板实例化）**当前测试体系覆盖为零**，
   `docs/bench/m3-80pct-goal-2026-05-25/session-2026-05-26/TESTING-GAPS.md` 已如实
   记录 4 类缺口，至今未关闭。
3. kernel 形状边界（kv_len > 共享内存预算 → INVALID_VALUE）没有参数化形状测试。
4. 发布脚本用"真模型行为 + 精确字符串匹配"做断言 → flake → 近期连续 10+ 个
   `test(release): stabilize/harden ... probe` 提交都在事后救火。"产品正确性"
   （确定性可断言）和"模型行为"（统计性）混在一种断言方式里。

### 痛点 3：回归测试没完没了

1. 没有单命令的 README 模型矩阵回归；逐模型 gate 是手工散装脚本。
2. 回归没有按改动爆炸半径分层：改一个 CUDA kernel 和改 engine 调度器触发的回归
   成本一样，全靠人决定跑什么。
3. 乒乓本质是痛点 1（共享代码耦合）+ 痛点 2（平台验证时机不对称）的复利。

## 目标

三个互相支撑的结果，全部有量化验收：

1. **平台隔离**：只动 `backend/cuda/**` 的改动在机制上不可能改变 Metal 行为
   （反之亦然）；跨后端共享代码对后端差异的表达只允许走"trait 方法 + 正确默认
   回退"，不允许 env 分支和 cfg 分支。
2. **问题前置**：多轮 / EOS / stop / 流式 / 并发 / 重复失控 / OpenAI wire 这些
   历史 bug 类，在每次 `cargo test --workspace`（无 GPU、无下载、≤90 秒的全栈
   tiny-model 套件）就能暴露；用真实历史 bug 回杀率证明套件不是摆设。
3. **回归提速**：按爆炸半径分层（L0 / L1-metal / L1-cuda / L2 矩阵），每层一条
   命令、有时间预算、有 PASS line；README 模型矩阵 100% 自动化覆盖。

## 当前基线（2026-06-10 量化）

| 维度 | 现状 |
| --- | --- |
| 单元测试 | 约 565 个，0 个默认 ignore；`ferrum-interfaces` 为 0 |
| 全栈行为测试 | 15 套 ferrum-cli 集成测试，全部 `#[ignore]` + 需真模型 + Metal-only，nightly 跑（约 45 min 预算） |
| CUDA 行为测试 | **0**（PR CI 仅 `cargo check --features cuda`；无任何自动化 CUDA 推理测试） |
| 跨后端算子对拍 | `ferrum-testkit::op_diff` 仅 1 个算子（rms_norm） |
| models+engine 后端 cfg 分支 | 44 行（含 multimodal；LLM 主路径关键分支约 8 处） |
| models+engine 直接 `env::var` | 15 处（models 12 + engine 3） |
| FERRUM_* 环境变量总面 | 100+ 个 |
| Backend trait 面 | 核心 62 方法 + BackendPagedKv 12 方法，约 23 个默认 unsupported |
| backend_boundary_audit | 已存在但只审计字符串后端分支，不审计 env 读取；allowlist 2 条 |
| README 模型 | 8 个架构家族、13+ 变体；PR 期真模型覆盖 3 个（仅 Metal） |
| 历史 bug 回杀机制 | 无 |
| 多轮第 3 轮 CUDA panic / marlin 乱码 / kv_len 边界 | TESTING-GAPS.md 记录在案，无测试覆盖 |

## 非目标

- 不在 GitHub Actions 搭常驻 GPU runner。CUDA lane 仍是按需 pod，但必须一条命令、
  有预算、有 PASS line。
- 不追求行覆盖率指标。度量单位是"场景、算子、历史 bug 回杀"，不是 lines covered。
- 不重写性能测试体系。`docs/bench/PLAYBOOK.md`、`ferrum-bench-core`、各性能 goal
  gate 不动。
- 不消灭 FERRUM_* 环境变量本身（调参旋钮保留），只消灭 **LLM 主路径共享代码里的
  直接读取**，统一收口到 `ferrum-types` 的类型化 runtime 配置解析。
- multimodal（Whisper / TTS / CLIP）第一期只进矩阵 smoke，不做指纹回归。
- 不为凑回杀率注入人造缺陷。回杀清单只允许 revert 真实历史修复（见回杀清单节），
  禁止 `if model.contains("X")` 式反模式注入。

## 设计原则：收紧既有机制，不另起炉灶

| 既有资产 | 本目标的动作 |
| --- | --- |
| `scripts/release/run_gate.py` lane 体系 | L1/L2 作为新 lane 或收紧现有 `metal` / `cuda-smoke` lane，沿用 manifest + PASS line 约定 |
| `scripts/release/backend_boundary_audit.py` | 扩展审计面：增加 `env::var("FERRUM_*")` 与 `cfg(feature = "cuda"/"metal")` / `target_os` 在 LLM 主路径的检查 |
| `ferrum-testkit::op_diff`（1 算子） | 扩成算子对拍矩阵（manifest 驱动） |
| `reference_match.rs` 指纹机制 | 从 Metal-only 扩到 CUDA + 更多模型 |
| `TESTING-GAPS.md` 4 项提案 | 全部纳入（CUDA 多轮 lane、kernel 形状测试、mixed batch、CUDA 指纹） |
| testkit Mock 体系 | 保留（调度器/engine 逻辑层用）；新增"真 forward"tiny-model 层与之并存 |
| `moe_bucketed_parity_test.rs` 的随机权重模式 | 作为 tiny-model 构造器的种子实现 |

**能力回退律**（本目标的核心架构规则）：共享代码中后端差异只允许两种表达——
(a) 构造期由类型化 capability/placement 决策一次；(b) trait 方法调用，其默认实现
必须是"由已有原语组合出的正确慢路径"，后端用融合 kernel 覆写（本仓已有范例：
`Backend::write_f32_to_activation` 默认 `from_slice + copy_slice`，CUDA 覆写为
直写）。decode/prefill 热路径里不允许出现 `if B::supports_*()` 行为分支。

## 实现计划

每阶段可独立合并，每阶段有自己的验证命令。阶段 0–3 零 GPU 费用。

### 阶段 0：基线固化与验收工装（本地）

- 落地 `scripts/release/test_arch_goal_gate.py --self-test`。
- 产出 `docs/goals/test-architecture-2026-06-10/baseline.json`：上表全部计数的
  机器可重算版本（audit 扩展模式 dry-run 输出 + 各 lane 实测耗时）。
- 固化 `historical_bugs.json` + 每条 bug 的最小复现 patch（从修复 diff 反推，
  人工 review 确认等价于历史缺陷本身，不是人造反模式）。
- 固化算子对拍 manifest `conformance_ops.json`：枚举 LlamaFamily + Qwen3MoE
  decode/prefill 主路径全部算子（rms_norm、fused_add_rms_norm、rope/qk_norm_rope、
  split_qkv_norm_rope(±paged ±varlen)、embedding_lookup、linear(f16 + marlin int4)、
  fused_silu_mul、residual_add、paged append/decode/varlen attention、
  moe route_topk_softmax / align_block_size / combine、argmax_rows_f16、
  int8 kv 变体；预计 15–18 项），含每算子数值容差。
- 验证：`python3 scripts/release/test_arch_goal_gate.py --self-test` 通过；
  baseline.json 数字与本文一致。

### 阶段 1：tiny-model 全栈套件（问题前置的主体）

- testkit 新增确定性 tiny 模型构造器：随机种子权重的 LlamaFamily（hidden≈64、
  2–4 层、vocab≈128）+ tiny Qwen3MoE（复用 `synth_gptq` 模式），CPU 后端，
  毫秒级 forward，无下载。
- 公共化既有私有测试零件：`ParityLoader`（现为 `llama_family_pipeline.rs` 测试
  模块私有）与 `synth_gptq` 上移到 testkit，统一为 tiny 模型构造 API，
  禁止各测试文件再自造合成 loader。
- 进程内组合点重构：从 `ferrum-cli/src/commands/serve.rs::execute` 提取
  `build_app(注入依赖) -> Router` 形式的组装函数（autosizer / env 解析 / 模型
  解析留在 CLI 层），使场景 9 的 OpenAI wire 断言进程内运行，不再 spawn
  子进程加载真模型。
- 新增全栈场景套件（真实 `ContinuousBatchEngine` + 真实模型 forward + 真实
  sampler + 真实 paged KV，**无 mock**），进 `cargo test --workspace` 默认执行：
  1. 多轮对话 ≥5 轮（KV 累积状态，对应历史 turn-3 panic 类）
  2. EOS 终止
  3. stop_sequences（含复合 token 内的 stop 文本，对应 `2a3a6332` 类）
  4. 流式 chunk 契约（chunk 数、finish_reason、usage）
  5. 并发 ≥3 会话隔离
  6. 取消/断连
  7. 重复失控守卫（构造倾向重复的 tiny 权重，断言 max-tokens/重复检测路径）
  8. guided/tool 约束解码（控制 token 掩码，对应 `b2ec992c` 系列）
  9. OpenAI wire：`/v1/models` 非空、空 messages → 400、usage 字段
  10. KV 容量边界与抢占（容量收紧到刚好不够，断言不 panic、合理排队/报错）
- 验证：套件全绿；`cargo test --workspace` 总耗时增量 ≤ 90 秒（M1，release 无关，
  debug 模式计）；场景清单作为 gate manifest 机械核对测试名存在。

### 阶段 2：算子对拍矩阵 + 能力回退律测试

- op_diff 按 `conformance_ops.json` 扩展：每算子 CPU 参考实现 ×（Metal、CUDA 各自
  声明支持的）对拍，固定 shape 网格 + 边界 shape（含 kv_len > 共享内存预算类）。
  Metal 部分本地跑；CUDA 部分进 L1-cuda lane。
- CUDA kernel 启动参数计算提为纯函数（launch plan：grid/block/shared-mem 尺寸
  由可在任意平台编译的纯函数给出，launch 调用只消费 plan），使形状边界断言
  （含 kv_len > 共享内存预算）降级到 L0 在 Mac 上跑；L1-cuda 只保留真实启动冒烟。
- 能力回退律测试：CPU 后端（全部可选 capability=false）跑通阶段 1 全套件 →
  证明共享代码每条 capability 分支都有合法回退（对应 #159 类）。
- 热路径 capability 分支清理为 trait 默认方法（能力回退律），
  `decode_batch.rs` 等处的 `if B::supports_varlen_qkv()` 行为分支迁入 trait。
- 验证：manifest 覆盖率 = 100%（基线 1/15+）；CPU 全栈套件绿；
  Metal `reference_match` 指纹 0 漂移（行为不变性证据）。

### 阶段 3：主路径去耦（平台隔离的主体）

- `ferrum-models` / `ferrum-engine` LLM 主路径的 15 处直接 `env::var` 全部迁移到
  `ferrum-types` 类型化 runtime 配置（沿用既有 auto_config/preset 机制），模型代码
  只见类型化字段。
- **硬约束**：迁移后的配置必须构造期传参、随对象流动；禁止迁移成另一个
  `OnceLock`/进程级全局访问器（现状如 `llama_batched_runtime_config()` 的
  `static CONFIG: OnceLock` 会把配置进程级冻结——一个测试进程只能存在一种配置，
  这是 `--test-threads=1` 无法解除的根因）。audit 增加对该模式的检查。
- LLM 主路径 cfg 后端分支迁入 backend crates 或构造期 placement 决策。
- `backend_boundary_audit.py` 扩展：增加 env 读取与 cfg 分支审计模式，审计范围
  `crates/ferrum-models/src/models/**`、`crates/ferrum-engine/src/**`
  （multimodal/TTS executor 路径进 allowlist，标注 review_condition）。
- 验证：扩展版 audit 计数 = 0（allowlist 外）；Metal 指纹 + CPU tiny 套件 0 漂移；
  `BACKEND BOUNDARY AUDIT PASS` 照常。

### 阶段 4：回归分层与 README 矩阵（回归提速的主体）

- **L1-cuda 一条命令落地**（TESTING-GAPS 提案兑现）：基于 `run_gate.py` 新增/收紧
  cuda lane，内容 = CUDA 多轮 chat（≥5 轮，0.5b GPTQ 级模型）+ server smoke +
  CUDA 版 `reference_match` 指纹 + 算子对拍 CUDA 部分 + kernel 形状边界测试。
- **L1-metal 收紧**：现有 7 套 metal `--ignored` 套件 + 指纹，确认单命令入口。
- **README 模型矩阵 runner**：`scripts/release/readme_model_matrix.py`，模型清单
  来自检入的 manifest（README 支持表由 manifest 生成，杜绝文档与回归脱钩），
  每模型 × 每声明平台 跑 {pull, run 3 轮多轮, serve smoke, 贪心指纹}，逐模型
  PASS/FAIL artifact，支持断点续跑。
- **爆炸半径触发规则**写入 CLAUDE.md / AGENTS.md 并配路径分类脚本
  （L1-metal 本地零成本，合并前直接跑；L1-cuda 攒批执行，见下面协议）：
  - 仅 `backend/cuda/**` → L0 必须；记入 L1-cuda 攒批范围
  - 仅 `backend/metal/**` → L0 + L1-metal
  - `models/ engine/ scheduler/ sampler/ server/`（共享面）→ L0 + L1-metal；记入 L1-cuda 攒批范围
  - 其余 → L0

**L1-cuda 攒批协议**（2026-06-10 拍板：攒批统一执行，不逐 PR 开 pod）：

- 合并门槛只到 L0 + 本地 `cargo check --features cuda`（沿用既有纪律）。
  L0 中的 launch-plan 形状数学、算子 CPU 参考对拍、能力回退律负责把大部分
  CUDA 可达错误先挡在本地。
- 攒批范围不维护清单文件：由路径分类脚本从"上次 L1-cuda PASS 的 SHA
  （gate artifact 记录）..HEAD"自动推导出涉及 CUDA 爆炸半径的提交集。
- 触发批跑（先到先触发）：
  1. 攒批范围累计 5 个 CUDA 相关 PR；
  2. 距上次 L1-cuda PASS 超过 7 天且范围非空；
  3. 发版前（范围必须清空）。
- 批跑 = 一个 pod session 跑一次 L1-cuda 单命令（≤ 4 GPU 时）。
  绿 → artifact 记录新 PASS SHA；红 → 在批内提交范围 bisect
  （L1-cuda 即判据），定位肇事 PR 后 revert 或 fix-forward，复跑确认。
- 已知代价与监控：main 上的 CUDA 隐性破损窗口最长约一周，可接受的前提是
  L0 先行兜底。若连续 3 次批跑中红 ≥ 2 次，说明 L0 网眼不够，
  收紧触发条件（5 PR → 3 PR）并把漏掉的 bug 类补进 L0。
- 常态成本预期：每 1–2 周一个 session，≤ 4 GPU 时。

- 验证：各 lane 实测耗时 ≤ 预算（见 Gate C）；矩阵在两平台各完整跑通一次并存档；
  攒批协议由分类脚本 self-test 覆盖（含范围推导与触发条件判定）。

### 阶段 5：回杀验证与稳定性（证明体系有效）

- 回杀跑批：逐条应用 `historical_bugs.json` 的复现 patch，断言映射 lane 变红
  （CPU 类本地跑；CUDA 类合并进一次 pod session）。
- 稳定性跑批：同一 SHA 上 L0 连续 10/10 绿、L1-metal 10/10 绿、L1-cuda 同 pod 3/3 绿。
- 模型行为型 probe（发布脚本里的 recall/tool 类）改造为 quorum 断言
  （N≥3 取多数）或结构化断言，单次精确字符串匹配不再直接 FAIL。
- 验证：Gate B2 / C3 达标 → 跑最终验证器，打印 `TEST_ARCH GOAL PASS`。

## 验收 Gate（量化）

### Gate A：平台隔离

| # | 指标 | 基线 | 验收值 | 机械检查 |
| --- | --- | --- | --- | --- |
| A1 | LLM 主路径直接 `env::var("FERRUM_*")` 处数 | 15 | **0**（allowlist 外） | 扩展版 backend_boundary_audit |
| A2 | LLM 主路径后端 cfg 分支行数 | 约 8 处关键分支（全量 44 行含 multimodal） | **0**（allowlist 外，multimodal 入 allowlist） | 同上 |
| A3 | 热路径 `if B::supports_*()` 行为分支 | ≥4 处 | **0**（迁为 trait 默认方法或构造期决策） | 同上（新审计模式） |
| A4 | 算子对拍矩阵覆盖 | 1/15+ | **manifest 100%**，每算子在每个声明支持的后端有 parity 测试且绿 | gate 解析 `conformance_ops.json` 对照 `cargo test -- --list` 与测试结果 |
| A5 | CPU 全 capability=false 跑通全栈场景套件 | 不成立 | 成立 | L0 套件结果 |
| A6 | 单后端改动隔离 | 无机制 | 仅动 `backend/cuda/**` 的变更：Metal 指纹 fixtures 0 改动且 Metal lane 绿（反向同理） | 路径分类脚本 + 回杀清单中 2 个真实后端 PR 重放验证 |

### Gate B：问题前置

| # | 指标 | 基线 | 验收值 | 机械检查 |
| --- | --- | --- | --- | --- |
| B1 | tiny-model 全栈场景套件 | 0 | 10 场景全部存在且绿；运行于 `cargo test --workspace`；增量耗时 ≤ 90s | gate manifest 核对测试名 + 计时 artifact |
| B2 | 历史 bug 回杀率 | 无机制 | CPU 可达类 **≥ 80%**（≥6/7）被 L0 抓红；CUDA 专属类 **3/3** 被 L1-cuda 抓红 | 回杀跑批 artifact（每条：patch、lane、红的测试名） |
| B3 | TESTING-GAPS 4 类缺口 | 0 覆盖 | 每类 ≥1 个测试：CUDA 多轮、kv_len 共享内存边界、mixed prefill+decode CUDA、CUDA 乱码指纹 | L1-cuda 套件清单 |
| B4 | kernel 形状边界测试 | 0 | paged_varlen_attn 等入口有参数化 shape 测试，含历史 INVALID_VALUE 触发 shape；launch-plan 数学部分在 L0 跑 | L0（launch-plan 纯函数）+ L1-cuda（真实启动）artifact |

### Gate C：回归提速

| # | 指标 | 基线 | 验收值 | 机械检查 |
| --- | --- | --- | --- | --- |
| C1 | L0（每次提交，无 GPU 无下载） | 行为级覆盖 0 | `cargo test --workspace` 全绿，M1 上总墙钟 ≤ 10 min（阶段 0 实测后可下修，只许收紧） | 计时 artifact |
| C2 | L1-metal 单命令 | 已有但散装 | 一条命令，热缓存 ≤ 15 min，含多轮 + 指纹 + server smoke | run_gate manifest + 计时 |
| C3 | L1-cuda 单命令 | **不存在** | 一条命令：warm pod ≤ 20 min；冷 pod（含构建 + 拉模型）≤ 60 min | run_gate manifest + 计时 |
| C4 | README 模型矩阵 | 无单命令；PR 期 3 模型 | manifest 驱动单命令；README LLM 模型覆盖 **100%**；每平台声明 ✓ 的模型 100% 在该平台有记录；单卡 4090 矩阵 ≤ 3h、2x4090 大模型子集 ≤ 1.5h、Metal 子集 ≤ 1.5h | 矩阵 artifact（逐模型 PASS/FAIL + 耗时） |
| C5 | 稳定性（反 flake） | recall probe 连续 10+ 次救火提交 | 同一 SHA：L0 10/10、L1-metal 10/10、L1-cuda 同 pod 3/3 全绿；行为型 probe 全部 quorum/结构化断言 | 稳定性跑批 artifact |
| C6 | 爆炸半径规则 | 口头约定 | 路径分类脚本 + CLAUDE.md/AGENTS.md 成文 | 脚本 self-test |

最终验证器要求 A/B/C 全部 Gate 的 artifact 齐备且在同一 base SHA 的干净工作区
产出，然后打印 `TEST_ARCH GOAL PASS: <out_dir>`。

## 历史 Bug 回杀清单（B2 的输入）

回杀方式：从修复 diff 反推最小复现 patch，人工 review 确认与历史缺陷等价。
禁止人造反模式注入。

| # | 历史 bug | 修复证据 | 可达层 | 期望抓红的 lane |
| --- | --- | --- | --- | --- |
| 1 | chat REPL EOS / stop_sequences / stream / pos_offset 全断 | `34c57c2b` (#188) | CPU | L0 |
| 2 | 复合 token 内 stop 文本漏检 | `2a3a6132` | CPU | L0 |
| 3 | guided decoding 控制 token 未掩码（系列） | `b2ec992c` `3190dbc7` `bf4573d3` | CPU | L0 |
| 4 | tool 约束未强制 full logits | `fd8e99e0` | CPU | L0 |
| 5 | forced tool 参数字符串无界 | `e7badefc` | CPU | L0 |
| 6 | `/v1/models` 空、空 messages→200、stop 字符泄漏、贪心非确定 | PR #8 server smoke 发现（memory 2026-05-19） | CPU | L0 |
| 7 | `supports_varlen_qkv()=false` 缺 per-item 回退 | `fbdfcfe2` (#159) | CPU（回退律） | L0 |
| 8 | ensure_scratch realloc 后 paged_batch scratch 失效 | `4701cc84` (#155) | CPU/形状 | L0 |
| 9 | 多轮第 3 轮 `paged_varlen_attn` CUDA_ERROR_INVALID_VALUE | TESTING-GAPS #1（`qwen3_moe_forward_unified.rs:368`） | CUDA | L1-cuda |
| 10 | vllm-moe-marlin 打包不匹配 / CUDA 13 乱码 | `049b3a42` `241dbc0` | CUDA | L1-cuda |
| 11 | kv_len > 24576 共享内存预算 kernel 启动失败 | TESTING-GAPS #2 | CUDA（launch-plan 数学 CPU 可达） | L0（launch-plan）+ L1-cuda |

CPU 可达 8 条（#1–8），验收 ≥ 80% 即 ≥7/8 被 L0 抓红（注：#8 若最小 patch 无法
在 tiny shape 下复现，允许标注豁免并降档为 ≥6/7）；CUDA 专属 3 条（#9–11）要求
3/3 被 L1-cuda 抓红。

## 付费 GPU Lane 约束

按 AGENTS.md 付费 GPU 契约执行。本目标预计 GPU 支出仅在阶段 4–5：

```text
Lane: L1-cuda 落地验证 + 回杀 CUDA 批 + 稳定性 3/3 + 矩阵首跑
预期 pod: 2 个 session（1x 4090 一个；2x4090 大模型矩阵子集一个）
预算上限: 每 session ≤ 4 GPU 时
正确性 gate: 各 lane 自身 PASS line
停止条件: 单 session 超 4h 或同一故障复现 3 次未定位 → 停，回本地分析
```

阶段 0–3 全部本地完成，零 GPU 费用。CUDA 构建在本地先 `cargo check --features cuda`
（CI 容器同款）通过后才允许上 pod（沿用既有 feedback 纪律）。

## Checkpoint 和提交

- 每阶段独立 PR，英文 conventional commit；阶段内不混合 kernel/模型改动与 gate
  改动（AGENTS.md 规则）。
- 阶段 3 的去耦改动必须附"行为不变"证据：Metal 指纹 0 漂移 + L0 全绿，写进 PR。
- 本文件同目录维护 `STATUS.md`（沿用 layer-split goal 约定）：每阶段完成后更新
  量化基线表的"现状"列。

## 待确认问题

1. **L1-cuda 节奏与预算**：已拍板（2026-06-10）——攒批统一执行，
   协议见阶段 4 "L1-cuda 攒批协议"。
2. **tiny GPTQ INT4 fixture 可行性**：marlin/量化路径要进 L0 需要一个极小 GPTQ
   模型（随机权重离线量化，几 MB，检入或测试期生成）。阶段 1 先验证可行性；
   不可行则该路径只在 L1-cuda 覆盖，Gate B 相应条目标注。
3. **README 与 manifest 的生成方向**：已拍板（2026-06-10）——方案 A，
   `models.json` manifest 作为单一事实源，README 支持表由脚本生成；
   阶段 4 的矩阵 runner 与 README 生成共用同一 manifest。
4. **L0 总耗时上限**：565 个存量测试 + 新套件在 M1 的实测值待阶段 0 锁定；
   本文先给 10 min 上限，只许收紧不许放宽。
5. **多轮 panic 的当下修复**：回杀条目 #9 的 bug 如果在当前 HEAD 仍可复现
   （TESTING-GAPS 未标记已修），则它先是一个待修 bug、后是回杀条目——阶段 4
   首个 pod session 顺带验证。
