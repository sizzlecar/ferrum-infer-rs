# Ferrum SLO 恢复实施方案

2026 年 10 月 2 日，用户要求在充分理解原目标、现状和连续失败经验后形成方案，并已要求创建和启动 goal。目标状态为 active。本方案从 `62b92fb1473bf11f93703d7db41c0dbf9c32ba65` 继续，集成分支为 `slo/recovery-20261002`。

当前仍未交付完整的 SLO 自动闭环版本。首先验证原预算下的覆盖和规划可行性，再完成正常入口的自动闭环，随后进行双后端、完整性能和发布验收。每一步都必须产出可复现的行为证据；局部测试、成本快照数量和源码规模不作为完成度。

## 目标与验收依据

完整功能范围沿用 [原实施清单](slo-implementation-plan.zh.md) 和 [算法设计](slo-algorithm-design.zh.md)。这两份文档保存的是历史进度，其中相对 llama.cpp 的旧胜负硬门已被后续要求覆盖。

最新范围通过交接和 `performance-execution-plan-r5-preparation` 中的机器可读规则恢复。其 `primary-matrix.candidate.json` 的 `execution.promote_default_gate` 明确保留绝对 SLO、自身吞吐与延迟回退限制，删除相对 llama.cpp 的胜负硬门。交接引用的原 `GOAL.zh.md`、`RFC-v2.md` 和历史 `PROGRESS.zh.md` 当前无法读取；本方案是依据现存材料的重建，不冒充找回了原文件。后续发现原文时核对差异，用户最新指令始终优先。

| 范围 | 必须完成的行为 |
| --- | --- |
| K0 执行效率 | 以真实热点证据选择算子和执行路径优化；保持模型精度、KV 和输出语义；基础执行收益与调度收益分开测量 |
| R0 基线与观测 | 源码、构建、配置和证据可追溯；低成本默认观测与重型诊断成本分开 |
| M0 指标契约 | 入口时间、TTFT、TPOT、可见文本 ITL、联合达标、错误分母、开放负载和队列证据准确 |
| M1 有界执行 | 实际 wave 工作上限、prefill 分段、split/mixed、提交和部分失败回执、资源生命周期成立 |
| M2 输出隔离 | 输出额度有界；慢消费者、断连和取消不阻塞其他请求；流式与非流式终态正确 |
| M3 成本模型 | 自动准备和在线学习、独立资格验证、有效快照发布和更新、覆盖与 Unknown、过期和漂移；Observe 保持原调度语义 |
| M4 规划与采用 | 有限合法候选、共同可行计划、有界前瞻、独立提交校验；只提交当前第一波，并依据真实回执继续 |
| M5 时间准入 | 动态准入、有界等待、独立时间唤醒、prefix 等待、恢复与维护、过载时公平进展 |
| 交付 | 正常 run 和 serve、双后端验证、完整比较与容量/消融、发布产物和准确的中英文文档/官网 |

正常用户通过 typed 配置使用功能，不需要先手工训练或导入成本文件。默认完成优先：成本 Unknown、预测失约和搜索失败不能自动取消请求、缩短输出或永久等待。实际接收容量、上下文限制和显式请求超时/取消仍按各自合同处理。

以下边界保持：默认规划预算 2ms；自动成本探测 120s、2048 个请求、16384 offered waves；原 Fitting、Residual、Qualification 独立阶段及其资格要求；模型精度、数值策略和完整输出政策。不通过扩大预算、降低资格门或删慢样本制造成功。配置原本允许的显式选择仍保留其产品语义。

| 正式客户端门槛 | CUDA | Metal |
| --- | --- | --- |
| TTFT P99 | 200ms | 3400ms |
| TPOT P99 | 15ms | 212ms |
| 非空可见 SSE 文本事件 ITL P99 | 50ms | 359ms |
| 同构建 Ferrum Off 的 successful output TPS | 至少 95% | 至少 95% |
| 自身延迟回退 | 同一分位在两轮均恶化超过 10%，不提升为默认 | 同左 |

llama.cpp 继续作为同机、同模型、同负载的比较基线，但不再要求各延迟都胜出 10% 或吞吐达到其 95% 才能发布。两轮配对结果冲突时，沿用 r5 至多补一次配对复测的限制，保留所有结果。保留各后端原始 client SLO 文件的联合达标、错误、拒绝和资格规则，不能把 pooled ITL 与请求级联合概率混用。有限样本结果不声称证明总体 P99 或全局最优容量。

## 已核实的起点与独立风险

交接分支仍指向 `62b92fb1`，交接工作区原先干净。原工作区 `perf/prefill-decode-isolation` 保留 314 个 status 项。本轮复用交接 worktree 新建集成分支，避免再复制一套源码。外部实验和日志继续留在仓库外，Cargo 使用现有共享 target。

| 当前事实 | 证据与影响 |
| --- | --- |
| 自动入口仍用 cold cohort | `prepared_owner/startup.rs` 调用 `run_probe_cohort`；两个正式声明仍为 `native_prefix_acquisition: None`。新恢复协议尚未带来正常启动的准备工作节省 |
| 已有真实恢复协议测试 | capture、关闭 seed、fresh restore ACK 和后续执行有 CPU 证明；该测试有意不满足资格下限，不能证明发布与普通请求采用 |
| 已有受控 cold 发布和采用测试 | 已发布 4 个 epoch 并完成 2 次控制器 witness；多跨度测试给 1 秒规划预算，实测两次分别约 32.29ms、31.30ms，不是产品 2ms 证明 |
| 已有产品预算探针 | `plan_e2e.rs` 两个 2ms wall-clock 探针为 ignored，需在优化构建和空闲主机上显式运行 |
| 已有工作量对账 | `source_work.rs` 遍历真实 `series.requests_for` 核对请求、token、wave；原测试使用结构体 Default，与 Enforce 有效默认数值策略不同 |
| 动作菜单已有部分统一 | 校准和调度已共用 prefill chunk 候选函数；仍需验证产品策略、宽度、上下文与数值支持组合，不能把旧 H41 缺口原样当作最终代码缺口 |
| 最终验证未齐 | 最终 workspace check 有成功日志；最终全量测试、Clippy、双后端特性和运行、正式性能矩阵未完成 |

交接 CUDA normal run 一行抄错：原始累计计数是 declared 472、matched 456、abandoned 16、paired 456。它们是运行时累计计数，缺少普通请求完整执行分母，不能转换成普通请求覆盖率。交接的“32ms 测试预算”也应理解为上述实测耗时，实际测试 allowance 为 1 秒。

准备成本、成本覆盖、规划开销、实际采用、客户端 SLO 是五个独立风险。前缀复用只直接影响准备工作；不能用其通过推断其他四项已解决。

## 防止重复失败的执行规则

1. 每轮修改只对应一个阻断当前验收门的已定位原因。开始前说明触发场景、预期不变量、修改边界和能否否定方案的检查；完成后给实际结果。
2. 先用已有测试和接口取得反例。接口、状态、计费和支持域错误进入 CPU 最小复现；后端语义进入对应后端；性能问题使用适当优化构建和真实测量。
3. 没有新增源码解释或测量证据，不重复同一昂贵实验。发现新问题时先分类，禁止顺手新增一套框架或扩大算子修改范围。
4. 失败后保留原始断言与负结果。不能通过放宽时限、降低资格成员数、关闭安全守卫、忽略失败、伪造 warmup 或重置样本年龄过门。
5. 原子集成单位是一个可重复的完整行为。草稿逐部分审查、合入和对账；不把未编译的 24 个文件一次复制后交给实机找组合问题。
6. root 负责集成、Cargo、GPU 和远端运行；代理只修改明确文件范围。一个共享构建缓存，同一时刻一个相应设备负载，禁止多分支同时覆盖构建产物。
7. 编译、协议、语义、性能四类结论分别报告。所有新结论绑定源码和配置；历史结果只作为诊断依据。每个稳定代码里程碑执行规定检查，避免每个小补丁都重跑全矩阵。

## 第一关 原预算下的可行性

这一步必须在接入 native 大改动前得到可解释的结果。准备工作账与 2ms 规划分别判定：若 cold 重复准备已证明超预算，允许先审查 native 复用的静态工作账，再小步实现用于消除该重复的必要路径；不要求旧 cold 路径先满足它本来无法满足的时限。它不要求预先穷举所有请求，也不把统计资格保证化。

**覆盖和工作量。** 复用 `Case`、`CheckedCaseInventory`、`CheckedPopulationKey`、`CheckedSelection`、现有 manifest 和预算接口。给出现有计划中“实际执行候选或支持范围 → 数值 population → 独立阶段要求 → 准备和测量工作 → 选入或具体 gap”的对应关系。

首先扩展现有 `source_work.rs`：保留历史 Default 场景，增加从生产 `SloConfig` typed 解析取得 Enforce 有效配置的场景；核对所有真实实例化请求和工作量，报告原额度、preflight 支出、selected 总账、剩余额度以及未归入 batch 的 gap。测试必须验证原有不执行设备波的边界。

工作账分为三种量：资格阶段的 offered clock、实际执行 actions、实际 token work。一次 seed setup 与其复用后的各次 restore/suffix 分别计费；同一 reservation 不再重复扣除实际支出。保留失败时已发生的工作。计划可安排不代表数值一定合格，未覆盖项始终为 Unknown。

默认拟合预算装不下常见需求时，先审查 source 分组和优先级是否把完整可学习的 population 被不必要地捆绑。确需统计共享时只在已有 `NumericalFamilyKeyV1` 语义上证明支持范围；执行身份、权限和提交校验继续精确匹配。不能靠 hash 相近或 journal 合并借用资格。

**默认 2ms。** 显式运行两个现有优化构建探针，先基本路径，再带前瞻的普通请求 witness；验证实际提交与 reconcile，不能只检查搜索返回。保持 required observer Disabled，检查完整控制事务的现有阶段计时和超预算完成优先路径。

当前已经有增量 `advance(parent.state)`、最终 fresh replay 和首波 canonical evidence 传递，不重新实现历史上已修好的路径。若失败，先用已有阶段计时区分 capture、查询/search-replay、publication，再决定是否调整不可变快照索引、事务内临时分配或重复查询。以上仅为待测候选热点；不提前引入跨 epoch 缓存，不省 TTL、请求 generation 或最终提交校验。

第一关交付：生产有效配置下的实际工作账和明确覆盖缺口；当前源码的 2ms 探针结果和失败归因。若尚不可行，则当前工作集中在已识别的根因；只有能解释工作缩减且保持资格/容量规则的 native 改动才进入下一小步，规划问题也单独验证。暂停与这些根因无关的接线扩张和硬件矩阵。CPU 小场景通过仍不等于真实后端和并发性能通过。

## 第二关 正常入口的完整自动闭环

复用现有共享 engine builder。CLI `run` 和 `serve` 已通过 automatic-cost-probe 构建入口，不增加另一条用户手工校准流程。

保留现有 guarded capture/restore、source collector、数值验证、发布、回执和资源 authority。将集成控制流明确为：

```text
候选与原资格义务
→ 按完整 source 预留准备与执行工作
→ inventory 内获取当前 source 所需 checkpoint
→ 在收样前决定 native 或重新核算的 cold 或跳过
→ 关闭 seed 和 inventory 并完成实际退役
→ 冻结真实 scope、manifest、phase cuts 和预算
→ fresh owners 执行原 Fitting / Residual / Qualification
→ 资格验证与发布
→ drain、释放额外 pins、退役 source
→ 普通请求执行前查询、实际采用、提交与 actual 配对
```

按以下依赖顺序实施，每项保留最小检查：

| 改动 | 保留的机制 | 必须新增的证明 |
| --- | --- | --- |
| checkpoint 与目标 admission 合同 | 模型、前缀、owner、scope、时间、真实容量校验 | 删除过强的 source/target maximum sequence 相等限制；短上限 seed 到长目标及反向合法恢复；目标自身不足仍零提交拒绝 |
| 准备工作与选择账本 | 已有实际支出/selection reservation 和 F/R/Q | 每 distinct acquisition key 一次 seed；不同合法 preset/max-output 可共享确定性前缀；ordinary ActualPrefill 保持 cold；token/actions/offers 对账 |
| source 准备和冻结 | 已有 inventory、collector、source series | source 打开前绑定真实 scope；准备结果区分 native、cold、可解释跳过和致命错误；失败清理后其他可执行 source 能继续 |
| cold 回退 | 原代表 cases、原资格规则和 cold layout | collector 打开前重建可执行 cohort plan、ranges、phase schedule、manifest/parent 绑定及额外预算；seed 已消费不退款；不挪用其他 source 预留；不足则保留 gap |
| 驻留与清理 | 真实后端 ledger、现有缓存 eviction 和 completion fence | 当前 source 全部驻留 checkpoint，加各 cohort 和 transfer 阶段的真实峰值；部分准备或恢复失败的安全 drain；额外 pins 释放后 cache 残留可归因，下一 source 可进入 |

外部草稿 A 的 offered/actions 分账和一次 setup、草稿 B 的 source 前 acquisition 可参考。必须纠正 B 把所有收样前 `ColdFallback` 变成错误并终止后续 series 的做法。现有 cursor 已对整个 series 预留，series 已保存 cohort ranges 和 parent 身份；因此 cold 回退必须更新真实可执行计划及其绑定，不能只改 source header。

已经冻结并开始收样的 native source 恢复失败时，保留现有共享 drain 后结束 series、保留此前有效 catalog 的路径，不能静默混入 cold 样本。现有 `retire_startup_owner_source` 只接受已完成并 checkpointed 的 source，不能拿它退役未完成源，更不能直接清 collector 字段。若确需失败源之后继续收样，先实现并验证显式 abort 与 worker barrier，再扩大继续执行范围。

第一版采用 source 内复用、source 间顺序执行，沿用真实缓存 ledger 和 eviction。峰值考虑获取后续 key 时已有 checkpoint、seed 与 capture 目标共存，也考虑较窄但上下文/输出上限更长的 cohort；不能只用最大宽度代表最大驻留。生产 capture 当前同时进入共享 prefix cache；释放 `AcquiredProbePrefix` 的额外 Arc 不等于全部设备容量归零。若证据显示共享缓存无法表达正确隔离，再设计明确的 private capture 用途，不能先虚构一套主机字节数代替物理驻留。

取消与 deadline 在提交边界处理：提交前检查，提交后等待真实 completion/ACK 并 drain。不得用外层 timeout 丢弃仍拥有设备事务的 future。保留主错误和清理结果；身份失配、未知提交状态或清理失败不允许假装可继续。

独立 F/R/Q 只复用采样前的确定性 prefix state，每个阶段仍有 fresh 请求和独立实际观测。setup 不是资格成员，不复用 Fit 行作 Qualification，不重置样本时间、checkpoint deadline 或成本 TTL。

第二关正例必须从共享正常 automatic startup 的空模型状态出发，完成原资格与有效发布，再通过普通请求验证 Known、witness、真实提交、actual 配对和完整输出。普通请求使用不同内容但受支持的几何，避免仅靠 startup seed 的缓存命中取得假阳性。身份、ACK、账本和过期负例复用最小协议测试；共享入口只保留能检验跨层组合的代表失败，如部分 acquisition 后容量失败、部分 restore ACK 后取消、收样前可跳过源之后后续源成功。完整资格正例不能由局部测试替代，也不让每个负例都重跑完整资格。

## 第三关 第一份双后端可运行检查点

完成第二关后冻结同一源码，执行规定 workspace 检查和后端编译，再在 Metal 与 CUDA 各验证正常 run 和 serve。真实硬件身份、模型、native operator-set lock 和资源状态重新核对，不能使用历史机器名作为测试条件。

每个入口报告自动准备耗时和预算、发布与有效状态、普通请求可查询机会数、Known 数、采用数、issued/actual 配对及失败原因。不能把几个快照或一个非零 witness 当作覆盖充分的证据；对事先声明必须覆盖的正常场景，逐项检查其实际动作与支持域。

覆盖正常输出与 finish/usage、真实容量边界、取消/断连、缓存过期和回退。共享推理改动两个入口都测；HTTP 的 SSE/非流式和 CLI 的 stdout 慢消费者分别验证。正式默认路径先保持 ProfileOff 与 required observer Disabled；重型诊断另测 producer 开销、上限和丢失。

这份检查点要能复现正常使用和自动成本采用，但还不能标记整个目标完成。若硬件失败，先把原因缩至最小后端场景，再重新验证该修改的影响范围。

## 第四关 完整功能和性能验收

基于重建的 r5 计划保留 224 个观测单元，并在执行前绑定实际 binary、client、配置、硬件、数据和选样身份。原计划状态为 candidate，数值候选与历史冻结项要分开，不把候选率值或参数网格写成用户亲定要求。

| 组 | 范围 |
| --- | --- |
| 主表 64 | 两后端、C4/C8/C16/C32、Off/Observe/Enforce/llama.cpp、两轮 |
| 容量 96 | 两后端和四实现，A/A、粗扫、细扫、所有点独立确认；16 个服务会话 |
| 消融 48 | 两后端、12 个配置、两轮；六个累积阶段与目标/前瞻等单因素 |
| prefix 16 | 两后端、冷热、等待关闭/开启、两轮 |

主模型为 Qwen3.5-9B Q4_K_M、KV fp16。每后端内部四实现使用同一模型和固定选样；Ferrum 三模式使用同一构建。历史候选容量是 32 slots、2048 batch tokens，CUDA context 2048/24GiB，Metal context 4096/8GiB。后端数值策略原本不同，分表披露。

固定 ShareGPT 64 样本由 32 warmup 和 32 measured 组成，完整参考输出长度及原 EOS 政策保留；自然 EOS 正确性另有真实场景。服务容量在并发扫描中保持固定。容量测试保存 scheduled/actual send、backlog、原始队列与 oldest age、排空和右删失，不用闭环客户端冒充开放到达率。

每次报告 TTFT、last-visible TPOT、非空可见 SSE ITL 的 P50/P99，successful output TPS、错误/拒绝/pending、样本与重复、SLO 状态，以及 sampled GPU/Metal memory、host footprint/RSS 的独立口径。usage tokens 与 SSE text events 分开。受传输合并影响的可见停顿仍保留，strict-token eligibility 仅作诊断。

224 单元之外仍保留 RFC 的混合长短请求和自然 EOS、稳定 decode 插入长 prefill、突发及变化到达率、增长上下文、慢消费者/断连、缓存冷热、必要维护和能力降级场景。为每项建立适用性和当前源码证据对应，不新造一个无依据的固定数字矩阵。

消融必须证明所消融机制实际被执行；零预测/零采用不允许解释为算法参数没有影响。K0 只在可信 profile 指向执行热点后展开，以同配置 Off 对照和数值/语义测试证明其收益；每次采用的新参数或执行优化按影响范围重新绑定和测量，不能继承旧配置的通过结论。消融导致最终保留或默认配置改变时，该配置必须完成整个相应容量网格和独立确认，不能只补测有利点。

## 检查 命令与交付

初始可行性检查复用现有 Rust 测试；使用同一 Cargo target，优化构建也可用于工作账测试，以避免重复编译。

```sh
cargo test --locked --offline --release -p ferrum-engine --lib source_work_matches_frozen_original_requests -- --nocapture --test-threads=1
cargo test --locked --offline --release -p ferrum-engine --lib source8_automatic_probe_plan_with_product_planning_budget -- --ignored --nocapture --test-threads=1
cargo test --locked --offline --release -p ferrum-engine --lib source8_automatic_forward_witness_with_product_planning_budget -- --ignored --nocapture --test-threads=1
```

第一个过滤词同时运行历史 Default 和新增的有效产品配置账目测试。这些测试只验证对应 CPU 范围，不替代真实后端。

稳定源码里程碑执行：

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -A warnings
cargo check --workspace --all-targets --features metal
cargo check -p ferrum-cli --bin ferrum --features cuda,vllm-moe-marlin,vllm-paged-attn-v2
```

最后一项在配置好的 CUDA 主机使用 pinned `FERRUM_NATIVE_OPERATOR_SET_LOCK`；Metal 编译在 macOS。后端编译与后端运行结论分开。忽略项、未跑项和容忍失败显式列出。

提交和 push 按稳定、可审查的修改集进行。正式发布准备包含版本检查、构建产物、release notes、同一份双后端证据支持的 README/中文 README/官网内容和发布后验证，遵守既有发布流程及实际授权。未经验证的检查点保持明确 WIP 状态。

本轮外部证据使用 `/private/tmp/ferrum-slo-recovery-20261002`。初始只保留当前计划、结果摘要、必要原始记录和对应源码身份；复用现有 Cargo/cache，不重复归档源码、依赖树和大 trace。当前磁盘容量紧张，构建前先检查并通过 Cargo 清理明确可再生的本任务构建产物，清理范围与结果留在外部记录。

当前仍在第一关。2026 年 10 月 2 日，当前源码的优化构建与以下最小检查已完成：

| 检查 | 实际结果 | 证明边界 |
| --- | --- | --- |
| 历史 Default 与 Enforce 有效配置的工作账 | 2 项通过，合计 0.26s；均为 4 sources、6 populations、724 requests、3188 serial waves、3524 token work | 真正实例化的请求与计划账一致，且纯输入检查未执行设备波；不证明数值资格或后端 120s 可完成 |
| 默认 2ms 基本探针 | 1 项通过，0.41s | 现有受控 CPU 校准/普通控制器场景在默认规划预算下完成 witness、提交与 reconcile |
| 默认 2ms 前瞻探针 | 1 项通过，0.42s；三次规划约 0.298/0.225/0.135ms | 同一受控场景的非空前瞻序列实际采用；不是 Metal/CUDA 或并发尾延迟保证 |
| fmt 和差异检查 | 通过 | 仅格式/差异检查，不替代 workspace/backend 最终验证 |

原始结果分别在本轮外部证据目录的 `g1-source-work-release.log`、`g1-product-2ms-basic-release.log`、`g1-product-2ms-forward-release.log`。构建使用仓库标准 release 配置，共享 target，首次优化构建 13 分钟，后续探针复用同一产物。2ms 探针保持原受控校准 fixture，只将其规划 allowance 设为产品默认；新增产品有效配置的检查仍只是输入工作账，两者不能拼成完整生产启动证明。

新得到的限制是：当前 CPU 输入计划还有 1324 requests 和 13196 waves 的计划余量，但保留 1 个 UnknownPopulation 和 4 个 RetainedSourceCapacity(maximum_sources=4) gap。下一项判别检查是确定排除的 population 是否影响普通 Configured 请求，并核对 source 分组/保留语义；不增加上限来抹掉缺口。现有小场景的 2ms 证据已成立，暂不展开没有测量依据的规划器重写。其余关卡保持未验收，后续按证据和下一项判别检查更新，不给无验收依据的百分比或完成时间承诺。

## 来源位置

- 仓库：`docs/slo-implementation-plan.zh.md`、`docs/slo-algorithm-design.zh.md`、`docs/performance-evaluation.md` 与上述对应 Rust 源码。
- 交接：`/private/tmp/ferrum-efficiency-evidence-20260920/HANDOFF-20261001-STOPPED.zh.md`。
- 计划与草稿公共根：`/private/tmp/ferrum-efficiency-evidence-20260920/slo-goal-20260922/resume-20260926-r1/structured-live-integration-root-r1`。
- 公共根下的 `performance-execution-plan-r5-preparation`：`execution-plan.candidate.json`、`primary-matrix.candidate.json`、`case-inventory.json`、`client-slo-cuda.json`、`client-slo-metal.json`、选样和各组候选文件。
- 公共根下的 `automatic-live-hardware-r41`：原始 CUDA issued/actual、客户端结果、Metal 覆盖和查询审计。属于历史源码。
- 两份草稿：`h42-native-prefix-plan-accounting-r1-work`、`h42-native-prefix-automatic-startup-r1-work`。初始均未编译、未集成。
