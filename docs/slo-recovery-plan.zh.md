# Ferrum SLO 恢复实施方案

2026 年 10 月 2 日，用户要求在充分理解原目标、现状和连续失败经验后形成方案，并已要求创建和启动 goal。目标状态为 active。本方案从 `62b92fb1473bf11f93703d7db41c0dbf9c32ba65` 继续，集成分支为 `slo/recovery-20261002`。

当前仍未交付完整的 SLO 自动闭环版本。首先验证原预算下的覆盖和规划可行性，再完成正常入口的自动闭环，随后进行双后端、完整性能和发布验收。每一步都必须产出可复现的行为证据；局部测试、成本快照数量和源码规模不作为完成度。

下一份可运行检查点以同一条产品链路验收：正常 `run` / `serve` 从空成本状态自动完成准备和原独立资格阶段，发布有效成本模型，内容不同但处于支持范围内的普通请求在执行前得到预测，Enforce 实际采用，真实提交与回执配对，输出完整且资源可退役。用户无需手工训练、导入成本文件或设置隐藏环境组合。CPU 正例先约束组合行为，再以同一源码验证真实后端；该检查点通过后才进入完整性能验收，不把它标成最终发布完成。

当前唯一实现主线是把已有 native 前缀准备接入共享自动启动。每个子修改必须说明它消除上述链路中的哪个阻断；与链路无关的优化单独登记，不能混进当前修改。如果失败推翻预算、支持域或生命周期假设，先修订对应合同与判别用例，再继续实现。没有新的原因证据，不重复昂贵测试，也不根据报错不断追加旁支设计。

2026 年 10 月 3 日进度：正常共享 engine builder 的 CPU cache-off 场景已通过，普通不同内容请求取得执行前预测、实际采用并完整输出，校准结束后的额外私有与共享 lease 均为零。生产源码为 `87cd94c7`，之后仅补测试诊断和部分 capture 成功后的容量失败清理用例。本轮完整 workspace tests 已 exit 0，engine 为 1522 pass、0 fail、6 ignored；当前生产源码的两个默认 2ms 优化构建探针及两项容量清理用例均通过。fmt、workspace check、Clippy 和 Metal compile check 也已通过。前一次全量运行曾发生 Decode 资格欠估，本轮未复现，仍保留原负结果，不声称数值根因已消除。真实 `run` / `serve`、双后端及性能仍未验收，正式性能为 0/224。原 Metal mini 当前仍无法连接，已询问可用访问方式；保存 CPU 检查点后推进实机验证，不继续扩张功能。

收敛以三个可交付结果推进：先得到可重复的正常自动链路检查点，再得到同一源码的 Metal/CUDA `run` / `serve` 可用版本，最后完成原性能范围。每个结果都列明失败和未测项。新失败必须落到具体触发、被否定的假设和最小修改；若它推翻原容量、资格或执行合同，则先停止依赖该假设的扩大验证，重新判定该方案的可行性。不能以修改数量、通过率或更多重跑替代这一判定。

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
| 交接时正式 cache-off 配置也关闭了校准 checkpoint 能力 | 原 `vnext_executor.rs` 由普通 prefix cache 开关控制 `checkpoint_capacity`，guarded capture 同时写共享缓存；本轮已分离用途并取得真实 Metal 生命周期证据，尚未接通完整自动启动 |
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
| 校准私有 checkpoint 用途 | 原物理 ledger、总内存上限、guard、completion 和 ACK | 普通 prefix cache 关闭时校准仍可使用有界私有 capture/restore；私有 capture 不插入共享缓存，普通请求不因此增加命中；最后一个真实 owner 退役后容量可回收 |
| 准备工作与选择账本 | 已有实际支出/selection reservation 和 F/R/Q | 每 distinct acquisition key 一次 seed；不同合法 preset/max-output 可共享确定性前缀；ordinary ActualPrefill 保持 cold；token/actions/offers 对账 |
| source 准备和冻结 | 已有 inventory、collector、source series | source 打开前绑定真实 scope；准备结果区分 native、cold、可解释跳过和致命错误；失败清理后其他可执行 source 能继续 |
| cold 回退 | 原代表 cases、请求成员与顺序、原资格规则 | collector 打开前重算 cold 工作与 schedule，验证原 cycles 足够，再冻结当前 source 的执行选择、manifest/parent 绑定及额外预算；原 members/ranges 不变；seed 已消费不退款；不足则保留 gap |
| 驻留与清理 | 真实后端 ledger、现有缓存 eviction 和 completion fence | 当前 source 全部驻留 checkpoint，加各 cohort 和 transfer 阶段的真实峰值；部分准备或恢复失败的安全 drain；额外 pins 释放后 cache 残留可归因，下一 source 可进入 |

外部草稿 A 的 offered/actions 分账和一次 setup、草稿 B 的 source 前 acquisition 可参考。必须纠正 B 把所有收样前 `ColdFallback` 变成错误并终止后续 series 的做法。现有 cursor 已对整个 series 预留，series 已保存 cohort ranges 和 parent 身份；因此 cold 回退必须更新当前 source 的真实可执行选择及绑定，不能只改 source header。第一版固定原 representative cases、cohort 成员、顺序、cycles 和 seed，重算 cold schedule 与资格输入义务；原 cycles 不足则在收样前 Skip，避免重新选择或重排兄弟 source。

重算必须保留原 `CaseOpportunity.minimum_fresh_members`，不能从“该 case 已选中”反推它必然贡献一个资格成员。沿用现有 `batch_plan` 的条件：`original_cycles × cold_minimum_cycle >= cold_required_original_offers + cold_schedule.block_offered`，包括最后一个原保护 block，以及 schedule/numerical/sample/shape 容量检查。固定成员使父 ranges 保持准确；当前 source 应持有最终 cohort Vec 与父 series/range 的不可变借用，通过自身 cohort 指针校验后再调用原父 execution plan 的真实请求生成。cold 额外执行预约只取 collection 动作的正增量，原 setup 全部预约和已消费费用保留；若未来改变成员，这个单维增量规则不再适用。

已经冻结并开始收样的 native source 恢复失败时，保留现有共享 drain 后结束 series、保留此前有效 catalog 的路径，不能静默混入 cold 样本。现有 `retire_startup_owner_source` 只接受已完成并 checkpointed 的 source，不能拿它退役未完成源，更不能直接清 collector 字段。若确需失败源之后继续收样，先实现并验证显式 abort 与 worker barrier，再扩大继续执行范围。

第一版采用 source 内复用、source 间顺序执行，沿用真实 checkpoint ledger 和既有 completion/reaper。峰值考虑获取后续 key 时已有 checkpoint、seed 与 capture 目标共存，也考虑较窄但上下文/输出上限更长的 cohort；不能只用最大宽度代表最大驻留。接线审查已确认必须区分校准私有 capture 与普通共享缓存用途：正式主表关闭普通 prefix cache，现有开关却同时移除底层 checkpoint capacity，而成功 capture 又同时进入共享 index。正确修复应保留同一总内存限制，使校准 lease 能独立拥有 checkpoint，私有 capture 正常完成 ACK 但不插共享 index；普通 lookup 和隐式 capture 仍遵守用户 cache 开关。不能直接打开共享缓存、更改基准配置或另造主机字节账本来替代物理容量，也不需要重写整个缓存框架。

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

2026 年 10 月 2 日，第一关的 CPU 工作账和默认规划预算探针已完成，当前正在实施第二关。覆盖缺口与后端准备耗时仍须在完整链路验证，以下最小检查不能代表整个目标通过：

| 检查 | 实际结果 | 证明边界 |
| --- | --- | --- |
| 历史 Default 与 Enforce 有效配置的工作账 | 2 项通过，合计 0.26s；均为 4 sources、6 populations、724 requests、3188 serial waves、3524 token work | 真正实例化的请求与计划账一致，且纯输入检查未执行设备波；不证明数值资格或后端 120s 可完成 |
| 默认 2ms 基本探针 | 1 项通过，0.41s | 现有受控 CPU 校准/普通控制器场景在默认规划预算下完成 witness、提交与 reconcile |
| 默认 2ms 前瞻探针 | 1 项通过，0.42s；三次规划约 0.298/0.225/0.135ms | 同一受控场景的非空前瞻序列实际采用；不是 Metal/CUDA 或并发尾延迟保证 |
| fmt 和差异检查 | 通过 | 仅格式/差异检查，不替代 workspace/backend 最终验证 |

原始结果分别在本轮外部证据目录的 `g1-source-work-release.log`、`g1-product-2ms-basic-release.log`、`g1-product-2ms-forward-release.log`。构建使用仓库标准 release 配置，共享 target，首次优化构建 13 分钟，后续探针复用同一产物。2ms 探针保持原受控校准 fixture，只将其规划 allowance 设为产品默认；新增产品有效配置的检查仍只是输入工作账，两者不能拼成完整生产启动证明。

当前 CPU 输入计划还有 1324 requests 和 13196 waves 的计划余量，保留 1 个 UnknownPopulation 和 4 个 RetainedSourceCapacity(maximum_sources=4) gap。逐 batch 核对后，四个 source 容量 gap 全属于辅助 GreedyLength Decode；本 fixture 的 Configured FullLogits Prefill/Decode 已选入。maximum_sources 约束真实保留模型 origin，执行预算余量不能兑换这个容量；现有证据不支持提高上限或修改分组。这个结论仅适用于当前 fixture，不是所有用户采样配置的覆盖保证。

UnknownPopulation 仍需定位到原 case 及真实 admission，区分正常越界负例和可执行的 Configured 覆盖缺口。已确认该聚合项也包含未 admission 的 WarmResidencyUnproven；日志中的 44 次资源拒绝则是已经建立 owner 后投影的未来工作超过 backing，不能表述成入口拒绝了长请求。后续完整链路必须用普通请求的真实查询检查支持范围，选入计划不等于完成资格和采用。现有小场景的 2ms 证据已成立，暂不展开没有测量依据的规划器重写。

第二关的独立 admission 合同修复已通过最小检查：只删除 seed/target 最大序列长度相等的附加限制，保留目标自己的真实 admission 与容量检查。原同长生命周期及短 seed→长 target、长 seed→短 target、原 host-capacity 回退合计 4 项通过；真实 native request authority 的组级检查确认 seed 为 3/5/6、target 组最小值为 5，保留每个 cohort 的两个实际 restore ACK、完整输出、预算和原资格失败断言。它不证明完整资格或双后端运行。

另有 3 项既有底层测试通过：恢复 extension 保持 request fit 且不能越界；错误 layout/超 backing 的恢复初始化拒绝；guard 拒绝时零提交并保留原 owner。格式和 diff 检查通过，workspace/backend 全量验收尚未完成。首轮新增测试错误地使用 native 路径不会调用的 observation hook，结果 1 pass / 3 fail；改用现有真实资源 authority 读取后重跑通过，原失败日志保留，没有修改生产行为或放宽断言来消除该失败。证据为 `g2-prefix-admission-dev.log`、`g2-prefix-admission-native-authority-dev.log`、`g2-restore-backing-boundary.log`、`g2-restore-guard-rejection.log` 和 `g2-restore-request-capacity.log`。

校准私有 checkpoint 已完成用途隔离。只有有效的 automatic、非 Off、CompleteRequests 配置启用私有能力；普通 prefix cache 仍遵守原开关，两者共享原物理容量。私有 capture 必须持有匹配的 live interest，完成真实 ACK 后由 lease 持有，不插共享索引。恢复要求原执行器签发的精确 lease 和 native guard；私有重试保留原到期时间。缺少原生 checkpoint 能力的 provider 仍可执行原有冷推理。

| 私有用途改动的检查 | 实际结果 | 证明边界 |
| --- | --- | --- |
| 模型 prefix cache 单元测试 | 63 项通过 | 包括有效产品配置、原总容量和 cache 开关隔离；默认特性 CPU 测试，不是设备运行 |
| 真实 Metal 私有 checkpoint | 3 项通过，2.43s | 捕获并关闭 seed、不同合法 target admission 下恢复、ACK 前后 frontier、相同分段冷计算 logits 有限且绝对差不超过 1e-5、共享 cache 无条目或命中、最后 lease 释放后 checkpoint claim/physical bytes 归零并可再次捕获；cache-off 下缺少 lease/guard 或 guard 拒绝时不提交 checkpoint copy；不支持 checkpoint 的 provider 冷推理可用 |
| engine native acquisition | 4 项通过，4.02s | 原相同和不同 admission 生命周期、容量回退及预算/完整输出；保留原资格失败断言 |
| 原 engine 共享 checkpoint 回归 | 12 项通过，145.66s | 包括 guard/ACK、取消/退役、普通共享前缀恢复、既有自动冷准备及普通请求执行；不证明新私有准备已进入自动路径 |
| Metal workspace 全目标编译 | `cargo check --workspace --all-targets --features metal` 通过，2m07s，存在 warnings | 覆盖接口调用方的编译兼容；不是全量运行测试或 CUDA 证明 |
| 格式与差异 | 通过 | 全 workspace/backend 验收仍另行执行 |

Metal 用例运行于本机 Apple M1 Max（24 GPU cores），采用测试中真实支持 checkpoint 的 F32-master provider；它不是正式性能验收机器或主模型。首轮测试代码有 3 处类型错误；修正后 1 pass / 2 fail，揭示 seed 到达边界后登记私有 interest 被旧的未来区间校验拒绝。生产修复仅允许私有用途在恰好已完成边界登记 Pending interest；真实 native capture 仍以已完成区间验证合法性，然后才分配和提交。修复后 3 项通过，没有放宽 native 捕获规则。失败和最终日志分别保存在 `g2-private-prefix-metal.log`、`g2-private-prefix-metal-runtime.log`、`g2-private-prefix-metal-retired-boundary.log`；其余证据为 `g2-private-prefix-cache-unit.log`、`g2-private-prefix-engine.log`、`g2-private-prefix-engine-checkpoint.log` 和 `g2-private-prefix-workspace-metal-check.log`。此次有界修改未运行默认特性 workspace check、workspace 全量 tests、Clippy 或 CUDA 检查；完整自动闭环源码稳定后执行全套，当前提交不标记为 PR 已完成验收。

准备工作账的第一批代码已接入共享计划计算：候选选择、schedule 输入、legacy build 和最终冻结共用 `case_work`；源内 distinct acquisition key 只加一次 seed/setup，合并 source 后重新去重。source offered clock、serial inference rows 和 execution actions 分开，setup/capture/restore 不增加数值资格成员；冻结时重新核对实际 order 总账与 selection reservation。不同未来 sampling preset/max-output 不改变 proper prefix key，但每个 case 的原策略绑定仍单独校验。生产 Case 当前仍为 `acquisition: None`，尚未启用私有 native 自动准备。

此批先运行 7 个工作账单测，再运行整个 `prepared_owner::plan::` 范围：144 项通过、0 fail、0 ignored，61.44s，包含这 7 项及 2 个新的 selector/coalesce 与预算边界用例。两个实际 `series.requests_for` 生成并重新分词的 cold 对账场景仍为 724 requests、3188 serial inference rows/actions、3524 token work；原资格规则、剩余预算和 gaps 保留。新 audit 的 `declared_offer_row_bound` 与实际生成请求的行数一致。证据为 `g2-native-accounting-work-tests.log` 和 `g2-native-accounting-plan-tests.log`，均为带 Metal 特性的 debug engine CPU 测试，不能视为设备计时或自动 native 执行证明。格式/diff 检查通过，未重复 workspace/backend 全量检查。

上述已提交的工作账检查点之后，自动入口接入正在验证：`checked.rs::freeze_inventory` 已分别限制 action 与 sample/shape rows；冻结 inventory 前保留各 source 的原 representative `Case + CaseOpportunity`，并计入 retained budget；当前 source 的真实 lease/执行选择在 collector 打开前绑定，执行循环通过最终选择生成原请求。冷回退保持原成员和 cycles，保留已花费 setup 与后续 source 的原预约，不能满足资格或容量时在收样前 Skip。额外 token evidence 的运行成本、当前 source 全部 checkpoint 的真实峰值及双后端效果仍未证明。其余关卡保持未验收，不给无验收依据的百分比或完成时间承诺。

本轮先通过了真实 seed 执行后 interest 容量不足的组合负例：花费未退款、collector 未打开、owner 清理完毕，释放容量后原后继 source 仍能准备，1 项通过、4.30s（`g2-native-startup-interest-fallback.log`）。该用例仅覆盖第一个 key 的 seed 已执行后失败，不代表多个成功 checkpoint 同时驻留后的失败。受控 session 的自动计划也完成原资格、2 次实际 capture、258 次 restore、普通不同内容请求的 Known/witness/提交/配对与完整输出，1 项通过、15.18s（`g2-native-startup-qualified-adoption-reconciled.log`）。结束后的私有与共享 lease 数均为零；普通请求没有继承 startup checkpoint。前两次正例检查分别因误用仅保留最近 32 条的诊断计数、沿用“所有提交都是推理”的旧断言失败；改为累计真实完成数并分别核对推理/capture/restore，保留失败日志。

随后正常 `finish_automatic_startup_with_probes`、有效 Enforce、共享 cache 关闭的 CPU 正例失败，不能用上述受控 session 的通过替代它。原始日志 `g2-native-startup-shared-builder-cache-off.log` 显示 source 0 发布 epoch 1，source 1 第一次 restore 后发生 `prefix actual settlement/FIFO coverage is incomplete`；普通请求 Decode 为 WrongDomain，采用失败。根因已定位：真实产品维护观测和推理观测共用 FIFO，restore 的维护记录已占用 ordinal，但 native ACK 没有推进 prefix preparation、source collector 和回放 ledger 的对应位置。它与 chunk 本身无关。

该修复从原 `offer_prefix` 返回值取得 exact transfer authority 对应的 ordinal，以一个有界 weak receipt 传给原 native ACK；每层继续严格要求 `last_fifo + 1`。维护记录不增加 inference offer、call、shape 或资格成员；缺失证据不能用全局 cutoff 补齐。真实 authority 的 clone/foreign/过期检查 1 项通过、0.04s（`g2-native-startup-weak-receipt-authority.log`）；scheduler native acquisition 组 7 项通过、0.20s，包含连续维护 FIFO 正例和跳号、重复、未 ACK 的拒绝（`g2-native-startup-maintenance-fifo-protocol.log`）。

首次接线后仍失败：维护 ACK 到来时，原 block 尚未由第一条推理打开，scheduler 还未继承源的 FIFO cut（`g2-native-startup-shared-builder-cache-off-fifo.log`）。最终修复在物理 restore 提交前调用原 block 打开机制，保留原 cutoff，保证 opening clock 早于 ACK；已打开的 block 不重开。没有重置时钟、扩大样本或放松顺序断言。正常 builder cache-off 重跑 1 项通过、30.48s，四个 source 依次激活 epoch 1–4（`g2-native-startup-shared-builder-cache-off-block.log`）。这是 ControlledExecutor CPU 协议证据，fixture 仍使用 1s 规划 allowance、24-token 输入、3-token 输出及最大并发 2，不能称为产品 2ms 或设备性能证明。

同一源码随后完成：计划组 147 项通过、62.83s（`g2-native-startup-plan-regression.log`）；正常共享 checkpoint 组 6 项通过、147.58s，包含 shared cache、自然 EOS、等待/恢复、普通 cache reuse、重新捕获及 cache-off（`g2-native-startup-shared-checkpoint-regression.log`）；自动计划与普通采用、原三阶段 acquisition、容量回退、预算和 prefix cost 组合回归 37 项通过、2 项原 wall-clock 探针 ignored、29.28s（`g2-native-startup-lifecycle-budget-regression.log`）。ignored 两项需在优化构建显式运行，旧源码结果不能充作本批验证。格式和 diff 检查通过，完整 workspace/backend 检查正在进行，第二关整体和产品可用版本仍未验收。

当前集成源码已提交并 push 为 `b6e16ade`。默认 `cargo check --workspace --all-targets` 已通过、1m31s；Metal 同范围 compile check 已通过、59.38s，均存在 warnings；`cargo clippy --workspace --all-targets -- -A warnings` 已通过、2m19s。证据为 `g2-native-startup-workspace-check.log`、`g2-native-startup-workspace-metal-check.log` 和 `g2-native-startup-workspace-clippy.log`。Clippy 使用仓库规定的允许 warnings 配置，不能表述成零 warning。

完整 workspace tests 首次在编译阶段因磁盘余量降至 341MiB 被主动 SIGINT（exit 130）；只清理旧 CLI/server/devtools 缓存后第二次仍因 linker/query-cache 的 ENOSPC 退出 101。这两次都不是测试语义通过。随后用 Cargo dry-run 确认并清理共享 target 的整个 dev profile（53.8GiB），保留 release、registry、模型、源码与证据，按原构建选项重建。新编译完成用时 11m21s；engine lib 为 1518 pass、1 fail、6 ignored、505.87s，后续 workspace targets 因 engine 失败未执行。日志为 `g2-native-startup-workspace-test-fresh-debug.log`。部分 engine 父测试启动子进程，不能累加所有 `test result` 行充作唯一测试数。

失败为原 `source8_checked_algorithm_subset_executes_unobserved_mixed_rows_with_real_witness`，独立复跑在同一断言失败、13.19s（`g2-combination-source-isolated.log`）。其四个 raw source 都已实际收集和导入，但不存在独立的 A+B 组合 child。该 fixture 没有 native checkpoint 能力，因此不是本轮 capture/restore 失败。最小 typed 同宽 A/B 选择用例在 0.06s 复现：同宽 raw journal 合并发生在组合候选选择之前，合并后只剩一份纯 Decode journal，旧算法要求的另一份已安排、不同 family 的 journal 已消失。原不同宽正例避开了这个场景；不能据静态比较认定该缺口由 `b6e16ade` 引入。

修复范围限定为原选择器：保留独立 raw A/B 的完整资格计划，再从原 checked inputs 声明一份另外执行完整 F/R/Q 的组合源。遵守既有 Configured 优先于辅助 GreedyLength 的顺序，保护合并前原遍历在三类预算和 source cap 内已经能安排的全部 population，新增组合还需通过完整工作量和源名额检查。当前反例合并前的前四份原候选全部是 Configured；合并后可用的名额不能自动全部归入低优先级辅助采样。辅助源仍保留完整声明，放不下时显式报告 gap，不缩短阶段或抬高 cap。

选择器的两项新增回归通过、0.03s（`g2-combination-same-width-fixed.log`）：同宽两 raw families 及独立组合执行；三账各少一单位、source cap 不足、seed 不完整时原 raw 不受影响；同策略和较低策略的原后续 source 保留且组合另算完整工作。原 `context_family` 端到端测试未修改，重跑通过、13.92s（`g2-combination-source-fixed.log`），覆盖独立 raw A/B 和组合的资格、未采集过的混合 A+B 查询、真实 witness 采用及完整输出。实际四份 source 为 Configured Prefill raw、Configured Decode raw、Configured Decode union、首份 Greedy raw；原后续 Greedy Decode 成为显式容量 gap。修复只证明该 CPU 场景闭合，不是双后端性能或全量 workspace 通过。

同一选择器修复的整个 `prepared_owner::plan::` CPU/default 回归为 149 pass、0 fail、0 ignored、21.79s（`g2-combination-plan-regression.log`），包含原不同宽组合、native 静态工作账、覆盖顺序与容量规则。格式和 diff 检查通过；不把该组结果替代整个 workspace 或正常产品入口的验收。

`87cd94c7` 的完整 workspace tests 为 exit 101：engine 1520 pass、1 fail、6 ignored、510.00s，后续 targets 未执行（`g2-combination-workspace-test.log`）。原组合端到端已在此次全量运行通过。唯一失败是 `native_prefix_cpu_ordinary_qualified_hold_restore_and_first_commit`：Configured 与 Greedy 两份 Decode source 的首轮均发生 `QualificationUnderestimate`，后续独立阶段未在原 source horizon 内形成有效 child，普通请求因 Decode 为 WrongDomain 无法取得有限时间 witness；失败在 follower 提交及 hold/restore 之前。该 fixture 的四份计划、工作量与此前通过的正常启动一致，且不满足新增组合路径的条件。空 child 导入产生的 `profile frozen child inventory differs` 是后续报错，不能据此认定持久化元数据损坏。

同一 workspace 二进制单独运行原失败项通过、30.48s，四个 source 均导入（`g2-qualified-hold-workspace-isolated.log`）。这只说明资格结果受运行条件影响；没有失败成员的实际耗时和冻结预测数值，不能断言是环境噪声，也不能用单独通过覆盖全量失败。当前仅在测试夹具接入既有 WARN 诊断，输出原 qualification 成员的 wall、fitted upper、residual、margin、planning 和 excess；不改样本、预算或资格规则。使用全局测试 subscriber 才能接收既有工作线程事件，保留已安装的 subscriber；日志扫描、格式化和 stderr 本身有观察开销。

曾尝试只运行尚未到达的 workspace packages，但排除前面 packages 改变了 Cargo feature unification，引发额外构建，磁盘余量降至约 3.1GiB；编译阶段主动中断、exit 130，没有得到测试结果（`g2-combination-workspace-remaining.log`）。确认无 Cargo/rustc 后，只清理共享 target 的 `debug/incremental`，保留依赖对象、可执行文件、release 缓存及证据，磁盘余量恢复约 16GiB。后续沿用完整 workspace package 集合并加 `--no-fail-fast`，一次记录全部 targets 的实际结果，不再用改变依赖组合的方式补测。

当前生产源码 `87cd94c7` 的两个默认 2ms 优化构建探针通过：2 pass、0 fail、0 ignored、0.85s，构建 12m45s（`g1-current-product-budget-release.log`）。前瞻场景实际决定、提交和配对均为 4 次、共 7 waves，包含 3 次非空 tail，4 次规划累计 0.942ms；基本场景为 1 次、1 wave，规划 0.199ms。两者均完成原请求输出。这是未启用 native checkpoint 的受控 CPU 小场景，不能替代正常 native builder 的默认预算或真实设备结果。二进制含测试诊断的初版，后续仅调整诊断级别上限；探针不安装该 subscriber。

私有准备的后续 key 容量失败边界已补齐，无生产修改。原第一 key 容量失败测试保留，新增真实 2/3-token 模板经原 cursor 形成多 key source；先填满真实 interest 容量再释放一个槽，要求实际完成 1 次 capture 后第二 key 被拒绝。检查全部已获取 private lease 退役、shared lease 为零、已花费 seed/capture 不退款、原预约与 deadline 不重置，以及释放压力后原后继 source 能真实准备。两项均通过、4.96s（`g2-native-later-key-capacity-workspace-filter.log`）；命令保持完整 workspace feature graph，其他测试仅被过滤，不能算全量通过。它覆盖后续 interest 容量失败，不冒充 capture 已提交后的失败或 partial restore ACK 故障证明。

带上述测试诊断和容量补测的完整 `cargo test --workspace --all-targets --no-fail-fast` 已 exit 0（`g2-current-workspace-test-diagnostics.log`），不再因首个 target 失败而提前停止。engine library 为 1522 pass、0 fail、6 ignored、509.61s；interfaces 为 936/0/3，scheduler 为 1013/0/28，models 为 586/0/6，CLI 为 542/0/0，server 为 418/0/0。这里分别列 library 终态，不把父测试启动的子进程结果重复累加成全仓总数。原 hold/restore 断言本次通过；生产算法未为此前资格失败作修改，因此不能将本次通过写成已修复数值欠估，也不据此追加无新证据的重跑。

Ignored 包括父测试专用 helper、真实模型或外部服务、GPU/算子依赖、优化构建计时，以及原始归档/settings 诊断。两项 2ms 探针引用其独立 release 结果；未启用后端 feature 导致的 0 tests 也不算设备验证。CUDA compile/runtime、真实双后端正常入口和正式性能仍待完成，本批不标记 PR 或完整目标已验收。

同一源码已完成规定的本地检查：fmt/diff 通过；workspace 全目标 check 通过、3m40s（`g2-current-workspace-check.log`）；Clippy 按 `-A warnings` 通过、2m22s（`g2-current-workspace-clippy.log`）；Metal 全目标 compile check 通过、1m21s（`g2-current-workspace-metal-check.log`）。构建仍有 warnings，Clippy 不是零 warning 证明；Metal 编译也不是 Metal 运行或 CUDA 兼容证明。下一项实机工作使用固定的当前检查点、原正常入口和模型，不再用新增 CPU 大场景延后真实链路验收。

## 来源位置

- 仓库：`docs/slo-implementation-plan.zh.md`、`docs/slo-algorithm-design.zh.md`、`docs/performance-evaluation.md` 与上述对应 Rust 源码。
- 交接：`/private/tmp/ferrum-efficiency-evidence-20260920/HANDOFF-20261001-STOPPED.zh.md`。
- 计划与草稿公共根：`/private/tmp/ferrum-efficiency-evidence-20260920/slo-goal-20260922/resume-20260926-r1/structured-live-integration-root-r1`。
- 公共根下的 `performance-execution-plan-r5-preparation`：`execution-plan.candidate.json`、`primary-matrix.candidate.json`、`case-inventory.json`、`client-slo-cuda.json`、`client-slo-metal.json`、选样和各组候选文件。
- 公共根下的 `automatic-live-hardware-r41`：原始 CUDA issued/actual、客户端结果、Metal 覆盖和查询审计。属于历史源码。
- 两份草稿：`h42-native-prefix-plan-accounting-r1-work`、`h42-native-prefix-automatic-startup-r1-work`。初始均未编译、未集成。
