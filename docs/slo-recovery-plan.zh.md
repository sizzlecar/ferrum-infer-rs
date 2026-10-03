# Ferrum SLO 恢复实施方案

2026 年 10 月 2 日，用户要求在充分理解原目标、现状和连续失败经验后形成方案，并已要求创建、启动和恢复 goal。本方案从 `62b92fb1473bf11f93703d7db41c0dbf9c32ba65` 继续，集成分支为 `slo/recovery-20261002`。10 月 3 日最新核对，goal 工具已返回 `active`，实际工作通过 Tailscale 继续。

当前仍未交付完整的 SLO 自动闭环版本。首先验证原预算下的覆盖和规划可行性，再完成正常入口的自动闭环，随后进行双后端、完整性能和发布验收。每一步都必须产出可复现的行为证据；局部测试、成本快照数量和源码规模不作为完成度。

最终容量目标继续沿用[原算法设计中的 C*](slo-algorithm-design.zh.md)：固定请求到达与长度分布族、模型质量、硬件和资源，在完整完成请求的合法策略中，寻找同时满足 TTFT、TPOT、可见文本 ITL、成功率及队列稳定条件的最高可持续请求到达率。实际 successful output tokens/s、原请求级联合达标以及本文件声明的最新配对门槛另行验收；到达率与输出吞吐分别计量。有限窗口只提供持续容量证据，不证明所有策略中的全局最优或无限时间的队列稳定。自动校准、成本模型、有限规划和干预都是实现该目标的手段。

下一份可运行检查点以同一条产品链路验收：正常 `run` / `serve` 从空成本状态自动完成准备和原独立资格阶段，发布有效成本模型，内容不同但处于支持范围内的普通请求在执行前得到预测，Enforce 实际采用，真实提交与回执配对，输出完整且资源可退役。用户无需手工训练、导入成本文件或设置隐藏环境组合。CPU 正例先约束组合行为，再以同一源码验证真实后端；该检查点通过后才进入完整性能验收，不把它标成最终发布完成。

## 当前收敛计划：先关闭联动缺口，再冻结实机检查点

此前虽有分阶段计划，实际仍过多依赖完整实机运行寻找组合故障。局部选源、维护、发布与调度测试分别通过，没有证明它们在同一真实计划和原预算下共同成立。G15 首版通过 123 项 layout 测试却破坏原普通查询，就是这类验收缺口；后续以原端到端失败和撤回对照定位，不能把补回两个测试当作整条产品链路已经完整。

当前生产检查点为 `f1c6ccac`：在 `5ace3e72` 的联动修复基础上，G19 减少冷几何的重复正交化计算。真实 CUDA 输入的全部几何结果已在本地保持原 32M 和 scratch 上限通过对照；完整回归及同源码正常后端入口仍须验证。验收义务来自原目标、声明的模型能力和目标负载，沿用现有 typed 输入、选择账与回执，不新增另一套运行框架。只统计能支持具体行为的证据，不以源码规模、来源数、测试数量或非零 witness 判定完成。

| 关口 | 当前已知状态 | 本关完成条件 |
| --- | --- | --- |
| 必要策略与几何覆盖 | G19 对 G18 实际输入的 162 项几何全部完成，使用 26,627,160 / 32,000,000 visits，秩、枢轴和最终 case 与完整原核一致；新后端选源、独立资格和普通采用仍缺 | 将必要普通请求的可执行路径对应到原 family、选源、独立 F/R/Q 和最终 catalog；列出未覆盖项及原因，明确 C8 的可行路径和支持边界。C8 不等于要求所有 rows8 或所有声明成为 Known，选择成功也不等于数值资格成功 |
| 准备、容量与工作预算 | G16 已有原冻结预算内维护、受影响来源完整收样及普通采用的 CPU 联动证明；真实 Metal 仍只有修复前 78/815 cohorts 的失败记录 | 从真实冻结计划核对 seed、维护、restore、F/R/Q 的请求、动作、offer rows、owner 和存活内存；证明原 reservation 可完成相关维护并继续完整收样。120s 的实际耗时仍由后端测量判定 |
| 数值支持到普通采用 | 原 basic/forward E2E 已恢复；新增 GreedyLength 联动严格绑定最终采用的 IndependentReplay，维护联动严格绑定受影响来源和普通实际 Matched 收据，两项 CPU 用例已通过 | 新保留的策略必须由原 catalog 查询成功并产生普通执行 witness；私有维护必须与真实计划、完整 F/R/Q、发布和后续采用连成一条测试路径。保留原断言、数值解释和所有预算 |
| 默认规划时间 | 普通逻辑 E2E 使用宽松测试时限；当前优化构建的两个原 2ms 探针另已通过 | 实机再核普通事务的预算、合法选择、实际提交与配对，解释预算耗尽；两个受控 CPU 场景不证明真实模型或并发负载可行 |
| 正常入口和资源生命周期 | 旧硬件记录跨不同源码，部分 run 输出被 token 上限截断 | 同一冻结源码在两后端的 run/serve 完成必要自动准备、普通采用、完整输出及清理；取消、过期、漂移、能力降级等原义务各自有适用证据 |
| SLO 与收益 | CUDA TPOT、Metal 三项 P99 仍失败，无当前同构建 Off 配对 | 固定原选样和配置，先验证同构建 Off/Enforce 的绝对 SLO、自身吞吐和延迟退化门，再完成原 224 单元及原功能场景；全过程保留错误和未完成请求 |

推进顺序如下，下面各关的详细合同继续有效：

1. **完整负载前的可行性与联动验证。** 复用 `CheckedCaseInventory`、`CheckedSelection`、`source_work.rs`、现有 native/普通 E2E 和原日志，集中处理上述已知缺口。优先补齐“新保留策略实际采用”和“维护处于原冻结工作预算内的完整收样”两处联动证明。所需工作超出原容量或覆盖仍无解释时，当前候选留在本关；先给出能改变该结论的机制及工作量依据。历史 artifact 没有完整保存 geometry axes，manifest 和 hash 不能冒充可重放输入；若缺少必要原输入，只进行取得该证据的有限后端诊断，保留原预算，再判断是否适合进入完整负载。
2. **冻结完整功能检查点。** 必要 CPU 联动、优化构建 2ms 探针和规定检查通过后，再绑定同一源码及配置开展后端验证。每个作业预先列明假设、输入、通过条件和失败分类，首先检查启动及普通窗口证据；前置条件不成立时不扩展整轮负载。实机失败先归入已有覆盖、资源、数值、时间或后端语义合同，取得足以否定具体假设的最小证据，再决定是否改动及重测范围。
3. **性能收敛与完整交付。** 功能检查点成立后做配对性能和正式矩阵。性能失败按真实阶段耗时与执行热点归因；若必须修改共享执行或调度，按影响范围重过功能门并重新冻结。224 单元、原额外功能场景和发布准备全部保留；正式发布或合并仍需对应授权。

近期联动验证直接复用现有 `plan_e2e.rs`、`source_preparation.rs` 和生产 cursor：新增策略使用原 catalog 查询及实际 witness 对账，绑定采用模型的 capture identity；维护场景使用真实冻结 reservation，保持原工作额度，不沿用局部反例额外加入的 `setup.actions()+1` 作为整体成功条件。联合资源账复用 `work::case_work/setup_for_indices`、`budget::plan_groups`、`packing_valid` 和现有 memory 计费，分别核对请求、projection、执行 action、offer rows、geometry、owner 与共存内存。总额度有余额不代表当前选中 source 仍有执行信用。

若必要路径的工作量或共存资源已证明超过原上限、必要 geometry 在原 32M 内无已证明的完整执行办法，或修复无进展循环后必要准备仍超过真实 120s，则判定当前执行安排未通过可行性关，提出机制及代价明确的设计修订。不能用放宽资格、删必要支持或新增预算结束本关，也不把当前安排失败夸大成所有架构都不可能。

预算核算与实测结论分别报告。离线账能否定请求、动作、geometry、owner 或内存容量上的不可行候选；历史吞吐只能用于估计准备时长，不能证明新计划满足 120s。一次回归的修复也不能关闭其他已知缺口。每次汇报给出当前关口、已关闭义务、仍缺证据以及下一项能改变结论的检查；出现原方案不可行的证据时明确修订机制和代价，不以继续重复完整运行替代判断。

G13 CUDA 原正常 inventory 的几何预算账已补核：原 32,000,000 visits 中使用 31,998,248，余 1,752，但下一项原子计费已无法支付；98 个 Configured population 完整计算，使用 23,351,472，64 个 GreedyLength population 中 39 个完整、25 个不完整，使用 8,646,776。耗尽发生在一个 width5 FullLogits population，之后 width6/7/8 的 24 个 population 均为零访问。对应 rows8 候选未开始计算，不能把它们说成“各自需要超过 32M”，也不能因 packing 修复就认为其几何缺口已消失。[原几何预算审计](/private/tmp/ferrum-slo-recovery-20261002/g16-c8-input-geometry-feasibility-audit.json)

G18 几何采集已实现为显式测试构建支持：复用真实 CLI serve 模板与 Enforce builder，在原 selection 借用写出全部有序轴的原始 f64 bits、anchor、settings 和前后账；inventory 清理后、Source8 收样前，经原 series finalization 和 engine shutdown 退出。普通非测试构建未开启该支持时不包含新增逻辑。采集保留原 120s/32M，不运行客户端负载，不建立数值资格或性能结论。输出使用原 StagedFile 和 64KiB 缓冲，有独立文件硬上限；不足时保留部分文件并报告不完整，原清理仍须执行。同步写入不能强制逐行受 120s 中断，因此另记录实际到期状态，采集完整不等于启动时间可行。[采集合同](/private/tmp/ferrum-slo-recovery-20261002/g18-geometry-capture-draft/CONTRACT.json)

实机前补齐了同次 `original_cases` 元数据：原 Case、模板 prompt 长度、chunk 和 row ceiling 均借用序列化，矩阵行的 case 索引可以对应真实批宽、上下文和执行域。仅凭 family key 和旧 G13 索引不能识别本次 C8。正常启动正例检查该映射实际存在且索引有效；空间不足负例检查原 owner、backing、lease 和 scheduler 状态清空，后台 trainer 的结束依据仍是原 awaited shutdown 合同。最终采集源码 `0d0f9ecf` 的规定检查已完成：fmt、workspace check/test、Clippy `-A warnings`、Metal 全目标编译及 pinned CUDA CLI 编译全部退出 0。全仓去除子进程重复后为 **7886 pass / 0 fail / 109 ignored**；ignored 不算执行。[最终全仓审计](/private/tmp/ferrum-slo-recovery-20261002/g18-final-workspace-audit.json)、[清理独立审查](/private/tmp/ferrum-slo-recovery-20261002/g18-normal-startup-independent-review.json)

G18 CUDA 实际采集已结束：上述同一源码的 release CLI 测试 **1 pass / 0 fail、109.22s**，guard 退出 0，原服务恢复并验证健康。原配置和预算未改，完整保存 **162/162 个矩阵、5038 个原 case、47 个模板长度**；文件 9,638,654 B，SHA256 为 `1abea0b3483da567c01aab1e396090d8c023cd883196200689255ec783ac9c16`。inventory 结束时原 cost-probe deadline 尚余 81.22s；109.22s 是包含其他启动阶段的测试总时长，两者不可相减。此作业在 Source8 收样之前退出，不能作为普通启动闭环或 SLO 通过。[实际构建、输入和恢复独立审计](/private/tmp/ferrum-slo-recovery-20261002/g18-cuda-inventory-final-audit.json)

测试提交 `49f03cd7` 使用 Rust 按原顺序重放同一共享 32M 账本，保留 f64 原始位，逐项核对原结果、最终 case 集、全部前后访问数和 exhausted 状态；实际重放 **1 pass / 0 fail、4.97s**。原 32,000,000 visits 使用 **31,997,348**：98 个 Configured 全部完成，使用 23,389,456；64 个 GreedyLength 中 39 个完成、25 个未完成，使用 8,607,892。耗尽发生在 width5 Full 候选，GreedyLength 的八个 width8 候选均零访问。该结果来自本次输入，不混用旧 G13 的索引和计费。[同次重放报告](/private/tmp/ferrum-slo-recovery-20261002/g18-geometry-replay-run-r1/original-geometry-audit.json)、[执行日志](/private/tmp/ferrum-slo-recovery-20261002/g18-geometry-replay-real-r1.log)

结构统计发现 36 组严格相同计算输入、37 次额外调用；其中已完成重复调用消耗 4,676,784 visits。全部 403,776 个原坐标中有 19,456 个全行零轴坐标，约 4.8%。这些只是优化机会，不是可直接抵扣的节省。随后以同一输入独立测量每项原内核的完整工作需求，并检查原归一化后是否存在严格位相同输入，结果见下文；诊断参考账与生产共享账分开，不能扩大生产预算。只有计入新增扫描、比对、缓存和共存内存后，必要路径仍可在原额度内完成，才进入后端完整负载。几何通过也不代替 selection、独立 F/R/Q、普通采用及最终容量验收。[离线核算设计](/private/tmp/ferrum-slo-recovery-20261002/g18-offline-geometry-audit-design.json)

G18 后续完整需求测量已完成：原 162 项全部独立计算需要 **38,193,728 visits**。仅复用原始位完全相同输入的理想剩余为 **32,627,168**；仅复用原归一化后相同输入、保留每次原 4ND 前处理的理想剩余为 **32,213,072**，均未计缓存、比对等开销。两者单独不足以完整完成当前调用序列，因此未将其直接实现为生产缓存。这不证明所有必要覆盖都不可行。[需求测量](/private/tmp/ferrum-slo-recovery-20261002/g18-geometry-reference-run-r1/demand-summary.json)

G19 改为在单次冷几何调用内复用原正交化的**第一遍前缀残差**。原 basis 只追加且旧方向不变，所以对新增方向继续第一遍，与从原行重新执行第一遍保持同一浮点操作顺序；每次第二遍仍在副本上从头完整执行。原 normalized rows 存储被就地复用，两个标量记录 anchor 与其余行的进度，没有跨调用矩阵缓存或新增堆分配；largest norm 在修改原行之前计算。正常数值 Fit 的原 core 不变，原预算、数值阈值、scratch 门和不可退款语义保留。

`f1c6ccac` 的定向测试 **13 pass / 0 fail / 1 ignored**，覆盖 anchor 转换、相同残差的原顺序、依赖与病态边界、精确预算、差一单位失败及持续耗尽。另以原真实捕获执行显式 Rust 对照 **1 pass / 0 fail、6.86s**：先复现原共享账，再用未改原 core 取得完整参考，最后让生产公开 API 按原顺序共用原 32M。新实现 **162/162 完成、26,627,160 visits、exhausted=false**，与工作量预测完全一致；每项 rank、anchor rank、pivot 和最终 case 均匹配原完整参考，原 scratch 限制不变。减少的 11,566,568 visits 是计算工作，不是实测吞吐或延迟收益。真实后端 Source8、普通采用、完整输出和 SLO 尚未因本项对照而通过，正式验收仍 **0/224**。[独立重放审计](/private/tmp/ferrum-slo-recovery-20261002/g19-first-pass-replay-audit.json)、[当前检查状态](/private/tmp/ferrum-slo-recovery-20261002/g19-first-pass-status.json)

G19 的 fmt、workspace check（56.37s）、完整 workspace test、Clippy `-A warnings`（54.66s）和 Metal 全目标编译（38.56s）均退出 0。全仓按顶层 harness 去重为 **7897 pass / 0 fail / 110 ignored**，另有 16 项 Criterion smoke，不作性能成绩。新增四项优化测试、G17 联动、原 basic/forward/multispan E2E 均通过；显式实机输入重放在全仓仍计 ignored，其单独运行结果不重复累加。上述生产提交已 push。同源码正常产品的 CUDA/Metal 构建材料已准备，CUDA pinned 编译和两后端实际入口验证尚待完成。[全仓审计](/private/tmp/ferrum-slo-recovery-20261002/g19-workspace-audit.json)

新增 GreedyLength 联动测试 r3 为 **1 pass / 0 fail、11.59s**。真实 fresh Prefill 通过原 time-admission，最终采用的 replay 为 transaction1/replay1，其后续 attempt19/alternative1 对新 FullLogits/PlainTextGreedyV1 child 返回 Known；普通后续实际 Decode 仍提交 GreedyToken。独立 F/R/Q、原 numerical scope、输出与清理断言保留，协议测试沿用原宽松规划时限，不是实机或默认 2ms 性能证据。r1 的四次普通采用成功，但目标 child 只出现在 Search，最终绑定断言失败，不能计通过；r2 因新增维护测试访问 calibration 私有方法而编译失败，没有执行测试。两次失败原日志保留。[r1](/private/tmp/ferrum-slo-recovery-20261002/g17-greedy-family-joint-r1.log)、[r2](/private/tmp/ferrum-slo-recovery-20261002/g17-greedy-family-joint-r2.log)、[r3](/private/tmp/ferrum-slo-recovery-20261002/g17-greedy-family-joint-r3.log)

新增维护联动测试 r2 为 **1 pass / 0 fail、41.07s**。保持原默认请求、动作、projection 和 120s 额度，从正常 automatic builder 触发一次实际 private seed 维护延期；恢复后真实 checkpoint authority 对应原 Source8 restored 记录，完整原 journal 重放要求生命周期及独立 F/R/Q 完成。已安装 child 的 source/capture 身份继续与普通请求原实际 `Matched` 收据的 source/capture/domain/parameters/epoch 对齐，并独立检查 decision/submitted/reconciled、完整输出和 owner/lease/ticket 清理。这只证明受维护影响的来源完成并被采用，不把它扩大成所有后续来源或真实后端已通过。[通过日志](/private/tmp/ferrum-slo-recovery-20261002/g17-maintenance-joint-r2.log)

维护联动 r1 为 **0 pass / 1 fail、40.18s**：上述启动、来源重放和独立资格断言通过，但立即读取普通回执时原 OnceLock 尚为空。正常 builder 使用后台 worker，`drain_calibration_fixture` 在该模式不执行；执行 ACK 先于后台原 `SealedCostCall::resolve` 生成收据。r2 只在原普通请求 20s 总时限内等待同一条已有收据，再检查 Matched 和完整身份，没有创建收据、额外 checkpoint 或新 epoch。普通路径不能使用 manual calibration 专用 receipt 的问题也在独立审查中提前排除，没有开启 manual driver。新增读取器和 enum re-export 均为 `cfg(test)`，生产行为继续冻结于 `5ace3e72`。[失败日志](/private/tmp/ferrum-slo-recovery-20261002/g17-maintenance-joint-r1.log)、[保留诊断文件 pins](/private/tmp/ferrum-slo-recovery-20261002/g17-maintenance-joint-r1-diagnostic-pins.json)

G17 首轮全仓在原 interfaces 用例停止：engine 为 **1544/0/6 ignored、107.33s**，两项新联动和原 E2E 均通过；已完成区间去除子进程重复后为 **4259/1/68 ignored**，不能称完整全仓。失败是 `ready_checkpoint_survives_retired_producer_with_native_restore_and_ack` 要求立即 Known，却收到 `ReadUnavailable(DeferredCleanup)`。原实现该 reason 仅表示进程共享 cleanup registry 的 try-lock 竞争，不表示 producer 有待清理任务；实际饱和仍是 BusyOrUnavailable。测试沿用相邻 fixture 的 5s 有界读取，仅对此临时 reason 重读，全部实际安全负例保留，产品协议不改。原 workspace 依赖配置下相关模块 **8/0、0.15s**；全仓第二轮已退出 0。这个附带测试修正独立报告，不作为新增 SLO 功能完成。[首轮审计](/private/tmp/ferrum-slo-recovery-20261002/g17-joint-workspace-first-audit.json)、[模块回归](/private/tmp/ferrum-slo-recovery-20261002/g17-checkpoint-projection-module.log)

G17 最终全仓为 **7880 pass / 0 fail / 109 ignored**：229 个顶层 Rust harness，排除 17 条子进程重复汇总；另有两个 Criterion harness 的 16 项 smoke，不作性能证据。engine 为 **1544/0/6、91.23s**，原接口失败用例及两项新联动均通过。最终 fmt、workspace check（0.69s）、Clippy `-A warnings`（18.07s）、Metal 全目标编译（49.98s）均退出 0。联动测试提交为 `d94897ca`，附带接口测试修正单独提交为 `b0852d8e`；该检查点未更改普通构建的执行行为，仍对应 `5ace3e72`。CUDA 编译与双后端实际运行尚缺，原默认 2ms release 探针保留为 `5ace3e72` 的既有证据，没有在本次测试提交上重跑。G17 当时的几何采集只有仓库外草稿；后续 G18 实际构建和采集结果见上文。[最终全仓审计](/private/tmp/ferrum-slo-recovery-20261002/g17-joint-workspace-final-audit.json)、[联动及检查状态](/private/tmp/ferrum-slo-recovery-20261002/g17-joint-gates-status.json)

## G15/G16：已修复端到端回归，完整检查进行中

当前实现检查点为 `5ace3e729ba54a92baefb9357f3c2ecefdff551c`，本地规定检查已完成，仍处于上面的联合可行性关。上一份 `f22e70b5695dff7ba0cb6c0deffe263d5b47a3e6` 包含 G15（`12192140`）及 G16 私有维护循环修复；其 G15 回归及失败证据继续保留。

G15 在原来源上限和请求、动作、inference rows 三项容量内，尝试让同产品的不同 host family 共用一份 Source8 journal。首版为此构建共同 U，虽然保留了 host categorical、EOS/length 语义、全部代表和各 family 的独立 F/R/Q，完整验证仍发现它改变原数值解释后导致选择冲突，不能作为可用修复。覆盖增益规则仍保留：完整重算必须保留原可安排覆盖、且实际新增至少一个原未安排候选；只减少 journal 数而未新增覆盖则保持原来源，geometry 不完整的候选不能借此成为完整覆盖。

G16 为私有 prefix acquisition 接入与正常 cohort driver 相同的 Maintenance turn：在原 deadline 内消费原 one-use 维护票据，然后重新捕获 frontier。首次零提交 Wave 和每个后续 Wave 仍扣费；Maintenance 沿原合同不计 inference offer 或 native-copy action，没有退款或扩预算。CPU 反例使用明确的 `setup.actions()+1` 有限额度承担一次真实零提交重试，并验证少一个动作仍拒绝 capture、owner/lease 可清理；这不证明实机冻结计划已有足够重试余量。

G15 有效 before 为 **0 pass / 1 fail、0.05s**，命中 selector 未安排完整 family；前两轮分别是 Prefill 超物理域和 seed 包含不支持的 Prefill，均保留为 fixture 失败，不算产品反例。修复后 focused 为 **1/0**，最新 layout 组为 **123/0、7.50s**。G16 有效 before 为 **0/1、3.24s**，原 reservation 耗尽且 maintenance_calls 为 0；修复后 prefix acquisition 组为 **5/0、3.38s**。严格未观测轴数值回归另为 **1/0**。这些结果不累加成全仓计数。[G15 before](/private/tmp/ferrum-slo-recovery-20261002/g15-cross-policy-before-r3.log)、[layout 回归](/private/tmp/ferrum-slo-recovery-20261002/g15-layout-tests-r2.log)、[G16 before](/private/tmp/ferrum-slo-recovery-20261002/g16-prefix-maintenance-before-r2.log)、[prefix 回归](/private/tmp/ferrum-slo-recovery-20261002/g16-prefix-maintenance-after-suite.log)、[数值回归](/private/tmp/ferrum-slo-recovery-20261002/g15-unobserved-axis-test.log)。

`f22e70b5` 的 fmt check 与 workspace check 均退出 0（后者 40.90s）；workspace test 已退出 **101**，engine 为 **1539 pass / 2 fail / 6 ignored**，全仓中途停止，不报告完整全仓计数。basic `plan_e2e` 的原普通 rows2 / FullLogits / Decode step2 查询返回 `QualificationCoverage`；forward 用例在原 witness 计数处得到 0 而非 1，其 Required journal 记录 rows1 / FullLogits 查询同样缺少资格覆盖，不能把两处失败都写成 step2 断言。[全仓失败日志](/private/tmp/ferrum-slo-recovery-20261002/g15-g16-workspace-tests.log)。

同一 basic E2E 单独复跑仍失败（11.51s）。在保留 G16 的情况下临时撤回六个 G15 文件，原断言通过（9.86s，带完整日志复跑为 9.70s）；随后六文件均恢复，未把临时对照当作候选提交。失败路径新增的有界诊断使用同一个不可变快照和原 query 时刻：新 U child 的 F/R/Q 成员数为 40/16/16，直接预测可用；旧 raw child 为 36/8/8，直接预测返回 `QualificationCoverage`。两者没有 phase support membership 判别，原 catalog 优先选中 raw child，故可用 U 被遮蔽。该对照把回归定位到 G15 改变原数值解释，不支持修改资格门或在预测失败后换模型重试。[撤回对照](/private/tmp/ferrum-slo-recovery-20261002/g16-plan-e2e-without-g15.log)、[同快照 child 诊断](/private/tmp/ferrum-slo-recovery-20261002/g16-plan-e2e-child-diagnostic.log)。

当前 `5ace3e72` 修正只共享 journal，保留原解释：raw 与 raw 合并后仍为 raw；原 U 完全相同才保留该 U 合并；raw/U 混合或不同 U 拒绝新合并。独立 owner、原代表、F/R/Q、关联 recipe 闭合、geometry、原来源覆盖和三项工作预算继续检查，catalog 及原 E2E 断言不变。真实 G13 CUDA 的目标 Configured 和 GreedyLength 候选原本具有相同 U，因此此约束仍针对已发现的实机 host-policy 缺口；是否能通过真实执行仍待验证。[实际候选 scope 审计](/private/tmp/ferrum-slo-recovery-20261002/g16-g13-cuda-equal-universe-audit.json)

修正后原两项失败 E2E 已通过：**2 pass / 0 fail、22.46s**，包含普通查询预测、实际 controller witness 和非终态后续波；同一命令中的默认 2ms 探针为 ignored，不计通过。选源 layout 组为 **124/0、17.04s**，prefix acquisition 组为 **5/0、3.82s**，fmt check 为 0。完整 workspace check 退出 0（37.03s），workspace test 退出 0：229 个顶层 Rust harness 去除 17 条子进程重复汇总后为 **7878 pass / 0 fail / 109 ignored**；另有 16 项 Criterion smoke 通过，不作性能结论。engine 为 1542/0/6 ignored、79.62s。当前优化构建默认 2ms 探针另已通过，结果见下文；`f22e70b5` 的旧硬件包保持停用。[原 E2E 修正后](/private/tmp/ferrum-slo-recovery-20261002/g16-preserved-scope-e2e-r2.log)、[layout](/private/tmp/ferrum-slo-recovery-20261002/g16-preserved-scope-layout.log)、[prefix](/private/tmp/ferrum-slo-recovery-20261002/g16-preserved-scope-prefix.log)、[全仓去重审计](/private/tmp/ferrum-slo-recovery-20261002/g16-preserved-scope-workspace-audit.json)。首轮命令漏设已有共享 Cargo target，依赖编译被主动中止（exit130），未运行测试，不作为行为证据。

当前源码的 release 默认 2ms 探针已退出 0：**2 pass / 0 fail / 0 ignored、1.06s**。basic 有 1 次实际 decision/submitted/reconciled，规划耗时 199,500ns；forward 为 4 次，规划耗时总和 958,917ns，规划波总数 7、其中 tail 3。总和不是单次或 P99 统计，这些只证明受控 CPU 场景在原预算内得到实际采用，不能代替真实后端。[优化构建 2ms 探针](/private/tmp/ferrum-slo-recovery-20261002/g16-preserved-scope-release-2ms.log)

Clippy（`-A warnings`）退出 0、41.32s；Metal 全目标编译退出 0、34.95s。G15/G16 当时仍缺 pinned CUDA 编译和双后端正常入口实机验证；后续 G18 已完成前者及限定范围采集，完整入口验证仍缺。本地编译不能充当后端运行证据。下方 G13/G14 实机结果不含 G15/G16 修复，不能作为其 after 证据。必要 host-policy、宽度支持和 SLO 尚未闭环，正式验收仍为 **0/224**。

## G13：最近完成的四入口实机结果，仍未达到可用检查点

2026 年 10 月 3 日，冻结候选 `1ad2642e` 的 Metal/CUDA 正常 `run`、`serve` 均已结束，guard/product 退出码为 0，**但普通请求预测采用、必要覆盖和 SLO 尚未闭环，正式验收仍为 0/224**。本轮包含 G12 重复 gaps 计费修正和 G13 完整关联轨迹声明修正；旧失败没有删除。

规定的 fmt、workspace check/test、Clippy（`-A warnings`）及 Metal 全目标编译通过；全仓按顶层 Rust harness 去重统计为 **7875 pass、0 fail、109 ignored**，ignored 不算已执行。两端均以这份 3353 文件源码完成 release 构建，实际 Cargo artifact 与冻结二进制对应；CUDA 原 pinned native lock 下的 CLI check 也通过。编译和 CPU 测试不能替代实机覆盖、默认 2ms 规划或 SLO 证明。

| 入口 | 本轮实际结果 | 当前缺口与边界 |
| --- | --- | --- |
| CUDA `run` | 92.81s 完成当前冻结计划的 632/632 cohorts，发布四份来源、epoch4；整个 bootstrap 约 99.45s | G11 拒绝未重现，但旧计划为 606 cohorts，不能声称逐波重放了原失败轨迹。后续 source7 因容量超出 2689 B 失败，确实进入下一代并结算 5 个 offer，但未再发布；1536-token 输出仍截断 |
| CUDA `serve` | 59.45s 完成当前 353 cohorts，启动 epoch4；在线实际新增、替换至 epoch5/6 | 普通窗口 witness decision/submitted/reconciled 为 **0/0/0**；437 个候选查询均 WrongDomain，Known 为 0，保持完成请求的回退执行；TPOT 和 joint SLO 失败 |
| Metal `run` | 完成 287/448 cohorts、两份来源、epoch2 | source2 在原 120s 时限停止，尚有请求与动作额度；reference probe 另耗约 62s，整个 bootstrap 约 182s。512-token 输出仍截断，未取得普通请求独立采用分母 |
| Metal `serve` | composition 获授权：32,943,962 B ≤ 34,195,039 B；约 21.69s 后停于 78/815 cohorts、首份 Prefill 来源 | source1 的 15 个 key 已 ACK，尚未进入收样即停止；具体拒绝未记录，不能称超时。普通窗口 witness 为 164/119/119，但三项延迟 P99 均失败 |

来源数和 cohort 数只描述当前计划，既不是覆盖率，也不是新增的 4/4 硬门。CUDA `run` 的累计 1069 issued、1037 paired 含启动和私有请求，不能写成普通采用。CUDA `serve` 的普通窗口确有 2833 次实际提交与结算，但预测 witness 为零；同期 planner phase 耗尽 1933 次、hard budget 耗尽 368 次，与 437 次查询不是同一分母。G10 相同选样窗口曾有 25/21/21 witness，本轮零采用不能被启动完成、后续模型发布或 retrospective Known902 掩盖。原正常运行的 WrongDomain 聚合缺少 query 身份，437 次仍不能逐次归因；后续独立 Required 诊断已确定一项必要 host-policy 覆盖缺口。

同一 `1ad2642e` 产品开启 `StructuredRequiredV1` 的 CUDA 诊断已结束，guard/server/client 均为 0，原服务恢复且健康。health-before 的事务数为 0，health-after 为 3158，故保留的事务属于 32 warmup + 32 measured 普通请求窗口；168974 个事件、3158 个完整事务无丢失，原 Rust audit 为 0 issues。377 个 query 均构造成功，其中 369 次 lookup 全在 epoch4 返回 WrongDomain、Known 为 0。**这 369 次是开启 observer 后的独立轨迹，不能替代或逐次解释原正常运行的 437 次，也不作为性能成绩。** 原工具的 `successful_close_attested=false` 保留，成功关闭另由退出与恢复回执佐证。[诊断状态](/private/tmp/ferrum-slo-recovery-20261002/g13-cuda-required-query-status.json)

这 369 次查询的实际 host policy 均为 `PlainTextGreedyV1`，epoch4 的两个数值 decode child 却均为 `PlainTextInstalledV2`、`model_eos=true/user_stop=false`，没有相同 host-policy family。事务 1239 的 rows=1、原 owner algorithm 与已安装 Greedy child 相同，仍因成本不可用结束；总规划 1,832,318ns，planner/hard exhausted 均为 false。因此该反例不能归咎于 2ms 耗尽或算法集合完全缺失。缺少同策略 child 足以使 catalog selection 无法选中适用模型，但不证明补齐后算法、数值支持和采用必然成功。epoch5 后有 372 个 snapshot，却无 query 构造或 lookup，不能据此判断新增 child 的覆盖；更晚的 `MissingProducer` 也不能解释前面的 epoch4 失败。[原 query 身份审计](/private/tmp/ferrum-slo-recovery-20261002/g13-cuda-required-query-identity-audit.json)

原正常 serve 的完整冻结声明进一步确认：GreedyLength 已生成 1996 个 case、64 个 population、32 个候选来源，选入数为 0。50 个 Configured 候选排序在前，首四份来源先占满原 `maximum_sources=4`。真正匹配该 query host policy 的 rows1 批次为 **52/53**（chat SSE、`include_usage=true`）；两者 geometry 完整，各需 107 请求、368 动作、253 inference rows、99 cohorts，仅被来源保留容量拒绝。rows8 的对应批次 **79/81** 还存在 geometry work 耗尽，不能把单行候选塞入计划就宣称覆盖 C8。正常与 Required 两轮的首四来源及 52/53 账目一致，但完整 inventory 并非逐项相同。该证据定位了选源缺口，并未证明简单改优先级或追加来源能保留全部原义务、通过原预算。[策略与选源审计](/private/tmp/ferrum-slo-recovery-20261002/g13-cuda-required-policy-selection-audit.json)

两端仍为原 pinned ShareGPT、固定 C8、32 warmup + 32 measured、1 次重复，64 请求均成功、错误为 0。CUDA 完整包已核：主 TPOT 终点为最后可见输出，ITL 为相邻非空可见 SSE 文本事件间隔。

| 后端 | TTFT P50/P99 ms | TPOT P50/P99 ms | 可见 SSE ITL P50/P99 ms | successful output tokens/s | 峰值内存 | 原 SLO |
| --- | --- | --- | --- | --- | --- | --- |
| CUDA | 71.396 / 138.781 | 18.614 / 20.880 | 17.345 / 40.022 | 350.957 | runtime requested 23,216,118,580 B；NVML 整卡 23,672 MiB；host max RSS 6,684,404 KiB | 原 200/15/50ms P99 阈值下 TTFT、pooled ITL 通过，TPOT 失败；joint 1/32，低于 99% |

CUDA measured 为 9210 usage tokens、9179 可见文本事件、9147 间隔；两条 event/usage 不一致请求仍计入可见停顿，无传输合并。Runtime memory 为 833 个样本、NVML 为 844 个样本，覆盖启动至退出；不同口径不相加，WSL per-process NVML 不可用。原服务已恢复为 PID699052，10:10:45 UTC 独立健康检查为 ok，这是该作业结束时的历史状态。Metal 完整归档已核：TTFT/TPOT/可见 ITL P99 为 4849.539/233.788/868.979ms，输出 37.014 tokens/s，原 3400/212/359ms 阈值均失败，joint 3/32。终态 Metal allocation 峰值 8,600,748,032 B（2116 样本、0 错误），独立 host 采样峰值 RSS 533,659,648 B、physical footprint 2,445,871,672 B；口径不相加，采样可能漏过瞬时峰值。单次重复不能建立性能收益或正式主表结论。

G14 Metal 错误诊断候选 `a08a3257` 已结束，guard、aggregate、server、client 四项退出码均为 0，但 startup 仍在约 21.85s 后停于 **78/815 cohorts、首份来源、epoch1**。source1 在 15 个 private key ACK 后，255-token seed、boundary254、offset0 处累计 2550 次 Blocked，最后一次为 `MaintenanceUnavailable`；随后 `actual_attempts_remaining=selection_attempts_remaining=13647`，原扣费返回 `native action has no original selected reservation`。2550 不是同一种 reason 的计数，也不能回填成 G13 的精确轨迹。该 Wave 分支在检测到待处理 maintenance ticket 时、创建 2ms controller budget 前返回；当时私有 acquisition loop 只重试 Wave，没有正常 driver 的 Maintenance turn，且没有后台执行循环。具体 ticket 的建立事件及底层容量种类仍未确定；上方 G16 CPU 反例只验证这个维护生命周期缺口。[错误证据审计](/private/tmp/ferrum-slo-recovery-20261002/g14-metal-preparation-error-audit.json)

G14 普通 health 窗口涵盖 32 warmup + 32 measured，64 请求成功、错误为 0，witness decision/submitted/reconciled 为 **180/133/133**。measured 32 条输出检查通过；原固定 C8、1 次重复下，TTFT/TPOT/可见 SSE ITL P99 为 **4840.607/212.830/877.658ms**，仍分别超过原 **3400/212/359ms** 阈值，joint 为 3/32，输出 37.146 tokens/s。usage tokens 9210、可见事件 9179、间隔 9147；两条 event/usage 不一致仍纳入可见停顿。完整 runtime 归档已核对配置、14 项 input pins 和原进程身份，并与 `a08a3257` 的 3353 文件源码及实际 Cargo 产物回执对应，产品 SHA 为 `408b7c54…6977f`。Apple M4 的 Metal allocation 峰值为 **8,598,831,104 B**，2114 个样本覆盖权重加载前至 shutdown，终态 complete、0 错误；host 2128 个样本的峰值 RSS 为 **540,049,408 B**、physical footprint 为 **2,401,257,920 B**，同一 PID 与 birth、0 错误，末样本距进程结束 371.122ms，但 host sampler 没有显式 complete 标志。各口径不相加，采样可能漏过瞬时峰值。automatic-reuse cache payload 未完整取回、未重放，不能宣称 warm restart catalog 有效；这些 G14 结果也不是 G15/G16 修复后的硬件证据。[终态摘要](/private/tmp/ferrum-slo-recovery-20261002/g14-metal-serve-summary-runtime-audit.json)、[runtime 与内存审计](/private/tmp/ferrum-slo-recovery-20261002/g14-metal-serve-runtime-audit.json)

**下一步先完成当前候选的规定检查，再验证原预算内必要 host-policy、宽度覆盖及私有维护的真实后端闭环。** G15/G16 的局部通过不能代替正常 `run` / `serve` 的独立资格、普通 query Known、实际采用和完整输出；不能混合策略、丢弃原覆盖义务、增来源槽或把单行资格当作 C8 支持。原独立 F/R/Q、2ms、120s/2048/16384 和完整输出要求不变；当前不启动 224 单元正式矩阵。

外部审计：[CPU 全仓计数](/private/tmp/ferrum-slo-recovery-20261002/g13-workspace-test-audit.json)、[CUDA run 运行](/private/tmp/ferrum-slo-recovery-20261002/g13-cuda-run-runtime-audit.json)与[覆盖边界](/private/tmp/ferrum-slo-recovery-20261002/g13-cuda-run-coverage-audit.json)、[CUDA serve 完整结果](/private/tmp/ferrum-slo-recovery-20261002/g13-cuda-serve-runtime-audit.json)、[Metal run](/private/tmp/ferrum-slo-recovery-20261002/g13-metal-run-results.json)、[Metal serve 摘要](/private/tmp/ferrum-slo-recovery-20261002/g13-metal-serve-summary-audit.json)与[完整归档](/private/tmp/ferrum-slo-recovery-20261002/g13-metal-serve-full-audit.json)。两端构建身份另见 `g13-metal-build-audit.json`、`g13-cuda-build-and-runtime-audit.json`。

G14 仅新增失败证据：已有 epoch 时仍记录原 `last_error`、来源位置和实际/预约余额，seed 动作扣费失败时记录 Blocked 总数及最后 typed reason。原返回值、扣费和资格未改；它与尚待实机验证的 G16 维护修复是不同源码。G13 静态账已包含 seed prefill、capture 和 restore，但不能把 2550 次阻塞都认定为同一种原因。

## G10：同一候选的四入口实机结果（历史）

2026 年 10 月 3 日，`39eebad8` 的 Metal/CUDA 正常 `run`、`serve` 已全部结束。仍无完整可交付版本，正式验收 **0/224**。本次包含两项修复：同一次 captured outcome 的原始配方共享，以及区分保留全部 native commands 的 adaptive replay 与 sealed direct invocation。前者保持完整配方存活期和引用计账，后者仅恢复原物理凭据，不把 GraphPath Unknown 改成数值合格。两项分别有修复前失败与修复后通过的 Rust 反例，真实 CUDA cold graph 及取消后 fence 生命周期检查也通过。

规定的 fmt、workspace check/test、Clippy（`-A warnings`）、Metal 全目标编译、CUDA pinned native lock 下的 CLI check 均通过。全仓测试为 7870 pass、0 fail、109 ignored；两个 release 默认 2ms 受控 CPU 探针通过。两端 release 实际产物与同一份 3352 文件源码清单已独立核对。上述检查不证明真实并发规划满足 2ms，也不替代完整功能和 SLO 验收。

| 入口 | 冻结候选的实机结果 | 未通过的条件 |
| --- | --- | --- |
| Metal `run` | 合并候选获授权；完成 286/448 cohorts，两份来源发布 | 原成本校准序列用尽约 120s；独立 reference probe 另耗约 62s，整个 bootstrap 约 182s；原 512-token 计数输出仍被上限截断 |
| Metal `serve` | 完成 714/1773 cohorts，首份来源发布；普通窗口实际提交、配对各增加 85 | 合并需 42,333,242 bytes、可用 34,195,039，仍未获授权；source1 出现一次 700,296ns qualification underestimate，随后临近期限发生 restore unavailable，底层拒绝原因未记录；三项延迟 P99 失败 |
| CUDA `run` | 首份来源完成 132 cohorts、F/R/Q 为 48/32/32，发布 epoch1；已跨过旧 physical settlement 阻断 | source1 第二个 cohort 出现 `invalid structured numerical replay`，最终 133/621 cohorts、一份来源；不是预算耗尽；原 1536-token 计数输出仍被上限截断 |
| CUDA `serve` | 输入清单完成；两份来源、170/353 cohorts、epoch2；普通窗口实际提交、配对各增加 21，上一版为 0 | source2 第二个 cohort 在序列约 50.59s 时同样出现数值回放拒绝；必要域覆盖和真实规划预算仍未证明，TPOT P99 失败 |

来源数量仅用于描述选中计划的执行状态，不是独立验收门槛或覆盖率。普通窗口由 health-before/after 差值确定，覆盖各 32 warmup + 32 measured 请求：Metal witness decision/submitted/reconciled 为 131/85/85，CUDA 为 25/21/21；不能用启动与 live 的累计 issued/actual 替换此窗口，也不能把非零采用写成完整覆盖。CUDA startup 有 14 次实际 capture、114 次 restore；普通窗口两者没有增长，健康截点的临时 checkpoint 占用和清理队列为 0。

普通窗口进一步核对：CUDA 有 3361 个完成的控制器 transaction，规划阶段耗尽 2402 次、hard budget 耗尽 527 次；Metal 对应 2914、1916、55。三个计数不是同一事件分类，不能相加。两端实际采用的 witness 均只有一波、tail 为 0，生成/展开候选数均等于 witness 样本数。此轮没有展示实机多候选、带后续波的前瞻计划，也没有 Off 配对来证明改变排程及净收益；非零 issued/actual 配对只证明局部真实采用。依据为原普通观测 `health-before/after`，没有开启 Required 重型追踪；原始差值和 hash 另存 `g10-ordinary-planning-window.json`。

这只描述已采用 witness 子集，不代表整个窗口没有搜索多个候选。单波也可能合法满足当前 horizon，lookahead3 是上限，不是每次必须三波。全窗口同时存在真实成本覆盖拒绝和预算停止，健康聚合缺少逐 witness 的 depth、Unknown 和 deadline 关联，不能把每次单波选择归到同一个原因。代码路径与现有计数的界限另存 `g10-ordinary-single-wave-causal-limits.json`；下一步仍以必要覆盖、原预算内有用的选择及同负载 Off 对照判断。

两端服务保持 Qwen3.5-9B Q4_K_M / KV fp16、同机、同配置、固定 C8、原 pinned ShareGPT 64 条选样与输出长度策略，1 次重复，各 64 请求完成且错误为 0。主口径取原 Rust SLO sidecar；TPOT 终点为最后可见输出，ITL 为相邻非空可见 SSE 文本事件间隔。

| 后端 | TTFT P50/P99 ms | TPOT P50/P99 ms | 可见 SSE ITL P50/P99 ms | successful output tokens/s | 峰值内存 | 原 SLO |
| --- | --- | --- | --- | --- | --- | --- |
| Metal | 972.894 / 4869.827 | 193.295 / 219.297 | 176.316 / 873.740 | 36.748 | MTL 8,598,831,104 B；host RSS 651,296,768 B、footprint 2,101,611,184 B | 三项 P99 失败，joint 3/32 |
| CUDA | 69.468 / 113.617 | 19.045 / 20.896 | 17.839 / 42.406 | 338.596 | runtime requested 23,216,118,580 B；NVML 整卡 23,651 MiB；host max RSS 6,686,984 KiB | TTFT、pooled ITL P99 通过，TPOT 失败；joint 0/32 |

每端 measured 均为 9210 usage tokens、9179 可见文本事件、9147 间隔；两条 event/usage 不一致请求仍计入可见停顿，没有观察到传输合并。角色、空与结束消息不计入 ITL。内存峰值覆盖启动到退出，采样可能漏过瞬时高点；口径重叠不相加，WSL per-process NVML 不可用。单轮诊断没有重复间置信区间或同构建 Off 配对，不能据此认定性能收益或正式矩阵通过。

两端另以既有算术任务 `What is 17 + 25? Reply with only the number.` 验证正常 CLI 的单次完整输出：stdout 均恰为 `42`，usage 为 28 prompt / 3 completion tokens，terminal `finish_reason=stop`，guard/product 均退出 0。该任务增加了 `--profile-jsonl` 以取得终态证据；即使 detail 为 off，也写入约 0.95GB 的启动与执行事件。完整文件保留在原主机，外部证据包保存末条 terminal record 及全文件 hash/size 收据。本项只证明该任务语义和自然终态，不作为性能、完整模型质量或相同校准轨迹证明。CUDA 原服务恢复并健康，最后 PID 436776。

本轮没有满足“正常自动链路检查点”。下一步先取数值拒绝的 typed reason、触发位置与原事件身份，不根据固定 universe 的静态可达路径猜测失败算法，也不把 WrongDomain 全部改成非成员；同时拆解 Metal 合并峰值与必要工作量，重判原预算下的覆盖可行性。四入口完整闭环未通过前，不启动 224 单元正式矩阵。

独立结果位于外部证据目录 `/private/tmp/ferrum-slo-recovery-20261002`：`g10-backend-build-identity-audit.json`、四份 `g10-final-{metal,cuda}-{run,serve}-results.json`、`g10-basic-output-results.json` 和各原始 evidence archive；状态汇总为 `g10-validation-status.json`。源码已 push，原 dirty 工作区未改。

## G10 后的定位与局部修复

`f1fade2f`（G11）仅增加错误路径诊断：保留原 `invalid structured numerical replay` 错误和 poison 行为，另记录 typed reason、固定触发位置及原 Source8 event/cohort/ticket/FIFO/call，不序列化完整记录。真实 journal 对照确认成功前缀没有新增日志，日志开关不改变失败记账、原记录或 source receipt。最小测试 1 pass，受影响 structured replay 组 166 pass、0 fail、27 项原外部证据测试未执行，fmt/diff 检查通过。CUDA 诊断构建使用该独立冻结版本，保持 G10 正常 run 的任务、配置、预算及观测设置；3353 文件源码清单、实际 Cargo artifact、冻结二进制、native lock/PTX 收据和各构建退出码已独立核对。

G11 正常 CUDA run 已在 10 月 3 日 08:47 UTC 前结束，guard/product 均退出 0，但自动校准仍失败。准确拒绝为 `PhysicalEnvelopeProjection / WrongDomain`；原 Source8 `completed` 记录位于 source1、phase0、cohort1、ticket5、FIFO1070、call877，之前接受到 FIFO1069，offered4，固定 declared universe 存在、discovery 未启用。拒绝发生在约 41.99s，最终 133/606 cohorts、一份来源、epoch1，因此不是 120s 超时。日志定位到了 `contract.project_input`，仍未记录具体算法名单，不能据此断言缺哪一项或把所有 WrongDomain 作为合法非成员放过。相同配置下本轮计划为 606 cohorts，G10 为 621；不声称运行选择轨迹完全一致。

原计数任务 stdout 与 G10 字节一致，仍达到 1536-token 上限；它只作因果诊断，不证明完整输出或性能提升。CUDA 原服务已按原 exe/cwd/cmdline 恢复，PID 560440、health 为 ok。原始证据为 `g11-cuda-run-evidence.tar.gz`，SHA256 `ea36df4bdda04af865bd35e9cd19599e33beca9d1ad0e34efa4457fdf6031bce`；该轮没有加入 G12 内存修复。下一步以准确拒绝路径建立最小因果反例，保留原物理校验、独立资格、计时和计账规则。

`089a49b8`（G12）修正合并候选的一项重复计费：当前选择结果只有一份 `gaps` Vec，原 `memory::plan` 已按完整分组拒绝、早停、geometry、append 和全局缺口上界收费，并包含扩容时旧/新 backing；composition 没有第二份 gaps 缓冲，却再次预约其空间。仅删除后者，其余原配方、scope、builder、机会数组、候选和增长账保持。

最小反例保留真实 A/B algorithm union 与独立 FullLogits terminal 组；增加不产生新 key/member 的空 Unknown 声明时，原有同一条聚合 Unknown gap 已足够。旧代码将同一存活阶段的额度从 255,036 抬至 318,524 bytes，完整计划在原额度处被拒，0 pass/1 fail。修复后两种声明数均为 234,204 bytes，并保持全部代表、scope、schedule、5 条 gaps 和三项实际工作量账；零工作额度场景仍保留 14 条 gaps 及全部请求/动作/行数拒绝。少 1 字节仍在 geometry 前拒绝。最小测试 1 pass，整个 layout 组 120 pass、0 fail、0 ignored，fmt/diff 通过。首个 fixture 曾把 terminal 放入同一个 Greedy family而先触发夹具断言，已保留该失败并以真实独立 product 修正；它不算生产修复前证据。

按 G10 Metal 原 inventory 及实际类型尺寸核算，重复项为 9,389,280 bytes，去重后的保守峰为 32,943,962，低于当时可用 34,195,039。此为同一存活对象的计费纠正，不是新增预算、删除样本或真实内存占用节省的测量。G12 未单独运行实机，已随 G13 组合候选完成规定 workspace/backend checks 和四入口运行；Metal serve 的真实 composition 授权通过，但准备、收样和 SLO 缺口仍见开头，不能把容量门通过写成完整校准通过。

此外，只读审查确认 Source8 可在一次真实 capture 中保留多个独立 host/product family，各自完成 F/R/Q。当前 selector 的分组、优先级和单一 universe 限制了打包。减少来源槽可能因共同采样周期反增执行工作，必须先重算请求、动作、行数、owner 和内存账；没有证据证明该方向能在 120s/2048 内覆盖必要策略，尚未实现。外部记录为 `g10-multi-family-capture-design-audit.json`。G11/G12 的局部结果分别存于 `g11-validation-status.json`、`g12-validation-status.json`，不覆盖上面的 G10 冻结负结果。

G13 修复声明前的一处已复现遗漏：`local_universe` 曾先按目标 numerical family 过滤关联配方，再构建固定算法集合；同一 case 的合法后继轨迹若属于另一 host/product family，其算法也被一起删除。但完整 cohort 的物理投影早于 numerical family 成员判定，因此不能用训练族筛选代替原物理轨迹的声明。现在 builder 读取所选原 case 的全部已关联配方，仍保留原 facts 的统一 family、代表、成员下限及独立 F/R/Q；未关联算法不引入，seed 不包含后继算法或 builder 越界时仍放弃整个 scoped candidate，原 raw 计划保留。collector、数值接受规则及各预算未改。

最小 typed Rust 反例保留目标 A/B、同 case 关联的另一 product C，以及仅在 global seed 中的无关 D。旧代码在 C 的声明成员检查处失败，0 pass/1 fail、0.06s；修复后 C 可物理投影但仍不同于目标 family，D 仍被排除。seed 缺 C 时的回归保持原代表、成员、cycles 和三项工作量。相关 `local_scope_` 六项通过，整个 layout 组 122 pass、0 fail、0 ignored。另一个完整独立 F/R/Q 与证书重放回归同时覆盖旧 FittedResidual 和生产 IdentifiedFitGlobalResidual：仅测 A 可以资格并预测 A，已声明但未测的 B 严格拒绝为 QualificationCoverage。首轮测试曾错误地要求更后的 UnidentifiedDirection，失败已保留；查明 phase coverage 先于数值求值后修正断言，生产门未改。输入 readiness V3 与 envelope challenge 是两个不同字段，本回归不声称覆盖整个输入准备流程。

G13 组合候选 `1ad2642e` 已完成全仓检查、准确源码对应的双后端构建及四入口实机运行，结果见开头。CUDA 当前 632-cohort 计划完成，G11 的物理投影拒绝未重现；但 G11 没有记录具体缺失算法，且原计划为 606 cohorts，不能把这次成功写成原 wave 的逐项复现。新增声明没有放宽资格，普通 CUDA serve 的预测采用仍为零，必要域覆盖和 SLO 未完成，不进入正式矩阵。局部反例和失败记录继续保留在 `g13-validation-status.json`。

## G9：同一候选的四入口实机结果（历史）

2026 年 10 月 3 日，冻结候选 `c00a0b30` 的 Metal/CUDA 正常 `run` 与 `serve` 均已结束。业务代码对应 `88e298de`，后续差异只有两个 inventory 测试计账和文档。当前仍无完整可交付版本，正式验收为 **0/224**。完整校准计划成功、普通请求实际采用和客户端 SLO 分别判定；来源数量不是独立的 4/4 门槛。

规定的本地 fmt、workspace check/test、Clippy（`-A warnings`）和 Metal 全目标编译均通过；全仓测试为 7863 pass、0 fail、109 ignored，未重复累计子进程测试。另有两个 release 默认 2ms 受控 CPU 探针通过；它们不证明真实并发规划可行。配置好的 CUDA 主机完成 pinned H39 lock 下的规定 CLI check 与 release 构建，两端实际 Cargo artifact、冻结二进制和同一份 3351 文件源码清单已核对。

| 入口 | 本次实机结果 | 结论边界 |
| --- | --- | --- |
| Metal `run` | 120s 内完成 271/448 cohorts，保留两份已发布来源；原 512-token 作业结束 | 来源数不作为覆盖率，达到原输出上限不算完整计数任务；不能替代 serving SLO |
| Metal `serve` | 120s 内仍为 705/1773 cohorts、一份已发布来源；普通窗口 witness 实际提交和回执均增加 108 | 三项延迟 P99 仍失败；合并候选在保留原始输入时的容量门仍不通过 |
| CUDA `run` | 9 次实际 private acquisition ACK；在 source 0 的 cohort 9、call 521 停止，启动未发布；原 1536-token 输出与 G8、先前 G9 逐字节一致 | 原主机结算完整，但物理凭据为空，报 `incomplete original source5 preparation settlement`；随后 live 发布不能补成启动成功 |
| CUDA `serve` | 完整 inventory 消耗 16156 次投影，55.22s 完成选定的 297 cohorts，并发布四份来源 | 准备复用首次在原 16384 上限内通过这条实机路径；普通窗口 witness decision/submitted/reconciled 均为 0，尚无实际采用闭环，TPOT P99 失败 |

两端 serving 都是原 pinned ShareGPT、固定 C8、32 warmup + 32 measured、1 次重复、Qwen3.5-9B Q4_K_M / KV fp16；各 64 请求完成且错误为 0。measured 均为 9210 usage tokens、9179 非空可见 SSE 文本事件、9147 可见间隔，32 条输出长度均匹配原 reference policy。两条 event/usage 不一致请求仍计入可见停顿。以下仅为因果诊断，不是正式主表或自身 Off 对比。

| 后端 | TTFT P50/P99 ms | TPOT P50/P99 ms | 可见 SSE ITL P50/P99 ms | 输出 tokens/s | 峰值内存 | 原 SLO |
| --- | --- | --- | --- | --- | --- | --- |
| Metal | 967.888 / 4835.555 | 192.967 / 235.942 | 174.600 / 872.762 | 37.083 | MTL 8.008 GiB；host RSS 0.536 GiB、footprint 1.980 GiB | 三项 P99 失败，joint 3/32 |
| CUDA | 69.362 / 138.113 | 18.975 / 20.523 | 18.013 / 39.315 | 344.629 | runtime requested 21.622 GiB；NVML 整卡 23.121 GiB；host max RSS 6.366 GiB | TTFT、pooled ITL P99 通过，TPOT 失败；joint 1/32 |

峰值覆盖进程启动至退出；采样可能漏过瞬时高点。内存口径重叠、不相加；WSL 的 NVML per-process accounting 不可用，整卡占用不能称产品进程独占。单轮不提供重复间置信区间；pooled ITL 达标不等于逐请求联合达标。CUDA 普通候选查询仅 1 Known、439 WrongDomain，且发生大量规划预算耗尽；现有计数不含逐请求/family/shape 关联，不能据此断言唯一原因或宣称必要域已覆盖。

本轮纠正了一个关键推断：Metal 实际 composition 需要 42,333,242 bytes，可用 28,278,996；先前引用的 55,717,403 是拒绝 composition 并释放原始输入后的 raw selection 额度。二者不能互换。局部 CPU 回归没有证明真实释放前的容量门可行，27,438,407 bytes 差额也不是新增 geometry cache 的持久占用。下一步针对同一真实 outcome 被各 case 深复制的原配方，在完整存活期和引用账下验证共享，先取得最小反例，不提高预算或提前丢弃 scope 投影所需输入。CUDA serving 同一 composition 容量门也未通过，尚不能把四份 raw 来源视作完整合并支持域。

CUDA `run` 的失败已沿原凭据链定位：实际 adaptive graph capture/replay 保留全部原 native commands 并标为 Replayed，而 direct invocation 使用单条物理调用及 sealed logical expansion；当前投影把两种表示都要求完整 direct replay segments。真实日志显示 capture/upload/replay 各一次、原主机行完整，但 physical evidence 为 None。后续最小验证保留 GraphPath Unknown、direct 缺账拒绝和原 FIFO/时钟边界，核实 runtime 已声明的 direct replay operation 能否让物理投影正确区分两种表示；不伪造 Known shape 或放宽最终 collector。

Metal 首次 serve 在产品启动前因新增 `RUST_LOG` 与原 guard 的自有过滤器冲突退出，已保留失败并用移除该覆盖的独立 r2 目录运行。CUDA 首次 release 因原容器 `sleep 21600` 到期中断，既有缓存下重新构建通过；这两项不混算成产品回归或成功运行。四入口工作结束后 CUDA 原服务已恢复，最终 PID 197273、健康正常。原 dirty 工作区未改。

本轮证据在 `/private/tmp/ferrum-slo-recovery-20261002`：`g9-final-local-test-audit.json`、`g9-c00-backend-build-identity-audit.json`、`g9-final-metal-run-results.json`、`g9-final-cuda-run-results.json`、`g9-final-metal-serve-results.json`、`g9-final-cuda-serve-results.json`，以及四个 `g9-final-*-evidence/` 原始目录。`g9-metal-composition-pre-release-audit.json` 和 `g9-c00-normal-workload-coverage-checklist.json` 分别记录容量推断纠正和验收边界。原 300 条计数 prompt 是历史 CLI 因果诊断，不是另加的最终业务门槛；自然终态与完整输出合同另用已有明确语义任务核验。

随后仅新增最小 Rust 反例，未修改生产：真实 `collect_ready → freeze_inventory` 在释放前因同 outcome 原配方深复制而拒绝合并，0 pass / 1 fail、0.32s；完整原 replay commands 的 typed projector 仍无物理凭据，0 pass / 1 fail、0.06s。日志为 `g10-shared-recipe-before.log` 和 `g10-adaptive-replay-physical-before.log`。新增真实 CUDA cold graph 结构断言尚未执行；CPU 失败不冒充该 GPU 证明。下一轮改动与验证另记，不覆盖本轮冻结负结果。

## G9 局部检查与先前诊断（历史）

native 前缀准备已接入共享自动启动。以下记录早于上面的 `c00a0b30` 四入口结果；其中的待验证状态不得当作当前结论。G8 四入口证据固定对应 `8307ffec`，先前 G9 诊断对应 `186c0008`。

G9 当前已完成两项局部修复：`38e8b069` 将 Metal 合并候选的 retained scope 预留从重复 key 引用数改为实际存活候选上界；原失败回归修复后通过，selection 全组 54 pass。`196c8df3` 在同一次 captured view 内复用已验证的 prefill 路径；完整小场景从 33 次实际投影降为 23 次，保留 Share/Replay 全部输出、Unknown、分支和容量检查，geometry 全组 29 pass。首次全组的旧 eviction 计费断言失败已保留：单槽复用使合法准备从 14 次降为 13 次，按逐动作账更新后通过。两项修复均未扩大预算；尚未完成这份候选的 workspace 和实机验证。

完整 CUDA 准备账的静态估计表明，同 view 合法复用可能将剩余完整 inventory 所需总投影从 18076–18124 降至 16242–16290。该范围假设没有新增 readiness 工作，仅余 94–142 次，不是已满足 16384 上限的实测证明。缓存不跨 captured view，元数据单独计费；淘汰只导致重放，不删分支或资格工作。

G9 CUDA `run` 诊断实际执行的是仅增加观测的 `186c0008`，不含上述两项业务修复。原 selector/submission 对接通过，call 506 在 source 0 第 6 个 cohort 的 `outside_host_work_or_policy` 联合门拒绝；启动 0/4、5/471，剩余请求和执行额度充足。该日志不能独自区分政策、generated counter 与 work 三项条件。源码追踪发现真实前缀安装刻意保持 empirical domain 为 None，而普通 outside 校验要求普通内容资格；后续私有反馈证明及最终 prefix 收据还各有 actual-shape 要求。G9 与 G8 计划分别为 471、632，不能称同一 call 的精确复现或性能改善。

G9 当前候选将实际物理工作凭据与数值 graph 资格分开：只有原始、单次完成的提交及精确 row 绑定可保留物理凭据；真实安装的每行前缀授权、完整 host 结算和同一 Arc 的私有证明共同进入 source8 最终消费。原 `GraphPath` / `Unknown` 不变，不授予训练或普通请求资格；source5 writer 没有新增 outside wrapper。新增保留对象纳入原 rows/bytes 容量账，原请求、动作、时间和来源上限不变。

真实安装→Core selector/submission→采样→FIFO→source8 collector 的受控 CPU 回归，修复前因缺少 prefix host settlement 失败，最终修复后 1 pass、3.18s；两个请求完整输出并释放，仍无成本快照。最小前缀正反例 2 pass，模型物理投影边界 3 pass，记录器边界 6 pass，scheduler prefix 组 7 pass。这些组含重叠检查，不累计成完成度。中间夹具失败（缺少后台 worker、误拒合法参数绑定、错误历史触发 builder sticky error）及模型测试的私有 API 编译失败均保留日志；最终只修正夹具，未放宽生产接受条件。完整 workspace、后端编译和这份候选的实机结果仍待完成，不能把 CPU 协议通过写成 CUDA 实际前缀复制或完整 F/R/Q 成功。

业务修复冻结于 `88e298de`，默认全目标 workspace check 通过（1m32s）。首次 workspace test 在 engine 取得 1531 pass、2 fail、6 ignored 后停止；两项失败均是 inventory 的旧精确投影账：跨宽度复用后实际为 5、旧断言为 7，三个 continuation span 复用前驱后实际为 3、旧断言为 6。按原每次必要查询和新增 owner 准备动作重新推导，仅更新两个测试计数及说明，生产源码不变。随后 inventory 全组 8 pass、0 fail、0 ignored（0.26s），原资格、gaps、未提交和清理断言均保留并执行。首次失败原始日志保留；全仓复测、Clippy、后端编译和实机仍待完成。

G9 CUDA guard/product 均退出 0，服务恢复记录为 `restored_verified`；独立检查确认 PID 4147753、服务 active、健康响应正常。原 1536-token 输出与 G8 字节一致，但仍在计数任务中途达到上限，不能称完整任务完成。原始证据及独立审核位于外部 `g9-cuda-route-evidence/`；Metal 计费证明及 after 审核分别为 `g9-metal-composition-memory-proof.json`、`g9-metal-composition-memory-after-review.json`。

## G8 已冻结的实机结果与当时判断

2026 年 10 月 3 日 G8 实机结果：同一 `8307ffec` 源码的四个正常入口均已运行结束，Metal 和 CUDA 构建及 CUDA 真实 guarded checkpoint 生命周期用例通过。规定的本地 workspace 检查通过，engine 全量结果为 1525 pass、0 fail、6 ignored；两个 release 默认 2ms 小场景探针通过。这些结果不替代下面的产品失败。原工作区的 314 项改动保持不动；CUDA 原服务已恢复并验证健康。正式验收仍为 **0/224，没有可交付的完整 SLO 版本**。

| 正常入口 | 已取得的实机证据 | 尚未通过的部分 |
| --- | --- | --- |
| Metal `run` | 原 512-token 输出与上轮同参数结果逐字节一致；417 次普通推理 witness 提交并配对，欠估为 0 | 启动仅完成 2/4 sources；417 次 wave 与 512 tokens 不是覆盖率；原 token 上限使计数任务停在中途，不能称任务全部完成；未证明延迟 SLO |
| Metal `serve` | 32 warmup + 32 measured 请求完成、错误为 0；77 次普通 witness 提交并配对 | 120s 内完成 705/1773 cohorts、1/4 sources；TTFT/TPOT/可见 SSE ITL 三项 P99 均超门槛 |
| CUDA `run` | 原 1536-token 输出逐字节一致；9 次私有 prefix ACK；独立真实 capture/restore 测试通过 | source 0 的第 9 个 cohort 在恢复后推理收到 GraphPath Unknown，缺少原始 host settlement，启动完成 0/4；后续 live 发布不补成 startup 成功 |
| CUDA `serve` | 64 请求完成、错误为 0；196 次普通 witness 提交并配对 | 准备阶段耗尽 16384 projections，43 个模板完成、4 个待完成，未形成 startup source；TPOT P99 超 15ms，联合达标为 0/32 |

两端 serve 使用固定 ShareGPT 选样、C8、32 warmup + 32 measured、1 次重复、Qwen3.5-9B Q4_K_M / KV fp16。以下是本次诊断结果，不计入正式主表，也不证明相对自身 Off 的收益。ITL 为客户端相邻非空可见 SSE 文本事件间隔；角色、空与结束消息不计入。

| 后端 | TTFT P50/P99 ms | TPOT P50/P99 ms | 可见 SSE ITL P50/P99 ms | 输出 tokens/s | 峰值内存（进程启动至退出） | SLO |
| --- | --- | --- | --- | --- | --- | --- |
| Metal | 956.954 / 4872.694 | 193.360 / 224.611 | 176.589 / 877.344 | 36.729 | MTL allocation 8.008 GiB；host RSS 0.528 GiB、footprint 1.981 GiB | 三项 P99 失败；请求联合 3/32 |
| CUDA | 67.484 / 128.315 | 18.627 / 20.937 | 17.515 / 37.942 | 339.576 | runtime allocation 21.622 GiB；NVML 全设备 23.124 GiB；host RSS 6.362 GiB | TPOT P99 失败；请求联合 0/32 |

每端 measured 输出均为 9210 usage tokens、9179 文本事件、9147 可见间隔；32 条输出长度均匹配原选样。两条 usage/event 不相等，不能改用单 token 诊断删掉这些请求。内存口径存在重叠，不能相加；采样峰值可能漏过瞬时高点。单轮没有重复间不确定度，pooled ITL 分位通过也不等于逐请求联合达标。

本轮已经被实机否定或限制的假设：

- `c6a9495b` 的零起点 readiness 完整结果复用有 CPU 反例和修复证明，但 CUDA serve G6/G8 的 219 次短探测消耗序列完全相同，总准备投影仍为 16384。核对真实顺序后确认：42 次执行准备和 13 次资源动作之后的探测全部从非零位置继续，没有进入新增的从零复用分支。模板 21 的三次从零扫描属于不同 group 的初次扫描，不能当成同一结果重复捕获。此前把 CPU 目标零场景的收益外推到真实 CUDA 负载是错误的，该修改没有解决当前产品预算问题。
- `24635d3f` 的兼容算法合并在 Metal run 形成 34 算法、130 cohorts 的独立 F/R/Q source，并带来普通请求实际采用；Metal serve 仍是原 1773 cohorts，不能把 run 的节省外推到 serve。当前证据尚不能区分全 inventory 内存授权与局部算法投影兼容门的影响。
- 修复 guarded transfer 的逻辑归因后，真实 CUDA capture/restore 已通过；正常启动却在恢复后的推理反馈处失败。这是不同执行层。现有日志不能唯一确定 selector 的 partial-program 路径还是后续 graph/host 证明被拒，不能再将 transfer 通过写成完整启动通过。

G8 后首先执行上述三处的可证伪判定：准备总账区分必要独立资格工作、可消除重复及未测成本；查明 Metal serve 未使用合并的实际原因，并按 CUDA 实际探测顺序建立预算反例；以最小诊断取得 CUDA 失败调用的原 selector/submission/settlement 拒绝证据。诊断源码 `186c0008` 不改变业务接受条件；开关等价定向测试通过，受影响 route-population 组为 70 pass、0 fail、1 ignored（原 settings 外部诊断），238.90s，fmt 和 diff 检查通过。后续 G9 结果见本文开头；完整 workspace 与四入口实机证据仍只对应 G8。

再次修改业务源码的前提是：旧行为有最小失败复现，修改保持原资格和资源规则，预期工作量或错误路径有直接依据。若现有设计在原预算下无法容纳必要覆盖，应修订准备/选择设计并重新证明，而不是增加超时、降低资格门或再补一条特殊分支。下一次集成仍由同一候选的四个正常入口和原 SLO 判断；四入口未满足前不启动 224 单元正式矩阵。

预算复核确认：Metal serve 原计划预约 1792 requests、5942 execution actions，readiness 已使用 181 个实际请求，完整预约后请求余量仅 75；两轮均先耗尽时间。G8 source 1 的 13 次 native setup 约为 1.96s，后续 collection 约为 98.77s，继续优化 capture 本身不能解决主要成本。当前完整计划两轮实际失败，但必要支持集合的最少请求、投影、动作与耗时下界仍未知；不能将 705/1773 线性外推成任何合法方案都不可行。CUDA 尚未完成 inventory，更没有可供评估的完整 source 采样计划。下一项可审查交付应是“所需 workload 支持 → 原代表义务 → 合法分组 → 独立资格样本 → 准备及执行总账”，先判定可行性，再恢复业务实现。

完整原始结果与独立核对位于外部证据目录的 `g8-metal-run-evidence/INDEPENDENT-FUNCTIONAL-REVIEW.json`、`g8-metal-serve-evidence/INDEPENDENT-FUNCTIONAL-REVIEW.json`、`g8-cuda-run-native-results.json`、`g8-cuda-serve-results.json`；统一状态为 `g8-validation-status.json`。下方实施步骤和末尾 G1/G2 记录保留其历史范围；涉及 `87cd94c7`、连接不可用和旧 raw+额外 union 方案的表述不代表当前状态。

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

## 交接时的起点与独立风险（历史）

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

本轮外部证据使用 `/private/tmp/ferrum-slo-recovery-20261002`。初始只保留当前计划、结果摘要、必要原始记录和对应源码身份；复用现有 Cargo/cache，不重复归档依赖树。硬件候选保留对应的冻结源码和原始运行证据。构建前检查磁盘容量，必要时通过 Cargo 清理明确可再生的本任务构建产物，清理范围与结果留在外部记录。

## G1/G2 历史实施记录（由开头的 G8 状态更新）

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
