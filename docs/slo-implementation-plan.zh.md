# SLO 持续吞吐优化：完整实施与验收清单

状态：实施计划；下列未勾选项均未完成，不能依据文档存在宣称功能已交付。
依据：`Ferrum_SLO_Throughput_RFC_v2_2c9998a.md`（2026-09-22，RFC v2）。
历史入口：`/private/tmp/ferrum-efficiency-evidence-20260920/HANDOFF-20260922-STOPPED.zh.md`。
本轮证据目录：`/private/tmp/ferrum-efficiency-evidence-20260920/slo-goal-20260922/`。
用户已明确恢复实施；旧 handoff 的停止状态保留为历史记录，不作为当前执行指令。

2026-09-24 用户要求算法作为长期核心，先系统调研和设计，并参考类似工程但不照抄。
新增 [核心算法设计与取舍](slo-algorithm-design.zh.md)：固定第三方阅读版本，区分可借鉴机制与不适用假设，
采用统一纯状态转移、共同计划优先的有界规划及独立提交校验，保留默认完整完成请求和双后端验收。
设计正分阶段迁移；A0–A5 是本清单 M3–M5/K0 的细化，不替代 R0–M5 的全部完成条件。
2026-09-24 A0 独立有限枚举器、A1 scheduler 不可变后继及 engine 增量路线适配已提交；
A2 共同计划构造、按需合法前沿及共同时间基点评分已提交为 `79570512`。
当前集成工作树的默认 workspace 测试为 5876 通过、72 忽略，fmt/check/clippy 和 Metal 编译通过；
另有 91 项 Metal 控制器测试通过，包含真实 partial→final、decode 和 mixed 投影/执行对照。
这不等同于所有 Metal 运行测试、独立 clean HEAD 或服务性能验收。A1 新构建 9B 校准的 32 请求、6432 输出 tokens 已完成；
独立 heldout 仍有 59 次 Unknown 和 48 次低估，不能将采集完成当作预测器或 SLO 达标。
A1/A2 服务收益继续使用各自固定构建单列验证。
原前缀缓存候选保留为仓库外实验，不继续作为主线补丁集成。研究结论不能用来宣称性能目标完成。

## 目标、基线与边界

2026-09-23 用户明确产品定位为并发服务，**C4/C8/C16/C32 是主验收范围，C1 降为回归诊断**。
Metal 与 CUDA 都按该并发范围推进；C1 结果如实保留，但不以追平单请求性能阻塞并发优化。
该优先级调整不改变下述延迟、吞吐、完整输出和错误/拒绝要求，也不删除已测负结果。

2026-09-22 用户补充的最终交付要求：使用 **Qwen3.5-9B** 做同硬件并发比较，
证明 TTFT、TPOT、客户端可见文本 ITL 明显优于 llama.cpp，同时吞吐基本持平；
表格必须标注峰值 GPU/Metal 内存，并在达标后**发布新版本、更新中英文 README 和官网**。
这些是当前 goal 的必要完成条件，不以成本模型/调度代码完成替代性能或发布验收。
2026-09-23 用户明确优先完成请求：默认不因预测或实际延迟超标、成本 Unknown、
搜索无可行 SLO 序列而拒绝、取消或缩短输出。时间准入控制活跃执行时机及服务承诺；
失约后保留原始计时和失约记录，继续安全、有界、公平的 best-effort 调度。
SLO deadline 不默认等于请求终止时间；独立配置的请求超时和客户端取消保持原语义。
无效请求、模型/上下文硬限制、实际接收容量耗尽等仍明确处理，不能无限扩充等待区。
严格时间承诺下的快速拒绝仅作为显式可选策略实现，主性能比较不启用它来降低负载。
工程预声明默认：三项延迟的 P50/P99 六项各降低至少 10%，吞吐至少为 llama.cpp 的 95%；
这是执行方的量化默认，用户后续指定优先，正式 A/B 前冻结比较并发范围和观察窗口。
2026-09-23 用户补充“吞吐不能下降太多”：另要求 successful output tokens/s
至少为同构建 Ferrum Off 的 95%，同模型/精度、输出传输、固定容量和工作负载测量。
5% 降幅是工程默认，不能声称用户给定了该数值；配对重复的不确定性须一并报告。
两条吞吐线均为硬条件，不能用相对 llama.cpp 达标掩盖 Ferrum 自身吞吐明显回退。
报告全扫描和交错重复、不确定性、输出有效性、失败/拒绝及样本数；不能看候选结果后缩范围、放宽阈值。
GPU/Metal sampled peak、OS footprint、max RSS 分列，明确采样与统一内存边界。
最终 README/官网表格使用同一份核实证据；无实测结果前不填写胜出数据。
2026-09-23 用户补充：**算子执行效率属于核心范围**。新增 K0 主线，与调度闭环并行；
基础执行明显不足时优先优化真实关键路径，不能以更复杂调度掩盖算子低效。

2026-09-23 用户新增 **RTX 5090 / CUDA 开发测试与硬验收**，保留 MAC MINI 2 / Metal 原主线及全部目标。
CUDA 于 2026-09-23 已完成 5090 连接核验、CUDA 12.8.1 工具链、四组原生算子及 CLI release 编译，
并通过实机运行时/算子绑定验证；**尚未完成 9B 模型质量或并发性能验收**。
不再以“有主机再测”或可跳过处理。历史 unavailable/未测记录保留为原时点事实。
连接配置位于 `.env.local`，文档不读取或记录 endpoint、凭据与密钥；开发测试授权不包含
任意删除远端文件、修改系统服务或干扰无关负载。先核验驱动/工具链、资源和 pinned operator-set lock。
Metal 与 CUDA **分别**使用 Qwen3.5-9B，在各自同机对比 llama.cpp 与同构建 Ferrum Off；
同后端内模型内容/精度、数值/KV、pinned ShareGPT 选样、完整输出政策和计时口径一致。
每个后端均须六项延迟达到冻结门槛、successful output TPS 同时达到两条基线门槛，
并完成固定服务容量下的完整并发扫描、交错重复及完成率/错误/拒绝/pending/稳态验收。
跨后端分表记录配置和证据，不合并指标或互相代替结论；各自公开峰值 GPU 内存口径、采样覆盖，
CUDA allocation/reserved/进程使用量不能混称，主机 footprint/RSS 按平台可得性单列，缺项如实标明。
新版本与中英文 README/官网须包含两后端已核实结果；任一后端未完成，整体目标仍未完成。
2026-09-23 07:55 UTC 已通过 goal 工具确认原目标为 active；沿用原目标，不重建或宣布完成。

- [ ] Qwen3.5-9B 同硬件、同精度、同 ShareGPT 样本/输出政策的并发验收达到上述延迟/吞吐目标。
- [ ] 同构建 Ferrum Off 对照确认开启 SLO 后成功输出吞吐下降不超过预声明上限；Observe 开销独立测量，不减少接收工作量换取延迟。
- [ ] 记录峰值 GPU/Metal allocation、OS footprint、RSS 与完整测量条件、误差及复现证据。
- [ ] 按仓库流程实际发布新版本，release notes 说明实现与已验证支持范围。
- [ ] README.md、README_zh.md 和官网更新相同数据表与条件，并核查发布后的线上内容。

目标是在固定模型质量、硬件和工作负载分布下，提高满足 TTFT、TPOT、ITL、成功率及拒绝率要求且无持续积压的最大到达强度 `λ_safe`，同时报告实际输出吞吐。
完整范围包括 K0、R0 与 M0–M5：算子与执行效率、有限候选、有界前瞻、动态时间准入、输出隔离及实机容量验收；仅实现 guard、配置、Observe 或静态 admission 不算完成。
首个 Enforce 配置必须具有真实分段/回执能力；其他后端明确报告支持范围，禁止无声回退后仍标注 Enforce。
共享引擎同时覆盖 `ferrum run`、`ferrum serve`；HTTP 特有计时、协议、过载响应另测，CLI 无网络边界但仍有慢消费者。
不在线改变模型精度、KV 格式、并行拓扑、用户输出上限，不新增推测解码或 P/D 分离以替代本任务。

| 已有基础（需要复用） | 未具备或尚未验证的能力 |
|---|---|
| ContinuousBatchScheduler、工作 generation、物理资源 authority/readiness | 统一入口时间预算、候选纯投影、时间准入及 deadline 控制 |
| native prefill chunk 回执、split/unified 执行路径 | 每条 Enforce 路径的强工作上限、真实返回边界与 fallback 保证 |
| ShareGPT 固定抽样、closed/open loop、SSE visible gap | 联合 ITL SLO、last-visible TPOT、开放负载送达/积压及容量验收 |
| v3 raw request timing 已构建、历史 workspace/Metal 检查 | v3 真实运行验证；当前 dirty capture 补丁的编译、语义与选路验证 |
| prefix rendezvous、压力释放、completion fence | 同一时间预算下的 prefix 等待、必要维护、独立时间唤醒 |

RFC 代码基线为 `2c9998a2`，交接分支 HEAD 为 `10cfda02`，二者与当前 dirty 内容必须分别记录。
历史 C4 仅单轮 N32：吞吐领先约 35.35%，TTFT P99 落后约 8.51%；其成功、编译和测试记录均不覆盖后续未完成补丁。
历史结果是诊断入口，不是新实现验收；模型输出 token 数一致也不等于数值或语义等价。

## 依赖与交付方式

顺序为 `R0 → M0 → M1 → M2 → M3 → M4 → M5 → 完整容量验收`；无依赖的类型/虚拟时钟测试可以并行。
K0 在恢复可信测量后与上述实现并行，真实执行瓶颈的优化不等待 M5 完成。
每阶段交付代码、配置/help/effective-config、必要 Rust 测试和外部验证证据；更新本表须附实际路径和结果。
M1/M2 的执行/隔离收益、M4/M5 的控制算法收益分开测量，避免一个改动同时改变 kernel、数值策略、缓存和调度。
恢复修复保留既有无关改动；禁止全仓 reset，禁止把所有 dirty 内容当成本轮新增。
root 统一管理 Cargo、SSH 和 GPU；代理限明确文件范围，一次只运行一个设备 workload。

## K0：算子与执行路径效率（核心交付）

- [ ] 复核交接中 Qwen3.5-9B 的真实 profile 和负实验，记录 prefill/decode 主要耗时及未知部分。
- [ ] 当前配置中分清 GPU 算子、上传/回读/同步、CPU 编码与调度等待；采集开销单独测量，不把 wall 残差当作已知瓶颈。
- [ ] 按实测耗时占比和改善空间选择关键量化矩阵乘、注意力/GDN、输出头、融合或搬运路径；名字和理论算量不能替代热点证据。
- [ ] 受影响算子验证数值/语义及真实后端行为，共享变化覆盖 run/serve；不隐改模型精度、KV 格式、输出政策换取速度。
- [ ] 以同配置 Ferrum Off 的优化前后对照验证9B基础执行收益，再比较同一优化构建 SLO On/Off，分别报告算子收益与调度开销。
- [ ] 重复测量并报告整波墙钟、端到端六项延迟、成功输出吞吐和峰值内存；微基准提升不代替最终服务验收。
- [ ] F0/Q128、M128 staging 等既有负实验仅在新机制或测量证据支持时重试，保留失败结果，不筛选有利样本。

落点：`ferrum-kernels`、`ferrum-models` 及必要的执行编排/原生算子实现；复用 Rust benchmark 和真实后端测试。
完成条件：已识别的关键执行瓶颈有可复现实验和优化结论；基础执行与 SLO 调度收益分别核实，最终联合验收不降低标准。

## R0：恢复到可验证的执行与观测基线

- [ ] 核对当前代码、worktree/upstream、历史源快照与冻结 binary；登记既有、半成品、本轮改动的归属。
- [ ] 完成或精确撤回首 N 帧采集半成品；核查 CLI 模块缺失测试、run/serve flag、配置映射与有效配置 provenance。
- [ ] 将日志捕获额度与执行路径能力分离；达到 cap 前后不得因局部 timing 变 Off 而改变 reusable dispatch。
- [ ] 验证 request incarnation、共享 physical wave、混合 cap、成功/失败/取消 terminal summary；partial 范围必须显式。
- [ ] 首 token 前多 chunk 的覆盖按实际回执验证，不假定 N 足够；低成本观测不得依赖无限 Kernel JSONL。
- [ ] v3 client 先验证 raw timing/关联 ID/失败占位，完成最新 llama C4 采样及 host/thermal 审计。
- [ ] 以同样本、原输出预算和 C4 找到实际两条 TTFT 尾请求，关联入口、commit、chunk、physical wave 与输出事件。

落点：`ferrum-cli/{config,runtime_env,commands/profile_capture,commands/run,commands/serve}`、types runtime config、interfaces event/device、engine profile、models executor。
验证：最小受影响 Rust 测试、默认路径等价、cap 前后路径一致、必要 workspace/Metal 检查，再做有界诊断。
完成条件：当前工作树可验证，观测边界与开销明确；旧检查不可借用，未知残差不得填成“排队”。

## M0：时间、SLO、负载及报告契约

- [ ] 在 `ferrum-types/src/slo.rs` 定义可序列化 service class、边界、模式、预算、分位/联合达标率、拒绝率及观察窗口。
- [ ] 在 interfaces 建 runtime envelope/context；可信单调 ingress 从入口前部贯穿模板、tokenization、内部等待、engine，明确 body 接收范围。
- [ ] `Instant` 不进入 wire JSON；旧入口给明确默认，不能丢 context 仍声称 Enforce；未授权 metadata 不决定 priority/时钟。
- [ ] 内部 ingress→token commit 与外部客户端 SLO 分开；传输/投影预算须验证，预算非正拒绝配置，不能相加多个 P99 伪造端到端保证。
- [ ] 保留 legacy `(terminal−first_visible)/(usage−1)` TPOT/schema，新增 last-visible、terminal、token-commit 各自时间及证据状态。
- [ ] 新版联合判定覆盖成功、TTFT、TPOT、request-max visible ITL；同时保留 pooled visible ITL 和 strict-token eligibility。
- [ ] accepted 与 offered 联合达标分别命名；前者须有可观测服务接纳证据，后者保留拒绝，不能以 offered−rejected 推算；接纳不等于 GPU 准入或时间承诺。
- [ ] Unknown 不算 Pass，单 token/少于两次可见更新为 N/A；失败、拒绝、pending/右删失均留分母和原始记录。
- [ ] 区分 raw output、successful output、request goodput、SLO output goodput；消除当前代码含失败部分 token 与文档成功 TPS 的歧义。
- [ ] 开放负载保存 scheduled/actual send、客户端 backlog、实际到达率；发送计划不因响应完成而退化成隐性闭环。
- [ ] 增加服务器队列长度/最老等待年龄、offered/received/promised/completed/rejected/failed/pending 和分类统计。
- [ ] 明确发送窗口加排空与持续时间窗/cohort 两种统计，分别标注跨 repeat P99 均值和 pooled P99。
- [ ] 预声明 workload-specific SLO；无绝对阈值时只作同负载 llama 三项 P99 相对参照，不能包装成绝对服务承诺。

落点：types requests/config、interfaces engine 入口、CLI run/bench_serve、server axum_server、bench-core `{lib,slo,sse_text_event_gap}`、performance-evaluation 文档。
验证：虚拟时钟、旧 JSON 兼容、0/1 token、合并事件/长停顿、无 usage、失败/拒绝/超时、原始分位数复算；同配置 A/A。
完成条件：每个指标的起止、分母、资格和失败语义可解释，完整原始记录可重算；不得静默改写历史表。

## M1：执行工作上限、返回边界与回执

- [ ] 在 scheduler/model_executor interfaces 声明 supported wave、合法 chunk/对齐/最大工作量和 capability；不支持组合显式拒绝 Enforce。
- [ ] `tokens_to_process` 为强上限，prefill `[offset, offset+count)` 连续且未越界；普通 decode 每请求最多一个新目标 token。
- [ ] legacy prefill 拆开一次初始化与单段执行；提交 KV/recurrent/position/offset 后非最终段返回调度器，不采样、不重置历史。
- [ ] 最终 prefill 保持首 token sampling、stop、usage 语义；prefix restore 后仅推进未恢复范围。
- [ ] PlanRuntime split 一次提交一个真实 wave，D→P/P→D 是两个可重规划动作；保留原生 unified mixed 能力。
- [ ] capacity shrink 只执行较小合法前缀并返回实际范围；Unsupported fallback 不扩原授权范围或循环吞掉全部 backlog。
- [ ] 回执区分 NotSubmitted/Deferred/Completed/PartiallyCompleted/FailedAfterSubmit；部分或未知提交禁止盲重试。
- [ ] 提交前校验 identity、work generation、取消、资源和实际当前时间；只释放本次尚未提交 reservation。
- [ ] 复用现有 completion fence/pressure authority，取消和超时不能提前释放设备仍引用的资源。
- [ ] 执行期间不持全局 scheduler/sequence 写锁；真实回执确认前不提前提交逻辑进度。

落点：`interfaces/{scheduler,model_executor}`、`engine/continuous_engine/inner/{batch,prefill,decode,mixed}`、models executor、受影响 backend。
验证：范围/进度而非调用次数；容量缩小、fallback、部分失败、重复回执、generation 失效、取消、restore/recurrent/stop。
完成条件：所声明 Enforce 路径都真实受限且可返回；原样 Off 路径及模型语义回归通过。

## M2：全链路有界输出与慢消费者隔离

- [ ] 保留单一请求语义 owner 的 sampling、stop、UTF-8/工具投影、token commit；独立输出任务承担等待消费者的传输。
- [ ] 可能产出 token 的 wave 执行前非阻塞预留请求级 output credit；最终 prefill 需要首 token credit。
- [ ] 额度同时覆盖队列字节、待投影缓冲和 terminal 元数据；不是只限制事件数，也不是改成无界通道。
- [ ] credit 不足仅将该请求置为 OutputBlocked；其他请求继续，资源占用与 SLO 时间不暂停。
- [ ] credit 恢复、断连/慢消费超时、取消有独立唤醒；保留有界 terminal 容量，满队列也能有序收尾。
- [ ] output reservation 与物理 permit 类型分离；提交失败释放未用 credit，已提交工作等待真实 completion。
- [ ] 昂贵 detokenization 离开全局 map 写锁；请求局部增量解码仅在等价测试通过后替换完整历史路径。
- [ ] HTTP SSE、非流式聚合、CLI stdout 均不得让一个慢消费者阻塞健康请求的调度。
- [ ] streaming headers 后拒绝/失败采用一致流内终态协议；不能伪造后发 HTTP 429。

落点：`engine/continuous_engine/{sequence,inner/completion,inner}`、scheduler readiness、server 流式/非流式输出与 CLI run。
验证：中文/emoji 跨 token、多 token stop、reasoning、工具/结构化输出、单 token、终态 flush、长输出、不消费、断连、取消竞态。
完成条件：有界、有序、无丢失/重复终态，健康请求不等别人的发送；同时报告健康子集和全体失败/延迟。

## M3：轻量时间状态、真实成本预测与 Observe

- [ ] 请求保留 ingress/first/last commit、committed_tokens、resolved budgets 和不可抹掉的失约状态；不复制第二套生命周期。
- [ ] deadline 为 `ingress+F` 与 `min(last+I, first+n×P)`；包含未入选和资源/output blocked 请求的义务。
- [ ] 给已承诺 prefill 建参考工作量、最晚启动窗口和里程碑；量化误差最多一个合法粒度，最终首字无误差豁免。
- [ ] 实际不可返回 wave 为预测单元；split/mixed、fallback、restore、backing growth、首次 graph 准备分别建模。
- [ ] 形状指纹覆盖模型/权重/KV/数值 profile、设备/路径/provider、decode 数和 KV 长度、prefill 长度/offset/片段、状态与采样。
- [ ] 模型返回典型/规划成本、样本/覆盖、可信状态、版本、边界；未知返回 Unknown 或已实测保守形状。
- [ ] 离线代表形状查表/分段回归与分桶残差余量；记录低估频率/幅度、漂移与实际失约，不把经验上界称硬保证。
- [ ] 观测包含 prepare/device/commit、计划/实际形状、真实进度和 outcome；NotSubmitted 不是成功 0ms，fallback 不混快路径桶。
- [ ] 快路径只读模型版本，慢路径校准发布；低估收紧余量/准入，降低余量需新鲜证据且不能擦除历史失约。
- [ ] Observe 只评价实际执行的旧策略 wave，不执行候选、不变 dispatch；规划/记录开销独立测量且有预算。

落点：`scheduler/implementations/continuous/{slo,cost_model}`、`engine/continuous_engine/{slo_runtime,sequence,profile}`、实际 executor 回执。
验证：虚拟时钟公式/索引/溢出、bucket 隔离、unknown/漂移、真实 shape 标签、Off/Observe 行为等价与实机开销。
完成条件：成本覆盖和误差可量化，轻量热路径不依赖重型诊断；尚不能据 Observe 宣称 Enforce 或容量提高。

## M4：有限候选、有界前瞻及完整 Enforce 快路径

- [ ] 只读 snapshot 包含 identity/generation、真实 logical progress/readiness、output credit、capacity 读视图和模型版本。
- [ ] 候选生成不修改队列/frontier、分配 KV 或启动设备；资源读视图永远不是 authority。
- [ ] 枚举合法 Decode/Prefill/Mixed/必要有界 Maintenance；按 deadline slack、prefill 欠账和稳定公平轮转选择有限集合。
- [ ] 依次检查语义身份、资源可能性、输出 credit、成本可信、本轮 deadline、所有未入选请求的未来义务。
- [ ] 实现有界 beam rollout：推进模拟时间、prefill 范围、最终首字/单步 decode、first/last/count、KV/context 和 credit。
- [ ] 自然 EOS 未知时在用户上限内保守存活；禁止读取 ShareGPT 参考答案长度预测实际结束或改变 max_tokens。
- [ ] horizon 终点检查共同可行的后续救援序列、首字期限和里程碑；逐请求各自假定独占“下一轮”不算共同见证。
- [ ] 三值结果区分 FeasibleWithinHorizon、ProvenImpossibleUnderModel、Unknown；搜索耗尽不是不可能，Unknown 不授予更多承诺。
- [ ] 只在可行序列间评分：真实预测输出、净首字参考进度与末端欠账/规划耗时；虚拟 prefill 信用不计外部 TPS。
- [ ] candidate 数、beam 宽度、深度、wall-time、重试次数均有硬上限；不复制 prompt/KV，公平 tie-break 不依赖 HashMap 顺序。
- [ ] 提交时重新读实际 now，预留 output credit 和权威资源，重验 identity/generation/取消/模型版本；只 commit 第一 wave。
- [ ] 消耗真实回执更新进度/成本后重新规划；过期候选只回收未提交 reservation，禁止预先推进未来动作。
- [ ] 无可信 SLO 候选时关闭新时间承诺，尝试已验证最小救援；默认进入有界、公平的 best-effort 进展路径，不因时间不可行永久等待或失败，不退回任意大 batch。
- [ ] 已失约永久记账，默认继续完成请求并保护其他请求；严格终止仅依显式服务合同或独立请求超时/取消，不删晚请求、重置时钟或借拒绝掩盖容量下降。

落点：`scheduler/implementations/continuous/{slo,candidate,cost_model}`、interfaces scheduler、engine slo_runtime/batch 提交闭环。
验证：纯候选无副作用、共同不可行例、未入选 decoder、prefill 饥饿、规划超预算、stale/cancel/输出变满、真实 wave 上限。
完成条件：静态保守容量范围内首条受支持路径的完整 Enforce 闭环有实机证据；单 deadline guard 或 greedy 单步不替代有界前瞻。

## M5：动态时间准入、等待、缓存与维护

- [ ] 有界等待区从 ingress 计 TTFT；同时限制数量、prompt 字节/token 和最长等待，不能排队后重置预算。
- [ ] 模型/上下文等硬约束不合法可拒绝；可靠时间下界表明无法达标时默认记录风险并继续 best-effort，只有显式严格服务合同才允许时间拒绝；信息不足保持 Unknown，不伪称数学不可行。
- [ ] 在只读状态加入候选，用 M4 构造包含已有承诺的有限服务见证；检查 prefill→decode、上下文增长、credit 和维护。
- [ ] 检查 decode 延伸负担满足 TPOT 与 ITL 且为未来首字留空间；用已实测并发/长度范围约束 horizon 外风险。
- [ ] 同批新增请求依服务类份额和年龄逐个更新模拟负担；不只接短请求，最终仍通过原物理 authority。
- [ ] 区分保守用户输出上限模式与统计长度分布模式，记录风险/版本；统计预测不授权超卖、不修改用户上限。
- [ ] Admit 携版本/horizon，Defer 携 retry_at/expiry/reason，Reject/Unknown 携明确原因；独立 TimeReached 定时唤醒。
- [ ] 显式区分完成优先默认与严格时间承诺策略：Defer/expiry 不把 SLO deadline 偷换成默认请求终止时间；成本 Unknown、已失约和规划无解仍有安全进展，不能隐式取消或永久饥饿。
- [ ] CapacityChanged/RequestStateReady/OutputCreditAvailable/TimeReached 可组合；取消始终唤醒，不伪造 capacity epoch。
- [ ] prefix rendezvous 比较 producer+restore+suffix+首字与直接 prefill；hold 不超过剩余 TTFT，producer 失败/代际改变解除等待。
- [ ] backing growth、restore/recompute、graph 冷启动、串行 fallback 纳入 wave 或有界维护；未知维护收紧承诺而不省正确性步骤。
- [ ] 漂移/过载先收紧新 admission，保持在途资源收尾；按每类 offered/promised/completed/rejected/failed/pending 公开结果。

落点：`scheduler/implementations/continuous/{time_admission,prefix_rendezvous,pressure}`、engine readiness/timer/prefix_restore、server admission 协议。
验证：两请求分别可行但合并无解、仅时间能唤醒、长上下文增长、突发/到达率变化、prefix 取消、维护 fallback、等待区真实边界。
完成条件：动态准入已接真实运行路径，负载变化不产生无界积压；只保留固定并发上限不算 M5 完成。

## 完整验证、性能验收与完成判定

- [ ] Rust 虚拟时钟/状态机与接口测试先行；随后执行真实后端的数值/语义回归，覆盖 position/KV/recurrent/stop/usage/工具及请求 RNG。
- [ ] 稳定代码里程碑执行 fmt、workspace all-targets check/test/clippy（`-A warnings`）和 macOS Metal compile；必须另在已授权 RTX 5090 核验并完成锁定 operator-set 的 CUDA 编译与受影响数值/语义、run/serve 运行回归，当前待 probe，缺项属于未完成。
- [ ] 编译、协议有效、语义正确、实测收益分栏；ignored/容忍失败/未跑后端不算通过，历史快照不替代当前代码检查。
- [ ] 同硬件、同模型/精度/数值策略、同 pinned ShareGPT、同选样顺序与输出/EOS 规则，llama.cpp 为初始对照；synthetic 仅诊断。
- [ ] Metal 与 CUDA 分别完成 Qwen3.5-9B 同机六项延迟、llama.cpp/Ferrum Off 双吞吐基线、完成率与峰值内存验收；各自冻结并发扫描和重复合同，跨后端分表，不能用一方通过覆盖另一方未测。
- [ ] 服务 slots/context/batch/预算固定，独立改变客户端 concurrency/开放到达率；记录实现差异、binary/config/data 指纹与 host/thermal 状态。
- [ ] 先 A/A，后成对交错独立重复；容量粗扫再在边界邻域细扫，不假定通过率完全单调；保留失败与不利轮次。
- [ ] 报告 TTFT/TPOT/pooled visible ITL P50/P99、request-max ITL/联合通过、raw/success/SLO output TPS、错误/拒绝/样本/重复及送达/队列趋势。
- [ ] 固定表保留 sampled peak Metal/GPU allocation、OS footprint、maximum RSS 三列及采样窗口/间隔；不可相加，缺失明确未采集。
- [ ] 持续服务区间足够观察队列/最老年龄，排空或记录右删失；给原始请求证据与不确定性，不将 N32 或三次均值当稳定 P99 保证。
- [ ] 分别消融 M1 工作边界、M2 输出隔离、M4 deadline/里程碑/beam/评分、M5 时间准入；复杂度调整需同目标实证和明确设计变更。
- [ ] Off/Observe/Enforce、未知形状/漂移/fallback、缓存冷热、慢消费者及取消均有回退与有效配置说明；外部证据不提交 Git。

最终完成要求：K0、R0、M0–M5 功能和契约均闭合，Metal 与 CUDA 各自指定支持范围内所有分项/联合 SLO、双吞吐基线、完成率、峰值内存、正确性与稳态条件通过，并证明相对各自同机基线的持续达标容量提高；新版本及中英文 README/官网交付全部完成。
若只证明隔离/过载保护、局部吞吐或部分后端能力，准确报告该结果，完整性能目标继续保持未完成。
前瞻参数与内部里程碑可依据消融调整，但不得把未实现能力改名为“简化完成”，也不得将没找到收益当作验收通过。
