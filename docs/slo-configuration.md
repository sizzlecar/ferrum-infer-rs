# SLO 配置与测量口径

运行时策略与 HTTP benchmark 使用两个独立合同。`ferrum run/serve --slo-config`
读取 `SloConfig`，内部时间从可信服务入口计到模型 token commit；
`ferrum bench-serve --slo-client-config` 读取 `SloClientVisibleConfig`，
测量客户端请求开始到可见输出。外部目标不能由内部预算推导。

`off` 保持原执行策略；`observe` 提供入口、token commit 和终态计时，
保持既有调度策略，具有完整执行证据的波次用于成本训练。
`enforce` 已接入有界规划、首波提交前重验和默认完成请求路径，但只允许具备
真实 guarded eager-wave 能力的 PlanRuntime 执行器、`complete-requests` 与
`credited` 输出组合。启动必须实际加载匹配身份的 `prefill_reference` 和非空、
未过期的 `cost_profile`；配置路径或此前保存的 receipt 不能替代本次加载。
推测执行、缺少该执行能力和严格 `require-slo` 目前仍不支持 Enforce。
这不表示任意模型、后端或请求已获得成本覆盖，也不表示 M3–M5 和性能验收完成。
完整完成条件见 [实施清单](slo-implementation-plan.zh.md)。

Observe 在每次执行迭代和空闲/资源等待期间记录尚未完成请求的内部超时。
等待时使用独立定时器，TTFT 从原入口计时，后续 token 同时检查 ITL 与累计 TPOT；
等于期限仍合格，超过后保留失约记录。已记录的同类失约不反复唤醒，达到输出上限
的请求不再承担下一 token 的义务。定时器只更新观测，不触发新的选批、资源探测或
时间准入；它不表示完整 M5 过载策略已经交付。HTTP 和 CLI 共用此引擎行为。

## run / serve 运行时策略

策略文件可为 `.toml` 或 `.json`。CLI `--slo-config` 优先于主配置文件中的
`runtime.slo_config`，无环境变量覆盖；未配置时默认为 Off。
配置内容、绝对路径和 SHA-256 进入运行时配置快照。
相对 `cost_profile`、`prefill_reference.artifact_path` 和成本导出路径
均以策略文件所在目录为基准。

下面所有延迟、比例、并发和请求数量均为**语法示例**，不是实测合格阈值、
推荐容量或服务承诺；实际验收必须事先指定工作负载对应的目标。

```toml
# /tmp/ferrum-slo-observe.toml
mode = "observe"
default_service_class = "interactive"

[[services]]
id = "interactive"

[services.server_token_commit]
ttft_ms = 500
tpot_ms = 40
itl_ms = 80

[services.attainment]
ttft_percentile = 99.0
tpot_percentile = 99.0
itl_percentile = 99.0
itl_percentile_scope = "pooled-gaps"
min_accepted_joint_attainment = 0.99
max_reject_rate = 0.001
max_error_rate = 0.001
```

```console
ferrum run /models/model --prompt "Hello" --slo-config /tmp/ferrum-slo-observe.toml
ferrum serve /models/model --served-model-name benchmark-model --slo-config /tmp/ferrum-slo-observe.toml
```

Observe/Enforce 要求显式 `default_service_class`，且该 ID 必须存在于 `services`。
未识别字段、重复服务 ID、零延迟预算、溢出预算、非有限数值和越界比例会被拒绝。
延迟预算为正整数毫秒，百分位为 `(0, 100]`；联合达标目标为 `(0, 1]`，
拒绝率和错误率上限为 `[0, 1]`。

| 配置分组 | 字段与默认值 | 当前含义 |
| --- | --- | --- |
| `planner` | `candidate_limit=16`, `beam_width=4`, `lookahead_waves=3`, `max_planning_us=2000`, `max_replan_attempts=2`, `retry_backoff_ms=1` | 有界规划参数合同；重试定时器仅处理瞬时竞争或计算预算耗尽，不授予容量或重置请求时间 |
| `planner` | `search_budget_percent=60`, `publication_reserve_percent=20` | 同一同步事务的阶段份额：已有完整共同见证时，可选探索最迟在原始预算 60% 处停止，并按已测完整路径工作量提前预留复验时间；共同序列最终回放必须在 80% 前结束，余下 20% 留给真实资源、路线、输出重验和发布。capture 消耗的时间已计入，不重置预算；份额不是性能保证 |
| `planner` | `prefill_credit_beta=1.0`, `prefill_debt_gamma=1.0`, `enable_prefill_milestones=true` | 预填充收益与债务的显式参数 |
| `cost_observation` | `max_queued_samples=256`, `max_queued_shape_rows=8192`, `max_samples_per_update=256` | 待训练队列与单次后台更新的硬上限，shape rows 按已分配容量计数 |
| `cost_observation` | `max_waves_per_call=4`, `max_rows_per_wave=1024`, `max_retained_rows_per_call=4096` | 单次执行观察器的波次和行存储上限；不授权增加模型工作 |
| `cost_observation` | `structured_capture="disabled"` | `host_settled_v1` 显式采集候选/实际执行结构及真实终态回执旁证；不切换预测器，不建立新 profile 的训练资格，详见[结构化成本模型](slo-structured-cost-model.md) |
| `admission` | `max_active_requests=8`, `max_waiting_requests=128`, `max_waiting_prompt_tokens=1048576`, `max_waiting_prompt_bytes=16777216` | 时间策略的请求/等待存储上限，不替代物理资源许可 |
| `admission` | `time_policy="complete-requests"` | 默认优先完成请求；`require-slo` 显式选择严格时间准入，不能追溯拒绝已接受请求 |
| `admission` | `max_wait_ms=30000`, `max_sequence_tokens=32768`, `output_length_policy="conservative-upper-bound"` | 输出上界及等待合同，不授权缩短用户请求 |
| `output` | `transport="legacy"` | 输出路径选择；显式 `credited` 启用已经证明容量与编码上界的传输路径 |
| `output` | `max_queued_events_per_request=256`, `max_queued_bytes_per_request=1048576`, `max_projection_bytes_per_request=1048576` | 每请求输出空间预算 |
| `output` | `terminal_reserve_bytes_per_request=4096`, `max_total_buffer_bytes=67108864`, `slow_consumer_timeout_ms=30000` | 终态保留、总输出空间与慢消费者超时预算 |

两个规划阶段百分比都必须大于 0，且总和小于 100。阶段截止按纳秒向下取整；
调用方给出的时间窗过短、导致阶段为空时返回 `Unknown`，不会扩大配置预算。
软停止通过审计字段 `search_soft_stops` 记录搜索截断，不表示最优结果；最终回放、
身份、资源、输出或 TTL 重验失败仍然返回 `Unknown`，默认完成优先路径继续负责进展。
审计 `planner_budget_exhausted` 表示搜索/回放额度停止（包含 80% 阶段截止），
`budget_exhausted` 只在同一事务的真实 100% 硬截止到达时置位，两项分别导出指标。
因此 1.6ms 阶段停止不再被报告成默认 2ms 总预算已经耗尽。

形成完整共同计划后，规划器用该路径 `begin` 与成功 `advance` 的实际墙钟耗时，
加上原配置中搜索截止与回放截止之间的余量，预留最终复验时间。同一事务只会
收紧可选搜索截止；没有完整计划时不会提前终止构造。这个估计不保证复验一定
来得及，最终仍须重新回放并通过真实时钟与提交前检查。审计中的
`measured_replay_work_ns`、`replay_reserve_ns` 和 `replay_reserve_stops` 分别记录
完整路径耗时高水位、预留量和因此提前停止探索的次数。`phase=Finalization`
包含统一排序、独立回放及末端检查，不能据此把全部耗时归因于执行形状投影。

`output_length_policy` 还接受 `statistical-capacity`，但启用统计容量需要独立校准证据；
单纯选择枚举不能建立覆盖率。输出总预算必须容纳 `max_active_requests` 个请求的
queued 与 projection 份额，终态保留必须小于每请求 queued 字节预算。
这些有界默认值没有通过同硬件容量验收。

### 每波 prefill 工作量

`ferrum.toml` 的 `runtime.scheduler_active_decode_prefill_chunk` 限制有可运行
decode 请求时每个 prefill 请求的块大小；
`runtime.scheduler_active_decode_prefill_token_budget` 限制同一物理波内所有
prefill 请求的 token 总量，`0` 表示禁用该总量限制。两者同时受模型分块粒度、
剩余 prompt 和实际资源容量约束，不能用不合法的短块满足配置。

SLO 规划、Unknown 下的请求完成路径和最终提交重验使用同一组工作量限制。
`runtime.prefill_decode_execution = "split"` 时，每次物理提交只含一个阶段；
prefill-only 波仍按当前可运行的 decode 请求计算限制。规划中的后继状态也重新
计算这些条件，包括最后一块 prefill 产生首 token 后转入 decode 的情况。
共享引擎中的 `run` 与 `serve` 使用相同规则；手动校准继续执行 manifest 声明的
精确 cohort，不将服务策略悄悄施加到校准样本。

token 上限不是毫秒上限，也不证明端到端 SLO 达标。尤其当总量限制为 `0` 时，
多个合法 prefill 块的总执行时间仍可能很长；应结合完整波成本预测和真实并发测量。

### 请求完成与时间承诺

默认 `admission.time_policy = "complete-requests"`。预测延迟超标、已经失约或成本
Unknown，都不能单独成为拒绝、取消或缩短输出的理由；原始 TTFT 和失约记录保留，
请求应走安全、有界且公平的 best-effort 执行路径。暂时没有活跃执行容量时等待，
模型/上下文硬限制与实际等待存储耗尽仍由相应资源和协议边界处理。

`Enforce + complete-requests` 在接受新请求前，按同一等待队列锁校验
`admission.max_waiting_requests`、`max_waiting_prompt_tokens` 和
`max_waiting_prompt_bytes`。请求数还受原 `scheduler.max_waiting_requests` 限制。
token 数来自引擎实际分词后覆盖的内部计数，缺失或非法值不能按零计；字节数是
实际 rendered prompt 的 UTF-8 正文长度之和，不包含 API 消息/metadata、重复持有的
字符串、分配器余量或整个进程 RSS，也不是 KV/执行资源许可。`run` 与 `serve` 的
共同引擎入口使用同一规则。超限只对尚未接受的新请求返回 `ResourceExhausted`
（当前 HTTP 映射为 503）；不会因预测成本 Unknown、延迟超标或原始等待时间过长拒绝。
请求离开等待队列进入 prefill、取消或错误移除后，其等待份额自然释放。
已接受请求因真实容量变化返队时继续保留，即使此时超过新请求上限，也只阻止新增请求，
不撤销已有工作、重置 ingress 或缩短输出。Off/Observe 保留原等待策略。

`max_wait_ms` 从原始 ingress 计算，默认仅是等待复审时间，不能把它当作隐含的
请求终止时间。仅显式 `require-slo` 且请求尚未被接受时，它才是时间准入到期条件；
客户端取消和独立配置的请求超时不受此策略替代。

Enforce 在真实 Waiting→Prefill 入场前执行时间活跃上限，默认
`admission.max_active_requests=8`。它与物理槽位、beam/search 请求数上限不同；
物理槽位设为 32 不会自动把时间上限改为 32。新的 fresh 请求超出时间上限时保持
有界 Waiting，原 ingress/SLO 时钟继续运行；已活跃请求及容量回退后的重算继续服务。
实际完成、取消、相关容量变化及声明的 `review_at` 会触发复审；Defer 独立记账，
不被记成模型 Unknown。单纯成本缺证不增加时间等待，未到时间活跃上限时仍可进行
受物理许可约束的完成优先入场。Off/Observe 的原入场行为不变。

CompleteRequests 的原等待复审时间到期后，最多允许一个额外逻辑活跃 owner
竞争真实入场，因此正常上限为配置值、超时完成例外最多为配置值加一。
该例外只由实际 Admitted 占用，维护、NotSubmitted 或容量回退不会释放它；
直到该 owner 离开，或实际完成腾位使它归入正常上限，才能产生下一个例外。
未能入场者仍遵循原队列的真实 capacity/source epoch 等待；到期不构造许可、
不缩短输出，也不把快循环每轮的重试算成新的名额。此默认 8 的实际执行约束可能
改变 Enforce 的并发行为，吞吐影响须用固定声明的配置测量；并发扫描不得逐 C
改变上限后仍声称配置相同。

纯时间准入评估保留所有请求义务，新请求的有限见证必须覆盖 prefill、首 token
及下一步 decode；用户只要求一个输出 token 时完成即可。该见证仅覆盖声明的 horizon，
不代表整请求生命周期保证，也不提供 GPU/输出资源许可。成本缺证、搜索预算耗尽、
未覆盖长度及其他请求的失约，不能被包装成目标请求数学上不可行。

默认完成策略在实际接受后记录原入口、输入长度及输出上界；每轮至多对一个尚无
首 token 的请求评估共同有限见证，使用同一规划时间预算。只有实际 guarded
提交才记录见证已开始；撤回、Unknown 或 Defer 不产生承诺，也不取消已接受请求。
CPU 控制后端测试覆盖此生命周期，真实模型的覆盖率、调度开销与收益仍须实测。
严格接受前时间准入及其完整过载策略尚未完成，不能据此称 M5 已交付。
HTTP 和 CLI 使用同一份配置和 credited 引擎入口；旧的直接 `infer` / `infer_stream`
接口不携带输出额度合同，在 Enforce 下会在接受请求前返回 Unsupported。

### 成本观测与后台训练

Observe 在真实准备、设备执行及逐请求结果提交边界收集成本证据。只有身份、实际
形状、提交结果和时钟完整的独立波次才进入训练；取消、失败、复合区间及未知路径
不生成成功成本样本。输出投影策略和历史长度也是成本形状的一部分。

每个启用观测的引擎使用一个 CPU 工作线程。推理线程只尝试投递有界样本并发送
合并唤醒信号，队列满或锁竞争时丢弃并计数，不等待训练。工作线程按
`max_samples_per_update` 分批更新，发布不可变模型快照；引擎关闭时先停止推理，
再排空已接收样本并等待工作线程结束。Off 不启动该线程。
入队成功自身触发唤醒，即使外层批次随后取消也能训练已接收样本；关闭等待被取消后，
再次关闭仍等待同一工作线程的完成结果，不会因句柄已被取走而提前报告结束。

训练保留真实样本时间，按已消费样本的时间发布；后台排队不会刷新样本年龄。
预测时仍检查当前时间与有效期。`ferrum.engine.cost_training_update_seconds` 记录
后台更新耗时；`cost_observation_lost` 和 `cost_observation_queue_lossless`（同一
`ferrum.engine.` 前缀）仅描述队列损失，不能证明完整采集或预测覆盖。
固定大小的审计分别记录初始化/准备/调用拒绝、实际完整观察的入队或丢失、训练的
Recorded/Skipped/Error、模型发布及导出结果，并按波次类型分项。计数饱和显式标记；
运行中读取的多个计数不是同一个原子快照，未埋点的物理波次仍属未知。
后台运行仍消耗 CPU，其实机开销待测。

在线模型参数通过 `cost_observation.model` 配置，默认每桶至少 8 个样本、最多
128 个样本，最多 1024 个桶和 16384 个保留样本、262144 个形状行；有效期
300 秒，经验残差分位为 0.99、额外余量为 100 微秒，KV 长度及 prefill offset
默认精确匹配。这些是起始参数，不是延迟保证。真实 ShareGPT 覆盖率和调度器对
快照的使用仍待验收；缺少覆盖时保持 Unknown。

接收新有效样本时，训练器先检查回收后的容量，再移除过期样本和空桶；校验失败
或回收后仍超限时不改变训练状态。回收不会刷新旧快照的样本年龄。被移除桶的
历史规划值按波次类型及计时边界归入固定数量的只增下限，防止重建同类桶时
静默降低余量；prefill/decode、设备时间/准备到提交时间分别保存。
这些下限仍须配合新的最少样本数和形状覆盖，不能单独生成 Known 预测。

模型默认使用 `feature_model.kind = "exact_v1"`，保留原来的精确执行身份和查询行为。
显式选择 `bounded_numeric_v1` 后，实际 provider、执行路径、物理行顺序、采样/输出
分支仍精确匹配，生成长度、采样历史、重复惩罚历史及完整前缀解码容量作为数值工作量。
输出上限保留为容量/终态元数据，是否到达上限的分支仍独立；此模式不会改变用户上限。
例如以下仅为配置语法，分段宽度必须在实际校准及对比前声明：

```toml
[cost_observation.model]
context_bucket_tokens = 128
prefill_offset_bucket_tokens = 128

[cost_observation.model.feature_model]
kind = "bounded_numeric_v1"
host_history_bucket_tokens = 64
```

数值查询须落在同组新鲜观测范围内，并由**一条真实联合观测**同时覆盖所有工作轴；
不能分别取不同请求、不同样本的坐标最大值拼出未测过的组合。规划值采用组内完整
波次耗时的最大包络，再加余量及历史下限；典型值、残差分位和包络分别保留。
这是明确的经验模型假设，不是 GPU 延迟单调性的证明或统计置信上界。
缺少数值证据、物理行顺序未证明、样本过期或联合支持不足时保持 Unknown。
新数值行同时计入队列、训练和导入导出的保留上限；不是免费增加的存储。
当前数值证据只来自已有有界输出 owner；真实覆盖、误差及调度收益仍需实机校准。

`empirical_host_content_v1` 使用完整的准备到 host-settled 实测区间，包括符合合同的
请求终态处理。它仅支持声明的普通文本贪心输出域，将具体未来文本引起的主机分支
差异作为经验扰动；实际全波输出产品、读回方式、provider 和终态分支仍区分。
规划值采用典型耗时加经验残差余量及历史下限，不是确定性的最坏时间上界。

`empirical_row_multiset_v2` 在该输出域下进一步合并完整行的统计置换，配置示例为：

```toml
[cost_observation.model.feature_model]
kind = "empirical_row_multiset_v2"
host_history_bucket_tokens = 64
```

新模式需要真实采集的逐行类别和数值证据。它只在连续相同角色的行段内，将
`(行类别, 当前工作, 完整数值特征)` 一起排序用于统计查询；实际物理行、训练原件和
提交许可保持原序。不能分别排序 KV、历史长度或终态标记，也不能把 prefill/decode
跨段交换。新增行的已分配容量计入观察器、队列、训练及导入导出限额。
此模式只解决行排列造成的样本分散：prefill 总长度和当前块大小、波次宽度及实际
provider 路径等仍有精确匹配限制；固定参考曲线覆盖输入范围不等于成本预测覆盖。
`empirical_prompt_range_v3` 进一步允许同一路径、同块大小的 prefill 样本按实际
输入总长度共享统计：

```toml
[cost_observation.model.feature_model]
kind = "empirical_prompt_range_v3"
host_history_bucket_tokens = 64
```

完整输入仍影响主机准备成本，因此总长度作为每行的联合工作坐标保留：查询须处于
新鲜样本的实测范围内，并由一条完整观测同时覆盖总长度和其他工作轴。实际分块
大小、末块标记、provider、输出及主机分支仍精确匹配；原请求、执行身份和提交
检查不变。该模式不证明耗时随输入长度单调，也不能补齐未采集的块大小、波次宽度
或冷启动路径。默认仍为 `exact_v1`，旧模式不会自动改用新统计规则。

这些经验模式都保留最少样本数、原始样本年龄和联合支持要求，缺证时保持 Unknown。
其真实并发误差、控制开销和 SLO 收益仍在验收中，不能依据模式已启用宣称达标。

### 导入成本 profile

在策略文件顶层设置 `cost_profile = "profiles/model.json"`，Observe 启动时会读取
对应文件；路径相对于策略文件目录。`cost_observation.model` 必须与 profile 的
训练设置一致，当前执行器也必须提供与文件一致的模型权重、数值策略、设备运行时
和执行配置指纹。缺少身份、文件损坏或设置不一致会让初始化失败。

导入同时支持严格的 v1 与 v2 文件。v1 保留原结构和精确模型，不能从旧摘要反推
数值特征或用于数值模式；v2 显式携带模型选择、旧精确形状和版本化数值证据。
模式、特征版本或其他设置不匹配会报错，不会静默切换到另一模型。
v3 文件用于 `empirical_host_content_v1`；v4 文件用于 `empirical_row_multiset_v2`；
v5 文件显式用于 `empirical_prompt_range_v3`。v5 沿用 v4 的实际观测字段，并
保留同一真实 entry 的来源序号、完整物理行特征与原始测量时间。旧文件不能通过
改版本号升级或从不可逆摘要推造逐行证据。新模式导出 raw v6、cut v4；旧模式
继续使用原 wire 结构，即使运行时观察器已经具备新增字段。

导入还要求显式填写
`cost_observation.profile_import.declared_local_clock_max_error_ns`，其含义是本机
墙上时钟误差的已知上界；默认未声明，不能仅凭读取系统时间推定误差为零。
导入器同时计入来源时钟误差，以保守样本年龄映射到本次进程的单调时钟。重新启动、
加载或后台消费都不会让旧证据重新变新；过期和未覆盖的查询仍返回 Unknown。

`cost_observation.profile_import` 默认限制文件 16 MiB、16384 个样本、262144 个
形状行、来源字段 4096 字节、profile 年龄一天及总时钟误差一秒。模型自己的
300 秒样本有效期仍独立生效。放宽文件年龄不会放宽样本有效期。

引擎返回独立的 `slo_cost_profile_receipt`，记录实际文件绝对路径、SHA-256、字节数、
样本年龄/计数、模型版本和来源。CLI 将这份实际 receipt 写入运行配置；用户策略及其
原始 hash 保持原值。Off 不加载 profile，也不保留调用方旧的导入 receipt。

### 导出成本 profile

Observe 可通过 `cost_observation.profile_export` 在引擎关闭、后台训练排空后导出
profile JSON 和配套观察 JSONL。两个路径相对于策略文件目录解析，父目录须已存在，
目标文件须不存在；写入不会覆盖旧证据。例如在前面的 Observe 策略中添加：

```toml
[cost_observation.profile_export]
path = "profiles/calibration.json"
observations_path = "profiles/calibration-observations.jsonl"
max_file_bytes = 16777216
max_samples = 16384
max_total_shape_rows = 262144
# 仅示例：须替换为已知的本机墙上时钟误差上界，不能假定为零。
declared_clock_max_error_ns = 10000000
```

推理线程不写这些文件；后台工作线程保留受限样本、计算实际观察文件的 SHA-256，
并先发布观察文件，再发布引用它的 profile。后一个文件发布失败时，已有观察文件
保留供诊断，关闭返回错误；两个文件不是一次跨文件原子事务。
来源记录包括实际运行程序的内容 hash、时钟锚点和采集计数，不凭工作树 HEAD
声称二进制来自某个提交。导入过的旧样本不会被导出成刚测得的新证据。

原始观察 JSONL 使用显式 v3 格式，有界保留已消费的 Completed 观察及真实训练结果，
包括训练器因容量等原因拒绝的完整形状；可导入 profile 仍只包含 Recorded 子集，
新导出使用 v2 格式保存数值证据。摘要分别报告源观察、队列、训练和导出的分母，以及样本/形状行/字节
限额造成的丢弃。`retained_samples` 保留可入 profile 样本数的原意，
`raw_retained_observations` 单列原始观察数。没有 Recorded 样本时仍可留下原始诊断
文件，关闭明确报告无法生成 profile。因此原始文件不能单独证明全部请求或全部
候选的成本覆盖率，也不是并发性能报告。
有效的成本校准还需使用已支持的 `credited` 输出路径；legacy 流式输出同步工作
使产 token 区间被标为复合成本，不能将它当作独立 decode 样本。

### 手动驱动真实校准

`ferrum calibrate-slo` 使用与产品入口共用的模型来源、数值配置和原生引擎构造，
由独占 `CalibrationSession` 驱动真实请求。每个 cohort 明确选择请求集合、
prefill chunk 和 split/mixed 工作；最终提交仍检查真实 owner、前沿、资源和输出额度。
它不启动常规调度后台循环，也不以模拟耗时替代实际执行。

manifest 的 `validation_model` 选择验证对象：

- `{"kind":"live_frozen"}`：训练阶段结束后，`freeze_cost_model()` 等待已接收的
  观察全部处理，保存一次不可变在线模型和队列 ordinal。这是机制诊断，不发布可部署文件。
- `{"kind":"exported_profile","profile":"cut-profile.json","source":"cut-source.jsonl"}`：
  在 accepted cut 导出真实 v2 artifact，再通过产品的严格 loader 导入**同一文件**，
  验证只查询这个固定 imported predictor。相对路径以 manifest 所在目录为基准。
  报告保存实际截止 ordinal、两个文件的路径/hash/字节数和加载 receipt。

旧 v1 manifest 省略此字段时保持 `live_frozen`，报告会显式记录；不会自动升级成
profile 验证。导入模型不要求等于原在线模型，二者的发布历史和保守下限可能不同。
验证均使用新请求 owner 和新的实际观察；后续在线训练不会改变已选的验证模型。
当前查询发生在执行结束后，输入是实际 shape，因此仍是**独立样本的事后成本检查**，
不能证明执行前 route 预测成功、候选覆盖率或 SLO 达标。Unknown 和覆盖不足原样报告。
可选 `reference` 字段现在接入固定 prefill reference 发布，见下面的阶段与限制。

命令要求策略显式选择 `observe`、`complete-requests` 和 `credited`，不修改
请求的完整 `sampling.max_tokens`。EOS、stop 和输出错误保留真实结果；不会为了补齐
成本样本而延长或缩短响应。`--startup-usage run|serve` 必须给出，用于选择目标产品的
启动默认值；所有 cohort 共用一次实际容量配置，不能随 cohort 宽度扩容。
需要与部署固定资源预算匹配时，显式传入 `--runtime-memory-budget-bytes`；该值进入
共享产品启动解析及报告中的有效配置，不要求设置环境变量。

每个 cohort 还可显式设置 `"rolling_window":{"maximum_in_flight":8}`。
此时 `prompts` 是有限、有序的请求索引列表；最多保留指定数量的在途请求，
一个请求的终态输出与成功 completion 都消费完后，才补入下一个索引。
例如，64 个原请求可以在固定窗口 8 下持续补位，无需等最慢的同批请求结束。
省略该字段或设为 `null` 时仍按原整组方式运行，上一组全部结束后才开始下一组。

窗口必须非零且不超过 `protocol.maximum_requests`。它不扩大引擎容量、输出额度
或资源许可；所有请求仍使用原 prompt、采样策略和完整输出预算。遇到消费失败或
取消时中止采集并执行原关闭流程，不以失败请求空出的槽继续收集。
未设置下述 `wave_plan` 时，每个 cohort 的 chunk、split/mixed 和 decode route 仍固定。
滚动入场不代表覆盖了所有执行形状。此选项只影响校准命令，普通 `run` / `serve` 的行为不变。

该字段进入完整 manifest 的协议摘要，因此滚动与整组采集具有不同身份。
fit→冻结→独立 residual→导出/重新导入→heldout 的阶段隔离不变。采集前须按完整
输出工作量声明有界 timeout；窗口不延长 profile TTL，也不改变残差分位数或样本门槛。
原始记录新增 `rolling_admission`、`rolling_completion` 和
`rolling_cohort_drained`，保留实际 request ID、输入序号和在途数量。

cohort 可另显式声明有界波次计划，例如：

```json
"wave_plan": {
  "prefill_chunks": [16, 32, 64, 128],
  "decode_routes": ["actual", "full_logits"]
}
```

两条周期各为 1–16 个预声明选项，至少提供一条；省略的轴沿用原静态字段。
每次 phase/case/repetition 从两个序号 0 开始。只有本次真实工作逐行成功提交并完成
host reconciliation，且 owner/frontier/work 与尝试一致时，才推进对应轴一次；mixed
波同时推进两轴。Prefill 末块按剩余输入裁短，但仍算一次成功 prefill 波。
Blocked、NotSubmitted、维护和未执行重试不推进；失败、不可判定提交或缺少成功行证据
终止采集并保留原关闭/清理流程，不靠再次执行猜测进度。它不依据预测 Known、入队成败、
耗时或质量结果改选项。额外成功终态清理不一定有合格成本样本，也不会阻止已完成工作推进。

计划及其顺序进入完整协议摘要。raw 的 `wave_plan_attempt` / `wave_plan_result`
记录全局尝试序号、两轴成功波序号、周期索引、所选 chunk/route、请求及实际提交结果；
并保留既有真实 wave/host 记录。Reference discovery/trials 和其 warmup 拒绝此计划，
须使用独立静态 reference manifest。该选项不会改变 split/mixed 策略、输出预算、
请求列表、阶段隔离、TTL、样本门槛或资源许可，也不保证每个实际族都获得足够样本。

每个显式 cohort 可选 `"token_policy_residency":"invalidate_before_cohort"`，
在每次 repetition 添加请求前，清除上一次真实 token-policy 上传的 residency 记录。
省略或设为 `"preserve"` 保持原行为，普通 `run` / `serve` 不调用此校准操作。
清除要求上一 cohort 全部终态，且没有尚待处理的执行、维护、发布或 restore；
executor 还会独立检查真实 registry、completion worker 和 lane。Unsupported 或
Unavailable 会写入原始 receipt 并使该次采集明确失败，不作为一次成功的清除。

raw 的 `token_policy_residency_invalidation` 事件保留 phase/case/repetition 和实际
`cleared_entries`，report 分别统计尝试、成功操作和清除条目。空缓存成功计数为零条目。
它只忘记 token-policy residency，**不清 GPU/驱动缓存，也不代表完整冷启动**；
下一次原 prepare 必须自己决定并执行上传，是否实际发生仍以 wave 的实际 route 为准。
这可为冷 residency 路线采集独立试次，不能替代 `min_samples`、TTL 或其他 shape 覆盖。
同一个 cohort 内后续 wave 仍会自然命中热缓存；请求的采样、完整输出预算与参考评分不变。

以下是可解析的最小语法例子，使用原始文本 `Hello`，**不是 ShareGPT 校准数据、
统计质量门槛或 9B 容量建议**。将运行参数合并到当前工作目录的 `ferrum.toml`：

```toml
[runtime]
paged_max_seqs = 2
max_batched_tokens = 2048
prefix_cache = false
reusable_execution = false
```

这里显式选择当前支持的无 prefix checkpoint 额外波次、无 reusable device-program
路线，workspace 桶仍由实际运行时使用。要复用校准结果，目标部署也必须使用匹配的
执行配置；命令不会偷偷关闭部署已有的选项。

保存下面策略为 `/tmp/calibration-observe.toml`：

```toml
mode = "observe"
default_service_class = "calibration"

[admission]
time_policy = "complete-requests"

[output]
transport = "credited"

[[services]]
id = "calibration"

[services.server_token_commit]
ttft_ms = 500
tpot_ms = 40
itl_ms = 80
```

保存下面完整 manifest 为 `/tmp/calibration-inputs.json`。摘要分别对应 UTF-8 字符串
`documentation:raw-utf8-no-template:Hello:v1` 和 `Hello`，均不带换行。
实际实验须替换为冻结的输入准备协议和真实提示词摘要，不能沿用示例身份。

<!-- calibrate-slo-manifest-v1 -->
```json
{
  "schema_version": 1,
  "validation_model": {"kind": "live_frozen"},
  "input_preprocessing_sha256": [134,32,164,16,145,167,29,99,166,166,126,187,52,170,36,103,32,149,187,240,4,237,133,43,218,179,195,73,148,94,68,199],
  "protocol": {
    "total_timeout_ms": 600000,
    "shutdown_timeout_ms": 30000,
    "maximum_wave_attempts": 10000,
    "maximum_raw_bytes": 16777216,
    "maximum_requests": 2,
    "output": {"kind": "cli_text"}
  },
  "prompts": [
    {
      "source_id": "documentation:hello",
      "rendered_prompt": "Hello",
      "rendered_prompt_sha256": [24,95,141,179,34,113,254,37,245,97,166,252,147,139,46,38,67,6,236,48,78,218,81,128,7,209,118,72,38,56,25,105],
      "sampling": {
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": null,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop_sequences": [],
        "seed": null,
        "min_p": null,
        "tfs": null,
        "typical_p": null,
        "mirostat": null,
        "response_format": {"type": "Text"},
        "structured_output_start": {"mode": "immediate"},
        "response_completion_boundary": {"mode": "immediate"},
        "model_output_protocol": "text"
      }
    }
  ],
  "training": [
    {"prompts": [0,0], "repetitions": 8, "prefill_chunk_tokens": 4, "execution": "mixed"}
  ],
  "validation": [
    {"prompts": [0,0], "repetitions": 2, "prefill_chunk_tokens": 4, "execution": "mixed"}
  ]
}
```
<!-- /calibrate-slo-manifest-v1 -->

```console
ferrum calibrate-slo /models/model --backend metal --startup-usage serve --slo-config /tmp/calibration-observe.toml --manifest /tmp/calibration-inputs.json --observations /tmp/calibration-raw.jsonl --out /tmp/calibration-report.json
```

要检查可部署 artifact，将 manifest 的 `validation_model` 改为上面的
`exported_profile`，并在 SLO 策略中配置原观察保留和真实时钟误差声明：

```toml
[cost_observation.profile_export]
path = "/tmp/live-profile.json"
observations_path = "/tmp/live-source.jsonl"
# 示例值：实际实验必须声明有根据的误差上界，不能由读取系统时间推断。
declared_clock_max_error_ns = 10000000

[cost_observation.profile_import]
declared_local_clock_max_error_ns = 10000000
```

cut 的两个路径、这里普通关闭导出的两个路径，以及命令的 raw/report 路径必须互不
重合，父目录须存在且目标不存在。缺少原始观察 exporter 或导入时钟声明会在加载
模型前报错。cut 导出及导入失败不会退回 `live_frozen`。普通关闭导出仍可能包含
验证阶段观察，不能用它替换报告中标明的 cut artifact。训练器或 exporter 的保留上限
会影响 artifact 覆盖；accepted ordinal 是队列序号，不等于文件中的 `source_record`。

输出父目录必须存在，目标文件必须不存在。报告保存 manifest、实际引擎配置、模型来源、
原始记录摘要、模型冻结点及 Known/Unknown/低估分母；原始记录包含实际 wave、
owner/incarnation/generation、逐行实际 host 特征、提交工作、耗时与输出终态。
请求输入摘要来自引擎实际 tokenizer 输出，不是 CLI 重做 tokenize；队列丢失的观察没有
accepted ordinal。原始报告中的本地单调时间
不能直接当作跨进程 Unix 时间导入 profile。输出 wire hash 包含协议帧，不等于模型文本
或 token 的语义一致性证明。

`prompts` 的索引可重复，表示不同真实 owner；`repetitions` 是预先声明的实验参数，
不是合格比例或固定通过次数。每次重复都完成原请求，后续重复不会接着使用其 KV。
`execution="split"` 先执行剩余 prefill，再执行 decode；`mixed` 提交当轮完整的
prefill/decode 集合，真实物理行顺序由资源 authority 决定。`prefill_chunk_tokens`
是每行上限，真实 tail 保留实际剩余长度。输出可改为
`{"kind":"completions_sse","include_usage":true}`，其 host 成本身份与 CLI 文本不同。
schema 2 使用 `{"kind":"chat_sse","include_usage":true}`，经产品 Chat 转换后
采集对应的输出投影成本；它不经过实际 HTTP 传输，不能据此宣称覆盖完整 Chat 服务成本。

manifest 文件上限 8 MiB，最多 4096 个提示词、训练和验证各 256 个 cohort；单 cohort
最多 256 个请求、64 次重复，总请求 owner 最多 65536。总时限最多一天，关闭时限最多
十分钟，wave 尝试最多一百万次，原始证据最多 256 MiB。配置应在这些硬上限内另设
适合实验的实际界限。超时、输出错误或不确定提交会停止新的 wave 并尝试真实关闭，
失败报告保留；Unknown 不算零成本，也不算已通过。退出成功仅表示采集流程完成。

#### 使用已有 ShareGPT selection

schema 2 从已有 `bench-serve` 报告恢复固定 selection：省略或清空顶层 `prompts`，
添加 `sharegpt`，并将 `protocol.output` 设为上述 Chat SSE。`sharegpt` 必须提供
`report_path`、`dataset_path`、精确 `tokenizer.json` 的 `tokenizer_path`、`repeat_index`，
以及 32 字节数组形式的 `report_sha256` 和 `selection_sha256`。相对路径以 manifest
目录为基准；cohort 的 `prompts` 索引指向所选 repeat 的完整样本序列，包括 warmup。
命令逐项核对源文件、tokenizer、selection、记录 ID、文本摘要、长度和过滤条件，
完整保留原 `requested_output_tokens`，不重新抽样或缩短输入输出。

`sharegpt.request_policy` 显式提供 `requested_model_name`、`sampling`、`ignore_eos`、`enable_thinking`、
`reasoning_effort`、`server_default_enable_thinking` 和 `interleaved_system_coalescing`。
三个可选值字段也必须出现；用 `null` 表示未指定。采样、ignore_eos 和 enable_thinking
必须与报告相符；旧报告没有 reasoning_effort，因此该值只记作本次 manifest 声明，
不能证明历史请求曾发送同一字段。`sharegpt.read_limits` 可收紧源文件、记录大小和
记录总数的读取上限。

`requested_model_name` 是实际 HTTP 请求中的公开模型名，例如 `"benchmark-model"`，
必须非空，且原样用于 Chat 请求 body 和 SSE output contract；模板解析与执行请求
仍使用实际内部 model ID。不要把 GGUF 路径自动当作公开模型名。报告的 `model` 是
可带 `--tag` 后缀的展示标签，不能可靠还原 wire 名称；当前校准会分别记录该标签与
manifest 的显式公开名声明，不拆后缀猜名称，也不声称已核实历史 body。
公开名长度和 JSON 转义会影响真实输出 envelope/host cost identity，部署 profile
时必须使用实际目标服务的公开名。schema 1 raw-input pilot 不受此字段影响。

恢复的 human 文本走与 `bench-serve` 共用的请求构造，再走产品 Chat 校验、模型模板、
thinking 和 completion/EOS 转换。证据分别记录 Chat body、渲染文本摘要及引擎实际
token 身份；缺失历史服务端 token 摘要时不声称已经证明跨运行 token 完全一致。
这条接线已通过输入恢复回归，真实 9B 校准仍待验证；受限成本校准不能替代完整
ShareGPT 服务 benchmark。schema 1 的手写示例仍是独立诊断输入。

### Selected whole-wave 三阶段模型（profile 6）

`cost_observation.predictor = "selected_whole_wave_v1"` 显式启用独立的
selected whole-wave 预测器；默认仍是原 legacy 预测器。它消费实际与未来共同提供的
selected algorithm family、数值工作量，以及完整 host-settled wave 的真实观察。
执行 canonical、资源权限和提交 guard 不因统计汇聚而改变。当前支持域是执行器
实际声明的 `PlainTextGreedyV1`；请求配置相似或某个 provider 名称相同不能替代
这项声明。任一必需 producer 缺失或路线不支持，仍返回 Unknown。

在已有 Observe/CompleteRequests/credited 策略上设置：

```toml
[cost_observation]
predictor = "selected_whole_wave_v1"

[cost_observation.model]
min_samples = 8
residual_quantile = 0.99
drift_margin_ns = 100000
context_bucket_tokens = 1
prefill_offset_bucket_tokens = 1
[cost_observation.model.feature_model]
kind = "exact_v1"

[cost_observation.profile_import]
# 仅示例：替换为有实际依据的本机墙钟误差上界。
declared_local_clock_max_error_ns = 10000000
```

这里 legacy feature selector/bucket 必须保留 `exact_v1`/1，避免将旧特征模式
解释成新模型；新预测器本身不使用旧精确键作为统计 family。未列的容量、TTL
等字段保留类型默认值；部署策略须与 artifact 设置一致。`min_samples` 在 fit 和
residual **各自**生效，不能相加达标。`max_samples_per_bucket` 的上限为 4096，
约束每个 family 的 fit+residual 合计样本数；长 decode cohort 可能填满小桶，
配置前应同时预算两个阶段。容量不足明确失败，不靠静默截断或缩短原输出补救。

在已有合法 manifest 中保留 `training` 和 `validation`，将 `validation_model`
替换为以下结构。这里只展示结构；索引、重复数、输出和容量须在采集前按目标域
声明，示例中的一次执行不代表足够的 family 支持：

```json
{
  "kind": "selected_whole_wave_v1",
  "export": {
    "path": "selected-profile.json",
    "observations_path": "selected-source.jsonl",
    "declared_clock_max_error_ns": 10000000
  },
  "residual": [
    {
      "prompts": [0],
      "repetitions": 1,
      "prefill_chunk_tokens": 128,
      "execution": "split",
      "decode_route": "actual",
      "token_policy_residency": "preserve"
    }
  ]
}
```

继续使用原 `ferrum calibrate-slo` 命令。顺序固定为所有 `training` 完整执行 →
冻结 fit receipt → 所有 `residual` 完整执行 → 密封 source/profile 6 并通过产品
loader 重新导入 → `validation` 独立 heldout。每阶段创建真实请求并完成原输出
预算、排空 owner 与消费者；相同预声明内容可独立重跑，但不能将一次执行的记录
复制到两个阶段，也不能称它验证了未见过的内容。heldout 只查询冻结模型，不更新 fit
或残差；不以 planned cancellation 代替请求完成。

该模式的 export 路径由 manifest 独占，必须省略策略中的旧
`cost_observation.profile_export`；manifest 也必须省略 `reference`。需要参考时，
使用独立命令先完成真实 reference 采集与 source join，再单独运行本三阶段采集。
不能把参考先前采集的 source 重新标成后来的 fit/residual capture。路径仍相对
manifest 目录解析，文件必须不存在；raw/report/source/profile 互不覆盖。

profile 6 带有独立 capture 身份、冻结 fit 参数摘要、fit/residual accepted cut、
原始 source 摘要和测量时钟，导入器校验真实执行指纹及设置并重建模型。旧 profile
1–5 不含这些证据，不能改版本号升级，也不能在新 predictor 下退回 legacy Known。
导入 receipt 的 `selected_whole_wave` 单列两阶段记录数、cut 和身份。普通观察
不会重训或刷新已加载的不可变模型；加载、密封和 heldout 都不会重置 TTL。

报告的 `selected_validation_offered/known/unknown/underestimates` 及
`selected_validation_unknown_reasons` 专门描述 heldout；`capture_support.family_support`
分别列 fit/residual 支持数。后者及 retained-point Known 不是 heldout 命中率。
完整 role pattern、参与者数、真实算法链、输出分支和 dispatch 结构仍可能拆分
family，prefill/terminal/尾部宽度的一次观测通常不能满足 min8。联合数值域外、
`InsufficientFit`、`InsufficientResidual`、过期或缺 producer 都继续 Unknown。
导入成功不能证明未来候选全覆盖，heldout 是实际完成波的回顾检验，不是提交前
时延承诺；真实 planner witness、完整 ShareGPT 服务质量及六项延迟/吞吐仍需独立验证。

### 独立 attention family 与 work-support（profile 7 / 8）

两种后续模式均为显式选择，默认预测器不变：

| `cost_observation.predictor` 与 manifest `validation_model.kind` | profile / source header | family schema | 模型行为 |
|---|---|---|---|
| `selected_independent_attention_v2` | 7 / 2 | 2 | 消费 producer 已证明的独立 attention 行子链，保留原全部支持坐标 |
| `selected_work_support_v1` | 8 / 3 | 2 | 同一个 V2 family，仅从统计支持匹配中排除 `output_budget_sum` |

family 版本描述 producer 的算法结构证据；model revision 描述如何拟合和判断支持域，
两者是不同身份。work-support 的 revision 为
`whole_wave_piecewise_affine_independent_attention_work_support_v1`，进入参数摘要、
profile 和查询标识；它没有生成新的 family，也没有扩大 Graph 或其他未声明的执行域。
缺少真正 V2 producer 仍为 Unknown。

`maximum_output_tokens` 继续保留在原始记录、canonical 和 host metadata，参与请求
授权、终态判断及资源容量；work-support 仅不再拿它的求和作为本波成本支持坐标。
终态/首 token 类别、真实已生成历史、KV、prompt、selected 工作量及其余支持坐标不变。
原 7 项回归本来不含该预算轴。移除一个非工作坐标不证明未知族、低样本、其他出域或
Known 低估已解决，也不把经验 residual q99 变成确定时延保证。

使用新模式时，在原合法策略中只显式设置：

```toml
[cost_observation]
predictor = "selected_work_support_v1"
```

该片段应并入既定策略，保留原 min_samples、residual_quantile、TTL、静态 margin、
声明 clock bound 和容量；不要用省略字段重新采用不同默认值。manifest 将
`validation_model.kind` 同样设为 `selected_work_support_v1`，并选择全新的
`export.path`（profile 8）和 `export.observations_path`（source 3）。其余字段及真实
请求恢复方式复用上述三阶段协议，不减少原 `max_tokens`、请求、输出或改变 EOS 规则。

先使用目标新 binary 独立采集真实 reference，再以相同执行配置运行新的 fit → freeze →
residual → 产品导入 → heldout。selected manifest 仍不得组合 `reference`，也不得使用
旧在线 `profile_export`；各阶段必须完成真实 owner 和原输出。旧 profile 6/7、source 1/2
不能通过改 header、重写时间或重标参数摘要迁移到新模式。profile 8 也不能由旧模式加载。
导入、普通观察和 heldout 都不刷新原样本年龄或重新训练被冻结的 fit/residual。

报告继续分别记录两阶段支持数与独立 heldout 的 Known/Unknown/低估；新模式及其导入
receipt 会明确显示 profile 8 和模型 revision，查询标识仍显示 family schema 2。
采集成功、产品导入成功及已有 CPU 回归均不能替代同配置 primary 服务、完整输出、
实际 witness 与六项延迟/吞吐验收。

### 从真实 singleton 试验发布固定参考

在 schema 1 或 schema 2 manifest 顶层添加 `reference`，并使用
`validation_model.kind = "exported_profile"`。以下只演示语法；索引、分区、重复数
和预热方案须在实际运行前按输入与容量确定，不是统计质量或性能验收门槛：

<!-- calibrate-slo-reference -->
```json
{
  "revision": 1,
  "request_policy": {"kind": "fixed_reference_output", "max_tokens": 16, "eos": "ignore"},
  "artifact_path": "prefill-reference.json",
  "frozen_plan_path": "reference-plan.json",
  "warmup": [],
  "curve_prompt_indices": [0],
  "granule_tokens": 128,
  "repetitions": 3,
  "decode_unit": {"prompt_index": 0, "generated_before": 1}
}
```
<!-- /calibrate-slo-reference -->

`warmup` 必须显式出现；空数组声明无预热，非空时使用与普通 cohort 相同的结构。
运行顺序固定为预热 → singleton discovery → 保存并同步完整冻结计划 → 新 owner
singleton trials → 普通 cost training → 同一个 accepted cut 的 profile 导出、真实导入和
reference source join → 独立 heldout。每个请求都按自己声明的完整 `max_tokens`、EOS 和 stop
规则运行至真实终态并排空输出。采到目标只停止参考收证；后续波仍写 raw，但不进入
reference 评分。目标之前提前 EOS、训练未 Recorded、源记录丢失或实际路线变化均失败，
不会换目标、选最快波、重写时间或把 batch wall 均摊成 singleton 耗时。

发现阶段只确定实际 provider/output/host 身份与完整原输入分区；冻结文件同时保存
实际引擎配置、输入来源、声明方案、plan hash、accepted cut 和 discovery ordinals。
完整文件先原子发布，再创建 trial owner。最后从同一 cut 的原始 `source_record`、
原始 wall 时间和实际 commit 链组装，调用产品 loader 验证后发布 `artifact_path`。
所有路径相对 manifest 父目录解析，并与 raw、report、live export、cut 互斥且不覆盖。

`request_policy` 缺省为 `{"kind":"original_input"}`，保留原请求的输出政策。
`fixed_reference_output` 必须显式声明正 `max_tokens` 和 `eos`（`respect` 或 `ignore`）；
它只创建独立的 warmup/discovery/reference 请求，复用原实际 Chat 模板并检查最终
prompt 字节相同。普通 training/heldout 和 ShareGPT selection 的原预算保持不变。
raw 分别记录来源预算、参考预算、EOS 及真实请求摘要；首次 engine frontier 提供 N/token
digest，报告 `reference_input_identities` 区分参考政策和原政策前沿是否实际观察到，
并核对同一来源在各阶段的 token identity。构造原请求不表示执行过原任务。

V1 只支持每条曲线一个实际原 prompt token 长度、统一参考 host 输出策略及完整 granule
端点；相同实际 token 长度的不同输入不能事后去重。`original_input` 仍拒绝不同原
`max_tokens`；独立固定策略允许原服务预算不同，但实际 host/路线不兼容仍明确失败，
不会归一化已测证据。参考标尺按原 N 和固定版本绑定服务请求，不要求服务预算与
参考预算相同，也不证明参考政策样本覆盖原服务成本。decode 单位冻结实际执行身份，
不从摘要推断 greedy。expanded recompute 或缺失端点仍无参考证据。

报告的 `phases` 分列 warmup/discovery/reference/training/heldout 请求数和波数；
参考阶段另列 `target_waves`、`preparation_waves` 与 `after_target_waves`。
`summary.reference` 给出可部署路径、协议摘要和 τ_ref。它是固定工作评分单位，
不是未来时延上界、成本 Known 覆盖或服务性能证明。固定参考不要求成本预测 Known；
成本 profile 导入仍按原样本年龄检查 TTL，heldout 期间也持续老化。
配置中可选 `limits` 复用 `SloPrefillReferenceLimits` 的文件、曲线、点数和样本硬上限。

V2 可在同一个 `reference` 对象中显式增加 `piecewise`：

```json
"piecewise": {
  "minimum_prompt_tokens": 1,
  "maximum_prompt_tokens": 513,
  "body_endpoints": [1, 32, 128, 256, 512]
}
```

这是参考协议语法示例，不是已有 9B 校准数据。`curve_prompt_indices` 仍指定真实输入；
discovery 后实际引擎 token 长度必须包含声明域两端，且每条曲线长度在域内，不会补齐、
截短或扩大已有 artifact 的覆盖域。`body_endpoints` 严格递增，最后一个为 Nmax−1。
仅 discovery/reference 的 prefill 使用这些绝对端点，先在各原 N−1 截止，再执行真实
final=1 波；decode unit 的准备也使用该分区。训练、heldout、预热与服务输出预算保持原声明。
后端不能合法执行该分区、真实尾波没有首 token 提交或来源链缺失时，采集失败。

schema 2 artifact 从最长输入的非最终分段冻结单调累计 B(p)，从少量不同原 N 的真实
最终单 token 波冻结 F(N)，仍使用真实 singleton decode 的固定 τ_ref。声明域内整数线性
插值只定义工作分数：p<N 时 W(N,p)=B(p)，p=N 时 W(N,N)=B(N−1)+F(N)。F 包含最终
首 token 波的全部工作，不是从 wall 中减出一个估算 logits 时间。插值不是实际波耗时
上界，不扩展成本 profile 的 Known 证据；最终首 token deadline、资源与输出提交门不变。

省略 `piecewise` 仍输出 schema 1，保持精确 N/endpoint 查表。V2 的域、分区和插值算法
进入独立协议 SHA；运行时按真实原 N 绑定不可变函数，不按 online 成本模型更新，不重置
原 admission 或有效 prefix 高水位。合法候选由后端约束筛选，不必结束于实测参考点；
中间进度只容忍一个实际合法 granule 的量化差，最终 deadline 不容忍该差。

### 显式选择有界输出

在同一个 `--slo-config` 文件中添加：

```toml
[output]
transport = "credited"
```

该选择与 `mode` 分开：Off/Observe 默认仍使用 `legacy`，显式 `credited`
可用于输出隔离验证和后续成本校准；它不表示 Enforce 闭环或性能验收已经完成。
配置继承现有 CLI/配置文件优先级和运行快照记录，不接受环境变量覆盖。

当前 HTTP 接线支持 `POST /v1/completions` 的 `stream=true`、单个文本 prompt
和 `n=1`。请求在提交模型工作前预留输出容量，使用唯一 codec 编码 SSE，
Body 直接接收带额度的字节；最后一个本地 Bytes owner 释放时才归还相应额度。
此边界不是远端客户端收包确认。finish、usage、`[DONE]` 保持此端点的既有规则；
当前 Completions 请求没有 `stream_options` 字段，因此仍输出 usage。

Chat SSE 支持普通 Text 的 content/reasoning 投影，复用模板实际生成的 reasoning
状态；`stream_options.include_usage` 仅显式为 true 时输出 usage。一次模型 token
可以产生多个语义事件，owner 保留有界游标逐个交付，终态先排空剩余文本再发
finish、可选 usage 和 `[DONE]`，不会再次序列化或使用旧的无界中间队列。
模板打开 reasoning 的请求保留原有 `AfterDelimiterAndPayload` 与 EOS 屏蔽规则：
需要词表原子 delimiter 或 tokenizer 冷编译的精确多 token 标记，以及声明的有界
strict-prefix 增量能力。HF 启动时从类型声明的 reasoning/tool 标记生成有限表，
请求时只借用并校验，不为未知标记临时运行通用 encode。matcher 的 token/failure
存储在额度取得后分配。alternate envelope 已支持通用 matcher 和 CLI 原样文本，
Chat 的工具投影完成前仍明确拒绝 alternate；不会篡改模板状态或 max_tokens。
`ignore_eos` 保持既有语义：关闭自动模型 EOS 停止及相应 completion gate，
保留显式用户 stop 和原有输出 token 上限；不会因为关闭 EOS 而自动扩大预算。
工具 envelope 的词法完成只允许 EOS，不代表工具参数已经通过协议验证。

显式选择 `credited` 后，非流式 Chat、Responses 和非流式 Completions 当前会在
推理前返回明确的 Unsupported 请求错误，不会退回旧队列。Chat tools、native/Harmony、结构化输出、
未声明边界的证据历史等也必须有各自的投影存储证明；引擎会拒绝缺少所需能力的组合。
这些是本阶段的未完成项，不缩小最终协议覆盖目标。

CLI 支持 `ferrum run ... --prompt "Hello" --output-format text` 的原样文本输出：
直接写出 codec 的 UTF-8 字节，不裁剪空白、不额外复制完整响应，额度持有到
实际写入和 flush 返回。一个请求使用一个阻塞 writer，没有中间输出队列；
操作系统的阻塞写无法由取消异步等待直接中断，未返回的写仍占用额度。
`--bench-mode` 消费并释放字节但不输出正文；统计写 stderr，区分 token 和文本事件。
单次文本请求支持 `--profile-detail basic|latency|kernel` 及 `--profile-jsonl`，
终态记录与已编码文本共享原请求的额度合同。交互历史、JSONL 输出、CLI request dump、
memory/scheduler lifecycle sinks 及 replay/debug/verify/full bundle 当前仍明确拒绝；
`--device-memory-jsonl` 独立可用。这些变体后续仍需各自的输出/历史额度，
不能用任意文本复制绕过预算。

HTTP credited 文本端点也可保存上述终态 profile；配置 request dump 时，成功请求的
prompt token ID 证据同样由原请求额度持有。请求在开始执行前按有效 prompt/output
上限预留证据空间；commit 时间序列和阶段记录不会另建无界历史。阶段记录额度
用尽后显式计数遗漏，不减少模型工作量、不截断用户输出，也不把部分记录称为完整。
序列化直接借用带额度的终态对象，写出结束前不释放其内存额度；这不等于客户端
已经收包，也不证明磁盘持久化。CLI 单次执行会等待该写出结果后再结束。

profile JSONL 可以同时包含 vNext 执行记录与 `phase = "credited_generation"`
终态记录，应按 phase 及 `attributes.execution_request_id` 关联，不能假定文件
只有一行或每行都是请求终态。

服务关闭先等待 HTTP drain，再等待已登记的终态证据写出，之后清理引擎；
这些步骤及并发关闭的串行等待共用 `HttpServer::stop(timeout)` 的同一个截止时间。
写出失败或超时会使关闭返回错误，同时仍尝试在原截止时间内清理引擎；
实际写出错误在再次关闭时仍保留。超时不能中断操作系统的阻塞文件写，
尚未完成的写仍持有原额度，后续关闭可以继续等待；不得将超时返回解释为内存已经释放。
这不保证真实引擎清理过程被取消后可以安全重试，也不保证 Tokio runtime 销毁有相同时间上限。

上述额度覆盖已声明的输出、解码工作区和 completion matcher；当前用户
`stop_sequences` 在引擎构造停止条件时仍调用通用 tokenizer encode。该编码的
临时工作区尚未建立有界能力，因此不能据当前实现宣称全部请求侧 host 内存已受额度约束。

当前 credited 引擎要求非 speculative 的 PlanRuntime 与声明有界解码、原始 token
字节能力的 tokenizer；预算使用请求的有效 `max_tokens`，不足时拒绝，不能静默缩短
输出。完整输出的字节储备与可循环 event slot 分别计账；默认字节上限不是所有模型或
请求长度都可接纳的承诺。持续输出阻塞超过 `slow_consumer_timeout_ms` 会取消推理，
已经交给传输或仍被投影持有的内存不会提前归还。

输出 owner 按 `max_queued_events`、实际请求额度及编码上限建立有界 data 队列，
终态槽独立保留。用于前瞻的 token 窗口只计算当前持有的事件额度和队列空槽，
Chat 按一次 token 可能产生多个文本事件计算；不假设消费者会及时读取或释放额度。
生命周期字节储备已计入池占用，规划时不会再从全局空闲字节重复扣除。
该只读窗口不持有执行许可，实际提交仍须取得当前 Ready grant。

Observe 的终态计数器为 `ferrum.engine.slo_terminal_requests_total`，标签包含
`service_class` 和 `internal_timing`（pass/fail/unknown）。可信 token 计时可产生
`ferrum.engine.slo_token_ttft_seconds`、`ferrum.engine.slo_token_tpot_seconds`、
`ferrum.engine.slo_request_max_token_itl_seconds` 直方图。
启用 scheduler/profile journal 时，`engine_slo_terminal` 事件保留相对入口的
first/last token commit、first/last engine text prepared 和 engine terminal。
text prepared 是引擎文本准备边界，不能称为 HTTP 发送或客户端可见；
engine terminal 也早于物理资源清理和传输终态。
内部 timing pass 仅表示已观察 token 时序，不表示客户端 SLO 或请求成功。

## HTTP benchmark 独立 sidecar

客户端文件是 JSON `SloClientVisibleConfig`，不接受完整运行时策略。
下面仍仅演示字段与 CLI 语法，必须替换为验收前声明的预算。

```json
{
  "latency": { "ttft_ms": 500, "tpot_ms": 40, "itl_ms": 80 },
  "attainment": {
    "ttft_percentile": 99.0,
    "tpot_percentile": 99.0,
    "itl_percentile": 99.0,
    "itl_percentile_scope": "pooled-gaps",
    "min_accepted_joint_attainment": 0.99,
    "min_offered_joint_attainment": 0.98,
    "max_reject_rate": 0.001,
    "max_error_rate": 0.001
  }
}
```

保存为 `/tmp/ferrum-client-slo.json` 后，可对同硬件 Ferrum 或 llama.cpp 执行：

```console
ferrum bench-serve --base-url http://127.0.0.1:8000 --model benchmark-model \
  --tokenizer /models/model --dataset sharegpt --sharegpt-path /datasets/sharegpt.json \
  --seed 42 --num-prompts 64 --warmup-requests 4 --n-repeats 3 --concurrency 4 \
  --output json --out /tmp/ferrum-legacy.json \
  --slo-client-config /tmp/ferrum-client-slo.json --slo-out /tmp/ferrum-slo.jsonl \
  --slo-fail-on-violation
```

`--slo-client-config` 与 `--slo-out` 必须同时提供。sidecar 按 cell 追加一行 JSON，
包含版本号、合同摘要、完整旧报表以及每个 repeat 的原始时间/间隔证据与新评估。
新 capture 开关和合同摘要进入原有环境 hash；旧报表 schema 和既有 TPOT 定义保留。
`--slo-fail-on-violation` 在报表输出后让任一 repeat 的非 Pass 返回错误。
当前 sidecar 支持 `--scenario standard` 的闭环、并发扫描和开放负载，
与 `decode-isolation` 联用会在运行前拒绝。

所有输出的父目录必须已存在。`--slo-out`、`--out` 和配置输入必须指向不同文件；
检查会解析已存在路径或父目录，拒绝符号链接别名，Unix 上也检查硬链接身份。
不存在的父目录会在发请求前报错，不会把包含 `missing/..` 的路径当作安全别名。
输出文件为追加或旧报表原有写入方式；请为独立实验选择独立路径。

## 指标和分母

| 指标 | 新 sidecar 定义 |
| --- | --- |
| TTFT | 客户端请求开始到首个非空可见文本更新 |
| TPOT | `(last_visible - first_visible) / (usage_output_tokens - 1)` |
| pooled visible ITL | 所有 offered attempt 已观察到的连续非空可见文本更新间隔；每个 gap 一个样本 |
| request max visible ITL | 每请求最大可见间隔，联合达标必须逐请求检查 |
| accepted joint | 明确被服务接纳的请求中，任务成功且 TTFT/TPOT/max-gap 均满足者比例 |
| offered joint | 所有有效 offered 请求中的联合通过比例；拒绝、失败、pending、unknown 不算通过 |
| reject / error | rejected / offered；failed / accepted，二者分母不同 |

旧报表 TPOT 仍使用 `(terminal - first_visible) / (output_tokens - 1)`，
因此尾部 usage/finish/EOF 等等待可导致旧值与新值不同。不得直接混用两个定义。
TTFT/TPOT 和 request-max 的聚合分布当前明确标为 `completed_requests`；
主 pooled ITL 标为 `all_offered_observed_gaps`，包含失败流和未完成流已观察到的停顿。
角色、空文本和仅 finish 消息不构成可见事件。传输合并与 event/usage 不匹配会披露，
不会移除真实可见停顿；严格单 token 时序资格另存为诊断证据。

Accepted 表示可观测服务接纳，不等于 GPU admission 或未来 TimePromise。
HTTP 成功流中可解析的 OpenAI SSE 响应提供接纳证据；晚发流内错误仍保持 Accepted
并计错误。HTTP 429 为 Rejected；连接失败、HTTP 500 和没有有效流证据的失败为
Unknown。旧记录中成功完成且确有输出可证明 Accepted；失败旧记录不能凭空补接纳证据。

`min_accepted_joint_attainment` 默认 0.99；`min_offered_joint_attainment` 默认不启用，
但始终报告 offered 联合计数和比例下界。accepted 不通过 offered 减 rejected 推算。
缺失接纳或计时覆盖不能判 Pass；零成功样本、空分布不会以零延迟通过。
单 token TPOT 和不足两个可见事件的 ITL 为 N/A；已完成但零输出不是成功，
无法观察首个文本则为 Unknown。Pending 的可见 gap 仍保留，但分布覆盖为 Unknown。

Raw output、successful output、SLO output 吞吐分别保留原始、任务成功、联合通过的
token 集合，分母都是该 repeat 的完整测量时长（含 drain）。Raw count 带来源；
SSE 文本 event 不冒充 usage token。Successful/SLO TPS 使用 usage，缺失 token 数量
或资格证据时 TPS 为未知，同时保留已知 token 下界和缺失覆盖计数。

## 开放负载证据与容量边界

将示例的 `--concurrency 4` 改为 `--request-rate 1.0` 会选择固定种子的 Poisson
开放负载调度。该数值也只是语法示例，不代表可持续请求率。
sidecar 的每请求时间均相对 measured repeat 起点：

- `scheduled_arrival_ms`：预先生成的到达目标，仅开放负载有值。
- `dispatched_ms`：客户端开始分派任务的时间。
- `request_started_ms`：现有 collector 在 reqwest 提交前记录的时间，不是 socket 发包或服务入口时间。
- `client_dispatch_backlog`：已经到时但尚未分派的到达数，包含本次请求；不包含在途、连接池或服务端队列。

`observed_request_start_rate_rps` 使用 `(observed starts - 1)/(last start - first start)`，
不把 drain 算进到达窗口；不足两个有效 start 或窗口为零时不产生数值。
这只描述客户端 collector 启动过程；网络、连接池和服务实际入口仍需独立证据。

`latency_and_outcome_status=pass` 只说明本次延迟与结果合同通过，不能证明稳态、
到达过程在服务端完整兑现、等待队列无漂移或最大安全容量。
最终容量结论仍需固定服务器配置、相同 ShareGPT 样本与输出策略、相同硬件/模型精度、
足够样本和重复、实际到达与 backlog/drain 证据，以及错误、拒绝和 Peak VRAM/内存。
流程见 [性能评估](performance-evaluation.md)；历史结果不能替代当前 dirty 代码的验收。

### Resource workspace preparation before readiness

The shared `run` / `serve` startup hook can prepare the resolved plan's existing
Step and Invocation workspace buckets without executing model waves:

```toml
[runtime]
reusable_execution = false
workspace_preparation = "startup"
```

`workspace_preparation` defaults to `"demand_driven"`, which preserves lazy
resource preparation. `"startup"` is an explicit resource-only option: it admits
disposable owners within the configured capacity, prepares each declared bucket
using ordinary bounded resource maintenance, drops the prepared wave and aborts
those owners. Idle lane slots remain resident. It neither captures/replays a
device program nor executes token generation, initialization uploads, sampling,
observations or training. It does not warm all GPU/driver caches.

This mode requires a reusable workspace plan and also supports plans with a
device-program policy. Resource preparation runs before the separately configured
program preparation: on-demand capture still starts with an empty program catalog;
startup capture still performs its real warmup, capture and validation. Resource
preparation alone cannot establish a warm Graph route or a known execution cost.
Unsupported buckets, capacity exhaustion, or incomplete cleanup fail
startup; the mode never shrinks the configured request capacity, skips a bucket,
or changes the memory budget to fit. It prepares declared capacity classes,
not every future request's KV backing or every provider route. Runtime guards,
future resource checks, model coverage and profile TTL still apply.

The startup report includes actual prepared bucket IDs, Step/invocation IDs,
peak simultaneous disposable-owner count, before/after resource epochs and pool
resident/free bytes, process-claimed bytes and the effective device ceiling.
The zero model-wave counts describe this resource-only path. Ordinary startup
and device-program preparation retain their separate existing report fields.
The typed option is included in the effective runtime configuration snapshot.

### 独立结构化 V2 发现与暖机

`calibrate-slo` 可先执行独立发现，再用另一份事前冻结的 manifest 采集模型。
发现模式不需要预先知道 owner/domain，仍使用实际 native executor、完整请求预算、
credited 输出和原有 cohort driver。SLO 配置须显式选择 `observe`、`complete-requests`、
`credited` transport、`structured_whole_wave_v2` predictor 与 `host_settled_v1` capture；
不能导入已有 cost profile 或启用 legacy profile export。

在原 manifest 中设置：

```json
"validation_model": {
  "kind": "structured_discovery_v2",
  "warmup": []
},
"validation": []
```

`training` 此时是独立发现 cohort 列表；`warmup` 接受相同的 cohort 结构。
先完整执行并排空所有暖机请求，再完整执行发现请求。Reference 模式与发现模式互斥。
重复数、输入索引、rolling window、wave plan、全部 fresh owner、超时及 raw 字节限制
同时覆盖暖机与发现，达到目标波次后不会缩短请求。

报告的 `summary.structured_discovery_v2` 只汇总 Discovery phase 中的原始 typed
receipt，保留 owner/domain、pending/Length 计数和位置、同一波次的联合计数，以及
与 raw 逐波数据同序的 basis/support 数值范围。范围端点不代表实际共同出现的输入；
位置并集也不代表同时出现的 pending/Length 组合。Unknown 原因单独计数。
清单最多保留 128 个域、65,536 个数值坐标、16,384 个联合计数项；达到限额后
`inventory_truncated` 会保持为 true，原逐波 raw 仍受原字节上限约束。
`collection_completed` 只表示完整 cohort driver 已成功返回；模型资格与 SLO 达标
不能从此字段、样本最小数或发现清单推出。

随后显式使用 `structured_whole_wave_v2` 采集模式，冻结实际发现的 owner、numeric
membership windows、覆盖挑战及三阶段完整请求计划。其 `capture` 也接受可选
`warmup` cohort 列表：完整 manifest 及采集 options 在暖机前固定，原 source 只在
暖机排空后读取真实 opening clock/FIFO cut。暖机进入完整 manifest 摘要，但不成为
fit/residual/qualification 的请求槽或成员。省略 `capture.warmup` 保持原行为。
发现本身不会创建模型 source、训练成员或 profile10，不会授予未观测未来分支资格。
