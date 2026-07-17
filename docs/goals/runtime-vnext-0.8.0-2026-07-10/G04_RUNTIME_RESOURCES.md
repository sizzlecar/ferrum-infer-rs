# G04: 共享 Execution Runtime、Scheduler 与资源事务

## 状态与依赖

- 状态：Open
- 依赖：S0A contract split；S0B/S1 actual Qwen3.5-4B CUDA runtime；随 S2-S5 强化
- 下游：S1-S7、G05、G06、G08-G10

## 目标

建立唯一共享执行 runtime。模型只描述 program/block/state；runtime 管理 batch、prefill、
decode、资源、执行次序和退出。解决当前 engine/model/KV manager 多重所有权以及模型复制
unified runner 的问题。

动态资源、transaction/lease、physical submission、fence/reaper/recovery 是核心 runtime 的不可约
复杂度，不以缩短总 LOC 为目标。S0A 必须先按 capacity/provisioning、pool/extent、transaction/
lease、request/sequence/session、step/invocation 和 completion/recovery 所有权拆分；S0B/S1 再允许
由实际生产 consumer 驱动 breaking rewrite。固定并发、模型/GPU 特判或弱化 defer/resume 合同均不
属于可接受的“简化”。

## Runtime 职责

1. request admission、capacity、defer、resume、cancel。
2. prefill/decode/mixed-batch 计划和公平调度。
3. immutable `ExecutionPlan` 驱动的 layer/block traversal。
4. KV、recurrent state、scratch、graph workspace 的唯一事务。
5. logits/final-token selection 与 sampler 边界。
6. 所有正常、错误、cancel、disconnect、timeout 路径的 deterministic cleanup。

Admission 必须采用 continuous-batching 语义：配置并发只是 ceiling，实际 active 数由当前
KV blocks、recurrent state、workspace、显存余量和请求形状共同决定。资源不足只 defer 当前
请求；不得提交其 prefill，也不得停止已有 decode 或其他当前可满足的 waiting request。释放任何
相关 lease 都递增 monotonic global audit `capacity_release_epoch` 和 exact availability source
generation；等待队列只在自身 `CapacityWaitCondition` 相关 source、请求取消或策略发生变化时
重新求值。无关 domain 即使推进 global audit epoch 也不得触发 admission probe。该合同避免 busy
retry，同时必须有 aging/fairness 防止大请求永久饥饿。

capacity 必须是 typed vector/domain authority，至少区分 KV group/page、recurrent arena、
workspace arena、sequence count 和不可互换的 backend pool；禁止用单一 total bytes 假装这些资源
可互换。共享 backing arena 可以在加载时创建或按显式 grow transaction 扩容，动态 request lease
只领取 block/slice/slot，不执行每请求 device allocation。active sequence id 由 admission 内部分配
并在 release 后回收，不能让调用者任选 slot，也不能按 ceiling 建 Vec 或保留无限 slot history。

每个动态 descriptor 必须带 core-derived storage contract 与 pool id；pool compatibility key 至少包含
allocator kind、contiguous/paged view、usage、dtype、layout fingerprint 和 alignment，并进入
`MemoryPlan`/plan hash。workspace storage 由 provider estimate 显式声明；value/state layout 由 core
从绑定推导，禁止从 `Pages` 等 size formula 猜物理 view。资源层只能消费 plan pool spec，不能再按
descriptor 建 domain 或按比例瓜分 `usable-static`。同 pool 的 actual demand 先 checked sum 后只产生
一个 capacity entry/physical claim；不同 key 必须分池。

semantic `StateSpec` 只描述 logical tensor/lifetime 和 fixed-per-scope 或 token-scaled capacity；不保留
含糊的 `PageScaled`。provider binding/workspace requirement、device offered profile 与 typed runtime
preference 共同选择 `LinearArena|FixedBlockArena` 和 `Contiguous|PagedRegions`。同一模型/StateSpec 在
CUDA/Metal 可选择不同物理合同，模型代码不得引用 allocator、pool 或 backend 名称。

`MemoryPlan` 保存 per-pool minimum/maximum 与 typed elastic provisioning policy；实际 initial capacity
由全局 device account 在 transaction 时按当前 live claim grant，并写入 runtime binding/artifact，不得
把 plan build 时的 `usable-static` 伪装成已安装容量。常见单模型路径可以由显式 policy 尽量使用剩余
容量；多 plan 争用时仍以真实 grant 为准，不能按模型名、GPU 名或显存档位分配。

逻辑 capacity 必须由已经成功分配并安装的真实 backing segment 推导，不能由
`usable_memory - static_bytes` 之类的剩余数字直接发布。全局 device account 只对 static buffer 和
backing segment 的真实物理字节各 claim 一次；request block/slice/slot 只改变所属 typed pool 的
逻辑 occupied/free，不得再次计入全局物理字节。每个逻辑 lease 必须线性持有 exact arena/segment、
segment generation、offset 和 length，并能向 provider 借出同一 committed device buffer 的有界
region view；只有 resource id、size、usage 等 metadata 而没有实际 buffer ownership 的 authority
无效。多个 plan/pool 不得分别把同一段全局剩余显存当作自己的 capacity。

initial arena 与 grow 都必须遵循 `global physical claim -> device allocate -> validate/install segment ->
publish typed capacity/epoch` 的提交顺序；任一步失败时 capacity total 与 epoch 均不变，并 rollback
或把真实 buffer 连同 global claim 一起 quarantine。request admission、page extend 和 defer/retry
不得隐式 grow。segment 只有在 live extent 数为 `0` 后才能从 domain 摘除并释放物理 claim。

真实 pool 由一个或多个 chunk 组成；`BackingSegment` 必须包含 chunk generation/ordinal、offset 和
length，region translator 通过 chunk identity 找 exact buffer。contiguous allocator 在碎片化时只能
defer/compact/grow，不能返回随后会被 provider 拒绝的多 extent；paged allocator 才能跨 extent/chunk。

每次 admission demand 必须分开记录：本次 step/chunk 的 `immediate claim`、证明完整 input 或
指定 frontier 可被当前 backing 容纳但不占用未来 block 的 `fit requirement`，以及不占资源的
max-context/concurrency ceiling。三者不得折叠成一个 peak bytes；fit 策略（full-input-must-fit、
immediate-only）、chunked prefill 和 preempt/recompute 必须是 typed 产品配置并进入 plan/artifact，
不能由隐藏环境变量或模型/GPU 名分支决定。若未来引入真正的 capacity reservation，必须使用第四个
独立 vector 计账并在 release 时归还；不得把 fit gate 伪称已经预留未来 KV。

每个 deferred 必须携带 coordinator identity 和 canonical exact source generations，至少覆盖被阻塞
的 capacity domain 与 active-sequence slot。global release/capacity epochs 只作为 monotonic audit 和
共享通知版本，不是 per-request retry predicate。waiter 必须遵循 subscribe -> snapshot -> exact
recheck -> park，封闭“检查失败与挂起之间发生释放”的 lost-wakeup 窗口；任何提高 effective
available capacity 的 transition 都必须同时推进对应 source generation 和 global audit epoch。无关
通知可以唤醒 listener，但 exact recheck 必须继续等待且 admission probe 增量为 `0`。独占全部
capacity 仍不满足的请求返回 typed `Impossible`。暂时不满足的队首可以被后续 eligible 请求跳过；
typed fairness policy 的默认 `max_relevant_bypass_events=8`，只统计该请求 exact source 变化、
重新求值后仍被后续请求绕过的事件；达到阈值后只暂停接纳新的 bypass 请求，不得停止 active
decode。Ferrum 借鉴 vLLM 的 active-first、capacity defer 和 fence-delayed
release，但不接受裸 `None`、1ms polling、队头 allocation failure `break` 或默认清空 KV 重算。

`da9c1ee8` 的真实 Qwen3.5-4B CUDA artifact 已否定 phase-local `DecodeProgressLease` 足以保证
全局活性的假设。该实现能够在 owner generation `49 -> 50` 后解除 admission barrier，但 victim
进入 recompute prefill 后与原 decode owner 同时等待 domain `4` generation `73`，最终形成
`1 blocked prefill + 1 blocked decode`，A/B 均无 `[DONE]`。因此当前 lease 只能作为迁移期
diagnostic，不能计入 G04 或 S1 PASS，也不能继续通过增加 phase-specific flag 修补。

vNext 的资源仲裁必须改为唯一的 typed logical work frontier。scheduler 只维护 request 的 logical
token frontier、resident/computed frontier、已提交但未完成 frontier、committed output generation 和
recompute origin；prefill、recompute、decode、speculative work 只是 executor `WorkKind`，不得成为
彼此独立的资源活性 authority 或必须跨队列拼接的不完整状态机。该设计借鉴 vLLM 统一 computed-token
frontier 和同轮 preemption 的原则，但不复制 Python object graph、裸 `None` allocation failure、
运行时字典拼装或无 fence 的立即复用。

权威 allocator 返回 temporary capacity failure 后才创建 cold-path `PressureEpisode`，并按 canonical
exact capacity source 建立索引。episode 必须分别保存 immutable `progress_owner`、held participant set 和
last transaction victim，禁止用一个可变 victim/owner 字段同时表达三者。状态至少覆盖
`Open -> YieldPlanned -> AwaitReleaseFence -> Resumable -> Closed`；当 stable owner 在仍有 held peer 时
释放自身资源进入 recompute，还必须经过
`OwnerAdmissionPending --typed admission receipt--> Resumable`。token progress、wait-source topology 变化和
另一个 participant 的排队顺序都不能转移 owner，也不能解除 held peer；owner 只有在 terminal release 后
才释放 peer。若 recompute owner 无法取得执行 claim，必须返回 typed invariant/blocker，不能把 held peer
晋升为新 owner 或静默轮转。一次 scheduler transaction 在提交 batch 前必须满足二选一：至少一个
frontier 获得可执行 claim，或已经产生一个 typed yield/release transaction，且被让渡资源在 fence
terminal 前不能复用；禁止把所有 frontier 先标成 passive wait 再期待另一个 phase 推进 source。
unchanged exact source 不重新调用 allocator，也不产生逐 tick event 洪泛，只更新预分配 episode counter
和有界 first/last sample。

机制与策略必须分离。资源机制只负责 exact claim、rollback、fence-delayed release 和 transition
ordering；调度策略基于 priority、相关 bypass age、resident/recompute cost、剩余 logical work、可形成的
batch width 和预计释放收益选择继续当前 cohort、yield victim 或等待。选择只发生在真实 capacity
boundary，不能按每 token、时间、模型名、GPU 型号或显存档位轮转。backend/provider 只提供 typed
capacity snapshot、cost/capability hint 和 release fence；scheduler 核心不得出现 CUDA/Metal/model
分支。

## Transaction 与 Lease 状态机

```text
waiting -> reserved -> committed -> released
   ^          | capacity changed        \-> rolled_back
   |          v
   +------ deferred
waiting -> impossible
committed -> deferred -> resumed -> committed
committed/deferred -> cancelled -> released
```

每个 transition 记录 request、resource kind、owner、amount、before/after、reason、plan node。
禁止通过 allocator/kernel OOM 才发现容量不足。

资源生命周期必须由类型明确分成五个语义 scope：`PlanRuntimeResources ->
AdmittedRequestResources -> AdmittedSequenceResources -> StepResourceLease ->
InvocationResourceLease`。`Step` 对每个 participant 等于一个 exact `ExecutionFrameId`，覆盖该 frame
的完整 topology；`Invocation` 才等于一次 exact batch node/provider 调用。Request state 在 `n`、
beam、speculative 等多个 child sequence 间只分配一次，Sequence state 每 child 独立，Step state
在各 participant 同一 frame 的所有 node 间保持，Invocation scratch 对整个实际 batch 只分配一次。
禁止把 Request 并入 Sequence、把 Step 并入 Invocation，或用含糊的 Node lifetime 同时表示
semantic state 与 provider scratch；Node 只可作为 `AllocationKind` owner identity。

`PlanRuntimeResources<R>` 是唯一 plan-lifetime owning root。zero-static 路径消费
`NoStatic::into_plan_runtime(self)`，static 路径消费
`ResourceTransaction<D, TransactionCommitted>::into_plan_runtime(self)`；handoff 必须连同 runtime、
driver、ledger、static buffers、dynamic pools、global capacity claim 和 recovery authority 一次转移。
`StaticProvisioningLease` 本身不能生成 trusted binding。`TrustedPlanRuntimeBinding<R>` 持有 root `Arc`，
Request -> Sequence -> `SequenceSession` -> Step -> Invocation 的 durable 类型全部删除 `'plan`、满足
`Send + 'static`，且所有被 `Arc` 共享的 target 满足 `Sync`。`OperationInvocation<'a, _>` 只允许作为
encode 期间的短借用，不能进入 fence/reaper record。

root close 先原子转为 Closing 并停止新 admission；任一 child 或 completion record 仍持有 root 时，
只能返回保留 root 的 typed `Referenced { resources, strong_count }`，不得释放任何 backing。取得唯一
所有权后，按 dynamic extent/pool -> resident chunk buffer -> chunk capacity grant -> static buffer ->
static capacity claim -> runtime 的顺序关闭。partial cleanup 必须返回拥有剩余资源的 retry/quarantine
authority；未显式 close 的 Drop 复用 transaction abandon，不能裸 drop 不确定资源。root 不得强持
completion reaper；engine 将二者作为 sibling，或只保存 `Weak`，杜绝
`root -> reaper -> record -> invocation -> root`。

continuous batch 的 Step/Invocation authority 必须持有 canonical non-empty participant set；每个
participant 都是 owning `Arc<AdmittedSequenceResources>`/Step hold，允许跨 Request 但必须共享 exact
plan/hash/device/runtime/coordinator/node/provider。public admission 不接受 raw `DynamicResourceShape`
或 caller aggregate token/page 数。model program 显式标注 token-bearing value/axis，core 将其解析到
plan node exact `ResolvedValueBinding`；page metric 绑定 exact state/resource/pool/storage profile 和
committed page-set authority。core 从每个 participant 的 token span、full-input 与 committed page set
checked 聚合不可伪造的 `BatchWorkShape`，其 sequence 数必须等于 exact participant 数；batch child
capacity lease 对全部 parent 建立 release edge，但 demand、physical extent、workspace 和 scratch 只
claim 一次。禁止按 tensor 元素数/模型名/backend 猜 work、canonical-leader 临时 authority、每
participant 重复 scratch 或 per-sequence execution stream。

`BatchWorkShape` 必须同时保存 canonical participant projections、immediate/fit typed metric vector 和
source fingerprint；相同 aggregate 但 participant partition、token span、page generation 或 page metric
不同，fingerprint 必须不同。shape evaluation 生成私有 `ClaimedBackingTransaction`，一起拥有 shape
authority、exact evaluated demand、logical capacity claim 和 physical extents；Step、Invocation、
`OperationInvocation` 与 in-flight record 只能移动/引用该 transaction，不能复制数字重新求值。空
demand 也保留 shape authority；dispatch 必须用同一 immediate shape 验证 dynamic extent/workspace
exact evaluated bytes，不能只比较 minimum bytes 或在 claim 后替换 work。

物理 scheduler step/invocation 使用 core-minted `BatchStepId`/`BatchInvocationId`；每个 participant 同时保留自己的连续
`ExecutionFrameId` 和 active-session fingerprint。长序列 frame=7、新加入序列 frame=2 可以进入同一
batch，Step guard 按各自 frame 获取；request journal node invocation identity 也由各 participant
cursor 独立生成。禁止要求 participant frame/node id 相等或复制 leader identity。

Step 内的 invocation registry 必须以 canonical
`ParticipantNodeKey = (sequence authority, request authority, ExecutionFrameId, NodeId)` 占位；
`BatchInvocationId` 仅是 physical attempt identity，不能用于规避重复检测。一次 prepare 对全部
participant keys 原子 transition：任何 key 已为 `Prepared`、`InFlight` 或 `Retired` 时整批
失败，Retired tombstone 保留到 Step 终态。合法状态机为
`Vacant -> Prepared -> InFlight -> Retired`，或
`Prepared -> NotSubmitted -> Prepared`。只有 typed `DefinitelyNotSubmitted` 可生成 sealed retry
authority；retry 必须保持 exact topology/work fingerprint 并使用 fresh attempt id。possibly-submitted、
indeterminate、changed-shape、普通 Drop 或 provider 自报 retryable 均不能进入该 retry 边。

每层 child authority 都必须绑定 exact parent authority，且 parent 的 backing/capacity 在最后一个
child 终态前不可释放。共享 scratch domain 由 plan 的 allocator/view compatibility 与依赖闭包
共同决定；只有依赖关系证明相关 node 全序时才可按 liveness row 取 `max`，否则先按可能并发的
invocation 保守求和，直到最大权重 antichain或显式 scheduler wave 提供更强证明。operation dispatch
必须按值消费 exact invocation lease，不能从 request/sequence/step lease 伪造 transient view。

这里的全序必须是 completion 全序：前驱 fence terminal 后后继才可 claim 同 pool extent。仅证明
node submission/dependency 顺序而允许两个 fence 同时 pending 时，仍按 concurrent sum 计账。

Invocation lease 的终态是设备 completion fence，而不是 `encode_and_submit` 返回。execution
stream 由 scheduler/device `ExecutionLane` 所有，不携带 sequence authority；同一 lane 可提交多个
mixed batch，sequence completion 只等待自己的 participant-flight counter，不能 drain 整条 lane。
提交前 durable reaper slot 必须已经预留；提交后 lease、physical extent、全部 participant hold 和
state hazard permit 一并进入 typed in-flight ownership，外部 handle Drop 只 detach/cancel，不释放
资源。scheduler 通过 poll/await 完成，不阻塞请求线程，cancel/disconnect 把 ownership 留给
completion reaper。
`CompletionReaper<R>` 与其 record 必须拥有上述 `'static` invocation chain，不得带 plan lifetime；
任何 `CompletionReaper<'plan, ...>` 临时合同都属于 G01/G04 blocker，不能作为后续再修的过渡实现。

`SequenceSession` 冻结 exact run/request/plan/coordinator、sequence authority generation 和
active-session fingerprint；Step/Invocation/in-flight record 必须持有 owning session/sequence parent。
session completion/abort 只等待自身 participant-flight 归零，不能通过 drain scheduler-owned shared
lane 使无关 request 阻塞，也不能在 live lifetime 内换 identity/parent resources，或在 exact
Step/in-flight projection 仍存活时替换其 frozen work authority。

`DeviceRuntime::submit` 的错误必须是 typed definitely-not-submitted；任何 partial enqueue、可能已
提交或 fence record 后置失败都必须返回 fence，再以 succeeded、failed-but-quiescent 或
indeterminate 终态处理。只有 exact fence quiescent 才能归还 extent/capacity；indeterminate 依次由
reaper blocking wait、独立 recovery worker lane drain、最终 quarantine 收敛。Drop 不得把仍可能被
CUDA/Metal 异步访问的 region 提前放回 allocator；阻塞 wait 只能属于 reaper/shutdown/panic 的
correctness fallback，不能成为正常请求路径。

Request lifetime mutable state 由 plan 的 typed state effect 生成跨 child sequence hazard authority：
read/read 可并行，任何 write 与其他 read/write 互斥，并且 permit 同样持有到 device fence。单 frame
内的 state DAG 不能冒充跨 sequence 的 Request state 同步。

当前普通非 state value 仍被 planner 保守归为 Request lifetime，这是正确但非内存最优的明确
G04 blocker。完成 producer/last-consumer 与跨 node alias liveness 证明前，不得宣称 activation
内存或吞吐已达到 vLLM 同级；后续必须把真正中间 value 下沉到 Step/Invocation，而不改变 program
output、共享 request input 或 state 的语义生命周期。

transaction state 与单个 lease state 必须分离。一次 request 可以原子包含 KV、recurrent、
scratch 和 graph workspace 多个 lease：

- reserve 阶段要么全部成功，要么按逆序补偿，不能留下 partial reservation；
- commit 顺序、provider failure after partial commit 和补偿 action 必须写入 contract；
- rollback/release/cancel transition 必须幂等，重复事件返回明确 already-finalized 而非 underflow；
- defer 明确哪些 lease 保留、哪些归还，resume 必须重新验证 generation/version；
- prefix/session cache retention 是显式 policy，最终 expected balance 与 leak 分开计算；
- disconnect/timeout 在任意 transition 点都有唯一合法终态。
- physical backing transaction 与 logical extent transaction 分层计账：前者拥有 buffer/global
  claim，后者拥有 segment extent；逻辑 lease 释放不得直接释放仍被其他 extent 使用的 segment。

## 状态类型

- full-attention paged KV；
- linear-attention recurrent state；
- hybrid model 同时拥有 KV + recurrent state；
- model/backend scratch；
- CUDA graph/Metal pipeline workspace；
- prefix/session cache lease。

这些状态通过 `StateSpec` 与 `ResourceLease` 管理，模型代码不得保存独立 hashmap/manager
作为第二真相。

## 测试

- seeded model checking / property test 至少 100,000 个状态序列。
- capacity=0/1/exact/overflow；并发 reserve；cancel at every transition；provider failure；
  partial prefill；mixed final/non-final；reallocation。
- full-input fit exact/minus-one、chunk immediate claim、可选 future reservation release、
  preempt/recompute；断言 ceiling/fit/immediate/reservation 四个计数互不冒充。
- Request/Sequence/Step/Invocation lifetime 交错：Request state 对 N child 只计费一次、Sequence
  state 每 child 独立、同 frame 多 node 共享 Step state；顺序 invocation scratch `max`、可能并发
  scratch `sum`。operation success/failure/cancel 只有 fence 完成后才可复用；无 in-flight work 时
  Invocation scratch 占用量必须为 `0`。
- 32 participant、1 execution lane、1 command、1 fence：participant set 非空/去重/同 plan，
  `shape.sequences=32`，batch child/scratch physical claim 均为 `1`；取消其中 1 个 participant 后其余
  31 个继续，sequence A terminal 不得 drain 或阻塞 lane 上 sequence B。
- participant frame `7/2/19` 保真；一份物理 batch ledger 与三个 request projection 共享同一
  submit/completion fingerprint，command/fence/throughput 物理计数均为 `1`。
- Request mutable state 跨 child sequence 的 read/read、read/write、write/write hazard grid；permit
  必须持续到 fence，非法并发 dispatch 成功数 `0`。
- submit 返回但 fence 未完成、cancel 后 reaper 未完成、completion error、Drop fallback 四条路径；
  fence 前 extent 重用数 `0`，正常 scheduler 请求线程阻塞 wait 数 `0`。
- definitely-not-submitted、pending N 次、failed-but-quiescent、indeterminate->wait 成功、
  indeterminate->lane drain 成功、lane drain 失败 quarantine 全矩阵；submit/poll/wait panic、handle/
  sequence/lane/reaper Drop 顺序均 exactly-one terminal/release。
- NoStatic/Static consuming handoff、Closing/Referenced/retry/quarantine 全矩阵；在 Request、Sequence、
  Session、Step、Invocation 和 reaper record 各层保留一个 hold 时 close 必须逐层拒绝，全部释放后
  close exactly once。用 drop probe 验证 dynamic buffer-before-grant、static buffer-before-claim。
- compile-pass 断言 root/binding 与 Request -> Invocation durable chain 为 `Send + 'static`、Arc target
  为 `Sync`；compile-fail 覆盖 root/binding 外部构造、static lease mint binding 和 consuming handoff 后
  再使用 transaction。source scan 禁止 owning chain 的 `'plan` 与 `CompletionReaper<'plan, ...>`。
- lost-wakeup race、temporary defer 与 permanent impossible 分类、global audit epoch 与 exact
  availability generation 单调、waiter register/recheck；只改变无关 domain 的 `10,000` 个事件中
  admission probe 增量为 `0`，相关 domain/slot 释放后的 retry 成功率为 `100%`；队首
  large-ineligible + 后续 small-eligible 在下一 tick admission `100/100`，aging 达阈值后不饿死且
  active decode 不停。
- mixed-size waiting queue：队首请求暂时不可满足时，active decode 与后续 eligible request
  继续推进；相关 source generation 推进后队首自动重试；无关 global audit 变化不得重新 probe；
  aging 后不存在永久饥饿。
- 从 `da9c1ee8` CUDA trace 固化 deterministic replay：最终 `active=2`、`blocked prefill=1`、
  `blocked decode=1`、两者等待同一 domain/generation。replay 在进入 passive park 前必须产生一个
  frontier claim 或 `YieldPlanned`，复现旧终态的次数为 `0/100`。
- phase 变化不能改变 request 的资源活性 identity：recompute request 从 waiting 到 executor prefill、
  再到 decode 的 transition 使用同一 frontier/pressure episode；跨 phase 复制或丢失 exact wait、
  fairness age、logical progress 和 release obligation 的 case 数为 `0`。
- transition journal 使用 scheduler-owned 单调 ordinal；同一 transaction 的 yield、release-fence、
  admission、submit 顺序必须由 ordinal 验证，不能用 wall-clock timestamp 推断。trace sink 延迟、批量
  flush 或关闭时 replay transition 与 batch membership 保持完全一致。
- 对 ceiling=`1/32/4096/u32::MAX` 的同一 graph，plan descriptor 数和 build allocation 数保持
  不变；除 `0` 外无任意固定 concurrency guard，实际资源 claim 只随 admitted request 数增长。
- 多资源第 N 个 reserve/commit 失败、逆序补偿、重复 rollback/release、defer retain/release、
  cache-retention policy 的组合 fault injection。
- initial arena allocation success/failure、显式 grow success/failure、两个 plan 争用同一 device
  account；断言只有已安装 segment 发布 capacity，且物理字节不重复计费。
- provision 后对 `10,000` 次 admit/release/extend 记录 device allocate counter，稳态增量必须为
  `0`；defer、Impossible、provider encode/submit 的同项增量也必须为 `0`。
- logical extent 的 offset/length/alignment/bounds/non-overlap、paged non-contiguous extent、segment
  generation stale reject；provider 必须观察到 exact committed buffer region，metadata-only proof
  必须在 dispatch 前被拒绝。
- Qwen3.5 hybrid state：KV 释放不提前释放 recurrent state，反之亦然；两者最终均释放。
- scheduler trace 可 replay 并得到相同 transition 和 batch membership。

## 验收

- 100,000 状态序列中 leak、underflow、double release、use-after-release 均为 `0`。
- owned plan-root 两条 handoff 通过率 `100%`；borrowed plan-runtime mint path 数 `0`，Request ->
  Invocation durable chain 的 `'plan` 参数数 `0`，`CompletionReaper<'plan, ...>` 数 `0`。
- 每个 child/reaper hold 下 close 提前释放数 `0`；全部 hold 清空后的 close 成功率 `100%`；partial
  release retry/quarantine ownership 丢失数 `0`，root/reaper strong cycle 数 `0`，两类 drop 顺序违规数 `0`。
- multi-resource partial reserve/commit fault grid `100%` 达到 contract 指定终态，补偿遗漏 `0`。
- vNext program/runtime 与 G04 L0/L1 纵切中 model-owned KV/recurrent manager 数量 `0`；
  legacy 模型的全仓/source-tree 零值由 G08D 验收。
- EngineBuilder 构造后丢弃 scheduler/manager 的路径数量 `0`。
- 所有 capacity reject/defer 在 kernel launch 前发生 `100%`。
- 所有 capacity defer 的 provider encode/device submit/prefill launch 增量均为 `0`；同时存在
  eligible work 时 scheduler idle tick=`0`、全局 HOL block=`0`。
- 任一 pressure snapshot 中，若所有 live frontier 在重叠 exact source 上不可执行，则必须存在一个
  未完成的 typed yield/release-fence transaction；`all blocked + no pending release` 状态数为 `0`。
- no-pressure c1/c4/c16/c32 路径创建 `PressureEpisode` 数为 `0`，跨队列/全历史 request scan 数为
  `0`，batch membership 不因启用动态 pressure 机制而改变。pressure path 的每次 recompute/yield 都
  必须释放本次失败 demand 涉及的 exact source 或产生明确 REJECT 诊断，盲目轮转数为 `0`。
- cancel/disconnect 后同容量下一请求成功率 `100%`。
- global release/capacity audit epoch 与每个 availability source generation 单调、无丢失唤醒；相关
  domain/slot 释放后 deferred request 成功重试 `100%`，无关 source 变化时 admission probe、prefill
  submit 和 device submit 增量均为 `0`。
- `global_claimed_bytes == sum(live static buffer bytes) + sum(live backing segment bytes)`；
  `domain.total_units == sum(installed segment usable units)`；`domain.used_units == sum(live logical
  extent units)`，在正常、失败、cancel、panic 和 quarantine 路径均成立 `100%`。
- 两个 plan/pool 共享一个 device 时重复认领同一物理余量的 case 数 `0`；逻辑 slice/page 重复
  计入 global bytes 的 case 数 `0`；metadata-only 动态 dispatch 成功数 `0`。
- Invocation scratch 被 Request/Sequence/Step lease 持有的 case 数 `0`；证明全序的 non-overlap
  scratch peak 与 plan liveness peak 完全相等 `100/100`，未证明互斥时按 conservative concurrent
  peak 验收；device fence 终态后 invocation lease balance 为 `0`。
- actual batch participant 数与 shape 不一致、空/重复/跨 plan/provider participant、leader-only
  capacity authority 的成功数均为 `0`；32 participant lane 的 batch scratch/device submit/fence 数
  均精确为 `1`。
- 外部构造 raw shape、少报 token/page、stale token/page generation、跨 plan/frame/node work
  authority、fit 小于 immediate、metric sum overflow、claim 后替换 work 的成功数均为 `0`；相同
  aggregate/different source fingerprint 区分率 `100%`，empty-demand shape 保留率 `100%`。
- 同 participant/frame/node 换 `BatchInvocationId` 的并发 prepare 成功数最多 `1`；overlap subset
  partial registry mutation 数 `0`。`DefinitelyNotSubmitted` 后 exact retry 成功率 `100%`，changed
  topology/work、stale retry authority、pending fence 或 Retired 后 retry 成功数均为 `0`。
- `SequenceSession` live identity mutation、非零 participant-flight terminalization、跨 session parent
  substitution、单 sequence terminalization drain shared lane 的成功数均为 `0`。
- N child sequence 的 Request extent 数保持 `1`，Sequence extent 数为 `N`；同一 frame 的 Step
  extent generation 保持不变，跨 frame generation 必须变化，错误层级 view 成功数 `0`。
- mixed-batch output 与逐请求 reference 一致 `100/100` seeds。
- resource final state 在所有 G04 L0/L1 scenario 中为 empty/expected cache `100%`；actual
  L2-L3 的同项验收归 G08。
- scheduler/replay determinism `100/100`。
- Qwen3.5 c32 historical resource fixtures `100%` 被本地 state-machine 或 replay gate 捕获，
  paid GPU 不再是第一次完整 invariant 测试。
- synthetic dense、MoE、hybrid 三类 program 的 setup/admission/state/finalize/cleanup 全部
  使用同一 lifecycle implementation，重复主循环数量 `0`。G08 在真实三模型迁移后重新
  执行同一 ownership analyzer；不再使用可通过改变分母操纵的“复用率”指标。

## 性能约束

- L1 reference workload scheduler bookkeeping 占 runtime wall time `<=5%`；真实 CUDA c32
  `<=2%` 由 G09 验收。
- disabled event path overhead `<=1%`。
- resource transaction 不增加每 token host allocation；steady decode host allocations/token=`0`；
  trace disabled 时 unchanged/irrelevant deferred scan 的 wait-condition clone/allocation 增量=`0`。
- steady admission/decode 的 device allocations/request=`0`；arena grow 只能作为独立、可观测的
  backing transaction，不能伪装成 request admission。
- plan build 的资源 descriptor/内存复杂度为 `O(graph)`，与 admission ceiling 无关；动态
  admission hot path 不扫描全部历史请求，scheduler bookkeeping 仍满足上述 wall-time 门。
- steady decode 的 frontier 更新由 batch completion 一次提交，不增加每 token shared-atomic RMW；
  pressure source index 只扫描相关 live cohort，不能扫描 waiting/prefill/decode 三套队列。
- capacity pressure 解除后下一 tick 回到与无压力路径相同的 stable batch builder；不得因 fairness
  bookkeeping 永久降低 decode cohort width。recompute/yield 次数、重算 token 数和被破坏 batch width
  必须进入 artifact，并与策略选择时的 cost evidence 对应。

## 产物与 PASS

```text
docs/release/runtime-vnext/0.8.0/g04-runtime-resources/
  state-model-report.json
  resource-fixtures/
  scheduler-replays/
  allocation-profile.json
  physical-logical-accounting.json
  backing-region-conformance.json
  qwen35-resource-kills.json
```

```text
FERRUM RUNTIME VNEXT G04 RUNTIME RESOURCES PASS: <out_dir>
FERRUM GATE vnext-g04 PASS: <out_dir>
```
