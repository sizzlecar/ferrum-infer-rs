# Runtime vNext v0.8.0 正确性矩阵收敛修订（2026-08-07）

## 状态与效力

- 状态：Active。
- 本修订只改变 R1 中跨模型重复用例的采样数量，优先于
  [`MODEL_MATRIX.md`](MODEL_MATRIX.md) 和
  [`RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md`](RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md)
  中要求每个主模型重复执行同一完整分母的条款。
- 三个主模型、CUDA/Metal 双后端、C01-C21 场景、产品入口、正确性 oracle、资源不变量、
  provider conformance 和 `production_legacy_selection_count=0` 均不删减。
- legacy baseline 的冻结 703/702/783/782 分母不变。本修订只作用于
  `g08-model-matrix-v1` candidate contract。
- 首次取得 R1 PASS 后，后续阶段修复采用累积证据和预定义影响范围回归；本文件中任何旧的
  source-SHA freshness 表述不得解释为“修改产品代码后从 R0 或完整 R1 重新开始”。具体计算规则以
  [`CHANGE_IMPACT_REGRESSION_PLAN_2026-08-12.md`](CHANGE_IMPACT_REGRESSION_PLAN_2026-08-12.md)
  为准。

## 决策

三个主模型不能互相替代：

| 模型 | 不可由其他模型覆盖的结构 |
|---|---|
| M1 Qwen3.5-4B | dense FFN、DeltaNet/recurrent state、稀疏 full attention、BF16 CUDA 主路径 |
| M2 Qwen3.5-35B-A3B | DeltaNet + full attention、routed + shared expert MoE、GPTQ/Marlin、最大资源压力 |
| M3 Qwen3-30B-A3B | 全层 full attention、routed-only MoE、Qwen3 soft thinking、GPTQ/Marlin 稳定控制组 |

因此不删除模型，而采用“公共协议全矩阵 + 架构差异矩阵”：

- M1 是公共产品/API canary，继续执行 CUDA `703`、Metal `702`。
- M2 只压缩跨模型重复采样，执行 CUDA `112`、Metal `111`。
- M3 只压缩跨模型重复采样，执行 CUDA `120`、Metal `119`。
- 六 lane 总分母从 `4375` 降为 `1867`，减少 `2508` cases（`57.3%`）。
- 已通过 M1 CUDA 后，剩余 CUDA 分母从 `1486` 降为 `232`，减少 `84.4%`。

## M2/M3 精确分母

| 场景 | M2 | M3 | 保留的硬覆盖 |
|---|---:|---:|---|
| C01 | 20 | 20 | config、template、special token、unknown fail-closed 各 5 |
| C02 | 4 | 4 | known-answer 与 natural EOS |
| C03 | 2 | 2 | multi-turn，逐组 3 rounds |
| C04 | 1 | 1 | `>=512` output tokens |
| C05/C06 | 4/4 | 4/4 | non-stream/stream 成对、唯一 `[DONE]`、usage、重组等价 |
| C07 | 2 | 2 | 两个隔离会话，各 5 turns |
| C08 | 6 | 6 | stop、natural EOS、max-tokens 各 2 |
| C09 | 6 | 6 | cancel、timeout、disconnect 各 2，释放后容量复用 |
| C10-C13 | 每项 4 | 每项 6 | no-thinking/thinking；M3 额外保留 soft-think/soft-no-think |
| C14 | 8 | 8 | required/type/additional-properties/enum x no-thinking/thinking |
| C15 | 4 | 4 | json-object x no-thinking/thinking |
| C16 | 5 | 5 | 五类 invalid request 各 1 |
| C17 | 6 | 6 | Chinese/emoji/combining x run/serve |
| C18 CUDA/Metal | 4/3 | 4/3 | CUDA c1/c4/c16/c32；Metal c1/c4/c16，资源 trace 全保留 |
| C19 | 10 | 10 | 五种 thinking mode x run/serve |
| C20 | 5 | 5 | image/data/video/mixed 显式拒绝 + text-array positive |
| C21 | 5 | 5 | run plain、serve stream、required tool、strict schema、json-object |

所有 case 必须 PASS；`known-fail`、`blocked`、skip、waiver、error、unexpected 均为 `0`。
M2/M3 的旧 703/702/783/782 candidate 重复矩阵转入 v0.8.1/0.9 nightly hardening，
不得被报告为 v0.8.0 已完成工作。

## C17 Unicode oracle 边界

C17 验证产品字节边界，不验证生成模型是否逐字服从复述提示。正式 PASS 必须同时证明：

- Unicode marker 逐字进入 `ferrum run` 的 user event 和 `ferrum serve` 的 HTTP request；
- `run` 的全部 `assistant_delta.raw_text_delta` 按连续 index 拼接后与最终 assistant content
  完全相等，每段 `utf8_bytes` 和最终 `raw_text_sha256` 均与真实 UTF-8 bytes 一致；
- `serve` 的 non-stream reference 与 streaming reconstruction 内容、reasoning、finish reason 和
  usage 完全相等；
- 逐字节读取 SSE 时确实跨越多字节 UTF-8 边界，重组 bytes SHA 一致，且 replacement character、
  mojibake、invalid UTF-8、malformed SSE 均为 0；
- 生成内容必须非空且实际包含至少一个多字节 Unicode scalar。

不得把“模型输出与提示 marker 完全相等”作为 engine/transport 正确性 oracle。M3 历史
`cff4c477` 和当前 `f0f61a17` 的 `c17-001` 都稳定将“中文正确”回答为“正确”；输入事件、
assistant delta、最终 content 与 SHA 均保持一致。这属于模型措辞行为，不是 Ferrum 丢失了
“中文”两个字。本修订后的预测是该 case 只有在上述真实字节链全部闭合时才从
`c17-contract-violation` 变为 PASS；任何乱码、截断、delta/content 不一致仍必须 REJECT。

## C19 thinking oracle 边界

C19 验证 reasoning 开关、final 分离和真实两轮 history，不把模型逐字服从提示当成运行时
正确性。两种状态分别执行：

- reasoning 开启时，两个 response 都必须有非空 reasoning，final 必须严格等于本轮 marker，
  且 `<think>` 标签和 reasoning 文本不得泄漏到 final；
- reasoning 关闭时，两个 response 都不得出现 reasoning 字段或 `<think>` 标签；普通 content
  可以包含模型给出的计算过程，但必须且只能包含一次本轮 marker，不能包含另一轮 marker；
- `run` 必须记录连续的两轮 user/assistant 事件和递增 history receipt；`serve` 第二个 request
  必须逐字段携带第一个 assistant message，包括 reasoning；硬开关必须覆盖 soft prompt，M3 的
  `/think`、`/no_think` 不得被错误转换为硬开关。

不得要求 no-thinking 模式把所有计算内容隐藏到不存在的 reasoning 通道。当前
`093d684f` M3 CUDA `c19-009` 正确关闭 reasoning、正确保持两轮历史并生成本轮 marker，但在
marker 前复述算式；这类输出只要满足上述边界即为 PASS。缺少 marker、混入另一轮 marker、
reasoning 模式错误或 history 不一致仍必须 REJECT。

## 失败后的执行策略

1. 首次失败只运行 exact case reproducer。
2. 在修复代码落地、任何回归命令启动前，由 validator 按版本化的通用 change-impact 规则计算
   affected case closure；输出必须列出 diff、规则 ID、受影响 case/lane/入口/后端/架构/模式和因果边。
3. exact case 通过后，只执行 closure 中尚未由当前修复验证的 affected scenario 和 architecture
   sentinel。只有调用链确实跨入口、后端、架构或阶段 contract，才扩大到该相邻范围。
4. closure 内所有规定 case PASS 后，该 failure class 关闭，原阶段资格恢复，并从已累积的下一阶段
   进度继续；不得退回 R0，也不得重启 M1 的 703/702、M2/M3 的 112/111/120/119 或其他完整 lane。
5. 首次取得 R1 PASS 仍要求本修订定义的完整 `1867` case 矩阵，且 R1 PASS 前不得进入 R2；本条不把
   后续阶段修复变成第二次完整 R1。
6. release candidate staged binary 复验仍使用目标要求的 R3 完整矩阵和 exact staged bytes，不恢复旧的
   跨模型重复分母，也不得用开发期 focused regression 缩减 R3。

## 证据复用边界

正式证据是按 case 和 contract domain 累积的，不按“当前 source SHA 是否等于 artifact SHA”整体
失效。首次 R1 PASS 后发生代码变化时，validator 必须计算从 baseline 到 current source 的实际 diff
到 case 的传递影响闭包：

- 闭包之外的历史 PASS case 保持有效，即使其 artifact 绑定的是当前 SHA 的祖先；聚合器必须保留其
  原始 source/binary identity，不能伪装成当前 SHA 的新执行。
- 闭包之内的历史 case 暂停用于当前阶段资格，必须在修复后的 source/binary 上按规定范围重跑；全部
  PASS 后恢复资格。若 closure 同时触及 R0 contract，只重验受影响的 R0 gate；若未触及，不得退回 R0。
- 影响面由机器可执行、版本化、可复用的语义规则计算，不能依赖某个 commit、SHA、blob 转换、日期、
  一次性 bridge 或人工写出的 reviewed-diff 白名单。人工说明可以补充因果理由，不能缩小 validator
  计算出的集合，也不能作为 waiver。
- 无法分类的 diff 必须 fail closed，并先补充通用映射规则及其正反 self-test；fail closed 的含义是
  阻止阶段继续，不是默认要求从零执行全部 R0/R1。规则补齐后由 validator 重新计算精确 closure。
- validator 必须拒绝漏掉真实 consumer 的映射，并证明：backend-local 不污染另一 backend，
  profile-only 不污染 profile-off/default correctness，无关模型/入口保持可复用，shared contract 则
  只扩展到实际消费者。
- `known-fail`、`blocked`、skip、waiver、error 和 unexpected 仍必须为 `0`；累积复用不是降低 case
  oracle，也不是把旧 failure 提升为 PASS。

任何旧的一次性 G02 roster bridge、特定文件 allowlist 或 exact blob 例外均由上述通用机制取代，
不得继续作为后续改动的验收依据。为深度认证首次完整 R1 的原始历史 artifact，validator 仍可按该
artifact 当时记录的 sealed bridge 重建其原始 provenance；这种历史重验只证明 baseline 本身未被
篡改，不能缩小 baseline 到 current 的影响闭包。最终 R3 仍必须对 exact staged bytes 执行 active amendments
规定的完整复验；祖先 source 的累积证据不能替代该最终 binary-level 签字。
