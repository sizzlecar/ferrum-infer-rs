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
2. exact case 通过后运行 affected scenario 和同架构 sentinel。
3. 只有达到新的 R1 milestone 才重新运行该 lane 的 112/111/120/119 矩阵。
4. 不因单 case 失败重启 M1 的 703/702 或其他 backend 的完整 lane。
5. release candidate staged binary 复验使用同一 R1 分层矩阵，不恢复旧的跨模型重复分母。

## 证据复用边界

M1 已通过工件只有在记录 SHA 是当前 SHA 的祖先，且中间变化严格限于本修订的文档与 R1
矩阵控制面文件时才可复用。任何 `crates/`、Cargo、模型锁、运行配置、产品场景 manifest 或生产
实现变化都会使工件 stale。M2/M3 工件原则上必须与当前 source 完全一致；唯一的祖先复用例外
是 diff 严格限于不参与 matrix 执行的 R1 控制面：`scripts/release/run_scenarios.py`、本修订文档、
`scripts/release/runtime_vnext_r1_product_correctness.py`，以及只记录 reviewed source SHA 的
`scripts/release/configs/runtime_vnext_g08a_source_contract.json`。该例外不覆盖任何 `crates/`、Cargo、
模型锁、运行配置、matrix runner、matrix scenario manifest 或生产实现变化，也不免除 Llama 自身
在当前 source 上重跑完整 `run`/`serve`/stream 三用例。validator 必须有正例接受上述因果隔离 diff，
并有反例拒绝任一产品源码 diff。

`scripts/release/runtime_vnext_g08a_same_history_collector.py` 的 bounded worker containment 只参与
G08A numerics，不参与 S2 或 R1 模型矩阵执行。R0 可跨该文件的单独 containment 变化消费内部所有
child 仍严格同 SHA 的 S2 聚合证据，但不得因此复用旧 numerics；R1 模型矩阵可跨该 exact 文件及
R0/R1 aggregator 的因果隔离变化。该例外不扩展为目录或相邻文件白名单，Llama 三用例仍必须在
最终 source SHA 上重跑。

### 2026-08-12 一次性 G02 roster bridge

从 clean source `05a5d2f8611ed3a3fedb5c69ff3ba11e533bc4c7` 到其紧接的 G02 roster
修复 checkpoint，只允许以下五个文件出现在 source diff 中：

- `scripts/release/runtime_vnext_g02_core.py`
- `scripts/release/runtime_vnext_s2_cuda_product_contract.py`
- `scripts/release/runtime_vnext_r0_core_closure.py`
- `scripts/release/runtime_vnext_r1_product_correctness.py`
- `docs/goals/runtime-vnext-0.8.0-2026-07-10/CORRECTNESS_ACCEPTANCE_AMENDMENT_2026-08-07.md`

其中 G02 文件必须是 Git blob
`38b832c95ecee833240a1477678fb5ce350f52fb` 到
`fa369a3ee52535ead59aefb4b3f675844feb09b8` 的精确转换：只补登记已经由
`c1faf845f821d60c8aab01542eaa58f6bf9d5900` 加入的
`legacy_reusable_memory_plan_wire_and_plan_hash_remain_stable` 测试，并增加该 12-test
精确 roster 的 self-test。其余三个 Python 文件只可实现和验证这次两域 source closure；本段文档
只记录该闭包。任一额外路径、不同 G02 blob、`crates/`、Cargo、model lock、runtime config、
产品场景 manifest、matrix runner 或生产实现变化都必须使 bridge fail closed，不得按文件名前缀或
相邻目录扩展白名单。

该 bridge 允许重新执行当前 source 的 G02 L0/L1，并让新的 S2、R0、R1 aggregate 对仍绑定
`05a5d2f8` 的未变产品 raw evidence 和已完成 matrix evidence 做深度重验；它不把旧 G02 failure
提升为 PASS，也不免除 G02、S2 和 R0 在当前 source 重新打印正式 PASS。Llama CUDA/Metal
`run`、`serve`、stream 三用例仍必须严格在最终 current source SHA 上重新执行，不能消费该 bridge。
