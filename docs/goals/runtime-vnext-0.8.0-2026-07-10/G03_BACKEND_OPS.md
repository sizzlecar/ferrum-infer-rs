# G03: Operation Contracts 与 Backend Providers

## 状态与依赖

- 状态：Open
- 依赖：S1 从 Qwen3.5-4B CUDA production slice 提取最小 live catalog；S3-S5 随模型扩展；full G03 在 S6 前
- 下游：S1-S6、G04、G08-G10

## 目标

用细粒度、版本化 operation contract 替换架构语义化的 `Backend` 大 trait。保留已验证
CUDA/Metal kernel 实现，但所有调用都通过统一 contract、capability catalog 和 conformance。

operation catalog 不再在 production caller 之前一次性穷举并冻结。S1 只提取 Qwen3.5-4B CUDA
实际需要的通用 op 和 provider；S3 增加同 semantic fixture 的 Metal provider；S4/S5 仅在模型确实
需要时增加 MoE/Marlin/full-attention 等 op。novel op 必须同时具有 CPU oracle、目标 backend
provider、negative fixture 和 live model consumer，planner/runtime 主循环改动仍为 `0`。

## Operation family

至少覆盖：

- allocation/copy/command/sync 基础设备能力；
- embedding、RMSNorm/LayerNorm、activation、elementwise/gating；
- dense linear、GGUF quant linear、GPTQ/Marlin linear；
- full/paged/varlen attention、RoPE/QK norm；
- Gated DeltaNet prepare/update/decode/prefill；
- dense FFN、routed MoE、shared expert、expert dispatch/combine；
- logits、sampling 所需 tensor primitives；
- graph/capture provider，但 graph 是实现能力而不是模型语义。

## Contract 规则

- operation name 描述数学/数据语义，禁止包含模型名。
- 每个 op 明确 input/output aliasing、dtype、shape、stride/layout、workspace、stream ordering。
- 每个 supported backend 有 capability version；planner 只选择完整满足 contract 的 provider。
- fallback 是 planner 中可见的另一 provider，不允许 backend method 内部静默切 host。
- provider 错误必须保留 operation/node/request identity。
- CPU oracle 默认 FP32；量化 op 保存 dequant/reference 和误差预算来源。
- CUDA 与 Metal 可使用不同 kernel，但消费相同 semantic fixture。
- provider 必须声明版本化 execution semantics：同 runtime repeatability，以及是否承诺
  bitwise eager/replay equivalence；缺字段、未知版本或 fingerprint 不一致均 fail-closed。
- policy 的 determinism requirement 参与 provider compatibility；要求 replay equivalence 时，
  eager-only provider 必须在 plan build 前以 typed rejection 退出。
- execution semantics 必须进入 plan hash、compiled node/wave identity、runtime binding、
  receipt/event/profile；provider 返回的 reusable topology 不得扩大 descriptor 的声明。

## Determinism Conformance

同实现确定性与 CPU/reference 数值正确性是两个正交 contract：

- same-runtime determinism 对 raw bytes 使用 exact equality，不读取 numerical tolerance catalog；
- oracle parity 使用 checked-in tolerance row，不允许反向放宽 determinism；
- replay-ineligible provider 仍必须通过 eager/eager；replay-equivalent provider 还必须通过
  replay/replay 和 eager/replay；
- RNG、initial state、workspace 初始化、logical input 或 immutable binding 任一不同的样本不得
  被错误配对为 determinism case。

CUDA conformance runner 必须从 live catalog 和 resolved model plans 生成 provider coverage，
并复用 G02 定义的 artifact schema。新增 provider 或修改 implementation fingerprint 后，没有
当前硬件 proof 时可以编译和运行 focused source tests，但 G03、对应模型 correctness、G09 和
G10 必须保持未完成，不能依靠 descriptor 自声明形成 PASS。

生产启动期的 executable inventory 检查只可命名为 preparation/inventory validation；它证明
capture/upload/residency 没漂移，不得命名或记录为 numerical replay validation。正式数值证明
由独立 CUDA determinism gate 给出，避免把昂贵 readback 放进默认热路径。

## Qwen3.5 重点

把当前 Qwen3.5 架构命名方法拆成通用 op：indexed recurrent state、packed GDN
prepare、recurrent prefill/decode、attention/token gate、partial RoPE、shared-expert MoE。
拆分后通用 trait 中 `qwen35` 字符串出现次数必须为 `0`。

## Conformance

- shape grid：batch、token、heads、head_dim、experts、top-k、dtype、contiguous/strided。
- boundary：0/1 token、max supported、unaligned、reallocation、partial final chunk。
- backend parity：CPU oracle vs CUDA、CPU oracle vs Metal。
- lifecycle：同 stream、cross stream、cancel、provider error、workspace resize。
- numerical tolerance 在每个 op manifest 固定，validator 禁止实现自行放宽。

所有正式 tolerance 的唯一机器可读来源是 checked-in
`scripts/release/configs/runtime_vnext_numerical_tolerances.json`。每行必须包含 `tolerance_id`、
operation/schema version、checkpoint kind、dtype、quant format、shape domain、oracle identity、
cosine/relative-L2/absolute bound、依据、owner 和 review commit。G03 validator 按 Git blob SHA
加载；provider、runner 和 artifact 只能引用 `tolerance_id + row fingerprint`，不能嵌入覆盖值。
Qwen3.5 Metal 的最低数值门以 MODEL_MATRIX 7.2 为准；任何更宽 row 必须在 G03 阶段 hard fail。

## 验收

- vNext `DeviceRuntime`、operation contracts 和 providers 中架构命名方法为 `0`。
- legacy 架构命名 methods 必须全部进入 `legacy_adapter_inventory.json`，并且只能单向调用
  vNext op；vNext op 反向依赖 legacy 数量 `0`。全局 legacy 零值由 G08 验收。
- vNext model/runtime path 中未批准 backend `cfg` 为 `0`；全仓相对 G00 减少 `>=80%`
  的目标移至 G08，G03 期间新增 legacy cfg 为 `0`。
- supported op conformance cell 通过率 `100%`。
- same-runtime determinism coverage：所有 supported provider eager/eager `100%`；所有
  replay-equivalent CUDA provider 的 eager/eager、replay/replay、eager/replay `100%`，
  mismatch/waiver/skip `0`。
- 三个主模型 CUDA resolved plans 的 replay-equivalent provider proof coverage `100%`；artifact
  中 provider implementation fingerprint 与 plan/catalog/binary 不一致数量 `0`。
- checked-in numerical tolerance catalog coverage `100%`，missing/ambiguous/unowned row、artifact-local
  override 和 post-hoc widening 数量均为 `0`。
- unsupported cell 在 plan build 时 fail `100%`，进入 kernel 后才失败数量 `0`。
- host fallback 未记录数量 `0`。
- 每个 op 至少有 1 个 negative/fault fixture。
- CUDA/Metal 相同 semantic fixture 覆盖率 `100%`；backend-only fixture 不得替代。
- disabled profiling 下 provider dispatch overhead：GPU op `<0.5%` wall time；CPU tiny-op
  microbench `<2%`，否则改用 monomorphized fast path并保留同一 contract。
- novel-op 扩展演练不修改 planner/runtime 主循环。

## 迁移与删除

compat layer 只能从 legacy method 调新 op，禁止新 op 反向调用 legacy Backend。G03 冻结
adapter inventory；数量从此只能单调下降。每个模型在 G08 子阶段切换 production entry 时，
同一子阶段删除它不再需要的 legacy method、default impl 和 `supports_*`。G03 不为满足零值
而提前迁移 G08 的全部模型 call sites。

## 2026-07-25 Mixed Execution-Weight ABI Checkpoint

| SHA | 结果 | G03 事实 |
|---|---|---|
| `61398b66` | CUDA build REJECT | 引入 capability-driven Marlin FP8 materializer、严格 CUDA component resolver 和 eligible GDN projection wiring；没有模型名、GPU 名或环境变量分支 |
| `cc4f8130` | build PASS，plan compile REJECT | 修复两个 CUDA 编译错误；暴露 channelwise grouping 错把矩阵 K 编进稳定 quantization ABI |
| `bfdbf5db` | build PASS，static initialization REJECT | 用 `WholeAxis`/fixed grouping 表达 shape-relative quantization；暴露 mixed schema 中 unchanged component 未由 materializer 返回 |
| `0c9a2c31` | build PASS，首个 `ferrum run` REJECT | materializer 显式保留 unchanged payload；暴露 provider 把 enclosing schema format 错当 component ABI |
| `0b72bab2` | 窄 CUDA smoke/profile/c1 KEEP，后续 correctness REJECT | enclosing `schema_format_id` 降为 crate-private planning key；CUDA/Metal provider 只按 component physical layout/encoding 校验，但默认选择仍把 kernel capability 错当作 F16 -> FP8 数值授权 |
| `7bc46122` | C03 PASS，C17 REJECT | deterministic Unicode 期望 `中文正确`，实际只生成 `正确`；transport UTF-8、进程和资源均正常，失败属于数值语义回归 |
| `5149bbfb` | source gate PASS，focused C17 KEEP | materializer descriptor 增加 `Exact`/`Approximate`；compiler 默认只选 exact，Marlin FP8 被标记 approximate；C17 恢复 token `99986,97901` |
| `883ee9e0` | source/replay/real-Metal numerics PASS | schema v6 把 source F16 QKVZ+BA byte-exact 冷打包为 QKVZBA；CUDA/Metal 各执行一个精确 projection，不改变 source/compute/accumulation dtype |
| `557cdcf5` | focused CUDA correctness/topology PASS，formal G09 REJECT | 9-case current-HEAD product correctness KEEP；GDN topology 为 `4,500` projection GEMV、`300` dispatch/correlation，decode device-duration sum `7.6973 ms`；c1 `73.3800 < 76.1583 tok/s`，不能形成 G03/G08B/G09 canonical PASS |

该 checkpoint 只证明 G03 的 schema/container、component ABI 和 materializer fidelity
边界已在源码层收敛；v6 数值目录保留 v4/v5 历史并新增 2 个 operation、4 个 state
和 1 个真实 layer contract，阈值未放宽；
CUDA topology 的 `9,000`-instance cuBLAS kernel-name aggregate 包含全模型多个 projection
shape，不能当作 GDN-only 计数；typed operation trace 与持久 CUDA graph node shape 共同确认
GDN 自身为 `2,250` QKVZBA 加 `2,250` output GEMV。它没有产生 `vnext-g03`
canonical PASS。完整 CUDA/Metal conformance、numerical catalog
和 dispatch-overhead 仍未完成，G03 状态保持 Open。

## 产物与 PASS

```text
docs/release/runtime-vnext/0.8.0/g03-backend-ops/
  operation-catalog.json
  provider-capabilities.json
  conformance/
  numerical-tolerances.json
  numerical-tolerance-catalog-binding.json
  boundary-audit.json
  legacy-adapter-inventory.json
  dispatch-overhead.json
  cuda-determinism/
```

```text
FERRUM RUNTIME VNEXT G03 BACKEND OPS PASS: <out_dir>
FERRUM GATE vnext-g03 PASS: <out_dir>
```
