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
```

```text
FERRUM RUNTIME VNEXT G03 BACKEND OPS PASS: <out_dir>
FERRUM GATE vnext-g03 PASS: <out_dir>
```
