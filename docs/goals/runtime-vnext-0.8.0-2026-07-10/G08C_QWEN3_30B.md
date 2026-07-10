# G08C: Qwen3-30B-A3B 传统 MoE 迁移

## 依赖与目标

- 依赖：G08B
- 下游：G08D
- 目标：迁移成熟 full-attention MoE，证明 Qwen3.5 op/runtime 没有污染传统路径。

## 必需交付

- CUDA GPTQ/Marlin 与 Metal GGUF vNext product paths。
- CUDA c32 与 Metal c16 的 active floor 分别为 `32/16`，eligible interval duty-cycle `>=0.80`。
- legacy token/plan/resource parity 和对冻结 G00 legacy 执行 G08 统一 performance smoke。
- thinking hard/soft switch、tools/schema/stream、paged KV、continuous batching。
- 删除 Qwen3Moe model-specific unified runner、legacy factory、capability branches 和 adapters。

## 验收

- M3 CUDA/Metal C01-C21 `2/2 PASS`。
- G08 统一 performance smoke candidate median `>=0.90x` legacy；不计算 LCB，完整 ABBA
  no-regression 留给 G09。
- Qwen3.5-only operation 出现在 M3 plan 的数量 `0`。
- Qwen3Moe legacy adapter/runner 数量 `0`。
- M1/M2 artifact freshness 重新校验；共享 contract 变化时旧 artifact 已重跑。

```text
FERRUM RUNTIME VNEXT G08C QWEN3 30B A3B PASS: <out_dir>
FERRUM GATE vnext-g08c PASS: <out_dir>
```
