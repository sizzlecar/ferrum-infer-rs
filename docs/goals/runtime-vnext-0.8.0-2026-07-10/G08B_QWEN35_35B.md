# G08B: Qwen3.5-35B-A3B Hybrid-MoE 迁移

## 依赖与目标

- 依赖：G08A
- 下游：G08C
- 目标：在 M1 hybrid 基础上加入 256-expert/shared-expert MoE、GPTQ/Marlin 和高压资源路径。

## 必需交付

- CUDA official GPTQ-Int4 full product path。
- Metal Q4_K_S full product path，固定 32GB M1 Max，>=2 GiB 实测 headroom、swap growth 0，无 waiver。
- requested/effective concurrency 分离；CUDA client c32、typed active cap至少 16，并记录
  observed max-active；Metal required client c1/4/16、typed active floor `4`。CUDA/Metal 最高
  cell 的 eligible interval active duty-cycle 均须 `>=0.80`。
- recurrent + KV + scratch 多资源事务 fault grid。
- G00 legacy CUDA binary parity；Metal 使用 HF/CPU + same-GGUF llama.cpp new-lane reference。
- 删除全部 Qwen3.5 family legacy runner/factory/arch-named adapter，包括 G08A test-only adapter。

## 验收

- M2 CUDA/Metal C01-C21 `2/2 PASS`。
- Metal op/layer/full-vocab-logit/token reference 全部满足 MODEL_MATRIX 固定数值门，并绑定 checked-in
  tolerance blob/row；missing/post-hoc-widened tolerance 数量 `0`。
- G02 Qwen3.5 resource/output historical mutations kill `100%`。
- Qwen3.5 family legacy production/test adapter 数量 `0`。
- Qwen3.5 架构专属执行脚手架相对 G00 减少 `>=60%`。
- CUDA client c32/admission-cap 路径资源终态正确，OOM/livelock/leak `0`。
- G08 统一 performance smoke：CUDA `>=0.90x` G00 legacy，Metal `>=0.70x` same-host
  llama.cpp；两者都只作 diagnostic，完整正式门留给 G09。

```text
FERRUM RUNTIME VNEXT G08B QWEN35 35B A3B PASS: <out_dir>
FERRUM GATE vnext-g08b PASS: <out_dir>
```
