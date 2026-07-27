# G08A: Qwen3.5-4B Dense-Hybrid 迁移

## 依赖与目标

- 依赖：S1/S2 已有 M1 CUDA live G01/G03/G04/G05/G06 slice；S3 补齐 M1 Metal；不等待全量 G03-G07
- 下游：G08B
- 目标：用 M1 完成第一个主模型 CUDA -> Metal vNext 产品纵切。

## 必需交付

- 官方 config/weight schema -> dense-hybrid `ModelProgram`。
- CUDA BF16 run/serve/tools/schema/stream/recurrent-state/concurrency。
- Metal Q4_K_M op/layer HF/CPU reference、run/serve 和 client c1/4/16；CUDA/Metal 最高 cell 的
  active floor 分别为 `32/16`，eligible interval duty-cycle `>=0.80`。
- G00 若有可执行 legacy M1 lane则 parity；否则明确走 new-lane reference，不伪造 baseline。
- M1 legacy production entry、dense-only factory/flag 删除。
- shared Qwen3.5 legacy adapter 只允许 test build，写明 `sunset=G08B`。

## 验收

- M1 CUDA/Metal C01-C21 `2/2 PASS`，waiver `0`。
- Qwen3.5 Metal numerical reference 按 MODEL_MATRIX 的 op/layer/logit/token 数值门 PASS，并绑定
  checked-in `runtime_vnext_numerical_tolerances.json` blob/row SHA；artifact-local tolerance 数量 `0`。
- lifecycle 五类 ownership 由 shared runtime负责 `5/5`。
- M1 product binary 选择 legacy path 次数 `0`。
- M1 model-specific production files `<=8`、LOC `<=1500`（novel op provider 不计）。
- historical corpus 中适用于 dense/recurrent/product 的 mutation 全部被杀死。
- G08 统一 performance smoke：legacy PASS 时 `>=0.90x` legacy，否则 `>=0.70x` same-host
  vLLM/llama.cpp；该结果只作 diagnostic。

```text
FERRUM RUNTIME VNEXT G08A QWEN35 4B PASS: <out_dir>
FERRUM GATE vnext-g08a PASS: <out_dir>
```
