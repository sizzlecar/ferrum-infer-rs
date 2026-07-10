# G08: 三主模型迁移、Parity 与 Legacy 删除

## 状态与依赖

- 状态：Open
- 依赖：G03、G04、G05、G06、G07
- 内部依赖：G08A -> G08B -> G08C -> G08D；G08 只聚合四个未 stale PASS
- 下游：G09、G10

## 目标

用三个主模型证明 vNext 不只适合简单 dense decoder：先迁 CUDA，再迁 Metal；每个模型
parity 后立即删除对应 legacy 路径。最后处置所有其余 support row，使 release binary 只包含
vNext runtime。

## 迁移顺序

### M1: Qwen3.5-4B

1. CUDA BF16：作为第一个真实 vertical slice，跑通 config -> program -> plan -> runtime ->
   run/serve。
2. CUDA tools/schema/stream/recurrent-state/concurrency。
3. Metal GGUF：实现 Qwen3.5 dense hybrid providers、state 和 quant weight mapping。
4. 有可执行 legacy M1 lane 时完成 parity；没有时走 HF/CPU/external new-lane reference。
5. 切断 M1 legacy production entry；Qwen3.5 shared legacy 仅以 test-only adapter 保留到 G08B，
   并声明 `sunset=G08B`。

M1 用于证明 dense hybrid 不被错误绑定到 MoE。仅 config probe 或 toy reference 不算迁移。

### M2: Qwen3.5-35B-A3B

1. CUDA GPTQ：复用 M1 hybrid blocks，增加 routed/shared-expert MoE、Marlin 和高压资源计划。
2. 修复 c32 recurrent/KV/defer/rollback 历史失败；完整 correctness/concurrency gate。
3. Metal GGUF：同一 semantic program，Metal providers + GGUF weight mapping。
4. 拆除当前 18,239 行 `qwen35.rs` 单体中已经被 program/runtime/op 取代的执行脚手架。

### M3: Qwen3-30B-A3B

1. CUDA GPTQ：迁移成熟 full-attention MoE，作为共享 runtime 性能控制。
2. Metal GGUF：迁移现有稳定路径。
3. 验证 Qwen3.5 新 op 没有污染传统 MoE plan。
4. 删除旧 Qwen3Moe model-specific unified runner 和 capability branches。

## 每模型完成合同

每个模型子阶段必须在对应的 G08A/G08B/G08C 文档内完成：

- typed config/weight/program；
- CUDA/Metal plan snapshot；
- [`MODEL_MATRIX.md`](MODEL_MATRIX.md) correctness；
- 对 G00 legacy PASS lane 做 token、plan、resource parity；对 G00 BLOCKED/new lane 做
  HF/CPU op+layer reference、全模型 known-answer 和 external implementation gate；
- performance smoke，完整性能留给 G09；
- 对应 legacy code、flag、fallback、adapter 删除；
- support contract 和 README row 指向 vNext artifact。

此外，G02 historical corpus 必须在 vNext production mutation/revert-to-bug 上达到
`15/15 family` 和 `M/M concrete case` kill；仅复用 G02 synthetic analyzer fixture 不算完成。

G08 的 `performance smoke` 统一为低成本 diagnostic：random `64 input / 32 output`、c=1 和该
backend 最高 required client concurrency、`--fail-on-error --seed 9271 --n-repeats 3`、每 repeat
100 requests，baseline 后 candidate 各运行一次并各自 warmup 10 requests。它不使用
`--require-ci`、不产生 LCB，也不能支撑 performance/no-regression claim；legacy PASS lane 的
candidate median 须 `>=0.90x` legacy，新 lane 须 `>=0.70x` same-host external。正式
`ABBA-BAAB`、不回退和 80% 门全部在 G09 完成。

禁止先宣布模型迁移完成，再把旧路径删除留给以后。

## 代码规模与复用

- Qwen3.5 family 的架构专属执行脚手架相对 G00 inventory 减少 `>=60%`。
- setup、admission、state transition、finalize、cleanup 五类 lifecycle ownership `5/5`
  只属于 `ExecutionRuntime`；model hook allowlist 不得实现这五类步骤，重复实现数量 `0`。
- 新增只使用 existing ops 的 model provider：生产文件 `<=8`、生产 LOC `<=1500`。
- novel op 实现不计入模型 LOC，但必须归属 G03 contract/provider。
- 单个模型生产文件不得再次增长为 >5K LOC 单体；超过时必须按 config/weights/program/
  blocks/tests 拆分并保持单向依赖。

## 长尾处置

v0.8.0 不能同时发布 vNext 主模型和 legacy 长尾 runtime。当前 README/support manifest 中每个
family 必须二选一：

1. 迁移到 vNext，并有最少 L2 actual-model artifact；或
2. 从 v0.8.0 支持矩阵、alias 和 release claims 中明确移除。

Qwen3-Coder 与 DeepSeek-R1 补充 lane必须迁移，因为它们承担 agent/reasoning gate。Llama
8B-class 必须迁移以满足 release policy。其他 modality 可保留独立非 LLM executor，但不得
依赖已删除的 legacy decoder/runtime contract。

## 验收

- G08A/G08B/G08C/G08D 四个 PASS artifact 均存在、fresh 且来自 canonical `run_gate.py`。
- 三主模型 x 双后端 correctness matrix `6/6 PASS`。
- 主模型 required correctness scenario 总失败数 `0`。
- legacy 可比 lane token parity `20/20/model/backend`，generated token exception 数量 `0`；near-tie
  logits 只作 diagnostic。
- Qwen3.5 Metal reference numerical gate PASS，waiver `0`。
- lifecycle ownership `5/5` 位于共享 runtime，模型重复实现 `0`。
- Qwen3.5 架构专属脚手架 LOC 减少 `>=60%`。
- arch-named backend API `0`；model-specific scheduler/KV manager `0`。
- main-model legacy executor/factory/feature/env toggle/runner 数量 `0`。
- 全部 support rows 有 migrated 或 removed 决议 `100%`。
- release binary、test build 和 source tree 的 legacy implementation/symbol/toggle 数量 `0`；
  parity 只允许由外部 harness 启动 G00 冻结的独立 legacy binary。
- compatibility adapter 数量在 G08D 最终为 `0`。

## Artifact invalidation

M2/M3 如果修改 G01-G04 contract，必须 reopen 受影响 Goal，并自动 stale 之前 M1/M2 artifact。
不允许以“另一个模型还能跑”为理由继续迁移。

G08 的开发迁移 PASS 不能跨过 v0.8.0 release freeze 自动成为发布候选正确性证据。G10A 生成
唯一 `release_candidate_sha` 后必须执行：

```text
python3 scripts/release/run_gate.py vnext-g08-rc \
  --g10a <g10a-manifest> \
  --g08 <g08-manifest> \
  --out <external-out>
```

G08-RC 在 `release_candidate_sha` clean checkout 上重新执行 M1-M3 x CUDA/Metal C01-C21，不能
复用 G08-dev 的 case result。manifest 必须绑定六 lane product binary SHA、model lock、scenario
corpus、effective config 和 G10A manifest。其 Metal/CUDA candidate binary SHA 必须分别与 G09-RC
和 staged tarball binary 完全一致；任一 source/config/binary hash 不同即 stale。必需 line：

```text
FERRUM RUNTIME VNEXT G08 RELEASE CANDIDATE CORRECTNESS PASS: <out_dir>
FERRUM GATE vnext-g08-rc PASS: <out_dir>
```

## 产物与 PASS

```text
docs/release/runtime-vnext/0.8.0/g08-model-migration/
  qwen35-4b/
  qwen35-35b-a3b/
  qwen3-30b-a3b/
  supplemental/
  legacy-inventory-before.json
  legacy-inventory-after.json
  support-disposition.json
  reuse-report.json
```

子 Goal：

- [`G08A_QWEN35_4B.md`](G08A_QWEN35_4B.md)
- [`G08B_QWEN35_35B.md`](G08B_QWEN35_35B.md)
- [`G08C_QWEN3_30B.md`](G08C_QWEN3_30B.md)
- [`G08D_LEGACY_ZERO.md`](G08D_LEGACY_ZERO.md)

```text
FERRUM RUNTIME VNEXT G08 MODEL MIGRATION PASS: <out_dir>
FERRUM GATE vnext-g08 PASS: <out_dir>
```
