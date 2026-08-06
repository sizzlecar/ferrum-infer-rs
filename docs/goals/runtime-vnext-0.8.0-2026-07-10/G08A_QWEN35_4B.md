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

Metal token parity 必须由 checked-in collector 生成，不能手写 JSON 或用自定义 Qwen 模板替代模型内置
chat template。collector 启动一个固定 `parallel=1`、固定线程上限的外部 `llama-server`，逐 case 调用
`/apply-template`、`/tokenize`、`/completion`，并逐个执行真实 `ferrum run`。20 个 prompt 的 prompt token
必须完全相同，每个 prompt 的 64 个 greedy output token 必须完全相同，共 `1280/1280`，waiver/exception
均为 `0`：

```text
python3 scripts/release/runtime_vnext_g08a_token_parity_collector.py \
  --ferrum-binary <ferrum> \
  --llama-server-binary <llama-server> \
  --llama-cpp-source <clean-llama.cpp-worktree> \
  --model <locked-Qwen3.5-4B-Q4_K_M.gguf> \
  --out <external-out>

FERRUM RUNTIME VNEXT G08A TOKEN PARITY COLLECTOR PASS: <external-out>
```

源码所有权与 legacy 删除先由独立、低成本 child gate 冻结：

```text
python3 scripts/release/run_gate.py vnext-g08a-source \
  --coupling-inventory <G00-coupling-inventory.json> \
  --out <external-out>
```

该 gate 必须使用 G00 同一 checked-in inventory analyzer 对冻结树和候选树计算函数级 logical LOC。
`<=1500 LOC` 指 G00 定义内的 model provider glue/执行脚手架；权重声明、真实数学 program/op 描述和
格式 parser 仍按 G00 规则单列排除。artifact 同时必须报告未排除的完整 Qwen3.5 family production
LOC/file 数，禁止用排除口径隐藏总体代码量。任一新增 family 文件或 provider 函数没有 checked-in
owner/reason/classification 时 fail closed。provider glue 必须按完整 provider production LOC 减去逐函数
review 的 parser/weights/math span 计算，顶层 type/const/impl 外壳不得漏算。源码 owner map 必须从真实
generic runtime symbol 证明 setup/admission/state-transition/finalize/cleanup `5/5` 只有一个共享 owner，
并验证 Qwen3.5 注册链最终进入共享 `VNextModelExecutor<R>`、family source 不持有 lifecycle authority、
`ModelFamilyProvider` 没有重新获得 lifecycle hook。独立 bounded dense/MoE/hybrid 测试只证明三种资源形状
执行同一状态轨迹并最终零占用，不允许由测试常量自报实现数量。

```text
FERRUM RUNTIME VNEXT G08A SOURCE OWNERSHIP PASS: <out_dir>
FERRUM GATE vnext-g08a-source PASS: <out_dir>
```

该 child PASS 不替代双后端 C01-C21、product binary legacy selection、完整 numerics、historical
production mutation、performance smoke 或最终 G08A PASS。

最终 G08A 只能由统一聚合门签发；七个输入都必须是同一 clean source SHA/tree 的 canonical
`run_gate.py` 外层 `gate.manifest.json`：source ownership、CUDA 703-case、Metal 702-case、完整
numerics、S2 product contract（含 `H02.1/H12.1-H12.4` 五个历史资源问题的 production
tests/replays）、以及 M1 CUDA/Metal 各一份 G08 performance smoke。CUDA 与 Metal performance
artifact 必须独立，不能用一份 backend 结果重复占位。

```text
python3 scripts/release/run_gate.py vnext-g08a \
  --g08a-source <vnext-g08a-source/gate.manifest.json> \
  --g08a-cuda <vnext-g08a-cuda/gate.manifest.json> \
  --g08a-metal <vnext-g08a-metal/gate.manifest.json> \
  --g08a-numerics <vnext-g08a-numerics/gate.manifest.json> \
  --g08a-s2 <vnext-s2/gate.manifest.json> \
  --g08a-cuda-performance <m1-cuda-performance/gate.manifest.json> \
  --g08a-metal-performance <m1-metal-performance/gate.manifest.json> \
  --out <external-out>
```

聚合器必须重新校验每个外层/child manifest、child stdout PASS、validation/input hash、source
identity、case denominator、waiver、C18 active floor/duty-cycle、numerical tolerance/token parity、
历史问题 denominator 和 performance ratio；matrix、numerics、performance 必须回到原始 artifact
重新执行 canonical validator，performance 阈值固定为 legacy `0.90` / external `0.70`。仅凭手写
summary、降低 artifact 自报阈值、伪造 delegated command/receipt 或使用旧 SHA 均不得通过。

```text
FERRUM RUNTIME VNEXT G08A QWEN35 4B PASS: <out_dir>
FERRUM GATE vnext-g08a PASS: <out_dir>
```
