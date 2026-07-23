# G07: Build Graph、开发循环与 Native Operators

## 状态与依赖

- 状态：Open
- 总 PASS 依赖：G00P、live G03 catalog 和 S1 production build graph
- G07A 在 S1 后立即与 S2/S3 并行，S4 前达到开发反馈目标；G07B 随 live operation catalog/version
- 下游：S2-S7、G08-G10

## 目标

把模型开发、Rust runtime、core PTX 和重模板 native op 的编译失效域分开。普通开发不再
因为 Marlin/CUTLASS 模板和 release LTO 等待 30 分钟；正式构建仍保持优化和可验证资产。

## Checkpoint

- G07A：crate graph、dev/release profile、Cargo/nvcc invalidation 和 timing harness。
- G07B：与 G03 operation id/version 对齐的 native ABI、artifact manifest、resolver、link、
  runtime selection 和 negative fixtures。

G07 总 PASS 必须聚合 G07A/G07B；不能用编译时间 PASS 代替 ABI correctness，也不能用
native artifact PASS 代替增量构建证据。

G07 不再等待完整 G01/G02/G03 后启动。S1 一旦形成真实 model/runtime/provider/product dependency
graph，就开始测量并拆分 invalidation domain；否则到 S4 大模型迁移时仍会承受 30 分钟反馈循环。
普通 model program 修改运行 nvcc 的次数必须在 S4 前降为 `0`。

G07A/G07B 是 canonical DAG checkpoint：

```text
python3 scripts/release/run_gate.py vnext-g07a --g00 <g00-manifest> --g01 <g01-manifest> --out <external-out>
python3 scripts/release/run_gate.py vnext-g07b --g03 <g03-manifest> --g07a <g07a-manifest> --out <external-out>
python3 scripts/release/run_gate.py vnext-g07 --g07a <g07a-manifest> --g07b <g07b-manifest> --out <external-out>
```

G07A manifest 必须绑定 G00 build-input inventory、fixed build-host fingerprint、crate graph、timing
harness blob 和 raw samples。G07B 必须绑定 G03 operation catalog/version、native ABI、source lock、
resolver fixtures 和 G07A manifest。aggregate G07 逐字节消费两个 child manifest 并验证 source、
crate graph、operation catalog 和 ABI freshness；任何 child stale 或 catalog hash 分叉都必须失败。

## 目标 build graph

1. `ferrum-runtime-core` 类稳定 contract 不依赖具体 backend 实现。
2. CUDA/Metal provider 按 operation family 分 crate/module invalidation domain。
3. Marlin、MoE Marlin、FA2 等重型第三方实现走版本化 native operator ABI/artifact。
4. source build 独立 lane 产出 manifest、ABI、compiler flags、SM target、SHA256。
5. 普通 workspace 只解析/验证/链接 artifact，不把全部第三方 TU 当输入。
6. dev profile 与 official release profile 分离；dev binary 保持相同产品语义和 feature path。

## Native artifact 安全

- compatibility key：operator ABI、Ferrum native ABI、CUDA/runtime、SM、compiler、input hash。
- resolver fail-closed；checksum/ABI/SM 不匹配不得退回隐藏 source build 或慢实现。
- artifact 来源、构建命令、license、source revision 可追溯。
- CUDA tarball gate 检查无 Python/Torch/vLLM runtime linkage 和缺失 shared library。

## 增量编译场景

在同一固定 CUDA build host 各执行 5 次，记录 cold/warm、Cargo timings、nvcc TU、link：

| 场景 | 目标 p95 | 失效域 |
|---|---:|---|
| no-op | `<=30s` | 无重编 |
| Rust model leaf edit | `<=90s` | model + 必要 downstream，不能重编 native ops |
| runtime leaf edit | `<=90s` | runtime + product downstream |
| 单 core PTX edit | `<=120s` | 仅受影响 PTX + link |
| 单 Marlin/MoE TU | `<=5min` | 独立 source-build，只重编受影响 TU |
| clean official CUDA release | `<=15min` | 全部已解析 native artifacts + release LTO |

Metal-only 文件修改不得使 CUDA provider/native artifact dirty，反向同理。

五个样本的 p95 使用 nearest-rank 定义：排序后取 `ceil(0.95 * 5)=5`，即最慢样本；禁止使用插值
把最大值平滑掉。每个样本必须在独立的 clean timing worktree 中执行并记录以下边界：

- no-op：先完成一次相同 argv 的成功 warm build；计时从第二次相同 argv 启动到 binary
  `--version`/smoke 成功，输入内容变化为 `0`；
- Rust/PTX/TU leaf：使用 harness 锁定的 sentinel，在两个预先 SHA256 固定且语义等价的内容间
  切换；edit/fsync 在计时前完成，计时从 build argv 启动到新 binary smoke 成功；恢复文件不进入
  timed interval；
- clean release：先删除该 timing worktree 的 Cargo target，再从 official release argv 启动计时；
  下载好的、manifest/SHA 已验证的 native operator artifact cache允许保留，但其 cache key、hit/miss
  和路径必须保存；
- `sccache`/`ccache`、Cargo target、native artifact 和 linker cache 状态逐样本显式记录。未声明
  compiler cache、后台预编译或在计时前生成目标 object 的样本一律 REJECT；
- 每次保存 edit before/after SHA、Cargo argv/messages、实际 rustc/nvcc/link invocation、起止
  monotonic/wall time、return code、binary SHA256 和 smoke receipt。五次必须使用相同 host/power
  policy/compiler/toolchain；混合 cold/warm 样本后只报一个 p95 禁止通过。

### 2026-07-24 失效域诊断

clean source `3ac6b65a` 只修改了 CUDA provider 的 Rust replay shape 和一项 Rust test，
没有修改 CUDA/C++ TU、header、build script 或 feature set。retained RTX 4090 build host
上的 release test target 仍重新执行整套 Marlin/MoE native 编译，耗时 `16m57s`；随后同一
缓存上的正式 candidate release build 仍耗时 `4m54s`。这些是单次 diagnostic 数字，不是本文件
要求的五样本 p95，也不能形成 G07A PASS。

该样本已经证明当前 graph 未达到“Rust model/runtime leaf edit 不重编 native ops”的目标。
test target 与 product target 使用不同的 `ferrum-kernels` Cargo build-output identity，使
OUT_DIR 内的 native static-lib stamp 不能跨目标复用；在 native artifact 从 Cargo package
fingerprint 中解耦前，继续微调 `nvcc --threads` 不能解决该失效域。G07A 必须保存本次
`cuda-unit.log`、candidate `cargo.log`、两个 build output identity 和实际 nvcc invocation
作为 invalidation fixture，并用目标 graph 证明相同 Rust leaf edit 的 nvcc TU 数量为 `0`。

## 验收

- 普通仓库中继续 vendored 的大体量第三方 CUDA/C++ template build input 数量 `0`。
- G00 定义的 large third-party native source tree 在 `crates/`、`scripts/` 下完整 vendored
  副本数量 `0`；只允许 checked-in ABI/shim、patch、license、source revision 和 fixture，独立
  source-build lane 从锁定 upstream source 产出 artifact。移动/删除前必须先过 inventory gate。
- 每个 native op 有 source-build、resolve、link、runtime-select、negative fixture 和 artifact gate。
- 单 TU 修改实际重编 TU 数量 `1`；共享 header 真正变化除外，必须列出 dependency proof。
- 上表全部 p95 达标，5 次 raw timing 保存。
- CI/G0 build stdout+stderr 非空保存率 `100%`。
- release features 与 dev correctness features 的 semantic plan hash 相同；只允许优化/strip/LTO 差异。
- no-content touch 不触发 nvcc 数量 `0`。
- native artifact cache hit 不运行 compiler；cache miss 有明确原因。
- `cargo test --workspace --all-targets` 与 source validator 在新 crate graph 上通过。

## 清理要求

移动/删除 `crates/` 或 `scripts/` 文件前运行 inventory gate，并保留必要 compatibility wrapper。
大规模目录清理不得与 kernel correctness 修改放在同一 patch。

## 产物与 PASS

以下均为 canonical external `<out_dir>` 下的逻辑路径：

```text
g07a-build-iteration/
  manifest.json
  crate-graph.json
  build-timings/
  invalidation-report.json
g07b-native-operators/
  manifest.json
  native-operator-catalog.json
  resolver-fixtures/
  build-logs/
g07-build-native/
  manifest.json
  crate-graph.json
  native-operator-catalog.json
  build-timings/
  invalidation-report.json
  resolver-fixtures/
  build-logs/
```

```text
FERRUM RUNTIME VNEXT G07A BUILD ITERATION PASS: <out_dir>
FERRUM GATE vnext-g07a PASS: <out_dir>
FERRUM RUNTIME VNEXT G07B NATIVE OPERATORS PASS: <out_dir>
FERRUM GATE vnext-g07b PASS: <out_dir>
FERRUM RUNTIME VNEXT G07 BUILD NATIVE OPS PASS: <out_dir>
FERRUM GATE vnext-g07 PASS: <out_dir>
```
