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

### 2026-07-28 CUDA correctness build source checkpoint

clean commit `9b318ea9` 建立了下一次 M2 CUDA exact replay 的 bounded 开发构建路径：

- `cuda-correctness` 继承 release 的 `opt-level=3` 和正式 features，只关闭 release LTO、
  放宽 codegen units、启用 incremental 并保留符号；这些差异不能改变产品语义。
- core PTX、Marlin、MoE Marlin 和 paged-attention archive 通过
  `ferrum-native-ops::NativeBuildArtifactCache` 在 profile/OUT_DIR 间按内容地址复用。
  payload 与 manifest 均校验 SHA256；复制使用同一打开文件的 copy+reread 验证，restore
  在发布 destination 前再次核对实际复制内容。
- compatibility key 绑定 source/header SHA256、SM、CUDA header SHA256、
  `nvcc`/host compiler/archiver identity、`TARGET/HOST` 和影响 nvcc 的 ambient flags。
  `quant_utils_stub.cuh` 已补入 paged-attention dependency closure；修改该 header 必须使
  Cargo 和 shared cache 同时失效。
- publish 使用 OS advisory lock；进程被 deadline kill、写锁信息失败或留下旧 lock 文件
  后，kernel cache 不会永久进入 30 秒超时循环。
- correctness build 会先强制 ferrum 非 fresh relink 和 `ferrum-kernels` build-script
  重跑，要求每个 native artifact 都有 cache evidence，发现任意 nvcc TU 重编即 REJECT。
  inventory 只打印 `IMPORT INVENTORY READY`，可执行产物只打印 `BINARY READY`，两者都不是
  PASS。
- semantic validator 必须同时绑定 build manifest、execution manifest、focused
  `c13-022` report、实际执行 binary SHA256 和 scheduler trace；只有 trace 的
  `vnext.plan_built` hash 精确匹配 reference 才打印
  `FERRUM CUDA CORRECTNESS SEMANTIC TRACE PASS`。

本地 staged evidence：

| Gate | Result | Artifact |
|---|---|---|
| native cache focused tests | `15/15 PASS` | `/Users/chejinxuan/ferrum-artifacts/runtime-vnext-g07-cache-local-20260728/native-ops-test-r4/bounded.receipt.json` |
| `ferrum-kernels --no-default-features` check | PASS | `/Users/chejinxuan/ferrum-artifacts/runtime-vnext-g07-cache-local-20260728/kernels-check-r2/bounded.receipt.json` |
| `cuda-correctness` profile check | PASS | `/Users/chejinxuan/ferrum-artifacts/runtime-vnext-g07-cache-local-20260728/profile-check-r2/bounded.receipt.json` |
| build/semantic validator self-test | PASS | `FERRUM CUDA CORRECTNESS BUILD SELFTEST PASS` |

这不是 G07A PASS：固定 CUDA host 上的真实 `BINARY READY`、release/dev semantic-plan
equivalence、五样本增量 p95、workspace source gate 和 canonical G07A validator 仍未完成。
下一 paid lane 的 stop condition 是 native rebuild、build deadline、semantic hash mismatch
或 exact `c13-022` 结果中的任意一个；在 exact case 通过前不运行 suffix 或 703 full。

### 2026-07-28 Replay leaf build diagnostic

Clean source `f9bb4070939eb4f318308b7446602389f89d0272` only changed CUDA replay
Rust code and a source contract. Local bounded validation passed the replay contract
`10/10` and the typed timing-mode unit `1/1`.

The retained RTX 4090 build host then produced two distinct bounded build REJECTs:

- the first release retry omitted the cached build's `FERRUM_NVCC_THREADS=4`, used
  the old default `0`, invalidated the native signature, entered vLLM Marlin nvcc,
  and hit the `480.020663s` deadline; peak usage was 11 processes and 43 group
  threads, with no resource violation and complete process-group cleanup;
- the corrected retry kept `FERRUM_NVCC_THREADS=4` and observed `nvcc=0`, proving
  the existing native archive remained reusable, but release LTO still exceeded the
  separate 300-second development deadline.

These results do not reject the replay Rust change and do not form a CUDA feature
PASS. They prove that a correctness iteration must use the no-LTO
`cuda-correctness` profile and a recorded native-worker policy rather than a manual
release command. The follow-up source change makes `--nvcc-threads` a typed
correctness-builder argument and manifest field, defaults it to `4`, and rejects
values outside `[1,8]` before compiler startup. `build.rs` and the legacy FA2
diagnostic use the same bounded default. Local builder self-test and
`ferrum-kernels --no-default-features` check pass; real CUDA `BINARY READY` remains
the next stop condition.

### 2026-07-28 Correctness-profile native inventory REJECT

Clean source `60efb84c3d234c8b77ee25154f650d8ab3a94ffa` used the typed
`cuda-correctness` builder on retained RTX 4090 instance `46083877`. Plan-only
found every expected Ferrum PTX/archive filename and printed inventory readiness,
but the bounded build exposed two independent contract failures:

- `static.marlin` existed in the release OUT_DIR, but its stamp was not compatible
  with the exact correctness-profile signature. The build silently fell through
  to nvcc, rebuilt it in `3838ms`, and published a new cache entry.
- `candle-kernels 0.9.2` is outside the Ferrum native inventory. Its non-fresh
  build script unconditionally compiled Candle PTX and `libmoe.a`, while the old
  parser counted only Ferrum `[cuda-build-summary]` events.

The lane was stopped at the declared first-native-recompile condition after
`164.682216s`; the bounded receipt recorded peaks of 49 processes, 102
process-group threads, and 33 threads in one process, with no resource violation
and complete process-group cleanup. Decision:
`REJECT/third_party_candle_and_static_marlin_native_compile_escaped_plan_inventory`.

The follow-up makes correctness builds fail closed:

- `FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only` is a typed build policy. Any
  exact Ferrum artifact miss now emits a rejected build summary and fails before
  starting its compiler; the default source-build policy remains `allow`.
- the validator records `native-build-signal.json` even when Cargo fails and
  counts actual execution of known unmanaged CUDA build scripts, including
  Candle, in addition to Ferrum summaries and visible nvcc commands;
- inventory readiness now means candidate files exist. Exact compatibility is
  resolved by the same Rust build-script signature implementation under the
  cache-only policy, rather than duplicated in Python.

The downloaded diagnostic bundle is independently SHA256- and zstd-verified at
`/Users/chejinxuan/ferrum-artifacts/github/cuda-g06-g07-diagnostics-60efb84c-20260728/verification.json`.
The GitHub draft-release asset is
`cuda-g06-g07-diagnostics-60efb84c.tar.zst` (asset `492570669`, SHA256
`2267c0a438aeef2190f85bc9be32a4dcc24c2e8ebc1f9e9a136bb47d58d849dd`).
Vast instances `46083877` and `45897840` are both `stopped/exited`, with
`potentially_billable=[]`.

This is not G07A or G07B PASS. A warm target can now prove that a Rust-only edit
does not invoke native compilers without allowing a hidden fallback, but the cold
graph still contains Candle's source-building CUDA dependency. G07B must either
replace that dependency with a signed Candle PTX/`libmoe.a` native artifact
resolver or remove Candle CUDA from the vNext production feature graph; the
five-sample timing matrix and canonical G07 validators remain outstanding.

### 2026-07-28 Release plan reference provenance checkpoint

Clean source `645a43717a49032c5a2d8fa57ee0f9f2d9f9c671` reused the retained native
cache and produced a runnable `cuda-correctness` binary in approximately
`20.1s` with no native compiler execution. Exact product `c13-022` printed
focused `KEEP`; its scheduler trace contained one canonical plan hash:
`96efe5ea30955542290d58ea76d4768a46c79886391cf06ffb538e16b726b586`.
The previous semantic check REJECTed only because the build plan still embedded
the old release hash
`54963e9ddc468d44eaf72227c603a0f64d19e1f151de58e41fa33fdd402cc09d`.

Source and trace audit classified this as a stale oracle, not nondeterministic
execution. Commit `f9bb4070` intentionally changed CUDA runtime/replay source
covered by the implementation fingerprint; effective config, model lock,
operation set and provider selection did not drift. A plan identity that includes
implementation provenance must change in that situation, so a manually carried
hash cannot prove release/development equivalence.

Clean pushed commit `fae4f6e777a2b6538f9382af9aa74ee8196ca21c` replaces the naked hash
with a portable release-plan reference artifact. Capture and validation bind:

- clean source commit/tree and ancestry policy;
- exact official release command, release build receipt and binary SHA256;
- hardware identity, model revision, model-file lock and models-lock digest;
- typed and actual effective configs;
- focused `c13-022` product report and actual vNext scheduler trace.

The correctness build imports the complete reference under its own artifact
root and revalidates every referenced file before compilation. Semantic
validation consumes only that build-bound reference. Reference ancestry permits
an unrelated newer candidate, but the candidate product trace must still match
the recorded plan identity exactly; implementation drift therefore fails closed.
The module self-test rejects a structurally consistent forged plan identity and
a forged debug build receipt even after the attacker updates the enclosing file
reference.

The complete scenario self-test printed
`FERRUM RUNTIME VNEXT G00 SCENARIOS SELFTEST PASS` under bounded execution in
`177.685141s`; peaks were 3 processes, 34 process-group threads and 17 threads
in one process, with zero violation and complete cleanup. The same change adds
an independent C18 client worker cap of 32 before the first thread spawn.

This remains a source/test checkpoint. G07A still requires one same-clean-SHA,
same-hardware release-reference plus correctness-profile product comparison,
followed by the five-sample build timing matrix and canonical G07 validator.
G07B's cold Candle native dependency remains open.

### 2026-07-28 Candle CUDA boundary and scheduler-neutral native cache identity

Clean pushed source `ecd6d1e8739467ab290bdb64df7a12e6a9c2cde6`, tree
`8e363e3b0efa90f11ffaf967f069a9ac00b77584`, closes the cold Candle CUDA
dependency from the official vNext CUDA feature graph:

- the official `cuda,vllm-moe-marlin,vllm-paged-attn-v2` graph resolves
  `candle-core` and `candle-nn` with CPU `default` only, and does not resolve
  `candle-kernels`;
- an explicit `candle-cuda-compat` feature preserves the legacy CUDA tensor
  modules and proves that Candle CUDA can only re-enter through an intentional
  compatibility request;
- the official cudarc feature set declares `std,dynamic-linking` directly
  rather than relying on Candle feature unification to select a link strategy;
- the boundary validator printed
  `FERRUM RUNTIME VNEXT CUDA CANDLE BOUNDARY PASS: /workspace/ferrum-artifacts/runtime-vnext-g07-candle-boundary-ecd6d1e8-20260728/dependency-graph-r2`.

The first real official build then found that `FERRUM_NVCC_THREADS` was part of
the static-library content identity. That was incorrect: nvcc worker count is a
scheduler policy, not an ABI, source, flag, or output-content input. The final
implementation:

- removes worker count from the canonical Marlin, MoE Marlin and paged-attention
  identities while continuing to pass bounded `--threads` to nvcc;
- accepts only enumerated historical full content/toolchain signatures for
  shared-cache promotion;
- keeps weak legacy metadata/stamp signatures confined to explicitly configured
  import directories;
- restores and verifies the historical payload before publishing it under the
  canonical identity, and requires promoted SHA256 and size to remain exact.

On retained instance `45897840`, exactly one RTX 4090 with driver `580.126.20`
and `24564 MiB`, the bounded official command was:

```text
env CARGO_TARGET_DIR=/workspace/ferrum-infer-rs/target \
  CARGO_BUILD_JOBS=4 CUDA_COMPUTE_CAP=89 FERRUM_NVCC_THREADS=4 \
  FERRUM_CUDA_NATIVE_BUILD_CACHE=/workspace/ferrum-native-build-cache \
  FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only \
  cargo build --release --locked --jobs 4 -p ferrum-cli --bin ferrum \
  --features cuda,vllm-moe-marlin,vllm-paged-attn-v2
```

It completed with `rc=0` in `471.000627s` (`7m50s`). This is below the
`15min` ceiling but retained the Cargo target, so it is an official-feature
incremental release build and cannot count as a clean-release timing sample.
The bounded receipt observed peaks of `5` processes, `35` process-group threads
and `17` threads in one process, no violation, and complete process-group
cleanup. All three heavy archives printed `status=promoted` followed by
`reason=promoted-compatible-native-build-cache`; actual native compiler
executions were `0`. The binary printed `ferrum 0.7.7`, had SHA256
`a6e5059c7cab467ea8cf79b37beb102644a19dea8267d76b038ce63531b11a11`,
and its dynamic dependency scan contained CUDA/cuBLAS but no Python, Torch or
vLLM runtime dependency.

The exact dependency and build evidence is saved in GitHub draft-release asset
`cuda-g07-candle-boundary-ecd6d1e8.tar.zst` (asset `492770357`, asset SHA256
`2fed9d75c9535b5437210c42d8a7a50fb3670cc9ee9e68ba7b5ffd0ee61af5d0`) and
was downloaded and SHA256/zstd verified at
`/Users/chejinxuan/ferrum-artifacts/github/cuda-g07-candle-boundary-ecd6d1e8-20260728`.
After evidence capture, instance `45897840` reached
`cur_state=stopped`, `actual_status=exited`; account inventory reported
`potentially_billable=[]`.

This proves that the official vNext CUDA **GPU execution graph** no longer
depends on Candle CUDA; it does not prove that the full product binary is
Candle-free. `vnext_executor` still creates CPU Candle tensors at the product
token/logit adapter boundary, and Metal, GGUF, embedding, audio and multimodal
implementations still use Candle. Replacing those host/API adapters is a
separate dependency cleanup and must not be confused with CUDA kernel
execution. This checkpoint is also one release-build sample, not the required
five-sample p95. G07 remains Open pending dev/release semantic equivalence,
the complete five-sample invalidation matrix, workspace source gate and
canonical G07A/G07B/G07 aggregate validators.

### 2026-07-29 CUDA release/correctness plan-equivalence checkpoint

Clean release-reference source `5e2aaaed73984d77951d68014b919fdcc3e84bd0`,
tree `5ff26a23b51c0161311a16f0cf4265da40397121`, ran the exact official
`cuda,vllm-moe-marlin,vllm-paged-attn-v2` release build on Vast instance
`46127509`, one RTX 4090 with driver `580.142` and `24564 MiB`. The bounded
build passed in `448.366051s`; peaks were 7 processes, 35 process-group
threads and 17 threads in one process, and cleanup confirmed the process group
was gone. No `nvcc` or `ptxas` process ran. The release binary SHA256 was
`e70114ed09c710eede78415e7f82c20a355ec04cc64ceff6f6e5089f6db21210`.

The release binary then ran exact product `c13-022` through `ferrum serve`.
It printed focused `KEEP` in `104.948784s`; the scheduler trace contained 17
`vnext.plan_built` events with one plan hash,
`96efe5ea30955542290d58ea76d4768a46c79886391cf06ffb538e16b726b586`.
The release reference bound the official build receipt and binary, model and
file locks, typed and actual effective configs, focused report, hardware
identity and the complete scheduler trace.

This lane also exposed two fail-closed evidence bugs instead of hiding them:

- fixed-architecture Marlin always compiles `compute_80` PTX, but its cache
  identity included the reported device capability. The canonical identity now
  excludes that phantom input and permits only an exact one-line historical
  signature migration; source, toolchain and payload SHA checks remain strict.
- the semantic validator compared the whole typed-config evidence envelope,
  which necessarily contains different release and correctness binary SHA
  values. It now compares the exact non-empty `typed_effective_config`
  sub-contract while retaining independent SHA binding for the release binary,
  correctness build binary and correctness execution binary. A changed device
  config remains a negative fixture.

Clean current source `775758780baf66c3a508f8b07237035483c97410`,
tree `135ac8a1dffa9add13b918478d46a87ec2304288`, then used the ancestor
release reference under the checked ancestry policy. Its forced, cache-only
`cuda-correctness` build passed in `24.897233s`, versus `448.366051s` for the
release build on the same host. All 44 managed native artifacts were cache
hits, native recompile count was `0`, and the runnable binary SHA256 was
`00c4db8060e60718c63e7bd4e3267c0c03f2fbd70e08659bdcc8f64685b3bbae`.
The current binary independently ran `c13-022` in `112.409712s` with focused
`KEEP`; the final validator observed 17/17 exact plan hashes and printed:

```text
FERRUM CUDA CORRECTNESS SEMANTIC TRACE PASS: /workspace/ferrum-artifacts/runtime-vnext-typed-boundary-77575878-20260729-semantic-trace
```

The evidence is preserved in GitHub draft-release asset
`runtime-vnext-g07-plan-equivalence-77575878-20260729.tar.zst`
(asset `493152796`, `118590834` bytes), SHA256
`b9d739bc80d05fcaf1ad679f2a9a1deaa78880035377f46eabe3822d3101b06a`.
It was downloaded through GitHub and SHA256/zstd verified at
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-g07-plan-equivalence-77575878-20260729`.
Instance `46127509` was then stopped and reached `actual_status=exited`;
inventory reported `potentially_billable=[]`.

This closes the real CUDA release/correctness semantic-plan comparison and
provides one Rust-only cache-hit iteration sample. It is not the five-sample
p95 matrix, workspace source gate, canonical G07A PASS, G07B native-operator
PASS, or aggregate G07 PASS. Those items remain Open and must not trigger
another C13 or 703 sweep unless runtime-affecting source changes invalidate
this reference.

### 2026-07-30 source/package provenance checkpoint

Clean pushed production source `e229ccadeb852a9198a26f5d21eecc61c543ff7f`,
tree `5618341821cf41ff41d504c9534416c4b608989b`, closes the local
source-build-to-package-to-runtime evidence chain. Package specs can no longer
self-report source package, input hash or build summary and can no longer
accept an arbitrary archive. Artifact-set assembly also requires caller-owned
SHA256 pins for every package receipt and for the G03 catalog; it does not
derive its trust anchor from the package being verified. The packager and
assembler mechanically verify:

- exact operator and single built compute capability;
- locked source inventory, plan/source package/input hashes, deterministic
  environment, tool identities, command lines and object cache keys;
- source and final archive SHA256, exact non-metadata member sets and every
  member's SHA256, size and ELF/Mach-O/COFF ABI identity;
- descriptor/source object compatibility by actual format/class/endianness/
  machine rather than raw compiler-target spelling;
- absolute descriptor compiler/archiver identities, cleared deterministic
  package environment, exact package commands and non-empty logs;
- package spec/source receipt/plan/manifest semantic projection during
  assembly, including build summary, exports, operation bindings, licenses,
  toolkit/runtime and external catalog pin.

The package copies its spec plus source receipt, plan, command logs and
licenses into its provenance tree. Artifact-set schema v3 carries and
re-hashes those files plus the manifest, package receipt and package build
logs during every load. Negative tests reject coherent provenance edits
against an external receipt pin, stale source semantics even after an outer
pin is updated, a replaced final member with updated manifest/binary pins,
mixed object ABIs, false-success archiver output and tampered manifest/spec/
license/log evidence. Source object names use fixed eight-digit indices so a
101+ TU build keeps deterministic lexical archive order.

Bounded local evidence is under
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-g07-final-provenance-20260730`:

- `builder-lib-final-2`: `23/23` pass in `17.771347s`, peak 4 processes /
  21 process-group threads / 15 per-process threads;
- `native-ops-lib-final-2`: `23/23` pass in `2.315004s`, peak 4 processes /
  15 process-group threads / 10 per-process threads;
- `builder-check-final-2`: all-targets check PASS in `0.982159s`, peak
  3 processes / 14 process-group threads / 7 per-process threads.

All bounded receipts report no violation and complete process-group cleanup.
This is a source/test checkpoint, not G07A, G07B or aggregate G07 PASS.
Dependency depfiles, CUDA toolkit/subtool provenance, one fixed-RTX-4090
source-build-to-link/load chain, the five-sample timing matrix, workspace
source gate and canonical validators remain Open.

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
