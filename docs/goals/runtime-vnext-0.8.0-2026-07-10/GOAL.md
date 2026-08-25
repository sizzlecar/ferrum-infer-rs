# Ferrum Runtime vNext / v0.8.0 总目标

## 状态

Open。创建于 2026-07-10。

2026-08-06 起，v0.8.0 的发布阻塞范围和执行口径以
[`RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md`](RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md)
和后续
[`PERFORMANCE_ACCEPTANCE_AMENDMENT_2026-08-06.md`](PERFORMANCE_ACCEPTANCE_AMENDMENT_2026-08-06.md)
为准。后者将 v0.8.0 性能门改为量化的 Ferrum 自身基线回归；vLLM/llama.cpp 只作非阻断诊断，
竞争性比较转入 v0.8.1/0.9。修订保留三主模型 CUDA/Metal、真实 `run`/`serve`、动态资源、正确性、性能、profile、
编译阈值和发布后安装验证硬门，但把完整历史治理、未使用 provider、全仓 support disposition、
legacy physical zero 和平台完备性转入 v0.8.1/0.9 hardening。v0.8.0 正式进度分母改为
`R0-R3 1/4 PASS`；下文各时期的 `G00-G10 0/11` 是原 exhaustive roadmap 的历史状态，不再作为
release freeze 前置条件。

2026-08-12 起，开发期每个代码改动的回归范围按
[`CHANGE_IMPACT_REGRESSION_PLAN_2026-08-12.md`](CHANGE_IMPACT_REGRESSION_PLAN_2026-08-12.md)
执行：只允许 `exact reproducer -> affected contract -> affected product/architecture sentinel` 逐层扩大，
阶段完整矩阵只在源码冻结/阶段退出时执行一次。该执行计划不修改本目标或 active amendments 的最终
验收分母与阈值。

### 当前 release-blocking 状态（2026-08-07）

- 正式分母：R0 `PASS`，R1-R3 `OPEN`，合计 `R0-R3 1/4 PASS`。本段是当前状态源；下方按日期
  保留的 `0/11`、S2 child 计数和 focused checkpoint 均为历史快照，不能覆盖本段。
- R0 product/raw evidence source 为 clean、已推送
  `84d6b72419b21b39b68ac1102cbc6fbf031d5790`；R0 validator/authenticity 修复已推送至
  `7a1a79b0f64c5e901438cfd6e0be247927acd8d7`。正式 S2 和 R0 聚合分别打印：

```text
FERRUM RUNTIME VNEXT S2 CUDA PRODUCT CONTRACT PASS: /workspace/ferrum-artifacts/r0-s2-84d6b724/s2-current-r6
FERRUM GATE vnext-s2 PASS: /workspace/ferrum-artifacts/r0-s2-84d6b724/s2-current-r6
FERRUM RUNTIME VNEXT R0 CORE CLOSURE PASS: /workspace/ferrum-artifacts/r0-s2-84d6b724/r0-current-r2
FERRUM GATE vnext-r0 PASS: /workspace/ferrum-artifacts/r0-s2-84d6b724/r0-current-r2
```

- R0 证据归档 `runtime-vnext-r0-evidence-7a1a79b0.tar.zst` 为 `506452352` bytes，SHA256 为
  `32a5bdbc5e53176d5834ad194de1b695e424a8600b17fd08c84fdb1c5190b7b3`，已上传至 draft
  [GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-9ed36af1f2445efff73d)，
  并从 GitHub 下载到
  `/Users/chejinxuan/ferrum-artifacts/runtime-vnext-r0-transfer-7a1a79b0/` 完成 SHA256 和
  `zstd -t` 回环校验。
- R0 付费 RTX 4090 实例 `46998761` 已停止并确认 `cur_state=stopped`、
  `actual_status=exited`；最终 inventory 中无 running/loading/scheduling sibling。保留停止实例中的
  build/model cache，R1 需要 CUDA 时按单实例策略复用。
- 当前关键路径切换为 R1 Product Correctness：复用 G08A/G08B/G08C CUDA/Metal、产品场景和
  Llama dense 证据，补齐薄 `vnext-r1` 聚合及缺失的真实模型 correctness；R1 精确 PASS 前不得进入
  R2 正式性能结论。

截至 2026-08-04，clean、已推送 production source
`04699472d17b72a5368692bacb81a4b38cdae04f` 已关闭 S2 CUDA API/modality
checkpoint。真实 Qwen3.5-4B/RTX 4090 产品回归覆盖 C16 negative API `30/30` 和
C20 text-only modality `50/50`；其中 `40` 个 media 请求按 text-only contract 拒绝，
`10` 个 text-array 请求实际推理成功。release build bounded duration 为 `503.040129s`，
binary SHA256 为
`c76002ba57203182dedc76e258225f057f148878800aa722d3215ce2febd8494`；正式执行
bounded duration 为 `36.07631s`，峰值为 `2` processes、`12` group threads 和 `11`
per-process threads，无 violation。正式 validator 和统一 gate 打印：

```text
FERRUM RUNTIME VNEXT S2 API MODALITY PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-api-modality-04699472-20260804/gate-r2
FERRUM GATE vnext-s2-api-modality PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-api-modality-04699472-20260804/gate-r2
```

完整 archive `runtime-vnext-s2-api-modality-04699472-cuda-pass-20260804.tar.zst` 为
`625833` bytes，SHA256 为
`8943583fcf9c846767345b9777f425fe31272844f04bbfa93e5008ab1e5b6317`，位于 draft
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-37f92b8e7c62a636b72c)。
本机从 GitHub 回下载后完成 size/SHA256、`zstd -t` 和 required-member 复验。

这只关闭 S2 API/modality 子门，不等于完整 S2、G05、G08、Metal、性能或 release PASS。
8 个 S2 child lane 当前正式 artifact 状态为 `6/8 PASS`；剩余 M1 CUDA determinism 真实
硬件 witness 和 response-format。两个子门关闭后，仍须在同一最终 source identity 上执行
change-impact 复验和 S2 aggregate。G00-G10 总目标 PASS 仍为 `0/11`。

截至 2026-08-04，clean、已推送 source
`04699472d17b72a5368692bacb81a4b38cdae04f` 已关闭 S2 CUDA latency/first-failure
checkpoint。真实 Qwen3.5-4B/RTX 4090 artifact 同时覆盖 `ferrum run` success/failure 和
`ferrum serve` success/failure/recovery；失败终态保留真实 wall duration、typed failure status、
error detail 和统一 request identity，不再因为 `event_kind=error` 被通用 analyzer 丢掉。聚焦
`run` 注入失败的首个终态为 `actual_run_generation_failed`、`event_kind=timed_span`、
`duration_us=256196`，随后正式 validator 和统一 gate 打印：

```text
FERRUM RUNTIME VNEXT S2 LATENCY FIRST FAILURE PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-latency-04699472-20260803/gate-r1
FERRUM GATE vnext-s2-latency-first-failure PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-latency-04699472-20260803/gate-r1
```

CUDA release build bounded duration 为 `456.63442s`，binary SHA256 为
`c76002ba57203182dedc76e258225f057f148878800aa722d3215ce2febd8494`。同硬件
ABBA-BAAB profile-overhead 结果为 off `67.473362 tok/s`、latency `52.720071 tok/s`，
mean overhead `21.8654%`、median overhead `21.7777%`，两组 CV 分别为 `0.6297%` 和
`0.8774%`。该结果按 G06 合同标记 `target_miss`、`blocking=false`：它关闭 attribution
正确性边界，但没有达到 latency profile `<=5%` 的报告目标，也不是默认关闭路径的性能证据。

完整 archive
`runtime-vnext-s2-latency-04699472-cuda-pass-20260803.tar.zst` 为 `198342023` bytes，
SHA256 为 `59429fbe22bff760ecc5dab866227c356c359b60a80bb6809d63f5566e84dff5`，位于 draft
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-7b3e03ffaf497735b17d)。
远端原包与 12 个分片均为 uploaded；本机从 GitHub 回下载后完成 `12/12` 分片 SHA256、整体
SHA256、`zstd -t` 和 required-member 复验，并打印
`FERRUM S2 LATENCY GITHUB ROUNDTRIP PASS`。

这只关闭 S2 latency/first-failure 子门，不等于完整 S2、G05、G06、性能、Metal 或 release
PASS。该 checkpoint 当时的 S2 child lane 正式 artifact 状态为 `5/8 PASS`；剩余 M1 CUDA determinism
真实硬件 witness、response-format 和 API/modality。`dcf8e46c` 只完成 determinism collector/gate
接线，没有真实 4090 PASS artifact，不能计入完成。三个子门关闭后，仍须在同一最终 source
identity 上执行 change-impact 复验和 S2 aggregate。G00-G10 总目标 PASS 仍为 `0/11`。

截至 2026-08-03，S2 CUDA 又关闭两个 focused product checkpoint。clean、已推送 source
`a4e8472cc1aa3fa46cf8f95d89e8b6385f09576f` 的真实 Qwen3.5-4B Unicode stream/C09
artifact 中，stream/non-stream exact content、finish reason、usage 和 multibyte split 为
`20/20`；已经 admission 的 cancel/timeout/disconnect 为 `3/3`，runtime release wall 分别为
`9.742219/8.701876/8.229095 ms`，scheduler tick delta 均为 `0`，随后同容量请求为 `3/3`。
正式 validator 和统一 gate 打印：

```text
FERRUM RUNTIME VNEXT S2 STREAM DISCONNECT PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-stream-disconnect-a4e8472c-20260803/gate-r1
FERRUM GATE vnext-s2-stream-disconnect PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-stream-disconnect-a4e8472c-20260803/gate-r1
```

随后 clean、已推送 source `e894c59e4a2da5ea97303776b72e8532574736f9` 关闭 tool/schema
checkpoint：required-tool 与 strict-schema priority 的四个 marker 在 sync/stream 共 `8/8`，
omitted/explicit-auto/required/named tool choice 和 tool-result fill 全部通过。原失败不是 Ferrum
丢失 tool result，而是 2026-08-02 新增测试要求模型暴露一个用户从未要求的内部 receipt；同一
binary 已经正确使用 city/temp/unit/desc。测试现在从首轮明确要求返回不可猜的查询回执，继续以
exact nonce 证明 tool-result history 到达模型，没有放宽为语义近似。正式 PASS 为：

```text
FERRUM RUNTIME VNEXT S2 TOOL SCHEMA PRIORITY PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-tool-schema-e894c59e-20260803/gate-r1
FERRUM GATE vnext-s2-tool-schema PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-tool-schema-e894c59e-20260803/gate-r1
```

两条证据复用 binary SHA256
`41dcb29df52ebefc1052b990911a3f3f26e0ee697471a04ed51f1e8f8edbe605`。stream archive
`runtime-vnext-s2-stream-disconnect-a4e8472c-cuda-pass-20260803.tar.zst` 的 SHA256 为
`d56252bee7173a144a25c4a9d9a37b496dd08f85ead52b729909f69c3964f2d3`，位于 draft
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-a187c237c3a6487c949e)；
tool/schema archive `runtime-vnext-s2-tool-schema-e894c59e-cuda-pass-20260803.tar.zst` 的 SHA256 为
`3c3c2e07362d470b9337ed2345a2b4d363beb8225aaadd6aa04abdc56833b03f`，位于 draft
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-9b22242822f2734b5c02)。
两者均经 GitHub 回下载、SHA256、`zstd -t` 和 required-member 复验。付费 GPU 实例随后已停止并
确认 `actual_status=exited`。

这些证据只关闭 Unicode stream/C09 和 tool/schema 子门，不等于完整 S2、G04、G05、性能、
Metal 或 release PASS。在该 checkpoint 当时，尚未完成 response-format、API/modality、M1
determinism、五项 historical resource、latency/first-failure attribution 和最终 aggregate；
当前剩余项以本文件顶部最新状态为准。

截至 2026-08-03，clean、已推送 source
`cc2556610d30749e761f85fafe25ccf1a13ce116` 已关闭当前 source 的 S2 CUDA
multi-turn/concurrency sentinel。真实 Qwen3.5-4B CUDA 覆盖 `ferrum run` 五轮默认预算、
`ferrum serve` 两轮 recall，以及 serve c1/c4 隔离；五轮 run 无 length finish，c4 达到
`max_in_flight=4`、六对请求全部重叠且 crosstalk/error=`0`。共 `12` 个实际请求均有唯一
engine-owned scheduler lifecycle，其中 run=`5`、serve=`7`。

本轮先用同机 vLLM 作诊断 oracle，而不是 Ferrum 依赖或性能替代：旧的随机项目密钥提示在
vLLM 和 Ferrum 上都会触发拒绝/不复述，不能证明多轮状态损坏；改为模型稳定遵循的
`banana` codeword 后，两侧均可复述，Ferrum gate 仍要求 exact match。随后生产修复补齐交互式
`run` 每轮 success/failure observability 和 replay bundle，并把成功请求的 scheduler 生命周期
收敛到 engine 唯一 owner；CLI 不再重复写 synthetic open/close，也不能用 fallback 隐藏 engine
trace 缺失。正式 validator 和统一 gate 分别打印：

```text
FERRUM RUNTIME VNEXT S2 MULTITURN CONCURRENCY PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-multiturn-concurrency-cc255661-20260803/gate-r1
FERRUM GATE vnext-s2-multiturn-concurrency PASS: /workspace/ferrum-artifacts/runtime-vnext-s2-multiturn-concurrency-cc255661-20260803/gate-r1
```

CUDA release build bounded duration 为 `403.82842s`，binary SHA256 为
`41dcb29df52ebefc1052b990911a3f3f26e0ee697471a04ed51f1e8f8edbe605`。完整证据归档
`runtime-vnext-s2-multiturn-concurrency-cc255661-cuda-pass-20260803.tar.zst` 的 SHA256 为
`49918196a8d68de6b8de3403ac2dde4b4670412ca50c6135ee9e8ceb0e1b7a36`，已由 GPU 机直接上传
到 draft [GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-74e720b07de638ea68c0)，
并经 GitHub 回下载、SHA256、`zstd -t` 和 required-member 复验。

该证据只关闭 S2 multi-turn/concurrency 子门，不等于完整 S2、G02、G04、G05、G06、性能、
Metal 或 release PASS。在该 checkpoint 当时，aggregate 尚须收齐或按 change-impact 复验 S1
inputs、determinism、response-format、API/modality、五项 historical resource 和 latency
attribution；当前剩余项以本文件顶部最新状态为准。

截至 2026-08-03，clean、已推送 source
`61399d6907677af3373f543c01b5f0cad720dbe4` 已关闭当前 source 的 S1 CUDA
decode execution-capacity 子门。此前物理 backing maintenance deferral 在 executor 到
engine 的边界被折叠成 `None`，导致 scheduler trace 同时缺少 logical shortfall 和
backing blocker；本次将 pressure evidence 收敛为唯一 typed owner/kind/source，并在生产
trace 中保留同一证据。真实 Qwen3.5-4B CUDA `ferrum run` 和 `ferrum serve` 强制
decode-pressure 序列中，原失败事件现在为 `owner=backing`、
`kind=backing_deferred`、`backing_blockers=1`、`pressure=device_capacity`、
`yield_kind=peer_handoff`，不再产生无因果来源的 capacity deferral。正式 validator 和
统一 gate 分别打印：

```text
FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY PASS: /workspace/ferrum-artifacts/runtime-vnext-g04-typed-evidence-61399d69-20260803/decode-capacity-r3/validation-r1
FERRUM GATE vnext-s1-cuda-decode-capacity PASS: /workspace/ferrum-artifacts/runtime-vnext-g04-typed-evidence-61399d69-20260803/decode-capacity-r3/gate
```

CUDA catalog 保持 `12` 个 provider，fingerprint
`a1d87c8ba2714fbb012897c28939d2f88dc3e2214ae20673d62fc1d491f02f96`；因此 native
object/archive 复用既有 content-addressed artifact，仅重新链接 Rust 生产二进制。release
build bounded duration 为 `463.95347s`，binary SHA256 为
`5f8a370e1bb8db183608ae6c63c89e440060f059e82b44ea01aded142ec05e76`。CUDA 和本地
focused artifact 均已上传到 draft
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-b0701503cde3cb34993b)，
archive SHA256 分别为
`ede721013129a07aca0b18ce8e9e84a6e44fe5fcad2972bbf0620ac9503cbdde` 和
`2b6e41578f605eea1204785edb08115a30fedfdcfe647c600d8f03d55fe8991b`；两者均已从
GitHub 回下载并通过 SHA256、`zstd -t` 和 required-member 复验。

该证据只关闭 S1 decode-capacity 补充 lane，不等于 G04、S1、G08B 或总 Goal PASS，
也不是性能证据。G00-G10 总目标 PASS 仍为 `0/11`；下一条产品关键路径为 S2 CUDA
实际 `run`/`serve` 正确性，先执行受影响场景和 multi-turn/concurrency gate，达到阶段
milestone 后才执行完整 matrix。

截至 2026-07-30，当前分支最新 clean、已推送的 production-source checkpoint 为
`e229ccadeb852a9198a26f5d21eecc61c543ff7f`，tree
`5618341821cf41ff41d504c9534416c4b608989b`。该 G07 checkpoint 将 native package
输入从任意 archive 收紧为 terminal PASS source-build receipt + locked plan，并要求
artifact-set 调用方从 package 外部提供逐 receipt SHA256 和 G03 catalog SHA256。
packager/assembler 机械重算 source/plan/input/cache/command identity，逐项核对 source
和 final archive object SHA256/size/ABI identity，并把 package spec、source/package
receipt、plan、manifest、license、toolchain 和非空日志贯穿到 schema v3 artifact lock。
bounded builder `23/23`、native-ops `23/23` 及 builder all-targets check 均通过；证据根目录为
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-g07-final-provenance-20260730`。
这仍不是 G07 PASS；depfile/CUDA toolkit 子工具 provenance、固定 RTX 4090
source-build→package→resolve→link/load、五样本 p95、workspace source gate 和三个
canonical validator 仍未完成。

最近一次完整 M2 Metal correctness source 为
`02a2ef072eadadee77cd9f825d7d6ad985112433`，tree
`2b8deda6b8310420d17961274ef137a1abf84d7c`。`a1d92aaf` 修复 structured-output
合法 token 不收敛，`02a2ef07` 将临时 backing pressure 从终止错误改为携带 blocker
的 typed deferral，并由 engine 缩批、释放 workspace 后等待重试。

相同 source 的 M2 Metal canonical correctness 已完成 `702/702 pass`，21 个
scenario 全部通过；正式 validator 打印：

```text
FERRUM RUNTIME VNEXT G08B METAL MODEL MATRIX PASS: /Users/chejinxuan/ferrum-artifacts/runtime-vnext-g08b-metal-full-02a2ef07-20260730/gate-vnext-g08b-metal
FERRUM GATE vnext-g08b-metal PASS: /Users/chejinxuan/ferrum-artifacts/runtime-vnext-g08b-metal-full-02a2ef07-20260730/gate-vnext-g08b-metal
```

由于上述共享生产代码晚于 `2ba731e2` 的 CUDA `703/703` checkpoint，当前 source
没有立刻重跑全量 CUDA。按 G02 staged policy，仅执行
`c03-001/c05-001/c06-001/c13-022/c15-063/c18-001/c18-002/c18-003`；
真实 `ferrum run`/`serve` 为 `8/8 pass`，focused report 打印
`FERRUM RUNTIME VNEXT FOCUSED DIAGNOSTIC KEEP`。bounded duration 为
`185.072369s`，峰值为 `2` processes、`83` group threads、`66` per-process
threads，无 violation，进程组清理成功。证据保存在 draft
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-2bca46f9154f90ab4f50)，
archive SHA256 为
`79314b77f3ebf6dffd9234d8629d8b84fa5fd78f00809ddc775b94c171666230`，
并已从 GitHub 下载后通过 SHA256、`zstd -t`、required-member 和 summary 复验。
Vast instance `46217231` 已确认 `stopped/exited`，最终
`potentially_billable=[]`。

这不是 CUDA `703` 正式 PASS，也不把双后端验证改成逐修复全量乒乓回归。已登记的
formal correctness matrix 仍为 `1/6`（M2 Metal），但当前 `e229ccad` 尚无同 SHA
完整双后端证据；M2 CUDA 在 `02a2ef07` 只有 affected `KEEP`。完整 `703` 只在下一次
runtime source freeze 执行。G00-G10 总目标 PASS 仍为 `0/11`。下一条关键路径先关闭
G07 depfile/toolchain provenance、固定硬件链、五样本 invalidation timing、workspace
source gate 和 canonical validator；Metal 性能继续遵守 owner 的空闲主机约束。

截至 2026-07-29，clean、已推送 source
`2ba731e2bb70dc09763f8582c8b67b8b7e42ac77`，tree
`4a2b83d28bc04b87001a9b80bac589ba19db28fe` 已修复 partial reusable program
边界：request-lifetime embedding/logits 保持 eager，中间 5 个 lane-stable CUDA
segments 保持 resident/direct；单个不稳定边界不再错误否决整条 wave。真实
`run`/`serve` 启动得到 `prepared_programs=5`，聚焦 C02/C05/C06 为 `60/60 pass`，
direct fallback 为 `0`。

相同 clean source 的 M2 Qwen3.5-35B-A3B-GPTQ CUDA canonical correctness 已重新完成
`703/703 pass`，21 个 scenario 的 known-fail/blocked/error/unexpected 均为 `0`。
bounded duration 为 `2590.064111s`，峰值为 `3` processes、`85` process-group
threads 和 `52` per-process threads，无 violation，进程组清理成功。C18 typed
admission cap 与 observed max-active 均为 `16`，active duty-cycle 为
`0.9635368787`；request slot、model cache 和 backend workspace 的资源事务全部平衡，
leak/underflow 为 `0`。正式 validator 打印：

```text
FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX PASS: /workspace/ferrum-artifacts/runtime-vnext-g08b-cuda-full-2ba731e2-20260729-r1/gate-vnext-g08b-cuda
FERRUM GATE vnext-g08b-cuda PASS: /workspace/ferrum-artifacts/runtime-vnext-g08b-cuda-full-2ba731e2-20260729-r1/gate-vnext-g08b-cuda
```

证据通过 draft
[GitHub transfer release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/untagged-aa32bc08b2eb70787bea)
保存；重组 archive SHA256 为
`cf588945f13d4ee9cd58228725f1f20ccf21f463873b9d07c93abe4b37e0bf07`，
已从 GitHub 回下载并通过分段/整体 SHA256、`zstd -t` 和 tar member 复验。Vast
实例 `45897840` 已确认 `stopped/exited`，最终 `potentially_billable=[]`。
这刷新而不增加 M2 CUDA cell，三主模型 x 双后端 fresh correctness matrix 仍为
`1/6`，G00-G10 总目标 PASS 仍为 `0/11`。下一条开发关键路径仍在 S4/G08B：
M2 Metal correctness；按 owner 2026-07-29 决定，工作软件占用资源期间不采集
Metal 性能证据。

截至 2026-07-28，最近一次完整 M2 CUDA correctness 的 clean source checkpoint 为
`b0431ca5b384a86c4e1f57406ad7267bdd3c3705`，tree 为
`0896756c962c6d983d6d251ad2c7d5bd99f122ad`。`27dd5c7f` 将 scenario request
manifest 绑定到实际 wire bytes，`b0431ca5` 修正 C01 negative-layout checker：
typed resolver 必须在读取 weight 前以非零状态拒绝，并且错误必须携带测试注入的
unknown architecture/model_type identity；通用 `unsupported architecture/layout`
仍会被 mutation test 拒绝，不能用弱化 oracle 冒充修复。本地 bounded self-test
用时 `173.578729s`，峰值 `2` processes / `4` threads，无 violation、cleanup 成功，
并打印 `FERRUM RUNTIME VNEXT G00 SCENARIOS SELFTEST PASS`。

clean `b0431ca5` 的 M2 Qwen3.5-35B-A3B-GPTQ CUDA canonical correctness 已完成
`703/703`，C01-C21 共 `21` 个 scenario 全部通过；正式 checkpoint 和 unified gate
分别打印
`FERRUM RUNTIME VNEXT G08B CUDA MODEL MATRIX PASS: /workspace/ferrum-artifacts/cuda-703-b0431ca5/gate-vnext-g08b-cuda`
与
`FERRUM GATE vnext-g08b-cuda PASS: /workspace/ferrum-artifacts/cuda-703-b0431ca5/gate-vnext-g08b-cuda`。
这使三主模型 x 双后端 fresh correctness matrix 从 `0/6` 前进到 `1/6`。正式
G00-G10 总目标 PASS 仍为 `0/11`；G08B 整体也仍为 Open，因为 Metal correctness、
CUDA/Metal performance、legacy/reference parity 和其余验收项尚未关闭。

`7ae12059` 已修复 MoE pair alignment 的跨运行顺序漂移。相同 clean source、相同
binary SHA256
`0a9c825fe70fc45b721554f9561ffe32d3a2166f80f6e5af1588e8d33a25285c`
的四次真实 CUDA prefill capture 在 embedding、layer-0 attention、layer-0 output 和
full logits 四个边界均为 `unique_count=1`，证明当前产品执行在这些观察点已可重复；
它不证明 C13 语义正确。随后 exact `c13-022` 仍打印 focused diagnostic REJECT，
错误为 `did not incorporate the tool result`。两个 artifact 已在 GitHub branch
`artifact/runtime-vnext-c13-7ae12059-20260728` commit `cf3ee553` 保存；determinism
archive SHA256 为
`39f3be353dd17253cf57eb73a75ad928aae81fd23f8cdd5162953e5f6e56d9ab`，
exact REJECT archive SHA256 为
`f190007d828c95bbe98822d3344985754dba2fd5caf4c2e43dd0827884b16910`。

source/artifact audit 已证伪 rendered prompt、tool history、template/source hash、seed
传播、reusable execution 和 checkpoint/product readback 竞争是当前缺失 `21` 的直接原因。
旧 `6fa8e215` 使用 nondeterministic atomic MoE ordering，因此其单次 C13 PASS 不能作为
bitwise oracle。对 batch-1 finite/unique top-k route 的 host 枚举也证明 direct 和 generic
Marlin metadata 在 consumed prefix 等价；非有限 router logits 的 duplicate route 仍需
独立 invariant/test，但没有证据表明它解释当前请求。当时首要未决边界是 decode 期间的
数值/token 分叉，而不是产品输入丢失。

clean source `addaf796` 的 tool-result sensitivity capture 进一步把 calculator result
从 `21` 改为 `99`：两次 prompt 都是 `61` tokens，但 full-logits SHA 不同，
`248,157/248,320` 个 F16 值变化，cosine `0.9262496`、relative L2 `0.3805839`，
top-20 只重合 `13/20`。因此“runtime 在模型执行前丢掉 tool-result history”已被真实
CUDA 证据排除。artifact 位于 GitHub branch
`artifact/runtime-vnext-c13-sensitivity-addaf796-20260728` commit
`13da606bc52b1ac1b4086ed75c9356c1a49e5fea`；它是诊断 KEEP，不是 C13 PASS。

同一 historical `61`-token fingerprint
`38cdf236189f6b93770ff50c0b231ce640b3577199b51e90311c6883dc7580ed`
下的 Ferrum/vLLM 采样前 raw-logits 对照已经完成。`seed=9271` 不要求两个实现选择相同 token：
vLLM 使用请求级 PyTorch generator 和 exponential-noise race，Ferrum 使用版本化
ChaCha12 inverse-CDF；两者分布可等价而 seeded realization 不同。只有 raw-logits
argmax、top-20/nucleus overlap、centered cosine 和 affine residual 能判定模型执行是否
偏离参考。结果为 `KEEP_REFERENCE_ALIGNMENT`：双方 argmax 均为 token `760`，
top-20 overlap `0.95`、nucleus Jaccard `1.0`、centered cosine
`0.9910251391`、affine relative L2 `0.1336756283`，全部满足预注册 KEEP 阈值。
validator 打印
`FERRUM C13 LOGITS REFERENCE KEEP: .../comparison`。GitHub evidence branch 为
`artifact/runtime-vnext-c13-vllm-reference-4ff1a5d9-20260728`，artifact commit
`e1cdc94c525feb9ed4297c1034731f1331eeb7a5`。因此 historical C13 的首 token
model execution 不再是当前 blocker，不得继续为它修改 kernel 或更换 RNG。

本轮尝试构建“旧基线 + deterministic alignment”和“当前源码强制 generic MoE”两个
诊断二进制，分别在 `720s + 360s` 的 native compile continuation 和 `360s` 的 release
LTO 边界超时，均未产出可执行二进制，因此没有形成 A/B 结论。该结果把下一步前置条件
收敛到 G07：先提供与正式 feature/产品语义相同、但不被完整 native 重编或 release LTO
阻塞的 correctness 开发构建路径，并证明可运行 binary 和 semantic plan hash；禁止再用
付费 GPU 重复等待相同编译失败。

clean source `9b318ea9` 已完成该路径的 source checkpoint：新增与正式 CUDA features
一致的 `cuda-correctness` profile、内容寻址 native artifact cache、跨 profile 的显式
import、bounded binary build，以及独立的真实 execution-trace semantic validator。cache
key 绑定 source/header SHA256、SM、CUDA headers、`nvcc`/host compiler/archiver version、
`TARGET/HOST` 和影响 nvcc 的 ambient flags；paged-attention 漏记的
`quant_utils_stub.cuh` 也已进入 invalidation closure。构建阶段只能打印
`FERRUM CUDA CORRECTNESS BINARY READY`，不能形成 PASS；只有 build/execution/focused
三份 manifest 的 source 与 binary SHA256 一致，并且实际 `ferrum serve` trace 中所有
`vnext.plan_built` hash 精确等于经审计的 release reference，validator 才能打印 semantic
trace PASS。该 source checkpoint 的本地聚焦测试已通过；
在 `b0431ca5` 上复用同一 native cache 的 CUDA binary rebuild 约 `6s` 完成，证明
correctness 开发构建不再必然等待完整 native rebuild。G07A 的 semantic trace、p95
和 workspace gate 仍须按其正式验收单独关闭，不能由 G08B correctness PASS 代替。

`54963e9ddc468d44eaf72227c603a0f64d19e1f151de58e41fa33fdd402cc09d`
现在只保留为旧源码的 historical release reference。`f9bb4070` 修改了被 CUDA runtime
implementation fingerprint 覆盖的 replay/runtime 源码；clean `645a4371` 的实际
`c13-022` 产品请求通过并打印 focused `KEEP`，其唯一 plan hash 为
`96efe5ea30955542290d58ea76d4768a46c79886391cf06ffb538e16b726b586`。
旧 semantic validator 因仍携带 `54963e...` 而 REJECT；这证明旧 reference 已 stale，
不表示该产品请求发生 correctness 回归。证据通过 GitHub draft-release asset
`cuda-c13-022-645a4371.tar.zst` 保存，archive SHA256 为
`c4f3a4ffe7e068c6ca8b025e44b09ed6d601eec80dae73e25bad0db0997a22e5`。

clean、已推送 source `fae4f6e777a2b6538f9382af9aa74ee8196ca21c`
新增独立 `runtime_vnext_plan_reference.py`：reference capture 必须绑定 clean
source/tree、M2 model revision 与文件锁、typed/actual effective config、固定硬件、
官方 release build receipt、release binary SHA256、focused report 和真实 scheduler
trace。correctness build 不再接受裸 plan hash，而是复制并再次验证完整 reference；
reference source 只能等于或祖先于 candidate，最终 trace plan identity 仍必须精确相等。
伪造 plan identity、同步伪造 `plan_id`、更新外层文件哈希后把 build mode 改成 debug
均被负例拒绝。完整场景自测在
`/Users/chejinxuan/ferrum-artifacts/runtime-vnext-plan-reference-local-20260728/baseline-selftest-r2`
用时 `177.685141s`，峰值 `3` processes / `34` group threads /
`17` per-process threads，无 violation、进程组清理成功，并打印
`FERRUM RUNTIME VNEXT G00 SCENARIOS SELFTEST PASS`。这是 source/test-system
checkpoint，不是 G07、G08B 或 CUDA hardware PASS。

此前 G07 CUDA/Candle boundary 的 clean source 为
`ecd6d1e8739467ab290bdb64df7a12e6a9c2cde6`，tree 为
`8e363e3b0efa90f11ffaf967f069a9ac00b77584`。正式 CUDA feature graph 已把
Candle CUDA 隔离到显式 `candle-cuda-compat`：official graph 的 `candle-core` /
`candle-nn` 只有 CPU `default` feature，`candle-kernels` 不存在；vNext CUDA GPU
执行使用 `cudarc`、cuBLAS 和 Ferrum/vLLM native kernels。相同 clean source 在
1x RTX 4090 上以 `cache-only` 构建正式 CUDA binary，用时 `471.000627s`，
native compiler 数为 `0`，binary SHA256 为
`a6e5059c7cab467ea8cf79b37beb102644a19dea8267d76b038ce63531b11a11`。
这仍不是“整个 CUDA binary 零 Candle”：vNext 产品边界仍用 CPU Candle tensor
包装 token/logits，Metal、GGUF、embedding 和多模态实现也继续依赖 Candle。
本次 GitHub draft-release evidence asset 为
`cuda-g07-candle-boundary-ecd6d1e8.tar.zst`（asset `492770357`），archive
SHA256 为
`2fed9d75c9535b5437210c42d8a7a50fb3670cc9ee9e68ba7b5ffd0ee61af5d0`。
它是 G07 CUDA/Candle boundary 和单次 release-build diagnostic，不是五样本 p95、
G07A、G07B 或 G07 aggregate PASS。

当前最新已验证并推送的 G07 clean source 为
`775758780baf66c3a508f8b07237035483c97410`，tree 为
`135ac8a1dffa9add13b918478d46a87ec2304288`。同一 RTX 4090 上，正式 release
reference build 在 clean ancestor `5e2aaaed` 用时 `448.366051s`，binary SHA256
为 `e70114ed09c710eede78415e7f82c20a355ec04cc64ceff6f6e5089f6db21210`；
current `cuda-correctness` cache-only build 用时 `24.897233s`，44 个 native artifact
全部 cache hit、native recompile `0`，binary SHA256 为
`00c4db8060e60718c63e7bd4e3267c0c03f2fbd70e08659bdcc8f64685b3bbae`。
两者分别执行 exact `c13-022` 并得到 focused `KEEP`；current trace 的 17/17 个
`vnext.plan_built` event 均精确等于 release reference
`96efe5ea30955542290d58ea76d4768a46c79886391cf06ffb538e16b726b586`，最终打印
`FERRUM CUDA CORRECTNESS SEMANTIC TRACE PASS`。

证据保存在 GitHub draft-release asset
`runtime-vnext-g07-plan-equivalence-77575878-20260729.tar.zst`
（asset `493152796`，SHA256
`b9d739bc80d05fcaf1ad679f2a9a1deaa78880035377f46eabe3822d3101b06a`），并已通过
GitHub 下载到本机完成 SHA256/zstd 复验。这关闭 G07A 的真实 release/correctness
semantic-plan 对照和一个 Rust-only cache-hit 样本；五样本 p95、workspace source gate、
canonical G07A/G07B/G07 PASS 仍未完成，因此 G07 仍为 Open。

paid inventory 已核对：RTX 4090 实例 `46127509` 与 `45897840` 均为
`cur_state=stopped`、`actual_status=exited`，`potentially_billable=[]`。在 runtime-affecting
source 变化前不再重跑 focused C13 或 703 全量；G07 下一步转为五样本 invalidation timing、
workspace source gate 和 canonical validator，CUDA 主模型工作继续剩余正确性/性能 lane。
historical `7*3 -> 21` 文件只保留作 raw-logits attribution，不再冒充当前 C13 replay；
禁止输出过滤、模型名特判、降低 C13 oracle 或因单个失败重跑全量。

2026-07-30 的后续 G07A fixed-host diagnostic 已在 clean pushed source
`9e68055481b8ade0f3e9600e202e814414a48fbc` 上证明 profile 分层修复：
no-op 为 `0.819575s`，Rust model leaf 为 `7.079652s`，两者均使用
`cuda-correctness`，native TU 重编数均为 `0`；正式 clean release 仍绑定
`--release`。完整 GitHub asset `495338367` 的 SHA256
`eb2fe227492ca146cd322d88d1ffff00bd1d2e5bf9fc40a0b76abea31a34e722`
已在本机复验。该结果是 focused `KEEP`，不是五样本 p95 或 G07A PASS。
bootstrap 与首次 prewarm 重复使用两个冷 target 的问题也已收敛为共享声明式
correctness target，待下一轮 diagnostic 验证后再进入 canonical matrix。实例
`46247151` 已停止到 `actual_status=exited`，当前 potentially billable 实例数为 `0`。

当前 source checkpoint 已注册正式 `run_gate.py vnext-g07a` lane。其 child
checkpoint 会重算 canonical 五样本 raw timing，并显式绑定 current-clean-source 的
G00F、S1 `run`/`serve` CUDA slice、bounded workspace unit source gate 和
release/correctness semantic-plan trace；unit child receipt 绑定测试前后完全相同的
`HEAD/tree/status`，semantic artifact 自带 release/correctness 两个真实二进制、
typed config、模型锁和 trace，并独立重算全部 SHA256 与 plan hash。外层 gate 再次
验证全部引用和当前 checkout。
`FERRUM RUNTIME VNEXT G07A CHECKPOINT SELFTEST PASS` 与
`FERRUM RUN GATE SELFTEST PASS` 已通过。这只关闭 validator 接线缺口，不是 G07A
完成证据；仍需在同一 clean SHA 上取得五样本 raw evidence 以及精确 child/outer PASS。

截至 2026-07-25，正式 G00-G10 PASS 仍为 `0/11`，三主模型 x 双后端 fresh
correctness matrix 仍为 `0/6`。当前生产纵切是
`G03 weight ABI/fidelity contract -> G08B M2 CUDA current-HEAD correctness -> G09
exact-precision performance`。`0b72bab2` 的隐式 F16 -> FP8 candidate 虽曾取得窄
`run`/`serve` smoke、Marlin dispatch profile 和单次 c1 bounded KEEP，随后在 `7bc46122`
的 C17 deterministic Unicode case 中丢失首 token，已判定 correctness REJECT。`5149bbfb`
禁止默认近似重量化并恢复 exact-F16 C17 输出；`883ee9e0` 进一步把 exact QKVZ+BA
冷打包为单个 QKVZBA projection，并在 CUDA/Metal 共用 schema v6。源码、replay contract
和真实 Metal 数值用例已通过。clean `557cdcf5` 的 1x RTX 4090 focused CUDA lane 随后通过
9 个 C03/C05/C06/C17 用例、`4,500` 个 GDN projection GEMV、每 correlation `300`
dispatch 和 `7.6973 ms` device-duration target；bounded c1 为 `73.3800 tok/s`，与
`73.3855 tok/s` exact reference 持平，但仍比 `76.1583 tok/s` floor 低 `3.65%`。
2026-07-25 owner 决定按 G09 的窄范围 M2 CUDA c1 修订把该差距作为 development
checkpoint 接受，并停止继续进行 c1-only 优化。

后续 clean `acf9a669` 首次把 typed device token selection 接入产品执行，但默认
`repetition_penalty=1.1` 仍触发 full-logits fallback；同机 bounded c32 为
`109.1643 tok/s`，trace 是 `greedy_token_waves=0`、
`greedy_policy_fallback_waves=164`、readback `1,167,104,000 bytes`，因此形成
product-default integration REJECT。commit `90057af1` 把 selection contract 升级到 v2，
用固定容量 typed token IDs、offsets 和 penalty 在 CUDA/Metal provider 内精确执行 sparse
repetition，再返回单个 token。clean 1x RTX 4090 release build 用时 `5m12s`，9 个
C03/C05/C06/C17 `run`/`serve`/stream/Unicode 用例全部通过；相同 c32 workload 为
`121.0025 tok/s`，比 `acf9a669` 提升 `10.84%`，decode fallback 降为 `0`，readback 降至
`61,592,288 bytes`（`-94.72%`）。该 artifact 仍只有一次 repeat、无 CI，且不是 703-case
CUDA matrix，所以仍不是 G03、G08B 或 G09 canonical PASS，也不降低 same-host vLLM、
并发矩阵、正确性或最终发布标准。最新证据在 GitHub branch
`artifact/runtime-vnext-sparse-repetition-90057af1-20260725` commit `65dfab41`，
archive SHA256 `1cd3b1fc3e656ecce8cd9f71f23877404f208665ed4e1596b1304917983637d2`；
付费实例已确认 `stopped/exited`，无 paid/transitional sibling。

commit `d995201e` 随后把共享物理 token 区域的多请求 GDN wave 合并为一个 packed
projection/prepare/conv/delta/output 链，并以每序列 state pointer table 保留独立 recurrent
state；`affacd2c` 修复基础 CUDA feature 的 `c_void` 编译边界。clean `affacd2c` 的官方
CUDA feature build 用时 `5m05s`，binary SHA256 为
`4123b86531951cb802693776ec93c798ac953a065d0d2ec570402fbb60ee2d2d`；真实 CUDA
GDN/linear-attention 数值测试 `14/14` 通过。focus-scenario 命令实际选择了 `110` 个
C03/C05/C06/C17 产品用例而不是预期的 9 个，结果 `110/110` 通过；相同 4090、相同
random `64/32` c32 workload 完成 `64/64`、零错误，得到 `155.1381 tok/s`，比
`90057af1` 的 `121.0025 tok/s` 提升 `28.21%`。本轮性能命令漏传
`--profile-jsonl`，scheduler trace 只能证明 GDN node 执行，不能直接证明 native
`batching_form=packed` 和物理 dispatch 计数，因此结论是
`KEEP_CODE_AND_PERFORMANCE_PENDING_PACKED_ATTRIBUTION_TRACE`，不是 G08B/G09 PASS。
远端压缩 artifact SHA256 为
`7116188660add41efa654d4256fc0c1d2f9b8d6ffef08e7230853125b2c65a49`；因远端
GitHub credential/SSH key 不可用，archive 与 commit 暂留 retained stopped instance，
本机只有 stop/inventory metadata，未使用 SCP。实例已确认 `stopped/exited`，无
paid/transitional sibling。

后续 `e0a74fa0` 加入通用 native-work attribution gate，并通过临时 stdin credential
先把旧 archive 完整推到 GitHub。`profile-detail=basic` 的首轮 closure 明确 REJECT：
有 `3,240` 个 GDN node event，但没有 native-work event；源码确认 CUDA 只有 kernel
attribution 或 reusable capture 才构造物理 command attribution。`profile-detail=full`
随后产生 `22,906` 个 native-work event；其中 `2,520/2,520` 个多参与者 GDN compute
event 全部满足 `batching_form=packed`、compute dispatch `10`、transfer `2`，参与者数
覆盖 `3/4/13/19/32`，并打印：

```text
FERRUM NATIVE WORK ATTRIBUTION PASS: /workspace/ferrum-artifacts/runtime-vnext-packed-gdn-trace-affacd2c-20260725T161940Z/validation-full/native-work-attribution
```

因此 packed GDN 的“代码存在但产品未调用”风险已关闭，结论升级为
`KEEP_PACKED_GDN_NATIVE_ATTRIBUTION`。这仍不是 G08B/G09 PASS：完整 CUDA matrix、
三模型双后端、置信区间、legacy/external comparison 和 release gate 均未完成。compact
evidence 在 GitHub branch
`artifact/runtime-vnext-packed-gdn-trace-affacd2c-20260725` commit `fab7ac75`，
archive SHA256 为
`ea93671ead07eec23bafc921a24962a0140ee9c21c933ffc09f90ea8f77cb909`；原始
`2,612,056,519` bytes profile/trace 由逐文件 SHA256 绑定并保留在 stopped instance。
本次 paid window 为 `26m04s`、约 `$0.2039`，比 `$0.157` cap 超出约 `$0.0469`，
失败分类为 full-profile overcollection；后续只采最小请求数并在 attribution 命中后立即
停止。实例已确认 `stopped/exited`，无 paid/transitional sibling。

clean、已推送 source `e2136178627d86ed9b520778f41138dcb37772d2`
随后完成 CUDA causal-attention packed shared work：RMSNorm、Q/K/V、output projection
与 residual 不再按 participant 重复。当前 official-feature release build 用时
`448.616s`，binary SHA256 为
`e2e09bee9e5ea3f801772c8f60ee6dda6c3cd358e76e0496e4f35d8d80ae7953`；
精确父提交 `7f8ff122` 的同配置 release build 用时 `442.695s`，binary SHA256 为
`e70114ed09c710eede78415e7f82c20a355ec04cc64ceff6f6e5089f6db21210`。
CUDA focused tests `15/15 + 3/3`、C03/C05/C06/C17 产品用例 `9/9` 和短 c32
`32/32` 均通过。

最小 full profile 的 `80/80` causal compute events 全部为 packed，participants 覆盖
`2/30/32`，V1、varlen-q4 和 mixed path 均满足 dispatch `6 + 3P`、transfer `0`，
三个 validator 分别打印 `FERRUM NATIVE WORK ATTRIBUTION PASS`。同机、同配置、
profile-off random `64/32` c32 单变量 A/B 为 `106.287877 -> 116.038538 tok/s`
（`+9.1738%`），双方均 `64/64`、零错误。因此代码与真实产品消费结论为 KEEP，
但 G09 仍为 REJECT：仅一次 repeat、无 CI/同机 vLLM，且比另一台 4090 的历史
`155.138143 tok/s` checkpoint 低 `25.20%`，未授权 full sweep。

本轮同时暴露新的 G03/G06 blocker：typed effective config 报告 V2 与 admission `16`，
真实 native work 却执行 V1/varlen/mixed 并达到 participant `32`。下一步必须让 compiled
kernel-selection policy、effective config 和 physical admission 共享同一 typed authority，
之后才允许继续付费性能验证。compact archive SHA256 为
`0bf70cb8e308695e958f03a079611902d030b083018f248807aaa757f094db1c`；
付费 host 到 GitHub 的 push 在 `133.402s` 超时，未使用 SCP，archive 与
`578,812,941` raw bytes 由 SHA256 绑定并保留在 stopped instance `46127509`。
本次 paid window 为 `54m54s`、约 `$0.3558`，未超过 `$0.58` cap；实例
`46127509` 和 `45897840` 均为 `stopped/exited`，`potentially_billable=[]`。

clean、已推送 source `fab3996956f9daefe7487904a04a0c66ecd07bb3`
关闭了上述 typed-authority blocker。`c8bba60f` 将 attention execution policy、
executor admission limits、scheduler phase snapshot 和 provider implementation
fingerprint 收敛到 PlanRuntime/DeviceRuntime 的 typed authority；`ead75924` 修复
CUDA-only error mapping，`fab39969` 把 Portable policy 的测试边界改为“禁止可选
vLLM V1/V2 provider、允许 Ferrum builtin PTX”，没有把动态 V1/V2/varlen 选择重新写死。
official-feature release binary 构建用时 `468.558363s`，SHA256 为
`35d6aa421cc33be164b6f6b75e3fe5e1a60197fdd7851c3b333d4155405d18cb`；
base CUDA policy tests `2/2` 通过。

当前 HEAD 的真实 CUDA `run`/`serve` 哨兵覆盖 multi-turn、non-stream、stream usage /
`[DONE]`、中文和 admission，共 `6/6` 通过。独立 c32 `serve` 压力完成 `32/32`，
typed cap 与 observed max-active 均为 `16`，active duty-cycle 为 `0.9591967864`，
error/OOM/panic/crosstalk 均为 `0`；`request_slot`、`model_cache_ref` 和
`backend_workspace` reserve/release 全部平衡，leak/underflow 为 `0`。最小公开
`--profile-detail full --profile-jsonl` 产品诊断随后生成真实 CUDA profile；
`320/320` 个 addressed V1 causal-attention event 携带 64-hex provider fingerprint
`7fb8cf0ce185a9ac0abe61946adf9ecd7fdf163181e7a10e4d148607c913625e`，
且全部满足 compute dispatch `9`、transfer `0`。validator 打印：

```text
FERRUM NATIVE WORK ATTRIBUTION PASS: /workspace/ferrum-artifacts/runtime-vnext-authority-fab39969-20260729/g06-provider-profile/attribution/causal-paged-attention
```

因此 G03/G06 的“配置声称固定 V2、产品实际执行另一条不可归因路径”failure class 已关闭，
共享 G04 runtime/scheduler/resource transaction 没有重新设计；本轮只验证新 authority wiring
没有绕过它。以上均为 current-HEAD focused KEEP/PASS 诊断，不替代 stale 后必须重跑的
G08B 完整 correctness、G03/G04/G06 canonical gate 或 G09 正式性能门。下一步允许执行
profile-off CUDA 性能诊断，正式性能证据仍须先绑定当前 HEAD 的 canonical correctness。

2026-07-14 起，开发顺序和阶段依赖以
[`EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md`](EXECUTION_STRATEGY_AMENDMENT_2026-07-14.md)
为准。G00-G10 继续定义最终能力与验收维度；S0-S7 定义实际生产纵切顺序。修订不降低本文件的
三模型、双后端、正确性、性能和发布标准。该修订中的“收敛”只冻结无证据扩散和补丁堆叠；真实
模型证据若暴露正确性、性能、扩展性或产品力的系统性架构缺陷，必须中止 gate 并彻底重构根抽象，
不能用冻结策略保留错误设计。

本目标不是在现有 Architecture v2、`Backend` 大 trait、模型专属 runner 和
`run`/`serve` 分叉上继续打补丁。目标是重新设计 Ferrum 的核心推理架构，并把
已经可靠的测试、benchmark、artifact、kernel 和 release 能力收敛到新架构。

本目标的产品北极星不是“完成一次内部重构”，而是在已声明支持的单机推理场景中具备
蚕食 vLLM 市场的竞争力。设计、正确性、功能完整性、延迟和吞吐都不得用“项目规模较小”
作为降级理由；凡是主三模型矩阵声明支持的 `run`、OpenAI-compatible `serve`、streaming、
tool calling、structured output 和 multi-turn 路径，功能/正确性缺口必须为 `0`。CUDA 同机
vLLM 的逐 cell throughput ratio LCB 必须 `>=0.90`、主矩阵几何平均必须 `>=0.95`，资源压力
continuous-batching lane 必须 `>=0.95`；TTFT/TPOT p95 ratio 必须 `<=1.15`。低于这些门槛
只能形成诊断 artifact，不能以“可用”替代“有竞争力”进入发布。

完成本目标意味着实际发布 `v0.8.0`，而不是仅达到 release-ready。只有发布后的
Metal/CUDA 安装产物通过最终验证器并打印下面这一行，目标才算完成：

```text
FERRUM RUNTIME VNEXT V0.8.0 RELEASE GOAL PASS: <out_dir>
```

计划中的总验证器：

```text
scripts/release/runtime_vnext_goal_gate.py
```

该脚本只允许聚合 `run_gate.py` 产生的 manifest、校验 DAG/freshness 和打印最终 line，
不得自行重跑或重新解释业务 gate。

## 1. 为什么必须彻底重构

当前问题不是缺少一个 registry 或几条测试，而是五个核心边界同时失效：

1. 模型语义进入通用 `Backend` trait，Qwen3.5 接入新增架构命名方法和热路径
   capability 分支。
2. 模型自行复制 prefill/decode/unified runner、KV/recurrent state 和 scratch 管理，
   通用生命周期无法复用。
3. `run` 与 `serve` 独立解析 source、alias、config、preset、capability 和 runtime
   defaults，曾使 `run` 未应用 `serve` 已使用的 resolved auto-config；同一真实模型修复前后
   从约 `9.5` 提升到 `54.3 tok/s`，证明产品组合分叉会直接隐藏正确性和性能路径。
4. 测试数量很多，但真实入口、真实 feature set、真实模型和 artifact freshness 没有
   形成闭合证据图；顶层自测甚至可以遗漏已失败的子门禁。
5. CUDA 编译、GPU 正确性集成和性能定位共用一个几十分钟级反馈循环，导致 GPU 被
   当作资源状态机和架构假设的第一次完整测试。

历史量级支持彻底重构：当前 `qwen35.rs` 单文件有 `18,239` 个物理行（其中包含 test
module），六个 Qwen3.5 命名生产源码文件的物理行合计 `24,317`；production LOC 必须由
G00 按统一分类器另算。2026-06-17 至 06-26 的 git subject 中有 `255` 个匹配
`Qwen3.5|Qwen35`，其中 `13` 个含 revert/rollback；这些主题提交触达 `86` 个去重后的
`crates/`/`scripts/` Rust、CUDA、build、test 或 gate 路径。已有复盘还记录了最近 500
提交中 `qwen35.rs` 被改动 114 次、128 个 Qwen35 artifact 目录和约 40.28 小时本地 ledger。
继续局部收敛只会把现有耦合固化为下一轮重构的前置债务。

上述证据来自历史 test-architecture handoff 与 W3 Qwen3.5 retrospective；两份过程性
文档已在 2026-08-25 清理，耐久结论仍保留在各自 STATUS/GOAL 和提交历史中。
G00 必须把统计命令和结果重新写入 artifact；本段数字只用于立项，不替代 baseline PASS。

### 1.1 仓库证据快照

| 痛点 | 当前代码/历史证据 | 对应重构 Goal |
|---|---|---|
| 核心抽象泄漏模型语义 | [`Backend`](../../../crates/ferrum-kernels/src/backend/traits.rs) 文件 2,341 个物理行，当前含 32 处 Qwen35 命名符号；[`ModelExecutor`](../../../crates/ferrum-interfaces/src/model_executor.rs) 与 [`ContinuousEngine`](../../../crates/ferrum-engine/src/continuous_engine.rs) 共同分担 lifecycle | G01、G03、G04 |
| 单模型接入膨胀 | [`qwen35.rs`](../../../crates/ferrum-models/src/models/qwen35.rs) 18,239 个物理行；同名六个生产源码文件共 24,317 行 | G01、G03、G08 |
| `run`/`serve` 组合分叉 | [`run.rs`](../../../crates/ferrum-cli/src/commands/run.rs) 与 [`serve.rs`](../../../crates/ferrum-cli/src/commands/serve.rs) 分别解析 source/config；历史 handoff 记录真实 CUDA 路径 `9.5 -> 54.3 tok/s` 修复 | G05 |
| 测试晚发现 | W3 复盘记录 128 个 Qwen35 artifact 目录，GPU 逐次发现 resource-state bug；近期 500 提交中 test/docs/perf/fix 高密度交替 | G02、G04、G10 |
| 性能定位反复试错 | 同一复盘记录约 40.28 小时 ledger，多个 c32 candidate 只改变局部 trace、仍远离目标 | G06、G09 |
| CUDA 编译阻塞 | test-architecture handoff 记录 CUDA L1 cold `1906s`、warm `18s`；重型 native source 与 release LTO 扩大失效域 | G07 |
| CUDA/Metal 乒乓回归 | [`release regression hardening goal`](../release-regression-hardening-2026-06-28/GOAL.md) 已归档多次 backend 边界、资源和人工 release smoke 问题 | G02、G03、G10 |

物理行数和 symbol 次数是 2026-07-10 的只读快照，不是最终 LOC 指标；G00 analyzer 才是后续
减少比例、分类和 PASS 的唯一数据源。

### 1.2 当前核心执行链

```text
ferrum run / serve
  -> ferrum-cli source/alias/config/preset resolution
  -> ferrum-engine registry + builder
  -> ContinuousEngine + ferrum-scheduler + ferrum-kv/recurrent state
  -> ferrum-interfaces::ModelExecutor
  -> ferrum-models architecture-specific loader/executor/runner
  -> ferrum-kernels Backend traits + CUDA/Metal providers/native kernels
  -> sampler/tokenizer -> terminal or ferrum-server OpenAI/SSE response
```

`bench-serve` 经 `ferrum-bench-core::BenchReport` 测 HTTP 产品路径，release scripts 再把
unit、Metal、CUDA、tarball、Homebrew 和 completion artifact 聚合。问题不在这些模块完全不存在，
而在 model semantics、backend capability、resource lifecycle 和 product defaults 横跨多层并形成
第二真相；vNext 保留已验证实现，重建它们之间的 ownership 和 contract。

## 2. 总策略

采用四类处理方式，禁止含糊的“以后再清理”：

| 类别 | 处理方式 |
|---|---|
| 核心 contract、trait、runner、资源所有权、产品组合根 | 重新设计并替换 |
| CUDA/Metal kernel、quant 实现、`bench-serve`、`BenchReport` | 保留实现，适配新 contract |
| resource invariant、observability/replay、scenario/gate schema | 修复依赖图后收敛复用 |
| legacy factory、架构命名 Backend API、重复入口、env-only 产品策略 | 迁移完成后删除 |

迁移使用受控双轨：legacy 只能作为 baseline、shadow comparison 和回滚参考，不能进入
`v0.8.0` release binary。每个 adapter 必须声明 owner、创建 Goal、删除 Goal 和最晚删除
里程碑；没有 sunset 的 adapter 不得合并。

## 3. 优先级

1. CUDA 是第一实现和性能优化后端。
2. Metal 在核心 contract 稳定后跟进，但最终发布不允许 waiver。
3. 正确性先于性能；任一真实模型 correctness cell 失败时，对应性能数据只能标记为
   diagnostic。
4. 三个主模型优先于长尾兼容；主模型迁移成功后，再迁移或撤销其他 support row。
5. 快速开发循环优先于单次漂亮 benchmark；正式 release 构建与开发构建分离。

## 4. 主三模型

主矩阵选择新、热门且能覆盖不同执行结构的 Qwen 系模型。热度快照和精确格式见
[`MODEL_MATRIX.md`](MODEL_MATRIX.md)。

| 模型 | 角色 | 核心覆盖 |
|---|---|---|
| `Qwen/Qwen3.5-4B` | 高频 dense-hybrid canary | Gated DeltaNet、full attention、recurrent state、dense FFN；目标态低成本双端回归（当前 Metal unsupported） |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | 战略主模型和最难验收对象 | hybrid attention、recurrent state、256 experts、8 routed + 1 shared、GPTQ、资源压力 |
| `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | 成熟传统 MoE 控制组 | full attention、128 experts/top-8、QK norm、Marlin、paged KV、历史性能锚点 |

三个主模型均只认证 language path。Qwen3.5 vision tower 不在本目标范围；图片输入必须
显式返回 unsupported/4xx，禁止静默忽略并按文本请求处理。

附加但不占主三模型名额的强制 lane：

- `Qwen/Qwen3-Coder-30B-A3B-Instruct`：agent/tool XML 格式和历史 CUDA empty-answer bug kill。
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`：reasoning、特殊 EOS、think-history 和模板回归。
- `meta-llama/Llama-3.1-8B-Instruct`：只作为仓库 G0 发布政策要求的 compatibility lane，不作为产品
  优先模型，也不能替代任何主模型 PASS。

## 5. vNext 核心结构

### 5.1 稳定基础层

`DeviceRuntime` 只负责设备、buffer、stream/command、同步和错误边界。它不知道模型、
Transformer、Qwen、Llama、MoE 或 scheduler。

### 5.2 可版本化 operation contract

Attention、linear attention、MoE、quantized linear、norm、sampling 等均为独立 operation
contract。每个 operation 必须具备：

- stable operation id 与 schema version；
- shape/dtype/layout/resource contract；
- CPU oracle 或明确的高精度 reference；
- 每个支持 backend 的 provider 与 conformance test；
- 显式 unsupported 结果；
- profiler phase、resource scope 和 fault-injection point。

禁止为某个模型向通用设备 trait 添加 `qwen35_*`、`gemma_*` 一类方法。

### 5.3 模型语义层

`ModelFamilyProvider` 负责：

- 解析官方 typed config；
- 声明 weight schema 与映射；
- 构造由 semantic blocks 和 state specs 组成的 `ModelProgram`；
- 声明需要的 operation contracts；
- 提供 chat/template/EOS 元数据，不决定 backend 实现。

### 5.4 计划层

`ExecutionPlanner` 在模型加载阶段把 `ModelProgram + BackendCapabilities + RuntimePolicy`
解析为不可变 `ExecutionPlan`。计划必须可序列化、可 snapshot、可 diff，并包含每个选择、
fallback 和拒绝原因。capability 判断不得留在 token hot loop。

权重物化的数值保真度是计划契约，不是 kernel registry 的附带属性：

- 每个 materializer descriptor 必须声明 `Exact` 或 `Approximate`，计划证据同时记录 source
  dtype/quantization、selected materializer、目标 physical format 和 fidelity；
- 默认配置、正式正确性矩阵、legacy no-regression 和 release gate 只能选择 `Exact`；
  layout/repack 只有在数值无损时才属于 `Exact`；
- backend 存在更快 kernel 只证明 capability，不能授权改变 checkpoint 精度。`Approximate`
  必须由用户可见的 typed CLI/config policy 显式选择，并有独立精度预算、正确性矩阵和性能
  artifact；hidden env、backend feature 或自动 preset 不得授权；
- approximate lane 不能替代同模型原始格式的 required correctness/performance cell，也不能把
  不同数值合同的吞吐提升计入 exact no-regression；
- compiler、static initialization 和 trusted catalog 都必须 fail closed：exact policy 遇到
  approximate descriptor 时在分配设备权重前拒绝。

### 5.5 执行与资源层

共享 `ExecutionRuntime` 负责 batching、prefill/decode、layer traversal、logits、sampling
边界和清理。唯一 `ResourceTransaction` 管理 KV、recurrent state、scratch、graph workspace、
admission、commit、rollback、release。scheduler、engine 和 model 不得分别拥有半套资源真相。

资源 authority 必须分为 `Plan -> Request -> Sequence -> Step(ExecutionFrame) -> Invocation` 五个
语义 scope；Request state 可被同 request 的多个 child sequence 共享但只计费一次，Step state 跨
同一 participant frame 的多个 node，Invocation scratch 只属于一次真实 batch node/provider 调用。
continuous batch 的 Step/Invocation 必须持有 canonical non-empty participant parent set，而不是把
batch authority 临时挂在首个 sequence；实际 shape 的 `sequences` 必须等于 participant 数，batch
workspace/scratch 只 claim 一次。`BatchStepId`/`BatchInvocationId` 标识物理 scheduler step/submit，
每个 participant 另保留自身连续的 `ExecutionFrameId`/request-journal node identity；新旧请求同批时
这些本地 id 可以不同，不能投影成一个 leader identity。
每个 participant 以 owning parent hold 保活 Request/Sequence/Step，
适配 scheduler/in-flight reaper；不得用词法自引用、每 sequence stream 或合并不同 lifetime。

device execution stream 属于 scheduler/device execution lane，不属于单个 sequence。`submit` 只能
返回 `DefinitelyNotSubmitted` 或持有 typed fence 的 in-flight authority；任何 possibly-submitted
错误都必须由 fence 终态报告。一次 batch invocation 的 extent、所有 participant hold 和 Request
state hazard 由 durable completion reaper 持有到 fence quiescent；单个 sequence cancel/completion
不得 drain 共享 lane 或阻塞其他 participant。

`maximum_active_sequences` 只能是用户/协议 ceiling，不能等同于启动时预分配数量。Plan 以
`O(graph)` scoped descriptor 保存每实例资源公式；静态资源在模型加载时提交，KV、recurrent
state、request/sequence workspace 按实际请求和当前可用容量动态 claim。资源不足的请求必须
进入 typed waiting/deferred 状态，在 claim 成功前 provider encode、kernel 和 prefill submit 次数
均为 `0`；已有 decode 和其他可运行请求继续推进。每次资源释放递增 capacity-release epoch，
等待请求据此重试，禁止用模型名、GPU 名、固定并发或显存档位硬编码正常 admission。

### 5.6 产品组合层

`run` 和 `serve` 通过唯一 `ResolvedModelPlan` 进入同一个 engine。二者只保留 terminal 与
HTTP/SSE 的 I/O 适配，不得各自解析模型 alias、能力、默认值、优化开关或 chat template。

## 6. 架构硬约束

以下指标是最终状态，不是“尽量减少”：

| 指标 | v0.8.0 目标 |
|---|---:|
| 通用 trait 中架构命名方法 | `0` |
| token hot loop 中 `supports_*` / backend feature 决策 | `0` |
| model/engine 中未批准的 `cfg(cuda|metal)` | `0` |
| 核心 runtime 直接读取隐藏 `FERRUM_*` 环境变量 | `0` |
| 同一模型的产品 source/config/capability 决策实现 | `1` |
| model-owned scheduler/KV/recurrent manager | `0` |
| 正常路径按 admission ceiling 预物化 per-slot 资源 | `0` |
| 从未安装真实 backing segment 的数字发布动态 capacity | `0` |
| metadata-only 动态资源 authority 到达 provider dispatch | `0` |
| logical slice/page 重复计入全局物理显存 | `0` |
| Request state 被每 child sequence 重复 claim | `0` |
| Step state 被降级为单 node/invocation lifetime | `0` |
| Invocation scratch 被提升为 Request/Sequence/Step lifetime | `0` |
| batch scratch 按 participant 重复 claim | `0` |
| batch child capacity 仅绑定 canonical leader sequence | `0` |
| continuous batch 强制所有 participant 使用同一 request frame id | `0` |
| execution stream 绑定单个 sequence | `0` |
| dynamic provider 可见无 physical region 的 arena 裸 buffer | `0` |
| device fence 完成前复用 Invocation extent/hazard permit | `0` |
| sequence completion 通过 drain 共享 execution lane 阻塞其他请求 | `0` |
| per-request projection 重复计数同一物理 command/fence | `0` |
| capacity defer 后发生 provider/prefill submit | `0` |
| 一个资源不足请求造成全局 scheduler HOL 阻塞 | `0` |
| release binary 中 legacy executor/factory/runtime | `0` |
| silent fallback / silent default success | `0` |
| 未声明 sunset 的 compatibility adapter | `0` |
| 声明 replay-equivalent 但没有当前 provider/binary 的 CUDA 数值证明 | `0` |
| 用 graph inventory、最终文本或 tolerance 代替同实现 bitwise 证明 | `0` |

必须通过四个扩展演练：

1. 新增只使用现有 op 的 synthetic model family，核心 runtime、Backend、`run`、`serve`
   生产代码改动均为 `0`。
2. 新增一个 novel op，只修改 op contract、provider、capability catalog 和 conformance test；
   planner/runtime 主循环改动为 `0`。
3. 新增 reference backend，模型生产代码改动为 `0`。
4. 注入 unsupported capability、kernel failure、资源泄漏和坏输出，单个 artifact 必须给出
   request、plan node、operation、phase、资源状态和 first failure event。

### 6.1 CUDA vNext 确定性执行边界

CUDA reusable execution 不是单纯的性能开关。provider 一旦声明
`bitwise_eager_equivalent`，该声明必须贯穿 provider descriptor、policy selection、plan hash、
compiled wave identity、runtime binding、receipt/event/profile 和发布证据，任何层都不能推断、
补默认值或静默降级。

在同一 immutable plan/provider/runtime/device binding 下，给定完全相同的逻辑输入、显式 RNG、
初始 KV/recurrent state 和已初始化 workspace，以下三组结果必须逐字节相同：

1. eager A 与 eager B；
2. replay A 与 replay B；
3. eager witness 与 replay witness。

比较集合必须包含每个 plan node 的全部 declared outputs，以及所有 `Write`/`ReadWrite`
`PlanStateEffect` 的 post-state；只比较最终 token、文本、logits 摘要、graph 数量或 executable
inventory 均不合格。scratch 使用至少两个确定性 poison pattern 后仍必须产生相同 witness，
从而暴露未初始化读取。该 bitwise 门只证明同实现执行语义；CPU/FP32/reference parity 继续使用
G03 tolerance catalog，两类门互不替代。

统一 child gate 为：

```text
FERRUM RUNTIME VNEXT CUDA DETERMINISM PASS: <out_dir>
FERRUM GATE vnext-cuda-determinism PASS: <out_dir>
```

它由 `run_gate.py` 注册，必须在 M1/M2/M3 CUDA correctness matrix、CUDA performance 和 G10
release candidate gate 之前通过。任一 mismatch 立即停止 expensive matrix，保存 first differing
provider/node/value/state、byte offset、两侧 SHA256 和执行路径；修复后先复跑 exact case，再跑
affected provider/shape，直到下一冻结点才允许完整矩阵。

## 7. 正确性总标准

三个主模型在 CUDA 和 Metal 上都必须真实执行，不能由另一个模型、mock、stub 或
synthetic fixture 代替：

- `ferrum run`：单轮、JSONL 三轮、交互三轮、长输出、多字节 UTF-8。
- `ferrum serve`：non-stream、stream、multi-turn、六轮 stateful loop。
- known-answer `20/20`。
- legacy 可比 lane：相同 backend/format 下，冻结旧 binary 与 vNext 在 `temperature=0`
  的前 64 个生成 token 完全一致，`20/20`，PASS exception 数量 `0`；near-tie logit margin 只作
  诊断，不能把 generated token flip 改写为 PASS。
- 新增 Metal/Qwen3.5 lane：官方 HF config/tokenizer/template 是 metadata 真值；独立 CPU
  FP32/Transformers checkpoint 是 op/layer 数值真值；同 GGUF llama.cpp 只做量化端到端
  token/logit 交叉验证和性能参考。prompt tokenization 必须 `20/20` 精确；自由生成序列分歧只作
  路径诊断，数值硬门使用 canonical teacher history 对 `20 x 64 = 1280` 个 full-vocab decision
  逐项裁决并要求 `1280/1280`。三层证据均通过 [`MODEL_MATRIX.md`](MODEL_MATRIX.md) 的明确数值门，
  并绑定 checked-in `runtime_vnext_numerical_tolerances.json` row/blob SHA；artifact 不得自带或事后
  放宽 tolerance。
- required tool call `20/20`，auto tool call `20/20`，tool-result 回填 `20/20`。
- streamed tool-call delta 重组 `20/20`，arguments 必须通过声明的 JSON schema。
- strict `json_schema` `50/50`，`json_object` `50/50`。这是 Ferrum server contract，
  不是模型卡能力声明。
- stream 与 non-stream 内容、finish reason、usage 一致 `20/20`。
- 每个 stream 恰好一个 `[DONE]`、恰好一个 usage、至少一个输出 token。
- natural EOS、custom stop、`max_tokens`、context limit、cancel、timeout 全部通过。
- Qwen3.5 默认 thinking 与 `enable_thinking=false` 硬切换、Qwen3 的硬/软 thinking 切换、
  content/final/history 隔离均按 [`MODEL_MATRIX.md`](MODEL_MATRIX.md) 分别验证。
- 所有 CUDA model plan 中被选择且声明 replay-equivalent 的 provider，必须由当前
  provider implementation fingerprint、plan hash 和 binary SHA256 绑定的 CUDA determinism
  artifact 覆盖；coverage `100%`，waiver/skip `0`。
- panic、OOM、resource leak、串话、`<unk>`、`[PAD]`、U+FFFD、mojibake、特殊 token
  泄漏、missing/duplicate DONE 均为 `0`。
- CUDA client c=1/4/16/32、Metal client c=1/4/16 marker/checksum 隔离全部正确；每 cell
  必须记录 typed admission cap、observed max-active 和 active timeline。最高 required cell 的硬
  active floor 为 CUDA M1/M2/M3=`32/16/32`、Metal M1/M2/M3=`16/4/16`，eligible interval
  duty-cycle `>=0.80`；typed cap 等于 floor 时 observed max-active 必须等于 cap。floor/cap
  变化必须走 reviewed Goal amendment，不能把排队的 client c32 冒充 active c32。
- `temperature=0` 只用于 deterministic parity；另以官方推荐 sampling + fixed seed 跑用户
  默认 smoke，不要求逐 token 相等但仍要求无垃圾、正确终止、tools/schema 和资源闭合。C21
  五组各 `4`；required-tool 与 strict response format 同时出现时必须确定性选择 tool priority，
  standalone strict-schema 另行成功。

任何 required probe 被 skip、waive、placeholder 或手填为 PASS 时，总 Goal 必须失败。

## 8. 性能总标准

### 8.1 证据协议

G00 先在冻结 legacy SHA `cff4c47765ef3259b8a04890187d99c60da86394` 上采集同机
基线。正式比较必须：

- 冻结 SHA 标识 legacy `run`/`serve` product binary；HTTP 采集使用单独锁定 source/tree/binary
  SHA 的 G00 canonical `bench-serve` client。该 client 只增加 payload 和证据字段，不改变
  被测 server；同一 comparison 的 A/B 必须使用完全相同的 client binary；
- 使用同一台机器、模型 revision、文件 SHA256、dataset、seed、CLI/config 和 feature set；
- 保存 Git SHA、dirty status、binary SHA256、硬件、driver/runtime、完整命令和非空日志；
- `serve-legacy`、`serve-external`、`run-legacy`、`run-vs-serve` 按适用范围分别执行独立
  `ABBA-BAAB`，不同 comparator 不复用 A rows 或 CI；
- 每个 serve comparison 的每实现每 cell 至少 `1200` measured requests 和 12 个 repeat
  samples；run comparison 每实现 12 个真实 `ferrum run` measured samples；
- 正确性 gate 先通过；
- HTTP 吞吐只使用 `ferrum bench-serve` 与 `ferrum-bench-core::BenchReport`。
- G00 legacy `ferrum run` 只比较 JSONL 暴露的完整 `engine.infer` E2E：
  `generated_tokens * 1000 / assistant.ms`。该边界包含 prefill/decode/sampling/text、排除 load 和
  shutdown，不得改名为 TTFT 或 steady decode；G06 后另采 token-commit 指标，并保留同一 E2E
  边界用于 legacy no-regression。

指标词典必须区分两个 ITL 来源。G00 `sse_delta_events` 仅是 client 收到完整 SSE output event 的
间隔 proxy；一个 request 只有在 usage completion tokens 等于 non-empty delta event 数、interval
数等于 `tokens-1`、且 transport 没有把多个 output event 合并成一次可观察 read 时才 eligible。
任一 paired request ineligible 时对应 repeat/cell 不生成正式 client-SSE ITL ratio，不能从 eligible
子集补数。G06 `engine_token_events` 使用同一 monotonic clock 的 token-commit timestamp，是独立
指标；两种 source 禁止合并、互相改名或用 tokenizer 重切 delta 合成不存在的到达时间。

标准命令必须包含：

```text
--fail-on-error --require-ci --seed 9271 --enable-thinking false --num-prompts 100 --warmup-requests 10 --n-repeats 3
```

CUDA 固定 1x RTX 4090，覆盖 random `256/128` 与固定 ShareGPT 数据集，
`c=1/4/16/32`。Metal 固定本机 `32GB / 24-GPU-core Apple M1 Max`，M2 使用固定
Q4_K_S，覆盖 `c=1/4/16`；Qwen3.5-35B 不允许因 32GB 机器不足而 waiver。物理硬件确实无法
满足 headroom/active floor 时，只能按 MODEL_MATRIX 的 reviewed amendment policy 更换明确硬件或
format，并重采全部受影响 baseline；不能原地降低 correctness、active floor 或 performance target。

### 8.2 不回退标准

对 legacy 已支持 cell：

- 每个 cell candidate throughput 中位数 `>= legacy`，且 ratio 的 95% CI 下界 `>=0.97`；
- 全部 cell throughput 几何平均值 `>=1.00x legacy`；
- TTFT、TPOT 的 candidate 中位数不得高于 legacy，ratio 95% CI 上界 `<=1.05`；只有 A/B
  paired request 全部 eligible 时，client-SSE-event ITL 执行同一门；G06 Ferrum token-commit ITL
  必须另行采集并满足对应 no-regression；
- `ferrum run` 单请求 decode tok/s 中位数 `>= legacy`，ratio LCB `>=0.97`；
- peak device/unified memory 增加 `<=3%`；
- completed `100%`，所有错误计数为 `0`。

对原本 unsupported 的 Metal/Qwen3.5 cell，不能伪称 no-regression：

- 与同机、同 GGUF、同 workload 的 llama.cpp 比较；
- Metal 全部 required c=1/4/16 的 throughput ratio LCB `>=0.90`，主矩阵几何平均 LCB
  `>=0.95`；
- TTFT/TPOT p95 不高于 llama.cpp `1.15x`；全 paired request eligible 时 client-SSE-event ITL p95
  也不高于 `1.15x`，否则不生成该 ratio；G06 Ferrum token-commit ITL 仍为必需证据；
- 正确性先通过 reference gate。

CUDA 三个主模型还必须达到同机 vLLM 相同模型/格式/数据集 throughput LCB 的
逐 cell `>=0.90`、主矩阵几何平均 `>=0.95`。仅守住低性能 legacy 基线不足以完成目标。

Qwen3-30B-A3B 有两套独立历史向量，禁止拼成一个不存在的 baseline：0.7.7 默认路径为
`164.2 / 353.3 / 636.9 / 706.0 tok/s`；历史 FA2 direct 路径为
`160.4 / 446.3 / 1185.1 / 1641.9 tok/s`。G00 必须分别绑定 artifact、SHA、feature、preset
和命令；只有证明模型、workload、硬件和产品可见配置可比后，才允许把逐 cell 最大有效
LCB 作为绝对防倒退线。不能把两套均值直接拼接，也不能静默降低。

Qwen3.5-35B-A3B 历史 vLLM ShareGPT 均值约为 c=1/4/16/32
`136.1 / 405.4 / 1190.7 / 1708.5 tok/s`；按该 artifact 的正式 LCB 计算，80% 历史参考线为
`107.495 / 324.046 / 896.239 / 1349.917 tok/s`。这些数字不是当前同机结论，最终以 G00
新鲜 same-host LCB 为准。

## 9. 构建和开发效率标准

固定 CUDA 构建机同机运行 5 次，记录 p50/p95；p95 使用 nearest-rank，五个样本时等于最慢
样本。cache、edit/fsync、Cargo argv 到 runnable-binary smoke 的精确边界按 G07 冻结：

| 场景 | p95 上限 |
|---|---:|
| no-op / 无内容变化 | `30s` |
| Rust model leaf edit 到可运行 dev binary | `90s` |
| runtime leaf edit 到可运行 dev binary | `90s` |
| 单个 core PTX 修改到可运行 dev binary | `120s` |
| 单个 Marlin/MoE native TU 修改 | `5min`，且只重编受影响 TU |
| clean official CUDA release build | `15min` |

正式 release 继续使用 LTO；开发 profile 禁止因 release LTO/link 阻塞反馈。大体量第三方
C++/CUDA 模板源码不得继续作为普通 workspace build 输入，必须走版本化 native operator
artifact 和独立 source-build lane。

## 10. 子目标与依赖

本节 G00-G10 表保留为能力来源和 post-release hardening backlog。v0.8.0 的 release-critical
依赖、PASS 分母和执行顺序已由 2026-08-06 发布加速修订替换为 `R0 -> R1 -> R2 -> R3`；不得再要求
G00-G10 十一个 exhaustive aggregate 全部 PASS 后才进入 release freeze。

| ID | 文档 | 依赖 | 目标 |
|---|---|---|---|
| G00 | [`G00_BASELINE.md`](G00_BASELINE.md) | G00F 无前置；G00M1-M3 随模型；G00P 在 G09 前 | 事实锁、逐模型 baseline、最终完整 external/legacy/build baseline |
| G01 | [`G01_CORE_CONTRACTS.md`](G01_CORE_CONTRACTS.md) | S0A<-G00F；S0B 与 S1 同里程碑 | 拆分现有 contract，并由真实 Qwen3.5-4B CUDA `run`/`serve` consumer 收敛核心边界 |
| G02 | [`G02_TEST_EVIDENCE.md`](G02_TEST_EVIDENCE.md) | L0 随 S0；L1/impact 随 S1-S2；full 在 S6 | 分层测试、artifact 图和 historical bug kill |
| G03 | [`G03_BACKEND_OPS.md`](G03_BACKEND_OPS.md) | S1 提取最小 CUDA ops；随 S3-S5 扩展 | operation contracts、CPU oracle、CUDA/Metal providers |
| G04 | [`G04_RUNTIME_RESOURCES.md`](G04_RUNTIME_RESOURCES.md) | S0 contract split；S1 首个 production runtime | 共享 runtime、动态 scheduler、资源事务与状态所有权 |
| G05 | [`G05_PRODUCT_API.md`](G05_PRODUCT_API.md) | S1 basic composition；S2 完整 M1 产品合同 | 唯一产品组合根和 OpenAI API 语义 |
| G06 | [`G06_OBSERVABILITY_PERF_LAB.md`](G06_OBSERVABILITY_PERF_LAB.md) | basic/resource 随 S1；full kernel/replay 在 S6 前 | 定位、replay、统一 profile 和性能实验协议 |
| G07 | [`G07_BUILD_NATIVE_OPS.md`](G07_BUILD_NATIVE_OPS.md) | G07A 在 S1 后并行；G07B 随 operation catalog | crate/build graph、native ops、增量编译；S4 前完成开发反馈目标 |
| G08 | [`G08_MODEL_MIGRATION.md`](G08_MODEL_MIGRATION.md) | S1-S5 逐模型 CUDA->Metal；每模型立即删除 legacy | 三主模型逐个迁移、parity、legacy 删除和长尾处置 |
| G09 | [`G09_PERFORMANCE.md`](G09_PERFORMANCE.md) | G00P,G06,G07,G08；S6 | 三模型双端性能恢复及竞争性外部线 |
| G10 | [`G10_RELEASE.md`](G10_RELEASE.md) | G10A<-R0-R2；G10A->fresh staged correctness/performance->G10B | release freeze、候选 SHA 重验、发布、安装后回归和最终 PASS |

### Canonical gate 入口

所有 milestone completion 和最终阶段必须注册到现有 `scripts/release/run_gate.py`，不能形成一套
可独立 PASS 的 sidecar。内部 mechanical commit 运行 focused tests，不要求先写完整 collector；
S milestone 声称退出前才必须有 canonical artifact：

```text
python3 scripts/release/run_gate.py vnext-g00 --out <out>
...
python3 scripts/release/run_gate.py vnext-g10 --out <out>
```

`run_gate.py --list-lanes` 在 v0.8.0 发布前必须列出 R0-R3、G10A/G08-RC/G09-RC/G10B，
以及 G10 定义的三模型 source/published/prepromotion lanes。G00-G10、G07A/G07B、G08A-G08D 的
现有 lanes 继续作为可复用 child 或 hardening 入口，但不要求所有 exhaustive aggregate 成为 R3 前置。
stage-specific validator 可以作为内部模块存在，但有效 PASS 必须来自 `run_gate.py` 写出的统一
`gate.manifest.json`。
manifest 记录 command、SHA、dirty、binary/model/config hashes、child artifacts 和 PASS line。

所有 canonical `--out` 必须解析到 Git 源码工作树之外。Goal 文档中的 artifact tree 都是相对
`<out_dir>` 的逻辑布局，不表示 collector 应直接写入未忽略的 `docs/release/`。collector、被测
legacy worktree 和 validator worktree 必须分别保持可辨识；不能通过忽略 output path、过滤整个
`git status` 或把证据预先放进源码树来伪造 clean source。需要随仓库保存的 compact manifest 或
结论在 gate 完成后另行提交，它只能引用 canonical artifact SHA，不能替代原始 artifact。

现有 G0 lane 与 vNext 重合时只执行一次并引用同一 artifact；禁止复制数据生成两个互相
独立的 PASS。`g0_release_summary.py` 和 completion manifest 必须把三主模型矩阵设为
v0.8.0 required input，不能把它留成可漏跑的旁路。

开发 DAG 与最终证据 DAG 分开。`G00F` 是只读 inventory、historical bug catalog、模型解析与
preset 锁形成的事实 checkpoint；它解锁 S0A contract/test 结构拆分和 S1 production slice，
不代表 G00P、性能或模型迁移 PASS：

```text
G00F -> S0A contract/test structural split
  -> S0B + S1 Qwen3.5-4B CUDA basic run/serve
  -> S2 Qwen3.5-4B CUDA complete product contract
  -> S3 Qwen3.5-4B Metal + M1 legacy deletion
  -> S4 Qwen3.5-35B-A3B CUDA -> Metal + M2 legacy deletion
  -> S5 Qwen3-30B-A3B CUDA -> Metal + M3 legacy deletion

S1 -> G07A in parallel with S2/S3
per-model G00M -> corresponding legacy deletion/parity claim
G02 determinism evidence + G03 provider execution contract
  -> CUDA determinism gate
  -> per-model CUDA correctness matrix
  -> CUDA performance
release-critical G00/G06/G07/G08 inputs -> R2
  -> G10A-release-freeze / R3
  -> G08-RC + G09-RC
  -> G10B-stage-publish-promote
  -> S7/G10
```

`G10A-release-freeze` 生成唯一 `release_candidate_sha`，完成 version/release notes/workflow policy
修改、以 production workflow 冻结 staged tarball/binary SHA 并保持 checkout clean；未来 `v0.8.0`
tag 只能指向该 SHA。G08-RC 必须直接用 staged binary 在该 SHA 重跑完整六
lane correctness，G09-RC 必须在该 SHA 重跑全部正式 comparison。二者的 candidate binary SHA
必须相同，并与 staged Metal/CUDA tarball 中对应 binary SHA 一致；不一致时重新构建并重跑，不能
拼接旧 G08/G09 rows。G10B 只能消费 fresh G08-RC/G09-RC。G00 legacy binary 仍固定
`cff4c477...`，这是 comparator 身份，不受 candidate SHA 相等规则影响。

### G00F 事实检查点

G00F 的 canonical 输入沿用现有 G00a facts contract：源码工作树外的两个真实 artifact，冻结
`cff4c477...` 由当前
checked-in analyzer 重算得到的 `coupling-inventory.json`，以及 clean current HEAD 通过实时
Hugging Face HTTPS 解析得到的 `model-resolution.json`。有效入口是：

```text
python3 scripts/release/run_gate.py vnext-g00a \
  --coupling-inventory <external-cff4-inventory.json> \
  --model-resolution <external-current-head-resolution.json> \
  --out <external-g00a-out>
python3 scripts/release/run_gate.py vnext-g00f \
  --g00a <external-g00a-out>/gate.manifest.json \
  --out <external-g00f-out>
```

`vnext-g00f` 是 G00a 事实 artifact 的 freshness-bound DAG 引用，不复制或重跑同一事实
collector；它只把开发解锁范围收敛到 `S0A`/`S1`。

checkpoint 必须冻结 12/12 model/backend lane、M1-M3 四类 generation preset、16 个 historical
bug family/29 个 concrete case 的 catalog 事实和完整 analyzer/catalog/goal source identity。collector
必须先把两个外部输入读取成不可变快照，再用 checked-in resolver 重新执行一次实时 HF 解析；
调用方解析与 live recheck 的完整 model facts 必须相等，二者都作为 artifact 保存；HF
`model-info`/`repo-tree` 原始响应必须可按 SHA/size 重放，safetensors index 的 weight map、shard
集合和编号必须完全一致，不能只比较自报 request 摘要。M2 Metal Q4_K_S 的 live LFS OID/size
还必须分别等于 model catalog 的 `expected_sha256`/`expected_size_bytes`；catalog 约束不能只被复制
进 lock 而不执行。产物为 `manifest.json`、
`model-facts.lock.json`、inventory 输入快照、resolution
输入快照、live resolution recheck 和四个 catalog 副本；8 个非 manifest artifact 必须被
SHA256/size index 100% 覆盖。historical 部分明确是 `catalog_only`，不冒充 G00 要求的完整
reproducer corpus。child validator 和统一入口分别必须打印：

```text
FERRUM RUNTIME VNEXT G00A FACT CHECKPOINT PASS: <out_dir>
FERRUM GATE vnext-g00a PASS: <out_dir>
```

任一 collector contract、model catalog、resolver、inventory analyzer、模型解析请求、目标文档、
Git HEAD/tree 或 clean 状态变化都会使 checkpoint stale。策略修订后的 manifest 必须把
`S0A`/`S1` 作为唯一开发解锁项；`G00P`、正式模型迁移、性能和发布仍必须位于 `does_not_prove`。
`model-facts.lock.json` 只保存 normalized facts/fingerprint，必须在相同事实下字节确定；带
`generated_at`、绝对路径和原始输入 SHA 的采集 provenance 只进入 manifest/index。

现有 G01A artifact 保留为 historical isolated-contract checkpoint，但不再解锁生产迁移或冻结当前
contract。S0A 先保持语义拆分，S0B 必须与 S1 production consumer 同里程碑；G01 不再依赖完整
G00P。G07A 在 S1 后与 S2/S3 并行，G07B 仍消费已被真实 provider 使用的 operation catalog。
任何后续 Goal 修改已通过的核心 contract，都必须自动 invalidate 受影响的上游/模型 artifact
并重新运行，不能靠人工判断“应该没影响”。

## 11. 里程碑统筹

| 里程碑 | 包含 | 退出条件 |
|---|---|---|
| S0 合同拆分 | G00F、G01-S0A | 46K contract/test 按责任拆分，既有动态资源语义保持，focused/aggregate bounded tests 通过 |
| S1 CUDA 基础纵切 | G01/G03/G04/G05/G06 basic slice | actual Qwen3.5-4B CUDA 同时跑通 vNext `run`/`serve`、动态 admission 和 basic/resource trace |
| S2 CUDA 完整产品 | G02 core、G04/G05/G06 M1 | M1 CUDA tools/schema/stream/multi-turn/cancel/concurrency 与历史资源问题通过 |
| S3 M1 双端 | G03 Metal、G08A | 同一 program 跑通 Metal，M1 dual-backend milestone PASS，M1 legacy 已删除 |
| S4 M2 双端 | G08B、G07 | Qwen3.5-35B-A3B CUDA->Metal，c32 资源合同和开发编译目标通过，M2 legacy 已删除 |
| S5 M3 双端 | G08C/G08D | Qwen3-30B-A3B CUDA->Metal，主模型 legacy/runtime 分叉清零 |
| S6 严格证据 | G00P、G02 full、G06、G07、G08、G09 | 六 lane correctness、正式 performance、historical kill、build/profile 全部通过 |
| S7 发布 | G10A -> G08-RC/G09-RC -> G10B -> G10 | exact staged/published/installed binary 双端复验并发布 `v0.8.0` |

S0-S7 是 2026-07-14 至 2026-08-06 的历史生产纵切分解。当前发布进度和剩余关键路径必须使用
R0-R3；S6 中未被 R0-R2 明确保留的 exhaustive evidence 转入 post-release backlog。

CUDA 优先顺序：Qwen3.5-4B -> Qwen3.5-35B-A3B -> Qwen3-30B-A3B。Metal 在每个模型
CUDA contract 稳定后开始，但不得把三个 Metal lane 全部拖到发布前一次性补做。

S1 CUDA 基础纵切的 production evidence 通过统一 gate 固化：

```text
python3 scripts/release/runtime_vnext_s1_cuda_basic_collector.py collect \
  --repo <clean-source-root> \
  --model <qwen35-4b-hf-snapshot> \
  --native-operator-set-lock <complete-cuda-native-operator-set.lock.json> \
  --out <qwen35-4b-cuda-raw-artifact>

python3 scripts/release/run_gate.py vnext-s1-cuda \
  --s1-artifact <qwen35-4b-cuda-raw-artifact> \
  --out <external-out>
```

collector 必须把 schema-5 native operator lock 及其全部相对路径证据闭包复制到
raw artifact，并从该快照构建。统一 gate 只依赖 raw artifact 独立复验，不允许依赖
采集机上的外部 lock 目录，也不允许正式 bounded-overhead lane 降级接受 legacy schema。

child validator 必须从原始 `run`、`serve`、stream、`bench-serve`、scheduler trace 和 ABBA-BAAB
样本重新计算结果，不能信任手工摘要。要求 basic trace 每请求只捕获一个完整 execution frame、operation
identity 完整、terminal token 与 usage 对账、trace `<=1 MiB/request`。profile 必须由正式
`--profile-detail off|basic|debug|full` 与 artifact path 控制，默认 `off`；off slot 不得创建 profile/
scheduler-trace artifact，device completion timing 样本必须为 `0`。同一 RTX 4090 的 ABBA-BAAB
仍重算并报告均值/中位数开销、两组 CV 与硬件 telemetry，但 `<=2%` overhead 和 `<=5%` CV 是
profile-on 质量目标，不再阻塞 S1。默认关闭路径的真实性能回归由 G09 的 legacy/vLLM 同硬件门负责，
不得用 profile lane 的高方差替代或豁免。精确 PASS 行为：

collector 固定 workload、seed、repeat 和 slot order，不暴露性能口径参数；每个 slot 必须保存 bench
前、bench 中至少 `3` 个样本和 bench 后的同一 GPU UUID/P-state/graphics-SM-memory clocks/power/
temperature/utilization/memory 以及 host CPU ticks/load/memory/swap。telemetry 只观测、不得设置 clocks、
power limit 或产品隐藏环境变量。首四 slot 的 `overhead.first-half.json` 只保存当时的真实诊断状态，允许
PASS 或 REJECT；正式 validator 必须重算它，禁止要求、伪造或复用历史 noisy REJECT。

```text
FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS: <out_dir>
FERRUM GATE vnext-s1-cuda PASS: <out_dir>
```

该 checkpoint 只解锁 G01B 的 production-reference 重构；它不证明 S1 里程碑、G01B、aggregate
G01、full G06、完整模型迁移或发布完成。S1 仍须由 G01B 中的共享动态 admission/backpressure 和
零 legacy runtime fallback 证据闭环。

S1 的共享动态 admission/backpressure 使用独立的有界 CUDA capacity-pressure lane，避免为一次
容量语义诊断重跑 ABBA/BAAB 性能 sweep：

```text
python3 scripts/release/runtime_vnext_s1_cuda_capacity.py collect \
  --binary target/release/ferrum --model <qwen35-4b-model-dir> --out <raw-out>
python3 scripts/release/run_gate.py vnext-s1-cuda-capacity \
  --s1-artifact <raw-out> --out <external-out>
```

collector 必须先在同一 clean SHA/binary 上通过真实 `ferrum run`，再用 `serve` 校准 A+C 的已安装
backing 并通过 typed `--runtime-memory-budget-bytes` 重放精确预算。压力序列固定为 A active、B 先到
但 `WaitForRelease`、C 后到且先完成；B 在 unchanged epoch 下 probe/submit 增量为 `0`，A 在等待窗
继续 decode，release epoch 前进后 B admission/submit/completion。stream 必须各有且仅有一个
`[DONE]` 和 usage，最终 active/queued/pending/maintenance 均为 `0`。精确 PASS 行为：

```text
FERRUM RUNTIME VNEXT S1 CUDA CAPACITY PRESSURE PASS: <out_dir>
FERRUM GATE vnext-s1-cuda-capacity PASS: <out_dir>
```

该 lane 只证明当前 SHA 的 Qwen3.5-4B CUDA 容量压力纵切，不单独证明 G01B、S1、性能或发布完成。

decode execution-capacity 使用独立的补充 lane，证明 active decode 在 Step/Invocation 动态容量
耗尽时不会热循环、互相等待或依赖某个请求先终态：

```text
python3 scripts/release/runtime_vnext_s1_cuda_decode_capacity.py collect \
  --binary target/release/ferrum --model <qwen35-4b-model-dir> --out <raw-out>
python3 scripts/release/run_gate.py vnext-s1-cuda-decode-capacity \
  --s1-artifact <raw-out> --out <external-out>
```

collector 必须在一个有界 A/B/C 压力序列中观察 wide work cohort split、exact-source park、typed
yield/recompute transaction 和 fence-delayed release。每个 pressure boundary 必须由 scheduler-owned
transition ordinal 证明：至少一个 logical work frontier 获得可执行 claim，或存在一个尚未完成的
`YieldPlanned/AwaitReleaseFence`；禁止出现 overlapping exact source 上
`all live frontiers blocked + no pending release`。unchanged source 的 allocator/admission probe 增量
为 `0`，无压力路径不得创建 pressure episode、改变 batch membership 或增加 host allocation。

同一 target server 还必须把 plan-owner 跨池回收与上述 decode 压力分成两个因果窗口。先复放
target-sizing workload，直到 quiescent pool snapshot 恰好占满校准后的全局动态预算；再发送一个
仍处于 `max_model_len` 内、但 token demand 高于 A/B/C 的真实 stream 请求，只接受带非零
`pools_reclaimed`、`chunks_reclaimed` 和 `reclaimed_bytes` 的 typed maintenance receipt。随后才运行
A/B/C decode-pressure 序列。validator 必须分别从 rebalance-probe 和 decode 窗口重算两类证据，
禁止用任意历史 maintenance 事件补足 decode 结果，也禁止硬编码 pool hash、domain id、GPU 名称或
显存档位来制造回收。

`DecodeProgressLease` baseline/release trace 可以保留为迁移期诊断，但不能再作为该 lane 的充分
PASS 条件。正式 validator 必须消费统一 logical frontier、pressure episode、resource transaction 和
ordered transition journal；不能依赖 wall-clock event 顺序，也不得用 token 阈值、时间、模型、GPU、
显存档位代替 capacity/cost contract。A/B/C 最终各有且仅有一个 `[DONE]` 和 usage，任一角色连续
30 秒无 token progress 时 REJECT，所有 scheduler/resource ownership 和 pending fence 清零。精确
PASS 行为：

```text
FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY PASS: <out_dir>
FERRUM GATE vnext-s1-cuda-decode-capacity PASS: <out_dir>
```

该补充 lane 是 correctness-only 证据，不执行性能 sweep，也不单独证明 G01B、S1、性能或发布完成。

`da9c1ee8363c686e71420fd5df8042c496e69757` 的 1x RTX 4090/Qwen3.5-4B collection 是
`cross_phase_capacity_progress_deadlock` REJECT：lease 在 generation `49 -> 50` 后解除，但最终
snapshot 为 `active=2`、`blocked prefill=1`、`blocked decode=1`，共同等待 domain `4` generation
`73`；A/B/C content 为 `81/33/16`，仅 C 有 `[DONE]`，B progress timeout `30.010s`。该 artifact
否定 phase-local lease 方案并强制上述 unified frontier/pressure transaction 重构；不得原样重跑或
通过提高 timeout 改写结果。

## 12. 分支、提交和停止规则

- 每个子 Goal 使用小而可审阅的提交；核心 contract、kernel 优化、release gate 大改不得混在
  同一个 patch。
- S0A 先做保持语义的 mechanical split；S0B 才允许与 production consumer 同里程碑做 breaking
  semantic rewrite。除修复已复现的 runner blocker外，连续 gate-only commit 不得超过 `2` 个。
- 长期分支提交前执行 `git pull --rebase --autostash`，验证后及时 push。
- correctness 失败时停止性能 sweep。
- 同一 paid GPU failure class 连续两个 REJECT 后，必须回到 source/artifact 分析。
- 每个性能候选只验证一个主要假设，必须预先写 expected signal 和 reject threshold。
- 双轨期间 legacy/vNext 差异必须有 artifact；不能用输出过滤掩盖 token 或状态错误。
- 任何子 Goal 的 PASS artifact 在受影响代码变化后自动 stale。

## 13. 最终发布条件

G10 必须实际完成：

1. workspace version 升到 `0.8.0`，迁移说明和 release notes 完整。
2. 三个主模型在 Metal/CUDA 上 correctness 与 performance 全过，且 final G08-RC/G09-RC 均
   绑定未来 tag SHA 和 staged/published binary SHA。
3. Llama 8B-class dense supplemental release evidence 满足仓库政策。
4. unit、Metal、CUDA、tarball、Homebrew、release summary 和 completion gate 全过。
5. GitHub tag/release、正式资产和 checksum 已发布。
6. workspace crates 已按依赖顺序发布，crates.io 可查询 `0.8.0`，clean
   `cargo install ferrum-cli --version 0.8.0 --locked` 通过。
7. 从已发布 tarball/Homebrew 安装的 binary 再运行 `run` 与 `serve`。
8. 最终 `G0 RELEASE PASS: docs/release/g0/0.8.0` 存在。
9. `FERRUM RELEASE COMPLETION PASS` 引用发布后的资产和安装验证。
10. Docker 当前不维护：v0.8.0 Docker image/tag 发布数量 `0`，现有 tag-trigger workflow 已禁用。

禁止在第 5 步之前打印总 Goal PASS，也禁止把 source-ready、release-ready 或 staged asset
描述为已经发布。
