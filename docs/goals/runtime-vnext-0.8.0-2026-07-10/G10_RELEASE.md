# G10: v0.8.0 发布、资产与安装后回归

## 状态与依赖

- 状态：Open
- G10A 依赖：G00-G09 development PASS，且在 release-freeze 前未 stale
- G08-RC/G09-RC 依赖：G10A；二者必须在 G10A 的 release-candidate SHA 和 exact staged binary 上重跑
- G10B 依赖：fresh G08-RC、G09-RC 和 staged asset gates
- G10 aggregate 依赖：G10A、G08-RC、G09-RC、G10B 全部 PASS
- 这是总 Goal 最后一项

## 目标

实际发布 Ferrum `v0.8.0`，并用用户最终获得的 Metal/CUDA binary 对三个主模型重新运行
正确性和性能。source gate、staged asset 或 release-ready 不等于完成。

## G10A：发布候选冻结

1. `git pull --rebase --autostash`，确认进入 freeze 前 G00-G09 development manifest 未 stale。
2. workspace 全部 crate version 升到 `0.8.0`。
3. 生成 breaking-change migration guide、release notes、support matrix 和性能报告。
4. 改造并自测 `.github/workflows/release.yml`、`release-cuda.yml` 和 `docker.yml`：`v0.8.0`
   tag 只能创建 `prerelease=true` 的 GitHub release；Docker 当前不维护，必须移除/禁用 `v*`
   tag 发布触发，禁止推送 `0.8.0`、`stable`、`latest` 或任何 candidate image。单独的
   post-validation promotion workflow 必须只接受下文 `runtime-vnext-prepromotion` PASS artifact，
   并只把 GitHub release 置为 `prerelease=false`。
5. pre-tag workflow policy gate 必须解析 workflow YAML 并以 negative fixture 证明：直接正式
   release、任何 Docker publish/tag job、缺 prepromotion child、SHA 不一致或重复使用 manifest
   都会失败。
6. 将上述变化提交为唯一 clean release-candidate commit；G10A manifest 保存
   `release_candidate_sha`/tree，未来 annotated `v0.8.0` tag 只能指向该 SHA。freeze 后任何 source、
   Cargo metadata、lockfile 或 workflow 变化都必须废弃 G10A，并回到新的 G10A SHA。
7. 从该 SHA 使用 production workflow、`publish_release=false` 构建 Metal/CUDA/CPU staged
   tarball，保存 tarball/binary SHA256。G08-RC/G09-RC 必须直接使用 staged Metal/CUDA tarball 中
   的 binary；G10B 发布同一 tarball bytes，禁止验证后重新编译。

有效命令与必需 line：

```text
python3 scripts/release/run_gate.py vnext-g10a \
  --g09 <g09-development-manifest> \
  --out <external-out>

FERRUM RUNTIME VNEXT G10A RELEASE FREEZE PASS: <out_dir>
FERRUM GATE vnext-g10a PASS: <out_dir>
```

G10A 只证明 release candidate SHA、workflow policy 和 staged bits 已冻结，不证明正确性、性能或
发布完成。release commit 后不得复用 G08/G09 development candidate rows；只允许继续使用固定
`cff4...` legacy comparator、model/download cache、dataset 和 external binary bits。

## Source gates

必须依次执行并保存 artifact：

- `cargo test --workspace --all-targets`。
- format、clippy advisory、validator self-tests。
- backend boundary、runtime preset、scenario、model contract、observability、native op。
- G10A staged Metal binary 的 source/binary gate，然后 G08-RC/G09-RC Metal 三主模型 gate。
- G10A staged CUDA binary 的 source/binary gate，然后 G08-RC/G09-RC CUDA 三主模型
  correctness/performance gate。
- Llama 8B-class dense supplemental CUDA/Metal evidence。

任一 correctness gate 失败时停止 release，不运行发布性能或 asset publication。

## Paid CUDA 发布合同

G10 的 G08-RC/G09-RC、staged binary gate 和 published-asset CUDA 验证分别是独立 paid lane。
每条 lane 开始前
都必须重新 inventory 并记录 instance、小时价、预计 runtime/cost、commands 和 stop
condition；不能引用 G00/G09 的旧成本声明。

- 优先复用 G09 retained stopped 4090 和模型/build cache，同时最多 1 个 billable instance。
- G08-RC/G09-RC 合并的三模型 full lane 与 published full lane 各最长 `5h`；staged binary gate
  最长 `1h`；三者总计上限
  `12 GPU-hours`。
- 任一 correctness failure 立即终止当前及后续性能/发布动作。
- G08-RC/G09-RC 与 published tarball 必须各自实际执行完整 correctness，并为每个 required
  `comparison_id` 重新执行 same-host `ABBA-BAAB`；不能只复用 external/legacy 的 A rows。
- staged tarball 先过 binary/dependency/version gate；随后 G08-RC/G09-RC 直接执行 tarball 内 binary。
  G10B 只允许发布同一 tarball SHA 和 binary SHA。任一 SHA 不同即废弃 G08-RC/G09-RC 并重新
  冻结/全量执行，不能引用部分 rows 或把另一个 source-built binary 当作等价物。
- G08-RC/G09-RC 与 published 之间不得复用 measured comparison；只可复用模型/download cache、dataset
  和相同 external binary bits，external server 仍须在各自 outer slots 重新运行。
- 超过上限需要用户明确批准；不允许因为 release 已开始而自动续费。
- 每条 lane结束 copy back artifacts并确认实例 `actual_status=exited`。

## G08-RC/G09-RC 最终 source matrix

M1/M2/M3 x Metal/CUDA 必须重新执行：

- [`MODEL_MATRIX.md`](MODEL_MATRIX.md) C01-C21；
- run + serve + stream + stateful；
- tools required/auto/stream delta/tool result；
- strict schema/json object；
- concurrency isolation；
- G09 正式性能 workload；
- candidate 与 G00 baseline、vLLM/llama.cpp ratio。

六个 lane PASS `6/6`，waiver/skipped/stale `0`。

G08-RC/G09-RC 的 `source_git_sha` 必须等于 G10A `release_candidate_sha`，dirty=false；两者
Metal/CUDA candidate binary SHA 必须分别相等，并等于 G10A staged tarball 内 binary。G09-RC
可以重新启动固定 G00 legacy comparator，但 G00 的 `source_git_sha=cff4...` 不参与 candidate SHA
相等检查。任何 candidate binary/source/config hash 不同都必须使两项 RC artifact stale。

必须新增并注册 canonical lanes：

```text
python3 scripts/release/run_gate.py runtime-vnext-metal-three-model \
  --g10a <g10a-manifest> --g08-rc <g08-rc-manifest> --g09-rc <g09-rc-manifest> --out <out>
python3 scripts/release/run_gate.py runtime-vnext-cuda-three-model \
  --g10a <g10a-manifest> --g08-rc <g08-rc-manifest> --g09-rc <g09-rc-manifest> --out <out>
python3 scripts/release/run_gate.py runtime-vnext-published-assets \
  --g10a <g10a-manifest> --g08-rc <g08-rc-manifest> --g09-rc <g09-rc-manifest> --out <out>
python3 scripts/release/run_gate.py runtime-vnext-prepromotion \
  --published-assets <published-assets-manifest> --out <out>
```

`run_gate.py --list-lanes`、`g0_release_summary.py` 和 release completion validator 都必须
把这四项列为 v0.8.0 required input。旧 `metal`/`cuda-full` 中相同 SHA/binary/model/config
的 row 直接引用同一 child artifact，不重复执行也不复制 summary。

四个 lane 的 child 与 canonical outer line 均为硬合同：

```text
FERRUM RUNTIME VNEXT THREE MODEL METAL SOURCE PASS: <out_dir>
FERRUM GATE runtime-vnext-metal-three-model PASS: <out_dir>
FERRUM RUNTIME VNEXT THREE MODEL CUDA SOURCE PASS: <out_dir>
FERRUM GATE runtime-vnext-cuda-three-model PASS: <out_dir>
FERRUM RUNTIME VNEXT PUBLISHED ASSETS PASS: <out_dir>
FERRUM V0.8.0 THREE MODEL METAL CUDA RELEASE PASS: <out_dir>
FERRUM GATE runtime-vnext-published-assets PASS: <out_dir>
FERRUM V0.8.0 PREPROMOTION PASS: <out_dir>
FERRUM GATE runtime-vnext-prepromotion PASS: <out_dir>
```

## Staged assets

G10A 在打 tag 前使用 production workflow 且 `publish_release=false` 构建并冻结：

- macOS aarch64 Metal tarball；
- Linux x86_64 CUDA sm89 tarball；
- Linux x86_64 CPU tarball（现有产品资产）；

每个 tarball 有 adjacent SHA256、version、dependency/ABI manifest。对 staged Metal/CUDA
tarball 运行 release binary gate，再由 G08-RC/G09-RC 直接用 tarball 内 binary 执行主三模型
run/serve correctness 与 full comparison。G10B 发布必须复用同一 tarball bytes；workflow
重新编译、tarball SHA 或内部 binary SHA 变化一律使 RC 证据 stale。不能用 source-built binary
代替实际 staged product path。

## G10B：发布、验证与提升

1. G10A staged assets、G08-RC 和 G09-RC 全过后，为全部 publishable workspace crates 生成
   `.crate` package，在 ephemeral
   local registry/index 中按 `cargo metadata` 拓扑顺序发布，并从打包内容 clean build/test；这一步
   验证下游的内部 `0.8.0` dependency，不假设 crates.io 已存在该版本。
2. 创建 annotated `v0.8.0` tag；tag target 必须精确等于 G10A `release_candidate_sha`。先发布
   GitHub prerelease，上传的资产 bytes/SHA 必须精确等于 G10A staged assets，禁止 rebuild。
3. 用 prerelease URL/资产完成发布后 tarball 三模型验证；失败时保留不可变 tag/asset 作为
   failed prerelease，修复进入 `v0.8.1`，禁止删除重建同名 tag。
4. prerelease 资产通过后，按依赖顺序对每个 crate 串行执行：
   `cargo publish --dry-run --locked` -> `cargo publish --locked` -> 轮询 crates.io API/index 和
   clean dependency resolution，成功后才进入下一个 crate。确认 crates.io 可查询全部 `0.8.0`
   crate，并在 clean environment 执行：

   ```text
   cargo install ferrum-cli --version 0.8.0 --locked
   ferrum --version
   ferrum --help
   ```

5. 更新 Homebrew Metal formula 和 CUDA fetch path/checksum。
6. crates.io、Homebrew 和 installed-asset 验证完成后，运行 `runtime-vnext-prepromotion`。
   它必须聚合同一 release id/tag SHA 下的 `runtime-vnext-published-assets`、crates.io、Homebrew
   Metal、Homebrew CUDA fetch 和 workflow-policy PASS，并绑定正式 asset SHA256；缺 child、
   stale、SHA 不一致或 manifest 已消费都必须失败。
7. promotion workflow 执行前再次确认 `draft=false`、`prerelease=true`，原子记录 manifest
   consumed 状态后把 GitHub prerelease 提升为正式 release。
8. promotion 后重新读取 GitHub release，证明 `prerelease=false` 且 asset ids/SHA256 未变化，
   再运行 final `g0_release_summary.py` 和 release completion validator；promotion 前的 dry-run
   不得打印 `G0 RELEASE PASS`。
9. 生成 GitHub release URL、asset ids、sizes、SHA256、workflow run ids、crates.io versions
   和 Homebrew commit artifact。

全部动作完成后的 canonical 聚合命令与 line：

```text
python3 scripts/release/run_gate.py vnext-g10b \
  --g10a <g10a-manifest> \
  --g08-rc <g08-rc-manifest> \
  --g09-rc <g09-rc-manifest> \
  --published-assets <published-assets-manifest> \
  --prepromotion <prepromotion-manifest> \
  --out <external-out>

FERRUM RUNTIME VNEXT G10B PUBLISHED RELEASE PASS: <out_dir>
FERRUM GATE vnext-g10b PASS: <out_dir>
```

G10B validator 必须重新读取 GitHub tag/release、crates.io 和 Homebrew 状态；tag target 不等于
G10A SHA、published asset SHA 不等于 staged SHA、G08-RC/G09-RC binary SHA 不等于 asset 内
binary、release 仍为 prerelease 或任一 manifest 已被其他 release id 消费时都必须失败。

crates.io 中途失败时停止后续发布。只有 auth/index/network 一类不改变任何 `.crate` bytes、
tag 或 asset 的外部失败，才可重试同一 `0.8.0`；tag 后任何 source/package metadata/content
修复，无论对应 crate 是否已发布，都必须提升到 `0.8.1`。已发布 crate 永远不能覆盖。

## 发布后验证

### Metal

- 从 GitHub tarball 安装并跑三主模型 correctness + performance。
- 从 Homebrew 安装并至少跑 M1 快速全 API、M2/M3 run/serve smoke；正式性能使用与
  tarball 相同 binary SHA 时可引用 tarball artifact。

### CUDA

- 从 GitHub CUDA tarball 安装到 clean runtime host。
- 检查无 Python/Torch/vLLM runtime linkage、缺库和错误 SM。
- 三主模型 correctness + performance 全量重跑。
- `homebrew-cuda-fetch` 只证明 URL/checksum/fetch，不得描述为 CUDA runtime install。若项目
  提供可运行的 Linuxbrew CUDA formula，必须在真实 CUDA host 安装并跑 M1 API smoke；
  否则 CUDA 运行时产品验证明确由 release tarball lane 承担。

prerelease 后任一主模型错误或性能不达标：保留 failed artifact/tag，停止提升为正式版并
发布修复版本；不得删除/重写 tag，也不得只修改 README 掩盖。

## 既有与新增 release gates

必须产生：

```text
FERRUM GATE vnext-g10a PASS: <out_dir>
FERRUM GATE vnext-g08-rc PASS: <out_dir>
FERRUM GATE vnext-g09-rc PASS: <out_dir>
FERRUM GATE runtime-vnext-metal-three-model PASS: <out_dir>
FERRUM GATE runtime-vnext-cuda-three-model PASS: <out_dir>
FERRUM GATE runtime-vnext-published-assets PASS: <out_dir>
FERRUM GATE unit PASS: <out_dir>
FERRUM GATE metal PASS: <out_dir>
FERRUM GATE cuda-full PASS: <out_dir>
FERRUM GATE cuda-llama-dense PASS: <out_dir>
METAL TARBALL GATE PASS: <out_dir>
CUDA TARBALL GATE PASS: <out_dir>
HOMEBREW METAL GATE PASS: <out_dir>
HOMEBREW CUDA FETCH GATE PASS: <out_dir>
FERRUM RELEASE WORKFLOW POLICY PASS: <out_dir>
FERRUM CRATES IO V0.8.0 PASS: <out_dir>
FERRUM V0.8.0 PREPROMOTION PASS: <out_dir>
FERRUM GATE runtime-vnext-prepromotion PASS: <out_dir>
FERRUM GATE vnext-g10b PASS: <out_dir>
FERRUM GATE vnext-g10 PASS: <out_dir>
G0 RELEASE PASS: docs/release/g0/0.8.0
FERRUM RELEASE COMPLETION PASS: <out_dir>
```

还必须新增三模型发布后聚合 line：

```text
FERRUM V0.8.0 THREE MODEL METAL CUDA RELEASE PASS: <out_dir>
```

该 line 必须由 `runtime-vnext-published-assets` lane 产生，并被 canonical
`G0 RELEASE PASS` 与 completion manifest 强制消费；它不是可独立忽略的 sidecar。

## 最终验收

- GitHub `v0.8.0` release 已存在，且 `draft=false`、`prerelease=false`。
- G10A/G08-RC/G09-RC/G10B 四个 manifest fresh；tag target 精确等于 G10A
  `release_candidate_sha`，G08-RC/G09-RC candidate binary SHA 与 staged/published tarball 内
  Metal/CUDA binary SHA 全部相等。
- final manifest 校验 GitHub release id、tag SHA、published timestamp 和正式 asset ids；
  任一字段仍指向 prerelease/staged asset 时不得 PASS。
- Metal/CUDA/CPU 资产和 checksum 完整。
- 三主模型安装后二端 correctness/performance `6/6 PASS`。
- Llama supplemental evidence PASS。
- Homebrew Metal/CUDA fetch PASS。
- crates.io workspace versions 完整，`cargo install ferrum-cli --version 0.8.0 --locked` PASS。
- release summary/completion manifest 引用发布后 asset，而不是 staged path。
- release source checkout 的 repo/worktree/tag SHA 一致且 dirty=false；repo 外 evidence workspace
  由 manifest 绑定该 SHA，不参与 source dirty 判定。
- release notes 不声明 vision/Qwen3.5 multimodal 支持。
- release notes 明确 v0.8.0 不发布、也不承诺官方维护的 Docker distribution；Docker
  asset/tag 数量 `0`。
- 所有 performance claim 带硬件、命令、SHA、binary hash、config 和 artifact link。

## 产物与最终 PASS

以下目录是 repo 外 immutable evidence workspace 中的逻辑布局，不是 release source checkout
里的未提交文件。为兼容既有 G0 validator，该 evidence workspace 内仍保留相对路径
`docs/release/g0/0.8.0`，并从该 workspace 产生要求的同名 PASS line；release source/tag checkout
始终 clean，manifest 单独绑定其 tag SHA。发布后 evidence 不能回写并改变已发布 tag。

```text
<evidence-workspace>/docs/release/g0/0.8.0/runtime-vnext-final/
  subgoals.json
  g10a-release-freeze/
  g08-release-candidate/
  g09-release-candidate/
  three-model-source-matrix.json
  staged-assets/
  published-assets.json
  prepromotion.json
  installed-metal/
  installed-cuda/
  homebrew/
  crates-io/
  g10b-published-release/
  release-completion.json
```

G10 PASS：

```text
FERRUM RUNTIME VNEXT G10 V0.8.0 RELEASE PASS: <out_dir>
FERRUM GATE vnext-g10 PASS: <out_dir>
```

总 Goal 最终 PASS：

```text
FERRUM RUNTIME VNEXT V0.8.0 RELEASE GOAL PASS: <out_dir>
```
