# Runtime vNext v0.8.0 正确性矩阵收敛修订（2026-08-07）

## 状态与效力

- 状态：Active。
- 本修订只改变 R1 中跨模型重复用例的采样数量，优先于
  [`MODEL_MATRIX.md`](MODEL_MATRIX.md) 和
  [`RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md`](RELEASE_ACCELERATION_AMENDMENT_2026-08-06.md)
  中要求每个主模型重复执行同一完整分母的条款。
- 三个主模型、CUDA/Metal 双后端、C01-C21 场景、产品入口、正确性 oracle、资源不变量、
  provider conformance 和 `production_legacy_selection_count=0` 均不删减。
- legacy baseline 的冻结 703/702/783/782 分母不变。本修订只作用于
  `g08-model-matrix-v1` candidate contract。

## 决策

三个主模型不能互相替代：

| 模型 | 不可由其他模型覆盖的结构 |
|---|---|
| M1 Qwen3.5-4B | dense FFN、DeltaNet/recurrent state、稀疏 full attention、BF16 CUDA 主路径 |
| M2 Qwen3.5-35B-A3B | DeltaNet + full attention、routed + shared expert MoE、GPTQ/Marlin、最大资源压力 |
| M3 Qwen3-30B-A3B | 全层 full attention、routed-only MoE、Qwen3 soft thinking、GPTQ/Marlin 稳定控制组 |

因此不删除模型，而采用“公共协议全矩阵 + 架构差异矩阵”：

- M1 是公共产品/API canary，继续执行 CUDA `703`、Metal `702`。
- M2 只压缩跨模型重复采样，执行 CUDA `112`、Metal `111`。
- M3 只压缩跨模型重复采样，执行 CUDA `120`、Metal `119`。
- 六 lane 总分母从 `4375` 降为 `1867`，减少 `2508` cases（`57.3%`）。
- 已通过 M1 CUDA 后，剩余 CUDA 分母从 `1486` 降为 `232`，减少 `84.4%`。

## M2/M3 精确分母

| 场景 | M2 | M3 | 保留的硬覆盖 |
|---|---:|---:|---|
| C01 | 20 | 20 | config、template、special token、unknown fail-closed 各 5 |
| C02 | 4 | 4 | known-answer 与 natural EOS |
| C03 | 2 | 2 | multi-turn，逐组 3 rounds |
| C04 | 1 | 1 | `>=512` output tokens |
| C05/C06 | 4/4 | 4/4 | non-stream/stream 成对、唯一 `[DONE]`、usage、重组等价 |
| C07 | 2 | 2 | 两个隔离会话，各 5 turns |
| C08 | 6 | 6 | stop、natural EOS、max-tokens 各 2 |
| C09 | 6 | 6 | cancel、timeout、disconnect 各 2，释放后容量复用 |
| C10-C13 | 每项 4 | 每项 6 | no-thinking/thinking；M3 额外保留 soft-think/soft-no-think |
| C14 | 8 | 8 | required/type/additional-properties/enum x no-thinking/thinking |
| C15 | 4 | 4 | json-object x no-thinking/thinking |
| C16 | 5 | 5 | 五类 invalid request 各 1 |
| C17 | 6 | 6 | Chinese/emoji/combining x run/serve |
| C18 CUDA/Metal | 4/3 | 4/3 | CUDA c1/c4/c16/c32；Metal c1/c4/c16，资源 trace 全保留 |
| C19 | 10 | 10 | 五种 thinking mode x run/serve |
| C20 | 5 | 5 | image/data/video/mixed 显式拒绝 + text-array positive |
| C21 | 5 | 5 | run plain、serve stream、required tool、strict schema、json-object |

所有 case 必须 PASS；`known-fail`、`blocked`、skip、waiver、error、unexpected 均为 `0`。
M2/M3 的旧 703/702/783/782 candidate 重复矩阵转入 v0.8.1/0.9 nightly hardening，
不得被报告为 v0.8.0 已完成工作。

## 失败后的执行策略

1. 首次失败只运行 exact case reproducer。
2. exact case 通过后运行 affected scenario 和同架构 sentinel。
3. 只有达到新的 R1 milestone 才重新运行该 lane 的 112/111/120/119 矩阵。
4. 不因单 case 失败重启 M1 的 703/702 或其他 backend 的完整 lane。
5. release candidate staged binary 复验使用同一 R1 分层矩阵，不恢复旧的跨模型重复分母。

## 证据复用边界

M1 已通过工件只有在记录 SHA 是当前 SHA 的祖先，且中间变化严格限于本修订的文档与 R1
矩阵控制面文件时才可复用。任何 `crates/`、Cargo、模型锁、运行配置、产品场景 manifest 或生产
实现变化都会使工件 stale。M2/M3 工件必须与当前 source 完全一致。

