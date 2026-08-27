# Qwen3.8-27B CUDA 最小产品闭环目标

## 状态与完成口径

- 2026-08-26 提议；2026-08-27 按单人资源和 v0.8.0 复盘收敛，并修正为同架构小模型
  优先验证。
- 这是一个 **model-adoption source goal**，不是发布审计、性能竞赛或正式发布目标。
- 文档合并、代码能编译、单元测试通过、模型开始输出，都不单独代表目标完成。
- 本目标不授权创建 release、tag、Homebrew 更新或付费 GPU 实例。

只有最终 artifact 目录存在，且薄 validator 的最后一行完全等于下面内容时，才能标记完成：

```text
QWEN38 CUDA ADOPTION GOAL PASS: <out_dir>
```

这条 PASS 只证明一个固定 Qwen3.8 checkpoint 的 CUDA 产品闭环，不代表官方
release-ready。正式发布仍须另行执行 `AGENTS.md` 的 G0 source、CUDA full、Llama dense、
release asset 和 release summary gates。

## 一句话目标

让 Ferrum 在单张 RTX 4090 上原生运行固定 revision 的
`cyankiwi/Qwen3.8-27B-AWQ-INT4`，准确支持它实际使用的
`compressed-tensors` W4/group32/asymmetric/mixed-dense 合同，并让
`ferrum run` 与 `ferrum serve` 都完成真模型生成；日常 CUDA 编译和小张量验证全部放在
`panda-pad` WSL 环境。

## 为什么只做这一项

以下数据是 2026-08-26 的时间点快照。Hugging Face downloads 不是独立用户数，也不能把
多个 repo 简单相加；计数口径见
[Hugging Face 文档](https://huggingface.co/docs/hub/en/models-download-stats)。

| 候选 | 可核验需求信号 | Ferrum 当前复用/缺口 | 决策 |
|---|---:|---|---|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | official 3,298,569 downloads；2026-08-05 发布 | 已有 Qwen3.5 hybrid text core；缺正式 Qwen3.8 config、格式 loader 和产品 gate | **当前唯一交付** |
| [Qwen3.8-27B AWQ INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) | 752,710 downloads | 24GB 级 CUDA 用户可运行，但其 `compressed-tensors` layout 当前不能由 GPTQ loader 正确读取 | **固定 checkpoint** |
| [Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 7,638,591 downloads | Metal/GGUF 有基础，但 CUDA GGUF kernel 与 Metal 新模型资格会扩大范围 | 后续独立候选 |
| [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | official 837,131；热门 GGUF repo 12,687,574 | Metal 已通过；CUDA GPTQ 是已有 checkpoint 的空回答故障 | 后续独立故障 goal |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 6,950,594 downloads | 同时缺架构、Harmony 行为和 MXFP4 | 暂不做 |

GitHub Issue 用来确认真实集成场景，不把 bug 数量当用户数：

- Qwen3.8 已在 [vLLM #52564](https://github.com/vllm-project/vllm/issues/52564)、
  [vLLM #53887](https://github.com/vllm-project/vllm/issues/53887)、
  [SGLang #36048](https://github.com/sgl-project/sglang/issues/36048) 和
  [llama.cpp #27615](https://github.com/ggml-org/llama.cpp/issues/27615) 出现 INT4、单卡、
  tool-call 和性能问题，说明部署场景真实存在。
- Responses API 的兼容需求也很明确，见
  [vLLM #14721](https://github.com/vllm-project/vllm/issues/14721)、
  [Ollama #9659](https://github.com/ollama/ollama/issues/9659) 和
  [llama.cpp #19138](https://github.com/ggml-org/llama.cpp/issues/19138)，但它复用的是 API 层，
  与本轮量化 loader 没有共同关键路径，混做只会推迟模型落地。
- [vLLM 量化文档](https://docs.vllm.ai/en/latest/features/quantization/) 已把
  compressed-tensors W4A16/Marlin 列为正式 CUDA 路径，因此这里实现精确格式子集比再支持一个
  低热度自制 GPTQ repo 更有复用价值。

## 从 v0.8.0 吸取的时间成本教训

仓库历史显示，v0.8.0 初始目标提交 `04001c6c` 一次新增 **17 个文件、2678 行**；随后又用
`a5d053fc` 把 11 阶段收敛为 4 个 release 阶段，用 `be18e0fc` 撤掉 vLLM/llama.cpp 外部性能
硬门，最后仍需要 `dcdc9f56` 的 587 行 handoff。问题不是验证本身无价值，而是功能、平台建设、
全仓审计、竞争基准和发布资产同时进入同一个完成分母，导致已有产品证据因无关 gate 变化反复
stale。

本目标把教训转成以下硬约束：

1. **一个模型、一个 checkpoint、一个产品后端。** 不在执行中增加 P1/P2。
2. **先出真 token，再补薄聚合。** 在 4090 真模型 `run` 首 token 前，不开发新的通用 validator、
   profiler、benchmark client、evidence collector 或 dashboard。
3. **最终只有四类证据。** `source/`、`panda-pad/`、`qwen38-4090/` 和根 manifest/validator；
   final validator 只读取文件，不负责启动测试、下载模型或采集性能。
4. **全 workspace source gate 只跑一次。** 开发中只跑 exact reproducer 和受影响 crate；候选 SHA
   形成后才执行一次 `run_gate.py unit`。单 case 失败不从头重跑全量。
5. **不做外部性能竞赛。** 本 goal 不启动 vLLM、不跑 c=1/4/16/32、不要求三次 CI、不建立 80%
   ratio。只保留一个很低的交互可用性止损线，防止“能输出但实际上不可用”。
6. **验收项不自动增长。** 新发现的 bug 先成为 focused reproducer；只有它属于固定 checkpoint 的
   正确性阻塞时，才能替换现有用例或经 goal amendment 加入，不能顺手扩成全格式/全模型审计。
7. **总工程时间盒为 7 个 active developer-days。** 超时不自动加 gate 或继续 GPU sweep，而是保存
   KEEP/REJECT、列出唯一阻塞和剩余工作量，由用户决定是否 amendment。下载等待和外部机器不可用
   单独记录，不伪装成开发进度。

## 当前代码基线与精确缺口

1. 当前正式矩阵只有 Qwen3.5、Qwen3 和 Llama 3.1 8B；见
   [`SUPPORT_MATRIX.md`](../../release/runtime-vnext/0.8.0/SUPPORT_MATRIX.md)。
2. `crates/ferrum-models/src/qwen35_config.rs` 已覆盖 `qwen3_5_text`、linear/full attention、
   attention gate、partial rotary、mRoPE 和 recurrent state。Qwen3.8 应尽量复用该 text core，
   但必须用官方 config fixture 证明，不能只加 model alias。
3. `WeightFormat` 目前只有 Safetensors 和 GGUF。`NativeSafetensorsLoader` 看到 `.qweight` 后进入
   GPTQ loader；它不理解目标权重的 `weight_packed`、`weight_scale`、
   `weight_zero_point`、`weight_shape`。
4. 固定 checkpoint revision
   `63768c10df38c0395e12ef49edac1bd539eaeeea` 的合同是
   `compressed-tensors/pack-quantized`、W4、group size 32、asymmetric zero point、无 activation
   quant，并在同一模型中混合 dense 与 quantized projections。
5. 现有 CUDA Marlin/GPTQ 小张量测试以 CPU dequant-GEMM 为 reference，CUDA 相对误差上限为
   `5%`。新路径应复用该测试方法和已有 Marlin ABI，不建立第二套 kernel 验证系统。

## 固定范围

| 项 | 固定值 |
|---|---|
| backend | CUDA；最终硬件 exactly one RTX 4090（sm89） |
| model | `cyankiwi/Qwen3.8-27B-AWQ-INT4` |
| revision | `63768c10df38c0395e12ef49edac1bd539eaeeea` |
| source format | `compressed-tensors`, `pack-quantized`, int4, group32, asymmetric, no activation quant |
| execution | 复用现有 CUDA Marlin W4A16 ABI；cold-path repack；禁止完整 F16 权重 fallback |
| modality | text-only |
| product entrypoints | `ferrum run`、`ferrum serve` |

若 M0 证明 revision、license、tensor schema 或单 4090 显存边界不成立，保存 REJECT artifact
并停止。更换 checkpoint 必须先 amendment 本文档，不能在代码里按模型名静默回退。

## CUDA 环境分工

| 环境 | 固定职责 | 能证明 | 不能证明 |
|---|---|---|---|
| `panda-pad` WSL；RTX 4050 Laptop 6GiB；Ubuntu 24.04；CUDA Toolkit 12.6 | CUDA build、格式 fixture、Marlin 小张量 parity、固定 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` 同架构双入口 smoke，以及 `qwen3:0.6b` 公共路径哨兵 | 源码可编译、CUDA 小张量正确、`qwen3_5_text` 混合执行图可走通、产品公共路径未坏 | 27B 能装入、真实 compressed-tensors 权重的完整模型组合、4090 性能、release-ready |
| 单 RTX 4090 | 固定 Qwen3.8 checkpoint 的最终真模型 smoke 和最低可用性测量 | 本 goal 的模型产品闭环 | 其他 GPU/模型/格式的泛化支持 |

源码只通过 GitHub `git`/`gh` 同步到 panda-pad 的仓库，不使用 SCP/rsync 传源码。artifact 可以
按既有安全流程取回，但不得包含 Tailscale IP、账户名、密钥路径、token 或代理凭据。

每个 panda-pad evidence run 重新记录 git SHA/dirty status、binary SHA256、`nvidia-smi`、
`nvcc --version`、WSL 有效 CPU/RAM/swap 和磁盘余量。不得把安装当天的口头状态当证据。

## 最小可量化验收

所有计数由 artifact/validator 读取；表外项目不阻塞本目标。

| ID | 必须证明 | 精确 PASS 判据 | artifact |
|---|---|---|---|
| G1 | 源码合同正确 | 固定 repo/revision、config、weight index、shard 名称/大小被 lock；目标配置 1/1 识别为精确格式合同；错误 bits/group、activation quant、缺 tensor 三个 fixture 3/3 在 GPU 分配前返回 typed error；四个固定 fixture 4/4 覆盖 asymmetric packing、single projection、fused QKV/gate-up 和 mixed dense/linear-attention，CUDA 对 CPU reference `rel_err < 0.05` 且 NaN/Inf=0；官方 config fixture 1/1、四个 template golden 4/4；候选 SHA 上 `FERRUM GATE unit PASS` 1/1 | `source/model-lock.json`、`source/contract-tests.json`、`source/unit/` |
| G2 | panda-pad CUDA 开发路径可复现 | exact release build exit=0；固定 revision 的 `Qwen3.5-0.8B`（`qwen3_5_text`，24 层：18 linear attention + 6 full attention）完成 `run` 单轮/两轮 2/2 非空及 `serve` non-stream/stream 2/2 成功；`qwen3:0.6b` 只做 `run` 单轮和 `serve` non-stream 2/2 公共路径哨兵；stream 恰好一个 `[DONE]` 且 usage output tokens >0；panic、OOM、`<unk>`、`[PAD]`、mojibake、空回答计数均为 0 | `panda-pad/host.json`、`panda-pad/build/`、`panda-pad/smoke/` |
| G3 | 4090 真模型可用 | 同一候选 SHA 和固定 revision：`run` known-answer/两轮 2/2；`serve` non-stream、stream、required tool、strict schema 4/4；stream 恰好一个 `[DONE]` 且 usage output tokens >0；上述错误计数均为 0。随后只跑 c=1、3 requests 的短测：completed=3、failed=0、服务端 usage 计数率=100%、median output throughput `>=5 tok/s`、p50 TTFT `<=30s` | `qwen38-4090/host.json`、`qwen38-4090/correctness/`、`qwen38-4090/usability/` |
| G4 | 声明与证据一致 | README/支持矩阵只声明固定 checkpoint、CUDA、text-only 和实测边界；根 manifest 只引用 G1-G3 同一候选 SHA 的 receipts；薄 validator 最后一行精确打印目标 PASS | `goal.manifest.json`、`validator.log` |

G3 的 `5 tok/s` 与 `30s` 是故意宽松的灾难性回归止损线，不是竞争性性能宣传。低于止损线时
本目标产生 REJECT 并停止，性能优化必须另立目标；不得为通过而加入 hidden env、缩短到零输出、
过滤坏 token 或改用另一个 checkpoint。

本目标明确**不要求**：c=4/16/32、100 requests、三次 repeats/CI、vLLM/llama.cpp ABBA、80%
竞争比、Metal 全门、CUDA full、Llama dense、tarball/Homebrew 或 release summary。它们属于正式
release goal，而不是新模型首次落地的完成分母。

## 实施里程碑与时间盒

### M0：checkpoint 合同预检（0.5 active day）

- 写入 `MODEL_MATRIX.json`：repo/revision、license、config/template/index SHA、shard 清单、
  architecture、quantization config、关键 tensor 名称/shape/dtype 和预计 VRAM/磁盘。
- 只下载小 metadata 和 safetensors header/index 做预检；不为了写审计文档先下载整模型。
- 输出 `KEEP` 或 `REJECT`。REJECT 即停，不自动寻找第二个模型。

### M1：精确 compressed-tensors W4 loader（2 active days）

- 在 typed weight-format/config 层识别目标合同，不用 repo 名、模型名或文件名猜格式。
- 读取 `weight_packed/weight_scale/weight_zero_point/weight_shape`，支持同模型中的 ignored dense
  projection；不假设每个 Linear 都量化。
- cold path 转换到已有 Marlin ABI；禁止产品路径完整反量化为 F16。
- 先过 3 个 reject fixtures 和 4 个 parity fixtures，再进入真模型。

若第 2 天仍没有四个 fixture 通过，记录唯一失败层（format、packing、zero point、shape mapping 或
Marlin ABI），停止扩大通用 compressed-tensors 支持面。

### M2：Qwen3.8 text core 与 panda-pad 产品路径（2 active days）

- 用官方 config fixture 证明 Qwen3.5 text core 可复用；只补真实字段差异。
- 四个 template golden 固定为 single-turn、multi-turn、thinking-off、tool continuation；模板错误
  必须显式失败，不得 fallback 到 builtin ChatML。
- 在 panda-pad 完成 exact release build；以固定 revision 的 `Qwen3.5-0.8B` 完成主要同架构
  双入口 smoke，并以 `qwen3:0.6b` 各跑一个入口作为廉价公共路径哨兵。

`Qwen3.5-0.8B` 只证明与目标相同的 `qwen3_5_text` 混合架构执行图，不证明目标
compressed-tensors W4 合同。后者由 G1 四个精确 CUDA parity fixture 与 G3 固定 27B 真权重共同
证明；不得用任一层证据替代另一层。

最迟在累计第 4.5 个 active day 形成可在 4090 尝试的候选 SHA。未形成时输出 REJECT/剩余估算，
不先写新的 gate 框架。

### M3：一次 4090 真模型闭环与聚合（最多 2.5 active days）

- 先跑 G3 六个 correctness 场景；任一失败只跑 exact reproducer。
- correctness 全过后才跑 3-request c=1 usability short run。
- 候选通过后执行一次 workspace unit gate、更新精确支持声明、运行薄 validator。

总计超过 7 个 active developer-days 时停止并给出 KEEP/REJECT review。延长、换 checkpoint、加入
性能优化或扩大格式支持都需要用户明确修改目标。

## 长命令和 GPU 成本边界

| lane | 预计时长 | hard deadline | 进度信号 | deadline 动作 |
|---|---:|---:|---|---|
| panda focused CUDA test | 2-10 分钟 | 15 分钟 | cargo log、CPU/GPU activity 或 receipt 增长 | 无进展立即停并分类 |
| panda release CUDA build | 6-12 分钟；已有 6m16s 冷编译基线 | 20 分钟 | rustc/nvcc/link log 或 CPU activity | 区分下载、Rust、nvcc、link、RAM/swap，不盲等 |
| panda 单个 product smoke | 1-5 分钟 | 10 分钟 | load log、GPU activity、首 token/HTTP event | 保存末尾日志和 GPU 状态后停止 |
| unit source gate | 20-45 分钟 | 60 分钟 | 日志、CPU 或 artifact 字节增长 | 定位最后 crate/test；之后只跑 reproducer |
| 4090 cold model lane | 2-3 小时 | 4 小时 | download bytes、model load、GPU memory、首 token 或 artifact 增长 | 复制 KEEP/REJECT 后停止计费 |

本目标不自动授权付费 GPU。启动前必须查询现有实例，优先复用有 model/build cache 的单 RTX
4090，并先写明 offer 单价、预计总成本、命令、progress signal 和 stop condition。同一候选 SHA
最多启动一次 full G3 lane；失败后只允许在同一 warmed session 跑 focused case。相同 failure class
连续两次 REJECT 后停止付费运行，直到有源码假设、本地测试和可观测指标变化预测。

## 最终 artifact 形状

```text
<out_dir>/
  source/
    model-lock.json
    contract-tests.json
    unit/
  panda-pad/
    host.json
    build/
    smoke/
  qwen38-4090/
    host.json
    correctness/
    usability/
  goal.manifest.json
  validator.log
```

最终新增的 `scripts/release/model_adoption_goal_gate.py` 必须是只读聚合器：只检查上面四类
evidence、同一 SHA、计数和阈值，不运行 cargo、模型、HTTP client、下载或 benchmark。现有
`ferrum bench-serve` 是唯一 HTTP 性能客户端，不再新增第二套。

## 完成定义

只有以下条件全部成立，validator 才能打印 PASS：

- G1-G4 全部满足，且 required artifact 同属一个候选 SHA；
- `run` 与 `serve` 都有 panda 同架构小模型和 4090 真模型 evidence，且另有 `qwen3:0.6b`
  双入口公共路径哨兵；
- 没有 hidden env、模型名 hack、输出过滤、完整 F16 fallback、waiver 或静默 feature fallback；
- 文档只声明 artifact 实际证明的 checkpoint、格式、后端、模态和性能边界；
- validator 最后一行精确为 `QWEN38 CUDA ADOPTION GOAL PASS: <out_dir>`。

## 非目标

- OpenAI `/v1/responses`、远程 MCP 或新的 agent runtime。
- Qwen3.8 Metal/GGUF 产品资格。
- Qwen3-Coder CUDA GPTQ 空回答修复。
- Qwen3.8 vision/video、多模态执行或 MTP speculative decoding。
- CUDA GGUF Q4_K/Q6_K kernel。
- gpt-oss/Harmony/MXFP4、NVFP4、外部 FP8 ingestion。
- 并发性能优化、分布式、tensor parallel 或双 4090。
- release tag、GitHub Release、crates.io、Homebrew 或 release-ready 声明。
- 为增加支持模型数量而添加未经真模型 gate 的 alias。

## PASS 后才做的下一次选择

本目标完成后重新依据最近 30/90 天 HF 下载变化、主流引擎 issue/PR 互动、Ferrum 代码复用率和
目标硬件可运行性，只选一个新 goal：

1. 无状态 `/v1/responses` adapter；
2. Qwen3.8 Metal GGUF 产品资格；
3. Qwen3-Coder CUDA 空回答故障闭环；
4. gpt-oss-20b + MXFP4 架构/格式 spike。

这些候选不因写在这里自动进入当前范围。
