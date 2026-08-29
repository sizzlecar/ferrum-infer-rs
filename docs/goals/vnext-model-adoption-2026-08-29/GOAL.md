# Ferrum vNext 热门本地模型顺序接入目标

## 状态、用途与完成权威

- 创建日期：2026-08-29。
- 状态：Active；2026-08-29 已激活 A1，本地实现已开始；用户已授权由执行者自行决策最终门禁所需
  的付费 GPU，但付费容量只允许在同架构小模型的 4050 完整 E2E 和其他本地 correctness 门禁通过后
  启动，不得用于开发迭代。
- 仓库内本文是目标执行与 amendment 的权威副本；外部草稿不再随执行状态更新。
- 这是一个 **model-adoption portfolio goal**：固定多个热门 checkpoint，按顺序逐个接入；不是把
  多个模型绑成一次性大版本，也不是 v0.8.x release-ready 审计。
- 同一时间只允许一个 checkpoint 处于 `ACTIVE`。后一个 checkpoint 失败，不得抹掉前一个已经取得
  的 `PASS`。
- 每个 checkpoint 独立得到 `PASS`、`BLOCKED` 或 `REJECT`。本文故意没有“所有模型全部通过”的
  聚合硬门。
- 编译通过、模型成功加载或输出首 token，都不能单独代表 checkpoint 完成。

每个 checkpoint 只有在薄 validator 的最后一行完全等于该行时才完成：

```text
FERRUM VNEXT MODEL ADOPTION PASS: qwen38-27b-fp8 <out_dir>
FERRUM VNEXT MODEL ADOPTION PASS: qwen36-27b-fp8 <out_dir>
FERRUM VNEXT MODEL ADOPTION PASS: qwen36-35b-a3b-fp8 <out_dir>
FERRUM VNEXT MODEL ADOPTION PASS: gpt-oss-20b-mxfp4 <out_dir>
FERRUM VNEXT MODEL ADOPTION PASS: gemma4-12b-w4a16-ct <out_dir>
```

未取得 PASS、但需要关闭当前 lane 时，同一 validator 必须完成 receipt 完整性校验，并把最后一行
打印为以下唯一一种终态；只有该终态 artifact 存在，队列才可继续：

```text
FERRUM VNEXT MODEL ADOPTION BLOCKED: <checkpoint_id> <out_dir>
FERRUM VNEXT MODEL ADOPTION REJECT: <checkpoint_id> <out_dir>
```

这些 PASS 只证明固定 revision、CUDA、text-only、单卡和本文最小产品闭环；不能改写成“支持整个
模型系列”“通用 FP8/MXFP4”“多模态”或“release-ready”。

## 决策结论

下一步不是只实现 `Qwen/Qwen3.6-27B-FP8`，而是建立以下顺序队列：

1. 先补齐现有 `vnext::qwen35` 的官方 block-FP8 source contract，以最新的
   `Qwen/Qwen3.8-27B-FP8` 做首个产品闭环。
2. 用同一套实现低成本资格化 `Qwen/Qwen3.6-27B-FP8`，再覆盖
   `Qwen/Qwen3.6-35B-A3B-FP8` 的 MoE 路径；三者分别验收，不互相连坐。
3. 第二个真正的新 family 做 `openai/gpt-oss-20b` 原生 MXFP4。它有明确的本地部署需求，也能让
   Ferrum 获得非 Qwen 的差异化能力。
4. 第三个新 family 做 `google/gemma-4-12B-it-qat-w4a16-ct`，首版只承诺 text-only，覆盖常见
   16GB 级设备市场。
5. `GLM-4.7-Flash` 只做候选审计；在前述队列没有至少两个 PASS 前，不投入付费 GPU。

这是一条队列，不是五个并行分支。当前唯一 `ACTIVE` 是 `qwen38-27b-fp8`。

## 当前项目边界

### production vNext 已有能力

| family | 已注册架构 identity | 当前可用权重/量化合同 | 本目标如何使用 |
|---|---|---|---|
| `family.qwen3_5.hybrid` | `Qwen3_5ForConditionalGeneration`、`Qwen3_5MoeForConditionalGeneration` | safetensors dense、GPTQ-Marlin INT4、特定 compressed-tensors pack-quantized W4、typed GGUF source | 复用现有 family；只补官方 block-FP8 source/materialization/provider 合同 |
| `vnext::qwen3_moe` | `Qwen3MoeForCausalLM` | GPTQ-Marlin INT4、typed GGUF source | 保留为回归和未来 coder 候选；本队列不复制它 |

`LlamaForCausalLM`、`Qwen2ForCausalLM`、dense `Qwen3ForCausalLM`、Gemma 3、Mistral、Phi 等
仍在显式 legacy 列表中。legacy 产品运行不能作为本目标的 vNext 证据，也不能为了快速支持新模型
继续往 legacy 列表加 identity。

当前已有回归面精确锁定的是 `cyankiwi/Qwen3.8-27B-AWQ-INT4` 的 compressed-tensors、
pack-quantized W4、group 32、asymmetric 合同；这不是通用 AWQ/INT4 支持，也不能替代官方 FP8
checkpoint 的 source contract。

### 唯一允许的生产链路

```text
ProductionModelSourceBundle
  -> production vNext model registry
  -> family prepare_from_sources
  -> PreparedModelFamily / WeightSchema / ModelProgram
  -> ProgramPlanCompiler
  -> backend capability catalog
  -> registered materializer and operation providers
  -> shared ferrum run / ferrum serve executor
```

架构不够时，必须改造共享 vNext contract、对应 family package、plan/catalog 或 provider 注册，不能
创建另一条能跑但不受 vNext 约束的路径。

## 固定候选与顺序

热度数据是 2026-08-29 的方向性快照：Hugging Face downloads 不是独立用户数，Ollama pulls 可能
是 family/tag 聚合值。它们用于排序，不进入验收分母。

| 顺序 | 状态 | 精确 checkpoint | 架构与量级 | 权重与量化格式 | vNext 增量 |
|---|---|---|---|---|---|
| A1 | `ACTIVE` | [`Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `Qwen3_5ForConditionalGeneration`；27B dense | safetensors；官方 `fp8` metadata，E4M3、dynamic、128x128 block | 复用 `vnext::qwen35`；新增 typed block-FP8 ingestion 和有质量凭证的执行转换 |
| A2 | `QUEUED` | [`Qwen/Qwen3.6-27B-FP8@e89b16ebf1988b3d6befa7de50abc2d76f26eb09`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | `Qwen3_5ForConditionalGeneration`；27B dense | 与 A1 同类的官方 block-FP8 safetensors | 资格化同一 source/provider；不得复制模型或 kernel 路径 |
| A3 | `QUEUED` | [`Qwen/Qwen3.6-35B-A3B-FP8@95a723d08a9490559dae23d0cff1d9466213d989`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | `Qwen3_5MoeForConditionalGeneration`；35B-A3B MoE | 官方 block-FP8 safetensors | 复用同一 FP8 合同并验证现有 qwen35 MoE program/provider；不新建 Qwen3.6 family |
| B | `QUEUED` | [`openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee`](https://huggingface.co/openai/gpt-oss-20b) | `GptOssForCausalLM`；约 21B、3.6B active MoE | safetensors；原生 MXFP4 专家权重和较高精度 exclusions | 新建 production vNext family、typed MXFP4 source/layout/provider，并实现 Harmony 输出语义 |
| C | `QUEUED` | [`google/gemma-4-12B-it-qat-w4a16-ct@1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct) | `Gemma4UnifiedForConditionalGeneration`；约 12B dense | safetensors；compressed-tensors W4A16、INT4、group 32、symmetric、pack-quantized | 新建 production vNext family 和 CT mapping；补齐或证明可复用 sliding-window/full attention、双 RoPE、GELU、logit softcap typed operations/providers；首版 typed text-only |
| D | `WATCH` | [`zai-org/GLM-4.7-Flash@7dd20894a642a0aa287e9827cb1a1f7f91386b67`](https://huggingface.co/zai-org/GLM-4.7-Flash)；第三方 Q4_K_M 尚未锁定 | `Glm4MoeLiteForConditionalGeneration`；30B-A3B，MLA + MoE | 候选第三方 GGUF Q4_K_M 约 18GB | 只做 M0 source/architecture spike；离开 WATCH 前必须 amendment 固定 GGUF repo、revision、文件名和 digest；当前不进入实现分母 |

选择依据：

- [Qwen3.8 在 Ollama](https://ollama.com/library/qwen3.8) 有强新品动量；它与当前 qwen35 identity 对齐，
  是最小架构成本的热门增量。A2/A3 用来证明这不是只对一个 dense checkpoint 写特判。
- [gpt-oss 的官方定位](https://openai.com/index/introducing-gpt-oss/)就是本地和自有基础设施部署；
  [Ollama gpt-oss family](https://ollama.com/library/gpt-oss) 与 Hugging Face 的长期生态信号都强，
  因而排在新 family 第一位。
- [Gemma 4 官方 model card](https://ai.google.dev/gemma/docs/core/model_card_4) 和
  [Ollama Gemma 4 family](https://ollama.com/library/gemma4) 说明 12B 档位有明确用户面；选择官方 CT
  checkpoint 是为了锁定可审计的 W4A16 合同。Ollama Q4_0 只作为热度信号，不能与该 checkpoint
  的 compressed-tensors 格式混写。
- GLM-4.7-Flash 有真实关注度，但新增 MLA 与 MoE 的耦合风险最高，先观察，不抢占当前预算。

## 所有 lane 共同遵守的 vNext 约束

1. 模型 identity 只由锁定 source metadata 通过 production registry 解析；不按 repo id、文件名、
   GPU 名称或显存大小写行为分支。
2. 量化 recipe、物理 layout、scale/zero-point sidecar、dense exclusions 都必须 typed 表达，并在
   GPU allocation 前完成 dtype/shape/classification 校验；checkpoint 全部 tensor 必须被划分为
   execution-eligible、typed non-executed 或 rejected，unknown tensor 必须为 0。
3. provider 必须通过 backend capability catalog 注册并由 typed plan 选择；family/loader 不得直接
   调 kernel。
4. `ferrum run` 与 `ferrum serve` 必须消费同一份 prepared family、plan、weight decision、tokenizer
   和 chat template identity。
5. 不允许 legacy、CPU、dense 或另一量化格式 silent fallback。不能用隐藏环境变量选择 loader、
   provider、text-only 或 fallback。
6. 新 family 只能注册在 production vNext registry。Gemma 4 不能伪装成 Gemma 3；gpt-oss 不能
   塞进 Qwen MoE；Qwen3.6/3.8 也不能复制成版本专用 executor。
7. 多模态 checkpoint 的首版 text-only 必须是 typed capability/result；不得通过删视觉/音频文件、
   模型名判断或异常捕获后继续运行实现。
8. approximate materializer 必须绑定 materializer id/version、implementation fingerprint、source
   schema fingerprint、execution schema 和 checked-in quality vector digest；不得使用全局布尔或
   环境变量放行。
9. 如果唯一 vNext 路径在时间盒内仍不能表达 checkpoint，结果是 `BLOCKED`；不能开第二条路径，
   也不能临时降低验收标准制造 PASS。
10. A1 不预设现有 FP8 provider 已覆盖完整 Qwen3.8 program。M0 必须锁定 execution-eligible
    tensor/op 集合及 catalog coverage matrix；缺失的共享二维 block layout、materializer、
    operation provider 或 compiler quality-approval authority 都是 A1 明示交付物，不能用现有
    F16 -> channelwise FP8 路径代替。
11. 每个 checkpoint 都先选择可在本地 RTX 4050 6GB 上运行的同架构官方小模型，完成共享
    `ferrum run`、`ferrum serve`、stream 和 provider attribution E2E；若官方小模型的权重格式不同，
    再从该固定小模型确定性地产生同架构、目标量化布局的开发派生件以覆盖 ingestion/materializer。
    小模型及派生件只形成 correctness canary，不替代目标 checkpoint 的最终 receipt；付费 GPU 只运行
    已经通过本地门禁的官方目标 checkpoint 最终测试。

本文区分四类 digest：`checkpoint_content_digest` 锁定 checkpoint 文件内容，
`source_schema_fingerprint` 锁定 typed source tensor/layout，`execution_contract_fingerprint` 锁定
执行 schema/provider 合同，`quality_vector_digest` 锁定数值批准输入和 reference。只有对应字段
逐项相同的静态证据才可复用，不得用“内容相同”笼统替代。

## 独立 lane 的最小验收

以下验收对 A1、A2、A3、B、C 分别执行。A2/A3 只有在 `source_schema_fingerprint`、
`execution_contract_fingerprint` 和 `quality_vector_digest` 分别完全一致时才可复用对应静态 fixture；
`checkpoint_content_digest`、tokenizer/template digest、产品运行和 provider attribution 必须逐
checkpoint 重新锁定，不能复用别的 checkpoint 的结果。

| ID | 必须证明 | 精确 PASS 判据 |
|---|---|---|
| M0 source lock | checkpoint 没有靠猜测接入 | model、完整 revision、license、config、tokenizer/template、index/shard 清单 1/1 锁定；checkpoint 全部 tensors 100% 划分为 execution-eligible、typed non-executed 或 rejected，unknown=0；从 manifest 推导 expected execution quant tensor/op 精确集合、数量和 catalog coverage matrix；锁定 quality-vector generator/input/reference 语义及四类 digest；峰值 host/device memory 有书面估算 |
| M1 fail closed | 错格式不会跑到 GPU 后才暴露 | 两个代表性坏合同 2/2 在 GPU allocation 前 typed-reject：一个 metadata/recipe 不匹配，一个 tensor dtype/shape/sidecar 不匹配 |
| M2 local path | 4050/普通开发机能完成日常迭代 | exact CUDA release build exit=0；family/source、weight/materializer、plan/provider、run/serve shared identity 四组 affected commands 4/4 exit=0；固定同架构小模型的 `run`、`serve` non-stream、stream E2E 全过，目标量化格式不同时再跑确定性派生件 ingestion/materializer canary；候选 SHA 冻结后 `run_gate.py unit` 1/1 PASS |
| M3 numeric | 新量化执行不是未验证黑盒 | 对 M0 锁定的 quality vector/reference 跑 2 个 weight shape x 2 个 activation batch，共 4/4 CUDA case；reference 固定为“source quantized values 按锁定 scales/layout 解码后”的 matmul 输出；relative L2 `<=0.05`；NaN=0、Inf=0；通过绑定 `quality_vector_digest` 的 typed approval 后 compiler 才能选择 approximate materializer；reuse 完全相同 provider 时只重验 source/layout 和一个 canary，不重复整套矩阵 |
| M4 product | 用户入口真的可用 | 预缓存权重后 load-to-ready `<=600s`；`run` 1/1、`serve` non-stream 1/1、stream 1/1；c2 固定短稳定性 4/4，每请求 output tokens `>=16`；HTTP 500、panic、OOM、CUDA error、invalid UTF-8、raw control/special token leakage 均为 0 |
| M5 usability | 没有退化成无意义慢路或 fallback | 使用 canonical `ferrum bench-serve --fail-on-error --seed 9271 --n-repeats 1` 跑三个固定 c1 短请求，median output throughput `>=5 tok/s`，p50 TTFT `<=60s`；attribution 分母是 M0 在全量 partition 后锁定的 execution-eligible quant tensor/op 集合且必须为 100%，typed non-executed tensors 仍须入 inventory 但不进入执行 attribution；silent/dense/legacy fallback=0 |
| M6 audit | 声明与架构闭合 | 5 项 checklist 5/5：registry family、typed quant/layout、plan/catalog provider、shared run/serve identity、no bypass/hidden-env/fallback；`scripts/release/vnext_model_adoption_gate.py` 通过自测、校验 versioned schemas 和本 checkpoint receipts，并打印唯一终态行 |

相对 L2 定义：

```text
relative_l2 = ||output - reference||2 / max(||reference||2, 1e-6)
```

产品场景只验证客观不变量，不用主观回答质量做硬门：

- `run`：固定短问题，exit=0、assistant 非空、包含固定客观 marker。
- `serve` non-stream：HTTP 200、JSON 可解析、assistant content 非空。
- `serve` stream：HTTP 200、`[DONE]` 恰好一次、usage chunk 恰好一次、output tokens > 0。
- gpt-oss 额外增加一个 Harmony/tool-call 客观 case；其他 checkpoint 不强制 tool calling 或 JSON
  schema。
- c2 只跑一次、四个固定 256-input/32-output 请求；不是并发或性能宣传。

Qwen A lane 的固定产品场景通过公开 typed 选项禁用 thinking：`ferrum run --disable-thinking`，
server 请求使用 `chat_template_kwargs.enable_thinking=false`，bench 使用
`--enable-thinking false`。不得靠隐藏环境变量获得稳定 marker。

c2 是 `concurrency=2` 的稳定性检查，不形成吞吐或扩展性声明。checkpoint PASS 也不自动更新
README/support matrix；如需更新支持声明，必须另行满足仓库现有 model-onboarding contract，不能
把本目标的四份 receipt 冒充 onboarding 证据。

本目标明确不要求 c16/c32、100 个请求、三次 repeats、95% CI、vLLM/llama.cpp 同机基线、外部
性能比例、Metal/CUDA 全 release matrix、长上下文、多模态、MTP/speculative decoding、release
asset 或 Homebrew 验收。

## 最小证据包

每个 checkpoint 只保存四个 JSON receipt 和一个非 receipt 日志：

```text
<out_dir>/
  model-lock.json
  validation.json
  product.json
  manifest.json
  validator.log
```

- `model-lock.json`：revision、license、文件/tensor schema、typed execution/non-execution partition、
  expected quant tensor/op 集合、coverage matrix、quality-vector generator/input/reference 语义、四类
  digest 和内存估算。
- `validation.json`：build/test 命令、candidate SHA、dirty status、binary SHA256、数值批准结果和 5 项
  架构 checklist。
- `product.json`：candidate SHA/dirty status、binary SHA256、GPU/driver/runtime、完整命令与时间、
  effective config、run/serve/stability/usability、provider attribution。
- `manifest.json`：前三个 receipt 的 SHA256、checkpoint id、validator version 和最终状态。
- 静态 source/execution/quality fixture 只有在各自 fingerprint/digest 完全一致时才可复用；代码执行
  receipt 必须绑定当前 candidate SHA，产品 receipt 还必须绑定实际 binary SHA256。

validator 只校验 receipt schema、digest、固定阈值和 checkpoint identity，不重新实现一套 release
审计框架。

validator 固定为 `scripts/release/vnext_model_adoption_gate.py`，并提供 `--self-test`。四份 receipt
使用 `scripts/release/schemas/vnext_model_adoption/` 下的 versioned JSON schema。每份 receipt 的
共同 envelope 至少包含 `schema_version`、`artifact_type`、checkpoint id/full revision、candidate
git SHA/dirty status、sanitized environment、argv、start/finish/duration、声明的 deadline/progress
signal，以及引用文件的 path/size/SHA256。适用时必须记录 binary SHA256、GPU/driver/runtime 和
effective config。

`manifest.json.final_status` 只能是 `PASS`、`BLOCKED` 或 `REJECT`，并绑定另外三份 receipt 的
SHA256 与 terminal line。三种终态都必须生成完整四份 JSON；未执行阶段以 typed `not_run` 和唯一
原因表示，不能省略 receipt。只有 PASS 才要求 M0-M6 全满足；BLOCKED/REJECT 只关闭 lane，不构成
产品或性能成功声明。

## 时间盒、硬件与预算

RTX 4050 Laptop 6GB 负责 workspace 编译、affected tests、metadata fixture、小张量 CUDA parity，
并负责同架构官方小模型及必要量化派生件的完整 `run`/`serve` E2E；它不承担官方目标 checkpoint
容量或性能验收。所有官方目标模型最终测试只租 **一张卡**，本目标不引入多卡执行。A1 的固定
本地 canary 是
`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`；其官方 BF16 权重验证共享架构，
确定性 block-FP8 派生件验证 A1 权重路径，二者都不能替代 27B FP8 receipt。

| lane | active developer-days 上限 | 付费 GPU 上限 | 推荐单卡 | 超限结果 |
|---|---:|---:|---|---|
| A1 Qwen3.8 dense block-FP8 | 8 天 | 6 小时且 `<= USD 12` | L40S 48GB | `BLOCKED`，记录唯一阻塞并评审共享 FP8 方案 |
| A2 Qwen3.6 dense qualification | 2 天 | 3 小时且 `<= USD 6` | 复用 A1 同型 48GB 卡 | 若需要第二套 provider，直接 `BLOCKED`，不复制实现 |
| A3 Qwen3.6 MoE qualification | 3 天 | 4 小时且 `<= USD 10` | M0 优先 L40S 48GB；峰值不安全才用单张 80GB | `BLOCKED`；不改成多卡来凑 PASS |
| B gpt-oss MXFP4 | 10 天 | 4 小时且 `<= USD 10` | RTX 4090 24GB | `BLOCKED`，保留 source/provider spike 证据 |
| C Gemma 4 W4A16 | 8 天 | 4 小时且 `<= USD 8` | 16GB 以上 CUDA，优先 24GB | `BLOCKED`，不回退到 Gemma 3 legacy |
| D GLM watch | 1 天 M0 spike | `USD 0` | 无 | 输出 GO/DEFER/REJECT，不进入实现 |

用户已为本目标的最终门禁授予付费 GPU 自主决策权。每次实际启动前仍必须先核对现有实例 inventory，
并公开写清：复用哪台实例或为何新租、预计时长/成本、correctness command、product command、
停止条件和 artifact 目录。只有本地 M0-M3、同架构小模型 E2E 和目标格式派生 canary 已通过时才可
启动；下载模型和编译应尽量在计费前完成，失败后先复制证据并停机，不能让实例空转等待下一次尝试。

## 各 lane 的特殊边界

### A：Qwen official block-FP8

- 官方 metadata 是 `quant_method=fp8`，不是 compressed-tensors FP8。
- 若当前 `Qwen35QuantizationConfig` 无法无损表达 E4M3、dynamic metadata、128x128 block scale，
  就重构为 tagged/typed recipe；不得给 `bits/group_size/sym` 填假值。
- 当前 `marlin_fp8_materializer` 的 F16 -> channelwise FP8 不能冒充官方 block-FP8 ingestion。
  新路径必须消费锁定的 FP8 value + 2D scale，并明确记录 source format 与 execution format。
- A1 的首选实现是有质量凭证的冷路径 materialization 到通过 capability catalog 注册的 CUDA
  execution providers；M0 必须先审计并补齐 Qwen3.8 所需 DenseLinear、DenseSwiGLU、attention 等
  operation coverage，不能假定当前 channelwise FP8 provider 已经完整。只允许单组件 transient
  buffer，不得完整展开并长期保存整模型 dense 权重。
- A2/A3 若 source schema、execution contract 和 quality vector 与 A1 的对应 fingerprint/digest 相同，
  必须复用同一实现。只允许 family program/shape 差异，不允许版本专用 provider。
- A1/A2/A3 的 vision 和 `mtp.*` tensors 都必须被完整分类为 typed non-executed components；本目标
  不实现或调度 vision、MTP/speculative decoding，也不允许通过不下载 shard 或忽略未知 tensor
  达到 text-only。

### B：gpt-oss-20b MXFP4

- 新建 `GptOssForCausalLM` production vNext family；不复用 Qwen MoE identity。
- M0 必须分类 MXFP4 expert tensors、较高精度 attention/router/embed/lm_head exclusions 和 scale
  layout；U8 storage 不能未经 typed decode 就当普通 INT8。
- Harmony chat/response semantics 是产品合同的一部分，但首版只要求普通 chat、stream 和一个客观
  tool-call case；不扩成完整 agent/tool/schema 兼容矩阵。
- 官方“可在 16GB 内运行”只作为容量参考；验收固定 24GB 单卡并保留余量。

### C：Gemma 4 12B W4A16 compressed-tensors

- 新建 `Gemma4UnifiedForConditionalGeneration` production vNext family；不迁移或复用 Gemma 3
  legacy executor。
- 固定 checkpoint 是官方 safetensors compressed-tensors W4A16，不是 Ollama Q4_0 GGUF。
- 首版 text-only 必须在 source/plan 中明确不调度 vision/audio components，同时保留对非法多模态
  请求的 typed rejection。
- group 32、symmetric、pack-quantized 和 dense exclusions 必须来自 checkpoint metadata/header，
  不从模型名推断。
- M0 必须逐项确认并在需要时新增 typed operation contracts/providers：sliding-window attention
  （window 1024）、full/sliding 两套 RoPE、`gelu_pytorch_tanh` 和 final-logit softcap；不能把 Gemma 3
  或标准 causal 默认值静默套用到 Gemma 4。

## 队列推进规则

1. 只有当前 lane 得到 `PASS`、`BLOCKED` 或 `REJECT` 后，才能把下一个 lane 改为 `ACTIVE`。
2. A1 PASS 后，A2/A3 是共享 FP8 实现的资格化；它们各自失败不撤销 A1 PASS。
3. B 与 C 都是独立新 family。B BLOCKED 不自动切换到旁路；可以结束该 lane 后按既定队列启动 C。
4. D 只有在 A/B/C 中至少两个 checkpoint PASS，且用户再次批准后，才能升级成新的独立 goal。
5. 新发现只有在让固定 checkpoint 无法正确运行时才是 blocker；泛化、性能优化、第二量化格式、
   Metal、多模态、长上下文、release packaging 默认进入 backlog。
6. checkpoint 被删除、license 改变、锁定 schema 被证伪、单卡容量不可行或阈值确需改变时，必须先
   amendment 本文并获得用户确认；不能在付费运行后向下移动门槛。

## 立即执行的第一个里程碑

不租卡，先完成 A1 的 M0/M1，并准备 M3 的小张量 fixture：

1. 从固定 Qwen3.8 revision 锁定 config、tokenizer/template、safetensors index/header、FP8 value/
   scale tensor 命名、dtype、shape、block grid 和 dense exclusions。
2. 把 `Qwen35QuantizationConfig` 改造成能无损表达官方 block-FP8 的 typed recipe，并让未知 recipe
   fail closed。
3. 扩展共享 typed contract 表达 128x128 二维 block scale，并在 `vnext::qwen35` 中建立 source
   layout -> execution layout 的显式合同；不得改变 registry identity，也不得增加 Qwen3.8 executor。
4. 实现两个 bad-contract fixtures；同时落四个小张量 CUDA parity fixture/case，实际 4/4 CUDA
   结果仍属于 M3，不能提前算作 M0/M1 PASS。
5. 锁定 Qwen3.8 execution op coverage matrix、typed quality approval authority 和缺失 provider 的
   实施边界；不得把当前部分 FP8 provider coverage 当成现成完整终点。
6. 本地四组 affected tests 全过后，提交一份包含峰值显存估算、租卡命令和 `<= USD 12` 上限的
   A1 GPU 执行申请；得到批准后才启动 L40S。

这六步是当前唯一实施范围。A2、A3、B、C 已经锁定方向和验收，但不会在 A1 结束前并行开工。
