# 2026-06 模型覆盖目标(Model Coverage W1/W2/W3)

## 状态

草案目标文件。调研基线:2026-06-12,三路独立证据(发布雷达 / 需求侧数据 / 架构-格式矩阵),
方法与来源见文末"调研记录"。

本目标分三个 wave,各自有独立 PASS 行。只有验证器打印对应行才算完成:

```text
MODEL_COVERAGE_W1 GOAL PASS: <out_dir>
MODEL_COVERAGE_W2 GOAL PASS: <out_dir>
```

W3 不在本目标内宣称完成,本目标只交付 W3 的立项合同(见下)。

## 目标

把 `ferrum run <model>` 对 2026-06 本地部署需求榜 top-15 的可跑数从当前约 4 个
(Qwen3 dense ≤8B / Qwen3-30B-A3B / Llama-3.x / Mistral-24B 线)提升到 ≥9 个,
且每个新模型按 "agent-grade" 标准认证后才算支持:

- `ferrum run` 与 `ferrum serve` 走产品路径(alias + autosizer,无手工 env);
- 正确性:通过"验收 gate 定义"一节的 L0–L5 全部层级;
- 性能:通过"验收 gate 定义"一节的性能 gate(按同构复用 / 新家族分类);
- 文档:README 支持矩阵行 + alias + 模板/EOS 来源说明 + release gate cell。

产品依据:产品力 = 模型覆盖 × 服务质量。0.7.7 已把服务质量做到位
(0.83× vLLM、tools/json_schema、gate 体系),覆盖是当前乘法里的小因子;
本目标只补"需求榜上且引擎税可控"的模型,不与 llama.cpp 拼广度。

## 2026-06 需求与格局快照(调研结论)

影响选型的硬事实:

| 事实 | 来源信号 | 对本目标的含义 |
|---|---|---|
| Qwen3.5(2026-02)/ Qwen3.6(2026-04)成为 24GB 第一家族 | Ollama qwen3.5 13.4M pulls;qwen3.6 2.2M/约1个月;多篇 2026-05 评测把 Qwen3.6-27B 列为 24GB 最强 | 但全系改用 Gated DeltaNet 混合线性注意力 + 256 专家 MoE + 原生视觉 —— 对 ferrum 是 LARGE→XL,放 W3 |
| Gemma 4(2026-04-02)是增速最快的本地模型 | Ollama gemma4 13.3M pulls / 约2个月 | K=V 共享投影(新 KV 语义)+ 逐层异构 head_dim + softcap,且暂无 GPTQ/AWQ —— 跳过,待量化生态与 W2 地基成熟后重评 |
| Meta 退出开源:Llama 5 不存在,2026-04 转向闭源 Muse Spark | meta-llama HF org 无 Llama 4 之后的新开源权重 | Llama 系是遗产盘;不再为其新增投入,现有支持吃存量 |
| DeepSeek R2 从未发布;V4/V4-Flash(2026-04)需求巨大但 INT4 远超 48GB | HF 6.8M+ 下载;Ollama 404;llama.cpp 仅社区 fork | 出 envelope,跳过;R1 蒸馏系(87.4M Ollama 遗产盘)是可白捡的替代 |
| 小激活 MoE(约30B-A3B/A4B)成为 24GB 标准形态 | GLM-4.7-Flash / Qwen3.6-35B-A3B / Gemma4-26B-A4B / Nemotron3-Nano 全是此形态 | ferrum 的 Qwen3-MoE 资产正中主流形态,M3 性能投资可直接复用 |
| 官方低比特权重成为标配 | Qwen3.5 官方 GPTQ-Int4 + FP8;Kimi K2.6 原生 INT4 QAT;NVIDIA NVFP4 | GPTQ-Marlin 主线继续有效;W3 解锁 Qwen3.5 线时可直接吃官方 GPTQ-Int4 |
| 24GB 实际使用以 GGUF Q4_K_M / UD-Q4_K_XL 为主 | unsloth GGUF 下载榜;Ollama 默认 q4 | GGUF 路线(Metal + 双卡 layer-split 已验证)与社区使用习惯一致 |
| 现任引擎对 2026 新架构服务质量差 | Ollama 跑 Qwen3.6 prefill 231–630 t/s,llama.cpp 同期约 2000 t/s;llama.cpp DeltaNet CUDA 路径仍慢(issue #17822) | W3 若做成,是差异化窗口而非跟跑 |

## 硬件与格式约束

### 硬件 envelope

- 主力:单卡 RTX 4090(24GB)。入选模型 INT4 权重 + 中等 ctx KV 必须放得下(权重 ≤ 约 18GB)。
- 次级:双卡 RTX 4090(48GB)layer-split,已由 `docs/goals/llama33-70b-4bit-2x4090/` 验证 70B 4bit 路线。
- 不为任何超出 48GB INT4 的模型立项(expert CPU offload 不在本目标范围)。

### 量化/权重格式路线

| 格式 | 地位 | 动作 |
|---|---|---|
| safetensors BF16/FP16 | 正确性基线 | 不变 |
| GPTQ-INT4(Marlin) | CUDA 主力 | 不变;每个新模型入 gate 前必须核对社区 GPTQ(QuantTrio/cpatonn/JunHowie)的 group size / desc_act 与 Marlin 兼容性 |
| GGUF Q4_K/Q6_K/Q8_0 | Metal 主力 + 双卡 layer-split 主力 | 不变;新模型按需扩 gguf loader 的 arch 白名单 |
| AWQ → Marlin repack | 候补 | W2 期间审计:`weight_format.rs` 现有 AWQ 引用的真实深度,以 vLLM awq_marlin 为参考;仅当某入选模型无可用 GPTQ 时升级为必做 |
| FP8 | defer | sm89 可做 W8A16,但 24GB 上 INT4 仍占优;不做 |
| MXFP4 | defer | 仅 gpt-oss 需要,绑定 W3 备选决策 |
| NVFP4 / Q4_0 / Q3_K / IQ 系 | 不做 | 无 Blackwell 测试卡;Q4_0(Gemma QAT)等待有刚需模型再说 |

## 验收 gate 定义(每个模型;全过才算"支持",缺一不可)

### 正确性 gate(L0–L5,顺序执行,首败即停,失败不进下一层也不进 bench)

| 层 | 测试内容 | 通过判据 |
|---|---|---|
| L0 模板/分词 | chat template golden 对照:ferrum 渲染结果 vs HF `apply_chat_template` 输出**逐字符相等**,固定覆盖 5 种用例:单轮 / 多轮 / 带 system / 带 tools 注入 / reasoning 历史剥离(R1 系);特殊 token 断言:EOS/BOS id 必须来自该模型 `generation_config`(不许按家族猜) | 5/5 相等 + token 断言全过;另需一条"渲染失败显式报错"测试——禁止静默 fallback 到 builtin ChatML |
| L1 数值(BF16) | safetensors BF16 greedy 输出 vs HF transformers 同 prompt **byte-equal**,N≥20 条 prompt(沿用 `reference_match` 基线机制,fixture 进库);W2 新家族额外:上卡前先 Mac/CPU 逐层激活 dump 对照,定位首个发散层 | 20/20 byte-equal |
| L2 量化路径 | GPTQ-INT4(CUDA)与 GGUF(Metal / 双卡 layer-split)各自跑 Paris 式 known-answer smoke N≥10,并与同引擎 BF16 greedy 输出做行为对照 | known-answer 10/10;允许 token 级分歧,语义级答案必须正确;量化路径**不要求** byte-match HF |
| L3 行为 | 多轮 KV 复用、stream 与 non-stream 输出内容一致、EOS 自然停止断言(在 max_tokens 之前停)、自定义 stop sequences、reasoning 模型 `<think>` 正确进 `message.reasoning` | 现有 chat/server smoke 套件扩 alias 后全绿 |
| L4 agent | `server_agent_tools` + `server_structured_output` 真模型 smoke,沿用 agent-serving.md 的既有标准:strict json_schema **20/20**、required tool-call **10/10**,temperature=0 | 全过才允许写进 README 支持矩阵(这就是 "agent-grade" 的定义) |
| L5 并发 | `bench-serve` random 256/128,c=1/4/16(30B 级补 c=32) | 全 cell 100% 完成、零错误、consecutive-repeat 退化检测器通过 |

### 性能 gate(按接入类型分类,数字门槛如下)

| 类型 | 适用对象 | 通过判据 |
|---|---|---|
| 同构复用类 | W1 全部(Qwen3-Coder-30B、R1 蒸馏系、Qwen3-32B/14B、Qwen2.5-Coder、Mistral 24B 线、70B 双卡) | 同卡同量化同 c 下,吞吐与同架构已 gate 模型差距 **≤10%**(例:Qwen3-Coder-30B-A3B 对照 M3 的 G0 数字)。代码路径相同,显著更慢=接入 bug(模板低效 / 错误 fallback / KV 配置错),按正确性问题处理,不立优化议题 |
| 新家族类 | W2(GLM-4.7-Flash、Gemma 3 27B)及以后 | ① 记录 c=1/4/16/32 tok/s artifact 进 release 证据目录;② 与 llama.cpp 同卡同量化对照;③ sanity floor:decode 吞吐 **≥0.5× llama.cpp**,低于即 gate fail——停手、另立单模型 perf goal,禁止在覆盖目标内就地展开优化(anti-drift);④ 0.5×–0.8× 区间记录为 known-gap,不阻塞覆盖发布 |
| 存量保护 | 所有 wave 的公共改动(模板引擎、loader、逐层注意力调度) | M2 / M3 / Llama-8B 的 Gate C7 decode floor 与 G0 gate **不回退**;每个公共工程项合入前跑一次存量回归 |
| Metal | 有 GGUF 路线的模型 | 纳入 `metal_readme_regression` 矩阵,16/64 throughput cell 完成零错误 |

阈值校准说明:±10% 与 0.5× 为初始值,W1 第一个模型与 W2 第一个模型跑完后按实测
复核一次并回写本节;"0.80× vLLM"级别的优化目标明确**不属于本目标**(那是 M3 式
单模型 perf goal 的范畴)。

## W1 — FREE/SMALL 即收(P0)

全部满足:需求榜上有名 + 架构差量为零或仅限 tokenizer/模板/loader + 单卡 24GB INT4 放得下。

| # | 模型 | 需求依据 | 架构差量(vs 现有 LlamaFamily/Qwen3MoE) | 格式计划 | INT4 VRAM | 工作量 |
|---|---|---|---|---|---|---|
| 1 | Qwen3-Coder-30B-A3B-Instruct | coding 是本地第一用例;qwen2.5-coder 遗产 16.6M pulls,本模型是其 MoE 直系后继、Coder-Next 的 24GB 替身 | 与 M3(Qwen3-30B-A3B)同构,仅 rope_theta=1e7、ctx 262144、XML 工具模板 | cpatonn GPTQ-4bit(验 group size)+ unsloth GGUF;官方 BF16/FP8 | ~16–17GB | FREE |
| 2 | DeepSeek-R1-0528-Qwen3-8B | reasoning 名号引流;R1 系 Ollama 87.4M 遗产盘;R1-0528 HF 6.49M | = Qwen3-8B + DeepSeek 模板/EOS(151643)+ YaRN factor 4(>32k 才需要) | QuantTrio GPTQ + GGUF 遍地 | ~5GB | FREE |
| 3 | R1-Distill-Qwen-32B(同批捎带 14B) | 同上,32B 是 24GB reasoning 经典位 | = Qwen2.5-32B + DeepSeek 模板(强制 `<think>\n`) | bartowski/unsloth GGUF + 社区 GPTQ/AWQ | ~17–18GB | FREE |
| 4 | Qwen3-32B / Qwen3-14B dense | 2025 代 24GB 标配对(Ollama qwen3 30.6M);填 README 矩阵 8B 以上空档 | 纯 config(与现有 Qwen3 dense 同) | 无官方 GPTQ;JunHowie/QuantTrio 社区 GPTQ + 官方 AWQ/GGUF | 32B ~17–18GB(ctx 收紧) | FREE |
| 5 | Qwen2.5-Coder-32B/14B | 2026 评测仍推荐其做补全;架构已支持 | 零(验证+alias+gate;FIM 模板不做) | 官方 GPTQ-Int4 / GGUF | ~18GB | 验证项 |
| 6 | Devstral Small 2 + Mistral Small 3.2 + Magistral Small(24B 线) | Devstral 2 在 Ollama 缺位 = 空档;24B 线是 Mistral 唯一 24GB 可跑线 | SMALL:Tekken tokenizer(vocab 131072)+ `language_model.*` 前缀重映射 + `[THINK]` token;Small 3.x 起无 SWA、全注意力;Pixtral 视觉塔跳过 | GGUF 为主(官方+bartowski);GPTQ 薄 → AWQ 审计的首个受益者 | ~13GB | SMALL |
| 7 | R1-Distill-Llama-70B(双卡) | 70B reasoning;搭 `llama33-70b-4bit-2x4090` 现成 lane(同为 Llama-3.3-70B 底) | = Llama-3.3-70B + DeepSeek 模板 | 同 70B lane(GGUF 4bit) | ~40GB(2×4090) | FREE(继承该 goal 前置) |

W1 公共工程项(做一次,全员受益):

- chat template 引擎补 DeepSeek/Mistral 模板用到的 Python 风格方法
  (`chat_template.rs` 目前手工 patch `.rstrip` 一例;R1 模板需要 `.split('</think>')[-1]`)。
  渲染失败时**禁止静默 fallback 到 builtin ChatML**——必须显式报错(W1 最大隐患)。
- EOS/BOS 一律取自各模型 `generation_config`,审计是否存在按家族猜测 EOS 的路径。
- YaRN rope_scaling 现状落证:支持则过 >32k 用例,不支持则文档声明 ctx 上限(不阻塞 W1)。
- 默认采样:reasoning 模型(R1 系/Magistral)按官方建议 temp 0.6 / top_p 0.95;
  对 `8c79fe8f` 引入的 repeat_penalty=1.1 在长推理链上做 A/B,防误伤。

预算与停损:W1 合计 ≤2 个 pod-day(Mac/Metal 先行,CUDA gate 批量一次跑);
单模型卡壳超过 1 pod-day → 记录、降级到 W2 末尾,不阻塞同批其余模型。

## W2 — MEDIUM 新家族(P1)

| # | 模型 | 需求依据 | 架构差量 | 战略副产物 | INT4 VRAM | 工作量 |
|---|---|---|---|---|---|---|
| 1 | GLM-4.7-Flash(30B-A3B,2026-01) | Ollama 1.3M pulls/约5个月;2026 上半年 30B 级 coding 口碑第一梯队 | sigmoid 路由 + e_score_correction_bias + shared expert + partial RoPE 0.5 + MTP 层跳过(Glm4MoeLite,细节 UNVERIFIED 需落证 config) | "DeepSeek 系路由包"可复用于未来 GLM/Kimi/Nemotron3 | ~16GB | MEDIUM |
| 2 | Gemma 3 27B(文本先行) | Gemma 线增速第一(gemma4 13.3M/2个月),Gemma3 自有大遗产盘;接入后才有资格谈 Gemma 4 | 5:1 SWA 逐层调度 + 双 rope 表(θ=10k/1M)+ GeGLU + 三明治 norm + query_pre_attn_scalar;视觉塔跳过 | "逐层异构注意力调度"是 Gemma 4 / gpt-oss / OLMo3 的共同前置地基 | ~15–16GB | MEDIUM |

约束:每模型先 Mac/CPU 数值对照(逐层 dump vs HF transformers)再上卡;
每模型 ≤2 pod-day;Paris 级正确性 gate 不过不进 bench。
Gemma 3 量化走 ISTA-DASLab GPTQ(4b-128g,Marlin 候选)+ Q4_K_M GGUF 双路。

## W3 — 战略大件(P2,决策门控,独立 goal)

本目标只交付立项合同,不在此宣称完成:

- **默认选项:Gated DeltaNet 混合注意力子系统**(recurrent state cache + chunked
  delta-rule kernel + shared-expert/512 专家 MoE 变体)。
  解锁:Qwen3.5-35B-A3B / Qwen3.6 线(当前 24GB 需求第一家族,官方 GPTQ-Int4 可直接
  喂 Marlin)+ Qwen3-Coder-Next-80B-A3B(INT4 约 40–43GB,2×4090 layer-split 内)。
- **备选(二选一):gpt-oss-20b**(attention sinks 内核 + MXFP4 路径 + harmony 模板,
  三件套只解锁一个家族;但它是工具调用口碑王,与 agent 定位契合,Ollama 10.1M pulls)。
- 默认 DeltaNet 优先的理由:解锁一条 2026 主线家族 + 差异化窗口
  (llama.cpp 合入耗时 2.5 个月且 CUDA 路径仍慢;vLLM 走 fla Triton kernel,
  ferrum 已有 triton_ptx 离线编译嵌入通道作为移植路径)。
- 进入条件:W1+W2 PASS;按下方执行合同立项;先做 50 行级 fla kernel 可移植性
  microbench 再谈整模型;每周 stop/go。

## 明确跳过(本轮不做,含理由)

| 模型 | 理由 |
|---|---|
| Gemma 4(26B-A4B/31B) | K=V 共享投影 = 新 KV 语义 + 逐层异构 head_dim(256/512)+ softcap;暂无 GPTQ/AWQ。W2 地基 + 量化生态成熟后重评 |
| DeepSeek V4-Pro/Flash、GLM-4.5-Air、Llama-4-Scout、gpt-oss-120b、Mistral Small 4 | INT4 超出 48GB envelope |
| Kimi-Linear-48B | KDA + MLA 双新 cache 类型(XL),量化生态最弱 |
| Mamba2 系(Granite 4.0-H、Nemotron Nano v2 / 3-Nano-30B) | 新 state cache 类型,需求位于头部之下;Granite 4.1 已回归纯 dense,若企业需求出现按 SMALL 候补 |
| OLMo 3/3.1 | research niche,无 GPTQ;post-norm + 全宽 QK-norm 不是现有 toggle |
| Qwen3-VL | XL 视觉管线,归属独立 VLM goal(DeepStack + MRoPE),不混入本目标 |
| Phi-4、Seed-OSS-36B | 需求证据不足(top-15 之外)/ 24GB 边缘;按 SMALL 候补池保留 |
| DiffusionGemma | text-diffusion 解码范式与引擎不符,观察 |

## 非目标

- 不与 llama.cpp/Ollama 拼架构数量;只做需求榜 + 税收可控的交集。
- 不做 expert CPU offload、不做 FP8/MXFP4/NVFP4(除非 W3 选了 gpt-oss)。
- 不降低既有 Metal/CUDA G0 gate 标准;不为新模型破坏 M2/M3 既有路径。
- 不接受静默 fallback(模板渲染失败静默换模板、GPTQ 不兼容静默走 dequant 慢路径
  而不告知)。
- 不在无同硬件 artifact 时做性能宣称。

## 执行合同模板(每模型开 pod 前填,CLAUDE.md 惯例)

```text
Lever:          <模型 + 接入层面(config/模板/loader/kernel)>
Expected gain:  <需求榜位次 + 解锁的用例>
Files:          <crates/ferrum-models/... 等>
Correctness gate: 验收 gate L0–L5 全过(见"验收 gate 定义",首败即停)
Benchmark gate: 按类型——同构复用 ≤10% 偏差 / 新家族 ≥0.5× llama.cpp + 全量 artifact;存量 C7/G0 不回退
Budget cap:     <pod-hour 数>
Stop condition: <正确性 gate 首败即停;卡壳 1 pod-day 降级>
```

## UNVERIFIED 清单(开工前必须落证)

1. ferrum YaRN rope_scaling 支持现状(影响 R1-0528-Qwen3-8B >32k、Qwen3 dense 长 ctx)。
2. `weight_format.rs` 中 AWQ 引用的真实能力(只是枚举还是有完整 loader)。
3. gguf loader 的 arch 白名单对 mistral(Tekken)/glm 的覆盖。
4. cpatonn / QuantTrio / JunHowie 各 GPTQ 的 group size、desc_act 与 Marlin 兼容矩阵。
5. GLM-4.7-Flash `Glm4MoeLite` config 全量字段(路由细节)。
6. Qwen3-Coder XML 工具模板与 ferrum caller-owned 工具注入的协同方式
  (用原生模板渲染 tools,还是沿用通用注入;二者 A/B 后定)。
7. Qwen3.6 线是否有官方 GPTQ-Int4(已确认 3.5 有、3.6 仅确认 FP8)——影响 W3 收益测算。

## 调研记录(2026-06-12)

三路独立调研,结论交叉验证:

1. **发布雷达(2026-01→06)**:Qwen3.5/3.6、Gemma 4、GLM-5/5.1、DeepSeek-V4、
   Kimi K2.5/2.6、MiniMax M2.5/M3、Nemotron 3、Granite 4.1、Mistral Small 4、
   Arcee Trinity、LFM2.5 等;关键否定项:Llama 5 不存在(Meta 转闭源)、
   DeepSeek R2 未发布、gpt-oss 无 2026 更新、GLM-4.6-Air 未发布、无 Phi-5。
2. **需求侧**:Ollama library 拉取数(2026-06-12 实时)、HF 下载/趋势榜、
   LM Studio 目录、llm-stats 开源榜、2026-04/05 多篇 24GB 实测指南;
   注意 Ollama 累计数偏向老模型,按"月均增速"修正。
3. **架构-格式矩阵**:15 个候选逐一核对 HF config.json + llama.cpp/vLLM 合入记录,
   输出 FREE/SMALL/MEDIUM/LARGE/XL 分级与量化可得性(本文各表的依据)。

原始三份报告未入库;若需要,从本 session 记录归档到本目录 `research/` 子目录。
