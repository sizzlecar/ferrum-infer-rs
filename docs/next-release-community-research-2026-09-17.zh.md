# 下一功能版建议：让现有硬件容纳可用的长上下文

调研日期：2026-09-17。对象：GitHub issues/discussions、Hugging Face 模型讨论与论坛、Reddit、Hacker News，以及 Ferrum 当前实现和正式发行。本文是产品建议，不是功能已经实现或性能已经验证的声明。

## 决策

**下一功能版只选一个主增量：把 8-bit KV cache 真正接入 vNext，让现有模型在相同硬件上容纳更长上下文，并在现有多会话路径中保持可用。自动持久化推理状态排第二，不与第一项捆成大版本。**

目标用户是已经用 CLI/API 跑本地 coding agent、受显存或统一内存限制的开发者。样本中已有用户为容量更换 runner；Ferrum 能否促成迁移仍待验证，产品应尽量让用户沿用现有模型和客户端。研究范围是 Ferrum 的模型与执行能力，不限定 Qwen3.5-9B；社区 Qwen 帖较多不代表其余架构不重要。

这不是独创技术：Ollama、llama.cpp 等已有量化 KV。它首先补 Ferrum 的实质能力缺口。**能否吸引用户换引擎，要靠“同机、同权重、同任务下，长上下文与少量并发的组合更好用”证明，不能靠新增参数名称。** 如果最终只达到已有替代方案的水平，应该按补齐能力发布，不能宣传性能领先。

## 为什么选择这个，而不是延续旧清单

| 候选 | 外部需求证据 | Ferrum 实际增量 | 本轮取舍 |
| --- | --- | --- | --- |
| 8-bit KV 支撑长上下文 | 用户明确为容量换 KV 类型、换 fork；不同模型也出现状态内存挤占空间 | vNext Metal/CUDA 标准 causal provider 固定 F16，旧 INT8 接口没有接入 | **P0，下一功能版主线**；先一档明确格式，完整打通相关资源与算子 |
| 自动持久状态恢复 | 慢 prefill 的长 agent 会话用户明确要求重启后自动恢复、无需客户端管理 | 有进程内 checkpoint，缺产品级落盘/恢复/磁盘预算 | **P1，后续独立功能**；可复用现有身份和合法恢复边界 |
| MTP / speculative decoding | 有用户报告明显速度提升，也有精度、显存与上下文取舍 | legacy 有，vNext 明确不支持 | 独立性能项目；先建立验证与回滚合约，不能拿别人的混合优化倍数承诺收益 |
| 部分权重 offload / SSD MoE | 模型权重本身装不下的人有明确需求 | 未见 vNext 产品实现 | 另一个容量项目；不与 KV 量化混淆，工程面更广 |
| 有界多模型常驻/热切换 | 有常驻小模型、按需大模型的工作流 | 当前单基座引擎；alias/LoRA 不等于多模型管理 | 后备；llama-swap、Ollama、llama.cpp router、oMLX 已有强替代 |
| 原生 Anthropic Messages、图像输入 | 有本地 Claude Code 和截图辅助编程场景 | 当前缺 `/v1/messages`，生产 Chat 拒绝多模态内容 | 是真增量，但本轮证据不足以压过容量主线；不能混称已有 OpenAI 接口完全覆盖 |
| 简单安装、权重复用、接客户端、基础 batching、再造压测 | 需求存在，但 Ferrum 已有基础能力 | 主要是局部修正或验收 | 不作为新主功能；保留必要修复和准确支持说明 |
| 合并短 kernel dispatch | 有源码空间，已有 DS4 零收益记录 | 未证明 Ferrum 端到端瓶颈 | 降为实验候选，不作为获客承诺 |

优先级是对用户痛点、代码缺口、替代方案和交付范围的综合判断，不是 issue 数、点赞数或人为加权分数的排名。没有 Ferrum 的新增用户转化/留存数据，不能预测增长百分比。

## 已核实的代码与发行基线

- 本地 HEAD：`b68576eb7e4f6ad4dc9526207bc7c41ba09bdb50`；当次 GitHub REST 核远端 main 为 `1cff9e1328a7048cb2d4b12af2e63665ecac1f7c`。
- 正式版已经是 [v0.10.0](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.10.0)，2026-09-16 发布，候选提交 `0a7d94f3a5fe8994a9bb12d0ff4b16de7fa2027c`。不能把 9/9、9/11 文档里的待办当成今日缺口。
- 安装、HF/GGUF、本地文件/目录、revision pin、CPU/Metal/CUDA 路径、Chat/无状态 Responses、工具调用、strict JSON、continuous batching、prefix/state cache、Rust bench-serve/decode-isolation 已有，不重新立项。
- 正式 CUDA 制品只覆盖 **sm89**；不能因 Reddit 上有人用 3090/5090，就声称 Ferrum 当前预编译包可直接覆盖这些设备。扩大发行硬件范围需要对应制品与真实设备验收。
- 本地已有的 9/17 旧建议撤回记录和 9/11 需求记录继续保留；它们未纳入本仓库，本次结论与证据在本文独立列出。

最关键的静态证据如下。行号基于本地 HEAD；审计也对照了 main，未运行新模型来验证参数的实际运行表现。

| 判断 | 实现位置 |
| --- | --- |
| registered vNext 在进入旧模型路径前返回 | [registry.rs](../crates/ferrum-engine/src/registry.rs)，1417–1424 行 |
| `config.kv_cache.dtype` 的 INT8 消费在 legacy Llama 分支 | 同文件 1620–1699 行；看到 `--kv-dtype int8` 不能认定 vNext 已压缩 |
| Metal causal KV 内存按 F16 计算，绑定拒绝非 F16 | [Metal causal_attention.rs](../crates/ferrum-kernels/src/backend/metal/vnext_ops/causal_attention.rs)，658–662、2150–2157 行 |
| CUDA causal KV 同样固定 F16 | [CUDA causal_attention.rs](../crates/ferrum-kernels/src/backend/cuda/vnext_ops/transformer/causal_attention.rs)，1146–1150、3595–3603 行 |
| 可参考的旧 CUDA INT8 kernel 已存在 | [int8_kv.rs](../crates/ferrum-kernels/src/backend/cuda/int8_kv.rs)，33、92 行；复用仍需满足 vNext 资源/状态合约 |
| vNext speculative 明确拒绝 | [builder.rs](../crates/ferrum-engine/src/builder.rs)，408–414 行；不是仅缺 CLI 开关 |
| native checkpoint 是进程内状态复用基础 | [checkpoint_access.rs](../crates/ferrum-interfaces/src/vnext/completion/checkpoint_access.rs)、[prefix_cache.rs](../crates/ferrum-models/src/executor/vnext_executor/prefix_cache.rs) |
| 当前服务是单引擎；session memory 保存消息而非磁盘 KV | [axum_server.rs](../crates/ferrum-server/src/axum_server.rs)，363–372、599–664 行 |

## 一手需求证据与反证

以下是决策用的目的性抽样，不是社区普查。优先近半年，旧讨论只采用相关的新回复。原帖是用户报告，不能直接作为技术根因、当前所有版本仍有故障、或 Ferrum 已胜出的证明。关联 issue/PR 和同一事件不重复计需求。

### 容量、长上下文与并发

| 编号 | 来源与时间 | 看到了什么 | 对结论的限制 |
| --- | --- | --- | --- |
| R1 | [Reddit：16GB 想要 100K+ context](https://www.reddit.com/r/LocalLLaMA/comments/1w7w2lv/which_quant_of_qwen38_27b_is_the_best_for_16gb/)，2026-09-05 | 用户现有配置约 64K，想在同一张卡上扩大本地编码上下文 | 证明容量诉求，不证明更长上下文必然提高任务成功率 |
| R2 | [Reddit：为 KV 类型改用 beellama](https://www.reddit.com/r/LocalLLaMA/comments/1w1lq7u/qwen_38_27b_at_50_toks_with_100k_context_on_a/)，08-29 | 用户为了长上下文更换 runner 分支，组合 KV 量化、权重、MTP 与模板 | 同帖试用者报告能装下却循环、未完成任务；不引用标题速度作为 Ferrum 收益或质量证明 |
| R3 | [Reddit：16GB 配置 200K context](https://www.reddit.com/r/LocalLLaMA/comments/1w04a5j/over_200k_context_on_16gb_vram_with_qwen_38_27b/)，08-27 | 通过低位权重与量化 KV，关闭额外组件换容量 | 作者明确 prefill 变慢、质量未测；容量与速度不是同一指标 |
| H1 | [HF：上下文与 VRAM 实测](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/87)，页面约 08-20 | 一位用户在同一设备/构建上比较不同 KV 类型；正文后续修正了“只取决于 KV”的过强结论 | 配置的内存增长还含其他 runtime 分配和 MTP；不能把该经验公式变成 Ferrum 通用容量预测器 |
| G1 | [mlx-lm #1631](https://github.com/ml-explore/mlx-lm/issues/1631)，07-28，08-01修正；Open | 低位 KV、旋转缓存的 sink policy 与 batch merge/extract 不能一致组合 | 部分模型使用自定义 keep=0 路径；不是所有模型均受影响。提示我们必须交付组合可用性 |
| G2 | [LM Studio lms #565](https://github.com/lmstudio-ai/lms/issues/565)，05-25；Open | 用户报告长 prompt 的少量并发请求出现空/截断输出，有单请求对照 | 未复现，正文与脚本 context 配置不一致；根因未定，不直接映射为 Ferrum bug |
| R4 | [Reddit：Concurrent MoE sessions](https://www.reddit.com/r/ollama/comments/1udftfx/concurrent_moe_sessions/)，06-23 | 用户要多个会话；回复明确希望按会话增长动态分配上下文容量 | 单会话内存不能简单乘成并发数；Ferrum 已有动态 admission，要测剩余限制而非重做 batching |
| R5 | [Reddit：Gemma 4 显存优化](https://www.reddit.com/r/LocalLLaMA/comments/1sb80yv/vram_optimization_for_gemma_4/)，04-03 | 权重能装下，状态与并行槽位却限制实际上下文；需求不限 Qwen | 帖子包含版本特定问题和未核技术解释；部分已修，不作为当前竞品缺陷宣传 |
| G3 | [vLLM #44294](https://github.com/vllm-project/vllm/issues/44294) / [PR #44295](https://github.com/vllm-project/vllm/pull/44295)，06-02；issue/PR 均 Open | offload 与 shared prefix 结合造成请求等待；修复仍需区分真实加载依赖 | 初版直接取消等待在中度内存压力下反而更慢。缓存命中不等于用户等待更短 |

### 持久恢复、切换和其他用户价值

| 编号 | 来源与时间 | 看到了什么 | 对结论的限制 |
| --- | --- | --- | --- |
| R6 | [Reddit：重启后长会话恢复](https://www.reddit.com/r/LocalLLaMA/comments/1vt8v4c/how_do_you_deal_with_longcontext_sessions_after/)，08-20 | 明确要服务端自动保存、最长有效前缀恢复、磁盘容量/LRU；知道 slot API，但不想每个客户端自己管理 | 主要是慢 prefill、长会话和频繁重启场景；评论指出身份兼容和读盘成本。公开信息未发现作者与 Ferrum 关联，匿名身份不能独立确认 |
| R7 | [Reddit：CachyLLama 操作者反馈](https://www.reddit.com/r/LocalLLaMA/comments/1v5k08a/cachyllamas_llamacpp_fork_with_persistent_kv/)，07-24 | 自称第三方使用者因重复 prefill 换 fork；评论希望不用编译多个分支、重启后仍能继续 | 作者承认没有 controlled benchmark，转引项目速度不算独立测量；有人指出上游已有 slot save |
| G4 | [llama.cpp #24043](https://github.com/ggml-org/llama.cpp/discussions/24043)，06-02，09-15有回复；Unanswered | recurrent/SSM 用户把保存恢复与 context-shift 结合时遇到限制 | 不能据此说 llama.cpp 所有 recurrent 状态都无法持久化；context-shift 与精确恢复是不同能力 |
| G5 | [llama.cpp cache 教程及回复](https://github.com/ggml-org/llama.cpp/discussions/13606)，采用2026-03-25、08-27回复 | 有分叉会话、不同调用者、冷启动与模型切换后复用诉求 | 教程本身已给手动 save/restore API；不要把“新增保存接口”当独特功能 |
| N1 | [HN：llama.cpp 多模型讨论](https://news.ycombinator.com/item?id=49267928)，页面35天前，约08月 | 常驻两个模型、按内存驱逐比固定超时更合适；有人用 llama-swap，有人满意原生 router | 精确日期 API 未取到；也有人只用单模型。替代充分、诉求不同 |
| G6 | [mlx-lm #1757](https://github.com/ml-explore/mlx-lm/issues/1757) / [PR #1837](https://github.com/ml-explore/mlx-lm/pull/1837)，08-18；09-11修复合并 | 用户报告切换模型后旧 buffer pool 驻留 | 搜索索引曾显示 Open，但当前已修；是否进入具体发行版未核，不能宣传竞品仍有此问题 |
| R8 | [Reddit：截图支持帮助编程](https://www.reddit.com/r/LocalLLaMA/comments/1w3vcvh/dont_sleep_on_vision_support_for_coding/)，08-31 | 用户描述模型通过截图发现测试未覆盖的 UI 错误 | 是个人工作流线索，未控制模型/客户端因素；多模态需要完整输入与执行支持，不是加一个路由 |

### 质量与接入：作为交付条件，不冒充新增主功能

| 编号 | 来源与时间 | 看到了什么 | 当前判断 |
| --- | --- | --- | --- |
| G7 | [llama.cpp #27966](https://github.com/ggml-org/llama.cpp/issues/27966)，08-29；Open | Pi 场景中并行工具的 SSE ID 重复，客户端误合并 | Ferrum 已有工具协议；将类似输入放入现有兼容性验证，不从零“做 agent 支持” |
| G8 | [llama.cpp #27129](https://github.com/ggml-org/llama.cpp/issues/27129)，08-15；Open/stale | 模板不支持 tools 时仍返回200，工具被静默忽略 | 保持能力校验和明确错误，不能靠客户端/模型名字特判 |
| H2 | [HF：工具链突然停止 #5](https://huggingface.co/DavidAU/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF/discussions/5)，用户标明09-11复现；页面 Closed | 工具结果后模型只描述下一步却不调用工具；多种量化均有人报告 | 维护者已引向模板修复并关闭；不是仍未解决的拉新线索。模型/模板问题不能全部算成 engine bug |

HF API 在本次 shell 读取中连接失败，HF 来源来自成功打开的官方网页；显示相对时间的记录保留近似日期。Reddit 内容包含公开索引/页面快照，不承诺实时完整评论清单。HN 精确时间未核。Open 只表示跟踪状态，不证明最新构建仍能复现。

## 竞品已有什么：避免虚构差异

- [Ollama FAQ](https://docs.ollama.com/faq)明确有 F16、Q8、Q4 KV，并提醒质量影响依模型和任务而变。Ferrum 首先需要补齐自己的 vNext 路径；“支持 Q8”单独不足以吸引别人迁移。
- [llama.cpp server 文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)和前述缓存教程已有 router、model presets、slot save/restore。自动匹配和有界清理才是持久恢复的具体产品增量。
- [oMLX](https://github.com/jundot/omlx)声明在 Apple Silicon 提供热/冷 KV、跨重启恢复、多模型 LRU/TTL，以及多种客户端入口。已有覆盖这些场景的替代方案，不能声称 Ferrum 将是首家；本次没有独立验证其稳定性或性能。
- DS4 自身也有持久恢复；此前阅读发现的 attention dispatch 合并在其单项提交中是 1.00×。因此不以“学 Redis 作者”替代用户价值证明。

## P0 的最小完整交付

1. **一个明确的 8-bit KV 存储 profile。** 复用现有 typed CLI/config，明确 scale、布局与数值语义。用户请求了不支持的模型/backend组合时准确拒绝，不能只接受参数却继续使用 F16。保留原有默认精度，避免静默数值变化。
2. **沿真实推理路径贯通。** 包含 prefill 写入、decode 读取、分页/复制、容量计算，以及现有 prefix checkpoint 捕获与恢复。相关共享改动同时覆盖 `ferrum run` 和 `ferrum serve`。这不是只在 microbench 里跑一个 INT8 kernel。
3. **按 provider 与状态形状选样。** 验证受影响的标准 causal 和混合 recurrent+causal 调用者；具体模型选能暴露边界且目标硬件可容纳的样本。Gemma/GPT-OSS 的专用 attention contract 必须单列支持状态。不要以一个9B成功推出所有模型都支持，也不把某参数量作为功能分支。
4. **首发后端边界明确。** Metal/CUDA 分别需要真实 kernel 和模型验证；哪一个未完成，就明确该 profile 尚不支持，而不是让 backend 编译通过代替 runtime 验证。CPU 可继续使用已有精度，不以新增 CPU INT8 为本轮隐含范围。
5. **不同时加入 Q4/FP8/多档混合精度、GDN recurrent 状态量化、权重 offload、MTP 和磁盘恢复。** 它们改变不同资源或计算合约，分别立项，避免一个用户问题变成整引擎改写。

先核目标负载的内存账：权重、KV、recurrent state、workspace、缓存副本分别占多少。如果主要瓶颈是权重或 F32 recurrent 状态，量化 KV 的收益可能很小，应在这个节点缩减或调整主线。这个检查利用现有 profile/容量工具即可，不新建一套平台。

F16 原始数据是2字节/值，8-bit是1字节/值，另有scale/对齐开销。**接近减半的只是 KV 数据部分，绝不是总显存或模型大小。** 实际可达上下文、并发数、TTFT、TPOT和质量都必须测量；没有证据承诺上下文翻倍、速度提升或无损。

## 怎样才足以发布并吸引试用

技术验收应回答三个用户问题：

- **同一台机器、同一权重，原来放不下的真实上下文，现在是否能完整运行？** 同时给进程/设备峰值、KV字节、有效上下文与并发，不能只给allocator理论值。
- **装下后能否继续完成任务？** 用固定token历史检查logits/参考续写概率，再跑真实多轮工具与长上下文任务；量化允许声明的数值差异，但不能把循环、工具丢失、空输出隐藏在吞吐数字里。沿用现有Rust工具。
- **多开一个任务的代价是多少？** 记录TTFT、输出token间隔/TPOT、排队、总吞吐和错误；沿用decode-isolation，覆盖长prefill对活跃decode的影响。SSE事件数不等于token数。

用同机配对比较当前正式 Ferrum 与候选，再比较该硬件上用户实际使用的成熟替代方案；权重、量化、输入、输出预算、上下文、采样与并发保持可比，注明有意差异。既展示单会话，也展示目标并发。遵守 AGENTS.md 的最小测试、工作区和受影响后端检查；本次仅文档调研，不将任何CI/模型检查标记为已完成。

发布文案以实测结果填空：

> 在【已验证硬件】上，用同一份【模型/权重】，Ferrum 能保留【实测上下文】并同时处理【实测会话数】；峰值内存【实测值】，首字等待和任务结果见对照。

不要预填“16GB跑任意27B”“100K无损”“快一倍”。Rust、单二进制、无需Python是降低试用摩擦的现有优势，新增容量结果才是这版的明确理由。

分发上可准备 [Hugging Face Local Apps](https://huggingface.co/docs/hub/local-apps) 的模型页入口：仅展示已验证来源和能力，复用现有run/serve命令，不新做GUI或扫描器。是否收录由上游决定，本轮未提交PR、未发帖、未联系社区。公开对照结果与可复现命令比重复发广告更有说服力。

衡量采用看“装好后完成目标任务”“继续第二次使用”“愿意用同一任务与原runner比较”，不以下载、star、帖子回复当留存。本次没有这些实际用户数据；需要发布后的自愿反馈才能验证吸引力，不能预先编一个增长目标。

## P1 保留什么

自动持久恢复的产品定义很清楚：客户端照常发送历史，服务端寻找兼容的最长合法前缀；重启后读盘恢复；磁盘有预算和清理；权重、tokenizer/template、数值profile、backend布局或状态ABI不兼容时miss重算。复用现有native checkpoint，不能把诊断tensor导出当会话恢复。

适合慢prefill、长会话、频繁重启或换出状态的用户。它节省重复prefill，不提高后续每token的基础decode速度。只在读取/恢复明显小于重算成本的目标场景宣传；先不与context-shift、语义摘要、跨backend恢复或多模型管理捆绑。

## 本次工作边界

完成外部来源阅读、状态/反证检查、Ferrum实现与发行去重及优先级建议。未执行新性能实验、未实现推理代码、未修改原有未跟踪文档、未提交GitHub内容或向社区发送消息。当前缺口仍是：量化KV在Ferrum目标负载的真实容量/质量/延迟收益，以及用户是否因此持续采用。
