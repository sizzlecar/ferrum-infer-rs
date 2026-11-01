# RTX 5090 热门文本模型并发对比（2026-11-01–02，UTC+8）

本轮使用正式版 Ferrum 0.18.3，对比同一张 RTX 5090 上的 llama.cpp 和 vLLM。7 组对照已全部完成：6 组每端各 256/256 个正式请求成功，GPT-OSS 自然 EOS 短测每端各 64/64 成功，但 Ferrum 未满足流式延迟 SLO。只比较同源权重、相同负载的配对结果，不跨模型或量化格式混排。

目标是满足 TTFT、TPOT 和可见文本 ITL 的延迟 SLO 后提高输出吞吐。超出任一必需 SLO 的结果会保留，但不计为达标优化。[27B 历史结果](performance-popular-models-p0.zh.md)单独保留，本轮不重复测试，也不将旧版本数字混入新表。

## 主要结果

预先固定 P99 SLO：TTFT ≤ 2000ms、TPOT ≤ 50ms、非空 SSE 文本事件 ITL ≤ 100ms，正式请求成功率 100%。合并吞吐为两轮成功 usage 输出 tokens 总数除以两轮测量时间总和。每端每档均为 128 个请求；Qwen3.5 每档输出 25412 tokens，Qwen3-30B 每档输出 25087 tokens。延迟分位数保留逐轮原值。

| 模型 / 源权重 | 对照 | C | Ferrum tok/s | 对照 tok/s | Ferrum / 对照 | 两轮 SLO |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B Q4_K_M | llama.cpp | 8 | 960.8 | 634.7 | 1.514× | 双方通过 |
| Qwen3.5-4B Q4_K_M | llama.cpp | 16 | 1294.8 | 729.4 | 1.775× | Ferrum 第 1 轮 ITL 失败；对照通过 |
| Qwen3.5-9B Q4_K_M | llama.cpp | 8 | 731.6 | 550.5 | 1.329× | 双方通过 |
| Qwen3.5-9B Q4_K_M | llama.cpp | 16 | 942.0 | 654.7 | 1.439× | Ferrum 第 1 轮 ITL 失败；对照通过 |
| Qwen3.5-4B 官方 BF16 | vLLM | 8 | 840.0 | 684.6 | 1.227× | 双方通过 |
| Qwen3.5-4B 官方 BF16 | vLLM | 16 | 1306.8 | 1153.9 | 1.132× | 双方通过 |
| Qwen3.5-9B 官方 BF16 | vLLM | 8 | 563.4 | 508.2 | 1.109× | 双方通过 |
| Qwen3.5-9B 官方 BF16 | vLLM | 16 | 875.0 | 864.3 | 1.012× | 双方通过 |
| Qwen3-30B-A3B GPTQ INT4 | vLLM | 8 | 630.5 | 824.0 | 0.765× | 双方通过 |
| Qwen3-30B-A3B GPTQ INT4 | vLLM | 16 | 848.8 | 1263.7 | 0.672× | 双方通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM | 8 | 717.6 | 625.6 | 1.147× | 双方通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM | 16 | 978.3 | 947.4 | 1.033× | Ferrum 通过；vLLM 第 1 轮 ITL 失败 |

官方 BF16 源权重对照中，Ferrum 实际采用 F16 激活/KV，vLLM 采用原生 BF16 激活/KV；两者均为 16 位，但数值路线不同。4B C16 第 1 轮 Ferrum 的 TTFT P99 为 486.18ms，高于 vLLM 的 173.06ms；可见 ITL P99 则为 38.00ms，低于 vLLM 的 63.88ms，双方均满足上述 SLO。吞吐优势不代表所有延迟分位都更低。

9B 官方 BF16 源组 C16 两轮分别为 Ferrum 802.5/953.9、vLLM 843.8/883.9 tok/s。两轮领先方向不同，合并仅差 1.2%，不作稳定显著优势结论。

30B GPTQ 组的 Ferrum 吞吐在 C8/C16 分别低 23.5%/32.8%。该组两端均为 F16 激活/KV，均满足 SLO；C16 第 1 轮 TTFT P99 为 578.95/109.97ms、TPOT P99 为 21.39/12.37ms（Ferrum/vLLM）。双方固定配置的两轮均已计入。

35B GPTQ 的 Ferrum 四轮均通过 SLO；vLLM C16 第 1 轮可见 ITL P99 为 136.73ms，超出 100ms 阈值。C16 合并吞吐差仅 3.3%，且逐轮领先方向不同，不能据此宣称稳定吞吐优势。该组 Ferrum 采用 F16 激活/KV，vLLM 为原生 BF16 激活/KV。

GPT-OSS 使用独立短测：默认 medium、自然 EOS、每请求最多 512 tokens，每档 32 个请求、仅 1 轮。吞吐包括推理 tokens，实际输出长度允许不同，下面的比例只描述该负载的原始吞吐，不是同答案加速倍数，也没有重复稳定性证据。

| GPT-OSS-20B MXFP4 | Ferrum tok/s | vLLM tok/s | Ferrum / vLLM | TTFT P99 ms（F/V） | SLO |
| --- | ---: | ---: | ---: | ---: | --- |
| C8 | 72.4 | 819.8 | 0.088× | 56304.24/387.13 | Ferrum 失败且缺 ITL；vLLM 通过 |
| C16 | 74.3 | 1429.8 | 0.052× | 109512.78/183.29 | Ferrum 失败且缺 ITL；vLLM 通过 |

Ferrum 的 Harmony 流式接口在生成结束后才交付整段文本，因此 TPOT 显示 0、ITL 无连续间隔，不能解释为零延迟；这不等于已经定位了整体吞吐差距的原因。双方大部分请求触及 512 上限，未验证每个请求都完成了最终答案。

## 比较范围

| 模型 | 源权重/量化格式 | 对照 | 状态 |
| --- | --- | --- | --- |
| Qwen3.5-4B | 同一份 Q4_K_M GGUF | Ferrum / llama.cpp | 双方测量完成 |
| Qwen3.5-4B | 同一份官方 BF16 权重 | Ferrum / vLLM | 双方测量完成；内部数值路线不同 |
| Qwen3.5-9B | 同一份 Q4_K_M GGUF | Ferrum / llama.cpp | 双方测量完成 |
| Qwen3.5-9B | 同一份官方 BF16 权重 | Ferrum / vLLM | 双方测量完成；内部数值路线不同 |
| Qwen3-30B-A3B | 同一份 GPTQ INT4 权重 | Ferrum / vLLM | 双方测量完成 |
| Qwen3.5-35B-A3B | 同一份 GPTQ INT4 权重 | Ferrum / vLLM | 双方测量完成；内部数值路线不同 |
| GPT-OSS-20B | 同一份 MXFP4 权重 | Ferrum / vLLM | 默认 medium 自然 EOS 双方完成；Ferrum SLO 失败，vLLM 通过；另保留固定输出协议失败 |

<details>
<summary>模型身份、冻结负载与服务端配置</summary>

GGUF 与 BF16/GPTQ/MXFP4 分开比较，不将不同量化文件的结果称为相同权重的三端对照。BF16 在上表中描述源权重格式，不代表所有引擎使用 BF16 中间激活。每组保留模型仓库 revision、权重 SHA256、config、tokenizer 与 chat template 身份，实际数值 profile、激活和 KV 精度另外记录。

已现场核对的 GGUF 字节身份：

上游来源沿用[已归档的 GGUF 输入清单](next-release-gguf-sources.zh.md#权重与语义来源)：[4B 权重固定 revision](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523) 与 [9B 权重固定 revision](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5)。本轮现场完整文件的 SHA256 与该清单记录一致。独立语义来源分别为 [Qwen 4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)、[Qwen 9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)，不以缓存文件名代替字节校验。

| 模型 | 权重字节数 | 权重 SHA256 | semantic config SHA256 |
| --- | ---: | --- | --- |
| Qwen3.5-4B Q4_K_M | 2740937888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` | `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| Qwen3.5-9B Q4_K_M | 5680522464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` | `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` |

两组使用的 tokenizer JSON SHA256 均为 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`，chat template SHA256 均为 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`。这是本轮实际文件身份，不仅是模型文件名。

4B 官方 SafeTensors 源为 `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，两片权重共 9319828096 bytes，逐文件 SHA256 为 `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61`、`cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188`；下载后逐片核对大小及 SHA256。其 config、tokenizer 和 template 与上面的 4B semantic source 相同。

9B 官方 SafeTensors 源为 `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`，四片权重共 19306310880 bytes，逐片 SHA256 为 `db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e`、`31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330`、`7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec`、`b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6`。完整 13 文件的实际大小/SHA256 已核对，其 config、tokenizer 和 template 与上面的 9B semantic source 相同。

30B GPTQ 源为 [Qwen/Qwen3-30B-A3B-GPTQ-Int4 的固定 revision](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/tree/9b534e4318b7ebc3c961a839f13eb18b1833f441)，量化声明为 INT4、group size 128、对称、`desc_act=false`。完整权重 16933256392 bytes，SHA256 为 `1ad50ff713f7be134a479990e9d70472f5c8ab12e0745d824e25d22f389c8495`；tokenizer JSON 为 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，包含模板的 tokenizer config 为 `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`。

35B GPTQ 源为 [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 的固定 revision](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/tree/3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b)，14 片权重共 24419939760 bytes，完整 25 文件的实际大小/SHA256 核验清单 SHA256 为 `272404f8f1796469cdbdf77cb3e7af69a1f3c0c41be93064f9481ca3bafacf00`。config SHA256 为 `13dd6a453236a33f16873dced7558fe7c9751e09f6c86ccc0cab66e3f23d4154`，tokenizer 和模板分别与上面的 Qwen3.5 tokenizer/template 相同。量化配置为 INT4、group size 128、对称、`desc_act=false`，并声明部分 attention/shared-expert 等模块不量化；不把源格式名称解释为所有模块均以 INT4 执行。源文件完整性与下文运行结果分开记录。

GPT-OSS 使用 [openai/gpt-oss-20b 的固定 revision](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee)，原生 MXFP4 三片权重共 13761316904 bytes，逐片 SHA256 为 `16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427`、`4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea`、`a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303`。tokenizer JSON 为 `0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3`，Harmony 模板为 `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146`；源文件完整性核验与运行结果分开记录。

## 冻结负载与 SLO

| 项目 | 设置 |
| --- | --- |
| 客户端并发 | C8、C16；服务端容量在整个 sweep 内固定 |
| 重复 | 默认每档 2 轮；每轮 10 次预热、64 次正式请求；GPT-OSS 自然 EOS 短测例外见下文 |
| 请求超时 | 120 秒 |
| 数据集 | ShareGPT_V3_unfiltered_cleaned_split.json |
| 原始文件 SHA256 | `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| 取样 | 首个 human/GPT 对；seed 42；每轮选样池 74；过滤，不截断 |
| 输入长度 | 原始用户文本 4–1024 tokens，未包含服务端模板 |
| 输出长度 | 默认按参考回答的实际 token 长度，限定为 128–256；GPT-OSS 的 512 上限与自然 EOS 例外见下文 |
| 总长度过滤 | 输入 + 请求输出 + 32 tokens 模板预留 ≤ 2048 |
| 生成策略 | `temperature=0`、`repetition_penalty=1`、`ignore_eos=true` |
| 思考策略 | 支持模板开关的模型两端均关闭；其他模型的策略单独记录 |
| P99 延迟 SLO | TTFT ≤ 2000ms；TPOT ≤ 50ms；可见 SSE 文本事件 ITL ≤ 100ms |
| 成功率要求 | 正式请求 100% 成功；预热失败另行记录 |

每个模型使用相同 tokenizer 选样，同一轮在两个引擎及两个并发档位使用完全相同的请求。两轮使用确定性的不同选样。核对 `dataset_evidence.repeats[].selection_sha256`、逐条 prompt/answer hash 和请求长度向量，并对齐服务端实际输入 token 数与生成 token 数。模板预留只是过滤预算，不能代替实际模板长度核对。

GPT-OSS-20B 最初冻结为 `reasoning_effort=low`，不发送 `enable_thinking`，保留 Harmony 模板；参考回答按 128–256 tokens 筛选，固定请求输出 512 tokens。该负载的 `ignore_eos=true` 触发了 Ferrum 0.18.3 的 Harmony 终止协议错误，原尝试保留为失败，不用于计算吞吐优势。

自然 EOS 的首个 `low` 诊断轮完成 64 个请求、无请求错误，正式测量耗时 390.0s，未形成完整矩阵。随后还发现正式客户端只将 `low` 放入 `chat_template_kwargs`，而此版本 vLLM 的 GPT-OSS 内部 Harmony builder 读取顶层 `reasoning_effort`，不能据此假定双方实际采用相同 `low`。[vLLM 渲染路径](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/renderers/online_renderer.py#L459)。

最终单独冻结默认策略短测：双方均不传 `ignore_eos` 和 `reasoning_effort`，使用模型默认 `medium`、每请求 `max_tokens=512`、C8/C16 每档 32 个正式请求和 2 次预热、1 轮。官方模板与实际 vLLM 镜像的 Harmony 默认均已确认为 `medium`；输入过滤、2048 总长度上限和 SLO 不变。此时 512 是上限，实际输出长度可以不同，应分别报告输出 token 分布；只核对选样、输入和请求预算一致，不强求实际输出向量相等。功能预检必须确认最终可见文本，缺失可见时序不能记为通过。该短测没有重复稳定性证据，不与固定输出组混排，也不覆盖此前两种诊断。

GPT-OSS 时序还受客户端口径限制：本版将 SSE 中非空 `content`、`reasoning` 或 `reasoning_content` 统一计为文本更新，性能报告没有分别保留 final 与 reasoning 的 token 用量。因此 TTFT 可能对应首段流式推理，TPOT 分母仍为全部 `completion_tokens - 1`；不能将其称为 final-only 首字延迟或最终答案的严格逐 token 速度。最终答案有效性单独检查，原始时序指标保留。

Ferrum 0.18.3 还有具体的 Harmony 流式限制：[`chat_stream.rs`](https://github.com/sizzlecar/ferrum-infer-rs/blob/e991c6955c3022f80d7e96f9d16d9306314054cf/crates/ferrum-server/src/axum_server/chat_stream.rs#L99) 对该协议启用完整输出缓冲，到引擎结束后才解析并发布内容。因此首个可见 SSE 可能接近整次完成时间；一次性文本交付的 TPOT 数值为 0、可见 ITL 无连续间隔，不能解释为零延迟或流畅逐 token 输出。缺失时序样本及 SLO 失败必须保留。这解释可见流式行为，不证明整体吞吐差距也由缓冲造成；算子吞吐根因尚未据此定位。

已完成的 Qwen3.5 4B/9B 报告均记录 `enable_thinking=false`，该推理通道限制不用于改写这些关闭思考的实测结果。

## 服务端容量与缓存口径

| 设置 | Ferrum | llama.cpp | vLLM |
| --- | --- | --- | --- |
| 单请求 context 上限 | `--max-model-len 2048` | unified KV：`--kv-unified-per-slot 2048`，总池 65536、32 slots | `--max-model-len 2048` |
| 最大活动序列 | `--max-num-seqs 32` | `--parallel 32` | `--max-num-seqs 32` |
| 调度逻辑 token 上限 | `--max-num-batched-tokens 2048` | `--batch-size 2048` | `--max-num-batched-tokens 2048` |
| 物理 microbatch | 引擎实际执行策略 | 默认 `--ubatch-size 512`，单独披露 | 引擎实际执行策略 |
| 前缀缓存 | `--disable-prefix-cache` | `--no-cache-prompt --cache-ram 0 --no-cache-idle-slots` | `--no-enable-prefix-caching` |
| GGUF 组 KV 精度 | FP16 | K/V 均 F16 | 不适用 |
| 自动容量调整 | 保存 effective config | `--fit off --no-context-shift` | 保存实际启动配置及 KV 容量 |

这些参数对齐请求容量及逻辑预算，不表示不同引擎的内核批大小、prefill/decode 调度或物理显存预留方式相同。llama.cpp 沿用既有基线的 `--kv-unified --kv-unified-per-slot 2048 --ctx-size 65536 --parallel 32 --ctx-checkpoints 0`，本轮短测确认后冻结；共享池总量与每请求上限分别记录。`--cache-reuse 0` 本身不能关闭 prompt cache。

统一 KV 布局与 checkpoint 设置属于性能配置：固定版本的 hybrid 非 unified 路线要求同一 microbatch 内的序列 ID 相邻；checkpoint 也不会随 `cache_prompt=false` 自动关闭。本表仅使用上述冻结的 unified、checkpoints=0 基线，不把其他配置诊断混入主结果。[相邻序列限制](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/llama-batch.cpp#L583)、[checkpoint 条件](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-context.cpp#L3377)。

Ferrum/vLLM 的 32 序列和 2048 context 是逻辑上限，不等于物理预留 65536 个 KV tokens。结果须附实际 KV dtype、池容量、GPU 层放置和注意力实现。GGUF 对照使用全部层 GPU offload；未达到该条件的结果单列，不混入同配置结论。

4B Ferrum 启动回执确认有效上限为 32/2048/2048，KV 为 F16，session cache 为 off，权重位于单设备，注意力策略为 `cuda_native_adaptive`。模型同时具有每序列固定状态，不能仅按 KV tokens 推算完整显存用量。

4B、9B llama.cpp 启动日志均确认 33/33 层 GPU offload、32 slots、总 context 65536、每 slot context 2048、Flash Attention enabled、物理 microbatch 512。CUDA KV buffer 均为 2048MiB，recurrent state buffer 均为 1608MiB；这些分配与整卡采样峰值的含义不同。

同权重不表示内部算术完全相同。4B、9B GGUF 的 Ferrum 均自动选中 `qwen3_5.f32-master.gguf-q6-head.q6-f16-mmq-v1`：主隐藏态边界为 F32，已准备的量化投影记录 F16 输入/输出，含按几何选择的 Q6 F16 路径，KV 为 F16；不能将整图笼统标成纯 FP16。4B llama.cpp 日志确认 KV 为 F16、recurrent R/S 为 F32，未据此推断其全部中间激活精度。

4B 官方 BF16 SafeTensors 组的 Ferrum 实际自动选择 `qwen3_5.f16`，主激活边界和 KV 为 F16，容量仍为 32/2048/2048。Ferrum 行按“BF16 源 / F16 执行”列出，与上面的 GGUF 组分开。

4B、9B SafeTensors 的 vLLM 基线采用原生 BF16 路线，命令为 `--dtype bfloat16 --kv-cache-dtype auto`，SSM 状态设为 F32。启动预检确认强制 `--dtype half` 会关闭融合 CUDA GDN 路径并回退 Triton；BF16 激活搭配 F16 KV 又在 Flash Attention warmup 因 query/key dtype 不同而失败。这些预检没有产生正式性能数据，也不计入正式请求错误。此组对齐官方源权重、负载和逻辑容量；Ferrum F16 KV 与 vLLM BF16 KV 均为 16 位存储，但保留不同数值范围，不能声称激活精度或内核算术相同。

30B GPTQ 的 Ferrum 自动选中 `qwen3_moe.f16`，KV 为 F16；vLLM 的 `--dtype auto --kv-cache-dtype auto` 实际解析为 F16 激活/KV，启动日志确认 `MarlinLinearKernel` 和 `MARLIN` WNA16 MoE。两端逻辑容量均保持 32/2048/2048。该组使用自己的 tokenizer 重新冻结选样，不套用 Qwen3.5 的长度向量；同为 F16 不表示全部融合、累加或调度策略相同。

35B GPTQ 的 Ferrum 自动选中 `qwen3_5.f16`、KV F16；vLLM 的 auto 实际为 BF16 激活/KV、SSM F32，GDN decode 为融合 CUDA，专家路径为 `MARLIN` WNA16 MoE / `MarlinExperts`。两端使用相同 GPTQ 检查点、模板和 32/2048/2048 逻辑容量，保留原生数值路线差异。

GPT-OSS 的 Ferrum 自动选中 `gpt_oss.f16-mxfp4-bf16`：外层 F16、MXFP4 专家计算路径为 BF16，KV 为 F16。vLLM 的 auto 实际为 BF16 激活/KV，使用 `MARLIN` MXFP4 MoE、FlashInfer attention，原生数值路线不同。双方默认 reasoning 为 medium，Harmony 系统日期均为 Asia/Shanghai 的 2026-11-02；容器只读挂载相同 `/etc/localtime` 并记录 `TZ=Asia/Shanghai`，没有修改模板。vLLM 实际 reasoning parser 为 `openai_gptoss`。

</details>

<details>
<summary>逐轮延迟、错误与样本数量</summary>

## 逐轮结果

每个模型/精度、引擎和并发档位保留两行，分别展示第 1、2 轮；GPT-OSS 默认 medium 短测仅一轮，明确标注。

| 模型 / 精度 | 引擎 | C | 轮次 | 成功/请求 | 错误/拒绝 | TTFT P50/P99 ms | TPOT P50/P99 ms | SSE ITL P50/P99 ms | 输出 tok/s | SLO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B Q4_K_M | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 59.18/261.45 | 8.06/10.95 | 5.51/56.44 | 906.7 | 通过 |
| Qwen3.5-4B Q4_K_M | llama.cpp b11429 | 8 | 1 | 64/64 | 0/0 | 81.68/370.26 | 12.11/13.62 | 9.23/63.81 | 622.9 | 通过 |
| Qwen3.5-4B Q4_K_M | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 47.12/242.25 | 7.00/8.93 | 5.44/53.29 | 1016.3 | 通过 |
| Qwen3.5-4B Q4_K_M | llama.cpp b11429 | 8 | 2 | 64/64 | 0/0 | 71.68/139.30 | 11.89/13.00 | 9.36/65.57 | 645.9 | 通过 |
| Qwen3.5-4B Q4_K_M | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 61.32/813.27 | 11.59/20.04 | 6.67/136.70 | 1103.6 | 失败：ITL |
| Qwen3.5-4B Q4_K_M | llama.cpp b11429 | 16 | 1 | 64/64 | 0/0 | 83.02/387.91 | 20.82/23.15 | 16.79/88.60 | 719.9 | 通过 |
| Qwen3.5-4B Q4_K_M | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 49.47/270.25 | 9.23/11.01 | 6.60/51.99 | 1538.7 | 通过 |
| Qwen3.5-4B Q4_K_M | llama.cpp b11429 | 16 | 2 | 64/64 | 0/0 | 76.22/244.57 | 20.39/22.23 | 16.74/82.23 | 738.3 | 通过 |
| Qwen3.5-9B Q4_K_M | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 74.04/500.18 | 10.87/14.95 | 7.17/86.44 | 673.9 | 通过 |
| Qwen3.5-9B Q4_K_M | llama.cpp b11429 | 8 | 1 | 64/64 | 0/0 | 80.27/273.54 | 14.03/15.11 | 11.03/65.62 | 543.7 | 通过 |
| Qwen3.5-9B Q4_K_M | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 51.09/268.68 | 9.13/10.89 | 7.11/69.95 | 793.7 | 通过 |
| Qwen3.5-9B Q4_K_M | llama.cpp b11429 | 8 | 2 | 64/64 | 0/0 | 70.26/270.60 | 13.62/14.80 | 11.17/71.81 | 556.8 | 通过 |
| Qwen3.5-9B Q4_K_M | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 146.82/1170.01 | 16.05/27.91 | 8.44/182.94 | 804.5 | 失败：ITL |
| Qwen3.5-9B Q4_K_M | llama.cpp b11429 | 16 | 1 | 64/64 | 0/0 | 85.41/642.24 | 22.92/25.25 | 18.71/86.18 | 638.5 | 通过 |
| Qwen3.5-9B Q4_K_M | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 91.98/537.05 | 11.98/16.53 | 8.29/95.13 | 1116.5 | 通过 |
| Qwen3.5-9B Q4_K_M | llama.cpp b11429 | 16 | 2 | 64/64 | 0/0 | 69.81/451.34 | 22.11/23.74 | 18.78/75.28 | 670.2 | 通过 |
| Qwen3.5-4B BF16 源 / F16 执行 | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 44.99/242.70 | 8.94/10.55 | 8.00/29.87 | 821.1 | 通过 |
| Qwen3.5-4B BF16 源 / BF16 执行 | vLLM 0.31.0 | 8 | 1 | 64/64 | 0/0 | 67.54/167.12 | 11.02/11.75 | 9.19/58.24 | 684.9 | 通过 |
| Qwen3.5-4B BF16 源 / F16 执行 | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 45.01/98.21 | 8.67/9.10 | 7.96/28.71 | 857.9 | 通过 |
| Qwen3.5-4B BF16 源 / BF16 执行 | vLLM 0.31.0 | 8 | 2 | 64/64 | 0/0 | 68.88/112.55 | 10.76/11.78 | 9.27/49.63 | 684.4 | 通过 |
| Qwen3.5-4B BF16 源 / F16 执行 | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 45.54/486.18 | 11.20/14.57 | 8.91/38.00 | 1215.9 | 通过 |
| Qwen3.5-4B BF16 源 / BF16 执行 | vLLM 0.31.0 | 16 | 1 | 64/64 | 0/0 | 78.41/173.06 | 12.46/13.85 | 9.68/63.88 | 1144.1 | 通过 |
| Qwen3.5-4B BF16 源 / F16 执行 | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 46.45/130.76 | 10.23/11.18 | 8.88/31.14 | 1402.6 | 通过 |
| Qwen3.5-4B BF16 源 / BF16 执行 | vLLM 0.31.0 | 16 | 2 | 64/64 | 0/0 | 80.28/115.79 | 12.34/13.94 | 9.63/70.85 | 1163.0 | 通过 |
| Qwen3.5-9B BF16 源 / F16 执行 | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 45.31/221.66 | 13.44/14.41 | 12.25/42.86 | 560.1 | 通过 |
| Qwen3.5-9B BF16 源 / BF16 执行 | vLLM 0.31.0 | 8 | 1 | 64/64 | 0/0 | 78.13/216.27 | 14.65/16.45 | 12.82/54.36 | 508.9 | 通过 |
| Qwen3.5-9B BF16 源 / F16 执行 | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 45.11/173.46 | 13.13/13.96 | 12.20/37.22 | 566.5 | 通过 |
| Qwen3.5-9B BF16 源 / BF16 执行 | vLLM 0.31.0 | 8 | 2 | 64/64 | 0/0 | 74.28/126.06 | 14.54/15.51 | 12.88/54.35 | 507.5 | 通过 |
| Qwen3.5-9B BF16 源 / F16 执行 | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 49.57/702.47 | 16.98/22.92 | 13.27/72.25 | 802.5 | 通过 |
| Qwen3.5-9B BF16 源 / BF16 执行 | vLLM 0.31.0 | 16 | 1 | 64/64 | 0/0 | 92.03/411.67 | 16.59/17.93 | 13.66/76.87 | 843.8 | 通过 |
| Qwen3.5-9B BF16 源 / F16 执行 | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 46.92/193.56 | 15.09/16.29 | 13.15/44.69 | 953.9 | 通过 |
| Qwen3.5-9B BF16 源 / BF16 执行 | vLLM 0.31.0 | 16 | 2 | 64/64 | 0/0 | 94.14/161.30 | 15.98/17.50 | 13.49/77.87 | 883.9 | 通过 |
| Qwen3-30B-A3B GPTQ INT4 | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 46.71/339.73 | 12.18/15.46 | 11.20/44.13 | 601.4 | 通过 |
| Qwen3-30B-A3B GPTQ INT4 | vLLM 0.31.0 | 8 | 1 | 64/64 | 0/0 | 45.61/164.53 | 8.93/9.74 | 7.83/40.01 | 814.7 | 通过 |
| Qwen3-30B-A3B GPTQ INT4 | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 45.56/154.62 | 11.34/12.73 | 9.90/39.86 | 663.3 | 通过 |
| Qwen3-30B-A3B GPTQ INT4 | vLLM 0.31.0 | 8 | 2 | 64/64 | 0/0 | 29.57/92.61 | 8.76/10.29 | 7.76/40.81 | 833.7 | 通过 |
| Qwen3-30B-A3B GPTQ INT4 | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 48.00/578.95 | 17.97/21.39 | 14.22/81.90 | 791.2 | 通过 |
| Qwen3-30B-A3B GPTQ INT4 | vLLM 0.31.0 | 16 | 1 | 64/64 | 0/0 | 61.37/109.97 | 11.52/12.37 | 9.71/55.88 | 1238.0 | 通过 |
| Qwen3-30B-A3B GPTQ INT4 | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 46.99/337.25 | 15.56/18.28 | 13.23/50.53 | 916.7 | 通过 |
| Qwen3-30B-A3B GPTQ INT4 | vLLM 0.31.0 | 16 | 2 | 64/64 | 0/0 | 59.15/93.07 | 11.11/12.28 | 9.48/51.63 | 1291.1 | 通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | Ferrum 0.18.3 | 8 | 1 | 64/64 | 0/0 | 47.08/246.26 | 10.92/12.61 | 9.24/42.04 | 693.1 | 通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM 0.31.0 | 8 | 1 | 64/64 | 0/0 | 103.55/232.90 | 11.75/13.54 | 9.02/89.98 | 627.2 | 通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | Ferrum 0.18.3 | 8 | 2 | 64/64 | 0/0 | 44.04/131.38 | 10.15/10.72 | 9.18/35.06 | 741.6 | 通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM 0.31.0 | 8 | 2 | 64/64 | 0/0 | 100.94/237.23 | 12.00/13.28 | 9.18/89.33 | 624.2 | 通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | Ferrum 0.18.3 | 16 | 1 | 64/64 | 0/0 | 48.89/736.40 | 16.29/21.21 | 11.57/75.99 | 890.6 | 通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM 0.31.0 | 16 | 1 | 64/64 | 0/0 | 137.97/388.13 | 15.50/18.63 | 10.84/136.73 | 921.9 | 失败：ITL |
| Qwen3.5-35B-A3B GPTQ INT4 | Ferrum 0.18.3 | 16 | 2 | 64/64 | 0/0 | 50.09/208.18 | 13.58/15.86 | 11.57/48.11 | 1075.2 | 通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | vLLM 0.31.0 | 16 | 2 | 64/64 | 0/0 | 115.97/228.37 | 14.79/17.90 | 10.82/93.94 | 972.0 | 通过 |
| GPT-OSS-20B MXFP4 / 默认 medium / 自然 EOS | Ferrum 0.18.3 | 8 | 1（短测） | 32/32 | 0/0 | 54648.74/56304.24 | 0/0（单事件） | N/A（0 间隔） | 72.4 | 失败：TTFT；ITL 缺失 |
| GPT-OSS-20B MXFP4 / 默认 medium / 自然 EOS | vLLM 0.31.0 | 8 | 1（短测） | 32/32 | 0/0 | 94.75/387.13 | 8.71/8.96 | 8.16/33.00 | 819.8 | 通过 |
| GPT-OSS-20B MXFP4 / 默认 medium / 自然 EOS | Ferrum 0.18.3 | 16 | 1（短测） | 32/32 | 0/0 | 100934.84/109512.78 | 0/0（单事件） | N/A（0 间隔） | 74.3 | 失败：TTFT；ITL 缺失 |
| GPT-OSS-20B MXFP4 / 默认 medium / 自然 EOS | vLLM 0.31.0 | 16 | 1（短测） | 32/32 | 0/0 | 174.08/183.29 | 10.14/10.38 | 9.47/34.24 | 1429.8 | 通过 |

4B Ferrum 客户端退出码为 1，原因是完整测量后 SLO 检查失败，不是请求失败。40 次预热及 256 次正式请求均成功，报告中的协议、坏输出、HTTP 500、拒绝及 panic 计数均为 0。C16 不能因第 2 轮通过而写成整档通过。

4B llama.cpp 的 40 次预热及 256 次正式请求同样全部成功，质量/协议错误计数均为 0，四轮均通过 SLO，客户端退出码为 0。双方完整选样记录、原始输入长度、服务端输入长度、输出长度及逐请求索引已逐项对齐；对应规范化证据 SHA256 为 `31b92e9a169b76e0c71c78ec5d3060a6ab90349c1e2a78cb6080ceb7c81e0139`。

9B 两端各 40 次预热及 256 次正式请求全部成功，质量/协议错误计数均为 0。Ferrum C16 第 1 轮可见 ITL P99 为 182.94ms，其他三轮满足 SLO；退出码 1 同样由完整测量后的 SLO 检查产生。llama.cpp 四轮均满足 SLO，退出码为 0。9B 的完整选样、长度向量和请求索引核对同样通过，规范化证据 SHA256 与 4B 相同；两组使用相同 tokenizer 和冻结负载。

9B 两端四轮各有 64 个 TTFT 和 64 个 TPOT 样本。按 C8 第 1/2 轮、C16 第 1/2 轮顺序，Ferrum 可见 ITL 间隔数为 12061/13171/12051/13167，llama.cpp 为 12064/13170/12064/13177；SSE 事件数与 usage 不同的请求数分别为 4/11/4/9 和 4/9/4/7。两端检出的传输合并、缺失 usage 及缺失时序均为 0。

4B 官方 BF16 源权重组两端各正式请求 256/256、预热 40/40 成功，四轮均满足 SLO，客户端退出码均为 0，质量/协议错误计数均为 0。完整选样、三组长度向量及请求索引逐项核对通过，规范化 SHA256 同为 `31b92e9a169b76e0c71c78ec5d3060a6ab90349c1e2a78cb6080ceb7c81e0139`。

该组两端四轮各有 64 个 TTFT、64 个 TPOT 样本。按相同轮次顺序，Ferrum 可见 ITL 间隔数为 12064/13146/12064/13148，vLLM 为 12054/13145/12050/13151；SSE 事件数与 usage 不同的请求数分别为 3/8/3/7 和 6/12/7/9。两端检出的传输合并、缺失 usage 及缺失时序均为 0。

9B 官方 BF16 源权重组的 Ferrum 实际采用 `qwen3_5.f16`、KV F16，vLLM 为 BF16 激活/KV、SSM F32、融合 CUDA GDN decode。两端各正式请求 256/256、预热 40/40 全部成功，四轮 SLO 通过、客户端退出 0，无质量/协议/请求错误。每轮各有 64 个 TTFT/TPOT 样本，Ferrum/vLLM 可见 ITL 间隔数为 12063/13166/12062/13166 和 12022/13160/12028/13159，SSE 事件数与 usage 不同的请求数为 3/9/4/9 和 9/10/9/10；检出的传输合并、缺失 usage 和缺失时序均为 0。两端完整选样、三组长度向量和请求索引逐项一致，规范化 SHA256 同为 `31b92e9a169b76e0c71c78ec5d3060a6ab90349c1e2a78cb6080ceb7c81e0139`。

30B GPTQ 两端各正式请求 256/256、预热 40/40 全部成功，四轮均满足 SLO，质量/协议错误计数均为 0。每轮各有 64 个 TTFT 和 TPOT 样本。按 C8 第 1/2 轮、C16 第 1/2 轮顺序，Ferrum 可见 ITL 间隔数为 12520/12301/12510/12303，vLLM 为 12527/12297/12543/12300；SSE 事件数与 usage 不同的请求数分别为 28/26/28/25 和 30/28/27/30。两端检出的传输合并、缺失 usage 及缺失时序均为 0。

30B 第 1 轮原始输入/服务端输入/输出总数为 6213/6981/12657 tokens，第 2 轮为 3912/4680/12430 tokens，每档两轮输出共 25087 tokens。两端完整选样、三组长度向量及请求索引逐项一致，规范化证据 SHA256 为 `fdffb2ae8cdf0e6b7f88e477d2bf5b526541b91268181913d2b2f7ae29b85146`。

35B GPTQ 两端正式请求各 256/256、预热各 40/40 全部成功，无质量/协议/请求错误。Ferrum 四轮 SLO 通过、客户端退出 0；vLLM C16 第 1 轮 ITL 失败，完整报告落盘后退出 1，另三轮通过。每轮各有 64 个 TTFT/TPOT 样本，按 C8 第 1/2 轮、C16 第 1/2 轮顺序，Ferrum/vLLM 可见 ITL 间隔数为 12018/13190/12019/13181 和 12014/13121/12002/13114，SSE 事件数与 usage 不同的请求数为 5/6/5/8 和 11/9/11/11；检出的传输合并、缺失 usage 和缺失时序均为 0。两端完整选样、实际输入/服务端输入/输出向量与请求索引逐项一致，规范化 SHA256 同为 `31b92e9a169b76e0c71c78ec5d3060a6ab90349c1e2a78cb6080ceb7c81e0139`。独立功能请求两端均正常返回 `4`，输入/输出为 26/2 tokens。

GPT-OSS 原固定输出尝试在 C8 第 1 轮记录 10 条 Harmony 协议错误后中断，没有完整 `bench.json`，控制器记录客户端退出码 15，后续轮次未执行。错误为 `terminal control token was followed by duplicate terminal data or trailing garbage`，完成 usage 缺失；这是不完整的失败尝试，不能写成整组请求成功率或吞吐。源码对应行为是 `ignore_eos` 取消 EOS 停止，却仍允许采样 Harmony 终止标记，随后生成的尾随数据被严格解析器拒绝。独立正常 EOS 功能请求返回最终答案 `4`，并不使该固定输出负载转为通过。

GPT-OSS 默认 medium 的 Ferrum 短测两档各 32/32 正式请求、2/2 预热成功，协议/质量/请求错误均为 0，客户端在保存完整 JSON 后因 SLO 失败退出 1。两档各只有 32 个非空文本事件、0 个连续间隔，32 个请求的事件数均不等于 usage tokens；无缺失 usage，未检出传输合并。`last_visible_tpot_ms` 的 0 是一次性交付的计算结果，不能作为逐 token 延迟证据。

该短测每档相同的原始输入/服务端输入合计为 4440/6584 tokens，选样 SHA256 为 `ceec98e3917466c7dc7347fc0522b2ef1626dd35651747d1018b65b05c562616`。Ferrum C8 输出 15176 tokens，单请求范围 190–512、均值 474.25，27/32 个请求触及 512 上限；C16 输出 14999 tokens，范围 194–512、均值 468.72，26/32 个触及上限。测量时间分别为 209.551576669s 和 201.874913683s。独立默认 medium 功能请求最终文本为 `4`，HTTP 200、正常结束，输入/输出为 81/36 tokens。

vLLM 短测两档各 32/32 正式请求、2/2 预热成功，无质量/协议/请求错误，客户端退出 0、两档 SLO 均通过。C8/C16 分别输出 15064/15115 tokens，范围 188–512/117–512、均值 470.75/472.34；上限命中分别为 27/32、26/32。测量时间为 18.375894987s/10.571745072s。每档各有 32 个 TTFT/TPOT 样本，可见文本事件为 14879/14934 个、连续 ITL 间隔为 14847/14902 个；全部 32 个请求的事件数与 usage 不同，没有缺失 usage、缺失可见时序或检出的传输合并，故可见事件 SLO 通过不等于严格单 token 时序通过。

两端完整选样、512 请求预算、原始/服务端输入长度和请求索引逐项一致；规范化输入配对证据 SHA256 为 `09f8d53a5bbc40b1804d261dbd09960d8a115763bb63a53d23dc2481a71915e7`，实际输出向量按自然 EOS 策略分别保留。vLLM 独立功能请求也返回最终文本 `4`、HTTP 200、正常结束，输入/输出同为 81/36 tokens；其功能原件另记录 25 个 reasoning tokens，不用该单次样本补造整轮性能报告的 reasoning/final 分拆。

GPT-OSS 吞吐按总 `completion_tokens` 计，包含推理，不是仅最终答案 token 数。上述大部分请求触及输出预算，存在回答截断；Rust 性能报告未单独保存每请求 final 文本及完整性覆盖，不能据“32/32 协议成功”或单独功能请求正确，声称 32 个完整最终答案均正确。保留上限命中和输出长度分布，不将截断产生的较短工作量解释为同答案速度优势。

Qwen3.5 两轮取样的长度不同：第 1 轮每档原始输入/服务端输入/输出总数为 8041/8809/12138 tokens，第 2 轮为 4097/4865/13274 tokens。跨引擎应按同一轮配对，不将轮间吞吐差单独归因于暖机或性能波动。

| 4B 时序样本 | TTFT 请求数 | TPOT 请求数 | 可见 ITL 间隔数 | SSE 事件数与 usage 不同的请求 | 检出的传输合并请求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ferrum C8 第 1 轮 | 64 | 64 | 12066 | 4 | 0 |
| llama.cpp C8 第 1 轮 | 64 | 64 | 12068 | 2 | 0 |
| Ferrum C8 第 2 轮 | 64 | 64 | 13178 | 5 | 0 |
| llama.cpp C8 第 2 轮 | 64 | 64 | 13174 | 5 | 0 |
| Ferrum C16 第 1 轮 | 64 | 64 | 12068 | 2 | 0 |
| llama.cpp C16 第 1 轮 | 64 | 64 | 12063 | 5 | 0 |
| Ferrum C16 第 2 轮 | 64 | 64 | 13150 | 5 | 0 |
| llama.cpp C16 第 2 轮 | 64 | 64 | 13161 | 5 | 0 |

上述轮次均无缺失 usage、缺失可见时序或间隔数量不一致；事件数不等于 token 数的请求仍保留在可见 SSE ITL 中，不作为严格逐 token 时序证据。

指标直接取自 Rust 客户端的 `client_slo.repeats[]`：

- TTFT：`ttft_ms`，请求开始到首个非空 SSE 文本更新；若服务端流式返回 reasoning，该字段也计入。
- TPOT：`last_visible_tpot_ms`，首个到最后一个可见更新的时间除以 `usage output tokens - 1`；不包含末尾控制消息等待。
- ITL：`visible_sse_itl_ms`，连续非空 SSE 文本更新的客户端可见间隔，包括流式 content/reasoning。排除 role-only、空内容和 finish-only 消息；保留传输合并造成的停顿及相关计数。
- 输出吞吐：`successful_output_throughput_tps`，成功请求的服务端 usage 输出 token 数除以整轮测量时间；失败工作占用的时间不会被扣掉。
- 状态和错误：`status`、`completed_requests`、`offered_requests`、`errors`、`rejected_requests`；拒绝数包含 HTTP 429/503，其他 HTTP 错误计入 errors。

同时保留 TTFT/TPOT 请求样本数、ITL 间隔样本数、预热结果、传输合并计数、缺失 usage/时序、坏输出、协议异常和 HTTP 500。严格单 token ITL 只作诊断，不替代可见 SSE ITL。

不把各轮 P50/P99 的平均值标为池化分位数。默认每轮 TTFT/TPOT 只有 64 个请求样本，GPT-OSS 短测每轮仅 32 个；本轮结果不建立置信区间，细小差异不作稳定优劣结论。缺失指标或输出有效性失败不记为 SLO 通过。

</details>

<details>
<summary>显存、Host 内存与有效性证据</summary>

## 资源与有效性证据

| 模型 / 引擎 | 采样窗口 | 采样间隔及覆盖 | Peak VRAM | RSS 范围及峰值 | 缺失项 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B Q4_K_M / Ferrum | C8+C16 整个 sweep，含准备/预热，253.010s | 配置 250ms；GPU 978、Host 1014 点；采样错误 0；最大间隔 GPU 1804ms、Host 252ms | 整卡 6.387GiB | Ferrum 进程窗口采样峰值 1.862GiB | WSL 目标进程显存不可取得 |
| Qwen3.5-4B Q4_K_M / llama.cpp | C8+C16 整个 sweep，含准备/预热，177.505s | 配置 250ms；GPU 642、Host 712 点；GPU 查询失败 4、Host 错误 0；最大间隔 GPU 2105ms、Host 259ms | 整卡已观测峰值 8.087GiB | llama.cpp 进程窗口采样峰值 1.240GiB | WSL 目标进程显存不可取得；GPU 存在采样缺口 |
| Qwen3.5-9B Q4_K_M / Ferrum | C8+C16 整个 sweep，含准备/预热，110.755s | 配置 250ms；GPU 361、Host 445 点；GPU 查询失败 3、Host 错误 0；最大间隔 GPU 2209ms、Host 251ms | 整卡已观测峰值 9.238GiB | Ferrum 进程窗口采样峰值 1.924GiB | WSL 目标进程显存不可取得；GPU 存在采样缺口 |
| Qwen3.5-9B Q4_K_M / llama.cpp | C8+C16 整个 sweep，含准备/预热，156.766s | 配置 250ms；GPU 499、Host 629 点；GPU 查询失败 8、Host 错误 0；最大间隔 GPU 2172ms、Host 259ms | 整卡已观测峰值 10.304GiB | llama.cpp 进程窗口采样峰值 1.300GiB | WSL 目标进程显存不可取得；GPU 存在采样缺口 |
| Qwen3.5-4B BF16 源 / F16 执行 / Ferrum | C8+C16 整个 sweep，含准备/预热，96.754s | 配置 250ms；GPU 382、Host 389 点；采样错误 0；最大间隔 GPU 1066ms、Host 252ms | 整卡已观测峰值 11.104GiB | Ferrum 进程窗口采样峰值 1.455GiB | WSL 目标进程显存不可取得 |
| Qwen3.5-4B BF16 源 / BF16 执行 / vLLM | C8+C16 整个 sweep，含准备/预热，156.505s | 配置 250ms；GPU 623、Host 627 点；采样错误 0；最大间隔 GPU 836ms、Host 271ms | 整卡已观测峰值 29.384GiB | EngineCore worker 窗口采样峰值 3.253GiB | WSL 目标进程显存不可取得；worker RSS 不含 API/其他进程 |
| Qwen3.5-9B BF16 源 / F16 执行 / Ferrum | C8+C16 整个 sweep，含准备/预热，92.005s | 配置 250ms；GPU 342、Host 370 点；采样错误 0；最大间隔 GPU 1877ms、Host 251ms | 整卡已观测峰值 19.987GiB | Ferrum 进程窗口采样峰值 1.396GiB | WSL 目标进程显存不可取得 |
| Qwen3.5-9B BF16 源 / BF16 执行 / vLLM | C8+C16 整个 sweep，含准备/预热，98.004s | 配置 250ms；GPU 356、Host 394 点；GPU 查询失败 1、Host 错误 0；最大间隔 GPU 2212ms、Host 265ms | 整卡已观测峰值 28.675GiB | EngineCore worker 窗口采样峰值 3.311GiB | WSL 目标进程显存不可取得；GPU 存在采样缺口；worker RSS 不含其他进程 |
| Qwen3-30B-A3B GPTQ INT4 / Ferrum | C8+C16 整个 sweep，含准备/预热，170.505s | 配置 250ms；GPU 678、Host 684 点；采样错误 0；最大间隔 GPU 832ms、Host 252ms | 整卡已观测峰值 18.668GiB | Ferrum 进程窗口采样峰值 1.711GiB | WSL 目标进程显存不可取得 |
| Qwen3-30B-A3B GPTQ INT4 / vLLM | C8+C16 整个 sweep，含准备/预热，141.755s | 配置 250ms；GPU 558、Host 569 点；采样错误 0；最大间隔 GPU 1201ms、Host 259ms | 整卡已观测峰值 30.008GiB | EngineCore worker 窗口采样峰值 3.130GiB | WSL 目标进程显存不可取得；worker RSS 不含 API/其他进程 |
| Qwen3.5-35B-A3B GPTQ INT4 / Ferrum | C8+C16 整个 sweep，含准备/预热，75.005s | 配置 250ms；GPU 287、Host 302 点；采样错误 0；最大间隔 GPU 1244ms、Host 275ms | 整卡已观测峰值 23.630GiB | Ferrum 进程窗口采样峰值 3.066GiB | WSL 目标进程显存不可取得 |
| Qwen3.5-35B-A3B GPTQ INT4 / vLLM | C8+C16 整个 sweep，含准备/预热，81.005s | 配置 250ms；GPU 284、Host 326 点；GPU 查询失败 1、Host 错误 0；最大间隔 GPU 2186ms、Host 275ms | 整卡已观测峰值 29.802GiB | EngineCore worker 窗口采样峰值 3.494GiB | WSL 目标进程显存不可取得；GPU 存在采样缺口；worker RSS 不含其他进程 |
| GPT-OSS-20B 默认 medium / Ferrum | C8+C16 短测整个 sweep，含准备/预热，452.509s | 配置 250ms；GPU 1804、Host 1812 点；采样错误 0；最大间隔 GPU 1273ms、Host 269ms | 整卡已观测峰值 15.990GiB | Ferrum 进程窗口采样峰值 1.389GiB | WSL 目标进程显存不可取得 |
| GPT-OSS-20B 默认 medium / vLLM | C8+C16 短测整个 sweep，含准备/预热，37.512s | 配置 250ms；GPU 147、Host 152 点；采样错误 0；最大间隔 GPU 744ms、Host 271ms | 整卡已观测峰值 29.937GiB | EngineCore worker 窗口采样峰值 3.145GiB | WSL 目标进程显存不可取得；worker RSS 不含 API/其他进程 |

复用已有 Rust sampler，以 250ms 间隔采样。若一个 monitor 覆盖整个 C8/C16 sweep，峰值明确标为整个 sweep（含预热），不伪装成各并发档位独立峰值。WSL 若只能提供整卡显存，列名及说明保留“整卡”；vLLM 的 API 父进程 RSS 不代表全部 worker。采样失败、最大间隔、其他进程驻留和不可取得的指标均保留。

本机为 RTX 5090，驱动 610.88，设备报告总显存 32607MiB、功耗上限 600W；CPU 为 Core Ultra 9 285K、24 个可见逻辑处理器，系统为 Ubuntu 26.04.1 / WSL2。4B 功能检查前观测到整卡已用 1181MiB、GPU 利用率 0%；原有服务进程保留。后续整卡显存峰值包含这些既有驻留，不声称整张卡的全部显存只属于被测进程。

4B Ferrum 的最后一次 `VmHWM` 为 3.389GiB，这是进程生命周期高水位，包含采样窗口前分配，与表中的窗口 RSS 峰值不同。显存采样最大间隔曾达到 1.804s，不能排除间隔内更高的瞬时峰值。原始字节峰值分别为整卡 6857687040、窗口 RSS 1999106048、生命周期 RSS 3639144448。

4B llama.cpp 的生命周期 `VmHWM` 为 2.919GiB，也不作为窗口峰值。其 GPU 查询存在 4 次失败，表中 8.087GiB 仅是成功样本的峰值，不据此证明没有更高的瞬时显存占用。

9B Ferrum 生命周期 `VmHWM` 为 6.170GiB，llama.cpp 为 5.655GiB；窗口采样范围和 GPU 查询缺口同样不能替代进程从启动开始的连续内存观测。

4B BF16 源 / F16 执行组的 Ferrum 生命周期 `VmHWM` 为 8.663GiB，包含窗口前的权重加载分配；表中的 1.455GiB 仅为测量窗口内 RSS 采样峰值。

9B BF16 源 / F16 执行组的 Ferrum 生命周期 `VmHWM` 为 17.589GiB，窗口 RSS 峰值 1.396GiB；生命周期包含加载分配。

9B 官方 BF16 源组的 vLLM EngineCore 生命周期 `VmHWM` 为 6.627GiB，独立容器 cgroup `memory.peak` 为 11133145088 bytes（10.369GiB），OOM/kill 计数为 0。其 0.9 利用率预留的 KV 可用内存为 9.53GiB、池容量 172909 tokens，GPU 存在一次查询缺口。

4B vLLM EngineCore 生命周期 `VmHWM` 为 6.451GiB；独立新容器的 cgroup `memory.peak` 为 8023486464 bytes（7.472GiB），包含加载、预热、全部 worker 和 page cache，不是进程 RSS。其 `memory.events` 中 OOM/kill 计数为 0。vLLM 按 0.9 利用率预留 KV，启动报告 KV 可用内存 19.07GiB、池容量 346112 tokens；GDN decode 为融合 CUDA。29.384GiB 整卡占用包含该预留，不能直接将与 Ferrum 的显存差解读为同等活跃数据的内存效率差。

30B Ferrum 生命周期 `VmHWM` 为 16.670GiB，包含权重加载阶段；窗口 RSS 1.711GiB 不能代表完整进程生命周期峰值。

30B vLLM EngineCore 生命周期 `VmHWM` 为 17.268GiB；容器 cgroup `memory.peak` 为 22336495616 bytes（20.802GiB），包含全部 worker、加载和 page cache，OOM/kill 计数为 0。按 0.9 利用率预留 KV，启动报告 KV 可用内存 12.0GiB、池容量 131056 tokens；仍使用固定 32 活动序列和 2048 单请求上限。

35B GPTQ 的 Ferrum 生命周期 `VmHWM` 为 22.367GiB，包含权重加载；窗口采样 RSS 峰值为 3.066GiB，二者分开记录。

35B GPTQ 的 vLLM EngineCore 生命周期 `VmHWM` 为 4.531GiB，独立容器 cgroup `memory.peak` 为 8915484672 bytes（8.303GiB），OOM/kill 计数为 0。按 0.9 利用率预留的 KV 可用内存为 6.87GiB、池容量 139673 tokens；GPU 查询缺口限制了瞬时峰值结论。

GPT-OSS Ferrum 生命周期 `VmHWM` 为 13.482GiB，包含加载和此前独立诊断阶段；默认 medium 短测前已核实 active/queued 为 0，窗口 RSS 峰值仅为 1.389GiB，不把复用服务进程的生命周期高水位冒充本轮峰值。

GPT-OSS vLLM EngineCore 生命周期 `VmHWM` 为 5.925GiB，独立容器 cgroup `memory.peak` 为 8277020672 bytes（7.709GiB），OOM/kill 计数为 0。其 0.9 利用率预留的 KV 可用内存为 12.65GiB、池容量 275324 tokens；整卡显存同样包含预留池和其他驻留，不能与 Ferrum 的按需峰值直接推导内存效率倍数。

功能检查与性能分开：先验证模型加载及正常回答，再检查正式请求的 HTTP/SSE、完成标记、usage 和长度。协议通过不等于通用回答质量评测。本轮不发布模型准确率结论。

</details>

<details>
<summary>版本、复现命令与原始报告 SHA256</summary>

## 复现

Ferrum 服务端与 benchmark 客户端均为正式 0.18.3，源码提交 `e991c6955c3022f80d7e96f9d16d9306314054cf`，Linux SM120 CLI SHA256 为 `76094c69c77d0a1e6d675900c0a91b28bdee3d4237c930193abdac709ebee087`。llama.cpp 为 b11429（`d81235049384534c167caea52b85a694f6103d14`）。vLLM 为 0.31.0（`db9527a46873454610df6dbedf79a36d6bf1a7f6`），本机缓存镜像 ID 为 `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`。镜像标签记录 `vllm/vllm-openai:v0.31.0`，但现有 RepoDigest 仅指向本地缓存 registry；未归档可公开拉取的 registry digest，不能将该镜像 ID 拼成官方下载地址。后续复现应核对实际版本、源码 revision 和镜像身份。

GGUF 服务端命令模板如下；4B、9B 只替换对应模型文件、语义目录和输出文件。双方分别启动、测量、退出，不同时占用 GPU。

```sh
ferrum serve /path/to/model-Q4_K_M.gguf \
  --semantic-source /path/to/official-model-metadata \
  --tokenizer-source /path/to/official-model-metadata \
  --backend cuda --disable-thinking \
  --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --kv-dtype fp16 --gpu-memory-utilization 0.9 --disable-prefix-cache \
  --host 127.0.0.1 --port 19081 --served-model-name comparison \
  --effective-config-json /path/to/effective.json

llama-server --model /path/to/model-Q4_K_M.gguf --alias comparison \
  --host 127.0.0.1 --port 19082 \
  --ctx-size 65536 --parallel 32 --batch-size 2048 --ubatch-size 512 \
  --cache-type-k f16 --cache-type-v f16 --n-gpu-layers all --flash-attn on \
  --kv-unified --kv-unified-per-slot 2048 --ctx-checkpoints 0 \
  --no-cache-prompt --cache-ram 0 --no-cache-idle-slots --no-context-shift \
  --chat-template-file /path/to/official-model-metadata/chat_template.jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --metrics --fit off --log-verbosity 4
```

Qwen3.5 官方 SafeTensors 组的 Ferrum 沿用上述容量和缓存参数，将模型、`--semantic-source`、`--tokenizer-source` 三处均替换为同一个完整 SafeTensors 目录，数值 profile 保持默认 auto。对应 vLLM 命令如下；其原生 BF16 数值差异已在主表说明。

```sh
vllm serve /path/to/official-safetensors \
  --tokenizer /path/to/official-safetensors --served-model-name comparison \
  --host 127.0.0.1 --port 19083 \
  --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --dtype bfloat16 --kv-cache-dtype auto --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching --language-model-only \
  --mamba-ssm-cache-dtype float32 --generation-config vllm \
  --chat-template /path/to/official-safetensors/chat_template.jinja
```

30B GPTQ 的 Ferrum 同样替换三处模型目录并保留其余参数。vLLM 使用上面的固定容量/缓存参数，改为 `--dtype auto --kv-cache-dtype auto`，去掉 Qwen3.5 专用的 `--mamba-ssm-cache-dtype`、`--language-model-only` 和外部 `--chat-template`，使用该 GPTQ 源目录自带的 tokenizer 模板；其实际激活/KV 类型为 F16。

35B GPTQ 的 Ferrum 将三处目录替换为该完整检查点，其余参数不变。vLLM 沿用 Qwen3.5 的命令，包括 `--language-model-only --mamba-ssm-cache-dtype float32` 和该目录的 `chat_template.jinja`，仅将 `--dtype` 改为 `auto`；实际解析为 BF16 激活/KV。

GPT-OSS 的 Ferrum 替换为完整 MXFP4 模型目录，删除 `--disable-thinking`，其余容量/缓存参数不变。vLLM 沿用 30B 的固定容量参数和 `--dtype auto --kv-cache-dtype auto`，增加 `--reasoning-parser openai_gptoss`，使用内部 Harmony 渲染；容器 `--env TZ=Asia/Shanghai --mount type=bind,source=/etc/localtime,target=/etc/localtime,readonly` 与 Ferrum 主机日期对齐。双方不覆盖默认 reasoning effort。

以下是客户端模板；将模型名、地址、tokenizer、数据集和输出路径替换为对应组保存的命令。每个引擎只运行一次 sweep，不为生成 Markdown 再测一次。

```sh
ferrum bench-serve \
  --base-url http://127.0.0.1:19081 --model comparison \
  --tokenizer /path/to/model-tokenizer --target-backend cuda \
  --dataset sharegpt --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-prepared-cache-dir /path/to/same-model-prepared-cache \
  --seed 42 --concurrency-sweep 8,16 \
  --num-prompts 64 --warmup-requests 10 --n-repeats 2 \
  --sharegpt-selection-pool-size 74 \
  --sharegpt-min-input-tokens 4 --sharegpt-max-input-tokens 1024 \
  --sharegpt-min-output-tokens 128 --sharegpt-max-output-tokens 256 \
  --sharegpt-max-total-tokens 2048 --sharegpt-chat-template-reserve-tokens 32 \
  --ignore-eos --temperature 0 --repetition-penalty 1 \
  --latency-slo ttft:2000,tpot:50,itl:100 \
  --latency-slo-fail-on-violation --fail-on-error --max-error-rate 0 \
  --timeout 120 --output json --out /path/to/new-report.json
```

模板支持时，两端相同地增加 `--enable-thinking false`。GPT-OSS 最终默认策略短测不添加该开关及 `--reasoning-effort`，增加 `--sharegpt-fixed-output-tokens 512`，删除 `--ignore-eos`，改为 `--num-prompts 32 --warmup-requests 2 --n-repeats 1 --sharegpt-selection-pool-size 34`；其服务端不添加 `--disable-thinking`。使用独立报告路径，并由 Rust 客户端按请求策略校验缓存。该 CLI 参数固定请求上限，实际输出必须等于 512 的断言仅在启用 `ignore_eos` 时执行。SLO 不通过时客户端先保存完整报告，再返回非零；不得因此丢弃较慢结果。原始 JSON、服务端命令/effective config、权重与选样身份、日志和 sampler 收据保存在仓库外的 `20261101-popular-model-comparison` 证据目录。

原始报告及采样记录保存在仓库外；以下 SHA256 用于标识本轮实际原件。本页提供复现配置与结果，不另行发布原始日志或权重。

| 原件 | SHA256 |
| --- | --- |
| `4b-ferrum-bench.json` | `6c59b6e64f0a0c1eef3e8d41cf58ec1fe2a70cca26048e66bcadcc362cfd25fe` |
| `4b-ferrum-resource.summary.json` | `4c5dce33d6f332f14643d7786f6b396cc018377cc595985f820b744cc8c5266e` |
| `4b-ferrum-evidence.tar.gz`（报告、effective config、日志及资源采样） | `92cbbbe7339cf01b3a17c374ca8d045bdcdffbce8f6cae977ae53957e8821b6f` |
| `4b-llama-unified-bench.json` | `cafec6b3474bf748466780846a0888165d831c38b3a9be05c93c9097a1328890` |
| `4b-llama-unified-resource.summary.json` | `a7f9b5f023ed75e5f6fc887373d790ae75210b9fd11a07cd171450865f79b458` |
| `4b-llama-unified-evidence.tar.gz`（主基线报告、日志及资源采样） | `178349fdf04105ff541537a0a0fe2e7b2d0ef1c7a824624feaaa45c97cc79c41` |
| `9b-ferrum-bench.json` | `a02a401b59044736c049bcd7b92eb66fd9d70fc66007cbebcbeb45ced8c475be` |
| `9b-ferrum-resource.summary.json` | `0872128eb2c73b592cc436e873ef21cd73187b3072ed1c1cbd75046d8ea2a55a` |
| `9b-ferrum-evidence.tar.gz`（报告、配置、资源及功能检查） | `8b82fd67b566e4006fa9c1d48cd3e9fd60a7e7d2011016d860e29ec8213ae6ce` |
| `9b-llama-unified-bench.json` | `aeaaacbfb07c460b9790d3f5c78cd389e51eb4dbd7d8b1a43661cfddb58e1216` |
| `9b-llama-unified-resource.summary.json` | `2b2050d0f5a813be97aed1a9e8619e77a6ecec7161b8f0147af126a2b57fa174` |
| `9b-llama-unified-evidence.tar.gz`（主基线报告、日志及资源采样） | `d1f107c52bfff9c75aeda156066f944f507dcecf9dbd145c0b479fc6baccc4b7` |
| `4b-f16-ferrum-bench.json` | `e482c46cf526782e769f1b90e4e023d639ae485dbe53b87ef9d4914a001a3992` |
| `4b-f16-ferrum-resource.summary.json` | `c14a071f5e99dfd16ac8d3714cc83c312ad0a605cb263ee3e14988d9c2a1afcf` |
| `4b-f16-ferrum-evidence.tar.gz`（报告、配置、资源及功能检查） | `af21c9488b072f2e9889f6561eb4feba575c95af1aa531a08295812f5affe9ea` |
| `4b-native-vllm-bench.json` | `8d0977a59fbb0739d8ecd87f9b27a14c60a24842d63d4fff80f2cd791e3bd4fe` |
| `4b-native-vllm-resource.summary.json` | `b93a68363459d1f31dec4982316e1c162896bcf1c33362a82af1104a2f0912d7` |
| `4b-native-vllm-evidence.tar.gz`（正式主基线报告、配置、资源及功能检查） | `c45d8406646fda6cfadcd51bf83d5a2dc705a87ce1cb8d560234f71f5b1f9b35` |
| `9b-f16-ferrum-bench.json` | `b5a29267ad3119a366c02f1346a5de12b62575105f01c473cabb9a6a687c3a93` |
| `9b-f16-ferrum-monitor.summary.json` | `4e6185b295153a7e174aaa402b94699316ba58987fc3d846e154441e678394a9` |
| `9b-f16-ferrum-evidence.tar.gz`（报告、配置、资源及功能检查） | `c24af9939bf2581f822ddd60ce1faf48d6ae2121102f6e24949202c17b712b02` |
| `9b-native-vllm-bench.json` | `2e8f987a51bdf52ef09e24fa89f2b7146612e7da4d4f2fd292ff0e8e5bfe55c9` |
| `9b-native-vllm-monitor.summary.json` | `d5f54a31f1a75521ac0cc470aeb94fffb068e3beaf08f1d52522bbf99aed7b78` |
| `9b-native-vllm-evidence.tar.gz`（报告、配置、资源及功能检查） | `92d854086f5ea44b0abe1fb8598ff5ca38f4e360967d21d2a7e34c1b9d9e9243` |
| `qwen3-30b-gptq-ferrum-bench.json` | `1d178440ce20401ffcf47ad192e28d3fa68232b78fa0bd1504bf5f94d1299733` |
| `qwen3-30b-gptq-ferrum-resource.summary.json` | `689b70577162cfa9643a0239bd0cf0ef64ae025673c592b62ad582b12d54083f` |
| `qwen3-30b-gptq-ferrum-evidence.tar.gz`（报告、配置、资源及功能检查） | `faf8e50334444db5ff527f7bc658785ea160e1209f4f409da3c889edbe7bedd9` |
| `qwen3-30b-gptq-vllm-bench.json` | `4d419202610b611243dd2bd3b2194ed652b9118f91f6f904bff59482dd742a03` |
| `qwen3-30b-gptq-vllm-resource.summary.json` | `f50f921df94cf62bc05e1304510a1049141add4c89af754be8bc019fce90860d` |
| `qwen3-30b-gptq-vllm-evidence.tar.gz`（报告、配置、资源及功能检查） | `9560230806638acfd29cc3999620940de964405406c9549d2c1b4260fc8ec5b7` |
| `qwen35-35b-gptq-ferrum-bench.json` | `883dffab4471b7347e6cdcef917b03946648fb2b2d3eb400cdcce4bc6164d2c5` |
| `qwen35-35b-gptq-ferrum-monitor.summary.json` | `08686c3df5223b59446cf4b3fd58522803cddcb51dfb5a0be1b9659edf90934e` |
| `qwen35-35b-gptq-ferrum-evidence.tar.gz`（报告、配置、资源及功能检查） | `a958e8c110be9bbaad730d7e6a5c7e82a0113bd111964cf2c307b47ab177c5a1` |
| `qwen35-35b-gptq-vllm-bench.json` | `086a5db4678f8e21abfb99f8bb63d74123dc6d2dc23e229dc0ead88e9ad5b544` |
| `qwen35-35b-gptq-vllm-monitor.summary.json` | `7f0e677666e17b0c5dbef5c8e9ca37690d5194368ccdd84d0688faf21d8e33f9` |
| `qwen35-35b-gptq-vllm-evidence.tar.gz`（报告、配置、资源及功能检查） | `39a35e6700dfad57943d762b0ee13e25a0995e7409e7dae25ab23c11778bfe8d` |
| `gpt-oss-20b-ferrum-fixed-failure-evidence.tar.gz`（中断记录、失败日志快照、配置及独立功能检查） | `55d261542160a061f188a36f234f2b5aa709e3a8ab9c6d57267a4fe64ad39071` |
| `gpt-oss-20b-native-eos-ferrum-evidence.tar.gz`（不完整的 low 自然 EOS 诊断、中断记录及日志快照） | `04415aa3a6ef258f1e897bdddb3e0cc49e72333bb42ea5fe4b7014e576078ab6` |
| `gpt-oss-20b-default-medium-ferrum-bench.json` | `4ccf52e8a8311e91373ef028948fb2683d8a1a12d83d0d5ff1e3d97332a8edcc` |
| `gpt-oss-20b-default-medium-ferrum-monitor.summary.json` | `ca71c429f44d28ab170911d0f7ac13d877a1e92da1470bbd278b58da34ad88ae` |
| `gpt-oss-20b-default-medium-ferrum-evidence.tar.gz`（完整短测报告、配置、资源及功能检查） | `1660ea98a399f06366f8e76ac385bdaf9fea60e0f69a223e8360c4626faa9cd3` |
| `gpt-oss-20b-default-medium-vllm-bench.json` | `739474333c8a43b0ba08513e6080ef61800455f8150e474ec7886fa4ed32c16b` |
| `gpt-oss-20b-default-medium-vllm-monitor.summary.json` | `a50f73a450619d1e0845a6b6517c3e71e315f5bebe0e02c954503a6964535ca6` |
| `gpt-oss-20b-default-medium-vllm-evidence.tar.gz`（完整短测报告、配置、资源及功能检查） | `a684d6f62c9e62aa405e353bc5385c3d651b97463b7d0beb3faf6e1d039538e7` |

</details>
