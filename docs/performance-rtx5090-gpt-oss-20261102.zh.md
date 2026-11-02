# RTX 5090 GPT-OSS 修复验证（2026-11-02，UTC+8）

[English](performance-rtx5090-gpt-oss-20261102.md) | 简体中文

本文记录 **0.18.4 发布前的优化候选 R11** `e36e1cc7e`。同一候选在两个全新进程中分别完成 C8→C16，对照同硬件、同源 MXFP4 权重的 Ferrum 0.18.3 和已保存的 vLLM 基线。

**两个全新进程均达到两档 ≥90% vLLM 的目标，并通过原定延迟 SLO。** C8 两次为 890.93/892.30 tok/s，达到 vLLM 的 108.68%/108.85%；C16 为 1383.53/1385.86 tok/s，达到 96.77%/96.93%。四个测量单元合计128/128成功、零错误和拒绝；这是本负载的两次独立启动复测，不代表长期稳定性或所有模型的性能。

## 正式结果

预先固定 P99 SLO：TTFT ≤2000ms、TPOT ≤50ms、非空 SSE 文本事件 ITL ≤100ms，正式请求成功率100%。吞吐门槛为 vLLM 的90%：C8 **737.79**、C16 **1286.78 tok/s**。每个进程每档32次正式请求、2次客户端预热、1轮；容量和配置在 C8/C16 间不变。表内延迟为 **P50/P99，单位 ms**。

| C | 引擎 / 全新进程 | 成功/请求 | 错误/拒绝 | 输出 tok/s | TTFT | TPOT | 可见 SSE ITL | 延迟 SLO / 吞吐门槛 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | Ferrum 0.18.3 | 32/32 | 0/0 | 72.42 | 54648.74/56304.24 | 不可作流式证据 | 缺失 | 失败 / 失败 |
| 8 | R11，第1个进程 | 32/32 | 0/0 | **890.93** | 155.70/227.27 | 8.17/8.62 | 8.00/14.38 | 通过 / 通过 |
| 8 | R11，第2个进程 | 32/32 | 0/0 | **892.30** | 157.81/226.31 | 8.21/8.61 | 7.95/13.02 | 通过 / 通过 |
| 8 | vLLM 0.31.0 | 32/32 | 0/0 | 819.77 | 94.75/387.13 | 8.71/8.96 | 8.16/33.00 | 通过 / 基线 |
| 16 | Ferrum 0.18.3 | 32/32 | 0/0 | 74.30 | 100934.84/109512.78 | 不可作流式证据 | 缺失 | 失败 / 失败 |
| 16 | R11，第1个进程 | 32/32 | 0/0 | **1383.53** | 346.93/462.27 | 10.00/10.29 | 9.88/11.73 | 通过 / 通过 |
| 16 | R11，第2个进程 | 32/32 | 0/0 | **1385.86** | 340.35/444.62 | 10.10/10.39 | 9.89/12.27 | 通过 / 通过 |
| 16 | vLLM 0.31.0 | 32/32 | 0/0 | 1429.75 | 174.08/183.29 | 10.14/10.38 | 9.47/34.24 | 通过 / 基线 |

旧版每请求仅交付一个文本事件，原报告 last-visible TPOT 为0、ITL缺失，不能解释为零延迟。R11四个单元均有全部32个请求的连续文本时序，无缺失 usage 或检出的传输合并。严格单 token ITL 仍不具备资格，因为文本事件数与 usage tokens 不相等。R11的C16首段延迟仍高于vLLM，未宣称所有延迟指标领先。

GPT-OSS 使用自然 EOS、默认 `medium`、每请求最多512 tokens。TTFT包含首段非空 reasoning 或 final 文本；TPOT使用首末可见事件和总 completion tokens，吞吐按成功请求的总 usage 输出 tokens / 正式测量时间计算，均不是 final-only 指标。两次独立启动仍是有限样本，未对 vLLM 重测或平均分位数。

## 修复范围

Harmony 增量解析提前发布已确定的 reasoning/content，保留跨分片控制符；工具参数和 JSON 等硬约束继续完整验证。状态引用已计费原始文本的偏移，不逐 token 重扫或复制全部历史。Chat、Responses 与 CLI 文本 REPL 的共享路径均已验证。

CUDA attention 改进上下文并行和页寻址，保留窗口、sink、YaRN及独立 KV pages 语义。Host通过 typed provider能力避免重复构建不支持的静态描述，每 wave 的 KV、请求权限和资源边界检查仍保留。锁定的 FlashInfer XQA decode直接读现有 pages，sink转换、workspace、semaphore均计入预算，single-split不再做无用reset。

XQA当前仅接入 GPT-OSS operation 的 **SM120/F16、head dimension64、64 query heads /8 KV heads decode**；其他硬件或缺少锁定算子时使用原有路径，未证明所有模型或 GPU 受益。

默认 `Auto` 现在按已准入容量和预算在 ready 前执行 warm/capture/replay，同时保持 bounded on-demand 学习和驱逐。显式 `OnDemand` 仍懒准备，`Startup` 仍封闭目录。合成请求释放后新产品请求可复用图，当前采样 mask和 terminal readback仍逐 wave绑定；run/serve共用构建屏障，无需新增用户调参。

运行期的可选构图按近期成功执行可节省的Host开销与实测构图成本准入。收益证据不足时继续正常推理，暂缓构图，后续仍可学习；不按模型名、并发档或固定第几次请求判断。启动声明的图不受经济门禁限制，显式旧模式保留原行为。未知形状的成本估计仍不能保证绝无新的尖峰。

## 启动成本与资源

两次启动均准备**34个程序/图**：decode宽度1–32、单请求prefill 1/256 tokens。目录容量3400，启动拒绝/容量延期均0；启动结束 active/pending/executing/maintenance均0，正式计数重置。正式负载前后发布的驻留目录均为34，lane epoch保持1，后续有界学习仍开放。

| R11进程 | spawn→首次成功 HTTP health | executor准备报告 | 启动采样窗 | 启动整卡显存峰值 | 启动进程RSS峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第1个 | 73.746s | 3.065s | 73.756s | 15.060GiB | 13.917GiB |
| 第2个 | 74.743s | 4.266s | 74.764s | 15.066GiB | 13.917GiB |

executor准备是启动内部区间，不与完整启动相加。首次health的executor preparation已ready，`text_ready` 在算术功能请求后为true。vLLM历史约51.25s的engine init与spawn→HTTP health边界不同，不作启动速度比。

以下为**整个C8+C16 sweep**的独立采样峰值，包含客户端准备和预热，不是各档峰值；间隔配置250ms。

| 引擎 | 采样窗 | 整卡显存峰值 | 窗口RSS峰值 | 生命周期RSS高水位 | GPU/Host采样点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ferrum 0.18.3 | 452.509s | 15.990GiB | 1.389GiB | 13.482GiB | 1804/1812 |
| R11，第1个进程 | 37.258s | 15.978GiB | 1.177GiB | 13.915GiB | 132/151 |
| R11，第2个进程 | 36.764s | 16.165GiB | 1.192GiB | 13.914GiB | 131/149 |
| vLLM | 37.512s | 29.937GiB | 3.145GiB | 5.925GiB | 147/152 |

采样错误均0；R11正式窗最大GPU/Host间隔为2016/251ms、1488/251ms，瞬时峰值可能遗漏。整卡包含其他驻留进程和驱动，WSL无可靠目标进程显存；Ferrum RSS为单服务进程，vLLM为EngineCore worker，不等于全进程树。生命周期高水位包含加载，不是窗口峰值。

vLLM按0.9利用率预留KV，池容量275324 tokens，与Ferrum按需占用不同，不能据此宣称显存效率倍数。两个R11进程各69/69成功（64正式、4预热、1算术预检），结束active/queued均0，随后只清理本轮服务。

<details>
<summary>构图阻塞的定位与未采用结果</summary>

R9两次全新进程的C8为889.90/899.10、C16为1307.43/1250.71 tok/s；第二次C16只有vLLM的87.48%，未达到目标。首16个请求输出都同为512 tokens，TTFT却从约298.56增至766.42ms，不能归因于自然EOS长度变化。R9结果保留，不作为最终主表。

同负载basic分段诊断中，C16首次与同进程热态的prefill Host encode为538.679/107.130ms，其中validate_and_prepare为449.824/0.209ms，而设备事件总时长为626.436/630.879ms。此诊断没有正式前的算术probe，仅用于定位，不替换主表。

R10仅增加分段观测，同一个单序列142-token prefill调用的Host encode为515.826ms，其中 `cuGraphInstantiateWithFlags` 占512.906ms；图录制、拓扑归因、上传分别仅0.591、0.077、0.820ms，该调用没有进入输入包分配同步。这确认了请求路径上的构图阻塞，尚不解释CUDA驱动内部为什么需要这段时间。另有独立资源准备峰值247.037ms，其中submission wave准备215.521ms，不能混为同一原因；各层包含关系也不能相加。

最终R11的两个正式窗口中，142-token单序列形状均有两次目录miss，发布目录均保持34张驻留图、lane epoch为1。正式 `profile=off` 不采集attempt记录及 `AmortizationDeferred` 细计数，字段0不能解释为没有发生；不据此声称该形状的具体延期次数或构图子段耗时。成本控制的状态转换另由CPU及真实CUDA测试验证。

</details>

<details>
<summary>输出长度与有效性边界</summary>

旧版、R11两进程和vLLM的完整选样、逐条prompt hash、输出预算、原始及服务端输入token向量均一致；每档输入总量4440/6584 tokens。自然EOS实际输出不强求一致。

| 引擎 | C | 输出tokens | 单请求范围 | 命中512上限 |
| --- | ---: | ---: | --- | ---: |
| Ferrum 0.18.3 | 8 | 15176 | 190–512 | 27/32 |
| R11，第1个进程 | 8 | 15223 | 190–512 | 27/32 |
| R11，第2个进程 | 8 | 15126 | 190–512 | 26/32 |
| vLLM | 8 | 15064 | 188–512 | 27/32 |
| Ferrum 0.18.3 | 16 | 14999 | 194–512 | 26/32 |
| R11，第1个进程 | 16 | 15062 | 190–512 | 27/32 |
| R11，第2个进程 | 16 | 15148 | 190–512 | 27/32 |
| vLLM | 16 | 15115 | 117–512 | 26/32 |

R11两次C8有4/32条输出长度变化，总计减少97 tokens；C16有3/32条变化，总计增加86 tokens。不宣称端到端逐位等价或同答案加速倍数。

多数请求触及上限，报告没有逐请求完整final答案覆盖；协议成功和单独功能题正确不能扩展成全部答案完整正确。原 `ignore_eos=true` 协议失败保留在[历史报告](performance-rtx5090-popular-models-20261101.zh.md)，本次自然EOS通过不覆盖该问题。

</details>

<details>
<summary>验证与中间结果</summary>

R11四项真实CUDA测试分别通过：成本不足时暂缓运行期构图且保留启动图；Auto准备后释放合成owner、新请求使用当前decode mask及新形状正常执行；显式hybrid仍按原warm/capture/replay学习；旧sealed保留精确描述符。早期新fixture错误要求prefill返回GreedyToken，仅纠正为既有FullLogits契约；decode mask/replay断言保持严格，原失败保留。

同一R11 binary的两次serve算术预检返回final `4`、HTTP200、DONE/usage完整；独立run Text REPL输出 `4` 并正常退出。早期长回答检查产生72个content事件，无Harmony控制符泄露；CPU测试覆盖Chat/Responses提前输出、终态不重复和错误不伪装正常完成。

R11源码通过workspace fmt/check/test/Clippy（`-A warnings`）、Metal编译；tests为6873 passed、0 failed、79 ignored。另通过规定的CUDA CLI feature编译检查。忽略项、feature-off零测试及编译均不冒充运行验证；Metal仅编译。

XQA真实5090数值/图重放/工作区边界通过，独立F64 reference容差为 `1e-3 × max(|reference|, 1)`。原scalar/packed F16用PartialEq逐元素比较，不区分±0，不称严格bitwise；prepare/YaRN/KV仅对旧实现，不称独立CPU YaRN oracle。

中间正式C8/C16 tok/s：R5 573.32/859.04、R6 865.50/1198.10、R7 862.49/1190.36、R8 794.78/1176.36。未择优替代最新候选。R8同进程热态C16曾达1389.03，但仅为诊断，未排除未采集环境因素，不计默认产品验收。

</details>

<details>
<summary>版本、负载身份与复现命令</summary>

硬件为同一张 RTX 5090（驱动报告 32607MiB，610.88），Intel Core Ultra 9 285K，Linux/WSL2。测试期间原驻留服务保持不变，各被测引擎依次运行。

| 项目 | 固定身份 |
| --- | --- |
| R11 候选服务端 | `e36e1cc7e8b095faaaf72c66e85e5b6d9448bf8b`；未发布 |
| 候选 CLI SHA256 | `8c09ebc1ffddd0869e45c0092065dddbd630fe1abc3135fd84a2950c0d1264a8` |
| 旧 Ferrum 服务端及三端测试客户端 | 正式 0.18.3，`e991c6955c3022f80d7e96f9d16d9306314054cf` |
| 正式客户端 SHA256 | `76094c69c77d0a1e6d675900c0a91b28bdee3d4237c930193abdac709ebee087` |
| vLLM | 0.31.0，`db9527a46873454610df6dbedf79a36d6bf1a7f6`，复用历史基线 |
| 模型 | `openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee`，同一份官方 MXFP4 权重 |
| tokenizer SHA256 | `0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3` |
| Harmony 模板 SHA256 | `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146` |
| ShareGPT 文件 SHA256 | `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| 本轮选样 SHA256 | `ceec98e3917466c7dc7347fc0522b2ef1626dd35651747d1018b65b05c562616` |

完整权重逐片身份和 vLLM 缓存镜像身份见历史报告。该缓存镜像 ID 不等于可公开拉取的 registry digest，复现时应核对实际版本与源码 revision。

候选和旧 Ferrum 自动选中 `gpt_oss.f16-mxfp4-bf16`：外层 F16、MXFP4 专家路径 BF16、KV F16。vLLM 为原生 BF16 激活/KV，使用 MARLIN MXFP4 MoE 和 FlashInfer attention。同源权重不意味着内部算术相同。本轮 Harmony 默认均为 `medium`，系统日期同为 Asia/Shanghai 的 2026-11-02。

输入筛选为 4–1024 tokens，参考回答为 128–256 tokens；输入 + 请求上限 + 32 模板预留 ≤ 2048，过滤而不截断。seed 42，选样池 34。实际服务端模板后的输入向量另行逐条匹配。正式测量 profile 为 `off`，服务端固定 32 序列、2048 context、2048 batched tokens，关闭前缀缓存。

启动准备采用各自产品默认：vLLM 在接受请求前自动完成模型预热及 CUDA graph capture；R11 默认在 ready 前准备声明的图，并按实测成本继续 bounded on-demand 学习。客户端仍仅每档 2 次预热，未另加 benchmark 专用预热来隐藏首次形状的成本。

候选服务端：

```sh
/path/to/candidate-ferrum serve /path/to/gpt-oss-20b \
  --semantic-source /path/to/gpt-oss-20b \
  --tokenizer-source /path/to/gpt-oss-20b \
  --backend cuda --max-model-len 2048 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 --kv-dtype fp16 \
  --gpu-memory-utilization 0.9 --disable-prefix-cache \
  --host 127.0.0.1 --port 19081 --served-model-name gpt-oss-20b \
  --effective-config-json /path/to/effective.json
```

使用未修改的正式 0.18.3 Rust 客户端：

```sh
/path/to/ferrum-0.18.3 bench-serve \
  --base-url http://127.0.0.1:19081 --model gpt-oss-20b \
  --tokenizer /path/to/gpt-oss-20b --target-backend cuda \
  --dataset sharegpt --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-prepared-cache-dir /path/to/gpt-oss-prepared-cache \
  --seed 42 --concurrency-sweep 8,16 \
  --num-prompts 32 --warmup-requests 2 --n-repeats 1 \
  --sharegpt-selection-pool-size 34 \
  --sharegpt-min-input-tokens 4 --sharegpt-max-input-tokens 1024 \
  --sharegpt-min-output-tokens 128 --sharegpt-max-output-tokens 256 \
  --sharegpt-max-total-tokens 2048 --sharegpt-chat-template-reserve-tokens 32 \
  --sharegpt-fixed-output-tokens 512 --temperature 0 --repetition-penalty 1 \
  --latency-slo ttft:2000,tpot:50,itl:100 \
  --latency-slo-fail-on-violation --fail-on-error --max-error-rate 0 \
  --timeout 120 --output json --out /path/to/report.json
```

这里 `--sharegpt-fixed-output-tokens 512` 固定请求预算；未启用 `--ignore-eos`，允许正常 EOS 提前结束。服务端和客户端都不覆盖默认 thinking/effort。vLLM 的实际复现命令保留在历史报告，使用 `--reasoning-parser openai_gptoss`、相同容量/缓存设置与时区。

</details>

<details>
<summary>原始证据</summary>

JSON、配置、命令、日志和资源采样保存在仓库外 `20261102-gpt-oss`；历史原件在 `20261101-popular-model-comparison`。R11两份正式归档及其51/39个成员已逐项核验SHA256。

| 原件 | SHA256 |
| --- | --- |
| R11第1个全新进程完整包 | `ae6d62c84d31dda4719d89ce00ee30977329e0ac5ac49677616a46db28a95b2c` |
| R11第2个全新进程完整包 | `ab2a59c4beb12b3c4529cd729d603c27b47bd7db40cc8d935c065073590af3c4` |
| R11真实CUDA验证包（21个成员） | `adfd853ccac69f7c7bcc13df9963f813266d7361203655f84ba5ca8419d31bc1` |
| R11独立run验证包 | `f9f19093b3e79b695ce7e10a147eae24e379bd56d877f94607788a45aaee149b` |
| R9第1/2个全新进程完整包 | `2b8489199700b12b0bea645c6440bb297f875e25831addd415bb523ffa1ec1fc` / `5d1f3cd4b24fd388970fb7ab01b0c482f13790d21b38c096ce029f79f9693944` |
| R9 basic分段诊断包（32个成员已核验） | `af2c9f6805014d11a661bcafa731246b764a7cc077da7f7e71b38c01c8e27921` |
| R10构图分段诊断包（43个成员已核验） | `494fa8306c07f5e8025dc604bb5def6c586893c8bd95355856a9d23f1cd6750a` |
| R11编译源码清单 | `18b1da7dec18f8cc348ce44287972c8d5fb41af6a435ca502ef9185df18be7c7` |

</details>

## 已测 MoE 范围

以下为[Ferrum 0.18.3历史结果](performance-rtx5090-popular-models-20261101.zh.md)；本次只重测GPT-OSS。

| 模型 | C8 tok/s（F/V） | C16 tok/s（F/V） | 历史SLO |
| --- | ---: | ---: | --- |
| Qwen3-30B-A3B GPTQ INT4 | 630.5/824.0 | 848.8/1263.7 | 两端两档两轮均通过 |
| Qwen3.5-35B-A3B GPTQ INT4 | 717.6/625.6 | 978.3/947.4 | Ferrum全过；vLLM C16第1轮ITL超限 |
| GPT-OSS-20B MXFP4 | 72.4/819.8 | 74.3/1429.8 | Ferrum两档失败且缺ITL；vLLM通过 |

Qwen每档两轮共128请求，GPT-OSS历史每档一轮32请求。模型、量化、注意力和数值路线不同，不跨模型排名；35B C16两轮领先方向不同，合并约3.3%不代表稳定优势。
