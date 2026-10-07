# 热门模型并发 P0（2026-10-07，进行中）

[目标](goal-popular-models-concurrency.zh.md)尚未完成。当前成功完成 **11/44** 格：Metal 27B llama.cpp C4/C8与Ferrum C4，以及CUDA 27B两引擎GGUF C4/C8/C16/C32全部8格，均零错误。CUDA guard退出0、原服务已恢复核验，模型/镜像下载已恢复；Metal Ferrum C8原尝试已自行结束，29成功/3超时，另记1格失败、不增加成功数。1800秒边界的六格队列已启动，结果待完成。全部仅一次探索，SLO unknown，未进入P1或完成验收。

推理产品冻结在main `0680840becae342e481df4ac0705c43fc08633a7`；客户端仅增加显式selection-pool以保证跨请求数的样本前缀。原工作区旧SLO修改保留。Metal对照仅llama.cpp，vLLM仅CUDA。旧3格因llama仅49/66层GPU及容量/样本口径不同退出新比较，完整证据保留。

## 固定输入与对照

| 输入 | 固定版本与状态 |
| --- | --- |
| Qwen3.8 GGUF | `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502`，`Qwen3.8-27B-UD-Q4_K_M.gguf`，16,464,440,224 B；本机与CUDA远端完整 SHA256 均已核对：`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Qwen3.6 GGUF | `unsloth/Qwen3.6-35B-A3B-GGUF@a483e9e6cbd595906af30beda3187c2663a1118c`，`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`，20,893,015,008 B；已下载并核对完整 SHA256 `a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c` |
| Qwen3.8 AWQ | `cyankiwi/Qwen3.8-27B-AWQ-INT4@6e134bae811fb5adac50ee042ae5f029ac6779aa`；实际为 compressed-tensors INT4/G32/非对称；5/5分片的399组量化头部布局已核对；CUDA主测后已恢复下载，完整权重尚待校验 |
| Qwen3.6 AWQ | `QuantTrio/Qwen3.6-35B-A3B-AWQ@119886a1072372348f73ef0df2d801cdcc0f455b`；AWQ/GEMM、4bit、group128、zero-point；已核对索引与2/9分片头部，完整覆盖第0/1层；CUDA主测后已恢复下载队列，尚未完整验证 |
| semantic | Qwen3.8 官方 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；Qwen3.6 官方 `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| llama.cpp | 固定 [b11429](https://github.com/ggml-org/llama.cpp/releases/tag/b11429)，commit `d81235049384534c167caea52b85a694f6103d14`；macOS arm64 包按发行摘要校验，新全GPU27B C4/C8完成；CUDA r3构建成功，27B GGUF四档探索均完成 |
| CUDA vLLM | 官方候选 `vllm/vllm-openai:v0.31.0-cu129`、`linux/amd64` 固定digest `sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`；镜像配置声明CUDA12.9.1/SM12.0目标，备用源已核对相同manifest SHA；CUDA主测后已恢复拉取，尚未部署或在5090验证 |

ShareGPT 延用 `anon8231489123/ShareGPT_Vicuna_unfiltered@745745adf6cd15b84e4f1c4a5a051fb4304f9342` 的 `ShareGPT_V3_unfiltered_cleaned_split.json`，本机已下载并核对完整 SHA256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。

为保留 #402 的原始样本与输出预算，选择器继续使用已核对 SHA256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` 的 tokenizer；服务端各用目标模型自己的 tokenizer 和模板。27B 官方 tokenizer 的 SHA256 是 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`，不能直接换入选择器并声称样本未变。

新口径：预热 8；探索正式请求 max(32, 4×C)、一次重复；Metal C4/C8/C16、CUDA C4/C8/C16/C32。验收 Metal 正式 max(64, 4×C)、CUDA 正式 200，均两次重复。各模型先短测并冻结 Metal 全 GPU 槽位，CUDA 固定32槽；同一冻结样本选择按所需数量取前缀，不因 C 改变抽样。seed/sampling-seed 42；输入 4–1024、输出至少 4、输入+参考输出+32-token 模板预留 ≤2048；参考输出长度、ignore_eos、thinking off、temperature 0、top_p 1、repetition penalty 1。旧三格使用 32 预热 + 64 测量，其完整样本选择一致，SHA均为旧冻结值 `03676b00ee0b9e8407f59e3fb531eeff99dde5ccc9e59844b6abf7d22f1d5df2`。新选择及容量须单独绑定，旧样本 SHA 不代表新口径。FP16 KV、prefix/session cache off。新模型 S 尚未按目标规则计算，不预填 SLO pass。

## 新口径探索结果（11/44）

11格均为Qwen3.8-27B同一UD-Q4_K_M、每请求2048、FP16 KV、pool208、8预热、一次重复、fresh HTTP；C4/C8各32正式，C16为64，C32为128，按同一冻结选择取前缀。Metal两引擎全GPU、固定16槽，CUDA固定32槽；llama均66/66层GPU、共享KV16384，Ferrum动态KV、Metal utilization1.0/CUDA0.9。这些总KV/预算策略差异保留，不跨硬件比较比值。

C4/C8、C16、C32各格正式usage输出分别9,704、16,699、36,430 token；预热8个均成功、各2,519 token。三种样本规模的输入均为5–964 token，均值依次116.6875/106.015625/104.4375；参考输出范围依次19–816/19–816/19–837，均值303.25/260.921875/284.609375。延迟单位ms，均为P50/P99。

| 后端 / 引擎 / 模型 / 客户端并发 | TTFT P50/P99 | TPOT P50/P99（last-visible） | ITL P50/P99（可见SSE文本事件） | 输出 tok/s | 设备采样峰值 | 采样footprint峰值（B） | 采样RSS峰值（B） | 成功/错误/拒绝；重复 | SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metal / llama.cpp / 27B / C4 | 1846.615 / 13422.359 | 346.014 / 436.661 | 337.620 / 433.329 | 10.524548 | 未采集 | 3,886,755,456 | 4,039,245,824 | 32/0/0；1 | unknown |
| Metal / Ferrum / 27B / C4 | 1846.992 / 19019.053 | 351.164 / 471.327 | 337.967 / 1773.182 | 10.221030 | 未采集 | 1,906,946,176 | 364,331,008 | 32/0/0；1 | unknown |
| Metal / llama.cpp / 27B / C8 | 2034.976 / 19729.125 | 576.920 / 734.440 | 552.246 / 1345.558 | 12.330679 | 未采集 | 3,908,071,168 | 4,037,050,368 | 32/0/0；1 | unknown |
| CUDA / llama.cpp / 27B / C4 | 114.191 / 372.604 | 24.727 / 27.188 | 23.737 / 53.060 | 147.284807 | 22,211 MiB（NVML全卡） | 未提供 | 未提供 | 32/0/0；1 | unknown |
| CUDA / Ferrum / 27B / C4 | 637.959 / 8413.991 | 97.366 / 162.883 | 92.385 / 198.642 | 33.591733 | 18,034 MiB（NVML全卡） | 未提供 | 未提供 | 32/0/0；1 | unknown |
| CUDA / llama.cpp / 27B / C8 | 126.935 / 585.610 | 29.005 / 32.475 | 26.310 / 92.286 | 225.204408 | 22,212 MiB（NVML全卡） | 未提供 | 未提供 | 32/0/0；1 | unknown |
| CUDA / Ferrum / 27B / C8 | 749.334 / 11959.664 | 132.401 / 207.660 | 121.207 / 450.456 | 44.399208 | 18,748 MiB（NVML全卡） | 未提供 | 未提供 | 32/0/0；1 | unknown |
| CUDA / llama.cpp / 27B / C16 | 139.638 / 1291.395 | 44.217 / 48.839 | 39.092 / 129.246 | 302.624067 | 22,210 MiB（NVML全卡） | 未提供 | 未提供 | 64/0/0；1 | unknown |
| CUDA / Ferrum / 27B / C16 | 940.480 / 20568.832 | 260.429 / 320.730 | 210.219 / 1006.224 | 50.371303 | 20,438 MiB（NVML全卡） | 未提供 | 未提供 | 64/0/0；1 | unknown |
| CUDA / llama.cpp / 27B / C32 | 147.768 / 2332.627 | 75.373 / 90.288 | 66.511 / 201.783 | 382.626783 | 22,223 MiB（NVML全卡） | 未提供 | 未提供 | 128/0/0；1 | unknown |
| CUDA / Ferrum / 27B / C32 | 1731.755 / 36427.941 | 498.273 / 730.451 | 373.448 / 1518.763 | 58.550886 | 23,809 MiB（NVML全卡） | 未提供 | 未提供 | 128/0/0；1 | unknown |

同硬件、同并发的Ferrum/llama输出吞吐比为：Metal C4 **0.971161**（本次低2.88%），CUDA C4 **0.228073**，CUDA C8 **0.197151**，C16 **0.166448**，C32 **0.153023**；本次均未超过llama。这是固定先llama后Ferrum的一次探索比较，无重复方差/CI，也不代表SLO达标。两引擎仍有上述动态KV/固定共享KV和内存预算策略差异；未完成并发档不外推比值。

C4/C8七份原生报告的40条完整选择（含8预热）已逐项核对一致，SHA均为 `6faca429d03d2f9718daf3fffa1a10382b88842c3432b41643b2596bcf0552c8`，原生元数据均记录pool208。Metal C4两份报告位于 `metal-explore-fullgpu-v3/serve-guard/qwen38-dense-{llama,ferrum}-explore-c4/client/report.jsonl`，文件SHA依次为 `69b6e911dd7ef83bbe9430da3a9075649989614c398a4ea82701dcc36d979093`、`b6fc308192514716b545d41df93634cc736b7e57a1120bd7e96501cc2fafc6de`。Metal llama C8报告为 `metal-explore-fullgpu-v3/serve-guard/qwen38-dense-llama-explore-c8/client/report.jsonl`，文件SHA为 `261b4dacecd9988a2e1eb2c3c910c97b4995d121dc59d0c231d9a2005cac49d3`。CUDA本机取证副本位于 `analysis/cuda-primary-results/qwen38-dense-{llama,ferrum}-explore-c{4,8}/client/report.jsonl`；按表中CUDA C4/C8行序，文件SHA为 `39bda2159a4dd643a3e9ecc20effd9026038f731a5db9f341ddedae79c9082b4`、`030414c10c42dbc61efbf9141dbeebc80ce805ebacb3a53aeb6901add3c98e49`、`b5fd0d0eaa59467cf57174af7a3d148b216511ea3c2179456128211386e5d3cb`、`de9a4a49d8e1ec9f9055ac3968dfc18986d959e1b91673c9876942bde2b250da`。11格cell均退出0，Metal另有client/server退出0原始记录。既有Rust汇总位于 `analysis/primary-summaries/` 的 `metal-dense-{llama,ferrum}-explore-c4-fullgpu-v3.jsonl`、`metal-dense-llama-explore-c8-fullgpu-v3.jsonl`、`cuda-dense-{llama,ferrum}-explore-c{4,8,16,32}.jsonl`。

CUDA C16/C32的原生副本同在 `analysis/cuda-primary-results/qwen38-dense-{llama,ferrum}-explore-c{16,32}/client/report.jsonl`。按llama/Ferrum顺序，C16报告SHA为 `bcaa5b971098a3ef9a3686e3d0cfda6d678f5ad712cb75a957f37cbc797ef5ea` / `ff6794d0f66584f2f0b82db1d5c1089203dba8ef127ce1117dd74212ac8d2470`；C32为 `c8dcc61fdc79e31dd28f83c25fd9ba42a8c682c2a5a3b508f508b7199e19955a` / `a8ce11d7ba91326a6d2320d2e1d193ce09b1d0942e96d2fb8b603dfcacc03031`。72条与136条选择（各含8预热）的SHA分别 `d39ec5b805599b92c2f1082bd474933b2ed223b89610af8db18e4efc0efa75d4`、`20c56f74aac0ca23927cf36376e71030f18e9d6bec969a5dc78a19961993e9bf`。同C跨引擎选择逐项一致，跨C前缀相同；原生元数据均为pool208。完成的C16原生报告可绑定新64题质量回放，尚未执行。

正式窗口按主表行序为922.0348755 / 949.41503775 / 786.980206959 / 65.885953952 / 288.880603834 / 43.089742774 / 218.562456841 / 55.18067409 / 331.51812379 / 95.210271681 / 622.193831604秒。

TPOT截止最后可见文本，按usage输出token减一归一；ITL保留全部可见SSE文本事件间隔。按表中行序，正式可见事件/间隔为9,684/9,652、9,680/9,648、9,684/9,652、9,683/9,651、9,680/9,648、9,678/9,646、9,680/9,648、16,674/16,610、16,665/16,601、36,344/36,216、36,364/36,236，观测到的transport coalescing均为0；严格单token诊断分别27/32、27/32、27/32、28/32、27/32、27/32、27/32、58/64、55/64、113/128、112/128，事件数与usage不一致的请求仍计入可见SSE ITL。预热与正式的全部协议质量计数均为0，这不等于语义质量或完整token一致率已验收。

Metal宿主内存来自整个client调用窗口（含预热）的4,620 / 4,791 / 3,935次采样，三格缺失及采集错误均为0；RSS与physical footprint重叠、不能相加，是采样下界，不能当作Metal设备分配峰值，也不是正式阶段或逐repeat峰值，不据较低Ferrum宿主值声称GPU内存更省。CUDA设备峰值来自各格既有Rust guard的 `nvml-summary.json`，250ms采样覆盖单个服务命令冷启动、加载、预热、正式执行至关闭，未按正式阶段裁剪。表中NVML值均为全卡已用内存采样峰，含驱动或其他分配，非该模型独占值或瞬时真峰；不能把差值直接解释为同等缓存容量下的模型内存节省。WSL逐进程NVML值不可用，宿主RSS/footprint未提供；各格两个采样器均正常退出0。

仅一次重复，无跨重复方差/CI；未声明SLO阈值，状态unknown。CUDA全部8格在原600秒单请求边界内完成。Metal Ferrum C8原尝试已自行完成并保存完整报告：正式29成功/3超时（预算742/781/816）/0拒绝，8预热全成功，窗口1183.778783125秒。guard/会话88606退出1（`client failed1`），旧guard42215/server55742/client56073均消失；root检查准备SIGTERM前任务已退出，kill返回no-such-process，实际未中断。归类为基准请求超时，不是provider不支持，也不计成功格。

该失败尝试的成功usage输出为7,365 token，成功usage吞吐6.221602 tok/s，不能作为完整公平吞吐或计算C8引擎比值；终端legacy约7.8包含不完整样本，不采用。29个成功请求的TTFT P50/P99为3397.588/29154.199 ms，last-visible TPOT为922.931/1154.916 ms；可见SSE ITL为849.546/3924.954 ms，保留全部32请求已观察的9,129个间隔，包括失败请求的停顿。协议记录3个missing DONE、3个malformed stream与3个bad output。原始报告及stderr在 `metal-explore-fullgpu-v3/serve-guard/qwen38-dense-ferrum-explore-c8/`，Rust汇总为 `analysis/primary-summaries/metal-dense-ferrum-explore-c8-fullgpu-v3-failed.jsonl`。

旧失败证据保留；root已启动独立1800秒请求边界六格队列，具体配置见下方生命周期记录。早期风险分析为 `metal-explore-fullgpu-v3/timeout-risk-review.json`。

## 历史诊断数据（不计入新矩阵或公平基线）

M1 Max/32GiB，27B同一UD-Q4_K_M文件，均固定32槽、每请求上下文2048、FP16 KV。llama.cpp为49层GPU加CPU、共享KV16384；Ferrum为全Metal、动态KV、`--gpu-memory-utilization 1.0`，不能忽略这些配置差异。下表各格一次重复，32预热与64测量全成功，错误/拒绝及全部协议质量计数均为0；无跨重复方差或CI。按行顺序测量窗口为3050.697375708 / 2108.819163458 / 1776.230751792秒，各19,801个usage输出token；选择器输入均为5–954 token，均值101.453125。延迟单位为ms，P50/P99；绝对SLO阈值未声明，状态为unknown。

| 引擎 / 模型 / 客户端并发 | TTFT P50/P99 | TPOT P50/P99（last-visible） | ITL P50/P99（可见SSE文本事件） | 输出 tok/s | Metal分配峰值 | 采样footprint峰值（B） | 采样RSS峰值（B） | 成功/错误/拒绝；重复 | SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama.cpp / 27B / C1 | 2168.744 / 15679.401 | 142.544 / 154.879 | 143.285 / 196.639 | 6.490647 | 未采集 | 8,740,024,000 | 11,430,707,200 | 64/0/0；1 | unknown |
| llama.cpp / 27B / C4 | 2820.175 / 15638.016 | 396.565 / 511.273 | 366.985 / 1690.317 | 9.389615 | 未采集 | 8,786,964,288 | 11,858,051,072 | 64/0/0；1 | unknown |
| Ferrum / 27B / C4 | 1767.372 / 16092.720 | 335.614 / 434.085 | 321.494 / 1512.212 | 11.147763 | 未采集 | 2,054,533,440 | 378,109,952 | 64/0/0；1 | unknown |

主表TPOT使用既有Rust reader的last-visible口径，不混入原报告的终止时刻TPOT；旧 `itl_ms=0` 也不作为主表ITL。按表中行序，可见文本事件为19,617 / 19,622 / 19,610，间隔为19,553 / 19,558 / 19,546，观测到的transport coalescing均为0；严格单token诊断分别51 / 52 / 50个请求符合，其余13 / 12 / 14个事件数与usage不一致，仍保留在可见SSE ITL中。

上述数据只保留历史取证。llama.cpp 部分层在 CPU、Ferrum 全 Metal，执行放置不同，吞吐及延迟比值不能用于公平性能结论。新比较须按全 GPU 固定槽位重测，绝对 SLO 仍 unknown。

内存为 `proc_pid_rusage.RUSAGE_INFO_V2` 在各自整个client调用窗口（含预热）对服务进程采样：按行序17,267 / 11,717 / 10,075次，缺失/采集错误均为0。RSS与physical footprint重叠、不能相加；它们是采样下界，不是独立测量阶段、逐repeat或Metal设备分配峰值。尤其不能把Ferrum的较低值解释为模型仅占这些内存或内存更省；原始采样、进程身份和ABI字段复核见 `analysis/metal-host-memory-accounting-review.json`。Ferrum C4的两份 `host-{midrun,late-measurement}-observation.json` 仅为全机中途快照，未覆盖全窗口，也无匹配llama窗口，不能据此归因换页或量化温度/降频。

Rust提取均在 `analysis/primary-summaries/`：`metal-dense-llama-c1-r4.jsonl`、`metal-dense-llama-c4.jsonl`、`metal-dense-ferrum-c4.jsonl`；各自 `source_report` 指向原生报告，相邻命令和退出状态保留取证，client/server/cell均退出0。协议计数通过不等于64题语义质量或完整token一致率已验收。

## 当前缺口与矩阵状态

下表是准备/支持状态，不是性能排名。CUDA 每模型四组×四档、Metal 每模型两组×三档，共44格；新口径11/44格，旧数据限制见历史表。

| 后端 / 模型 | Ferrum GGUF | llama.cpp GGUF | Ferrum AWQ / vLLM（仅 CUDA） |
| --- | --- | --- | --- |
| CUDA / 27B | GGUF C4/C8/C16/C32均完成，零错误；指标见主表 | 66/66层GPU；GGUF四档均完成，零错误 | CT的399组头部与实际 weight_shape 已核对；权重/镜像下载已恢复，AWQ/vLLM未实跑 |
| CUDA / 35B-A3B | provider 格式缺口，未实跑 | 待下载与运行 | AWQ及混合BF16专家支持缺口，未实跑；vLLM 待部署 |
| M1 Max / 27B | 新探索固定16槽，C4主负载32/32完成、零错误，10.221030 tok/s；C8原尝试29成功/3超时；1800秒重测已启动 | 16槽66/66层GPU；新C4/C8各32/32完成、零错误，10.524548 / 12.330679 tok/s | 不适用 |
| M1 Max / 35B-A3B | run（1槽）与serve（32槽）均实测失败：混合 MoE provider 不接受 Q6_K | 全GPU16槽短测OOM；8槽语义短测通过但并发OOM；7槽41/41层GPU，C16短测32/32成功；固定7槽已入主负载队列 | 不适用 |

- Ferrum CUDA 27B compressed-tensors AWQ：399组实际CT量化头部的dtype、布局和配置维度均匹配现有路径，未发现头部层面的阻塞；BF16 scales会转为F16，比较时须披露该精度差异。399 个 `weight_shape` 实际值已通过 6,384B 范围读取核对，零不匹配（`analysis/cuda-ct-dense-weight-shapes.json`）；该检查点尚未验证完整权重payload、分片hash或CUDA加载/计算，不能用GGUF短测代替。证据为 `metadata/dense-ct-headers/` 与 `analysis/cuda-ct-dense-gap.json`，只计元数据审计完成。
- Ferrum CUDA 35B：`qwen35_config.rs` 不接受原生AWQ。固定头部确认第0层768个专家张量均为BF16，第1层768组三元组为I32 qweight/qzeros与F16 scales；qweight为 `[K,N/8]`，不同于现有GPTQ的 `[K/8,N]`，router为F32、共享专家为BF16。现有CUDA routed/shared MoE仅GPTQ/FP8 Marlin，因此除AWQ零点与repack外，还需按层选择dense专家路径；GGUF亦未覆盖。证据为 `metadata/awq-moe-headers/` 与 `analysis/cuda-awq-moe-gap.json`；payload、完整hash及CUDA实机行为未验证，不计实机失败。
- Ferrum Metal 35B 实跑：context2048/FP16 KV/Auto，run（1槽）与serve（32槽）均退出1，在 `qwen3_5.f32-master (ProgramCompilation)` 阶段失败；`operation.routed_shared_swiglu_moe` 的唯一候选 `provider.metal.routed_shared_swiglu_moe.f16.q4k` 返回 `unsupported_quantization_formats`，具体格式为 `quantization.gguf.q6-k`。现有Rust `gguf_inventory` 已核对文件：`blk.{34,38,39}.ffn_down_exps.weight` 均为Q6_K，每个220,200,960B，其余117个专家栈为Q4_K。混合 MoE provider 的 routed gate/up/down 要求Q4_K，额外Q6_K down支持仅用于 routed-only Qwen3；本轮已确认是实际启动失败。两份原始错误分别在 `metal-moe-smoke/run-guard/qwen36-moe-natural-run/rust-fixture.stderr`、`metal-moe-smoke/serve-guard/qwen36-moe-short-smoke/rust-fixture.stderr`，相邻 `rust-fixture.command.json` 保留完整命令。尚无模型输出，不记吞吐0。
- Metal早期27B Ferrum短测：默认utilization0.9的一槽run通过，KV为正确两句话/44个完整ID（含终止ID248046）；默认32槽serve因21,359,593,088B计划超过20,615,498,410B预算拒绝。仅改文档化utilization1.0后，同32槽serve的算术42/usage3与KV usage44/`[DONE]`通过，guard退出0。证据为 `metal-dense-smoke-memory-retry/`、`metal-dense-serve-memory-default32/`、`metal-dense-serve-memory-util10032/`；初始 `metal-dense-smoke/` 失败另保留。启动预算/计划成本均不是内存实测峰值，不填入主表。
- Metal早期27B llama短测：32槽全GPU在warmup发生Metal OOM/首请求HTTP500（`metal-dense-llama-smoke/`）；单槽/共享KV2048两题通过（`metal-dense-llama-c1-smoke/`）。32槽auto fit选择49/66层GPU后两题通过（`metal-dense-llama-fit-smoke/`），但含CPU层，退出新公平比较。UD文件混合F32/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ3_S/IQ4_XS，不能视为纯Q4_K。
- 旧队列 `metal-primary-remaining-combined-v2/serve.guard.json` 在llama C8途中因用户改口径正常终止并清理；已完成旧C4及部分日志全部保留，不计产品失败或新矩阵完成。Rust guard继续独占生命周期和 `NO_PROXY/no_proxy`。
- KV差异：Ferrum动态KV/状态池按总运行时字节预算增长，2048是每请求长度，不是共享池大小；llama显式共享16384-token KV池。启动与短测不能证明所有槽同时容纳完整2048上下文。旧32槽审计见 `analysis/metal-kv-capacity-comparison.json`；本轮Metal固定16槽、CUDA32槽，实际结果和错误单独记录。
- Metal早期35B llama保持32槽/shared16384/per2048/FP16：auto fit全41层GPU在warmup OOM（`metal-moe-llama-fit-smoke/`）；改为16层GPU/fit off/no-repack后，算术42/usage3与KV两句话/usage39自然stop、有效 `[DONE]`，guard退出0（`metal-moe-llama-gpu16-smoke/`）。该CPU offload诊断不计主负载或公平全GPU基线。
- 新全GPU容量短测：llama.cpp显式 `--gpu-layers all --fit off`，共享KV16384/每请求2048/FP16、batch2048/ubatch512固定。27B16槽32请求/C16随机128输入+32输出全部成功；35B8槽在第8槽加入prefill后Metal OOM，32请求失败（不是仅启动失败），随后只减槽位，4/6/7槽均32/32成功，8槽失败与7槽通过形成相邻边界。新探索采用27B16槽、35B7槽，客户端仍扫C4/8/16，C高于槽位时排队。短测不计主矩阵、不证明所有长上下文组合；证据分别 `metal-fullgpu-capacity-saturated-r1/` 与 `metal-moe-fullgpu-slots{4,6,7}-saturation-r1/`。
- 新采样客户端：固定pool208（8预热+最多200正式）、seed42、官方35B tokenizer及原长度过滤，每格仅取所需前缀。不同请求数的完整selection SHA应不同，但实际样本前缀须相同；同格跨引擎SHA完全一致。原选择器会随请求总数重新reservoir抽样，因此不能只改数量并声称跨C同样本。改动限Rust基准参数/选择器/报告元数据，源码commit `59244b3380abe474389c3ad1deb659d53221c430`；本机13项最小测试与workspace fmt通过，远端CPU客户端同13项通过，两份release均已构建。Metal客户端固定为 `bin/ferrum-bench-client-selection-pool`，SHA256 `28245cdbbc3a2dd51ec25ddd48729660f4466c3f303822126e4c62a096090a6c`；构建、原生help和实际Cargo产物回执为 `analysis/client-selection-pool-local/client-artifact-receipt.json`。推理二进制单独冻结，Metal SHA `d1797e90dd419b50d802a2c14592d899910a1d4923b1240371a69d71230d1356`，CUDA服务器release SHA为 `efad1116ee4fac2f5a3b9302315cf3fffa00fc28d4b113d6156e46d0eed58c86`。尚未执行本次完整workspace check/test/clippy，不据此声明PR已验证。
- Metal v3队列 `metal-explore-fullgpu-v3/serve.guard.json` 在3格成功后因Ferrum C8的3条超时自行退出1，进程已清理。root已启动 `metal-explore-fullgpu-v4-timeout1800/serve.guard.json`，SHA `554832a8472b1b79b9eba8debd08c94a64a66f62607c6e8468acf378452a05cb`，会话49978；顺序为27B Ferrum C8重试、llama/Ferrum C16、35B llama C4/C8/C16，共6格。仅deadline和新输出目录改变，模型/样本/服务参数保持冻结；原失败与已完成格保留。测量期间不运行本机构建/测试。
- CUDA引擎与控制器已构建：远端任务根 `/home/seekee/ferrum-slo-20260923/popular-models-20261007` 下，客户端SHA256 `114ee119d853b8f585ea049021aa9f62eb397c82a4b7ab249afc718b2c00db49`；guard SHA256 `8ff31e5df75b7721a51060144b528012d45481191e63acfc5b8afc32245b27c8`，fmt/test/release退出0、66项测试通过。llama r3构建退出0，server SHA256 `a290792b8b47609741f46a260fd683a93d2b4459f3e82f4cb4f56cc315ddf8cc`、CLI SHA256 `e76209300148f6c20ea58f013b50337816c4cf3db74bdf43d37238c2489420b9`。早期构建回执中的 `not_runtime_validated` 仅代表当时快照；后续原生进程短测与恢复已实际执行，Docker/vLLM模型生命周期仍待验证。
- CUDA 27B GGUF两引擎run/serve短测均通过；serve固定32槽/context2048/FP16，llama66/66层GPU、Ferrumutilization0.9；算术42/usage3与KV两句话/usage44、流式 `[DONE]` 均正确。llama CLI r1因过时 `--conversation` 参数失败，移除此参数的独立r2通过；所有短测均完成原服务恢复。证据为 `analysis/cuda-preparation/receipts/cuda-dense-runtime-and-launch-r1.json` 及其远端receipt路径；不计主矩阵或完整上下文容量验收。
- CUDA dense GGUF探索8/8完成：配置 `guard/configs/cuda-dense-gguf-primary-alternating.guard.bound.json` SHA `33dd572e23cc94df461f1eabc21e8fc3a69f78f1cb1090e60d4c6d432709df1e`；原guard PID1346180退出0，结束于2026-10-07T02:17:23Z。按llama/Ferrum C4、C8、C16、C32交替，固定32槽、pool208、8预热、32/32/64/128正式、重复1、单请求600秒、每格6000秒。CUDA测量期间准备进程全部暂停；其后恢复下载r5 primary PID1413198/quant PID1413199及镜像dockerproxy-r7 PID1413200，输入pins不变。完成与恢复原始证据在 `analysis/cuda-primary-completion-r1/`，启动/暂停历史分别见 `analysis/cuda-preparation/receipts/cuda-dense-runtime-and-launch-r1.json` 与远端 `downloads/pauses/dense-gguf-explore-r1/intent.json`。
- 主机为M1 Max/32GiB/macOS15.1.1与RTX5090/32607MiB，CUDA仅用指定helper经Mini→WSL。CUDA主队列已 `restored_verified`，原waifugemma4配置身份不变，恢复PID1410227、health `ok`；root同时报告18502MiB。旧PID4162656仅为准备快照；不据静态PID判断后续状态。私钥仅在Mini。
- 测量与质量工具：既有Rust `analysis/report-reader/` 通过2项测试并读取全部11格；TPOT取last-visible，拒绝缺字段旧schema。`analysis/quality-replay/` collector通过7项、compare通过8项硬件无关测试及各自release；真实64题回放/一致率尚未执行，Linux直接Cargo构建材料已备于 `analysis/cuda-preparation/quality-linux-r1/`。测试和编译不等于语义质量验收。

Metal失败格及后续主负载已按声明的新时限启动；CUDA当前GGUF队列已结束，其他权重/镜像准备恢复。P0仍缺完整矩阵、真实64题质量/一致率、两后端对照profile及按耗时排序的缺口表。Qwen3.5-9B #402 CUDA C4/C8回归线仍为284.688/414.436 tok/s。Metal vLLM已取消，不再下载/测试。

独立工作区：`/private/tmp/ferrum-popular-models-20261007`；全部过程、配置、元数据、下载和日志在仓库外 `/private/tmp/ferrum-popular-models-p0-20261007/`。恢复工作须核验进程/会话的当前状态，不能仅凭这些路径判断任务仍在运行。
