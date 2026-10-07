# 热门模型并发 P0（2026-10-07，进行中）

[目标](goal-popular-models-concurrency.zh.md)尚未完成。当前以 `main` 的 `0680840becae342e481df4ac0705c43fc08633a7` 采集基线，用户已调整矩阵：Metal 12 格、CUDA 32 格，共 44 格，目前新口径 **3/44**。Metal 27B llama.cpp/Ferrum C4和CUDA 27B llama.cpp C4均完成，各32个正式请求成功、零错误；当前Metal继续llama C8、CUDA继续Ferrum C4。旧 3 格退出新比较：llama.cpp C1/C4 实际仅 49/66 层在 GPU，不能作公平基线；Ferrum C4 也须按新容量与请求数重测。旧队列已因口径调整终止，进程清理完成；不是产品失败。推理服务与内核仍冻结在 main，尚未进入 P1。为跨请求数复用同一批样本，仅基准客户端增加显式 `--sharegpt-selection-pool-size`；Metal/CPU 两种构建的13项 ShareGPT 测试均通过，release 客户端均已固定。原工作区的旧 SLO 修改保留。按用户最新决定，Metal 仅比较 llama.cpp，vLLM 仅在 CUDA 测试。

## 固定输入与对照

| 输入 | 固定版本与状态 |
| --- | --- |
| Qwen3.8 GGUF | `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502`，`Qwen3.8-27B-UD-Q4_K_M.gguf`，16,464,440,224 B；本机与CUDA远端完整 SHA256 均已核对：`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Qwen3.6 GGUF | `unsloth/Qwen3.6-35B-A3B-GGUF@a483e9e6cbd595906af30beda3187c2663a1118c`，`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`，20,893,015,008 B；已下载并核对完整 SHA256 `a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c` |
| Qwen3.8 AWQ | `cyankiwi/Qwen3.8-27B-AWQ-INT4@6e134bae811fb5adac50ee042ae5f029ac6779aa`；实际为 compressed-tensors INT4/G32/非对称；5/5分片的399组量化头部布局已核对；为CUDA主测暂停下载，完整权重尚待校验 |
| Qwen3.6 AWQ | `QuantTrio/Qwen3.6-35B-A3B-AWQ@119886a1072372348f73ef0df2d801cdcc0f455b`；AWQ/GEMM、4bit、group128、zero-point；已核对索引与2/9分片头部，完整覆盖第0/1层；下载队列随主测暂停，尚未完整验证 |
| semantic | Qwen3.8 官方 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；Qwen3.6 官方 `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| llama.cpp | 固定 [b11429](https://github.com/ggml-org/llama.cpp/releases/tag/b11429)，commit `d81235049384534c167caea52b85a694f6103d14`；macOS arm64 包按发行摘要校验，新全GPU27B C4完成32正式请求；CUDA r3构建成功，27B run/serve短测通过 |
| CUDA vLLM | 官方候选 `vllm/vllm-openai:v0.31.0-cu129`、`linux/amd64` 固定digest `sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`；镜像配置声明CUDA12.9.1/SM12.0目标，备用源已核对相同manifest SHA；为CUDA主测主动暂停拉取，尚未部署或在5090验证 |

ShareGPT 延用 `anon8231489123/ShareGPT_Vicuna_unfiltered@745745adf6cd15b84e4f1c4a5a051fb4304f9342` 的 `ShareGPT_V3_unfiltered_cleaned_split.json`，本机已下载并核对完整 SHA256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。

为保留 #402 的原始样本与输出预算，选择器继续使用已核对 SHA256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` 的 tokenizer；服务端各用目标模型自己的 tokenizer 和模板。27B 官方 tokenizer 的 SHA256 是 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`，不能直接换入选择器并声称样本未变。

新口径：预热 8；探索正式请求 max(32, 4×C)、一次重复；Metal C4/C8/C16、CUDA C4/C8/C16/C32。验收 Metal 正式 max(64, 4×C)、CUDA 正式 200，均两次重复。各模型先短测并冻结 Metal 全 GPU 槽位，CUDA 固定32槽；同一冻结样本选择按所需数量取前缀，不因 C 改变抽样。seed/sampling-seed 42；输入 4–1024、输出至少 4、输入+参考输出+32-token 模板预留 ≤2048；参考输出长度、ignore_eos、thinking off、temperature 0、top_p 1、repetition penalty 1。旧三格使用 32 预热 + 64 测量，其完整样本选择一致，SHA均为旧冻结值 `03676b00ee0b9e8407f59e3fb531eeff99dde5ccc9e59844b6abf7d22f1d5df2`。新选择及容量须单独绑定，旧样本 SHA 不代表新口径。FP16 KV、prefix/session cache off。新模型 S 尚未按目标规则计算，不预填 SLO pass。

## 新口径探索结果（3/44）

三格均为Qwen3.8-27B同一UD-Q4_K_M、每请求2048、FP16 KV，客户端固定pool208、C4、8预热+32正式、一次重复、fresh HTTP。Metal两引擎均全GPU、固定16槽；llama.cpp b11429为66/66层GPU、共享KV16384，Ferrum为动态KV、utilization1.0。CUDA llama同为66/66层GPU、共享KV16384，固定32槽；不跨硬件计算引擎比值。按表中行序，正式窗口922.0348755 / 949.41503775 / 65.885953952秒，各9,704个usage输出token；预热8个也全成功，预算各2,519输出token。正式选择器输入5–964 token、均值116.6875；参考输出19–816 token、均值303.25。下表延迟单位ms，均为P50/P99。

| 后端 / 引擎 / 模型 / 客户端并发 | TTFT P50/P99 | TPOT P50/P99（last-visible） | ITL P50/P99（可见SSE文本事件） | 输出 tok/s | 设备采样峰值 | 采样footprint峰值（B） | 采样RSS峰值（B） | 成功/错误/拒绝；重复 | SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Metal / llama.cpp / 27B / C4 | 1846.615 / 13422.359 | 346.014 / 436.661 | 337.620 / 433.329 | 10.524548 | 未采集 | 3,886,755,456 | 4,039,245,824 | 32/0/0；1 | unknown |
| Metal / Ferrum / 27B / C4 | 1846.992 / 19019.053 | 351.164 / 471.327 | 337.967 / 1773.182 | 10.221030 | 未采集 | 1,906,946,176 | 364,331,008 | 32/0/0；1 | unknown |
| CUDA / llama.cpp / 27B / C4 | 114.191 / 372.604 | 24.727 / 27.188 | 23.737 / 53.060 | 147.284807 | 22,211 MiB（NVML全卡） | 未提供 | 未提供 | 32/0/0；1 | unknown |

Metal C4的Ferrum/llama输出吞吐比为 **0.971161**（本次低2.88%），未超过llama；这是固定先llama后Ferrum的一次探索比较，无重复方差/CI，也不代表SLO达标。两引擎仍有上述动态KV/固定共享KV和内存预算策略差异。CUDA Ferrum C4尚在运行，不计算CUDA引擎比值。

三份原生报告的40条完整选择（含8预热）已逐项核对一致，SHA均为 `6faca429d03d2f9718daf3fffa1a10382b88842c3432b41643b2596bcf0552c8`，原生元数据均记录pool208。Metal两份报告位于 `metal-explore-fullgpu-v3/serve-guard/qwen38-dense-{llama,ferrum}-explore-c4/client/report.jsonl`，文件SHA依次为 `69b6e911dd7ef83bbe9430da3a9075649989614c398a4ea82701dcc36d979093`、`b6fc308192514716b545d41df93634cc736b7e57a1120bd7e96501cc2fafc6de`；CUDA本机取证副本为 `analysis/cuda-primary-results/qwen38-dense-llama-explore-c4/client/report.jsonl`，SHA为 `39bda2159a4dd643a3e9ecc20effd9026038f731a5db9f341ddedae79c9082b4`。三格cell均退出0，Metal另有client/server退出0原始记录。既有Rust汇总位于 `analysis/primary-summaries/` 的 `metal-dense-{llama,ferrum}-explore-c4-fullgpu-v3.jsonl`、`cuda-dense-llama-explore-c4.jsonl`。

TPOT截止最后可见文本，按usage输出token减一归一；ITL保留全部可见SSE文本事件间隔。按表中行序，正式可见事件/间隔为9,684/9,652、9,680/9,648、9,683/9,651，观测到的transport coalescing均为0；严格单token诊断分别27/32、27/32、28/32，事件数与usage不一致的请求仍计入可见SSE ITL。预热与正式的全部协议质量计数均为0，这不等于语义质量或完整token一致率已验收。

Metal宿主内存来自整个client调用窗口（含预热）的4,620 / 4,791次采样，两格缺失及采集错误均为0；RSS与physical footprint重叠、不能相加，是采样下界，不能当作Metal设备分配峰值，也不是正式阶段或逐repeat峰值，不据较低Ferrum宿主值声称GPU内存更省。CUDA设备峰值来自既有Rust guard的 `nvml-summary.json`，250ms采样覆盖单个服务命令冷启动、加载、执行至关闭，未按正式阶段裁剪；22,211MiB是全卡已用内存采样峰，含驱动或其他分配，非该模型独占值或瞬时真峰。WSL逐进程NVML值不可用，宿主RSS/footprint未提供；两个采样器正常退出0。

仅一次重复，无跨重复方差/CI；未声明SLO阈值，状态为unknown。600秒单请求边界保持不变；这些报告只证明各自C4的前32个正式样本已完成，不揭示C16新增后32个样本的输出长度或高并发时长。风险说明见 `metal-explore-fullgpu-v3/timeout-risk-review.json`。

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

下表是准备/支持状态，不是性能排名。CUDA 每模型四组×四档、Metal 每模型两组×三档，共44格；新口径3/44格，旧数据限制见历史表。

| 后端 / 模型 | Ferrum GGUF | llama.cpp GGUF | Ferrum AWQ / vLLM（仅 CUDA） |
| --- | --- | --- | --- |
| CUDA / 27B | run与32槽serve短测均通过，GGUF C4主负载运行中 | 66/66层GPU；C4主负载32/32完成、零错误，147.284807 tok/s | CT的399组头部与实际 weight_shape 已核对；权重/镜像下载已为主测暂停，AWQ/vLLM未实跑 |
| CUDA / 35B-A3B | provider 格式缺口，未实跑 | 待下载与运行 | AWQ及混合BF16专家支持缺口，未实跑；vLLM 待部署 |
| M1 Max / 27B | 新探索固定16槽，C4主负载32/32完成、零错误，10.221030 tok/s | 16槽66/66层GPU；新C4主负载32/32完成、零错误，C8运行中 | 不适用 |
| M1 Max / 35B-A3B | run（1槽）与serve（32槽）均实测失败：混合 MoE provider 不接受 Q6_K | 全GPU16槽短测OOM；8槽语义短测通过但并发OOM；7槽41/41层GPU，C16短测32/32成功；固定7槽已入主负载队列 | 不适用 |

- Ferrum CUDA 27B compressed-tensors AWQ：399组实际CT量化头部的dtype、布局和配置维度均匹配现有路径，未发现头部层面的阻塞；BF16 scales会转为F16，比较时须披露该精度差异。399 个 `weight_shape` 实际值已通过 6,384B 范围读取核对，零不匹配（`analysis/cuda-ct-dense-weight-shapes.json`）；该检查点尚未验证完整权重payload、分片hash或CUDA加载/计算，不能用GGUF短测代替。证据为 `metadata/dense-ct-headers/` 与 `analysis/cuda-ct-dense-gap.json`，只计元数据审计完成。
- Ferrum CUDA 35B：`qwen35_config.rs` 不接受原生AWQ。固定头部确认第0层768个专家张量均为BF16，第1层768组三元组为I32 qweight/qzeros与F16 scales；qweight为 `[K,N/8]`，不同于现有GPTQ的 `[K/8,N]`，router为F32、共享专家为BF16。现有CUDA routed/shared MoE仅GPTQ/FP8 Marlin，因此除AWQ零点与repack外，还需按层选择dense专家路径；GGUF亦未覆盖。证据为 `metadata/awq-moe-headers/` 与 `analysis/cuda-awq-moe-gap.json`；payload、完整hash及CUDA实机行为未验证，不计实机失败。
- Ferrum Metal 35B 实跑：context2048/FP16 KV/Auto，run（1槽）与serve（32槽）均退出1，在 `qwen3_5.f32-master (ProgramCompilation)` 阶段失败；`operation.routed_shared_swiglu_moe` 的唯一候选 `provider.metal.routed_shared_swiglu_moe.f16.q4k` 返回 `unsupported_quantization_formats`，具体格式为 `quantization.gguf.q6-k`。现有Rust `gguf_inventory` 已核对文件：`blk.{34,38,39}.ffn_down_exps.weight` 均为Q6_K，每个220,200,960B，其余117个专家栈为Q4_K。混合 MoE provider 的 routed gate/up/down 要求Q4_K，额外Q6_K down支持仅用于 routed-only Qwen3；本轮已确认是实际启动失败。两份原始错误分别在 `metal-moe-smoke/run-guard/qwen36-moe-natural-run/rust-fixture.stderr`、`metal-moe-smoke/serve-guard/qwen36-moe-short-smoke/rust-fixture.stderr`，相邻 `rust-fixture.command.json` 保留完整命令。尚无模型输出，不记吞吐0。
- 27B Ferrum Metal复测：相同main产品 SHA256 `d1797e90dd419b50d802a2c14592d899910a1d4923b1240371a69d71230d1356`、context2048/FP16 KV/Auto，系统限制未改。默认 `--gpu-memory-utilization 0.9` 的1槽run通过，KV回答为正确两句话，完整输出44个ID含终止ID248046，guard=0（`metal-dense-smoke-memory-retry/`）。默认32槽serve明确拒绝：显式并发需要21,359,593,088B，超过20,615,498,410B预算（`metal-dense-serve-memory-default32/`）。只改文档化参数为 `--gpu-memory-utilization 1.0`，同32槽serve短测试通过：算术42、usage3；KV流式usage44且有 `[DONE]`，guard=0（`metal-dense-serve-memory-util10032/`）。原先 `metal-dense-smoke/` 的静态内存失败仍保留，但未暴露当时的采样预算，不能用本次快照回填；不将失败记为吞吐0。
- 上述成功复测的有效配置中，1槽run的available/usable为22,906,109,952 / 20,615,498,410B，context/decode计划峰值为16,906,678,832 / 16,433,928,768B；32槽serve显式1.0的available=usable=22,906,109,952B，context/decode计划峰值为16,979,635,344 / 21,359,593,088B。这些是启动计划成本，**不是实测分配峰值**，不能填入主表Peak Memory。
- 同文件 llama.cpp：32槽、共享KV16384、每槽2048、FP16KV、全GPU层在warmup出现 `kIOGPUCommandBufferCallbackErrorOutOfMemory`，首个算术请求HTTP500。单槽/共享KV2048诊断则回答42，第二条KV解释44个usage输出token、自然stop，guard退出0。单槽启动分配为MTL模型15356.48MiB、KV128MiB、递归状态149.62MiB、compute144.02MiB；不是主负载或逐格峰值。UD文件包含F32/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ3_S/IQ4_XS，不能当作所有张量均为Q4_K。两次配置与原始结果分别在 `metal-dense-llama-smoke/`、`metal-dense-llama-c1-smoke/`，进程均已退出并清理。
- 27B llama.cpp保持32槽/共享KV16384/每槽2048/FP16KV，改用文档化 `--gpu-layers auto --fit on` 后，两条相同短请求通过并自然stop，guard退出0。自动放置实选49/66层到GPU；日志报告启动时空闲设备内存17056MiB，目标保留1024MiB，未改变显式上下文。KV为GPU768MiB+CPU256MiB，部分递归状态和层计算在CPU，与全GPU运行的执行配置不同。短测试证据在 `metal-dense-llama-fit-smoke/`。
- 旧队列 `metal-primary-remaining-combined-v2/serve.guard.json` 的 llama C4、Ferrum C4 已完成，llama C8 在途时收到新口径后停止。guard16832 收到 SIGTERM 并完成清理，session83720 退出1，原 server21379/client21431 已消失。保留所有旧报告与部分日志；终止原因为用户调整方案，不计产品失败或新矩阵完成。仓库外 Rust guard v2 仍保留已声明作业时限和 `NO_PROXY/no_proxy`，不以超时或代理中断重测已完成格。
- 27B容量比较说明：两引擎均声明32槽、每请求输入与请求输出合计最多2048 token、FP16 KV，但总KV策略不同。Ferrum的KV/状态池按总运行时字节预算动态增长，2048不是共享池大小；llama.cpp显式预留共享16384-token KV池。Ferrum启动检查覆盖单序列完整上下文及32序列各一个已提交token的decode状态，不表示能同时容纳32个完整2048上下文；已完成C4也不证明该极限容量。Ferrum全Metal与llama的49层GPU加CPU放置同样是有意配置差异。审计证据为 `analysis/metal-kv-capacity-comparison.json`；其余并发能力和实际错误仍须由主负载验证。
- 35B llama.cpp保持32槽/共享KV16384/每槽2048/FP16KV：`--gpu-layers auto --fit on` 实选41/41层GPU，在warmup发生Metal OOM，首个请求HTTP500，guard退出1（`metal-moe-llama-fit-smoke/`）。随后改为 `--gpu-layers 16 --fit off --no-repack`，实选16/41层GPU、其余CPU；算术回答42（3个usage输出token），KV解释为两句话（39个usage输出token），均自然stop，流式含有效 `[DONE]`，guard退出0（`metal-moe-llama-gpu16-smoke/`）。启动日志为KV CPU192+GPU128MiB、递归状态CPU1273+GPU737MiB、Metal模型映射19914.65MiB及compute207.02MiB；这些不是驻留峰值或主负载结果。
- 新全GPU容量短测：llama.cpp显式 `--gpu-layers all --fit off`，共享KV16384/每请求2048/FP16、batch2048/ubatch512固定。27B16槽32请求/C16随机128输入+32输出全部成功；35B8槽在第8槽加入prefill后Metal OOM，32请求失败（不是仅启动失败），随后只减槽位，4/6/7槽均32/32成功，8槽失败与7槽通过形成相邻边界。新探索采用27B16槽、35B7槽，客户端仍扫C4/8/16，C高于槽位时排队。短测不计主矩阵、不证明所有长上下文组合；证据分别 `metal-fullgpu-capacity-saturated-r1/` 与 `metal-moe-fullgpu-slots{4,6,7}-saturation-r1/`。
- 新采样客户端：固定pool208（8预热+最多200正式）、seed42、官方35B tokenizer及原长度过滤，每格仅取所需前缀。不同请求数的完整selection SHA应不同，但实际样本前缀须相同；同格跨引擎SHA完全一致。原选择器会随请求总数重新reservoir抽样，因此不能只改数量并声称跨C同样本。改动限Rust基准参数/选择器/报告元数据，源码commit `59244b3380abe474389c3ad1deb659d53221c430`；本机13项最小测试与workspace fmt通过，远端CPU客户端同13项通过，两份release均已构建。Metal客户端固定为 `bin/ferrum-bench-client-selection-pool`，SHA256 `28245cdbbc3a2dd51ec25ddd48729660f4466c3f303822126e4c62a096090a6c`；构建、原生help和实际Cargo产物回执为 `analysis/client-selection-pool-local/client-artifact-receipt.json`。推理二进制单独冻结，旧Metal SHA不变，CUDA服务器release SHA为 `efad1116ee4fac2f5a3b9302315cf3fffa00fc28d4b113d6156e46d0eed58c86`。尚未执行本次完整workspace check/test/clippy，不据此声明PR已验证。
- 新Metal队列由Rust guard串行执行：`metal-explore-fullgpu-v3/serve.guard.json`，guard PID42215、工具会话88606；llama/Ferrum C4已正常完成，当前执行llama C8。顺序为27B llama/Ferrum C4、C8、C16交替，再35B llama C4/C8/C16，共9个可运行格；Ferrum 35B的3格仍列provider缺口。27B固定16槽、35B固定7槽，llama显式all GPU/fit off，FP16 KV；每格8预热，正式32/32/64、重复1，pool208，单请求600秒。llama各格生命周期6000秒、Ferrum各格7200秒。启动时的原始PID、命令及小型证据索引保留在 `metal-explore-fullgpu-v3/execution-receipt.json`，该回执是当时快照，不代表首格仍在运行。本机测量窗口内不运行构建、测试或其他重负载。
- CUDA引擎与控制器已构建：远端任务根 `/home/seekee/ferrum-slo-20260923/popular-models-20261007` 下，客户端SHA256 `114ee119d853b8f585ea049021aa9f62eb397c82a4b7ab249afc718b2c00db49`；guard SHA256 `8ff31e5df75b7721a51060144b528012d45481191e63acfc5b8afc32245b27c8`，fmt/test/release退出0、66项测试通过。llama r3构建退出0，server SHA256 `a290792b8b47609741f46a260fd683a93d2b4459f3e82f4cb4f56cc315ddf8cc`、CLI SHA256 `e76209300148f6c20ea58f013b50337816c4cf3db74bdf43d37238c2489420b9`。早期构建回执中的 `not_runtime_validated` 仅代表当时快照；后续原生进程短测与恢复已实际执行，Docker/vLLM模型生命周期仍待验证。
- CUDA 27B GGUF功能短测：Ferrum `run-smoke-r1` 加载约6.6秒、KV解释正确且为两句话/44输出token，guard退出0；`serve-smoke-r1` 固定32槽/context2048/FP16、默认utilization0.9，启动预算27.18GiB（计划预算，不是实测峰值），算术42/usage3、KV usage44且有有效 `[DONE]`，guard退出0。llama `serve-smoke-r1` 核实66/66层GPU、n_seq_max32、每槽2048，算术42/usage3与KV usage44/`[DONE]`均通过。llama `run-smoke-r1` 因不支持旧 `--conversation` 参数而退出1，属于命令参数错误，保留该失败；仅移除该参数、其余参数不变的独立r2正确回答KV，guard退出0。所有成功短测均记录 `restored_verified`，失败r1亦已恢复原服务；这些短测不计44格完成数，也不证明32个完整上下文同时容纳。
- CUDA 8格GGUF探索由guard继续执行：`guard/configs/cuda-dense-gguf-primary-alternating.guard.bound.json` SHA256 `33dd572e23cc94df461f1eabc21e8fc3a69f78f1cb1090e60d4c6d432709df1e`，launcher PID1346179、guard PID1346180；首格llama C4已完成，当前执行Ferrum C4，输出根为 `receipts/cuda-dense-gguf-explore-alternating-r1/output`。顺序为llama/Ferrum C4、C8、C16、C32交替，两引擎固定32槽，pool208、8预热、正式32/32/64/128、重复1，单请求600秒、每格6000秒。Ferrum保留动态KV与utilization0.9，llama保留共享KV16384、每槽2048、全GPU/fit off；并非相同总KV策略。仅llama C4计入完成数，SLO仍unknown。模型下载与镜像拉取已主动暂停以避免测量争用：两个aria父进程退出7、镜像拉取退出130，6个拥有的准备进程均已消失；远端意图记录为 `downloads/pauses/dense-gguf-explore-r1/intent.json`。本机回执 `analysis/cuda-preparation/receipts/cuda-dense-runtime-and-launch-r1.json` 保存启动时版本、短测分类及root中继的运行事实，该快照不代表首格仍在运行。
- 主机：本机为 M1 Max/32GiB/macOS15.1.1，CUDA RTX 5090为32607MiB，按指定helper经本机→Mini→WSL连接。此前“主机认证阻塞”记录撤回。waifugemma4使用原有服务身份配置；PID4162656、占用18502MiB及8081 HTTP200是准备时快照，guard恢复后PID已改变，不能当作当前身份。短测已多次停止并验证恢复该服务；本次主队列的生命周期为 `temporarily_stopped_restore_required`，退出后仍须核验 `restored_verified`。私钥仅在Mini，服务切换与恢复均由既有guard负责。
- 测量：仓库外Rust `analysis/report-reader/` 的last-visible读取通过2项测试，已成功读取上述三格原生报告；历史schema缺字段仍被明确拒绝。`analysis/quality-replay/` 的collector已通过7项测试、fmt与锁定离线release构建，pin未变；独立原始ID比较工具通过8项测试、fmt和release构建，有1条unused `token_record` helper警告，证据为 `analysis/quality-replay/comparison-validation.json`。真实64题质量/一致率套件仍未运行；其余主负载、设备峰值与质量证据待完成，短测试不替代验收。

当前两台机器按冻结配置并行执行Metal与CUDA GGUF探索队列，同机串行；其他权重和镜像准备暂停，待测量窗口结束再恢复，并补真实64题质量/一致率。P0 尚缺完整运行矩阵、两后端各自对照的 profile 与按耗时排序的缺口表。Qwen3.5-9B 的 #402 CUDA C4/C8 回归基线仍为284.688/414.436 tok/s，不以本轮结果替代回归。已取消的 vllm-metal 环境及其uv下载缓存已删除以释放磁盘，仓库外小型元数据保留；不再测试或下载Metal vLLM权重，不属于当前验收范围。

独立工作区：`/private/tmp/ferrum-popular-models-20261007`；全部过程、配置、元数据、下载和日志在仓库外 `/private/tmp/ferrum-popular-models-p0-20261007/`。恢复工作须核验进程/会话的当前状态，不能仅凭这些路径判断任务仍在运行。
