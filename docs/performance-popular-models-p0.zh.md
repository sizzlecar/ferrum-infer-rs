# 热门模型并发 P0（2026-10-07，进行中）

[目标](goal-popular-models-concurrency.zh.md)尚未完成。当前在 `main` 的 `0680840becae342e481df4ac0705c43fc08633a7` 上准备基线；未修改产品代码，未取得本轮主负载吞吐比值，也未进入 P1。原工作区的旧 SLO 修改保留。按用户最新决定，Metal 仅比较 llama.cpp，vLLM 仅在 CUDA 测试。

## 固定输入与对照

| 输入 | 固定版本与状态 |
| --- | --- |
| Qwen3.8 GGUF | `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502`，`Qwen3.8-27B-UD-Q4_K_M.gguf`，16,464,440,224 B；本机完整 SHA256 已核对：`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Qwen3.6 GGUF | `unsloth/Qwen3.6-35B-A3B-GGUF@a483e9e6cbd595906af30beda3187c2663a1118c`，`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`，20,893,015,008 B；已下载并核对完整 SHA256 `a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c` |
| Qwen3.8 AWQ | `cyankiwi/Qwen3.8-27B-AWQ-INT4@6e134bae811fb5adac50ee042ae5f029ac6779aa`；实际为 compressed-tensors INT4/G32/非对称；5/5分片的399组量化头部布局已核对，未下载权重payload |
| Qwen3.6 AWQ | `QuantTrio/Qwen3.6-35B-A3B-AWQ@119886a1072372348f73ef0df2d801cdcc0f455b`；AWQ/GEMM、4bit、group128、zero-point；已核对索引与2/9分片头部，完整覆盖第0/1层；未下载或验证权重payload |
| semantic | Qwen3.8 官方 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；Qwen3.6 官方 `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| llama.cpp | 固定 [b11429](https://github.com/ggml-org/llama.cpp/releases/tag/b11429)，commit `d81235049384534c167caea52b85a694f6103d14`；macOS arm64 包已按发行摘要校验；27B、35B短测试通过，27B主负载C1的r4已启动、待完成 |
| CUDA vLLM | 官方候选 `vllm/vllm-openai:v0.31.0-cu129`、`linux/amd64` 固定digest `sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f`；镜像配置声明CUDA12.9.1/SM12.0目标，尚未拉取、部署或在5090验证 |

ShareGPT 延用 `anon8231489123/ShareGPT_Vicuna_unfiltered@745745adf6cd15b84e4f1c4a5a051fb4304f9342` 的 `ShareGPT_V3_unfiltered_cleaned_split.json`，本机已下载并核对完整 SHA256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。

为保留 #402 的原始样本与输出预算，选择器继续使用已核对 SHA256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` 的 tokenizer；服务端各用目标模型自己的 tokenizer 和模板。27B 官方 tokenizer 的 SHA256 是 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`，不能直接换入选择器并声称样本未变。

每格 32 预热 + 64 测量、一次重复、seed/sampling-seed 42；输入 4–1024、输出至少 4、输入+参考输出+32-token 模板预留 ≤2048；参考输出长度、ignore_eos、thinking off、temperature 0、top_p 1、repetition penalty 1。预期选择 SHA 为 `03676b00ee0b9e8407f59e3fb531eeff99dde5ccc9e59844b6abf7d22f1d5df2`，以新客户端实取证据再核对。不同引擎和并发档复用同一选择，服务容量固定32，FP16 KV、prefix/session cache off。新模型 S 尚未按目标规则计算，不预填 SLO pass。

## 当前缺口与矩阵状态

下表是准备/支持状态，不是性能排名。CUDA 每模型四组、Metal 每模型两组，共 12 组 × 5 个并发档 = 60 格；尚无已完成的本轮主负载格。

| 后端 / 模型 | Ferrum GGUF | llama.cpp GGUF | Ferrum AWQ / vLLM（仅 CUDA） |
| --- | --- | --- | --- |
| CUDA / 27B | 待运行 | 待运行 | CT的399组头部布局已核对，未实跑；主机认证阻塞 |
| CUDA / 35B-A3B | provider 格式缺口，未实跑 | 待下载与运行 | AWQ及混合BF16专家支持缺口，未实跑；vLLM 待部署 |
| M1 Max / 27B | run 1槽默认0.9通过；serve 32槽默认0.9容量拒绝，显式1.0短测试通过 | 全GPU32槽OOM；短测试通过；固定49层GPU/32槽的主负载C1 r4进行中，尚无完成结果 | 不适用 |
| M1 Max / 35B-A3B | run（1槽）与serve（32槽）均实测失败：混合 MoE provider 不接受 Q6_K | 32槽自动放置OOM；固定16层GPU/32槽短测试通过，未完成主负载 | 不适用 |

- Ferrum CUDA 27B：399组实际CT量化头部的dtype、布局和配置维度均匹配现有路径，未发现头部层面的阻塞；BF16 scales会转为F16，比较时须披露该精度差异。尚未验证 `weight_shape` 实际值、权重payload、完整分片hash或CUDA加载/计算。证据为 `metadata/dense-ct-headers/` 与 `analysis/cuda-ct-dense-gap.json`，只计元数据审计完成。
- Ferrum CUDA 35B：`qwen35_config.rs` 不接受原生AWQ。固定头部确认第0层768个专家张量均为BF16，第1层768组三元组为I32 qweight/qzeros与F16 scales；qweight为 `[K,N/8]`，不同于现有GPTQ的 `[K/8,N]`，router为F32、共享专家为BF16。现有CUDA routed/shared MoE仅GPTQ/FP8 Marlin，因此除AWQ零点与repack外，还需按层选择dense专家路径；GGUF亦未覆盖。证据为 `metadata/awq-moe-headers/` 与 `analysis/cuda-awq-moe-gap.json`；payload、完整hash及CUDA实机行为未验证，不计实机失败。
- Ferrum Metal 35B 实跑：context2048/FP16 KV/Auto，run（1槽）与serve（32槽）均退出1，在 `qwen3_5.f32-master (ProgramCompilation)` 阶段失败；`operation.routed_shared_swiglu_moe` 的唯一候选 `provider.metal.routed_shared_swiglu_moe.f16.q4k` 返回 `unsupported_quantization_formats`，具体格式为 `quantization.gguf.q6-k`。现有Rust `gguf_inventory` 已核对文件：`blk.{34,38,39}.ffn_down_exps.weight` 均为Q6_K，每个220,200,960B，其余117个专家栈为Q4_K。混合 MoE provider 的 routed gate/up/down 要求Q4_K，额外Q6_K down支持仅用于 routed-only Qwen3；本轮已确认是实际启动失败。两份原始错误分别在 `metal-moe-smoke/run-guard/qwen36-moe-natural-run/rust-fixture.stderr`、`metal-moe-smoke/serve-guard/qwen36-moe-short-smoke/rust-fixture.stderr`，相邻 `rust-fixture.command.json` 保留完整命令。尚无模型输出，不记吞吐0。
- 27B Ferrum Metal复测：相同main产品 SHA256 `d1797e90dd419b50d802a2c14592d899910a1d4923b1240371a69d71230d1356`、context2048/FP16 KV/Auto，系统限制未改。默认 `--gpu-memory-utilization 0.9` 的1槽run通过，KV回答为正确两句话，完整输出44个ID含终止ID248046，guard=0（`metal-dense-smoke-memory-retry/`）。默认32槽serve明确拒绝：显式并发需要21,359,593,088B，超过20,615,498,410B预算（`metal-dense-serve-memory-default32/`）。只改文档化参数为 `--gpu-memory-utilization 1.0`，同32槽serve短测试通过：算术42、usage3；KV流式usage44且有 `[DONE]`，guard=0（`metal-dense-serve-memory-util10032/`）。原先 `metal-dense-smoke/` 的静态内存失败仍保留，但未暴露当时的采样预算，不能用本次快照回填；不将失败记为吞吐0。
- 上述成功复测的有效配置中，1槽run的available/usable为22,906,109,952 / 20,615,498,410B，context/decode计划峰值为16,906,678,832 / 16,433,928,768B；32槽serve显式1.0的available=usable=22,906,109,952B，context/decode计划峰值为16,979,635,344 / 21,359,593,088B。这些是启动计划成本，**不是实测分配峰值**，不能填入主表Peak Memory。
- 同文件 llama.cpp：32槽、共享KV16384、每槽2048、FP16KV、全GPU层在warmup出现 `kIOGPUCommandBufferCallbackErrorOutOfMemory`，首个算术请求HTTP500。单槽/共享KV2048诊断则回答42，第二条KV解释44个usage输出token、自然stop，guard退出0。单槽启动分配为MTL模型15356.48MiB、KV128MiB、递归状态149.62MiB、compute144.02MiB；不是主负载或逐格峰值。UD文件包含F32/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ3_S/IQ4_XS，不能当作所有张量均为Q4_K。两次配置与原始结果分别在 `metal-dense-llama-smoke/`、`metal-dense-llama-c1-smoke/`，进程均已退出并清理。
- 27B llama.cpp保持32槽/共享KV16384/每槽2048/FP16KV，改用文档化 `--gpu-layers auto --fit on` 后，两条相同短请求通过并自然stop，guard退出0。自动放置实选49/66层到GPU；日志报告启动时空闲设备内存17056MiB，目标保留1024MiB，未改变显式上下文。KV为GPU768MiB+CPU256MiB，部分递归状态和层计算在CPU，与全GPU运行的执行配置不同。短测试证据在 `metal-dense-llama-fit-smoke/`。
- 27B主负载C1固定 `--gpu-layers 49 --fit off` 和32槽容量，`metal-dense-llama-primary-c1-r4/` 已启动（session20613），同一冻结参数/样本、6000秒时限和 `NO_PROXY/no_proxy=127.0.0.1,localhost,::1`。r3因发现仓库外guard普通client仍有3600秒上限而主动停止，不计产品或性能失败；更早的时限/代理重试也不产生完成格。guard v2已将普通client截止时间改为作业声明时限，61项测试、fmt和release构建通过，产品代码未改。后续14格已合并为串行队列 `metal-primary-remaining-combined-v2/serve.guard.json`：27B两引擎的C4/8/16/32、Ferrum C1及35B llama的五档，尚未启动；不再重复执行各子集。所有主负载吞吐、延迟、SLO和实测峰值仍待完成。
- 27B容量比较说明：两引擎均声明32槽、每请求输入与请求输出合计最多2048 token、FP16 KV，但总KV策略不同。Ferrum的KV/状态池按总运行时字节预算动态增长，2048不是共享池大小；llama.cpp显式预留共享16384-token KV池。Ferrum启动检查覆盖单序列完整上下文及32序列各一个已提交token的decode状态，不表示能同时容纳32个完整2048上下文；现有短测也只覆盖两个顺序请求。Ferrum全Metal与llama的49层GPU加CPU放置同样是有意配置差异。审计证据为 `analysis/metal-kv-capacity-comparison.json`；并发能力和实际错误须由主负载验证。
- 35B llama.cpp保持32槽/共享KV16384/每槽2048/FP16KV：`--gpu-layers auto --fit on` 实选41/41层GPU，在warmup发生Metal OOM，首个请求HTTP500，guard退出1（`metal-moe-llama-fit-smoke/`）。随后改为 `--gpu-layers 16 --fit off --no-repack`，实选16/41层GPU、其余CPU；算术回答42（3个usage输出token），KV解释为两句话（39个usage输出token），均自然stop，流式含有效 `[DONE]`，guard退出0（`metal-moe-llama-gpu16-smoke/`）。启动日志为KV CPU192+GPU128MiB、递归状态CPU1273+GPU737MiB、Metal模型映射19914.65MiB及compute207.02MiB；这些不是驻留峰值或主负载结果。
- 主机：已确认本机为 M1 Max/32GiB/macOS15.1.1。CUDA 的文档路由为本机→Mini→WSL；Mini Tailscale 可达，但现有密码和本机公钥认证均失败，未进入5090、未改原服务。
- 测量：main保留每请求last-visible timing；无S时常规 `tpot_ms` 是终止时刻口径，不能直接入主表。仓库外Rust `analysis/report-reader/` 的last-visible读取通过2项测试；当前传输失败诊断保留errors=1、延迟null、SLO=unknown，旧schema缺字段被拒绝。`analysis/quality-replay/` 已通过7项测试、fmt与锁定离线release构建，用于保留原始响应和完整ID证据；尚未运行真实64题质量/一致率套件。主负载报告、逐格实测内存与质量证据仍待完成，短测试不替代验收。

下一步完成27B C1采集、核验样本与测量口径，再按冻结配置采集其余可运行格；P0 尚缺运行矩阵、两后端各自对照的 profile 与按耗时排序的缺口表。Qwen3.5-9B 的 #402 CUDA C4/C8 回归基线仍为284.688/414.436 tok/s，不以本轮准备工作替代回归。已取消的 vllm-metal 环境及其uv下载缓存已删除以释放磁盘，仓库外小型元数据保留；不再测试或下载Metal vLLM权重，不属于当前验收范围。

独立工作区：`/private/tmp/ferrum-popular-models-20261007`；全部过程、配置、元数据、下载和日志在仓库外 `/private/tmp/ferrum-popular-models-p0-20261007/`。恢复工作须核验进程/会话的当前状态，不能仅凭这些路径判断任务仍在运行。
