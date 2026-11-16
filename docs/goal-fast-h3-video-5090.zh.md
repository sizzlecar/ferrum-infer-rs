# Goal：FastH3 原生视频、连续批处理与 RTX 5090 性能

## 1. 目标与授权

用户目标：Ferrum 原生支持 `FastVideo/FastH3-Omni-NVFP4-QAD`，在指定 RTX 5090 上比 FastVideo 更快，并支持视频连续批处理。以核心抽象承载功能，不堆叠模型特判或另起平行运行时。

完成意味着：同一 Rust pipeline 贯通 `ferrum run` / `ferrum serve`，真实模型输出通过正确性和容量验证；多个视频请求能够在去噪步边界动态加入、完成、取消；在同硬件、同 checkpoint、同数值/画质配置和同工作负载下，有可复查的端到端优于 FastVideo 的测量。接口占位、模型能加载、单个算子通过、视频文件可播放或微基准变快，均不等于完成。

用户追加约束：**容量不足允许变慢，不应直接让有效请求报错**。对已接纳的受支持请求，共批候选放不下时自动缩小批次、串行推进，或等待现有任务释放资源；保持原精度、PDD 步数和解码质量。将“两条已接纳请求在 B2 容量不足时都通过 B1 最终完成”作为明确验收项，覆盖编译计划容量和真实动态 backing 压力。内部候选容量拒绝不等于请求失败；仍保留真实原生执行故障与取消的原始结果，等待不能变为无进展的挂起。

用户已授权将该 5090 用于测试并停止现有服务。使用已有 SSH 密钥与管理助手；保存被停服务的定义、配置及恢复方法。原工作区的未提交改动保持原样。大权重、第三方源码、视频、日志、trace、构建产物和报告都放库外。

2026-11-14 用户追加执行顺序：**先完成本文件的视频功能、容量与性能验收，再处理第 10 节的 Windows v0.19.1 外部测试反馈**。该后续阶段已获授权，视频完成后直接接续；其工作不替代视频退出条件。完整工作队列包含两个阶段，视频验收结束不代表全部追加事项完成。

## 2. 固定起点与事实边界

- 开发基线：`origin/main` 的 `7bbea02877df04132ae7407a194ecb9d669265c2`（v0.19.1）。独立分支 `feat/fast-h3-video-5090-20261113`，工作树 `/Users/chejinxuan/rust_ws/ferrum-fast-h3-video-20261113`。
- 模型：`FastVideo/FastH3-Omni-NVFP4-QAD`，已部署 revision `f01e6a23ac6066aa9eb622cf94b6259442e70ec2`。
- 参考运行时：FastVideo revision `866a3cdb91a85aef8561847365fc8564add9e4c8`；部署版本、依赖、源码及本地修改一并归档，不能仅凭 revision 假定安装字节相同。
- 设备：RTX 5090 32 GB，WSL2，SM120。现成部署目录 `/home/seekee/fastvideo-h3-5090-20261111`，原生 API `127.0.0.1:18196`。
- 固定质量配置：NVFP4 QAD、八次 transformer evaluation、24 FPS、36 层质量解码器、CFG 1；视频 sparse attention、FP8 conditioning prefix、P 两级量化及 K smoothing 保持参考配置。八层快速解码器或改变精度不能冒充同质量提速。
- 部署起点配置：FFN chunk 4096、CUDA allocator cap 28 GiB、文本编码器卸载、AdaLN 表。正式对照后来将两侧 cap 同时固定为 29 GiB，见第 9 节。配置上限不是实测显存峰值，历史成功耗时不替代固定配置的同场测量。
- 部署开启 `FASTVIDEO_MINIMAX_H3_FUSIONS=all`，优先于 `EXACT_KERNELS=all`；SwiGLU、RMS/modulation、QK RMS/RoPE 保留融合路径的 F32 中间值，不能用 eager 的额外 BF16 舍入代替。视频 VSA 的 P 是两级 FP4、V 为 FP4、MMA 累加 F32、最终 BF16；prefix 分支独立使用 FP8 QK 和 BF16 V。旧 pilot provenance 的“BF16 P.V”短句仅是简写，正式比较必须分开记录两条数值路线。
- 启动本目标时，历史 832×480 样本已有 124/243 帧和完整音视频解码记录；768p 当时只有页面选项。本阶段新增的实际资格见第 9 节。
- 证据根：本机 `/Users/chejinxuan/ferrum-handoffs/20261113-ferrum-fast-h3`；5090 `/home/seekee/ferrum-fast-h3-20261113`。旧部署资料保留在 `20261111-fastvideo-h3-5090`。

来源：[模型卡](https://huggingface.co/FastVideo/FastH3-Omni-NVFP4-QAD)、[FastVideo 推理说明](https://github.com/hao-ai-lab/FastVideo/blob/866a3cdb91a85aef8561847365fc8564add9e4c8/docs/inference/fasth3_omniref.md)。实现依据固定源码和实际下载的元数据，不跟随浮动 main 修改语义。

## 3. 产品范围

首个完整执行切片为一张参考图、832×480、124 帧、音视频联合生成；它只是最小验收入口，不是最终交付范围。后续完成模型声明的有序图片、视频和声音参考处理，以及受容量约束的 5/10/15 秒请求。纯文本输入是否支持以这个 Ref2VA checkpoint 的真实合同为准，不能根据其他 H3 变体推定。

CLI 与服务共享模型来源解析、精度、帧数对齐、资源预算、输入限制和模型默认值。API 返回请求值和实际值，例如请求约五秒时产生 124 帧，不能把 124/24 秒标成精确五秒。不支持的参考组合、分辨率、数量、时长和精度显式拒绝；能力声明反映运行中的模型与后端，而非页面提供的选项。

首个性能主表固定 480p 五秒档；十秒、十五秒及长参考承担功能/容量验证并分别报告性能。768p 在独立正确性和容量验证后才宣告支持。CUDA 是本目标交付后端，Metal 不因共享 Rust 接口而自动获得视频资格。

## 4. 已有基础与必须改变的边界

已有 `ferrum-runtime` 的 Tensor work、`ferrum-execution` 的 TensorComponentSession、受管 Host/device 资源、组件驻留/退休、结构化 attention、图像去噪步协议和输出额度。继续使用这些权威，不能新增第二套设备分配账本。

目标启动时的图像路径有明确领域限制：`SingleFrameLatentShape` 拒绝多帧；P-H3 图片 packing、音频行和 VAE 存在单帧语义；P-H3 的 Comfy W4A8/INT8 与 NVFP4 不兼容；P-H3 图片 Euler 日程也不能仅因都是八步就替代 FastH3 PDD。

代码入口：

- [张量工作](../crates/ferrum-runtime/src/execution/work.rs)
- [组件会话](../crates/ferrum-execution/src/component.rs)
- [共享迭代执行协议](../crates/ferrum-engine/src/iterative_service/steps.rs)
- [输出准入](../crates/ferrum-engine/src/iterative_service/output_gate.rs)
- [单帧 latent](../crates/ferrum-diffusion/src/spatial_latent.rs)
- [P-H3 图片布局](../crates/ferrum-models/src/p_h3_image/layout.rs)

## 5. 核心抽象与依赖

优先在现有 crate 内按责任建立模块。抽象必须有图像和视频的实际调用者；不先造插件框架或大而全的推理图 DSL。以下名称表达职责，最终 Rust 命名可以调整。

| 所在层 | 应拥有的职责 | 边界 |
|---|---|---|
| `ferrum-types` | 类型化任务参数、能力、精度策略、输出描述、进度和配置 | 不拥有设备 handle、模型推导规则或调度算法 |
| `ferrum-diffusion` | checked 时空形状、布局变换、可复用 solver 数学合同 | 不认识 FastH3 checkpoint 字段、HTTP 或设备分配策略 |
| `ferrum-models` | 模型包、conditioning、时空位置、PDD 多模态日程、组件计划和批组成的语义校验 | 不签发物理执行授权，不启动服务，不私建 allocator，不以模型名选择服务调度 |
| `ferrum-quantization` | NVFP4 等物理存储格式、scale/packing 校验与来源映射基础 | 格式不等于模型；读取权重不授权 GPU 分配或执行 |
| `ferrum-runtime` | 规范算子语义、形状/精度约束、资源授权、提交/完成/退休 | 不认识视频 API、模型文件名或性能 SLO |
| `ferrum-kernels` / native ops | SM120 算子 provider、能力、workspace 和原生生命周期 | 不决定请求是否入队、模型采样日程或输出质量档 |
| `ferrum-execution` | 组件准备/执行、受管会话、批次执行事实与恢复所有权 | 无模型专属旁路；不另起服务线程或资源池 |
| `ferrum-engine` | 通用迭代生成生命周期、条件准备/去噪/解码协调、取消、产物交接 | HTTP 和编解码不进入去噪热循环 |
| `ferrum-scheduler` | 基于声明能力、合法批次和资源/成本证据选择就绪工作 | 不读取 latent 内容，不改 solver，不自行分配 KV/显存 |
| `ferrum-output` | 有界帧/音频输出、编码接口、产物所有权与读取额度 | 生命周期不跟随 GPU 请求结束而提前释放 |
| CLI / server | 共用装配入口、协议校验、作业查询与产物传输 | 不各写一条模型流程或隐含环境组合 |

依赖遵循已有方向：模型/量化提供描述，runtime 定义执行端口，kernels 实现设备操作，execution 驱动组件，engine 组合领域流程，scheduler 只作策略选择。不得通过“公共类型”反向引用高层实现形成依赖环；需调整公共契约时以实际依赖闭包决定归属。

### 5.1 图像和视频共享迭代执行协议

从现有 image service 抽取真正通用的请求准备、就绪、组批、单步提交、提交结果、暂停、退出和输出准入机制。图像保持 typed image 请求/结果，视频保持 typed video 请求/结果；核心 worker 不判断 `Image`/`Video` 或模型名字来决定生命周期。

每个请求独立拥有 conditioning 引用、目标 latent、模型 solver continuation、RNG 身份、取消状态及资源授权。模型 adapter 负责判断多个 continuation 能否被一个 forward 承载，并生成精确成员、行范围、mask、每请求时钟和输出切片的执行描述。调度器不能将 Vec 长度当作合法 batch 证明。

模型描述只证明语义合法；managed buffer identity、生命周期、真实容量和提交许可由 runtime/execution 结合活跃 owner 验证并签发。engine 的唯一受管 worker 持有队列、continuation、in-flight batch 和恢复状态；scheduler 只消费不可变快照并返回候选，execution 持有具体提交与 fence。不能保留两套会推进同一请求的 engine/scheduler 循环，也不为视频强行改写文本 scheduler。

一次 forward 的 batch owner 持有完整提交快照、授权与 completion fence。设备完成后先验证/准备所有 successor，再按既有提交协议发布请求进度。提交后的失败不得退化为整批逐个重试；只有证明未提交的工作才能重试。修改成员、卸载权重或复用 workspace 必须发生在对应 fence 之后。

提交快照绑定每个请求的 continuation generation、solver index 和资源 incarnation；发布时重新验证，每步只能提交一次。旧 completion 不得覆盖已取消、已迁移或新一代状态。successor 准备使用明确的 ping-pong/别名计划，不要求复制整个 batch；RNG 按请求与步骤绑定，入批次序不能改变噪声或输出。

迁移需同时接入现有 Qwen/P-H3 图像实现并保留现有行为测试。允许短期兼容 adapter，但视频交付前删除被替代的平行 worker/账本；仅改名、复制 image service 为 video service 或给 causal token executor 塞视频分支都不算迁移完成。图像专属 PNG、参考图解析和尺寸规则留在 image adapter。

### 5.2 时空数据与多模态时钟

新增独立的 checked 时空 latent 合同，明确 batch、channel、time、height、width、patch、轴序和有效区域；保留单帧合同。时间轴不能借用 batch 或折叠进 height 后遗失语义。所有乘积、对齐、字节数和缓存需求在分配前检查溢出和资源界限。

通用几何不硬编码 FastH3：模型 adapter 声明帧对齐、时间/空间压缩、channel、参考布局和位置映射。该固定版本的实际公式是视频帧对齐到 `17n+5`、视觉 latent 为 `5n+2`、音频每声道长度按 `round(frames/24*40)`；这些规则属于 FastH3，不能提升为全部视频模型的规则。

有序参考和生成目标是不同的 typed 区域。参考图/视频/音频的 conditioning 与 latent 不得因去噪步推进被错误修改。音视频目标共享一次模型调用，但拥有各自的 schedule/时钟；保留联合生成语义。首版输出也保留音轨，不能靠跳过音频计算获得虚假提速。

### 5.3 FastH3 PDD 与预计算

按模型包验证 `fastvideo_inference.json`、transformer、视频/音频 scheduler 的一致性；未知 schema、冲突字段、非法分区和不支持的 attention 语义失败关闭。PDD 的分区、加权 heads、输出 channel-major 布局和实际更新公式逐项对齐固定可执行源码。普通 Euler 不能充当 PDD 的 adapter。

该版本的 video/audio shift 分别为 12/3，32 个 PDD 子步分为八次 evaluation。每个请求持有自己的步索引，不共享全局游标；同一 batch 可包含不同进度仅在模型明确支持每请求时间及对应 AdaLN 时成立。

混进度还必须表达每个请求的 PDD head 区间和音视频融合系数：不同区间对应不同 final projection 融合权重，不能用一组全局 `proj_out`/`audio_proj_out` 处理所有成员。模型计划可共享 transformer 主体，再执行显式分组或按请求索引的输出 projection。初期可按同一区间分组，但跨步动态补位的最终验收仍需实际实现合法的混进度执行。所有 attention 区域与文档边界保证请求之间互不可见。

AdaLN 表是可验证的派生产物：身份包含权重内容、schedule、参考模态、dtype 与数值策略。Rust 端生成或读取明确版本化的中立格式，并验证其来源；生产推理不依赖执行 Python、pickle 或 `.pt` 反序列化。源表可作为库外参考，不能作为无身份的隐藏全局缓存。

### 5.4 量化与原生算子合同

NVFP4 独立描述 E2M1、group16、E4M3 block scale、128×4 scale swizzle、全局 scale 与 FP32 alpha；不得复用 MXFP4 group32/E8M0 或 Comfy codebook 的格式 ID。源包命名由显式 source adapter 映射，通用算子不认识 checkpoint tensor 名字。

保留实际 FP32 alpha，不能用已舍入的 BF16 global scale 重算。静态/动态 activation scale、量化舍入、零/饱和值、矩阵转置、输出 BF16 和 bias 的舍入顺序都是数值合同。实际未量化层按声明保留其精度；不得静默全量反量化成 BF16，破坏 32 GB 目标。

通用 NVFP4 projection 与结构化稀疏 attention 分离。FastH3 adapter 声明 VSA 的 tile/区域/参考保留规则，provider 实现 FP4 视频 query 与高精度 prefix 的数值路线；模型 conditioning 的精度保护不能被“全 FP4 更快”替代。

native ABI 接收 caller-owned buffers、stream 和有界 workspace；缓存、library 内部容量及其失效纳入现有 runtime。第三方 kernel 来源与工具链沿 pinned native operator-set 机制集成，不复制整个依赖树到仓库，不调用 Python 推理进程完成模型执行。

### 5.5 容量、驻留与连续批处理

统一预算包括权重、量化 scale、AdaLN、conditioning、参考/目标 latent、gather/scatter、attention scratch、提交前后同时存活的 solver 状态、VAE 时序/分块缓存、编码缓冲、Host staging 和原生库 allowance。输出落盘有独立有界容量，不把磁盘当无限内存。设备授权依据实际计划，不依赖显存空闲值的瞬时采样。

候选计划的容量预测应计算新增需求。只有借用真实存活的共享权重 owner，并通过原运行时、物理布局与准入合同校验，才能免计已驻留静态权重；编译元数据或历史收据不是驻留证明。旧批次的闲置临时缓冲必须在确认完成的边界通过原 owner 实际释放，再观察剩余额度，不能先减去假定可回收字节。预测不产生许可，最终仍由现有准入与回滚负责；拒绝新批次后，旧请求通过原路径重新取得临时缓冲并推进。

先支持同模型、数值 profile 和输出规格的请求在去噪步边界补位；同规格仍需区分参考长度/模态与 prefix 语义。随后利用合法 packed/ragged 计划支持不同长度和进度。兼容性至少包含模型权重/adapter 身份、PDD/head 布局、精度、attention/patch 合同及可执行形状；每请求 seed 和 cursor 继续独立。

条件编码、去噪和解码共享同一设备预算。入队不意味着立即加载辅助模型；调度器比较已就绪请求和组件切换成本，避免频繁卸载驻留 DiT。活跃请求、候选 batch、pending 输入与解码输出分别设界限，压力通过可观察的资源事件唤醒，避免忙轮询或无限等待。

取消区别于删除 HTTP 记录。待处理请求可直接退出，已提交请求在安全 fence 后回收；GPU 终态、solver 提交、编码完成、产物发布、客户端读取是独立状态。decoder/output 受压不得丢失已有请求或完成所有权，服务关闭必须能收敛并保留不可恢复的原生错误。

真实连续批处理的证据必须包括：到达与入批时刻、每次实际 forward 的成员/步号/行范围、完成退出、取消退出和新请求补位。至少一个真实 GPU forward 含多个独立请求，且有运行期间的加入/退出；将请求排队串行执行、单次固定 batch 或拼接结果不算。

### 5.6 作业和媒体输出

服务提供异步作业的提交、查询、取消和产物读取；状态包含阶段和实际进度。进度事件不是可播放视频帧，不据此计算首帧延迟。`run` 等待同一 pipeline 并写产物，不经过本地 HTTP 绕行。

媒体输入使用有界解码和明确重采样规则，保持参考顺序、时间戳和音画对齐。输出编码通过明确的 codec 接口；如采用外部 ffmpeg，作为显式产品依赖检查并限制进程/缓冲生命周期，不能隐含 Python 运行时。产物发布原子化，失败/取消清理临时输出，读取者持有的产物不能在任务结束时提前删除。

独立 artifact owner 持有 encoder staging、临时文件额度和 reader lease。进入不可逆生成前必须获得可有界消费/落盘的输出计划；先取得 decoder 所需输出额度，再提交对应解码。慢下载可以阻塞新产物准入，不能持有 DiT forward grant 或让 VAE 与输出 gate 循环等待。codec 只依赖低层帧/音频描述，不反向引用模型实现。

## 6. 分阶段交付与退出条件

| 阶段 | 交付 | 必须取得的证据 |
|---|---|---|
| G0 基线与设计 | 固定资料、正式目标、FastVideo Rust 客户端、恢复记录 | 原模型/配置/依赖身份，当前基线完整产物和实际容量 |
| G1 合同与来源 | 真实模型元数据/PDD/时空布局、NVFP4 来源 | Rust 正确性/拒绝测试，实际 checkpoint 读取与独立数值 oracle |
| G2 执行核心 | 共用迭代协议迁移，NVFP4/attention provider，受管资源 | 图像回归；真实 SM120 算子对齐、容量/失败/退休测试 |
| G3 完整短片 | conditioning→联合八步→36L视频/音频解码→编码 | paired 中间结果和完整短片；run/serve 共用链路 |
| G4 连续批处理 | 同规格动态入退批，再扩展合法混合规格 | 真实共享 forward/混进度/补位/取消；单独执行与批执行正确性 |
| G5 产品与性能 | 参考模态、长片容量、可用API、端到端优化 | 固定主表胜过 FastVideo且达SLO，完整回归，限制与恢复说明 |

每阶段的小切片须可运行并有实际消费者；不能先铺大量未使用 trait/配置、把关键执行留成 `unsupported` 后宣称支持。性能优化围绕测得的关键路径，先用短而充分的 paired 试验筛选，再在稳定候选执行完整验证。失败和回退保留记录。

当前媒体输入切片使用 `VideoGenerationRequest.references` 的有序 typed 媒体列表；模型能力分别声明图片、视频、音频及总数，编码字节始终属于原 Host 账户。声音时长进入 conditioning 缓存键；容量预估覆盖原波形/补零、CNN 输出/投影转置、posterior/归一化和派生权重的真实重叠。输入裁剪在样本缓冲分配之前完成，声音 CNN 关闭后才安装投影；进入后续去噪的仍是原 continuation 和唯一 worker。

本次产品候选的 CLI 为 `ferrum run --video-config CONFIG --reference audio=REFERENCE.wav --reference image=REFERENCE.png --prompt PROMPT --output CLIP.mp4`；重复 `--reference` 保留顺序；视频参考候选增加 `--reference video=REFERENCE.mp4`。HTTP 为 `references: [{"type":"audio","data_url":"data:audio/wav;base64,..."},{"type":"image","data_url":"data:image/png;base64,..."}]`。视频容器候选使用 `{"type":"video","data_url":"data:video/mp4;base64,..."}`，同时接受 WebM、QuickTime、Matroska MIME；由共享 worker 的 codec 校验实际内容，HTTP 不读取外部 URL。原 `--image` / `image_reference` 是纯图片入口的兼容写法，不能与新表示混用。当前声音输入为 32kHz 的单/双声道 PCM8/16/24/32 或 IEEE-F32 WAVE，单声道复制到两个独立声道，长声音按请求时长裁剪。独立声音文件的其他采样率/压缩音频，以及参考视频候选，暂未取得产品资格；第 9 节记录实际通过的组合，接口存在不代表验收完成。

## 7. 正确性与质量验收

- hardware-independent Rust 测试：模型 schema 与跨组件一致性、维度/字节溢出、packing/positions、PDD 系数与音视频时钟、量化编码、参考区域不变性、批成员身份和故障状态转换。
- 后端：SM120 上逐项验证 NVFP4 量化/GEMM、VSA prefix 隔离、完整 block、单步 DiT、完整 trajectory、VAE chunk/tile 和音画对齐。CPU 参考或编译通过不替代真实 CUDA 检查。
- 中间参考保留固定输入与数值 profile，误差预算在候选比较前由 reference 重复性和算子合同确定并冻结。出现越界先定位，不能事后放宽以接受性能候选。
- 完整输出检查尺寸、实际帧数/FPS、非空音轨、完整解码、帧内容与时序连续性、参考遵循；有效容器不证明语义正确。使用固定输入的 paired 帧/音频比较和可复查人工画质检查，不以任意单个 SSIM 分数代替模型正确性。
- 固定检查输入必须覆盖图片、短视频、声音、有序混合参考、取消/失败、输出背压和真实容量边界；测量数据不以“成功样本占比”筛掉失败。
- 稳定代码里程碑执行 AGENTS.md 的 fmt/check/test/Clippy 与 Metal 编译兼容检查，以及实际配置的 CUDA CLI/native checks；忽略、失败和未运行项分别记录。

## 8. 性能比较合同

### 8.1 固定主工作负载

首轮主表为既有成功规格：单图参考、832×480、124 帧、24 FPS、8 evaluations、36L decoder、同步音频。固定不可变的请求清单、prompt/reference 内容、seed、输出编码、原模型和量化/attention profile；首轮可以采用受控小样本，但明确是工程对照而非代表性用户数据集。

FastVideo 与 Ferrum 使用同一 5090，串行独占 GPU；主表预先指定 C1/C2，C4 单列为容量/调度诊断。服务容量/预算在客户端 sweep 中固定。若后续改变服务容量，作为另一次实验，不混入原主表。两侧都保留合法缓存与卸载策略，记录差异；不得故意关闭基线默认加速。

在首次候选比较前固定五秒档 SLO：**客户端提交至完整产物接收的 P99 ≤120 秒，所有已提交测量请求成功率 100%，无未收敛任务**。吞吐只计已完整接收且验证有效的产物，失败、超时、拒绝和未完成仍在分母。该阈值是本任务的初始验收目标，不是已有达标结论。十/十五秒及768p单独预先声明SLO后才比较，不能混成同一延迟分布。

故意取消/容量拒绝的正确性负载单独运行，不混入上述完整完成负载。超时后停止后续计时单元直到确认原任务已退休；FastVideo 的 DELETE 仅删除作业记录，不能证明在途 CUDA 已停止。

### 8.2 测量口径与胜出条件

- 胜出主表固定为暖进程、新参考输入；预热使用清单外输入，每个重复的测量参考内容不命中既有结果缓存。首次启动/首请求与相同参考缓存命中另报。请求顺序在两侧和各并发格一致，不能仅更换文件路径冒充新参考。
- 每格报告端到端 P50/P99、有效 clips/min 与有效输出视频秒/墙钟秒、队列/conditioning/DiT/VAE/编码时间、Peak VRAM/RSS、错误、样本数和重复数。GPU采样峰值是采样下界；配置cap和框架allocated不充当全设备峰值。
- 主延迟包含最终产物传输；终态被轮询观察到的时间另记，披露轮询粒度。编码/下载是产品路径的一部分。验证解码放在计时窗外并保持所有产物可追溯。
- 主目标同时要求 C1 每轮端到端中位延迟降低、C2 有效吞吐提高，均满足预声明SLO；C2须有真实联合forward证据。冻结 paired 重复的统计方法：以实验重复为单位计算 Ferrum/FastVideo 比值和95%置信区间，C1区间整体低于1，C2整体高于1才建立胜出；完整报告全部预声明单元，不能挑选最佳并发格。
- 短试验只用于筛选及估计方差，不进入正式胜出数据。正式A/B开始前将重复数、每轮请求数、AB/BA顺序、区间估计方法与停止规则写入不可变实验清单。结果不明确时报告未证明；追加样本必须先声明新实验，不能反复查看结果直到显著。小样本P99明确标为经验分位，不冒称稳定生产尾延迟。不得用固定PASS文字、任意重复总数或通过比例代替证据。
- 本目标的视频主表不套文字 TTFT/TPOT/SSE ITL。已有文字性能/正确性回归继续遵守仓库规定，不能因视频测试通过而省略受影响路径。


视频输入的可选 `input` 配置包含 `ffmpeg`、`ffprobe`、`directory`、`threads`（默认2）、`timeout_seconds`（默认120）。省略路径时，FFmpeg 继承 `output.ffmpeg`，ffprobe 为同目录可执行文件，临时目录为 `output.directory/reference-input`。输入暂存文件保留至子进程回收后删除；默认探测16秒、至多8192源帧，模型选取前15秒源片段并裁剪到请求时长。`source.files.video_processor` 可显式指定原 `video_preprocessor_config.json`，省略时取 image processor 同目录的该文件。

## 9. 当前进度与下一步

2026-11-17 发布候选补充：视频与文本改动已合入主线，v0.20.0 冻结源码为 `a08e735d3d1da090bb858fecffb54868513a9836`，包含独立 Cargo 包的源码材料归属修复。此前 `a50b9bff5` 候选被替换，其原始验收和失败记录保留，不能替代新包验证。新 Linux SM120 压缩包解出的二进制 SHA-256 为 `f4988a7fd24f466d3a9927a0221a82db78037f5e70910f6c7d8df703e72cc506`，已通过 35 项文本功能案例、显式准备选项的 6 项案例，以及 CLI 视频生成。CLI 输出为 832×480、124 帧、24 FPS，包含 AAC 音频，完整音视频解码通过。

新包 HTTP r1 的 C1/C2 共 6 个任务全部生成成功（4 次计时、2 次预热），媒体完整验证通过、零请求错误；各格仅一次功能重复，经验 P99 为 47.10/97.13 秒，均低于预声明 120 秒门槛。但 C2 有一次 `VmRSS is unavailable` 采样错误，因此 r1 基准退出码为 1，不能记为完整资源验收通过。仅针对 C2 的 r2 复核保持同一配置、输入、200 ms 采样间隔和 2 秒查询期限，1 次预热及 2 次计时均通过，退出码为 0，采样完整；计时 P50/P99 为 95.28/96.89 秒、有效吞吐为 1.238 视频/分钟，峰值 VRAM 为 29,584,523,264 B、计时 RSS 峰值为 33,721,020,416 B（含预热的峰值为 34,064,781,312 B）。r1 和 r2 分别保留，不拼接为新的配对性能结论。报告、媒体和 CLI 记录保存在库外 `/Users/chejinxuan/ferrum-handoffs/20261116-release-v0200` 及 5090 的 `candidate-a08e735d3` 目录；正式分发与公开安装验证属于独立发布检查，本段不替代这些检查。

2026-11-16：**G0–G5 视频阶段验收完成，接续第 10 节 Windows 反馈。** `run`/`serve` 共用原生 FastH3 管线，图片与视频复用通用迭代 worker；真实连续入退批、混合进度、取消和补位已验证。受支持已接纳请求在容量压力下缩批或等待后完成；长参考的 362 帧 CLI/HTTP 与 243 帧 C2 资格均完成。源码位于本文件第 2 节的独立工作树，使用入口和固定 5090 配置见 [视频指南](video-generation.md)。以下历史记录保留失败与修复过程，以本段和本段所指证据作为最终状态。

正式主表 `video-fasth3-final29-r2` 使用同机、同 checkpoint/质量配置、两侧 29 GiB、C1/C2 三个配对重复，24/24 计时请求有效。C1 延迟配对比值 0.676750，95%CI `[0.651494, 0.702986]`；C2 有效吞吐比值 1.437047，95%CI `[1.366575, 1.511152]`。Ferrum 六格经验 P99 全部 ≤120 秒，最大 115.018090 秒；满足预声明主表门槛。后续原生分配容量修复的产品二进制 `77b4148b…87b9d` 已通过真实 CUDA 容量恢复及 CLI/HTTP 回归，输出身份与正式计时版本一致。正式计时版本身份、完整源清单、原始报告和条件差异均保留，不混算后续功能回归的计时。

最终冻结 `video-final-stable-r5`：Linux workspace check/test/Clippy、常规 CUDA CLI 和 `cuda-fast-h3` CLI 编译全部退出0；完整 CPU suite 8541 通过、0 失败、97 忽略，包含 348 个测试目标的结果。Mac 全目标 Metal 编译及 Rust fmt 通过；编译兼容不等于 Metal 视频运行资格。两项受影响的 SM120 注意力与 B1/B2 混进度 CUDA 用例各通过一次，最大 BF16 ULP=0；没有放宽数值门槛。最后 6 项失败均来自旧测试对优化前算子注册、行映射、路由分块或生命周期布局的假设，已用物理绑定、逐行语义、实际资源占用和独立数值结果修正。R4 的生命周期对照仍错误地排除已读完输入副本的合法复用，定向失败保留；R5 修正后该用例及全套通过。最新源码变化仅在测试和文档，不需要重跑已完成的产品与正式性能实验。

验收边界：上述是固定短片工程工作负载，不能推定生产尾延迟或所有参考组合；C2 未提高 Ferrum 自身 C1 吞吐。长参考 15 秒和 C2 完成但超过其 600 秒目标，不记作长视频性能通过；768p 仅有一项 124 帧图片参考资格。GPU 采样存在缺口，显存峰值仅为观测下界；完整音轨解码已验证，没有人工听音结论。Rust fmt、源码/文档链接和自有文件 diff 检查通过；完整 staged diff 的 whitespace 检查仍报告两份已资格化 Sage 来源头文件 `params.h`、`quantize_core.cuh` 的行尾空格，保留其构建字节和源哈希，不将该检查记为全通过。视频目标完成不代表第 10 节队列完成。

2026-11-16 00:10（Asia/Shanghai）：SSH 已恢复，原生容量修复的新产品 `77b4148b…87b9d` 已完成 HTTP 原图 C1、新图 C2 和同输入 CLI 回归，4 份音视频均完整有效，MP4 SHA 与此前资格一致。CLI 请求 65.922802 秒，HTTP 冷请求 73.271320 秒，新图 C2 为 114.322956/110.794757 秒，记录了 8 次真实 B2 prediction handoff；driver、服务及 CLI 均 exit0。HTTP 测量客户端因 GPU 采样缺口返回 exit1，请求和媒体验证本身无错误；采样峰值仍仅为下界，不记为完整资源测量成功。完整原始性能包已下载并通过 SHA 校验，Rust 对完整报告重算得到相同配对比值和区间；FastVideo 原启动配置已确认恢复。

最终 Linux 检查的 workspace check 已通过；首次测试遇到远端磁盘空间耗尽，释放的仅是本任务旧二进制缓存及 Cargo 已被替代的测试可执行文件，原始证据和当前构建缓存保留。同一 bench-core 测试二进制重跑 450 通过、1 忽略。续跑发现 5 个模型测试的算子注册/图断言没有跟上已资格化的优化：修复测试注册表，校验 K/V 共享物理行映射及逐行调制路由，不修改推理算术或放宽数值门槛。冻结 `video-final-stable-r3` 仅增加 5 个测试文件修改和文档更新，模型库定向回归 1170 通过、0 失败、14 忽略（25.70 秒），Mac 全目标 Metal 编译通过（23.24 秒），fmt/diff 通过。两项受影响 CUDA 数值测试和剩余最终检查正在执行；视频尚未标记验收完成，第 10 节仍按顺序接续。

2026-11-15 22:46（Asia/Shanghai）：最终候选的 Mac `cargo check --locked --offline -j2 --profile release-tools --workspace --all-targets --features metal` 退出0（1分39秒），fmt/diff检查通过；这是全部目标的编译兼容证据，不是Metal视频运行资格。源冻结 `video-final-stable-r2` 已准备，包含最新原生容量修复，待新二进制产品回归成功后执行一次Linux workspace check/test/Clippy及两组CUDA CLI编译检查。未重复正式性能实验。

`video-native-allocation-product-r1` 已在5090启动，但当前尚未取回终态。原Windows跳板的SSH banner连续超时，Mac直连Windows SSH、WSL SSH及局域网页面TCP探测也超时；既有备用Mac管理会话已结束。冻结源上传未确认成功，不能记为远端检查已开始。已保存本机最终检查记录 `video-final-stable-r2.local-qualification.json`；恢复连接后先核查现有回归容器及产物，避免重复提交。正式24请求性能对照与CUDA原生容量测试的已完成证据不受影响，视频验收和第10节后续工作仍未标记完成。

2026-11-15 22:34（Asia/Shanghai）：完整计时单元已通过Rust `video_job_compare` 复核。C1 P50配对比值几何均值0.676750，95%CI `[0.651494,0.702986]`；C2吞吐比值1.437047，95%CI `[1.366575,1.511152]`。每轮C1中位耗时均更低，六格Ferrum经验P99均满足120秒，24/24计时请求有效；本固定5秒480p工程工作负载的预声明性能门槛达标。下载的18,690B分析包保留全部原始cell及每份原报告SHA，省略的输入payload仍保存在5090原报告；Rust结果、门槛核查和逐格资源数据为库外 `video-fasth3-final29-r2.{comparison,acceptance,timed-cells}.json`。各格GPU资源采样均不完整，RSS有明显跨次波动，不能宣称完整显存峰值或更省内存。

原生容量修复的5090 CUDA测试2/2通过（1.68秒），确实到达受限私有pool的原生分配OOM，随后原B1与释放后的B2都完成并逐字节校验。CPU第二块分配失败测试另证先前分配/额度全部回滚；GPU回滚收据为0，不能混称GPU已发布多块后的回收。新产品构建exit0（3分29秒），SHA `77b4148bc9dd4a40e92cc35765797b6b5de133926772b3156b4d3b4a01687b9d`；当前定向Core、Reference、Metal CLI编译、fmt/diff均通过。`video-native-allocation-product-r1` 为独立功能回归：HTTP原图C1、新图C2，再同输入CLI，保持原质量/预算；其计时不混入正式对照。产品回归、最终稳定版本检查与第10节后续工作仍待完成。SSH备用管理会话曾完成上传与统计下载，现已结束，继续沿原Windows路径操作；部署资料保留两条路径和原配置。

2026-11-15 22:10（Asia/Shanghai）：`video-fasth3-final29-r2` 全部12个计时单元与12个预热单元完成，orchestration exit0，24条计时请求全部有效。C1三次P50：Ferrum 55.366461/55.880548/55.017292秒，FastVideo 80.385232/83.178989/82.135532秒；C2三次有效吞吐：Ferrum 1.049902/1.047857/1.043017 clips/min，FastVideo 0.713976/0.740222/0.731618。Ferrum六格经验P99全部≤120秒，最大115.018090；FastVideo三格C2经验P99为167.326541/161.384602/163.279820，保留其未达标结果。Ferrum全部原fresh产物SHA与旧资格一致；两侧GPU采样均有缺口，不能作完整峰值或显存效率主张。原始完整报告及服务日志已打包在5090的 `/tmp/video-fasth3-final29-r2.reports.tar.gz`（20,138,933B，SHA `05b0bee523d868e5380486abe8405bf84e387bd9c2bf35c8966ea62fcd7e6cca`）；配对Rust统计仍待取回完整单元数据后执行。

同轮容量审查发现一项实际遗漏：Core动态backing把原生分配错误转为普通文本，无法可靠区分安全容量回退与执行故障。新增通用 `DeviceAllocationFailureKind` 和 `DeviceAllocationPressure`，仅对后端确认的零产出分配容量失败保留typed原因；CUDA同步/回收错误、非法访问和仅有retryable标记的错误均不获回退资格。候选仍须经原prepare owner回滚后才能缩批。`video-native-allocation-pressure-r1` Core第二块分配失败/零部分发布/额度回收/再执行测试1项通过，Reference B2拒绝→原B1→释放→B2测试1项通过；实际CUDA用例以私有pool的64MiB原生上限和充足Core预算触发真正分配OOM，避免占满整机，目前尚未执行。12文件冻结包SHA `55f74608d0b8d125339e72d03cd76f84eeb3ff6f5eb3a44c64de70875553e832`，CPU日志与scope在库外。正式计时源保持不变，新候选不能冒用旧binary身份。SSH间歇成功但上传反复断开，正在恢复连接；原定最终整仓检查推迟到此修复稳定后只执行一次。视频目标与第10节后续队列尚未完成。

2026-11-15 21:02（Asia/Shanghai）：`video-long-capacity-product-r11` 已全部结束，driver与服务exit0、无容器OOM，4/4请求完整音视频通过。图片控制68.376337秒；长参考362帧HTTP635.536278秒、SHA与同源CLI完全一致（`6600e4d8…fa13118`）；长参考243帧C2两请求421.464989/868.260970秒，seed1201与先前CLI也相同。全程累计32个B1 forward，末态executed_requests=4，active/pending/preparing=0、rejected=0；此处证明等待后串行完成，不声称长参考B2。长15和C2经验P99均超过600秒，保留未达标。结果 `video-long-capacity-product-r11.results-final.json`；旧模板中若干描述性provenance字段未刷新，原始报告不改，真实请求/配置/二进制及纠正列表记于 `.identity-amendment.json`。当前源正式对照 `video-fasth3-final29-r2` 已启动：原固定5秒图片参考、29GiB、C1/C2、3个配对重复，24条计时+12条预热，原AB/BA顺序；当前产品SHA b365e9b6…23db47，新Rust客户端a57cfc68…104a6，1000ms资源周期/10000ms查询截止，失败/漏采全部保留。主胜出要求C1 P50配对比值95%CI上界<1、C2吞吐下界>1、每格P99≤120秒且全完成；不混入旧数据。本轮正式比较尚待结果。已增加基于实测显式配置的 `fast-h3-video-cuda.json` 与使用文档草稿，尚未完成最终整体验证。

2026-11-15 20:11（Asia/Shanghai）：`video-consumer-chains-r1` 原15秒视频来源→362帧请求首次完整成功，11:51:04→12:03:11 UTC进程exit0、无容器OOM；请求704.833365秒。原29GiB Device/4GiB Host、32768节点/命令预算、NVFP4、8次评估与36层解码器均保持。完整音视频解码exit0；ffprobe确认832×480、24FPS、362帧、32kHz双声道AAC，9,725,689B；MP4 SHA `6600e4d8d5b641cedbe5a6673d4ce51288e576a975b3cbeac1be2bf73fa13118`。每两秒抽帧可见人物/湖岸场景和连续镜头变化，无明显空白或损坏；这不是完整语义或与FastVideo配对质量证明。产物资格与终态位于库外 `video-consumer-chains-r1.results-final.json`、`.probe.json`、`.decode-exit`、`.contact.jpg`。本轮仍仅为功能/容量证据，705秒超过长参考预声明600秒门槛，且有CPU编译重叠，不能记性能通过。已提交同一二进制HTTP `video-long-capacity-product-r11`：图片控制→长参考362帧→长参考243帧C2；等待各格终态及同源CLI/HTTP校验。后续仍需最新源FastVideo配对比较/真实显存采样、默认配置与文档、稳定里程碑回归，再处理第10节；整体目标未完成。

2026-11-15 19:59（Asia/Shanghai）：`video-consumer-chains-r1` Reference调度5/5通过，CUDA产品构建exit0（2分35秒），二进制SHA `b365e9b6b99958dd330df8d354f246a5ffa02749193e6ffc1a5a21f64a23db47`。原362帧请求11:51:04 UTC启动；11:57:28记录ConsumerChains、4096行计划通过原29GiB预算，新增需求（含初始化stream）28,794,987,248B，比原方案减少397,623,296B；加原生存活2,298,478,592B后尚余45,047,056B。这是编译/账本用量，不是全设备实测峰值。11:57:34实际预取到5个原Core动态backing块、共16,848,977,632B，开始去噪阶段；八步、解码与最终产物仍待完成。没有提高预算、降质或裁减参考。资源探针CLI补测已在真正的`--lib`目标7/7通过（0.18秒），证据 `video-gpu-probe-cli-lib-r2.results.json`；旧`--bin`0项记录保留。本轮CLI功能运行期间有CPU编译，不能作比较性能。HTTP接续r11仅准备，仍以本请求成功为启动条件。

2026-11-15 19:46（Asia/Shanghai）：`video-depth-first-r1` Reference调度5/5通过，CUDA产品构建exit0（2分31秒），二进制SHA `cabd3cef31f28e0d8d6582b800e995422ae02fea2944f059c96a717d617f5219`；原362帧请求11:28:46→11:35:42 UTC exit1，无OOM/产物。全局深度优先把U8收费降到2,224,373,760B，但BF16收费升至19,409,143,808B，单池即超剩余动态预算，因此弃用；原最小计划及352,576,240B缺口未变。终态为 `video-depth-first-r1.results-final.json`，HTTP r10未启动。该候选已从工作树移除，改为通用ConsumerChains候选：只连续完成单消费者链及连接处已就绪的依赖，共享边界继续按提前压缩优先级，最后延后扩张；原三候选保留。三文件冻结 `video-consumer-chains-r1` 覆盖SHA `435f0c3a7e7d74a2fd3d66d20662dfaedcc41506297f3102c56da00432584ea3`，父源为下面的probe-owner-r1；本候选测试/完整重放待结果。

同日：已修复图片/视频两个基准客户端的GPU探针退出等待，共用 `ferrum-bench-core::resource_usage::GpuMemorySampler`。查询超时后仅请求终止，保留原Child并用try_wait回收；旧进程未退出时跳过新探针，避免无限wait及进程累积；采样缺失仍使资源报告不完整。Mac生命周期2项和子进程fixture、视频窗口4项通过；Linux共用资源13项（含fixture）与视频窗口4项通过。六文件 `video-gpu-probe-owner-r1` 覆盖SHA `ee93846f76dca14a0d18328c337a4e86b5e26e80a9df458e451ed216561e8a99`，新Rust视频客户端SHA `a57cfc684651be51934fb7265850edfe96d4c073e385026e498d0d82c14104a6`。CLI `--bin ferrum`测试目标仅编译通过、实际0项，不能算入口回归；真正的 `--lib` 资源窗口测试待补。尚未用新探针建立实际5090峰值/性能证据。

2026-11-15 19:23（Asia/Shanghai）：`video-upload-reuse-r1` CUDA数值/所有权资格1/1通过（1.76秒），模型调度Reference组4/4通过（0.01秒），产品构建exit0（3分27秒），二进制SHA `1f38cff73bc9f53ca72673f0262e83e18351a1544fcee3f8fb63334130fa722c`。原362帧请求11:04:45→11:12:27 UTC exit1，无OOM、无产物；中途SSH暂时失联后恢复，未重启或取消容器。最低计划从29,358,474,480降至29,175,833,328B（减少182,641,152B），加原生存活量与初始化stream仍缺352,576,240B，容量问题未解决。终态为 `video-upload-reuse-r1.results-final.json`；HTTP r9未启动。逐slot诊断进一步表明：U8计划3,098,466,560B、顺序有效载荷峰值2,700,843,264B，额外收费来自多个同时存活的FFN分块量化缓冲，单纯提前压缩会破坏分块连续处理。已移除无收益的LifetimeAwareCompaction候选，冻结三文件 `video-depth-first-r1`（父源upload-reuse-r1，覆盖SHA `ca260ef6c759a675d0574fc8bf9161fcb7dc5a78be751bd959cd9205ceaf53ae`）：通用固定无状态图按输出依赖深度优先排列，显式栈避免深图递归，所有原节点、未消费工作、算术、输入和权重保留，Core继续验证别名/完成顺序及真实容量，原三种候选仍在。新增多分支实际Reference执行/独立数值oracle与容量边界回归；测试和原请求重放尚待结果。没有调整质量或Host/Device预算，也没有新性能结论。

2026-11-15 19:02（Asia/Shanghai）：`video-lifetime-schedule-r1` 原362帧请求10:39:34→10:46:40 UTC exit1，无OOM/产物。新调度虽把U8收费降至2,644,017,152B，却延长了大BF16输入寿命：BF16收费升至18,721,699,840B，最低计划34,894,666,736B，自动弃用，最终仍选择父方案29,358,474,480B并遇到原容量不足。终态 `video-lifetime-schedule-r1.results-final.json`。已冻结5文件 `video-upload-reuse-r1`（父源lifetime-schedule-r1、覆盖SHA `58b3db7f08d83d8bcc9ea9309f2c07efe7d278ead041366d385d44f5774b290e`）：调度收窄为只延后严格压缩，且每个输入必须本来就有不早于目的位置的其他消费者；Core仅允许已证明串行的固定张量波次复用只读Step输入副本，输入寿命从波次开始，最后读取完成后才能覆盖，保留输出、晚读、别名写入、token/混合或非串行工作仍受保护。原不可变Host输入始终保留。核心生命周期7/7、slot/对齐9/9通过（另1项CPU profiling测试未运行），真实Reference执行1/1通过（0.19秒），覆盖数据依赖/串行两种计划、重复请求和Host输入不变；fmt/diff通过，未跑全仓。CUDA同组数值/所有权测试、产品构建和原请求重放正在接续。模型调度小组在CLI功能运行期间复核，不与性能测量混用。新候选实际用量与最终产物仍待结果。

2026-11-15 18:43（Asia/Shanghai）：`video-lifetime-schedule-r1` 共享调度定向测试4/4通过（Reference实际输出、容量边界、关闭后资源清零、checkpoint角色保护；测试执行0.01秒，编译3分18秒）。CUDA产品构建exit0（2分32秒），二进制SHA `70f04601b1a8fef22cddca3118ef6dc18b0af67bd32c1bdeab347c1b149ee4b2`；原362帧请求正在运行，尚未取得新计划用量或完整视频。HTTP接续验收`video-long-capacity-product-r8`仅在本机准备，CLI成功前不启动；包括单图控制、原长参考15秒、长参考10秒C2，保持原预算和质量。为避免已发现的GPU资源查询清理挂起，本次仅作功能验收并明确禁用GPU采样，正式资源/性能资格仍待补齐。

2026-11-15 18:33（Asia/Shanghai）：只读诊断 `video-step-census-r1` 构建exit0（3分32秒），二进制SHA `123fcfe15b2d224b68d68e56b0b44c14c9a8fe49123c5772ab0cdfd4af06ded8`；原请求10:19:33→10:26:28 UTC exit1，无OOM/产物，容量结果与父候选相同。首次拿到选中EarlyCompaction的逐节点存活量：BF16收费12,731,058,176B、顺序存活峰值12,694,921,216B，峰值在`vsa.key_center.layer0`，原K/中心化K/V各2,699,296,768B、video-Q2,510,290,944B、残差1,694,106,624B并存；所以主因是活跃张量重叠，非大幅slot碎片。U8收费3,098,466,560B、顺序峰值2,700,843,264B，含提前存活的gate量化缓冲423,526,656B；F32收费与顺序峰值同为1,067,237,312B，峰值是softmax输入/输出各484,700,384B并存。顺序census是诊断，不代替Core完成顺序/别名安全证明。证据 `video-step-census-r1.results-final.json`、`video-step-census-r1.long15-census.json`。据此新增通用`LifetimeAwareCompaction`候选：只有内部输入的最后消费者才能计入退休量，延后不能释放原输入的压缩，保留净零扩张的压缩位置；已有候选全部保留，仅以完整编译容量选择。冻结5文件`video-lifetime-schedule-r1`，父源census-r1、覆盖SHA `ce557ef25f0ac0a14d13f39373ed66e542624fcd3625ad2b3471f37fd1c50775`，含此前两处测试修正。定向Reference生命周期/容量测试和CUDA产品重放待结果；没有新算子、精度/预算变更或性能结论。

2026-11-15 18:14（Asia/Shanghai）：`video-indexed-fp8-r1` 产品构建exit0（3分29秒），二进制SHA `7074d22ba94419276f1c6ecb506709db6789b451f862010f000f5b7c5cd25118`。原362帧请求09:59:51→10:06:17 UTC exit1，无容器OOM、无产物。虽然完整计划减少50节点至16555，最低计划仍29,358,474,480B，与父候选完全相同；加原生存活容量仍缺535,217,392B，不能将被删张量体积称为峰值改善。2048行最低29,395,862,768B，也无改善；原backing全部回滚。终态证据 `video-indexed-fp8-r1.results-final.json`。下一步在共享Core增加只读的逐节点同时存活量与实际收费slot诊断，检查被静态规划接受但被真实存活容量拒绝的候选，定位峰值后再选择修改；预算、精度和输入保持原样。另外，旧保留输出复用改变了7个fixed liveness测试的容量预期，首轮1通过/6失败；修订为仍禁止外部输入/存活别名复用、允许已死中间值供终态输出使用，并保留真实容量边界后7/7通过（`video-retained-slot-existing-liveness-r1/r2.log`）。这只是旧优化的测试修正，不是新的运行时优化；尚未进行全仓检查或最新候选HTTP/性能验收。

2026-11-15 17:59（Asia/Shanghai）：冻结19文件 `video-indexed-fp8-r1`，父源 `video-phase-capacity-r1`、覆盖SHA `15c6f8546160da7190ec8161848774dfccb84fc98639a049d457c55f16fe2cca`。通用indexed head FP8算子将原始I32行表与完整head内amax/div.full/E4M3FN/BF16舍入组合，保持原精度和无别名合同，直接产出有效K行；Reference/CUDA共同保留原整行操作，未引入隐藏scratch。CPU实际数值/所有权测试1项、模型完整注意力/映射容量测试2项通过；5090原整行+indexed数值/所有权资格1项通过（1.87秒），覆盖重复/无效索引、跨batch、head宽度及grid-stride边界。最初CPU固定数组编译错误已修复并保留失败日志；冻结后仅修正模型测试ID的unwrap，生产源与远端冻结全部一致，区别见 `video-indexed-fp8-r1.local-qualification.json`。fmt/diff通过；没有跑全仓测试。产品构建/原362帧请求重放进行中，尚无本候选峰值或完整视频结论。

2026-11-15 17:54（Asia/Shanghai）：`video-phase-capacity-r1` CUDA增量构建exit0（2分00秒），二进制SHA `195caaee3d79b7c7d17afcee0a2797ed8c7a99e062f2e1cecdcce9ca5914008a`；Host阶段信用回收生命周期测试1/1通过。原362帧seed1301请求09:35:53→09:42:36 UTC exit1、无OOM、无产物。Host自等待已消除，推进至真实Step backing分配；工作区准入不足并按原owner回滚16,896,762,048B，最终仍为initial denoising capacity错误，尚未满足容量验收。4096行EarlyCompaction最低29,358,474,480B；2048行29,905节点、最低29,395,862,768B，未降低峰值；1024行触及节点上限，保留原最小有效计划。终态证据 `video-phase-capacity-r1.results-final.json`。下一候选正在实现通用indexed head FP8 roundtrip，保留原head内算术并直接按I32行表输出有效K，消除完整padded quantized-K中间值；未提高预算或降低质量。CPU/CUDA数值、生命周期和原长片重放尚待结果，不能把张量体积当作实际峰值降幅。

2026-11-15 17:34（Asia/Shanghai）：`video-rounded-projection-r2` CUDA产品构建exit0，二进制SHA `701bda0f26aba49e9b23b78efd2c8c9aea0e332558f9a58bfa7733218eb9c2f0`。同一362帧请求的最低计划由31,982,091,280B降至29,358,474,480B（27.34GiB），Invocation峰值仍532,480,032B；完整图16605节点。额外真实准入量29,375,251,696B，加原生已占2,298,478,592B仍比29GiB多535,217,392B，不能称设备准入成功。随后发现Host自等待：已承诺3,130,059,485B，B1又需2,076,454,854B，超过4GiB共911,547,043B；条件编码的Input预留已不再使用却由请求继续持有。09:20:24 UTC后两次观察均停在同一Host准入边界，主动SIGINT取消，09:24:19 exit1、无OOM，错误为取消而非新崩溃，未生成产物；终态证据 `video-rounded-projection-r2.results-final.json`。已冻结6文件 `video-phase-capacity-r1`（覆盖SHA `a53583b9e5a0ce2a1f204aff9c941183ba1077bea7a585a1e05255a9e5caa871`）：在已完成的conditioning边界释放过时Input信用，保留后续真实输入上界、其他类预留及存活数据；FFN在原配置最大行数内依次尝试更小完整计划，节点上限仍有效。定向Host生命周期测试及新产品构建/原请求重放进行中。配置-only2048诊断已准备但没有启动，避免重复执行必然遇到的Host等待。15秒、最新候选性能与最终容量验收仍未完成。

2026-11-15 17:14（Asia/Shanghai）：新增通用 `affine_linear.f32_to_bf16` 合同及 FastH3 media consumer：保持原F32输入/权重、完整K轴、F32 dot-store→F32 bias→BF16 RNE；CUDA按4096行流式复用Core计费的Invocation scratch，不再声明完整F32投影为Step值。分块可能改变SGEMM归约树，不声称与全行GEMM逐位等价。模型3项、Reference6项、Core2项通过，含小精度差、偏置前提前舍入的反例、全行/尾部、输出保留和真实容量拒绝。5090新多分块及5376宽度数值资格、原F32 affine/rank5 patch回归共2项通过（2.35秒）。冻结18文件 `video-rounded-projection-r2` 覆盖SHA `025a533569d1f5b9552512834083e13eff71022695096a303040d5e0de702f45`，父源为 `video-early-compaction-r2`；r1仅准备/上传，未运行。r2额外修正Reference操作名/工作量/dispatch计数，CPU实际算术证据早于这项观测修订，最终稳定检查仍待完成。产品增量构建及原362帧seed1301重放进行中，保持原29GiB/4GiB与质量、显式32768节点/命令预算；尚无本候选完整视频或性能结论。

2026-11-15 16:58（Asia/Shanghai）：`video-early-compaction-r2` CUDA构建exit0，二进制SHA `6c0be2ed67bbb0c96cf246809c95bf11b6e4b1362af11b5836e82515b44f6a3c`；父候选为已通过真实GPU保留输出复用测试的 `video-retained-slot-r1`，新增3文件覆盖SHA `3aacf299090d8d62ac93615e6e02fb9b8bb2e330ed23145e1600a595afa7dadd`。同一362帧seed1301、原29GiB Device/4GiB Host及质量配置，CLI 08:37:59–08:43:39 UTC exit1、无容器OOM，仍未生成15秒视频。完整候选比较的最低计划降至31,982,091,280B（29.79GiB），比修复前35,921,420,304B减少3,939,329,024B；其中BF16池12,731,058,176B，F32池3,690,854,112B。仍超过31,138,512,896B预算，且原生运行时已占2,298,478,592B，不能将计划差额当作全部待释放容量。下一目标是F32输入投影3,108,317,184B中间结果的有界执行。共享Core9项、checkpoint8项、请求状态7项、Reference实际输出1项通过；调度3项最终通过，首轮因保留输出优化消除了旧fixture的严格容量差而2项断言失败，修订为真实容量边界与全数值验证，失败记录保留。证据 `video-early-compaction-r2.results-final.json` 与各定向测试日志在库外，未运行全仓测试；最新二进制尚无完整生成或性能资格。

2026-11-15 16:30（Asia/Shanghai）：`video-plan-capacity-r1` 的显式32768节点/命令诊断已结束：362帧完整分块图为16607节点，最低35,921,420,304B（static 11,929,232,400B、step 23,459,707,872B、invocation 532,480,032B），超过原31,138,512,896B设备预算；CLI exit1、无容器OOM。仅提高节点预算不能完成15秒支持。最后容量明细包含BF16池17,025,833,984B，F32池3,733,030,624B（最大3,108,317,184B来自输入投影），终态输出独占1,694,106,624B；证据 `video-plan-capacity-r1.results-final.json`。已修改通用Core生命周期：保留输出可复用已完成的早期值，但之后禁止覆盖；外部输入仍不复用，多个保留值仍互斥。局部9项通过（旧CPU profile测试仍ignored）、Reference完整逐值对照1项通过，新增早期临时值→后期保留输出实际共享与所有先前输出共同读回；5090同一fixture1项通过（1.68秒）。冻结5文件 `video-retained-slot-r1` 源覆盖SHA `6fce3400b37aefeb0f5ec5399c38fb7b12c8d7819f9e3c08ad63e588792911a5`，产品构建尚在完成。另在验证固定无状态图的EarlyCompaction候选：优先就绪的最后使用/压缩操作，再延迟展开，保持全部节点/来源/算术，由Core重新证明别名与完整容量；仅在旧候选放不下时比较。尚未取得该候选完整长视频结果，未提高预算或宣称15秒通过。

2026-11-15 16:09（Asia/Shanghai）：`video-routed-chunks-r1` 两项真实CUDA测试通过：BF16紧凑/展开路由及独立标量对照覆盖68段、119行、32+32+4参数包；融合RMS覆盖67行跨包且不拆归约轴。Core1、Reference2项通过；模型4项在修正旧“33段必拒绝”测试为实际行容量拒绝后通过（测试修订不在r1冻结源，生产代码不变）。产品构建exit0，二进制SHA `ad819988f5218ba14b1a66f4d830b5e8dd089d282a53ec68e8ea6c3737fddef8`。原362帧seed1301 CLI越过旧路由限制，07:52:34–07:57:45 UTC退出1：分块计算图至少16600节点，超过显式16384节点预算；没有容器OOM，也尚无这次完整分块图的设备峰值。证据 `video-routed-chunks-r1.results-final.json`。下一诊断 `video-plan-capacity-r1` 复用同一二进制，仅将已有显式 `maximum_plan_nodes` 与 `cuda.maximum_commands_per_submission` 设为32768，保留29GiB Device、4GiB Host、原输入和质量；用于查清后续真实容量边界，不据此宣称默认产品支持或性能通过。15秒、长参考HTTP并发、最新候选性能复测仍未完成。

2026-11-15 15:48（Asia/Shanghai）：`video-long-capacity-product-r6` 的原图像控制1/1完整有效、67.889255秒，MP4保持原SHA `a6278466a355a0cba2d1eea8484b04d66e926ab74de75b11106a9a8b867ca229`。客户端exit1来自GPU采样不完整：即使配置10秒超时，一次nvidia-smi查询包含清理实际耗时75.747秒；峰值28,922,871,808B仅为已采样下界，不能当完整资源资格。后续362帧/15秒参考在229.137秒后失败于`FastH3 modulation: request exceeds bounded row-routing capacity`，尚未进入去噪容量规划；不是本次OOM证据。客户端停止后正常关闭服务，无容器OOM，C2未启动；完整报告 `video-long-capacity-product-r6.results-final.json`。已将Core路由语义与CUDA每次32段参数包分离：逻辑列表遵循wire计数、完整张量覆盖及模型metadata限制，两个CUDA provider按连续行区间流式分包，原紧凑表不展开、RMS归约轴不拆、真实dispatch计数随包数记录。Core1、Reference2项通过；GPU与模型资格进行中。冻结12文件覆盖SHA `1d0573ccd4df4e4b1f338f6ea402c8b14fefd2c575ef2eb4e30fcf77a29e44bb`，候选 `video-routed-chunks-r1` 先跑原生对照，再按原输入/预算/质量重放362帧seed1301；未声称15秒或整个目标通过。

2026-11-15 15:30（Asia/Shanghai）：长参考CLI首次完整成功。`video-native-handoff-r2` 在原29GiB Device/4GiB Host、NVFP4、8次评估、36层解码器下完成原15秒视频来源→243帧请求（参考仍按原模型规则裁到请求范围），8轮去噪及预测全部退役，无ERROR/容器OOM，CLI exit0。请求433.731129秒、完整进程488.723795秒；这是一次功能资格，不是FastVideo性能比较。原生FFmpeg全音视频解码exit0，ffprobe确认832×480、24FPS、243帧、32kHz双声道AAC，7,110,039B；MP4 SHA `3d5dc217dcc7c538df7e4c683af5bf1a3a7cdb7f7624027777520ef6f988cee6`。二进制SHA `403e640ea21678df433926a89468d1afe90d8c97a197b51cf8d1740076f77081`；结果 `video-native-handoff-r2.results-final.json`。当前最低计划28,723,836,656B，原真实动态backing取得16,795,594,944B/5块，未提高预算或修改数值/质量。已启动同一二进制HTTP `video-long-capacity-product-r6`：先原图像控制，再完整362帧/15秒参考边界，最后243帧C2；各格预声明600秒经验P99功能门槛、完整音视频验证，资源采样改为1000ms/10000ms超时。15秒和HTTP并发结果仍待完成；整体目标未完成。

2026-11-15 15:19（Asia/Shanghai）：`video-native-handoff-r2` 的真实5090交接测试通过（1项、1.65秒）：前一组件关闭、原输出非零偏移切片、InputPreprocess容量占满、下一组件逐字节正确及最终回收均已执行。Mac上的3项Metal managed IO测试也通过，包含两种Host来源的实际上传/读回、无复制持有与错误类/偏移/设备拒绝；Reference 1项通过。测试日志为库外 `video-native-handoff-r2.cuda-test.log`、`video-native-handoff-metal-r2.log`、`video-native-handoff-reference-r2.log`，fmt/diff通过。产品增量构建与同一长参考完整CLI仍待结果；这里只证明交接合同，不宣称完整长视频成功或新版性能胜出。

2026-11-15 15:16（Asia/Shanghai）：`video-host-handoff-r1` 已通过同一长参考的显存准入并完成首轮去噪，原1.21GB的Host重复复制消失；随后CUDA上传层拒绝原Output类输入，CLI exit1、无容器OOM，仍未产生完整视频。二进制SHA `36f8549e8a7fd9133b6412079ee169bf6320bfefe0ff80ebf106955cfb187d7f`，原结果 `video-host-handoff-r1.results-final.json`。现补齐CUDA/Metal的只读Output→Activations上传合同，原Output计费、命令持有、动态绑定阶段与权重/状态限制保持。新增两阶段真实执行fixture：上一组件关闭后读取非零偏移输出切片，在InputPreprocess额度耗尽时仍逐字节正确，最终Host/设备回收；Reference 1项通过。首次fixture因通用close错误类型未实现Debug而编译失败，保留r1记录并修正测试断言；r2的GPU定向测试与产品构建待完成，源覆盖SHA `ecde5da81fad61f4ae5886b67d9c3dc2776487437c1bbda8eb1440b638fac207`（继承r1原生修复覆盖 `1a074c2db48817650bf9a1661c38eb72fa4f80cb6cddc7947c03df8022114c86`）。未提高预算、降低质量或裁短参考；HTTP后续仍须等待完整CLI通过。

2026-11-15 14:58（Asia/Shanghai）：`video-live-capacity-r1` 已跨过真实Device准入：原账本已有2,298,478,592B占用，自动选择合并对齐后的DelayedExpansion，最低计划28,723,836,656B；原动态backing实际取得16,795,594,944B/5块。CUDA构建exit0，二进制SHA `72dd89182c99adc4588acd96109cd11127e2ef0a8e73e2eddb30d603a4ca5687`。首次去噪执行后，在Host预测交接失败：复制完整hidden到请求InputPreprocess申请1,211,465,727B，而该scope仅余844,898,302B；CLI exit1、无容器OOM，完整产物未完成。原日志SHA `58711d3ec6db034a5c8eefed850009656bda51bcd1869e4bc42881f0bb27bdb5`，结果 `video-live-capacity-r1.results-final.json`。修复采用Core显式 `ManagedHostTensor::from_retained_output`：下一组件只读原不可变Output backing，保持完整物理计费与原退款路径，不复制、不重分类，也不把输入视图当作执行/分配权限。模型继续检查真实完成输出、成员行区间及Host父账户。2项Core生命周期/容量测试通过；模型混合成员数值测试与新CUDA候选 `video-host-handoff-r1` 构建进行中，4文件覆盖SHA `a079422f797d980d5c072b3b04d3f6cae84e1be73edf05f3a7b24a34e26465e9`。尚未宣称长参考完整生成或当前版本性能通过。

2026-11-15 14:41（Asia/Shanghai）：`video-alignment-capacity-r1` CUDA构建exit0（二进制 SHA `60c0b13f8909c660652217aa163d2220df4cdbab989c84caae2a9570e62c4bb3`），同一长参考CLI仍exit1、无容器OOM。Core确认BF16的16/256字节对齐可以合并，4096 FFN去噪计划比合并前再省1,665,955,712B，变为29,998,361,712B；但实际Step backing依然触发MaintenanceLimit。完整结果 `video-alignment-capacity-r1.results-final.json`，日志SHA `a692a96d7f4f123dae79eed97d3daafe476eccf6ed568ca5a58762581e77c23b`。本次因分块计划通过编译门槛，尚未选择DelayedExpansion，不能把两项节省简单叠加宣称生成成功。下一候选 `video-live-capacity-r1` 对分块候选同样检查原DeviceCapacityOwner的实际追加额度，不足时比较延迟展开的完整计划；较大或计划容量失败的替代方案保留原有效计划。另记录Core实际维护结果以定位真实差额，未增加预算或绕过准入。2文件覆盖SHA `6bd68dbd1c35e62d0e186527a23e1ed09593443f330726f16f11be8eb54ce2c6`，已提交增量构建及原输入重放；预备HTTP批次均未启动。

2026-11-15 14:30（Asia/Shanghai）：`video-alignment-capacity-r1` 将兼容固定大小 Step activation 的对齐复用放在 Core 内存规划层：按相同 dtype、usage、storage profile 和逻辑布局分组，取最高原始对齐要求，重新计算完整分组的生命周期、保留输出和填充；只有总 slot bytes 严格下降才采用。变量形状、状态与 provider workspace 保留原合同。9项定向测试通过、0失败、1项原有性能剖析测试跳过，覆盖真实容量边界、wire 往返、无依赖不可复用及保留输出；fmt/diff检查通过。冻结6文件覆盖 SHA `9327fad77360a4384ee597def95e7787fc4d1ca7351292eb42c5d307281e18c3`，5090增量构建及同输入重放已启动，尚无本候选长参考生成结果。测试日志为库外 `video-alignment-capacity-cpu-r1.log`。

2026-11-15 14:17（Asia/Shanghai）：`video-delayed-capacity-r1` CUDA构建成功（二进制 SHA `4e3113b94b4feb0df20fdbdeca2fbe167ec01e3cf45f82cad778def3d93210ca`），原长参考的计划最低用量降到30,513,456,752B，较4096 FFN原始计划共减少4,381,420,928B；通过29GiB的静态规划门槛。实际动态 backing 准入仍在8次维护后返回 `MaintenanceLimit`，CLI exit1、无容器OOM，未生成产物；不能将计划通过当作容量完成。完整结果为库外 `video-delayed-capacity-r1.results-final.json`。后补压缩/扩张生命周期 fixture 的2项测试也通过，补充文件晚于该冻结源。下一步需取得原输入的实际分配与完整音视频生成成功，再执行HTTP并发及15秒覆盖；预备HTTP批次尚未启动。

2026-11-15 14:07（Asia/Shanghai）：`video-dependency-capacity-r1` 实机失败，DFS按依赖重排使最低计划从31,664,317,424B增加到32,897,828,336B，CLI exit1、无容器OOM；完整记录为库外 `video-dependency-capacity-r1.results-final.json`。小图通过不能代替真实模型容量结果，该候选未采用，HTTP后续验证未启动。现已替换为 `DelayedExpansion`：按原图逆序，把输出激活总量不小于输入的节点推迟到首个消费者之前，量化/缩减操作保留原位置；仍保留所有节点和输入，Core继续检查MustAlias最终消费者等约束。12项共享组合回归通过，冻结3文件覆盖SHA `0e0c85958e629157f6dd99e964a109ec1071a9a5d8240843a663b0349e99e4c8`，候选 `video-delayed-capacity-r1` 准备同输入实机重放。当前确认减少的容量仍仅为3,230,560,256B，未宣称长参考生成成功。

2026-11-15 13:58（Asia/Shanghai）：去除无用 K/V payload 后，原长参考实测计划从34,894,877,680B降到31,664,317,424B，准确减少3,230,560,256B；仍比29GiB预算高525,804,528B，CLI仍exit1，不能称容量修复完成。新 `DependencyOrder` 的2项定向测试及完整12项共享组合测试全部通过，包含真实 Core 规划边界、两种输入的受管执行与完全回收；较低容量能规划/执行，低于自身最低容量仍正确拒绝。新候选 `video-dependency-capacity-r1` 源覆盖 SHA `4474e010210ee38145a373ab838ac0178c36873a9725669bd40f22447c1ff3e4`、CUDA二进制 SHA `8c02f1e9c9bb741ba95a3beca78a73b8d80ed2f9d33623d9aa0275768133cc2a`，增量构建3分28秒成功；同一原长参考CLI重放进行中。三个容量候选均保留实际错误与明细，不以编排exit0替代生成通过；后续HTTP/长15秒资格须在当前候选通过后执行。

2026-11-15 13:50（Asia/Shanghai）：长参考容量诊断已完成，`video-long-capacity-diagnostic-r1` CUDA 构建 exit0，重放同一243帧视频参考的 CLI exit1、无容器 OOM。4096行 FFN 下实际仍需34,894,877,680B；明细定位到 cuDNN prefix cross-attention 为已经直接读取的 K/V 额外预留3,230,560,256B payload，另有512B长度数组。该无用预留已在共享 provider 的准备/估算/执行共同布局处移除，保留原 native signature 与算术；6项 CUDA 后端布局测试通过，产品增量构建和同输入重放进行中，尚未宣称容量通过。另修正直接 K/V 在第二个16样本 mask tile 的起始偏移，独立布局测试覆盖19样本，此修正不是当前B1容量成功证据。候选 `video-cross-scratch-capacity-r1` 覆盖 SHA `a17ff81d537b8ea17b7e5e8bb8f2b903e01e6f310a3725438d60e19ee32d9613`。完整原始容量明细与采集时状态在库外 `video-long-capacity-and-scratch-r1.observation.json`；最新结果仍以各冻结候选结束记录为准。

同时准备通用 stateless tensor composition 的显式 `DependencyOrder` 调度：保持每个节点、操作数、权重、数值合同及输出不变，只按输出依赖推迟独立分支；仍由 Core 的实际计划决定容量。产品仅在整行与分块 FFN 均无法规划时尝试该候选，短请求保留声明顺序。CPU 真实容量边界/数值/回收测试正在运行；未验证前不作为长参考支持或性能结论。禁止用增大显存预算、裁掉更多参考、减少评估步数或降低解码质量掩盖当前缺口。

2026-11-15 13:33（Asia/Shanghai）：短参考视频的 C2 和内嵌音轨资格已补齐。`video-reference-coverage-r2` 两条不同 seed 均完整有效，P50/P99为196.341783/238.080745秒；seed1101仍保持原CLI/HTTP的MP4 SHA。实际为16次B1、零请求拒绝，不作为视频参考B2共批证明。相同视频加入32kHz PCM音轨后1/1有效、122.300767秒，SHA `5f3e4f7d90ce218c6402519ec63232deab5f93bb9afee6b57bda0665b9feaff2`。两格GPU采样不完整、客户端exit1；功能通过不等于资源或正式性能通过。首轮768p测试请求错误地包含接口不接受的`size`字段，HTTP400发生在推理前；保留失败并仅修正请求。随后`video-768-product-r2`完成1344×768、124帧、24FPS、带音轨的完整解码，1/1、108.215037秒，MP4 8,850,188字节，SHA `dff9abea8a9d1f3468f7207a8defb642f9bf01ea613e8a2e6195a48dde6798c2`；600秒功能门槛内通过，仍有GPU采样缺口，不扩称768p长片或性能对照。

**当前阻塞：长参考仍会在规划容量不足时失败。** `video-reference-longclip-r1`使用15秒循环源、只输入视频、按243目标帧裁剪参考，目标832×480/243帧；完成视频VAE、视频Qwen与文本条件后，于207.949秒失败。整行FFN计划需41,137,438,704B；自动4096行分块后仍需34,894,877,680B，超出31,138,512,896B预算，其中静态11,928,241,712B、step19,463,295,840B、invocation峰值3,503,340,128B。服务正常exit0、非进程OOM；请求0/1，不能计为长参考支持，也不符合用户容量不足应继续完成的要求。正在给Core的原容量拒绝补充只读张量/槽位说明，以定位可减少的真实重叠；不改变预算和拒绝规则。后续须修复并复测这条原请求。

同seed的短视频参考首/中/末三帧已与FastVideo人工比较：主体和场景可辨识，未见空白、噪声或明显拼接损坏；FastVideo的推近幅度更明显，Ferrum运动较弱。这不是完整时间序列、音频语义或感知质量等价证明。另一个独立输入边界修改统一限制codec探测/画面/音轨只读取声明容器及本地协议；5项实际FFmpeg/ffprobe测试通过，包含相邻文件播放列表拒绝、网络读取拒绝及原像素/PCM/取消/容量回收。该本地改动尚不在上述冻结r2产品二进制中。

2026-11-15 13:05（Asia/Shanghai）：`video-reference-product-r2` 的独立 CLI 已正常退出0，内部请求耗时129.991888秒；与 HTTP 的同PNG/2秒无声视频、同prompt/seed1101产物逐字节一致，SHA `c9665cce67298b47f31892313be3a0655a0da004f6ceb675a3def68c56bfbcc8`。CLI进程整体生命周期含冷启动及退出回收，不能将内部时间直接当作HTTP延迟。随后同5090、原NVFP4/8步/36层、相同媒体/prompt/seed/832×480/124帧，FastVideo新进程首请求167.916421秒，Ferrum对应132.892554秒，本轮耗时低20.86%；双方1/1、完整音视频解码通过。**仅每侧一次首次请求诊断，非正式重复主表或质量等价证明**；双方分别4/6次GPU查询超时，客户端均exit1，资源峰值只是观察下界。FastVideo原启动文件已恢复SHA `6f0810758f3951fd435d462c6566b4507f147858dfd9e6e5c35a916ba705c514`，服务inactive/disabled。完整报告、配置、日志摘要与CLI证据归档库外 `video-reference-product-r2.results-final.json`。下一轮沿用该二进制验证双请求参考视频、内嵌32kHz PCM音轨及1344×768/124帧生成；均预声明600秒功能门槛，不替代120秒原单图性能主表。

2026-11-15 12:42（Asia/Shanghai）：同一 r2 服务随后完成原图片控制请求，1/1有效，68.872838秒，完整音视频解码通过；MP4 SHA `a6278466a355a0cba2d1eea8484b04d66e926ab74de75b11106a9a8b867ca229` 与接入视频参考前的原控制逐字节相同。客户端仍因GPU采样超时 exit1，不能当作完整资源通过。已检查视频参考输出的首/中/尾画面：人物、湖边场景保持可辨识，未见空白或噪声输出；这只作基本视觉有效性检查，不是量化质量对照。HTTP服务已结束，开始独立CLI的相同PNG/视频、提示词、seed1101、124帧生成，核对两入口完整MP4一致性。

2026-11-15 12:38（Asia/Shanghai）：`video-reference-product-r2` 首条“图片＋2秒参考视频” HTTP 完整生成通过：832×480、124帧、24FPS，存在音轨且完整AV解码有效，1/1成功、0生成错误，首次请求132.892554秒，MP4 4,205,928字节，SHA `c9665cce67298b47f31892313be3a0655a0da004f6ceb675a3def68c56bfbcc8`。新二进制 SHA `c3e1edb0d71f2160134fb9dde868ff7d58100a27e534dd29b367ee864fda96c5`，增量构建2分31秒；沿用原NVFP4、8步、36层解码器、29GiB/26.25GiB缓存配置。视频VAE实际执行84个时空块，之后完成视频Qwen、文本条件、去噪及音视频解码。**客户端 exit1：4次GPU采样超时导致资源证据不完整**；采样显存28,922,871,808字节、RSS43,509,555,200字节只是下界，不能声称资源验收通过。仅1次功能诊断，600秒门槛内完成，不是人口P99或FastVideo胜出结论。HTTP原报告在 `video-reference-product-r2-runtime-video/report.json`；正在进行图片路径回归和同种子CLI一致性，参考视频的性能比较仍待完成。r1的CLI3项、HTTP5项定向测试已全部通过，CPU复用标识/容量3项也通过；未重复全仓测试。

2026-11-15 12:31（Asia/Shanghai）：视频参考产品 r1 的 CUDA 二进制构建通过（3分27秒，SHA `536b0e547fd787d6f162e0dcdf3c2e02529cab20420e6f33f33bad530ace66e0`），真实 HTTP“PNG＋2秒参考视频”已提交，但在8.34秒时失败：共享 CUDA composition 未注册 `operation.conv3d.f32`，因而尚未执行视频 VAE。原图片编码/视觉阶段已执行；不是端到端通过。Rust 客户端 exit1、0/1 完成，资源采样另有一次 GPU 查询超时；编排 exit0 只表示收集结束，不代表验收成功。原报告及服务日志保留为库外/5090 `video-reference-product-r1-runtime*`，服务已停止。r2 补齐同一 composition 的三维卷积、分帧归一化和 NCTHW 残差合同/provider，并将原始视频输出线程数也纳入显式 codec 配置；3文件覆盖 SHA `8a204bfc4d2358a34c823abae8116a3c6a9f6419bc8c43e44ae51e9fb5f88695` 已冻结并进入增量产品构建。r1初次请求与CPU测试编译有时间重叠，不能用于速度比较。

2026-11-15 12:22（Asia/Shanghai）：`video-qwen-r2` 的实际 CUDA 视频视觉编码、主输出与三层 DeepStack 时间隔离通过；同一构建的 Qwen CPU 回归 34 项、呈现 4 项、时间线 2 项、视频分块 2 项及图像归一化 1 项通过。已纠正视频归一化的除法/倒数乘法舍入差异，与图像消费者共用规则。原始权重 `[1,3,22,320,320]` 分块编码输出 `[1,24,7,20,20]`，执行/读回/采样 4.470821 秒；posterior SHA `e16215c51a5645c087f6fdb7b29f57e5f24bf31a1597ff6344da53002213348e`，测试二进制 SHA `477cf56fadb3b78b3072f11a126be16fec6aa56b4e2bb7e3fef2cd10ce237554`，完整证据 `video-qwen-r2.results-final.json`。r1 曾因 CUDA 测试夹具的注意力头维度 8 不受原算子支持而失败，r2 改用原模型维度 72，通过后保留原失败记录。

视频容器输入采用显式 FFmpeg/ffprobe 配置、串行暂存文件和取消/超时 kill/reap；只保留所需帧的有界读取，输出由原 Host 账户管理。3 项真实 codec 测试通过，包括精确帧值、容量释放后重试、取消/超时资源退还、32 kHz PCM 原值与 48 kHz 单声道重采样。音轨的跨采样率预处理明确使用 libswresample（32 taps / phase 10 / cutoff .97），与 FastVideo 的 torchaudio/scipy 滤波器不同，不能声称该预处理逐位一致。现已接入有序参考、共享视觉编码、VAE 分块与音轨条件、时间戳和完整 Host 峰值规划；`video-reference-product-r1` 冻结 23 文件覆盖 SHA `eec4da466570cce8d28cebda0f694fa825a70767d4aeb89eca4d02e5a2d2bf68`，进入 CUDA 产品构建及 CLI/HTTP 定向验证。**这些改动尚未取得参考视频端到端生成资格，未增加新的性能结论。**

2026-11-15 11:35（Asia/Shanghai）：`video-encoder-r3` 原始权重分块视频编码在 5090 通过，编排/容器均 exit0。输入 `[1,3,22,320,320]`，原17帧分片×2、256空间分块/64重叠共8块，拼接、尾部裁剪及完整序列posterior输出 `[1,24,7,20,20]`，执行/读回/采样4.963617秒；moments SHA `16a390bb391ddc68c2f771127274655f39313bcc5b6486f82c9a1a8592d7f13a`，posterior SHA `0189d0d219868b22549f8fdd3aef20bdcb82dfae572e99a304cfe529076b8c66`。5项编码器测试及1项分块容量重试/通道时间拼接测试通过；二进制 SHA `d1e0dd49351c44240f9a59a836bb11ce9ce04ecfa0c201fd53fa0eb5e3e78413`，结果保存库外 `video-encoder-r3.results-final.json`。这仍是编码组件测量，视频参考的媒体读取、Qwen逐时间块呈现及run/serve完整生成尚待接入，不作为端到端或FastVideo性能比较结论。

2026-11-15 11:26（Asia/Shanghai）：`video-encoder-r2` 已完成原始权重 GPU 编码资格，编排/容器 exit0。原六级编码器以完整17帧、256×256、F32全时间核执行，输出 `[1,48,5,16,16]`；两次执行/读回0.276539/0.528451秒，初始化0.142731秒。修改最后一帧后所有更早 latent 逐位相同、末时刻改变，有限输出及最终Host committed=0通过。首输出 SHA `61391e50aea6d4e777a196c29f36d78f07fb054f2fd47766aca4184aea350ca3`，试验二进制 SHA `3f21966031f382711c414f4800d4c387d3bef35b836aa751f8499a516ccbf972`；完整记录/SHA在库外 `video-encoder-r2.results-final.json`。完整小型存储权重 CPU/CUDA 独立 F64/因果性、容量形状拒绝、时间分片1项、posterior3项和原图像编码器回归均通过。r1曾因CUDA默认能力清单遗漏五维残差在执行前失败，r2补齐声明，原失败记录保留。

视频分片消费者沿用原组件会话/serial actor和Host账户，逐17帧/空间块编码，原始RGB均值标准差、重复末帧、raw-neighbour拼接、全序列仅一次尾部裁剪、posterior完整采样均已实现。CPU独立F64全图＋四块/两片拼接、通道时间顺序和Host容量释放后继续的测试通过；`video-encoder-r3` 已冻结8文件覆盖包SHA `aa10c77e7f97b3bb347d0ff4d88b2d60d16ea34be0a711cd466e44c805e068c1`，接续原始权重22帧320×320的完整分片编码。**这些是编码组件资格，不是HTTP/CLI视频参考已交付或FastVideo端到端提速结论**；视频容器解码、Qwen视频帧/时间戳呈现、共享条件装配和完整请求验收仍需完成。

2026-11-15 11:08（Asia/Shanghai）：视频参考底层资格已推进至实际 CUDA：`video-conv3d-r2` 的四种三维形状/独立 F64 对照与因果性、31项 cuDNN owner、实际音频转置卷积及完整六级图像编码器均通过。r1 在执行前因合同禁止实际 scratch 被拒；r2 将 scratch 声明为 optional，实际 workspace 仍由原资源权威核算，没有放松容量检查。`video-frame-norm-r1` 随后通过 NCTHW 分帧归一化＋三维卷积独立 F64/未来帧隔离，以及原图像编码器 GPU 回归，编排/容器均 exit0。原日志 SHA 和结果在库外 `video-volume-primitives-results-r1.json`。完整因果视频编码器已形成独立模型图，复用现有执行/权重/容量协议，保留所有原始 F32 时间卷积核；17文件冻结 `video-encoder-r1` 覆盖包 SHA `3fb2245b65d0baa6e8b12770b264a132bf12f1a5f43d2619466c77a836686f32`，接续小型存储权重的完整图 CPU/CUDA 数值资格。**尚未完成真实原始权重视频编码或视频参考产品入口，不宣称视频参考已可用。**

补齐混合参考对照记录：`video-audio-pair-r1` 的 Ferrum/FastVideo 预热及 C2 四个 Rust 客户端 wait 状态均为 exit1，生成输出仍全部有效，GPU资源采样失败已保留；不能将编排 exit0 当成整轮基准全绿。双方同 seed 首/中/末三帧抽查未见明显损坏，但完整感知质量等价和音频语义未验收。

2026-11-15 10:43（Asia/Shanghai）：`video-audio-pair-r1` 同机诊断已结束，原八步/36层/NVFP4、832×480/124帧、相同图像→声音输入顺序、相同 seeds1101/1102。双方各一条原图预热、两条混合参考计时，6/6 完整音视频有效。Ferrum/FastVideo 的 C2 P50 为 91.282567/120.698675 秒，P99 为 114.487313/155.454493 秒，吞吐 1.043833/0.768424 clips/min，本轮 Ferrum 吞吐约高35.8%，只有 Ferrum 满足预声明120秒门槛。**仅一轮、每侧n2，不能替代正式重复或证明感知质量等价；Ferrum 原生日志为两条B1，无联合B2提交**。C2 VRAM观察峰值分别28,922,871,808/27,719,106,560 B，RSS分别43,120,250,880/50,504,953,856 B；GPU查询失败2/8次、跳过501/679个采样间隔，所有峰值仅为下界。原报告/SHA/实际成员记录与配置恢复在库外 `video-audio-pair-r1.results-final.json`，FastVideo 启动文件已恢复原SHA `6f081075…c514`。

声音在前的上一轮与本轮图片在前的输出本来就有不同文本位置，未进行跨顺序 hash 等价断言。上一轮的 CLI library target 已实际执行3项并通过；bin target执行0项和Mac磁盘不足记录仍保留。视频参考下一步是完整时序CNN；新增F32 Conv3D采用各轴独立补边、原始全时间核，复用既有cuDNN owner、准备身份和真实workspace，不以逐帧图像编码替代。2项几何边界、2项Reference物理范围/外部owner、1项实际受管组件测试（四种形状、独立F64相关和未来帧隔离）通过；25文件冻结 `video-conv3d-r1` 覆盖包SHA `d52be898d7271a313e85b17cded1c2432a27ff8a63a84550c76246e74f0e6bcb` 准备在5090验证CUDA及既有2D编码器。尚未建立视频参考产品资格。

2026-11-15 10:34（Asia/Shanghai）：有序图片＋声音参考已经贯通共享请求、条件缓存、真实 Host 预估和 run/serve。`video-audio-product-r1` 的 CUDA 产品构建通过，二进制 SHA256 `3c5bf3fe1cf5079b0f787cacd367105c2f4b3fe31a513c17804cea6bbe4f0c08`；原纯图片控制 71.883046 秒且保持原 MP4 SHA。声音在前、图片在后的 C2 两请求完整有效，P50/P99 92.793145/115.998265 秒、1.030292 clips/min；同媒体缓存 C1 为 47.954538 秒，独立 CLI 的请求耗时 67.611080 秒。同 seed1101 的这三个入口/调度场景 MP4 字节一致，SHA256 `16d67127f26502a23501f7145cd4b28579d3544d4ac59a8afc100671de7c3986`，124 帧、24FPS、832×480、双声道音轨。四个 HTTP 请求及 CLI 全部生成有效；原 GPU 采样存在超时，观察峰值 28,922,871,808 B 仅为下界，不把脚本 exit0 等同于全部采样完整。本轮混合参考预声明门槛为 600 秒、一次功能资格；不继承旧二进制的正式性能胜出结论。

首次编排在启动前因库外 provenance 文件属主不符退出；修正该部署文件属主后以同一冻结二进制恢复，保留原 exit1，恢复链 exit0。原 provenance 的 `reference_png_sha256` 误继承了上一实验的新图 SHA；请求字节复核证明本轮实际使用原图 `b5416687…e9c9`、WAVE `fde04fcb…7fa65`，原记录不覆盖，按此更正解释。报告及原始 SHA 保存为库外 `video-audio-product-r1.results-final.json`。WAVE、媒体文本呈现各 3 项、engine 22 项、server 8 项及实际权重 backing 预估 3 项定向测试通过；server 原有依赖外部 ffmpeg 的测试 1 项明确忽略。Mac CLI 测试因磁盘不足未执行；迁移到 5090 后首次误选 bin target 实际执行 0 项，不能称测试通过，已改为 library target 运行同一过滤条件。

已冻结 `video-audio-pair-r1` 同机配对诊断：双方独立启动，原图预热后，以相同图片→声音顺序、原采样率 WAVE、提示词和 seeds1101/1102 测新条件 C2，固定 29GiB、原八步/36层质量、120秒经验 P99 门槛与完整有效率1。FastVideo 分组 API 固定图片在前，故不复用上轮声音在前的计时作对照。沿用原 Rust benchmark，无重新编译产品；一次配对只作功能与瓶颈诊断。视频参考原始时序编码、其他音频采样率/压缩格式和两侧完整感知质量资格仍未完成。

2026-11-15 09:57（Asia/Shanghai）：原始声音参考编码器已完成 5090 实测 `video-audio-encoder-r2`，CUDA 构建和执行均 exit0。输入为既有有效 MP4 提取的 64,001 个 32kHz 双声道样本；完整 F32 CNN、因果投影、posterior mode 和归一化得到有限的 `[2,32,81]`，SHA256 `7172969473e57739cf889aa9b9210235eb151cbf945bcad599d70d615dc6b0ed`。编码准备/执行/读回约 0.623643 秒，runtime 初始化约 1.886747 秒，最终 Host committed=0。二进制 SHA256 `b96a0f700b55149b4dbc9882b31dd6f2eb86d611278aad38bd5506abd2369c4e`，原日志 SHA256 `517f83748cd70f4a65e9307d81b3ca40c5d2506c3b51a2ba6a49e98c4e329620`。r1 首次执行结束后验证程序仍持有转换前的交错输入，回收检查 exit1；r2 仅修复该消费者所有权，保留 r1 失败记录，没有绕过回收检查。此前上传中断的网络已恢复。WAVE 输入 3 项与有序图像/声音文本呈现 3 项定向测试通过；当前正在把 typed 有序媒体参考接入原请求、缓存、Host 预估和 run/serve，**尚无图像＋声音完整生成或 FastVideo 声音质量对照结论**，当前输入切片为模型原采样率的 WAVE。

2026-11-15 09:33（Asia/Shanghai）：声音参考编码已实现原始 F32 CNN、直接参数 Snake、因果注意力投影、两级均值/池化、GeGLU、posterior mode 和独立 F32 归一化。沿用现有组件会话、Host/device 权威和 serial actor；CNN 完成关闭后才安装投影，不另建推理运行时。5 项声音 VAE 定向测试通过，包含原解码器回归、容量回滚、声道隔离、补零/膨胀卷积以及独立 F64 投影对照和未来时刻不影响过去输出；另外 signal/channelwise/pointwise 各 4 项算子测试通过。`cargo check -p ferrum-engine --lib --example encode_audio_reference`（默认特性）通过；**这不等于 CUDA 编译或产品声音输入已验收**。冻结 `video-audio-encoder-r1` 源码覆盖包 SHA256 `cc3a1216964d42ce3c6acc3141b9bec8d26fb07e09058ee44d27a5a6975f636f`，准备在 5090 用已有 MP4 的 64,001 个 32kHz 双声道样本运行完整编码链路。HTTP/CLI 声音 ingress、真实音频生成和 FastVideo 配对对照尚未接续完成。

2026-11-15 09:02（Asia/Shanghai）：运行中取消/补位实验`video-lifecycle-product-r1`客户端及服务正常exit0。health确认两请求已完成两个联合去噪步后，于45.950秒DELETE首请求，并于45.961秒接纳替补。原生日志表明取消到达时第三个B2已提交，因此原成员`request.fast_h3.51`在evaluation2完成fence后退出，没有继续evaluation3；存活成员`.59`随后与替补`.66`在同一forward组成evaluation3/0，继续4/1、5/2、6/3、7/4，最后替补独立完成5–7。两份完整音视频分别保持原seed1102/1101的MP4 hash；最终active/pending/preparing均0、rejected=0。客户端404仍只保留删除事实；结合实际成员/步号/retirement日志与服务正常收敛，建立该单图480p场景的运行中取消和混进度补位资格。完整原日志SHA256 `f937f2d740c793972637dd999fdaaf2ca8118bf1745a2bce64d0b708e723605f`，报告SHA256 `9be8665fd8debf9c5b4a08d509e453dbf65623ea7dbe55729ad2548c39095412`；本机副本为`video-lifecycle-product-r1.results-final.json`及`.native-events.json`。未因此扩称所有取消阶段、参考组合或物理OOM均已覆盖。

2026-11-15 08:57（Asia/Shanghai）：有序双图产品候选`video-multi-image-product-r1` CUDA构建、HTTP三格、双图CLI均exit0，二进制SHA256 `93aebf2898c5d03eb638fd3547c290ed2a83be8d0413c9f7d22a5d55af94e42e`。原图冷请求68.130069秒，新参考C2 P50/P99 113.321577/114.927802秒、1.043836 clips/min，三条控制MP4全部保持原hash；这是一轮回归，不能代替旧冻结二进制的三轮正式对照。两张按序参考的HTTP请求124.103633秒，完整124帧/24FPS/双声道音轨有效；CLI同输入完整解码exit0且MP4 SHA256同为`1871159918905015a8caccf5d014eb1bedbb80b8bfb082bdfb28bc9004a28f11`（3,788,533字节）。原报告SHA、资源漏采和CLI容器状态保存在库外`video-multi-image-product-r1.results-http.json`。当前实测覆盖两张参考，九张上限和不同组合仍受独立token/Host/device预算，不把一例双图扩称所有组合已测。

连续批处理补充资格`video-lifecycle-product-r1`已冻结Rust客户端6文件overlay `1fdfb1d83b1ed15b62dc68e65234538a6a990464cb3e8dff7bcd4dedbb258e41`，本机10项定向协议测试通过，Linux构建exit0，客户端SHA256 `6f89dda93449237c97969ad57fb63ec2beb8895567227307e7bdb7ceccff78e4`。复用原有有界HTTP、产物下载和完整解码；新增显式`--lifecycle-plan`，按真实health数值等待联合B2完成步，再DELETE原seed1101并提交替补同seed请求，最后观察剩余两条产物和调度器收敛。故意取消独立于性能主表，不靠固定sleep猜测GPU阶段；DELETE后404保留`remote_terminal_unknown=true`，必须结合原服务fence/成员日志判定退休。该GPU实验正在执行，尚无取消/补位通过结论。

2026-11-15 08:47（Asia/Shanghai）：`video-duration15-r1` 实际 CUDA build、HTTP 与 CLI 全部 exit0。二进制 SHA256 `4a025877b6064ca6639cdedc4ed3f92f23281ed2e59c834f3951c856419b863c`；原124帧冷请求67.619824秒、MP4保持原hash。两条362帧/24FPS新参考请求均完整音视频验证通过，分别122.094630/236.975383秒，经验P50/P99为179.535007/235.826575秒、0.506382 clips/min，满足事前600秒/全部完成的容量门槛。真实日志只取得B1动态backing，没有把客户端C2计为联合B2。单独CLI生成相同seed1102的15.083333秒视频，362帧、32000Hz双声道；MP4 SHA256 `c48263639f14daff94316f4d7587d1c266796bf46a2c78661d3c1bb3e0ac2e33`与HTTP相同。完整解码exit0，视频与音轨起点均0；AAC容器末尾含编码padding，不能把15.107秒容器时长当成视觉时长。报告和原始hash在库外`video-capacity-current-0843.json`、`video-duration15-r1.qualification-final.json`。GPU/RSS采样有超时和漏采，28,922,871,808B显存峰值只是下界；该轮是容量资格，不是新的正式性能对照。

同轮外部8GiB显存占用测试`video-external-pressure29-r1`完成exit0，两请求175.177582/179.967428秒，完整产物分别保持原fresh1101/1102 hash。GPU采样下界33,554,432,000B且有长时间查询阻塞；原始报告保留失败及漏采。缓存预备及B1/B2实际backing均成功，因此只证明外部占用下可慢速完成，尚未覆盖真实原生分配拒绝后的等待/缩批。多图产品候选已接续CUDA编译；下一项验证运行中取消、退休和新请求补位，不将HTTP删除记录单独视为GPU退休证据。

2026-11-15 08:34（Asia/Shanghai）：10秒独立容量对照`video-ten-second-capacity29-r1`结束，双方各3/3有效、exit0。Ferrum冷101.419785秒，C2 P50/P99为140.205113/176.755847秒、0.676049 clips/min；FastVideo冷143.716682秒，C2为172.199846/222.669997秒、0.536433 clips/min。双方满足预声明300秒；只有一次容量配对，不并入5秒正式区间。Ferrum真实日志显示两条10秒请求各走8步B1完成，不能将客户端C2当成联合B2。原始报告/SHA、资源失败和产物均在库外`video-ten-second-capacity29-r1.results-final.json`，运行事件摘要为`video-formal29-and-ten-second-runtime-events-r1.jsonl`。

外部压力辅助程序首次因运行容器缺`libnccl.so.2`在分配前退出，15秒/多图依赖任务随之在构建前退出。`video-capacity-continuation-r2`复用已构建Rust辅助二进制，补原NCCL挂载/搜索路径、修正Docker readiness格式；ldd已确认全部依赖，首轮exit/log/container state保存在远端`video-capacity-continuation-r2-attempt1/`，不覆盖成成功。恢复链root PID2041733按原先冻结清单顺序重接压力→15秒→多图，无新模型参数或质量更改。

恢复后的`video-external-pressure29-r1`在08:28:10观察到辅助进程完成独立8GiB清零/同步并保持240秒。冷C2两请求均有效，P50/P99为177.572505/179.919530秒、0.666787 clips/min，服务与辅助进程均正常结束。原26.25GiB启动预备成功，随后实际B2额外backing也取得；没有观察到真实原生分配拒绝。因此本次是外部占用下的成功/变慢证据，不冒称已经覆盖物理OOM后B1回退或等待分支。15秒候选已开始构建。

多图候选`video-multi-image-product-r1`冻结16文件overlay `d591781c9cf56c4ba0aad6f5910bc6d0cf53a3a40d2a33e4f1705f781c5d394c`，父源为待完成的`source-video-duration15-r1`。文本2项定向测试再次通过，新增实际Host footprint与分配字节一致性；未跑全仓。新输入保持模型上限9张、合计33MiB编码输入和32768文本token等独立限制，两个入口采用同一ordered-image conditioning与实际几何容量计划；原Host账户负责最终准入。已预登记旧单图冷+新参考C2回归、双图HTTP C1及同双图/seed1101 CLI，短控制120秒、双图生存600秒，全部完整音视频并核对对应SHA；尚无CUDA编译/运行结果。该候选不在已完成的正式单图性能证据中。

2026-11-15 08:15（Asia/Shanghai）：正式`video-fasth3-formal29-r1`全部12格结束、orchestration exit0，24/24计时请求及12/12暖机均有效。Rust配对分析取得：C1中位延迟Ferrum/FastVideo几何均值比0.752526（95%CI 0.697732–0.811624，每轮下降），C1 P99比0.779473（0.715738–0.848884）；C2有效吞吐比1.273769（1.173945–1.382081）。Ferrum全部六格满足120秒/全完成；三轮C2 P99分别115.934147、116.963542、115.936568秒；FastVideo对应141.589709、151.233710、149.853826秒，全部miss保留。C2中位延迟比0.993868（0.917153–1.076998），没有显著改善，不据吞吐增益声称各项延迟都变快。

本组固定小样本的性能统计门槛已满足；每格仅两条请求，P99仍为经验分位，未证明生产尾延迟或全面质量等价。Ferrum计时格显存采样下界均26.9365GiB；FastVideo为24.7646–25.3486GiB；相同29GiB配置上限不等于相同实际占用。Host RSS计时格分别35.5214–42.7853GiB、36.8053–47.0747GiB；GPU查询超时以及FastVideo最后一格的一次进程采样失效均保留。六次独立冷启动首请求Ferrum为68.8675–75.4318秒，FastVideo为99.4118–114.3543秒，单列不混入主表。报告为库外`video-fasth3-formal29-r1.results-final.json`（原远端SHA和去输入body副本）、`video-fasth3-formal29-r1.comparison.json`和不可变实验清单；原参考与同参考seed复用口径见下条。10秒档Ferrum首条243帧已有效完成，101.419785秒，正在C2；物理压力/15秒仍排队。

多图文本准备已通过2项定向测试，包含旧单图输入、两种空间尺寸的有序标签/视觉替换/DeepStack和实际容量回收；原图/视频/声音latent准备的3项测试此前已通过。当前本地进一步接入多图模型/CLI/HTTP共同流程：原始图片逐张编码、文本一次合并、所有参考latent按序冻结，prepare先按PNG尺寸提示和原processor/模型几何计算同时存活的Host需求；实际解码和原账户准入继续决定有效性。此新增接线尚无CUDA构建或真实多图生成资格，不能由已完成的单图性能报告推定通过；完整视频/声音参考仍待encoder和输入协议完成。

2026-11-15 07:57（Asia/Shanghai）：正式29GiB已完成9/12个独立服务运行、18/18条计时请求有效。前两轮C2的Ferrum/FastVideo P99分别为115.934147/141.589709秒、116.963542/151.233710秒，吞吐分别为1.034755/0.843762、1.025673/0.789943 clips/min；Ferrum均满足120秒，FastVideo两次超限均保留。第三轮Ferrum C1 P50/P99为56.261614/64.903735秒。仍等待剩余三个单元，未计算完整配对区间或宣布目标达成。测量格内两个seed使用同一参考和prompt，允许第二请求复用首请求已完成的conditioning；“新参考”指每个独立服务测量开始时不同于暖机输入，不表示两条请求均无缓存。所有显存采样均作为下界，GPU查询超时单列。

首轮C1的双方原参考和新参考四份MP4已下载并逐份校验原报告SHA；FFmpeg比较全部124帧，SSIM分别0.618959/0.661855，不设事后阈值、不据此推定语义等价。实际查看第0/60/123帧：主体、衣着和湖边场景保留；原参考Ferrum多一人抬手，新参考双方末帧人物都向右边缘移动，和严格居中的慢推近有差异。完整音视频解码已通过，但未听审音轨、未建立完整时间序列质量等价。证据为库外`video-qualitative-pair-r1.inspection.json`及对应MP4/帧图。

15秒资格已排在10秒与物理压力实验之后：`video-duration15-r1`仅将公开类型默认帧数上限从260改为362（15.083秒），保留实际逐请求容量准入；冻结单文件overlay为`02461ac9f5a326415254cdd2e285324f13d3715629ea3b44ca1405c691bb5e34`。将执行124帧短请求回归、362帧C2两条及同输入CLI一条，长片预声明600秒生存界限。此时尚未构建/运行，不算15秒已验证。

本地模型准备入口已扩展为有序`FastH3ReferenceLatents`：图片/视频视觉latent按原tensor边界依次取条件噪声，再取目标视频/音频噪声；独立音频和视频音轨按原声道/时间顺序复制，不加参考噪声。布局从实际encoder输出重建并核对，所有payload/scratch仍走原Host账户；现有单图调用实际委托此入口。3项定向测试通过（包含原单图回归、混合顺序/打包和RNG、foreign owner/真实容量拒绝及释放），无全仓测试。该本地改动不在上述GPU冻结候选中；多参考encoder和产品接线仍待完成，不能宣称图片/视频/声音参考全支持。

2026-11-15 07:29（Asia/Shanghai）：正式29GiB第一对C1完成，Ferrum/FastVideo P50 **55.629236/76.550148秒**、P99 **63.893535/84.959760秒**，各2/2有效；Ferrum首轮C2 P99 **115.934147秒**、**1.034755 clips/min**、2/2有效，满足120秒，FastVideo对应C2仍在运行。后两轮未完成，不能宣告稳定胜出。Rust分析器额外保留第8.2节原定C1中位数逐对下降及其95%区间，同时报告冻结实验的P99指标，不以较有利指标替代原目标；2项定向测试通过，无GPU源码变化。

已按顺序登记后续两个独立容量实验，复用二进制 `42fd4948…bc46`，不干扰正式计时。`video-ten-second-capacity29-r1` 为双方各1条冷请求和2条新参考C2、243帧/24FPS（10.125秒），同29GiB上限，预声明P99≤300秒/全成功；不混入124帧主表，也不以一次配对主张置信区间胜出。`video-external-pressure29-r1` 使用仓库现有Rust `cuda_memory_pressure` 在独立上下文实际清零并同步8GiB、保持240秒，Ferrum仍维持29GiB配置、冷C2两条124帧请求，生存界限600秒。必须从真实分配拒绝、回退/等待和最终原MP4 SHA验证压力路径；仅成功完成或配置了8GiB不算已覆盖。后者等待长片实验正常清理后才构建辅助程序及执行；当前均无运行结果。

2026-11-15 07:16（Asia/Shanghai）：同一r2二进制29GiB/26.25GiB预备筛选完成：新参考C2 P50/P99 **113.313149/114.918613秒**、**1.043919 clips/min**，冷67.851783秒，3/3有效且MP4 SHA与28GiB一致；8个B2步骤均取得设备预测handoff，新增并存峰值16,130,427,824 B，无超过1秒分配。显存采样下界28,922,871,808 B、C2 RSS44,736,876,544 B，GPU查询2次超时。相邻FastVideo仅将相同公开cap改为29GiB（原文件归档、测试后原28GiB字节级恢复），C2 P50/P99 **108.093455/139.279505秒**、**0.857658 clips/min**，暖机99.184563秒，3/3有效；显存下界27,217,887,232 B，C2 GPU查询6次超时及1次VmRSS不可得，保留采样不完整。单次吞吐比约1.2172，不能代替正式统计。

20GiB实际双请求回归已完成：2/2有效、所有对应MP4 SHA相同，P99 124.808311秒、最后产物125.287151秒（容量诊断预声明300秒）。B2要求24,279,080,976 B超出21,474,836,480 B后，两条各8步B1完成；无超过1秒分配，显存下界19,158,532,096 B，2次GPU查询超时。此证据覆盖逻辑预算下的真实产品回退，仍不代替外部进程物理backing压力验收。

正式 `video-fasth3-formal29-r1` 已冻结并启动：12个独立服务运行、C1/C2各3对，双方每格相同fresh图/提示词/1101与1102两个种子，共24计时请求+12原参考暖机。每格独立服务保证C2不复用先测C1的条件结果；全程配置固定，第二轮交换实现先后顺序，筛选数据排除。P99≤120秒/全成功，C1 P99成对log比95%CI上界<1、C2吞吐比下界>1；保留所有SLO miss，失败停止并保存，不挑选重跑。计划、命令及报告均库外，后续用Rust分析复用现有ScalarStats。

2026-11-15 07:00（Asia/Shanghai）：`video-shared-ffn-forecast-r2` CUDA构建1分57秒完成，二进制 `42fd4948c545314da600769cb1380e8f89b1f3dd10b07155599266c76344bc46`。`video-shared-ffn-runtime-r2` 同28GiB/25GiB预备的4HTTP+20GiB CLI全部有效、MP4 SHA分别与对应父版精确一致。冷66.841581秒，新参考直接C2 P50/P99 **118.362175/119.966847秒**、**1.000003 clips/min**，随后同参考B1 47.423376秒；本次刚满足120秒，但余量仅0.033秒，尚无正式重复/CI，不宣告验收。C2显存采样下界28,184,674,304 B、RSS44,110,839,808 B，有1次GPU查询超时。原配置父版C2 P99121.751636秒，FastVideo28GiB同工作负载138.412628秒；跨次波动仍需配对重复。

实际B2已选择整行，预测新增15,515,674,544 B、真实预准入持有15,498,897,328 B，移除了重复分块编译；无超过1秒分配。B2未取得设备预测handoff，仍走Host中转（B1有16次device handoff）；不能将device handoff计数当成全部实际batch数。20GiB双请求回退回归 `video-shared-ffn-capacity20-r2` 正在运行，预声明300秒P99/全成功、同精度/步数/decoder；该格为容量诊断，不混入28GiB性能主表。后续同二进制 `video-shared-ffn-budget29-r1` 仅通过公开配置测试29GiB上限和26.25GiB启动预备，考察整行B2与设备预测并存；保留所有native安全预算。两项内存配置相对父版的差异已预声明，不能冒充同预算提速。

2026-11-15 06:48（Asia/Shanghai）：共享权重预测r1完整4HTTP+1CLI均有效、MP4 SHA与原版本逐项相同，CLI exit0且非OOM；但C2 P50/P99 **134.823972/137.292516秒**、0.873725 clips/min，单条54.746096秒，未达性能目标。实际日志确认B1预准入仍持有 **7,749,450,032 B** 空闲动态backing，导致B2整行预测被拒；该backing原先在候选选完并attach之后才释放。r2把现有release移到已确认fence、同原B1域、尚未编译的新cardinality预测之前；保留原scope、continuation和shared installation，不创造可回收额度。扩展Reference真实边界测试保留B1 session，只退其idle backing，再执行少1B拒绝/释放后B2完整执行；6项通过（8.75秒增量编译、0.03秒运行），无全仓测试。2文件源码归档 `7985e184feeb9ffd6f21d685d25d6692ae0db26cc17d08f833b823ea79b90e76`；CUDA构建和同配置4HTTP+1CLI已串行排队。

运行波动仍单列：r1没有超过1秒的分配，但未改动的单条执行也比原对照慢。生成结束后GPU空闲观测为P8/225MHz/42°C/11.06W、600W限额；这不证明生成期间的频率或温度，不能据此解释时延。原报告及命令、CLI媒体验证、GPU采样缺口和空闲观测已保存至库外 `video-shared-ffn-runtime-r1.results-raw.json`。

2026-11-15 06:29（Asia/Shanghai）：解码计划复用候选已按原文件 SHA 精确撤回；保留显存池启动预备与 MP4 时间戳修复。相邻原二进制 `33930e5e…47158` 对照 `video-decoder-domain-parent-r1` 完成：冷75.448935秒，新参考直接C2 P50/P99 **120.142716/121.748169秒**、**0.985376 clips/min**，随后同参考seed1102 B1 **48.213351秒**。4/4完整音视频有效且MP4与对应父版SHA完全一致；16次B1/8次B2，没有超过1秒的分配调用。候选对应135.054265/56.498437秒，故不保留仅CPU少编译但端到端未获益的改动；底层执行变慢机制仍未证明。

同批 FastVideo 新参考直接C2基线 `video-fresh-c2-fastvideo-r1` 已完成，原参考暖机100.935082秒；新参考seed1101/1102的C2 P50/P99 **107.593288/138.412628秒**，**0.863051 clips/min**，3/3有效。相邻Ferrum吞吐为其约1.142倍，但两侧C2均未满足预先声明的120秒P99；只是一次筛选，未完成正式重复与置信区间，不能称目标达成。Ferrum/FastVideo新参考C2显存采样下界分别27,580,694,528/27,217,887,232 B，Host RSS峰值45,154,959,360/49,576,812,544 B；GPU查询超时分别2/4次，峰值不完整。报告的少数字段沿用了旧实验描述，保留原始报告并在 `video-fresh-c2-final-controls-r1.results.json` 明确更正；实际请求、配置哈希与执行命令均保留。

同一二进制自动分块对照 `video-ffn-auto-fresh-r1` 已完成4/4有效且对应MP4 SHA一致：冷67.879946秒、新参考C2 P50/P99 120.144670/121.751636秒、0.985347 clips/min，随后seed1102 B1 47.451328秒。日志确认B1整行，B2先编译整行后退回4096行分块，第二次编译约4秒。候选 `video-shared-ffn-forecast-r1` 将新增内存预测放在现有 `CompiledTensorComponent`：必须借用真实共享权重并验证运行时、物理布局及容量合同，才能免计已驻留静态权重；预测不签发准入。6项Reference共享权重测试通过，含精确容量、少1字节等待/恢复、不同运行时/布局/关闭中owner拒绝；没有全仓测试。CUDA构建2分34秒成功，二进制 `20e5edd45ec95190605fab17eba23117731d81eda281afd3522e27b96d1431b3`、7文件源码归档 `c5b10b0a982687f0e691238439bf7b59faef3a2a4b2e773ea44b4f59666502e3`。相同配置的4HTTP及20GiB/16GiB预备CLI已预登记并开始，效果待实测，不以本地测试代替GPU资格。

2026-11-15 06:17（Asia/Shanghai）：`video-decoder-domain-r1`（二进制`5dc0738e897117a926318858a53d92f1a5601ef3ad1116782fd81f680e5227cb`）实际性能未通过：新参考C2 P50/P99 **132.584854/135.054265秒**、0.888200 clips/min，比相同配置父版121.190187秒更慢。复用本身生效：整服务视频/音频decoder各只编译1次；没有超过1秒的分配调用；8次真实B2。冷请求68.136386秒、随后缓存参考B1 56.498437秒；四条HTTP和20GiB/16GiB预备CLI全部完整有效，分别与同编码父版MP4 SHA一致，CLI音视频起点均为0、24FPS。不能把CPU少编译当成端到端优化成功。

阶段区间显示候选冷B1仍约5.53秒/步，但新参考后的B2从父版约11.1秒/步变成12.8秒/步，随后B1从5.56变成6.37秒/步，decoder也变慢。原因尚未证实；Windows侧单次NVIDIA观测超时，没有可用时钟/温度因果证据。FastVideo新参考C2正在同机运行，另已预登记`video-decoder-domain-parent-r1`，随后复用未修改父二进制作相邻对照，无新构建。候选未晋级，正式重复统计仍未启动。原始结果、各阶段时刻和CLI完成收据均在库外对应文件。

2026-11-15 06:05（Asia/Shanghai）：独立时间戳修复已完成5090构建（`33930e5ed61de7e66aec0a481d7f40a90ceb4c1e6159c3336bf93b7129547158`）及实际FFmpeg9的4项Rust codec回归，全部通过。`video-encode-aligned-r2` 4/4完整有效：冷70.381303秒，新参考C2 P50/P99 **119.461318/121.190187秒**、0.989891 clips/min，seed1102单独48.700364秒；C2仍未满足120秒。8次真实B2，无超过1秒的分配调用，seed1102跨批MP4 SHA一致。原图修复前后124个H264/163个AAC packet的编码payload逐一相同，修复只更正封装时钟；x264 SEI与FastVideo对照一致。各格GPU采样分别1/2/2次超时，显存观察下界27,580,694,528 B，不能称完整峰值证据。编码对齐前后端到端差异约0.02秒，不是剩余瓶颈。

`video-decoder-domain-r1`现已在此独立修复之上增量构建；原有两文件overlay不变。后续已预登记并串行排队：同条件4条HTTP与1条CLI的`video-decoder-domain-runtime-r1`，随后FastVideo原图预热→新参考seed1101/1102直接C2的`video-fresh-c2-fastvideo-r1`。后者修正此前先测新参考C1导致C2缓存命中的比较口径；仍是筛选，不进入正式配对统计。原始记录与摘要在库外对应`.results-raw.json`/`.results.json`，尚无解码计划复用的完整实测结果。

2026-11-15 05:58（Asia/Shanghai）：编码对齐r1首请求虽服务端完成，但媒体校验失败：124帧/832×480，平均帧率23.851977，第一帧duration=0.073730秒，其后才是1/24秒。保留失败；不放宽24FPS门槛，也不计有效性能。实物SEI已确认与FastVideo相同core165/ultrafast/CRF23/15线程。问题定位为非seekable MP4立即写`empty_moov`时未表达codec延迟，影响首帧时钟；现有veryfast路径在本机扩展测试中也复现视频起点偏移。

独立输出修复`video-mp4-timestamps-r1`改为`delay_moov+frag_keyframe+default_base_moof`，等待首片段时间戳可用再写头，继续使用原有有界spool/pipe和进程回收。依据[FFmpeg MP4 muxer文档](https://ffmpeg.org/ffmpeg-formats.html#Options-9)，并以Rust实际编码/ffprobe逐帧PTS验证，不能仅凭文档断言正确。本机FFmpeg8.1.2原版定向测试失败、修复后veryfast/ultrafast均从0起以严格1/24秒推进；完整解码、编码额度、取消/丢弃、超时共4项真实codec测试通过。此修复不改变模型或生成像素；编码字节随容器时间戳改变。三文件源码SHA `793868d50f0e33cfd5aa5fdc71a789e83ca2ed7c8e82a32cccf3d4a73ef81ef4`正在5090增量构建，之后用实际FFmpeg9重跑同4项，再执行`video-encode-aligned-r2`新参考C2。解码域复用的首轮构建因对照失败在编译前退出；已改为基于此独立输出修复，等待r2对照结束再构建，保留各次失败与源码父链。仍未开始正式配对统计。

2026-11-15 05:49（Asia/Shanghai）：新参考直接C2的完整复核已完成，4/4有效，seed1101与既有B1一致；seed1102的C2/B1 MP4 SHA均为`2b6438adceb3758f057f6d013ed4ee98c800df94a683e701d22e4e4e1e3a624c`。采样记录显示GPU查询最长101.47秒，超过配置超时的回收耗时也被保留；只报告显存采样下界，不把这些缺口解释成推理失败或证明采样造成延迟。

`video-encode-aligned-r1`使用同一个pool-prepare二进制，仅将输出改为FastVideo实际imageio FFmpeg 9.0.2（SHA `0d709a0bed1a43f24cd2c420878c63afea4580671b6949ad19ed124572f3bc3d`）、ultrafast/CRF23/15线程。固定原图预热→新参考C2 seed1101/1102→seed1102 B1，Host/GPU200ms采样；编码后的新SHA独立核对，旧CRF18哈希不作为相等门槛。首次编排因provenance缺字段在服务启动前退出，原日志保留，补齐后已重新启动；不改模型、量化、步数或36层解码器，不新增构建。

阶段日志确认第二个新参考请求已命中条件缓存，不能归因于重复参考编码。C2前的B1/B2图准备各占数秒，其中相同目标的音视频解码图随参考域变化重编译。候选`video-decoder-domain-r1`让不可变解码计划按完整目标几何、实际接纳的tile group和后处理策略共享，仍限定同一资产/runtime/limits组合；不共享可变session、时序状态或请求额度，不新增全局缓存。两文件源码已冻结（archive SHA `e57919b4884a7ee1137f0e4eb857f822d819d222e89de44ea4537ea78286c1b1`），格式/diff检查通过；CUDA增量构建将等待上述对照结束，再用实际生成核查性能、输出SHA和跨请求隔离。尚无此候选的GPU结果或正式胜出结论。

2026-11-15 05:38（Asia/Shanghai）：同二进制关闭预备的对照已完成：冷69.630397秒、新参考C1 65.316033秒、复用C1参考的首次C2 P50/P99 137.628405/139.357802秒，0.860875 clips/min，4/4完整有效且SHA等于对应父版。与开启预备相比，C2 P99减少25.678%、吞吐提高34.538%；两侧配置解析后只有`cuda.initial_allocation_cache_bytes`一项不同。关闭时6,374,152,448 B与44,859,568 B的分配分别18.763006/16.651861秒，合计35.414867秒，开启时无超过1秒的调用，与端到端减少35.784015秒相符。单条没有新增收益（开启66.33秒，关闭65.32秒），完整结果为库外 `video-pool-prepare-control-r1.results.json` / `.results-raw.json`。

后续 `video-pool-prepare-fresh-c2-r1` 复用同一二进制与25GiB预备：原图预热后，直接提交新参考的seed1101/1102两请求，未先运行新参考C1。2/2成功，P50/P99 **119.484310/121.210154秒**，0.989728 clips/min，仍超120秒1.210154秒；没有以先前缓存参考103.57秒替代此新参考结果。随后seed1102单独生成48.691691秒，完整产物已生成，跨批SHA核对及阶段记录取回中。Host+GPU200ms观测显存下界27,580,694,528 B；每格有2次NVIDIA查询超时，不能称完整峰值证据。下一步核对准备与编码成本；既有实物SEI已证实FastVideo使用x264 core165/ultrafast/CRF23/15线程，当前Ferrum core164/veryfast/CRF18/4线程，正式对照需消除这个差异。尚未启动正式重复统计或Windows阶段。

2026-11-15 05:27（Asia/Shanghai）：`video-pool-prepare-r1` 的实际GPU缓存/存活数据/额度边界检查、配置检查各1项通过，CUDA产品构建2分48秒。二进制SHA `bf54f1b391a6e826d01fa1c9c302dc3dcc4e3d901e756ebe586e74613e990932`；7文件源码SHA `f7baf32615699ae18622e8f7585f9df8a2557d5f51e0f3ffbf33354732996e9c`。首次启动编排因现有`/tmp`文件属主保护，在服务启动前退出；已保存失败日志，只修正编排权限后复用原二进制，未重复构建或生成。

25GiB预备实际耗时 **0.560081秒**，完成后private pool reserved=26,843,545,600 B、used=0；HTTP就绪等待19.028213秒另报，不能将其称完整进程启动耗时。4条HTTP与20GiB/16GiB预备的CLI共5条完整音视频均与对应父版MP4 SHA相同。冷C1 **74.200582秒**、新参考C1 **66.330146秒**、复用C1参考的首次C2 P50/P99 **101.852508/103.573787秒**，有效吞吐 **1.158201 clips/min**，错误0，8次真实B2；整轮没有超过1秒的私有池分配调用。最近FastVideo相同单轮筛选为新参考72.943707秒、C2 P99 120.577522秒/0.990288 clips/min。只支持筛选改善，尚无正式新参考C2或配对置信区间；Peak VRAM仍缺失，编码差异保留。

预声明的同二进制 `video-pool-prepare-control-r1` 已启动：仅关闭可选预备，其他28GiB/4096/maxseq2/输入/编码均相同，冷C1、新参考C1、首次C2共4请求；无额外构建。候选/原始日志为库外 `video-pool-prepare-runtime-r1.results.json` / `.results-raw.json`，控制组结果待完成。未宣布目标达成；Windows阶段仍在视频验收之后。

2026-11-15 05:05（Asia/Shanghai）：`video-swiglu-rows-runtime-r2` 的4条HTTP及20GiB CLI全部完成，完整音视频SHA分别等于对应父版；冷70.372880秒、新参考C1 65.818088秒、复用C1参考的首次C2 P50/P99 133.440688/135.045122秒，0.888375 clips/min，错误0。24次实际去噪前向为16次B1和8次B2。C2两次原生分配分别6,374,152,448 B / 15.803666秒和44,859,568 B / 15.720061秒，合计31.523727秒；不是Core锁等待。整轮BLAS只创建一次，不支持反复建句柄的假设。

相邻未修改父版 `video-swiglu-rows-control-nsys-r2` 已完成：profile新参考68.081892秒，对比候选68.587378秒；3452次SwiGLU累计GPU 0.790540秒，对比候选0.819347秒。先前接近减半的kernel耗时不能归因改动。**四个SwiGLU候选文件已按逐文件SHA恢复原region-args-r2字节**，原始报告、二进制和不采用记录全部保留，没有以候选65.82秒宣称已保留的代码获得此收益。

同机相邻 `video-swiglu-rows-fastvideo-r2` 四请求全部成功，冷96.869816秒、新参考C1 72.943707秒、相同参考C2 P50/P99 91.213485/120.577522秒，0.990288 clips/min；各MP4 SHA等于原FastVideo对照。两侧C2均未满足120秒门槛，Ferrum仍落后。仍为一次重复的诊断，C2参考已缓存；编码差异为Ferrum veryfast/CRF18/4线程，FastVideo ultrafast/CRF23/15线程，未做正式配对统计。Peak VRAM缺失；FastVideo冷/C1 Host采样各有一次失败，保留原错误，不能称完整内存峰值证据。

下一定向候选 `video-pool-prepare-r1`：在冷初始化完成、服务接纳请求之前，通过原Core账本临时申领、写入并退役一块私有池缓存。类型化配置 `cuda.initial_allocation_cache_bytes` 默认不启用；试验指定25GiB，未提高28GiB总预算，启动成本单列。容量不足或明确分配OOM只跳过可选预备；未知原生错误/清理失败保留原pool/context/完整额度，不盲重试。后续张量仍走原准入和初始化，缓存可被正常容量压力回收，不承诺物理显存预留。先做实际GPU额度/数据/退役验证，再仅测冷C1、新参考C1、首次C2及CLI；无收益即撤回，不启动全仓测试。尚未取得该候选性能结果。

2026-11-15 04:49（Asia/Shanghai）：`video-swiglu-rows-nsys-r2` 已完成，二进制SHA `74d821727f1f666dddace283a94fa72982cb212c1920760050f14af11aa9ccda`，两条完整音视频的SHA与对应父版相同。profile新参考68.587378秒，父版76.145129秒；3452次SwiGLU累计GPU1.558507→0.819347秒。但未修改的NVFP4 GEMM也从11.902880→9.629695秒、gather2.247451→1.240126秒，dense/sparse attention则基本不变，因此不能将全请求减少的7.56秒归因本次改动。SQLite SHA `18533450ed2d238f59b8d369d0e0af4bbff47dda0528381221b0214244363d27`，逐项对照及完整原始报告保存在库外同前缀文件。

未插桩 `video-swiglu-rows-runtime-r2` 已完成冷70.372880秒、新参考C1 **65.818088秒**，两条输出正确；首次B2的两个请求及CLI仍在执行。预先登记并串行排队：复用现成原二进制的 `video-swiglu-rows-control-nsys-r2` 相邻父版诊断，然后 `video-swiglu-rows-fastvideo-r2` 冷C1/同一新参考C1/复用C1参考的C2四请求，固定质量/容量/编码设置。复测用于排除未改内核的速度漂移，不重建、不跑全仓测试，也不把65.82与旧72.44的非同期差直接宣称正式胜出。仍缺完整GPU峰值、正式新参考C2与配对重复，整体目标未完成。

2026-11-15 04:38（Asia/Shanghai）：`video-swiglu-rows-r2` CUDA产品构建3分06秒完成；受管GPU SwiGLU资格通过，两种舍入profile、九种形状各两次，覆盖8/16 lane、2040/2048/2056列、14336列与65536行；有限值最大0 BF16 ULP，特殊值及原舍入反例、owner关闭/回收均通过。只跑相关fixture，未跑全仓测试。四文件源包SHA `ce697dedc78c7c609fdbe0244f14e5d7e03394e07a6e22c4afca268c97c388cc`，复用已验证的region-args-r2 native锁。`video-swiglu-rows-nsys-r2` 两条完整视频诊断已启动，尚无性能结论。旧平铺向量化的三个源文件已由未改动远端快照恢复，并与旧构建命令的逐文件SHA完全相同；恢复的导出归档明确不是原压缩包字节，旧测量/二进制/决策未改。

2026-11-15 04:27（Asia/Shanghai）：相邻旧版对照 `video-region-parent-nsys-r1` 完成，两条音视频有效，冷69.596288秒、profile新参考76.138598秒；新版对应69.859273/76.145129秒。主要GPU kernel数量和耗时基本一致，不能把之前未插桩enqueue变慢直接归因新上传。新上传1200次累计GPU仅0.000927秒，相关Host launch累计0.390804秒。400次prefix SDPA新/旧21.432485/21.439087秒，而FastVideo旧采样对应约22.96秒，此最大热点没有已知性能落后证据。两份SQLite的SHA及具体kernel/API汇总保留在库外 `video-region-adjacent-nsys-r1.analysis.json`。

下一候选 `video-swiglu-rows-r2` 针对更明确的逐元素差距：当前SwiGLU GPU累计1.558507秒（3452次），FastVideo旧DiT采样0.620652秒（3200次，范围差异明确）。通用算子对16字节对齐且宽度为8倍数的BF16行使用八lane读写，每CTA按行推进，移除逐元素64位地址除法；每lane保持原表达式、FP32及BF16舍入，其他宽度仍走原标量路径。复用原Core owner与同一CUDA模块，没有模型名特判或精度变化。新增现有GPU fixture的向量/部分CTA/真实宽度/65536行边界。先完成两种数值profile及owner资格，再只跑预定冷C1、新参考C1的相同Nsight诊断；有算子收益且完整输出正确才追加未插桩并发与CLI。初次CPU测试构建暴露fixture容量整数类型不匹配，已修复且Reference测试通过。核对历史发现旧 `video-swiglu-vector-r1` 已因整体性能撤回：旧内核是平铺vector索引，仍逐vector除法；本候选按CTA行所有权去除该除法。部署重名由远端已有目录保护在构建前拒绝，未覆盖旧源码；本机同名源导出归档曾被新导出覆盖，另记恢复记录，不混用实验身份。以新名继续相同的最小profile流程，尚无新GPU或性能结论。

2026-11-15 04:13（Asia/Shanghai）：`video-region-args-r2` 的2项真实GPU路由测试（含257 key块的跨参数包边界）通过；CUDA CLI构建2分52秒、GPU测试构建1分12秒，实际执行2.51秒，未跑全仓测试。4条HTTP和20GiB CLI共5条完整音视频均成功且与对应旧SHA逐字节一致，二进制SHA `65cd15ebcef16aa89b94f9434f4e74b3706c6beb4df238640c2faf585ace1b7e`。冷C1 82.494533秒，新参考C1 74.646769秒（父版79.431684秒、此前最佳74.121203秒、FastVideo72.436527秒）；C2 P50/P99 140.874218/143.344661秒、0.836849 clips/min，仍违反120秒SLO。这里C2复用了C1参考，属于首次B2诊断，并非正式新参考并发主表。注册调用归零，但6,374,152,448 B设备分配仍耗6.852827秒、44,859,568 B分配耗15.990965秒，B1/B2每步enqueue也从约4.86/10.15升至5.64/11.77秒，候选尚不能接受为性能提升。其他五个native原archive及Sage上游header保持相同；仅预期的边界头和新上传头不同。

20GiB真实容量诊断已通过：两个独立seed都被接纳并完整完成（73.383256/121.499469秒），8次B2候选需要24,306,260,720 B超过20GiB，实际16次forward全部B1；两条SHA分别等于对应单独运行对照。容量不足没有变成请求报错，但经验P99 121.018308秒仍不满足主表SLO。库外 `video-region-args-capacity20-r1.results.json` / `.results-raw.json` 保留报告、8条拒绝候选证据与16个实际B1记录。

新增定位只抓C1：`video-region-args-nsys-r1` 完成冷69.859273秒及profile新参考76.145129秒，均有效；CUDA/OSRT SQLite已生成，不把profile延迟混入未插桩主表。已预先排同配置、不改代码的 `video-region-parent-nsys-r1` 相邻父版对照，先排除硬件/时序漂移再归因新增kernel开销。视频性能目标未完成，Windows阶段仍未开始。

2026-11-15 03:54（Asia/Shanghai）：`video-region-args-r2` 的 CUDA native 编译、六包绑定与锁生成成功，Sage archive SHA `042b9d6e3ad85b5ed4c038077fe58af7a182c660f1986442c530e39dde9e2f3c`，新锁 SHA `b77017927918e1137ef5d418302299d93df7e9edd2774efe5d64b48d74fde0c6`，正在链接 CLI 与运行两项实际GPU路由fixture。预先登记后续容量诊断 `video-region-args-capacity20-r1`：使用同一二进制，仅把主表配置的设备预算从28改到20GiB，其他字段相同；max_num_seqs仍为2，两个独立seed1101/1102均须完成并等于各自B1完整音视频对照，查实际缩批/等待记录。它在主要四条HTTP及CLI全部完成后串行启动，不混入28GiB性能主表；报告保留120秒门槛及所有慢请求，正确性以完整完成而非计时通过判定。容量诊断配置SHA `1e8e00c22e20bfd2977d4fc5d28ebef2929bcc515cc392d90c0f81374b83929d`。

2026-11-15 03:49（Asia/Shanghai）：`video-region-args-r1` 尚未进入 GPU 测试或性能计时。目录与 Rust builder 编译成功，初次部署因 CPU 镜像缺少 Python 中止，已将辅助编排移至宿主机；随后 CUDA 编译明确失败在 NVCC 生成的匿名命名空间 stub 与 CuTe 名称冲突。原失败日志/凭据保留。`video-region-args-r2` 仅把新上传 kernel/参数类型改入具名命名空间，数值与上传逻辑不变；16文件源包 SHA `74cbdb5952d9f6a1401e31b09455475f0009668332baa790eee0282d6609a470`。新快照重建后的 GPU 正确性测试成功才允许计时，上一候选失败不会被当作请求生成错误或已有性能收益。

2026-11-15 03:44（Asia/Shanghai）：`video-demand-source-r1` 的 CUDA 构建、4 条 HTTP 与 20 GiB CLI 共5条生成全部成功，完整音视频 SHA 与各自对照相同。冷 C1 为70.111656秒，新参考 C1 为79.431684秒，首次新参考 C2 P50/P99 为132.446463/134.052083秒、0.894956 clips/min；C2仍违反120秒SLO。仅进一步延后B2来源准备没有稳定提速，最新结果不可替换先前74.121203秒的最佳单轮成绩为“已胜出”。新参考C1又有13.436957秒的4 KiB Host注册，C2注册16.011361秒及44,859,568 B设备分配14.554767秒；这些是本次已测调用，并非推断所有停顿同源。库外两个候选的 `.results.json` / `.results-raw.json` 已保留完成结果；峰值显存缺失、单轮限制不变。

下一候选 `video-region-args-r1` 新增独立的region kernel参数上传ABI，每包最多2048 B、按实际包数申报kernel次数，保留原完整几何校验、Core Host费用、设备scratch、stream/fence/隔离回收，并复用原选择和attention函数。原pinned-DMA ABI保留；没有把pageable内存直接交给异步DMA。14项Sage CPU合同测试与kernels CPU编译通过；新增真实GPU跨包几何与原Host/动态路由测试待跑，不以这些编译测试声称性能收益。16文件源包SHA `1cea6e2ecd0cd7972ba8c7a84ca7746fdf068877c58d6dd3bb100014de3ac788`，将重建Sage native archive并重新生成六包的共同catalog/锁，其他五包保留原编译凭据。5090编译/测试部署中。本机磁盘再次耗尽后仅删除24小时前可重建的Cargo debug archive约2.5 GB，源码/证据保留。

2026-11-15 03:25（Asia/Shanghai）：`video-demand-batch-r1` 的 CUDA 构建、6 条 HTTP 与 20 GiB CLI 均成功，7 条完整音视频 SHA 分别与既有输入/seed 对照相同；另有 engine 编译、共享权重 5 项、容量回退 3 项及格式/diff 检查通过。新参考 C1 从 78.662732 降至 74.121203 秒，仍比 FastVideo 72.436527 秒慢 1.684676 秒；冷 C1 为 76.913889 秒。首次新参考 C2 的 P50/P99 为 130.171378/131.776071 秒、0.910409 clips/min，2/2 成功但仍超过 120 秒 SLO，不能因 C1 改善称目标完成。真实中途加入的同一前向依次出现 evaluation3/0、4/1、5/2、6/3、7/4，随后后一请求以 B1 完成5–7；原请求与 seed1102 输出均逐字节等于各自 B1 对照。二进制 SHA `23165aa426c864c4c3b5edc9fd942123c873af9a5fae20568e6f6544b3395fea`，库外 `video-demand-batch-r1.results-raw.json` / `.mixed-progress.json`。CLI 正确性运行与下一候选 CPU 编译有重叠，不用于性能比较；单轮、Host-only 及缺少峰值显存的限制保留。

本次首次 C2 的慢调用转移到 4096 B Host 注册（16.057303 秒）及后段 44,859,568 B 分配（14.073798 秒），不能只按“大块 cudaMalloc”处理。5090 Windows 宿主机实测提交内存 84.419/85.679 GB，`vmmemWSL` 为主要提交占用；分页文件自动管理、C 盘尚余约 319 GB，瞬时分页读取为0。这只是压力线索，不证明分页/驱动根因；一份历史 Windows 查询来自办公室跳板机，已明确排除。下一候选 `video-demand-source-r1` 进一步将 B2 的来源/几何构建纳入同一按需准备，保留完整来源校验和 Host 容量等待，2 文件源包 SHA `123619c511b67b39e24064579da642b796e3a24bcb1a136f6b74d685e83e3545`，CUDA 构建2分钟通过，冷/新参考 C1、C2及20 GiB CLI共5条正在验证。小块几何元数据的无锁页上传仍处于设计核查，尚未修改 native ABI 或实施新分配器。

2026-11-15 03:02（Asia/Shanghai）：`video-demand-batch-r1` 将可选 B2 执行计划延后到实际加入请求时编译，B1 不再支付未使用计划的准备成本。冷 domain 仍检查两种 cardinality 的来源合同；新增计划只在已确认的成员变更安全边界通过原 Core shared-weights owner 挂接，挂起/恢复保留已准备计划。固定计划容量上限仍触发原缩批路径，其他错误不缓存成永久容量拒绝。本机 engine 编译、共享权重 5 项测试及受改文件格式/diff 检查通过；首轮本机检查因磁盘满失败，清理可再生成的旧 debug 测试二进制后重试成功。6 文件源包 SHA `2d735b839610e7226695e38b5f3fe988efc62ed869ddbee9d18b37c519d9b93c`，5090 CUDA 构建中。预声明冷 C1、新参考 C1/C2、中途加入两请求和 20 GiB CLI 共 7 条生成，尚无本候选性能或连续批处理结论；当前有效主表仍为新参考 78.662732 秒对 FastVideo 72.436527 秒。

2026-11-15 02:48（Asia/Shanghai）：`video-register-nsys-r1` 已完成，4/4 完整音视频有效，原/新输入 MP4 SHA 不变。profile 下新参考 C1 为 73.624598 秒，首次 C2 P99 为 129.521383 秒（0.926236 clips/min，仍违反 120 秒 SLO），不把插桩数据并入未插桩主表。已捕获两次 `cuMemAllocFromPoolAsync`：6,374,152,448 B 耗时 17.549403 秒，44,859,568 B 耗时 7.220781 秒；区间内该 trace 的 kernel/memcpy/memset 均为 0，同线程 OSRT ioctl 覆盖 17.548751 / 7.220723 秒。这是约 24.77 秒已定位的分配路径停顿，不再笼统称为 GPU 计算或 Core 分配前的锁等待；仍不能据此断言驱动缺陷、其他上下文完全空闲或所有历史停顿同源。原先新参考约 6.65 秒的 Host 注册在本次 profile 中仅毫秒级，未复现，Nsight 启动及插桩改变时序是明确限制。库外 `video-register-nsys-r1.results.json` / `.results-raw.json` 保留全部结果，远端 SQLite 为 92,028,928 B，SHA `baf764e29a71f9b9d409868e0d85467f2e6029c0636221c0f76a528ea7505342`；服务已停止，未增加新的分配器变体。

2026-11-15 02:37（Asia/Shanghai）：`video-program-index-r1` 的 1 项索引、117 项执行、17 项模型 wire、22 项计划编译检查通过，2 项 CPU profiling 默认忽略，CUDA 构建 3 分 22 秒完成。两条 HTTP 及 20 GiB CLI 全部成功，原/新输入 MP4 SHA 不变，B1/B2 计划 ID 与全部 602 条布局费用字段保持相同。冷 C1 为 80.190367 秒（上版 77.175069 秒），新参考 C1 为 78.662732 秒（上版 79.694305 秒）；新参考仍比最新 FastVideo 72.436527 秒慢 6.226205 秒。单轮混合结果不能证明正式胜出或冷启动提升；新参考 conditioning 至首次 handoff 从 21.527340 降到 20.693652 秒，冷请求该区间基本不变。二进制 SHA `6b4a38c08ff5260c7e7c8495806d73a4ad46ae41c0d9e2c11ba187b533825170`，库外证据 `video-program-index-r1.results.json` / `.results-raw.json`。峰值显存仍缺失，旧 C2 超限记录不变。

新参考再次记录 4 KiB Host 注册耗时 6.651427 秒。下一步已冻结 `video-register-nsys-r1` 四请求诊断：同二进制先冷 C1，再抓新参考 C1 及首次实际 B2（两请求）的 CUDA/OSRT 时间线。只定位 Host 注册和设备池分配是否重叠已记录 GPU 工作或卡在同线程系统调用；不把 profile 运行当未插桩性能比较，失败与 profiler 诊断全部保留，不新增 allocator 试配。视频目标尚未完成，Windows 后续仍排在视频验收后。

2026-11-15 02:22（Asia/Shanghai）：`video-composition-owner-r1` 的 10 项组合测试、3 项 FastH3 批处理测试及 CUDA 构建通过；2 条 HTTP / 20 GiB CLI 共 3 条视频均完整有效，原/新输入 MP4 SHA 不变，B1/B2 计划 ID 和 16 组大型布局费用不变。冷 C1 为 77.175069 秒，新参考 C1 为 79.694305 秒，较原 r4 累计缩短 38.710%。新参考 conditioning 至首次 handoff 从 25.012115 降到 21.527340 秒；去噪步间隔同时恢复为约 5.5 秒，不能将全部端到端下降归因于建图优化。二进制 SHA `f7697a29a545ef13f551546e2108ba0019e18c35ed12152a068672447a1c68a4`，证据为库外 `video-composition-owner-r1.results.json` / `.results-raw.json`。

考虑旧二进制也出现执行速率漂移，预先声明并紧接候选复测 `video-current-fastvideo-r1` 两请求，均完整成功：FastVideo 冷 C1 102.066928 秒、新参考 C1 72.436527 秒。Ferrum 新参考仍慢 7.257778 秒，目标未达成；新旧基线全部保留，未挑选最佳轮次，来源为库外同名 `.experiment.json` / `.results.json` / `.results-raw.json`。基线推理已再次停止，自启动仍 disabled。以上仍是单轮诊断、Host-only，C2 最新 129 秒超限记录不变。

下一候选 `video-program-index-r1` 为已验证且不可变的 ModelProgram 建立共享只读节点/状态/权重索引，消除编译和校验时反复线性扫描。索引不进入 wire 或 fingerprint，反序列化仍走原构造与校验，构造后无节点/状态/权重可变入口。新测试覆盖多块、排序、clone、wire 往返和拒绝外部注入索引；首编译发现测试 ID 辅助函数错误地统一要求 VNextError（NumericalProfileId 返回 String），已修正测试辅助函数，失败日志保留。截至 02:28，新索引测试 1 项、执行层 117 项、模型 wire 17 项、计划编译合同 22 项全部通过；2 项显式 CPU profiling 测试未运行，受改文件格式检查通过。6 文件源码已冻结（archive SHA `e9ae267622eaca470938ce86a80cee141b528ce8084f5e9d54f7c801752f9380`）；预先声明冷 C1、新参考 C1、20 GiB CLI 共 3 条生成，进入上传和 CUDA 构建。尚无此候选性能结论。

2026-11-15 02:03（Asia/Shanghai）：`video-joint-move-r1` 已完成远端 70 项执行检查与 CUDA 构建，2 条 HTTP 及 20 GiB CLI 共 3 条视频全部完整有效，原/新输入 MP4 SHA 均保持相同，16 组大型布局费用完全一致。冷 C1 90.285946 秒，新参考 C1 91.517575 秒；相对原 r4 130.027443 秒累计缩短 29.617%，仍比最近 FastVideo 75.369531 秒慢 16.148 秒，未完成目标。真实 B1/B2 联合选择已从约 4.05 / 8.60 秒降至 0.031619 / 0.030269 秒，计划 ID 相同。二进制 SHA `b186cf31c8bf132fa3b4d21ee039f700a81525cdab25b21dbbb5fcb228452b74`，证据为库外 `video-joint-move-r1.results.json` / `.results-raw.json`。

本轮去噪步间隔从旧轮约 5.5 秒变成约 6.4 秒，不能将全部端到端差异归于源码改动；预先声明 `video-joint-control-r1` 用未改的旧仪器版二进制仅重测冷 C1 / 新参考 C1 两请求。旧二进制冷 C1 为 102.392679 秒（此前 94.067349 秒）、新参考 C1 为 102.099897 秒；相邻候选新参考少 10.582322 秒。两版去噪 enqueue 中位数为 5.641049 / 5.636973 秒，步间隔均约 6.4 秒，支持当前差异来自准备开销且原去噪变慢不是本次求解器改动引入；执行速率整体波动的原因尚未确定。补充对照保留在 `video-joint-control-r1.results.json` / `.results-raw.json`。C2 未复测，仍保留上一轮超 120 秒记录；当前仍是单轮诊断、Host-only。下一候选 `video-composition-owner-r1` 让已验证 Stored composition 同时提供模型 family 和编译几何，通过原 Core 定义验证路径，减少同一配置反复建图；正在做定向一致性检查，尚未冻结或取得 GPU 收益。

2026-11-15 01:46（Asia/Shanghai）：`video-plan-profile-r1` 分段诊断完成，两条 HTTP 视频和 20 GiB CLI 全部成功、完整音视频解码通过，并保持对应原/新输入 MP4 SHA 不变；新参考 C1 为 95.568959 秒（一次诊断，不能把仪器版波动当作新增优化）。实测 denoising 的单/双请求计划中，联合 provider/storage 选择分别约 4.05 / 8.60 秒；内存规划已仅约 0.20 / 0.33 秒。二进制 SHA `7a11400b796e65c822a51f84d802dde490945ac6ef5db95ee1f957a80bd3c02b`。原始报告继承了较早轮次的部分说明字段，实际执行顺序以库外 `video-plan-profile-r1.experiment.json`、编排日志和各请求报告为准：只运行冷 C1、新参考 C1、20 GiB CLI，未重跑基线或 C2；完整原件及文件 SHA 保留。

下一候选 `video-joint-move-r1` 在共享求解器中让最后一个候选分支接管已消费的 frontier 状态，避免单候选路径逐节点复制整个累计约束和选择列表；不改目标函数、偏好、分支去重、容量或模型算术。穷举小图与独立完整赋值 oracle 对照通过，执行范围 70 项通过、0 失败、2 项诊断默认忽略；其中联合选择 CPU 诊断已显式运行，同一 4096 节点链从 1.937894 降到 0.005491 秒，选择指纹 `60fff79072d98280a1ae2022e1d783e2fa14277aeb7ae537d50dd8c790811f8b` 相同。源包 SHA `7cc93791660618c75c1e5e9463ab95f226bbd0db177ea64ad2506a24d206a8c3`，预声明同样只测三条生成，尚无该候选 GPU 收益结论；不重复已缓存且不执行该准备阶段的 C2 或全仓测试。视频目标未完成，Windows 外部反馈仍排在视频验收之后。

2026-11-15 01:28（Asia/Shanghai）：`video-order-index-r1` 的远端 69 项定向测试和 CUDA 构建已通过，四条实际视频全部有效并保持原/新输入各自 MP4 字节相同。冷 C1 为 95.827958 秒；新参考 C1 为 97.824146 秒，较上一候选再少 13.109 秒、相对原 r4 累计缩短 24.767%。仍慢于同轮附近 FastVideo 新参考 75.369531 秒，剩余约 22.455 秒。相同参考首次 C2 为 125.515202 / 129.042645 秒，经验 P99 129.007371 秒、0.929924 clips/min，8 次实际 B2，但仍超 120 秒。16 组大型布局费用与前候选全部相同；原生两次分配调用仍为 15.601 / 15.098 秒。新参考 conditioning 至首次 prediction handoff 还有 39.302 秒，正在补充通用计划构建分段时序以定位其余准备成本；不把这些混合间隔全部归为 CPU。二进制 SHA `e1164d98e5503eb57eec1a453fa84a19a8f6add2f8636ba0fde6b8deaba6fadc`，证据为库外 `video-order-index-r1.results.json` / `.results-raw.json`。仅单轮诊断、Host-only，尚无正式胜出或完整目标完成结论。

此前 `video-slot-neighbors-r1` 的 20 GiB `run` 已完成：generation exit0、ffprobe 与完整音视频解码通过，MP4 SHA 仍为 `d3de4e5a70933e54f2187f1277cc5274cc1a94f85b999348af028e8d511bd5a0`。下一次计划分段诊断保持已验证模型算法，追加一次 CLI 以覆盖最新共享 Core；不重跑全仓测试或已有 FastVideo 基线。

2026-11-15 01:10（Asia/Shanghai）：`video-slot-neighbors-r1` 同轮八请求诊断已结束，8/8 完整音视频有效。原 r4 / 本次候选冷 C1 为 129.795443 / 110.966090 秒；暖进程同一新参考 C1 为 130.027443 / 110.932872 秒，缩短 14.685%，两种输入的 MP4 均与各自旧版逐字节一致。FastVideo 同轮冷 C1 为 96.630179 秒、新参考 C1 为 75.369531 秒；候选仍慢约 47.2%，没有达到胜出目标。候选随后相同参考首次 C2 为 124.027196 / 127.554466 秒，经验 P99 127.519193 秒、0.940772 clips/min，均成功但违反 120 秒 SLO；日志确认 8 次实际 B2。该 C2 参考已缓存且没有同期 C2 基线，不能冒充新参考并发主表。每格一次重复、Host-only，峰值显存仍缺失；报告与原始文件 SHA 为库外 `video-slot-neighbors-r1.results.json` / `.results-raw.json`，完整原件仍在 5090。

这次优化保留：16 组大型布局费用记录前后全部相同，新参考 conditioning 完成至首次 prediction handoff 的间隔由 70.806 秒降至 51.495 秒，与端到端约 19.1 秒收益相符；这段仍混合编译、初始化和设备等待，不能全部称为 CPU 布局耗时。C2 另有 15.157 / 14.331 秒的原生分配调用，保留设备等待可能。20 GiB 的 `run` 资格已启动（PID 1199734），尚待完成，不冒称两个入口已全部验证。

下一候选 `video-order-index-r1` 为共享 completion-order 增加线性存储的完整 DAG 索引：依赖深度证明不可能有序的节点对，真实依赖组成的森林证明可达节点对，其余沿原 DFS；缺节点与环仍保留原行为。新增全小型前向 DAG 与独立 DFS 对照，定向执行检查 69 项通过、0 失败；同一 4096 节点 CPU 诊断进一步从 0.437679 降为 0.018965 秒，布局指纹与字节数相同。尚无这一候选的 GPU 性能结论。三文件源包 SHA `9c6a85c863065f71840050455a19f53b07322c0595d8096844be38e1325135a5`，预声明仅冷 C1、新参考 C1、相同参考 C2 共四请求，复用已测基线；等待上述 CLI 结束后才构建和测量，避免资源争用。

2026-11-15：按用户要求继续推进，连接故障期间改查已保存的真实模型日志与共享内存规划器。r4 冷单条中 conditioning 于约 12.75 秒结束，首次 prediction 于约 76.68 秒开始；多次固定 Step 布局计算的相邻日志区间为 6–16 秒，不能把这段全部归为 GPU 分配。`video-slot-neighbors-r1` 在 Core 中一次索引资源首末使用位置；对已证明内部有序的复用槽，利用传递性检查插入位置的相邻区间，保留原四种放置候选、费用比较、容量规则和无法建立顺序时的原逐对检查。没有修改 CUDA allocator 或模型算术。

本机优化编译下，Rust 4096 节点/跳连工作负载由 4.531353 秒降至 0.437679 秒（约 10.35 倍）；69,648,384 字节的槽预算和完整布局指纹 `e3b0532f3665e44fe8815b72d813229c621f162a6b9bd63f3d12ca80eb212134` 均相同。受影响执行/布局/容量测试 68 项通过、0 失败，性能用例显式独立运行；这是 CPU 单次诊断，尚不是 5090 视频加速证据。源包 SHA `04e6fdd1b691356606c44463c04256c5c61eeb4065c11a63f70f524cfa8fe160`，结果为库外 `.cpu-results.json`；沿用原 r4 父源并保留独立 row-write 测试导入修复。

已恢复 SSH 并复用远端已有新参考 PNG。31,784 字节冻结包已传输并核对 SHA，独立远端任务于 2026-11-15 00:36（Asia/Shanghai）启动。远端局部 runtime 检查 68 项通过、0 失败，CUDA 产品增量构建 3 分 24 秒完成；候选二进制 SHA 为 `ab694325a5d4e62d54f46be042e2c7ef79b53116ef7b9bbe54664763dd5e579e`。首次编排在输入准备阶段因旧 `/tmp` 文件属主不一致退出，尚未发起生成；已保留失败日志，仅修正本次暂存文件属主，从测量阶段续跑，没有重编译。

本轮依次测原 r4/FastVideo 各预热与新输入 C1，再测候选冷 C1/同一新输入 C1/首次 C2，共八请求；不并行构建与测量。全局 120 秒与完整成功门槛不变，首次 C2 不冒充已有同期新输入 C2 基线。原编排 PID 为 1140925，测量续跑 PID 为 1153500；启动与恢复记录为库外 `video-slot-neighbors-r1.pending.json` / `.resume.json`，观察连接断开不影响远端执行，须先检查原任务再决定续跑。以下 11 月 14 日尚未启动记录为历史状态。

2026-11-14 23:47（Asia/Shanghai）：新输入 C1 配对诊断已经冻结为库外 `video-fresh-input-r1.experiment.json`：两侧各原输入预热一次，再测同一新图/新提示词一次，共四请求，沿用 120 秒门槛及原质量配置。复用 r4 产品和既有 Rust benchmark，无新编译；新图内容 SHA 为 `b560782a09aaf232480a3aca72938e358ed6b1c15e84d53be3de3e51c3810197`，两侧使用相同 data URI 和显式 seed1101。该实验**尚未启动**：三次上传在 SSH 握手/传输中失败，最后一次已缩为 601,454 字节压缩包，现停止重复重试。启动命令从未执行；压缩包身份、完整请求复原校验、远端独立编排与恢复步骤保存在 `.pending.json`。连接恢复后先检查远端状态及包 SHA，不重跑已经完成的 r2 实验。本地撤回源哈希与定向 diff 检查均已通过，原工作区未改；当前没有正在运行的新测量，也没有性能达标结论。

2026-11-14 23:38（Asia/Shanghai）：连接恢复后直接取回原 r2 后台实验，未重复执行；编排 exit0、产品构建 2m49s、两项真实 CUDA 定向检查通过。四条视频全部完整解码且与原 MP4 SHA 一致，实际八步 B2，无共批容量拒绝。冷 C1 146.146799 秒，暖 C1 48.437364 秒，首次 B2 经验 P99 134.051559 秒：暖单条恢复，但共批仍未达到预声明 120 秒，**候选不采用，未建立比 FastVideo 更快的结论**。二进制 SHA `37eb0e99df807b6c7fcfb625652456962aa813a26580805d2df358b442efa9fc`；完整结果及撤回记录保存为库外 `video-native-reclaim-r2.results.json` / `.disposition.json`。

r2 暖 C1 的 107 次分配全部命中、无 native 增长；B2 为 139 次命中、27 次新分配，新分配调用累计 35.007691 秒，其中 736,985,088 B 为 13.087769 秒，14,610,176 B 为 18.542712 秒。调用区间可能包含等待先前 GPU 工作，不能全部视为可消除的 allocator CPU 开销。过早回收修复有效，但尚不足以改善双条端到端。已按冻结 SHA 撤回本轮实验性缓存源码及其 Core API，保留独立的 row-write 测试导入修复；默认私有池与既有 r4 产品路径保持一致。接下来复用已有 Rust 客户端和 r4 二进制，先测暖进程下新参考图、新提示词与 FastVideo 的小型配对诊断，核对真实输入的 conditioning 与去噪成本，不再对同一缓存命中负载盲调分配策略。该诊断不替代正式主表。

2026-11-14 23:27（Asia/Shanghai）：r2 新增整批预留边界的实际 CUDA allocator 测试 1 项通过，fresh Core 会话/数值/额度回收测试 1 项通过。最后观察到产品正在编译 `ferrum-cli`，**最终构建与四请求视频结果尚未取回，不计为通过，也不计为推理失败**。远端独立编排 PID 993902 已启动，日志与退出码分别为 `full-forward-r1/video-native-reclaim-r2.orchestration.log` / `.orchestration.exit`；应先读取这轮状态，不重复启动。

当前连接异常已分别核查：Windows `100.89.56.34:22`、WSL 内网 `172.31.40.3:2222` 均 TCP/SSH 超时；Tailscale 点对点探测及 `http://100.89.56.34:18194/` 页面也超时。控制面仍显示 Windows Online/Active，最后 handshake 为 23:22，这不能证明可达。另存不复用连接的 `ssh-reconnect-r5.conf` / `ssh-5090-reconnect-r5.cjs`，原配置未改。库外 `video-native-reclaim-r2.pending.json` 记录精确待办；可选 r1 原始日志下载中断文件保留为 `.download-interrupted`，已取回的完整结果 JSON/源包不受影响。用户已被询问机器/网络状态或新直连地址。候选仍未采用，默认私有池保持不变，目标未完成；若性能筛选通过，还需同二进制控制、CLI/容量和最终稳定里程碑资格。

2026-11-14 23:11（Asia/Shanghai）：`video-native-reclaim-r1` 完成，**不采用**：冷 C1 147.704939 秒、暖 C1 64.570005 秒、首次真实 B2 经验 P99 124.497008 秒，均未改善原私有池暖单条表现，B2 也未达到 120 秒。4/4 完整有效、原 MP4 SHA 一致，8 步真实 B2。三项定向资格及产品构建通过；二进制 SHA `acd9c9cb1d6abf8bf69f50133adb07bafd457191e6011a69a8f3b9dd5d5098c0`。SSH 观察会话退出 255，但远端脚本继续执行并完成全部单元，不记作推理错误。原始结果、分配事件及不采用理由已保存，未执行条件性的私有池复测或 CLI 交付检查。

暖 C1 实际 99 次缓存命中、8 次新分配，后者合计 15.906948 秒；其中 626,626,560 B 等待 2.169670 秒，99,401,728 B 等待 13.718817 秒，单独 allocation-stream 同步合计仅 0.000271 秒。发现候选的过早回收：Core 在 `dynamic_pool_set` 先预留整批增长额度，再逐块调用 allocator，尚未取用的预留额度不能等同于物理占用。r1 即使当前命中、不增加物理内存，也会按剩余额度淘汰后续可复用块。`video-native-reclaim-r2` 将这项回收限定到实际 native miss；命中仍消耗原新 permit，但不凭预留额度触发物理回收。补充实际 CUDA 边界用例：整批额度已满时，两块原 backing 都须保持可复用；其他真实增长压力与原生 owner 回收仍保留。两文件 overlay SHA `e8541a5ec3e412ee4e314ea7dc58f880fd22910692ed6730fb626af7e6864287`，父源为 r1；只重跑受影响 CUDA 资格及必要产品构建，沿用原四请求筛选，尚无 r2 性能结果。远端编排改为独立后台进程并记录退出码，避免 SSH 观察会话决定实验寿命。

2026-11-14 22:47（Asia/Shanghai）：候选 `video-native-reclaim-r1` 已写入并开始 5090 局部资格/产品构建，**尚无新视频性能结果，默认仍是私有池**。它在既有 CUDA runtime 下提供 typed `tensor_allocation=driver_cached`，由原 allocator owner 保留无活跃视图的 native backing；每次取用仍要求新的 Core permit，按共享设备账户在该准入后的剩余额度回收闲置块，原生 cuDNN/CUTLASS owner 准入前仍执行既有显存回收入口。没有再加固定 2/4 GiB 配额，也未改变模型算术、迭代器、B2 准入或容量回退。`DeviceCapacityOwner::available_bytes()` 只返回原账户观察值，不生成授权；独立 CPU 测试已通过，覆盖共享 claim、更小 ceiling、观察不推进 epoch、回收及 fail-closed。

本次定向测试编译暴露已存在的 `row_write_bf16` 测试缺少 `SemanticValue` 导入，已单独补齐；初始 exit101 日志保留，这项编译修复不计作分配器优化。冻结源 12 文件 SHA 为 `f1e6234ceb5e8f3be46319cf06ca0d5deab484e504be11e8e60d0f68e84fb28c`，原父源与新增文件清单保存在库外。预声明仅冷 C1×1、暖 C1×1、首次 C2×2，共 4 请求，28 GiB/4 GiB、原图片/seed/质量配置、Host-only；筛选要求暖 C1 ≤51.5 秒、C2 经验 P99 ≤120 秒、100% 完整有效，并核日志是否真实 B2。旧私有池控制只作筛选参考；候选通过后仍需同二进制配对和最终交付验证。没有新增全仓测试或调参矩阵。

2026-11-14 22:15（Asia/Shanghai）：最近归还优先的两轮缓存筛选均完成，**不采用；11 个候选源码文件已按冻结 SHA 校验后逐字节恢复为 r4 父版本**。没有撤回此前已验证的视频、连续批处理及容量回退实现。`video-allocator-lru-r1` 的 2 GiB 缓存：冷 C1 149.465104 秒、暖 C1 71.897751 秒、首次真实 B2 经验 P99 138.326872 秒。暖 C1 循环中可缓存工作区合计 3,144,373,920 B，超过缓存容量，626,626,560 B prediction 仍为 0 命中/8 次未命中。不能只因淘汰策略改变就声称获得复用收益。

`video-allocator-lru-4g-r1` 复用同一二进制，仅将缓存调至 4 GiB，仍计入原 28 GiB 总额：冷 C1 145.170320 秒、暖 C1 65.826099 秒、C2 经验 P99 134.700961 秒。暖 C1 的 prediction 已有 7 次命中/1 次未命中，但大于单条目 1 GiB 上限的 1,264,406,528 B 等工作区仍反复直接申请，分配调用总计 16.854809 秒；可能包含设备同步，不能全算作纯 CPU 开销。额外固定缓存预留压缩动态 backing 的可用额度，日志出现可选 B2 的 `MaintenanceLimit` 及原始 rollback，两条请求最终按 B1 完成；这轮不能称为真实 B2 性能。原私有池暖 C1 约 49 秒，因此没有总体收益。

两轮合计 8/8 视频完整音视频验证通过，全部保持原 MP4 SHA。局部真实 CUDA 缓存 5 项、独立进程隔离资源 1 项及覆盖两种 backing/两种缓存策略的 Core 复用 1 项通过，产品构建通过；未为失败候选追加 CLI、容量矩阵或全仓检查。二进制 SHA 为 `be1e61544538dad1a27ec32c24a0b5358a602e66986cd485452dd2caebf69330`，库外 `video-allocator-lru{,-4g}-r1.results.json` 与各自 `disposition.json` 保存结果及撤回记录。最后的测试容器已正常退出。以上仍为原参考缓存、每格一次、Host-only 诊断，缺少峰值显存和正式新参考配对重复。

对 FastVideo 差异的结论：当前部署用 PyTorch native caching allocator，且 HTTP generation lock 将 C2 串行为两个 B1；Ferrum 首次真实 B2 则经历约 3.20→6.39 GB 工作区增长。FastVideo 既有 trace 也记录过 6.383143 秒的 `cudaMalloc`，不能称完全没有等待；两个 trace 的捕获窗口不同。Ferrum 内部同负载对照将一次 6.39 GB 分配从 26.405063 秒降至 16.268 毫秒，证实 backing 路径影响该等待，但直接分配及外加固定缓存都引入了端到端退化，尚未解决问题，也未证明驱动缺陷。

下一候选须先解决工作区复用与容量维护的共同边界：由既有 owner 统一管理活跃与可回收的闲置 backing，容量压力下可回收闲置块，不能把额外固定缓存额度当作 PyTorch 复用机制的等价替代。保留共享 run/serve、真实 B2、原数值与容量不足时降批完成的合同。先依据现有 B1/B2 分配寿命证据审查设计，再进行一个有明确退出条件的实现；不继续盲扫缓存大小，也不把这一待实现方向写成已完成优化。

2026-11-14 21:46（Asia/Shanghai）：`video-allocator-cache-r1` 对照已完成，8/8 视频完整有效，但**不采用该缓存配置**。原私有池/直接分配加 largest 缓存：冷 C1 129.831481/142.196538 秒，暖 C1 48.954394/65.571769 秒，首次 B2 P99 139.345796/124.215350 秒，两侧 B2 均未满足 120 秒。缓存保留了冷阶段的 1,063,895,040 B 和 1,054,323,200 B 两块，之后都没有命中；626,626,560 B prediction 工作区 0 命中、16 次未命中。额外缓存没有消除上一候选的单条退化。没有运行条件性的 CLI/容量交付检查，原始结果和 disposition 已保存。

后续 `video-allocator-lru-r1` 只为现有有预算缓存增加 typed `native_allocation_retention=recent`，原 `largest` 保持默认。最近归还的条目排在末尾，先淘汰最早归还的闲置条目；活跃条目不在缓存中，不能被淘汰。仍为总额 2 GiB、单块 1 GiB、16 项并计入原 28 GiB。11 文件冻结 SHA `1049c62115c9f8dbb2f32003e2fe045c6486631d3d0e9b06e29b70d4ace8263c`；构建、局部真实 CUDA 验证与性能筛选现已完成，结果及撤回见本节 22:15 记录。为了避免每个缓存候选都重跑未改变的基线，本轮预声明仅候选冷 C1×1、暖 C1×1、首次 B2×2；参考上一轮完整配对控制，明确二进制不同、仅筛选，不以此宣布正式性能胜出。

2026-11-14 21:24（Asia/Shanghai）：`video-allocator-cache-r1` 复用已有 `CudaNativeAllocationReuse`，将直接驱动分配与原有回收缓存组合。候选固定缓存额度 2 GiB、单条目至多 1 GiB、16 条目，**从原 28 GiB 内扣除**；对照为私有池且缓存关闭，默认行为不变。没有修改 Core 分块规划器或申请权限。5 项缓存测试通过，另 1 项真实 Core 测试同时覆盖两种分配方式、新输入数值、缓存实际命中、重新计费及最终额度退还。首次同进程运行因故意隔离未知释放资源的用例保留原 owner，影响随后不同配置的用例；保留原 4 通过/1 失败日志，改为将故意隔离资源的测试单独进程执行后全部通过，无源码豁免。产品构建中，随后进行同二进制 8 请求对照；没有性能结论。11 文件冻结包 SHA `493ae0be4f505d19c5f0124fc84a860b55fd9ef931e1087af2e18a17ad023ede`，实验、构建重试和预声明对照均保存在库外。

2026-11-14 21:08（Asia/Shanghai）：`video-allocator-driver-r1` 对照完成，**不采用并已撤回全部 10 文件候选改动**。3 项真实 CUDA 所有权测试通过；同二进制 8/8 HTTP 视频完整有效，全部保持原 MP4 SHA。私有池/直接分配分别为：冷 C1 129.071229/142.673030 秒，暖 C1 48.437493/65.594074 秒，首次真实 B2 P99 137.109762/119.958854 秒、吞吐 0.875002/0.999986 clips/min。首次 6,390,058,240 B 分配由 26.405063 秒降至 16.268 毫秒；但直接分配时 prediction 动态工作区反复申请 626,626,560 B，单次约 1–2.7 秒，暖 C1 的全部分配调用合计 16.472404 秒（其中该大小合计 15.270678 秒），单条端到端退化约 35%。这些调用可能含设备同步，不能全部称为纯 CPU 开销。该方案没有同时满足单条和双条目标，不做额外 CLI/全仓验证；原实现、容量和数值路径均已恢复。

冻结源码包 SHA `b5717520f7eeb22a5860cd5bc62002ba895ec3e0155c22864e90a4c481969886`、二进制 SHA `f3495d7fe3242893a754acf8db416161e47d0da9b7fe5cff335dedf5d321c9dc`，命令、结果和撤回记录在库外 `video-allocator-driver-r1.*`。一次原参考缓存诊断、Host-only，双条候选虽刚好满足这一轮经验 P99 120 秒，不能称为稳定 SLO 或正式 FastVideo 胜出。已确认参考侧与 Ferrum 的差异同时包括分配缓存策略和实际 batch：不能用 FastVideo 串行 C2 的结果解释或证明真实 B2 的扩容行为。下一步审查现有 Core 的连续张量放置与 backing chunk 增长策略；固定块 profile 本身只改变分配粒度，不能假定它已经把一个连续大申请拆开。

验证节奏（用户 2026-11-14 明确要求）：性能筛选阶段只运行受影响的局部正确性/容量测试和必要的 5090 对照，不为每个候选重复全仓测试，也不让全仓构建占用实机筛选窗口。配置对照复用已验证的同一二进制。完整 workspace 与相关后端检查集中在最终实现稳定、准备交付的里程碑执行；新失败或明确的共享回归风险才扩大中途验证范围。

2026-11-14 20:38（Asia/Shanghai）：`video-pool-sync-r1` 对照已完成，**不采用并已撤回全部 8 文件改动**。3 项真实 CUDA 池测试及产品构建通过，8/8 视频有效且保持原 MP4 SHA。同二进制原策略/扩容前同步：暖 C1 48.699853/48.694871 秒；首次 B2 P99 140.120533/133.560400 秒，两侧均违反 120 秒。候选在大分配前的全上下文同步仅 8 微秒，随后 6,390,058,240 B 分配仍等 23.340158 秒，另一次 44,988,608 B 分配等 11.475 秒。已排除这次大分配在等待之前排队的模型 GPU 工作，但还不能断定底层驱动缺陷。库外 `video-pool-sync-r1.*` 保存冻结源码、命令、结果及撤回记录；仅为一轮原参考缓存诊断，未获得稳定性能胜出。

FastVideo 对照路径已核实：部署虚拟环境为 PyTorch 2.12.0+cu130、allocator backend `native`，既有 Nsight trace 记录的是 `cudaMalloc`，2317 次合计 11.147184 秒、最长 6.383143 秒；它并非没有分配停顿。这套服务的 C2 为单条串行，Ferrum 的 C2 为真实双条 forward，后者还经历约 3.20→6.39 GB 的连续工作区增长。后续只在同一 Ferrum 真实负载中比较私有池和直接驱动分配，保持 Core 准入、容量、buffer/fence 退役所有权与图像计算不变；不能仅凭跨实现差异认定根因或将串行测试称为批处理性能。

2026-11-14 20:14（Asia/Shanghai）：复用既有 Rust `cuda_allocation_phase_probe` 完成独立三分配器对照，无新增代码或构建。按私有池 reserved=18,454,937,600 B、used=15,192,262,512 B、B1/B2 工作区=3,195,029,120/6,390,058,240 B 设置 prime/anchor/small/large，逐字节清零、同步，28 GiB 显式逻辑预算。私有池/默认异步/直接驱动的大块调用为 13.243/20.599/15.665 ms，全部清理完成；聚合驻留量与大小本身没有复现原等待，不足以支持更换 allocator。既有二进制 SHA、帮助、完整 JSONL 和限制记在库外 `video-allocation-shape-r1.*`；没有声称其等于完整模型的碎片布局、原生 provider 额外分配或 PyTorch 缓存。

2026-11-14 20:03（Asia/Shanghai）：Nsight 首次 B2 时间线完成（r1 因新版本 launch 不接受 cpuctxsw 参数，在提交请求前失败；r2 参数移至 start 后完成）。仍为 r4 原二进制，4/4 请求完整有效。记录到 44,988,608 B 分配等待 4.589007 秒，其中同线程 ioctl 重叠 4.588945 秒，期间没有记录到 kernel、memcpy、memset 活动；这支持该次停顿在驱动调用内，不是等待已记录的模型 GPU 工作。原 6,390,058,240 B 大分配在分析器下仅 26.083 ms，说明观测改变现场，不能将本轮 B2 P99 106.341050 秒当成产品提速，也不能将小分配的因果证据无条件推广到原 24–28 秒大分配。Unified Memory 跟踪不可用、线程调度采样未开启；主要进程记录 663,225 个 CUDA 事件，子 FFmpeg 无 CUDA 事件是独立诊断。原始 nsys/sqlite 在 5090 库外，摘要 `video-pool-nsys-r2.analysis.json` 已取回。下一步复用仓库已有 Rust `cuda_allocation_phase_probe`，按实际 pool reserved/used 和首次扩批字节构造独立分配对照，不再新建重复测试工具。

2026-11-14 19:49（Asia/Shanghai）：`video-pool-growth-r1` 同二进制 AB 已完成，**不采用并已撤回全部 8 文件候选改动**。3 项真实 CUDA 池测试通过，8/8 视频完整有效且均保持原 MP4 SHA；原策略/扩容前归还闲置页分别为：冷 C1 129.312292/127.040585 秒，暖 C1 48.945927/48.701521 秒，首次 B2 P99 132.538724/135.817461 秒，吞吐 0.905172/0.883309 clips/min，两者 B2 均违反 120 秒。6,390,058,240 B 分配仍分别等 24.121393/26.677094 秒，44,988,608 B 分配等 9.591039/9.949338 秒；候选没有解决首次增长等待。只完成一轮诊断，Host-only，非正式胜出证据。原始结果、命令、冻结源码和撤回记录在库外 `video-pool-growth-r1.*`；当前工作树回到 r4 原分配器和配置，不留下无收益开关。下一步用现有 Nsight 记录同一请求的 CUDA API 与 GPU 执行时间线，区分分配内部耗时和等待在途设备工作的时间；不继续盲试参数。

2026-11-14 19:13（Asia/Shanghai）：隔离复测完成，Ferrum 6/6、FastVideo 5/5 均完整有效并保持各自原 MP4 SHA。Ferrum 仍使用 r4 同二进制、28 GiB/4 GiB、强制 4096 分块：冷 131.284436 秒，暖 C1 48.687655 秒；首次 B2 P50/P99 136.847160/138.577175 秒、0.865721 clips/min，违反 120 秒；第二次 B2 97.163580/98.767776 秒、1.214568 clips/min。16 步真实 B2；首次两次原生分配等 27.723377、11.600133 秒，合计 39.323510 秒。停止用户级图片服务没有消除等待，不将它认定为根因。

紧邻 FastVideo：冷 88.658474 秒；暖 C1 两条 P50/P99 62.354489/68.532309 秒、0.962238 clips/min；C2 两条 84.810576/112.519232 秒、1.061150 clips/min。均各一次重复、Host-only、原参考缓存诊断；Ferrum 首次 B2 仍超目标，不能拿第二次 B2 替换后宣称总体胜出。原报告留在远端，摘要含每份报告 SHA、逐请求有效性/资源与原生分配记录，库外文件 `video-isolation-recheck-r1.results.json`。

`video-nvfp4-k256-r1` 已停止推进并从当前工作树完整撤回，候选及父版本均另存库外。局部 5 项合同测试和 CUDA 原生库构建通过，但没有 GPU 数值或完整视频资格；全算子打包分别遇到镜像缺 Python、既有 CUTLASS 绝对符号链接挂载问题。该候选不是已采用的优化，当前继续使用 r4 原生算子锁。为集中解决首次 B2 约 39 秒等待，本轮不再并行展开计算内核优化。

2026-11-14 18:53（Asia/Shanghai）实验条件更正：发现 `seekee` 的**用户级** `ferrum-h3-image.service` 自 11 月 13 日 10:16:49 起仍在运行（PID 906466、NRestarts=0），尽管同名系统级单元为 inactive。本次实际 GPU 进程查询含该 PID；停止用户级单元后，空闲设备占用从 697 MiB 降为 84 MiB，计算进程列表清空。定义、状态和前后读数保存为外部 `gpu-user-service-isolation-r1.txt`。这次观察不能证明它过去每时刻的占用或认定它造成分配等待；此前该时段的性能记录不能声称已验证 GPU 独占，原始延迟与正确性结果保留。使用同一 r4 二进制和同配置开始新的隔离诊断（`video-isolation-recheck-r1.manifest.json`），Ferrum 冷/暖 C1、首次/第二次 B2，再顺序运行 FastVideo 冷/暖 C1/C2，共 11 个请求、每格一次；仍为 Host-only、原参考缓存诊断，不作正式胜出推断，无额外构建或单元测试。

2026-11-14 18:49（Asia/Shanghai）：`video-ffn-row-write-r4` 已完成本轮资格：局部组合器 8 项、FFN 5 项及 CUDA 产品构建通过；HTTP 8/8、20 GiB `ferrum run` 1/1 完整音视频解码成功，全部原 Ferrum MP4 SHA 一致。没有重复全仓检查。显式 `maximum_plan_nodes=8192` 时，可选 B2 在 8200 节点被类型化拒绝，已编译 B1 保留，两条请求串行完成，P99 140.672008 秒，满足独立预声明的 300 秒容量目标；证明容量回退而非只扩大默认值。新默认节点预算 16384 同时作用于 run/serve。

正常 28 GiB、强制 4096 分块的四格各一次：冷 C1 124.992576 秒；暖 C1 48.936751 秒，与旧 concat 的 48.947927 秒基本相同；首次 B2 的 P50/P99 133.064954/134.793207 秒、0.890019 clips/min，违反 120 秒目标；同进程第二次 B2 为 97.181846/98.787778 秒、1.214321 clips/min，观察值满足 120 秒。第二次 B2 单元在 r4 正常服务启动前已加入声明，不与首次混合。16 步真实 B2 均设备内交接；首次 B2 两次 `cuMemAllocFromPoolAsync` 分别等 24.594934、10.759839 秒，合计 35.354773 秒，接近两格 P50 差 35.883107 秒；驱动内部原因仍未证明，不能把预热差异归为算法提速。

结论：逐块写入降低计划临时容量，run/serve 容量回退已验证，本轮没有单请求速度收益，默认仍按容量需要才分块。原始报告、失败 r3、CLI 终态与完整汇总保存在外部 `video-ffn-row-write-r4-evidence/`、`video-ffn-row-write-r4.results.json`。Host RSS 峰值已记录，GPU 峰值仍缺失；一轮诊断不能证明正式胜过 FastVideo。下一步对照 FastVideo 的工作区复用与融合实现，先处理首次扩批分配等待及已测执行热点；不扩大无关验证。

2026-11-14 18:20（Asia/Shanghai）：4096 行同配置对照的旧 concat 版 4/4 完整成功：冷 137.869961 秒，暖 C1 48.947927 秒，暖 C2 P50/P99 131.076472/132.806267 秒、0.903331 clips/min。8 次真实 B2 均设备内预测交接；两次原生分配分别等待 23.130180、9.452891 秒。新 r3 的完整管线 HTTP 0/1、CLI 0/1，在可选 B2 计划准备时失败，尚无新视频性能数据；不将孤立 FFN 资格算作集成通过。旧 B2 图 8107 节点，新的逐块写入多 100 节点，超出原固定 8192 上限。

r4 正在修复规划容量边界：通用图组合返回带实际节点数/上限的类型化容量拒绝；可选 B2 超限只保留已编译 B1，worker 通过原 `Ok(None)` 准入协议推迟加入，已有请求继续推进并最终串行完成。共享 run/serve JSON 配置新增正整数 `maximum_plan_nodes`，默认 16384；这是规划复杂度预算，不是 GPU 字节配额。低上限仍须进行实机双请求回退验证，不能只调大默认值后计通过。服务日志补充原失败诊断，客户端仍保留现有错误边界。只增加实际节点边界局部测试，继续复用已有 CUDA 算子资格；r4 尚未构建或计通过。

2026-11-14 17:59（Asia/Shanghai）：直接行写入候选推进到 `video-ffn-row-write-r3`。r1 CUDA 编译暴露枚举数组长度遗漏，r2 CUDA 注册暴露能力声明遗漏，均保留原失败并修正；r3 独立 FFN CUDA 构建与四个实机单元完成，各连续执行两次。129 行整块/128 分块与独立 dyadic oracle 逐位一致；真实 block0 的 8193 行整块/4096 分块 SHA 相同，且与之前 concat 结果相同，静态来源未重复读取，关闭并释放输出后 Host payload 为零。真实样本计划临时字节从旧 concat 的 605,602,816 降为 561,533,952，整块仍为 947,093,504；这是计划容量而非实测 GPU 峰值，也不证明端到端提速。局部行搬运 10 项、FFN 图/来源 5 项通过。17 个实现文件与冻结源 manifest 一致；产品候选正在构建，同配置 4096 行 concat/直接写入的一轮 AB 诊断已准备，尚无完整视频结论。原始证据在外部目录 `video-ffn-row-write-*-r3.json` 与 `video-ffn-row-write-r3.qualification-summary.json`。

2026-11-14 17:41（Asia/Shanghai）：16K 配置筛选完成、4/4 原输出 SHA 一致，但不采用：冷 96.087316 秒、暖 C1 56.264437 秒、暖 C2 P50/P99 128.788318/131.257829 秒，0.913879 clips/min，C2 违反 120 秒。8 步 B2 均获设备内预测交接；首次 6,390,058,240 B 分配等 8.300469 秒，另一次仅 44,988,608 B 的分配也等 8.067845 秒，因此不能把等待简单归为单块过大。各格一次、Host-only，状态波动尚在；不把全部耗时差因果归于分块参数。

重新对照本地 FastVideo `models/dits/minimax_h3.py` 184–191 行：参考实现分配一个输出并逐块写回；Ferrum 的旧分块图保留各块结果再树形 concat。候选 `video-ffn-row-write-r1` 引入通用 BF16 行写入合同：完整初始化的 base、输出严格同 backing、更新源不重叠、行范围检查、位级搬运；Reference/CUDA 均遵守同一合同。FFN 用显式初始复制与 MustAlias 更新链替代末尾拼接，权重、GEMM/SwiGLU 和量化舍入不变，run/serve 共用。不是借原生指针绕过 Core，也未引入隐式分块循环。

局部 Reference 位级/别名测试通过。首轮 FFN 测试 4 通过、1 失败，发现 `row_start=0` 被错误的正整数属性规则拒绝，已在远端构建前修正并复测；失败日志保留。15 文件最终冻结包 SHA `f45e57fcbb163a37cbbfac27480ae06fb2962fb5fe3eeb06ac677297d72aefa1`，父源为 decode-overlap-r2；CUDA 编译、真实 FFN 数值/容量资格与整视频性能尚未计通过。此候选只做相关局部检查，不重新运行全仓测试。

2026-11-14 17:15（Asia/Shanghai）：`video-decode-overlap-r2` 已恢复真实 B2，日志有 8 次联合 forward；HTTP 冷/暖/20 GiB 共 7/7、CLI 1/1 输出完整有效，均为原 Ferrum MP4 SHA。暖 C1 P50/P99 **47.684342/47.691120 秒**，1.258268 clips/min；暖 C2 P50/P99 **139.388495/141.115604 秒**，0.850154 clips/min，**违反 120 秒目标**。B2 准入后首次申请 9,366,957,312 B 动态 backing，`cuMemAllocFromPoolAsync` 实际等待 **39.415766 秒**后成功；不是永久死锁。显存池增长的驱动内部原因尚未证明。B2 仍使用 Host 预测交接，首条解码可选配额不足时正常回退 inline。20 GiB 冷双请求串行完成，P99 173.984571 秒，零生成错误，满足独立 300 秒容量目标。

同一 r1 二进制的解码开关对照：inline 暖单条 48.727169 秒，overlap 暖两条中位 47.693792 秒；解码收尾 Host 区间约 4.53 → 3.43 秒。只支持约 1 秒收益的小样本诊断，尚非正式 FastVideo 胜出。r2 Mac workspace/Metal check、Clippy、12 项 engine 测试通过；17:20 回收 Linux 完整 workspace 结果：343 项汇总、8472 通过、0 失败、90 忽略，随后必需 CUDA compile check 通过（29.99 秒），容器 exit0。资源为 Host-only，GPU 峰值缺失。原始报告保留；冷/暖 HTTP provenance 的 binary 字段因编排误写旧 r1 清单而仍为占位，真实执行二进制由运行命令和 CLI 归档 SHA 独立确认，该证据缺陷不改写成预先完整。

下一项固定诊断 `video-ffn16k-overlap-r1` 复用 r2 二进制，仅设 `feed_forward_partition=always`、`feed_forward_chunk_rows=16384`，容量仍为 28 GiB/4 GiB Host。预声明冷 C1 n1、暖 C1 n1、暖 C2 n2，各一次；检查较小工作区能否避免扩批长等待并让设备内预测交接共存，再据结果决定容量选择策略。命令、清单和原始结果均在库外部署资料目录。

2026-11-14 16:57（Asia/Shanghai）：解码重叠 r1 找到了明确的准入设计问题：新增回读配额被预留在整个请求生命周期，28 GiB C2 的 Host 使用在尝试扩批时已承诺约 3.953 GB，4 GiB 总额度不足以再取得 B2 所需增量；日志反复尝试 B2 后仍执行 B1。该格虽 2/2 正确、P99 94.583962 秒、1.262424 clips/min，实际是串行，**不计真实联合 forward 或连续合批提速证据**。r1 暖 C1 两条为 47.695868 /47.691716 秒，冷 79.701335 秒；20 GiB 冷 C2 P99 172.935544 秒、2/2 正确。合计 HTTP 7/7、CLI 1/1 原 Ferrum SHA 相同；资源为 Host-only，未补全 GPU 峰值。

`video-decode-overlap-r2` 将可选配额移至完成去噪后的解码 fence：原请求与合批 Host claims 恢复；此时从原 parent 原子申请独立的两个回读 backing，不能取得就使用原已准入 inline 路径。零长度通道约束仍是两个回读共存；独立授信随组件 phase 进入原生初始化、执行和关闭错误的 custody，不提前退款，也不借用其他请求的授权。12 项 engine 局部测试通过，覆盖实际 Host 配额、原合批 grant 保留、晚申请回退、双回读上限、错误与退款；workspace check 通过。7 文件增量包 SHA `6979c56f87dfd20fe58e3c54d9595c5ad4ad4d1af89b3d0d352d8fe95c105e4d`。r1 同二进制 inline C1 对照与 r2 CUDA/实机验证排队执行，未提前宣称 r2 恢复合批或胜出。

2026-11-14 16:35（Asia/Shanghai）：当前阻塞是执行效率，非连接、基线或模型文件缺失。代码中的主要差距是 VAE `blocks/program.rs` 将偏置、残差、RMS、cast、量化拆开执行，以及 `decode/video.rs` 原来的 GPU→CPU 拼接串行顺序；现有 RMS/RoPE 融合算子的舍入语义不同，不能直接替换。主注意力的历史同机 GPU 诊断已接近 FastVideo；双请求联合 forward 仍未带来吞吐收益，尚未完成目标。

相邻 FastVideo 补测完成，5/5 完整音视频有效、原 FastVideo SHA 一致：冷 C1 96.205952 秒；暖 C1 两条 65.196165 /58.303549 秒，P50/P99 61.749857/65.127239 秒、0.971659 clips/min；暖 C2 两条 58.885413 /118.203962 秒，P50/P99 88.544688/117.610776 秒、1.015194 clips/min。各格 1 次重复；Host-only 诊断、无 GPU 峰值，暖 C1 另有一次 VmRSS 不可用，资源证据不完整。不能把该 C1 中位当作稳态值，也不能与历史约 48 秒混成同轮比较。库外证据 `fastvideo-adjacent-r1.results.json` 保留原始报告路径。

`video-swiglu-vector-r1` 拒绝并已恢复三个父文件：CUDA 两 profile、七 shape 资格通过，HTTP 5/5 与 CLI 1/1 输出有效且原 Ferrum SHA 相同，但暖 C1 P50/P99 56.012693/56.019630 秒，暖 C2 P50/P99 **133.726047/136.565799 秒**、0.878324 clips/min，违反预设 120 秒。运行状态波动仍存在，不将全部退步因果归于向量化；没有足够证据保留该候选。决策与实测在库外 `video-swiglu-vector-r1.decision.json`、`.results.jsonl`。

`video-decode-overlap-r1` 独立于该已拒绝内核候选。共享 run/serve 的类型化配置 `video_postprocessing` 默认 `inline`，可选 `overlap_if_capacity_allows`：GPU 始终使用原串行 actor，仅将已经完成 fence 的 Host 回读交给 CPU 工作者；零长度通道限制为当前处理块和下一回读块共存，后一块单独计入 Host Output 配额。物理容量不足先回退 inline，再缩小 tile group；临时占用继续等原 admission。错误先 join CPU，保留原生错误及 owner；线程启动失败在提交前回退 inline。局部 engine 11 项、spatial 8 项通过（1 忽略），Mac workspace/Metal check 与 Clippy 通过。13 文件源已冻结，CUDA 构建及 HTTP/CLI/20 GiB 实机验证进行中，未计性能收益或完整资格通过。

2026-11-14 16:10（Asia/Shanghai）：相邻旧/新/旧诊断中，`video-claim-digest-r1` 暖 C1 先为 **48.705952 秒**，`video-cold-pool-policy-r1` 为 **56.749226 秒**，再次切回同一旧二进制为 **56.252366 秒**；各 n1、1 次重复。对应冷请求为 84.719503 /90.257201 /90.029468 秒。不能将最初约 16.5% 的差值单独归因于容量修复；两版四个热点内核的内嵌 PTX 完全相同，源码差异也仅为已冻结的 10 个容量策略文件。历史 Nsight 中主 cuDNN 注意力耗时近似，变慢集中于搬运类内核，但具体运行状态原因尚未确定。此次 Rust bench 仅采 Host，外部持续 NVML 查询仍有超过两分钟的缺口，不能证明 GPU 时钟/温度或完整峰值。全部 6 份完整音视频有效、原 MP4 SHA 一致。库外证据为 `video-adjacent-binaries-r1.results.json` 与相邻服务日志；同期 FastVideo 基线正在补测。

空间 BF16 舍入微优化已撤回：同机 CPU 拼接中位约 0.632→0.545 秒，整体只省约 0.087 秒，未做完整推理收益声明，不为该收益保留额外数值实现。新的 `video-swiglu-vector-r1` 仅将 packed SwiGLU 对齐读写改为八元素搬运，逐元素指数/除法及两个舍入 profile 不变，奇数宽度保留标量路径。真实 CUDA 边界资格与产品构建已排在基线完成后，尚未计通过或宣称提速。

2026-11-14 15:46（Asia/Shanghai）：`video-cold-pool-policy-r1` 已完成容量修复资格：HTTP 5/5 与 CLI 1/1 完整音视频有效、原 seed1101 MP4 SHA 相同、零生成错误。冷请求 91.288123 秒；暖 C2（n2、1 次重复）P50/P99 **115.466220/118.429745 秒**、**1.012741 clips/min**；20 GiB 冷 C2 成功回退为 B1 顺序执行，P50/P99 **154.995735/179.201090 秒**、0.667798 clips/min，满足该容量单元预设 300 秒。本候选没有测暖 C1，不能沿用父候选的 48.81 秒。最新 C2 比父候选慢，尚未定位原因；配置内容完全相同，服务分阶段日志显示 B1 DiT 每步约 5.4→6.18 秒、B2 约 11.2→12.76 秒，视频解码也变慢，正在做同机相邻对照。

全量 Linux `cargo test --workspace --all-targets` 退出 0，343 个测试汇总项合计 8466 通过、0 失败、90 忽略；必需 CUDA CLI 特性 check 通过。Mac workspace/Metal check、Clippy（允许 warnings）、fmt/diff 通过。旧图片容量失败已修复且原日志保留。二进制 SHA `54452a2f49f78412885aa3e0f2455142f89d27f55f57c2cd96add748969352c5`，冻结源为库外 `video-cold-pool-policy-r1.source.json`。GPU 查询在冷/C2/20 GiB 单元分别失败 2/1/1 次；VRAM 观察峰值 21,411,921,920 /28,827,451,392 /20,036,190,208 B，RSS 37,614,743,552 /37,807,894,528 /38,841,139,200 B，仍只是采样下界，资源证据不完整。未建立超过 FastVideo 的结论，长视频/参考视频音频与 768p 资格仍待完成。

2026-11-14 15:14（Asia/Shanghai）：`video-claim-digest-r1` 暖 C1 P50/P99 为 **48.814155/48.943087 秒**、1.229144 clips/min；C2 为 **101.094765/103.317156 秒**、1.160962 clips/min，各 n2、1 次重复。冷请求 82.702112 秒；20 GiB 两条请求成功串行回退，P50/P99 154.464812/178.660166 秒，满足该单元预设 300 秒。HTTP 7/7 有效、零生成错误，另有 CLI 完整音视频解码通过，8 份 MP4 都与原 seed1101 SHA 相同。仍未证明超过 FastVideo 稳定约 48 秒和 C2 1.245904 clips/min；GPU 查询仍有超时，资源证据不完整。原始报告为库外 `video-claim-digest-{cold,warm,pressure-c2}-r1.report.json`，CLI 证据归档为 `video-claim-digest-profile-r1.evidence.tar.gz`。

本机 workspace check、Metal check、Clippy 通过；远端全量测试在 engine 库出现 7 项图片原生容量回退失败（928 通过、1 忽略），后续 CUDA 检查因此未执行。单项在本机也复现，不能归为并行测试噪声。根因是前轮把冷初始化统一改为只装静态资源，移动了图片消费者依赖的容量拒绝边界。正在修正为类型化 `ColdPoolInitializationPolicy`：既有调用默认在静态上传前取得初始动态池；视频显式在执行准入时取得动态 backing，继续使用原账本、回滚和恢复 owner。两种模式的真实容量测试已通过；组件生命周期 20/20、图片服务边界 56/56 通过，覆盖原 7 项失败。本机 workspace check、Metal check、Clippy 也通过；旧失败日志保留。冻结源 `video-cold-pool-policy-r1` 的 CUDA/实机/完整 workspace 资格仍待完成。

同一二进制切换 `feed_forward_partition=always` 的筛选已经完成：冷 135.604650 秒，违反预设 120 秒；暖 C1 P50/P99 50.467071/50.467543 秒、1.188887 clips/min，C2 106.514292/108.608124 秒、1.104455 clips/min，均 n2、1 次重复。5/5 输出完整有效且原 SHA 相同；资源采样仍不完整。完整分块继续只用于容量回退，不采用为默认性能路径；保留 `if_capacity_needed`，不把降低显存占用等同提速。报告为库外 `video-claim-digest-chunks-{cold,warm}-r1.report.json`。

2026-11-14 14:48（Asia/Shanghai）：`video-ffn-auto-r1` 冷/C1/C2 与 20 GiB 双请求合计 **7/7 有效、零生成错误**，均完整音视频可解码且原 seed1101 MP4 SHA 相同。28 GiB B1/B2 选择完整 FFN：冷 80.466275 秒；暖 C1 P50/P99 49.581872/49.699366 秒、1.210110 clips/min，C2 101.352713/103.574579 秒、1.158078 clips/min，各 n2、1 次重复。20 GiB 两条回退成功，P99 183.393273 秒，满足该容量单元预设 300 秒。仍未超过 FastVideo 稳定约 48 秒与 C2 1.245904 clips/min。

冷/C1/C2/压力 GPU 查询分别失败 2/1/1/3 次；暖 GPU 仅 4/5 次成功采样，最大间隔约 123/113 秒，Host 同样有联合采样间隙。VRAM 观察峰值分别为 21,411,921,920 /21,411,921,920 /28,827,451,392 /20,036,190,208 B，均为下界，资源证据不完整。完整报告为库外 `video-ffn-auto-runtime-pressure-r1.reports-final.jsonl`。

下一候选 `video-claim-digest-r1` 复用不可变物理 claim 的成员摘要，并将每次分配的 owner/代际/范围/初始化等完整字段纳入版本化的嵌套摘要；真实相等、成员、lease、容量校验不变。本机 runtime 1018 通过、5 忽略，新增身份/代际/投影敏感性测试通过；workspace all-target check 通过。CUDA 构建通过，二进制 SHA `62a16306db3b328fdd52895a16f245ea135e972caaac929930900f71a810a693`；首条实际去噪的 backing claim 从父候选约 40 ms 降到约 3.2–3.8 ms，尚未形成完整端到端结论。28 GiB、20 GiB、CLI profile 与最终 Linux workspace/CUDA 检查已按顺序提交，结果待回收，不能提前计通过。

2026-11-14 14:29（Asia/Shanghai）：固定 4096 行分块在 28 GiB 下出现性能退步：冷请求 133.827606 秒，超过预设 120 秒；暖 C1 P50/P99 53.985775/53.985991 秒、1.111398 clips/min，C2 109.394003/111.615009 秒、1.074688 clips/min，各 n2、1 次重复。5/5 生成输出完整音视频有效、原 MP4 SHA 一致、零生成错误；GPU 查询仍有失败，资源证据不完整，不能宣称全项通过。原整体目标尚未完成，FastVideo 后续稳定单条约 48 秒、C2 1.245904 clips/min。

`video-ffn-chunks-r3` CLI profile 与完整音视频解码通过，MP4 SHA 仍为 `d3de4e5a70933e54f2187f1277cc5274cc1a94f85b999348af028e8d511bd5a0`。DiT GPU 44.021955 秒、8 次调用跨度 48.982846 秒；步骤间隙 1.714719 秒的 851 个 Running CPU 样本中，411 为 JSON 字符串编码、280 为 SHA256。调用栈指向 `LogicalBackingSliceAllocationEvidence` 对同一共享物理 claim 成员列表的逐投影重复编码。这是诊断证据，不替代暖 HTTP 主表；正在修改不可变身份摘要复用，保留实际 owner/代际/范围校验。

新 `video-ffn-auto-r1` CUDA 构建通过，二进制 SHA `6ac6588f92d359b44e81df5a7c944d535251cdc35590bab36efaf61a94e368e7`。共享 run/serve 配置 `feed_forward_partition=if_capacity_needed` 根据现有设备账本检查完整冷峰值，足够时使用原整行图，不足时使用声明的 `feed_forward_chunk_rows=4096` 图；该预测不签发容量 permit，运行时仍走原准入。28 GiB 日志已证明 B1/B2 选择整行图，首轮分阶段计时显示 backing claim 约 40 ms，其他准入阶段远小于它。28 GiB 冷/暖以及 20 GiB 回退测试正在顺序运行，尚未计通过；本机 Reference 组件 19 项与 engine/CLI all-target check 通过。尚未实现继续缩小分块或覆盖所有长视频容量边界。

2026-11-14 14:04（Asia/Shanghai）：分块 FFN 的实际 CUDA 资格通过：129 行独立 dyadic oracle 的整块/128 行分块均精确，8193 行真实 block0 的整块/4096 行分块输出 SHA 相同，各连续执行 2 次，短尾块覆盖到 1 行；真实 FFN 计划工作区 947,093,504 → 605,602,816 B，静态字节不变，关闭后 Host payload 为零。诊断首轮因重复构造同一个 CUDA 原生模块触及所有权配额而失败；改为一次声明完整操作集合后通过，没有增加模块配额。`source-video-ffn-chunks-r2` 只修测试规范化比较与诊断注册；产品二进制仍是 r1，SHA `39e95bdfbf46baaf3a736b0e9422eafb064eba5e57ff061da42142ae7a7270b6`。

20 GiB 分块首轮仍有 2 条失败：B1 5657 节点已经满足显存预算，但超过旧配置的 4096 条 Host 命令记录上限，整批尚未提交即拒绝。按允许 8192 节点的图规模，将显式 `cuda.maximum_commands_per_submission` 与 5090 默认预设调整为 16384，包含上传/初始化命令；**没有增大 20 GiB device 上限、Host payload 上限或降低后端预留**。这属于配置的 Host 记录容量调整，尚未实现跨提交的命令分段。

随后 `video-ffn-chunks-pressure-r2` **2/2 完成、零生成错误**，两条均完整音视频有效、原 seed1101 MP4 逐字节一致；返回 120.465264 / 173.622035 秒，P99 173.090467 秒，满足本容量单元预设 300 秒。B2 计划超过 20 GiB 后使用 B1 顺序完成。GPU 采样 334 成功/3 失败，Host 337 成功/0 失败，最大联合采样间隔约 63.9 秒；VRAM 观察峰值 20,036,190,208 B、RSS 43,226,537,984 B 均为下界，资源证据仍不完整，不能把生成通过写成全部检查通过。该冷容量单元不并入 120 秒暖态性能主表。28 GiB 固定配置的冷/C1/C2 对比正在运行。

本机最新组件生命周期 19 项、execution component 15 项、FastH3 135 项通过（2 项忽略）；workspace/Metal all-target check、Clippy（允许 warnings）通过。完整 workspace tests 与必需 CUDA 特性检查待最终源冻结后补齐；默认命令容量改动的 r3 已冻结，未在 GPU 运行期间并发构建。

2026-11-14 13:50（Asia/Shanghai）：SSH 已恢复，确认 `video-device-handoff-r2` 的 CLI 音视频完整解码与原 seed1101 MP4 SHA 一致；该候选常规 HTTP、晚加入补跑和 CLI 合计 8/8 有效输出。先前晚加入编排中断的两条失败继续单列，不抹去。新 profile 确认 DiT 隐状态转为设备内复制 2,505,043,968 B，去掉此前同量的 D2H/H2D；DiT GPU 43.774294 秒、8 次调用跨度 45.166183 秒，视频 GPU 2.315058 秒、7 次跨度 4.011261 秒，均为诊断区间，不能代替暖 HTTP 对比。原性能结论不变：尚未超过 FastVideo。

断线期间实际执行的 `video-device-pressure-r1` 20 GiB 双请求测试为 **0/2 成功、2 条生成失败**，服务初始化工作区时停止。完整 B1 计划静态加动态最低需要 19,686,492,816 B，后端固定预留 2,298,478,592 B，总计超过 20 GiB 上限 510,134,928 B；单纯等待不能解决。没有调低预留、放宽上限或把失败改写成成功。

新候选 `video-ffn-chunks-r1` 将按行独立的 FFN 显式组合成共享权重的子图，使用原有 gather、NVFP4、SwiGLU、concat 操作及原舍入边界；共享 run/serve 配置 `feed_forward_chunk_rows` 默认 4096，可声明正整数上限。权重安装与动态工作区准备分离，临时容量拒绝通过原 admission/rollback 返回，不能把不确定退役当作可重试。实际 Reference 容量回归以及组件生命周期 19 项通过；FFN 5 项通过，覆盖尾块、权重解释一致和图规模限制。首轮 FFN 测试错误地比较原始 schema 与规范化 schema，改为比较同一准备流程的结果后通过，生产图未因此改动。21 文件冻结包 SHA 为 `71da1a4695c67db26d971ca06be157551e8b62641a9a2515b160584212c2e7a7`，5090 CUDA 编译进行中；真实 NVFP4 资格、20 GiB 重测和暖态速度均未计通过，尚未实现自动选择更小 FFN 分块。

FastVideo 原配置的 CUDA Graph 节点采样已完成，冷/暖各 1 条完整输出均与原基线 MP4 哈希一致；这是 profiler 诊断，不并入主性能表。GPU 已空闲后才启动候选编译。下一步先完成分块的实际数值/容量资格，再判断对双请求设备交接和端到端速度的收益。

2026-11-14 13:20（Asia/Shanghai）：同一 r2 二进制的晚加入 r3 补跑已完成，两条音视频均有效、与各自 seed 原 MP4 精确一致，零生成错误。第一条带冷启动 123.264274 秒，超过预先声明的 120 秒门槛；第二条 91.034498 秒。该功能补测不纳入暖态主性能表，不能把第一条 benchmark exit1 写成全项通过。CLI profile/完整解码编排 exit0，CLI MP4 的原始 SHA 尚未取回核对。

随后办公室入口 `100.89.56.34:22` 的新 SSH 连接及 Tailscale ping 超时，Mac 直连 `172.31.40.3:2222` 同样超时；本机 Tailscale 运行且网络诊断正常。20 GiB 容量资格命令的 SSH helper 无输出 exit255，远端是否收到命令未知，不能计通过；连接恢复后先检查已命名容器，避免重复提交或重叠 GPU 工作。新 profile 分析取回失败，FastVideo CUDA Graph 节点跟踪仅已准备、未启动。状态和精确恢复命令保存在库外 `video-device-handoff-r2.recovery-state.json`；性能目标仍未完成。

2026-11-14 13:10（Asia/Shanghai）：`video-device-handoff-r2` 完成冷请求与暖 C1/C2，共 5/5 音视频完整有效、零生成错误，全部与原 MP4 逐字节一致。冷 87.743374 秒；暖 C1 P50/P99 49.705257/49.707916 秒、1.207111 clips/min，C2 101.963795/104.062932 秒、1.152673 clips/min，各 n2、1 次重复，经验 P99 在预设 120 秒内。单请求相对 bias-vector 的 51.596644 秒减少约 1.89 秒；双请求未提速，日志只有 B1 交接获准，B2 仍走原 Host 中转。FastVideo 后续稳定单请求约 48 秒、C2 1.245904 clips/min，仍未超过；不能以其混合小样本 C1 P50 52.115772 秒宣称胜出。

冷/C1/C2 GPU 查询各失败 1 次，暖格仅 4/7 次 GPU 成功采样，查询最长阻塞约 119/108 秒；Host 虽零失败，联合采样同样缺失长区间。暖 VRAM 观察峰值 22,485,663,744 /28,827,451,392 B，RSS 43,142,807,552 /43,158,249,472 B，均为采样下界，资源证据不完整；benchmark 进程因此非零退出，不能称全项通过。TTFT/TPOT/SSE 文本事件 ITL 对视频作业不适用。原始报告保存在库外 `video-device-handoff-r2.runtime-summary-raw.jsonl`。

本轮 r1 冻结包的 epoch mtime 导致共享 Cargo 缓存复用旧二进制，服务在配置解析时即拒绝，未纳入性能；核对冻结源 SHA 后刷新 mtime，r2 实际重新编译，二进制为 `22c2295e92b05f1650db0a3a24239471fe729bd9d0a6c241b7e22d2c1a24bd78`。后续晚加入 r2 编排误等旧 r1 客户端名，触发清理并中断两条请求，两次失败保留且不计通过；正在用相同二进制/配置单独补跑 r3，CLI profile 与 20 GiB 容量资格尚待完成。Mac workspace/Metal check、Clippy（允许 warnings）、fmt/diff 通过；runtime 1017 通过/5 忽略、视频服务 15 通过、FastH3 134 通过/2 忽略，通用交接 7 项与预测头 4 项通过。完整工作区 tests 和必需 CUDA 特性检查最近完整结果仍属于 packed-staged 父候选。

2026-11-14 12:44（Asia/Shanghai）：`video-bias-vector-r1` 完成 5/5 常规 HTTP、2/2 晚到请求与 1 CLI；全部完整音视频解码有效，各 seed 均与其原单请求 MP4 逐字节一致，零生成错误。冷 85.970719 秒；暖 C1 P50/P99 51.596644/51.719019 秒、1.162863 clips/min，C2 100.976291/103.323223 秒、1.160865 clips/min，各 n2、1 次重复，经验 P99 在 120 秒内，仍未超过 FastVideo。冷/C1/C2 GPU 查询失败 1/1/4 次，暖格分别仅 3/6 次 GPU 成功采样；Host 虽零失败，也因联合采样被阻塞而缺失长区间。暖 VRAM 观察峰值 21,076,377,600 /28,827,451,392 B，RSS 37,100,056,576 /37,181,796,352 B，资源证据不完整，不能建立峰值上限。TTFT/TPOT/SSE 文本事件 ITL 不适用于这些视频作业。证据及命令均在库外 `.summary.json` 与 HTTP/CLI 两份归档。

晚到请求在首条完成两次实际 forward 后提交，实际加入时进度为 3/0；日志证明同一 joint forward 依次覆盖 3/0 至 7/4，首条解码后余下一条以 B1 继续 5、6、7 步并成功。整个服务仅一次 DiT 安装，新增保留策略未造成该流程重装或结果变化。此项是功能资格，不纳入主性能表。

剩余差距已发现明确的跨阶段数据往返：bias-vector 的 8 次 DiT 回读 2,505,043,968 B、GPU 拷贝 1.098313 秒，预测头上传 2,507,968,512 B、0.234781 秒；清零不是主要因素。待测 `video-device-handoff` 复用 Core 既有只读 Request state 初始化，增加受检 leading-axis 选择与一个 producer 到多个独立 consumer 的交接；全部输出明确交给设备时只等待原完成栅栏，不制造 Host 输出。每个 consumer 先走原容量准入，producer 再做真实零提交共存预检；可退役的容量拒绝关闭已建 consumer 后走原 Host 路径。默认 `prediction_handoff=host`，显式 `device_if_capacity_allows` 供 run/serve 共用；不降低精度、步数或编码质量。

通用 Reference 实际生命周期 7 项通过，包括无 Host 回读、多请求切片、producer 关闭后的独立 consumer、越界拒绝和无泄漏；FastH3 预测头 4 项通过，其中真实 producer 后的 Host/Device 两路在第 1/7 个原 PDD evaluation 上最终 F32 视频/音频张量完全一致。修复了测试模块路径、空输出仍调用 Host 空预留、泛型设备余量错误转换三处接入问题，原失败日志保留。新路径尚无 5090 完整视频或性能结果，不能计为已实现提速。

2026-11-14 12:18（Asia/Shanghai）：`video-residency-r1` 完成 5/5 HTTP + 1 CLI，六份输出均与原 MP4 逐字节一致、完整音视频解码通过，零生成错误。冷 87.522260 秒；暖 C1 P50/P99 51.983548/51.985658 秒、1.154205 clips/min，C2 101.214540/103.558671 秒、1.158228 clips/min，各 n2、1 次重复，经验 P99 在预设 120 秒内。服务日志仅一次安装 11,926,349,960 B DiT 权重，五次解码前预测均允许保留；相对父候选 C1 约省 1.52 秒，与被消除的暖权重上传相符。仍未超过 FastVideo 后续稳定请求约 48 秒及 C2 1.245904 clips/min，非同期小样本不构成正式胜出或稳定尾延迟结论。

冷/C1/C2 GPU 查询失败 1/3/1 次，Host 零失败但联合采样仍会因 GPU 查询阻塞而出现缺口；暖 VRAM 观察峰值 21,076,377,600 /28,827,451,392 B，RSS 35,335,266,304 /35,172,704,256 B，均为采样下界，资源证据不完整。TTFT/TPOT/SSE 文本事件 ITL 对视频作业不适用。两平台 runtime 1017 项通过、5 项显式忽略，Linux 视频服务 15 项通过；Mac workspace/Metal check、Clippy（允许 warnings）、fmt/diff 及产品 CUDA 构建通过，全工作区 tests/必需 CUDA 特性检查的最近完整结果仍属于 packed-staged 父候选。证据为库外 `video-residency-r1.summary.json`、HTTP/CLI 两份归档。

下一候选 `video-bias-vector-r1` 在通用 BF16 channel-bias 算子中，对齐且列宽为 8 倍数时采用 uint4 传输，逐元素 RN32 加法及 BF16 舍入保持；其他宽度沿用有界标量路径。CPU Reference 与真实 CUDA 六种形状、每形状两次提交、独立 oracle、全部 BF16 编码/尾部/owner 资格通过，产品 CUDA 构建完成，完整视频性能待测。保留权重策略改变了完成作业后剩余批次的生命周期，还需在该候选验证真实晚到请求及先完成一条后的继续执行。

2026-11-14 11:55（Asia/Shanghai）：`video-host-slices-r1` 已完成 5/5 HTTP + 1 CLI，全部原 MP4 精确一致、完整音视频解码通过。冷 88.027706 秒；暖 C1 P50/P99 53.501127/53.502523 秒、1.121467 clips/min，C2 103.126964/105.349598 秒、1.138574 clips/min，各 n2、1 次重复，零生成错误、经验 P99 在 120 秒内。冷/C1/C2 GPU 查询失败 1/1/5 次，Host 零失败；暖 VRAM 观察峰值 21,076,377,600 /28,827,451,392 B，RSS 32,381,542,400 /32,320,581,632 B，资源证据不完整，均为采样下界。未配对小样本与父候选整体相当，没有确立端到端提速。

同台主机 Rust CPU 微基准：空间拼接 0.598311/0.585669/0.585455 秒；RGB 查表 0.100683–0.106793 秒，标量参考 0.439105–0.441934 秒，四次交替顺序且输出精确一致，不能用该倍率推算推理收益。新 CLI Nsight 视频块间 Host 间隙合计 1.138357 秒（父记录 1.369967），整个视频解码跨度 4.123222 秒、GPU 2.374655 秒；DiT GPU 43.862142 秒、8 次跨度 47.148699 秒。均为诊断区间，未替代 HTTP 主表。证据 `video-host-slices-r1.summary.json`、两份归档及 `.cpu-gpu-summary.json` 均位于库外。

2026-11-14 11:46（Asia/Shanghai）：待测候选 `video-residency-r1` 为共享 run/serve 配置增加 `denoiser_residency`，默认 `release_before_decode` 保持原生命周期；显式 `retain_if_capacity_allows` 在完成栅栏释放所有闲置动态 backing，按视频/音频两个顺序计划的最大完整冷峰值检查现有进程容量账本及实际物理余量。不够则走原有 suspend/restore；关闭、取消、不同源计划替换仍由原 worker 持有并退役真实 owner。该只读检查不预留字节、不发分配 permit，后续普通 admission 仍权威；不宣称快照能消除并发外部进程导致的原生分配竞争。通用账本观察复用 claim 的同一检查函数，新增真实容量边界/观察后抢占/较小共享 ceiling/异常账户测试 2 项通过，实机保留与性能尚未验证。

2026-11-14 11:38（Asia/Shanghai）：`video-packed-staged-r1` 已完成共享 CLI 和冷/C1/C2 HTTP：共 6/6 原 MP4 逐字节一致、完整音视频解码通过，零生成错误。冷 88.004040 秒；暖 C1 P50/P99 53.495715/53.499786 秒、1.121582 clips/min，C2 102.725741/105.071486 秒、1.141559 clips/min，各 n2、1 次重复，均在预设 120 秒经验 P99 内；未配对、没有置信区间，尚未超过 FastVideo。冷/C1/C2 GPU 查询失败 1/1/12 次，Host 查询零失败；暖 VRAM 观察峰值 21,076,377,600 /28,827,451,392 B，RSS 32,551,673,856 /33,356,230,656 B，均为采样下界，资源证据不完整。视频作业的 TTFT/TPOT/SSE 文本事件 ITL 不适用。保留双方编码设置不同及尚无质量等价验收的限制，详见库外 `.summary.json` 和两份证据归档。

该候选 CPU/Reference 契约、真实 CUDA packed SwiGLU 两 profile/两形状/连续提交资格通过（有限值最大 0 ULP）；全工作区 all-target tests、CUDA CLI 必需特性编译退出 0，Mac workspace/Metal check、Clippy（允许 warnings）、fmt/diff 通过。模型 CUDA 首轮同进程混用 1/2 GiB fixture 容量导致 live metadata 冲突；未弱化预算或断言，另外两例按独立进程补跑通过，原失败日志保留，不将首轮计为通过。

带 CPU 调用栈的 Nsight 诊断显示：视频解码 GPU 从父候选 2.667531 秒降到 2.353573 秒，7 次调用跨度 4.327746 秒；区间包含异步执行，不能与 Host 耗时相加或替换暖 HTTP 表。6 个块间隙合计 1.369967 秒，采样热点为空间/时间拼接、输出布局转换、RGB 转换及逐元素受管缓冲区访问。下一候选 `video-host-slices-r1` 将 slice 获取移出循环，并在保持 BF16 舍入和事务发布的前提下合并非有限检查；本机相关 29 项通过、2 项诊断忽略，5090 微基准及产品验证待完成。DiT 间隙 1.411949 秒的 CPU 样本大多未解析（driver/libc），不归因于求解器或分配器。另已确认暖请求重装 11,926,349,960 B DiT 权重耗时 1.43–1.59 秒，正在核算容量允许时保留安装的收益，尚无实现或收益结论。

2026-11-14 10:57（Asia/Shanghai）：`video-head-warp-r1` 暖 C1 P50/P99 为 54.120803/54.235492 秒、1.108627 clips/min；C2 为 104.397930/107.120573 秒、1.119652 clips/min，各 n2、1 次重复。冷/C1/C2 共 5/5 完整音视频解码通过且原 MP4 字节一致，零生成错误；GPU 查询失败 1/1/8 次，资源证据仍不完整。暖 VRAM 观察峰值 21,076,377,600 /28,827,451,392 B，RSS 31,490,772,992 /31,239,143,424 B，均为采样下界。非同期小样本尚未超过 FastVideo；详见库外 `video-head-warp-r1.summary.json` 和归档。

CPU 空间拼接 `video-spatial-vector-r1` 在同台 5090 主机的既有 Rust 优化构建微基准为 0.645661/0.629917/0.624423 秒，旧诊断记录为 1.094/1.085/1.080 秒；未配对，不能直接换算整条视频加速。模型库 1134 通过、9 忽略；共享 CLI 完整生成、音视频解码和原 MP4 字节一致均通过，也补齐前两项 CUDA 优化的 run 入口资格。没有单独跑该微小候选的 HTTP sweep，下一步和解码器 packed SwiGLU 一并验证 serve 与性能，避免重复冷启动。Nsight 的步骤间分配/释放 API 累计不足 0.4 ms，不支持把约 1.49 秒 Host 间隙归咎于分配器；保持既有容量策略。

后续候选 `video-packed-staged-r1` 扩展通用 packed SwiGLU 的显式数值 profile，保留 SiLU 的中间 BF16 舍入，把 decoder 的两次切片和激活合为一次；未改原单次舍入 profile。CPU 受管 Reference 数值校验通过；首次导入错误已修正并保留日志。Mac 后续测试链接因磁盘满失败，清理可再生成的编译缓存后恢复 6.8 GiB 空间；未记作测试通过。契约、模型和真实 CUDA 资格及产品构建在 5090 继续，完整视频与收益尚待验证。源码、日志、trace 均在库外保留，未改变模型质量设置。

2026-11-14 10:33（Asia/Shanghai）：`video-row-gather-r1` 完成冷/C1/C2，5/5 原 MP4 精确一致且完整音视频解码通过。冷92.546962秒；暖 C1 P50/P99 54.384009/54.516109秒、1.103262 clips/min，C2 104.873602/107.594730秒、1.114721 clips/min，各n2、1次。相比父候选 C1 约减少1.01秒、C2约增加1.65%，只是非同期筛查；冷单样本反而较88.001841秒慢，不能泛称所有情形改善。冷/C1/C2 GPU 查询失败1/1/12，资源证据不完整；暖 VRAM 21,076,377,600 /28,827,451,392 B，RSS 29,973,463,040 /30,435,913,728 B，均为采样下界。暖B1 DiT Host区间合计44.984766/45.245325秒、跨度46.318294/46.620241秒；decoder合计3.230600/3.237971秒、跨度4.851290/4.976315秒，包含CPU、等待及读回，不当作GPU纯耗时。库外证据：`video-row-gather-r1.evidence.tar.gz`、`.summary.json`、`.host-phases.json`。该独立候选的CLI尚未另跑，将随下一候选共享资格验证。

基线逐条澄清：FastVideo C1 是56.118791、48.112753秒；C2响应延迟48.284628、96.315528秒，后者服务器串行，第二条增量约48.030899秒。因此52.115772秒小样本P50不能当作充分预热后的固定速度，当前差距并非只剩约2秒；仍需按正式预热规则重新配对比较。

`video-head-warp-r1` 保持原max/NaN类、div.full非FTZ、E4M3整数舍入和最终BF16边界，将≤256维短head改成每warp一组、每CTA四组；更宽维度保留旧CTA路径。真实CUDA资格覆盖warp尾组、31/32/33及256/257边界、多个batch、超过网格上限的262145组，两个连续提交owner检查通过。CPU Reference及workspace/Metal check、Clippy（允许warnings）、fmt/diff通过；首条完整视频89.008354秒，原MP4精确一致，热C1/C2正在测量。测试fixture首轮Host容量u64→usize类型错误已修正并保留失败日志。更后续的CPU拼接候选只把每像素早退改为累计中间/最终非有限标记，写画布前仍验证，7项空间数值/容量回归通过；尚未有5090主机微基准，未混入head候选。

2026-11-14 10:09（Asia/Shanghai）：补齐 FastVideo Nsight 的 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` 后，暖视频解码包含 20 次 graph replay、累计 1.367968 秒，另有普通 kernel 0.088752 秒；不能将未展开 graph 的普通 kernel 表误读为整个 decoder 仅 0.09 秒。Ferrum 同阶段 GPU kernel 合计约 2.670444 秒，仍有明确差距；双方被采样后的 stage 墙钟受 profiler 干扰，不替换未采样主表。另一方面，FastVideo 暖 DiT dense attention 22.961088 秒，Ferrum 冷 CLI 对应 21.791799 秒，目前没有证据将最大耗时的 dense attention 判断为性能落差来源。原始图执行摘要及逐 kernel 明细留在库外 `fastvideo-kernel-profile-r1.graph-summary.json`、`video-spatial-groups-profile-r2.kernel-details.json`。

下一候选 `video-row-gather-r1` 只优化通用 BF16 indexed gather：对齐且列宽为 8 倍数时按行作 uint4 位拷贝，其余仍走有界标量路径；非法索引在形成源地址前补正零。未改模型数值、精度、输出编码或资源所有权。CPU Reference 与真实 5090 managed CUDA 资格通过，覆盖全部 BF16 编码、批次隔离、动态索引更新、非法索引、奇数宽度、7168 列和超过 65,535 CTA 的行数；产品 CUDA 构建通过，完整视频与性能待测。另一项短 head 的 FP8 warp reduction 候选只在本机准备，未混入当前测量。

`spatial-groups-r2` 的完整 workspace/all-target tests 已运行完；首轮 root 绕过只读权限、次轮 macOS AppleDouble `._` 元数据被误当 Rust 源码及降权 runner 被 trybuild 递归调用，均保留原失败。使用普通用户 runner，且在独立副本中仅排除验证过 magic 的 4,230 个 AppleDouble 文件（其余 4,386 个常规文件逐字节一致），失败的两个目标补跑通过：81 个真实 UI 编译合同用例、source audit 3 项；只读权限用例也通过。没有删除或弱化测试断言，其他全量目标已在 r3 通过，显式 ignored 仍不算 runtime 资格。该结果属于 spatial-groups-r2；新增 CUDA 优化另行验证。日志和副本记录在库外 `video-spatial-groups-workspace-tests-r{2,3,4}*`。

2026-11-14 09:49（Asia/Shanghai）：`video-spatial-groups-r2` 暖测量完成，C1 P50/P99 55.395131/55.526476 秒、1.083120 clips/min；C2 106.528423/109.365184 秒、1.096661 clips/min，各 n2、1 次重复。冷/C1/C2 GPU 查询失败 3/1/7 次，峰值证据不完整；暖 VRAM 21,076,377,600 /28,827,451,392 B，RSS 41,989,754,880 /42,744,512,512 B。5/5 HTTP 与新增 CLI 均音视频完整解码通过，六份文件均与原 MP4 字节相同。C1 较父候选约减 0.99 秒，C2 基本不变；相比 FastVideo 52.115772 秒、1.245904 clips/min 仍落后，未达到目标。

CLI 与 Nsight 资格合并执行，exit0、832×480/124 帧/24FPS/H264、AAC32k双声道。新 trace 共 104,366 kernels，52.260730 秒；8 次 DiT 的 Host 执行区间合计 46.880855 秒、跨度 48.371478 秒，其中 GPU dense SDPA 21.791799、NVFP4 GEMM 9.032585、稀疏 attention 5.269608、row gather 1.821512 秒。视频真实调用从 21 降至 7，Host 区间合计 3.274572 秒、跨度 4.938775 秒，GPU GEMM0.602186/SDPA0.420350 秒。Host/GPU 区间不得相加，也不将该冷 CLI trace 当暖态性能表。证据：`video-spatial-groups-r2.evidence.tar.gz`、`.summary.json`、`video-spatial-groups-profile-r2.evidence.tar.gz`、`.phase-summary.json`；26.6 MB 原始 trace 留在 5090 库外。

FastVideo 原启动脚本/配置亦完成同 PNG、同 seed 的 Nsight 冷+暖诊断，两份完整音视频均与原 FastVideo 文件字节相同。采样下冷91.451250/暖61.808596 秒仅诊断，不替换未采样的性能基线；已停止 profiler 及其服务。正在分析双方同阶段 GPU 分项，不能仅凭 Ferrum 最大耗时项判断差距来源。全工作区 CPU all-target tests 已在 GPU 采样结束后启动，结果待完成。

2026-11-14 09:35（Asia/Shanghai）：共享空间分组 `video-spatial-groups-r2` 已在 5090 实际选择 `tiles=12,total_tiles=12`，首条完整视频 88.001841 秒；音视频完整解码通过、与原 MP4 字节相同。暖 C1/C2 尚在执行。模型库 1134 项通过、9 项显式忽略，Host 减组/少量占用/原错误边界 5 项通过；workspace all-target check、Metal check、Clippy（允许 warnings）、fmt/diff 通过，完整 workspace all-target tests 尚未完成。CUDA r1 发现组合入口缺少 anyhow→既有 BatchError 的转换，r2 仅补齐该入口映射，构建已通过并保留初次失败。无失败请求被吞掉或转成容量重试。两入口共用此分组实现，CLI 完整出片待暖测量后验证。

缩放校验的阶段核对：同冷请求 vision forward 结束到 text forward 开始的 Host 间隔由 28.968729 秒降至 5.680478 秒，约 23.29 秒，与端到端约 22.92 秒下降一致；间隔包含计划/初始化等工作，不是单个 GPU kernel 时长。其冷/C1/C2 GPU 查询失败分别为 2/1/5 次，已保留在 `video-scale-validation-r1.summary.json`。本轮各指标仍为非配对诊断筛查，不能宣称已超过 FastVideo。

2026-11-14 09:25（Asia/Shanghai）：通用 NVFP4 缩放校验候选 `video-scale-validation-r1` 完成冷请求及暖 C1/C2。首请求从同配置旧值 111.677368 秒降到 88.753160 秒（单次非配对约减少 22.92 秒）；FastVideo 同 PNG 首请求 81.884680 秒，仍未胜出。暖 C1 P50/P99 为 56.384921/56.502261 秒、1.064109 clips/min；C2 106.632152/109.349101 秒、1.096846 clips/min，各 n2、1 次重复。5/5 音视频完整有效、零生成错误、均与原 MP4 字节一致。暖 GPU 查询仍失败 1/5 次，峰值证据不完整；C1/C2 VRAM 观察值 21,076,377,600 /28,827,451,392 B，RSS 42,244,526,080 /42,721,226,752 B。优化构建校验微基准 16 MiB 原逻辑约 27–28 ms、storage scan 约 4 ms，不能用微基准倍率推算整个请求。16 项缩放正确性测试及扩展真实 NVFP4 CUDA 资格通过；产品构建完成，CLI 将与后续解码候选一并验证，尚不称两入口均已完成本轮验证。

解码抽象调整为共享 `SpatialTileBatch`（正数偏好）与 `SpatialTileGroups`（已按真实画布裁剪的有限分组）。视频不再借用图片策略的 1–4 上限或列宽上限；以真实 tile 总数的降序因数选择同形分组，保持一个计划/权重安装，Host/GPU 容量不足继续缩组，临时占用继续等待。图片既有策略仍通过共同分组逻辑保留原列宽及 1–4 限制。`video_decoder_tile_batch` 配置字段及默认 1 保持，12 是本轮显式筛查值。此抽象不合并请求/time 维度、不改变单 tile 的 attention/RoPE/原邻居 BF16 拼接、不引入填充。初步空间顺序/原图片语义/实际 Output backing 及少 1 B 回归 11 项通过、1 项性能诊断忽略；新增画布上限和 12→6→4 Host 减组检查正在完整构建中。当前还没有新分组的 GPU 出片或性能结果。

2026-11-14 09:13（Asia/Shanghai）：N256 GEMM 候选已拒绝并恢复原 N128 原生实现及数值身份。真实 5090 的动态/Stored、跨 M/N 边界、非有限填充与两次提交 owner 资格通过；完整 HTTP 5/5 视频音频解码有效、均为原 MP4，零生成错误。但冷请求 114.961253 秒，暖 C1 P50/P99 61.805393/62.057685 秒、0.970782 clips/min，C2 120.629454/123.712106 秒、0.969501 clips/min，双请求经验 P99 超过预设 120 秒；各暖格 n2、1 次重复，不能保留为提速。冷/C1/C2 GPU 查询失败 3/1/1 次，暖格各仅 10/12 次成功采样，最长查询阻塞约 132/130 秒，峰值证据严重不完整。暖 VRAM 观察值为 21,076,377,600 /28,827,451,392 B，RSS 42,116,079,616 /42,192,416,768 B，均非上限。拒绝候选不追加 CLI 或正式 A/B；完整记录位于库外 `video-gemm-n256-r1.evidence.tar.gz`、`.summary.json`。保留有用的跨多 N tile 正确性测试。

下一候选仅优化 NVFP4 权重源的通用缩放校验：完整物理 tile 直接检查编码合法性，不进行无用浮点解码或坐标换算；部分 tile 仍仅检查有效位置，保留任意填充字节语义。全部 256 编码、每个有效位置、长度边界及已有 source 回归共 16 项通过，1 项优化构建诊断 benchmark 暂忽略。还没有 5090 性能收益证据。冻结源以已验证的 N128 上传阈值版本为父，继续固定 8 MiB 下限及原精度/预算；初次远端 CPU 构建因无 native-op-artifact feature 却继承原生锁而中止，已保留失败，并仅对 CPU 测试取消该构建变量后继续。正式 GPU/产品构建仍使用原生锁。

2026-11-14 08:50（Asia/Shanghai）：8 MiB 可选权重注册阈值 `video-upload-upper-r1` 已完成，同一已验证二进制，仅调整显式配置。冷请求 111.677368 秒；暖 C1 P50/P99 56.003251/56.003729 秒、1.071364 clips/min，C2 107.386557/110.101622 秒、1.089354 clips/min，各 n2、1 次重复。5/5 音视频完整解码通过、零生成错误、全部与原 MP4 字节相同。相较同 PNG FastVideo 的 C1 52.115772 秒和 C2 1.245904 clips/min，仍有约 7.5% 的延迟差距及 12.6% 的吞吐差距；这些非同期小样本不构成正式性能结论。冷/C1/C2 GPU 查询失败 1/1/5 次，峰值证据仍不完整。暖 C1/C2 采样 VRAM 为 21,076,377,600 / 28,827,451,392 B，进程树 RSS 为 42,879,414,272 / 43,403,931,648 B。证据：库外 `video-upload-upper-r1.evidence.tar.gz`、`.summary.json`。

8 MiB 候选全进程仅观察到 4 次必要 Host 注册，累计 3,064 微秒、单次最大 1,263 微秒；此前的注册阻塞已消除，未改变原 fence、Host 所有权或容量预算。阈值版本的共享 CLI 验证、workspace all-target check、Metal check、Clippy（`-A warnings`）及 fmt/diff 检查均通过；完整 workspace all-target tests 仍待完成。当前下一候选 `video-gemm-n256-r1` 将通用 SM120 NVFP4 GEMM 的输出列分块由 128 改为 256，保留 K128 归约、精度边界、static persistent 调度与原 owner；同步更新数值/来源身份，重新生成编译器实际依赖闭包及原生锁。原生编译已通过，新增跨 N256 边界的动态/Stored 独立 dyadic oracle 资格正在构建。初次打包编排因容器缺 Python 而 exit127，已保留日志，并用容器现有工具继续，复用成功的原生编译产物。该候选尚无 GPU 正确性或性能结论。

2026-11-14 08:22（Asia/Shanghai）：分组 segmented-attention 候选完成真实 GPU block、完整冷请求和暖 C1/C2；共 5/5 音视频完整解码通过，均与原 `d3de4e5a…bd5a0` MP4 逐字节相同。暖 C1 P50/P99 为 60.283095/61.520712 秒、0.995302 clips/min；C2 为 107.744825/111.068489 秒、1.079754 clips/min，各 n2、1 次重复。冷/C1/C2 GPU 采样失败 10/19/5 次，峰值证据不完整，不能称全项通过。该候选尚未保留为性能优化；已验证最快记录仍是 tile-groups-r4 的 C1 56.887743 秒。证据：库外 `video-segment-groups-r3.evidence.tar.gz`、`.summary.json`。原生分组及固定 CPU 后处理的新结果不构成 FastVideo 胜出结论。

回退定位已收敛到阶段间的小块 Host 注册等待：新候选 DiT 执行阶段总计仍约 45.95 秒，完整跨度却扩大到 49.81/52.04 秒；这些是包含提交与回读的 Host 区间，不是纯 GPU kernel 时间。旧 r4 Nsight 按 Host 日志窗口重算，8 次预测头初始化合计 0.029426 秒，其中 32 次 `cuMemHostRegister` 0.015987 秒；主干 GPU kernel 分项为密集 cuDNN 21.742797 秒、NVFP4 GEMM 9.023336 秒、稀疏 attention 5.266300 秒、索引 gather 1.826559 秒（冷 CLI trace，非暖态主表）。新服务 DEBUG 细分直接捕捉到仅 4096 B 的原生注册等待约 0.330–0.557 秒，锁及 context bind 接近零；计时暖请求 60.287120 秒内注册累计 4.561620 秒。资源采样 OFF/ON 各 1 次为 58.245790/59.294778 秒，仅诊断，不能由此确立采样因果或以无采样数据替代主表。证据：`video-tile-groups-profile-r4.phase-summary.json`、`video-upload-timing-r1.evidence.tar.gz`、`.summary.json`。

2026-11-14 08:39（Asia/Shanghai）更新：`video-upload-threshold-r1` 已完成 4 MiB 阈值的完整验证。冷请求 114.174410 秒；暖 C1 P50/P99 为 58.362083/58.977904 秒、1.028062 clips/min，C2 为 108.271500/111.850227 秒、1.072163 clips/min，各 n2、1 次重复。HTTP 5/5 完整音视频有效、零生成错误；共享 CLI 亦 exit0、完整解码通过，六份 MP4 均与原 `d3de4e5a…bd5a0` 字节相同。冷/C1/C2 GPU 查询分别失败 2/4/11 次，资源证据不完整；观察 P99 小于 120 秒不证明稳定尾延迟。此候选没有超过 r4 最好记录，更没有超过 FastVideo。证据：库外 `video-upload-threshold-r1.evidence.tar.gz`、`.summary.json`、`video-upload-threshold-cli-r1.evidence.tar.gz`。

该阈值是共享 CUDA 受管权重上传的物理 backing 下限，原调用方及缺省配置维持 0，FastH3 run/serve 共用序列化字段 `cuda.minimum_pinned_weight_upload_bytes`；较小权重沿用 pageable 受管复制与原 fence/owner，必须固定的原生 metadata 不受此下限影响。Device28GiB、Host4GiB、注册总上限16MiB、精度、层数及编码不变，不增加缓存。3 项范围/配额检查、2 项真实 CUDA 注册生命周期/配置身份检查及 required-flags 产品构建通过；Mac workspace all-target 编译、fmt/diff 检查通过，当前完整 workspace tests 仍未补齐。

细分日志确认本轮 B1 的八次预测头初始化合计仅 0.007965/0.008240 秒，主干执行 Host 区间 45.820800/45.970919 秒、跨度 47.335977/47.467611 秒；小块注册等待已消除。视频和音频初始化仍出现较大权重 backing 的原生注册等待，最长约 2.7 秒，故继续用相同二进制、固定其余配置，测试显式 8 MiB 下限（`video-upload-upper-r1`）。这仍是非配对筛查；不会将配置变化或采样缺失换算为性能胜出。后续 GPU 优化优先依据上述密集 attention / NVFP4 GEMM 实测分项，不把 API 阻塞区间冒充 kernel 耗时。

2026-11-14 07:37（Asia/Shanghai）：同字节 PNG 的 FastVideo 复测 r3 已完成，冷请求 81.884680 秒；暖 C1 P50/P99 52.115772/56.038731 秒、1.151257 clips/min，C2 72.300078/95.835219 秒、1.245904 clips/min，各格 n2、1 次重复，5/5 完整音视频解码通过且同 seed 文件一致。冷/C1/C2 GPU 资源查询失败 5/8/2 次，报告完整保留，不能建立完整峰值上限。VRAM 暖 C1/C2 观察峰值 27,799,846,912 / 27,804,041,216 B，进程树 RSS 50,267,615,232 / 40,059,138,048 B（含共享页重复计数）。服务已正常停止，仍为 disabled。证据：库外 `fastvideo-recheck-r3.evidence.tar.gz`、`fastvideo-recheck-r3.summary.json`。实物 MP4 的 x264 SEI 还确认 FastVideo 为 core165、ultrafast 对应选项、CRF23、15 threads；Ferrum 为 core164、veryfast、CRF18、4 threads（`fastvideo-ferrum-encoder-sei-r1.jsonl`），因此这些非配对筛查仍不构成同编码设置的正式主表，未宣称胜出。

保留版本 r4 的共享 CLI 经 Nsight 执行 exit0，完整 ffmpeg 解码通过，MP4 与原 `d3de4e5a…bd5a0` 字节相同，补齐了分组解码的 run 入口证据。新 trace 为 125,702 个 GPU kernels、累计 52.190175 秒，其中密集 cuDNN SDPA 22.715567 秒、NVFP4 GEMM 9.905921 秒、稀疏 attention 5.266300 秒；这些是冷 CLI 的 kernel 累计区间，不是端到端暖态耗时。分配 API 合计约 0.093 秒，当前不优先改分配策略。证据：`video-tile-groups-profile-r4.evidence.tar.gz`、`.summary.json`，大 trace 保留在 5090 库外。

正在验证的新候选尚未成为性能结论：`video-host-postprocess-r1` 保留原时间 BF16 逐步舍入，按帧计算重叠系数、非重叠区整帧复制；原 F32 RGB 传递使用编译期 192 KiB 只读表。Mac 与 5090 的 6 项 CPU 检查通过，含全部有限 BF16 编码、非有限输入拒绝和 Output grant 回收；5090 优化构建的 RGB 微基准四次交替次序为原式 0.444–0.451 秒、查表 0.240–0.247 秒，结果逐字节相同，只代表该 CPU 转换。其上 `video-segment-groups-r1` 将连续等长的隔离段作为独立 cuDNN batch，超过原生上限拆组，保持每个成员的长度、地址跨度、原 scratch 与 owner；不同长度不合并。实际 D64 非零完整 block 的 CUDA 资格新增 17 个等长段及 ragged 尾部，以免把旧 scalar-attention/零 residual fixture 误当本后端验证。截至 07:47，`video-segment-groups-r2` 的分组 5 项、mask 5 项、原生 SDPA ABI 13 项检查通过；Mac 解码模块 28 项通过、2 项显式忽略。r1/r2 分别发现测试的 sum 类型推断、VariantPack 新参数遗漏及 generic 验证闭包类型推断编译问题，已在不可变 r2/r3 overlay 修正并保留失败日志；生产候选逻辑未变。r3 正在构建实际 GPU block 资格和产品 CLI，完整请求及性能均待实测，当前最快已验证结果仍是 r4。

2026-11-14 07:16（Asia/Shanghai）：cuDNN B→A→FALLBACK 排序候选已完成，决定不保留，`native.rs` 已恢复试验前的完整文件。3 项既有 CUDA 数值/隔离资格通过；冷请求 113.667302 秒，随后暖 C1 P50/P99 57.364920/57.487273 秒、1.045933 clips/min，C2 107.878292/111.083584 秒、1.079631 clips/min。共 5/5 完整音视频验证通过，MP4 均与此前 seed1101 结果逐字节相同；暖 C1 未改善，C2 不足 1% 的差异不能由这次非配对 n2 筛查证明。冷/C1/C2 资源查询分别失败 6/17/2 次，客户端因此 exit1；不作完整峰值或正式胜出证据。原始记录、状态与撤回理由位于库外 `video-cudnn-heuristic-b-r1.evidence.tar.gz`、`video-cudnn-heuristic-b-r1.summary.json`。

FastVideo 原服务已重新启动但保持 disabled，启动脚本和配置哈希与旧基线相同；本轮改用与 Ferrum 完全相同的 PNG 字节（`b5416687…e9c9`），放在原允许的 inputs 目录，旧 pilot 使用 JPEG 的输入形式差异明确保留。首次客户端替换误将 `image_reference` 列表改成字符串，HTTP400 在推理前拒绝；`fastvideo-recheck-cold-r2.report.json` 保留失败。r3 从旧基线原始请求复制列表结构后已进入冷生成，尚无新基线性能结论。下一步完成暖 C1/C2，再跟踪保留的 r4 CLI 路径及 GPU/API 开销；不会为无明确收益的排序试验继续扩测。

2026-11-14 06:55（Asia/Shanghai）：r4 已解决上述容量失败，实际 4 tiles/group、21 次视频调用全部完成。冷 HTTP 请求 111.652201 秒、完整音视频解码通过，MP4 与原 `d3de4e5a…bd5a0` 字节相同；随后固定配置暖 C1/C2（各 n2、1 次重复，冷请求后无额外预热，未先预热 B2）4/4 成功且所有文件相同。C1 P50/P99 56.887743/57.264102 秒、1.054706 clips/min；C2 108.499083/111.831323 秒、1.072392 clips/min。相较前候选 C1 59.394843 秒及 C2 1.019479 clips/min 有改善，但仍落后 FastVideo 原暖缓存 pilot 的 52.711489 秒及 1.274351 clips/min；这仍是非同期同参考缓存筛查，不是新参考 paired 胜出主表。资源采样不完整，benchmark 客户端因此 exit1，不能将它写成全项通过或生成失败；请求错误为 0，服务最终正常停止。VRAM 观察峰值 C1/C2 为 21,076,377,600 / 28,827,451,392 B，RSS 为 43,148,709,888 / 43,062,882,304 B。完整证据在库外 `video-tile-groups-r4.evidence.tar.gz`、`video-tile-groups-r4.summary.json`。

r4 暖 B1 的视频 decoder host 区间从前候选约 7.24 秒降为 5.32/5.65 秒，实际调用 84→21；DiT 8 步 host 区间仍为 47.66/47.99 秒（非 GPU kernel 时长）。因此下一候选仅比较 cuDNN SDPA 的确定性候选排序：固定库原策略 A→FALLBACK→B 改为 B→A→FALLBACK，保留原数值合同、workspace 上限、确定性过滤及资源所有权，不进行 native autotuning。部署的 `cudnn_graph.h` 已核实枚举值；源码 `video-cudnn-heuristic-b-r1` 已冻结，正在先跑既有 CUDA prefix 数值 oracle、完整 clip 批隔离与 causal GQA 数值验证，尚无性能或画质结论。r4 本地 engine 全量 934 通过/1 忽略，Clippy（允许 warnings）通过；首轮 Clippy 的“反向空 range”错误是故意构造的非法输入测试，已以局部注释与 allow 保留拒绝测试语义。完整 workspace all-target tests 仍未完成，r4 的分组 CLI 入口尚未单独重跑，不能称 PR 已验证。

2026-11-14 06:43（Asia/Shanghai）：分组解码 r2/r3 的首次完整请求未通过，均在第一时间块末尾终止，未产生可计入性能的产物。r3 的保留所有权诊断确认是 Host Output scope：申请 11,014,143 B、剩余 10,723,337 B，差 290,806 B；不是 GPU OOM。此前新缓存保留的原始 Output backing 与解码同时存在，旧 admission 将两阶段视为互斥，分组扩大读回后暴露漏计。候选 r4 将可保留条件结果的物理上限与 decoder peak 相加，再与 conditioning floor 取最大值；不改变 Host 4 GiB / Device 28 GiB 总上限。若硬容量不足先缩组，所有组仍不适用时放弃可选缓存再尝试；临时占用仍等待。4 项 Rust 容量/原 owner 生命周期测试通过，含真实分配、关闭 scope 后缓存继续记账、少 1 B、缩组与放弃可选缓存；r4 正在构建，完整 GPU 复测尚待完成。r2 模型 26 通过/1 忽略、engine 932 通过/1 忽略，workspace all-target 与 Metal 编译检查通过；这些不替代 r4 的实际复测。

2026-11-14 06:29（Asia/Shanghai）：条件复用已在 5090 完成资格与性能筛查。实际条件结果保留 79,884,286 B，因此默认 64 MiB 策略正常放弃缓存并重新计算；未修改默认值。显式 128 MiB 配置（其余 Host 4 GiB / Device 28 GiB 不变）下，seed1102 中途加入 seed1101，联合前向实际进度从 2/0 到 7/5，期间不重新加载 DiT 权重。两请求均完整解码且分别与各自 B1 MP4 字节相同。暖 C1/C2 各 n2、1 次重复、4/4 成功：C1 P50/P99 59.394843/59.527651 秒、1.010185 clips/min；C2 112.667769/117.606331 秒、1.019479 clips/min。观察到的 P99 均小于预声明 120 秒，但小样本不证明稳定尾延迟；仍慢于 FastVideo 对照（C1 P50 52.711489 秒、C2 1.274351 clips/min）。C1/C2 资源查询各 10/2 次失败，VRAM 观察峰值 28,827,451,392 B，RSS 42,997,927,936 / 43,359,588,352 B，采样不完整。候选 DEBUG 日志、非同期测量和 FFN 分块差异仍需披露。证据：库外 `video-conditioning-reuse-functional-r2.evidence.tar.gz`、`video-conditioning-reuse-warm-r2.evidence.tar.gz` 与对应摘要。

同一二进制及 128 MiB 配置的 `ferrum run` 于 06:20:39 正常退出，完整 H264/AAC 解码通过，832×480 / 24FPS / 32kHz 双声道、MP4 SHA256 `d3de4e5a70933e54f2187f1277cc5274cc1a94f85b999348af028e8d511bd5a0` 与原文件一致。全进程约 145.97 秒只作冷启动正确性证据；`video-conditioning-reuse-run-r1.evidence.tar.gz` 保存日志、状态、探测及解码结果。该快照 engine 930 通过/1 忽略，workspace all-target check、Metal check、Clippy（允许 warnings）、fmt/diff 通过；完整 workspace all-target tests 尚未完成。

下一候选 `video-tile-groups-r2` 在原解码计划上增加显式 `video_decoder_tile_batch`（1–4，默认仍 1）。以最大可容纳的真实均匀空间分组复用一份计划及权重，保留 request/time 独立维度、每 tile 原始 BF16 邻居融合，不填充虚构样本。480p 可将 12 tiles×7 chunks 的 84 次调用减为 21 次；Host 硬容量及编译设备容量不足时缩组，临时占用仍走原接纳等待。模型边界测试 26 通过/1 忽略；第一轮 CUDA 构建的私有模块引用错误已修正，第二轮构建与 Host 容量测试进行中，尚无实际 GPU 分组性能结果。

2026-11-14 06:02（Asia/Shanghai）：`video-stitch-axis-r1` 已完成原输入的完整视频复测。空间拼接保留原 BF16 系数及逐次舍入、先纵后横与原始邻居语义，将逐像素分支改为行复制和重叠区融合；5090 主机上优化构建的 Rust 微基准（480p、12 tiles×7 chunks，3 次）由 2.865/2.858/2.862 秒降到 1.094/1.085/1.080 秒，仅代表 CPU 拼接。完整 C1 筛查为一次冷预热、两个热请求、1 次重复，P50/P99 68.194213/68.324444 秒、0.879836 clips/min，2/2 成功且 MP4 与原 `d3de4e5a…bd5a0` 逐字节相同；前候选 C1 P50 69.484601 秒，FastVideo 原 pilot 52.711489 秒，仍未达到更快目标。这是非同期小样本，不将微基准收益直接换算为整体提升。测量阶段 11 次 GPU 查询超时、393 个采样间隔跳过；显存/RSS 观察峰值 21,076,377,600 / 43,090,030,592 B，资源采样不完整；前候选 C1 的设备池曾被 B2 预热，本次仅 B1。证据为库外 `video-stitch-axis-r1.evidence.tar.gz`、`video-stitch-axis-warm-r1.summary.json`。

后续 `video-conditioning-reuse-r1` 已完成构建，实际混合进度与暖性能待验证。两入口共享显式 `conditioning_reuse` 策略：默认 `{"kind":"last_result","maximum_retained_bytes":67108864}`，可配置 `{"kind":"disabled"}`。只在同一不可变资产/处理器实例内，按有长度分帧的原始 prompt 与有序图片内容 SHA256 复用已完成的 posterior/refined-text Host owner；生成 seed、目标几何不改变这两个条件计算，后续噪声仍逐请求创建。只保留最后一个结果，额度按完整物理 tensor backing 计算，元数据另受现有模型限制约束。接纳受临时容量限制时先清除缓存再真实重试，仍不足则等待；活跃借用的实际账目不会因清除索引而消失。命中结果在接纳后固定，worker 据此决定是否需要暂停 DiT、加载辅助模型；未完成/失败的 native work 不入缓存。身份、显式配置与额度边界测试通过，engine 全量 930 通过/1 忽略。首次编译因误用 `BatchPreparation::get` 失败、首次配置测试暴露 serde unit variant 未拒绝未知字段，均修正并保留失败日志；不把 CPU 测试当作 CUDA 缓存资格。

2026-11-14 05:22（Asia/Shanghai）：`video-phase-workspace-r1` 已在 5090 完成实际 B2 全部 8 轮和两段完整音视频，2/2 成功且 MP4 与 B1 原文件逐字节相同。另以 seed1101/1102 做真实中途加入：同一联合前向 `run.fast_h3.119.denoise` 的 slot evaluation 分别为 6/0，下一次为 7/1；首请求完成后，后一请求以 B1 继续 evaluation2–7。两个文件分别精确等于各自 B1 对照，客户端 exit0；这是该单图 480p 路径真实混合进度连续批处理的资格证据，不泛化到全部参考类型/分辨率/取消路径。

暖进程相同输入筛查已完成（固定服务器/28GiB，C1→C2，各 n2、1 次，无额外预热）。Ferrum C1 P50/P99 为 69.484601/69.602339 秒、0.863499 clips/min；C2 为 132.686706/137.993969 秒、0.868921 clips/min，后者未达到 P99≤120 秒。FastVideo 原 pilot 对应 C1 52.711489/58.219412 秒、1.138267 clips/min，C2 70.755746/93.697331 秒、1.274351 clips/min。候选 4/4 成功且文件一致，但没有吞吐提升或性能胜出；包含此前功能用例共 9 个成功 HTTP 请求，服务正常停止退出 0。候选 DEBUG 日志、FastVideo 条件缓存命中而 Ferrum 重算、FFN 分块不同及非同期小样本必须保留在比较中。候选 C1/C2 分别 4/3 次 GPU 查询超时；显存观察峰值同为 28,827,451,392 B，RSS 分别 42,485,760,000 / 42,724,474,880 B，采样不完整，不能据此证明资源上限。原始报告、状态和比较表为库外 `video-phase-warm-r1.evidence.tar.gz`、`video-phase-r1-comparison.json`。

下一性能重点已缩到准备/解码：暖 B1 的 DiT 8 次 host retirement 合计 45.866 秒，视频 decoder 的 84 次调用合计 5.073 秒，但首调用开始到最后 retirement 跨 9.031 秒；这些包含 CPU、等待和读回，并非 GPU kernel 时长，也不与 FastVideo stage 边界严格等同。FastVideo 原视频解码 stage 为约 1.5 秒。应沿现有空间 tile 分组/同源权重复用与 Host 输出所有权设计减少调用和拼接成本，维持原局部 attention、RoPE、BF16 邻居融合及逐帧语义；容量不足缩组，不把执行错误转为容量。当前没有改这些后续实现。阶段容量 Reference 测试及组件套件 16 项通过，engine 928 通过/1 忽略；workspace all-target check、Metal all-target check、Clippy（允许 warnings）、fmt/diff 通过。全工作区 all-target tests 未完成，不称 PR 已验证。同一候选的 CLI 完整出片已于 05:22:34 正常退出 0；832×480、24FPS、H264/AAC 双声道 32kHz、完整音视频解码通过，MP4 仍精确等于原 `d3de4e5a…bd5a0`。CLI 全进程约 150.44 秒只作冷启动正确性证据，不混入暖性能表；日志、状态、ffprobe 和解码结果为库外 `video-phase-workspace-run-r1.evidence.tar.gz`。

2026-11-14 05:01（Asia/Shanghai）：诊断 r2 已重现并取得原始错误：PDD 初始化其动态池时申请 939,757,056 B，进程已占 29,747,760,880 / 30,064,771,072 B，只余 317,010,192 B。主 DiT B2 正常完成后仍保留 15,520,054,144 B idle backing，导致两个串行阶段的 workspace 叠加。候选修复把显式工作区策略调整为 `ActivePhaseOnly`：完成 Host 读回及 retirement 后，经原 session 的 idle-release 释放临时池；共享权重和 Host 输出保留，后续执行仍走原 admission。没有提高 28 GiB 上限，也没有吞掉原始未知错误或尝试在部分失败的 PDD 初始化上盲重试。engine 编译检查已通过；精确阶段容量/少 1 B/输出存活/权重不重传的 Reference 回归和实际 CUDA 共批待验证。两入口使用同一阶段实现。

2026-11-14 04:55（Asia/Shanghai）：诊断 r1 已确认 B2 取得 15,520,054,144 B 动态 backing，主 DiT 联合前向正常完成（含等待/读回的 host retirement 11.446 秒）；随后 PDD 输出预测失败，两个请求均未完成。故之前“无联合前向”应理解为没有完整成功的联合 step，不能再推断主干执行失败。PDD 错误枚举原 Display 隐藏了初始化/执行/关闭的具体来源；诊断 r2 已补齐借用原错误的输出，正在同机重现，暂不将失败归为容量。单请求暖 P50 仍为 68.44 秒，FastVideo 对应 pilot 为 52.71 秒，性能目标未达成。产物证据和错误日志均在库外 `video-batch-diagnostic-r1.evidence.tar.gz`，没有用计划容量代替完整共批资格。

串行顺序候选的暖 C1 为 68.439948 秒（经验 P99 68.532714 秒，0.876679 clips/min），2/2 成功；显存采样峰值 22,485,663,744 B。实际 C2 准备却失败：0/2 成功、无联合前向，服务停止接收任务，客户端 exit1；不能作为 B2 通过或有效吞吐结果。原始失败保留在 worker 恢复对象中，但现有服务仅输出通用错误；当前补充统一退休边界的原始错误日志，格式化异常不会替换或释放原始 owner。下一轮仅重跑一个 B1 预热加 C2 来获取根因，未将未知错误改判为容量。失败证据完整归档于库外 `evidence/video-serial-warm-r1/`。

串行算子顺序候选 `video-serial-order-r1` 的 required-flags CUDA 构建已成功（3 分 24 秒，复用原缓存、4 个编译任务）。真实 CLI 正常出片，113.263 秒冷请求、147.143 秒全进程，MP4 仍为原 `d3de4e5a…bd5a0`，完整音视频解码通过。完整 B1/B2 计划最低预留分别为 19,686,492,816 / 27,446,518,512 B；B2 首次低于原 28 GiB（30,064,771,072 B），仍须实际两请求动态 backing/联合前向验证。当前 workspace all-target check、Metal check、Clippy（允许 warnings）与 fmt/diff 均通过；全工作区 all-target tests 未完成。暖 C1/C2 已启动，不能把计划容量满足换成实际共批或提速结论。证据为库外 `video-serial-order-{build,run}-r1.*`。

只读视图 r2 暖进程 C1 已完成：两条请求成功、零生成错误，经验 P50/P99 为 68.557541 / 68.561669 秒，0.875175 clips/min。相对 r4 的 71.336147 秒下降约 3.9%，仍慢于 FastVideo pilot 的 52.711489 秒；两请求小样本、非配对实验，不能据此宣称正式胜出。采样峰值 VRAM 24,398,266,368 B、RSS 42,907,353,088 B，采样不完整，均为观察下界。C2 同样两条成功、零生成错误，经验 P50/P99 为 106.605049 / 135.239366 秒，0.883498 clips/min，仍未达到 P99≤120 秒；FastVideo 对应吞吐为 1.274351 clips/min。4/4 测量产物均完整解码且与旧 MP4 字节一致。含预热共 48 次 B1 执行、8 次缩批，结束队列为空、拒绝数 0；客户端 exit1（SLO/资源采样不足）、服务正常退出 0。C1 有一次、C2 有三次 GPU 查询超时；完整原始报告及终态保存在库外 `evidence/video-readonly-warm-r2/` 和同名前缀文件中。

下一容量候选采用通用、可序列化的 `PlanExecutionOrder`：默认保留数据/状态依赖；显式 `serial` 将相邻节点的完成依赖写入实际计划，内存规划继续使用原 DAG 证明。它牺牲独立分支并行机会，不把遍历位置当作完成证据，也不缩短外部输入、保留输出或别名读者的真实寿命。FastH3 的现有单 lane 执行可通过 `cuda.execution_order` 配置选择；两入口共享同一策略。分支复用、精确容量/少 1 B、迟读和保留输出回归通过；runtime 1015 通过/5 忽略，component 15 项、执行图 11 项、编译器 22 项通过。该顺序约束作用于单次 forward 内部算子，不限制每个 forward 所含请求数；B2 仍须真实联合执行。5090 和当前全工作区验证尚未完成，不预报容量或耗时收益。

只读视图 r2 已在 RTX 5090 完成真实 CUDA 资格验证：双方向 copy/view、B2 与非整齐批次几何、全部 BF16 编码、重复执行、实际 storage 别名和关闭后 owner 退役均通过（1 个组合 Rust 用例，1.42 秒，日志 `video-readonly-view-gpu-qualification-r2.log`）。这不是完整模型或性能验证。产品 required-flags 构建已于 2026-11-14 03:53:51（Asia/Shanghai）正常完成，冻结二进制 SHA256 为 `251f78f469f813971cb0795599e7cee7ac0839db94f71500d1fd2da361e14b2a`；输入、精度、8 步、36 层 decoder 与 28 GiB 预算未变。完整 CLI 已正常出片，MP4 SHA256 与原产品 r3/r4 的 `d3de4e5a…bd5a0` 一致；ffprobe 124 帧、832×480、24FPS、AAC 32kHz/双声道与完整解码通过。冷请求 115.978 秒、全进程 156.859 秒，只用于正确性及阶段诊断。B1 完整最低预留为 21,575,428,624 B，B2 为 31,224,390,128 B，仍比 28 GiB 上限多 1,159,619,056 B；相对 r4 的 B2 节省 5,023,854,592 B，尚不能共批。日志及摘要在库外 `video-readonly-view-run-r2.*`。

2026-11-14 03:36（Asia/Shanghai）：用户要求重试后，跳板 SSH 已成功返回 `Seekee5090`；内网直连仍超时。GPU 观察值 697 MiB、利用率 0%，仅为当时空闲观察。r2 增量已重新传输成功，继续以冻结候选执行 CUDA 资格验证与产品构建。此前 r1 构建实际为主动停止后的 exit137、OOMKilled=false，未执行 GPU 资格；不能算作完成构建或 GPU 失败。以下 00:37 阻塞记录保留为历史。

2026-11-14 00:37（Asia/Shanghai）：两条已授权连接在连续三轮目标执行中均未恢复；最新跳板 SSH banner 超时、内网 `172.31.40.3:2222` 超时、Tailscale ping 无回复。目标状态转为网络阻塞，目标未完成；r2 尚未传输完成或启动远端验证。恢复连接后的首步是校验并传输已冻结增量，运行现成 CUDA 资格命令，再进行完整生成及同机性能测量。审计记录位于库外 `video-readonly-view-network-blocked-r1.json`。

2026-11-14 瓶颈定位：同一 r4 二进制、原输入与精度使用隔离解压的官方 Nsight Systems 2026.6.1 完成 GPU trace（旧 2024.6 只有 API 数据，不作 GPU 分析）。全冷进程 204,366 个 kernel 的累计时长为 55.5005 秒；跨流累计时间不等于关键路径或端到端时间。BF16 dense attention 22.6517 秒、NVFP4 GEMM 9.9660 秒、稀疏 attention 5.2324 秒、仅形状桥接的 `transport_copy_bf16` 7,600 次共 3.4280 秒、索引 gather 1.8169 秒。产物与原 r4 的 MP4 SHA256 一致；原始 trace 留在 5090，筛选摘要为库外 `video-slot-profile-r2.kernel-summary.json`。

当前候选优先移除形状桥接复制：新增独立 `ReadOnlyView` 契约及显式 merge/split view 操作，保留原 NoAlias copy 行为。只允许同 dtype、完整相等字节范围的连续固定张量；所有别名读者共享生命周期，整组别名禁止任何后续写入，仅允许唯一原始 producer 初始化。元数据操作保留图依赖和容量所有权但不提交设备命令，不伪报 kernel。该候选尚在正确性验证，不把 3.4280 秒直接扣成端到端收益，也不声称完整 B2 已满足 28 GiB。另有 CPU latent 连续行拷贝候选，Mac 诊断约 3.86→1.87 ms/块，尚未建立 5090 收益。

只读视图候选 CPU 验证：runtime 1014 项通过、5 项忽略；execution 274 项与 3 项容量边界通过；Reference 的双方向 copy/view 真实管理执行、全部 BF16 位模式、重复执行与 owner 退役通过。FastH3 首轮 121 项通过、5 项因旧 fixture 强制 NoAlias/Write 失败；修正 fixture 按声明端口构造别名后，受影响 package 54 项全部通过。局部算子允许相同完整存储的只读输入，但全图仍拒绝未声明的别名；部分重叠和任意别名路径写入都有回归覆盖。执行图 11 项和计划编译器 22 项集成回归亦通过，保留原 MustAlias 覆盖保护、物理存储等价与计划 wire 校验。workspace all-target check、Metal 编译、Clippy（允许 warnings）、fmt/diff 检查通过；完整 workspace tests 尚未运行，不称 PR 全验证。

5090 候选 r1 已冻结并启动 required-flags 优化构建；检查发现 CUDA factory 和构造策略还未声明两项 view 能力，主动停止该构建，r2 已补齐并冻结（库外 `video-readonly-view-r2.source.json`）。随后 SSH/内网直连和 Tailscale 探测出现超时，45 KiB 增量传输中断；r2 CUDA 构建、GPU资格、实际成片及暖 C1/C2 尚未执行。恢复后先跑这些已有命令，不重复权重下载或改动已冻结源；不从消除复制的代码推断实际收益。此前已实测的性能主数据仍为 r4。

当前验收状态：**正式 `ferrum run` 与 HTTP 服务均已完成真实参考图条件下的 480p MP4；HTTP 两条已接纳请求在 B2 容量拒绝后均以 B1 完成，下载和完整解码通过。正式质量对齐、完整模型连续共批及快于 FastVideo 均未完成，当前 C2 延迟未达标。** 产物为 832×480、124 帧、24FPS、32kHz 立体声音轨；抽帧未见首版明显彩色重影。两层 DiT 的真实 B2 资格通过，不能替代全 50 层、8 步的共同 forward。下一步先定位完整 B2 计划的槽位容量与执行开销，再做保持原精度和预算的通用优化。以下组件证据不能换算为整体完成百分比。

槽位诊断 r1 已在原产品配置完成，容器 14:06:29–14:09:07 UTC 正常退出。只读诊断基于原解析后的资源与依赖图，揭示旧 B2 的完整最低预留为 39,101,272,240 B；此前 31,665,289,328 B 是逐池检查提前拒绝时的数字，不能当作完整 B2 占用。单纯按增量字节贪心复用仍需要 38,315,156,912 B，而且 BF16 池反而增大，故未直接采用这一策略。通用规划现对固定尺寸池比较原布局、按时间最小增长布局与大尺寸优先布局，仅在完整池的实际量化字节严格减少时替换；动态尺寸池保留原行为。所有候选沿用原外部输入/别名/保留输出隔离与逐成员 DAG 完成证明，不改变执行顺序、预算或精度。新增四项边界回归通过，覆盖容量少 1 B、无依赖不得复用、保留输出、局部贪心反例和大小交错链；Mac runtime 全量 1006 通过、5 忽略，fmt 与 diff 检查通过。生产源码候选为 `video-slot-placement-r3`；完整模型及真实 B2 的 GPU 验证尚待完成，不由小测试推断实际节省或性能。

生产布局 r3 随后于 14:35:20–14:38:19 UTC 完成正式 CLI 生成，exit0；MP4 SHA256 与正式产品 r3 的 `d3de4e5a…bd5a0` 完全一致，保留原音视频长度。二进制 SHA256 为 `bbc8627db72b3c93ce1e09c74fb0c06e15c79a44f8229700b6ef15e37c646aa5`。B1 完整计划最低预留为 24,539,142,160 B，B2 仍在逐池检查中以 30,806,659,184 B 拒绝（这是提前拒绝的下界，不是完整 B2 总量），因此没有继续运行已知会串行回退的 C2。冷请求阶段为 137.350582 秒、全进程约 179.06 秒，不作为性能胜出证据。该快照的 Mac 全工作区 check、Clippy（允许 warnings）与 Metal 编译均通过，execution 274 项单元及 3 项容量边界通过；完整 workspace all-target tests 尚未完成。

候选 `video-slot-placement-r4` 增加大尺寸优先的 first-fit 布局，并以有界祖先/后代缓存加速两种方向的依赖查询；每个共用槽仍检查全部成员。相邻成员推导的尝试被现有“无序使用者”回归拒绝，已撤回，没有放宽安全证明。新增小型 DAG 全枚举与独立传递闭包核验、双方向缓存/缺失节点错误核验后，runtime 实际 1007 通过、5 忽略，execution 274 项单元及 3 项容量边界通过，fmt/diff 检查通过。通用 tensor 执行在现有 DEBUG 日志下收集固定容量的 dispatch 分段耗时，默认路径仍使用编译期关闭的计时 sink；嵌套 host 时间不充作 GPU 时间。

r4 的 required-flags CUDA 优化构建及真实 CLI 已完成，运行于 15:09:39–15:12:22 UTC 正常退出，二进制 SHA256 为 `9d0d064b911d5a48f956c44d8e80864cf1fe949d0850524bb2f044b964f16113`，MP4 与已完整解码的产品 r3 字节一致。B1 完整最低预留降至 24,087,355,920 B；B2 完整最低预留仍为 36,248,244,720 B，超过原 28 GiB，继续合法回退 B1。布局优化没有解决完整共批，不能用提前拒绝的部分池数字声称 B2 更差或已经放入显存。冷请求 121.793 秒、全进程约 162.36 秒，包含 DEBUG 观测，不作为 A/B 提速证据。原始数据位于 `evidence/video-slot-placement-{build,run}-r4.*`。

分段观测已把单轮去噪约 6.1 秒中的约 5.2 秒定位到 CUDA 后端 command enqueue；Rust 节点准备及绑定仅约 0.04 秒。enqueue 的 host 区间可能包含 GPU/驱动等待，不能直接当作可消除的 CPU 开销。下一步以 CUDA API 与 kernel 时间线区分真实计算、同步和驱动阻塞，优先改实测热点。与此同时运行默认日志级别下的相同参考暖进程 C1/C2 诊断（`video-slot-warm-r4`，每格先预热一次、测量两请求、一次重复），与既有 FastVideo 暖缓存诊断比较；保持预算、输入和精度，披露 FastVideo FFN 4096 分块与 Ferrum 当前整行 FFN 的实现差异。此小样本仍不替代正式新参考、配对重复和质量验收。不同种子的真实 B2 验证输入已准备，待容量允许共批后执行。

暖进程 r4 的 C1 测量已取回：71.591420 / 71.080874 秒，经验 P50/P99 71.336147 / 71.586315 秒，0.841086 clips/min，两条完整解码通过且与原产品 MP4 字节一致。相对既有 FastVideo 暖缓存诊断的 P50 52.711489 秒，当前中位延迟仍高约 35.3%；不能把去掉冷启动后的变化全归因于布局优化。C1 采样峰值 VRAM 26,914,848,768 B、进程树 RSS 43,505,115,136 B，685 次成功、0 次失败、30 个跳过间隔、39 次超期，`resources.complete=false`，峰值仅为采样下界。15:24 UTC 观察到客户端已退出 1、六条含预热的请求均已退休、服务拒绝数 0，随后正常 SIGTERM 退出 0；完整 C2 报告已归档远端，但 SSH 跳板故障暂阻取回，尚不填写未取得的 C2 数字。已使用冻结 r4 二进制派发 Nsight CLI 诊断 `video-slot-profile-r1`；须实际检查 trace 完整性和 GPU 兼容性后才解读，诊断不进入性能主表。

恢复连接后，暖进程 C2 完整报告已取回：80.913397 / 143.625558 秒，经验 P50/P99 112.269477 / 142.998436 秒，0.835506 clips/min；2/2 下载及完整解码通过、零生成错误，均为原 MP4 字节。相对 FastVideo C2 的 1.274351 clips/min，当前吞吐低约 34.4%，且 P99 超过 120 秒。GPU 661 次成功/1 次查询超时，Host 662 次成功，跳过 58 个间隔、26 次超期，采样仍不完整；VRAM 26,914,848,768 B、RSS 43,538,952,192 B 为观察下界。完整报告与终态保存在 `evidence/video-slot-warm-r4/` 及同前缀文件中。六条含预热请求共 48 次 B1 执行、8 次缩批，未建立 B2 共批或性能胜出。

基线原始报告还记录了服务端阶段时间：四条暖缓存请求的去噪均约 43.13–43.18 秒，视频解码约 1.48–1.53 秒，conditioning 约 0.04–0.05 秒。Ferrum 冷诊断的 tensor host 区间合计去噪 49.38 秒、视频/音频解码 4.90 秒，而包含 decoder 的收尾区间约 10.31 秒。两侧计时边界和冷热不同，不能直接相减作为严格贡献归因；但解码的 Host 拼接/转换与重复提交需要和去噪 CUDA 一起排查。Nsight r1 在 15:25:16–15:27:51 UTC 正常退出；实际只获得 612,523 条 CUDA API 记录，未获得 kernel 活动表。诊断明确警告该 2024.6.2 版本不支持安装的 CUDA driver 13.3，因此不能声称已取得完整 GPU 时间线。API 汇总中 `cuLaunchKernel` host 区间累计约 40.58 秒，仍不等于 kernel 时间；正在使用任务目录中的新版 CLI 重新采集，不更换 GPU 驱动。

产品接入的新证据：共享 CUDA runtime/provider/actor 初始化实际 5090 测试 r2 通过（`evidence/video-engine-provider-gpu-r2.*`，18.63 秒），此前 r1 因无 VSA regions 的 registry 错误声明 region operation 被拒绝，现按实际能力修正，未放宽容量。视频 facade 4 项、取消/core 6 项、原图片容量唤醒 4 项、MP4 spool 自唤醒回归 1 项通过。HTTP 状态/输入/registry 7 项与真实 ffmpeg MP4 下载 owner 测试 1 项通过。目标 latent 提取 2 项验证参考区域排除、原 ID/shape/Host parent 与容量回滚。CLI CPU 编译及两项媒体/文本参数分流检查通过；完整 CUDA CLI（含 required vllm flags）r2 编译、r1 可执行构建通过。最终产物由原 MP4 owner 读取后在同目录无覆盖发布；run 的信号取消继续等待原 job 与 worker 退役。

首次正式 `ferrum run` 请求于 2026-11-13 11:47:07–11:52:28 UTC 运行，exit1，未产出 MP4：计划要求 31,665,289,328 B，超过 28 GiB 的 30,064,771,072 B 预算（`evidence/video-product-run-r1.*`）。其单请求路径强制编译可选 B2；已改为仅对 B2 的精确 `PlanCapacityExceeded` 保留合法 B1，其他错误仍传播，设备上限未增加。同期补齐原 source Arc 随 denoising stage 的真实生命周期；按原 BF16 tile/temporal/audio owner 并存计算请求 Output 上界，480p 仍由 384 MiB floor 覆盖，不再使用 RGB×5 的粗略预留。

优化构建后的正式 CLI r2 已越过上述可选 B2 容量失败，于 12:15:08–12:16:38 UTC 因 RoPE 输入 rank 不符退出（`evidence/video-product-run-r2.*`），未产出 MP4。模型适配现从原二维位置输入构造 `[1,rows,rotary_dim]` 的 B1 view，保留原 ID、同一 byte region 和 Host owner，不增加分配，不放宽执行端 geometry 校验；Mac 独立 trig/owner 测试实际 1 项通过（`single-request-rotary-mac-r1.log`）。

正式 CLI r3 已于 12:57:15.134–12:59:53.066 UTC 成功退出，发布 `ferrum-product-480p-r3.mp4`（3,463,109 B，SHA256 `d3de4e5a70933e54f2187f1277cc5274cc1a94f85b999348af028e8d511bd5a0`），ffprobe 计数 124 帧、832×480、24FPS、AAC/32kHz/双声道，完整视频及音频解码 exit0。同一 reference/prompt/seed1101、8 步和 36 层解码器、28 GiB 设备预算保持不变。`cuda-correctness` 优化二进制 SHA256 为 `26bebc60195353864464222dd35d1cd7a2f7f57d9c4ba9ba63ee5a32d4120f2a`；请求阶段计时 116.280829 秒，含进程初始化的容器总时长约 157.93 秒。该次为首次正确性运行，期间有一次约 3.6 秒的 CPU 客户端构建，没有完整资源峰值采样，不进入性能对比或 SLO 判定。原始启动命令、二进制 hash、终态、stdout、ffprobe 与 full-decode 记录均在 `full-forward-integration-r1-stable/`。

同一二进制的真实 HTTP C2 功能验收已完成（`video-http-c2-r3/summary.json`）：1 次重复、无 warmup、两条相同图片/prompt/seed 请求，2/2 下载及完整解码通过，零生成/证据错误、无未知终态，两个 MP4 的 SHA256 均与 CLI r3 完全一致。结束时服务健康，16 次执行、8 次容量缩批、最后批量 1、拒绝数 0、active/pending/preparing 均为 0；该结果证明本输入下已接纳任务可以退化串行完成，不是全模型 B2 共批证明。服务收到 SIGTERM 后正常退出 0，原容器/日志/UUID/产物均保留。

该 C2 诊断的完整产物耗时为 189.913184 / 127.983248 秒，经验 P50/P99 为 158.948216 / 189.293884 秒，吞吐 0.631868 clips/min（1.305860 生成帧/秒）；未满足预声明 P99≤120 秒。TTFT/TPOT/ITL 不适用于此无文本 SSE 的异步视频协议。采样观察到 Peak VRAM 26.379 GiB、服务进程树 RSS 42.486 GiB；942 次成功、0 次失败，但跳过 9 个间隔、30 次设备查询超期，`resources.complete=false`，峰值仅是采样下界且 RSS 可重复计算共享页。客户端原始 exit1 保留，不能以 2/2 功能成功宣称性能验收通过，也不能由两个样本建立稳定 P99。完整远端报告保留；首次 SSH 取回的截断副本另存 `.partial`，本地结论依据随后从原报告提取并完整取回的摘要及逐请求验证文件。

用户要求的容量回退已有最小行为证据：Mac Rust 测试实际 3 项通过（`capacity-fallback-mac-r1.log`），覆盖 B2 typed capacity 与动态预检回滚拒绝时两条已接纳请求均以 B1 完成、没有运行中 GPU 请求时等待真实 Output owner 退款后继续，以及真实 Fatal 保留原失败 owner。Reference 后端动态预检实际 1 项通过，预检不提交 GPU 工作并保留原 B1。5090 r5 验证容器 exit0，恢复 SSH 跳转后已取回逐项日志：相同容量测试 3 项、RoPE 测试 1 项、视频基准输入协议测试 3 项通过，完整 required flags 的 CUDA CLI check 通过。CUDA 动态预检压力 fixture 随后实际匹配 1 项并通过（`evidence/video-dynamic-preparation-gpu-r5.*`，2.17 秒），覆盖实际 CUDA quota 拒绝后原 B1 继续、释放后 B2 执行；该次拒绝在物理 backing 分配前发生，回滚设备字节为 0，不能据此声称已覆盖部分物理分配后 OOM，更不能替代完整模型双请求出片。

刷新冻结源的远端 mtime 后，r3b 三项 Rust 检查各匹配 1 项并通过：最后一个 decoder 请求在安全边界取消后，原服务可接新请求；空间解码精确 Host 上限成功/少 1 B 回滚；新的 prepared-plan 初始化与原路径在 mixed-evaluation fixture 中保持独立提交并回收 Host。r3 首次取消命令误用旧 Cargo test binary 导致 0 匹配，不记通过；同轮 mixed-evaluation test 因测试仍持有原 plan/region 的 120 B owner 失败，释放该真实 owner 后通过，未放宽零回收断言。后续传输显式刷新 source mtime。完整 50 层 B2 的真实动态容量预检和 HTTP GPU 请求仍未完成，不把编译或 C2 客户端数当作连续批处理证明。

最新实物结果在库外 `full-forward-integration-r1-stable/`：文本编码器使用检查点声明的 64 层中的全部 50 个已导出层，原权重实际上传 13,714,459,004 B，诊断 token 输入输出 BF16 `[3,5120]` 全部有限；真实完整 36 层视频解码器执行 1162 节点，原权重实际上传 1,374,793,876 B，非零诊断 latent `[1,24,6,4,4]` 输出 18 帧 64×64 RGB。两者均正常关闭并回收 Host。解码器实际暴露的 cuDNN D64 支持缺口已修复并由该运行覆盖。以上不证明提示词语义、生成质量、480p 容量或性能胜出。

真实全 50 层 DiT 已在 5090 执行 3150 节点、单次上传 11,922,924,656 B，8 个不同条件 evaluation 均输出有限 BF16 `[28,5376]`，源读取保持 1359 次、Host 最终归零。该诊断每次使用固定 hidden，不是 PDD 更新，也不是生成质量或性能比较。完整图片参考 CNN 同样已实际执行原权重 71 节点，非零 `[1,3,64,64]` 输入输出 F32 posterior moments `[1,48,4,4]`；这条 T=1 路径不能替代多帧视频编码。

本轮已在 5090 跑通完整 Media/Packing/50 层 backbone/输出 projection 与实际 PDD 更新，8 步全部正常提交。`full-forward-integration-r1-stable/pdd-diagnostic-r1-output/manifest.json` 记录主干静态初始化 1 次、每步 source reads 固定 1363、最终 Host 为 0，产物为 F32 video `[8,96]` 与 audio `[16,32]`。输入是已声明的诊断零 latent 和诊断文本特征；同轮最终 latent 已通过原 36 层视频解码器及完整音频解码器，得到 64×64、5 帧、24FPS、双声道的 `pdd-diagnostic-r1.mp4`。ffprobe 与完整解码均通过，两侧 decoder 的 Host 最终为 0。这是完整计算链诊断，不能称真实提示词生成视频或质量/性能胜出。复用已验证的 StoredComposition 消除了重复构图；同一 CPU fixture 从约 20.14 秒降至 3.34 秒，只作为本地准备阶段诊断。

音频 F32 转置卷积已有 Reference 2 项、CUDA 构建下 cuDNN owner 10 项检查；实际 5090 backward-data 的 3 组不同几何均与独立 F64 scatter oracle 精确一致，并验证重复执行和关闭后 Host 归零。完整音频解码图的原来源/实际 Reference 执行与容量检查 2 项通过，包含 729 节点、两个请求的独立立体声输出；原权重音频 GPU 消费者亦已通过同轮 PDD `[16,32]` 输入，输出 6400 帧、32kHz 立体声。

图片参考后验的 fresh seed 42、F32 sample→F16→F32、原 channel statistics 归一化已成为共享模型适配；数学/批次独立性/Host 容量 2 项和单流 RNG 1 项 CPU 检查通过。它直接借用执行输出的受管 bytes 与 shape，不把 Output 伪装为 InputPreprocess。CPU scalar-libm RNG 明确不声称 Torch AVX2 逐位一致；原权重 CNN 输出接该适配的 5090 消费者亦已完成，输出 normalized F32 `[1,24,1,4,4]`；仍是诊断像素输入。完整音频解码图和实际随机输入生产继续集成，尚不改变上述未完成验收状态。

真实参考准备已核原部署：Ref2VA 先按 short edge 2048 和 32 对齐做 RGB Lanczos，832×480 原图对应 3552×2048；同一 prepared 图必须交给视觉模型和 VAE。原 VAE 已启用 256×256 tile、64 overlap；整图 GroupNorm 不能作为等价替代。共享参考画布/原 F32 像素归一化 1 项、视觉 BF16 processor 2 项和复用 resize 回归 1 项通过。复用 causal tile plan 与原邻居 moments 拼接后，实际 CNN 分块对照 1 项、共享 canvas 4 项和例子编译通过。5090 已用原真实参考图执行 209 块，moments `[1,48,128,222]` 与 normalized posterior `[1,24,1,128,222]` 完成，Host 最终归零（`reference-keyframe-r3.report.json`）。同一 prepared 图的原 27 层视觉模型和 3 路 DeepStack 也已完成，四份 BF16 `[7104,5120]` 输出有限且 Host 归零（`reference-vision-r3/manifest.json`）。首次 sequential attention 运行因耗时过长主动停止并保留状态；后续明确配置已有 cuDNN segmented attention，未改变 D72 或 Q/K/V 精度。长序列因果 GQA 复用原 cuDNN owner/cache/workspace，以官方虚拟图保留紧凑 KV 和严格因果；5090 的 B2/S129/Hq16/Hkv2/D128 两次执行对独立 F64 oracle 最大绝对误差 0.01869，在预先声明的 0.04 fixture 界内，Host 回收归零。真实图文已随后跑完 7175 token、全部 50 个已导出文本层，BF16 `[7175,5120]` 特征有限并关闭正常（`reference-text-r5.report.json`）。真实参考后验与上述特征已在 832×480/124帧/seed1101 下完成8步 PDD；主干只初始化一次、source reads每步1363、Host最终0，输出video `[14430,96]` / audio `[414,32]`。原36层视频与完整音频解码后得到 `ferrum-reference-480p-r1.mp4`，124帧/H264/24FPS/AAC立体声/32k，完整解码通过。**视觉质量未通过**：抽帧存在重影和纹理异常；已核原baseline decoder也启用256像素tile/64 overlap，而首轮消费者为整空间attention，当前恢复原tile语义后将重解同一latent。该首样片不证明质量对齐或速度胜出。失败记录包括旧例子二进制拒绝新枚举、packed行数上限未覆盖VSA padding；后者将显式maximum_rows设65536，设备预算仍为28GiB。

空间 decoder 已实现原 256 像素 tile、最小重叠 64、局部 RoPE、原始邻居 BF16 分步融合。480p 使用 12 个空间块 × 7 个时间窗口，同一 session 只上传一次权重。两个 CPU 正确性测试和 example check 通过，CUDA 增量构建及同 latent 重解均取得真实 exit0。新产物 `ferrum-reference-480p-r2.mp4` 完整解码通过；0/2/4 秒抽帧中首版明显彩色重影与扭曲纹理消失，但动作与基线单样例不同，尚不建立正式质量对齐。该次 debug 解码用时不作性能胜出证据。

双请求实际 B2 模型路径已有 CPU 混进度/换槽/提交回滚验证；两层 CUDA 数值资格 r3 在同一 runtime、原生 owner 与共享权重安装下通过：133 compute + 7 输入上传共 140 commands，67 个来源读取一次、上传 269360 B；evaluation 0/1 后换槽为 1/2，与两个 B1 对照的最大 BF16 ULP 为 0，第二份 prediction 失败时双方不提交，所有 owner 正常退役。前两轮测试配置漏算提交命令而被容量检查拒绝，失败记录保留，生产容量保护未放宽。该资格只有两层，不是完整 50 层视频连续批处理。

共享 MP4 输出库已完成 H264/AAC 编码与受管输入持有、有限 spool、取消后的 kill/wait 及最后 reader 退役；38 项普通测试、4 项真实 ffmpeg 测试和独立音视频尾长测试通过。它保留 124/24 秒的视频与 165600/32000 秒的音频，不截断较长音轨。尚未接入产品服务，外部编码进程 RSS 不冒称已有硬限额。

下一步把已实跑的参考编码、PDD 共批与原视频/音频解码接入现有 iterative worker 和共享 run/serve。各阶段接收同一 runtime/Host/serial actor，在实际 fence/close 后转换驻留，不把例子子进程或中间文件当产品路径。按效率要求复用保存的 conditioning/latent、增量传输和既有 Cargo 缓存；完整链路稳定后再做 workspace 与性能验收。

内网直连已于2026-11-13 05:32 UTC实际验证：`ssh -p 2222 seekee@172.31.40.3`使用现有密钥成功到达Seekee5090，后续同步/构建观察使用这一路由；原Tailscale跳板仍保留。该次观察显存697MiB，仅表示当时占用，不是峰值或空闲基线。`origin/main`重新核查仍为本目标所基于的`7bbea02877df04132ae7407a194ecb9d669265c2`。

2026-11-13：目标与基线来源已确认，独立工作树已建立。5090 原 `ferrum-h3-image.service` 已正常停止，定义/配置保存在库外；空闲时观察GPU占用约84MiB。FastVideo 服务已启动用于基线，未启用开机自启动。新 Rust 客户端已完成一次当前配置的冷进程首请求诊断：832×480、124 帧、24 FPS、带音轨；完整解码通过。服务报告推理82.686秒，客户端观察到终态84.924秒，接收完整产物110.627秒。该客户端位于Mac并经过SSH转发，只有一次样本、没有完整显存采样，不能作为正式同机比较或稳定P99证据。产物与报告在证据根 `fastvideo-pilot-cold-r1/`。

设计已完成核心所有权、PDD模型语义与性能合同的独立审阅，修订了物理授权边界、唯一worker、版本化提交、混进度输出projection和产物容量闭环。内容、相对文件链接、固定上游链接与diff检查已完成。

首批基础实现已进入独立工作树：NVFP4来源/布局定向Rust测试10项通过；FastH3受管PDD continuation具备独立音视频时钟、冻结参考和prepare/commit回滚，加上完整权重包合计15项FastH3测试通过。通用时空latent/patch合同4项测试通过，已有P-H3图片packing与FastH3几何使用同一合同，P-H3编辑prefix独立oracle回归2项通过。Rust异步视频基准客户端16项测试与复用资源采样核心10项测试通过。

5090本机暖进程、重复相同参考的诊断已完成（`fastvideo-pilot-local-warm-r1/report.json`）：C1两次完整产物耗时58.332/47.091秒，经验P50/P99为52.711/58.219秒，1.138 clips/min，采样显存25.292GiB、服务进程树RSS38.117GiB；C2同时提交两次，完整产物耗时94.166/47.346秒，经验P50/P99为70.756/93.697秒，1.274 clips/min，采样显存25.296GiB、RSS37.680GiB。每格一次重复、两请求，4/4完整音视频验证通过、零错误，观察值满足预声明120秒阈值；样本不足以建立稳定P99。200ms采样分别529/472次，零失败/漏采；显存是采样下界，进程树RSS可能重复计算共享页。此缓存命中诊断不进入正式新参考A/B数据。计时期间暂停其他任务构建与GPU工作；结束后已确认全部任务终态并停止FastVideo，GPU回到约84MiB。

实际checkpoint已通过5090上的Rust CPU inspector检查（`metadata/package-inspection-r2.json`）：350个NVFP4矩阵覆盖50个block，338个dense参数，共688个逻辑权重、1838个物理组件；350个权威FP32 alpha及100个静态输入scale通过验证。`skip_blocks=[]`，未发现冗余的原始量化矩阵。源payload为39,675,309,764字节；检查包括完整清单与NVFP4 scale，未逐值扫描dense payload，未执行forward。这也要求驻留计划使用明确可验证的AdaLN预计算，不能将全源包直接加载到32GB显存。原生CUTLASS C ABI已用兼容工具链编译通过；NVFP4布局/ABI测试与统一静态模块容量owner定向测试通过，FP8/既有融合算子真实迁移到同一受界限机制。CUDA受管provider正在做实际GPU资格验证，尚无SM120数值结论。图像服务唯一worker向通用迭代core的迁移已开始；量化权重到图计划的物理视图仍需补齐，不能用伪Dense U8绕过格式与所有权。

受管SM120算子资格已取得第一份证据（`nvfp4-managed-qualification-r2.json`）：dynamic输入的`(M,N,K)=(3,8,128)/(129,136,256)`及Stored量化权重的`(129,136,256)`，各运行两次，独立FP64 dyadic参考转BF16后逐位一致；覆盖128行scale边界、padding隔离、原Stored来源只初始化一次、上传/工作区/提交/读回和关闭后的Host回收。Stored路径保留PackedValues与Quantized编码、原flat scale及运行时arena；不是把量化权重伪装为Dense。该微型fixture不证明实际FastH3层、任意输入精度或端到端性能。

原图片服务的队列、唯一worker及step/cohort/completion状态机已实际迁到`ferrum-engine::iterative_service`，图片facade和Qwen/P-H3消费者直接委托；随后容量/search/efficiency和lifecycle事件也迁为通用合同，图片wire观测由adapter转接原有有界collector。Linux图片领域431项、通用core10项、旧image_trace兼容1项通过，另有独立时钟/动态加入的非图片请求fixture。它证明共享生命周期，尚不是GPU视频连续批处理。Rust内部分阶段事实现在使用`ExecutionStageKey`，部分公开Rust struct literal需显式转换；HTTP/trace字段保持原格式。CLI `run` 的单请求生命周期迁移仍未完成。

本机测试曾因ENOSPC中断，已清理可再生dev缓存并将主要检查移到5090；中断的检查不计通过。原生首次宿主编译遇到CUDA12.8与宿主glibc头不兼容，改用已有Ubuntu24构建容器，不修改系统头。首次全workspace Metal编译发现新物理视图缺少显式拒绝分支，已补齐；随后Metal全工作区编译通过。物理组件与静态初始化43项、NVFP4格式/舍入14项定向测试通过。

基础集成快照`source-integration-r3`加模型测试编译修正已完成workspace check、Clippy（`-A warnings`），固定快照的required CUDA CLI flags加NVFP4编译通过。完整all-target测试观察到8147通过、82忽略、1项trybuild因原共享cache权限失败；该trybuild单独恢复默认容器权限后通过，完整单次命令的exit101仍保留，不冒称一次全绿。此前并行链接被signal9终止、镜像缺Ruby、root绕过文件权限测试等环境失败均记录在`integration-validation-r3.json`，改为单任务链接及已有Ruby镜像，没有删除或禁用测试。此快照早于下面的run和新模型组件，不自动验证后续修改。

后续切片：`run`已改为封闭有限队列驱动同一worker，保留真实shutdown receipts及失败owner；29项最小回归通过，较广图像与CLI CUDA回归待新的集成检查。FastH3静态QAD FFN接入原始B/scales/SF和受管派生alpha，模型来源/实际节点绑定等20项FastH3测试通过；激活量化CUDA provider与独立FlashInfer数值对照正在实现。F32 unary SiLU已进入原pointwise执行路径；TimeMLP→F32 SiLU→BF16 AdaLN组件、独立请求时钟和受管sinusoidal输入5项测试通过，含Reference实际执行/F64 oracle和真实容量拒绝。CPU libm与GPU libdevice身份分开，尚无完整FFN或AdaLN的真实5090数值结论。

这些只证明已测基础合同和组件，**尚未建立 Ferrum 视频推理、视频连续批处理或性能胜出结论**。VSA、完整transformer、参考编码、36L时序视频/音频解码和产物产品路径仍需实际接通。

继续推进后的数值审查发现上述 fusion 优先级差异，已从实际 Triton 3.7.0/SM120 缓存取得 SwiGLU PTX：`ex2.approx.f32`、`div.full.f32`、两次 F32 乘法后仅一次 BF16 转换。新 packed SwiGLU 合同保留该顺序，FFN 已切为五节点模型图；旧 SiLU 合同保持原行为，Reference 与 GPU 的指数近似身份分别记录。最新 FastH3 CPU 测试33项通过，包含时间调制、FFN来源/图合同与8项VSA元数据测试，仍不等于新算子 GPU 数值通过。

[VSA 模块说明](../crates/ferrum-models/src/fast_h3/vsa/README.md)固定 tile128、空间边界 tile 的真实有效行、dense prefix、分区域 top-k 预算和每 batch/head 独立动态 mask 的降级规则。现阶段输出 checked block lists/half-valid；pooled QK 选择和 FP4 sparse attention 的 GPU provider 尚未完成。上游 FP4 attention/quant 使用 fast-math，必须单独固定其原生编译 profile，不能沿用 GEMM 的精确编译选项后称同数值实现。

[AdaLN 预计算工具](../crates/ferrum-models/examples/precompute_fast_h3_time/README.md)已成为时间调制组件的实际消费者：构造完整 PDD 时钟并集，逐 block 上传原始权重、执行、关闭，再写带来源/计划/时间路由的 safetensors；全部选择 block 成功后才发布 manifest。它与 FFN 工具复用 bounded package loader 和失败 owner 保留机制。时间组件5项测试及工具CPU编译/2项入口与失败保留测试通过；真实 checkpoint 的 CUDA 执行尚待运行。NVFP4 激活量化/GEMM 原生库和 Rust CUDA 测试二进制已编译，独立 FlashInfer 参考库也已编译；本轮长 SSH 连接中断留下无终态测试，未计为通过，改用独立连接核对进程与日志后重试。

纯量化 producer 暴露了原数值 profile 对所有主要输出一律要求浮点的领域耦合。核心已将规则分层：结构 profile 要求主要值属于实际 node output boundaries，TensorComponent 可产生整数/编码存储；CausalLanguage 在 typed definition 与 prepared-family 构造时继续强制浮点。`primary_storage_type()`显式读取存储边界，旧`activation_type()`继续只返回浮点；语言限制由`ModelSemanticMetadata`承担，通用数值模块不依赖语言语义。没有增加假浮点输出、放宽实际推导 dtype 或改变旧 wire 字段。52项模型wire/数值profile合同测试通过（`tensor-storage-primary-contract-tests-r3.log`），覆盖语言路径拒绝、typed/wire伪造、输出集合和旧profile编译/指纹；首轮只因测试ID构造器错误编译失败，已记录。GPU测试改为独立容器保留终态后，新版packed GEMM真实exit0；纯激活量化资格正在继续，旧无终态SSH日志仍不计通过。

后续独立容器取得量化与GEMM终态（证据根 `nvfp4-quantizer-qualification-r6.json`、`nvfp4-quantize-r6.log`、`nvfp4-gemm-r6.log`）：RTX5090 激活量化测试1项、packed GEMM回归1项均exit0。量化38组输入/global组合各重复两次，覆盖K16尾部、128行scale边界、全部有限BF16值及每个正有限BF16组最大值、极小/极大全局scale；packed/scales逐字节符合独立固定FlashInfer参考库，padding为零。PTX审查识别并显式实现基线的`setp.eq.ftz.f32`，使subnormal组最大值进入零分支；不是依靠整个生产库fast-math掩盖数值顺序。GEMM三例含真实Stored物理组件桥，独立dyadic FP64→BF16 oracle逐位通过，覆盖来源仅初始化一次与关闭后Host回收。旧SSH无终态日志仍保留、不计通过；本快照早于后续core API分层收口。此证据不证明完整FFN、encoder、视频质量、模块显存实测上界或性能胜出。

固定快照 `source-execution-r6`（SHA256 `fae11991cf6537d9317e809969f281e1ffb1d1502e80421c5c1f2f5191699140`）的两个 CUDA 执行例子已编译通过。完整五节点 FFN 在5090上取得两种证据：129行小型dyadic fixture两次执行逐位符合独立FP64/BF16完整FFN oracle；真实checkpoint第0层、3行输入两次执行均有限且输出SHA相同，原始8个物理来源只在初始化读取，close后保留输出仍有效、释放后Host计费归零。真实层尚无独立整层数值对照，不能由有限/重复结果推断模型准确性。原始报告与容器终态在证据根 `evidence/ffn-{dyadic,real}-r1.{report,state}.json`。

同一快照的真实第0层 TimeMLP/AdaLN 也已在5090完成：17个联合时钟值产生F32 `[17,2688]` embedding与BF16 `[17,96768]` modulation；两次输出有限且逐字节相同，6个来源只在初始化读取，关闭并释放输出后Host计费归零。实际上传583,758,336字节，逐层预计算使用2GiB设备准入额度；额度不等于实测显存峰值。完整manifest在 `time-conditioning-block0-r1.manifest.json`，保留来源hash、计划、初始化和退役receipt。CPU libm与CUDA算术身份仍明确，尚未建立Torch逐位或完整视频质量对齐。随后50层表预计算取得真实exit0（`time-conditioning-all-r1.manifest.json`、`evidence/time-all-r1.state.json`）：0至49层各两次输出有限且相同，source读取次数保持6，全部关闭并释放后Host归零；modulation payload总计164,505,600字节。该数值是表数据大小，不是整模型显存或速度结论。

同快照的融合SwiGLU独立GPU测试也exit0：`(rows,width)=(2,17)/(3,129)`各执行两次，覆盖不齐整形状、原BF16分阶段舍入反例、signed zero及特殊值；相对保留指定F32边界的独立FP64-exp2参考，有限非零值最坏BF16 ULP为0，输出/owner退役通过。证据为 `evidence/packed-swiglu-r6.{log,state.json,sha256}`。这是声明输入域内的算子资格，不外推至完整层或视频质量。

后续Time输出到调制的模型桥已落地：`FastH3ConditioningTable`保持真实Output owner，检查两个输出的ID/dtype/shape与有限性；匹配block、宽度及精确时间行后，才为下游申请一次独立InputPreprocess copy。实际Reference执行链覆盖Time关闭后继续使用、两种调制阶段、同time不同text/vision标签、错层/错时间拒绝、非有限输出回收及真实Host额度拒绝，最小测试通过（`time-modulation-bridge-cpu-r2.log`）。首轮fixture遗漏Ref2VA必需参考而失败，已改为合法单图参考并保留r1日志。该桥声明当前模型adapter绑定live输出，不把任意磁盘表的同shape当作checkpoint来源证明；此后续修改未包含在r6 GPU证据中。

融合归一化/调制取得独立r7 GPU证据：宽度2/5376/8191，各运行普通norm与residual+norm两种模式，每组两次。相对独立FP64参考，H2普通norm最坏2 BF16 ULP，其余五组为0，符合事前声明的≤2 ULP预算；两轮输出及关闭/Host回收通过，不称FastVideo逐位等价。来源为r6加15文件RMS overlay（`rms-execution-r7-source.json`），原始终态/日志/二进制和实际编译源hash保留在 `evidence/rms-r7/`。首编译exit101暴露了不同快照同挂`/source`、归档保留旧mtime导致Cargo复用旧库的问题；核对远端新源码后刷新隔离快照mtime，真实重编runtime/kernels后exit0，再运行GPU。旧失败在 `rms-build-r7-failed.log`，没有作为数值失败隐藏或计入通过。

实际NVFP4文本encoder接入又发现global_sf的来源合同缺口：普通受管Dense F32 unit常量与原checkpoint Metadata F32都合法。核心新增显式`NumericalOrPhysicalComponent`，只有该量化数值参数端口声明两种来源，分别执行原数值或精确physical role/encoding验证；原physical-only权重端口及默认数值端口规则不变。6项核心合同、4项Qwen NVFP4来源/完整Stored端口测试通过，先前真实绑定失败保留。文本图复用既有Qwen3-VL，明确原64层、执行前50层及pre-final-norm输出；还未执行完整encoder。真实H5120超出现CUDA HeadRMS的pow2≤4096范围，正在按独立后端reduction plan扩展，保持两次BF16舍入；不能将CPU图通过当作GPU encoder已支持。

Q/K/V/gate投影现为一个实际模型组件：共享一次unit-scale量化，再执行四个GEMM，输出宽度按真实`heads×head_dim`（当前7168），保留原F32 alpha与physical权重来源。最新FastH3 CPU测试40项通过（`evidence/models-qkv-fast-h3-r2.log`），包含完整13个来源/端口绑定与原FFN回归；首次图profile遗漏内部U8输出而失败，已按实际node输出补齐，原失败保留。独立dyadic oracle的受管CUDA模型测试已写，尚未编译/运行；gate裁剪、QK norm与RoPE未由此证明。

Ref2VA坐标/RoPE producer已接入共享受管输入：`FastH3PositionPlan`同时构造原有packed layout和有界region元数据，逐行生成位置，不另存完整坐标tensor；保留image/audio/video参考顺序、video与附带音频共享origin、双声道位置及独立目标时钟。解析inverse frequency遵循F32边界，使用声明的libm CPU profile，BF16 cosine/sine均由现有InputPreprocess账户持有；不称Torch CUDA逐位等价。5项CPU测试通过（`evidence/fast-h3-positions-cpu-r2.log`），覆盖混合参考、不同宽度、37个latent的时间周期、独立F64 trig对照与真实Host容量拒绝/回收。QK融合consumer仍在实现，此producer尚未由GPU完整attention执行验证。

VSA取得首个受管执行资格（证据根 `vsa-managed-r1-stable/`，远端 `vsa-managed-r1/`）：固定源码快照SHA为`f3287af01d3de824c51edd5a9cf514eec7883216c2e7c2d72ad33bc90dfb3b42`，仅复用实际旧4包的catalog行、刷新NVFP4并追加Sage后，6包锁SHA为`cbee891750da2256a10f3c4299b7e5c76975787d59778f8eac84ca34a080119f`。CPU几何2项、原生ABI6项、Host owner/layout4项通过；Linux CUDA编译和独立5090容器均取得真实exit0。GPU三组`(B,H,Qblocks,KVblocks)=(1,1,1,1)/(2,3,2,4)/(1,2,2,3)`各两次，Q=K=0、dyadic V，557056个BF16结果逐位符合独立均匀attention平均值oracle；覆盖每head独立矩形列表、不同query稀疏选择、64行半块、错误digest在prepare前拒绝、同一Core提交/读回及Host计费退役。三份原始TransferStaging owner及其padding通过现有Host账户、pinned registry、prepared retention和command quarantine持有，没有另建分配器。测试还识别固定上游正常noncausal路径未写LSE；新operation只发布Y，ABI的LSE地址保留为已计费scratch，不读回未初始化结果。该资格仅验证不可变Host pattern的路由、有限特殊算术与成功路径owner；**不证明一般softmax/两级P舍入、GPU动态mask、QKV量化、注入后的部分上传失败、完整FastH3或性能**。下一切片是由实际pooled QK、区域预算产生动态模式及FP4前处理，不能把此固定pattern fixture当成完整VSA。原生源码/许可证与编译闭包见[Sage入口说明](../native-operators/cuda/sage-vsa-fp4/README.md)。

QK融合与实际PositionPlan输入桥的Reference串联已通过，最新FastH3 CPU测试49项通过（`evidence/models-fast-h3-qk-r3.log`）；其中包含PDD源算术定向测试。该CPU闭包未替代CUDA QK/RoPE资格，尚无完整attention或视频结论。

PDD最终projection已成为实际Stored F32组件：按音频/视频及evaluation选择原channel-major widened heads，F64归一化后以F32系数融合原参数，精确记录原dtype/content SHA、区间、integration bits、coefficient bits和CPU separate-mul/add profile。两份融合参数直接由原Host WeightBacking账户持有，没有全局fusion游标或未计费F32源展开；不同请求进度通过不同来源身份绑定。源算术/非法值2项通过（`evidence/fast-h3-pdd-source-cpu-r1.log`），真实package→融合权重→Reference affine及容量/身份拒绝2项通过（`pdd-projection-managed-cpu-r2.log`）。两种模态各两次执行符合独立raw decode与F64 affine参考，source仅初始化读两次、关闭回收；首轮集成fixture误把规范化schema组件顺序当作weight/bias顺序，已按rank识别并保留r1失败日志。这里明确CPU融合与Torch CUDA einsum reduction的逐位关系尚未验证，最终norm与完整GPU输出链仍需接通。

最终norm需要的BF16→F32精确promotion已加入通用cast合同；两个方向共用typed provider/physical preflight及受管执行，原F32→BF16保留RNE与NaN quieting规则。新方向按位展开，CPU实际Core执行覆盖全部65536种BF16编码（含subnormal、signed zero和NaN payload），每种两次、逐位正确，旧窄化回归同时通过（`evidence/float-cast-managed-cpu-r2.log`，2项）。运行时几何、错误dtype/extent、alias、foreign owner拒绝共4项通过（`evidence/float-cast-contracts-cpu-r1.log`）。CUDA接线与同域测试已写，尚未编译/运行，不由较早的HeadRMS冻结源验证。

输入和最终输出已形成新的模型组合切片。输入投影显式区分Text冷阶段（原BF16 context affine）与Media每步阶段（原F32音视频affine后转BF16），packing只接受已refined text，不能把context投影冒充refiner。最终输出在同一图中执行原BF16 RMS/逐阶段BF16调制、按模态选行、精确转F32与两个PDD affine；只对外返回音视频F32输出，组件之间不增加Host读回。归一化子组件现自然输出BF16，promotion由选行后的消费者负责。两个evaluation各重复执行的Reference结果符合独立原参数融合、时间路由及F64 affine误差界，所有来源只在初始化读取，关闭后保留输出有效，最终Host与设备owner归零。原来53项回归对应更早的分组件实现；新切片FastH3回归60项通过（`evidence/models-fast-h3-final-r4.log`）。该切片还没有CUDA数值证据，不代表完整视频forward。

组合抽象从原P-H3实现提取为crate内部stateless tensor program fragment，由原DiT与FastH3共同消费。只映射显式输入、内部value/node与逻辑weight ID，原物理component及numerical合同不改写；无权重transport保留None。接口只接受几何，不吞掉完整compile-options中的后端选择；有state、checkpoint输入角色、动态work或非默认schema specialization明确拒绝。旧P-H3 DiT相关22项回归通过（`evidence/ph3-dit-fragment-cpu-r2.log`）；Fragment映射/几何/来源边界4项通过（`evidence/tensor-fragment-cpu-r2.log`），fmt/diff检查通过。原两次新测试编译失败分别涉及未完成的测试模块及fixture缺字段，日志`models-final-output-r1/r2.log`保留，不计通过。

跨组件审查还发现“两个局部合法配置属于不同evaluation”的风险。现在PDD计划保留可重建的typed `FastH3PddSchedule`（fine-grid容量、partition与两个模态shift），原计划与反序列化组合配置复用同一数学入口。最终输出保存typed segment分区，从这一schedule重建完整时钟、head区间/系数，并检查归一化的精确时间值和行路由。合法但属于另一时钟的norm routing也拒绝；没有用同shape、自由组合配置或一份可伪造摘要替代语义一致性。

5090上的HeadRMS扩展已完成独立GPU资格：3/128/4096/4097/5120/6145/8192七种宽度，每形状三类输入各两次，符合独立RN32 pair-carry及两次BF16舍入oracle；有限输入相对F64 RMS最坏BF16 ULP为0（预算2）。超界8193/16384、关闭后保留输出及Host回收也覆盖，metadata边界7项通过。证据位于`head-rms-qwen-r1/`，冻结源码archive SHA为`60263e92601d8c032e1459894f41b474f23bd9473cbf38f06e6fd8555aa2d1c4`。真实Qwen NVFP4 checkpoint的1层与50层managed CUDA计划构建随后均取得exit0；例子泛型ID和canonical device ID的两次失败及单文件overlay单独保留。此阶段没有上传/执行完整encoder权重，计划容量不是显存实测或性能结论。

动态VSA已完成新的编译器观测源码闭包、原生archive、6包catalog/lock及5090限定GPU资格（`vsa-dynamic-r1-stable/validation-r3-r6.json`，最终容器真实exit0）。同一受管command/stream先从原F32全域pooled scores按不可变region预算产生索引，再执行原FP4 attention；不接受外部I32列表。动态`(B,H,Qtiles,Ktiles)=(2,3,4,5)/(1,2,4,6)/(1,1,129,130)`各两次变化scores，5,275,648个BF16结果逐位符合独立Q=K=0、dyadic V均匀attention oracle；旧Host路由同时回归，总计5,832,704个结果。实际覆盖多区域、dense prefix、半块、跨128lane索引、NaN/无穷和cutoff tie，第二轮三形状分别观察4/0/129个cutoff-tied区域；这验证声明的radix/index规则，不证明Torch tie选集相同。原始两份Host owner/padding、私有设备scratch、错误digest拒绝和正常退役经过现Core账户与prepared retention。首轮原生命名冲突、包exports排序失败、GPU整组因两份测试runtime能力指纹冲突的exit101均保留；后者只统一测试能力声明，容量保护未改。最终test源码archive SHA为`6b51229d296745fd2737184bd27a94bc6dc02ce89a64a0e3e596c0ed247d6970`，新锁SHA为`dc20fa626edac8390d099365a6a9f7d5dd90cd931b6e1f57a5976d262894f1e6`。仍未证明一般softmax/两级P舍入、真实pooled QK精度、Sage QKV量化、部分提交失败注入、完整FastH3或性能。最新修改尚未完成全workspace和所有受影响后端检查，旧集成快照不自动验证这些新增代码。

视频行搬运已有实际Stored消费者：从checked VSA layout派生唯一I32 row map，Tile/Untile/ExpandTileValues均通过原Host WeightBacking与Core初始化；分块的负索引填正零，有效行逐位保留全部BF16编码。Reference搬运回归9项与混合参考模型执行/容量2项通过（`evidence/row-transport-reference-r3.log`、`vsa-row-transport-model-r2.log`）；另有runtime几何1项。Masked tile mean直接消费真实Tile输出，以有界I32有效行数忽略padding，输出明确的F32[B,H,T,D]；合同1、Reference3、模型2项通过，含NaN padding不污染均值。这里分组件测试的Host桥只用于资格验证，生产组合图不依赖该桥。

矩阵[B*S,C]与序列[B,S,C]现通过两个通用leading-batch BF16 bitcopy合同连接，显式检查轴顺序、乘积、地址容量与NoAlias；不是伪装storage view。Reference真实Core执行覆盖全部65536种编码、非整齐维度、两个方向各两次及关闭后输出持有（`evidence/leading-batch-reference-r2.log`，1项），几何1项也通过。CUDA实现与同域fixture已写，尚未编译/运行；复制带来的带宽/峰值容量计入真实计划，未来优化需由通用view/liveness合同承担。

共享组合器现在由最终输出与注意力输入共同消费。实测发现Core禁止同一physical component在多个logical layout树重复引用，因此组合时只对完整声明及解释均相同的权重统一为一个logical weight/value；各节点显式引用该值，不放松Core所有权规则。7项组合回归通过，其中真实Stored receipt证明同一8B来源只读取/上传一次，两个节点在source释放后仍正确执行并回收（`evidence/stored-composition-cpu-r4.log`）。此前r1暴露上述重复layout错误，r2/r3是测试配置未采用JSON object/漏Clone与新测试操作名错误，失败日志均保留。

注意力前段现为22节点同图：packed hidden/cos/sin分块、QKV投影、QK/RoPE、三份未平滑tile mean，以及packed gate。投影通过typed Qkv/Gate分组复用来源；固定上游在tile行上量化QKV、在原packed行上独立量化gate，这两个入口被忠实保留。映射、counts和unit标量分别共享一个来源；F32 alpha派生及原NVFP4物理组件身份不变。另有四节点pooled attention图，显式产生QK→F32 scalar multiply后的scores，供动态选择与softmax/PV共用，未将scale折进GEMM alpha改变舍入。新增matmul保留B/H两轴、禁止隐式广播；CUDA声明pedantic F32，Reference/CUDA reduction身份独立，不宣称Torch逐位等价。

最新FastH3 CPU回归76/76通过（`evidence/models-fast-h3-attention-r2.log`），包含原最终输出、文本refiner、注意力输入的来源/图合同和独立oracle，以及实际Reference pooled attention；NVFP4注意力输入在CPU没有执行。新增F32合同2、Reference3项通过。完整22节点attention input、pooled attention、FMA输出等真实CUDA fixture已准备，等待冻结源后的5090资格。首次全组75通过/1失败源于pooled测试错误地给Ref2VA空参考，修复fixture后重跑；FMA输出的CPU图/源验证不冒称GPU前向。旧workspace全量检查早于这些修改，尚不能据此宣布当前PR已全量验证。

新一轮CPU回归87/87通过（`evidence/models-fast-h3-key-center-r1.log`，真实exit0），包含head-axis pack、K centering与视频VAE包/时序新增测试。K centering采用通用一输入两输出合同：完整padded key序列→BF16 centered与BF16 mean，八路F32归约后先舍入mean，再做BF16 subtraction；两输出均为Core预算内张量，CUDA同stream两次launch，没有额外分配器或Host读回。Core合同1项、Reference执行/拒绝3项、实际模型受管执行/容量/布局3项分别通过（`row-center-contract-r1.log`、`row-center-reference-r1.log`、`key-center-model-r2.log`）。共享fixture含独立F64 dyadic oracle、两轮变化输入、BF16均值tie、subnormal/signed-zero，以及能区分归约重排的抵消案例；关闭后账户仍计入两个Host输出的额度，释放后账户归零。该归约有独立数值身份，不宣称Torch reduction tree逐位等价，CUDA资格仍待下一冻结源。

本轮attention integration r4的默认CUDA全部descriptor回归、6个kernels测试目标、models CUDA lib和required CUDA CLI检查均真实exit0。此前r1测试导入错误、r2默认配置遗漏三项transport能力、r3缺WSL loader路径及RMS两分支实现指纹未包含实际kind的失败均保留；修复分别落在fixture导入、实际默认能力、测试容器库挂载及实际operation identity，没有关闭校验。首个22节点GPU fixture已于2026-11-13 06:55 UTC通过：B2、hidden128、heads2、D128、rot96，packed256→tiled1536；17个来源读取/80452B上传各一次，两次实际执行，独立exact-domain oracle、close和最终Host释放通过。原始receipt/log在 `attention-integration-r1-stable/evidence-r4/evidence/attention-input.state.json` / `attention-input.log`；二进制SHA为 `6a0abf25a02e85e467ecfa826c5e37b54d60eee612b0c7c409feb73752b3054d`。该结果不证明任意checkpoint数值、Torch归约等价或完整attention，更不是端到端性能。r4冻结源早于下述新增算子，不能据此声明当前工作树CUDA已全量通过。

BF16 channel bias采用精确矩阵/通道向量合同和原Core owner，RN32加法后一次BF16舍入；Core1、Reference物理拒绝1、真实受管fixture2通过（`decoder-bias-and-sage-core-cpu-r1.json`）。fixture覆盖所有BF16输入编码、tie、subnormal、signed zero、溢出/非有限、关闭后重新读取输出与最终账户归零。Sage QKV专用量化的Core2测试也通过：三个显式BF16输入、六种原U8布局输出、每原输入signed32索引边界；新CUDA provider已复用attention同一native module owner/prepare receipt，没有另造分配器，尚未CUDA编译/GPU验证。

prefix现在通过11节点Reference图先center K，再逐head FP8 roundtrip，最后按tile行序compact有效K/V；V保持原BF16，prefix/video query切片保留padding。模型2项与query map1项通过（`prefix-attention-models-r1.log`、`vsa-query-maps-r1.log`），包括B2/D128、独立oracle、两次执行和来源共享/回收；显式VendorFused profile复用原cuDNN容量与cache，CUDA资格待完成。视频decoder已有完整32节点block图、原NVFP4来源及严格层range；包/时序/图/来源/独立oracle共8项CPU测试通过（`video-vae-block-cpu-r2.log`），这是NVFP4 GPU前向之前的资格，尚未生成clip。

以上保留分阶段历史记录，最新验收状态以第 9 节开头为准。本文件仍约束完整目标（包括用户追加的视频连续批处理）；目标工具仍保留此前连接故障的 blocked 状态，当前 SSH 已恢复，用户已明确要求继续，工作按既有授权推进，尚未标记完成。

## 10. 视频验收完成后：Windows v0.19.1 外部测试反馈

状态：视频阶段已验收，接续执行本节。依据为 CJ Zwart / OpenSourcesAI 的 `OSAI-TR-20261113-FE Rev 1`、随附邮件及用户 2026-11-14 的追加授权。报告 19 页，SHA-256 为 `b3cbd30ee964f4453ed0c648abb2088e820570454d19e6b79846defc9d6cda5b`；已核对附件字节并按附录重算 54 个计时单元，共 2448/2448 完成，其中 v0.19.1 自身 816/816。此项是报告内部核查，尚未独立复现实机结果。

报告与重算数据保存在库外 `/Users/chejinxuan/ferrum-handoffs/20261114-v0191-windows-review/`，文件分别为 `Ferrum-v0.19.1-windows-retest-addendum.pdf` 与 `appendix-recalculation.json`。既有 5090 对比依据为 [PR #514](https://github.com/sizzlecar/ferrum-infer-rs/pull/514)。

### 10.1 待处理事项与交付

| 优先顺序 | 事项 | 当前证据 | 处理与完成依据 |
| --- | --- | --- | --- |
| 1 | 并发首 token 延迟差距 | RTX 4070 Ti、Qwen3.5-4B Q4_K_M：C8/C16 TTFT 中位数 95.3/178.8 ms，对照 llama.cpp 为 76.4/136.3 ms，仍高约 25%/31%。 | 按入口排队、prefill、首次采样与传输定位时间；沿共享执行与调度抽象优化。以同机同负载复测证明改善，并同时核查吞吐、TPOT、可见 ITL、错误和内存；未解决差距明确保留。 |
| 2 | 单请求吞吐差距 | C1 为 93.0 对 106.4 tok/s，即约 0.87×；首 token 29.2 对 29.6 ms 已接近。 | 优先剖析后续 decode 的 CPU 提交、同步、设备执行与内存成本，分别验证候选和完整 run/serve。目标缩小直至消除单请求差距，同时保留并发场景的吞吐与延迟资格。 |
| 3 | Windows 发布签名 | 对方确认 `ferrum.exe` 未签名；未测试安装程序。 | 核查 EXE/安装器发布链与现有签名凭据；接通正式代码签名、时间戳及发布后验签，分别保存两类产物证据。证书主体、资质或凭据缺失须列为明确外部依赖，先完成可审查的流程，不能把未签产物计为完成。 |
| 4 | 解释 4070 Ti 与 5090 的中位数变化 | 对方是跨 v0.18.5/v0.19.1 的短提示词测试；我们主表是同一二进制切换 profile 的 ShareGPT A/B，5090 的短提示词 C16 中位数也从 112.37 降至 86.77 ms。 | 依据工作负载、对照版本、预热、输入长度及首 token 采样改动解释差异；分清已测事实与硬件瓶颈假设。准备对报告第 8 节四问的逐项答复，未建立因果证据前不归因于 SM89 架构。 |
| 5 | temperature=0 的跨批次可复现性 | 单独与 C8 同时执行的文本一致率很低，双方引擎及新旧 Ferrum 都存在；该报告没有发现新版独有回归，也未验证答案质量。 | 明确数值 profile、不同批宽与 provider 的确定性边界；用固定请求与 seed 区分算术路径差异、采样差异和跨请求状态污染。实际隔离/正确性问题须修复；允许的数值差异据实说明并检查质量，不能仅用“预期行为”代替验证。 |
| 6 | 性能证据和对外表述 | 报告提供单元 P50 与平均单元最大值；缺少完整 TPOT P99、可见 ITL 和预声明 SLO。历史“约 2.5×”来自另一次测试，本次同场旧版 TTFT 比值约 1.87×/2.09×。 | 补齐可复用 Rust 客户端、固定 prompt/输入 token 长度及原始逐请求数据的测量口径；需要外部原始文件时明确所需材料。区分历史与同场数据、平均分位数与池化分位数，按实际硬件/模型/版本/负载限定结论；回信和发布说明使用同一证据。 |

### 10.2 实验与退出条件

- 首先复现对方工作负载：Windows SM89 发布包、同一 GGUF、llama.cpp b12069、固定容量 32、上下文与 batch 2048、FP16 KV、关闭 thinking/prefix cache，C1/C8/C16。原模型 revision 为 `e87f176479d0855a907a41277aca2f8ee7a09523`，GGUF SHA-256 为 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`。5090 或其他 SM89 卡的结果只作自身证据，不能替代 RTX 4070 Ti 的复测。
- 按仓库性能合同预先固定 TTFT/TPOT/可见 SSE 文本事件 ITL 的阈值、百分位与完成率；固定服务容量后再扫客户端并发。报告 P50/P99、输出吞吐、峰值显存/RSS、采样缺失、错误、样本量和重复数。原短提示词作为回归负载，正式服务结论同时使用固定 ShareGPT 样本；“吞吐基本不变”不能代替无回归或等价性证据。
- 共享执行修改覆盖 `ferrum run` 与 `ferrum serve`，保持声明的数值、容量、所有权和请求隔离合同；使用硬件无关正确性测试加受影响后端实测，在稳定候选执行仓库要求的检查。
- API key、自动选择 profile、下载/收据一致性和请求完成率是报告已通过项；保留相关回归，仅在代码触及时补充针对性检查。未测的 TOML 鉴权、benchmark 快照、安装程序等不能由该报告推定通过。
- 最终交付包括可复查的性能前后对比、可复现性结论、Windows 签名产物/验签结果与四问答复草稿。性能未改善、关键验证缺失或签名仍有外部依赖时，分别保留未完成项。用户当前授权包含处理与准备答复，实际发送邮件需另有明确发送指令。
