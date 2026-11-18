# Ferrum v0.20.1 产品修复、验收与发布

2026-11-18 16:30（北京时间），Ferrum v0.20.1 已从 `cc098c3a0` 正式发布。[GitHub Release](https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.20.1)
[产品修复](https://github.com/sizzlecar/ferrum-infer-rs/pull/523) · [版本准备](https://github.com/sizzlecar/ferrum-infer-rs/pull/524) · [发行任务](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/44232521500)

## 冻结发行包验收

- Linux SM120 包在 RTX 5090 完成 4 项安装与启动检查，3 个文本模型通过全部 35 个用例。另在无 cuDNN、FFmpeg 和 CUDA SDK 的文本环境中，使用安装器装入同一二进制，Qwen3.5-0.8B 的 6 项 CLI 和 HTTP 用例均通过。
- 同一 SM120 二进制的 P-H3 鉴权服务连续完成生成、编辑、再生成，两次生成的像素相同。生成和编辑遇到缺失或错误的密钥均返回 401。FastH3 使用三字段预设、系统 FFmpeg 与 PNG 参考图，生成 832×480、24 fps、124 帧的 H.264 视频及 32 kHz 双声道 AAC。进程重启后，下载的 MP4 与重启前逐字节一致。
- SM120 归档为 90,312,338 字节，v0.20.0 为 90,202,645 字节，本次增加 109,693 字节（约 0.12%）。两者均不含模型权重。选择图像或视频所需的系统库与媒体工具另计。[系统 GPU 依赖与安装成本](image-generation.md#p-h3-setup) · [视频编码与安装成本](video-generation.md#setup)
- Apple Silicon 使用冻结的 Metal 二进制完成 4 项安装与启动检查。Llama 3.1 8B GGUF 通过 14 项，Qwen3.5-0.8B safetensors 通过 6 项，Qwen3.5-4B GGUF 通过 18 项，合计 38 项全部通过。[Metal 原包模型验收](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/44232521500/job/134583871861)
- RTX 4050（SM89）使用冻结 Linux CUDA 包完成 Llama 3.2 1B、Qwen3.5 0.8B、Qwen3.5 4B 三个文本配置的运行与服务验收，FP8 独立数值对照通过。Linux CPU 包通过 Qwen3.5 0.8B 的 GGUF 和 safetensors 两种配置验收；两个包各通过 4 项启动检查。原 SM89 任务下载超时，其后排队的 CPU 任务被跳过。两项手工复验均使用冻结候选的相应二进制，归档 SHA-256 为 `bc78888b76d37f45cc4da3cf2d51f2e6c6daf6b3460658e12c7912e8c2bfd2ca`。
- Windows x86_64 CPU 和 CUDA SM89 两种发行包都通过了打包与程序启动验证。CPU 包来自原发行任务，CUDA 包来自同一候选标签的[独立 Windows 构建](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/44244618049)。
- 正式发布包含 47 个文件。公开 Mac Metal 与 Linux SM120 归档的 SHA-256 均与实机验收使用的归档一致。Homebrew 的 macOS 配方已更新到同一公开 Metal 归档。[发布记录](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/44264837074) · [Homebrew 更新](https://github.com/sizzlecar/homebrew-ferrum/commit/5dc6dafe113304bece13d3ddbf6b5b915b5aa2e0)
- Apple Silicon 在全新用户目录中从官网零参数安装，自动选择 Metal。官网返回 HTTP 200 并传输完整的 20,899,079 字节；安装后的二进制与冻结 Metal 包一致，`--version`、`--help`、`run --help`、`serve --help` 均成功退出。
- RTX 5090 在全新用户目录使用官网安装器自动选择 SM120，完成 4 项安装与启动检查。随后使用已安装程序运行 Qwen3.5-0.8B 的 6 项 CLI 和 HTTP 检查，服务报告 CUDA 计算能力 12.0。公开验收归档 SHA-256 为 `79b24081a763c95c5170856c41f62964e4feae506b7c23b00d4d5461ce80a509`。[发布后 SM120 实机验收](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/44269332887/job/134682019352)
- crates.io 的 22 个可发布 crate 均已发布 0.20.1，均未撤回，注册表的校验值与正式发行记录全部一致。
- 官网全新安装与 Homebrew 升级分别验收。首轮公开工作流中，Mac Homebrew 在更新上游索引时下载失败，未执行 Ferrum 升级。网络恢复后，在同一台 Mac 上从公开 tap 将 0.20.0 升级为 0.20.1。安装的程序与官网 Metal 包 SHA-256 一致，4 项启动检查均通过。托管 macOS 安装未执行，另有 Windows 和 Linux Homebrew 任务失败。复验时，macOS、Linux 和 Windows 的官网安装、macOS 与 Linux 的 Homebrew 安装、Cargo 安装及 RTX 5090 实机验收全部通过。[首轮记录](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/44269332887) · [最终公开安装工作流](https://github.com/sizzlecar/ferrum-infer-rs/actions/runs/44283769582)

首次在 RTX 5090 部署 P-H3 仍需填写 `--image-config` 模型路径与资源配置。FastH3 需要 `--video-config`、检查点和输出目录。仅传模型目录即可启动两种服务的工作由 [#528](https://github.com/sizzlecar/ferrum-infer-rs/issues/528) 跟踪。

## 交付目标

图像用户能通过 CLI、局部配置和网页完成鉴权及调用；视频服务能持续接单、
有界保留结果并跨重启查询；FastH3 的正常安装不再要求用户查找 cuDNN、
FFmpeg 和 ffprobe 的绝对路径。通过仓库正式发布流程交付实际验收的产物。

RTX 5090 的 P-H3 和 FastH3 复用系统 CUDA 12.8 与 CUDA 12 版 cuDNN 9.20。视频输出使用系统 FFmpeg，视频参考输入还需要 ffprobe。纯文本入口沿用对应后端的依赖。模型及编码器变更的正确性与性能分别验收。

用户明确的产品边界：

- 只用文本时，不安装、下载、检查 cuDNN 或媒体组件；CPU/Metal/CUDA 各自只需要对应后端的必要依赖。
- 图像、视频能力须由用户明确选择；不能因为支持媒体生成就扩大所有人的安装成本。
- 视频也不能以包体积翻倍数倍为代价。分别核算总下载量、展开占用和升级增量；拆包、首次下载或缓存不能用来掩盖成本。
- 依赖方案未定前只做审查与测量，不继续实现捆绑、下载器或新媒体架构。现有试作已归档并撤出工作分支。

依赖验收从用户选择的模型和输入输出格式开始：

| 用户操作 | 实际需要的能力 | 验收方式 |
| --- | --- | --- |
| 只跑文本 | 所选 CPU、Metal 或 CUDA 文本后端 | 在无 cuDNN 和媒体工具的环境运行 `run` 和 `serve`；对比 0.20.0 的下载、磁盘占用和 NVIDIA 驱动要求 |
| P-H3 在 5090 上出 PNG | 当前 GPU 算子使用 cuDNN；PNG 编码在 Rust 中完成 | 未安装 FFmpeg 时完成真实出图；记录图像新增的库，随后启用 FastH3 时核对哪些能复用 |
| FastH3 用提示词和 PNG 参考图生成 MP4 | 已发布的 Ref2VA 至少需要一份视觉参考。PNG 输入在 Rust 内解码；模型使用 cuDNN 做卷积和注意力。生成的 RGB 帧和 PCM 音频经 H.264/AAC 编码后写入 MP4 | 无需填写库或工具路径，在缺少 ffprobe 时用 PNG 生成视频并完整解码，检查帧数、音轨与取消回收 |
| FastH3 使用参考视频 | 额外需要探测、解码输入容器和音视频轨道 | 实际输入多种已支持的编码并核对旋转、时序和音频；没有输入解码能力时在加载或排队前给出可操作的提示 |

5090 的发布候选沿用 SM120 的 CUDA 12.8 构建，P-H3 和 FastH3 共用 cuDNN
9.20.0.48 的 CUDA 12 构建。该组合已开始验证。原有图像服务同时加载两个 CUDA
主版本，原有图像和视频示例配置显式列出了六个 CUDA 13 库。验证时使用隔离的
CUDA 12 环境，分别保存 P-H3 真实 PNG 和 FastH3 真实 MP4、同一请求的运行库加载
记录及产物解码结果。保留原有精度和显存预算，记录 cuDNN 的实际下载与展开体积。
NVIDIA 列出的 Blackwell 支持条件包括 CUDA 12.8 及以上；具体模型的验收另行记录。
[cuDNN 9.20.0 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.20.0/reference/support-matrix.html)

媒体先验证复用已安装且能力匹配的编解码器，保留已有机器的显式路径覆盖。部署方
只选 MP4 输出时需要输出编码；输入探测与解码属于参考视频能力。若干净机器缺少
所选能力，须给出明确的安装途径并计入完整安装成本。最终以只有驱动的机器能实际
交付原有 MP4 为准；内部生成原始帧只是其中一个步骤。

## 待办和验收标准

| 项目 | 必须达到的行为 | 验收证据 |
| --- | --- | --- |
| 图像 CLI 鉴权 | JSON 和模型目录入口均接受 `serve --api-key`；非法 key 在模型加载前拒绝；优先级、脱敏正确 | 真实 CLI 解析/执行入口 Rust 回归，正确/错误/缺失密钥 HTTP 调用 |
| 局部配置 | 仅 `[server] api_key=...`、空表、仅覆盖 host/port 均继承其余默认值；错误类型仍拒绝 | 实际 TOML load/save 测试，最小配置执行 `doctor` |
| 图像网页 | 可输入、更新、清除 key；配置/生成/编辑请求带 Bearer；明确提示 401；旧响应不覆盖新连接；key 不写 URL/静态资源/持久存储 | 路由测试、JS 语法、实际浏览器交互及真实图像生成/编辑抽样 |
| 健康检查 | `unhealthy` 返回 503；正常可用返回 200，保留诊断正文和无鉴权探针 | 正常、关闭、运行期故障和带鉴权服务的 HTTP 测试 |
| 上传时限 | typed idle/总时限默认 30/300 秒；超时 408；释放原始额度/内存；不限制后续模型生成 | 可控时钟的停传、慢传、空块、取消及阻塞校验所有权测试 |
| 持续视频服务 | 终态不占活动额度；终态有条数、字节、TTL 上限；TTL 从完成起算；仅回收安全终态 | 小容量连续完成仍能接单、并发预留、TTL、取消、不安全 owner 边界回归 |
| 视频持久化 | 重启可查已完成结果，未完成明确中断；落盘后公布完成；删除失败不报成功；下载保持文件/配额所有权 | 重启、写入中断、发布/删除失败、独占目录、下载并发删除；未知文件保留 |
| FastH3 依赖准备 | 明确最小 GPU/媒体能力及交付方式；自动定位且保留显式覆盖；无需手配路径；文本用户不承担视频专属依赖；加载大权重前尽早诊断 | 实际下载与解压增量、运行时依赖闭包、升级/损坏包和离线测试；仅有 NVIDIA 驱动的干净机器实际启动；首次与重复启动成本 |
| 模型与预算配置 | `run/serve` 共用解析；明确模型内容、已验证预设与有效默认；普通用户无需猜内部预算 | 最小文档命令分别完成两入口；记录实际配置；超出能力/预算明确失败 |
| 正式发布 | 候选版本一致，必需门禁通过，验收与发布资产一致；公开安装可用 | 完整发布任务计划、模型证据、公开安装及 release delivery 最终 `complete` |

视频保留初始默认：最多 128 条、8 GiB、24 小时；容量压力可提前回收最旧安全终态。
活动任务额度沿用配置，编码临时 spool 默认 2 GiB，二者与保留配额分别计量。
下载者和未完成原生工作仍占用真实容量。历史匿名 spool 不因新机制上线而被冒领或擅自删除。

## 执行顺序

1. 完成行为修复、Rust 反例与独立审查；确定依赖交付取舍后再接入发行工作流。
2. 在稳定节点执行工作区 `fmt --check`、`check --all-targets`、
   `test --all-targets`、`clippy --all-targets -- -A warnings`，均带 `--workspace`。
3. 执行适用的 macOS Metal 与配置完整的 CUDA 后端编译检查；区分编译和真实运行。
4. 实际图像/视频验证：尺寸、帧数、音频、完整解码、取消、鉴权和重启恢复。
   存储签名 fixture 与 HTTP 200 不能替代真实推理和媒体有效性。
5. 使用候选发行包在干净 Linux SM120 环境测试自动依赖准备，不继承开发环境的
   CUDA/cuDNN/FFmpeg 路径。驱动仍为系统前置；不泛化为 Windows/macOS FastH3 支持。
6. 按上次正式版到候选的联合 diff 生成发布回归计划，完成必需项，执行
   [正式发布流程](release-candidate-automation.md)，核验公开发行包、包注册表和安装器。

失败、未执行、不确定、缺证据分别记录，不能用测试数或容忍失败的 CI 绿灯抵消。
日志、模型、构建缓存和原始结果保存在仓库外。若触及性能策略，预先声明工作负载、
TTFT/TPOT/ITL SLO 和退化边界，以同硬件测量验证，不由编译或出片成功推断收益。

## 单独跟踪

图像 API 缺省 seed、混合 GPU 的安装目标选择、文本默认容量档位、历史特定 GPU
OOM/性能报告继续单独跟踪；本目标通过不会自动关闭它们。依赖交付触及相关行为时补相应回归。
Qwen3 0.6B CPU 的多轮语义测试也单独跟踪：当前修复与已发布 0.20.0 都在算术轮
重复上一轮的 OK。相同权重、官方模板和消息在 PyTorch/Transformers 参考实现下也产生
相同回答，因此该用例不能单独证明 Ferrum 引擎缺陷；保留失败记录，不把参考失败改记为语义通过。

## 开发阶段的验证记录

- 正式 0.20.0 已复现图像 `--api-key` 拒绝和最小 `[server]` 配置失败。
- 配置、鉴权、健康、上传和视频生命周期修复已进入工作分支，
  [草稿 PR #523](https://github.com/sizzlecar/ferrum-infer-rs/pull/523) 已创建；
  产品源码检查点为 `5be6ae817`。Windows 的条件编译参数告警已修正，其 CI 检查通过。
- server 单元/路由测试阶段记录：497 通过、0 失败、1 个显式忽略。
  后续代码变化需核实证据适用范围。
- 两项显式运行的真实 MP4 编码/保留/重启测试通过；这不是 GPU 模型推理验收。
- Chrome 157 真实浏览器鉴权测试通过：覆盖无/错误/正确 key、生成 JSON、
  编辑 multipart、更新/清除和刷新不持久化；业务层使用测试模型，真实图像推理仍待验收。
- 依赖解析、运行库打包、安装器及新媒体架构草案试作已归档至仓库外，未提交或发布。
  此前试作的测试结果不能作为当前依赖交付已完成的证据。
- 撤出试作后，本机 `fmt --check`、工作区 `check --all-targets`、
  `test --all-targets`、`clippy --all-targets -- -A warnings` 及 Metal 全目标编译通过。
  默认 feature 的完整测试结果为 8584 通过、0 失败、97 显式忽略；忽略项不算通过，
  上述编译和单元/集成测试不代替真实 CUDA/Metal 模型验收。
  新增 segmented 小用例后上述工作区检查再次通过；该用例另在 5090 显式执行并通过，
  不计入默认 feature 的通过数。
- 5090 已通过 Windows SSH 桥恢复连接；本机上传超时后，改为远端从公开 GitHub
  浅克隆并核对 `5be6ae817`，复用既有 CUDA 12.8.1 和 native 算子锁。
  必需的 `cuda,vllm-moe-marlin,vllm-paged-attn-v2` 编译通过；额外的
  `cuda-fast-h3,vllm-moe-marlin,vllm-paged-attn-v2` 编译也通过，后者有 6 条告警。
  未启用 GPU 或停止在线 P-H3 服务；这不是完整发行 feature 组合或真实模型验收。
- Qwen3 0.6B CPU 在移除额外动态库路径、PATH 仅含系统工具的环境中完成模型加载、
  HTTP 启动和关闭，但 `run`/`serve` 的 basic 多轮语义检查均失败。已用相同模型和
  参数在公开 0.20.0 与候选 `release-tools` 优化二进制上复现，不据不同构建比较性能。
  PyTorch 2.8.0 / Transformers 4.57.6 使用同一快照、CPU F32、官方模板、关闭思考、
  显式贪心、32 token 上限和仅作用于生成 token 的 1.1 重复惩罚，三轮同样输出
  `OK.` / `OK.` / `cobalt-731`；固定重放 Ferrum 第二轮消息也输出 `OK.`。
  本地 tokenizer 配置与官方同提交文件逐字节相同。该对照不是性能或所有输入的等价证明。
- Qwen3.5 0.8B 的 dev 构建超时记录保留；同一源码的 `release-tools` 优化构建在原有
  120 秒启动/请求、240 秒 run 预算内通过 basic：`run` 多轮、同步/SSE 算术与回忆、
  正常关闭均通过。没有放宽断言或超时，也不据单次功能测试宣称性能。
- 5090 上现有 F32 Conv3d、BF16 causal GQA，以及独立 segmented provider 的小张量
  oracle 测试已实际通过。新增 D72、不同长度分段用例使用超过 F16 有限范围但 BF16
  可精确表示的数值，检查有限输出、段内均值和段间隔离。测试前后在线 P-H3 服务健康，
  未运行第二个大型模型。这些是现有算子基线，不是去除 cuDNN 或完整 FastH3 的验收。
- 5090 隔离环境换用 CUDA 12.8 与 `nvidia-cudnn-cu12==9.20.0.48`。官方 Linux x86_64
  wheel 下载 657,910,570 字节，SHA-256 已校验，8 个动态库展开共 996,737,320 字节。
  复用原有 CUDA 12.8 运行库、显式预加载列表留空时，F32 Conv3d、BF16 causal GQA
  和独立 segmented D72 三项数值测试通过。该体积描述本次测试使用的 cuDNN wheel。
- 同一隔离环境使用公开 v0.20.0 SM120 二进制完成 P-H3 真实出图。1024×1024、
  seed 42 的雪地狐狸提示词输出解码为 RGBA8，像素 SHA-256 与此前相同请求的
  0.19.1 线上结果一致。动态加载日志仅见 CUDA 12 与对应的 cuDNN。
- 同一二进制重放 v0.20.0 的 FastH3 发布请求，使用原 PNG 参考图、seed 1101、
  832×480、124 帧、24 fps 和原编码参数。生成的 MP4 与发布参考逐字节一致
  （SHA-256 `ef9680e507015ef0439bf78ea7c2b462386a17ef88c60fc1501d59b02cdbbdb0`）。
  H.264 视频与 32 kHz 双声道 AAC 均完整解码成功。该生成进程加载五个 CUDA 12 库
  和七个 cuDNN 库，未加载 CUDA 13。以上实测仍使用显式 cuDNN 主库路径。
- 相同 v0.20.0 SM120 程序还在 CUDA 12.8 文本环境中运行了 Qwen3.5-0.8B。
  容器未挂载 cuDNN，系统中未发现 cuDNN、ffmpeg 或 ffprobe。基本语义回归覆盖
  CLI 多轮和 HTTP 同步、流式请求，均通过并正常释放进程。
- 参考视频在探测和解码阶段均应用解码器像素上限。旧实现实际接受了超限帧，
  修复后 7 项真实 FFmpeg 测试通过，包括探测拒绝、非对齐合法尺寸和取消回收。
  工作区默认测试 8584 通过，99 显式忽略，fmt、check、clippy 和 Metal 编译通过。
- 已从 `v0.20.0` 到 `5be6ae817` 生成完整发布回归计划：152 项必需行为、8 个选定
  profile、0 个计划覆盖缺口。74 项扩展行为、10 项 CPU 扩展语义和 18 项性能测量
  被明确列为未执行或推迟；计划本身不是通过证据，也不代替本页的媒体产品验收。
- Linux SM120 统一 CUDA 13 的体积估算中，现有 CUDA 12 三库合计
  828.64 MiB，是候选能完全移除 CUDA 12 后的展开文件减少量。保留完整 cuDNN 八库
  对比时，统一 CUDA 12 的模型化净减少仅 100.68 MiB；其对应 cuDNN 更大。
  这些不是现有发行 tar 的缩小量，候选实际动态闭包仍需验证。CUDA 13 还涉及
  原生产物重建，以及 SM120 文本用户最低驱动从 R570 到 R580 的兼容变化。
- 媒体库裁剪仍在测量阶段；删除 FFmpeg 命令行前端本身不能大幅降低完整依赖大小。
  候选需保留现有 H.264/AAC/fMP4 输出与输入解码能力，不能用仅 H.264 的实现替代
  现有 FFV1/PCM 测试或无说明地移除 AV1/VP9 等参考输入。
- 产品代码 `e6a6d7d0c` 在 5090 上编译通过。最小 Ubuntu 容器内通过系统加载器发现
  CUDA 12 版 cuDNN，没有 CUDA SDK、媒体工具或 `LD_LIBRARY_PATH` 时，P-H3 CLI
  和已鉴权 HTTP 生成的 PNG 逐字节一致；真实编辑成功，缺失和错误密钥均返回 401。
- 在同一产品代码上，FastH3 三字段预设配合系统 FFmpeg 6.1.1 和 PNG 参考图，在没有
  可执行 ffprobe 时生成可完整解码的 832×480、24 fps、124 帧 H.264 与 32 kHz
  双声道 AAC。与旧编码器的对照逐帧解码画面一致，音频采样与容器字节存在差异。
- 视频启动增加单帧和立体声音频试编码，沿用产品输出的编码参数和进程回收流程，
  在加载模型前诊断缺少 libx264、AAC 或 MP4 输出支持。已在本机用真实 FFmpeg
  和可执行但无法编码 MP4 的程序分别验证成功与失败；最终候选包须重验。
- 托管 macOS 测试发现图像单输出名额的顺序请求偶发 429。通过控制线程交付时序，
  已在真实 HTTP 路径复现：第一张图的响应和数据均已释放，工作线程仍短暂持有
  名额，第二次请求因此返回“output session limit reached”。回归测试用客户端
  收到结果的通知检查该名额，另有测试覆盖仍持有响应数据时应继续限制接单。
