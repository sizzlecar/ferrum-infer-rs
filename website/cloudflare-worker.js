import shellInstaller from "../scripts/install.sh";
import powerShellInstaller from "../scripts/install.ps1";

const SITE_ORIGIN = "https://ferrum.pandaailabs.com";
const REPOSITORY = "https://github.com/sizzlecar/ferrum-infer-rs";
const RELEASE_VERSION = "0.15.0";
const RELEASE_URL = `${REPOSITORY}/releases/tag/v${RELEASE_VERSION}`;
const IMAGE_PERFORMANCE_URL = `${REPOSITORY}/releases/tag/v0.15.0#qwen-image-performance`;

const css = String.raw`
:root {
  color-scheme: dark;
  --bg: #0b0d0f;
  --panel: #12161a;
  --panel-2: #181d22;
  --text: #f5f3ee;
  --muted: #a9b0b7;
  --line: #2b3238;
  --rust: #f0682f;
  --rust-light: #ff9b69;
  --green: #81d8a7;
  --max: 1120px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 12% 4%, rgba(240,104,47,.16), transparent 32rem),
    radial-gradient(circle at 90% 24%, rgba(129,216,167,.09), transparent 30rem),
    var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.6;
}
a { color: inherit; }
.wrap { width: min(calc(100% - 40px), var(--max)); margin-inline: auto; }
nav {
  position: sticky;
  top: 0;
  z-index: 10;
  border-bottom: 1px solid rgba(255,255,255,.08);
  background: rgba(11,13,15,.8);
  backdrop-filter: blur(18px);
}
.nav-inner { min-height: 68px; display: flex; align-items: center; justify-content: space-between; gap: 24px; }
.brand { display: inline-flex; align-items: center; gap: 11px; text-decoration: none; }
.brand span { font-size: .95rem; font-weight: 800; letter-spacing: .16em; }
.mark { width: 34px; height: 34px; flex: 0 0 auto; }
.nav-links { display: flex; align-items: center; gap: 22px; color: var(--muted); font-size: .94rem; }
.nav-links a { text-decoration: none; }
.nav-links a:hover { color: var(--text); }
.lang { border: 1px solid var(--line); border-radius: 999px; padding: 5px 11px; }
header { padding: 92px 0 68px; }
.eyebrow { color: var(--rust-light); font-size: .8rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
h1 { max-width: 900px; margin: 14px 0 22px; font-size: clamp(3.1rem, 8vw, 6.9rem); line-height: .96; letter-spacing: -.065em; }
.hero-copy { max-width: 760px; color: #c8cdd1; font-size: clamp(1.12rem, 2vw, 1.35rem); }
.objective { max-width: 760px; margin-top: 28px; padding: 20px; border: 1px solid var(--line); border-radius: 16px; background: var(--panel); }
.objective p { margin: 0 0 12px; }
.objective math { margin: 20px 0 4px; font-size: clamp(.95rem, 2vw, 1.15rem); }
.objective-definitions { margin: 20px 0 0; padding-left: 18px; color: var(--muted); font-size: .88rem; }
.objective-definitions li + li { margin-top: 8px; }
p code { overflow-wrap: anywhere; }
.actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 34px; }
.button { display: inline-flex; align-items: center; justify-content: center; min-height: 46px; padding: 0 19px; border-radius: 10px; border: 1px solid var(--line); background: var(--panel); text-decoration: none; font-weight: 700; }
.button.primary { border-color: var(--rust); background: var(--rust); color: #180b06; }
.button:hover { transform: translateY(-1px); border-color: #59636c; }
.button.primary:hover { border-color: var(--rust-light); background: var(--rust-light); }
section { padding: 68px 0; scroll-margin-top: 68px; }
.section-head { max-width: 760px; margin-bottom: 30px; }
h2 { margin: 0 0 10px; font-size: clamp(2rem, 5vw, 3.25rem); line-height: 1.05; letter-spacing: -.045em; }
.section-head p, .muted { color: var(--muted); }
.code-shell { overflow: hidden; border: 1px solid #303840; border-radius: 16px; background: #080a0b; box-shadow: 0 26px 80px rgba(0,0,0,.28); }
.code-top { display: flex; align-items: center; justify-content: space-between; min-height: 48px; padding: 0 18px; border-bottom: 1px solid #20262b; color: var(--muted); font-size: .82rem; }
.install-commands { display: grid; gap: 16px; }
.copy-button { margin: 8px 0 8px 12px; padding: 7px 12px; border: 1px solid var(--line); border-radius: 7px; background: var(--panel-2); color: var(--text); font: inherit; cursor: pointer; flex-shrink: 0; }
.copy-button:hover { border-color: var(--rust); }
.copy-button:focus-visible { outline: 2px solid var(--rust-light); outline-offset: 3px; }
.copy-status { color: var(--muted); font-size: .9rem; }
.copy-status:empty { display: none; }
pre { margin: 0; padding: 24px; overflow-x: auto; color: #e6e3dc; font: 500 .92rem/1.75 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
.code-shell pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.command { color: #ffad80; }
.note { margin-top: 18px; padding: 15px 18px; border-left: 3px solid var(--rust); background: rgba(240,104,47,.07); color: #cdd1d4; }
details { margin-top: 24px; }
summary { cursor: pointer; color: var(--rust-light); font-weight: 700; }
details .section-head { margin-top: 24px; }
.support-list { margin: 0; }
.support-list > div { display: grid; grid-template-columns: 180px 1fr; gap: 18px; padding: 18px 0; border-bottom: 1px solid var(--line); }
.support-list dt { font-weight: 700; }
.support-list dd { margin: 0; color: var(--muted); }
.benchmark-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }
.table-wrap { min-width: 0; max-width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
caption { padding: 18px 14px; text-align: left; font-weight: 700; }
caption span { display: block; color: var(--muted); font-size: .8rem; font-weight: 400; }
th, td { padding: 11px 10px; border-top: 1px solid var(--line); text-align: right; }
thead th { color: var(--muted); font-size: .82rem; font-weight: 500; }
th:first-child { text-align: left; }
tbody th { font-weight: 500; }
tbody td:last-child { color: var(--green); }
footer { margin-top: 52px; padding: 34px 0 48px; border-top: 1px solid var(--line); color: var(--muted); font-size: .9rem; }
.footer-inner { display: flex; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
@media (max-width: 820px) {
  header { padding-top: 72px; }
  .benchmark-grid { grid-template-columns: 1fr; }
  .support-list > div { grid-template-columns: 1fr; gap: 6px; }
  .nav-links a:not(.lang) { display: none; }
}
@media (max-width: 400px) {
  .wrap { width: calc(100% - 28px); }
  .objective { padding: 16px 12px; }
  .code-top { padding-inline: 12px; }
  .code-shell pre { padding: 18px 12px; }
  th, td { padding: 10px 7px; font-size: .83rem; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .button:hover { transform: none; }
}
`;

const logoShapes = `<rect x="4" y="4" width="56" height="56" rx="13" fill="#0b0d0f"/><path d="M16 14H50L47 24H29V31H38L35 41H29V52H16Z" fill="#f0682f"/><path d="M38 31H50L47 41H35Z" fill="#81d8a7"/>`;
const mark = `<svg class="mark" viewBox="0 0 64 64" aria-hidden="true">${logoShapes}</svg>`;
const logoSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Ferrum">${logoShapes}</svg>`;

const throughput = {
  gguf: [
    [4, "179.25", "170.13", "105.36%"],
    [8, "356.42", "291.60", "122.23%"],
    [16, "535.51", "377.41", "141.89%"],
    [32, "666.24", "435.57", "152.96%"],
  ],
  ct: [
    [4, "210.32", "233.45", "90.09%"],
    [8, "374.32", "447.06", "83.73%"],
    [16, "654.04", "733.86", "89.12%"],
    [32, "836.79", "1035.09", "80.84%"],
  ],
};

const imageThroughput = [
  ["generation", "1024²", 8, "0.5193", "0.4428", "1.173×"],
  ["editing", "1024²", 1, "0.2943", "0.2631", "1.119×"],
  ["generation", "2048²", 2, "0.0799", "0.0772", "1.035×"],
  ["editing", "2048²", 2, "0.0686", "0.0685", "≈1.00×*"],
];

const pages = {
  en: {
    lang: "en",
    path: "/",
    alternatePath: "/zh/",
    alternateLabel: "中文",
    title: "Ferrum — Text and image inference in Rust",
    description: "Local text inference, Qwen-Image generation and editing, and OpenAI-compatible APIs in one Rust binary.",
    nav: [["Quick start", "#quick-start"], ["Images", "#images"], ["Supported scope", "#support"], ["Throughput", "#performance"], ["Project", "#project"]],
    eyebrow: `v${RELEASE_VERSION} · Released`,
    h1: "Text and images.<br>One Rust binary.",
    lead: "Run LLMs and Qwen-Image locally, with CLI and OpenAI-compatible APIs.",
    goal: "Goal: maximize output throughput while meeting TTFT, TPOT and ITL SLOs.",
    goalContext: "Fix hardware, model and output policy; choose execution policy π from eligible policies Π under predeclared SLOs.",
    goalMathLabel: "Maximize valid output tokens from requests completed within fixed window T, divided by T, over eligible execution policies pi; require each latency percentile limit and completion fraction at least rho.",
    goalDefinitions: [
      "W fixes the request set and arrival times; T > 0 fixes the measurement window and common cutoff. N_out counts only valid output tokens from requests successfully completed within that window.",
      'L_k is the latency sample distribution, Q_p_k its chosen quantile, and S_k its upper limit. TTFT includes queueing; missing latency does not count as meeting its SLO. <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/performance-popular-models-p0.zh.md">Sampling rules</a>.',
      "C is the fraction of W successfully completed by the cutoff; its denominator includes rejected, failed, timed-out and unfinished requests. Declare p_k, S_k and 0 < ρ ≤ 1 before measuring.",
    ],
    primary: `Install ${RELEASE_VERSION}`,
    secondary: "View on GitHub",
    quickTitle: "Install and run",
    installTitle: "Installation commands",
    copyLabel: "Copy",
    copiedLabel: "Copied!",
    copyFailedLabel: "Copy was blocked. Select the command and copy it manually.",
    installNote: "Installers verify checksums and configure PATH. Open a new terminal on macOS/Linux; PowerShell is ready immediately. Repeat the command to upgrade; models and configuration are preserved.",
    runtimeNote: 'macOS uses Metal. Linux/WSL selects sm89 (CUDA 12.4) or sm120 (CUDA 12.8) when all visible GPUs report the matching compute capability. Compatible NVIDIA driver and CUDA/NCCL runtimes are required. Windows and Homebrew CUDA target sm89; Windows bundles CUDA/VC runtimes. Unsupported or mixed GPUs select CPU automatically, as does a CUDA load failure. Explicit CUDA fails when unavailable; missing matching assets also fail. <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/release-regression.md#prepare-the-candidate">Runtime requirements</a>.',
    runLabel: "Terminal chat",
    serveLabel: "API server",
    runNote: "Weights download on first use. Omit --disable-thinking to preserve the model template's default reasoning behavior.",
    apiNote: 'API URL: <code>http://127.0.0.1:8000/v1</code>, model <code>ferrum</code>. <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/openai-api-compatibility.md">Streaming, tools and structured output</a>.',
    defaultsNote: 'Defaults adapt execution and capacity to capabilities, hardware and memory; explicit settings prevail. Qualified SM120 GGUF paths can use 8-bit activation quantization and integer dot products. <code>--numerical-profile qwen3_5.f32-master</code> retains the original mixed-precision reference. Batch layouts can change output. <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/numerical-execution.zh.md">Numerical profiles</a> · <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/cache-product.md">Caching</a>.',
    imageTitle: "Generate and edit images",
    imageLead: 'Use a complete local <a href="https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo">Qwen-Image-2.1-Turbo</a> directory. Validated on RTX 5090, Linux CUDA 12.8 and cuDNN 9.',
    imageMetalNote: 'Metal image support is being validated for the next release. M4 Mac mini, 16 GiB unified memory: 1024×1024 generation and editing completed with a Q4_K_M GGUF DiT, FP8 text encoder and BF16 vision/VAE, using an explicit <code>--image-config</code> for the backend, model assets and memory limits. This configuration has not completed 2K generation; throughput qualification remains in progress.',
    imageMetalSetup: 'Metal: <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/README.md#image-generation-and-editing">configuration and commands</a>.',
    imageRunLabel: "CUDA: text to image",
    imageEditLabel: "CUDA: edit one PNG",
    imageServeLabel: "CUDA: image API server",
    imageApi: 'HTTP: <code>POST /v1/images/generations</code> (JSON), <code>POST /v1/images/edits</code> (multipart). One 8-bit RGB/RGBA PNG input; PNG output as <code>b64_json</code>. Editing may redraw background/details.',
    imageConfigNote: 'The local-directory preset supports up to 1024×1024. 2K and the measured resident-service configurations use <code>--image-config</code>.',
    imagePerformanceTitle: "Qwen-Image vs vLLM-Omni",
    imagePerformanceLead: "RTX 5090 · 8 steps · FP8 text encoder/DiT, BF16 VAE. Four paired pre-release runs per workload; 640 measured requests, zero errors. Mean images/s; C is client concurrency.",
    imageTasks: { generation: "generation", editing: "editing" },
    workloadLabel: "Workload",
    imagePerformanceNote: `Both implementations met P99 limits of 20 / 5 / 30 / 30 seconds, in row order. *2K editing is approximately equal: its 95% ratio interval, 0.9976–1.0077, includes 1. Resident-service measurements do not represent cold CLI defaults. <a href="${IMAGE_PERFORMANCE_URL}">Configuration, latency and memory</a>.`,
    textPerformanceLabel: "Historical text results · 0.13 candidate",
    supportTitle: "Architectures and formats",
    supportRows: [
      ["Text architectures", "Llama, Mistral, Qwen2, Qwen3 dense/MoE, Qwen3.5 hybrid dense/MoE, Gemma 3 text."],
      ["Weights", "GGUF, including Q4_K/Q5_K/Q6_K/Q8_0; SafeTensors. Selected CUDA paths: GPTQ INT4, compressed-tensors INT4, block-FP8."],
      ["Specific paths", "Gemma 4 compressed-tensors W4A16 and GPT-OSS MXFP4: CUDA. PQ2_0: complete-model validation on Metal; CUDA operator coverage only."],
      ["Images", "Qwen-Image-2.1-Turbo generation and single-PNG editing; validated on Linux CUDA SM120. Upcoming Metal support: 1024×1024 validated on M4 16 GiB with the configuration above."],
    ],
    supportNote: 'Support varies by architecture, format and backend. CPU is a compatibility backend; text support does not imply vision support. Embedding, ASR and TTS APIs were removed in 0.14.0. <a href="https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.14.0">Release notes</a> · <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/release-regression-catalog.md">Validation scope</a>.',
    performanceTitle: "Recent 27B throughput",
    performanceLead: "RTX 5090 · Qwen3.8-27B · frozen ShareGPT workload. Accepted 0.13 candidate measurements: fixed 32 slots, context/batch 2048; 200 requests × 2 repeats per cell, 8 warmups per repeat, zero protocol errors. These tables fix the request count and use each repeat’s actual elapsed time, rather than a fixed window. Output tok/s = total output tokens / total measured time; C in the tables is client concurrency.",
    ggufCaption: "GGUF Q4_K_M · same weights",
    ctCaption: "Compressed-tensors INT4 · group size 32",
    concurrency: "C",
    ratio: "Ferrum / baseline",
    performanceNote: 'Several cells fail latency SLOs. CT C32’s second repeat reached <strong>78.91%</strong>; 80% was not met in every repeat. CT uses FP16 activations/KV in Ferrum versus BF16 in vLLM (KV inferred from its auto setting). Same GGUF weights do not imply identical arithmetic. <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/performance-popular-models-p0.zh.md">Full results</a> include latency percentiles, memory and quality limits. Joint SLO optimization follows in the next version.',
    projectTitle: "An evolving open-source project",
    projectBody: "Ferrum uses AI-assisted development and iterates quickly. Internal APIs may change; model coverage and performance evolve. Quality remains a priority through correctness checks, model validation and honest reporting of failures. Personal GPU resources are limited; hardware access and contributions are welcome.",
    attribution: 'Thanks to <a href="https://github.com/vllm-project/vllm">vLLM</a> and <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> for code and implementation references. Component licenses and notices are preserved.',
    docsLabel: "Documentation",
    releaseLabel: `${RELEASE_VERSION} release`,
    footer: "Ferrum · MIT licensed",
  },
  zh: {
    lang: "zh-CN",
    path: "/zh/",
    alternatePath: "/",
    alternateLabel: "English",
    title: "Ferrum — Rust 文本与图像推理",
    description: "一个 Rust 二进制提供本地大模型推理、Qwen-Image 图像生成与编辑，以及 OpenAI 兼容 API。",
    nav: [["快速开始", "#quick-start"], ["图像", "#images"], ["支持范围", "#support"], ["吞吐结果", "#performance"], ["项目说明", "#project"]],
    eyebrow: `v${RELEASE_VERSION} · 已发布`,
    h1: "对话与生图，<br>一个 Rust 二进制。",
    lead: "本地大模型与 Qwen-Image 图像生成、编辑，提供 CLI 和 OpenAI 兼容 API。",
    goal: "目标：在满足 TTFT、TPOT、ITL SLO 的前提下，最大化输出吞吐。",
    goalContext: "固定硬件、模型与输出策略，在预设 SLO 下从可用策略集合 Π 中选择执行策略 π。",
    goalMathLabel: "在可用执行策略π中，最大化固定窗口T内成功完成请求的有效输出token数除以T，并满足各延迟分位上限及最低完成比例ρ。",
    goalDefinitions: [
      "W 固定请求集合与到达时序；T > 0 为固定测量窗口，结束即共同统计截止。N_out 只计窗口内成功完成请求的有效输出 token。",
      'L_k 为延迟采样分布，Q_p_k 为指定分位数，S_k 为上限。TTFT 包含排队；缺失延迟不视为达标。<a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/performance-popular-models-p0.zh.md">采样规则</a>。',
      "C 为同一 W 在截止前成功完成的比例，分母包含拒绝、失败、超时和未完成请求；p_k、S_k 与 0 < ρ ≤ 1 均须预先给定。",
    ],
    primary: `安装 ${RELEASE_VERSION}`,
    secondary: "查看 GitHub",
    quickTitle: "安装与使用",
    installTitle: "安装命令",
    copyLabel: "复制",
    copiedLabel: "已复制",
    copyFailedLabel: "浏览器未允许复制，请选中命令后手动复制。",
    installNote: "安装器校验发布包并配置 PATH。macOS/Linux 安装后打开新终端，PowerShell 当前会话即可使用。再次执行即可升级，模型与配置保留。",
    runtimeNote: 'macOS 使用 Metal。Linux/WSL 在全部可见 GPU 计算能力匹配时选择 sm89（CUDA 12.4）或 sm120（CUDA 12.8），需兼容的 NVIDIA 驱动及 CUDA/NCCL 运行库。Windows 与 Homebrew CUDA 面向 sm89；Windows 包内含 CUDA/VC 运行库。自动模式在 GPU 不受支持、混合架构或 CUDA 无法启动时回退 CPU；显式 CUDA 不满足条件或缺少对应资产会报错。<a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/release-regression.md#prepare-the-candidate">运行环境要求</a>。',
    runLabel: "终端对话",
    serveLabel: "API 服务",
    runNote: "首次运行自动下载权重；删除 --disable-thinking 即恢复模型模板默认的思考行为。",
    apiNote: 'API 地址：<code>http://127.0.0.1:8000/v1</code>，模型名 <code>ferrum</code>。<a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/openai-api-compatibility.md">流式输出、工具与结构化输出</a>。',
    defaultsNote: '默认执行与容量按声明能力、硬件和内存选择，显式设置优先。合资格的 SM120 GGUF 路径可采用 8 位激活量化及整数点积。<code>--numerical-profile qwen3_5.f32-master</code> 保留原混合精度参考路径；批布局可能改变输出。<a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/numerical-execution.zh.md">数值策略</a> · <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/cache-product.md">缓存说明</a>。',
    imageTitle: "图像生成与编辑",
    imageLead: '使用完整的本地 <a href="https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo">Qwen-Image-2.1-Turbo</a> 模型目录。已验证 RTX 5090、Linux CUDA 12.8、cuDNN 9。',
    imageMetalNote: 'Metal 图像后端正在为下一版本完成验收。M4 Mac mini、16 GiB 统一内存已完成 1024×1024 生成与编辑，使用 Q4_K_M GGUF DiT、FP8 文本编码器和 BF16 视觉编码器/VAE，通过 <code>--image-config</code> 显式配置后端、模型资产与内存上限。这套配置尚未完成 2K 出图，吞吐验收仍在进行。',
    imageMetalSetup: 'Metal：<a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/README_zh.md#图像生成与编辑">配置与使用命令</a>。',
    imageRunLabel: "CUDA 文生图",
    imageEditLabel: "CUDA 编辑单张 PNG",
    imageServeLabel: "CUDA 图像 API 服务",
    imageApi: 'HTTP：<code>POST /v1/images/generations</code> 接收 JSON，<code>POST /v1/images/edits</code> 接收 multipart。编辑支持单张 8 位 RGB/RGBA PNG，返回 PNG 的 <code>b64_json</code>；背景和细节可能重绘。',
    imageConfigNote: '本地目录预设最高支持 1024×1024；2K 及下表的常驻服务配置使用 <code>--image-config</code>。',
    imagePerformanceTitle: "Qwen-Image 对比 vLLM-Omni",
    imagePerformanceLead: "RTX 5090 · 8 步 · 文本编码器/DiT FP8、VAE BF16。每项 4 对发布候选配对测试，共 640 个正式请求，零错误。单位：张/秒；C 为客户端并发。",
    imageTasks: { generation: "生成", editing: "编辑" },
    workloadLabel: "任务",
    imagePerformanceNote: `两端均满足对应 P99 上限：20 / 5 / 30 / 30 秒。*2K 编辑约持平，比值 95% 区间为 0.9976–1.0077，差异尚不显著。表格采用常驻服务配置，不代表默认 CLI 冷启动性能。<a href="${IMAGE_PERFORMANCE_URL}">配置、延迟与内存</a>。`,
    textPerformanceLabel: "历史文本结果 · 0.13 候选",
    supportTitle: "架构与格式",
    supportRows: [
      ["文本架构", "Llama、Mistral、Qwen2、Qwen3 dense/MoE、Qwen3.5 hybrid dense/MoE、Gemma 3 文本。"],
      ["权重格式", "GGUF（含 Q4_K/Q5_K/Q6_K/Q8_0）、SafeTensors。部分 CUDA 路径支持 GPTQ INT4、compressed-tensors INT4、block-FP8。"],
      ["专用路径", "Gemma 4 compressed-tensors W4A16、GPT-OSS MXFP4 为 CUDA 路径。PQ2_0 完整模型仅在 Metal 实测，CUDA 目前为算子级验证。"],
      ["图像", "Qwen-Image-2.1-Turbo 生成与单张 PNG 编辑；已验证 Linux CUDA SM120。下一版本的 Metal 支持已在 M4 16 GiB、上述配置下验证 1024×1024。"],
    ],
    supportNote: '支持范围取决于架构、格式与后端组合。CPU 为兼容后端，文本支持不代表视觉支持。0.14.0 已移除 embedding、ASR 和 TTS API。<a href="https://github.com/sizzlecar/ferrum-infer-rs/releases/tag/v0.14.0">发布说明</a> · <a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/release-regression-catalog.md">验证范围</a>。',
    performanceTitle: "近期 27B 吞吐",
    performanceLead: "RTX 5090 · Qwen3.8-27B · 冻结 ShareGPT 负载。已验收的 0.13 候选测量：固定 32 槽、上下文/批处理 2048；每格正式 200 请求 × 2 轮，每轮另有 8 次预热，零协议错误。本表固定请求数，以各轮实际耗时计算吞吐，不采用固定窗口。输出 token/s＝总输出 token / 总正式时长；表中 C 为客户端并发。",
    ggufCaption: "GGUF Q4_K_M · 同权重",
    ctCaption: "Compressed-tensors INT4 · group size 32",
    concurrency: "C",
    ratio: "Ferrum / 对照",
    performanceNote: '多档延迟 SLO 未通过。CT C32 第二轮仅 <strong>78.91%</strong>，未证明每轮稳定达到 80%。CT 对比中 Ferrum 激活/KV 为 FP16，vLLM 为 BF16（KV 由 auto 配置推导）。同 GGUF 权重不等于内部算术相同。<a href="https://github.com/sizzlecar/ferrum-infer-rs/blob/main/docs/performance-popular-models-p0.zh.md">完整结果</a>列出延迟分位、内存与质量边界；联合 SLO 优化留到下一版本。',
    projectTitle: "持续迭代的开源项目",
    projectBody: "Ferrum 使用 AI 辅助开发，仍在快速迭代。内部 API 暂不承诺稳定，模型支持与性能持续更新。质量始终重要：保留正确性检查、模型验证并如实报告失败。个人 GPU 资源有限，欢迎提供硬件测试支持和贡献。",
    attribution: '感谢 <a href="https://github.com/vllm-project/vllm">vLLM</a> 与 <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a>，部分实现参考或复用了其代码；相关许可证与署名随组件保留。',
    docsLabel: "文档",
    releaseLabel: `${RELEASE_VERSION} 发布说明`,
    footer: "Ferrum · MIT 开源",
  },
};

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[character]);
}

function render(page) {
  const canonical = `${SITE_ORIGIN}${page.path}`;
  const schema = JSON.stringify({
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    name: "Ferrum",
    applicationCategory: "DeveloperApplication",
    operatingSystem: "macOS, Linux, Windows",
    softwareVersion: RELEASE_VERSION,
    description: page.description,
    url: canonical,
    codeRepository: REPOSITORY,
    downloadUrl: `${REPOSITORY}/releases`,
    license: `${REPOSITORY}/blob/main/LICENSE`,
    isAccessibleForFree: true,
  }).replace(/</g, "\\u003c");
  const nav = page.nav.map(([label, href]) => `<a href="${href}">${label}</a>`).join("");
  const support = page.supportRows.map(([title, body]) => `<div><dt>${title}</dt><dd>${body}</dd></div>`).join("");
  const codeBlock = (label, command, id) => `<div class="code-shell"><div class="code-top"><span>${label}</span><button class="copy-button" type="button" data-copy="${id}" aria-label="${page.copyLabel}: ${label}" hidden>${page.copyLabel}</button></div><pre><code class="command" id="${id}">${escapeHtml(command)}</code></pre></div>`;
  const installCommands = [
    ["macOS Apple Silicon / Linux x86_64", "curl -fsSL https://ferrum.pandaailabs.com/install.sh | sh"],
    ["Windows x64 · PowerShell", "irm https://ferrum.pandaailabs.com/install.ps1 | iex"],
  ].map(([platform, command], index) => codeBlock(platform, command, `install-command-${index}`)).join("");
  const runCommands = [
    [page.runLabel, "ferrum run qwen3.5:4b-q4_k_m --disable-thinking"],
    [page.serveLabel, "ferrum serve --model qwen3.5:4b-q4_k_m --served-model-name ferrum --disable-thinking --port 8000"],
  ].map(([label, command], index) => codeBlock(label, command, `run-command-${index}`)).join("");
  const table = (caption, baseline, values) => `<div class="table-wrap"><table><caption>${caption}<span>output tok/s</span></caption><thead><tr><th scope="col">${page.concurrency}</th><th scope="col">Ferrum</th><th scope="col">${baseline}</th><th scope="col">${page.ratio}</th></tr></thead><tbody>${values.map(([c, ferrum, reference, ratio]) => `<tr><th scope="row">${c}</th><td>${ferrum}</td><td>${reference}</td><td>${ratio}</td></tr>`).join("")}</tbody></table></div>`;
  const imageCommands = [
    [page.imageRunLabel, 'ferrum run /path/to/Qwen-Image-2.1-Turbo --image-precision fp8_linear_experimental_v1 --prompt "A red fox in a snowy forest" --width 1024 --height 1024 --output fox.png'],
    [page.imageEditLabel, 'ferrum run /path/to/Qwen-Image-2.1-Turbo --image fox.png --image-precision fp8_linear_experimental_v1 --prompt "Make the fox blue" --width 1024 --height 1024 --output edited.png'],
    [page.imageServeLabel, "ferrum serve --model /path/to/Qwen-Image-2.1-Turbo --served-model-name qwen-image --image-precision fp8_linear_experimental_v1 --port 8000"],
  ].map(([label, command], index) => codeBlock(label, command, `image-command-${index}`)).join("");
  const imageTable = `<div class="table-wrap"><table><caption>Qwen-Image-2.1-Turbo<span>images/s</span></caption><thead><tr><th scope="col">${page.workloadLabel}</th><th scope="col">C</th><th scope="col">Ferrum</th><th scope="col">vLLM-Omni</th><th scope="col">${page.ratio}</th></tr></thead><tbody>${imageThroughput.map(([task, size, c, ferrum, omni, ratio]) => `<tr><th scope="row">${size} ${page.imageTasks[task]}</th><td>${c}</td><td>${ferrum}</td><td>${omni}</td><td>${ratio}</td></tr>`).join("")}</tbody></table></div>`;
  return `<!doctype html>
<html lang="${page.lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${page.title}</title>
  <meta name="description" content="${escapeHtml(page.description)}">
  <meta name="robots" content="index,follow,max-image-preview:large">
  <link rel="canonical" href="${canonical}">
  <link rel="alternate" hreflang="en" href="${SITE_ORIGIN}/">
  <link rel="alternate" hreflang="zh-CN" href="${SITE_ORIGIN}/zh/">
  <link rel="alternate" hreflang="x-default" href="${SITE_ORIGIN}/">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <meta property="og:type" content="website">
  <meta property="og:site_name" content="Ferrum">
  <meta property="og:title" content="${page.title}">
  <meta property="og:description" content="${escapeHtml(page.description)}">
  <meta property="og:url" content="${canonical}">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="${page.title}">
  <meta name="twitter:description" content="${escapeHtml(page.description)}">
  <script type="application/ld+json">${schema}</script>
  <style>${css}</style>
</head>
<body>
  <nav aria-label="${page.lang === "en" ? "Primary navigation" : "主导航"}"><div class="wrap nav-inner">
    <a class="brand" href="${page.path}">${mark}<span>FERRUM</span></a>
    <div class="nav-links">${nav}<a class="lang" href="${page.alternatePath}" hreflang="${page.lang === "en" ? "zh-CN" : "en"}">${page.alternateLabel}</a></div>
  </div></nav>
  <main>
    <header><div class="wrap">
      <a class="eyebrow" href="${RELEASE_URL}">${page.eyebrow}</a>
      <h1>${page.h1}</h1>
      <p class="hero-copy">${page.lead}</p>
      <div class="objective">
        <p><strong>${page.goal}</strong></p>
        <p class="muted">${page.goalContext}</p>
        <math display="block" aria-label="${escapeHtml(page.goalMathLabel)}">
          <mtable columnalign="center" rowspacing="0.5em">
            <mtr><mtd><mrow><munder><mo>max</mo><mrow><mi>π</mi><mo>∈</mo><mi>Π</mi></mrow></munder><mspace width="0.4em"/><mfrac><mrow><msub><mi>N</mi><mtext>out</mtext></msub><mo>(</mo><mi>π</mi><mo>;</mo><mi>W</mi><mo>,</mo><mi>T</mi><mo>)</mo></mrow><mi>T</mi></mfrac></mrow></mtd></mtr>
            <mtr><mtd><mtext>s.t.</mtext></mtd></mtr>
            <mtr><mtd><mrow><msub><mi>Q</mi><msub><mi>p</mi><mi>k</mi></msub></msub><mo>(</mo><msub><mi>L</mi><mi>k</mi></msub><mo>(</mo><mi>π</mi><mo>;</mo><mi>W</mi><mo>,</mo><mi>T</mi><mo>)</mo><mo>)</mo><mo>≤</mo><msub><mi>S</mi><mi>k</mi></msub></mrow></mtd></mtr>
            <mtr><mtd><mrow><mo>∀</mo><mi>k</mi><mo>∈</mo><mo>{</mo><mtext>TTFT</mtext><mo>,</mo><mtext>TPOT</mtext><mo>,</mo><mtext>ITL</mtext><mo>}</mo></mrow></mtd></mtr>
            <mtr><mtd><mrow><mi>C</mi><mo>(</mo><mi>π</mi><mo>;</mo><mi>W</mi><mo>,</mo><mi>T</mi><mo>)</mo><mo>≥</mo><mi>ρ</mi></mrow></mtd></mtr>
          </mtable>
        </math>
        <ul class="objective-definitions">${page.goalDefinitions.map((definition) => `<li>${definition}</li>`).join("")}</ul>
      </div>
      <div class="actions"><a class="button primary" href="#quick-start">${page.primary}</a><a class="button" href="${REPOSITORY}">${page.secondary}</a></div>
    </div></header>
    <section id="quick-start"><div class="wrap">
      <div class="section-head"><h2>${page.quickTitle}</h2></div>
      <div class="install-commands" aria-label="${page.installTitle}">${installCommands}</div>
      <p class="muted">${page.installNote}</p>
      <p class="muted">${page.runtimeNote}</p>
      <div class="install-commands">${runCommands}</div>
      <p class="copy-status" role="status" aria-live="polite"></p>
      <p class="muted">${page.runNote}</p>
      <p>${page.apiNote}</p>
      <p class="note">${page.defaultsNote}</p>
    </div></section>
    <section id="images"><div class="wrap">
      <div class="section-head"><h2>${page.imageTitle}</h2><p>${page.imageLead}</p></div>
      <p class="note">${page.imageMetalNote}</p>
      <p>${page.imageMetalSetup}</p>
      <div class="install-commands">${imageCommands}</div>
      <p class="copy-status" role="status" aria-live="polite"></p>
      <p>${page.imageApi}</p>
      <p class="note">${page.imageConfigNote}</p>
    </div></section>
    <section id="support"><div class="wrap">
      <div class="section-head"><h2>${page.supportTitle}</h2></div>
      <dl class="support-list">${support}</dl>
      <p class="muted">${page.supportNote}</p>
    </div></section>
    <section id="performance"><div class="wrap">
      <div class="section-head"><h2>${page.imagePerformanceTitle}</h2><p>${page.imagePerformanceLead}</p></div>
      ${imageTable}
      <p class="note">${page.imagePerformanceNote}</p>
      <details><summary>${page.textPerformanceLabel}</summary>
        <div class="section-head"><h3>${page.performanceTitle}</h3><p>${page.performanceLead}</p></div>
        <div class="benchmark-grid">${table(page.ggufCaption, "llama.cpp b11429", throughput.gguf)}${table(page.ctCaption, "vLLM 0.31", throughput.ct)}</div>
        <p class="note">${page.performanceNote}</p>
      </details>
    </div></section>
    <section id="project"><div class="wrap">
      <div class="section-head"><h2>${page.projectTitle}</h2></div>
      <p>${page.projectBody}</p>
      <p class="muted">${page.attribution}</p>
      <div class="actions"><a class="button" href="${REPOSITORY}${page.lang === "en" ? "#readme" : "/blob/main/README_zh.md"}">${page.docsLabel}</a><a class="button" href="${RELEASE_URL}">${page.releaseLabel}</a></div>
    </div></section>
  </main>
  <footer><div class="wrap footer-inner"><span>${page.footer}</span><span><a href="${REPOSITORY}">GitHub</a> · <a href="${REPOSITORY}/blob/main/LICENSE">MIT License</a></span></div></footer>
  <script>
    document.querySelectorAll('[data-copy]').forEach((button) => {
      button.hidden = false;
      button.addEventListener('click', async () => {
        const status = button.closest('section').querySelector('.copy-status');
        const command = document.getElementById(button.dataset.copy).textContent;
        let copied = false;
        try {
          await navigator.clipboard.writeText(command);
          copied = true;
        } catch {
          const field = document.createElement('textarea');
          field.value = command;
          field.style.position = 'fixed';
          field.style.opacity = '0';
          field.setAttribute('readonly', '');
          document.body.appendChild(field);
          field.select();
          try { copied = document.execCommand('copy'); } catch {}
          field.remove();
          button.focus({ preventScroll: true });
        }
        status.textContent = copied ? ${JSON.stringify(page.copiedLabel)} : ${JSON.stringify(page.copyFailedLabel)};
        button.textContent = copied ? ${JSON.stringify(page.copiedLabel)} : ${JSON.stringify(page.copyLabel)};
        clearTimeout(button.copyTimer);
        button.copyTimer = setTimeout(() => { button.textContent = ${JSON.stringify(page.copyLabel)}; }, 2000);
      });
    });
  </script>
</body>
</html>`;
}

function response(body, contentType, status = 200) {
  return new Response(body, {
    status,
    headers: {
      "content-type": `${contentType}; charset=utf-8`,
      "cache-control": contentType === "text/html" ? "public, max-age=60, s-maxage=60" : "public, max-age=300, s-maxage=3600",
      "content-security-policy": "default-src 'none'; script-src 'unsafe-inline' https://static.cloudflareinsights.com; connect-src 'self' https://cloudflareinsights.com; style-src 'unsafe-inline'; img-src 'self' data: https://ferrum-downloads.pandaailabs.com https://github.com https://github-production-user-asset-6210df.s3.amazonaws.com; media-src https://github.com https://github-production-user-asset-6210df.s3.amazonaws.com; base-uri 'none'; form-action 'none'; frame-ancestors 'none';",
      "referrer-policy": "strict-origin-when-cross-origin",
      "x-content-type-options": "nosniff",
      "x-frame-options": "DENY",
    },
  });
}

function installerResponse(request, source) {
  const allowed = request.method === "GET" || request.method === "HEAD";
  const result = response(
    allowed ? (request.method === "HEAD" ? null : source) : "Method Not Allowed\n",
    "text/plain",
    allowed ? 200 : 405,
  );
  result.headers.set("cache-control", "no-cache");
  if (!allowed) result.headers.set("allow", "GET, HEAD");
  return result;
}

export default {
  async fetch(request) {
    const { pathname } = new URL(request.url);
    if (pathname.startsWith("/downloads/")) {
      if (request.method !== "GET" && request.method !== "HEAD") {
        return new Response("Method Not Allowed\n", { status: 405, headers: { allow: "GET, HEAD" } });
      }
      const asset = pathname.match(/^\/downloads\/(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\/ferrum-[A-Za-z0-9._-]+)$/);
      if (!asset) return new Response("Not Found\n", { status: 404 });
      const headers = new Headers();
      for (const name of ["range", "if-range", "if-none-match", "if-modified-since"]) {
        if (request.headers.has(name)) headers.set(name, request.headers.get(name));
      }
      // Cloudflare's fetch cache streams large immutable release objects. Do not
      // buffer or tee installer bodies into the Worker's limited memory.
      const options = {
        method: request.method,
        headers,
        cf: { cacheEverything: true, cacheTtlByStatus: { "200-299": 31536000, "400-599": -1 } },
      };
      const mirrored = await fetch(`https://ferrum-downloads.pandaailabs.com/${asset[1]}`, { ...options, redirect: "manual" });
      if (mirrored.status !== 404) return mirrored;
      await mirrored.body?.cancel();
      // Newly published versions work before an optional R2 mirror is populated.
      // Resolve GitHub's asset redirect at the edge, keeping downloads on the CDN.
      return fetch(`${REPOSITORY}/releases/download/${asset[1]}`, { ...options, redirect: "follow" });
    }
    if (pathname === "/install.sh") {
      return installerResponse(request, shellInstaller);
    }
    if (pathname === "/install.ps1") {
      return installerResponse(request, powerShellInstaller);
    }
    if (pathname === "/robots.txt") {
      return response(`User-agent: *\nAllow: /\nSitemap: ${SITE_ORIGIN}/sitemap.xml\n`, "text/plain");
    }
    if (pathname === "/sitemap.xml") {
      const today = new Date().toISOString().slice(0, 10);
      return response(`<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>${SITE_ORIGIN}/</loc><lastmod>${today}</lastmod></url><url><loc>${SITE_ORIGIN}/zh/</loc><lastmod>${today}</lastmod></url></urlset>\n`, "application/xml");
    }
    if (pathname === "/favicon.svg" || pathname === "/logo.svg") {
      return response(logoSvg, "image/svg+xml");
    }
    if (pathname === "/zh") {
      return Response.redirect(`${SITE_ORIGIN}/zh/`, 308);
    }
    if (pathname === "/" || pathname === "/index.html") {
      return response(render(pages.en), "text/html");
    }
    if (pathname === "/zh/" || pathname === "/zh/index.html") {
      return response(render(pages.zh), "text/html");
    }
    return response("Not Found\n", "text/plain", 404);
  },
};
