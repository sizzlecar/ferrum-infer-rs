const SITE_ORIGIN = "https://ferrum.pandaailabs.com";
const REPOSITORY = "https://github.com/sizzlecar/ferrum-infer-rs";

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
.actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 34px; }
.button { display: inline-flex; align-items: center; justify-content: center; min-height: 46px; padding: 0 19px; border-radius: 10px; border: 1px solid var(--line); background: var(--panel); text-decoration: none; font-weight: 700; }
.button.primary { border-color: var(--rust); background: var(--rust); color: #180b06; }
.button:hover { transform: translateY(-1px); border-color: #59636c; }
.button.primary:hover { border-color: var(--rust-light); background: var(--rust-light); }
.proof { display: flex; flex-wrap: wrap; gap: 18px 30px; margin-top: 38px; color: var(--muted); font-size: .93rem; }
.proof span::before { content: ""; display: inline-block; width: 7px; height: 7px; margin-right: 9px; border-radius: 50%; background: var(--green); }
section { padding: 68px 0; }
.section-head { max-width: 760px; margin-bottom: 30px; }
h2 { margin: 0 0 10px; font-size: clamp(2rem, 5vw, 3.25rem); line-height: 1.05; letter-spacing: -.045em; }
.section-head p, .muted { color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.card { padding: 25px; border: 1px solid var(--line); border-radius: 16px; background: linear-gradient(145deg, rgba(24,29,34,.94), rgba(15,18,21,.94)); }
.card h3 { margin: 0 0 8px; font-size: 1.15rem; }
.card p { margin: 0; color: var(--muted); }
.code-shell { overflow: hidden; border: 1px solid #303840; border-radius: 16px; background: #080a0b; box-shadow: 0 26px 80px rgba(0,0,0,.28); }
.code-top { display: flex; align-items: center; justify-content: space-between; min-height: 48px; padding: 0 18px; border-bottom: 1px solid #20262b; color: var(--muted); font-size: .82rem; }
.dots { display: flex; gap: 7px; }
.dots i { display: block; width: 9px; height: 9px; border-radius: 50%; background: #3b434a; }
pre { margin: 0; padding: 24px; overflow-x: auto; color: #e6e3dc; font: 500 .92rem/1.75 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
.comment { color: #78838c; }
.command { color: #ffad80; }
.note { margin-top: 18px; padding: 15px 18px; border-left: 3px solid var(--rust); background: rgba(240,104,47,.07); color: #cdd1d4; }
.split { display: grid; grid-template-columns: 1.05fr .95fr; gap: 42px; align-items: start; }
.list { display: grid; gap: 12px; }
.list-item { display: grid; grid-template-columns: 28px 1fr; gap: 12px; padding: 16px 0; border-bottom: 1px solid var(--line); }
.list-item b { color: var(--rust-light); }
.faq { display: grid; gap: 10px; }
details { border: 1px solid var(--line); border-radius: 13px; background: rgba(18,22,26,.85); }
summary { cursor: pointer; padding: 17px 20px; font-weight: 700; }
details p { margin: 0; padding: 0 20px 19px; color: var(--muted); }
.cta { padding: 38px; border: 1px solid rgba(240,104,47,.45); border-radius: 20px; background: linear-gradient(120deg, rgba(240,104,47,.17), rgba(129,216,167,.07)); }
.cta h2 { max-width: 780px; }
footer { margin-top: 52px; padding: 34px 0 48px; border-top: 1px solid var(--line); color: var(--muted); font-size: .9rem; }
.footer-inner { display: flex; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
@media (max-width: 820px) {
  header { padding-top: 72px; }
  .grid, .split { grid-template-columns: 1fr; }
  .nav-links a:not(.lang) { display: none; }
  .cta { padding: 28px; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .button:hover { transform: none; }
}
`;

const logoShapes = `<rect x="4" y="4" width="56" height="56" rx="13" fill="#0b0d0f"/><path d="M16 14H50L47 24H29V31H38L35 41H29V52H16Z" fill="#f0682f"/><path d="M38 31H50L47 41H35Z" fill="#81d8a7"/>`;
const mark = `<svg class="mark" viewBox="0 0 64 64" aria-hidden="true">${logoShapes}</svg>`;
const logoSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Ferrum">${logoShapes}</svg>`;

const pages = {
  en: {
    lang: "en",
    path: "/",
    alternatePath: "/zh/",
    alternateLabel: "中文",
    title: "Ferrum — Rust-native local LLM inference for Metal and CUDA",
    description: "Run and serve local LLMs with one Rust binary. Ferrum provides Apple Silicon Metal and NVIDIA CUDA acceleration plus OpenAI-compatible APIs without a Python runtime.",
    nav: [
      ["Quick start", "#quick-start"],
      ["Features", "#features"],
      ["FAQ", "#faq"],
    ],
    eyebrow: "Local inference, without the runtime stack",
    h1: "One Rust binary for local LLM inference.",
    lead: "Ferrum runs language models on Apple Silicon Metal and NVIDIA CUDA, then serves them through an OpenAI-compatible API — without requiring Python, PyTorch, or vLLM at runtime.",
    primary: "View on GitHub",
    secondary: "Start locally",
    proof: ["MIT licensed", "Metal + CUDA", "Run + OpenAI-compatible serve"],
    featureTitle: "The direct path from model to API",
    featureLead: "Ferrum keeps the first experience small: inspect the install, name a model explicitly, run a prompt, or expose the same model over HTTP.",
    cards: [
      ["Rust-native product", "A single CLI and server binary with no Python runtime in the supported prebuilt release path."],
      ["Two accelerator backends", "Apple Silicon uses Metal and GGUF; NVIDIA sm89 uses CUDA with GPTQ or safetensors models."],
      ["OpenAI-compatible", "Chat Completions, streaming usage, Responses text, function tools, and structured output."],
      ["Explicit model choice", "Ferrum never silently selects a model. Every run and server starts from a model you named."],
      ["Server controls", "Continuous batching, paged KV cache, prefix cache, session cache, and typed admission controls."],
      ["Release-gated", "Both ferrum run and ferrum serve are validated against documented release flows before promotion."],
    ],
    quickTitle: "From install to first answer",
    quickLead: "The first run downloads model weights. Ferrum prints download progress; the Metal model below is about 2.55 GiB and the CUDA repository is about 8.7 GiB.",
    note: "The quick start disables verbose reasoning for a short first response. Remove --disable-thinking when you want the model template's default reasoning behavior.",
    platformTitle: "Built for local and private serving",
    platformLead: "Use the interactive CLI while exploring, then serve the same model behind a familiar HTTP contract.",
    platformRows: [
      ["01", "Inspect before download", "ferrum doctor maps aliases and checks the compiled accelerator without downloading model weights."],
      ["02", "Run interactively", "ferrum run opens a local chat loop and keeps multi-turn context in the process."],
      ["03", "Serve over HTTP", "ferrum serve exposes health, model discovery, Chat Completions, and Responses endpoints."],
    ],
    faqTitle: "Frequently asked questions",
    faq: [
      ["What is Ferrum?", "Ferrum is an open-source Rust workspace and command-line product for running and serving language models locally on supported Metal and CUDA hardware."],
      ["Does Ferrum require Python?", "No Python runtime is required for the official prebuilt Metal and CUDA binaries. The CUDA host still needs compatible NVIDIA driver, CUDA runtime, and NCCL runtime libraries."],
      ["Which models are in the formal release scope?", "Ferrum v0.8 formally covers Qwen3.5 4B and 35B-A3B, Qwen3 30B-A3B, and Llama 3.1 8B dense. Other model work may appear as development evidence before entering the support matrix."],
      ["Can existing OpenAI clients connect to Ferrum?", "Ferrum implements OpenAI-compatible Chat Completions and stateless Responses surfaces. Check the compatibility document for the exact request contract."],
      ["Is Ferrum a hosted AI service?", "No. Ferrum is inference software that runs on hardware you control. Model licenses and data-handling choices remain yours."],
    ],
    ctaTitle: "Run your next local model with fewer moving pieces.",
    ctaPrimary: "Read the documentation",
    ctaSecondary: "Browse releases",
    footer: "Ferrum is open-source software maintained by Panda AI Labs contributors.",
  },
  zh: {
    lang: "zh-CN",
    path: "/zh/",
    alternatePath: "/",
    alternateLabel: "English",
    title: "Ferrum — 面向 Metal 与 CUDA 的 Rust 原生本地大模型推理",
    description: "使用一个 Rust 二进制运行与部署本地大模型。Ferrum 支持 Apple Silicon Metal、NVIDIA CUDA 和 OpenAI 兼容 API，无需 Python runtime。",
    nav: [
      ["快速开始", "#quick-start"],
      ["功能", "#features"],
      ["常见问题", "#faq"],
    ],
    eyebrow: "本地推理，不必背负复杂 runtime",
    h1: "一个 Rust 二进制，完成本地大模型推理。",
    lead: "Ferrum 在 Apple Silicon Metal 与 NVIDIA CUDA 上运行语言模型，并通过 OpenAI 兼容 API 提供服务——运行官方预编译版本无需 Python、PyTorch 或 vLLM。",
    primary: "查看 GitHub",
    secondary: "开始使用",
    proof: ["MIT 开源", "Metal + CUDA", "命令行运行 + OpenAI 兼容服务"],
    featureTitle: "从模型直接到 API",
    featureLead: "Ferrum 让首次体验保持简单：检查安装、明确指定模型、运行一次对话，或者把同一模型开放为 HTTP 服务。",
    cards: [
      ["Rust 原生产品", "受支持的预编译发布路径只有一个 CLI/Server 二进制，不依赖 Python runtime。"],
      ["两种加速后端", "Apple Silicon 使用 Metal 与 GGUF；NVIDIA sm89 使用 CUDA 与 GPTQ/safetensors。"],
      ["OpenAI 兼容", "支持 Chat Completions、streaming usage、Responses 文本、函数工具与结构化输出。"],
      ["明确选择模型", "Ferrum 不会静默选择默认模型；每次 run 与 serve 都从你明确指定的模型开始。"],
      ["服务端能力", "支持 continuous batching、paged KV cache、prefix cache、session cache 与 typed admission。"],
      ["发布门禁", "正式提升版本前，ferrum run 与 ferrum serve 都会按公开文档流程完成回归。"],
    ],
    quickTitle: "从安装到第一次回答",
    quickLead: "首次运行需要下载模型权重。Ferrum 会输出下载进度；下面的 Metal 模型约为 2.55 GiB，CUDA 仓库约为 8.7 GiB。",
    note: "快速开始通过 --disable-thinking 缩短第一次回答。需要模型模板默认推理行为时，删除这个参数即可。",
    platformTitle: "为本地与私有服务而构建",
    platformLead: "探索阶段使用交互式 CLI，随后通过熟悉的 HTTP 契约提供同一个模型。",
    platformRows: [
      ["01", "下载前检查", "ferrum doctor 会解析 alias 并检查已编译的加速后端，不会下载模型权重。"],
      ["02", "交互式运行", "ferrum run 启动本地对话循环，并在进程内保留多轮上下文。"],
      ["03", "通过 HTTP 服务", "ferrum serve 提供健康检查、模型发现、Chat Completions 与 Responses endpoint。"],
    ],
    faqTitle: "常见问题",
    faq: [
      ["Ferrum 是什么？", "Ferrum 是一个开源 Rust workspace 与命令行产品，用于在受支持的 Metal 和 CUDA 硬件上本地运行并提供语言模型服务。"],
      ["Ferrum 需要 Python 吗？", "官方预编译 Metal 与 CUDA 二进制无需 Python runtime。CUDA 主机仍需兼容的 NVIDIA driver、CUDA runtime 与 NCCL runtime。"],
      ["哪些模型属于正式发布范围？", "Ferrum v0.8 正式覆盖 Qwen3.5 4B 与 35B-A3B、Qwen3 30B-A3B 和 Llama 3.1 8B dense。其他模型可能先以开发证据出现，完成发布级门禁后才进入支持矩阵。"],
      ["现有 OpenAI client 能连接 Ferrum 吗？", "Ferrum 实现 OpenAI 兼容的 Chat Completions 与无状态 Responses 接口；精确请求契约请查看兼容性文档。"],
      ["Ferrum 是托管 AI 服务吗？", "不是。Ferrum 是运行在你控制的硬件上的推理软件；模型许可与数据处理方式仍由你决定。"],
    ],
    ctaTitle: "用更少的组件运行你的下一个本地模型。",
    ctaPrimary: "阅读文档",
    ctaSecondary: "查看发布版本",
    footer: "Ferrum 是由 Panda AI Labs 贡献者维护的开源软件。",
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
  const alternate = `${SITE_ORIGIN}${page.alternatePath}`;
  const schema = JSON.stringify({
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    name: "Ferrum",
    applicationCategory: "DeveloperApplication",
    operatingSystem: "macOS, Linux",
    description: page.description,
    url: canonical,
    codeRepository: REPOSITORY,
    downloadUrl: `${REPOSITORY}/releases`,
    license: `${REPOSITORY}/blob/main/LICENSE`,
    isAccessibleForFree: true,
  }).replace(/</g, "\\u003c");
  const cards = page.cards.map(([title, body]) => `<article class="card"><h3>${title}</h3><p>${body}</p></article>`).join("");
  const rows = page.platformRows.map(([number, title, body]) => `<div class="list-item"><b>${number}</b><div><strong>${title}</strong><div class="muted">${body}</div></div></div>`).join("");
  const faq = page.faq.map(([question, answer]) => `<details><summary>${question}</summary><p>${answer}</p></details>`).join("");
  const nav = page.nav.map(([label, href]) => `<a href="${href}">${label}</a>`).join("");
  const proof = page.proof.map((item) => `<span>${item}</span>`).join("");
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
  <nav aria-label="Primary navigation"><div class="wrap nav-inner">
    <a class="brand" href="${page.path}">${mark}<span>FERRUM</span></a>
    <div class="nav-links">${nav}<a class="lang" href="${page.alternatePath}" hreflang="${page.lang === "en" ? "zh-CN" : "en"}">${page.alternateLabel}</a></div>
  </div></nav>
  <main>
    <header><div class="wrap">
      <div class="eyebrow">${page.eyebrow}</div>
      <h1>${page.h1}</h1>
      <p class="hero-copy">${page.lead}</p>
      <div class="actions"><a class="button primary" href="${REPOSITORY}">${page.primary}</a><a class="button" href="#quick-start">${page.secondary}</a></div>
      <div class="proof">${proof}</div>
    </div></header>
    <section id="features"><div class="wrap">
      <div class="section-head"><h2>${page.featureTitle}</h2><p>${page.featureLead}</p></div>
      <div class="grid">${cards}</div>
    </div></section>
    <section id="quick-start"><div class="wrap">
      <div class="section-head"><h2>${page.quickTitle}</h2><p>${page.quickLead}</p></div>
      <div class="code-shell"><div class="code-top"><span>Terminal</span><span class="dots"><i></i><i></i><i></i></span></div><pre><span class="comment"># macOS Apple Silicon</span>
<span class="command">brew tap sizzlecar/ferrum
brew install ferrum</span>
ferrum doctor qwen3.5:4b-q4_k_m
ferrum run qwen3.5:4b-q4_k_m --disable-thinking

<span class="comment"># Linux x86_64 · NVIDIA CUDA sm89</span>
<span class="command">brew tap sizzlecar/ferrum
brew install ferrum-cuda</span>
ferrum doctor qwen3.5:4b
ferrum run qwen3.5:4b --disable-thinking</pre></div>
      <p class="note">${page.note}</p>
    </div></section>
    <section><div class="wrap split">
      <div><div class="section-head"><h2>${page.platformTitle}</h2><p>${page.platformLead}</p></div><a class="button" href="${REPOSITORY}/blob/main/docs/openai-api-compatibility.md">OpenAI API contract</a></div>
      <div class="list">${rows}</div>
    </div></section>
    <section id="faq"><div class="wrap"><div class="section-head"><h2>${page.faqTitle}</h2></div><div class="faq">${faq}</div></div></section>
    <section><div class="wrap"><div class="cta"><h2>${page.ctaTitle}</h2><div class="actions"><a class="button primary" href="${REPOSITORY}#quick-start">${page.ctaPrimary}</a><a class="button" href="${REPOSITORY}/releases">${page.ctaSecondary}</a></div></div></div></section>
  </main>
  <footer><div class="wrap footer-inner"><span>${page.footer}</span><span><a href="${REPOSITORY}">GitHub</a> · <a href="${REPOSITORY}/blob/main/LICENSE">MIT License</a></span></div></footer>
</body>
</html>`;
}

function response(body, contentType, status = 200) {
  return new Response(body, {
    status,
    headers: {
      "content-type": `${contentType}; charset=utf-8`,
      "cache-control": "public, max-age=300, s-maxage=3600",
      "content-security-policy": "default-src 'none'; script-src 'unsafe-inline' https://static.cloudflareinsights.com; connect-src 'self' https://cloudflareinsights.com; style-src 'unsafe-inline'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none';",
      "referrer-policy": "strict-origin-when-cross-origin",
      "x-content-type-options": "nosniff",
      "x-frame-options": "DENY",
    },
  });
}

export default {
  async fetch(request) {
    const { pathname } = new URL(request.url);
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
