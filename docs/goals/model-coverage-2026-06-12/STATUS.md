# STATUS — model-coverage-2026-06-12

进度日志,倒序。

## 2026-06-12(午后)— gate 矩阵 + 验证器落地;两个 GOAL 修订提案待批

- **`w1_matrix.json` + `scripts/w1_goal_validator.py` 落地**:7 个模型 ×
  9 个 cell = 63 cell,当前 10/63 满足(L0 ×6 + waived ×4)。验证器是唯一
  允许打印 `MODEL_COVERAGE_W1 GOAL PASS` 的程序;cell 必须 pass(带
  artifact)或 waived(带理由),引用的 artifact 必须存在。
- **L0 完成度**:43/43 golden 全过(9 个 fixture 模型,新增 Mistral 线
  ×3 + Llama-3.1;`strftime_now` 时钟注入 + `tojson(indent=N)` 两个真实
  渲染缺口由 L0 抓出并已修,commit `c8f3703e`)。
- smoke 阶梯扩充:known-answer 1x→10x(对齐 L2 判据),新增自定义 stop
  机制断言 + max_tokens 截断断言(L3 缺口)。
- **修订提案 #1(L1,需用户决定)**:L1 BF16 byte-equal 对 14B+ 在现有
  硬件上物理不可行(14B BF16=28GB>24GB 单卡;32B=64GB;70B=140GB)。
  提案:L1 按"代码路径代表"执行——每条代码路径取硬件放得下的最大代表
  (Qwen3 dense → 8B/0.6B 已有 reference_match;Qwen3-MoE → 30B-A3B 需
  pod 上 BF16?同样放不下,24GB 单卡上 MoE BF16 60GB 也不可行 → MoE 路径
  L1 只能 waive 到"Mac/CPU 逐层激活对照"或双卡)。大尺寸模型靠"同代码
  路径 + L2 行为对照"传递。**未批前 5 个 l1_bf16 cell 保持 pending。**
- **修订提案 #2(L3,需用户决定)**:blast-radius 套件断言对 0.6B 哨兵
  模型定制(canonical id、即答行为),对 8B-32B reasoning 模型强行参数化
  会又重又脆。提案:L3 判据改为"model_coverage_smoke 的 L3 段全绿"
  (多轮/stream/自然 EOS/自定义 stop/max_tokens/reasoning 提取,行为
  断言与套件同源),blast-radius 套件保持小模型哨兵职责(引擎级回归)。
  **未批前 L3 cell 不以 smoke 结果记 pass。**

## 2026-06-12(午前)— L0 扩面:模板同一性 + Mistral/Llama golden

- **模板同一性(HF raw tokenizer_config,sha256 前 16 位)**:
  Qwen3-0.6B / 14B / 32B 模板逐字节同一份(`a55ee1b1660128b7`,EOS
  `<|im_end|>`);R1-Distill-Qwen-14B / 32B 同一份(`56a1447ad31926fd`,
  EOS `<｜end▁of▁sentence｜>`)。**结论:Qwen3-14B/32B 与 R1-Distill-14B
  的 L0 由现有 golden fixture 直接覆盖**,各自只剩 per-model
  EOS/generation_config 断言(T4 机制已通用)。
- Mistral 24B 线 + Llama-3.1 golden fixture 生成中(来源 = serve 实际用的
  tokenizer 仓库:unsloth 镜像 ×2 + mistralai 上游 ×1 + unsloth Llama)。
  环境坑:huggingface_hub 新版走 httpx,SOCKS 代理需要 `httpx[socks]`
  (socksio),`pysocks` 只管 requests。
- Coder-30B GGUF(17.28GB)断点续传循环推进中(代理频繁断流,每次
  尝试落 1–3GB,`.incomplete` blob 在涨)。

## 2026-06-12(深夜 III)— ✅ R1-0528-Qwen3-8B GGUF Metal smoke 全绿

**W1 第一个模型过本地阶梯**:`FERRUM W1 SMOKE PASS: deepseek-r1:8b-q4_k_m`
(8/8:known-answer、自然 EOS、reasoning 提取、think 不漏入 content、
多轮记忆、stream==non-stream、required tool 10/10、strict json_schema
10/10)。证据:`artifacts/smoke_deepseek-r1-8b-q4_k_m_metal_2026-06-12.txt`。
serve 参数:`--kv-capacity 8192 --max-num-seqs 4`(见下条 thrash 诊断)。
注意:这是 L2/L3/L4 的可跑子集;最终认证仍需完整套件
(json_schema 20/20 走 server_structured_output)+ CUDA 侧 gate。

## 2026-06-12(深夜 II)— GGUF pull 产品缺口修复 + KV 池 thrash 诊断

R1-8B GGUF smoke 调试中钉死三个真实产品问题(全部影响 W1 每个 GGUF alias):

1. **pull sidecar 全量下载 bug(已修)**:GGUF 仓库缺 tokenizer.json 时,
   兜底走 `HfDownloader::download(sibling)` —— 会把 sibling 的 **safetensors
   权重(8B≈16GB)整库拉下来**,只为拿 tokenizer。磁盘紧张时必死,这就是
   此前需要手工拷 tokenizer 的根因。新增
   `HfDownloader::download_sidecar_files`(只拉指定小文件),pull 改用之,
   清单补上 `generation_config.json`(EOS 解析第一优先级)+
   `chat_template.jinja`。
2. **bartowski 系 sibling 映射全断(已修)**:HF API 实测 9 个 W1 GGUF 仓库
   **全部不带 tokenizer.json**,sibling 兜底是必经之路;而 strip `-GGUF`
   约定对 bartowski/*(无 safetensors 镜像)全部失效。
   `tokenizer_sibling_repo` 加显式映射(2026-06-12 HF API 逐个核实
   tokenizer.json 存在):Qwen2.5-Coder→Qwen 官方;Mistral-Small-3.2 /
   Magistral→unsloth 镜像(**mistralai 上游只有 tekken 格式,无 HF
   tokenizer.json**);Devstral 2→mistralai 上游;Llama 系→unsloth 镜像
   (meta-llama 上游 gated)。
3. **`--kv-capacity` 单独抬高 = 32GB Mac 内存灾难(smoke 已加防护)**:
   KV 池 = `max_num_seqs × kv_capacity`。autosizer server 档默认
   (32, 512)≈2GB;只把 capacity 提到 8192 会得到 32×8192≈36GB 池
   (8B/36 层/8KV头/128hd),Metal 分配直接把机器打进内存压缩 thrash
   (实测:health 能过、首个请求触发 `ensure_kv` 后 600s 超时,压缩器
   存页 38GB)。smoke 的 reasoning 档改为
   `--kv-capacity 8192 --max-num-seqs 4`(池 32K token,与默认同量级)。
   **autosizer 产品缺口升级**:reasoning 模型需要的不是"调大 capacity",
   而是 (seqs × capacity) 在显存预算内的联合推导 + 长上下文低并发档位;
   `--kv-capacity` 作为独立产品 flag 缺少联动护栏。

## 2026-06-12(深夜)— 本地验证推进与环境修正

- **修正**:HF 缓存里的 R1-0528-8B / R1-Distill-32B / Qwen3-Coder-30B /
  Qwen2.5-Coder-32B 仅为 6–11MB 元数据壳(config/tokenizer),**无权重**。
  W1 端到端一律需要下载。
- 磁盘:删除 target/debug(15GB)后约 16GB 可用;R1-8B Q4_K_M GGUF(~5GB)
  下载中(第一次因网络/代理 "error decoding response body" 失败,重试中);
  Qwen3-Coder-30B Q4_K_M(~18.6GB)需要更多空间——待用户清理或换机。
- 新增 `scripts/model_coverage_smoke.sh <alias> [--reasoning]`:
  L2/L3/L4 阶梯(known-answer + 自然 EOS / 多轮 / stream==non-stream /
  reasoning 提取 / required tool 10x / strict schema 10x),所有 W1 模型复用。
- 下一步(按序):R1-8B GGUF smoke(--reasoning)→ 视磁盘跑
  qwen3-coder:30b-q4_k_m → W1 收尾(README 矩阵 + 验证器)→ pod 合同。

## 2026-06-12(深夜)— blast-radius 存量回归结果

T3/T4/T5 处于 EOS/stop/模板爆炸半径,全套件(release + Metal,真模型)结果:

- ✅ chat_smoke 13 / server_smoke 10 / chat_pty 3 / chat_stress 2 / server_stress 2
- ✅ server_openai_compat 7/7 — 其中两处修复:
  - `test_python_openai_sdk_*`:本机环境缺 `openai`/`socksio`(SOCKS 代理),
    已 pip --user 安装,非代码问题。
  - `test_openai_client_tools_stream_*`:模板修正后 prompt 与 transformers
    字节一致(差 1 token),0.6B 贪心解码改为真的调用工具——服务器输出了
    规范的 tool_calls delta + finish=tool_calls + usage。测试断言改为
    "文本 XOR 合法工具调用"(7c69e2a7),钉住流式机制而非模型选择。
- ⏸ reference_match:1 行 drift **等用户审核后 re-baseline**(分类器按
  CLAUDE.md 拦截了自动重置,正确):case `qwen3-0.6b-arith-2-plus-3`
  内容与 token 数完全一致,仅 `finish_reason: length → stop` ——
  这是 EOS 修复的直接证据(此前 tokenizer 探测不到 Qwen EOS,自然停止
  被误归因为 budget 耗尽)。审核通过后执行:
  `FERRUM_UPDATE_FIXTURES=1 cargo test --release -p ferrum-cli --features metal --test reference_match -- --ignored --test-threads=1`

## 2026-06-12(晚)

- **T5 完成:L0 golden 基建落地并修出 7 处真实偏差**(PR #234,auto-merge):
  - `scripts/gen_chat_template_goldens.py` + 5 模型 23 用例 fixture 入库,
    `chat_template_golden` 测试 23/23 与 transformers 字节级一致。
  - 修复项:trim_blocks/lstrip_blocks 对齐 transformers;tojson 改 Python
    json.dumps 风格(自定义 filter);minijinja+serde_json 双 preserve_order
    (minijinja 对 Rust struct 字段强制字母序,tools 改为有序 JSON 值进模板);
    `PromptMessage::new` 不再急切剥离 assistant 历史的 `<think>`
    (剥不剥是模板的政策:DeepSeek 剥、Qwen3-Coder 保留)。
- **W1 全模型 alias 配齐**(均经 HF API 核实文件名):safetensors/GPTQ/GGUF
  三组,含 deepseek-r1:8b/14b/32b、qwen3-coder:30b、qwen3:14b/32b、
  qwen2.5-coder、mistral-small/devstral/magistral 24b 线。
- **YaRN clamp 落地**:不支持的 rope_scaling → `max_seq_len` clamp 到
  `original_max_position_embeddings` + 启动警告(R1-0528 由 131072 clamp 到
  32768),含单测。
- **环境约束发现**:本机磁盘 100%(HF 缓存 42GB);已清理 target/debug/
  incremental 释放 7.3GB。**新模型权重无法下载**,但缓存中已有
  R1-0528-Qwen3-8B、R1-Distill-32B、Qwen3-Coder-30B、Qwen2.5-Coder-32B 的
  safetensors + blast-radius 三小模型 → 本地验证用缓存模型推进。
- blast-radius 套件(chat_smoke/pty/stress + server 三件 + reference_match)
  在后台执行中——T3/T4/T5 改动处于 EOS/stop/模板爆炸半径,存量回归必须绿。

## 2026-06-12(下午)

- **T3 完成并提交(`778082a6`)**:minijinja-contrib pycompat 接入;模板渲染失败/
  渲染为空改为硬错误(消灭静默 fallback);tools-unaware 模板改为"注入工具 spec 后
  仍走模型模板渲染"(此前会静默丢弃工具定义)。新增 5 条防回归测试,
  workspace 测试全绿。
- **T2 完成(外部核查,逐仓库 config 原文)**,三个重要修正:
  - **GLM-4.7-Flash 是 MLA 注意力**(q_lora 768/kv_lora 512)+ noaux_tc 路由 + MTP,
    接入成本 MEDIUM→LARGE,已从 W2 移出(W2 只剩 Gemma 3 27B)。
  - **R1-0528-Qwen3-8B 无 Marlin-clean GPTQ**(QuantTrio 版 sym=false+4/8 混合)
    → CUDA 走 BF16 + GGUF。**Devstral 2 同样无 Marlin GPTQ** → GGUF/BF16。
  - W1 各模型的 Marlin-clean GPTQ 仓库已逐一锁定(jart25 / OPEA / JunHowie /
    Qwen 官方 / Intel AutoRound),写入 GOAL.md UNVERIFIED #4。
  - Qwen3.6 无官方 GPTQ-Int4(仅 FP8);官方 GPTQ 停在 Qwen3.5 代。

## 2026-06-12

- GOAL.md 建立并提交(分支 `goal/model-coverage-20260612`)。
- 验收 gate 定义(L0–L5 正确性 + 分类性能门槛)写入 GOAL.md。
- UNVERIFIED 落证(本地 4 项,全部完成):
  - #1 YaRN:不支持(仅 Llama3 变体);发现 max_seq_len 不 clamp 的隐患,
    已追加为 W1 公共工程项。
  - #2 AWQ:无 loader,纯 Future 注释;维持 defer。
  - #3 gguf arch 白名单:`qwen3|qwen3moe|qwen2|qwen|llama|mistral`,
    W1 够用,GLM(W2)需新增。
  - #8(新增)模板渲染失败静默 fallback 实锤(`chat_template.rs:226/488`),
    待 T3 消灭。
- UNVERIFIED #4/#5/#7(GPTQ group size / GLM config / Qwen3.6 官方 GPTQ)
  由后台 web 核查进行中。
- 任务分解:12 个任务建于会话任务系统(T1–T12),T1 完成。

### 下一步

- T3:模板引擎改造(minijinja pycompat 路线 + 渲染失败显式报错 + 防回归测试)。
- T4:EOS/BOS generation_config 审计。
- T5:L0 golden 测试基建(需本机 Python transformers 生成 fixture)。

### 阻塞项(预先声明)

- CUDA 侧 gate(L2-GPTQ / L5 / C7 回归)需要 4090 pod:开 pod 前按 GOAL
  执行合同填表并征得用户预算批准(CLAUDE.md 要求)。当前无可用 pod
  (上一台 38237968 已失;见 memory)。本地(Metal/CPU)可推进项先行。
