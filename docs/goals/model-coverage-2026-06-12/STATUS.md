# STATUS — model-coverage-2026-06-12

进度日志,倒序。

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
