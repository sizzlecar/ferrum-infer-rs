# Runtime vNext 模型与场景矩阵

## 1. 目的

本文是 Runtime vNext 三个主模型、格式、硬件、正确性场景和性能 workload 的唯一文字
来源。实现阶段必须增加机器可读 manifest，并由 validator 检查本文、manifest、命令和
artifact 一致。

热度数据是 2026-07-10 的 Hugging Face 近 30 天快照，只用于解释模型选择，不作为
Goal PASS 条件。

## 2. 主模型选择

| ID | 官方模型 | 热度快照 | 为什么进入主矩阵 |
|---|---|---:|---|
| M1 | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 约 778 万下载 / 720 likes | 新一代 dense hybrid；体积适合高频 CUDA/Metal gate；覆盖 DeltaNet、full attention、recurrent state 和 dense FFN |
| M2 | [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 约 226 万下载 / 1461 likes | 战略主模型；35B/3B active、hybrid recurrent、256 experts、shared expert，是最困难资源和并发路径 |
| M3 | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 约 308 万下载 / 905 likes | 成熟传统 MoE 控制组；现有 Metal/CUDA release 证据最完整，可证明重构没有破坏稳定路径 |

M1/M2 官方模型包含 vision 能力，但本 Goal 只认证 language path。vision encoder、图片和
视频推理另立目标；当前 API 收到多模态内容时必须显式拒绝，不能忽略 image token。

## 3. 精确 lane

所有正式 artifact 必须固定 revision 和文件 SHA256。下面的 repo/file 是目标输入；G00
负责锁定具体 revision 与 SHA256。

| 模型 | CUDA lane | Metal lane | 当前事实 |
|---|---|---|---|
| M1 Qwen3.5-4B | `Qwen/Qwen3.5-4B`, BF16 safetensors | `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-Q4_K_M.gguf` | 代码声明 Qwen3.5 CUDA product path；缺少正式实模产品/perf artifact；Metal 当前明确 unsupported |
| M2 Qwen3.5-35B-A3B | [官方 GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) | `unsloth/Qwen3.5-35B-A3B-GGUF`, `Qwen3.5-35B-A3B-Q4_K_S.gguf` | CUDA 有大量 correctness/L5 diagnostic，但没有 W3 release PASS；Metal 当前 unsupported |
| M3 Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | `Qwen/Qwen3-30B-A3B-GGUF`, `Qwen3-30B-A3B-Q4_K_M.gguf` | CUDA 有 0.7.7 release 锚点；冻结 cff4 Metal 在 32 GiB M1 Max 上有 reviewed resource blocker，vNext 最终双端目标不变 |

G00 固定 M1 GGUF revision `e87f176479d0855a907a41277aca2f8ee7a09523`、M2 GGUF revision
`bc014a17be43adabd7066b7a86075ff935c6a4e2`，并通过 Hugging Face revision/file probe 从该
commit 取得文件 SHA256。upstream `main` 后续移动不改变本 Goal；固定 commit/file 不可获取或
SHA 不一致时 hard fail 并走正式 Goal amendment，禁止自动挑选“最接近”的 quant。

M2 Metal 根据实际可用机器固定为 Q4_K_S：`20,673,845,888` bytes，约
`20.67GB / 19.25GiB`，LFS SHA256
`ee93ceffed5ce4df8b09bcbaf59a286d531025a1ebde9cf204c74e800c47d57e`。正式 Metal
reference hardware 是本机 `32GB / 24-GPU-core Apple M1 Max`；选择 Q4_K_S 只调整
Metal weight format，不减少 M2 架构、正确性、工具、结构化输出或性能场景。启动与每个测量
cell 必须保留至少 `2 GiB` 实测物理 headroom 且 swap growth 为 `0`，否则 hard fail 并重新
评估 typed admission/KV budget，不能 waiver。机器可读 model catalog 必须同时锁定上述 size 和
`expected_sha256`，resolver 必须证明它与 immutable revision 的 Hugging Face LFS OID 完全相等。

若独立 preflight 证明指定硬件无法满足本文的物理 headroom 或 active-concurrency floor，只允许通过
reviewed Goal amendment 更换明确命名的 reference hardware 或 weight format。amendment 必须保留
C01-C21、工具/流式/结构化输出和 G09 竞争性外部门，重采 hardware/model lock、legacy/external baseline
及所有受影响 artifact；不得在同一次 amendment 中静默降低并发、正确性或性能门槛。硬件阻塞在
amendment 合并前保持 `BLOCKED`，不构成 waiver 或 PASS。

## 4. 补充 lane

| Lane | CUDA 输入 | Metal 输入 | 最低要求 |
|---|---|---|---|
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq` | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`, `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` | 官方 Qwen config/tokenizer/template；首 token EOS/empty-answer bug-kill、required/auto tool、streamed tool delta、tool result、strict schema；不占三模型性能总分 |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | official BF16 safetensors | `unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF`, `DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf` | 必须使用 DeepSeek repo 自身 config/tokenizer/template/EOS；run/serve、stream、reasoning history；CUDA/Metal smoke |
| `meta-llama/Llama-3.1-8B-Instruct` compatibility | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`, `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`，tokenizer `unsloth/Meta-Llama-3.1-8B-Instruct` | 只为仓库 accelerator release policy 提供非 Qwen dense control；现有 CUDA/Metal run/serve/perf，不能替代 M1-M3 |

G00 对上述每个 repo/file 冻结 revision 和 SHA256。社区 quant 只提供 weight artifact；语义
config/tokenizer/template 必须来自表中明确的官方 upstream，禁止因架构相似改用 Qwen3 或
Llama 默认配置。

## 5. 模型结构覆盖

| 能力 | M1 | M2 | M3 |
|---|:---:|:---:|:---:|
| Dense FFN | yes | no | no |
| Routed MoE | no | 256 experts, top-8 | 128 experts, top-8 |
| Shared expert | no | yes | no |
| Gated DeltaNet / linear attention | yes | yes | no |
| Full attention | 每 4 层 | 每 4 层 | 每层 |
| Recurrent state | yes | yes | no |
| Paged KV | full-attention 层 | full-attention 层 | 全层 |
| BF16 safetensors | CUDA main | reference only | reference/optional |
| GPTQ/Marlin | optional | CUDA main | CUDA main |
| GGUF quant | Metal main | Metal main | Metal main |
| Thinking/final content 分离 | default thinking + hard switch | default thinking + hard switch | hard + soft switch |
| 官方 tool-call behavior | required | required | required |
| Ferrum constrained `json_schema/json_object` | required | required | required |

## 6. 正确性场景

每一行必须在 M1-M3 x CUDA/Metal 的六个 lane 上执行，除非表中明确只属于某个模型。

### 6.1 生成 preset

G00 在 `models.lock.json` 为每个模型冻结以下 machine-readable preset。每个 preset 必须保存
model revision、generation-config/chat-template revision、`temperature`、`top_p`、`top_k`、
`min_p`、repetition penalty、seed、max tokens、stop/EOS 和 template kwargs；“使用官方推荐值”
但没有具体字段的 artifact 一律 REJECT。

| Preset | 定义 | 使用场景 |
|---|---|---|
| `P_DETERMINISTIC` | `temperature=0`、`seed=9271`、`enable_thinking=false`，其余字段显式锁定 | C02-C09、C16-C18；legacy/token/stream parity |
| `P_NO_THINKING` | 官方 non-thinking sampling、`enable_thinking=false`、`seed=9271` | C10-C15 的 non-thinking 组 |
| `P_THINKING` | 官方 thinking sampling、`enable_thinking=true`、`seed=9271` | C10-C15 的 thinking 组、C19 |
| `P_OFFICIAL_DEFAULT` | 不覆盖模型默认 thinking mode，逐字段冻结官方推荐 sampling、`seed=9271` | C21 |

C01/C20 不生成模型输出的 negative case 不应用 sampling；C20 的纯文本 content-array positive
case 使用 `P_DETERMINISTIC`。M3 的 soft `/think`、`/no_think` 只改变 C19 和明确标记的 tool
子组，不得改变其他 preset 的 hard `enable_thinking` 值。

下表是 vNext 在 G08/G10 的最终硬合同。G00 仍逐项运行可执行的 legacy case，但必须按
`runtime_vnext_legacy_correctness_expectations.json` 冻结为 `pass`、`known-fail` 或
`blocked`；未知事实先进入不产生 PASS 的 discovery 阶段。不得因为下表是最终标准，就把 G05
才新增的能力手填成 legacy PASS；也不得把 G00 的 known-fail 当作 G08/G10 waiver。

| ID | 场景 | 输入/次数 | 硬断言 |
|---|---|---|---|
| C01 | config/template golden | 20 fixtures：official config/template bytes/special-token+EOS/unknown+fail-closed 各 `5` | 每个 fixture 有冻结 input/expected bytes/hash；execution manifest 显式绑定 `models.lock.json` 的 `semantic_source_root` 和独立 `tokenizer_source_root`，实际 runtime `resolution_evidence` 与其 revision/path/SHA 逐项一致；官方 config 字段完整，template bytes 与官方 tokenizer 一致，special ids 精确；五个 materially distinct 的未知 architecture/layout fixture 都必须通过真实 `ferrum run` 在读取权重/选择 kernel 前 hard error，保存 argv/stdout/stderr/receipt，且未知字段不得触发 Llama fallback；普通缺文件或权重错误不能代替该失败类 |
| C02 | `ferrum run` 单轮 | 20 known-answer | 20/20；非空；自然 EOS；无保留 token/乱码 |
| C03 | `ferrum run` JSONL 多轮 | 10 组三轮 | 10/10 记忆正确；每轮 token/usage 完整；无 KV/state 串话 |
| C04 | `ferrum run` 交互长输出 | 3 组，每组 >=512 output tokens | 不 hang/crash；UTF-8 完整；增量输出；结束状态正确 |
| C05 | serve non-stream | 20 known-answer | 20/20；OpenAI response shape、finish_reason、usage 正确 |
| C06 | serve stream | 与 C05 相同 20 个 prompt id/payload hash | 每个 C06 case 引用对应 C05 artifact；拼接 content/reasoning、finish reason、usage 与 non-stream `20/20` 一致；一个 `[DONE]`、一个 usage、至少一个 content/tool delta |
| C07 | stateful loop | 6 个独立 conversation x 5 个顺序轮次 | 第 N 轮 payload 必须携带该 conversation 前 N-1 轮完整 user/assistant history，并含唯一 conversation/round nonce；30/30；无 prefix 重复、length 早停、跨 conversation 污染；重复同一 stateless payload 不计 |
| C08 | stop/EOS/max_tokens | 每类 20 | finish reason 正确；stop 不泄漏；EOS 来源为模型元数据；max-token 子组关闭 EOS/stop 或使用不会早停的 fixture，精确生成 N 个 usage-counted token、无第 N+1 个 token、`finish_reason=length`；natural-EOS 子组允许 `<N` |
| C09 | cancel/timeout/disconnect | 每类 20 | cancel 被 runtime 接收后 `<=2` 个 scheduler tick 且 wall `<=5s` 到达 released/expected-cache 终态；后续同容量请求成功；无 double release/leak |
| C10 | required tool | 每模型 `P_NO_THINKING`/`P_THINKING` 各 20；M3 soft mode 各 10 | `finish_reason=tool_calls`；name/arguments/schema 正确；reasoning 不污染 arguments |
| C11 | auto tool | 同 C10，强触发 prompt | 产生预期 tool；不得把 JSON 塞入 content |
| C12 | streamed tool delta | M1/M2 40：P_NO/P_THINK=`20/20`；M3 60：P_NO/P_THINK=`30/30`，其中 base/soft-think/soft-no-think=`40/10/10` | 每个 ordinal 与同 model/preset/ordinal 的 C11 auto-tool 配对，先 non-stream 后 stream；使用相同 payload，按 tool index/id 重组 name/arguments 后与 C11 deep-equal；arguments JSON/schema valid；一个 `[DONE]`、一个 usage；只统计 delta 数量不计 PASS |
| C13 | tool result continuation | 同 C10 | tool role/template 正确；最终回答引用 tool result；无重复 call/history 污染 |
| C14 | strict `json_schema` | 每模型 `P_NO_THINKING` 50：required/type/additionalProperties/enum=`13/13/12/12`；`P_THINKING` 20：四类各 `5` | non-thinking 50/50、thinking 20/20 valid；四类使用不同 schema、prompt 和预期值，不能用同一 payload 重复计数；reasoning 与最终 JSON 分离 |
| C15 | `json_object` | 每模型 `P_NO_THINKING` 50、`P_THINKING` 20 | non-thinking 50/50、thinking 20/20 可解析对象；无 markdown fence 和前后垃圾；reasoning 不进入 JSON |
| C16 | negative API | 30：非法 tool/schema/stream option/model/context 各 `6` | 五类分别 `6/6` OpenAI-shaped 4xx；任一类不得借另一类的重复请求补足总数 |
| C17 | UTF-8 | 中文、emoji、组合字符各 20：每类 run incremental `10`、serve stream `10` | transport read 允许任意切开多字节字符；跨 read 增量组帧后，每个完整 SSE line/event 必须是合法 UTF-8，拼接结果与 expected bytes/hash 相等；U+FFFD/mojibake 为 0；non-stream response 不能替代 stream 子组 |
| C18 | concurrent isolation | CUDA client c=1/4/16/32；Metal client c=1/4/16 | 完成率 100%；串话/坏 checksum/错误/500/panic/OOM 均为 0；每 cell 保存 requested concurrency、typed admission cap、observed max-active 和 active duty-cycle；最高 client cell 必须达到本文件 8.1/8.2 的 backend/model active floor 且 eligible-interval duty-cycle `>=0.80`，不能把 client c32 冒充 active c32 |
| C19 | thinking separation | 每模型 20；M1/M2：default/hard-on/hard-off/hard-off+`/think`/hard-on+`/no_think` 各 `4`；M3：default/hard-on/hard-off/soft-`/think`/soft-`/no_think` 各 `4` | M1/M2 默认 thinking，硬 off+`/think` 仍 off、硬 on+`/no_think` 仍 on；M3 同时验证硬/软切换；每组含 history-carrying multi-turn，reasoning/final/history 正确分离 |
| C20 | multimodal fail-closed | M1/M2/M3 每类 10 | image URL、data URL、video URL、mixed text+media 在 template 插入 vision token 前明确 4xx；纯 text content-array 成功；`/v1/models` 只声明 text modality |
| C21 | 官方采样 smoke | 每模型/backend 20：run plain/serve stream/required-tool+strict-response-format/standalone strict-schema/json_object 各 `4` | 全部使用锁定的 `P_OFFICIAL_DEFAULT`；五组分别 `4/4`，非垃圾、结构/终止/资源终态正确。required-tool 组同时发送 strict `response_format`，必须确定性选择 tool priority：`finish_reason=tool_calls`、name/arguments/schema valid、content 不得伪造 schema JSON；standalone strict-schema 组 `4/4` schema success；普通 smoke 不要求逐 token 相等 |

## 7. 数值与 parity

### 7.1 Legacy 已支持 lane

- 固定 20 个 temp=0 prompt。
- 同 backend、同 format、同模型 revision 下，legacy 与 vNext 前 64 个生成 token `20/20` 完全
  一致，PASS exception 数量 `0`。reference top-2 margin `<1e-3` 时保存 logits、margin 和高精度
  reference 作为 near-tie diagnostic；generated token flip 仍为失败，且不得影响 tools/schema/stop。
- plan decision、resource lifecycle 终态、selected op provider 和显式 fallback 必须可 diff。

### 7.2 新 Qwen3.5 Metal lane

- 官方 HF config、tokenizer、generation config 和 chat template 是语义/metadata 真值。
- 每类关键 op 有独立 CPU FP32 oracle；layer/full-model checkpoint 优先使用 Transformers
  reference，不能只用同一个 GGUF converter + llama.cpp 自证正确。
- 单层与全模型 checkpoint 对比保存 tensor SHA、shape、dtype、abs max、rel max、relative L2、
  cosine 和完整 tolerance id。shape/dtype 必须精确相等，NaN/Inf 数量为 `0`。
- 同权重 FP32 oracle 对 FP16/BF16 op/layer 默认要求 cosine `>=0.999`、relative L2 `<=1e-2`。
  量化 op 使用其 checked-in contract tolerance，且不得宽于现有 CUDA/GGUF conformance；provider
  不得在 artifact 中覆盖 tolerance。
- 同 GGUF、同 token history 的 Ferrum Metal 与独立 CPU reference/llama.cpp next-token full-vocab
  logits 必须达到 cosine `>=0.999`、relative L2 `<=1e-2`。固定 20 个 deterministic prompt 的前
  64 个生成 token 必须 `20/20` 完全一致，PASS exception 数量 `0`。reference top-2 margin
  `<1e-3` 时仍须保存两侧 logits、margin 和高精度裁决作为 near-tie diagnostic，但任何 generated
  token flip 仍使本门失败；diagnostic 不得改写序列相等结果。
- G03 在实现 provider 前必须提交
  `scripts/release/configs/runtime_vnext_numerical_tolerances.json`。每个 tolerance row 至少绑定
  `tolerance_id`、op/schema version、checkpoint kind、dtype、quant format、shape domain、oracle、
  cosine/relative-L2/absolute bound、source commit 和 owner。G08A/G08B artifact 必须引用该文件的
  Git blob SHA 与 row fingerprint；缺 row、post-hoc 放宽或 artifact 自带 tolerance 一律 REJECT。
- 全模型 known-answer、tool、schema 和 stream 全过后才允许性能测试。
- 同 GGUF 的 llama.cpp 只用于量化端到端 token/logit 交叉验证和性能基线，不单独充当官方
  语义真值。
- 社区 GGUF artifact 必须记录 upstream model/revision、converter/quantizer version、GGUF
  metadata、tensor inventory、文件 SHA256 和许可证。

### 7.3 Deterministic 与用户默认路径

- legacy token equality、stream/non-stream equality 使用明确的 `temperature=0` diagnostic
  设置；这是回归工具，不是 Qwen 官方推荐的 thinking sampling。
- C21 使用官方推荐参数和固定 seed，验证真实用户默认路径。因为 sampling 和 backend 数值
  差异，不要求 legacy/vNext 逐 token 相等，但 correctness/termination/resource hard gate 不变。

## 8. 性能 workload

### 8.1 CUDA

- Hardware: exactly one RTX 4090, fixed host/driver/CUDA.
- Client cells: c=1/4/16/32。typed active admission 硬下限为 M1=`32`、M2=`16`、M3=`32`。
  最高 client cell 的 `observed_max_active` 必须达到该下限；在 warmup 后、仍有至少该下限数量
  outstanding request 的 eligible interval 内，active sequences `>=` 下限的 wall-time fraction
  必须 `>=0.80`。正式 external comparison 使用相同 active cap，并保存完整 scheduler active
  timeline；只记录瞬时 max 不构成 duty-cycle PASS。typed cap 等于 floor 时，
  `observed_max_active == typed_admission_cap`；cap 高于 floor 时 observed max-active 至少达到 floor。
  cap/floor 关系只能通过本文件前述 reviewed Goal amendment 修改。
- Synthetic: random 256 input / 128 output tokens, `enable_thinking=false`、`ignore_eos=true`。
- Realistic: frozen ShareGPT JSONL, tokenizer-counted input，128 output cap，
  `enable_thinking=false`、`ignore_eos=false`。
- Warmup: 10 requests before each measured repeat.
- Both `serve` throughput and `run` single-request decode are required.

### 8.2 Metal

- Hardware: fixed `32GB / 24-GPU-core Apple M1 Max`; M2 uses the pinned Q4_K_S file.
- Client cells: c=1/4/16；typed active admission 硬下限为 M1=`16`、M2=`4`、M3=`16`。
  c16 cell 的 `observed_max_active` 必须达到对应下限；在 warmup 后、仍有至少该下限数量
  outstanding request 的 eligible interval 内，active sequences `>=` 下限的 wall-time fraction
  必须 `>=0.80`。client c32 仅作 admission stress，必须记录 typed cap、observed max-active 和
  active timeline，不计入 required performance cells。typed cap 等于 floor 时，
  `observed_max_active == typed_admission_cap`；cap 高于 floor 时 observed max-active 至少达到 floor。
  cap/floor 关系只能通过 reviewed Goal amendment 修改。
- Workload: random 64 input / 128 output plus fixed real-chat dataset；thinking/EOS 规则与 CUDA 一致。
- Start/end thermal、power mode、memory、swap and process high-water must be saved.
- No active swap growth during measured cells; otherwise artifact is REJECT.

### 8.3 统一统计

```text
ferrum bench-serve ... \
  --fail-on-error \
  --require-ci \
  --seed 9271 \
  --enable-thinking false \
  --num-prompts 100 \
  --warmup-requests 10 \
  --n-repeats 3
```

- output token count source 必须是 usage。
- 每个 `comparison_id` 独立执行一套 `ABBA-BAAB`；同一套 outer sessions 可以按固定顺序覆盖
  该 comparison 的完整 cell matrix，但每个 `(model, backend, dataset, concurrency,
  comparison_id)` 独立统计。不同 comparator 的 slot、repeat、A rows 和 CI 不能混用或拼接。
  required comparison 如下：

| comparison_id | A | B | 适用范围 |
|---|---|---|---|
| `serve-legacy` | G00 冻结 legacy server | vNext server | G00 legacy PASS/comparable cell |
| `serve-external` | same-host vLLM 或 llama.cpp | vNext server | 全部主模型 required HTTP cells |
| `run-legacy` | G00 冻结 legacy `ferrum run` | vNext `ferrum run` | G00 legacy PASS/comparable run lane |
| `run-vs-serve` | vNext serve-c1 | 同 binary vNext `ferrum run` | 全部主模型/backend |

- 每个 HTTP comparison 的 slot 都运行上面的 3 个 inner repeats。因此每实现每
  `(cell, comparison_id)` 有 4 个 outer slots、12 个 repeat samples、`1200/1200` measured
  requests，另有 `120` 个 warmup requests；warmup 不进入性能统计，但完成率也必须 `100%`、
  error/bad-output `0`。
- 一个 HTTP outer slot 是一个
  model/backend server session：只加载模型一次，再按 manifest 的固定 cell 顺序跑完整 required
  matrix；下一 outer slot 才重启 server。每个 inner repeat 先完成 10 个 warmup；model load 不计入
  request latency，但启动/load time 单独保存。八个 slot 的最近邻配对固定为 `(1,2)`、`(4,3)`、`(6,5)`、
  `(7,8)`，得到 4 个顺序交替的 A/B pair cluster。
- throughput/run-decode ratio 定义为 `B/A`；latency/memory ratio 定义为 `B/A`（越低越好）。
  每个 pair 内按 inner-repeat ordinal 配对，产生 3 个 ratio；point estimate 是 12 个 paired
  ratios 的 median，同时另算 `median(B_12)/median(A_12)` 并执行文档中的绝对中位数门。
- 95% CI 使用固定 seed 的 100,000 次 hierarchical paired bootstrap：先有放回抽样 4 个
  pair cluster，再在每个抽中的 cluster 内有放回抽样 3 个 paired repeat；单 request 永远不是
  统计样本。validator 保存 bootstrap seed/code SHA 和完整 paired ratios。
- throughput 几何平均按 `comparison_id` 分别覆盖该 model/backend 下全部 required dataset x
  concurrency cell 的 point ratios；不同 comparison 不合并，任一单 cell 中位数门失败不能被
  其他 cell 抵消。
- 保存 throughput、TTFT、TPOT、E2E、request throughput、goodput、peak memory，以及逐 request
  client-SSE-event ITL provenance/eligibility；G06 engine token-commit ITL 以独立 source 保存。
- 正式比较使用 ratio bootstrap 95% CI；不能只比较单次均值。
- client-SSE-event ITL 只有 paired A/B 的全部 request 均 eligible 时才产生 repeat/cell ratio；任一
  request ineligible 时不得从其余 eligible subset 计算。transport chunk coalescing 只影响 ITL
  eligibility，不得归因 server bulk flush 或改写 request correctness。

`ferrum run` 使用同一 frozen prompt/tokenizer、`enable_thinking=false`、128 output token cap 和
明确 EOS policy。G00 legacy one-shot 每个 slot 运行 3 个独立冷进程首请求，共 12 个真实命令；
唯一正式指标是 JSONL assistant 事件暴露的完整 `engine.infer` E2E：
`generated_tokens * 1000 / assistant.ms`。它包含 prefill、decode、sampling 和 text，排除 model
load/shutdown；另存进程 wall time但不得混算，也不得伪称 TTFT/steady decode/chunk timing。

G06 后 vNext 才以同一 monotonic clock 保存 first/last token commit，从而计算 TTFT 和
steady-decode。`run-legacy` 同时保留上述 E2E 边界与 G00 比较；`run-vs-serve` 将同一 vNext
binary 的 run 作为 `B`、serve-c1 作为 `A`，两侧使用 G06 token-commit steady-decode tok/s，
不能拿 serve 总请求吞吐与 run E2E 或 decode 混比；方向仍为 `B/A`。

### 8.4 外部引擎公平性

- vLLM 固定 version/commit、container/binary SHA、build flags、CUDA/driver；Qwen3.5 使用
  `--language-model-only`，M2 GPTQ 固定 `moe_wna16`。
- vLLM/Ferrum 的 MTP、speculative decode、prefix cache、max model len、stream、EOS、
  scheduler/admission、KV budget 和 thinking 设置必须成对相同或同时关闭。
- llama.cpp 固定 commit、binary SHA、Metal build flags、threads、GPU layers、context 和 batch。
- Ferrum、vLLM、llama.cpp 全部由同一个 `ferrum bench-serve` 发压，不使用对方自带客户端
  生成正式比较数字。
- 同一个 dataset 经各自 tokenizer 后的实际输入 token 均需保存；平均长度差异必须 `<=1%`，
  否则该比较 REJECT。

## 9. 性能门槛

| 类型 | 门槛 |
|---|---|
| legacy non-regression | 每 cell candidate throughput 中位数 `>=legacy`，ratio LCB `>=0.97`；cell 几何平均 `>=1.00` |
| latency non-regression | TTFT/TPOT candidate 中位数 `<=legacy`，ratio UCB `<=1.05`；client-SSE-event ITL 仅在全 paired request eligible 时执行同一门；G06 token-commit ITL 独立执行 no-regression |
| memory non-regression | peak memory `<=1.03x` legacy；leak/swap growth/OOM `0` |
| new-lane memory | CUDA peak `<=` typed preflight budget 且保留 `>=512 MiB` physical headroom；Metal peak 不越过 typed unified-memory budget、measured cell swap growth `0` |
| CUDA external | 三模型每 cell throughput LCB `>=0.90x` same-host vLLM，required-cell ratio 几何平均 LCB `>=0.95` |
| CUDA external latency | 三模型所有 new/BLOCKED cell 的 TTFT/TPOT p95 `<=1.15x` same-host vLLM，paired latency ratio UCB `<=1.15`；client-SSE-event ITL 仅在全 paired request eligible 时执行同一门，否则 ratio 不得存在；legacy PASS cell 仍执行更严格门，G06 token-commit ITL 另行必采 |
| Metal external | 三模型 c1/c4/c16 throughput LCB `>=0.90x` same-host llama.cpp、required-cell ratio 几何平均 LCB `>=0.95`，同 GGUF；TTFT/TPOT p95 和 eligible client-SSE-event ITL `<=1.15x` |
| product run/serve | G06+ 同 binary、model、prompt、prefill/output 设置下，`ferrum run` token-commit 稳态 decode / Ferrum serve-c1 decode median `>=0.95`；legacy PASS lane 另以匹配的 `engine.infer` E2E 边界做 no-regression |
| reliability | 每 required HTTP comparison 的每实现每 cell `1200/1200` measured requests completed；run comparison `12/12`；全部 warmup/error counters `0` |

已保存的参考数字全部是 stale historical evidence，只为 G00 提供审计线索：

- M2 Qwen3.5-35B CUDA Ferrum
  [artifact](../model-coverage-2026-06-12/artifacts/w3_qwen35_default_full_l5_fixed_output_20260623_39ffe5db/perf/bench_ferrum_sharegpt_sweep_100x3.json)
  （SHA `39ffe5db`、driver 570、1x4090 `24564 MiB`）：约
  `85.1/263.7/622.7/627.2 tok/s`；requested c32 实际 active admission cap 为 c16，不能
  视为真实 active-c32 baseline。
- 历史 vLLM ShareGPT
  [artifact](../model-coverage-2026-06-12/artifacts/w3_vllm_sharegpt_baseline_20260619/bench_vllm_sharegpt_sweep_100x3.json)
  来自另一台 driver 580 主机，并报告异常的 `49140 MiB` visible VRAM，active cap 也未可靠
  锁定：`136.1/405.4/1190.7/1708.5 tok/s`。它与上一行不是 same-host A/B，不能进入 validator。
- M3 Qwen3-30B CUDA v0.7.7
  [artifact](../../release/g0/0.7.7/cuda-full/summary.json) 位于 SHA
  `22b8b7ada3fd586f018a95ba7c1d550d1c57001e`：`164.2/353.3/636.9/706.0 tok/s`；历史
  [FA2 direct artifact](../../bench/cuda-rtx4090-2026-05-30-m3-80pct-confirmed/README.md)
  为 `160.4/446.3/1185.1/1641.9 tok/s`。二者不是当前 `cff4c477` legacy，也不能拼接。
- M3 Metal v0.7.7
  [artifact](../../release/g0/0.7.7/metal/metal-readme/summary.md) 同样位于 SHA `22b8b7ad`；
  旧 `16 input / 64 output` c16 约 `68.5 tok/s`，与本 Goal `64/128` 不同，只能作 sanity
  reference，不能直接成为硬 floor。冻结 `cff4` 在固定 32 GiB 主机上的 fresh legacy lane 按
  [`G00_M3_METAL_32GB_AMENDMENT.md`](G00_M3_METAL_32GB_AMENDMENT.md) 标记 resource-blocked，
  不生成 legacy ratio；G08C/G09 仍在同一主机验证 vNext Metal。

M1 没有可信现有实模性能 artifact。G00 必须先尝试冻结 legacy CUDA 基线；如果 legacy
产品路径失败，保存 BLOCKED artifact。vNext 仍必须通过完整正确性并达到 `>=0.90x vLLM`，
TTFT/TPOT p95 `<=1.15x vLLM`；全 paired request eligible 时 client-SSE-event ITL 也执行该门；
`ferrum run` 稳态 decode 不低于同 binary Ferrum
serve-c1 的 `0.95x`。不得把“旧版没有数字”解释为没有性能要求。

## 10. Artifact 最低字段

每个模型/backend artifact 至少包含：

- goal/model/lane/scenario id；
- model repo、revision、文件清单和 SHA256；
- git SHA、dirty status、binary SHA256、Cargo features；
- hardware、OS、driver、CUDA/Metal capability、memory；
- 完整 server/run/bench 命令和 typed effective config；
- dataset id/SHA、seed、tokenizer id/SHA；
- correctness counts、bad-output scan、resource final state；
- performance raw repeats、aggregate、CI 和 baseline ratio；
- `itl_evidence_per_request`、逐 repeat eligibility counts、expected/observed interval totals、ITL
  source 与 ratio eligibility；ineligible 时 ratio 字段必须不存在；
- typed admission cap、raw active timeline、observed max-active、eligible interval、active duty-cycle；
- 每个 capacity domain 的 arena/block/slot unit、absolute total/effective/free capacity、reserve/
  watermark、block/cache dtype，以及 immediate claim、fit requirement、实际 future reservation、
  defer/impossible/release epoch、preemption/recompute timeline；Ferrum/vLLM 比较必须锁定并报告
  相同的绝对有效 KV budget；
- numerical checkpoint 的 tolerance catalog blob SHA、tolerance id/row fingerprint 和 raw tensor/logit metrics；
- stdout/stderr 非空日志；
- PASS/FAIL/REJECT/BLOCKED 与精确 failure class。

缺字段、手填结果、跨 SHA 拼接或 artifact stale 时不得进入最终矩阵。
