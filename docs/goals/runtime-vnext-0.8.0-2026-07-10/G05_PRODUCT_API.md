# G05: 统一产品组合根与 API 语义

## 状态与依赖

- 状态：Open
- 依赖：G02、G04
- 下游：G06、G08-G10

## 目标

让 `ferrum run` 和 `ferrum serve` 使用同一个 `ResolvedModelPlan`、同一个 engine 和同一套
token/template/stop/sampling semantics。入口差异只允许存在于 terminal、HTTP 和 SSE I/O。

## ResolvedModelPlan

必须包含并可保存：

- original source、resolved local source、model revision/file hash；
- architecture/family provider、weight format、tokenizer、chat template；
- EOS/BOS/pad/special ids；
- backend/device/provider capabilities；
- typed runtime preset、memory/admission plan；
- selected execution plan/hash；
- sampling、stop、structured-output policy；
- 每项值的 source 和 decision reason。

禁止通过 materialize hidden env 把 plan 再翻译成第二套运行时真相。兼容环境变量只能在
最外层解析为 typed config，并在 artifact 中标明 deprecated source。

## 产品语义

- run/serve 模型解析和自动下载策略统一。
- template 使用模型 metadata；渲染失败 hard error。
- Unicode 增量解码共享。
- EOS/stop/max_tokens/context/cancel 共享。
- stream/non-stream 共享同一 token/result accumulator。
- tools、tool_choice、tool result、structured output 使用同一 server contract。
- `tool_choice=required` 与 strict `response_format` 同时出现时，v0.8.0 的确定语义是 tool priority：
  返回 `finish_reason=tool_calls`，function arguments 按 tool schema valid，assistant content 不得伪造
  response-format JSON；相同 typed request 不得在 tool/content 两条路径间随机切换。standalone strict
  response format 仍按其 schema contract 成功。
- 扩展 G00 已加入的 typed bool `--enable-thinking`，增加显式 `model-default` 三态；它必须序列化为
  `chat_template_kwargs.enable_thinking`（`model-default` 时省略字段），并与现有 `run`
  thinking flags、serve request 共用同一 typed enum/validation，禁止靠进程环境变量补齐。
- unsupported image/vision 明确 4xx。
- OpenAI errors、usage、finish_reason、`[DONE]` 有唯一实现。

v0.8.0 将当前 best-effort `json_object` 提升为硬产品合同：它与 strict `json_schema` 共用
constrained-decoding/validation 基础设施，前者至少保证单一合法 JSON object、无 fence/前后
文本，后者再执行 schema 约束。达到 token/context limit 前无法生成合法结果必须返回明确
error/finish metadata，不能返回 200 + 非 JSON。同步更新 `docs/openai-api-compatibility.md` 和
migration guide。

G00 必须把 legacy `json_object` 的实际结果记录为锁定的 pass/known-fail observation，而不是
用本段未来合同判定冻结二进制。G05 完成本合同后，G08/G10 对三个主模型执行 C15 全量硬门；
届时不继承任何 G00 known-fail 豁免。

## 验收

- 同输入下 run/serve `ResolvedModelPlan` 去除 I/O 字段后的 hash 一致率 `100/100`。
- source/config/capability/template/preset decision 的生产实现各 `1` 处。
- RunCommand/ServeCommand 重叠业务字段由共享 typed options 表达 `100%`。
- 产品核心依赖 CLI helper 的反向 import 数 `0`。
- 核心 runtime 直接 `std::env` read 数 `0`。
- 使用 G01 tiny runtime 和每后端至少一个 G00 既有 actual-model legacy adapter，证明真实
  `run`/`serve` 通过同一个 composition root；main-model vNext 六 lane 由 G08 验收。
- sentinel stream/non-stream exact equivalence `20/20/backend`。
- required/auto/streamed tool 与 strict schema 的 engine-level sentinel 全部通过；三模型
  全矩阵归 G08。
- engine-level required-tool+strict-response-format sentinel `20/20` 重放选择 tool priority，
  name/arguments schema valid、schema-content fabrication `0`；standalone strict-schema sentinel
  `20/20` 成功。G08 C21 再按每 model/backend 两个相关子组各 `4/4` 实模验证。
- malformed SSE、missing/duplicate DONE、usage mismatch、UTF-8 replacement 均为 `0`。
- invalid requests 4xx `30/30`，不得 panic 或 200 silently ignored。
- `run` slow-path config materialization historical mutation 被 G02 gate 杀死。
- `bench-serve --enable-thinking false/true/model-default` payload golden 各 `20/20`，server
  捕获值与 effective config 完全一致；非法值在发请求前失败。

G05 不得打印任何主模型迁移 PASS。它只完成产品组合与 API contract；G08 才把该 contract
应用到三个真实模型和两个 backend。

## 兼容性

保持已发布 CLI flag 和 OpenAI-compatible request shape，除非 migration guide 明确列为
v0.8.0 breaking change。废弃 hidden env 必须提供 typed CLI/config replacement，并至少保留
一个版本的明确 warning；release evidence 不允许使用 deprecated env。

## 产物与 PASS

```text
docs/release/runtime-vnext/0.8.0/g05-product-api/
  resolved-plan-pairs/
  api-conformance/
  run-serve-matrix/
  unicode-streaming/
  tool-schema/
  deprecated-config.json
```

```text
FERRUM RUNTIME VNEXT G05 PRODUCT API PASS: <out_dir>
FERRUM GATE vnext-g05 PASS: <out_dir>
```
