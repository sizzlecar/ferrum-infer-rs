# 有界输出预算与所有权契约

本模块提供由真实请求和 tokenizer 能力导出的输出计划、有界编码器、move-only frame/session 类型与预算。接口定义 `LlmInferenceEngine::infer_credited_stream`；真实 engine output actor、server lease adapter 与 HTTP/CLI 消费实现在各自 crate，接口测试本身不替代其生命周期验证。只有使用同一套编码器，并让 projection grant 覆盖全部声明的保留存储，才能使用这里的预算结论。

## 已支持的协议边界

固定构造器支持 CLI 原样 UTF-8 文本、OpenAI Completions SSE，以及普通流式 Text Chat SSE。Chat 必须对应真实 Chat 请求，具有已解析的模板 reasoning 状态；include_usage 必须与请求一致。`BoundedChatProjection` 复用 Text reasoning parser，在有界 raw/scratch 中分别投影 reasoning 和 content，可能暂缓不完整标记；一次 raw command 最多产生两个语义帧，必须处理完当前 cursor 才能接受下一条。它不拥有模型或 token frontier。

带 tools、legacy functions、指定工具调用或结构化 response format 的 Chat 不受此契约支持；无工具的 auto/none tool_choice 可通过请求校验。Responses、JSONL、Harmony 等 native protocol、任意工具/schema 输出没有对应预算契约。显式 prompt-token/engine-token-timing evidence 使用下述独立 retained-storage 子合同。CLI 原样文本不执行 reasoning/native-protocol 转换；不能用它的预算冒充 Chat 语义投影、CLI 装饰或统计输出。

延迟 completion 使用真实 atomic token 或 tokenizer 不可变 prepared token sequence，按精确 marker 长度预留匹配存储，并在取得存储后复验。它还要求 tokenizer 声明 bounded incremental equivalence。Chat 只接受 prompt 已打开 reasoning 且没有 alternate envelope 的延迟边界；Completions 不接受该延迟投影。CLI 可以保留 alternate envelope 原文，但这不表示已经实现工具语义。

## 字节、事件与终态预算

`RequestOutputPlan::derive` 要求 tokenizer 同时声明 bounded decoder（完整文本及 caller-owned workspace）和 bounded context-free token bytes；只有 decoded_text_bound 不足以准入。使用请求的有效 max_tokens，不再截短。对 N 个输出 token，按 N 条 raw command 加一次末尾 UTF-8/stop flush 计算生命周期帧 envelope；Chat 每条最多两帧，累计语义字节另按 Text reasoning parser 声明的投影 epoch 上限计算。JSON 最坏转义及固定 id/model envelope 由实际序列化器计数，调用方不能自报展开系数。

`RequestOutputBudget::open` 在公布 owner 前预付整个生命周期的 data wire reservoir、projection 高水位和固定 terminal escrow。普通单帧使用方式逐帧取得一个真实 data event；`reserve_future_event_window` 还可预留有界的真实未来 event escrow。失败或竞争不能产生额度。CLI 的最低事件容量为一个 data event 加一个 terminal event；两种 SSE codec 的 terminal 成功包包含 finish、可选 usage 和 DONE，因此预留两个或三个 terminal events。生命周期最大帧数不等于一次预留 N 个队列位置。

真实编码前必须取得 frame permit 和字节所有权。编码后精确输出额分给帧，未用字节移回 reservoir，全程不释放后再申请。消费者释放帧会归还 pool 额度，但不会增加该请求未花掉的生命周期字节。未提交 permit 可原样归还；直接 Drop 放弃其 reservoir，不能伪造后续可用帧。

固定 terminal 字节/event escrow 不被 data 输出消耗；大段末尾文本仍必须先经 data frame 投影。成功终态按真实 prompt/max-output usage、最长 finish reason 和最大时间戳估算；失败包只保留最多 512 字节且保持 UTF-8 边界，wire buffer 与错误字符串分别计账。CLI 成功终态没有额外文本。此接口没有 transport queue，终态能在满队列中发布还需要后续 actor 另保留真实 terminal wire slot，不能仅由 budget 推出。

`PrepaidOutputCapacityView` 只读、不保有执行权限。它验证同 account 的实际 Data frame permit，并把剩余 token command、真实 event escrow 和 owner 提供的可用 wire slots 共同限制为 no-drain 窗口。Chat 窗口考虑每 command 最多两帧，最后一条已接受 command 可停在有界 pending projection；该数字不保证全部文本或 terminal 已交付。planner adapter 仅复制数值，模拟时扣减命令数，不重复扣已预付字节，也不假设消费者释放。发布该 view 的 actor 必须同时持有对应 Ready permit；真正提交仍需取得权限并复验。

## 保留内存与后续接线

projection 基础高水位包含四份最大文本（decoded output、engine pending、consumer history、consumer transform scratch）、三份 TokenId 历史、三份 stop 字符串、固定协议 id/model 和 bounded error。Chat 另计各一份 raw/scratch arena；延迟 incremental 检查另计两份 decoded text 与一份 token history，marker 匹配存储单独计入。decoder workspace 与 UTF-8 校验 workspace 取较大值，因为这两个阶段不重叠；后者为两份最大 token 原始字节加旧/新 UTF-8 片段的固定上限。派生预算的容量运算检查溢出，不允许退回隐含临时分配或不受该契约约束的 decoder cache。

消费者保留的 history 可在终态后连同 projection grant 转移；最后 owner 释放前账户和预算仍留账。后续 engine actor 必须保证 incarnation/frontier、无重复提交、命令物理 slot、投影顺序及 cancel/terminal 生命周期；server 字节 owner 与 CLI writer flush 也必须持有对应 lease。本 checkpoint 的契约测试不证明这些产品接线已经完成，也不能据此解除 Enforce 启动保护。

## Bounded opt-in execution evidence

Ordinary credited text output can retain requested prompt/output token IDs and
complete engine token-commit timing independently of its SSE/stdout frames.
The engine derives an `EngineEvidenceRetentionPlan` after model-context checks
and reserves its bytes in both existing per-request and global projection credit
before admitting the request. Timing disabled retains no added arrays. Timing
always keeps every commit up to the resolved output limit; decode-stage attempts
retain at most the first `3 * max_tokens` intervals and explicitly count omitted
intervals (saturating at u64::MAX). This stage prefix is never a full-trace claim.
Native Kernel first-N capture remains a separate policy and lifetime contract.

Successful terminal evidence moves with `LeasedOutput<OutputCompletion>`;
cancellation/error destroys unused evidence while its original grant is live.
External consumers retain that lease through reporting. The profile serializer
borrows token/stage arrays, and the owned JSONL record path writes directly into
the existing fixed-size journal writer without an encoded payload Vec. The
completion and prompt artifact may share an Arc over the same lease; credit is
released only when its last owner finishes. Output wire credit never pays for
these histories. Actual Vec capacities and complete commit counts are checked
before terminal handoff.

This covers the existing ordinary text and reasoning streaming contracts plus
one-shot text `run`. It does not add tool/structured output, interactive run,
non-streaming HTTP, or full replay/verification protocols. Observed latency is
engine-commit timing; socket delivery/visible SSE ITL remains a separate measure.
