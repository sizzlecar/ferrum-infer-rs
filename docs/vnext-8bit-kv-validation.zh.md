# vNext 8-bit KV 验证记录

关联：[开发设计](vnext-8bit-kv-development.zh.md)。本记录区分合约、设备执行、模型质量与性能；前两项通过不代表后两项通过。当前尚未取得发布资格。

## 1. 验证环境与选样

2026-09-17 本机环境为 MacBookPro18,4、Apple M1 Max、32 GiB 统一内存、macOS 15.1.1（24B91）、Rust 1.91.0。清理旧临时代码索引约 19 GiB 后，可用磁盘从约 14 GiB 增至 33 GiB；源码、模型、已有测试记录均保留。Cargo 复用本机已有构建缓存。

模型按本次影响的状态结构选样，不能将一个模型的结果推广到所有模型：

| 模型及固定 revision | 权重输入 | 覆盖目的 |
| --- | --- | --- |
| Qwen/Qwen3-30B-A3B-GGUF，`e4d4bafdfb96a411a163846265362aceb0b9c63a`；语义/tokenizer 为 Qwen/Qwen3-30B-A3B，`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | 本地 Q4_K_M GGUF | 纯 causal + MoE，48 层、4 KV heads、head dimension 128；待本机验证 |
| Qwen/Qwen3.5-0.8B，`2fc06364715b967f1860aea9cf38778875588b17` | 本地 safetensors | recurrent + causal，24 层、每 4 层一个 full attention、2 KV heads、head dimension 256 |
| Qwen/Qwen3.5-2B，`15852e8c16360a2fea060d615a32b45270f8a8fc` | 本地 safetensors | 混合状态模型的指令、状态、结构化与工具任务校准 |

选择依据是 operation/state 覆盖和真实已注册的 vNext family。Qwen3-0.6B 初始校准显示它进入显式 legacy 路径，`kv_storage` 为 null，故从本项 vNext 证据中排除；其成功生成不算本功能通过。GGUF 权重路径另需真实模型验证，当前 Q4_K provider 的合成权重 continuation 测试不能替代它。30B GGUF 仅适合本机容量范围，不能直接拿去要求 6 GiB CUDA 设备运行。

按未分页的活跃 KV 计算，30B 纯 causal 样本每 token 从 98,304 字节变为 50,688 字节；0.8B 混合样本的 full-attention KV 每 token 从 12,288 字节变为 6,240 字节。此处是布局公式，不是进程内存实测；分页、checkpoint、recurrent、权重与 workspace 需单独计入。

0.8B 的 F16 校准中，run/serve basic 和实际 KV 选择验证通过；原 state 测试要求仅返回 `cobalt-731`，模型返回 `The code to remember is cobalt-731.`，run/serve state 均判失败。保留失败及原 oracle，不据此宣称 INT8 质量通过；更强指令任务改用 2B 做事前校准。

## 2. 配对方法与事前约束

所有配对使用同一设备、二进制、模型文件、模板、输入与 sampling；唯一主动变化为 `--kv-dtype fp16` / `--kv-dtype int8`。CUDA 同时记录实际 attention policy；INT8 仅 portable，F16 auto 选择 native 时属于必须声明的额外差异。

先跑 F16 校准，检查模型本身能完成语义任务、测量重复运行噪声；随后冻结最终评估输入和接受预算，再执行 INT8 最终评估。不能看过候选结果后放宽门槛。校准失败要修正模型/任务适配或记录不适用，不能算作 INT8 通过。

质量覆盖：

- 固定 token 历史的 teacher-forced 配对：保留相同的真实 token 序列和原始 full logits，记录目标 token NLL、ΔNLL、KL(F16‖INT8)，按历史长度分别报告。续写分叉后的 logits 不做直接配对。
- 事实提取：由确定性输入给出键值事实，要求取回指定字段；短上下文与跨多个 KV 页的长上下文分别检查。
- 严格 JSON：验证 schema 和字段事实，不能只检查 JSON 可解析。
- 工具调用与续答：验证请求参数、注入的工具返回值及最终回答引用该返回值；不以 HTTP 200 判定内容正确。
- 相同前缀恢复及追加：确认实际恢复 tokens、与冷算的状态/输出对照，并保留失败原因。

性能覆盖同负载与同预算两组：先冻结相同输入/输出 token 数、并发与 prefill chunk 配置，再测 prefill/TTFT、decode、端到端延迟及实际容量。性能测量使用优化构建并避开本机同时编译/CI 的干扰；GPU 单元测试的运行时长不作为推理性能。

最终语料、质量预算、性能预算及 F16 校准结果尚待填写；在其冻结并验证之前，不宣称模型质量或性能达标。

## 3. 已实际执行的检查

以下为本机运行结果，代码里程碑包括 `c414a179`、`8fddf851`、`bf484f93`、`821fb996`。提交号用于复核来源，不是通过条件。

| 检查 | 结果 | 证明范围 |
| --- | --- | --- |
| interfaces `kv_storage` | 4 通过 | typed storage 完整性、格式选择、scale 容量与 activation 边界 |
| interfaces `int8_kv` | 5 通过，部分与上一过滤器重叠 | 两个状态的合约、预算、持有及 capture/restore 生命周期 |
| Metal `causal_attention::conformance_tests::int8` | 6 通过，0 ignored | 实际量化写入、独立页边界、GQA/MQA、general/direct/tiled 读取及 Rust reference 对照 |
| Metal `completed_device_status` | 2 通过，0 ignored | 数值失败在 fence 后可见，复用 slot/stream 不改写旧 fence 的终态 |
| Metal `vnext_metal_checkpoint_continuation` 中 `causal_int8_kv` | 3 通过，0 ignored | public provider 的 Q4_K projection、跨页双状态恢复、追加 suffix、跨 dtype 拒绝 |
| `cargo check --locked --workspace --all-targets --features metal` | 通过（`RUSTFLAGS=-D warnings`） | 当前 workspace 的 Metal 编译；不等同于整个 workspace runtime 测试 |

首次 Metal 编译在 `-D warnings` 下因废弃 helper 未使用失败；已删除无用路径并重跑上述设备测试通过。全部 workspace 检查、真实模型验证、CUDA runtime 与发行验证仍需分别完成，不能由本表替代。

完整日志与模型输出放仓库外。完成发布资格时补充具体命令、设备/驱动信息、配置、原始结果位置及实际限制。

## 4. 可复用的 run / serve 配对验收

`ferrum-devtools` 的 `model_regression` 复用现有 JSONL、HTTP、SSE 与 basic/state 语义验证，新增公开测试参数 `--kv-dtype fp16|int8`。它会核验实际 run effective-config 和 serve health / effective-config 中来自 `resolved_model_plan` 的 requested、selected 与 numerical profile；仅有 CLI 输入回显不能通过。当前 prepared release task 尚未声明 KV 选择，因此该参数不能和 `--expected-task` 混用。

可先直接运行 runner 做 F16 校准，例如：

```text
model_regression --ferrum-bin /absolute/path/to/ferrum --model /absolute/path/to/vnext-model --backend metal --kv-dtype fp16 --report-dir /outside/repository/f16-calibration --checks basic,state --disable-thinking --context-tokens 4096 --max-num-seqs 1 --max-tokens 128 --long-context-records 128 --long-context-min-prompt-tokens 1024
```

长输入由 Rust 生成编号记录，在中间放置唯一目标键值，核验 run、HTTP JSON、SSE 和追加历史续答中的精确答案，并保留完整请求。`--long-context-records` 决定记录数；`--long-context-min-prompt-tokens` 用实际 tokenizer usage 验证输入长度，记录数本身不等于 token 数。示例长度须经所选模型的 F16 校准，并结合该模型 KV 页容量确认跨页覆盖；它不是已经测得的合格配置。

冻结校准配置后，`ferrum-cli/tests/vnext_kv_product.rs` 可在同一设备依次调用共享 runner，覆盖每个模型的 fp16/int8 两种格式。测试配置使用 typed JSON；以下路径均需换成当前环境的真实路径，报告目录必须为空并放在仓库外：

```json
{
  "ferrum_bin": "/absolute/path/to/ferrum",
  "runner_bin": "/absolute/path/to/model_regression",
  "backend": "metal",
  "hardware_label": "Apple M1 Max, 32 GiB, macOS 15.1.1",
  "report_dir": "/outside/repository/kv-paired",
  "context_tokens": 4096,
  "max_num_seqs": 1,
  "runtime_memory_budget_bytes": null,
  "max_tokens": 128,
  "disable_thinking": true,
  "repetitions": 1,
  "timeouts": { "startup_secs": 600, "request_secs": 300, "run_secs": 1200 },
  "models": [
    {
      "id": "hybrid-sample",
      "model": "/absolute/path/to/vnext-model",
      "source_label": "repository and immutable revision",
      "precision_label": "actual weight precision",
      "checks": ["basic", "state"],
      "expect_prefix_restore": true,
      "long_context": { "records": 128, "min_prompt_tokens": 1024 }
    }
  ]
}
```

`backend` 可为 `metal` 或 `cuda`。`models` 可包含多个本地模型；代码不会按模型名调整行为。每个模型的 `checks` 在 F16 校准后冻结，省略时默认为 basic/state；可使用共享 runner 支持的 basic、state、structured、tools 等检查。`long_context: null` 不追加长输入。某样本的 F16 无法完成 state 时应保留该失败，改由能完成该任务的样本验证 state；不能在看过 INT8 后删除失败检查。`expect_prefix_restore: true` 要求 native checkpoint health 的实际 hits 与 saved_prefill_tokens 增加；为 false 时仍保留实际观测及 unsupported reasons，不把 server 文本前缀重合当作 KV 恢复。

```text
cargo build -p ferrum-devtools --bin model_regression
FERRUM_KV_PRODUCT_CONFIG=/outside/repository/kv-config.json cargo test -p ferrum-cli --test vnext_kv_product -- --ignored --test-threads=1 --nocapture
```

环境变量仅为测试加载显式配置；Ferrum 产品行为仍通过公开 CLI 参数选择。Ferrum 二进制应另行按被测设备构建并传入配置。

产物包括 `configuration.json`、`paired.json`、每次 runner 的完整 report、实际命令、输入、原始 JSONL / HTTP / SSE、effective-config、前后 health 与子进程日志。失败结果会保留，其他独立配对仍继续记录；没有按通过率放宽语义门槛。HTTP `elapsed_ms` 是完整请求墙钟耗时，`first_body_chunk_ms` 只表示首个传输数据块，不能作为首个生成 token 的 TTFT。最终 health 必须没有失败请求或 executor failure；该产品检查不能证明所有内部浮点数有限，内部量化数值边界仍由设备测试覆盖。此夹具也不替代 teacher-forced KL/NLL、容量测量或完整发布资格。
