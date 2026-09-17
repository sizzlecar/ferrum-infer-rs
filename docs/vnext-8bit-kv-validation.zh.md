# vNext 8-bit KV 验证记录

关联：[开发设计](vnext-8bit-kv-development.zh.md)。本记录区分合约、设备执行、模型质量与性能；前两项通过不代表后两项通过。当前尚未取得发布资格。

## 1. 验证环境与选样

2026-09-17 本机环境为 MacBookPro18,4、Apple M1 Max、32 GiB 统一内存、macOS 15.1.1（24B91）、Rust 1.91.0。清理旧临时代码索引约 19 GiB 后，可用磁盘从约 14 GiB 增至 33 GiB；源码、模型、已有测试记录均保留。Cargo 复用本机已有构建缓存。

模型按本次影响的状态结构选样，不能将一个模型的结果推广到所有模型：

| 模型及固定 revision | 权重输入 | 覆盖目的 |
| --- | --- | --- |
| Qwen/Qwen3-0.6B，`c1899de289a04d12100db370d81485cdf75e47ca` | 本地 safetensors | 纯 causal attention，28 层、8 KV heads、head dimension 128 |
| Qwen/Qwen3.5-0.8B，`2fc06364715b967f1860aea9cf38778875588b17` | 本地 safetensors | recurrent + causal，24 层、每 4 层一个 full attention、2 KV heads、head dimension 256 |

两个模型大小用于适配本机与现有 CUDA 设备；选择依据是 operation/state 覆盖。GGUF 权重路径另需真实模型验证，当前 Q4_K provider 的合成权重 continuation 测试不能替代它。

按未分页的活跃 KV 计算，纯 causal 样本每 token 从 114,688 字节变为 59,136 字节；混合样本的 full-attention KV 每 token 从 12,288 字节变为 6,240 字节。此处是布局公式，不是进程内存实测；分页、checkpoint、recurrent、权重与 workspace 需单独计入。

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
