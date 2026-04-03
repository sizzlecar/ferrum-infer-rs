# Tensor Parallel Decode — Qwen3-4B 调试复盘

**日期**: 2026-04-02
**分支**: feat/tensor-parallel
**硬件**: 2× NVIDIA RTX PRO 6000 Blackwell Server Edition

---

## 1. 目标

在 TinyLlama TP=2 成功运行（52 tok/s）的基础上，让 Qwen3-4B TP=2 正确运行。

## 2. 问题现象

Qwen3-4B TP=2 输出全 NaN logits，无法生成任何 token。单 GPU Qwen3-4B 正常（80 tok/s）。

## 3. 发现的 Bug（按定位顺序）

### Bug 1: candle stream 与 runner stream 未同步（根因）

**位置**: `crates/ferrum-models/src/loader/tp_weight_loader.rs`

**原因**: `load_sharded_weights` 中，candle 的 tensor 操作（`vb.get`、`narrow`、`cat`、`to_device`）在 candle 的默认 CUDA stream 上执行。但 `GpuWeight::from_tensor` 用 runner 自己的 stream（`rs`）做 `clone_dtod`。两个 stream 之间没有同步，导致 `clone_dtod` 读到的是 **未完成写入的 stale GPU 数据**。

**影响**: 所有 TP 权重数据全部错误。gate_up 权重本应是 ~0.004，实际读到的是 ~1.0（stale 数据）。MLP 输出因此放大 190 倍，residual 在 3 层内溢出 FP16 范围。

**修复**: 在每次 `GpuWeight::from_tensor` 前调用 `candle_stream.synchronize()`。封装为 `to_gpu`/`to_gpu_linear` helper。

**验证**:
- Python 直接读 safetensors 确认原始数据正确
- `[TP LOAD]` 诊断确认 VarBuilder 返回的 tensor 数据正确（first5 匹配 Python）
- `[WEIGHT]` 诊断确认 runner 里的权重在 sync 后也正确

### Bug 2: VarBuilder 在不同 GPU 上 BF16→F16 转换结果不同

**位置**: candle 的 `VarBuilder::from_mmaped_safetensors`

**原因**: Qwen3-4B 的 safetensors 存储格式是 BF16。每个 rank 在自己的 GPU 上独立加载 VarBuilder 并做 BF16→F16 转换。两个 GPU 对相同数据的转换结果不一致（L1 input_layernorm.weight max_diff=0.507）。

**影响**: replicated 权重（norm、embed、lm_head、RoPE）在两个 rank 之间不同。norm_out 不同导致 MLP 输入不同，errors 随层数累积。

**修复**: 加载完成后，从 rank 0 读取所有 replicated 权重（D2H），覆盖写入到 rank 1（H2D）。见 `CudaDecodeRunner::sync_replicated_weights_from()`。

**验证**:
- PyTorch 验证同一数据通过 `to_device` 在两个 GPU 间传输 diff=0
- `[TP WEIGHT]` 诊断确认 sync 后 max_diff=0.000000

**注**: Bug 1 修复后，此 bug 可能已被间接解决（VarBuilder "divergence" 可能也是 stale data 导致）。但 `sync_replicated_weights_from` 作为安全网保留。

## 4. 犯的错误

### 4.1 诊断方法不系统（最大的错误）

一开始就该做的事情：
1. **对比单 GPU 和 TP 的中间值**（embed → norm → attn → oproj → mlp → residual）
2. **对比两个路径的权重数据**

但实际花了大量时间在：
- 检查 all-reduce 是否工作（工作正常）
- 检查 fused_add_rms_norm kernel 是否正确（正确）
- 检查维度是否匹配（匹配）
- 检查 weight sharding 逻辑（逻辑正确）

**教训**: 遇到数值错误，第一步应该是 dump 两个路径的 **权重数据** 和 **中间激活值**，直接对比，而不是猜测哪个模块有问题。

### 4.2 没有先在 GPU 机器上用 Python 验证假设

多次在本地猜测原因、改代码、push、让用户编译测试。应该先 SSH 到 GPU 机器，用 Python 快速验证假设（如 BF16→F16 在不同 GPU 上是否一致、safetensors 数据是否正确）。

### 4.3 修复 bind_to_thread 问题时不够谨慎

尝试将 VarBuilder 加载改为单一 GPU + cross-device 复制，但 `bind_to_thread` 在这个项目中反复出问题（cudarc 的 CUDA context 管理不可靠）。应该记住这个历史经验，避免依赖 `bind_to_thread`。

### 4.4 盲目尝试去掉 thread::scope

没有在 GPU 机器上先验证 "NCCL all-reduce 能否顺序调用" 就直接改代码 push，结果导致死锁。之前的对话中已经验证过顺序调用会死锁，应该记住这个结论。

### 4.5 诊断代码反复添加/修改

每次只添加一小部分诊断，需要多轮 push/compile/run 才能定位问题。应该一次性添加完整的诊断（权重 dump + 中间值 dump + rank 对比），减少 GPU 机器上的编译次数。

## 5. 最终状态

### 正确性：已修复 ✓

- Qwen3-4B TP=2 生成 16 个 token，logits 正常（±15）
- Residual 量级与单 GPU 一致（L0~6, L5~50）
- 所有 36 层无 NaN/inf

### 性能：13 tok/s（待优化）

| 配置 | Decode tok/s | TPOT |
|------|-------------|------|
| 单 GPU Qwen3-4B | ~80 | ~12ms |
| TP=2 Qwen3-4B | 13 | ~77ms |

**瓶颈**: 每个 decode step 调用 72 次 `std::thread::scope`（36 层 × 2 次 all-reduce），每次创建/销毁 2 个线程。thread spawn/join 开销远大于 NCCL 通信本身。

### 性能优化方向：持久化 per-rank 线程

当前架构：主线程驱动，`for_each_rank!` 串行 enqueue，`thread::scope` 做 all-reduce。

目标架构：
- 每个 rank 一个持久线程，跑完整 decode 循环
- NCCL all-reduce 是线程间唯一同步点
- 主线程通过 channel 发送 token_id，从 rank 0 线程接收 logits
- 这是 vLLM / Megatron-LM 的标准做法

预期提升：去掉 thread::scope 开销后，TPOT 应接近 `单GPU_TPOT / 2 + NCCL_latency`。

## 6. 关键代码变更

| 文件 | 变更 | 类型 |
|------|------|------|
| `tp_weight_loader.rs` | `to_gpu`/`to_gpu_linear` helper，每次 from_tensor 前 sync candle stream | Bug fix |
| `cuda_decode.rs` | `sync_replicated_weights_from()` 方法 | Bug fix |
| `candle_executor.rs` | 加载后调用 `sync_replicated_weights_from` 广播 replicated 权重 | Bug fix |
| `tp_decode.rs` | decode_step 清理（去掉诊断代码） | Cleanup |

## 7. 给未来自己的建议

1. **数值问题 → 先 dump 权重和中间值，直接对比**。不要从代码逻辑推导。
2. **跨 stream 操作 → 永远显式 sync**。cudarc 的 event tracking 不可靠。
3. **NCCL all-reduce → 必须多线程同时调用**。cudarc wrapper 不是真正的 async。
4. **bind_to_thread → 别用**。在这个项目里反复出问题。
5. **GPU 机器上先用 Python 验证假设**，再改 Rust 代码。
