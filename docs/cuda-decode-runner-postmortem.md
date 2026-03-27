# CUDA Decode Runner 开发复盘

## 项目目标

绕过 candle 的 per-op 张量分配，用 cuBLAS + 自定义 CUDA kernel + 预分配 buffer 实现 decode 热路径，最终通过 CUDA Graphs 进一步减少 kernel launch 开销。

## 时间线与关键问题

### Phase 1: 基础设施搭建
**做了什么：** 4 个 CUDA kernel (rms_norm, rope, decode_attention, residual_add) + cuBLAS wrapper + 预分配 buffer pool + 权重提取

**遇到的问题：** 无（Mac 上只做编译检查，CUDA 代码在 `#[cfg(feature = "cuda")]` 后面）

**教训：** Mac 上无法验证 CUDA 代码的正确性，所有 API 用法都是读 cudarc 源码推测的，导致后续大量编译错误。

---

### Phase 2: CUDA 编译错误（RunPod 首次编译）

| 问题 | 根因 | 解决 |
|------|------|------|
| `CublasError` 路径不对 | cudarc 的类型在 `result` 子模块 | 改为正确路径 |
| `builder.launch()` 返回类型不匹配 | 返回 `Option<(CudaEvent, CudaEvent)>` 不是 `()` | 加 `.map(|_| ())` |
| `impl DeviceSlice` 不满足 `DeviceRepr` | kernel helper 用了错误的泛型约束 | 改为具体类型 `CudaSlice/CudaView` |
| `end_capture(0)` 类型不对 | 需要枚举不是 integer | 用正确的枚举值 |
| 借用检查器冲突 | `self.kv_states.get_mut()` 和 `self.launch_*()` 冲突 | launch helper 改为关联函数，kv_state 用 `remove()` 取出 |

**教训：** `#[cfg(feature = "cuda")]` 的代码在 Mac 上完全不编译检查。应该在 CI 加 CUDA 编译检查（即使没有 GPU）。

---

### Phase 3: 运行时 panic

| 问题 | 根因 | 诊断方法 | 解决 |
|------|------|---------|------|
| `CudaView::slice() unwrap on None` | RoPE 临时 buffer 用了 `norm_out`（hidden_size=2560），但 Q 需要 `q_dim=4096` | 加 debug logging 打印 buffer 大小 | 加专用 `rope_q_temp/rope_k_temp` buffer |
| KV cache "No KV cache for sequence" | `init_kv_cache()` 从未被调用 | 代码审查 | 加 `ensure_runner_kv_cache()` 在首次 decode 时迁移 |
| `CudaSlice::clone()` panic | cudarc 跨 stream 事件同步对 candle 的 CudaSlice 不兼容 | 读 cudarc 源码 line 1839 | 改为直接接管 candle Tensor 的 CudaSlice（不 clone） |
| 乱码输出 | attention kernel KV layout 错误：`[kv_heads, seq, dim]` 应为 `[seq, kv_heads, dim]` | 对比 candle 的 PreAllocKvCache layout | 修改 kernel 索引 |

**教训：** Qwen3-4B 的 `hidden_size=2560 ≠ q_dim=4096`（因为 head_dim=128），不能假设相等。buffer 大小必须用实际 dim 计算。

---

### Phase 4: CUDA Graph capture 失败

| 问题 | 根因 | 诊断方法 | 解决 |
|------|------|---------|------|
| `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` | candle 用 default stream（blocking），Graph 要求 non-blocking | 读 candle 源码：`context.default_stream()` vs `context.new_stream()` | runner 创建自己的 non-blocking stream |
| `CUBLAS_STATUS_EXECUTION_FAILED` during capture | cudarc 的 `DevicePtr::device_ptr_mut()` 在 capture 期间调 `stream.wait(event)` | 逐个 kernel 测试 capture：rms_norm OK, cuBLAS FAIL | `context.disable_event_tracking()` 在 capture 前 |
| cuBLAS workspace 未预分配 | cuBLAS 在 capture 期间动态分配 workspace（不是 capture-safe） | NVIDIA 文档 | `cublasSetWorkspace_v2()` 预分配 32MB |
| capture 失败后所有请求 0 token | `end_capture` 未被调用，stream 卡在 capture 模式 | 代码审查 | `capture_one()` helper 保证 always call `end_capture` |
| Graph 比 eager 慢（100 vs 109 tok/s） | 37 个 graph launch（~100µs/个）> 500 个直接 kernel launch（~5µs/个） | benchmark 对比 | 默认禁用 graph（WARMUP=usize::MAX） |

**教训：**
1. vLLM 的 CUDA Graph 策略是对的——piecewise capture，attention 走 eager
2. cudarc 的 safe API（event tracking）不兼容 CUDA Graph capture
3. Graph launch 开销大于预期，37 个小 graph 不如直接 launch
4. 要让 graph 有收益需要合成 1 个大 graph（需要 `cuGraphExecKernelNodeSetParams`，cudarc 不支持）

---

### Phase 5: 跨 GPU 兼容性（RTX PRO 6000 / Blackwell）

| 问题 | 根因 | 诊断方法 | 解决 |
|------|------|---------|------|
| `cuInit` 返回 999 | `NVIDIA_VISIBLE_DEVICES=void` 导致 GPU 不可见 | `env` 查看环境变量 | `export NVIDIA_VISIBLE_DEVICES=all` |
| 链接错误 `-lnvrtc -lcublas` | CUDA 库在非标准路径 | `find / -name "libcublas.so*"` | 创建符号链接 + `.cargo/config.toml` 设置 link args |
| `CUDA_COMPUTE_CAP=120` 不支持 | nvcc 12.8 只支持到 cc90 | 编译错误信息 | `CUDA_COMPUTE_CAP=89`（PTX 向前兼容） |
| 输出乱码（交错重复） | 初始诊断误导：以为是跨 stream 事件/attention kernel/权重 offset | 见下方详细分析 | `stream.synchronize()` |

#### 乱码问题的完整诊断过程

**现象：** "你好Another！way有什么to我可以say help you你？？"——两个回复交错

**错误诊断路径（浪费了大量时间）：**
1. ❌ 以为是 `CudaSlice::clone()` 跨 stream 事件同步问题 → 改为直接接管 → 没修好
2. ❌ 以为是权重拷贝未同步 → 加 `stream.synchronize()` 在权重拷贝后 → 没修好
3. ❌ 以为是 `mem::forget(SyncOnDrop)` 破坏事件链 → 恢复 safe wrapper → 部分改善但仍有问题
4. ❌ 以为是 warp-cooperative attention kernel 的跨 warp 归约 bug → 改为简单 kernel → 没修好
5. ❌ 以为是权重 tensor 的 `start_offset` 未应用 → 加了 offset 处理 → 没修好
6. ❌ 加诊断代码对比 candle logits → 诊断代码本身调了 `forward_decode` 污染了 candle KV cache → 误以为修好了

**正确诊断路径：**
1. ✅ 加 `FERRUM_DISABLE_CUDA_RUNNER=1` 环境变量，不重编就能切换测试
2. ✅ 加 `FERRUM_LOG_TOKENS=1` 打印每步 argmax → 发现 CUDA runner 的 argmax **和 candle 完全一致**
3. ✅ 对比 top-5 logits → token ID 一致，logit 值相差 <0.03
4. ✅ 确认 logits shape/dtype 一致（`[1, 1, 151936]` F16）
5. ✅ 结论：logits 计算正确，sampler 读到了错误的数据

**根因：** `clone_dtod` 在 runner 的 non-blocking stream 上产生 logits CudaSlice，但 `CudaStorage::wrap_cuda_slice` 用 candle 的 default stream 包装。sampler 在 default stream 上读数据时，non-blocking stream 的写入可能还没完成。

**修复：** 在返回 logits 前 `stream.synchronize()`，确保数据写入完成。

**为什么在 RTX 5090 上没问题：** 5090 的 GPU 足够快，non-blocking stream 上的操作在 sampler 读取前已经完成（race condition 没被触发）。

---

## 关键教训总结

### 1. 不要在没有 GPU 的环境下写 CUDA 代码
`#[cfg(feature = "cuda")]` 让编译器完全跳过 CUDA 代码。应该在 CI 加 CUDA 编译检查。

### 2. 不要猜 API，读源码
cudarc 的文档不完整。每次猜测 API 用法都导致编译错误。应该在实现前完整读源码。

### 3. Stream 同步是 CUDA 编程的核心问题
non-blocking stream 和 default stream 之间没有隐式同步。任何跨 stream 的数据传递都需要显式同步或事件机制。

### 4. 诊断代码不能有副作用
对比 CUDA runner 和 candle 的 logits 时，诊断代码调了 `forward_decode` 修改了 candle 的 KV cache。这导致：
- 误以为 bug 已修复（因为诊断污染了状态）
- 去掉诊断后 bug "回来了"
- 浪费了大量时间在错误的方向上

### 5. 先研究最佳实践，再实现
CUDA Graphs 的实现走了弯路：
- 先尝试 capture 整个 forward pass → 失败（attention 变长）
- 再尝试 per-layer piecewise → 失败（cudarc event tracking 不兼容）
- 研究 vLLM 后才知道正确做法
- 最终 graph 比 eager 慢，因为 launch 开销

应该在第一行代码前就研究 vLLM 的实现。

### 6. 环境变量 > 重新编译
加了 `FERRUM_DISABLE_CUDA_RUNNER=1` 和 `FERRUM_LOG_TOKENS=1` 后，在 RunPod 上不需要重新编译就能切换测试。这省了大量时间（每次编译 2-3 分钟）。

### 7. 跨 GPU 测试是必要的
RTX 5090 上正确的代码在 RTX PRO 6000 上乱码。不同 GPU 的 timing、stream 行为、memory 分配顺序都不同。race condition 在快 GPU 上可能不触发。

---

## 当前状态

| 组件 | 状态 |
|------|------|
| CudaDecodeRunner | ✅ 正确运行（RTX 5090 + RTX PRO 6000） |
| 自定义 CUDA kernels | ✅ rms_norm, rope, decode_attention, flash_decode_attention, paged_decode_attention, residual_add, fused_add_rms_norm, fused_silu_mul (+interleaved) |
| cuBLAS GEMM | ✅ 支持 batch decode（m=batch） |
| 预分配 DecodeBuffers | ✅ 支持 max_batch_size 缩放 |
| 双缓冲 residual | ✅ 跨层 norm 融合，减少 108 次 kernel launch |
| Flash Decoding (split-K) | ✅ 长上下文自动启用（kv_len > 256） |
| Paged KV Attention | ✅ GPU block pool + block-table 间接寻址 + free-list 回收 |
| Batch Decode | ✅ batched GEMM + per-item attention，executor/engine 端到端集成 |
| CUDA Graphs | ⚠️ 架构就绪但默认禁用（比 eager 慢） |
| TransformerWeights 抽象 | ✅ Qwen3 已实现，Llama/Qwen2 待做 |
| 运行时诊断 | ✅ FERRUM_DIAG=1 控制 shapes/attn/timing 日志 |
| Bench CLI | ✅ 顺序/并发/长上下文模式 |

## 性能数据

### RTX PRO 6000, Qwen3-4B FP16（最新）

| 场景 | Decode tok/s | TPOT | 备注 |
|------|-------------|------|------|
| 优化前基线 | 73.5 | 13.60ms | 旧 CUDA runner |
| 单请求（优化后） | **88.8** | 11.26ms | +21%，双缓冲+跨层融合 |
| 4 并发 batch decode | **109.4** | 4.75ms | batched GEMM |
| 4 并发 paged KV | **102.9** | 5.05ms | block-table attention |
| 长 decode (1024 tok) | 79.7 | 12.54ms | flash decode 自动启用 |
| 长 prompt (~2k tok) | 78.1 | 12.81ms | flash decode + 长 prefill |

### RTX 5090, Qwen3-4B FP16
| 路径 | Decode tok/s | TPOT |
|------|-------------|------|
| Candle (fused kernels) | ~100 | ~10ms |
| CUDA Runner (eager) | ~109 | ~9.15ms |

### Mac (CPU), Qwen2.5-0.5B
| 路径 | Decode tok/s | TPOT |
|------|-------------|------|
| CPU | 14.1 | 71ms |
| Metal | 31.9 | 31ms |
