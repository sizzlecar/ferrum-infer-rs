# Metal Backend 性能现状与问题记录

## 当前性能数据

| 路径 | tok/s | 备注 |
|------|-------|------|
| 旧 Candle 路径 | ~19 tok/s | Qwen3-0.6B, Metal backend 选项但实际用 Candle tensor ops |
| 新 ModelRunner + MetalBackend | ~5 tok/s | rms_norm/attention/add 用 Metal shader，GEMM 用 cblas |
| 新 ModelRunner + CpuBackend | ~9 tok/s | 全 CPU，Accelerate cblas + scalar loops |

测试环境：本机 Mac，Qwen3-0.6B (28层, hidden=1024, head_dim=128)，decode 模式 (tokens=1)。

## 已验证的事实

### 正确性
- MetalBackend vs CpuBackend cosine=1.000000（release 模式，真实 Qwen3-0.6B 权重）
- E2E 交互输出正确（"你好！有什么可以帮助你的吗？"，多轮对话正常）
- gemm_v2 Metal shader 在所有实际模型维度上 max_diff=0（1x4096x1024, 1x151936x1024 等）

### Metal command buffer 问题
- `to_owned()` 在 metal-rs 0.31 下 **hang**（release 模式卡死，debug 模式 SIGABRT）
- 原因：`foreign_obj_type!` 宏的 `obj_clone` 调 `msg_send![retain]`，但 autoreleased command buffer 可能在 retain 前已被 pool drain
- llama.cpp 的做法：`[queue commandBufferWithUnretainedReferences]` + 手动 `[cmd_buf retain]` + `@autoreleasepool` 包裹
- Rust 的 metal-rs 0.31 没有 `commandBufferWithUnretainedReferences` API，也没有安全的 autorelease pool 管理
- **结果：无法在 Rust 中安全持有 command buffer 跨函数调用，每个 Metal op 必须独立 cmd buffer + commit + wait**

### 性能分析
- 每层 3 次 Metal shader dispatch（rms_norm, attention, fused_add_rms_norm），每次 commit+wait
- 28 层 × 3 = 84 次 GPU sync
- 4 次 cblas GEMM/层（QKV, O-proj, gate_up, down）— 这部分速度跟 Candle 一样
- **瓶颈是 84 次 GPU sync 的开销，不是计算本身**

## 旧 Candle 路径为什么快

旧路径（Qwen3ModelExecutor + Candle）在 Metal 设备上的实际执行方式：
- **GEMM: cblas_sgemm**（通过 candle_nn::Linear → Candle dispatch → Accelerate）
- **RMS Norm: Candle CPU ops**（broadcast_div + broadcast_mul 或自定义 fused kernel）
- **Attention: Candle CPU standard attention**（matmul + softmax + matmul，全 CPU）
- **SiLU: Candle CPU**
- **零 GPU dispatch，零 Metal sync**

也就是说旧路径标称 "Metal backend" 但 **实际没有用任何 Metal GPU 计算**，全是 CPU（Accelerate BLAS + Candle scalar ops）。

## 未解决的问题

1. **Metal pipeline 无法实现**：metal-rs 0.31 的 `to_owned()` broken，无法持有 cmd buffer 跨多个 Backend 方法调用
2. **CpuBackend 有 bug**：`kv_cache_append` 在 `--backend cpu` 时 slice 越界（nkv * cl * hd 计算在 cl=0 时 from_raw_parts 长度为 0 但仍被索引）
3. **MetalBackend 的 gemm_v2 shader**：单独测试正确（max_diff=0），但放到 pipeline 中（to_owned cmd buffer）产生错误结果——原因是 cmd buffer hang 不是 shader 精度问题

## 文件位置

- MetalBackend: `crates/ferrum-kernels/src/backend/metal.rs`
- CpuBackend: `crates/ferrum-kernels/src/backend/cpu.rs`
- layer_forward: `crates/ferrum-kernels/src/backend/layer_forward.rs`
- ModelRunner: `crates/ferrum-kernels/src/backend/runner.rs`
- Backend trait: `crates/ferrum-kernels/src/backend/traits.rs`
- Metal pipeline tests: `crates/ferrum-models/tests/metal_runner_test.rs`
- Candle vs Runner compare: `crates/ferrum-models/tests/runner_compare_test.rs`

## 下一步可能的方向

1. 修 metal-rs `to_owned()` 或换 `metal` crate 版本（0.27 可能没有这个问题，ferrum-engine 用的就是 0.27）
2. 绕过 Backend trait，在 `layer_forward_fused` 里直接用函数局部 cmd buffer 编码所有 GPU ops（不跨函数）
3. decode 时全走 CPU（shared memory 零拷贝），只在 prefill 时用 Metal GPU
4. 修 CpuBackend 的 kv_cache_append bug，让纯 CPU 路径能跑
