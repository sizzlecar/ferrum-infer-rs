# Metal Backend 性能现状

## 当前性能数据 (Qwen3-0.6B, Mac, ModelRunner<MetalBackend>)

| 路径 | decode tok/s | TPOT | TTFT | 备注 |
|------|--------------|------|------|------|
| **新 MetalBackend (current)** | **32.5 tok/s** | **31ms** | 80ms | 全 Metal shader + m=1 GEMV + 整个 decode 1 个 cmd buffer |
| MetalBackend phase 2 (per-decode cmd buffer) | 13.0 tok/s | 77ms | 85ms | gemm_v2 tile kernel（m=1 时 tile 浪费）|
| MetalBackend phase 1 (per-layer cmd buffer) | 8.8 tok/s | 115ms | 123ms | 全 Metal shader，28 次 sync/decode |
| 旧 MetalBackend (per-op cmd buffer) | 5.4 tok/s | 184ms | 63ms | 每层 3 cmd buffer + cblas GEMM + CPU 标量 split/rope/kv_append |
| CpuBackend | 跑不完 (>5min/64tok) | — | — | 全 CPU，无 GPU 加速 |
| Candle legacy (Qwen3) | broken | — | — | Qwen3 在 Metal device 上 `no metal impl for rms-norm`，CPU 上 dtype mismatch；实际 baseline 不可用 |

**累计提升 6x** throughput，TPOT **降 83%**。满足底线：Metal > CPU，Metal > Candle。

测试：Qwen3-0.6B (28层, h=1024, nh=16, nkv=8, hd=128, im=3072)，decode tokens=1，release build。

## 当前 MetalBackend 结构 (ferrum-kernels/src/backend/metal.rs)

`MetalContext` 持有一个 `Option<&'static CommandBufferRef>`（queue 在 `OnceLock` 里永久存活，lifetime 安全），所有 op 把 encoder 添加到这个共享 cmd buffer。`B::sync()` 才 commit+wait+释放。

结果：ModelRunner::decode 的 `embedding_lookup + 28 × layer_forward_fused + final_rms_norm`（全部 GPU）积累到**同一个 cmd buffer**，之后 `lm_head gemm`（cblas CPU）自动触发 `ctx.flush()`，最后 `to_vec`。**整个 decode 只有 1 次 GPU sync**。

每层 13 个 encoder 顺序串起来（不 commit）：

每层 12 个 encoder 顺序串起来，一次 commit+wait：
1. `rms_norm_enc` (input norm)
2. `gemm_v2` (fused QKV projection)
3. `split_qkv_enc` (**新 shader** — fused qkv → Q/K/V 三个 buffer)
4. `qk_norm_rope` ×3 (Q/K 走 mode 1 norm+rope+transpose，V 走 mode 0 transpose only)
5. `kv_cache_append` ×2 (K, V 写入预分配 max_seq_len 的 cache)
6. `flash_attn_v2` (head-major Q/K/V, kv_seq_stride=max_seq_len)
7. `transpose_out` (head-major → token-major)
8. `gemm_v2` (O projection)
9. `fused_residual_norm_enc` (residual add + post-attn norm)
10. `gemm_v2` (gate_up projection)
11. `silu_mul_split_enc` (**新 shader** — fused gate_up → SiLU(gate)*up)
12. `gemm_v2` (down projection)
13. `add_enc` (final residual add)

KV cache 在**首次调用时**按 `cfg.max_seq_len` 预分配 `[nkv, max_seq_len, hd]`，之后永远复用（不再 realloc）。

整个 ModelRunner::decode() 的 sync 开销：**1 次 commit+wait**（在 lm_head 的 cblas 调用前由 `gemm()` 内部 `ctx.flush()` 触发）。

## 关键设计决策

- **Fused QKV 权重保留**：`LayerWeights.qkv_proj_w` 是 `[q_dim+2*kv_dim, hidden]` 融合矩阵。用 1 次 gemm_v2 + split_qkv_f32 替代 3 次独立 GEMM。
- **cblas 保留在 `MetalBackend::gemm()`**：只在 ModelRunner 末尾的 lm_head GEMM (m=1, n=vocab=152k) 时被调用；对于 m=1 的极端 tall-thin 矩阵，Accelerate 很难被 GPU shader 打败（GPU dispatch overhead 占比太大）。
- **per-op trait methods (split_qkv, qk_norm, rope, kv_cache_append, ...) 保留作为 CPU fallback**：不再被 layer_forward_fused 调用，但 Backend trait 要求实现。后续清理会把这些从 trait 中移除。

## 历史误诊与修正

`1c466ba` 的 commit message 和前一版的本文档都断言 "metal-rs 0.31 `to_owned()` hang 导致无法跨方法持有 cmd buffer" 是性能瓶颈。**这个结论是错的**。

真正瓶颈是 `layer_forward_fused` 被拆成 3 个 cmd buffer + cblas GEMM + CPU 标量循环，每层 3 次 GPU sync + 6+ 次 CPU-GPU 切换。

同 workspace 里 `ferrum-attention::metal_layer_forward_v2` 早就证明可以在一个函数栈帧里串起全部 encoder 并一次 commit（TTS 在用）。本次改造就是把这个模式搬到 MetalBackend 上，完全不需要 `to_owned()` 或跨函数 cmd buffer 持有。

## 下一步可能的优化

1. **GEMV shader 进一步优化**（已实现基础版本）
   - 当前 gemv_f32: 32 threads/tg, simd_sum K-reduction，每 threadgroup 算 1 个输出列
   - 参考 llama.cpp `kernel_mul_mv_f32_f32`: 每 threadgroup 算多列 + cross-simd reduction，memory coalescing 更好
   - 对 lm_head (N=152064) 尤其有效，当前是每层开销的大头

2. **Prefill 路径优化**（TTFT 80ms 还有空间）
   - tokens > 1 走 gemm_v2 tile kernel，m=8 时 tile 利用率 8/64
   - 方向：写 m<=16 的中等 GEMM kernel，或多流水线 prefill

3. **scratch buffer 复用问题**
   - 当前用 `o_proj_out` 和 `gate_up_out` 作为 K/V head-major staging，依赖 `2*im >= nkv*hd` 和 `h >= nkv*hd`，脆弱
   - 修复：给 `LayerScratch` 加专用 `k_head_buf` / `v_head_buf` 字段

4. **删除死代码**
   - MetalBackend::split_qkv / qk_norm / rope / kv_cache_append / silu_mul_split / add_inplace / transpose_* 现在不再被调用（layer_forward_fused override 了）
   - 待 Backend trait 做减法（把"整个 layer"而不是"单 op"作为强约束）后可以移除

5. **合并 ferrum-attention 到 ferrum-kernels**
   - Cargo.toml 里已标注 "to be merged"
   - `metal_layer_forward_v2`（TTS 用）和 `MetalBackend::layer_forward_fused`（LLM 用）大量重复
   - 迁移后 TTS 也走统一的 `ModelRunner<MetalBackend>`

## 文件位置

- MetalBackend (override): `crates/ferrum-kernels/src/backend/metal.rs`
- 新增 shader: `crates/ferrum-attention/src/metal/shaders/transformer_ops.metal` (split_qkv_f32, silu_mul_split_f32)
- ferrum-attention pipelines (复用): `crates/ferrum-attention/src/metal/pipelines.rs`
- 参考实现 (TTS): `crates/ferrum-attention/src/metal/transformer.rs::metal_layer_forward_v2`
- Parity test: `crates/ferrum-models/tests/metal_parity_test.rs` — prefill + 5 decode step 对拍 CpuBackend，每步 cosine > 0.9999、argmax 一致
