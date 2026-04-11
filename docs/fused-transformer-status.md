# Fused Transformer Implementation Status

## 当前卡点

**TTS 运行超时**（>60s 无输出）。需要定位是哪一步卡住：
- Speech tokenizer encoder (candle Metal, 63帧×32 codebooks)
- Prefill (73 tokens × 28 layers, 新的 fused path)
- Decode loop

**已加计时日志**但未能成功运行收集数据。管道 (`| grep`) 会截断后台进程的 stdout 导致进程被 SIGPIPE 杀死。

**正确的调试方式**：`2>&1 > /tmp/log.txt` 重定向到文件，不用管道。

## 架构

```
ferrum-attention/
├── src/
│   ├── lib.rs          # FusedTransformer — N层 transformer，自动选 Metal/CPU
│   ├── cpu/
│   │   ├── mod.rs      # Accelerate sgemm fused attention
│   │   └── transformer.rs  # CPU transformer layer (Accelerate sgemm + fused softmax)
│   └── metal/
│       ├── mod.rs
│       ├── pipelines.rs    # Metal pipeline cache + dispatch (GEMM 改用 Accelerate on shared buf)
│       ├── transformer.rs  # Metal transformer layer forward
│       └── shaders/
│           ├── flash_attn.metal       # Fused flash attention (tested, 1.68ms/prefill)
│           └── transformer_ops.metal  # RMSNorm, SiLU, add (tested), GEMM (8x8, 太慢已弃用)
└── tests/
    ├── metal_test.rs       # 4 tests pass (flash attention correctness + perf)
    └── transformer_test.rs # 1 test pass (single layer prefill + decode)
```

## Metal Transformer Layer 数据流

```
每层 forward (metal_layer_forward):

1. RMSNorm        → Metal kernel, 1 dispatch
2. Q/K/V proj     → cblas_sgemm on shared buffer (零拷贝，同步)
3. QK-norm + RoPE → CPU (element-wise)
4. KV cache       → CPU Vec append
5. Flash Attention → Metal kernel, 1 dispatch
6. Untranspose    → CPU
7. O proj         → cblas_sgemm
8. Residual add   → Metal kernel  ┐
9. Post RMSNorm   → Metal kernel  ┘ 1 dispatch
10. Gate/Up proj  → cblas_sgemm (×2)
11. SiLU×gate     → Metal kernel, 1 dispatch
12. Down proj     → cblas_sgemm
13. Residual add  → Metal kernel, 1 dispatch

总计：每层 4 次 Metal dispatch + 5 次 sgemm 同步调用
28 层 = 112 Metal dispatch + 140 sgemm
```

## 性能问题

1. **Metal GEMM 8x8 kernel 太慢** — 已替换为 Accelerate cblas_sgemm on shared buffers
2. **每层 4 次 `cmd.commit(); wait()`** — GPU sync 开销。可优化为批量提交
3. **Speech tokenizer encoder** — 原 candle Metal 代码，63帧×32 codebooks，可能也很慢
4. **管道截断问题** — `| grep` 导致进程被 SIGPIPE，看不到日志

## 已验证的精度数据

| 配置 | past_hidden 差异 vs Python |
|------|---------------------------|
| candle 原始 (Metal) | 2.0-5.0 |
| fused attention only (CPU) | 0.1-0.6 |
| fused transformer (Metal+Accel) | 未测到 |

## 下一步

1. **定位卡点**：重定向日志到文件，找出哪步耗时最长
2. **减少 Metal sync**：合并 Metal dispatches，或改为 CPU-only path 先验证正确性
3. **验证 ASR 输出**
4. **优化 Metal GEMM**：移植 llama.cpp 的 `kernel_mul_mm`（64×32 tiles, 4 simdgroups）或用 MPS
5. **Metal QK-norm + RoPE kernel**：消除 CPU 来回

## 关键文件

- `crates/ferrum-attention/src/metal/transformer.rs` — Metal 层 forward 逻辑
- `crates/ferrum-attention/src/metal/pipelines.rs` — Metal dispatch 和 GEMM（Accelerate）
- `crates/ferrum-models/src/architectures/qwen3_tts.rs` — Talker forward_step 集成
- `crates/ferrum-models/src/executor/tts_executor.rs` — TTS 执行器（有计时日志）
- `~/rust_ws/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` — 参考实现（kernel_mul_mm, kernel_flash_attn_ext）
