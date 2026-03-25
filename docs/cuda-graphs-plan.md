# CUDA Graphs Decode 优化开发计划

## 背景

### 问题
Qwen3 decode 阶段每生成一个 token 需要 ~550 次 CUDA kernel launch：
- 36 layers × ~15 launches/layer (QKV proj, Q/K norm, RoPE, flash_attn, O proj, fused_add_rms_norm, gate/up proj, fused_silu_mul, down proj)
- 每次 launch 的 CPU 开销 ~5µs → 总计 ~2.75ms/token 纯 CPU 开销
- 对于小模型 (0.6B-3B) 这个开销占比很大，限制了 tok/s 上限

### 目标
用 CUDA Graph 录制整个 decode forward pass，后续 replay 只需 1 次 CPU 调用，消除 CPU kernel launch 瓶颈。

## 现状分析

### 依赖栈
```
Qwen3 Model (candle-core 0.9.2)
    ↓
candle (CUDA backend) + candle-flash-attn
    ↓
cudarc 0.19 (仅 "driver" feature)
    ↓
CUDA Runtime
```

### 关键限制
**cudarc 0.19 的 `driver` feature 不暴露 CUDA Graph API。**

cudarc 是 candle 的 CUDA 绑定层。当前配置只启用了 `driver` feature，不包含 graph capture/launch 功能。

### Decode 数据流 (单 token)
```
input_ids [1,1] (i64)
    → embed_tokens → [1,1,hidden_size]
    → 36× DecoderLayer:
        → input_layernorm (RMS Norm)
        → qkv_proj (Linear) → split Q/K/V
        → q_norm, k_norm (RMS Norm)
        → RoPE (cos/sin lookup + rotate)
        → KV cache append (slice_set, O(1))
        → flash_attn / manual attention
        → o_proj (Linear)
        → fused_add_rms_norm (residual + norm)
        → gate_proj + up_proj (Linear)
        → fused_silu_mul
        → down_proj (Linear)
        → residual add
    → final norm (RMS Norm)
    → lm_head (Linear)
    → logits [1,1,vocab_size]
```

## 方案选择

### 方案 A: cudarc CUDA Graph (推荐)
通过启用 cudarc 的额外 feature 或升级版本获取 Graph API。

**实现路径：**
1. 检查 cudarc 0.19 是否有隐藏的 graph feature（查看 crates.io 文档）
2. 如果没有，升级到支持 graph 的 cudarc 版本（需要与 candle 0.9.2 兼容）
3. 如果 candle 锁定了 cudarc 版本，可能需要 fork candle 或等待上游支持

**风险：** candle 内部硬绑定 cudarc 版本，升级可能破坏兼容性

### 方案 B: 原生 CUDA FFI Graph
绕过 cudarc，直接 FFI 调用 CUDA Runtime API。

**需要的 CUDA Runtime 函数：**
```c
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)
cudaStreamEndCapture(stream, &graph)
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0)
cudaGraphLaunch(graphExec, stream)
cudaGraphExecUpdate(graphExec, graph, &resultInfo)  // 更新图参数
```

**实现路径：**
1. 新建 `crates/ferrum-cuda-kernels/src/cuda_graph.rs`
2. 通过 `cuda-sys` crate 或手写 FFI 绑定获取 CUDA Runtime 函数
3. 获取 candle CUDA stream：`tensor.device().as_cuda_device()` → 内部 cudarc stream
4. 在 stream 上 capture decode forward pass
5. 需要处理 stream 获取问题（candle 可能不暴露底层 stream handle）

**风险：** 需要获取 candle 内部使用的 CUDA stream handle，可能需要 unsafe hack

### 方案 C: 更多 Fused Kernels (保守备选)
如果 Graph 方案受阻，继续写更多 fused CUDA kernel 减少 launch 次数。

**候选 fused kernels：**

| Fused Kernel | 融合操作 | 每层减少 launches | 总减少 |
|---|---|---|---|
| fused_qk_norm_rope | Q-norm + K-norm + RoPE | 4→1 | 36×3 = 108 |
| fused_gate_up_silu | gate_proj + up_proj + SiLU·mul | 3→1 | 36×2 = 72 |
| fused_qkv_proj | 已有 (candle fused) | 0 | 0 |

总计可从 ~550 减少到 ~370 launches (~33% 减少)

**风险：** 低，但收益有限（33% vs Graph 的 ~99%）

### 推荐路径
**先尝试 A → 如果受阻尝试 B → 如果都不行退回 C**

## 方案 A 详细设计

### Step 1: 验证 cudarc Graph 支持

```bash
# 检查 cudarc 0.19 所有 features
cargo doc -p cudarc --all-features 2>&1 | grep -i graph

# 或直接看源码
find ~/.cargo/registry/src -path "*/cudarc-0.19*" -name "*.rs" | xargs grep -l "graph\|Graph"
```

如果 cudarc 0.19 有 graph 模块，在 `ferrum-cuda-kernels/Cargo.toml` 启用：
```toml
cudarc = { version = "0.19", features = ["driver", "graph"] }  # 假设 feature 名
```

### Step 2: Graph Manager 结构

新建文件：`crates/ferrum-cuda-kernels/src/cuda_graph.rs`

```rust
pub struct CudaGraphManager {
    /// 已录制的 graph executable
    graph_exec: Option<CudaGraphExec>,
    /// 录制时使用的固定 buffer（decode 输入/输出形状固定）
    input_buffer: CudaSlice<i64>,   // [1, 1] token id
    output_buffer: CudaSlice<f32>,  // [1, 1, vocab_size] logits
    /// KV cache 指针（每步更新 offset，不改变 buffer 地址）
    kv_cache_ptrs: Vec<(*mut f16, *mut f16)>,  // per-layer (K, V) device pointers
    /// 当前 sequence position（用于 RoPE cos/sin 查表）
    position_buffer: CudaSlice<i64>,
    /// 是否已录制
    is_captured: bool,
    /// 最大支持的 sequence length（预分配 KV cache 大小）
    max_seq_len: usize,
}
```

### Step 3: Graph Capture 流程

```rust
impl CudaGraphManager {
    /// 首次 decode 调用时录制 graph
    pub fn capture_decode_graph(
        &mut self,
        model: &mut Model,  // Qwen3 model
        warmup_input: &Tensor,  // [1, 1] dummy token
        warmup_pos: usize,
        cache_key: &str,
    ) -> Result<()> {
        let stream = get_cuda_stream(warmup_input.device())?;

        // 1. Stream synchronize before capture
        stream.synchronize()?;

        // 2. Begin capture
        stream.begin_capture(StreamCaptureMode::Global)?;

        // 3. Run decode forward pass (all kernels recorded, not executed)
        let _logits = model.forward(warmup_input, warmup_pos, None, cache_key)?;

        // 4. End capture → get graph
        let graph = stream.end_capture()?;

        // 5. Instantiate executable graph
        self.graph_exec = Some(graph.instantiate()?);
        self.is_captured = true;

        Ok(())
    }

    /// 后续 decode 调用：更新输入 → replay graph
    pub fn replay_decode(
        &self,
        token_id: u32,
        position: usize,
    ) -> Result<Tensor> {
        // 1. 更新 input_buffer（token id）— host-to-device copy
        self.input_buffer.copy_from_host(&[token_id as i64])?;

        // 2. 更新 position_buffer（RoPE position）
        self.position_buffer.copy_from_host(&[position as i64])?;

        // 3. KV cache offset 已由 PreAllocKvCache 管理
        //    因为用了预分配 buffer + slice_set，GPU 指针不变，只是写入 offset 变了
        //    ⚠️ 这里是关键：slice_set 的 offset 参数需要通过 graph 参数更新

        // 4. Replay graph
        let stream = self.get_stream()?;
        self.graph_exec.as_ref().unwrap().launch(stream)?;

        // 5. 从 output_buffer 构造 Tensor 返回
        let logits = wrap_cuda_slice_as_tensor(&self.output_buffer, shape)?;
        Ok(logits)
    }
}
```

### Step 4: 集成到 Qwen3 Decode 路径

修改文件：`crates/ferrum-models/src/architectures/qwen3.rs`

```rust
pub struct ModelForCausalLM {
    base_model: Model,
    lm_head: Linear,
    // 新增
    #[cfg(feature = "cuda")]
    cuda_graph: Option<CudaGraphManager>,
}

impl ModelForCausalLM {
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
        cache_key: &str,
    ) -> CandleResult<Tensor> {
        #[cfg(feature = "cuda")]
        if seqlen_offset > 0 {
            // Decode phase: use graph if available
            if let Some(ref graph) = self.cuda_graph {
                if graph.is_captured() {
                    return graph.replay_decode(input_ids, seqlen_offset);
                }
            }
            // First decode call: capture graph
            if self.cuda_graph.is_none() {
                let mut graph = CudaGraphManager::new(...)?;
                graph.capture_decode_graph(&mut self.base_model, input_ids, seqlen_offset, cache_key)?;
                self.cuda_graph = Some(graph);
                // First call already executed during capture, return result
                return Ok(captured_output);
            }
        }

        // Prefill phase or non-CUDA: normal path
        // ... existing code ...
    }
}
```

### Step 5: KV Cache 兼容性

**关键挑战：** KV cache 每步增长，但 CUDA Graph 要求固定的 kernel 参数。

**解决方案：** 我们已经用了 `PreAllocKvCache`（预分配到 max_seq_len），所以：
- GPU buffer 指针**不变**（预分配的）
- 每步只改变 `current_len` offset（用于 `slice_set` 和 `narrow`）
- `slice_set` offset 是 kernel 参数，需要通过 `cudaGraphExecKernelNodeSetParams` 更新

```rust
// PreAllocKvCache 已有的结构（不需要改）
pub struct PreAllocKvCache {
    k_cache: Tensor,  // [batch, max_seq, kv_heads, head_dim] 预分配
    v_cache: Tensor,  // 同上
    current_len: usize,
}

impl PreAllocKvCache {
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // slice_set 写入 current_len 位置 — GPU 指针不变
        self.k_cache.slice_set(k, 1, self.current_len)?;
        self.v_cache.slice_set(v, 1, self.current_len)?;
        self.current_len += k.dim(1)?;
        // narrow 返回 view — GPU 指针不变
        Ok((
            self.k_cache.narrow(1, 0, self.current_len)?,
            self.v_cache.narrow(1, 0, self.current_len)?,
        ))
    }
}
```

**问题：** `narrow` 返回的 view 长度每步变化 → attention matmul 的矩阵大小变化 → **Graph 不兼容！**

**解决方案：用 padding mask 代替动态 narrow**

```rust
// Graph-compatible KV cache access
fn get_kv_for_graph(&self) -> (Tensor, Tensor) {
    // 返回完整 max_seq_len buffer，用 attention mask 屏蔽无效位置
    // 这样 tensor shape 永远固定 → Graph 兼容
    (self.k_cache.clone(), self.v_cache.clone())
}
```

配合 attention mask：
```rust
// 构造 causal mask，屏蔽 position > current_len 的 KV
let mask = create_causal_mask_with_kv_len(
    query_len=1,
    kv_len=max_seq_len,    // 固定大小
    valid_kv_len=current_len + 1,  // 实际有效长度
)?;
```

### Step 6: RoPE Position 更新

RoPE 需要当前 position 来查 cos/sin 表。在 Graph 中：
- cos/sin 表预计算到 max_seq_len → GPU buffer 固定
- position offset 作为 kernel 参数 → 需要通过 graph 参数更新

**方案：** 预计算 position 到 device buffer，graph replay 前用 memcpy 更新。

## 方案 B 详细设计 (FFI 备选)

如果 cudarc 0.19 不支持 graph，直接 FFI：

### 新建依赖

```toml
# ferrum-cuda-kernels/Cargo.toml
[dependencies]
cuda-sys = "0.3"  # 或手写 FFI bindings
```

### FFI Bindings

```rust
// crates/ferrum-cuda-kernels/src/cuda_graph_ffi.rs

extern "C" {
    fn cudaStreamBeginCapture(
        stream: cudaStream_t,
        mode: cudaStreamCaptureMode,
    ) -> cudaError_t;

    fn cudaStreamEndCapture(
        stream: cudaStream_t,
        pGraph: *mut cudaGraph_t,
    ) -> cudaError_t;

    fn cudaGraphInstantiate(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        pErrorNode: *mut cudaGraphNode_t,
        pLogBuffer: *mut c_char,
        bufferSize: usize,
    ) -> cudaError_t;

    fn cudaGraphLaunch(
        graphExec: cudaGraphExec_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
```

### 获取 Candle 的 CUDA Stream

```rust
// candle 通过 cudarc 管理 stream
// cudarc::driver::CudaStream 内部持有 raw stream handle
// 需要从 candle device 提取：

fn get_raw_stream(device: &CandleDevice) -> Result<cudaStream_t> {
    let cuda_dev = device.as_cuda_device()?;
    // cudarc 0.19 CudaDevice 可能暴露 stream()
    // 如果不暴露，需要用 default stream (0) 或 unsafe 提取
    unsafe { Ok(std::ptr::null_mut()) }  // default stream
}
```

**风险：** 使用 default stream 可能与 candle 内部 stream 冲突。

## 方案 C 详细设计 (Fused Kernels 备选)

### Kernel 1: fused_qk_norm_rope

融合 Q-Norm + K-Norm + RoPE 为一个 kernel。

```
输入: Q [batch, q_heads, 1, head_dim], K [batch, kv_heads, 1, head_dim]
      cos [1, head_dim/2], sin [1, head_dim/2]
      q_weight [head_dim], k_weight [head_dim]  (RMS norm weights)
输出: Q_rotated, K_rotated (same shapes)

每层减少: 4 launches → 1 launch (q_norm, k_norm, q_rope, k_rope)
总减少: 36 × 3 = 108 launches
```

**CUDA kernel 设计：**
```cuda
__global__ void fused_qk_norm_rope_kernel(
    const half* __restrict__ q,      // [num_heads, head_dim]
    const half* __restrict__ k,      // [num_kv_heads, head_dim]
    const float* __restrict__ q_w,   // [head_dim] rms norm weight
    const float* __restrict__ k_w,   // [head_dim] rms norm weight
    const float* __restrict__ cos,   // [head_dim/2]
    const float* __restrict__ sin,   // [head_dim/2]
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    int num_q_heads, int num_kv_heads, int head_dim
) {
    // Each block handles one head
    // Step 1: RMS norm (reduction for variance, then scale)
    // Step 2: RoPE rotation (pair-wise rotation using cos/sin)
}
```

### Kernel 2: fused_gate_up_silu

融合 gate_proj + up_proj + SiLU·mul。

```
输入: x [batch, 1, hidden_size]
      gate_weight [intermediate_size, hidden_size]
      up_weight [intermediate_size, hidden_size]
输出: silu(gate(x)) * up(x) [batch, 1, intermediate_size]

每层减少: 3 launches → 1 (gate matmul, up matmul, silu_mul)
总减少: 36 × 2 = 72 launches
```

**注意：** 这个 kernel 包含 matmul，性能优化难度大，可能不如分开调用 cuBLAS。
实际可行的是只融合 `silu_mul`（已有）+ 把 gate/up 合并为一个 matmul：

```rust
// 当前: 2 次 matmul
let gate = gate_proj.forward(x)?;  // matmul 1
let up = up_proj.forward(x)?;      // matmul 2
let out = fused_silu_mul(&gate, &up)?;

// 优化: 1 次 matmul (concat weights)
let gate_up_weight = Tensor::cat(&[gate_weight, up_weight], 0)?;  // 初始化时做一次
let gate_up = x.matmul(&gate_up_weight.t())?;  // 1 次 matmul
let (gate, up) = gate_up.chunk(2, -1)?;
let out = fused_silu_mul(&gate, &up)?;

// 减少: 1 matmul launch/layer → 36 total
```

## 文件清单

### 方案 A/B 需要创建的文件

| 文件 | 说明 |
|---|---|
| `crates/ferrum-cuda-kernels/src/cuda_graph.rs` | CUDA Graph manager |
| `crates/ferrum-cuda-kernels/src/cuda_graph_ffi.rs` | (方案B) FFI bindings |

### 方案 A/B 需要修改的文件

| 文件 | 修改内容 |
|---|---|
| `crates/ferrum-cuda-kernels/Cargo.toml` | 启用 cudarc graph feature 或添加 cuda-sys |
| `crates/ferrum-cuda-kernels/src/lib.rs` | 导出 cuda_graph 模块 |
| `crates/ferrum-models/src/architectures/qwen3.rs` | ModelForCausalLM 集成 graph capture/replay |
| `crates/ferrum-models/src/architectures/qwen3.rs` | Attention: graph 模式下用 full KV buffer + mask |
| `crates/ferrum-models/Cargo.toml` | 添加 ferrum-cuda-kernels 依赖(如果还没有) |

### 方案 C 需要创建的文件

| 文件 | 说明 |
|---|---|
| `crates/ferrum-cuda-kernels/kernels/fused_qk_norm_rope.cu` | CUDA kernel 源码 |
| `crates/ferrum-cuda-kernels/src/fused_qk_norm_rope.rs` | Rust wrapper |

### 方案 C 需要修改的文件

| 文件 | 修改内容 |
|---|---|
| `crates/ferrum-cuda-kernels/build.rs` | 编译新 kernel |
| `crates/ferrum-cuda-kernels/src/lib.rs` | 导出新模块 |
| `crates/ferrum-models/src/architectures/qwen3.rs` | Attention 层使用 fused kernel |

## 预期收益

| 方案 | Kernel Launches | CPU 开销 | 理论提升 |
|---|---|---|---|
| 当前 | ~550/token | ~2.75ms | baseline |
| 方案 C (fused) | ~370/token | ~1.85ms | ~33% 减少 |
| 方案 A/B (graph) | 1/token | ~0.005ms | ~99.8% 减少 |

对于 0.6B 模型（decode compute ~1-2ms），Graph 方案可能将 decode latency 从 ~5ms 降到 ~2ms，tok/s 从 ~200 提升到 ~500。

## 实施顺序

```
Week 1: 验证 cudarc graph 可行性
  ├─ 可行 → Step 2-6 实施方案 A (预计 1 week)
  └─ 不可行 → 评估方案 B FFI 可行性
       ├─ 可行 → 实施方案 B (预计 1.5 weeks)
       └─ 不可行 → 实施方案 C fused kernels (预计 1 week)

Week 2-3: 实施选定方案
Week 3: 性能测试 + 调优
```

## 注意事项

1. **CUDA Graph 不支持动态 shape** — decode 阶段 shape 固定 (batch=1, seq=1) 所以没问题，但 KV cache view 长度变化需要用 padding+mask 方案解决
2. **Graph 不支持 CPU-GPU 同步** — capture 期间不能有 `cudaMemcpy` (sync) 或 `tensor.to_vec()`
3. **Graph 不支持动态控制流** — 不能有 if/else 依赖 GPU 数据
4. **多 sequence 并发** — 每个 cache_key 可能需要自己的 graph instance（因为 KV cache buffer 不同）
5. **candle 内部分配** — candle 的 tensor 运算可能触发 GPU 内存分配，Graph capture 期间的分配会被固化，需要确保所有 buffer 预分配
6. **Flash Attention 兼容性** — candle-flash-attn 内部实现可能不支持 graph capture，需要验证
