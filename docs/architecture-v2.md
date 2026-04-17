# Architecture v2 RFC — Extensibility + Performance

## 1. 背景与目标

### 当前阶段的成果
- Qwen3 `ModelRunner<CpuBackend | MetalBackend>` 可用，Metal decode 33 tok/s（从 5.4 的 6x）
- Parity test 保证正确性
- 单 cmd buffer 模式（MetalContext）已验证

### 当前阶段的问题
1. `TransformerConfig` 是 "decoder-only + GQA + SwiGlu + 可选 QK-norm" 专用，塞 MoE / MLA / multimodal 只能靠膨胀 enum
2. `layer_forward_fused` 假设"标准 transformer layer"结构，无法覆盖 MoE / MLA / cross-attention
3. `LayerWeights<B>` 硬编码 dense f32 buffer，GPTQ/AWQ/GGUF 塞不进去
4. Backend trait 里混了 transformer-specific 方法（split_qkv / qk_norm / rope / kv_cache_append / transpose 等），MetalBackend 用 override 绕开后多数变成死代码
5. 非 LLM（Whisper/Bert/CLIP/TTS）绕过 registry

### 本 RFC 的目标（P0 优先级）
1. **性能第一**：热路径最小开销。Context 批处理保持现有 single-cmd-buffer 模式。量化 GEMM 用 fused kernel，不展中间 fp16 buffer
2. **扩展第一**：新模型 = 新文件，不改共享基础设施；新量化 = 新 `Linear` impl；新 attention = 新 Backend method
3. **主流量化必支持**：GPTQ / AWQ / GGUF（含 K-quants）
4. **近期目标**：MoE（Mixtral, DeepSeek-MoE）、MLA（DeepSeek-V2/V3）、多模态（Qwen-VL, LLaVA）

### 非目标（本期不做）
- Tensor Parallel（多 GPU 分片）—— 之后单独 RFC
- Pipeline Parallel
- Speculative Decoding
- Long-context extrapolation（RoPE scaling 以外的）

---

## 2. 设计原则

### 2.1 Model-as-Code，不是 Model-as-Config
每个模型一个独立文件，模型显式写 `forward`，调 Backend op。参考 vLLM / candle-transformers / mistral.rs。

**代价**：每模型 300-600 LOC，有"样板代码"。
**回报**：任何模型的特殊结构（MLA 压缩、MoE 路由、sliding window 位置、Qwen-VL 的视觉适配器）都有专属代码路径，不是 config 里的一个 if/else。
**共享**：通用 helper（RoPE 预计算、KV cache 管理、linear forward dispatch）放在 `ferrum-models/src/common/`。

### 2.2 正交分解三个维度
- **Backend**（在哪算）：CPU / Metal / CUDA
- **Linear**（权重格式）：Dense / GPTQ / AWQ / GGUF
- **Model**（怎么算）：Qwen3 / DeepSeek-V3 / Qwen-VL

这三者**独立变化**：加新 backend 不碰模型，加新量化不碰 backend，加新模型不碰 Linear。

### 2.3 Backend trait 只管硬件
Backend 不知道"transformer 是什么"。它提供 alloc/memcpy/math primitives/attention variants。Transformer 结构是模型代码的事。

### 2.4 零成本抽象优先
- `Box<dyn Linear<B>>` 的 vtable 开销 < 1 ns/call，对 GEMM（毫秒级）可忽略
- 但 hot loop 里的小 op 不走 dyn，直接调 `B::xxx`
- Enum + inline 的地方（KV cache block selection、dtype dispatch）用 monomorphization

---

## 3. 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│  ferrum-cli  /  ferrum-server                                │
│   run / serve / bench / transcribe / tts / embed             │
└────────────────────────┬─────────────────────────────────────┘
                         │ Box<dyn ModelExecutor>
┌────────────────────────▼─────────────────────────────────────┐
│  ferrum-engine :: Executor Layer                             │
│   LlmExecutor         ← Box<dyn DecoderOnlyLLM>              │
│   EncDecExecutor      ← Box<dyn EncoderDecoderLM>            │
│   MultimodalExecutor  ← Box<dyn MultimodalLLM>               │
│   EmbedExecutor       ← Box<dyn EmbeddingModel>              │
│   TtsExecutor         ← Box<dyn TtsModel>                    │
│   + Registry (arch → executor + loader)                      │
│   + Scheduler / KV manager / Sampler integration             │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  ferrum-models :: Model Layer (Model-as-Code)                │
│   qwen3.rs / llama.rs / mistral.rs / deepseek_v3.rs          │
│   mixtral.rs / qwen_vl.rs / llava.rs                         │
│   whisper.rs / bert.rs / clip.rs / qwen3_tts.rs              │
│   common/ (rope, sampler helpers, kv block mgmt)             │
│   → 每模型: struct + forward + impl DecoderOnlyLLM etc.      │
│   → 持有 Box<dyn Linear<B>> / B::Buffer for tensors          │
└────────────────────────┬─────────────────────────────────────┘
                         │ Box<dyn Linear<B>>, B::Buffer
┌────────────────────────▼─────────────────────────────────────┐
│  ferrum-quantization :: Linear Layer (权重抽象)              │
│   Dense<B> / Gptq<B> / Awq<B> / Gguf<B>                      │
│   WeightLoader (safetensors / gguf → 对应 Linear 变种)        │
└────────────────────────┬─────────────────────────────────────┘
                         │ Backend trait
┌────────────────────────▼─────────────────────────────────────┐
│  ferrum-kernels :: Backend Layer                             │
│   CpuBackend / MetalBackend / CudaBackend                    │
│   Ops: alloc, gemm_f{32,16}, gemm_gptq, gemm_awq,            │
│         gemm_gguf, rms_norm, rope, silu_mul, softmax,        │
│         embedding, flash_attn, paged_attn, cross_attn,       │
│         mla_attn (DeepSeek V2/V3)                            │
│   Context: GPU command buffer 批处理                          │
└──────────────────────────────────────────────────────────────┘
```

### Crate 变化

保留：`ferrum-types`, `ferrum-interfaces`, `ferrum-kernels`, `ferrum-scheduler`, `ferrum-sampler`, `ferrum-tokenizer`, `ferrum-kv`, `ferrum-runtime`, `ferrum-models`, `ferrum-engine`, `ferrum-server`, `ferrum-cli`, `ferrum-testkit`

新增：`ferrum-quantization`（Linear trait + GPTQ/AWQ/GGUF 实现）

合并：`ferrum-attention` → `ferrum-kernels::backend::metal::shaders`（已无独立存在必要）

---

## 4. Backend Trait 设计

### 4.1 最小接口

```rust
pub trait Backend: Send + Sync + Sized + 'static {
    type Context;
    type Buffer: Send + Sync;

    // ── Lifecycle ───────────────────────────────────────
    fn new_context() -> Self::Context;
    fn sync(ctx: &mut Self::Context);

    // ── Memory ──────────────────────────────────────────
    fn alloc(len: usize, dtype: DType) -> Self::Buffer;
    fn from_host(ctx: &mut Self::Context, data: &[u8], dtype: DType) -> Self::Buffer;
    fn to_host(buf: &Self::Buffer, dtype: DType) -> Vec<u8>;

    // ── Dense math ──────────────────────────────────────
    fn gemm(ctx, a: &Buffer, b: &Buffer, out: &mut Buffer,
            m: usize, n: usize, k: usize, dtype: DType);

    fn rms_norm(ctx, x, w, eps, out, tokens, dim, dtype);
    fn rope(ctx, q, k, cos, sin, positions, ...);
    fn silu_mul(ctx, gate, up, out, len, dtype);
    fn add(ctx, a, b, out, len, dtype);
    fn softmax(ctx, x, rows, cols, dtype);
    fn embedding(ctx, table, ids, out, dim, dtype);

    // ── Attention variants ──────────────────────────────
    fn flash_attention(ctx, q, k, v, out, params);
    fn paged_attention(ctx, q, kv_blocks, block_table, out, params);
    fn cross_attention(ctx, q, kv_encoded, out, params);
    fn mla_attention(ctx, q, kv_compressed, kv_uncompressed, out, params);

    // ── Quantized GEMM (optional, Backend 选择实现) ──────
    fn gemm_gptq(ctx, a, qweight, scales, zeros, g_idx, out,
                 m, n, k, bits, group_size) -> Result<()>;
    fn gemm_awq(ctx, a, qweight, scales, zeros, out,
                m, n, k, bits, group_size) -> Result<()>;
    fn gemm_gguf(ctx, a, qweight, out, m, n, k, quant_type) -> Result<()>;
}
```

### 4.2 被删除的方法

以下**从 Backend trait 移除**（都是 transformer-specific，应该由模型代码组合基础 op）：
- `split_qkv` — 模型自己管布局
- `qk_norm` — 变成 `rms_norm` 的特殊调用
- `kv_cache_append` — 变成 KV cache manager 的职责（`ferrum-kv`）
- `silu_mul_split` — 模型里两次 linear 后再 silu_mul
- `transpose_token_to_head` / `transpose_head_to_token` — 由 Backend 内部 attention 实现处理
- `add_inplace` — 变成 `add` 的 in-place 情况
- `layer_forward_fused` — Backend 不懂 layer 是什么

### 4.3 Context 批处理模式（保留）

Metal/CUDA 的性能关键。Context 持有 GPU command buffer，所有 op 往里加 encoder，`sync()` 才 commit+wait。这是当前 MetalContext 的模式。

CPU backend 的 Context = `()`，op 立即执行。

---

## 5. Linear Trait + 量化支持

### 5.1 Linear trait

```rust
pub trait Linear<B: Backend>: Send + Sync {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;

    /// Forward: out[m, out_f] = input[m, in_f] @ W^T
    fn forward(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        out: &mut B::Buffer,
        m: usize,
    );
}
```

### 5.2 实现

```rust
pub struct DenseLinear<B: Backend> {
    weight: B::Buffer,   // [out, in], dtype: F32/F16/BF16
    bias: Option<B::Buffer>,
    in_f: usize,
    out_f: usize,
    dtype: DType,
}
impl<B: Backend> Linear<B> for DenseLinear<B> {
    fn forward(...) {
        B::gemm(ctx, input, &self.weight, out, m, self.out_f, self.in_f, self.dtype);
        if let Some(b) = &self.bias { B::add(...); }
    }
}

pub struct GptqLinear<B: Backend> {
    qweight: B::Buffer,   // int4 packed as int32
    scales: B::Buffer,    // [groups, out] fp16
    zeros: B::Buffer,     // [groups, out / 8] int4 packed
    g_idx: Option<B::Buffer>,
    bits: u32,            // 4 or 8
    group_size: usize,    // 典型 128
    in_f: usize,
    out_f: usize,
}
impl<B: Backend> Linear<B> for GptqLinear<B> {
    fn forward(...) {
        B::gemm_gptq(ctx, input, &self.qweight, &self.scales, &self.zeros,
                     self.g_idx.as_ref(), out, m, self.out_f, self.in_f,
                     self.bits, self.group_size)?;
    }
}

pub struct AwqLinear<B: Backend> { ... }
pub struct GgufLinear<B: Backend> {
    weight: B::Buffer,
    quant_type: GgufQuant,  // Q4_0 / Q4_K / Q5_K / Q8_0 / ...
    in_f: usize, out_f: usize,
}
```

### 5.3 GEMM 内核策略（重点）

- **GPTQ INT4**：参考 vLLM 的 Marlin kernel。Metal 上需要写 shader；CUDA 可以直接调 Marlin PTX
- **AWQ INT4**：参考 AutoAWQ 的 kernel，CUDA 有现成
- **GGUF**：每种 quant type 独立 dequant 路径，推荐 Metal 参考 llama.cpp 的 `ggml-metal.metal`

这些 kernel 大部分工作在 CUDA 侧（因为 production 推理用 CUDA）。Metal 侧起步先支持 FP16 + GPTQ-INT4，AWQ/GGUF 后续加。

### 5.4 KV Cache 也要支持量化

DeepSeek 的 INT8 KV cache、SmoothQuant 的 FP8 KV cache 是近期方向。

引入 `KvCacheDType` 枚举（F32 / F16 / BF16 / INT8 / FP8），PagedKvCache 带 dtype，Backend 的 `paged_attention` 按 dtype 分派。

---

## 6. Attention 变体

Backend trait 暴露四种 attention：

| 方法 | 用途 | 典型模型 |
|------|------|---------|
| `flash_attention` | 标准 MHA/GQA，dense KV | Qwen3, Llama3, Mistral |
| `paged_attention` | 分页 KV（continuous batching 必需） | 所有 production LLM |
| `cross_attention` | Encoder 产的 K/V 固定 | Whisper, T5 |
| `mla_attention` | 压缩 KV 的 MLA | DeepSeek V2/V3 |

`sliding_window` 不是独立方法，而是 `flash_attention` 的 `params.window_size`。

`moe_attention` 不存在 —— MoE 是 MLP 层的事，不是 attention 的事。

### 扩展

未来加 sparse attention / linear attention，都是 Backend trait 加新方法。不破坏已有模型代码。

---

## 7. Model-as-Code Pattern

### 7.1 一个典型模型

```rust
// ferrum-models/src/qwen3.rs
use ferrum_kernels::backend::Backend;
use ferrum_quantization::Linear;

pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
}

pub struct Qwen3Layer<B: Backend> {
    input_ln: B::Buffer,                   // [hidden]
    qkv_proj: Box<dyn Linear<B>>,          // fused QKV
    q_norm: B::Buffer,                     // [head_dim]  ← Qwen3 特有
    k_norm: B::Buffer,                     // [head_dim]
    o_proj: Box<dyn Linear<B>>,
    post_ln: B::Buffer,
    gate_up_proj: Box<dyn Linear<B>>,      // fused gate+up
    down_proj: Box<dyn Linear<B>>,
}

pub struct Qwen3Model<B: Backend> {
    cfg: Qwen3Config,
    embed: B::Buffer,
    layers: Vec<Qwen3Layer<B>>,
    final_norm: B::Buffer,
    lm_head: Box<dyn Linear<B>>,
    // 运行时 scratch
    scratch: ModelScratch<B>,
    rope_cache: RopeCache<B>,
}

impl<B: Backend> Qwen3Model<B> {
    pub fn new(cfg: Qwen3Config, loader: &dyn WeightLoader) -> Result<Self> {
        // weight_loader 决定 lm_head 是 DenseLinear / GptqLinear / ...
        let lm_head = loader.load_linear::<B>("lm_head")?;
        let embed = loader.load_tensor::<B>("model.embed_tokens.weight")?;
        let layers = (0..cfg.num_layers).map(|i| Qwen3Layer {
            qkv_proj: loader.load_linear(&format!("model.layers.{i}.self_attn.qkv_proj"))?,
            // ...
        }).collect::<Result<Vec<_>>>()?;
        ...
    }

    fn layer_forward(
        &mut self,
        ctx: &mut B::Context,
        li: usize,
        residual: &mut B::Buffer,
        positions: &[u32],
        kv: &mut KvCache<B>,
        tokens: usize,
    ) {
        let s = &mut self.scratch;
        let layer = &self.layers[li];
        let cfg = &self.cfg;

        // 1. input rms_norm
        B::rms_norm(ctx, residual, &layer.input_ln, cfg.rms_norm_eps, &mut s.norm, ...);

        // 2. QKV projection (Linear dispatches to Dense/GPTQ/AWQ/GGUF)
        layer.qkv_proj.forward(ctx, &s.norm, &mut s.qkv, tokens);

        // 3. split QKV (Backend API is gone — model does this with memcpy or a small kernel)
        self.split_qkv(ctx, &s.qkv, &mut s.q, &mut s.k, &mut s.v, tokens);

        // 4. Qwen3-specific: QK-norm (just a rms_norm per head)
        B::rms_norm(ctx, &s.q, &layer.q_norm, cfg.rms_norm_eps, ..., tokens * cfg.num_heads, cfg.head_dim);
        B::rms_norm(ctx, &s.k, &layer.k_norm, cfg.rms_norm_eps, ..., tokens * cfg.num_kv_heads, cfg.head_dim);

        // 5. RoPE
        B::rope(ctx, &mut s.q, &mut s.k, &self.rope_cache.cos, &self.rope_cache.sin, positions, ...);

        // 6. append to KV cache
        kv.append(ctx, &s.k, &s.v, tokens);

        // 7. flash/paged attention
        B::flash_attention(ctx, &s.q, &kv.k_buf, &kv.v_buf, &mut s.attn, AttnParams {
            q_len: tokens, kv_len: kv.len, heads: cfg.num_heads, kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim, causal: tokens > 1, ...
        });

        // 8. O projection
        layer.o_proj.forward(ctx, &s.attn, &mut s.o, tokens);

        // 9. residual + post-attn norm
        B::add(ctx, residual, &s.o, residual, ...);
        B::rms_norm(ctx, residual, &layer.post_ln, cfg.rms_norm_eps, &mut s.norm, ...);

        // 10. gate_up -> silu_mul -> down
        layer.gate_up_proj.forward(ctx, &s.norm, &mut s.gu, tokens);
        self.silu_mul_from_fused(ctx, &s.gu, &mut s.silu, tokens);
        layer.down_proj.forward(ctx, &s.silu, &mut s.mlp, tokens);
        B::add(ctx, residual, &s.mlp, residual, ...);
    }
}

impl<B: Backend> DecoderOnlyLLM for Qwen3Model<B> {
    fn prefill(&mut self, tokens: &[u32], kv: &mut KvCache) -> Logits { ... }
    fn decode(&mut self, token: u32, pos: u32, kv: &mut KvCache) -> Logits { ... }
}
```

### 7.2 示例：DeepSeek-V3 MLA

```rust
pub struct DeepSeekV3Layer<B: Backend> {
    input_ln: B::Buffer,
    kv_a_proj_with_mqa: Box<dyn Linear<B>>,  // [kv_lora_rank + qk_rope_head_dim, hidden]
    kv_a_layernorm: B::Buffer,
    kv_b_proj: Box<dyn Linear<B>>,           // [nh * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    q_a_proj: Box<dyn Linear<B>>,            // Q 也有 compression
    q_a_layernorm: B::Buffer,
    q_b_proj: Box<dyn Linear<B>>,
    o_proj: Box<dyn Linear<B>>,
    // MoE MLP
    gate: Box<dyn Linear<B>>,                // expert router
    experts: Vec<DeepSeekExpert<B>>,         // 256 experts (V3)
    shared_experts: Vec<DeepSeekExpert<B>>,  // dense shared
}

impl<B> DeepSeekV3Model<B> {
    fn layer_forward(...) {
        // MLA attention: 
        //   kv_compressed = kv_a_proj(x)[:, :kv_lora_rank] + norm
        //   kv_rope = kv_a_proj(x)[:, kv_lora_rank:] + rope
        //   -> B::mla_attention
        // MoE MLP:
        //   scores = gate(x); top_k_experts = topk(scores, k=8)
        //   for expert in top_k: result += expert.forward(x) * score
        //   result += shared_experts.forward(x)
    }
}
```

这些复杂结构**每个模型自己写**，不要试图抽象。

### 7.3 多模态示例：Qwen-VL

```rust
pub struct QwenVLModel<B: Backend> {
    vision: VisionTransformer<B>,     // ViT 结构
    adapter: Box<dyn Linear<B>>,      // vision → text hidden
    llm: Qwen3Model<B>,               // 复用 Qwen3 当 LLM backbone
}

impl<B> MultimodalLLM for QwenVLModel<B> {
    fn encode_image(&mut self, pixels: &[u8]) -> VisualTokens {
        let vit_out = self.vision.forward(pixels);
        self.adapter.forward(&vit_out)
    }
}

impl<B> DecoderOnlyLLM for QwenVLModel<B> {
    fn prefill(&mut self, tokens: &[u32], kv: &mut KvCache) -> Logits {
        // 同 Qwen3
        self.llm.prefill(tokens, kv)
    }
    fn decode(&mut self, token: u32, pos: u32, kv: &mut KvCache) -> Logits {
        self.llm.decode(token, pos, kv)
    }
}
```

---

## 8. ModelFamily Traits

```rust
pub trait DecoderOnlyLLM: Send + Sync {
    fn prefill(&mut self, tokens: &[u32], kv: &mut KvCache) -> Logits;
    fn decode(&mut self, token: u32, pos: u32, kv: &mut KvCache) -> Logits;
    fn config(&self) -> &LlmRuntimeConfig;
}

pub trait EncoderDecoderLM: Send + Sync {
    fn encode(&mut self, input: &[u32]) -> EncoderState;
    fn decode_step(&mut self, token: u32, pos: u32, enc: &EncoderState,
                   kv: &mut KvCache) -> Logits;
}

pub trait MultimodalLLM: DecoderOnlyLLM {
    fn encode_image(&mut self, pixels: &ImageBuffer) -> VisualTokens;
    fn encode_audio(&mut self, pcm: &[f32]) -> AudioTokens;  // 可选
}

pub trait EmbeddingModel: Send + Sync {
    fn embed(&mut self, tokens: &[u32]) -> Vec<f32>;
}

pub trait TtsModel: Send + Sync {
    fn synthesize(&mut self, text: &str, speaker: Option<&SpeakerRef>) -> AudioBuffer;
}

pub trait Transcriber: Send + Sync {
    fn transcribe(&mut self, pcm: &[f32], language: Option<&str>) -> TranscriptSegments;
}
```

### `ModelExecutor` 统一入口

```rust
pub trait ModelExecutor: Any + Send + Sync {
    fn info(&self) -> &ModelInfo;
    fn as_llm(&mut self) -> Option<&mut dyn DecoderOnlyLLM> { None }
    fn as_enc_dec(&mut self) -> Option<&mut dyn EncoderDecoderLM> { None }
    fn as_multimodal(&mut self) -> Option<&mut dyn MultimodalLLM> { None }
    fn as_embedder(&mut self) -> Option<&mut dyn EmbeddingModel> { None }
    fn as_tts(&mut self) -> Option<&mut dyn TtsModel> { None }
    fn as_transcriber(&mut self) -> Option<&mut dyn Transcriber> { None }
}
```

CLI 从 registry 拿到 `Box<dyn ModelExecutor>`，根据命令调合适的 `as_*`。

---

## 9. WeightLoader

```rust
pub trait WeightLoader: Send + Sync {
    fn load_tensor<B: Backend>(&self, name: &str) -> Result<B::Buffer>;
    fn load_linear<B: Backend>(&self, name: &str) -> Result<Box<dyn Linear<B>>>;
    fn has_tensor(&self, name: &str) -> bool;
    fn quant_config(&self) -> &Option<QuantConfig>;
}

// 具体实现
pub struct SafeTensorsLoader { mmap, metadata, quant_config }
pub struct GgufLoader { mmap, metadata }
```

`load_linear` 返回不同的 Linear 实例：
- 无量化 → `DenseLinear`
- `quantize_config.json` 有 `"quant_method": "gptq"` → `GptqLinear`
- 同上 AWQ
- GGUF 文件 → `GgufLinear`

模型代码只调 `loader.load_linear("model.layers.0.self_attn.qkv_proj")`，loader 内部处理格式差异。

---

## 10. Registry & Factory

```rust
pub struct ModelFactory;

impl ModelFactory {
    pub fn build(
        model_id: &str,
        device: Device,
        overrides: ModelOverrides,  // force dtype, quant hint, etc.
    ) -> Result<Box<dyn ModelExecutor>> {
        let def = registry::resolve(model_id)?;
        let loader = WeightLoader::for_path(&def.path)?;

        match (device, def.architecture) {
            (Device::Metal, Architecture::Qwen3) => {
                let model = Qwen3Model::<MetalBackend>::new(cfg, &loader)?;
                Ok(Box::new(LlmExecutor::new(model)))
            }
            (Device::CUDA(_), Architecture::DeepSeekV3) => {
                let model = DeepSeekV3Model::<CudaBackend>::new(cfg, &loader)?;
                Ok(Box::new(LlmExecutor::new(model)))
            }
            (Device::Metal, Architecture::Whisper) => {
                let model = WhisperModel::<MetalBackend>::new(cfg, &loader)?;
                Ok(Box::new(TranscriberExecutor::new(model)))
            }
            ...
        }
    }
}
```

CLI 所有命令走这个：
```rust
// ferrum-cli/src/commands/run.rs
let exec = ModelFactory::build(&cmd.model, device, overrides)?;
let llm = exec.as_llm().ok_or("model is not a decoder-only LLM")?;
chat_loop(llm, ...);
```

---

## 11. 性能关键点

### 11.1 Context 批处理（保留）
Metal 的 single-cmd-buffer 模式继续保留。Backend::Context 持有 GPU 命令流，模型 forward 只在最后 sync 一次。

### 11.2 Linear dyn dispatch 开销
- `Box<dyn Linear<B>>` 每次 forward 一次 vtable lookup
- 典型场景：28 层 × 4 linear/layer = 112 次/decode step，30ms/step → 0.001% 开销
- 不是瓶颈

### 11.3 模型 hot path 不用 dyn
- `B::gemm` / `B::rms_norm` 等基础 op 是 monomorphized，零开销
- 只有 Linear（权重维度）走 dyn，因为量化类型多

### 11.4 量化 kernel 必须 fused
- GPTQ dequant 不展开成 fp16 中间 buffer
- Backend::gemm_gptq 一步完成 dequant+gemm（Marlin 做法）
- Metal/CUDA 都要实现

### 11.5 KV cache dtype
- 支持 FP16 / BF16 / INT8 KV（节省带宽和 VRAM）
- `PagedKvCache` 带 dtype 字段

---

## 12. 迁移计划

### Phase A：Backend trait 瘦身 + 量化 skeleton（1 周）
- **A1**: 从 Backend trait 删除 transformer-specific 方法（split_qkv/qk_norm/kv_cache_append/transpose_*/silu_mul_split/add_inplace/layer_forward_fused）
- **A2**: 引入 `ferrum-quantization` crate，定义 `Linear<B>` trait + `DenseLinear`
- **A3**: Backend trait 加 `gemm_gptq` / `gemm_awq` / `gemm_gguf`（先 unimplemented，后续补 kernel）
- **A4**: WeightLoader trait，`SafeTensorsLoader` 先返回 DenseLinear
- **验证**: 保持 Qwen3 ModelRunner 工作（parity + bench 不变）

### Phase B：Qwen3 Model-as-Code + GPTQ（1-2 周）
- **B1**: 新 `ferrum-models/src/qwen3.rs` 作为 Model-as-Code 示例
- **B2**: 迁移当前 ModelRunner 逻辑到 Qwen3Model::forward
- **B3**: Metal 实现 `gemm_gptq`（Marlin-style kernel 或参考 llama.cpp）
- **B4**: Qwen3-GPTQ end-to-end 工作，parity test 对 Candle
- **B5**: 删除 `ModelRunner<B>`（被 Qwen3Model 替换）
- **验证**: Qwen3 FP32 和 Qwen3 GPTQ-INT4 在 Metal 都过 parity + bench

### Phase C：其他 LLM 迁过来（1 周）
- **C1**: `llama.rs` / `mistral.rs`
- **C2**: 删除 `CandleModelExecutor`, `Qwen2ModelExecutor`, `Qwen3ModelExecutor`, `MetalLlamaExecutor`, `MetalQwen2Executor`
- **C3**: registry → ModelFactory 重构，CLI 统一入口
- **验证**: 所有 LLM bench，Llama / Qwen2 各一个 parity test

### Phase D：近期目标模型（3-4 周，分批）
按重要性：
- **D1**: DeepSeek-V3 MLA + MoE（`mla_attention` + MoE expert routing）
- **D2**: Qwen-VL 多模态（Vision backbone + adapter）
- **D3**: Mixtral MoE（最早 MoE 实例，简单 expert routing）
- **D4**: AWQ 支持（`gemm_awq` kernel）
- **D5**: GGUF 支持（`GgufLoader` + `gemm_gguf` kernel）
- **D6**: Whisper / Bert / CLIP 迁到新 executor framework

### Phase E：CUDA 完成度（持续，需 GPU 机器）
- E1: CudaBackend 所有 methods 实现
- E2: CUDA 版 gemm_gptq (Marlin)、gemm_awq
- E3: CI cargo check cuda feature 强制过

---

## 13. 关键决策记录

### D1. 为什么 Linear 用 trait object（`Box<dyn>`）而不是 enum？
- Enum 的 `match Dense | Gptq | Awq | Gguf` 对"加新量化类型" 不开放
- trait object 的 vtable 开销 ~1 ns，对 linear forward（>100us）可忽略
- **决定**：trait object

### D2. 为什么 Backend trait 不用 associated types 区分 dtype？
- 考虑过 `Backend<Dtype = F32>` + `Backend<Dtype = F16>`，复杂且违反"一个 Backend 一种设备"
- **决定**：Backend 方法带 `dtype: DType` 参数，内部 dispatch

### D3. 为什么不用 candle 作为 tensor 抽象？
- candle 带了自己的 autograd / tensor tracker，推理场景不需要
- candle 的 Metal/CUDA op 不够快（Metal 上的 Qwen3 一开始跑 19 tok/s 全靠 CPU Accelerate）
- **决定**：Backend trait 自己管 tensor，candle 只在 `WeightLoader` 里加载 safetensors 文件（可能后续也删）

### D4. 为什么 MoE 不做成 Backend trait 方法？
- MoE 是 model-level 逻辑（expert 路由 + combine），不是硬件操作
- 每层 MoE 的 expert 数量/路由策略不同（Mixtral 8/2、DeepSeek-V3 256/8+shared）
- **决定**：MoE 在模型代码里写，用 `B::gemm` 多次调用

### D5. 为什么 ModelRunner 被删除？
- ModelRunner 的 `layer_forward` 强假设了 transformer 结构
- MoE / MLA / multimodal 塞不进去
- 与"Model-as-Code"原则冲突
- **决定**：每个模型自己写 forward，ModelRunner 退休

### D6. ferrum-attention 合并去向？
- 现在只有 Metal pipelines 被 MetalBackend 用
- TTS 走 FusedTransformer 的代码迁到 `ferrum-models/src/qwen3_tts.rs` 里自己写
- **决定**：shader 迁到 `ferrum-kernels/src/metal/shaders/`, crate 删除

---

## 14. 开放问题（待讨论）

1. **RoPE 预计算 cache**: 放模型里（每模型各自），还是放 Backend 里全局复用？倾向后者（`RopeCache<B>` 作为 ferrum-kernels 里的 helper）
2. **Sampler 放哪**：`ferrum-sampler` 保留，但接口要不要从 `logits_cpu: Vec<f32>` 改成 `logits: B::Buffer`？会省一次 device→host 拷贝，但把 sampler 绑 Backend
3. **Continuous batching 的 attention**：`paged_attention` vs `flash_attention` 的选择时机。是否 `paged_attention` 作为默认，`flash_attention` 只在无 batching 时用？
4. **Quantized KV cache**：先 INT8 还是先 FP8？FP8 需要更新的 GPU 支持

---

## 15. 本 RFC 不包括

- TP / PP / 多 GPU
- Speculative decoding
- Structured output（已在 engine 层做）
- Prefix caching（已在 engine 层做）
- LoRA 热加载
- Draft model

这些保持现状或后续 RFC。

---

## 下一步

1. 等待这个 RFC review 反馈
2. Confirm 后开始 Phase A（Backend trait 瘦身 + Linear trait skeleton），1 周
3. 之后按 Phase B / C / D 顺序推进
