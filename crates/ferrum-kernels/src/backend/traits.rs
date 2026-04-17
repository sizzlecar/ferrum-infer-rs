//! Core Backend trait — the single abstraction over CUDA / Metal / CPU.

/// Configuration for RoPE (rotary position embeddings).
#[derive(Clone, Debug)]
pub struct RopeConfig {
    pub theta: f64,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

/// Configuration for attention dispatch.
#[derive(Clone, Debug)]
pub struct AttnConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub scale: f32,
}

/// Transformer model configuration (architecture-level).
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope: RopeConfig,
    pub has_qk_norm: bool,
    pub attn_type: AttnType,
    pub mlp_type: MlpType,
}

#[derive(Clone, Debug)]
pub enum AttnType {
    Gqa,
    SlidingWindow { window: usize },
}

#[derive(Clone, Debug)]
pub enum MlpType {
    SwiGlu,
    Moe { num_experts: usize, top_k: usize },
}

/// Per-layer weights stored in backend-native buffers.
pub struct LayerWeights<B: Backend> {
    pub input_ln_w: B::Buffer,
    pub qkv_proj_w: B::Buffer,
    pub o_proj_w: B::Buffer,
    pub post_ln_w: B::Buffer,
    pub gate_up_proj_w: B::Buffer,
    pub down_proj_w: B::Buffer,
    pub q_norm_w: Option<B::Buffer>,
    pub k_norm_w: Option<B::Buffer>,
}

/// Pre-allocated scratch buffers for one layer forward pass.
pub struct LayerScratch<B: Backend> {
    pub norm_out: B::Buffer,
    pub qkv_out: B::Buffer,
    pub q_buf: B::Buffer,
    pub k_buf: B::Buffer,
    pub v_buf: B::Buffer,
    pub attn_out: B::Buffer,
    pub o_proj_out: B::Buffer,
    pub gate_up_out: B::Buffer,
    pub silu_out: B::Buffer,
    pub mlp_out: B::Buffer,
}

/// Per-layer KV cache.
pub struct KvCache<B: Backend> {
    pub k: B::Buffer,
    pub v: B::Buffer,
    pub len: usize,
    pub capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Full model weights.
pub struct ModelWeights<B: Backend> {
    pub embed: B::Buffer,
    pub layers: Vec<LayerWeights<B>>,
    pub final_norm_w: B::Buffer,
    pub lm_head_w: B::Buffer,
}

/// The core abstraction over CUDA / Metal / CPU.
///
/// Key design: operations take a `&mut Self::Context` which accumulates work.
///   - **CPU**: Context is `()` — ops execute immediately.
///   - **Metal**: Context is a `CommandBuffer` — ops encode into it, flushed on `sync()`.
///   - **CUDA**: Context is a `CudaStream` — ops launch on the stream, synced on `sync()`.
///
/// `layer_forward` passes the context through all ops in a layer.
/// `ModelRunner` calls `sync()` only when it needs results (e.g., reading logits).
pub trait Backend: Send + Sync + Sized + 'static {
    type Buffer: Send + Sync;

    /// Execution context that accumulates GPU work.
    ///   - CPU: `()` (no-op, ops execute inline)
    ///   - Metal: wraps a CommandBuffer
    ///   - CUDA: wraps a CudaStream
    type Context;

    /// Create a new execution context (begin accumulating work).
    fn new_context() -> Self::Context;

    /// Flush accumulated work and wait for completion.
    /// CPU: no-op. Metal: commit + waitUntilCompleted. CUDA: stream sync.
    fn sync(ctx: &mut Self::Context);

    // ── GEMM ────────────────────────────────────────────────────────────

    fn gemm(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    );

    // ── Norms ───────────────────────────────────────────────────────────

    fn rms_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    );

    fn fused_add_rms_norm(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    );

    // ── Positional encoding ─────────────────────────────────────────────

    fn rope(
        ctx: &mut Self::Context,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        positions: &[u32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    );

    // ── Attention ───────────────────────────────────────────────────────

    fn decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        out: &mut Self::Buffer,
        kv_len: usize,
        cfg: &AttnConfig,
    );

    fn flash_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        q_len: usize,
        kv_len: usize,
        pos_offset: usize,
        cfg: &AttnConfig,
    );

    // ── Activations ─────────────────────────────────────────────────────

    fn silu_mul(
        ctx: &mut Self::Context,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    );

    // ── Element-wise ────────────────────────────────────────────────────

    fn add(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    );

    fn copy(ctx: &mut Self::Context, src: &Self::Buffer, dst: &mut Self::Buffer, len: usize);

    // ── Embedding ───────────────────────────────────────────────────────

    fn embedding_lookup(
        ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    );

    // ── Transformer-specific fused ops ─────────────────────────────────
    // These avoid CPU round-trips for data layout transformations.

    /// Split fused QKV [tokens, q_dim+2*kv_dim] into separate Q, K, V buffers.
    /// Q: [tokens, q_dim], K: [tokens, kv_dim], V: [tokens, kv_dim]
    fn split_qkv(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    );

    /// Split fused gate_up [tokens, 2*im] into gate [tokens, im] and up [tokens, im],
    /// then compute SiLU(gate) * up → out [tokens, im].
    fn fused_silu_mul_split(
        ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    );

    /// Per-head RMS norm in-place (QK-norm for Qwen3).
    /// data: [tokens, heads, head_dim], w: [head_dim]
    fn qk_norm(
        ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        w: &Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
    );

    /// Append new K/V to cache. Transposes from token-major to head-major.
    /// new_k/v: [tokens, nkv, hd] (token-major)
    /// cache k/v: [nkv, kv_len, hd] (head-major)
    fn kv_cache_append(
        ctx: &mut Self::Context,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cache_len: usize,
        new_k: &Self::Buffer,
        new_v: &Self::Buffer,
        new_tokens: usize,
        nkv: usize,
        hd: usize,
    ) -> (Self::Buffer, Self::Buffer);
    // Returns new cache buffers with appended data. Cache len updated by caller.

    /// Transpose [tokens, heads, dim] → [heads, tokens, dim]
    fn transpose_token_to_head(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    );

    /// Transpose [heads, tokens, dim] → [tokens, heads, dim]
    fn transpose_head_to_token(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    );

    /// residual[i] += x[i] (in-place)
    fn add_inplace(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        len: usize,
    );

    // ── Fused layer forward ────────────────────────────────────────────
    //
    // Optional: execute entire transformer layer as a single GPU batch.
    // Metal/CUDA can override to pipeline all ops in one command buffer.
    // Default: delegates to generic layer_forward (per-op dispatch).

    fn layer_forward_fused(
        ctx: &mut Self::Context,
        cfg: &TransformerConfig,
        weights: &super::LayerWeights<Self>,
        kv: &mut super::KvCache<Self>,
        scratch: &mut super::LayerScratch<Self>,
        residual: &mut Self::Buffer,
        positions: &[u32],
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        tokens: usize,
    ) {
        super::layer_forward::layer_forward::<Self>(
            ctx, cfg, weights, kv, scratch, residual, positions, cos, sin, tokens,
        );
    }

    // ── Buffer management (context-free) ────────────────────────────────

    fn alloc(len: usize) -> Self::Buffer;
    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32>;
    fn from_slice(data: &[f32]) -> Self::Buffer;
}
