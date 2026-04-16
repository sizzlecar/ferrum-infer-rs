//! Core Backend trait — the single abstraction over CUDA / Metal / CPU.

/// Configuration for RoPE (rotary position embeddings).
#[derive(Clone, Debug)]
pub struct RopeConfig {
    /// Base frequency (default 1_000_000 for Qwen3, 10_000 for LLaMA).
    pub theta: f64,
    /// Head dimension (full, not half).
    pub head_dim: usize,
    /// Max sequence length for precomputed cos/sin tables.
    pub max_seq_len: usize,
}

/// Configuration for attention dispatch.
#[derive(Clone, Debug)]
pub struct AttnConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Whether to apply causal mask.
    pub causal: bool,
    /// Softmax scale factor (typically 1/sqrt(head_dim)).
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
    /// Whether the model uses QK-norm (e.g., Qwen3 does, LLaMA doesn't).
    pub has_qk_norm: bool,
    /// Attention type.
    pub attn_type: AttnType,
    /// MLP type.
    pub mlp_type: MlpType,
}

/// Attention variant.
#[derive(Clone, Debug)]
pub enum AttnType {
    /// Standard grouped-query attention.
    Gqa,
    /// Sliding window attention (Mistral).
    SlidingWindow { window: usize },
}

/// MLP variant.
#[derive(Clone, Debug)]
pub enum MlpType {
    /// SwiGLU (gate + up projections, SiLU activation, down projection).
    SwiGlu,
    /// Mixture of Experts (future: DeepSeek-V3).
    Moe { num_experts: usize, top_k: usize },
}

/// Per-layer weights stored in backend-native buffers.
pub struct LayerWeights<B: Backend> {
    pub input_ln_w: B::Buffer,
    pub qkv_proj_w: B::Buffer,    // fused [q_dim + 2*kv_dim, hidden]
    pub o_proj_w: B::Buffer,
    pub post_ln_w: B::Buffer,
    pub gate_up_proj_w: B::Buffer, // fused [2*intermediate, hidden]
    pub down_proj_w: B::Buffer,
    // Optional QK-norm weights (Qwen3)
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
    /// Current number of tokens stored in cache.
    pub len: usize,
    /// Max tokens this cache can hold before reallocation.
    pub capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Full model weights (embedding + layers + final norm + lm_head).
pub struct ModelWeights<B: Backend> {
    pub embed: B::Buffer,
    pub layers: Vec<LayerWeights<B>>,
    pub final_norm_w: B::Buffer,
    pub lm_head_w: B::Buffer,
}

/// The core abstraction: all operations needed for a transformer forward pass.
///
/// Each method takes explicit dimensions — no shape metadata baked into buffers.
/// Buffers are raw contiguous memory; layout is row-major unless noted otherwise.
pub trait Backend: Send + Sync + 'static {
    /// The native buffer type for this backend.
    ///   - CUDA: `CudaSlice<f16>`
    ///   - Metal: `metal::Buffer`
    ///   - CPU: `Vec<f32>`
    type Buffer: Send + Sync;

    // ── GEMM ────────────────────────────────────────────────────────────

    /// C[m, n] = A[m, k] @ B[n, k]^T  (row-major, B transposed)
    ///
    /// This is the standard weight-projection GEMM:
    ///   - A = activations [tokens, hidden]
    ///   - B = weights [out_dim, hidden] (stored transposed)
    ///   - C = output [tokens, out_dim]
    fn gemm(
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    );

    // ── Norms ───────────────────────────────────────────────────────────

    /// RMS norm: out[i] = x[i] * w[i % dim] / sqrt(mean(x^2) + eps)
    ///
    /// x, out: [tokens * dim], w: [dim]
    fn rms_norm(
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    );

    /// Fused residual add + RMS norm:
    ///   residual[i] += x[i]
    ///   out[i] = residual[i] * w[i % dim] / sqrt(mean(residual^2) + eps)
    ///
    /// residual, x, out: [tokens * dim], w: [dim]
    fn fused_add_rms_norm(
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    );

    // ── Positional encoding ─────────────────────────────────────────────

    /// Apply rotary position embeddings to Q and K in-place.
    ///
    /// q: [tokens, num_heads, head_dim], k: [tokens, num_kv_heads, head_dim]
    /// cos, sin: precomputed tables [max_seq, head_dim/2]
    /// positions: [tokens] position index for each token
    fn rope(
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

    /// Single-query decode attention: Q[1, num_heads, head_dim] attends to
    /// full KV cache [kv_len, num_kv_heads, head_dim].
    ///
    /// Optimized for decode (M=1): warp-cooperative dot product, no
    /// intermediate score matrix materialization.
    fn decode_attention(
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        out: &mut Self::Buffer,
        kv_len: usize,
        cfg: &AttnConfig,
    );

    /// Full-sequence flash attention for prefill.
    ///
    /// Q: [batch, num_heads, q_len, head_dim]
    /// K: [batch, num_kv_heads, kv_len, head_dim]
    /// V: [batch, num_kv_heads, kv_len, head_dim]
    /// out: [batch, num_heads, q_len, head_dim]
    fn flash_attention(
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

    /// Fused SiLU(gate) * up: out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
    ///
    /// gate, up, out: [tokens * intermediate_size]
    fn silu_mul(
        gate: &Self::Buffer,
        up: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    );

    // ── Element-wise ────────────────────────────────────────────────────

    /// out[i] = a[i] + b[i]
    fn add(a: &Self::Buffer, b: &Self::Buffer, out: &mut Self::Buffer, len: usize);

    /// Copy src into dst.
    fn copy(src: &Self::Buffer, dst: &mut Self::Buffer, len: usize);

    // ── Embedding ───────────────────────────────────────────────────────

    /// Gather embedding vectors: out[i] = table[ids[i]]
    ///
    /// table: [vocab_size, dim], ids: token ids, out: [tokens * dim]
    fn embedding_lookup(
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    );

    // ── Buffer management ───────────────────────────────────────────────

    /// Allocate a zero-initialized buffer of `len` elements.
    fn alloc(len: usize) -> Self::Buffer;

    /// Read buffer contents back to CPU as f32 vec.
    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32>;

    /// Create a buffer from existing f32 data (upload to device if GPU).
    fn from_slice(data: &[f32]) -> Self::Buffer;
}
