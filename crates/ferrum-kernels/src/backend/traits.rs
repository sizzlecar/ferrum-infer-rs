//! Core Backend trait — the single abstraction over CUDA / Metal / CPU.

use ferrum_types::{FerrumError, Result};

/// Quantization flavour discriminator for `Backend::gemm_quant`.
///
/// Distinct schemes need distinct kernels. Carried as a parameter so the
/// Backend trait does not explode with one method per quantization type.
#[derive(Clone, Debug)]
pub enum QuantKind {
    /// GPTQ: group-wise int4/int8 with scales + zeros (asymmetric) + optional g_idx.
    Gptq { bits: u32, group_size: usize, desc_act: bool },
    /// AWQ: activation-aware int4 with scales + zeros, different packing from GPTQ.
    Awq { bits: u32, group_size: usize },
    /// GGUF: one of k-quants / legacy quants, fully specified by the inner type.
    Gguf { quant_type: GgufQuantType },
}

/// GGUF quantization sub-type (expand as kernels are added).
#[derive(Clone, Copy, Debug)]
pub enum GgufQuantType {
    Q4_0,
    Q4_1,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

/// Packed quantized weight buffers passed to `Backend::gemm_quant`.
///
/// Not every field is used by every `QuantKind` — e.g. GGUF packs scales
/// inside `qweight`, so `scales` / `zeros` may be dummies. The Backend
/// implementation is expected to validate the shape for the kind it handles.
pub struct QuantWeights<'a, B: Backend> {
    pub qweight: &'a B::Buffer,
    pub scales: Option<&'a B::Buffer>,
    pub zeros: Option<&'a B::Buffer>,
    pub g_idx: Option<&'a B::Buffer>,
}

/// Collective-op reduction kind for TP all_reduce.
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}


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
    /// Stride (in rows) between head blocks in the KV buffer.
    /// `0` means contiguous (use `kv_len`, legacy behaviour).
    /// Set to `cache_capacity` when flashing against a pre-allocated cache
    /// that only has `kv_len` valid slots out of `cache_capacity`.
    pub kv_seq_stride: usize,
}

impl Default for AttnConfig {
    fn default() -> Self {
        Self {
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            causal: false,
            scale: 1.0,
            kv_seq_stride: 0,
        }
    }
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

    /// Fused QK-norm + RoPE + transpose-to-head-major.
    ///
    /// `mode` selects the operation:
    ///   0 = transpose only (typical for V, which needs no norm and no RoPE)
    ///   1 = per-head RMS norm + RoPE + transpose  (Q/K with QK-norm, Qwen3)
    ///   2 = RoPE + transpose                       (Q/K without QK-norm, Llama/Mistral)
    ///
    /// input:   `[tokens, heads, head_dim]`  (token-major, output of split_qkv)
    /// output:  `[heads, tokens, head_dim]`  (head-major, ready for flash_attn / kv_cache_append)
    ///
    /// `pos_offset` is the position of token 0 (decode uses current seq len;
    /// prefill uses 0). Within the batch, positions are taken as `pos_offset + i`.
    ///
    /// This is the primary attention-input preparation op. Backends that have a
    /// fused kernel (Metal's `qk_norm_rope_transpose_f32`) will be dramatically
    /// faster than composing norm + rope + transpose separately; the CPU
    /// fallback lowers to the individual ops.
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        mode: i32,
    );

    /// (Legacy) Append new K/V to cache. Accepts token-major input and
    /// allocates fresh cache buffers sized for `cache_len + new_tokens`.
    /// Used by the generic `layer_forward` path (CpuBackend default impl).
    ///
    /// `new_k/v`: `[tokens, nkv, hd]` (token-major)
    /// `cache k/v`: `[nkv, cache_len, hd]` (head-major)
    /// Returns fresh `(cache_k, cache_v)` with appended data; caller updates
    /// `kv.len` on its side. Realloc-per-call makes this O(cache_len) per
    /// layer and is going away once Model-as-Code models use the pre-allocated
    /// `kv_cache_append_head_major` path below.
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

    /// Append new K/V into a pre-allocated head-major cache buffer.
    ///
    /// `cache_k` / `cache_v`: `[nkv, capacity, hd]` (head-major, pre-allocated)
    /// `new_k_head_major` / `new_v_head_major`: `[nkv, new_tokens, hd]`
    ///   — produced directly by `qk_norm_rope`, no extra transpose needed.
    ///
    /// In-place append at slot `[nkv, cache_len..cache_len+new_tokens, hd]`.
    /// Caller owns `cache_len` bookkeeping.
    #[allow(clippy::too_many_arguments)]
    fn kv_cache_append_head_major(
        ctx: &mut Self::Context,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cache_len: usize,
        cache_capacity: usize,
        new_k_head_major: &Self::Buffer,
        new_v_head_major: &Self::Buffer,
        new_tokens: usize,
        nkv: usize,
        hd: usize,
    );

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

    // ── Quantized GEMM (Phase A3 stubs) ─────────────────────────────────
    //
    // Backends override the kinds they actually support (e.g. Metal will
    // implement Gptq first; CUDA will implement Gptq + Awq via Marlin).
    // Default impl returns an `unsupported` error so missing kernels surface
    // as clean runtime errors instead of silent wrong output.

    /// GEMM with packed-quantized B matrix. `m`/`n`/`k` describe the dense
    /// equivalent (`[m,n] = [m,k] @ [k,n]^T`).
    #[allow(clippy::too_many_arguments)]
    fn gemm_quant(
        _ctx: &mut Self::Context,
        _a: &Self::Buffer,
        _weights: &QuantWeights<'_, Self>,
        _out: &mut Self::Buffer,
        _m: usize,
        _n: usize,
        _k: usize,
        kind: &QuantKind,
    ) -> Result<()> {
        Err(FerrumError::unsupported(format!(
            "gemm_quant({kind:?}) not implemented for this backend"
        )))
    }

    // ── TP collective ops (Phase A3 stubs) ──────────────────────────────
    //
    // Default impl is single-rank no-op: `world_size = 1`, `rank = 0`, and
    // the collective ops are identity. Multi-GPU backends (future
    // CudaBackend + NCCL) override these. Model code can call
    // `B::all_reduce_sum(...)` unconditionally; single-GPU paths pay zero.

    fn world_size(_ctx: &Self::Context) -> usize {
        1
    }
    fn rank(_ctx: &Self::Context) -> usize {
        0
    }
    fn all_reduce(
        _ctx: &mut Self::Context,
        _buf: &mut Self::Buffer,
        _len: usize,
        _op: ReduceOp,
    ) {
        // single-rank: no-op
    }
    fn all_gather(
        _ctx: &mut Self::Context,
        _local: &Self::Buffer,
        _global: &mut Self::Buffer,
        _local_len: usize,
    ) {
        // single-rank: no-op (caller is expected to handle the degenerate
        // case or arrange for `local == global`)
    }
    fn broadcast(
        _ctx: &mut Self::Context,
        _buf: &mut Self::Buffer,
        _len: usize,
        _src_rank: usize,
    ) {
        // single-rank: no-op
    }
}
