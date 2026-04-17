//! Core Backend trait — the single abstraction over CUDA / Metal / CPU.

use ferrum_types::{FerrumError, Result};

/// Quantization flavour discriminator for `Backend::gemm_quant`.
///
/// Distinct schemes need distinct kernels. Carried as a parameter so the
/// Backend trait does not explode with one method per quantization type.
#[derive(Clone, Debug)]
pub enum QuantKind {
    /// GPTQ: group-wise int4/int8 with scales + zeros (asymmetric) + optional g_idx.
    Gptq {
        bits: u32,
        group_size: usize,
        desc_act: bool,
    },
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
    /// Sliding-window attention size (Mistral v0.1, Gemma).
    /// `0` = disabled (full causal attention).
    /// `w > 0` = each query position attends to the previous `w` KV positions
    ///            (still bounded by `causal` + `pos_offset + qi + 1` as the upper end).
    pub sliding_window: usize,
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
            sliding_window: 0,
        }
    }
}

// Note: `TransformerConfig` / `AttnType` / `MlpType` / `RopeConfig` used to
// live here when `ModelRunner` needed a generic model config. They're now
// per-model (e.g. `Qwen3Config` in `ferrum-models::models::qwen3`) so each
// model can carry exactly the architecture parameters it cares about.
// Backend trait stays model-agnostic.

/// Per-layer KV cache. Each model owns its own `Vec<KvCache<B>>` per sequence.
pub struct KvCache<B: Backend> {
    pub k: B::Buffer,
    pub v: B::Buffer,
    pub len: usize,
    pub capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
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

    /// Opaque per-backend GPTQ weight representation.
    ///   - CPU: dequantized f32 weights (run as regular GEMM)
    ///   - Metal: `()` — unsupported; `gemm_gptq` errors
    ///   - CUDA: `MarlinWeight` — pre-repacked tiles + permuted scales
    ///
    /// Each backend repacks raw GPTQ tensors (qweight/scales/qzeros, all
    /// i32/f16) into its preferred format at model load time, so inference
    /// doesn't pay the repack cost per forward pass.
    type GptqStore: Send + Sync;

    /// Create a new execution context (begin accumulating work).
    fn new_context() -> Self::Context;

    /// Flush accumulated work and wait for completion.
    /// CPU: no-op. Metal: commit + waitUntilCompleted. CUDA: stream sync.
    fn sync(ctx: &mut Self::Context);

    // ── Graph capture / replay (CUDA only) ──────────────────────────────
    //
    // Decode-loop optimization: eliminate per-kernel launch overhead by
    // capturing the full step as a CUDA graph and replaying. CPU/Metal
    // have no equivalent — defaults return `unsupported`.
    //
    // Flow per decode step:
    //   1. Caller: `set_decode_state(ctx, token, step)` — memcpy to dev bufs
    //   2. Try `replay_last_graph(ctx)`:
    //        - Ok(true):  graph replayed, skip eager forward
    //        - Ok(false): no captured graph yet, run eager
    //        - Err(_):    not supported, run eager
    //   3. If running eager and in capture window:
    //      - `set_dev_state_mode(ctx, true)` so kernels use _dyn variants
    //      - `begin_graph_capture(ctx)`
    //      - run forward
    //      - `end_graph_capture(ctx)` — stores graph on ctx internally
    //      - `set_dev_state_mode(ctx, false)` — restore scalar kernels

    /// Update per-step dynamic state (token id, step/pos). Fast (3x memcpy).
    fn set_decode_state(_ctx: &mut Self::Context, _token: u32, _step: u32) {}

    /// Toggle between scalar-arg kernels (normal) and `_dyn` kernels that
    /// read their dynamic scalar args from device memory (graph-friendly).
    fn set_dev_state_mode(_ctx: &mut Self::Context, _enable: bool) {}

    /// Begin stream capture. Subsequent kernel launches are recorded into
    /// a pending graph instead of executing eagerly.
    fn begin_graph_capture(_ctx: &mut Self::Context) -> Result<()> {
        Err(FerrumError::unsupported("graph capture not supported"))
    }

    /// End stream capture and install the captured graph as this context's
    /// "last graph" for future `replay_last_graph` calls.
    fn end_graph_capture(_ctx: &mut Self::Context) -> Result<()> {
        Err(FerrumError::unsupported("graph capture not supported"))
    }

    /// Replay the last captured graph. Returns `Ok(false)` if no graph
    /// is cached; caller should run eager.
    fn replay_last_graph(_ctx: &mut Self::Context) -> Result<bool> {
        Ok(false)
    }

    // ── GPTQ (INT4 quantization) ────────────────────────────────────────
    //
    // Two-step: load (once per weight) → gemm (per forward). The store
    // holds whatever backend-specific format is fastest; caller code
    // (GptqLinear) is dtype-agnostic.

    /// Repack raw GPTQ tensors into the backend's preferred format.
    /// Called once per layer at model load time.
    ///
    /// Inputs are host-side slices (CPU memory) — the loader reads from
    /// safetensors and hands them off; each backend uploads + repacks
    /// per its own strategy. `bits` is typically 4; `group_size` is
    /// typically 128.
    #[allow(clippy::too_many_arguments)]
    fn load_gptq(
        _qweight: &[i32],
        _scales: &[f32],
        _qzeros: &[i32],
        _g_idx: Option<&[i32]>,
        _bits: u32,
        _group_size: usize,
        _k: usize,
        _n: usize,
    ) -> Result<Self::GptqStore> {
        Err(FerrumError::unsupported(
            "load_gptq not implemented for this backend",
        ))
    }

    /// GEMM with pre-loaded GPTQ weights.
    /// `out[m, n] = a[m, k] @ dequant(weight)^T`
    fn gemm_gptq(
        _ctx: &mut Self::Context,
        _a: &Self::Buffer,
        _weight: &Self::GptqStore,
        _out: &mut Self::Buffer,
        _m: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "gemm_gptq not implemented for this backend",
        ))
    }

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

    // ── Attention ───────────────────────────────────────────────────────

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

    /// Multi-Head Latent Attention — DeepSeek V2 / V3's compressed-KV
    /// attention variant. Extension point only; no backend implements it
    /// yet. DeepSeek V3 landing in Phase D/E will fill this in.
    ///
    /// `q`: full Q `[batch, num_heads, q_len, head_dim]`
    /// `kv_compressed`: latent KV `[batch, kv_len, kv_lora_rank]`
    /// `kv_rope`: per-position rope-applied key heads `[batch, kv_len, qk_rope_head_dim]`
    /// `out`: `[batch, num_heads, q_len, head_dim]`
    #[allow(clippy::too_many_arguments)]
    fn mla_attention(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _kv_compressed: &Self::Buffer,
        _kv_rope: &Self::Buffer,
        _out: &mut Self::Buffer,
        _batch: usize,
        _q_len: usize,
        _kv_len: usize,
        _pos_offset: usize,
        _cfg: &AttnConfig,
        _kv_lora_rank: usize,
        _qk_rope_head_dim: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "mla_attention not implemented for this backend; required by \
             DeepSeek V2/V3 (Phase D/E)",
        ))
    }

    // ── Element-wise ────────────────────────────────────────────────────
    //
    // Models use `add_inplace` for residual updates and `copy_slice` for the
    // row-extraction step in prefill. Offset-free copy / non-inplace add are
    // not needed by the current Model-as-Code path; they can return later if
    // a model actually requires them.

    /// Copy `len` floats from `src[src_offset..]` to `dst[dst_offset..]`.
    ///
    /// Needed for Qwen3Model::prefill to pluck the last token's hidden state
    /// out of `residual[seq_len, h]` without round-tripping through host RAM.
    /// `Backend::copy` is the offset-free variant; `copy_slice` additionally
    /// supports non-zero source and destination offsets.
    fn copy_slice(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        src_offset: usize,
        dst: &mut Self::Buffer,
        dst_offset: usize,
        len: usize,
    );

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

    /// Transpose [heads, tokens, dim] → [tokens, heads, dim].
    /// Called after `flash_attention` to restore token-major layout for O-proj.
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

    /// Broadcast bias add: `data[r, c] += bias[c]` for every row.
    /// Required by Bert / Clip / Whisper whose linear projections carry a bias.
    fn add_bias(
        ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    );

    /// Full LayerNorm (mean + variance normalisation + affine), distinct from
    /// the `rms_norm` used by Llama-family decoders.
    ///   `out[r, c] = ((x[r, c] - mean) / sqrt(var + eps)) * gamma[c] + beta[c]`
    /// Where `mean` and `var` are reduced over the last dim (cols).
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    );

    /// Element-wise GELU activation (erf-based, matches PyTorch default).
    fn gelu(ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize);

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
    fn all_reduce(_ctx: &mut Self::Context, _buf: &mut Self::Buffer, _len: usize, _op: ReduceOp) {
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
    fn broadcast(_ctx: &mut Self::Context, _buf: &mut Self::Buffer, _len: usize, _src_rank: usize) {
        // single-rank: no-op
    }
}
