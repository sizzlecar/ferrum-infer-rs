//! Core Backend trait — the single abstraction over CUDA / Metal / CPU.

use ferrum_types::{FerrumError, Result};

pub use super::capabilities::{
    BackendCollective, BackendGraph, BackendMoeFused, BackendQuantGguf, BackendQuantMarlin,
};
pub use super::types::MoeRouting;
use super::types::{AttnConfig, KvCacheQuant, SrcDtype};

/// Maximum decode-graph layer count. Per-layer call sites that share
/// graph-captured host staging arrays use this as the stride between
/// distinct slots. CUDA-only invariant (other backends ignore the
/// `slot` argument); 64 covers all current LLM families up to and
/// including Llama-3-70B (80 layers — but 70B doesn't run on a single
/// 4090 anyway, so 64 is safe in practice for v0.2).
pub const MAX_LAYERS_FOR_GRAPH: usize = 64;

// Note: `TransformerConfig` / `AttnType` / `MlpType` / `RopeConfig` used to
// live here when `ModelRunner` needed a generic model config. They're now
// per-model (e.g. `Qwen3Config` in `ferrum-models::models::qwen3`) so each
// model can carry exactly the architecture parameters it cares about.
// Backend trait stays model-agnostic.

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

    /// GPU-side timer scoped to this backend. See `super::timer` —
    /// CPU: `Instant`; Metal: sync-wrap; CUDA: `cuEvent`.
    /// PLAYBOOK § 1.1.
    type Timer: super::timer::BackendTimer<Self>;

    /// Factory for `Self::Timer` — exists so call sites that have a
    /// `<B: Backend>` parameter can spawn a timer without importing the
    /// concrete impl. PLAYBOOK § 1.2.
    fn make_timer() -> Self::Timer;

    /// Opaque per-backend GPTQ weight representation.
    ///   - CPU: dequantized f32 weights (run as regular GEMM)
    ///   - Metal: `()` — unsupported; `gemm_gptq` errors
    // Note (Phase 3e/4 + Phase C):
    // - `type QuantStore` (GGUF k-quant storage) was removed in Phase 3e/4
    //   — stacked-expert MoE GGUF goes through Box<dyn StackedExpertGgufLinear<Self>>
    //   returned by `load_quant_experts`.
    // - `type GptqStore` (Marlin/dequant GPTQ storage) was removed in Phase C
    //   step 4e — stacked-expert Marlin MoE goes through
    //   Arc<dyn MarlinExpertStack<Self>> returned by `load_gptq_stacked`,
    //   and single-tensor GPTQ goes through Box<dyn Linear<Self>> returned
    //   by `load_gptq`. Adding a new Marlin-capable backend is purely a
    //   new MarlinExpertStack<NewBackend> impl — no Backend trait edits.

    /// Create a new execution context (begin accumulating work).
    fn new_context() -> Self::Context;

    /// Run `body` while binding context-free backend operations to an
    /// explicit device ordinal when the backend supports multi-device scopes.
    ///
    /// Most backends have no per-ordinal concept and use the default no-op
    /// implementation. CUDA overrides this once its stream/context caches are
    /// device-keyed, allowing layer-split stages to load and execute on their
    /// selected GPU instead of relying on process-global defaults.
    fn with_device_ordinal<R>(_device_ordinal: Option<usize>, body: impl FnOnce() -> R) -> R {
        body()
    }

    /// Whether [`Self::with_device_ordinal`] actually switches backend
    /// execution to the requested ordinal.
    fn supports_device_ordinal_scope() -> bool {
        false
    }

    /// Flush accumulated work and wait for completion.
    /// CPU: no-op. Metal: commit + waitUntilCompleted. CUDA: stream sync.
    fn sync(ctx: &mut Self::Context);

    /// Whether the backend context is currently inside a graph-capture window.
    ///
    /// Synchronizing a CUDA stream while capture is active raises
    /// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`; diagnostic probes that time
    /// sub-ops with explicit sync boundaries must skip those boundaries while
    /// this returns true. Backends without graph capture use the default.
    fn graph_capture_in_flight(_ctx: &Self::Context) -> bool {
        false
    }

    /// Prepare pending GPU work for a following host readback.
    ///
    /// Most backends either execute eagerly or synchronize as part of their
    /// device-to-host copy. Metal shared-buffer reads use the CPU pointer
    /// directly, so Metal must flush its command buffer before `to_vec`.
    fn sync_before_host_readback(_ctx: &mut Self::Context) {}

    /// Byte width of buffers returned by [`Self::alloc`].
    ///
    /// CUDA activation scratch is fp16, while Metal and CPU scratch are fp32.
    /// Generic model code uses this for byte offsets into batched scratch
    /// buffers without checking concrete backend types.
    fn activation_elem_size_bytes() -> usize {
        std::mem::size_of::<half::f16>()
    }

    /// Whether `LlamaFamilyModel::decode_batch_internal` may use its optimized
    /// batched decode path on this backend.
    ///
    /// Backends that do not yet produce correct follow-up logits under
    /// concurrent dense decode should override this to force the per-item
    /// fallback until the optimized path is fixed.
    fn supports_llama_family_batched_decode() -> bool {
        true
    }

    // Graph capability moved to the `BackendGraph` supertrait at the end
    // of this file. CUDA implements its overrides; Metal/CPU inherit
    // unsupported defaults via empty `impl BackendGraph for X {}` blocks.

    // ── GPTQ (INT4 quantization) ────────────────────────────────────────
    //
    // Two-step: load (once per weight) → gemm (per forward). The store
    // holds whatever backend-specific format is fastest; caller code
    // (GptqLinear) is dtype-agnostic.

    /// Zero the first `len` elements of a Self::Buffer. CUDA path uses
    /// cuMemsetD16Async; default returns unsupported.
    fn zero_buffer(_ctx: &mut Self::Context, _buf: &mut Self::Buffer, _len: usize) -> Result<()> {
        Err(FerrumError::unsupported(
            "zero_buffer not implemented for this backend",
        ))
    }

    /// Phase D step 2+3: unified typed allocator. Replaces per-dtype
    /// `alloc_u32` / `alloc_typed_i32` / etc. The buffer is dtype-
    /// tagged at the wrapper level (`CudaBuf::U32`, `MetalBuf` with
    /// `Dtype::U32`, `CpuBuf::U32`), so reads/writes through `.as_<T>()`
    /// accessors get the correct byte count automatically.
    fn alloc_typed(dtype: super::Dtype, n: usize) -> Self::Buffer;

    /// Upload typed host data — replaces `from_slice_i32` /
    /// `from_slice_u32` etc. The host element type `T` carries its
    /// `Dtype` via the `HostDtype` marker so dispatch in the impl
    /// is a one-line `match T::DTYPE`.
    fn from_slice_typed<T: super::HostDtype>(data: &[T]) -> Self::Buffer;

    /// In-place typed write — replaces `write_u32` / `write_i32_into`
    /// / `write_f32_into`. The buffer must already be dtype-tagged
    /// matching `T::DTYPE` (typically alloc'd via `alloc_typed` or
    /// `from_slice_typed`).
    fn write_typed<T: super::HostDtype>(
        ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        data: &[T],
    );

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

    /// Recurrent gated DeltaNet update used by linear-attention layers.
    ///
    /// Layouts are token-major:
    /// - `query` / `key`: `[tokens, key_heads, key_dim]`
    /// - `value` / `out`: `[tokens, value_heads, value_dim]`
    /// - `g` / `beta`: `[tokens, value_heads]`
    /// - `initial_state` / `final_state`: `[value_heads, value_dim, key_dim]`
    ///
    /// Backends may require these buffers to be F32. CUDA currently provides
    /// the native W3 path; unsupported backends should use the model-level
    /// reference path instead of silently round-tripping through the host.
    #[allow(clippy::too_many_arguments)]
    fn recurrent_gated_delta_rule_f32(
        _ctx: &mut Self::Context,
        _query: &Self::Buffer,
        _key: &Self::Buffer,
        _value: &Self::Buffer,
        _g: &Self::Buffer,
        _beta: &Self::Buffer,
        _initial_state: &Self::Buffer,
        _out: &mut Self::Buffer,
        _final_state: &mut Self::Buffer,
        _tokens: usize,
        _key_heads: usize,
        _value_heads: usize,
        _key_dim: usize,
        _value_dim: usize,
        _use_qk_l2norm: bool,
        _scale: f32,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "recurrent_gated_delta_rule_f32 not implemented for this backend",
        ))
    }

    /// Prepare a gated-Delta linear-attention block:
    /// depthwise causal conv + SiLU over `mixed_qkv_raw`, split into Q/K/V,
    /// and compute GDN gates `g` and `beta`.
    #[allow(clippy::too_many_arguments)]
    fn linear_attention_prepare_f32(
        _ctx: &mut Self::Context,
        _mixed_qkv_raw: &Self::Buffer,
        _conv_weight: &Self::Buffer,
        _a_raw: &Self::Buffer,
        _b_raw: &Self::Buffer,
        _a_log: &Self::Buffer,
        _dt_bias: &Self::Buffer,
        _query: &mut Self::Buffer,
        _key: &mut Self::Buffer,
        _value: &mut Self::Buffer,
        _g: &mut Self::Buffer,
        _beta: &mut Self::Buffer,
        _tokens: usize,
        _key_heads: usize,
        _value_heads: usize,
        _key_dim: usize,
        _value_dim: usize,
        _conv_kernel: usize,
        _apply_qk_l2norm: bool,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "linear_attention_prepare_f32 not implemented for this backend",
        ))
    }

    /// Decode-time gated-Delta linear-attention preparation for one token.
    ///
    /// This is the stateful counterpart of [`Self::linear_attention_prepare_f32`]:
    /// it reads `[conv_channels, conv_kernel - 1]` causal-conv state, appends the
    /// current raw QKV token, writes the next conv state, then emits Q/K/V and
    /// GDN gates for the current token. The layout mirrors vLLM's Qwen GDN
    /// `conv_state` + temporal-state split.
    #[allow(clippy::too_many_arguments)]
    fn linear_attention_decode_prepare_f32(
        _ctx: &mut Self::Context,
        _mixed_qkv_raw: &Self::Buffer,
        _conv_weight: &Self::Buffer,
        _conv_state: &Self::Buffer,
        _a_raw: &Self::Buffer,
        _b_raw: &Self::Buffer,
        _a_log: &Self::Buffer,
        _dt_bias: &Self::Buffer,
        _query: &mut Self::Buffer,
        _key: &mut Self::Buffer,
        _value: &mut Self::Buffer,
        _g: &mut Self::Buffer,
        _beta: &mut Self::Buffer,
        _next_conv_state: &mut Self::Buffer,
        _key_heads: usize,
        _value_heads: usize,
        _key_dim: usize,
        _value_dim: usize,
        _conv_kernel: usize,
        _apply_qk_l2norm: bool,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "linear_attention_decode_prepare_f32 not implemented for this backend",
        ))
    }

    /// Gated RMSNorm used after recurrent DeltaNet core:
    /// `out = rms_norm(core) * weight * silu(z)`.
    #[allow(clippy::too_many_arguments)]
    fn gated_rms_norm_f32(
        _ctx: &mut Self::Context,
        _core: &Self::Buffer,
        _z: &Self::Buffer,
        _weight: &Self::Buffer,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _heads: usize,
        _dim: usize,
        _eps: f32,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "gated_rms_norm_f32 not implemented for this backend",
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

    /// Device-buffer variant of `embedding_lookup` for graph-capturable
    /// MoE routing — the gather step before phase-1 GEMM in
    /// `moe_forward_bucketed`. The host-slice `embedding_lookup` does
    /// `clone_htod(ids)` internally, which records stale host pointers
    /// under CUDA Graph capture replay.
    ///
    /// `ids: &Self::Buffer` must be a device I32 buffer of `batch`
    /// elements (e.g. `Qwen3MoeScratch::route_packed_idx_dev`).
    /// `batch` is passed explicitly since a typed CudaBuf carries
    /// its element count but the caller often wants a partial gather.
    ///
    /// Default impl: round-trip via `to_vec` + dispatch the host-slice
    /// variant. CUDA overrides.
    fn embedding_lookup_dev(
        ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        dim: usize,
    ) {
        // Default: round-trip. CUDA overrides with a direct device-arg
        // kernel launch (no clone_htod).
        let ids_host_f32 = Self::to_vec(ids, batch);
        let ids_host_u32: Vec<u32> = ids_host_f32.iter().map(|x| x.to_bits()).collect();
        Self::embedding_lookup(ctx, table, &ids_host_u32, out, dim);
    }

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

    /// GeGLU variant of [`Backend::fused_silu_mul_split`]:
    /// gelu_tanh(gate) * up → out. Matches HF `gelu_pytorch_tanh`
    /// (Gemma family MLP). Panics by default: wire a real kernel on any
    /// backend that loads a GeGLU model.
    fn fused_gelu_tanh_mul_split(
        _ctx: &mut Self::Context,
        _gate_up: &Self::Buffer,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _im: usize,
    ) {
        panic!("fused_gelu_tanh_mul_split not implemented for this backend");
    }

    /// `buf[i] *= scale` over the first `len` elements. Gemma-family
    /// embedding scaling (×√hidden_size on residual-stream entry).
    /// Default round-trips through host memory — correct but slow;
    /// override on backends that serve Gemma models.
    fn scale_inplace(ctx: &mut Self::Context, buf: &mut Self::Buffer, scale: f32, len: usize) {
        Self::sync(ctx);
        let mut v = Self::to_vec(buf, len);
        for x in v.iter_mut() {
            *x *= scale;
        }
        *buf = Self::from_slice(&v);
    }

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

    /// Q/K preparation variant for Qwen3.5 full attention.
    ///
    /// `input_stride` is the per-token feature width in `input`, and
    /// `input_offset` is the first feature for this projection inside each
    /// token row. This lets Q read the first `num_heads * head_dim` slice from
    /// Qwen3.5's gated `q_proj` output while leaving the attention gate slice
    /// in place for a later device-side post-op.
    ///
    /// `rope_dim` may be smaller than `head_dim`; dimensions outside
    /// `rope_dim` are normalized and copied but not rotated. `mode` follows
    /// [`Backend::qk_norm_rope`]: 0 transpose only, 1 RMSNorm+RoPE, 2 RoPE
    /// only, 3 RMSNorm+interleaved RoPE for Qwen3.5's mrope layout.
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope_partial(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        input_stride: usize,
        input_offset: usize,
        input_head_stride: usize,
        pos_offset: usize,
        eps: f32,
        mode: i32,
    ) -> Result<()> {
        if rope_dim == head_dim
            && input_stride == heads * head_dim
            && input_offset == 0
            && input_head_stride == head_dim
            && mode != 3
        {
            Self::qk_norm_rope(
                ctx, input, norm_w, cos, sin, output, tokens, heads, head_dim, pos_offset, eps,
                mode,
            );
            return Ok(());
        }
        Err(FerrumError::unsupported(
            "qk_norm_rope_partial not implemented for this backend",
        ))
    }

    /// Apply Qwen3.5 attention output gate in place. Gated Qwen3.5 full
    /// attention stores q_proj rows as per-head `[query, gate]` slices, so
    /// `context[token, head, dim] *= sigmoid(q_proj[token, head, head_dim + dim])`.
    ///
    /// The CUDA implementation is a single device kernel. Backends that do not
    /// implement it must fail rather than silently copying data through host
    /// memory on product paths.
    fn qwen35_apply_attention_gate(
        _ctx: &mut Self::Context,
        _context: &mut Self::Buffer,
        _query_raw: &Self::Buffer,
        _tokens: usize,
        _q_total: usize,
        _q_proj_total: usize,
        _head_dim: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "qwen35_apply_attention_gate not implemented for this backend",
        ))
    }

    /// Apply one scalar gate per token to a token-major hidden buffer:
    /// `values[token, dim] *= sigmoid(gate[token])`.
    fn qwen35_apply_token_gate(
        _ctx: &mut Self::Context,
        _values: &mut Self::Buffer,
        _gate: &Self::Buffer,
        _tokens: usize,
        _hidden_size: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "qwen35_apply_token_gate not implemented for this backend",
        ))
    }

    /// Batched kv_cache_append across M caches in one launch. Each item
    /// writes its (head-major) K-or-V row into its own cache at offset
    /// read from `cache_lens[i]`. Replaces M sequential
    /// `kv_cache_append_head_major` calls with a single dispatch.
    ///
    /// `new_data` layout: `[m, nkv, hd]` item-major (each item's slice
    /// is contiguous, identical to the `k/v_normed_batched` produced by
    /// `qk_norm_rope_batched_per_item`).
    /// `caches`: per-cache `[nkv, capacity, hd]` head-major.
    /// `cache_lens`: device buffer (u32 storage, length ≥ m). Caller
    /// fills via `B::write_u32_into` BEFORE the call. Required for
    /// CUDA-graph capture: the kernel reads from this stable device
    /// buffer, so a captured graph can be replayed with new lens by
    /// just rewriting the buffer between launches.
    fn kv_cache_append_batched_per_cache(
        _ctx: &mut Self::Context,
        _caches: &[&Self::Buffer],
        _new_data: &Self::Buffer,
        _cache_lens: &Self::Buffer,
        _capacity: usize,
        _m: usize,
        _nkv: usize,
        _hd: usize,
        _slot: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "kv_cache_append_batched_per_cache not implemented for this backend",
        ))
    }

    /// Batched flash_attention across M decode caches in one launch.
    /// Replaces the per-item `flash_attention(q_len=1, ...)` × M
    /// loop in the non-paged batched-decode path.
    ///
    /// API takes Vec<&Buffer> for the per-cache K/V buffers (each
    /// `[nkv, capacity, hd]` head-major) plus host-side `kv_lens`.
    /// Backends that implement it must extract per-cache device
    /// pointers, build the device arrays the kernel needs, and launch
    /// one kernel covering all M items.
    ///
    /// `q` layout: [m, nq, hd] item-major (matches the
    /// `qk_norm_rope_batched_per_item` output for q_len=1).
    /// `out` layout: [m, nq, hd] item-major — written directly into
    /// the caller's batched attn_out buffer, no per-item copy needed.
    ///
    /// CUDA-only for now (kernel `batched_decode_attention` exists in
    /// `kernels/batched_decode_attention.cu`).
    /// `kv_lens`: device buffer (u32 storage, length ≥ m) — same
    /// design as `kv_cache_append_batched_per_cache::cache_lens`.
    /// `sliding_window`: common decode window for every item; `0` means
    /// full causal attention, `w > 0` means each item attends only to the
    /// last `w` valid KV positions.
    fn flash_attention_batched_per_cache(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_caches: &[&Self::Buffer],
        _v_caches: &[&Self::Buffer],
        _kv_lens: &Self::Buffer,
        _out: &mut Self::Buffer,
        _nq: usize,
        _nkv: usize,
        _hd: usize,
        _scale: f32,
        _max_valid_kv: usize,
        _capacity: usize,
        _sliding_window: usize,
        _slot: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "flash_attention_batched_per_cache not implemented for this backend",
        ))
    }

    /// Batched per-item-position variant of `qk_norm_rope` for the
    /// non-paged batched-decode path. Each of the `m` items has its own
    /// absolute RoPE position (read from a device i32 buffer of length
    /// `m`). Layout is item-major in *both* input and output:
    ///
    ///   input  [m, heads, head_dim]
    ///   output [m, heads, head_dim]   (no head-major transpose)
    ///
    /// Item-major output keeps the per-item flash_attention slice
    /// contiguous (`output[i * heads * head_dim ..]` is item i's whole
    /// Q tensor in head-major-equivalent layout for q_len=1).
    ///
    /// Replaces the M sequential single-item launches in the existing
    /// `forward_layer_batched_decode` path with one batched dispatch.
    /// CUDA-only for now; other backends fall through to the default
    /// `unsupported` and the caller falls back to the per-item loop.
    fn qk_norm_rope_batched_per_item(
        _ctx: &mut Self::Context,
        _input: &Self::Buffer,
        _norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _output: &mut Self::Buffer,
        _positions: &Self::Buffer,
        _m: usize,
        _heads: usize,
        _head_dim: usize,
        _eps: f32,
        _mode: i32,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "qk_norm_rope_batched_per_item not implemented for this backend",
        ))
    }

    /// Fused split-QKV + QK-norm + RoPE + head-major transpose.
    ///
    /// Single-dispatch replacement for the (`split_qkv` → 3× `qk_norm_rope`)
    /// chain on the decode-attention prelude. Reads the linear-layer
    /// fused-QKV output once and writes head-major Q/K/V directly into
    /// attention scratch.
    ///
    /// `qkv` layout: `[tokens, q_heads*hd + 2*kv_heads*hd]`.
    /// `q_out`: `[q_heads, tokens, hd]`. `k_out`/`v_out`: `[kv_heads, tokens, hd]`.
    /// `qk_mode`: 1 = norm + half-split RoPE for Q/K (Qwen3 with QK-norm),
    ///            2 = half-split RoPE only for Q/K,
    ///            3 = interleaved RoPE only for Q/K (GGUF LLaMA / llama.cpp layout).
    /// V always falls through to transpose-only.
    ///
    /// Default returns Unsupported. Backends that implement it are
    /// expected to be dramatically faster than the four-dispatch chain.
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope(
        _ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _k_out: &mut Self::Buffer,
        _v_out: &mut Self::Buffer,
        _tokens: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _qk_mode: i32,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "split_qkv_norm_rope not implemented for this backend",
        ))
    }

    /// Variant of [`Backend::split_qkv_norm_rope`] that writes the new
    /// K and V directly into pre-allocated head-major KV cache buffers
    /// at slot `[kv_heads, cache_len .. cache_len + tokens, hd]`.
    /// Eliminates the trailing `kv_cache_append_head_major` dispatch on
    /// the decode hot path. Q still lands in per-token head-major
    /// scratch (flash-attention reads it as the query).
    ///
    /// Default returns Unsupported. Backends without the fused kernel
    /// can keep using `split_qkv_norm_rope` + `kv_cache_append_head_major`.
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_cache(
        _ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _tokens: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _qk_mode: i32,
        _cache_len: usize,
        _cache_capacity: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "split_qkv_norm_rope_into_cache not implemented for this backend",
        ))
    }

    // Phase D step 2: alloc_u32 / write_u32 deleted. Callers use the
    // unified `alloc_typed(Dtype::U32, n)` + `write_typed(&[u32])` API
    // declared above.

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

    /// Inverse of `transpose_head_to_token`: [tokens, heads, dim] →
    /// [heads, tokens, dim]. Used by the CUDA `paged_decode_attention`
    /// wrapper to convert `paged_varlen_attention`'s token-major output
    /// back to the head-major layout that Qwen3MoeModel expects.
    /// Default panics — backends without a paged-KV CUDA path don't
    /// hit this code.
    fn transpose_token_to_head(
        _ctx: &mut Self::Context,
        _src: &Self::Buffer,
        _dst: &mut Self::Buffer,
        _tokens: usize,
        _heads: usize,
        _dim: usize,
    ) {
        panic!("transpose_token_to_head not implemented for this backend");
    }

    /// residual[i] += x[i] (in-place)
    fn add_inplace(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        len: usize,
    );

    /// `dst[i] += scale * src[i]` — scalar-broadcast scaled add, in place.
    ///
    /// MoE per-token combine writes `out[b] += weight_k * expert_k(x[b])`
    /// for each top-K expert; this primitive is the per-call accumulate.
    /// Backends without a dedicated kernel can fall back to the default
    /// implementation, which round-trips through host memory — correct,
    /// but slow on a hot path. Override on any backend you actually
    /// dispatch MoE on.
    fn scaled_add_inplace(
        _ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        src: &Self::Buffer,
        scale: f32,
        len: usize,
    ) {
        let mut dst_v = Self::to_vec(dst, len);
        let src_v = Self::to_vec(src, len);
        for i in 0..len {
            dst_v[i] += scale * src_v[i];
        }
        // Move the new buffer into the slot pointed to by `dst`. Safe
        // because `Self::Buffer: Send + Sync` and the old buffer is
        // dropped here when overwritten.
        *dst = Self::from_slice(&dst_v);
    }

    /// Strided variant of [`Backend::fused_silu_mul_split`] for the
    /// bucketed MoE path: reads `gate_up` rows starting at
    /// `in_row_offset`, writes `out` rows starting at `out_row_offset`.
    #[allow(clippy::too_many_arguments)]
    fn fused_silu_mul_split_strided(
        _ctx: &mut Self::Context,
        _gate_up: &Self::Buffer,
        _in_row_offset: usize,
        _out: &mut Self::Buffer,
        _out_row_offset: usize,
        _tokens: usize,
        _intermediate: usize,
    ) {
        unimplemented!("fused_silu_mul_split_strided default impl missing");
    }

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

    fn write_f32_to_activation(ctx: &mut Self::Context, dst: &mut Self::Buffer, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        let src = Self::from_slice(data);
        Self::copy_slice(ctx, &src, 0, dst, 0, data.len());
    }

    /// Convert a typed F32 device buffer into the backend activation dtype.
    ///
    /// CUDA activations are FP16 for tensor-core/Marlin kernels, while the
    /// Qwen3.5 gated-Delta core keeps recurrent math in F32. Backends with
    /// non-F32 activations should override this with a device-side conversion.
    fn f32_to_activation(
        ctx: &mut Self::Context,
        input_f32: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    ) {
        Self::sync(ctx);
        let values = Self::to_vec(input_f32, len);
        Self::write_f32_to_activation(ctx, out, &values);
    }

    /// Whether this backend can keep Gemma-style sandwich residuals in a
    /// device-side F32 shadow while continuing to feed FP16 activations into
    /// projection kernels. The default is false so existing CPU/Metal paths
    /// keep their current host-side fallback behavior.
    fn supports_device_f32_residual_shadow() -> bool {
        false
    }

    /// Copy an activation buffer into a typed F32 shadow buffer.
    fn activation_to_f32_shadow(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst_f32: &mut Self::Buffer,
        len: usize,
    ) {
        let data = Self::to_vec(src, len);
        Self::write_typed::<f32>(ctx, dst_f32, &data);
    }

    /// RMSNorm an activation buffer and write the result into a typed F32
    /// scratch buffer. Used for Gemma post-attn/post-ffn branch norms.
    fn rms_norm_activation_to_f32(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        eps: f32,
        out_f32: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let input_h = Self::to_vec(input, tokens * dim);
        let weight_h = Self::to_vec(weight, dim);
        let mut out = vec![0.0f32; tokens * dim];
        for row in 0..tokens {
            let offset = row * dim;
            let mut variance = 0.0f32;
            for i in 0..dim {
                let x = input_h[offset + i];
                variance += x * x;
            }
            let inv_rms = (variance / dim as f32 + eps).sqrt().recip();
            for i in 0..dim {
                out[offset + i] = input_h[offset + i] * inv_rms * weight_h[i];
            }
        }
        Self::write_typed::<f32>(ctx, out_f32, &out);
    }

    /// RMSNorm an activation buffer and add the F32 result directly into an
    /// existing F32 residual shadow. `scratch_f32` is provided for backend
    /// fallbacks that need to materialize the normalized branch.
    fn rms_norm_activation_add_to_f32(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        eps: f32,
        residual_f32: &mut Self::Buffer,
        scratch_f32: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        Self::rms_norm_activation_to_f32(ctx, input, weight, eps, scratch_f32, tokens, dim);
        Self::add_inplace(ctx, residual_f32, scratch_f32, tokens * dim);
    }

    /// RMSNorm a typed F32 shadow buffer and write the normalized result back
    /// to the backend's regular activation dtype.
    fn rms_norm_f32_to_activation(
        ctx: &mut Self::Context,
        input_f32: &Self::Buffer,
        weight: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let input_h = Self::to_vec(input_f32, tokens * dim);
        let weight_h = Self::to_vec(weight, dim);
        let mut normed = vec![0.0f32; tokens * dim];
        for row in 0..tokens {
            let offset = row * dim;
            let mut variance = 0.0f32;
            for i in 0..dim {
                let x = input_h[offset + i];
                variance += x * x;
            }
            let inv_rms = (variance / dim as f32 + eps).sqrt().recip();
            for i in 0..dim {
                normed[offset + i] = input_h[offset + i] * inv_rms * weight_h[i];
            }
        }
        Self::write_f32_to_activation(ctx, out, &normed);
    }

    /// Greedy-decode fast path: GPU argmax over each row of a
    /// `[m, n]` FP16 logits buffer, returning the m token indices on the
    /// host. Saves `m × n × 2` bytes of D2H per call (e.g. 19.5 MB at
    /// c=32, vocab=152064) and the host-side argmax scan (~150 µs × m).
    ///
    /// Default impl falls back to the slow path: full `to_vec` + host
    /// argmax. CUDA overrides with a native kernel + tiny D2H (m × 4 B).
    /// Backends that don't override pay the same cost as
    /// `to_vec` + host argmax, so callers can call this unconditionally.
    fn argmax_rows_f16(
        _ctx: &mut Self::Context,
        logits: &Self::Buffer,
        m: usize,
        n: usize,
    ) -> Result<Vec<u32>> {
        let host = Self::to_vec(logits, m * n);
        let mut out = Vec::with_capacity(m);
        for row in 0..m {
            let slice = &host[row * n..(row + 1) * n];
            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in slice.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            out.push(max_idx as u32);
        }
        Ok(out)
    }

    fn argmax_rows_f16_masked(
        _ctx: &mut Self::Context,
        _logits: &Self::Buffer,
        _valid_token_mask: &Self::Buffer,
        _mask_len: usize,
        _m: usize,
        _n: usize,
    ) -> Result<Vec<u32>> {
        Err(FerrumError::unsupported(
            "masked GPU argmax is not implemented for this backend",
        ))
    }

    /// Load a weight tensor straight from its on-disk byte representation,
    /// letting the backend pick its preferred storage dtype.
    ///
    /// Default impl upcasts bf16/f16 to f32 via an intermediate Vec, matching
    /// pre-existing loader behaviour. Backends override this to go straight
    /// from raw bytes into a native half-precision buffer (e.g. Metal with
    /// `FERRUM_METAL_DTYPE=f16`), avoiding the transient 2× RAM spike.
    fn from_weight_bytes(raw: &[u8], src_dtype: SrcDtype) -> Self::Buffer {
        let data = src_dtype.to_f32_vec(raw);
        Self::from_slice(&data)
    }

    // (The Phase A3 unified `gemm_quant(QuantWeights, QuantKind)` stub
    // that used to live here is superseded by the `load_quant` /
    // `gemm_quant(QuantStore)` pair earlier in this trait — same idea,
    // but the store hides the per-kind buffer layout so callers don't
    // have to construct a per-kind `QuantWeights<'_, Self>` packet.)
}

// ════════════════════════════════════════════════════════════════════════
// BackendPagedKv capability (vLLM-style paged KV cache + paged attention)
// ════════════════════════════════════════════════════════════════════════
//
// Paged KV pool with block-table indirection, plus the paged attention
// kernel variants that read through that indirection. CUDA + Metal both
// implement the real kernels; CPU `impl BackendPagedKv for CpuBackend {}`
// inherits unsupported defaults.

/// Capability-trait for backends that support paged KV cache + paged attention.
pub trait BackendPagedKv: Backend {
    /// Whether this backend has a paged-KV decode path
    /// (`paged_decode_attention` etc.). Currently true for Metal, false
    /// for CPU. Used to decide the default of `FERRUM_METAL_PAGED_KV` —
    /// the `serve` path should opt in automatically when supported so
    /// users get the bench-quality concurrent-decode numbers without
    /// having to learn the flag.
    fn supports_paged_kv() -> bool {
        false
    }
    /// Pre-populate the per-slot device-pointer scratch arrays used by
    /// the batched kernels (`kv_cache_append_batched_per_cache` and
    /// `flash_attention_batched_per_cache`). Required by the CUDA-graph
    /// capture path: the captured graph contains only kernel launches
    /// (no captured `memcpy_htod`), so the device scratch must be fresh
    /// when the graph replays.
    ///
    /// Caller passes flat layer-major slices: `k_caches[li * m + i]` and
    /// `v_caches[li * m + i]`. Backend extracts each cache's device
    /// pointer and writes into its corresponding slot in the device
    /// scratch via SYNCHRONOUS memcpy (not captured by stream capture).
    ///
    /// CUDA-only; other backends fall through to the default
    /// `unsupported` and the caller skips the population call.
    fn populate_batched_pointers(
        _ctx: &mut Self::Context,
        _k_caches: &[&Self::Buffer],
        _v_caches: &[&Self::Buffer],
        _num_layers: usize,
        _m: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "populate_batched_pointers not implemented for this backend",
        ))
    }
    /// Paged-KV variant of [`Self::split_qkv_norm_rope_into_cache`].
    ///
    /// Same fused split + qk-norm + RoPE, but K/V are written into a
    /// paged pool `[num_blocks, kv_heads, block_size, head_dim]`
    /// indexed via `block_table[logical_block]` → physical_block.
    /// Q still goes to head-major scratch.
    ///
    /// Default returns Unsupported. Backends that lack a paged kernel
    /// keep using the contiguous variant.
    /// `qkv_byte_offset` / `q_out_byte_offset` let the caller pass a
    /// slice of a larger batched buffer (used by the multi-seq paged
    /// path in `decode_batch_internal`). For single-seq dispatch they
    /// should be 0.
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache(
        _ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _qkv_byte_offset: u64,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _q_out_byte_offset: u64,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _block_table: &Self::Buffer,
        _tokens: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _qk_mode: i32,
        _cache_len: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "split_qkv_norm_rope_into_paged_cache not implemented for this backend",
        ))
    }
    /// Paged-KV variant of [`Self::flash_attention`].
    ///
    /// Decode (`q_len == 1`):
    ///   `q`/`out`: `[num_seqs, num_heads, head_dim]` (token-major)
    ///
    /// Causal prefill (`q_len > 1`, single seq):
    ///   `q`/`out`: `[num_heads, q_len, head_dim]` (head-major — the
    ///              layout produced by `split_qkv_norm_rope_into_paged_cache`)
    ///   The kernel applies a per-q-token causal mask using
    ///   `context_lens[seq]` as the FINAL kv_len (= `pos_offset + q_len`):
    ///   token i sees positions `[0, context_lens - q_len + 1 + i)`.
    ///
    /// Common to both:
    ///   `k_pool`/`v_pool`: `[num_blocks, num_kv_heads, block_size, head_dim]`
    ///   `block_tables`: `[num_seqs, max_num_blocks_per_seq]` u32
    ///   `context_lens`: `[num_seqs]` u32
    ///
    /// Backends without a paged kernel return Unsupported; callers are
    /// expected to fall back to contiguous KV.
    #[allow(clippy::too_many_arguments)]
    fn paged_decode_attention(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_pool: &Self::Buffer,
        _v_pool: &Self::Buffer,
        _out: &mut Self::Buffer,
        _block_tables: &Self::Buffer,
        _context_lens: &Self::Buffer,
        _num_seqs: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
        _q_len: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "paged_decode_attention not implemented for this backend",
        ))
    }
    /// Capability: does this backend implement
    /// `split_qkv_norm_rope_into_paged_cache_varlen` and
    /// `paged_varlen_attention`? Required by the unified mixed-batch
    /// forward path used by `LlamaFamilyModel::unified_forward`. Default
    /// false; backends that ship the varlen kernels override.
    fn supports_varlen_qkv() -> bool {
        false
    }
    /// Varlen variant of [`Self::split_qkv_norm_rope_into_paged_cache`].
    ///
    /// Single launch covering ALL sequences in the batch. Reads
    /// `pos_offsets[seq]`, `cu_seqlens_q[seq]`, and the per-seq
    /// block_table from device buffers — graph-capturable (the per-iter
    /// state is in buffers, not kernel scalars). Replaces the per-item
    /// dispatch loop in `unified_forward_layer` with one call.
    ///
    /// Layouts:
    /// - `qkv`: `[m_total, q_dim + 2 * kv_dim]` token-major
    /// - `q_out`: `[m_total, q_heads, head_dim]` token-major (matches
    ///   what `paged_varlen_attention` reads)
    /// - `cache_k` / `cache_v`: paged pool same as `paged_varlen_attention`
    /// - `cu_seqlens_q`: `[num_seqs + 1]` u32 prefix sum
    /// - `pos_offsets`: `[num_seqs]` u32, starting kv_pos per seq
    /// - `block_tables`: `[num_seqs, max_blocks_per_seq]` i32 stacked
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache_varlen(
        _ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _cu_seqlens_q: &Self::Buffer,
        _pos_offsets: &Self::Buffer,
        _block_tables: &Self::Buffer,
        _num_seqs: usize,
        _m_total: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _eps: f32,
        _qk_mode: i32,
        _block_size: usize,
        _max_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "split_qkv_norm_rope_into_paged_cache_varlen not implemented for this backend",
        ))
    }
    /// Variable-length paged attention with GQA + causal mask.
    ///
    /// Supports a unified mixed batch where each sequence contributes
    /// 1 (decode) or N (prefill chunk) query tokens — the workhorse for
    /// chunked-prefill. See `kernels/paged_varlen_attention.cu` for the
    /// kernel itself.
    ///
    /// Layouts:
    /// - `q` / `out`: `[total_q_tokens, num_heads, head_dim]` (token-
    ///   major, FP16). `total_q_tokens` = `cu_seqlens_q[num_seqs]`.
    /// - `k_pool` / `v_pool`: paged block pool, layout matches
    ///   `paged_decode_attention`.
    /// - `cu_seqlens_q`: `[num_seqs + 1]` u32 prefix sum, with
    ///   `cu_seqlens_q[0] = 0` and `cu_seqlens_q[num_seqs] = total_q_tokens`.
    /// - `pos_offsets`: `[num_seqs]` u32, the starting absolute KV
    ///   position of each seq's first q token (= prior `kv_len`).
    /// - `block_tables`: `[num_seqs, max_num_blocks_per_seq]` i32 grid.
    ///
    /// Each query token attends causally to KV positions
    /// `[0, pos_offsets[s] + local_idx]` when `sliding_window == 0`, or
    /// only the most recent `sliding_window` positions when non-zero.
    #[allow(clippy::too_many_arguments)]
    fn paged_varlen_attention(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_pool: &Self::Buffer,
        _v_pool: &Self::Buffer,
        _out: &mut Self::Buffer,
        _cu_seqlens_q: &Self::Buffer,
        _pos_offsets: &Self::Buffer,
        _block_tables: &Self::Buffer,
        _num_seqs: usize,
        _total_q_tokens: usize,
        _max_kv_len: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _sliding_window: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "paged_varlen_attention not implemented for this backend",
        ))
    }

    /// Opt-in vLLM FlashAttention-2 FFI path for FA-layout paged KV.
    ///
    /// This is intentionally separate from [`Self::paged_varlen_attention`]:
    /// it needs the final per-sequence KV lengths (`seq_lens`) and an explicit
    /// LSE scratch buffer because the external FA2 runner writes softmax LSE.
    /// Default returns Err(unsupported); CUDA overrides when a runtime shim is
    /// provided via `FERRUM_FA2_DIRECT_FFI_SHIM`.
    #[allow(clippy::too_many_arguments)]
    fn paged_varlen_attention_fa2_ffi(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_pool: &Self::Buffer,
        _v_pool: &Self::Buffer,
        _out: &mut Self::Buffer,
        _lse: &mut Self::Buffer,
        _cu_seqlens_q: &Self::Buffer,
        _seq_lens: &Self::Buffer,
        _block_tables: &Self::Buffer,
        _num_seqs: usize,
        _total_q_tokens: usize,
        _max_q_len: usize,
        _max_kv_len: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "paged_varlen_attention_fa2_ffi not implemented for this backend",
        ))
    }

    /// Batched paged decode attention — multi-seq, single token per seq.
    /// Faster path for the unified_forward layer when m_total == num_seqs
    /// (every item is a single-token decode). Skips the cu_seqlens_q
    /// linear scan that `paged_varlen_attention` does in the fully-mixed
    /// case.
    ///
    /// Layouts:
    ///   q              : [num_seqs, num_q_heads, head_dim]
    ///   k_pool/v_pool  : paged pool (same as paged_varlen)
    ///   block_tables   : [num_seqs, max_num_blocks_per_seq]
    ///   valid_kv_lens  : [num_seqs] — current kv_len per seq
    ///   out            : [num_seqs, num_q_heads, head_dim]
    ///
    /// Default returns Err(unsupported); CUDA backend overrides.
    #[allow(clippy::too_many_arguments)]
    fn paged_batched_decode_attention(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_pool: &Self::Buffer,
        _v_pool: &Self::Buffer,
        _out: &mut Self::Buffer,
        _block_tables: &Self::Buffer,
        _valid_kv_lens: &Self::Buffer,
        _num_seqs: usize,
        _max_kv_len: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "paged_batched_decode_attention not implemented for this backend",
        ))
    }

    /// Capability: backend has vLLM-layout paged KV write kernels and the
    /// `paged_attention_v2` decode kernel. Models that opt into this layout
    /// at construction time (via `FERRUM_USE_VLLM_PAGED_ATTN=1`) must
    /// dispatch ALL paged writes and reads through the `_vllm` variants —
    /// the layouts are not compatible. Default `false`.
    fn supports_vllm_paged_attn() -> bool {
        false
    }

    /// Qwen3.5 full-attention uses separate q/k/v projections and partial
    /// RoPE/gated-Q layout, so it cannot use the fused-QKV paged writer
    /// directly. Backends that implement this method can write those
    /// separate projections into the same vLLM-style paged pool consumed by
    /// [`Self::paged_batched_decode_attention`].
    fn supports_qwen35_paged_qkv() -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_split_qkv_norm_rope_into_paged_cache_varlen(
        _ctx: &mut Self::Context,
        _query_raw: &Self::Buffer,
        _key_raw: &Self::Buffer,
        _value_raw: &Self::Buffer,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _cu_seqlens_q: &Self::Buffer,
        _pos_offsets: &Self::Buffer,
        _block_tables: &Self::Buffer,
        _num_seqs: usize,
        _total_q_tokens: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _rope_dim: usize,
        _q_proj_stride: usize,
        _q_head_stride: usize,
        _kv_proj_stride: usize,
        _eps: f32,
        _qk_mode: i32,
        _block_size: usize,
        _max_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "qwen35_split_qkv_norm_rope_into_paged_cache_varlen not implemented for this backend",
        ))
    }

    /// vLLM-layout variant of
    /// [`Self::split_qkv_norm_rope_into_paged_cache`]. K/V are written in
    /// vLLM's `paged_attention_v2` layout: K is
    /// `[num_blocks, kv_heads, head_dim/x, block_size, x]` (x = 16/sizeof(elem)),
    /// V is `[num_blocks, kv_heads, head_dim, block_size]`. Q output and
    /// every other argument matches the non-vllm variant exactly so the
    /// model layer can swap dispatchers based on a single flag.
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache_vllm(
        _ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _qkv_byte_offset: u64,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _q_out_byte_offset: u64,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _block_table: &Self::Buffer,
        _tokens: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _qk_mode: i32,
        _cache_len: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "split_qkv_norm_rope_into_paged_cache_vllm not implemented for this backend",
        ))
    }

    /// vLLM-layout variant of
    /// [`Self::split_qkv_norm_rope_into_paged_cache_varlen`]. Same signature
    /// — only the K/V cache layout changes.
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache_varlen_vllm(
        _ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _q_norm_w: &Self::Buffer,
        _k_norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _q_out: &mut Self::Buffer,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _cu_seqlens_q: &Self::Buffer,
        _pos_offsets: &Self::Buffer,
        _block_tables: &Self::Buffer,
        _num_seqs: usize,
        _m_total: usize,
        _q_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _eps: f32,
        _qk_mode: i32,
        _block_size: usize,
        _max_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "split_qkv_norm_rope_into_paged_cache_varlen_vllm not implemented for this backend",
        ))
    }

    /// vLLM `paged_attention_v2` — multi-partition split-K decode attention
    /// reading the vLLM K/V layout. `q_len` is implicitly 1 (decode only;
    /// vLLM's v2 kernel does not support q_len > 1). `max_seq_len` is the
    /// max kv_len across the batch — used to size the partition reduction.
    #[allow(clippy::too_many_arguments)]
    fn paged_decode_attention_v2(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_pool: &Self::Buffer,
        _v_pool: &Self::Buffer,
        _out: &mut Self::Buffer,
        _block_tables: &Self::Buffer,
        _context_lens: &Self::Buffer,
        _num_seqs: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
        _max_seq_len: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "paged_decode_attention_v2 not implemented for this backend",
        ))
    }

    /// q_len>1 prefill/chunk-prefill attention over vLLM-layout paged KV.
    /// This keeps cache layout consistent when `FERRUM_USE_VLLM_PAGED_ATTN=1`
    /// and the prompt path writes K/V in the layout consumed later by
    /// `paged_decode_attention_v2`.
    #[allow(clippy::too_many_arguments)]
    fn paged_varlen_attention_vllm_layout(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_pool: &Self::Buffer,
        _v_pool: &Self::Buffer,
        _out: &mut Self::Buffer,
        _block_tables: &Self::Buffer,
        _context_lens: &Self::Buffer,
        _num_seqs: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
        _q_len: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "paged_varlen_attention_vllm_layout not implemented for this backend",
        ))
    }

    /// Variable-length paged attention over vLLM-layout paged KV.
    ///
    /// Unlike [`Self::paged_varlen_attention_vllm_layout`], this accepts the
    /// same varlen index tensors as [`Self::paged_varlen_attention`] and writes
    /// token-major output directly. It is the unified mixed-batch companion for
    /// `split_qkv_norm_rope_into_paged_cache_varlen_vllm`.
    #[allow(clippy::too_many_arguments)]
    fn paged_varlen_attention_vllm(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_pool: &Self::Buffer,
        _v_pool: &Self::Buffer,
        _out: &mut Self::Buffer,
        _cu_seqlens_q: &Self::Buffer,
        _pos_offsets: &Self::Buffer,
        _block_tables: &Self::Buffer,
        _num_seqs: usize,
        _total_q_tokens: usize,
        _max_kv_len: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "paged_varlen_attention_vllm not implemented for this backend",
        ))
    }

    /// Q-tiled vLLM-layout varlen attention. `tile_seqs` and `tile_starts`
    /// describe a compact list of q-token tiles, avoiding empty grid blocks
    /// for mixed batches that contain both long prefill items and q_len=1
    /// decode items. Semantics match [`Self::paged_varlen_attention_vllm`].
    #[allow(clippy::too_many_arguments)]
    fn paged_varlen_attention_vllm_tiled_q4(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k_pool: &Self::Buffer,
        _v_pool: &Self::Buffer,
        _out: &mut Self::Buffer,
        _cu_seqlens_q: &Self::Buffer,
        _pos_offsets: &Self::Buffer,
        _block_tables: &Self::Buffer,
        _tile_seqs: &Self::Buffer,
        _tile_starts: &Self::Buffer,
        _num_tiles: usize,
        _max_kv_len: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _block_size: usize,
        _max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "paged_varlen_attention_vllm_tiled_q4 not implemented for this backend",
        ))
    }
}

// ════════════════════════════════════════════════════════════════════════
// Capability bundles — readable type aliases over the supertrait set
// ════════════════════════════════════════════════════════════════════════
//
// Models declare what they need via these bundles instead of spelling out
// every supertrait. Rust auto-derives the impl via blanket impls below,
// so any backend that satisfies the underlying supertraits automatically
// becomes a `LlmBackend` / `QuantLlmBackend` / `MoeLlmBackend`.

/// Minimum capability set for a decoder-only LLM: the core compute trait
/// plus paged-KV cache + graph-capture support. Every concrete backend
/// (CUDA / Metal / CPU) satisfies this.
pub trait LlmBackend: Backend + BackendGraph + BackendPagedKv {}
impl<T> LlmBackend for T where T: Backend + BackendGraph + BackendPagedKv {}

/// LLM backend that also supports quantized weight loading (GPTQ Marlin
/// for CUDA; GGUF k-quant for Metal). Required by models that hold
/// `Box<dyn Linear<B>>` where the Linear impl might be a quant variant.
pub trait QuantLlmBackend: LlmBackend + BackendQuantMarlin + BackendQuantGguf {}
impl<T> QuantLlmBackend for T where T: LlmBackend + BackendQuantMarlin + BackendQuantGguf {}

/// MoE-capable LLM backend: adds the fused MoE routing + post-op kernels
/// to the quant LLM bundle. Required by Qwen3-MoE / future MoE models.
pub trait MoeLlmBackend: QuantLlmBackend + BackendMoeFused {}
impl<T> MoeLlmBackend for T where T: QuantLlmBackend + BackendMoeFused {}

// ════════════════════════════════════════════════════════════════════════
// KV cache dtype axis (dim 5 of the 5-dimension architecture)
// ════════════════════════════════════════════════════════════════════════
//
// Each model's KV cache has its own precision independent of the model's
// compute precision. vLLM 0.6+ ships INT8 / FP8 KV caches that halve KV
// memory at small (<1%) accuracy hit. Today ferrum's KV is hardcoded
// FP16 on CUDA / Metal — to support INT8/FP8 KV in a future PR, the
// type system needs an explicit axis.
//
// Phase 4 scope: scaffolding only. All concrete backends impl
// `BackendKvDtype<KvFp16>` so existing models keep working unchanged.
// Future PR: implement BackendKvDtype<KvInt8> on CUDA + a new model
// type-parameter `K: KvDtypeKind` to wire it through.

// `KvDtypeKind` + `KvFp16` / `KvBf16` / `KvInt8` / `KvFp8` markers moved
// to `ferrum_interfaces::kv_dtype` (no GPU deps, so the right place is
// the contract crate). Re-exported here so existing callers keep
// compiling against `crate::backend::KvFp16` etc.
pub use ferrum_interfaces::kv_dtype::{KvBf16, KvDtypeKind, KvFp16, KvFp8, KvInt8};

/// Capability-trait for backends that can store + read a KV cache of
/// type `K`.
///
/// The two associated types carry the K-specific storage shape:
///   - `KvBuffer`: per-layer K/V element storage. For `K = KvFp16` it
///     is the backend's normal `Self::Buffer` (FP16). For `K = KvInt8`
///     it is the backend's INT8 buffer (e.g. `CudaSlice<i8>` on CUDA).
///   - `KvScales`: per-token-per-kv-head scales. For `K = KvFp16` this
///     is the unit type `()` (no scales). For `K = KvInt8` / `KvFp8`
///     it is a backend-specific FP16 buffer.
///
/// Models that want INT8 KV use:
///   `where B: BackendKvDtype<KvInt8>`
/// — the buffers in `KvCache<B, KvInt8>` are then `CudaSlice<i8>` and
/// `CudaSlice<f16>`, distinct from the FP16 path's `Self::Buffer`.
pub trait BackendKvDtype<K: KvDtypeKind>: BackendPagedKv {
    /// Per-layer K/V element storage.
    type KvBuffer: Send + Sync;
    /// Per-token per-kv-head scale storage. `()` for FP16 (no scales).
    type KvScales: Send + Sync + Default;
}

/// INT8 KV cache operations (Dim 5).
///
/// `BackendKvDtype<KvInt8>` only declares the storage types; it does not
/// know how to write INT8 K/V into a paged pool or run paged decode
/// attention against an INT8 cache. Those launchers live here so the
/// model layer can call them through a single `B: BackendInt8KvOps` bound
/// without dropping into backend-specific code.
///
/// Today only `CudaBackend` provides a real implementation (delegating to
/// [`crate::int8_kv::launch_int8_kv_cache_append`] and
/// [`crate::int8_kv::launch_int8_paged_decode_attention`]). Other backends
/// inherit the default `unimplemented!()` body — the registry factory
/// rejects `(Device::CPU/Metal, KvCacheDtype::Int8)` before the model
/// gets a chance to call into these.
#[allow(clippy::too_many_arguments)]
pub trait BackendInt8KvOps: Backend + BackendKvDtype<KvInt8> {
    /// Allocate the per-layer INT8 paged cache for one sequence.
    /// Default panics — backends without INT8 support never reach this
    /// path (factory rejects (Cpu/Metal, Int8) before ensure_kv runs).
    fn alloc_paged_int8_layer(
        _max_blocks_per_seq: usize,
        _block_size: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) -> KvCacheQuant<Self, KvInt8> {
        unimplemented!("alloc_paged_int8_layer not supported on this backend")
    }

    /// Append `tokens` FP16 K/V values into the paged INT8 pool.
    /// `paged_block_indices` is the host-side mirror of the per-seq
    /// logical→physical block table (already populated at `ensure_kv` time
    /// — see `KvCacheQuant::paged_block_indices`). Passing the host slice
    /// avoids a per-token D2H + sync barrier; backend computes the slot
    /// mapping host-side, async-H2D's it, and chains the append kernel
    /// on the same stream — fully overlapping with prior work.
    /// `cache_len_before` is the current number of valid tokens; the
    /// backend quantizes FP16 → INT8 with per-(token, kv-head) FP16 scale
    /// and writes both into the layer's INT8 / scale buffers.
    fn int8_kv_append_paged(
        _ctx: &mut Self::Context,
        _k_in: &Self::Buffer,
        _v_in: &Self::Buffer,
        _layer_k: &mut <Self as BackendKvDtype<KvInt8>>::KvBuffer,
        _layer_v: &mut <Self as BackendKvDtype<KvInt8>>::KvBuffer,
        _layer_k_scales: &mut <Self as BackendKvDtype<KvInt8>>::KvScales,
        _layer_v_scales: &mut <Self as BackendKvDtype<KvInt8>>::KvScales,
        _paged_block_indices: &[u32],
        _cache_len_before: usize,
        _tokens: usize,
        _block_size: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "int8_kv_append_paged not implemented for this backend",
        ))
    }

    /// Run paged decode attention reading from an INT8 cache. Q is FP16,
    /// output is FP16; the kernel dequantizes K/V on the fly using the
    /// per-token scales. `valid_kv_len` is the post-append cache length
    /// (i.e. the kernel attends over `[0, valid_kv_len)` tokens).
    fn int8_paged_decode_attention(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _layer_k: &<Self as BackendKvDtype<KvInt8>>::KvBuffer,
        _layer_v: &<Self as BackendKvDtype<KvInt8>>::KvBuffer,
        _layer_k_scales: &<Self as BackendKvDtype<KvInt8>>::KvScales,
        _layer_v_scales: &<Self as BackendKvDtype<KvInt8>>::KvScales,
        _block_table: &Self::Buffer,
        _output: &mut Self::Buffer,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _valid_kv_len: usize,
        _block_size: usize,
        _scale: f32,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "int8_paged_decode_attention not implemented for this backend",
        ))
    }
}

// Cpu/Metal NOT impl `BackendInt8KvOps` — the trait pivot to
// `KvLayer<B>` means `KvInt8: KvLayer<B>` only holds where
// `B: BackendInt8KvOps`, so `LlamaFamilyModel<CpuBackend, KvInt8>` is a
// compile error (no INT8 KvLayer impl satisfies it). Type system
// enforces the constraint without runtime stubs.
