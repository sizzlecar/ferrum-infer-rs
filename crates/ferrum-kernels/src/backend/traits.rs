//! Core Backend trait — the single abstraction over CUDA / Metal / CPU.

use ferrum_types::{FerrumError, Result};
use half::{bf16, f16};

/// Source dtype for a weight tensor read straight from safetensors mmap.
///
/// Passed to `Backend::from_weight_bytes` so each backend can choose whether
/// to upcast to its compute dtype or store as-is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrcDtype {
    F32,
    F16,
    BF16,
}

impl SrcDtype {
    /// Number of bytes per element in the raw on-disk representation.
    pub const fn bytes_per_elem(self) -> usize {
        match self {
            SrcDtype::F32 => 4,
            SrcDtype::F16 | SrcDtype::BF16 => 2,
        }
    }

    /// Materialise the raw byte slice into a `Vec<f32>`. Used by the default
    /// `Backend::from_weight_bytes` impl; fp16-preferring backends bypass it.
    pub fn to_f32_vec(self, raw: &[u8]) -> Vec<f32> {
        match self {
            SrcDtype::F32 => {
                debug_assert_eq!(raw.len() % 4, 0);
                let n = raw.len() / 4;
                let mut out = vec![0f32; n];
                for i in 0..n {
                    let b = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
                    out[i] = f32::from_le_bytes(b);
                }
                out
            }
            SrcDtype::F16 => {
                debug_assert_eq!(raw.len() % 2, 0);
                let n = raw.len() / 2;
                let mut out = vec![0f32; n];
                for i in 0..n {
                    out[i] = f16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]).to_f32();
                }
                out
            }
            SrcDtype::BF16 => {
                debug_assert_eq!(raw.len() % 2, 0);
                let n = raw.len() / 2;
                let mut out = vec![0f32; n];
                for i in 0..n {
                    out[i] = bf16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]).to_f32();
                }
                out
            }
        }
    }
}

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

    /// Single backend-specific store for **all GGUF k-quant flavours**
    /// (Q4_K_M today; Q5_K_M / Q6_K / Q8_0 etc. become enum variants
    /// without changing the trait shape).
    ///
    /// Each backend's `QuantStore` is typically an enum dispatching on
    /// the on-disk quant type — the public API (`load_quant`,
    /// `gemm_quant`) takes a [`QuantKind`] discriminator so callers
    /// don't see the variant boilerplate.
    ///
    /// **GPTQ stays on the older [`Self::GptqStore`] path** because its
    /// load inputs are split arrays (qweight / scales / qzeros), not
    /// the contiguous byte payload GGUF quants ship as. A future PR can
    /// fold GPTQ into `QuantStore` once an input-shape unification is
    /// agreed.
    type QuantStore: Send + Sync;

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

    /// Drop the cached decode graph — required when the KV cache it
    /// was captured against is about to be freed (e.g. request release),
    /// since the graph holds raw device pointers into that cache.
    fn reset_graph(_ctx: &mut Self::Context) {}

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

    /// Load GGUF k-quant weights into the backend's preferred format.
    ///
    /// `kind` discriminates Q4_K / Q5_K / Q6_K / Q8_0 etc. The CPU path
    /// typically eager-dequants to fp32; the Metal path keeps raw block
    /// bytes in MTLBuffer and dequants per matmul into a transient fp16
    /// buffer. Adding a new k-quant flavour is a matched pair of
    /// `QuantStore` variant + `match` arm, not a new trait method.
    ///
    /// `bytes`: contiguous on-disk payload — `n_blocks × block_size`.
    /// `n_rows`: out_features. `n_cols`: in_features. The block count
    /// is derived per-kind from these dims.
    fn load_quant(
        _kind: GgufQuantType,
        _bytes: &[u8],
        _n_rows: usize,
        _n_cols: usize,
    ) -> Result<Self::QuantStore> {
        Err(FerrumError::unsupported(
            "load_quant not implemented for this backend",
        ))
    }

    /// Build a fused `QuantStore` from multiple `(kind, bytes, n_rows)`
    /// parts that share `n_cols`. Used by `GgufLoader::load_fused` when
    /// parts have heterogeneous quant kinds (e.g. Qwen3 qkv_proj where
    /// q+k are Q4_K but v is Q6_K) — byte-concatenation isn't possible,
    /// so each part stays as its own QuantStore and the gemm dispatches
    /// one matvec per part with output offsets.
    ///
    /// Default: not supported. Backends that have a `Fused`-like variant
    /// override.
    fn load_quant_fused(
        _parts: &[(GgufQuantType, &[u8], usize)],
        _n_cols: usize,
    ) -> Result<Self::QuantStore> {
        Err(FerrumError::unsupported(
            "load_quant_fused not implemented for this backend",
        ))
    }

    /// GEMM with k-quant weights. Mirrors `gemm` / `gemm_gptq` shape:
    /// `out[m, n] = a[m, k] @ dequant(weight)^T`. The dispatch on the
    /// quant flavour happens inside the backend's `QuantStore` enum.
    fn gemm_quant(
        _ctx: &mut Self::Context,
        _a: &Self::Buffer,
        _weight: &Self::QuantStore,
        _out: &mut Self::Buffer,
        _m: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "gemm_quant not implemented for this backend",
        ))
    }

    /// Build a stacked-experts `QuantStore` from a contiguous 3-D weight
    /// payload `[num_experts, n_rows, n_cols/256]` super-blocks.
    /// Used for the MoE indirect-dispatch fast path; backends without
    /// such a kernel return `Err(unsupported)` and the model code falls
    /// back to the per-expert loop.
    ///
    /// Default: not supported. Override on backends with batched MoE
    /// kernels (e.g. Metal `gemv_q*kw_moe_id_f32`).
    fn load_quant_experts(
        _kind: GgufQuantType,
        _bytes: &[u8],
        _num_experts: usize,
        _n_rows: usize,
        _n_cols: usize,
    ) -> Result<Self::QuantStore> {
        Err(FerrumError::unsupported(
            "load_quant_experts not implemented for this backend",
        ))
    }

    /// MoE 2-D indirect-dispatch GEMM (prefill m > 1).
    ///
    /// Computes per (token, expert_slot) pair, batched across all
    /// experts in one launch:
    ///
    ///   `out[token, slot, :] = a[token, slot_or_0, :] @ dequant(weight[expert(token, slot), :])^T`
    ///
    /// `ids[expert][slot] = pair_id` encodes `(token_idx, slot_within_token)`
    /// so the kernel reads activations indirectly (src1 row for the
    /// pair) and writes outputs directly to the natural
    /// `[batch, top_k, M]` layout. `tpe[expert]` gives the count of
    /// pairs assigned to each expert — threadgroups past `tpe[e]`
    /// early-exit.
    ///
    /// `ne11` selects the src1 inner-batch shape:
    /// - `1` for `gate` / `up` (broadcast — all slots read the same
    ///   activation row per token).
    /// - `top_k` for `down` (per-slot — each pair reads its own row in
    ///   the upstream silu·gate output).
    ///
    /// Closes the prefill MoE gap: the per-token gemv loop becomes one
    /// batched gemm where each expert's slab handles m ≈ batch·top_k /
    /// num_experts pairs in parallel via simdgroup_half8x8 matmul.
    #[allow(clippy::too_many_arguments)]
    fn gemm_quant_moe_id(
        _ctx: &mut Self::Context,
        _a: &Self::Buffer,
        _weight: &Self::QuantStore,
        _ids: &Self::Buffer,
        _tpe: &Self::Buffer,
        _out: &mut Self::Buffer,
        _ne11: usize,
        _top_k: usize,
        _max_per_expert: usize,
        _batch: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "gemm_quant_moe_id not implemented for this backend",
        ))
    }

    /// GPU-side MoE router: `[batch, num_experts]` logits → `[batch, top_k]`
    /// expert IDs (i32) + `[batch, top_k]` combine weights (f32).
    ///
    /// Replaces the per-layer `B::sync + B::to_vec(router_logits) + host route()`
    /// round trip. The output buffers stay device-side for downstream
    /// `gemv_quant_moe_id` / `gemm_quant_moe_id` consumption — no host
    /// pipeline drain in the inner loop.
    ///
    /// `norm_topk_prob`: if true, divide each row's K weights by their
    /// sum so they total 1.0 (Qwen3-MoE / Mixtral default).
    #[allow(clippy::too_many_arguments)]
    fn route_topk_softmax(
        _ctx: &mut Self::Context,
        _logits: &Self::Buffer,
        _out_ids: &mut Self::Buffer,
        _out_weights: &mut Self::Buffer,
        _batch: usize,
        _num_experts: usize,
        _top_k: usize,
        _norm_topk_prob: bool,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "route_topk_softmax not implemented for this backend",
        ))
    }

    /// Stacked SiLU·gate over `[batch * top_k, ffn]` rows (prefill version
    /// of `silu_mul_stacked`).
    fn silu_mul_batched(
        _ctx: &mut Self::Context,
        _gate: &Self::Buffer,
        _up: &Self::Buffer,
        _out: &mut Self::Buffer,
        _total_pairs: usize,
        _ffn: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "silu_mul_batched not implemented for this backend",
        ))
    }

    /// Fused weighted-sum + residual-add: `residual[i] += Σ_k weights[k] · slots[k, i]`.
    /// Single dispatch replaces the (weighted_sum → moe_out) +
    /// (add_inplace residual += moe_out) pair on the decode hot path.
    fn weighted_sum_residual_stacked(
        _ctx: &mut Self::Context,
        _slots: &Self::Buffer,
        _weights: &Self::Buffer,
        _residual: &mut Self::Buffer,
        _n_slots: usize,
        _hidden: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "weighted_sum_residual_stacked not implemented for this backend",
        ))
    }

    /// Per-batch weighted sum: `out[b, h] = Σ_k weights[b, k] · slots[b, k, h]`.
    /// Single dispatch covers the whole batch (prefill version of
    /// `weighted_sum_stacked` which only handled one token).
    fn weighted_sum_batched(
        _ctx: &mut Self::Context,
        _slots: &Self::Buffer,
        _weights: &Self::Buffer,
        _out: &mut Self::Buffer,
        _batch: usize,
        _top_k: usize,
        _hidden: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "weighted_sum_batched not implemented for this backend",
        ))
    }

    /// MoE indirect-dispatch GEMV: `out[i, :] = a[i, :] @ dequant(weight[ids[i], :])^T`
    /// for each `i ∈ [0, n_selected)`. Single backend dispatch covers
    /// all selected (token, expert) pairs.
    ///
    /// `weight` must be a stacked-experts variant produced by
    /// [`Self::load_quant_experts`]. `ids` is a backend-side buffer of
    /// `n_selected` i32 expert IDs. `out` is sized `[n_selected, n_rows]`.
    /// `src1_stride` is the per-slot activation stride in **elements**:
    /// `0` ⇒ every slot reads the same activation row (broadcast — for
    /// `gate` / `up` projections); `n_cols` ⇒ each slot reads its own
    /// activation row (for `down` projections, where each expert
    /// consumes its own silu(gate)·up output).
    fn gemv_quant_moe_id(
        _ctx: &mut Self::Context,
        _a: &Self::Buffer,
        _weight: &Self::QuantStore,
        _ids: &Self::Buffer,
        _out: &mut Self::Buffer,
        _n_selected: usize,
        _src1_stride: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "gemv_quant_moe_id not implemented for this backend",
        ))
    }

    /// Allocate a backend buffer of i32-typed values for kernels that
    /// need integer indices (MoE expert IDs, scatter indices, etc.).
    ///
    /// Default impl bit-casts the i32s to f32s and uploads via
    /// `from_slice` — useful on backends where the buffer type is type-
    /// erased (CPU's `Vec<f32>`, Metal's untyped MTLBuffer). Backends
    /// that use a strongly-typed buffer override.
    fn from_slice_i32(data: &[i32]) -> Self::Buffer {
        let f: Vec<f32> = data.iter().map(|&i| f32::from_bits(i as u32)).collect();
        Self::from_slice(&f)
    }

    /// Overwrite an existing i32 buffer's contents in place. Used on
    /// the MoE decode hot path: per-layer expert-id updates do an
    /// in-place memcpy instead of allocating a fresh device buffer
    /// (48 layers × 128 tokens = 6144 fresh allocations per decode
    /// run otherwise — allocator pressure dominates the secondary cost).
    ///
    /// Default impl falls back to `from_slice_i32` + drop. Backends
    /// with shared CPU↔GPU memory (Metal `StorageModeShared`, CPU's
    /// `Vec<f32>`) override with a direct write.
    fn write_i32_into(buf: &mut Self::Buffer, data: &[i32]) {
        *buf = Self::from_slice_i32(data);
    }

    /// Overwrite an existing f32 buffer's contents in place. Counterpart
    /// to `write_i32_into` for f32 data — used to update the per-token
    /// MoE combine weights into a pre-allocated scratch buffer instead
    /// of allocating a fresh `from_slice` buffer 6144 times per decode
    /// run.
    fn write_f32_into(buf: &mut Self::Buffer, data: &[f32]) {
        *buf = Self::from_slice(data);
    }

    /// Stacked SiLU·gate over `[n_slots, ffn]` rows.
    ///
    /// Computes `out[s, i] = silu(gate[s, i]) * up[s, i]` for each slot
    /// `s`, element `i`. Single dispatch covers all slots — cuts the
    /// MoE decode silu staging from `top_k * (3 copy_slice + 1 silu)`
    /// = 32 dispatches per layer to 1.
    fn silu_mul_stacked(
        _ctx: &mut Self::Context,
        _gate: &Self::Buffer,
        _up: &Self::Buffer,
        _out: &mut Self::Buffer,
        _n_slots: usize,
        _ffn: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "silu_mul_stacked not implemented for this backend",
        ))
    }

    /// Weighted sum across `n_slots` rows of `[hidden]`.
    ///
    /// Computes `out[i] = Σ_s weights[s] * slots[s, i]`. Single
    /// dispatch replaces the per-slot `(copy_slice + scaled_add)`
    /// loop in the MoE decode path (16 dispatches per layer → 1).
    fn weighted_sum_stacked(
        _ctx: &mut Self::Context,
        _slots: &Self::Buffer,
        _weights: &Self::Buffer,
        _out: &mut Self::Buffer,
        _n_slots: usize,
        _hidden: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "weighted_sum_stacked not implemented for this backend",
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

    /// Fused split-QKV + QK-norm + RoPE + head-major transpose.
    ///
    /// Single-dispatch replacement for the (`split_qkv` → 3× `qk_norm_rope`)
    /// chain on the decode-attention prelude. Reads the linear-layer
    /// fused-QKV output once and writes head-major Q/K/V directly into
    /// attention scratch.
    ///
    /// `qkv` layout: `[tokens, q_heads*hd + 2*kv_heads*hd]`.
    /// `q_out`: `[q_heads, tokens, hd]`. `k_out`/`v_out`: `[kv_heads, tokens, hd]`.
    /// `qk_mode`: 1 = norm + RoPE for Q/K (Qwen3 with QK-norm),
    ///            2 = RoPE only for Q/K (no QK-norm; Llama-style).
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
