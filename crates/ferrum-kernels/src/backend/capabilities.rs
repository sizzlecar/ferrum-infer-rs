//! Optional backend capability traits layered on top of [`Backend`].

use ferrum_types::{FerrumError, Result};

use super::traits::Backend;
use super::types::{GgufQuantType, MoeRouting, ReduceOp};

// ════════════════════════════════════════════════════════════════════════
// BackendGraph capability (CUDA Graph capture/replay)
// ════════════════════════════════════════════════════════════════════════
//
// Decode-loop optimization: eliminate per-kernel launch overhead by
// capturing the full step as a CUDA graph and replaying. CPU/Metal have
// no equivalent — they `impl BackendGraph for X {}` with empty bodies and
// inherit the unsupported / no-op defaults below.
//
// Flow per decode step:
//   1. Caller: `set_decode_state(ctx, token, step)` — memcpy to dev bufs
//   2. Try `replay_graph(ctx, key)`:
//        - Ok(true):  graph replayed, skip eager forward
//        - Ok(false): no captured graph yet, run eager
//        - Err(_):    not supported, run eager
//   3. If running eager and in capture window:
//      - `set_dev_state_mode(ctx, true)` so kernels use _dyn variants
//      - `begin_graph_capture(ctx)`
//      - run forward
//      - `end_graph_capture(ctx, key)` — stores graph on ctx internally
//      - `set_dev_state_mode(ctx, false)` — restore scalar kernels

/// Capability-trait for backends that can capture and replay execution as
/// a graph (CUDA Graph). Models that call these methods bound their
/// generic on `B: BackendGraph`; backends without graph support
/// (Metal, CPU) impl this trait with an empty body and inherit
/// no-op / `unsupported` defaults.
pub trait BackendGraph: Backend {
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

    /// End stream capture and install the captured graph keyed by
    /// `_key` (opaque caller-chosen u64; the model uses `m_padded` so
    /// that different batch shapes don't thrash a single slot).
    fn end_graph_capture(_ctx: &mut Self::Context, _key: u64) -> Result<()> {
        Err(FerrumError::unsupported("graph capture not supported"))
    }

    /// Replay the captured graph for `_key`. Returns `Ok(false)` if no
    /// graph is cached for that key; caller should run eager.
    fn replay_graph(_ctx: &mut Self::Context, _key: u64) -> Result<bool> {
        Ok(false)
    }

    /// Drop the cached graph for `_key` — required when its kernel-arg
    /// pointers (KV cache, scratch) might no longer be valid. Use
    /// `reset_all_graphs` when EVERY cached graph should be evicted
    /// (hard model reload / scratch realloc).
    fn reset_graph(_ctx: &mut Self::Context, _key: u64) {}

    /// Drop ALL cached graphs — used by hard reset paths.
    fn reset_all_graphs(_ctx: &mut Self::Context) {}
}

// ════════════════════════════════════════════════════════════════════════
// BackendCollective capability (NCCL / RCCL multi-rank ops)
// ════════════════════════════════════════════════════════════════════════
//
// Tensor-parallel multi-GPU collective ops. CUDA wires these to NCCL via
// `crate::nccl_comm::NcclRank`; AMD would wire to RCCL similarly. CPU and
// Metal `impl BackendCollective for X {}` with empty bodies, inheriting
// single-rank no-ops (world_size=1, rank=0, ops are identity).

/// Capability-trait for backends that support multi-rank collective ops.
/// Single-GPU backends inherit the no-op defaults: `world_size = 1`,
/// `rank = 0`, and the collective ops are identity. Multi-rank backends
/// (CUDA + NCCL today, AMD + RCCL in the future) override these.
pub trait BackendCollective: Backend {
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

// ════════════════════════════════════════════════════════════════════════
// BackendQuantMarlin capability (CUDA Marlin INT4 / GPTQ)
// ════════════════════════════════════════════════════════════════════════
//
// CUDA-specific INT4 GEMM via Marlin tile kernels (Tensor Core required).
// Metal/CPU don't have Marlin; they `impl BackendQuantMarlin for X {}` empty
// and inherit `unsupported` Err defaults. GPTQ models targeting non-CUDA
// backends are loaded via the dequant-fallback path in the Linear impls.

/// Capability-trait for backends that natively support Marlin INT4 GEMM.
/// CUDA wires this to the Marlin (or vLLM marlin_moe_wna16) tile kernels;
/// other backends inherit defaults that error or no-op.
pub trait BackendQuantMarlin: Backend {
    /// Repack raw GPTQ tensors into a backend-specific `Linear<Self>` impl.
    /// Called once per layer at model load time.
    ///
    /// Inputs are host-side slices (CPU memory) — the loader reads from
    /// safetensors and hands them off; each backend uploads + repacks
    /// per its own strategy. `bits` is typically 4; `group_size` is
    /// typically 128. `bias_host` is optional `[out_features]` f32 (when
    /// the model has fused bias, e.g. Qwen2.5 attention projections).
    ///
    /// Phase 3e/2: returns `Box<dyn Linear<Self>>` directly (CUDA:
    /// `CudaMarlinLinear`, CPU: `CpuGptqLinear`). Kernel dispatch lives
    /// inside the boxed Linear's `forward` — the old `gemm_gptq` trait
    /// method is gone.
    #[allow(clippy::too_many_arguments)]
    fn load_gptq(
        _qweight: &[i32],
        _scales: &[f32],
        _qzeros: &[i32],
        _g_idx: Option<&[i32]>,
        _bias_host: Option<&[f32]>,
        _bits: u32,
        _group_size: usize,
        _k: usize,
        _n: usize,
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        Err(FerrumError::unsupported(
            "load_gptq not implemented for this backend",
        ))
    }
    /// Load num_experts GPTQ weight tiles into ONE stacked store, with
    /// the property that **each expert's packed bytes are contiguous**
    /// in the resulting store. This is what the offset GEMM needs to
    /// dispatch per expert via pointer offset alone.
    ///
    /// Why this is a separate API from `load_gptq` + post-hoc concat:
    /// Marlin's repack permutes data in `[K-tile-row outer, N-tile inner]`
    /// order. A single repack of `concat(all experts along N)` produces
    /// a buffer where expert e's bytes are spread across K-tile-rows,
    /// NOT contiguous. Per-expert repack-then-concat keeps each
    /// expert's data in one contiguous block.
    ///
    /// `qweights[i] / scales[i] / qzeros[i]` are each expert's raw GPTQ
    /// tensors. All share the same K + group_size + bits + g_idx.
    ///
    /// Default returns Err(unsupported); override on backends with a
    /// per-expert MoE GPTQ path.
    /// Phase C step 4e: returns the trait-object `MarlinExpertStack`
    /// directly. Internally, each backend constructs its own opaque
    /// repacked tile (Marlin: per-expert-then-concat; CPU: dequantized
    /// f32 weight slab) and wraps it in the concrete
    /// `{Cuda,Cpu}MarlinExpertStack` impl.
    ///
    /// Removing `Self::GptqStore` from the public API kills the type
    /// leak that previously forced `ExpertStack<B>` to carry
    /// `Option<Arc<B::GptqStore>>`. Adding a new Marlin backend now
    /// only requires implementing this method + a fresh
    /// `MarlinExpertStack<NewBackend>` impl — no Backend trait edits.
    #[allow(clippy::too_many_arguments)]
    fn load_gptq_stacked(
        _qweights: &[&[i32]],
        _scales: &[&[f32]],
        _qzeros: &[&[i32]],
        _g_idx: Option<&[i32]>,
        _bits: u32,
        _group_size: usize,
        _k: usize,
        _n_per_expert: usize,
    ) -> Result<std::sync::Arc<dyn crate::MarlinExpertStack<Self>>> {
        Err(FerrumError::unsupported(
            "load_gptq_stacked not implemented for this backend",
        ))
    }
    // Phase C step 4a: marlin_zero_stacked_workspace — body inlined into
    // MarlinExpertStack::zero_workspace.
    // Phase C step 4b: make_stacked_expert_linear — body inlined into
    // MarlinExpertStack::make_expert_linear.
    // Phase C step 4c+4d: moe_gemm_phase_batched + moe_gemm_phase_vllm —
    // bodies inlined into MarlinExpertStack::gemm_phase_batched /
    // gemm_phase_vllm (concrete impls in quant_linear/{cuda,cpu}_marlin_stack.rs).
    // Phase C step 4e: make_marlin_expert_stack subsumed by
    // load_gptq_stacked (now returns the trait object directly).
    // gemm_gptq_with_offset_strided — body inlined into CpuMarlinExpertStack
    // (the only remaining caller).
    /// Pre-grow any backend-internal scratch slots whose size depends
    /// on `m_total * intermediate_size` (the largest matmul fan-in
    /// inside `unified_forward_internal`). Default no-op. CUDA
    /// implements this to grow the perm-aware Marlin gather scratch
    /// EAGERLY before the caller enters a CUDA-graph capture region —
    /// `cuLaunchKernel` after a runtime alloc inside a captured
    /// stream returns `CUDA_ERROR_INVALID_VALUE`.
    fn pregrow_marlin_gather_scratch(_ctx: &mut Self::Context, _required: usize) {
        // default: no scratch to pre-grow
    }
    // Phase C step 4e: gemm_gptq_with_offset_strided removed —
    // body inlined into CpuMarlinExpertStack (the only caller after
    // step 4c moved the multi-stream pool dispatch into the CUDA
    // free function).
}

// ════════════════════════════════════════════════════════════════════════
// BackendQuantGguf capability (Metal GGUF Q4_K / Q6_K / Q8_0)
// ════════════════════════════════════════════════════════════════════════
//
// Metal-specific GGUF k-quant GEMM/GEMV via simdgroup_matmul shaders.
// CUDA/CPU don't ship GGUF kernels; they `impl BackendQuantGguf for X {}`
// empty and inherit unsupported defaults. GGUF models targeting non-Metal
// backends are loaded via dequant-fallback in the Linear impls.

/// Capability-trait for backends that natively dispatch GGUF k-quant
/// GEMM / GEMV. Metal wires its q4k/q6k shaders here; CUDA/CPU inherit
/// defaults that error.
pub trait BackendQuantGguf: Backend {
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
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        Err(FerrumError::unsupported(
            "load_quant not implemented for this backend",
        ))
    }
    /// Build a fused linear from multiple `(kind, bytes, n_rows)`
    /// parts that share `n_cols`. Used by `GgufLoader::load_fused` when
    /// parts have heterogeneous quant kinds (e.g. Qwen3 qkv_proj where
    /// q+k are Q4_K but v is Q6_K) — byte-concatenation isn't possible,
    /// so each part stays as its own QuantStore and the gemm dispatches
    /// one matvec per part with output offsets.
    ///
    /// Phase 3e/3: returns `Box<dyn Linear<Self>>` directly (Metal:
    /// `MetalGgufLinear` over a `Fused` MetalQuantStore variant).
    fn load_quant_fused(
        _parts: &[(GgufQuantType, &[u8], usize)],
        _n_cols: usize,
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        Err(FerrumError::unsupported(
            "load_quant_fused not implemented for this backend",
        ))
    }
    /// Build a stacked-experts MoE linear from a contiguous 3-D weight
    /// payload `[num_experts, n_rows, n_cols/256]` super-blocks. Used for
    /// the MoE indirect-dispatch fast path; backends without such a kernel
    /// return `Err(unsupported)` and the model code falls back to the
    /// per-expert `Box<dyn Linear<Self>>` loop.
    ///
    /// Phase 3e/4: returns `Box<dyn StackedExpertGgufLinear<Self>>` directly
    /// (Metal: `MetalStackedExpertGgufLinear` over Q4KExperts / Q6KExperts).
    /// Replaces the old `Result<Self::QuantStore>` API + the 7 sibling
    /// `*_moe_id*` Backend methods that consumed it.
    fn load_quant_experts(
        _kind: GgufQuantType,
        _bytes: &[u8],
        _num_experts: usize,
        _n_rows: usize,
        _n_cols: usize,
    ) -> Result<Box<dyn crate::StackedExpertGgufLinear<Self>>> {
        Err(FerrumError::unsupported(
            "load_quant_experts not implemented for this backend",
        ))
    }
}

// ════════════════════════════════════════════════════════════════════════
// BackendMoeFused capability (MoE routing + post-ops kernels)
// ════════════════════════════════════════════════════════════════════════
//
// Backend-specific MoE infrastructure: routing index buffers, expert
// dispatch align, weighted sum / silu/mul fused ops, top-k softmax.
// CUDA + Metal both implement (they're the real MoE backends);
// CPU inherits unsupported defaults.

/// Capability-trait for backends that natively dispatch MoE post-ops + routing.
pub trait BackendMoeFused: Backend {
    /// Routing inputs for `moe_gemm_phase_vllm` — host-built i32 arrays
    /// uploaded once per layer (or per token, depending on caller cadence).
    /// Matches the shape contract of `moe_align_block_size` outputs but is
    /// usable on backends that build the indices on host.
    ///
    /// Buffers are typed Self::Buffer (= fp16 on CUDA) for trait-object
    /// reasons; backends reinterpret as i32. Default returns unsupported.
    fn upload_moe_routing(
        _ctx: &mut Self::Context,
        _sorted_token_ids: &[i32],
        _expert_ids: &[i32],
        _num_tokens_past_padded: &[i32],
    ) -> Result<MoeRouting<Self>> {
        Err(FerrumError::unsupported(
            "upload_moe_routing not implemented for this backend",
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
    /// GPU-side fast-path for the host route() leg of the bucketed
    /// MoE forward (`moe_forward_bucketed` in ferrum-models). Replaces
    /// the `B::sync(ctx) + B::to_vec(logits) + crate::moe::router::
    /// route_into(...)` triple with a single GPU kernel + small D2H of
    /// `[batch, top_k]` ids + weights.
    ///
    /// The backend allocates / reuses its own device-side scratch for
    /// the kernel output; the caller only provides the host destination
    /// vectors (resized + overwritten on each call). Default impl
    /// returns `Err(unsupported)` so non-CUDA callers stay on the host
    /// route_into() path with no behavior change.
    #[allow(clippy::too_many_arguments)]
    fn try_gpu_route_topk_into_host(
        _ctx: &mut Self::Context,
        _logits_dev: &Self::Buffer,
        _out_ids_host: &mut Vec<u32>,
        _out_weights_host: &mut Vec<f32>,
        _batch: usize,
        _num_experts: usize,
        _top_k: usize,
        _norm_topk_prob: bool,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "try_gpu_route_topk_into_host not implemented for this backend",
        ))
    }
    /// GPU-side moe_align_block_size — prep for a future fused MoE
    /// Marlin kernel. Takes per-pair expert assignments (from
    /// [`Self::route_topk_softmax`]) and produces:
    ///   - `sorted_token_ids[N_padded]`: flat list of pair indices
    ///     in [0, batch * top_k), sorted by their assigned expert and
    ///     padded with sentinel `batch * top_k` inside each expert
    ///     group up to a `block_size` boundary.
    ///   - `block_ids[N_padded / block_size]`: which expert each
    ///     `block_size`-row tile of `sorted_token_ids` belongs to.
    ///   - `total_tokens_post_pad[1]`: actual padded token count.
    ///
    /// Layout matches vLLM's marlin_moe_wna16 kernel input
    /// expectation. The fused Marlin kernel reads a row from
    /// `a[sorted_token_ids[i] / top_k]` and weights from
    /// Build `pairs_by_token` + `packed_token_idx` device-side from
    /// device-side `expert_ids`. The counting-sort permutation that
    /// lets `moe_combine` (and the gather step before phase 1 GEMM)
    /// read routing output without a host round-trip — the prerequisite
    /// for graph-capturing the MoE bucketed path.
    ///
    /// Inputs (device):
    /// - `expert_ids: I32 [batch * top_k]` — top-K selected expert ids.
    ///
    /// Outputs (device):
    /// - `pairs_by_token: I32 [batch * top_k]` — sorted-by-expert
    ///   position of each (b, k) pair (the row index into `packed_down`
    ///   that `moe_combine` reads).
    /// - `packed_token_idx: I32 [batch * top_k]` — inverse map: for
    ///   each packed row, the original token b. Used by the gather
    ///   step (`embedding_lookup` of `x` into `x_packed` before phase 1).
    /// - `expert_offsets: I32 [num_experts + 1]` — exclusive prefix
    ///   sum of tokens-per-expert; phase 1/3 dispatchers use it to
    ///   compute each expert's row slice in the packed buffers.
    ///
    /// Default impl returns Err — only CUDA implements this.
    #[allow(clippy::too_many_arguments)]
    fn moe_build_pairs_by_token(
        _ctx: &mut Self::Context,
        _expert_ids: &Self::Buffer,
        _pairs_by_token: &mut Self::Buffer,
        _packed_token_idx: &mut Self::Buffer,
        _expert_offsets: &mut Self::Buffer,
        _batch_x_topk: usize,
        _num_experts: usize,
        _top_k: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "moe_build_pairs_by_token not implemented for this backend",
        ))
    }

    /// `b[block_ids[blockIdx.y] * n_per_expert + ...]`.
    ///
    /// Default impl returns Err — only CUDA implements this.
    #[allow(clippy::too_many_arguments)]
    fn moe_align_block_size(
        _ctx: &mut Self::Context,
        _expert_ids_per_pair: &Self::Buffer,
        _sorted_token_ids: &mut Self::Buffer,
        _block_ids: &mut Self::Buffer,
        _total_tokens_post_pad: &mut Self::Buffer,
        _batch_x_topk: usize,
        _num_experts: usize,
        _block_size: usize,
        _sorted_max_size: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "moe_align_block_size not implemented for this backend",
        ))
    }

    /// vLLM-native align variant: `sorted_token_ids` stores flattened
    /// `(token, top_k_slot)` pair ids, not Ferrum's pre-gathered packed rows.
    /// This lets marlin_moe read gate_up input as `A[pair_id / top_k]`.
    #[allow(clippy::too_many_arguments)]
    fn moe_align_block_size_pair_ids(
        _ctx: &mut Self::Context,
        _expert_ids_per_pair: &Self::Buffer,
        _sorted_token_ids: &mut Self::Buffer,
        _block_ids: &mut Self::Buffer,
        _total_tokens_post_pad: &mut Self::Buffer,
        _batch_x_topk: usize,
        _num_experts: usize,
        _block_size: usize,
        _sorted_max_size: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "moe_align_block_size_pair_ids not implemented for this backend",
        ))
    }

    /// GPU-side bucket sort: turn `[batch, top_k]` selected expert IDs
    /// (from [`Self::route_topk_softmax`]) into `tpe[num_experts]` /
    /// `ids[num_experts * row_stride]` arrays consumed by the batched
    /// MoE GEMM, and emit indirect-dispatch args for the consumer GEMM.
    ///
    /// The `ids` buffer's row stride is `batch * top_k` (worst case);
    /// only the first `tpe[e]` entries of each row are populated. The
    /// consumer GEMM kernel early-exits at `r1 >= tpe[e]`, so the over-
    /// strided indices cost nothing in the inner loop. The grid size,
    /// however, would still be worst-case unless we tighten it — this
    /// is what the `gate_up_args` / `down_args` outputs do: a 12-byte
    /// `(grid_x, grid_y, grid_z)` u32 triple per shape, ready for
    /// `dispatch_thread_groups_indirect`. `grid_x` is shared (depends
    /// only on `max(tpe[e])`); `grid_y` differs because gate/up has
    /// `M = m_gate_up` while down has `M = m_down`.
    ///
    /// All five output buffers are written in one kernel; no host
    /// roundtrip and no per-layer pipeline drain.
    #[allow(clippy::too_many_arguments)]
    fn compute_ids_tpe_gpu(
        _ctx: &mut Self::Context,
        _selected_ids: &Self::Buffer,
        _tpe: &mut Self::Buffer,
        _ids: &mut Self::Buffer,
        _gate_up_args: &mut Self::Buffer,
        _down_args: &mut Self::Buffer,
        _batch: usize,
        _num_experts: usize,
        _top_k: usize,
        _m_gate_up: usize,
        _m_down: usize,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "compute_ids_tpe_gpu not implemented for this backend",
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
    /// Fused weighted-sum-residual + RMSNorm: combines this layer's
    /// `weighted_sum_residual_stacked` with the next layer's leading
    /// `rms_norm` into a single dispatch.
    ///
    /// Computes
    ///   `residual[i] += Σ_s w[s] · slots[s, i]`
    ///   `normed_out[i] = residual[i] · (1 / sqrt(Σ residual² / hidden + eps)) · next_norm_w[i]`
    ///
    /// Caller is responsible for skipping the next layer's standalone
    /// `rms_norm` — `normed_out` IS that layer's `norm_out` input.
    /// Default returns Unsupported.
    #[allow(clippy::too_many_arguments)]
    fn weighted_sum_residual_norm_stacked(
        _ctx: &mut Self::Context,
        _slots: &Self::Buffer,
        _weights: &Self::Buffer,
        _residual: &mut Self::Buffer,
        _next_norm_w: &Self::Buffer,
        _normed_out: &mut Self::Buffer,
        _n_slots: usize,
        _hidden: usize,
        _eps: f32,
    ) -> Result<()> {
        Err(FerrumError::unsupported(
            "weighted_sum_residual_norm_stacked not implemented for this backend",
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
    /// Offset-aware variant of [`Self::weighted_sum_batched`] —
    /// `weights` reads from `weights_offset` (in elements, points at
    /// the start of `[batch, top_k]`), `out` writes from `out_offset`
    /// (in elements, points at start of `[batch, hidden]`). Used by
    /// the per-item batched-decode path to skip `copy_slice` round-trips.
    /// Default falls back to the non-offset variant via two copies.
    #[allow(clippy::too_many_arguments)]
    fn weighted_sum_batched_offset(
        ctx: &mut Self::Context,
        slots: &Self::Buffer,
        weights: &Self::Buffer,
        weights_offset: usize,
        out: &mut Self::Buffer,
        out_offset: usize,
        batch: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()> {
        // Default: stage through scratch — backends override for zero-copy.
        let _ = (
            ctx,
            slots,
            weights,
            weights_offset,
            out,
            out_offset,
            batch,
            top_k,
            hidden,
        );
        Err(FerrumError::unsupported(
            "weighted_sum_batched_offset not implemented for this backend",
        ))
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
    /// Capability probe for [`Self::gemv_quant_moe_id_gate_up_silu`].
    ///
    /// `true` ⇒ the fused kernel is wired in and the caller should
    /// prefer it on the MoE decode hot path. `false` ⇒ caller must use
    /// the 3-dispatch fallback (gate gemv + up gemv + silu_mul_stacked).
    /// Lets callers branch without paying the cost of an `Err(Unsupported)`
    /// allocation per (layer, step).
    fn supports_fused_moe_gate_up_silu() -> bool {
        false
    }
    /// Capability probe for [`Self::gemv_quant_moe_id_batched`].
    fn supports_batched_moe_gemv() -> bool {
        false
    }
    /// Capability probe for [`Self::gemv_quant_moe_id_gate_up_silu_batched`].
    fn supports_batched_moe_gate_up_silu() -> bool {
        false
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
    /// MoE combine: per-token weighted sum across `top_k` expert outputs.
    ///
    /// After the bucketed dispatch, `packed_down` holds `[total_pairs,
    /// hidden]` with one row per (token, k_slot) pair in expert-bucketed
    /// order. `pairs_by_token[b * top_k + k]` is the inverse map: which
    /// row of `packed_down` carries the (b, k_slot) contribution. A row
    /// index of `-1` means "skip" (unused slot).
    ///
    /// Computes:
    ///
    /// ```text
    /// out[b, h] = sum_k pair_weights[b * top_k + k] *
    ///                   packed_down[pairs_by_token[b * top_k + k], h]
    /// ```
    ///
    /// Default impl round-trips via host memory — correct but slow.
    /// CUDA backend launches a single fused kernel.
    ///
    /// Phase D follow-up: `pairs_by_token` (I32) and `pair_weights` (F32)
    /// are now device buffers so callers can build them on-device for
    /// graph capture (was `&[i32]` / `&[f32]` host slices with internal
    /// clone_htod, which records stale host pointers under CUDA Graph
    /// capture replay).
    #[allow(clippy::too_many_arguments)]
    fn moe_combine(
        ctx: &mut Self::Context,
        packed_down: &Self::Buffer,
        pairs_by_token: &Self::Buffer,
        pair_weights: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        hidden: usize,
        top_k: usize,
        total_pairs: usize,
    ) {
        // Reference default: D2H pairs/weights, run the host loop, H2D out.
        // CUDA backend overrides with a single device kernel.
        let _ = ctx;
        let packed = Self::to_vec(packed_down, total_pairs * hidden);
        let pairs_host_f32 = Self::to_vec(pairs_by_token, batch * top_k);
        let weights_host = Self::to_vec(pair_weights, batch * top_k);
        let mut out_h = vec![0.0f32; batch * hidden];
        for b in 0..batch {
            for k in 0..top_k {
                // `to_vec` returns f32; the device-side I32 buffer is
                // bit-cast to f32 by the trait's f16-default to_vec path,
                // so we re-extract via raw transmute. Backends override
                // this default with a typed kernel that doesn't go
                // through f16; on the default path callers are CPU
                // parity tests where the byte pattern is preserved.
                let pair_row = pairs_host_f32[b * top_k + k].to_bits() as i32;
                if pair_row < 0 {
                    continue;
                }
                let w = weights_host[b * top_k + k];
                let src = &packed[(pair_row as usize) * hidden..(pair_row as usize + 1) * hidden];
                let dst = &mut out_h[b * hidden..(b + 1) * hidden];
                for h in 0..hidden {
                    dst[h] += w * src[h];
                }
            }
        }
        *out = Self::from_slice(&out_h);
    }
}
