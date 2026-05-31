//! `Qwen3MoeModel<B>` — Qwen3-MoE family decoder (Qwen3-30B-A3B and friends).
//!
//! Architectural delta vs [`LlamaFamilyModel`]:
//!   * Each transformer layer's FFN is a top-K MoE block instead of a
//!     fused `gate_up_proj → silu → down_proj` MLP.
//!     - One small router linear (`[hidden] → [num_experts]`) picks
//!       top-K experts per token.
//!     - Each expert is itself a fused `gate_up + down` MLP with the
//!       same SwiGLU + RMSNorm structure as the dense path, just with
//!       `expert_intermediate_size` (typically much smaller than the
//!       dense `intermediate_size`).
//!     - Output is the weight-summed combination of the K selected
//!       expert outputs.
//!   * Attention path is unchanged from dense Qwen3 (GQA + QK-norm + RoPE).
//!
//! Implementation re-uses the dense layer's attention machinery
//! verbatim — RMSNorm, fused QKV, QK-norm + RoPE, KV cache append,
//! flash attention, O-projection, residual + post-norm. The only new
//! code is the MoE FFN block at the tail of each layer's forward.
//!
//! Memory model: experts are loaded as `QuantLinear<B>` per expert,
//! slicing the on-disk 3-D `ffn_{gate,up,down}_exps.weight` tensors
//! byte-wise so weights stay compressed (Q4_K / Q6_K). For a 32 GB
//! Mac to run Qwen3-30B-A3B at all, this is non-negotiable: an
//! eager-fp32 expert stack would weigh ~110 GB.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use ferrum_bench_core::{global_profile, profile_fields_from_json};
use ferrum_interfaces::kv_dtype::{KvDtypeKind, KvFp16};
use ferrum_kernels::backend::{
    Backend, BackendGraph, BackendMoeFused, BackendPagedKv, BackendQuantGguf, BackendQuantMarlin,
    KvCache, LlmBackend, MoeLlmBackend, QuantLlmBackend,
};
use ferrum_quantization::WeightLoader;
use ferrum_types::{FerrumError, Result};

use crate::common::paged_pool::{block_hash_chain, BlockHash};
use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig};
use crate::models::llama_family::{LlamaFamilyConfig, LlamaFamilyLayer, RopeCache};
use crate::models::qwen3_moe_profile::*;
use crate::models::qwen3_moe_runtime::Qwen3MoeRuntimeEnv;
use crate::moe::{moe_forward, ExpertStack};
use crate::moe_config::Qwen3MoeConfig;

mod api;
mod decode_batch;
mod forward_layer;
mod kv;
mod load;
mod prefill_decode;
mod prefix_cache;
mod scratch;

pub use scratch::{Qwen3MoeLayerState, Qwen3MoeScratch};

/// Qwen3-MoE decoder model.
///
/// Holds the same per-layer attention weights as [`LlamaFamilyModel`]
/// plus a [`Qwen3MoeLayerState`] per layer for the MoE FFN. Routing,
/// expert dispatch, and weighted combine all happen inside
/// [`moe_forward`]; this struct only owns the storage and orchestrates
/// the per-layer call sequence.
pub struct Qwen3MoeModel<B: MoeLlmBackend, K: KvDtypeKind = KvFp16> {
    pub cfg: Qwen3MoeConfig,
    pub runtime_cfg: LlmRuntimeConfig,
    pub(crate) runtime_env: Qwen3MoeRuntimeEnv,

    pub embed: B::Buffer,
    /// Per-layer attention weights (re-uses dense `LlamaFamilyLayer`).
    pub attn_layers: Vec<LlamaFamilyLayer<B>>,
    /// Per-layer MoE state (router + expert stack).
    pub moe_layers: Vec<Qwen3MoeLayerState<B>>,
    pub final_norm_w: B::Buffer,
    pub lm_head: Box<dyn ferrum_quantization::Linear<B>>,

    pub rope: RopeCache<B>,
    pub scratch: Qwen3MoeScratch<B>,

    pub kv_caches: HashMap<String, Vec<KvCache<B, K>>>,
    kv_free_pool: Vec<Vec<KvCache<B, K>>>,

    // ── Paged-KV multi-seq state ────────────────────────────────────────
    //
    // Mirrors `LlamaFamilyModel`. Only populated when
    // `FERRUM_METAL_PAGED_KV=1`. Kv_caches entries become metadata-only
    // views (block_table + context_lens) into the shared `paged_pools`.
    pub paged_pools: Option<Vec<(B::Buffer, B::Buffer)>>,
    // Optional second paged K/V pool in Ferrum's legacy/FlashAttention-friendly
    // layout `[block, slot, kv_head, head_dim]`. Used only for unified
    // prefill/mixed varlen attention when vLLM decode keeps the primary pool
    // in vLLM's paged-decode layout.
    pub paged_fa_pools: Option<Vec<(B::Buffer, B::Buffer)>>,
    pub paged_block_alloc: Option<std::sync::Mutex<crate::common::paged_pool::BlockAllocator>>,
    // Paged-batch dispatch dimensions. Pinned at the first `ensure_kv`
    // when paged-KV is on. Stored on the model (not on scratch) so
    // `ensure_scratch`'s realloc — which wipes scratch's
    // `paged_batch_block_tables`/`paged_batch_q`/etc. — can re-call
    // `enable_paged_batch` with the same dims afterwards. Without this
    // re-init the next forward enters `forward_layer_batched_decode`
    // and panics on `paged_batch_block_tables missing`.
    pub paged_dims: Option<(usize, usize)>, // (max_seqs, max_blocks_per_seq)

    // ── CUDA Graph capture state (FERRUM_MOE_GRAPH=1) ────────────────────
    //
    // Mirrors `LlamaFamilyModel`'s batched_graph_* fields. Captures the
    // layer loop + final rms_norm + lm_head into one graph keyed by
    // `m_padded` (next_power_of_two of batch size). Empty until the
    // first warmup completes; resets on `release(cid)` / `reset()`.
    //
    // Phase 3 prerequisites are now satisfied:
    //   * Phase 2: moe_forward_bucketed runs entirely device-side under
    //     FERRUM_MOE_DEVICE_ROUTE=1 + FERRUM_VLLM_MOE=1 (no D2H sync,
    //     no host pointer recording)
    //   * Phase 3a: paged-batch scratch (block_tables / context_lens /
    //     pos_offsets / cu_seqlens_q) pre-populated once before the
    //     layer loop, so per-layer write_typed is gone
    //   * Phase 3c: capture/replay wrapper (this state's consumer)
    pub(crate) batched_graph_warmup: usize,
    pub(crate) batched_graph_failed: bool,
    pub(crate) batched_graph_keys_seen: std::collections::HashSet<u64>,

    // ── vLLM paged_attention_v2 opt-in ──────────────────────────────────
    //
    // True when (a) the backend reports `supports_vllm_paged_attn() == true`
    // (CUDA built with the `vllm-paged-attn-v2` feature) AND (b) the user
    // did not force `FERRUM_USE_VLLM_PAGED_ATTN=0`. When set, the model writes K/V
    // in vLLM's layout end-to-end (prefill + decode) and uses vLLM's
    // `paged_attention_v2` multi-partition kernel for decode reads.
    //
    // Cached once at construction — flipping at runtime would corrupt the
    // KV cache (the layouts are not compatible).
    pub(crate) use_vllm_paged_attn: bool,
}
