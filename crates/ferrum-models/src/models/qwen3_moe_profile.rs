//! Profiling counters shared by the Qwen3-MoE forward paths.

use std::sync::atomic::{AtomicBool, AtomicU64};

// Decode-side per-op profile counters — same names as the dense path
// so existing tooling (`FERRUM_DECODE_OP_PROFILE=1` log scrapers) keeps
// working without a separate switch for MoE.
pub(crate) static ATTN_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_TIME_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_CALLS: AtomicU64 = AtomicU64::new(0);

// Fine-grained decode-only counters, populated by
// `moe_forward_stacked_decode_impl` when FERRUM_DECODE_OP_PROFILE is set.
// Each is per-layer summed over the layers in one decode token; drained
// at the bottom of `decode_internal`.
pub(crate) static DEC_ROUTE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_GATE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_UP_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_SILU_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_WSUM_US: AtomicU64 = AtomicU64::new(0);
// Single-shot per decode token (not per-layer).
pub(crate) static DEC_EMBED_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_FINAL_NORM_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static DEC_LM_HEAD_US: AtomicU64 = AtomicU64::new(0);

pub(crate) static MOE_GRAPH_UNCLEAN_WARNED: AtomicBool = AtomicBool::new(false);

// MoE batched-prefill sub-stage counters (gate / up / down mul_mm_id +
// silu + weighted_sum + host topk). Same FERRUM_DECODE_OP_PROFILE gate.
pub(crate) static MOE_PREFILL_HOST_TOPK_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_HOST_TOPK_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_GATE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_GATE_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_UP_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_UP_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_SILU_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_SILU_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_DOWN_CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_WSUM_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREFILL_WSUM_CALLS: AtomicU64 = AtomicU64::new(0);

// MoE batched-DECODE sub-stage counters (small-m path that uses the
// batched-pair GEMV in place of the per-token loop).
pub(crate) static MOE_BATCHED_DECODE_ROUTE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_GATE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_UP_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_SILU_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_DOWN_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_BATCHED_DECODE_WSUM_US: AtomicU64 = AtomicU64::new(0);

// Coarse stage counters for `forward_layer_batched_decode` so we can
// see where the time goes without per-op instrumentation. Summed
// across all layers in one decode_batch_internal call.
pub(crate) static BD_DENSE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static BD_ATTN_PERITEM_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static BD_MOE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static BD_LAYER_CALLS: AtomicU64 = AtomicU64::new(0);
