//! vLLM paged-attention wrappers.
//!
//! Calls the extern "C" launcher in `kernels/vllm_attn/launcher.cu` which
//! invokes vLLM's `paged_attention_v1_kernel` for short single-partition
//! decode, or `paged_attention_v2_kernel` + reduce kernel for longer
//! multi-partition decode. Only the HEAD_SIZE=128, BLOCK_SIZE=16, FP16
//! instantiation needed by Qwen3-30B-A3B is exported.
//!
//! Companion to the `split_qkv_norm_rope_into_paged_cache_varlen_vllm_f16`
//! PTX kernel which writes K/V in vLLM's layout.

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ferrum_types::{FerrumError, Result};
use half::f16;
use std::sync::Arc;

extern "C" {
    fn ferrum_vllm_paged_attention_v1_f16_h128_b16(
        out: *mut std::ffi::c_void,     // __half*
        query: *const std::ffi::c_void, // const __half*
        key_cache: *const std::ffi::c_void,
        value_cache: *const std::ffi::c_void,
        num_kv_heads: i32,
        scale: f32,
        block_tables: *const std::ffi::c_void,
        seq_lens: *const std::ffi::c_void,
        num_seqs: i32,
        num_heads: i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        max_seq_len: i32,
        stream: *mut std::ffi::c_void,
    );

    fn ferrum_vllm_paged_attention_v2_f16_h128_b16(
        out: *mut std::ffi::c_void,        // __half*
        exp_sums: *mut std::ffi::c_void,   // float*
        max_logits: *mut std::ffi::c_void, // float*
        tmp_out: *mut std::ffi::c_void,    // __half*
        query: *const std::ffi::c_void,    // const __half*
        key_cache: *const std::ffi::c_void,
        value_cache: *const std::ffi::c_void,
        num_kv_heads: i32,
        scale: f32,
        block_tables: *const std::ffi::c_void, // const int*
        seq_lens: *const std::ffi::c_void,
        num_seqs: i32,
        num_heads: i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32,
        kv_block_stride: i32,
        kv_head_stride: i32,
        max_seq_len: i32,
        stream: *mut std::ffi::c_void,
    );
}

// Process-global scratch (exp_sums / max_logits / tmp_out).  Mirrors the
// MARLIN_GATHER_SCRATCH pattern — stable GPU addresses so captured graphs
// can replay safely; sized once for the model's max (num_seqs, num_heads,
// max_partitions, head_dim).
struct PagedAttnScratch {
    exp_sums: CudaSlice<f32>,
    max_logits: CudaSlice<f32>,
    tmp_out: CudaSlice<f16>,
    capacity: PagedAttnCapacity,
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct PagedAttnCapacity {
    num_seqs: usize,
    num_heads: usize,
    max_partitions: usize,
    head_dim: usize,
}

static PA_SCRATCH: std::sync::OnceLock<std::sync::RwLock<Option<PagedAttnScratch>>> =
    std::sync::OnceLock::new();
static PA_V1_SHORT_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn pa_scratch_slot() -> &'static std::sync::RwLock<Option<PagedAttnScratch>> {
    PA_SCRATCH.get_or_init(|| std::sync::RwLock::new(None))
}

fn pa_v1_short_enabled() -> bool {
    *PA_V1_SHORT_ENABLED
        .get_or_init(|| std::env::var("FERRUM_VLLM_PAGED_ATTN_V1_SHORT").as_deref() != Ok("0"))
}

fn ensure_pa_scratch(
    stream: &Arc<CudaStream>,
    num_seqs: usize,
    num_heads: usize,
    max_partitions: usize,
    head_dim: usize,
) {
    let need = PagedAttnCapacity {
        num_seqs,
        num_heads,
        max_partitions,
        head_dim,
    };
    {
        let g = pa_scratch_slot().read().expect("PA_SCRATCH poisoned");
        if let Some(s) = g.as_ref() {
            if s.capacity.num_seqs >= need.num_seqs
                && s.capacity.num_heads >= need.num_heads
                && s.capacity.max_partitions >= need.max_partitions
                && s.capacity.head_dim >= need.head_dim
            {
                return;
            }
        }
    }
    // Allocate to the new max (round num_seqs up so growth amortises).
    let cap = PagedAttnCapacity {
        num_seqs: need.num_seqs.max(64),
        num_heads: need.num_heads,
        max_partitions: need.max_partitions.max(8),
        head_dim: need.head_dim,
    };
    let n_floats = cap.num_seqs * cap.num_heads * cap.max_partitions;
    let n_halves = cap.num_seqs * cap.num_heads * cap.max_partitions * cap.head_dim;
    let exp_sums = unsafe { stream.alloc::<f32>(n_floats) }.expect("PA exp_sums alloc");
    let max_logits = unsafe { stream.alloc::<f32>(n_floats) }.expect("PA max_logits alloc");
    let tmp_out = unsafe { stream.alloc::<f16>(n_halves) }.expect("PA tmp_out alloc");
    let mut w = pa_scratch_slot().write().expect("PA_SCRATCH poisoned");
    *w = Some(PagedAttnScratch {
        exp_sums,
        max_logits,
        tmp_out,
        capacity: cap,
    });
}

/// Dispatch vLLM paged attention for the FP16 / HEAD=128 / BLOCK=16 shape.
/// K/V cache must already be in vLLM layout (see
/// `split_qkv_norm_rope_into_paged_cache_varlen_vllm_f16`).
///
/// `q` is `[num_seqs, num_heads, head_dim]` (head-major within seq).
/// `block_tables` is `[num_seqs, max_num_blocks_per_seq]` (i32).
/// `seq_lens` is `[num_seqs]` (i32).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attention_v2(
    stream: &Arc<CudaStream>,
    out: &mut CudaSlice<f16>,
    q: &CudaSlice<f16>,
    k_cache: &CudaSlice<f16>,
    v_cache: &CudaSlice<f16>,
    block_tables: &CudaSlice<u32>,
    seq_lens: &CudaSlice<u32>,
    num_seqs: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_num_blocks_per_seq: usize,
    max_seq_len: usize,
) -> Result<()> {
    const PARTITION_SIZE: usize = 512;
    if head_dim != 128 {
        return Err(FerrumError::unsupported(format!(
            "vllm paged_attn_v2: only head_dim=128 instantiated, got {head_dim}"
        )));
    }
    if block_size != 16 {
        return Err(FerrumError::unsupported(format!(
            "vllm paged_attn_v2: only block_size=16 instantiated, got {block_size}"
        )));
    }
    // Layout strides (in halves):
    //   K, V cache  : [num_blocks, num_kv_heads, head_dim*block_size]
    //   query       : [num_seqs, num_heads, head_dim]
    let q_stride = (num_heads * head_dim) as i32;
    let kv_block_stride = (num_kv_heads * head_dim * block_size) as i32;
    let kv_head_stride = (head_dim * block_size) as i32;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let raw_stream = stream.cu_stream() as *mut std::ffi::c_void;

    let use_v1_short = max_seq_len <= PARTITION_SIZE && pa_v1_short_enabled();
    if use_v1_short {
        unsafe {
            let (out_dp, _o_recs) = out.device_ptr_mut(stream);
            let (q_dp, _q_recs) = q.device_ptr(stream);
            let (k_dp, _k_recs) = k_cache.device_ptr(stream);
            let (v_dp, _v_recs) = v_cache.device_ptr(stream);
            let (bt_dp, _bt_recs) = block_tables.device_ptr(stream);
            let (sl_dp, _sl_recs) = seq_lens.device_ptr(stream);
            ferrum_vllm_paged_attention_v1_f16_h128_b16(
                out_dp as *mut std::ffi::c_void,
                q_dp as *const std::ffi::c_void,
                k_dp as *const std::ffi::c_void,
                v_dp as *const std::ffi::c_void,
                num_kv_heads as i32,
                scale,
                bt_dp as *const std::ffi::c_void,
                sl_dp as *const std::ffi::c_void,
                num_seqs as i32,
                num_heads as i32,
                max_num_blocks_per_seq as i32,
                q_stride,
                kv_block_stride,
                kv_head_stride,
                max_seq_len as i32,
                raw_stream,
            );
        }
        return Ok(());
    }

    let max_partitions = max_seq_len.div_ceil(PARTITION_SIZE).max(1);
    ensure_pa_scratch(stream, num_seqs, num_heads, max_partitions, head_dim);

    let slot = pa_scratch_slot();
    let mut sg = slot.write().expect("PA_SCRATCH poisoned");
    let scratch = sg.as_mut().expect("ensure_pa_scratch must have populated");

    // SAFETY: all pointers come from CudaSlice (device pointers via cudarc).
    //   The kernel reads/writes them on `stream` and we issue a launch on
    //   the same stream. No host-pointer captures.
    unsafe {
        let (out_dp, _o_recs) = out.device_ptr_mut(stream);
        let (es_dp, _es_recs) = scratch.exp_sums.device_ptr_mut(stream);
        let (ml_dp, _ml_recs) = scratch.max_logits.device_ptr_mut(stream);
        let (to_dp, _to_recs) = scratch.tmp_out.device_ptr_mut(stream);
        let (q_dp, _q_recs) = q.device_ptr(stream);
        let (k_dp, _k_recs) = k_cache.device_ptr(stream);
        let (v_dp, _v_recs) = v_cache.device_ptr(stream);
        let (bt_dp, _bt_recs) = block_tables.device_ptr(stream);
        let (sl_dp, _sl_recs) = seq_lens.device_ptr(stream);
        // cudarc 0.19's `device_ptr*` returns `(u64, _records)` where the
        // u64 is the raw device address. Cast u64 → typed pointer directly
        // (the intermediate `as *const _` form fails to infer the element
        // type when the destination is `*const c_void`).
        ferrum_vllm_paged_attention_v2_f16_h128_b16(
            out_dp as *mut std::ffi::c_void,
            es_dp as *mut std::ffi::c_void,
            ml_dp as *mut std::ffi::c_void,
            to_dp as *mut std::ffi::c_void,
            q_dp as *const std::ffi::c_void,
            k_dp as *const std::ffi::c_void,
            v_dp as *const std::ffi::c_void,
            num_kv_heads as i32,
            scale,
            bt_dp as *const std::ffi::c_void,
            sl_dp as *const std::ffi::c_void,
            num_seqs as i32,
            num_heads as i32,
            max_num_blocks_per_seq as i32,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            max_seq_len as i32,
            raw_stream,
        );
    }
    Ok(())
}
