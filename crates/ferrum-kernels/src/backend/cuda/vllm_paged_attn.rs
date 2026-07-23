//! vLLM paged-attention wrappers.
//!
//! Calls the extern "C" launcher in `kernels/vllm_attn/launcher.cu` which
//! invokes vLLM's `paged_attention_v1_kernel` for short single-partition
//! decode, or `paged_attention_v2_kernel` + reduce kernel for longer
//! multi-partition decode. Only the HEAD_SIZE=128, BLOCK_SIZE=16, FP16
//! instantiations needed by the Qwen3/Qwen3.5 CUDA lanes are exported.
//!
//! Companion to the `split_qkv_norm_rope_into_paged_cache_varlen_vllm_f16`
//! PTX kernel which writes K/V in vLLM's layout.

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ferrum_types::{FerrumError, Result};
use half::f16;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

pub(crate) const VNEXT_VLLM_BLOCK_TOKENS: u64 = 16;
pub(crate) const VNEXT_VLLM_PARTITION_TOKENS: u64 = 512;

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

    fn ferrum_vllm_paged_attention_v1_f16_h256_b16(
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

    fn ferrum_vllm_paged_attention_v2_f16_h256_b16(
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

    fn ferrum_vnext_vllm_paged_attention_v1_f16_h128_b16_addressed(
        out: *mut std::ffi::c_void,
        query: *const std::ffi::c_void,
        num_kv_heads: i32,
        scale: f32,
        block_addresses: *const std::ffi::c_void,
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

    fn ferrum_vnext_vllm_paged_attention_v1_f16_h256_b16_addressed(
        out: *mut std::ffi::c_void,
        query: *const std::ffi::c_void,
        num_kv_heads: i32,
        scale: f32,
        block_addresses: *const std::ffi::c_void,
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

    fn ferrum_vnext_vllm_paged_attention_v2_f16_h128_b16_addressed(
        out: *mut std::ffi::c_void,
        exp_sums: *mut std::ffi::c_void,
        max_logits: *mut std::ffi::c_void,
        tmp_out: *mut std::ffi::c_void,
        query: *const std::ffi::c_void,
        num_kv_heads: i32,
        scale: f32,
        block_addresses: *const std::ffi::c_void,
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

    fn ferrum_vnext_vllm_paged_attention_v2_f16_h256_b16_addressed(
        out: *mut std::ffi::c_void,
        exp_sums: *mut std::ffi::c_void,
        max_logits: *mut std::ffi::c_void,
        tmp_out: *mut std::ffi::c_void,
        query: *const std::ffi::c_void,
        num_kv_heads: i32,
        scale: f32,
        block_addresses: *const std::ffi::c_void,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VnextAddressedPagedAttentionKernel {
    V1,
    V2,
}

impl VnextAddressedPagedAttentionKernel {
    pub(crate) fn for_sequence_length(sequence_tokens: u64) -> Self {
        if sequence_tokens <= VNEXT_VLLM_PARTITION_TOKENS {
            Self::V1
        } else {
            Self::V2
        }
    }

    pub(crate) fn native_kernel_id(self) -> &'static str {
        match self {
            Self::V1 => "vllm.paged_attention_v1.addressed",
            Self::V2 => "vllm.paged_attention_v2.addressed",
        }
    }

    pub(crate) fn dispatch_count(self) -> u64 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dispatch_vnext_addressed_paged_attention_raw(
    stream: &CudaStream,
    out: u64,
    query: u64,
    block_addresses: u64,
    sequence_length_device: u64,
    sequence_length: u64,
    exp_sums: Option<u64>,
    max_logits: Option<u64>,
    temporary_output: Option<u64>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    max_num_blocks_per_seq: i32,
) -> Result<VnextAddressedPagedAttentionKernel> {
    if out == 0
        || query == 0
        || block_addresses == 0
        || sequence_length_device == 0
        || sequence_length == 0
    {
        return Err(FerrumError::model(
            "vNext addressed paged attention received a null pointer or empty sequence",
        ));
    }
    if !matches!(head_dim, 128 | 256)
        || num_heads <= 0
        || num_kv_heads <= 0
        || num_heads % num_kv_heads != 0
        || max_num_blocks_per_seq <= 0
    {
        return Err(FerrumError::unsupported(format!(
            "vNext addressed paged attention requires heads>0, divisible GQA, head_dim=128/256, and a non-empty block table; got heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, blocks={max_num_blocks_per_seq}"
        )));
    }
    let sequence_length_i32 = i32::try_from(sequence_length).map_err(|_| {
        FerrumError::model("vNext addressed paged attention sequence length exceeds i32")
    })?;
    let required_blocks = sequence_length.div_ceil(VNEXT_VLLM_BLOCK_TOKENS);
    if u64::try_from(max_num_blocks_per_seq).unwrap_or(0) < required_blocks {
        return Err(FerrumError::model(format!(
            "vNext addressed paged attention block table has {max_num_blocks_per_seq} entries but needs {required_blocks}"
        )));
    }
    let q_stride = num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| FerrumError::model("vNext addressed query stride overflows"))?;
    let kv_head_stride = head_dim
        .checked_mul(VNEXT_VLLM_BLOCK_TOKENS as i32)
        .ok_or_else(|| FerrumError::model("vNext addressed KV head stride overflows"))?;
    let kv_block_stride = num_kv_heads
        .checked_mul(kv_head_stride)
        .ok_or_else(|| FerrumError::model("vNext addressed KV block stride overflows"))?;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let raw_stream = stream.cu_stream() as *mut std::ffi::c_void;
    let kernel = VnextAddressedPagedAttentionKernel::for_sequence_length(sequence_length);
    match kernel {
        VnextAddressedPagedAttentionKernel::V1 => {
            if head_dim == 128 {
                ferrum_vnext_vllm_paged_attention_v1_f16_h128_b16_addressed(
                    out as *mut std::ffi::c_void,
                    query as *const std::ffi::c_void,
                    num_kv_heads,
                    scale,
                    block_addresses as *const std::ffi::c_void,
                    sequence_length_device as *const std::ffi::c_void,
                    1,
                    num_heads,
                    max_num_blocks_per_seq,
                    q_stride,
                    kv_block_stride,
                    kv_head_stride,
                    sequence_length_i32,
                    raw_stream,
                );
            } else {
                ferrum_vnext_vllm_paged_attention_v1_f16_h256_b16_addressed(
                    out as *mut std::ffi::c_void,
                    query as *const std::ffi::c_void,
                    num_kv_heads,
                    scale,
                    block_addresses as *const std::ffi::c_void,
                    sequence_length_device as *const std::ffi::c_void,
                    1,
                    num_heads,
                    max_num_blocks_per_seq,
                    q_stride,
                    kv_block_stride,
                    kv_head_stride,
                    sequence_length_i32,
                    raw_stream,
                );
            }
        }
        VnextAddressedPagedAttentionKernel::V2 => {
            let (Some(exp_sums), Some(max_logits), Some(temporary_output)) =
                (exp_sums, max_logits, temporary_output)
            else {
                return Err(FerrumError::model(
                    "vNext addressed paged attention v2 requires caller-owned scratch",
                ));
            };
            if exp_sums == 0 || max_logits == 0 || temporary_output == 0 {
                return Err(FerrumError::model(
                    "vNext addressed paged attention v2 received null scratch",
                ));
            }
            if head_dim == 128 {
                ferrum_vnext_vllm_paged_attention_v2_f16_h128_b16_addressed(
                    out as *mut std::ffi::c_void,
                    exp_sums as *mut std::ffi::c_void,
                    max_logits as *mut std::ffi::c_void,
                    temporary_output as *mut std::ffi::c_void,
                    query as *const std::ffi::c_void,
                    num_kv_heads,
                    scale,
                    block_addresses as *const std::ffi::c_void,
                    sequence_length_device as *const std::ffi::c_void,
                    1,
                    num_heads,
                    max_num_blocks_per_seq,
                    q_stride,
                    kv_block_stride,
                    kv_head_stride,
                    sequence_length_i32,
                    raw_stream,
                );
            } else {
                ferrum_vnext_vllm_paged_attention_v2_f16_h256_b16_addressed(
                    out as *mut std::ffi::c_void,
                    exp_sums as *mut std::ffi::c_void,
                    max_logits as *mut std::ffi::c_void,
                    temporary_output as *mut std::ffi::c_void,
                    query as *const std::ffi::c_void,
                    num_kv_heads,
                    scale,
                    block_addresses as *const std::ffi::c_void,
                    sequence_length_device as *const std::ffi::c_void,
                    1,
                    num_heads,
                    max_num_blocks_per_seq,
                    q_stride,
                    kv_block_stride,
                    kv_head_stride,
                    sequence_length_i32,
                    raw_stream,
                );
            }
        }
    }
    Ok(kernel)
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

static PA_SCRATCH: std::sync::OnceLock<std::sync::RwLock<HashMap<usize, PagedAttnScratch>>> =
    std::sync::OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VllmPagedAttnRuntimeConfig {
    v1_short: bool,
}

impl VllmPagedAttnRuntimeConfig {
    fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut config = Self { v1_short: true };
        for (name, value) in vars {
            if name.as_ref() == "FERRUM_VLLM_PAGED_ATTN_V1_SHORT" {
                config.v1_short = value.as_ref() != "0";
            }
        }
        config
    }
}

fn vllm_paged_attn_runtime_config() -> &'static VllmPagedAttnRuntimeConfig {
    static CONFIG: OnceLock<VllmPagedAttnRuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(VllmPagedAttnRuntimeConfig::from_env)
}

fn pa_scratch_slots() -> &'static std::sync::RwLock<HashMap<usize, PagedAttnScratch>> {
    PA_SCRATCH.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
}

fn pa_v1_short_enabled() -> bool {
    vllm_paged_attn_runtime_config().v1_short
}

fn ensure_pa_scratch(
    stream: &Arc<CudaStream>,
    ordinal: usize,
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
        let g = pa_scratch_slots().read().expect("PA_SCRATCH poisoned");
        if let Some(s) = g.get(&ordinal) {
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
    let mut w = pa_scratch_slots().write().expect("PA_SCRATCH poisoned");
    w.insert(
        ordinal,
        PagedAttnScratch {
            exp_sums,
            max_logits,
            tmp_out,
            capacity: cap,
        },
    );
}

/// Dispatch vLLM paged attention for the FP16 / HEAD={128,256} / BLOCK=16
/// shapes used by the Qwen3/Qwen3.5 CUDA lanes.
/// K/V cache must already be in vLLM layout (see
/// `split_qkv_norm_rope_into_paged_cache_varlen_vllm_f16`).
///
/// `q` is `[num_seqs, num_heads, head_dim]` (head-major within seq).
/// `block_tables` is `[num_seqs, max_num_blocks_per_seq]` (i32).
/// `seq_lens` is `[num_seqs]` (i32).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attention_v2(
    stream: &Arc<CudaStream>,
    ordinal: usize,
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
    if !matches!(head_dim, 128 | 256) {
        return Err(FerrumError::unsupported(format!(
            "vllm paged_attn_v2: only head_dim=128/256 instantiated, got {head_dim}"
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
            if head_dim == 128 {
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
            } else {
                ferrum_vllm_paged_attention_v1_f16_h256_b16(
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
        }
        return Ok(());
    }

    let max_partitions = max_seq_len.div_ceil(PARTITION_SIZE).max(1);
    ensure_pa_scratch(
        stream,
        ordinal,
        num_seqs,
        num_heads,
        max_partitions,
        head_dim,
    );

    let mut sg = pa_scratch_slots().write().expect("PA_SCRATCH poisoned");
    let scratch = sg
        .get_mut(&ordinal)
        .expect("ensure_pa_scratch must have populated");

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
        if head_dim == 128 {
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
        } else {
            ferrum_vllm_paged_attention_v2_f16_h256_b16(
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
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{VllmPagedAttnRuntimeConfig, VnextAddressedPagedAttentionKernel};

    #[test]
    fn vllm_paged_attn_runtime_config_defaults_short_v1_on() {
        let config = VllmPagedAttnRuntimeConfig::from_env_vars(std::iter::empty::<(&str, &str)>());
        assert!(config.v1_short);
    }

    #[test]
    fn vllm_paged_attn_runtime_config_parses_short_v1_opt_out() {
        let config =
            VllmPagedAttnRuntimeConfig::from_env_vars([("FERRUM_VLLM_PAGED_ATTN_V1_SHORT", "0")]);
        assert!(!config.v1_short);
    }

    #[test]
    fn vnext_addressed_dispatch_is_typed_and_env_independent() {
        let v1 = VnextAddressedPagedAttentionKernel::for_sequence_length(512);
        let v2 = VnextAddressedPagedAttentionKernel::for_sequence_length(513);
        assert_eq!(v1, VnextAddressedPagedAttentionKernel::V1);
        assert_eq!(v2, VnextAddressedPagedAttentionKernel::V2);
        assert_eq!(v1.native_kernel_id(), "vllm.paged_attention_v1.addressed");
        assert_eq!(v2.native_kernel_id(), "vllm.paged_attention_v2.addressed");
        assert_eq!(v1.dispatch_count(), 1);
        assert_eq!(v2.dispatch_count(), 2);
    }
}
