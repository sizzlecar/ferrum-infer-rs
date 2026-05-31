//! Backend data types shared by the capability traits and model code.

use half::{bf16, f16};

use super::traits::{Backend, BackendKvDtype};
use ferrum_interfaces::kv_dtype::{KvDtypeKind, KvFp16};

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

/// Per-layer KV cache. Each model owns its own `Vec<KvCache<B, K>>` per
/// sequence. The `K: KvDtypeKind` parameter selects the cache element
/// type — defaults to [`KvFp16`] so existing call sites that wrote
/// `KvCache<B>` keep compiling unchanged.
///
/// Two layouts are supported, selected at allocation time:
/// 1. **Contiguous** (default): `k`/`v` are `[num_kv_heads, capacity, head_dim]`
///    f32 buffers. `block_size == 0` and `block_table` / `context_lens` are
///    `None`. Original ferrum layout — used when `FERRUM_METAL_PAGED_KV` is
///    unset.
/// 2. **Paged** (vLLM-style): `k`/`v` are `[num_blocks, num_kv_heads,
///    block_size, head_dim]` block pools. `block_size > 0` and
///    `block_table` (`u32[max_num_blocks_per_seq]`) + `context_lens`
///    (`u32[1]` single-seq for now) are populated. Multi-seq sharing
///    is a Phase 4 concern; today every paged cache_id has its own
///    pool but the kernel-level indirection works.
///
/// The `K` parameter is currently a phantom-type marker — the buffer
/// fields stay `B::Buffer` regardless. Future PRs will switch backends
/// to `BackendKvDtype<KvInt8>` etc. and the kernel dispatch will read
/// `K::NAME` / `K::BYTES_PER_ELEM` to pick the right append / attention
/// kernel without any `KvCache` struct change.
pub struct KvCache<B: Backend, K: KvDtypeKind = KvFp16> {
    pub k: B::Buffer,
    pub v: B::Buffer,
    pub len: usize,
    pub capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Paged: KV positions per physical block. `0` => contiguous layout.
    pub block_size: usize,
    /// Paged: `[max_num_blocks_per_seq]` u32 — logical → physical block.
    pub block_table: Option<B::Buffer>,
    /// Paged: `[1]` u32 — current context length for the kernel to read.
    pub context_lens: Option<B::Buffer>,
    /// Paged: host-side mirror of the physical block indices owned by
    /// this cache. Lets the model's release path return blocks to the
    /// shared allocator without reading them back from device.
    pub paged_block_indices: Vec<u32>,
    /// Marker — KV cache element type. Zero-sized.
    pub _kv_dtype: std::marker::PhantomData<K>,
}

/// Quantized-KV cache (Dim 5 INT8 / future FP8 paths). Sibling of
/// [`KvCache`] for backends that store K/V in a non-FP16 element type
/// plus per-token per-kv-head scales.
///
/// Why a separate struct: the FP16 `KvCache<B, K>` uses `B::Buffer`
/// uniformly, which is FP16 on every concrete backend. Stuffing INT8
/// storage into that buffer would require unsafe transmutes; making
/// the FP16 struct generic over the storage type would force every
/// existing call site (4 model files, ~20 functions) to pick up an
/// equality-bound on the associated type. Keeping a parallel struct
/// for INT8 is the cheaper trade — the kernel launchers in
/// [`crate::int8_kv`] take cudarc primitives directly anyway.
///
/// `KStorage` and `ScaleStorage` come from `BackendKvDtype<K>::KvBuffer`
/// and `BackendKvDtype<K>::KvScales`. On CUDA they wrap `CudaSlice<i8>`
/// and `CudaSlice<f16>`.
pub struct KvCacheQuant<B: BackendKvDtype<K>, K: KvDtypeKind> {
    pub k: <B as BackendKvDtype<K>>::KvBuffer,
    pub v: <B as BackendKvDtype<K>>::KvBuffer,
    pub k_scales: <B as BackendKvDtype<K>>::KvScales,
    pub v_scales: <B as BackendKvDtype<K>>::KvScales,
    pub len: usize,
    pub capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub block_table: Option<B::Buffer>,
    pub context_lens: Option<B::Buffer>,
    pub paged_block_indices: Vec<u32>,
    pub _kv_dtype: std::marker::PhantomData<K>,
}

/// Routing buffers consumed by `moe_gemm_phase_vllm` — held by the
/// caller across phase 1 and phase 3 of one MoE forward. All three
/// fields are i32 device tensors in disguise (`Self::Buffer = fp16` on
/// CUDA; the backend reinterprets the underlying device pointer).
pub struct MoeRouting<B: Backend + ?Sized> {
    pub sorted_token_ids: B::Buffer,
    pub expert_ids: B::Buffer,
    pub num_tokens_past_padded: B::Buffer,
}
