//! Metal quant: GGUF k-quant container + matmul/MoE dispatchers + the
//! `BackendQuantGguf` impl. Extracted from the monolithic `metal.rs`
//! (Audit #8 split mirror of the cuda/quant.rs layout).
//!
//! Public surface preserved via `pub use` in `super` (metal/mod.rs), so
//! external `crate::backend::metal::{MetalQuantStore, ...}` paths keep
//! working without changes.

use super::{slice_is_in_registered_mmap, st, MetalBackend};
use crate::backend::{Backend, GgufQuantType};
use ferrum_types::{FerrumError, Result};
use metal::MTLResourceOptions;
use std::ffi::c_void;
use std::sync::OnceLock;

// ── Profiling counters for `gemm_quant` (off by default) ─────────────

static QUANT_GEMM_TIME_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static QUANT_GEMM_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static QUANT_GEMM_LAST_M: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static QUANT_GEMM_LAST_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static QUANT_GEMM_LAST_K: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn debug_per_call_flush() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("FERRUM_METAL_QUANT_PROFILE").is_ok())
}

// ── MetalQuantStore (GGUF k-quant container) ──────────────────────────

/// Metal-side container for any GGUF k-quant flavour. Each variant
/// keeps its raw on-disk block bytes in MTLBuffer and dequants on
/// demand inside `gemm_quant` (per-call transient fp16 buffer).
///
/// Persistent footprint stays at the on-disk Q4 size (~5 GB for an 8B
/// Q4_K_M) instead of inflating to fp16 (~16 GB) or fp32 (~32 GB).
/// New k-quant types add new variants (and matched dequant kernel +
/// `gemm_quant` arm) without touching the trait surface.
pub enum MetalQuantStore {
    Q4K {
        /// Either a private allocation owning `[n_blocks * 144]` bytes
        /// (copy fallback) or a clone of the shared zero-copy mmap buffer.
        /// Use `setBuffer:offset:` with `byte_offset` to bind the right
        /// region either way.
        blocks: metal::Buffer,
        /// Byte offset into `blocks` where this tensor's payload starts.
        /// `0` for the copy fallback; non-zero when `blocks` is the
        /// shared mmap buffer.
        byte_offset: u64,
        n_rows: usize,
        n_cols: usize,
        n_blocks: usize,
    },
    Q6K {
        /// See `Q4K::blocks`.
        blocks: metal::Buffer,
        byte_offset: u64,
        n_rows: usize,
        n_cols: usize,
        n_blocks: usize,
    },
    /// Row-concatenation of multiple parts that may have different
    /// quant types. Each part is a leaf (Q4K / Q6K) with the same
    /// `n_cols`. Used for `qkv_proj` on Qwen3 Q4_K_M, where q & k
    /// are Q4_K but v is Q6_K — bytes can't be concatenated, so we
    /// keep them separate and dispatch one gemv per part with output
    /// offsets.
    Fused {
        parts: Vec<MetalQuantStore>,
        total_rows: usize,
        n_cols: usize,
    },
    /// 3-D expert stack for MoE indirect dispatch. Holds **all experts'
    /// weights for one matmul role** (e.g. all `ffn_gate_exps` rows for
    /// every expert) in one contiguous Metal buffer with byte stride
    /// `nb02` between expert slabs. Consumed by the
    /// `gemv_q4kw_moe_id_f32` / `gemv_q6kw_moe_id_f32` Metal kernels in
    /// a single dispatch covering all selected (token, expert) pairs.
    Q4KExperts {
        blocks: metal::Buffer,
        byte_offset: u64,
        num_experts: usize,
        n_rows: usize, // per-expert out_features
        n_cols: usize, // in_features
    },
    Q6KExperts {
        blocks: metal::Buffer,
        byte_offset: u64,
        num_experts: usize,
        n_rows: usize,
        n_cols: usize,
    },
}

impl MetalQuantStore {
    fn n_rows(&self) -> usize {
        match self {
            MetalQuantStore::Q4K { n_rows, .. } | MetalQuantStore::Q6K { n_rows, .. } => *n_rows,
            MetalQuantStore::Fused { total_rows, .. } => *total_rows,
            MetalQuantStore::Q4KExperts { n_rows, .. }
            | MetalQuantStore::Q6KExperts { n_rows, .. } => *n_rows,
        }
    }
}

// SAFETY: metal::Buffer wraps an Objective-C handle. metal-rs marks it
// Send+Sync via internal unsafe impls; we just propagate that.
unsafe impl Send for MetalQuantStore {}
unsafe impl Sync for MetalQuantStore {}

/// Resolve `bytes` to a `(MTLBuffer, byte_offset_in_buffer)` pair.
///
/// Two paths:
///   * **Zero-copy**: if `bytes` lies inside a registered mmap region
///     (i.e., the GGUF file is mmap'd and registered via
///     `register_gguf_mmap`), we wrap the page-aligned host range
///     covering this tensor in a fresh `newBufferWithBytesNoCopy` and
///     return the buffer + the tensor's offset within the page-aligned
///     window. Memory cost: nothing — Metal references the same physical
///     pages as the file mmap.
///   * **Copy**: otherwise (slice is heap memory, e.g. a fused
///     byte-concat'd tensor), allocate a fresh shared buffer and copy
///     the bytes in. Memory cost: `bytes.len()` GPU-resident bytes.
///
/// Why per-tensor `newBufferWithBytesNoCopy` instead of one big buffer
/// covering the whole file: a 16-GiB MTLBuffer regresses decode tok/s
/// ~30× on M1 Max — Apple's GPU residency tracking on huge buffers is
/// expensive per dispatch. Many small per-tensor buffers fit Apple's
/// model better (this is the same pattern llama.cpp uses for tensors
/// that fit in one view, just at finer granularity).
fn buffer_for_quant_bytes(bytes: &[u8]) -> (metal::Buffer, u64) {
    const PAGE: usize = 16384;
    let trace = std::env::var("FERRUM_MMAP_TRACE").is_ok();
    if slice_is_in_registered_mmap(bytes) {
        // Zero-copy: page-align the host range covering this tensor and
        // wrap it in an MTLBuffer. The keeper Arc on the registry entry
        // keeps the underlying mmap alive as long as the registry has
        // any entry for it; that outlives any MetalQuantStore we ever
        // hand back, so the buffer's pointer stays valid.
        let ptr_addr = bytes.as_ptr() as usize;
        let aligned_start = ptr_addr & !(PAGE - 1);
        let aligned_end = (ptr_addr + bytes.len()).div_ceil(PAGE) * PAGE;
        let aligned_len = aligned_end - aligned_start;
        let byte_offset = (ptr_addr - aligned_start) as u64;
        let buf = st().pipes.device.new_buffer_with_bytes_no_copy(
            aligned_start as *const c_void,
            aligned_len as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        if buf.length() != 0 {
            if trace {
                eprintln!(
                    "[mmap] zero-copy: tensor ptr=0x{ptr_addr:x} len={} -> buf @0x{aligned_start:x} len={aligned_len} off={byte_offset}",
                    bytes.len()
                );
            }
            return (buf, byte_offset);
        }
        // newBufferWithBytesNoCopy can refuse very rare edge cases
        // (fragmented host pages?). Fall through to the copy path.
        if trace {
            eprintln!(
                "[mmap] zero-copy refused for tensor ptr=0x{ptr_addr:x} len={} aligned_len={aligned_len} — copying",
                bytes.len()
            );
        }
    }
    if trace {
        eprintln!("[mmap] copy: ptr={:p} len={}", bytes.as_ptr(), bytes.len());
    }
    let buf = st().pipes.device.new_buffer_with_data(
        bytes.as_ptr() as *const c_void,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    (buf, 0)
}

// ── MoE expert stack builders ────────────────────────────────────────

/// Build a `Q4KExperts` MoE stack from a contiguous block-bytes payload.
///
/// `bytes` must be exactly `num_experts * n_rows * (n_cols/256) * 144`
/// bytes — typically the raw `ffn_gate_exps` / `ffn_up_exps` data slab
/// straight off the GGUF. When the slice belongs to a registered mmap,
/// the returned store points into the shared zero-copy buffer with a
/// non-zero `byte_offset`; otherwise a fresh allocation is made.
pub fn load_q4k_experts(
    bytes: &[u8],
    num_experts: usize,
    n_rows: usize,
    n_cols: usize,
) -> Result<MetalQuantStore> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144;
    if n_cols % QK_K != 0 {
        return Err(FerrumError::model(format!(
            "load_q4k_experts: n_cols {n_cols} not a multiple of {QK_K}"
        )));
    }
    let expected = num_experts * n_rows * (n_cols / QK_K) * BLOCK_BYTES;
    if bytes.len() != expected {
        return Err(FerrumError::model(format!(
            "load_q4k_experts: bytes {} != expected {expected} ({num_experts}E × {n_rows}R × {n_cols}C)",
            bytes.len()
        )));
    }
    let (blocks, byte_offset) = buffer_for_quant_bytes(bytes);
    Ok(MetalQuantStore::Q4KExperts {
        blocks,
        byte_offset,
        num_experts,
        n_rows,
        n_cols,
    })
}

/// Build a `Q6KExperts` MoE stack from a contiguous block-bytes payload.
/// Honours the mmap registry the same way as [`load_q4k_experts`].
pub fn load_q6k_experts(
    bytes: &[u8],
    num_experts: usize,
    n_rows: usize,
    n_cols: usize,
) -> Result<MetalQuantStore> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = crate::q6_k_gemv::Q6_K_BLOCK_BYTES;
    if n_cols % QK_K != 0 {
        return Err(FerrumError::model(format!(
            "load_q6k_experts: n_cols {n_cols} not a multiple of {QK_K}"
        )));
    }
    let expected = num_experts * n_rows * (n_cols / QK_K) * BLOCK_BYTES;
    if bytes.len() != expected {
        return Err(FerrumError::model(format!(
            "load_q6k_experts: bytes {} != expected {expected}",
            bytes.len()
        )));
    }
    let (blocks, byte_offset) = buffer_for_quant_bytes(bytes);
    Ok(MetalQuantStore::Q6KExperts {
        blocks,
        byte_offset,
        num_experts,
        n_rows,
        n_cols,
    })
}

/// Dispatch the MoE indirect-gemv on an existing compute encoder.
/// `ids` is a Metal buffer of `n_selected` i32 expert IDs (one per
/// selected slot). The kernel writes `[n_selected, n_rows]` outputs.
/// `src1_stride` is the per-slot activation stride in elements: 0 for
/// broadcast (gate/up), `n_cols` for per-slot (down).
pub fn dispatch_gemv_moe_id(
    enc: &metal::ComputeCommandEncoderRef,
    a: &metal::Buffer,
    weights: &MetalQuantStore,
    ids: &metal::Buffer,
    out: &metal::Buffer,
    n_selected: usize,
    src1_stride: usize,
) -> Result<()> {
    match weights {
        MetalQuantStore::Q4KExperts {
            blocks,
            byte_offset,
            n_rows,
            n_cols,
            ..
        } => {
            crate::q4_k_moe_id_gemv::dispatch_gemv_q4k_moe_id_on_encoder(
                &st().pipes.device,
                enc,
                a,
                blocks,
                *byte_offset,
                ids,
                out,
                *n_rows,
                *n_cols,
                n_selected,
                src1_stride,
            );
            Ok(())
        }
        MetalQuantStore::Q6KExperts {
            blocks,
            byte_offset,
            n_rows,
            n_cols,
            ..
        } => {
            crate::q6_k_moe_id_gemv::dispatch_gemv_q6k_moe_id_on_encoder(
                &st().pipes.device,
                enc,
                a,
                blocks,
                *byte_offset,
                ids,
                out,
                *n_rows,
                *n_cols,
                n_selected,
                src1_stride,
            );
            Ok(())
        }
        _ => Err(FerrumError::model(
            "dispatch_gemv_moe_id: weights must be Q4KExperts or Q6KExperts variant".to_string(),
        )),
    }
}

/// Offset-aware variant of [`dispatch_gemv_moe_id`].
///
/// `a_byte_offset` lets the activation pointer skip into a stacked
/// `[M, K]` buffer at row `i*K`; `ids_byte_offset` skips into a stacked
/// `[M, top_k]` selected-experts buffer at the i-th token's block.
///
/// Eliminates the `copy_slice` round-trips in the per-item batched
/// decode path on Qwen3-MoE — saves 2 dispatches per (item × layer)
/// at no GPU compute cost (Metal's `set_buffer(buf, offset)` is free).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemv_moe_id_offset(
    enc: &metal::ComputeCommandEncoderRef,
    a: &metal::Buffer,
    a_byte_offset: u64,
    weights: &MetalQuantStore,
    ids: &metal::Buffer,
    ids_byte_offset: u64,
    out: &metal::Buffer,
    n_selected: usize,
    src1_stride: usize,
) -> Result<()> {
    match weights {
        MetalQuantStore::Q4KExperts {
            blocks,
            byte_offset,
            n_rows,
            n_cols,
            ..
        } => {
            crate::q4_k_moe_id_gemv::dispatch_gemv_q4k_moe_id_offset_on_encoder(
                &st().pipes.device,
                enc,
                a,
                a_byte_offset,
                blocks,
                *byte_offset,
                ids,
                ids_byte_offset,
                out,
                *n_rows,
                *n_cols,
                n_selected,
                src1_stride,
            );
            Ok(())
        }
        MetalQuantStore::Q6KExperts {
            blocks,
            byte_offset,
            n_rows,
            n_cols,
            ..
        } => {
            crate::q6_k_moe_id_gemv::dispatch_gemv_q6k_moe_id_offset_on_encoder(
                &st().pipes.device,
                enc,
                a,
                a_byte_offset,
                blocks,
                *byte_offset,
                ids,
                ids_byte_offset,
                out,
                *n_rows,
                *n_cols,
                n_selected,
                src1_stride,
            );
            Ok(())
        }
        _ => Err(FerrumError::model(
            "dispatch_gemv_moe_id_offset: weights must be Q4KExperts or Q6KExperts variant"
                .to_string(),
        )),
    }
}

// ── Fused-store leaf dispatchers (used by metal_gemm_quant_dispatch) ─

/// Dispatch a single mul_mm for one Q4K / Q6K leaf part with output
/// column offset + row stride. Used by the Fused m>1 path.
fn dispatch_part_gemm(
    enc: &metal::ComputeCommandEncoderRef,
    a_buf: &metal::Buffer,
    part: &MetalQuantStore,
    out_buf: &metal::Buffer,
    c_offset_cols: usize,
    m: usize,
    part_rows: usize,
    stride_c: usize,
    n_cols: usize,
) -> Result<()> {
    if part_rows % 4 != 0 {
        return Err(FerrumError::model(format!(
            "gemm_quant Fused: part rows {part_rows} not divisible by 4"
        )));
    }
    match part {
        MetalQuantStore::Q4K {
            blocks,
            byte_offset,
            ..
        } => {
            crate::q4_k_gemm::dispatch_gemm_q4k_part(
                &st().pipes.device,
                enc,
                a_buf,
                blocks,
                *byte_offset,
                out_buf,
                c_offset_cols,
                m,
                part_rows,
                stride_c,
                n_cols,
            );
        }
        MetalQuantStore::Q6K {
            blocks,
            byte_offset,
            ..
        } => {
            crate::q6_k_gemm::dispatch_gemm_q6k_part(
                &st().pipes.device,
                enc,
                a_buf,
                blocks,
                *byte_offset,
                out_buf,
                c_offset_cols,
                m,
                part_rows,
                stride_c,
                n_cols,
            );
        }
        MetalQuantStore::Fused { .. }
        | MetalQuantStore::Q4KExperts { .. }
        | MetalQuantStore::Q6KExperts { .. } => {
            return Err(FerrumError::model(
                "gemm_quant Fused: only Q4K/Q6K leaf parts supported here".to_string(),
            ));
        }
    }
    Ok(())
}

/// Dispatch a single fused gemv for one Q4K / Q6K leaf part with
/// activation and output byte offsets. Used by the Fused m=1 path.
fn dispatch_part_gemv_offset(
    enc: &metal::ComputeCommandEncoderRef,
    a_buf: &metal::Buffer,
    a_offset_bytes: u64,
    part: &MetalQuantStore,
    out_buf: &metal::Buffer,
    c_offset_bytes: u64,
    n_cols: usize,
) -> Result<()> {
    match part {
        MetalQuantStore::Q4K {
            blocks,
            byte_offset,
            n_rows,
            ..
        } => {
            if *n_rows % 4 != 0 {
                crate::q4_k_gemv::dispatch_gemv_q4k_on_encoder(
                    &st().pipes.device,
                    enc,
                    a_buf,
                    blocks,
                    *byte_offset,
                    out_buf,
                    *n_rows,
                    n_cols,
                );
                if a_offset_bytes != 0 || c_offset_bytes != 0 {
                    return Err(FerrumError::model(
                        "gemm_quant Fused: q4k v1 path doesn't support offsets yet".to_string(),
                    ));
                }
                return Ok(());
            }
            crate::q4_k_gemv_v2::dispatch_gemv_q4k_v2_offset(
                &st().pipes.device,
                enc,
                a_buf,
                a_offset_bytes,
                blocks,
                *byte_offset,
                out_buf,
                c_offset_bytes,
                *n_rows,
                n_cols,
            );
        }
        MetalQuantStore::Q6K {
            blocks,
            byte_offset,
            n_rows,
            ..
        } => {
            if *n_rows % 4 != 0 {
                return Err(FerrumError::model(format!(
                    "gemm_quant Fused: Q6K part n_rows={n_rows} not divisible by 4"
                )));
            }
            crate::q6_k_gemv::dispatch_gemv_q6k_v2_offset(
                &st().pipes.device,
                enc,
                a_buf,
                a_offset_bytes,
                blocks,
                *byte_offset,
                out_buf,
                c_offset_bytes,
                *n_rows,
                n_cols,
            );
        }
        MetalQuantStore::Fused { .. }
        | MetalQuantStore::Q4KExperts { .. }
        | MetalQuantStore::Q6KExperts { .. } => {
            return Err(FerrumError::model(
                "gemm_quant Fused: only Q4K/Q6K leaf parts supported here".to_string(),
            ));
        }
    }
    Ok(())
}

// ── BackendQuantGguf ──────────────────────────────────────────────────

/// Internal helper: build a `MetalQuantStore` from raw GGUF block bytes.
/// Phase 3e/3 extracted from the old `load_quant` body so `load_quant_fused`
/// can compose multiple stores into a `Fused` variant without going
/// through the `Box<dyn Linear>` wrapper.
fn metal_load_quant_store_helper(
    kind: GgufQuantType,
    bytes: &[u8],
    n_rows: usize,
    n_cols: usize,
) -> Result<MetalQuantStore> {
    const QK_K: usize = 256;
    match kind {
        GgufQuantType::Q4K => {
            const BLOCK_BYTES: usize = 144;
            let total_elems = n_rows * n_cols;
            if total_elems % QK_K != 0 {
                return Err(FerrumError::model(format!(
                    "load_quant Q4K: elements {total_elems} not multiple of {QK_K}"
                )));
            }
            let n_blocks = total_elems / QK_K;
            let expected = n_blocks * BLOCK_BYTES;
            if bytes.len() != expected {
                return Err(FerrumError::model(format!(
                    "load_quant Q4K: bytes {} != expected {} ({n_blocks} blocks)",
                    bytes.len(),
                    expected
                )));
            }
            let (blocks, byte_offset) = buffer_for_quant_bytes(bytes);
            Ok(MetalQuantStore::Q4K {
                blocks,
                byte_offset,
                n_rows,
                n_cols,
                n_blocks,
            })
        }
        GgufQuantType::Q6K => {
            const BLOCK_BYTES: usize = crate::q6_k_gemv::Q6_K_BLOCK_BYTES; // 210
            let total_elems = n_rows * n_cols;
            if total_elems % QK_K != 0 {
                return Err(FerrumError::model(format!(
                    "load_quant Q6K: elements {total_elems} not multiple of {QK_K}"
                )));
            }
            let n_blocks = total_elems / QK_K;
            let expected = n_blocks * BLOCK_BYTES;
            if bytes.len() != expected {
                return Err(FerrumError::model(format!(
                    "load_quant Q6K: bytes {} != expected {} ({n_blocks} blocks)",
                    bytes.len(),
                    expected
                )));
            }
            let (blocks, byte_offset) = buffer_for_quant_bytes(bytes);
            Ok(MetalQuantStore::Q6K {
                blocks,
                byte_offset,
                n_rows,
                n_cols,
                n_blocks,
            })
        }
        other => Err(FerrumError::unsupported(format!(
            "Metal load_quant: {other:?} not yet implemented"
        ))),
    }
}

impl crate::backend::BackendQuantGguf for MetalBackend {
    fn load_quant(
        kind: GgufQuantType,
        bytes: &[u8],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        let store = metal_load_quant_store_helper(kind, bytes, n_rows, n_cols)?;
        // Phase 3e/3: dispatch via MetalGgufLinear::forward.
        Ok(Box::new(crate::quant_linear::metal_gguf::MetalGgufLinear {
            store,
            in_features: n_cols,
            out_features: n_rows,
        }))
    }
    fn load_quant_experts(
        kind: GgufQuantType,
        bytes: &[u8],
        num_experts: usize,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn crate::StackedExpertGgufLinear<Self>>> {
        let store = match kind {
            GgufQuantType::Q4K => load_q4k_experts(bytes, num_experts, n_rows, n_cols)?,
            GgufQuantType::Q6K => load_q6k_experts(bytes, num_experts, n_rows, n_cols)?,
            other => {
                return Err(FerrumError::unsupported(format!(
                    "Metal load_quant_experts: {other:?} not implemented (only Q4K / Q6K)"
                )));
            }
        };
        Ok(Box::new(
            crate::quant_linear::metal_gguf_moe::MetalStackedExpertGgufLinear::new(store)?,
        ))
    }
    fn load_quant_fused(
        parts: &[(GgufQuantType, &[u8], usize)],
        n_cols: usize,
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        let mut sub_stores: Vec<MetalQuantStore> = Vec::with_capacity(parts.len());
        let mut total_rows = 0;
        for (kind, bytes, n_rows) in parts {
            // Phase 3e/3: use the helper to get the raw QuantStore (the
            // trait method now returns Box<dyn Linear>, which we can't
            // unwrap into Fused).
            let store = metal_load_quant_store_helper(*kind, bytes, *n_rows, n_cols)?;
            // Only leaf (Q4K / Q6K) parts are valid; nested Fused isn't.
            if matches!(store, MetalQuantStore::Fused { .. }) {
                return Err(FerrumError::model(
                    "Metal load_quant_fused: nested Fused not supported".to_string(),
                ));
            }
            total_rows += n_rows;
            sub_stores.push(store);
        }
        let fused = MetalQuantStore::Fused {
            parts: sub_stores,
            total_rows,
            n_cols,
        };
        Ok(Box::new(crate::quant_linear::metal_gguf::MetalGgufLinear {
            store: fused,
            in_features: n_cols,
            out_features: total_rows,
        }))
    }
}

// ── Free-fn matmul dispatcher (called by MetalGgufLinear::forward) ───

/// Metal GGUF k-quant matmul dispatcher — extracted from the old
/// `BackendQuantGguf::gemm_quant` impl in Phase 3e/3 so the new
/// `MetalGgufLinear::forward` (in `quant_linear::metal_gguf`) can call
/// it from outside the trait. Trait method `gemm_quant` itself is
/// gone; this free fn is the sole entry point now.
pub fn metal_gemm_quant_dispatch(
    ctx: &mut <MetalBackend as Backend>::Context,
    a: &<MetalBackend as Backend>::Buffer,
    weight: &MetalQuantStore,
    out: &mut <MetalBackend as Backend>::Buffer,
    m: usize,
) -> Result<()> {
    // Fused path: row-concatenation of mixed-quant parts (used for
    // Qwen3 qkv_proj where q+k are Q4_K but v is Q6_K). For m=1
    // dispatch each part with the right output offset; for m>1
    // currently bail (prefill of mixed-fused matmuls is rare and
    // would need a strided per-row inner loop).
    if let MetalQuantStore::Fused {
        parts,
        total_rows,
        n_cols,
    } = weight
    {
        if m != 1 {
            // **Fused mul_mm path** for prefill. Each part dispatches
            // one mul_mm into a [m, part_rows] slice of the
            // [m, total_rows] fused output via the strided variant
            // (kernel uses `strideC = total_rows`, write width = part_rows,
            // dst pre-offset to part column). Replaces the previous
            // m × per-part gemv loop which scaled linearly with prompt
            // length and was the dominant remaining prefill bottleneck
            // on Qwen3 (mixed Q4+Q6 qkv).
            let a_buf = a.expect_f32("gemm_quant a (fused)");
            let out_buf = out.expect_f32_mut("gemm_quant out (fused)");
            let enc = ctx.compute_encoder();
            let mut col_off = 0usize;
            for part in parts {
                let part_rows = part.n_rows();
                dispatch_part_gemm(
                    enc,
                    a_buf,
                    part,
                    out_buf,
                    col_off,
                    m,
                    part_rows,
                    *total_rows,
                    *n_cols,
                )?;
                col_off += part_rows;
            }
            return Ok(());
        }
        // m == 1 — single output row of length total_rows.
        let a_buf = a.expect_f32("gemm_quant a (fused m=1)");
        let out_buf = out.expect_f32_mut("gemm_quant out (fused m=1)");
        let enc = ctx.compute_encoder();
        let mut row_off_elems = 0usize;
        for part in parts {
            let part_rows = part.n_rows();
            let c_off = (row_off_elems * 4) as u64;
            dispatch_part_gemv_offset(enc, a_buf, 0, part, out_buf, c_off, *n_cols)?;
            row_off_elems += part_rows;
        }
        return Ok(());
    }

    // Borrow checker: `cmd` / `compute_encoder` need `&mut ctx` while
    // we hold a borrow into `weight` for `blocks`. Snapshot the
    // primitive fields out of `weight` first; `blocks` is a `Buffer`
    // ref that is independent of ctx.
    let (blocks, blocks_off, n_rows, n_cols, n_blocks, is_q6k) = match weight {
        MetalQuantStore::Q4K {
            blocks,
            byte_offset,
            n_rows,
            n_cols,
            n_blocks,
        } => (blocks, *byte_offset, *n_rows, *n_cols, *n_blocks, false),
        MetalQuantStore::Q6K {
            blocks,
            byte_offset,
            n_rows,
            n_cols,
            n_blocks,
        } => (blocks, *byte_offset, *n_rows, *n_cols, *n_blocks, true),
        MetalQuantStore::Fused { .. } => unreachable!("handled above"),
        MetalQuantStore::Q4KExperts { .. } | MetalQuantStore::Q6KExperts { .. } => {
            return Err(FerrumError::model(
                "gemm_quant: ExpertsStacked must be dispatched via gemv_moe_id".to_string(),
            ));
        }
    };

    let _t0 = if debug_per_call_flush() {
        Some(std::time::Instant::now())
    } else {
        None
    };

    let a_buf = a.expect_f32("gemm_quant a");
    let out_buf = out.expect_f32_mut("gemm_quant out");

    if m == 1 {
        // **Fused path** for decode (m=1): one kernel reads the
        // Q4_K / Q6_K super-blocks, decodes them inline, and reduces
        // against `A`. No transient fp16 buffer materialised.
        //
        // All variants use N_R0=2, N_SG=2 layout from llama.cpp:
        // 64 threads/threadgroup, 4 output rows/threadgroup. Requires
        // N divisible by 4. Q6_K has no v1 fallback yet — Qwen3 /
        // Llama always satisfy the constraint, so we just panic if
        // not.
        let enc = ctx.compute_encoder();
        if is_q6k {
            if n_rows % 4 != 0 {
                return Err(FerrumError::model(format!(
                    "gemm_quant Q6K: n_rows={n_rows} not divisible by 4"
                )));
            }
            crate::q6_k_gemv::dispatch_gemv_q6k_v2_on_encoder(
                &st().pipes.device,
                enc,
                a_buf,
                blocks,
                blocks_off,
                out_buf,
                n_rows,
                n_cols,
            );
        } else if n_rows % 4 == 0 {
            crate::q4_k_gemv_v2::dispatch_gemv_q4k_v2_on_encoder(
                &st().pipes.device,
                enc,
                a_buf,
                blocks,
                blocks_off,
                out_buf,
                n_rows,
                n_cols,
            );
        } else {
            crate::q4_k_gemv::dispatch_gemv_q4k_on_encoder(
                &st().pipes.device,
                enc,
                a_buf,
                blocks,
                blocks_off,
                out_buf,
                n_rows,
                n_cols,
            );
        }
    } else if is_q6k {
        // **Q6_K m>1 fused mul_mm path** — ported from llama.cpp's
        // `kernel_mul_mm_q6_K_f32`. Same 64×32 simdgroup-matmul
        // tile + inline dequant as the Q4_K version. Replaces the
        // prior per-row gemv loop which scaled linearly with m
        // and was the dominant prefill bottleneck on Q4_K_M models
        // where down_proj / lm_head live as Q6_K.
        let enc = ctx.compute_encoder();
        crate::q6_k_gemm::dispatch_gemm_q6k_on_encoder(
            &st().pipes.device,
            enc,
            a_buf,
            blocks,
            blocks_off,
            out_buf,
            m,
            n_rows,
            n_cols,
        );
    } else {
        // **Fused mul_mm path** for prefill (m > 1) Q4_K — ported
        // from llama.cpp's `kernel_mul_mm_q4_K_f32`. Inlines Q4_K
        // dequant into the threadgroup-memory load and uses
        // `simdgroup_half8x8` matmul, eliminating both the fp16
        // transient buffer (~2× memory traffic) and the scalar
        // gemm_v2_f16w inner loop.
        let enc = ctx.compute_encoder();
        crate::q4_k_gemm::dispatch_gemm_q4k_on_encoder(
            &st().pipes.device,
            enc,
            a_buf,
            blocks,
            blocks_off,
            out_buf,
            m,
            n_rows,
            n_cols,
        );
        let _ = n_blocks; // not consumed by mul_mm path; kept for the load-time block accounting
    }

    // Optional per-call timing: commit + wait the cmd buffer right
    // here so we measure the GPU work for *this* matmul. Off by
    // default (would serialize the whole pipeline).
    if let Some(t0) = _t0 {
        ctx.flush();
        let elapsed_us = t0.elapsed().as_micros();
        QUANT_GEMM_TIME_US.fetch_add(elapsed_us as u64, std::sync::atomic::Ordering::Relaxed);
        QUANT_GEMM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        QUANT_GEMM_LAST_M.store(m as u64, std::sync::atomic::Ordering::Relaxed);
        QUANT_GEMM_LAST_N.store(n_rows as u64, std::sync::atomic::Ordering::Relaxed);
        QUANT_GEMM_LAST_K.store(n_cols as u64, std::sync::atomic::Ordering::Relaxed);
        if QUANT_GEMM_CALLS.load(std::sync::atomic::Ordering::Relaxed) <= 16 {
            eprintln!(
                "[gemm_quant] m={} n={} k={} took {} us",
                m, n_rows, n_cols, elapsed_us
            );
        }
    }
    Ok(())
}
