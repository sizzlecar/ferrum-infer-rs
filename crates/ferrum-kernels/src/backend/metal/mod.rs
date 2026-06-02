//! Metal backend — one unified backend with dtype-tagged buffers.
//!
//! Precision is carried as a runtime tag on [`MetalBuf`] rather than a
//! separate backend type. Ops branch on buffer dtype where it matters
//! (`gemm`, `embedding_lookup`); the rest assert F32 and go through the
//! existing f32 shaders.
//!
//! The default policy:
//!   - Activations (`alloc`, `from_slice`, scratch) are F32 — precision-
//!     sensitive paths (norm / rope / attention) stay unchanged.
//!   - Weights loaded via `from_weight_bytes` go F16 for tensors past a
//!     size threshold when `FERRUM_METAL_DTYPE=f16` is set; otherwise
//!     they stay F32.
//!
//! The same shape generalises: an INT8 / GPTQ variant would add another
//! `Dtype` case + the corresponding shader, without introducing a new
//! `Backend` type. Hardware (Metal / CUDA / CPU) and precision remain
//! orthogonal, matching how the `Backend::GptqStore` + `gemm_gptq`
//! trait plumbing already handles INT4 on CUDA.
//!
//! Command buffer lifecycle mirrors llama.cpp: one retained buffer per
//! forward pass that accumulates encodes; `sync()` commits and waits.
//!
//! ## Layout (Audit #8 split)
//!
//! - `mod.rs` (this file) — shared state, `MetalBuf`/`MetalContext`,
//!   `impl Backend for MetalBackend`, empty stubs for `BackendGraph`,
//!   `BackendCollective`, `BackendQuantMarlin`, and `BackendKvDtype<KvFp16>`.
//! - [`quant`] — `MetalQuantStore` + `BackendQuantGguf` + GGUF k-quant
//!   matmul dispatcher + MoE expert-stack builders.
//! - [`paged`] — `BackendPagedKv` (paged-KV decode + split_qkv into pool).
//! - [`moe`]   — `BackendMoeFused` (route/topk, silu/mul, weighted-sum).

pub mod moe;
pub mod paged;
pub mod quant;

// GGUF k-quant kernels (Audit #9 moved here from kernels-crate top level).
// Re-exported via `pub use backend::metal::{...}` in `crate::lib` so the
// historical `ferrum_kernels::q4_k_gemm::*` and internal `crate::q4_k_*`
// paths keep working without rippling import updates through the
// metal_gguf_moe.rs et al. consumers.
pub mod moe_post_ops;
pub mod moe_post_ops_batched;
pub mod moe_router;
pub mod q4_k;
pub mod q4_k_gemm;
pub mod q4_k_gemv;
pub mod q4_k_gemv_v2;
pub mod q4_k_moe_id_gate_up_silu;
pub mod q4_k_moe_id_gate_up_silu_batched;
pub mod q4_k_moe_id_gemm;
pub mod q4_k_moe_id_gemv;
pub mod q4_k_moe_id_gemv_batched;
pub mod q6_k_gemm;
pub mod q6_k_gemv;
pub mod q6_k_moe_id_gemm;
pub mod q6_k_moe_id_gemv;
pub mod q6_k_moe_id_gemv_batched;

// Preserve the historical `crate::backend::metal::*` public surface that
// external callers (`quant_linear::metal_gguf*`, model tests,
// `ferrum-engine::registry`) depend on.
pub use quant::{
    dispatch_gemv_moe_id, dispatch_gemv_moe_id_offset, load_q4k_experts, load_q6k_experts,
    metal_gemm_quant_dispatch, MetalQuantStore,
};

use super::{AttnConfig, Backend, SrcDtype};
use crate::attention::metal::pipelines::MetalPipelines;
use crate::attention::AttentionParams;
use ferrum_types::{FerrumError, Result};
use half::{bf16, f16};
use metal::{Device, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

// ── Shared Metal state ────────────────────────────────────────────────

/// One registered host-memory region eligible for zero-copy Metal binding.
///
/// `base_addr / len` describe the host range (`as_ptr() as usize` for Send/
/// Sync). `_keeper` holds an Arc to whatever owns the host memory
/// (typically `Arc<GgufFile>`) so the mmap stays alive as long as any
/// Metal buffer wraps a part of it.
///
/// Unlike the early "one big buffer" attempt, we do **not** create a
/// single MTLBuffer for the whole region here — that approach worked but
/// regressed decode tok/s ~30× on M1 Max because Apple's GPU residency
/// logic on large buffers (16 GiB) is very expensive per dispatch.
/// Instead, `quant::buffer_for_quant_bytes` creates a small per-tensor
/// MTLBuffer via `newBufferWithBytesNoCopy` covering only the
/// page-aligned region around each tensor — same memory footprint, but
/// many small buffers fit Apple's GPU residency model the way llama.cpp
/// observed.
struct MetalMmapEntry {
    base_addr: usize,
    len: usize,
    _keeper: Arc<dyn std::any::Any + Send + Sync>,
}

pub(crate) struct MetalState {
    pub(crate) pipes: MetalPipelines,
    /// Registered mmap regions wrapped as zero-copy Metal buffers. Looked
    /// up at `load_quant*` time so weight tensors that already live in a
    /// registered mmap can reuse the big buffer with an offset instead of
    /// being copied (`new_buffer_with_data`) into a fresh allocation.
    mmaps: Mutex<Vec<MetalMmapEntry>>,
}
static METAL_STATE: OnceLock<MetalState> = OnceLock::new();
pub(crate) fn st() -> &'static MetalState {
    METAL_STATE.get_or_init(|| MetalState {
        pipes: MetalPipelines::new(&Device::system_default().unwrap()),
        mmaps: Mutex::new(Vec::new()),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MetalRuntimeConfig {
    mmap_trace: bool,
    capture_path: Option<String>,
    mtl_capture_enabled: bool,
    prefer_f16_weights: bool,
}

fn metal_runtime_config() -> &'static MetalRuntimeConfig {
    static CONFIG: OnceLock<MetalRuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let mut config = MetalRuntimeConfig {
            mmap_trace: false,
            capture_path: None,
            mtl_capture_enabled: false,
            prefer_f16_weights: false,
        };
        for (name, value) in std::env::vars() {
            match name.as_str() {
                "FERRUM_MMAP_TRACE" => config.mmap_trace = true,
                "FERRUM_METAL_CAPTURE" => config.capture_path = Some(value),
                "FERRUM_METAL_DTYPE" => {
                    config.prefer_f16_weights = value.eq_ignore_ascii_case("f16")
                }
                "MTL_CAPTURE_ENABLED" => config.mtl_capture_enabled = true,
                _ => {}
            }
        }
        config
    })
}

pub(crate) fn metal_mmap_trace_enabled() -> bool {
    metal_runtime_config().mmap_trace
}

/// Register a host-memory region (typically the full mmap of a GGUF file)
/// so subsequent `load_quant*` calls whose input slice lives inside this
/// range can use the shared zero-copy `MTLBuffer` instead of allocating a
/// fresh device-resident copy.
///
/// `keeper` is anything that, while alive, guarantees `slice` stays mapped.
/// For GGUF the natural choice is `Arc<GgufFile>`. The Metal state holds
/// onto it for the lifetime of the registration entry, which is the
/// process lifetime — registrations are not removed today.
///
/// Constraints (`newBufferWithBytesNoCopy`):
///   * the slice base pointer must be page-aligned (16 KB on Apple Silicon)
///   * the wrapped length must be a multiple of the page size
///
/// `mmap` returns a page-aligned base, so the address constraint is met
/// for free. For length we round **up** to the next page; the kernel
/// zero-fills any tail past EOF, but our reads never go past the file
/// length so that's harmless.
///
/// Returns `Ok(())` on success; on alignment failure or duplicate
/// registration, returns an error and does not mutate the registry.
pub fn register_gguf_mmap(
    slice: &[u8],
    keeper: Arc<dyn std::any::Any + Send + Sync>,
) -> Result<()> {
    const PAGE: usize = 16384;
    let base_addr = slice.as_ptr() as usize;
    if !base_addr.is_multiple_of(PAGE) {
        return Err(FerrumError::model(format!(
            "register_gguf_mmap: base pointer 0x{base_addr:x} not page-aligned (need {PAGE})"
        )));
    }
    let trace = metal_mmap_trace_enabled();
    if trace {
        eprintln!(
            "[mmap] register file at 0x{base_addr:x} len={} ({:.2} GB)",
            slice.len(),
            slice.len() as f64 / 1e9
        );
    }
    let mut guard = st()
        .mmaps
        .lock()
        .map_err(|e| FerrumError::model(format!("register_gguf_mmap: registry poisoned: {e}")))?;
    if guard
        .iter()
        .any(|e| e.base_addr == base_addr && e.len == slice.len())
    {
        return Ok(());
    }
    guard.push(MetalMmapEntry {
        base_addr,
        len: slice.len(),
        _keeper: keeper,
    });
    Ok(())
}

/// Check whether `bytes` lives inside a registered mmap region. Returns
/// the (registered_base, registered_len) for the matching entry — the
/// caller uses this only to know "yes, the host memory will outlive any
/// MTLBuffer we wrap around it" via the entry's keeper Arc.
#[inline(never)]
pub(crate) fn slice_is_in_registered_mmap(bytes: &[u8]) -> bool {
    let ptr = bytes.as_ptr() as usize;
    let len = bytes.len();
    let end = match ptr.checked_add(len) {
        Some(e) => e,
        None => return false,
    };
    let guard = match st().mmaps.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    for entry in guard.iter() {
        let entry_end = match entry.base_addr.checked_add(entry.len) {
            Some(e) => e,
            None => continue,
        };
        if ptr >= entry.base_addr && end <= entry_end {
            return true;
        }
    }
    false
}

// ── Frame capture ─────────────────────────────────────────────────────

/// Begin a Metal frame capture if `FERRUM_METAL_CAPTURE` is set to an
/// output path. The result is a `.gputrace` file you can open in Xcode
/// to view per-kernel GPU timing, occupancy, instruction counts, etc.
///
/// Requirements:
///   - The process must have been launched with `MTL_CAPTURE_ENABLED=1`
///     in its environment (Metal silently rejects capture otherwise).
///   - The output path must not exist already.
///
/// Returns `true` if a capture started, `false` if no env var set or
/// capture failed (in which case stderr explains).
pub fn maybe_begin_frame_capture() -> bool {
    use metal::{CaptureDescriptor, CaptureManager, MTLCaptureDestination};
    let Some(out_path) = metal_runtime_config().capture_path.as_deref() else {
        return false;
    };
    if !metal_runtime_config().mtl_capture_enabled {
        eprintln!(
            "[capture] FERRUM_METAL_CAPTURE set but MTL_CAPTURE_ENABLED is not — Metal will reject. Re-launch with MTL_CAPTURE_ENABLED=1."
        );
        return false;
    }
    let mgr = CaptureManager::shared();
    if !mgr.supports_destination(MTLCaptureDestination::GpuTraceDocument) {
        eprintln!("[capture] device does not support GpuTraceDocument");
        return false;
    }
    let desc = CaptureDescriptor::new();
    desc.set_capture_device(&st().pipes.device);
    desc.set_destination(MTLCaptureDestination::GpuTraceDocument);
    desc.set_output_url(&out_path);
    match mgr.start_capture(&desc) {
        Ok(()) => {
            eprintln!("[capture] started → {out_path}");
            true
        }
        Err(e) => {
            eprintln!("[capture] start_capture failed: {e}");
            false
        }
    }
}

/// Stop the active frame capture and flush the `.gputrace` to disk.
pub fn end_frame_capture() {
    metal::CaptureManager::shared().stop_capture();
    eprintln!("[capture] stopped — open .gputrace in Xcode");
}

// ── Dtype tag + tagged buffer ─────────────────────────────────────────

/// Element storage type for a [`MetalBuf`]. Same shape generalises to INT8
/// / bf16 etc. when their shaders land — just add a variant + wire it in
/// the op dispatches.
// Phase A: Dtype moved to `crate::backend::dtype::Dtype` (shared across
// CPU/CUDA/Metal). Re-export here so existing `metal::Dtype` users keep
// working without touching them.
pub use super::dtype::Dtype;

/// Metal device buffer with a runtime dtype tag and logical element count.
///
/// Two buffers of identical raw bytes but different `dtype` are treated as
/// different types for shader selection. `n` is the number of logical
/// elements; `raw.length()` is `n * dtype.bytes_per_elem()`.
pub struct MetalBuf {
    pub(crate) raw: metal::Buffer,
    pub(crate) dtype: Dtype,
    pub(crate) n: usize,
}

impl MetalBuf {
    pub fn raw(&self) -> &metal::Buffer {
        &self.raw
    }
    pub fn raw_mut(&mut self) -> &mut metal::Buffer {
        &mut self.raw
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn is_f16(&self) -> bool {
        matches!(self.dtype, Dtype::F16)
    }

    #[inline]
    pub(crate) fn expect_f32<'a>(&'a self, what: &str) -> &'a metal::Buffer {
        debug_assert!(
            matches!(self.dtype, Dtype::F32),
            "{what}: expected F32 buffer, got {:?}",
            self.dtype
        );
        &self.raw
    }
    #[inline]
    pub(crate) fn expect_f32_mut<'a>(&'a mut self, what: &str) -> &'a mut metal::Buffer {
        debug_assert!(
            matches!(self.dtype, Dtype::F32),
            "{what}: expected F32 buffer, got {:?}",
            self.dtype
        );
        &mut self.raw
    }
}

// Safety: metal::Buffer wraps a retained NSObject pointer that's safe to move
// and share across threads (Metal serialises on its command queue).
unsafe impl Send for MetalBuf {}
unsafe impl Sync for MetalBuf {}

// ── Context ───────────────────────────────────────────────────────────

/// Metal context — one in-flight command buffer that accumulates encodes
/// across multiple `Backend` method calls. `sync()` commits + waits and
/// drops the handle so the next call creates a fresh buffer.
pub struct MetalContext {
    cmd: Option<&'static metal::CommandBufferRef>,
    /// Sticky compute encoder. Held open across multiple `Backend`
    /// method calls so consecutive compute dispatches share one encoder
    /// — Metal serializes dispatches within a single encoder and prior
    /// dispatch's device-memory writes are visible to the next
    /// dispatch's reads, so this is correct for the model's dataflow.
    /// Closing happens on `sync`, on `compute_encoder_end` (e.g. before
    /// a blit encoder), or naturally on context drop.
    encoder: Option<&'static metal::ComputeCommandEncoderRef>,
}

impl MetalContext {
    /// Return the current in-flight command buffer, creating one on first
    /// use. The `'static` is safe: the command queue lives in a `OnceLock`
    /// for the program's lifetime, and autoreleased command buffers are
    /// retained for as long as that queue lives.
    fn cmd(&mut self) -> &'static metal::CommandBufferRef {
        match self.cmd {
            Some(c) => c,
            None => {
                // Tried `new_command_buffer_with_unretained_references`
                // here (matching llama.cpp): output regressed to "The The
                // The…" — likely some buffer we bind isn't kept alive
                // long enough between encode and execute on this code
                // path. The retained variant is correct and the CPU
                // overhead from retains turned out to be negligible at
                // our dispatch rate.
                let c = st().pipes.queue.new_command_buffer();
                let c_static: &'static metal::CommandBufferRef =
                    unsafe { std::mem::transmute::<&metal::CommandBufferRef, _>(c) };
                self.cmd = Some(c_static);
                c_static
            }
        }
    }

    /// Return a compute command encoder, opening one on the current cmd
    /// buffer if there isn't already one. Subsequent calls during the
    /// same window return the same encoder — set the pipeline state
    /// before each dispatch.
    pub(crate) fn compute_encoder(&mut self) -> &'static metal::ComputeCommandEncoderRef {
        if let Some(enc) = self.encoder {
            return enc;
        }
        let cmd = self.cmd();
        let enc = cmd.new_compute_command_encoder();
        let enc_static: &'static metal::ComputeCommandEncoderRef =
            unsafe { std::mem::transmute::<&metal::ComputeCommandEncoderRef, _>(enc) };
        self.encoder = Some(enc_static);
        enc_static
    }

    /// Close the active compute encoder if one is open. Caller must do
    /// this before switching to a blit encoder, before commit, or before
    /// recording into a different cmd buffer.
    fn compute_encoder_end(&mut self) {
        if let Some(enc) = self.encoder.take() {
            enc.end_encoding();
        }
    }

    pub(crate) fn flush(&mut self) {
        self.compute_encoder_end();
        if let Some(cmd) = self.cmd.take() {
            cmd.commit();
            cmd.wait_until_completed();
        }
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        self.flush();
    }
}

// ── Policy: should big weights land as f16? ───────────────────────────
// Cached on first read so each tensor load doesn't re-parse env.

fn prefer_f16_weights() -> bool {
    metal_runtime_config().prefer_f16_weights
}

/// Element-count threshold above which a weight tensor goes to F16 storage
/// (when `FERRUM_METAL_DTYPE=f16`). Tiny tensors (norm weights, biases) stay
/// F32 so the existing f32 shaders continue to see f32 inputs.
///
/// 1M elements = 4 MB as f32 / 2 MB as f16. Anything smaller saves little
/// memory and would force the f32-only shaders to sprout f16 variants.
const F16_MIN_ELEMS: usize = 1_048_576;

// ── Buffer allocation helpers (context-free) ─────────────────────────

fn alloc_f32_raw(n: usize) -> metal::Buffer {
    st().pipes.buffer_empty(n)
}

fn buffer_from_f32_slice(data: &[f32]) -> metal::Buffer {
    st().pipes.buffer_from_data(data)
}

fn buffer_from_f16_bytes(bytes: &[u8]) -> metal::Buffer {
    debug_assert_eq!(bytes.len() % 2, 0);
    st().pipes.device.new_buffer_with_data(
        bytes.as_ptr() as *const c_void,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Pack a slice of f32 into a fresh f16 metal::Buffer.
fn buffer_f16_from_f32(data: &[f32]) -> metal::Buffer {
    let n = data.len();
    let mut f16_bytes = vec![0u8; n * 2];
    for i in 0..n {
        let h = f16::from_f32(data[i]).to_le_bytes();
        f16_bytes[i * 2] = h[0];
        f16_bytes[i * 2 + 1] = h[1];
    }
    buffer_from_f16_bytes(&f16_bytes)
}

// ── Backend impl ──────────────────────────────────────────────────────

pub struct MetalBackend;

impl Backend for MetalBackend {
    type Buffer = MetalBuf;
    type Context = MetalContext;
    type Timer = crate::backend::timer::MetalTimer;
    fn make_timer() -> Self::Timer {
        crate::backend::timer::MetalTimer::new()
    }
    // type GptqStore: removed in Phase C step 4e. Metal has no Marlin
    // GPTQ path (GGUF is the quant story on Metal). Adding GPTQ later
    // means impl-ing MarlinExpertStack<MetalBackend> + load_gptq_stacked.

    fn new_context() -> Self::Context {
        MetalContext {
            cmd: None,
            encoder: None,
        }
    }
    fn sync(ctx: &mut Self::Context) {
        ctx.flush();
    }

    // ── Q4_K_M ────────────────────────────────────────────────────────

    /// Phase D step 2+3: unified typed uploader. Replaces from_slice_i32 +
    /// the legacy `from_slice` (which is kept as f32-default convenience).
    fn from_slice_typed<T: crate::backend::HostDtype>(data: &[T]) -> Self::Buffer {
        let bytes = data.len() * std::mem::size_of::<T>();
        let raw = st().pipes.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        MetalBuf {
            raw,
            dtype: T::DTYPE,
            n: data.len(),
        }
    }

    /// Phase D step 2+3: unified typed in-place write. Replaces
    /// write_i32_into + write_f32_into. Unified memory on Apple
    /// Silicon means CPU writes the buffer's contents pointer
    /// directly — no blit encoder.
    fn write_typed<T: crate::backend::HostDtype>(
        _ctx: &mut Self::Context,
        buf: &mut Self::Buffer,
        data: &[T],
    ) {
        debug_assert_eq!(
            buf.dtype,
            T::DTYPE,
            "Metal write_typed: buf.dtype {:?} != T::DTYPE {:?}",
            buf.dtype,
            T::DTYPE
        );
        let dst = buf.raw.contents() as *mut T;
        let n = data.len().min(buf.n);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, n);
        }
    }

    // ── GEMM — dispatches on B-weight dtype ──────────────────────────

    fn gemm(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        let a_buf = a.expect_f32("gemm a");
        let out_buf = out.expect_f32_mut("gemm out");
        let enc = ctx.compute_encoder();
        match b.dtype {
            Dtype::F16 => {
                // f16 weights — route through the f32-activation / f16-weight kernels.
                if m == 1 {
                    st().pipes.gemv_enc_f16w(enc, a_buf, &b.raw, out_buf, n, k);
                } else {
                    st().pipes
                        .gemm_v2_f16w(enc, a_buf, &b.raw, out_buf, m, n, k);
                }
            }
            Dtype::F32 => {
                if m == 1 {
                    // GEMV with K-reduction via simd_sum — good for lm_head
                    // (N = vocab = 152k for Qwen3).
                    st().pipes.gemv_enc(enc, a_buf, &b.raw, out_buf, n, k);
                } else {
                    // GPU simdgroup-matrix tiled GEMM. Replaces an earlier
                    // cblas_sgemm fallback that serialised against GPU work.
                    st().pipes.gemm_v2(enc, a_buf, &b.raw, out_buf, m, n, k);
                }
            }
            // Phase A: integer dtypes are reserved for the upcoming typed-
            // buffer migration (alloc_u32 / write_u32 / from_slice_i32 →
            // typed-buffer alloc + write). No caller creates an integer
            // MetalBuf yet, so reaching this arm = caller bug.
            other => panic!(
                "MetalBackend::gemm: b.dtype = {} unsupported (only F16 / F32)",
                other.name()
            ),
        }
    }

    // ── Norm / attention / fused ops — all f32 ───────────────────────

    fn rms_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let x = x.expect_f32("rms_norm x");
        let w = w.expect_f32("rms_norm w");
        let out = out.expect_f32_mut("rms_norm out");
        let enc = ctx.compute_encoder();
        st().pipes.rms_norm_enc(enc, x, w, out, tokens, dim, eps);
    }

    fn fused_add_rms_norm(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let residual = residual.expect_f32_mut("fused_add_rms_norm residual");
        let x = x.expect_f32("fused_add_rms_norm x");
        let w = w.expect_f32("fused_add_rms_norm w");
        let out = out.expect_f32_mut("fused_add_rms_norm out");
        let enc = ctx.compute_encoder();
        st().pipes.fused_residual_norm_enc(
            enc, residual, x, None, w, residual, out, tokens, dim, eps, 0,
        );
    }

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
    ) {
        let p = AttentionParams {
            batch,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            q_len,
            kv_len,
            head_dim: cfg.head_dim,
            causal: cfg.causal,
            pos_offset,
            sliding_window: cfg.sliding_window,
        };
        let q = q.expect_f32("flash_attention q");
        let k = k.expect_f32("flash_attention k");
        let v = v.expect_f32("flash_attention v");
        let out = out.expect_f32_mut("flash_attention out");
        // flash_attn_v2 opens its own compute encoder internally; close
        // the sticky one first so we don't have two open at once on the
        // same cmd buffer.
        ctx.compute_encoder_end();
        let cmd = ctx.cmd();
        st().pipes
            .flash_attn_v2(cmd, q, k, v, out, &p, cfg.kv_seq_stride);
    }

    fn copy_slice(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        src_offset: usize,
        dst: &mut Self::Buffer,
        dst_offset: usize,
        len: usize,
    ) {
        // Blit encoder stays in the same command buffer as neighbouring
        // compute encoders, keeping the single-command-buffer invariant.
        // Close the sticky compute encoder first — Metal forbids two
        // active encoders on one cmd buffer.
        let src = src.expect_f32("copy_slice src");
        let dst = dst.expect_f32_mut("copy_slice dst");
        ctx.compute_encoder_end();
        let cmd = ctx.cmd();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(
            src,
            (src_offset * 4) as u64,
            dst,
            (dst_offset * 4) as u64,
            (len * 4) as u64,
        );
        blit.end_encoding();
    }

    fn embedding_lookup(
        ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    ) {
        // CPU-side gather into the f32 activation output. `ids.len()` is
        // small (one per batch item); flush any pending GPU work first.
        let out = out.expect_f32_mut("embedding_lookup out");
        ctx.flush();
        unsafe {
            let o = std::slice::from_raw_parts_mut(out.contents() as *mut f32, ids.len() * dim);
            match table.dtype {
                Dtype::F32 => {
                    let t = std::slice::from_raw_parts(
                        table.raw.contents() as *const f32,
                        table.raw.length() as usize / 4,
                    );
                    for (i, &id) in ids.iter().enumerate() {
                        let s = id as usize * dim;
                        o[i * dim..(i + 1) * dim].copy_from_slice(&t[s..s + dim]);
                    }
                }
                Dtype::F16 => {
                    let t = std::slice::from_raw_parts(
                        table.raw.contents() as *const f16,
                        table.raw.length() as usize / 2,
                    );
                    for (i, &id) in ids.iter().enumerate() {
                        let s = id as usize * dim;
                        for j in 0..dim {
                            o[i * dim + j] = t[s + j].to_f32();
                        }
                    }
                }
                other => panic!(
                    "MetalBackend::embedding_lookup: table.dtype = {} unsupported",
                    other.name()
                ),
            }
        }
    }

    fn split_qkv(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        let qkv = qkv.expect_f32("split_qkv qkv");
        let q = q.expect_f32_mut("split_qkv q");
        let k = k.expect_f32_mut("split_qkv k");
        let v = v.expect_f32_mut("split_qkv v");
        let enc = ctx.compute_encoder();
        st().pipes
            .split_qkv_enc(enc, qkv, q, k, v, tokens, q_dim, kv_dim);
    }

    fn fused_silu_mul_split(
        ctx: &mut Self::Context,
        gu: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        let gu = gu.expect_f32("fused_silu_mul_split gate_up");
        let out = out.expect_f32_mut("fused_silu_mul_split out");
        let enc = ctx.compute_encoder();
        st().pipes.silu_mul_split_enc(enc, gu, out, tokens, im);
    }

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
    ) {
        let input = input.expect_f32("qk_norm_rope input");
        let norm_w = norm_w.expect_f32("qk_norm_rope norm_w");
        let cos = cos.expect_f32("qk_norm_rope cos");
        let sin = sin.expect_f32("qk_norm_rope sin");
        let output = output.expect_f32_mut("qk_norm_rope output");
        let enc = ctx.compute_encoder();
        st().pipes.qk_norm_rope(
            enc, input, norm_w, cos, sin, output, tokens, heads, head_dim, pos_offset, eps, mode,
        );
    }

    fn split_qkv_norm_rope(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        k_out: &mut Self::Buffer,
        v_out: &mut Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()> {
        let qkv = qkv.expect_f32("split_qkv_norm_rope qkv");
        let q_norm_w = q_norm_w.expect_f32("split_qkv_norm_rope q_norm_w");
        let k_norm_w = k_norm_w.expect_f32("split_qkv_norm_rope k_norm_w");
        let cos = cos.expect_f32("split_qkv_norm_rope cos");
        let sin = sin.expect_f32("split_qkv_norm_rope sin");
        let q_out = q_out.expect_f32_mut("split_qkv_norm_rope q_out");
        let k_out = k_out.expect_f32_mut("split_qkv_norm_rope k_out");
        let v_out = v_out.expect_f32_mut("split_qkv_norm_rope v_out");
        let enc = ctx.compute_encoder();
        st().pipes.split_qkv_norm_rope(
            enc, qkv, q_norm_w, k_norm_w, cos, sin, q_out, k_out, v_out, tokens, q_heads, kv_heads,
            head_dim, pos_offset, eps, qk_mode,
        );
        Ok(())
    }

    fn split_qkv_norm_rope_into_cache(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
        cache_len: usize,
        cache_capacity: usize,
    ) -> Result<()> {
        let qkv = qkv.expect_f32("split_qkv_norm_rope_kvc qkv");
        let q_norm_w = q_norm_w.expect_f32("split_qkv_norm_rope_kvc q_norm_w");
        let k_norm_w = k_norm_w.expect_f32("split_qkv_norm_rope_kvc k_norm_w");
        let cos = cos.expect_f32("split_qkv_norm_rope_kvc cos");
        let sin = sin.expect_f32("split_qkv_norm_rope_kvc sin");
        let q_out = q_out.expect_f32_mut("split_qkv_norm_rope_kvc q_out");
        let cache_k = cache_k.expect_f32_mut("split_qkv_norm_rope_kvc cache_k");
        let cache_v = cache_v.expect_f32_mut("split_qkv_norm_rope_kvc cache_v");
        let enc = ctx.compute_encoder();
        st().pipes.split_qkv_norm_rope_into_cache(
            enc,
            qkv,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            q_out,
            cache_k,
            cache_v,
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            pos_offset,
            eps,
            qk_mode,
            cache_len,
            cache_capacity,
        );
        Ok(())
    }

    /// Phase D step 2+3 unified typed allocator. Replaces alloc_u32.
    fn alloc_typed(dtype: Dtype, n: usize) -> Self::Buffer {
        let bytes = (n * dtype.bytes_per_elem()) as u64;
        let raw = st()
            .pipes
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        MetalBuf { raw, dtype, n }
    }

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
    ) {
        debug_assert!(cache_len + new_tokens <= cache_capacity);
        let cache_k = cache_k.expect_f32_mut("kv_cache_append cache_k");
        let cache_v = cache_v.expect_f32_mut("kv_cache_append cache_v");
        let new_k_head_major = new_k_head_major.expect_f32("kv_cache_append new_k");
        let new_v_head_major = new_v_head_major.expect_f32("kv_cache_append new_v");
        let enc = ctx.compute_encoder();
        st().pipes.kv_cache_append(
            enc,
            new_k_head_major,
            cache_k,
            nkv,
            hd,
            cache_len,
            new_tokens,
            cache_capacity,
        );
        st().pipes.kv_cache_append(
            enc,
            new_v_head_major,
            cache_v,
            nkv,
            hd,
            cache_len,
            new_tokens,
            cache_capacity,
        );
    }

    fn transpose_head_to_token(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        let src = src.expect_f32("transpose_head_to_token src");
        let dst = dst.expect_f32_mut("transpose_head_to_token dst");
        let enc = ctx.compute_encoder();
        st().pipes.transpose_out(enc, src, dst, tokens, heads, dim);
    }

    fn add_bias(
        ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let data = data.expect_f32_mut("add_bias data");
        let bias = bias.expect_f32("add_bias bias");
        let enc = ctx.compute_encoder();
        st().pipes.add_bias_enc(enc, data, bias, rows, cols);
    }

    fn layer_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let x = x.expect_f32("layer_norm x");
        let gamma = gamma.expect_f32("layer_norm gamma");
        let beta = beta.expect_f32("layer_norm beta");
        let out = out.expect_f32_mut("layer_norm out");
        let enc = ctx.compute_encoder();
        st().pipes
            .layer_norm_enc(enc, x, gamma, beta, out, tokens, dim, eps);
    }

    fn gelu(ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        let x = x.expect_f32("gelu x");
        let out = out.expect_f32_mut("gelu out");
        let enc = ctx.compute_encoder();
        st().pipes.gelu_enc(enc, x, out, len);
    }

    fn add_inplace(ctx: &mut Self::Context, r: &mut Self::Buffer, x: &Self::Buffer, len: usize) {
        let r = r.expect_f32_mut("add_inplace r");
        let x = x.expect_f32("add_inplace x");
        let enc = ctx.compute_encoder();
        st().pipes.add_enc(enc, r, x, r, len);
    }

    fn scaled_add_inplace(
        ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        src: &Self::Buffer,
        scale: f32,
        len: usize,
    ) {
        let dst_buf = dst.expect_f32_mut("scaled_add_inplace dst");
        let src_buf = src.expect_f32("scaled_add_inplace src");
        let enc = ctx.compute_encoder();
        st().pipes
            .scaled_add_inplace_enc(enc, dst_buf, src_buf, scale, len);
    }

    // ── Buffer management ────────────────────────────────────────────

    fn alloc(len: usize) -> Self::Buffer {
        // Scratch / output buffers are always f32 — the precision-sensitive
        // compute paths expect it.
        MetalBuf {
            raw: alloc_f32_raw(len),
            dtype: Dtype::F32,
            n: len,
        }
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        match buf.dtype {
            Dtype::F32 => {
                let ptr = buf.raw.contents() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
            }
            Dtype::F16 => {
                let ptr = buf.raw.contents() as *const f16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                slice.iter().map(|h| h.to_f32()).collect()
            }
            other => panic!(
                "MetalBackend::to_vec: buf.dtype = {} unsupported (expected F32 or F16)",
                other.name()
            ),
        }
    }

    fn argmax_rows_f16(
        ctx: &mut Self::Context,
        logits: &Self::Buffer,
        m: usize,
        n: usize,
    ) -> Result<Vec<u32>> {
        if !matches!(logits.dtype, Dtype::F32) {
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
            return Ok(out);
        }

        #[repr(C)]
        struct ArgmaxParams {
            n: i32,
        }

        let out = st().pipes.device.new_buffer(
            (m * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = ArgmaxParams { n: n as i32 };
        let enc = ctx.compute_encoder();
        enc.set_compute_pipeline_state(st().pipes.pipeline("argmax_rows_f32"));
        enc.set_buffer(0, Some(&logits.raw), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<ArgmaxParams>() as u64,
            &params as *const _ as *const c_void,
        );
        enc.dispatch_thread_groups(MTLSize::new(1, m as u64, 1), MTLSize::new(256, 1, 1));
        ctx.flush();
        let ptr = out.contents() as *const u32;
        let tokens = unsafe { std::slice::from_raw_parts(ptr, m).to_vec() };
        Ok(tokens)
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        // Activations, cos/sin tables, temporary scratch — all f32.
        MetalBuf {
            raw: buffer_from_f32_slice(data),
            dtype: Dtype::F32,
            n: data.len(),
        }
    }

    fn from_weight_bytes(raw: &[u8], src_dtype: SrcDtype) -> Self::Buffer {
        let n = raw.len() / src_dtype.bytes_per_elem();
        let want_f16 = prefer_f16_weights() && n >= F16_MIN_ELEMS;

        if !want_f16 {
            // Default behaviour: materialise as f32, matches pre-refactor
            // MetalBackend byte-for-byte.
            let data = src_dtype.to_f32_vec(raw);
            return MetalBuf {
                raw: buffer_from_f32_slice(&data),
                dtype: Dtype::F32,
                n,
            };
        }

        // f16 storage path — go directly from raw bytes where possible to
        // avoid the transient 2× RAM spike.
        match src_dtype {
            SrcDtype::F16 => MetalBuf {
                raw: buffer_from_f16_bytes(raw),
                dtype: Dtype::F16,
                n,
            },
            SrcDtype::BF16 => {
                // bf16 → f16 via f32. Loses magnitude range (bf16 has a
                // broader exponent) but gains mantissa precision, which for
                // typical weight magnitudes (|w| < 32) is a net upgrade.
                let mut f16_bytes = vec![0u8; n * 2];
                for i in 0..n {
                    let v = bf16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]).to_f32();
                    let h = f16::from_f32(v).to_le_bytes();
                    f16_bytes[i * 2] = h[0];
                    f16_bytes[i * 2 + 1] = h[1];
                }
                MetalBuf {
                    raw: buffer_from_f16_bytes(&f16_bytes),
                    dtype: Dtype::F16,
                    n,
                }
            }
            SrcDtype::F32 => {
                // f32 → f16 downcast. Halves storage.
                let data = src_dtype.to_f32_vec(raw);
                MetalBuf {
                    raw: buffer_f16_from_f32(&data),
                    dtype: Dtype::F16,
                    n,
                }
            }
        }
    }
}

// Metal has no graph-capture analogue; inherit BackendGraph defaults.
impl crate::backend::BackendGraph for MetalBackend {}

// Metal has no multi-GPU collectives; inherit BackendCollective single-rank defaults.
impl crate::backend::BackendCollective for MetalBackend {}

// Metal does not ship Marlin INT4 kernels; inherit unsupported defaults.
impl crate::backend::BackendQuantMarlin for MetalBackend {}

// Metal: existing KV cache path is FP16.
impl crate::backend::BackendKvDtype<crate::backend::KvFp16> for MetalBackend {
    type KvBuffer = <Self as crate::backend::Backend>::Buffer;
    type KvScales = ();
}
