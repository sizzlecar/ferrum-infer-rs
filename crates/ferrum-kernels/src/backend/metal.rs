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

use super::{AttnConfig, Backend, SrcDtype};
use ferrum_attention::metal::pipelines::MetalPipelines;
use ferrum_attention::AttentionParams;
use ferrum_types::{FerrumError, Result};
use half::{bf16, f16};
use metal::{Device, MTLResourceOptions};
use std::ffi::c_void;
use std::sync::OnceLock;

// ── Shared Metal state ────────────────────────────────────────────────

struct MetalState {
    pipes: MetalPipelines,
}
static METAL_STATE: OnceLock<MetalState> = OnceLock::new();
fn st() -> &'static MetalState {
    METAL_STATE.get_or_init(|| MetalState {
        pipes: MetalPipelines::new(&Device::system_default().unwrap()),
    })
}

// ── Dtype tag + tagged buffer ─────────────────────────────────────────

/// Element storage type for a [`MetalBuf`]. Same shape generalises to INT8
/// / bf16 etc. when their shaders land — just add a variant + wire it in
/// the op dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
}

impl Dtype {
    pub const fn bytes_per_elem(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
        }
    }
}

/// Metal device buffer with a runtime dtype tag and logical element count.
///
/// Two buffers of identical raw bytes but different `dtype` are treated as
/// different types for shader selection. `n` is the number of logical
/// elements; `raw.length()` is `n * dtype.bytes_per_elem()`.
pub struct MetalBuf {
    raw: metal::Buffer,
    dtype: Dtype,
    n: usize,
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
    fn expect_f32<'a>(&'a self, what: &str) -> &'a metal::Buffer {
        debug_assert!(
            matches!(self.dtype, Dtype::F32),
            "{what}: expected F32 buffer, got {:?}",
            self.dtype
        );
        &self.raw
    }
    #[inline]
    fn expect_f32_mut<'a>(&'a mut self, what: &str) -> &'a mut metal::Buffer {
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
///
/// Also owns a per-shape pool of fp16 dequant transients used by
/// `gemm_quant`. Without it, every Q4 matmul allocates a fresh
/// StorageModePrivate buffer that the in-flight command buffer retains
/// until commit; on a 36-layer model with 4 quant matmuls per layer
/// that grew to multi-GB of leaked transients between flushes.
/// With pooling, ~4 distinct shapes per model survive — bounded ~200 MB.
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
    /// Reusable fp16 dequant targets, keyed by byte-size. We index by
    /// size rather than (n_rows,n_cols) because shape isn't observed by
    /// the dequant/gemm kernels — they just need a buffer of the right
    /// length. Same size from any matmul shares a slot.
    quant_transients: std::collections::HashMap<u64, metal::Buffer>,
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
    fn compute_encoder(&mut self) -> &'static metal::ComputeCommandEncoderRef {
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

    /// Get or create a private-storage fp16 transient buffer of exactly
    /// `size_bytes`. Reused across `gemm_quant` calls so the in-flight
    /// command buffer retains a single object per shape, not one per call.
    fn quant_transient(&mut self, size_bytes: u64) -> &metal::Buffer {
        // HashMap::entry borrow gymnastics: insert if missing, then return ref.
        if !self.quant_transients.contains_key(&size_bytes) {
            let buf = st()
                .pipes
                .device
                .new_buffer(size_bytes, MTLResourceOptions::StorageModePrivate);
            self.quant_transients.insert(size_bytes, buf);
        }
        self.quant_transients.get(&size_bytes).unwrap()
    }

    fn flush(&mut self) {
        self.compute_encoder_end();
        if let Some(cmd) = self.cmd.take() {
            cmd.commit();
            cmd.wait_until_completed();
        }
    }
}

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

// ── Policy: should big weights land as f16? ───────────────────────────
// Cached on first read so each tensor load doesn't re-parse env.

fn prefer_f16_weights() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("FERRUM_METAL_DTYPE")
            .map(|v| v.eq_ignore_ascii_case("f16"))
            .unwrap_or(false)
    })
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
        blocks: metal::Buffer, // [n_blocks * 144] bytes
        n_rows: usize,
        n_cols: usize,
        n_blocks: usize,
    },
}

// SAFETY: metal::Buffer wraps an Objective-C handle. metal-rs marks it
// Send+Sync via internal unsafe impls; we just propagate that.
unsafe impl Send for MetalQuantStore {}
unsafe impl Sync for MetalQuantStore {}

impl Backend for MetalBackend {
    type Buffer = MetalBuf;
    type Context = MetalContext;
    type GptqStore = (); // Metal GPTQ not yet wired; load_gptq/gemm_gptq return unsupported.
    type QuantStore = MetalQuantStore;

    fn new_context() -> Self::Context {
        MetalContext {
            cmd: None,
            encoder: None,
            quant_transients: std::collections::HashMap::new(),
        }
    }
    fn sync(ctx: &mut Self::Context) {
        ctx.flush();
    }

    // ── Q4_K_M ────────────────────────────────────────────────────────

    fn load_quant(
        kind: super::GgufQuantType,
        bytes: &[u8],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Self::QuantStore> {
        use super::GgufQuantType;
        match kind {
            GgufQuantType::Q4K => {
                const Q4_K_QK: usize = 256;
                const Q4_K_BLOCK_BYTES: usize = 144;
                let total_elems = n_rows * n_cols;
                if total_elems % Q4_K_QK != 0 {
                    return Err(FerrumError::model(format!(
                        "load_quant Q4K: elements {total_elems} not multiple of {Q4_K_QK}"
                    )));
                }
                let n_blocks = total_elems / Q4_K_QK;
                let expected = n_blocks * Q4_K_BLOCK_BYTES;
                if bytes.len() != expected {
                    return Err(FerrumError::model(format!(
                        "load_quant Q4K: bytes {} != expected {} ({n_blocks} blocks)",
                        bytes.len(),
                        expected
                    )));
                }
                let blocks = st().pipes.device.new_buffer_with_data(
                    bytes.as_ptr() as *const c_void,
                    bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                Ok(MetalQuantStore::Q4K {
                    blocks,
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

    fn gemm_quant(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        weight: &Self::QuantStore,
        out: &mut Self::Buffer,
        m: usize,
    ) -> Result<()> {
        let MetalQuantStore::Q4K {
            blocks,
            n_rows,
            n_cols,
            n_blocks,
        } = weight;

        let total_elems = (*n_rows * *n_cols) as u64;
        let f16_bytes = total_elems * 2;
        // Snapshot the values we need before the mutable borrow on ctx,
        // since the borrow checker can't see that `quant_transient` and
        // `cmd` touch disjoint fields of MetalContext.
        let n_rows = *n_rows;
        let n_cols = *n_cols;
        let n_blocks = *n_blocks;

        let _t0 = if debug_per_call_flush() {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let a_buf = a.expect_f32("gemm_quant Q4K a");
        let out_buf = out.expect_f32_mut("gemm_quant Q4K out");

        if m == 1 {
            // **Fused path** for decode (m=1): one kernel reads the Q4
            // super-blocks, decodes them inline, and reduces against `A`.
            // No transient fp16 buffer materialised — saves ~64 MB of
            // intermediate write+read per 4K×4K matmul, which is a big
            // chunk of the per-token wall on memory-bandwidth-bound
            // hardware. Requires `K % 256 == 0` (always holds for Q4_K_M
            // tensors).
            let enc = ctx.compute_encoder();
            crate::q4_k_gemv::dispatch_gemv_q4k_on_encoder(
                &st().pipes.device,
                enc,
                a_buf,
                blocks,
                out_buf,
                n_rows,
                n_cols,
            );
        } else {
            // **Dequant→transient→GEMM** path for prefill (m > 1). Pool
            // the fp16 transient by byte-size so repeated shapes share
            // one allocation; same encoder for both dispatches lets the
            // GEMM see dequant's writes without an explicit fence.
            let transient_ptr: *const metal::Buffer = ctx.quant_transient(f16_bytes);
            // SAFETY: transient_ptr points into ctx.quant_transients (a
            // HashMap we own); the buffer outlives this function and the
            // cmd buffer we encode into. The raw pointer dance is needed
            // because the next call takes &mut ctx again for `cmd`.
            let transient: &metal::Buffer = unsafe { &*transient_ptr };
            let enc = ctx.compute_encoder();
            crate::q4_k::dispatch_dequant_q4_k_on_encoder(
                &st().pipes.device,
                enc,
                blocks,
                transient,
                n_blocks,
            );
            st().pipes
                .gemm_v2_f16w(enc, a_buf, transient, out_buf, m, n_rows, n_cols);
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
        }
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
