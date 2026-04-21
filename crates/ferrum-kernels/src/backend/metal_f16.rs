//! `MetalF16Backend` — Metal backend with fp16 weight storage, f32 activations.
//!
//! Parallel to [`metal::MetalBackend`] but designed for memory-constrained Macs
//! (e.g. M1 16 GB running a 4B model). Only the big weight tensors (attention
//! + MLP projections, lm_head) are stored as half-precision — halving their
//! footprint. Activations, norms, KV cache, and everything allocated via
//! `alloc`/`from_slice` stay as f32 so the precision-sensitive paths (norm,
//! rope, attention) are unchanged.
//!
//! The `Buffer` type is an enum tagged with its dtype; `gemm` dispatches on
//! the weight dtype: when `b` is `F16` we run the `gemm_f32a_f16w_v2` /
//! `gemv_f32a_f16w` kernels; otherwise we fall back to the f32 path used by
//! `MetalBackend`. All other trait methods assert their inputs are f32.
//!
//! This backend reuses `MetalBackend`'s command queue / pipeline cache
//! via the shared `metal::st()` singleton.
//!
//! Shaders live in `ferrum-attention::metal::shaders::gemm_f16w.metal` and
//! pipelines are pre-compiled by `MetalPipelines::new`.

use super::{AttnConfig, Backend, SrcDtype};
use ferrum_attention::metal::pipelines::MetalPipelines;
use ferrum_attention::AttentionParams;
use half::{bf16, f16};
use metal::{Device, MTLResourceOptions};
use std::ffi::c_void;
use std::sync::OnceLock;

// ── Shared state ──────────────────────────────────────────────────────

struct MetalState {
    pipes: MetalPipelines,
}
static METAL_F16_STATE: OnceLock<MetalState> = OnceLock::new();
fn st() -> &'static MetalState {
    METAL_F16_STATE.get_or_init(|| MetalState {
        pipes: MetalPipelines::new(&Device::system_default().unwrap()),
    })
}

// ── Buffer ────────────────────────────────────────────────────────────

/// Tagged Metal buffer. Weights live as `F16`; activations / scratch / norm
/// weights / KV cache live as `F32`.
pub enum F16Buffer {
    F32 { raw: metal::Buffer, n: usize },
    F16 { raw: metal::Buffer, n: usize },
}

impl F16Buffer {
    pub fn raw(&self) -> &metal::Buffer {
        match self {
            F16Buffer::F32 { raw, .. } | F16Buffer::F16 { raw, .. } => raw,
        }
    }
    pub fn raw_mut(&mut self) -> &mut metal::Buffer {
        match self {
            F16Buffer::F32 { raw, .. } | F16Buffer::F16 { raw, .. } => raw,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            F16Buffer::F32 { n, .. } | F16Buffer::F16 { n, .. } => *n,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn is_f16(&self) -> bool {
        matches!(self, F16Buffer::F16 { .. })
    }
    fn expect_f32(&self, what: &str) -> &metal::Buffer {
        match self {
            F16Buffer::F32 { raw, .. } => raw,
            F16Buffer::F16 { .. } => panic!("{what}: expected F32 buffer, got F16"),
        }
    }
    fn expect_f32_mut(&mut self, what: &str) -> &mut metal::Buffer {
        match self {
            F16Buffer::F32 { raw, .. } => raw,
            F16Buffer::F16 { .. } => panic!("{what}: expected F32 buffer, got F16"),
        }
    }
}

// Safety: metal::Buffer is Send + Sync (wraps an NSObject pointer).
unsafe impl Send for F16Buffer {}
unsafe impl Sync for F16Buffer {}

// ── Context ───────────────────────────────────────────────────────────

pub struct MetalF16Context {
    cmd: Option<&'static metal::CommandBufferRef>,
}

impl MetalF16Context {
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

    fn flush(&mut self) {
        if let Some(cmd) = self.cmd.take() {
            cmd.commit();
            cmd.wait_until_completed();
        }
    }
}

// ── Backend impl ──────────────────────────────────────────────────────

pub struct MetalF16Backend;

/// Allocate a device buffer of `n` f32 elements, zero-initialised (Metal
/// guarantees shared-mode buffers are zero on alloc).
fn alloc_f32_raw(n: usize) -> metal::Buffer {
    st().pipes.buffer_empty(n)
}

/// Allocate a device buffer of `n` f16 elements.
fn alloc_f16_raw(n: usize) -> metal::Buffer {
    st().pipes.device.new_buffer(
        (n * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Copy f32 data into a newly-allocated shared Metal buffer.
fn buffer_from_f32(data: &[f32]) -> metal::Buffer {
    st().pipes.device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Copy raw half-precision bytes into a newly-allocated shared Metal buffer.
fn buffer_from_f16_bytes(bytes: &[u8]) -> metal::Buffer {
    debug_assert_eq!(bytes.len() % 2, 0);
    st().pipes.device.new_buffer_with_data(
        bytes.as_ptr() as *const c_void,
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

impl Backend for MetalF16Backend {
    type Buffer = F16Buffer;
    type Context = MetalF16Context;
    type GptqStore = (); // Not yet wired — GPTQ stays on the CUDA / Marlin path.

    fn new_context() -> Self::Context {
        MetalF16Context { cmd: None }
    }
    fn sync(ctx: &mut Self::Context) {
        ctx.flush();
    }

    // ── GEMM — mixed-precision path ────────────────────────────────────

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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        match b {
            F16Buffer::F16 { raw: b_buf, .. } => {
                // Big weight in fp16 — use the f32-activation / f16-weight kernels.
                if m == 1 {
                    st().pipes.gemv_enc_f16w(enc, a_buf, b_buf, out_buf, n, k);
                } else {
                    st().pipes.gemm_v2_f16w(enc, a_buf, b_buf, out_buf, m, n, k);
                }
            }
            F16Buffer::F32 { raw: b_buf, .. } => {
                // Fallback for any f32-stored weights (e.g. if a backend
                // caller hasn't threaded fp16 through). Same paths as MetalBackend.
                if m == 1 {
                    st().pipes.gemv_enc(enc, a_buf, b_buf, out_buf, n, k);
                } else {
                    st().pipes.gemm_v2(enc, a_buf, b_buf, out_buf, m, n, k);
                }
            }
        }
        enc.end_encoding();
    }

    // ── Norm / attention / fused ops — all f32 ─────────────────────────

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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.rms_norm_enc(enc, x, w, out, tokens, dim, eps);
        enc.end_encoding();
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.fused_residual_norm_enc(
            enc, residual, x, None, w, residual, out, tokens, dim, eps, 0,
        );
        enc.end_encoding();
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
        // Both src and dst must be f32 (copy_slice is only used for activation
        // slicing). The offsets are in elements; stride = 4 bytes.
        let src = src.expect_f32("copy_slice src");
        let dst = dst.expect_f32_mut("copy_slice dst");
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
        // CPU-side gather (ids.len() is small — one per batch item), writing
        // into the shared-memory output buffer. Must flush any pending GPU
        // work that might read `out` before this write.
        let out = out.expect_f32_mut("embedding_lookup out");
        ctx.flush();
        unsafe {
            let o = std::slice::from_raw_parts_mut(out.contents() as *mut f32, ids.len() * dim);
            match table {
                F16Buffer::F32 { raw: table, .. } => {
                    let t = std::slice::from_raw_parts(
                        table.contents() as *const f32,
                        table.length() as usize / 4,
                    );
                    for (i, &id) in ids.iter().enumerate() {
                        let s = id as usize * dim;
                        o[i * dim..(i + 1) * dim].copy_from_slice(&t[s..s + dim]);
                    }
                }
                F16Buffer::F16 { raw: table, .. } => {
                    // Upcast fp16 → f32 row by row. Loop is cheap: dim ≤
                    // few thousand, called once per token.
                    let t = std::slice::from_raw_parts(
                        table.contents() as *const f16,
                        table.length() as usize / 2,
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes
            .split_qkv_enc(enc, qkv, q, k, v, tokens, q_dim, kv_dim);
        enc.end_encoding();
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.silu_mul_split_enc(enc, gu, out, tokens, im);
        enc.end_encoding();
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.qk_norm_rope(
            enc, input, norm_w, cos, sin, output, tokens, heads, head_dim, pos_offset, eps, mode,
        );
        enc.end_encoding();
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
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
        enc.end_encoding();
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.transpose_out(enc, src, dst, tokens, heads, dim);
        enc.end_encoding();
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.add_bias_enc(enc, data, bias, rows, cols);
        enc.end_encoding();
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes
            .layer_norm_enc(enc, x, gamma, beta, out, tokens, dim, eps);
        enc.end_encoding();
    }

    fn gelu(ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        let x = x.expect_f32("gelu x");
        let out = out.expect_f32_mut("gelu out");
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.gelu_enc(enc, x, out, len);
        enc.end_encoding();
    }

    fn add_inplace(ctx: &mut Self::Context, r: &mut Self::Buffer, x: &Self::Buffer, len: usize) {
        let r = r.expect_f32_mut("add_inplace r");
        let x = x.expect_f32("add_inplace x");
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.add_enc(enc, r, x, r, len);
        enc.end_encoding();
    }

    // ── Buffer management ─────────────────────────────────────────────

    fn alloc(len: usize) -> Self::Buffer {
        F16Buffer::F32 {
            raw: alloc_f32_raw(len),
            n: len,
        }
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        match buf {
            F16Buffer::F32 { raw, .. } => {
                let ptr = raw.contents() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
            }
            F16Buffer::F16 { raw, .. } => {
                let ptr = raw.contents() as *const f16;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                slice.iter().map(|h| h.to_f32()).collect()
            }
        }
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        // `from_slice` is used for activations, cos/sin tables, temporary
        // scratch — all precision-sensitive. Always store as f32 on this
        // backend; weights go through `from_weight_bytes`.
        F16Buffer::F32 {
            raw: buffer_from_f32(data),
            n: data.len(),
        }
    }

    fn from_weight_bytes(raw: &[u8], src_dtype: SrcDtype) -> Self::Buffer {
        // Size-threshold dispatch: the tiny tensors (norm weights, QK-norm
        // weights, biases) only save kilobytes if put in f16 but would force
        // every shader that reads them to pick a dtype. Keep them f32.
        //
        // 1M elements × 4 bytes = 4 MB as f32, 2 MB as f16 — anything
        // smaller isn't worth the complexity. Embed table and projection
        // weights are all > 1M elements, so they take the f16 path.
        let n = raw.len() / src_dtype.bytes_per_elem();
        if n < F16_MIN_ELEMS {
            return F16Buffer::F32 {
                raw: buffer_from_f32(&src_dtype.to_f32_vec(raw)),
                n,
            };
        }

        // Materialise the weight directly into a half-precision Metal buffer.
        // This is the whole point of MetalF16Backend — avoids the 2× host
        // RAM spike the default impl incurs.
        match src_dtype {
            SrcDtype::F16 => F16Buffer::F16 {
                raw: buffer_from_f16_bytes(raw),
                n,
            },
            SrcDtype::BF16 => {
                // bf16 → f16: go via f32. Loses a tiny bit of magnitude
                // range (bf16's broader exponent is the only thing we can't
                // keep) but gains mantissa precision.
                let mut f16_bytes = vec![0u8; n * 2];
                for i in 0..n {
                    let v = bf16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]).to_f32();
                    let h = f16::from_f32(v).to_le_bytes();
                    f16_bytes[i * 2] = h[0];
                    f16_bytes[i * 2 + 1] = h[1];
                }
                F16Buffer::F16 {
                    raw: buffer_from_f16_bytes(&f16_bytes),
                    n,
                }
            }
            SrcDtype::F32 => {
                // Source is f32 → downcast. Halves storage.
                let mut f16_bytes = vec![0u8; n * 2];
                for i in 0..n {
                    let bytes = [
                        raw[i * 4],
                        raw[i * 4 + 1],
                        raw[i * 4 + 2],
                        raw[i * 4 + 3],
                    ];
                    let v = f32::from_le_bytes(bytes);
                    let h = f16::from_f32(v).to_le_bytes();
                    f16_bytes[i * 2] = h[0];
                    f16_bytes[i * 2 + 1] = h[1];
                }
                F16Buffer::F16 {
                    raw: buffer_from_f16_bytes(&f16_bytes),
                    n,
                }
            }
        }
    }
}

/// Element-count threshold above which `from_weight_bytes` stores as f16.
/// 1 M elems = 4 MB as f32 / 2 MB as f16 — anything smaller (norm weights,
/// small biases) stays f32 to avoid feeding f16 into shaders that expect f32.
const F16_MIN_ELEMS: usize = 1_048_576;

// Silence unused-var complaints for helpers that will land with GPTQ later.
#[allow(dead_code)]
fn _touch_alloc_f16() {
    let _ = alloc_f16_raw(0);
}
