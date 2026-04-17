//! Metal backend — pipelined command buffer with manual retain/release.
//!
//! Pattern from llama.cpp: [queue commandBuffer] + [cmd_buf retain].
//! One command buffer per forward pass; GPU ops encode into it; sync() commits.

use super::{AttnConfig, Backend};
use ferrum_attention::metal::pipelines::MetalPipelines;
use ferrum_attention::AttentionParams;
use metal::Device;
use std::sync::OnceLock;

struct MetalState {
    pipes: MetalPipelines,
}
static METAL_STATE: OnceLock<MetalState> = OnceLock::new();
fn st() -> &'static MetalState {
    METAL_STATE.get_or_init(|| MetalState {
        pipes: MetalPipelines::new(&Device::system_default().unwrap()),
    })
}

pub struct MetalBackend;

#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Metal context — accumulates GPU work in a single command buffer across
/// multiple Backend method calls. `sync()` commits and creates a fresh one on demand.
///
/// The `&'static CommandBufferRef` lifetime is safe because the command queue lives in
/// a `OnceLock` (see [`st`]) for the program's lifetime; autoreleased command buffers
/// are retained for the duration of the queue.
pub struct MetalContext {
    cmd: Option<&'static metal::CommandBufferRef>,
}

impl MetalContext {
    /// Return the current in-flight command buffer, creating one on first use.
    fn cmd(&mut self) -> &'static metal::CommandBufferRef {
        match self.cmd {
            Some(c) => c,
            None => {
                let c = st().pipes.queue.new_command_buffer();
                // Erase the queue's lifetime; the queue is static (OnceLock) so this is safe.
                let c_static: &'static metal::CommandBufferRef =
                    unsafe { std::mem::transmute::<&metal::CommandBufferRef, _>(c) };
                self.cmd = Some(c_static);
                c_static
            }
        }
    }

    /// Commit pending work and wait for completion.
    fn flush(&mut self) {
        if let Some(cmd) = self.cmd.take() {
            cmd.commit();
            cmd.wait_until_completed();
        }
    }
}

/// One-shot helper: open a cmd buffer, encode, commit+wait.
/// Used only when ctx-driven batching is not needed (tests, isolated ops).
fn run(f: impl FnOnce(&metal::CommandBufferRef)) {
    let cmd = st().pipes.queue.new_command_buffer();
    f(cmd);
    cmd.commit();
    cmd.wait_until_completed();
}

impl Backend for MetalBackend {
    type Buffer = metal::Buffer;
    type Context = MetalContext;

    fn new_context() -> Self::Context {
        MetalContext { cmd: None }
    }
    fn sync(ctx: &mut Self::Context) {
        ctx.flush();
    }

    fn gemm(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        if m == 1 {
            // GPU GEMV: one threadgroup per output col, K-reduction via simd_sum.
            // Great occupancy for lm_head (N = vocab = 152k for Qwen3).
            let cmd = ctx.cmd();
            let enc = cmd.new_compute_command_encoder();
            st().pipes.gemv_enc(enc, a, b, out, n, k);
            enc.end_encoding();
        } else {
            // Multi-row: Accelerate cblas on shared memory. Needs flush first.
            ctx.flush();
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    112,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.contents() as *const f32,
                    k as i32,
                    b.contents() as *const f32,
                    k as i32,
                    0.0,
                    out.contents() as *mut f32,
                    n as i32,
                );
            }
        }
    }

    fn rms_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
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
        // CPU table lookup writing to shared buffer; must flush any readers of `out`.
        ctx.flush();
        unsafe {
            let t = std::slice::from_raw_parts(
                table.contents() as *const f32,
                table.length() as usize / 4,
            );
            let o = std::slice::from_raw_parts_mut(out.contents() as *mut f32, ids.len() * dim);
            for (i, &id) in ids.iter().enumerate() {
                let s = id as usize * dim;
                o[i * dim..(i + 1) * dim].copy_from_slice(&t[s..s + dim]);
            }
        }
    }

    // ── Metal shader dispatches, no ctx flush ────────────────────────────
    // These were CPU scalar loops on shared memory in the layer_forward_fused
    // era (where the override bypassed the trait). Now that models call them
    // directly they must encode into ctx.cmd() like every other GPU op so the
    // single-command-buffer batching stays intact.

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
        // Fused norm (optional) + RoPE (optional) + transpose, all in one
        // Metal dispatch. See ferrum_attention::metal::shaders::norm_rope.
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
        // Pre-allocated head-major append: data already laid out as
        // [nkv, new_tokens, hd] (e.g. produced by qk_norm_rope). Writes into
        // cache slot [nkv, cache_len .. cache_len+new_tokens, hd].
        //
        // No allocation; no transpose; one compute encoder on ctx.cmd().
        debug_assert!(cache_len + new_tokens <= cache_capacity);
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
        // [heads, tokens, dim] → [tokens, heads, dim] via dedicated Metal shader.
        // Called after flash_attention to return Q in row-major layout for O-proj.
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
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes
            .layer_norm_enc(enc, x, gamma, beta, out, tokens, dim, eps);
        enc.end_encoding();
    }

    fn gelu(ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.gelu_enc(enc, x, out, len);
        enc.end_encoding();
    }

    fn add_inplace(ctx: &mut Self::Context, r: &mut Self::Buffer, x: &Self::Buffer, len: usize) {
        // Fused in-place add via add_enc with output == residual input.
        // Metal's `add_f32` kernel reads a[tid] + b[tid] -> out[tid]; using the
        // same buffer for a and out is well-defined because each thread handles
        // exactly one element with no cross-thread dependency.
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.add_enc(enc, r, x, r, len);
        enc.end_encoding();
    }

    fn alloc(len: usize) -> Self::Buffer {
        st().pipes.buffer_empty(len)
    }
    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        MetalPipelines::read_buffer(buf, len)
    }
    fn from_slice(data: &[f32]) -> Self::Buffer {
        st().pipes.buffer_from_data(data)
    }
}

