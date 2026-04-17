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

    fn decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        kc: &Self::Buffer,
        vc: &Self::Buffer,
        out: &mut Self::Buffer,
        kv_len: usize,
        cfg: &AttnConfig,
    ) {
        let p = AttentionParams {
            batch: 1,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            q_len: 1,
            kv_len,
            head_dim: cfg.head_dim,
            causal: false,
            pos_offset: 0,
        };
        let cmd = ctx.cmd();
        st().pipes.flash_attn(cmd, q, kc, vc, out, &p);
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
        };
        let cmd = ctx.cmd();
        st().pipes.flash_attn(cmd, q, k, v, out, &p);
    }

    fn silu_mul(
        ctx: &mut Self::Context,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    ) {
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.silu_mul_enc(enc, gate, up, out, len);
        enc.end_encoding();
    }

    fn add(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    ) {
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes.add_enc(enc, a, b, out, len);
        enc.end_encoding();
    }

    fn copy(ctx: &mut Self::Context, src: &Self::Buffer, dst: &mut Self::Buffer, len: usize) {
        // CPU memcpy on shared buffers; must flush pending GPU writes first.
        ctx.flush();
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.contents() as *const u8,
                dst.contents() as *mut u8,
                len * 4,
            );
        }
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

    fn qk_norm(
        ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        w: &Self::Buffer,
        tokens: usize,
        heads: usize,
        hd: usize,
        eps: f32,
    ) {
        // Per-head RMS norm in-place: reshape [tokens, heads, hd] as
        // `tokens * heads` independent rows of length `hd`. The rms_norm_f32
        // shader does exactly one normalized row per threadgroup, so this is
        // the same kernel, just with a larger row count.
        let cmd = ctx.cmd();
        let enc = cmd.new_compute_command_encoder();
        st().pipes
            .rms_norm_enc(enc, data, w, data, tokens * heads, hd, eps);
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

    fn rope(
        _ctx: &mut Self::Context,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        positions: &[u32],
        nh: usize,
        nkv: usize,
        hd: usize,
    ) {
        let tokens = positions.len();
        let half = hd / 2;
        let cv = Self::to_vec(cos, cos.length() as usize / 4);
        let sv = Self::to_vec(sin, sin.length() as usize / 4);
        unsafe {
            rope_cpu(
                std::slice::from_raw_parts_mut(q.contents() as *mut f32, tokens * nh * hd),
                tokens,
                nh,
                hd,
                half,
                &cv,
                &sv,
                positions,
            );
            rope_cpu(
                std::slice::from_raw_parts_mut(k.contents() as *mut f32, tokens * nkv * hd),
                tokens,
                nkv,
                hd,
                half,
                &cv,
                &sv,
                positions,
            );
        }
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

    fn kv_cache_append(
        _ctx: &mut Self::Context,
        ck: &mut Self::Buffer,
        cv: &mut Self::Buffer,
        cl: usize,
        nk: &Self::Buffer,
        nv: &Self::Buffer,
        nt: usize,
        nkv: usize,
        hd: usize,
    ) -> (Self::Buffer, Self::Buffer) {
        let nl = cl + nt;
        let fk = st().pipes.buffer_empty(nkv * nl * hd);
        let fv = st().pipes.buffer_empty(nkv * nl * hd);
        unsafe {
            let ok = std::slice::from_raw_parts(ck.contents() as *const f32, nkv * cl * hd);
            let ov = std::slice::from_raw_parts(cv.contents() as *const f32, nkv * cl * hd);
            let nkd = std::slice::from_raw_parts(nk.contents() as *const f32, nt * nkv * hd);
            let nvd = std::slice::from_raw_parts(nv.contents() as *const f32, nt * nkv * hd);
            let fko = std::slice::from_raw_parts_mut(fk.contents() as *mut f32, nkv * nl * hd);
            let fvo = std::slice::from_raw_parts_mut(fv.contents() as *mut f32, nkv * nl * hd);
            for h in 0..nkv {
                if cl > 0 {
                    fko[h * nl * hd..h * nl * hd + cl * hd]
                        .copy_from_slice(&ok[h * cl * hd..(h + 1) * cl * hd]);
                    fvo[h * nl * hd..h * nl * hd + cl * hd]
                        .copy_from_slice(&ov[h * cl * hd..(h + 1) * cl * hd]);
                }
                for t in 0..nt {
                    let s = t * nkv * hd + h * hd;
                    let d = h * nl * hd + (cl + t) * hd;
                    fko[d..d + hd].copy_from_slice(&nkd[s..s + hd]);
                    fvo[d..d + hd].copy_from_slice(&nvd[s..s + hd]);
                }
            }
        }
        (fk, fv)
    }

    fn transpose_token_to_head(
        _ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        unsafe {
            let s = std::slice::from_raw_parts(src.contents() as *const f32, tokens * heads * dim);
            let d =
                std::slice::from_raw_parts_mut(dst.contents() as *mut f32, heads * tokens * dim);
            for t in 0..tokens {
                for h in 0..heads {
                    d[(h * tokens + t) * dim..(h * tokens + t) * dim + dim]
                        .copy_from_slice(&s[(t * heads + h) * dim..(t * heads + h) * dim + dim]);
                }
            }
        }
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

    fn layer_forward_fused(
        ctx: &mut Self::Context,
        cfg: &super::TransformerConfig,
        weights: &super::LayerWeights<Self>,
        kv: &mut super::KvCache<Self>,
        scratch: &mut super::LayerScratch<Self>,
        residual: &mut Self::Buffer,
        positions: &[u32],
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        tokens: usize,
    ) {
        // All-Metal layer forward. Encoders are appended to the caller's command buffer
        // (via `ctx.cmd()`), so an entire forward pass — embedding + N × layer_forward
        // + final norm + lm_head — can be bundled into a single commit+wait.
        //
        // Pipeline: rms_norm → qkv GEMM → split → qk_norm+RoPE+transpose → kv_append
        //           → flash_attn → untranspose → o_proj GEMM → fused residual+post_norm
        //           → gate_up GEMM → silu_mul_split → down GEMM → residual add
        //
        // KV buffers are pre-allocated to cfg.max_seq_len on first call and reused thereafter.

        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let qkv_dim = q_dim + 2 * kv_dim;

        let pipes = &st().pipes;

        // Ensure KV cache is pre-allocated to max_seq_len.
        // First call: kv.capacity == 0 — alloc full-size buffers.
        let max_len = cfg.max_seq_len;
        if kv.capacity == 0 {
            kv.k = pipes.buffer_empty(nkv * max_len * hd);
            kv.v = pipes.buffer_empty(nkv * max_len * hd);
            kv.capacity = max_len;
        }
        debug_assert!(
            kv.len + tokens <= kv.capacity,
            "kv overflow: len={}, new={}, cap={}",
            kv.len,
            tokens,
            kv.capacity
        );

        let cmd = ctx.cmd();

        // GEMM dispatcher: m=1 (decode) uses gemv; m>1 (prefill) uses tiled gemm_v2.
        // Closure lets us pick implementation once at layer entry.
        let gemm = |enc: &metal::ComputeCommandEncoderRef,
                    a: &metal::Buffer,
                    b: &metal::Buffer,
                    c: &metal::Buffer,
                    m: usize,
                    n: usize,
                    k: usize| {
            if m == 1 {
                pipes.gemv_enc(enc, a, b, c, n, k);
            } else {
                pipes.gemm_v2(enc, a, b, c, m, n, k);
            }
        };

        // ── Prelude encoder: rms_norm → QKV gemm → split → qk_norm_rope × 3 → kv_append × 2 ──
        // Metal guarantees commands in a single compute encoder are ordered and sees
        // writes from earlier dispatches when later dispatches read the same resource,
        // so merging saves ~11 encoder end/create pairs per layer vs one-encoder-per-op.
        let pos_offset = positions[0] as usize;
        let q_mode: i32 = if cfg.has_qk_norm { 1 } else { 2 };
        let (q_norm_w, k_norm_w) =
            (weights.q_norm_w.as_ref(), weights.k_norm_w.as_ref());
        // Dummy weight buffer for mode-0/2 paths (kernel requires a buffer even when unused).
        let dummy = &weights.input_ln_w;
        let q_norm_buf = q_norm_w.unwrap_or(dummy);
        let k_norm_buf = k_norm_w.unwrap_or(dummy);

        {
            let enc = cmd.new_compute_command_encoder();

            // 1. Input RMSNorm
            pipes.rms_norm_enc(
                enc,
                residual,
                &weights.input_ln_w,
                &scratch.norm_out,
                tokens,
                h,
                eps,
            );

            // 2. Fused QKV GEMM
            gemm(
                enc,
                &scratch.norm_out,
                &weights.qkv_proj_w,
                &scratch.qkv_out,
                tokens,
                qkv_dim,
                h,
            );

            // 3. Split fused qkv → q, k, v (token-major)
            pipes.split_qkv_enc(
                enc,
                &scratch.qkv_out,
                &scratch.q_buf,
                &scratch.k_buf,
                &scratch.v_buf,
                tokens,
                q_dim,
                kv_dim,
            );

            // 4. QK-norm + RoPE + transpose to head-major.
            //    Q → attn_out; K → o_proj_out (staging); V → gate_up_out (staging).
            pipes.qk_norm_rope(
                enc,
                &scratch.q_buf,
                q_norm_buf,
                cos,
                sin,
                &scratch.attn_out,
                tokens,
                nh,
                hd,
                pos_offset,
                eps,
                q_mode,
            );
            pipes.qk_norm_rope(
                enc,
                &scratch.k_buf,
                k_norm_buf,
                cos,
                sin,
                &scratch.o_proj_out,
                tokens,
                nkv,
                hd,
                pos_offset,
                eps,
                q_mode,
            );
            pipes.qk_norm_rope(
                enc,
                &scratch.v_buf,
                k_norm_buf,
                cos,
                sin,
                &scratch.gate_up_out,
                tokens,
                nkv,
                hd,
                pos_offset,
                eps,
                0,
            );

            // 5. Append K/V into pre-allocated cache
            pipes.kv_cache_append(
                enc,
                &scratch.o_proj_out,
                &kv.k,
                nkv,
                hd,
                kv.len,
                tokens,
                kv.capacity,
            );
            pipes.kv_cache_append(
                enc,
                &scratch.gate_up_out,
                &kv.v,
                nkv,
                hd,
                kv.len,
                tokens,
                kv.capacity,
            );

            enc.end_encoding();
        }
        let kv_len = kv.len + tokens;
        kv.len = kv_len;

        // 6. Flash attention with strided KV (kv_seq_stride = max_len so cache reads skip unused slots)
        {
            let params = AttentionParams {
                batch: 1,
                num_heads: nh,
                num_kv_heads: nkv,
                q_len: tokens,
                kv_len,
                head_dim: hd,
                causal: tokens > 1,
                pos_offset,
            };
            pipes.flash_attn_v2(
                cmd,
                &scratch.attn_out, // head-major Q
                &kv.k,
                &kv.v,
                &scratch.q_buf, // reuse q_buf for head-major attn output
                &params,
                kv.capacity,
            );
        }

        // ── Postlude encoder: transpose → O-proj → residual+norm → gate_up → silu → down → add ──
        // Again a single encoder per Metal's in-order guarantee.
        {
            let enc = cmd.new_compute_command_encoder();

            // 7. Untranspose: [nh, tokens, hd] → [tokens, nh, hd]
            pipes.transpose_out(enc, &scratch.q_buf, &scratch.attn_out, tokens, nh, hd);

            // 8. O-projection GEMM
            gemm(
                enc,
                &scratch.attn_out,
                &weights.o_proj_w,
                &scratch.o_proj_out,
                tokens,
                h,
                q_dim,
            );

            // 9. Fused residual + post-attention RMSNorm
            pipes.fused_residual_norm_enc(
                enc,
                residual,
                &scratch.o_proj_out,
                None,
                &weights.post_ln_w,
                residual,
                &scratch.norm_out,
                tokens,
                h,
                eps,
                0,
            );

            // 10. Gate+Up GEMM (fused)
            gemm(
                enc,
                &scratch.norm_out,
                &weights.gate_up_proj_w,
                &scratch.gate_up_out,
                tokens,
                2 * im,
                h,
            );

            // 11. SiLU(gate) * up
            pipes.silu_mul_split_enc(enc, &scratch.gate_up_out, &scratch.silu_out, tokens, im);

            // 12. Down GEMM
            gemm(
                enc,
                &scratch.silu_out,
                &weights.down_proj_w,
                &scratch.mlp_out,
                tokens,
                h,
                im,
            );

            // 13. Final residual add
            pipes.add_enc(enc, residual, &scratch.mlp_out, residual, tokens * h);

            enc.end_encoding();
        }

        // No commit here — caller (ModelRunner / B::sync) is responsible for flushing
        // the accumulated command buffer after all layers + final norm + lm_head GEMM.
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

fn rope_cpu(
    data: &mut [f32],
    tokens: usize,
    heads: usize,
    hd: usize,
    half: usize,
    cos: &[f32],
    sin: &[f32],
    pos: &[u32],
) {
    for t in 0..tokens {
        let p = pos[t] as usize;
        for h in 0..heads {
            let b = t * heads * hd + h * hd;
            for i in 0..half {
                let c = cos[p * half + i];
                let s = sin[p * half + i];
                let x0 = data[b + i];
                let x1 = data[b + half + i];
                data[b + i] = x0 * c - x1 * s;
                data[b + half + i] = x1 * c + x0 * s;
            }
        }
    }
}
