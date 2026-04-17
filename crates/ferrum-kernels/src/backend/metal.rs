//! Metal backend — pipelined command buffer with manual retain/release.
//!
//! Pattern from llama.cpp: [queue commandBuffer] + [cmd_buf retain].
//! One command buffer per forward pass; GPU ops encode into it; sync() commits.

use super::{AttnConfig, Backend};
use ferrum_attention::metal::pipelines::MetalPipelines;
use ferrum_attention::AttentionParams;
use metal::{CommandBufferRef, Device};
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

/// Metal context — empty marker. Command buffers are function-local (not stored).
pub struct MetalContext;

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
        MetalContext
    }
    fn sync(_ctx: &mut Self::Context) {}

    fn gemm(
        _ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
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

    fn rms_norm(
        _ctx: &mut Self::Context,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        run(|cmd| st().pipes.rms_norm(cmd, x, w, out, tokens, dim, eps));
    }

    fn fused_add_rms_norm(
        _ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        run(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            st().pipes.fused_residual_norm_enc(
                enc, residual, x, None, w, residual, out, tokens, dim, eps, 0,
            );
            enc.end_encoding();
        });
    }

    fn decode_attention(
        _ctx: &mut Self::Context,
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
        run(|cmd| st().pipes.flash_attn(cmd, q, kc, vc, out, &p));
    }

    fn flash_attention(
        _ctx: &mut Self::Context,
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
        run(|cmd| st().pipes.flash_attn(cmd, q, k, v, out, &p));
    }

    fn silu_mul(
        _ctx: &mut Self::Context,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    ) {
        run(|cmd| st().pipes.silu_mul(cmd, gate, up, out, len));
    }

    fn add(
        _ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    ) {
        run(|cmd| st().pipes.add(cmd, a, b, out, len));
    }

    fn copy(_ctx: &mut Self::Context, src: &Self::Buffer, dst: &mut Self::Buffer, len: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.contents() as *const u8,
                dst.contents() as *mut u8,
                len * 4,
            );
        }
    }

    fn embedding_lookup(
        _ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    ) {
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

    // ── CPU ops on shared memory: flush GPU first ────────────────────────

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
        let qd = q_dim + 2 * kv_dim;
        unsafe {
            let s = std::slice::from_raw_parts(qkv.contents() as *const f32, tokens * qd);
            let qo = std::slice::from_raw_parts_mut(q.contents() as *mut f32, tokens * q_dim);
            let ko = std::slice::from_raw_parts_mut(k.contents() as *mut f32, tokens * kv_dim);
            let vo = std::slice::from_raw_parts_mut(v.contents() as *mut f32, tokens * kv_dim);
            for t in 0..tokens {
                let b = t * qd;
                qo[t * q_dim..(t + 1) * q_dim].copy_from_slice(&s[b..b + q_dim]);
                ko[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&s[b + q_dim..b + q_dim + kv_dim]);
                vo[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&s[b + q_dim + kv_dim..b + qd]);
            }
        }
    }

    fn fused_silu_mul_split(
        ctx: &mut Self::Context,
        gu: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        unsafe {
            let s = std::slice::from_raw_parts(gu.contents() as *const f32, tokens * 2 * im);
            let d = std::slice::from_raw_parts_mut(out.contents() as *mut f32, tokens * im);
            for t in 0..tokens {
                for i in 0..im {
                    let g = s[t * 2 * im + i];
                    let u = s[t * 2 * im + im + i];
                    d[t * im + i] = (g / (1.0 + (-g).exp())) * u;
                }
            }
        }
    }

    fn qk_norm(
        _ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        w: &Self::Buffer,
        tokens: usize,
        heads: usize,
        hd: usize,
        eps: f32,
    ) {
        unsafe {
            let d =
                std::slice::from_raw_parts_mut(data.contents() as *mut f32, tokens * heads * hd);
            let wv = std::slice::from_raw_parts(w.contents() as *const f32, hd);
            for t in 0..tokens {
                for h in 0..heads {
                    let o = (t * heads + h) * hd;
                    let mut ss = 0.0f32;
                    for i in 0..hd {
                        ss += d[o + i] * d[o + i];
                    }
                    let inv = 1.0 / (ss / hd as f32 + eps).sqrt();
                    for i in 0..hd {
                        d[o + i] *= inv * wv[i];
                    }
                }
            }
        }
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
        _ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        unsafe {
            let s = std::slice::from_raw_parts(src.contents() as *const f32, heads * tokens * dim);
            let d =
                std::slice::from_raw_parts_mut(dst.contents() as *mut f32, tokens * heads * dim);
            for h in 0..heads {
                for t in 0..tokens {
                    d[(t * heads + h) * dim..(t * heads + h) * dim + dim]
                        .copy_from_slice(&s[(h * tokens + t) * dim..(h * tokens + t) * dim + dim]);
                }
            }
        }
    }

    fn add_inplace(ctx: &mut Self::Context, r: &mut Self::Buffer, x: &Self::Buffer, len: usize) {
        unsafe {
            let rv = std::slice::from_raw_parts_mut(r.contents() as *mut f32, len);
            let xv = std::slice::from_raw_parts(x.contents() as *const f32, len);
            for i in 0..len {
                rv[i] += xv[i];
            }
        }
    }

    fn layer_forward_fused(
        _ctx: &mut Self::Context,
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
        let h = cfg.hidden_size;
        let nh = cfg.num_heads;
        let nkv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let im = cfg.intermediate_size;
        let eps = cfg.rms_norm_eps;
        let q_dim = nh * hd;
        let kv_dim = nkv * hd;
        let qkv_dim = q_dim + 2 * kv_dim;
        let s = st();
        let mut mc = MetalContext;

        // --- CMD 1: rms_norm ---
        let cmd = s.pipes.queue.new_command_buffer();
        s.pipes.rms_norm(
            cmd,
            residual,
            &weights.input_ln_w,
            &scratch.norm_out,
            tokens,
            h,
            eps,
        );
        cmd.commit();
        cmd.wait_until_completed();

        // --- GEMM (cblas): QKV ---
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                tokens as i32,
                qkv_dim as i32,
                h as i32,
                1.0,
                scratch.norm_out.contents() as *const f32,
                h as i32,
                weights.qkv_proj_w.contents() as *const f32,
                h as i32,
                0.0,
                scratch.qkv_out.contents() as *mut f32,
                qkv_dim as i32,
            );
        }

        // --- CPU: split, qk_norm, rope, kv_append ---
        Self::split_qkv(
            &mut mc,
            &scratch.qkv_out,
            &mut scratch.q_buf,
            &mut scratch.k_buf,
            &mut scratch.v_buf,
            tokens,
            q_dim,
            kv_dim,
        );
        if cfg.has_qk_norm {
            if let Some(ref w) = weights.q_norm_w {
                Self::qk_norm(&mut mc, &mut scratch.q_buf, w, tokens, nh, hd, eps);
            }
            if let Some(ref w) = weights.k_norm_w {
                Self::qk_norm(&mut mc, &mut scratch.k_buf, w, tokens, nkv, hd, eps);
            }
        }
        Self::rope(
            &mut mc,
            &mut scratch.q_buf,
            &mut scratch.k_buf,
            cos,
            sin,
            positions,
            nh,
            nkv,
            hd,
        );
        let (nk, nv) = Self::kv_cache_append(
            &mut mc,
            &mut kv.k,
            &mut kv.v,
            kv.len,
            &scratch.k_buf,
            &scratch.v_buf,
            tokens,
            nkv,
            hd,
        );
        kv.k = nk;
        kv.v = nv;
        kv.len += tokens;

        // --- CMD 2: attention ---
        let cmd = s.pipes.queue.new_command_buffer();
        if tokens == 1 {
            let p = AttentionParams {
                batch: 1,
                num_heads: nh,
                num_kv_heads: nkv,
                q_len: 1,
                kv_len: kv.len,
                head_dim: hd,
                causal: false,
                pos_offset: 0,
            };
            s.pipes
                .flash_attn(cmd, &scratch.q_buf, &kv.k, &kv.v, &scratch.attn_out, &p);
        } else {
            Self::transpose_token_to_head(
                &mut mc,
                &scratch.q_buf,
                &mut scratch.o_proj_out,
                tokens,
                nh,
                hd,
            );
            let p = AttentionParams {
                batch: 1,
                num_heads: nh,
                num_kv_heads: nkv,
                q_len: tokens,
                kv_len: kv.len,
                head_dim: hd,
                causal: true,
                pos_offset: positions[0] as usize,
            };
            s.pipes.flash_attn(
                cmd,
                &scratch.o_proj_out,
                &kv.k,
                &kv.v,
                &scratch.attn_out,
                &p,
            );
        }
        cmd.commit();
        cmd.wait_until_completed();
        if tokens > 1 {
            let mut temp = Self::alloc(tokens * q_dim);
            Self::transpose_head_to_token(&mut mc, &scratch.attn_out, &mut temp, tokens, nh, hd);
            Self::copy(&mut mc, &temp, &mut scratch.attn_out, tokens * q_dim);
        }

        // --- GEMM (cblas): O-proj ---
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                tokens as i32,
                h as i32,
                q_dim as i32,
                1.0,
                scratch.attn_out.contents() as *const f32,
                q_dim as i32,
                weights.o_proj_w.contents() as *const f32,
                q_dim as i32,
                0.0,
                scratch.o_proj_out.contents() as *mut f32,
                h as i32,
            );
        }

        // --- CMD 3: fused_add_rms_norm ---
        let cmd = s.pipes.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        s.pipes.fused_residual_norm_enc(
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
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // --- GEMM (cblas): gate_up ---
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                tokens as i32,
                (2 * im) as i32,
                h as i32,
                1.0,
                scratch.norm_out.contents() as *const f32,
                h as i32,
                weights.gate_up_proj_w.contents() as *const f32,
                h as i32,
                0.0,
                scratch.gate_up_out.contents() as *mut f32,
                (2 * im) as i32,
            );
        }

        // --- CPU: silu_mul_split ---
        Self::fused_silu_mul_split(
            &mut mc,
            &scratch.gate_up_out,
            &mut scratch.silu_out,
            tokens,
            im,
        );

        // --- GEMM (cblas): down ---
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                tokens as i32,
                h as i32,
                im as i32,
                1.0,
                scratch.silu_out.contents() as *const f32,
                im as i32,
                weights.down_proj_w.contents() as *const f32,
                im as i32,
                0.0,
                scratch.mlp_out.contents() as *mut f32,
                h as i32,
            );
        }

        // --- CPU: add_inplace ---
        Self::add_inplace(&mut mc, residual, &scratch.mlp_out, tokens * h);
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
