//! Metal backend using ferrum-attention's MetalPipelines.
//!
//! Buffer type: `metal::Buffer` (shared memory = zero-copy on Apple Silicon).
//! GEMM uses cblas_sgemm on shared buffer contents (no GPU dispatch needed).
//! Other ops use Metal compute shaders via MetalPipelines.

use super::{AttnConfig, Backend};
use ferrum_attention::metal::pipelines::MetalPipelines;
use ferrum_attention::AttentionParams;
use metal::{Device, MTLResourceOptions};
use std::ffi::c_void;
use std::sync::OnceLock;

/// Lazily initialized Metal state (device + compiled pipelines).
struct MetalState {
    pipes: MetalPipelines,
}

static METAL_STATE: OnceLock<MetalState> = OnceLock::new();

fn get_state() -> &'static MetalState {
    METAL_STATE.get_or_init(|| {
        let device = Device::system_default().expect("no Metal device");
        let pipes = MetalPipelines::new(&device);
        MetalState { pipes }
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

impl Backend for MetalBackend {
    type Buffer = metal::Buffer;

    fn gemm(
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        // Apple Silicon shared memory: cblas directly on buffer contents (zero-copy)
        unsafe {
            cblas_sgemm(
                101,
                111,
                112, // RowMajor, NoTrans, Trans
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
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        _dim: usize,
    ) {
        let st = get_state();
        let cmd = st.pipes.queue.new_command_buffer();
        st.pipes.rms_norm(cmd, x, w, out, tokens, _dim, eps);
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn fused_add_rms_norm(
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let st = get_state();
        let cmd = st.pipes.queue.new_command_buffer();
        // Use fused_residual_norm: residual = residual + x, out = rms_norm(residual) * w
        let enc = cmd.new_compute_command_encoder();
        st.pipes.fused_residual_norm_enc(
            enc, residual, x, None, // no layer_scale
            w, residual, // out_res = updated residual (in-place)
            out,      // out_norm
            tokens, dim, eps, 0, // scale_len unused
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn rope(
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        positions: &[u32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        // Metal RoPE is fused into qk_norm_rope_transpose, which expects a different layout.
        // For now, fall back to CPU RoPE on shared buffer contents (zero-copy).
        let tokens = positions.len();
        let half = head_dim / 2;
        let cos_vec = Self::to_vec(cos, cos.length() as usize / 4);
        let sin_vec = Self::to_vec(sin, sin.length() as usize / 4);

        unsafe {
            let q_ptr = q.contents() as *mut f32;
            let q_slice = std::slice::from_raw_parts_mut(q_ptr, tokens * num_heads * head_dim);
            apply_rope_cpu(
                q_slice, tokens, num_heads, head_dim, half, &cos_vec, &sin_vec, positions,
            );

            let k_ptr = k.contents() as *mut f32;
            let k_slice = std::slice::from_raw_parts_mut(k_ptr, tokens * num_kv_heads * head_dim);
            apply_rope_cpu(
                k_slice,
                tokens,
                num_kv_heads,
                head_dim,
                half,
                &cos_vec,
                &sin_vec,
                positions,
            );
        }
    }

    fn decode_attention(
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        out: &mut Self::Buffer,
        kv_len: usize,
        cfg: &AttnConfig,
    ) {
        let st = get_state();
        let params = AttentionParams {
            batch: 1,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            q_len: 1,
            kv_len,
            head_dim: cfg.head_dim,
            causal: false, // decode: single token, no mask needed
            pos_offset: 0,
        };
        let cmd = st.pipes.queue.new_command_buffer();
        st.pipes.flash_attn(cmd, q, k_cache, v_cache, out, &params);
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn flash_attention(
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
        let st = get_state();
        let params = AttentionParams {
            batch,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            q_len,
            kv_len,
            head_dim: cfg.head_dim,
            causal: cfg.causal,
            pos_offset,
        };
        let cmd = st.pipes.queue.new_command_buffer();
        st.pipes.flash_attn(cmd, q, k, v, out, &params);
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn silu_mul(gate: &Self::Buffer, up: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        let st = get_state();
        let cmd = st.pipes.queue.new_command_buffer();
        st.pipes.silu_mul(cmd, gate, up, out, len);
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn add(a: &Self::Buffer, b: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        let st = get_state();
        let cmd = st.pipes.queue.new_command_buffer();
        st.pipes.add(cmd, a, b, out, len);
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn copy(src: &Self::Buffer, dst: &mut Self::Buffer, len: usize) {
        let bytes = len * 4;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.contents() as *const u8,
                dst.contents() as *mut u8,
                bytes,
            );
        }
    }

    fn embedding_lookup(table: &Self::Buffer, ids: &[u32], out: &mut Self::Buffer, dim: usize) {
        // CPU gather on shared memory (zero-copy)
        let tbl = unsafe {
            std::slice::from_raw_parts(table.contents() as *const f32, table.length() as usize / 4)
        };
        let o =
            unsafe { std::slice::from_raw_parts_mut(out.contents() as *mut f32, ids.len() * dim) };
        for (i, &id) in ids.iter().enumerate() {
            let src = id as usize * dim;
            o[i * dim..(i + 1) * dim].copy_from_slice(&tbl[src..src + dim]);
        }
    }

    fn alloc(len: usize) -> Self::Buffer {
        let st = get_state();
        st.pipes.buffer_empty(len)
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        MetalPipelines::read_buffer(buf, len)
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        let st = get_state();
        st.pipes.buffer_from_data(data)
    }
}

// CPU RoPE fallback on shared memory contents
fn apply_rope_cpu(
    data: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    half: usize,
    cos: &[f32],
    sin: &[f32],
    positions: &[u32],
) {
    for t in 0..tokens {
        let pos = positions[t] as usize;
        for h in 0..heads {
            let base = t * heads * head_dim + h * head_dim;
            for i in 0..half {
                let c = cos[pos * half + i];
                let s = sin[pos * half + i];
                let x0 = data[base + i];
                let x1 = data[base + half + i];
                data[base + i] = x0 * c - x1 * s;
                data[base + half + i] = x1 * c + x0 * s;
            }
        }
    }
}
