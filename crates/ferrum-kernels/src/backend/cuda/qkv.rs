//! Instance-owned token-major scratch for the existing split/norm/RoPE chain.

use super::{ptx, Backend, CudaBackend, CudaState};
use crate::backend::CudaBuf;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use ferrum_types::{FerrumError, Result};
use std::sync::Arc;

#[derive(Default)]
pub(super) struct QkvScratch {
    q: Option<CudaBuf>,
    k: Option<CudaBuf>,
    v: Option<CudaBuf>,
}

fn prepare(ctx: &mut CudaState, q_len: usize, kv_len: usize) {
    let grow_q = ctx
        .qkv_scratch
        .q
        .as_ref()
        .is_none_or(|buf| buf.len() < q_len);
    let grow_k = ctx
        .qkv_scratch
        .k
        .as_ref()
        .is_none_or(|buf| buf.len() < kv_len);
    let grow_v = ctx
        .qkv_scratch
        .v
        .as_ref()
        .is_none_or(|buf| buf.len() < kv_len);
    if !grow_q && !grow_k && !grow_v {
        return;
    }
    if ctx.capture_in_flight {
        // This cannot be a recoverable "unsupported fused kernel" result:
        // callers may otherwise select another route inside the same capture.
        std::panic::panic_any(FerrumError::backend(
            "CUDA QKV scratch must be prepared before graph capture",
        ));
    }
    // Publish each allocated owner immediately, before another allocation or
    // any launch can fail. Old scratch may still be referenced by a cached
    // graph, so use the original context's capture closure until retirement.
    for (grow, len, slot) in [(grow_q, q_len, 0), (grow_k, kv_len, 1), (grow_v, kv_len, 2)] {
        if !grow {
            continue;
        }
        let buffer = CudaBackend::alloc_in_context(ctx, len);
        let previous = match slot {
            0 => ctx.qkv_scratch.q.replace(buffer),
            1 => ctx.qkv_scratch.k.replace(buffer),
            _ => ctx.qkv_scratch.v.replace(buffer),
        };
        if let Some(previous) = previous {
            CudaBackend::retain_buffer_until_completion(ctx, previous);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn split_qkv_norm_rope(
    ctx: &mut CudaState,
    qkv: &CudaBuf,
    q_norm_w: &CudaBuf,
    k_norm_w: &CudaBuf,
    cos: &CudaBuf,
    sin: &CudaBuf,
    q_out: &mut CudaBuf,
    k_out: &mut CudaBuf,
    v_out: &mut CudaBuf,
    tokens: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pos_offset: usize,
    eps: f32,
    qk_mode: i32,
) -> Result<()> {
    let q_dim = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| FerrumError::model("QKV query size overflow"))?;
    let kv_dim = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| FerrumError::model("QKV key/value size overflow"))?;
    let q_len = tokens
        .checked_mul(q_dim)
        .ok_or_else(|| FerrumError::model("QKV query scratch size overflow"))?;
    let kv_len = tokens
        .checked_mul(kv_dim)
        .ok_or_else(|| FerrumError::model("QKV key/value scratch size overflow"))?;
    prepare(ctx, q_len, kv_len);
    let split = ctx.func("split_qkv", ptx::SPLIT_QKV, "split_qkv_f16");
    let use_dyn = ctx.use_dev_state && tokens == 1;
    let norm = ctx.func(
        "qk_norm_rope",
        ptx::QK_NORM_ROPE,
        if use_dyn {
            "qk_norm_rope_transpose_f16_dyn"
        } else {
            "qk_norm_rope_transpose_f16"
        },
    );
    // Borrow fields in place: a launch panic cannot move scratch into a
    // temporary that unwinds before the frame's completion observation.
    let QkvScratch { q, k, v } = &mut ctx.qkv_scratch;
    let q = q.as_mut().expect("prepared Q scratch");
    let k = k.as_mut().expect("prepared K scratch");
    let v = v.as_mut().expect("prepared V scratch");
    launch_split_qkv(&ctx.stream, &split, qkv, q, k, v, tokens, q_dim, kv_dim);
    let position = use_dyn.then_some(&ctx.decode_state.pos);
    launch_qk_norm_rope(
        &ctx.stream,
        &norm,
        position,
        q,
        q_norm_w,
        cos,
        sin,
        q_out,
        tokens,
        q_heads,
        head_dim,
        pos_offset,
        eps,
        qk_mode,
    );
    launch_qk_norm_rope(
        &ctx.stream,
        &norm,
        position,
        k,
        k_norm_w,
        cos,
        sin,
        k_out,
        tokens,
        kv_heads,
        head_dim,
        pos_offset,
        eps,
        qk_mode,
    );
    // Mode zero copies/transposes V; the weight and trigonometric inputs are
    // ignored by the existing kernel, exactly as in the original chain.
    launch_qk_norm_rope(
        &ctx.stream,
        &norm,
        position,
        v,
        q_norm_w,
        cos,
        sin,
        v_out,
        tokens,
        kv_heads,
        head_dim,
        pos_offset,
        eps,
        0,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_split_qkv(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    qkv: &CudaBuf,
    q: &mut CudaBuf,
    k: &mut CudaBuf,
    v: &mut CudaBuf,
    tokens: usize,
    q_dim: usize,
    kv_dim: usize,
) {
    let tokens_i32 = tokens as i32;
    let q_dim_i32 = q_dim as i32;
    let kv_dim_i32 = kv_dim as i32;
    let mut b = stream.launch_builder(func);
    b.arg(qkv);
    b.arg(q);
    b.arg(k);
    b.arg(v);
    b.arg(&tokens_i32);
    b.arg(&q_dim_i32);
    b.arg(&kv_dim_i32);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .expect("split_qkv launch");
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_qk_norm_rope(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    position: Option<&CudaSlice<i32>>,
    input: &CudaBuf,
    norm_w: &CudaBuf,
    cos: &CudaBuf,
    sin: &CudaBuf,
    output: &mut CudaBuf,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    pos_offset: usize,
    eps: f32,
    mode: i32,
) {
    let tokens_i32 = tokens as i32;
    let heads_i32 = heads as i32;
    let head_dim_i32 = head_dim as i32;
    let pos_offset_i32 = pos_offset as i32;
    let mut b = stream.launch_builder(func);
    b.arg(input);
    b.arg(norm_w);
    b.arg(cos);
    b.arg(sin);
    b.arg(output);
    b.arg(&tokens_i32);
    b.arg(&heads_i32);
    b.arg(&head_dim_i32);
    if let Some(position) = position {
        b.arg(position);
    } else {
        b.arg(&pos_offset_i32);
    }
    b.arg(&eps);
    b.arg(&mode);
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (tokens as u32, heads as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .expect("qk_norm_rope launch");
}

#[cfg(test)]
mod tests;
