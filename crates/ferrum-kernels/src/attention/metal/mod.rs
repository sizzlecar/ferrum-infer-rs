//! Metal flash attention through an owned execution context.

pub(crate) mod pipelines;

#[cfg(test)]
mod pipeline_tests;

use crate::attention::AttentionParams;
use crate::backend::metal::{MetalBackend, MetalContext};
use crate::backend::Backend;
use ferrum_types::{BackendContextConfig, Result};
use metal::{Buffer, Device};

pub fn is_available() -> bool {
    Device::system_default().is_some()
}

/// Run the scalar attention kernel on the original context and require a
/// successful completion before returning. Buffers belong to that context's
/// device; any producer on another context must already have completed.
fn fused_attention_metal_buffers(
    ctx: &mut MetalContext,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    o_buf: &Buffer,
    p: &AttentionParams,
) -> Result<()> {
    {
        let (pipes, cmd, _) = ctx.command_parts();
        pipes.flash_attn_legacy(cmd, q_buf, k_buf, v_buf, o_buf, p);
    }
    MetalBackend::sync_checked(ctx)
}

/// One-shot scalar attention from f32 slices, including upload and readback.
/// The execution context owns its queue and shares only the immutable device
/// pipeline catalog. Model inference uses its existing long-lived context.
pub fn fused_attention(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], p: &AttentionParams) {
    // The one-shot call returns only host values. Drain autoreleased command
    // objects here so repeated calls do not retain their independent queues
    // until a caller-owned pool happens to drain.
    metal::objc::rc::autoreleasepool(|| {
        let mut ctx = MetalBackend::new_context(&BackendContextConfig::default())
            .expect("create Metal attention context");
        let q_buf = MetalBackend::from_slice_in_context(&mut ctx, q);
        let k_buf = MetalBackend::from_slice_in_context(&mut ctx, k);
        let v_buf = MetalBackend::from_slice_in_context(&mut ctx, v);
        let o_buf = MetalBackend::alloc_in_context(&mut ctx, out.len());

        fused_attention_metal_buffers(
            &mut ctx,
            q_buf.raw(),
            k_buf.raw(),
            v_buf.raw(),
            o_buf.raw(),
            p,
        )
        .expect("complete Metal attention");

        // The checked original-context completion above precedes every host read.
        unsafe {
            std::ptr::copy_nonoverlapping(
                o_buf.raw().contents() as *const f32,
                out.as_mut_ptr(),
                out.len(),
            );
        }
    });
}
