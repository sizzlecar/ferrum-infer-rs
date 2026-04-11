//! Metal flash attention backend.
//!
//! Fused single-kernel attention on GPU. No intermediate buffers.
//! Accepts raw Metal Buffer references to avoid CPU↔GPU copies.

use crate::AttentionParams;
use metal::{Buffer, CompileOptions, ComputePipelineState, Device, MTLSize, MTLResourceOptions};
use std::ffi::c_void;
use std::sync::OnceLock;

#[repr(C)]
struct FlashAttnParams {
    batch: i32,
    num_heads: i32,
    num_kv_heads: i32,
    q_len: i32,
    kv_len: i32,
    head_dim: i32,
    scale: f32,
    causal: i32,
    pos_offset: i32,
}

struct MetalFlashAttn {
    device: Device,
    pipeline: ComputePipelineState,
    queue: metal::CommandQueue,
}

static INSTANCE: OnceLock<MetalFlashAttn> = OnceLock::new();

fn get_or_init_device(device: &Device) -> &'static MetalFlashAttn {
    INSTANCE.get_or_init(|| {
        let queue = device.new_command_queue();
        let shader_src = include_str!("shaders/flash_attn.metal");
        let library = device.new_library_with_source(shader_src, &CompileOptions::new())
            .expect("failed to compile flash_attn.metal");
        let func = library.get_function("flash_attn_f32", None)
            .expect("flash_attn_f32 not found");
        let pipeline = device.new_compute_pipeline_state_with_function(&func)
            .expect("failed to create pipeline");
        MetalFlashAttn {
            device: device.clone(),
            pipeline,
            queue,
        }
    })
}

pub fn is_available() -> bool {
    Device::system_default().is_some()
}

/// Run fused flash attention directly on Metal buffers.
///
/// q_buf, k_buf, v_buf: existing Metal buffers (from candle tensors)
/// o_buf: pre-allocated output Metal buffer
/// All buffers must be StorageModeShared.
pub fn fused_attention_metal_buffers(
    device: &Device,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    o_buf: &Buffer,
    p: &AttentionParams,
) {
    let ma = get_or_init_device(device);

    let params = FlashAttnParams {
        batch: p.batch as i32,
        num_heads: p.num_heads as i32,
        num_kv_heads: p.num_kv_heads as i32,
        q_len: p.q_len as i32,
        kv_len: p.kv_len as i32,
        head_dim: p.head_dim as i32,
        scale: 1.0 / (p.head_dim as f32).sqrt(),
        causal: if p.causal { 1 } else { 0 },
        pos_offset: p.pos_offset as i32,
    };

    let params_buf = ma.device.new_buffer_with_data(
        &params as *const _ as *const c_void,
        std::mem::size_of::<FlashAttnParams>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = ma.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ma.pipeline);
    enc.set_buffer(0, Some(q_buf), 0);
    enc.set_buffer(1, Some(k_buf), 0);
    enc.set_buffer(2, Some(v_buf), 0);
    enc.set_buffer(3, Some(o_buf), 0);
    enc.set_buffer(4, Some(&params_buf), 0);

    // One threadgroup per (query_pos, head, batch), 32 threads each (1 simdgroup)
    let grid_size = MTLSize::new(p.q_len as u64, p.num_heads as u64, p.batch as u64);
    let tg_size = MTLSize::new(32, 1, 1);
    enc.dispatch_thread_groups(grid_size, tg_size);
    enc.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();
}

/// Convenience: run attention from f32 slices (copies to/from GPU).
/// For integration testing. Production code should use fused_attention_metal_buffers.
pub fn fused_attention(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    p: &AttentionParams,
) {
    let device = Device::system_default().expect("no Metal device");
    let ma = get_or_init_device(&device);

    let mk_buf = |data: &[f32]| -> Buffer {
        ma.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };

    let q_buf = mk_buf(q);
    let k_buf = mk_buf(k);
    let v_buf = mk_buf(v);
    let o_buf = ma.device.new_buffer((out.len() * 4) as u64, MTLResourceOptions::StorageModeShared);

    fused_attention_metal_buffers(&device, &q_buf, &k_buf, &v_buf, &o_buf, p);

    unsafe {
        std::ptr::copy_nonoverlapping(o_buf.contents() as *const f32, out.as_mut_ptr(), out.len());
    }
}
