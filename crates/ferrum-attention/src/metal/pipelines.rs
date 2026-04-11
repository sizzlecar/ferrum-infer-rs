//! Metal compute pipeline cache and dispatch helpers.

use metal::*;
use std::collections::HashMap;
use std::ffi::c_void;

pub struct MetalPipelines {
    pub device: Device,
    pub queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl MetalPipelines {
    pub fn new(device: &Device) -> Self {
        let queue = device.new_command_queue();

        // Compile shaders
        let opts = CompileOptions::new();
        let fa_src = include_str!("shaders/flash_attn.metal");
        let ops_src = include_str!("shaders/transformer_ops.metal");

        let fa_lib = device.new_library_with_source(fa_src, &opts)
            .expect("failed to compile flash_attn.metal");
        let ops_lib = device.new_library_with_source(ops_src, &opts)
            .expect("failed to compile transformer_ops.metal");

        let mut pipelines = HashMap::new();
        for (lib, names) in [
            (&fa_lib, &["flash_attn_f32"][..]),
            (&ops_lib, &["rms_norm_f32", "silu_mul_f32", "add_f32", "gemm_f32"][..]),
        ] {
            for name in names {
                let func = lib.get_function(name, None)
                    .unwrap_or_else(|e| panic!("kernel {name} not found: {e}"));
                let pso = device.new_compute_pipeline_state_with_function(&func)
                    .unwrap_or_else(|e| panic!("pipeline {name} failed: {e}"));
                pipelines.insert(*name, pso);
            }
        }

        MetalPipelines { device: device.clone(), queue, pipelines }
    }

    pub fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.pipelines.get(name).unwrap_or_else(|| panic!("pipeline {name} not found"))
    }

    /// Create a shared buffer from f32 data.
    pub fn buffer_from_data(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create an empty shared buffer.
    pub fn buffer_empty(&self, num_floats: usize) -> Buffer {
        self.device.new_buffer((num_floats * 4) as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Read f32 data from a buffer.
    pub fn read_buffer(buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    /// Run GEMM: C[M,N] = A[M,K] @ B[N,K]^T
    /// Uses Accelerate cblas_sgemm on shared Metal buffers (zero-copy on Apple Silicon).
    pub fn gemm(&self, _cmd: &CommandBufferRef, a: &Buffer, b: &Buffer, c: &Buffer, m: usize, n: usize, k: usize) {
        extern "C" {
            fn cblas_sgemm(
                order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32,
                alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32,
                beta: f32, c: *mut f32, ldc: i32,
            );
        }
        unsafe {
            cblas_sgemm(
                101, 111, 112, // RowMajor, NoTrans, Trans
                m as i32, n as i32, k as i32,
                1.0,
                a.contents() as *const f32, k as i32,
                b.contents() as *const f32, k as i32,
                0.0,
                c.contents() as *mut f32, n as i32,
            );
        }
    }

    /// Run RMS norm: out = rms_norm(input) * weight
    pub fn rms_norm(&self, cmd: &CommandBufferRef, input: &Buffer, weight: &Buffer, output: &Buffer, rows: usize, dim: usize, eps: f32) {
        #[repr(C)]
        struct P { dim: i32, eps: f32 }
        let params = P { dim: dim as i32, eps };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void, 8, MTLResourceOptions::StorageModeShared);

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipeline("rms_norm_f32"));
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        // Shared memory for cross-simdgroup reduction
        enc.set_threadgroup_memory_length(0, 128); // 32 floats
        let grid = MTLSize::new(rows as u64, 1, 1);
        let tg = MTLSize::new(32, 1, 1); // 1 simdgroup per row
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    /// Run silu_mul: out = silu(gate) * up
    pub fn silu_mul(&self, cmd: &CommandBufferRef, gate: &Buffer, up: &Buffer, output: &Buffer, n: usize) {
        #[repr(C)]
        struct P { n: i32 }
        let params = P { n: n as i32 };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void, 4, MTLResourceOptions::StorageModeShared);

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipeline("silu_mul_f32"));
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        let grid = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    /// Run add: out = a + b
    pub fn add(&self, cmd: &CommandBufferRef, a: &Buffer, b: &Buffer, output: &Buffer, n: usize) {
        #[repr(C)]
        struct P { n: i32 }
        let params = P { n: n as i32 };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void, 4, MTLResourceOptions::StorageModeShared);

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipeline("add_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        let grid = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    /// Run flash attention
    pub fn flash_attn(&self, cmd: &CommandBufferRef,
        q: &Buffer, k: &Buffer, v: &Buffer, o: &Buffer,
        params: &crate::AttentionParams)
    {
        #[repr(C)]
        struct P { batch: i32, num_heads: i32, num_kv_heads: i32, q_len: i32, kv_len: i32, head_dim: i32, scale: f32, causal: i32, pos_offset: i32 }
        let p = P {
            batch: params.batch as i32, num_heads: params.num_heads as i32,
            num_kv_heads: params.num_kv_heads as i32, q_len: params.q_len as i32,
            kv_len: params.kv_len as i32, head_dim: params.head_dim as i32,
            scale: 1.0 / (params.head_dim as f32).sqrt(),
            causal: if params.causal { 1 } else { 0 }, pos_offset: params.pos_offset as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &p as *const _ as *const c_void,
            std::mem::size_of::<P>() as u64,
            MTLResourceOptions::StorageModeShared);

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipeline("flash_attn_f32"));
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k), 0);
        enc.set_buffer(2, Some(v), 0);
        enc.set_buffer(3, Some(o), 0);
        enc.set_buffer(4, Some(&params_buf), 0);
        let grid = MTLSize::new(params.q_len as u64, params.num_heads as u64, params.batch as u64);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }
}
