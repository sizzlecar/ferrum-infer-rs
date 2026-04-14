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
        let gemm_src = include_str!("shaders/gemm_f32.metal");
        let nr_src = include_str!("shaders/norm_rope.metal");
        let sm_src = include_str!("shaders/softmax.metal");

        let fa_lib = device
            .new_library_with_source(fa_src, &opts)
            .expect("failed to compile flash_attn.metal");
        let ops_lib = device
            .new_library_with_source(ops_src, &opts)
            .expect("failed to compile transformer_ops.metal");
        let gemm_lib = device
            .new_library_with_source(gemm_src, &opts)
            .expect("failed to compile gemm_f32.metal");
        let nr_lib = device
            .new_library_with_source(nr_src, &opts)
            .expect("failed to compile norm_rope.metal");
        let sm_lib = device
            .new_library_with_source(sm_src, &opts)
            .expect("failed to compile softmax.metal");

        let mut pipelines = HashMap::new();
        for (lib, names) in [
            (&fa_lib, &["flash_attn_f32"][..]),
            (
                &ops_lib,
                &[
                    "rms_norm_f32",
                    "silu_mul_f32",
                    "add_f32",
                    "mul_scale_f32",
                    "fused_scale_add_f32",
                    "fused_residual_norm_f32",
                    "gemm_f32",
                ][..],
            ),
            (&gemm_lib, &["gemm_f32_v2"][..]),
            (
                &nr_lib,
                &[
                    "qk_norm_rope_transpose_f32",
                    "transpose_out_f32",
                    "kv_cache_append_f32",
                ][..],
            ),
            (
                &sm_lib,
                &["softmax_last_dim_f32", "softmax_last_dim_f32_out"][..],
            ),
        ] {
            for name in names {
                let func = lib
                    .get_function(name, None)
                    .unwrap_or_else(|e| panic!("kernel {name} not found: {e}"));
                let pso = device
                    .new_compute_pipeline_state_with_function(&func)
                    .unwrap_or_else(|e| panic!("pipeline {name} failed: {e}"));
                pipelines.insert(*name, pso);
            }
        }

        MetalPipelines {
            device: device.clone(),
            queue,
            pipelines,
        }
    }

    pub fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.pipelines
            .get(name)
            .unwrap_or_else(|| panic!("pipeline {name} not found"))
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
        self.device.new_buffer(
            (num_floats * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Read f32 data from a buffer.
    pub fn read_buffer(buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    pub fn read_buffer_u32(buf: &Buffer, len: usize) -> Vec<u32> {
        let ptr = buf.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    /// Run GEMM: C[M,N] = A[M,K] @ B[N,K]^T
    /// Uses Accelerate cblas_sgemm on shared Metal buffers (zero-copy on Apple Silicon).
    pub fn gemm(
        &self,
        _cmd: &CommandBufferRef,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
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
                c.contents() as *mut f32,
                n as i32,
            );
        }
    }

    /// RMS norm on an existing encoder (no end_encoding)
    pub fn rms_norm_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        #[repr(C)]
        struct P {
            dim: i32,
            eps: f32,
        }
        let params = P {
            dim: dim as i32,
            eps,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            8,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("rms_norm_f32"));
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        enc.set_threadgroup_memory_length(0, 128);
        let grid = MTLSize::new(rows as u64, 1, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// SiLU×gate on an existing encoder
    pub fn silu_mul_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        gate: &Buffer,
        up: &Buffer,
        output: &Buffer,
        n: usize,
    ) {
        #[repr(C)]
        struct P {
            n: i32,
        }
        let params = P { n: n as i32 };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("silu_mul_f32"));
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        let grid = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Add on an existing encoder
    pub fn add_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        output: &Buffer,
        n: usize,
    ) {
        #[repr(C)]
        struct P {
            n: i32,
        }
        let params = P { n: n as i32 };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("add_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        let grid = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Run RMS norm: out = rms_norm(input) * weight
    pub fn rms_norm(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        #[repr(C)]
        struct P {
            dim: i32,
            eps: f32,
        }
        let params = P {
            dim: dim as i32,
            eps,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            8,
            MTLResourceOptions::StorageModeShared,
        );

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
    pub fn silu_mul(
        &self,
        cmd: &CommandBufferRef,
        gate: &Buffer,
        up: &Buffer,
        output: &Buffer,
        n: usize,
    ) {
        #[repr(C)]
        struct P {
            n: i32,
        }
        let params = P { n: n as i32 };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );

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
        struct P {
            n: i32,
        }
        let params = P { n: n as i32 };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );

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

    // ── New all-Metal dispatch methods (encoder-based, no commit) ──────

    /// GEMM v2: C[M,N] = A[M,K] @ B[N,K]^T — all on Metal, 64x32 tiles
    pub fn gemm_v2(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        #[repr(C)]
        struct P {
            m: i32,
            n: i32,
            k: i32,
        }
        let params = P {
            m: m as i32,
            n: n as i32,
            k: k as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            12,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("gemm_f32_v2"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        enc.set_threadgroup_memory_length(0, 12288);
        let grid = MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
        let tg = MTLSize::new(128, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// QK-norm + RoPE + transpose (fused). Set apply_norm=false for V (transpose only).
    pub fn qk_norm_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &Buffer,
        weight: &Buffer,
        cos: &Buffer,
        sin: &Buffer,
        output: &Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        norm_mode: i32,
    )
    // norm_mode: 0=transpose only (V), 1=norm+RoPE (Q/K with QK-norm), 2=RoPE only (Q/K without QK-norm)
    {
        #[repr(C)]
        struct P {
            tokens: i32,
            heads: i32,
            head_dim: i32,
            half_dim: i32,
            pos_offset: i32,
            eps: f32,
            apply_norm: i32,
        }
        let params = P {
            tokens: tokens as i32,
            heads: heads as i32,
            head_dim: head_dim as i32,
            half_dim: (head_dim / 2) as i32,
            pos_offset: pos_offset as i32,
            eps,
            apply_norm: norm_mode,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            std::mem::size_of::<P>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("qk_norm_rope_transpose_f32"));
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(cos), 0);
        enc.set_buffer(3, Some(sin), 0);
        enc.set_buffer(4, Some(output), 0);
        enc.set_buffer(5, Some(&params_buf), 0);
        let grid = MTLSize::new(tokens as u64, heads as u64, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Untranspose: [heads, tokens, hd] -> [tokens, heads*hd]
    pub fn transpose_out(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) {
        #[repr(C)]
        struct P {
            tokens: i32,
            heads: i32,
            head_dim: i32,
        }
        let params = P {
            tokens: tokens as i32,
            heads: heads as i32,
            head_dim: head_dim as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            12,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("transpose_out_f32"));
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(output), 0);
        enc.set_buffer(2, Some(&params_buf), 0);
        let n = tokens * heads * head_dim;
        let grid = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// KV cache append: copy new data into pre-allocated cache
    pub fn kv_cache_append(
        &self,
        enc: &ComputeCommandEncoderRef,
        new_data: &Buffer,
        cache: &Buffer,
        heads: usize,
        head_dim: usize,
        old_len: usize,
        new_len: usize,
        max_len: usize,
    ) {
        #[repr(C)]
        struct P {
            heads: i32,
            head_dim: i32,
            old_len: i32,
            new_len: i32,
            max_len: i32,
        }
        let params = P {
            heads: heads as i32,
            head_dim: head_dim as i32,
            old_len: old_len as i32,
            new_len: new_len as i32,
            max_len: max_len as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            20,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("kv_cache_append_f32"));
        enc.set_buffer(0, Some(new_data), 0);
        enc.set_buffer(1, Some(cache), 0);
        enc.set_buffer(2, Some(&params_buf), 0);
        let n = heads * new_len * head_dim;
        let grid = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Softmax last dim (in-place): data[rows, cols] → softmax over cols
    pub fn softmax_last_dim_inplace(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &Buffer,
        rows: usize,
        cols: usize,
    ) {
        #[repr(C)]
        struct P {
            rows: i32,
            cols: i32,
        }
        let params = P {
            rows: rows as i32,
            cols: cols as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            8,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("softmax_last_dim_f32"));
        enc.set_buffer(0, Some(data), 0);
        enc.set_buffer(1, Some(&params_buf), 0);
        enc.set_threadgroup_memory_length(0, 128);
        let grid = MTLSize::new(rows as u64, 1, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Element-wise multiply with broadcast scale: out[i] = a[i] * scale[i % scale_len]
    pub fn mul_scale_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        scale: &Buffer,
        output: &Buffer,
        n: usize,
        scale_len: usize,
    ) {
        #[repr(C)]
        struct P {
            n: i32,
            scale_len: i32,
        }
        let params = P {
            n: n as i32,
            scale_len: scale_len as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            8,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("mul_scale_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(scale), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        let grid = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Fused scale-add: out = a + b * scale (replaces mul_scale + add)
    pub fn fused_scale_add_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        scale: &Buffer,
        output: &Buffer,
        n: usize,
        scale_len: usize,
    ) {
        #[repr(C)]
        struct P {
            n: i32,
            scale_len: i32,
        }
        let params = P {
            n: n as i32,
            scale_len: scale_len as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            8,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("fused_scale_add_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(scale), 0);
        enc.set_buffer(3, Some(output), 0);
        enc.set_buffer(4, Some(&params_buf), 0);
        let grid = MTLSize::new(((n + 255) / 256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Fused residual-add + RMSNorm: out_res = a + b*scale, out_norm = rms_norm(out_res)*weight
    pub fn fused_residual_norm_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        scale: Option<&Buffer>,
        weight: &Buffer,
        out_res: &Buffer,
        out_norm: &Buffer,
        tokens: usize,
        dim: usize,
        eps: f32,
        scale_len: usize,
    ) {
        #[repr(C)]
        struct P {
            tokens: i32,
            dim: i32,
            eps: f32,
            has_scale: i32,
            scale_len: i32,
        }
        let dummy_buf = self
            .device
            .new_buffer(4, MTLResourceOptions::StorageModeShared);
        let params = P {
            tokens: tokens as i32,
            dim: dim as i32,
            eps,
            has_scale: if scale.is_some() { 1 } else { 0 },
            scale_len: scale_len as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            std::mem::size_of::<P>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        enc.set_compute_pipeline_state(self.pipeline("fused_residual_norm_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(scale.unwrap_or(&dummy_buf)), 0);
        enc.set_buffer(3, Some(weight), 0);
        enc.set_buffer(4, Some(out_res), 0);
        enc.set_buffer(5, Some(out_norm), 0);
        enc.set_buffer(6, Some(&params_buf), 0);
        enc.set_threadgroup_memory_length(0, 128);
        let grid = MTLSize::new(tokens as u64, 1, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Run flash attention. kv_seq_stride: 0 = contiguous (use kv_len), >0 = strided cache.
    pub fn flash_attn_v2(
        &self,
        cmd: &CommandBufferRef,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        o: &Buffer,
        params: &crate::AttentionParams,
        kv_seq_stride: usize,
    ) {
        #[repr(C)]
        struct P {
            batch: i32,
            num_heads: i32,
            num_kv_heads: i32,
            q_len: i32,
            kv_len: i32,
            head_dim: i32,
            scale: f32,
            causal: i32,
            pos_offset: i32,
            kv_seq_stride: i32,
        }
        let p = P {
            batch: params.batch as i32,
            num_heads: params.num_heads as i32,
            num_kv_heads: params.num_kv_heads as i32,
            q_len: params.q_len as i32,
            kv_len: params.kv_len as i32,
            head_dim: params.head_dim as i32,
            scale: 1.0 / (params.head_dim as f32).sqrt(),
            causal: if params.causal { 1 } else { 0 },
            pos_offset: params.pos_offset as i32,
            kv_seq_stride: kv_seq_stride as i32,
        };
        let params_buf = self.device.new_buffer_with_data(
            &p as *const _ as *const c_void,
            std::mem::size_of::<P>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipeline("flash_attn_f32"));
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k), 0);
        enc.set_buffer(2, Some(v), 0);
        enc.set_buffer(3, Some(o), 0);
        enc.set_buffer(4, Some(&params_buf), 0);
        let grid = MTLSize::new(
            params.q_len as u64,
            params.num_heads as u64,
            params.batch as u64,
        );
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    /// Run flash attention (legacy API, contiguous KV)
    pub fn flash_attn(
        &self,
        cmd: &CommandBufferRef,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        o: &Buffer,
        params: &crate::AttentionParams,
    ) {
        self.flash_attn_v2(cmd, q, k, v, o, params, 0);
    }
}
