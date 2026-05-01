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
        let gemm_f16w_src = include_str!("shaders/gemm_f16w.metal");
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
        let gemm_f16w_lib = device
            .new_library_with_source(gemm_f16w_src, &opts)
            .expect("failed to compile gemm_f16w.metal");
        let nr_lib = device
            .new_library_with_source(nr_src, &opts)
            .expect("failed to compile norm_rope.metal");
        let sm_lib = device
            .new_library_with_source(sm_src, &opts)
            .expect("failed to compile softmax.metal");

        let mut pipelines = HashMap::new();
        for (lib, names) in [
            (
                &fa_lib,
                &[
                    "flash_attn_f32",
                    "flash_attn_q_tiled_f32",
                    "flash_attn_decode_f32",
                    "flash_attn_decode_paged_f32",
                ][..],
            ),
            (
                &ops_lib,
                &[
                    "rms_norm_f32",
                    "silu_mul_f32",
                    "add_f32",
                    "scaled_add_inplace_f32",
                    "mul_scale_f32",
                    "fused_scale_add_f32",
                    "fused_residual_norm_f32",
                    "gemm_f32",
                    "argmax_f32",
                    "embedding_lookup_f32",
                    "split_qkv_f32",
                    "silu_mul_split_f32",
                    "gemv_f32",
                    "layer_norm_f32",
                    "gelu_f32",
                    "add_bias_f32",
                ][..],
            ),
            (&gemm_lib, &["gemm_f32_v2"][..]),
            (&gemm_f16w_lib, &["gemm_f32a_f16w_v2", "gemv_f32a_f16w"][..]),
            (
                &nr_lib,
                &[
                    "qk_norm_rope_transpose_f32",
                    "transpose_out_f32",
                    "kv_cache_append_f32",
                    "split_qkv_norm_rope_f32",
                    "split_qkv_norm_rope_kvc_f32",
                    "split_qkv_norm_rope_paged_kvc_f32",
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
        enc.set_compute_pipeline_state(self.pipeline("rms_norm_f32"));
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 8, &params as *const _ as *const c_void as *const _);
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
        enc.set_compute_pipeline_state(self.pipeline("silu_mul_f32"));
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
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
        enc.set_compute_pipeline_state(self.pipeline("add_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Scalar-scaled in-place add: `dst[i] += scale * src[i]`. Used by the
    /// MoE per-(token, expert) combine where each expert's down-projection
    /// is weighted by a router-derived scalar before summing into the
    /// per-token output. Inlining the multiply avoids materialising
    /// `scale * src` into a transient buffer.
    pub fn scaled_add_inplace_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        dst: &Buffer,
        src: &Buffer,
        scale: f32,
        n: usize,
    ) {
        #[repr(C)]
        struct P {
            n: i32,
            scale: f32,
        }
        let params = P { n: n as i32, scale };
        enc.set_compute_pipeline_state(self.pipeline("scaled_add_inplace_f32"));
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(src), 0);
        enc.set_bytes(2, 8, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
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

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipeline("rms_norm_f32"));
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 8, &params as *const _ as *const c_void as *const _);
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

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipeline("silu_mul_f32"));
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
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

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipeline("add_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 4, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    // ── New all-Metal dispatch methods (encoder-based, no commit) ──────

    /// GEMV: m=1 specialization. C[1, N] = A[1, K] @ B[N, K]^T.
    /// One threadgroup per output column, 32 threads; K-reduction via simd_sum.
    pub fn gemv_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
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
            m: 1,
            n: n as i32,
            k: k as i32,
        };
        enc.set_compute_pipeline_state(self.pipeline("gemv_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        enc.set_bytes(3, 12, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n as u64, 1, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

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
        enc.set_compute_pipeline_state(self.pipeline("gemm_f32_v2"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        enc.set_bytes(3, 12, &params as *const _ as *const c_void as *const _);
        enc.set_threadgroup_memory_length(0, 12288);
        let grid = MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1);
        let tg = MTLSize::new(128, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// GEMM with f16 weights: C[M,N] f32 = A[M,K] f32 @ B[N,K]^T f16.
    /// Same tile shape as `gemm_v2`; B is read as half and upcast on load.
    pub fn gemm_v2_f16w(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b_f16: &Buffer,
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
        enc.set_compute_pipeline_state(self.pipeline("gemm_f32a_f16w_v2"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b_f16), 0);
        enc.set_buffer(2, Some(c), 0);
        enc.set_bytes(3, 12, &params as *const _ as *const c_void as *const _);
        enc.set_threadgroup_memory_length(0, 12288);
        let grid = MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1);
        let tg = MTLSize::new(128, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// GEMV with f16 weights: C[1,N] f32 = A[1,K] f32 @ B[N,K]^T f16.
    /// Used for decode (m=1) against the large weight matrices.
    pub fn gemv_enc_f16w(
        &self,
        enc: &ComputeCommandEncoderRef,
        a: &Buffer,
        b_f16: &Buffer,
        c: &Buffer,
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
            m: 1,
            n: n as i32,
            k: k as i32,
        };
        enc.set_compute_pipeline_state(self.pipeline("gemv_f32a_f16w"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b_f16), 0);
        enc.set_buffer(2, Some(c), 0);
        enc.set_bytes(3, 12, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n as u64, 1, 1);
        let tg = MTLSize::new(32, 1, 1);
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
        enc.set_compute_pipeline_state(self.pipeline("qk_norm_rope_transpose_f32"));
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(cos), 0);
        enc.set_buffer(3, Some(sin), 0);
        enc.set_buffer(4, Some(output), 0);
        enc.set_bytes(
            5,
            std::mem::size_of::<P>() as u64,
            &params as *const _ as *const c_void as *const _,
        );
        let grid = MTLSize::new(tokens as u64, heads as u64, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Fused split-QKV + QK-norm + RoPE + transpose.
    ///
    /// Replaces (`split_qkv` → 3× `qk_norm_rope_transpose`) with a single
    /// dispatch that reads the linear-layer fused-QKV output once and
    /// writes head-major Q/K (with norm+RoPE) and V (transpose only)
    /// directly to attention scratch.
    ///
    /// `qk_mode`: 1 = norm + RoPE for Q/K (Qwen3 with QK-norm),
    ///            2 = RoPE only for Q/K (no QK-norm; Llama-style).
    #[allow(clippy::too_many_arguments)]
    pub fn split_qkv_norm_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        qkv: &Buffer,
        q_norm_w: &Buffer,
        k_norm_w: &Buffer,
        cos: &Buffer,
        sin: &Buffer,
        q_out: &Buffer,
        k_out: &Buffer,
        v_out: &Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) {
        #[repr(C)]
        struct P {
            tokens: i32,
            q_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            half_dim: i32,
            pos_offset: i32,
            eps: f32,
            qk_mode: i32,
        }
        let params = P {
            tokens: tokens as i32,
            q_heads: q_heads as i32,
            kv_heads: kv_heads as i32,
            head_dim: head_dim as i32,
            half_dim: (head_dim / 2) as i32,
            pos_offset: pos_offset as i32,
            eps,
            qk_mode,
        };
        enc.set_compute_pipeline_state(self.pipeline("split_qkv_norm_rope_f32"));
        enc.set_buffer(0, Some(qkv), 0);
        enc.set_buffer(1, Some(q_norm_w), 0);
        enc.set_buffer(2, Some(k_norm_w), 0);
        enc.set_buffer(3, Some(cos), 0);
        enc.set_buffer(4, Some(sin), 0);
        enc.set_buffer(5, Some(q_out), 0);
        enc.set_buffer(6, Some(k_out), 0);
        enc.set_buffer(7, Some(v_out), 0);
        enc.set_bytes(
            8,
            std::mem::size_of::<P>() as u64,
            &params as *const _ as *const c_void as *const _,
        );
        let grid = MTLSize::new(tokens as u64, (q_heads + 2 * kv_heads) as u64, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Variant of `split_qkv_norm_rope` that writes K/V straight into the
    /// pre-allocated head-major KV cache at slot `cache_len + tok`.
    /// Eliminates the trailing `kv_cache_append_head_major` dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn split_qkv_norm_rope_into_cache(
        &self,
        enc: &ComputeCommandEncoderRef,
        qkv: &Buffer,
        q_norm_w: &Buffer,
        k_norm_w: &Buffer,
        cos: &Buffer,
        sin: &Buffer,
        q_out: &Buffer,
        cache_k: &Buffer,
        cache_v: &Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
        cache_len: usize,
        cache_capacity: usize,
    ) {
        #[repr(C)]
        struct P {
            tokens: i32,
            q_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            half_dim: i32,
            pos_offset: i32,
            eps: f32,
            qk_mode: i32,
            cache_len: i32,
            cache_capacity: i32,
        }
        let params = P {
            tokens: tokens as i32,
            q_heads: q_heads as i32,
            kv_heads: kv_heads as i32,
            head_dim: head_dim as i32,
            half_dim: (head_dim / 2) as i32,
            pos_offset: pos_offset as i32,
            eps,
            qk_mode,
            cache_len: cache_len as i32,
            cache_capacity: cache_capacity as i32,
        };
        enc.set_compute_pipeline_state(self.pipeline("split_qkv_norm_rope_kvc_f32"));
        enc.set_buffer(0, Some(qkv), 0);
        enc.set_buffer(1, Some(q_norm_w), 0);
        enc.set_buffer(2, Some(k_norm_w), 0);
        enc.set_buffer(3, Some(cos), 0);
        enc.set_buffer(4, Some(sin), 0);
        enc.set_buffer(5, Some(q_out), 0);
        enc.set_buffer(6, Some(cache_k), 0);
        enc.set_buffer(7, Some(cache_v), 0);
        enc.set_bytes(
            8,
            std::mem::size_of::<P>() as u64,
            &params as *const _ as *const c_void as *const _,
        );
        let grid = MTLSize::new(tokens as u64, (q_heads + 2 * kv_heads) as u64, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Paged-KV variant of [`Self::split_qkv_norm_rope_into_cache`].
    ///
    /// Same fused split + qk-norm + RoPE + cache-write, but K/V are
    /// written into a paged pool `[num_blocks, kv_heads, block_size,
    /// head_dim]` indexed via `block_table[logical_block]` →
    /// physical_block. Q still goes to head-major scratch.
    ///
    /// `block_table` is the per-sequence logical→physical map for the
    /// caller's single sequence; multi-seq dispatch threads `tgpig.z`
    /// into a `[num_seqs, max_blocks_per_seq]` table — that's a future
    /// PR. For now this signature handles single-seq decode + prefill
    /// (the only call sites we're integrating in Phase 3).
    #[allow(clippy::too_many_arguments)]
    /// Dispatch the paged split+norm+RoPE+kvc kernel.
    ///
    /// `qkv_byte_offset` lets the caller pass a slice into a larger
    /// batched buffer (per-item dispatch from `decode_batch_internal`'s
    /// paged path). `q_out_byte_offset` likewise positions the per-item
    /// Q output into a stacked batched buffer. Both default to 0 for
    /// the single-seq case.
    #[allow(clippy::too_many_arguments)]
    pub fn split_qkv_norm_rope_into_paged_cache(
        &self,
        enc: &ComputeCommandEncoderRef,
        qkv: &Buffer,
        qkv_byte_offset: u64,
        q_norm_w: &Buffer,
        k_norm_w: &Buffer,
        cos: &Buffer,
        sin: &Buffer,
        q_out: &Buffer,
        q_out_byte_offset: u64,
        cache_k: &Buffer,
        cache_v: &Buffer,
        block_table: &Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
        cache_len: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
    ) {
        // Layout matches `SplitQkvNormRopePagedKvcParams` in norm_rope.metal.
        #[repr(C)]
        struct P {
            tokens: i32,
            q_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            half_dim: i32,
            pos_offset: i32,
            eps: f32,
            qk_mode: i32,
            cache_len: i32,
            block_size: i32,
            max_num_blocks_per_seq: i32,
        }
        let params = P {
            tokens: tokens as i32,
            q_heads: q_heads as i32,
            kv_heads: kv_heads as i32,
            head_dim: head_dim as i32,
            half_dim: (head_dim / 2) as i32,
            pos_offset: pos_offset as i32,
            eps,
            qk_mode,
            cache_len: cache_len as i32,
            block_size: block_size as i32,
            max_num_blocks_per_seq: max_num_blocks_per_seq as i32,
        };
        enc.set_compute_pipeline_state(self.pipeline("split_qkv_norm_rope_paged_kvc_f32"));
        enc.set_buffer(0, Some(qkv), qkv_byte_offset);
        enc.set_buffer(1, Some(q_norm_w), 0);
        enc.set_buffer(2, Some(k_norm_w), 0);
        enc.set_buffer(3, Some(cos), 0);
        enc.set_buffer(4, Some(sin), 0);
        enc.set_buffer(5, Some(q_out), q_out_byte_offset);
        enc.set_buffer(6, Some(cache_k), 0);
        enc.set_buffer(7, Some(cache_v), 0);
        enc.set_buffer(8, Some(block_table), 0);
        enc.set_bytes(
            9,
            std::mem::size_of::<P>() as u64,
            &params as *const _ as *const c_void as *const _,
        );
        let grid = MTLSize::new(tokens as u64, (q_heads + 2 * kv_heads) as u64, 1);
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
        enc.set_compute_pipeline_state(self.pipeline("transpose_out_f32"));
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(output), 0);
        enc.set_bytes(2, 12, &params as *const _ as *const c_void as *const _);
        let n = tokens * heads * head_dim;
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
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
        enc.set_compute_pipeline_state(self.pipeline("kv_cache_append_f32"));
        enc.set_buffer(0, Some(new_data), 0);
        enc.set_buffer(1, Some(cache), 0);
        enc.set_bytes(2, 20, &params as *const _ as *const c_void as *const _);
        let n = heads * new_len * head_dim;
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
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
        enc.set_compute_pipeline_state(self.pipeline("softmax_last_dim_f32"));
        enc.set_buffer(0, Some(data), 0);
        enc.set_bytes(1, 8, &params as *const _ as *const c_void as *const _);
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
        enc.set_compute_pipeline_state(self.pipeline("mul_scale_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(scale), 0);
        enc.set_buffer(2, Some(output), 0);
        enc.set_bytes(3, 8, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
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
        enc.set_compute_pipeline_state(self.pipeline("fused_scale_add_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(scale), 0);
        enc.set_buffer(3, Some(output), 0);
        enc.set_bytes(4, 8, &params as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(n.div_ceil(256) as u64, 1, 1);
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
        enc.set_compute_pipeline_state(self.pipeline("fused_residual_norm_f32"));
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(scale.unwrap_or(&dummy_buf)), 0);
        enc.set_buffer(3, Some(weight), 0);
        enc.set_buffer(4, Some(out_res), 0);
        enc.set_buffer(5, Some(out_norm), 0);
        enc.set_bytes(
            6,
            std::mem::size_of::<P>() as u64,
            &params as *const _ as *const c_void as *const _,
        );
        enc.set_threadgroup_memory_length(0, 128);
        let grid = MTLSize::new(tokens as u64, 1, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Run flash attention. kv_seq_stride: 0 = contiguous (use kv_len), >0 = strided cache.
    ///
    /// Dispatches the Q-tiled simdgroup_matmul kernel for the prefill-shaped
    /// hot path (head_dim=128, no sliding window, q_len ≥ Q_TILE) and falls
    /// back to the legacy per-query kernel otherwise (decode m=1, special
    /// head sizes, sliding-window models).
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
            sliding_window: i32,
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
            sliding_window: params.sliding_window as i32,
        };

        // Three kernel variants, picked by query length and head shape:
        //
        //   q_len ≥ 8  + head_dim=128         → flash_attn_q_tiled_f32
        //                                       (4-simdgroup tile, simdgroup_matmul)
        //   q_len == 1 + head_dim=128 + no
        //                sliding_window       → flash_attn_decode_f32
        //                                       (32-simdgroup wide TG; the MLX
        //                                        sdpa_vector port — 32× more
        //                                        active threads than the legacy
        //                                        scalar path used to give us)
        //   everything else                   → flash_attn_f32 (legacy scalar)
        //
        // Override via `FERRUM_FA_LEGACY=1` to force the scalar path (debugging
        // / numerical comparison). `FERRUM_FA_DECODE=0` disables the new
        // decode kernel specifically while leaving Q-tiled prefill enabled.
        const Q_TILE_R: usize = 8;
        let force_legacy = std::env::var("FERRUM_FA_LEGACY").as_deref() == Ok("1");
        let allow_decode_widen = std::env::var("FERRUM_FA_DECODE").as_deref() != Ok("0");
        let use_q_tiled = !force_legacy
            && params.head_dim == 128
            && params.sliding_window == 0
            && params.q_len >= Q_TILE_R;
        let use_decode_widen = !force_legacy
            && allow_decode_widen
            && params.head_dim == 128
            && params.sliding_window == 0
            && params.q_len == 1;

        let enc = cmd.new_compute_command_encoder();
        if use_q_tiled {
            enc.set_compute_pipeline_state(self.pipeline("flash_attn_q_tiled_f32"));
            enc.set_buffer(0, Some(q), 0);
            enc.set_buffer(1, Some(k), 0);
            enc.set_buffer(2, Some(v), 0);
            enc.set_buffer(3, Some(o), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<P>() as u64,
                &p as *const _ as *const c_void as *const _,
            );
            let q_tiles = params.q_len.div_ceil(Q_TILE_R) as u64;
            let grid = MTLSize::new(q_tiles, params.num_heads as u64, params.batch as u64);
            let tg = MTLSize::new(128, 1, 1); // 4 simdgroups
            enc.dispatch_thread_groups(grid, tg);
        } else if use_decode_widen {
            enc.set_compute_pipeline_state(self.pipeline("flash_attn_decode_f32"));
            enc.set_buffer(0, Some(q), 0);
            enc.set_buffer(1, Some(k), 0);
            enc.set_buffer(2, Some(v), 0);
            enc.set_buffer(3, Some(o), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<P>() as u64,
                &p as *const _ as *const c_void as *const _,
            );
            // q_len always 1 here. One TG per (head, batch); each TG has
            // SDPA_BN=32 simdgroups × SDPA_BD=32 threads = 1024 threads.
            let grid = MTLSize::new(1, params.num_heads as u64, params.batch as u64);
            let tg = MTLSize::new(32, 32, 1);
            enc.dispatch_thread_groups(grid, tg);
        } else {
            enc.set_compute_pipeline_state(self.pipeline("flash_attn_f32"));
            enc.set_buffer(0, Some(q), 0);
            enc.set_buffer(1, Some(k), 0);
            enc.set_buffer(2, Some(v), 0);
            enc.set_buffer(3, Some(o), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<P>() as u64,
                &p as *const _ as *const c_void as *const _,
            );
            let grid = MTLSize::new(
                params.q_len as u64,
                params.num_heads as u64,
                params.batch as u64,
            );
            let tg = MTLSize::new(32, 1, 1);
            enc.dispatch_thread_groups(grid, tg);
        }
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

    /// Split fused QKV: qkv [tokens, q_dim + 2*kv_dim] → q, k, v separate buffers.
    pub fn split_qkv_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        qkv: &Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        #[repr(C)]
        struct P {
            tokens: i32,
            q_dim: i32,
            kv_dim: i32,
        }
        let p = P {
            tokens: tokens as i32,
            q_dim: q_dim as i32,
            kv_dim: kv_dim as i32,
        };
        enc.set_compute_pipeline_state(self.pipeline("split_qkv_f32"));
        enc.set_buffer(0, Some(qkv), 0);
        enc.set_buffer(1, Some(q), 0);
        enc.set_buffer(2, Some(k), 0);
        enc.set_buffer(3, Some(v), 0);
        enc.set_bytes(4, 12, &p as *const _ as *const c_void as *const _);
        let total = tokens * (q_dim + 2 * kv_dim);
        let grid = MTLSize::new(total.div_ceil(256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// LayerNorm with learnable gamma/beta.
    pub fn layer_norm_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        gamma: &Buffer,
        beta: &Buffer,
        out: &Buffer,
        tokens: usize,
        dim: usize,
        eps: f32,
    ) {
        #[repr(C)]
        struct P {
            dim: i32,
            eps: f32,
        }
        let p = P {
            dim: dim as i32,
            eps,
        };
        enc.set_compute_pipeline_state(self.pipeline("layer_norm_f32"));
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(gamma), 0);
        enc.set_buffer(2, Some(beta), 0);
        enc.set_buffer(3, Some(out), 0);
        enc.set_bytes(4, 8, &p as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(tokens as u64, 1, 1);
        let tg = MTLSize::new(32, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Element-wise GELU (erf-based, matches torch default).
    pub fn gelu_enc(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, out: &Buffer, len: usize) {
        #[repr(C)]
        struct P {
            n: i32,
        }
        let p = P { n: len as i32 };
        enc.set_compute_pipeline_state(self.pipeline("gelu_f32"));
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_bytes(2, 4, &p as *const _ as *const c_void as *const _);
        let grid = MTLSize::new(len.div_ceil(256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Broadcast bias add: data[r, c] += bias[c].
    pub fn add_bias_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &Buffer,
        bias: &Buffer,
        rows: usize,
        cols: usize,
    ) {
        #[repr(C)]
        struct P {
            rows: i32,
            cols: i32,
        }
        let p = P {
            rows: rows as i32,
            cols: cols as i32,
        };
        enc.set_compute_pipeline_state(self.pipeline("add_bias_f32"));
        enc.set_buffer(0, Some(data), 0);
        enc.set_buffer(1, Some(bias), 0);
        enc.set_bytes(2, 8, &p as *const _ as *const c_void as *const _);
        let total = rows * cols;
        let grid = MTLSize::new(total.div_ceil(256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// SiLU(gate) * up with fused gate_up split: gate_up [tokens, 2*im] → out [tokens, im].
    pub fn silu_mul_split_enc(
        &self,
        enc: &ComputeCommandEncoderRef,
        gate_up: &Buffer,
        out: &Buffer,
        tokens: usize,
        im: usize,
    ) {
        #[repr(C)]
        struct P {
            tokens: i32,
            im: i32,
        }
        let p = P {
            tokens: tokens as i32,
            im: im as i32,
        };
        enc.set_compute_pipeline_state(self.pipeline("silu_mul_split_f32"));
        enc.set_buffer(0, Some(gate_up), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_bytes(2, 8, &p as *const _ as *const c_void as *const _);
        let total = tokens * im;
        let grid = MTLSize::new(total.div_ceil(256) as u64, 1, 1);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
    }

    /// Run paged-KV attention on the caller's existing compute encoder.
    /// Handles both decode (`q_len=1`) and causal prefill (`q_len>1`).
    ///
    /// Q/O layout selected by `q_layout`:
    ///   `TokenMajor` (decode): `[num_seqs, num_heads, head_dim]`
    ///   `HeadMajor`  (prefill, single seq batch=1):
    ///                          `[num_heads, q_len, head_dim]`
    ///
    /// K/V cache: `[num_blocks, num_kv_heads, block_size, head_dim]`
    /// `block_tables`: `[num_seqs, max_num_blocks_per_seq]` u32
    /// `context_lens`: `[num_seqs]` u32 — FINAL kv_len after this batch's
    ///   writes. The kernel computes per-q-token causal limit as
    ///   `context_lens[seq] - (q_len - 1 - q_token_idx)`, so token i
    ///   sees positions [0, context_lens - q_len + 1 + i).
    ///
    /// Caller is responsible for opening / closing the encoder.
    ///
    /// Restrictions: `head_dim == 128` (will panic otherwise on the
    /// debug_assert). Block size configurable.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_attention_on_encoder(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        o: &Buffer,
        block_tables: &Buffer,
        context_lens: &Buffer,
        params: &PagedAttnDispatchParams,
    ) {
        debug_assert_eq!(
            params.head_dim, 128,
            "paged_decode_attention currently only supports head_dim=128"
        );
        debug_assert!(
            params.num_heads % params.num_kv_heads == 0,
            "GQA: num_heads must be divisible by num_kv_heads"
        );
        debug_assert!(params.q_len >= 1, "q_len must be ≥ 1");

        // Q/O strides depend on layout:
        //   TokenMajor (q_len=1 typical): q_head_stride = head_dim
        //   HeadMajor  (q_len>1, single seq): q_head_stride = q_len * head_dim
        let (q_head_stride, o_head_stride) = match params.q_layout {
            PagedAttnQLayout::TokenMajor => (params.head_dim as i32, params.head_dim as i32),
            PagedAttnQLayout::HeadMajor => {
                let s = (params.q_len * params.head_dim) as i32;
                (s, s)
            }
        };

        // Kernel-side struct mirror of `PagedAttnParams` in flash_attn.metal.
        #[repr(C)]
        struct P {
            num_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            scale: f32,
            block_size: i32,
            max_num_blocks_per_seq: i32,
            kv_block_stride: i32,
            kv_head_stride: i32,
            q_len: i32,
            q_head_stride: i32,
            o_head_stride: i32,
        }
        let kv_head_stride = (params.block_size * params.head_dim) as i32;
        let kv_block_stride = (params.num_kv_heads as i32) * kv_head_stride;
        let p = P {
            num_heads: params.num_heads as i32,
            num_kv_heads: params.num_kv_heads as i32,
            head_dim: params.head_dim as i32,
            scale: 1.0 / (params.head_dim as f32).sqrt(),
            block_size: params.block_size as i32,
            max_num_blocks_per_seq: params.max_num_blocks_per_seq as i32,
            kv_block_stride,
            kv_head_stride,
            q_len: params.q_len as i32,
            q_head_stride,
            o_head_stride,
        };

        enc.set_compute_pipeline_state(self.pipeline("flash_attn_decode_paged_f32"));
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k_cache), 0);
        enc.set_buffer(2, Some(v_cache), 0);
        enc.set_buffer(3, Some(o), 0);
        enc.set_buffer(4, Some(block_tables), 0);
        enc.set_buffer(5, Some(context_lens), 0);
        enc.set_bytes(
            6,
            std::mem::size_of::<P>() as u64,
            &p as *const _ as *const c_void as *const _,
        );

        // Grid: (q_len, num_heads, num_seqs). For q_len=1 this is
        // identical to the previous (1, num_heads, num_seqs) shape;
        // q_len>1 spawns one TG per query token to walk causal KV in
        // parallel across token positions.
        let grid = MTLSize::new(
            params.q_len as u64,
            params.num_heads as u64,
            params.num_seqs as u64,
        );
        let tg = MTLSize::new(32, 32, 1);
        enc.dispatch_thread_groups(grid, tg);
    }
}

/// Q/O memory layout for paged attention.
///
/// `TokenMajor` matches the contiguous decode-step layout: rows are
/// per-sequence per-head per-token vectors. With q_len=1 (decode) it's
/// `[num_seqs, num_heads, head_dim]`.
///
/// `HeadMajor` is what `split_qkv_norm_rope` writes for prefill:
/// `[num_heads, q_len, head_dim]` with each head's q_len rows
/// contiguous. Used when paged attention runs on prefill output
/// directly without a transpose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagedAttnQLayout {
    TokenMajor,
    HeadMajor,
}

/// Caller-side parameters for [`MetalPipelines::paged_decode_attention_on_encoder`].
///
/// Layout / stride conventions match the comments on the dispatch
/// helper. `kv_block_stride` / `kv_head_stride` / `q_head_stride` /
/// `o_head_stride` are computed inside the dispatch from these fields.
#[derive(Clone, Copy, Debug)]
pub struct PagedAttnDispatchParams {
    pub num_seqs: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub max_num_blocks_per_seq: usize,
    /// 1 for decode (most common), >1 for causal prefill.
    pub q_len: usize,
    /// Layout of `q` and `o` buffers. See [`PagedAttnQLayout`].
    pub q_layout: PagedAttnQLayout,
}
