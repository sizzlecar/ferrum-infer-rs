//! CUDA decode runner — full decode forward pass bypassing candle.
//!
//! All intermediate buffers are pre-allocated. The runner uses cuBLAS for GEMM
//! and custom CUDA kernels for norm/rope/attention/activation directly via cudarc.
//! No candle Tensor allocation in the hot path.
//!
//! This module is the foundation for CUDA Graph capture (Phase 3).

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::cuda_backend::cudarc;
use candle_core::cuda_backend::CudaDevice;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::decode_buffers::{DecodeBuffers, ModelDims};
use crate::ptx;
use crate::weight_store::{GpuWeight, Qwen3Weights};

/// Per-sequence KV cache managed by the decode runner.
struct SequenceKvState {
    /// Per-layer K cache: [num_kv_heads * max_len * head_dim] each
    k_caches: Vec<CudaSlice<half::f16>>,
    /// Per-layer V cache: same layout
    v_caches: Vec<CudaSlice<half::f16>>,
    /// Number of valid KV positions
    current_len: usize,
    /// Allocated max length
    max_len: usize,
}

/// CUDA decode runner for Qwen3-style transformer models.
///
/// Executes the full decode forward pass using:
/// - cuBLAS GEMM for all linear projections
/// - Custom CUDA kernels for norm, RoPE, attention, activation
/// - Pre-allocated intermediate buffers (DecodeBuffers)
///
/// Created from a loaded candle model by extracting weight GPU pointers.
pub struct CudaDecodeRunner {
    weights: Qwen3Weights,
    buffers: DecodeBuffers,
    blas: Arc<CudaBlas>,
    device: CudaDevice,
    stream: Arc<CudaStream>,
    dims: ModelDims,
    /// Per-sequence KV cache state
    kv_states: HashMap<String, SequenceKvState>,
}

impl CudaDecodeRunner {
    /// Create a new decode runner.
    ///
    /// - `weights`: GPU weight pointers extracted from the candle model
    /// - `dims`: Model dimensions
    /// - `device`: candle CudaDevice (used for kernel loading and cuBLAS handle)
    pub fn new(
        weights: Qwen3Weights,
        dims: ModelDims,
        device: CudaDevice,
    ) -> candle_core::Result<Self> {
        let stream = device.cuda_stream();
        let blas = device.cublas_handle();
        let buffers = DecodeBuffers::new(dims.clone(), &stream)
            .map_err(|e| candle_core::Error::Msg(format!("DecodeBuffers alloc: {e}")))?;

        tracing::info!(
            "CudaDecodeRunner initialized: {}MB decode buffers, {} layers",
            buffers.memory_bytes() / (1024 * 1024),
            dims.num_layers,
        );

        Ok(Self {
            weights,
            buffers,
            blas,
            device,
            stream,
            dims,
            kv_states: HashMap::new(),
        })
    }

    /// Initialize KV cache for a sequence from prefill data.
    ///
    /// `kv_data`: per-layer (K, V) CudaSlice pairs from prefill, each
    /// [num_kv_heads * prefill_len * head_dim].
    pub fn init_kv_cache(
        &mut self,
        cache_key: &str,
        kv_data: Vec<(CudaSlice<half::f16>, CudaSlice<half::f16>)>,
        prefill_len: usize,
    ) -> candle_core::Result<()> {
        let kv_head_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let max_len = (prefill_len * 8).max(2048).min(self.dims.max_seq_len);

        let mut k_caches = Vec::with_capacity(self.dims.num_layers);
        let mut v_caches = Vec::with_capacity(self.dims.num_layers);

        for (k_src, v_src) in &kv_data {
            // Allocate full-size KV buffer
            let k_buf = unsafe {
                self.stream
                    .alloc::<half::f16>(kv_head_dim * max_len)
                    .map_err(|e| candle_core::Error::Msg(format!("KV alloc: {e}")))?
            };
            let v_buf = unsafe {
                self.stream
                    .alloc::<half::f16>(kv_head_dim * max_len)
                    .map_err(|e| candle_core::Error::Msg(format!("KV alloc: {e}")))?
            };

            // Copy prefill data into the beginning of the buffer
            let copy_bytes = kv_head_dim * prefill_len * std::mem::size_of::<half::f16>();
            let k_src_view = k_src.slice(..kv_head_dim * prefill_len);
            let mut k_dst_view = k_buf.slice_mut(..kv_head_dim * prefill_len);
            self.stream
                .memcpy_dtod(&k_src_view, &mut k_dst_view)
                .map_err(|e| candle_core::Error::Msg(format!("KV memcpy: {e}")))?;

            let v_src_view = v_src.slice(..kv_head_dim * prefill_len);
            let mut v_dst_view = v_buf.slice_mut(..kv_head_dim * prefill_len);
            self.stream
                .memcpy_dtod(&v_src_view, &mut v_dst_view)
                .map_err(|e| candle_core::Error::Msg(format!("KV memcpy: {e}")))?;

            k_caches.push(k_buf);
            v_caches.push(v_buf);
        }

        self.kv_states.insert(
            cache_key.to_string(),
            SequenceKvState {
                k_caches,
                v_caches,
                current_len: prefill_len,
                max_len,
            },
        );

        Ok(())
    }

    /// Release KV cache for a completed sequence.
    pub fn release_kv_cache(&mut self, cache_key: &str) {
        self.kv_states.remove(cache_key);
    }

    /// Execute one decode step: single token in, logits out.
    ///
    /// Returns a CudaSlice containing logits [vocab_size].
    pub fn decode_step(
        &mut self,
        token_id: u32,
        position: usize,
        cache_key: &str,
    ) -> candle_core::Result<CudaSlice<half::f16>> {
        let h = self.dims.hidden_size;
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let inter = self.dims.intermediate_size;
        let head_dim = self.dims.head_dim;
        let num_q = self.dims.num_attention_heads;
        let num_kv = self.dims.num_kv_heads;
        let half_dim = head_dim / 2;
        let eps = 1e-6f32; // Qwen3 default

        // 1. Embedding lookup: copy row token_id from embed table
        let embed_offset = (token_id as usize) * h;
        let embed_src = self
            .weights
            .embed_table
            .slice
            .slice(embed_offset..embed_offset + h);
        self.stream
            .memcpy_dtod(&embed_src, &mut self.buffers.embed_out)
            .map_err(|e| candle_core::Error::Msg(format!("embed memcpy: {e}")))?;

        // Copy embedding to residual accumulator
        let embed_view = self.buffers.embed_out.slice(..);
        self.stream
            .memcpy_dtod(&embed_view, &mut self.buffers.residual)
            .map_err(|e| candle_core::Error::Msg(format!("residual init: {e}")))?;

        // Get KV state for this sequence
        let kv_state = self.kv_states.get_mut(cache_key).ok_or_else(|| {
            candle_core::Error::Msg(format!("No KV cache for sequence: {cache_key}"))
        })?;

        // RoPE cos/sin for this position
        let cos_offset = position * half_dim;
        let sin_offset = position * half_dim;

        // 2. Decoder layers
        for layer_idx in 0..self.dims.num_layers {
            let lw = &self.weights.layers[layer_idx];

            // 2a. Input LayerNorm
            self.launch_rms_norm(
                &self.buffers.residual,
                &lw.input_ln_w.slice,
                &mut self.buffers.norm_out,
                h,
                eps,
            )?;

            // 2b. Fused QKV projection: [1, h] @ [qkv_dim, h]^T → [1, qkv_dim]
            crate::cublas::linear_f16(
                &self.blas,
                &self.buffers.norm_out,
                &lw.qkv_w.slice,
                &mut self.buffers.qkv_out,
                1,
                qkv_dim as i32,
                h as i32,
            )
            .map_err(|e| candle_core::Error::Msg(format!("QKV gemm: {e}")))?;

            // 2c. Split QKV (pointer arithmetic — no compute)
            let q_slice = self.buffers.qkv_out.slice(..q_dim);
            let k_slice = self.buffers.qkv_out.slice(q_dim..q_dim + kv_dim);
            let v_slice = self.buffers.qkv_out.slice(q_dim + kv_dim..qkv_dim);

            // 2d. Q-norm and K-norm
            // Q: treat as [num_q_heads, head_dim], norm each row
            self.launch_rms_norm(
                &q_slice,
                &lw.q_norm_w.slice,
                &mut self.buffers.q_rotated, // reuse as temp
                head_dim,
                eps,
            )?;
            // K: treat as [num_kv_heads, head_dim]
            self.launch_rms_norm(
                &k_slice,
                &lw.k_norm_w.slice,
                &mut self.buffers.k_rotated, // reuse as temp
                head_dim,
                eps,
            )?;

            // 2e. RoPE (fused Q+K rotation)
            let cos_view = self
                .weights
                .rope_cos
                .slice
                .slice(cos_offset..cos_offset + half_dim);
            let sin_view = self
                .weights
                .rope_sin
                .slice
                .slice(sin_offset..sin_offset + half_dim);
            // RoPE writes back to q_rotated and k_rotated (in-place via separate out buffers)
            // We need temp buffers for in→out. Reuse norm_out and o_proj_out as temps.
            self.launch_rope(
                &self.buffers.q_rotated,
                &self.buffers.k_rotated,
                &cos_view,
                &sin_view,
                &mut self.buffers.norm_out,   // temp for q_out
                &mut self.buffers.o_proj_out, // temp for k_out
                num_q,
                num_kv,
                head_dim,
            )?;
            // Now q_rotated data is in norm_out, k_rotated data is in o_proj_out.
            // Swap back by memcpy.
            let norm_view = self.buffers.norm_out.slice(..q_dim);
            self.stream
                .memcpy_dtod(&norm_view, &mut self.buffers.q_rotated)
                .map_err(|e| candle_core::Error::Msg(format!("rope q copy: {e}")))?;
            let oproj_view = self.buffers.o_proj_out.slice(..kv_dim);
            self.stream
                .memcpy_dtod(&oproj_view, &mut self.buffers.k_rotated)
                .map_err(|e| candle_core::Error::Msg(format!("rope k copy: {e}")))?;

            // 2f. KV cache append
            let kv_head_dim_total = num_kv * head_dim;
            let kv_offset = kv_state.current_len * kv_head_dim_total;
            {
                let k_src_view = self.buffers.k_rotated.slice(..kv_dim);
                let mut k_dst =
                    kv_state.k_caches[layer_idx].slice_mut(kv_offset..kv_offset + kv_dim);
                self.stream
                    .memcpy_dtod(&k_src_view, &mut k_dst)
                    .map_err(|e| candle_core::Error::Msg(format!("KV append k: {e}")))?;

                let v_src_view = v_slice.slice(..kv_dim);
                let mut v_dst =
                    kv_state.v_caches[layer_idx].slice_mut(kv_offset..kv_offset + kv_dim);
                self.stream
                    .memcpy_dtod(&v_src_view, &mut v_dst)
                    .map_err(|e| candle_core::Error::Msg(format!("KV append v: {e}")))?;
            }
            let valid_kv_len = kv_state.current_len + 1;

            // 2g. Decode attention
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            self.launch_decode_attention(
                &self.buffers.q_rotated,
                &kv_state.k_caches[layer_idx],
                &kv_state.v_caches[layer_idx],
                &mut self.buffers.attn_out,
                num_q,
                num_kv,
                head_dim,
                kv_state.max_len,
                valid_kv_len,
                scale,
            )?;

            // 2h. O projection: [1, q_dim] @ [h, q_dim]^T → [1, h]
            crate::cublas::linear_f16(
                &self.blas,
                &self.buffers.attn_out,
                &lw.o_w.slice,
                &mut self.buffers.o_proj_out,
                1,
                h as i32,
                q_dim as i32,
            )
            .map_err(|e| candle_core::Error::Msg(format!("O proj gemm: {e}")))?;

            // 2i. Fused add + RMS norm:
            //   residual_updated = o_proj_out + residual
            //   post_norm_out = rms_norm(residual_updated, post_ln_w)
            self.launch_fused_add_rms_norm(
                &self.buffers.o_proj_out,
                &self.buffers.residual,
                &lw.post_ln_w.slice,
                &mut self.buffers.post_norm_out,
                &mut self.buffers.post_norm_residual,
                h,
                eps,
            )?;
            // Update residual to post_norm_residual
            let pnr_view = self.buffers.post_norm_residual.slice(..h);
            self.stream
                .memcpy_dtod(&pnr_view, &mut self.buffers.residual)
                .map_err(|e| candle_core::Error::Msg(format!("residual update: {e}")))?;

            // 2j. Gate+Up projection: [1, h] @ [2*inter, h]^T → [1, 2*inter]
            crate::cublas::linear_f16(
                &self.blas,
                &self.buffers.post_norm_out,
                &lw.gate_up_w.slice,
                &mut self.buffers.gate_up_out,
                1,
                (2 * inter) as i32,
                h as i32,
            )
            .map_err(|e| candle_core::Error::Msg(format!("gate_up gemm: {e}")))?;

            // 2k. Fused SiLU * mul
            let gate_view = self.buffers.gate_up_out.slice(..inter);
            let up_view = self.buffers.gate_up_out.slice(inter..2 * inter);
            self.launch_fused_silu_mul(&gate_view, &up_view, &mut self.buffers.mlp_act, inter)?;

            // 2l. Down projection: [1, inter] @ [h, inter]^T → [1, h]
            crate::cublas::linear_f16(
                &self.blas,
                &self.buffers.mlp_act,
                &lw.down_w.slice,
                &mut self.buffers.down_out,
                1,
                h as i32,
                inter as i32,
            )
            .map_err(|e| candle_core::Error::Msg(format!("down gemm: {e}")))?;

            // 2m. Residual add: residual = residual + down_out
            self.launch_residual_add(
                &self.buffers.residual,
                &self.buffers.down_out,
                &mut self.buffers.norm_out, // temp output
                h,
            )?;
            let norm_view = self.buffers.norm_out.slice(..h);
            self.stream
                .memcpy_dtod(&norm_view, &mut self.buffers.residual)
                .map_err(|e| candle_core::Error::Msg(format!("residual add copy: {e}")))?;
        }

        // Update KV length after processing all layers
        kv_state.current_len += 1;

        // 3. Final RMS norm
        self.launch_rms_norm(
            &self.buffers.residual,
            &self.weights.final_norm_w.slice,
            &mut self.buffers.final_norm_out,
            h,
            eps,
        )?;

        // 4. LM head: [1, h] @ [vocab, h]^T → [1, vocab]
        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.final_norm_out,
            &self.weights.lm_head_w.slice,
            &mut self.buffers.logits,
            1,
            self.dims.vocab_size as i32,
            h as i32,
        )
        .map_err(|e| candle_core::Error::Msg(format!("lm_head gemm: {e}")))?;

        // Return a clone of the logits buffer
        let logits = self
            .stream
            .clone_dtod(&self.buffers.logits)
            .map_err(|e| candle_core::Error::Msg(format!("logits clone: {e}")))?;

        Ok(logits)
    }

    // ======================== Kernel Launch Helpers ========================

    fn launch_rms_norm(
        &self,
        input: &impl cudarc::driver::DeviceSlice<half::f16>,
        weight: &impl cudarc::driver::DeviceSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        row_size: usize,
        eps: f32,
    ) -> candle_core::Result<()> {
        let num_rows = input.len() / row_size;
        let func =
            self.device
                .get_or_load_custom_func("rms_norm_f16", "rms_norm", ptx::RMS_NORM)?;
        let block_size = row_size.min(1024) as u32;
        let row_size_i32 = row_size as i32;

        let mut builder = func.builder();
        builder.arg(input);
        builder.arg(weight);
        builder.arg(output);
        builder.arg(&row_size_i32);
        builder.arg(&eps);

        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("rms_norm launch: {e}")))
    }

    fn launch_rope(
        &self,
        q: &impl cudarc::driver::DeviceSlice<half::f16>,
        k: &impl cudarc::driver::DeviceSlice<half::f16>,
        cos: &impl cudarc::driver::DeviceSlice<half::f16>,
        sin: &impl cudarc::driver::DeviceSlice<half::f16>,
        q_out: &mut CudaSlice<half::f16>,
        k_out: &mut CudaSlice<half::f16>,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
    ) -> candle_core::Result<()> {
        let func = self
            .device
            .get_or_load_custom_func("rope_f16", "rope", ptx::ROPE)?;
        let total_heads = (num_q_heads + num_k_heads) as u32;
        let half_dim = (head_dim / 2).min(1024) as u32;
        let num_q_i32 = num_q_heads as i32;
        let num_k_i32 = num_k_heads as i32;
        let head_dim_i32 = head_dim as i32;

        let mut builder = func.builder();
        builder.arg(q);
        builder.arg(k);
        builder.arg(cos);
        builder.arg(sin);
        builder.arg(q_out);
        builder.arg(k_out);
        builder.arg(&num_q_i32);
        builder.arg(&num_k_i32);
        builder.arg(&head_dim_i32);

        let cfg = LaunchConfig {
            grid_dim: (total_heads, 1, 1),
            block_dim: (half_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("rope launch: {e}")))
    }

    fn launch_decode_attention(
        &self,
        q: &impl cudarc::driver::DeviceSlice<half::f16>,
        k_cache: &CudaSlice<half::f16>,
        v_cache: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_kv_len: usize,
        valid_kv_len: usize,
        scale: f32,
    ) -> candle_core::Result<()> {
        let func = self.device.get_or_load_custom_func(
            "decode_attention_f16",
            "decode_attention",
            ptx::DECODE_ATTENTION,
        )?;
        let block_size = 256u32;
        let shared_mem = (max_kv_len as u32) * 4; // f32 per position

        let num_q_i32 = num_q_heads as i32;
        let num_kv_i32 = num_kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let max_kv_i32 = max_kv_len as i32;
        let valid_kv_i32 = valid_kv_len as i32;

        let mut builder = func.builder();
        builder.arg(q);
        builder.arg(k_cache);
        builder.arg(v_cache);
        builder.arg(output);
        builder.arg(&num_q_i32);
        builder.arg(&num_kv_i32);
        builder.arg(&head_dim_i32);
        builder.arg(&max_kv_i32);
        builder.arg(&valid_kv_i32);
        builder.arg(&scale);

        let cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("decode_attention launch: {e}")))
    }

    fn launch_fused_add_rms_norm(
        &self,
        input: &impl cudarc::driver::DeviceSlice<half::f16>,
        residual: &impl cudarc::driver::DeviceSlice<half::f16>,
        weight: &impl cudarc::driver::DeviceSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        residual_out: &mut CudaSlice<half::f16>,
        hidden_size: usize,
        eps: f32,
    ) -> candle_core::Result<()> {
        let func = self.device.get_or_load_custom_func(
            "fused_add_rms_norm_f16",
            "fused_add_rms_norm",
            ptx::FUSED_ADD_RMS_NORM,
        )?;
        let block_size = hidden_size.min(1024) as u32;
        let hidden_i32 = hidden_size as i32;

        let mut builder = func.builder();
        builder.arg(input);
        builder.arg(residual);
        builder.arg(weight);
        builder.arg(output);
        builder.arg(residual_out);
        builder.arg(&hidden_i32);
        builder.arg(&eps);

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1), // 1 token in decode
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("fused_add_rms_norm launch: {e}")))
    }

    fn launch_fused_silu_mul(
        &self,
        gate: &impl cudarc::driver::DeviceSlice<half::f16>,
        up: &impl cudarc::driver::DeviceSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        n: usize,
    ) -> candle_core::Result<()> {
        let func = self.device.get_or_load_custom_func(
            "fused_silu_mul_f16",
            "fused_silu_mul",
            ptx::FUSED_SILU_MUL,
        )?;
        let block_size = 256u32;
        let grid_size = (n as u32 + block_size - 1) / block_size;
        let n_i32 = n as i32;

        let mut builder = func.builder();
        builder.arg(gate);
        builder.arg(up);
        builder.arg(output);
        builder.arg(&n_i32);

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("fused_silu_mul launch: {e}")))
    }

    fn launch_residual_add(
        &self,
        a: &impl cudarc::driver::DeviceSlice<half::f16>,
        b: &impl cudarc::driver::DeviceSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        n: usize,
    ) -> candle_core::Result<()> {
        let func = self.device.get_or_load_custom_func(
            "residual_add_f16",
            "residual_add",
            ptx::RESIDUAL_ADD,
        )?;
        let block_size = 256u32;
        let grid_size = (n as u32 + block_size - 1) / block_size;
        let n_i32 = n as i32;

        let mut builder = func.builder();
        builder.arg(a);
        builder.arg(b);
        builder.arg(output);
        builder.arg(&n_i32);

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("residual_add launch: {e}")))
    }
}
