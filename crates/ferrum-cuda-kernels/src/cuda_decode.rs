//! CUDA decode runner with piecewise CUDA Graphs (vLLM pattern).
//!
//! Captures fixed-shape operations (GEMM, norm, activation) as CUDA Graphs.
//! Attention runs in eager mode because KV cache length varies per step.
//! Pre-allocated buffers eliminate per-op allocation overhead.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::cuda_backend::CudaDevice;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};

use crate::decode_buffers::{DecodeBuffers, ModelDims};
use crate::ptx;
use crate::weight_store::Qwen3Weights;

/// Per-sequence KV cache managed by the decode runner.
struct SequenceKvState {
    k_caches: Vec<CudaSlice<half::f16>>,
    v_caches: Vec<CudaSlice<half::f16>>,
    current_len: usize,
    max_len: usize,
}

/// Piecewise CUDA Graphs (vLLM pattern).
/// Only captures fixed-shape ops; attention runs eager.
struct PiecewiseGraphs {
    pre_attn: Vec<cudarc::driver::CudaGraph>,
    post_attn: Vec<cudarc::driver::CudaGraph>,
    final_graph: Option<cudarc::driver::CudaGraph>,
}

unsafe impl Send for PiecewiseGraphs {}
unsafe impl Sync for PiecewiseGraphs {}

/// CUDA decode runner with piecewise graph acceleration.
pub struct CudaDecodeRunner {
    weights: Qwen3Weights,
    buffers: DecodeBuffers,
    blas: Arc<CudaBlas>,
    device: CudaDevice,
    stream: Arc<CudaStream>,
    dims: ModelDims,
    kv_states: HashMap<String, SequenceKvState>,
    graphs: Option<PiecewiseGraphs>,
    warmup_count: usize,
    capture_attempted: bool,
}

impl CudaDecodeRunner {
    /// Create with a pre-created non-blocking stream.
    /// Weights must already be on this stream (via GpuWeight::from_tensor with stream).
    pub fn new(
        weights: Qwen3Weights,
        dims: ModelDims,
        device: CudaDevice,
        stream: Arc<CudaStream>,
    ) -> candle_core::Result<Self> {
        let blas = Arc::new(
            cudarc::cublas::CudaBlas::new(stream.clone())
                .map_err(|e| candle_core::Error::Msg(format!("cublas new: {e}")))?,
        );
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
            graphs: None,
            warmup_count: 0,
            capture_attempted: false,
        })
    }

    pub fn init_kv_cache(
        &mut self,
        cache_key: &str,
        kv_data: Vec<(CudaSlice<half::f16>, CudaSlice<half::f16>)>,
        prefill_len: usize,
        max_len: usize,
    ) -> candle_core::Result<()> {
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for (k, v) in kv_data {
            ks.push(k);
            vs.push(v);
        }
        self.kv_states.insert(
            cache_key.to_string(),
            SequenceKvState {
                k_caches: ks,
                v_caches: vs,
                current_len: prefill_len,
                max_len,
            },
        );
        Ok(())
    }

    pub fn has_kv_cache(&self, cache_key: &str) -> bool {
        self.kv_states.contains_key(cache_key)
    }

    pub fn release_kv_cache(&mut self, cache_key: &str) {
        self.kv_states.remove(cache_key);
    }

    // ======================== Sub-methods ========================

    fn embed_eager(&mut self, token_id: u32) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let off = (token_id as usize) * h;
        let src = self
            .weights
            .embed_table
            .slice
            .try_slice(off..off + h)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "embed OOB: off={off}, len={}",
                    self.weights.embed_table.slice.len()
                ))
            })?;
        self.stream
            .memcpy_dtod(&src, &mut self.buffers.embed_out)
            .map_err(|e| candle_core::Error::Msg(format!("embed: {e}")))?;
        let v = self.buffers.embed_out.slice(..);
        self.stream
            .memcpy_dtod(&v, &mut self.buffers.residual)
            .map_err(|e| candle_core::Error::Msg(format!("residual init: {e}")))?;
        Ok(())
    }

    fn pre_attention_eager(&mut self, li: usize, position: usize) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let hd = self.dims.head_dim;
        let half_dim = hd / 2;
        let eps = 1e-6f32;
        let lw = &self.weights.layers[li];

        Self::launch_rms_norm(
            &self.device,
            &self.buffers.residual,
            &lw.input_ln_w.slice,
            &mut self.buffers.norm_out,
            h,
            eps,
        )?;
        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.norm_out,
            &lw.qkv_w.slice,
            &mut self.buffers.qkv_out,
            1,
            qkv_dim as i32,
            h as i32,
        )?;

        let q = self.buffers.qkv_out.slice(..q_dim);
        let k = self.buffers.qkv_out.slice(q_dim..q_dim + kv_dim);
        Self::launch_rms_norm_view(
            &self.device,
            &q,
            &lw.q_norm_w.slice,
            &mut self.buffers.rope_q_temp,
            hd,
            eps,
        )?;
        Self::launch_rms_norm_view(
            &self.device,
            &k,
            &lw.k_norm_w.slice,
            &mut self.buffers.rope_k_temp,
            hd,
            eps,
        )?;

        let co = position * half_dim;
        let cos = self.weights.rope_cos.slice.slice(co..co + half_dim);
        let sin = self.weights.rope_sin.slice.slice(co..co + half_dim);
        Self::launch_rope(
            &self.device,
            &self.buffers.rope_q_temp,
            &self.buffers.rope_k_temp,
            &cos,
            &sin,
            &mut self.buffers.q_rotated,
            &mut self.buffers.k_rotated,
            self.dims.num_attention_heads,
            self.dims.num_kv_heads,
            hd,
        )?;
        Ok(())
    }

    fn attention_eager(&mut self, li: usize, kv: &mut SequenceKvState) -> candle_core::Result<()> {
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let hd = self.dims.head_dim;
        let nq = self.dims.num_attention_heads;
        let nkv = self.dims.num_kv_heads;
        let kv_stride = nkv * hd;
        let off = kv.current_len * kv_stride;

        let ks = self.buffers.k_rotated.slice(..kv_dim);
        let mut kd = kv.k_caches[li].slice_mut(off..off + kv_dim);
        self.stream
            .memcpy_dtod(&ks, &mut kd)
            .map_err(|e| candle_core::Error::Msg(format!("KV k: {e}")))?;

        let qkv_dim = nq * hd + 2 * kv_dim;
        let vs = self.buffers.qkv_out.slice(nq * hd + kv_dim..qkv_dim);
        let mut vd = kv.v_caches[li].slice_mut(off..off + kv_dim);
        self.stream
            .memcpy_dtod(&vs, &mut vd)
            .map_err(|e| candle_core::Error::Msg(format!("KV v: {e}")))?;

        let scale = 1.0f32 / (hd as f32).sqrt();
        Self::launch_decode_attention(
            &self.device,
            &self.buffers.q_rotated,
            &kv.k_caches[li],
            &kv.v_caches[li],
            &mut self.buffers.attn_out,
            nq,
            nkv,
            hd,
            kv.max_len,
            kv.current_len + 1,
            scale,
        )
    }

    fn post_attention_eager(&mut self, li: usize) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let inter = self.dims.intermediate_size;
        let eps = 1e-6f32;
        let lw = &self.weights.layers[li];

        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.attn_out,
            &lw.o_w.slice,
            &mut self.buffers.o_proj_out,
            1,
            h as i32,
            q_dim as i32,
        )?;
        Self::launch_fused_add_rms_norm(
            &self.device,
            &self.buffers.o_proj_out,
            &self.buffers.residual,
            &lw.post_ln_w.slice,
            &mut self.buffers.post_norm_out,
            &mut self.buffers.post_norm_residual,
            h,
            eps,
        )?;
        let pnr = self.buffers.post_norm_residual.slice(..h);
        self.stream
            .memcpy_dtod(&pnr, &mut self.buffers.residual)
            .map_err(|e| candle_core::Error::Msg(format!("res: {e}")))?;

        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.post_norm_out,
            &lw.gate_up_w.slice,
            &mut self.buffers.gate_up_out,
            1,
            (2 * inter) as i32,
            h as i32,
        )?;
        let g = self.buffers.gate_up_out.slice(..inter);
        let u = self.buffers.gate_up_out.slice(inter..2 * inter);
        Self::launch_fused_silu_mul(&self.device, &g, &u, &mut self.buffers.mlp_act, inter)?;

        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.mlp_act,
            &lw.down_w.slice,
            &mut self.buffers.down_out,
            1,
            h as i32,
            inter as i32,
        )?;
        Self::launch_residual_add(
            &self.device,
            &self.buffers.residual,
            &self.buffers.down_out,
            &mut self.buffers.norm_out,
            h,
        )?;
        let nv = self.buffers.norm_out.slice(..h);
        self.stream
            .memcpy_dtod(&nv, &mut self.buffers.residual)
            .map_err(|e| candle_core::Error::Msg(format!("res add: {e}")))
    }

    fn final_eager(&mut self) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        Self::launch_rms_norm(
            &self.device,
            &self.buffers.residual,
            &self.weights.final_norm_w.slice,
            &mut self.buffers.final_norm_out,
            h,
            1e-6f32,
        )?;
        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.final_norm_out,
            &self.weights.lm_head_w.slice,
            &mut self.buffers.logits,
            1,
            self.dims.vocab_size as i32,
            h as i32,
        )
    }

    // ======================== Piecewise Graph Capture ========================

    /// Helper: capture a single graph segment. Always calls end_capture even on error.
    fn capture_one<F>(
        &mut self,
        label: &str,
        f: F,
    ) -> candle_core::Result<cudarc::driver::CudaGraph>
    where
        F: FnOnce(&mut Self) -> candle_core::Result<()>,
    {
        let mode = cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
        let flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

        self.stream
            .begin_capture(mode)
            .map_err(|e| candle_core::Error::Msg(format!("{label} begin: {e}")))?;

        let result = f(self);

        // ALWAYS end capture, even if f() failed
        let graph = self.stream.end_capture(flags);

        // Check f() result first
        result?;

        match graph {
            Ok(Some(g)) => {
                g.upload().ok();
                Ok(g)
            }
            Ok(None) => Err(candle_core::Error::Msg(format!("{label} empty"))),
            Err(e) => Err(candle_core::Error::Msg(format!("{label} end: {e}"))),
        }
    }

    fn capture_piecewise_graphs(&mut self) -> candle_core::Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;

        let n = self.dims.num_layers;
        let mut pre = Vec::with_capacity(n);
        let mut post = Vec::with_capacity(n);

        for li in 0..n {
            let g =
                self.capture_one(&format!("pre_attn L{li}"), |s| s.pre_attention_eager(li, 0))?;
            pre.push(g);

            let g =
                self.capture_one(&format!("post_attn L{li}"), |s| s.post_attention_eager(li))?;
            post.push(g);
        }

        let fg = self.capture_one("final", |s| s.final_eager()).ok(); // final graph is optional

        tracing::info!(
            "Piecewise graphs captured: {} pre + {} post + final",
            pre.len(),
            post.len()
        );
        self.graphs = Some(PiecewiseGraphs {
            pre_attn: pre,
            post_attn: post,
            final_graph: fg,
        });
        Ok(())
    }

    // ======================== Public API ========================

    pub fn decode_step(
        &mut self,
        token_id: u32,
        position: usize,
        cache_key: &str,
    ) -> candle_core::Result<CudaSlice<half::f16>> {
        self.embed_eager(token_id)?;
        let mut kv = self
            .kv_states
            .remove(cache_key)
            .ok_or_else(|| candle_core::Error::Msg(format!("No KV cache: {cache_key}")))?;
        for li in 0..self.dims.num_layers {
            self.pre_attention_eager(li, position)?;
            self.attention_eager(li, &mut kv)?;
            self.post_attention_eager(li)?;
        }
        kv.current_len += 1;
        self.kv_states.insert(cache_key.to_string(), kv);
        self.final_eager()?;
        self.stream
            .clone_dtod(&self.buffers.logits)
            .map_err(|e| candle_core::Error::Msg(format!("logits: {e}")))
    }

    pub fn decode_step_graphed(
        &mut self,
        token_id: u32,
        position: usize,
        cache_key: &str,
    ) -> candle_core::Result<CudaSlice<half::f16>> {
        const WARMUP: usize = 5;

        if self.warmup_count < WARMUP {
            self.warmup_count += 1;
            return self.decode_step(token_id, position, cache_key);
        }

        // Try capture once after warmup
        if self.graphs.is_none() && !self.capture_attempted {
            self.capture_attempted = true;
            match self.capture_piecewise_graphs() {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!("Piecewise capture failed: {e}");
                    // Recreate cuBLAS handle — capture failure corrupts its state
                    match cudarc::cublas::CudaBlas::new(self.stream.clone()) {
                        Ok(new_blas) => {
                            self.blas = Arc::new(new_blas);
                            tracing::info!("cuBLAS handle recreated after capture failure");
                        }
                        Err(e2) => {
                            tracing::error!("cuBLAS recreate failed: {e2}");
                        }
                    }
                    self.stream.synchronize().ok();
                }
            }
        }

        // Use graphs if captured, otherwise eager
        if let Some(ref graphs) = self.graphs {
            self.embed_eager(token_id)?;
            let mut kv = self
                .kv_states
                .remove(cache_key)
                .ok_or_else(|| candle_core::Error::Msg(format!("No KV cache: {cache_key}")))?;

            for li in 0..self.dims.num_layers {
                graphs.pre_attn[li]
                    .launch()
                    .map_err(|e| candle_core::Error::Msg(format!("pre L{li}: {e}")))?;
                self.attention_eager(li, &mut kv)?;
                graphs.post_attn[li]
                    .launch()
                    .map_err(|e| candle_core::Error::Msg(format!("post L{li}: {e}")))?;
            }

            kv.current_len += 1;
            self.kv_states.insert(cache_key.to_string(), kv);

            if let Some(ref g) = graphs.final_graph {
                g.launch()
                    .map_err(|e| candle_core::Error::Msg(format!("final: {e}")))?;
            } else {
                self.final_eager()?;
            }

            self.stream
                .clone_dtod(&self.buffers.logits)
                .map_err(|e| candle_core::Error::Msg(format!("logits: {e}")))
        } else {
            self.decode_step(token_id, position, cache_key)
        }
    }

    // ======================== Kernel Launch Helpers ========================

    fn launch_rms_norm(
        device: &CudaDevice,
        input: &CudaSlice<half::f16>,
        weight: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        row_size: usize,
        eps: f32,
    ) -> candle_core::Result<()> {
        let num_rows = input.len() / row_size;
        let func = device.get_or_load_custom_func("rms_norm_f16", "rms_norm", ptx::RMS_NORM)?;
        let inp = input.slice(..);
        let w = weight.slice(..);
        let rs = row_size as i32;
        let mut b = func.builder();
        b.arg(&inp);
        b.arg(&w);
        b.arg(output);
        b.arg(&rs);
        b.arg(&eps);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_rows as u32, 1, 1),
                block_dim: (row_size.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("rms_norm: {e}")))
    }

    fn launch_rms_norm_view(
        device: &CudaDevice,
        input: &CudaView<half::f16>,
        weight: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        row_size: usize,
        eps: f32,
    ) -> candle_core::Result<()> {
        let num_rows = input.len() / row_size;
        let func = device.get_or_load_custom_func("rms_norm_f16", "rms_norm", ptx::RMS_NORM)?;
        let w = weight.slice(..);
        let rs = row_size as i32;
        let mut b = func.builder();
        b.arg(input);
        b.arg(&w);
        b.arg(output);
        b.arg(&rs);
        b.arg(&eps);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_rows as u32, 1, 1),
                block_dim: (row_size.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("rms_norm: {e}")))
    }

    fn launch_rope(
        device: &CudaDevice,
        q: &CudaSlice<half::f16>,
        k: &CudaSlice<half::f16>,
        cos: &CudaView<half::f16>,
        sin: &CudaView<half::f16>,
        q_out: &mut CudaSlice<half::f16>,
        k_out: &mut CudaSlice<half::f16>,
        nq: usize,
        nk: usize,
        hd: usize,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func("rope_f16", "rope", ptx::ROPE)?;
        let qv = q.slice(..);
        let kv = k.slice(..);
        let nqi = nq as i32;
        let nki = nk as i32;
        let hdi = hd as i32;
        let mut b = func.builder();
        b.arg(&qv);
        b.arg(&kv);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(k_out);
        b.arg(&nqi);
        b.arg(&nki);
        b.arg(&hdi);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((nq + nk) as u32, 1, 1),
                block_dim: ((hd / 2).min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("rope: {e}")))
    }

    fn launch_decode_attention(
        device: &CudaDevice,
        q: &CudaSlice<half::f16>,
        kc: &CudaSlice<half::f16>,
        vc: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        nq: usize,
        nkv: usize,
        hd: usize,
        max_kv: usize,
        valid_kv: usize,
        scale: f32,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func(
            "decode_attention_f16",
            "decode_attention",
            ptx::DECODE_ATTENTION,
        )?;
        let qv = q.slice(..);
        let kv = kc.slice(..);
        let vv = vc.slice(..);
        let nqi = nq as i32;
        let nkvi = nkv as i32;
        let hdi = hd as i32;
        let mki = max_kv as i32;
        let vki = valid_kv as i32;
        let mut b = func.builder();
        b.arg(&qv);
        b.arg(&kv);
        b.arg(&vv);
        b.arg(output);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&mki);
        b.arg(&vki);
        b.arg(&scale);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (nq as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (max_kv as u32) * 4,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("attn: {e}")))
    }

    fn launch_fused_add_rms_norm(
        device: &CudaDevice,
        input: &CudaSlice<half::f16>,
        residual: &CudaSlice<half::f16>,
        weight: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        residual_out: &mut CudaSlice<half::f16>,
        h: usize,
        eps: f32,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func(
            "fused_add_rms_norm_f16",
            "fused_add_rms_norm",
            ptx::FUSED_ADD_RMS_NORM,
        )?;
        let iv = input.slice(..);
        let rv = residual.slice(..);
        let wv = weight.slice(..);
        let hi = h as i32;
        let mut b = func.builder();
        b.arg(&iv);
        b.arg(&rv);
        b.arg(&wv);
        b.arg(output);
        b.arg(residual_out);
        b.arg(&hi);
        b.arg(&eps);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (h.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("fused_add_rms: {e}")))
    }

    fn launch_fused_silu_mul(
        device: &CudaDevice,
        gate: &CudaView<half::f16>,
        up: &CudaView<half::f16>,
        output: &mut CudaSlice<half::f16>,
        n: usize,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func(
            "fused_silu_mul_f16",
            "fused_silu_mul",
            ptx::FUSED_SILU_MUL,
        )?;
        let ni = n as i32;
        let mut b = func.builder();
        b.arg(gate);
        b.arg(up);
        b.arg(output);
        b.arg(&ni);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (((n + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("silu_mul: {e}")))
    }

    fn launch_residual_add(
        device: &CudaDevice,
        a: &CudaSlice<half::f16>,
        b_: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        n: usize,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func(
            "residual_add_f16",
            "residual_add",
            ptx::RESIDUAL_ADD,
        )?;
        let av = a.slice(..);
        let bv = b_.slice(..);
        let ni = n as i32;
        let mut b = func.builder();
        b.arg(&av);
        b.arg(&bv);
        b.arg(output);
        b.arg(&ni);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (((n + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("res_add: {e}")))
    }
}
