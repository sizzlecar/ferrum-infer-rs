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
use crate::weight_store::{LayerWeights, TransformerGpuWeights};

// ======================== Runtime Diagnostics ========================
//
// Enable via env vars (checked once at runner init, zero cost when off):
//   FERRUM_DIAG_SHAPES=1   — log tensor shapes at key points per step
//   FERRUM_DIAG_ATTN=1     — log attention details (kv_len, num_splits, etc.)
//   FERRUM_DIAG_TIMING=1   — log per-layer timing breakdown
//   FERRUM_DIAG_NUMERICAL=1 — sync + log first/last few values of key tensors
//   FERRUM_DIAG=1           — enable ALL of the above

/// Runtime diagnostic flags, read once from env vars.
#[derive(Debug, Clone, Copy)]
struct DiagConfig {
    shapes: bool,
    attn: bool,
    timing: bool,
    numerical: bool,
}

impl DiagConfig {
    fn from_env() -> Self {
        let all = std::env::var("FERRUM_DIAG").map_or(false, |v| v == "1");
        Self {
            shapes: all || std::env::var("FERRUM_DIAG_SHAPES").map_or(false, |v| v == "1"),
            attn: all || std::env::var("FERRUM_DIAG_ATTN").map_or(false, |v| v == "1"),
            timing: all || std::env::var("FERRUM_DIAG_TIMING").map_or(false, |v| v == "1"),
            numerical: all || std::env::var("FERRUM_DIAG_NUMERICAL").map_or(false, |v| v == "1"),
        }
    }

    fn any_enabled(self) -> bool {
        self.shapes || self.attn || self.timing || self.numerical
    }
}

/// A single decode request in a batch.
pub struct BatchDecodeRequest<'a> {
    pub token_id: u32,
    pub position: usize,
    pub cache_key: &'a str,
}

/// Per-sequence contiguous KV cache (legacy).
struct SequenceKvState {
    k_caches: Vec<CudaSlice<half::f16>>,
    v_caches: Vec<CudaSlice<half::f16>>,
    current_len: usize,
    max_len: usize,
}

/// Per-sequence paged KV state.
/// KV data lives in the shared `GpuPagedKvPool`; this holds the block table.
struct PagedSequenceKvState {
    /// Block table: logical block → physical block ID.
    block_table_cpu: Vec<i32>,
    /// GPU copy of the block table (re-uploaded when new blocks are allocated).
    block_table_gpu: CudaSlice<i32>,
    /// Current sequence length (number of valid KV tokens).
    current_len: usize,
    /// Whether the GPU block table needs re-upload.
    dirty: bool,
}

/// Merged piecewise CUDA Graphs.
/// Each graph captures all non-attention ops between two attention calls.
/// Layout for N layers:
///   graphs[0]     = pre_attn_0                       (before first attention)
///   graphs[1..N]  = post_attn_{i-1} + pre_attn_i     (between attention calls)
///   graphs[N]     = post_attn_{N-1} + final_norm + lm_head  (after last attention)
/// Total: N+1 graphs instead of 2*N+1, each with ~12-15 kernels.
struct PiecewiseGraphs {
    /// graphs[0] = pre_attn_0
    /// graphs[i] for i in 1..num_layers = post_attn_{i-1} + pre_attn_i
    /// graphs[num_layers] = post_attn_{num_layers-1} + final
    segments: Vec<cudarc::driver::CudaGraph>,
}

unsafe impl Send for PiecewiseGraphs {}
unsafe impl Sync for PiecewiseGraphs {}

/// CUDA decode runner with piecewise graph acceleration.
pub struct CudaDecodeRunner {
    weights: TransformerGpuWeights,
    buffers: DecodeBuffers,
    blas: Arc<CudaBlas>,
    device: CudaDevice,
    stream: Arc<CudaStream>,
    dims: ModelDims,
    /// Contiguous KV states (used when paged KV is off).
    kv_states: HashMap<String, SequenceKvState>,
    /// Paged KV states (used when paged KV is on).
    paged_kv_states: HashMap<String, PagedSequenceKvState>,
    /// Shared GPU block pool for paged KV (None when paged KV is off).
    kv_pool: Option<crate::gpu_paged_kv::GpuPagedKvPool>,
    /// Simple block allocator: next fresh block ID + free list for reuse.
    next_block_id: usize,
    free_blocks: Vec<usize>,
    /// Whether paged KV is enabled (FERRUM_PAGED_KV=1).
    use_paged_kv: bool,
    graphs: Option<PiecewiseGraphs>,
    /// cuBLAS workspace buffer (must stay alive while graphs use it)
    _cublas_workspace: Option<CudaSlice<u8>>,
    warmup_count: usize,
    capture_attempted: bool,
    diag: DiagConfig,
}

impl CudaDecodeRunner {
    /// Create with a pre-created non-blocking stream.
    /// Weights must already be on this stream (via GpuWeight::from_tensor with stream).
    pub fn new(
        weights: TransformerGpuWeights,
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

        let diag = DiagConfig::from_env();
        let use_paged_kv = std::env::var("FERRUM_PAGED_KV").map_or(false, |v| v == "1");

        let (kv_pool, next_block_id) = if use_paged_kv {
            let max_blocks: usize = std::env::var("FERRUM_KV_BLOCKS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1024);
            let block_size: usize = 16;
            let pool = crate::gpu_paged_kv::GpuPagedKvPool::new(
                crate::gpu_paged_kv::GpuPagedKvConfig {
                    block_size,
                    max_blocks,
                    num_kv_heads: dims.num_kv_heads,
                    head_dim: dims.head_dim,
                    num_layers: dims.num_layers,
                },
                stream.clone(),
            )
            .map_err(|e| candle_core::Error::Msg(format!("GpuPagedKvPool: {e}")))?;
            tracing::warn!(
                "Paged KV enabled: {} blocks × {} tok/block = {}MB pool",
                max_blocks,
                block_size,
                pool.memory_bytes() / (1024 * 1024),
            );
            (Some(pool), 0usize)
        } else {
            (None, 0)
        };

        tracing::warn!(
            "CudaDecodeRunner initialized: {}MB decode buffers, {} layers, paged_kv={}{}",
            buffers.memory_bytes() / (1024 * 1024),
            dims.num_layers,
            use_paged_kv,
            if diag.any_enabled() {
                format!(", diag={diag:?}")
            } else {
                String::new()
            },
        );

        Ok(Self {
            weights,
            buffers,
            blas,
            device,
            stream,
            dims,
            kv_states: HashMap::new(),
            paged_kv_states: HashMap::new(),
            kv_pool,
            next_block_id,
            free_blocks: Vec::new(),
            use_paged_kv,
            graphs: None,
            _cublas_workspace: None,
            warmup_count: 0,
            capture_attempted: false,
            diag,
        })
    }

    /// Access weight layers (diagnostic only).
    pub fn weight_layers(&self) -> &[LayerWeights] {
        &self.weights.layers
    }

    // ======================== TP Sub-Phase Methods ========================
    //
    // These expose individual pipeline stages for TpDecodeGroup to orchestrate.
    // The full decode_step calls them in sequence; TP inserts all-reduce between
    // o_proj and post_attn_norm, and between down_proj and post_mlp_norm.

    /// TP Phase: embed token into residual buffer.
    pub(crate) fn tp_embed(&mut self, token_id: u32) -> candle_core::Result<()> {
        self.embed_eager(token_id)
    }

    /// TP Phase: first layer's input norm (only for layer 0).
    pub(crate) fn tp_first_norm(&mut self) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let eps = 1e-6f32;
        Self::launch_rms_norm(
            &self.device,
            &self.stream,
            &self.buffers.residual,
            &self.weights.layers[0].input_ln_w.slice,
            &mut self.buffers.norm_out,
            h,
            eps,
        )
    }

    /// TP Phase: QKV projection + Q/K norm + RoPE + attention for one layer.
    /// After this, attn_out contains the attention output.
    pub(crate) fn tp_pre_o_proj(
        &mut self,
        li: usize,
        position: usize,
        cache_key: &str,
    ) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let hd = self.dims.head_dim;
        let half_dim = hd / 2;
        let eps = 1e-6f32;

        let lw = &self.weights.layers[li];

        // QKV GEMM
        Self::linear(
            &self.blas,
            &self.device,
            &self.stream,
            &self.buffers.norm_out,
            &lw.qkv_w,
            &mut self.buffers.qkv_out,
            &mut self.buffers.temp_dequant,
            1,
            qkv_dim as i32,
            h as i32,
        )?;

        // Q/K norm
        let q = self.buffers.qkv_out.slice(..q_dim);
        let k = self.buffers.qkv_out.slice(q_dim..q_dim + kv_dim);
        if let Some(ref qnw) = lw.q_norm_w {
            Self::launch_rms_norm_view(
                &self.device,
                &self.stream,
                &q,
                &qnw.slice,
                &mut self.buffers.rope_q_temp,
                hd,
                eps,
            )?;
        } else {
            self.stream
                .memcpy_dtod(&q, &mut self.buffers.rope_q_temp)
                .map_err(|e| candle_core::Error::Msg(format!("q copy: {e}")))?;
        }
        if let Some(ref knw) = lw.k_norm_w {
            Self::launch_rms_norm_view(
                &self.device,
                &self.stream,
                &k,
                &knw.slice,
                &mut self.buffers.rope_k_temp,
                hd,
                eps,
            )?;
        } else {
            self.stream
                .memcpy_dtod(&k, &mut self.buffers.rope_k_temp)
                .map_err(|e| candle_core::Error::Msg(format!("k copy: {e}")))?;
        }

        // RoPE
        let co = position * half_dim;
        let cos = self.weights.rope_cos.slice.slice(co..co + half_dim);
        let sin = self.weights.rope_sin.slice.slice(co..co + half_dim);
        Self::launch_rope(
            &self.device,
            &self.stream,
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

        // Attention (contiguous KV only for TP — paged KV is separate concern)
        let mut kv = self
            .kv_states
            .remove(cache_key)
            .ok_or_else(|| candle_core::Error::Msg(format!("No KV cache: {cache_key}")))?;
        self.attention_eager(li, &mut kv)?;
        self.kv_states.insert(cache_key.to_string(), kv);

        Ok(())
    }

    /// TP Phase: O projection GEMM only. Output in o_proj_out.
    /// After this, TpDecodeGroup calls all_reduce(o_proj_out).
    pub(crate) fn tp_o_proj(&mut self, li: usize) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let lw = &self.weights.layers[li];
        Self::linear(
            &self.blas,
            &self.device,
            &self.stream,
            &self.buffers.attn_out,
            &lw.o_w,
            &mut self.buffers.o_proj_out,
            &mut self.buffers.temp_dequant,
            1,
            h as i32,
            q_dim as i32,
        )
    }

    /// TP Phase: post-attention residual add + norm (after all-reduce of o_proj_out).
    pub(crate) fn tp_post_attn_norm(&mut self, li: usize) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let eps = 1e-6f32;
        let lw = &self.weights.layers[li];
        Self::launch_fused_add_rms_norm(
            &self.device,
            &self.stream,
            &self.buffers.o_proj_out,
            &self.buffers.residual,
            &lw.post_ln_w.slice,
            &mut self.buffers.post_norm_out,
            &mut self.buffers.post_norm_residual,
            h,
            eps,
        )
    }

    /// TP Phase: MLP GEMMs (gate_up + SiLU + down). Output in down_out.
    /// After this, TpDecodeGroup calls all_reduce(down_out).
    pub(crate) fn tp_mlp(&mut self, li: usize) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let inter = self.dims.intermediate_size;
        let lw = &self.weights.layers[li];

        // gate_up GEMM
        Self::linear(
            &self.blas,
            &self.device,
            &self.stream,
            &self.buffers.post_norm_out,
            &lw.gate_up_w,
            &mut self.buffers.gate_up_out,
            &mut self.buffers.temp_dequant,
            1,
            (2 * inter) as i32,
            h as i32,
        )?;

        // SiLU * mul
        let g = self.buffers.gate_up_out.slice(..inter);
        let u = self.buffers.gate_up_out.slice(inter..2 * inter);
        Self::launch_fused_silu_mul(
            &self.device,
            &self.stream,
            &g,
            &u,
            &mut self.buffers.mlp_act,
            inter,
        )?;

        // down GEMM
        Self::linear(
            &self.blas,
            &self.device,
            &self.stream,
            &self.buffers.mlp_act,
            &lw.down_w,
            &mut self.buffers.down_out,
            &mut self.buffers.temp_dequant,
            1,
            h as i32,
            inter as i32,
        )
    }

    /// TP Phase: post-MLP residual add + next layer norm (after all-reduce of down_out).
    pub(crate) fn tp_post_mlp_norm(&mut self, li: usize) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        let n = self.dims.num_layers;
        let eps = 1e-6f32;

        if li < n - 1 {
            Self::launch_fused_add_rms_norm(
                &self.device,
                &self.stream,
                &self.buffers.down_out,
                &self.buffers.post_norm_residual,
                &self.weights.layers[li + 1].input_ln_w.slice,
                &mut self.buffers.norm_out,
                &mut self.buffers.residual,
                h,
                eps,
            )
        } else {
            Self::launch_fused_add_rms_norm(
                &self.device,
                &self.stream,
                &self.buffers.down_out,
                &self.buffers.post_norm_residual,
                &self.weights.final_norm_w.slice,
                &mut self.buffers.final_norm_out,
                &mut self.buffers.residual,
                h,
                eps,
            )
        }
    }

    /// TP Phase: LM head GEMM. Output in logits buffer.
    pub(crate) fn tp_lm_head(&mut self) -> candle_core::Result<()> {
        let h = self.dims.hidden_size;
        Self::linear(
            &self.blas,
            &self.device,
            &self.stream,
            &self.buffers.final_norm_out,
            &self.weights.lm_head_w,
            &mut self.buffers.logits,
            &mut self.buffers.temp_dequant,
            1,
            self.dims.vocab_size as i32,
            h as i32,
        )
    }

    /// Access o_proj_out buffer for all-reduce.
    pub(crate) fn o_proj_out_mut(&mut self) -> &mut CudaSlice<half::f16> {
        &mut self.buffers.o_proj_out
    }

    /// Access down_out buffer for all-reduce.
    pub(crate) fn down_out_mut(&mut self) -> &mut CudaSlice<half::f16> {
        &mut self.buffers.down_out
    }

    /// Sync runner's stream.
    pub(crate) fn sync_stream(&self) -> candle_core::Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("stream sync: {e}")))
    }

    /// Clone logits from runner's buffer.
    pub(crate) fn clone_logits(&self) -> candle_core::Result<CudaSlice<half::f16>> {
        self.stream
            .clone_dtod(&self.buffers.logits)
            .map_err(|e| candle_core::Error::Msg(format!("logits: {e}")))
    }

    pub fn init_kv_cache(
        &mut self,
        cache_key: &str,
        kv_data: Vec<(CudaSlice<half::f16>, CudaSlice<half::f16>)>,
        prefill_len: usize,
        max_len: usize,
    ) -> candle_core::Result<()> {
        if self.use_paged_kv {
            return self.init_kv_cache_paged(cache_key, kv_data, prefill_len);
        }
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

    /// Initialize paged KV cache from prefill's contiguous KV tensors.
    /// Allocates blocks from the pool and bulk-copies the data.
    pub fn init_kv_cache_paged(
        &mut self,
        cache_key: &str,
        kv_data: Vec<(CudaSlice<half::f16>, CudaSlice<half::f16>)>,
        prefill_len: usize,
    ) -> candle_core::Result<()> {
        let (bs, max_blk) = {
            let pool = self
                .kv_pool
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("paged KV not enabled".into()))?;
            (pool.block_size(), pool.max_blocks())
        };
        let num_blocks_needed = (prefill_len + bs - 1) / bs;

        // Allocate physical blocks (reuse free blocks first)
        let mut block_table = Vec::with_capacity(num_blocks_needed);
        for _ in 0..num_blocks_needed {
            let block_id = self.alloc_block(max_blk)?;
            block_table.push(block_id as i32);
        }

        // Bulk copy contiguous KV → paged blocks, per layer
        let pool = self
            .kv_pool
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("paged KV not enabled".into()))?;
        for (li, (k_cont, v_cont)) in kv_data.iter().enumerate() {
            pool.copy_contiguous_to_paged(li, k_cont, v_cont, prefill_len, &block_table)
                .map_err(|e| candle_core::Error::Msg(format!("paged copy L{li}: {e}")))?;
        }

        // Upload block table to GPU
        let block_table_gpu = pool
            .upload_block_table(&block_table)
            .map_err(|e| candle_core::Error::Msg(format!("block table upload: {e}")))?;

        tracing::warn!(
            "init_kv_cache_paged key={cache_key} prefill={prefill_len} \
             blocks={num_blocks_needed} ids={block_table:?} free_list={} next_id={}",
            self.free_blocks.len(),
            self.next_block_id,
        );

        self.paged_kv_states.insert(
            cache_key.to_string(),
            PagedSequenceKvState {
                block_table_cpu: block_table,
                block_table_gpu,
                current_len: prefill_len,
                dirty: false,
            },
        );
        Ok(())
    }

    pub fn has_kv_cache(&self, cache_key: &str) -> bool {
        self.kv_states.contains_key(cache_key) || self.paged_kv_states.contains_key(cache_key)
    }

    pub fn release_kv_cache(&mut self, cache_key: &str) {
        let had_contiguous = self.kv_states.remove(cache_key).is_some();
        let had_paged = self.paged_kv_states.contains_key(cache_key);
        // Return freed paged blocks to the free list for reuse
        if let Some(paged) = self.paged_kv_states.remove(cache_key) {
            let freed = paged.block_table_cpu.len();
            for &block_id in &paged.block_table_cpu {
                self.free_blocks.push(block_id as usize);
            }
            tracing::warn!(
                "release_kv_cache(paged) key={cache_key} freed={freed} blocks, \
                 free_list={}, next_id={}",
                self.free_blocks.len(),
                self.next_block_id,
            );
        } else {
            tracing::warn!(
                "release_kv_cache key={cache_key} cont={had_contiguous} paged={had_paged} \
                 (nothing in paged_kv_states)",
            );
        }
    }

    /// Allocate a physical block: reuse from free list, else bump allocator.
    fn alloc_block(&mut self, max_blocks: usize) -> candle_core::Result<usize> {
        if let Some(id) = self.free_blocks.pop() {
            Ok(id)
        } else if self.next_block_id < max_blocks {
            let id = self.next_block_id;
            self.next_block_id += 1;
            Ok(id)
        } else {
            Err(candle_core::Error::Msg("KV block pool exhausted".into()))
        }
    }

    // ======================== Linear dispatch ========================

    /// Dispatch linear projection: FP16 cuBLAS or INT4 dequant+cuBLAS.
    fn linear(
        blas: &std::sync::Arc<cudarc::cublas::CudaBlas>,
        device: &CudaDevice,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        input: &CudaSlice<half::f16>,
        weight: &crate::weight_store::LinearWeight,
        output: &mut CudaSlice<half::f16>,
        temp: &mut Option<CudaSlice<half::f16>>,
        m: i32,
        n: i32,
        k: i32,
    ) -> candle_core::Result<()> {
        match weight {
            crate::weight_store::LinearWeight::Fp16(w) => {
                crate::cublas::linear_f16(blas, input, &w.slice, output, m, n, k)
            }
            crate::weight_store::LinearWeight::Int4(qw) => {
                let t = temp.as_mut().ok_or_else(|| {
                    candle_core::Error::Msg("INT4 requires temp_dequant buffer".into())
                })?;
                crate::quant::dequant_int4(device, qw, t)?;
                crate::cublas::linear_f16(blas, input, t, output, m, n, k)
            }
            crate::weight_store::LinearWeight::Marlin(mw) => {
                crate::marlin::marlin_gemm(stream, input, mw, output, m)
            }
        }
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
            &self.stream,
            &self.buffers.residual,
            &lw.input_ln_w.slice,
            &mut self.buffers.norm_out,
            h,
            eps,
        )?;
        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.norm_out,
            lw.qkv_w.as_fp16(),
            &mut self.buffers.qkv_out,
            1,
            qkv_dim as i32,
            h as i32,
        )?;

        let q = self.buffers.qkv_out.slice(..q_dim);
        let k = self.buffers.qkv_out.slice(q_dim..q_dim + kv_dim);
        // Q/K normalization (Qwen3 has this, Llama doesn't)
        if let Some(ref qnw) = lw.q_norm_w {
            Self::launch_rms_norm_view(
                &self.device,
                &self.stream,
                &q,
                &qnw.slice,
                &mut self.buffers.rope_q_temp,
                hd,
                eps,
            )?;
        } else {
            self.stream
                .memcpy_dtod(&q, &mut self.buffers.rope_q_temp)
                .map_err(|e| candle_core::Error::Msg(format!("q copy: {e}")))?;
        }
        if let Some(ref knw) = lw.k_norm_w {
            Self::launch_rms_norm_view(
                &self.device,
                &self.stream,
                &k,
                &knw.slice,
                &mut self.buffers.rope_k_temp,
                hd,
                eps,
            )?;
        } else {
            self.stream
                .memcpy_dtod(&k, &mut self.buffers.rope_k_temp)
                .map_err(|e| candle_core::Error::Msg(format!("k copy: {e}")))?;
        }

        let co = position * half_dim;
        let cos = self.weights.rope_cos.slice.slice(co..co + half_dim);
        let sin = self.weights.rope_sin.slice.slice(co..co + half_dim);
        Self::launch_rope(
            &self.device,
            &self.stream,
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
        let valid_kv = kv.current_len + 1;
        let num_splits = Self::compute_num_splits(valid_kv);

        if self.diag.attn && li == 0 {
            tracing::info!(
                "[diag:attn] layer=0 valid_kv={valid_kv} num_splits={num_splits} \
                 nq={nq} nkv={nkv} hd={hd} scale={scale:.4}",
            );
        }

        if num_splits <= 1 {
            // Short KV: use original single-block kernel (no Phase 2 overhead)
            Self::launch_decode_attention(
                &self.device,
                &self.stream,
                &self.buffers.q_rotated,
                &kv.k_caches[li],
                &kv.v_caches[li],
                &mut self.buffers.attn_out,
                nq,
                nkv,
                hd,
                kv.max_len,
                valid_kv,
                scale,
            )
        } else {
            // Long KV: flash decode with split-K
            Self::launch_flash_decode_attention(
                &self.device,
                &self.stream,
                &self.buffers.q_rotated,
                &kv.k_caches[li],
                &kv.v_caches[li],
                &mut self.buffers.flash_partial_out,
                &mut self.buffers.flash_partial_m,
                &mut self.buffers.flash_partial_l,
                &mut self.buffers.attn_out,
                nq,
                nkv,
                hd,
                valid_kv,
                scale,
                num_splits,
            )
        }
    }

    /// Paged attention: append K/V to block pool, run paged kernel.
    fn attention_paged(
        &mut self,
        li: usize,
        paged: &mut PagedSequenceKvState,
    ) -> candle_core::Result<()> {
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let hd = self.dims.head_dim;
        let nq = self.dims.num_attention_heads;
        let nkv = self.dims.num_kv_heads;

        let (bs, max_blk) = {
            let pool = self
                .kv_pool
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("paged KV not enabled".into()))?;
            (pool.block_size(), pool.max_blocks())
        };

        // Allocate new block only if this logical block doesn't exist yet.
        // attention_paged is called once per layer, but all layers share the
        // same block table — only the first layer should allocate.
        let logical_block = paged.current_len / bs;
        let slot = paged.current_len % bs;
        if logical_block >= paged.block_table_cpu.len() {
            let block_id = self.alloc_block(max_blk)?;
            paged.block_table_cpu.push(block_id as i32);
            paged.dirty = true;
        }
        let physical_block = paged.block_table_cpu[logical_block] as usize;

        // Append K to pool
        let pool = self
            .kv_pool
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("paged KV not enabled".into()))?;
        let ks = self.buffers.k_rotated.slice(..kv_dim);
        pool.write_k_token(li, physical_block, slot, &ks)
            .map_err(|e| candle_core::Error::Msg(format!("paged K write: {e}")))?;

        // Append V to pool
        let qkv_dim = nq * hd + 2 * kv_dim;
        let vs = self.buffers.qkv_out.slice(nq * hd + kv_dim..qkv_dim);
        pool.write_v_token(li, physical_block, slot, &vs)
            .map_err(|e| candle_core::Error::Msg(format!("paged V write: {e}")))?;

        // Re-upload block table if dirty
        if paged.dirty {
            paged.block_table_gpu = pool
                .upload_block_table(&paged.block_table_cpu)
                .map_err(|e| candle_core::Error::Msg(format!("block table: {e}")))?;
            paged.dirty = false;
        }

        let valid_kv = paged.current_len + 1;
        let scale = 1.0f32 / (hd as f32).sqrt();

        if self.diag.attn && li == 0 {
            tracing::info!(
                "[diag:attn] paged layer=0 valid_kv={valid_kv} blocks={} bs={bs}",
                paged.block_table_cpu.len(),
            );
        }

        // Launch paged attention kernel
        Self::launch_paged_decode_attention(
            &self.device,
            &self.stream,
            &self.buffers.q_rotated,
            pool.k_pool(li),
            pool.v_pool(li),
            &paged.block_table_gpu,
            &mut self.buffers.attn_out,
            nq,
            nkv,
            hd,
            valid_kv,
            bs,
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
            lw.o_w.as_fp16(),
            &mut self.buffers.o_proj_out,
            1,
            h as i32,
            q_dim as i32,
        )?;
        Self::launch_fused_add_rms_norm(
            &self.device,
            &self.stream,
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
            lw.gate_up_w.as_fp16(),
            &mut self.buffers.gate_up_out,
            1,
            (2 * inter) as i32,
            h as i32,
        )?;
        let g = self.buffers.gate_up_out.slice(..inter);
        let u = self.buffers.gate_up_out.slice(inter..2 * inter);
        Self::launch_fused_silu_mul(
            &self.device,
            &self.stream,
            &g,
            &u,
            &mut self.buffers.mlp_act,
            inter,
        )?;

        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.mlp_act,
            lw.down_w.as_fp16(),
            &mut self.buffers.down_out,
            1,
            h as i32,
            inter as i32,
        )?;
        Self::launch_residual_add(
            &self.device,
            &self.stream,
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
            &self.stream,
            &self.buffers.residual,
            &self.weights.final_norm_w.slice,
            &mut self.buffers.final_norm_out,
            h,
            1e-6f32,
        )?;
        crate::cublas::linear_f16(
            &self.blas,
            &self.buffers.final_norm_out,
            self.weights.lm_head_w.as_fp16(),
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

        // Pre-allocate cuBLAS workspace BEFORE capture.
        // cuBLAS internally calls cudaMalloc for workspace during GEMM, which is
        // NOT capture-safe. By setting a pre-allocated workspace, cuBLAS uses it
        // instead of allocating dynamically during graph capture.
        // Pre-allocate cuBLAS workspace BEFORE capture.
        // cuBLAS internally calls cudaMalloc for workspace during GEMM, which is
        // NOT capture-safe. By setting a pre-allocated workspace via
        // cublasSetWorkspace_v2, cuBLAS uses it instead of allocating dynamically.
        let ws_size: usize = 32 * 1024 * 1024; // 32MB
        let ws_buf = unsafe {
            self.stream
                .alloc::<u8>(ws_size)
                .map_err(|e| candle_core::Error::Msg(format!("cublas ws alloc: {e}")))?
        };
        {
            use cudarc::driver::DevicePtr;
            let (ws_ptr, _guard) = ws_buf.device_ptr(&self.stream);
            unsafe {
                let status = cudarc::cublas::sys::cublasSetWorkspace_v2(
                    *self.blas.handle(),
                    ws_ptr as *mut std::ffi::c_void,
                    ws_size,
                );
                if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(candle_core::Error::Msg(format!(
                        "cublasSetWorkspace failed: {status:?}"
                    )));
                }
            }
        }
        self._cublas_workspace = Some(ws_buf);
        tracing::info!(
            "cuBLAS workspace pre-allocated: {}MB",
            ws_size / (1024 * 1024)
        );

        // Disable cudarc event tracking during capture.
        // cudarc's DevicePtr::device_ptr_mut() calls stream.wait(event) when
        // event tracking is on, which is not capture-safe. Since all our
        // buffers and weights are on the same stream, we don't need
        // cross-stream event synchronization.
        unsafe {
            self.stream.context().disable_event_tracking();
        }

        let n = self.dims.num_layers;
        let mut segments = Vec::with_capacity(n + 1);

        // Segment 0: pre_attn_0 only
        let g = self.capture_one("seg0_pre0", |s| s.pre_attention_eager(0, 0))?;
        segments.push(g);

        // Segments 1..N-1: post_attn_{i-1} + pre_attn_i (merged across layer boundary)
        for li in 1..n {
            let g = self.capture_one(&format!("seg{li}"), |s| {
                s.post_attention_eager(li - 1)?;
                s.pre_attention_eager(li, 0)
            })?;
            segments.push(g);
        }

        // Segment N: post_attn_{N-1} + final_norm + lm_head
        let g = self.capture_one(&format!("seg{n}_final"), |s| {
            s.post_attention_eager(n - 1)?;
            s.final_eager()
        })?;
        segments.push(g);

        // Re-enable event tracking after capture
        unsafe {
            self.stream.context().enable_event_tracking();
        }

        tracing::info!(
            "Merged piecewise graphs captured: {} segments ({} layers)",
            segments.len(),
            n,
        );
        Ok(())
    }

    // ======================== Public API ========================

    /// Fused decode step with double-buffered residual and cross-layer norm fusion.
    ///
    /// Optimization over the naive per-layer loop:
    /// - Double-buffer residual: `residual` (res_a) and `post_norm_residual` (res_b)
    ///   alternate, eliminating all residual memcpy operations.
    /// - Cross-layer fusion: each layer's MLP residual-add is fused with the next
    ///   layer's input RMS norm via `fused_add_rms_norm`, eliminating separate
    ///   `residual_add` and `rms_norm` kernel launches.
    /// - Last layer fuses MLP residual-add with final norm.
    ///
    /// Net savings: 108 fewer kernel launches per token (580 → 472 for 36-layer model).
    pub fn decode_step(
        &mut self,
        token_id: u32,
        position: usize,
        cache_key: &str,
    ) -> candle_core::Result<CudaSlice<half::f16>> {
        let n = self.dims.num_layers;
        let h = self.dims.hidden_size;
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let hd = self.dims.head_dim;
        let half_dim = hd / 2;
        let inter = self.dims.intermediate_size;
        let eps = 1e-6f32;

        let step_start = if self.diag.timing {
            self.stream.synchronize().ok();
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Diagnostic: log first step's embed values
        if position <= 2 {
            self.stream.synchronize().ok();
            eprintln!(
                "[runner] q_norm_w={}",
                self.weights.layers[0].q_norm_w.is_some()
            );
        }

        self.embed_eager(token_id)?;

        // Extract KV state (contiguous or paged)
        let mut kv_cont = if !self.use_paged_kv {
            Some(
                self.kv_states
                    .remove(cache_key)
                    .ok_or_else(|| candle_core::Error::Msg(format!("No KV cache: {cache_key}")))?,
            )
        } else {
            None
        };
        let mut kv_paged = if self.use_paged_kv {
            Some(self.paged_kv_states.remove(cache_key).ok_or_else(|| {
                candle_core::Error::Msg(format!("No paged KV cache: {cache_key}"))
            })?)
        } else {
            None
        };

        if self.diag.shapes {
            let kv_len = kv_cont.as_ref().map_or_else(
                || kv_paged.as_ref().map_or(0, |p| p.current_len),
                |c| c.current_len,
            );
            tracing::info!(
                "[diag:shapes] decode_step token={token_id} pos={position} kv_len={kv_len} paged={}",
                self.use_paged_kv,
            );
        }

        // First layer: standalone rms_norm (subsequent layers get norm from
        // previous layer's exit fused_add_rms_norm)
        Self::launch_rms_norm(
            &self.device,
            &self.stream,
            &self.buffers.residual,
            &self.weights.layers[0].input_ln_w.slice,
            &mut self.buffers.norm_out,
            h,
            eps,
        )?;

        for li in 0..n {
            // ---- QKV projection (norm_out already computed) ----
            {
                let lw = &self.weights.layers[li];
                Self::linear(
                    &self.blas,
                    &self.device,
                    &self.stream,
                    &self.buffers.norm_out,
                    &lw.qkv_w,
                    &mut self.buffers.qkv_out,
                    &mut self.buffers.temp_dequant,
                    1,
                    qkv_dim as i32,
                    h as i32,
                )?;

                // Q/K norm
                let q = self.buffers.qkv_out.slice(..q_dim);
                let k = self.buffers.qkv_out.slice(q_dim..q_dim + kv_dim);
                if let Some(ref qnw) = lw.q_norm_w {
                    Self::launch_rms_norm_view(
                        &self.device,
                        &self.stream,
                        &q,
                        &qnw.slice,
                        &mut self.buffers.rope_q_temp,
                        hd,
                        eps,
                    )?;
                } else {
                    self.stream
                        .memcpy_dtod(&q, &mut self.buffers.rope_q_temp)
                        .map_err(|e| candle_core::Error::Msg(format!("q copy: {e}")))?;
                }
                if let Some(ref knw) = lw.k_norm_w {
                    Self::launch_rms_norm_view(
                        &self.device,
                        &self.stream,
                        &k,
                        &knw.slice,
                        &mut self.buffers.rope_k_temp,
                        hd,
                        eps,
                    )?;
                } else {
                    self.stream
                        .memcpy_dtod(&k, &mut self.buffers.rope_k_temp)
                        .map_err(|e| candle_core::Error::Msg(format!("k copy: {e}")))?;
                }

                // RoPE
                let co = position * half_dim;
                let cos = self.weights.rope_cos.slice.slice(co..co + half_dim);
                let sin = self.weights.rope_sin.slice.slice(co..co + half_dim);
                Self::launch_rope(
                    &self.device,
                    &self.stream,
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
            }

            // ---- Attention (contiguous or paged) ----
            if let Some(ref mut kv) = kv_cont {
                self.attention_eager(li, kv)?;
            } else if let Some(ref mut paged) = kv_paged {
                self.attention_paged(li, paged)?;
            }

            // ---- O proj + post-attn residual/norm + MLP + layer exit ----
            {
                let lw = &self.weights.layers[li];

                // O projection
                Self::linear(
                    &self.blas,
                    &self.device,
                    &self.stream,
                    &self.buffers.attn_out,
                    &lw.o_w,
                    &mut self.buffers.o_proj_out,
                    &mut self.buffers.temp_dequant,
                    1,
                    h as i32,
                    q_dim as i32,
                )?;

                // Post-attention: residual + norm
                // Double-buffer: residual (res_a) → post_norm_residual (res_b)
                Self::launch_fused_add_rms_norm(
                    &self.device,
                    &self.stream,
                    &self.buffers.o_proj_out,
                    &self.buffers.residual,
                    &lw.post_ln_w.slice,
                    &mut self.buffers.post_norm_out,
                    &mut self.buffers.post_norm_residual,
                    h,
                    eps,
                )?;

                // MLP
                Self::linear(
                    &self.blas,
                    &self.device,
                    &self.stream,
                    &self.buffers.post_norm_out,
                    &lw.gate_up_w,
                    &mut self.buffers.gate_up_out,
                    &mut self.buffers.temp_dequant,
                    1,
                    (2 * inter) as i32,
                    h as i32,
                )?;
                let g = self.buffers.gate_up_out.slice(..inter);
                let u = self.buffers.gate_up_out.slice(inter..2 * inter);
                Self::launch_fused_silu_mul(
                    &self.device,
                    &self.stream,
                    &g,
                    &u,
                    &mut self.buffers.mlp_act,
                    inter,
                )?;
                Self::linear(
                    &self.blas,
                    &self.device,
                    &self.stream,
                    &self.buffers.mlp_act,
                    &lw.down_w,
                    &mut self.buffers.down_out,
                    &mut self.buffers.temp_dequant,
                    1,
                    h as i32,
                    inter as i32,
                )?;

                // Layer exit: fuse MLP residual-add with next layer's input norm
                // Double-buffer: post_norm_residual (res_b) → residual (res_a)
                if li < n - 1 {
                    Self::launch_fused_add_rms_norm(
                        &self.device,
                        &self.stream,
                        &self.buffers.down_out,
                        &self.buffers.post_norm_residual,
                        &self.weights.layers[li + 1].input_ln_w.slice,
                        &mut self.buffers.norm_out,
                        &mut self.buffers.residual,
                        h,
                        eps,
                    )?;
                } else {
                    // Last layer: fuse with final norm
                    Self::launch_fused_add_rms_norm(
                        &self.device,
                        &self.stream,
                        &self.buffers.down_out,
                        &self.buffers.post_norm_residual,
                        &self.weights.final_norm_w.slice,
                        &mut self.buffers.final_norm_out,
                        &mut self.buffers.residual,
                        h,
                        eps,
                    )?;
                }
            }
        }

        // Update KV state and put it back
        if let Some(mut kv) = kv_cont {
            kv.current_len += 1;
            let cur_len = kv.current_len;
            self.kv_states.insert(cache_key.to_string(), kv);
            if let Some(t0) = step_start {
                tracing::info!(
                    "[diag:timing] decode_step total={:.2}ms (pos={position}, kv_len={cur_len})",
                    t0.elapsed().as_secs_f64() * 1000.0,
                );
            }
        } else if let Some(mut paged) = kv_paged {
            paged.current_len += 1;
            let cur_len = paged.current_len;
            self.paged_kv_states.insert(cache_key.to_string(), paged);
            if let Some(t0) = step_start {
                tracing::info!(
                    "[diag:timing] decode_step total={:.2}ms (pos={position}, kv_len={cur_len}, paged)",
                    t0.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }

        // LM head only — final norm already computed in loop
        Self::linear(
            &self.blas,
            &self.device,
            &self.stream,
            &self.buffers.final_norm_out,
            &self.weights.lm_head_w,
            &mut self.buffers.logits,
            &mut self.buffers.temp_dequant,
            1,
            self.dims.vocab_size as i32,
            h as i32,
        )?;

        // Synchronize runner stream before returning logits.
        self.stream
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("stream sync: {e}")))?;
        self.stream
            .clone_dtod(&self.buffers.logits)
            .map_err(|e| candle_core::Error::Msg(format!("logits: {e}")))
    }

    /// Batched decode: process multiple sequences in one forward pass.
    ///
    /// GEMMs use m=batch for tensor core utilization.
    /// Q/K norm, RoPE, and attention loop per-item (different positions/KV caches).
    ///
    /// Returns `[batch * vocab_size]` logits.
    pub fn batch_decode_step(
        &mut self,
        requests: &[BatchDecodeRequest<'_>],
    ) -> candle_core::Result<CudaSlice<half::f16>> {
        let batch = requests.len();
        if batch == 0 {
            return Err(candle_core::Error::Msg("empty batch".into()));
        }
        if batch == 1 {
            return self.decode_step(
                requests[0].token_id,
                requests[0].position,
                requests[0].cache_key,
            );
        }
        if batch > self.dims.max_batch_size {
            return Err(candle_core::Error::Msg(format!(
                "batch {batch} > max_batch_size {}",
                self.dims.max_batch_size
            )));
        }

        let n = self.dims.num_layers;
        let h = self.dims.hidden_size;
        let q_dim = self.dims.num_attention_heads * self.dims.head_dim;
        let kv_dim = self.dims.num_kv_heads * self.dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let hd = self.dims.head_dim;
        let inter = self.dims.intermediate_size;
        let eps = 1e-6f32;
        let nq = self.dims.num_attention_heads;
        let nkv = self.dims.num_kv_heads;
        let m = batch as i32;

        let step_start = if self.diag.timing {
            self.stream.synchronize().ok();
            Some(std::time::Instant::now())
        } else {
            None
        };

        if self.diag.shapes {
            let positions: Vec<_> = requests.iter().map(|r| r.position).collect();
            tracing::info!("[diag:shapes] batch_decode batch={batch} positions={positions:?}",);
        }

        // ---- Batched embedding: gather B rows into residual[0..B*H] ----
        for (b, req) in requests.iter().enumerate() {
            let off = (req.token_id as usize) * h;
            let src = self
                .weights
                .embed_table
                .slice
                .try_slice(off..off + h)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("embed OOB: token={}", req.token_id))
                })?;
            let mut dst = self.buffers.residual.slice_mut(b * h..(b + 1) * h);
            self.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| candle_core::Error::Msg(format!("batch embed: {e}")))?;
        }

        // Extract all KV states for the batch.
        // If any is missing, restore already-extracted ones before returning error.
        let mut kv_batch: Vec<SequenceKvState> = Vec::with_capacity(batch);
        for (i, req) in requests.iter().enumerate() {
            match self.kv_states.remove(req.cache_key) {
                Some(kv) => kv_batch.push(kv),
                None => {
                    // Restore already-extracted KV states
                    for (j, kv) in kv_batch.into_iter().enumerate() {
                        self.kv_states.insert(requests[j].cache_key.to_string(), kv);
                    }
                    return Err(candle_core::Error::Msg(format!(
                        "No KV cache: {} (item {i}/{batch})",
                        req.cache_key
                    )));
                }
            }
        }

        // First layer: batched rms_norm (num_rows = B*H / H = B)
        Self::launch_rms_norm(
            &self.device,
            &self.stream,
            &self.buffers.residual,
            &self.weights.layers[0].input_ln_w.slice,
            &mut self.buffers.norm_out,
            h,
            eps,
        )?;

        for li in 0..n {
            // ---- Batched QKV GEMM: [B*H] × [qkv_dim, H]^T → [B*qkv_dim] ----
            {
                let lw = &self.weights.layers[li];
                Self::linear(
                    &self.blas,
                    &self.device,
                    &self.stream,
                    &self.buffers.norm_out,
                    &lw.qkv_w,
                    &mut self.buffers.qkv_out,
                    &mut self.buffers.temp_dequant,
                    m,
                    qkv_dim as i32,
                    h as i32,
                )?;
            }

            // ---- Per-item Q/K norm → batch-offset positions in rope_q/k_temp ----
            {
                let lw = &self.weights.layers[li];
                for b in 0..batch {
                    let q_view = self.buffers.qkv_out.slice(b * qkv_dim..b * qkv_dim + q_dim);
                    let k_view = self
                        .buffers
                        .qkv_out
                        .slice(b * qkv_dim + q_dim..b * qkv_dim + q_dim + kv_dim);

                    // Q norm → rope_q_temp[b*q_dim..(b+1)*q_dim]
                    if let Some(ref qnw) = lw.q_norm_w {
                        let mut q_dst = self
                            .buffers
                            .rope_q_temp
                            .slice_mut(b * q_dim..(b + 1) * q_dim);
                        Self::launch_rms_norm_into(
                            &self.device,
                            &self.stream,
                            &q_view,
                            &qnw.slice,
                            &mut q_dst,
                            hd,
                            eps,
                        )?;
                    } else {
                        let mut q_dst = self
                            .buffers
                            .rope_q_temp
                            .slice_mut(b * q_dim..(b + 1) * q_dim);
                        self.stream
                            .memcpy_dtod(&q_view, &mut q_dst)
                            .map_err(|e| candle_core::Error::Msg(format!("q copy: {e}")))?;
                    }
                    // K norm → rope_k_temp[b*kv_dim..(b+1)*kv_dim]
                    if let Some(ref knw) = lw.k_norm_w {
                        let mut k_dst = self
                            .buffers
                            .rope_k_temp
                            .slice_mut(b * kv_dim..(b + 1) * kv_dim);
                        Self::launch_rms_norm_into(
                            &self.device,
                            &self.stream,
                            &k_view,
                            &knw.slice,
                            &mut k_dst,
                            hd,
                            eps,
                        )?;
                    } else {
                        let mut k_dst = self
                            .buffers
                            .rope_k_temp
                            .slice_mut(b * kv_dim..(b + 1) * kv_dim);
                        self.stream
                            .memcpy_dtod(&k_view, &mut k_dst)
                            .map_err(|e| candle_core::Error::Msg(format!("k copy: {e}")))?;
                    }
                }
            }

            // ---- Batched RoPE: single launch for all items ----
            // Upload positions once on first layer (same across layers).
            if li == 0 {
                let pos_vec: Vec<i32> = requests.iter().map(|r| r.position as i32).collect();
                self.stream
                    .memcpy_htod(&pos_vec, &mut self.buffers.batched_positions)
                    .map_err(|e| candle_core::Error::Msg(format!("positions: {e}")))?;
            }
            Self::launch_batched_rope(
                &self.device,
                &self.stream,
                &self.buffers.rope_q_temp,
                &self.buffers.rope_k_temp,
                &self.weights.rope_cos.slice,
                &self.weights.rope_sin.slice,
                &self.buffers.batched_positions,
                &mut self.buffers.q_rotated,
                &mut self.buffers.k_rotated,
                nq,
                nkv,
                hd,
                batch,
            )?;

            // ---- Per-item KV cache append ----
            let kv_stride = nkv * hd;
            for b in 0..batch {
                let kv = &mut kv_batch[b];
                let kv_off = kv.current_len * kv_stride;
                {
                    let ks = self.buffers.k_rotated.slice(b * kv_dim..(b + 1) * kv_dim);
                    let mut kd = kv.k_caches[li].slice_mut(kv_off..kv_off + kv_dim);
                    self.stream
                        .memcpy_dtod(&ks, &mut kd)
                        .map_err(|e| candle_core::Error::Msg(format!("KV k: {e}")))?;
                }
                {
                    let vs = self
                        .buffers
                        .qkv_out
                        .slice(b * qkv_dim + q_dim + kv_dim..b * qkv_dim + qkv_dim);
                    let mut vd = kv.v_caches[li].slice_mut(kv_off..kv_off + kv_dim);
                    self.stream
                        .memcpy_dtod(&vs, &mut vd)
                        .map_err(|e| candle_core::Error::Msg(format!("KV v: {e}")))?;
                }
            }

            // ---- Attention: batched when all items have short KV, else per-item ----
            let scale = 1.0f32 / (hd as f32).sqrt();
            let max_valid_kv = kv_batch
                .iter()
                .map(|kv| kv.current_len + 1)
                .max()
                .unwrap_or(1);
            let all_short = max_valid_kv <= 256;

            if all_short {
                // Build device pointer arrays for K/V caches
                use cudarc::driver::DevicePtr;
                let mut k_ptrs_host = Vec::with_capacity(batch);
                let mut v_ptrs_host = Vec::with_capacity(batch);
                let mut kv_lens_host = Vec::with_capacity(batch);
                for b in 0..batch {
                    let kv = &kv_batch[b];
                    let (kp, _) = kv.k_caches[li].device_ptr(&self.stream);
                    let (vp, _) = kv.v_caches[li].device_ptr(&self.stream);
                    k_ptrs_host.push(kp);
                    v_ptrs_host.push(vp);
                    kv_lens_host.push((kv.current_len + 1) as i32);
                }
                self.stream
                    .memcpy_htod(&k_ptrs_host, &mut self.buffers.batched_k_ptrs)
                    .map_err(|e| candle_core::Error::Msg(format!("k_ptrs: {e}")))?;
                self.stream
                    .memcpy_htod(&v_ptrs_host, &mut self.buffers.batched_v_ptrs)
                    .map_err(|e| candle_core::Error::Msg(format!("v_ptrs: {e}")))?;
                self.stream
                    .memcpy_htod(&kv_lens_host, &mut self.buffers.batched_kv_lens)
                    .map_err(|e| candle_core::Error::Msg(format!("kv_lens: {e}")))?;

                Self::launch_batched_decode_attention(
                    &self.device,
                    &self.stream,
                    &self.buffers.q_rotated,
                    &self.buffers.batched_k_ptrs,
                    &self.buffers.batched_v_ptrs,
                    &self.buffers.batched_kv_lens,
                    &mut self.buffers.attn_out,
                    batch,
                    nq,
                    nkv,
                    hd,
                    max_valid_kv,
                    scale,
                )?;
            } else {
                // Fallback: per-item attention for long KV (flash decode)
                for b in 0..batch {
                    let kv = &kv_batch[b];
                    let valid_kv = kv.current_len + 1;
                    let num_splits = Self::compute_num_splits(valid_kv);

                    // Q is at q_rotated[b*q_dim..(b+1)*q_dim]
                    let q_view = self.buffers.q_rotated.slice(b * q_dim..(b + 1) * q_dim);

                    if num_splits <= 1 {
                        Self::launch_decode_attention_view(
                            &self.device,
                            &self.stream,
                            &q_view,
                            &kv.k_caches[li],
                            &kv.v_caches[li],
                            &mut self.buffers.scratch_attn,
                            nq,
                            nkv,
                            hd,
                            kv.max_len,
                            valid_kv,
                            scale,
                        )?;
                    } else {
                        Self::launch_flash_decode_attention_view(
                            &self.device,
                            &self.stream,
                            &q_view,
                            &kv.k_caches[li],
                            &kv.v_caches[li],
                            &mut self.buffers.flash_partial_out,
                            &mut self.buffers.flash_partial_m,
                            &mut self.buffers.flash_partial_l,
                            &mut self.buffers.scratch_attn,
                            nq,
                            nkv,
                            hd,
                            valid_kv,
                            scale,
                            num_splits,
                        )?;
                    }
                    // Copy scratch → attn_out[b*q_dim..]
                    {
                        let src = self.buffers.scratch_attn.slice(..q_dim);
                        let mut dst = self.buffers.attn_out.slice_mut(b * q_dim..(b + 1) * q_dim);
                        self.stream
                            .memcpy_dtod(&src, &mut dst)
                            .map_err(|e| candle_core::Error::Msg(format!("attn cp: {e}")))?;
                    }
                }
            }

            // ---- Batched O proj + post-attn residual/norm + MLP + layer exit ----
            {
                let lw = &self.weights.layers[li];

                // O proj: [B*q_dim] → [B*H]
                Self::linear(
                    &self.blas,
                    &self.device,
                    &self.stream,
                    &self.buffers.attn_out,
                    &lw.o_w,
                    &mut self.buffers.o_proj_out,
                    &mut self.buffers.temp_dequant,
                    m,
                    h as i32,
                    q_dim as i32,
                )?;

                // Post-attention: fused_add_rms_norm, grid_dim = (B,)
                Self::launch_fused_add_rms_norm(
                    &self.device,
                    &self.stream,
                    &self.buffers.o_proj_out,
                    &self.buffers.residual,
                    &lw.post_ln_w.slice,
                    &mut self.buffers.post_norm_out,
                    &mut self.buffers.post_norm_residual,
                    h,
                    eps,
                )?;

                // MLP: batched gate+up → silu_mul → down
                Self::linear(
                    &self.blas,
                    &self.device,
                    &self.stream,
                    &self.buffers.post_norm_out,
                    &lw.gate_up_w,
                    &mut self.buffers.gate_up_out,
                    &mut self.buffers.temp_dequant,
                    m,
                    (2 * inter) as i32,
                    h as i32,
                )?;
                // Interleaved silu_mul: gate_up is [B, 2*inter] → mlp_act is [B, inter]
                Self::launch_fused_silu_mul_interleaved(
                    &self.device,
                    &self.stream,
                    &self.buffers.gate_up_out,
                    &mut self.buffers.mlp_act,
                    inter,
                    batch,
                )?;
                Self::linear(
                    &self.blas,
                    &self.device,
                    &self.stream,
                    &self.buffers.mlp_act,
                    &lw.down_w,
                    &mut self.buffers.down_out,
                    &mut self.buffers.temp_dequant,
                    m,
                    h as i32,
                    inter as i32,
                )?;

                // Layer exit: fuse MLP residual-add with next norm
                if li < n - 1 {
                    Self::launch_fused_add_rms_norm(
                        &self.device,
                        &self.stream,
                        &self.buffers.down_out,
                        &self.buffers.post_norm_residual,
                        &self.weights.layers[li + 1].input_ln_w.slice,
                        &mut self.buffers.norm_out,
                        &mut self.buffers.residual,
                        h,
                        eps,
                    )?;
                } else {
                    Self::launch_fused_add_rms_norm(
                        &self.device,
                        &self.stream,
                        &self.buffers.down_out,
                        &self.buffers.post_norm_residual,
                        &self.weights.final_norm_w.slice,
                        &mut self.buffers.final_norm_out,
                        &mut self.buffers.residual,
                        h,
                        eps,
                    )?;
                }
            }
        }

        // Update KV state lengths and put them back
        for (req, mut kv) in requests.iter().zip(kv_batch.into_iter()) {
            kv.current_len += 1;
            self.kv_states.insert(req.cache_key.to_string(), kv);
        }

        // Batched LM head: [B*H] → [B*vocab]
        Self::linear(
            &self.blas,
            &self.device,
            &self.stream,
            &self.buffers.final_norm_out,
            &self.weights.lm_head_w,
            &mut self.buffers.logits,
            &mut self.buffers.temp_dequant,
            m,
            self.dims.vocab_size as i32,
            h as i32,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("stream sync: {e}")))?;

        if let Some(t0) = step_start {
            tracing::info!(
                "[diag:timing] batch_decode total={:.2}ms batch={batch}",
                t0.elapsed().as_secs_f64() * 1000.0,
            );
        }

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
        // Graph capture is architecturally working but currently slower than
        // eager (~100 vs ~109 tok/s) due to per-graph launch overhead.
        // Set WARMUP to usize::MAX to disable graph capture by default.
        // To enable: change to 5.
        const WARMUP: usize = usize::MAX;

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

        // Take graphs out to avoid borrow conflict with &mut self methods
        if let Some(graphs) = self.graphs.take() {
            let n = self.dims.num_layers;
            self.embed_eager(token_id)?;
            let mut kv = self
                .kv_states
                .remove(cache_key)
                .ok_or_else(|| candle_core::Error::Msg(format!("No KV cache: {cache_key}")))?;

            // segments[0] = pre_attn_0
            graphs.segments[0]
                .launch()
                .map_err(|e| candle_core::Error::Msg(format!("seg0: {e}")))?;
            self.attention_eager(0, &mut kv)?;

            // segments[1..N-1] = post_attn_{i-1} + pre_attn_i
            for li in 1..n {
                graphs.segments[li]
                    .launch()
                    .map_err(|e| candle_core::Error::Msg(format!("seg{li}: {e}")))?;
                self.attention_eager(li, &mut kv)?;
            }

            // segments[N] = post_attn_{N-1} + final
            graphs.segments[n]
                .launch()
                .map_err(|e| candle_core::Error::Msg(format!("seg{n}: {e}")))?;

            kv.current_len += 1;
            self.kv_states.insert(cache_key.to_string(), kv);
            self.graphs = Some(graphs);

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
        stream: &Arc<CudaStream>,
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
        let mut b = stream.launch_builder(&func);
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
        stream: &Arc<CudaStream>,
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
        let mut b = stream.launch_builder(&func);
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

    /// RMS norm from CudaView input to CudaViewMut output (for batch-offset writes).
    fn launch_rms_norm_into(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
        input: &CudaView<half::f16>,
        weight: &CudaSlice<half::f16>,
        output: &mut cudarc::driver::CudaViewMut<half::f16>,
        row_size: usize,
        eps: f32,
    ) -> candle_core::Result<()> {
        let num_rows = input.len() / row_size;
        let func = device.get_or_load_custom_func("rms_norm_f16", "rms_norm", ptx::RMS_NORM)?;
        let w = weight.slice(..);
        let rs = row_size as i32;
        let mut b = stream.launch_builder(&func);
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
        stream: &Arc<CudaStream>,
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
        let mut b = stream.launch_builder(&func);
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
        stream: &Arc<CudaStream>,
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
        let mut b = stream.launch_builder(&func);
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
        // Shared: max_kv_len floats for attention scores
        let shared_bytes = (max_kv as u32) * 4;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (nq as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("attn: {e}")))
    }

    /// Decode attention accepting CudaView for Q (for batch fallback with sliced Q).
    #[allow(clippy::too_many_arguments)]
    fn launch_decode_attention_view(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
        q: &CudaView<half::f16>,
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
        let kv = kc.slice(..);
        let vv = vc.slice(..);
        let nqi = nq as i32;
        let nkvi = nkv as i32;
        let hdi = hd as i32;
        let mki = max_kv as i32;
        let vki = valid_kv as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(q);
        b.arg(&kv);
        b.arg(&vv);
        b.arg(output);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&mki);
        b.arg(&vki);
        b.arg(&scale);
        let shared_bytes = (max_kv as u32) * 4;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (nq as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("attn: {e}")))
    }

    /// Flash decode attention accepting CudaView for Q.
    #[allow(clippy::too_many_arguments)]
    fn launch_flash_decode_attention_view(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
        q: &CudaView<half::f16>,
        kc: &CudaSlice<half::f16>,
        vc: &CudaSlice<half::f16>,
        partial_out: &mut CudaSlice<f32>,
        partial_m: &mut CudaSlice<f32>,
        partial_l: &mut CudaSlice<f32>,
        output: &mut CudaSlice<half::f16>,
        nq: usize,
        nkv: usize,
        hd: usize,
        valid_kv: usize,
        scale: f32,
        num_splits: usize,
    ) -> candle_core::Result<()> {
        let func1 = device.get_or_load_custom_func(
            "flash_decode_attn_f16",
            "flash_decode_attention",
            ptx::FLASH_DECODE_ATTENTION,
        )?;
        let kv = kc.slice(..);
        let vv = vc.slice(..);
        let nqi = nq as i32;
        let nkvi = nkv as i32;
        let hdi = hd as i32;
        let vki = valid_kv as i32;
        let nsi = num_splits as i32;
        let chunk_size = (valid_kv + num_splits - 1) / num_splits;
        let shared_bytes = (chunk_size as u32) * 4;

        let mut b1 = stream.launch_builder(&func1);
        b1.arg(q);
        b1.arg(&kv);
        b1.arg(&vv);
        b1.arg(&mut *partial_out);
        b1.arg(&mut *partial_m);
        b1.arg(&mut *partial_l);
        b1.arg(&nqi);
        b1.arg(&nkvi);
        b1.arg(&hdi);
        b1.arg(&vki);
        b1.arg(&scale);
        b1.arg(&nsi);
        unsafe {
            b1.launch(LaunchConfig {
                grid_dim: (nq as u32, num_splits as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("flash_decode phase1: {e}")))?;

        let func2 = device.get_or_load_custom_func(
            "flash_decode_reduce_f16",
            "flash_decode_attention",
            ptx::FLASH_DECODE_ATTENTION,
        )?;
        let po = partial_out.slice(..);
        let pm = partial_m.slice(..);
        let pl = partial_l.slice(..);
        let mut b2 = stream.launch_builder(&func2);
        b2.arg(&po);
        b2.arg(&pm);
        b2.arg(&pl);
        b2.arg(output);
        b2.arg(&hdi);
        b2.arg(&nsi);
        unsafe {
            b2.launch(LaunchConfig {
                grid_dim: (nq as u32, 1, 1),
                block_dim: (hd.min(256) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("flash_decode phase2: {e}")))
    }

    /// Batched decode attention: single launch for all batch items.
    /// Grid: (num_q_heads, batch, 1). Each block handles one (head, item) pair.
    #[allow(clippy::too_many_arguments)]
    fn launch_batched_decode_attention(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<half::f16>,          // [B * nq * hd]
        k_ptrs: &CudaSlice<u64>,           // [B] device pointers
        v_ptrs: &CudaSlice<u64>,           // [B] device pointers
        kv_lens: &CudaSlice<i32>,          // [B] valid kv lengths
        output: &mut CudaSlice<half::f16>, // [B * nq * hd]
        batch: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        max_valid_kv: usize,
        scale: f32,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func(
            "batched_decode_attention_f16",
            "batched_decode_attention",
            ptx::BATCHED_DECODE_ATTENTION,
        )?;
        let qv = q.slice(..);
        let kp = k_ptrs.slice(..);
        let vp = v_ptrs.slice(..);
        let kl = kv_lens.slice(..);
        let nqi = nq as i32;
        let nkvi = nkv as i32;
        let hdi = hd as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kp);
        b.arg(&vp);
        b.arg(output);
        b.arg(&kl);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&scale);
        let shared_bytes = (max_valid_kv as u32) * 4;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (nq as u32, batch as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("batched_attn: {e}")))
    }

    /// Batched RoPE: single launch for all batch items with per-item positions.
    #[allow(clippy::too_many_arguments)]
    fn launch_batched_rope(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<half::f16>,         // [B * nq * hd]
        k: &CudaSlice<half::f16>,         // [B * nk * hd]
        cos_table: &CudaSlice<half::f16>, // full cos table
        sin_table: &CudaSlice<half::f16>, // full sin table
        positions: &CudaSlice<i32>,       // [B]
        q_out: &mut CudaSlice<half::f16>, // [B * nq * hd]
        k_out: &mut CudaSlice<half::f16>, // [B * nk * hd]
        nq: usize,
        nk: usize,
        hd: usize,
        batch: usize,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func("batched_rope_f16", "rope", ptx::ROPE)?;
        let qv = q.slice(..);
        let kv = k.slice(..);
        let cv = cos_table.slice(..);
        let sv = sin_table.slice(..);
        let pv = positions.slice(..);
        let nqi = nq as i32;
        let nki = nk as i32;
        let hdi = hd as i32;
        let bi = batch as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kv);
        b.arg(&cv);
        b.arg(&sv);
        b.arg(q_out);
        b.arg(k_out);
        b.arg(&pv);
        b.arg(&nqi);
        b.arg(&nki);
        b.arg(&hdi);
        b.arg(&bi);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (((nq + nk) * batch) as u32, 1, 1),
                block_dim: ((hd / 2).min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("batched_rope: {e}")))
    }

    /// Determine number of KV splits for flash decoding.
    /// Returns 1 for short KV (uses original kernel), >1 for long KV.
    fn compute_num_splits(valid_kv_len: usize) -> usize {
        use crate::decode_buffers::DecodeBuffers;
        const MIN_KV_PER_SPLIT: usize = 256;
        if valid_kv_len <= MIN_KV_PER_SPLIT {
            return 1;
        }
        (valid_kv_len / MIN_KV_PER_SPLIT)
            .min(DecodeBuffers::MAX_SPLITS)
            .max(2)
    }

    /// Flash Decoding: two-phase split-K attention for long KV sequences.
    #[allow(clippy::too_many_arguments)]
    fn launch_flash_decode_attention(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<half::f16>,
        kc: &CudaSlice<half::f16>,
        vc: &CudaSlice<half::f16>,
        partial_out: &mut CudaSlice<f32>,
        partial_m: &mut CudaSlice<f32>,
        partial_l: &mut CudaSlice<f32>,
        output: &mut CudaSlice<half::f16>,
        nq: usize,
        nkv: usize,
        hd: usize,
        valid_kv: usize,
        scale: f32,
        num_splits: usize,
    ) -> candle_core::Result<()> {
        // Phase 1: split-K attention
        let func1 = device.get_or_load_custom_func(
            "flash_decode_attn_f16",
            "flash_decode_attention",
            ptx::FLASH_DECODE_ATTENTION,
        )?;
        let qv = q.slice(..);
        let kv = kc.slice(..);
        let vv = vc.slice(..);
        let nqi = nq as i32;
        let nkvi = nkv as i32;
        let hdi = hd as i32;
        let vki = valid_kv as i32;
        let nsi = num_splits as i32;
        let chunk_size = (valid_kv + num_splits - 1) / num_splits;
        let shared_bytes = (chunk_size as u32) * 4; // floats for local scores

        let mut b1 = func1.builder();
        b1.arg(&qv);
        b1.arg(&kv);
        b1.arg(&vv);
        b1.arg(&mut *partial_out);
        b1.arg(&mut *partial_m);
        b1.arg(&mut *partial_l);
        b1.arg(&nqi);
        b1.arg(&nkvi);
        b1.arg(&hdi);
        b1.arg(&vki);
        b1.arg(&scale);
        b1.arg(&nsi);
        unsafe {
            b1.launch(LaunchConfig {
                grid_dim: (nq as u32, num_splits as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("flash_decode phase1: {e}")))?;

        // Phase 2: reduce across splits
        let func2 = device.get_or_load_custom_func(
            "flash_decode_reduce_f16",
            "flash_decode_attention",
            ptx::FLASH_DECODE_ATTENTION,
        )?;
        let po = partial_out.slice(..);
        let pm = partial_m.slice(..);
        let pl = partial_l.slice(..);
        let mut b2 = func2.builder();
        b2.arg(&po);
        b2.arg(&pm);
        b2.arg(&pl);
        b2.arg(output);
        b2.arg(&hdi);
        b2.arg(&nsi);
        unsafe {
            b2.launch(LaunchConfig {
                grid_dim: (nq as u32, 1, 1),
                block_dim: (hd.min(256) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("flash_decode phase2: {e}")))
    }

    /// Paged decode attention kernel launcher.
    #[allow(clippy::too_many_arguments)]
    fn launch_paged_decode_attention(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<half::f16>,
        k_pool: &CudaSlice<half::f16>,
        v_pool: &CudaSlice<half::f16>,
        block_table: &CudaSlice<i32>,
        output: &mut CudaSlice<half::f16>,
        nq: usize,
        nkv: usize,
        hd: usize,
        valid_kv: usize,
        block_size: usize,
        scale: f32,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func(
            "paged_decode_attention_f16",
            "paged_decode_attention",
            ptx::PAGED_DECODE_ATTENTION,
        )?;
        let qv = q.slice(..);
        let kp = k_pool.slice(..);
        let vp = v_pool.slice(..);
        let bt = block_table.slice(..);
        let nqi = nq as i32;
        let nkvi = nkv as i32;
        let hdi = hd as i32;
        let vki = valid_kv as i32;
        let bsi = block_size as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kp);
        b.arg(&vp);
        b.arg(&bt);
        b.arg(output);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&vki);
        b.arg(&bsi);
        b.arg(&scale);
        let shared_bytes = (valid_kv as u32) * 4;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (nq as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("paged_attn: {e}")))
    }

    fn launch_fused_add_rms_norm(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
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
        let mut b = stream.launch_builder(&func);
        b.arg(&iv);
        b.arg(&rv);
        b.arg(&wv);
        b.arg(output);
        b.arg(residual_out);
        b.arg(&hi);
        b.arg(&eps);
        let num_tokens = (input.len() / h) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_tokens, 1, 1),
                block_dim: (h.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("fused_add_rms: {e}")))
    }

    fn launch_fused_silu_mul(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
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
        let mut b = stream.launch_builder(&func);
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

    /// Interleaved silu_mul for batched gate+up GEMM output.
    /// gate_up: [batch * 2 * inter], layout [gate_0, up_0, gate_1, up_1, ...]
    /// output:  [batch * inter], layout [act_0, act_1, ...] (contiguous)
    fn launch_fused_silu_mul_interleaved(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
        gate_up: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        inter: usize,
        batch: usize,
    ) -> candle_core::Result<()> {
        let func = device.get_or_load_custom_func(
            "fused_silu_mul_interleaved_f16",
            "fused_silu_mul",
            ptx::FUSED_SILU_MUL,
        )?;
        let gv = gate_up.slice(..);
        let inter_i = inter as i32;
        let total = (batch * inter) as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&gv);
        b.arg(output);
        b.arg(&inter_i);
        b.arg(&total);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: ((((batch * inter) + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("silu_mul_interleaved: {e}")))
    }

    fn launch_residual_add(
        device: &CudaDevice,
        stream: &Arc<CudaStream>,
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
        let mut b = stream.launch_builder(&func);
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
