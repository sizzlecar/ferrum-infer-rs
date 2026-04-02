//! Llama model executor using our custom Llama implementation.
//!
//! Uses GenericKvCacheHandle (like Qwen3) with per-request cache_id.
//! Supports CUDA decode runner for GPU acceleration.

use async_trait::async_trait;
use candle_core::Tensor;
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus,
        MemoryRequirements, PrefillInput, PrefillOutput,
    },
    KvCacheHandle, ModelExecutor, TensorRef,
};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, Result};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tracing::{debug, info};

use super::common::{self, GenericKvCacheHandle};
use crate::architectures::llama::LlamaModelWrapper;
use crate::tensor_wrapper::CandleTensorWrapper;
use parking_lot::Mutex;

struct LlamaCacheState {
    sequence_length: usize,
}

/// Llama model executor
pub struct CandleModelExecutor {
    model: Arc<LlamaModelWrapper>,
    info: ModelInfo,
    states: Mutex<HashMap<String, LlamaCacheState>>,
    next_cache_id: AtomicU64,
    #[cfg(feature = "cuda")]
    cuda_runner: Mutex<Option<ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner>>,
    #[cfg(feature = "tensor-parallel")]
    tp_group: Mutex<Option<ferrum_cuda_kernels::tp_decode::TpDecodeGroup>>,
}

impl CandleModelExecutor {
    pub fn new(model: LlamaModelWrapper, info: ModelInfo) -> Self {
        info!("Created CandleModelExecutor (Llama) for: {}", info.model_id);
        Self {
            model: Arc::new(model),
            info,
            states: Mutex::new(HashMap::new()),
            next_cache_id: AtomicU64::new(1),
            #[cfg(feature = "cuda")]
            cuda_runner: Mutex::new(None),
            #[cfg(feature = "tensor-parallel")]
            tp_group: Mutex::new(None),
        }
    }

    fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<Tensor> {
        Tensor::new(token_ids, self.model.device())
            .map_err(|e| FerrumError::model(format!("tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))
    }

    fn wrap_tensor(&self, tensor: Tensor) -> TensorRef {
        Arc::new(CandleTensorWrapper::new(tensor))
    }

    fn tensor_to_tokens(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
        common::tensor_to_tokens(tensor)
    }

    /// Get TP size from FERRUM_TP env var (0 or 1 = disabled).
    fn tp_size() -> usize {
        std::env::var("FERRUM_TP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    /// Initialize TP decode group if FERRUM_TP > 1.
    #[cfg(feature = "tensor-parallel")]
    fn ensure_tp_group(&self) -> bool {
        if self.tp_group.lock().is_some() {
            return true;
        }
        let tp = Self::tp_size();
        if tp <= 1 {
            return false;
        }

        info!("Initializing tensor parallel group: tp_size={tp}");

        let model_dir = match self.model.model_dir.as_ref() {
            Some(d) => d.clone(),
            None => {
                tracing::warn!("TP requires model_dir");
                return false;
            }
        };

        // Load sharded weights for each rank
        let loader = crate::loader::SafeTensorsLoader::new(&model_dir);
        let vb = match loader.load_varbuilder(self.model.device(), self.model.dtype()) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("TP weight load failed: {e}");
                return false;
            }
        };

        let cfg = self.model.config();
        let tp_cfg = crate::loader::tp_weight_loader::TpWeightConfig {
            num_hidden_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_attention_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            vocab_size: cfg.vocab_size,
            max_seq_len: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta as f64,
            // Auto-detect Q/K norm by probing safetensors
            has_qk_norm: crate::loader::SafeTensorsLoader::new(&model_dir)
                .load_varbuilder(self.model.device(), self.model.dtype())
                .map(|vb| {
                    vb.get(cfg.head_dim, "model.layers.0.self_attn.q_norm.weight")
                        .is_ok()
                })
                .unwrap_or(false),
            tp_size: tp,
            rank: 0,
        };
        info!(
            "TP config: has_qk_norm={}, head_dim={}, nq={}, nkv={}",
            tp_cfg.has_qk_norm, tp_cfg.head_dim, tp_cfg.num_attention_heads, tp_cfg.num_kv_heads
        );

        // Load VarBuilder ONCE on GPU 0 to guarantee replicated weights are
        // bit-identical across ranks. Candle's mmaped VarBuilder can produce
        // subtly different BF16→F16 conversions when loaded independently on
        // different GPUs. Using a single VarBuilder + cross-device to_device()
        // eliminates this. Weight sharding (narrow/cat) and GPU transfer happen
        // sequentially per rank — acceptable since this is a one-time init cost.
        let dtype = self.model.dtype();
        let device0 = match candle_core::Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Cannot create CUDA device 0: {e}");
                return false;
            }
        };
        let loader = crate::loader::SafeTensorsLoader::new(&model_dir);
        let vb = match loader.load_varbuilder(&device0, dtype) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("TP VarBuilder load failed: {e}");
                return false;
            }
        };

        let mut runners = Vec::with_capacity(tp);
        let mut nccl_streams = Vec::with_capacity(tp);

        for rank in 0..tp {
            let mut rank_cfg = tp_cfg.clone();
            rank_cfg.rank = rank;
            // Target device for this rank's weights
            let device = match candle_core::Device::new_cuda(rank) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("Cannot create CUDA device {rank}: {e}");
                    return false;
                }
            };

            let (weights, dims, stream) =
                match crate::loader::tp_weight_loader::load_sharded_weights(
                    &vb, &rank_cfg, &device,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("TP shard {rank} failed: {e}");
                        return false;
                    }
                };

            let cuda_dev = match device.as_cuda_device() {
                Ok(d) => d.clone(),
                Err(e) => {
                    tracing::warn!("TP rank {rank} not CUDA: {e}");
                    return false;
                }
            };

            let runner = match ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner::new(
                weights,
                dims,
                cuda_dev,
                stream.clone(),
            ) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("TP runner {rank} failed: {e}");
                    return false;
                }
            };

            runners.push(runner);
            nccl_streams.push(stream);
        }

        // Retain primary context for all GPUs on main thread.
        // Init threads retained their own, but those may not carry over.
        let _main_thread_devices: Vec<_> = (0..tp)
            .filter_map(|r| candle_core::Device::new_cuda(r).ok())
            .collect();

        // Init NCCL using ncclCommInitAll (single thread, no deadlock)
        let nccl_ranks = match ferrum_cuda_kernels::nccl_comm::NcclRank::init_all(nccl_streams) {
            Ok(ranks) => ranks,
            Err(e) => {
                tracing::warn!("NCCL init_all failed: {e}");
                return false;
            }
        };

        match ferrum_cuda_kernels::tp_decode::TpDecodeGroup::new(runners, nccl_ranks) {
            Ok(group) => {
                info!("Tensor parallel group initialized: {tp} GPUs");
                *self.tp_group.lock() = Some(group);
                true
            }
            Err(e) => {
                tracing::warn!("TpDecodeGroup init: {e}");
                false
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ensure_cuda_runner(&self) -> bool {
        if self.cuda_runner.lock().is_some() {
            return true;
        }
        if std::env::var("FERRUM_DISABLE_CUDA_RUNNER").map_or(false, |v| v == "1") {
            return false;
        }
        match self.model.create_decode_runner() {
            Ok(runner) => {
                info!("CUDA decode runner initialized for Llama");
                *self.cuda_runner.lock() = Some(runner);
                true
            }
            Err(e) => {
                tracing::warn!("CUDA runner init failed for Llama: {e}");
                false
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ensure_runner_kv_cache(&self, cache_id: &str, _seq_len: usize) -> Result<()> {
        use candle_core::Storage;

        let mut runner_guard = self.cuda_runner.lock();
        let runner = match runner_guard.as_mut() {
            Some(r) => r,
            None => return Ok(()),
        };
        if runner.has_kv_cache(cache_id) {
            return Ok(());
        }

        // Export from our custom model's per-request KV cache
        let kv_data = self
            .model
            .export_kv_cache(cache_id)
            .or_else(|| {
                cache_id
                    .rfind("-clone-")
                    .and_then(|pos| self.model.export_kv_cache(&cache_id[..pos]))
            })
            .ok_or_else(|| FerrumError::model(format!("No KV cache to export for: {cache_id}")))?;

        if kv_data.is_empty() {
            return Err(FerrumError::model("Empty KV cache export"));
        }
        let prefill_len = kv_data[0].2;
        let max_len = kv_data[0].3;

        let mut kv_slices = Vec::new();
        for (k_tensor, v_tensor, _len, _max) in &kv_data {
            let (k_s, _) = k_tensor.storage_and_layout();
            let (v_s, _) = v_tensor.storage_and_layout();
            let k_cuda = match &*k_s {
                Storage::Cuda(cs) => cs
                    .as_cuda_slice::<half::f16>()
                    .map_err(|e| FerrumError::model(format!("KV extract: {e}")))?
                    .clone(),
                _ => return Err(FerrumError::model("KV not on CUDA")),
            };
            let v_cuda = match &*v_s {
                Storage::Cuda(cs) => cs
                    .as_cuda_slice::<half::f16>()
                    .map_err(|e| FerrumError::model(format!("KV extract: {e}")))?
                    .clone(),
                _ => return Err(FerrumError::model("KV not on CUDA")),
            };
            drop(k_s);
            drop(v_s);
            kv_slices.push((k_cuda, v_cuda));
        }

        runner
            .init_kv_cache(cache_id, kv_slices, prefill_len, max_len)
            .map_err(|e| FerrumError::model(format!("KV init: {e}")))?;

        if !self.states.lock().contains_key(cache_id) {
            self.states.lock().insert(
                cache_id.to_string(),
                LlamaCacheState {
                    sequence_length: prefill_len,
                },
            );
        }

        debug!("Migrated Llama KV to CUDA runner: {cache_id}");
        Ok(())
    }

    pub fn release_sequence(&self, cache_id: &str) {
        self.states.lock().remove(cache_id);
        self.model.release_cache(cache_id);
        #[cfg(feature = "cuda")]
        if let Some(ref mut runner) = *self.cuda_runner.lock() {
            runner.release_kv_cache(cache_id);
        }
    }
}

#[async_trait]
impl ModelExecutor for CandleModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        debug!("Llama Prefill: seq_len={}", input.sequence_length());

        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Empty input"));
        }

        let cache_id = format!(
            "llama-cache-{}",
            self.next_cache_id.fetch_add(1, Ordering::Relaxed)
        );

        let input_tensor = self.tokens_to_tensor(&tokens)?;
        let logits = self.model.forward_prefill(&input_tensor, &cache_id)?;

        let logits = match logits.dims().len() {
            2 => logits
                .unsqueeze(1)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?,
            3 => logits,
            d => return Err(FerrumError::model(format!("Unexpected logits rank: {d}"))),
        };

        let logits_ref = self.wrap_tensor(logits);
        let cfg = self.model.config();
        let handle = Arc::new(GenericKvCacheHandle::new(
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            self.model.device().clone(),
            tokens.len(),
            cache_id.clone(),
        ));

        self.states.lock().insert(
            cache_id,
            LlamaCacheState {
                sequence_length: tokens.len(),
            },
        );

        Ok(PrefillOutput::new(logits_ref, handle))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let handle = input
            .kv_cache
            .as_any()
            .downcast_ref::<GenericKvCacheHandle>()
            .ok_or_else(|| FerrumError::model("Invalid KV handle for Llama"))?;
        let cache_id = handle.request_cache_id().to_string();

        let seq_len = {
            let mut states = self.states.lock();
            if let Some(s) = states.get(&cache_id) {
                s.sequence_length
            } else {
                let len = handle.block_table().sequence_length;
                states.insert(
                    cache_id.clone(),
                    LlamaCacheState {
                        sequence_length: len,
                    },
                );
                len
            }
        };

        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Empty decode input"));
        }

        // Try TP path first (FERRUM_TP > 1)
        #[cfg(feature = "tensor-parallel")]
        {
            if Self::tp_size() > 1 && self.ensure_tp_group() {
                // Ensure KV cache exists on all TP ranks.
                // Export candle KV and shard by heads across ranks.
                {
                    let mut group = self.tp_group.lock();
                    if let Some(ref mut g) = *group {
                        if !g.has_kv_cache(&cache_id) {
                            let tp = g.world_size();
                            let kv_data = self
                                .model
                                .export_kv_cache(&cache_id)
                                .or_else(|| {
                                    cache_id.rfind("-clone-").and_then(|pos| {
                                        self.model.export_kv_cache(&cache_id[..pos])
                                    })
                                })
                                .ok_or_else(|| {
                                    FerrumError::model(format!("No KV for TP: {cache_id}"))
                                })?;

                            if !kv_data.is_empty() {
                                let prefill_len = kv_data[0].2;
                                let max_len = kv_data[0].3;
                                let num_kv_heads = self.model.config().num_key_value_heads;
                                let heads_per_rank = num_kv_heads / tp;

                                let mut per_rank_kv: Vec<
                                    Vec<(
                                        candle_core::cuda_backend::cudarc::driver::CudaSlice<
                                            half::f16,
                                        >,
                                        candle_core::cuda_backend::cudarc::driver::CudaSlice<
                                            half::f16,
                                        >,
                                    )>,
                                > = (0..tp).map(|_| Vec::new()).collect();

                                // Shard on GPU 0, D2H via candle, H2D via runner
                                for (k_tensor, v_tensor, _len, _max) in &kv_data {
                                    for rank in 0..tp {
                                        let start = rank * heads_per_rank;
                                        let k_shard = k_tensor
                                            .narrow(1, start, heads_per_rank)
                                            .and_then(|t| t.contiguous())
                                            .map_err(|e| {
                                                FerrumError::model(format!("KV shard: {e}"))
                                            })?;
                                        let v_shard = v_tensor
                                            .narrow(1, start, heads_per_rank)
                                            .and_then(|t| t.contiguous())
                                            .map_err(|e| {
                                                FerrumError::model(format!("KV shard: {e}"))
                                            })?;

                                        if rank == 0 {
                                            // Same GPU: extract CudaSlice directly
                                            use candle_core::Storage;
                                            let (ks, _) = k_shard.storage_and_layout();
                                            let (vs, _) = v_shard.storage_and_layout();
                                            let kc = match &*ks {
                                                Storage::Cuda(cs) => cs
                                                    .as_cuda_slice::<half::f16>()
                                                    .map_err(|e| {
                                                        FerrumError::model(format!("KV: {e}"))
                                                    })?
                                                    .clone(),
                                                _ => return Err(FerrumError::model("KV not CUDA")),
                                            };
                                            let vc = match &*vs {
                                                Storage::Cuda(cs) => cs
                                                    .as_cuda_slice::<half::f16>()
                                                    .map_err(|e| {
                                                        FerrumError::model(format!("KV: {e}"))
                                                    })?
                                                    .clone(),
                                                _ => return Err(FerrumError::model("KV not CUDA")),
                                            };
                                            drop(ks);
                                            drop(vs);
                                            per_rank_kv[rank].push((kc, vc));
                                        } else {
                                            // Cross-GPU: D2H via candle → H2D via runner
                                            let k_host = k_shard
                                                .flatten_all()
                                                .and_then(|t| t.to_vec1::<half::f16>())
                                                .map_err(|e| {
                                                    FerrumError::model(format!("KV d2h: {e}"))
                                                })?;
                                            let v_host = v_shard
                                                .flatten_all()
                                                .and_then(|t| t.to_vec1::<half::f16>())
                                                .map_err(|e| {
                                                    FerrumError::model(format!("KV d2h: {e}"))
                                                })?;
                                            let kc = g
                                                .runner_mut(rank)
                                                .upload_to_self(&k_host)
                                                .map_err(|e| {
                                                    FerrumError::model(format!(
                                                        "KV h2d r{rank}: {e}"
                                                    ))
                                                })?;
                                            let vc = g
                                                .runner_mut(rank)
                                                .upload_to_self(&v_host)
                                                .map_err(|e| {
                                                    FerrumError::model(format!(
                                                        "KV h2d r{rank}: {e}"
                                                    ))
                                                })?;
                                            per_rank_kv[rank].push((kc, vc));
                                        }
                                    }
                                }
                                g.init_kv_cache(&cache_id, per_rank_kv, prefill_len, max_len)
                                    .map_err(|e| FerrumError::model(format!("TP KV init: {e}")))?;
                            }
                        }
                    }
                }

                let logits = {
                    let mut group = self.tp_group.lock();
                    let group = group
                        .as_mut()
                        .ok_or_else(|| FerrumError::model("TP group gone"))?;
                    group
                        .decode_step(tokens[0], seq_len, &cache_id)
                        .map_err(|e| FerrumError::model(format!("tp_decode: {e}")))?
                };

                let cuda_dev = self
                    .model
                    .candle_device()
                    .as_cuda_device()
                    .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;
                let vocab = self.info.vocab_size;
                let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                    logits,
                    cuda_dev.clone(),
                );
                let logits_tensor = candle_core::Tensor::from_storage(
                    candle_core::Storage::Cuda(storage),
                    (1, 1, vocab),
                    candle_core::op::BackpropOp::none(),
                    false,
                );
                // Diagnostic: print TP logits top-5
                if let Ok(vals) = logits_tensor
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<half::f16>())
                {
                    let has_nan = vals.iter().any(|v| v.is_nan());
                    let has_inf = vals.iter().any(|v| v.is_infinite());
                    let max_val = vals
                        .iter()
                        .map(|v| v.to_f32())
                        .fold(f32::NEG_INFINITY, f32::max);
                    let min_val = vals
                        .iter()
                        .map(|v| v.to_f32())
                        .fold(f32::INFINITY, f32::min);
                    let mut indexed: Vec<(usize, f32)> = vals
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i, v.to_f32()))
                        .collect();
                    indexed
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let top5: Vec<String> = indexed[..5.min(indexed.len())]
                        .iter()
                        .map(|(i, v)| format!("{i}:{v:.2}"))
                        .collect();
                    eprintln!("[TP DIAG] logits: len={} nan={has_nan} inf={has_inf} min={min_val:.2} max={max_val:.2} top5=[{}]", vals.len(), top5.join(", "));
                }

                let logits_ref = self.wrap_tensor(logits_tensor);
                let new_seq_len = seq_len + 1;
                {
                    let mut states = self.states.lock();
                    if let Some(s) = states.get_mut(&cache_id) {
                        s.sequence_length = new_seq_len;
                    }
                }
                let new_handle = Arc::new(handle.with_sequence_length(new_seq_len));
                return Ok(DecodeOutput::new(logits_ref, new_handle));
            }
        }

        // Try single-GPU CUDA runner path
        #[cfg(feature = "cuda")]
        {
            if std::env::var("FERRUM_DISABLE_CUDA_RUNNER").map_or(true, |v| v != "1") {
                if self.ensure_cuda_runner() {
                    self.ensure_runner_kv_cache(&cache_id, seq_len)?;

                    let logits = {
                        let mut runner = self.cuda_runner.lock();
                        let runner = runner
                            .as_mut()
                            .ok_or_else(|| FerrumError::model("CUDA runner gone"))?;
                        runner
                            .decode_step(tokens[0], seq_len, &cache_id)
                            .map_err(|e| FerrumError::model(format!("decode: {e}")))?
                    };

                    let cuda_dev = self
                        .model
                        .candle_device()
                        .as_cuda_device()
                        .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;
                    let vocab = self.info.vocab_size;
                    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                        logits,
                        cuda_dev.clone(),
                    );
                    let logits_tensor = candle_core::Tensor::from_storage(
                        candle_core::Storage::Cuda(storage),
                        (1, 1, vocab),
                        candle_core::op::BackpropOp::none(),
                        false,
                    );

                    let logits_ref = self.wrap_tensor(logits_tensor);

                    let new_seq_len = seq_len + 1;
                    {
                        let mut states = self.states.lock();
                        if let Some(s) = states.get_mut(&cache_id) {
                            s.sequence_length = new_seq_len;
                        }
                    }
                    let new_handle = Arc::new(handle.with_sequence_length(new_seq_len));
                    return Ok(DecodeOutput::new(logits_ref, new_handle));
                }
            }
        }

        // Candle fallback
        let input_tensor = self.tokens_to_tensor(&tokens)?;
        let logits = self
            .model
            .forward_decode(&input_tensor, seq_len, &cache_id)?;
        let logits = match logits.dims().len() {
            2 => logits
                .unsqueeze(1)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?,
            3 => logits,
            _ => logits,
        };
        let logits_ref = self.wrap_tensor(logits);

        let new_seq_len = seq_len + tokens.len();
        {
            let mut states = self.states.lock();
            if let Some(s) = states.get_mut(&cache_id) {
                s.sequence_length = new_seq_len;
            }
        }
        let new_handle = Arc::new(handle.with_sequence_length(new_seq_len));
        Ok(DecodeOutput::new(logits_ref, new_handle))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::MultiHead, AttentionType::GroupedQuery],
            supports_dynamic_batching: false,
            supports_continuous_batching: true,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP16, DataType::FP32],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: (self.info.num_parameters * 2) as u64,
                activation_memory_per_token: 4 * self.info.hidden_size,
                kv_cache_memory_per_token: 2 * self.info.num_layers * self.info.hidden_size,
                overhead_memory: 1024 * 1024 * 1024,
            },
        }
    }

    fn release_cache(&self, cache_id: &str) {
        self.release_sequence(cache_id);
    }

    fn status(&self) -> ExecutorStatus {
        common::default_executor_status()
    }
}
