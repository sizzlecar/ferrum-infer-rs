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
    #[cfg(feature = "cuda")]
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
            #[cfg(feature = "cuda")]
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

    /// Get TP size: FERRUM_TP env overrides, otherwise auto-detect GPU count.
    /// FERRUM_TP=0 or FERRUM_TP=1 explicitly disables TP.
    #[cfg(feature = "cuda")]
    fn tp_size() -> usize {
        if let Ok(v) = std::env::var("FERRUM_TP") {
            if let Ok(n) = v.parse::<usize>() {
                return n;
            }
        }
        // Auto-detect: use all available GPUs
        candle_core::cuda_backend::cudarc::driver::CudaContext::device_count()
            .map(|n| n as usize)
            .unwrap_or(1)
    }

    #[cfg(not(feature = "cuda"))]
    fn tp_size() -> usize {
        0
    }

    /// Initialize TP decode group if FERRUM_TP > 1.
    #[cfg(feature = "cuda")]
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

        // Each rank loads weights in its own thread (correct CUDA context).
        // After loading, we sync replicated weights from rank 0 to other ranks
        // to fix candle VarBuilder producing different BF16→F16 conversions
        // when loaded independently on different GPUs.
        type RankResult = candle_core::Result<(
            ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner,
            std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
        )>;

        let mut handles: Vec<std::thread::JoinHandle<RankResult>> = Vec::with_capacity(tp);
        let dtype = self.model.dtype();
        for rank in 0..tp {
            let mut rank_cfg = tp_cfg.clone();
            rank_cfg.rank = rank;
            let model_dir = model_dir.clone();

            handles.push(std::thread::spawn(move || {
                let device = candle_core::Device::new_cuda(rank)?;
                let loader = crate::loader::SafeTensorsLoader::new(&model_dir);
                let vb = loader
                    .load_varbuilder(&device, dtype)
                    .map_err(|e| candle_core::Error::Msg(format!("VB rank {rank}: {e}")))?;

                let (weights, dims, stream) =
                    crate::loader::tp_weight_loader::load_sharded_weights(&vb, &rank_cfg, &device)
                        .map_err(|e| candle_core::Error::Msg(format!("shard {rank}: {e}")))?;

                let cuda_dev = device.as_cuda_device()?.clone();

                let runner = ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner::new(
                    weights,
                    dims,
                    cuda_dev,
                    stream.clone(),
                )?;
                Ok((runner, stream))
            }));
        }

        let mut runners = Vec::with_capacity(tp);
        let mut nccl_streams = Vec::with_capacity(tp);
        for (rank, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok((runner, stream))) => {
                    runners.push(runner);
                    nccl_streams.push(stream);
                }
                Ok(Err(e)) => {
                    tracing::warn!("TP rank {rank} failed: {e}");
                    return false;
                }
                Err(_) => {
                    tracing::warn!("TP rank {rank} panicked");
                    return false;
                }
            }
        }

        // Sync replicated weights: copy from rank 0 to all other ranks.
        // Fixes candle VarBuilder BF16→F16 divergence across GPUs.
        for rank in 1..tp {
            let (src, dst) = if rank == 1 {
                let (first, rest) = runners.split_at_mut(1);
                (&first[0], &mut rest[0])
            } else {
                let (first, rest) = runners.split_at_mut(1);
                (&first[0], &mut rest[rank - 1])
            };
            if let Err(e) = dst.sync_replicated_weights_from(src) {
                tracing::warn!("TP weight sync rank 0→{rank} failed: {e}");
                return false;
            }
            info!("Synced replicated weights: rank 0 → rank {rank}");
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
        #[cfg(feature = "cuda")]
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

                                use ferrum_cuda_kernels::tp_decode::KvSource;

                                let mut per_rank_kv: Vec<Vec<(KvSource, KvSource)>> =
                                    (0..tp).map(|_| Vec::new()).collect();

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
                                            per_rank_kv[rank]
                                                .push((KvSource::Gpu(kc), KvSource::Gpu(vc)));
                                        } else {
                                            // Cross-GPU: D2H here, worker does H2D
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
                                            per_rank_kv[rank].push((
                                                KvSource::Host(k_host),
                                                KvSource::Host(v_host),
                                            ));
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
