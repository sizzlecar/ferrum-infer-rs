//! Direct token execution for model-owned KV implementations.
//!
//! The model owns numerical execution and physical resources. This executor
//! retains exact token incarnations and uses an independently owned worker
//! when the implementation requires device submission.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::{Mutex, MutexGuard};

use ferrum_execution::{
    model_executor::{
        AttentionType, ExecutorCapabilities, ExecutorStatus, KvSlotCapacitySnapshot, KvSlotRequest,
        KvSlotReservation, MemoryRequirements,
    },
    ModelExecutor,
};
use ferrum_interfaces::{adapter::AdapterExecutionSupport, RecurrentStateSpec};
use ferrum_types::{DataType, FerrumError, ModelInfo, RequestId, Result, TokenId};

use crate::common::DecoderOnlyLLM;
use crate::lora::ActiveLoraAdapter;
use crate::vnext::ProductionModelSourceBundle;

use super::common;

mod device_worker;
mod token_execution;

pub struct LlmExecutor {
    // A submitted device task must be able to retain the actual model-owned
    // weights, KV and scratch independently of the awaiting caller/executor.
    // This changes ownership, not the single exclusive model lock per batch.
    model: Arc<Mutex<Box<dyn DecoderOnlyLLM>>>,
    adapter_support: AdapterExecutionSupport,
    info: ModelInfo,
    bound_sources: Option<Arc<ProductionModelSourceBundle>>,
    next_cache_id: AtomicU64,
    total_model_lock_wait_us: AtomicU64,
    model_lock_wait_samples: AtomicU64,
    token_issuer: ferrum_execution::token_execution::ExecutorSequenceIssuer,
    token_sequences: Arc<Mutex<token_execution::TokenSequences>>,
    token_worker_required: bool,
    token_worker: OnceLock<Result<device_worker::LlmDeviceWorker>>,
}

impl LlmExecutor {
    pub fn new(model: Box<dyn DecoderOnlyLLM>, info: ModelInfo) -> Self {
        Self::new_with_optional_model_sources(model, info, None)
    }

    /// Retains the exact product sources used to construct the model.
    pub fn new_with_model_sources(
        model: Box<dyn DecoderOnlyLLM>,
        info: ModelInfo,
        sources: Arc<ProductionModelSourceBundle>,
    ) -> Self {
        Self::new_with_optional_model_sources(model, info, Some(sources))
    }

    fn new_with_optional_model_sources(
        model: Box<dyn DecoderOnlyLLM>,
        info: ModelInfo,
        bound_sources: Option<Arc<ProductionModelSourceBundle>>,
    ) -> Self {
        let adapter_support = model.adapter_execution_support();
        let token_worker_required = model.token_execution_requires_worker();
        Self {
            model: Arc::new(Mutex::new(model)),
            adapter_support,
            info,
            bound_sources,
            next_cache_id: AtomicU64::new(0),
            total_model_lock_wait_us: AtomicU64::new(0),
            model_lock_wait_samples: AtomicU64::new(0),
            token_issuer: ferrum_execution::token_execution::ExecutorSequenceIssuer::new(),
            token_sequences: Arc::new(Mutex::new(token_execution::TokenSequences::default())),
            token_worker_required,
            token_worker: OnceLock::new(),
        }
    }

    fn lock_model(&self) -> MutexGuard<'_, Box<dyn DecoderOnlyLLM>> {
        let start = std::time::Instant::now();
        let guard = self.model.lock();
        self.record_model_lock_wait(start.elapsed());
        guard
    }

    fn record_model_lock_wait(&self, duration: std::time::Duration) {
        self.total_model_lock_wait_us.fetch_add(
            duration.as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.model_lock_wait_samples.fetch_add(1, Ordering::Relaxed);
    }

    fn model_lock_metrics_json(&self) -> serde_json::Value {
        let samples = self.model_lock_wait_samples.load(Ordering::Relaxed);
        let total_us = self.total_model_lock_wait_us.load(Ordering::Relaxed);
        serde_json::json!({
            "schema_version": 1,
            "samples": samples,
            "total_wait_time_us": total_us,
            "avg_wait_time_ms": if samples == 0 {
                0.0
            } else {
                total_us as f64 / samples as f64 / 1000.0
            },
        })
    }

    fn attach_model_lock_metrics(&self, mut snapshot: serde_json::Value) -> serde_json::Value {
        let lock_metrics = self.model_lock_metrics_json();
        if let Some(obj) = snapshot.as_object_mut() {
            obj.insert("executor_model_lock".to_string(), lock_metrics);
            snapshot
        } else {
            serde_json::json!({
                "cache_metrics": snapshot,
                "executor_model_lock": lock_metrics,
            })
        }
    }

    fn gen_cache_id(&self) -> String {
        format!(
            "llm-cache-{}",
            self.next_cache_id.fetch_add(1, Ordering::Relaxed)
        )
    }
}

#[async_trait::async_trait]
impl ModelExecutor for LlmExecutor {
    fn observe_token_allocator_reclaim(
        &self,
        blocked: &ferrum_execution::token_execution::ExecutionFrontier,
        domain: &ferrum_types::AllocatorCapacityDomain,
        candidates: &[ferrum_execution::token_execution::ExecutorSequence],
    ) -> Result<Option<ferrum_execution::token_execution::TokenAllocatorReclaim>> {
        self.observe_allocator_reclaim(blocked, domain, candidates)
    }

    async fn admit_tokens(
        &self,
        input: ferrum_execution::token_execution::ExecutionAdmission<'_>,
    ) -> Result<ferrum_execution::token_execution::TokenAdmissionOutcome> {
        self.admit_token_sequence_async(input).await
    }

    async fn execute_tokens(
        &self,
        work: &[ferrum_execution::token_execution::ExecutionWork<'_>],
    ) -> Result<ferrum_execution::token_execution::ExecutionBatchOutcome> {
        if self.token_worker_required {
            self.token_device_worker()?.execute(work).await
        } else {
            self.execute_token_batch(work)
        }
    }

    async fn finish_tokens(
        &self,
        sequence: &ferrum_execution::token_execution::ExecutorSequence,
        reason: ferrum_execution::token_execution::ExecutionEndReason,
    ) -> Result<ferrum_execution::token_execution::ExecutionCloseOutcome> {
        if self.token_worker_required {
            self.token_device_worker()?.close(sequence, reason)
        } else {
            self.close_token_incarnation(sequence, reason)
        }
    }

    fn cancel_tokens(
        &self,
        sequence: &ferrum_execution::token_execution::ExecutorSequence,
    ) -> Result<ferrum_execution::token_execution::ExecutionCloseOutcome> {
        let reason = ferrum_execution::token_execution::ExecutionEndReason::Cancelled;
        if self.token_worker_required {
            self.token_device_worker()?.close(sequence, reason)
        } else {
            self.close_token_incarnation(sequence, reason)
        }
    }

    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn bound_resolved_sources(&self) -> Option<&ferrum_runtime::ResolvedModelSources> {
        self.bound_sources
            .as_deref()
            .map(ProductionModelSourceBundle::resolved_sources)
    }

    fn adapter_execution_support(&self) -> AdapterExecutionSupport {
        self.adapter_support
    }

    fn kv_capacity(&self) -> Option<usize> {
        Some(self.lock_model().kv_capacity())
    }

    fn reserve_kv_slots(&self, requests: &[KvSlotRequest]) -> Result<Option<KvSlotReservation>> {
        self.lock_model().reserve_kv_slots(requests)
    }

    fn kv_slot_capacity_snapshot(&self) -> Option<KvSlotCapacitySnapshot> {
        self.lock_model().kv_slot_capacity_snapshot()
    }

    fn recurrent_state_spec(
        &self,
        request_id: &RequestId,
        input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        let mut spec = self
            .lock_model()
            .recurrent_state_spec(request_id, input_tokens)?;
        if let Some(spec) = spec.as_mut() {
            spec.device = self.info.device.clone();
        }
        Ok(spec)
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        let cfg = self.lock_model().config().clone();
        ExecutorCapabilities {
            max_batch_size: 256,
            max_sequence_length: cfg.max_seq_len,
            attention_mechanisms: vec![AttentionType::GroupedQuery],
            supports_dynamic_batching: true,
            supports_continuous_batching: true,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                typed_sequence_state: None,
                parameter_memory: (self.info.num_parameters * 4) as u64,
                activation_memory_per_token: cfg.hidden_size * 4,
                kv_cache_memory_per_token: cfg.hidden_size * 2,
                overhead_memory: 256 * 1024 * 1024,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        common::default_executor_status()
    }

    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        let snapshot = self.lock_model().cache_metrics_snapshot()?;
        Some(self.attach_model_lock_metrics(snapshot))
    }

    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        self.lock_model().lora_metrics_snapshot()
    }
}

#[cfg(test)]
mod tests;
