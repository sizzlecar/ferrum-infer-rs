//! Direct token execution for model-owned KV implementations.
//!
//! The model owns numerical execution and physical resources. This executor
//! retains exact token incarnations and uses an independently owned worker
//! when the implementation requires device submission.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};

use ferrum_execution::{
    model_executor::{
        AttentionType, ExecutorCapabilities, ExecutorStatus, KvSlotCapacitySnapshot, KvSlotRequest,
        KvSlotReservation, MemoryRequirements, PagedAdmissionCapacitySnapshot,
    },
    ModelExecutor,
};
use ferrum_interfaces::adapter::AdapterExecutionSupport;
#[cfg(test)]
use ferrum_types::TokenId;
use ferrum_types::{DataType, FerrumError, ModelInfo, RequestId, Result};

use crate::common::{DecoderOnlyLLM, LlmRuntimeConfig, ModelTokenExecutionMode};
use crate::lora::ActiveLoraAdapter;
use crate::vnext::ProductionModelSourceBundle;

use super::common;

mod device_worker;
mod token_execution;

type ModelSlot = Arc<Mutex<Option<Box<dyn DecoderOnlyLLM>>>>;

fn lock_live_model(
    slot: &Mutex<Option<Box<dyn DecoderOnlyLLM>>>,
) -> Result<MappedMutexGuard<'_, Box<dyn DecoderOnlyLLM>>> {
    MutexGuard::try_map(slot.lock(), Option::as_mut)
        .map_err(|_| FerrumError::request_validation("model executor has retired"))
}

pub struct LlmExecutor {
    // A submitted device task must be able to retain the actual model-owned
    // weights, KV and scratch independently of the awaiting caller/executor.
    // This changes ownership, not the single exclusive model lock per batch.
    model: ModelSlot,
    model_config: LlmRuntimeConfig,
    model_capacity: usize,
    adapter_support: AdapterExecutionSupport,
    info: ModelInfo,
    bound_sources: Option<Arc<ProductionModelSourceBundle>>,
    next_cache_id: AtomicU64,
    total_model_lock_wait_us: AtomicU64,
    model_lock_wait_samples: AtomicU64,
    token_issuer: ferrum_execution::token_execution::ExecutorSequenceIssuer,
    token_sequences: Arc<Mutex<token_execution::TokenSequences>>,
    token_execution_mode: ModelTokenExecutionMode,
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
        let token_execution_mode = model.token_execution_mode();
        let model_config = model.config().clone();
        let model_capacity = model.kv_capacity();
        Self {
            model: Arc::new(Mutex::new(Some(model))),
            model_config,
            model_capacity,
            adapter_support,
            info,
            bound_sources,
            next_cache_id: AtomicU64::new(0),
            total_model_lock_wait_us: AtomicU64::new(0),
            model_lock_wait_samples: AtomicU64::new(0),
            token_issuer: ferrum_execution::token_execution::ExecutorSequenceIssuer::new(),
            token_sequences: Arc::new(Mutex::new(token_execution::TokenSequences::default())),
            token_execution_mode,
            token_worker: OnceLock::new(),
        }
    }

    fn lock_model(&self) -> Result<MappedMutexGuard<'_, Box<dyn DecoderOnlyLLM>>> {
        let start = std::time::Instant::now();
        let guard = lock_live_model(&self.model);
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
    fn prefill_shape_policy(&self) -> ferrum_execution::model_executor::PrefillShapePolicy {
        // This executor submits the validated token range without a separate
        // checkpoint-tail rewrite. Actual partial completion is reported by
        // the execution result, independently of this planning rule.
        ferrum_execution::model_executor::PrefillShapePolicy::Identity
    }

    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
    ) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ferrum_execution::executor_lifecycle::ExecutorShutdownObserver::retained(
                Arc::new(self),
                error,
            );
        }
        self.token_sequences.lock().seal();
        match self
            .token_device_worker()
            .and_then(device_worker::LlmDeviceWorker::begin_shutdown)
        {
            Ok(observer) => observer,
            Err(error) => ferrum_execution::executor_lifecycle::ExecutorShutdownObserver::retained(
                Arc::new(self),
                error,
            ),
        }
    }
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
        match self.token_execution_mode {
            ModelTokenExecutionMode::ManagedWorker => {
                self.token_device_worker()?.execute(work).await
            }
            ModelTokenExecutionMode::InlineSynchronous => self.execute_token_batch(work),
            ModelTokenExecutionMode::Unsupported => Err(FerrumError::unsupported(
                "model/backend has not qualified token execution",
            )),
        }
    }

    async fn finish_tokens(
        &self,
        sequence: &ferrum_execution::token_execution::ExecutorSequence,
        reason: ferrum_execution::token_execution::ExecutionEndReason,
    ) -> Result<ferrum_execution::token_execution::ExecutionCloseOutcome> {
        match self.token_execution_mode {
            ModelTokenExecutionMode::ManagedWorker => {
                self.token_device_worker()?.close(sequence, reason)
            }
            ModelTokenExecutionMode::InlineSynchronous => {
                self.close_token_incarnation(sequence, reason)
            }
            ModelTokenExecutionMode::Unsupported => Err(FerrumError::unsupported(
                "model/backend has not qualified token execution",
            )),
        }
    }

    fn cancel_tokens(
        &self,
        sequence: &ferrum_execution::token_execution::ExecutorSequence,
    ) -> Result<ferrum_execution::token_execution::ExecutionCloseOutcome> {
        let reason = ferrum_execution::token_execution::ExecutionEndReason::Cancelled;
        match self.token_execution_mode {
            ModelTokenExecutionMode::ManagedWorker => {
                self.token_device_worker()?.close(sequence, reason)
            }
            ModelTokenExecutionMode::InlineSynchronous => {
                self.close_token_incarnation(sequence, reason)
            }
            ModelTokenExecutionMode::Unsupported => Err(FerrumError::unsupported(
                "model/backend has not qualified token execution",
            )),
        }
    }

    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn implementation_identity(&self) -> Option<&'static str> {
        Some(std::any::type_name::<Self>())
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
        Some(self.model_capacity)
    }

    fn reserve_kv_slots(&self, requests: &[KvSlotRequest]) -> Result<Option<KvSlotReservation>> {
        if self.token_sequences.lock().is_closing() {
            return Err(FerrumError::request_validation("model executor is closing"));
        }
        self.lock_model()?.reserve_kv_slots(requests)
    }

    fn kv_slot_capacity_snapshot(&self) -> Option<KvSlotCapacitySnapshot> {
        self.lock_model().ok()?.kv_slot_capacity_snapshot()
    }

    fn paged_admission_capacity_snapshot(&self) -> Option<PagedAdmissionCapacitySnapshot> {
        self.lock_model().ok()?.paged_admission_capacity_snapshot()
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        let cfg = &self.model_config;
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
        let snapshot = self.lock_model().ok()?.cache_metrics_snapshot()?;
        Some(self.attach_model_lock_metrics(snapshot))
    }

    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        self.lock_model().ok()?.lora_metrics_snapshot()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod shutdown_tests;
