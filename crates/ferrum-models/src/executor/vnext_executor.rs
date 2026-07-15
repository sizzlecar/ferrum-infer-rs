//! Shared production executor for immutable vNext model programs.
//!
//! Model families provide semantics and weights; backend compositions provide
//! typed operation providers and a device runtime. This executor owns the
//! remaining product lifecycle: compile once, initialize once, dynamically
//! admit exact live work, submit immutable-plan waves, and retire resources
//! only after a terminal fence.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use ferrum_interfaces::kv_cache::{BlockTable, CacheHandleStats};
use ferrum_interfaces::model_executor::{
    AttentionType, DecodeInput, DecodeOutput, ExecutionResourceOwnership, ExecutorAdmissionEpochs,
    ExecutorCapabilities, ExecutorMemoryUsage, ExecutorPrefillAdmission,
    ExecutorPrefillAdmissionDecision, ExecutorPrefillAdmissionReceipt,
    ExecutorPrefillMaintenanceDeferral, ExecutorPrefillMaintenanceOutcome, ExecutorState,
    ExecutorStatus, MemoryRequirements, PrefillInput, PrefillOutput,
};
use ferrum_interfaces::vnext::*;
use ferrum_interfaces::{KvCacheHandle, ModelExecutor, TensorRef};
use ferrum_types::{
    Device, EngineConfig, FerrumError, ModelInfo, RequestId, Result, SchedulingPolicy,
};
use parking_lot::{Mutex, RwLock};
use tokio::sync::Mutex as AsyncMutex;

use crate::vnext::PreparedProductionModel;

use super::common;

const POLICY_ID: &str = "policy.ferrum.product.vnext.default";
const POLICY_VERSION: ContractVersion = ContractVersion::new(1, 0);
const DEFAULT_STATIC_STAGING_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_STATIC_COMMANDS_PER_BATCH: usize = 64;
const DEFAULT_CANCELLATION_CHECK_INTERVAL_STEPS: u32 = 1;
const MAX_DEFINITELY_NOT_SUBMITTED_RETRIES: u32 = 1;
const MAX_BACKING_MAINTENANCE_ATTEMPTS: u32 = 2;
const MAX_EXTENSION_RECHECKS: u32 = 2;

type VNextDriver<R> = RuntimeResourceDriver<R>;

/// Typed product policy resolved before plan compilation. None of these
/// values are inferred from a model name, GPU name, or hidden environment
/// combination.
#[derive(Debug, Clone)]
pub struct VNextExecutorConfig {
    pub maximum_model_tokens: usize,
    pub static_initialization: StaticInitializationPolicy,
    pub runtime_policy: ResolvedRuntimePolicy,
}

impl VNextExecutorConfig {
    pub fn from_engine_config<R: DeviceRuntime>(
        engine: &EngineConfig,
        info: &ModelInfo,
        runtime: &R,
    ) -> Result<Self> {
        let descriptor = runtime.descriptor();
        descriptor
            .validate()
            .map_err(|error| FerrumError::config(format!("invalid vNext runtime: {error}")))?;

        let maximum_model_tokens = engine
            .runtime
            .max_model_len
            .unwrap_or(info.max_sequence_length)
            .min(info.max_sequence_length);
        if maximum_model_tokens == 0 {
            return Err(FerrumError::config(
                "vNext maximum model length must be greater than zero",
            ));
        }

        let memory_budget = engine
            .memory
            .resolve_capacity_budget(descriptor.total_memory_bytes)
            .map_err(FerrumError::config)?;

        let maximum_active_sequences = u32::try_from(engine.scheduler.max_running_requests)
            .map_err(|_| {
                FerrumError::config("scheduler.max_running_requests exceeds the vNext limit")
            })?;
        let maximum_queue_depth =
            u32::try_from(engine.scheduler.max_waiting_requests).map_err(|_| {
                FerrumError::config("scheduler.max_waiting_requests exceeds the vNext limit")
            })?;
        let maximum_scheduled_tokens = u64::try_from(engine.batching.max_num_batched_tokens)
            .map_err(|_| {
                FerrumError::config("batching.max_num_batched_tokens exceeds the vNext limit")
            })?;
        let scheduling = match engine.scheduler.policy {
            SchedulingPolicy::Priority | SchedulingPolicy::FairShare => {
                SchedulingDiscipline::Priority
            }
            SchedulingPolicy::FCFS
            | SchedulingPolicy::SJF
            | SchedulingPolicy::RoundRobin
            | SchedulingPolicy::ContinuousBatch => SchedulingDiscipline::FirstReady,
        };
        let dynamic_storage_profile_order = descriptor
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect::<Vec<_>>();

        let runtime_policy = ResolvedRuntimePolicy::new(
            POLICY_ID,
            POLICY_VERSION,
            scheduling,
            RuntimeMemoryPolicy {
                capacity_bytes: memory_budget.capacity_bytes,
                reserve_bytes: memory_budget.reserve_bytes,
                maximum_active_sequences,
                dynamic_storage_profile_order,
            },
            AdmissionPolicy {
                maximum_queue_depth,
                maximum_scheduled_tokens,
                allow_defer: true,
                cancellation_check_interval_steps: DEFAULT_CANCELLATION_CHECK_INTERVAL_STEPS,
            },
        )
        .map_err(|error| FerrumError::config(format!("invalid vNext policy: {error}")))?;
        let static_initialization = StaticInitializationPolicy::new(
            DEFAULT_STATIC_STAGING_BYTES,
            DEFAULT_STATIC_COMMANDS_PER_BATCH,
        )
        .map_err(|error| FerrumError::config(error.to_string()))?;

        Ok(Self {
            maximum_model_tokens,
            static_initialization,
            runtime_policy,
        })
    }
}

#[derive(Debug, Clone)]
struct VNextIoBinding {
    input_node_id: NodeId,
    input_ordinal: u32,
    output_node_id: NodeId,
    output_resource_id: ResourceId,
    output_offset_bytes: u64,
    output_layout: HostTransferLayout,
    output_element_type: ElementType,
    output_elements: usize,
}

#[derive(Default)]
struct VNextExecutorMetrics {
    prefill_operations: AtomicU64,
    decode_operations: AtomicU64,
    submitted_waves: AtomicU64,
    completed_waves: AtomicU64,
    failed_waves: AtomicU64,
    definitely_not_submitted_retries: AtomicU64,
    request_deferrals: AtomicU64,
    sequence_deferrals: AtomicU64,
    extension_deferrals: AtomicU64,
    step_deferrals: AtomicU64,
    wave_deferrals: AtomicU64,
    backing_deferrals: AtomicU64,
    uploaded_bytes: AtomicU64,
    readback_bytes: AtomicU64,
    total_prefill_us: AtomicU64,
    total_decode_us: AtomicU64,
    last_failure: Mutex<Option<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeFailureDisposition {
    PreserveForCapacityRetry,
    AbortSequence,
}

impl DecodeFailureDisposition {
    fn from_error(error: &FerrumError) -> Self {
        if matches!(error, FerrumError::ResourceExhausted { .. }) {
            Self::PreserveForCapacityRetry
        } else {
            Self::AbortSequence
        }
    }
}

impl VNextExecutorMetrics {
    fn record_failure(&self, message: impl Into<String>) {
        self.failed_waves.fetch_add(1, Ordering::Relaxed);
        *self.last_failure.lock() = Some(message.into());
    }

    fn average_ms(total_us: u64, operations: u64) -> f64 {
        if operations == 0 {
            0.0
        } else {
            total_us as f64 / operations as f64 / 1000.0
        }
    }
}

fn reported_allocated_bytes(budget_claimed_bytes: Option<u64>, static_bytes: u64) -> u64 {
    budget_claimed_bytes.unwrap_or(static_bytes)
}

enum JournaledSubmission {
    Captured {
        receipt: SubmittedOperationReceipt,
        selected: Vec<usize>,
    },
    Suppressed {
        slot_id: CompletionSlotId,
    },
}

struct VNextExecutionJournal {
    emitter: ExecutionEventEmitter<'static>,
    topology: TrustedExecutionTopology,
    active: TrustedActiveSequenceBinding,
    capture_policy: ExecutionEventCapturePolicy,
    completed_frames: u64,
    started: Instant,
    last_timestamp_nanos: u64,
    root_span: SpanId,
    pending_submission: Option<JournaledSubmission>,
}

impl VNextExecutionJournal {
    fn error(error: impl fmt::Display) -> ExecutionEventSinkError {
        ExecutionEventSinkError::new(error.to_string())
    }

    fn open(
        sink: Arc<dyn ExecutionEventSink>,
        plan: &ExecutionPlan,
        active: TrustedActiveSequenceBinding,
    ) -> std::result::Result<Self, ExecutionEventSinkError> {
        let topology = TrustedExecutionTopology::from_plan(plan).map_err(Self::error)?;
        let root_span =
            SpanId::new(format!("vnext/request/{}", active.fingerprint())).map_err(Self::error)?;
        let capture_policy = sink.capture_policy();
        let mut journal = Self {
            emitter: ExecutionEventEmitter::from_shared(
                sink,
                active.run_id().clone(),
                active.request_id().clone(),
            ),
            topology,
            active,
            capture_policy,
            completed_frames: 0,
            started: Instant::now(),
            last_timestamp_nanos: 0,
            root_span,
            pending_submission: None,
        };
        let accepted = journal.event(
            ExecutionPhase::Resolution,
            ExecutionEventKind::RequestAccepted,
            journal.base_parts(1, journal.root_span.clone(), None),
            ExecutionEventDetail::None,
        )?;
        journal.emitter.emit(
            &accepted,
            &TrustedExecutionEventContext::pre_plan(
                journal.active.run_id(),
                journal.active.request_id(),
            ),
        )?;
        let plan_span = SpanId::new(format!("{}/plan", journal.root_span)).map_err(Self::error)?;
        let planned_parts =
            journal.bind_plan(journal.base_parts(2, plan_span, Some(journal.root_span.clone())));
        let planned = journal.event(
            ExecutionPhase::Planning,
            ExecutionEventKind::PlanBuilt,
            planned_parts,
            ExecutionEventDetail::None,
        )?;
        journal.emitter.emit(
            &planned,
            &TrustedExecutionEventContext::bound(
                journal.active.run_id(),
                journal.active.request_id(),
                &journal.topology,
            ),
        )?;
        Ok(journal)
    }

    fn next_timestamp(&mut self) -> MonotonicTimestamp {
        let elapsed = self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let next = elapsed.max(self.last_timestamp_nanos.saturating_add(1));
        self.last_timestamp_nanos = next;
        MonotonicTimestamp {
            nanos_since_run_start: next,
        }
    }

    fn base_parts(
        &self,
        sequence: u64,
        span_id: SpanId,
        parent_span_id: Option<SpanId>,
    ) -> ExecutionIdentityParts {
        ExecutionIdentityParts {
            version: EXECUTION_IDENTITY_VERSION,
            run_id: self.active.run_id().clone(),
            request_id: self.active.request_id().clone(),
            sequence,
            plan_id: None,
            plan_hash: None,
            frame_id: None,
            node_invocation_id: None,
            node_id: None,
            operation_id: None,
            provider_id: None,
            device_id: None,
            resource_pool_id: None,
            resource_pool_identity_fingerprint: None,
            provisioning_run_id: None,
            provisioning_request_id: None,
            transaction_id: None,
            active_sequence_slot: None,
            admission_generation: None,
            activation_epoch: None,
            runtime_implementation_fingerprint: None,
            active_sequence_fingerprint: None,
            completed_sequence_fingerprint: None,
            aborted_sequence_fingerprint: None,
            resource_id: None,
            resource_generation: None,
            resource_batch_fingerprint: None,
            span_id,
            parent_span_id,
            async_links: Vec::new(),
        }
    }

    fn bind_plan(&self, mut parts: ExecutionIdentityParts) -> ExecutionIdentityParts {
        parts.plan_id = Some(self.topology.plan_id().clone());
        parts.plan_hash = Some(self.topology.plan_hash().clone());
        parts.device_id = Some(self.topology.device_id().clone());
        parts.runtime_implementation_fingerprint = Some(
            self.topology
                .device_runtime_implementation_fingerprint()
                .to_owned(),
        );
        parts
    }

    fn bind_active(&self, mut parts: ExecutionIdentityParts) -> ExecutionIdentityParts {
        let provisioning = self.active.static_provisioning_identity();
        parts.resource_pool_id = self.active.static_pool_id();
        parts.resource_pool_identity_fingerprint = self.active.static_pool_identity_fingerprint();
        parts.provisioning_run_id = provisioning.map(|identity| identity.run_id().clone());
        parts.provisioning_request_id = provisioning.map(|identity| identity.request_id().clone());
        parts.transaction_id = provisioning.map(|identity| identity.transaction_id().clone());
        parts.active_sequence_slot = Some(self.active.sequence_authority().sparse_id());
        parts.admission_generation = Some(self.active.sequence_authority().generation());
        parts.activation_epoch = Some(self.active.activation_epoch());
        parts.active_sequence_fingerprint = Some(self.active.fingerprint().to_owned());
        parts
    }

    fn event(
        &mut self,
        phase: ExecutionPhase,
        kind: ExecutionEventKind,
        parts: ExecutionIdentityParts,
        detail: ExecutionEventDetail,
    ) -> std::result::Result<ExecutionEvent, ExecutionEventSinkError> {
        let identity = ExecutionIdentityEnvelope::new(parts).map_err(Self::error)?;
        ExecutionEvent::new(self.next_timestamp(), phase, kind, identity, detail)
            .map_err(Self::error)
    }

    fn frame_event(
        &mut self,
        operation: &ExecutionIdentityEnvelope,
        kind: ExecutionEventKind,
    ) -> std::result::Result<ExecutionEvent, ExecutionEventSinkError> {
        let operation_parts = operation.parts();
        let frame_id = operation_parts
            .frame_id
            .ok_or_else(|| Self::error("operation identity lacks frame id"))?;
        let sequence = match kind {
            ExecutionEventKind::FrameStarted => operation_parts.sequence.checked_sub(2),
            ExecutionEventKind::FrameCompleted => operation_parts.sequence.checked_add(2),
            _ => None,
        }
        .ok_or_else(|| Self::error("frame event sequence overflow"))?;
        let frame_span =
            SpanId::new(format!("{}/frame/{frame_id}", self.root_span)).map_err(Self::error)?;
        let mut parts = self.bind_active(self.bind_plan(self.base_parts(
            sequence,
            frame_span,
            Some(self.root_span.clone()),
        )));
        parts.frame_id = Some(frame_id);
        self.event(
            ExecutionPhase::Execution,
            kind,
            parts,
            ExecutionEventDetail::None,
        )
    }

    fn node_event(
        &mut self,
        operation: &ExecutionIdentityEnvelope,
        kind: ExecutionEventKind,
    ) -> std::result::Result<ExecutionEvent, ExecutionEventSinkError> {
        let mut parts = operation.parts().clone();
        let node_span = parts
            .parent_span_id
            .clone()
            .ok_or_else(|| Self::error("operation identity lacks node span"))?;
        let frame_id = parts
            .frame_id
            .ok_or_else(|| Self::error("operation identity lacks frame id"))?;
        parts.sequence = match kind {
            ExecutionEventKind::NodeStarted => parts.sequence.checked_sub(1),
            ExecutionEventKind::NodeRetired => parts.sequence.checked_add(1),
            _ => None,
        }
        .ok_or_else(|| Self::error("node event sequence overflow"))?;
        parts.span_id = node_span;
        parts.parent_span_id =
            Some(SpanId::new(format!("{}/frame/{frame_id}", self.root_span)).map_err(Self::error)?);
        self.event(
            ExecutionPhase::Execution,
            kind,
            parts,
            ExecutionEventDetail::None,
        )
    }

    fn operation_event(
        &mut self,
        operation: &ExecutionIdentityEnvelope,
    ) -> std::result::Result<ExecutionEvent, ExecutionEventSinkError> {
        ExecutionEvent::new(
            self.next_timestamp(),
            ExecutionPhase::Execution,
            ExecutionEventKind::OperationSubmitted,
            operation.clone(),
            ExecutionEventDetail::None,
        )
        .map_err(Self::error)
    }

    fn submitted(
        &mut self,
        submission: &SubmittedOperationReceipt,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        if self.pending_submission.is_some() {
            return Err(Self::error(
                "execution journal already has an in-flight physical submission",
            ));
        }
        if !self.capture_policy.captures_frame(self.completed_frames) {
            self.pending_submission = Some(JournaledSubmission::Suppressed {
                slot_id: submission.slot_id(),
            });
            return Ok(());
        }
        let selected = submission
            .participants()
            .iter()
            .enumerate()
            .filter_map(|(index, participant)| {
                let identity = participant.identity().parts();
                (&identity.run_id == self.active.run_id()
                    && &identity.request_id == self.active.request_id())
                .then_some(index)
            })
            .collect::<Vec<_>>();
        let Some(&first_index) = selected.first() else {
            return Err(Self::error(
                "physical submission has no participant for this request journal",
            ));
        };
        let first_identity = submission.participants()[first_index].identity();
        let frame_started = self.frame_event(first_identity, ExecutionEventKind::FrameStarted)?;
        self.emitter.emit(
            &frame_started,
            &TrustedExecutionEventContext::active(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
            ),
        )?;
        let node_started = self.node_event(first_identity, ExecutionEventKind::NodeStarted)?;
        self.emitter.emit(
            &node_started,
            &TrustedExecutionEventContext::active(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
            ),
        )?;
        let operation_submitted = self.operation_event(first_identity)?;
        self.emitter.emit(
            &operation_submitted,
            &TrustedExecutionEventContext::operation_submitted(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
                submission,
            ),
        )?;
        self.pending_submission = Some(JournaledSubmission::Captured {
            receipt: submission.clone(),
            selected,
        });
        Ok(())
    }

    fn completed(
        &mut self,
        completion: &OperationCompletionReceipt,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        let pending = self
            .pending_submission
            .take()
            .ok_or_else(|| Self::error("completion has no journaled physical submission"))?;
        let JournaledSubmission::Captured {
            receipt: submission,
            selected,
        } = pending
        else {
            let JournaledSubmission::Suppressed { slot_id } = pending else {
                unreachable!();
            };
            if completion.submission().slot_id() != slot_id {
                return Err(Self::error(
                    "completion differs from the suppressed journal submission",
                ));
            }
            self.completed_frames = self.completed_frames.saturating_add(1);
            return Ok(());
        };
        if completion.submission().fingerprint() != submission.fingerprint() {
            return Err(Self::error(
                "completion differs from the journaled physical submission",
            ));
        }
        for (position, participant_index) in selected.iter().copied().enumerate() {
            let participant = completion
                .participants()
                .get(participant_index)
                .ok_or_else(|| Self::error("completion participant index is missing"))?;
            let identity = participant.submission().identity();
            let retired = self.node_event(identity, ExecutionEventKind::NodeRetired)?;
            self.emitter.emit(
                &retired,
                &TrustedExecutionEventContext::node_retired(
                    self.active.run_id(),
                    self.active.request_id(),
                    &self.topology,
                    &self.active,
                    participant,
                ),
            )?;
            if let Some(next_index) = selected.get(position + 1).copied() {
                let next_identity = submission.participants()[next_index].identity();
                let started = self.node_event(next_identity, ExecutionEventKind::NodeStarted)?;
                self.emitter.emit(
                    &started,
                    &TrustedExecutionEventContext::active(
                        self.active.run_id(),
                        self.active.request_id(),
                        &self.topology,
                        &self.active,
                    ),
                )?;
                let submitted = self.operation_event(next_identity)?;
                self.emitter.emit(
                    &submitted,
                    &TrustedExecutionEventContext::operation_submitted(
                        self.active.run_id(),
                        self.active.request_id(),
                        &self.topology,
                        &self.active,
                        &submission,
                    ),
                )?;
            }
        }
        let last_index = *selected
            .last()
            .ok_or_else(|| Self::error("completion participant set is empty"))?;
        let frame_completed = self.frame_event(
            submission.participants()[last_index].identity(),
            ExecutionEventKind::FrameCompleted,
        )?;
        self.emitter.emit(
            &frame_completed,
            &TrustedExecutionEventContext::active(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
            ),
        )?;
        self.completed_frames = self.completed_frames.saturating_add(1);
        Ok(())
    }

    fn complete_sequence(
        &mut self,
        receipt: &SequenceSessionTerminalReceipt,
        input_tokens: u64,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        if self.pending_submission.is_some() {
            return Err(Self::error(
                "sequence completed with an in-flight journal submission",
            ));
        }
        let completed =
            TrustedCompletedSequenceBinding::from_session_receipt(receipt, &self.active)
                .map_err(Self::error)?;
        let sequence_number = self.emitter.cursor().last_sequence().saturating_add(1);
        let sequence_span =
            SpanId::new(format!("{}/sequence-completed", self.root_span)).map_err(Self::error)?;
        let mut parts = self.bind_active(self.bind_plan(self.base_parts(
            sequence_number,
            sequence_span,
            Some(self.root_span.clone()),
        )));
        parts.completed_sequence_fingerprint = Some(completed.fingerprint().to_owned());
        let sequence_completed = self.event(
            ExecutionPhase::Completion,
            ExecutionEventKind::SequenceCompleted,
            parts,
            ExecutionEventDetail::None,
        )?;
        self.emitter.emit(
            &sequence_completed,
            &TrustedExecutionEventContext::completed(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
                &completed,
            ),
        )?;
        let request_sequence = self.emitter.cursor().last_sequence().saturating_add(1);
        let mut parts = self.bind_active(self.bind_plan(self.base_parts(
            request_sequence,
            self.root_span.clone(),
            None,
        )));
        parts.completed_sequence_fingerprint = Some(completed.fingerprint().to_owned());
        let request_completed = self.event(
            ExecutionPhase::Completion,
            ExecutionEventKind::RequestCompleted,
            parts,
            ExecutionEventDetail::Counters {
                input: input_tokens,
                output: self.completed_frames,
            },
        )?;
        self.emitter.emit(
            &request_completed,
            &TrustedExecutionEventContext::completed(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
                &completed,
            ),
        )
    }
}

struct VNextSequence<R: DeviceRuntime> {
    cache_id: String,
    request_id: RequestId,
    session: Arc<SequenceSession<R>>,
    tokens: Mutex<Vec<u32>>,
    maximum_tokens: usize,
    active: AtomicBool,
    operation: AsyncMutex<()>,
    events: Option<Mutex<VNextExecutionJournal>>,
    prompt_tokens: u64,
}

impl<R: DeviceRuntime> VNextSequence<R> {
    fn complete(&self) {
        if !self.active.swap(false, Ordering::AcqRel) {
            return;
        }
        if let Ok(receipt) = self.session.try_complete() {
            if let Some(events) = &self.events {
                let _ = events
                    .lock()
                    .complete_sequence(&receipt, self.prompt_tokens);
            }
            return;
        }
        let _ = self.session.request_cancel();
        let _ = self.session.try_abort();
    }

    fn abort(&self) {
        if !self.active.swap(false, Ordering::AcqRel) {
            return;
        }
        let _ = self.session.request_cancel();
        let _ = self.session.try_abort();
    }
}

impl<R: DeviceRuntime> Drop for VNextSequence<R> {
    fn drop(&mut self) {
        if self.active.load(Ordering::Acquire) {
            let _ = self.session.request_cancel();
            let _ = self.session.try_abort();
            self.active.store(false, Ordering::Release);
        }
    }
}

struct VNextKvCacheHandle<R: DeviceRuntime> {
    block_table: BlockTable,
    cache_id: String,
    sequence: Weak<VNextSequence<R>>,
    device: Device,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    maximum_tokens: usize,
}

impl<R: DeviceRuntime> VNextKvCacheHandle<R> {
    fn new(sequence: &Arc<VNextSequence<R>>, info: &ModelInfo, tokens: usize) -> Self {
        let mut block_table = BlockTable::new(16);
        block_table.sequence_length = tokens;
        Self {
            block_table,
            cache_id: sequence.cache_id.clone(),
            sequence: Arc::downgrade(sequence),
            device: info.device.clone(),
            num_layers: info.num_layers,
            num_heads: info.num_kv_heads,
            head_dim: info.hidden_size / info.num_heads.max(1),
            maximum_tokens: sequence.maximum_tokens,
        }
    }
}

impl<R: DeviceRuntime> fmt::Debug for VNextKvCacheHandle<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VNextKvCacheHandle")
            .field("cache_id", &self.cache_id)
            .field("tokens", &self.block_table.sequence_length)
            .field("maximum_tokens", &self.maximum_tokens)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> KvCacheHandle for VNextKvCacheHandle<R> {
    fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn key_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }

    fn value_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }

    fn clone_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
        Err(FerrumError::unsupported(
            "vNext cache cloning requires an explicit typed copy-on-write contract",
        ))
    }

    fn stats(&self) -> CacheHandleStats {
        let tokens = self.block_table.sequence_length;
        CacheHandleStats {
            memory_bytes: 0,
            blocks_allocated: tokens.div_ceil(self.block_table.block_size),
            tokens_stored: tokens,
            utilization: tokens as f32 / self.maximum_tokens.max(1) as f32,
            last_access: Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        self.sequence
            .upgrade()
            .is_some_and(|sequence| sequence.active.load(Ordering::Acquire))
    }

    fn cache_id(&self) -> String {
        self.cache_id.clone()
    }
}

enum DispatchOutcome<R: DeviceRuntime> {
    Submitted(CompletionHandle<R>),
    QuiescentFailure(String),
    SubmissionIndeterminate {
        message: String,
        recovery: IndeterminateSubmissionHandle<R>,
    },
    PostSubmitContract {
        message: String,
        completion: CompletionHandle<R>,
    },
}

enum VNextSequenceAdmissionDecision<R: DeviceRuntime> {
    Admitted(Arc<SequenceSession<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

enum PendingPrefillMaintenance {
    Logical(AdmissionDeferred),
    Backing(DynamicBackingDeferred),
}

struct VNextSequenceRegistry<R: DeviceRuntime> {
    pending: HashMap<RequestId, Arc<VNextSequence<R>>>,
    active: HashMap<String, Arc<VNextSequence<R>>>,
    prefill_maintenance: HashMap<RequestId, PendingPrefillMaintenance>,
}

impl<R: DeviceRuntime> Default for VNextSequenceRegistry<R> {
    fn default() -> Self {
        Self {
            pending: HashMap::new(),
            active: HashMap::new(),
            prefill_maintenance: HashMap::new(),
        }
    }
}

impl<R: DeviceRuntime> VNextSequenceRegistry<R> {
    fn total_len(&self) -> usize {
        self.pending.len() + self.active.len() + self.prefill_maintenance.len()
    }

    fn remove_pending_if(
        &mut self,
        request_id: &RequestId,
        expected: &Arc<VNextSequence<R>>,
    ) -> Option<Arc<VNextSequence<R>>> {
        if self
            .pending
            .get(request_id)
            .is_some_and(|sequence| Arc::ptr_eq(sequence, expected))
        {
            self.pending.remove(request_id)
        } else {
            None
        }
    }

    fn activate(&mut self, request_id: &RequestId, sequence: &Arc<VNextSequence<R>>) -> Result<()> {
        if !self
            .pending
            .get(request_id)
            .is_some_and(|pending| Arc::ptr_eq(pending, sequence))
        {
            return Err(FerrumError::cancelled(format!(
                "vNext prefill admission for `{request_id}` is no longer active"
            )));
        }
        if self.active.contains_key(&sequence.cache_id) {
            return Err(FerrumError::already_exists(format!(
                "vNext cache `{}` raced with another prefill",
                sequence.cache_id
            )));
        }
        let pending = self
            .pending
            .remove(request_id)
            .expect("pointer-checked pending sequence remains present");
        self.active.insert(sequence.cache_id.clone(), pending);
        Ok(())
    }
}

/// Backend-neutral executor over one concrete device runtime and operation
/// registry. CUDA and Metal factories differ only in composition creation.
pub struct VNextModelExecutor<R: DeviceRuntime> {
    info: ModelInfo,
    executable: ExecutablePlan,
    runtime: Arc<R>,
    registry: OperationRuntimeRegistry<R>,
    policy: ResolvedRuntimePolicy,
    plan_resources: Arc<PlanRuntimeResources<R>>,
    lane: Arc<ExecutionLane<R>>,
    reaper: Arc<CompletionReaper<R>>,
    io: VNextIoBinding,
    maximum_model_tokens: usize,
    run_id: RunId,
    family_fingerprint: String,
    program_fingerprint: String,
    static_bytes: u64,
    sequences: Mutex<VNextSequenceRegistry<R>>,
    event_sink: RwLock<Option<Arc<dyn ExecutionEventSink>>>,
    metrics: VNextExecutorMetrics,
}

impl<R: DeviceRuntime> fmt::Debug for VNextModelExecutor<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VNextModelExecutor")
            .field("model_id", &self.info.model_id)
            .field(
                "plan_id",
                self.executable.execution_plan().payload().plan_id(),
            )
            .field("device", &self.runtime.descriptor().id)
            .field("maximum_model_tokens", &self.maximum_model_tokens)
            .field("retained_sequences", &self.sequences.lock().total_len())
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub fn from_runtime_composition(
        prepared: PreparedProductionModel,
        info: ModelInfo,
        engine_config: &EngineConfig,
        runtime: Arc<R>,
        registry: OperationRuntimeRegistry<R>,
        catalog: CapabilityCatalog,
    ) -> Result<Self> {
        let config =
            VNextExecutorConfig::from_engine_config(engine_config, &info, runtime.as_ref())?;
        let family = prepared.family();
        let input_id = match family.program().inputs() {
            [input] => input.clone(),
            inputs => {
                return Err(FerrumError::model(format!(
                    "vNext language executor requires exactly one token input, got {}",
                    inputs.len()
                )))
            }
        };
        let output_id = match family.program().outputs() {
            [output] => output.clone(),
            outputs => {
                return Err(FerrumError::model(format!(
                    "vNext language executor requires exactly one logits output, got {}",
                    outputs.len()
                )))
            }
        };
        let input_capacity = u64::try_from(config.maximum_model_tokens)
            .map_err(|_| FerrumError::config("vNext model length exceeds u64"))?;
        let compile_options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            input_id.clone(),
            ProgramTensorSpec {
                dimensions: vec![input_capacity],
                element_type: ElementType::U32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .map_err(|error| FerrumError::model(format!("vNext compile input: {error}")))?;
        let compilation = ProgramPlanCompiler::compile(
            family,
            &catalog,
            &config.runtime_policy,
            &registry.planning(),
            &compile_options,
        )
        .map_err(|error| FerrumError::model(format!("vNext plan compile: {error}")))?;
        let (executable, _, _) = compilation.into_parts();
        let io = Self::resolve_io(&executable, &input_id, &output_id, info.vocab_size)?;
        let family_fingerprint = family
            .fingerprint()
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let program_fingerprint = family
            .program()
            .fingerprint()
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let static_bytes = executable
            .execution_plan()
            .payload()
            .memory()
            .static_bytes();
        let run_id = RunId::new(format!("run.vnext.{}", uuid::Uuid::new_v4()))
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let provision_request = RequestIdentity::new(format!("request.vnext.provision.{run_id}"))
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let provisioned = executable
            .execution_plan()
            .provision_static(Arc::clone(&runtime), provision_request)
            .map_err(|error| FerrumError::device(format!("vNext static provision: {error}")))?;
        let plan_resources = match provisioned.into_provisioning() {
            StaticProvisioning::NoStatic(no_static) => no_static.into_plan_runtime(),
            StaticProvisioning::Required(permit) => {
                let identity = ResourceTransactionIdentity::for_admission(
                    permit.binding(),
                    run_id.clone(),
                    TransactionId::new(format!("transaction.vnext.provision.{run_id}"))
                        .map_err(|error| FerrumError::internal(error.to_string()))?,
                );
                let driver = RuntimeResourceDriver::new(Arc::clone(&runtime))
                    .map_err(|error| FerrumError::device(error.to_string()))?;
                let transaction = ResourceTransaction::<VNextDriver<R>, TransactionNew>::begin(
                    driver, identity, permit,
                )
                .map_err(|error| FerrumError::device(error.to_string()))?;
                let reserved = match transaction.reserve() {
                    Ok(reserved) => reserved,
                    Err(error) => {
                        let message = format!("{:?}", error.failure());
                        drop(error);
                        return Err(FerrumError::device(format!(
                            "vNext static reserve failed: {message}"
                        )));
                    }
                };
                let committed = match reserved.commit() {
                    Ok(committed) => committed,
                    Err(ResourceCommitTransitionError::Recoverable(error)) => {
                        let message = format!("{:?}", error.failure());
                        drop(error);
                        return Err(FerrumError::device(format!(
                            "vNext static commit failed: {message}"
                        )));
                    }
                    Err(ResourceCommitTransitionError::Poisoned(error)) => {
                        let message = format!("{:?}", error.failure());
                        drop(error);
                        return Err(FerrumError::device(format!(
                            "vNext static commit was indeterminate: {message}"
                        )));
                    }
                };
                let initialized = match committed.initialize_static(
                    family,
                    executable.execution_plan(),
                    prepared.weights(),
                    config.static_initialization,
                ) {
                    Ok(initialized) => initialized,
                    Err(error) => {
                        let message = error.failure().message().to_owned();
                        drop(error);
                        return Err(FerrumError::device(format!(
                            "vNext static initialization failed: {message}"
                        )));
                    }
                };
                match initialized.into_plan_runtime() {
                    Ok(resources) => resources,
                    Err(error) => {
                        let message = error.error().to_string();
                        drop(error);
                        return Err(FerrumError::device(format!(
                            "vNext runtime handoff failed: {message}"
                        )));
                    }
                }
            }
        };
        let lane = ExecutionLane::create(Arc::clone(&runtime)).map_err(|error| {
            FerrumError::device(format!("vNext execution lane creation failed: {error:?}"))
        })?;
        let reaper = CompletionReaper::new();

        Ok(Self {
            info,
            executable,
            runtime,
            registry,
            policy: config.runtime_policy,
            plan_resources,
            lane,
            reaper,
            io,
            maximum_model_tokens: config.maximum_model_tokens,
            run_id,
            family_fingerprint,
            program_fingerprint,
            static_bytes,
            sequences: Mutex::new(VNextSequenceRegistry::default()),
            event_sink: RwLock::new(None),
            metrics: VNextExecutorMetrics::default(),
        })
    }

    fn resolve_io(
        executable: &ExecutablePlan,
        input_id: &ProgramValueId,
        output_id: &ProgramValueId,
        expected_vocab: usize,
    ) -> Result<VNextIoBinding> {
        let nodes = executable.execution_plan().payload().nodes();
        let input_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == input_id && value.role() == ResolvedValueRole::Input
            })
            .collect::<Vec<_>>();
        let output_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == output_id && value.role() == ResolvedValueRole::Output
            })
            .collect::<Vec<_>>();
        let [(input_node_id, input)] = input_matches.as_slice() else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind the token input exactly once",
            ));
        };
        let [(output_node_id, output)] = output_matches.as_slice() else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind the logits output exactly once",
            ));
        };
        if input.tensor().element_type() != ElementType::U32 {
            return Err(FerrumError::model(
                "compiled vNext token input must use U32 elements",
            ));
        }
        let [component] = output.storage().components() else {
            return Err(FerrumError::model(
                "compiled vNext logits output must use one physical component",
            ));
        };
        let output_elements_u64 = output
            .tensor()
            .dimensions()
            .iter()
            .try_fold(1_u64, |total, extent| total.checked_mul(*extent))
            .ok_or_else(|| FerrumError::model("vNext logits element count overflows u64"))?;
        let output_elements = usize::try_from(output_elements_u64)
            .map_err(|_| FerrumError::model("vNext logits exceed host address space"))?;
        if output_elements != expected_vocab {
            return Err(FerrumError::model(format!(
                "compiled vNext logits contain {output_elements} elements, expected vocabulary {expected_vocab}"
            )));
        }
        let output_element_type = output.tensor().element_type();
        if !matches!(
            output_element_type,
            ElementType::F16 | ElementType::Bf16 | ElementType::F32
        ) {
            return Err(FerrumError::model(format!(
                "compiled vNext logits use unsupported element type {output_element_type:?}"
            )));
        }
        let output_layout = HostTransferLayout::new(output_element_type, output_elements_u64)
            .map_err(|error| FerrumError::model(error.to_string()))?;
        Ok(VNextIoBinding {
            input_node_id: (*input_node_id).clone(),
            input_ordinal: input.ordinal(),
            output_node_id: (*output_node_id).clone(),
            output_resource_id: component.resource_id().clone(),
            output_offset_bytes: component.offset_bytes(),
            output_layout,
            output_element_type,
            output_elements,
        })
    }

    fn maintain_backing(&self, deferred: &DynamicBackingDeferred) -> Result<()> {
        self.plan_resources
            .maintain_for_deferred(deferred)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        Ok(())
    }

    fn retain_prefill_maintenance(
        &self,
        request_id: &RequestId,
        pending: PendingPrefillMaintenance,
    ) -> Result<ExecutorPrefillMaintenanceDeferral> {
        let projection = match &pending {
            PendingPrefillMaintenance::Logical(deferred) => {
                ExecutorPrefillMaintenanceDeferral::from_admission(request_id, deferred)?
            }
            PendingPrefillMaintenance::Backing(deferred) => {
                ExecutorPrefillMaintenanceDeferral::from_backing(request_id, deferred)?
            }
        };
        let mut sequences = self.sequences.lock();
        if sequences.pending.contains_key(request_id)
            || sequences.prefill_maintenance.contains_key(request_id)
            || sequences
                .active
                .values()
                .any(|sequence| sequence.request_id == *request_id)
        {
            return Err(FerrumError::already_exists(format!(
                "vNext request `{request_id}` already retained admission state"
            )));
        }
        sequences
            .prefill_maintenance
            .insert(request_id.clone(), pending);
        Ok(projection)
    }

    fn current_prefill_admission_epochs(&self) -> Result<ExecutorAdmissionEpochs> {
        self.plan_resources
            .dynamic_pool_status()
            .map(|status| ExecutorAdmissionEpochs::from_capacity(status.epochs()))
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn maintain_admission_growth(
        &self,
        scope: &str,
        deferred: &AdmissionDeferred,
        attempts: &mut u32,
    ) -> Result<bool> {
        if deferred.action() != DeferredAction::AwaitBackingGrowth {
            return Ok(false);
        }
        if *attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
            return Err(Self::deferred(scope, deferred));
        }
        *attempts += 1;
        self.plan_resources
            .maintain_for_admission_deferred(deferred)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        Ok(true)
    }

    fn deferred(scope: &str, deferred: &AdmissionDeferred) -> FerrumError {
        FerrumError::resource_exhausted(format!(
            "vNext {scope} deferred with action {:?} until capacity epoch changes: {:?}",
            deferred.action(),
            deferred.blockers()
        ))
    }

    fn backing_deferred(scope: &str, deferred: &DynamicBackingDeferred) -> FerrumError {
        FerrumError::resource_exhausted(format!(
            "vNext {scope} remains deferred after bounded backing maintenance: {:?}",
            deferred.blockers()
        ))
    }

    fn try_admit_sequence(
        &self,
        request_id: RequestIdentity,
        work: ResourceWorkShape,
    ) -> Result<VNextSequenceAdmissionDecision<R>> {
        let binding = self
            .plan_resources
            .trusted_runtime_binding()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let admission = RequestResourceAdmissionRequest::new(
            work.clone(),
            AdmissionFitPolicy::FullInputMustFit,
            AdmissionPressureAction::WaitForRelease,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let request = match binding
            .try_admit_request(admission, self.run_id.clone(), request_id)
            .map_err(|error| FerrumError::backend(error.to_string()))?
        {
            RequestResourceAdmissionDecision::Admitted(request) => request,
            RequestResourceAdmissionDecision::Deferred(deferred) => {
                self.metrics
                    .request_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(VNextSequenceAdmissionDecision::Deferred(deferred));
            }
            RequestResourceAdmissionDecision::BackingDeferred(deferred) => {
                self.metrics
                    .backing_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(VNextSequenceAdmissionDecision::BackingDeferred(deferred));
            }
            RequestResourceAdmissionDecision::PermanentRejected(rejected) => {
                return Ok(VNextSequenceAdmissionDecision::PermanentRejected(rejected));
            }
        };

        let admission = SequenceResourceAdmissionRequest::new(
            work,
            AdmissionFitPolicy::FullInputMustFit,
            AdmissionPressureAction::WaitForRelease,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let sequence = match request
            .try_admit_sequence(admission)
            .map_err(|error| FerrumError::backend(error.to_string()))?
        {
            SequenceResourceAdmissionDecision::Admitted(sequence) => sequence,
            SequenceResourceAdmissionDecision::Deferred(deferred) => {
                self.metrics
                    .sequence_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(VNextSequenceAdmissionDecision::Deferred(deferred));
            }
            SequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                self.metrics
                    .backing_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(VNextSequenceAdmissionDecision::BackingDeferred(deferred));
            }
            SequenceResourceAdmissionDecision::PermanentRejected(rejected) => {
                return Ok(VNextSequenceAdmissionDecision::PermanentRejected(rejected));
            }
        };
        let session = sequence
            .open_session()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        Ok(VNextSequenceAdmissionDecision::Admitted(session))
    }

    fn execution_journal(
        &self,
        session: &Arc<SequenceSession<R>>,
    ) -> Result<Option<VNextExecutionJournal>> {
        let Some(sink) = self.event_sink.read().clone() else {
            return Ok(None);
        };
        let active = TrustedActiveSequenceBinding::from_session(session)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        VNextExecutionJournal::open(sink, self.executable.execution_plan(), active)
            .map(Some)
            .map_err(|error| FerrumError::backend(format!("vNext execution journal: {error}")))
    }

    fn extend_sequence(
        &self,
        sequence: &VNextSequence<R>,
        target: ResourceWorkShape,
    ) -> Result<()> {
        let mut backing_attempts = 0;
        let mut rechecks = 0;
        loop {
            if !sequence.active.load(Ordering::Acquire) {
                return Err(FerrumError::cancelled(
                    "vNext sequence was released while awaiting capacity",
                ));
            }
            let request = SequenceResourceExtensionRequest::new(
                target.clone(),
                AdmissionPressureAction::WaitForRelease,
            )
            .map_err(|error| FerrumError::backend(error.to_string()))?;
            match sequence
                .session
                .try_extend_backing(request)
                .map_err(|error| FerrumError::backend(error.to_string()))?
            {
                SequenceResourceExtensionDecision::Current(_)
                | SequenceResourceExtensionDecision::Extended(_) => return Ok(()),
                SequenceResourceExtensionDecision::RetryRequired(_) => {
                    if rechecks >= MAX_EXTENSION_RECHECKS {
                        return Err(FerrumError::resource_exhausted(
                            "vNext sequence extension is waiting for the prior frame to retire",
                        ));
                    }
                    rechecks += 1;
                    std::thread::yield_now();
                }
                SequenceResourceExtensionDecision::Deferred(deferred) => {
                    self.metrics
                        .extension_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if self.maintain_admission_growth(
                        "sequence extension",
                        &deferred,
                        &mut backing_attempts,
                    )? {
                        continue;
                    }
                    return Err(Self::deferred("sequence extension", &deferred));
                }
                SequenceResourceExtensionDecision::BackingDeferred(deferred) => {
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::backing_deferred("sequence extension", &deferred));
                    }
                    backing_attempts += 1;
                    self.maintain_backing(&deferred)?;
                }
                SequenceResourceExtensionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::request_validation(format!(
                        "vNext sequence extension exceeds the configured fit ceiling: {rejected:?}"
                    )))
                }
            }
        }
    }

    fn begin_step(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        span: &TokenSpanWork,
    ) -> Result<Arc<StepResourceLease<R>>> {
        let mut backing_attempts = 0;
        loop {
            let request = StepResourceAdmissionRequest::new(
                batch
                    .bind_work_shape(vec![span.clone()])
                    .map_err(|error| FerrumError::backend(error.to_string()))?,
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .map_err(|error| FerrumError::backend(error.to_string()))?;
            match batch
                .try_begin_step(request)
                .map_err(|error| FerrumError::backend(error.to_string()))?
            {
                StepResourceAdmissionDecision::Admitted(step) => return Ok(step),
                StepResourceAdmissionDecision::Deferred(deferred) => {
                    self.metrics.step_deferrals.fetch_add(1, Ordering::Relaxed);
                    if self.maintain_admission_growth(
                        "step admission",
                        &deferred,
                        &mut backing_attempts,
                    )? {
                        continue;
                    }
                    return Err(Self::deferred("step admission", &deferred));
                }
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::backing_deferred("step admission", &deferred));
                    }
                    backing_attempts += 1;
                    self.maintain_backing(&deferred)?;
                }
                StepResourceAdmissionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::backend(format!(
                        "vNext execution step exceeds its immutable plan: {rejected:?}"
                    )))
                }
            }
        }
    }

    fn prepare_wave(
        &self,
        step: &Arc<StepResourceLease<R>>,
        span: &TokenSpanWork,
    ) -> Result<PreparedStepSubmissionWave<R>> {
        let mut backing_attempts = 0;
        loop {
            let requests = self
                .executable
                .execution_plan()
                .payload()
                .nodes()
                .iter()
                .map(|node| {
                    InvocationResourceAdmissionRequest::for_all_step_participants(
                        node.id().clone(),
                        step.bind_all_invocation_work_shape(vec![span.clone()])?,
                        AdmissionFitPolicy::ImmediateOnly,
                        AdmissionPressureAction::WaitForRelease,
                    )
                })
                .collect::<std::result::Result<Vec<_>, VNextError>>()
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            match step
                .try_prepare_submission_wave(requests)
                .map_err(|error| FerrumError::backend(error.to_string()))?
            {
                StepSubmissionWaveAdmissionDecision::Prepared(wave) => return Ok(wave),
                StepSubmissionWaveAdmissionDecision::Deferred(deferred) => {
                    self.metrics.wave_deferrals.fetch_add(1, Ordering::Relaxed);
                    if self.maintain_admission_growth(
                        "submission wave",
                        &deferred,
                        &mut backing_attempts,
                    )? {
                        continue;
                    }
                    return Err(Self::deferred("submission wave", &deferred));
                }
                StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::backing_deferred("submission wave", &deferred));
                    }
                    backing_attempts += 1;
                    self.maintain_backing(&deferred)?;
                }
                StepSubmissionWaveAdmissionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::backend(format!(
                        "vNext submission wave exceeds its immutable plan: {rejected:?}"
                    )))
                }
            }
        }
    }

    fn dispatch_wave(
        &self,
        session: &Arc<SequenceSession<R>>,
        tokens: &[u32],
        span: &TokenSpanWork,
        wave: PreparedStepSubmissionWave<R>,
    ) -> DispatchOutcome<R> {
        let active = match TrustedActiveSequenceBinding::from_session(session) {
            Ok(active) => active,
            Err(error) => return DispatchOutcome::QuiescentFailure(error.to_string()),
        };
        let active_bindings = wave
            .nodes()
            .iter()
            .map(|_| vec![active.clone()])
            .collect::<Vec<_>>();
        let providers = match self
            .executable
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .map(|node| self.registry.bind(&self.executable, node.id()))
            .collect::<std::result::Result<Vec<_>, VNextError>>()
        {
            Ok(providers) => providers,
            Err(error) => return DispatchOutcome::QuiescentFailure(error.to_string()),
        };
        let range = span.immediate_token_range();
        let host_range = Range {
            start: range.start as usize,
            end: range.end as usize,
        };
        let Some(host_tokens) = tokens.get(host_range.clone()) else {
            return DispatchOutcome::QuiescentFailure(format!(
                "vNext token upload range {host_range:?} exceeds host token length {}",
                tokens.len()
            ));
        };
        let host_bytes = host_tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        let logical_offset_bytes = match range.start.checked_mul(ElementType::U32.size_bytes()) {
            Some(offset) => offset,
            None => {
                return DispatchOutcome::QuiescentFailure(
                    "vNext token upload offset overflows u64".to_owned(),
                )
            }
        };
        let source_layout = match HostTransferLayout::new(ElementType::U32, span.immediate_tokens())
        {
            Ok(layout) => layout,
            Err(error) => return DispatchOutcome::QuiescentFailure(error.to_string()),
        };
        let upload = match SubmissionWaveInputUpload::new(
            self.io.input_node_id.clone(),
            0,
            self.io.input_ordinal,
            logical_offset_bytes,
            source_layout,
            host_bytes,
        ) {
            Ok(upload) => upload,
            Err(error) => return DispatchOutcome::QuiescentFailure(error.to_string()),
        };
        self.metrics
            .uploaded_bytes
            .fetch_add(source_layout.byte_len().unwrap_or(0), Ordering::Relaxed);

        let mut wave = wave;
        let mut retries = 0;
        loop {
            let identity = match OperationDispatch::bind_submission_wave_identity(
                &self.executable,
                &active_bindings,
                &wave,
                &self.lane,
            ) {
                Ok(identity) => identity,
                Err(error) => return DispatchOutcome::QuiescentFailure(error.to_string()),
            };
            match OperationDispatch::encode_and_submit_wave_with_inputs(
                &providers,
                &self.executable,
                &identity,
                &active_bindings,
                std::slice::from_ref(&upload),
                wave,
                &self.lane,
                &self.reaper,
            ) {
                Ok(completion) => {
                    self.metrics.submitted_waves.fetch_add(1, Ordering::Relaxed);
                    return DispatchOutcome::Submitted(completion);
                }
                Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { failures, retry })
                    if retries < MAX_DEFINITELY_NOT_SUBMITTED_RETRIES =>
                {
                    retries += 1;
                    self.metrics
                        .definitely_not_submitted_retries
                        .fetch_add(1, Ordering::Relaxed);
                    match retry.retry() {
                        Ok(retry_wave) => wave = retry_wave,
                        Err(error) => {
                            return DispatchOutcome::QuiescentFailure(format!(
                                "vNext wave retry authority failed after {failures:?}: {error}"
                            ))
                        }
                    }
                }
                Err(error @ SubmissionWaveDispatchError::DefinitelyNotSubmitted { .. })
                | Err(error @ SubmissionWaveDispatchError::Contract(_))
                | Err(error @ SubmissionWaveDispatchError::Provider(_))
                | Err(error @ SubmissionWaveDispatchError::InputUpload(_)) => {
                    return DispatchOutcome::QuiescentFailure(error.to_string())
                }
                Err(SubmissionWaveDispatchError::SubmissionIndeterminate { recovery }) => {
                    return DispatchOutcome::SubmissionIndeterminate {
                        message: "vNext wave submission is indeterminate".to_owned(),
                        recovery,
                    }
                }
                Err(SubmissionWaveDispatchError::PostSubmitContract { error, completion }) => {
                    return DispatchOutcome::PostSubmitContract {
                        message: error.to_string(),
                        completion,
                    }
                }
            }
        }
    }

    async fn abort_step(
        &self,
        step: Arc<StepResourceLease<R>>,
        message: impl Into<String>,
    ) -> FerrumError {
        let message = message.into();
        self.metrics.record_failure(message.clone());
        match step.try_abort() {
            Ok(_) => FerrumError::backend(message),
            Err(failure) => FerrumError::backend(format!(
                "{message}; vNext step abort failed: {}",
                failure.error()
            )),
        }
    }

    fn abort_unsubmitted_step(
        &self,
        step: Arc<StepResourceLease<R>>,
        error: FerrumError,
    ) -> FerrumError {
        if !matches!(&error, FerrumError::ResourceExhausted { .. }) {
            self.metrics.record_failure(error.to_string());
        }
        match step.try_abort() {
            Ok(_) => error,
            Err(failure) => FerrumError::backend(format!(
                "{error}; vNext unsubmitted step abort failed: {}",
                failure.error()
            )),
        }
    }

    async fn execute_step(
        &self,
        sequence: &Arc<VNextSequence<R>>,
        tokens: &[u32],
        span: TokenSpanWork,
    ) -> Result<Vec<f32>> {
        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&sequence.session)])
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let step = self.begin_step(&batch, &span)?;
        let wave = match self.prepare_wave(&step, &span) {
            Ok(wave) => wave,
            Err(error) => return Err(self.abort_unsubmitted_step(step, error)),
        };
        let dispatch = self.dispatch_wave(&sequence.session, tokens, &span, wave);
        let completion = match dispatch {
            DispatchOutcome::Submitted(completion) => completion,
            DispatchOutcome::QuiescentFailure(message) => {
                return Err(self.abort_step(step, message).await)
            }
            DispatchOutcome::SubmissionIndeterminate { message, recovery } => {
                let recovered =
                    tokio::task::spawn_blocking(move || recovery.recover_by_draining_lane())
                        .await
                        .map_err(|error| FerrumError::backend(format!("{message}: {error}")))?;
                match recovered {
                    Ok(_) => return Err(self.abort_step(step, message).await),
                    Err(error) => {
                        self.metrics
                            .record_failure(format!("{message}; recovery failed: {error}"));
                        return Err(FerrumError::backend(format!(
                            "{message}; recovery failed: {error}"
                        )));
                    }
                }
            }
            DispatchOutcome::PostSubmitContract {
                message,
                completion,
            } => {
                let observed = tokio::task::spawn_blocking(move || completion.wait())
                    .await
                    .map_err(|error| FerrumError::backend(format!("{message}: {error}")))?;
                match observed {
                    Ok(CompletionObservation::Terminal(_)) => {
                        return Err(self.abort_step(step, message).await)
                    }
                    Ok(other) => {
                        self.metrics.record_failure(format!(
                            "{message}; post-submit drain remained nonterminal: {other:?}"
                        ));
                        return Err(FerrumError::backend(format!(
                            "{message}; post-submit drain remained nonterminal"
                        )));
                    }
                    Err(error) => {
                        self.metrics
                            .record_failure(format!("{message}; drain failed: {error}"));
                        return Err(FerrumError::backend(format!(
                            "{message}; drain failed: {error}"
                        )));
                    }
                }
            }
        };
        let mut execution_event_error = sequence.events.as_ref().and_then(|events| {
            events
                .lock()
                .submitted(completion.receipt())
                .err()
                .map(|error| error.to_string())
        });

        let readback = CompletionReadbackRequest::new(
            self.io.output_node_id.clone(),
            0,
            self.io.output_resource_id.clone(),
            self.io.output_offset_bytes,
            self.io.output_layout,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let observation =
            tokio::task::spawn_blocking(move || completion.wait_with_readback(readback))
                .await
                .map_err(|error| {
                    FerrumError::backend(format!("vNext completion task failed: {error}"))
                })?
                .map_err(|error| FerrumError::backend(error.to_string()))?;
        let receipt = match observation {
            CompletionReadbackObservation::Terminal(receipt) => receipt,
            other => {
                self.metrics
                    .record_failure(format!("vNext completion remained nonterminal: {other:?}"));
                return Err(FerrumError::backend(
                    "vNext completion did not reach a quiescent terminal",
                ));
            }
        };
        if execution_event_error.is_none() {
            execution_event_error = sequence.events.as_ref().and_then(|events| {
                events
                    .lock()
                    .completed(receipt.completion())
                    .err()
                    .map(|error| error.to_string())
            });
        }
        if !matches!(
            receipt.completion().disposition(),
            OperationCompletionDisposition::Succeeded
        ) {
            let message = format!(
                "vNext device wave failed: {:?}",
                receipt.completion().disposition()
            );
            drop(receipt);
            return Err(self.abort_step(step, message).await);
        }
        let logits = match receipt.disposition() {
            CompletionReadbackDisposition::Succeeded(output) => {
                match Self::decode_logits(output.bytes(), self.io.output_element_type) {
                    Ok(logits) => logits,
                    Err(error) => {
                        drop(receipt);
                        return Err(self.abort_step(step, error.to_string()).await);
                    }
                }
            }
            disposition => {
                let message = format!("vNext logits readback failed: {disposition:?}");
                drop(receipt);
                return Err(self.abort_step(step, message).await);
            }
        };
        self.metrics.readback_bytes.fetch_add(
            self.io.output_layout.byte_len().unwrap_or(0),
            Ordering::Relaxed,
        );
        drop(receipt);
        match step.try_retire_normal() {
            Ok(_) => {
                self.metrics.completed_waves.fetch_add(1, Ordering::Relaxed);
                if let Some(error) = execution_event_error {
                    let message = format!("vNext execution event emission failed: {error}");
                    self.metrics.record_failure(message.clone());
                    Err(FerrumError::backend(message))
                } else {
                    Ok(logits)
                }
            }
            Err(failure) => {
                let message = format!("vNext step retirement failed: {}", failure.error());
                self.metrics.record_failure(message.clone());
                Err(FerrumError::backend(message))
            }
        }
    }

    fn decode_logits(bytes: &[u8], element_type: ElementType) -> Result<Vec<f32>> {
        match element_type {
            ElementType::F16 => Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                })
                .collect()),
            ElementType::Bf16 => Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                })
                .collect()),
            ElementType::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            other => Err(FerrumError::backend(format!(
                "unsupported vNext logits element type {other:?}"
            ))),
        }
    }

    fn prefill_tensor(&self, logits: Vec<f32>) -> Result<TensorRef> {
        let tensor = candle_core::Tensor::from_vec(
            logits,
            (1, 1, self.io.output_elements),
            &candle_core::Device::Cpu,
        )
        .map_err(|error| FerrumError::model(format!("vNext prefill logits tensor: {error}")))?;
        Ok(common::wrap_tensor(tensor))
    }

    fn decode_tensor(&self, logits: Vec<f32>) -> Result<TensorRef> {
        let tensor = candle_core::Tensor::from_vec(
            logits,
            (1, self.io.output_elements),
            &candle_core::Device::Cpu,
        )
        .map_err(|error| FerrumError::model(format!("vNext decode logits tensor: {error}")))?;
        Ok(common::wrap_tensor(tensor))
    }

    fn cache_handle(
        &self,
        sequence: &Arc<VNextSequence<R>>,
        tokens: usize,
    ) -> Arc<dyn KvCacheHandle> {
        Arc::new(VNextKvCacheHandle::new(sequence, &self.info, tokens))
    }

    fn sequence_for_cache(&self, cache_id: &str) -> Result<Arc<VNextSequence<R>>> {
        self.sequences
            .lock()
            .active
            .get(cache_id)
            .cloned()
            .ok_or_else(|| {
                FerrumError::not_found(format!("vNext cache `{cache_id}` is not active"))
            })
    }

    fn metrics_snapshot(&self) -> serde_json::Value {
        let pool_status = self
            .plan_resources
            .dynamic_pool_status()
            .ok()
            .and_then(|status| serde_json::to_value(status).ok());
        let cleanup = serde_json::to_value(self.plan_resources.deferred_cleanup_status()).ok();
        let (pending_sequences, active_sequences, pending_prefill_maintenance) = {
            let sequences = self.sequences.lock();
            (
                sequences.pending.len(),
                sequences.active.len(),
                sequences.prefill_maintenance.len(),
            )
        };
        serde_json::json!({
            "schema": "ferrum.runtime-vnext.executor-trace.v1",
            "model_id": self.info.model_id.to_string(),
            "family_fingerprint": self.family_fingerprint,
            "program_fingerprint": self.program_fingerprint,
            "plan_id": self.executable.execution_plan().payload().plan_id().to_string(),
            "plan_hash": self.executable.execution_plan().plan_hash().to_string(),
            "policy_id": self.policy.policy_id(),
            "policy_fingerprint": self.policy.fingerprint_str(),
            "device_id": self.runtime.descriptor().id.to_string(),
            "runtime_fingerprint": self.runtime.descriptor().runtime_implementation_fingerprint,
            "maximum_model_tokens": self.maximum_model_tokens,
            "runtime_memory_policy": self.policy.memory(),
            "pending_sequences": pending_sequences,
            "active_sequences": active_sequences,
            "pending_prefill_maintenance": pending_prefill_maintenance,
            "static_bytes": self.static_bytes,
            "counters": {
                "prefill_operations": self.metrics.prefill_operations.load(Ordering::Relaxed),
                "decode_operations": self.metrics.decode_operations.load(Ordering::Relaxed),
                "submitted_waves": self.metrics.submitted_waves.load(Ordering::Relaxed),
                "completed_waves": self.metrics.completed_waves.load(Ordering::Relaxed),
                "failed_waves": self.metrics.failed_waves.load(Ordering::Relaxed),
                "definitely_not_submitted_retries": self.metrics.definitely_not_submitted_retries.load(Ordering::Relaxed),
                "request_deferrals": self.metrics.request_deferrals.load(Ordering::Relaxed),
                "sequence_deferrals": self.metrics.sequence_deferrals.load(Ordering::Relaxed),
                "extension_deferrals": self.metrics.extension_deferrals.load(Ordering::Relaxed),
                "step_deferrals": self.metrics.step_deferrals.load(Ordering::Relaxed),
                "wave_deferrals": self.metrics.wave_deferrals.load(Ordering::Relaxed),
                "backing_deferrals": self.metrics.backing_deferrals.load(Ordering::Relaxed),
                "uploaded_bytes": self.metrics.uploaded_bytes.load(Ordering::Relaxed),
                "readback_bytes": self.metrics.readback_bytes.load(Ordering::Relaxed),
            },
            "dynamic_pools": pool_status,
            "deferred_cleanup": cleanup,
            "last_failure": self.metrics.last_failure.lock().clone(),
        })
    }
}

#[async_trait::async_trait]
impl<R: DeviceRuntime> ModelExecutor for VNextModelExecutor<R> {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn execution_resource_ownership(&self) -> ExecutionResourceOwnership {
        ExecutionResourceOwnership::ExecutorManaged
    }

    fn kv_capacity(&self) -> Option<usize> {
        Some(self.maximum_model_tokens)
    }

    fn attach_execution_event_sink(&self, sink: Arc<dyn ExecutionEventSink>) {
        *self.event_sink.write() = Some(sink);
    }

    fn prefill_admission_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        self.plan_resources
            .dynamic_pool_status()
            .map(|status| Some(ExecutorAdmissionEpochs::from_capacity(status.epochs())))
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        if input.input_tokens.is_empty() {
            return Err(FerrumError::request_validation(
                "executor-owned vNext prefill admission requires at least one input token",
            ));
        }
        if input.maximum_sequence_tokens < input.input_tokens.len()
            || input.maximum_sequence_tokens > self.maximum_model_tokens
        {
            return Err(FerrumError::request_validation(format!(
                "request sequence ceiling {} must cover prompt {} and not exceed {}",
                input.maximum_sequence_tokens,
                input.input_tokens.len(),
                self.maximum_model_tokens
            )));
        }
        {
            let sequences = self.sequences.lock();
            if sequences.pending.contains_key(input.request_id)
                || sequences.prefill_maintenance.contains_key(input.request_id)
                || sequences
                    .active
                    .values()
                    .any(|sequence| sequence.request_id == *input.request_id)
            {
                return Err(FerrumError::already_exists(format!(
                    "vNext request `{}` already retained an admission",
                    input.request_id
                )));
            }
        }

        let tokens = input
            .input_tokens
            .iter()
            .map(|token| token.get())
            .collect::<Vec<_>>();
        let span = TokenSpanWork::from_token_ids_with_fit(
            &tokens,
            0..tokens.len(),
            input.maximum_sequence_tokens,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let work = ResourceWorkShape::single(span)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let identity = RequestIdentity::new(format!("request.product.{}", input.request_id))
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let session = match self.try_admit_sequence(identity, work)? {
            VNextSequenceAdmissionDecision::Admitted(session) => session,
            VNextSequenceAdmissionDecision::Deferred(deferred) => {
                if deferred.action() == DeferredAction::AwaitBackingGrowth {
                    let projection = self.retain_prefill_maintenance(
                        input.request_id,
                        PendingPrefillMaintenance::Logical(deferred),
                    )?;
                    return Ok(ExecutorPrefillAdmissionDecision::MaintenanceDeferred(
                        projection,
                    ));
                }
                return Ok(ExecutorPrefillAdmissionDecision::Deferred(deferred));
            }
            VNextSequenceAdmissionDecision::BackingDeferred(deferred) => {
                let projection = self.retain_prefill_maintenance(
                    input.request_id,
                    PendingPrefillMaintenance::Backing(deferred),
                )?;
                return Ok(ExecutorPrefillAdmissionDecision::MaintenanceDeferred(
                    projection,
                ));
            }
            VNextSequenceAdmissionDecision::PermanentRejected(rejected) => {
                return Ok(ExecutorPrefillAdmissionDecision::PermanentRejected(
                    rejected,
                ));
            }
        };
        let events = match self.execution_journal(&session) {
            Ok(events) => events,
            Err(error) => {
                let _ = session.request_cancel();
                let _ = session.try_abort();
                return Err(error);
            }
        };
        let prompt_tokens = u64::try_from(tokens.len())
            .map_err(|_| FerrumError::request_validation("prompt token count exceeds u64"))?;
        let sequence = Arc::new(VNextSequence {
            cache_id: format!("vnext-cache-{}", input.request_id),
            request_id: input.request_id.clone(),
            session,
            tokens: Mutex::new(tokens),
            maximum_tokens: input.maximum_sequence_tokens,
            active: AtomicBool::new(true),
            operation: AsyncMutex::new(()),
            events: events.map(Mutex::new),
            prompt_tokens,
        });
        let raced = {
            let mut sequences = self.sequences.lock();
            if sequences.pending.contains_key(input.request_id)
                || sequences.prefill_maintenance.contains_key(input.request_id)
                || sequences.active.contains_key(&sequence.cache_id)
            {
                true
            } else {
                sequences
                    .pending
                    .insert(input.request_id.clone(), sequence.clone());
                false
            }
        };
        if raced {
            sequence.abort();
            return Err(FerrumError::already_exists(format!(
                "vNext request `{}` raced with another admission",
                input.request_id
            )));
        }
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }

    fn cancel_prefill_admission(&self, request_id: &RequestId) -> bool {
        let (pending, maintenance) = {
            let mut sequences = self.sequences.lock();
            (
                sequences.pending.remove(request_id),
                sequences.prefill_maintenance.remove(request_id),
            )
        };
        let removed = pending.is_some() || maintenance.is_some();
        if let Some(sequence) = pending {
            sequence.abort();
        }
        removed
    }

    fn maintain_prefill_backing(
        &self,
        request_id: &RequestId,
    ) -> Result<ExecutorPrefillMaintenanceOutcome> {
        let pending = self.sequences.lock().prefill_maintenance.remove(request_id);
        let Some(pending) = pending else {
            return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending);
        };
        let outcome = match pending {
            PendingPrefillMaintenance::Logical(deferred) => self
                .plan_resources
                .maintain_for_admission_deferred(&deferred),
            PendingPrefillMaintenance::Backing(deferred) => {
                self.plan_resources.maintain_for_deferred(&deferred)
            }
        }
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        match outcome {
            DynamicDeferredMaintenanceOutcome::RetryWithoutGrowth { current_epochs } => {
                Ok(ExecutorPrefillMaintenanceOutcome::RetryWithoutGrowth {
                    current: ExecutorAdmissionEpochs::from_capacity(current_epochs),
                })
            }
            DynamicDeferredMaintenanceOutcome::Maintained(receipt) => {
                let allocated_bytes = receipt
                    .growths()
                    .iter()
                    .try_fold(0_u64, |total, growth| {
                        total.checked_add(growth.chunk_bytes())
                    })
                    .ok_or_else(|| {
                        FerrumError::internal("vNext prefill maintenance byte count overflow")
                    })?;
                Ok(ExecutorPrefillMaintenanceOutcome::Maintained {
                    current: self.current_prefill_admission_epochs()?,
                    pools_grown: receipt.growths().len(),
                    allocated_bytes,
                })
            }
        }
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let started = Instant::now();
        if input.batch_size() != 1 {
            return Err(FerrumError::unsupported(
                "vNext prefill currently requires one sequence per typed submission wave",
            ));
        }
        let request_id = input.request_id.clone().ok_or_else(|| {
            FerrumError::request_validation(
                "executor-owned vNext prefill requires a typed request_id",
            )
        })?;
        let tokens = common::tensor_to_tokens(&input.input_ids)?;
        let maximum_tokens = input.maximum_sequence_tokens.ok_or_else(|| {
            FerrumError::request_validation(
                "executor-owned vNext prefill requires maximum_sequence_tokens",
            )
        })?;
        if maximum_tokens < tokens.len() || maximum_tokens > self.maximum_model_tokens {
            return Err(FerrumError::request_validation(format!(
                "request sequence ceiling {maximum_tokens} must cover prompt {} and not exceed {}",
                tokens.len(),
                self.maximum_model_tokens
            )));
        }
        let sequence = self
            .sequences
            .lock()
            .pending
            .get(&request_id)
            .cloned()
            .ok_or_else(|| {
                FerrumError::request_validation(format!(
                    "vNext prefill for `{request_id}` has no retained admission authority"
                ))
            })?;
        if sequence.maximum_tokens != maximum_tokens || *sequence.tokens.lock() != tokens {
            self.sequences
                .lock()
                .remove_pending_if(&request_id, &sequence);
            sequence.abort();
            return Err(FerrumError::request_validation(format!(
                "vNext prefill input for `{request_id}` differs from its admitted work"
            )));
        }
        let span = TokenSpanWork::from_token_ids_with_fit(&tokens, 0..tokens.len(), maximum_tokens)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let logits = match self.execute_step(&sequence, &tokens, span).await {
            Ok(logits) => logits,
            Err(error) => {
                self.sequences
                    .lock()
                    .remove_pending_if(&request_id, &sequence);
                sequence.abort();
                return Err(error);
            }
        };
        let logits = match self.prefill_tensor(logits) {
            Ok(logits) => logits,
            Err(error) => {
                self.sequences
                    .lock()
                    .remove_pending_if(&request_id, &sequence);
                sequence.abort();
                return Err(error);
            }
        };
        if let Err(error) = self.sequences.lock().activate(&request_id, &sequence) {
            self.sequences
                .lock()
                .remove_pending_if(&request_id, &sequence);
            sequence.abort();
            return Err(error);
        }
        self.metrics
            .prefill_operations
            .fetch_add(1, Ordering::Relaxed);
        self.metrics.total_prefill_us.fetch_add(
            started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        let cache = self.cache_handle(&sequence, tokens.len());
        Ok(PrefillOutput::new(logits, cache))
    }

    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        futures::future::try_join_all(inputs.iter().map(|input| self.prefill(input))).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let started = Instant::now();
        if input.batch_size() != 1 {
            return Err(FerrumError::unsupported(
                "vNext decode currently requires one sequence per typed submission wave",
            ));
        }
        let cache_id = input.kv_cache.cache_id();
        let sequence = self.sequence_for_cache(&cache_id)?;
        if input
            .request_id
            .as_ref()
            .is_some_and(|request_id| request_id != &sequence.request_id)
        {
            return Err(FerrumError::request_validation(
                "vNext decode request identity differs from its cache owner",
            ));
        }
        let _operation = sequence.operation.lock().await;
        if !sequence.active.load(Ordering::Acquire) {
            return Err(FerrumError::cancelled(format!(
                "vNext cache `{cache_id}` is no longer active"
            )));
        }
        let next = common::tensor_to_tokens(&input.input_ids)?;
        let [next_token] = next.as_slice() else {
            return Err(FerrumError::request_validation(
                "vNext decode requires exactly one input token",
            ));
        };
        let (tokens, previous_len) = {
            let current = sequence.tokens.lock();
            let previous_len = current.len();
            if previous_len >= sequence.maximum_tokens {
                return Err(FerrumError::request_validation(format!(
                    "vNext sequence reached its {} token ceiling",
                    sequence.maximum_tokens
                )));
            }
            let mut tokens = current.clone();
            tokens.push(*next_token);
            (tokens, previous_len)
        };
        let extension_span = TokenSpanWork::from_token_ids(&tokens, 0..tokens.len())
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let extension = ResourceWorkShape::single(extension_span)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        if let Err(error) = self.extend_sequence(&sequence, extension) {
            if DecodeFailureDisposition::from_error(&error)
                == DecodeFailureDisposition::AbortSequence
            {
                self.sequences.lock().active.remove(&cache_id);
                sequence.abort();
            }
            return Err(error);
        }
        let step_span = TokenSpanWork::from_token_ids(&tokens, previous_len..tokens.len())
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let logits = match self.execute_step(&sequence, &tokens, step_span).await {
            Ok(logits) => logits,
            Err(error) => {
                if DecodeFailureDisposition::from_error(&error)
                    == DecodeFailureDisposition::AbortSequence
                {
                    self.sequences.lock().active.remove(&cache_id);
                    sequence.abort();
                }
                return Err(error);
            }
        };
        *sequence.tokens.lock() = tokens;
        self.metrics
            .decode_operations
            .fetch_add(1, Ordering::Relaxed);
        self.metrics.total_decode_us.fetch_add(
            started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        let logits = self.decode_tensor(logits)?;
        let cache = self.cache_handle(&sequence, previous_len + 1);
        Ok(DecodeOutput::new(logits, cache))
    }

    async fn batch_decode(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>> {
        futures::future::try_join_all(inputs.iter().map(|input| self.decode(input))).await
    }

    fn release_cache(&self, cache_id: &str) {
        if let Some(sequence) = self.sequences.lock().active.remove(cache_id) {
            sequence.complete();
        }
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        let head_dim = self.info.hidden_size / self.info.num_heads.max(1);
        ExecutorCapabilities {
            max_batch_size: self.policy.memory().maximum_active_sequences as usize,
            max_sequence_length: self.maximum_model_tokens,
            attention_mechanisms: vec![AttentionType::GroupedQuery, AttentionType::Paged],
            supports_dynamic_batching: true,
            supports_continuous_batching: true,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![self.info.dtype],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: self.static_bytes,
                activation_memory_per_token: self.info.hidden_size * self.info.dtype.size_bytes(),
                kv_cache_memory_per_token: self.info.num_kv_heads
                    * head_dim
                    * 2
                    * self.info.dtype.size_bytes(),
                overhead_memory: self.policy.memory().reserve_bytes,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        let prefill_operations = self.metrics.prefill_operations.load(Ordering::Relaxed);
        let decode_operations = self.metrics.decode_operations.load(Ordering::Relaxed);
        let pool_status = self.plan_resources.dynamic_pool_status().ok();
        let allocated_bytes = reported_allocated_bytes(
            pool_status
                .as_ref()
                .map(|status| status.budget_claimed_bytes()),
            self.static_bytes,
        );
        let used_dynamic = pool_status
            .as_ref()
            .map(|status| {
                status.pools().iter().fold(0_u64, |total, pool| {
                    total.saturating_add(pool.resident_bytes().saturating_sub(pool.free_bytes()))
                })
            })
            .unwrap_or(0);
        let used_bytes = self.static_bytes.saturating_add(used_dynamic);
        let capacity = self.policy.memory().capacity_bytes;
        ExecutorStatus {
            state: if self.sequences.lock().total_len() == 0 {
                ExecutorState::Ready
            } else {
                ExecutorState::Busy
            },
            is_ready: !self.plan_resources.is_closing(),
            current_batch_size: self.sequences.lock().active.len(),
            prefill_operations,
            decode_operations,
            avg_prefill_time_ms: VNextExecutorMetrics::average_ms(
                self.metrics.total_prefill_us.load(Ordering::Relaxed),
                prefill_operations,
            ),
            avg_decode_time_ms: VNextExecutorMetrics::average_ms(
                self.metrics.total_decode_us.load(Ordering::Relaxed),
                decode_operations,
            ),
            memory_usage: ExecutorMemoryUsage {
                allocated_bytes: usize::try_from(allocated_bytes).unwrap_or(usize::MAX),
                used_bytes: usize::try_from(used_bytes).unwrap_or(usize::MAX),
                peak_bytes: usize::try_from(allocated_bytes).unwrap_or(usize::MAX),
                utilization_percent: if capacity == 0 {
                    0.0
                } else {
                    used_bytes as f32 / capacity as f32 * 100.0
                },
            },
            last_operation: Some(Instant::now()),
        }
    }

    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        Some(self.metrics_snapshot())
    }
}

#[cfg(test)]
mod tests {
    use super::{reported_allocated_bytes, DecodeFailureDisposition, FerrumError};

    #[test]
    fn allocated_memory_does_not_count_static_claim_twice() {
        assert_eq!(reported_allocated_bytes(Some(64), 64), 64);
        assert_eq!(reported_allocated_bytes(None, 64), 64);
    }

    #[test]
    fn decode_capacity_deferral_preserves_executor_owned_sequence() {
        let error = FerrumError::resource_exhausted("dynamic pool is waiting for release");

        assert_eq!(
            DecodeFailureDisposition::from_error(&error),
            DecodeFailureDisposition::PreserveForCapacityRetry
        );
    }

    #[test]
    fn decode_permanent_failure_aborts_executor_owned_sequence() {
        let error = FerrumError::request_validation("sequence exceeds its configured ceiling");

        assert_eq!(
            DecodeFailureDisposition::from_error(&error),
            DecodeFailureDisposition::AbortSequence
        );
    }
}

#[cfg(feature = "cuda")]
impl VNextModelExecutor<ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime> {
    pub fn create_cuda(
        ordinal: usize,
        prepared: PreparedProductionModel,
        info: ModelInfo,
        engine_config: &EngineConfig,
    ) -> Result<Self> {
        use ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition;

        let device_id = DeviceId::new(format!("device.cuda.{ordinal}"))
            .map_err(|error| FerrumError::device(error.to_string()))?;
        let composition = CudaVNextComposition::create(ordinal, device_id)
            .map_err(|error| FerrumError::device(format!("create vNext CUDA runtime: {error}")))?;
        let (runtime, registry, catalog) = composition.into_parts();
        Self::from_runtime_composition(prepared, info, engine_config, runtime, registry, catalog)
    }
}
