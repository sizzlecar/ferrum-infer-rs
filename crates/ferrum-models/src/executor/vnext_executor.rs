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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use ferrum_interfaces::kv_cache::{BlockTable, CacheHandleStats};
use ferrum_interfaces::model_executor::{
    AttentionType, DecodeInput, DecodeOutput, ExecutionResourceAuthority, ExecutorAdmissionEpochs,
    ExecutorBatchDecodeOutcome, ExecutorCapabilities, ExecutorCapacityWaitRegistration,
    ExecutorExecutionCapacityDeferral, ExecutorExecutionCapacityPreemption,
    ExecutorExecutionCapacityPreemptionAuthority, ExecutorExecutionCapacityPreemptionReceipt,
    ExecutorExecutionCapacityStage, ExecutorMemoryUsage, ExecutorPrefillAdmission,
    ExecutorPrefillAdmissionDecision, ExecutorPrefillAdmissionReceipt, ExecutorPrefillCompletion,
    ExecutorPrefillMaintenanceDeferral, ExecutorPrefillMaintenanceOutcome, ExecutorPrefillOutcome,
    ExecutorSequenceCompletion, ExecutorState, ExecutorStatus, MemoryRequirements,
    PlanRuntimeResourceSnapshot, PrefillChunk, PrefillInput, PrefillOutput,
};
use ferrum_interfaces::vnext::*;
use ferrum_interfaces::{KvCacheHandle, ModelExecutor, TensorRef};
use ferrum_types::{
    Device, EngineConfig, FerrumError, ModelInfo, RequestId, Result, SchedulingPolicy,
    SequenceFitPolicy, TokenId,
};
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use tokio::sync::Mutex as AsyncMutex;

use crate::vnext::PreparedProductionModel;

use super::{
    common,
    vnext_completion_worker::{VNextCompletionTaskKind, VNextCompletionWorker},
    vnext_timing::AtomicDurationMetrics,
};

const POLICY_ID: &str = "policy.ferrum.product.vnext.default";
const POLICY_VERSION: ContractVersion = ContractVersion::new(2, 0);
const UNIFORM_QUERY_REUSABLE_CLASS: &str = "execution.uniform-query-token";
const PACKED_TOKEN_REUSABLE_CLASS: &str = "execution.single-sequence-packed-token";
const DEFAULT_STATIC_STAGING_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_STATIC_COMMANDS_PER_BATCH: usize = 64;
const DEFAULT_CANCELLATION_CHECK_INTERVAL_STEPS: u32 = 1;
const MAX_DEFINITELY_NOT_SUBMITTED_RETRIES: u32 = 1;
const MAX_BACKING_MAINTENANCE_ATTEMPTS: u32 = 2;
const MAX_EXTENSION_RECHECKS: u32 = 2;

type VNextDriver<R> = RuntimeResourceDriver<R>;

const fn resolved_sequence_fit_policy(policy: SequenceFitPolicy) -> AdmissionFitPolicy {
    match policy {
        SequenceFitPolicy::FullInputMustFit => AdmissionFitPolicy::FullInputMustFit,
        SequenceFitPolicy::ImmediateOnly => AdmissionFitPolicy::ImmediateOnly,
    }
}

fn resolve_reusable_execution_policy(
    maximum_active_sequences: u32,
    maximum_scheduled_tokens: u64,
    maximum_model_tokens: usize,
    prefill_token_counts: &[usize],
) -> Result<ReusableExecutionPolicy> {
    let maximum_active_sequences = usize::try_from(maximum_active_sequences)
        .map_err(|_| FerrumError::config("vNext active sequence limit exceeds usize"))?;
    let maximum_scheduled_tokens = usize::try_from(maximum_scheduled_tokens)
        .map_err(|_| FerrumError::config("vNext scheduled token limit exceeds usize"))?;
    let maximum_width = maximum_active_sequences.min(maximum_scheduled_tokens);
    if maximum_width == 0 {
        return Err(FerrumError::config(
            "vNext reusable execution requires a non-zero decode width",
        ));
    }

    let uniform_class = ReusableExecutionClassId::new(UNIFORM_QUERY_REUSABLE_CLASS)
        .map_err(|error| FerrumError::config(error.to_string()))?;
    let packed_class = ReusableExecutionClassId::new(PACKED_TOKEN_REUSABLE_CLASS)
        .map_err(|error| FerrumError::config(error.to_string()))?;
    let mut buckets = Vec::new();
    let mut width = 1_usize;
    loop {
        let width_u32 = u32::try_from(width)
            .map_err(|_| FerrumError::config("vNext decode width exceeds u32"))?;
        let width_u64 = u64::try_from(width)
            .map_err(|_| FerrumError::config("vNext decode width exceeds u64"))?;
        buckets.push(
            ReusableExecutionBucketSpec::new(
                uniform_class.clone(),
                ReusableExecutionCapacity::new(width_u32, width_u64, 1)
                    .map_err(|error| FerrumError::config(error.to_string()))?,
            )
            .map_err(|error| FerrumError::config(error.to_string()))?,
        );
        if width == maximum_width {
            break;
        }
        width = width.saturating_mul(2).min(maximum_width);
    }

    let mut prefill_token_counts = prefill_token_counts
        .iter()
        .copied()
        .filter(|token_count| *token_count > 0)
        .map(|token_count| {
            token_count
                .min(maximum_scheduled_tokens)
                .min(maximum_model_tokens)
        })
        .filter(|token_count| *token_count > 0)
        .collect::<Vec<_>>();
    prefill_token_counts.sort_unstable();
    prefill_token_counts.dedup();
    for token_count in prefill_token_counts {
        buckets.push(
            ReusableExecutionBucketSpec::new(
                packed_class.clone(),
                ReusableExecutionCapacity::new(
                    1,
                    u64::try_from(token_count).map_err(|_| {
                        FerrumError::config("vNext prefill token capacity exceeds u64")
                    })?,
                    1,
                )
                .map_err(|error| FerrumError::config(error.to_string()))?,
            )
            .map_err(|error| FerrumError::config(error.to_string()))?,
        );
    }
    ReusableExecutionPolicy::new(1, buckets).map_err(|error| FerrumError::config(error.to_string()))
}

/// Typed product policy resolved before plan compilation. None of these
/// values are inferred from a model name, GPU name, or hidden environment
/// combination.
#[derive(Debug, Clone)]
pub struct VNextExecutorConfig {
    pub maximum_model_tokens: usize,
    pub static_initialization: StaticInitializationPolicy,
    pub runtime_policy: ResolvedRuntimePolicy,
    pub reusable_execution_enabled: bool,
    pub reusable_execution_prefill_token_counts: Vec<usize>,
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

        let mut reusable_execution_prefill_token_counts = [
            engine.scheduler.prefill_step_chunk,
            engine.scheduler.active_decode_prefill_chunk,
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        reusable_execution_prefill_token_counts.sort_unstable_by(|left, right| right.cmp(left));
        reusable_execution_prefill_token_counts.dedup();
        let reusable_execution_supported = descriptor
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID);
        let reusable_execution_policy =
            if engine.backend.enable_cuda_graphs && reusable_execution_supported {
                Some(resolve_reusable_execution_policy(
                    maximum_active_sequences,
                    maximum_scheduled_tokens,
                    maximum_model_tokens,
                    &reusable_execution_prefill_token_counts,
                )?)
            } else {
                None
            };

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
                sequence_fit_policy: resolved_sequence_fit_policy(
                    engine.scheduler.sequence_fit_policy,
                ),
                allow_defer: true,
                cancellation_check_interval_steps: DEFAULT_CANCELLATION_CHECK_INTERVAL_STEPS,
            },
            reusable_execution_policy,
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
            reusable_execution_enabled: engine.backend.enable_cuda_graphs,
            reusable_execution_prefill_token_counts,
        })
    }
}

const REUSABLE_EXECUTION_WARMUP_PASSES: usize = 1;
const REUSABLE_EXECUTION_CAPTURE_PASSES: usize = 1;
const REUSABLE_EXECUTION_REPLAY_VALIDATION_PASSES: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "topology", rename_all = "snake_case")]
enum VNextReusableExecutionDescriptor {
    UniformDecode {
        query_tokens_per_sequence: usize,
        token_capacity: usize,
        request_capacity: usize,
    },
    Prefill {
        token_capacity: usize,
        request_capacity: usize,
    },
}

impl VNextReusableExecutionDescriptor {
    const fn uniform_decode(width: usize) -> Self {
        Self::UniformDecode {
            query_tokens_per_sequence: 1,
            token_capacity: width,
            request_capacity: width,
        }
    }

    const fn prefill(token_count: usize) -> Self {
        Self::Prefill {
            token_capacity: token_count,
            request_capacity: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct VNextReusableExecutionStartupPlan {
    descriptors: Vec<VNextReusableExecutionDescriptor>,
    maximum_decode_sequence_tokens: usize,
    device_plan: DeviceReusableExecutionPlan,
}

impl VNextReusableExecutionStartupPlan {
    fn resolve(
        maximum_active_sequences: u32,
        maximum_scheduled_tokens: u64,
        maximum_model_tokens: usize,
        prefill_token_counts: &[usize],
        execution_node_count: usize,
    ) -> Result<Self> {
        let maximum_active_sequences = usize::try_from(maximum_active_sequences)
            .map_err(|_| FerrumError::config("vNext active sequence limit exceeds usize"))?;
        let maximum_scheduled_tokens = usize::try_from(maximum_scheduled_tokens)
            .map_err(|_| FerrumError::config("vNext scheduled token limit exceeds usize"))?;
        let maximum_width = maximum_active_sequences.min(maximum_scheduled_tokens);
        if maximum_width == 0 {
            return Err(FerrumError::config(
                "vNext reusable execution requires a non-zero decode width",
            ));
        }

        let mut decode_widths = Vec::new();
        let mut width = 1_usize;
        while width < maximum_width {
            decode_widths.push(width);
            width = width.checked_mul(2).unwrap_or(maximum_width);
        }
        if decode_widths.last().copied() != Some(maximum_width) {
            decode_widths.push(maximum_width);
        }
        decode_widths.sort_unstable_by(|left, right| right.cmp(left));

        let mut prefill_token_counts = prefill_token_counts
            .iter()
            .copied()
            .filter(|token_count| *token_count > 0)
            .map(|token_count| {
                token_count
                    .min(maximum_scheduled_tokens)
                    .min(maximum_model_tokens)
            })
            .filter(|token_count| *token_count > 0)
            .collect::<Vec<_>>();
        prefill_token_counts.sort_unstable_by(|left, right| right.cmp(left));
        prefill_token_counts.dedup();

        let passes_per_width = REUSABLE_EXECUTION_WARMUP_PASSES
            + REUSABLE_EXECUTION_CAPTURE_PASSES
            + REUSABLE_EXECUTION_REPLAY_VALIDATION_PASSES;
        let maximum_decode_sequence_tokens = decode_widths
            .len()
            .checked_mul(passes_per_width)
            .and_then(|decode_tokens| decode_tokens.checked_add(1))
            .ok_or_else(|| FerrumError::config("vNext startup token ceiling overflowed"))?;
        if maximum_decode_sequence_tokens > maximum_model_tokens {
            return Err(FerrumError::config(format!(
                "vNext model length {maximum_model_tokens} cannot cover reusable execution startup ceiling {maximum_decode_sequence_tokens}"
            )));
        }

        let descriptors = decode_widths
            .into_iter()
            .map(VNextReusableExecutionDescriptor::uniform_decode)
            .chain(
                prefill_token_counts
                    .into_iter()
                    .map(VNextReusableExecutionDescriptor::prefill),
            )
            .collect::<Vec<_>>();

        let maximum_executables = execution_node_count
            .max(1)
            .checked_mul(descriptors.len())
            .ok_or_else(|| FerrumError::config("vNext reusable executable capacity overflowed"))?;
        let device_plan = DeviceReusableExecutionPlan::new(maximum_executables)
            .map_err(|error| FerrumError::config(error.to_string()))?;
        Ok(Self {
            descriptors,
            maximum_decode_sequence_tokens,
            device_plan,
        })
    }

    fn decode_widths(&self) -> Vec<usize> {
        self.descriptors
            .iter()
            .filter_map(|descriptor| match descriptor {
                VNextReusableExecutionDescriptor::UniformDecode {
                    request_capacity, ..
                } => Some(*request_capacity),
                VNextReusableExecutionDescriptor::Prefill { .. } => None,
            })
            .collect()
    }

    fn prefill_token_counts(&self) -> Vec<usize> {
        self.descriptors
            .iter()
            .filter_map(|descriptor| match descriptor {
                VNextReusableExecutionDescriptor::UniformDecode { .. } => None,
                VNextReusableExecutionDescriptor::Prefill { token_capacity, .. } => {
                    Some(*token_capacity)
                }
            })
            .collect()
    }

    fn widths_for_available_sequences(&self, available: usize) -> Vec<usize> {
        let mut widths = self
            .decode_widths()
            .into_iter()
            .filter(|width| *width <= available)
            .collect::<Vec<_>>();
        if available > 0 && !widths.contains(&available) {
            widths.insert(0, available);
        }
        widths
    }
}

#[derive(Debug, Clone, Serialize)]
struct VNextReusableExecutionStartupReport {
    enabled: bool,
    supported: bool,
    requested_descriptors: Vec<VNextReusableExecutionDescriptor>,
    prepared_descriptors: Vec<VNextReusableExecutionDescriptor>,
    requested_decode_widths: Vec<usize>,
    prepared_decode_widths: Vec<usize>,
    requested_prefill_token_counts: Vec<usize>,
    prepared_prefill_token_counts: Vec<usize>,
    synthetic_sequences: usize,
    eager_warmup_waves: usize,
    capture_waves: usize,
    replay_validation_waves: usize,
    device_preparation: DeviceReusableExecutionPreparation,
    elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum VNextStartupPreparationState {
    Pending,
    Preparing,
    Ready {
        report: VNextReusableExecutionStartupReport,
    },
    Failed {
        message: String,
    },
}

impl VNextStartupPreparationState {
    const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
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
    prefill_frontier_narrowings: AtomicU64,
    decode_operations: AtomicU64,
    prepared_wave_topology: VNextPreparedWaveTopologyMetrics,
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
    wave_timing: VNextWaveTimingMetrics,
    prefill_wave_timing: VNextWaveTimingMetrics,
    decode_wave_timing: VNextWaveTimingMetrics,
    device_timing: VNextDeviceTimingMetrics,
    prefill_device_timing: VNextDeviceTimingMetrics,
    decode_device_timing: VNextDeviceTimingMetrics,
    last_failure: Mutex<Option<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VNextExecutionWaveKind {
    Prefill,
    Decode,
}

impl VNextExecutionWaveKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }

    const fn reusable_execution_class(self) -> &'static str {
        match self {
            Self::Prefill => PACKED_TOKEN_REUSABLE_CLASS,
            Self::Decode => UNIFORM_QUERY_REUSABLE_CLASS,
        }
    }
}

#[derive(Default)]
struct VNextWaveTimingMetrics {
    resource_prepare_attempt: AtomicDurationMetrics,
    host_encode_submit: AtomicDurationMetrics,
    token_upload_prepare: AtomicDurationMetrics,
    wave_identity_bind: AtomicDurationMetrics,
    provider_encode_submit: AtomicDurationMetrics,
    contract_validate_reserve: AtomicDurationMetrics,
    backing_input_encode: AtomicDurationMetrics,
    provider_node_encode: AtomicDurationMetrics,
    lane_reserve_submit_arm: AtomicDurationMetrics,
    lane_reserve: AtomicDurationMetrics,
    device_runtime_submit: AtomicDurationMetrics,
    device_submit_validate_prepare: AtomicDurationMetrics,
    device_submit_begin_timing: AtomicDurationMetrics,
    device_submit_enqueue_commands: AtomicDurationMetrics,
    device_submit_record_fence_account: AtomicDurationMetrics,
    reusable_execution: VNextReusableExecutionMetrics,
    completion_arm: AtomicDurationMetrics,
    completion_round_trip: AtomicDurationMetrics,
    host_postprocess: AtomicDurationMetrics,
    submitted_wave_total: AtomicDurationMetrics,
}

#[derive(Default)]
struct VNextPreparedWaveTopologyMetrics {
    wave_authorities: AtomicU64,
    covered_nodes: AtomicU64,
    participant_flights: AtomicU64,
    node_participant_projections: AtomicU64,
    physical_ledger_entries: AtomicU64,
}

impl VNextPreparedWaveTopologyMetrics {
    fn record<R: DeviceRuntime>(&self, wave: &PreparedStepSubmissionWave<R>) {
        self.record_counts(
            wave.node_count(),
            wave.prepared_participant_flight_count(),
            wave.node_participant_projection_count(),
            wave.physical_invocation_ledger_entry_count(),
        );
    }

    fn record_counts(
        &self,
        covered_nodes: usize,
        participant_flights: usize,
        node_participant_projections: usize,
        physical_ledger_entries: usize,
    ) {
        self.wave_authorities.fetch_add(1, Ordering::Relaxed);
        self.covered_nodes.fetch_add(
            u64::try_from(covered_nodes).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.participant_flights.fetch_add(
            u64::try_from(participant_flights).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.node_participant_projections.fetch_add(
            u64::try_from(node_participant_projections).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.physical_ledger_entries.fetch_add(
            u64::try_from(physical_ledger_entries).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "wave_authorities": self.wave_authorities.load(Ordering::Relaxed),
            "covered_nodes": self.covered_nodes.load(Ordering::Relaxed),
            "participant_flights": self.participant_flights.load(Ordering::Relaxed),
            "node_participant_projections": self.node_participant_projections.load(Ordering::Relaxed),
            "physical_ledger_entries": self.physical_ledger_entries.load(Ordering::Relaxed),
        })
    }

    fn reset(&self) {
        for counter in [
            &self.wave_authorities,
            &self.covered_nodes,
            &self.participant_flights,
            &self.node_participant_projections,
            &self.physical_ledger_entries,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

impl VNextWaveTimingMetrics {
    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "clock": "host_monotonic",
            "scope": "executor_host_wall_boundaries",
            "resource_prepare_attempt": self.resource_prepare_attempt.snapshot(),
            "host_encode_submit": self.host_encode_submit.snapshot(),
            "host_encode_submit_breakdown": {
                "collection": "profile_attached_only",
                "token_upload_prepare": self.token_upload_prepare.snapshot(),
                "wave_identity_bind": self.wave_identity_bind.snapshot(),
                "provider_encode_submit": self.provider_encode_submit.snapshot(),
                "provider_encode_submit_breakdown": {
                    "contract_validate_reserve": self.contract_validate_reserve.snapshot(),
                    "backing_input_encode": self.backing_input_encode.snapshot(),
                    "provider_node_encode": self.provider_node_encode.snapshot(),
                    "lane_reserve_submit_arm": self.lane_reserve_submit_arm.snapshot(),
                    "lane_reserve_submit_arm_breakdown": {
                        "lane_reserve": self.lane_reserve.snapshot(),
                        "device_runtime_submit": self.device_runtime_submit.snapshot(),
                        "device_runtime_submit_breakdown": {
                            "validate_and_prepare": self.device_submit_validate_prepare.snapshot(),
                            "begin_timing": self.device_submit_begin_timing.snapshot(),
                            "enqueue_commands": self.device_submit_enqueue_commands.snapshot(),
                            "record_fence_and_account": self.device_submit_record_fence_account.snapshot(),
                            "reusable_execution": self.reusable_execution.snapshot(),
                        },
                        "completion_arm": self.completion_arm.snapshot(),
                    },
                },
            },
            "completion_round_trip": self.completion_round_trip.snapshot(),
            "host_postprocess": self.host_postprocess.snapshot(),
            "submitted_wave_total": self.submitted_wave_total.snapshot(),
            "limitations": [
                "resource_prepare_attempt includes capacity-deferred attempts and is outside submitted_wave_total",
                "host_encode_submit breakdown is collected only while a typed profile sink is attached",
                "provider_encode_submit breakdown covers contract validation and completion reservation, backing/input encoding, provider node encoding, and lane reserve/submit/arm",
                "lane_reserve_submit_arm breakdown isolates lane acquisition, DeviceRuntime::submit, and successful completion arming; failed submissions do not emit completion_arm",
                "device_runtime_submit breakdown isolates backend validation/preparation, timing start, ordered command enqueue, and fence/accounting for runtimes that implement typed attribution",
                "completion_round_trip includes async queue wait, device fence wait, and readback",
                "these host intervals are not kernel or device-busy time"
            ],
        })
    }

    fn reset(&self) {
        for metrics in [
            &self.resource_prepare_attempt,
            &self.host_encode_submit,
            &self.token_upload_prepare,
            &self.wave_identity_bind,
            &self.provider_encode_submit,
            &self.contract_validate_reserve,
            &self.backing_input_encode,
            &self.provider_node_encode,
            &self.lane_reserve_submit_arm,
            &self.lane_reserve,
            &self.device_runtime_submit,
            &self.device_submit_validate_prepare,
            &self.device_submit_begin_timing,
            &self.device_submit_enqueue_commands,
            &self.device_submit_record_fence_account,
            &self.completion_arm,
            &self.completion_round_trip,
            &self.host_postprocess,
            &self.submitted_wave_total,
        ] {
            metrics.reset();
        }
        self.reusable_execution.reset();
    }
}

#[derive(Default)]
struct VNextReusableExecutionMetrics {
    candidate_segments: AtomicU64,
    captured_segments: AtomicU64,
    uploaded_segments: AtomicU64,
    cache_hit_segments: AtomicU64,
    cached_rejected_segments: AtomicU64,
    capture_rejected_segments: AtomicU64,
    quiescence_deferred_segments: AtomicU64,
    capacity_deferred_segments: AtomicU64,
    outside_preparation_segments: AtomicU64,
    evicted_segments: AtomicU64,
    replayed_segments: AtomicU64,
    replayed_commands: AtomicU64,
    eager_commands: AtomicU64,
}

impl VNextReusableExecutionMetrics {
    fn record(&self, observation: DeviceReusableExecutionObservation) {
        self.candidate_segments
            .fetch_add(observation.candidate_segments(), Ordering::Relaxed);
        self.captured_segments
            .fetch_add(observation.captured_segments(), Ordering::Relaxed);
        self.uploaded_segments
            .fetch_add(observation.uploaded_segments(), Ordering::Relaxed);
        self.cache_hit_segments
            .fetch_add(observation.cache_hit_segments(), Ordering::Relaxed);
        self.cached_rejected_segments
            .fetch_add(observation.cached_rejected_segments(), Ordering::Relaxed);
        self.capture_rejected_segments
            .fetch_add(observation.capture_rejected_segments(), Ordering::Relaxed);
        self.quiescence_deferred_segments.fetch_add(
            observation.quiescence_deferred_segments(),
            Ordering::Relaxed,
        );
        self.capacity_deferred_segments
            .fetch_add(observation.capacity_deferred_segments(), Ordering::Relaxed);
        self.outside_preparation_segments.fetch_add(
            observation.outside_preparation_segments(),
            Ordering::Relaxed,
        );
        self.evicted_segments
            .fetch_add(observation.evicted_segments(), Ordering::Relaxed);
        self.replayed_segments
            .fetch_add(observation.replayed_segments(), Ordering::Relaxed);
        self.replayed_commands
            .fetch_add(observation.replayed_commands(), Ordering::Relaxed);
        self.eager_commands
            .fetch_add(observation.eager_commands(), Ordering::Relaxed);
    }

    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "candidate_segments": self.candidate_segments.load(Ordering::Relaxed),
            "captured_segments": self.captured_segments.load(Ordering::Relaxed),
            "uploaded_segments": self.uploaded_segments.load(Ordering::Relaxed),
            "cache_hit_segments": self.cache_hit_segments.load(Ordering::Relaxed),
            "cached_rejected_segments": self.cached_rejected_segments.load(Ordering::Relaxed),
            "capture_rejected_segments": self.capture_rejected_segments.load(Ordering::Relaxed),
            "quiescence_deferred_segments": self.quiescence_deferred_segments.load(Ordering::Relaxed),
            "capacity_deferred_segments": self.capacity_deferred_segments.load(Ordering::Relaxed),
            "outside_preparation_segments": self.outside_preparation_segments.load(Ordering::Relaxed),
            "evicted_segments": self.evicted_segments.load(Ordering::Relaxed),
            "replayed_segments": self.replayed_segments.load(Ordering::Relaxed),
            "replayed_commands": self.replayed_commands.load(Ordering::Relaxed),
            "eager_commands": self.eager_commands.load(Ordering::Relaxed),
        })
    }

    fn reset(&self) {
        for counter in [
            &self.candidate_segments,
            &self.captured_segments,
            &self.uploaded_segments,
            &self.cache_hit_segments,
            &self.cached_rejected_segments,
            &self.capture_rejected_segments,
            &self.quiescence_deferred_segments,
            &self.capacity_deferred_segments,
            &self.outside_preparation_segments,
            &self.evicted_segments,
            &self.replayed_segments,
            &self.replayed_commands,
            &self.eager_commands,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

impl DeviceSubmissionTimingSink for VNextWaveTimingMetrics {
    const ENABLED: bool = true;

    fn record_device_submission(&self, stage: DeviceSubmissionStage, elapsed: Duration) {
        match stage {
            DeviceSubmissionStage::ValidateAndPrepare => {
                self.device_submit_validate_prepare.record(elapsed)
            }
            DeviceSubmissionStage::BeginTiming => self.device_submit_begin_timing.record(elapsed),
            DeviceSubmissionStage::EnqueueCommands => {
                self.device_submit_enqueue_commands.record(elapsed)
            }
            DeviceSubmissionStage::RecordFenceAndAccount => {
                self.device_submit_record_fence_account.record(elapsed)
            }
        }
    }

    fn record_reusable_execution(&self, observation: DeviceReusableExecutionObservation) {
        self.reusable_execution.record(observation);
    }
}

impl SubmissionWaveDispatchTimingSink for VNextWaveTimingMetrics {
    fn record(&self, stage: SubmissionWaveDispatchStage, elapsed: Duration) {
        match stage {
            SubmissionWaveDispatchStage::ContractValidateAndReserve => {
                self.contract_validate_reserve.record(elapsed)
            }
            SubmissionWaveDispatchStage::BackingAndInputEncode => {
                self.backing_input_encode.record(elapsed)
            }
            SubmissionWaveDispatchStage::ProviderNodeEncode => {
                self.provider_node_encode.record(elapsed)
            }
            SubmissionWaveDispatchStage::LaneReserve => self.lane_reserve.record(elapsed),
            SubmissionWaveDispatchStage::DeviceRuntimeSubmit => {
                self.device_runtime_submit.record(elapsed)
            }
            SubmissionWaveDispatchStage::CompletionArm => self.completion_arm.record(elapsed),
            SubmissionWaveDispatchStage::LaneReserveSubmitAndArm => {
                self.lane_reserve_submit_arm.record(elapsed)
            }
        }
    }
}

struct VNextWaveTimingSink<'metrics> {
    aggregate: &'metrics VNextWaveTimingMetrics,
    phase: &'metrics VNextWaveTimingMetrics,
}

impl DeviceSubmissionTimingSink for VNextWaveTimingSink<'_> {
    const ENABLED: bool = true;

    fn record_device_submission(&self, stage: DeviceSubmissionStage, elapsed: Duration) {
        self.aggregate.record_device_submission(stage, elapsed);
        self.phase.record_device_submission(stage, elapsed);
    }

    fn record_reusable_execution(&self, observation: DeviceReusableExecutionObservation) {
        self.aggregate.record_reusable_execution(observation);
        self.phase.record_reusable_execution(observation);
    }
}

impl SubmissionWaveDispatchTimingSink for VNextWaveTimingSink<'_> {
    fn record(&self, stage: SubmissionWaveDispatchStage, elapsed: Duration) {
        SubmissionWaveDispatchTimingSink::record(self.aggregate, stage, elapsed);
        SubmissionWaveDispatchTimingSink::record(self.phase, stage, elapsed);
    }
}

#[derive(Default)]
struct VNextDeviceTimingMetrics {
    device_execution: AtomicDurationMetrics,
    fence_wait_host: AtomicDurationMetrics,
    readback_host: AtomicDurationMetrics,
    readback_calls: AtomicU64,
    readback_bytes: AtomicU64,
    device_unavailable: AtomicU64,
    fence_wait_unavailable: AtomicU64,
    readback_unavailable: AtomicU64,
}

impl VNextDeviceTimingMetrics {
    fn record(&self, receipt: &CompletionReadbackBatchReceipt) {
        let fence = receipt.completion().fence_timing();
        match fence.device_execution() {
            DeviceTimingMeasurement::Measured(timing) => self
                .device_execution
                .record(Duration::from_nanos(timing.elapsed_ns())),
            DeviceTimingMeasurement::Unavailable(_) => {
                self.device_unavailable.fetch_add(1, Ordering::Relaxed);
            }
            DeviceTimingMeasurement::NotRequested => {}
        }
        match fence.blocking_wait_host_ns() {
            DeviceTimingMeasurement::Measured(nanoseconds) => self
                .fence_wait_host
                .record(Duration::from_nanos(nanoseconds)),
            DeviceTimingMeasurement::Unavailable(_) => {
                self.fence_wait_unavailable.fetch_add(1, Ordering::Relaxed);
            }
            DeviceTimingMeasurement::NotRequested => {}
        }
        if let Some(readbacks) = receipt.readback_timings() {
            for readback in readbacks {
                match readback {
                    DeviceTimingMeasurement::Measured(timing) => {
                        self.readback_host
                            .record(Duration::from_nanos(timing.host_elapsed_ns()));
                        self.readback_calls
                            .fetch_add(u64::from(timing.calls()), Ordering::Relaxed);
                        self.readback_bytes
                            .fetch_add(timing.bytes(), Ordering::Relaxed);
                    }
                    DeviceTimingMeasurement::Unavailable(_) => {
                        self.readback_unavailable.fetch_add(1, Ordering::Relaxed);
                    }
                    DeviceTimingMeasurement::NotRequested => {}
                }
            }
        }
    }

    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "scope": "exact_submission_completion",
            "device_execution": self.device_execution.snapshot(),
            "fence_wait_host": self.fence_wait_host.snapshot(),
            "readback_host": self.readback_host.snapshot(),
            "readback_calls": self.readback_calls.load(Ordering::Relaxed),
            "readback_bytes": self.readback_bytes.load(Ordering::Relaxed),
            "unavailable": {
                "device_execution": self.device_unavailable.load(Ordering::Relaxed),
                "fence_wait_host": self.fence_wait_unavailable.load(Ordering::Relaxed),
                "readback_host": self.readback_unavailable.load(Ordering::Relaxed),
            },
            "clocks": {
                "device_execution": "backend_device_event_elapsed",
                "fence_wait_host": "host_monotonic",
                "readback_host": "host_monotonic",
            },
            "limitations": [
                "device execution and fence host wait may overlap and must not be added",
                "readback host time includes backend synchronization, host allocation, and transfer",
                "device event elapsed has no cross-clock anchor and is diagnostic-only"
            ],
        })
    }

    fn reset(&self) {
        self.device_execution.reset();
        self.fence_wait_host.reset();
        self.readback_host.reset();
        for counter in [
            &self.readback_calls,
            &self.readback_bytes,
            &self.device_unavailable,
            &self.fence_wait_unavailable,
            &self.readback_unavailable,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }
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
    fn wave_timing_for(&self, kind: VNextExecutionWaveKind) -> &VNextWaveTimingMetrics {
        match kind {
            VNextExecutionWaveKind::Prefill => &self.prefill_wave_timing,
            VNextExecutionWaveKind::Decode => &self.decode_wave_timing,
        }
    }

    fn device_timing_for(&self, kind: VNextExecutionWaveKind) -> &VNextDeviceTimingMetrics {
        match kind {
            VNextExecutionWaveKind::Prefill => &self.prefill_device_timing,
            VNextExecutionWaveKind::Decode => &self.decode_device_timing,
        }
    }

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

    fn reset_after_startup(&self) {
        for counter in [
            &self.prefill_operations,
            &self.prefill_frontier_narrowings,
            &self.decode_operations,
            &self.submitted_waves,
            &self.completed_waves,
            &self.failed_waves,
            &self.definitely_not_submitted_retries,
            &self.request_deferrals,
            &self.sequence_deferrals,
            &self.extension_deferrals,
            &self.step_deferrals,
            &self.wave_deferrals,
            &self.backing_deferrals,
            &self.uploaded_bytes,
            &self.readback_bytes,
            &self.total_prefill_us,
            &self.total_decode_us,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        self.wave_timing.reset();
        self.prepared_wave_topology.reset();
        self.prefill_wave_timing.reset();
        self.decode_wave_timing.reset();
        self.device_timing.reset();
        self.prefill_device_timing.reset();
        self.decode_device_timing.reset();
        *self.last_failure.lock() = None;
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
    active: Arc<TrustedActiveSequenceBinding>,
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
        active: Arc<TrustedActiveSequenceBinding>,
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
        let plan_span = SpanId::new(format!("{}/plan", journal.root_span)).map_err(Self::error)?;
        let planned_parts =
            journal.bind_plan(journal.base_parts(2, plan_span, Some(journal.root_span.clone())));
        let planned = journal.event(
            ExecutionPhase::Planning,
            ExecutionEventKind::PlanBuilt,
            planned_parts,
            ExecutionEventDetail::None,
        )?;
        let events = [accepted, planned];
        let contexts = [
            TrustedExecutionEventContext::pre_plan(
                journal.active.run_id(),
                journal.active.request_id(),
            ),
            TrustedExecutionEventContext::bound(
                journal.active.run_id(),
                journal.active.request_id(),
                &journal.topology,
            ),
        ];
        journal.emitter.emit_batch(events.into(), &contexts)?;
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
        let node_started = self.node_event(first_identity, ExecutionEventKind::NodeStarted)?;
        let operation_submitted = self.operation_event(first_identity)?;
        let events = [frame_started, node_started, operation_submitted];
        let contexts = [
            TrustedExecutionEventContext::active(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
            ),
            TrustedExecutionEventContext::active(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
            ),
            TrustedExecutionEventContext::operation_submitted(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
                submission,
            ),
        ];
        self.emitter.emit_batch(events.into(), &contexts)?;
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
        enum CompletionEventEvidence {
            Active,
            Submitted,
            Retired(usize),
        }
        let mut events = Vec::with_capacity(selected.len().saturating_mul(3));
        let mut evidence = Vec::with_capacity(selected.len().saturating_mul(3));
        for (position, participant_index) in selected.iter().copied().enumerate() {
            let participant = completion
                .participants()
                .get(participant_index)
                .ok_or_else(|| Self::error("completion participant index is missing"))?;
            let identity = participant.submission().identity();
            let retired = self.node_event(identity, ExecutionEventKind::NodeRetired)?;
            events.push(retired);
            evidence.push(CompletionEventEvidence::Retired(participant_index));
            if let Some(next_index) = selected.get(position + 1).copied() {
                let next_identity = submission.participants()[next_index].identity();
                let started = self.node_event(next_identity, ExecutionEventKind::NodeStarted)?;
                events.push(started);
                evidence.push(CompletionEventEvidence::Active);
                let submitted = self.operation_event(next_identity)?;
                events.push(submitted);
                evidence.push(CompletionEventEvidence::Submitted);
            }
        }
        let last_index = *selected
            .last()
            .ok_or_else(|| Self::error("completion participant set is empty"))?;
        let frame_completed = self.frame_event(
            submission.participants()[last_index].identity(),
            ExecutionEventKind::FrameCompleted,
        )?;
        events.push(frame_completed);
        evidence.push(CompletionEventEvidence::Active);
        let contexts = evidence
            .iter()
            .map(|evidence| match evidence {
                CompletionEventEvidence::Active => TrustedExecutionEventContext::active(
                    self.active.run_id(),
                    self.active.request_id(),
                    &self.topology,
                    &self.active,
                ),
                CompletionEventEvidence::Submitted => {
                    TrustedExecutionEventContext::operation_submitted(
                        self.active.run_id(),
                        self.active.request_id(),
                        &self.topology,
                        &self.active,
                        &submission,
                    )
                }
                CompletionEventEvidence::Retired(participant_index) => {
                    TrustedExecutionEventContext::node_retired(
                        self.active.run_id(),
                        self.active.request_id(),
                        &self.topology,
                        &self.active,
                        &completion.participants()[*participant_index],
                    )
                }
            })
            .collect::<Vec<_>>();
        self.emitter.emit_batch(events, &contexts)?;
        self.completed_frames = self.completed_frames.saturating_add(1);
        Ok(())
    }

    fn complete_sequence(
        &mut self,
        receipt: &SequenceSessionTerminalReceipt,
        input_tokens: u64,
        output_tokens: u64,
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
            sequence_completed,
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
                output: output_tokens,
            },
        )?;
        self.emitter.emit(
            request_completed,
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
    active_binding: Arc<TrustedActiveSequenceBinding>,
    tokens: Mutex<Vec<u32>>,
    maximum_tokens: usize,
    active: AtomicBool,
    operation: AsyncMutex<()>,
    events: Option<Mutex<VNextExecutionJournal>>,
    prompt_tokens: u64,
    prefill_tokens_processed: AtomicUsize,
}

struct PreparedVNextPrefill<R: DeviceRuntime> {
    step: Arc<StepResourceLease<R>>,
    wave: PreparedStepSubmissionWave<R>,
}

struct VNextExecutionParticipant<'a, R: DeviceRuntime> {
    sequence: &'a Arc<VNextSequence<R>>,
    tokens: &'a [u32],
    span: &'a TokenSpanWork,
}

struct VNextDecodeCandidate<R: DeviceRuntime> {
    original_index: usize,
    sequence: Arc<VNextSequence<R>>,
    cache_id: String,
    next_token: u32,
}

impl<R: DeviceRuntime> VNextSequence<R> {
    fn preempt_for_recompute(&self) -> Result<()> {
        if !self.active.load(Ordering::Acquire) {
            return Err(FerrumError::already_exists(format!(
                "vNext request `{}` is already terminal",
                self.request_id
            )));
        }
        self.session
            .try_abort_if_quiescent()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        self.active.store(false, Ordering::Release);
        Ok(())
    }

    fn complete(&self, completion: &ExecutorSequenceCompletion) -> Result<()> {
        if completion.request_id() != &self.request_id {
            self.abort();
            return Err(FerrumError::request_validation(format!(
                "vNext completion request `{}` differs from cache owner `{}`",
                completion.request_id(),
                self.request_id
            )));
        }
        if completion.input_tokens() != self.prompt_tokens {
            self.abort();
            return Err(FerrumError::request_validation(format!(
                "vNext completion input count {} differs from admitted prompt count {}",
                completion.input_tokens(),
                self.prompt_tokens
            )));
        }
        if !self.active.swap(false, Ordering::AcqRel) {
            return Err(FerrumError::already_exists(format!(
                "vNext request `{}` is already terminal",
                self.request_id
            )));
        }
        let receipt = self.session.try_complete().map_err(|error| {
            let _ = self.session.request_cancel();
            let _ = self.session.try_abort();
            FerrumError::backend(format!("vNext sequence completion: {error}"))
        })?;
        if let Some(events) = &self.events {
            events
                .lock()
                .complete_sequence(
                    &receipt,
                    completion.input_tokens(),
                    completion.output_tokens(),
                )
                .map_err(|error| {
                    FerrumError::backend(format!("vNext execution journal completion: {error}"))
                })?;
        }
        Ok(())
    }

    fn abort(&self) {
        self.active.store(false, Ordering::Release);
        let _ = self.session.request_cancel();
        let _ = self.session.try_abort();
    }
}

impl<R: DeviceRuntime> Drop for VNextSequence<R> {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        let _ = self.session.request_cancel();
        let _ = self.session.try_abort();
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
    fn new(
        sequence: &Arc<VNextSequence<R>>,
        info: &ModelInfo,
        attention_head_dimension: usize,
        tokens: usize,
    ) -> Self {
        let mut block_table = BlockTable::new(16);
        block_table.sequence_length = tokens;
        Self {
            block_table,
            cache_id: sequence.cache_id.clone(),
            sequence: Arc::downgrade(sequence),
            device: info.device.clone(),
            num_layers: info.num_layers,
            num_heads: info.num_kv_heads,
            head_dim: attention_head_dimension,
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

enum VNextExecutionCapacityDecision<T> {
    Ready(T),
    Deferred(ExecutorExecutionCapacityDeferral),
}

enum VNextSequenceAdmissionDecision<R: DeviceRuntime> {
    Admitted(Arc<SequenceSession<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(VNextPrefillBackingDeferral<R>),
    PermanentRejected(AdmissionRejected),
}

enum VNextPrefillBackingDeferral<R: DeviceRuntime> {
    InitialSequence(InitialSequenceBackingDeferral<R>),
    Step(StepAdmissionBackingDeferral<R>),
    SubmissionWave(StepSubmissionWaveBackingDeferral<R>),
}

impl<R: DeviceRuntime> VNextPrefillBackingDeferral<R> {
    fn evidence(&self) -> &DynamicBackingDeferred {
        match self {
            Self::InitialSequence(deferred) => deferred.evidence(),
            Self::Step(deferred) => deferred.evidence(),
            Self::SubmissionWave(deferred) => deferred.evidence(),
        }
    }

    fn maintain(&self) -> std::result::Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        match self {
            Self::InitialSequence(deferred) => deferred.maintain(),
            Self::Step(deferred) => deferred.maintain(),
            Self::SubmissionWave(deferred) => deferred.maintain(),
        }
    }
}

enum PendingPrefillMaintenance<R: DeviceRuntime> {
    Logical(AdmissionDeferred),
    Backing(VNextPrefillBackingDeferral<R>),
}

impl<R: DeviceRuntime> PendingPrefillMaintenance<R> {
    fn projection(&self, request_id: &RequestId) -> Result<ExecutorPrefillMaintenanceDeferral> {
        match self {
            Self::Logical(deferred) => {
                ExecutorPrefillMaintenanceDeferral::from_admission(request_id, deferred)
            }
            Self::Backing(deferred) => {
                ExecutorPrefillMaintenanceDeferral::from_backing(request_id, deferred.evidence())
            }
        }
    }
}

enum VNextPrefillSlotState<R: DeviceRuntime> {
    Probing,
    Deferred {
        maintenance: Option<PendingPrefillMaintenance<R>>,
        maintaining: bool,
    },
    Ready(Arc<VNextSequence<R>>),
    Executing(Arc<VNextSequence<R>>),
    Terminal,
}

enum VNextPrefillProbeResolution<R: DeviceRuntime> {
    Deferred(AdmissionDeferred),
    MaintenanceDeferred {
        pending: PendingPrefillMaintenance<R>,
    },
    Ready(Arc<VNextSequence<R>>),
    PermanentRejected(AdmissionRejected),
}

impl<R: DeviceRuntime> VNextPrefillProbeResolution<R> {
    fn abort(self) {
        match self {
            Self::Deferred(_) => {}
            Self::MaintenanceDeferred { pending } => drop(pending),
            Self::Ready(sequence) => sequence.abort(),
            Self::PermanentRejected(_) => {}
        }
    }
}

impl<R: DeviceRuntime> VNextPrefillSlotState<R> {
    fn abort(self) {
        match self {
            Self::Deferred { maintenance, .. } => drop(maintenance),
            Self::Ready(sequence) | Self::Executing(sequence) => sequence.abort(),
            Self::Probing | Self::Terminal => {}
        }
    }
}

struct VNextPrefillSlot<R: DeviceRuntime> {
    request_id: RequestId,
    work_shape: ResourceWorkShape,
    cancelled: AtomicBool,
    state: Mutex<VNextPrefillSlotState<R>>,
}

impl<R: DeviceRuntime> VNextPrefillSlot<R> {
    fn new(request_id: RequestId, work_shape: ResourceWorkShape) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            work_shape,
            cancelled: AtomicBool::new(false),
            state: Mutex::new(VNextPrefillSlotState::Probing),
        })
    }
}

impl<R: DeviceRuntime> Drop for VNextPrefillSlot<R> {
    fn drop(&mut self) {
        let state = std::mem::replace(self.state.get_mut(), VNextPrefillSlotState::Terminal);
        state.abort();
    }
}

struct VNextSequenceRegistry<R: DeviceRuntime> {
    prefills: HashMap<RequestId, Arc<VNextPrefillSlot<R>>>,
    active: HashMap<String, Arc<VNextSequence<R>>>,
}

impl<R: DeviceRuntime> Default for VNextSequenceRegistry<R> {
    fn default() -> Self {
        Self {
            prefills: HashMap::new(),
            active: HashMap::new(),
        }
    }
}

impl<R: DeviceRuntime> VNextSequenceRegistry<R> {
    fn total_len(&self) -> usize {
        self.prefills.len() + self.active.len()
    }

    fn begin_prefill_probe(
        &mut self,
        request_id: &RequestId,
        work_shape: &ResourceWorkShape,
    ) -> Result<Arc<VNextPrefillSlot<R>>> {
        if self
            .active
            .values()
            .any(|sequence| sequence.request_id == *request_id)
        {
            return Err(FerrumError::already_exists(format!(
                "vNext request `{request_id}` is already active"
            )));
        }
        if let Some(slot) = self.prefills.get(request_id).cloned() {
            if slot.work_shape != *work_shape {
                return Err(FerrumError::request_validation(format!(
                    "vNext prefill retry for `{request_id}` differs from its deferred work shape"
                )));
            }
            if slot.cancelled.load(Ordering::Acquire) {
                return Err(FerrumError::cancelled(format!(
                    "vNext prefill probe for `{request_id}` was cancelled"
                )));
            }
            let mut state = slot.state.lock();
            let prior = std::mem::replace(&mut *state, VNextPrefillSlotState::Probing);
            match prior {
                VNextPrefillSlotState::Deferred {
                    maintenance: None,
                    maintaining: false,
                } => {
                    drop(state);
                    Ok(slot)
                }
                other => {
                    *state = other;
                    Err(FerrumError::already_exists(format!(
                        "vNext request `{request_id}` already retained prefill state"
                    )))
                }
            }
        } else {
            let slot = VNextPrefillSlot::new(request_id.clone(), work_shape.clone());
            self.prefills.insert(request_id.clone(), Arc::clone(&slot));
            Ok(slot)
        }
    }

    fn begin_prefill_execution(
        &mut self,
        request_id: &RequestId,
    ) -> Result<(Arc<VNextPrefillSlot<R>>, Arc<VNextSequence<R>>)> {
        let slot = self.prefills.get(request_id).cloned().ok_or_else(|| {
            FerrumError::request_validation(format!(
                "vNext prefill for `{request_id}` has no retained admission authority"
            ))
        })?;
        if slot.cancelled.load(Ordering::Acquire) {
            return Err(FerrumError::cancelled(format!(
                "vNext prefill admission for `{request_id}` is no longer active"
            )));
        }
        let mut state = slot.state.lock();
        let prior = std::mem::replace(&mut *state, VNextPrefillSlotState::Terminal);
        let VNextPrefillSlotState::Ready(sequence) = prior else {
            *state = prior;
            return Err(FerrumError::request_validation(format!(
                "vNext prefill for `{request_id}` is not ready for execution"
            )));
        };
        *state = VNextPrefillSlotState::Executing(Arc::clone(&sequence));
        drop(state);
        Ok((slot, sequence))
    }

    fn activate(
        &mut self,
        slot: &Arc<VNextPrefillSlot<R>>,
        sequence: &Arc<VNextSequence<R>>,
    ) -> Result<()> {
        let request_id = &slot.request_id;
        if slot.cancelled.load(Ordering::Acquire)
            || !self
                .prefills
                .get(request_id)
                .is_some_and(|current| Arc::ptr_eq(current, slot))
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
        let mut state = slot.state.lock();
        let executing = matches!(
            &*state,
            VNextPrefillSlotState::Executing(current) if Arc::ptr_eq(current, sequence)
        );
        if !executing {
            return Err(FerrumError::cancelled(format!(
                "vNext prefill execution for `{request_id}` lost its slot authority"
            )));
        }
        *state = VNextPrefillSlotState::Terminal;
        drop(state);
        self.prefills.remove(request_id);
        self.active
            .insert(sequence.cache_id.clone(), Arc::clone(sequence));
        Ok(())
    }

    fn restore_prefill_ready(
        &mut self,
        slot: &Arc<VNextPrefillSlot<R>>,
        sequence: &Arc<VNextSequence<R>>,
    ) -> Result<()> {
        let request_id = &slot.request_id;
        if slot.cancelled.load(Ordering::Acquire)
            || !self
                .prefills
                .get(request_id)
                .is_some_and(|current| Arc::ptr_eq(current, slot))
        {
            return Err(FerrumError::cancelled(format!(
                "vNext prefill admission for `{request_id}` is no longer active"
            )));
        }
        let mut state = slot.state.lock();
        if !matches!(
            &*state,
            VNextPrefillSlotState::Executing(current) if Arc::ptr_eq(current, sequence)
        ) {
            return Err(FerrumError::internal(format!(
                "vNext prefill execution for `{request_id}` lost its slot authority"
            )));
        }
        *state = VNextPrefillSlotState::Ready(Arc::clone(sequence));
        Ok(())
    }

    fn cancel_prefill(&mut self, request_id: &RequestId) -> bool {
        let Some(slot) = self.prefills.get(request_id).cloned() else {
            return false;
        };
        slot.cancelled.store(true, Ordering::Release);
        let mut state = slot.state.lock();
        if let VNextPrefillSlotState::Executing(sequence) = &*state {
            let sequence = Arc::clone(sequence);
            drop(state);
            sequence.abort();
            return true;
        }
        let defer_cleanup = matches!(&*state, VNextPrefillSlotState::Probing)
            || matches!(
                &*state,
                VNextPrefillSlotState::Deferred {
                    maintaining: true,
                    ..
                }
            );
        if defer_cleanup {
            return true;
        }
        let prior = std::mem::replace(&mut *state, VNextPrefillSlotState::Terminal);
        drop(state);
        self.prefills.remove(request_id);
        prior.abort();
        true
    }

    fn write_execution_capacity_release_sources(
        &self,
        preemption: &ExecutorExecutionCapacityPreemption,
        sources: &mut Vec<CapacityAvailabilitySource>,
    ) -> Result<bool> {
        sources.clear();
        let retained = self.prefills.get(preemption.request_id()).and_then(|slot| {
            let state = slot.state.lock();
            match &*state {
                VNextPrefillSlotState::Ready(sequence)
                    if sequence.cache_id == preemption.cache_id() =>
                {
                    Some(Arc::clone(sequence))
                }
                _ => None,
            }
        });
        let sequence = retained.or_else(|| {
            self.active
                .get(preemption.cache_id())
                .filter(|sequence| sequence.request_id == *preemption.request_id())
                .cloned()
        });
        let Some(sequence) = sequence else {
            return Ok(false);
        };
        if !sequence.active.load(Ordering::Acquire) {
            return Ok(false);
        }
        sequence
            .session
            .write_release_capacity_sources(sources)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        Ok(true)
    }

    fn preempt_execution_capacity(
        &mut self,
        preemption: &ExecutorExecutionCapacityPreemption,
    ) -> Result<ExecutorExecutionCapacityPreemptionAuthority> {
        let request_id = preemption.request_id();
        if let Some(slot) = self.prefills.get(request_id).cloned() {
            let mut state = slot.state.lock();
            let sequence = match &*state {
                VNextPrefillSlotState::Ready(sequence) => Arc::clone(sequence),
                VNextPrefillSlotState::Executing(_) => {
                    return Err(FerrumError::internal(format!(
                        "vNext request `{request_id}` cannot preempt an executing prefill"
                    )));
                }
                _ => {
                    return Err(FerrumError::request_validation(format!(
                        "vNext request `{request_id}` has no releasable retained prefill authority"
                    )));
                }
            };
            if sequence.cache_id != preemption.cache_id() {
                return Err(FerrumError::request_validation(format!(
                    "vNext request `{request_id}` preemption cache mismatch: expected {}, found {}",
                    preemption.cache_id(),
                    sequence.cache_id
                )));
            }
            sequence.preempt_for_recompute()?;
            *state = VNextPrefillSlotState::Terminal;
            slot.cancelled.store(true, Ordering::Release);
            drop(state);
            self.prefills.remove(request_id);
            return Ok(ExecutorExecutionCapacityPreemptionAuthority::RetainedPrefill);
        }

        let sequence = self
            .active
            .get(preemption.cache_id())
            .cloned()
            .ok_or_else(|| {
                let detail = self
                    .active
                    .values()
                    .find(|sequence| sequence.request_id == *request_id)
                    .map_or_else(
                        || "no active sequence".to_string(),
                        |sequence| format!("active cache is {}", sequence.cache_id),
                    );
                FerrumError::request_validation(format!(
                    "vNext request `{request_id}` preemption did not match an authority: {detail}"
                ))
            })?;
        if sequence.request_id != *request_id {
            return Err(FerrumError::request_validation(format!(
                "vNext cache `{}` belongs to request {}, not {request_id}",
                preemption.cache_id(),
                sequence.request_id
            )));
        }
        sequence.preempt_for_recompute()?;
        self.active.remove(preemption.cache_id());
        Ok(ExecutorExecutionCapacityPreemptionAuthority::ActiveSequence)
    }

    fn finish_prefill_execution(
        &mut self,
        slot: &Arc<VNextPrefillSlot<R>>,
        sequence: &Arc<VNextSequence<R>>,
    ) {
        if !self
            .prefills
            .get(&slot.request_id)
            .is_some_and(|current| Arc::ptr_eq(current, slot))
        {
            return;
        }
        let mut state = slot.state.lock();
        if !matches!(
            &*state,
            VNextPrefillSlotState::Executing(current) if Arc::ptr_eq(current, sequence)
        ) {
            return;
        }
        *state = VNextPrefillSlotState::Terminal;
        drop(state);
        self.prefills.remove(&slot.request_id);
    }
}

struct VNextPrefillExecutionGuard<'a, R: DeviceRuntime> {
    registry: &'a Mutex<VNextSequenceRegistry<R>>,
    slot: Arc<VNextPrefillSlot<R>>,
    sequence: Arc<VNextSequence<R>>,
    armed: bool,
}

impl<'a, R: DeviceRuntime> VNextPrefillExecutionGuard<'a, R> {
    fn new(
        registry: &'a Mutex<VNextSequenceRegistry<R>>,
        slot: Arc<VNextPrefillSlot<R>>,
        sequence: Arc<VNextSequence<R>>,
    ) -> Self {
        Self {
            registry,
            slot,
            sequence,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    fn restore_ready(&mut self) -> Result<()> {
        self.registry
            .lock()
            .restore_prefill_ready(&self.slot, &self.sequence)?;
        self.disarm();
        Ok(())
    }
}

impl<R: DeviceRuntime> Drop for VNextPrefillExecutionGuard<'_, R> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.registry
            .lock()
            .finish_prefill_execution(&self.slot, &self.sequence);
        self.sequence.abort();
    }
}

/// Backend-neutral executor over one concrete device runtime and operation
/// registry. CUDA and Metal factories differ only in composition creation.
pub struct VNextModelExecutor<R: DeviceRuntime> {
    info: ModelInfo,
    resolved_plan: ResolvedModelPlan,
    runtime: Arc<R>,
    providers: BoundOperationProviderSet<R>,
    policy: ResolvedRuntimePolicy,
    plan_resources: Arc<PlanRuntimeResources<R>>,
    lane: Arc<ExecutionLane<R>>,
    completion_worker: VNextCompletionWorker,
    reaper: Arc<CompletionReaper<R>>,
    io: VNextIoBinding,
    maximum_model_tokens: usize,
    attention_head_dimension: usize,
    run_id: RunId,
    family_fingerprint: String,
    program_fingerprint: String,
    static_bytes: u64,
    reusable_execution_enabled: bool,
    reusable_execution_supported: bool,
    reusable_execution_startup_plan: Option<VNextReusableExecutionStartupPlan>,
    startup_preparation: Mutex<VNextStartupPreparationState>,
    sequences: Mutex<VNextSequenceRegistry<R>>,
    event_sink: RwLock<Option<Arc<dyn ExecutionEventSink>>>,
    device_timing_mode: AtomicU8,
    metrics: VNextExecutorMetrics,
}

impl<R: DeviceRuntime> fmt::Debug for VNextModelExecutor<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VNextModelExecutor")
            .field("model_id", &self.info.model_id)
            .field(
                "plan_id",
                self.resolved_plan.execution_plan().payload().plan_id(),
            )
            .field("device", &self.runtime.descriptor().id)
            .field("maximum_model_tokens", &self.maximum_model_tokens)
            .field("retained_sequences", &self.sequences.lock().total_len())
            .finish_non_exhaustive()
    }
}

struct VNextStartupSequence {
    request_id: RequestId,
    kv_cache: Arc<dyn KvCacheHandle>,
}

struct VNextStartupSequenceGuard<'executor, R: DeviceRuntime> {
    executor: &'executor VNextModelExecutor<R>,
    pending_request: Option<RequestId>,
    sequences: Vec<VNextStartupSequence>,
}

impl<'executor, R: DeviceRuntime> VNextStartupSequenceGuard<'executor, R> {
    fn new(executor: &'executor VNextModelExecutor<R>) -> Self {
        Self {
            executor,
            pending_request: None,
            sequences: Vec::new(),
        }
    }

    fn begin_request(&mut self, request_id: RequestId) {
        debug_assert!(self.pending_request.is_none());
        self.pending_request = Some(request_id);
    }

    fn activate(&mut self, kv_cache: Arc<dyn KvCacheHandle>) {
        let request_id = self
            .pending_request
            .take()
            .expect("startup sequence activation requires a pending request");
        self.sequences.push(VNextStartupSequence {
            request_id,
            kv_cache,
        });
    }

    fn cancel_pending(&mut self) {
        if let Some(request_id) = self.pending_request.take() {
            self.executor.sequences.lock().cancel_prefill(&request_id);
        }
    }
}

impl<R: DeviceRuntime> Drop for VNextStartupSequenceGuard<'_, R> {
    fn drop(&mut self) {
        self.cancel_pending();
        let sequences = self
            .sequences
            .drain(..)
            .filter_map(|startup| {
                self.executor
                    .sequences
                    .lock()
                    .active
                    .remove(&startup.kv_cache.cache_id())
            })
            .collect::<Vec<_>>();
        for sequence in sequences {
            sequence.abort();
        }
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub fn from_runtime_composition<F>(
        prepared: PreparedProductionModel,
        info: ModelInfo,
        engine_config: &EngineConfig,
        runtime: Arc<R>,
        registry: OperationRuntimeRegistry<R>,
        catalog: CapabilityCatalog,
        resolve_plan: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &PreparedProductionModel,
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        let attention_head_dimension = prepared.descriptor().attention_head_dimension();
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
        let resolved_plan =
            resolve_plan(&prepared, &config.runtime_policy, &catalog, &compilation)?;
        if resolved_plan.execution_plan() != compilation.executable().execution_plan() {
            return Err(FerrumError::internal(
                "product composition returned a different execution plan than the compiler",
            ));
        }
        let providers = registry
            .bind_plan(&resolved_plan)
            .map_err(|error| FerrumError::model(format!("vNext provider binding: {error}")))?;
        let io = Self::resolve_io(&resolved_plan, &input_id, &output_id, info.vocab_size)?;
        let family_fingerprint = family
            .fingerprint()
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let program_fingerprint = family
            .program()
            .fingerprint()
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let static_bytes = resolved_plan
            .execution_plan()
            .payload()
            .memory()
            .static_bytes();
        let run_id = RunId::new(format!("run.vnext.{}", uuid::Uuid::new_v4()))
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let provision_request = RequestIdentity::new(format!("request.vnext.provision.{run_id}"))
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let provisioned = resolved_plan
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
                    resolved_plan.execution_plan(),
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
        let completion_worker = VNextCompletionWorker::new().map_err(|error| {
            FerrumError::device(format!("vNext completion worker creation failed: {error}"))
        })?;
        let reaper = CompletionReaper::new();
        let reusable_execution_supported = runtime
            .descriptor()
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID);
        let reusable_execution_startup_plan =
            if config.reusable_execution_enabled && reusable_execution_supported {
                Some(VNextReusableExecutionStartupPlan::resolve(
                    config.runtime_policy.memory().maximum_active_sequences,
                    config.runtime_policy.admission().maximum_scheduled_tokens,
                    config.maximum_model_tokens,
                    &config.reusable_execution_prefill_token_counts,
                    resolved_plan.execution_plan().payload().nodes().len(),
                )?)
            } else {
                None
            };

        Ok(Self {
            info,
            resolved_plan,
            runtime,
            providers,
            policy: config.runtime_policy,
            plan_resources,
            lane,
            completion_worker,
            reaper,
            io,
            maximum_model_tokens: config.maximum_model_tokens,
            attention_head_dimension,
            run_id,
            family_fingerprint,
            program_fingerprint,
            static_bytes,
            reusable_execution_enabled: config.reusable_execution_enabled,
            reusable_execution_supported,
            reusable_execution_startup_plan,
            startup_preparation: Mutex::new(VNextStartupPreparationState::Pending),
            sequences: Mutex::new(VNextSequenceRegistry::default()),
            event_sink: RwLock::new(None),
            device_timing_mode: AtomicU8::new(DeviceTimingMode::Off as u8),
            metrics: VNextExecutorMetrics::default(),
        })
    }

    fn startup_token_tensor(tokens: &[u32]) -> Result<TensorRef> {
        if tokens.is_empty() {
            return Err(FerrumError::internal(
                "vNext startup token tensor must be non-empty",
            ));
        }
        let tensor = candle_core::Tensor::new(tokens, &candle_core::Device::Cpu)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| FerrumError::model(format!("vNext startup token tensor: {error}")))?;
        Ok(common::wrap_tensor(tensor))
    }

    async fn admit_startup_sequence(
        &self,
        resources: &mut VNextStartupSequenceGuard<'_, R>,
        input_tokens: &[TokenId],
        input_tensor: &TensorRef,
        maximum_sequence_tokens: usize,
    ) -> Result<bool> {
        let request_id = RequestId::new();
        resources.begin_request(request_id.clone());
        let mut maintenance_attempts = 0_u32;
        loop {
            match self.try_admit_prefill(ExecutorPrefillAdmission::new(
                &request_id,
                input_tokens,
                maximum_sequence_tokens,
            ))? {
                ExecutorPrefillAdmissionDecision::Admitted(receipt) => {
                    if receipt.request_id != request_id {
                        return Err(FerrumError::internal(
                            "vNext startup prefill admission changed request identity",
                        ));
                    }
                    break;
                }
                ExecutorPrefillAdmissionDecision::MaintenanceDeferred(_) => {
                    if maintenance_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(FerrumError::resource_exhausted(format!(
                            "vNext startup prefill backing did not converge after {maintenance_attempts} attempts"
                        )));
                    }
                    maintenance_attempts += 1;
                    match self.maintain_prefill_backing(&request_id)? {
                        ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. } => continue,
                        ExecutorPrefillMaintenanceOutcome::WaitForRelease { .. } => {
                            resources.cancel_pending();
                            return Ok(false);
                        }
                        ExecutorPrefillMaintenanceOutcome::NoLongerPending => {
                            return Err(FerrumError::internal(
                                "vNext startup backing maintenance lost its retained request",
                            ));
                        }
                    }
                }
                ExecutorPrefillAdmissionDecision::Deferred(_) => {
                    resources.cancel_pending();
                    return Ok(false);
                }
                ExecutorPrefillAdmissionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::resource_exhausted(format!(
                        "vNext startup prefill was permanently rejected: {rejected:?}"
                    )));
                }
            }
        }

        let input_token_count = input_tokens.len();
        let chunk = PrefillChunk::new(0, input_token_count, input_token_count)?;
        let input = PrefillInput::new(Arc::clone(input_tensor))
            .with_request_context(request_id, maximum_sequence_tokens)
            .with_chunk(chunk);
        match self.execute_prefill_with_capacity(&input).await? {
            ExecutorPrefillOutcome::Completed(completion) => {
                let (output, planned, completed, _) = completion.into_parts();
                if planned != chunk || completed != chunk {
                    return Err(FerrumError::internal(format!(
                        "vNext startup prefill did not complete its exact {input_token_count}-token frontier"
                    )));
                }
                resources.activate(output.kv_cache);
                Ok(true)
            }
            ExecutorPrefillOutcome::Deferred(_) => {
                resources.cancel_pending();
                Ok(false)
            }
        }
    }

    async fn execute_startup_prefill_request(
        &self,
        token_count: usize,
        phase: &'static str,
    ) -> Result<()> {
        let raw_tokens = vec![0_u32; token_count];
        let input_tokens = raw_tokens
            .iter()
            .copied()
            .map(TokenId::new)
            .collect::<Vec<_>>();
        let input_tensor = Self::startup_token_tensor(&raw_tokens)?;
        let mut resources = VNextStartupSequenceGuard::new(self);
        if !self
            .admit_startup_sequence(&mut resources, &input_tokens, &input_tensor, token_count)
            .await?
        {
            return Err(FerrumError::resource_exhausted(format!(
                "vNext startup {phase} could not admit a {token_count}-token prefill descriptor"
            )));
        }
        if resources.sequences.len() != 1 {
            return Err(FerrumError::internal(format!(
                "vNext startup {phase} retained {} sequences for one prefill descriptor",
                resources.sequences.len()
            )));
        }
        Ok(())
    }

    async fn execute_startup_decode_pass(
        &self,
        resources: &mut VNextStartupSequenceGuard<'_, R>,
        input_tensor: &TensorRef,
        width: usize,
        phase: &'static str,
    ) -> Result<()> {
        if width == 0 || width > resources.sequences.len() {
            return Err(FerrumError::internal(format!(
                "vNext startup {phase} width {width} exceeds {} retained sequences",
                resources.sequences.len()
            )));
        }
        let inputs = resources
            .sequences
            .iter()
            .take(width)
            .map(|sequence| {
                DecodeInput::new(Arc::clone(input_tensor), Arc::clone(&sequence.kv_cache))
                    .with_request_id(sequence.request_id.clone())
            })
            .collect::<Vec<_>>();
        match self.execute_decode_batch(&inputs).await? {
            ExecutorBatchDecodeOutcome::Completed(outputs) => {
                if outputs.len() != width {
                    return Err(FerrumError::internal(format!(
                        "vNext startup {phase} returned {} outputs for width {width}",
                        outputs.len()
                    )));
                }
                for (sequence, output) in resources.sequences.iter_mut().take(width).zip(outputs) {
                    sequence.kv_cache = output.kv_cache;
                }
                Ok(())
            }
            ExecutorBatchDecodeOutcome::Deferred(deferred) => {
                Err(FerrumError::resource_exhausted(format!(
                    "vNext startup {phase} width {width} deferred at {:?}",
                    deferred.stage()
                )))
            }
        }
    }

    async fn prepare_reusable_execution_startup(
        &self,
    ) -> Result<VNextReusableExecutionStartupReport> {
        let started = Instant::now();
        let Some(plan) = self.reusable_execution_startup_plan.clone() else {
            return Ok(VNextReusableExecutionStartupReport {
                enabled: self.reusable_execution_enabled,
                supported: self.reusable_execution_supported,
                requested_descriptors: Vec::new(),
                prepared_descriptors: Vec::new(),
                requested_decode_widths: Vec::new(),
                prepared_decode_widths: Vec::new(),
                requested_prefill_token_counts: Vec::new(),
                prepared_prefill_token_counts: Vec::new(),
                synthetic_sequences: 0,
                eager_warmup_waves: 0,
                capture_waves: 0,
                replay_validation_waves: 0,
                device_preparation: DeviceReusableExecutionPreparation::unsupported(),
                elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            });
        };

        let requested_descriptors = plan.descriptors.clone();
        let requested_decode_widths = plan.decode_widths();
        let requested_prefill_token_counts = plan.prefill_token_counts();
        for _ in 0..REUSABLE_EXECUTION_WARMUP_PASSES {
            for token_count in requested_prefill_token_counts.iter().copied() {
                self.execute_startup_prefill_request(token_count, "eager prefill warmup")
                    .await?;
            }
        }

        let input_tokens = [TokenId::new(0)];
        let input_tensor = Self::startup_token_tensor(&[0])?;
        let mut resources = VNextStartupSequenceGuard::new(self);
        let requested_sequences = requested_decode_widths.first().copied().ok_or_else(|| {
            FerrumError::internal("vNext reusable execution plan has no decode widths")
        })?;
        for _ in 0..requested_sequences {
            if !self
                .admit_startup_sequence(
                    &mut resources,
                    &input_tokens,
                    &input_tensor,
                    plan.maximum_decode_sequence_tokens,
                )
                .await?
            {
                break;
            }
        }
        if resources.sequences.is_empty() {
            return Err(FerrumError::resource_exhausted(
                "vNext reusable execution startup could not admit one synthetic sequence",
            ));
        }
        let prepared_decode_widths = plan.widths_for_available_sequences(resources.sequences.len());

        for _ in 0..REUSABLE_EXECUTION_WARMUP_PASSES {
            for width in prepared_decode_widths.iter().copied() {
                self.execute_startup_decode_pass(
                    &mut resources,
                    &input_tensor,
                    width,
                    "eager warmup",
                )
                .await?;
            }
        }

        let configured = self
            .lane
            .configure_reusable_executables(plan.device_plan)
            .map_err(|error| {
                FerrumError::device(format!(
                    "vNext reusable execution configuration failed: {error}"
                ))
            })?;
        if configured.state() != DeviceReusableExecutionPreparationState::Preparing {
            return Err(FerrumError::internal(format!(
                "vNext reusable execution capability configured as {:?}",
                configured.state()
            )));
        }

        for _ in 0..REUSABLE_EXECUTION_CAPTURE_PASSES {
            for width in prepared_decode_widths.iter().copied() {
                self.execute_startup_decode_pass(&mut resources, &input_tensor, width, "capture")
                    .await?;
            }
        }
        let captured = self
            .lane
            .reusable_executable_preparation()
            .map_err(|error| {
                FerrumError::device(format!(
                    "vNext reusable execution capture inspection failed: {error}"
                ))
            })?;
        if captured.state() != DeviceReusableExecutionPreparationState::Preparing
            || captured.resident_executables() == 0
            || captured.rejected_executables() != 0
            || captured.capacity_deferred_executables() != 0
            || captured.captured_executables() != captured.uploaded_executables()
            || captured.uploaded_executables() != captured.resident_executables()
        {
            return Err(FerrumError::device(format!(
                "vNext reusable execution capture receipt is incomplete: {captured:?}"
            )));
        }
        for _ in 0..REUSABLE_EXECUTION_REPLAY_VALIDATION_PASSES {
            for width in prepared_decode_widths.iter().copied() {
                self.execute_startup_decode_pass(
                    &mut resources,
                    &input_tensor,
                    width,
                    "replay validation",
                )
                .await?;
            }
        }
        let replayed = self
            .lane
            .reusable_executable_preparation()
            .map_err(|error| {
                FerrumError::device(format!(
                    "vNext reusable execution replay inspection failed: {error}"
                ))
            })?;
        if replayed != captured {
            return Err(FerrumError::device(format!(
                "vNext replay validation compiled or changed executable state: before={captured:?}, after={replayed:?}"
            )));
        }

        let synthetic_sequences = resources.sequences.len();
        drop(resources);

        let mut captured = replayed;
        for token_count in requested_prefill_token_counts.iter().copied() {
            self.execute_startup_prefill_request(token_count, "prefill capture")
                .await?;
            let prefill_captured =
                self.lane
                    .reusable_executable_preparation()
                    .map_err(|error| {
                        FerrumError::device(format!(
                            "vNext {token_count}-token prefill capture inspection failed: {error}"
                        ))
                    })?;
            if prefill_captured.state() != DeviceReusableExecutionPreparationState::Preparing
                || prefill_captured.resident_executables() == 0
                || prefill_captured.rejected_executables() != 0
                || prefill_captured.capacity_deferred_executables() != 0
                || prefill_captured.captured_executables()
                    != prefill_captured.uploaded_executables()
                || prefill_captured.uploaded_executables()
                    != prefill_captured.resident_executables()
                || prefill_captured.captured_executables() < captured.captured_executables()
            {
                return Err(FerrumError::device(format!(
                    "vNext {token_count}-token prefill capture receipt is incomplete: before={captured:?}, after={prefill_captured:?}"
                )));
            }

            self.execute_startup_prefill_request(token_count, "fresh-request prefill replay")
                .await?;
            let prefill_replayed = self
                .lane
                .reusable_executable_preparation()
                .map_err(|error| {
                    FerrumError::device(format!(
                        "vNext {token_count}-token fresh-request prefill replay inspection failed: {error}"
                    ))
                })?;
            if prefill_replayed != prefill_captured {
                return Err(FerrumError::device(format!(
                    "vNext {token_count}-token fresh-request prefill replay changed executable state: before={prefill_captured:?}, after={prefill_replayed:?}"
                )));
            }
            captured = prefill_replayed;
        }

        let device_preparation = self.lane.seal_reusable_executables().map_err(|error| {
            FerrumError::device(format!("vNext reusable execution sealing failed: {error}"))
        })?;
        if device_preparation.state() != DeviceReusableExecutionPreparationState::Ready
            || device_preparation.resident_executables() == 0
            || device_preparation.uploaded_executables() < device_preparation.resident_executables()
            || device_preparation.captured_executables() != captured.captured_executables()
            || device_preparation.uploaded_executables() != captured.uploaded_executables()
        {
            return Err(FerrumError::device(format!(
                "vNext reusable execution sealing produced an unusable receipt: {device_preparation:?}"
            )));
        }
        let prepared_descriptors = prepared_decode_widths
            .iter()
            .copied()
            .map(VNextReusableExecutionDescriptor::uniform_decode)
            .chain(
                requested_prefill_token_counts
                    .iter()
                    .copied()
                    .map(VNextReusableExecutionDescriptor::prefill),
            )
            .collect::<Vec<_>>();
        let prepared_wave_shapes =
            prepared_decode_widths.len() + requested_prefill_token_counts.len();
        Ok(VNextReusableExecutionStartupReport {
            enabled: true,
            supported: true,
            requested_descriptors,
            prepared_descriptors,
            requested_decode_widths,
            prepared_decode_widths: prepared_decode_widths.clone(),
            requested_prefill_token_counts: requested_prefill_token_counts.clone(),
            prepared_prefill_token_counts: requested_prefill_token_counts,
            synthetic_sequences,
            eager_warmup_waves: prepared_wave_shapes * REUSABLE_EXECUTION_WARMUP_PASSES,
            capture_waves: prepared_wave_shapes * REUSABLE_EXECUTION_CAPTURE_PASSES,
            replay_validation_waves: prepared_wave_shapes
                * REUSABLE_EXECUTION_REPLAY_VALIDATION_PASSES,
            device_preparation,
            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        })
    }

    fn reset_request_metrics_after_startup(&self) -> Result<()> {
        if self.sequences.lock().total_len() != 0 {
            return Err(FerrumError::internal(
                "vNext startup cleanup retained synthetic sequence authority",
            ));
        }
        if !self.completion_worker.reset_metrics_if_idle() {
            return Err(FerrumError::internal(
                "vNext startup cleanup left a completion task in flight",
            ));
        }
        self.metrics.reset_after_startup();
        Ok(())
    }

    fn device_timing_mode(&self) -> DeviceTimingMode {
        match self.device_timing_mode.load(Ordering::Acquire) {
            value if value == DeviceTimingMode::Completion as u8 => DeviceTimingMode::Completion,
            _ => DeviceTimingMode::Off,
        }
    }

    fn host_dispatch_timing_enabled(&self) -> bool {
        self.device_timing_mode() != DeviceTimingMode::Off
    }

    fn resolve_io(
        executable: &impl ExecutablePlanView,
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

    fn fail_prefill_probe(&self, slot: &Arc<VNextPrefillSlot<R>>) {
        let mut sequences = self.sequences.lock();
        if !sequences
            .prefills
            .get(&slot.request_id)
            .is_some_and(|current| Arc::ptr_eq(current, slot))
        {
            return;
        }
        let mut state = slot.state.lock();
        if matches!(&*state, VNextPrefillSlotState::Probing) {
            *state = VNextPrefillSlotState::Terminal;
            drop(state);
            sequences.prefills.remove(&slot.request_id);
        }
    }

    fn publish_prefill_probe(
        &self,
        slot: &Arc<VNextPrefillSlot<R>>,
        resolution: VNextPrefillProbeResolution<R>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        let projection = match &resolution {
            VNextPrefillProbeResolution::MaintenanceDeferred { pending, .. } => {
                pending.projection(&slot.request_id).map(Some)
            }
            _ => Ok(None),
        };
        let projection = match projection {
            Ok(projection) => projection,
            Err(error) => {
                resolution.abort();
                self.fail_prefill_probe(slot);
                return Err(error);
            }
        };
        let mut sequences = self.sequences.lock();
        let current = sequences
            .prefills
            .get(&slot.request_id)
            .is_some_and(|current| Arc::ptr_eq(current, slot));
        let mut state = slot.state.lock();
        if !current
            || slot.cancelled.load(Ordering::Acquire)
            || !matches!(&*state, VNextPrefillSlotState::Probing)
        {
            let prior = std::mem::replace(&mut *state, VNextPrefillSlotState::Terminal);
            drop(state);
            if current {
                sequences.prefills.remove(&slot.request_id);
            }
            drop(sequences);
            prior.abort();
            resolution.abort();
            return Err(FerrumError::cancelled(format!(
                "vNext prefill probe for `{}` lost its request authority",
                slot.request_id
            )));
        }

        let (decision, terminal) = match resolution {
            VNextPrefillProbeResolution::Deferred(deferred) => {
                *state = VNextPrefillSlotState::Deferred {
                    maintenance: None,
                    maintaining: false,
                };
                (ExecutorPrefillAdmissionDecision::Deferred(deferred), false)
            }
            VNextPrefillProbeResolution::MaintenanceDeferred { pending } => {
                *state = VNextPrefillSlotState::Deferred {
                    maintenance: Some(pending),
                    maintaining: false,
                };
                (
                    ExecutorPrefillAdmissionDecision::MaintenanceDeferred(
                        projection.expect("maintenance projection was constructed"),
                    ),
                    false,
                )
            }
            VNextPrefillProbeResolution::Ready(sequence) => {
                *state = VNextPrefillSlotState::Ready(sequence);
                (
                    ExecutorPrefillAdmissionDecision::Admitted(ExecutorPrefillAdmissionReceipt {
                        request_id: slot.request_id.clone(),
                    }),
                    false,
                )
            }
            VNextPrefillProbeResolution::PermanentRejected(rejected) => {
                *state = VNextPrefillSlotState::Terminal;
                (
                    ExecutorPrefillAdmissionDecision::PermanentRejected(rejected),
                    true,
                )
            }
        };
        drop(state);
        if terminal {
            sequences.prefills.remove(&slot.request_id);
        }
        Ok(decision)
    }

    fn resolve_prefill_probe(
        &self,
        request_id: &RequestId,
        maximum_tokens: usize,
        tokens: Vec<u32>,
        work: ResourceWorkShape,
    ) -> Result<VNextPrefillProbeResolution<R>> {
        let identity = RequestIdentity::new(format!("request.product.{request_id}"))
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let session = match self.try_admit_sequence(identity, work)? {
            VNextSequenceAdmissionDecision::Admitted(session) => session,
            VNextSequenceAdmissionDecision::Deferred(deferred) => {
                if deferred.action() == DeferredAction::AwaitBackingGrowth {
                    return Ok(VNextPrefillProbeResolution::MaintenanceDeferred {
                        pending: PendingPrefillMaintenance::Logical(deferred),
                    });
                }
                return Ok(VNextPrefillProbeResolution::Deferred(deferred));
            }
            VNextSequenceAdmissionDecision::BackingDeferred(deferred) => {
                return Ok(VNextPrefillProbeResolution::MaintenanceDeferred {
                    pending: PendingPrefillMaintenance::Backing(deferred),
                });
            }
            VNextSequenceAdmissionDecision::PermanentRejected(rejected) => {
                return Ok(VNextPrefillProbeResolution::PermanentRejected(rejected));
            }
        };
        let active_binding = match TrustedActiveSequenceBinding::from_session(&session) {
            Ok(active_binding) => Arc::new(active_binding),
            Err(error) => {
                let _ = session.request_cancel();
                let _ = session.try_abort();
                return Err(FerrumError::backend(error.to_string()));
            }
        };
        let events = match self.execution_journal(&active_binding) {
            Ok(events) => events,
            Err(error) => {
                let _ = session.request_cancel();
                let _ = session.try_abort();
                return Err(error);
            }
        };
        let prompt_tokens = match u64::try_from(tokens.len()) {
            Ok(prompt_tokens) => prompt_tokens,
            Err(_) => {
                let _ = session.request_cancel();
                let _ = session.try_abort();
                return Err(FerrumError::request_validation(
                    "prompt token count exceeds u64",
                ));
            }
        };
        let sequence = Arc::new(VNextSequence {
            cache_id: format!("vnext-cache-{request_id}"),
            request_id: request_id.clone(),
            session,
            active_binding,
            tokens: Mutex::new(tokens),
            maximum_tokens,
            active: AtomicBool::new(true),
            operation: AsyncMutex::new(()),
            events: events.map(Mutex::new),
            prompt_tokens,
            prefill_tokens_processed: AtomicUsize::new(0),
        });

        Ok(VNextPrefillProbeResolution::Ready(sequence))
    }

    fn current_execution_capacity_epochs(&self) -> Result<ExecutorAdmissionEpochs> {
        self.plan_resources
            .dynamic_pool_status()
            .map(|status| ExecutorAdmissionEpochs::from_capacity(status.epochs()))
            .map_err(|error| FerrumError::backend(error.to_string()))
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
        let request = RequestResourceAdmissionRequest::new(
            work.clone(),
            AdmissionFitPolicy::FullInputMustFit,
            AdmissionPressureAction::WaitForRelease,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let sequence = SequenceResourceAdmissionRequest::new(
            work,
            self.policy.admission().sequence_fit_policy,
            AdmissionPressureAction::WaitForRelease,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let sequence = match binding
            .try_admit_initial_sequence(request, sequence, self.run_id.clone(), request_id)
            .map_err(|error| FerrumError::backend(error.to_string()))?
        {
            InitialSequenceResourceAdmissionDecision::Admitted(sequence) => sequence,
            InitialSequenceResourceAdmissionDecision::Deferred(deferred) => {
                self.metrics
                    .sequence_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(VNextSequenceAdmissionDecision::Deferred(deferred));
            }
            InitialSequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                self.metrics
                    .backing_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(VNextSequenceAdmissionDecision::BackingDeferred(
                    VNextPrefillBackingDeferral::InitialSequence(deferred),
                ));
            }
            InitialSequenceResourceAdmissionDecision::PermanentRejected(rejected) => {
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
        active: &Arc<TrustedActiveSequenceBinding>,
    ) -> Result<Option<VNextExecutionJournal>> {
        let Some(sink) = self.event_sink.read().clone() else {
            return Ok(None);
        };
        VNextExecutionJournal::open(
            sink,
            self.resolved_plan.execution_plan(),
            Arc::clone(active),
        )
        .map(Some)
        .map_err(|error| FerrumError::backend(format!("vNext execution journal: {error}")))
    }

    fn execution_maintenance_decision(
        stage: ExecutorExecutionCapacityStage,
        outcome: DynamicDeferredMaintenanceOutcome,
        source: Option<&AdmissionDeferred>,
    ) -> Result<Option<ExecutorExecutionCapacityDeferral>> {
        match outcome {
            DynamicDeferredMaintenanceOutcome::RetryAdmission { .. }
            | DynamicDeferredMaintenanceOutcome::Maintained(_) => Ok(None),
            DynamicDeferredMaintenanceOutcome::WaitForRelease {
                current_epochs,
                wait_condition,
                ..
            } => match source {
                Some(source) => ExecutorExecutionCapacityDeferral::from_maintenance(
                    source,
                    ExecutorAdmissionEpochs::from_capacity(current_epochs),
                    wait_condition,
                    stage,
                ),
                None => ExecutorExecutionCapacityDeferral::new(
                    ExecutorAdmissionEpochs::from_capacity(current_epochs),
                    wait_condition,
                    stage,
                ),
            }
            .map(Some),
        }
    }

    fn execution_capacity_error(deferral: &ExecutorExecutionCapacityDeferral) -> FerrumError {
        FerrumError::resource_exhausted(format!(
            "vNext {:?} is waiting for an exact capacity source change: {:?}",
            deferral.stage(),
            deferral.wait_condition().observed()
        ))
    }

    fn extend_sequence_with_capacity(
        &self,
        sequence: &VNextSequence<R>,
        target: ResourceWorkShape,
    ) -> Result<VNextExecutionCapacityDecision<()>> {
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
                .try_ensure_backing_covers(request)
                .map_err(|error| FerrumError::backend(error.to_string()))?
            {
                SequenceResourceExtensionDecision::Current(_)
                | SequenceResourceExtensionDecision::Extended(_) => {
                    return Ok(VNextExecutionCapacityDecision::Ready(()))
                }
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
                    if deferred.action() == DeferredAction::WaitForRelease {
                        return ExecutorExecutionCapacityDeferral::from_admission(
                            &deferred,
                            ExecutorExecutionCapacityStage::SequenceExtension,
                        )
                        .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    if deferred.action() != DeferredAction::AwaitBackingGrowth {
                        return Err(Self::deferred("sequence extension", &deferred));
                    }
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::deferred("sequence extension", &deferred));
                    }
                    backing_attempts += 1;
                    let outcome = self
                        .plan_resources
                        .maintain_for_admission_deferred(&deferred)
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = Self::execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SequenceExtension,
                        outcome,
                        Some(&deferred),
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                SequenceResourceExtensionDecision::BackingDeferred(deferred) => {
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::backing_deferred(
                            "sequence extension",
                            deferred.evidence(),
                        ));
                    }
                    backing_attempts += 1;
                    let outcome = deferred
                        .maintain()
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = Self::execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SequenceExtension,
                        outcome,
                        None,
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                SequenceResourceExtensionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::request_validation(format!(
                        "vNext sequence extension exceeds the configured fit ceiling: {rejected:?}"
                    )))
                }
            }
        }
    }

    fn extend_sequence(
        &self,
        sequence: &VNextSequence<R>,
        target: ResourceWorkShape,
    ) -> Result<()> {
        match self.extend_sequence_with_capacity(sequence, target)? {
            VNextExecutionCapacityDecision::Ready(()) => Ok(()),
            VNextExecutionCapacityDecision::Deferred(deferred) => {
                Err(Self::execution_capacity_error(&deferred))
            }
        }
    }

    fn try_begin_step_once(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        span: &TokenSpanWork,
    ) -> Result<StepResourceAdmissionDecision<R>> {
        self.try_begin_step_for_spans(
            batch,
            std::slice::from_ref(span),
            VNextExecutionWaveKind::Decode,
        )
    }

    fn try_begin_step_for_spans(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
    ) -> Result<StepResourceAdmissionDecision<R>> {
        let work_shape = batch
            .bind_work_shape(spans.to_vec())
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let reusable_bucket_id = self
            .resolved_plan
            .execution_plan()
            .payload()
            .memory()
            .reusable_execution()
            .and_then(|plan| {
                plan.buckets().iter().find(|resolved| {
                    let bucket = resolved.bucket();
                    bucket.class_id().as_str() == kind.reusable_execution_class()
                        && bucket.capacity().covers(
                            work_shape.immediate_sequences(),
                            work_shape.immediate_tokens(),
                            work_shape.immediate_pages(),
                        )
                })
            })
            .map(|resolved| resolved.bucket().bucket_id().clone());
        let request = StepResourceAdmissionRequest::new(
            work_shape,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let request = match reusable_bucket_id {
            Some(bucket_id) => request.with_reusable_execution_bucket(bucket_id),
            None => request,
        };
        batch
            .try_begin_step(request, &self.lane)
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn begin_step(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        span: &TokenSpanWork,
    ) -> Result<Arc<StepResourceLease<R>>> {
        self.begin_step_for_spans(batch, std::slice::from_ref(span))
    }

    fn begin_step_for_spans(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        spans: &[TokenSpanWork],
    ) -> Result<Arc<StepResourceLease<R>>> {
        match self.begin_step_for_spans_with_capacity(
            batch,
            spans,
            VNextExecutionWaveKind::Decode,
        )? {
            VNextExecutionCapacityDecision::Ready(step) => Ok(step),
            VNextExecutionCapacityDecision::Deferred(deferred) => {
                Err(Self::execution_capacity_error(&deferred))
            }
        }
    }

    fn begin_step_for_spans_with_capacity(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
    ) -> Result<VNextExecutionCapacityDecision<Arc<StepResourceLease<R>>>> {
        let mut backing_attempts = 0;
        loop {
            match self.try_begin_step_for_spans(batch, spans, kind)? {
                StepResourceAdmissionDecision::Admitted(step) => {
                    return Ok(VNextExecutionCapacityDecision::Ready(step))
                }
                StepResourceAdmissionDecision::Deferred(deferred) => {
                    self.metrics.step_deferrals.fetch_add(1, Ordering::Relaxed);
                    if deferred.action() == DeferredAction::WaitForRelease {
                        return ExecutorExecutionCapacityDeferral::from_admission(
                            &deferred,
                            ExecutorExecutionCapacityStage::StepAdmission,
                        )
                        .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    if deferred.action() != DeferredAction::AwaitBackingGrowth {
                        return Err(Self::deferred("step admission", &deferred));
                    }
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::deferred("step admission", &deferred));
                    }
                    backing_attempts += 1;
                    let outcome = self
                        .plan_resources
                        .maintain_for_admission_deferred(&deferred)
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = Self::execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::StepAdmission,
                        outcome,
                        Some(&deferred),
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::backing_deferred(
                            "step admission",
                            deferred.evidence(),
                        ));
                    }
                    backing_attempts += 1;
                    let outcome = deferred
                        .maintain()
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = Self::execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::StepAdmission,
                        outcome,
                        None,
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                StepResourceAdmissionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::backend(format!(
                        "vNext execution step exceeds its immutable plan: {rejected:?}"
                    )))
                }
            }
        }
    }

    fn try_prepare_wave_once(
        &self,
        step: &Arc<StepResourceLease<R>>,
        span: &TokenSpanWork,
    ) -> Result<StepSubmissionWaveAdmissionDecision<R>> {
        self.try_prepare_wave_for_spans(step, std::slice::from_ref(span))
    }

    fn try_prepare_wave_for_spans(
        &self,
        step: &Arc<StepResourceLease<R>>,
        spans: &[TokenSpanWork],
    ) -> Result<StepSubmissionWaveAdmissionDecision<R>> {
        let work_shape = Arc::new(
            step.bind_all_invocation_work_shape(spans.to_vec())
                .map_err(|error| FerrumError::backend(error.to_string()))?,
        );
        let requests = self
            .resolved_plan
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .map(|node| {
                InvocationResourceAdmissionRequest::for_all_step_participants(
                    node.id().clone(),
                    Arc::clone(&work_shape),
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
            })
            .collect::<std::result::Result<Vec<_>, VNextError>>()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        step.try_prepare_submission_wave(requests)
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn prepare_wave(
        &self,
        step: &Arc<StepResourceLease<R>>,
        span: &TokenSpanWork,
    ) -> Result<PreparedStepSubmissionWave<R>> {
        self.prepare_wave_for_spans(step, std::slice::from_ref(span))
    }

    fn prepare_wave_for_spans(
        &self,
        step: &Arc<StepResourceLease<R>>,
        spans: &[TokenSpanWork],
    ) -> Result<PreparedStepSubmissionWave<R>> {
        match self.prepare_wave_for_spans_with_capacity(step, spans)? {
            VNextExecutionCapacityDecision::Ready(wave) => Ok(wave),
            VNextExecutionCapacityDecision::Deferred(deferred) => {
                Err(Self::execution_capacity_error(&deferred))
            }
        }
    }

    fn prepare_wave_for_spans_with_capacity(
        &self,
        step: &Arc<StepResourceLease<R>>,
        spans: &[TokenSpanWork],
    ) -> Result<VNextExecutionCapacityDecision<PreparedStepSubmissionWave<R>>> {
        let mut backing_attempts = 0;
        loop {
            match self.try_prepare_wave_for_spans(step, spans)? {
                StepSubmissionWaveAdmissionDecision::Prepared(wave) => {
                    self.metrics.prepared_wave_topology.record(&wave);
                    return Ok(VNextExecutionCapacityDecision::Ready(wave));
                }
                StepSubmissionWaveAdmissionDecision::Deferred(deferred) => {
                    self.metrics.wave_deferrals.fetch_add(1, Ordering::Relaxed);
                    if deferred.action() == DeferredAction::WaitForRelease {
                        return ExecutorExecutionCapacityDeferral::from_admission(
                            &deferred,
                            ExecutorExecutionCapacityStage::SubmissionWave,
                        )
                        .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    if deferred.action() != DeferredAction::AwaitBackingGrowth {
                        return Err(Self::deferred("submission wave", &deferred));
                    }
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::deferred("submission wave", &deferred));
                    }
                    backing_attempts += 1;
                    let outcome = self
                        .plan_resources
                        .maintain_for_admission_deferred(&deferred)
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = Self::execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SubmissionWave,
                        outcome,
                        Some(&deferred),
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(Self::backing_deferred(
                            "submission wave",
                            deferred.evidence(),
                        ));
                    }
                    backing_attempts += 1;
                    let outcome = deferred
                        .maintain()
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = Self::execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SubmissionWave,
                        outcome,
                        None,
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                StepSubmissionWaveAdmissionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::backend(format!(
                        "vNext submission wave exceeds its immutable plan: {rejected:?}"
                    )))
                }
            }
        }
    }

    fn dispatch_participant_wave(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        wave: PreparedStepSubmissionWave<R>,
        kind: VNextExecutionWaveKind,
    ) -> DispatchOutcome<R> {
        if participants.is_empty() {
            return DispatchOutcome::QuiescentFailure(
                "vNext submission wave requires at least one participant".to_owned(),
            );
        }
        let active_bindings = || {
            participants
                .iter()
                .map(|participant| participant.sequence.active_binding.as_ref())
        };
        let timing_enabled = self.host_dispatch_timing_enabled();
        let phase_timing = self.metrics.wave_timing_for(kind);
        let uploads = match {
            let _timing = self
                .metrics
                .wave_timing
                .token_upload_prepare
                .start_if(timing_enabled);
            let _phase_timing = phase_timing.token_upload_prepare.start_if(timing_enabled);
            participants
                .iter()
                .enumerate()
                .map(|(participant_index, participant)| {
                    let range = participant.span.immediate_token_range();
                    let host_range = Range {
                        start: usize::try_from(range.start).map_err(|_| {
                            FerrumError::backend(
                                "vNext token upload start exceeds host address space",
                            )
                        })?,
                        end: usize::try_from(range.end).map_err(|_| {
                            FerrumError::backend(
                                "vNext token upload end exceeds host address space",
                            )
                        })?,
                    };
                    let host_tokens =
                        participant.tokens.get(host_range.clone()).ok_or_else(|| {
                            FerrumError::backend(format!(
                                "vNext token upload range {host_range:?} exceeds host token length {}",
                                participant.tokens.len()
                            ))
                        })?;
                    let host_bytes = host_tokens
                        .iter()
                        .flat_map(|token| token.to_le_bytes())
                        .collect::<Vec<_>>();
                    let logical_offset_bytes = range
                        .start
                        .checked_mul(ElementType::U32.size_bytes())
                        .ok_or_else(|| {
                            FerrumError::backend("vNext token upload offset overflows u64")
                        })?;
                    let source_layout = HostTransferLayout::new(
                        ElementType::U32,
                        participant.span.immediate_tokens(),
                    )
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
                    let participant_index = u32::try_from(participant_index).map_err(|_| {
                        FerrumError::backend("vNext token upload participant index exceeds u32")
                    })?;
                    SubmissionWaveInputUpload::new(
                        self.io.input_node_id.clone(),
                        participant_index,
                        self.io.input_ordinal,
                        logical_offset_bytes,
                        source_layout,
                        host_bytes,
                    )
                    .map_err(|error| FerrumError::backend(error.to_string()))
                })
                .collect::<Result<Vec<_>>>()
        } {
            Ok(uploads) => uploads,
            Err(error) => return DispatchOutcome::QuiescentFailure(error.to_string()),
        };
        let uploaded_bytes = uploads.iter().fold(0_u64, |total, upload| {
            total.saturating_add(upload.source_layout().byte_len().unwrap_or(0))
        });
        self.metrics
            .uploaded_bytes
            .fetch_add(uploaded_bytes, Ordering::Relaxed);

        let mut wave = wave;
        let mut retries = 0;
        loop {
            let identity = match {
                let _timing = self
                    .metrics
                    .wave_timing
                    .wave_identity_bind
                    .start_if(timing_enabled);
                let _phase_timing = phase_timing.wave_identity_bind.start_if(timing_enabled);
                OperationDispatch::bind_submission_wave_identity(
                    &self.resolved_plan,
                    active_bindings(),
                    &wave,
                    &self.lane,
                )
            } {
                Ok(identity) => identity,
                Err(error) => return DispatchOutcome::QuiescentFailure(error.to_string()),
            };
            let timing_sink = VNextWaveTimingSink {
                aggregate: &self.metrics.wave_timing,
                phase: phase_timing,
            };
            let submission = {
                let _timing = self
                    .metrics
                    .wave_timing
                    .provider_encode_submit
                    .start_if(timing_enabled);
                let _phase_timing = phase_timing.provider_encode_submit.start_if(timing_enabled);
                if timing_enabled {
                    OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
                        self.providers.providers(),
                        &self.resolved_plan,
                        &identity,
                        active_bindings(),
                        self.device_timing_mode(),
                        &uploads,
                        &timing_sink,
                        wave,
                        &self.lane,
                        &self.reaper,
                    )
                } else {
                    OperationDispatch::encode_and_submit_wave_with_inputs(
                        self.providers.providers(),
                        &self.resolved_plan,
                        &identity,
                        active_bindings(),
                        self.device_timing_mode(),
                        &uploads,
                        wave,
                        &self.lane,
                        &self.reaper,
                    )
                }
            };
            match submission {
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
                | Err(error @ SubmissionWaveDispatchError::Initialization(_))
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
        let prepared = {
            let _timing = self.metrics.wave_timing.resource_prepare_attempt.start();
            let _phase_timing = self
                .metrics
                .decode_wave_timing
                .resource_prepare_attempt
                .start();
            let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&sequence.session)])
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            let step = self.begin_step(&batch, &span)?;
            let wave = match self.prepare_wave(&step, &span) {
                Ok(wave) => wave,
                Err(error) => return Err(self.abort_unsubmitted_step(step, error)),
            };
            PreparedVNextPrefill { step, wave }
        };
        self.execute_prepared_step(
            sequence,
            tokens,
            span,
            prepared,
            VNextExecutionWaveKind::Decode,
        )
        .await
    }

    async fn execute_batch_step(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        sequences: &[Arc<VNextSequence<R>>],
        token_batches: &[Vec<u32>],
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
    ) -> Result<VNextExecutionCapacityDecision<Vec<Vec<f32>>>> {
        if sequences.is_empty()
            || sequences.len() != token_batches.len()
            || sequences.len() != spans.len()
            || sequences.len() != batch.sessions().len()
            || batch
                .sessions()
                .iter()
                .zip(sequences)
                .any(|(session, sequence)| !Arc::ptr_eq(session, &sequence.session))
        {
            return Err(FerrumError::internal(
                "vNext decode batch differs from its canonical participant set",
            ));
        }
        let prepared = {
            let _timing = self.metrics.wave_timing.resource_prepare_attempt.start();
            let _phase_timing = self
                .metrics
                .wave_timing_for(kind)
                .resource_prepare_attempt
                .start();
            let step = match self.begin_step_for_spans_with_capacity(batch, spans, kind)? {
                VNextExecutionCapacityDecision::Ready(step) => step,
                VNextExecutionCapacityDecision::Deferred(deferred) => {
                    return Ok(VNextExecutionCapacityDecision::Deferred(deferred))
                }
            };
            let wave = match self.prepare_wave_for_spans_with_capacity(&step, spans)? {
                VNextExecutionCapacityDecision::Ready(wave) => wave,
                VNextExecutionCapacityDecision::Deferred(deferred) => {
                    step.try_rollback_unsubmitted().map_err(|failure| {
                        FerrumError::backend(format!(
                            "vNext capacity-deferred unsubmitted step rollback failed: {}",
                            failure.error()
                        ))
                    })?;
                    return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                }
            };
            PreparedVNextPrefill { step, wave }
        };
        let participants = sequences
            .iter()
            .zip(token_batches)
            .zip(spans)
            .map(|((sequence, tokens), span)| VNextExecutionParticipant {
                sequence,
                tokens,
                span,
            })
            .collect::<Vec<_>>();
        self.execute_prepared_participants(&participants, prepared, kind)
            .await
            .map(VNextExecutionCapacityDecision::Ready)
    }

    async fn execute_prepared_step(
        &self,
        sequence: &Arc<VNextSequence<R>>,
        tokens: &[u32],
        span: TokenSpanWork,
        prepared: PreparedVNextPrefill<R>,
        kind: VNextExecutionWaveKind,
    ) -> Result<Vec<f32>> {
        let participant = VNextExecutionParticipant {
            sequence,
            tokens,
            span: &span,
        };
        let mut logits = self
            .execute_prepared_participants(std::slice::from_ref(&participant), prepared, kind)
            .await?;
        logits
            .pop()
            .ok_or_else(|| FerrumError::internal("vNext single execution returned no logits"))
    }

    async fn execute_prepared_participants(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        prepared: PreparedVNextPrefill<R>,
        kind: VNextExecutionWaveKind,
    ) -> Result<Vec<Vec<f32>>> {
        let _execution_timing = self.metrics.wave_timing.submitted_wave_total.start();
        let phase_timing = self.metrics.wave_timing_for(kind);
        let _phase_execution_timing = phase_timing.submitted_wave_total.start();
        let PreparedVNextPrefill { step, wave } = prepared;
        let dispatch = {
            let _timing = self.metrics.wave_timing.host_encode_submit.start();
            let _phase_timing = phase_timing.host_encode_submit.start();
            self.dispatch_participant_wave(participants, wave, kind)
        };
        let completion = match dispatch {
            DispatchOutcome::Submitted(completion) => completion,
            DispatchOutcome::QuiescentFailure(message) => {
                return Err(self.abort_step(step, message).await)
            }
            DispatchOutcome::SubmissionIndeterminate { message, recovery } => {
                let reaper = Arc::clone(&self.reaper);
                let recovered = self
                    .completion_worker
                    .execute(VNextCompletionTaskKind::IndeterminateRecovery, move || {
                        let recovered = recovery.recover_by_draining_lane();
                        drop(reaper);
                        recovered
                    })
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
                let reaper = Arc::clone(&self.reaper);
                let observed = self
                    .completion_worker
                    .execute(VNextCompletionTaskKind::PostSubmitDrain, move || {
                        let observed = completion.wait();
                        drop(reaper);
                        observed
                    })
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
        let mut execution_event_error = None;
        for participant in participants {
            if let Some(events) = &participant.sequence.events {
                if let Err(error) = events.lock().submitted(completion.receipt()) {
                    execution_event_error.get_or_insert_with(|| error.to_string());
                }
            }
        }

        let readbacks = participants
            .iter()
            .enumerate()
            .map(|(participant_index, _)| {
                CompletionReadbackRequest::new(
                    self.io.output_node_id.clone(),
                    u32::try_from(participant_index).map_err(|_| {
                        FerrumError::backend("vNext readback participant index exceeds u32")
                    })?,
                    self.io.output_resource_id.clone(),
                    self.io.output_offset_bytes,
                    self.io.output_layout,
                )
                .map_err(|error| FerrumError::backend(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let readbacks = CompletionReadbackBatchRequest::new(readbacks)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let reaper = Arc::clone(&self.reaper);
        let observation = {
            let _timing = self.metrics.wave_timing.completion_round_trip.start();
            let _phase_timing = phase_timing.completion_round_trip.start();
            self.completion_worker
                .execute(VNextCompletionTaskKind::WaveReadback, move || {
                    let observation = completion.wait_with_readbacks(readbacks);
                    drop(reaper);
                    observation
                })
                .await
                .map_err(|error| {
                    FerrumError::backend(format!("vNext completion task failed: {error}"))
                })?
                .map_err(|error| FerrumError::backend(error.to_string()))?
        };
        let _postprocess_timing = self.metrics.wave_timing.host_postprocess.start();
        let _phase_postprocess_timing = phase_timing.host_postprocess.start();
        let receipt = match observation {
            CompletionReadbackBatchObservation::Terminal(receipt) => receipt,
            other => {
                self.metrics
                    .record_failure(format!("vNext completion remained nonterminal: {other:?}"));
                return Err(FerrumError::backend(
                    "vNext completion did not reach a quiescent terminal",
                ));
            }
        };
        self.metrics.device_timing.record(&receipt);
        self.metrics.device_timing_for(kind).record(&receipt);
        for participant in participants {
            if let Some(events) = &participant.sequence.events {
                if let Err(error) = events.lock().completed(receipt.completion()) {
                    execution_event_error.get_or_insert_with(|| error.to_string());
                }
            }
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
        let logits = receipt
            .dispositions()
            .iter()
            .map(|disposition| match disposition {
                CompletionReadbackDisposition::Succeeded(output) => {
                    Self::decode_logits(output.bytes(), self.io.output_element_type)
                }
                disposition => Err(FerrumError::backend(format!(
                    "vNext logits readback failed: {disposition:?}"
                ))),
            })
            .collect::<Result<Vec<_>>>();
        let logits = match logits {
            Ok(logits) => logits,
            Err(error) => {
                drop(receipt);
                return Err(self.abort_step(step, error.to_string()).await);
            }
        };
        let readback_bytes = self
            .io
            .output_layout
            .byte_len()
            .unwrap_or(0)
            .saturating_mul(participants.len() as u64);
        self.metrics
            .readback_bytes
            .fetch_add(readback_bytes, Ordering::Relaxed);
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
        Arc::new(VNextKvCacheHandle::new(
            sequence,
            &self.info,
            self.attention_head_dimension,
            tokens,
        ))
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

    fn abort_decode_candidates(&self, candidates: &[VNextDecodeCandidate<R>]) {
        {
            let mut registry = self.sequences.lock();
            for candidate in candidates {
                if registry
                    .active
                    .get(&candidate.cache_id)
                    .is_some_and(|current| Arc::ptr_eq(current, &candidate.sequence))
                {
                    registry.active.remove(&candidate.cache_id);
                }
            }
        }
        for candidate in candidates {
            candidate.sequence.abort();
        }
    }

    async fn execute_decode_batch(
        &self,
        inputs: &[DecodeInput],
    ) -> Result<ExecutorBatchDecodeOutcome> {
        let started = Instant::now();
        if inputs.is_empty() {
            return Ok(ExecutorBatchDecodeOutcome::Completed(Vec::new()));
        }

        let mut candidates = Vec::with_capacity(inputs.len());
        for (original_index, input) in inputs.iter().enumerate() {
            if input.batch_size() != 1 {
                return Err(FerrumError::unsupported(
                    "each vNext batch-decode input must contain exactly one sequence",
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
                    "vNext batch-decode request identity differs from its cache owner",
                ));
            }
            let next = common::tensor_to_tokens(&input.input_ids)?;
            let [next_token] = next.as_slice() else {
                return Err(FerrumError::request_validation(
                    "each vNext batch-decode participant requires exactly one input token",
                ));
            };
            candidates.push(VNextDecodeCandidate {
                original_index,
                sequence,
                cache_id,
                next_token: *next_token,
            });
        }

        let batch = ExecutionBatchParticipants::new(
            candidates
                .iter()
                .map(|candidate| Arc::clone(&candidate.sequence.session))
                .collect(),
        )
        .map_err(|error| FerrumError::request_validation(error.to_string()))?;
        let mut candidates_by_authority = BTreeMap::new();
        for candidate in candidates {
            let authority = candidate.sequence.session.sequence_authority();
            if candidates_by_authority
                .insert(authority, candidate)
                .is_some()
            {
                return Err(FerrumError::request_validation(
                    "vNext batch-decode inputs contain a duplicate sequence",
                ));
            }
        }
        let canonical_candidates = batch
            .sessions()
            .iter()
            .map(|session| {
                candidates_by_authority
                    .remove(&session.sequence_authority())
                    .ok_or_else(|| {
                        FerrumError::internal(
                            "vNext canonical decode participant is absent from its input batch",
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if !candidates_by_authority.is_empty() {
            return Err(FerrumError::internal(
                "vNext decode input is absent from its canonical participant batch",
            ));
        }

        let mut operation_guards = Vec::with_capacity(canonical_candidates.len());
        for candidate in &canonical_candidates {
            operation_guards.push(candidate.sequence.operation.lock().await);
        }

        let mut token_batches = Vec::with_capacity(canonical_candidates.len());
        let mut previous_lengths = Vec::with_capacity(canonical_candidates.len());
        let mut spans = Vec::with_capacity(canonical_candidates.len());
        for candidate in &canonical_candidates {
            if !candidate.sequence.active.load(Ordering::Acquire) {
                return Err(FerrumError::cancelled(format!(
                    "vNext cache `{}` is no longer active",
                    candidate.cache_id
                )));
            }
            let (tokens, previous_len) = {
                let current = candidate.sequence.tokens.lock();
                let previous_len = current.len();
                if previous_len >= candidate.sequence.maximum_tokens {
                    return Err(FerrumError::request_validation(format!(
                        "vNext sequence reached its {} token ceiling",
                        candidate.sequence.maximum_tokens
                    )));
                }
                let mut tokens = current.clone();
                tokens.push(candidate.next_token);
                (tokens, previous_len)
            };
            let extension_span = TokenSpanWork::from_token_ids(&tokens, 0..tokens.len())
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            let extension = ResourceWorkShape::single(extension_span)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            match self.extend_sequence_with_capacity(&candidate.sequence, extension) {
                Ok(VNextExecutionCapacityDecision::Ready(())) => {}
                Ok(VNextExecutionCapacityDecision::Deferred(deferred)) => {
                    return Ok(ExecutorBatchDecodeOutcome::Deferred(deferred));
                }
                Err(error) => {
                    if DecodeFailureDisposition::from_error(&error)
                        == DecodeFailureDisposition::AbortSequence
                    {
                        self.abort_decode_candidates(std::slice::from_ref(candidate));
                    }
                    return Err(error);
                }
            }
            let span = TokenSpanWork::from_token_ids(&tokens, previous_len..tokens.len())
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            token_batches.push(tokens);
            previous_lengths.push(previous_len);
            spans.push(span);
        }

        let sequences = canonical_candidates
            .iter()
            .map(|candidate| Arc::clone(&candidate.sequence))
            .collect::<Vec<_>>();
        let logits = match self
            .execute_batch_step(
                &batch,
                &sequences,
                &token_batches,
                &spans,
                VNextExecutionWaveKind::Decode,
            )
            .await
        {
            Ok(VNextExecutionCapacityDecision::Ready(logits)) => logits,
            Ok(VNextExecutionCapacityDecision::Deferred(deferred)) => {
                return Ok(ExecutorBatchDecodeOutcome::Deferred(deferred));
            }
            Err(error) => {
                if DecodeFailureDisposition::from_error(&error)
                    == DecodeFailureDisposition::AbortSequence
                {
                    self.abort_decode_candidates(&canonical_candidates);
                }
                return Err(error);
            }
        };
        if logits.len() != canonical_candidates.len() {
            self.abort_decode_candidates(&canonical_candidates);
            return Err(FerrumError::internal(format!(
                "vNext batch decode returned {} logits rows for {} participants",
                logits.len(),
                canonical_candidates.len()
            )));
        }

        let logits = match logits
            .into_iter()
            .map(|logits| self.decode_tensor(logits))
            .collect::<Result<Vec<_>>>()
        {
            Ok(logits) => logits,
            Err(error) => {
                self.abort_decode_candidates(&canonical_candidates);
                return Err(error);
            }
        };
        let mut ordered_outputs = (0..inputs.len()).map(|_| None).collect::<Vec<_>>();
        for (((candidate, tokens), previous_len), logits) in canonical_candidates
            .iter()
            .zip(token_batches)
            .zip(previous_lengths)
            .zip(logits)
        {
            if candidate.sequence.active.load(Ordering::Acquire) {
                *candidate.sequence.tokens.lock() = tokens;
            }
            let cache = self.cache_handle(&candidate.sequence, previous_len + 1);
            ordered_outputs[candidate.original_index] = Some(DecodeOutput::new(logits, cache));
        }

        let participant_count = u64::try_from(inputs.len()).unwrap_or(u64::MAX);
        self.metrics
            .decode_operations
            .fetch_add(participant_count, Ordering::Relaxed);
        let elapsed_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        self.metrics.total_decode_us.fetch_add(
            elapsed_us.saturating_mul(participant_count),
            Ordering::Relaxed,
        );
        let outputs = ordered_outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    FerrumError::internal(
                        "vNext batch decode lost the original participant ordering",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ExecutorBatchDecodeOutcome::Completed(outputs))
    }

    async fn execute_prefill_with_capacity(
        &self,
        input: &PrefillInput,
    ) -> Result<ExecutorPrefillOutcome> {
        let started = Instant::now();
        if input.batch_size() != 1 {
            return Err(FerrumError::unsupported(
                "vNext prefill currently requires one sequence per typed submission wave",
            ));
        }
        let request_id = input.request_id.clone().ok_or_else(|| {
            FerrumError::request_validation(
                "plan-runtime vNext prefill requires a typed request_id",
            )
        })?;
        let tokens = common::tensor_to_tokens(&input.input_ids)?;
        let maximum_tokens = input.maximum_sequence_tokens.ok_or_else(|| {
            FerrumError::request_validation(
                "plan-runtime vNext prefill requires maximum_sequence_tokens",
            )
        })?;
        if maximum_tokens < tokens.len() || maximum_tokens > self.maximum_model_tokens {
            return Err(FerrumError::request_validation(format!(
                "request sequence ceiling {maximum_tokens} must cover prompt {} and not exceed {}",
                tokens.len(),
                self.maximum_model_tokens
            )));
        }
        let planned_chunk = match input.chunk {
            Some(chunk) => chunk,
            None => PrefillChunk::new(0, tokens.len(), tokens.len())?,
        };
        if planned_chunk.total_prompt_tokens() != tokens.len() {
            return Err(FerrumError::request_validation(format!(
                "vNext prefill chunk declares {} prompt tokens for input length {}",
                planned_chunk.total_prompt_tokens(),
                tokens.len()
            )));
        }

        let (slot, sequence) = self.sequences.lock().begin_prefill_execution(&request_id)?;
        let mut execution = VNextPrefillExecutionGuard::new(
            &self.sequences,
            Arc::clone(&slot),
            Arc::clone(&sequence),
        );
        let _operation = sequence.operation.lock().await;
        if sequence.maximum_tokens != maximum_tokens || *sequence.tokens.lock() != tokens {
            return Err(FerrumError::request_validation(format!(
                "vNext prefill input for `{request_id}` differs from its admitted work"
            )));
        }
        let processed = sequence.prefill_tokens_processed.load(Ordering::Acquire);
        if processed != planned_chunk.tokens_processed() {
            return Err(FerrumError::request_validation(format!(
                "vNext prefill chunk for `{request_id}` starts at {}, expected {processed}",
                planned_chunk.tokens_processed()
            )));
        }
        if slot.cancelled.load(Ordering::Acquire) {
            return Err(FerrumError::cancelled(format!(
                "vNext prefill for `{request_id}` was cancelled before submission"
            )));
        }

        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&sequence.session)])
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let mut completed_chunk = planned_chunk;
        let mut capacity_probe_count = 0_u32;
        let mut logits = loop {
            let extension_tokens = &tokens[..completed_chunk.end()];
            let extension_span =
                TokenSpanWork::from_token_ids(extension_tokens, 0..extension_tokens.len())
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
            let extension = ResourceWorkShape::single(extension_span)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            if let VNextExecutionCapacityDecision::Deferred(deferred) =
                self.extend_sequence_with_capacity(&sequence, extension)?
            {
                let Some(next_tokens) =
                    deferred.narrower_prefill_tokens(completed_chunk.tokens_to_process())
                else {
                    execution.restore_ready()?;
                    return Ok(ExecutorPrefillOutcome::Deferred(deferred));
                };
                capacity_probe_count = capacity_probe_count.checked_add(1).ok_or_else(|| {
                    FerrumError::internal("vNext prefill capacity probe count overflow")
                })?;
                self.metrics
                    .prefill_frontier_narrowings
                    .fetch_add(1, Ordering::Relaxed);
                completed_chunk = PrefillChunk::new(
                    completed_chunk.tokens_processed(),
                    next_tokens,
                    completed_chunk.total_prompt_tokens(),
                )?;
                continue;
            }

            let span = TokenSpanWork::from_token_ids_with_fit(
                &tokens,
                completed_chunk.range(),
                maximum_tokens,
            )
            .map_err(|error| FerrumError::backend(error.to_string()))?;
            match self
                .execute_batch_step(
                    &batch,
                    std::slice::from_ref(&sequence),
                    std::slice::from_ref(&tokens),
                    std::slice::from_ref(&span),
                    VNextExecutionWaveKind::Prefill,
                )
                .await?
            {
                VNextExecutionCapacityDecision::Ready(logits) => break logits,
                VNextExecutionCapacityDecision::Deferred(deferred) => {
                    let Some(next_tokens) =
                        deferred.narrower_prefill_tokens(completed_chunk.tokens_to_process())
                    else {
                        execution.restore_ready()?;
                        return Ok(ExecutorPrefillOutcome::Deferred(deferred));
                    };
                    capacity_probe_count =
                        capacity_probe_count.checked_add(1).ok_or_else(|| {
                            FerrumError::internal("vNext prefill capacity probe count overflow")
                        })?;
                    self.metrics
                        .prefill_frontier_narrowings
                        .fetch_add(1, Ordering::Relaxed);
                    completed_chunk = PrefillChunk::new(
                        completed_chunk.tokens_processed(),
                        next_tokens,
                        completed_chunk.total_prompt_tokens(),
                    )?;
                }
            }
        };
        let logits = logits.pop().ok_or_else(|| {
            FerrumError::internal("vNext single prefill execution returned no logits")
        })?;
        let logits = self.prefill_tensor(logits)?;
        sequence
            .prefill_tokens_processed
            .store(completed_chunk.end(), Ordering::Release);
        let cache = self.cache_handle(&sequence, completed_chunk.end());
        if completed_chunk.is_final() {
            self.sequences.lock().activate(&slot, &sequence)?;
            execution.disarm();
        } else {
            execution.restore_ready()?;
        }
        self.metrics
            .prefill_operations
            .fetch_add(1, Ordering::Relaxed);
        self.metrics.total_prefill_us.fetch_add(
            started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        Ok(ExecutorPrefillOutcome::Completed(
            ExecutorPrefillCompletion::new(
                PrefillOutput::new(logits, cache),
                planned_chunk,
                completed_chunk,
                capacity_probe_count,
            )?,
        ))
    }

    fn metrics_snapshot(&self) -> serde_json::Value {
        let pool_status = self
            .plan_resources
            .dynamic_pool_status()
            .ok()
            .and_then(|status| serde_json::to_value(status).ok());
        let cleanup = serde_json::to_value(self.plan_resources.deferred_cleanup_status()).ok();
        let (pending_sequences, active_sequences, pending_prefill_maintenance, executing_prefills) = {
            let sequences = self.sequences.lock();
            let mut ready = 0;
            let mut maintenance = 0;
            let mut executing = 0;
            for slot in sequences.prefills.values() {
                match &*slot.state.lock() {
                    VNextPrefillSlotState::Deferred {
                        maintenance: pending,
                        maintaining,
                    } => {
                        if pending.is_some() || *maintaining {
                            maintenance += 1;
                        }
                    }
                    VNextPrefillSlotState::Ready(_) => ready += 1,
                    VNextPrefillSlotState::Executing(_) => executing += 1,
                    VNextPrefillSlotState::Probing | VNextPrefillSlotState::Terminal => {}
                }
            }
            (ready, sequences.active.len(), maintenance, executing)
        };
        serde_json::json!({
            "schema": "ferrum.runtime-vnext.executor-trace.v1",
            "model_id": self.info.model_id.to_string(),
            "family_fingerprint": self.family_fingerprint,
            "program_fingerprint": self.program_fingerprint,
            "resolved_plan_fingerprint": self.resolved_plan.fingerprint(),
            "plan_id": self.resolved_plan.execution_plan().payload().plan_id().to_string(),
            "plan_hash": self.resolved_plan.execution_plan().plan_hash().to_string(),
            "policy_id": self.policy.policy_id(),
            "policy_fingerprint": self.policy.fingerprint_str(),
            "device_id": self.runtime.descriptor().id.to_string(),
            "runtime_fingerprint": self.runtime.descriptor().runtime_implementation_fingerprint,
            "maximum_model_tokens": self.maximum_model_tokens,
            "runtime_memory_policy": self.policy.memory(),
            "runtime_admission_policy": self.policy.admission(),
            "pending_sequences": pending_sequences,
            "active_sequences": active_sequences,
            "staged_prefill_requests": 0,
            "staged_prefill_sequences": 0,
            "pending_prefill_maintenance": pending_prefill_maintenance,
            "executing_prefills": executing_prefills,
            "static_bytes": self.static_bytes,
            "counters": {
                "prefill_operations": self.metrics.prefill_operations.load(Ordering::Relaxed),
                "prefill_frontier_narrowings": self.metrics.prefill_frontier_narrowings.load(Ordering::Relaxed),
                "decode_operations": self.metrics.decode_operations.load(Ordering::Relaxed),
                "prepared_wave_topology": self.metrics.prepared_wave_topology.snapshot(),
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
            "wave_timing": self.metrics.wave_timing.snapshot(),
            "wave_timing_by_phase": {
                VNextExecutionWaveKind::Prefill.as_str(): self.metrics.prefill_wave_timing.snapshot(),
                VNextExecutionWaveKind::Decode.as_str(): self.metrics.decode_wave_timing.snapshot(),
            },
            "device_timing": self.metrics.device_timing.snapshot(),
            "device_timing_by_phase": {
                VNextExecutionWaveKind::Prefill.as_str(): self.metrics.prefill_device_timing.snapshot(),
                VNextExecutionWaveKind::Decode.as_str(): self.metrics.decode_device_timing.snapshot(),
            },
            "completion_worker": self.completion_worker.metrics_snapshot(),
            "dynamic_pools": pool_status,
            "deferred_cleanup": cleanup,
            "startup_preparation": serde_json::to_value(&*self.startup_preparation.lock())
                .unwrap_or_else(|error| serde_json::json!({"state": "serialization_failed", "message": error.to_string()})),
            "last_failure": self.metrics.last_failure.lock().clone(),
        })
    }
}

#[async_trait::async_trait]
impl<R: DeviceRuntime> ModelExecutor for VNextModelExecutor<R> {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prepare_startup(&self) -> Result<()> {
        {
            let mut state = self.startup_preparation.lock();
            match &*state {
                VNextStartupPreparationState::Pending => {
                    *state = VNextStartupPreparationState::Preparing;
                }
                VNextStartupPreparationState::Ready { .. } => return Ok(()),
                VNextStartupPreparationState::Preparing => {
                    return Err(FerrumError::internal(
                        "vNext startup preparation is already running",
                    ));
                }
                VNextStartupPreparationState::Failed { message } => {
                    return Err(FerrumError::device(format!(
                        "vNext startup preparation previously failed: {message}"
                    )));
                }
            }
        }

        let preparation = self
            .prepare_reusable_execution_startup()
            .await
            .and_then(|report| {
                self.reset_request_metrics_after_startup()?;
                Ok(report)
            });
        match preparation {
            Ok(report) => {
                *self.startup_preparation.lock() = VNextStartupPreparationState::Ready { report };
                Ok(())
            }
            Err(error) => {
                *self.startup_preparation.lock() = VNextStartupPreparationState::Failed {
                    message: error.to_string(),
                };
                Err(error)
            }
        }
    }

    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }

    fn resolved_model_plan(&self) -> Option<&ResolvedModelPlan> {
        Some(&self.resolved_plan)
    }

    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        let status = self
            .plan_resources
            .dynamic_pool_status()
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let mut resident_bytes = 0_u64;
        let mut free_bytes = 0_u64;
        let mut pending_growth_bytes = 0_u64;
        let mut quarantined_bytes = 0_u64;
        for pool in status.pools() {
            resident_bytes = resident_bytes
                .checked_add(pool.resident_bytes())
                .ok_or_else(|| FerrumError::internal("dynamic resident bytes overflow u64"))?;
            free_bytes = free_bytes
                .checked_add(pool.free_bytes())
                .ok_or_else(|| FerrumError::internal("dynamic free bytes overflow u64"))?;
            pending_growth_bytes = pending_growth_bytes
                .checked_add(pool.pending_growth_bytes())
                .ok_or_else(|| FerrumError::internal("pending growth bytes overflow u64"))?;
            quarantined_bytes = quarantined_bytes
                .checked_add(pool.quarantined_bytes())
                .ok_or_else(|| FerrumError::internal("quarantined bytes overflow u64"))?;
        }
        PlanRuntimeResourceSnapshot::new(
            status.device_capacity_bytes(),
            status.effective_device_usable_ceiling_bytes(),
            status.process_claimed_bytes(),
            status.budget_claimed_bytes(),
            self.static_bytes,
            resident_bytes,
            free_bytes,
            pending_growth_bytes,
            quarantined_bytes,
        )
        .map(Some)
    }

    fn kv_capacity(&self) -> Option<usize> {
        Some(self.maximum_model_tokens)
    }

    fn attach_execution_event_sink(&self, sink: Arc<dyn ExecutionEventSink>) {
        self.device_timing_mode
            .store(sink.device_timing_mode() as u8, Ordering::Release);
        *self.event_sink.write() = Some(sink);
    }

    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        self.plan_resources
            .dynamic_pool_status()
            .map(|status| Some(ExecutorAdmissionEpochs::from_capacity(status.epochs())))
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<ferrum_interfaces::vnext::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        self.plan_resources
            .write_dynamic_capacity_availability(availability)
            .map(|epochs| Some(ExecutorAdmissionEpochs::from_capacity(epochs)))
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn register_execution_capacity_waiter(
        &self,
        observed: &CapacityWaitCondition,
    ) -> Result<Option<ExecutorCapacityWaitRegistration>> {
        let registration = self
            .plan_resources
            .register_capacity_waiter(observed)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        Ok(Some(ExecutorCapacityWaitRegistration::new(async move {
            registration
                .wait_for_change()
                .await
                .map(ExecutorAdmissionEpochs::from_capacity)
                .map_err(|error| FerrumError::backend(error.to_string()))
        })))
    }

    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        if input.input_tokens.is_empty() {
            return Err(FerrumError::request_validation(
                "plan-runtime vNext prefill admission requires at least one input token",
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
        let tokens = input
            .input_tokens
            .iter()
            .map(|token| token.get())
            .collect::<Vec<_>>();
        let span =
            TokenSpanWork::from_token_ids_with_fit(&tokens, 0..1, input.maximum_sequence_tokens)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
        let work = ResourceWorkShape::single(span.clone())
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let slot = self
            .sequences
            .lock()
            .begin_prefill_probe(input.request_id, &work)?;
        let resolution = match self.resolve_prefill_probe(
            input.request_id,
            input.maximum_sequence_tokens,
            tokens,
            work,
        ) {
            Ok(resolution) => resolution,
            Err(error) => {
                self.fail_prefill_probe(&slot);
                return Err(error);
            }
        };
        self.publish_prefill_probe(&slot, resolution)
    }

    fn cancel_prefill_admission(&self, request_id: &RequestId) -> bool {
        self.sequences.lock().cancel_prefill(request_id)
    }

    fn write_execution_capacity_release_sources(
        &self,
        preemption: &ExecutorExecutionCapacityPreemption,
        sources: &mut Vec<CapacityAvailabilitySource>,
    ) -> Result<bool> {
        self.sequences
            .lock()
            .write_execution_capacity_release_sources(preemption, sources)
    }

    async fn preempt_execution_capacity(
        &self,
        preemption: ExecutorExecutionCapacityPreemption,
    ) -> Result<ExecutorExecutionCapacityPreemptionReceipt> {
        let authority = self
            .sequences
            .lock()
            .preempt_execution_capacity(&preemption)?;
        Ok(ExecutorExecutionCapacityPreemptionReceipt::new(
            preemption.request_id().clone(),
            preemption.cache_id().to_string(),
            authority,
        ))
    }

    fn maintain_prefill_backing(
        &self,
        request_id: &RequestId,
    ) -> Result<ExecutorPrefillMaintenanceOutcome> {
        let (slot, pending) = {
            let sequences = self.sequences.lock();
            let Some(slot) = sequences.prefills.get(request_id).cloned() else {
                return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending);
            };
            let mut state = slot.state.lock();
            let pending = match &mut *state {
                VNextPrefillSlotState::Deferred {
                    maintenance,
                    maintaining,
                    ..
                } if !*maintaining => {
                    let Some(pending) = maintenance.take() else {
                        return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending);
                    };
                    *maintaining = true;
                    pending
                }
                _ => return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending),
            };
            drop(state);
            (slot, pending)
        };

        let outcome = match &pending {
            PendingPrefillMaintenance::Logical(deferred) => self
                .plan_resources
                .maintain_for_admission_deferred(deferred),
            PendingPrefillMaintenance::Backing(deferred) => deferred.maintain(),
        };
        drop(pending);

        let mut sequences = self.sequences.lock();
        let current = sequences
            .prefills
            .get(request_id)
            .is_some_and(|current| Arc::ptr_eq(current, &slot));
        if !current {
            return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending);
        }
        let mut state = slot.state.lock();
        let cancelled = slot.cancelled.load(Ordering::Acquire);
        let owns_maintenance = matches!(
            &*state,
            VNextPrefillSlotState::Deferred {
                maintenance: None,
                maintaining: true,
                ..
            }
        );
        if cancelled || outcome.is_err() || !owns_maintenance {
            let prior = std::mem::replace(&mut *state, VNextPrefillSlotState::Terminal);
            drop(state);
            sequences.prefills.remove(request_id);
            drop(sequences);
            prior.abort();
            if cancelled {
                return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending);
            }
            return match outcome {
                Err(error) => Err(FerrumError::backend(error.to_string())),
                Ok(_) => Err(FerrumError::internal(format!(
                    "vNext prefill maintenance for `{request_id}` lost its slot state"
                ))),
            };
        }
        let VNextPrefillSlotState::Deferred { maintaining, .. } = &mut *state else {
            unreachable!("maintenance ownership was checked")
        };
        *maintaining = false;
        drop(state);
        drop(sequences);
        let outcome = outcome.expect("maintenance error was handled above");
        match outcome {
            DynamicDeferredMaintenanceOutcome::RetryAdmission { current_epochs } => {
                Ok(ExecutorPrefillMaintenanceOutcome::RetryAdmission {
                    current: ExecutorAdmissionEpochs::from_capacity(current_epochs),
                })
            }
            DynamicDeferredMaintenanceOutcome::WaitForRelease {
                current_epochs,
                wait_condition,
                pressure,
            } => Ok(ExecutorPrefillMaintenanceOutcome::WaitForRelease {
                current: ExecutorAdmissionEpochs::from_capacity(current_epochs),
                wait_condition,
                pressure,
            }),
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
                let rebalance = receipt.rebalance().cloned();
                let (pools_reclaimed, chunks_reclaimed, reclaimed_bytes) =
                    rebalance.as_ref().map_or((0, 0, 0), |rebalance| {
                        (
                            rebalance.pools().len(),
                            rebalance.reclaimed_chunks(),
                            rebalance.reclaimed_bytes(),
                        )
                    });
                Ok(ExecutorPrefillMaintenanceOutcome::Maintained {
                    current: self.current_execution_capacity_epochs()?,
                    pools_grown: receipt.growths().len(),
                    allocated_bytes,
                    pools_reclaimed,
                    chunks_reclaimed,
                    reclaimed_bytes,
                    rebalance,
                })
            }
        }
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        match self.execute_prefill_with_capacity(input).await? {
            ExecutorPrefillOutcome::Completed(completion) => {
                let (output, _, _, _) = completion.into_parts();
                Ok(output)
            }
            ExecutorPrefillOutcome::Deferred(deferred) => {
                Err(Self::execution_capacity_error(&deferred))
            }
        }
    }

    async fn prefill_with_capacity(&self, input: &PrefillInput) -> Result<ExecutorPrefillOutcome> {
        self.execute_prefill_with_capacity(input).await
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
        match self.execute_decode_batch(inputs).await? {
            ExecutorBatchDecodeOutcome::Completed(outputs) => Ok(outputs),
            ExecutorBatchDecodeOutcome::Deferred(deferred) => {
                Err(Self::execution_capacity_error(&deferred))
            }
        }
    }

    async fn batch_decode_with_capacity(
        &self,
        inputs: &[DecodeInput],
    ) -> Result<ExecutorBatchDecodeOutcome> {
        self.execute_decode_batch(inputs).await
    }

    fn release_cache(&self, cache_id: &str) {
        if let Some(sequence) = self.sequences.lock().active.remove(cache_id) {
            sequence.abort();
        }
    }

    fn complete_cache(&self, completion: ExecutorSequenceCompletion) -> Result<()> {
        let sequence = self
            .sequences
            .lock()
            .active
            .remove(completion.cache_id())
            .ok_or_else(|| {
                FerrumError::not_found(format!(
                    "vNext completion cache `{}` is not active",
                    completion.cache_id()
                ))
            })?;
        sequence.complete(&completion)
    }

    fn capabilities(&self) -> ExecutorCapabilities {
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
                    * self.attention_head_dimension
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
        let startup = self.startup_preparation.lock();
        let startup_ready = startup.is_ready();
        let startup_failed = matches!(&*startup, VNextStartupPreparationState::Failed { .. });
        drop(startup);
        ExecutorStatus {
            state: if startup_failed {
                ExecutorState::Error
            } else if !startup_ready {
                ExecutorState::Initializing
            } else if self.sequences.lock().total_len() == 0 {
                ExecutorState::Ready
            } else {
                ExecutorState::Busy
            },
            is_ready: startup_ready && !self.plan_resources.is_closing(),
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
    use super::{
        reported_allocated_bytes, resolved_sequence_fit_policy, AdmissionFitPolicy,
        DecodeFailureDisposition, FerrumError, SequenceFitPolicy, VNextDeviceTimingMetrics,
        VNextExecutionWaveKind, VNextPreparedWaveTopologyMetrics, VNextReusableExecutionDescriptor,
        VNextReusableExecutionMetrics, VNextReusableExecutionStartupPlan, VNextWaveTimingMetrics,
        VNextWaveTimingSink,
    };
    use ferrum_interfaces::vnext::{
        DeviceReusableExecutionObservation, DeviceSubmissionTimingSink,
    };

    #[test]
    fn reusable_execution_metrics_aggregate_typed_preparation_outcomes() {
        let mut observation = DeviceReusableExecutionObservation::default();
        observation.observe_candidate_segment();
        observation.observe_captured_segment();
        observation.observe_uploaded_segment();
        observation.observe_cache_hit_segment();
        observation.observe_cached_rejected_segment();
        observation.observe_capture_rejection();
        observation.observe_quiescence_deferred_segment();
        observation.observe_capacity_deferred_segment();
        observation.observe_outside_preparation_segment();
        observation.observe_evicted_segment();
        observation.observe_replayed_segment(4);
        observation.observe_eager_command();

        let metrics = VNextReusableExecutionMetrics::default();
        metrics.record(observation);
        metrics.record(observation);
        let snapshot = metrics.snapshot();

        for field in [
            "candidate_segments",
            "captured_segments",
            "uploaded_segments",
            "cache_hit_segments",
            "cached_rejected_segments",
            "capture_rejected_segments",
            "quiescence_deferred_segments",
            "capacity_deferred_segments",
            "outside_preparation_segments",
            "evicted_segments",
            "replayed_segments",
            "eager_commands",
        ] {
            assert_eq!(snapshot[field], 2, "counter {field} must aggregate");
        }
        assert_eq!(snapshot["replayed_commands"], 8);
    }

    #[test]
    fn wave_timing_sink_attributes_replay_to_aggregate_and_exact_phase() {
        let aggregate = VNextWaveTimingMetrics::default();
        let decode = VNextWaveTimingMetrics::default();
        let sink = VNextWaveTimingSink {
            aggregate: &aggregate,
            phase: &decode,
        };
        let mut observation = DeviceReusableExecutionObservation::default();
        observation.observe_candidate_segment();
        observation.observe_replayed_segment(3);

        sink.record_reusable_execution(observation);

        assert_eq!(
            aggregate.reusable_execution.snapshot()["candidate_segments"],
            1
        );
        assert_eq!(
            decode.reusable_execution.snapshot()["candidate_segments"],
            1
        );
        assert_eq!(VNextExecutionWaveKind::Prefill.as_str(), "prefill");
        assert_eq!(VNextExecutionWaveKind::Decode.as_str(), "decode");
    }

    #[test]
    fn prepared_wave_topology_metrics_separate_owners_from_node_projections() {
        let metrics = VNextPreparedWaveTopologyMetrics::default();

        metrics.record_counts(131, 1, 131, 1);
        metrics.record_counts(131, 4, 524, 1);
        let snapshot = metrics.snapshot();

        assert_eq!(snapshot["wave_authorities"], 2);
        assert_eq!(snapshot["covered_nodes"], 262);
        assert_eq!(snapshot["participant_flights"], 5);
        assert_eq!(snapshot["node_participant_projections"], 655);
        assert_eq!(snapshot["physical_ledger_entries"], 2);

        metrics.reset();
        assert_eq!(metrics.snapshot()["wave_authorities"], 0);
    }

    #[test]
    fn reusable_execution_startup_plan_is_policy_derived_largest_first_and_bounded() {
        let plan =
            VNextReusableExecutionStartupPlan::resolve(32, 2_048, 128, &[64, 64], 23).unwrap();

        assert_eq!(plan.decode_widths(), [32, 16, 8, 4, 2, 1]);
        assert_eq!(plan.prefill_token_counts(), [64]);
        assert_eq!(plan.maximum_decode_sequence_tokens, 19);
        assert_eq!(plan.device_plan.maximum_executables(), 161);
        assert_eq!(
            plan.descriptors.last(),
            Some(&VNextReusableExecutionDescriptor::prefill(64))
        );
        assert_eq!(
            plan.widths_for_available_sequences(20),
            [20, 16, 8, 4, 2, 1]
        );

        let non_power_of_two =
            VNextReusableExecutionStartupPlan::resolve(7, 7, 64, &[64, 4], 2).unwrap();
        assert_eq!(non_power_of_two.decode_widths(), [7, 4, 2, 1]);
        assert_eq!(non_power_of_two.prefill_token_counts(), [7, 4]);
        assert_eq!(non_power_of_two.device_plan.maximum_executables(), 12);
        assert!(VNextReusableExecutionStartupPlan::resolve(32, 2_048, 18, &[64], 23).is_err());
    }

    #[test]
    fn wave_timing_snapshot_exposes_honest_host_boundaries() {
        let snapshot = VNextWaveTimingMetrics::default().snapshot();

        assert_eq!(snapshot["clock"], "host_monotonic");
        assert_eq!(snapshot["resource_prepare_attempt"]["samples"], 0);
        assert_eq!(snapshot["submitted_wave_total"]["samples"], 0);
        assert_eq!(
            snapshot["host_encode_submit_breakdown"]["collection"],
            "profile_attached_only"
        );
        assert_eq!(
            snapshot["host_encode_submit_breakdown"]["wave_identity_bind"]["samples"],
            0
        );
        assert_eq!(
            snapshot["host_encode_submit_breakdown"]["provider_encode_submit_breakdown"]
                ["provider_node_encode"]["samples"],
            0
        );
        assert_eq!(
            snapshot["host_encode_submit_breakdown"]["provider_encode_submit_breakdown"]
                ["lane_reserve_submit_arm_breakdown"]["device_runtime_submit"]["samples"],
            0
        );
        assert_eq!(
            snapshot["host_encode_submit_breakdown"]["provider_encode_submit_breakdown"]
                ["lane_reserve_submit_arm_breakdown"]["device_runtime_submit_breakdown"]
                ["enqueue_commands"]["samples"],
            0
        );
        assert_eq!(
            snapshot["host_encode_submit_breakdown"]["provider_encode_submit_breakdown"]
                ["lane_reserve_submit_arm_breakdown"]["device_runtime_submit_breakdown"]
                ["reusable_execution"]["replayed_commands"],
            0
        );
        assert!(snapshot["limitations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry.as_str().unwrap().contains("not kernel")));
    }

    #[test]
    fn device_timing_snapshot_distinguishes_device_and_host_clocks() {
        let snapshot = VNextDeviceTimingMetrics::default().snapshot();

        assert_eq!(snapshot["device_execution"]["samples"], 0);
        assert_eq!(snapshot["fence_wait_host"]["samples"], 0);
        assert_eq!(snapshot["readback_host"]["samples"], 0);
        assert_eq!(
            snapshot["clocks"]["device_execution"],
            "backend_device_event_elapsed"
        );
        assert!(snapshot["limitations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry.as_str().unwrap().contains("must not be added")));
    }

    #[test]
    fn product_sequence_fit_policy_maps_exhaustively_to_runtime_contract() {
        assert_eq!(
            resolved_sequence_fit_policy(SequenceFitPolicy::ImmediateOnly),
            AdmissionFitPolicy::ImmediateOnly
        );
        assert_eq!(
            resolved_sequence_fit_policy(SequenceFitPolicy::FullInputMustFit),
            AdmissionFitPolicy::FullInputMustFit
        );
    }

    #[test]
    fn allocated_memory_does_not_count_static_claim_twice() {
        assert_eq!(reported_allocated_bytes(Some(64), 64), 64);
        assert_eq!(reported_allocated_bytes(None, 64), 64);
    }

    #[test]
    fn decode_capacity_deferral_preserves_plan_runtime_sequence() {
        let error = FerrumError::resource_exhausted("dynamic pool is waiting for release");

        assert_eq!(
            DecodeFailureDisposition::from_error(&error),
            DecodeFailureDisposition::PreserveForCapacityRetry
        );
    }

    #[test]
    fn decode_permanent_failure_aborts_plan_runtime_sequence() {
        let error = FerrumError::request_validation("sequence exceeds its configured ceiling");

        assert_eq!(
            DecodeFailureDisposition::from_error(&error),
            DecodeFailureDisposition::AbortSequence
        );
    }
}
