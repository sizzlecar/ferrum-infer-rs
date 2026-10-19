//! Shared production executor for immutable model programs.
//!
//! Model families provide semantics and weights; backend compositions provide
//! typed operation providers and a device runtime. This executor owns the
//! remaining product lifecycle: compile once, initialize once, dynamically
//! admit exact live work, submit immutable-plan waves, and retire resources
//! only after a terminal fence.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::model_executor::{
    AttentionType, ExecutionResourceAuthority, ExecutorAdmissionEpochs, ExecutorCapabilities,
    ExecutorCapacityWaitRegistration, ExecutorExecutionCapacityDeferral,
    ExecutorExecutionCapacityPreemption, ExecutorExecutionCapacityPreemptionAuthority,
    ExecutorExecutionCapacityPreemptionReceipt, ExecutorExecutionCapacityStage,
    ExecutorExecutionDeferral, ExecutorMemoryUsage, ExecutorPrefillAdmission,
    ExecutorPrefillAdmissionDecision, ExecutorPrefillAdmissionReceipt,
    ExecutorPrefillMaintenanceDeferral, ExecutorPrefillMaintenanceOutcome, ExecutorRequestOrigin,
    ExecutorRequestStateDeferral, ExecutorSamplingOutput, ExecutorSequenceCompletion,
    ExecutorState, ExecutorStatus, LogitsReturnPolicy, MemoryRequirements,
    PlanRuntimePrefixRestoreDeferral, PlanRuntimePrefixRestoreInput,
    PlanRuntimePrefixRestoreOutcome, PlanRuntimePrefixRestoreOutput, PlanRuntimeResourceSnapshot,
    PrefillChunk, PrefixCaptureBoundary, PrefixCaptureLease, PrefixCapturePlan,
    PrefixCaptureRequest, TypedSequenceStateMemory,
};
use crate::prepared_model::PreparedExecutionInput;
use crate::token_execution::{
    ExecutedTokenWork, ExecutionAdmission, ExecutionBatchOutcome, ExecutionFrontier,
    ExecutionInterruptionCause, ExecutionRecoveryKey, ExecutionWork, ExecutorSequence,
    ExecutorSequenceIssuer, TokenAdmissionOutcome, TokenWork,
};
use crate::{ModelExecutor, ProductExecutionEventSink};
use ferrum_runtime::*;
use ferrum_types::{
    AttentionExecutionPolicy, EngineConfig, ExecutorAdmissionLimits, FerrumError,
    InvocationPreparationStrategy, ModelInfo, ObservabilityProfileDetail, ProfileEntrypoint,
    RequestId, Result, ReusableExecutionCaptureConfig, ReusableExecutionPreparationMode,
    SamplingReadbackMode, SchedulingPolicy, SequenceFitPolicy, TokenId, VNextDiagnosticFault,
    MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH,
};
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use tokio::sync::Mutex as AsyncMutex;

mod checkpoint;
mod completion_worker;
mod timing;

use self::{
    checkpoint::{
        VNextCheckpointArtifactRecord, VNextCheckpointCapture, VNextCheckpointClaim,
        VNextCheckpointProductOutputMode, VNextCheckpointProductOutputRecord,
        VNextCheckpointSelection, VNextTeacherForcedDecision,
    },
    completion_worker::{VNextCompletionTaskKind, VNextCompletionWorker},
    timing::{log_static_initialization_receipt, AtomicDurationMetrics, StartupPhaseTimer},
};

mod backing_maintenance;
mod composition;
mod determinism;
mod input_uploads;
mod invocation_preparation;
mod resident_decode;
pub use composition::{CompiledProgram, ProgramCompilationError, RuntimeComposition};
use resident_decode::{
    PreparedDispatch, PreparedDispatchResult, PreparedRetirement, ResidentCache, ResidentMetrics,
};
mod prefix_cache;
mod program_binding_upload;
mod request;
mod reusable_catalog;
mod reusable_cold_path;
mod sampling_readback;
mod sequence_settlement;
mod shutdown;
mod terminal_facts;
use sequence_settlement::{PreparedWaveGuard, RetiredWave, SequenceSettlement, WaveFailure};
use terminal_facts::PendingSubmissionObservation;
mod state_memory;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
mod token_execution;
pub use determinism::{
    DeterminismExecutionMode, DeterminismExecutionSpec, DeterminismInitialState,
    DeterminismParticipantSpec, DeterminismPhase, DeterminismWorkspacePoison,
    MAX_DETERMINISM_PARTICIPANTS,
};
use request::{terminalize_unsubmitted_session, VNextRequestRoot};
use sampling_readback::ProductReadbackModes;

const POLICY_ID: &str = "policy.ferrum.product.vnext.default";
const POLICY_VERSION: ContractVersion = ContractVersion::new(3, 0);
const UNIFORM_QUERY_REUSABLE_CLASS: &str = "execution.uniform-query-token";
const PACKED_TOKEN_REUSABLE_CLASS: &str = "execution.single-sequence-packed-token";
const DEFAULT_STATIC_STAGING_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_STATIC_COMMANDS_PER_BATCH: usize = 64;
const DEFAULT_CANCELLATION_CHECK_INTERVAL_STEPS: u32 = 1;
const MAX_DEFINITELY_NOT_SUBMITTED_RETRIES: u32 = 1;
const MAX_BACKING_MAINTENANCE_ATTEMPTS: u32 = 2;
const MAX_EXTENSION_RECHECKS: u32 = 2;
const MAX_PROFILED_REUSABLE_EXECUTABLES: usize = 256;
const MAX_PRODUCT_TOKEN_MASK_SLOT_CACHE_ENTRIES: usize = 1_024;
const REUSABLE_EXECUTION_WARMUP_PASSES: u32 = 1;
const REUSABLE_EXECUTION_CAPTURE_PASSES: u32 = 1;
const REUSABLE_EXECUTION_REPLAY_VALIDATION_PASSES: u32 = 1;
type VNextDriver<R> = RuntimeResourceDriver<R>;

const fn submission_execution_policy_for_timing(
    timing_mode: DeviceTimingMode,
) -> SubmissionExecutionPolicy {
    match timing_mode {
        DeviceTimingMode::Verification => SubmissionExecutionPolicy::determinism_eager(0),
        DeviceTimingMode::Off
        | DeviceTimingMode::Completion
        | DeviceTimingMode::Replay
        | DeviceTimingMode::Kernel => SubmissionExecutionPolicy::adaptive(),
    }
}

const fn reusable_catalog_lookup_allowed(
    has_startup_plan: bool,
    direct_reusable_execution_allowed: bool,
    direct_reusable_execution_already_attempted: bool,
) -> bool {
    has_startup_plan
        && direct_reusable_execution_allowed
        && !direct_reusable_execution_already_attempted
}

const fn reusable_program_identity_required(
    has_startup_plan: bool,
    catalog_installed: bool,
    direct_reusable_execution_allowed: bool,
    direct_reusable_execution_already_attempted: bool,
) -> bool {
    has_startup_plan
        && (!catalog_installed
            || reusable_catalog_lookup_allowed(
                true,
                direct_reusable_execution_allowed,
                direct_reusable_execution_already_attempted,
            ))
}

const fn resolved_sequence_fit_policy(policy: SequenceFitPolicy) -> AdmissionFitPolicy {
    match policy {
        SequenceFitPolicy::FullInputMustFit => AdmissionFitPolicy::FullInputMustFit,
        SequenceFitPolicy::ImmediateOnly => AdmissionFitPolicy::ImmediateOnly,
    }
}

fn resolved_sequence_backing_growth(
    policy: ferrum_types::SequenceBackingGrowth,
) -> SequenceBackingGrowthPolicy {
    match policy {
        ferrum_types::SequenceBackingGrowth::Exact => SequenceBackingGrowthPolicy::Exact,
        ferrum_types::SequenceBackingGrowth::PagedGroup4 => {
            SequenceBackingGrowthPolicy::PagedGroup(
                std::num::NonZeroU32::new(4).expect("four pages is a positive reservation group"),
            )
        }
        ferrum_types::SequenceBackingGrowth::PagedGroup4GrowOnly => {
            SequenceBackingGrowthPolicy::PagedGroupGrowOnly(
                std::num::NonZeroU32::new(4).expect("four pages is a positive reservation group"),
            )
        }
    }
}

fn resolve_runtime_attention_authority(
    requested: AttentionExecutionPolicy,
    native_adaptive_supported: bool,
    installed: AttentionExecutionPolicy,
) -> Result<AttentionExecutionPolicy> {
    let requested = requested
        .resolve(native_adaptive_supported)
        .map_err(|reason| {
            FerrumError::config(format!(
                "invalid vNext attention execution policy: {reason}"
            ))
        })?;
    if !installed.is_resolved() {
        return Err(FerrumError::config(
            "vNext runtime exposed unresolved auto attention policy",
        ));
    }
    if requested != installed {
        return Err(FerrumError::config(format!(
            "vNext attention policy authority mismatch: product configuration resolves to {}, but the runtime composition installed {}",
            requested.as_runtime_value(),
            installed.as_runtime_value(),
        )));
    }
    Ok(installed)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum VNextReusableExecutionDecodeWidthSource {
    Automatic,
    Explicit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct VNextReusableExecutionCaptureResolution {
    source: VNextReusableExecutionDecodeWidthSource,
    admission_maximum_decode_width: usize,
    requested_decode_widths: Vec<usize>,
    effective_decode_widths: Vec<usize>,
    reduction_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct VNextReusableExecutionPolicyResolution {
    policy: ReusableExecutionPolicy,
    capture: Option<VNextReusableExecutionCaptureResolution>,
}

fn reusable_execution_anchor_decode_widths(maximum_width: usize) -> Vec<usize> {
    let mut widths = Vec::new();
    let mut width = 1_usize;
    while width < maximum_width {
        widths.push(width);
        width = width
            .checked_mul(2)
            .unwrap_or(maximum_width)
            .min(maximum_width);
    }
    if widths.last().copied() != Some(maximum_width) {
        widths.push(maximum_width);
    }
    widths
}

fn reusable_execution_maximum_decode_sequence_tokens(decode_width_count: usize) -> Result<usize> {
    let passes_per_width = usize::try_from(
        REUSABLE_EXECUTION_WARMUP_PASSES
            .checked_add(REUSABLE_EXECUTION_CAPTURE_PASSES)
            .and_then(|passes| passes.checked_add(REUSABLE_EXECUTION_REPLAY_VALIDATION_PASSES))
            .ok_or_else(|| FerrumError::config("vNext reusable startup pass count overflowed"))?,
    )
    .map_err(|_| FerrumError::config("vNext reusable startup pass count exceeds usize"))?;
    decode_width_count
        .checked_mul(passes_per_width)
        .and_then(|decode_tokens| decode_tokens.checked_add(1))
        .ok_or_else(|| FerrumError::config("vNext reusable startup token ceiling overflowed"))
}

fn resolve_reusable_execution_policy(
    maximum_active_sequences: u32,
    maximum_scheduled_tokens: u64,
    maximum_model_tokens: usize,
    prefill_chunks: &[PrefillChunk],
    capture_config: &ReusableExecutionCaptureConfig,
    prepare_device_programs: bool,
) -> Result<VNextReusableExecutionPolicyResolution> {
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
    for width in reusable_execution_anchor_decode_widths(maximum_width) {
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
    }

    let mut prefill_token_counts = prefill_chunks
        .iter()
        .map(|chunk| chunk.tokens_to_process())
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
    let mut policy = ReusableExecutionPolicy::new(1, buckets)
        .map_err(|error| FerrumError::config(error.to_string()))?;
    if !prepare_device_programs {
        return Ok(VNextReusableExecutionPolicyResolution {
            policy,
            capture: None,
        });
    }

    if capture_config.maximum_automatic_exact_decode_width == 0
        || capture_config.maximum_automatic_exact_decode_width
            > MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH
    {
        return Err(FerrumError::config(format!(
            "runtime.reusable_execution_max_automatic_exact_decode_width must be within 1..={MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH}"
        )));
    }
    let mut resolution_reasons = Vec::new();
    let (source, mut requested_decode_widths) = match &capture_config.exact_decode_widths {
        Some(widths) => {
            if widths.is_empty() {
                return Err(FerrumError::config(
                    "runtime.reusable_execution_exact_decode_widths must not be empty",
                ));
            }
            (
                VNextReusableExecutionDecodeWidthSource::Explicit,
                widths.clone(),
            )
        }
        None => {
            let automatic_maximum_width = maximum_width
                .min(capture_config.maximum_automatic_exact_decode_width)
                .min(MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH);
            if automatic_maximum_width < maximum_width {
                resolution_reasons.push(format!(
                    "automatic startup capture is bounded at width {automatic_maximum_width}; admitted widths {}..={maximum_width} use eager fallback",
                    automatic_maximum_width + 1
                ));
            }
            (
                VNextReusableExecutionDecodeWidthSource::Automatic,
                (1..=automatic_maximum_width).collect(),
            )
        }
    };
    requested_decode_widths.sort_unstable();
    requested_decode_widths.dedup();
    if requested_decode_widths.iter().any(|width| {
        *width == 0
            || *width > maximum_width
            || *width > MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH
    }) {
        return Err(FerrumError::config(format!(
            "vNext reusable exact decode widths must be within 1..={} and may not exceed the independent startup capture hard bound {MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH}",
            maximum_width.min(MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH)
        )));
    }

    let on_demand = capture_config.preparation == ReusableExecutionPreparationMode::OnDemand;
    let requested_sequence_tokens = if on_demand {
        0
    } else {
        reusable_execution_maximum_decode_sequence_tokens(requested_decode_widths.len())?
    };
    let (effective_decode_widths, budget_reduction_reason) = if requested_sequence_tokens
        <= maximum_model_tokens
    {
        (requested_decode_widths.clone(), None)
    } else if source == VNextReusableExecutionDecodeWidthSource::Automatic {
        let requested_maximum_width = requested_decode_widths.last().copied().ok_or_else(|| {
            FerrumError::config("vNext automatic reusable decode matrix is empty")
        })?;
        let anchors = reusable_execution_anchor_decode_widths(requested_maximum_width);
        let anchor_sequence_tokens =
            reusable_execution_maximum_decode_sequence_tokens(anchors.len())?;
        if anchor_sequence_tokens > maximum_model_tokens {
            return Err(FerrumError::config(format!(
                    "vNext model length {maximum_model_tokens} cannot cover reusable execution startup ceiling {anchor_sequence_tokens}"
                )));
        }
        (
                anchors,
                Some(format!(
                    "automatic exact matrix needs {requested_sequence_tokens} synthetic sequence tokens, exceeding model length {maximum_model_tokens}; reduced to canonical power-of-two anchors"
                )),
            )
    } else {
        return Err(FerrumError::config(format!(
                "explicit reusable exact decode matrix needs {requested_sequence_tokens} synthetic sequence tokens, exceeding model length {maximum_model_tokens}"
            )));
    };
    if let Some(reason) = budget_reduction_reason {
        resolution_reasons.push(reason);
    }
    let reduction_reason = (!resolution_reasons.is_empty()).then(|| resolution_reasons.join("; "));

    let mut program_specs = effective_decode_widths
        .iter()
        .copied()
        .map(|width| {
            let shape = ReusableExecutionProgramShape::uniform_decode(
                u32::try_from(width)
                    .map_err(|_| FerrumError::config("vNext decode width exceeds u32"))?,
                1,
            )
            .map_err(|error| FerrumError::config(error.to_string()))?;
            ReusableExecutionProgramSpec::new(uniform_class.clone(), shape)
                .map_err(|error| FerrumError::config(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    for chunk in prefill_chunks.iter().copied() {
        if chunk.tokens_processed() > 0 {
            let shape = ReusableExecutionProgramShape::prefill(
                0,
                u64::try_from(chunk.tokens_processed())
                    .map_err(|_| FerrumError::config("vNext prefill prefix exceeds u64"))?,
                u64::try_from(chunk.total_prompt_tokens())
                    .map_err(|_| FerrumError::config("vNext prefill prompt exceeds u64"))?,
            )
            .map_err(|error| FerrumError::config(error.to_string()))?;
            program_specs.push(
                ReusableExecutionProgramSpec::new(packed_class.clone(), shape)
                    .map_err(|error| FerrumError::config(error.to_string()))?,
            );
        }
        let shape = ReusableExecutionProgramShape::prefill(
            u64::try_from(chunk.tokens_processed())
                .map_err(|_| FerrumError::config("vNext prefill frontier exceeds u64"))?,
            u64::try_from(chunk.tokens_to_process())
                .map_err(|_| FerrumError::config("vNext prefill chunk exceeds u64"))?,
            u64::try_from(chunk.total_prompt_tokens())
                .map_err(|_| FerrumError::config("vNext prefill prompt exceeds u64"))?,
        )
        .map_err(|error| FerrumError::config(error.to_string()))?;
        program_specs.push(
            ReusableExecutionProgramSpec::new(packed_class.clone(), shape)
                .map_err(|error| FerrumError::config(error.to_string()))?,
        );
    }
    let program_policy = if on_demand {
        ReusableExecutionProgramPolicy::exact_on_demand(program_specs)
    } else {
        ReusableExecutionProgramPolicy::exact_startup_sealed(
            REUSABLE_EXECUTION_WARMUP_PASSES,
            REUSABLE_EXECUTION_CAPTURE_PASSES,
            REUSABLE_EXECUTION_REPLAY_VALIDATION_PASSES,
            program_specs,
        )
    }
    .map_err(|error| FerrumError::config(error.to_string()))?;
    policy = policy
        .with_program_policy(program_policy)
        .map_err(|error| FerrumError::config(error.to_string()))?;
    Ok(VNextReusableExecutionPolicyResolution {
        policy,
        capture: Some(VNextReusableExecutionCaptureResolution {
            source,
            admission_maximum_decode_width: maximum_width,
            requested_decode_widths,
            effective_decode_widths,
            reduction_reason,
        }),
    })
}

/// Typed lifetime policy for values observed after a terminal device fence.
/// Product plans do not pay for all-node diagnostic retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VNextPlanObservationPolicy {
    Product,
    /// Derives a diagnostic observation plan from the same family, catalog,
    /// policy, providers, and dispatch contracts. Its memory layout and plan
    /// hash intentionally differ from the product plan because outputs remain
    /// live until terminal readback.
    DeterminismTerminalWitnesses,
}

impl VNextPlanObservationPolicy {
    fn apply(
        self,
        family: &PreparedModelFamily,
        options: &mut ProgramPlanCompileOptions,
    ) -> Result<()> {
        match self {
            Self::Product => Ok(()),
            Self::DeterminismTerminalWitnesses => options
                .retain_all_outputs_for_determinism(family)
                .map_err(|error| {
                    FerrumError::model(format!(
                        "vNext determinism terminal-witness retention: {error}"
                    ))
                }),
        }
    }

    fn validate_compilation(
        self,
        family: &PreparedModelFamily,
        compilation: &ProgramPlanCompilation,
    ) -> Result<()> {
        if self == Self::Product {
            return Ok(());
        }
        let expected = CompletionRetentionSpec::for_determinism_outputs(family)
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let retained = compilation
            .executable()
            .execution_plan()
            .payload()
            .retained_completion_values()
            .iter()
            .map(|value| value.value_id().clone())
            .collect::<BTreeSet<_>>();
        if compilation.completion_retention() != &expected || retained != *expected.values() {
            return Err(FerrumError::internal(
                "vNext determinism compilation did not retain every operation output exactly once",
            ));
        }
        Ok(())
    }
}

fn budget_reusable_decode_seed_prefill(
    chunks: &mut Vec<PrefillChunk>,
    prepare_device_programs: bool,
) -> Result<()> {
    if prepare_device_programs {
        chunks.push(PrefillChunk::new(0, 1, 1)?);
    }
    Ok(())
}

/// Typed executor policy resolved before plan compilation. None of these
/// values are inferred from a model name, GPU name, or hidden environment
/// combination.
#[derive(Debug, Clone)]
pub struct ProgramExecutorConfig {
    pub maximum_model_tokens: usize,
    pub startup_memory_plan: Option<ferrum_types::StartupMemoryPlan>,
    pub static_initialization: StaticInitializationPolicy,
    pub runtime_policy: ResolvedRuntimePolicy,
    pub invocation_preparation_strategy: InvocationPreparationStrategy,
    pub completion_readback_policy: ferrum_types::CompletionReadbackPolicy,
    pub sampling_readback_mode: SamplingReadbackMode,
    pub device_reusable_execution_enabled: bool,
    pub reusable_execution_prefill_chunks: Vec<PrefillChunk>,
    reusable_execution_capture_resolution: Option<VNextReusableExecutionCaptureResolution>,
    pub diagnostic_fault: Option<VNextDiagnosticFault>,
    plan_observation: VNextPlanObservationPolicy,
}

impl ProgramExecutorConfig {
    pub fn from_engine_config<R: DeviceRuntime>(
        engine: &EngineConfig,
        info: &ModelInfo,
        runtime: &R,
    ) -> Result<Self> {
        Self::from_engine_config_with_prefill_chunks(
            engine,
            info,
            runtime,
            &[],
            VNextPlanObservationPolicy::Product,
        )
    }

    pub fn for_determinism_collection<R: DeviceRuntime>(
        engine: &EngineConfig,
        info: &ModelInfo,
        runtime: &R,
    ) -> Result<Self> {
        // This diagnostic requests a fixed eager/replay matrix immediately
        // after preparation. Its automatic mode must prepare that matrix even
        // when product requests would normally populate the cache on demand.
        let mut engine = engine.clone();
        match engine.backend.reusable_execution_capture.preparation {
            ReusableExecutionPreparationMode::Auto => {
                engine.backend.reusable_execution_capture.preparation =
                    ReusableExecutionPreparationMode::Startup;
            }
            ReusableExecutionPreparationMode::Startup => {}
            ReusableExecutionPreparationMode::OnDemand => {
                return Err(FerrumError::config(
                    "fixed determinism collection requires startup reusable execution preparation",
                ))
            }
        }
        let required_chunks = [
            PrefillChunk::new(0, 1, 1)?,
            PrefillChunk::new(0, 4, 4)?,
            PrefillChunk::new(4, 4, 8)?,
        ];
        let config = Self::from_engine_config_with_prefill_chunks(
            &engine,
            info,
            runtime,
            &required_chunks,
            VNextPlanObservationPolicy::DeterminismTerminalWitnesses,
        )?;
        if config.runtime_policy.memory().maximum_active_sequences
            < u32::try_from(MAX_DETERMINISM_PARTICIPANTS)
                .map_err(|_| FerrumError::config("determinism width exceeds u32"))?
            || config.runtime_policy.admission().maximum_scheduled_tokens
                < u64::try_from(MAX_DETERMINISM_PARTICIPANTS)
                    .map_err(|_| FerrumError::config("determinism width exceeds u64"))?
        {
            return Err(FerrumError::config(format!(
                "vNext determinism collection requires at least {MAX_DETERMINISM_PARTICIPANTS} active sequences and scheduled tokens"
            )));
        }
        Ok(config)
    }

    fn from_engine_config_with_prefill_chunks<R: DeviceRuntime>(
        engine: &EngineConfig,
        info: &ModelInfo,
        runtime: &R,
        additional_prefill_chunks: &[PrefillChunk],
        plan_observation: VNextPlanObservationPolicy,
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
            SchedulingPolicy::ContinuousBatch => SchedulingDiscipline::FirstReady,
        };
        let dynamic_storage_profile_order = descriptor
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect::<Vec<_>>();

        let device_reusable_execution_supported = descriptor
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID);
        let prepare_device_programs =
            engine.backend.enable_reusable_execution && device_reusable_execution_supported;
        let mut capture_config = engine.backend.reusable_execution_capture.clone();
        if engine.backend.enable_reusable_execution {
            capture_config.preparation = capture_config
                .preparation
                .resolve(descriptor.capabilities.iter().any(|capability| {
                    capability.as_str() == DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID
                }))
                .map_err(FerrumError::config)?;
        }
        let mut reusable_execution_prefill_chunks = [
            engine.scheduler.prefill_step_chunk,
            engine.scheduler.active_decode_prefill_chunk,
        ]
        .into_iter()
        .flatten()
        .filter_map(|token_count| {
            let token_count = token_count
                .min(engine.batching.max_num_batched_tokens)
                .min(maximum_model_tokens);
            (token_count > 0).then_some(token_count)
        })
        .map(|token_count| PrefillChunk::new(0, token_count, token_count))
        .collect::<Result<Vec<_>>>()?;
        reusable_execution_prefill_chunks.extend_from_slice(additional_prefill_chunks);
        // Decode startup admission executes this exact prefill before the
        // synthetic sequence can participate in capture. Budget it in the
        // immutable program policy instead of observing an unplanned case.
        budget_reusable_decode_seed_prefill(
            &mut reusable_execution_prefill_chunks,
            prepare_device_programs,
        )?;
        if reusable_execution_prefill_chunks.iter().any(|chunk| {
            !chunk.is_final()
                || chunk.total_prompt_tokens() > maximum_model_tokens
                || chunk.tokens_to_process() > engine.batching.max_num_batched_tokens
        }) {
            return Err(FerrumError::config(
                "vNext reusable prefill capture requires a final chunk within model and scheduled-token limits",
            ));
        }
        reusable_execution_prefill_chunks.sort_unstable_by(|left, right| {
            right
                .tokens_to_process()
                .cmp(&left.tokens_to_process())
                .then_with(|| left.tokens_processed().cmp(&right.tokens_processed()))
                .then_with(|| left.total_prompt_tokens().cmp(&right.total_prompt_tokens()))
        });
        reusable_execution_prefill_chunks.dedup();
        // Workspace buckets remain backend-independent capacity policy. The
        // exact device-program matrix is attached only when this runtime will
        // prepare it, and becomes part of the resolved policy fingerprint.
        let reusable_execution_resolution = resolve_reusable_execution_policy(
            maximum_active_sequences,
            maximum_scheduled_tokens,
            maximum_model_tokens,
            &reusable_execution_prefill_chunks,
            &capture_config,
            prepare_device_programs,
        )?;
        let reusable_execution_policy = Some(reusable_execution_resolution.policy);
        let reusable_execution_capture_resolution = reusable_execution_resolution.capture;
        let execution_determinism = if prepare_device_programs {
            ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay
        } else {
            ExecutionDeterminismRequirement::BitwiseSameRuntime
        };
        let attention_execution = resolve_runtime_attention_authority(
            engine.runtime.attention_execution_policy,
            descriptor.capabilities.iter().any(|capability| {
                capability.as_str() == DEVICE_NATIVE_ADAPTIVE_ATTENTION_CAPABILITY_ID
            }),
            runtime.attention_execution_policy(),
        )?;

        let runtime_policy = ResolvedRuntimePolicy::new(
            POLICY_ID,
            POLICY_VERSION,
            scheduling,
            RuntimeMemoryPolicy {
                checkpoint_capacity: engine
                    .runtime
                    .prefix_state_cache_enabled
                    .then(|| {
                        CheckpointCapacityPolicy::new(
                            memory_budget.capacity_bytes - memory_budget.reserve_bytes,
                        )
                    })
                    .transpose()
                    .map_err(|error| FerrumError::config(error.to_string()))?,
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
                sequence_backing_growth: resolved_sequence_backing_growth(
                    engine.runtime.sequence_backing_growth,
                ),
                allow_defer: true,
                cancellation_check_interval_steps: DEFAULT_CANCELLATION_CHECK_INTERVAL_STEPS,
            },
            attention_execution,
            execution_determinism,
            reusable_execution_policy,
        )
        .map_err(|error| FerrumError::config(format!("invalid vNext policy: {error}")))?;
        let static_initialization = StaticInitializationPolicy::new(
            DEFAULT_STATIC_STAGING_BYTES,
            DEFAULT_STATIC_COMMANDS_PER_BATCH,
        )
        .map_err(|error| FerrumError::config(error.to_string()))?;
        let diagnostic_fault = engine.runtime.vnext_diagnostic_fault;
        if diagnostic_fault.is_some()
            && (engine.runtime.profile_detail != ObservabilityProfileDetail::Latency
                || engine.runtime.profile_jsonl.is_none()
                || !matches!(
                    engine.runtime.profile_entrypoint,
                    Some(ProfileEntrypoint::Run | ProfileEntrypoint::Serve)
                ))
        {
            return Err(FerrumError::config(
                "vNext diagnostic faults require a product run/serve latency profile and --profile-jsonl",
            ));
        }

        Ok(Self {
            maximum_model_tokens,
            startup_memory_plan: engine.runtime.startup_memory_plan.clone(),
            static_initialization,
            runtime_policy,
            invocation_preparation_strategy: engine.runtime.invocation_preparation_strategy,
            completion_readback_policy: engine.runtime.completion_readback_policy,
            sampling_readback_mode: engine.runtime.sampling_readback_mode,
            device_reusable_execution_enabled: engine.backend.enable_reusable_execution,
            reusable_execution_prefill_chunks,
            reusable_execution_capture_resolution,
            diagnostic_fault,
            plan_observation,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(tag = "topology", rename_all = "snake_case")]
enum VNextReusableExecutionDescriptor {
    UniformDecode {
        query_tokens_per_sequence: usize,
        token_capacity: usize,
        request_capacity: usize,
    },
    Prefill {
        tokens_processed: usize,
        token_capacity: usize,
        total_prompt_tokens: usize,
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

    const fn prefill(chunk: PrefillChunk) -> Self {
        Self::Prefill {
            tokens_processed: chunk.tokens_processed(),
            token_capacity: chunk.tokens_to_process(),
            total_prompt_tokens: chunk.total_prompt_tokens(),
            request_capacity: 1,
        }
    }

    fn from_program_shape(shape: ReusableExecutionProgramShape) -> Result<Self> {
        match shape {
            ReusableExecutionProgramShape::UniformDecode {
                query_tokens_per_sequence,
                token_capacity,
                request_capacity,
            } => Ok(Self::UniformDecode {
                query_tokens_per_sequence: usize::try_from(query_tokens_per_sequence)
                    .map_err(|_| FerrumError::config("vNext decode query width exceeds usize"))?,
                token_capacity: usize::try_from(token_capacity).map_err(|_| {
                    FerrumError::config("vNext decode token capacity exceeds usize")
                })?,
                request_capacity: usize::try_from(request_capacity).map_err(|_| {
                    FerrumError::config("vNext decode request capacity exceeds usize")
                })?,
            }),
            ReusableExecutionProgramShape::Prefill {
                tokens_processed,
                token_capacity,
                total_prompt_tokens,
            } => Ok(Self::Prefill {
                tokens_processed: usize::try_from(tokens_processed)
                    .map_err(|_| FerrumError::config("vNext prefill frontier exceeds usize"))?,
                token_capacity: usize::try_from(token_capacity)
                    .map_err(|_| FerrumError::config("vNext prefill capacity exceeds usize"))?,
                total_prompt_tokens: usize::try_from(total_prompt_tokens)
                    .map_err(|_| FerrumError::config("vNext prefill prompt exceeds usize"))?,
                request_capacity: 1,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct VNextReusableExecutionStartupPlan {
    descriptors: Vec<VNextReusableExecutionDescriptor>,
    prefill_chunks: Vec<PrefillChunk>,
    maximum_decode_sequence_tokens: usize,
    warmup_passes: usize,
    capture_passes: usize,
    replay_validation_passes: usize,
    program_policy: ReusableExecutionProgramPolicy,
    capture_resolution: VNextReusableExecutionCaptureResolution,
    device_plan: DeviceReusableExecutionPlan,
}

impl VNextReusableExecutionStartupPlan {
    fn resolve(
        program_policy: &ReusableExecutionProgramPolicy,
        capture_resolution: VNextReusableExecutionCaptureResolution,
        maximum_model_tokens: usize,
        prefill_chunks: &[PrefillChunk],
        maximum_device_executables: u64,
    ) -> Result<Self> {
        let mut prefill_chunks = prefill_chunks.iter().copied().collect::<Vec<_>>();
        if prefill_chunks
            .iter()
            .any(|chunk| !chunk.is_final() || chunk.total_prompt_tokens() > maximum_model_tokens)
        {
            return Err(FerrumError::config(
                "vNext reusable execution prefill chunk exceeds its immutable startup limits",
            ));
        }
        prefill_chunks.sort_unstable_by(|left, right| {
            right
                .tokens_to_process()
                .cmp(&left.tokens_to_process())
                .then_with(|| left.tokens_processed().cmp(&right.tokens_processed()))
                .then_with(|| left.total_prompt_tokens().cmp(&right.total_prompt_tokens()))
        });
        prefill_chunks.dedup();

        let mut descriptors = program_policy
            .programs()
            .iter()
            .map(|program| {
                let shape = program.shape();
                let expected_class = match shape {
                    ReusableExecutionProgramShape::UniformDecode { .. } => {
                        UNIFORM_QUERY_REUSABLE_CLASS
                    }
                    ReusableExecutionProgramShape::Prefill { .. } => {
                        PACKED_TOKEN_REUSABLE_CLASS
                    }
                };
                if program.class_id().as_str() != expected_class {
                    return Err(FerrumError::config(format!(
                        "vNext reusable {:?} program is bound to workspace class `{}`, expected `{expected_class}`",
                        shape,
                        program.class_id().as_str()
                    )));
                }
                VNextReusableExecutionDescriptor::from_program_shape(shape)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut decode_widths = descriptors
            .iter()
            .filter_map(|descriptor| match descriptor {
                VNextReusableExecutionDescriptor::UniformDecode {
                    query_tokens_per_sequence,
                    token_capacity,
                    request_capacity,
                } if *query_tokens_per_sequence == 1 && token_capacity == request_capacity => {
                    Some(*request_capacity)
                }
                VNextReusableExecutionDescriptor::UniformDecode { .. } => None,
                VNextReusableExecutionDescriptor::Prefill { .. } => None,
            })
            .collect::<Vec<_>>();
        decode_widths.sort_unstable();
        if decode_widths.is_empty()
            || decode_widths
                .iter()
                .any(|width| *width > MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH)
        {
            return Err(FerrumError::config(format!(
                "vNext reusable startup decode widths must be non-empty and within the independent hard bound 1..={MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH}"
            )));
        }
        if decode_widths != capture_resolution.effective_decode_widths {
            return Err(FerrumError::config(
                "vNext reusable startup widths differ from the resolved program policy",
            ));
        }
        decode_widths.sort_unstable_by(|left, right| right.cmp(left));

        let warmup_passes = usize::try_from(program_policy.warmup_passes())
            .map_err(|_| FerrumError::config("vNext reusable warmup passes exceed usize"))?;
        let capture_passes = usize::try_from(program_policy.capture_passes())
            .map_err(|_| FerrumError::config("vNext reusable capture passes exceed usize"))?;
        let replay_validation_passes = usize::try_from(program_policy.replay_validation_passes())
            .map_err(|_| {
            FerrumError::config("vNext reusable replay validation passes exceed usize")
        })?;
        let passes_per_width = warmup_passes
            .checked_add(capture_passes)
            .and_then(|passes| passes.checked_add(replay_validation_passes))
            .ok_or_else(|| FerrumError::config("vNext reusable startup passes overflow usize"))?;
        let maximum_decode_sequence_tokens = if program_policy.catalog_lifetime()
            == ReusableExecutionCatalogLifetime::OnDemandBounded
        {
            0
        } else {
            decode_widths
                .len()
                .checked_mul(passes_per_width)
                .and_then(|decode_tokens| decode_tokens.checked_add(1))
                .ok_or_else(|| FerrumError::config("vNext startup token ceiling overflowed"))?
        };
        if maximum_decode_sequence_tokens > maximum_model_tokens {
            return Err(FerrumError::config(format!(
                "vNext model length {maximum_model_tokens} cannot cover reusable execution startup ceiling {maximum_decode_sequence_tokens}"
            )));
        }

        let expected_prefill_descriptors = prefill_chunks
            .iter()
            .copied()
            .flat_map(|chunk| {
                let prefix = (chunk.tokens_processed() > 0).then(|| {
                    VNextReusableExecutionDescriptor::Prefill {
                        tokens_processed: 0,
                        token_capacity: chunk.tokens_processed(),
                        total_prompt_tokens: chunk.total_prompt_tokens(),
                        request_capacity: 1,
                    }
                });
                prefix
                    .into_iter()
                    .chain([Self::descriptor_for_chunk(chunk)])
            })
            .collect::<BTreeSet<_>>();
        let resolved_prefill_descriptors = descriptors
            .iter()
            .copied()
            .filter(|descriptor| {
                matches!(descriptor, VNextReusableExecutionDescriptor::Prefill { .. })
            })
            .collect::<BTreeSet<_>>();
        if expected_prefill_descriptors != resolved_prefill_descriptors {
            return Err(FerrumError::config(
                "vNext reusable prefill startup work differs from the resolved program policy",
            ));
        }
        descriptors.sort_unstable();
        let decode_descriptor_count = descriptors
            .iter()
            .take_while(|descriptor| {
                matches!(
                    descriptor,
                    VNextReusableExecutionDescriptor::UniformDecode { .. }
                )
            })
            .count();
        descriptors[..decode_descriptor_count].reverse();

        let maximum_executables = usize::try_from(maximum_device_executables)
            .map_err(|_| FerrumError::config("vNext reusable executable capacity exceeds usize"))?;
        let device_plan = if program_policy.catalog_lifetime()
            == ReusableExecutionCatalogLifetime::OnDemandBounded
        {
            DeviceReusableExecutionPlan::on_demand(maximum_executables)
        } else {
            DeviceReusableExecutionPlan::new(maximum_executables)
        }
        .map_err(|error| FerrumError::config(error.to_string()))?;
        Ok(Self {
            descriptors,
            prefill_chunks,
            maximum_decode_sequence_tokens,
            warmup_passes,
            capture_passes,
            replay_validation_passes,
            program_policy: program_policy.clone(),
            capture_resolution,
            device_plan,
        })
    }

    const fn descriptor_for_chunk(chunk: PrefillChunk) -> VNextReusableExecutionDescriptor {
        VNextReusableExecutionDescriptor::prefill(chunk)
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

    fn prefill_chunks(&self) -> Vec<PrefillChunk> {
        self.prefill_chunks.clone()
    }

    fn prefill_token_counts(&self) -> Vec<usize> {
        let mut token_counts = self
            .prefill_chunks()
            .into_iter()
            .map(PrefillChunk::tokens_to_process)
            .collect::<Vec<_>>();
        token_counts.sort_unstable_by(|left, right| right.cmp(left));
        token_counts.dedup();
        token_counts
    }

    fn prefill_wave_shapes(&self) -> usize {
        self.prefill_chunks()
            .iter()
            .map(|chunk| usize::from(chunk.tokens_processed() > 0) + 1)
            .sum()
    }

    fn decode_catalog_omits_admitted_widths(&self) -> bool {
        self.capture_resolution
            .effective_decode_widths
            .iter()
            .copied()
            .ne(1..=self.capture_resolution.admission_maximum_decode_width)
    }
}

#[derive(Debug, Clone, Serialize)]
struct VNextReusableExecutionCatalogProgramReceipt {
    program_id: DeviceReusableExecutionProgramId,
    program_fingerprint: String,
    state: DeviceReusableExecutionProgramState,
    node_count: u32,
    eager_boundary_node_indices: Vec<u32>,
    resident_segments: Vec<DeviceReusableExecutionSegment>,
    per_wave_binding_node_indices: Vec<u32>,
    gaps: Vec<DeviceReusableExecutionProgramGap>,
}

impl VNextReusableExecutionCatalogProgramReceipt {
    fn from_program(program: &DeviceReusableExecutionProgram) -> Self {
        Self {
            program_id: program.program_id().clone(),
            program_fingerprint: program.program_id().fingerprint(),
            state: program.state(),
            node_count: program.node_count(),
            eager_boundary_node_indices: program.eager_boundary_node_indices().to_vec(),
            resident_segments: program.segments().to_vec(),
            per_wave_binding_node_indices: program.per_wave_binding_node_indices().to_vec(),
            gaps: program.gaps().to_vec(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct VNextReusableExecutionCaptureCaseReceipt {
    descriptor: VNextReusableExecutionDescriptor,
    observed_program_fingerprints: Vec<String>,
    resident_program_fingerprints: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct VNextReusableExecutionStartupReport {
    enabled: bool,
    supported: bool,
    eager_fallback_required: bool,
    resolved_runtime_policy_fingerprint: String,
    resolved_program_policy: Option<ReusableExecutionProgramPolicy>,
    decode_width_resolution: Option<VNextReusableExecutionCaptureResolution>,
    maximum_device_executables: usize,
    requested_descriptors: Vec<VNextReusableExecutionDescriptor>,
    prepared_descriptors: Vec<VNextReusableExecutionDescriptor>,
    capture_case_receipts: Vec<VNextReusableExecutionCaptureCaseReceipt>,
    catalog_programs: Vec<VNextReusableExecutionCatalogProgramReceipt>,
    requested_decode_widths: Vec<usize>,
    prepared_decode_widths: Vec<usize>,
    requested_prefill_token_counts: Vec<usize>,
    prepared_prefill_token_counts: Vec<usize>,
    requested_prefill_chunks: Vec<PrefillChunk>,
    prepared_prefill_chunks: Vec<PrefillChunk>,
    synthetic_sequences: usize,
    eager_warmup_waves: usize,
    capture_waves: usize,
    replay_inventory_check_waves: usize,
    prepared_programs: usize,
    device_preparation: DeviceReusableExecutionPreparation,
    elapsed_ms: u64,
}

fn reusable_executable_inventory_matches(
    before: DeviceReusableExecutionPreparation,
    after: DeviceReusableExecutionPreparation,
) -> bool {
    before.maximum_executables() == after.maximum_executables()
        && before.resident_executables() == after.resident_executables()
        && before.rejected_executables() == after.rejected_executables()
        && before.captured_executables() == after.captured_executables()
        && before.uploaded_executables() == after.uploaded_executables()
}

fn reusable_execution_requires_eager_fallback(
    preparation: DeviceReusableExecutionPreparation,
) -> bool {
    preparation.resident_executables() == 0
        || preparation.rejected_executables() != 0
        || preparation.capacity_deferred_executables() != 0
}

fn reusable_execution_program_catalog_is_usable(
    preparation: DeviceReusableExecutionPreparation,
    prepared_programs: usize,
) -> bool {
    preparation.resident_executables() == 0 || prepared_programs != 0
}

fn reusable_startup_case_budget_violation<'a, D: Ord, P: Ord>(
    requested_cases: &BTreeSet<D>,
    observed_programs: &'a BTreeMap<D, BTreeSet<P>>,
) -> Option<(&'a D, usize)> {
    observed_programs
        .iter()
        .find(|(case, programs)| !requested_cases.contains(*case) || programs.len() > 1)
        .map(|(case, programs)| (case, programs.len()))
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
struct VNextLanguageIoIds {
    token_input: ProgramValueId,
    token_mask_input: ProgramValueId,
    repetition_token_ids_input: ProgramValueId,
    repetition_offsets_input: ProgramValueId,
    repetition_penalty_input: ProgramValueId,
    logits_output: ProgramValueId,
    greedy_token_output: ProgramValueId,
}

fn is_language_token_embedding_operation(operation_id: &str) -> bool {
    operation_id == TOKEN_EMBEDDING_OPERATION_ID
        || operation_id == TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID
}

fn is_language_masked_argmax_operation(operation_id: &str) -> bool {
    operation_id == LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID
        || operation_id == LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID
}

#[derive(Debug, Clone)]
struct VNextIoBinding {
    input_node_id: NodeId,
    input_ordinal: u32,
    token_mask_input_node_id: NodeId,
    token_mask_input_ordinal: u32,
    token_mask_residency_eligible: bool,
    repetition_token_ids_input_node_id: NodeId,
    repetition_token_ids_input_ordinal: u32,
    repetition_offsets_input_node_id: NodeId,
    repetition_offsets_input_ordinal: u32,
    repetition_penalty_input_node_id: NodeId,
    repetition_penalty_input_ordinal: u32,
    repetition_capacity: usize,
    output_node_id: NodeId,
    output_resource_id: ResourceId,
    output_offset_bytes: u64,
    output_layout: HostTransferLayout,
    output_element_type: ElementType,
    output_elements: usize,
    greedy_token_output_node_id: NodeId,
    greedy_token_output_resource_id: ResourceId,
    greedy_token_output_offset_bytes: u64,
    greedy_token_output_layout: HostTransferLayout,
}

struct VNextReusableExecutionCatalog {
    lane_epoch: u64,
    programs: BTreeMap<DeviceReusableExecutionProgramId, DeviceReusableExecutionProgram>,
}

const MAX_REUSABLE_EXECUTION_CATALOG_MISS_KEYS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum VNextReusableExecutionCatalogMissReason {
    ProgramIdentityUnavailable,
    CatalogEmpty,
    ProgramAbsent,
    ProgramNonResident,
    EpochMismatch,
}

impl VNextReusableExecutionCatalogMissReason {
    const fn is_epoch_mismatch(self) -> bool {
        matches!(self, Self::EpochMismatch)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct VNextReusableExecutionCatalogMissKey {
    immediate_sequences: u32,
    immediate_tokens: u64,
    immediate_pages: u64,
    topology_fingerprint: DeviceReusableExecutionTopologyFingerprint,
    reason: VNextReusableExecutionCatalogMissReason,
}

impl VNextReusableExecutionCatalogMissKey {
    fn from_program_id(
        program_id: &DeviceReusableExecutionProgramId,
        reason: VNextReusableExecutionCatalogMissReason,
    ) -> Self {
        Self {
            immediate_sequences: program_id.immediate_sequences(),
            immediate_tokens: program_id.immediate_tokens(),
            immediate_pages: program_id.immediate_pages(),
            topology_fingerprint: program_id.topology_fingerprint(),
            reason,
        }
    }

    fn without_program_identity(
        work_shape: &BatchWorkShape,
        reason: VNextReusableExecutionCatalogMissReason,
    ) -> Self {
        Self {
            immediate_sequences: work_shape.immediate_sequences(),
            immediate_tokens: work_shape.immediate_tokens(),
            immediate_pages: work_shape.immediate_pages(),
            topology_fingerprint: DeviceReusableExecutionTopologyFingerprint::static_program(),
            reason,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct VNextReusableExecutionCatalogMissRow {
    key: VNextReusableExecutionCatalogMissKey,
    count: u64,
}

#[derive(Debug, Clone, Serialize)]
struct VNextReusableExecutionCatalogMissOverflowRow {
    reason: VNextReusableExecutionCatalogMissReason,
    count: u64,
}

#[derive(Debug, Clone, Serialize)]
struct VNextReusableExecutionCatalogMissSnapshot {
    maximum_distinct_keys: usize,
    distinct_keys: usize,
    rows: Vec<VNextReusableExecutionCatalogMissRow>,
    overflow: Vec<VNextReusableExecutionCatalogMissOverflowRow>,
}

#[derive(Default)]
struct VNextReusableExecutionCatalogMissLedger {
    counts: BTreeMap<VNextReusableExecutionCatalogMissKey, u64>,
    overflow: BTreeMap<VNextReusableExecutionCatalogMissReason, u64>,
}

impl VNextReusableExecutionCatalogMissLedger {
    fn record(&mut self, key: VNextReusableExecutionCatalogMissKey) {
        if let Some(count) = self.counts.get_mut(&key) {
            *count = count.saturating_add(1);
        } else if self.counts.len() < MAX_REUSABLE_EXECUTION_CATALOG_MISS_KEYS {
            self.counts.insert(key, 1);
        } else {
            let count = self.overflow.entry(key.reason).or_default();
            *count = count.saturating_add(1);
        }
    }

    fn snapshot(&self) -> VNextReusableExecutionCatalogMissSnapshot {
        VNextReusableExecutionCatalogMissSnapshot {
            maximum_distinct_keys: MAX_REUSABLE_EXECUTION_CATALOG_MISS_KEYS,
            distinct_keys: self.counts.len(),
            rows: self
                .counts
                .iter()
                .map(|(key, count)| VNextReusableExecutionCatalogMissRow {
                    key: *key,
                    count: *count,
                })
                .collect(),
            overflow: self
                .overflow
                .iter()
                .map(
                    |(reason, count)| VNextReusableExecutionCatalogMissOverflowRow {
                        reason: *reason,
                        count: *count,
                    },
                )
                .collect(),
        }
    }

    fn reset(&mut self) {
        self.counts.clear();
        self.overflow.clear();
    }
}

#[derive(Default)]
struct VNextExecutorMetrics {
    prefix_cache: Arc<prefix_cache::PrefixCacheMetrics>,
    invocation_preparation: invocation_preparation::PreparationMetrics,
    program_binding_upload: program_binding_upload::ProgramBindingUploadMetrics,
    prefill_operations: AtomicU64,
    prefill_frontier_narrowings: AtomicU64,
    decode_operations: AtomicU64,
    prepared_wave_topology: VNextPreparedWaveTopologyMetrics,
    submitted_waves: AtomicU64,
    completed_waves: AtomicU64,
    failed_waves: AtomicU64,
    direct_reusable_waves: AtomicU64,
    direct_reusable_segments: AtomicU64,
    direct_reusable_logical_nodes: AtomicU64,
    direct_reusable_binding_nodes: AtomicU64,
    direct_reusable_fallbacks: AtomicU64,
    reusable_catalog_misses: AtomicU64,
    reusable_catalog_epoch_misses: AtomicU64,
    reusable_catalog_miss_ledger: Mutex<VNextReusableExecutionCatalogMissLedger>,
    reusable_cold_path: reusable_cold_path::Ledger,
    identity_waves: AtomicU64,
    identity_logical_nodes: AtomicU64,
    identity_nodes_materialized_before_submit: AtomicU64,
    identity_full_participant_materializations_before_submit: AtomicU64,
    definitely_not_submitted_retries: AtomicU64,
    request_deferrals: AtomicU64,
    sequence_deferrals: AtomicU64,
    extension_deferrals: AtomicU64,
    step_deferrals: AtomicU64,
    wave_deferrals: AtomicU64,
    backing_deferrals: AtomicU64,
    uploaded_bytes: AtomicU64,
    readback_bytes: AtomicU64,
    full_logits_readback_waves: AtomicU64,
    greedy_token_readback_waves: AtomicU64,
    hybrid_readback_waves: AtomicU64,
    full_logits_readback_participants: AtomicU64,
    greedy_token_readback_participants: AtomicU64,
    no_selectable_finite_logit_participants: AtomicU64,
    product_readback_bytes: AtomicU64,
    greedy_policy_fallback_waves: AtomicU64,
    token_mask_upload_participants: AtomicU64,
    token_mask_cache_hit_participants: AtomicU64,
    sparse_repetition_waves: AtomicU64,
    sparse_repetition_participants: AtomicU64,
    sparse_repetition_token_ids_uploaded: AtomicU64,
    total_prefill_us: AtomicU64,
    total_decode_us: AtomicU64,
    wave_timing: VNextWaveTimingMetrics,
    prefill_wave_timing: VNextWaveTimingMetrics,
    decode_wave_timing: VNextWaveTimingMetrics,
    mixed_wave_timing: VNextWaveTimingMetrics,
    device_timing: VNextDeviceTimingMetrics,
    prefill_device_timing: VNextDeviceTimingMetrics,
    decode_device_timing: VNextDeviceTimingMetrics,
    mixed_device_timing: VNextDeviceTimingMetrics,
    last_failure: Mutex<Option<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VNextExecutionWaveKind {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VNextProductOutputMode {
    FullLogits,
    GreedyToken,
}

/// Product consumption is distinct from the complete device plan. An
/// intermediate prefill advances state without authorizing token sampling.
#[derive(Debug, Clone)]
enum VNextParticipantOutputRole {
    IntermediatePrefill,
    FinalPrefill,
    Decode(LogitsReturnPolicy),
}

impl VNextParticipantOutputRole {
    fn prefill(chunk: PrefillChunk) -> Self {
        if chunk.is_final() {
            Self::FinalPrefill
        } else {
            Self::IntermediatePrefill
        }
    }

    fn logits_policy(&self) -> Option<&LogitsReturnPolicy> {
        match self {
            Self::Decode(policy) => Some(policy),
            Self::IntermediatePrefill | Self::FinalPrefill => None,
        }
    }

    fn matches_frontier(
        &self,
        kind: VNextExecutionWaveKind,
        span: &TokenSpanWork,
        token_count: usize,
    ) -> bool {
        let end = span.immediate_token_range().end;
        match self {
            Self::IntermediatePrefill => {
                kind != VNextExecutionWaveKind::Decode && end < token_count as u64
            }
            Self::FinalPrefill => {
                kind != VNextExecutionWaveKind::Decode && end == token_count as u64
            }
            Self::Decode(_) => {
                kind != VNextExecutionWaveKind::Prefill
                    && span.immediate_tokens() == 1
                    && end == token_count as u64
            }
        }
    }
}

impl VNextProductOutputMode {
    const fn checkpoint_mode(self) -> VNextCheckpointProductOutputMode {
        match self {
            Self::FullLogits => VNextCheckpointProductOutputMode::FullLogits,
            Self::GreedyToken => VNextCheckpointProductOutputMode::GreedyToken,
        }
    }
}

#[derive(Debug, Clone)]
enum VNextProductTokenMaskContent {
    AllValid {
        vocabulary_size: usize,
    },
    Selection {
        vocabulary_size: usize,
        fingerprint: u64,
        valid_token_mask: Arc<[i8]>,
    },
}

impl VNextProductTokenMaskContent {
    fn from_policy(
        policy: Option<&LogitsReturnPolicy>,
        output_mode: VNextProductOutputMode,
        vocabulary_size: usize,
    ) -> Self {
        if output_mode == VNextProductOutputMode::GreedyToken {
            if let Some(LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(token_mask),
                ..
            }) = policy
            {
                return Self::Selection {
                    vocabulary_size,
                    fingerprint: token_mask.fingerprint,
                    valid_token_mask: Arc::clone(&token_mask.valid_token_mask),
                };
            }
        }
        Self::AllValid { vocabulary_size }
    }

    fn normalized(&self) -> Vec<u8> {
        match self {
            Self::AllValid { vocabulary_size } => vec![1_u8; *vocabulary_size],
            Self::Selection {
                vocabulary_size,
                valid_token_mask,
                ..
            } => {
                let mut output = vec![0_u8; *vocabulary_size];
                for (destination, source) in output.iter_mut().zip(valid_token_mask.iter().copied())
                {
                    *destination = u8::from(source != 0);
                }
                output
            }
        }
    }
}

#[derive(Debug)]
enum VNextResidentProductTokenMaskContent {
    AllValid {
        vocabulary_size: usize,
    },
    Selection {
        vocabulary_size: usize,
        fingerprint: u64,
        source_len: usize,
        valid_token_mask: Weak<[i8]>,
    },
}

impl VNextResidentProductTokenMaskContent {
    fn from_requested(requested: &VNextProductTokenMaskContent) -> Self {
        match requested {
            VNextProductTokenMaskContent::AllValid { vocabulary_size } => Self::AllValid {
                vocabulary_size: *vocabulary_size,
            },
            VNextProductTokenMaskContent::Selection {
                vocabulary_size,
                fingerprint,
                valid_token_mask,
            } => Self::Selection {
                vocabulary_size: *vocabulary_size,
                fingerprint: *fingerprint,
                source_len: valid_token_mask.len(),
                valid_token_mask: Arc::downgrade(valid_token_mask),
            },
        }
    }

    fn exactly_matches(&self, requested: &VNextProductTokenMaskContent) -> bool {
        match (self, requested) {
            (
                Self::AllValid {
                    vocabulary_size: resident_vocabulary_size,
                },
                VNextProductTokenMaskContent::AllValid {
                    vocabulary_size: requested_vocabulary_size,
                },
            ) => resident_vocabulary_size == requested_vocabulary_size,
            (
                Self::Selection {
                    vocabulary_size: resident_vocabulary_size,
                    fingerprint: resident_fingerprint,
                    source_len,
                    valid_token_mask: resident_mask,
                },
                VNextProductTokenMaskContent::Selection {
                    vocabulary_size: requested_vocabulary_size,
                    fingerprint: requested_fingerprint,
                    valid_token_mask: requested_mask,
                },
            ) => {
                if resident_vocabulary_size != requested_vocabulary_size
                    || resident_fingerprint != requested_fingerprint
                    || *source_len != requested_mask.len()
                {
                    return false;
                }
                resident_mask.upgrade().is_some_and(|resident_mask| {
                    Arc::ptr_eq(&resident_mask, requested_mask)
                        || resident_mask.as_ref() == requested_mask.as_ref()
                })
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VNextProductTokenMaskSlotIdentity {
    LaneStable(Arc<LaneStableArenaSlotIdentity>),
    #[cfg(test)]
    Test(u64),
}

impl VNextProductTokenMaskSlotIdentity {
    fn slot_id(&self) -> u64 {
        match self {
            Self::LaneStable(identity) => identity.slot_id(),
            #[cfg(test)]
            Self::Test(slot_id) => *slot_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VNextProductTokenMaskSlotTarget {
    identity: VNextProductTokenMaskSlotIdentity,
    participant_index: usize,
}

impl VNextProductTokenMaskSlotTarget {
    fn cache_key(&self) -> (u64, usize) {
        (self.identity.slot_id(), self.participant_index)
    }
}

#[derive(Debug)]
struct VNextResidentProductTokenMaskEntry {
    identity: VNextProductTokenMaskSlotIdentity,
    content: VNextResidentProductTokenMaskContent,
}

#[derive(Debug, Clone)]
struct VNextProductTokenMaskSubmissionPlan {
    target: Option<VNextProductTokenMaskSlotTarget>,
    content: VNextProductTokenMaskContent,
    upload_required: bool,
}

#[derive(Debug, Default)]
struct VNextProductTokenMaskResidency {
    entries: BTreeMap<(u64, usize), VNextResidentProductTokenMaskEntry>,
}

impl VNextProductTokenMaskResidency {
    fn prepare(
        &mut self,
        target: Option<VNextProductTokenMaskSlotTarget>,
        content: VNextProductTokenMaskContent,
    ) -> VNextProductTokenMaskSubmissionPlan {
        let upload_required = target.as_ref().is_none_or(|target| {
            let key = target.cache_key();
            let hit = self.entries.get(&key).is_some_and(|entry| {
                entry.identity == target.identity && entry.content.exactly_matches(&content)
            });
            if !hit {
                // An upload becomes resident only after terminal device success
                // publishes it below while the Step slot is still leased.
                self.entries.remove(&key);
            }
            !hit
        });
        VNextProductTokenMaskSubmissionPlan {
            target,
            content,
            upload_required,
        }
    }

    fn publish(&mut self, plans: &[VNextProductTokenMaskSubmissionPlan]) {
        let cacheable = plans
            .iter()
            .filter(|plan| plan.upload_required && plan.target.is_some())
            .count();
        if cacheable == 0 {
            return;
        }
        if cacheable > MAX_PRODUCT_TOKEN_MASK_SLOT_CACHE_ENTRIES {
            self.entries.clear();
            return;
        }
        let new_entries = plans
            .iter()
            .filter(|plan| plan.upload_required)
            .filter_map(|plan| plan.target.as_ref())
            .filter(|target| !self.entries.contains_key(&target.cache_key()))
            .count();
        if self.entries.len().saturating_add(new_entries)
            > MAX_PRODUCT_TOKEN_MASK_SLOT_CACHE_ENTRIES
        {
            // Eviction can only turn future hits into uploads. Device contents
            // remain authoritative and are never inferred from a missing entry.
            self.entries.clear();
        }
        for plan in plans {
            if plan.upload_required {
                let Some(target) = &plan.target else {
                    continue;
                };
                self.entries.insert(
                    target.cache_key(),
                    VNextResidentProductTokenMaskEntry {
                        identity: target.identity.clone(),
                        content: VNextResidentProductTokenMaskContent::from_requested(
                            &plan.content,
                        ),
                    },
                );
            }
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

struct VNextProductTokenMaskResidencyTransaction<'a> {
    residency: &'a Mutex<VNextProductTokenMaskResidency>,
    plans: Vec<VNextProductTokenMaskSubmissionPlan>,
    published: bool,
    settled: bool,
}

impl<'a> VNextProductTokenMaskResidencyTransaction<'a> {
    fn prepare(
        residency: &'a Mutex<VNextProductTokenMaskResidency>,
        slot_identity: Option<LaneStableArenaSlotIdentity>,
        contents: impl IntoIterator<Item = VNextProductTokenMaskContent>,
    ) -> Self {
        let slot_identity = slot_identity
            .map(Arc::new)
            .map(VNextProductTokenMaskSlotIdentity::LaneStable);
        Self::prepare_with_identity(residency, slot_identity, contents)
    }

    fn prepare_with_identity(
        residency: &'a Mutex<VNextProductTokenMaskResidency>,
        slot_identity: Option<VNextProductTokenMaskSlotIdentity>,
        contents: impl IntoIterator<Item = VNextProductTokenMaskContent>,
    ) -> Self {
        let mut ledger = residency.lock();
        let plans = contents
            .into_iter()
            .enumerate()
            .map(|(participant_index, content)| {
                let target =
                    slot_identity
                        .clone()
                        .map(|identity| VNextProductTokenMaskSlotTarget {
                            identity,
                            participant_index,
                        });
                ledger.prepare(target, content)
            })
            .collect();
        drop(ledger);
        Self {
            residency,
            plans,
            published: false,
            settled: false,
        }
    }

    #[cfg(test)]
    fn prepare_for_test(
        residency: &'a Mutex<VNextProductTokenMaskResidency>,
        slot_id: Option<u64>,
        contents: impl IntoIterator<Item = VNextProductTokenMaskContent>,
    ) -> Self {
        Self::prepare_with_identity(
            residency,
            slot_id.map(VNextProductTokenMaskSlotIdentity::Test),
            contents,
        )
    }

    fn plans(&self) -> &[VNextProductTokenMaskSubmissionPlan] {
        &self.plans
    }

    fn publish(&mut self) {
        debug_assert!(!self.published);
        if self
            .plans
            .iter()
            .any(|plan| plan.upload_required && plan.target.is_some())
        {
            self.residency.lock().publish(&self.plans);
        }
        self.published = true;
    }

    fn settle_success(&mut self) {
        debug_assert!(self.published);
        self.settled = true;
    }

    fn invalidate_before_slot_release(&mut self) {
        debug_assert!(!self.published);
        self.residency.lock().clear();
        self.settled = true;
    }
}

impl Drop for VNextProductTokenMaskResidencyTransaction<'_> {
    fn drop(&mut self) {
        if !self.settled {
            // Any unclassified submission or host-side terminal failure loses
            // the proof, never the device allocation. The next wave uploads.
            self.residency.lock().clear();
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VNextProductRepetitionInput<'a> {
    token_ids: &'a [u32],
    penalty: f32,
}

impl VNextProductRepetitionInput<'_> {
    const NONE: Self = Self {
        token_ids: &[],
        penalty: 1.0,
    };

    fn is_active(self) -> bool {
        !self.token_ids.is_empty() && self.penalty != 1.0
    }
}

fn product_output_mode_for_roles<'a>(
    kind: VNextExecutionWaveKind,
    roles: impl IntoIterator<Item = &'a VNextParticipantOutputRole>,
) -> VNextProductOutputMode {
    if kind == VNextExecutionWaveKind::Prefill {
        return VNextProductOutputMode::FullLogits;
    }
    let mut has_decode = false;
    for role in roles {
        match role {
            VNextParticipantOutputRole::IntermediatePrefill
                if kind == VNextExecutionWaveKind::Mixed => {}
            VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax { .. }) => {
                has_decode = true;
            }
            _ => return VNextProductOutputMode::FullLogits,
        }
    }
    if has_decode {
        VNextProductOutputMode::GreedyToken
    } else {
        VNextProductOutputMode::FullLogits
    }
}

#[cfg(test)]
fn normalized_product_token_mask(
    policy: Option<&LogitsReturnPolicy>,
    output_mode: VNextProductOutputMode,
    vocabulary_size: usize,
) -> Vec<u8> {
    VNextProductTokenMaskContent::from_policy(policy, output_mode, vocabulary_size).normalized()
}

fn product_repetition_input(
    policy: Option<&LogitsReturnPolicy>,
    output_mode: VNextProductOutputMode,
) -> VNextProductRepetitionInput<'_> {
    if output_mode != VNextProductOutputMode::GreedyToken {
        return VNextProductRepetitionInput::NONE;
    }
    let Some(LogitsReturnPolicy::GreedyArgmax {
        repetition_penalty: Some(repetition),
        ..
    }) = policy
    else {
        return VNextProductRepetitionInput::NONE;
    };
    if repetition.is_empty() {
        VNextProductRepetitionInput::NONE
    } else {
        VNextProductRepetitionInput {
            token_ids: repetition.token_ids(),
            penalty: repetition.penalty(),
        }
    }
}

fn padded_repetition_token_id_bytes(token_ids: &[u32], capacity: usize) -> Result<Vec<u8>> {
    if token_ids.len() > capacity {
        return Err(FerrumError::backend(format!(
            "vNext sparse repetition input contains {} ids, capacity is {capacity}",
            token_ids.len()
        )));
    }
    let byte_capacity = capacity
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| FerrumError::backend("vNext repetition token-id bytes overflow usize"))?;
    let mut bytes = vec![0_u8; byte_capacity];
    for (destination, token_id) in bytes
        .chunks_exact_mut(std::mem::size_of::<u32>())
        .zip(token_ids)
    {
        destination.copy_from_slice(&token_id.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_selected_token(bytes: &[u8], vocabulary_size: usize) -> Result<TokenId> {
    let token_bytes: [u8; 4] = bytes.try_into().map_err(|_| {
        FerrumError::backend(format!(
            "vNext selected-token readback contains {} bytes, expected 4",
            bytes.len()
        ))
    })?;
    let token = u32::from_le_bytes(token_bytes);
    if usize::try_from(token)
        .ok()
        .is_none_or(|token| token >= vocabulary_size)
    {
        return Err(FerrumError::backend(format!(
            "vNext masked argmax returned invalid token {token} for vocabulary {vocabulary_size}"
        )));
    }
    Ok(TokenId::new(token))
}

fn apply_teacher_forced_decision(
    outputs: &mut [ExecutorSamplingOutput],
    decision: VNextTeacherForcedDecision,
) -> Result<()> {
    let [output] = outputs else {
        return Err(FerrumError::internal(format!(
            "vNext teacher-forced decision requires one output, got {}",
            outputs.len()
        )));
    };
    let ExecutorSamplingOutput::FullLogits(logits) = output else {
        return Err(FerrumError::internal(
            "vNext teacher-forced decision requires full logits",
        ));
    };
    let token_index = usize::try_from(decision.token_id().get())
        .map_err(|_| FerrumError::internal("vNext teacher-forced token id exceeds usize"))?;
    let selected = logits.get(token_index).copied().ok_or_else(|| {
        FerrumError::internal(format!(
            "vNext teacher-forced token {} at decision {} exceeds logits width {}",
            decision.token_id().get(),
            decision.token_index(),
            logits.len()
        ))
    })?;
    if !selected.is_finite() {
        return Err(FerrumError::model(format!(
            "vNext teacher-forced token {} at decision {} has a non-finite raw logit",
            decision.token_id().get(),
            decision.token_index()
        )));
    }
    logits.fill(f32::NEG_INFINITY);
    logits[token_index] = 0.0;
    Ok(())
}

fn nonterminal_completion_message(observation: &CompletionReadbackBatchObservation) -> String {
    format!("vNext completion did not reach a quiescent terminal: {observation:?}")
}

enum VNextTerminalReadbacks {
    Batch(CompletionReadbackBatchRequest),
    Collection(CompletionReadbackCollectionRequest),
    Selection(CompletionReadbackSelectionRequest),
}

impl VNextTerminalReadbacks {
    fn terminal_packet_request(
        &self,
        policy: ferrum_types::CompletionReadbackPolicy,
        output_mode: VNextProductOutputMode,
        checkpoint_capture: bool,
    ) -> Option<&CompletionReadbackBatchRequest> {
        if policy != ferrum_types::CompletionReadbackPolicy::BatchedU32
            || output_mode != VNextProductOutputMode::GreedyToken
            || checkpoint_capture
        {
            return None;
        }
        let Self::Batch(batch) = self else {
            return None;
        };
        // The batch constructor establishes canonical order and one typed
        // node/resource group. Core validates its exact fresh wave authority.
        ((1..=32).contains(&batch.len())
            && batch.requests().iter().all(|request| {
                let layout = request.output_layout();
                layout.element_type() == ElementType::U32 && layout.element_count() == 1
            }))
        .then_some(batch)
    }
}

impl VNextExecutionWaveKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Mixed => "mixed",
        }
    }

    const fn reusable_execution_class(self) -> &'static str {
        match self {
            Self::Prefill | Self::Mixed => PACKED_TOKEN_REUSABLE_CLASS,
            Self::Decode => UNIFORM_QUERY_REUSABLE_CLASS,
        }
    }
}

#[derive(Default)]
struct VNextStepAdmissionTimingMetrics {
    authority_and_policy_validate: AtomicDurationMetrics,
    demand_evaluate: AtomicDurationMetrics,
    backing_claim: AtomicDurationMetrics,
    logical_capacity_claim: AtomicDurationMetrics,
    transaction_validate_and_fingerprint: AtomicDurationMetrics,
    frame_capture_and_lease: AtomicDurationMetrics,
}

impl VNextStepAdmissionTimingMetrics {
    fn record(&self, phase: StepResourceAdmissionProfilePhase, duration: Duration) {
        match phase {
            StepResourceAdmissionProfilePhase::AuthorityAndPolicyValidate => {
                &self.authority_and_policy_validate
            }
            StepResourceAdmissionProfilePhase::DemandEvaluate => &self.demand_evaluate,
            StepResourceAdmissionProfilePhase::BackingClaim => &self.backing_claim,
            StepResourceAdmissionProfilePhase::LogicalCapacityClaim => &self.logical_capacity_claim,
            StepResourceAdmissionProfilePhase::TransactionValidateAndFingerprint => {
                &self.transaction_validate_and_fingerprint
            }
            StepResourceAdmissionProfilePhase::FrameCaptureAndLease => {
                &self.frame_capture_and_lease
            }
        }
        .record(duration);
    }

    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "collection": "profile_attached_only",
            "authority_and_policy_validate": self.authority_and_policy_validate.snapshot(),
            "demand_evaluate": self.demand_evaluate.snapshot(),
            "backing_claim": self.backing_claim.snapshot(),
            "logical_capacity_claim": self.logical_capacity_claim.snapshot(),
            "transaction_validate_and_fingerprint": self.transaction_validate_and_fingerprint.snapshot(),
            "frame_capture_and_lease": self.frame_capture_and_lease.snapshot(),
        })
    }

    fn reset(&self) {
        for metrics in [
            &self.authority_and_policy_validate,
            &self.demand_evaluate,
            &self.backing_claim,
            &self.logical_capacity_claim,
            &self.transaction_validate_and_fingerprint,
            &self.frame_capture_and_lease,
        ] {
            metrics.reset();
        }
    }
}

#[derive(Default)]
struct VNextEnqueueCallMetrics {
    wall: AtomicDurationMetrics,
    thread_cpu: AtomicDurationMetrics,
    thread_cpu_paired_wall: AtomicDurationMetrics,
    thread_cpu_unavailable_samples: AtomicU64,
}

impl VNextEnqueueCallMetrics {
    fn record(&self, wall: Duration, thread_cpu: Option<Duration>) {
        self.wall.record(wall);
        if let Some(cpu) = thread_cpu {
            self.thread_cpu.record(cpu);
            self.thread_cpu_paired_wall.record(wall);
        } else {
            self.thread_cpu_unavailable_samples
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> serde_json::Value {
        let thread_cpu = self.thread_cpu.snapshot();
        let has_cpu_samples = thread_cpu["samples"].as_u64().is_some_and(|n| n > 0);
        serde_json::json!({
            "wall": self.wall.snapshot(),
            "thread_cpu": has_cpu_samples.then_some(thread_cpu),
            "thread_cpu_paired_wall": has_cpu_samples.then(|| self.thread_cpu_paired_wall.snapshot()),
            "thread_cpu_unavailable_samples": self.thread_cpu_unavailable_samples.load(Ordering::Relaxed),
        })
    }

    fn reset(&self) {
        self.wall.reset();
        self.thread_cpu.reset();
        self.thread_cpu_paired_wall.reset();
        self.thread_cpu_unavailable_samples
            .store(0, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct VNextWaveTimingMetrics {
    resource_prepare_attempt: AtomicDurationMetrics,
    resource_step_request_prepare: AtomicDurationMetrics,
    resource_step_admission: AtomicDurationMetrics,
    resource_step_admission_breakdown: VNextStepAdmissionTimingMetrics,
    resource_submission_wave_prepare: AtomicDurationMetrics,
    host_encode_submit: AtomicDurationMetrics,
    token_upload_prepare: AtomicDurationMetrics,
    wave_identity_bind: AtomicDurationMetrics,
    provider_encode_submit: AtomicDurationMetrics,
    contract_validate_reserve: AtomicDurationMetrics,
    backing_input_encode: AtomicDurationMetrics,
    provider_node_encode: AtomicDurationMetrics,
    segment_hit_provider_node_encode: AtomicDurationMetrics,
    segment_binding_prepare_encode: AtomicDurationMetrics,
    segment_fresh_authority_windows: AtomicDurationMetrics,
    segment_backing_permit_metadata: AtomicDurationMetrics,
    segment_backing_snapshot_lookup: AtomicDurationMetrics,
    segment_backing_dedup_reserve_pool_resolution: AtomicDurationMetrics,
    segment_backing_lock_acquisition: AtomicDurationMetrics,
    segment_backing_locked_validation: AtomicDurationMetrics,
    segment_backing_window_intersections: AtomicDurationMetrics,
    segment_backing_immutable_metadata_validation: AtomicDurationMetrics,
    segment_node_dependencies_regions: AtomicDurationMetrics,
    segment_backend_encode_validate: AtomicDurationMetrics,
    node_identity_materialize: AtomicDurationMetrics,
    node_invocation_construct: AtomicDurationMetrics,
    provider_dynamic_binding_encode: AtomicDurationMetrics,
    binding_validate_coalesce: AtomicDurationMetrics,
    lane_reserve_submit_arm: AtomicDurationMetrics,
    lane_reserve: AtomicDurationMetrics,
    device_runtime_submit: AtomicDurationMetrics,
    device_submit_validate_prepare: AtomicDurationMetrics,
    device_submit_begin_timing: AtomicDurationMetrics,
    device_submit_enqueue_commands: AtomicDurationMetrics,
    device_enqueue_graph_launch: VNextEnqueueCallMetrics,
    device_enqueue_eager_command: VNextEnqueueCallMetrics,
    device_enqueue_deferred_snapshot_readback: VNextEnqueueCallMetrics,
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
            "resource_prepare_breakdown": {
                "collection": "profile_attached_only",
                "step_request_prepare": self.resource_step_request_prepare.snapshot(),
                "step_admission": self.resource_step_admission.snapshot(),
                "step_admission_breakdown": self.resource_step_admission_breakdown.snapshot(),
                "submission_wave_prepare": self.resource_submission_wave_prepare.snapshot(),
            },
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
                    "segment_hit_provider_node_encode": self.segment_hit_provider_node_encode.snapshot(),
                    "provider_node_encode_breakdown": {
                        "collection": "profile_attached_only",
                        "identity_materialize": self.node_identity_materialize.snapshot(),
                        "invocation_construct": self.node_invocation_construct.snapshot(),
                        "dynamic_binding_encode": self.provider_dynamic_binding_encode.snapshot(),
                        "binding_validate_coalesce": self.binding_validate_coalesce.snapshot(),
                        "segment_binding_prepare_encode": self.segment_binding_prepare_encode.snapshot(),
                        "segment_binding_prepare_encode_breakdown": self.segment_binding_prepare_encode_breakdown(),
                    },
                    "lane_reserve_submit_arm": self.lane_reserve_submit_arm.snapshot(),
                    "lane_reserve_submit_arm_breakdown": {
                        "lane_reserve": self.lane_reserve.snapshot(),
                        "device_runtime_submit": self.device_runtime_submit.snapshot(),
                        "device_runtime_submit_breakdown": {
                            "validate_and_prepare": self.device_submit_validate_prepare.snapshot(),
                            "begin_timing": self.device_submit_begin_timing.snapshot(),
                            "enqueue_commands": self.device_submit_enqueue_commands.snapshot(),
                            "enqueue_commands_breakdown": self.enqueue_commands_breakdown(),
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
                "resource_prepare breakdown is collected only while a typed profile sink is attached",
                "resource_prepare breakdown samples low-level admission attempts; retries may produce more breakdown samples than outer resource_prepare_attempt samples",
                "step_admission breakdown records completed admission phases; an error returned inside a phase remains in the outer step_admission interval",
                "resource_prepare residual includes caller-side participant construction and orchestration not attributed to the three child intervals",
                "host_encode_submit breakdown is collected only while a typed profile sink is attached",
                "provider_encode_submit breakdown covers contract validation and completion reservation, backing/input encoding, provider node encoding, and lane reserve/submit/arm",
                "provider_node_encode children sample node or patch operations, not waves; compare total_ns over parent wave samples rather than adding child averages",
                "segment_hit_provider_node_encode is the overlapping subset of fully encoded hit waves, not an additive stage; segment_binding_prepare_encode measures hot-path attempts, including unsupported or failed attempts",
                "segment_binding_prepare_encode children are mutually exclusive host intervals within that parent, not additional work; early returns record only entered stages and panic skips the active stage; compare child totals and sample counts before calculating shares",
                "segment preparation includes pool lock waits, allocations and backend host calls, not pure CPU; errors may include local cleanup in the active child; successful parent residual includes timer transitions, sink recording, final local cleanup and dispatch result map/accounting; snapshots are not atomic",
                "backing_permit_and_metadata children are nested, mutually exclusive host intervals, not additional work; only entered stages have samples and pool-stage observations are recorded after releasing pool guards, so compare counts and totals before calculating shares",
                "identity_materialize includes one bulk identity materialization on eager waves; dynamic_binding_encode covers reusable binding payloads only; eager provider compute encoding and command assembly remain in the parent residual",
                "binding_validate_coalesce includes per-node binding validation and the final exact-layout coverage and backend coalescing pass; child timers include elapsed work before an error but skip unwinding",
                "lane_reserve_submit_arm breakdown isolates lane acquisition, DeviceRuntime::submit, and successful completion arming; failed submissions do not emit completion_arm",
                "device_runtime_submit breakdown isolates backend validation/preparation, timing start, ordered command enqueue, and fence/accounting for runtimes that implement typed attribution",
                "enqueue_commands children are nested per-call host intervals, not additional wave time; optional thread_cpu and its paired wall use only calls with valid CPU samples; unavailable includes clocks not collected; these clocks are not additive, and their difference cannot identify driver waits or GPU idle; snapshots are not atomic",
                "completion_round_trip includes async queue wait, device fence wait, and readback",
                "these host intervals are not kernel or device-busy time"
            ],
        })
    }

    fn enqueue_commands_breakdown(&self) -> serde_json::Value {
        serde_json::json!({
            "collection": "profile_attached_only",
            "scope": "per_call_nested_in_enqueue_commands",
            "wall_clock": "host_monotonic",
            "thread_cpu_clock": "calling_thread_cpu_time_when_available",
            "graph_launch": self.device_enqueue_graph_launch.snapshot(),
            "eager_command": self.device_enqueue_eager_command.snapshot(),
            "deferred_snapshot_readback": self.device_enqueue_deferred_snapshot_readback.snapshot(),
        })
    }

    fn segment_binding_prepare_encode_breakdown(&self) -> serde_json::Value {
        serde_json::json!({
            "collection": "profile_attached_only",
            "fresh_authority_and_windows": self.segment_fresh_authority_windows.snapshot(),
            "backing_permit_and_metadata": self.segment_backing_permit_metadata.snapshot(),
            "backing_permit_and_metadata_breakdown": self.segment_backing_permit_metadata_breakdown(),
            "node_dependencies_and_regions": self.segment_node_dependencies_regions.snapshot(),
            "backend_encode_and_validate": self.segment_backend_encode_validate.snapshot(),
        })
    }

    fn segment_backing_permit_metadata_breakdown(&self) -> serde_json::Value {
        serde_json::json!({
            "collection": "profile_attached_only",
            "snapshot_lookup": self.segment_backing_snapshot_lookup.snapshot(),
            "dedup_reserve_and_pool_resolution": self.segment_backing_dedup_reserve_pool_resolution.snapshot(),
            "lock_acquisition": self.segment_backing_lock_acquisition.snapshot(),
            "locked_validation": self.segment_backing_locked_validation.snapshot(),
            "window_intersections": self.segment_backing_window_intersections.snapshot(),
            "immutable_metadata_validation": self.segment_backing_immutable_metadata_validation.snapshot(),
        })
    }

    fn reset(&self) {
        for metrics in [
            &self.resource_prepare_attempt,
            &self.resource_step_request_prepare,
            &self.resource_step_admission,
            &self.resource_submission_wave_prepare,
            &self.host_encode_submit,
            &self.token_upload_prepare,
            &self.wave_identity_bind,
            &self.provider_encode_submit,
            &self.contract_validate_reserve,
            &self.backing_input_encode,
            &self.provider_node_encode,
            &self.segment_hit_provider_node_encode,
            &self.segment_binding_prepare_encode,
            &self.segment_fresh_authority_windows,
            &self.segment_backing_permit_metadata,
            &self.segment_backing_snapshot_lookup,
            &self.segment_backing_dedup_reserve_pool_resolution,
            &self.segment_backing_lock_acquisition,
            &self.segment_backing_locked_validation,
            &self.segment_backing_window_intersections,
            &self.segment_backing_immutable_metadata_validation,
            &self.segment_node_dependencies_regions,
            &self.segment_backend_encode_validate,
            &self.node_identity_materialize,
            &self.node_invocation_construct,
            &self.provider_dynamic_binding_encode,
            &self.binding_validate_coalesce,
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
        self.resource_step_admission_breakdown.reset();
        self.device_enqueue_graph_launch.reset();
        self.device_enqueue_eager_command.reset();
        self.device_enqueue_deferred_snapshot_readback.reset();
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
    warmup_required_segments: AtomicU64,
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
        self.warmup_required_segments
            .fetch_add(observation.warmup_required_segments(), Ordering::Relaxed);
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
            "warmup_required_segments": self.warmup_required_segments.load(Ordering::Relaxed),
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
            &self.warmup_required_segments,
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

    fn record_device_enqueue(
        &self,
        stage: DeviceEnqueueStage,
        wall: Duration,
        thread_cpu: Option<Duration>,
    ) {
        match stage {
            DeviceEnqueueStage::GraphLaunch => &self.device_enqueue_graph_launch,
            DeviceEnqueueStage::EagerCommand => &self.device_enqueue_eager_command,
            DeviceEnqueueStage::DeferredSnapshotReadback => {
                &self.device_enqueue_deferred_snapshot_readback
            }
        }
        .record(wall, thread_cpu);
    }

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
            SubmissionWaveDispatchStage::SegmentHitProviderNodeEncode => {
                self.segment_hit_provider_node_encode.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentBindingPrepareAndEncode => {
                self.segment_binding_prepare_encode.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentFreshAuthorityAndWindows => {
                self.segment_fresh_authority_windows.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata => {
                self.segment_backing_permit_metadata.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentBackingSnapshotLookup => {
                self.segment_backing_snapshot_lookup.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentBackingDedupReserveAndPoolResolution => self
                .segment_backing_dedup_reserve_pool_resolution
                .record(elapsed),
            SubmissionWaveDispatchStage::SegmentBackingLockAcquisition => {
                self.segment_backing_lock_acquisition.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentBackingLockedValidation => {
                self.segment_backing_locked_validation.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentBackingWindowIntersections => {
                self.segment_backing_window_intersections.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentBackingImmutableMetadataValidation => self
                .segment_backing_immutable_metadata_validation
                .record(elapsed),
            SubmissionWaveDispatchStage::SegmentNodeDependenciesAndRegions => {
                self.segment_node_dependencies_regions.record(elapsed)
            }
            SubmissionWaveDispatchStage::SegmentBackendEncodeAndValidate => {
                self.segment_backend_encode_validate.record(elapsed)
            }
            SubmissionWaveDispatchStage::NodeIdentityMaterialize => {
                self.node_identity_materialize.record(elapsed)
            }
            SubmissionWaveDispatchStage::NodeInvocationConstruct => {
                self.node_invocation_construct.record(elapsed)
            }
            SubmissionWaveDispatchStage::ProviderDynamicBindingEncode => {
                self.provider_dynamic_binding_encode.record(elapsed)
            }
            SubmissionWaveDispatchStage::BindingValidateAndCoalesce => {
                self.binding_validate_coalesce.record(elapsed)
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

    fn record_device_enqueue(
        &self,
        stage: DeviceEnqueueStage,
        wall: Duration,
        thread_cpu: Option<Duration>,
    ) {
        self.aggregate
            .record_device_enqueue(stage, wall, thread_cpu);
        self.phase.record_device_enqueue(stage, wall, thread_cpu);
    }

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
struct VNextPhysicalSpanDurationSummary {
    samples: u64,
    total_ns: u64,
    max_ns: u64,
}

impl VNextPhysicalSpanDurationSummary {
    fn record(&mut self, elapsed_ns: u64) {
        self.samples = self.samples.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(elapsed_ns);
        self.max_ns = self.max_ns.max(elapsed_ns);
    }

    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "samples": self.samples,
            "total_ns": self.total_ns,
            "average_us": if self.samples == 0 {
                0.0
            } else {
                self.total_ns as f64 / self.samples as f64 / 1_000.0
            },
            "max_us": self.max_ns as f64 / 1_000.0,
        })
    }
}

#[derive(Default)]
struct VNextPhysicalSpanTimingMetrics {
    measured_submissions: AtomicU64,
    unavailable_submissions: AtomicU64,
    eager_commands: AtomicDurationMetrics,
    reusable_executables: AtomicDurationMetrics,
    unavailable_spans: AtomicU64,
    reusable_without_fingerprint: AtomicU64,
    fingerprint_capacity_exhausted: AtomicU64,
    reusable_by_fingerprint: Mutex<BTreeMap<String, VNextPhysicalSpanDurationSummary>>,
}

impl VNextPhysicalSpanTimingMetrics {
    fn record(&self, measurement: &DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>) {
        let timing = match measurement {
            DeviceTimingMeasurement::Measured(timing) => {
                self.measured_submissions.fetch_add(1, Ordering::Relaxed);
                timing
            }
            DeviceTimingMeasurement::Unavailable(_) => {
                self.unavailable_submissions.fetch_add(1, Ordering::Relaxed);
                return;
            }
            DeviceTimingMeasurement::NotRequested => return,
        };
        for span in timing.spans() {
            let Some(elapsed_ns) = span.measurement().elapsed_ns() else {
                self.unavailable_spans.fetch_add(1, Ordering::Relaxed);
                continue;
            };
            match span.kind() {
                DeviceExecutionSpanKind::EagerCommand => {
                    self.eager_commands.record(Duration::from_nanos(elapsed_ns));
                }
                DeviceExecutionSpanKind::ReusableExecutable => {
                    self.reusable_executables
                        .record(Duration::from_nanos(elapsed_ns));
                    let Some(fingerprint) = span.reusable_executable_fingerprint() else {
                        self.reusable_without_fingerprint
                            .fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let mut summaries = self.reusable_by_fingerprint.lock();
                    if let Some(summary) = summaries.get_mut(fingerprint) {
                        summary.record(elapsed_ns);
                    } else if summaries.len() < MAX_PROFILED_REUSABLE_EXECUTABLES {
                        let mut summary = VNextPhysicalSpanDurationSummary::default();
                        summary.record(elapsed_ns);
                        summaries.insert(fingerprint.to_owned(), summary);
                    } else {
                        self.fingerprint_capacity_exhausted
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    fn snapshot(&self) -> serde_json::Value {
        let reusable_by_fingerprint = self
            .reusable_by_fingerprint
            .lock()
            .iter()
            .map(|(fingerprint, summary)| {
                serde_json::json!({
                    "reusable_executable_fingerprint": fingerprint,
                    "timing": summary.snapshot(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "scope": "physical_execution_span",
            "measured_submissions": self.measured_submissions.load(Ordering::Relaxed),
            "unavailable_submissions": self.unavailable_submissions.load(Ordering::Relaxed),
            "eager_commands": self.eager_commands.snapshot(),
            "reusable_executables": self.reusable_executables.snapshot(),
            "unavailable_spans": self.unavailable_spans.load(Ordering::Relaxed),
            "reusable_without_fingerprint": self.reusable_without_fingerprint.load(Ordering::Relaxed),
            "fingerprint_capacity": MAX_PROFILED_REUSABLE_EXECUTABLES,
            "fingerprint_capacity_exhausted": self.fingerprint_capacity_exhausted.load(Ordering::Relaxed),
            "reusable_by_fingerprint": reusable_by_fingerprint,
        })
    }

    fn reset(&self) {
        self.measured_submissions.store(0, Ordering::Relaxed);
        self.unavailable_submissions.store(0, Ordering::Relaxed);
        self.eager_commands.reset();
        self.reusable_executables.reset();
        self.unavailable_spans.store(0, Ordering::Relaxed);
        self.reusable_without_fingerprint
            .store(0, Ordering::Relaxed);
        self.fingerprint_capacity_exhausted
            .store(0, Ordering::Relaxed);
        self.reusable_by_fingerprint.lock().clear();
    }
}

#[derive(Default)]
struct VNextDeviceTimingMetrics {
    device_execution: AtomicDurationMetrics,
    fence_wait_host: AtomicDurationMetrics,
    readback_host: AtomicDurationMetrics,
    physical_spans: VNextPhysicalSpanTimingMetrics,
    readback_calls: AtomicU64,
    readback_bytes: AtomicU64,
    device_unavailable: AtomicU64,
    fence_wait_unavailable: AtomicU64,
    readback_unavailable: AtomicU64,
}

impl VNextDeviceTimingMetrics {
    fn record(&self, receipt: &CompletionReadbackBatchReceipt) {
        let fence = receipt.completion().fence_timing();
        self.physical_spans
            .record(receipt.completion().submission_timing());
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
            "physical_submission_spans": self.physical_spans.snapshot(),
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
        self.physical_spans.reset();
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

impl VNextExecutorMetrics {
    fn wave_timing_for(&self, kind: VNextExecutionWaveKind) -> &VNextWaveTimingMetrics {
        match kind {
            VNextExecutionWaveKind::Prefill => &self.prefill_wave_timing,
            VNextExecutionWaveKind::Decode => &self.decode_wave_timing,
            VNextExecutionWaveKind::Mixed => &self.mixed_wave_timing,
        }
    }

    fn device_timing_for(&self, kind: VNextExecutionWaveKind) -> &VNextDeviceTimingMetrics {
        match kind {
            VNextExecutionWaveKind::Prefill => &self.prefill_device_timing,
            VNextExecutionWaveKind::Decode => &self.decode_device_timing,
            VNextExecutionWaveKind::Mixed => &self.mixed_device_timing,
        }
    }

    fn record_failure(&self, message: impl Into<String>) {
        self.failed_waves.fetch_add(1, Ordering::Relaxed);
        *self.last_failure.lock() = Some(message.into());
    }

    fn record_reusable_catalog_miss(&self, key: VNextReusableExecutionCatalogMissKey) {
        if key.reason.is_epoch_mismatch() {
            self.reusable_catalog_epoch_misses
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.reusable_catalog_misses.fetch_add(1, Ordering::Relaxed);
        }
        self.reusable_catalog_miss_ledger.lock().record(key);
    }

    fn average_ms(total_us: u64, operations: u64) -> f64 {
        if operations == 0 {
            0.0
        } else {
            total_us as f64 / operations as f64 / 1000.0
        }
    }

    fn reset_after_startup(&self) {
        self.prefix_cache.reset();
        self.invocation_preparation.reset();
        for counter in [
            &self.prefill_operations,
            &self.prefill_frontier_narrowings,
            &self.decode_operations,
            &self.submitted_waves,
            &self.completed_waves,
            &self.failed_waves,
            &self.direct_reusable_waves,
            &self.direct_reusable_segments,
            &self.direct_reusable_logical_nodes,
            &self.direct_reusable_binding_nodes,
            &self.direct_reusable_fallbacks,
            &self.reusable_catalog_misses,
            &self.reusable_catalog_epoch_misses,
            &self.identity_waves,
            &self.identity_logical_nodes,
            &self.identity_nodes_materialized_before_submit,
            &self.identity_full_participant_materializations_before_submit,
            &self.definitely_not_submitted_retries,
            &self.request_deferrals,
            &self.sequence_deferrals,
            &self.extension_deferrals,
            &self.step_deferrals,
            &self.wave_deferrals,
            &self.backing_deferrals,
            &self.uploaded_bytes,
            &self.readback_bytes,
            &self.full_logits_readback_waves,
            &self.greedy_token_readback_waves,
            &self.hybrid_readback_waves,
            &self.full_logits_readback_participants,
            &self.greedy_token_readback_participants,
            &self.no_selectable_finite_logit_participants,
            &self.product_readback_bytes,
            &self.greedy_policy_fallback_waves,
            &self.token_mask_upload_participants,
            &self.token_mask_cache_hit_participants,
            &self.sparse_repetition_waves,
            &self.sparse_repetition_participants,
            &self.sparse_repetition_token_ids_uploaded,
            &self.total_prefill_us,
            &self.total_decode_us,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        self.wave_timing.reset();
        self.prepared_wave_topology.reset();
        self.reusable_catalog_miss_ledger.lock().reset();
        self.reusable_cold_path.reset();
        self.prefill_wave_timing.reset();
        self.decode_wave_timing.reset();
        self.mixed_wave_timing.reset();
        self.device_timing.reset();
        self.prefill_device_timing.reset();
        self.decode_device_timing.reset();
        self.mixed_device_timing.reset();
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

const VNEXT_MONOTONIC_CLOCK_SOURCE: &str = "rust_std_instant";

const fn journal_clock_anchor_required(timing_mode: DeviceTimingMode) -> bool {
    matches!(timing_mode, DeviceTimingMode::Kernel)
}

fn system_time_unix_nanos(time: SystemTime) -> std::result::Result<i64, ExecutionEventSinkError> {
    let duration = time.duration_since(UNIX_EPOCH).map_err(|error| {
        ExecutionEventSinkError::contract(format!("system clock predates Unix epoch: {error}"))
    })?;
    i64::try_from(duration.as_nanos())
        .map_err(|_| ExecutionEventSinkError::contract("system clock Unix nanos exceed i64"))
}

fn bounded_wall_anchor(wall_before: i64, wall_after: i64) -> (i64, u64) {
    let lower = wall_before.min(wall_after);
    let upper = wall_before.max(wall_after);
    let span = upper.saturating_sub(lower);
    (
        lower.saturating_add(span / 2),
        u64::try_from(span).unwrap_or(u64::MAX),
    )
}

fn capture_monotonic_wall_anchor(
) -> std::result::Result<(Instant, i64, u64), ExecutionEventSinkError> {
    let wall_before = system_time_unix_nanos(SystemTime::now())?;
    let started = Instant::now();
    let wall_after = system_time_unix_nanos(SystemTime::now())?;
    let (wall_anchor_unix_nanos, max_error_nanos) = bounded_wall_anchor(wall_before, wall_after);
    Ok((started, wall_anchor_unix_nanos, max_error_nanos))
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
    first_failure: Option<IdentifiedFailure>,
}

impl VNextExecutionJournal {
    fn error(error: impl fmt::Display) -> ExecutionEventSinkError {
        ExecutionEventSinkError::contract(error.to_string())
    }

    fn open(
        sink: Arc<dyn ExecutionEventSink>,
        plan: &ExecutionPlan,
        active: Arc<TrustedActiveSequenceBinding>,
        request_origin: ExecutorRequestOrigin,
    ) -> std::result::Result<Self, ExecutionEventSinkError> {
        let clock_anchor = journal_clock_anchor_required(sink.device_timing_mode())
            .then(capture_monotonic_wall_anchor)
            .transpose()?;
        let started = clock_anchor
            .as_ref()
            .map_or_else(Instant::now, |(started, _, _)| *started);
        let topology = TrustedExecutionTopology::from_plan(plan).map_err(Self::error)?;
        let root_span =
            SpanId::new(format!("vnext/request/{}", active.fingerprint())).map_err(Self::error)?;
        let capture_policy = sink.capture_policy_for_request(request_origin);
        let mut journal = Self {
            emitter: ExecutionEventEmitter::from_shared_with_capture_policy(
                sink,
                active.run_id().clone(),
                active.request_id().clone(),
                capture_policy,
            ),
            topology,
            active,
            capture_policy,
            completed_frames: 0,
            started,
            last_timestamp_nanos: 0,
            root_span,
            pending_submission: None,
            first_failure: None,
        };
        let accepted_detail = clock_anchor.map_or(
            ExecutionEventDetail::None,
            |(_, wall_anchor_unix_nanos, max_error_nanos)| {
                ExecutionEventDetail::MonotonicClockAnchor {
                    clock_source: VNEXT_MONOTONIC_CLOCK_SOURCE.to_string(),
                    wall_anchor_unix_nanos,
                    max_error_nanos,
                }
            },
        );
        let accepted = journal.event(
            ExecutionPhase::Resolution,
            ExecutionEventKind::RequestAccepted,
            journal.base_parts(1, journal.root_span.clone(), None),
            accepted_detail,
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

    fn observe_resource_failure(
        &mut self,
        snapshot: PlanRuntimeResourceSnapshot,
        code: &str,
        message: &str,
    ) -> std::result::Result<String, ExecutionEventSinkError> {
        if self.first_failure.is_some() {
            return Err(Self::error(
                "execution journal already observed its first failure",
            ));
        }
        let failed_operation = match self.pending_submission.as_ref() {
            Some(JournaledSubmission::Captured { receipt, selected }) => {
                let first = *selected
                    .first()
                    .ok_or_else(|| Self::error("captured submission has no request participant"))?;
                receipt
                    .participants()
                    .get(first)
                    .ok_or_else(|| Self::error("captured failure participant is missing"))?
                    .identity()
                    .clone()
            }
            Some(JournaledSubmission::Suppressed { .. }) => {
                return Err(Self::error(
                    "resource failure attribution requires a captured operation",
                ))
            }
            None => {
                return Err(Self::error(
                    "resource failure attribution requires one submitted operation",
                ))
            }
        };
        let envelope = FailureEnvelope::new(FailureDomain::Resource, code, message, false)
            .and_then(|failure| failure.with_resource_snapshot(snapshot))
            .map_err(Self::error)?;
        let failure =
            IdentifiedFailure::new(failed_operation.clone(), envelope).map_err(Self::error)?;
        let failure_fingerprint = failure.fingerprint();
        let sequence = self.emitter.cursor().last_sequence().saturating_add(1);
        let mut parts = failed_operation.parts().clone();
        parts.sequence = sequence;
        parts.parent_span_id = Some(failed_operation.parts().span_id.clone());
        parts.span_id =
            SpanId::new(format!("{}/failure/{sequence}", self.root_span)).map_err(Self::error)?;
        let event = self.event(
            ExecutionPhase::Execution,
            ExecutionEventKind::FailureObserved,
            parts,
            ExecutionEventDetail::Failure(failure.clone()),
        )?;
        self.emitter.emit(
            event,
            &TrustedExecutionEventContext::failure(
                self.active.run_id(),
                self.active.request_id(),
                Some(&self.topology),
                Some(&self.active),
                &failure,
            ),
        )?;
        self.first_failure = Some(failure);
        Ok(failure_fingerprint)
    }

    fn settle_failed_submission(
        &mut self,
        completion: &OperationCompletionReceipt,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        if self.first_failure.is_none() {
            return Err(Self::error(
                "cannot settle a failed submission before observing its failure",
            ));
        }
        let pending = self
            .pending_submission
            .take()
            .ok_or_else(|| Self::error("failed submission is no longer journaled"))?;
        let JournaledSubmission::Captured { receipt, .. } = pending else {
            return Err(Self::error(
                "failed submission was suppressed from the execution journal",
            ));
        };
        if completion.submission().fingerprint() != receipt.fingerprint() {
            return Err(Self::error(
                "settled failure differs from the journaled physical submission",
            ));
        }
        Ok(())
    }

    fn fail_sequence(
        &mut self,
        receipt: &SequenceSessionTerminalReceipt,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        if self.pending_submission.is_some() {
            return Err(Self::error(
                "sequence failure still has an in-flight journal submission",
            ));
        }
        let failure = self
            .first_failure
            .clone()
            .ok_or_else(|| Self::error("sequence failure lacks its first observed failure"))?;
        let aborted = TrustedAbortedSequenceBinding::from_session_receipt(receipt, &self.active)
            .map_err(Self::error)?;
        let sequence_number = self.emitter.cursor().last_sequence().saturating_add(1);
        let sequence_span =
            SpanId::new(format!("{}/sequence-aborted", self.root_span)).map_err(Self::error)?;
        let mut parts = self.bind_active(self.bind_plan(self.base_parts(
            sequence_number,
            sequence_span,
            Some(self.root_span.clone()),
        )));
        parts.aborted_sequence_fingerprint = Some(aborted.fingerprint().to_owned());
        let sequence_aborted = self.event(
            ExecutionPhase::Completion,
            ExecutionEventKind::SequenceAborted,
            parts,
            ExecutionEventDetail::None,
        )?;
        self.emitter.emit(
            sequence_aborted,
            &TrustedExecutionEventContext::aborted(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
                &aborted,
            ),
        )?;

        let request_sequence = self.emitter.cursor().last_sequence().saturating_add(1);
        let mut parts = self.bind_active(self.bind_plan(self.base_parts(
            request_sequence,
            self.root_span.clone(),
            None,
        )));
        parts.aborted_sequence_fingerprint = Some(aborted.fingerprint().to_owned());
        let request_failed = self.event(
            ExecutionPhase::Completion,
            ExecutionEventKind::RequestFailed,
            parts,
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint: failure.fingerprint(),
            },
        )?;
        self.emitter.emit(
            request_failed,
            &TrustedExecutionEventContext::failure_with_disposition(
                self.active.run_id(),
                self.active.request_id(),
                &self.topology,
                &self.active,
                None,
                Some(&aborted),
                &failure,
            ),
        )
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
    token_identity: ExecutorSequence,
    token_recovery_key: ExecutionRecoveryKey,
    token_issuer: ExecutorSequenceIssuer,
    prefix_capture_interests:
        Mutex<Vec<std::sync::Weak<prefix_cache::rendezvous::NativePrefixCapture<R>>>>,
    cache_id: String,
    request: Arc<VNextRequestRoot<R>>,
    session: Arc<SequenceSession<R>>,
    active_binding: Arc<TrustedActiveSequenceBinding>,
    request_origin: ExecutorRequestOrigin,
    tokens: Mutex<Vec<u32>>,
    maximum_tokens: usize,
    active: AtomicBool,
    settlement: Mutex<SequenceSettlement<R>>,
    recovery_wake: Arc<completion_worker::VNextRecoveryWake>,
    operation: AsyncMutex<()>,
    events: Option<Mutex<VNextExecutionJournal>>,
    product_prompt_tokens: u64,
    replayed_output_tokens: u64,
    prefill_tokens_processed: AtomicUsize,
}

struct PreparedVNextPrefill<R: DeviceRuntime> {
    retirement: PreparedRetirement<R>,
    dispatch: PreparedDispatch<R>,
}

struct VNextExecutionParticipant<'a, R: DeviceRuntime> {
    sequence: &'a Arc<VNextSequence<R>>,
    tokens: &'a [u32],
    span: &'a TokenSpanWork,
    output_role: &'a VNextParticipantOutputRole,
}

fn validate_sequence_completion_accounting(
    request_id: &RequestId,
    product_prompt_tokens: u64,
    replayed_output_tokens: u64,
    completion: &ExecutorSequenceCompletion,
) -> Result<()> {
    if completion.request_id() != request_id {
        return Err(FerrumError::request_validation(format!(
            "vNext completion request `{}` differs from cache owner `{request_id}`",
            completion.request_id()
        )));
    }
    if completion.input_tokens() != product_prompt_tokens {
        return Err(FerrumError::request_validation(format!(
            "vNext completion input count {} differs from admitted product prompt count {product_prompt_tokens}",
            completion.input_tokens()
        )));
    }
    if completion.output_tokens() < replayed_output_tokens {
        return Err(FerrumError::request_validation(format!(
            "vNext completion output count {} precedes recompute baseline {replayed_output_tokens}",
            completion.output_tokens()
        )));
    }
    Ok(())
}

/// Own cancellation while the registry retains this closing incarnation.
/// Other already-admitted callers can still hold Arcs, so the sequence's final
/// Drop cannot be relied upon to cancel an abandoned completion future.
struct PendingSequenceCompletion<'a, R: DeviceRuntime> {
    sequence: &'a VNextSequence<R>,
    operation: Option<tokio::sync::MutexGuard<'a, ()>>,
    completed: bool,
}

impl<R: DeviceRuntime> Drop for PendingSequenceCompletion<'_, R> {
    fn drop(&mut self) {
        if !self.completed {
            // Cancel before the operation guard is released. An already-owned
            // decode caller must see an inactive sequence when it next enters.
            self.sequence.abort();
        }
    }
}

impl<R: DeviceRuntime> VNextSequence<R> {
    fn request_id(&self) -> &RequestId {
        self.request.product_request_id()
    }

    fn preempt_for_recompute(&self) -> Result<()> {
        if !self.active.load(Ordering::Acquire) {
            return Err(FerrumError::already_exists(format!(
                "vNext request `{}` is already terminal",
                self.request_id()
            )));
        }
        let receipt = self
            .session
            .try_abort_if_quiescent()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        self.active.store(false, Ordering::Release);
        self.record_token_terminal(receipt);
        Ok(())
    }

    fn complete(&self, completion: &ExecutorSequenceCompletion) -> Result<()> {
        if let Err(error) = validate_sequence_completion_accounting(
            self.request_id(),
            self.product_prompt_tokens,
            self.replayed_output_tokens,
            completion,
        ) {
            self.abort();
            return Err(error);
        }
        self.complete_with_counts(completion.input_tokens(), completion.output_tokens())
    }

    fn complete_startup(&self) -> Result<()> {
        self.complete_with_counts(self.product_prompt_tokens, self.replayed_output_tokens)
    }

    fn complete_with_counts(&self, input_tokens: u64, output_tokens: u64) -> Result<()> {
        if self.settlement.lock().terminal.is_some() {
            return Err(FerrumError::already_exists(format!(
                "vNext request `{}` is already terminal",
                self.request_id()
            )));
        }
        self.begin_closing();
        let receipt = self
            .session
            .try_complete()
            .map_err(|error| FerrumError::backend(format!("vNext sequence completion: {error}")))?;
        self.record_token_terminal(receipt.clone());
        if let Some(events) = &self.events {
            events
                .lock()
                .complete_sequence(&receipt, input_tokens, output_tokens)
                .map_err(|error| {
                    FerrumError::backend(format!("vNext execution journal completion: {error}"))
                })?;
        }
        Ok(())
    }

    fn abort(&self) {
        self.begin_closing();
        self.settlement.lock().cancel_requested = true;
        // Only the real lower terminal receipt permits registry removal.
        // A held frame/transfer remains owned for the independent worker.
        let _ = self.try_end_cancelled();
        self.recovery_wake.request();
    }

    fn abort_after_observed_failure(&self) -> Result<()> {
        if !self.active.swap(false, Ordering::AcqRel) {
            return Err(FerrumError::already_exists(format!(
                "vNext request `{}` is already terminal",
                self.request_id()
            )));
        }
        self.session.request_cancel().map_err(|error| {
            FerrumError::backend(format!("vNext failure cancellation: {error}"))
        })?;
        let receipt = self
            .session
            .try_abort()
            .map_err(|error| FerrumError::backend(format!("vNext failure abort: {error}")))?;
        self.record_token_terminal(receipt.clone());
        self.events
            .as_ref()
            .ok_or_else(|| {
                FerrumError::internal(
                    "vNext diagnostic failure requires an execution event journal",
                )
            })?
            .lock()
            .fail_sequence(&receipt)
            .map_err(|error| {
                FerrumError::backend(format!("vNext execution journal failure: {error}"))
            })
    }
}

impl<R: DeviceRuntime> Drop for VNextSequence<R> {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        let _ = self.session.request_cancel();
        let _ = self.session.try_abort();
    }
}

enum DispatchOutcome<R: DeviceRuntime> {
    Submitted {
        completion: CompletionHandle<R>,
        observation: SubmissionObservation,
    },
    QuiescentFailure(FerrumError),
    SubmissionIndeterminate {
        message: String,
        recovery: IndeterminateSubmissionHandle<R>,
    },
    PostSubmitContract {
        message: String,
        completion: CompletionHandle<R>,
        observation: SubmissionObservation,
    },
}

enum VNextExecutionCapacityDecision<T> {
    Ready(T),
    Deferred(ExecutorExecutionCapacityDeferral),
    RequestStateDeferred(ExecutorRequestStateDeferral),
}

enum VNextExecutionMaintenanceSource<'a> {
    Logical(&'a AdmissionDeferred),
    Backing(&'a DynamicBackingDeferred),
}

enum VNextSequenceAdmissionDecision<R: DeviceRuntime> {
    Admitted(Arc<SequenceSession<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(InitialSequenceBackingDeferral<R>),
    PermanentRejected(AdmissionRejected),
}

enum PendingPrefillMaintenance<R: DeviceRuntime> {
    Logical(AdmissionDeferred),
    Backing(InitialSequenceBackingDeferral<R>),
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
    admission: Mutex<Option<token_execution::AdmissionBinding>>,
}

impl<R: DeviceRuntime> VNextPrefillSlot<R> {
    fn new(request_id: RequestId, work_shape: ResourceWorkShape) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            work_shape,
            cancelled: AtomicBool::new(false),
            state: Mutex::new(VNextPrefillSlotState::Probing),
            admission: Mutex::new(None),
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
    // Identity lookup only; the existing prefill/active entries own resources.
    token_index: HashMap<ExecutorSequence, Weak<VNextSequence<R>>>,
    recovery_wake: Option<Arc<completion_worker::VNextRecoveryWake>>,
}

impl<R: DeviceRuntime> Default for VNextSequenceRegistry<R> {
    fn default() -> Self {
        Self {
            prefills: HashMap::new(),
            active: HashMap::new(),
            token_index: HashMap::new(),
            recovery_wake: None,
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
            .any(|sequence| sequence.request_id() == request_id)
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

    fn begin_prefill_batch_execution(
        &mut self,
        request_ids: &[RequestId],
    ) -> Result<Vec<(Arc<VNextPrefillSlot<R>>, Arc<VNextSequence<R>>)>> {
        let mut prepared = Vec::with_capacity(request_ids.len());
        for request_id in request_ids {
            if prepared.iter().any(
                |(slot, _): &(Arc<VNextPrefillSlot<R>>, Arc<VNextSequence<R>>)| {
                    slot.request_id == *request_id
                },
            ) {
                return Err(FerrumError::request_validation(
                    "vNext batch prefill inputs contain a duplicate request",
                ));
            }
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
            let sequence = match &*slot.state.lock() {
                VNextPrefillSlotState::Ready(sequence) => Arc::clone(sequence),
                _ => {
                    return Err(FerrumError::request_validation(format!(
                        "vNext prefill for `{request_id}` is not ready for batch execution"
                    )))
                }
            };
            prepared.push((slot, sequence));
        }

        let mut states = prepared
            .iter()
            .map(|(slot, _)| slot.state.lock())
            .collect::<Vec<_>>();
        if states.iter().zip(&prepared).any(|(state, (_, sequence))| {
            !matches!(
                &**state,
                VNextPrefillSlotState::Ready(current) if Arc::ptr_eq(current, sequence)
            )
        }) {
            return Err(FerrumError::internal(
                "vNext prefill authorities changed during atomic batch acquisition",
            ));
        }
        for (state, (_, sequence)) in states.iter_mut().zip(&prepared) {
            **state = VNextPrefillSlotState::Executing(Arc::clone(sequence));
        }
        drop(states);
        Ok(prepared)
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
        if let VNextPrefillSlotState::Ready(sequence) = &*state {
            let sequence = Arc::clone(sequence);
            drop(state);
            sequence.abort();
            self.remove_ended_exact(&sequence);
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
                .filter(|sequence| sequence.request_id() == preemption.request_id())
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
            self.token_index.remove(&sequence.token_identity);
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
                    .find(|sequence| sequence.request_id() == request_id)
                    .map_or_else(
                        || "no active sequence".to_string(),
                        |sequence| format!("active cache is {}", sequence.cache_id),
                    );
                FerrumError::request_validation(format!(
                    "vNext request `{request_id}` preemption did not match an authority: {detail}"
                ))
            })?;
        if sequence.request_id() != request_id {
            return Err(FerrumError::request_validation(format!(
                "vNext cache `{}` belongs to request {}, not {request_id}",
                preemption.cache_id(),
                sequence.request_id()
            )));
        }
        sequence.preempt_for_recompute()?;
        self.active.remove(preemption.cache_id());
        self.token_index.remove(&sequence.token_identity);
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
        let state = slot.state.lock();
        if !matches!(
            &*state,
            VNextPrefillSlotState::Executing(current) if Arc::ptr_eq(current, sequence)
        ) {
            return;
        }
        drop(state);
        sequence.abort();
        self.remove_ended_exact(sequence);
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
pub struct ProgramExecutor<R: DeviceRuntime> {
    token_issuer: ExecutorSequenceIssuer,
    info: ModelInfo,
    resolved_plan: ResolvedModelPlan,
    capability_catalog: CapabilityCatalog,
    runtime: Arc<R>,
    providers: BoundOperationProviderSet<R>,
    policy: ResolvedRuntimePolicy,
    invocation_preparation_strategy: InvocationPreparationStrategy,
    completion_readback_policy: ferrum_types::CompletionReadbackPolicy,
    resident_decode: Mutex<Option<ResidentCache<R>>>,
    resident_metrics: ResidentMetrics,
    sampling_readback_mode: SamplingReadbackMode,
    plan_resources: Arc<PlanRuntimeResources<R>>,
    lane: Arc<ExecutionLane<R>>,
    submission_wave_identity: CompiledSubmissionWaveIdentity,
    completion_worker: VNextCompletionWorker,
    reaper: Arc<CompletionReaper<R>>,
    io: VNextIoBinding,
    maximum_model_tokens: usize,
    startup_memory_plan: Option<ferrum_types::StartupMemoryPlan>,
    run_id: RunId,
    family_fingerprint: String,
    program_fingerprint: String,
    static_provider_attribution: Option<StaticProviderAttributionWitness>,
    checkpoint_capture: Option<VNextCheckpointCapture>,
    static_bytes: u64,
    sequence_state_memory: TypedSequenceStateMemory,
    device_reusable_execution_enabled: bool,
    reusable_execution_supported: bool,
    reusable_execution_startup_plan: Option<VNextReusableExecutionStartupPlan>,
    reusable_execution_catalog: RwLock<Option<Arc<VNextReusableExecutionCatalog>>>,
    reusable_execution_catalog_refresh_needed: AtomicBool,
    startup_reusable_programs: Mutex<
        BTreeMap<VNextReusableExecutionDescriptor, BTreeSet<DeviceReusableExecutionProgramId>>,
    >,
    startup_preparation: Mutex<VNextStartupPreparationState>,
    sequences: Arc<Mutex<VNextSequenceRegistry<R>>>,
    prefix_cache: Mutex<prefix_cache::PrefixIndex<SequenceCheckpoint<R>>>,
    prefix_capture_identity: Arc<()>,
    product_token_mask_residency: Mutex<VNextProductTokenMaskResidency>,
    event_sink: RwLock<Option<Arc<dyn ProductExecutionEventSink>>>,
    device_timing_mode: AtomicU8,
    diagnostic_fault: Option<VNextDiagnosticFault>,
    diagnostic_fault_armed: AtomicBool,
    metrics: VNextExecutorMetrics,
}

impl<R: DeviceRuntime> fmt::Debug for ProgramExecutor<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProgramExecutor")
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

struct VNextStartupSequence<R: DeviceRuntime> {
    sequence: Arc<VNextSequence<R>>,
    frontier: ExecutionFrontier,
}

/// Startup owns the same admitted incarnations as ordinary token execution.
/// Keep the actual owner immediately after admission, including intermediate
/// prerequisite prefill; neither request IDs nor copied cache handles authorize
/// cleanup of a later incarnation.
struct VNextStartupSequenceGuard<'executor, R: DeviceRuntime> {
    executor: &'executor ProgramExecutor<R>,
    pending: Option<Arc<VNextPrefillSlot<R>>>,
    sequences: Vec<VNextStartupSequence<R>>,
}

impl<'executor, R: DeviceRuntime> VNextStartupSequenceGuard<'executor, R> {
    fn new(executor: &'executor ProgramExecutor<R>) -> Self {
        Self {
            executor,
            pending: None,
            sequences: Vec::new(),
        }
    }

    fn track_pending(&mut self, request_id: &RequestId) {
        self.pending = self
            .executor
            .sequences
            .lock()
            .prefills
            .get(request_id)
            .cloned();
    }

    fn activate(&mut self, frontier: ExecutionFrontier) -> Result<usize> {
        let sequence = self.executor.sequence_for_token(frontier.sequence())?;
        if sequence.request_origin != ExecutorRequestOrigin::Startup
            || !self.pending.as_ref().is_some_and(|slot| {
                slot.request_id == *sequence.request_id()
                    && matches!(&*slot.state.lock(), VNextPrefillSlotState::Ready(current)
                        if Arc::ptr_eq(current, &sequence))
            })
        {
            return Err(FerrumError::internal(
                "startup admission lost its original sequence owner",
            ));
        }
        let index = self.sequences.len();
        self.sequences
            .push(VNextStartupSequence { sequence, frontier });
        self.pending = None;
        Ok(index)
    }

    fn complete(mut self) -> Result<()> {
        self.cancel_pending();
        for startup in &self.sequences {
            let sequence = &startup.sequence;
            if sequence.token_identity != *startup.frontier.sequence()
                || sequence.request_origin != ExecutorRequestOrigin::Startup
            {
                return Err(FerrumError::internal(
                    "startup completion changed its admitted incarnation",
                ));
            }
            // This emits the original Startup accounting, without product
            // usage or optional product-prefix retention. A journal error may
            // coexist with actual Ended and must still be returned.
            let result = sequence.complete_startup();
            self.executor.sequences.lock().remove_ended_exact(sequence);
            result.map_err(|error| {
                FerrumError::backend(format!("vNext startup sequence completion: {error}"))
            })?;
            if sequence.settlement.lock().terminal.is_none() {
                return Err(FerrumError::backend(
                    "startup completion has no native terminal receipt",
                ));
            }
        }
        self.sequences.clear();
        Ok(())
    }

    fn cancel_pending(&mut self) {
        if let Some(slot) = self.pending.take() {
            let mut registry = self.executor.sequences.lock();
            if registry
                .prefills
                .get(&slot.request_id)
                .is_some_and(|current| Arc::ptr_eq(current, &slot))
            {
                registry.cancel_prefill(&slot.request_id);
            }
        }
    }
}

impl<R: DeviceRuntime> Drop for VNextStartupSequenceGuard<'_, R> {
    fn drop(&mut self) {
        self.cancel_pending();
        for startup in &self.sequences {
            startup.sequence.abort();
            self.executor
                .sequences
                .lock()
                .remove_ended_exact(&startup.sequence);
        }
        // A nonterminal owner remains in the original registry/reaper. Drop
        // initiates cancellation; it does not assert that held work is released.
    }
}

impl<R: DeviceRuntime> ProgramExecutor<R> {
    fn claim_prefill_resource_diagnostic_fault(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        kind: VNextExecutionWaveKind,
    ) -> bool {
        matches!(
            self.diagnostic_fault,
            Some(VNextDiagnosticFault::PrefillResourceAfterSubmitOnce)
        ) && kind == VNextExecutionWaveKind::Prefill
            && matches!(participants, [participant] if participant.sequence.request_origin == ExecutorRequestOrigin::Product)
            && self
                .diagnostic_fault_armed
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
    }

    fn resolve_language_io_ids(program: &ModelProgram) -> Result<VNextLanguageIoIds> {
        let embedding_nodes = program
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .filter(|node| is_language_token_embedding_operation(node.operation_id.as_str()))
            .collect::<Vec<_>>();
        let [embedding] = embedding_nodes.as_slice() else {
            return Err(FerrumError::model(format!(
                "vNext language program requires exactly one token embedding operation, got {}",
                embedding_nodes.len()
            )));
        };
        let token_input = embedding.inputs.first().cloned().ok_or_else(|| {
            FerrumError::model("vNext token embedding operation has no token input")
        })?;

        let argmax_nodes = program
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .filter(|node| is_language_masked_argmax_operation(node.operation_id.as_str()))
            .collect::<Vec<_>>();
        let [argmax] = argmax_nodes.as_slice() else {
            return Err(FerrumError::model(format!(
                "vNext language program requires exactly one masked argmax operation, got {}",
                argmax_nodes.len()
            )));
        };
        let [logits_output, token_mask_input, repetition_token_ids_input, repetition_offsets_input, repetition_penalty_input] =
            argmax.inputs.as_slice()
        else {
            return Err(FerrumError::model(
                "vNext masked argmax operation must consume logits, a token mask, and typed sparse repetition policy",
            ));
        };
        let [greedy_token_output] = argmax.outputs.as_slice() else {
            return Err(FerrumError::model(
                "vNext masked argmax operation must produce one token",
            ));
        };
        let expected_inputs = [
            &token_input,
            token_mask_input,
            repetition_token_ids_input,
            repetition_offsets_input,
            repetition_penalty_input,
        ];
        if program.inputs().len() != expected_inputs.len()
            || expected_inputs
                .iter()
                .any(|expected| !program.inputs().contains(expected))
        {
            return Err(FerrumError::model(
                "vNext language program inputs must expose token ids plus typed selection and repetition policy",
            ));
        }
        let expected_outputs = [logits_output, greedy_token_output];
        if program.outputs().len() != expected_outputs.len()
            || expected_outputs
                .iter()
                .any(|expected| !program.outputs().contains(expected))
        {
            return Err(FerrumError::model(
                "vNext language program outputs must expose full logits and the selected token",
            ));
        }
        Ok(VNextLanguageIoIds {
            token_input,
            token_mask_input: token_mask_input.clone(),
            repetition_token_ids_input: repetition_token_ids_input.clone(),
            repetition_offsets_input: repetition_offsets_input.clone(),
            repetition_penalty_input: repetition_penalty_input.clone(),
            logits_output: logits_output.clone(),
            greedy_token_output: greedy_token_output.clone(),
        })
    }

    pub fn from_runtime_composition<F>(
        prepared: PreparedExecutionInput<'_>,
        info: ModelInfo,
        engine_config: &EngineConfig,
        runtime: Arc<R>,
        registry: OperationRuntimeRegistry<R>,
        weight_materializers: WeightMaterializerRegistry,
        weight_materializer_selection: WeightMaterializerSelection,
        catalog: CapabilityCatalog,
        resolve_plan: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        prepared.validate_info(&info)?;
        let config =
            ProgramExecutorConfig::from_engine_config(engine_config, &info, runtime.as_ref())?;
        Self::from_runtime_composition_with_config(
            prepared,
            info,
            engine_config,
            config,
            runtime,
            registry,
            weight_materializers,
            weight_materializer_selection,
            catalog,
            resolve_plan,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_runtime_composition_with_config<F>(
        prepared: PreparedExecutionInput<'_>,
        info: ModelInfo,
        engine_config: &EngineConfig,
        config: ProgramExecutorConfig,
        runtime: Arc<R>,
        registry: OperationRuntimeRegistry<R>,
        weight_materializers: WeightMaterializerRegistry,
        weight_materializer_selection: WeightMaterializerSelection,
        catalog: CapabilityCatalog,
        resolve_plan: F,
    ) -> Result<Self>
    where
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        let composition = RuntimeComposition::new(runtime, registry, weight_materializers, catalog);
        composition
            .compile_model(
                prepared,
                info,
                engine_config,
                config,
                weight_materializer_selection,
            )?
            .initialize(resolve_plan)
    }

    fn from_compiled_model<F>(compiled: CompiledProgram<'_, R>, resolve_plan: F) -> Result<Self>
    where
        F: FnOnce(
            &ResolvedRuntimePolicy,
            &CapabilityCatalog,
            &ProgramPlanCompilation,
        ) -> Result<ResolvedModelPlan>,
    {
        let CompiledProgram {
            composition,
            prepared,
            info,
            config,
            compilation,
            language_io,
            checkpoint_selection,
            repetition_capacity,
            executor_startup,
        } = compiled;
        let runtime = Arc::clone(&composition.runtime);
        let registry = &composition.registry;
        let catalog = composition.catalog.clone();
        let family = prepared.family();
        let resolve_bind_phase = StartupPhaseTimer::start("plan_resolve_and_bind");
        let resolved_plan = resolve_plan(&config.runtime_policy, &catalog, &compilation)?;
        if resolved_plan.execution_plan() != compilation.executable().execution_plan() {
            return Err(FerrumError::internal(
                "product composition returned a different execution plan than the compiler",
            ));
        }
        prepared.validate_resolved_plan(&resolved_plan)?;
        let sequence_state_memory = state_memory::logical_sequence_state_memory(
            &family.numerical_profile().states,
            &family.numerical_profile().kv_storage,
        )?;
        let providers = registry
            .bind_plan(&resolved_plan)
            .map_err(|error| FerrumError::model(format!("vNext provider binding: {error}")))?;
        resolve_bind_phase.finish();
        let io = Self::resolve_io(
            &resolved_plan,
            &language_io,
            info.vocab_size,
            usize::try_from(repetition_capacity)
                .map_err(|_| FerrumError::config("vNext repetition capacity exceeds usize"))?,
        )?;
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
        let provision_phase = StartupPhaseTimer::start("static_provision");
        let provisioned = resolved_plan
            .execution_plan()
            .provision_static(Arc::clone(&runtime), provision_request)
            .map_err(|error| FerrumError::device(format!("vNext static provision: {error}")))?;
        provision_phase.finish();
        let (plan_resources, static_provider_attribution) = match provisioned.into_provisioning() {
            StaticProvisioning::NoStatic(no_static) => (no_static.into_plan_runtime(), None),
            StaticProvisioning::Required(permit) => {
                let identity = ResourceTransactionIdentity::for_admission(
                    permit.binding(),
                    run_id.clone(),
                    TransactionId::new(format!("transaction.vnext.provision.{run_id}"))
                        .map_err(|error| FerrumError::internal(error.to_string()))?,
                );
                let driver = RuntimeResourceDriver::new(Arc::clone(&runtime))
                    .map_err(|error| FerrumError::device(error.to_string()))?;
                let transaction_begin_phase =
                    StartupPhaseTimer::start("resource_transaction_begin");
                let transaction = ResourceTransaction::<VNextDriver<R>, TransactionNew>::begin(
                    driver, identity, permit,
                )
                .map_err(|error| FerrumError::device(error.to_string()))?;
                transaction_begin_phase.finish();
                let reserve_phase = StartupPhaseTimer::start("resource_reserve");
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
                reserve_phase.finish();
                let commit_phase = StartupPhaseTimer::start("resource_commit");
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
                commit_phase.finish();
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
                log_static_initialization_receipt(initialized.receipt());
                let provider_attribution =
                    StaticProviderAttributionWitness::from_completed_static_initialization(
                        &resolved_plan,
                        initialized.receipt(),
                    )
                    .map_err(|error| {
                        FerrumError::model(format!(
                            "vNext static provider attribution failed: {error}"
                        ))
                    })?;
                let handoff_phase = StartupPhaseTimer::start("runtime_handoff");
                match initialized.into_plan_runtime() {
                    Ok(resources) => {
                        handoff_phase.finish();
                        (resources, provider_attribution)
                    }
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
        executor_startup.finish();
        let lane = ExecutionLane::create(Arc::clone(&runtime)).map_err(|error| {
            FerrumError::device(format!("vNext execution lane creation failed: {error:?}"))
        })?;
        let submission_wave_identity =
            OperationDispatch::compile_submission_wave_identity(&resolved_plan, &lane).map_err(
                |error| {
                    FerrumError::model(format!(
                        "vNext submission-wave identity compilation failed: {error}"
                    ))
                },
            )?;
        let completion_worker = VNextCompletionWorker::new().map_err(|error| {
            FerrumError::device(format!("vNext completion worker creation failed: {error}"))
        })?;
        let reaper = CompletionReaper::new();
        let sequences = Arc::new(Mutex::new(VNextSequenceRegistry {
            recovery_wake: Some(completion_worker.recovery_wake()),
            ..VNextSequenceRegistry::default()
        }));
        sequence_settlement::install_recovery(&completion_worker, &sequences, &reaper).map_err(
            |error| FerrumError::device(format!("vNext recovery initialization failed: {error}")),
        )?;
        let reusable_execution_supported = runtime
            .descriptor()
            .capabilities
            .iter()
            .any(|capability| capability.as_str() == DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID);
        let reusable_execution_startup_plan =
            if config.device_reusable_execution_enabled && reusable_execution_supported {
                let reusable_memory = resolved_plan
                    .execution_plan()
                    .payload()
                    .memory()
                    .reusable_execution()
                    .ok_or_else(|| {
                        FerrumError::internal(
                            "vNext reusable execution is enabled without a resolved memory plan",
                        )
                    })?;
                let program_policy = reusable_memory.program_policy().ok_or_else(|| {
                    FerrumError::internal(
                        "vNext reusable execution is enabled without a resolved program policy",
                    )
                })?;
                let capture_resolution = config
                    .reusable_execution_capture_resolution
                    .clone()
                    .ok_or_else(|| {
                        FerrumError::internal(
                        "vNext reusable execution is enabled without capture resolution evidence",
                    )
                    })?;
                Some(VNextReusableExecutionStartupPlan::resolve(
                    program_policy,
                    capture_resolution,
                    config.maximum_model_tokens,
                    &config.reusable_execution_prefill_chunks,
                    reusable_memory.maximum_device_executables(),
                )?)
            } else {
                None
            };
        let checkpoint_capture = checkpoint_selection
            .map(|selection| {
                selection.bind(
                    resolved_plan.execution_plan(),
                    info.model_id.to_string(),
                    family_fingerprint.clone(),
                    program_fingerprint.clone(),
                    &run_id,
                    info.vocab_size,
                )
            })
            .transpose()?;

        let metrics = VNextExecutorMetrics::default();
        metrics
            .program_binding_upload
            .reset_baseline(runtime.program_binding_upload_snapshot());
        Ok(Self {
            info,
            token_issuer: ExecutorSequenceIssuer::new(),
            resolved_plan,
            capability_catalog: catalog,
            runtime,
            providers,
            policy: config.runtime_policy,
            invocation_preparation_strategy: config.invocation_preparation_strategy,
            resident_decode: Mutex::new(None),
            resident_metrics: ResidentMetrics::default(),
            completion_readback_policy: config.completion_readback_policy,
            sampling_readback_mode: config.sampling_readback_mode,
            plan_resources,
            lane,
            submission_wave_identity,
            completion_worker,
            reaper,
            io,
            maximum_model_tokens: config.maximum_model_tokens,
            startup_memory_plan: config.startup_memory_plan,
            run_id,
            family_fingerprint,
            program_fingerprint,
            static_provider_attribution,
            checkpoint_capture,
            static_bytes,
            sequence_state_memory,
            device_reusable_execution_enabled: config.device_reusable_execution_enabled,
            reusable_execution_supported,
            reusable_execution_startup_plan,
            reusable_execution_catalog: RwLock::new(None),
            reusable_execution_catalog_refresh_needed: AtomicBool::new(false),
            startup_reusable_programs: Mutex::new(BTreeMap::new()),
            startup_preparation: Mutex::new(VNextStartupPreparationState::Pending),
            sequences,
            prefix_cache: Mutex::new(prefix_cache::PrefixIndex::default()),
            prefix_capture_identity: Arc::new(()),
            product_token_mask_residency: Mutex::new(VNextProductTokenMaskResidency::default()),
            event_sink: RwLock::new(None),
            device_timing_mode: AtomicU8::new(DeviceTimingMode::Off as u8),
            diagnostic_fault: config.diagnostic_fault,
            diagnostic_fault_armed: AtomicBool::new(config.diagnostic_fault.is_some()),
            metrics,
        })
    }

    pub fn resolved_plan(&self) -> &ResolvedModelPlan {
        &self.resolved_plan
    }

    pub fn capability_catalog(&self) -> &CapabilityCatalog {
        &self.capability_catalog
    }

    async fn admit_startup_sequence(
        &self,
        resources: &mut VNextStartupSequenceGuard<'_, R>,
        input_tokens: Arc<[TokenId]>,
        maximum_sequence_tokens: usize,
    ) -> Result<bool> {
        let Some(index) =
            self.reserve_startup_sequence(resources, &input_tokens, maximum_sequence_tokens)?
        else {
            return Ok(false);
        };
        let chunk = PrefillChunk::new(0, input_tokens.len(), input_tokens.len())?;
        if !self
            .execute_startup_prefill_chunk(resources, index, chunk, "decode-sequence admission")
            .await?
        {
            return Err(FerrumError::resource_exhausted(
                "vNext startup decode-sequence prefill was not submitted; refusing an unprepared capture participant",
            ));
        }
        Ok(true)
    }

    fn reserve_startup_sequence(
        &self,
        resources: &mut VNextStartupSequenceGuard<'_, R>,
        input_tokens: &[TokenId],
        maximum_sequence_tokens: usize,
    ) -> Result<Option<usize>> {
        let request_id = RequestId::new();
        let mut maintenance_attempts = 0_u32;
        loop {
            let decision = self.admit_native_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_startup(
                    &request_id,
                    input_tokens,
                    maximum_sequence_tokens,
                ),
                adapter: None,
            });
            // Capture the exact pending slot before propagating any failure.
            // Admitted incarnations are promoted into the guard synchronously.
            resources.track_pending(&request_id);
            match decision? {
                TokenAdmissionOutcome::Admitted(frontier) => {
                    return resources.activate(frontier).map(Some)
                }
                TokenAdmissionOutcome::MaintenanceDeferred(_) => {
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
                            return Ok(None);
                        }
                        ExecutorPrefillMaintenanceOutcome::NoLongerPending => {
                            return Err(FerrumError::internal(
                                "vNext startup backing maintenance lost its retained request",
                            ));
                        }
                    }
                }
                TokenAdmissionOutcome::Deferred(_) | TokenAdmissionOutcome::CapacityDeferred(_) => {
                    resources.cancel_pending();
                    return Ok(None);
                }
                TokenAdmissionOutcome::PermanentRejected(rejected) => {
                    return Err(FerrumError::resource_exhausted(format!(
                        "vNext startup prefill was permanently rejected: {rejected:?}"
                    )));
                }
            }
        }
    }

    fn startup_execution_failure(
        outcome: ExecutionBatchOutcome,
        phase: &'static str,
    ) -> FerrumError {
        match outcome {
            ExecutionBatchOutcome::Interrupted(interruption) => interruption
                .causes()
                .iter()
                .find_map(ExecutionInterruptionCause::as_error)
                .cloned()
                .unwrap_or_else(|| {
                    FerrumError::resource_exhausted(format!(
                        "vNext startup {phase} was interrupted"
                    ))
                }),
            ExecutionBatchOutcome::ContractViolation(violation) => violation.error,
            ExecutionBatchOutcome::NotSubmitted(_) => {
                FerrumError::resource_exhausted(format!("vNext startup {phase} was not submitted"))
            }
            ExecutionBatchOutcome::Completed(_) => {
                FerrumError::internal("startup rejected a completed token batch")
            }
        }
        // Exact native owners stay in VNextStartupSequenceGuard and the
        // registry/reaper. Dropping an observation never stands in for Ended.
    }

    async fn execute_startup_prefill_chunk(
        &self,
        resources: &mut VNextStartupSequenceGuard<'_, R>,
        index: usize,
        chunk: PrefillChunk,
        phase: &'static str,
    ) -> Result<bool> {
        let policy = LogitsReturnPolicy::FullLogits;
        let input = ExecutionWork::new(
            &resources.sequences[index].frontier,
            TokenWork::Prefill(chunk),
            &policy,
        )?;
        let outcome = self
            .execute_native_tokens(std::slice::from_ref(&input))
            .await?;
        let completed = match outcome {
            ExecutionBatchOutcome::Completed(completed) => completed,
            ExecutionBatchOutcome::NotSubmitted(_) => return Ok(false),
            other => return Err(Self::startup_execution_failure(other, phase)),
        };
        let mut completions = completed.into_completions();
        if completions.len() != 1 {
            return Err(FerrumError::internal(
                "startup prefill returned an incorrect participant count",
            ));
        }
        let completion = completions.pop().expect("one startup completion");
        completion.validate_for(0, &input, self.info.vocab_size)?;
        if !matches!(completion.executed(), ExecutedTokenWork::Prefill { chunk: actual, capacity_probe_count: 0 } if actual == chunk)
        {
            return Err(FerrumError::internal(format!(
                "vNext startup {phase} did not complete its exact {:?} frontier",
                chunk.range()
            )));
        }
        resources.sequences[index].frontier = completion.into_parts().1;
        Ok(true)
    }

    async fn execute_startup_prefill_request(
        &self,
        chunk: PrefillChunk,
        phase: &'static str,
    ) -> Result<()> {
        let input_tokens: Arc<[TokenId]> = (0..chunk.total_prompt_tokens())
            .map(|_| TokenId::new(0))
            .collect::<Vec<_>>()
            .into();
        let mut resources = VNextStartupSequenceGuard::new(self);
        let Some(index) = self.reserve_startup_sequence(
            &mut resources,
            &input_tokens,
            chunk.total_prompt_tokens(),
        )?
        else {
            return Err(FerrumError::resource_exhausted(format!(
                "vNext startup {phase} could not admit prefill chunk {:?}",
                chunk.range()
            )));
        };
        if chunk.tokens_processed() > 0 {
            let prefix =
                PrefillChunk::new(0, chunk.tokens_processed(), chunk.total_prompt_tokens())?;
            if !self
                .execute_startup_prefill_chunk(&mut resources, index, prefix, phase)
                .await?
            {
                return Err(FerrumError::resource_exhausted(format!(
                    "vNext startup {phase} deferred the prerequisite prefill chunk {:?}",
                    prefix.range()
                )));
            }
        }
        if !self
            .execute_startup_prefill_chunk(&mut resources, index, chunk, phase)
            .await?
        {
            return Err(FerrumError::resource_exhausted(format!(
                "vNext startup {phase} deferred prefill chunk {:?}",
                chunk.range()
            )));
        }
        resources.complete()
    }

    async fn execute_startup_decode_pass(
        &self,
        resources: &mut VNextStartupSequenceGuard<'_, R>,
        input_token: TokenId,
        width: usize,
        phase: &'static str,
    ) -> Result<()> {
        if width == 0 || width > resources.sequences.len() {
            return Err(FerrumError::internal(format!(
                "vNext startup {phase} width {width} exceeds {} retained sequences",
                resources.sequences.len()
            )));
        }
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = resources
            .sequences
            .iter()
            .take(width)
            .map(|startup| {
                ExecutionWork::new(
                    &startup.frontier,
                    TokenWork::Decode {
                        token: input_token,
                        position: startup.frontier.consumed_tokens(),
                    },
                    &policy,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let completed = match self.execute_native_tokens(&inputs).await? {
            ExecutionBatchOutcome::Completed(completed) => completed,
            other => return Err(Self::startup_execution_failure(other, phase)),
        };
        let completions = completed.into_completions();
        if completions.len() != width {
            return Err(FerrumError::internal(format!(
                "vNext startup {phase} returned {} outputs for width {width}",
                completions.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for completion in &completions {
            let index = completion.input_index();
            let input = inputs.get(index).ok_or_else(|| {
                FerrumError::internal("startup decode returned a foreign participant index")
            })?;
            if !seen.insert(index) {
                return Err(FerrumError::internal(
                    "startup decode returned a duplicate participant",
                ));
            }
            completion.validate_for(index, input, self.info.vocab_size)?;
        }
        drop(inputs);
        for completion in completions {
            let (index, frontier, _, _) = completion.into_parts();
            resources.sequences[index].frontier = frontier;
        }
        Ok(())
    }

    async fn prepare_reusable_execution_startup(
        &self,
    ) -> Result<VNextReusableExecutionStartupReport> {
        let started = Instant::now();
        let Some(plan) = self.reusable_execution_startup_plan.clone() else {
            self.install_reusable_execution_catalog(VNextReusableExecutionCatalog {
                lane_epoch: self.lane.reusable_execution_epoch(),
                programs: BTreeMap::new(),
            })?;
            return Ok(VNextReusableExecutionStartupReport {
                enabled: self.device_reusable_execution_enabled,
                supported: self.reusable_execution_supported,
                eager_fallback_required: self.device_reusable_execution_enabled
                    && !self.reusable_execution_supported,
                resolved_runtime_policy_fingerprint: self.policy.fingerprint_str().to_owned(),
                resolved_program_policy: None,
                decode_width_resolution: None,
                maximum_device_executables: 0,
                requested_descriptors: Vec::new(),
                prepared_descriptors: Vec::new(),
                capture_case_receipts: Vec::new(),
                catalog_programs: Vec::new(),
                requested_decode_widths: Vec::new(),
                prepared_decode_widths: Vec::new(),
                requested_prefill_token_counts: Vec::new(),
                prepared_prefill_token_counts: Vec::new(),
                requested_prefill_chunks: Vec::new(),
                prepared_prefill_chunks: Vec::new(),
                synthetic_sequences: 0,
                eager_warmup_waves: 0,
                capture_waves: 0,
                replay_inventory_check_waves: 0,
                prepared_programs: 0,
                device_preparation: DeviceReusableExecutionPreparation::unsupported(),
                elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            });
        };

        let requested_descriptors = plan.descriptors.clone();
        let requested_decode_widths = plan.decode_widths();
        let requested_prefill_chunks = plan.prefill_chunks();
        let requested_prefill_token_counts = plan.prefill_token_counts();
        if plan.device_plan.catalog_lifetime() == ReusableExecutionCatalogLifetime::OnDemandBounded
        {
            return self.prepare_on_demand_reusable_execution(&plan, started);
        }
        for _ in 0..plan.warmup_passes {
            for chunk in requested_prefill_chunks.iter().copied() {
                self.execute_startup_prefill_request(chunk, "eager prefill warmup")
                    .await?;
            }
        }

        let input_tokens: Arc<[TokenId]> = Arc::from([TokenId::new(0)]);
        let input_token = TokenId::new(0);
        let mut resources = VNextStartupSequenceGuard::new(self);
        let requested_sequences = requested_decode_widths.first().copied().ok_or_else(|| {
            FerrumError::internal("vNext reusable execution plan has no decode widths")
        })?;
        if requested_sequences > MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH {
            return Err(FerrumError::config(format!(
                "vNext reusable startup requested {requested_sequences} synthetic sequences, exceeding the independent hard bound {MAXIMUM_REUSABLE_EXECUTION_STARTUP_CAPTURE_WIDTH}"
            )));
        }
        for _ in 0..requested_sequences {
            if !self
                .admit_startup_sequence(
                    &mut resources,
                    Arc::clone(&input_tokens),
                    plan.maximum_decode_sequence_tokens,
                )
                .await?
            {
                break;
            }
        }
        if resources.sequences.len() != requested_sequences {
            return Err(FerrumError::resource_exhausted(
                format!(
                    "vNext reusable execution startup admitted {} of {requested_sequences} synthetic sequences; refusing to shrink the fingerprinted exact matrix",
                    resources.sequences.len()
                ),
            ));
        }
        let prepared_decode_widths = requested_decode_widths.clone();

        for _ in 0..plan.warmup_passes {
            for width in prepared_decode_widths.iter().copied() {
                self.execute_startup_decode_pass(
                    &mut resources,
                    input_token,
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

        for _ in 0..plan.capture_passes {
            for width in prepared_decode_widths.iter().copied() {
                self.execute_startup_decode_pass(&mut resources, input_token, width, "capture")
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
            || captured.captured_executables() != captured.uploaded_executables()
            || captured.uploaded_executables() != captured.resident_executables()
        {
            return Err(FerrumError::device(format!(
                "vNext reusable execution capture receipt is incomplete: {captured:?}"
            )));
        }
        for _ in 0..plan.replay_validation_passes {
            for width in prepared_decode_widths.iter().copied() {
                self.execute_startup_decode_pass(
                    &mut resources,
                    input_token,
                    width,
                    "replay inventory check",
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
        if replayed.state() != DeviceReusableExecutionPreparationState::Preparing
            || !reusable_executable_inventory_matches(captured, replayed)
        {
            return Err(FerrumError::device(format!(
                "vNext replay inventory check compiled or changed executable state: before={captured:?}, after={replayed:?}"
            )));
        }

        let synthetic_sequences = resources.sequences.len();
        resources.complete()?;

        let mut captured = replayed;
        for _ in 0..plan.capture_passes {
            for chunk in requested_prefill_chunks.iter().copied() {
                self.execute_startup_prefill_request(chunk, "prefill capture")
                    .await?;
                let prefill_captured =
                    self.lane
                        .reusable_executable_preparation()
                        .map_err(|error| {
                            FerrumError::device(format!(
                                "vNext {:?} prefill capture inspection failed: {error}",
                                chunk.range()
                            ))
                        })?;
                if prefill_captured.state() != DeviceReusableExecutionPreparationState::Preparing
                    || prefill_captured.captured_executables()
                        != prefill_captured.uploaded_executables()
                    || prefill_captured.uploaded_executables()
                        != prefill_captured.resident_executables()
                    || prefill_captured.captured_executables() < captured.captured_executables()
                {
                    return Err(FerrumError::device(format!(
                        "vNext {:?} prefill capture receipt is incomplete: before={captured:?}, after={prefill_captured:?}",
                        chunk.range()
                    )));
                }
                captured = prefill_captured;
            }
        }

        for _ in 0..plan.replay_validation_passes {
            for chunk in requested_prefill_chunks.iter().copied() {
                self.execute_startup_prefill_request(chunk, "fresh-request prefill replay")
                    .await?;
                let prefill_replayed =
                    self.lane
                        .reusable_executable_preparation()
                        .map_err(|error| {
                            FerrumError::device(format!(
                            "vNext {:?} fresh-request prefill replay inspection failed: {error}",
                            chunk.range()
                        ))
                        })?;
                if prefill_replayed.state() != DeviceReusableExecutionPreparationState::Preparing
                    || !reusable_executable_inventory_matches(captured, prefill_replayed)
                {
                    return Err(FerrumError::device(format!(
                        "vNext {:?} fresh-request prefill replay changed executable state: before={captured:?}, after={prefill_replayed:?}",
                        chunk.range()
                    )));
                }
            }
        }

        let device_preparation = self.lane.seal_reusable_executables().map_err(|error| {
            FerrumError::device(format!("vNext reusable execution sealing failed: {error}"))
        })?;
        if device_preparation.state() != DeviceReusableExecutionPreparationState::Ready
            || device_preparation.uploaded_executables() < device_preparation.resident_executables()
            || !reusable_executable_inventory_matches(captured, device_preparation)
        {
            return Err(FerrumError::device(format!(
                "vNext reusable execution sealing produced an unusable receipt: {device_preparation:?}"
            )));
        }
        let catalog = self.lane.reusable_execution_catalog().map_err(|error| {
            FerrumError::device(format!(
                "vNext reusable execution catalog inspection failed: {error}"
            ))
        })?;
        let (catalog_epoch, catalog) = catalog.into_parts();
        let mut catalog_by_id = BTreeMap::new();
        for program in catalog {
            let program_id = program.program_id();
            if program_id.plan_hash() != self.resolved_plan.execution_plan().plan_hash()
                || program_id.runtime_implementation_fingerprint()
                    != self.runtime.descriptor().runtime_implementation_fingerprint
                || program_id.lane_id() != self.lane.id()
                || program.segments().iter().any(|segment| {
                    segment.end_node_index() as usize
                        > self.resolved_plan.execution_plan().payload().nodes().len()
                })
            {
                return Err(FerrumError::internal(
                    "vNext reusable execution catalog differs from its immutable plan or lane",
                ));
            }
            if catalog_by_id.insert(program_id.clone(), program).is_some() {
                return Err(FerrumError::internal(
                    "vNext reusable execution catalog contains a duplicate program identity",
                ));
            }
        }
        if catalog_by_id.len() > plan.program_policy.programs().len() {
            return Err(FerrumError::internal(format!(
                "vNext reusable execution observed {} physical programs for {} budgeted startup capture cases",
                catalog_by_id.len(),
                plan.program_policy.programs().len()
            )));
        }
        let startup_programs = self.startup_reusable_programs.lock().clone();
        let requested_descriptor_set = requested_descriptors
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if let Some((descriptor, observed_program_count)) =
            reusable_startup_case_budget_violation(&requested_descriptor_set, &startup_programs)
        {
            return Err(FerrumError::internal(format!(
                "vNext reusable startup case {descriptor:?} observed {} physical variants; the resolved policy budgets exactly one",
                observed_program_count
            )));
        }
        let observed_program_ids = startup_programs
            .values()
            .flat_map(|programs| programs.iter())
            .collect::<BTreeSet<_>>();
        if catalog_by_id
            .keys()
            .any(|program_id| !observed_program_ids.contains(program_id))
        {
            return Err(FerrumError::internal(
                "vNext reusable execution catalog contains a physical program not observed by a resolved startup capture case",
            ));
        }
        let prepared_programs = catalog_by_id
            .values()
            .filter(|program| program.has_resident_segments())
            .count();
        if !reusable_execution_program_catalog_is_usable(device_preparation, prepared_programs) {
            return Err(FerrumError::internal(format!(
                "vNext reusable execution sealed {} resident executable segments but registered no typed programs",
                device_preparation.resident_executables()
            )));
        }
        let catalog_programs = catalog_by_id
            .values()
            .map(VNextReusableExecutionCatalogProgramReceipt::from_program)
            .collect::<Vec<_>>();
        let capture_case_receipts = requested_descriptors
            .iter()
            .copied()
            .map(|descriptor| {
                let observed = startup_programs.get(&descriptor);
                let observed_program_fingerprints = observed
                    .into_iter()
                    .flat_map(|programs| programs.iter())
                    .map(DeviceReusableExecutionProgramId::fingerprint)
                    .collect::<Vec<_>>();
                let resident_program_fingerprints = observed
                    .into_iter()
                    .flat_map(|programs| programs.iter())
                    .filter(|program_id| {
                        catalog_by_id
                            .get(*program_id)
                            .is_some_and(DeviceReusableExecutionProgram::has_resident_segments)
                    })
                    .map(DeviceReusableExecutionProgramId::fingerprint)
                    .collect::<Vec<_>>();
                VNextReusableExecutionCaptureCaseReceipt {
                    descriptor,
                    observed_program_fingerprints,
                    resident_program_fingerprints,
                }
            })
            .collect::<Vec<_>>();
        let prepared_descriptors = capture_case_receipts
            .iter()
            .filter(|receipt| !receipt.resident_program_fingerprints.is_empty())
            .map(|receipt| receipt.descriptor)
            .collect::<Vec<_>>();
        let prepared_descriptor_set = prepared_descriptors
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let resident_prepared_decode_widths = prepared_descriptors
            .iter()
            .filter_map(|descriptor| match descriptor {
                VNextReusableExecutionDescriptor::UniformDecode {
                    request_capacity, ..
                } => Some(*request_capacity),
                VNextReusableExecutionDescriptor::Prefill { .. } => None,
            })
            .collect::<Vec<_>>();
        let prepared_prefill_chunks = requested_prefill_chunks
            .iter()
            .copied()
            .filter(|chunk| {
                let main = VNextReusableExecutionStartupPlan::descriptor_for_chunk(*chunk);
                if !prepared_descriptor_set.contains(&main) {
                    return false;
                }
                if chunk.tokens_processed() == 0 {
                    return true;
                }
                prepared_descriptor_set.contains(&VNextReusableExecutionDescriptor::Prefill {
                    tokens_processed: 0,
                    token_capacity: chunk.tokens_processed(),
                    total_prompt_tokens: chunk.total_prompt_tokens(),
                    request_capacity: 1,
                })
            })
            .collect::<Vec<_>>();
        let mut prepared_prefill_token_counts = prepared_prefill_chunks
            .iter()
            .map(|chunk| chunk.tokens_to_process())
            .collect::<Vec<_>>();
        prepared_prefill_token_counts.sort_unstable_by(|left, right| right.cmp(left));
        prepared_prefill_token_counts.dedup();
        let incomplete_capture_cases = prepared_descriptors.len() != requested_descriptors.len();
        self.install_reusable_execution_catalog(VNextReusableExecutionCatalog {
            lane_epoch: catalog_epoch,
            programs: catalog_by_id,
        })?;
        let requested_wave_shapes = prepared_decode_widths.len() + plan.prefill_wave_shapes();
        Ok(VNextReusableExecutionStartupReport {
            enabled: true,
            supported: true,
            eager_fallback_required: reusable_execution_requires_eager_fallback(device_preparation)
                || plan.decode_catalog_omits_admitted_widths()
                || incomplete_capture_cases,
            resolved_runtime_policy_fingerprint: self.policy.fingerprint_str().to_owned(),
            resolved_program_policy: Some(plan.program_policy.clone()),
            decode_width_resolution: Some(plan.capture_resolution.clone()),
            maximum_device_executables: plan.device_plan.maximum_executables(),
            requested_descriptors,
            prepared_descriptors,
            capture_case_receipts,
            catalog_programs,
            requested_decode_widths,
            prepared_decode_widths: resident_prepared_decode_widths,
            requested_prefill_token_counts: requested_prefill_token_counts.clone(),
            prepared_prefill_token_counts,
            requested_prefill_chunks: requested_prefill_chunks.clone(),
            prepared_prefill_chunks,
            synthetic_sequences,
            eager_warmup_waves: requested_wave_shapes * plan.warmup_passes,
            capture_waves: requested_wave_shapes * plan.capture_passes,
            replay_inventory_check_waves: requested_wave_shapes * plan.replay_validation_passes,
            prepared_programs,
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
        self.reaper.reset_checkpoint_timings();
        // Startup waves are synthetic evidence. Forget their residency proof so
        // the first product wave establishes and accounts for its own upload.
        self.release_resident_decode();
        for counter in [
            &self.resident_metrics.submitted,
            &self.resident_metrics.bootstraps,
            &self.resident_metrics.installations,
            &self.resident_metrics.unavailable,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        self.product_token_mask_residency.lock().clear();
        self.metrics.reset_after_startup();
        self.metrics
            .program_binding_upload
            .reset_baseline(self.runtime.program_binding_upload_snapshot());
        Ok(())
    }

    fn device_timing_mode(&self) -> DeviceTimingMode {
        match self.device_timing_mode.load(Ordering::Acquire) {
            value if value == DeviceTimingMode::Verification as u8 => {
                DeviceTimingMode::Verification
            }
            value if value == DeviceTimingMode::Kernel as u8 => DeviceTimingMode::Kernel,
            value if value == DeviceTimingMode::Replay as u8 => DeviceTimingMode::Replay,
            value if value == DeviceTimingMode::Completion as u8 => DeviceTimingMode::Completion,
            _ => DeviceTimingMode::Off,
        }
    }

    fn host_dispatch_timing_enabled(&self) -> bool {
        self.device_timing_mode() != DeviceTimingMode::Off
    }

    fn token_mask_residency_eligible(
        executable: &impl ExecutablePlanView,
        token_mask_input: &ResolvedValueBinding,
    ) -> bool {
        let [component] = token_mask_input.storage().components() else {
            return false;
        };
        let memory = executable.execution_plan().payload().memory();
        let Some(descriptor) = memory
            .dynamic_descriptors()
            .iter()
            .find(|descriptor| descriptor.base_resource_id() == component.resource_id())
        else {
            return false;
        };
        if token_mask_input.usage() != BufferUsage::Activations
            || descriptor.lifetime() != AllocationLifetime::Step
            || descriptor.kind() != &AllocationKind::Value
            || descriptor.usage() != BufferUsage::Activations
            || descriptor.element_type() != ElementType::U8
            || descriptor.initialization() != StateInitialization::None
            || !matches!(
                descriptor.demand(),
                DynamicResourceDemand::ActualSequences { .. }
            )
        {
            return false;
        }
        let mut matching_slots = memory
            .dynamic_pools()
            .iter()
            .flat_map(DynamicBackingPoolSpec::step_resource_slots)
            .filter(|slot| slot.resource_ids().contains(&component.resource_id()));
        let Some(slot) = matching_slots.next() else {
            return false;
        };
        matching_slots.next().is_none()
            && slot.kind() == StepResourceSlotKind::Dedicated
            && slot.resource_ids() == std::slice::from_ref(component.resource_id())
    }

    fn resolve_io(
        executable: &impl ExecutablePlanView,
        language_io: &VNextLanguageIoIds,
        expected_vocab: usize,
        expected_repetition_capacity: usize,
    ) -> Result<VNextIoBinding> {
        let nodes = executable.execution_plan().payload().nodes();
        let input_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == &language_io.token_input
                    && value.role() == ResolvedValueRole::Input
            })
            .collect::<Vec<_>>();
        let token_mask_input_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == &language_io.token_mask_input
                    && value.role() == ResolvedValueRole::Input
            })
            .collect::<Vec<_>>();
        let repetition_token_ids_input_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == &language_io.repetition_token_ids_input
                    && value.role() == ResolvedValueRole::Input
            })
            .collect::<Vec<_>>();
        let repetition_offsets_input_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == &language_io.repetition_offsets_input
                    && value.role() == ResolvedValueRole::Input
            })
            .collect::<Vec<_>>();
        let repetition_penalty_input_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == &language_io.repetition_penalty_input
                    && value.role() == ResolvedValueRole::Input
            })
            .collect::<Vec<_>>();
        let logits_output_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == &language_io.logits_output
                    && value.role() == ResolvedValueRole::Output
            })
            .collect::<Vec<_>>();
        let greedy_token_output_matches = nodes
            .iter()
            .flat_map(|node| node.values().iter().map(move |value| (node.id(), value)))
            .filter(|(_, value)| {
                value.value_id() == &language_io.greedy_token_output
                    && value.role() == ResolvedValueRole::Output
            })
            .collect::<Vec<_>>();
        let [(input_node_id, input)] = input_matches.as_slice() else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind the token input exactly once",
            ));
        };
        let [(token_mask_input_node_id, token_mask_input)] = token_mask_input_matches.as_slice()
        else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind the token-selection mask exactly once",
            ));
        };
        let [(repetition_token_ids_input_node_id, repetition_token_ids_input)] =
            repetition_token_ids_input_matches.as_slice()
        else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind sparse repetition token ids exactly once",
            ));
        };
        let [(repetition_offsets_input_node_id, repetition_offsets_input)] =
            repetition_offsets_input_matches.as_slice()
        else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind sparse repetition offsets exactly once",
            ));
        };
        let [(repetition_penalty_input_node_id, repetition_penalty_input)] =
            repetition_penalty_input_matches.as_slice()
        else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind sparse repetition penalty exactly once",
            ));
        };
        let [(output_node_id, output)] = logits_output_matches.as_slice() else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind the logits output exactly once",
            ));
        };
        let [(greedy_token_output_node_id, greedy_token_output)] =
            greedy_token_output_matches.as_slice()
        else {
            return Err(FerrumError::model(
                "compiled vNext plan must bind the selected-token output exactly once",
            ));
        };
        if input.tensor().element_type() != ElementType::U32 {
            return Err(FerrumError::model(
                "compiled vNext token input must use U32 elements",
            ));
        }
        let expected_vocab_u64 = u64::try_from(expected_vocab)
            .map_err(|_| FerrumError::model("vNext vocabulary exceeds u64"))?;
        if token_mask_input.tensor().element_type() != ElementType::U8
            || token_mask_input.tensor().dimensions() != [expected_vocab_u64]
            || !matches!(
                token_mask_input.tensor().layout(),
                ResolvedTensorLayout::Contiguous
            )
        {
            return Err(FerrumError::model(
                "compiled vNext token-selection mask must be contiguous U8[vocab]",
            ));
        }
        let expected_repetition_capacity_u64 = u64::try_from(expected_repetition_capacity)
            .map_err(|_| FerrumError::model("vNext repetition capacity exceeds u64"))?;
        let contiguous = |value: &ResolvedValueBinding| {
            matches!(value.tensor().layout(), ResolvedTensorLayout::Contiguous)
        };
        if repetition_token_ids_input.tensor().element_type() != ElementType::U32
            || repetition_token_ids_input.tensor().dimensions()
                != [expected_repetition_capacity_u64]
            || !contiguous(repetition_token_ids_input)
        {
            return Err(FerrumError::model(
                "compiled vNext sparse repetition ids must be contiguous U32[capacity]",
            ));
        }
        if repetition_offsets_input.tensor().element_type() != ElementType::U32
            || repetition_offsets_input.tensor().dimensions() != [2]
            || !contiguous(repetition_offsets_input)
        {
            return Err(FerrumError::model(
                "compiled vNext sparse repetition offsets must be contiguous U32[2]",
            ));
        }
        if repetition_penalty_input.tensor().element_type() != ElementType::F32
            || repetition_penalty_input.tensor().dimensions() != [1]
            || !contiguous(repetition_penalty_input)
        {
            return Err(FerrumError::model(
                "compiled vNext sparse repetition penalty must be contiguous F32[1]",
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
        let [greedy_token_component] = greedy_token_output.storage().components() else {
            return Err(FerrumError::model(
                "compiled vNext selected-token output must use one physical component",
            ));
        };
        if greedy_token_output.tensor().element_type() != ElementType::U32
            || greedy_token_output.tensor().dimensions() != [1]
        {
            return Err(FerrumError::model(
                "compiled vNext selected-token output must be U32[1]",
            ));
        }
        let greedy_token_output_layout = HostTransferLayout::new(ElementType::U32, 1)
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let token_mask_residency_eligible =
            Self::token_mask_residency_eligible(executable, token_mask_input);
        Ok(VNextIoBinding {
            input_node_id: (*input_node_id).clone(),
            input_ordinal: input.ordinal(),
            token_mask_input_node_id: (*token_mask_input_node_id).clone(),
            token_mask_input_ordinal: token_mask_input.ordinal(),
            token_mask_residency_eligible,
            repetition_token_ids_input_node_id: (*repetition_token_ids_input_node_id).clone(),
            repetition_token_ids_input_ordinal: repetition_token_ids_input.ordinal(),
            repetition_offsets_input_node_id: (*repetition_offsets_input_node_id).clone(),
            repetition_offsets_input_ordinal: repetition_offsets_input.ordinal(),
            repetition_penalty_input_node_id: (*repetition_penalty_input_node_id).clone(),
            repetition_penalty_input_ordinal: repetition_penalty_input.ordinal(),
            repetition_capacity: expected_repetition_capacity,
            output_node_id: (*output_node_id).clone(),
            output_resource_id: component.resource_id().clone(),
            output_offset_bytes: component.offset_bytes(),
            output_layout,
            output_element_type,
            output_elements,
            greedy_token_output_node_id: (*greedy_token_output_node_id).clone(),
            greedy_token_output_resource_id: greedy_token_component.resource_id().clone(),
            greedy_token_output_offset_bytes: greedy_token_component.offset_bytes(),
            greedy_token_output_layout,
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
                sequences
                    .token_index
                    .insert(sequence.token_identity.clone(), Arc::downgrade(&sequence));
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
        request_origin: ExecutorRequestOrigin,
        maximum_tokens: usize,
        tokens: Vec<u32>,
        product_prompt_tokens: usize,
        replayed_output_tokens: usize,
        work: ResourceWorkShape,
    ) -> Result<VNextPrefillProbeResolution<R>> {
        let product_prompt_tokens = u64::try_from(product_prompt_tokens).map_err(|_| {
            FerrumError::request_validation("product prompt token count exceeds u64")
        })?;
        let replayed_output_tokens = u64::try_from(replayed_output_tokens).map_err(|_| {
            FerrumError::request_validation("replayed output token count exceeds u64")
        })?;
        let identity = RequestIdentity::new(format!(
            "request.{}.{request_id}",
            request_origin.namespace()
        ))
        .map_err(|error| FerrumError::internal(error.to_string()))?;
        let session = match self.try_admit_sequence(identity.clone(), work)? {
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
        let request = match VNextRequestRoot::bind_initial(request_id.clone(), &identity, &session)
        {
            Ok(request) => request,
            Err(error) => {
                return Err(terminalize_unsubmitted_session(&session, error));
            }
        };
        let active_binding = match TrustedActiveSequenceBinding::from_session(&session) {
            Ok(active_binding) => Arc::new(active_binding),
            Err(error) => {
                return Err(terminalize_unsubmitted_session(
                    &session,
                    FerrumError::backend(error.to_string()),
                ));
            }
        };
        let events = match self.execution_journal(&active_binding, request_origin) {
            Ok(events) => events,
            Err(error) => {
                return Err(terminalize_unsubmitted_session(&session, error));
            }
        };
        let token_identity = self
            .token_issuer
            .issue_sequence()
            .map_err(|error| terminalize_unsubmitted_session(&session, error))?;
        let token_recovery_key = self
            .token_issuer
            .recovery_key(&token_identity)
            .map_err(|error| terminalize_unsubmitted_session(&session, error))?;
        let sequence = Arc::new(VNextSequence {
            token_identity,
            token_recovery_key,
            token_issuer: self.token_issuer.clone(),
            prefix_capture_interests: Mutex::new(Vec::new()),
            cache_id: format!(
                "vnext-cache-{request_id}-{}-{}",
                session.sequence_authority().sparse_id(),
                session.sequence_authority().generation()
            ),
            request,
            session,
            active_binding,
            request_origin,
            tokens: Mutex::new(tokens),
            maximum_tokens,
            active: AtomicBool::new(true),
            settlement: Mutex::new(SequenceSettlement::default()),
            recovery_wake: self.completion_worker.recovery_wake(),
            operation: AsyncMutex::new(()),
            events: events.map(Mutex::new),
            product_prompt_tokens,
            replayed_output_tokens,
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

    fn try_admit_sequence(
        &self,
        request_id: RequestIdentity,
        work: ResourceWorkShape,
    ) -> Result<VNextSequenceAdmissionDecision<R>> {
        // A new participant will change the cohort. Release idle decode claims
        // before admitting its Request/Sequence resources or doing maintenance.
        self.release_resident_decode();
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
        let mut prefix_maintenance = self.prefix_pressure_maintenance();
        let mut prefix_backing_attempts = 0;
        let sequence = loop {
            match binding
                .try_admit_initial_sequence(
                    request.clone(),
                    sequence.clone(),
                    self.run_id.clone(),
                    request_id.clone(),
                )
                .map_err(|error| FerrumError::backend(error.to_string()))?
            {
                InitialSequenceResourceAdmissionDecision::Admitted(sequence) => break sequence,
                InitialSequenceResourceAdmissionDecision::Deferred(deferred) => {
                    match self.recover_prefix_pressure(&mut prefix_maintenance, &deferred)? {
                        prefix_cache::PrefixPressureRecovery::Maintained(_)
                        | prefix_cache::PrefixPressureRecovery::Evicted => continue,
                        prefix_cache::PrefixPressureRecovery::Unchanged => {}
                    }
                    self.metrics
                        .sequence_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(VNextSequenceAdmissionDecision::Deferred(deferred));
                }
                InitialSequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                    if prefix_maintenance.allows_backing_attempt(prefix_backing_attempts)
                        && self.prefix_checkpoint_may_block(deferred.evidence().wait_condition())
                    {
                        prefix_backing_attempts += 1;
                        match deferred
                            .maintain()
                            .map_err(|error| FerrumError::backend(error.to_string()))?
                        {
                            DynamicDeferredMaintenanceOutcome::Maintained(_)
                            | DynamicDeferredMaintenanceOutcome::RetryAdmission { .. } => continue,
                            DynamicDeferredMaintenanceOutcome::WaitForRelease {
                                wait_condition,
                                ..
                            } => {
                                if prefix_maintenance.eviction_after_wait(self, &wait_condition) {
                                    continue;
                                }
                            }
                        }
                    }
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(VNextSequenceAdmissionDecision::BackingDeferred(deferred));
                }
                InitialSequenceResourceAdmissionDecision::PermanentRejected(rejected) => {
                    return Ok(VNextSequenceAdmissionDecision::PermanentRejected(rejected));
                }
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
        request_origin: ExecutorRequestOrigin,
    ) -> Result<Option<VNextExecutionJournal>> {
        let Some(sink) = self.event_sink.read().clone() else {
            return Ok(None);
        };
        if sink.enablement() == ferrum_runtime::ExecutionEventSinkEnablement::None {
            // Timing is configured independently when the sink is attached. A
            // metrics-only observer needs no transactional event cursor or payloads.
            return Ok(None);
        }
        // Preserve the same sink allocation and queue; the runtime only sees
        // the lower execution-event contract, without product prefix decisions.
        let runtime_sink: Arc<dyn ExecutionEventSink> = sink;
        VNextExecutionJournal::open(
            runtime_sink,
            self.resolved_plan.execution_plan(),
            Arc::clone(active),
            request_origin,
        )
        .map(Some)
        .map_err(|error| FerrumError::backend(format!("vNext execution journal: {error}")))
    }

    fn execution_maintenance_decision<'a>(
        &self,
        stage: ExecutorExecutionCapacityStage,
        outcome: DynamicDeferredMaintenanceOutcome,
        source: VNextExecutionMaintenanceSource<'_>,
        participants: impl IntoIterator<Item = &'a VNextSequence<R>>,
        progress_receipts: &mut Vec<DynamicPoolGrowthBatchReceipt>,
        prefix_maintenance: &mut prefix_cache::PrefixPressureMaintenance,
    ) -> Result<Option<ExecutorExecutionCapacityDeferral>> {
        match outcome {
            DynamicDeferredMaintenanceOutcome::RetryAdmission { .. } => Ok(None),
            DynamicDeferredMaintenanceOutcome::Maintained(receipt) => {
                progress_receipts.push(receipt.clone());
                let Some(sink) = self.event_sink.read().clone() else {
                    return Ok(None);
                };
                if !sink.records_execution_resource_maintenance() {
                    return Ok(None);
                }
                let stage = match stage {
                    ExecutorExecutionCapacityStage::SequenceExtension => {
                        ExecutionResourceMaintenanceStage::SequenceExtension
                    }
                    ExecutorExecutionCapacityStage::StepAdmission => {
                        ExecutionResourceMaintenanceStage::StepAdmission
                    }
                    ExecutorExecutionCapacityStage::SubmissionWave => {
                        ExecutionResourceMaintenanceStage::SubmissionWave
                    }
                };
                let maintenance = BoundExecutionResourceMaintenance::bind(
                    stage,
                    participants
                        .into_iter()
                        .map(|sequence| sequence.active_binding.as_ref()),
                    receipt,
                )
                .map_err(|error| {
                    FerrumError::backend(format!(
                        "vNext execution resource maintenance binding: {error}"
                    ))
                })?;
                crate::event_sink::resolve_event_delivery(
                    sink.as_ref(),
                    sink.record_execution_resource_maintenance(maintenance),
                )
                .map_err(|error| {
                    FerrumError::backend(format!(
                        "vNext execution resource maintenance event: {error}"
                    ))
                })?;
                Ok(None)
            }
            DynamicDeferredMaintenanceOutcome::WaitForRelease {
                current_epochs,
                wait_condition,
                pressure,
                maintenance_boundary,
            } => {
                if prefix_maintenance.eviction_after_wait(self, &wait_condition) {
                    return Ok(None);
                }
                match source {
                    VNextExecutionMaintenanceSource::Logical(source) => {
                        ExecutorExecutionCapacityDeferral::from_admission_maintenance(
                            source,
                            ExecutorAdmissionEpochs::from_capacity(current_epochs),
                            wait_condition,
                            pressure,
                            maintenance_boundary,
                            stage,
                        )
                    }
                    VNextExecutionMaintenanceSource::Backing(source) => {
                        ExecutorExecutionCapacityDeferral::from_backing_maintenance(
                            source,
                            ExecutorAdmissionEpochs::from_capacity(current_epochs),
                            wait_condition,
                            pressure,
                            maintenance_boundary,
                            stage,
                        )
                    }
                }
                .map(Some)
            }
        }
    }

    fn bind_execution_maintenance_retry(
        &self,
        deferral: ExecutorExecutionCapacityDeferral,
        attempts: u32,
        receipts: &[DynamicPoolGrowthBatchReceipt],
        affected_request_ids: Vec<RequestId>,
    ) -> Result<ExecutorExecutionCapacityDeferral> {
        if receipts.is_empty() {
            return Ok(deferral);
        }
        let status = self
            .plan_resources
            .dynamic_pool_status()
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        deferral.with_relevant_maintenance_retry(
            attempts,
            receipts,
            status.pools(),
            affected_request_ids,
        )
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
        let mut prefix_maintenance = self.prefix_pressure_maintenance();
        let mut backing_attempts = 0;
        let mut maintenance_receipts = Vec::new();
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
            let request = match self.policy.admission().sequence_backing_growth {
                SequenceBackingGrowthPolicy::Exact => request,
                SequenceBackingGrowthPolicy::PagedGroup(pages) => {
                    request.with_paged_growth_group(pages)
                }
                SequenceBackingGrowthPolicy::PagedGroupGrowOnly(pages) => {
                    request.with_paged_growth_group_grow_only(pages)
                }
            };
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
                    let retained_resident =
                        if deferred.action() == DeferredAction::AwaitBackingGrowth {
                            self.retain_resident_decode_for_growth()
                        } else {
                            self.release_resident_decode();
                            false
                        };
                    match self.recover_prefix_pressure(&mut prefix_maintenance, &deferred)? {
                        prefix_cache::PrefixPressureRecovery::Maintained(receipt) => {
                            if !receipt.growths().is_empty() {
                                let _ = self.execution_maintenance_decision(
                                    ExecutorExecutionCapacityStage::SequenceExtension,
                                    DynamicDeferredMaintenanceOutcome::Maintained(receipt),
                                    VNextExecutionMaintenanceSource::Logical(&deferred),
                                    std::iter::once(sequence),
                                    &mut maintenance_receipts,
                                    &mut prefix_maintenance,
                                )?;
                            }
                            continue;
                        }
                        prefix_cache::PrefixPressureRecovery::Evicted => continue,
                        prefix_cache::PrefixPressureRecovery::Unchanged => {}
                    }
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
                    if !backing_maintenance::allows_backing_attempt(
                        &prefix_maintenance,
                        backing_attempts,
                        &maintenance_receipts,
                    ) {
                        if retained_resident {
                            self.release_resident_decode();
                            continue;
                        }
                        let deferral = ExecutorExecutionCapacityDeferral::from_pending_maintenance(
                            &deferred,
                            ExecutorExecutionCapacityStage::SequenceExtension,
                        )?;
                        return self
                            .bind_execution_maintenance_retry(
                                deferral,
                                backing_attempts,
                                &maintenance_receipts,
                                vec![sequence.request_id().clone()],
                            )
                            .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    backing_attempts += 1;
                    let outcome = self
                        .plan_resources
                        .maintain_for_admission_deferred(&deferred)
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = self.execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SequenceExtension,
                        outcome,
                        VNextExecutionMaintenanceSource::Logical(&deferred),
                        std::iter::once(sequence),
                        &mut maintenance_receipts,
                        &mut prefix_maintenance,
                    )? {
                        if retained_resident {
                            // Retry after releasing retained claims before
                            // exposing a pressure deferral to the scheduler.
                            self.release_resident_decode();
                            continue;
                        }
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                SequenceResourceExtensionDecision::BackingDeferred(deferred) => {
                    let retained_resident = self.retain_resident_decode_for_growth();
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if !backing_maintenance::allows_backing_attempt(
                        &prefix_maintenance,
                        backing_attempts,
                        &maintenance_receipts,
                    ) {
                        if retained_resident {
                            self.release_resident_decode();
                            continue;
                        }
                        let deferral = ExecutorExecutionCapacityDeferral::from_backing(
                            deferred.evidence(),
                            ExecutorExecutionCapacityStage::SequenceExtension,
                        )?;
                        return self
                            .bind_execution_maintenance_retry(
                                deferral,
                                backing_attempts,
                                &maintenance_receipts,
                                vec![sequence.request_id().clone()],
                            )
                            .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    backing_attempts += 1;
                    let outcome = deferred
                        .maintain()
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = self.execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SequenceExtension,
                        outcome,
                        VNextExecutionMaintenanceSource::Backing(deferred.evidence()),
                        std::iter::once(sequence),
                        &mut maintenance_receipts,
                        &mut prefix_maintenance,
                    )? {
                        if retained_resident {
                            self.release_resident_decode();
                            continue;
                        }
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
            VNextExecutionCapacityDecision::RequestStateDeferred(_) => Err(FerrumError::internal(
                "sequence extension unexpectedly produced a Request-state deferral",
            )),
        }
    }

    fn step_request_for_spans(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
    ) -> Result<StepResourceAdmissionRequest> {
        let timing_enabled = self.host_dispatch_timing_enabled();
        let phase_timing = self.metrics.wave_timing_for(kind);
        let _timing = self
            .metrics
            .wave_timing
            .resource_step_request_prepare
            .start_if(timing_enabled);
        let _phase_timing = phase_timing
            .resource_step_request_prepare
            .start_if(timing_enabled);
        let work_shape = Arc::new(
            batch
                .bind_work_shape(spans.to_vec())
                .map_err(|error| FerrumError::backend(error.to_string()))?,
        );
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
        Ok(match reusable_bucket_id {
            Some(bucket_id) => request.with_reusable_execution_bucket(bucket_id),
            None => request,
        })
    }

    fn try_begin_step_for_spans(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
    ) -> Result<StepResourceAdmissionDecision<R>> {
        let timing_enabled = self.host_dispatch_timing_enabled();
        let phase_timing = self.metrics.wave_timing_for(kind);
        let request = self.step_request_for_spans(batch, spans, kind)?;
        {
            let _timing = self
                .metrics
                .wave_timing
                .resource_step_admission
                .start_if(timing_enabled);
            let _phase_timing = phase_timing
                .resource_step_admission
                .start_if(timing_enabled);
            let decision = if timing_enabled {
                batch.try_begin_step_profiled(request, &self.lane, |phase, duration| {
                    self.metrics
                        .wave_timing
                        .resource_step_admission_breakdown
                        .record(phase, duration);
                    phase_timing
                        .resource_step_admission_breakdown
                        .record(phase, duration);
                })
            } else {
                batch.try_begin_step(request, &self.lane)
            };
            decision.map_err(|error| FerrumError::backend(error.to_string()))
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    fn begin_step(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        sequence: &Arc<VNextSequence<R>>,
        span: &TokenSpanWork,
    ) -> Result<Arc<StepResourceLease<R>>> {
        self.begin_step_for_spans(
            batch,
            std::slice::from_ref(sequence),
            std::slice::from_ref(span),
        )
    }

    fn begin_step_for_spans(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        sequences: &[Arc<VNextSequence<R>>],
        spans: &[TokenSpanWork],
    ) -> Result<Arc<StepResourceLease<R>>> {
        match self.begin_step_for_spans_with_capacity(
            batch,
            sequences,
            spans,
            VNextExecutionWaveKind::Decode,
        )? {
            VNextExecutionCapacityDecision::Ready(step) => Ok(step),
            VNextExecutionCapacityDecision::Deferred(deferred) => {
                Err(Self::execution_capacity_error(&deferred))
            }
            VNextExecutionCapacityDecision::RequestStateDeferred(_) => Err(FerrumError::internal(
                "step admission unexpectedly produced a Request-state deferral",
            )),
        }
    }

    fn begin_step_for_spans_with_capacity(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        sequences: &[Arc<VNextSequence<R>>],
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
    ) -> Result<VNextExecutionCapacityDecision<Arc<StepResourceLease<R>>>> {
        if sequences.is_empty()
            || sequences.len() != batch.sessions().len()
            || batch
                .sessions()
                .iter()
                .zip(sequences)
                .any(|(session, sequence)| !Arc::ptr_eq(session, &sequence.session))
        {
            return Err(FerrumError::internal(
                "vNext step maintenance participants differ from the canonical batch",
            ));
        }
        let mut prefix_maintenance = self.prefix_pressure_maintenance();
        let mut backing_attempts = 0;
        let mut maintenance_receipts = Vec::new();
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
                    if !backing_maintenance::allows_backing_attempt(
                        &prefix_maintenance,
                        backing_attempts,
                        &maintenance_receipts,
                    ) {
                        let deferral = ExecutorExecutionCapacityDeferral::from_pending_maintenance(
                            &deferred,
                            ExecutorExecutionCapacityStage::StepAdmission,
                        )?;
                        return self
                            .bind_execution_maintenance_retry(
                                deferral,
                                backing_attempts,
                                &maintenance_receipts,
                                sequences
                                    .iter()
                                    .map(|sequence| sequence.request_id().clone())
                                    .collect(),
                            )
                            .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    backing_attempts += 1;
                    let outcome = self
                        .plan_resources
                        .maintain_for_admission_deferred(&deferred)
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = self.execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::StepAdmission,
                        outcome,
                        VNextExecutionMaintenanceSource::Logical(&deferred),
                        sequences.iter().map(Arc::as_ref),
                        &mut maintenance_receipts,
                        &mut prefix_maintenance,
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if !backing_maintenance::allows_backing_attempt(
                        &prefix_maintenance,
                        backing_attempts,
                        &maintenance_receipts,
                    ) {
                        let deferral = ExecutorExecutionCapacityDeferral::from_backing(
                            deferred.evidence(),
                            ExecutorExecutionCapacityStage::StepAdmission,
                        )?;
                        return self
                            .bind_execution_maintenance_retry(
                                deferral,
                                backing_attempts,
                                &maintenance_receipts,
                                sequences
                                    .iter()
                                    .map(|sequence| sequence.request_id().clone())
                                    .collect(),
                            )
                            .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    backing_attempts += 1;
                    let outcome = deferred
                        .maintain()
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = self.execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::StepAdmission,
                        outcome,
                        VNextExecutionMaintenanceSource::Backing(deferred.evidence()),
                        sequences.iter().map(Arc::as_ref),
                        &mut maintenance_receipts,
                        &mut prefix_maintenance,
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

    fn try_prepare_wave_for_spans(
        &self,
        step: &Arc<StepResourceLease<R>>,
        spans: &[TokenSpanWork],
    ) -> Result<StepSubmissionWaveAdmissionDecision<R>> {
        let work_shape = step
            .shared_all_invocation_work_shape(spans)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        step.try_prepare_full_plan_submission_wave(
            work_shape,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn validate_step_maintenance_participants(
        step: &StepResourceLease<R>,
        sequences: &[Arc<VNextSequence<R>>],
    ) -> Result<()> {
        if sequences.is_empty()
            || step.participant_count() as usize != sequences.len()
            || step
                .participants()
                .zip(sequences)
                .any(|(resources, sequence)| !Arc::ptr_eq(resources, sequence.session.resources()))
        {
            return Err(FerrumError::internal(
                "vNext submission-wave maintenance participants differ from the exact step",
            ));
        }
        Ok(())
    }

    #[cfg(any(test, feature = "test-support"))]
    fn prepare_wave(
        &self,
        step: &Arc<StepResourceLease<R>>,
        sequence: &Arc<VNextSequence<R>>,
        span: &TokenSpanWork,
    ) -> Result<PreparedStepSubmissionWave<R>> {
        self.prepare_wave_for_spans(
            step,
            std::slice::from_ref(sequence),
            std::slice::from_ref(span),
        )
    }

    fn prepare_wave_for_spans(
        &self,
        step: &Arc<StepResourceLease<R>>,
        sequences: &[Arc<VNextSequence<R>>],
        spans: &[TokenSpanWork],
    ) -> Result<PreparedStepSubmissionWave<R>> {
        match self.prepare_wave_for_spans_with_capacity(
            step,
            sequences,
            spans,
            VNextExecutionWaveKind::Decode,
        )? {
            VNextExecutionCapacityDecision::Ready(wave) => Ok(wave),
            VNextExecutionCapacityDecision::Deferred(deferred) => {
                Err(Self::execution_capacity_error(&deferred))
            }
            VNextExecutionCapacityDecision::RequestStateDeferred(deferred) => {
                Err(FerrumError::resource_exhausted(format!(
                    "vNext synchronous submission wave is waiting for Request-state hazards: {:?}",
                    deferred.hazard().blockers()
                )))
            }
        }
    }

    fn prepare_wave_for_spans_with_capacity(
        &self,
        step: &Arc<StepResourceLease<R>>,
        sequences: &[Arc<VNextSequence<R>>],
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
    ) -> Result<VNextExecutionCapacityDecision<PreparedStepSubmissionWave<R>>> {
        if sequences.len() != spans.len() {
            return Err(FerrumError::internal(
                "vNext submission-wave maintenance participants differ from the work spans",
            ));
        }
        Self::validate_step_maintenance_participants(step, sequences)?;
        let timing_enabled = self.host_dispatch_timing_enabled();
        let phase_timing = self.metrics.wave_timing_for(kind);
        let _timing = self
            .metrics
            .wave_timing
            .resource_submission_wave_prepare
            .start_if(timing_enabled);
        let _phase_timing = phase_timing
            .resource_submission_wave_prepare
            .start_if(timing_enabled);
        let mut prefix_maintenance = self.prefix_pressure_maintenance();
        let mut backing_attempts = 0;
        let mut maintenance_receipts = Vec::new();
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
                    if !backing_maintenance::allows_backing_attempt(
                        &prefix_maintenance,
                        backing_attempts,
                        &maintenance_receipts,
                    ) {
                        let deferral = ExecutorExecutionCapacityDeferral::from_pending_maintenance(
                            &deferred,
                            ExecutorExecutionCapacityStage::SubmissionWave,
                        )?;
                        return self
                            .bind_execution_maintenance_retry(
                                deferral,
                                backing_attempts,
                                &maintenance_receipts,
                                sequences
                                    .iter()
                                    .map(|sequence| sequence.request_id().clone())
                                    .collect(),
                            )
                            .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    backing_attempts += 1;
                    let outcome = self
                        .plan_resources
                        .maintain_for_admission_deferred(&deferred)
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = self.execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SubmissionWave,
                        outcome,
                        VNextExecutionMaintenanceSource::Logical(&deferred),
                        sequences.iter().map(Arc::as_ref),
                        &mut maintenance_receipts,
                        &mut prefix_maintenance,
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                    self.metrics
                        .backing_deferrals
                        .fetch_add(1, Ordering::Relaxed);
                    if !backing_maintenance::allows_backing_attempt(
                        &prefix_maintenance,
                        backing_attempts,
                        &maintenance_receipts,
                    ) {
                        let deferral = ExecutorExecutionCapacityDeferral::from_backing(
                            deferred.evidence(),
                            ExecutorExecutionCapacityStage::SubmissionWave,
                        )?;
                        return self
                            .bind_execution_maintenance_retry(
                                deferral,
                                backing_attempts,
                                &maintenance_receipts,
                                sequences
                                    .iter()
                                    .map(|sequence| sequence.request_id().clone())
                                    .collect(),
                            )
                            .map(VNextExecutionCapacityDecision::Deferred);
                    }
                    backing_attempts += 1;
                    let outcome = deferred
                        .maintain()
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = self.execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SubmissionWave,
                        outcome,
                        VNextExecutionMaintenanceSource::Backing(deferred.evidence()),
                        sequences.iter().map(Arc::as_ref),
                        &mut maintenance_receipts,
                        &mut prefix_maintenance,
                    )? {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                }
                StepSubmissionWaveAdmissionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::backend(format!(
                        "vNext submission wave exceeds its immutable plan: {rejected:?}"
                    )))
                }
                StepSubmissionWaveAdmissionDecision::RequestStateDeferred(deferred) => {
                    let mut request_ids = Vec::new();
                    for sequence in sequences {
                        let request_authority = sequence.session.resources().request_authority();
                        if deferred
                            .blockers()
                            .iter()
                            .any(|blocker| blocker.request() == request_authority)
                            && !request_ids.contains(sequence.request_id())
                        {
                            request_ids.push(sequence.request_id().clone());
                        }
                    }
                    return ExecutorRequestStateDeferral::new(
                        ExecutorExecutionCapacityStage::SubmissionWave,
                        request_ids,
                        deferred,
                    )
                    .map(VNextExecutionCapacityDecision::RequestStateDeferred);
                }
                StepSubmissionWaveAdmissionDecision::RequestStateSplitRequired(split) => {
                    return Err(FerrumError::internal(format!(
                        "vNext product executor reached an unsplit sibling Request-state wave for request {:?}: {:?}",
                        split.request(),
                        split.resource_ids()
                    )))
                }
                StepSubmissionWaveAdmissionDecision::RequestStatePoisoned(poison) => {
                    return Err(FerrumError::backend(format!(
                        "vNext Request-state resource is poisoned: {poison:?}"
                    )))
                }
            }
        }
    }

    fn startup_reusable_descriptor(
        participants: &[VNextExecutionParticipant<'_, R>],
        kind: VNextExecutionWaveKind,
    ) -> Option<VNextReusableExecutionDescriptor> {
        match kind {
            VNextExecutionWaveKind::Decode
                if !participants.is_empty()
                    && participants
                        .iter()
                        .all(|participant| participant.span.immediate_tokens() == 1) =>
            {
                Some(VNextReusableExecutionDescriptor::uniform_decode(
                    participants.len(),
                ))
            }
            VNextExecutionWaveKind::Prefill if participants.len() == 1 => {
                let participant = &participants[0];
                let range = participant.span.immediate_token_range();
                Some(VNextReusableExecutionDescriptor::Prefill {
                    tokens_processed: usize::try_from(range.start).ok()?,
                    token_capacity: usize::try_from(range.end.checked_sub(range.start)?).ok()?,
                    total_prompt_tokens: participant.tokens.len(),
                    request_capacity: 1,
                })
            }
            VNextExecutionWaveKind::Prefill
            | VNextExecutionWaveKind::Decode
            | VNextExecutionWaveKind::Mixed => None,
        }
    }

    fn record_startup_reusable_program(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        kind: VNextExecutionWaveKind,
        program_id: &DeviceReusableExecutionProgramId,
    ) {
        if let Some(descriptor) = Self::startup_reusable_descriptor(participants, kind) {
            self.startup_reusable_programs
                .lock()
                .entry(descriptor)
                .or_default()
                .insert(program_id.clone());
        }
    }

    fn dispatch_participant_wave(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        wave: PreparedStepSubmissionWave<R>,
        kind: VNextExecutionWaveKind,
        output_modes: &ProductReadbackModes,
        token_mask_plans: &[VNextProductTokenMaskSubmissionPlan],
        mut cold_wave: Option<&mut reusable_cold_path::WaveObservation<'_>>,
        resident_bootstrap: Option<Arc<ResidentDecodeReservation<R>>>,
        resident_install: &mut Option<Arc<ResidentDecodeProgram<R>>>,
        facts_capture: ExecutionFactsCapture,
    ) -> DispatchOutcome<R> {
        if participants.is_empty()
            || participants.len() != token_mask_plans.len()
            || participants.len() != output_modes.len()
        {
            return DispatchOutcome::QuiescentFailure(FerrumError::internal(
                "vNext submission wave requires matching participants and token-mask decisions",
            ));
        }
        let active_bindings = || {
            participants
                .iter()
                .map(|participant| participant.sequence.active_binding.as_ref())
        };
        let timing_enabled = self.host_dispatch_timing_enabled();
        let phase_timing = self.metrics.wave_timing_for(kind);
        let uploads = match self.prepare_participant_uploads(
            participants,
            kind,
            output_modes,
            token_mask_plans,
        ) {
            Ok(uploads) => uploads,
            Err(error) => return DispatchOutcome::QuiescentFailure(error),
        };

        let mut wave = wave;
        let mut retries = 0;
        let mut reusable_direct_attempted = false;
        loop {
            let identity = match {
                let _timing = self
                    .metrics
                    .wave_timing
                    .wave_identity_bind
                    .start_if(timing_enabled);
                let _phase_timing = phase_timing.wave_identity_bind.start_if(timing_enabled);
                OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
                    &self.submission_wave_identity,
                    active_bindings(),
                    &wave,
                    &self.lane,
                    self.invocation_preparation_strategy,
                )
            } {
                Ok(identity) => identity,
                Err(error) => {
                    return DispatchOutcome::QuiescentFailure(FerrumError::backend(
                        error.to_string(),
                    ))
                }
            };
            let timing_sink = VNextWaveTimingSink {
                aggregate: &self.metrics.wave_timing,
                phase: phase_timing,
            };
            let device_timing_mode = self.device_timing_mode();
            let execution_policy = submission_execution_policy_for_timing(device_timing_mode);
            let mut reusable_catalog_miss = None;
            let catalog_snapshot = self.reusable_execution_catalog.read().clone();
            let catalog = catalog_snapshot.as_deref();
            let mut diagnostic_program_id = None;
            let reusable_program = if !reusable_program_identity_required(
                self.reusable_execution_startup_plan.is_some(),
                catalog.is_some(),
                device_timing_mode.direct_reusable_execution_allowed(),
                reusable_direct_attempted,
            ) {
                None
            } else {
                let program_id = match OperationDispatch::reusable_execution_program_id_for_wave(
                    self.providers.providers(),
                    &self.resolved_plan,
                    &wave,
                    &self.lane,
                ) {
                    Ok(program_id) => program_id,
                    Err(error) => {
                        return DispatchOutcome::QuiescentFailure(FerrumError::backend(
                            error.to_string(),
                        ));
                    }
                };
                match program_id {
                    Some(program_id) => {
                        if cold_wave.is_some() {
                            diagnostic_program_id = Some(program_id.clone());
                        }
                        if catalog.is_none() {
                            self.record_startup_reusable_program(participants, kind, &program_id);
                        }
                        match catalog {
                            Some(catalog)
                                if catalog.lane_epoch != self.lane.reusable_execution_epoch() =>
                            {
                                reusable_catalog_miss =
                                    Some(VNextReusableExecutionCatalogMissKey::from_program_id(
                                        &program_id,
                                        VNextReusableExecutionCatalogMissReason::EpochMismatch,
                                    ));
                                None
                            }
                            Some(catalog) if catalog.programs.is_empty() => {
                                reusable_catalog_miss =
                                    Some(VNextReusableExecutionCatalogMissKey::from_program_id(
                                        &program_id,
                                        VNextReusableExecutionCatalogMissReason::CatalogEmpty,
                                    ));
                                None
                            }
                            Some(catalog) => match catalog.programs.get(&program_id) {
                                Some(program) if program.has_resident_segments() => Some(program),
                                Some(_) => {
                                    reusable_catalog_miss = Some(
                                        VNextReusableExecutionCatalogMissKey::from_program_id(
                                            &program_id,
                                            VNextReusableExecutionCatalogMissReason::ProgramNonResident,
                                        ),
                                    );
                                    None
                                }
                                None => {
                                    reusable_catalog_miss = Some(
                                        VNextReusableExecutionCatalogMissKey::from_program_id(
                                            &program_id,
                                            VNextReusableExecutionCatalogMissReason::ProgramAbsent,
                                        ),
                                    );
                                    None
                                }
                            },
                            None => None,
                        }
                    }
                    None if catalog.is_some() => {
                        reusable_catalog_miss = Some(
                            VNextReusableExecutionCatalogMissKey::without_program_identity(
                                wave.claimed_backing().work_shape(),
                                VNextReusableExecutionCatalogMissReason::ProgramIdentityUnavailable,
                            ),
                        );
                        None
                    }
                    None => None,
                }
            };
            *resident_install = None;
            if let (Some(reservation), Some(program)) =
                (resident_bootstrap.as_ref(), reusable_program)
            {
                match OperationDispatch::try_compile_resident_decode(
                    Arc::clone(reservation),
                    &self.resolved_plan,
                    &wave,
                    &self.submission_wave_identity,
                    program,
                    &self.lane,
                ) {
                    Ok(compiled) => *resident_install = compiled,
                    Err(error) => {
                        return DispatchOutcome::QuiescentFailure(FerrumError::backend(
                            error.to_string(),
                        ))
                    }
                }
            }
            reusable_direct_attempted |= reusable_program.is_some();
            if reusable_program.is_none() && self.on_demand_reusable_execution_enabled() {
                self.reusable_execution_catalog_refresh_needed
                    .store(true, Ordering::Release);
            }
            let reusable_program_stats = reusable_program.map(|program| {
                (
                    program.segments().len() as u64,
                    program
                        .segments()
                        .iter()
                        .map(|segment| {
                            u64::from(segment.end_node_index() - segment.start_node_index())
                        })
                        .sum::<u64>(),
                    program.per_wave_binding_node_indices().len() as u64,
                )
            });
            let observation = timing_enabled.then(|| {
                reusable_cold_path::AttemptObservation::new(
                    &timing_sink,
                    diagnostic_program_id,
                    self.lane.reusable_execution_epoch(),
                    catalog.map(|catalog| catalog.lane_epoch),
                    reusable_catalog_miss.map(|miss| miss.reason),
                )
            });
            let mut provider_elapsed = None;
            let submission = {
                let timing = self
                    .metrics
                    .wave_timing
                    .provider_encode_submit
                    .start_if(timing_enabled);
                let phase_timer = phase_timing.provider_encode_submit.start_if(timing_enabled);
                let submission = if let Some(reusable_program) = reusable_program {
                    if let Some(timing_sink) = observation.as_ref() {
                        OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_preparation(
                            self.providers.providers(),
                            &self.resolved_plan,
                            &identity,
                            active_bindings(),
                            device_timing_mode,
                            facts_capture,
                            &uploads,
                            reusable_program,
                            execution_policy,
                            self.invocation_preparation_strategy,
                            timing_sink,
                            &self.metrics.invocation_preparation,
                            wave,
                            &self.lane,
                            &self.reaper,
                        )
                        .map(ProfiledSubmissionHandle::into_parts)
                    } else {
                        OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_preparation(
                            self.providers.providers(),
                            &self.resolved_plan,
                            &identity,
                            active_bindings(),
                            device_timing_mode,
                            facts_capture,
                            &uploads,
                            reusable_program,
                            execution_policy,
                            self.invocation_preparation_strategy,
                            &invocation_preparation::DisabledTiming,
                            &self.metrics.invocation_preparation,
                            wave,
                            &self.lane,
                            &self.reaper,
                        )
                        .map(ProfiledSubmissionHandle::into_parts)
                    }
                } else if let Some(timing_sink) = observation.as_ref() {
                    OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
                        self.providers.providers(),
                        &self.resolved_plan,
                        &identity,
                        active_bindings(),
                        device_timing_mode,
                        facts_capture,
                        &uploads,
                        execution_policy,
                        self.invocation_preparation_strategy,
                        timing_sink,
                        &self.metrics.invocation_preparation,
                        wave,
                        &self.lane,
                        &self.reaper,
                    )
                    .map(ProfiledSubmissionHandle::into_parts)
                } else {
                    OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
                        self.providers.providers(),
                        &self.resolved_plan,
                        &identity,
                        active_bindings(),
                        device_timing_mode,
                        facts_capture,
                        &uploads,
                        execution_policy,
                        self.invocation_preparation_strategy,
                        &invocation_preparation::DisabledTiming,
                        &self.metrics.invocation_preparation,
                        wave,
                        &self.lane,
                        &self.reaper,
                    )
                    .map(ProfiledSubmissionHandle::into_parts)
                };
                drop(phase_timer);
                if cold_wave.is_some() {
                    provider_elapsed = timing.and_then(|timer| timer.finish());
                }
                submission
            };
            if let (Some(cold_wave), Some(observation)) = (cold_wave.as_mut(), observation) {
                let outcome = match &submission {
                    Ok(_) => reusable_cold_path::Outcome::Submitted,
                    Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { .. }) => {
                        reusable_cold_path::Outcome::DefinitelyNotSubmitted
                    }
                    Err(SubmissionWaveDispatchError::Contract(_)) => {
                        reusable_cold_path::Outcome::ContractError
                    }
                    Err(SubmissionWaveDispatchError::Provider(_)) => {
                        reusable_cold_path::Outcome::ProviderError
                    }
                    Err(SubmissionWaveDispatchError::Initialization(_)) => {
                        reusable_cold_path::Outcome::InitializationError
                    }
                    Err(SubmissionWaveDispatchError::InputUpload(_)) => {
                        reusable_cold_path::Outcome::InputUploadError
                    }
                    Err(SubmissionWaveDispatchError::SubmissionIndeterminate { .. }) => {
                        reusable_cold_path::Outcome::SubmissionIndeterminate
                    }
                    Err(SubmissionWaveDispatchError::PostSubmitContract { .. }) => {
                        reusable_cold_path::Outcome::PostSubmitContract
                    }
                };
                cold_wave.record_attempt(
                    observation,
                    identity.preparation_snapshot(),
                    outcome,
                    provider_elapsed,
                );
            }
            match submission {
                Ok((completion, observation)) => {
                    let identity_materialization = identity.materialization_snapshot();
                    self.metrics.submitted_waves.fetch_add(1, Ordering::Relaxed);
                    self.metrics.identity_waves.fetch_add(1, Ordering::Relaxed);
                    self.metrics.identity_logical_nodes.fetch_add(
                        u64::from(identity_materialization.logical_nodes()),
                        Ordering::Relaxed,
                    );
                    self.metrics
                        .identity_nodes_materialized_before_submit
                        .fetch_add(
                            u64::from(identity_materialization.materialized_nodes()),
                            Ordering::Relaxed,
                        );
                    self.metrics
                        .identity_full_participant_materializations_before_submit
                        .fetch_add(
                            u64::from(identity_materialization.full_participant_projection()),
                            Ordering::Relaxed,
                        );
                    if let Some((segments, logical_nodes, binding_nodes)) = reusable_program_stats {
                        self.metrics
                            .direct_reusable_waves
                            .fetch_add(1, Ordering::Relaxed);
                        self.metrics
                            .direct_reusable_segments
                            .fetch_add(segments, Ordering::Relaxed);
                        self.metrics
                            .direct_reusable_logical_nodes
                            .fetch_add(logical_nodes, Ordering::Relaxed);
                        self.metrics
                            .direct_reusable_binding_nodes
                            .fetch_add(binding_nodes, Ordering::Relaxed);
                    } else if let Some(miss) = reusable_catalog_miss {
                        self.metrics.record_reusable_catalog_miss(miss);
                    }
                    return DispatchOutcome::Submitted {
                        completion,
                        observation,
                    };
                }
                Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { failures, retry })
                    if retries < MAX_DEFINITELY_NOT_SUBMITTED_RETRIES =>
                {
                    retries += 1;
                    if reusable_program_stats.is_some() {
                        self.metrics
                            .direct_reusable_fallbacks
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    self.metrics
                        .definitely_not_submitted_retries
                        .fetch_add(1, Ordering::Relaxed);
                    match retry.retry() {
                        Ok(retry_wave) => wave = retry_wave,
                        Err(error) => {
                            return DispatchOutcome::QuiescentFailure(FerrumError::backend(
                                format!(
                                    "vNext wave retry authority failed after {failures:?}: {error}"
                                ),
                            ))
                        }
                    }
                }
                Err(error @ SubmissionWaveDispatchError::DefinitelyNotSubmitted { .. })
                | Err(error @ SubmissionWaveDispatchError::Contract(_))
                | Err(error @ SubmissionWaveDispatchError::Provider(_))
                | Err(error @ SubmissionWaveDispatchError::Initialization(_))
                | Err(error @ SubmissionWaveDispatchError::InputUpload(_)) => {
                    return DispatchOutcome::QuiescentFailure(FerrumError::backend(
                        error.to_string(),
                    ))
                }
                Err(SubmissionWaveDispatchError::SubmissionIndeterminate { recovery }) => {
                    return DispatchOutcome::SubmissionIndeterminate {
                        message: "vNext wave submission is indeterminate".to_owned(),
                        recovery,
                    }
                }
                Err(SubmissionWaveDispatchError::PostSubmitContract {
                    error,
                    completion,
                    observation,
                }) => {
                    return DispatchOutcome::PostSubmitContract {
                        message: error.to_string(),
                        completion,
                        observation,
                    }
                }
            }
        }
    }

    fn checkpoint_token_evidence_enabled(&self) -> bool {
        self.prefix_restore_enabled()
    }

    fn retain_checkpoint_token_evidence(
        &self,
        span: TokenSpanWork,
        tokens: &[u32],
    ) -> Result<TokenSpanWork> {
        prefix_cache::retain_token_evidence(self.resolved_plan.execution_plan(), span, tokens)
    }

    async fn execute_batch_step(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        sequences: &[Arc<VNextSequence<R>>],
        token_batches: &[Vec<u32>],
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
        output_roles: &[VNextParticipantOutputRole],
    ) -> std::result::Result<
        VNextExecutionCapacityDecision<RetiredWave<Vec<ExecutorSamplingOutput>, R>>,
        WaveFailure,
    > {
        if sequences.is_empty()
            || sequences.len() != token_batches.len()
            || sequences.len() != spans.len()
            || output_roles.len() != sequences.len()
            || output_roles
                .iter()
                .zip(spans)
                .zip(token_batches)
                .any(|((role, span), tokens)| !role.matches_frontier(kind, span, tokens.len()))
            || sequences.len() != batch.sessions().len()
            || batch
                .sessions()
                .iter()
                .zip(sequences)
                .any(|(session, sequence)| !Arc::ptr_eq(session, &sequence.session))
        {
            return Err(WaveFailure::not_submitted(FerrumError::internal(
                "vNext decode batch differs from its canonical participant set",
            )));
        }
        let retained_spans = if self.checkpoint_token_evidence_enabled() {
            Some(
                spans
                    .iter()
                    .zip(token_batches)
                    .map(|(span, tokens)| {
                        self.retain_checkpoint_token_evidence(span.clone(), tokens)
                    })
                    .collect::<Result<Vec<_>>>()
                    .map_err(WaveFailure::not_submitted)?,
            )
        } else {
            None
        };
        // Unsupported/default plans keep the original slice and do not copy
        // full token inputs or repeat their hash calculation.
        let spans = retained_spans.as_deref().unwrap_or(spans);
        let participants = sequences
            .iter()
            .zip(token_batches)
            .zip(spans)
            .enumerate()
            .map(
                |(participant_index, ((sequence, tokens), span))| VNextExecutionParticipant {
                    sequence,
                    tokens,
                    span,
                    output_role: &output_roles[participant_index],
                },
            )
            .collect::<Vec<_>>();
        let prepared = {
            let _timing = self.metrics.wave_timing.resource_prepare_attempt.start();
            let _phase_timing = self
                .metrics
                .wave_timing_for(kind)
                .resource_prepare_attempt
                .start();
            if let Some(prepared) = self
                .try_prepare_resident_decode(
                    batch,
                    spans,
                    kind,
                    output_roles.iter().all(|role| {
                        matches!(
                            role.logits_policy(),
                            Some(LogitsReturnPolicy::GreedyArgmax { .. })
                        )
                    }),
                )
                .map_err(WaveFailure::not_submitted)?
            {
                prepared
            } else {
                let step = match self
                    .begin_step_for_spans_with_capacity(batch, sequences, spans, kind)
                    .map_err(WaveFailure::not_submitted)?
                {
                    VNextExecutionCapacityDecision::Ready(step) => step,
                    VNextExecutionCapacityDecision::Deferred(deferred) => {
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred))
                    }
                    VNextExecutionCapacityDecision::RequestStateDeferred(deferred) => {
                        return Ok(VNextExecutionCapacityDecision::RequestStateDeferred(
                            deferred,
                        ))
                    }
                };
                let wave_decision = match self
                    .prepare_wave_for_spans_with_capacity(&step, sequences, spans, kind)
                {
                    Ok(decision) => decision,
                    Err(error) => {
                        let mut guard = PreparedWaveGuard::new(self, step.into(), &participants);
                        if let Err(cleanup_error) = guard.rollback_unsubmitted() {
                            return Err(guard
                                .fail(FerrumError::backend(format!("{error}; {cleanup_error}"))));
                        }
                        return Err(WaveFailure::not_submitted(error));
                    }
                };
                let wave = match wave_decision {
                    VNextExecutionCapacityDecision::Ready(wave) => wave,
                    VNextExecutionCapacityDecision::Deferred(deferred) => {
                        let mut guard = PreparedWaveGuard::new(self, step.into(), &participants);
                        guard
                            .rollback_unsubmitted()
                            .map_err(|error| guard.fail(error))?;
                        return Ok(VNextExecutionCapacityDecision::Deferred(deferred));
                    }
                    VNextExecutionCapacityDecision::RequestStateDeferred(deferred) => {
                        let mut guard = PreparedWaveGuard::new(self, step.into(), &participants);
                        guard
                            .rollback_unsubmitted()
                            .map_err(|error| guard.fail(error))?;
                        return Ok(VNextExecutionCapacityDecision::RequestStateDeferred(
                            deferred,
                        ));
                    }
                };
                PreparedVNextPrefill::ordinary(step, wave)
            }
        };
        self.execute_prepared_participants(&participants, prepared, kind)
            .await
            .map(VNextExecutionCapacityDecision::Ready)
    }

    async fn execute_prepared_participants(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        prepared: PreparedVNextPrefill<R>,
        kind: VNextExecutionWaveKind,
    ) -> std::result::Result<RetiredWave<Vec<ExecutorSamplingOutput>, R>, WaveFailure> {
        let _execution_timing = self.metrics.wave_timing.submitted_wave_total.start();
        let phase_timing = self.metrics.wave_timing_for(kind);
        let _phase_execution_timing = phase_timing.submitted_wave_total.start();
        let PreparedVNextPrefill {
            retirement,
            dispatch,
        } = prepared;
        let mut guard = PreparedWaveGuard::new(self, retirement, participants);
        // Prepared wave owners must drop before the guard publishes recovery,
        // including a panic in an external observer's capture query.
        let mut wave = dispatch;
        // Snapshot once for this attempt, before dispatch. Late attachment does
        // not retarget a submitted observation or recreate missing work facts.
        let observation_sink = self.event_sink.read().clone();
        let facts_capture = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            observation_sink
                .as_ref()
                .map_or(ExecutionFactsCapture::Off, |sink| {
                    sink.execution_facts_capture()
                })
        }))
        .map_err(|_| {
            guard.fail(FerrumError::backend(
                "execution observer capture contract panicked",
            ))
        })?;
        let legacy_attribution = self.host_dispatch_timing_enabled();
        let mut resident_install = None;
        let capture_claim = match (kind, self.checkpoint_capture.as_ref()) {
            (VNextExecutionWaveKind::Prefill, Some(capture)) => {
                let first = participants.first();
                let is_final_prefill = first.is_some_and(|participant| {
                    participant.span.immediate_token_range().end
                        == participant.span.full_input_tokens()
                });
                capture
                    .claim_prefill_wave(
                        participants.len(),
                        first.map(|participant| participant.sequence.request_id()),
                        first.map(|participant| participant.tokens),
                        is_final_prefill,
                    )
                    .map_err(|error| guard.fail(error))?
            }
            (VNextExecutionWaveKind::Decode, Some(capture)) => capture
                .claim_decode_wave(
                    participants.len(),
                    participants
                        .first()
                        .map(|participant| participant.sequence.request_id()),
                    participants.first().map(|participant| participant.tokens),
                )
                .map_err(|error| guard.fail(error))?,
            (VNextExecutionWaveKind::Mixed, Some(_)) => {
                return Err(guard.fail(FerrumError::internal(
                    "mixed execution must fall back before diagnostic checkpoint capture",
                )));
            }
            (_, None) => None,
        };
        let teacher_forced_decision = match (capture_claim, self.checkpoint_capture.as_ref()) {
            (Some(claim), Some(capture)) => capture
                .teacher_forced_decision(claim)
                .map_err(|error| guard.fail(error))?,
            _ => None,
        };
        let output_modes = ProductReadbackModes::resolve(
            self.sampling_readback_mode,
            kind,
            self.io.output_layout.element_type(),
            self.io.output_elements,
            self.checkpoint_capture.is_some(),
            teacher_forced_decision.is_some(),
            participants
                .iter()
                .map(|participant| participant.output_role),
        );
        let readbacks = self
            .prepare_terminal_readbacks(participants, capture_claim, &output_modes)
            .map_err(|error| guard.fail(error))?;
        if let Some(request) = output_modes.uniform().and_then(|mode| {
            readbacks.terminal_packet_request(
                self.completion_readback_policy,
                mode,
                capture_claim.is_some(),
            )
        }) {
            wave.set_terminal_readbacks(request.clone())
                .map_err(|error| guard.fail(FerrumError::backend(error.to_string())))?;
        }
        let mut cold_wave = self
            .metrics
            .reusable_cold_path
            .begin(self.host_dispatch_timing_enabled(), kind);
        let (mut token_mask_residency, dispatch) = loop {
            let timing = self.metrics.wave_timing.host_encode_submit.start();
            let phase_timer = phase_timing.host_encode_submit.start();
            // Product-fixed inputs live in the lane-stable Step arena. One Step
            // slot lease is exclusive through terminal completion, while a
            // released slot retains its physical backing for later waves. Bind
            // residency to that exact slot and participant range; Invocation
            // program-binding slots refer to different backing and are not used.
            let token_mask_slot_identity = self
                .io
                .token_mask_residency_eligible
                .then(|| wave.lane_stable_slot_identity())
                .flatten();
            let token_mask_residency = VNextProductTokenMaskResidencyTransaction::prepare(
                &self.product_token_mask_residency,
                token_mask_slot_identity,
                participants
                    .iter()
                    .zip(output_modes.iter())
                    .map(|(participant, output_mode)| {
                        VNextProductTokenMaskContent::from_policy(
                            participant.output_role.logits_policy(),
                            output_mode,
                            self.io.output_elements,
                        )
                    }),
            );
            let dispatch = self.dispatch_prepared_wave(
                participants,
                wave,
                kind,
                &output_modes,
                token_mask_residency.plans(),
                cold_wave.as_mut(),
                &mut resident_install,
                facts_capture,
            );
            let dispatch = match dispatch {
                PreparedDispatchResult::Outcome(dispatch) => dispatch,
                PreparedDispatchResult::Unavailable => {
                    let mut token_mask_residency = token_mask_residency;
                    token_mask_residency.invalidate_before_slot_release();
                    self.release_resident_decode();
                    guard
                        .rollback_unsubmitted()
                        .map_err(|error| guard.fail(error))?;
                    let sequences = participants
                        .iter()
                        .map(|p| Arc::clone(p.sequence))
                        .collect::<Vec<_>>();
                    let spans = participants
                        .iter()
                        .map(|p| p.span.clone())
                        .collect::<Vec<_>>();
                    let batch = ExecutionBatchParticipants::new(
                        sequences.iter().map(|s| Arc::clone(&s.session)).collect(),
                    )
                    .map_err(|e| guard.fail(FerrumError::backend(e.to_string())))?;
                    let ordinary = self
                        .begin_step_for_spans(&batch, &sequences, &spans)
                        .map_err(|error| guard.fail(error))?;
                    guard.replace_retirement(PreparedRetirement::Ordinary(Arc::clone(&ordinary)));
                    let mut ordinary_wave =
                        match self.prepare_wave_for_spans(&ordinary, &sequences, &spans) {
                            Ok(wave) => wave,
                            Err(error) => return Err(guard.fail(error)),
                        };
                    if let Some(request) = output_modes.uniform().and_then(|mode| {
                        readbacks.terminal_packet_request(
                            self.completion_readback_policy,
                            mode,
                            capture_claim.is_some(),
                        )
                    }) {
                        ordinary_wave
                            .set_terminal_readbacks(request.clone())
                            .map_err(|e| guard.fail(FerrumError::backend(e.to_string())))?;
                    }
                    drop(ordinary);
                    wave = PreparedDispatch::Ordinary {
                        wave: ordinary_wave,
                        reservation: None,
                    };
                    continue;
                }
            };
            drop(phase_timer);
            if let Some(cold_wave) = cold_wave.take() {
                cold_wave.finish(
                    timing.finish(),
                    reusable_cold_path::Outcome::from_dispatch(&dispatch),
                );
            }
            break (token_mask_residency, dispatch);
        };
        let mut execution_event_error = None;
        let mut event_error_is_terminal_observation = false;
        let completion = match dispatch {
            DispatchOutcome::Submitted {
                completion,
                observation,
            } => {
                guard.observe_submission(&completion);
                guard.set_submission_observation(
                    observation_sink.clone(),
                    facts_capture,
                    observation,
                    legacy_attribution,
                );
                completion
            }
            DispatchOutcome::QuiescentFailure(error) => {
                token_mask_residency.invalidate_before_slot_release();
                return Err(guard.fail(error));
            }
            DispatchOutcome::SubmissionIndeterminate { message, recovery } => {
                guard.observe_indeterminate(&recovery);
                let route = guard.worker_route();
                let reaper = Arc::clone(&self.reaper);
                let recovered = self
                    .completion_worker
                    .execute(VNextCompletionTaskKind::IndeterminateRecovery, move || {
                        let recovered = recovery.recover_by_draining_lane();
                        drop(reaper);
                        route.wrap(recovered)
                    })
                    .await
                    .map_err(|error| {
                        guard.fail(FerrumError::backend(format!("{message}: {error}")))
                    })?
                    .into_inner(&mut guard);
                let message = match &recovered {
                    Ok(CompletionRecoveryOutcome::Drained(_)) => message,
                    Ok(CompletionRecoveryOutcome::Quarantined(_)) => {
                        format!("{message}; recovery retained quarantined work")
                    }
                    Err(error) => format!("{message}; recovery failed: {error}"),
                };
                guard.accept_recovery(recovered);
                token_mask_residency.invalidate_before_slot_release();
                self.metrics.record_failure(message.clone());
                return Err(guard.fail(FerrumError::backend(message)));
            }
            DispatchOutcome::PostSubmitContract {
                message,
                completion,
                observation,
            } => {
                guard.observe_submission(&completion);
                guard.set_submission_observation(
                    observation_sink.clone(),
                    facts_capture,
                    observation,
                    legacy_attribution,
                );
                let route = guard.worker_route();
                let reaper = Arc::clone(&self.reaper);
                let observed = self
                    .completion_worker
                    .execute(VNextCompletionTaskKind::PostSubmitDrain, move || {
                        let observed = completion.wait();
                        drop(reaper);
                        route.wrap(observed)
                    })
                    .await
                    .map_err(|error| {
                        guard.fail(FerrumError::backend(format!("{message}: {error}")))
                    })?
                    .into_inner(&mut guard);
                let message = match &observed {
                    Ok(CompletionObservation::Terminal(_)) => message,
                    Ok(other) => {
                        format!("{message}; post-submit drain remained nonterminal: {other:?}")
                    }
                    Err(error) => format!("{message}; drain failed: {error}"),
                };
                guard.accept_observation(observed);
                token_mask_residency.invalidate_before_slot_release();
                self.metrics.record_failure(message.clone());
                return Err(guard.fail(FerrumError::backend(message)));
            }
        };
        for participant in participants {
            if let Some(events) = &participant.sequence.events {
                if let Err(error) = events.lock().submitted(completion.receipt()) {
                    execution_event_error.get_or_insert_with(|| error.to_string());
                }
            }
        }
        let diagnostic_failure_requested =
            self.claim_prefill_resource_diagnostic_fault(participants, kind);
        let mut diagnostic_failure_observed = false;
        if diagnostic_failure_requested {
            let observation = self
                .plan_runtime_resource_snapshot()
                .and_then(|snapshot| {
                    snapshot.ok_or_else(|| {
                        FerrumError::internal(
                            "vNext diagnostic resource failure lacks a plan runtime snapshot",
                        )
                    })
                })
                .and_then(|snapshot| {
                    participants[0]
                        .sequence
                        .events
                        .as_ref()
                        .ok_or_else(|| {
                            FerrumError::internal(
                                "vNext diagnostic resource failure lacks an execution journal",
                            )
                        })?
                        .lock()
                        .observe_resource_failure(
                            snapshot,
                            "diagnostic_resource_after_submit",
                            "typed diagnostic resource failure after prefill submission",
                        )
                        .map_err(|error| FerrumError::backend(error.to_string()))
                });
            match observation {
                Ok(_) => diagnostic_failure_observed = true,
                Err(error) => {
                    execution_event_error.get_or_insert_with(|| error.to_string());
                }
            }
        }

        // Take the optional hint before the completion handle moves to the
        // worker. The ticket owns no Step or Sequence resources.
        let segment_publication = completion.take_segment_binding_publication();
        let route = guard.worker_route();
        let reaper = Arc::clone(&self.reaper);
        let observation = {
            let _timing = self.metrics.wave_timing.completion_round_trip.start();
            let _phase_timing = phase_timing.completion_round_trip.start();
            self.completion_worker
                .execute(VNextCompletionTaskKind::WaveReadback, move || {
                    let observation = match readbacks {
                        VNextTerminalReadbacks::Batch(request) => {
                            completion.wait_with_readbacks(request)
                        }
                        VNextTerminalReadbacks::Collection(request) => {
                            completion.wait_with_readback_collection(request)
                        }
                        VNextTerminalReadbacks::Selection(request) => {
                            completion.wait_with_readback_selection(request)
                        }
                    };
                    drop(reaper);
                    route.wrap(observation)
                })
                .await
                .map_err(|error| {
                    guard.fail(FerrumError::backend(format!(
                        "vNext completion task failed: {error}"
                    )))
                })?
                .into_inner(&mut guard)
        };
        let _postprocess_timing = self.metrics.wave_timing.host_postprocess.start();
        let _phase_postprocess_timing = phase_timing.host_postprocess.start();
        let observation_error = match &observation {
            Ok(CompletionReadbackBatchObservation::Terminal(_)) => None,
            Ok(other) => Some(nonterminal_completion_message(other)),
            Err(error) => Some(error.to_string()),
        };
        guard.accept_readback_observation(observation);
        if let Some(error) = guard.observation_error() {
            if execution_event_error.is_none() {
                execution_event_error = Some(error.to_string());
                event_error_is_terminal_observation = true;
            }
        }
        if let Some(message) = observation_error {
            self.metrics.record_failure(message.clone());
            token_mask_residency.invalidate_before_slot_release();
            return Err(guard.fail(FerrumError::backend(message)));
        }
        self.refresh_on_demand_reusable_execution_catalog()
            .map_err(|error| guard.fail(error))?;
        let receipt = guard.readbacks();
        self.metrics.device_timing.record(receipt);
        self.metrics.device_timing_for(kind).record(receipt);
        if !diagnostic_failure_requested {
            for participant in participants {
                if let Some(events) = &participant.sequence.events {
                    if let Err(error) = events.lock().completed(receipt.completion()) {
                        execution_event_error.get_or_insert_with(|| error.to_string());
                    }
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
            token_mask_residency.invalidate_before_slot_release();
            return Err(guard.fail(FerrumError::backend(message)));
        }
        // A terminally successful command batch proves every encoded mask
        // upload reached this still-owned Step slot. Host output processing and
        // event attribution cannot mutate that input backing.
        token_mask_residency.publish();
        token_mask_residency.settle_success();
        if diagnostic_failure_requested {
            if diagnostic_failure_observed {
                if let Err(error) = participants[0]
                    .sequence
                    .events
                    .as_ref()
                    .expect("observed diagnostic failure has an execution journal")
                    .lock()
                    .settle_failed_submission(receipt.completion())
                {
                    execution_event_error.get_or_insert_with(|| error.to_string());
                }
            }
            let retired = guard.retire_normal(())?;
            if let Some(error) = execution_event_error {
                participants[0].sequence.abort();
                let message =
                    format!("vNext diagnostic resource failure attribution failed closed: {error}");
                self.metrics.record_failure(message.clone());
                let failure = retired.retirement.fail(FerrumError::backend(message));
                return Err(
                    guard.attach_observation_failure(failure, event_error_is_terminal_observation)
                );
            }
            if let Err(error) = participants[0].sequence.abort_after_observed_failure() {
                return Err(retired.retirement.fail(error));
            }
            let message = "typed diagnostic resource failure after prefill submission".to_string();
            self.metrics.record_failure(message.clone());
            return Err(retired
                .retirement
                .fail(FerrumError::resource_exhausted(message)));
        }
        // Only this successful, non-diagnostic terminal may qualify a hint.
        // Readback/output/event/retirement failures below still discard it.
        let ready_segment_publication =
            segment_publication.and_then(|ticket| ticket.bind_completion(receipt.completion()));
        let processed = (|| -> Result<(
            Vec<ExecutorSamplingOutput>,
            u64,
            u64,
            Vec<VNextCheckpointArtifactRecord>,
            Vec<VNextCheckpointProductOutputRecord>,
        )> {
            let mut logits = vec![None; participants.len()];
            let mut readback_bytes = 0_u64;
            let mut product_readback_bytes = 0_u64;
            let mut checkpoint_records = Vec::new();
            let mut product_output_records = Vec::new();
            for disposition in receipt.dispositions() {
                let CompletionReadbackDisposition::Succeeded(output) = disposition else {
                    return Err(FerrumError::backend(format!(
                        "vNext terminal readback failed: {disposition:?}"
                    )));
                };
                let request = output.request();
                let output_bytes = request.output_layout().byte_len()
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
                readback_bytes = readback_bytes.saturating_add(output_bytes);
                let participant_index = usize::try_from(request.participant_index())
                    .map_err(|_| FerrumError::internal("vNext readback participant index exceeds usize"))?;
                let output_mode = output_modes.at(participant_index).ok_or_else(|| {
                    FerrumError::internal("vNext readback participant index exceeds submitted participants")
                })?;
                let is_product_output = match output_mode {
                    VNextProductOutputMode::FullLogits => {
                        request.node_id() == &self.io.output_node_id
                            && request.resource_id() == &self.io.output_resource_id
                            && request.logical_offset_bytes() == self.io.output_offset_bytes
                            && request.output_layout() == self.io.output_layout
                    }
                    VNextProductOutputMode::GreedyToken => {
                        request.node_id() == &self.io.greedy_token_output_node_id
                            && request.resource_id() == &self.io.greedy_token_output_resource_id
                            && request.logical_offset_bytes()
                                == self.io.greedy_token_output_offset_bytes
                            && request.output_layout() == self.io.greedy_token_output_layout
                    }
                };
                let checkpoint = capture_claim
                    .and_then(|_| self.checkpoint_capture.as_ref())
                    .and_then(|capture| capture.checkpoint_for_output(output));
                if !is_product_output && checkpoint.is_none() {
                    return Err(FerrumError::internal(format!(
                        "vNext terminal readback returned unowned node/resource {}/{}",
                        request.node_id(),
                        request.resource_id()
                    )));
                }
                if is_product_output {
                    product_readback_bytes = product_readback_bytes.saturating_add(output_bytes);
                    let participant_index =
                        usize::try_from(request.participant_index()).map_err(|_| {
                            FerrumError::internal(
                                "vNext product-output participant index exceeds usize",
                            )
                        })?;
                    let slot = logits.get_mut(participant_index).ok_or_else(|| {
                        FerrumError::internal(
                            "vNext product-output participant index exceeds submitted participants",
                        )
                    })?;
                    if slot.is_some() {
                        return Err(FerrumError::internal(
                            "vNext terminal readback returned duplicate participant output",
                        ));
                    }
                    *slot = Some(if output_modes.uniform().is_none()
                        && output_mode == VNextProductOutputMode::GreedyToken
                        && output.bytes() == u32::MAX.to_le_bytes()
                    {
                        // The scalar sentinel describes this request's sampling
                        // result. Preserve the previous mixed host path's error
                        // isolation instead of aborting other completed rows.
                        ExecutorSamplingOutput::NoSelectableFiniteLogit
                    } else {
                        self.decode_product_output(output.bytes(), output_mode)?
                    });
                    if let (Some(capture_claim), Some(capture)) =
                        (capture_claim, self.checkpoint_capture.as_ref())
                    {
                        if capture.captures_product_output() {
                            let participant =
                                participants.get(participant_index).ok_or_else(|| {
                                    FerrumError::internal(
                                        "vNext product-output checkpoint participant index exceeds submitted participants",
                                    )
                                })?;
                            product_output_records.push(capture.write_product_output(
                                capture_claim,
                                participant.sequence.request_id(),
                                participant.span,
                                output_mode.checkpoint_mode(),
                                output,
                            )?);
                        }
                    }
                }
                if let (Some(capture_claim), Some(capture), Some(checkpoint)) =
                    (capture_claim, self.checkpoint_capture.as_ref(), checkpoint)
                {
                    let participant_index =
                        usize::try_from(request.participant_index()).map_err(|_| {
                            FerrumError::internal(
                                "vNext checkpoint participant index exceeds usize",
                            )
                        })?;
                    let participant = participants.get(participant_index).ok_or_else(|| {
                        FerrumError::internal(
                            "vNext checkpoint participant index exceeds submitted participants",
                        )
                    })?;
                    checkpoint_records.push(capture.write_output(
                        capture_claim,
                        participant.sequence.request_id(),
                        participant.span,
                        checkpoint,
                        output,
                    )?);
                }
            }
            let logits = logits
                .into_iter()
                .enumerate()
                .map(|(participant_index, logits)| {
                    logits.ok_or_else(|| {
                        FerrumError::internal(format!(
                            "vNext terminal readback omitted participant {participant_index} product output"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((
                logits,
                readback_bytes,
                product_readback_bytes,
                checkpoint_records,
                product_output_records,
            ))
        })();
        let (
            mut logits,
            readback_bytes,
            product_readback_bytes,
            checkpoint_records,
            product_output_records,
        ) = match processed {
            Ok(processed) => processed,
            Err(error) => {
                return Err(guard.fail(error));
            }
        };
        if let (Some(capture_claim), Some(capture)) =
            (capture_claim, self.checkpoint_capture.as_ref())
        {
            if let Err(error) = capture.finish_wave(
                capture_claim,
                participants.len(),
                receipt.completion().fingerprint(),
                receipt.fingerprint(),
                checkpoint_records,
                product_output_records,
            ) {
                return Err(guard.fail(error));
            }
        }
        if let Some(decision) = teacher_forced_decision {
            if let Err(error) = apply_teacher_forced_decision(&mut logits, decision) {
                return Err(guard.fail(error));
            }
        }
        self.metrics
            .readback_bytes
            .fetch_add(readback_bytes, Ordering::Relaxed);
        self.metrics
            .no_selectable_finite_logit_participants
            .fetch_add(
                logits
                    .iter()
                    .filter(|output| {
                        matches!(output, ExecutorSamplingOutput::NoSelectableFiniteLogit)
                    })
                    .count() as u64,
                Ordering::Relaxed,
            );
        for plan in token_mask_residency.plans() {
            if plan.upload_required {
                self.metrics
                    .token_mask_upload_participants
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                self.metrics
                    .token_mask_cache_hit_participants
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        let (sparse_repetition_participants, sparse_repetition_token_ids) = participants
            .iter()
            .zip(output_modes.iter())
            .map(|(participant, output_mode)| {
                product_repetition_input(participant.output_role.logits_policy(), output_mode)
            })
            .filter(|input| input.is_active())
            .fold((0_u64, 0_u64), |(participants, token_ids), input| {
                (
                    participants.saturating_add(1),
                    token_ids.saturating_add(input.token_ids.len() as u64),
                )
            });
        if sparse_repetition_participants != 0 {
            self.metrics
                .sparse_repetition_waves
                .fetch_add(1, Ordering::Relaxed);
            self.metrics
                .sparse_repetition_participants
                .fetch_add(sparse_repetition_participants, Ordering::Relaxed);
            self.metrics
                .sparse_repetition_token_ids_uploaded
                .fetch_add(sparse_repetition_token_ids, Ordering::Relaxed);
        }
        let full_participants = output_modes.full_participants() as u64;
        let greedy_participants = participants.len() as u64 - full_participants;
        self.metrics
            .full_logits_readback_participants
            .fetch_add(full_participants, Ordering::Relaxed);
        self.metrics
            .greedy_token_readback_participants
            .fetch_add(greedy_participants, Ordering::Relaxed);
        self.metrics
            .product_readback_bytes
            .fetch_add(product_readback_bytes, Ordering::Relaxed);
        match output_modes.uniform() {
            Some(VNextProductOutputMode::FullLogits) => {
                self.metrics
                    .full_logits_readback_waves
                    .fetch_add(1, Ordering::Relaxed);
            }
            Some(VNextProductOutputMode::GreedyToken) => {
                self.metrics
                    .greedy_token_readback_waves
                    .fetch_add(1, Ordering::Relaxed);
            }
            None => {
                self.metrics
                    .hybrid_readback_waves
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        if kind != VNextExecutionWaveKind::Prefill
            && participants
                .iter()
                .zip(output_modes.iter())
                .any(|(participant, mode)| {
                    mode == VNextProductOutputMode::FullLogits
                        && matches!(
                            participant.output_role.logits_policy(),
                            Some(LogitsReturnPolicy::GreedyArgmax { .. })
                        )
                })
        {
            self.metrics
                .greedy_policy_fallback_waves
                .fetch_add(1, Ordering::Relaxed);
        }
        let retired = guard.retire_normal(logits)?;
        self.metrics.completed_waves.fetch_add(1, Ordering::Relaxed);
        if let Some(error) = execution_event_error {
            let message = format!("vNext execution event emission failed: {error}");
            self.metrics.record_failure(message.clone());
            let failure = retired.retirement.fail(FerrumError::backend(message));
            return Err(
                guard.attach_observation_failure(failure, event_error_is_terminal_observation)
            );
        }
        if let Some(ready) = ready_segment_publication {
            // Publication checks exact retirement and current entry.
            // A stale or cancelled hint cannot fail a completed wave.
            let _ = ready.publish(retired.retirement.receipt());
        }
        if let Some(program) = resident_install {
            self.install_resident_decode(program);
        }
        Ok(retired)
    }

    fn prepare_terminal_readbacks(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        capture_claim: Option<VNextCheckpointClaim>,
        output_modes: &ProductReadbackModes,
    ) -> Result<VNextTerminalReadbacks> {
        if output_modes.len() != participants.len() {
            return Err(FerrumError::internal(
                "vNext readback decisions must cover every submitted participant",
            ));
        }
        let Some(output_mode) = output_modes.uniform() else {
            if capture_claim.is_some() {
                return Err(FerrumError::internal(
                    "vNext participant readbacks cannot replace checkpoint capture",
                ));
            }
            let alternatives = [
                VNextProductOutputMode::FullLogits,
                VNextProductOutputMode::GreedyToken,
            ]
            .into_iter()
            .map(|mode| self.product_readback_batch(participants.len(), mode))
            .collect::<Result<Vec<_>>>()?;
            let selected = output_modes
                .iter()
                .map(|mode| match mode {
                    VNextProductOutputMode::FullLogits => 0,
                    VNextProductOutputMode::GreedyToken => 1,
                })
                .collect();
            return CompletionReadbackSelectionRequest::new(alternatives, selected)
                .map(VNextTerminalReadbacks::Selection)
                .map_err(|error| FerrumError::backend(error.to_string()));
        };
        let product_readbacks = self.product_readback_batch(participants.len(), output_mode)?;
        let mut readback_batches = vec![product_readbacks.clone()];
        if capture_claim.is_some() {
            let capture = self.checkpoint_capture.as_ref().ok_or_else(|| {
                FerrumError::internal("vNext checkpoint capture index has no capture owner")
            })?;
            let token_spans = participants
                .iter()
                .map(|participant| participant.span)
                .collect::<Vec<_>>();
            for batch in
                capture.readback_batches(self.resolved_plan.execution_plan(), &token_spans)?
            {
                let first = &batch.requests()[0];
                let product_first = &product_readbacks.requests()[0];
                if first.node_id() == product_first.node_id()
                    && first.resource_id() == product_first.resource_id()
                    && first.logical_offset_bytes() == product_first.logical_offset_bytes()
                {
                    if batch != product_readbacks {
                        return Err(FerrumError::internal(
                            "retained output readback differs from the product output layout",
                        ));
                    }
                    continue;
                }
                readback_batches.push(batch);
            }
        }
        if readback_batches.len() == 1 {
            return Ok(VNextTerminalReadbacks::Batch(
                readback_batches.pop().ok_or_else(|| {
                    FerrumError::internal("vNext terminal readback batch disappeared")
                })?,
            ));
        }
        CompletionReadbackCollectionRequest::new(readback_batches)
            .map(VNextTerminalReadbacks::Collection)
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn product_readback_batch(
        &self,
        participant_count: usize,
        output_mode: VNextProductOutputMode,
    ) -> Result<CompletionReadbackBatchRequest> {
        let (output_node_id, output_resource_id, output_offset_bytes, output_layout) =
            match output_mode {
                VNextProductOutputMode::FullLogits => (
                    &self.io.output_node_id,
                    &self.io.output_resource_id,
                    self.io.output_offset_bytes,
                    self.io.output_layout,
                ),
                VNextProductOutputMode::GreedyToken => (
                    &self.io.greedy_token_output_node_id,
                    &self.io.greedy_token_output_resource_id,
                    self.io.greedy_token_output_offset_bytes,
                    self.io.greedy_token_output_layout,
                ),
            };
        let product_readbacks = (0..participant_count)
            .map(|participant_index| {
                CompletionReadbackRequest::new(
                    output_node_id.clone(),
                    u32::try_from(participant_index).map_err(|_| {
                        FerrumError::backend("vNext readback participant index exceeds u32")
                    })?,
                    output_resource_id.clone(),
                    output_offset_bytes,
                    output_layout,
                )
                .map_err(|error| FerrumError::backend(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        CompletionReadbackBatchRequest::new(product_readbacks)
            .map_err(|error| FerrumError::backend(error.to_string()))
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

    fn decode_product_output(
        &self,
        bytes: &[u8],
        output_mode: VNextProductOutputMode,
    ) -> Result<ExecutorSamplingOutput> {
        match output_mode {
            VNextProductOutputMode::FullLogits => ExecutorSamplingOutput::full_logits(
                Self::decode_logits(bytes, self.io.output_element_type)?,
            ),
            VNextProductOutputMode::GreedyToken => Ok(ExecutorSamplingOutput::greedy_token(
                decode_selected_token(bytes, self.io.output_elements)?,
            )),
        }
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
        let product_readback = serde_json::json!({
            "mode": self.sampling_readback_mode,
            "full_logits_waves": self.metrics.full_logits_readback_waves.load(Ordering::Relaxed),
            "greedy_token_waves": self.metrics.greedy_token_readback_waves.load(Ordering::Relaxed),
            "hybrid_waves": self.metrics.hybrid_readback_waves.load(Ordering::Relaxed),
            "full_logits_participants": self.metrics.full_logits_readback_participants.load(Ordering::Relaxed),
            "greedy_token_participants": self.metrics.greedy_token_readback_participants.load(Ordering::Relaxed),
            "no_selectable_finite_logit_participants": self.metrics.no_selectable_finite_logit_participants.load(Ordering::Relaxed),
            "product_bytes": self.metrics.product_readback_bytes.load(Ordering::Relaxed),
            "greedy_policy_fallback_waves": self.metrics.greedy_policy_fallback_waves.load(Ordering::Relaxed),
            "token_mask_residency_eligible": self.io.token_mask_residency_eligible,
            "token_mask_upload_participants": self.metrics.token_mask_upload_participants.load(Ordering::Relaxed),
            "token_mask_cache_hit_participants": self.metrics.token_mask_cache_hit_participants.load(Ordering::Relaxed),
            "sparse_repetition_waves": self.metrics.sparse_repetition_waves.load(Ordering::Relaxed),
            "sparse_repetition_participants": self.metrics.sparse_repetition_participants.load(Ordering::Relaxed),
            "sparse_repetition_token_ids_uploaded": self.metrics.sparse_repetition_token_ids_uploaded.load(Ordering::Relaxed),
        });
        let mut snapshot = serde_json::json!({
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
                "reusable_execution": {
                    "direct_waves": self.metrics.direct_reusable_waves.load(Ordering::Relaxed),
                    "catalog": self.reusable_execution_catalog_snapshot(),
                    "direct_segments": self.metrics.direct_reusable_segments.load(Ordering::Relaxed),
                    "direct_logical_nodes": self.metrics.direct_reusable_logical_nodes.load(Ordering::Relaxed),
                    "direct_binding_nodes": self.metrics.direct_reusable_binding_nodes.load(Ordering::Relaxed),
                    "direct_fallbacks": self.metrics.direct_reusable_fallbacks.load(Ordering::Relaxed),
                    "catalog_misses": self.metrics.reusable_catalog_misses.load(Ordering::Relaxed),
                    "catalog_epoch_misses": self.metrics.reusable_catalog_epoch_misses.load(Ordering::Relaxed),
                    "catalog_miss_ledger": self.metrics.reusable_catalog_miss_ledger.lock().snapshot(),
                    "cold_path": self.metrics.reusable_cold_path.snapshot(),
                },
                "identity_materialization": {
                    "waves": self.metrics.identity_waves.load(Ordering::Relaxed),
                    "logical_nodes": self.metrics.identity_logical_nodes.load(Ordering::Relaxed),
                    "nodes_materialized_before_submit": self.metrics.identity_nodes_materialized_before_submit.load(Ordering::Relaxed),
                    "full_participant_materializations_before_submit": self.metrics.identity_full_participant_materializations_before_submit.load(Ordering::Relaxed),
                },
                "definitely_not_submitted_retries": self.metrics.definitely_not_submitted_retries.load(Ordering::Relaxed),
                "request_deferrals": self.metrics.request_deferrals.load(Ordering::Relaxed),
                "sequence_deferrals": self.metrics.sequence_deferrals.load(Ordering::Relaxed),
                "extension_deferrals": self.metrics.extension_deferrals.load(Ordering::Relaxed),
                "step_deferrals": self.metrics.step_deferrals.load(Ordering::Relaxed),
                "wave_deferrals": self.metrics.wave_deferrals.load(Ordering::Relaxed),
                "backing_deferrals": self.metrics.backing_deferrals.load(Ordering::Relaxed),
                "uploaded_bytes": self.metrics.uploaded_bytes.load(Ordering::Relaxed),
                "readback_bytes": self.metrics.readback_bytes.load(Ordering::Relaxed),
                "product_readback": product_readback,
            },
            "wave_timing": self.metrics.wave_timing.snapshot(),
            "wave_timing_by_phase": {
                VNextExecutionWaveKind::Prefill.as_str(): self.metrics.prefill_wave_timing.snapshot(),
                VNextExecutionWaveKind::Decode.as_str(): self.metrics.decode_wave_timing.snapshot(),
                VNextExecutionWaveKind::Mixed.as_str(): self.metrics.mixed_wave_timing.snapshot(),
            },
            "device_timing": self.metrics.device_timing.snapshot(),
            "device_timing_by_phase": {
                VNextExecutionWaveKind::Prefill.as_str(): self.metrics.prefill_device_timing.snapshot(),
                VNextExecutionWaveKind::Decode.as_str(): self.metrics.decode_device_timing.snapshot(),
                VNextExecutionWaveKind::Mixed.as_str(): self.metrics.mixed_device_timing.snapshot(),
            },
            "completion_worker": self.completion_worker.metrics_snapshot(),
            "checkpoint_timings": self.reaper.checkpoint_timing_snapshot(),
            "dynamic_pools": pool_status,
            "deferred_cleanup": cleanup,
            "startup_preparation": serde_json::to_value(&*self.startup_preparation.lock())
                .unwrap_or_else(|error| serde_json::json!({"state": "serialization_failed", "message": error.to_string()})),
            "last_failure": self.metrics.last_failure.lock().clone(),
        });
        let fields = snapshot
            .as_object_mut()
            .expect("vNext executor snapshot is an object");
        // Cache health consumes these top-level fields. Always publish native
        // values, including disabled/zero, so text-LCP fallback cannot stand in
        // for sequence checkpoint reuse on this executor.
        fields.extend(self.prefix_cache_metrics_snapshot());
        fields.insert(
            "program_binding_upload".to_owned(),
            self.metrics
                .program_binding_upload
                .snapshot(self.runtime.program_binding_upload_snapshot()),
        );
        fields.insert(
            "invocation_preparation".to_owned(),
            self.metrics.invocation_preparation.snapshot(),
        );
        fields.insert(
            "invocation_preparation_strategy".to_owned(),
            serde_json::json!(self.invocation_preparation_strategy),
        );
        fields.insert(
            "resident_decode".to_owned(),
            serde_json::json!({
                "submitted": self.resident_metrics.submitted.load(Ordering::Relaxed),
                "bootstraps": self.resident_metrics.bootstraps.load(Ordering::Relaxed),
                "installations": self.resident_metrics.installations.load(Ordering::Relaxed),
                "unavailable": self.resident_metrics.unavailable.load(Ordering::Relaxed),
            }),
        );
        fields.insert(
            "completion_readback_policy".to_owned(),
            serde_json::json!(self.completion_readback_policy),
        );
        fields.insert(
            "sampling_readback_mode".to_owned(),
            serde_json::json!(self.sampling_readback_mode),
        );
        fields.insert(
            "attention_execution_policy".to_owned(),
            serde_json::json!(self.policy.attention_execution()),
        );
        fields.insert(
            "numerical_execution".to_owned(),
            serde_json::json!(self.resolved_plan.parts().numerical_execution),
        );
        let numerical = &self.resolved_plan.parts().numerical_execution;
        fields.insert(
            "kv_storage".to_owned(),
            serde_json::json!({
                "source": "resolved_model_plan",
                "requested": numerical.requested_kv_storage(),
                "selected": numerical.selected_kv_storage(),
                "numerical_profile": numerical.selected_profile(),
                "logical_sequence_state": self.sequence_state_memory,
                "logical_bytes_scope": "complete_model_excluding_alignment_workspace_and_checkpoint_copies",
                "resident_usage_source": "dynamic_pools",
            }),
        );
        snapshot
    }
}

#[async_trait::async_trait]
impl<R: DeviceRuntime> ModelExecutor for ProgramExecutor<R> {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &crate::executor_lifecycle::ExecutorShutdownPermit,
    ) -> crate::executor_lifecycle::ExecutorShutdownObserver {
        shutdown::begin(self, permit)
    }

    fn plan_prompt_tail_capture_boundary(&self, chunk: PrefillChunk) -> Option<PrefixCapturePlan> {
        self.prompt_tail_boundary(chunk)
    }

    fn plan_prefix_capture_boundary(
        &self,
        input: PrefixCaptureBoundary<'_>,
    ) -> Option<PrefixCapturePlan> {
        self.rendezvous_boundary(input)
    }

    fn retain_prefix_capture_interest(
        &self,
        input: PrefixCaptureRequest<'_>,
    ) -> Result<Option<Arc<dyn PrefixCaptureLease>>> {
        self.arm_prefix_capture(input)
    }

    fn supports_plan_runtime_prefix_restore(&self) -> bool {
        self.prefix_restore_enabled()
    }

    async fn try_restore_plan_runtime_prefix(
        &self,
        input: PlanRuntimePrefixRestoreInput<'_>,
    ) -> Result<PlanRuntimePrefixRestoreOutcome> {
        self.restore_prefix(input).await
    }

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
                if let Some(capture) = &self.checkpoint_capture {
                    capture.arm();
                }
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

    fn admission_limits(&self) -> Result<Option<ExecutorAdmissionLimits>> {
        ExecutorAdmissionLimits::new(
            self.policy.memory().maximum_active_sequences,
            self.policy.admission().maximum_scheduled_tokens,
        )
        .map(Some)
        .map_err(|reason| {
            FerrumError::internal(format!(
                "resolved vNext admission limits violated their typed contract: {reason}"
            ))
        })
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

    fn startup_memory_plan(&self) -> Option<&ferrum_types::StartupMemoryPlan> {
        self.startup_memory_plan.as_ref()
    }

    fn attach_execution_event_sink(&self, sink: Arc<dyn ProductExecutionEventSink>) {
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
        availability: &mut Vec<ferrum_runtime::CapacityAvailabilityEpoch>,
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
        self.admit_bound_prefill(input, false)
    }

    async fn admit_tokens(
        &self,
        input: crate::token_execution::ExecutionAdmission<'_>,
    ) -> Result<crate::token_execution::TokenAdmissionOutcome> {
        self.admit_native_tokens(input)
    }

    async fn execute_tokens(
        &self,
        work: &[crate::token_execution::ExecutionWork<'_>],
    ) -> Result<crate::token_execution::ExecutionBatchOutcome> {
        self.execute_native_tokens(work).await
    }

    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: crate::token_execution::ExecutionEndReason,
    ) -> Result<crate::token_execution::ExecutionCloseOutcome> {
        self.finish_native_tokens(sequence, reason).await
    }

    fn cancel_tokens(
        &self,
        sequence: &ExecutorSequence,
    ) -> Result<crate::token_execution::ExecutionCloseOutcome> {
        self.cancel_native_tokens(sequence)
    }

    fn write_token_capacity_release_sources(
        &self,
        sequence: &ExecutorSequence,
        sources: &mut Vec<CapacityAvailabilitySource>,
    ) -> Result<bool> {
        self.write_native_token_release_sources(sequence, sources)
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
                maintenance_boundary: _,
            } => {
                if self.evict_prefix_for_wait(&wait_condition) {
                    return Ok(ExecutorPrefillMaintenanceOutcome::RetryAdmission {
                        current: self.current_execution_capacity_epochs()?,
                    });
                }
                Ok(ExecutorPrefillMaintenanceOutcome::WaitForRelease {
                    current: ExecutorAdmissionEpochs::from_capacity(current_epochs),
                    wait_condition,
                    pressure,
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
                kv_cache_memory_per_token: 0,
                typed_sequence_state: Some(self.sequence_state_memory),
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

    fn execution_attribution_snapshot(&self) -> Option<serde_json::Value> {
        self.static_provider_attribution
            .as_ref()
            .and_then(|witness| serde_json::to_value(witness).ok())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::time::Duration;

    use super::{
        apply_teacher_forced_decision, bounded_wall_anchor, budget_reusable_decode_seed_prefill,
        decode_selected_token, is_language_masked_argmax_operation,
        is_language_token_embedding_operation, journal_clock_anchor_required,
        nonterminal_completion_message, normalized_product_token_mask,
        padded_repetition_token_id_bytes, product_output_mode_for_roles, product_repetition_input,
        reported_allocated_bytes, resolve_reusable_execution_policy,
        resolve_runtime_attention_authority, resolved_sequence_fit_policy,
        reusable_catalog_lookup_allowed, reusable_executable_inventory_matches,
        reusable_execution_program_catalog_is_usable, reusable_execution_requires_eager_fallback,
        reusable_program_identity_required, reusable_startup_case_budget_violation,
        submission_execution_policy_for_timing, validate_sequence_completion_accounting,
        AdmissionFitPolicy, FerrumError, SequenceFitPolicy, VNextDeviceTimingMetrics,
        VNextExecutionWaveKind, VNextParticipantOutputRole, VNextPhysicalSpanTimingMetrics,
        VNextPreparedWaveTopologyMetrics, VNextProductOutputMode, VNextProductTokenMaskContent,
        VNextProductTokenMaskResidency, VNextProductTokenMaskResidencyTransaction,
        VNextProductTokenMaskSlotIdentity, VNextProductTokenMaskSlotTarget,
        VNextReusableExecutionCatalogMissKey, VNextReusableExecutionCatalogMissLedger,
        VNextReusableExecutionCatalogMissReason, VNextReusableExecutionDescriptor,
        VNextReusableExecutionMetrics, VNextReusableExecutionStartupPlan,
        VNextTeacherForcedDecision, VNextWaveTimingMetrics, VNextWaveTimingSink,
        MAX_PRODUCT_TOKEN_MASK_SLOT_CACHE_ENTRIES, MAX_REUSABLE_EXECUTION_CATALOG_MISS_KEYS,
    };
    use crate::model_executor::{
        ExecutorSamplingOutput, ExecutorSequenceCompletion, GreedyRepetitionPenalty,
        LogitsReturnPolicy, PrefillChunk, TokenSelectionMask,
    };
    use ferrum_runtime::{
        CompletionReadbackBatchObservation, DeviceComputePathRequirement, DeviceExecutionInterval,
        DeviceExecutionIntervalKind, DeviceExecutionSpanKind, DeviceReusableExecutionObservation,
        DeviceReusableExecutionPlan, DeviceReusableExecutionPreparation,
        DeviceReusableExecutionTopologyFingerprint, DeviceSubmissionExecutionSpan,
        DeviceSubmissionExecutionTiming, DeviceSubmissionTimingSink, DeviceTimingMeasurement,
        DeviceTimingMode, StepResourceAdmissionProfilePhase, DENSE_SWIGLU_OPERATION_ID,
        LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID, LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
        TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID, TOKEN_EMBEDDING_OPERATION_ID,
    };
    use ferrum_types::{
        AttentionExecutionPolicy, RequestId, ReusableExecutionCaptureConfig, TokenId,
    };

    fn resolve_test_reusable_startup_plan(
        maximum_active_sequences: u32,
        maximum_scheduled_tokens: u64,
        maximum_model_tokens: usize,
        prefill_chunks: &[PrefillChunk],
        execution_node_count: usize,
        capture_config: &ReusableExecutionCaptureConfig,
    ) -> ferrum_types::Result<VNextReusableExecutionStartupPlan> {
        let resolution = resolve_reusable_execution_policy(
            maximum_active_sequences,
            maximum_scheduled_tokens,
            maximum_model_tokens,
            prefill_chunks,
            capture_config,
            true,
        )?;
        let program_policy = resolution
            .policy
            .program_policy()
            .cloned()
            .ok_or_else(|| FerrumError::internal("test capture policy is missing"))?;
        let maximum_device_executables = u64::try_from(
            execution_node_count
                .checked_mul(program_policy.programs().len())
                .ok_or_else(|| FerrumError::internal("test executable capacity overflowed"))?,
        )
        .map_err(|_| FerrumError::internal("test executable capacity exceeds u64"))?;
        VNextReusableExecutionStartupPlan::resolve(
            &program_policy,
            resolution
                .capture
                .ok_or_else(|| FerrumError::internal("test capture resolution is missing"))?,
            maximum_model_tokens,
            prefill_chunks,
            maximum_device_executables,
        )
    }

    #[test]
    fn language_io_resolution_accepts_legacy_and_fp32_master_contracts_only() {
        for operation_id in [
            TOKEN_EMBEDDING_OPERATION_ID,
            TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
        ] {
            assert!(is_language_token_embedding_operation(operation_id));
        }
        for operation_id in [
            LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
            LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
        ] {
            assert!(is_language_masked_argmax_operation(operation_id));
        }
        assert!(!is_language_token_embedding_operation(
            DENSE_SWIGLU_OPERATION_ID
        ));
        assert!(!is_language_masked_argmax_operation(
            DENSE_SWIGLU_OPERATION_ID
        ));
    }

    #[test]
    fn runtime_attention_authority_rejects_plan_provider_drift() {
        assert_eq!(
            resolve_runtime_attention_authority(
                AttentionExecutionPolicy::Auto,
                true,
                AttentionExecutionPolicy::NativeAdaptive,
            )
            .unwrap(),
            AttentionExecutionPolicy::NativeAdaptive
        );
        assert!(resolve_runtime_attention_authority(
            AttentionExecutionPolicy::Auto,
            true,
            AttentionExecutionPolicy::Portable,
        )
        .is_err());
        assert!(resolve_runtime_attention_authority(
            AttentionExecutionPolicy::Portable,
            false,
            AttentionExecutionPolicy::Auto,
        )
        .is_err());
    }

    #[test]
    fn verification_timing_selects_typed_eager_submission_without_owning_other_paths() {
        assert_eq!(
            submission_execution_policy_for_timing(DeviceTimingMode::Verification).compute_path(),
            DeviceComputePathRequirement::EagerOnly
        );
        for timing_mode in [
            DeviceTimingMode::Off,
            DeviceTimingMode::Completion,
            DeviceTimingMode::Replay,
            DeviceTimingMode::Kernel,
        ] {
            assert_eq!(
                submission_execution_policy_for_timing(timing_mode).compute_path(),
                DeviceComputePathRequirement::Adaptive
            );
        }
    }

    #[test]
    fn reusable_catalog_lookup_requires_a_startup_plan_and_one_direct_attempt() {
        assert!(reusable_catalog_lookup_allowed(true, true, false));
        assert!(!reusable_catalog_lookup_allowed(false, true, false));
        assert!(!reusable_catalog_lookup_allowed(true, false, false));
        assert!(!reusable_catalog_lookup_allowed(true, true, true));
    }

    #[test]
    fn reusable_program_identity_is_recorded_before_catalog_installation() {
        for timing_mode in [DeviceTimingMode::Kernel, DeviceTimingMode::Verification] {
            assert!(reusable_program_identity_required(
                true,
                false,
                timing_mode.direct_reusable_execution_allowed(),
                false,
            ));
            assert!(!reusable_program_identity_required(
                true,
                true,
                timing_mode.direct_reusable_execution_allowed(),
                false,
            ));
        }
        for timing_mode in [
            DeviceTimingMode::Off,
            DeviceTimingMode::Completion,
            DeviceTimingMode::Replay,
        ] {
            assert_eq!(
                reusable_program_identity_required(
                    true,
                    true,
                    timing_mode.direct_reusable_execution_allowed(),
                    false,
                ),
                reusable_catalog_lookup_allowed(
                    true,
                    timing_mode.direct_reusable_execution_allowed(),
                    false,
                ),
            );
        }
        assert!(reusable_program_identity_required(true, false, false, true));
        assert!(!reusable_program_identity_required(
            false, false, true, false
        ));

        assert!(reusable_program_identity_required(true, true, true, false));
        assert!(!reusable_program_identity_required(true, true, true, true));
    }

    #[test]
    fn reusable_startup_case_budget_allows_observed_many_to_one_program_identity() {
        let requested_cases = BTreeSet::from(["prefill-a", "prefill-b"]);
        let shared_program = BTreeSet::from(["physical-program"]);
        let observations = BTreeMap::from([
            ("prefill-a", shared_program.clone()),
            ("prefill-b", shared_program),
        ]);
        assert!(
            reusable_startup_case_budget_violation(&requested_cases, &observations).is_none(),
            "each logical case actually observed the same stable physical program"
        );

        let invalid = BTreeMap::from([(
            "prefill-a",
            BTreeSet::from(["physical-program-a", "physical-program-b"]),
        )]);
        assert!(reusable_startup_case_budget_violation(&requested_cases, &invalid).is_some());
    }

    #[test]
    fn nonterminal_completion_message_preserves_typed_failure_class() {
        assert_eq!(
            nonterminal_completion_message(
                &CompletionReadbackBatchObservation::ObservationPanicked
            ),
            "vNext completion did not reach a quiescent terminal: ObservationPanicked"
        );
    }

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
    fn reusable_catalog_miss_ledger_is_bounded_and_preserves_overflow_reason() {
        let mut ledger = VNextReusableExecutionCatalogMissLedger::default();
        for width in 1..=(MAX_REUSABLE_EXECUTION_CATALOG_MISS_KEYS + 1) {
            ledger.record(VNextReusableExecutionCatalogMissKey {
                immediate_sequences: u32::try_from(width).unwrap(),
                immediate_tokens: u64::try_from(width).unwrap(),
                immediate_pages: 1,
                topology_fingerprint: DeviceReusableExecutionTopologyFingerprint::static_program(),
                reason: VNextReusableExecutionCatalogMissReason::ProgramAbsent,
            });
        }
        ledger.record(VNextReusableExecutionCatalogMissKey {
            immediate_sequences: 1,
            immediate_tokens: 1,
            immediate_pages: 1,
            topology_fingerprint: DeviceReusableExecutionTopologyFingerprint::static_program(),
            reason: VNextReusableExecutionCatalogMissReason::ProgramAbsent,
        });

        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.maximum_distinct_keys, 64);
        assert_eq!(snapshot.distinct_keys, 64);
        assert_eq!(snapshot.rows[0].count, 2);
        assert_eq!(snapshot.overflow.len(), 1);
        assert_eq!(
            snapshot.overflow[0].reason,
            VNextReusableExecutionCatalogMissReason::ProgramAbsent
        );
        assert_eq!(snapshot.overflow[0].count, 1);

        ledger.reset();
        let empty = ledger.snapshot();
        assert_eq!(empty.distinct_keys, 0);
        assert!(empty.rows.is_empty());
        assert!(empty.overflow.is_empty());
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
        let chunk_64 = PrefillChunk::new(0, 64, 64).unwrap();
        let plan = resolve_test_reusable_startup_plan(
            32,
            2_048,
            128,
            &[chunk_64, chunk_64],
            23,
            &ReusableExecutionCaptureConfig::default(),
        )
        .unwrap();

        assert_eq!(plan.decode_widths(), (1..=32).rev().collect::<Vec<_>>());
        assert_eq!(plan.prefill_token_counts(), [64]);
        assert_eq!(plan.prefill_chunks(), [chunk_64]);
        assert_eq!(plan.maximum_decode_sequence_tokens, 97);
        assert_eq!(plan.device_plan.maximum_executables(), 759);
        assert!(!plan.decode_catalog_omits_admitted_widths());
        assert_eq!(
            plan.descriptors.last(),
            Some(&VNextReusableExecutionDescriptor::prefill(chunk_64))
        );

        let chunk_7 = PrefillChunk::new(0, 7, 7).unwrap();
        let chunk_4 = PrefillChunk::new(0, 4, 4).unwrap();
        let non_power_of_two = resolve_test_reusable_startup_plan(
            7,
            7,
            64,
            &[chunk_7, chunk_4],
            2,
            &ReusableExecutionCaptureConfig::default(),
        )
        .unwrap();
        assert_eq!(
            non_power_of_two.decode_widths(),
            (1..=7).rev().collect::<Vec<_>>()
        );
        assert_eq!(non_power_of_two.prefill_token_counts(), [7, 4]);
        assert_eq!(non_power_of_two.maximum_decode_sequence_tokens, 22);
        assert_eq!(non_power_of_two.device_plan.maximum_executables(), 18);

        let wider_admission = resolve_test_reusable_startup_plan(
            65,
            2_048,
            128,
            &[],
            1,
            &ReusableExecutionCaptureConfig::default(),
        )
        .unwrap();
        assert_eq!(
            wider_admission.decode_widths(),
            (1..=32).rev().collect::<Vec<_>>()
        );
        assert!(wider_admission.decode_catalog_omits_admitted_widths());
        assert!(wider_admission
            .capture_resolution
            .reduction_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("bounded at width 32")));

        let explicit_wide = ReusableExecutionCaptureConfig {
            exact_decode_widths: Some(vec![1, 2, 4, 8, 16, 32, 64, 65]),
            ..ReusableExecutionCaptureConfig::default()
        };
        assert!(
            resolve_test_reusable_startup_plan(65, 2_048, 128, &[], 1, &explicit_wide).is_err()
        );

        let unbounded_explicit = ReusableExecutionCaptureConfig {
            exact_decode_widths: Some(vec![u32::MAX as usize]),
            ..ReusableExecutionCaptureConfig::default()
        };
        assert!(resolve_reusable_execution_policy(
            u32::MAX,
            u64::MAX,
            usize::MAX,
            &[],
            &unbounded_explicit,
            true,
        )
        .is_err());

        let short_model = resolve_test_reusable_startup_plan(
            32,
            2_048,
            19,
            &[],
            1,
            &ReusableExecutionCaptureConfig::default(),
        )
        .unwrap();
        assert_eq!(short_model.decode_widths(), [32, 16, 8, 4, 2, 1]);
        assert_eq!(short_model.maximum_decode_sequence_tokens, 19);
        assert_eq!(short_model.device_plan.maximum_executables(), 6);
        assert!(short_model.capture_resolution.reduction_reason.is_some());
        assert!(short_model.decode_catalog_omits_admitted_widths());
        assert!(resolve_test_reusable_startup_plan(
            32,
            2_048,
            18,
            &[],
            1,
            &ReusableExecutionCaptureConfig::default(),
        )
        .is_err());
        let explicit_all = ReusableExecutionCaptureConfig {
            exact_decode_widths: Some((1..=32).collect()),
            ..ReusableExecutionCaptureConfig::default()
        };
        assert!(resolve_test_reusable_startup_plan(32, 2_048, 19, &[], 1, &explicit_all).is_err());
    }

    #[test]
    fn reusable_execution_on_demand_keeps_capacity_without_synthetic_token_budget() {
        let config = ReusableExecutionCaptureConfig {
            preparation: ferrum_types::ReusableExecutionPreparationMode::OnDemand,
            ..Default::default()
        };
        let plan = resolve_test_reusable_startup_plan(32, 32, 2, &[], 7, &config).unwrap();
        assert_eq!(plan.maximum_decode_sequence_tokens, 0);
        assert_eq!(plan.decode_widths(), (1..=32).rev().collect::<Vec<_>>());
        assert_eq!(plan.device_plan.maximum_executables(), 32 * 7);
        assert_eq!(
            plan.device_plan.catalog_lifetime(),
            ferrum_runtime::ReusableExecutionCatalogLifetime::OnDemandBounded
        );
        assert_eq!(plan.program_policy.replay_validation_passes(), 0);
        assert!(resolve_test_reusable_startup_plan(
            32,
            32,
            2,
            &[],
            7,
            &ReusableExecutionCaptureConfig::default()
        )
        .is_err());
    }

    #[test]
    fn reusable_execution_startup_plan_preserves_exact_chunk_boundaries() {
        let chunk_single = PrefillChunk::new(0, 1, 1).unwrap();
        let chunk_multi = PrefillChunk::new(0, 4, 4).unwrap();
        let chunk_boundary = PrefillChunk::new(4, 4, 8).unwrap();
        let plan = resolve_test_reusable_startup_plan(
            32,
            2_048,
            128,
            &[chunk_single, chunk_multi, chunk_boundary],
            23,
            &ReusableExecutionCaptureConfig::default(),
        )
        .unwrap();

        assert_eq!(
            plan.prefill_chunks(),
            [chunk_multi, chunk_boundary, chunk_single]
        );
        assert_eq!(plan.prefill_token_counts(), [4, 1]);
        assert_eq!(plan.prefill_wave_shapes(), 4);
        assert_eq!(plan.device_plan.maximum_executables(), 828);
    }

    #[test]
    fn reusable_execution_product_policy_budgets_decode_seed_prefill() {
        let seed = PrefillChunk::new(0, 1, 1).unwrap();
        let mut enabled_chunks = Vec::new();
        budget_reusable_decode_seed_prefill(&mut enabled_chunks, true).unwrap();
        assert_eq!(enabled_chunks, [seed]);

        let plan = resolve_test_reusable_startup_plan(
            32,
            2_048,
            128,
            &enabled_chunks,
            23,
            &ReusableExecutionCaptureConfig::default(),
        )
        .unwrap();
        assert!(plan
            .descriptors
            .contains(&VNextReusableExecutionDescriptor::prefill(seed)));

        let mut disabled_chunks = Vec::new();
        budget_reusable_decode_seed_prefill(&mut disabled_chunks, false).unwrap();
        assert!(disabled_chunks.is_empty());
    }

    #[test]
    fn reusable_execution_safe_misses_use_eager_fallback_without_inventory_drift() {
        let plan = DeviceReusableExecutionPlan::new(4).unwrap();
        let captured =
            DeviceReusableExecutionPreparation::preparing_with_progress(plan, 2, 1, 2, 2, 0)
                .unwrap();
        let replayed =
            DeviceReusableExecutionPreparation::preparing_with_progress(plan, 2, 1, 2, 2, 3)
                .unwrap();
        let sealed = DeviceReusableExecutionPreparation::ready(plan, 2, 1, 2, 2, 3).unwrap();

        assert!(reusable_executable_inventory_matches(captured, replayed));
        assert!(reusable_executable_inventory_matches(replayed, sealed));
        assert!(reusable_execution_requires_eager_fallback(sealed));
    }

    #[test]
    fn reusable_execution_complete_preparation_needs_no_eager_fallback() {
        let plan = DeviceReusableExecutionPlan::new(4).unwrap();
        let prepared = DeviceReusableExecutionPreparation::ready(plan, 2, 0, 2, 2, 0).unwrap();
        let drifted = DeviceReusableExecutionPreparation::ready(plan, 1, 0, 2, 2, 0).unwrap();

        assert!(!reusable_execution_requires_eager_fallback(prepared));
        assert!(!reusable_executable_inventory_matches(prepared, drifted));
    }

    #[test]
    fn resident_reusable_segments_require_a_typed_program_catalog() {
        let plan = DeviceReusableExecutionPlan::new(4).unwrap();
        let resident = DeviceReusableExecutionPreparation::ready(plan, 2, 0, 2, 2, 0).unwrap();
        let eager_only = DeviceReusableExecutionPreparation::ready(plan, 0, 0, 0, 0, 0).unwrap();

        assert!(!reusable_execution_program_catalog_is_usable(resident, 0));
        assert!(reusable_execution_program_catalog_is_usable(resident, 1));
        assert!(reusable_execution_program_catalog_is_usable(eager_only, 0));
    }

    #[test]
    fn reusable_workspace_policy_is_backend_neutral_and_bucketed() {
        let chunks = [
            PrefillChunk::new(0, 128, 128).unwrap(),
            PrefillChunk::new(0, 64, 64).unwrap(),
        ];
        let resolution = resolve_reusable_execution_policy(
            16,
            2_048,
            4_096,
            &chunks,
            &ReusableExecutionCaptureConfig::default(),
            true,
        )
        .unwrap();
        let policy = resolution.policy;
        let decode_widths = policy
            .buckets()
            .iter()
            .filter(|bucket| bucket.class_id().as_str() == super::UNIFORM_QUERY_REUSABLE_CLASS)
            .map(|bucket| bucket.capacity().maximum_sequences())
            .collect::<Vec<_>>();
        let prefill_tokens = policy
            .buckets()
            .iter()
            .filter(|bucket| bucket.class_id().as_str() == super::PACKED_TOKEN_REUSABLE_CLASS)
            .map(|bucket| bucket.capacity().maximum_tokens())
            .collect::<Vec<_>>();

        assert_eq!(policy.maximum_reusable_lanes(), 1);
        assert_eq!(decode_widths, [1, 2, 4, 8, 16]);
        assert_eq!(prefill_tokens, [64, 128]);
    }

    #[test]
    fn wave_timing_snapshot_exposes_honest_host_boundaries() {
        let snapshot = VNextWaveTimingMetrics::default().snapshot();

        assert_eq!(snapshot["clock"], "host_monotonic");
        assert_eq!(snapshot["resource_prepare_attempt"]["samples"], 0);
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["collection"],
            "profile_attached_only"
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_request_prepare"]["samples"],
            0
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_admission"]["samples"],
            0
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_admission_breakdown"]["demand_evaluate"]
                ["samples"],
            0
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["submission_wave_prepare"]["samples"],
            0
        );
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
    fn wave_timing_enqueue_calls_preserve_cpu_availability_phase_and_reset() {
        use ferrum_runtime::{DeviceEnqueueStage, DeviceSubmissionStage};

        let aggregate = VNextWaveTimingMetrics::default();
        let decode = VNextWaveTimingMetrics::default();
        let prefill = VNextWaveTimingMetrics::default();
        let sink = VNextWaveTimingSink {
            aggregate: &aggregate,
            phase: &decode,
        };
        sink.record_device_submission(
            DeviceSubmissionStage::EnqueueCommands,
            Duration::from_nanos(900),
        );
        sink.record_device_enqueue(
            DeviceEnqueueStage::GraphLaunch,
            Duration::from_nanos(100),
            Some(Duration::from_nanos(40)),
        );
        sink.record_device_enqueue(
            DeviceEnqueueStage::GraphLaunch,
            Duration::from_nanos(300),
            None,
        );
        // An available zero CPU interval is a sample, distinct from unavailable.
        sink.record_device_enqueue(
            DeviceEnqueueStage::GraphLaunch,
            Duration::ZERO,
            Some(Duration::ZERO),
        );
        sink.record_device_enqueue(
            DeviceEnqueueStage::EagerCommand,
            Duration::from_nanos(70),
            None,
        );
        sink.record_device_enqueue(
            DeviceEnqueueStage::DeferredSnapshotReadback,
            Duration::from_nanos(80),
            None,
        );

        for metrics in [&aggregate, &decode] {
            let snapshot = metrics.snapshot();
            let submit = &snapshot["host_encode_submit_breakdown"]
                ["provider_encode_submit_breakdown"]["lane_reserve_submit_arm_breakdown"]
                ["device_runtime_submit_breakdown"];
            let children = &submit["enqueue_commands_breakdown"];
            assert_eq!(children["scope"], "per_call_nested_in_enqueue_commands");
            let graph = &children["graph_launch"];
            assert_eq!(graph["wall"]["samples"], 3);
            assert_eq!(graph["wall"]["total_ns"], 400);
            assert_eq!(graph["thread_cpu"]["samples"], 2);
            assert_eq!(graph["thread_cpu"]["total_ns"], 40);
            assert_eq!(graph["thread_cpu_paired_wall"]["samples"], 2);
            assert_eq!(graph["thread_cpu_paired_wall"]["total_ns"], 100);
            assert_eq!(graph["thread_cpu_unavailable_samples"], 1);
            for (field, wall_ns) in [("eager_command", 70), ("deferred_snapshot_readback", 80)] {
                assert_eq!(children[field]["wall"]["samples"], 1);
                assert_eq!(children[field]["wall"]["total_ns"], wall_ns);
                assert!(children[field]["thread_cpu"].is_null());
                assert!(children[field]["thread_cpu_paired_wall"].is_null());
                assert_eq!(children[field]["thread_cpu_unavailable_samples"], 1);
            }
            // Nested calls neither synthesize a wave nor add to its parent.
            assert_eq!(submit["enqueue_commands"]["samples"], 1);
            assert_eq!(submit["enqueue_commands"]["total_ns"], 900);
            assert_eq!(snapshot["submitted_wave_total"]["samples"], 0);
            metrics.reset();
            assert_eq!(
                metrics.enqueue_commands_breakdown(),
                prefill.enqueue_commands_breakdown()
            );
        }
        let untouched = prefill.enqueue_commands_breakdown();
        for field in [
            "graph_launch",
            "eager_command",
            "deferred_snapshot_readback",
        ] {
            assert_eq!(untouched[field]["wall"]["samples"], 0);
            assert_eq!(untouched[field]["wall"]["total_ns"], 0);
            assert!(untouched[field]["thread_cpu"].is_null());
            assert!(untouched[field]["thread_cpu_paired_wall"].is_null());
            assert_eq!(untouched[field]["thread_cpu_unavailable_samples"], 0);
        }
    }

    #[test]
    fn wave_timing_node_encode_children_are_phase_scoped_and_reset() {
        use super::{SubmissionWaveDispatchStage, SubmissionWaveDispatchTimingSink};

        let aggregate = VNextWaveTimingMetrics::default();
        let decode = VNextWaveTimingMetrics::default();
        let other_phase = VNextWaveTimingMetrics::default();
        let sink = VNextWaveTimingSink {
            aggregate: &aggregate,
            phase: &decode,
        };
        let stages = [
            (
                SubmissionWaveDispatchStage::NodeIdentityMaterialize,
                "identity_materialize",
            ),
            (
                SubmissionWaveDispatchStage::NodeInvocationConstruct,
                "invocation_construct",
            ),
            (
                SubmissionWaveDispatchStage::ProviderDynamicBindingEncode,
                "dynamic_binding_encode",
            ),
            (
                SubmissionWaveDispatchStage::BindingValidateAndCoalesce,
                "binding_validate_coalesce",
            ),
            (
                SubmissionWaveDispatchStage::SegmentFreshAuthorityAndWindows,
                "segment_binding_prepare_encode_breakdown/fresh_authority_and_windows",
            ),
            (
                SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata,
                "segment_binding_prepare_encode_breakdown/backing_permit_and_metadata",
            ),
            (
                SubmissionWaveDispatchStage::SegmentBackingSnapshotLookup,
                "segment_binding_prepare_encode_breakdown/backing_permit_and_metadata_breakdown/snapshot_lookup",
            ),
            (
                SubmissionWaveDispatchStage::SegmentBackingDedupReserveAndPoolResolution,
                "segment_binding_prepare_encode_breakdown/backing_permit_and_metadata_breakdown/dedup_reserve_and_pool_resolution",
            ),
            (
                SubmissionWaveDispatchStage::SegmentBackingLockAcquisition,
                "segment_binding_prepare_encode_breakdown/backing_permit_and_metadata_breakdown/lock_acquisition",
            ),
            (
                SubmissionWaveDispatchStage::SegmentBackingLockedValidation,
                "segment_binding_prepare_encode_breakdown/backing_permit_and_metadata_breakdown/locked_validation",
            ),
            (
                SubmissionWaveDispatchStage::SegmentBackingWindowIntersections,
                "segment_binding_prepare_encode_breakdown/backing_permit_and_metadata_breakdown/window_intersections",
            ),
            (
                SubmissionWaveDispatchStage::SegmentBackingImmutableMetadataValidation,
                "segment_binding_prepare_encode_breakdown/backing_permit_and_metadata_breakdown/immutable_metadata_validation",
            ),
            (
                SubmissionWaveDispatchStage::SegmentNodeDependenciesAndRegions,
                "segment_binding_prepare_encode_breakdown/node_dependencies_and_regions",
            ),
            (
                SubmissionWaveDispatchStage::SegmentBackendEncodeAndValidate,
                "segment_binding_prepare_encode_breakdown/backend_encode_and_validate",
            ),
        ];
        for (index, (stage, _)) in stages.iter().enumerate() {
            sink.record(*stage, Duration::from_nanos(100 + index as u64));
            sink.record(*stage, Duration::from_nanos(200 + index as u64));
        }
        let child = |metrics: &VNextWaveTimingMetrics| {
            metrics.snapshot()["host_encode_submit_breakdown"]["provider_encode_submit_breakdown"]
                ["provider_node_encode_breakdown"]
                .clone()
        };
        for metrics in [&aggregate, &decode] {
            let snapshot = child(metrics);
            assert_eq!(snapshot["collection"], "profile_attached_only");
            for (index, (_, field)) in stages.iter().enumerate() {
                let sample = snapshot.pointer(&format!("/{field}")).unwrap();
                assert_eq!(sample["samples"], 2);
                assert_eq!(sample["total_ns"], 300 + 2 * index as u64);
            }
            // Recording a nested interval does not invent a parent wave sample.
            assert_eq!(metrics.snapshot()["submitted_wave_total"]["samples"], 0);
            metrics.reset();
            for (_, field) in stages {
                let snapshot = child(metrics);
                let sample = snapshot.pointer(&format!("/{field}")).unwrap();
                assert_eq!(sample["samples"], 0);
                assert_eq!(sample["total_ns"], 0);
            }
        }
        for (_, field) in stages {
            assert_eq!(
                child(&other_phase).pointer(&format!("/{field}/samples")),
                Some(&serde_json::json!(0))
            );
        }
    }

    #[test]
    fn wave_timing_backing_partial_observations_keep_independent_parent_samples() {
        use super::{SubmissionWaveDispatchStage, SubmissionWaveDispatchTimingSink};

        let metrics = VNextWaveTimingMetrics::default();
        metrics.record(
            SubmissionWaveDispatchStage::SegmentBackingSnapshotLookup,
            Duration::from_nanos(11),
        );
        metrics.record(
            SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata,
            Duration::from_nanos(100),
        );
        metrics.record(
            SubmissionWaveDispatchStage::SegmentBackingPermitAndMetadata,
            Duration::from_nanos(200),
        );

        let snapshot = metrics.segment_binding_prepare_encode_breakdown();
        assert_eq!(snapshot["backing_permit_and_metadata"]["samples"], 2);
        assert_eq!(snapshot["backing_permit_and_metadata"]["total_ns"], 300);
        let children = &snapshot["backing_permit_and_metadata_breakdown"];
        assert_eq!(children["snapshot_lookup"]["samples"], 1);
        assert_eq!(children["snapshot_lookup"]["total_ns"], 11);
        assert_eq!(children["lock_acquisition"]["samples"], 0);
        assert_eq!(children["immutable_metadata_validation"]["samples"], 0);
    }

    #[test]
    fn wave_timing_reset_clears_resource_prepare_breakdown() {
        let metrics = VNextWaveTimingMetrics::default();
        metrics
            .resource_step_request_prepare
            .record(Duration::from_micros(11));
        metrics
            .resource_step_admission
            .record(Duration::from_micros(13));
        metrics.resource_step_admission_breakdown.record(
            StepResourceAdmissionProfilePhase::DemandEvaluate,
            Duration::from_micros(7),
        );
        metrics
            .resource_submission_wave_prepare
            .record(Duration::from_micros(17));

        let snapshot = metrics.snapshot();
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_request_prepare"]["samples"],
            1
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_admission"]["average_us"],
            13.0
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_admission_breakdown"]["demand_evaluate"]
                ["average_us"],
            7.0
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["submission_wave_prepare"]["average_us"],
            17.0
        );

        metrics.reset();
        let snapshot = metrics.snapshot();
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_request_prepare"]["samples"],
            0
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_admission"]["samples"],
            0
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["step_admission_breakdown"]["demand_evaluate"]
                ["samples"],
            0
        );
        assert_eq!(
            snapshot["resource_prepare_breakdown"]["submission_wave_prepare"]["samples"],
            0
        );
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
    fn physical_span_metrics_group_replay_time_by_executable_fingerprint() {
        let eager = DeviceSubmissionExecutionSpan::measured(
            0,
            1,
            DeviceExecutionSpanKind::EagerCommand,
            vec![
                DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Transfer, 0, 10).unwrap(),
            ],
        )
        .unwrap();
        let replay = DeviceSubmissionExecutionSpan::measured(
            1,
            2,
            DeviceExecutionSpanKind::ReusableExecutable,
            vec![
                DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Compute, 10, 50).unwrap(),
            ],
        )
        .unwrap()
        .with_reusable_executable_fingerprint("a".repeat(64))
        .unwrap();
        let timing = DeviceSubmissionExecutionTiming::from_spans(2, vec![eager, replay]).unwrap();
        let metrics = VNextPhysicalSpanTimingMetrics::default();

        metrics.record(&DeviceTimingMeasurement::Measured(timing));
        let snapshot = metrics.snapshot();

        assert_eq!(snapshot["measured_submissions"], 1);
        assert_eq!(snapshot["eager_commands"]["total_ns"], 10);
        assert_eq!(snapshot["reusable_executables"]["total_ns"], 40);
        assert_eq!(
            snapshot["reusable_by_fingerprint"][0]["reusable_executable_fingerprint"],
            "a".repeat(64)
        );
        assert_eq!(
            snapshot["reusable_by_fingerprint"][0]["timing"]["total_ns"],
            40
        );

        metrics.reset();
        assert_eq!(metrics.snapshot()["measured_submissions"], 0);
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
    fn monotonic_wall_anchor_uses_sample_midpoint_and_bounds_full_capture_span() {
        assert_eq!(bounded_wall_anchor(1_000, 1_100), (1_050, 100));
        assert_eq!(bounded_wall_anchor(1_101, 1_000), (1_050, 101));
        assert_eq!(bounded_wall_anchor(7, 7), (7, 0));
    }

    #[test]
    fn journal_clock_anchor_is_kernel_profile_only() {
        assert!(journal_clock_anchor_required(DeviceTimingMode::Kernel));
        for timing_mode in [
            DeviceTimingMode::Off,
            DeviceTimingMode::Completion,
            DeviceTimingMode::Replay,
            DeviceTimingMode::Verification,
        ] {
            assert!(!journal_clock_anchor_required(timing_mode));
        }
    }

    #[test]
    fn completion_readback_policy_declares_only_canonical_greedy_scalar_batches() {
        use super::{ElementType, VNextTerminalReadbacks};
        use ferrum_runtime::{
            CompletionReadbackBatchRequest, CompletionReadbackCollectionRequest,
            CompletionReadbackRequest, CompletionReadbackSelectionRequest, HostTransferLayout,
            NodeId, ResourceId,
        };
        use ferrum_types::CompletionReadbackPolicy::{BatchedU32, Synchronous};
        let batch = |count: u32, element, elements| {
            CompletionReadbackBatchRequest::new(
                (0..count)
                    .map(|participant| {
                        CompletionReadbackRequest::new(
                            NodeId::new("node.token").unwrap(),
                            participant,
                            ResourceId::new("output.token").unwrap(),
                            u64::from(participant) * 16,
                            HostTransferLayout::new(element, elements).unwrap(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap()
        };
        for count in [1, 8, 32] {
            let expected = batch(count, ElementType::U32, 1);
            let readbacks = VNextTerminalReadbacks::Batch(expected.clone());
            assert_eq!(
                readbacks.terminal_packet_request(
                    BatchedU32,
                    VNextProductOutputMode::GreedyToken,
                    false,
                ),
                Some(&expected)
            );
            assert!(readbacks
                .terminal_packet_request(Synchronous, VNextProductOutputMode::GreedyToken, false,)
                .is_none());
            assert!(readbacks
                .terminal_packet_request(BatchedU32, VNextProductOutputMode::FullLogits, false,)
                .is_none());
            assert!(readbacks
                .terminal_packet_request(BatchedU32, VNextProductOutputMode::GreedyToken, true,)
                .is_none());
        }
        for request in [
            batch(33, ElementType::U32, 1),
            batch(2, ElementType::U32, 2),
            batch(2, ElementType::F32, 1),
        ] {
            assert!(VNextTerminalReadbacks::Batch(request)
                .terminal_packet_request(BatchedU32, VNextProductOutputMode::GreedyToken, false,)
                .is_none());
        }
        let collection =
            CompletionReadbackCollectionRequest::new(vec![batch(2, ElementType::U32, 1)]).unwrap();
        assert!(VNextTerminalReadbacks::Collection(collection)
            .terminal_packet_request(BatchedU32, VNextProductOutputMode::GreedyToken, false,)
            .is_none());
        let selection = CompletionReadbackSelectionRequest::new(
            vec![batch(2, ElementType::F16, 4), batch(2, ElementType::U32, 1)],
            vec![0, 1],
        )
        .unwrap();
        assert!(VNextTerminalReadbacks::Selection(selection)
            .terminal_packet_request(BatchedU32, VNextProductOutputMode::GreedyToken, false)
            .is_none());
    }

    #[test]
    fn product_output_mode_requires_a_uniform_exact_greedy_decode_wave() {
        let greedy = VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: None,
        });
        let full = VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits);
        let repetition = VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(GreedyRepetitionPenalty::new(1.1, vec![7, 11])),
        });

        assert_eq!(
            product_output_mode_for_roles(VNextExecutionWaveKind::Decode, [&greedy, &greedy],),
            VNextProductOutputMode::GreedyToken
        );
        assert_eq!(
            product_output_mode_for_roles(VNextExecutionWaveKind::Decode, [&greedy, &repetition],),
            VNextProductOutputMode::GreedyToken
        );
        assert_eq!(
            product_output_mode_for_roles(VNextExecutionWaveKind::Decode, [&greedy, &full],),
            VNextProductOutputMode::FullLogits
        );
        assert_eq!(
            product_output_mode_for_roles(VNextExecutionWaveKind::Prefill, [&greedy]),
            VNextProductOutputMode::FullLogits
        );
        assert_eq!(
            product_output_mode_for_roles(VNextExecutionWaveKind::Decode, std::iter::empty()),
            VNextProductOutputMode::FullLogits
        );
    }

    #[test]
    fn teacher_forcing_masks_only_the_engine_facing_logits_copy() {
        let mut outputs = vec![ExecutorSamplingOutput::full_logits(vec![1.0, 2.0, 3.0]).unwrap()];
        apply_teacher_forced_decision(
            &mut outputs,
            VNextTeacherForcedDecision::new(7, TokenId::new(1)),
        )
        .unwrap();

        assert_eq!(
            outputs,
            [ExecutorSamplingOutput::FullLogits(vec![
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
            ])]
        );
        assert!(apply_teacher_forced_decision(
            &mut [ExecutorSamplingOutput::greedy_token(TokenId::new(1))],
            VNextTeacherForcedDecision::new(0, TokenId::new(1)),
        )
        .is_err());
        assert!(apply_teacher_forced_decision(
            &mut [ExecutorSamplingOutput::full_logits(vec![0.0]).unwrap()],
            VNextTeacherForcedDecision::new(0, TokenId::new(2)),
        )
        .is_err());
    }

    #[test]
    fn product_repetition_input_is_typed_and_neutral_outside_greedy_decode() {
        let policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(GreedyRepetitionPenalty::new(1.25, vec![3, 9])),
        };
        let active = product_repetition_input(Some(&policy), VNextProductOutputMode::GreedyToken);
        assert_eq!(active.token_ids, [3, 9]);
        assert_eq!(active.penalty, 1.25);
        assert!(active.is_active());

        let neutral = product_repetition_input(Some(&policy), VNextProductOutputMode::FullLogits);
        assert!(neutral.token_ids.is_empty());
        assert_eq!(neutral.penalty, 1.0);
        assert!(!neutral.is_active());
    }

    #[test]
    fn repetition_token_ids_are_zero_padded_to_the_typed_participant_capacity() {
        let bytes = padded_repetition_token_id_bytes(&[3, 9], 4).unwrap();
        assert_eq!(
            bytes,
            [
                3_u32.to_le_bytes(),
                9_u32.to_le_bytes(),
                0_u32.to_le_bytes(),
                0_u32.to_le_bytes(),
            ]
            .concat()
        );
        assert!(padded_repetition_token_id_bytes(&[3, 9], 1).is_err());
    }

    #[test]
    fn product_token_mask_preserves_short_mask_semantics_without_hidden_defaults() {
        let policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(TokenSelectionMask::new(vec![1, -7, 0])),
            repetition_penalty: None,
        };
        assert_eq!(
            normalized_product_token_mask(Some(&policy), VNextProductOutputMode::GreedyToken, 5,),
            [1, 1, 0, 0, 0]
        );
        assert_eq!(
            normalized_product_token_mask(Some(&policy), VNextProductOutputMode::FullLogits, 5,),
            [1, 1, 1, 1, 1]
        );
        assert_eq!(
            normalized_product_token_mask(None, VNextProductOutputMode::GreedyToken, 3),
            [1, 1, 1]
        );
    }

    fn test_token_mask_target(
        slot_id: u64,
        participant_index: usize,
    ) -> VNextProductTokenMaskSlotTarget {
        VNextProductTokenMaskSlotTarget {
            identity: VNextProductTokenMaskSlotIdentity::Test(slot_id),
            participant_index,
        }
    }

    fn test_selection_content(mask: &TokenSelectionMask) -> VNextProductTokenMaskContent {
        let policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(mask.clone()),
            repetition_penalty: None,
        };
        VNextProductTokenMaskContent::from_policy(
            Some(&policy),
            VNextProductOutputMode::GreedyToken,
            5,
        )
    }

    #[test]
    fn product_token_mask_residency_requires_exact_slot_participant_and_content() {
        let first_mask = TokenSelectionMask::new(vec![1, 0, 1]);
        let first_content = test_selection_content(&first_mask);
        let target = test_token_mask_target(7, 0);
        let mut residency = VNextProductTokenMaskResidency::default();

        let first = residency.prepare(Some(target.clone()), first_content.clone());
        assert!(first.upload_required);
        residency.publish(std::slice::from_ref(&first));

        assert!(
            !residency
                .prepare(Some(target.clone()), first_content.clone())
                .upload_required
        );
        assert!(
            residency
                .prepare(Some(test_token_mask_target(7, 1)), first_content.clone())
                .upload_required
        );
        assert!(
            residency
                .prepare(Some(test_token_mask_target(8, 0)), first_content.clone())
                .upload_required
        );

        // Distinct Arcs with byte-identical masks may safely share residency.
        let identical = TokenSelectionMask::new(vec![1, 0, 1]);
        assert!(
            !residency
                .prepare(Some(target.clone()), test_selection_content(&identical))
                .upload_required
        );

        // The public 64-bit fingerprint is a fast filter, not correctness
        // authority: an exact byte mismatch must still upload.
        let mut forged_collision = TokenSelectionMask::new(vec![0, 1, 0]);
        forged_collision.fingerprint = first_mask.fingerprint;
        assert!(
            residency
                .prepare(Some(target), test_selection_content(&forged_collision))
                .upload_required
        );
    }

    #[test]
    fn product_token_mask_residency_overwrite_and_failed_transactions_are_fail_closed() {
        let residency = parking_lot::Mutex::new(VNextProductTokenMaskResidency::default());
        let first = test_selection_content(&TokenSelectionMask::new(vec![1, 0, 1]));
        let changed = test_selection_content(&TokenSelectionMask::new(vec![1, 1, 0]));

        {
            let mut transaction = VNextProductTokenMaskResidencyTransaction::prepare_for_test(
                &residency,
                Some(7),
                [first.clone()],
            );
            assert!(transaction.plans()[0].upload_required);
            transaction.publish();
            transaction.settle_success();
        }
        {
            let mut transaction = VNextProductTokenMaskResidencyTransaction::prepare_for_test(
                &residency,
                Some(7),
                [changed.clone()],
            );
            assert!(transaction.plans()[0].upload_required);
            // Definitely-not-submitted or any earlier failure cannot publish.
            transaction.invalidate_before_slot_release();
            assert!(residency.lock().entries.is_empty());
            drop(transaction);
        }
        {
            let transaction = VNextProductTokenMaskResidencyTransaction::prepare_for_test(
                &residency,
                Some(7),
                [first.clone()],
            );
            assert!(transaction.plans()[0].upload_required);
        }
        {
            let mut transaction = VNextProductTokenMaskResidencyTransaction::prepare_for_test(
                &residency,
                Some(7),
                [first.clone()],
            );
            transaction.publish();
            // An interruption between publication and typed settlement must
            // still fail closed. Production settles immediately after terminal
            // device success.
        }
        let transaction = VNextProductTokenMaskResidencyTransaction::prepare_for_test(
            &residency,
            Some(7),
            [first],
        );
        assert!(transaction.plans()[0].upload_required);
    }

    #[test]
    fn product_token_mask_residency_tracks_participant_reordering_by_physical_range() {
        let mask_a = TokenSelectionMask::new(vec![1, 0, 1]);
        let mask_b = TokenSelectionMask::new(vec![0, 1, 1]);
        let content_a = test_selection_content(&mask_a);
        let content_b = test_selection_content(&mask_b);
        let target_zero = test_token_mask_target(7, 0);
        let target_one = test_token_mask_target(7, 1);
        let mut residency = VNextProductTokenMaskResidency::default();

        let original = [
            residency.prepare(Some(target_zero.clone()), content_a.clone()),
            residency.prepare(Some(target_one.clone()), content_b.clone()),
        ];
        assert!(original.iter().all(|plan| plan.upload_required));
        residency.publish(&original);

        let reordered = [
            residency.prepare(Some(target_zero.clone()), content_b.clone()),
            residency.prepare(Some(target_one.clone()), content_a.clone()),
        ];
        assert!(reordered.iter().all(|plan| plan.upload_required));
        residency.publish(&reordered);

        assert!(
            !residency
                .prepare(Some(target_zero), content_b)
                .upload_required
        );
        assert!(
            !residency
                .prepare(Some(target_one), content_a)
                .upload_required
        );
    }

    #[test]
    fn product_token_mask_residency_rejects_copy_on_write_mutation() {
        let mut mask = TokenSelectionMask::new(vec![1, 0, 1]);
        let target = test_token_mask_target(7, 0);
        let mut residency = VNextProductTokenMaskResidency::default();
        let original = residency.prepare(Some(target.clone()), test_selection_content(&mask));
        residency.publish(std::slice::from_ref(&original));

        assert!(mask.set_tokens_validity(&[1], true));
        assert!(
            residency
                .prepare(Some(target), test_selection_content(&mask))
                .upload_required,
            "copy-on-write mutation must not inherit the prior device residency proof"
        );
    }

    #[test]
    fn product_token_mask_residency_without_stable_slot_always_uploads() {
        let residency = parking_lot::Mutex::new(VNextProductTokenMaskResidency::default());
        let content = VNextProductTokenMaskContent::AllValid { vocabulary_size: 5 };
        for _ in 0..2 {
            let mut transaction = VNextProductTokenMaskResidencyTransaction::prepare_for_test(
                &residency,
                None,
                [content.clone()],
            );
            assert!(transaction.plans()[0].upload_required);
            transaction.publish();
            transaction.settle_success();
        }
        assert!(residency.lock().entries.is_empty());
    }

    #[test]
    fn product_token_mask_residency_is_entry_bounded_and_does_not_retain_masks() {
        let mut residency = VNextProductTokenMaskResidency::default();
        for slot_id in 0..=MAX_PRODUCT_TOKEN_MASK_SLOT_CACHE_ENTRIES as u64 {
            let plan = residency.prepare(
                Some(test_token_mask_target(slot_id, 0)),
                VNextProductTokenMaskContent::AllValid { vocabulary_size: 5 },
            );
            residency.publish(std::slice::from_ref(&plan));
            assert!(residency.entries.len() <= MAX_PRODUCT_TOKEN_MASK_SLOT_CACHE_ENTRIES);
        }

        residency.clear();
        let target = test_token_mask_target(17, 0);
        {
            let content = test_selection_content(&TokenSelectionMask::new(vec![1, 0, 1]));
            let plan = residency.prepare(Some(target.clone()), content);
            residency.publish(std::slice::from_ref(&plan));
        }
        assert!(
            residency
                .prepare(
                    Some(target),
                    test_selection_content(&TokenSelectionMask::new(vec![1, 0, 1])),
                )
                .upload_required,
            "a dead request mask must not be retained solely by the residency ledger"
        );
    }

    #[test]
    fn selected_token_readback_rejects_out_of_vocabulary_values() {
        assert_eq!(
            decode_selected_token(&3_u32.to_le_bytes(), 8).unwrap(),
            TokenId::new(3)
        );
        assert!(decode_selected_token(&8_u32.to_le_bytes(), 8).is_err());
        assert!(decode_selected_token(&u32::MAX.to_le_bytes(), 8).is_err());
        assert!(decode_selected_token(&[0, 1, 2], 8).is_err());
    }

    #[test]
    fn recompute_completion_preserves_product_usage_and_replay_baseline() {
        let request_id = RequestId::new();
        let valid = ExecutorSequenceCompletion::new(request_id.clone(), "cache-valid".into(), 2, 3)
            .unwrap();
        validate_sequence_completion_accounting(&request_id, 2, 1, &valid).unwrap();

        let wrong_prompt =
            ExecutorSequenceCompletion::new(request_id.clone(), "cache-prompt".into(), 3, 3)
                .unwrap();
        assert!(validate_sequence_completion_accounting(&request_id, 2, 1, &wrong_prompt).is_err());

        let before_replay =
            ExecutorSequenceCompletion::new(request_id.clone(), "cache-output".into(), 2, 0)
                .unwrap();
        assert!(
            validate_sequence_completion_accounting(&request_id, 2, 1, &before_replay).is_err()
        );

        let other_request =
            ExecutorSequenceCompletion::new(RequestId::new(), "cache-other".into(), 2, 3).unwrap();
        assert!(
            validate_sequence_completion_accounting(&request_id, 2, 1, &other_request).is_err()
        );
    }
}
