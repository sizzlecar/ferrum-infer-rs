//! Product-owned composition of one immutable vNext model plan.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::collections::BTreeSet;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    CapabilityCatalog, ContractVersion, DeviceRuntime, EngineSelection, ExecutablePlanView,
    ModelArtifactSourceRole, ModelConfigFingerprint, NumericalProfileRejection,
    NumericalProfileRejectionStage, NumericalProfileResolution, OperationRuntimeRegistry,
    ProgramPlanCompilation, RationalValue, ResolutionArtifactId, ResolutionDecisionBinding,
    ResolutionDecisionSource, ResolutionField, ResolutionFingerprint, ResolutionReasonId,
    ResolutionSourceEvidence, ResolutionSourceProvenance, ResolvedModelPlan,
    ResolvedModelPlanInputs, ResolvedPlanValidationContext, ResolvedRuntimePolicy, SamplingPolicy,
    SpecialTokenRole, StopPolicy, StopTokenCollisionPolicy, StructuredOutputPolicy,
    TokenizerDescriptor, TokenizerId, TriStatePolicy, WeightMaterializerRegistry,
    WeightMaterializerSelection, JSON_RESOLUTION_SOURCE_PARSER,
};
use ferrum_models::vnext::{
    DefinedProductionModel, PreparedProductionModel, ProductionModelFamilyRegistry,
};
use ferrum_models::{
    VNextCompiledModel, VNextExecutorConfig, VNextModelCompilationError, VNextModelExecutor,
    VNextRuntimeComposition,
};
use ferrum_types::{EngineConfig, FerrumError, ModelInfo, ResponseFormat, Result, SamplingParams};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const PRODUCT_COMPOSITION_VERSION: ContractVersion = ContractVersion::new(2, 0);
const PRODUCT_COMPOSITION_PRODUCER: &str = "ferrum.product-composition";
const DEFAULT_SAMPLER_SEED: u64 = 42;

/// The current addressed native CUDA ABI consumes F16 pages. Resolve Auto
/// within the requested storage format instead of letting an incompatible
/// native provider silently change the user's numerical request.
pub(crate) fn cuda_attention_policy_for_kv(
    requested: ferrum_types::AttentionExecutionPolicy,
    dtype: ferrum_types::KvCacheDtype,
) -> Result<ferrum_types::AttentionExecutionPolicy> {
    use ferrum_types::{AttentionExecutionPolicy, KvStorageFormat};
    match (KvStorageFormat::try_from(dtype).map_err(FerrumError::config)?, requested) {
        (KvStorageFormat::Int8PerTokenHeadF32ScaleV1, AttentionExecutionPolicy::NativeAdaptive) => {
            Err(FerrumError::unsupported(
                "native-adaptive CUDA attention does not implement INT8 KV storage; use portable/auto attention or --kv-dtype fp16",
            ))
        }
        (KvStorageFormat::Int8PerTokenHeadF32ScaleV1, _) => Ok(AttentionExecutionPolicy::Portable),
        (KvStorageFormat::F16, requested) => Ok(requested),
    }
}

pub(crate) fn create_vnext_executor<R: DeviceRuntime>(
    engine: &EngineConfig,
    defined: &DefinedProductionModel,
    runtime: Arc<R>,
    operation_registry: OperationRuntimeRegistry<R>,
    weight_materializers: WeightMaterializerRegistry,
    catalog: CapabilityCatalog,
    select_materializer: impl Fn(
        &ferrum_interfaces::vnext::PreparedModelFamily,
    ) -> std::result::Result<
        WeightMaterializerSelection,
        ferrum_interfaces::vnext::VNextError,
    >,
    q8_predecoded_composition: Option<&mut dyn FnMut() -> Result<VNextRuntimeComposition<R>>>,
) -> Result<VNextModelExecutor<R>> {
    create_vnext_executor_with_configuration(
        engine,
        defined,
        runtime,
        operation_registry,
        weight_materializers,
        catalog,
        select_materializer,
        q8_predecoded_composition,
        |engine, info, runtime| VNextExecutorConfig::from_engine_config(engine, info, runtime),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_vnext_executor_with_configuration<R: DeviceRuntime>(
    engine: &EngineConfig,
    defined: &DefinedProductionModel,
    runtime: Arc<R>,
    operation_registry: OperationRuntimeRegistry<R>,
    weight_materializers: WeightMaterializerRegistry,
    catalog: CapabilityCatalog,
    select_materializer: impl Fn(
        &ferrum_interfaces::vnext::PreparedModelFamily,
    ) -> std::result::Result<
        WeightMaterializerSelection,
        ferrum_interfaces::vnext::VNextError,
    >,
    mut q8_predecoded_composition: Option<&mut dyn FnMut() -> Result<VNextRuntimeComposition<R>>>,
    executor_config: impl Fn(&EngineConfig, &ModelInfo, &R) -> Result<VNextExecutorConfig>,
) -> Result<VNextModelExecutor<R>> {
    let requested_kv_storage = ferrum_types::KvStorageFormat::try_from(engine.kv_cache.dtype)
        .map_err(FerrumError::config)?;
    let composition =
        VNextRuntimeComposition::new(runtime, operation_registry, weight_materializers, catalog);
    let candidates = defined
        .definition()
        .numerical_profiles()
        .candidates(&engine.numerical_execution, requested_kv_storage)
        .map_err(|error| FerrumError::config(error.to_string()))?;
    let mut rejected = Vec::new();
    for profile in candidates {
        if let Some(rejection) = auto_numerical_profile_rejection(
            &engine.numerical_execution,
            profile,
            &engine.backend.device,
            composition.runtime().descriptor().ordinal,
            &composition.runtime().descriptor().capabilities,
            probe_auto_cuda_attributes,
        ) {
            rejected.push(rejection);
            continue;
        }
        if engine.runtime.q8_projection_weight_preparation
            == ferrum_types::Q8ProjectionWeightPreparation::PredecodedF32
            && !profile.operations.iter().any(|operation| {
                operation
                    .composite_arithmetic
                    .as_ref()
                    .is_some_and(composite_supports_q8_preparation)
            })
        {
            return Err(FerrumError::unsupported(
                "predecoded-f32 Q8 projection weights require a selected supported upstream causal or GDN numerical route",
            ));
        }
        if engine.runtime.q4_mmvq_weight_decode == ferrum_types::Q4MmvqWeightDecode::SharedM8
            && !profile_has_shared_m8_q4_route(profile)
        {
            return Err(FerrumError::unsupported(
                "shared-m8 Q4 weight decode requires a selected causal or GDN numerical profile declaring Q4_K Columns M8 MarkerV2 MMVQ; select an appropriate explicit numerical profile",
            ));
        }
        if engine.runtime.q5_mmvq_weight_decode == ferrum_types::Q5MmvqWeightDecode::SharedM8
            && !profile_has_shared_m8_q5_route(profile)
        {
            return Err(FerrumError::unsupported(
                "shared-m8 Q5 weight decode requires a selected causal or GDN numerical profile declaring Q5_K Columns M8 MarkerV2 MMVQ; select an appropriate explicit numerical profile",
            ));
        }
        if engine.runtime.q4_mmvq_weight_decode == ferrum_types::Q4MmvqWeightDecode::SharedM4M8
            && !profile_has_shared_m4_m8_route(
                profile,
                ferrum_interfaces::vnext::ProjectionBlockFormat::Q4K,
            )
        {
            return Err(FerrumError::unsupported(
                "shared-m4-m8 Q4 weight decode requires a selected causal, GDN or SwiGLU numerical profile declaring Q4_K Columns M4 or M8 MarkerV2 MMVQ",
            ));
        }
        if engine.runtime.q5_mmvq_weight_decode == ferrum_types::Q5MmvqWeightDecode::SharedM4M8
            && !profile_has_shared_m4_m8_route(
                profile,
                ferrum_interfaces::vnext::ProjectionBlockFormat::Q5K,
            )
        {
            return Err(FerrumError::unsupported(
                "shared-m4-m8 Q5 weight decode requires a selected causal, GDN or SwiGLU numerical profile declaring Q5_K Columns M4 or M8 MarkerV2 MMVQ",
            ));
        }
        let prepared = match defined.prepare(&profile.id) {
            Ok(prepared) => prepared,
            Err(error) => {
                rejected.push(NumericalProfileRejection {
                    profile_id: profile.id.clone(),
                    stage: NumericalProfileRejectionStage::FamilyPreparation,
                    reason: error.to_string(),
                });
                continue;
            }
        };
        let model_info =
            prepared.model_info(engine.model.model_id.clone(), engine.backend.device.clone());
        let mut selected_engine = engine.clone();
        if selected_engine.runtime.q8_projection_weight_preparation
            == ferrum_types::Q8ProjectionWeightPreparation::Auto
        {
            selected_engine.runtime.q8_projection_weight_preparation =
                ferrum_types::Q8ProjectionWeightPreparation::Compressed;
        }
        if let Some(request) = &engine.runtime.startup_memory_request {
            selected_engine.memory.usable_capacity_bytes = Some(
                usize::try_from(request.usable_capacity_bytes)
                    .map_err(|_| FerrumError::config("startup memory budget exceeds usize"))?,
            );
            if !request.batch_is_explicit {
                selected_engine.batching.max_num_batched_tokens = selected_engine
                    .batching
                    .max_num_batched_tokens
                    .max(selected_engine.scheduler.max_running_requests);
            }
        }
        let config = executor_config(&selected_engine, &model_info, composition.runtime())?;
        let compiled_context_tokens = config.maximum_model_tokens;
        let materializer = match select_materializer(prepared.family()) {
            Ok(selection) => selection,
            Err(error) => {
                rejected.push(NumericalProfileRejection {
                    profile_id: profile.id.clone(),
                    stage: NumericalProfileRejectionStage::WeightMaterializer,
                    reason: error.to_string(),
                });
                continue;
            }
        };
        let compiled = match composition.compile_model(
            &prepared,
            model_info.clone(),
            &selected_engine,
            config,
            materializer.clone(),
        ) {
            Ok(compiled) => compiled,
            Err(error) => {
                rejected.push(NumericalProfileRejection {
                    profile_id: profile.id.clone(),
                    stage: NumericalProfileRejectionStage::ProgramCompilation,
                    reason: error.to_string(),
                });
                continue;
            }
        };
        // Fit the already selected providers and materialized weight layout.
        // No upload or allocation has occurred. Compile the selected bounds
        // and budget, then initialize that exact retained final plan.
        let compiled = if let Some(request) = &engine.runtime.startup_memory_request {
            let context_tokens = selected_engine
                .runtime
                .max_model_len
                .unwrap_or(model_info.max_sequence_length)
                .min(model_info.max_sequence_length)
                .min(selected_engine.runtime.kv_capacity.unwrap_or(usize::MAX));
            let limits = ferrum_types::StartupResourceLimits {
                context_tokens,
                max_sequences: selected_engine.scheduler.max_running_requests,
                max_batch_tokens: selected_engine.batching.max_num_batched_tokens,
            };
            let mut fitted = ferrum_types::fit_startup_resources(request, limits, |workload| {
                compiled
                    .startup_peak_bytes(workload)
                    .map_err(|error| error.to_string())
            })
            .map_err(|error| FerrumError::config(format!("automatic startup capacity: {error}")))?;
            fitted
                .apply_to_engine_config(&mut selected_engine)
                .map_err(FerrumError::config)?;
            let model_context_tokens = model_info.max_sequence_length;
            let mut final_compiled = if fitted.selected == limits
                && limits.context_tokens == compiled_context_tokens
            {
                compiled
            } else {
                drop(compiled);
                let config = executor_config(&selected_engine, &model_info, composition.runtime())?;
                composition.compile_model(
                    &prepared,
                    model_info.clone(),
                    &selected_engine,
                    config,
                    materializer.clone(),
                )?
            };
            // Compilation may select different provider layouts for the final
            // bounds. Verify those actual layouts before authorizing upload.
            let exact_request = ferrum_types::StartupMemoryRequest {
                context_is_explicit: true,
                sequences_is_explicit: true,
                batch_is_explicit: true,
                ..request.clone()
            };
            let verified =
                ferrum_types::fit_startup_resources(&exact_request, fitted.selected, |workload| {
                    final_compiled
                        .startup_peak_bytes(workload)
                        .map_err(|error| error.to_string())
                })
                .map_err(|error| {
                    FerrumError::config(format!("final compiled startup capacity: {error}"))
                })?;
            fitted.context_peak_bytes = verified.context_peak_bytes;
            fitted.decode_peak_bytes = verified.decode_peak_bytes;
            fitted
                .apply_to_engine_config(&mut selected_engine)
                .map_err(FerrumError::config)?;
            tracing::info!(
                model_context_tokens,
                effective_context_tokens = fitted.selected.context_tokens,
                max_sequences = fitted.selected.max_sequences,
                max_batch_tokens = fitted.selected.max_batch_tokens,
                usable_capacity_bytes = fitted.request.usable_capacity_bytes,
                context_peak_bytes = fitted.context_peak_bytes,
                decode_peak_bytes = fitted.decode_peak_bytes,
                reasons = ?fitted.reasons,
                "Adapted runtime limits to the compiled model and available device memory"
            );
            final_compiled.set_startup_memory_plan(fitted)?;
            final_compiled
        } else {
            compiled
        };
        let initialize = |compiled: VNextCompiledModel<'_, R>, selected_engine: &EngineConfig| {
            tracing::info!(requested_numerical_policy = ?engine.numerical_execution,
            requested_kv_dtype = engine.kv_cache.dtype.as_str(),
            kv_storage = ?requested_kv_storage,
            numerical_profile = %profile.id, rejected_numerical_profiles = ?rejected,
            family_id = %prepared.family().family_id(),
            q8_preparation_requested = engine.runtime.q8_projection_weight_preparation.as_runtime_value(),
            q8_preparation_selected = selected_engine.runtime.q8_projection_weight_preparation.as_runtime_value(),
            "Selected numerical profile using the complete static program");
            compiled.initialize(|prepared, runtime, catalog, compilation| {
                let numerical_execution = NumericalProfileResolution::from_static_plan(
                    selected_engine.numerical_execution.clone(),
                    requested_kv_storage,
                    defined.definition(),
                    prepared.family(),
                    catalog,
                    runtime,
                    compilation.executable().execution_plan(),
                    rejected,
                )
                .map_err(|error| FerrumError::model(error.to_string()))?;
                resolve_model_plan(
                    selected_engine,
                    prepared,
                    catalog,
                    runtime,
                    compilation,
                    &numerical_execution,
                )
            })
        };
        if q8_predecoded_composition.is_some()
            && q8_auto_preparation_eligible(
                engine,
                profile,
                composition.runtime().descriptor().ordinal,
                probe_auto_cuda_attributes,
            )?
        {
            // The compressed plan has already fixed context, concurrency and
            // batch capacity. The optional layout cannot buy speed by lowering
            // any of those limits. It owns a separate immutable provider cache.
            let optional_composition = q8_predecoded_composition.as_mut().unwrap()()?;
            let mut optional_engine = selected_engine.clone();
            optional_engine.runtime.q8_projection_weight_preparation =
                ferrum_types::Q8ProjectionWeightPreparation::PredecodedF32;
            let config = executor_config(
                &optional_engine,
                &model_info,
                optional_composition.runtime(),
            )?;
            match optional_composition.compile_model(
                &prepared,
                model_info.clone(),
                &optional_engine,
                config,
                materializer.clone(),
            ) {
                Ok(mut optional) => {
                    let baseline_static = compiled
                        .compilation()
                        .executable()
                        .execution_plan()
                        .payload()
                        .memory()
                        .static_bytes();
                    let optional_static = optional
                        .compilation()
                        .executable()
                        .execution_plan()
                        .payload()
                        .memory()
                        .static_bytes();
                    // The existing provider checks physical Q8_0 leaf ABI,
                    // transforms, staging, K and N. No added destination means
                    // no eligible work; retain the original compressed plan.
                    if optional_static > baseline_static {
                        let baseline = selected_engine
                            .runtime
                            .startup_memory_plan
                            .as_ref()
                            .ok_or_else(|| {
                                FerrumError::internal(
                                    "Q8 Auto lost its fitted compressed memory plan",
                                )
                            })?;
                        let verified = verify_optional_q8_capacity(
                            baseline,
                            |workload| optional.startup_peak_bytes(workload),
                            || ferrum_kernels::backend::probe_device_memory(&engine.backend.device),
                        )?;
                        if let Some(verified) = verified {
                            verified
                                .apply_to_engine_config(&mut optional_engine)
                                .map_err(FerrumError::config)?;
                            optional.set_startup_memory_plan(verified)?;
                            // No static reserve, commit, upload or KV allocation
                            // has occurred. Release the losing runtime/modules
                            // before initializing the selected plan exactly once.
                            drop(compiled);
                            drop(composition);
                            return initialize(optional, &optional_engine);
                        }
                    }
                    drop(optional);
                }
                Err(VNextModelCompilationError::Capacity {
                    required_bytes,
                    usable_capacity_bytes,
                }) => {
                    tracing::info!(required_bytes, usable_capacity_bytes, "Q8 Auto retained compressed storage: optional static layout exceeds capacity");
                }
                Err(error) => return Err(error.into()),
            }
            // Drop the failed probe's provider caches, modules and runtime
            // before reserving any resources for the compressed plan.
            drop(optional_composition);
        }
        return initialize(compiled, &selected_engine);
    }
    Err(FerrumError::unsupported(format!(
        "no declared numerical profile satisfies {:?} with KV dtype {}: {}",
        engine.numerical_execution,
        engine.kv_cache.dtype.as_str(),
        rejected
            .iter()
            .map(|rejection| format!(
                "{} ({:?}): {}",
                rejection.profile_id, rejection.stage, rejection.reason
            ))
            .collect::<Vec<_>>()
            .join("; ")
    )))
}

fn q8_auto_preparation_eligible(
    engine: &EngineConfig,
    profile: &ferrum_interfaces::vnext::NumericalExecutionProfile,
    runtime_ordinal: u32,
    probe: impl FnOnce(
        usize,
    )
        -> std::result::Result<ferrum_kernels::backend::CudaDeviceAttributes, String>,
) -> Result<bool> {
    if engine.runtime.q8_projection_weight_preparation
        != ferrum_types::Q8ProjectionWeightPreparation::Auto
        || engine.runtime.startup_memory_request.is_none()
        || !profile.operations.iter().any(|operation| {
            operation
                .composite_arithmetic
                .as_ref()
                .is_some_and(composite_supports_q8_preparation)
        })
    {
        return Ok(false);
    }
    let ferrum_types::Device::CUDA(ordinal) = &engine.backend.device else {
        return Ok(false);
    };
    if u32::try_from(*ordinal).ok() != Some(runtime_ordinal) {
        return Err(FerrumError::device(
            "Q8 Auto requested device differs from the actual runtime ordinal",
        ));
    }
    let attributes = probe(*ordinal).map_err(FerrumError::device)?;
    Ok(attributes.compute_capability_major == 12
        && attributes.compute_capability_minor == 0
        && attributes.multiprocessor_count > 0)
}

/// Verify an optional layout at the already fitted limits. Only concrete
/// capacity comparisons may decline it; estimation/probe errors propagate.
fn verify_optional_q8_capacity(
    baseline: &ferrum_types::StartupMemoryPlan,
    mut peak: impl FnMut(ferrum_types::StartupWorkload) -> Result<u64>,
    probe: impl FnOnce() -> Result<ferrum_types::DeviceMemorySnapshot>,
) -> Result<Option<ferrum_types::StartupMemoryPlan>> {
    let limits = baseline.selected;
    let context_peak = peak(ferrum_types::StartupWorkload::Prefill {
        context_tokens: limits.context_tokens,
        chunk_tokens: limits.max_batch_tokens.min(limits.context_tokens),
    })?;
    let decode_peak = peak(ferrum_types::StartupWorkload::Decode {
        context_tokens: limits.context_tokens,
        active_sequences: limits.max_sequences,
    })?;
    let budget = baseline.request.usable_capacity_bytes;
    if context_peak.max(decode_peak) > budget {
        tracing::info!(
            context_peak,
            decode_peak,
            budget,
            "Q8 Auto retained compressed storage: optional layout exceeds fixed startup capacity"
        );
        return Ok(None);
    }
    // Both runtimes are still alive here, so the observed free bytes already
    // account for the trial's module/driver footprint. Keep the original budget
    // fully backed rather than silently shrinking it to fit the trial.
    let observed = probe()?;
    if observed.capacity_bytes != baseline.request.device.capacity_bytes
        || observed.available_bytes > observed.capacity_bytes
    {
        return Err(FerrumError::device(
            "Q8 Auto memory observation differs from the selected device capacity",
        ));
    }
    if observed.available_bytes < budget {
        tracing::info!(available_bytes = observed.available_bytes, budget, "Q8 Auto retained compressed storage: driver/module footprint leaves insufficient budget headroom");
        return Ok(None);
    }
    let mut verified = baseline.clone();
    verified.context_peak_bytes = context_peak;
    verified.decode_peak_bytes = decode_peak;
    Ok(Some(verified))
}

fn probe_auto_cuda_attributes(
    ordinal: usize,
) -> std::result::Result<ferrum_kernels::backend::CudaDeviceAttributes, String> {
    #[cfg(feature = "cuda")]
    {
        ferrum_kernels::backend::probe_cuda_device_attributes(ordinal)
            .map_err(|error| error.to_string())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        Err("CUDA device attribute discovery is unavailable in this build".into())
    }
}

/// Default qualification is narrower than explicit operator availability.
/// Keep the runtime catalog intact: compiled native artifacts still require
/// their complete live provider bindings, including nonselected providers.
fn auto_numerical_profile_rejection(
    requested: &ferrum_types::NumericalExecutionPolicy,
    profile: &ferrum_interfaces::vnext::NumericalExecutionProfile,
    device: &ferrum_types::Device,
    runtime_ordinal: u32,
    capabilities: &BTreeSet<ferrum_interfaces::vnext::CapabilityId>,
    probe: impl FnOnce(
        usize,
    )
        -> std::result::Result<ferrum_kernels::backend::CudaDeviceAttributes, String>,
) -> Option<NumericalProfileRejection> {
    use ferrum_interfaces::vnext::{
        UpstreamMarkerV2Profile as Profile, LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID,
    };
    if !matches!(requested, ferrum_types::NumericalExecutionPolicy::Auto) {
        return None;
    }
    let required = [
        Profile::SwiGluQ6F16,
        Profile::GatedDeltaQ6F16,
        Profile::CausalQ6F16,
    ]
    .into_iter()
    .filter(|kind| {
        profile
            .operations
            .iter()
            .any(|operation| operation.operation_id.as_str() == kind.operation_id())
    })
    .map(Profile::capability_id)
    .collect::<Vec<_>>();
    if required.is_empty() {
        return None;
    }
    let qualification = (|| {
        let ferrum_types::Device::CUDA(ordinal) = device else {
            return Err("Q6 F16 Auto qualification requires a CUDA device".to_owned());
        };
        if u32::try_from(*ordinal).ok() != Some(runtime_ordinal) {
            return Err(
                "Q6 F16 Auto requested device differs from the actual runtime ordinal".to_owned(),
            );
        }
        // These capabilities are advertised only by providers whose compiled
        // native feature, static ABI and complete exports have been checked.
        // The usual composition binding and program checks remain authoritative.
        for required in required
            .into_iter()
            .chain([LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID])
        {
            if !capabilities
                .iter()
                .any(|capability| capability.as_str() == required)
            {
                return Err(format!("Q6 F16 Auto requires native capability {required}"));
            }
        }
        let actual = probe(*ordinal)
            .map_err(|error| format!("Q6 F16 Auto device qualification unavailable: {error}"))?;
        // This is the currently validated architecture scope, not a GPU-name
        // or build-toolchain guess. Other architectures retain strict Auto.
        if actual.compute_capability_major != 12
            || actual.compute_capability_minor != 0
            || actual.multiprocessor_count == 0
        {
            return Err(format!(
                "Q6 F16 Auto is not qualified for CUDA compute capability {}.{} with {} multiprocessors",
                actual.compute_capability_major,
                actual.compute_capability_minor,
                actual.multiprocessor_count,
            ));
        }
        Ok(())
    })();
    qualification.err().map(|reason| NumericalProfileRejection {
        profile_id: profile.id.clone(),
        stage: NumericalProfileRejectionStage::ProgramCompilation,
        reason,
    })
}

fn composite_supports_q8_preparation(
    arithmetic: &ferrum_interfaces::vnext::CompositeNumericalArithmetic,
) -> bool {
    use ferrum_interfaces::vnext::UpstreamMarkerV2Profile as Profile;
    // Match declared numerical contracts, not names of models or tensors.
    // The provider separately decides whether any physical leaf is eligible.
    [
        Profile::GatedDelta,
        Profile::GatedDeltaPrefill,
        Profile::GatedDeltaExtraPrefill,
        Profile::GatedDeltaExtraLargePrefill,
        Profile::GatedDeltaExtraAllRows,
        Profile::GatedDeltaQ6F16,
        Profile::Causal,
        Profile::CausalPrefill,
        Profile::CausalExtraPrefill,
        Profile::CausalExtraLargePrefill,
        Profile::CausalExtraAllRows,
        Profile::CausalQ6F16,
    ]
    .into_iter()
    .any(|profile| profile.arithmetic() == *arithmetic)
}

fn profile_has_shared_m8_q4_route(
    profile: &ferrum_interfaces::vnext::NumericalExecutionProfile,
) -> bool {
    profile
        .operations
        .iter()
        .filter_map(|operation| operation.composite_arithmetic.as_ref())
        .any(composite_has_shared_m8_q4_route)
}

fn composite_has_shared_m8_q4_route(
    arithmetic: &ferrum_interfaces::vnext::CompositeNumericalArithmetic,
) -> bool {
    composite_has_shared_m8_route(
        arithmetic,
        ferrum_interfaces::vnext::ProjectionBlockFormat::Q4K,
    )
}

fn profile_has_shared_m8_q5_route(
    profile: &ferrum_interfaces::vnext::NumericalExecutionProfile,
) -> bool {
    profile
        .operations
        .iter()
        .filter_map(|operation| operation.composite_arithmetic.as_ref())
        .any(composite_has_shared_m8_q5_route)
}

fn composite_has_shared_m8_q5_route(
    arithmetic: &ferrum_interfaces::vnext::CompositeNumericalArithmetic,
) -> bool {
    composite_has_shared_m8_route(
        arithmetic,
        ferrum_interfaces::vnext::ProjectionBlockFormat::Q5K,
    )
}

fn composite_has_shared_m8_route(
    arithmetic: &ferrum_interfaces::vnext::CompositeNumericalArithmetic,
    format: ferrum_interfaces::vnext::ProjectionBlockFormat,
) -> bool {
    use ferrum_interfaces::vnext::{
        NumericalArithmeticStage, CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
    };
    // These two protocol contracts own the factories wired to this option.
    // A route in an unrelated composite (including FFN) cannot qualify it.
    matches!(
        arithmetic.strict_base.operation_id.as_str(),
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID
            | GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID
    ) && arithmetic
        .projections
        .iter()
        .flat_map(|projection| &projection.leaves)
        .flat_map(|leaf| &leaf.arithmetic.stages)
        .any(|stage| {
            matches!(stage,
            NumericalArithmeticStage::UpstreamProjection { policy }
                if policy.format == format
                    && policy.routes.iter().any(route_supports_shared_m8))
        })
}

fn route_supports_shared_m8(
    route: &ferrum_interfaces::vnext::UpstreamProjectionRouteDeclaration,
) -> bool {
    use ferrum_interfaces::vnext::{UpstreamProjectionArithmetic, UpstreamProjectionLayout};
    route.arithmetic == UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2
        && route.layout == UpstreamProjectionLayout::Columns
        && route.contains_rows(8)
}

fn profile_has_shared_m4_m8_route(
    profile: &ferrum_interfaces::vnext::NumericalExecutionProfile,
    format: ferrum_interfaces::vnext::ProjectionBlockFormat,
) -> bool {
    profile
        .operations
        .iter()
        .filter_map(|operation| operation.composite_arithmetic.as_ref())
        .any(|arithmetic| composite_has_shared_m4_m8_route(arithmetic, format))
}

fn composite_has_shared_m4_m8_route(
    arithmetic: &ferrum_interfaces::vnext::CompositeNumericalArithmetic,
    format: ferrum_interfaces::vnext::ProjectionBlockFormat,
) -> bool {
    use ferrum_interfaces::vnext::{
        NumericalArithmeticStage, UpstreamProjectionArithmetic, UpstreamProjectionLayout,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID, DENSE_SWIGLU_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
    };
    // The new policy is also wired to the SwiGLU factory. Keep the original
    // SharedM8 attention-only eligibility above unchanged.
    matches!(
        arithmetic.strict_base.operation_id.as_str(),
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID
            | GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID
            | DENSE_SWIGLU_OPERATION_ID
    ) && arithmetic
        .projections
        .iter()
        .flat_map(|projection| &projection.leaves)
        .flat_map(|leaf| &leaf.arithmetic.stages)
        .any(|stage| {
            matches!(stage,
            NumericalArithmeticStage::UpstreamProjection { policy }
                if policy.format == format
                    && policy.routes.iter().any(|route| {
                        route.arithmetic == UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2
                            && route.layout == UpstreamProjectionLayout::Columns
                            && (route.contains_rows(4) || route.contains_rows(8))
                    }))
        })
}

fn resolve_model_plan(
    engine: &EngineConfig,
    prepared: &PreparedProductionModel,
    catalog: &CapabilityCatalog,
    runtime: &ResolvedRuntimePolicy,
    compilation: &ProgramPlanCompilation,
    numerical_execution: &NumericalProfileResolution,
) -> Result<ResolvedModelPlan> {
    let sources = prepared.sources();
    let config_sha256 = sources
        .fingerprint(ModelArtifactSourceRole::Semantic, "config.json")
        .ok_or_else(|| FerrumError::internal("semantic config.json fingerprint was lost"))?
        .sha256
        .clone();
    let tokenizer_sha256 = sources
        .fingerprint(ModelArtifactSourceRole::Tokenizer, "tokenizer.json")
        .ok_or_else(|| FerrumError::internal("tokenizer.json fingerprint was lost"))?
        .sha256
        .clone();
    let original_sources = sources.original_sources().clone();
    let resolved_sources = sources.resolved_sources().clone();
    let family = prepared.family();

    let engine_provider = match catalog
        .engine_providers()
        .values()
        .collect::<Vec<_>>()
        .as_slice()
    {
        [provider] => *provider,
        providers => {
            return Err(FerrumError::config(format!(
                "vNext product composition requires exactly one engine provider, got {}",
                providers.len()
            )))
        }
    };
    let engine_selection = EngineSelection {
        provider_id: engine_provider.provider_id().clone(),
        contract_version: engine_provider.contract_version(),
        implementation_fingerprint: engine_provider.implementation_fingerprint().to_owned(),
    };
    let (sampling, stop, structured_output) = generation_defaults(engine, prepared)?;

    let inputs = ResolvedModelPlanInputs {
        original_sources,
        resolved_sources,
        config: ModelConfigFingerprint {
            source_file: "config.json".to_owned(),
            sha256: config_sha256,
            typed_config_sha256: family.config_fingerprint().to_owned(),
        },
        external_metadata_id: family.external_metadata_id().clone(),
        prepared_family: family.clone(),
        numerical_execution: numerical_execution.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: TokenizerId::new("tokenizer.huggingface.json")
                .map_err(|error| FerrumError::tokenizer(error.to_string()))?,
            source_file: "tokenizer.json".to_owned(),
            sha256: tokenizer_sha256,
            vocabulary_size: prepared.descriptor().vocabulary_size() as u64,
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime.clone(),
        engine: engine_selection,
        execution_plan: compilation.executable().execution_plan().clone(),
        sampling,
        stop,
        structured_output,
    };
    let (source_evidence, bindings) = resolution_evidence(&inputs)?;
    let registry = ProductionModelFamilyRegistry::new()?;
    let context = ResolvedPlanValidationContext::new(
        &registry,
        &source_evidence,
        compilation.node_resolutions(),
        catalog.device(),
        catalog,
        runtime,
        numerical_execution,
    )
    .with_completion_retention(compilation.completion_retention().clone());
    ResolvedModelPlan::new(inputs, bindings, &context)
        .map_err(|error| FerrumError::model(format!("resolve vNext product plan: {error}")))
}

fn generation_defaults(
    engine: &EngineConfig,
    prepared: &PreparedProductionModel,
) -> Result<(SamplingPolicy, StopPolicy, StructuredOutputPolicy)> {
    let params = &engine.sampling.default_params;
    validate_representable_sampling(params)?;
    let top_k = params
        .top_k
        .map(|value| {
            u32::try_from(value)
                .map_err(|_| FerrumError::config("sampling.top_k exceeds the vNext limit"))
        })
        .transpose()?;
    let sampling = SamplingPolicy::new(
        rational_from_f32(params.temperature, "sampling.temperature")?,
        rational_from_f32(params.top_p, "sampling.top_p")?,
        top_k,
        rational_from_f32(params.min_p.unwrap_or(0.0), "sampling.min_p")?,
        rational_from_f32(params.presence_penalty, "sampling.presence_penalty")?,
        rational_from_f32(params.repetition_penalty, "sampling.repetition_penalty")?,
        params.seed.unwrap_or(DEFAULT_SAMPLER_SEED),
        TriStatePolicy::ModelDefault,
    )
    .map_err(|error| FerrumError::config(error.to_string()))?;
    let maximum_output_tokens = u32::try_from(params.max_tokens)
        .map_err(|_| FerrumError::config("sampling.max_tokens exceeds the vNext limit"))?;
    let mut strings = params.stop_sequences.clone();
    strings.sort();
    strings.dedup();
    let token_ids = prepared
        .family()
        .metadata()
        .special_tokens
        .eos_token_ids
        .clone();
    let special_tokens = &prepared.family().metadata().special_tokens;
    let collision_policy = stop_collision_policy(
        &token_ids,
        special_tokens.bos_token_id,
        special_tokens.pad_token_id,
    )?;
    let stop = StopPolicy {
        maximum_output_tokens,
        token_ids,
        strings,
        collision_policy,
    };
    let structured_output = match &params.response_format {
        ResponseFormat::Text => StructuredOutputPolicy::Disabled,
        ResponseFormat::JsonObject => StructuredOutputPolicy::JsonObject,
        ResponseFormat::JsonSchema(schema) => StructuredOutputPolicy::JsonSchema {
            schema_sha256: format!("{:x}", Sha256::digest(schema.as_bytes())),
        },
    };
    Ok((sampling, stop, structured_output))
}

fn stop_collision_policy(
    token_ids: &BTreeSet<u32>,
    bos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
) -> Result<StopTokenCollisionPolicy> {
    let mut model_roles = BTreeSet::new();
    if !token_ids.is_empty() {
        model_roles.insert(SpecialTokenRole::Eos);
    }
    if bos_token_id.is_some_and(|token_id| token_ids.contains(&token_id)) {
        model_roles.insert(SpecialTokenRole::Bos);
    }
    if pad_token_id.is_some_and(|token_id| token_ids.contains(&token_id)) {
        model_roles.insert(SpecialTokenRole::Pad);
    }
    StopTokenCollisionPolicy::new(model_roles)
        .map_err(|error| FerrumError::config(error.to_string()))
}

fn validate_representable_sampling(params: &SamplingParams) -> Result<()> {
    params.validate()?;
    if params.frequency_penalty != 0.0
        || params.tfs.is_some()
        || params.typical_p.is_some()
        || params.mirostat.is_some()
    {
        return Err(FerrumError::unsupported(
            "the resolved vNext startup policy does not yet represent frequency_penalty, tfs, typical_p, or mirostat",
        ));
    }
    Ok(())
}

fn rational_from_f32(value: f32, field: &'static str) -> Result<RationalValue> {
    if !value.is_finite() {
        return Err(FerrumError::config(format!("{field} must be finite")));
    }
    let rendered = value.to_string();
    let (mantissa, exponent) = rendered
        .split_once(['e', 'E'])
        .map(|(mantissa, exponent)| exponent.parse::<i32>().map(|exponent| (mantissa, exponent)))
        .transpose()
        .map_err(|error| FerrumError::config(format!("parse {field}: {error}")))?
        .unwrap_or((rendered.as_str(), 0));
    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.trim_start_matches(['-', '+']);
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let digits = format!("{whole}{fraction}");
    let mut numerator = digits
        .parse::<i128>()
        .map_err(|error| FerrumError::config(format!("parse {field}: {error}")))?;
    if negative {
        numerator = -numerator;
    }
    let decimal_places = i32::try_from(fraction.len())
        .map_err(|_| FerrumError::config(format!("{field} decimal width exceeds i32")))?;
    let scale = decimal_places - exponent;
    let denominator = if scale > 0 {
        10_u128
            .checked_pow(scale as u32)
            .ok_or_else(|| FerrumError::config(format!("{field} denominator overflow")))?
    } else {
        let multiplier = 10_i128
            .checked_pow((-scale) as u32)
            .ok_or_else(|| FerrumError::config(format!("{field} numerator overflow")))?;
        numerator = numerator
            .checked_mul(multiplier)
            .ok_or_else(|| FerrumError::config(format!("{field} numerator overflow")))?;
        1
    };
    let numerator = i64::try_from(numerator)
        .map_err(|_| FerrumError::config(format!("{field} numerator exceeds i64")))?;
    let denominator = u64::try_from(denominator)
        .map_err(|_| FerrumError::config(format!("{field} denominator exceeds u64")))?;
    RationalValue::new(numerator, denominator)
        .map_err(|error| FerrumError::config(error.to_string()))
}

fn resolution_evidence(
    inputs: &ResolvedModelPlanInputs,
) -> Result<(
    Vec<ResolutionSourceEvidence<'static>>,
    Vec<ResolutionDecisionBinding>,
)> {
    const USER_FIELDS: &[ResolutionField] = &[ResolutionField::OriginalSources];
    const MODEL_FIELDS: &[ResolutionField] = &[
        ResolutionField::ResolvedSources,
        ResolutionField::Config,
        ResolutionField::ExternalMetadata,
        ResolutionField::Family,
        ResolutionField::WeightSchema,
        ResolutionField::WeightFormat,
        ResolutionField::Tokenizer,
        ResolutionField::Template,
        ResolutionField::SpecialTokens,
    ];
    const CAPABILITY_FIELDS: &[ResolutionField] = &[
        ResolutionField::Device,
        ResolutionField::Capabilities,
        ResolutionField::Engine,
    ];
    const RUNTIME_FIELDS: &[ResolutionField] = &[
        ResolutionField::RuntimePreset,
        ResolutionField::RuntimeMemory,
        ResolutionField::Admission,
    ];
    const PLANNER_FIELDS: &[ResolutionField] = &[
        ResolutionField::ExecutionPlan,
        ResolutionField::NumericalExecution,
    ];
    const DEFAULT_FIELDS: &[ResolutionField] = &[
        ResolutionField::Sampling,
        ResolutionField::Stop,
        ResolutionField::StructuredOutput,
    ];
    let groups = [
        (ResolutionDecisionSource::UserInput, USER_FIELDS),
        (ResolutionDecisionSource::TypedModelResolution, MODEL_FIELDS),
        (
            ResolutionDecisionSource::CapabilityResolution,
            CAPABILITY_FIELDS,
        ),
        (ResolutionDecisionSource::RuntimePreset, RUNTIME_FIELDS),
        (ResolutionDecisionSource::Planner, PLANNER_FIELDS),
        (ResolutionDecisionSource::ProductDefault, DEFAULT_FIELDS),
    ];
    let implementation_fingerprint = ResolutionFingerprint::new(format!(
        "{:x}",
        Sha256::digest(include_bytes!("product_composition.rs"))
    ))
    .map_err(|error| FerrumError::internal(error.to_string()))?;
    let mut evidence = Vec::with_capacity(groups.len());
    let mut bindings = Vec::new();
    for (source, fields) in groups {
        let source_name = decision_source_name(source);
        let artifact_id = ResolutionArtifactId::new(format!("artifact.product.{source_name}"))
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let mut document = Map::new();
        let mut field_paths = BTreeSet::new();
        for field in fields {
            let name = field.as_str();
            document.insert(name.to_owned(), resolution_value(inputs, *field)?);
            let path = format!("/{name}");
            field_paths.insert(path.clone());
            bindings.push(
                ResolutionDecisionBinding::new(
                    *field,
                    source,
                    ResolutionReasonId::new(format!("reason.product.{name}"))
                        .map_err(|error| FerrumError::internal(error.to_string()))?,
                    artifact_id.clone(),
                    path,
                )
                .map_err(|error| FerrumError::internal(error.to_string()))?,
            );
        }
        let source_bytes = serde_json::to_vec(&Value::Object(document)).map_err(|error| {
            FerrumError::internal(format!(
                "serialize {source_name} resolution evidence: {error}"
            ))
        })?;
        evidence.push(
            ResolutionSourceEvidence::new(
                artifact_id,
                source,
                ResolutionSourceProvenance::Upstream {
                    producer_id: PRODUCT_COMPOSITION_PRODUCER.to_owned(),
                    producer_version: PRODUCT_COMPOSITION_VERSION,
                    producer_implementation_fingerprint: implementation_fingerprint.clone(),
                    revision: env!("CARGO_PKG_VERSION").to_owned(),
                    artifact_locator: format!("product-composition/{source_name}"),
                },
                source_bytes,
                field_paths,
                &JSON_RESOLUTION_SOURCE_PARSER,
            )
            .map_err(|error| FerrumError::internal(error.to_string()))?,
        );
    }
    Ok((evidence, bindings))
}

fn resolution_value(inputs: &ResolvedModelPlanInputs, field: ResolutionField) -> Result<Value> {
    let value = match field {
        ResolutionField::OriginalSources => serde_json::to_value(&inputs.original_sources),
        ResolutionField::ResolvedSources => serde_json::to_value(&inputs.resolved_sources),
        ResolutionField::Config => serde_json::to_value(&inputs.config),
        ResolutionField::ExternalMetadata => serde_json::to_value(&inputs.external_metadata_id),
        ResolutionField::Family => serde_json::to_value(inputs.prepared_family.family_id()),
        ResolutionField::NumericalExecution => serde_json::to_value(&inputs.numerical_execution),
        ResolutionField::WeightSchema => {
            serde_json::to_value(inputs.prepared_family.weight_schema())
        }
        ResolutionField::WeightFormat => {
            serde_json::to_value(&inputs.prepared_family.weight_schema().format_id)
        }
        ResolutionField::Tokenizer => serde_json::to_value(&inputs.tokenizer),
        ResolutionField::Template => {
            serde_json::to_value(&inputs.prepared_family.metadata().template)
        }
        ResolutionField::SpecialTokens => {
            serde_json::to_value(&inputs.prepared_family.metadata().special_tokens)
        }
        ResolutionField::Device => serde_json::to_value(&inputs.device),
        ResolutionField::Capabilities => serde_json::to_value(&inputs.capabilities),
        ResolutionField::RuntimePreset => Ok(json!({
            "policy_id": inputs.runtime.policy_id(),
            "version": inputs.runtime.version(),
            "scheduling": inputs.runtime.scheduling(),
        })),
        ResolutionField::RuntimeMemory => serde_json::to_value(inputs.runtime.memory()),
        ResolutionField::Admission => serde_json::to_value(inputs.runtime.admission()),
        ResolutionField::Engine => serde_json::to_value(&inputs.engine),
        ResolutionField::ExecutionPlan => Ok(json!(inputs.execution_plan.plan_hash().as_str())),
        ResolutionField::Sampling => serde_json::to_value(&inputs.sampling),
        ResolutionField::Stop => serde_json::to_value(&inputs.stop),
        ResolutionField::StructuredOutput => serde_json::to_value(&inputs.structured_output),
    };
    value.map_err(|error| FerrumError::internal(format!("serialize resolution field: {error}")))
}

const fn decision_source_name(source: ResolutionDecisionSource) -> &'static str {
    match source {
        ResolutionDecisionSource::UserInput => "user-input",
        ResolutionDecisionSource::CommandLine => "command-line",
        ResolutionDecisionSource::ConfigFile => "config-file",
        ResolutionDecisionSource::ModelMetadata => "model-metadata",
        ResolutionDecisionSource::TypedModelResolution => "typed-model",
        ResolutionDecisionSource::ProductDefault => "product-default",
        ResolutionDecisionSource::RuntimePreset => "runtime-preset",
        ResolutionDecisionSource::CapabilityResolution => "capability",
        ResolutionDecisionSource::Planner => "planner",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q6_auto_profile() -> ferrum_interfaces::vnext::NumericalExecutionProfile {
        use ferrum_interfaces::vnext::{
            ElementType, ModelFamilyId, NumericalExecutionProfile, NumericalOperationContract,
            NumericalProfileId, OperationId, ProgramValueId, Q6MmqF32Policy,
            UpstreamMarkerV2Profile as Profile, LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID,
        };
        let mut operations = [
            Profile::SwiGluQ6F16,
            Profile::GatedDeltaQ6F16,
            Profile::CausalQ6F16,
        ]
        .into_iter()
        .map(|kind| NumericalOperationContract {
            operation_id: OperationId::new(kind.operation_id()).unwrap(),
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: Some(kind.arithmetic()),
        })
        .collect::<Vec<_>>();
        operations.push(NumericalOperationContract {
            operation_id: OperationId::new(LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID)
                .unwrap(),
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: Some(Q6MmqF32Policy::new().staged()),
            composite_arithmetic: None,
        });
        let activation = ProgramValueId::new("value.activation").unwrap();
        let profile = NumericalExecutionProfile {
            id: NumericalProfileId::new("numerical.test.q6").unwrap(),
            version: ContractVersion::new(1, 0),
            family_id: ModelFamilyId::new("family.test").unwrap(),
            primary_activation: activation.clone(),
            boundaries: [(activation, ElementType::F32)].into_iter().collect(),
            states: Vec::new(),
            kv_storage: Vec::new(),
            operations,
        };
        profile.validate().unwrap();
        profile
    }

    fn q6_auto_capabilities() -> BTreeSet<ferrum_interfaces::vnext::CapabilityId> {
        use ferrum_interfaces::vnext::{
            CapabilityId, UpstreamMarkerV2Profile as Profile,
            LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID,
        };
        [
            Profile::SwiGluQ6F16.capability_id(),
            Profile::GatedDeltaQ6F16.capability_id(),
            Profile::CausalQ6F16.capability_id(),
            LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_CAPABILITY_ID,
        ]
        .into_iter()
        .map(|id| CapabilityId::new(id).unwrap())
        .collect()
    }

    fn qualified_q6_device() -> ferrum_kernels::backend::CudaDeviceAttributes {
        ferrum_kernels::backend::CudaDeviceAttributes {
            compute_capability_major: 12,
            compute_capability_minor: 0,
            multiprocessor_count: 32,
        }
    }

    fn q8_compressed_capacity() -> ferrum_types::StartupMemoryPlan {
        use ferrum_types::{DeviceMemorySnapshot, StartupMemoryRequest, StartupResourceLimits};
        ferrum_types::fit_startup_resources(
            &StartupMemoryRequest {
                device: DeviceMemorySnapshot {
                    capacity_bytes: 16_384,
                    available_bytes: 12_288,
                    source: "selected-device-memory".into(),
                },
                usable_capacity_bytes: 8192,
                context_is_explicit: false,
                sequences_is_explicit: true,
                batch_is_explicit: false,
            },
            StartupResourceLimits {
                context_tokens: 128,
                max_sequences: 4,
                max_batch_tokens: 16,
            },
            |_| Ok(4096),
        )
        .unwrap()
    }

    #[test]
    fn q8_auto_requires_budget_exact_profile_and_selected_device_attributes() {
        use ferrum_types::Q8ProjectionWeightPreparation as Preparation;
        let profile = q6_auto_profile();
        let mut engine = EngineConfig::default();
        engine.backend.device = ferrum_types::Device::CUDA(3);
        engine.runtime.q8_projection_weight_preparation = Preparation::Auto;
        engine.runtime.startup_memory_request = Some(q8_compressed_capacity().request);
        assert!(
            q8_auto_preparation_eligible(&engine, &profile, 3, |ordinal| {
                assert_eq!(ordinal, 3);
                Ok(qualified_q6_device())
            })
            .unwrap()
        );
        assert!(q8_auto_preparation_eligible(&engine, &profile, 0, |_| {
            panic!("must not probe a device other than the runtime owner")
        })
        .is_err());
        assert!(q8_auto_preparation_eligible(&engine, &profile, 3, |_| {
            Err("driver query failed".into())
        })
        .is_err());
        for (major, minor, count) in [(8, 9, 32), (12, 1, 32), (12, 0, 0)] {
            assert!(!q8_auto_preparation_eligible(&engine, &profile, 3, |_| {
                Ok(ferrum_kernels::backend::CudaDeviceAttributes {
                    compute_capability_major: major,
                    compute_capability_minor: minor,
                    multiprocessor_count: count,
                })
            })
            .unwrap());
        }
        let mut unrelated = profile.clone();
        unrelated.operations.iter_mut().for_each(|operation| {
            operation.composite_arithmetic = None;
        });
        assert!(!q8_auto_preparation_eligible(&engine, &unrelated, 3, |_| {
            panic!("unsupported numerical contracts must not reach the CUDA probe")
        })
        .unwrap());
        for mode in [Preparation::Compressed, Preparation::PredecodedF32] {
            engine.runtime.q8_projection_weight_preparation = mode;
            assert!(!q8_auto_preparation_eligible(&engine, &profile, 3, |_| {
                panic!("explicit preparation keeps its original behavior")
            })
            .unwrap());
        }
        engine.runtime.q8_projection_weight_preparation = Preparation::Auto;
        engine.runtime.startup_memory_request = None;
        assert!(!q8_auto_preparation_eligible(&engine, &profile, 3, |_| {
            panic!("Auto cannot allocate optional storage without a fitted budget")
        })
        .unwrap());
        engine.runtime.startup_memory_request = Some(q8_compressed_capacity().request);
        engine.backend.device = ferrum_types::Device::CPU;
        assert!(!q8_auto_preparation_eligible(&engine, &profile, 0, |_| {
            panic!("CPU Auto must remain compressed")
        })
        .unwrap());
    }

    #[test]
    fn q8_auto_optional_capacity_preserves_limits_and_checks_both_workloads() {
        use ferrum_types::StartupWorkload;
        let baseline = q8_compressed_capacity();
        let mut seen = Vec::new();
        let verified = verify_optional_q8_capacity(
            &baseline,
            |workload| {
                seen.push(workload);
                Ok(match workload {
                    StartupWorkload::Prefill { .. } => 8192,
                    StartupWorkload::Decode { .. } => 8191,
                })
            },
            || {
                Ok(ferrum_types::DeviceMemorySnapshot {
                    available_bytes: 8192,
                    ..baseline.request.device.clone()
                })
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            seen,
            [
                StartupWorkload::Prefill {
                    context_tokens: 128,
                    chunk_tokens: 16
                },
                StartupWorkload::Decode {
                    context_tokens: 128,
                    active_sequences: 4
                },
            ]
        );
        assert_eq!(verified.request, baseline.request);
        assert_eq!(verified.requested, baseline.requested);
        assert_eq!(verified.selected, baseline.selected);
        assert_eq!(verified.reasons, baseline.reasons);
        assert_eq!(
            (verified.context_peak_bytes, verified.decode_peak_bytes),
            (8192, 8191)
        );
        for prefill_overflow in [true, false] {
            assert!(verify_optional_q8_capacity(
                &baseline,
                |workload| Ok(
                    if matches!(workload, StartupWorkload::Prefill { .. }) == prefill_overflow {
                        8193
                    } else {
                        8192
                    }
                ),
                || panic!("a static capacity refusal needs no driver probe"),
            )
            .unwrap()
            .is_none());
        }
        assert!(
            verify_optional_q8_capacity(
                &baseline,
                |_| Ok(8192),
                || Ok(ferrum_types::DeviceMemorySnapshot {
                    available_bytes: 8191,
                    ..baseline.request.device.clone()
                }),
            )
            .unwrap()
            .is_none(),
            "module footprint must not reduce the previously fixed budget"
        );
    }

    #[test]
    fn q8_auto_optional_capacity_never_masks_estimator_or_driver_errors() {
        let baseline = q8_compressed_capacity();
        assert!(matches!(
            verify_optional_q8_capacity(
                &baseline,
                |_| Err(FerrumError::model("invalid provider shape")),
                || panic!("invalid provider metadata must abort before the memory probe"),
            ),
            Err(FerrumError::Model { .. })
        ));
        assert!(matches!(
            verify_optional_q8_capacity(
                &baseline,
                |_| Ok(4096),
                || Err(FerrumError::device("memory query failed")),
            ),
            Err(FerrumError::Device { .. })
        ));
        assert!(matches!(
            verify_optional_q8_capacity(
                &baseline,
                |_| Ok(4096),
                || Ok(ferrum_types::DeviceMemorySnapshot {
                    capacity_bytes: baseline.request.device.capacity_bytes / 2,
                    available_bytes: 8192,
                    source: "changed-capacity".into(),
                }),
            ),
            Err(FerrumError::Device { .. })
        ));
    }

    #[test]
    fn q6_auto_requires_all_native_capabilities_and_selected_device_attributes() {
        let profile = q6_auto_profile();
        let capabilities = q6_auto_capabilities();
        assert!(auto_numerical_profile_rejection(
            &ferrum_types::NumericalExecutionPolicy::Auto,
            &profile,
            &ferrum_types::Device::CUDA(3),
            3,
            &capabilities,
            |ordinal| {
                assert_eq!(ordinal, 3);
                Ok(qualified_q6_device())
            },
        )
        .is_none());
        assert!(auto_numerical_profile_rejection(
            &ferrum_types::NumericalExecutionPolicy::Auto,
            &profile,
            &ferrum_types::Device::CUDA(3),
            0,
            &capabilities,
            |_| panic!("qualification must not probe a device other than the runtime owner"),
        )
        .is_some());
        for missing in &capabilities {
            let mut incomplete = capabilities.clone();
            incomplete.remove(missing);
            let rejection = auto_numerical_profile_rejection(
                &ferrum_types::NumericalExecutionPolicy::Auto,
                &profile,
                &ferrum_types::Device::CUDA(0),
                0,
                &incomplete,
                |_| panic!("missing native capability must reject before device discovery"),
            )
            .unwrap();
            assert_eq!(rejection.profile_id, profile.id);
            assert_eq!(
                rejection.stage,
                NumericalProfileRejectionStage::ProgramCompilation
            );
            assert!(rejection.reason.contains(missing.as_str()));
        }
        assert_eq!(
            capabilities,
            q6_auto_capabilities(),
            "qualification does not remove catalog capabilities"
        );
    }

    #[test]
    fn q6_auto_rejects_unknown_unqualified_or_non_cuda_devices() {
        let profile = q6_auto_profile();
        let capabilities = q6_auto_capabilities();
        for (major, minor, count) in [(8, 9, 32), (12, 1, 32), (12, 0, 0)] {
            assert!(auto_numerical_profile_rejection(
                &ferrum_types::NumericalExecutionPolicy::Auto,
                &profile,
                &ferrum_types::Device::CUDA(0),
                0,
                &capabilities,
                |_| Ok(ferrum_kernels::backend::CudaDeviceAttributes {
                    compute_capability_major: major,
                    compute_capability_minor: minor,
                    multiprocessor_count: count,
                }),
            )
            .is_some());
        }
        let failure = auto_numerical_profile_rejection(
            &ferrum_types::NumericalExecutionPolicy::Auto,
            &profile,
            &ferrum_types::Device::CUDA(0),
            0,
            &capabilities,
            |_| Err("driver attribute read failed".into()),
        )
        .unwrap();
        assert_eq!(failure.profile_id, profile.id);
        assert_eq!(
            failure.stage,
            NumericalProfileRejectionStage::ProgramCompilation
        );
        assert!(failure.reason.contains("driver attribute read failed"));
        let devices = [
            ferrum_types::Device::CPU,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ferrum_types::Device::Metal,
        ];
        for device in devices {
            assert!(auto_numerical_profile_rejection(
                &ferrum_types::NumericalExecutionPolicy::Auto,
                &profile,
                &device,
                0,
                &capabilities,
                |_| panic!("non-CUDA profiles must not query the CUDA driver"),
            )
            .is_some());
        }
    }

    #[test]
    fn q6_auto_preserves_explicit_requests_and_other_profiles() {
        let profile = q6_auto_profile();
        assert!(auto_numerical_profile_rejection(
            &ferrum_types::NumericalExecutionPolicy::Require(profile.id.clone()),
            &profile,
            &ferrum_types::Device::CPU,
            0,
            &BTreeSet::new(),
            |_| panic!("explicit numerical requests keep existing provider validation"),
        )
        .is_none());
        let mut strict = profile;
        strict.id =
            ferrum_interfaces::vnext::NumericalProfileId::new("numerical.test.strict").unwrap();
        strict
            .operations
            .retain(|operation| operation.composite_arithmetic.is_none());
        let head = &mut strict.operations[0];
        head.operation_id = ferrum_interfaces::vnext::OperationId::new(
            ferrum_interfaces::vnext::LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
        )
        .unwrap();
        head.staged_arithmetic = None;
        head.multiplication_type = Some(ferrum_interfaces::vnext::ElementType::F32);
        head.accumulation_type = Some(ferrum_interfaces::vnext::ElementType::F32);
        strict.validate().unwrap();
        assert!(auto_numerical_profile_rejection(
            &ferrum_types::NumericalExecutionPolicy::Auto,
            &strict,
            &ferrum_types::Device::CPU,
            0,
            &BTreeSet::new(),
            |_| panic!("strict fallback does not need CUDA qualification"),
        )
        .is_none());
    }

    #[test]
    fn q8_projection_weight_preparation_requires_wired_attention_numerics() {
        use ferrum_interfaces::vnext::{
            OperationId, UpstreamMarkerV2Profile as Profile, DENSE_SWIGLU_OPERATION_ID,
        };
        for profile in [
            Profile::GatedDelta,
            Profile::Causal,
            Profile::GatedDeltaExtraAllRows,
            Profile::CausalExtraAllRows,
            Profile::GatedDeltaQ6F16,
            Profile::CausalQ6F16,
        ] {
            assert!(composite_supports_q8_preparation(&profile.arithmetic()));
        }
        for profile in [
            Profile::SwiGlu,
            Profile::SwiGluG32MmqPrefill,
            Profile::SwiGluExtraAllRows,
            Profile::SwiGluQ6F16,
            Profile::GatedDeltaG32MmqPrefill,
            Profile::CausalG32MmqPrefill,
        ] {
            assert!(!composite_supports_q8_preparation(&profile.arithmetic()));
        }
        let mut unrelated = Profile::GatedDelta.arithmetic();
        unrelated.strict_base.operation_id = OperationId::new(DENSE_SWIGLU_OPERATION_ID).unwrap();
        assert!(!composite_supports_q8_preparation(&unrelated));
    }

    #[test]
    fn int8_kv_resolves_attention_without_changing_storage_precision() {
        use ferrum_types::{AttentionExecutionPolicy as Policy, KvCacheDtype as Dtype};
        for requested in [Policy::Auto, Policy::Portable] {
            assert_eq!(
                cuda_attention_policy_for_kv(requested, Dtype::Int8).unwrap(),
                Policy::Portable,
            );
        }
        assert!(cuda_attention_policy_for_kv(Policy::NativeAdaptive, Dtype::Int8).is_err());
        for requested in [Policy::Auto, Policy::Portable, Policy::NativeAdaptive] {
            assert_eq!(
                cuda_attention_policy_for_kv(requested, Dtype::Fp16).unwrap(),
                requested,
            );
        }
        for unsupported in [Dtype::Bf16, Dtype::Fp8] {
            assert!(cuda_attention_policy_for_kv(Policy::Auto, unsupported).is_err());
        }
    }

    #[test]
    fn decimal_sampling_values_are_canonical_rationals() {
        let value = rational_from_f32(1.1, "test").unwrap();
        assert_eq!(value.numerator(), 11);
        assert_eq!(value.denominator(), 10);
        let value = rational_from_f32(0.000_001, "test").unwrap();
        assert_eq!(value.numerator(), 1);
        assert_eq!(value.denominator(), 1_000_000);
    }

    #[test]
    fn stop_collision_policy_includes_every_model_role_aliased_by_default_eos() {
        let token_ids = BTreeSet::from([10, 20]);
        let policy = stop_collision_policy(&token_ids, Some(10), Some(20)).unwrap();
        assert_eq!(
            policy.allowed_model_roles(),
            &BTreeSet::from([
                SpecialTokenRole::Bos,
                SpecialTokenRole::Eos,
                SpecialTokenRole::Pad,
            ])
        );

        let distinct = stop_collision_policy(&BTreeSet::new(), Some(10), Some(20)).unwrap();
        assert!(distinct.allowed_model_roles().is_empty());
    }
    #[test]
    fn shared_m8_q4_requires_original_marker_mmvq_columns_and_declared_eight_rows() {
        use ferrum_interfaces::vnext::{
            UpstreamProjectionArithmetic as Arithmetic, UpstreamProjectionLayout as Layout,
            UpstreamProjectionRouteDeclaration,
        };
        for arithmetic in [
            Arithmetic::MmvqQ8_1MarkerV2,
            Arithmetic::MmvqQ8_1V1,
            Arithmetic::MmqDs4MarkerV2,
        ] {
            for layout in [Layout::Columns, Layout::Channels] {
                for rows in [vec![1, 4], vec![8], vec![16, 32]] {
                    let route = UpstreamProjectionRouteDeclaration {
                        arithmetic,
                        layout,
                        local_rows: rows.iter().copied().collect(),
                        prefill_rows: None,
                    };
                    assert_eq!(
                        route_supports_shared_m8(&route),
                        arithmetic == Arithmetic::MmvqQ8_1MarkerV2
                            && layout == Layout::Columns
                            && rows.contains(&8)
                    );
                }
            }
        }
    }

    #[test]
    fn shared_m8_q4_requires_a_supported_attention_contract_with_original_mmvq() {
        use ferrum_interfaces::vnext::{
            OperationId, UpstreamMarkerV2Profile as Profile, DENSE_SWIGLU_OPERATION_ID,
        };
        // Extended formats retain the independently declared original Q4 route.
        for profile in [
            Profile::Causal,
            Profile::GatedDelta,
            Profile::CausalExtraAllRows,
            Profile::GatedDeltaExtraAllRows,
            Profile::CausalQ6F16,
            Profile::GatedDeltaQ6F16,
        ] {
            assert!(composite_has_shared_m8_q4_route(&profile.arithmetic()));
        }
        for profile in [
            Profile::SwiGlu,
            Profile::CausalG32MmqPrefill,
            Profile::GatedDeltaG32MmqPrefill,
            Profile::SwiGluExtraAllRows,
            Profile::SwiGluQ6F16,
        ] {
            assert!(!composite_has_shared_m8_q4_route(&profile.arithmetic()));
        }
        let mut unrelated = Profile::Causal.arithmetic();
        unrelated.strict_base.operation_id = OperationId::new(DENSE_SWIGLU_OPERATION_ID).unwrap();
        assert!(
            !composite_has_shared_m8_q4_route(&unrelated),
            "an eligible leaf in an unwired composite cannot activate the product option"
        );
    }

    #[test]
    fn shared_m8_q5_requires_a_supported_attention_contract_with_original_mmvq() {
        use ferrum_interfaces::vnext::{
            OperationId, UpstreamMarkerV2Profile as Profile, DENSE_SWIGLU_OPERATION_ID,
        };
        // Extended formats retain the independently declared original Q5 route.
        for profile in [
            Profile::Causal,
            Profile::GatedDelta,
            Profile::CausalExtraAllRows,
            Profile::GatedDeltaExtraAllRows,
            Profile::CausalQ6F16,
            Profile::GatedDeltaQ6F16,
        ] {
            assert!(composite_has_shared_m8_q5_route(&profile.arithmetic()));
        }
        for profile in [
            Profile::SwiGlu,
            Profile::CausalG32MmqPrefill,
            Profile::GatedDeltaG32MmqPrefill,
            Profile::SwiGluExtraAllRows,
            Profile::SwiGluQ6F16,
        ] {
            assert!(!composite_has_shared_m8_q5_route(&profile.arithmetic()));
        }
        let mut q4_only = Profile::Causal.arithmetic();
        for projection in &mut q4_only.projections {
            projection
                .leaves
                .retain(|leaf| leaf.format != ferrum_interfaces::vnext::ProjectionBlockFormat::Q5K);
        }
        assert!(composite_has_shared_m8_q4_route(&q4_only));
        assert!(!composite_has_shared_m8_q5_route(&q4_only));
        let mut unrelated = Profile::Causal.arithmetic();
        unrelated.strict_base.operation_id = OperationId::new(DENSE_SWIGLU_OPERATION_ID).unwrap();
        assert!(
            !composite_has_shared_m8_q5_route(&unrelated),
            "an eligible leaf in an unwired composite cannot activate the product option"
        );
    }

    #[test]
    fn shared_m4_m8_accepts_wired_attention_and_swiglu_routes_only() {
        use ferrum_interfaces::vnext::{
            NumericalArithmeticStage, OperationId, ProjectionBlockFormat as Format,
            UpstreamMarkerV2Profile as Profile, UpstreamProjectionArithmetic as Arithmetic,
            UpstreamProjectionLayout as Layout,
        };
        for format in [Format::Q4K, Format::Q5K] {
            for profile in [
                Profile::Causal,
                Profile::GatedDelta,
                Profile::SwiGlu,
                Profile::CausalExtraAllRows,
                Profile::GatedDeltaExtraAllRows,
                Profile::SwiGluExtraAllRows,
                Profile::CausalQ6F16,
                Profile::GatedDeltaQ6F16,
                Profile::SwiGluQ6F16,
            ] {
                assert!(composite_has_shared_m4_m8_route(
                    &profile.arithmetic(),
                    format
                ));
            }
            for profile in [
                Profile::CausalG32MmqPrefill,
                Profile::GatedDeltaG32MmqPrefill,
                Profile::SwiGluG32MmqPrefill,
            ] {
                assert!(!composite_has_shared_m4_m8_route(
                    &profile.arithmetic(),
                    format
                ));
            }
            let mut unrelated = Profile::Causal.arithmetic();
            unrelated.strict_base.operation_id = OperationId::new("test.unwired").unwrap();
            assert!(!composite_has_shared_m4_m8_route(&unrelated, format));

            for (rows, layout, arithmetic, expected) in [
                (
                    vec![1],
                    Layout::Columns,
                    Arithmetic::MmvqQ8_1MarkerV2,
                    false,
                ),
                (vec![4], Layout::Columns, Arithmetic::MmvqQ8_1MarkerV2, true),
                (vec![8], Layout::Columns, Arithmetic::MmvqQ8_1MarkerV2, true),
                (
                    vec![16, 32],
                    Layout::Columns,
                    Arithmetic::MmvqQ8_1MarkerV2,
                    false,
                ),
                (
                    vec![4, 8],
                    Layout::Channels,
                    Arithmetic::MmvqQ8_1MarkerV2,
                    false,
                ),
                (vec![4, 8], Layout::Columns, Arithmetic::MmvqQ8_1V1, false),
            ] {
                let mut contract = Profile::SwiGlu.arithmetic();
                for stage in contract
                    .projections
                    .iter_mut()
                    .flat_map(|projection| &mut projection.leaves)
                    .flat_map(|leaf| &mut leaf.arithmetic.stages)
                {
                    if let NumericalArithmeticStage::UpstreamProjection { policy } = stage {
                        for route in &mut policy.routes {
                            route.local_rows = rows.iter().copied().collect();
                            route.prefill_rows = None;
                            route.layout = layout;
                            route.arithmetic = arithmetic;
                        }
                    }
                }
                assert_eq!(
                    composite_has_shared_m4_m8_route(&contract, format),
                    expected
                );
            }
        }
        let mut q4_only = Profile::SwiGlu.arithmetic();
        for projection in &mut q4_only.projections {
            projection.leaves.retain(|leaf| leaf.format != Format::Q5K);
        }
        assert!(composite_has_shared_m4_m8_route(&q4_only, Format::Q4K));
        assert!(!composite_has_shared_m4_m8_route(&q4_only, Format::Q5K));
    }
}
