use super::{
    backend::Backend,
    shared::{id, initialize_resources, TensorPolicy},
};
use anyhow::{bail, Context, Result};
use ferrum_execution::{PreparedTensorExecutor, TensorExecutionOutput, TensorExecutorConfig};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::qwen_image21::{
    QwenImage21Block0QkNormFamily as Family, QwenImage21Config, QwenImage21QkFrontend as Frontend,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::Arc};

pub fn execute(
    config: &QwenImage21Config,
    source: &SafetensorsArchive,
    backend: &Backend,
    frontend: Frontend,
    input: &[u8],
    frequencies: Option<&[u8]>,
    inspect: impl FnOnce(&[TensorExecutionOutput], &Value) -> Result<Value>,
) -> Result<(Value, Value)> {
    match (frontend, frequencies) {
        (Frontend::Norm, None) => {}
        (Frontend::Rotary, Some(bytes))
            if config.head_dim() % 2 == 0
                && u64::try_from(bytes.len())?
                    == config
                        .head_dim()
                        .checked_mul(4)
                        .context("frequency size overflows")? => {}
        _ => bail!("frontend and explicit F32 [1,D/2,2] frequency input disagree"),
    }
    match backend {
        Backend::Reference => {
            let composition = ReferenceVNextComposition::create(id(
                "device.reference.qwen_image21.block0-qk-norm",
            )?)?;
            run(
                config,
                source,
                frontend,
                input,
                frequencies,
                composition.runtime().clone(),
                composition.registry(),
                composition.catalog(),
                composition.weight_materializers(),
                frontend.reference_profile_id(),
                composition.runtime().descriptor().total_memory_bytes,
                65536,
                json!({"backend":"Reference", "dense":"FP32 sequential; BF16 boundary", "head_rms_norm":"reference pairwise FP32; sqrt/div; normalize B then scale B", "adjacent_rope":(frontend==Frontend::Rotary).then_some("FP32 cross products then explicit FMA; BF16 boundary")}),
                inspect,
            )
        }
        Backend::Cuda(options) => {
            #[cfg(not(feature = "cuda"))]
            {
                let _ = options;
                bail!("CUDA requires --features cuda")
            }
            #[cfg(feature = "cuda")]
            {
                use ferrum_kernels::backend::cuda::{
                    vnext_ops::*,
                    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
                };
                let runtime = Arc::new(CudaDeviceRuntime::new_tensor_eager(
                    CudaTensorRuntimeConfig {
                        device: cuda_tensor_runtime_config(
                            options.device,
                            id(format!("device.cuda.{}", options.device))?,
                        )?,
                        usable_memory_bytes: options.usable_memory_bytes,
                        bootstrap_memory_bytes: options.bootstrap_memory_bytes,
                        stream_memory_bytes: options.stream_memory_bytes,
                        maximum_streams: options.maximum_streams,
                        maximum_in_flight_per_stream: options.maximum_in_flight_per_stream,
                        maximum_commands_per_submission: options.maximum_commands_per_submission,
                        opaque_memory_profile: options.opaque_memory_profile.clone(),
                    },
                )?);
                let mut providers = match frontend {
                    Frontend::Norm => cuda_vnext_tensor_bf16_providers(&runtime)?,
                    Frontend::Rotary => cuda_vnext_tensor_component_providers(&runtime)?,
                };
                providers.retain(|provider| {
                    let id = provider.descriptor().operation_id().as_str();
                    id == HEAD_RMS_NORM_BF16_OPERATION_ID
                        || (frontend == Frontend::Rotary && id == ADJACENT_ROPE_BF16_OPERATION_ID)
                });
                providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
                let mut contracts: Vec<Box<dyn OperationContract>> = vec![
                    Box::new(dense_linear_bf16_contract()?),
                    Box::new(head_rms_norm_bf16_contract()?),
                ];
                if frontend == Frontend::Rotary {
                    contracts.push(Box::new(adjacent_rope_bf16_contract()?));
                }
                let registry = OperationRuntimeRegistry::new(contracts, providers)?;
                let materializers = WeightMaterializerRegistry::identity_only()?;
                let descriptor = runtime.descriptor();
                let engine = EngineProviderDescriptor::new(
                    id("provider.engine.cuda.tensor.qk-norm")?,
                    ContractVersion::new(1, 0),
                    descriptor.runtime_implementation_fingerprint.clone(),
                    descriptor.id.clone(),
                    descriptor.capabilities.clone(),
                )?;
                let catalog = materializers.augment_catalog(
                    registry.capability_catalog(descriptor.clone(), vec![engine])?,
                )?;
                run(
                    config,
                    source,
                    frontend,
                    input,
                    frequencies,
                    runtime,
                    &registry,
                    &catalog,
                    &materializers,
                    frontend.cuda_profile_id(),
                    options.usable_memory_bytes,
                    options.transfer_staging_bytes,
                    json!({"backend":"CUDA", "allowances":options, "dense_profile":CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID, "head_rms_norm_profile":CUDA_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_ID, "adjacent_rope_profile":(frontend==Frontend::Rotary).then_some(CUDA_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID), "cublas_version":cuda_dense_bf16_cublas_version()?, "qualification":"sampled mathematical bounds and explicit rotary probes; no Torch/vLLM or Reference bitwise equality"}),
                    inspect,
                )
            }
        }
    }
}

fn run<R: DeviceRuntime>(
    config: &QwenImage21Config,
    source: &SafetensorsArchive,
    frontend: Frontend,
    input: &[u8],
    frequencies: Option<&[u8]>,
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
    profile: &str,
    capacity: u64,
    staging: usize,
    backend: Value,
    inspect: impl FnOnce(&[TensorExecutionOutput], &Value) -> Result<Value>,
) -> Result<(Value, Value)> {
    let family = TypedFamilyRegistration::new(Family::with_frontend(frontend)?)
        .prepare_with_profile(
            &serde_json::to_value(config)?,
            &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
        )?;
    let spec = ProgramTensorSpec {
        dimensions: vec![1, config.hidden_size()],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let mut input_specs = BTreeMap::from([(id(Family::INPUT_ID)?, spec)]);
    if frontend == Frontend::Rotary {
        input_specs.insert(
            id(Family::FREQUENCIES_ID)?,
            ProgramTensorSpec {
                dimensions: vec![1, config.head_dim() / 2, 2],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        );
    }
    let mut options = ProgramPlanCompileOptions::new(input_specs)?;
    for output in family.program().outputs() {
        options.retain_completion_value(output.clone());
    }
    options.require_weight_materializer(id(IDENTITY_WEIGHT_MATERIALIZER_ID)?);
    let policy = TensorPolicy {
        capacity_bytes: capacity,
        storage_profiles: runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let compiled = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        catalog,
        &policy,
        &registry.planning(),
        materializers,
        &options,
    )?;
    let executable = compiled.executable();
    let plan = executable.execution_plan();
    let output_bytes = usize::try_from(
        config
            .hidden_size()
            .checked_mul(u64::try_from(frontend.output_ids().len())?)
            .and_then(|n| n.checked_mul(2))
            .context("output bytes overflow")?,
    )?;
    let input_bytes = input
        .len()
        .checked_add(frequencies.map_or(0, <[u8]>::len))
        .context("input bytes overflow")?;
    let total = input_bytes
        .checked_add(output_bytes)
        .and_then(|n| n.checked_add(staging))
        .context("Host total overflows")?;
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        total,
        HostMemoryClassLimits {
            input_preprocess_bytes: input_bytes,
            output_bytes,
            transfer_staging_bytes: staging,
            weight_backing_bytes: 0,
        },
    )?);
    let weight_bytes =
        family
            .stored_weight_schema()?
            .components
            .iter()
            .try_fold(0_u64, |total, component| {
                total
                    .checked_add(component.physical_bytes()?)
                    .ok_or_else(|| VNextError::InvalidExecutionPlan {
                        reason: "Q/K norm weight bytes overflow".into(),
                    })
            })?;
    let resources = initialize_resources(
        runtime.clone(),
        &family,
        plan,
        source,
        weight_bytes,
        &host,
        staging,
    )?;
    let mut executor = PreparedTensorExecutor::new(
        executable.clone(),
        registry.bind_plan(executable)?,
        resources.clone(),
        host.clone(),
        TensorExecutorConfig::default(),
    )
    .map_err(|error| anyhow::anyhow!("Q/K norm executor: {error}"))?;
    let provenance = json!({"family":frontend.family_id(),"frontend":frontend, "profile":profile, "plan_hash":plan.plan_hash(), "runtime_fingerprint":runtime.descriptor().runtime_implementation_fingerprint, "backend":backend, "initialized_weight_bytes":weight_bytes, "managed_host_bytes":total, "managed_staging_bytes":staging, "managed_input_bytes":input_bytes,"managed_frequency_bytes":frequencies.map_or(0,<[u8]>::len), "managed_output_bytes":output_bytes, "nodes":plan.payload().nodes().iter().map(|node| json!({"node":node.id(), "operation":node.operation_id(), "provider":node.provider_resources().provider_id(), "implementation_fingerprint":node.provider_implementation_fingerprint()})).collect::<Vec<_>>()});
    let mut managed = host.allocate_zeroed(HostMemoryClass::InputPreprocess, input.len())?;
    managed.as_mut().copy_from_slice(input);
    let tensor = ManagedHostTensor::new(
        id(Family::INPUT_ID)?,
        ResolvedTensorSpec::new(
            vec![1, config.hidden_size()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?,
        managed.freeze(),
    )?;
    let mut tensors = vec![tensor];
    if let Some(bytes) = frequencies {
        let mut managed = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())?;
        managed.as_mut().copy_from_slice(bytes);
        tensors.push(ManagedHostTensor::new(
            id(Family::FREQUENCIES_ID)?,
            ResolvedTensorSpec::new(
                vec![1, config.head_dim() / 2, 2],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            managed.freeze(),
        )?);
    }
    let outputs = executor
        .execute(
            id("run.qwen-image21.block0-qk-norm.basis0")?,
            id("request.qwen-image21.block0-qk-norm.basis0")?,
            tensors,
        )
        .map_err(|error| anyhow::anyhow!("Q/K norm execution: {error}"))?;
    let report = inspect(&outputs, &provenance)?;
    drop(outputs);
    drop(executor);
    let released = match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt.released_static_resources(),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            bail!("Q/K norm close retains {strong_count} owners")
        }
        Err(error) => bail!("Q/K norm close failed: {:?}", error.failure()),
    };
    let usage = host.usage()?;
    if usage.committed_bytes() != 0 || usage.allocated_bytes() != 0 {
        bail!("Q/K norm Host owners remain after close");
    }
    Ok((
        report,
        json!({"released_static_resources":released, "managed_host_committed_bytes":usage.committed_bytes(), "managed_host_allocated_bytes":usage.allocated_bytes(), "scope":"plan close and Host account; not CUDA opaque memory qualification or process RSS"}),
    ))
}
