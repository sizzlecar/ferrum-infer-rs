use super::{
    backend::Backend,
    shared::{id, initialize_resources, TensorPolicy},
};
use anyhow::{bail, Context, Result};
use ferrum_execution::{PreparedTensorExecutor, TensorExecutionOutput, TensorExecutorConfig};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::qwen_image21::{QwenImage21Block0MlpFamily as Family, QwenImage21Config};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::Arc};

pub fn execute(
    config: &QwenImage21Config,
    source: &SafetensorsArchive,
    backend: &Backend,
    input: &[u8],
    inspect: impl FnOnce(&[TensorExecutionOutput], &Value) -> Result<Value>,
) -> Result<(Value, Value)> {
    match backend {
        Backend::Reference => {
            let composition =
                ReferenceVNextComposition::create(id("device.reference.qwen_image21.block0-mlp")?)?;
            run(
                config,
                source,
                input,
                composition.runtime().clone(),
                composition.registry(),
                composition.catalog(),
                composition.weight_materializers(),
                Family::REFERENCE_PROFILE_ID,
                composition.runtime().descriptor().total_memory_bytes,
                65536,
                json!({"backend":"Reference", "dense":"FP32 sequential; BF16 boundary", "silu_mul":"reference.libm.exp-f32; BF16 activation boundary then product"}),
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
                        opaque_memory_profile: options.opaque_memory_profile.clone(),
                    },
                )?);
                let mut providers = cuda_vnext_tensor_pointwise_providers(&runtime)?;
                providers.retain(|provider| {
                    provider.descriptor().operation_id().as_str() == SILU_MUL_BF16_OPERATION_ID
                });
                providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
                let registry = OperationRuntimeRegistry::new(
                    vec![
                        Box::new(dense_linear_bf16_contract()?),
                        Box::new(silu_mul_bf16_contract()?),
                    ],
                    providers,
                )?;
                let materializers = WeightMaterializerRegistry::identity_only()?;
                let descriptor = runtime.descriptor();
                let engine = EngineProviderDescriptor::new(
                    id("provider.engine.cuda.tensor.mlp")?,
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
                    input,
                    runtime,
                    &registry,
                    &catalog,
                    &materializers,
                    Family::CUDA_PROFILE_ID,
                    options.usable_memory_bytes,
                    options.transfer_staging_bytes,
                    json!({"backend":"CUDA", "allowances":options, "dense_profile":CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID, "silu_mul_profile":CUDA_SILU_MUL_BF16_NUMERICAL_PROFILE_ID, "cublas_version":cuda_dense_bf16_cublas_version()?, "qualification":"sampled mathematical bounds; no Torch/vLLM or Reference bitwise equality"}),
                    inspect,
                )
            }
        }
    }
}

fn run<R: DeviceRuntime>(
    config: &QwenImage21Config,
    source: &SafetensorsArchive,
    input: &[u8],
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
    let family = TypedFamilyRegistration::new(Family::new()?).prepare_with_profile(
        &serde_json::to_value(config)?,
        &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
    )?;
    let spec = ProgramTensorSpec {
        dimensions: vec![1, config.hidden_size()],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let mut options =
        ProgramPlanCompileOptions::new(BTreeMap::from([(id(Family::INPUT_ID)?, spec)]))?;
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
            .mlp_hidden_size()
            .checked_mul(3)
            .and_then(|n| n.checked_add(config.hidden_size()))
            .and_then(|n| n.checked_mul(2))
            .context("output bytes overflow")?,
    )?;
    let total = input
        .len()
        .checked_add(output_bytes)
        .and_then(|n| n.checked_add(staging))
        .context("Host total overflows")?;
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        total,
        HostMemoryClassLimits {
            input_preprocess_bytes: input.len(),
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
                        reason: "MLP weight bytes overflow".into(),
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
    .map_err(|error| anyhow::anyhow!("MLP executor: {error}"))?;
    let provenance = json!({"family":Family::FAMILY_ID, "profile":profile, "plan_hash":plan.plan_hash(), "runtime_fingerprint":runtime.descriptor().runtime_implementation_fingerprint, "backend":backend, "initialized_weight_bytes":weight_bytes, "managed_host_bytes":total, "managed_staging_bytes":staging, "managed_input_bytes":input.len(), "managed_output_bytes":output_bytes, "nodes":plan.payload().nodes().iter().map(|node| json!({"node":node.id(), "operation":node.operation_id(), "provider":node.provider_resources().provider_id(), "implementation_fingerprint":node.provider_implementation_fingerprint()})).collect::<Vec<_>>()});
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
    let outputs = executor
        .execute(
            id("run.qwen-image21.block0-mlp.basis0")?,
            id("request.qwen-image21.block0-mlp.basis0")?,
            vec![tensor],
        )
        .map_err(|error| anyhow::anyhow!("MLP execution: {error}"))?;
    let report = inspect(&outputs, &provenance)?;
    drop(outputs);
    drop(executor);
    let released = match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt.released_static_resources(),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            bail!("MLP close retains {strong_count} owners")
        }
        Err(error) => bail!("MLP close failed: {:?}", error.failure()),
    };
    let usage = host.usage()?;
    if usage.committed_bytes() != 0 || usage.allocated_bytes() != 0 {
        bail!("MLP Host owners remain after close");
    }
    Ok((
        report,
        json!({"released_static_resources":released, "managed_host_committed_bytes":usage.committed_bytes(), "managed_host_allocated_bytes":usage.allocated_bytes(), "scope":"plan close and Host account; not CUDA opaque memory qualification or process RSS"}),
    ))
}
