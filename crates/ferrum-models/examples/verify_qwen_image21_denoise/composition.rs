use super::{
    arguments::Arguments,
    backend::Backend,
    execution::{self, ResultArtifact},
    inputs::{HostBudget, Inputs},
    shared::{id, StoredComponentRuntime},
};
use anyhow::Result;
use ferrum_diffusion::FlowMatchEulerSchedule;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::qwen_image21::{
    QwenImage21TransformerConfig, QwenImage21TransformerFamily as Family,
};
use ferrum_runtime::*;
use serde_json::{json, Value};

pub fn run(
    config: &QwenImage21TransformerConfig,
    source: &dyn WeightComponentSource,
    args: &Arguments,
    host: &HostMemoryAccount,
    budget: &HostBudget,
    inputs: Inputs,
    schedule: FlowMatchEulerSchedule,
    inspect: impl FnMut(&Value, &ManagedHostTensor) -> Result<()>,
) -> Result<ResultArtifact> {
    let profile = match &args.backend {
        Backend::Reference => Family::REFERENCE_PROFILE_ID,
        Backend::Cuda(_) => Family::CUDA_PROFILE_ID,
    };
    let family = TypedFamilyRegistration::new(Family::new()?).prepare_with_profile(
        &serde_json::to_value(config)?,
        &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
    )?;
    match &args.backend {
        Backend::Reference => {
            let c = ReferenceVNextComposition::create_with_capacity(
                id("device.reference.denoise")?,
                args.reference_memory_bytes,
            )?;
            execution::execute(
                config,
                &family,
                source,
                StoredComponentRuntime {
                    runtime: c.runtime().clone(),
                    registry: c.registry(),
                    catalog: c.catalog(),
                    materializers: c.weight_materializers(),
                    capacity_bytes: args.reference_memory_bytes,
                    staging_bytes: budget.staging,
                    provenance: json!({"backend":"Reference","capacity_bytes":args.reference_memory_bytes}),
                },
                host,
                budget,
                inputs,
                schedule,
                inspect,
            )
        }
        Backend::Cuda(options) => {
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (options, inputs, schedule, inspect);
                anyhow::bail!("CUDA requires --features cuda")
            }
            #[cfg(feature = "cuda")]
            {
                use ferrum_kernels::backend::cuda::{
                    vnext_ops::*,
                    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
                };
                let nodes = family
                    .program()
                    .blocks()
                    .iter()
                    .map(|b| b.nodes.len())
                    .sum::<usize>();
                anyhow::ensure!(
                    options.maximum_commands_per_submission >= nodes + 4,
                    "CUDA command limit must cover {} nodes plus four input uploads",
                    nodes
                );
                let runtime = std::sync::Arc::new(CudaDeviceRuntime::new_tensor_eager(
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
                use CudaTensorOperation as Op;
                let mut providers = cuda_vnext_tensor_providers(
                    &runtime,
                    &[
                        Op::Silu,
                        Op::SiluMul,
                        Op::Tanh,
                        Op::ModulationScale,
                        Op::GatedResidual,
                        Op::LayerNorm,
                        Op::HeadRmsNorm,
                        Op::AdjacentRope,
                        Op::StructuredAttention,
                        Op::ZeroCenteredRmsNorm,
                        Op::GeluTanh,
                        Op::RowGather,
                        Op::BatchedRowConcat,
                    ],
                )?;
                providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
                let registry = OperationRuntimeRegistry::new(
                    vec![
                        Box::new(dense_linear_bf16_contract()?),
                        Box::new(silu_bf16_contract()?),
                        Box::new(silu_mul_bf16_contract()?),
                        Box::new(tanh_bf16_contract()?),
                        Box::new(modulation_scale_bf16_contract()?),
                        Box::new(gated_residual_bf16_contract()?),
                        Box::new(layer_norm_bf16_contract()?),
                        Box::new(head_rms_norm_bf16_contract()?),
                        Box::new(adjacent_rope_bf16_contract()?),
                        Box::new(structured_attention_bf16_contract()?),
                        Box::new(zero_centered_rms_norm_bf16_contract()?),
                        Box::new(gelu_tanh_bf16_contract()?),
                        Box::new(row_gather_bf16_contract()?),
                        Box::new(batched_row_concat_bf16_contract()?),
                    ],
                    providers,
                )?;
                let materializers = WeightMaterializerRegistry::identity_only()?;
                let descriptor = runtime.descriptor();
                let engine = EngineProviderDescriptor::new(
                    id("provider.engine.cuda.denoise")?,
                    ContractVersion::new(1, 0),
                    descriptor.runtime_implementation_fingerprint.clone(),
                    descriptor.id.clone(),
                    descriptor.capabilities.clone(),
                )?;
                let catalog = materializers.augment_catalog(
                    registry.capability_catalog(descriptor.clone(), vec![engine])?,
                )?;
                let provenance = json!({"backend":"CUDA","allowances":options,"cublas_version":cuda_dense_bf16_cublas_version()?,
                    "dense_profile":CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID,"qualification":"execution smoke only; no full-DiT numerical or Torch parity proof"});
                execution::execute(
                    config,
                    &family,
                    source,
                    StoredComponentRuntime {
                        runtime,
                        registry: &registry,
                        catalog: &catalog,
                        materializers: &materializers,
                        capacity_bytes: options.usable_memory_bytes,
                        staging_bytes: budget.staging,
                        provenance,
                    },
                    host,
                    budget,
                    inputs,
                    schedule,
                    inspect,
                )
            }
        }
    }
}
