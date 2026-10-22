use super::{
    arguments::Arguments,
    backend::Backend,
    inputs::Inputs,
    shared::{execute_stored_component_with_options, id, ComponentInput, StoredComponentRuntime},
};
use anyhow::{bail, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::qwen_image21::{
    QwenImage21TransformerConfig, QwenImage21TransformerFamily as Family,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::{json, Value};

pub fn execute(
    config: &QwenImage21TransformerConfig,
    source: &SafetensorsArchive,
    args: &Arguments,
    payload: &Inputs,
    inspect: impl FnOnce(&[TensorExecutionOutput], &Value) -> Result<Value>,
) -> Result<(Value, Value)> {
    let specs = config.input_tensor_specs()?;
    let inputs = Family::INPUT_IDS
        .iter()
        .zip(&payload.bytes)
        .map(|(name, bytes)| {
            let value_id = id(*name)?;
            let spec = &specs[&value_id];
            Ok(ComponentInput {
                value_id,
                tensor: ResolvedTensorSpec::new(
                    spec.dimensions.clone(),
                    spec.element_type,
                    spec.layout.clone(),
                )?,
                bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let profile = match &args.backend {
        Backend::Reference => Family::REFERENCE_PROFILE_ID,
        Backend::Cuda(_) => Family::CUDA_PROFILE_ID,
    };
    let family = TypedFamilyRegistration::new(Family::new()?).prepare_with_profile(
        &serde_json::to_value(config)?,
        &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
    )?;
    // Includes seven attribute-derived transport shapes; do not reconstruct
    // compile options from the four inputs and silently discard those shapes.
    let compile_options = config.compile_options()?;
    match &args.backend {
        Backend::Reference => {
            let composition = ReferenceVNextComposition::create_with_capacity(
                id("device.reference.qwen_image21.transformer")?,
                args.reference_memory_bytes,
            )?;
            execute_stored_component_with_options(
                &family,
                source,
                StoredComponentRuntime {
                    runtime: composition.runtime().clone(),
                    registry: composition.registry(),
                    catalog: composition.catalog(),
                    materializers: composition.weight_materializers(),
                    capacity_bytes: args.reference_memory_bytes,
                    staging_bytes: 65536,
                    provenance: json!({"backend":"Reference","capacity_bytes":args.reference_memory_bytes,
                    "qualification":"execution and finite velocity only; no whole-transformer numerical qualification or Torch/vLLM parity"}),
                },
                inputs,
                compile_options,
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
                let node_count = family
                    .program()
                    .blocks()
                    .iter()
                    .map(|block| block.nodes.len())
                    .sum::<usize>();
                if options.maximum_commands_per_submission < node_count + inputs.len() {
                    bail!("CUDA command limit must cover at least {} graph commands and {} input uploads; configure --cuda-maximum-commands-per-submission",node_count,inputs.len());
                }
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
                    id("provider.engine.cuda.tensor.transformer")?,
                    ContractVersion::new(1, 0),
                    descriptor.runtime_implementation_fingerprint.clone(),
                    descriptor.id.clone(),
                    descriptor.capabilities.clone(),
                )?;
                let catalog = materializers.augment_catalog(
                    registry.capability_catalog(descriptor.clone(), vec![engine])?,
                )?;
                execute_stored_component_with_options(
                    &family,
                    source,
                    StoredComponentRuntime {
                        runtime,
                        registry: &registry,
                        catalog: &catalog,
                        materializers: &materializers,
                        capacity_bytes: options.usable_memory_bytes,
                        staging_bytes: options.transfer_staging_bytes,
                        provenance: json!({"backend":"CUDA","allowances":options,"cublas_version":cuda_dense_bf16_cublas_version()?,
                        "dense_profile":CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID,
                        "qualification":"execution and finite velocity only; no whole-transformer numerical qualification or Torch/vLLM parity"}),
                    },
                    inputs,
                    compile_options,
                    inspect,
                )
            }
        }
    }
}
