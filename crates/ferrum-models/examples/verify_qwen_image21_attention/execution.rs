use super::{
    backend::Backend,
    shared::{execute_stored_component, id, ComponentInput, StoredComponentRuntime},
};
use anyhow::{bail, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::qwen_image21::{
    QwenImage21AttentionConfig, QwenImage21Block0AttentionFamily as Family,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::{json, Value};

pub fn execute(
    config: &QwenImage21AttentionConfig,
    source: &SafetensorsArchive,
    backend: &Backend,
    input: &[u8],
    frequencies: &[u8],
    inspect: impl FnOnce(&[TensorExecutionOutput], &Value) -> Result<Value>,
) -> Result<(Value, Value)> {
    let transformer = config.transformer();
    let inputs = vec![
        ComponentInput {
            value_id: id(Family::INPUT_ID)?,
            tensor: ResolvedTensorSpec::new(
                vec![config.layout().rows(), transformer.hidden_size()],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?,
            bytes: input,
        },
        ComponentInput {
            value_id: id(Family::FREQUENCIES_ID)?,
            tensor: ResolvedTensorSpec::new(
                vec![
                    u64::from(config.layout().sequence_length()),
                    transformer.head_dim() / 2,
                    2,
                ],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            bytes: frequencies,
        },
    ];
    let profile = match backend {
        Backend::Reference => Family::REFERENCE_PROFILE_ID,
        Backend::Cuda(_) => Family::CUDA_PROFILE_ID,
    };
    let family = TypedFamilyRegistration::new(Family::new()?).prepare_with_profile(
        &serde_json::to_value(config)?,
        &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
    )?;
    match backend {
        Backend::Reference => {
            let composition = ReferenceVNextComposition::create(id(
                "device.reference.qwen_image21.block0-attention",
            )?)?;
            execute_stored_component(
                &family,
                source,
                StoredComponentRuntime {
                    runtime: composition.runtime().clone(),
                    registry: composition.registry(),
                    catalog: composition.catalog(),
                    materializers: composition.weight_materializers(),
                    capacity_bytes: composition.runtime().descriptor().total_memory_bytes,
                    staging_bytes: 65536,
                    provenance: json!({"backend":"Reference","qualification":"candidate-independent real-weight diagnostic; no Torch/vLLM parity","scale_f32_bits":config.effective_scale().to_bits()}),
                },
                inputs,
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
                        opaque_memory_profile: options.opaque_memory_profile.clone(),
                    },
                )?);
                let mut providers = cuda_vnext_tensor_attention_providers(&runtime)?;
                providers.retain(|provider| {
                    matches!(
                        provider.descriptor().operation_id().as_str(),
                        HEAD_RMS_NORM_BF16_OPERATION_ID
                            | ADJACENT_ROPE_BF16_OPERATION_ID
                            | STRUCTURED_ATTENTION_BF16_OPERATION_ID
                    )
                });
                providers.push(cuda_vnext_dense_bf16_provider(&runtime)?);
                let registry = OperationRuntimeRegistry::new(
                    vec![
                        Box::new(dense_linear_bf16_contract()?),
                        Box::new(head_rms_norm_bf16_contract()?),
                        Box::new(adjacent_rope_bf16_contract()?),
                        Box::new(structured_attention_bf16_contract()?),
                    ],
                    providers,
                )?;
                let materializers = WeightMaterializerRegistry::identity_only()?;
                let descriptor = runtime.descriptor();
                let engine = EngineProviderDescriptor::new(
                    id("provider.engine.cuda.tensor.attention")?,
                    ContractVersion::new(1, 0),
                    descriptor.runtime_implementation_fingerprint.clone(),
                    descriptor.id.clone(),
                    descriptor.capabilities.clone(),
                )?;
                let catalog = materializers.augment_catalog(
                    registry.capability_catalog(descriptor.clone(), vec![engine])?,
                )?;
                execute_stored_component(
                    &family,
                    source,
                    StoredComponentRuntime {
                        runtime,
                        registry: &registry,
                        catalog: &catalog,
                        materializers: &materializers,
                        capacity_bytes: options.usable_memory_bytes,
                        staging_bytes: options.transfer_staging_bytes,
                        provenance: json!({"backend":"CUDA","allowances":options,"dense_profile":CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID,"head_rms_norm_profile":CUDA_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_ID,"rope_profile":CUDA_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID,"attention_profile":CUDA_STRUCTURED_ATTENTION_BF16_NUMERICAL_PROFILE_ID,"cublas_version":cuda_dense_bf16_cublas_version()?,"scale_f32_bits":config.effective_scale().to_bits(),"qualification":"sampled candidate-independent mathematical envelope; not full-F32-exp qualification or Torch/vLLM/Reference parity"}),
                    },
                    inputs,
                    inspect,
                )
            }
        }
    }
}
