use super::{
    backend::MetalOptions,
    execution,
    latent::Budget,
    shared::{id, StoredComponentRuntime},
};
use anyhow::Result;
use ferrum_execution::TensorExecutionOutput;
use ferrum_kernels::backend::metal::{
    vnext_ops::tensor::{
        metal_tensor_providers, metal_tensor_runtime_config, MetalTensorOperation as Op,
    },
    vnext_runtime::{MetalDeviceRuntime, MetalTensorRuntimeConfig},
};
use ferrum_models::qwen_image21::QwenImage21VaeDecoderConfig;
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::{json, Value};
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
pub fn run(
    runtime: Arc<MetalDeviceRuntime>,
    family: &PreparedModelFamily,
    config: &QwenImage21VaeDecoderConfig,
    input: ManagedHostTensor,
    host: &HostMemoryAccount,
    budget: &Budget,
    source: &SafetensorsArchive,
    weight_bytes: u64,
    options: &MetalOptions,
    inspect: impl FnOnce(&TensorExecutionOutput) -> Result<Value>,
) -> Result<(Value, Value, Value)> {
    let selected = [
        Op::Conv2d,
        Op::ChannelL2Norm,
        Op::ChannelL2NormImage,
        Op::SpatialAttention,
        Op::SpatialUpsampleGather,
        Op::SiluNchw,
        Op::PlainResidualAddNchw,
    ];
    let registry = OperationRuntimeRegistry::new(
        selected
            .iter()
            .map(|kind| Ok(Box::new(kind.contract()?) as Box<dyn OperationContract>))
            .collect::<Result<_>>()?,
        metal_tensor_providers(&runtime, &selected)?,
    )?;
    let materializers = WeightMaterializerRegistry::identity_only()?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.metal.vae.verify")?,
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )?;
    let catalog =
        materializers.augment_catalog(registry.capability_catalog(d.clone(), vec![engine])?)?;
    // Runtime validation has checked these sums; expose only storage to planning.
    let storage = options.usable_memory_bytes
        - options.bootstrap_memory_bytes
        - options.stream_memory_bytes * u64::try_from(options.maximum_streams)?;
    execution::execute(
        family,
        config,
        input,
        host,
        budget,
        source,
        weight_bytes,
        StoredComponentRuntime {
            runtime,
            registry: &registry,
            catalog: &catalog,
            materializers: &materializers,
            capacity_bytes: storage,
            staging_bytes: budget.staging,
            provenance: json!({"backend":"Metal","allowances":options,"storage_capacity_bytes":storage,"qualification":"complete decoder execution with declared resource allowances; one component timing, not image-generation throughput or cross-backend parity"}),
        },
        inspect,
    )
}

pub fn create(options: &MetalOptions) -> Result<Arc<MetalDeviceRuntime>> {
    Ok(Arc::new(MetalDeviceRuntime::new_tensor_eager(
        MetalTensorRuntimeConfig {
            device: metal_tensor_runtime_config(id("device.metal.0")?)?,
            usable_memory_bytes: options.usable_memory_bytes,
            bootstrap_memory_bytes: options.bootstrap_memory_bytes,
            stream_memory_bytes: options.stream_memory_bytes,
            maximum_streams: options.maximum_streams,
            maximum_in_flight_per_stream: options.maximum_in_flight_per_stream,
            maximum_commands_per_submission: options.maximum_commands_per_submission,
            opaque_memory_profile: "metal.vae.verify.declared.v1".into(),
        },
    )?))
}
