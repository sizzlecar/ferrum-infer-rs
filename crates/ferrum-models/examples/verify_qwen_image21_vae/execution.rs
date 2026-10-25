use super::{
    arguments::Arguments,
    backend::Backend,
    latent::Budget,
    shared::{self, id, StoredComponentRuntime, TensorPolicy},
};
use anyhow::{bail, ensure, Context, Result};
use ferrum_execution::{PreparedTensorExecutor, TensorExecutionOutput, TensorExecutorConfig};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::qwen_image21::{
    QwenImage21VaeDecoderConfig as Config, QwenImage21VaeDecoderFamily as Family,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::{json, Value};

#[allow(clippy::too_many_arguments)]
pub fn run(
    config: &Config,
    input: ManagedHostTensor,
    host: &HostMemoryAccount,
    budget: &Budget,
    source: &SafetensorsArchive,
    args: &Arguments,
    runtime: &super::backend::Runtime,
    weight_bytes: u64,
    inspect: impl FnOnce(&TensorExecutionOutput) -> Result<Value>,
) -> Result<(Value, Value, Value)> {
    let profile = match &args.backend {
        Backend::Reference => Family::REFERENCE_PROFILE_ID,
        Backend::Cuda(_) => Family::CUDA_PROFILE_ID,
        Backend::Metal(_) => Family::METAL_PROFILE_ID,
    };
    let family = TypedFamilyRegistration::new(Family::new()?).prepare_with_profile(
        &serde_json::to_value(config)?,
        &NumericalProfileId::new(profile).map_err(anyhow::Error::msg)?,
    )?;
    match &args.backend {
        Backend::Metal(options) => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                super::metal::run(
                    runtime
                        .metal
                        .as_ref()
                        .context("missing admitted Metal runtime")?
                        .clone(),
                    &family,
                    config,
                    input,
                    host,
                    budget,
                    source,
                    weight_bytes,
                    options,
                    inspect,
                )
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                let _ = (options, inspect, runtime);
                bail!("Metal requires macOS and --features metal")
            }
        }
        Backend::Reference => {
            let c = ReferenceVNextComposition::create_with_capacity(
                id("device.reference.qwen_image21.vae")?,
                args.reference_memory_bytes,
            )?;
            execute(
                &family,
                config,
                input,
                host,
                budget,
                source,
                weight_bytes,
                StoredComponentRuntime {
                    runtime: c.runtime().clone(),
                    registry: c.registry(),
                    catalog: c.catalog(),
                    materializers: c.weight_materializers(),
                    capacity_bytes: args.reference_memory_bytes,
                    staging_bytes: budget.staging,
                    provenance: json!({"backend":"Reference","capacity_bytes":args.reference_memory_bytes,"qualification":"complete decoder execution/finite smoke only; no Torch parity"}),
                },
                inspect,
            )
        }
        Backend::Cuda(options) => {
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (options, inspect);
                bail!("CUDA requires --features cuda")
            }
            #[cfg(feature = "cuda")]
            {
                use ferrum_kernels::backend::cuda::{
                    vnext_ops::*,
                    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
                    CudaCudnnConfig,
                };
                use CudaTensorOperation as Op;
                let declared = args.cudnn.as_ref().context("missing typed cuDNN options")?;
                let cudnn = CudaCudnnConfig {
                    library_path: declared.library.clone(),
                    dependency_libraries: Vec::new(),
                    opaque_memory_bytes: declared.opaque_memory_bytes,
                    workspace_limit_bytes: declared.workspace_bytes,
                    opaque_memory_profile: declared.memory_profile.clone(),
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
                        maximum_commands_per_submission: options.maximum_commands_per_submission,
                        opaque_memory_profile: options.opaque_memory_profile.clone(),
                    },
                )?);
                let providers = cuda_vnext_tensor_providers_with_cudnn(
                    &runtime,
                    &cudnn,
                    &[
                        Op::ChannelL2Norm,
                        Op::ChannelL2NormImage,
                        Op::SpatialAttention,
                        Op::SpatialUpsampleGather,
                        Op::SiluNchw,
                        Op::PlainResidualAddNchw,
                    ],
                )?;
                let registry = OperationRuntimeRegistry::new(
                    vec![
                        Box::new(conv2d_bf16_contract()?),
                        Box::new(channel_l2_norm_bf16_contract()?),
                        Box::new(channel_l2_norm_image_bf16_contract()?),
                        Box::new(spatial_attention_bf16_contract()?),
                        Box::new(spatial_upsample_gather_bf16_contract()?),
                        Box::new(silu_nchw_bf16_contract()?),
                        Box::new(plain_residual_add_nchw_bf16_contract()?),
                    ],
                    providers,
                )?;
                let materializers = WeightMaterializerRegistry::identity_only()?;
                let d = runtime.descriptor();
                let engine = EngineProviderDescriptor::new(
                    id("provider.engine.cuda.tensor.vae")?,
                    ContractVersion::new(1, 0),
                    d.runtime_implementation_fingerprint.clone(),
                    d.id.clone(),
                    d.capabilities.clone(),
                )?;
                let catalog = materializers
                    .augment_catalog(registry.capability_catalog(d.clone(), vec![engine])?)?;
                execute(
                    &family,
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
                        capacity_bytes: options.usable_memory_bytes,
                        staging_bytes: budget.staging,
                        provenance: json!({"backend":"CUDA","allowances":options,"cudnn":declared,"qualification":"declared experimental opaque allowances; complete decoder execution/finite smoke only, no memory-size qualification or Torch parity"}),
                    },
                    inspect,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute<R: DeviceRuntime>(
    family: &PreparedModelFamily,
    config: &Config,
    input: ManagedHostTensor,
    host: &HostMemoryAccount,
    budget: &Budget,
    source: &SafetensorsArchive,
    weight_bytes: u64,
    c: StoredComponentRuntime<'_, R>,
    inspect: impl FnOnce(&TensorExecutionOutput) -> Result<Value>,
) -> Result<(Value, Value, Value)> {
    ensure!(
        host.owns(input.bytes())
            && input.tensor() == &config.input_tensor()?
            && input.value_id().as_str() == Family::INPUT_ID,
        "managed latent differs from prepared account/port"
    );
    let mut options = config.compile_options()?;
    options.retain_completion_value(id(Family::OUTPUT_ID)?);
    let policy = TensorPolicy {
        capacity_bytes: c.capacity_bytes,
        storage_profiles: c
            .runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let compiled = ProgramPlanCompiler::compile_with_weight_materializers(
        family,
        c.catalog,
        &policy,
        &c.registry.planning(),
        c.materializers,
        &options,
    )?;
    let executable = compiled.executable();
    let plan = executable.execution_plan();
    let ports = plan
        .payload()
        .tensor_ports()
        .context("decoder plan lacks tensor ports")?;
    ensure!(
        ports.outputs().len() == 1
            && ports.outputs()[0].value_id().as_str() == Family::OUTPUT_ID
            && ports.outputs()[0].tensor() == &config.output_tensor()?
            && usize::try_from(ports.outputs()[0].tensor().minimum_storage_bytes()?)?
                == budget.output,
        "compiled output differs from admitted Host extent"
    );
    let initialization_started = std::time::Instant::now();
    let resources = shared::initialize_resources(
        c.runtime.clone(),
        family,
        plan,
        source,
        weight_bytes,
        host,
        budget.staging,
    )?;
    let initialization_us = initialization_started.elapsed().as_micros();
    let execution: Result<_> = (|| {
        let mut executor = PreparedTensorExecutor::new(
            executable.clone(),
            c.registry.bind_plan(executable)?,
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .map_err(|e| anyhow::anyhow!("decoder executor: {e}"))?;
        let mut provenance = json!({"family":family.family_id(),"profile":family.numerical_profile().id,"family_fingerprint":family.fingerprint()?,"canonical_config":family.canonical_config(),"plan_hash":plan.plan_hash(),"runtime_fingerprint":c.runtime.descriptor().runtime_implementation_fingerprint,"backend":c.provenance,"initialized_weight_bytes":weight_bytes,"managed_host_budget":budget,
            "nodes":plan.payload().nodes().iter().map(|n|json!({"node":n.id(),"operation":n.operation_id(),"provider":n.provider_resources().provider_id(),"implementation_fingerprint":n.provider_implementation_fingerprint()})).collect::<Vec<_>>()});
        let execution_started = std::time::Instant::now();
        let outputs = executor
            .execute(
                id("run.vae.decoder.verify")?,
                id("request.vae.decoder.verify")?,
                vec![input],
            )
            .map_err(|e| anyhow::anyhow!("decoder execution: {e}"))?;
        provenance["timing"] = json!({"initialization_us":initialization_us,"execution_us":execution_started.elapsed().as_micros(),"scope":"managed static initialization and one executor call including input/output transfers; excludes weight hashing, compilation, image encoding and resource retirement"});
        let [rgba] = outputs.as_slice() else {
            bail!("decoder returned unexpected output ports")
        };
        ensure!(
            rgba.value_id().as_str() == Family::OUTPUT_ID
                && rgba.tensor() == &config.output_tensor()?,
            "actual decoder output differs from exact spec"
        );
        Ok((inspect(rgba)?, provenance))
    })();
    let closed = match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => Ok(receipt.released_static_resources()),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => Err(anyhow::anyhow!(
            "decoder plan retains {strong_count} owners"
        )),
        Err(error) => Err(anyhow::anyhow!(
            "decoder retirement failed: {:?}",
            error.failure()
        )),
    };
    let cleanup: Result<Value> = (|| {
        let released = closed?;
        let usage = host.usage()?;
        ensure!(
            usage.committed_bytes() == 0 && usage.allocated_bytes() == 0,
            "decoder Host owners remain after retirement"
        );
        Ok(
            json!({"released_static_resources":released,"managed_host_committed_bytes":usage.committed_bytes(),"managed_host_allocated_bytes":usage.allocated_bytes()}),
        )
    })();
    match (execution, cleanup) {
        (Ok((result, provenance)), Ok(cleanup)) => Ok((result, provenance, cleanup)),
        (Err(error), Ok(_)) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => {
            Err(error.context(format!("retirement also failed: {cleanup:#}")))
        }
    }
}
