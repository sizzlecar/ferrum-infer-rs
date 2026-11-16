use super::{load, Attention, Config, Failure};
use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_tensor_runtime_enable_cutlass_nvfp4,
        cuda_vnext_nvfp4_bf16_providers, cuda_vnext_tensor_providers,
        cuda_vnext_tensor_providers_with_cudnn_causal_gqa, CudaTensorOperation, CutlassNvfp4Config,
        CUDA_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_ID,
        CUDA_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_IDENTITY,
    },
    vnext_runtime::{
        CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig, CudnnSdpaPlanPolicy,
    },
};
use ferrum_models::qwen3_vl_text::Qwen3VLTextNvfp4Family;
use ferrum_runtime::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;
#[path = "cuda/forward.rs"]
mod forward;
fn id<T: TryFrom<String>>(value: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.to_owned()).expect("fixed qualification ID")
}
fn error(e: impl std::fmt::Display + 'static) -> Failure {
    Failure::retain("checkpoint/configuration", e, ())
}

pub(super) fn run(args: Config) -> Result<Value, Failure> {
    let (checkpoint, metadata, presentation) = load(&args).map_err(error)?;
    let config = checkpoint.config();
    let family = TypedFamilyRegistration::new(Qwen3VLTextNvfp4Family::new().map_err(error)?)
        .prepare_with_profile(
            &serde_json::to_value(config).map_err(error)?,
            &id(config.numerical_profile_id()),
        )
        .map_err(error)?;
    let ModelWeights::Stored(schema) = family.weights() else {
        return Err(error("encoder must have Stored weights"));
    };
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            args.host_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: args.host_bytes,
                transfer_staging_bytes: args.host_bytes,
                output_bytes: args.host_bytes,
                weight_backing_bytes: args.host_bytes,
            },
        )
        .map_err(error)?,
    );
    let source = checkpoint.managed_source(&host).map_err(error)?;
    let mut selected_bytes = 0u64;
    for c in &schema.components {
        let payload = source.component(c).map_err(error)?;
        selected_bytes = selected_bytes
            .checked_add(payload.bytes().len() as u64)
            .ok_or_else(|| error("selected source bytes overflow"))?;
    }
    let managed_source_backing_bytes = host.usage().map_err(error)?.allocated_bytes();
    drop(source);
    if host.usage().map_err(error)?.allocated_bytes() != 0 {
        return Err(error("source Host backing did not retire"));
    }
    let source_report = json!({"metadata_sha256":format!("{:x}",Sha256::digest(&metadata)),
        "declared_layers":config.declared_layers(),"exported_layers":config.exported_layers(),"executed_layers":config.executed_layers(),
        "hidden_size":config.text().architecture().hidden_size(),"sequence_length":config.text().layout().sequence_length(),
        "projection_count":config.projection_count(),"components":schema.components.len(),"selected_execution_weight_bytes":selected_bytes,
        "managed_source_backing_bytes":managed_source_backing_bytes,"source_host_bytes_after_drop":0,
        "scalar_profile":config.scalar_arithmetic_profile(),"family_fingerprint":family.fingerprint().map_err(error)?,
        "dense_values_scanned":false,"weight_payload_hashes_computed":false});
    let options = config.compile_options().map_err(error)?;
    let mut device =
        cuda_tensor_runtime_config(args.device, id(&format!("device.cuda.{}", args.device)))
            .map_err(error)?;
    cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device).map_err(error)?;
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device,
            usable_memory_bytes: args.device_bytes,
            bootstrap_memory_bytes: args.bootstrap_bytes,
            stream_memory_bytes: args.stream_bytes,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 4096,
            opaque_memory_profile: "explicit-qwen3-vl-nvfp4-compile-qualification-v1".into(),
        })
        .map_err(error)?,
    );
    let mut providers = cuda_vnext_nvfp4_bf16_providers(
        &runtime,
        &CutlassNvfp4Config {
            opaque_memory_bytes: args.native_bytes,
            opaque_memory_profile: "explicit-qwen3-vl-nvfp4-compile-native-v1".into(),
        },
    )
    .map_err(|e| Failure::retain("NVFP4 providers", e, runtime.clone()))?;
    let mut selected = vec![
        CudaTensorOperation::HeadRmsNorm,
        CudaTensorOperation::HalfSplitRope,
        CudaTensorOperation::PlainResidualAdd,
        CudaTensorOperation::SiluMul,
        CudaTensorOperation::RowScatterAdd,
    ];
    let attention_providers = match &args.attention {
        Attention::SequentialDiagnostic => {
            selected.push(CudaTensorOperation::StructuredGqaAttention);
            cuda_vnext_tensor_providers(&runtime, &selected)
        }
        Attention::CudnnCausal {
            library,
            dependencies,
            opaque_bytes,
            workspace_bytes,
            cache_bytes,
        } => cuda_vnext_tensor_providers_with_cudnn_causal_gqa(
            &runtime,
            &CudaCudnnConfig {
                library_path: library.clone(),
                dependency_libraries: dependencies.clone(),
                opaque_memory_bytes: *opaque_bytes,
                workspace_limit_bytes: *workspace_bytes,
                opaque_memory_profile: "explicit-qwen-causal-gqa-owner-v1".into(),
            },
            CudnnSdpaPlanPolicy {
                maximum_plans: 8,
                maximum_live_bindings: 128,
                opaque_memory_bytes: *cache_bytes,
                workspace_limit_bytes: *workspace_bytes,
            },
            &selected,
        ),
    }
    .map_err(|e| Failure::retain("text providers", e, runtime.clone()))?;
    providers.extend(attention_providers);
    let mut contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(nvfp4_quantize_bf16_contract().map_err(error)?),
        Box::new(nvfp4_matmul_bf16_contract().map_err(error)?),
        Box::new(head_rms_norm_bf16_contract().map_err(error)?),
        Box::new(half_split_rope_bf16_contract().map_err(error)?),
        Box::new(plain_residual_add_bf16_contract().map_err(error)?),
        Box::new(silu_mul_bf16_contract().map_err(error)?),
        Box::new(row_scatter_add_bf16_contract().map_err(error)?),
    ];
    match args.attention {
        Attention::SequentialDiagnostic => contracts.push(Box::new(
            structured_gqa_attention_bf16_contract().map_err(error)?,
        )),
        Attention::CudnnCausal { .. } => {
            contracts.push(Box::new(
                structured_gqa_attention_fused_bf16_contract().map_err(error)?,
            ));
            contracts.push(Box::new(
                structured_attention_fused_bf16_contract().map_err(error)?,
            ));
            contracts.push(Box::new(conv2d_bf16_contract().map_err(error)?));
        }
    }
    let registry = Arc::new(
        OperationRuntimeRegistry::new(contracts, providers)
            .map_err(|e| Failure::retain("registry", e, runtime.clone()))?,
    );
    let materializers = WeightMaterializerRegistry::identity_only().map_err(error)?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.qwen3_vl.compile"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .map_err(error)?;
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .map_err(error)?,
        )
        .map_err(error)?;
    let policy = TensorComponentPolicy::new(
        args.device_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .map_err(error)?;
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            family,
            &registry,
            &catalog,
            &policy,
            &materializers,
            &options,
        )
        .map_err(|e| {
            Failure::retain(
                "actual encoder plan compile",
                e,
                (runtime.clone(), registry.clone()),
            )
        })?,
    );
    let plan = compiled.executable().execution_plan();
    let mut report = json!({"schema":"ferrum.qwen3_vl.nvfp4.compile.v1","validation":"real checkpoint source and managed CUDA provider plan compilation",
        "forward_submitted":false,"weights_uploaded":false,"performance_claim":false,"source":source_report,
        "head_rms_profile":CUDA_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_ID,
        "attention":args.attention,"numerical_profile":config.numerical_profile_id(),
        "head_rms_numerical_identity":CUDA_HEAD_RMS_NORM_BF16_NUMERICAL_PROFILE_IDENTITY,
        "plan_hash":plan.plan_hash(),"static_bytes":plan.payload().memory().static_bytes(),
        "minimum_step_bytes":plan.payload().memory().minimum_step_bytes(),
        "nodes":plan.payload().nodes().iter().map(|n|json!({"provider":n.provider_resources().provider_id(),
            "implementation":n.provider_implementation_fingerprint()})).collect::<Vec<_>>(),
        "runtime_fingerprint":d.runtime_implementation_fingerprint});
    if args.forward.is_some() || args.image_forward.is_some() {
        let execution = forward::run(
            &args,
            presentation.as_ref(),
            &metadata,
            Arc::new(checkpoint),
            compiled,
            runtime,
            host,
        )?;
        report["forward_submitted"] = true.into();
        report["weights_uploaded"] = true.into();
        report["schema"] = "ferrum.qwen3_vl.nvfp4.forward.v1".into();
        report["validation"] = "real checkpoint weights, managed CUDA forward, finite BF16 features and closed session; no semantic quality or comparative performance claim".into();
        report["execution"] = execution;
    }
    Ok(report)
}
