//! Real managed CUDA execution shared by block and complete-clip fixtures.
//! Fixtures explicitly select scalar or admitted cuDNN segmented attention.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentInitializationReceipt,
        ComponentWeightSource, TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_tensor_runtime_enable_cutlass_nvfp4,
        cuda_vnext_dense_bf16_provider, cuda_vnext_nvfp4_bf16_providers,
        cuda_vnext_tensor_providers, cuda_vnext_tensor_providers_with_cudnn_selection,
        CudaCudnnOperation, CudaTensorOperation, CutlassNvfp4Config,
    },
    vnext_runtime::{
        CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig, CudnnSdpaPlanPolicy,
    },
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
pub(super) fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
struct Counted {
    source: Arc<dyn WeightComponentSource>,
    reads: AtomicUsize,
}
impl WeightComponentSource for Counted {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.component(c)
    }
}
pub(super) fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            16 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 8 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    )
}
pub(super) fn execute_twice(
    prepared: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    source: Arc<dyn WeightComponentSource>,
    inputs: Vec<ManagedHostTensor>,
    account: &HostMemoryAccount,
    verify: impl Fn(&[TensorExecutionOutput]),
) -> Vec<TensorExecutionOutput> {
    execute_twice_with_attention(prepared, options, source, inputs, account, verify, false)
}

pub(super) fn execute_twice_with_cudnn(
    prepared: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    source: Arc<dyn WeightComponentSource>,
    inputs: Vec<ManagedHostTensor>,
    account: &HostMemoryAccount,
    verify: impl Fn(&[TensorExecutionOutput]),
) -> Vec<TensorExecutionOutput> {
    execute_twice_with_attention(prepared, options, source, inputs, account, verify, true)
}

fn execute_twice_with_attention(
    prepared: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    source: Arc<dyn WeightComponentSource>,
    inputs: Vec<ManagedHostTensor>,
    account: &HostMemoryAccount,
    verify: impl Fn(&[TensorExecutionOutput]),
    use_cudnn: bool,
) -> Vec<TensorExecutionOutput> {
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    let components = schema.components.len();
    let source_bytes: u64 = schema
        .components
        .iter()
        .map(|c| c.physical_bytes().unwrap())
        .sum();
    let mut device = cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap();
    cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device,
            usable_memory_bytes: if use_cudnn { 2 << 30 } else { 1 << 30 },
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 128,
            opaque_memory_profile: "video-decoder-qualification".into(),
        })
        .unwrap(),
    );
    let mut providers = cuda_vnext_nvfp4_bf16_providers(
        &runtime,
        &CutlassNvfp4Config {
            opaque_memory_bytes: 64 << 20,
            opaque_memory_profile: "video-decoder-nvfp4-qualification".into(),
        },
    )
    .unwrap();
    providers.push(cuda_vnext_dense_bf16_provider(&runtime).unwrap());
    let mut ordinary = vec![
        CudaTensorOperation::HeadRmsNormF32,
        CudaTensorOperation::HeadRmsNorm,
        CudaTensorOperation::CastF32ToBf16,
        CudaTensorOperation::CastBf16ToF32,
        CudaTensorOperation::ChannelBiasAddBf16,
        CudaTensorOperation::PartialHalfSplitRope,
        CudaTensorOperation::SegmentedAttention,
        CudaTensorOperation::ChannelScaledResidualSeparateF32,
        CudaTensorOperation::RowGather,
        CudaTensorOperation::PackedSwiGlu,
        CudaTensorOperation::AffineLinear,
        CudaTensorOperation::BatchedSuffixAppendF32,
        CudaTensorOperation::LayerNormF32,
    ];
    let tensor_providers = if use_cudnn {
        ordinary.retain(|op| *op != CudaTensorOperation::SegmentedAttention);
        cuda_vnext_tensor_providers_with_cudnn_selection(
            &runtime,
            &CudaCudnnConfig {
                library_path: std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY")
                    .expect("explicit trusted cuDNN library required")
                    .into(),
                dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCIES")
                    .map(|s| std::env::split_paths(&s).collect())
                    .unwrap_or_default(),
                opaque_memory_bytes: 512 << 20,
                workspace_limit_bytes: 256 << 20,
                opaque_memory_profile: "video-decoder-test.cudnn".into(),
            },
            Some(CudnnSdpaPlanPolicy {
                maximum_plans: 8,
                maximum_live_bindings: 32,
                opaque_memory_bytes: 128 << 20,
                workspace_limit_bytes: 256 << 20,
            }),
            &[CudaCudnnOperation::SegmentedAttentionBf16],
            &ordinary,
        )
    } else {
        cuda_vnext_tensor_providers(&runtime, &ordinary)
    };
    providers.extend(tensor_providers.unwrap());
    let registry = Arc::new(
        OperationRuntimeRegistry::new(
            vec![
                Box::new(head_rms_norm_f32_contract().unwrap()),
                Box::new(head_rms_norm_bf16_contract().unwrap()),
                Box::new(cast_f32_to_bf16_contract().unwrap()),
                Box::new(cast_bf16_to_f32_contract().unwrap()),
                Box::new(channel_bias_add_bf16_contract().unwrap()),
                Box::new(partial_half_split_rope_bf16_contract().unwrap()),
                Box::new(segmented_attention_bf16_contract().unwrap()),
                Box::new(channel_scaled_residual_separate_f32_contract().unwrap()),
                Box::new(row_gather_bf16_contract().unwrap()),
                Box::new(packed_swiglu_bf16_contract().unwrap()),
                Box::new(nvfp4_quantize_bf16_contract().unwrap()),
                Box::new(nvfp4_matmul_bf16_contract().unwrap()),
                Box::new(dense_linear_bf16_contract().unwrap()),
                Box::new(affine_linear_bf16_contract().unwrap()),
                Box::new(batched_suffix_append_f32_contract().unwrap()),
                Box::new(layer_norm_f32_contract().unwrap()),
            ],
            providers,
        )
        .unwrap(),
    );
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.video_decoder.test"),
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(descriptor.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            prepared,
            &registry,
            &catalog,
            &TensorComponentPolicy::new(
                descriptor.total_memory_bytes,
                descriptor
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            )
            .unwrap(),
            &materializers,
            &options,
        )
        .unwrap(),
    );
    let source = Arc::new(Counted {
        source,
        reads: AtomicUsize::new(0),
    });
    let phase = Arc::new((
        source.clone(),
        inputs.clone(),
        runtime.clone(),
        registry.clone(),
        compiled.clone(),
    ));
    let mut session = compiled
        .initialize_retained(
            runtime.clone(),
            account.clone(),
            ComponentWeightSource::Stored(source.as_ref()),
            ComponentInitializationConfig {
                run_id: id("run.video_decoder.init"),
                request_id: id("request.video_decoder.init"),
                transaction_id: id("transaction.video_decoder.init"),
                static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
            phase.clone(),
        )
        .unwrap_or_else(|e| {
            std::mem::forget(e);
            panic!("decoder initialization failed; owners retained")
        });
    let ComponentInitializationReceipt::Initialized { receipt } =
        session.session().initialization_receipt()
    else {
        panic!("static upload")
    };
    assert_eq!(receipt.uploaded_component_count(), components);
    assert_eq!(receipt.uploaded_bytes(), source_bytes);
    let mut retained = None;
    for trial in 0..2 {
        let outputs = match session.session_mut().execute(
            id(format!("run.video_decoder.{trial}")),
            id(format!("request.video_decoder.{trial}")),
            inputs.clone(),
        ) {
            Ok(outputs) => outputs,
            Err(e) => {
                std::mem::forget((e, session, phase));
                panic!("decoder execution failed; owners retained")
            }
        };
        verify(&outputs);
        assert_eq!(source.reads.load(Ordering::Relaxed), components);
        retained = Some(outputs);
    }
    let (returned, close) = session.close().unwrap_or_else(|e| {
        std::mem::forget(e);
        panic!("decoder close failed; owners retained")
    });
    assert!(close.released_static_resources() > 0);
    drop(returned);
    drop(phase);
    drop(inputs);
    drop(compiled);
    drop(source);
    eprintln!("Decoder managed CUDA: {components} source components, {source_bytes} uploaded bytes once, two executions; close completed with retained Host output");
    retained.unwrap()
}
