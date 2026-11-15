use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_tensor_runtime_enable_cutlass_nvfp4,
        cuda_vnext_nvfp4_bf16_providers, CutlassNvfp4Config,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).unwrap()
}
struct Counted {
    inner: FastH3AttentionProjectionSource,
    reads: AtomicUsize,
}
impl WeightComponentSource for Counted {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}

#[test]
#[ignore = "requires SM120 CUDA host and locked NVFP4 artifact; exact dyadic subset only"]
fn actual_attention_package_executes_shared_quantization_four_projections_and_closes() {
    let (_directory, package) = dyadic::fixture();
    let account = HostMemoryAccount::new(
        HostMemoryLimits::new(
            16 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 16 << 20,
                transfer_staging_bytes: 16 << 20,
                weight_backing_bytes: 16 << 20,
            },
        )
        .unwrap(),
    );
    let (config, source) = FastH3AttentionProjectionSource::prepare(
        package.clone(),
        0,
        129,
        &account,
        projection_limits(),
    )
    .unwrap();
    let prepared = TypedFamilyRegistration::new(
        FastH3AttentionProjectionFamily::new(projection_limits()).unwrap(),
    )
    .prepare_with_profile(
        &serde_json::to_value(&config).unwrap(),
        &NumericalProfileId::new(FastH3AttentionProjectionFamily::PROFILE_ID).unwrap(),
    )
    .unwrap();
    let mut device = cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap();
    cuda_tensor_runtime_enable_cutlass_nvfp4(&mut device).unwrap();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device,
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "attention-projection-qualification".into(),
        })
        .unwrap(),
    );
    let registry = Arc::new(
        OperationRuntimeRegistry::new(
            vec![
                Box::new(nvfp4_quantize_bf16_contract().unwrap()),
                Box::new(nvfp4_matmul_bf16_contract().unwrap()),
            ],
            cuda_vnext_nvfp4_bf16_providers(
                &runtime,
                &CutlassNvfp4Config {
                    opaque_memory_bytes: 64 << 20,
                    opaque_memory_profile: "attention-projection-module-qualification".into(),
                },
            )
            .unwrap(),
        )
        .unwrap(),
    );
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.fast_h3.attention_projection.test"),
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
            &config.compile_options().unwrap(),
        )
        .unwrap(),
    );
    let source = Arc::new(Counted {
        inner: source,
        reads: AtomicUsize::new(0),
    });
    let x = dyadic::input(129);
    let mut bytes = account
        .allocate_zeroed(HostMemoryClass::InputPreprocess, x.len() * 2)
        .unwrap();
    for (target, value) in bytes.as_mut().chunks_exact_mut(2).zip(x) {
        target.copy_from_slice(&value.to_bits().to_le_bytes());
    }
    let input = ManagedHostTensor::new(
        id(FastH3AttentionProjectionFamily::INPUT_ID),
        ResolvedTensorSpec::new(
            vec![129, 128],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap();
    let phase = Arc::new((
        source.clone(),
        input.clone(),
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
                run_id: id("run.attention_projection.init"),
                request_id: id("request.attention_projection.init"),
                transaction_id: id("transaction.attention_projection.init"),
                static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
            phase.clone(),
        )
        .unwrap_or_else(|error| {
            std::mem::forget(error);
            panic!("attention initialization failed; original owner retained")
        });
    let expected = dyadic::expected(&package, 129);
    assert_eq!(source.reads.load(Ordering::Relaxed), 13);
    let mut retained = None;
    for trial in 0..2 {
        let outputs = match session.session_mut().execute(
            id(format!("run.attention_projection.{trial}")),
            id(format!("request.attention_projection.{trial}")),
            vec![input.clone()],
        ) {
            Ok(outputs) => outputs,
            Err(error) => {
                std::mem::forget((error, session, phase));
                panic!("attention execution failed; original owners retained")
            }
        };
        assert_eq!(outputs.len(), 4);
        for output in &outputs {
            assert_eq!(output.tensor().dimensions(), [129, 256]);
            let actual: Vec<_> = output
                .bytes()
                .bytes()
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(
                actual,
                expected[output.value_id().as_str()],
                "{}",
                output.value_id()
            );
        }
        assert_eq!(
            source.reads.load(Ordering::Relaxed),
            13,
            "weights must initialize only once"
        );
        retained = Some(outputs);
    }
    let (returned, _receipt) = session.close().unwrap_or_else(|error| {
        std::mem::forget(error);
        panic!("attention close failed; original owner retained")
    });
    drop(returned);
    drop(phase);
    drop(input);
    drop(compiled);
    drop(source);
    let retained = retained.unwrap();
    assert!(retained.iter().all(|output| account.owns(output.bytes())));
    assert!(account.usage().unwrap().committed_bytes() > 0);
    drop(retained);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    eprintln!("FastH3 Q/K/V/gate exact dyadic qualification: rows129 H128 output256, shared quantization, 13 source reads, 2 executions, checked close and Host retirement");
}
