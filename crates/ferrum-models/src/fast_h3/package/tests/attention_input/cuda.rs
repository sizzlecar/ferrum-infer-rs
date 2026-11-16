use super::*;
use ferrum_execution::{
    component::{
        ComponentInitializationConfig, ComponentInitializationReceipt, ComponentWeightSource,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_tensor_runtime_enable_cutlass_nvfp4,
        cuda_vnext_nvfp4_bf16_providers, cuda_vnext_tensor_providers, CudaTensorOperation,
        CutlassNvfp4Config,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Counted {
    inner: FastH3AttentionInputSource,
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
#[ignore = "requires locked SM120 NVFP4 and tensor modules; exact dyadic subset, not arbitrary model parity"]
fn actual_attention_input_package_runs_tiled_qkv_packed_gate_qk_rope_and_pools_on_device() {
    let (_directory, package) = dyadic::fixture();
    let layout = layout();
    let account = host(32 << 20);
    let (config, source) =
        FastH3AttentionInputSource::prepare(package.clone(), 0, &layout, 2, &account, capacities())
            .unwrap();
    let prepared = TypedFamilyRegistration::new(F::new(capacities()).unwrap())
        .prepare_with_profile(&serde_json::to_value(&config).unwrap(), &id(F::PROFILE_ID))
        .unwrap();
    let ModelWeights::Stored(schema) = prepared.weights() else {
        panic!("Stored")
    };
    let component_count = schema.components.len();
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
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "attention-input-qualification".into(),
        })
        .unwrap(),
    );
    let mut providers = cuda_vnext_nvfp4_bf16_providers(
        &runtime,
        &CutlassNvfp4Config {
            opaque_memory_bytes: 64 << 20,
            opaque_memory_profile: "attention-input-module-qualification".into(),
        },
    )
    .unwrap();
    providers.extend(
        cuda_vnext_tensor_providers(
            &runtime,
            &[
                CudaTensorOperation::RowIndexGather,
                CudaTensorOperation::MergeLeadingBatchView,
                CudaTensorOperation::SplitLeadingBatchView,
                CudaTensorOperation::HeadRmsPartialRopeSingleRound,
                CudaTensorOperation::MaskedTileMean,
            ],
        )
        .unwrap(),
    );
    let registry = Arc::new(
        OperationRuntimeRegistry::new(
            vec![
                Box::new(row_index_gather_or_zero_bf16_contract().unwrap()),
                Box::new(LeadingBatchBf16Kind::Merge.view_contract().unwrap()),
                Box::new(LeadingBatchBf16Kind::Split.view_contract().unwrap()),
                Box::new(head_rms_partial_rope_single_round_bf16_contract().unwrap()),
                Box::new(masked_tile_mean_bf16_f32_contract().unwrap()),
                Box::new(nvfp4_quantize_bf16_contract().unwrap()),
                Box::new(nvfp4_matmul_bf16_contract().unwrap()),
            ],
            providers,
        )
        .unwrap(),
    );
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.fast_h3.attention_input.test"),
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
            &config.compile_options(capacities()).unwrap(),
        )
        .unwrap(),
    );
    let source = Arc::new(Counted {
        inner: source,
        reads: AtomicUsize::new(0),
    });
    let inputs = [F::HIDDEN_ID, F::COSINE_ID, F::SINE_ID]
        .into_iter()
        .zip(dyadic::inputs(&layout, 2))
        .enumerate()
        .map(|(index, (name, values))| {
            let mut bytes = account
                .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
                .unwrap();
            for (target, bits) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
                target.copy_from_slice(&bits.to_le_bytes());
            }
            ManagedHostTensor::new(
                id(name),
                ResolvedTensorSpec::new(
                    vec![2, config.packed_rows(), if index == 0 { 128 } else { 96 }],
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
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
                run_id: id("run.attention_input.init"),
                request_id: id("request.attention_input.init"),
                transaction_id: id("transaction.attention_input.init"),
                static_policy: StaticInitializationPolicy::new(4096, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
            phase.clone(),
        )
        .unwrap_or_else(|error| {
            std::mem::forget(error);
            panic!("attention input initialization failed; original owner retained")
        });
    let ComponentInitializationReceipt::Initialized { receipt } =
        session.session().initialization_receipt()
    else {
        panic!("real static source upload")
    };
    assert_eq!(receipt.uploaded_component_count(), component_count);
    assert_eq!(receipt.uploaded_bytes(), source_bytes);
    assert_eq!(source.reads.load(Ordering::Relaxed), component_count);
    let expected = dyadic::expected(&package, &layout, 2);
    let mut retained = None;
    for trial in 0..2 {
        let outputs = match session.session_mut().execute(
            id(format!("run.attention_input.{trial}")),
            id(format!("request.attention_input.{trial}")),
            inputs.clone(),
        ) {
            Ok(v) => v,
            Err(error) => {
                std::mem::forget((error, session, phase));
                panic!("attention input execution failed; original owners retained")
            }
        };
        assert_eq!(outputs.len(), 7);
        for output in &outputs {
            let bytes = output.bytes().bytes();
            assert_eq!(
                bytes,
                expected[output.value_id().as_str()],
                "independent FP64/analytic oracle {}",
                output.value_id()
            );
        }
        assert_eq!(
            source.reads.load(Ordering::Relaxed),
            component_count,
            "one initialization for both executions"
        );
        retained = Some(outputs);
    }
    let (returned, close) = session.close().unwrap_or_else(|error| {
        std::mem::forget(error);
        panic!("attention input close failed; original owners retained")
    });
    assert!(close.released_static_resources() > 0);
    drop(returned);
    drop(phase);
    drop(inputs);
    drop(compiled);
    drop(source);
    let retained = retained.unwrap();
    assert!(retained.iter().all(|v| account.owns(v.bytes())));
    assert!(account.usage().unwrap().committed_bytes() > 0);
    drop(retained);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    eprintln!("FastH3 attention-input exact-domain qualification: B2 H128 heads2 D128 rot96, packed_rows={} tiled_rows={}, 22 managed device nodes, shared map/count/unit, {component_count} source reads and {source_bytes} uploaded bytes once, 2 executions, checked close and final Host retirement; no arbitrary-checkpoint or Torch reduction parity claim", config.packed_rows(), config.tiled_rows());
}
