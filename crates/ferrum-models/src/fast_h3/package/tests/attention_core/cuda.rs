use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use std::sync::atomic::{AtomicUsize, Ordering};
fn ident<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).unwrap()
}
struct Counted {
    inner: FastH3AttentionCoreSource,
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
#[ignore = "requires SM120, nine-export Sage and NVFP4 locked artifacts plus explicit cuDNN; constant-V zero-QK subset <=2 BF16 ULP, not general model quality"]
fn attention_core_cuda_runs_prefix_sparse_compression_and_projection_in_one_session() {
    let account = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            128 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64 << 20,
                output_bytes: 32 << 20,
                transfer_staging_bytes: 16 << 20,
                weight_backing_bytes: 16 << 20,
            },
        )
        .unwrap(),
        CudaDeviceRuntime::pinned_weight_upload_alignment().unwrap(),
    )
    .unwrap();
    let (_directory, package) = dyadic::fixture();
    let layout = layout();
    let bounds = capacities();
    let (config, source, regions) = FastH3AttentionCoreSource::prepare(
        package.clone(),
        0,
        &layout,
        2,
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::VendorFused,
        &account,
        bounds,
    )
    .unwrap();
    let prepared = TypedFamilyRegistration::new(F::new(bounds).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::try_from(config.profile_id().to_owned()).unwrap(),
        )
        .unwrap();
    let source_count = match prepared.weights() {
        ModelWeights::Stored(s) => s.components.len(),
        _ => panic!("Stored"),
    };
    let (runtime, registry) = super::cuda_factory::create(regions.clone(), &[], vec![]);
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        ident("provider.engine.attention_core"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            prepared,
            &registry,
            &catalog,
            &TensorComponentPolicy::new(
                d.total_memory_bytes,
                d.dynamic_storage_profiles.iter().copied().collect(),
            )
            .unwrap(),
            &materializers,
            &config.compile_options(bounds).unwrap(),
        )
        .unwrap(),
    );
    let source = Arc::new(Counted {
        inner: source,
        reads: AtomicUsize::new(0),
    });
    let inputs = dyadic::inputs(&account, &config, &layout);
    let expected = dyadic::expected(&package);
    let owners = Arc::new((
        source.clone(),
        inputs.clone(),
        runtime.clone(),
        registry.clone(),
        compiled.clone(),
        regions.clone(),
    ));
    let mut session = compiled
        .initialize_retained(
            runtime.clone(),
            account.clone(),
            ComponentWeightSource::Stored(source.as_ref()),
            ComponentInitializationConfig {
                run_id: ident("run.attention_core.init"),
                request_id: ident("request.attention_core.init"),
                transaction_id: ident("transaction.attention_core.init"),
                static_policy: StaticInitializationPolicy::new(8192, 32).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
            owners.clone(),
        )
        .unwrap_or_else(|error| {
            let diagnostic = format!("{error:?}");
            std::mem::forget(error);
            panic!("initialization retains original owners: {diagnostic}")
        });
    assert_eq!(source.reads.load(Ordering::Relaxed), source_count);
    let mut retained: Option<Vec<ferrum_execution::TensorExecutionOutput>> = None;
    let rows = layout.packed().sequence_rows();
    let mut maximum_ulp = 0u16;
    for trial in 0..2 {
        let outputs = match session.session_mut().execute(
            ident(format!("run.attention_core.{trial}")),
            ident(format!("request.attention_core.{trial}")),
            inputs.clone(),
        ) {
            Ok(outputs) => outputs,
            Err(error) => {
                let diagnostic = format!("{error:?}");
                std::mem::forget((error, session, owners));
                panic!("execution retains original owners: {diagnostic}");
            }
        };
        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
        assert_eq!(output.tensor().dimensions(), [2 * rows as u64, 128]);
        for (index, bytes) in output.bytes().bytes().chunks_exact(2).enumerate() {
            let actual = u16::from_le_bytes(bytes.try_into().unwrap());
            let row = index / 128;
            let wanted = expected[row / rows * 3 + row % rows % 3][index % 128];
            assert_ne!(actual & 0x7f80, 0x7f80, "nonfinite at {index}");
            assert_eq!(actual >> 15, wanted >> 15, "sign at {index}");
            let ulp = (actual & 0x7fff).abs_diff(wanted & 0x7fff);
            maximum_ulp = maximum_ulp.max(ulp);
            assert!(ulp<=2,"full core constant-V subset at {index}: {actual:04x}, expected {wanted:04x}, ULP={ulp}");
        }
        if let Some(previous) = &retained {
            assert_eq!(output.bytes().bytes(), previous[0].bytes().bytes());
        }
        assert_eq!(source.reads.load(Ordering::Relaxed), source_count);
        retained = Some(outputs);
    }
    let (returned, receipt) = session.close().unwrap_or_else(|error| {
        let diagnostic = format!("{error:?}");
        std::mem::forget(error);
        panic!("close retains original owners: {diagnostic}")
    });
    drop(receipt);
    drop(returned);
    drop(owners);
    drop(inputs);
    drop(compiled);
    drop(source);
    drop(registry);
    drop(regions);
    let retained = retained.unwrap();
    assert!(retained.iter().all(|o| account.owns(o.bytes())));
    assert!(account.usage().unwrap().allocated_bytes() > 0);
    drop(retained);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    eprintln!("FastH3 complete attention core: B2 H2 D128 packed_rows={rows}, prefix+shared scores+Sage quant/dynamic+two head packs+Untile/Expand+raw gate+NVFP4 projection; source_reads={source_count}, 2 runs, max_BF16_ULP={maximum_ulp}, closed/Host retired; constant-V Q=K=0 subset only");
}
