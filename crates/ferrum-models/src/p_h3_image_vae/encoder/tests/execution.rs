use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use std::sync::Arc;
struct Diagnostic {
    inner: Family,
    capture: bool,
}
impl ModelFamilyProvider for Diagnostic {
    type Config = PH3ImageVaeEncoderConfig;
    fn family_id(&self) -> &ModelFamilyId {
        self.inner.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.inner.external_metadata_ids()
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        self.inner.parse_config(r)
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        self.inner.validate_config_identity(r, c)
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.inner.validated_external_metadata_id(r, c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.inner.weights(c)
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        self.inner.numerical_profiles(c)
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let graph = self.inner.semantic_program(c, p)?;
        if !self.capture {
            return Ok(graph);
        }
        let outputs = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| n.outputs.clone())
            .collect();
        ModelProgram::new(
            graph.family_id().clone(),
            graph.inputs().to_vec(),
            graph.blocks().to_vec(),
            graph.states().to_vec(),
            graph.weights().to_vec(),
            outputs,
        )?
        .with_input_tensor_specs(graph.input_tensor_specs().clone())
    }
    fn semantic_metadata(&self, c: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        self.inner.semantic_metadata(c)
    }
}
fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    c: &PH3ImageVaeEncoderConfig,
    source: &SafetensorsArchive,
    x: &[f32],
    profile: &str,
    capture: bool,
) -> Values {
    let prepared = TypedFamilyRegistration::new(Diagnostic {
        inner: Family::new(limits()).unwrap(),
        capture,
    })
    .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(profile))
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = materializers.augment_catalog(catalog.clone()).unwrap();
    let options = c.compile_options().unwrap();
    let d = runtime.descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared,
        registry,
        &catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &materializers,
        &options,
    )
    .unwrap();
    let host = host();
    let checkpoint = PH3ImageVaeEncoderCheckpoint::new(c, source, host.clone(), limits()).unwrap();
    let mut session = compiled
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(&checkpoint),
            ComponentInitializationConfig {
                request_id: id("request.full_encoder.init"),
                run_id: id("run.full_encoder.init"),
                transaction_id: id("transaction.full_encoder.init"),
                static_policy: StaticInitializationPolicy::new(16 << 20, 512).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap_or_else(|f| panic!("full encoder init: {f:?}"));
    let values = bytes(x.iter().copied());
    let mut backing = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len())
        .unwrap();
    backing.as_mut().copy_from_slice(&values);
    let input = ManagedHostTensor::new(
        id(Family::INPUT_ID),
        ResolvedTensorSpec::new(
            vec![
                c.batch(),
                c.architecture().in_channels(),
                c.height(),
                c.width(),
            ],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        backing.freeze(),
    )
    .unwrap();
    let outputs = session
        .execute(
            id("run.full_encoder.execute"),
            id("request.full_encoder.execute"),
            vec![input],
        )
        .unwrap_or_else(|f| panic!("full encoder execute: {f:?}"));
    let actual = outputs
        .iter()
        .map(|output| {
            assert_eq!(output.tensor().element_type(), ElementType::F32);
            assert_eq!(
                output.tensor().dimensions(),
                options.tensor_specs()[output.value_id()].dimensions
            );
            assert!(host.owns(output.bytes()));
            (
                output.value_id().clone(),
                output
                    .bytes()
                    .bytes()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            )
        })
        .collect();
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("full encoder close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert!(outputs.iter().all(|o| host.owns(o.bytes())));
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    actual
}
fn near(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let bound = 1e-4 * (1. + e.abs());
        assert!(
            a.is_finite() && (a - e).abs() <= bound,
            "{label} element{i}: {a} vs {e}, bound{bound}"
        );
    }
}
fn verify<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    profile: &str,
) {
    let c = config(2);
    let (_dir, source) = archive(&originals(&c));
    let x = input(&c, false);
    let actual = run(
        runtime.clone(),
        registry,
        catalog,
        &c,
        &source,
        &x,
        profile,
        true,
    );
    let wanted = oracle::capture(&c, &source, &x);
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        wanted.keys().collect::<Vec<_>>()
    );
    for (port, expected) in wanted {
        near(&actual[&port], &expected, port.as_str());
    }
    for sample in 0..2 {
        let one = config(1);
        let per = x.len() / 2;
        let baseline = run(
            runtime.clone(),
            registry,
            catalog,
            &one,
            &source,
            &x[sample * per..(sample + 1) * per],
            profile,
            true,
        );
        for (port, expected) in &baseline {
            let per = expected.len();
            near(
                &actual[port][sample * per..(sample + 1) * per],
                expected,
                "full CNN B2 vs independent B1",
            );
        }
    }
    let changed = run(
        runtime.clone(),
        registry,
        catalog,
        &c,
        &source,
        &input(&c, true),
        profile,
        true,
    );
    for (port, values) in &actual {
        let per = values.len() / 2;
        assert_eq!(
            &values[..per],
            &changed[port][..per],
            "request contamination at {port}"
        );
    }
    assert!(actual[&id::<ProgramValueId>(Family::OUTPUT_ID)]
        .iter()
        .zip(&changed[&id::<ProgramValueId>(Family::OUTPUT_ID)])
        .any(|(a, b)| a != b));
    let production = run(runtime, registry, catalog, &c, &source, &x, profile, false);
    assert_eq!(
        actual[&id::<ProgramValueId>(Family::OUTPUT_ID)],
        production[&id::<ProgramValueId>(Family::OUTPUT_ID)]
    );
}
#[test]
fn actual_stored_full_six_level_reference_encoder_matches_f64_oracle_and_request_isolation() {
    let backend = ReferenceVNextComposition::create(id("device.reference.full_encoder")).unwrap();
    verify(
        backend.runtime().clone(),
        backend.registry(),
        backend.catalog(),
        Family::REFERENCE_PROFILE_ID,
    );
    assert_eq!(backend.runtime().snapshot().live_allocations, 0);
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA and explicit FERRUM_TEST_CUDNN_LIBRARY_PATH"]
fn actual_stored_full_six_level_cuda_encoder_uses_f32_fma_cudnn_and_matches_oracle() {
    use ferrum_kernels::backend::cuda::{
        vnext_ops::{
            cuda_tensor_runtime_config, cuda_vnext_cnn_f32_providers, CudaTensorOperation,
        },
        vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
        CudaCudnnConfig,
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 256,
            opaque_memory_profile: "unqualified-full-encoder-f32-tiny-v1".into(),
        })
        .unwrap(),
    );
    let config = CudaCudnnConfig {
        library_path: std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
            .map(std::path::PathBuf::from)
            .expect("explicit trusted cuDNN9 library path required"),
        dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCY_LIBRARIES")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default(),
        opaque_memory_bytes: 64 << 20,
        workspace_limit_bytes: 16 << 20,
        opaque_memory_profile: "unqualified-full-encoder-cudnn9-fma-v1".into(),
    };
    let registry = OperationRuntimeRegistry::new(
        F32CnnOp::ALL
            .into_iter()
            .map(|kind| Box::new(kind.contract().unwrap()) as Box<dyn OperationContract>)
            .collect(),
        cuda_vnext_cnn_f32_providers(
            &runtime,
            &config,
            &[
                CudaTensorOperation::GroupNormSiluNchwF32,
                CudaTensorOperation::PlainResidualAddNchwF32,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.full_encoder.fixture"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(d.clone(), vec![engine])
        .unwrap();
    verify(runtime, &registry, &catalog, Family::CUDA_PROFILE_ID);
}
