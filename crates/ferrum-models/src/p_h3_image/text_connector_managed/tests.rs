use super::*;
use crate::p_h3_image::{
    PH3ImageHiddenState, PH3ImageTextConnectorReference, PH3ImageTextConnectorWeights,
};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use std::{collections::BTreeMap, sync::Arc};

mod comfy;

type Parameters = BTreeMap<&'static str, (Vec<usize>, Vec<f32>)>;
fn parameters() -> Parameters {
    BTreeMap::from([
        ("input_mean", (vec![4], vec![1., 2., 3., 4.])),
        ("input_std", (vec![4], vec![1., 2., 1., 4.])),
        (
            "proj.weight",
            (vec![2, 4], vec![1., 2., 3., 4., -1., 0., 2., -2.]),
        ),
        ("output_mean", (vec![2], vec![10., -1.])),
        ("output_std", (vec![2], vec![2., 0.5])),
        ("sink_embedding", (vec![2], vec![99., -99.])),
    ])
}
fn config() -> PH3ImageTextConnectorConfig {
    PH3ImageTextConnectorConfig::from_hf_json(br#"{"_class_name":"PH3ImageTextConnector","hidden_state_indices":[2,1],"text_hidden_size":2,"out_channels":2}"#).unwrap()
}
fn limits() -> PH3ImageTextConnectorProjectionLimits {
    PH3ImageTextConnectorProjectionLimits {
        max_projection_bytes: 4096,
    }
}
struct ArchiveFixture {
    _directory: tempfile::TempDir,
    archive: SafetensorsArchive,
}
fn archive(parameters: &Parameters, bad_dtype: Option<&str>) -> ArchiveFixture {
    let directory = tempfile::tempdir().unwrap();
    let payloads: Vec<_> = parameters
        .iter()
        .map(|(&name, (shape, values))| {
            let dtype = if bad_dtype == Some(name) {
                safetensors::Dtype::BF16
            } else {
                safetensors::Dtype::F32
            };
            let bytes = if dtype == safetensors::Dtype::F32 {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            } else {
                values
                    .iter()
                    .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
                    .collect()
            };
            (name, shape.clone(), dtype, bytes)
        })
        .collect();
    let views = payloads
        .iter()
        .map(|(name, shape, dtype, bytes)| {
            (
                *name,
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    ArchiveFixture {
        archive: SafetensorsArchive::open(directory.path()).unwrap(),
        _directory: directory,
    }
}
fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn f32_values(tensor: &ManagedHostTensor) -> Vec<f32> {
    tensor
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}
fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}
fn state_spec(batch: u64, tokens: u64, hidden: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![batch * tokens, hidden],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

#[test]
fn normalized_input_is_owned_f32_in_declared_tap_order_with_zeroed_per_sequence_sinks() {
    let config = config();
    let parameters = parameters();
    let source = archive(&parameters, None);
    let checkpoint =
        PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).unwrap();
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 2, 3).unwrap();
    let spec = state_spec(2, 3, 2);
    let state_two = bf16_bytes(&[100., 200., 2., 6., 3., 2., -7., 8., 0., 2., 1., 4.]);
    let state_one = bf16_bytes(&[300., 400., 4., 12., 5., 4., 9., 10., 3., 8., 2., 0.]);
    let states = [
        PH3ImageBf16HiddenState::new(2, &spec, &state_two),
        PH3ImageBf16HiddenState::new(1, &spec, &state_one),
    ];
    let host = account(96);
    let normalized = adapter
        .managed_standardized_input(&host, &states, &checkpoint)
        .unwrap();
    assert_eq!(normalized.tensor().element_type(), ElementType::F32);
    assert_eq!(normalized.tensor().dimensions(), &[6, 4]);
    assert!(host.owns(normalized.bytes()));
    assert_eq!(
        f32_values(&normalized),
        [
            0., 0., 0., 0., 1., 2., 1., 2., 2., 0., 2., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0.,
            1., -1., -1.
        ]
    );
    assert_eq!(host.usage().unwrap().allocated_bytes(), 96);
    drop(normalized);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn output_affine_ignores_sink_projection_nonfinites_and_casts_only_at_final_boundary() {
    let config = config();
    let mut parameters = parameters();
    parameters.get_mut("sink_embedding").unwrap().1[0] = 1.00390625;
    let source = archive(&parameters, None);
    let checkpoint =
        PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).unwrap();
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 2, 2).unwrap();
    let spec = ResolvedTensorSpec::new(
        vec![4, 2],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let projected = [
        f32::NAN,
        f32::INFINITY,
        16.,
        -3.,
        f32::NEG_INFINITY,
        f32::NAN,
        3.,
        -1.,
    ];
    let bytes: Vec<_> = projected
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let host = account(128);
    let f32 = adapter
        .managed_conditioning_from_f32(
            &host,
            &spec,
            &bytes,
            &checkpoint,
            PH3ImageConditioningPrecision::F32,
        )
        .unwrap();
    assert_eq!(
        f32_values(&f32),
        [1.00390625, -99., 42., -2.5, 1.00390625, -99., 16., -1.5]
    );
    let bf16 = adapter
        .managed_conditioning_from_f32(
            &host,
            &spec,
            &bytes,
            &checkpoint,
            PH3ImageConditioningPrecision::Bf16,
        )
        .unwrap();
    let actual: Vec<_> = bf16
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
        .collect();
    assert_eq!(actual, [1., -99., 42., -2.5, 1., -99., 16., -1.5]);
    assert!(host.owns(f32.bytes()) && host.owns(bf16.bytes()));
    drop((f32, bf16));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn invalid_sources_states_and_host_capacity_do_not_publish_or_leak_allocations() {
    let config = config();
    for name in [
        "input_mean",
        "input_std",
        "proj.weight",
        "output_mean",
        "output_std",
        "sink_embedding",
    ] {
        let source = archive(&parameters(), Some(name));
        assert!(PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).is_err());
        let mut bad = parameters();
        bad.get_mut(name).unwrap().1[0] = f32::NAN;
        let source = archive(&bad, None);
        assert!(PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).is_err());
    }
    for value in [0., -1.] {
        let mut bad = parameters();
        bad.get_mut("input_std").unwrap().1[0] = value;
        let source = archive(&bad, None);
        assert!(PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).is_err());
    }
    let source = archive(&parameters(), None);
    assert!(PH3ImageTextConnectorCheckpoint::new(
        &config,
        &source.archive,
        PH3ImageTextConnectorProjectionLimits {
            max_projection_bytes: 31
        }
    )
    .is_err());
    let checkpoint =
        PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).unwrap();
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 1, 2).unwrap();
    let spec = state_spec(1, 2, 2);
    let hidden = bf16_bytes(&[1.; 4]);
    let states = [
        PH3ImageBf16HiddenState::new(2, &spec, &hidden),
        PH3ImageBf16HiddenState::new(1, &spec, &hidden),
    ];
    let host = account(31);
    assert!(adapter
        .managed_standardized_input(&host, &states, &checkpoint)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let host = account(1024);
    assert!(adapter
        .managed_standardized_input(&host, &states[..1], &checkpoint)
        .is_err());
    let mut nonfinite = hidden.clone();
    nonfinite[..2].copy_from_slice(&bf16::NAN.to_bits().to_le_bytes());
    assert!(adapter
        .managed_standardized_input(
            &host,
            &[
                PH3ImageBf16HiddenState::new(2, &spec, &nonfinite),
                states[1]
            ],
            &checkpoint
        )
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn invalid_projection_or_final_capacity_does_not_publish_partial_conditioning() {
    let config = config();
    let source = archive(&parameters(), None);
    let checkpoint =
        PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).unwrap();
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 1, 2).unwrap();
    let spec = ResolvedTensorSpec::new(
        vec![2, 2],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let valid = [0., 0., 1., 2.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let host = account(15);
    assert!(adapter
        .managed_conditioning_from_f32(
            &host,
            &spec,
            &valid,
            &checkpoint,
            PH3ImageConditioningPrecision::F32
        )
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let host = account(64);
    let mut invalid = valid.clone();
    invalid[12..16].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(adapter
        .managed_conditioning_from_f32(
            &host,
            &spec,
            &invalid,
            &checkpoint,
            PH3ImageConditioningPrecision::F32
        )
        .is_err());
    assert!(adapter
        .managed_conditioning_from_f32(
            &host,
            &spec,
            &valid[..12],
            &checkpoint,
            PH3ImageConditioningPrecision::F32
        )
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn stored_family_and_source_reject_shape_precision_capacity_and_unknown_metadata() {
    let config = config();
    let projection = PH3ImageTextConnectorProjectionConfig::from_connector(&config).unwrap();
    assert_eq!(projection.projection_bytes().unwrap(), 32);
    let family = Projection::new(PH3ImageTextConnectorProjectionLimits {
        max_projection_bytes: 32,
    })
    .unwrap();
    let definition = TypedFamilyRegistration::new(family)
        .define(&serde_json::to_value(&projection).unwrap())
        .unwrap();
    let schema = definition.stored_weight_schema().unwrap();
    assert_eq!(schema.components[0].external_names, ["proj.weight"]);
    assert_eq!(schema.components[0].dimensions, [2, 4]);
    assert_eq!(
        schema.components[0].physical_element_type(),
        ElementType::F32
    );
    let family = Projection::new(PH3ImageTextConnectorProjectionLimits {
        max_projection_bytes: 31,
    })
    .unwrap();
    assert!(TypedFamilyRegistration::new(family)
        .define(&serde_json::to_value(&projection).unwrap())
        .is_err());
    assert!(Projection::new(PH3ImageTextConnectorProjectionLimits {
        max_projection_bytes: 0
    })
    .is_err());
    assert!(PH3ImageTextConnectorProjectionConfig::new(u64::MAX, 2).is_err());
    assert!(PH3ImageTextConnectorProjectionConfig::new(0, 2).is_err());
    let mut parameters = parameters();
    parameters.get_mut("proj.weight").unwrap().0 = vec![4, 2];
    let source = archive(&parameters, None);
    assert!(PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).is_err());
    let family = Projection::new(limits()).unwrap();
    assert!(family
        .parse_config(&serde_json::json!({"input_dim":4,"output_dim":2,"downcast":true}))
        .is_err());
}

fn run_component<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    profile: &str,
) {
    let config = config();
    let parameters = parameters();
    let source = archive(&parameters, None);
    let checkpoint =
        PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).unwrap();
    let projection_config = checkpoint.projection_config().unwrap();
    let family = TypedFamilyRegistration::new(Projection::new(limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&projection_config).unwrap(),
            &NumericalProfileId::new(profile).unwrap(),
        )
        .unwrap();
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 2, 3).unwrap();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        ProgramValueId::new(Projection::INPUT_ID).unwrap(),
        projection_config.input_spec(6).unwrap(),
    )]))
    .unwrap();
    let policy = TensorComponentPolicy::new(
        runtime.descriptor().total_memory_bytes,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::new(vec![]).unwrap();
    let catalog = materializers.augment_catalog(catalog.clone()).unwrap();
    let component = CompiledTensorComponent::compile(
        family,
        registry,
        &catalog,
        &policy,
        &materializers,
        &options,
    )
    .unwrap();
    let host = account(8192);
    let mut session = component
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(checkpoint.archive()),
            ComponentInitializationConfig {
                run_id: RunId::new("run.connector.init").unwrap(),
                request_id: RequestIdentity::new("request.connector.init").unwrap(),
                transaction_id: TransactionId::new("transaction.connector.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(4096, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let two = [100., 200., 2., 6., 3., 2., -7., 8., 0., 2., 1., 4.];
    let one = [300., 400., 4., 12., 5., 4., 9., 10., 3., 8., 2., 0.];
    let two_bytes = bf16_bytes(&two);
    let one_bytes = bf16_bytes(&one);
    let spec = state_spec(2, 3, 2);
    let normalized = adapter
        .managed_standardized_input(
            &host,
            &[
                PH3ImageBf16HiddenState::new(2, &spec, &two_bytes),
                PH3ImageBf16HiddenState::new(1, &spec, &one_bytes),
            ],
            &checkpoint,
        )
        .unwrap();
    let projected = session
        .execute(
            RunId::new("run.connector.execute").unwrap(),
            RequestIdentity::new("request.connector.execute").unwrap(),
            vec![normalized],
        )
        .unwrap();
    let result = adapter
        .managed_conditioning(
            &host,
            &projected[0],
            &checkpoint,
            PH3ImageConditioningPrecision::F32,
        )
        .unwrap();
    let weights = PH3ImageTextConnectorWeights {
        input_mean: &parameters["input_mean"].1,
        input_std: &parameters["input_std"].1,
        projection: &parameters["proj.weight"].1,
        output_mean: &parameters["output_mean"].1,
        output_std: &parameters["output_std"].1,
        sink_embedding: &parameters["sink_embedding"].1,
    };
    let mut expected = [0.; 12];
    PH3ImageTextConnectorReference::new(&config, 2, 3)
        .unwrap()
        .write_f32(
            &[
                PH3ImageHiddenState {
                    index: 2,
                    values: &two,
                },
                PH3ImageHiddenState {
                    index: 1,
                    values: &one,
                },
            ],
            weights,
            &mut expected,
        )
        .unwrap();
    assert_eq!(f32_values(&result), expected);
    assert!(host.owns(result.bytes()) && host.owns(projected[0].bytes()));
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|failure| panic!("component close failed: {:?}", failure.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop((projected, result));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn stored_f32_projection_plus_managed_adapters_matches_complete_independent_reference() {
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.connector.component").unwrap(),
    )
    .unwrap();
    run_component(
        composition.runtime().clone(),
        composition.registry(),
        composition.catalog(),
        Projection::REFERENCE_PROFILE_ID,
    );
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}

#[test]
fn sink_normalization_overflow_is_irrelevant_but_original_inputs_are_still_checked() {
    let config = config();
    let mut parameters = parameters();
    parameters.get_mut("input_mean").unwrap().1.fill(-f32::MAX);
    parameters.get_mut("input_std").unwrap().1.fill(1.);
    parameters.get_mut("proj.weight").unwrap().1.fill(0.);
    let source = archive(&parameters, None);
    let checkpoint =
        PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).unwrap();
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 1, 2).unwrap();
    let spec = state_spec(1, 2, 2);
    let max = bf16::MAX.to_f32();
    let hidden = bf16_bytes(&[max, max, 0., 0.]);
    assert!((max - (-f32::MAX)).is_infinite());
    let host = account(128);
    let normalized = adapter
        .managed_standardized_input(
            &host,
            &[
                PH3ImageBf16HiddenState::new(2, &spec, &hidden),
                PH3ImageBf16HiddenState::new(1, &spec, &hidden),
            ],
            &checkpoint,
        )
        .unwrap();
    assert_eq!(&f32_values(&normalized)[..4], &[0.; 4]);
    let mut expected = [0.; 4];
    PH3ImageTextConnectorReference::new(&config, 1, 2)
        .unwrap()
        .write_f32(
            &[
                PH3ImageHiddenState {
                    index: 2,
                    values: &[max, max, 0., 0.],
                },
                PH3ImageHiddenState {
                    index: 1,
                    values: &[max, max, 0., 0.],
                },
            ],
            PH3ImageTextConnectorWeights {
                input_mean: &parameters["input_mean"].1,
                input_std: &parameters["input_std"].1,
                projection: &parameters["proj.weight"].1,
                output_mean: &parameters["output_mean"].1,
                output_std: &parameters["output_std"].1,
                sink_embedding: &parameters["sink_embedding"].1,
            },
            &mut expected,
        )
        .unwrap();
    let projected = [f32::NAN, f32::NAN, 0., 0.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let output_spec = ResolvedTensorSpec::new(
        vec![2, 2],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let output = adapter
        .managed_conditioning_from_f32(
            &host,
            &output_spec,
            &projected,
            &checkpoint,
            PH3ImageConditioningPrecision::F32,
        )
        .unwrap();
    assert_eq!(f32_values(&output), expected);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host with admitted fixture capacity"]
fn stored_cuda_f32_connector_component_matches_the_same_independent_reference() {
    use ferrum_kernels::backend::cuda::{
        vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_dense_f32_provider},
        vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, DeviceId::new("device.cuda.0").unwrap()).unwrap(),
            usable_memory_bytes: 2 << 30,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "connector-f32-operation-validation".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(dense_linear_f32_contract().unwrap())],
        vec![cuda_vnext_dense_f32_provider(&runtime).unwrap()],
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        ProviderId::new("provider.engine.cuda.connector.fixture").unwrap(),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(runtime.descriptor().clone(), vec![engine])
        .unwrap();
    run_component(runtime, &registry, &catalog, Projection::CUDA_PROFILE_ID);
}
