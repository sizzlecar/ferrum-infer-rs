use super::*;
use crate::p_h3_image::text_connector::PH3ImageTextConnectorAffineArithmetic as Affine;
use crate::p_h3_image::text_connector_projection::PH3ImageTextConnectorCheckpointLayout as Layout;

fn comfy_parameters() -> Parameters {
    // Exact six external names from the official merged TE header; shapes are
    // scaled down from [12800] / [5120] / [5120,12800] for real Stored execution.
    BTreeMap::from([
        (
            "text_connector.input_mean",
            (vec![4], vec![0., 0.25, 1., -0.5]),
        ),
        ("text_connector.input_std", (vec![4], vec![1., 2., 0.5, 4.])),
        (
            "text_connector.proj.weight",
            (vec![2, 4], vec![1., 0., 0., 0., 0., 0.5, -0.25, 1.]),
        ),
        ("text_connector.output_mean", (vec![2], vec![-1., 0.375])),
        (
            "text_connector.output_std",
            (vec![2], vec![f32::from_bits(0x3f7ffffe), 0.25]),
        ),
        (
            "text_connector.sink_embedding",
            (vec![2], vec![f32::from_bits(0x3f800001), -3.125]),
        ),
    ])
}
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn f32_spec(batch: u64, tokens: u64, hidden: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![batch, tokens, hidden],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn checkpoint<'a>(
    config: &'a PH3ImageTextConnectorConfig,
    source: &'a ArchiveFixture,
) -> PH3ImageTextConnectorCheckpoint<'a> {
    PH3ImageTextConnectorCheckpoint::new_with_layout(
        config,
        &source.archive,
        limits(),
        Layout::ComfyMergedTextEncoder,
    )
    .unwrap()
}
fn weights(parameters: &Parameters) -> PH3ImageTextConnectorWeights<'_> {
    PH3ImageTextConnectorWeights {
        input_mean: &parameters["text_connector.input_mean"].1,
        input_std: &parameters["text_connector.input_std"].1,
        projection: &parameters["text_connector.proj.weight"].1,
        output_mean: &parameters["text_connector.output_mean"].1,
        output_std: &parameters["text_connector.output_std"].1,
        sink_embedding: &parameters["text_connector.sink_embedding"].1,
    }
}

#[test]
fn checkpoint_layout_is_exact_and_default_projection_configuration_stays_canonical() {
    let config = config();
    let original = PH3ImageTextConnectorProjectionConfig::from_connector(&config).unwrap();
    assert_eq!(
        serde_json::to_value(&original).unwrap(),
        serde_json::json!({"input_dim":4,"output_dim":2})
    );
    assert_eq!(
        serde_json::from_value::<PH3ImageTextConnectorProjectionConfig>(
            serde_json::json!({"input_dim":4,"output_dim":2})
        )
        .unwrap(),
        original
    );
    let merged = original
        .clone()
        .with_checkpoint_layout(Layout::ComfyMergedTextEncoder);
    let raw = serde_json::to_value(&merged).unwrap();
    assert_eq!(raw["checkpoint_layout"], "comfy_merged_text_encoder");
    assert_eq!(
        serde_json::from_value::<PH3ImageTextConnectorProjectionConfig>(raw.clone()).unwrap(),
        merged
    );
    let definition = TypedFamilyRegistration::new(Projection::new(limits()).unwrap())
        .define(&raw)
        .unwrap();
    assert_eq!(
        definition.stored_weight_schema().unwrap().components[0].external_names,
        ["text_connector.proj.weight"]
    );

    let source = archive(&comfy_parameters(), None);
    assert!(PH3ImageTextConnectorCheckpoint::new(&config, &source.archive, limits()).is_err());
    assert_eq!(
        checkpoint(&config, &source).projection_config().unwrap(),
        merged
    );
    let separate = archive(&parameters(), None);
    assert!(PH3ImageTextConnectorCheckpoint::new_with_layout(
        &config,
        &separate.archive,
        limits(),
        Layout::ComfyMergedTextEncoder
    )
    .is_err());
    // No alias/strip-prefix fallback for any of the six required parameters.
    for name in [
        "text_connector.input_mean",
        "text_connector.input_std",
        "text_connector.proj.weight",
        "text_connector.output_mean",
        "text_connector.output_std",
        "text_connector.sink_embedding",
    ] {
        let mut bad = comfy_parameters();
        bad.remove(name);
        let source = archive(&bad, None);
        assert!(PH3ImageTextConnectorCheckpoint::new_with_layout(
            &config,
            &source.archive,
            limits(),
            Layout::ComfyMergedTextEncoder
        )
        .is_err());
        let source = archive(&comfy_parameters(), Some(name));
        assert!(PH3ImageTextConnectorCheckpoint::new_with_layout(
            &config,
            &source.archive,
            limits(),
            Layout::ComfyMergedTextEncoder
        )
        .is_err());
    }
}

fn compile_projection(
    composition: &ReferenceVNextComposition,
    projection: &PH3ImageTextConnectorProjectionConfig,
    rows: u64,
) -> CompiledTensorComponent<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    let family = TypedFamilyRegistration::new(Projection::new(limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(projection).unwrap(),
            &NumericalProfileId::new(Projection::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        ProgramValueId::new(Projection::INPUT_ID).unwrap(),
        projection.input_spec(rows).unwrap(),
    )]))
    .unwrap();
    let descriptor = composition.runtime().descriptor();
    let materializers = WeightMaterializerRegistry::new(vec![]).unwrap();
    let catalog = materializers
        .augment_catalog(composition.catalog().clone())
        .unwrap();
    CompiledTensorComponent::compile(
        family,
        composition.registry(),
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
    .unwrap()
}

#[test]
fn merged_stored_projection_preserves_native_f32_taps_and_matches_full_two_request_oracle() {
    let config = config();
    let mut parameters = comfy_parameters();
    // A plausible unprefixed decoy must never supply the merged projection.
    parameters.insert("proj.weight", (vec![2, 4], vec![0.; 8]));
    let source = archive(&parameters, None);
    let checkpoint = checkpoint(&config, &source);
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 2, 3)
        .unwrap()
        .with_output_affine(Affine::FusedAddcmulF32);
    let two = [
        100.,
        200.,
        f32::from_bits(0x3f800001),
        1.0001,
        2.0003,
        -3.0002,
        -7.,
        8.,
        -0.10003,
        f32::from_bits(0x40490fd0),
        4.00001,
        -0.00012345,
    ];
    let one = [
        300., 400., 4.0001, 12.0003, 5.0002, 4.0001, 9., 10., 3.0003, 8.0001, 2.0007, -0.00011,
    ];
    let two_bytes = f32_bytes(&two);
    let one_bytes = f32_bytes(&one);
    let spec = f32_spec(2, 3, 2);
    let host = account(8192);
    let native = adapter
        .managed_standardized_f32_input(
            &host,
            &[
                PH3ImageF32HiddenState::new(2, &spec, &two_bytes),
                PH3ImageF32HiddenState::new(1, &spec, &one_bytes),
            ],
            &checkpoint,
        )
        .unwrap();
    assert_eq!(f32_values(&native)[4].to_bits(), two[2].to_bits());
    let rounded_two = bf16_bytes(&two);
    let rounded_one = bf16_bytes(&one);
    let bf16_spec = state_spec(2, 3, 2);
    let promoted = adapter
        .managed_standardized_input(
            &host,
            &[
                PH3ImageBf16HiddenState::new(2, &bf16_spec, &rounded_two),
                PH3ImageBf16HiddenState::new(1, &bf16_spec, &rounded_one),
            ],
            &checkpoint,
        )
        .unwrap();
    assert_ne!(native.bytes().bytes(), promoted.bytes().bytes());
    drop(promoted);

    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.comfy.connector").unwrap(),
    )
    .unwrap();
    let projection_config = checkpoint.projection_config().unwrap();
    let component = compile_projection(&composition, &projection_config, 6);
    let mut session = component
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(checkpoint.archive()),
            ComponentInitializationConfig {
                run_id: RunId::new("run.comfy.connector.init").unwrap(),
                request_id: RequestIdentity::new("request.comfy.connector.init").unwrap(),
                transaction_id: TransactionId::new("transaction.comfy.connector.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(4096, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let projected = session
        .execute(
            RunId::new("run.comfy.connector.execute").unwrap(),
            RequestIdentity::new("request.comfy.connector.execute").unwrap(),
            vec![native],
        )
        .unwrap();
    let ports = component
        .executable()
        .execution_plan()
        .payload()
        .tensor_ports()
        .unwrap();
    let view = PH3ImageF32HiddenState::from_output(2, &ports.outputs()[0], &projected[0]).unwrap();
    assert_eq!(view.bytes.as_ptr(), projected[0].bytes().bytes().as_ptr());
    assert_eq!(view.spec, projected[0].tensor());
    assert!(PH3ImageF32HiddenState::from_output(2, &ports.inputs()[0], &projected[0]).is_err());
    let other_shape = compile_projection(&composition, &projection_config, 3);
    let other_port = &other_shape
        .executable()
        .execution_plan()
        .payload()
        .tensor_ports()
        .unwrap()
        .outputs()[0];
    assert!(PH3ImageF32HiddenState::from_output(2, other_port, &projected[0]).is_err());
    let mut wrong_dtype = serde_json::to_value(&ports.outputs()[0]).unwrap();
    wrong_dtype["tensor"] = serde_json::to_value(
        ResolvedTensorSpec::new(
            vec![6, 2],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
    )
    .unwrap();
    let wrong_dtype: CompiledTensorPort = serde_json::from_value(wrong_dtype).unwrap();
    assert!(PH3ImageF32HiddenState::from_output(2, &wrong_dtype, &projected[0]).is_err());

    let result = adapter
        .managed_conditioning(
            &host,
            &projected[0],
            &checkpoint,
            PH3ImageConditioningPrecision::F32,
        )
        .unwrap();
    let mut expected = [0.; 12];
    PH3ImageTextConnectorReference::new(&config, 2, 3)
        .unwrap()
        .with_output_affine(Affine::FusedAddcmulF32)
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
            weights(&parameters),
            &mut expected,
        )
        .unwrap();
    assert_eq!(f32_values(&result), expected);
    // The F32 micro-difference survives the actual Stored GEMM and final FMA.
    assert_eq!(expected[2].to_bits(), (-2.0_f32.powi(-46)).to_bits());
    assert_eq!(
        &expected[..2],
        &parameters["text_connector.sink_embedding"].1
    );
    assert_eq!(
        &expected[6..8],
        &parameters["text_connector.sink_embedding"].1
    );
    let mut separate = [0.; 12];
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
            weights(&parameters),
            &mut separate,
        )
        .unwrap();
    assert_eq!(separate[2], 0.);
    assert_ne!(expected[2].to_bits(), separate[2].to_bits());
    let bf16 = adapter
        .managed_conditioning(
            &host,
            &projected[0],
            &checkpoint,
            PH3ImageConditioningPrecision::Bf16,
        )
        .unwrap();
    let rounded = expected
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(bf16.bytes().bytes(), rounded);
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|failure| panic!("{:?}", failure.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop((projected, result, bf16));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}

#[test]
fn native_f32_validation_and_host_rejection_leave_previous_owner_and_credit_unchanged() {
    let config = config();
    let source = archive(&comfy_parameters(), None);
    let checkpoint = checkpoint(&config, &source);
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 2, 2).unwrap();
    let spec = f32_spec(2, 2, 2);
    let bytes = f32_bytes(&[1.0001; 8]);
    let states = [
        PH3ImageF32HiddenState::new(2, &spec, &bytes),
        PH3ImageF32HiddenState::new(1, &spec, &bytes),
    ];
    let host = account(64);
    let first = adapter
        .managed_standardized_f32_input(&host, &states, &checkpoint)
        .unwrap();
    let previous = first.bytes().bytes().to_vec();
    assert!(adapter
        .managed_standardized_f32_input(&host, &states, &checkpoint)
        .is_err());
    assert_eq!(first.bytes().bytes(), previous);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 64);
    drop(first);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let mut bad = bytes.clone();
    bad[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        adapter
            .managed_standardized_f32_input(
                &host,
                &[PH3ImageF32HiddenState::new(2, &spec, &bad), states[1]],
                &checkpoint
            )
            .is_err(),
        "even a replaced sink must retain finite original hidden values"
    );
    let wrong_dtype = state_spec(2, 2, 2);
    assert!(adapter
        .managed_standardized_f32_input(
            &host,
            &[
                PH3ImageF32HiddenState::new(2, &wrong_dtype, &bytes),
                states[1]
            ],
            &checkpoint
        )
        .is_err());
    assert!(adapter
        .managed_standardized_f32_input(&host, &[states[1], states[0]], &checkpoint)
        .is_err());
    assert!(adapter
        .managed_standardized_f32_input(
            &host,
            &[
                PH3ImageF32HiddenState::new(2, &spec, &bytes[..28]),
                states[1]
            ],
            &checkpoint
        )
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(
        adapter
            .managed_standardized_f32_input(&host, &states, &checkpoint)
            .unwrap(),
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn fused_output_affine_has_its_own_finite_and_final_precision_boundary() {
    let config = config();
    let mut parameters = comfy_parameters();
    parameters.get_mut("text_connector.output_std").unwrap().1[0] = 2.;
    parameters.get_mut("text_connector.output_mean").unwrap().1[0] = -f32::MAX;
    let source = archive(&parameters, None);
    let checkpoint = checkpoint(&config, &source);
    let adapter = PH3ImageTextConnectorManagedAdapter::new(&config, 1, 2).unwrap();
    assert_eq!(adapter.output_affine(), Affine::SeparateMulAddF32);
    let fused = PH3ImageTextConnectorManagedAdapter::new(&config, 1, 2)
        .unwrap()
        .with_output_affine(Affine::FusedAddcmulF32);
    let spec = ResolvedTensorSpec::new(
        vec![2, 2],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let projection = f32_bytes(&[f32::NAN, f32::INFINITY, f32::MAX, 1.]);
    let host = account(32);
    assert!(adapter
        .managed_conditioning_from_f32(
            &host,
            &spec,
            &projection,
            &checkpoint,
            PH3ImageConditioningPrecision::F32
        )
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    // Single-round FMA cancels the overflowing intermediate product.
    let result = fused
        .managed_conditioning_from_f32(
            &host,
            &spec,
            &projection,
            &checkpoint,
            PH3ImageConditioningPrecision::F32,
        )
        .unwrap();
    assert_eq!(f32_values(&result)[2], f32::MAX);
    assert!(fused
        .managed_conditioning_from_f32(
            &host,
            &spec,
            &projection,
            &checkpoint,
            PH3ImageConditioningPrecision::Bf16
        )
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 16);
    assert_eq!(f32_values(&result)[2], f32::MAX);
    drop(result);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
