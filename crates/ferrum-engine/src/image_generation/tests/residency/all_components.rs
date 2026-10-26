use super::*;
use ferrum_kernels::backend::reference::reference_vnext_operation_registry;
use ferrum_models::qwen3_vl_text::{Qwen3VLTextKeyRun, Qwen3VLTextLayout};

fn pipeline(
    name: &str,
) -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    let (original, directory) = numeric_pipeline();
    // Each capacity test has a separate device identity, so filling its shared
    // ledger cannot interfere with other simultaneously running Rust tests.
    let composition = composition(name);
    let mut pipeline = QwenImage21Pipeline::new(
        composition.runtime().clone(),
        original.host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        original.assets,
        original.profiles,
        original.config,
    )
    .unwrap();
    pipeline.config.residency = ImageResidencyPolicy::AllModelComponents;
    (pipeline, directory)
}

fn text_stage(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    sequence: u32,
) -> (
    PreparedStage<ReferenceDeviceRuntime>,
    Vec<ManagedHostTensor>,
) {
    let config = Qwen3VLTextConfig::new(
        text(),
        Qwen3VLTextLayout::new(sequence, vec![vec![Qwen3VLTextKeyRun::new(sequence, true)]])
            .unwrap(),
    )
    .unwrap();
    let family = TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(Qwen3VLTextEncoderFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let inputs = config
        .input_tensor_specs()
        .unwrap()
        .into_iter()
        .map(|(id, spec)| {
            let tensor =
                ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap();
            let mut bytes = pipeline
                .host
                .allocate_zeroed(
                    HostMemoryClass::InputPreprocess,
                    usize::try_from(tensor.minimum_storage_bytes().unwrap()).unwrap(),
                )
                .unwrap();
            for (i, element) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                let value = if id.as_str() == Qwen3VLTextEncoderFamily::EMBEDDINGS_ID {
                    bf16::from_f32((i + 1) as f32).to_bits()
                } else if id.as_str() == Qwen3VLTextEncoderFamily::COSINE_ID {
                    0x3f80
                } else {
                    0
                };
                element.copy_from_slice(&value.to_le_bytes());
            }
            ManagedHostTensor::new(id, tensor, bytes.freeze()).unwrap()
        })
        .collect();
    let stage = PreparedStage::compile(
        ImageStage::TextEncoder,
        1,
        family,
        &config.compile_options().unwrap(),
        pipeline,
    )
    .unwrap();
    (stage, inputs)
}

#[test]
fn all_components_reuses_real_text_session_then_attaches_changed_sequence_without_upload() {
    let (mut pipeline, _directory) = pipeline("device.image.residency.all.text");
    let mut stages = Vec::new();
    let mut outputs = Vec::new();
    for (job, sequence) in [1, 1, 2].into_iter().enumerate() {
        let (stage, inputs) = text_stage(&pipeline, sequence);
        let input_bytes = inputs
            .iter()
            .map(|input| input.bytes().len() as u64)
            .sum::<u64>();
        let before = pipeline.runtime.snapshot();
        let active = pipeline
            .start_stage(stage, job as u64, &mut stages)
            .unwrap_or_else(|_| panic!("text acquire"));
        assert_eq!(active.receipt.session_reused, job == 1);
        assert_eq!(active.receipt.weights_reused, job > 0);
        if job > 0 {
            assert_eq!(
                pipeline.runtime.snapshot().uploaded_bytes,
                before.uploaded_bytes
            );
        }
        let (active, result) = pipeline
            .execute_stage(active, job as u64, inputs)
            .unwrap_or_else(|_| panic!("real text forward"));
        // The fixture's learned weights are zero; both residuals preserve the
        // independently constructed nonzero embeddings through the real graph.
        let output = only_output(result, Qwen3VLTextEncoderFamily::OUTPUT_ID).unwrap();
        assert_eq!(output.tensor().dimensions(), [u64::from(sequence), 8]);
        for (i, bytes) in output.bytes().bytes().chunks_exact(2).enumerate() {
            assert_eq!(
                u16::from_le_bytes([bytes[0], bytes[1]]),
                bf16::from_f32((i + 1) as f32).to_bits()
            );
        }
        if job > 0 {
            assert_eq!(
                pipeline.runtime.snapshot().uploaded_bytes - before.uploaded_bytes,
                input_bytes
            );
        }
        pipeline
            .finish_stage(active, &mut stages)
            .unwrap_or_else(|_| panic!("text trim"));
        outputs.push(output);
    }
    let shutdown = pipeline.shutdown().unwrap();
    assert_eq!(shutdown.len(), 2);
    assert_eq!(shutdown[0].execution.plan.stage, ImageStage::TextEncoder);
    assert!(matches!(
        shutdown[0].retirement,
        ImageStageRetirement::Closed
    ));
    assert!(matches!(
        shutdown[1].retirement,
        ImageStageRetirement::SharedWeightsClosed { .. }
    ));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(
        pipeline.host.usage().unwrap().committed_bytes(),
        (1 + 1 + 2) * 8 * 2
    );
    assert_eq!(&outputs[0].bytes().bytes()[..2], &0x3f80_u16.to_le_bytes());
    drop(outputs);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn old_policies_keep_text_cold_and_new_policy_wire_is_explicit() {
    assert_eq!(ImageResidencyPolicy::default(), ImageResidencyPolicy::Cold);
    assert_eq!(
        serde_json::to_value(ImageResidencyPolicy::AllModelComponents).unwrap(),
        serde_json::json!("all_model_components")
    );
    assert_eq!(
        serde_json::from_value::<ImageResidencyPolicy>(serde_json::json!("all_model_components"))
            .unwrap(),
        ImageResidencyPolicy::AllModelComponents
    );
    for (index, policy) in [
        ImageResidencyPolicy::Cold,
        ImageResidencyPolicy::DenoiserAndDecoder,
    ]
    .into_iter()
    .enumerate()
    {
        let (mut pipeline, _directory) = pipeline(&format!("device.image.residency.old.{index}"));
        pipeline.config.residency = policy;
        let mut stages = Vec::new();
        let (stage, inputs) = text_stage(&pipeline, 1);
        let active = pipeline
            .start_stage(stage, 0, &mut stages)
            .unwrap_or_else(|_| panic!("cold text"));
        let (active, outputs) = active
            .execute(0, inputs)
            .unwrap_or_else(|_| panic!("cold forward"));
        pipeline
            .finish_stage(active, &mut stages)
            .unwrap_or_else(|_| panic!("cold retirement"));
        assert!(matches!(stages[0].retirement, ImageStageRetirement::Closed));
        assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
        assert!(pipeline.shutdown().unwrap().is_empty());
        drop(outputs);
        assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn text_retention_consumes_shared_capacity_and_shutdown_retires_all_sessions_before_weights() {
    let (mut pipeline, _directory) = pipeline("device.image.residency.all.capacity");
    let capacity = DeviceCapacityOwner::open(pipeline.runtime.descriptor(), DEVICE_BYTES).unwrap();
    let mut stages = Vec::new();
    let (stage, inputs) = text_stage(&pipeline, 1);
    drop(inputs);
    let text_static_bytes = stage.plan.static_bytes;
    let active = pipeline
        .start_stage(stage, 0, &mut stages)
        .unwrap_or_else(|_| panic!("text initialization"));
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("text residency"));
    let remaining = capacity
        .reserve(
            DeviceOpaqueMemoryKind::Bootstrap,
            DEVICE_BYTES - text_static_bytes,
        )
        .unwrap();
    assert!(capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1)
        .is_err());
    let before = pipeline.runtime.snapshot();
    let dit = prepared(&pipeline, 1);
    let failure = pipeline
        .start_stage(dit, 1, &mut stages)
        .err()
        .expect("TE bytes cannot be ignored");
    assert!(matches!(
        failure,
        ImageGenerationFailure::Initialization { .. }
    ));
    assert_eq!(pipeline.runtime.snapshot().allocations, before.allocations);
    assert_eq!(
        pipeline.runtime.snapshot().uploaded_bytes,
        before.uploaded_bytes
    );
    drop(failure);
    drop(remaining);

    let dit = prepared(&pipeline, 1);
    let active = pipeline
        .start_stage(dit, 2, &mut stages)
        .unwrap_or_else(|_| panic!("admit DiT after budget release"));
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("DiT residency"));
    let vae = decoder();
    let family = TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&vae).unwrap(),
            &NumericalProfileId::new(QwenImage21VaeDecoderFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let stage = PreparedStage::compile(
        ImageStage::Decoder,
        1,
        family,
        &vae.compile_options().unwrap(),
        &pipeline,
    )
    .unwrap();
    let active = pipeline
        .start_stage(stage, 3, &mut stages)
        .unwrap_or_else(|_| panic!("VAE initialization"));
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("VAE residency"));
    let receipts = pipeline.shutdown().unwrap();
    assert_eq!(receipts.len(), 6);
    for (i, stage) in [
        ImageStage::TextEncoder,
        ImageStage::Denoiser,
        ImageStage::Decoder,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(receipts[i].execution.plan.stage, stage);
        assert!(matches!(
            receipts[i].retirement,
            ImageStageRetirement::Closed
        ));
        assert_eq!(
            receipts[i]
                .close
                .as_ref()
                .unwrap()
                .released_static_resources(),
            0
        );
        assert_eq!(receipts[i + 3].execution.plan.stage, stage);
        assert!(matches!(
            receipts[i + 3].retirement,
            ImageStageRetirement::SharedWeightsClosed { .. }
        ));
    }
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    drop(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, DEVICE_BYTES)
            .unwrap(),
    );
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn text_execution_failure_and_snapshot_shutdown_preserve_recovery_owners() {
    let (mut pipeline, _directory) = pipeline("device.image.residency.all.error");
    let mut stages = Vec::new();
    let (stage, inputs) = text_stage(&pipeline, 1);
    drop(inputs);
    let active = pipeline
        .start_stage(stage, 0, &mut stages)
        .unwrap_or_else(|_| panic!("text initialization"));
    let ImageGenerationFailure::Execution { source, active } =
        active.execute(0, Vec::new()).err().unwrap()
    else {
        panic!("input error retains actual session")
    };
    let shutdown = pipeline.shutdown().unwrap_err();
    let ImageGenerationFailure::SharedWeightsReferenced {
        stage: ImageStage::TextEncoder,
        outcome: SharedComponentWeightsCloseOutcome::Referenced { weights },
    } = shutdown.failure
    else {
        panic!("snapshot cannot retire around the returned session")
    };
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    drop(source);
    active
        .close()
        .unwrap_or_else(|_| panic!("explicit quiescent recovery"));
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    assert!(matches!(
        weights.close(),
        Ok(SharedComponentWeightsCloseOutcome::Closed(_))
    ));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

fn capacity_stages(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
) -> [PreparedStage<ReferenceDeviceRuntime>; 3] {
    let (text, inputs) = text_stage(pipeline, 1);
    drop(inputs);
    let dit = prepared(pipeline, 1);
    let vae = decoder();
    let family = TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&vae).unwrap(),
            &NumericalProfileId::new(QwenImage21VaeDecoderFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let vae = PreparedStage::compile(
        ImageStage::Decoder,
        1,
        family,
        &vae.compile_options().unwrap(),
        pipeline,
    )
    .unwrap();
    [text, dit, vae]
}

#[test]
fn resident_preflight_uses_physical_plan_bytes_and_one_native_allowance_at_exact_boundary() {
    let (pipeline, _directory) = pipeline("device.image.residency.all.preflight");
    // Pure planning evidence: no native backend or opaque allocation is made.
    let opaque = 256;
    let mut pipeline = pipeline.with_device_opaque_allowance(opaque).unwrap();
    pipeline.config.plan_cache = ImagePlanCachePolicy::Disabled;
    let plans = capacity_stages(&pipeline);
    let memories = plans.each_ref().map(|plan| {
        plan.component
            .executable()
            .execution_plan()
            .payload()
            .memory()
    });
    let static_sum: u64 = memories.iter().map(|memory| memory.static_bytes()).sum();
    let dynamic_peak = memories
        .iter()
        .map(|memory| memory.minimum_runnable_request_bytes())
        .max()
        .unwrap();
    let exact = static_sum + dynamic_peak + opaque;
    for memory in memories {
        assert!(memory.static_bytes() + memory.minimum_runnable_request_bytes() < exact - 1);
    }
    drop(plans);

    for (capacity, accepted) in [(exact - 1, false), (exact, true)] {
        pipeline.config.device_capacity_bytes = capacity;
        pipeline.policy = TensorComponentPolicy::new(
            capacity,
            pipeline
                .runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();
        // Every component really compiles under this ceiling. Only their
        // simultaneous retained static installations make the lower case fail.
        let plans = capacity_stages(&pipeline);
        let before = pipeline.runtime.snapshot();
        let result = pipeline.validate_resident_capacity(&plans.iter().collect::<Vec<_>>());
        assert_eq!(result.is_ok(), accepted, "{result:?}");
        assert!(!pipeline.is_stopped());
        let after = pipeline.runtime.snapshot();
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.submissions, before.submissions);
        assert_eq!(after.live_allocations, 0);
        assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn cold_preflight_counts_one_active_stage_and_native_allowance_before_execution() {
    let (pipeline, _directory) = pipeline("device.image.residency.cold.preflight");
    let opaque = 256;
    let mut pipeline = pipeline.with_device_opaque_allowance(opaque).unwrap();
    pipeline.config.residency = ImageResidencyPolicy::Cold;
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
    pipeline.config.edit_prefix_policy = ImageEditPrefixPolicy::Recompute;
    pipeline.config.plan_cache = ImagePlanCachePolicy::Disabled;
    let plans = capacity_stages(&pipeline);
    let memories = plans.each_ref().map(|plan| {
        plan.component
            .executable()
            .execution_plan()
            .payload()
            .memory()
    });
    let active_peak = memories
        .iter()
        .map(|memory| memory.static_bytes() + memory.minimum_runnable_request_bytes())
        .max()
        .unwrap();
    let exact = active_peak + opaque;
    let resident_requirement = memories.iter().map(|m| m.static_bytes()).sum::<u64>()
        + memories
            .iter()
            .map(|m| m.minimum_runnable_request_bytes())
            .max()
            .unwrap()
        + opaque;
    // This fixture distinguishes sequential cold stages from simultaneous
    // residency: charging all static installations would reject `exact`.
    assert!(exact < resident_requirement);
    drop(plans);

    for (capacity, accepted) in [(exact - 1, false), (exact, true)] {
        pipeline.config.device_capacity_bytes = capacity;
        pipeline.policy = TensorComponentPolicy::new(
            capacity,
            pipeline
                .runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();
        let before = pipeline.runtime.snapshot();
        // All actual component plans fit independently at both ceilings. The
        // lower one fails only when their peak includes the native allowance.
        let plans = capacity_stages(&pipeline);
        let result = pipeline.validate_resident_capacity(&plans.iter().collect::<Vec<_>>());
        assert_eq!(result.is_ok(), accepted, "{result:?}");
        assert!(!pipeline.is_stopped());
        let after = pipeline.runtime.snapshot();
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.submissions, before.submissions);
        assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
        assert_eq!(after.live_allocations, 0);
        assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
    }
}
