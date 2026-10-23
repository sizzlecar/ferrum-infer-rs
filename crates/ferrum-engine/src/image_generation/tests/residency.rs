use super::{fixture::*, *};
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
use ferrum_models::qwen_image21::{IMAGE_PROJECTION_INPUT_ID, IMAGE_PROJECTION_OUTPUT_ID};
use half::bf16;

mod all_components;
mod dynamic;

pub(super) fn numeric_pipeline() -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    let (mut pipeline, directory) = super::adapter::pipeline(2);
    pipeline.config.residency = ImageResidencyPolicy::DenoiserAndDecoder;
    pipeline.config.static_initialization = StaticInitializationPolicy::new(1024, 8).unwrap();
    let path = directory.path().join("projection");
    std::fs::create_dir(&path).unwrap();
    // Nonzero diagonal projection: y[o] = x[o] for o<8. The expectation is
    // independent of either graph execution or reuse bookkeeping.
    let mut bytes = vec![0u8; 8 * 64 * 2];
    for channel in 0..8 {
        bytes[(channel * 64 + channel) * 2..][..2].copy_from_slice(&0x3f80_u16.to_le_bytes());
    }
    serialize_to_file(
        [(
            "img_in.weight",
            TensorView::new(Dtype::BF16, vec![8, 64], &bytes).unwrap(),
        )],
        &None,
        &path.join("model.safetensors"),
    )
    .unwrap();
    pipeline.assets.sources.transformer = Arc::new(SafetensorsArchive::open(&path).unwrap());
    (pipeline, directory)
}

pub(super) fn prepared(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    rows: u64,
) -> PreparedStage<ReferenceDeviceRuntime> {
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(IMAGE_PROJECTION_INPUT_ID).unwrap(),
        ProgramTensorSpec {
            dimensions: vec![rows, 64],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    PreparedStage::compile(
        ImageStage::Denoiser,
        rows as usize,
        projection(),
        &options,
        pipeline,
    )
    .unwrap()
}

pub(super) fn input(host: &HostMemoryAccount, rows: u64, offset: usize) -> ManagedHostTensor {
    let tensor = ResolvedTensorSpec::new(
        vec![rows, 64],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            tensor.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for (index, part) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
        part.copy_from_slice(
            &bf16::from_f32((offset + index + 1) as f32)
                .to_bits()
                .to_le_bytes(),
        );
    }
    ManagedHostTensor::new(
        id(IMAGE_PROJECTION_INPUT_ID).unwrap(),
        tensor,
        bytes.freeze(),
    )
    .unwrap()
}

#[test]
fn exact_plan_reuses_static_weights_trims_dynamic_and_preserves_independent_outputs() {
    let (mut pipeline, _directory) = numeric_pipeline();
    let mut stages = Vec::new();
    let mut retained = Vec::new();
    let mut previous_uploads = 0;
    for job in 0..2 {
        let mut stage = prepared(&pipeline, 1);
        stage.plan.compile_wall_duration_us = 123;
        let active = pipeline
            .start_stage(stage, job, &mut stages)
            .unwrap_or_else(|_| panic!("initialize/reuse"));
        assert_eq!(active.receipt.session_reused, job == 1);
        assert_eq!(active.receipt.weights_reused, job == 1);
        let (active, output) = active
            .execute(job, vec![input(&pipeline.host, 1, job as usize * 8)])
            .unwrap_or_else(|_| panic!("numeric projection"));
        let output = only_output(output, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
        for (index, part) in output.bytes().bytes().chunks_exact(2).enumerate() {
            assert_eq!(
                u16::from_le_bytes([part[0], part[1]]),
                bf16::from_f32((index + 1 + job as usize * 8) as f32).to_bits()
            );
        }
        pipeline
            .finish_stage(active, &mut stages)
            .unwrap_or_else(|_| panic!("idle trim"));
        let snapshot = pipeline.runtime.snapshot();
        if job == 1 {
            // The second batch uploads only X, never the 1024-byte weight.
            assert_eq!(snapshot.uploaded_bytes - previous_uploads, 128);
        }
        previous_uploads = snapshot.uploaded_bytes;
        assert!(snapshot.live_allocations > 0);
        let receipt = stages.last().unwrap();
        assert!(receipt.close.is_none());
        assert!(serde_json::to_value(receipt)
            .unwrap()
            .get("close")
            .is_none());
        let ImageStageRetirement::Resident { idle_release } = &receipt.retirement else {
            panic!("resident receipt")
        };
        assert!(idle_release.reclaimed_bytes() > 0);
        retained.push(output);
    }
    let shutdown = pipeline.shutdown().unwrap();
    assert_eq!(shutdown.len(), 2);
    assert!(shutdown[0].close.is_some());
    assert!(serde_json::to_value(&shutdown[0]).unwrap()["close"].is_object());
    assert!(matches!(
        shutdown[0].retirement,
        ImageStageRetirement::Closed
    ));
    assert_eq!(shutdown[0].execution.completed_executions, 0);
    assert_eq!(shutdown[0].execution.plan.compile_wall_duration_us, 0);
    let ImageStageRetirement::SharedWeightsClosed { receipt } = &shutdown[1].retirement else {
        panic!("snapshot retires after its session")
    };
    assert_eq!(receipt.released_static_resources(), 1);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 32);
    assert!(pipeline.shutdown().unwrap().is_empty());
    assert!(pipeline.is_stopped());
    assert_eq!(&retained[0].bytes().bytes()[..2], &0x3f80_u16.to_le_bytes());
    drop(retained);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn changed_plan_closes_previous_session_and_attaches_without_weight_upload() {
    let (mut pipeline, _directory) = numeric_pipeline();
    let mut stages = Vec::new();
    let mut first = prepared(&pipeline, 1);
    first.plan.compile_wall_duration_us = 123;
    let old_hash = first.plan.plan_hash.clone();
    let active = pipeline
        .start_stage(first, 0, &mut stages)
        .unwrap_or_else(|_| panic!("first initialization"));
    let (active, outputs) = active
        .execute(0, vec![input(&pipeline.host, 1, 0)])
        .unwrap_or_else(|_| panic!("first execute"));
    drop(outputs);
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("first trim"));
    let changed = prepared(&pipeline, 2);
    assert_ne!(changed.plan.plan_hash, old_hash);
    let before = pipeline.runtime.snapshot();
    let active = pipeline
        .start_stage(changed, 1, &mut stages)
        .unwrap_or_else(|_| panic!("replacement"));
    assert!(!active.receipt.session_reused);
    assert!(active.receipt.weights_reused);
    assert!(matches!(
        active.receipt.initialization,
        ComponentInitializationReceipt::SharedWeights { .. }
    ));
    assert_eq!(
        pipeline.runtime.snapshot().uploaded_bytes,
        before.uploaded_bytes
    );
    assert_eq!(pipeline.runtime.snapshot().allocations, before.allocations);
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[1].execution.plan.plan_hash, old_hash);
    assert!(stages[1].close.is_some());
    assert_eq!(
        stages[1]
            .close
            .as_ref()
            .unwrap()
            .released_static_resources(),
        0
    );
    assert_eq!(stages[1].execution.completed_executions, 0);
    assert_eq!(stages[1].execution.plan.compile_wall_duration_us, 0);
    let (active, outputs) = active
        .execute(1, vec![input(&pipeline.host, 2, 0)])
        .unwrap_or_else(|_| panic!("replacement execute"));
    assert_eq!(outputs[0].tensor().dimensions(), [2, 8]);
    assert_eq!(
        pipeline.runtime.snapshot().uploaded_bytes - before.uploaded_bytes,
        2 * 64 * 2
    );
    assert_eq!(
        &outputs[0].bytes().bytes()[16..18],
        &bf16::from_f32(65.).to_bits().to_le_bytes()
    );
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("replacement trim"));
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    drop(outputs);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn reused_session_failure_keeps_original_owner_outside_resident_slot() {
    let (mut pipeline, _directory) = numeric_pipeline();
    let mut stages = Vec::new();
    let first = prepared(&pipeline, 1);
    let active = pipeline
        .start_stage(first, 0, &mut stages)
        .unwrap_or_else(|_| panic!("initialization"));
    let (active, outputs) = active
        .execute(0, vec![input(&pipeline.host, 1, 0)])
        .unwrap_or_else(|_| panic!("execute"));
    drop(outputs);
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("trim"));
    let second = prepared(&pipeline, 1);
    let active = pipeline
        .start_stage(second, 1, &mut stages)
        .unwrap_or_else(|_| panic!("reuse"));
    let ImageGenerationFailure::Execution { source, active } =
        active.execute(1, Vec::new()).err().unwrap()
    else {
        panic!("missing input must retain executor owner")
    };
    assert!(active.receipt.session_reused);
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    // Shutdown reports the retained snapshot owner rather than claiming that
    // an active session returned in the earlier error has retired.
    let shutdown = pipeline.shutdown().unwrap_err();
    let ImageGenerationFailure::SharedWeightsReferenced {
        outcome: SharedComponentWeightsCloseOutcome::Referenced { weights },
        ..
    } = shutdown.failure
    else {
        panic!("snapshot close must return its referenced owner")
    };
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    drop(source);
    active
        .close()
        .unwrap_or_else(|_| panic!("quiescent explicit recovery"));
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    match weights.close() {
        Ok(SharedComponentWeightsCloseOutcome::Closed(_)) => {}
        _ => panic!("last recovered snapshot must retire"),
    }
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

#[test]
fn incompatible_weight_plan_is_rejected_without_replacement_upload() {
    let (mut pipeline, _directory) = numeric_pipeline();
    let mut stages = Vec::new();
    let first = prepared(&pipeline, 1);
    let active = pipeline
        .start_stage(first, 0, &mut stages)
        .unwrap_or_else(|_| panic!("initialize"));
    let (active, outputs) = active
        .execute(0, vec![input(&pipeline.host, 1, 0)])
        .unwrap_or_else(|_| panic!("execute"));
    drop(outputs);
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("trim"));
    let mut raw = projection().canonical_config().clone();
    raw["in_channels"] = serde_json::json!(32);
    let family = TypedFamilyRegistration::new(
        ferrum_models::qwen_image21::QwenImage21ImageProjectionFamily::new().unwrap(),
    )
    .prepare_with_profile(
        &raw,
        &NumericalProfileId::new(ferrum_models::qwen_image21::IMAGE_PROJECTION_PROFILE_ID).unwrap(),
    )
    .unwrap();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(IMAGE_PROJECTION_INPUT_ID).unwrap(),
        ProgramTensorSpec {
            dimensions: vec![1, 32],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    let changed =
        PreparedStage::compile(ImageStage::Denoiser, 1, family, &options, &pipeline).unwrap();
    let before = pipeline.runtime.snapshot();
    assert!(matches!(
        pipeline.start_stage(changed, 1, &mut stages),
        Err(ImageGenerationFailure::Initialization {
            source: ComponentInitializationError::Contract(_),
            ..
        })
    ));
    assert_eq!(
        pipeline.runtime.snapshot().uploaded_bytes,
        before.uploaded_bytes
    );
    assert_eq!(pipeline.runtime.snapshot().allocations, before.allocations);
    assert!(pipeline.runtime.snapshot().live_allocations > 0);
    let shutdown = pipeline.shutdown().unwrap();
    assert_eq!(shutdown.len(), 1);
    assert!(matches!(
        shutdown[0].retirement,
        ImageStageRetirement::SharedWeightsClosed { .. }
    ));
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}
