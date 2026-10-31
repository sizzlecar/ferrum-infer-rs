use super::*;
use ferrum_kernels::backend::reference::reference_vnext_operation_registry;
use std::sync::atomic::{AtomicU64, Ordering};

mod stage_budgets;

fn pipeline() -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    let (original, directory) = numeric_pipeline();
    // Capacity assertions fill the process-wide device ledger. Each fixture
    // therefore needs its own simulated device, even under parallel tests.
    static NEXT_DEVICE: AtomicU64 = AtomicU64::new(0);
    let instance = NEXT_DEVICE.fetch_add(1, Ordering::Relaxed);
    let composition = composition(&format!("device.image.dynamic.residency.{instance}"));
    let mut pipeline = QwenImage21Pipeline::new(
        composition.runtime().clone(),
        original.host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        Arc::try_unwrap(original.assets)
            .ok()
            .expect("unique fixture assets"),
        original.profiles,
        original.config,
    )
    .unwrap();
    // The same small numerical projection exercises a decoder-lifetime owner.
    Arc::get_mut(&mut pipeline.assets)
        .expect("unique fixture assets")
        .sources
        .vae = pipeline.assets.sources.transformer.clone();
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainDecoder {
        maximum_bytes: DEVICE_BYTES / 2,
    };
    (pipeline, directory)
}

fn decoder_stage(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    rows: u64,
) -> PreparedStage<ReferenceDeviceRuntime> {
    projection_stage(pipeline, ImageStage::Decoder, rows)
}

fn projection_stage(
    pipeline: &QwenImage21Pipeline<ReferenceDeviceRuntime>,
    stage: ImageStage,
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
    PreparedStage::compile(stage, 1, projection(), &options, pipeline).unwrap()
}

fn retain_decoder(
    pipeline: &mut QwenImage21Pipeline<ReferenceDeviceRuntime>,
    rows: u64,
    stages: &mut Vec<ImageStageReceipt>,
) -> IdleDynamicRetentionReceipt {
    let stage = decoder_stage(pipeline, rows);
    let active = pipeline
        .start_stage(stage, 0, stages)
        .unwrap_or_else(|_| panic!("decoder start"));
    let (active, outputs) = pipeline
        .execute_stage(active, 0, vec![input(&pipeline.host, rows, 0)])
        .unwrap_or_else(|_| panic!("decoder execute"));
    drop(outputs);
    pipeline
        .finish_stage(active, stages)
        .unwrap_or_else(|_| panic!("decoder retention"));
    let ImageStageRetirement::ResidentBackingRetained { retention } =
        &stages.last().unwrap().retirement
    else {
        panic!("decoder backing must be retained")
    };
    retention.clone()
}

#[test]
fn decoder_dynamic_policy_is_explicit_bounded_and_requires_session_residency() {
    assert_eq!(
        ImageDynamicResidencyPolicy::default(),
        ImageDynamicResidencyPolicy::Release
    );
    let policy = ImageDynamicResidencyPolicy::RetainDecoder {
        maximum_bytes: 1024,
    };
    let wire = serde_json::json!({"retain_decoder":{"maximum_bytes":1024}});
    assert_eq!(serde_json::to_value(policy).unwrap(), wire);
    assert_eq!(
        serde_json::from_value::<ImageDynamicResidencyPolicy>(wire).unwrap(),
        policy
    );
    assert!(serde_json::from_value::<ImageDynamicResidencyPolicy>(
        serde_json::json!({"retain_decoder":{"maximum_bytes":1024,"unknown":true}})
    )
    .is_err());
    let small = ImageDynamicResidencyPolicy::RetainSmallStages {
        maximum_bytes_per_stage: 1024,
    };
    let wire = serde_json::json!({"retain_small_stages":{"maximum_bytes_per_stage":1024}});
    assert_eq!(serde_json::to_value(small).unwrap(), wire);
    assert_eq!(
        serde_json::from_value::<ImageDynamicResidencyPolicy>(wire).unwrap(),
        small
    );
    assert!(serde_json::from_value::<ImageDynamicResidencyPolicy>(
        serde_json::json!({"retain_small_stages":{"maximum_bytes_per_stage":1024,"unknown":true}})
    )
    .is_err());
    for residency in [
        ImageResidencyPolicy::Cold,
        ImageResidencyPolicy::DenoiserAndDecoder,
        ImageResidencyPolicy::AllModelComponents,
    ] {
        ImageDynamicResidencyPolicy::Release
            .validate(residency)
            .unwrap();
        assert!(
            ImageDynamicResidencyPolicy::RetainDecoder { maximum_bytes: 0 }
                .validate(residency)
                .is_err()
        );
        assert_eq!(
            policy.validate(residency).is_ok(),
            residency != ImageResidencyPolicy::Cold
        );
        assert_eq!(
            small.validate(residency).is_ok(),
            residency != ImageResidencyPolicy::Cold
        );
        assert!(ImageDynamicResidencyPolicy::RetainSmallStages {
            maximum_bytes_per_stage: 0
        }
        .validate(residency)
        .is_err());
    }
}

#[test]
fn decoder_dynamic_reuse_preserves_outputs_and_original_device_capacity_grants() {
    let (mut pipeline, _directory) = pipeline();
    let capacity = DeviceCapacityOwner::open(pipeline.runtime.descriptor(), DEVICE_BYTES).unwrap();
    let mut stages = Vec::new();
    let mut outputs = Vec::new();
    let mut first_allocations = 0;
    for job in 0..2 {
        let stage = decoder_stage(&pipeline, 1);
        let static_bytes = stage.plan.static_bytes;
        let active = pipeline
            .start_stage(stage, job, &mut stages)
            .unwrap_or_else(|_| panic!("decoder start"));
        assert_eq!(active.receipt.session_reused, job == 1);
        let (active, output) = pipeline
            .execute_stage(
                active,
                job,
                vec![input(&pipeline.host, 1, job as usize * 8)],
            )
            .unwrap_or_else(|_| panic!("decoder forward"));
        let output = only_output(output, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
        for (index, bytes) in output.bytes().bytes().chunks_exact(2).enumerate() {
            assert_eq!(
                u16::from_le_bytes([bytes[0], bytes[1]]),
                bf16::from_f32((index + 1 + job as usize * 8) as f32).to_bits()
            );
        }
        pipeline
            .finish_stage(active, &mut stages)
            .unwrap_or_else(|_| panic!("decoder retain"));
        let ImageStageRetirement::ResidentBackingRetained { retention } =
            &stages.last().unwrap().retirement
        else {
            panic!("bounded decoder retention")
        };
        assert!(retention.retained_bytes() > 0 && retention.retained_chunks() > 0);
        let remaining = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                DEVICE_BYTES - static_bytes - retention.retained_bytes(),
            )
            .unwrap();
        assert!(capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1)
            .is_err());
        drop(remaining);
        let snapshot = pipeline.runtime.snapshot();
        if job == 0 {
            first_allocations = snapshot.allocations;
        } else {
            assert_eq!(snapshot.allocations, first_allocations);
        }
        outputs.push(output);
    }
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(&outputs[0].bytes().bytes()[..2], &0x3f80_u16.to_le_bytes());
    drop(outputs);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn decoder_dynamic_budget_counts_other_stages_and_actual_old_shape_without_double_counting() {
    let (mut pipeline, _directory) = pipeline();
    let mut stages = Vec::new();
    let retention = retain_decoder(&mut pipeline, 4, &mut stages);
    assert!(retention.retained_bytes() > 1);
    // Tightening a future retention ceiling cannot erase the actual old pool.
    pipeline.config.dynamic_residency =
        ImageDynamicResidencyPolicy::RetainDecoder { maximum_bytes: 1 };
    let dit = prepared(&pipeline, 1);
    let decoder = decoder_stage(&pipeline, 1);
    let memory = |stage: &PreparedStage<ReferenceDeviceRuntime>| {
        stage
            .component
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .minimum_runnable_request_bytes()
    };
    // The changed shape closes its old pool before allocating new backing.
    let own_peak = memory(&decoder);
    pipeline.config.device_capacity_bytes = decoder.plan.static_bytes + own_peak;
    pipeline.validate_resident_capacity(&[&decoder]).unwrap();
    let required = decoder.plan.static_bytes
        + dit.plan.static_bytes
        + own_peak.max(memory(&dit) + retention.retained_bytes());
    let before = pipeline.runtime.snapshot();
    pipeline.config.device_capacity_bytes = required - 1;
    assert!(pipeline
        .validate_resident_capacity(&[&dit, &decoder])
        .is_err());
    pipeline.config.device_capacity_bytes = required;
    pipeline
        .validate_resident_capacity(&[&dit, &decoder])
        .unwrap();
    assert_eq!(pipeline.runtime.snapshot(), before);
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

#[test]
fn decoder_plan_replacement_closes_retained_pool_before_admitting_new_dynamic_backing() {
    let (mut pipeline, _directory) = pipeline();
    let mut stages = Vec::new();
    let retention = retain_decoder(&mut pipeline, 4, &mut stages);
    let changed = decoder_stage(&pipeline, 1);
    let before = pipeline.runtime.snapshot();
    let active = pipeline
        .start_stage(changed, 1, &mut stages)
        .unwrap_or_else(|_| panic!("decoder replacement"));
    assert!(!active.receipt.session_reused);
    assert!(active.receipt.weights_reused);
    assert!(matches!(
        stages.last().unwrap().retirement,
        ImageStageRetirement::Closed
    ));
    let after = pipeline.runtime.snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(
        after.live_allocations + retention.retained_chunks() as u64,
        before.live_allocations
    );
    let (active, outputs) = active
        .execute(1, vec![input(&pipeline.host, 1, 0)])
        .unwrap_or_else(|_| panic!("replacement execute"));
    pipeline
        .finish_stage(active, &mut stages)
        .unwrap_or_else(|_| panic!("replacement retain"));
    drop(outputs);
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn decoder_pool_over_its_retention_ceiling_is_released_and_other_stages_still_trim() {
    let (mut pipeline, _directory) = pipeline();
    pipeline.config.dynamic_residency =
        ImageDynamicResidencyPolicy::RetainDecoder { maximum_bytes: 1 };
    let mut stages = Vec::new();
    for stage in [decoder_stage(&pipeline, 1), prepared(&pipeline, 1)] {
        let active = pipeline
            .start_stage(stage, 0, &mut stages)
            .unwrap_or_else(|_| panic!("stage start"));
        let (active, outputs) = active
            .execute(0, vec![input(&pipeline.host, 1, 0)])
            .unwrap_or_else(|_| panic!("stage execute"));
        drop(outputs);
        pipeline
            .finish_stage(active, &mut stages)
            .unwrap_or_else(|_| panic!("idle release"));
        let ImageStageRetirement::Resident { idle_release } = &stages.last().unwrap().retirement
        else {
            panic!("dynamic backing should be released")
        };
        assert!(idle_release.reclaimed_bytes() > 1);
    }
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

fn small_pipeline(
    maximum_bytes_per_stage: u64,
) -> (
    QwenImage21Pipeline<ReferenceDeviceRuntime>,
    tempfile::TempDir,
) {
    let (mut pipeline, directory) = pipeline();
    Arc::get_mut(&mut pipeline.assets)
        .expect("unique fixture assets")
        .sources
        .text_encoder = pipeline.assets.sources.transformer.clone();
    pipeline.config.residency = ImageResidencyPolicy::AllModelComponents;
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainSmallStages {
        maximum_bytes_per_stage,
    };
    (pipeline, directory)
}

#[test]
fn small_stages_reuse_two_independent_pools_and_preserve_numerical_outputs() {
    let (mut pipeline, _directory) = small_pipeline(4096);
    let mut stages = Vec::new();
    let mut first_allocations = 0;
    for job in 0..2 {
        for kind in [ImageStage::VisionEncoder, ImageStage::TextEncoder] {
            let stage = projection_stage(&pipeline, kind, 1);
            let active = pipeline
                .start_stage(stage, job, &mut stages)
                .unwrap_or_else(|_| panic!("small stage start"));
            assert_eq!(active.receipt.session_reused, job == 1);
            let (active, output) = pipeline
                .execute_stage(
                    active,
                    job,
                    vec![input(&pipeline.host, 1, job as usize * 8)],
                )
                .unwrap_or_else(|_| panic!("small stage execute"));
            let output = only_output(output, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
            for (index, bytes) in output.bytes().bytes().chunks_exact(2).enumerate() {
                assert_eq!(
                    u16::from_le_bytes([bytes[0], bytes[1]]),
                    bf16::from_f32((index + 1 + job as usize * 8) as f32).to_bits()
                );
            }
            pipeline
                .finish_stage(active, &mut stages)
                .unwrap_or_else(|_| panic!("small stage retain"));
            let ImageStageRetirement::ResidentBackingRetained { retention } =
                &stages.last().unwrap().retirement
            else {
                panic!("small stage retained")
            };
            assert!((1..=4096).contains(&retention.retained_bytes()));
            assert!(retention.retained_chunks() > 0);
        }
        let snapshot = pipeline.runtime.snapshot();
        if job == 0 {
            first_allocations = snapshot.allocations;
        } else {
            assert_eq!(snapshot.allocations, first_allocations);
        }
    }
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn small_stage_capacity_reserves_every_eligible_slot_and_keeps_actual_old_shape_bytes() {
    let limit = 4096;
    let (mut pipeline, _directory) = small_pipeline(limit);
    let kinds = [ImageStage::VisionEncoder, ImageStage::TextEncoder];
    let planned = kinds.map(|kind| projection_stage(&pipeline, kind, 1));
    let demands = planned.each_ref().map(|stage| {
        stage
            .component
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .minimum_runnable_request_bytes()
    });
    let static_sum = planned
        .iter()
        .map(|stage| stage.plan.static_bytes)
        .sum::<u64>();
    // AllModelComponents has five eligible slots; the active slot is excluded.
    let required = static_sum + demands[0].max(demands[1]) + 4 * limit;
    let before = pipeline.runtime.snapshot();
    pipeline.config.device_capacity_bytes = required - 1;
    assert!(pipeline
        .validate_resident_capacity(&planned.iter().collect::<Vec<_>>())
        .is_err());
    pipeline.config.device_capacity_bytes = required;
    pipeline
        .validate_resident_capacity(&planned.iter().collect::<Vec<_>>())
        .unwrap();
    assert_eq!(pipeline.runtime.snapshot(), before);
    pipeline.config.device_capacity_bytes = DEVICE_BYTES;

    let mut receipts = Vec::new();
    let mut actual = [0; 2];
    for (index, kind) in kinds.into_iter().enumerate() {
        let stage = projection_stage(&pipeline, kind, 4);
        let active = pipeline
            .start_stage(stage, 0, &mut receipts)
            .unwrap_or_else(|_| panic!("old shape start"));
        let (active, outputs) = active
            .execute(0, vec![input(&pipeline.host, 4, 0)])
            .unwrap_or_else(|_| panic!("old shape execute"));
        drop(outputs);
        pipeline
            .finish_stage(active, &mut receipts)
            .unwrap_or_else(|_| panic!("old shape retain"));
        let ImageStageRetirement::ResidentBackingRetained { retention } =
            &receipts.last().unwrap().retirement
        else {
            panic!("old pool retained")
        };
        actual[index] = retention.retained_bytes();
        assert!(actual[index] > 1);
    }
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainSmallStages {
        maximum_bytes_per_stage: 1,
    };
    let required = static_sum + (demands[0] + actual[1]).max(demands[1] + actual[0]) + 3;
    pipeline.config.device_capacity_bytes = required - 1;
    assert!(pipeline
        .validate_resident_capacity(&planned.iter().collect::<Vec<_>>())
        .is_err());
    pipeline.config.device_capacity_bytes = required;
    pipeline
        .validate_resident_capacity(&planned.iter().collect::<Vec<_>>())
        .unwrap();
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

#[test]
fn small_stages_over_their_individual_ceiling_release_backing() {
    let (mut pipeline, _directory) = small_pipeline(1);
    let mut stages = Vec::new();
    for kind in [ImageStage::VisionEncoder, ImageStage::TextEncoder] {
        let stage = projection_stage(&pipeline, kind, 1);
        let active = pipeline
            .start_stage(stage, 0, &mut stages)
            .unwrap_or_else(|_| panic!("small stage start"));
        let (active, outputs) = active
            .execute(0, vec![input(&pipeline.host, 1, 0)])
            .unwrap_or_else(|_| panic!("small stage execute"));
        drop(outputs);
        pipeline
            .finish_stage(active, &mut stages)
            .unwrap_or_else(|_| panic!("small stage release"));
        let ImageStageRetirement::Resident { idle_release } = &stages.last().unwrap().retirement
        else {
            panic!("oversized pool released")
        };
        assert!(idle_release.reclaimed_bytes() > 1);
    }
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}
