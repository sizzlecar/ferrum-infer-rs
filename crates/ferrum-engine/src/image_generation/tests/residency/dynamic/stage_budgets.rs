use super::*;
use std::num::NonZeroU64;

fn budgets() -> ImageStageRetentionBudgets {
    ImageStageRetentionBudgets {
        text_encoder: NonZeroU64::new(2048),
        vision_encoder: NonZeroU64::new(3072),
        decoder: NonZeroU64::new(4096),
        ..ImageStageRetentionBudgets::default()
    }
}

#[test]
fn stage_budgets_require_known_positive_selected_resident_stages() {
    let policy = ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: budgets(),
    };
    let wire = serde_json::json!({"retain_stage_budgets":{"maximum_bytes_by_stage":{
        "text_encoder":2048,"vision_encoder":3072,"decoder":4096
    }}});
    assert_eq!(serde_json::to_value(policy).unwrap(), wire);
    assert_eq!(
        serde_json::from_value::<ImageDynamicResidencyPolicy>(wire).unwrap(),
        policy
    );
    policy
        .validate(ImageResidencyPolicy::AllModelComponents)
        .unwrap();
    assert!(policy.validate(ImageResidencyPolicy::Cold).is_err());
    assert!(policy
        .validate(ImageResidencyPolicy::DenoiserAndDecoder)
        .is_err());
    ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: ImageStageRetentionBudgets {
            decoder: NonZeroU64::new(1),
            ..ImageStageRetentionBudgets::default()
        },
    }
    .validate(ImageResidencyPolicy::DenoiserAndDecoder)
    .unwrap();
    assert!(ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: ImageStageRetentionBudgets::default(),
    }
    .validate(ImageResidencyPolicy::AllModelComponents)
    .is_err());
    for fields in [
        serde_json::json!({"decoder":0}),
        serde_json::json!({"decoder":-1}),
        serde_json::json!({"noise":1024}),
        serde_json::json!({"unknown_stage":1024}),
        serde_json::json!({"decoder":1024,"unknown_stage":1}),
    ] {
        assert!(
            serde_json::from_value::<ImageDynamicResidencyPolicy>(serde_json::json!({
                "retain_stage_budgets":{"maximum_bytes_by_stage":fields}
            }))
            .is_err()
        );
    }
    assert_eq!(budgets().maximum_bytes(ImageStage::VaeEncoder), 0);
    assert_eq!(budgets().maximum_bytes(ImageStage::Noise), 0);
}

#[test]
fn selected_stage_budgets_reuse_real_backing_and_omit_other_stages() {
    let (mut pipeline, _directory) = small_pipeline(4096);
    let limits = budgets();
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: limits,
    };
    let mut receipts = Vec::new();
    for job in 0..2 {
        for kind in [
            ImageStage::VisionEncoder,
            ImageStage::TextEncoder,
            ImageStage::Decoder,
            ImageStage::Denoiser,
        ] {
            let before = pipeline.runtime.snapshot().allocations;
            let stage = projection_stage(&pipeline, kind, 1);
            let active = pipeline
                .start_stage(stage, job, &mut receipts)
                .unwrap_or_else(|_| panic!("stage start"));
            let (active, outputs) = pipeline
                .execute_stage(
                    active,
                    job,
                    vec![input(&pipeline.host, 1, job as usize * 8)],
                )
                .unwrap_or_else(|_| panic!("stage execute"));
            let output = only_output(outputs, IMAGE_PROJECTION_OUTPUT_ID).unwrap();
            for (index, bytes) in output.bytes().bytes().chunks_exact(2).enumerate() {
                assert_eq!(
                    u16::from_le_bytes([bytes[0], bytes[1]]),
                    bf16::from_f32((index + 1 + job as usize * 8) as f32).to_bits()
                );
            }
            drop(output);
            pipeline
                .finish_stage(active, &mut receipts)
                .unwrap_or_else(|_| panic!("stage retire"));
            if limits.maximum_bytes(kind) == 0 {
                let ImageStageRetirement::Resident { idle_release } =
                    &receipts.last().unwrap().retirement
                else {
                    panic!("unselected stage must release")
                };
                assert!(idle_release.reclaimed_bytes() > 0);
            } else {
                let ImageStageRetirement::ResidentBackingRetained { retention } =
                    &receipts.last().unwrap().retirement
                else {
                    panic!("selected stage retains")
                };
                assert!(retention.retained_bytes() > 0);
                assert!(retention.retained_bytes() < limits.maximum_bytes(kind));
                if job == 1 {
                    assert_eq!(pipeline.runtime.snapshot().allocations, before);
                }
            }
        }
    }
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
    assert_eq!(pipeline.host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn stage_budget_capacity_keeps_actual_old_receipts_after_ceiling_and_selection_changes() {
    let (mut pipeline, _directory) = small_pipeline(4096);
    let limits = budgets();
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: limits,
    };
    let kinds = [
        ImageStage::TextEncoder,
        ImageStage::VisionEncoder,
        ImageStage::Decoder,
        ImageStage::Denoiser,
    ];
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
    let static_bytes = planned
        .iter()
        .map(|stage| stage.plan.static_bytes)
        .sum::<u64>();
    let declared_sum = kinds
        .iter()
        .map(|&kind| limits.maximum_bytes(kind))
        .sum::<u64>();
    let required = static_bytes
        + kinds
            .iter()
            .enumerate()
            .map(|(index, &kind)| demands[index] + declared_sum - limits.maximum_bytes(kind))
            .max()
            .unwrap();
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
    let mut actual = [0_u64; 4];
    for (index, &kind) in kinds[..3].iter().enumerate() {
        let old = projection_stage(&pipeline, kind, 4);
        let active = pipeline
            .start_stage(old, 0, &mut receipts)
            .unwrap_or_else(|_| panic!("old shape start"));
        let (active, outputs) = pipeline
            .execute_stage(active, 0, vec![input(&pipeline.host, 4, 0)])
            .unwrap_or_else(|_| panic!("old shape execute"));
        drop(outputs);
        pipeline
            .finish_stage(active, &mut receipts)
            .unwrap_or_else(|_| panic!("old shape retain"));
        let ImageStageRetirement::ResidentBackingRetained { retention } =
            &receipts.last().unwrap().retirement
        else {
            panic!("old receipt")
        };
        actual[index] = retention.retained_bytes();
        assert!(actual[index] > 1);
    }
    // A lowered cap and an omitted stage cannot erase existing physical pools.
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: ImageStageRetentionBudgets {
            decoder: NonZeroU64::new(1),
            ..ImageStageRetentionBudgets::default()
        },
    };
    let actual_sum = actual.iter().sum::<u64>();
    let required = static_bytes
        + demands
            .iter()
            .enumerate()
            .map(|(index, &demand)| demand + actual_sum - actual[index])
            .max()
            .unwrap();
    pipeline.config.device_capacity_bytes = required - 1;
    assert!(pipeline
        .validate_resident_capacity(&planned.iter().collect::<Vec<_>>())
        .is_err());
    pipeline.config.device_capacity_bytes = required;
    pipeline
        .validate_resident_capacity(&planned.iter().collect::<Vec<_>>())
        .unwrap();
    pipeline.config.device_capacity_bytes = DEVICE_BYTES;
    // Changing the decoder's plan retires its old pool before the replacement.
    let replacement = projection_stage(&pipeline, ImageStage::Decoder, 1);
    let active = pipeline
        .start_stage(replacement, 1, &mut receipts)
        .unwrap_or_else(|_| panic!("replace decoder"));
    let eviction = receipts.last().unwrap();
    assert_eq!(eviction.execution.plan.stage, ImageStage::Decoder);
    assert!(eviction.close.is_some());
    // Replacement must remove the old decoder bytes from the capacity ledger.
    actual[2] = 0;
    let allowances = [actual[0], actual[1], 1, 0];
    let allowance_sum = allowances.iter().sum::<u64>();
    let required_after_replacement = static_bytes
        + demands
            .iter()
            .enumerate()
            .map(|(index, &demand)| demand + allowance_sum - allowances[index])
            .max()
            .unwrap();
    pipeline.config.device_capacity_bytes = required_after_replacement;
    pipeline
        .validate_resident_capacity(&planned.iter().collect::<Vec<_>>())
        .unwrap();
    pipeline.config.device_capacity_bytes = DEVICE_BYTES;
    pipeline
        .finish_stage(active, &mut receipts)
        .unwrap_or_else(|_| panic!("replacement retire"));
    pipeline.shutdown().unwrap();
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}
