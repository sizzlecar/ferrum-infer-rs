use super::{fixture::*, *};
use ferrum_models::qwen_image21::QwenImage21TransformerPhase;

#[test]
fn prefix_handoff_counts_two_caches_and_one_weight_installation_before_execution() {
    let (mut pipeline, _directory) = super::adapter::pipeline(1);
    // Two text rows, 64 VLM image slots, then two suffix rows. Each
    // image slot expands to four latent rows: 64 * 4 == 16 * 16.
    // Keep the large reference prefix and small target to expose the
    // simultaneous producer-output and decoder-state capacity boundary.
    let config = QwenImage21TransformerConfig::new(
        transformer(),
        QwenImage21T2iLayout::new(68, 2, 2, vec![68]).unwrap(),
    )
    .unwrap()
    .with_reference_image(2, 64, 16, 16)
    .unwrap();
    let prefill = pipeline
        .prepare_denoiser(
            &config
                .clone()
                .with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)
                .unwrap(),
            1,
        )
        .unwrap();
    let decode = pipeline
        .prepare_denoiser(
            &config
                .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
                .unwrap(),
            1,
        )
        .unwrap();
    let pm = prefill
        .component
        .executable()
        .execution_plan()
        .payload()
        .memory();
    let dm = decode
        .component
        .executable()
        .execution_plan()
        .payload()
        .memory();
    assert!(dm.minimum_request_bytes() > 0);
    assert!(pm.static_bytes() >= dm.static_bytes());
    let ordinary = pm.static_bytes()
        + pm.minimum_runnable_request_bytes()
            .max(dm.minimum_runnable_request_bytes());
    let transition = pm.static_bytes()
        + pm.minimum_runnable_request_bytes()
        + dm.minimum_request_bytes()
        + dm.minimum_sequence_bytes();
    assert!(
        transition > ordinary,
        "reference-prefix fixture must exercise simultaneous cache pressure"
    );
    for policy in [
        ImageResidencyPolicy::Cold,
        ImageResidencyPolicy::AllModelComponents,
    ] {
        pipeline.config.residency = policy;
        pipeline.config.device_capacity_bytes = transition - 1;
        assert!(pipeline
            .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)),)
            .is_err());
        pipeline.config.device_capacity_bytes = transition;
        pipeline
            .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
            .unwrap();
    }
    let decoder_retention = 4096;
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainDecoder {
        maximum_bytes: decoder_retention,
    };
    pipeline.config.device_capacity_bytes = transition + decoder_retention - 1;
    assert!(pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .is_err());
    pipeline.config.device_capacity_bytes = transition + decoder_retention;
    pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .unwrap();
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainSmallStages {
        maximum_bytes_per_stage: decoder_retention,
    };
    // Five retained-stage allowances, with the one shared Denoiser slot excluded
    // for both cooperating plans throughout the two-cache handoff.
    pipeline.config.device_capacity_bytes = transition + 4 * decoder_retention - 1;
    assert!(pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .is_err());
    pipeline.config.device_capacity_bytes = transition + 4 * decoder_retention;
    pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .unwrap();
    let stage_budgets = ImageStageRetentionBudgets {
        text_encoder: std::num::NonZeroU64::new(2048),
        vision_encoder: std::num::NonZeroU64::new(3072),
        decoder: std::num::NonZeroU64::new(4096),
        ..ImageStageRetentionBudgets::default()
    };
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: stage_budgets,
    };
    // The two-cache handoff coexists with every selected other stage's pool;
    // unselected Denoiser/VAE-encoder slots add no speculative allowance.
    let retained_other = 2048 + 3072 + 4096;
    pipeline.config.device_capacity_bytes = transition + retained_other - 1;
    assert!(pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .is_err());
    pipeline.config.device_capacity_bytes = transition + retained_other;
    pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .unwrap();
    pipeline.config.residency = ImageResidencyPolicy::AllStages;
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::RetainStageBudgets {
        maximum_bytes_by_stage: ImageStageRetentionBudgets {
            noise: std::num::NonZeroU64::new(1024),
            ..stage_budgets
        },
    };
    pipeline.config.device_capacity_bytes = transition + retained_other + 1024 - 1;
    assert!(pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .is_err());
    pipeline.config.device_capacity_bytes = transition + retained_other + 1024;
    pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .unwrap();
    assert_eq!(pipeline.runtime.snapshot().uploaded_bytes, 0);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}
