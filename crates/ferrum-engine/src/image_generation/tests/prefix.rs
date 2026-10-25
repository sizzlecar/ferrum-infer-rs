use super::{fixture::*, *};
use ferrum_models::qwen_image21::QwenImage21TransformerPhase;

#[test]
fn prefix_command_preflight_uses_selected_programs_and_all_state_copies() {
    use crate::image_generation::composition::{
        component_command_count, prefix_command_families, validate_prefix_command_limit,
    };
    let (pipeline, _directory) = super::adapter::pipeline(1);
    for layers in [1, 3] {
        let mut metadata = transformer_metadata();
        metadata["num_layers"] = serde_json::json!(layers);
        let architecture = QwenImage21Config::from_hf_json(
            &serde_json::to_vec(&metadata).unwrap(),
            StructureLimits::default(),
        )
        .unwrap();
        let config = QwenImage21TransformerConfig::new(
            architecture,
            QwenImage21T2iLayout::new(1, 2, 2, vec![1]).unwrap(),
        )
        .unwrap();
        for profile in [
            QwenImage21TransformerFamily::REFERENCE_PROFILE_ID,
            QwenImage21TransformerFamily::METAL_PROFILE_ID,
            QwenImage21TransformerFamily::METAL_FP8_PROFILE_ID,
        ] {
            let mut profiles = pipeline.profiles.clone();
            profiles.denoiser = NumericalProfileId::new(profile).unwrap();
            let ([prefill, decode], copies) = prefix_command_families(&config, &profiles).unwrap();
            assert_eq!(copies, 2 * layers as usize);
            assert!(prefill.program().states().is_empty());
            assert_eq!(decode.program().states().len(), copies);
            for state in decode.program().states() {
                assert_eq!(state.lifetime, StateLifetime::Request);
                assert!(prefill.program().outputs().contains(&state.value_id));
            }
            let nodes: Vec<_> = decode
                .program()
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .collect();
            assert_eq!(
                nodes
                    .iter()
                    .filter(|node| node.operation_id.as_str() == CROSS_ATTENTION_BF16_OPERATION_ID)
                    .count(),
                layers as usize
            );
            assert!(nodes
                .iter()
                .all(|node| node.operation_id.as_str() != CROSS_ATTENTION_FUSED_BF16_OPERATION_ID));
            let commands = component_command_count(&prefill).unwrap();
            assert!(validate_prefix_command_limit(&prefill, copies, commands).is_err());
            assert!(
                validate_prefix_command_limit(&prefill, copies, commands + copies - 1).is_err()
            );
            assert_eq!(
                validate_prefix_command_limit(&prefill, copies, commands + copies).unwrap(),
                commands + copies,
            );
            assert!(validate_prefix_command_limit(&prefill, usize::MAX, usize::MAX).is_err());
        }
    }
    assert_eq!(pipeline.runtime.snapshot().uploaded_bytes, 0);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}

#[test]
fn prefix_command_preflight_preserves_native_weight_encodings() {
    let (pipeline, _directory) = super::adapter::pipeline(1);
    let mut metadata = transformer_metadata();
    metadata["num_attention_heads"] = serde_json::json!(4);
    metadata["context_in_dim"] = serde_json::json!(32);
    let architecture = QwenImage21Config::from_hf_json(
        &serde_json::to_vec(&metadata).unwrap(),
        StructureLimits::default(),
    )
    .unwrap();
    let expected = ferrum_models::qwen_image21::ExpectedWeights::from_config(
        &architecture,
        StructureLimits::default(),
    )
    .unwrap();
    let encodings = expected
        .tensors()
        .values()
        .map(|tensor| {
            let encoding = if tensor.dimensions().len() == 1 {
                WeightEncoding::Dense {
                    element_type: ElementType::F32,
                }
            } else {
                WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: QuantizationFormatId::new("quantization.gguf.q8-0").unwrap(),
                    logical_values_per_block: 32,
                    bytes_per_block: 34,
                })
            };
            (tensor.external_name().to_owned(), encoding)
        })
        .collect::<BTreeMap<_, _>>();
    let config = QwenImage21TransformerConfig::new(
        architecture,
        QwenImage21T2iLayout::new(1, 2, 2, vec![1]).unwrap(),
    )
    .unwrap()
    .with_weight_encodings(encodings.clone())
    .unwrap();
    let mut profiles = pipeline.profiles.clone();
    profiles.denoiser =
        NumericalProfileId::new(QwenImage21TransformerFamily::METAL_NATIVE_BLOCK_PROFILE_ID)
            .unwrap();
    let (families, copies) =
        super::super::composition::prefix_command_families(&config, &profiles).unwrap();
    assert_eq!(copies, 2);
    for (family, phase) in families.iter().zip([
        QwenImage21TransformerPhase::PrefixPrefill,
        QwenImage21TransformerPhase::PrefixDecode,
    ]) {
        let checked: QwenImage21TransformerConfig =
            serde_json::from_value(family.canonical_config().clone()).unwrap();
        assert_eq!(checked.execution_phase(), phase);
        assert_eq!(checked.weight_encodings(), &encodings);
        assert!(family
            .program()
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .any(|node| node.operation_id.as_str() == DENSE_LINEAR_BLOCK_BF16_OPERATION_ID));
    }
    assert_eq!(pipeline.runtime.snapshot().uploaded_bytes, 0);
}

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
    assert!(
        pm.static_bytes() > dm.static_bytes(),
        "decode consumes a strict subset but still retains the full snapshot"
    );
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
        ImageResidencyPolicy::SelectedStages {
            stages: ImageResidentStageSet {
                vision_encoder: true,
                vae_encoder: true,
                denoiser: true,
                decoder: true,
                ..ImageResidentStageSet::default()
            },
        },
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
    // The remaining tests exercise the original all-component idle allowances;
    // do not let the selected subset above silently change those contracts.
    pipeline.config.residency = ImageResidencyPolicy::AllModelComponents;
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
    // Native declarations occupy the same total ceiling, even with a cold
    // handoff. Do not discount either cache or charge opaque storage twice.
    pipeline.config.residency = ImageResidencyPolicy::Cold;
    pipeline.config.dynamic_residency = ImageDynamicResidencyPolicy::Release;
    let opaque = 4096;
    pipeline.device_opaque_allowance_bytes = opaque;
    pipeline.config.device_capacity_bytes = transition + opaque - 1;
    assert!(pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .is_err());
    pipeline.config.device_capacity_bytes = transition + opaque;
    pipeline
        .validate_capacity_with_prefix(&[&prefill, &decode], Some((&prefill, &decode)))
        .unwrap();
    assert_eq!(pipeline.runtime.snapshot().uploaded_bytes, 0);
    assert_eq!(pipeline.runtime.snapshot().live_allocations, 0);
}
