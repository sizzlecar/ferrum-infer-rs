use super::*;
use serde_json::json;

fn transformer(heads: u64, head_dim: u64) -> QwenImage21Config {
    QwenImage21Config::from_hf_json(&serde_json::to_vec(&json!({"_class_name":"QwenImage21Transformer2DModel","num_layers":1,"num_attention_heads":heads,"attention_head_dim":head_dim,"context_in_dim":8,"in_channels":3,"out_channels":4,"patch_size":1,"mlp_ratio":3,"axes_dims_rope":[2,2,head_dim-4],"causal_condition":true,"eps":0.75})).unwrap(),StructureLimits::default()).unwrap()
}
fn config(full: bool) -> QwenImage21AttentionConfig {
    QwenImage21AttentionConfig::new(
        transformer(2, 8),
        QwenImage21AttentionLayout::uniform(2, full).unwrap(),
    )
    .unwrap()
}

#[test]
fn attention_family_reuses_frontend_and_preserves_all_nine_bf16_boundaries() {
    let config = config(false);
    let family = QwenImage21Block0AttentionFamily::new().unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    for profile in profiles.profiles() {
        let program = family.semantic_program(&config, profile).unwrap();
        assert_eq!(program.inputs().len(), 2);
        assert_eq!(program.outputs().len(), 9);
        assert_eq!(program.weights().len(), 6);
        let nodes = &program.blocks()[0].nodes;
        assert_eq!(nodes.len(), 9);
        assert_eq!(
            nodes[6].operation_id.as_str(),
            DENSE_LINEAR_BF16_OPERATION_ID
        );
        assert_eq!(
            nodes[7].operation_id.as_str(),
            STRUCTURED_ATTENTION_BF16_OPERATION_ID
        );
        assert_eq!(
            nodes[8].operation_id.as_str(),
            DENSE_LINEAR_BF16_OPERATION_ID
        );
        assert_eq!(
            nodes[7].inputs,
            [
                nodes[4].outputs[0].clone(),
                nodes[5].outputs[0].clone(),
                nodes[6].outputs[0].clone()
            ]
        );
        assert_eq!(nodes[8].inputs[0], nodes[7].outputs[0]);
        assert_eq!(nodes[7].attributes, config.attributes().unwrap());
        for output in program.outputs() {
            assert_eq!(profile.boundaries[output], ElementType::Bf16);
        }
        let frontend = Qk::with_frontend(QwenImage21QkFrontend::Rotary).unwrap();
        let source = frontend.numerical_profiles(config.transformer()).unwrap();
        let base = frontend
            .semantic_program(config.transformer(), &source.profiles()[0])
            .unwrap();
        assert_eq!(&nodes[..6], base.blocks()[0].nodes.as_slice());
        assert!(frontend
            .semantic_program(config.transformer(), profile)
            .is_err());
    }
    let real = QwenImage21AttentionConfig::new(
        transformer(32, 128),
        QwenImage21AttentionLayout::uniform(1, false).unwrap(),
    )
    .unwrap();
    let ModelWeights::Stored(schema) = family.weights(&real).unwrap() else {
        panic!("Stored required");
    };
    assert_eq!(schema.components.len(), 6);
    assert!(schema.components.iter().all(|c| c.required));
    assert_eq!(
        schema
            .components
            .iter()
            .map(|c| c.physical_bytes().unwrap())
            .sum::<u64>(),
        134_218_240
    );
}

#[test]
fn layout_is_canonical_config_and_changes_trusted_family_identity() {
    let family = QwenImage21Block0AttentionFamily::new().unwrap();
    let registration = TypedFamilyRegistration::new(family);
    let prepare = |config: &QwenImage21AttentionConfig| {
        registration
            .prepare_with_profile(
                &serde_json::to_value(config).unwrap(),
                &NumericalProfileId::new(QwenImage21Block0AttentionFamily::REFERENCE_PROFILE_ID)
                    .unwrap(),
            )
            .unwrap()
    };
    let causal = prepare(&config(false));
    let full = prepare(&config(true));
    assert_ne!(causal.config_fingerprint(), full.config_fingerprint());
    assert_ne!(causal.fingerprint().unwrap(), full.fingerprint().unwrap());
    assert_ne!(causal.program(), full.program());
    assert_eq!(causal.weights(), full.weights());
    let encoded = serde_json::to_value(config(false)).unwrap();
    assert_eq!(
        serde_json::from_value::<QwenImage21AttentionConfig>(encoded.clone()).unwrap(),
        config(false)
    );
    let mut changed = encoded.clone();
    changed["scale"] = json!({"numerator":1,"denominator":1});
    assert!(serde_json::from_value::<QwenImage21AttentionConfig>(changed).is_err());
    let mut changed = encoded.clone();
    changed["layout"]["hidden_mask"] = json!(true);
    assert!(serde_json::from_value::<QwenImage21AttentionConfig>(changed).is_err());
    let mut changed = encoded;
    changed["layout"]["segments"][0]["length"] = json!(1);
    assert!(serde_json::from_value::<QwenImage21AttentionConfig>(changed).is_err());
}

#[test]
fn layout_rejects_invalid_runs_and_keeps_adjacent_image_boundaries() {
    assert!(QwenImage21AttentionLayout::uniform(0, false).is_err());
    assert!(QwenImage21AttentionLayout::new(
        2,
        vec![QwenImage21AttentionSegment::new(2, false)],
        vec![vec![
            QwenImage21AttentionKeyRun::new(1, true),
            QwenImage21AttentionKeyRun::new(1, true)
        ]]
    )
    .is_err());
    let split = QwenImage21AttentionLayout::new(
        2,
        vec![
            QwenImage21AttentionSegment::new(1, true),
            QwenImage21AttentionSegment::new(1, true),
        ],
        vec![vec![QwenImage21AttentionKeyRun::new(2, true)]],
    )
    .unwrap();
    let split = QwenImage21AttentionConfig::new(transformer(2, 8), split).unwrap();
    let spec = ResolvedTensorSpec::new(
        vec![2, 16],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let geometry = StructuredAttentionBf16Geometry::new(
        &spec,
        &spec,
        &spec,
        &spec,
        &split.attributes().unwrap(),
    )
    .unwrap();
    assert_eq!(geometry.mask().visible_end(0), Some(1));
    let full = config(true);
    let geometry = StructuredAttentionBf16Geometry::new(
        &spec,
        &spec,
        &spec,
        &spec,
        &full.attributes().unwrap(),
    )
    .unwrap();
    assert_eq!(geometry.mask().visible_end(0), Some(2));
}

#[test]
fn scale_rounding_is_bound_as_an_exact_rational() {
    let scale = QwenImage21Block0AttentionFamily::scale_for_head_dim(128).unwrap();
    assert_eq!(
        ((scale.numerator() as f64 / scale.denominator() as f64) as f32).to_bits(),
        0x3db5_04f3
    );
    assert_eq!(
        scale,
        CanonicalRational::new(11_863_283, 134_217_728).unwrap()
    );
    assert_eq!(
        QwenImage21Block0AttentionFamily::scale_for_head_dim(256).unwrap(),
        CanonicalRational::new(1, 16).unwrap()
    );
    for dimension in [0, 1, 6, 512, u64::MAX] {
        assert!(QwenImage21Block0AttentionFamily::scale_for_head_dim(dimension).is_err());
    }
}
