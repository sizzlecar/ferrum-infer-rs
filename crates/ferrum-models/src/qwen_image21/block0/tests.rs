use super::*;
use crate::qwen_image21::{
    QwenImage21AttentionKeyRun, QwenImage21AttentionLayout, QwenImage21AttentionSegment,
    StructureLimits,
};
use serde_json::json;

type F = QwenImage21Block0Family;

#[test]
fn fused_block_profile_changes_attention_only_and_keeps_all_boundaries() {
    let config = QwenImage21Block0Config::new(
        QwenImage21AttentionConfig::new(
            transformer(1, 128, 0.75),
            QwenImage21AttentionLayout::uniform(2, false).unwrap(),
        )
        .unwrap(),
        QwenImage21Block0Broadcast::Elementwise,
    )
    .unwrap();
    let family = F::new().unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    let fused_id = profile_id(F::CUDA_FUSED_PROFILE_ID).unwrap();
    let original = family
        .semantic_program(
            &config,
            profiles
                .resolve(&profile_id(F::CUDA_PROFILE_ID).unwrap())
                .unwrap(),
        )
        .unwrap();
    let profile = profiles.resolve(&fused_id).unwrap();
    let candidate = family.semantic_program(&config, profile).unwrap();
    profile.validate_program(&candidate).unwrap();
    assert_eq!(candidate.outputs(), &F::OUTPUT_IDS.map(value));
    assert_eq!(original.weights(), candidate.weights());
    assert_eq!(
        original.input_tensor_specs(),
        candidate.input_tensor_specs()
    );
    assert!(!profiles.auto_preference().contains(&fused_id));
    let mut changed = 0;
    for (left, right) in original.blocks()[0]
        .nodes
        .iter()
        .zip(&candidate.blocks()[0].nodes)
    {
        let mut restored = right.clone();
        if right.operation_id.as_str() == STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID {
            assert_eq!(
                left.operation_id.as_str(),
                STRUCTURED_ATTENTION_BF16_OPERATION_ID
            );
            restored.operation_id = left.operation_id.clone();
            changed += 1;
        }
        assert_eq!(&restored, left);
    }
    assert_eq!(changed, 1);
    TypedFamilyRegistration::new(F::new().unwrap())
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &fused_id)
        .unwrap();
}

fn transformer(heads: u64, head_dim: u64, epsilon: f64) -> QwenImage21Config {
    QwenImage21Config::from_hf_json(
        &serde_json::to_vec(&json!({
            "_class_name":"QwenImage21Transformer2DModel", "num_layers":1,
            "num_attention_heads":heads, "attention_head_dim":head_dim,
            "context_in_dim":8, "in_channels":3, "out_channels":4,
            "patch_size":1, "mlp_ratio":3, "axes_dims_rope":[2,2,head_dim-4],
            "causal_condition":true, "eps":epsilon,
        }))
        .unwrap(),
        StructureLimits::default(),
    )
    .unwrap()
}

fn config(mode: QwenImage21Block0Broadcast, full: bool) -> QwenImage21Block0Config {
    QwenImage21Block0Config::new(
        QwenImage21AttentionConfig::new(
            transformer(2, 8, 0.75),
            QwenImage21AttentionLayout::uniform(2, full).unwrap(),
        )
        .unwrap(),
        mode,
    )
    .unwrap()
}

fn value(name: &str) -> ProgramValueId {
    ProgramValueId::new(name).unwrap()
}

#[test]
fn complete_block_reuses_branches_and_retains_raw_gate_and_both_residual_boundaries() {
    let config = config(QwenImage21Block0Broadcast::Row, false);
    let family = F::new().unwrap();
    for profile in family.numerical_profiles(&config).unwrap().profiles() {
        let program = family.semantic_program(&config, profile).unwrap();
        assert_eq!(program.inputs(), &F::INPUT_IDS.map(value));
        assert_eq!(program.outputs(), &F::OUTPUT_IDS.map(value));
        assert!(program.states().is_empty());
        assert_eq!(program.weights().len(), 9);
        assert_eq!(profile.boundaries.len(), 21);
        assert_eq!(profile.primary_activation, value(F::OUTPUT_ID));
        for output in program.outputs() {
            assert_eq!(profile.boundaries[output], ElementType::Bf16);
        }
        let nodes = &program.blocks()[0].nodes;
        assert_eq!(nodes.len(), 21);
        assert!(nodes
            .iter()
            .all(|node| node.work == ProgramNodeWorkSpec::Fixed));
        for (index, operation, inputs) in [
            (0, LAYER_NORM_BF16_OPERATION_ID, vec![F::INPUT_ID]),
            (
                1,
                MODULATION_SCALE_BF16_OPERATION_ID,
                vec![F::NORMALIZED1_ID, F::SCALE1_ID],
            ),
            (2, TANH_BF16_OPERATION_ID, vec![F::GATE1_ID]),
            (
                12,
                GATED_RESIDUAL_BF16_OPERATION_ID,
                vec![F::INPUT_ID, F::TANH_GATE1_ID, Attention::OUTPUT_ID],
            ),
            (13, LAYER_NORM_BF16_OPERATION_ID, vec![F::RESIDUAL1_ID]),
            (
                14,
                MODULATION_SCALE_BF16_OPERATION_ID,
                vec![F::NORMALIZED2_ID, F::SCALE2_ID],
            ),
            (15, TANH_BF16_OPERATION_ID, vec![F::GATE2_ID]),
            (
                20,
                GATED_RESIDUAL_BF16_OPERATION_ID,
                vec![F::RESIDUAL1_ID, F::TANH_GATE2_ID, Mlp::OUTPUT_ID],
            ),
        ] {
            assert_eq!(nodes[index].operation_id.as_str(), operation);
            assert_eq!(
                nodes[index].inputs,
                inputs.into_iter().map(value).collect::<Vec<_>>()
            );
        }
        for index in [1, 12, 14, 20] {
            assert_eq!(
                nodes[index].attributes,
                BTreeMap::from([(
                    AttributeId::new("broadcast").unwrap(),
                    SemanticValue::Text("row".into())
                )])
            );
        }
        for index in [0, 13] {
            assert_eq!(
                nodes[index].attributes[&AttributeId::new("epsilon").unwrap()],
                SemanticValue::Rational(config.transformer().epsilon())
            );
            assert_eq!(nodes[index].attributes.len(), 2); // no affine or shift
        }
        for index in [2, 15] {
            assert!(nodes[index].attributes.is_empty());
        }

        // Each branch is reused exactly except for its explicit entrance map and
        // enclosing node identity. This catches accidental loss of QK rounding,
        // the structured visibility layout or the MLP's SiLU-multiply boundary.
        let (_, attention_name, mlp_name) = profile_sources()
            .into_iter()
            .find(|(name, _, _)| *name == profile.id.as_str())
            .unwrap();
        let attention = Attention::new().unwrap();
        let profiles = attention.numerical_profiles(config.attention()).unwrap();
        let branch = attention
            .semantic_program(
                config.attention(),
                profiles
                    .resolve(&profile_id(attention_name).unwrap())
                    .unwrap(),
            )
            .unwrap();
        assert_branch(
            &nodes[3..12],
            &branch,
            &[
                (Attention::INPUT_ID, F::MODULATED1_ID),
                (Attention::FREQUENCIES_ID, F::FREQUENCIES_ID),
            ],
        );
        let mlp = Mlp::new().unwrap();
        let profiles = mlp.numerical_profiles(config.transformer()).unwrap();
        let branch = mlp
            .semantic_program(
                config.transformer(),
                profiles.resolve(&profile_id(mlp_name).unwrap()).unwrap(),
            )
            .unwrap();
        assert_branch(
            &nodes[16..20],
            &branch,
            &[(Mlp::INPUT_ID, F::MODULATED2_ID)],
        );
    }
}

fn assert_branch(actual: &[ProgramNode], branch: &ModelProgram, mapping: &[(&str, &str)]) {
    let source = branch
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), source.len());
    for (actual, source) in actual.iter().zip(source) {
        let mut expected = source.clone();
        assert_ne!(actual.id, expected.id);
        expected.id = actual.id.clone();
        for input in &mut expected.inputs {
            if let Some((_, to)) = mapping.iter().find(|(from, _)| input.as_str() == *from) {
                *input = value(to);
            }
        }
        assert_eq!(actual, &expected);
    }
}

#[test]
fn six_input_specs_bind_batch_sequence_width_dtype_and_broadcast() {
    let layout = QwenImage21AttentionLayout::new(
        3,
        vec![QwenImage21AttentionSegment::new(3, true)],
        vec![vec![QwenImage21AttentionKeyRun::new(3, true)]; 2],
    )
    .unwrap();
    let attention = QwenImage21AttentionConfig::new(transformer(2, 8, 0.75), layout).unwrap();
    for (mode, parameter_rows) in [
        (QwenImage21Block0Broadcast::Row, 1),
        (QwenImage21Block0Broadcast::Elementwise, 6),
    ] {
        let config = QwenImage21Block0Config::new(attention.clone(), mode).unwrap();
        let specs = config.input_tensor_specs().unwrap();
        assert_eq!(specs.len(), 6);
        assert_eq!(specs[&value(F::INPUT_ID)].dimensions, vec![6, 16]);
        for id in [F::SCALE1_ID, F::GATE1_ID, F::SCALE2_ID, F::GATE2_ID] {
            assert_eq!(specs[&value(id)].dimensions, vec![parameter_rows, 16]);
            assert_eq!(specs[&value(id)].element_type, ElementType::Bf16);
        }
        assert_eq!(specs[&value(F::FREQUENCIES_ID)].dimensions, vec![3, 4, 2]);
        assert_eq!(
            specs[&value(F::FREQUENCIES_ID)].element_type,
            ElementType::F32
        );
        let family = F::new().unwrap();
        let profiles = family.numerical_profiles(&config).unwrap();
        let program = family
            .semantic_program(&config, &profiles.profiles()[0])
            .unwrap();
        assert_eq!(program.input_tensor_specs(), &specs);
        for index in [1, 12, 14, 20] {
            assert_eq!(
                program.blocks()[0].nodes[index].attributes
                    [&AttributeId::new("broadcast").unwrap()],
                SemanticValue::Text(mode.as_str().into())
            );
        }
    }
}

#[test]
fn real_block_schema_has_only_the_nine_reused_weights_and_no_affine_or_modulation_weights() {
    let config = QwenImage21Block0Config::new(
        QwenImage21AttentionConfig::new(
            transformer(32, 128, 1e-6),
            QwenImage21AttentionLayout::uniform(1, false).unwrap(),
        )
        .unwrap(),
        QwenImage21Block0Broadcast::Row,
    )
    .unwrap();
    let ModelWeights::Stored(schema) = F::new().unwrap().weights(&config).unwrap() else {
        panic!("Stored required")
    };
    let ModelWeights::Stored(attention) = Attention::new()
        .unwrap()
        .weights(config.attention())
        .unwrap()
    else {
        panic!("Stored required")
    };
    let ModelWeights::Stored(mlp) = Mlp::new().unwrap().weights(config.transformer()).unwrap()
    else {
        panic!("Stored required")
    };
    assert_eq!(schema.components.len(), 9);
    assert_eq!(schema.tensors.len(), 9);
    assert!(schema.components.iter().all(|c| c.required));
    assert_eq!(
        schema
            .components
            .iter()
            .map(|c| c.physical_bytes().unwrap())
            .sum::<u64>(),
        436_208_128
    );
    assert_eq!(
        schema.components,
        attention
            .components
            .into_iter()
            .chain(mlp.components)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        schema.tensors,
        attention
            .tensors
            .into_iter()
            .chain(mlp.tensors)
            .collect::<Vec<_>>()
    );
    for name in F::WEIGHT_NAMES {
        assert!(schema
            .tensor(&WeightId::new(format!("weight.qwen_image21.{name}")).unwrap())
            .is_some());
    }
}

#[test]
fn broadcast_and_attention_layout_are_explicit_trusted_family_identity() {
    let registration = TypedFamilyRegistration::new(F::new().unwrap());
    let prepare = |config: QwenImage21Block0Config| {
        registration
            .prepare_with_profile(
                &serde_json::to_value(config).unwrap(),
                &profile_id(F::REFERENCE_PROFILE_ID).unwrap(),
            )
            .unwrap()
    };
    let row = prepare(config(QwenImage21Block0Broadcast::Row, false));
    let elementwise = prepare(config(QwenImage21Block0Broadcast::Elementwise, false));
    let full = prepare(config(QwenImage21Block0Broadcast::Row, true));
    for other in [&elementwise, &full] {
        assert_ne!(row.config_fingerprint(), other.config_fingerprint());
        assert_ne!(row.fingerprint().unwrap(), other.fingerprint().unwrap());
        assert_ne!(row.program(), other.program());
        assert_eq!(row.weights(), other.weights());
    }
}

#[test]
fn config_rejects_unknown_semantics_and_unsupported_layer_norm_epsilon() {
    let config = config(QwenImage21Block0Broadcast::Row, false);
    let encoded = serde_json::to_value(&config).unwrap();
    assert_eq!(
        serde_json::from_value::<QwenImage21Block0Config>(encoded.clone()).unwrap(),
        config
    );
    for (field, extra) in [
        ("shift", json!(0)),
        ("gate_is_tanh", json!(true)),
        ("broadcast", json!("inferred")),
    ] {
        let mut mutated = encoded.clone();
        mutated[field] = extra;
        assert!(serde_json::from_value::<QwenImage21Block0Config>(mutated).is_err());
    }
    let mut missing = encoded;
    missing.as_object_mut().unwrap().remove("broadcast");
    assert!(serde_json::from_value::<QwenImage21Block0Config>(missing).is_err());
    for epsilon in [1e-13, 2.0] {
        let attention = QwenImage21AttentionConfig::new(
            transformer(2, 8, epsilon),
            QwenImage21AttentionLayout::uniform(1, false).unwrap(),
        )
        .unwrap();
        assert!(QwenImage21Block0Config::new(attention, QwenImage21Block0Broadcast::Row).is_err());
    }
}

#[test]
fn profile_must_preserve_every_operation_and_bf16_boundary() {
    let family = F::new().unwrap();
    let config = config(QwenImage21Block0Broadcast::Row, false);
    let profiles = family.numerical_profiles(&config).unwrap();
    let profile = profiles.profiles()[0].clone();
    let mut changed = profile.clone();
    changed.boundaries.remove(&value(F::TANH_GATE1_ID));
    assert!(family.semantic_program(&config, &changed).is_err());
    let mut changed = profile;
    changed
        .operations
        .iter_mut()
        .find(|operation| operation.operation_id.as_str() == GATED_RESIDUAL_BF16_OPERATION_ID)
        .unwrap()
        .accumulation_type = Some(ElementType::Bf16);
    assert!(family.semantic_program(&config, &changed).is_err());
    let partial = Attention::new()
        .unwrap()
        .numerical_profiles(config.attention())
        .unwrap();
    assert!(family
        .semantic_program(&config, &partial.profiles()[0])
        .is_err());
}
