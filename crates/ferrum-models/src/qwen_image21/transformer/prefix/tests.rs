use super::*;
use serde_json::json;
type F = QwenImage21TransformerFamily;

fn config() -> QwenImage21TransformerConfig {
    // Two samples share geometry but the second has two invalid prefix keys.
    let transformer = QwenImage21Config::from_hf_json(
        &serde_json::to_vec(&json!({
            "_class_name":"QwenImage21Transformer2DModel","num_layers":2,
            "num_attention_heads":1,"attention_head_dim":128,"context_in_dim":128,
            "in_channels":3,"out_channels":4,"patch_size":1,"mlp_ratio":3,
            "axes_dims_rope":[16,56,56],"causal_condition":true,"eps":0.000001
        }))
        .unwrap(),
        StructureLimits::default(),
    )
    .unwrap();
    QwenImage21TransformerConfig::new(
        transformer,
        QwenImage21T2iLayout::new(7, 2, 2, vec![7, 5]).unwrap(),
    )
    .unwrap()
    .with_reference_image(2, 2, 2, 4)
    .unwrap()
}
fn graph(config: &QwenImage21TransformerConfig, name: &str) -> ModelProgram {
    let f = F::new().unwrap();
    let profiles = f.numerical_profiles(config).unwrap();
    let profile = profiles.resolve(&profile_id(name).unwrap()).unwrap();
    let graph = f.semantic_program(config, profile).unwrap();
    profile.validate_program(&graph).unwrap();
    graph
}
fn value(s: &str) -> ProgramValueId {
    ProgramValueId::new(s).unwrap()
}
fn attr<'a>(node: &'a ProgramNode, name: &str) -> &'a SemanticValue {
    &node.attributes[&AttributeId::new(name).unwrap()]
}

#[test]
fn prefix_phases_preserve_default_wire_and_share_exact_weight_schema() {
    let full = config();
    let wire = serde_json::to_value(&full).unwrap();
    assert!(wire.get("phase").is_none());
    assert_eq!(
        serde_json::from_value::<QwenImage21TransformerConfig>(wire).unwrap(),
        full
    );
    let mut fingerprints = BTreeSet::new();
    let family = F::new().unwrap();
    let weights = family.weights(&full).unwrap();
    for phase in [
        QwenImage21TransformerPhase::Full,
        QwenImage21TransformerPhase::PrefixPrefill,
        QwenImage21TransformerPhase::PrefixDecode,
    ] {
        let c = full.clone().with_execution_phase(phase).unwrap();
        assert_eq!(family.weights(&c).unwrap(), weights);
        assert_eq!(
            F::fp8_weight_groups(&c).unwrap(),
            F::fp8_weight_groups(&full).unwrap()
        );
        assert_eq!(
            serde_json::from_value::<QwenImage21TransformerConfig>(
                serde_json::to_value(&c).unwrap()
            )
            .unwrap(),
            c
        );
        for name in [
            F::REFERENCE_PROFILE_ID,
            F::CUDA_PROFILE_ID,
            F::CUDA_FUSED_PROFILE_ID,
            F::CUDA_FP8_FUSED_PROFILE_ID,
        ] {
            let program = graph(&c, name);
            assert_eq!(
                serde_json::from_value::<ModelProgram>(serde_json::to_value(&program).unwrap())
                    .unwrap(),
                program
            );
        }
        fingerprints.insert(graph(&c, F::REFERENCE_PROFILE_ID).fingerprint().unwrap());
    }
    assert_eq!(
        fingerprints.len(),
        3,
        "distinct phase programs must not share an execution identity"
    );
    assert!(super::super::tests::config(false)
        .with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)
        .is_err());
    assert!(super::super::tests::config(false)
        .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
        .is_err());
}

#[test]
fn text_entrance_is_logically_phase_optional_but_physically_required_and_fully_counted() {
    let full = config();
    let family = F::new().unwrap();
    let expected =
        ExpectedWeights::from_config(full.transformer(), StructureLimits::default()).unwrap();
    let original = expected.weight_schema().unwrap();
    let mut identity = None;
    for phase in [
        QwenImage21TransformerPhase::Full,
        QwenImage21TransformerPhase::PrefixPrefill,
        QwenImage21TransformerPhase::PrefixDecode,
    ] {
        let config = full.clone().with_execution_phase(phase).unwrap();
        let prepared = TypedFamilyRegistration::new(F::new().unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&config).unwrap(),
                &profile_id(F::REFERENCE_PROFILE_ID).unwrap(),
            )
            .unwrap();
        let schema = prepared.stored_weight_schema().unwrap();
        assert_eq!(schema.components, original.components);
        assert!(schema.components.iter().all(|component| component.required));
        let bytes: u64 = schema
            .components
            .iter()
            .filter(|component| component.required)
            .map(|component| component.physical_bytes().unwrap())
            .sum();
        assert_eq!(bytes, expected.payload_bytes());
        let optional = schema
            .tensors
            .iter()
            .filter(|tensor| !tensor.required)
            .map(|tensor| tensor.id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            optional,
            BTreeSet::from([
                "weight.qwen_image21.txt_in.text_norm.weight",
                "weight.qwen_image21.txt_in.in_layer.weight",
                "weight.qwen_image21.txt_in.out_layer.weight",
            ])
        );
        for name in optional {
            let reference = prepared
                .program()
                .weights()
                .iter()
                .find(|weight| weight.weight_id.as_str() == name);
            if phase == QwenImage21TransformerPhase::PrefixDecode {
                assert!(reference.is_none());
            } else {
                let reference = reference.expect("full/prefill must bind the text weight");
                assert!(prepared
                    .program()
                    .blocks()
                    .iter()
                    .flat_map(|block| &block.nodes)
                    .any(|node| node.inputs.contains(&reference.value_id)));
            }
        }
        let weights = ExecutionWeightPlan::identity(&prepared).unwrap();
        if let Some(original) = &identity {
            assert_eq!(&weights, original);
        } else {
            identity = Some(weights);
        }
        assert_eq!(family.weights(&config).unwrap(), prepared.weights().clone());
    }
}

#[test]
fn prefill_captures_rotated_keys_and_unmodified_values_into_compact_per_sample_ports() {
    let c = config()
        .with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)
        .unwrap();
    let program = graph(&c, F::REFERENCE_PROFILE_ID);
    let nodes = &program.blocks()[0].nodes;
    let ports = c.prefix_cache_ports().unwrap();
    assert_eq!(program.outputs()[0], value(F::VELOCITY_ID));
    assert_eq!(
        program.outputs()[1..],
        ports.iter().map(|p| p.value_id.clone()).collect::<Vec<_>>()
    );
    assert!(program.states().is_empty());
    for port in &ports {
        assert_eq!(port.tensor.dimensions, [26, 128]);
        assert_eq!(
            c.compile_options().unwrap().tensor_specs()[&port.value_id],
            port.tensor
        );
        let capture = nodes
            .iter()
            .find(|n| n.outputs == [port.value_id.clone()])
            .unwrap();
        assert_eq!(capture.operation_id.as_str(), ROW_GATHER_BF16_OPERATION_ID);
        assert_eq!(
            attr(capture, "run_lengths"),
            &SemanticValue::Integers(vec![13])
        );
        assert_eq!(
            attr(capture, "source_batch_strides"),
            &SemanticValue::Integers(vec![17])
        );
        assert_eq!(
            attr(capture, "source_starts"),
            &SemanticValue::Integers(vec![0])
        );
        let producer = nodes
            .iter()
            .find(|n| n.outputs.contains(&capture.inputs[0]))
            .unwrap();
        assert!(producer.id.as_str().starts_with(&format!(
            "node.qwen_image21.transformer.layer{}.",
            port.layer
        )));
        match port.part {
            QwenImage21PrefixPart::Key => {
                assert_eq!(
                    producer.operation_id.as_str(),
                    ADJACENT_ROPE_BF16_OPERATION_ID
                );
                let norm = nodes
                    .iter()
                    .find(|n| n.outputs.contains(&producer.inputs[0]))
                    .unwrap();
                assert_eq!(norm.operation_id.as_str(), HEAD_RMS_NORM_BF16_OPERATION_ID);
            }
            QwenImage21PrefixPart::Value => {
                assert_eq!(
                    producer.operation_id.as_str(),
                    DENSE_LINEAR_BF16_OPERATION_ID
                );
                let weight = program
                    .weights()
                    .iter()
                    .find(|w| w.value_id == producer.inputs[1])
                    .unwrap();
                assert!(weight.weight_id.as_str().ends_with(".attn.to_v.weight"));
            }
        }
    }
    assert_eq!(c.prefix_cache_bytes().unwrap(), 2 * 2 * 2 * 13 * 128 * 2);
}

#[test]
fn decode_reads_request_prefix_before_target_kv_and_preserves_padding() {
    let c = config()
        .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
        .unwrap();
    let program = graph(&c, F::CUDA_FP8_FUSED_PROFILE_ID);
    let nodes = &program.blocks()[0].nodes;
    let input_specs = c.input_tensor_specs().unwrap();
    assert_eq!(input_specs.len(), 3);
    assert!(!input_specs.contains_key(&value(F::TEXT_ID)));
    assert!(!input_specs.contains_key(&value(F::REFERENCE_LATENTS_ID)));
    assert_eq!(input_specs[&value(F::LATENTS_ID)].dimensions, [8, 3]);
    assert_eq!(
        input_specs[&value(F::FREQUENCIES_ID)].dimensions,
        [4, 64, 2]
    );
    assert_eq!(program.outputs(), [value(F::VELOCITY_ID)]);
    for port in c.prefix_cache_ports().unwrap() {
        let state = program
            .states()
            .iter()
            .find(|s| s.value_id == port.value_id)
            .unwrap();
        assert_eq!(state, &port.state_spec().unwrap());
        assert_eq!(state.lifetime, StateLifetime::Request);
        let consumers = nodes
            .iter()
            .filter(|n| n.inputs.contains(&state.value_id))
            .collect::<Vec<_>>();
        assert_eq!(consumers.len(), 1);
        let concat = consumers[0];
        assert_eq!(
            concat.operation_id.as_str(),
            BATCHED_ROW_CONCAT_BF16_OPERATION_ID
        );
        assert_eq!(concat.inputs[0], state.value_id);
        let attention = nodes
            .iter()
            .find(|n| n.inputs.contains(&concat.outputs[0]))
            .unwrap();
        assert_eq!(
            attention.operation_id.as_str(),
            CROSS_ATTENTION_FUSED_BF16_OPERATION_ID
        );
        assert_eq!(
            attr(attention, "key_run_counts"),
            &SemanticValue::Integers(vec![1, 3])
        );
        assert_eq!(
            attr(attention, "key_run_lengths"),
            &SemanticValue::Integers(vec![17, 11, 2, 4])
        );
        assert_eq!(
            attr(attention, "key_run_valid"),
            &SemanticValue::Integers(vec![1, 1, 0, 1])
        );
        assert!(!attention
            .attributes
            .contains_key(&AttributeId::new("segment_lengths").unwrap()));
        assert_eq!(
            c.compile_options().unwrap().tensor_specs()[&concat.outputs[0]].dimensions,
            [34, 128]
        );
    }
    for node in nodes {
        assert!(!node.inputs.iter().any(|v| v.as_str().contains("txt_in.")));
        assert!(!node
            .outputs
            .iter()
            .any(|v| program.states().iter().any(|s| s.value_id == *v)));
    }
    let conditioning = nodes
        .iter()
        .find(|n| n.outputs == [value(program::CONDITIONING[0])])
        .unwrap();
    assert_eq!(
        attr(conditioning, "run_lengths"),
        &SemanticValue::Integers(vec![4])
    );
    assert_eq!(
        attr(conditioning, "source_starts"),
        &SemanticValue::Integers(vec![0])
    );
    let target = nodes
        .iter()
        .find(|n| n.outputs == [value(program::TARGET_HIDDEN)])
        .unwrap();
    assert_eq!(
        attr(target, "source_starts"),
        &SemanticValue::Integers(vec![0])
    );
    assert_eq!(
        attr(target, "source_batch_strides"),
        &SemanticValue::Integers(vec![4])
    );
}

#[test]
fn decode_rotary_is_exact_original_joint_target_slice() {
    let full = config();
    let decode = full
        .clone()
        .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)
        .unwrap();
    let rope = full.rotary_frequencies().unwrap();
    let mut original = vec![0.0; rope.output_elements(full.layout()).unwrap()];
    rope.write_into(full.layout(), &mut original).unwrap();
    let mut target = vec![0.0; rope.phase_output_elements(&decode).unwrap()];
    rope.write_phase_into(&decode, &mut target).unwrap();
    let start = (full.layout().target_start() * full.transformer().head_dim()) as usize;
    assert_eq!(
        target.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        original[start..]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let mut bad = vec![f32::from_bits(0x12345678); target.len() - 1];
    assert!(rope.write_phase_into(&decode, &mut bad).is_err());
    assert!(bad.iter().all(|v| v.to_bits() == 0x12345678));
}
