//! Model/schema declarations only. Backend qualification uses the real Stored
//! mixed-GDN/full-attention consumer, not these metadata assertions.
use super::*;

#[test]
fn visual_profile_keeps_weights_and_gdn_but_declares_precise_input_roles_and_attention() {
    let config = test_config();
    let provider = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let definition = provider
        .define(&serde_json::to_value(config).unwrap())
        .unwrap();
    let raw = provider
        .prepare(&definition, &F16_NUMERICAL_PROFILE_ID.parse().unwrap())
        .unwrap();
    let visual = provider
        .prepare(
            &definition,
            &F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID.parse().unwrap(),
        )
        .unwrap();
    assert_eq!(raw.weights(), visual.weights());
    let conditioning = visual
        .program()
        .checkpoint_inputs()
        .unwrap()
        .conditioning_inputs();
    assert_eq!(
        conditioning,
        &[
            "value.input.visual_rows",
            "value.input.visual_replace_mask",
            "value.input.rope_cos",
            "value.input.rope_sin",
        ]
        .into_iter()
        .map(|id| value_id(id).unwrap())
        .collect()
    );
    let raw_nodes = &raw.program().blocks()[0].nodes;
    let nodes = &visual.program().blocks()[0].nodes;
    assert_eq!(raw_nodes[0], nodes[0]);
    assert_eq!(
        nodes[1].operation_id.as_str(),
        ferrum_runtime::ROW_SELECT_F16_OPERATION_ID
    );
    let full = nodes
        .iter()
        .find(|n| {
            n.operation_id.as_str()
                == ferrum_runtime::CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID
        })
        .unwrap();
    assert_eq!(full.inputs[9].as_str(), "value.input.rope_cos");
    assert_eq!(full.inputs[10].as_str(), "value.input.rope_sin");
    assert_eq!(
        full.attributes[&AttributeId::new("rope_interleaved").unwrap()],
        SemanticValue::Bool(false)
    );
    for node in nodes
        .iter()
        .filter(|n| n.operation_id.as_str() == GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID)
    {
        let original = raw_nodes.iter().find(|n| n.id == node.id).unwrap();
        assert_eq!(node.operation_id, original.operation_id);
        assert_eq!(node.attributes, original.attributes);
        assert_eq!(node.inputs[1..], original.inputs[1..]);
        assert!(node.inputs.iter().all(|i| !conditioning.contains(i)));
    }
    assert!(visual
        .program()
        .states()
        .iter()
        .all(|s| s.checkpoint == ferrum_runtime::StateCheckpointCapability::Unsupported));
    assert!(raw
        .program()
        .states()
        .iter()
        .all(|s| s.checkpoint != ferrum_runtime::StateCheckpointCapability::Unsupported));
}

#[test]
fn visual_profile_cannot_be_selected_for_native_block_quantized_sources() {
    let config = test_dense_gguf_config();
    let provider = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let definition = provider
        .define(&serde_json::to_value(config).unwrap())
        .unwrap();
    assert!(provider
        .prepare(
            &definition,
            &F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID.parse().unwrap()
        )
        .is_err());
}
