use super::*;

type Family = Qwen3VLTextEncoderFamily;

fn prepared(config: &Qwen3VLTextConfig) -> PreparedModelFamily {
    TypedFamilyRegistration::new(Family::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(Family::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap()
}

#[test]
fn empty_selection_preserves_canonical_identity_program_and_weight_inventory() {
    let base = tiny();
    let empty = base.clone().with_hidden_state_indices(Vec::new()).unwrap();
    let wire = serde_json::to_value(&base).unwrap();
    assert!(wire.get("hidden_state_indices").is_none());
    assert_eq!(
        serde_json::from_value::<Qwen3VLTextConfig>(wire.clone()).unwrap(),
        base
    );
    assert_eq!(serde_json::to_value(&empty).unwrap(), wire);
    let base_family = prepared(&base);
    let empty_family = prepared(&empty);
    assert_eq!(
        base_family.fingerprint().unwrap(),
        empty_family.fingerprint().unwrap()
    );
    assert_eq!(base_family.program(), empty_family.program());
    assert_eq!(base_family.weights(), empty_family.weights());
    assert_eq!(
        base_family.program().outputs(),
        &[ProgramValueId::new(Family::OUTPUT_ID).unwrap()]
    );
    assert!(base
        .expected_weights()
        .unwrap()
        .iter()
        .all(|w| w.external_name() != Family::FINAL_NORM_EXTERNAL_NAME));
}

#[test]
fn selected_states_change_identity_and_final_norm_is_required_only_for_last_index() {
    let base = tiny();
    let middle = base.clone().with_hidden_state_indices(vec![1]).unwrap();
    let selected = base.clone().with_hidden_state_indices(vec![1, 2]).unwrap();
    let base_family = prepared(&base);
    let middle_family = prepared(&middle);
    let selected_family = prepared(&selected);
    assert_ne!(
        base_family.fingerprint().unwrap(),
        middle_family.fingerprint().unwrap()
    );
    assert_ne!(
        middle_family.fingerprint().unwrap(),
        selected_family.fingerprint().unwrap()
    );
    assert_eq!(base_family.weights(), middle_family.weights());
    let norm = selected
        .expected_weights()
        .unwrap()
        .into_iter()
        .find(|w| w.external_name() == Family::FINAL_NORM_EXTERNAL_NAME)
        .unwrap();
    assert_eq!(norm.dimensions(), &[base.architecture().hidden_size()]);
    assert_eq!(norm.bytes(), base.architecture().hidden_size() * 2);
    let graph = selected_family.program();
    let mid_id = Family::hidden_state_output_id(1).unwrap();
    let last_id = Family::hidden_state_output_id(2).unwrap();
    assert_eq!(
        graph.outputs(),
        &[
            ProgramValueId::new(Family::OUTPUT_ID).unwrap(),
            mid_id.clone(),
            last_id.clone()
        ]
    );
    let middle_node = graph.blocks()[0].nodes.last().unwrap();
    assert_eq!(middle_node.outputs, [mid_id.clone()]);
    assert_eq!(graph.blocks()[1].nodes[0].inputs[0], mid_id);
    let norm_node = graph.blocks().last().unwrap().nodes.last().unwrap();
    assert_eq!(
        norm_node.operation_id.as_str(),
        HEAD_RMS_NORM_BF16_OPERATION_ID
    );
    assert_eq!(
        norm_node.inputs,
        [
            ProgramValueId::new(Family::OUTPUT_ID).unwrap(),
            weights::value_id(Family::FINAL_NORM_EXTERNAL_NAME).unwrap()
        ]
    );
    assert_eq!(norm_node.outputs, [last_id.clone()]);
    assert_eq!(
        selected_family.numerical_profile().boundaries[&last_id],
        ElementType::Bf16
    );
    let serialized = serde_json::to_value(&selected).unwrap();
    assert_eq!(serialized["hidden_state_indices"], json!([1, 2]));
    assert_eq!(
        serde_json::from_value::<Qwen3VLTextConfig>(serialized).unwrap(),
        selected
    );
}

#[test]
fn selected_state_indices_reject_embedding_out_of_range_duplicates_and_reordering() {
    for indices in [vec![0], vec![3], vec![1, 1], vec![2, 1], vec![1, u64::MAX]] {
        assert!(tiny().with_hidden_state_indices(indices.clone()).is_err());
        let mut wire = serde_json::to_value(tiny()).unwrap();
        wire["hidden_state_indices"] = json!(indices);
        assert!(serde_json::from_value::<Qwen3VLTextConfig>(wire).is_err());
    }
    assert!(Family::hidden_state_output_id(0).is_err());
}

#[test]
fn deepstack_hidden_state_ports_are_after_injection_including_last_layer() {
    let config = Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&hf(3, 8, 12, 4, 2, 2)).unwrap(),
        layout(true),
    )
    .unwrap()
    .with_deepstack(
        Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(1, 1).unwrap()]).unwrap(),
    )
    .unwrap()
    .with_hidden_state_indices(vec![1, 2, 3])
    .unwrap();
    let family = prepared(&config);
    for (layer, block) in family.program().blocks()[..3].iter().enumerate() {
        let injected = block.nodes.last().unwrap();
        assert_eq!(
            injected.operation_id.as_str(),
            edit::SCATTER_ADD_OPERATION_ID
        );
        let expected = if layer == 2 {
            ProgramValueId::new(Family::OUTPUT_ID).unwrap()
        } else {
            Family::hidden_state_output_id(layer as u64 + 1).unwrap()
        };
        assert_eq!(injected.outputs, [expected.clone()]);
        let next = &family.program().blocks()[layer + 1].nodes[0];
        assert_eq!(next.inputs[0], expected);
    }
}

mod execution;
