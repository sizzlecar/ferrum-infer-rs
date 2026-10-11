use super::*;

#[test]
fn extra_fixture_declares_six_real_formats_and_preserves_state_and_old_profiles() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let original = Family::new(kind);
        let old_schema = original.weight_schema(&kind).unwrap();
        let old_profile = serde_json::to_vec(&original.numerical_profiles(&kind).unwrap()).unwrap();
        let extra = Family::extra(kind);
        let schema = extra.weight_schema(&kind).unwrap();
        assert_eq!(extra.states(), original.states());
        assert_eq!(
            schema
                .tensors
                .iter()
                .map(|t| (&t.id, &t.dimensions))
                .collect::<Vec<_>>(),
            old_schema
                .tensors
                .iter()
                .map(|t| (&t.id, &t.dimensions))
                .collect::<Vec<_>>()
        );
        let mut node_counts = [0_u64; 2];
        for tensor in &schema.tensors {
            let PhysicalWeightLayout::Composite { parts } = &tensor.physical_layout else {
                continue;
            };
            let mut formats = BTreeSet::new();
            let mut actual = 0;
            for part in parts {
                actual += part.extents.iter().product::<u64>();
                if let PhysicalWeightLayout::BlockQuantized { blocks, .. } = part.layout.as_ref() {
                    let c = schema
                        .components
                        .iter()
                        .find(|c| c.id == blocks.component_id)
                        .unwrap();
                    let WeightEncoding::BlockQuantized(s) = &c.encoding else {
                        panic!("physical quantized leaf")
                    };
                    formats.insert(s.format_id.as_str());
                    assert_eq!(
                        c.dimensions.iter().product::<u64>()
                            * u64::from(s.logical_values_per_block),
                        part.extents.iter().product::<u64>()
                    );
                    if s.format_id.as_str() == "quantization.gguf.iq4-nl" {
                        assert_eq!((s.logical_values_per_block, s.bytes_per_block), (32, 18));
                    }
                }
            }
            assert_eq!(formats, BTreeSet::from(FORMATS));
            assert_eq!(actual, tensor.dimensions.iter().product::<u64>());
            node_counts[usize::from(tensor.id.as_str().contains("ffn_"))] += parts.len() as u64;
        }
        assert!(node_counts.iter().all(|count| count % 2 == 1));
        let weights = Weights::new(&schema);
        for c in &schema.components {
            weights.component(c).unwrap();
        }
        let registration = TypedFamilyRegistration::new(extra);
        let definition = registration
            .define(&serde_json::to_value(kind).unwrap())
            .unwrap();
        let prepared = registration.prepare(&definition, &id(PROFILE)).unwrap();
        definition
            .numerical_profiles()
            .resolve(&id(PROFILE))
            .unwrap()
            .validate_program(prepared.program())
            .unwrap();
        for name in ["node.attention", "node.swiglu"] {
            let node = prepared
                .program()
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .find(|n| n.id.as_str() == name)
                .unwrap();
            assert!(node
                .operation_id
                .as_str()
                .contains("q3k-q4k-q5k-iq3s-iq4nl-iq4xs"));
        }
        assert_eq!(original.weight_schema(&kind).unwrap(), old_schema);
        assert_eq!(
            serde_json::to_vec(&original.numerical_profiles(&kind).unwrap()).unwrap(),
            old_profile
        );
    }
}

#[test]
fn extra_fixture_weight_source_preserves_original_component_bytes() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let schema = Family::new(kind).weight_schema(&kind).unwrap();
        let old = attention_family::Weights::new(&schema);
        let wrapper = Weights::new(&schema);
        for component in &schema.components {
            // Exact source payload equality protects all pre-existing GPU gates.
            let a = old.component(component).unwrap();
            let b = wrapper.component(component).unwrap();
            assert_eq!(a.bytes(), b.bytes());
        }
    }
}
