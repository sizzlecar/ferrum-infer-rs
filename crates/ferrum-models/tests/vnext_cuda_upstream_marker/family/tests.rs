use super::*;

#[test]
fn upstream_prefill_fixture_preserves_mixed_weights_and_declares_real_context() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let old = Family::new(kind);
        let family = Family::prefill(kind, 2052);
        assert_eq!(
            family.weight_schema(&kind).unwrap(),
            old.weight_schema(&kind).unwrap()
        );
        let registration = TypedFamilyRegistration::new(family);
        let definition = registration
            .define(&serde_json::to_value(kind).unwrap())
            .unwrap();
        let id: NumericalProfileId = id("fixture.attention-ffn.require-upstream-marker-v2-prefill");
        let prepared = registration.prepare(&definition, &id).unwrap();
        let profile = definition.numerical_profiles().resolve(&id).unwrap();
        profile.validate_program(prepared.program()).unwrap();
        for state in prepared.program().states() {
            if let StateCapacityDemand::TokenScaled { maximum_tokens, .. } = state.capacity_demand {
                assert_eq!(maximum_tokens, 2052);
            }
        }
        let attention = &prepared.program().blocks()[0].nodes[1];
        assert_eq!(
            attention.operation_id.as_str(),
            match kind {
                AttentionKind::GatedDelta =>
                    UpstreamMarkerV2Profile::GatedDeltaPrefill.operation_id(),
                AttentionKind::Causal => UpstreamMarkerV2Profile::CausalPrefill.operation_id(),
                _ => unreachable!(),
            }
        );
        if kind == AttentionKind::Causal {
            assert_eq!(
                attention.attributes[&super::id("maximum_context_tokens")],
                SemanticValue::Unsigned(2052)
            );
        }
    }
}

#[test]
fn upstream_marker_fixture_odd_leaf_flags_require_padding_without_changing_tensors() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let family = Family::new(kind);
        let schema = family.weight_schema(&kind).unwrap();
        let base = attention_family::Family::new(kind)
            .weight_schema(&kind)
            .unwrap();
        let attention_names: &[&str] = match kind {
            AttentionKind::GatedDelta => &["qkvzba", "o"],
            AttentionKind::Causal => &["q", "k", "v", "o"],
            _ => unreachable!(),
        };
        for names in [attention_names, &["ffn_gate_up", "ffn_down"]] {
            let mut leaf_count = 0;
            for name in names {
                let tensor = schema
                    .tensors
                    .iter()
                    .find(|t| t.id.as_str() == format!("weight.{name}"))
                    .unwrap();
                if let Some(original) = base.tensors.iter().find(|t| t.id == tensor.id) {
                    assert_eq!(tensor.dimensions, original.dimensions);
                    assert_eq!(tensor.logical_element_type, original.logical_element_type);
                }
                let PhysicalWeightLayout::Composite { parts } = &tensor.physical_layout else {
                    panic!("physical fixture parts")
                };
                leaf_count += parts.len() as u64;
                let mut formats = BTreeSet::new();
                let mut values = 0;
                for part in parts {
                    values += part.extents.iter().product::<u64>();
                    match part.layout.as_ref() {
                        PhysicalWeightLayout::Dense { component_id } => {
                            let component = schema
                                .components
                                .iter()
                                .find(|c| &c.id == component_id)
                                .unwrap();
                            assert_eq!(component.dimensions, part.extents);
                        }
                        PhysicalWeightLayout::BlockQuantized { blocks, .. } => {
                            let component = schema
                                .components
                                .iter()
                                .find(|c| c.id == blocks.component_id)
                                .unwrap();
                            let WeightEncoding::BlockQuantized(spec) = &component.encoding else {
                                panic!("quantized fixture")
                            };
                            formats.insert(spec.format_id.as_str());
                        }
                        _ => panic!("unexpected physical leaf"),
                    }
                }
                assert_eq!(values, tensor.dimensions.iter().product::<u64>());
                assert_eq!(
                    formats,
                    BTreeSet::from([
                        "quantization.gguf.q4-k",
                        "quantization.gguf.q5-k",
                        "quantization.gguf.iq4-xs"
                    ])
                );
            }
            let logical = leaf_count * 8;
            let requirement = ProviderWorkspaceRequirement::new(
                logical,
                16,
                ProviderWorkspaceScope::Plan,
                ProviderWorkspaceReusePolicy::Preserve,
                DynamicStorageRequirement::contiguous(),
            )
            .unwrap();
            assert_eq!(leaf_count % 2, 1);
            assert_eq!(requirement.fixed_bytes(), Some(logical));
            assert_eq!(requirement.minimum_bytes().unwrap(), logical + 8);
        }
        // Real typed preparation also checks composite coverage/shape; summed
        // element counts alone could otherwise hide overlapping output parts.
        TypedFamilyRegistration::new(family)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(PROFILE))
            .unwrap();
    }
}

#[test]
fn upstream_marker_fixture_prepares_real_attention_norm_ffn_residual_boundaries() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let base = attention_family::Family::new(kind);
        let old_wire = serde_json::to_vec(&base.numerical_profiles(&kind).unwrap()).unwrap();
        let provider = Family::new(kind);
        let mut schema = provider.weight_schema(&kind).unwrap();
        // Typed preparation canonicalizes independent declarations, while each
        // composite's physical part offsets remain semantic and unsorted.
        schema.components.sort_by(|a, b| a.id.cmp(&b.id));
        schema.tensors.sort_by(|a, b| a.id.cmp(&b.id));
        let catalog = provider.numerical_profiles(&kind).unwrap();
        let profile = catalog.resolve(&id(PROFILE)).unwrap();
        let registration = TypedFamilyRegistration::new(provider);
        let prepared = registration
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(PROFILE))
            .unwrap();
        profile.validate_program(prepared.program()).unwrap();
        assert_eq!(prepared.weight_schema(), &schema);
        assert_eq!(prepared.program().states(), base.states());
        assert_eq!(
            profile.boundaries[&id::<ProgramValueId>("value.attention")],
            ElementType::F32
        );
        assert_eq!(
            profile.boundaries[&id::<ProgramValueId>("value.normalized")],
            ElementType::F16
        );
        assert_eq!(
            profile.boundaries[&id::<ProgramValueId>("value.ffn")],
            ElementType::F16
        );
        assert_eq!(
            profile.boundaries[&id::<ProgramValueId>("value.output")],
            ElementType::F32
        );
        let nodes = &prepared.program().blocks()[0].nodes;
        let find = |name: &str| nodes.iter().find(|node| node.id.as_str() == name).unwrap();
        assert_eq!(
            find("node.ffn_norm").inputs[0],
            find("node.attention").outputs[0]
        );
        assert_eq!(
            find("node.swiglu").inputs[0],
            find("node.ffn_norm").outputs[0]
        );
        assert_eq!(
            find("node.ffn_residual").inputs,
            [
                find("node.attention").outputs[0].clone(),
                find("node.swiglu").outputs[0].clone()
            ]
        );
        for name in ["ffn_gate_up", "ffn_down"] {
            let tensor = schema
                .tensors
                .iter()
                .find(|t| t.id.as_str() == format!("weight.{name}"))
                .unwrap();
            let PhysicalWeightLayout::Composite { parts } = &tensor.physical_layout else {
                panic!("mixed fixture")
            };
            let physical_ids: Vec<_> = parts
                .iter()
                .map(|part| match part.layout.as_ref() {
                    PhysicalWeightLayout::BlockQuantized { blocks, .. } => {
                        blocks.component_id.clone()
                    }
                    PhysicalWeightLayout::Dense { component_id } => component_id.clone(),
                    _ => panic!("fixture leaf"),
                })
                .collect();
            let mut sorted = physical_ids.clone();
            sorted.sort();
            assert_ne!(
                physical_ids, sorted,
                "physical offsets cannot be inferred from lexical identity"
            );
            assert!(parts
                .iter()
                .any(|p| matches!(p.layout.as_ref(), PhysicalWeightLayout::Dense { .. })));
        }
        assert_eq!(
            serde_json::to_vec(&base.numerical_profiles(&kind).unwrap()).unwrap(),
            old_wire
        );
    }
}

#[test]
fn upstream_marker_fixture_fingerprint_rejects_policy_or_dtype_substitution() {
    let family = Family::new(AttentionKind::Causal);
    let catalog = family.numerical_profiles(&AttentionKind::Causal).unwrap();
    let profile = catalog.resolve(&id(PROFILE)).unwrap();
    let bytes = serde_json::to_vec(profile).unwrap();
    let roundtrip: NumericalExecutionProfile = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        profile.fingerprint().unwrap(),
        roundtrip.fingerprint().unwrap()
    );
    assert_eq!(serde_json::to_vec(&roundtrip).unwrap(), bytes);
    let mut changed = profile.clone();
    changed
        .boundaries
        .insert(id("value.normalized"), ElementType::F32);
    assert_ne!(
        changed.fingerprint().unwrap(),
        profile.fingerprint().unwrap()
    );
    assert!(family
        .semantic_program(&AttentionKind::Causal, &changed)
        .is_err());
    let mut changed = profile.clone();
    changed
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == UpstreamMarkerV2Profile::SwiGlu.operation_id())
        .unwrap()
        .composite_arithmetic
        .as_mut()
        .unwrap()
        .projections
        .pop();
    changed.validate().unwrap();
    assert_ne!(
        changed.fingerprint().unwrap(),
        profile.fingerprint().unwrap()
    );
    assert!(family
        .semantic_program(&AttentionKind::Causal, &changed)
        .is_err());
}

#[test]
fn hybrid_fixture_keeps_atn_tensor_state_contract_and_distinct_arithmetic_ids() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let baseline = Family::g32_baseline(kind);
        let hybrid = Family::hybrid(kind, MAX_TOKENS);
        assert_eq!(
            baseline.weight_schema(&kind).unwrap(),
            hybrid.weight_schema(&kind).unwrap()
        );
        assert_eq!(baseline.states(), hybrid.states());
        let mut prepared = Vec::new();
        for family in [baseline, hybrid, Family::hybrid(kind, 2052)] {
            let profile_id: NumericalProfileId = id(family.profile_id());
            let registration = TypedFamilyRegistration::new(family);
            let definition = registration
                .define(&serde_json::to_value(kind).unwrap())
                .unwrap();
            let selected = registration.prepare(&definition, &profile_id).unwrap();
            definition
                .numerical_profiles()
                .resolve(&profile_id)
                .unwrap()
                .validate_program(selected.program())
                .unwrap();
            prepared.push(selected);
        }
        assert_eq!(
            prepared[0].program().states(),
            prepared[1].program().states()
        );
        for name in ["node.attention", "node.swiglu"] {
            let nodes = [&prepared[0], &prepared[1]].map(|p| {
                p.program()
                    .blocks()
                    .iter()
                    .flat_map(|b| &b.nodes)
                    .find(|n| n.id.as_str() == name)
                    .unwrap()
            });
            assert_ne!(nodes[0].operation_id, nodes[1].operation_id);
            assert_eq!(nodes[0].inputs, nodes[1].inputs);
            assert_eq!(nodes[0].outputs, nodes[1].outputs);
        }
    }
}
