use super::*;

fn block(format: &str, bytes: u32) -> BlockQuantizationSpec {
    BlockQuantizationSpec {
        format_id: id(format),
        logical_values_per_block: 256,
        bytes_per_block: bytes,
    }
}

#[test]
fn q8_attention_contracts_retain_strict_fused_ports_state_and_attributes() {
    for (selected, strict) in [
        (
            Q8ActAttentionProfile::GatedDelta,
            gated_delta_recurrent_attention_f32_master_contract().unwrap(),
        ),
        (
            Q8ActAttentionProfile::Causal,
            causal_paged_attention_f32_master_contract().unwrap(),
        ),
    ] {
        let contract = selected.contract().unwrap();
        let actual = contract.descriptor();
        let base = strict.descriptor();
        assert_ne!(actual.id, base.id);
        assert_eq!(actual.id.as_str(), selected.operation_id());
        assert_eq!(actual.version, base.version);
        assert_eq!(actual.inputs, base.inputs);
        assert_eq!(actual.outputs, base.outputs);
        assert_eq!(actual.attributes, base.attributes);
        assert_eq!(actual.resources, base.resources);
        assert_ne!(actual.provider, base.provider);
        let arithmetic = selected.arithmetic();
        arithmetic.validate().unwrap();
        assert_eq!(arithmetic.strict_base.operation_id, base.id);
        let restored: CompositeNumericalArithmetic =
            serde_json::from_slice(&serde_json::to_vec(&arithmetic).unwrap()).unwrap();
        assert_eq!(restored, arithmetic);
        for port in &arithmetic.projections {
            assert_eq!(port.activation_input_type, ElementType::F16);
            assert_eq!(port.activation_output_type, ElementType::F16);
        }
    }
}

#[test]
fn q8_attention_versions_do_not_extend_legacy_bases_or_accept_int8_kv() {
    let causal = Q8ActAttentionProfile::Causal.arithmetic();
    assert_eq!(causal.schema_version, 2);
    let gdn = Q8ActAttentionProfile::GatedDelta.arithmetic();
    assert_eq!(gdn.schema_version, 1);
    for (base, version) in [(&causal, 0), (&causal, 1), (&causal, 3), (&gdn, 2)] {
        let mut changed = base.clone();
        changed.schema_version = version;
        assert!(changed.validate().is_err());
    }
    for wrong_base in [
        CAUSAL_PAGED_ATTENTION_OPERATION_ID,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_INT8_KV_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
    ] {
        let mut changed = causal.clone();
        changed.strict_base.operation_id = id(wrong_base);
        assert!(changed.validate().is_err());
    }
    let mut wrong_version = causal;
    wrong_version.strict_base.version.minor += 1;
    assert!(wrong_version.validate().is_err());
}

#[test]
fn q8_attention_requires_exact_local_roles_and_weight_slots() {
    let causal = Q8ActAttentionProfile::Causal.arithmetic();
    let expected = [
        (ProjectionRole::CausalQuery, 2),
        (ProjectionRole::CausalKey, 3),
        (ProjectionRole::CausalValue, 4),
        (ProjectionRole::CausalOutput, 5),
    ];
    for (projection, &(role, slot)) in causal.projections.iter().zip(&expected) {
        assert_eq!(projection.role, role);
        assert_eq!(projection.weight_input_ordinal, slot);
    }
    for index in 0..causal.projections.len() {
        for wrong_slot in [0, 1, 6, 8, u32::MAX] {
            let mut changed = causal.clone();
            changed.projections[index].weight_input_ordinal = wrong_slot;
            assert!(changed.validate().is_err());
        }
        let mut wrong_dtype = causal.clone();
        wrong_dtype.projections[index].activation_output_type = ElementType::F32;
        assert!(wrong_dtype.validate().is_err());
    }
    // Q and K have the same rank/dtype; those facts cannot authorize a swap.
    let mut swapped = causal.clone();
    swapped.projections[0].weight_input_ordinal = 3;
    swapped.projections[1].weight_input_ordinal = 2;
    assert!(swapped.validate().is_err());
    let mut duplicate = causal.clone();
    duplicate.projections.push(causal.projections[0].clone());
    assert!(duplicate.validate().is_err());
    let mut foreign = causal;
    foreign.projections[0].role = ProjectionRole::GatedDeltaInput;
    assert!(foreign.validate().is_err());
}

#[test]
fn q8_attention_leaves_use_the_same_g32_policy_and_explicit_strict_fallbacks() {
    let ffn = Q8ActSwiGluProfile::Q4KQ5KIq4Xs.arithmetic();
    for selected in [
        Q8ActAttentionProfile::GatedDelta,
        Q8ActAttentionProfile::Causal,
    ] {
        let arithmetic = selected.arithmetic();
        for projection in &arithmetic.projections {
            assert_eq!(projection.leaves, ffn.projections[0].leaves);
            for (index, (format, bytes)) in [
                ("quantization.gguf.q4-k", 144),
                ("quantization.gguf.q5-k", 176),
                ("quantization.gguf.iq4-xs", 136),
            ]
            .into_iter()
            .enumerate()
            {
                let exact = block(format, bytes);
                assert_eq!(
                    arithmetic
                        .declared_projection_arithmetic(
                            projection.role,
                            Some(&exact),
                            256,
                            1,
                            false,
                        )
                        .unwrap(),
                    DeclaredProjectionArithmetic::Staged(&projection.leaves[index].arithmetic)
                );
                for (weight, k, transformed, reason) in [
                    (
                        Some(&exact),
                        257,
                        false,
                        StrictProjectionReason::ShapeNotDeclared,
                    ),
                    (
                        Some(&exact),
                        256,
                        true,
                        StrictProjectionReason::TransformedWeight,
                    ),
                    (None, 256, false, StrictProjectionReason::FormatNotDeclared),
                ] {
                    assert_eq!(
                        arithmetic
                            .declared_projection_arithmetic(
                                projection.role,
                                weight,
                                k,
                                1,
                                transformed,
                            )
                            .unwrap(),
                        DeclaredProjectionArithmetic::StrictBase(reason)
                    );
                }
            }
        }
    }
    let mut partial = Q8ActAttentionProfile::Causal.arithmetic();
    partial
        .projections
        .retain(|p| p.role == ProjectionRole::CausalQuery);
    assert_eq!(
        partial
            .declared_projection_arithmetic(
                ProjectionRole::CausalKey,
                Some(&block("quantization.gguf.q4-k", 144)),
                256,
                1,
                false,
            )
            .unwrap(),
        DeclaredProjectionArithmetic::StrictBase(StrictProjectionReason::UnmodifiedProjection)
    );
}

#[test]
fn q8_attention_wire_rejects_undeclared_ports_and_affine_arithmetic() {
    let base = Q8ActAttentionProfile::Causal.arithmetic();
    let mut wire = serde_json::to_value(&base).unwrap();
    wire["projections"][0]["role"] = json!("causal_rope");
    assert!(serde_json::from_value::<CompositeNumericalArithmetic>(wire).is_err());
    let mut wrong_min = base;
    if let NumericalArithmeticStage::Rescale { min_correction, .. } =
        &mut wrong_min.projections[0].leaves[0].arithmetic.stages[2]
    {
        *min_correction = AffineMinCorrection::None {};
    }
    assert!(wrong_min.validate().is_err());
}

#[test]
fn causal_q8_profile_validates_f32_program_boundary_without_relabeling_projection_ports() {
    let selected = Q8ActAttentionProfile::Causal;
    let kv_tensor = tensor(ElementType::F16, vec![2, 4, 256]);
    let kv_state = StateSpec {
        id: id("state.kv"),
        value_id: id("value.input-8"),
        capacity_demand: StateCapacityDemand::TokenScaled {
            bytes_per_token: kv_tensor.byte_len().unwrap(),
            maximum_tokens: 2048,
        },
        tensor: kv_tensor,
        lifetime: StateLifetime::Sequence,
        initialization: StateInitialization::None,
        checkpoint: StateCheckpointCapability::Unsupported,
    };
    let mut profile = NumericalExecutionProfile {
        id: id("fixture.attention-q8"),
        version: ContractVersion::new(1, 0),
        family_id: id("family.fixture.attention-q8"),
        primary_activation: id("value.output"),
        boundaries: BTreeMap::from([(id("value.output"), ElementType::F32)]),
        states: vec![kv_state.clone()],
        kv_storage: vec![KvStateStorage::F16 {
            state: kv_state.id.clone(),
        }],
        operations: vec![NumericalOperationContract {
            operation_id: id(selected.operation_id()),
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: Some(selected.arithmetic()),
        }],
    };
    let inputs: Vec<ProgramValueId> = (0..9).map(|i| id(&format!("value.input-{i}"))).collect();
    let mut attributes: BTreeMap<AttributeId, SemanticValue> = [
        ("hidden_size", 5120),
        ("query_heads", 24),
        ("key_value_heads", 4),
        ("head_dim", 256),
        ("query_features", 6144),
        ("query_projection_features", 12288),
        ("kv_features", 1024),
        ("rope_dim", 64),
        ("maximum_context_tokens", 2048),
        ("layer_index", 0),
    ]
    .into_iter()
    .map(|(key, value)| (id(key), SemanticValue::Unsigned(value)))
    .collect();
    for (key, numerator, denominator) in [("rope_theta", 10000, 1), ("epsilon", 1, 1_000_000)] {
        attributes.insert(
            id(key),
            SemanticValue::Rational(CanonicalRational::new(numerator, denominator).unwrap()),
        );
    }
    for (key, value) in [
        ("rope_interleaved", false),
        ("output_gate", true),
        ("causal", true),
    ] {
        attributes.insert(id(key), SemanticValue::Bool(value));
    }
    // Static validation retains the real KV state contract. Provider tests
    // exercise the allocation and contents; other inputs remain external.
    let program = ModelProgram::new(
        profile.family_id.clone(),
        inputs[..8].to_vec(),
        vec![ProgramBlock {
            id: id("block.attention"),
            nodes: vec![ProgramNode {
                id: id("node.attention"),
                operation_id: id(selected.operation_id()),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs,
                outputs: vec![id("value.output")],
                attributes,
            }],
        }],
        vec![kv_state],
        vec![],
        vec![id("value.output")],
    )
    .unwrap();
    profile.validate_program(&program).unwrap();
    profile
        .boundaries
        .insert(id("value.output"), ElementType::F16);
    assert!(profile.validate_program(&program).is_err());
    profile
        .boundaries
        .insert(id("value.output"), ElementType::F32);
    profile.operations[0]
        .composite_arithmetic
        .as_mut()
        .unwrap()
        .projections[0]
        .activation_output_type = ElementType::F32;
    assert!(profile.validate_program(&program).is_err());
}
