use super::*;

fn leaf(format: ProjectionBlockFormat) -> QuantizedProjectionLeafContract {
    let mut arithmetic = super::staged::arithmetic();
    if format == ProjectionBlockFormat::Iq4Xs {
        if let NumericalArithmeticStage::Rescale { min_correction, .. } = &mut arithmetic.stages[2]
        {
            *min_correction = AffineMinCorrection::None {};
        }
    }
    QuantizedProjectionLeafContract {
        format,
        shape: ProjectionShapeEligibility {
            minimum_input_features: 1024,
            minimum_output_features: 16,
            input_features_multiple: 256,
        },
        arithmetic,
    }
}

fn projection(role: ProjectionRole, slot: u32) -> ProjectionArithmeticOverride {
    ProjectionArithmeticOverride {
        role,
        weight_input_ordinal: slot,
        activation_input_type: ElementType::F16,
        activation_output_type: ElementType::F16,
        leaves: vec![
            leaf(ProjectionBlockFormat::Q4K),
            leaf(ProjectionBlockFormat::Iq4Xs),
        ],
        fallback: StrictProjectionFallback::RetainBaseArithmetic {},
    }
}

fn composite(gdn: bool) -> CompositeNumericalArithmetic {
    let base = if gdn {
        gated_delta_recurrent_attention_f32_master_contract().unwrap()
    } else {
        dense_swiglu_contract().unwrap()
    };
    CompositeNumericalArithmetic {
        schema_version: COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION,
        strict_base: StrictNumericalOperation {
            operation_id: base.descriptor().id.clone(),
            version: base.descriptor().version,
        },
        projections: if gdn {
            vec![
                projection(ProjectionRole::GatedDeltaInput, 2),
                projection(ProjectionRole::GatedDeltaOutput, 7),
            ]
        } else {
            vec![
                projection(ProjectionRole::SwiGluGateUp, 1),
                projection(ProjectionRole::SwiGluDown, 2),
            ]
        },
    }
}

fn profile(gdn: bool) -> NumericalExecutionProfile {
    NumericalExecutionProfile {
        id: id("fixture.projection-overrides"),
        version: ContractVersion::new(1, 0),
        family_id: id("family.composite-fixture"),
        primary_activation: id("value.output"),
        boundaries: BTreeMap::from([(
            id("value.output"),
            if gdn {
                ElementType::F32
            } else {
                ElementType::F16
            },
        )]),
        states: vec![],
        kv_storage: vec![],
        operations: vec![NumericalOperationContract {
            operation_id: id("operation.fixture.projection-overrides"),
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: Some(composite(gdn)),
        }],
    }
}

// This exercises graph/port declarations, not full model compilation: the
// operation/provider remains unregistered and external input tensors are not
// inferred here. The actual standard descriptor supplies the port contract.
fn program(profile: &NumericalExecutionProfile, input_count: usize) -> ModelProgram {
    let inputs: Vec<ProgramValueId> = (0..input_count)
        .map(|i| id(&format!("value.input-{i}")))
        .collect();
    let op = &profile.operations[0];
    ModelProgram::new(
        profile.family_id.clone(),
        inputs.clone(),
        vec![ProgramBlock {
            id: id("block.main"),
            nodes: vec![ProgramNode {
                id: id("node.fused"),
                operation_id: op.operation_id.clone(),
                required_version: op.version,
                work: ProgramNodeWorkSpec::Fixed,
                inputs,
                outputs: vec![id("value.output")],
                attributes: base_attributes(profile),
            }],
        }],
        vec![],
        vec![],
        vec![id("value.output")],
    )
    .unwrap()
}

fn base_attributes(profile: &NumericalExecutionProfile) -> BTreeMap<AttributeId, SemanticValue> {
    let gdn = profile.operations[0]
        .composite_arithmetic
        .as_ref()
        .unwrap()
        .strict_base
        .operation_id
        .as_str()
        == GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID;
    if !gdn {
        return BTreeMap::from([
            (id("hidden_size"), SemanticValue::Unsigned(5120)),
            (id("intermediate_size"), SemanticValue::Unsigned(17408)),
        ]);
    }
    let mut attributes: BTreeMap<_, _> = [
        ("hidden_size", 5120),
        ("key_heads", 16),
        ("value_heads", 48),
        ("key_head_dim", 128),
        ("value_head_dim", 128),
        ("qkv_features", 10240),
        ("value_features", 6144),
        ("qkvz_features", 16384),
        ("ba_features", 96),
        ("qkvzba_features", 16480),
        ("conv_kernel", 4),
        ("conv_state_width", 3),
        ("layer_index", 0),
    ]
    .into_iter()
    .map(|(key, value)| (id(key), SemanticValue::Unsigned(value)))
    .collect();
    attributes.insert(
        id("epsilon"),
        SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
    );
    attributes.insert(
        id("decay_parameterization"),
        SemanticValue::Text(GatedDeltaDecayParameterization::LogRate.as_str().into()),
    );
    attributes.insert(
        id("value_head_mapping"),
        SemanticValue::Text(GatedDeltaValueHeadMapping::GroupedByKeyHead.as_str().into()),
    );
    attributes
}

fn block(format: &str, bytes: u32) -> BlockQuantizationSpec {
    BlockQuantizationSpec {
        format_id: id(format),
        logical_values_per_block: 256,
        bytes_per_block: bytes,
    }
}

#[test]
fn composite_real_standard_projection_ports_preserve_fused_boundaries() {
    for (gdn, count) in [(false, 3), (true, 10)] {
        let p = profile(gdn);
        p.validate().unwrap();
        p.validate_program(&program(&p, count)).unwrap();
        let wire = serde_json::to_value(&p).unwrap();
        let restored: NumericalExecutionProfile = serde_json::from_value(wire).unwrap();
        assert_eq!(restored, p);
        assert_eq!(restored.fingerprint().unwrap(), p.fingerprint().unwrap());
        assert!(p.validate_program(&program(&p, count - 1)).is_err());

        let mut wrong_outer = p.clone();
        wrong_outer.boundaries.insert(
            id("value.output"),
            if gdn {
                ElementType::F16
            } else {
                ElementType::F32
            },
        );
        assert!(wrong_outer
            .validate_program(&program(&wrong_outer, count))
            .is_err());
        // GDN's F16 projection leaf must not be relabelled as its F32 fused
        // result; conversely changing the leaf to F32 cannot repair this.
        if gdn {
            let c = p.operations[0].composite_arithmetic.as_ref().unwrap();
            for override_ in &c.projections {
                assert_eq!(override_.activation_input_type, ElementType::F16);
                assert_eq!(override_.activation_output_type, ElementType::F16);
            }
            assert_eq!(
                p.boundaries[&id::<ProgramValueId>("value.output")],
                ElementType::F32
            );
        }
    }
}

#[test]
fn composite_checks_known_activation_inputs_and_strict_base_attributes() {
    let mut p = profile(true);
    let original = program(&p, 10);
    let mut blocks = original.blocks().to_vec();
    blocks[0].nodes[0].inputs[0] = id("value.before");
    blocks[0].nodes.insert(
        0,
        ProgramNode {
            id: id("node.before"),
            operation_id: id("operation.fixture.before"),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![id("value.input-0")],
            outputs: vec![id("value.before")],
            attributes: BTreeMap::new(),
        },
    );
    p.operations.push(NumericalOperationContract {
        operation_id: id("operation.fixture.before"),
        version: ContractVersion::new(1, 0),
        multiplication_type: Some(ElementType::F32),
        accumulation_type: Some(ElementType::F32),
        staged_arithmetic: None,
        composite_arithmetic: None,
    });
    p.boundaries.insert(id("value.before"), ElementType::F32);
    let input = ModelProgram::new(
        p.family_id.clone(),
        original.inputs().to_vec(),
        blocks.clone(),
        vec![],
        vec![],
        vec![id("value.output")],
    )
    .unwrap();
    p.validate_program(&input).unwrap();
    p.boundaries.insert(id("value.before"), ElementType::F16);
    assert!(p.validate_program(&input).is_err());
    p.boundaries.insert(id("value.before"), ElementType::F32);
    blocks[0].nodes[1]
        .attributes
        .insert(id("undeclared_nonlinearity"), SemanticValue::Bool(true));
    let bad_attributes = ModelProgram::new(
        p.family_id.clone(),
        original.inputs().to_vec(),
        blocks,
        vec![],
        vec![],
        vec![id("value.output")],
    )
    .unwrap();
    assert!(p.validate_program(&bad_attributes).is_err());
}

#[test]
fn composite_mixed_parts_and_ineligible_shapes_declare_strict_fallback() {
    let q4 = block("quantization.gguf.q4-k", 144);
    let iq4 = block("quantization.gguf.iq4-xs", 136);
    let q5 = block("quantization.gguf.q5-k", 176);
    for gdn in [false, true] {
        let c = composite(gdn);
        let role = c.projections[0].role;
        // Each physical part of packed gate/up or QKVZBA can have a different
        // format. One eligible part does not label the entire fused operation.
        for (b, expected) in [(&q4, 0), (&iq4, 1)] {
            for (k, n) in [(1024, 16), (5120, 17408), (17408, 5120)] {
                assert_eq!(
                    c.declared_projection_arithmetic(role, Some(b), k, n, false)
                        .unwrap(),
                    DeclaredProjectionArithmetic::Staged(
                        &c.projections[0].leaves[expected].arithmetic
                    )
                );
            }
        }
        for (b, k, n, transformed, reason) in [
            (
                Some(&q5),
                5120,
                17408,
                false,
                StrictProjectionReason::FormatNotDeclared,
            ),
            (
                None,
                5120,
                17408,
                false,
                StrictProjectionReason::FormatNotDeclared,
            ),
            (
                Some(&q4),
                768,
                17408,
                false,
                StrictProjectionReason::ShapeNotDeclared,
            ),
            (
                Some(&q4),
                1025,
                17408,
                false,
                StrictProjectionReason::ShapeNotDeclared,
            ),
            (
                Some(&q4),
                5120,
                15,
                false,
                StrictProjectionReason::ShapeNotDeclared,
            ),
            (
                Some(&q4),
                5120,
                17408,
                true,
                StrictProjectionReason::TransformedWeight,
            ),
        ] {
            assert_eq!(
                c.declared_projection_arithmetic(role, b, k, n, transformed)
                    .unwrap(),
                DeclaredProjectionArithmetic::StrictBase(reason)
            );
        }
        assert!(c
            .declared_projection_arithmetic(role, Some(&q4), 0, 16, false)
            .is_err());
        let mut wrong_abi = q4.clone();
        wrong_abi.bytes_per_block += 1;
        assert_eq!(
            c.declared_projection_arithmetic(role, Some(&wrong_abi), 5120, 16, false)
                .unwrap(),
            DeclaredProjectionArithmetic::StrictBase(StrictProjectionReason::FormatNotDeclared)
        );
        let mut subset = c.clone();
        let strict_role = subset.projections.pop().unwrap().role;
        assert_eq!(
            subset
                .declared_projection_arithmetic(strict_role, Some(&q4), 5120, 17408, false)
                .unwrap(),
            DeclaredProjectionArithmetic::StrictBase(StrictProjectionReason::UnmodifiedProjection)
        );
    }
    assert!(composite(false)
        .declared_projection_arithmetic(ProjectionRole::GatedDeltaInput, Some(&q4), 5120, 16, false)
        .is_err());
}

#[test]
fn composite_rejects_wrong_roles_slots_ports_conflicts_and_groups() {
    let mutations: &[fn(&mut CompositeNumericalArithmetic)] = &[
        |c| c.schema_version += 1,
        |c| c.strict_base.version.minor += 1,
        |c| c.strict_base.operation_id = id("operation.unknown-base"),
        |c| c.projections.clear(),
        |c| c.projections.push(c.projections[0].clone()),
        |c| c.projections[0].role = ProjectionRole::GatedDeltaInput,
        |c| c.projections[0].weight_input_ordinal = 0,
        |c| c.projections[0].weight_input_ordinal = u32::MAX,
        |c| c.projections[1].weight_input_ordinal = 1,
        |c| c.projections[0].activation_input_type = ElementType::F32,
        |c| c.projections[0].activation_output_type = ElementType::F32,
        |c| c.projections[0].leaves.clear(),
        |c| {
            let p = &mut c.projections[0];
            p.leaves.push(p.leaves[0].clone());
        },
        |c| c.projections[0].leaves[0].shape.minimum_input_features = 0,
        |c| c.projections[0].leaves[0].shape.minimum_output_features = 0,
        |c| c.projections[0].leaves[0].shape.input_features_multiple = 0,
        |c| c.projections[0].leaves[0].shape.input_features_multiple = 128,
        |c| {
            if let NumericalArithmeticStage::ActivationQuantization { group_values, .. } =
                &mut c.projections[0].leaves[0].arithmetic.stages[0]
            {
                *group_values = 96;
            }
        },
        |c| {
            let stages = &mut c.projections[0].leaves[0].arithmetic.stages;
            if let NumericalArithmeticStage::ActivationQuantization { group_values, .. } =
                &mut stages[0]
            {
                *group_values = 64;
            }
            if let NumericalArithmeticStage::IntegerDot {
                values_per_partial, ..
            } = &mut stages[1]
            {
                *values_per_partial = 64;
            }
        },
        |c| {
            if let NumericalArithmeticStage::OutputRounding { output_type, .. } =
                &mut c.projections[0].leaves[0].arithmetic.stages[4]
            {
                *output_type = ElementType::F32;
            }
        },
        |c| c.projections[0].leaves[0].arithmetic = leaf(ProjectionBlockFormat::Iq4Xs).arithmetic,
        |c| c.projections[0].leaves[1].arithmetic = leaf(ProjectionBlockFormat::Q4K).arithmetic,
    ];
    for mutate in mutations {
        let mut c = composite(false);
        mutate(&mut c);
        assert!(c.validate().is_err(), "accepted invalid composite: {c:?}");
    }
    let mut wrong_gdn_slot = composite(true);
    // Input 3 is another rank-2 F16 weight (the convolution), but not a
    // projection. Dtype/rank alone must never authorize that slot.
    wrong_gdn_slot.projections[0].weight_input_ordinal = 3;
    assert!(wrong_gdn_slot.validate().is_err());
}

#[test]
fn composite_requires_an_unambiguous_new_operation_contract() {
    let p = profile(false);
    let mutations: &[fn(&mut NumericalOperationContract)] = &[
        |op| op.staged_arithmetic = Some(super::staged::arithmetic()),
        |op| op.multiplication_type = Some(ElementType::F32),
        |op| op.accumulation_type = Some(ElementType::F32),
        |op| {
            op.operation_id = op
                .composite_arithmetic
                .as_ref()
                .unwrap()
                .strict_base
                .operation_id
                .clone()
        },
    ];
    for mutate in mutations {
        let mut candidate = p.clone();
        mutate(&mut candidate.operations[0]);
        assert!(candidate.validate().is_err());
    }
    let mut breaking = p.clone();
    let op = &mut breaking.operations[0];
    op.operation_id = op
        .composite_arithmetic
        .as_ref()
        .unwrap()
        .strict_base
        .operation_id
        .clone();
    op.version = ContractVersion::new(2, 0);
    breaking.validate().unwrap();
    assert_ne!(breaking.fingerprint().unwrap(), p.fingerprint().unwrap());
}

#[test]
fn composite_wire_rejects_unknown_or_missing_semantics() {
    let wire = serde_json::to_value(composite(false)).unwrap();
    let mutations: &[fn(&mut Value)] = &[
        |v| v["projections"][0]["role"] = json!("gated_delta_convolution"),
        |v| v["projections"][0]["activation_type"] = json!("f16"),
        |v| v["projections"][0]["leaves"][0]["format"] = json!("int4"),
        |v| v["projections"][0]["leaves"][0]["shape"]["min_batch"] = json!(8),
        |v| v["projections"][0]["fallback"]["kind"] = json!("silently_use_i8"),
        |v| v["projections"][0]["fallback"]["ignore"] = json!(true),
        |v| {
            v["projections"][0]
                .as_object_mut()
                .unwrap()
                .remove("fallback");
        },
        |v| v["strict_base"]["unverified_arithmetic"] = json!("i8"),
    ];
    for mutate in mutations {
        let mut candidate = wire.clone();
        mutate(&mut candidate);
        assert!(serde_json::from_value::<CompositeNumericalArithmetic>(candidate).is_err());
    }
    // The prior staged reader must reject this field, rather than reading
    // legacy None/None as a claim of unmodified or wholly integer arithmetic.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PriorOperation {
        #[serde(rename = "operation_id")]
        _operation_id: OperationId,
        #[serde(rename = "version")]
        _version: ContractVersion,
        #[serde(rename = "multiplication_type")]
        _multiplication_type: Option<ElementType>,
        #[serde(rename = "accumulation_type")]
        _accumulation_type: Option<ElementType>,
        #[serde(default, rename = "staged_arithmetic")]
        _staged: Option<StagedNumericalArithmetic>,
    }
    assert!(serde_json::from_value::<PriorOperation>(
        serde_json::to_value(&profile(false).operations[0]).unwrap()
    )
    .is_err());
}

#[test]
fn composite_fingerprint_commits_to_leaf_numerics_and_fallback_extent() {
    let base = profile(false);
    let fingerprint = base.fingerprint().unwrap();
    let mutations: &[fn(&mut CompositeNumericalArithmetic)] = &[
        |c| {
            c.projections.pop();
        },
        |c| {
            c.projections[0].leaves.pop();
        },
        |c| c.projections[0].leaves[0].shape.minimum_input_features += 256,
        |c| c.projections[0].leaves[0].shape.minimum_output_features += 1,
        |c| c.projections[0].leaves[0].shape.input_features_multiple *= 2,
        |c| {
            if let NumericalArithmeticStage::IntegerDot {
                values_per_partial, ..
            } = &mut c.projections[0].leaves[0].arithmetic.stages[1]
            {
                *values_per_partial = 32;
            }
        },
        |c| {
            if let NumericalArithmeticStage::ActivationQuantization { group_values, .. } =
                &mut c.projections[0].leaves[0].arithmetic.stages[0]
            {
                *group_values = 64;
            }
        },
    ];
    for mutate in mutations {
        let mut candidate = base.clone();
        mutate(
            candidate.operations[0]
                .composite_arithmetic
                .as_mut()
                .unwrap(),
        );
        candidate.validate().unwrap();
        assert_ne!(candidate.fingerprint().unwrap(), fingerprint);
    }
}
