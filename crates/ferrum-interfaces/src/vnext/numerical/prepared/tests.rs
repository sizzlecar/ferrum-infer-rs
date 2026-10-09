use super::*;
use crate::vnext::*;

#[test]
fn prepared_projection_fingerprint_is_cached_without_changing_wire_or_trust() {
    let policy = UpstreamMarkerV2Profile::Causal.arithmetic();
    let values = vec![
        binding(2, &[Some("quantization.gguf.iq4-xs")], 256, 512, false),
        binding(3, &[Some("quantization.gguf.q4-k")], 256, 64, false),
        binding(4, &[Some("quantization.gguf.q6-k")], 256, 64, false),
        binding(5, &[Some("quantization.gguf.q5-k")], 512, 256, false),
    ];
    let prepared = PreparedProjectionNumerics::prepare(&policy, &values).unwrap();
    let wire = serde_json::to_vec(&prepared).unwrap();
    let expected = format!("{:x}", Sha256::digest(&wire));
    assert!(prepared.data.fingerprint.get().is_none());
    assert_eq!(prepared.fingerprint(), expected);
    assert!(std::ptr::eq(
        prepared.fingerprint().as_ptr(),
        prepared.fingerprint().as_ptr()
    ));
    assert_eq!(serde_json::to_vec(&prepared).unwrap(), wire);
    let restored: PreparedProjectionNumerics = serde_json::from_slice(&wire).unwrap();
    assert!(restored.data.fingerprint.get().is_none());
    assert_eq!(restored, prepared);
    assert_eq!(restored.fingerprint(), expected);
    restored.validate_bindings(&policy, &values).unwrap();

    let mut changed = values.clone();
    changed[2] = binding(4, &[Some("quantization.gguf.q4-k")], 256, 64, false);
    let changed = PreparedProjectionNumerics::prepare(&policy, &changed).unwrap();
    assert_ne!(changed.fingerprint(), expected);
    let mut changed_policy = policy.clone();
    changed_policy.projections.pop();
    assert_ne!(
        PreparedProjectionNumerics::prepare(&changed_policy, &values)
            .unwrap()
            .fingerprint(),
        expected
    );
    let mut tampered: serde_json::Value = serde_json::from_slice(&wire).unwrap();
    tampered["projections"][0]["output_features"] = 513.into();
    let tampered: PreparedProjectionNumerics = serde_json::from_value(tampered).unwrap();
    assert_ne!(tampered.fingerprint(), expected);
    assert!(tampered.validate_bindings(&policy, &values).is_err());
}

#[test]
fn prepared_attention_q8act_preserves_causal_slots_shapes_and_fallbacks() {
    let policy = Q8ActAttentionProfile::Causal.arithmetic();
    let values = vec![
        binding(2, &[Some("quantization.gguf.iq4-xs")], 256, 512, false),
        binding(3, &[Some("quantization.gguf.q4-k")], 256, 64, false),
        binding(4, &[Some("quantization.gguf.q6-k")], 256, 64, false),
        binding(5, &[Some("quantization.gguf.q5-k")], 512, 256, false),
    ];
    let prepared = PreparedProjectionNumerics::prepare(&policy, &values).unwrap();
    for (role, ordinal, k, n, staged) in [
        (ProjectionRole::CausalQuery, 2, 256, 512, true),
        (ProjectionRole::CausalKey, 3, 256, 64, true),
        (ProjectionRole::CausalValue, 4, 256, 64, false),
        (ProjectionRole::CausalOutput, 5, 512, 256, true),
    ] {
        let projection = prepared.projection(role).unwrap();
        assert_eq!(
            (
                projection.weight_input_ordinal(),
                projection.input_features(),
                projection.output_features()
            ),
            (ordinal, k, n)
        );
        assert_eq!(projection.has_staged_leaf(), staged);
        assert_eq!(
            projection.leaves()[0].component_id(),
            &id::<WeightId>(&format!("component.{ordinal}.0"))
        );
    }
    assert!(matches!(
        prepared
            .projection(ProjectionRole::CausalValue)
            .unwrap()
            .leaves()[0]
            .route(),
        PreparedProjectionRoute::StrictBase {
            reason: StrictProjectionReason::FormatNotDeclared
        }
    ));
    let wire = serde_json::to_value(&prepared).unwrap();
    let roundtrip: PreparedProjectionNumerics = serde_json::from_value(wire.clone()).unwrap();
    roundtrip.validate_bindings(&policy, &values).unwrap();
    let mut tampered = wire;
    tampered["projections"][0]["weight_input_ordinal"] = 3.into();
    assert!(
        serde_json::from_value::<PreparedProjectionNumerics>(tampered)
            .unwrap()
            .validate_bindings(&policy, &values)
            .is_err()
    );
    let mut changed = values.clone();
    changed[0] = binding(2, &[Some("quantization.gguf.iq4-xs")], 256, 512, true);
    assert!(prepared.validate_bindings(&policy, &changed).is_err());
    assert!(PreparedProjectionNumerics::prepare(&policy, &values[..3]).is_err());
    let mut duplicate = values.clone();
    duplicate.push(values[1].clone());
    assert!(PreparedProjectionNumerics::prepare(&policy, &duplicate).is_err());
    let mut partial = policy;
    partial
        .projections
        .retain(|projection| projection.role != ProjectionRole::CausalKey);
    let partial = PreparedProjectionNumerics::prepare(&partial, &values).unwrap();
    assert!(matches!(
        partial
            .projection(ProjectionRole::CausalKey)
            .unwrap()
            .leaves()[0]
            .route(),
        PreparedProjectionRoute::StrictBase {
            reason: StrictProjectionReason::UnmodifiedProjection
        }
    ));
}

fn id<T: TryFrom<String>>(s: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    s.to_owned().try_into().unwrap()
}

pub(in crate::vnext::numerical) fn binding(
    ordinal: u32,
    formats: &[Option<&str>],
    k: u64,
    n: u64,
    transformed: bool,
) -> ResolvedValueBinding {
    let packed = ordinal == 1;
    let shape = if packed {
        vec![formats.len() as u64, n, k]
    } else {
        vec![n, k]
    };
    let mut components = Vec::new();
    let mut parts = Vec::new();
    for (i, format) in formats.iter().enumerate() {
        let component_id = id::<WeightId>(&format!("component.{ordinal}.{i}"));
        let encoding = match format {
            None => WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            Some(name) => WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: id(name),
                logical_values_per_block: if *name == "quantization.gguf.iq4-nl" {
                    32
                } else {
                    256
                },
                bytes_per_block: match *name {
                    "quantization.gguf.iq4-xs" => 136,
                    "quantization.gguf.q4-k" => 144,
                    "quantization.gguf.q5-k" => 176,
                    "quantization.gguf.q6-k" => 210,
                    "quantization.gguf.q3-k" | "quantization.gguf.iq3-s" => 110,
                    "quantization.gguf.iq4-nl" => 18,
                    _ => panic!("unknown fixture block ABI"),
                },
            }),
        };
        let columns = match &encoding {
            WeightEncoding::BlockQuantized(block) => k / u64::from(block.logical_values_per_block),
            _ => k,
        };
        components.push(ResolvedWeightComponentLayout::from_parts(
            component_id.clone(),
            if format.is_some() {
                WeightComponentRole::PackedValues
            } else {
                WeightComponentRole::Values
            },
            if packed {
                vec![1, n, columns]
            } else {
                vec![n, columns]
            },
            encoding,
        ));
        let mut layout = if format.is_some() {
            PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding::exact_contiguous(component_id),
                block_axis: if packed { 2 } else { 1 },
                block_padding: PhysicalWeightPadding::Exact,
            }
        } else {
            PhysicalWeightLayout::Dense { component_id }
        };
        if transformed {
            layout = PhysicalWeightLayout::Hadamard {
                values: Box::new(layout),
                transform: HadamardTransformSpec {
                    block_size: std::num::NonZeroU32::new(4).unwrap(),
                    signs: HadamardSigns::Identity,
                    application: HadamardApplication::BeforeMatmul {
                        input_permutation: None,
                    },
                },
            };
        }
        parts.push(CompositeWeightPart {
            layout: Box::new(layout),
            logical_offsets: if packed {
                vec![i as u64, 0, 0]
            } else {
                vec![0, 0]
            },
            extents: if packed { vec![1, n, k] } else { vec![n, k] },
        });
    }
    let storage = ResolvedValueStorage::composite(
        components
            .iter()
            .map(|component| {
                ResolvedStorageComponent::new(
                    Some(component.component_id().clone()),
                    id(&format!("resource.{}", component.component_id())),
                    0,
                    component.physical_bytes().unwrap(),
                    component.physical_element_type(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let weight = ResolvedWeightBinding::from_parts(
        id(&format!("weight.{ordinal}")),
        id("weight-format.gguf.native-block"),
        id("layout.fixture"),
        ContractVersion::new(1, 0),
        PhysicalWeightLayout::Composite { parts },
        components,
    )
    .unwrap();
    ResolvedValueBinding::new(
        id(&format!("value.{ordinal}")),
        ResolvedValueRole::Input,
        ordinal,
        ResolvedTensorSpec::new(shape, ElementType::F16, ResolvedTensorLayout::Contiguous).unwrap(),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(weight),
        storage,
    )
    .unwrap()
}

#[test]
fn prepared_q8act_preserves_mixed_physical_leaves_and_exact_fallbacks() {
    let policy = dense_swiglu_iq4xs_q8act_g32_arithmetic();
    let values = vec![
        binding(
            1,
            &[
                Some("quantization.gguf.iq4-xs"),
                Some("quantization.gguf.q4-k"),
            ],
            256,
            256,
            false,
        ),
        binding(2, &[None], 256, 256, false),
    ];
    let prepared = PreparedProjectionNumerics::prepare(&policy, &values).unwrap();
    assert_eq!(prepared.staged_leaf_count(), 1);
    let gate = prepared.projection(ProjectionRole::SwiGluGateUp).unwrap();
    assert_eq!(gate.output_features(), 512);
    assert_eq!(gate.leaves()[1].output_offset(), 256);
    assert!(gate.leaves()[0].is_staged());
    assert_eq!(
        gate.leaves()[1].route(),
        &PreparedProjectionRoute::StrictBase {
            reason: StrictProjectionReason::FormatNotDeclared
        }
    );
    prepared.validate_bindings(&policy, &values).unwrap();
    let wire = serde_json::to_value(&prepared).unwrap();
    let decoded: PreparedProjectionNumerics = serde_json::from_value(wire).unwrap();
    decoded.validate_bindings(&policy, &values).unwrap();
    let rotated = vec![
        binding(
            1,
            &[
                Some("quantization.gguf.iq4-xs"),
                Some("quantization.gguf.q4-k"),
            ],
            256,
            256,
            true,
        ),
        values[1].clone(),
    ];
    let strict = PreparedProjectionNumerics::prepare(&policy, &rotated).unwrap();
    assert_eq!(strict.staged_leaf_count(), 0);
    assert_eq!(
        strict.projections()[0].leaves()[0].route(),
        &PreparedProjectionRoute::StrictBase {
            reason: StrictProjectionReason::TransformedWeight
        }
    );
    assert!(prepared.validate_bindings(&policy, &rotated).is_err());
}

#[test]
fn prepared_q8act_rejects_tampered_routes_offsets_policy_and_missing_ports() {
    let policy = dense_swiglu_iq4xs_q8act_g32_arithmetic();
    let values = vec![
        binding(
            1,
            &[Some("quantization.gguf.iq4-xs"), None],
            256,
            256,
            false,
        ),
        binding(2, &[None], 256, 256, false),
    ];
    let good = PreparedProjectionNumerics::prepare(&policy, &values).unwrap();
    let mutations: &[fn(&mut PreparedProjectionNumericsData)] = &[
        |p| {
            p.projections[0].leaves[0].route = PreparedProjectionRoute::StrictBase {
                reason: StrictProjectionReason::FormatNotDeclared,
            }
        },
        |p| p.projections[0].leaves[1].route = PreparedProjectionRoute::Staged {},
        |p| p.projections[0].leaves[1].output_offset -= 1,
        |p| p.projections[0].leaves[0].component_id = id("component.wrong"),
        |p| p.projections[0].weight_input_ordinal = 0,
        |p| {
            p.projections.pop();
        },
        |p| {
            p.contract.projections[0].leaves[0]
                .shape
                .minimum_input_features *= 2
        },
    ];
    for mutate in mutations {
        let bad = tamper(&good, mutate);
        assert!(bad.validate_bindings(&policy, &values).is_err());
    }
    assert!(PreparedProjectionNumerics::prepare(&policy, &values[..1]).is_err());
    let mut duplicate = values.clone();
    duplicate.push(values[0].clone());
    assert!(PreparedProjectionNumerics::prepare(&policy, &duplicate).is_err());
}

#[test]
fn q8act_operation_preserves_strict_swiglu_signature_and_dot32_policy() {
    let original = dense_swiglu_contract().unwrap();
    let candidate = dense_swiglu_iq4xs_q8act_g32_contract().unwrap();
    let a = original.descriptor();
    let b = candidate.descriptor();
    assert_ne!(a.id, b.id);
    assert_eq!(a.inputs, b.inputs);
    assert_eq!(a.outputs, b.outputs);
    assert_eq!(a.attributes, b.attributes);
    assert_eq!(a.resources, b.resources);
    let policy = dense_swiglu_iq4xs_q8act_g32_arithmetic();
    policy.validate().unwrap();
    for projection in policy.projections {
        assert_eq!(projection.leaves.len(), 1);
        assert_eq!(projection.leaves[0].format, ProjectionBlockFormat::Iq4Xs);
        assert!(matches!(
            projection.leaves[0].arithmetic.stages[1],
            NumericalArithmeticStage::IntegerDot {
                values_per_partial: 32,
                ..
            }
        ));
    }
}

#[test]
fn prepared_three_format_q8act_preserves_leaf_identity_and_old_policy_fallbacks() {
    let selected = Q8ActSwiGluProfile::Q4KQ5KIq4Xs.arithmetic();
    let old = Q8ActSwiGluProfile::Iq4Xs.arithmetic();
    let values = vec![
        binding(
            1,
            &[
                Some("quantization.gguf.q4-k"),
                Some("quantization.gguf.q5-k"),
            ],
            256,
            256,
            false,
        ),
        binding(2, &[Some("quantization.gguf.iq4-xs")], 256, 256, false),
    ];
    let prepared = PreparedProjectionNumerics::prepare(&selected, &values).unwrap();
    assert_eq!(prepared.staged_leaf_count(), 3);
    let gate = prepared.projection(ProjectionRole::SwiGluGateUp).unwrap();
    assert_eq!(gate.weight_input_ordinal(), 1);
    assert_eq!(gate.output_features(), 512);
    assert_eq!(gate.leaves()[0].output_offset(), 0);
    assert_eq!(gate.leaves()[1].output_offset(), 256);
    assert_eq!(
        gate.leaves()[1].component_id(),
        &id::<WeightId>("component.1.1")
    );
    assert!(
        matches!(gate.leaves()[1].encoding(), WeightEncoding::BlockQuantized(spec)
        if spec.format_id.as_str() == "quantization.gguf.q5-k" && spec.bytes_per_block == 176)
    );
    prepared.validate_bindings(&selected, &values).unwrap();
    assert!(prepared.validate_bindings(&old, &values).is_err());
    let legacy = PreparedProjectionNumerics::prepare(&old, &values).unwrap();
    assert_eq!(legacy.staged_leaf_count(), 1);
    for leaf in legacy.projections()[0].leaves() {
        assert_eq!(
            leaf.route(),
            &PreparedProjectionRoute::StrictBase {
                reason: StrictProjectionReason::FormatNotDeclared,
            }
        );
    }
    let changed = tamper(&prepared, |data| {
        data.projections[0].leaves[1].route = PreparedProjectionRoute::StrictBase {
            reason: StrictProjectionReason::FormatNotDeclared,
        };
    });
    assert!(changed.validate_bindings(&selected, &values).is_err());

    let mixed = vec![
        binding(
            1,
            &[
                Some("quantization.gguf.q5-k"),
                Some("quantization.gguf.q6-k"),
            ],
            256,
            256,
            false,
        ),
        binding(2, &[None], 256, 256, false),
    ];
    let routes = PreparedProjectionNumerics::prepare(&selected, &mixed).unwrap();
    assert_eq!(routes.staged_leaf_count(), 1);
    for leaf in [
        &routes.projections()[0].leaves()[1],
        &routes.projections()[1].leaves()[0],
    ] {
        assert_eq!(
            leaf.route(),
            &PreparedProjectionRoute::StrictBase {
                reason: StrictProjectionReason::FormatNotDeclared,
            }
        );
    }
    let transformed = vec![
        binding(
            1,
            &[
                Some("quantization.gguf.q4-k"),
                Some("quantization.gguf.q5-k"),
            ],
            256,
            256,
            true,
        ),
        mixed[1].clone(),
    ];
    let routes = PreparedProjectionNumerics::prepare(&selected, &transformed).unwrap();
    assert_eq!(routes.staged_leaf_count(), 0);
    for leaf in routes.projections()[0].leaves() {
        assert_eq!(
            leaf.route(),
            &PreparedProjectionRoute::StrictBase {
                reason: StrictProjectionReason::TransformedWeight,
            }
        );
    }
}

// Mutations model untrusted wire data, never a live shared immutable object.
fn tamper(
    prepared: &PreparedProjectionNumerics,
    mutate: impl FnOnce(&mut PreparedProjectionNumericsData),
) -> PreparedProjectionNumerics {
    let mut changed: PreparedProjectionNumerics =
        serde_json::from_slice(&serde_json::to_vec(prepared).unwrap()).unwrap();
    assert!(changed.data.fingerprint.get().is_none());
    assert!(changed.data.static_contract_validation.get().is_none());
    mutate(Arc::get_mut(&mut changed.data).unwrap());
    changed
}

#[path = "sharing_tests.rs"]
mod sharing;
