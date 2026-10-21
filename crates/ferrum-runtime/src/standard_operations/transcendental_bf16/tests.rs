use super::*;
use crate::{
    BufferUsage, ProgramValueId, ResolvedValueBinding, ResolvedValueRole, ResolvedValueStorage,
    ResourceId, SemanticValue,
};

fn contracts() -> [(StandardOperationContract, &'static str, &'static str, usize); 3] {
    [
        (
            silu_bf16_contract().unwrap(),
            SILU_BF16_OPERATION_ID,
            SILU_BF16_CAPABILITY_ID,
            1,
        ),
        (
            silu_mul_bf16_contract().unwrap(),
            SILU_MUL_BF16_OPERATION_ID,
            SILU_MUL_BF16_CAPABILITY_ID,
            2,
        ),
        (
            tanh_bf16_contract().unwrap(),
            TANH_BF16_OPERATION_ID,
            TANH_BF16_CAPABILITY_ID,
            1,
        ),
    ]
}

fn tensor(dimensions: &[u64], element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        dimensions.to_vec(),
        element_type,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

fn binding(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    tensor: ResolvedTensorSpec,
    resource: &str,
    offset: u64,
) -> ResolvedValueBinding {
    let storage = ResolvedValueStorage::single(
        ResourceId::new(resource).unwrap(),
        offset,
        tensor.minimum_storage_bytes().unwrap(),
        tensor.element_type(),
    )
    .unwrap();
    ResolvedValueBinding::new(
        ProgramValueId::new(name).unwrap(),
        role,
        ordinal,
        tensor,
        if role == ResolvedValueRole::Input {
            TensorAccess::Read
        } else {
            TensorAccess::Write
        },
        AliasPolicy::NoAlias,
        BufferUsage::Activations,
        None,
        storage,
    )
    .unwrap()
}

fn bindings(input_count: usize) -> Vec<ResolvedValueBinding> {
    let tensor = tensor(&[2, 3], ElementType::Bf16);
    let mut bindings = (0..input_count)
        .map(|ordinal| {
            binding(
                &format!("value.input{ordinal}"),
                ResolvedValueRole::Input,
                ordinal as u32,
                tensor.clone(),
                &format!("resource.input{ordinal}"),
                0,
            )
        })
        .collect::<Vec<_>>();
    bindings.push(binding(
        "value.output",
        ResolvedValueRole::Output,
        0,
        tensor,
        "resource.output",
        0,
    ));
    bindings
}

#[test]
fn contracts_have_independent_capabilities_and_exact_same_shape_bf16_ports() {
    let mut fingerprints = BTreeSet::new();
    for (contract, operation_id, capability_id, input_count) in contracts() {
        let descriptor = contract.descriptor();
        assert_eq!(descriptor.id.as_str(), operation_id);
        assert_eq!(descriptor.version, ContractVersion::new(1, 0));
        assert_eq!(descriptor.inputs.len(), input_count);
        assert_eq!(descriptor.outputs.len(), 1);
        for port in descriptor.inputs.iter().chain(&descriptor.outputs) {
            assert_eq!(port.dimensions(), [symbol("rows"), symbol("width")]);
            assert_eq!(port.element_types(), &BTreeSet::from([ElementType::Bf16]));
            assert_eq!(port.layouts(), [LayoutConstraint::Contiguous]);
            assert_eq!(port.alias(), &AliasPolicy::NoAlias);
        }
        for input in &descriptor.inputs {
            assert_eq!(input.access(), TensorAccess::Read);
        }
        assert_eq!(descriptor.outputs[0].access(), TensorAccess::Write);
        assert!(descriptor.attributes.entries().is_empty());
        assert_eq!(descriptor.oracle, OracleSpec::OperationDefined);
        assert_eq!(descriptor.profile_phase, ProfilePhase::Forward);
        assert_eq!(descriptor.resources.minimum_value_alignment_bytes, 16);
        assert_eq!(
            descriptor.resources.scratch,
            ResourcePresenceRequirement::Optional
        );
        assert_eq!(
            descriptor.resources.binding,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            descriptor.resources.persistent,
            ResourcePresenceRequirement::Forbidden
        );
        assert_eq!(
            descriptor.provider.required_capabilities,
            BTreeSet::from([CapabilityId::new(capability_id).unwrap()])
        );
        contract
            .validate_signature(&descriptor.inputs, &descriptor.outputs)
            .unwrap();
        assert!(fingerprints.insert(descriptor.fingerprint().unwrap()));
    }
}

#[test]
fn empty_attributes_reject_broadcast_activation_and_hidden_parameters() {
    for (contract, _, _, _) in contracts() {
        let descriptor = contract.descriptor();
        descriptor.validate_attributes(&BTreeMap::new()).unwrap();
        for (name, value) in [
            ("broadcast", SemanticValue::Text("row".to_owned())),
            ("broadcast", SemanticValue::Text("elementwise".to_owned())),
            ("activation", SemanticValue::Text("silu".to_owned())),
            ("hidden_size", SemanticValue::Unsigned(3)),
            (
                "numerical_profile",
                SemanticValue::Text("alternate".to_owned()),
            ),
        ] {
            assert!(descriptor
                .validate_attributes(&BTreeMap::from([(AttributeId::new(name).unwrap(), value)]))
                .is_err());
        }
    }
}

#[test]
fn resolved_contracts_require_all_real_operands_and_outputs_to_match() {
    for (contract, _, _, input_count) in contracts() {
        let descriptor = contract.descriptor();
        descriptor
            .validate_resolved_bindings(&bindings(input_count))
            .unwrap();
        assert!(descriptor
            .validate_resolved_bindings(&bindings(input_count + 1))
            .is_err());
        assert!(descriptor
            .validate_resolved_bindings(&bindings(input_count - 1))
            .is_err());
        for invalid in [
            tensor(&[1, 3], ElementType::Bf16), // Row broadcasting is not supported.
            tensor(&[3, 2], ElementType::Bf16), // Same numel is insufficient.
            tensor(&[2, 3], ElementType::F16),
            tensor(&[2, 3], ElementType::F32),
            tensor(&[6], ElementType::Bf16),
            tensor(&[1, 2, 3], ElementType::Bf16),
            ResolvedTensorSpec::new(
                vec![2, 3],
                ElementType::Bf16,
                ResolvedTensorLayout::Strided {
                    byte_strides: vec![8, 2],
                },
            )
            .unwrap(),
        ] {
            for ordinal in 0..input_count {
                let mut values = bindings(input_count);
                values[ordinal] = binding(
                    &format!("value.input{ordinal}"),
                    ResolvedValueRole::Input,
                    ordinal as u32,
                    invalid.clone(),
                    &format!("resource.input{ordinal}"),
                    0,
                );
                assert!(descriptor.validate_resolved_bindings(&values).is_err());
            }
            let mut values = bindings(input_count);
            values[input_count] = binding(
                "value.output",
                ResolvedValueRole::Output,
                0,
                invalid,
                "resource.output",
                0,
            );
            assert!(descriptor.validate_resolved_bindings(&values).is_err());
        }
    }
}

#[test]
fn output_no_alias_rejects_full_and_partial_overlap_with_every_input() {
    for (contract, _, _, input_count) in contracts() {
        for ordinal in 0..input_count {
            for offset in [0, 2] {
                let mut values = bindings(input_count);
                values[input_count] = binding(
                    "value.output",
                    ResolvedValueRole::Output,
                    0,
                    tensor(&[2, 3], ElementType::Bf16),
                    &format!("resource.input{ordinal}"),
                    offset,
                );
                assert!(contract
                    .descriptor()
                    .validate_resolved_bindings(&values)
                    .is_err());
            }
        }
    }
}

#[test]
fn silu_mul_read_reuse_requires_the_same_logical_value() {
    let contract = silu_mul_bf16_contract().unwrap();
    let mut values = bindings(2);
    values[1] = binding(
        "value.input0",
        ResolvedValueRole::Input,
        1,
        tensor(&[2, 3], ElementType::Bf16),
        "resource.input0",
        0,
    );
    contract
        .descriptor()
        .validate_resolved_bindings(&values)
        .unwrap();
    values[1] = binding(
        "value.distinct",
        ResolvedValueRole::Input,
        1,
        tensor(&[2, 3], ElementType::Bf16),
        "resource.input0",
        0,
    );
    assert!(contract
        .descriptor()
        .validate_resolved_bindings(&values)
        .is_err());
}

#[test]
fn geometry_checks_rank_dtype_layout_and_shape_without_a_width_to_f32_limit() {
    let valid = tensor(&[2, 3], ElementType::Bf16);
    let geometry = Bf16SameShapeGeometry::new(&valid, &valid).unwrap();
    assert_eq!((geometry.rows(), geometry.width()), (2, 3));
    for invalid in [
        tensor(&[1, 3], ElementType::Bf16),
        tensor(&[3, 2], ElementType::Bf16),
        tensor(&[2, 3], ElementType::F16),
        tensor(&[2, 3], ElementType::F32),
        tensor(&[6], ElementType::Bf16),
        tensor(&[1, 2, 3], ElementType::Bf16),
        ResolvedTensorSpec::new(
            vec![2, 3],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![8, 2],
            },
        )
        .unwrap(),
    ] {
        assert!(Bf16SameShapeGeometry::new(&invalid, &valid).is_err());
        assert!(Bf16SameShapeGeometry::new(&valid, &invalid).is_err());
        if invalid.element_type() != ElementType::Bf16
            || invalid.layout() != &ResolvedTensorLayout::Contiguous
            || invalid.dimensions().len() != 2
        {
            assert!(Bf16SameShapeGeometry::new(&invalid, &invalid).is_err());
        }
    }
    for (rows, width) in [(1, 1), (1, (1 << 24) + 1), (2, u64::MAX / 4)] {
        let metadata = tensor(&[rows, width], ElementType::Bf16);
        let geometry = Bf16SameShapeGeometry::new(&metadata, &metadata).unwrap();
        assert_eq!((geometry.rows(), geometry.width()), (rows, width));
    }
    for dimensions in [[0, 3], [2, 0], [1, u64::MAX / 2 + 1], [2, u64::MAX / 4 + 1]] {
        assert!(ResolvedTensorSpec::new(
            dimensions.to_vec(),
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .is_err());
    }
}
