use super::*;
use crate::{
    BufferUsage, ProgramValueId, ResolvedValueBinding, ResolvedValueRole, ResolvedValueStorage,
    ResourceId, SemanticValue,
};

fn attributes(mode: &str) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("broadcast").unwrap(),
        SemanticValue::Text(mode.to_owned()),
    )])
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

fn bindings(has_update: bool, parameter_rows: u64) -> Vec<ResolvedValueBinding> {
    let base = tensor(&[2, 3], ElementType::Bf16);
    let mut values = vec![
        binding(
            "value.x",
            ResolvedValueRole::Input,
            0,
            base.clone(),
            "resource.x",
            0,
        ),
        binding(
            "value.parameter",
            ResolvedValueRole::Input,
            1,
            tensor(&[parameter_rows, 3], ElementType::Bf16),
            "resource.parameter",
            0,
        ),
    ];
    if has_update {
        values.push(binding(
            "value.update",
            ResolvedValueRole::Input,
            2,
            base.clone(),
            "resource.update",
            0,
        ));
    }
    values.push(binding(
        "value.output",
        ResolvedValueRole::Output,
        0,
        base,
        "resource.output",
        0,
    ));
    values
}

#[test]
fn contracts_declare_distinct_ports_and_closed_bf16_resource_boundaries() {
    for (contract, operation_id, capability_id, input_count) in [
        (
            modulation_scale_bf16_contract().unwrap(),
            MODULATION_SCALE_BF16_OPERATION_ID,
            MODULATION_SCALE_BF16_CAPABILITY_ID,
            2,
        ),
        (
            gated_residual_bf16_contract().unwrap(),
            GATED_RESIDUAL_BF16_OPERATION_ID,
            GATED_RESIDUAL_BF16_CAPABILITY_ID,
            3,
        ),
    ] {
        let descriptor = contract.descriptor();
        assert_eq!(descriptor.id.as_str(), operation_id);
        assert_eq!(descriptor.version, ContractVersion::new(1, 0));
        assert_eq!(descriptor.inputs.len(), input_count);
        assert_eq!(descriptor.outputs.len(), 1);
        assert_eq!(
            descriptor.inputs[0].dimensions(),
            [symbol("rows"), symbol("width")]
        );
        assert_eq!(
            descriptor.inputs[1].dimensions(),
            [symbol("parameter_rows"), symbol("width")]
        );
        if input_count == 3 {
            assert_eq!(
                descriptor.inputs[2].dimensions(),
                descriptor.inputs[0].dimensions()
            );
        }
        assert_eq!(
            descriptor.outputs[0].dimensions(),
            descriptor.inputs[0].dimensions()
        );
        for input in &descriptor.inputs {
            assert_eq!(input.access(), TensorAccess::Read);
        }
        assert_eq!(descriptor.outputs[0].access(), TensorAccess::Write);
        for port in descriptor.inputs.iter().chain(&descriptor.outputs) {
            assert_eq!(port.element_types(), &BTreeSet::from([ElementType::Bf16]));
            assert_eq!(port.layouts(), [LayoutConstraint::Contiguous]);
            assert_eq!(port.alias(), &AliasPolicy::NoAlias);
        }
        assert_eq!(descriptor.attributes.entries().len(), 1);
        assert_eq!(descriptor.oracle, OracleSpec::OperationDefined);
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
    }
    let modulation = modulation_scale_bf16_contract().unwrap();
    let gated = gated_residual_bf16_contract().unwrap();
    assert!(modulation
        .validate_signature(&gated.descriptor().inputs, &gated.descriptor().outputs)
        .is_err());
}

#[test]
fn broadcast_attribute_requires_exact_typed_modes_without_defaults_or_extra_fields() {
    assert_eq!(RowBroadcast::Elementwise.as_str(), "elementwise");
    assert_eq!(RowBroadcast::Row.as_str(), "row");
    for contract in [
        modulation_scale_bf16_contract().unwrap(),
        gated_residual_bf16_contract().unwrap(),
    ] {
        let descriptor = contract.descriptor();
        for mode in RowBroadcast::ALL {
            assert_eq!(RowBroadcast::parse(mode.as_str()), Some(mode));
            descriptor
                .validate_attributes(&attributes(mode.as_str()))
                .unwrap();
        }
        for invalid in [
            "",
            "Row",
            "rows",
            "scalar",
            "column",
            "segmented_rows",
            "row ",
        ] {
            assert_eq!(RowBroadcast::parse(invalid), None);
            assert!(descriptor
                .validate_attributes(&attributes(invalid))
                .is_err());
        }
        assert!(descriptor.validate_attributes(&BTreeMap::new()).is_err());
        for wrong_kind in [
            SemanticValue::Bool(true),
            SemanticValue::Unsigned(1),
            SemanticValue::Integers(vec![1]),
        ] {
            let wrong = BTreeMap::from([(AttributeId::new("broadcast").unwrap(), wrong_kind)]);
            assert!(descriptor.validate_attributes(&wrong).is_err());
        }
        for unknown in [
            "hidden_size",
            "shift",
            "activation",
            "row_segment_lengths",
            "parameter_row_indices",
        ] {
            let mut extra = attributes("row");
            extra.insert(
                AttributeId::new(unknown).unwrap(),
                SemanticValue::Bool(false),
            );
            assert!(descriptor.validate_attributes(&extra).is_err());
        }
    }
}

#[test]
fn resolved_contracts_accept_row_broadcast_and_require_matching_update_and_output_shapes() {
    for (contract, has_update) in [
        (modulation_scale_bf16_contract().unwrap(), false),
        (gated_residual_bf16_contract().unwrap(), true),
    ] {
        let descriptor = contract.descriptor();
        for parameter_rows in [1, 2] {
            descriptor
                .validate_resolved_bindings(&bindings(has_update, parameter_rows))
                .unwrap();
        }
        let mut wrong_width = bindings(has_update, 1);
        wrong_width[1] = binding(
            "value.parameter",
            ResolvedValueRole::Input,
            1,
            tensor(&[1, 6], ElementType::Bf16),
            "resource.parameter",
            0,
        );
        assert!(descriptor.validate_resolved_bindings(&wrong_width).is_err());
        let mut wrong_output = bindings(has_update, 1);
        let output = wrong_output.len() - 1;
        wrong_output[output] = binding(
            "value.output",
            ResolvedValueRole::Output,
            0,
            tensor(&[3, 2], ElementType::Bf16),
            "resource.output",
            0,
        );
        assert!(descriptor
            .validate_resolved_bindings(&wrong_output)
            .is_err());
        if has_update {
            let mut wrong_update = bindings(true, 1);
            wrong_update[2] = binding(
                "value.update",
                ResolvedValueRole::Input,
                2,
                tensor(&[3, 2], ElementType::Bf16),
                "resource.update",
                0,
            );
            assert!(descriptor
                .validate_resolved_bindings(&wrong_update)
                .is_err());
        }
    }
}

#[test]
fn no_alias_contract_rejects_output_overlap_with_each_read_operand() {
    for (contract, has_update) in [
        (modulation_scale_bf16_contract().unwrap(), false),
        (gated_residual_bf16_contract().unwrap(), true),
    ] {
        let descriptor = contract.descriptor();
        let resources = if has_update {
            vec!["resource.x", "resource.parameter", "resource.update"]
        } else {
            vec!["resource.x", "resource.parameter"]
        };
        for resource in resources {
            for offset in [0, 2] {
                let mut values = bindings(has_update, 2);
                let output = values.len() - 1;
                values[output] = binding(
                    "value.output",
                    ResolvedValueRole::Output,
                    0,
                    tensor(&[2, 3], ElementType::Bf16),
                    resource,
                    offset,
                );
                assert!(descriptor.validate_resolved_bindings(&values).is_err());
            }
        }
    }
}

#[test]
fn read_only_storage_reuse_requires_the_same_logical_value_and_never_allows_output_alias() {
    for (contract, has_update, reused_ordinal) in [
        (modulation_scale_bf16_contract().unwrap(), false, 1),
        (gated_residual_bf16_contract().unwrap(), true, 2),
    ] {
        let descriptor = contract.descriptor();
        let mut values = bindings(has_update, 2);
        values[reused_ordinal] = binding(
            "value.x",
            ResolvedValueRole::Input,
            reused_ordinal as u32,
            tensor(&[2, 3], ElementType::Bf16),
            "resource.x",
            0,
        );
        descriptor.validate_resolved_bindings(&values).unwrap();

        let mut different_value = values.clone();
        different_value[reused_ordinal] = binding(
            "value.other",
            ResolvedValueRole::Input,
            reused_ordinal as u32,
            tensor(&[2, 3], ElementType::Bf16),
            "resource.x",
            0,
        );
        assert!(descriptor
            .validate_resolved_bindings(&different_value)
            .is_err());

        let output = values.len() - 1;
        values[output] = binding(
            "value.output",
            ResolvedValueRole::Output,
            0,
            tensor(&[2, 3], ElementType::Bf16),
            "resource.x",
            0,
        );
        assert!(descriptor.validate_resolved_bindings(&values).is_err());
    }
}

#[test]
fn geometry_requires_the_selected_mode_instead_of_inferring_broadcast() {
    let base = tensor(&[2, 3], ElementType::Bf16);
    let row = tensor(&[1, 3], ElementType::Bf16);
    let elementwise =
        Bf16RowBroadcastGeometry::new(&base, &base, &base, RowBroadcast::Elementwise).unwrap();
    assert_eq!(
        (
            elementwise.rows(),
            elementwise.width(),
            elementwise.parameter_rows(),
            elementwise.mode()
        ),
        (2, 3, 2, RowBroadcast::Elementwise)
    );
    let broadcast = Bf16RowBroadcastGeometry::new(&base, &row, &base, RowBroadcast::Row).unwrap();
    assert_eq!(
        (
            broadcast.rows(),
            broadcast.width(),
            broadcast.parameter_rows(),
            broadcast.mode()
        ),
        (2, 3, 1, RowBroadcast::Row)
    );
    assert!(Bf16RowBroadcastGeometry::new(&base, &row, &base, RowBroadcast::Elementwise).is_err());
    assert!(Bf16RowBroadcastGeometry::new(&base, &base, &base, RowBroadcast::Row).is_err());
    let other_rows = tensor(&[3, 3], ElementType::Bf16);
    let same_numel = tensor(&[3, 2], ElementType::Bf16);
    for mode in RowBroadcast::ALL {
        assert!(Bf16RowBroadcastGeometry::new(&base, &other_rows, &base, mode).is_err());
        assert!(Bf16RowBroadcastGeometry::new(&base, &same_numel, &base, mode).is_err());
        assert!(Bf16RowBroadcastGeometry::new(&base, &row, &same_numel, mode).is_err());
    }
}

#[test]
fn single_row_and_large_width_metadata_keep_both_explicit_modes() {
    for width in [1, 3, (1 << 24) + 1, u64::MAX / 4] {
        let one_row = tensor(&[1, width], ElementType::Bf16);
        for mode in RowBroadcast::ALL {
            let geometry =
                Bf16RowBroadcastGeometry::new(&one_row, &one_row, &one_row, mode).unwrap();
            assert_eq!(
                (
                    geometry.rows(),
                    geometry.width(),
                    geometry.parameter_rows(),
                    geometry.mode()
                ),
                (1, width, 1, mode)
            );
        }
        let two_rows = tensor(&[2, width], ElementType::Bf16);
        for (mode, parameter) in [
            (RowBroadcast::Row, &one_row),
            (RowBroadcast::Elementwise, &two_rows),
        ] {
            let geometry =
                Bf16RowBroadcastGeometry::new(&two_rows, parameter, &two_rows, mode).unwrap();
            assert_eq!(geometry.rows(), 2);
            assert_eq!(geometry.width(), width);
        }
    }
}

#[test]
fn geometry_rejects_wrong_dtype_rank_or_layout_at_every_tensor_role() {
    let valid = tensor(&[1, 3], ElementType::Bf16);
    for invalid in [
        tensor(&[1, 3], ElementType::F16),
        tensor(&[1, 3], ElementType::F32),
        tensor(&[3], ElementType::Bf16),
        tensor(&[1, 1, 3], ElementType::Bf16),
        ResolvedTensorSpec::new(
            vec![1, 3],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![8, 2],
            },
        )
        .unwrap(),
    ] {
        for mode in RowBroadcast::ALL {
            assert!(Bf16RowBroadcastGeometry::new(&invalid, &valid, &invalid, mode).is_err());
            assert!(Bf16RowBroadcastGeometry::new(&valid, &invalid, &valid, mode).is_err());
            assert!(Bf16RowBroadcastGeometry::new(&valid, &valid, &invalid, mode).is_err());
        }
    }
}

#[test]
fn tensor_byte_overflow_and_zero_extents_fail_before_geometry() {
    for dimensions in [[0, 3], [2, 0], [1, u64::MAX / 2 + 1], [2, u64::MAX / 4 + 1]] {
        assert!(ResolvedTensorSpec::new(
            dimensions.to_vec(),
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous
        )
        .is_err());
    }
}
