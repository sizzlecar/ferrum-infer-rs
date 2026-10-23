use super::*;
use crate::backend::reference::{
    reference_vnext_gated_residual_bf16_provider, reference_vnext_modulation_scale_bf16_provider,
    ReferenceVNextComposition,
};
use ferrum_runtime::{
    AliasPolicy, BufferUsage, DeviceId, ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueStorage, ResourceId, TensorAccess,
};

fn tensor(dimensions: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dimensions.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}

fn value(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    tensor: ResolvedTensorSpec,
) -> ResolvedValueBinding {
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
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

fn attributes(mode: &str) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("broadcast").unwrap(),
        SemanticValue::Text(mode.to_owned()),
    )])
}

fn values(kind: ReferencePointwiseKind, parameter_rows: u64) -> Vec<ResolvedValueBinding> {
    let base = tensor(&[2, 3], ElementType::Bf16);
    let mut values = vec![
        value("pointwise-input", ResolvedValueRole::Input, 0, base.clone()),
        value(
            "pointwise-parameter",
            ResolvedValueRole::Input,
            1,
            tensor(&[parameter_rows, 3], ElementType::Bf16),
        ),
    ];
    if kind == ReferencePointwiseKind::GatedResidual {
        values.push(value(
            "pointwise-update",
            ResolvedValueRole::Input,
            2,
            base.clone(),
        ));
    }
    values.push(value(
        "pointwise-output",
        ResolvedValueRole::Output,
        0,
        base,
    ));
    values
}

#[test]
fn providers_bind_independent_contracts_and_estimators_without_weight_formats() {
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.pointwise-provider").unwrap(),
    )
    .unwrap();
    let modulation = reference_vnext_modulation_scale_bf16_provider(composition.runtime()).unwrap();
    let residual = reference_vnext_gated_residual_bf16_provider(composition.runtime()).unwrap();
    let contracts = [
        modulation_scale_bf16_contract().unwrap(),
        gated_residual_bf16_contract().unwrap(),
    ];
    for (provider, contract, capability) in [
        (
            modulation.as_ref(),
            &contracts[0],
            MODULATION_SCALE_BF16_CAPABILITY_ID,
        ),
        (
            residual.as_ref(),
            &contracts[1],
            GATED_RESIDUAL_BF16_CAPABILITY_ID,
        ),
    ] {
        let descriptor = provider.descriptor();
        assert_eq!(descriptor.operation_id(), &contract.descriptor().id);
        assert_eq!(
            descriptor.operation_fingerprint(),
            contract.descriptor().fingerprint().unwrap()
        );
        assert_eq!(descriptor.version(), ContractVersion::new(1, 0));
        assert_eq!(
            descriptor.device_id(),
            &composition.runtime().descriptor().id
        );
        assert_eq!(
            descriptor.capabilities(),
            &BTreeSet::from([CapabilityId::new(capability).unwrap()])
        );
        assert!(descriptor.supports_fixed_tensor_work());
        assert!(descriptor.accepted_weight_formats().is_empty());
        assert!(descriptor.accepted_quantization_formats().is_empty());
    }
    let modulation = modulation.descriptor();
    let residual = residual.descriptor();
    assert_ne!(modulation.provider_id(), residual.provider_id());
    assert_ne!(
        modulation.provider_implementation_fingerprint(),
        residual.provider_implementation_fingerprint()
    );
    assert_ne!(
        modulation.resource_estimator_id(),
        residual.resource_estimator_id()
    );
    assert_ne!(
        modulation.resource_estimator_implementation_fingerprint(),
        residual.resource_estimator_implementation_fingerprint()
    );
}

#[test]
fn provider_geometry_preserves_both_explicit_parameter_row_mappings() {
    for kind in [
        ReferencePointwiseKind::ModulationScale,
        ReferencePointwiseKind::GatedResidual,
    ] {
        for (mode, parameter_rows) in [(RowBroadcast::Row, 1), (RowBroadcast::Elementwise, 2)] {
            let checked = geometry(
                kind,
                &values(kind, parameter_rows),
                &attributes(mode.as_str()),
            )
            .unwrap();
            assert_eq!(
                (
                    checked.rows(),
                    checked.width(),
                    checked.parameter_rows(),
                    checked.mode()
                ),
                (2, 3, parameter_rows, mode)
            );
        }
        assert!(geometry(kind, &values(kind, 1), &attributes("elementwise")).is_err());
        assert!(geometry(kind, &values(kind, 2), &attributes("row")).is_err());
        assert!(geometry(kind, &values(kind, 3), &attributes("elementwise")).is_err());
    }
}

#[test]
fn provider_geometry_rejects_wrong_tensor_shape_dtype_layout_and_gated_update() {
    for kind in [
        ReferencePointwiseKind::ModulationScale,
        ReferencePointwiseKind::GatedResidual,
    ] {
        for wrong_parameter in [
            tensor(&[3, 1], ElementType::Bf16),
            tensor(&[1, 3], ElementType::F16),
            tensor(&[3], ElementType::Bf16),
            ResolvedTensorSpec::new(
                vec![1, 3],
                ElementType::Bf16,
                ResolvedTensorLayout::Strided {
                    byte_strides: vec![8, 2],
                },
            )
            .unwrap(),
        ] {
            let mut bindings = values(kind, 1);
            bindings[1] = value(
                "pointwise-wrong-parameter",
                ResolvedValueRole::Input,
                1,
                wrong_parameter,
            );
            assert!(geometry(kind, &bindings, &attributes("row")).is_err());
        }
        for wrong_base in [
            tensor(&[2, 3], ElementType::F32),
            tensor(&[6], ElementType::Bf16),
        ] {
            let mut bindings = values(kind, 1);
            bindings[0] = value(
                "pointwise-wrong-input",
                ResolvedValueRole::Input,
                0,
                wrong_base,
            );
            assert!(geometry(kind, &bindings, &attributes("row")).is_err());
        }
        let mut bindings = values(kind, 1);
        let output = bindings.len() - 1;
        bindings[output] = value(
            "pointwise-wrong-output",
            ResolvedValueRole::Output,
            0,
            tensor(&[3, 2], ElementType::Bf16),
        );
        assert!(geometry(kind, &bindings, &attributes("row")).is_err());
    }
    for wrong_update in [
        tensor(&[3, 2], ElementType::Bf16),
        tensor(&[2, 3], ElementType::F16),
    ] {
        let kind = ReferencePointwiseKind::GatedResidual;
        let mut bindings = values(kind, 1);
        bindings[2] = value(
            "pointwise-wrong-update",
            ResolvedValueRole::Input,
            2,
            wrong_update,
        );
        assert!(geometry(kind, &bindings, &attributes("row")).is_err());
    }
}

#[test]
fn provider_geometry_rejects_missing_or_untyped_modes_and_incomplete_bindings() {
    for kind in [
        ReferencePointwiseKind::ModulationScale,
        ReferencePointwiseKind::GatedResidual,
    ] {
        let bindings = values(kind, 1);
        assert!(geometry(kind, &bindings, &BTreeMap::new()).is_err());
        for mode in ["", "Row", "scalar", "segmented_rows"] {
            assert!(geometry(kind, &bindings, &attributes(mode)).is_err());
        }
        let wrong_kind = BTreeMap::from([(
            AttributeId::new("broadcast").unwrap(),
            SemanticValue::Unsigned(1),
        )]);
        assert!(geometry(kind, &bindings, &wrong_kind).is_err());
        assert!(geometry(kind, &bindings[..bindings.len() - 1], &attributes("row")).is_err());
        let mut wrong_ordinal = bindings;
        wrong_ordinal[1] = value(
            "pointwise-wrong-ordinal",
            ResolvedValueRole::Input,
            7,
            tensor(&[1, 3], ElementType::Bf16),
        );
        assert!(geometry(kind, &wrong_ordinal, &attributes("row")).is_err());
    }
}
