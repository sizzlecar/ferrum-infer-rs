//! Pure metadata/geometry tests; native arithmetic and module lifetime require
//! the separate managed CUDA execution tests on an actual device.

use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, DeviceClass, DeviceId, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageView, ProgramValueId, ResolvedTensorLayout,
    ResolvedTensorSpec, ResolvedValueStorage, ResourceId, RowBroadcast, TensorAccess,
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

fn values(kind: Kind, parameter_rows: u64) -> Vec<ResolvedValueBinding> {
    let mut values = (0..kind.inputs())
        .map(|ordinal| {
            let rows = if kind.broadcasts() && ordinal == 1 {
                parameter_rows
            } else {
                2
            };
            value(
                &format!("pointwise-input{ordinal}"),
                ResolvedValueRole::Input,
                ordinal,
                tensor(&[rows, 3], ElementType::Bf16),
            )
        })
        .collect::<Vec<_>>();
    values.push(value(
        "pointwise-output",
        ResolvedValueRole::Output,
        0,
        tensor(&[2, 3], ElementType::Bf16),
    ));
    values
}

fn attributes(mode: &str) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("broadcast").unwrap(),
        SemanticValue::Text(mode.into()),
    )])
}

#[test]
fn independent_descriptors_bind_contracts_capabilities_profiles_and_estimators() {
    let device = DeviceDescriptor {
        id: DeviceId::new("device.cuda.0").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: Kind::ALL
            .into_iter()
            .map(|kind| CapabilityId::new(kind.capability()).unwrap())
            .collect(),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    };
    let mut implementation_ids = BTreeSet::new();
    let mut estimator_ids = BTreeSet::new();
    for kind in Kind::ALL {
        let selected = descriptor(&device, kind).unwrap();
        let contract = kind.contract().unwrap();
        assert_eq!(selected.operation_id(), &contract.descriptor().id);
        assert_eq!(
            selected.operation_fingerprint(),
            contract.descriptor().fingerprint().unwrap()
        );
        assert_eq!(
            selected.capabilities(),
            &BTreeSet::from([CapabilityId::new(kind.capability()).unwrap()])
        );
        assert!(selected.supports_fixed_tensor_work());
        assert!(selected.accepted_weight_formats().is_empty());
        assert!(selected.accepted_quantization_formats().is_empty());
        assert!(
            implementation_ids.insert(selected.provider_implementation_fingerprint().to_owned())
        );
        assert!(estimator_ids.insert(
            selected
                .resource_estimator_implementation_fingerprint()
                .to_owned()
        ));
        assert_eq!(
            selected.provider_implementation_fingerprint(),
            descriptor(&device, kind)
                .unwrap()
                .provider_implementation_fingerprint()
        );
        let mut missing = device.clone();
        missing
            .capabilities
            .remove(&CapabilityId::new(kind.capability()).unwrap());
        assert!(descriptor(&missing, kind).is_err());
    }
}

#[test]
fn same_shape_paths_reject_broadcast_dtype_and_equal_numel_reshapes() {
    for kind in [
        Kind::Silu,
        Kind::SiluMul,
        Kind::Tanh,
        Kind::GeluTanh,
        Kind::PlainResidualAdd,
    ] {
        let bindings = values(kind, 2);
        let checked = geometry(kind, &bindings, &BTreeMap::new()).unwrap();
        assert_eq!(
            (
                checked.elements,
                checked.bytes,
                checked.arithmetic_operations
            ),
            (6, 12, 6 * kind.operations_per_element())
        );
        assert!(geometry(kind, &bindings, &attributes("row")).is_err());
        for wrong in [
            tensor(&[3, 2], ElementType::Bf16),
            tensor(&[1, 3], ElementType::Bf16),
            tensor(&[2, 3], ElementType::F16),
            tensor(&[6], ElementType::Bf16),
        ] {
            for ordinal in 0..kind.inputs() {
                let mut wrong_values = bindings.clone();
                wrong_values[ordinal as usize] =
                    value("wrong", ResolvedValueRole::Input, ordinal, wrong.clone());
                assert!(geometry(kind, &wrong_values, &BTreeMap::new()).is_err());
            }
        }
    }
}

#[test]
fn row_parameters_are_not_materialized_or_inferred_from_extents() {
    for kind in [Kind::Modulation, Kind::GatedResidual] {
        for (mode, rows, parameter_bytes) in [("row", 1, 6), ("elementwise", 2, 12)] {
            let checked = geometry(kind, &values(kind, rows), &attributes(mode)).unwrap();
            assert_eq!(
                (checked.elements, checked.width, checked.parameter_bytes),
                (6, 3, parameter_bytes)
            );
            assert_eq!(checked.row_broadcast, mode == "row");
            assert_eq!(checked.arithmetic_operations, 12);
        }
        assert!(geometry(kind, &values(kind, 1), &attributes("elementwise")).is_err());
        assert!(geometry(kind, &values(kind, 2), &attributes("row")).is_err());
        assert!(geometry(kind, &values(kind, 1), &attributes("auto")).is_err());
        assert!(geometry(kind, &values(kind, 1), &BTreeMap::new()).is_err());
    }
    let mut wrong_update = values(Kind::GatedResidual, 1);
    wrong_update[2] = value(
        "wrong-update",
        ResolvedValueRole::Input,
        2,
        tensor(&[3, 2], ElementType::Bf16),
    );
    assert!(geometry(Kind::GatedResidual, &wrong_update, &attributes("row")).is_err());
}

#[test]
fn metadata_preflight_rejects_duplicate_or_missing_ports_and_work_overflow() {
    for kind in Kind::ALL {
        let attrs = if kind.broadcasts() {
            attributes("row")
        } else {
            BTreeMap::new()
        };
        let mut bindings = values(kind, 1);
        bindings.pop();
        assert!(geometry(kind, &bindings, &attrs).is_err());
        bindings.push(bindings[0].clone());
        assert!(geometry(kind, &bindings, &attrs).is_err());
        assert!(Shape::new(kind, 0, 3, false).is_err());
        assert!(Shape::new(kind, u64::MAX, 2, false).is_err());
    }
    let fits_bytes_but_not_five_operations = u64::MAX / 5 + 1;
    if (fits_bytes_but_not_five_operations as u128) * 2 <= isize::MAX as u128 {
        assert!(Shape::new(Kind::Silu, 1, fits_bytes_but_not_five_operations, false).is_ok());
        assert!(Shape::new(Kind::SiluMul, 1, fits_bytes_but_not_five_operations, false).is_err());
    }
    assert!(geometry::validate_specs(
        Kind::SiluMul,
        &[
            &tensor(&[2, 3], ElementType::Bf16),
            &tensor(&[2, 3], ElementType::Bf16)
        ],
        &tensor(&[2, 3], ElementType::Bf16),
        Some(RowBroadcast::Row)
    )
    .is_err());
}

#[test]
fn physical_alias_checks_use_each_range_length_and_reject_address_overflow() {
    assert!(nonoverlapping_ranges(16, 6, 32, 12).is_ok());
    assert!(nonoverlapping_ranges(32, 12, 16, 16).is_ok());
    assert!(nonoverlapping_ranges(16, 6, 16, 12).is_err());
    assert!(nonoverlapping_ranges(16, 6, 20, 12).is_err());
    assert!(nonoverlapping_ranges(32, 12, 28, 8).is_err());
    assert!(nonoverlapping_ranges(u64::MAX - 4, 8, 16, 8).is_err());
    assert!(nonoverlapping_ranges(16, 8, u64::MAX - 4, 8).is_err());
}
