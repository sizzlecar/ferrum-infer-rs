//! Metadata-only tests. Native arithmetic is exercised through managed CUDA
//! integration tests, not simulated by a second implementation here.

use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, DeviceClass, DeviceId, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageView, ProgramValueId, ResolvedTensorLayout,
    ResolvedValueStorage, ResourceId, TensorAccess,
};

fn tensor(dims: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dims.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}

fn value(name: &str, role: ResolvedValueRole, ordinal: u32, dims: &[u64]) -> ResolvedValueBinding {
    let spec = tensor(dims, ElementType::Bf16);
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
        spec.minimum_storage_bytes().unwrap(),
        ElementType::Bf16,
    )
    .unwrap();
    ResolvedValueBinding::new(
        ProgramValueId::new(name).unwrap(),
        role,
        ordinal,
        spec,
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

fn attrs(kind: Kind, width: u64) -> BTreeMap<AttributeId, SemanticValue> {
    let mut values = BTreeMap::from([(
        AttributeId::new("epsilon").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
    )]);
    if let Some(name) = kind.width_attribute() {
        values.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Unsigned(width),
        );
    }
    values
}

#[test]
fn independent_norm_descriptors_preserve_weight_and_profile_boundaries() {
    let device = DeviceDescriptor {
        id: DeviceId::new("device.cuda.0").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: Kind::ALL
            .into_iter()
            .map(|k| CapabilityId::new(k.capability()).unwrap())
            .collect(),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    };
    let mut implementations = BTreeSet::new();
    let mut estimators = BTreeSet::new();
    for kind in Kind::ALL {
        let d = descriptor(&device, kind).unwrap();
        assert_eq!(d.operation_id().as_str(), kind.operation());
        assert_eq!(
            d.operation_fingerprint(),
            kind.contract().unwrap().descriptor().fingerprint().unwrap()
        );
        assert!(d.supports_fixed_tensor_work());
        assert!(d.accepted_quantization_formats().is_empty());
        assert_eq!(d.accepted_weight_formats().is_empty(), kind == Kind::Layer);
        assert!(implementations.insert(d.provider_implementation_fingerprint().to_owned()));
        assert!(estimators.insert(d.resource_estimator_implementation_fingerprint().to_owned()));
        assert_eq!(
            d.provider_implementation_fingerprint(),
            descriptor(&device, kind)
                .unwrap()
                .provider_implementation_fingerprint()
        );
        let mut absent = device.clone();
        absent
            .capabilities
            .remove(&CapabilityId::new(kind.capability()).unwrap());
        assert!(descriptor(&absent, kind).is_err());
    }
}

#[test]
fn actual_layer_and_head_geometry_use_bounded_shared_memory_without_expanded_gamma() {
    let epsilon = CanonicalRational::new(1, 1_000_000).unwrap();
    let input = tensor(&[3, 4096], ElementType::Bf16);
    let layer = validate_specs(Kind::Layer, &input, None, &input, 4096, epsilon).unwrap();
    assert_eq!(
        (layer.groups, layer.width, layer.bytes, layer.shared_bytes),
        (3, 4096, 24576, 16384)
    );
    assert_eq!(layer.arithmetic_operations, 3 * (6 * 4096 + 3));
    let gamma = tensor(&[128], ElementType::Bf16);
    let rms = validate_specs(Kind::HeadRms, &input, Some(&gamma), &input, 128, epsilon).unwrap();
    assert_eq!(
        (rms.groups, rms.width, rms.gamma_bytes, rms.shared_bytes),
        (96, 128, 256, 512)
    );
    assert_eq!(rms.arithmetic_operations, 96 * (4 * 128 + 3));
    assert_eq!(rms.elements, layer.elements);
    assert_eq!(
        rms.epsilon.to_bits(),
        ((1_f64 / 1_000_000_f64) as f32).to_bits()
    );
    let one = tensor(&[2, 1], ElementType::Bf16);
    assert!(validate_specs(Kind::Layer, &one, None, &one, 1, epsilon).is_ok());
    for width in [3, 8192] {
        let x = tensor(&[1, width], ElementType::Bf16);
        let g = tensor(&[width], ElementType::Bf16);
        assert!(validate_specs(Kind::Layer, &x, None, &x, width, epsilon).is_err());
        assert!(validate_specs(Kind::HeadRms, &x, Some(&g), &x, width, epsilon).is_err());
    }
}

#[test]
fn dtype_shape_head_grouping_and_epsilon_are_validated_before_encoding() {
    let x = tensor(&[2, 256], ElementType::Bf16);
    let gamma = tensor(&[128], ElementType::Bf16);
    let eps = CanonicalRational::new(1, 1_000_000).unwrap();
    for bad in [
        tensor(&[2, 128], ElementType::Bf16),
        tensor(&[128], ElementType::F16),
        tensor(&[64], ElementType::Bf16),
    ] {
        assert!(validate_specs(Kind::HeadRms, &x, Some(&bad), &x, 128, eps).is_err());
    }
    for bad in [
        tensor(&[4, 128], ElementType::Bf16),
        tensor(&[2, 256], ElementType::F16),
        tensor(&[512], ElementType::Bf16),
    ] {
        assert!(validate_specs(Kind::HeadRms, &x, Some(&gamma), &bad, 128, eps).is_err());
    }
    let indivisible = tensor(&[2, 192], ElementType::Bf16);
    assert!(validate_specs(
        Kind::HeadRms,
        &indivisible,
        Some(&gamma),
        &indivisible,
        128,
        eps
    )
    .is_err());
    assert!(validate_specs(Kind::HeadRms, &x, None, &x, 128, eps).is_err());
    assert!(validate_specs(Kind::Layer, &x, Some(&gamma), &x, 256, eps).is_err());
    for bad_eps in [
        CanonicalRational::new(0, 1).unwrap(),
        CanonicalRational::new(2, 1).unwrap(),
        CanonicalRational::new(1, 1_000_000_000_001).unwrap(),
    ] {
        assert!(validate_specs(Kind::HeadRms, &x, Some(&gamma), &x, 128, bad_eps).is_err());
        assert!(validate_specs(Kind::Layer, &x, None, &x, 256, bad_eps).is_err());
    }
}

#[test]
fn port_and_attribute_checks_are_shared_by_estimation_and_encoding() {
    for kind in Kind::ALL {
        let mut values = vec![value("input", ResolvedValueRole::Input, 0, &[2, 256])];
        if kind != Kind::Layer {
            let width = if kind == Kind::HeadRms { 128 } else { 256 };
            values.push(value("gamma", ResolvedValueRole::Input, 1, &[width]));
        }
        values.push(value("output", ResolvedValueRole::Output, 0, &[2, 256]));
        let width = if kind == Kind::HeadRms { 128 } else { 256 };
        let attributes = attrs(kind, width);
        assert!(geometry(kind, &values, &attributes).is_ok());
        let mut duplicate = values.clone();
        let first = duplicate[0].clone();
        *duplicate.last_mut().unwrap() = first;
        assert!(geometry(kind, &duplicate, &attributes).is_err());
        assert!(geometry(kind, &values[..values.len() - 1], &attributes).is_err());
        assert!(geometry(kind, &values, &BTreeMap::new()).is_err());
        let mut extra = attributes.clone();
        extra.insert(
            AttributeId::new("broadcast").unwrap(),
            SemanticValue::Text("row".into()),
        );
        assert!(geometry(kind, &values, &extra).is_err());
        let mut wrong = attributes;
        wrong.insert(
            AttributeId::new(kind.width_attribute().unwrap_or("hidden_size")).unwrap(),
            SemanticValue::Unsigned(width / 2),
        );
        assert!(geometry(kind, &values, &wrong).is_err());
    }
}

#[test]
fn metadata_only_byte_and_logical_work_limits_fail_without_allocating_payloads() {
    for kind in Kind::ALL {
        assert!(Shape::new(kind, 0, 128, 128, 1e-6).is_err());
        assert!(Shape::new(kind, 1, 128, 0, 1e-6).is_err());
        assert!(Shape::new(kind, u64::MAX, 128, 128, 1e-6).is_err());
        assert!(Shape::new(kind, 1, 128, 128, f32::NAN).is_err());
        let operations_per_group = match kind {
            Kind::Layer => 9,
            Kind::HeadRms => 7,
            Kind::ZeroCenteredRms => 8,
        };
        let groups = u64::MAX / operations_per_group + 1;
        if (groups as u128) * 2 <= isize::MAX as u128 {
            assert!(Shape::new(kind, groups, 1, 1, 1e-6).is_err());
        }
    }
}

#[test]
fn zero_centered_rms_is_whole_row_bf16_weight_with_one_epsilon_attribute() {
    let epsilon = CanonicalRational::new(1, 1_000_000).unwrap();
    let kind = Kind::ZeroCenteredRms;
    for width in [1, 128, 4096] {
        let x = tensor(&[3, width], ElementType::Bf16);
        let weight = tensor(&[width], ElementType::Bf16);
        let shape = validate_specs(kind, &x, Some(&weight), &x, width, epsilon).unwrap();
        assert_eq!(shape.groups, 3);
        assert_eq!(shape.width as u64, width);
        assert_eq!(shape.gamma_bytes, width * 2);
        assert_eq!(shape.bytes, 3 * width * 2);
        assert_eq!(shape.shared_bytes as u64, width * 4);
        assert_eq!(shape.arithmetic_operations, 3 * (5 * width + 3));
    }
    for width in [3, 8192] {
        let x = tensor(&[1, width], ElementType::Bf16);
        let weight = tensor(&[width], ElementType::Bf16);
        assert!(validate_specs(kind, &x, Some(&weight), &x, width, epsilon).is_err());
    }
    let x = tensor(&[2, 256], ElementType::Bf16);
    let weight = tensor(&[256], ElementType::Bf16);
    // A head-width gamma cannot implicitly broadcast across a whole row.
    for wrong in [
        tensor(&[128], ElementType::Bf16),
        tensor(&[1, 256], ElementType::Bf16),
        tensor(&[256], ElementType::F32),
        tensor(&[256], ElementType::F16),
    ] {
        assert!(validate_specs(kind, &x, Some(&wrong), &x, 256, epsilon).is_err());
    }
    for wrong in [
        tensor(&[4, 128], ElementType::Bf16),
        tensor(&[512], ElementType::Bf16),
        tensor(&[2, 256], ElementType::F16),
    ] {
        assert!(validate_specs(kind, &x, Some(&weight), &wrong, 256, epsilon).is_err());
    }
    assert!(validate_specs(kind, &x, None, &x, 256, epsilon).is_err());
    for bad in [
        CanonicalRational::new(0, 1).unwrap(),
        CanonicalRational::new(2, 1).unwrap(),
    ] {
        assert!(validate_specs(kind, &x, Some(&weight), &x, 256, bad).is_err());
    }
}
