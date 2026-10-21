use super::*;
use crate::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, DeviceId, ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueStorage, ResourceId, TensorAccess,
};

const KINDS: [ReferenceTranscendentalKind; 3] = [
    ReferenceTranscendentalKind::Silu,
    ReferenceTranscendentalKind::SiluMul,
    ReferenceTranscendentalKind::Tanh,
];

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

fn values(kind: ReferenceTranscendentalKind, dimensions: &[u64]) -> Vec<ResolvedValueBinding> {
    let tensor = tensor(dimensions, ElementType::Bf16);
    let mut values = (0..kind.input_count())
        .map(|ordinal| {
            value(
                &format!("activation-input{ordinal}"),
                ResolvedValueRole::Input,
                ordinal,
                tensor.clone(),
            )
        })
        .collect::<Vec<_>>();
    values.push(value(
        "activation-output",
        ResolvedValueRole::Output,
        0,
        tensor,
    ));
    values
}

#[test]
fn providers_select_independent_contracts_capabilities_and_numerical_estimators() {
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.activation-provider").unwrap(),
    )
    .unwrap();
    let mut providers = BTreeSet::new();
    let mut implementations = BTreeSet::new();
    let mut estimators = BTreeSet::new();
    let mut estimator_implementations = BTreeSet::new();
    for (kind, contract, capability) in [
        (
            ReferenceTranscendentalKind::Silu,
            silu_bf16_contract().unwrap(),
            SILU_BF16_CAPABILITY_ID,
        ),
        (
            ReferenceTranscendentalKind::SiluMul,
            silu_mul_bf16_contract().unwrap(),
            SILU_MUL_BF16_CAPABILITY_ID,
        ),
        (
            ReferenceTranscendentalKind::Tanh,
            tanh_bf16_contract().unwrap(),
            TANH_BF16_CAPABILITY_ID,
        ),
    ] {
        let provider = ReferenceTranscendentalProvider::new(composition.runtime(), kind).unwrap();
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
        assert!(providers.insert(descriptor.provider_id().as_str().to_owned()));
        assert!(implementations.insert(descriptor.provider_implementation_fingerprint().to_owned()));
        assert!(estimators.insert(descriptor.resource_estimator_id().to_owned()));
        assert!(estimator_implementations.insert(
            descriptor
                .resource_estimator_implementation_fingerprint()
                .to_owned()
        ));
        let repeated = ReferenceTranscendentalProvider::new(composition.runtime(), kind).unwrap();
        assert_eq!(
            descriptor.provider_implementation_fingerprint(),
            repeated.descriptor().provider_implementation_fingerprint()
        );
        assert_eq!(
            descriptor.resource_estimator_implementation_fingerprint(),
            repeated
                .descriptor()
                .resource_estimator_implementation_fingerprint()
        );
    }
}

#[test]
fn shared_provider_geometry_accepts_exact_shapes_and_rejects_implicit_up_broadcast() {
    for kind in KINDS {
        for dimensions in [[1, 1], [2, 3], [1, (1 << 24) + 1]] {
            let checked = geometry(kind, &values(kind, &dimensions), &BTreeMap::new()).unwrap();
            assert_eq!(
                (checked.rows(), checked.width()),
                (dimensions[0], dimensions[1])
            );
        }
        for wrong in [
            tensor(&[1, 3], ElementType::Bf16),
            tensor(&[3, 2], ElementType::Bf16),
            tensor(&[2, 3], ElementType::F16),
            tensor(&[2, 3], ElementType::F32),
            tensor(&[6], ElementType::Bf16),
            ResolvedTensorSpec::new(
                vec![2, 3],
                ElementType::Bf16,
                ResolvedTensorLayout::Strided {
                    byte_strides: vec![8, 2],
                },
            )
            .unwrap(),
        ] {
            for ordinal in 0..kind.input_count() {
                let mut bindings = values(kind, &[2, 3]);
                bindings[ordinal as usize] = value(
                    "activation-wrong-input",
                    ResolvedValueRole::Input,
                    ordinal,
                    wrong.clone(),
                );
                assert!(geometry(kind, &bindings, &BTreeMap::new()).is_err());
            }
            let mut bindings = values(kind, &[2, 3]);
            let output = bindings.len() - 1;
            bindings[output] = value(
                "activation-wrong-output",
                ResolvedValueRole::Output,
                0,
                wrong,
            );
            assert!(geometry(kind, &bindings, &BTreeMap::new()).is_err());
        }
    }
}

#[test]
fn provider_geometry_has_no_attributes_and_requires_complete_unique_port_ordinals() {
    for kind in KINDS {
        let bindings = values(kind, &[2, 3]);
        for (name, setting) in [
            ("broadcast", SemanticValue::Text("row".to_owned())),
            ("broadcast", SemanticValue::Text("elementwise".to_owned())),
            ("activation", SemanticValue::Text("silu".to_owned())),
            ("hidden_size", SemanticValue::Unsigned(3)),
        ] {
            let attributes = BTreeMap::from([(AttributeId::new(name).unwrap(), setting)]);
            assert!(geometry(kind, &bindings, &attributes).is_err());
        }
        assert!(geometry(kind, &bindings[..bindings.len() - 1], &BTreeMap::new()).is_err());
        let mut extra = bindings.clone();
        extra.push(value(
            "activation-extra",
            ResolvedValueRole::Input,
            9,
            tensor(&[2, 3], ElementType::Bf16),
        ));
        assert!(geometry(kind, &extra, &BTreeMap::new()).is_err());
        let mut wrong_input = bindings.clone();
        wrong_input[0] = value(
            "activation-wrong-ordinal",
            ResolvedValueRole::Input,
            9,
            tensor(&[2, 3], ElementType::Bf16),
        );
        assert!(geometry(kind, &wrong_input, &BTreeMap::new()).is_err());
        let mut wrong_output = bindings;
        let output = wrong_output.len() - 1;
        wrong_output[output] = value(
            "activation-wrong-output-ordinal",
            ResolvedValueRole::Output,
            1,
            tensor(&[2, 3], ElementType::Bf16),
        );
        assert!(geometry(kind, &wrong_output, &BTreeMap::new()).is_err());
    }
    let kind = ReferenceTranscendentalKind::SiluMul;
    let mut duplicated = values(kind, &[2, 3]);
    duplicated[1] = value(
        "activation-duplicate",
        ResolvedValueRole::Input,
        0,
        tensor(&[2, 3], ElementType::Bf16),
    );
    assert!(geometry(kind, &duplicated, &BTreeMap::new()).is_err());
}

#[test]
fn provider_geometry_rejects_logical_work_overflow_before_tensor_allocation() {
    // BF16 storage is representable, but four or five logical calls per
    // element exceed u64. These are descriptors only, with no payload backing.
    let width = u64::MAX / 4 + 1;
    let metadata = tensor(&[1, width], ElementType::Bf16);
    let bytes = width.checked_mul(2).unwrap();
    assert_eq!(metadata.minimum_storage_bytes().unwrap(), bytes);
    assert!(width.checked_mul(4).is_none());
    assert!(width.checked_mul(5).is_none());
    for kind in [
        ReferenceTranscendentalKind::Silu,
        ReferenceTranscendentalKind::SiluMul,
    ] {
        assert!(geometry(kind, &values(kind, &[1, width]), &BTreeMap::new()).is_err());
    }

    // Tanh has one logical call per element. Only the platform's actual
    // address-size conversion can prevent this metadata shape from fitting.
    let tanh = ReferenceTranscendentalKind::Tanh;
    let checked = geometry(tanh, &values(tanh, &[1, width]), &BTreeMap::new());
    if usize::try_from(width).is_ok() && usize::try_from(bytes).is_ok() {
        let checked = checked.unwrap();
        assert_eq!((checked.rows(), checked.width()), (1, width));
    } else {
        assert!(checked.is_err());
    }

    // This separate boundary distinguishes the SiLU-mul count from unary
    // SiLU instead of accepting one shared conservative four-call limit.
    let width = u64::MAX / 5 + 1;
    let bytes = width.checked_mul(2).unwrap();
    assert!(width.checked_mul(4).is_some());
    assert!(width.checked_mul(5).is_none());
    let silu_mul = ReferenceTranscendentalKind::SiluMul;
    assert!(geometry(silu_mul, &values(silu_mul, &[1, width]), &BTreeMap::new()).is_err());
    let silu = ReferenceTranscendentalKind::Silu;
    let checked = geometry(silu, &values(silu, &[1, width]), &BTreeMap::new());
    if usize::try_from(width).is_ok() && usize::try_from(bytes).is_ok() {
        assert_eq!(checked.unwrap().width(), width);
    } else {
        assert!(checked.is_err());
    }
}
