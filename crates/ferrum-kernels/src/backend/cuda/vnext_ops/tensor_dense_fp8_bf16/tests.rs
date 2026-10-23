use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, PhysicalWeightComponentBinding, PhysicalWeightLayout,
    QuantizationSpec, ResolvedStorageComponent, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueStorage, ResolvedWeightBinding, TensorAccess, WeightComponentRole,
    WeightComponentSpec, WeightEncoding, WeightSchema, WeightTensorSpec,
};

fn id<T: TryFrom<String, Error = VNextError>>(value: &str) -> T {
    T::try_from(value.to_owned()).unwrap()
}

fn schema() -> WeightSchema {
    WeightSchema {
        format_id: id(super::super::DENSE_SAFETENSORS_FORMAT_ID),
        layout_id: id("weight-layout.fp8-fixture"),
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: id("component.values"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["projection.weight".into()],
                dimensions: vec![32, 16],
                encoding: WeightEncoding::Quantized(QuantizationSpec::fp8_e4m3fn_whole_tensor_f32()),
                required: true,
            },
            WeightComponentSpec {
                id: id("component.scale"),
                role: WeightComponentRole::Scales,
                external_names: vec!["projection.scale".into()],
                dimensions: vec![1],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: id("weight.projection"),
            dimensions: vec![32, 16],
            logical_element_type: ElementType::Bf16,
            physical_layout: PhysicalWeightLayout::QuantizedTensor {
                packed_values: PhysicalWeightComponentBinding::exact_contiguous(id(
                    "component.values",
                )),
                packed_dimensions: vec![32, 16],
                scales: PhysicalWeightComponentBinding::exact_contiguous(id("component.scale")),
            },
            required: true,
        }],
    }
}

fn weight(schema: &WeightSchema) -> ResolvedValueBinding {
    let resolved = ResolvedWeightBinding::from_schema(schema, &id("weight.projection")).unwrap();
    let components = resolved
        .components()
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
        .collect();
    ResolvedValueBinding::new(
        id("value.weight"),
        ResolvedValueRole::Input,
        1,
        ResolvedTensorSpec::new(
            vec![32, 16],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(resolved),
        ResolvedValueStorage::composite(components).unwrap(),
    )
    .unwrap()
}

#[test]
fn whole_tensor_scale_is_resolved_by_identity_not_storage_order() {
    let binding = weight(&schema());
    // Canonical storage order puts scale first; the native ABI needs values first.
    assert_eq!(
        binding.storage().components()[0].element_type(),
        ElementType::F32
    );
    let components = weights::validate(&binding).unwrap();
    assert_eq!(components.packed.length_bytes(), 512);
    assert_eq!(components.packed.element_type(), ElementType::U8);
    assert_eq!(components.scale.length_bytes(), 4);
    assert_eq!(components.scale.element_type(), ElementType::F32);
}

#[test]
fn logically_valid_flattened_or_other_quantization_cannot_masquerade_as_oi_fp8() {
    let mut flattened = schema();
    flattened.components[0].dimensions = vec![512];
    let PhysicalWeightLayout::QuantizedTensor {
        packed_dimensions, ..
    } = &mut flattened.tensors[0].physical_layout
    else {
        unreachable!()
    };
    *packed_dimensions = vec![512];
    assert!(weights::validate(&weight(&flattened)).is_err());
    let mut different = schema();
    let WeightEncoding::Quantized(spec) = &mut different.components[0].encoding else {
        unreachable!()
    };
    spec.format_id = id("quantization.other-e4m3");
    assert!(weights::validate(&weight(&different)).is_err());
}

#[test]
fn fp8_factory_policy_is_explicit_bounded_and_fingerprinted() {
    let valid = CudaCutlassFp8Config {
        opaque_memory_bytes: 1,
        maximum_plans: 1,
        opaque_memory_profile: "experimental-unqualified-v1".into(),
    };
    assert!(valid.validate().is_ok());
    for invalid in [
        CudaCutlassFp8Config {
            opaque_memory_bytes: 0,
            ..valid.clone()
        },
        CudaCutlassFp8Config {
            maximum_plans: 0,
            ..valid.clone()
        },
        CudaCutlassFp8Config {
            maximum_plans: usize::MAX,
            ..valid.clone()
        },
        CudaCutlassFp8Config {
            opaque_memory_profile: "\n".into(),
            ..valid.clone()
        },
    ] {
        assert!(invalid.validate().is_err());
    }
    // Neither builder constructs a CUDA context. Default remains unadvertised.
    let ordinary = super::super::cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap();
    assert!(!ordinary
        .capabilities
        .contains(&id(DENSE_LINEAR_FP8_BF16_CAPABILITY_ID)));
    if !cutlass_fp8::compiled() {
        assert!(
            super::super::cuda_tensor_runtime_config_with_cutlass_fp8(0, id("device.cuda.0"))
                .is_err()
        );
    }
}

#[test]
fn prepared_identity_changes_with_native_geometry_and_workspace() {
    let device = Device {
        ordinal: 0,
        architecture: 1200,
        multiprocessors: 12,
        maximum_dynamic_shared_bytes: 100_000,
    };
    let a = cutlass_fp8::Plan::checked(Request::new(3, 16, 32, device).unwrap(), 4096, 0).unwrap();
    let b = cutlass_fp8::Plan::checked(a.request, 4096, 256).unwrap();
    let c = cutlass_fp8::Plan::checked(Request::new(4, 16, 32, device).unwrap(), 4096, 0).unwrap();
    let fingerprint = |p: &cutlass_fp8::Plan| {
        implementation_fingerprint(&[NUMERICAL_PROFILE_ID.as_bytes(), &p.fingerprint_material()])
    };
    assert_ne!(fingerprint(&a), fingerprint(&b));
    assert_ne!(fingerprint(&a), fingerprint(&c));
    assert!(a.scratch_bytes >= a.quantized_activation_bytes + a.activation_scale_bytes);
}
