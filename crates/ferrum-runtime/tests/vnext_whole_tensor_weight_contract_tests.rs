use std::collections::BTreeSet;

use ferrum_runtime::*;
use serde_json::json;

fn id<T: TryFrom<String, Error = VNextError>>(value: &str) -> T {
    T::try_from(value.to_owned()).unwrap()
}

fn binding(component: &str) -> PhysicalWeightComponentBinding {
    PhysicalWeightComponentBinding::exact_contiguous(id(component))
}

fn schema() -> WeightSchema {
    WeightSchema {
        format_id: id("weight-format.whole-tensor-fixture"),
        layout_id: id("weight-layout.whole-tensor-fixture"),
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: id("component.values"),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["projection.weight".to_owned()],
                dimensions: vec![3, 5],
                encoding: WeightEncoding::Quantized(QuantizationSpec::fp8_e4m3fn_whole_tensor_f32()),
                required: true,
            },
            WeightComponentSpec {
                id: id("component.scale"),
                role: WeightComponentRole::Scales,
                external_names: vec!["projection.weight_scale".to_owned()],
                dimensions: vec![1],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: id("weight.projection"),
            dimensions: vec![3, 5],
            logical_element_type: ElementType::Bf16,
            physical_layout: PhysicalWeightLayout::QuantizedTensor {
                packed_values: binding("component.values"),
                packed_dimensions: vec![3, 5],
                scales: binding("component.scale"),
            },
            required: true,
        }],
    }
}

fn spec(schema: &mut WeightSchema) -> &mut QuantizationSpec {
    let WeightEncoding::Quantized(spec) = &mut schema.components[0].encoding else {
        panic!("fixture values must carry a quantization spec");
    };
    spec
}

fn validate(schema: &WeightSchema) -> Result<(), VNextError> {
    schema.validate(&id("family.whole-tensor-fixture"))
}

fn axis_schema() -> WeightSchema {
    let mut result = schema();
    let spec = spec(&mut result);
    spec.format_id = id("quantization.per-row-fixture");
    spec.grouping = QuantizationGrouping::WholeAxis;
    result.components[1].dimensions = vec![3, 1];
    result.tensors[0].physical_layout = PhysicalWeightLayout::Quantized {
        packed_values: binding("component.values"),
        packed_dimensions: vec![3, 5],
        scales: binding("component.scale"),
        zero_points: None,
        zero_point_packed_dimensions: None,
        axis_indices: None,
        permutation: None,
        codebook: None,
        group_axis: 1,
        group_padding: PhysicalWeightPadding::Exact,
    };
    result
}

#[test]
fn whole_tensor_scalar_scale_has_exact_physical_resource_bytes() {
    let schema = schema();
    validate(&schema).unwrap();
    assert_eq!(schema.tensors[0].logical_bytes().unwrap(), 30);
    assert_eq!(schema.physical_bytes(&id("weight.projection")).unwrap(), 19);
    let resources = schema
        .physical_resource_requirements(&id("weight.projection"))
        .unwrap();
    let values = resources
        .iter()
        .find(|value| value.component_id == id("component.values"))
        .unwrap();
    let scale = resources
        .iter()
        .find(|value| value.component_id == id("component.scale"))
        .unwrap();
    assert_eq!(
        (values.element_type, values.resource_bytes),
        (ElementType::U8, 15)
    );
    assert_eq!(values.physical_dimensions, [3, 5]);
    assert_eq!(
        (scale.element_type, scale.resource_bytes),
        (ElementType::F32, 4)
    );
    assert_eq!(scale.physical_dimensions, [1]);
    assert_eq!(
        schema.quantization_formats(),
        BTreeSet::from([id(FP8_E4M3FN_WHOLE_TENSOR_F32_FORMAT_ID)])
    );
    let resolved = ResolvedWeightBinding::from_schema(&schema, &id("weight.projection")).unwrap();
    resolved
        .validate_logical(&[3, 5], ElementType::Bf16)
        .unwrap();
    assert!(resolved
        .validate_logical(&[3, 4], ElementType::Bf16)
        .is_err());
    assert!(resolved.validate_logical(&[3, 5], ElementType::U8).is_err());
}

#[test]
fn whole_tensor_wire_roundtrip_and_fingerprints_retain_storage_identity() {
    let original = schema();
    let wire = serde_json::to_value(&original).unwrap();
    assert_eq!(
        wire["components"][0]["encoding"]["quantized"]["grouping"],
        json!({"kind": "whole_tensor"})
    );
    assert!(wire["tensors"][0]["physical_layout"]["quantized_tensor"].is_object());
    let restored: WeightSchema = serde_json::from_value(wire).unwrap();
    validate(&restored).unwrap();
    assert_eq!(restored, original);
    assert_eq!(
        restored.fingerprint().unwrap(),
        original.fingerprint().unwrap()
    );

    let mut flattened = original.clone();
    flattened.components[0].dimensions = vec![15];
    let PhysicalWeightLayout::QuantizedTensor {
        packed_dimensions, ..
    } = &mut flattened.tensors[0].physical_layout
    else {
        unreachable!();
    };
    *packed_dimensions = vec![15];
    validate(&flattened).unwrap();
    assert_ne!(
        flattened.fingerprint().unwrap(),
        original.fingerprint().unwrap()
    );
    let mut source_changed = original.clone();
    source_changed.components[1].external_names = vec!["different.weight_scale".to_owned()];
    validate(&source_changed).unwrap();
    assert_ne!(
        source_changed.fingerprint().unwrap(),
        original.fingerprint().unwrap()
    );

    let binding = ResolvedWeightBinding::from_schema(&original, &id("weight.projection")).unwrap();
    let binding_wire = serde_json::to_vec(&binding).unwrap();
    let restored: ResolvedWeightBinding = serde_json::from_slice(&binding_wire).unwrap();
    restored
        .validate_logical(&[3, 5], ElementType::Bf16)
        .unwrap();
    assert_eq!(restored, binding);
}

#[test]
fn whole_axis_old_wire_keeps_per_row_scales_and_cannot_stand_in_for_whole_tensor() {
    assert_eq!(QuantizationGrouping::WholeAxis.resolved_size(5), 5);
    assert_eq!(QuantizationGrouping::WholeTensor.resolved_size(5), 0);
    assert_eq!(QuantizationGrouping::WholeTensor.fixed_size(), None);
    assert_eq!(QuantizationGrouping::WholeTensor.block_shape_2d(), None);
    let old = axis_schema();
    validate(&old).unwrap();
    assert_eq!(old.physical_bytes(&id("weight.projection")).unwrap(), 27);
    let wire = serde_json::to_value(&old).unwrap();
    assert_eq!(
        wire["components"][0]["encoding"]["quantized"]["grouping"],
        json!({"kind": "whole_axis"})
    );
    let restored: WeightSchema = serde_json::from_value(wire).unwrap();
    validate(&restored).unwrap();
    assert_eq!(restored, old);
    assert_ne!(old.fingerprint().unwrap(), schema().fingerprint().unwrap());

    let mut whole_in_axis = old;
    spec(&mut whole_in_axis).grouping = QuantizationGrouping::WholeTensor;
    assert!(validate(&whole_in_axis).is_err());
    let mut axis_in_whole = schema();
    let spec = spec(&mut axis_in_whole);
    spec.format_id = id("quantization.generic-fixture");
    spec.grouping = QuantizationGrouping::WholeAxis;
    assert!(validate(&axis_in_whole).is_err());
}

#[test]
fn reserved_fp8_identity_rejects_reinterpretation_of_its_physical_abi() {
    let original = QuantizationSpec::fp8_e4m3fn_whole_tensor_f32();
    original.validate().unwrap();
    let mutations: [fn(&mut QuantizationSpec); 6] = [
        |spec| spec.bits_per_weight = 4,
        |spec| spec.grouping = QuantizationGrouping::WholeAxis,
        |spec| spec.grouping = QuantizationGrouping::fixed(32),
        |spec| spec.packing = QuantizationPacking::Tiled,
        |spec| spec.scale_type = ElementType::Bf16,
        |spec| spec.zero_point_type = Some(ElementType::U8),
    ];
    for mutate in mutations {
        let mut changed = original.clone();
        mutate(&mut changed);
        assert!(changed.validate().is_err());
        let restored: QuantizationSpec =
            serde_json::from_value(serde_json::to_value(&changed).unwrap()).unwrap();
        assert!(restored.validate().is_err());
        let mut invalid_schema = schema();
        invalid_schema.components[0].encoding = WeightEncoding::Quantized(restored);
        assert!(validate(&invalid_schema).is_err());
    }
}

#[test]
fn whole_tensor_rejects_scale_payload_role_and_component_mismatches() {
    let mutations: [fn(&mut WeightSchema); 9] = [
        |schema| schema.components[1].dimensions = vec![3, 1],
        |schema| schema.components[1].dimensions = vec![1, 1],
        |schema| {
            schema.components[1].encoding = WeightEncoding::Dense {
                element_type: ElementType::F16,
            }
        },
        |schema| schema.components[0].dimensions = vec![14],
        |schema| {
            schema.components[0].encoding = WeightEncoding::Dense {
                element_type: ElementType::U8,
            }
        },
        |schema| schema.components[1].role = WeightComponentRole::Values,
        |schema| schema.components[1].id = id("component.values"),
        |schema| {
            schema.components.pop();
        },
        |schema| {
            let spec = spec(schema);
            spec.format_id = id("quantization.generic-fixture");
            spec.zero_point_type = Some(ElementType::U8);
        },
    ];
    for mutate in mutations {
        let mut changed = schema();
        mutate(&mut changed);
        assert!(validate(&changed).is_err());
    }
    let mut changed = schema();
    let PhysicalWeightLayout::QuantizedTensor { scales, .. } =
        &mut changed.tensors[0].physical_layout
    else {
        unreachable!()
    };
    *scales = binding("component.values");
    assert!(validate(&changed).is_err());
}

#[test]
fn whole_tensor_metadata_rejects_empty_zero_and_overflowing_extents_without_allocating_payload() {
    for dimensions in [
        vec![],
        vec![0, 5],
        vec![u64::MAX, 2],
        vec![u64::MAX / 8 + 1],
    ] {
        let mut changed = schema();
        changed.tensors[0].dimensions = dimensions;
        assert!(validate(&changed).is_err());
    }
    for dimensions in [vec![], vec![0], vec![u64::MAX, 2], vec![14], vec![16]] {
        let mut changed = schema();
        let PhysicalWeightLayout::QuantizedTensor {
            packed_dimensions, ..
        } = &mut changed.tensors[0].physical_layout
        else {
            unreachable!()
        };
        *packed_dimensions = dimensions;
        assert!(validate(&changed).is_err());
    }
    let mut changed = schema();
    changed.components[1].dimensions = vec![u64::MAX / 4 + 1];
    assert!(changed.components[1].physical_bytes().is_err());
    assert!(validate(&changed).is_err());
}

#[test]
fn untrusted_resolved_binding_is_revalidated_against_logical_contract() {
    let original = ResolvedWeightBinding::from_schema(&schema(), &id("weight.projection")).unwrap();
    let mut wire = serde_json::to_value(&original).unwrap();
    let values = wire["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|component| component["component_id"] == "component.values")
        .unwrap();
    values["encoding"]["quantized"]["scale_type"] = json!("bf16");
    let untrusted: ResolvedWeightBinding = serde_json::from_value(wire).unwrap();
    assert!(untrusted
        .validate_logical(&[3, 5], ElementType::Bf16)
        .is_err());

    let mut wire = serde_json::to_value(&original).unwrap();
    wire["components"].as_array_mut().unwrap().pop();
    assert!(serde_json::from_value::<ResolvedWeightBinding>(wire).is_err());
}
