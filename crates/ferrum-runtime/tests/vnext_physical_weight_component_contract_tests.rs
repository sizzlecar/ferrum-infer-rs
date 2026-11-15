use std::collections::BTreeSet;

use ferrum_runtime::*;
#[path = "vnext_physical_weight_component_contract_tests/numerical_or_physical.rs"]
mod numerical_or_physical;

fn id<T: TryFrom<String, Error = VNextError>>(value: &str) -> T {
    T::try_from(value.to_owned()).unwrap()
}

fn schema() -> WeightSchema {
    WeightSchema {
        format_id: id("weight-format.source"),
        layout_id: id("weight-layout.raw-view"),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: id("component.packed"),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["actual._nvfp4_weight".into()],
            dimensions: vec![8, 64],
            encoding: WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4()),
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: id("weight.raw"),
            dimensions: vec![8, 64],
            logical_element_type: ElementType::U8,
            physical_layout: PhysicalWeightLayout::PhysicalComponent(
                PhysicalWeightComponentView::new(id("component.packed")),
            ),
            required: true,
        }],
    }
}

fn contract() -> TensorContract {
    TensorContract::new(
        vec![
            DimensionConstraint::Exact(8),
            DimensionConstraint::Exact(64),
        ],
        BTreeSet::from([ElementType::U8]),
        vec![LayoutConstraint::Contiguous],
        TensorAccess::Read,
        AliasPolicy::NoAlias,
    )
    .unwrap()
}

fn descriptor(input: TensorContract) -> OperationDescriptor {
    OperationDescriptor {
        id: id("operation.physical-read"),
        version: ContractVersion::new(1, 0),
        inputs: vec![input],
        outputs: vec![TensorContract::new(
            vec![DimensionConstraint::Exact(1)],
            BTreeSet::from([ElementType::U8]),
            vec![LayoutConstraint::Contiguous],
            TensorAccess::Write,
            AliasPolicy::NoAlias,
        )
        .unwrap()],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 1,
            scratch: ResourcePresenceRequirement::Forbidden,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: OracleSpec::Exact,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::new(),
        },
        profile_phase: ProfilePhase::Forward,
    }
}

fn validate_input(contract: TensorContract, input: ResolvedValueBinding) -> Result<(), VNextError> {
    let output = ResolvedValueBinding::new(
        id("value.output"),
        ResolvedValueRole::Output,
        0,
        ResolvedTensorSpec::new(vec![1], ElementType::U8, ResolvedTensorLayout::Contiguous)
            .unwrap(),
        TensorAccess::Write,
        AliasPolicy::NoAlias,
        BufferUsage::Activations,
        None,
        ResolvedValueStorage::single(id("resource.output"), 0, 1, ElementType::U8).unwrap(),
    )
    .unwrap();
    descriptor(contract).validate_resolved_bindings(&[input, output])
}

fn binding(schema: &WeightSchema) -> ResolvedValueBinding {
    ResolvedValueBinding::new(
        id("value.packed"),
        ResolvedValueRole::Input,
        0,
        ResolvedTensorSpec::new(
            vec![8, 64],
            ElementType::U8,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(ResolvedWeightBinding::from_schema(schema, &id("weight.raw")).unwrap()),
        ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
            Some(id("component.packed")),
            id("resource.original-arena"),
            128,
            512,
            ElementType::U8,
        )
        .unwrap()])
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn numerical_ports_reject_raw_views_and_physical_ports_require_full_encoding_and_role() {
    let schema = schema();
    schema.validate(&id("family.physical")).unwrap();
    assert!(schema.tensors[0].logical_elements().is_err());
    assert!(schema.tensors[0].logical_bytes().is_err());
    let value = binding(&schema);
    assert!(validate_input(contract(), value.clone()).is_err());
    let opt_in = contract()
        .with_physical_weight_component(
            WeightComponentRole::PackedValues,
            schema.components[0].encoding.clone(),
        )
        .unwrap();
    validate_input(opt_in.clone(), value.clone()).unwrap();
    assert_eq!(
        value
            .weight()
            .unwrap()
            .physical_component_view()
            .unwrap()
            .encoding(),
        &schema.components[0].encoding
    );
    assert_eq!(
        value.weight().unwrap().quantization_formats(),
        BTreeSet::from([QuantizationSpec::nvfp4_e2m1_e4m3_128x4().format_id])
    );
    let mut changed = QuantizationSpec::nvfp4_e2m1_e4m3_128x4();
    changed.packing = QuantizationPacking::Linear;
    for policy in [
        contract()
            .with_physical_weight_component(
                WeightComponentRole::PackedValues,
                WeightEncoding::Quantized(changed),
            )
            .unwrap(),
        contract()
            .with_physical_weight_component(
                WeightComponentRole::Metadata,
                schema.components[0].encoding.clone(),
            )
            .unwrap(),
    ] {
        assert!(validate_input(policy, value.clone()).is_err());
    }
    let mut dense = schema.clone();
    dense.components[0].role = WeightComponentRole::Values;
    dense.components[0].encoding = WeightEncoding::Dense {
        element_type: ElementType::U8,
    };
    dense.tensors[0].physical_layout = PhysicalWeightLayout::Dense {
        component_id: id("component.packed"),
    };
    assert!(validate_input(opt_in, binding(&dense)).is_err());
}

#[test]
fn physical_views_reject_partial_ranges_dtype_conversion_nesting_and_unknown_wire_fields() {
    let original = schema();
    for dimensions in [vec![8, 63], vec![8, 65]] {
        let mut changed = original.clone();
        changed.tensors[0].dimensions = dimensions;
        assert!(changed.validate(&id("family.physical")).is_err());
    }
    let mut changed = original.clone();
    changed.tensors[0].logical_element_type = ElementType::F32;
    changed.tensors[0].dimensions = vec![128];
    assert!(changed.validate(&id("family.physical")).is_err());
    let mut nested = original.clone();
    nested.tensors[0].physical_layout = PhysicalWeightLayout::Composite {
        parts: vec![CompositeWeightPart {
            layout: Box::new(original.tensors[0].physical_layout.clone()),
            logical_offsets: vec![0, 0],
            extents: vec![8, 64],
        }],
    };
    assert!(nested.validate(&id("family.physical")).is_err());
    assert!(ResolvedWeightBinding::from_schema(&nested, &id("weight.raw")).is_err());
    for field in ["offset_bytes", "length_bytes", "storage", "encoding"] {
        let mut wire = serde_json::to_value(&original.tensors[0].physical_layout).unwrap();
        wire["physical_component"][field] = serde_json::json!(1);
        assert!(serde_json::from_value::<PhysicalWeightLayout>(wire).is_err());
    }
    let value = binding(&original);
    let wire = serde_json::to_value(&value).unwrap();
    let mut short = wire.clone();
    short["storage"]["components"][0]["length_bytes"] = serde_json::json!(511);
    assert!(serde_json::from_value::<ResolvedValueBinding>(short).is_err());
    let mut strided = wire;
    strided["tensor"]["layout"] = serde_json::json!({"strided": {"byte_strides": [64, 1]}});
    assert!(serde_json::from_value::<ResolvedValueBinding>(strided).is_err());
}

#[test]
fn rank_zero_source_scalar_is_only_legal_as_explicit_full_physical_view() {
    let mut scalar = schema();
    scalar.components[0].role = WeightComponentRole::Metadata;
    scalar.components[0].dimensions.clear();
    scalar.components[0].encoding = WeightEncoding::Dense {
        element_type: ElementType::F32,
    };
    scalar.tensors[0].dimensions = vec![1];
    scalar.tensors[0].logical_element_type = ElementType::F32;
    scalar.validate(&id("family.scalar")).unwrap();
    let resolved = ResolvedWeightBinding::from_schema(&scalar, &id("weight.raw")).unwrap();
    let decoded: ResolvedWeightBinding =
        serde_json::from_value(serde_json::to_value(&resolved).unwrap()).unwrap();
    decoded.validate_logical(&[1], ElementType::F32).unwrap();
    assert_eq!(decoded.components()[0].physical_bytes().unwrap(), 4);
    scalar.tensors[0].physical_layout = PhysicalWeightLayout::Dense {
        component_id: id("component.packed"),
    };
    scalar.components[0].role = WeightComponentRole::Values;
    assert!(scalar.validate(&id("family.scalar")).is_err());
}

#[test]
fn default_contract_serialization_keeps_legacy_fingerprint_bytes() {
    let numerical = contract();
    let old = serde_json::json!({
        "dimensions": [{"exact":8},{"exact":64}], "element_types":["u8"],
        "layouts":["contiguous"], "access":"read", "alias":"no_alias"
    });
    let decoded: TensorContract = serde_json::from_value(old.clone()).unwrap();
    assert_eq!(decoded, numerical);
    assert_eq!(serde_json::to_value(&numerical).unwrap(), old);
    let explicit = numerical
        .with_physical_weight_component(
            WeightComponentRole::PackedValues,
            schema().components[0].encoding.clone(),
        )
        .unwrap();
    assert_ne!(
        descriptor(decoded).fingerprint().unwrap(),
        descriptor(explicit.clone()).fingerprint().unwrap()
    );
    let roundtrip: TensorContract =
        serde_json::from_value(serde_json::to_value(&explicit).unwrap()).unwrap();
    assert_eq!(roundtrip, explicit);
    // Physical ports cannot expose mutable or strided raw source storage.
    for field in ["access", "layouts"] {
        let mut wire = serde_json::to_value(&explicit).unwrap();
        wire[field] = if field == "access" {
            serde_json::json!("read_write")
        } else {
            serde_json::json!([{ "strided": {"strides":[{"exact_bytes":64},{"exact_bytes":1}]}}])
        };
        assert!(serde_json::from_value::<TensorContract>(wire).is_err());
    }
}
