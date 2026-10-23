//! Internal comparison tests. Serialized declarations here do not confer plan
//! or allocation authority; the model integration test uses real compiled plans.
use super::*;
use crate::{
    ContractVersion, DynamicStorageAllocator, DynamicStorageContract, DynamicStorageProfile,
    DynamicStorageView, PhysicalWeightLayout, WeightComponentRole, WeightComponentSpec,
    WeightEncoding, WeightFormatId, WeightLayoutId, WeightSchema, WeightTensorSpec,
};
use serde_json::json;

fn allocation(bytes: u64) -> ResourceAllocation {
    let profile = DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap();
    serde_json::from_value(json!({
        "resource_id": "resource.shared.arena", "per_instance_bytes": bytes,
        "instance_stride_bytes": bytes, "instance_count": 1, "size_bytes": bytes,
        "alignment_bytes": 16, "usage": BufferUsage::Weights,
        "element_type": ElementType::Bf16, "lifetime": AllocationLifetime::Plan,
        "kind": AllocationKind::Value,
        "storage": DynamicStorageContract::resource_test_contract(profile, "a".repeat(64)).unwrap()
    }))
    .unwrap()
}

fn component(name: &str, offset: u64, bytes: u64, dtype: ElementType) -> ResolvedStorageComponent {
    ResolvedStorageComponent::new(
        Some(WeightId::new(name).unwrap()),
        ResourceId::new("resource.shared.arena").unwrap(),
        offset,
        bytes,
        dtype,
    )
    .unwrap()
}

fn source() -> SharedWeightContract {
    let components = ["weight.a", "weight.b"].map(|name| WeightComponentSpec {
        id: WeightId::new(name).unwrap(),
        role: WeightComponentRole::Values,
        external_names: vec![name.into()],
        dimensions: vec![4, 8],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        },
        required: true,
    });
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense").unwrap(),
        layout_id: WeightLayoutId::new("weight-layout.shared-subset.test").unwrap(),
        version: ContractVersion::new(1, 0),
        tensors: components
            .iter()
            .map(|c| WeightTensorSpec {
                id: WeightId::new(format!("{}.logical", c.id)).unwrap(),
                dimensions: c.dimensions.clone(),
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: c.id.clone(),
                },
                required: true,
            })
            .collect(),
        components: components.to_vec(),
    };
    let weights = serde_json::from_value(json!({
        "source_schema_fingerprint": schema.fingerprint().unwrap(),
        "materializer_id": "weight-materializer.shared.test", "materializer_version": ContractVersion::new(1, 0),
        "materializer_implementation_fingerprint": "b".repeat(64),
        "component_sources": { "weight.a": ["weight.a"], "weight.b": ["weight.b"] },
        "schema": schema
    })).unwrap();
    SharedWeightContract {
        weights,
        allocations: vec![allocation(128)],
        placements: BTreeMap::from([
            (
                components[0].id.clone(),
                component("weight.a", 0, 64, ElementType::Bf16),
            ),
            (
                components[1].id.clone(),
                component("weight.b", 64, 64, ElementType::Bf16),
            ),
        ]),
    }
}
fn subset(source: &SharedWeightContract) -> SharedWeightContract {
    let mut consumer = source.clone();
    consumer
        .placements
        .remove(&WeightId::new("weight.b").unwrap());
    consumer.allocations = vec![allocation(64)];
    consumer
}

#[test]
fn shared_subset_keeps_original_offsets_and_cannot_expand_or_reinterpret() {
    let source = source();
    let consumer = subset(&source);
    assert!(source.supports(&source));
    assert!(source.supports(&consumer));
    assert!(!consumer.supports(&source));
    for wrong in [
        component("weight.a", 16, 64, ElementType::Bf16),
        component("weight.a", 0, 32, ElementType::Bf16),
        component("weight.a", 0, 64, ElementType::F16),
        component("weight.unknown", 0, 64, ElementType::Bf16),
    ] {
        let mut bad = consumer.clone();
        bad.placements = BTreeMap::from([(wrong.component_id().unwrap().clone(), wrong)]);
        assert!(!source.supports(&bad));
    }
    let mut too_large = consumer.clone();
    too_large.allocations = vec![allocation(144)];
    assert!(!source.supports(&too_large));
}

#[test]
fn shared_subset_requires_complete_schema_source_and_materializer_identity() {
    let source = source();
    let consumer = subset(&source);
    for field in [
        "schema",
        "source_schema_fingerprint",
        "component_sources",
        "materializer_id",
        "materializer_implementation_fingerprint",
    ] {
        let mut wire = serde_json::to_value(&consumer.weights).unwrap();
        match field {
            "schema" => wire["schema"]["tensors"][0]["dimensions"] = json!([8, 4]),
            "component_sources" => wire["component_sources"]["weight.a"] = json!(["weight.b"]),
            "materializer_id" => wire[field] = json!("weight-materializer.different.test"),
            _ => wire[field] = json!("c".repeat(64)),
        }
        let mut bad = consumer.clone();
        bad.weights = serde_json::from_value(wire).unwrap();
        assert!(
            !source.supports(&bad),
            "identity difference must reject: {field}"
        );
    }
}
