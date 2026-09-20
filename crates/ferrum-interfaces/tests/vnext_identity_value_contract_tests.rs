use ferrum_interfaces::vnext::{DynamicBackingPoolId, NodeId, PlanHash, ResourceId, VNextError};
use std::collections::{hash_map::DefaultHasher, BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

fn hash(value: &impl Hash) -> u64 {
    let mut state = DefaultHasher::new();
    value.hash(&mut state);
    state.finish()
}

#[test]
fn identities_keep_string_wire_order_hash_and_owned_conversion() {
    let independent = NodeId::new("node.a").unwrap();
    let copied = {
        let original = NodeId::new("node.a".to_owned()).unwrap();
        original.clone()
    };
    assert_eq!(copied, independent);
    assert_eq!(hash(&copied), hash(&"node.a"));
    assert_eq!(copied.as_str(), "node.a");
    assert_eq!(copied.to_string(), "node.a");
    assert_eq!(format!("{copied:?}"), "NodeId(\"node.a\")");
    assert_eq!(String::from(copied.clone()), "node.a");
    assert_eq!(serde_json::to_string(&copied).unwrap(), "\"node.a\"");
    assert_eq!(
        serde_json::from_str::<NodeId>("\"node.a\"").unwrap(),
        copied
    );

    let mut values = BTreeMap::new();
    values.insert(NodeId::new("node.z").unwrap(), 2);
    values.insert(copied, 1);
    values.insert(independent, 3);
    let wire = serde_json::to_string(&values).unwrap();
    assert_eq!(wire, "{\"node.a\":3,\"node.z\":2}");
    assert_eq!(
        serde_json::from_str::<BTreeMap<NodeId, u32>>(&wire).unwrap(),
        values
    );
}

#[test]
fn identity_parsing_retains_real_portability_and_length_boundaries() {
    let boundary = "a".repeat(160);
    assert_eq!(
        ResourceId::new(boundary.clone()).unwrap().as_str(),
        boundary
    );
    for invalid in [
        String::new(),
        "a".repeat(161),
        "resource contains spaces".to_owned(),
        "resource\n".to_owned(),
        "资源".to_owned(),
    ] {
        assert!(matches!(
            ResourceId::new(invalid.clone()),
            Err(VNextError::InvalidIdentity { value, .. }) if value == invalid
        ));
        assert!(serde_json::from_value::<ResourceId>(serde_json::json!(invalid)).is_err());
    }
    let portable = "resource.AZaz09._-:/";
    assert_eq!(ResourceId::new(portable).unwrap().as_str(), portable);
}

#[test]
fn plan_hash_clones_preserve_canonical_string_and_value_semantics() {
    let canonical = "0123456789abcdef".repeat(4);
    let independent: PlanHash = serde_json::from_value(serde_json::json!(canonical)).unwrap();
    let copied = {
        let original: PlanHash = serde_json::from_value(serde_json::json!(canonical)).unwrap();
        original.clone()
    };
    assert_eq!(copied, independent);
    assert_eq!(hash(&copied), hash(&canonical));
    assert_eq!(copied.as_str(), canonical);
    assert_eq!(copied.to_string(), canonical);
    assert_eq!(serde_json::to_value(&copied).unwrap(), canonical);
    assert_eq!(BTreeSet::from([copied, independent]).len(), 1);

    for invalid in [
        String::new(),
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        assert!(serde_json::from_value::<PlanHash>(serde_json::json!(invalid)).is_err());
    }
}

#[test]
fn dynamic_pool_ids_preserve_string_wire_order_hash_and_independent_value_equality() {
    let canonical = format!("dynamic-pool/sha256/{}", "0123456789abcdef".repeat(4));
    let independent: DynamicBackingPoolId =
        serde_json::from_value(serde_json::json!(canonical)).unwrap();
    let copied = {
        let original: DynamicBackingPoolId =
            serde_json::from_value(serde_json::json!(canonical)).unwrap();
        original.clone()
    };
    assert_eq!(copied, independent);
    assert_eq!(copied.as_str(), canonical);
    assert_eq!(hash(&copied), hash(&canonical));
    assert_eq!(
        format!("{copied:?}"),
        format!("DynamicBackingPoolId({canonical:?})")
    );
    assert_eq!(serde_json::to_value(&copied).unwrap(), canonical);

    let later = format!("dynamic-pool/sha256/{}", "f".repeat(64));
    let later_id: DynamicBackingPoolId = serde_json::from_value(serde_json::json!(later)).unwrap();
    let mut values = BTreeMap::new();
    values.insert(later_id, 2);
    values.insert(copied, 1);
    values.insert(independent, 3);
    let wire = serde_json::to_string(&values).unwrap();
    assert_eq!(wire, format!("{{\"{canonical}\":3,\"{later}\":2}}"));
    assert_eq!(
        serde_json::from_str::<BTreeMap<DynamicBackingPoolId, u32>>(&wire).unwrap(),
        values
    );
}

#[test]
fn dynamic_pool_id_parsing_preserves_exact_prefix_and_canonical_sha256_boundaries() {
    for invalid in [
        String::new(),
        "a".repeat(64),
        format!("dynamic-pool/sha256:{}", "a".repeat(64)),
        format!("dynamic-pool/SHA256/{}", "a".repeat(64)),
        "dynamic-pool/sha256/".to_owned(),
        format!("dynamic-pool/sha256/{}", "a".repeat(63)),
        format!("dynamic-pool/sha256/{}", "a".repeat(65)),
        format!("dynamic-pool/sha256/{}", "A".repeat(64)),
        format!("dynamic-pool/sha256/{}", "g".repeat(64)),
        format!("dynamic-pool/sha256/{}\n", "a".repeat(64)),
    ] {
        assert!(
            serde_json::from_value::<DynamicBackingPoolId>(serde_json::json!(invalid)).is_err()
        );
    }
}
