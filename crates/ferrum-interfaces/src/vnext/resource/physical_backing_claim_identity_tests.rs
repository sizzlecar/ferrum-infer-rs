use super::*;

fn pool(digit: char) -> DynamicBackingPoolId {
    serde_json::from_value(serde_json::json!(format!(
        "dynamic-pool/sha256/{}",
        digit.to_string().repeat(64)
    )))
    .unwrap()
}

fn claim(digit: char, resources: &[&str]) -> PhysicalBackingClaimIdentity {
    PhysicalBackingClaimIdentity::new(
        pool(digit),
        resources
            .iter()
            .map(|resource| ResourceId::new(*resource).unwrap())
            .collect(),
    )
    .unwrap()
}

#[test]
fn physical_claim_identity_clone_shares_only_immutable_resource_ids() {
    let original = claim('a', &["resource-b", "resource-a"]);
    let cloned = original.clone();
    let independent = claim('a', &["resource-a", "resource-b"]);
    assert!(Arc::ptr_eq(&original.resource_ids, &cloned.resource_ids));
    assert!(!Arc::ptr_eq(
        &original.resource_ids,
        &independent.resource_ids
    ));
    assert_eq!(original, cloned);
    assert_eq!(original, independent);
    assert_eq!(original.cmp(&independent), std::cmp::Ordering::Equal);
    assert_eq!(original.pool_id(), &pool('a'));
    assert_eq!(original.resource_ids()[0].as_str(), "resource-a");
    assert!(original.is_shared());
    assert!(!claim('a', &["resource-a"]).is_shared());
}

#[test]
fn physical_claim_identity_order_and_equality_preserve_pool_and_resources() {
    let cases = [
        claim('a', &["resource-a"]),
        claim('a', &["resource-a", "resource-b"]),
        claim('a', &["resource-b"]),
        claim('b', &["resource-a"]),
    ];
    for (left_index, left) in cases.iter().enumerate() {
        for (right_index, right) in cases.iter().enumerate() {
            assert_eq!(left == right, left_index == right_index);
            assert_eq!(left.partial_cmp(right), Some(left_index.cmp(&right_index)));
            assert_eq!(left.cmp(right), left_index.cmp(&right_index));
        }
    }
    // Sharing the resource list never hides a different pool owner.
    let mut foreign_pool = cases[1].clone();
    foreign_pool.pool_id = pool('b');
    assert!(Arc::ptr_eq(
        &cases[1].resource_ids,
        &foreign_pool.resource_ids
    ));
    assert_ne!(cases[1], foreign_pool);
    assert!(cases[1] < foreign_pool);
}

#[test]
fn physical_claim_identity_serialization_matches_original_two_field_layout() {
    #[derive(Serialize)]
    struct LegacyIdentity {
        pool_id: DynamicBackingPoolId,
        resource_ids: Vec<ResourceId>,
    }
    for resources in [&["resource-a"][..], &["resource-b", "resource-a"][..]] {
        let current = claim('a', resources);
        let legacy = LegacyIdentity {
            pool_id: current.pool_id().clone(),
            resource_ids: current.resource_ids().to_vec(),
        };
        assert_eq!(
            serde_json::to_vec(&current).unwrap(),
            serde_json::to_vec(&legacy).unwrap()
        );
    }
    assert!(PhysicalBackingClaimIdentity::new(pool('a'), Vec::new()).is_err());
    let duplicate = ResourceId::new("resource-a").unwrap();
    assert!(
        PhysicalBackingClaimIdentity::new(pool('a'), vec![duplicate.clone(), duplicate]).is_err()
    );
}
