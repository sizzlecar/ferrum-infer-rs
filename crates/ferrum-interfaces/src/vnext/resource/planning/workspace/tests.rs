use super::*;

fn identity(names: &[&str]) -> PhysicalBackingClaimIdentity {
    let pool = serde_json::from_value(serde_json::json!(format!(
        "dynamic-pool/sha256/{}",
        "a".repeat(64)
    )))
    .unwrap();
    PhysicalBackingClaimIdentity::new(
        pool,
        names
            .iter()
            .map(|name| ResourceId::new(*name).unwrap())
            .collect(),
    )
    .unwrap()
}

#[test]
fn workspace_claim_identity_shared_storage_preserves_wire_and_value_order() {
    // A public caller can still invoke this API from a const function. Keeping
    // this wrapper const catches a source-compatibility regression at compile.
    const fn const_shared_query(identity: &PhysicalBackingClaimIdentity) -> bool {
        identity.is_shared()
    }
    let original = identity(&["resource.b", "resource.a"]);
    let shared = original.clone();
    let independent = identity(&["resource.a", "resource.b"]);
    assert!(original.shares_resource_id_storage(&shared));
    assert!(!original.shares_resource_id_storage(&independent));
    assert_eq!(original, independent);
    assert!(const_shared_query(&original));
    assert!(const_shared_query(&shared));
    assert!(!const_shared_query(&identity(&["resource.a"])));
    assert_eq!(original.cmp(&independent), std::cmp::Ordering::Equal);
    assert_eq!(
        serde_json::to_value(&original).unwrap(),
        serde_json::json!({
            "pool_id": format!("dynamic-pool/sha256/{}", "a".repeat(64)),
            "resource_ids": ["resource.a", "resource.b"],
        })
    );
    assert!(original < identity(&["resource.a", "resource.c"]));
}

#[test]
fn workspace_claim_identity_shortcut_requires_actual_shared_storage() {
    let original = identity(&["resource.a", "resource.b", "resource.c"]);
    let mut polls = 0;
    check_same_resource_ids(&original, &original.clone(), &mut || {
        polls += 1;
        true
    })
    .unwrap();
    assert_eq!(polls, 0, "immutable clones need no per-ID clock reads");
    let independent = identity(&["resource.a", "resource.b", "resource.c"]);
    check_same_resource_ids(&original, &independent, &mut || {
        polls += 1;
        true
    })
    .unwrap();
    assert_eq!(polls, original.resource_ids().len());
    assert_eq!(
        check_same_resource_ids(
            &original,
            &identity(&["resource.a", "resource.b", "resource.d"]),
            &mut || true
        ),
        Err(ResourcePlanningUnknown::InvalidDemand)
    );
    assert_eq!(
        check_same_resource_ids(&original, &independent, &mut || false),
        Err(ResourcePlanningUnknown::BudgetExhausted)
    );
}
