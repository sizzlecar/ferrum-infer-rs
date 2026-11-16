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
    assert_eq!(original.fingerprint, cloned.fingerprint);
    assert_eq!(original.fingerprint, independent.fingerprint);
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
            assert_eq!(
                left.fingerprint == right.fingerprint,
                left_index == right_index
            );
            assert_eq!(left.partial_cmp(right), Some(left_index.cmp(&right_index)));
            assert_eq!(left.cmp(right), left_index.cmp(&right_index));
        }
    }
    // Sharing the resource list never hides a different pool owner.
    let mut foreign_pool = claim('b', &["resource-a", "resource-b"]);
    foreign_pool.resource_ids = Arc::clone(&cases[1].resource_ids);
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

#[test]
fn allocation_fingerprint_binds_live_generation_and_projection_to_shared_membership() {
    let allocation = || LogicalBackingSliceAllocationEvidence {
        domain_id: CapacityDomainId::new(1).unwrap(),
        pool_id: pool('a'),
        resource_id: ResourceId::new("resource-a").unwrap(),
        pool_instance_id: 1,
        physical_claim_identity: claim('a', &["resource-a", "resource-b"]),
        reusable_execution_bucket_id: None,
        segment_generation: 1,
        segments: vec![BackingSegment::from_chunk(&pool('a'), 1, 1, 0, 64).unwrap()],
        physical_offset_bytes: 0,
        capacity_size_bytes: 64,
        physical_size_bytes: 128,
        alignment_bytes: 16,
        usage: BufferUsage::Scratch,
        element_type: ElementType::U8,
        storage_profile: DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            crate::DynamicStorageView::Contiguous,
        )
        .unwrap(),
        initialization: StateInitialization::None,
        fingerprint: String::new(),
    };
    let original = allocation().compute_fingerprint().unwrap();
    assert_eq!(original, allocation().compute_fingerprint().unwrap());
    let changes: &[fn(&mut LogicalBackingSliceAllocationEvidence)] = &[
        |a| a.pool_instance_id += 1,
        |a| a.segment_generation += 1,
        |a| a.domain_id = CapacityDomainId::new(2).unwrap(),
        |a| a.resource_id = ResourceId::new("resource-b").unwrap(),
        |a| a.physical_claim_identity = claim('a', &["resource-a", "resource-c"]),
        |a| {
            a.physical_claim_identity = claim('b', &["resource-a", "resource-b"]);
            a.pool_id = pool('b');
        },
        |a| a.physical_offset_bytes = 64,
        |a| a.capacity_size_bytes = 32,
        |a| a.physical_size_bytes = 256,
        |a| a.segments = vec![BackingSegment::from_chunk(&pool('a'), 1, 2, 0, 64).unwrap()],
        |a| a.segments = vec![BackingSegment::from_chunk(&pool('a'), 1, 1, 64, 64).unwrap()],
        |a| a.initialization = StateInitialization::Zero,
        |a| a.usage = BufferUsage::State,
        |a| a.element_type = ElementType::Bf16,
    ];
    for change in changes {
        let mut changed = allocation();
        change(&mut changed);
        assert_ne!(original, changed.compute_fingerprint().unwrap());
    }
}
