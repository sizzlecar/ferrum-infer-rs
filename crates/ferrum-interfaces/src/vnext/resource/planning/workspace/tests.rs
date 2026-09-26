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

struct HotClaimsFixture {
    domains: Vec<crate::vnext::resource::dynamic_pool::DynamicPoolDomainSpec>,
    claims: Vec<ClaimReadView>,
    projections: Vec<ProjectionReadView>,
}

#[test]
fn future_workspace_equality_keeps_slot_claim_and_projection_identity() {
    let fixture = HotClaimsFixture::new();
    let lane = ExecutionLaneId::mint().unwrap();
    let bucket = crate::vnext::ReusableExecutionBucketSpec::new(
        crate::vnext::ReusableExecutionClassId::new("workspace.equivalence").unwrap(),
        crate::vnext::ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let original = WorkspaceReadView {
        lane_id: lane,
        lane_epoch: 1,
        arena_clock: 2,
        next_slot_id: 2,
        slots: vec![SlotReadView {
            key: LaneStableArenaKey {
                lane_id: lane,
                lifetime: AllocationLifetime::Step,
                reusable_execution_bucket_id: bucket.bucket_id().clone(),
                layout_fingerprint: "a".repeat(64),
            },
            slot_id: 1,
            in_use: false,
            last_used: 1,
            claims: fixture.claims,
            projections: fixture.projections,
        }],
    };
    assert!(original
        .same_future_state(&original.clone(), &mut || true)
        .unwrap());
    let mut changed = original.clone();
    changed.slots[0].slot_id += 1;
    assert!(!original.same_future_state(&changed, &mut || true).unwrap());
    let mut changed = original.clone();
    changed.slots[0].in_use = true;
    assert!(!original.same_future_state(&changed, &mut || true).unwrap());
    let mut changed = original.clone();
    changed.slots[0].last_used += 1;
    assert!(!original.same_future_state(&changed, &mut || true).unwrap());
    let mut changed = original.clone();
    changed.slots[0].claims[0].pool_instance += 1;
    assert!(!original.same_future_state(&changed, &mut || true).unwrap());
    let mut changed = original.clone();
    changed.slots[0].projections[0].physical_offset += 1;
    assert!(!original.same_future_state(&changed, &mut || true).unwrap());
    let mut polls = 0;
    assert_eq!(
        original.same_future_state(&original, &mut || {
            polls += 1;
            polls < 5
        }),
        Err(ResourcePlanningUnknown::BudgetExhausted)
    );
}

impl HotClaimsFixture {
    fn new() -> Self {
        use crate::vnext::resource::dynamic_pool::DynamicPoolDomainSpec;
        use crate::vnext::{DynamicResourceDemand, DynamicStorageContract, MemoryPlan};

        let mut domains = Vec::new();
        let mut claims = Vec::new();
        let mut projections = Vec::new();
        // Distinct real pool descriptors, coalesced ranges and resource IDs.
        // The projection traversal deliberately interleaves the two claims.
        for (group, fingerprint) in ['d', 'e'].into_iter().enumerate() {
            let storage = DynamicStorageContract::resource_test_contract(
                DynamicStorageProfile::new(
                    DynamicStorageAllocator::LinearArena,
                    DynamicStorageView::Contiguous,
                )
                .unwrap(),
                fingerprint.to_string().repeat(64),
            )
            .unwrap();
            let descriptors: Vec<_> = (0..4)
                .map(|index| {
                    DynamicResourceDescriptor::resource_test_binding(
                        ResourceId::new(format!("resource/hot-{group}-{index}")).unwrap(),
                        DynamicResourceDemand::fixed(16 * (index + 1)).unwrap(),
                        16,
                        NodeId::new(format!("node/hot-{group}-{index}")).unwrap(),
                        storage.clone(),
                        8,
                    )
                    .unwrap()
                })
                .collect();
            let nodes: Vec<_> = descriptors
                .iter()
                .enumerate()
                .map(|(index, descriptor)| {
                    PlanNode::resource_test_node_with_binding(
                        NodeId::new(format!("node/hot-{group}-{index}")).unwrap(),
                        descriptor.base_resource_id().clone(),
                    )
                })
                .collect();
            let mut pools = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 160).unwrap();
            assert_eq!(pools.len(), 1);
            let identity = PhysicalBackingClaimIdentity::new(
                descriptors[0].pool_id().clone(),
                descriptors
                    .iter()
                    .map(|d| d.base_resource_id().clone())
                    .collect(),
            )
            .unwrap();
            let mut offset = 0;
            for (index, descriptor) in descriptors.iter().enumerate() {
                let capacity = 16 * (index as u64 + 1);
                projections.push(ProjectionReadView {
                    claim: group,
                    resource: descriptor.base_resource_id().clone(),
                    capacity,
                    physical_offset: offset,
                });
                offset += capacity;
            }
            claims.push(ClaimReadView {
                identity,
                physical_size: offset,
                pool_instance: group as u64 + 1,
                segment_generation: Some(1),
                // These fields are established by capture; matching is purely
                // numeric. The existing allocator regression covers capture.
                segments: Vec::new(),
            });
            domains.push(DynamicPoolDomainSpec {
                domain_id: CapacityDomainId::new(group as u32 + 1).unwrap(),
                pool: pools.remove(0),
                descriptors,
            });
        }
        projections.sort_by_key(|p| (p.physical_offset, p.claim));
        Self {
            domains,
            claims,
            projections,
        }
    }

    fn requests(&self) -> Vec<EvaluatedBackingRequest<'_>> {
        use crate::vnext::resource::dynamic_pool::EvaluatedBackingProjection;
        self.domains
            .iter()
            .enumerate()
            .map(|(claim_index, domain)| {
                let claim = &self.claims[claim_index];
                EvaluatedBackingRequest {
                    domain,
                    // Equal content in independent storage must still match.
                    claim_identity: PhysicalBackingClaimIdentity::new(
                        claim.identity.pool_id().clone(),
                        claim.identity.resource_ids().to_vec(),
                    )
                    .unwrap(),
                    capacity_size_bytes: claim.physical_size,
                    reusable_execution_bucket_id: None,
                    projections: domain
                        .descriptors
                        .iter()
                        .map(|descriptor| {
                            let p = self
                                .projections
                                .iter()
                                .find(|p| &p.resource == descriptor.base_resource_id())
                                .unwrap();
                            EvaluatedBackingProjection {
                                descriptor,
                                physical_offset_bytes: p.physical_offset,
                                logical_size_bytes: p.capacity - 1,
                                capacity_size_bytes: p.capacity,
                            }
                        })
                        .collect(),
                }
            })
            .collect()
    }
}

fn check_hot_claims(
    claims: &[ClaimReadView],
    projections: &[ProjectionReadView],
    requests: &[EvaluatedBackingRequest<'_>],
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    let mut canonical: Vec<_> = requests.iter().collect();
    canonical.sort_unstable_by(|a, b| a.claim_identity.cmp(&b.claim_identity));
    validate_idle_projections(claims, projections, &canonical, budget)
}

#[test]
fn workspace_hot_claim_match_keeps_every_interleaved_projection_check() {
    let fixture = HotClaimsFixture::new();
    let mut requests = fixture.requests();
    for (claim, request) in fixture.claims.iter().zip(&requests) {
        assert!(!claim
            .identity
            .shares_resource_id_storage(&request.claim_identity));
    }
    requests.reverse(); // A slot-local claim index is not a canonical index.
    let mut polls = 0;
    check_hot_claims(
        &fixture.claims,
        &fixture.projections,
        &requests,
        &mut || {
            polls += 1;
            true
        },
    )
    .unwrap();
    assert_eq!(polls, fixture.projections.len());
    for index in 2..fixture.projections.len() {
        // All these rows follow an already matched projection of this claim.
        for field in 0..4 {
            let mut projections = fixture.projections.clone();
            match field {
                0 => projections[index].physical_offset += 1,
                1 => projections[index].capacity += 1,
                2 => {
                    projections[index].resource = ResourceId::new("resource/not-in-claim").unwrap()
                }
                _ => projections[index].claim = fixture.claims.len(),
            }
            assert_eq!(
                check_hot_claims(&fixture.claims, &projections, &requests, &mut || true),
                Err(ResourcePlanningUnknown::InvalidDemand)
            );
        }
    }
}

#[test]
fn workspace_hot_claim_match_is_local_and_compares_complete_identity() {
    let fixture = HotClaimsFixture::new();
    let requests = fixture.requests();
    check_hot_claims(
        &fixture.claims,
        &fixture.projections,
        &requests,
        &mut || true,
    )
    .unwrap();
    let mut claims = fixture.claims.clone();
    let mut ids = claims[0].identity.resource_ids().to_vec();
    *ids.last_mut().unwrap() = ResourceId::new("resource/other-last-id").unwrap();
    claims[0].identity =
        PhysicalBackingClaimIdentity::new(claims[0].identity.pool_id().clone(), ids).unwrap();
    assert_eq!(
        check_hot_claims(&claims, &fixture.projections, &requests, &mut || true),
        Err(ResourcePlanningUnknown::InvalidDemand)
    );
    let mut claims = fixture.claims.clone();
    claims[1].physical_size += 16;
    assert_eq!(
        check_hot_claims(&claims, &fixture.projections, &requests, &mut || true),
        Err(ResourcePlanningUnknown::InvalidDemand)
    );
    check_hot_claims(
        &fixture.claims,
        &fixture.projections,
        &requests,
        &mut || true,
    )
    .unwrap();
}

#[test]
fn workspace_hot_claim_match_preserves_logical_demand_and_per_row_cancellation() {
    let fixture = HotClaimsFixture::new();
    for invalid_logical in [0, 65] {
        let mut requests = fixture.requests();
        requests[0].projections[3].logical_size_bytes = invalid_logical;
        assert_eq!(
            check_hot_claims(
                &fixture.claims,
                &fixture.projections,
                &requests,
                &mut || true
            ),
            Err(ResourcePlanningUnknown::InvalidDemand)
        );
    }
    let requests = fixture.requests();
    for stop in 1..=fixture.projections.len() {
        let mut polls = 0;
        assert_eq!(
            check_hot_claims(
                &fixture.claims,
                &fixture.projections,
                &requests,
                &mut || {
                    polls += 1;
                    polls < stop
                }
            ),
            Err(ResourcePlanningUnknown::BudgetExhausted)
        );
        assert_eq!(polls, stop);
    }
    check_hot_claims(
        &fixture.claims,
        &fixture.projections,
        &requests,
        &mut || true,
    )
    .unwrap();
}
