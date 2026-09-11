mod vnext_core_contract;
use vnext_core_contract::*;
#[path = "vnext_sequence_checkpoint_contract/fixture.rs"]
mod fixture;
use fixture::{Fixture, Spec};

fn policy(bytes: u64) -> Option<CheckpointCapacityPolicy> {
    Some(CheckpointCapacityPolicy::new(bytes).unwrap())
}

fn enabled(bytes: u64) -> Fixture {
    Fixture::build(Spec {
        checkpoint_capacity: policy(bytes),
        ..Spec::default()
    })
    .unwrap()
}

fn growth_pools(fixture: &Fixture) -> Vec<&DynamicBackingPoolSpec> {
    fixture
        .plan
        .payload()
        .memory()
        .dynamic_pools()
        .iter()
        .filter(|pool| pool.checkpoint_growth_ceiling_bytes() != 0)
        .collect()
}

#[test]
fn checkpoint_capacity_default_preserves_legacy_wire_and_fingerprints() {
    let fixture = Fixture::build(Spec::default()).unwrap();
    let wire = fixture.plan.to_json().unwrap();
    let mut value: Value = serde_json::from_slice(&wire).unwrap();
    let memory = &value["payload"]["memory"];
    assert!(memory.get("checkpoint_capacity").is_none());
    for pool in memory["dynamic_pools"].as_array().unwrap() {
        assert!(pool.get("checkpoint_growth_ceiling_bytes").is_none());
    }
    // The legacy payload contains none of the newly optional fields. Hashing
    // exactly that payload still yields the same plan identity.
    rehash_plan_json(&mut value);
    assert_eq!(
        value["plan_hash"],
        serde_json::to_value(fixture.plan.plan_hash()).unwrap()
    );
    assert_eq!(fixture.revalidate(&wire).unwrap().to_json().unwrap(), wire);
    let mut policy_wire = serde_json::to_value(&fixture.policy).unwrap();
    let memory_wire = &policy_wire["memory"];
    assert_eq!(
        *memory_wire,
        json!({
            "capacity_bytes": fixture.policy.memory().capacity_bytes,
            "reserve_bytes": fixture.policy.memory().reserve_bytes,
            "maximum_active_sequences": fixture.policy.memory().maximum_active_sequences,
            "dynamic_storage_profile_order": fixture.policy.memory().dynamic_storage_profile_order,
        })
    );
    let mut legacy_fingerprint_material = policy_wire.clone();
    legacy_fingerprint_material
        .as_object_mut()
        .unwrap()
        .remove("fingerprint");
    let digest = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&canonical_json(legacy_fingerprint_material)).unwrap())
    );
    assert_eq!(digest, fixture.policy.fingerprint_str());
    policy_wire["memory"]["checkpoint_capacity"] = Value::Null;
    let round_trip: ResolvedRuntimePolicy = serde_json::from_value(policy_wire).unwrap();
    assert_eq!(round_trip, fixture.policy);
}

#[test]
fn checkpoint_capacity_changes_only_authenticated_pool_growth_and_identity() {
    let ordinary = Fixture::build(Spec::default()).unwrap();
    let enabled = enabled(4096);
    let before = ordinary.plan.payload().memory();
    let after = enabled.plan.payload().memory();
    assert_ne!(
        ordinary.policy.fingerprint_str(),
        enabled.policy.fingerprint_str()
    );
    assert_ne!(ordinary.plan.plan_hash(), enabled.plan.plan_hash());
    assert_eq!(
        before.maximum_active_sequences(),
        after.maximum_active_sequences()
    );
    assert_eq!(after.maximum_active_sequences(), 3);
    assert_eq!(
        before.minimum_runnable_request_bytes(),
        after.minimum_runnable_request_bytes()
    );
    assert_eq!(
        before.theoretical_ceiling_bytes(),
        after.theoretical_ceiling_bytes()
    );
    assert_eq!(before.static_allocations(), after.static_allocations());
    assert_eq!(before.dynamic_descriptors(), after.dynamic_descriptors());
    assert_eq!(
        after
            .checkpoint_capacity()
            .unwrap()
            .maximum_retained_bytes(),
        4096
    );
    let allowed_pools = enabled
        .layout()
        .states()
        .iter()
        .map(|state| {
            after
                .dynamic_descriptors()
                .iter()
                .find(|descriptor| descriptor.base_resource_id() == state.resource_id())
                .unwrap()
                .pool_id()
        })
        .collect::<BTreeSet<_>>();
    assert!(!allowed_pools.is_empty());
    for pool in after.dynamic_pools() {
        let original = before
            .dynamic_pools()
            .iter()
            .find(|original| original.pool_id() == pool.pool_id())
            .unwrap();
        let extra = if allowed_pools.contains(pool.pool_id()) {
            4096
        } else {
            0
        };
        assert_eq!(pool.checkpoint_growth_ceiling_bytes(), extra);
        assert_eq!(
            pool.minimum_sequence_bytes(),
            original.minimum_sequence_bytes()
        );
        assert_eq!(
            pool.theoretical_ceiling_bytes(),
            original.theoretical_ceiling_bytes()
        );
        assert_eq!(
            u128::from(pool.provisioning().maximum_resident_bytes()),
            (pool.theoretical_ceiling_bytes()
                + u128::from(pool.reusable_workspace_ceiling_bytes())
                + u128::from(extra))
            .min(u128::from(
                after.usable_capacity_bytes() - after.static_bytes()
            ))
        );
    }
    assert_eq!(
        enabled
            .revalidate(&enabled.plan.to_json().unwrap())
            .unwrap(),
        enabled.plan
    );
}

#[test]
fn checkpoint_capacity_does_not_infer_support_from_sequence_state_or_unselected_provider() {
    for spec in [
        Spec {
            declare_inputs: false,
            checkpoint_capacity: policy(4096),
            ..Spec::default()
        },
        Spec {
            declare_provider: false,
            unselected_support: true,
            checkpoint_capacity: policy(4096),
            ..Spec::default()
        },
        Spec {
            hidden_persistent: true,
            checkpoint_capacity: policy(4096),
            ..Spec::default()
        },
    ] {
        let fixture = Fixture::build(spec).unwrap();
        assert!(!fixture.reasons().is_empty());
        assert!(growth_pools(&fixture).is_empty());
        assert_eq!(
            fixture
                .revalidate(&fixture.plan.to_json().unwrap())
                .unwrap(),
            fixture.plan
        );
    }
}

#[test]
fn checkpoint_capacity_is_per_pool_permission_not_per_alias_or_per_state_quota() {
    let mut spec = Spec {
        checkpoint_capacity: policy(4096),
        ..Spec::default()
    };
    spec.states[0].capacity_demand = StateCapacityDemand::FixedPerScope;
    spec.states[0].checkpoint = spec.states[1].checkpoint;
    spec.layouts[0] = ProviderCheckpointStateLayout::ContiguousBoundaryValue;
    spec.locations = vec![(id("resource.shared"), 0), (id("resource.shared"), 16)];
    let fixture = Fixture::build(spec).unwrap();
    assert_eq!(fixture.layout().states().len(), 2);
    assert_eq!(
        fixture
            .plan
            .checkpoint_byte_plan(3)
            .unwrap()
            .resources()
            .len(),
        1
    );
    let pools = growth_pools(&fixture);
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0].checkpoint_growth_ceiling_bytes(), 4096);
    assert_eq!(pools[0].resource_ids().len(), 1);

    let mut separate_pools = Spec {
        checkpoint_capacity: policy(4096),
        ..Spec::default()
    };
    separate_pools.states[1].tensor.element_type = ElementType::F32;
    let separate = Fixture::build(separate_pools).unwrap();
    let pools = growth_pools(&separate);
    assert_eq!(pools.len(), 2);
    assert!(pools
        .iter()
        .all(|pool| pool.checkpoint_growth_ceiling_bytes() == 4096));
    // The two physical compatibility classes share the same aggregate cap.
    assert_eq!(
        separate
            .plan
            .payload()
            .memory()
            .checkpoint_capacity()
            .unwrap()
            .maximum_retained_bytes(),
        4096
    );
}

#[test]
fn checkpoint_capacity_rounds_down_to_physical_quantum_and_clamps_growth_to_device_budget() {
    for (cap, expected) in [(65535, 0), (65537, 65536), (u64::MAX, u64::MAX - 65535)] {
        let fixture = Fixture::build(Spec {
            checkpoint_capacity: policy(cap),
            profile: paged_storage_profile(65536),
            port_profile: paged_storage_profile(65536),
            ..Spec::default()
        })
        .unwrap();
        assert_eq!(fixture.layout().states().len(), 2);
        let memory = fixture.plan.payload().memory();
        let state_pools = memory
            .dynamic_pools()
            .iter()
            .filter(|pool| pool.compatibility().usage() == BufferUsage::State)
            .collect::<Vec<_>>();
        assert!(!state_pools.is_empty());
        for pool in state_pools {
            assert_eq!(pool.checkpoint_growth_ceiling_bytes(), expected);
            assert!(
                pool.provisioning().maximum_resident_bytes()
                    <= memory.usable_capacity_bytes() - memory.static_bytes()
            );
        }
        assert_eq!(
            fixture
                .revalidate(&fixture.plan.to_json().unwrap())
                .unwrap(),
            fixture.plan
        );
    }
}

#[test]
fn checkpoint_capacity_rejects_invalid_policy_and_checked_ceiling_overflow() {
    assert!(CheckpointCapacityPolicy::new(0).is_err());
    for wire in [
        json!({"maximum_retained_bytes": 0}),
        json!({"maximum_retained_bytes": -1}),
        json!({"maximum_retained_bytes": 1, "unknown_mode": true}),
    ] {
        assert!(serde_json::from_value::<CheckpointCapacityPolicy>(wire).is_err());
    }
    let fixture = enabled(4096);
    let mut pool = serde_json::to_value(growth_pools(&fixture)[0]).unwrap();
    pool["theoretical_ceiling_bytes"] = json!(u128::MAX.to_string());
    let error = serde_json::from_value::<DynamicBackingPoolSpec>(pool).unwrap_err();
    assert!(error.to_string().contains("overflows"));
}

#[test]
fn checkpoint_capacity_wire_rejects_wrong_pool_and_inconsistent_quantum() {
    let fixture = enabled(4096);
    let memory = fixture.plan.payload().memory();
    let wire = serde_json::to_value(memory).unwrap();
    let mut wrong_pool = wire.clone();
    let pool = wrong_pool["dynamic_pools"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|pool| pool["compatibility"]["usage"] != json!("state"))
        .unwrap();
    pool["checkpoint_growth_ceiling_bytes"] = json!(4096);
    // Local MemoryPlan validation already rejects a capacity grant to an
    // activation pool, before external wire can acquire any plan authority.
    assert!(serde_json::from_value::<MemoryPlan>(wrong_pool).is_err());
    let mut partial_quantum = wire;
    let pool = partial_quantum["dynamic_pools"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|pool| pool["checkpoint_growth_ceiling_bytes"] == json!(4096))
        .unwrap();
    pool["checkpoint_growth_ceiling_bytes"] = json!(4095);
    assert!(serde_json::from_value::<MemoryPlan>(partial_quantum).is_err());
}

#[test]
fn checkpoint_capacity_semantic_rebuild_rejects_rehashed_policy_or_pool_omission() {
    let fixture = enabled(4096);
    let original: Value = serde_json::from_slice(&fixture.plan.to_json().unwrap()).unwrap();
    let mut missing_pool = original.clone();
    let pool = missing_pool["payload"]["memory"]["dynamic_pools"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|pool| pool["checkpoint_growth_ceiling_bytes"] == json!(4096))
        .unwrap();
    pool.as_object_mut()
        .unwrap()
        .remove("checkpoint_growth_ceiling_bytes");
    pool["provisioning"]["maximum_resident_bytes"] = json!(
        pool["provisioning"]["maximum_resident_bytes"]
            .as_u64()
            .unwrap()
            - 4096
    );
    rehash_plan_json(&mut missing_pool);
    let missing_wire = serde_json::to_vec(&missing_pool).unwrap();
    // Locally self-consistent and correctly rehashed; only semantic rebuild
    // can establish the required authenticated pool set.
    assert!(ExecutionPlan::decode_untrusted(&missing_wire).is_ok());
    assert!(fixture.revalidate(&missing_wire).is_err());

    let mut changed_cap = original;
    changed_cap["payload"]["memory"]["checkpoint_capacity"]["maximum_retained_bytes"] = json!(8192);
    for pool in changed_cap["payload"]["memory"]["dynamic_pools"]
        .as_array_mut()
        .unwrap()
    {
        if pool["checkpoint_growth_ceiling_bytes"] == json!(4096) {
            pool["checkpoint_growth_ceiling_bytes"] = json!(8192);
            pool["provisioning"]["maximum_resident_bytes"] = json!(
                pool["provisioning"]["maximum_resident_bytes"]
                    .as_u64()
                    .unwrap()
                    + 4096
            );
        }
    }
    rehash_plan_json(&mut changed_cap);
    let changed_wire = serde_json::to_vec(&changed_cap).unwrap();
    assert!(ExecutionPlan::decode_untrusted(&changed_wire).is_ok());
    assert!(fixture.revalidate(&changed_wire).is_err());
}
