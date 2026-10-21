use std::collections::BTreeSet;

use super::{
    device_capacity_account, issue_generation, Arc, DeviceCapacityOwner,
    DeviceCapacityPressureScope, DeviceCapacityReservation, DeviceDescriptor,
    DeviceOpaqueMemoryKind, VNextError,
};
use crate::{
    DeviceClass, DeviceId, DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView,
};

fn descriptor(capacity_bytes: u64) -> DeviceDescriptor {
    DeviceDescriptor {
        id: DeviceId::new(format!(
            "device.opaque-capacity-test-{}",
            issue_generation().unwrap()
        ))
        .unwrap(),
        class: DeviceClass::Reference,
        ordinal: 0,
        total_memory_bytes: capacity_bytes,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: BTreeSet::new(),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    }
}

#[test]
fn opaque_and_plan_reservations_share_the_process_capacity_account() {
    let device = descriptor(128);
    let owner = DeviceCapacityOwner::open(&device, 128).unwrap();
    let account = device_capacity_account(
        &device.id,
        &device.runtime_implementation_fingerprint,
        device.total_memory_bytes,
    )
    .unwrap();
    assert!(Arc::ptr_eq(&owner.budget.account, &account));
    let plan_budget = account.register_budget(128).unwrap();
    let bootstrap = owner
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 64)
        .unwrap();
    let plan = DeviceCapacityReservation::reserve(&plan_budget, 64).unwrap();

    assert!(matches!(
        owner.reserve(DeviceOpaqueMemoryKind::Stream, 1),
        Err(VNextError::DeviceCapacityUnavailable(ref pressure))
            if pressure.scope() == &DeviceCapacityPressureScope::ProcessWide
    ));
    assert!(matches!(
        DeviceCapacityReservation::reserve(&plan_budget, 1),
        Err(VNextError::DeviceCapacityUnavailable(_))
    ));
    let before_release = plan_budget.availability_snapshot().unwrap();
    drop(bootstrap);
    let after_release = plan_budget.availability_snapshot().unwrap();
    assert!(after_release.process_epoch() > before_release.process_epoch());
    assert_eq!(after_release.plan_epoch(), before_release.plan_epoch());
    let additional_plan = DeviceCapacityReservation::reserve(&plan_budget, 64).unwrap();
    assert_eq!(account.state.lock().unwrap().claimed_bytes, 128);
    drop((plan, additional_plan));
    assert_eq!(account.state.lock().unwrap().claimed_bytes, 0);
}

#[test]
fn clones_retain_one_claim_and_the_budget_until_the_last_native_owner() {
    let device = descriptor(64);
    let owner = DeviceCapacityOwner::open(&device, 64).unwrap();
    let same_owner = owner.clone();
    let other_owner = DeviceCapacityOwner::open(&device, 64).unwrap();
    let budget = Arc::downgrade(&owner.budget);
    let held = owner.reserve(DeviceOpaqueMemoryKind::Stream, 48).unwrap();
    let native_alias = held.clone();
    assert!(same_owner.owns(&native_alias));
    assert!(!other_owner.owns(&native_alias));
    assert_eq!(held.bytes(), 48);
    assert_eq!(held.kind(), DeviceOpaqueMemoryKind::Stream);
    assert_eq!(owner.budget.account.state.lock().unwrap().claimed_bytes, 48);

    drop((owner, same_owner, held));
    assert!(budget.upgrade().is_some());
    assert!(matches!(
        other_owner.reserve(DeviceOpaqueMemoryKind::Bootstrap, 17),
        Err(VNextError::DeviceCapacityUnavailable(_))
    ));
    drop(native_alias);
    assert!(budget.upgrade().is_none());
    let full = other_owner
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 64)
        .unwrap();
    assert_eq!(full.bytes(), 64);
}

#[test]
fn zero_reservations_keep_the_ceiling_without_claiming_a_byte() {
    let device = descriptor(128);
    let limited = DeviceCapacityOwner::open(&device, 96).unwrap();
    let other = DeviceCapacityOwner::open(&device, 128).unwrap();
    let zero = limited
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 0)
        .unwrap();
    let zero_alias = zero.clone();
    assert_eq!(zero.bytes(), 0);
    assert!(limited.owns(&zero));
    assert_eq!(
        limited.budget.account.state.lock().unwrap().claimed_bytes,
        0
    );
    drop((limited, zero));

    assert!(matches!(
        other.reserve(DeviceOpaqueMemoryKind::Stream, 97),
        Err(VNextError::DeviceCapacityUnavailable(_))
    ));
    let exact = other.reserve(DeviceOpaqueMemoryKind::Stream, 96).unwrap();
    drop(exact);
    drop(zero_alias);
    assert_eq!(
        other
            .reserve(DeviceOpaqueMemoryKind::Stream, 128)
            .unwrap()
            .bytes(),
        128
    );
}

#[test]
fn opening_an_owner_rejects_invalid_limits_and_conflicting_live_metadata() {
    let device = descriptor(128);
    assert!(DeviceCapacityOwner::open(&device, 0).is_err());
    assert!(DeviceCapacityOwner::open(&device, 129).is_err());
    let owner = DeviceCapacityOwner::open(&device, 128).unwrap();
    let held = owner
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 65)
        .unwrap();
    assert!(DeviceCapacityOwner::open(&device, 64).is_err());

    let mut conflicting = device.clone();
    conflicting.total_memory_bytes = 256;
    assert!(DeviceCapacityOwner::open(&conflicting, 128).is_err());
    conflicting = device.clone();
    conflicting.runtime_implementation_fingerprint = "b".repeat(64);
    assert!(DeviceCapacityOwner::open(&conflicting, 128).is_err());
    conflicting.runtime_implementation_fingerprint = "invalid".to_owned();
    assert!(DeviceCapacityOwner::open(&conflicting, 128).is_err());
    assert_eq!(held.bytes(), 65);
    assert_eq!(owner.budget.account.state.lock().unwrap().claimed_bytes, 65);
}

#[test]
fn overflowing_claims_leave_the_existing_reservation_intact() {
    let owner = DeviceCapacityOwner::open(&descriptor(u64::MAX), u64::MAX).unwrap();
    let held = owner
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, u64::MAX)
        .unwrap();
    assert!(owner.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
    assert_eq!(
        owner.budget.account.state.lock().unwrap().claimed_bytes,
        u64::MAX
    );
    drop(held);
    assert_eq!(owner.budget.account.state.lock().unwrap().claimed_bytes, 0);
}

#[test]
fn fail_closed_accounts_reject_zero_and_nonzero_reservations() {
    let owner = DeviceCapacityOwner::open(&descriptor(64), 64).unwrap();
    owner.budget.account.state.lock().unwrap().poisoned = true;
    for bytes in [0, 1] {
        assert!(matches!(
            owner.reserve(DeviceOpaqueMemoryKind::Stream, bytes),
            Err(VNextError::DynamicAdmissionContract { .. })
        ));
    }
}
