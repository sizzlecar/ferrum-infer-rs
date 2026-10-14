use super::*;

#[path = "../../../../tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../../tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::{fixture_with_provider_behavior, id, ProviderBehavior};
use vnext_device_operation_wave_contract::{prepare_wave, setup, setup_with_fixture, teardown};

#[test]
fn topology_request_and_current_scope_lookup_remain_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ReusableExecutionTopologyRequest<'static>>();
    assert_send_sync::<CurrentReusableAddressScopes<'static>>();
}

#[test]
fn scope_memo_preserves_static_none_and_lazy_error_order() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let memory = fixture.plan.payload().memory();
    let scopes = CurrentReusableAddressScopes::new(
        memory,
        wave.claimed_backing(),
        wave.step_resources().backing_slices(),
    );
    let static_resource = memory.static_allocations()[0].resource_id();
    let dynamic_resource = memory
        .dynamic_descriptors()
        .iter()
        .map(|descriptor| descriptor.base_resource_id())
        .find(|resource| scopes.lookup_uncached(resource).unwrap().is_none())
        .expect("ordinary fixture has submission-scoped dynamic resources");
    let unknown = id("resource.unknown.scope-memo");

    // No up-front validation of later resources: the first requested error
    // remains the first error, and unsuccessful lookups never enter the memo.
    let error = [&unknown, static_resource]
        .into_iter()
        .map(|resource| scopes.lookup(resource))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_err();
    assert!(error.to_string().contains("unknown memory resource"));
    assert!(scopes.memo.lock().unwrap().is_empty());
    assert_eq!(
        scopes.lookup(static_resource).unwrap(),
        Some(DeviceReusableAddressScope::Plan)
    );
    assert_eq!(scopes.lookup(dynamic_resource).unwrap(), None);
    assert_eq!(
        scopes.memo.lock().unwrap().get(dynamic_resource),
        Some(&None)
    );
    assert_eq!(
        scopes.lookup(static_resource).unwrap(),
        scopes.lookup_uncached(static_resource).unwrap()
    );
    assert_eq!(scopes.lookup(dynamic_resource).unwrap(), None);
    assert!(scopes.lookup(&unknown).is_err());
    assert!(!scopes.memo.lock().unwrap().contains_key(&unknown));

    drop(scopes);
    drop(wave);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn scope_memo_is_fresh_per_wave_and_preserves_lane_conflict_order() {
    let (first, first_sequence, first_session, first_batch, first_step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let (second, second_sequence, second_session, second_batch, second_step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let first_wave = prepare_wave(&first.plan_resources, &first.plan, &first_step);
    let second_wave = prepare_wave(&second.plan_resources, &second.plan, &second_step);
    let first_scopes = CurrentReusableAddressScopes::new(
        first.plan.payload().memory(),
        first_wave.claimed_backing(),
        first_wave.step_resources().backing_slices(),
    );
    let second_scopes = CurrentReusableAddressScopes::new(
        second.plan.payload().memory(),
        second_wave.claimed_backing(),
        second_wave.step_resources().backing_slices(),
    );
    let resource = first_wave
        .claimed_backing()
        .backing_slices()
        .iter()
        .find(|authority| authority.reusable_address_scope().is_some())
        .expect("program-binding fixture owns a lane-stable resource")
        .resource_id();
    let first_scope = Some(DeviceReusableAddressScope::ExecutionLane(
        first_step.execution_lane().id(),
    ));
    let second_scope = Some(DeviceReusableAddressScope::ExecutionLane(
        second_step.execution_lane().id(),
    ));
    assert_ne!(first_scope, second_scope);
    assert_eq!(first_scopes.lookup(resource).unwrap(), first_scope);
    assert_eq!(second_scopes.lookup(resource).unwrap(), second_scope);
    assert_eq!(first_scopes.lookup(resource).unwrap(), first_scope);

    // Deliberately combine real authorities from different lanes. The lookup
    // must reject them on demand without poisoning unrelated static queries.
    let mixed = CurrentReusableAddressScopes::new(
        first.plan.payload().memory(),
        first_wave.claimed_backing(),
        second_wave.claimed_backing().backing_slices(),
    );
    let static_resource = first.plan.payload().memory().static_allocations()[0].resource_id();
    let unknown = id("resource.unknown.scope-memo");
    assert_eq!(
        mixed.lookup(static_resource).unwrap(),
        Some(DeviceReusableAddressScope::Plan)
    );
    for (requested, expected) in [
        ([resource, &unknown], "different execution lanes"),
        ([&unknown, resource], "unknown memory resource"),
    ] {
        let error = requested
            .into_iter()
            .map(|resource| mixed.lookup(resource))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_err();
        assert!(error.to_string().contains(expected));
    }
    assert!(!mixed.memo.lock().unwrap().contains_key(resource));
    assert!(!mixed.memo.lock().unwrap().contains_key(&unknown));

    drop(mixed);
    drop(first_scopes);
    drop(second_scopes);
    drop(first_wave);
    drop(second_wave);
    teardown(
        first,
        first_sequence,
        first_session,
        first_batch,
        first_step,
    );
    teardown(
        second,
        second_sequence,
        second_session,
        second_batch,
        second_step,
    );
}
