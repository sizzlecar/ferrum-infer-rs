use super::*;
#[path = "../../../../tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../../tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;
use vnext_device_operation_contract::{fixture_with_device_id, id};
use vnext_device_operation_wave_contract::{
    prepare_wave, setup_with_fixture, teardown, wave_active_bindings,
};

static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn fixture() -> vnext_device_operation_contract::Fixture {
    fixture_with_device_id(id(format!(
        "device.fresh-invocation.{}",
        NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )))
}

#[test]
fn future_cost_request_uses_bound_resource_indices_without_inventing_residency() {
    use crate::vnext::operation::cost_route::{
        OperationCostRouteRequest, OperationCostWorkRow, ValidatedCostRows,
    };
    use crate::vnext::{
        DeviceReusableAddressScope, DynamicStorageView, ReusableExecutionTopologyView,
    };
    let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let work = [OperationCostWorkRow {
            offset: 0,
            count: std::num::NonZeroU64::new(1).unwrap(),
            full_input_tokens: std::num::NonZeroU64::new(1).unwrap(),
        }];
        let rows = ValidatedCostRows::new(&work).unwrap();
        let memory = fixture.plan.payload().memory();
        let mut checked_static = false;
        let mut checked_dynamic = false;
        let mut checked_outside_node = false;
        for node in fixture.plan.payload().nodes() {
            let provider = fixture.registry.bind(&fixture.resolved, node.id()).unwrap();
            let prepared = provider.dispatch();
            let request = OperationCostRouteRequest::new(node, memory, &rows, prepared, None);
            // Query every plan resource, including those belonging only to a
            // different node. Such callers retain the original lookup behavior.
            for allocation in memory.static_allocations() {
                let resource = allocation.resource_id();
                checked_static = true;
                checked_outside_node |= prepared.resource_source(resource).is_none();
                assert_eq!(
                    request.resource_reusable_address_scope(resource).unwrap(),
                    Some(DeviceReusableAddressScope::Plan)
                );
            }
            for descriptor in memory.dynamic_descriptors() {
                let resource = descriptor.base_resource_id();
                checked_dynamic = true;
                checked_outside_node |= prepared.resource_source(resource).is_none();
                assert_eq!(
                    request.resource_reusable_address_scope(resource).unwrap(),
                    None,
                    "a prepared descriptor is not proof of current lane residency"
                );
            }
            assert!(request
                .resource_reusable_address_scope(&id("resource.absent"))
                .is_err());
            for binding in node.values() {
                let [component] = binding.storage().components() else {
                    continue;
                };
                let expected = crate::vnext::operation::resolved_value::resource_uses_packed_batch_coordinates(memory, component.resource_id()).unwrap();
                assert_eq!(
                    request
                        .binding_uses_packed_batch_coordinates(binding.role(), binding.ordinal())
                        .unwrap(),
                    expected
                );
                assert_eq!(
                    ReusableExecutionTopologyView::binding_uses_packed_batch_coordinates(
                        &request,
                        binding.role(),
                        binding.ordinal()
                    )
                    .unwrap(),
                    expected
                );
                let alignment = memory
                    .dynamic_descriptor(component.resource_id())
                    .filter(|d| d.storage().profile().view() == DynamicStorageView::Contiguous)
                    .and_then(|d| std::num::NonZeroU64::new(d.alignment_bytes()));
                assert_eq!(
                    request
                        .binding_contiguous_base_alignment(binding.role(), binding.ordinal())
                        .unwrap(),
                    alignment
                );
            }
        }
        assert!(
            checked_static && checked_dynamic && checked_outside_node,
            "the fixture must exercise static, dynamic and out-of-node resource classifications"
        );
    }
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn replay_cost_identity_cache_probe_is_passive_and_preserves_absent_node_errors() {
    use crate::vnext::OperationDispatch;

    let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active = wave_active_bindings(&wave, &session);
        let lane = step.execution_lane();
        let topology =
            OperationDispatch::compile_submission_wave_identity(&fixture.resolved, lane).unwrap();
        let identity = OperationDispatch::bind_compiled_submission_wave_identity(
            &topology,
            active.iter(),
            &wave,
            lane,
        )
        .unwrap();
        let clone = identity.clone();
        let count = identity.node_count();
        assert!(count > 1);
        for index in 0..count {
            assert!(!identity.node_is_materialized(index));
        }
        assert_eq!(identity.materialization_snapshot().materialized_nodes(), 0);
        assert!(!identity.node_is_materialized(count));
        let before = identity.materialize_node(count).unwrap_err().to_string();
        assert!(!identity.node_is_materialized(count));
        assert_eq!(
            identity.materialize_node(count).unwrap_err().to_string(),
            before
        );
        assert_eq!(identity.materialization_snapshot().materialized_nodes(), 0);

        identity.materialize_node(0).unwrap();
        assert!(identity.node_is_materialized(0));
        assert!(clone.node_is_materialized(0));
        assert!(!identity.node_is_materialized(1));
        assert_eq!(identity.materialization_snapshot().materialized_nodes(), 1);
        assert!(!identity
            .materialization_snapshot()
            .full_participant_projection());

        // A second current wave identity shares the compiled topology but has
        // independent materialization slots. The probe cannot cache by plan.
        let separate = OperationDispatch::bind_compiled_submission_wave_identity(
            &topology,
            active.iter(),
            &wave,
            lane,
        )
        .unwrap();
        assert!(!separate.node_is_materialized(0));
        identity.nodes();
        assert!((0..count).all(|index| identity.node_is_materialized(index)));
        assert!(!identity.node_is_materialized(count));
        assert!(!separate.node_is_materialized(0));
        assert!(!identity
            .materialization_snapshot()
            .full_participant_projection());
        assert_eq!(
            serde_json::to_vec(&identity).unwrap(),
            serde_json::to_vec(&separate).unwrap()
        );
    }
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn fresh_invocation_matches_full_population_and_rejects_missing_or_wrong_node_identity() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active = wave_active_bindings(&wave, &session);
        let identity = crate::vnext::OperationDispatch::bind_submission_wave_identity(
            &fixture.resolved,
            active.iter(),
            &wave,
            step.execution_lane(),
        )
        .unwrap();
        for (index, node) in fixture.plan.payload().nodes().iter().enumerate() {
            let provider = fixture.registry.bind(&fixture.resolved, node.id()).unwrap();
            let node_identity = identity.materialize_node(index).unwrap();
            BatchedOperationInvocation::from_wave_node(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                node_identity,
                &wave,
                index,
                active.iter(),
            )
            .unwrap();
            BatchedOperationInvocation::validate_replay_cost_resources(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                node_identity,
                &wave,
                index,
                active.iter(),
            )
            .unwrap();
            let empty = &active[..0];
            assert!(BatchedOperationInvocation::from_wave_node(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                node_identity,
                &wave,
                index,
                empty.iter(),
            )
            .is_err());
            assert!(BatchedOperationInvocation::validate_replay_cost_resources(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                node_identity,
                &wave,
                index,
                empty.iter(),
            )
            .is_err());
            let wrong_index = (index + 1) % wave.nodes().len();
            assert_ne!(
                wrong_index, index,
                "fixture must exercise distinct actual nodes"
            );
            let wrong_identity = identity.materialize_node(wrong_index).unwrap();
            assert!(BatchedOperationInvocation::from_wave_node(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                wrong_identity,
                &wave,
                index,
                active.iter(),
            )
            .is_err());
            assert!(BatchedOperationInvocation::validate_replay_cost_resources(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                wrong_identity,
                &wave,
                index,
                active.iter(),
            )
            .is_err());
        }
    }
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn fresh_invocation_rechecks_current_static_and_dynamic_runtime_descriptors() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active = wave_active_bindings(&wave, &session);
        let identity = crate::vnext::OperationDispatch::bind_submission_wave_identity(
            &fixture.resolved,
            active.iter(),
            &wave,
            step.execution_lane(),
        )
        .unwrap();
        for (index, node) in fixture.plan.payload().nodes().iter().enumerate() {
            let provider = fixture.registry.bind(&fixture.resolved, node.id()).unwrap();
            let node_identity = identity.materialize_node(index).unwrap();
            for tamper in [false, true, false] {
                fixture
                    .runtime_trace
                    .lock()
                    .unwrap()
                    .tamper_buffer_descriptor = tamper;
                let old = BatchedOperationInvocation::from_wave_node(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    &identity,
                    node_identity,
                    &wave,
                    index,
                    active.iter(),
                )
                .is_ok();
                let fresh = BatchedOperationInvocation::validate_replay_cost_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    &identity,
                    node_identity,
                    &wave,
                    index,
                    active.iter(),
                )
                .is_ok();
                assert_eq!(fresh, old);
                assert_eq!(fresh, !tamper);
            }
        }
    }
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn borrowed_descriptor_invocation_parity_preserves_fresh_tamper_checks() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active = wave_active_bindings(&wave, &session);
        let identity = crate::vnext::OperationDispatch::bind_submission_wave_identity(
            &fixture.resolved,
            active.iter(),
            &wave,
            step.execution_lane(),
        )
        .unwrap();
        for (index, node) in fixture.plan.payload().nodes().iter().enumerate() {
            let provider = fixture.registry.bind(&fixture.resolved, node.id()).unwrap();
            let node_identity = identity.materialize_node(index).unwrap();
            let mut expected_work = None;
            for borrowed in [false, true] {
                // Reuse the same wave and identities across descriptor changes:
                // neither owned fallback nor a borrowed read may cache success.
                for tamper in [false, true, false] {
                    let (owned_before, borrowed_before) = {
                        let mut trace = fixture.runtime_trace.lock().unwrap();
                        trace.borrow_buffer_descriptors = borrowed;
                        trace.tamper_buffer_descriptor = tamper;
                        (
                            trace.owned_buffer_descriptor_reads,
                            trace.borrowed_buffer_descriptor_reads,
                        )
                    };
                    let full = BatchedOperationInvocation::from_wave_node(
                        fixture.runtime.as_ref(),
                        &fixture.resolved,
                        provider.dispatch(),
                        &identity,
                        node_identity,
                        &wave,
                        index,
                        active.iter(),
                    )
                    .map(|invocation| invocation.replay_cost_work().unwrap());
                    let numeric = BatchedOperationInvocation::validate_replay_cost_resources(
                        fixture.runtime.as_ref(),
                        &fixture.resolved,
                        provider.dispatch(),
                        &identity,
                        node_identity,
                        &wave,
                        index,
                        active.iter(),
                    );
                    assert_eq!(full.is_ok(), !tamper);
                    assert_eq!(numeric.is_ok(), !tamper);
                    if let (Ok(full), Ok(numeric)) = (full, numeric) {
                        assert_eq!(full, numeric);
                        if let Some(expected) = expected_work.as_ref() {
                            assert_eq!(&full, expected);
                        } else {
                            expected_work = Some(full);
                        }
                    }
                    let trace = fixture.runtime_trace.lock().unwrap();
                    if borrowed {
                        assert!(trace.borrowed_buffer_descriptor_reads > borrowed_before);
                        assert_eq!(trace.owned_buffer_descriptor_reads, owned_before);
                    } else {
                        assert!(trace.owned_buffer_descriptor_reads > owned_before);
                        assert_eq!(trace.borrowed_buffer_descriptor_reads, borrowed_before);
                    }
                }
            }
        }
    }
    teardown(fixture, sequence, session, batch, step);
}
