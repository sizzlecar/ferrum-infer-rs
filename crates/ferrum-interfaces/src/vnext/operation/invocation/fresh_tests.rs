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
