use super::*;

#[path = "../../../../../../tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../../../../tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;
use vnext_device_operation_contract::{
    fixture_with_device_id, fixture_with_provider_behavior, id, ProviderBehavior,
};
use vnext_device_operation_wave_contract::{prepare_wave, setup_with_fixture, teardown};

static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn fixture() -> vnext_device_operation_contract::Fixture {
    fixture_with_device_id(id(format!(
        "device.future-provider-selection.{}",
        NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )))
}

fn work(tokens: u64) -> [OperationCostWorkRow; 1] {
    [OperationCostWorkRow {
        offset: 0,
        count: std::num::NonZeroU64::new(tokens).unwrap(),
        full_input_tokens: std::num::NonZeroU64::new(tokens).unwrap(),
    }]
}

#[test]
fn future_provider_selection_default_matches_separate_route_and_topology() {
    let (f, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let mut trace = f.provider_trace.lock().unwrap();
        trace.cost_route_supported = true;
        trace.cost_route_eager_boundary = true;
    }
    {
        let providers = f.registry.bind_plan(&f.resolved).unwrap();
        for tokens in [1, 2, 7] {
            let work = work(tokens);
            let rows = ValidatedCostRows::new(&work).unwrap();
            let separate = providers
                .eager_cost_route_with_ranges(&f.resolved, &rows, None, &mut || Ok(()))
                .unwrap()
                .unwrap();
            let topology = providers
                .future_has_only_eager_boundaries(&f.resolved, &rows, None, &mut || Ok(()))
                .unwrap();
            let combined = providers
                .future_cost_route_with_ranges(
                    &f.resolved,
                    &rows,
                    None,
                    OperationCostTopologyRequirement::Required,
                    &mut || Ok(()),
                )
                .unwrap()
                .unwrap();
            assert_eq!(combined.route().physical_slots(), separate.physical_slots());
            for (a, b) in combined.route().nodes.iter().zip(&separate.nodes) {
                assert_eq!(a.route, b.route);
            }
            assert!(topology);
            assert_eq!(
                combined.only_eager_boundaries(&mut || Ok(())).unwrap(),
                topology
            );
            assert_eq!(combined.rows.immediate_tokens(), tokens);
        }
    }
    teardown(f, sequence, session, batch, step);
}

#[test]
fn future_provider_selection_requeries_changed_evidence_and_never_promotes_unknown() {
    let (f, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let mut trace = f.provider_trace.lock().unwrap();
        trace.cost_route_supported = true;
        trace.cost_route_eager_boundary = true;
    }
    {
        let providers = f.registry.bind_plan(&f.resolved).unwrap();
        let work = work(1);
        let rows = ValidatedCostRows::new(&work).unwrap();
        let select = |topology| {
            providers
                .future_cost_route_with_ranges(&f.resolved, &rows, None, topology, &mut || Ok(()))
        };
        let first = select(OperationCostTopologyRequirement::Required)
            .unwrap()
            .unwrap();
        assert!(first.only_eager_boundaries(&mut || Ok(())).unwrap());
        let queries_before = f.provider_trace.lock().unwrap().cost_route_queries.len();
        f.provider_trace.lock().unwrap().cost_route_eager_boundary = false;
        let replay = select(OperationCostTopologyRequirement::Required)
            .unwrap()
            .unwrap();
        assert!(!replay.only_eager_boundaries(&mut || Ok(())).unwrap());
        assert!(replay
            .topologies
            .as_ref()
            .unwrap()
            .iter()
            .all(Option::is_none));
        assert!(f.provider_trace.lock().unwrap().cost_route_queries.len() > queries_before);
        // A locally retained declaration is only for its own projection. The
        // next selection above did not borrow it or treat it as a live proof.
        assert!(first.only_eager_boundaries(&mut || Ok(())).unwrap());
        let eager = select(OperationCostTopologyRequirement::NotRequested)
            .unwrap()
            .unwrap();
        assert!(eager.topologies.is_none());
        assert!(eager.only_eager_boundaries(&mut || Ok(())).is_err());
        f.provider_trace.lock().unwrap().cost_route_supported = false;
        assert!(select(OperationCostTopologyRequirement::Required)
            .unwrap()
            .is_none());
    }
    teardown(f, sequence, session, batch, step);
}

#[test]
fn future_provider_selection_preserves_plan_binding_capacity_and_budget_boundaries() {
    let (f, sequence, session, batch, step) = setup_with_fixture(fixture());
    let (foreign, foreign_sequence, foreign_session, foreign_batch, foreign_step) =
        setup_with_fixture(fixture());
    f.provider_trace.lock().unwrap().cost_route_supported = true;
    {
        let providers = f.registry.bind_plan(&f.resolved).unwrap();
        let work = work(1);
        let rows = ValidatedCostRows::new(&work).unwrap();
        assert!(providers
            .future_cost_route_with_ranges(
                &foreign.resolved,
                &rows,
                None,
                OperationCostTopologyRequirement::Required,
                &mut || Ok(()),
            )
            .is_err());
        assert!(providers
            .future_cost_route_with_ranges(
                &f.resolved,
                &rows,
                None,
                OperationCostTopologyRequirement::Required,
                &mut || Err(invalid_operation("test budget expired")),
            )
            .is_err());
        assert!(f
            .provider_trace
            .lock()
            .unwrap()
            .cost_route_queries
            .is_empty());
        f.provider_trace
            .lock()
            .unwrap()
            .cost_route_extra_participants = 1;
        assert!(providers
            .future_cost_route_with_ranges(
                &f.resolved,
                &rows,
                None,
                OperationCostTopologyRequirement::Required,
                &mut || Ok(()),
            )
            .is_err());
    }
    teardown(
        foreign,
        foreign_sequence,
        foreign_session,
        foreign_batch,
        foreign_step,
    );
    teardown(f, sequence, session, batch, step);
}

#[test]
fn future_provider_selection_program_identity_matches_separate_query_on_real_arena() {
    let (f, sequence, session, batch, step) = setup_with_fixture(fixture_with_provider_behavior(
        false,
        ProviderBehavior::ProgramBinding,
    ));
    {
        let mut trace = f.provider_trace.lock().unwrap();
        trace.cost_route_supported = true;
        trace.cost_route_eager_boundary = true;
    }
    {
        let wave = prepare_wave(&f.plan_resources, &f.plan, &step);
        let backing = wave.claimed_backing();
        let layout = backing.program_binding_layout().unwrap();
        let slot = backing.program_binding_lane_slot_identity().unwrap();
        let providers = f.registry.bind_plan(&f.resolved).unwrap();
        let work = work(1);
        let rows = ValidatedCostRows::new(&work).unwrap();
        let combined = providers
            .future_cost_route_with_ranges(
                &f.resolved,
                &rows,
                None,
                OperationCostTopologyRequirement::Required,
                &mut || Ok(()),
            )
            .unwrap()
            .unwrap();
        let expected = providers
            .future_reusable_program_id(
                &f.resolved,
                &rows,
                None,
                layout,
                slot,
                slot.lane_id(),
                &mut || Ok(()),
            )
            .unwrap()
            .unwrap();
        let actual = combined
            .reusable_program_id(layout, slot, slot.lane_id(), &mut || Ok(()))
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
        assert!(combined
            .reusable_program_id(layout, slot, ExecutionLaneId::mint().unwrap(), &mut || Ok(
                ()
            ),)
            .is_err());
        assert!(combined
            .reusable_program_id(layout, slot, slot.lane_id(), &mut || Err(
                invalid_operation("expired before catalog identity")
            ),)
            .is_err());
        drop(wave);
    }
    teardown(f, sequence, session, batch, step);
}
