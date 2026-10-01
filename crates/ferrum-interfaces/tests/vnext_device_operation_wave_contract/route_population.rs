//! Selection against real prepared core waves and lane-produced snapshots.
use super::*;
use ferrum_interfaces::execution_cost::{
    PreparedCostRouteClassV1 as C, PreparedCostRouteReasonV1 as U,
};

#[test]
fn route_population_no_program_layout_requires_original_observed_catalogue() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(fixture());
    {
        let mut trace = fixture.provider_trace.lock().unwrap();
        assert!(!trace.cost_route_eager_boundary);
        // A provider may explicitly project this complete eager wave. The
        // original OnDemand selector still excludes its absent program layout
        // from the warm/graph-disabled numerical population.
        trace.cost_route_eager_boundary = true;
    }
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    assert!(wave.claimed_backing().program_binding_layout().is_none());
    assert!(wave
        .claimed_backing()
        .program_binding_lane_slot_identity()
        .is_none());
    fixture.runtime_trace.lock().unwrap().cost_graph_state = Some(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 0, 0, 0).unwrap(),
    );
    let catalog = lane
        .reusable_execution_catalog()
        .unwrap()
        .into_index()
        .unwrap();
    let placeholder =
        IndexedExecutionLaneReusableCatalog::unobserved_empty(lane.reusable_execution_epoch());
    for (snapshot, observe, expected) in [
        (Some(&catalog), true, C::OutsideProgramLayoutAbsent),
        (Some(&catalog), false, C::Unknown),
        (None, true, C::Unknown),
        (Some(&placeholder), true, C::Unknown),
    ] {
        let selected = OperationDispatch::select_reusable_execution_for_cost(
            providers.providers(),
            &fixture.resolved,
            &wave,
            &lane,
            snapshot,
            observe,
        )
        .unwrap();
        assert_eq!(selected.route().class(), expected);
        assert!(selected.route().program_id().is_none());
        assert!(selected.into_parts().0.is_none());
    }
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    drop(wave);
    drop(providers);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn route_population_requires_actual_configured_catalog_and_never_executes_a_probe() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let catalog = lane
        .reusable_execution_catalog()
        .unwrap()
        .into_index()
        .unwrap();
    let unknown = OperationDispatch::select_reusable_execution_for_cost(
        providers.providers(),
        &fixture.resolved,
        &wave,
        &lane,
        Some(&catalog),
        true,
    )
    .unwrap();
    assert_eq!(
        unknown.route().class(),
        C::Unknown,
        "empty catalog does not prove graph capability"
    );
    fixture.runtime_trace.lock().unwrap().cost_graph_state = Some(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 0, 0, 0).unwrap(),
    );
    let outside = OperationDispatch::select_reusable_execution_for_cost(
        providers.providers(),
        &fixture.resolved,
        &wave,
        &lane,
        Some(&catalog),
        true,
    )
    .unwrap();
    assert_eq!(outside.route().class(), C::OutsideProgramAbsent);
    assert_eq!(outside.route().reason(), U::CatalogEmpty);
    assert!(outside.route().program_id().is_some());
    assert!(outside.into_parts().0.is_none());
    let placeholder =
        IndexedExecutionLaneReusableCatalog::unobserved_empty(lane.reusable_execution_epoch());
    for snapshot in [None, Some(&placeholder)] {
        let unknown = OperationDispatch::select_reusable_execution_for_cost(
            providers.providers(),
            &fixture.resolved,
            &wave,
            &lane,
            snapshot,
            true,
        )
        .unwrap();
        assert_eq!(unknown.route().class(), C::Unknown);
    }
    let omitted = OperationDispatch::select_reusable_execution_for_cost(
        providers.providers(),
        &fixture.resolved,
        &wave,
        &lane,
        Some(&catalog),
        false,
    )
    .unwrap();
    assert_eq!(omitted.route().class(), C::Unknown);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    drop(wave);
    drop(providers);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn route_population_reuses_the_program_from_original_capture_and_rejects_partial_readiness() {
    let (fixture, sequence, session, batch, mut step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let handle = OperationDispatch::encode_and_submit_wave_with_inputs(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    drop(handle);
    drop(active);
    drop(identity);
    let capture = fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submitted_reusable_captures[0]
        .clone()
        .unwrap();
    let count = capture.node_count();
    assert!(count > 1);
    let resident = DeviceReusableExecutionProgram::new(
        &capture,
        vec![DeviceReusableExecutionSegment::new(0, 0, count, count).unwrap()],
        (0..count).collect(),
        vec![],
    )
    .unwrap();
    step.try_retire_normal().unwrap();
    step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let programs = [
        (resident.clone(), C::Warm, true),
        (
            DeviceReusableExecutionProgram::new(
                &capture,
                vec![],
                vec![],
                (0..count)
                    .map(|node| {
                        DeviceReusableExecutionProgramGap::new(
                            node,
                            DeviceReusableExecutionProgramGapReason::WarmupRequired,
                        )
                    })
                    .collect(),
            )
            .unwrap(),
            C::OutsideProgramNonResident,
            false,
        ),
        (
            DeviceReusableExecutionProgram::new(
                &capture,
                vec![DeviceReusableExecutionSegment::new(0, 0, count - 1, count - 1).unwrap()],
                (0..count - 1).collect(),
                vec![DeviceReusableExecutionProgramGap::new(
                    count - 1,
                    DeviceReusableExecutionProgramGapReason::WarmupRequired,
                )],
            )
            .unwrap(),
            C::Unknown,
            true,
        ),
    ];
    for (program, class, selected) in programs {
        {
            let mut trace = fixture.runtime_trace.lock().unwrap();
            trace.cost_graph_state = Some(
                DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 1, 1, 0)
                    .unwrap(),
            );
            trace.reusable_catalog_programs = vec![program.clone()];
        }
        let catalog = lane
            .reusable_execution_catalog()
            .unwrap()
            .into_index()
            .unwrap();
        let actual = OperationDispatch::select_reusable_execution_for_cost(
            providers.providers(),
            &fixture.resolved,
            &wave,
            &lane,
            Some(&catalog),
            true,
        )
        .unwrap();
        assert_eq!(actual.route().class(), class);
        assert_eq!(actual.route().program_id(), Some(capture.program_id()));
        let (chosen, _) = actual.into_parts();
        assert_eq!(chosen, selected.then_some(&program));
    }
    assert_eq!(
        fixture.runtime_trace.lock().unwrap().submit_calls,
        1,
        "classification may not warm up or execute a replacement sample"
    );
    drop(wave);
    drop(providers);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn route_population_no_graph_requires_actual_backend_or_stream_declaration() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let lane = Arc::clone(step.execution_lane());
    assert_eq!(
        OperationDispatch::observe_non_reusable_cost_route(&lane).class(),
        C::Unknown
    );
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .cost_route_projection_enabled = true;
    assert_eq!(
        OperationDispatch::observe_non_reusable_cost_route(&lane).class(),
        C::GraphDisabled
    );
    fixture.runtime_trace.lock().unwrap().cost_graph_state = Some(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::Unconfigured, 0, 0, 0)
            .unwrap(),
    );
    assert_eq!(
        OperationDispatch::observe_non_reusable_cost_route(&lane).class(),
        C::GraphDisabled
    );
    fixture.runtime_trace.lock().unwrap().cost_graph_state = Some(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::OnDemand, 0, 0, 0).unwrap(),
    );
    assert_eq!(
        OperationDispatch::observe_non_reusable_cost_route(&lane).class(),
        C::Unknown
    );
    teardown(fixture, sequence, session, batch, step);
}
