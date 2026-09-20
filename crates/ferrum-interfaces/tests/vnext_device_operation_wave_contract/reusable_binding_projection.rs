use super::*;

fn run_projection_case(resources: ReusableBindingResources, reusable: bool, drift: bool) {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_hybrid_state_and_provider_behavior(ProviderBehavior::ProgramBinding),
    );
    fixture
        .provider_trace
        .lock()
        .unwrap()
        .reusable_binding_resources = resources;
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        providers.providers(),
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .unwrap();
    let node_count = providers.len() as u32;
    let program = test_reusable_program(
        program_id,
        node_count,
        vec![],
        vec![DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap()],
        (0..node_count).collect(),
        vec![],
    );
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = drift;
    let result = if reusable {
        OperationDispatch::encode_and_submit_reusable_wave_with_inputs(
            providers.providers(),
            &fixture.resolved,
            &identity,
            active.iter(),
            DeviceTimingMode::Off,
            &[],
            &program,
            wave,
            &lane,
            &reaper,
        )
    } else {
        OperationDispatch::encode_and_submit_wave_with_inputs(
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
    };
    // Restore the injected runtime descriptor before cleanup, which also
    // validates the live allocations while releasing their authorities.
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = false;
    if drift {
        assert!(
            result.is_err(),
            "live backing descriptor drift must fail closed"
        );
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.encode_calls, 0);
        assert_eq!(trace.reusable_binding_encode_calls, 0);
        assert!(fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_commands
            .is_empty());
        drop(trace);
        drop(result);
    } else {
        let handle = result.unwrap();
        assert!(matches!(
            handle.wait().unwrap(),
            CompletionObservation::Terminal(_)
        ));
        let trace = fixture.provider_trace.lock().unwrap();
        if reusable {
            assert_eq!(trace.reusable_views.len(), providers.len());
            for (node_id, actual_views, actual_values) in &trace.reusable_views {
                let node = fixture
                    .plan
                    .payload()
                    .nodes()
                    .iter()
                    .find(|node| node.id() == node_id)
                    .unwrap();
                let all_values = node
                    .values()
                    .iter()
                    .map(|value| value.value_id().clone())
                    .collect::<BTreeSet<_>>();
                assert_eq!(
                    actual_values, &all_values,
                    "projection must retain full semantic metadata"
                );
                let mut expected_views = node
                    .values()
                    .iter()
                    .filter(|value| {
                        resources == ReusableBindingResources::All
                            || value.usage() == BufferUsage::State
                    })
                    .flat_map(|value| value.storage().components())
                    .map(|component| component.resource_id().clone())
                    .collect::<BTreeSet<_>>();
                expected_views.extend(node.binding_resource().cloned());
                assert_eq!(actual_views, &expected_views);
                assert!(actual_views.contains(&id("resource.state")));
                assert!(actual_views.contains(&id("resource.recurrent-state")));
            }
        } else {
            assert_eq!(trace.reusable_binding_encode_calls, 0);
            assert_eq!(trace.encode_calls as usize, providers.len());
            assert!(
                trace.component_resources.is_subset(&trace.view_resources),
                "eager still exposes captured weights"
            );
        }
        drop(trace);
        drop(handle);
    }
    drop(program);
    drop(providers);
    drop(active);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn reusable_binding_projection_preserves_live_paged_and_recurrent_state_and_all_metadata() {
    run_projection_case(
        ReusableBindingResources::RequestStateAndBinding,
        true,
        false,
    );
    run_projection_case(ReusableBindingResources::All, true, false);
}

#[test]
fn reusable_binding_opt_in_does_not_filter_eager_views() {
    run_projection_case(
        ReusableBindingResources::RequestStateAndBinding,
        false,
        false,
    );
}

#[test]
fn reusable_binding_projection_rejects_live_descriptor_drift_before_encoding() {
    run_projection_case(ReusableBindingResources::RequestStateAndBinding, true, true);
}
