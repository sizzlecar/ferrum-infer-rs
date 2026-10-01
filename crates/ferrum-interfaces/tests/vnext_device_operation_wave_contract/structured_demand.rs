//! Real core admission/dispatch against the typed CPU runtime. This proves the
//! frozen observation-input contract, not native capture, numeric quality or speed.
use super::*;
use ferrum_interfaces::execution_cost::StructuredCostSampleDemand;

#[test]
fn numeric_demand_preserves_cold_templates_and_resumes_resident_inputs() {
    let (fixture, sequence, session, batch, mut step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.structured_cost_capture_enabled = true;
        trace.observation_template_budget =
            Some(DeviceObservationTemplateBudget::new(1 << 20).unwrap());
    }
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let count = providers.len() as u32;
    let timing = RecordingSubmissionTimingSink::default();

    // Even with no dynamic-sample consumer, the first real encode must retain
    // its provider work and capture declaration for future resident execution.
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let expected_program = OperationDispatch::reusable_execution_program_id_for_wave(
        providers.providers(),
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .unwrap();
    let (handle, attribution) =
        OperationDispatch::encode_and_submit_wave_with_cost_evidence_demand(
            providers.providers(),
            &fixture.resolved,
            &identity,
            active.iter(),
            DeviceTimingMode::Off,
            &[],
            SubmissionExecutionPolicy::default(),
            None,
            true,
            DeviceCostObservationDemand::NotRequired,
            StructuredCostSampleDemand::NotRequested,
            &timing,
            wave,
            &lane,
            &reaper,
        )
        .unwrap()
        .into_parts();
    assert!(
        attribution.is_some(),
        "low-cost logical receipt remains available"
    );
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
        .expect("cold actual capture declaration must survive omitted samples");
    assert_eq!(capture.program_id(), &expected_program);
    assert_eq!(
        fixture.provider_trace.lock().unwrap().encode_calls,
        u64::from(count)
    );
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .observation_template_requests,
        vec![true; count as usize],
        "future replay templates are retained during the actual cold capture"
    );
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submitted_observation_demands,
        vec![DeviceCostObservationDemand::NotRequired],
        "cold-template retention does not silently request a current sample"
    );
    let program = DeviceReusableExecutionProgram::new(
        &capture,
        vec![DeviceReusableExecutionSegment::new(0, 0, count, count).unwrap()],
        (0..count).collect(),
        vec![],
    )
    .unwrap();

    // The same resident program remains selected across requested, omitted,
    // renewed and old-wrapper-equivalent demand. Unknown evidence is not
    // upgraded to Known by the test provider.
    for (numeric, demand, input_expected) in [
        (
            DeviceCostObservationDemand::Required,
            StructuredCostSampleDemand::Requested,
            true,
        ),
        (
            DeviceCostObservationDemand::NotRequired,
            StructuredCostSampleDemand::Requested,
            false,
        ),
        (
            DeviceCostObservationDemand::Required,
            StructuredCostSampleDemand::NotRequested,
            false,
        ),
        (
            DeviceCostObservationDemand::Required,
            StructuredCostSampleDemand::Requested,
            true,
        ),
        (
            DeviceCostObservationDemand::Required,
            StructuredCostSampleDemand::RuntimePolicy,
            true,
        ),
    ] {
        step.try_retire_normal().unwrap();
        step = begin_single_participant_step_on_lane_with_bucket(
            &batch,
            &lane,
            fixture.reusable_execution_bucket.as_ref(),
        );
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .replay_cost_queries
            .clear();
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .binding_observation_template_requests
            .clear();
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .reusable_observation_inputs
            .clear();
        timing.stages.lock().unwrap().clear();
        let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
        let tokens = wave.claimed_backing().work_shape().immediate_tokens();
        let active = wave_active_bindings(&wave, &session);
        let identity = OperationDispatch::bind_submission_wave_identity(
            &fixture.resolved,
            active.iter(),
            &wave,
            &lane,
        )
        .unwrap();
        let (handle, attribution) =
            OperationDispatch::encode_and_submit_wave_with_cost_evidence_demand(
                providers.providers(),
                &fixture.resolved,
                &identity,
                active.iter(),
                DeviceTimingMode::Off,
                &[],
                SubmissionExecutionPolicy::default(),
                Some(&program),
                true,
                numeric,
                demand,
                &timing,
                wave,
                &lane,
                &reaper,
            )
            .unwrap()
            .into_parts();
        let attribution = attribution.expect("logical attribution cannot depend on sampling");
        assert_eq!(attribution.device().replayed_segments().len(), 1);
        assert_eq!(
            attribution.device().replayed_segments()[0]
                .logical_commands()
                .len(),
            count as usize
        );
        assert!(matches!(
            handle.wait().unwrap(),
            CompletionObservation::Terminal(_)
        ));
        drop(handle);
        let provider = fixture.provider_trace.lock().unwrap();
        assert!(
            provider.replay_cost_queries.is_empty(),
            "observation demand never calls the old per-node projection hook"
        );
        assert_eq!(
            provider.binding_observation_template_requests,
            vec![numeric.is_required(); count as usize]
        );
        drop(provider);
        let runtime = fixture.runtime_trace.lock().unwrap();
        assert_eq!(runtime.submitted_observation_demands.last(), Some(&numeric));
        let [input] = runtime.reusable_observation_inputs.as_slice() else {
            panic!("one actual replay segment must carry one frozen input slot")
        };
        assert_eq!(input.is_some(), input_expected);
        if let Some(input) = input {
            assert_eq!(input.tokens(), tokens);
            assert_eq!(input.participant_ranges().last().unwrap().end, tokens);
            assert_eq!(
                input.source_ranges().len(),
                input.participant_ranges().len()
            );
        }
        drop(runtime);
        assert!(
            timing
                .stages
                .lock()
                .unwrap()
                .iter()
                .all(|stage| *stage != SubmissionWaveDispatchStage::ReplayCostProviderProjection),
            "frontend may freeze inputs but cannot resolve provider statistics"
        );
        assert_eq!(
            fixture.provider_trace.lock().unwrap().encode_calls,
            u64::from(count),
            "resident replay cannot fall back to eager provider encoding"
        );
    }
    // Omitting optional projection does not turn a stale live buffer into an
    // execution permission. Real binding construction still fails pre-submit.
    step.try_retire_normal().unwrap();
    step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let submitted = fixture.runtime_trace.lock().unwrap().submit_calls;
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = true;
    let stale = OperationDispatch::encode_and_submit_wave_with_cost_evidence_demand(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::default(),
        Some(&program),
        true,
        DeviceCostObservationDemand::NotRequired,
        StructuredCostSampleDemand::NotRequested,
        &timing,
        wave,
        &lane,
        &reaper,
    );
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = false;
    assert!(stale.is_err());
    assert_eq!(
        fixture.runtime_trace.lock().unwrap().submit_calls,
        submitted
    );
    drop(stale);
    drop(active);
    drop(identity);
    drop(program);

    drop(providers);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn numeric_demand_omits_eager_templates_without_omitting_execution_or_receipts() {
    for policy in [
        SubmissionExecutionPolicy::eager(),
        SubmissionExecutionPolicy::adaptive(),
    ] {
        let (fixture, sequence, session, batch, mut step) = setup_with_fixture(
            fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
        );
        {
            let mut trace = fixture.runtime_trace.lock().unwrap();
            trace.structured_cost_capture_enabled = true;
            // This is the real runtime capability used by graphless backends.
            // An adaptive execution policy alone cannot require cold graph metadata.
            trace.cost_route_projection_enabled = true;
            trace.observation_template_budget =
                Some(DeviceObservationTemplateBudget::new(1 << 20).unwrap());
        }
        let lane = Arc::clone(step.execution_lane());
        let reaper = CompletionReaper::new();
        let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
        let count = providers.len();
        for (index, numeric) in [
            DeviceCostObservationDemand::Required,
            DeviceCostObservationDemand::NotRequired,
            DeviceCostObservationDemand::Required,
        ]
        .into_iter()
        .enumerate()
        {
            if index != 0 {
                step.try_retire_normal().unwrap();
                step = begin_single_participant_step_on_lane_with_bucket(
                    &batch,
                    &lane,
                    fixture.reusable_execution_bucket.as_ref(),
                );
            }
            fixture
                .provider_trace
                .lock()
                .unwrap()
                .observation_template_requests
                .clear();
            let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
            let active = wave_active_bindings(&wave, &session);
            let identity = OperationDispatch::bind_submission_wave_identity(
                &fixture.resolved,
                active.iter(),
                &wave,
                &lane,
            )
            .unwrap();
            let (handle, attribution) =
                OperationDispatch::encode_and_submit_wave_with_cost_evidence_demand(
                    providers.providers(),
                    &fixture.resolved,
                    &identity,
                    active.iter(),
                    DeviceTimingMode::Off,
                    &[],
                    policy,
                    None,
                    true,
                    numeric,
                    StructuredCostSampleDemand::Requested,
                    &RecordingSubmissionTimingSink::default(),
                    wave,
                    &lane,
                    &reaper,
                )
                .unwrap()
                .into_parts();
            let attribution =
                attribution.expect("exact raw receipt survives an omitted numeric sample");
            assert!(attribution.device().replayed_segments().is_empty());
            assert!(matches!(
                handle.wait().unwrap(),
                CompletionObservation::Terminal(_)
            ));
            drop(handle);
            assert_eq!(
                fixture
                    .provider_trace
                    .lock()
                    .unwrap()
                    .observation_template_requests,
                vec![numeric.is_required(); count]
            );
            let trace = fixture.runtime_trace.lock().unwrap();
            assert_eq!(trace.submit_calls, (index + 1) as u64);
            assert_eq!(trace.submitted_observation_demands.last(), Some(&numeric));
            assert_eq!(
                trace.submitted_reusable_captures.last().unwrap().is_some(),
                policy.compute_path() != DeviceComputePathRequirement::EagerOnly,
                "observation demand does not alter the original execution capture declaration"
            );
        }
        drop(providers);
        drop(reaper);
        drop(lane);
        teardown(fixture, sequence, session, batch, step);
    }
}
