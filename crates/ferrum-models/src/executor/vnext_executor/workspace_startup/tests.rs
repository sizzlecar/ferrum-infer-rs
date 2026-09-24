use super::*;

#[path = "../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod contract;

fn bucket(class: &str, width: u32, tokens: u64) -> ReusableExecutionBucketSpec {
    ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new(class).unwrap(),
        ReusableExecutionCapacity::new(width, tokens, 1).unwrap(),
    )
    .unwrap()
}

#[test]
fn workspace_startup_declared_buckets_cover_widths_without_expanding_programs() {
    let policy = resolve_reusable_execution_policy(
        32,
        2048,
        4096,
        &[PrefillChunk::new(0, 128, 128).unwrap()],
        &ReusableExecutionCaptureConfig::default(),
        false,
    )
    .unwrap();
    assert!(policy.policy.program_policy().is_none());
    let cases = declared_cases(policy.policy.buckets().iter().cloned(), 32, 2048, 4096).unwrap();
    let widths = cases
        .iter()
        .filter(|case| case.kind == VNextExecutionWaveKind::Decode)
        .map(|case| case.sequences)
        .collect::<Vec<_>>();
    assert_eq!(widths, [32, 16, 8, 4, 2, 1]);
    assert!(cases
        .iter()
        .any(|case| case.kind == VNextExecutionWaveKind::Prefill
            && case.sequences == 1
            && case.tokens_per_sequence == 128));
}

#[test]
fn workspace_startup_rejects_unknown_duplicate_and_over_capacity_cases() {
    assert!(declared_cases([], 32, 2048, 4096).is_err());
    assert!(declared_cases([bucket("unrecognized", 1, 1)], 32, 2048, 4096).is_err());
    let one = bucket(UNIFORM_QUERY_REUSABLE_CLASS, 1, 1);
    assert!(declared_cases([one.clone(), one], 32, 2048, 4096).is_err());
    assert!(declared_cases([bucket(UNIFORM_QUERY_REUSABLE_CLASS, 8, 8)], 4, 2048, 4096).is_err());
    assert!(declared_cases(
        [bucket(UNIFORM_QUERY_REUSABLE_CLASS, 64, 64)],
        64,
        2048,
        4096
    )
    .is_err());
    assert!(declared_cases([bucket(PACKED_TOKEN_REUSABLE_CLASS, 1, 128)], 32, 64, 4096).is_err());
    assert!(declared_cases([bucket(PACKED_TOKEN_REUSABLE_CLASS, 1, 128)], 32, 2048, 64).is_err());
}

fn prepare_wave(
    step: &Arc<StepResourceLease<contract::TestRuntime>>,
) -> PreparedStepSubmissionWave<contract::TestRuntime> {
    let work = step
        .shared_all_invocation_work_shape(&[contract::one_token_span()])
        .unwrap();
    for _ in 0..4 {
        match step
            .try_prepare_full_plan_submission_wave(
                Arc::clone(&work),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        {
            StepSubmissionWaveAdmissionDecision::Prepared(wave) => return wave,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                deferred.maintain().unwrap();
            }
            _ => panic!("actual workspace wave did not prepare"),
        }
    }
    panic!("bounded fixture backing maintenance did not converge");
}

#[test]
fn workspace_startup_real_prepared_wave_aborts_owners_and_reuses_slots_without_encode() {
    let fixture =
        contract::fixture_with_provider_behavior(false, contract::ProviderBehavior::ProgramBinding);
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let sequence = contract::logical_resources(
        &fixture.plan_resources,
        "run.workspace-a",
        "request.workspace-a",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = contract::begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    let slot = step.claimed_backing().lane_stable_slot_identity();
    let submits = fixture.runtime_trace.lock().unwrap().submit_calls;
    let encodes = fixture.provider_trace.lock().unwrap().encode_calls;
    let wave = prepare_wave(&step);
    assert!(wave.node_count() > 0);
    assert!(wave.prepared_participant_flight_count() > 0);
    finish_resource_only_wave(wave, step).unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    let sequence = contract::logical_resources(
        &fixture.plan_resources,
        "run.workspace-b",
        "request.workspace-b",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let allocations = fixture.runtime_trace.lock().unwrap().allocation_calls;
    let step = contract::begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    assert_eq!(slot, step.claimed_backing().lane_stable_slot_identity());
    let wave = prepare_wave(&step);
    assert_eq!(
        allocations,
        fixture.runtime_trace.lock().unwrap().allocation_calls
    );
    finish_resource_only_wave(wave, step).unwrap();
    assert_eq!(submits, fixture.runtime_trace.lock().unwrap().submit_calls);
    assert_eq!(encodes, fixture.provider_trace.lock().unwrap().encode_calls);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn workspace_startup_abort_failure_does_not_report_cleanup_or_release_live_owner() {
    let fixture =
        contract::fixture_with_provider_behavior(false, contract::ProviderBehavior::ProgramBinding);
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let sequence = contract::logical_resources(
        &fixture.plan_resources,
        "run.workspace-abort",
        "request.workspace-abort",
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = contract::begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    let wave = prepare_wave(&step);
    // A real outstanding Step owner makes unique finalization impossible.
    // The helper must not turn that error into a successful startup receipt.
    assert!(finish_resource_only_wave(wave, Arc::clone(&step)).is_err());
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    // The retained owner still carries finalization authority; its explicit
    // abort reconciles the resources before root close can succeed.
    step.try_abort().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
