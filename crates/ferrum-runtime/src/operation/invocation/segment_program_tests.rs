//! The compiled frame runs against real Core admissions and the ordinary
//! invocation oracle. TestRuntime records exact write-region facts and owners;
//! it does not execute device bytes or stand in for CUDA numerical tests.
use super::segment_hot_tests::{compile, oracle_with_owners};
use super::*;
use crate::operation::dispatch_contract::DisabledSubmissionWaveDispatchTimingSink as NoopSubmissionWaveDispatchTimingSink;
use crate::operation::retained_dependency::append_dependencies;
use crate::operation::segment_compile::CompiledSegmentBindingRecipe;
use crate::operation::segment_dispatch::encode_compiled_segment_wave_with_timing;
use vnext_device_operation_wave_contract::{setup_with_fixture, teardown, wave_active_bindings};

fn identity(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    active: &[TrustedActiveSequenceBinding],
) -> BatchOperationIdentity {
    let lane = wave.step_resources().execution_lane();
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, lane).unwrap();
    OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
        &topology,
        active.iter(),
        wave,
        lane,
        ferrum_types::InvocationPreparationStrategy::DecodeSegmentProgram,
    )
    .unwrap()
}

fn enable(fixture: &Fixture) {
    let mut trace = fixture.runtime_trace.lock().unwrap();
    trace.segment_metadata_enabled = true;
    trace.segment_encoder_enabled = true;
    trace.segment_capture_owners = true;
}

fn compile_with_dependencies(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
) -> CompiledSegmentBindingRecipe {
    let original = compile(fixture, wave);
    let declarations = original
        .nodes
        .iter()
        .map(|node| {
            (
                node.node_index,
                SegmentBindingDeclaration::new(
                    node.declaration.regions().to_vec(),
                    vec![SegmentBindingDependency {
                        input_ordinal: 1,
                        component_id: id("weight.component.left"),
                        source_offset_bytes: 0,
                        source_length_bytes: 8,
                        persistent_offset_bytes: 0,
                        persistent_length_bytes: 4,
                        alignment_bytes: 4,
                        validation_identity: "fixture.exact-weight-validation.v1".into(),
                    }],
                    Arc::new(()),
                )
                .unwrap(),
            )
        })
        .collect();
    let mut recipe = CompiledSegmentBindingRecipe::compile(
        &fixture.resolved,
        original.program_id.clone(),
        declarations,
    )
    .unwrap();
    recipe.compile_backend(fixture.runtime.as_ref()).unwrap();
    recipe
}

#[test]
fn segment_program_frame_matches_current_regions_owners_and_rejects_stale_frames() {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_token_scaled_paged_state_and_provider_behavior(
            ProviderBehavior::ProgramBinding,
        ));
    enable(&fixture);
    let lane = Arc::clone(step.execution_lane());
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let old_identity = identity(&fixture, &wave, &active);
    let mut recipe = compile(&fixture, &wave);
    recipe.compile_backend(fixture.runtime.as_ref()).unwrap();
    let cold_program = Arc::clone(recipe.backend_program.as_ref().unwrap());
    let (expected, owners) = oracle_with_owners(&fixture, &wave, &old_identity, &active, &recipe);
    let (encoded, facts) = encode_compiled_segment_wave_with_timing(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        &old_identity,
        &wave,
        active.iter(),
        &recipe,
        &NoopSubmissionWaveDispatchTimingSink,
    )
    .unwrap()
    .unwrap();
    assert!(facts.dynamic_resource_requests > 0 && facts.unique_physical_buffers > 0);
    assert_eq!(
        encoded.program_bindings,
        vec![TestCommand::CoalescedProgramBinding]
    );
    assert_eq!(
        encoded.program_binding_nodes,
        recipe
            .nodes
            .iter()
            .map(|n| n.node_index as u32)
            .collect::<Vec<_>>()
    );
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.segment_regions, expected);
        assert_eq!(trace.segment_retained_owners.len(), owners.len());
        assert!(trace
            .segment_retained_owners
            .iter()
            .zip(&owners)
            .all(|(a, b)| a.same_owners(b)));
        assert_eq!(trace.segment_program_compile_calls, 1);
        assert_eq!(trace.segment_program_encode_calls, 1);
        trace.segment_retained_owners.clear();
    }
    drop(owners);
    drop(encoded);
    drop(active);
    drop(wave);
    step.try_retire_normal().unwrap();
    let step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let current_identity = identity(&fixture, &wave, &active);
    assert!(encode_compiled_segment_wave_with_timing(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        &old_identity,
        &wave,
        active.iter(),
        &recipe,
        &NoopSubmissionWaveDispatchTimingSink,
    )
    .is_err());
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .segment_program_encode_calls,
        1
    );
    let (expected, owners) =
        oracle_with_owners(&fixture, &wave, &current_identity, &active, &recipe);
    let (encoded, _) = encode_compiled_segment_wave_with_timing(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        &current_identity,
        &wave,
        active.iter(),
        &recipe,
        &NoopSubmissionWaveDispatchTimingSink,
    )
    .unwrap()
    .unwrap();
    assert!(Arc::ptr_eq(
        &cold_program,
        recipe.backend_program.as_ref().unwrap()
    ));
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.segment_regions, expected);
        assert!(trace
            .segment_retained_owners
            .iter()
            .zip(&owners)
            .all(|(a, b)| a.same_owners(b)));
        assert_eq!(trace.segment_program_compile_calls, 1);
        assert_eq!(trace.segment_program_encode_calls, 2);
        trace.segment_retained_owners.clear();
    }
    drop(owners);
    drop(encoded);
    drop(current_identity);
    drop(old_identity);
    drop(active);
    drop(wave);
    drop(recipe);
    drop(cold_program);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn segment_program_frame_consumes_fresh_dependencies_once_and_checks_returned_nodes() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_retained_dependencies(16, DependencyMode::Valid),
    );
    enable(&fixture);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = identity(&fixture, &wave, &active);
    let recipe = compile_with_dependencies(&fixture, &wave);
    let run = || {
        encode_compiled_segment_wave_with_timing(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            &identity,
            &wave,
            active.iter(),
            &recipe,
            &NoopSubmissionWaveDispatchTimingSink,
        )
    };
    let (expected, owners) = oracle_with_owners(&fixture, &wave, &identity, &active, &recipe);
    let (mut first, _) = run().unwrap().unwrap();
    assert_eq!(
        fixture.runtime_trace.lock().unwrap().segment_regions,
        expected
    );
    let (mut second, _) = run().unwrap().unwrap();
    assert!(first
        .nodes
        .iter()
        .zip(&second.nodes)
        .all(|(a, b)| !Arc::ptr_eq(&a.scope, &b.scope)));
    for (old, current) in first.nodes.iter_mut().zip(&mut second.nodes) {
        assert_eq!(old.bindings.retained_plan_dependency_count(), 1);
        assert_eq!(
            old.bindings
                .retained_dependency_identities()
                .collect::<Vec<_>>(),
            current
                .bindings
                .retained_dependency_identities()
                .collect::<Vec<_>>()
        );
        assert!(append_dependencies(
            &current.scope,
            old.bindings.take_retained_plan_dependencies(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .is_err());
        append_dependencies(
            &current.scope,
            current.bindings.take_retained_plan_dependencies(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap();
    }
    fixture.runtime_trace.lock().unwrap().segment_program_fault =
        SegmentProgramFault::LeaveDependency;
    assert!(run().is_err());
    assert!(recipe.nodes.len() > 1);
    fixture.runtime_trace.lock().unwrap().segment_program_fault = SegmentProgramFault::ReverseNodes;
    assert!(run().is_err());
    fixture.runtime_trace.lock().unwrap().segment_program_fault = SegmentProgramFault::None;
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = true;
    let before = fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_program_encode_calls;
    assert!(run().is_err());
    assert_eq!(
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .segment_program_encode_calls,
        before
    );
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .tamper_buffer_descriptor = false;
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_retained_owners
        .clear();
    drop(owners);
    drop(first);
    drop(second);
    drop(recipe);
    drop(identity);
    drop(active);
    drop(wave);
    teardown(fixture, sequence, session, batch, step);
}

fn with_two_participants(
    run: impl FnOnce(
        &Fixture,
        &PreparedStepSubmissionWave<TestRuntime>,
        &[TrustedActiveSequenceBinding],
    ),
) {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("execution.compiled-frame").unwrap(),
        ReusableExecutionCapacity::new(2, 2, 1).unwrap(),
    )
    .unwrap();
    let fixture = fixture_with_token_scaled_paged_state_and_bucket(bucket);
    enable(&fixture);
    let sequences = (0..2)
        .map(|i| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.compiled.{i}"),
                &format!("request.compiled.{i}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = sequences
        .iter()
        .map(|s| s.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![one_token_span(), one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
    .with_reusable_execution_bucket(
        fixture
            .reusable_execution_bucket
            .as_ref()
            .unwrap()
            .bucket_id()
            .clone(),
    );
    let step = (0..4)
        .find_map(
            |_| match batch.try_begin_step(request.clone(), &lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => Some(step),
                StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                    deferred.maintain().unwrap();
                    None
                }
                _ => panic!("real compiled-frame Step admission failed"),
            },
        )
        .unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = batch
        .sessions()
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    run(&fixture, &wave, &active);
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .segment_retained_owners
        .clear();
    drop(active);
    drop(wave);
    step.try_retire_normal().unwrap();
    drop(batch);
    for session in &sessions {
        session.try_complete().unwrap();
    }
}

#[test]
fn segment_program_frame_rejects_reordered_participants_and_foreign_admitted_wave() {
    with_two_participants(|fixture, wave, active| {
        let identity = identity(fixture, wave, active);
        let mut recipe = compile(fixture, wave);
        recipe.compile_backend(fixture.runtime.as_ref()).unwrap();
        let (expected, owners) = oracle_with_owners(fixture, wave, &identity, active, &recipe);
        let encoded = encode_compiled_segment_wave_with_timing(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            &identity,
            wave,
            active.iter(),
            &recipe,
            &NoopSubmissionWaveDispatchTimingSink,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            fixture.runtime_trace.lock().unwrap().segment_regions,
            expected
        );
        let reversed = active.iter().rev().cloned().collect::<Vec<_>>();
        assert!(encode_compiled_segment_wave_with_timing(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            &identity,
            wave,
            reversed.iter(),
            &recipe,
            &NoopSubmissionWaveDispatchTimingSink,
        )
        .is_err());
        with_two_participants(|donor, foreign_wave, foreign_active| {
            assert_eq!(fixture.plan.plan_hash(), donor.plan.plan_hash());
            assert_ne!(wave.batch_step_id(), foreign_wave.batch_step_id());
            assert!(encode_compiled_segment_wave_with_timing(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                &identity,
                foreign_wave,
                foreign_active.iter(),
                &recipe,
                &NoopSubmissionWaveDispatchTimingSink,
            )
            .is_err());
        });
        assert_eq!(
            fixture
                .runtime_trace
                .lock()
                .unwrap()
                .segment_program_encode_calls,
            1
        );
        drop(owners);
        drop(encoded);
    });
}

#[test]
fn segment_program_mixed_eager_binding_uses_ordinary_uploads_before_frame_consumption() {
    use crate::completion::LaneSubmitOutcome;
    use vnext_device_operation_wave_contract::test_reusable_program;

    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_provider_behavior(
            false,
            ProviderBehavior::ProgramBindingFirstNodeEagerBoundary,
        ));
    enable(&fixture);
    fixture.runtime_trace.lock().unwrap().segment_entry =
        Some(DeviceReusableExecutionEntryIdentity::new());
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let mut wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let complete = compile(&fixture, &wave);
    let tail = &complete.nodes[1];
    let mut partial = CompiledSegmentBindingRecipe::compile(
        &fixture.resolved,
        complete.program_id.clone(),
        vec![(
            tail.node_index,
            SegmentBindingDeclaration::new(
                tail.declaration.regions().to_vec(),
                vec![],
                Arc::new(()),
            )
            .unwrap(),
        )],
    )
    .unwrap();
    partial.compile_backend(fixture.runtime.as_ref()).unwrap();
    let partial = Arc::new(partial);
    let program = test_reusable_program(
        complete.program_id.clone(),
        2,
        vec![0],
        vec![DeviceReusableExecutionSegment::new(0, 1, 2, 1).unwrap()],
        vec![1],
        vec![],
    );
    // Publish a real successful/retired candidate for the tail resident segment.
    // The eager first node owns a different current binding slot.
    let active = wave_active_bindings(&wave, &session);
    let initial_identity = identity(&fixture, &wave, &active);
    let (entry, epoch) = lane
        .try_segment_binding_entry(program.program_id())
        .unwrap()
        .unwrap();
    wave.begin_dispatch().unwrap();
    let mut reservation =
        CompletionReaper::reserve_wave(&reaper, wave, Arc::clone(&lane), initial_identity).unwrap();
    reservation
        .set_segment_binding_candidate(Arc::clone(&partial), entry, epoch)
        .unwrap();
    let mut commands = DeviceCommandBatch::with_capacity(8);
    reservation
        .encode_backing_initializations(fixture.runtime.as_ref(), &mut commands)
        .unwrap();
    commands.push_compute(TestCommand::Provider);
    let mut enqueue = lane.reserve_enqueue().unwrap();
    reservation.mark_submission_started();
    let fence = match enqueue.submit(commands) {
        LaneSubmitOutcome::Submitted(fence) => fence,
        _ => panic!("fixture candidate submission did not return a fence"),
    };
    drop(enqueue);
    let handle = match reservation.arm(fence, DeviceTimingMode::Off) {
        Ok(handle) => handle,
        Err((error, _)) => panic!("fixture candidate fence installation failed: {error}"),
    };
    let ticket = handle.take_segment_binding_publication().unwrap();
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("fixture candidate fence did not terminalize");
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    drop(handle);
    drop(active);
    let retirement = step.try_retire_normal().unwrap();
    assert!(ticket
        .bind_completion(&receipt)
        .unwrap()
        .publish(&retirement)
        .unwrap());

    let step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = identity(&fixture, &wave, &active);
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let cached = lane
        .lookup_segment_binding_recipe(program.program_id())
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&cached, &partial));
    let handle = OperationDispatch::encode_and_submit_reusable_wave_with_inputs(
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
    .unwrap();
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("mixed binding fallback did not terminalize");
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    assert!(handle.take_segment_binding_publication().is_none());
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.segment_program_compile_calls, 1);
        assert_eq!(trace.segment_program_encode_calls, 0);
        assert_eq!(trace.program_binding_input_counts.last(), Some(&2));
        assert_eq!(
            trace.submitted_commands.last().unwrap(),
            &vec![
                TestCommand::CoalescedProgramBinding,
                TestCommand::Provider,
                TestCommand::ReusableExecution,
            ]
        );
    }
    let retained = lane
        .lookup_segment_binding_recipe(program.program_id())
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&retained, &cached));
    drop(retained);
    drop(cached);
    drop(handle);
    drop(providers);
    drop(identity);
    drop(active);
    drop(partial);
    drop(complete);
    drop(program);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}
