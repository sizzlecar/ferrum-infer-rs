mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

#[test]
fn determinism_eager_submission_restores_the_complete_typed_denominator() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ScratchOverwrite),
    );
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
        fixture.runtime.as_ref(),
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        &wave,
    )
    .unwrap();
    let valid_payloads = determinism_payloads(&layout, 0x31);
    let mut invalid_payloads = valid_payloads.clone();
    invalid_payloads[0][0].pop();
    assert!(layout.clone().bind(invalid_payloads).is_err());
    assert!(layout.clone().bind(vec![Vec::new()]).is_err());
    let restore = layout.bind(valid_payloads).unwrap();

    let handle = OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Kernel,
        &restore,
        0xa5,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let restore_fingerprint = restore.logical_fingerprint().unwrap();
    let initialization_identity = restore.initialization_identity().unwrap();
    assert_eq!(handle.restore_fingerprint(), restore_fingerprint);
    let attribution = handle
        .attribution()
        .expect("determinism eager path must preserve attribution");
    let submission_fingerprint = attribution.submission_fingerprint().to_owned();
    let expected_readbacks = handle.readback_plan().collection_request().request_count();
    let expected_witnesses = handle.readback_plan().witness_count();
    assert_eq!(
        expected_witnesses,
        fixture
            .plan
            .determinism_witness_plan()
            .unwrap()
            .witnesses()
            .len()
    );

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        let phases = &trace.submitted_command_phases[0];
        let first_compute = phases
            .iter()
            .position(|phase| *phase == DeviceCommandPhase::Compute)
            .expect("determinism submission has provider compute");
        assert!(phases[..first_compute]
            .iter()
            .all(|phase| *phase == DeviceCommandPhase::Initialization));
        assert!(phases[first_compute..]
            .iter()
            .all(|phase| *phase == DeviceCommandPhase::Compute));
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::EagerOnly]
        );
        assert_eq!(
            trace.submitted_attribution_requirements,
            vec![DeviceSubmissionAttributionRequirement::LogicalExecutionPath]
        );
        assert_eq!(trace.scratch_observations, vec![(0, 0xa5), (1, 0xa5)]);
        assert!(trace
            .uploaded_payloads
            .iter()
            .any(|payload| !payload.is_empty() && payload.iter().all(|byte| *byte == 0x31)));
    }
    let compute_rows = attribution
        .device()
        .commands()
        .iter()
        .filter(|command| command.command_phase() == DeviceCommandPhase::Compute)
        .collect::<Vec<_>>();
    assert_eq!(compute_rows.len(), providers.len());
    assert!(compute_rows
        .iter()
        .all(|command| command.execution_path() == DeviceExecutionPath::Eager));
    let evidence = handle.wait_into_evidence().unwrap();
    assert_eq!(evidence.restore_fingerprint(), restore_fingerprint);
    assert_eq!(evidence.initialization_identity(), &initialization_identity);
    assert_eq!(
        evidence.expected_execution_path(),
        DeviceExecutionPath::Eager
    );
    assert_eq!(evidence.physical_readbacks().len(), expected_readbacks);
    assert_eq!(evidence.witnesses().len(), expected_witnesses);
    assert!(evidence
        .physical_readbacks()
        .iter()
        .all(|readback| !readback.bytes().is_empty() && readback.raw_sha256().len() == 64));
    assert_eq!(
        evidence.attribution().submission_fingerprint(),
        submission_fingerprint
    );
    let artifact =
        serde_json::to_value(evidence.into_artifact_execution("eager-00").unwrap()).unwrap();
    assert_eq!(artifact["mode"], "eager");
    assert_eq!(
        artifact["initialization_identity"]["input_sha256"],
        initialization_identity.input_sha256()
    );
    assert!(artifact["attribution"]["physical_commands"]
        .as_array()
        .is_some_and(|commands| !commands.is_empty()));
    assert_eq!(
        artifact["attribution"]["replayed_segments"],
        serde_json::json!([])
    );
    assert!(artifact["witnesses"]
        .as_array()
        .is_some_and(|witnesses| !witnesses.is_empty()));

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_logical_restore_identity_ignores_fresh_physical_authority() {
    fn collect(fill_byte: u8) -> String {
        let (fixture, sequence, session, batch, step) = setup_with_fixture(
            fixture_with_determinism_provider_behavior(false, ProviderBehavior::ScratchOverwrite),
        );
        let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
        let active_bindings = wave_active_bindings(&wave, &session);
        let lane = Arc::clone(step.execution_lane());
        let providers = fixture
            .plan
            .payload()
            .nodes()
            .iter()
            .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
            .collect::<Vec<_>>();
        let batch_identity = OperationDispatch::bind_submission_wave_identity(
            &fixture.resolved,
            active_bindings.iter(),
            &wave,
            &lane,
        )
        .unwrap();
        let layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
            fixture.runtime.as_ref(),
            &providers,
            &fixture.resolved,
            &batch_identity,
            active_bindings.iter(),
            &wave,
        )
        .unwrap();
        let payloads = determinism_payloads(&layout, fill_byte);
        let fingerprint = layout
            .bind(payloads)
            .unwrap()
            .logical_fingerprint()
            .unwrap();

        drop(batch_identity);
        drop(providers);
        drop(active_bindings);
        drop(wave);
        drop(lane);
        teardown(fixture, sequence, session, batch, step);
        fingerprint
    }

    let first = collect(0x31);
    let second = collect(0x31);
    let changed_payload = collect(0x32);
    assert_eq!(first, second);
    assert_ne!(first, changed_payload);
}

#[test]
fn determinism_submission_rejects_unretained_outputs_before_provider_encode() {
    let (fixture, sequence, session, batch, step) = setup();
    assert!(fixture
        .plan
        .payload()
        .retained_completion_values()
        .is_empty());
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x31,
    );

    let error = match OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &restore,
        0xa5,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("unretained determinism output unexpectedly reached provider encode"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("is not retained as one exact terminal witness")
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_readback_collects_writable_state_with_exact_typed_usage() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(true, ProviderBehavior::Success),
    );
    let node_ids = [id("node.tail")];
    let witness_plan = fixture
        .plan
        .determinism_witness_plan_for_nodes(&node_ids)
        .unwrap();
    assert_eq!(witness_plan.node_ids(), &node_ids);
    assert!(witness_plan.initializations().iter().any(|initialization| {
        matches!(
            initialization.kind(),
            ExecutionDeterminismInitializationKind::ExternalInput { value_id }
                if value_id.as_str() == "value.intermediate"
        )
    }));
    assert!(witness_plan.witnesses().iter().any(|witness| {
        matches!(
            witness.kind(),
            ExecutionDeterminismWitnessKind::StateEffect { .. }
        ) && witness.location().usage() == BufferUsage::State
    }));
    let wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &node_ids,
    );
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x27,
    );

    let handle = OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Completion,
        &restore,
        0x5a,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(handle.readback_plan().targets().iter().any(|target| {
        target.witnesses().iter().any(|witness| {
            matches!(
                witness.kind(),
                ExecutionDeterminismWitnessKind::StateEffect { .. }
            )
        }) && target
            .batch()
            .requests()
            .iter()
            .all(|request| request.expected_usage() == BufferUsage::State)
    }));

    let readback = match handle.wait_with_determinism_readback().unwrap() {
        CompletionReadbackCollectionObservation::Terminal(receipt) => receipt,
        other => panic!("writable-state determinism readback did not terminate: {other:?}"),
    };
    assert!(readback.dispositions().iter().any(|disposition| {
        matches!(
            disposition,
            CompletionReadbackDisposition::Succeeded(output)
                if output.request().expected_usage() == BufferUsage::State
        )
    }));

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_chunk_binds_the_exact_paged_state_prefix_from_the_provider_view() {
    let fixture = fixture_with_token_scaled_paged_state();
    let full_input = [11, 12, 13, 14];
    let sequence = logical_resources_with_work(
        &fixture.plan_resources,
        "run.device-operation.paged-determinism",
        "request.device-operation.paged-determinism",
        TokenSpanWork::from_token_ids(&full_input, 0..full_input.len()).unwrap(),
    );
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step = begin_single_participant_step_on_lane_with_bucket_and_work(
        &batch,
        &lane,
        None,
        TokenSpanWork::from_token_ids(&full_input, 2..3).unwrap(),
    );
    let node_ids = [id("node.tail")];
    let wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &node_ids,
    );
    let active_bindings = wave_active_bindings(&wave, &session);
    let providers = wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
        fixture.runtime.as_ref(),
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        &wave,
    )
    .unwrap();

    let state_initialization_index = layout
        .witness_plan()
        .initializations()
        .iter()
        .position(|initialization| initialization.location().usage() == BufferUsage::State)
        .expect("token-scaled state must have one deterministic initialization");
    let state_initialization = &layout.witness_plan().initializations()[state_initialization_index];
    assert_eq!(state_initialization.location().declared_length_bytes(), 4);
    assert!(matches!(
        state_initialization.location().extent(),
        ExecutionDeterminismValueExtent::ActiveTokenPrefix {
            bytes_per_token: 4,
            maximum_tokens: 16,
            maximum_storage_bytes: 64,
        }
    ));
    let state_range =
        layout.participant_initialization_ranges(0).unwrap()[state_initialization_index];
    assert_eq!(state_range.logical_offset_bytes(), 0);
    assert_eq!(state_range.length_bytes(), 16);
    assert!(state_range.length_bytes() > 3 * 4);
    assert_eq!(state_range.length_bytes() % TEST_PAGED_BLOCK_BYTES, 0);

    let external_input_index = layout
        .witness_plan()
        .initializations()
        .iter()
        .position(|initialization| {
            matches!(
                initialization.kind(),
                ExecutionDeterminismInitializationKind::ExternalInput { .. }
            )
        })
        .expect("tail chunk must bind its external activation input");
    let external_input_range =
        layout.participant_initialization_ranges(0).unwrap()[external_input_index];
    assert_eq!(external_input_range.logical_offset_bytes(), 0);
    assert_eq!(external_input_range.length_bytes(), 4);

    let state_witness_index = layout
        .witness_plan()
        .witnesses()
        .iter()
        .position(|witness| {
            matches!(
                witness.kind(),
                ExecutionDeterminismWitnessKind::StateEffect { .. }
            )
        })
        .expect("tail chunk must retain its writable state witness");
    assert_eq!(
        layout
            .witness_participant_ranges(state_witness_index)
            .unwrap()[0],
        state_range
    );

    let participant_payloads = determinism_payloads(&layout, 0x27);
    let base_identity = layout
        .clone()
        .bind(participant_payloads.clone())
        .unwrap()
        .initialization_identity()
        .unwrap();
    let mut changed_input_payloads = participant_payloads.clone();
    changed_input_payloads[0][external_input_index].fill(0x28);
    let changed_input_identity = layout
        .clone()
        .bind(changed_input_payloads)
        .unwrap()
        .initialization_identity()
        .unwrap();
    assert_ne!(
        base_identity.input_sha256(),
        changed_input_identity.input_sha256()
    );
    assert_eq!(
        base_identity.initial_state_sha256(),
        changed_input_identity.initial_state_sha256()
    );
    assert_eq!(
        base_identity.rng_sha256(),
        changed_input_identity.rng_sha256()
    );

    let mut changed_state_payloads = participant_payloads.clone();
    changed_state_payloads[0][state_initialization_index].fill(0x29);
    let changed_state_identity = layout
        .clone()
        .bind(changed_state_payloads)
        .unwrap()
        .initialization_identity()
        .unwrap();
    assert_eq!(
        base_identity.input_sha256(),
        changed_state_identity.input_sha256()
    );
    assert_ne!(
        base_identity.initial_state_sha256(),
        changed_state_identity.initial_state_sha256()
    );
    assert_eq!(
        base_identity.rng_sha256(),
        changed_state_identity.rng_sha256()
    );

    let restore = layout.bind(participant_payloads).unwrap();
    let reaper = CompletionReaper::new();
    let handle = OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Completion,
        &restore,
        0x5a,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let state_readback = handle
        .readback_plan()
        .targets()
        .iter()
        .flat_map(|target| target.batch().requests())
        .find(|request| request.expected_usage() == BufferUsage::State)
        .expect("token-scaled state must have one exact terminal readback");
    assert_eq!(state_readback.logical_offset_bytes(), 0);
    assert_eq!(state_readback.output_layout().byte_len().unwrap(), 16);

    let readback = match handle.wait_with_determinism_readback().unwrap() {
        CompletionReadbackCollectionObservation::Terminal(receipt) => receipt,
        other => panic!("paged-state determinism readback did not terminate: {other:?}"),
    };
    assert!(readback.dispositions().iter().any(|disposition| {
        matches!(
            disposition,
            CompletionReadbackDisposition::Succeeded(output)
                if output.request().expected_usage() == BufferUsage::State
                    && output.bytes().len() == 16
        )
    }));
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert!(trace
            .uploaded_payloads
            .iter()
            .any(|payload| { payload.len() == 16 && payload.iter().all(|byte| *byte == 0x27) }));
        assert!(trace.readback_lengths.contains(&16));
    }

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_submission_rejects_state_overwritten_later_in_the_same_wave() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(true, ProviderBehavior::Success),
    );
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x27,
    );

    let error = match OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &restore,
        0x5a,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("overwritten state witness unexpectedly reached provider encode"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("is overwritten later in the same terminal readback scope")
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_submission_rejects_restore_for_a_different_wave_scope() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success),
    );
    let node_ids = [id("node.tail")];
    let lane = Arc::clone(step.execution_lane());
    let source_wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &node_ids,
    );
    let source_active_bindings = wave_active_bindings(&source_wave, &session);
    let source_providers = source_wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let source_batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        source_active_bindings.iter(),
        &source_wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &source_providers,
        &source_batch_identity,
        &source_active_bindings,
        &source_wave,
        0x27,
    );
    drop(source_providers);
    drop(source_active_bindings);
    drop(source_wave);
    step.try_retire_normal().unwrap();
    let target_step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );

    let wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &target_step,
        &node_ids,
    );
    let active_bindings = wave_active_bindings(&wave, &session);
    let reaper = CompletionReaper::new();
    let providers = wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();

    let error = match OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &restore,
        0x5a,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("foreign determinism restore scope unexpectedly reached provider encode"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("differs from its prepared wave or physical batch identity")
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, target_step);
}

#[test]
fn determinism_single_node_replay_preserves_the_immutable_plan_node_index() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let node_ids = [id("node.tail")];
    let wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &node_ids,
    );
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x42,
    );
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("single-node determinism probe has a reusable program identity");
    let program = DeviceReusableExecutionProgram::new(
        program_id,
        vec![DeviceReusableExecutionSegment::new(0, 0, 1, 1).unwrap()],
        vec![0],
    )
    .unwrap();

    let handle = OperationDispatch::encode_and_submit_determinism_replayed_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Replay,
        &restore,
        0xa5,
        &program,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert_eq!(handle.readback_plan().node_ids(), &node_ids);
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .program_binding_slots
            .keys()
            .copied()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([1])
    );
    assert!(matches!(
        handle.wait_with_determinism_readback().unwrap(),
        CompletionReadbackCollectionObservation::Terminal(_)
    ));

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_single_node_scopes_cannot_alias_static_reusable_program_identity() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    *fixture.provider_behavior.lock().unwrap() = ProviderBehavior::Success;
    let lane = Arc::clone(step.execution_lane());

    let main_node_ids = [id("node.main")];
    let main_wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &step,
        &main_node_ids,
    );
    let main_providers = main_wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let main_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &main_providers,
        &fixture.resolved,
        &main_wave,
        &lane,
    )
    .unwrap()
    .expect("static main-node probe has a reusable program identity");
    drop(main_providers);
    drop(main_wave);
    step.try_retire_normal().unwrap();

    let tail_node_ids = [id("node.tail")];
    let tail_step = begin_single_participant_step_on_lane_with_bucket(
        &batch,
        &lane,
        fixture.reusable_execution_bucket.as_ref(),
    );
    let tail_wave = prepare_determinism_wave_for_nodes(
        &fixture.plan_resources,
        &fixture.plan,
        &tail_step,
        &tail_node_ids,
    );
    let tail_providers = tail_wave
        .nodes()
        .iter()
        .map(|node| {
            fixture
                .registry
                .bind(&fixture.resolved, node.node_id())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let tail_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &tail_providers,
        &fixture.resolved,
        &tail_wave,
        &lane,
    )
    .unwrap()
    .expect("static tail-node probe has a reusable program identity");
    assert_ne!(main_program_id, tail_program_id);

    drop(tail_providers);
    drop(tail_wave);
    drop(lane);
    teardown(fixture, sequence, session, batch, tail_step);
}

#[test]
fn determinism_probe_wave_cannot_escape_through_the_product_dispatch_path() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success),
    );
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();

    let error = match OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("determinism probe wave unexpectedly escaped through product dispatch"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("purpose differs from its product or determinism dispatch path")
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_dispatch_rejects_a_product_wave_before_provider_encode() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x42,
    );

    let error = match OperationDispatch::encode_and_submit_determinism_eager_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &restore,
        0xa5,
        wave,
        &lane,
        &reaper,
    ) {
        Err(error) => error,
        Ok(_) => panic!("product wave unexpectedly entered determinism dispatch"),
    };
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains("purpose differs from its product or determinism dispatch path")
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_readback_rejects_a_foreign_plan_before_submission() {
    let foreign_fixture =
        fixture_with_determinism_provider_behavior(true, ProviderBehavior::Success);
    let foreign_witness_plan = foreign_fixture.plan.determinism_witness_plan().unwrap();
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success),
    );
    assert_ne!(foreign_witness_plan.plan_hash(), fixture.plan.plan_hash());
    let wave = prepare_determinism_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let restore = determinism_restore(
        &fixture,
        &providers,
        &batch_identity,
        &active_bindings,
        &wave,
        0x27,
    );

    let error = SubmissionWaveDeterminismReadbackPlan::from_restore(
        &foreign_fixture.resolved,
        &batch_identity,
        &wave,
        &restore,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("differs from the exact immutable plan initialization denominator"));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);

    drop(providers);
    drop(active_bindings);
    drop(lane);
    drop(wave);
    teardown(fixture, sequence, session, batch, step);

    drop(foreign_fixture.registry);
    drop(foreign_fixture.impostor_registry);
    drop(foreign_fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(foreign_fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
