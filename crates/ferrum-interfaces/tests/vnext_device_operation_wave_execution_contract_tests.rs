mod vnext_device_operation_contract;
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

#[test]
fn reusable_topology_observes_packed_step_token_coordinates() {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_token_scaled_paged_state_and_provider_behavior(
            ProviderBehavior::ProgramBinding,
        ));
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();

    OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("packed token wave must expose reusable identity");

    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_topology_packed_input_coordinates,
        [true, true]
    );

    drop(providers);
    drop(wave);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn kernel_profile_binds_native_work_to_exact_plan_nodes() {
    let (fixture, sequence, session, batch, step) = setup();
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
    let timing = RecordingSubmissionTimingSink::default();

    let profiled = OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Kernel,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        &timing,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let (handle, attribution) = profiled.into_parts();
    let attribution = attribution.expect("kernel profile must return bound native work");
    assert_eq!(
        attribution.submission_fingerprint(),
        handle.receipt().fingerprint()
    );
    let node_rows = attribution
        .device()
        .commands()
        .iter()
        .filter(|command| command.node_index().is_some())
        .collect::<Vec<_>>();
    assert_eq!(node_rows.len(), batch_identity.nodes().len());
    for (expected_index, command) in node_rows.into_iter().enumerate() {
        let identity_node = &batch_identity.nodes()[expected_index];
        let plan_node = &fixture.plan.payload().nodes()[expected_index];
        assert_eq!(
            identity_node.provider_implementation_fingerprint(),
            plan_node.provider_implementation_fingerprint()
        );
        assert_eq!(command.node_index(), Some(expected_index as u32));
        assert_eq!(command.native_op_id(), "test_provider");
        assert_eq!(command.execution_path(), DeviceExecutionPath::Eager);
        assert_eq!(command.batching_form(), DeviceBatchingForm::Scalar);
        assert_eq!(command.participant_count(), 1);
        assert_eq!(command.token_count(), 1);
        assert_eq!(command.compute_dispatch_count(), 1);
        assert_eq!(command.transfer_command_count(), 0);
    }
    let completion = match handle.wait().unwrap() {
        CompletionObservation::Terminal(completion) => completion,
        other => panic!("kernel-profiled wave did not terminate: {other:?}"),
    };
    let attribution = attribution
        .bind_terminal_timing(completion.submission_timing().clone())
        .unwrap();
    let DeviceTimingMeasurement::Measured(timing) = attribution.terminal_timing() else {
        panic!("kernel-profiled wave must bind terminal command timing")
    };
    assert_eq!(
        timing.command_count() as usize,
        attribution.device().commands().len()
    );
    for (timing, command) in timing.spans().iter().zip(attribution.device().commands()) {
        assert_eq!(timing.start_command_index(), command.command_index());
        assert_eq!(timing.end_command_index(), command.command_index() + 1);
        assert_eq!(
            timing.kind(),
            ferrum_interfaces::vnext::DeviceExecutionSpanKind::EagerCommand
        );
        assert_eq!(timing.measurement().intervals().unwrap().len(), 1);
        assert_eq!(timing.measurement().elapsed_ns(), Some(10));
    }

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn terminal_wave_reads_output_before_releasing_backing() {
    let (fixture, sequence, session, batch, step) = setup();
    let executable = ExecutablePlan::new(
        fixture.plan.clone(),
        fixture.resolved.parts().capabilities.clone(),
    )
    .unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&executable, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &executable,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &executable,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Completion,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let request = CompletionReadbackRequest::new(
        id("node.tail"),
        0,
        id("resource.output"),
        0,
        HostTransferLayout::new(ElementType::F32, 4).unwrap(),
    )
    .unwrap();
    let duplicate = CompletionReadbackBatchRequest::new(vec![request.clone(), request.clone()]);
    assert!(duplicate.is_err());
    assert_eq!(reaper.retained_count(), 1);

    let out_of_range = CompletionReadbackBatchRequest::new(vec![
        request.clone(),
        CompletionReadbackRequest::new(
            id("node.tail"),
            1,
            id("resource.output"),
            0,
            HostTransferLayout::new(ElementType::F32, 4).unwrap(),
        )
        .unwrap(),
    ])
    .unwrap();
    assert!(handle.wait_with_readbacks(out_of_range).is_err());
    assert_eq!(reaper.retained_count(), 1);

    let foreign = CompletionReadbackBatchRequest::new(vec![CompletionReadbackRequest::new(
        id("node.foreign"),
        0,
        id("resource.output"),
        0,
        HostTransferLayout::new(ElementType::F32, 4).unwrap(),
    )
    .unwrap()])
    .unwrap();
    assert!(handle.wait_with_readbacks(foreign).is_err());
    assert_eq!(reaper.retained_count(), 1);

    let receipt = match handle.wait_with_readback(request).unwrap() {
        CompletionReadbackObservation::Terminal(receipt) => receipt,
        other => panic!("wave output readback did not terminate: {other:?}"),
    };
    assert!(matches!(
        receipt.completion().disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    let fence_timing = receipt.completion().fence_timing();
    assert_eq!(fence_timing.timing_mode(), DeviceTimingMode::Completion);
    assert!(matches!(
        fence_timing.device_execution(),
        DeviceTimingMeasurement::Measured(timing) if timing.elapsed_ns() == 1_000_000
    ));
    assert!(matches!(
        fence_timing.blocking_wait_host_ns(),
        DeviceTimingMeasurement::Measured(_)
    ));
    assert!(matches!(
        receipt.readback_timing(),
        Some(DeviceTimingMeasurement::Measured(timing))
            if timing.calls() == 1 && timing.bytes() == 16
    ));
    let output = match receipt.disposition() {
        CompletionReadbackDisposition::Succeeded(output) => output,
        other => panic!("wave output readback failed: {other:?}"),
    };
    assert_eq!(output.request().node_id(), &id("node.tail"));
    assert_eq!(output.request().resource_id(), &id("resource.output"));
    assert_eq!(output.bytes(), &[0; 16]);
    assert_eq!(output.sha256().len(), 64);
    assert_eq!(receipt.fingerprint().len(), 64);
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert!(trace.readback_calls >= 1);
        assert_eq!(trace.readback_lengths.iter().sum::<u64>(), 16);
    }
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    assert!(handle.wait().is_err(), "terminal slot must be reaped once");

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn terminal_wave_reads_multiple_canonical_node_groups_under_one_fence() {
    let (fixture, sequence, session, batch, step) = setup();
    let executable = ExecutablePlan::new(
        fixture.plan.clone(),
        fixture.resolved.parts().capabilities.clone(),
    )
    .unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active_bindings = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&executable, node.id()).unwrap())
        .collect::<Vec<_>>();
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &executable,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &executable,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Completion,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let group = |node: &str, resource: &str| {
        CompletionReadbackBatchRequest::new(vec![CompletionReadbackRequest::new(
            id(node),
            0,
            id(resource),
            0,
            HostTransferLayout::new(ElementType::F32, 4).unwrap(),
        )
        .unwrap()])
        .unwrap()
    };
    assert!(CompletionReadbackCollectionRequest::new(Vec::new()).is_err());
    assert!(CompletionReadbackCollectionRequest::new(vec![
        group("node.tail", "resource.output"),
        group("node.tail", "resource.output"),
    ])
    .is_err());
    let collection = CompletionReadbackCollectionRequest::new(vec![
        group("node.tail", "resource.output"),
        group("node.main", "resource.intermediate"),
    ])
    .unwrap();
    assert_eq!(collection.len(), 2);
    assert_eq!(collection.request_count(), 2);

    let receipt = match handle.wait_with_readback_collection(collection).unwrap() {
        CompletionReadbackCollectionObservation::Terminal(receipt) => receipt,
        other => panic!("wave collection readback did not terminate: {other:?}"),
    };
    assert_eq!(receipt.dispositions().len(), 2);
    assert!(receipt
        .dispositions()
        .iter()
        .all(|disposition| matches!(disposition, CompletionReadbackDisposition::Succeeded(_))));
    let groups = receipt
        .dispositions()
        .iter()
        .map(|disposition| match disposition {
            CompletionReadbackDisposition::Succeeded(output) => (
                output.request().node_id().as_str(),
                output.request().resource_id().as_str(),
            ),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        groups,
        vec![
            ("node.main", "resource.intermediate"),
            ("node.tail", "resource.output"),
        ]
    );
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn provider_declared_binding_compute_and_result_phases_share_one_wave() {
    let (fixture, sequence, session, batch, step) = setup();
    *fixture.provider_behavior.lock().unwrap() = ProviderBehavior::SplitPhases;
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
    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();

    let trace = fixture.runtime_trace.lock().unwrap();
    assert_eq!(trace.submitted_command_counts, vec![6]);
    assert_eq!(
        trace.submitted_command_phases,
        vec![vec![
            DeviceCommandPhase::DynamicBinding,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::ResultBinding,
            DeviceCommandPhase::DynamicBinding,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::ResultBinding,
        ]]
    );
    drop(trace);
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn provider_program_bindings_are_coalesced_once_before_all_wave_compute() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
    assert!(batch_identity.nodes().iter().all(|node| {
        node.provider_execution_semantics()
            == ProviderExecutionSemantics::bitwise_eager_and_replay()
    }));
    let expected_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave must expose reusable capture identity");

    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.program_binding_coalesce_calls, 1);
        assert_eq!(trace.program_binding_input_counts, vec![2]);
        assert_eq!(trace.submitted_command_counts, vec![3]);
        assert_eq!(
            trace.submitted_command_phases,
            vec![vec![
                DeviceCommandPhase::DynamicBinding,
                DeviceCommandPhase::Compute,
                DeviceCommandPhase::Compute,
            ]]
        );
        assert_eq!(
            trace.submitted_commands,
            vec![vec![
                TestCommand::CoalescedProgramBinding,
                TestCommand::Provider,
                TestCommand::Provider,
            ]]
        );
        assert_eq!(trace.submitted_reusable_captures.len(), 1);
        let capture = trace.submitted_reusable_captures[0]
            .as_ref()
            .expect("full encode must carry reusable capture metadata");
        assert_eq!(capture.program_id(), &expected_program_id);
        assert_eq!(capture.per_wave_binding_node_indices(), &[0, 1]);
    }
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.program_binding_slots.len(), 2);
        for (node_index, node) in fixture.plan.payload().nodes().iter().enumerate() {
            assert_eq!(
                trace.program_binding_slots.get(&node_index),
                node.binding_resource()
            );
        }
        assert_eq!(
            trace.program_binding_plan_hashes,
            BTreeSet::from([fixture.plan.plan_hash().as_str().to_owned()])
        );
        assert_eq!(trace.program_binding_layout_fingerprints.len(), 1);
        assert_eq!(trace.program_binding_lane_slot_ids.len(), 1);
        assert_eq!(
            trace.program_binding_lifetimes,
            vec![
                AllocationLifetime::Invocation,
                AllocationLifetime::Invocation
            ]
        );
    }
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn provider_keeps_lane_stable_scratch_in_a_reusable_segment() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBindingWithScratchTail),
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

    let handle = OperationDispatch::encode_and_submit_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        let capture = trace.submitted_reusable_captures[0]
            .as_ref()
            .expect("eager-boundary wave must preserve its reusable topology identity");
        assert!(capture.eager_boundary_node_indices().is_empty());
    }
    assert_eq!(
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .reusable_topology_calls,
        2,
        "each provider must classify the exact workspace captured by its command"
    );
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn provider_contract_skips_identity_materialization_for_overwrite_scratch() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBindingWithScratchTail),
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
    assert_eq!(providers.len(), 2);
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let batch_identity = OperationDispatch::bind_compiled_submission_wave_identity(
        &topology,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    assert_eq!(
        batch_identity
            .materialization_snapshot()
            .materialized_nodes(),
        0
    );
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("scratch-tail wave must have a reusable program identity");
    let node_count = u32::try_from(providers.len()).unwrap();
    let program = test_reusable_program(
        program_id,
        node_count,
        vec![],
        vec![DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap()],
        vec![0],
        vec![],
    );

    let handle = OperationDispatch::encode_and_submit_reusable_wave_with_inputs(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        &program,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();

    assert_eq!(
        batch_identity
            .materialization_snapshot()
            .materialized_nodes(),
        1,
        "only the live program-binding node should materialize; overwrite scratch needs no identity"
    );
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));

    drop(handle);
    drop(topology);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn sealed_reusable_program_encodes_only_bindings_and_one_direct_segment() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let batch_identity = OperationDispatch::bind_compiled_submission_wave_identity(
        &topology,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    assert_eq!(
        batch_identity
            .materialization_snapshot()
            .materialized_nodes(),
        0
    );
    assert!(!batch_identity
        .materialization_snapshot()
        .full_participant_projection());
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave must have a reusable program identity");
    let node_count = u32::try_from(providers.len()).unwrap();
    let segment = DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap();
    let program = test_reusable_program(
        program_id,
        node_count,
        vec![],
        vec![segment],
        (0..node_count).collect(),
        vec![],
    );

    let handle = OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_policy(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        &program,
        SubmissionExecutionPolicy::determinism_replayed(0xa5),
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    assert!(!handle.receipt().has_materialized_participant_receipts());

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.program_binding_coalesce_calls, 1);
        assert_eq!(trace.program_binding_input_counts, vec![2]);
        assert_eq!(trace.submitted_command_counts, vec![2]);
        assert_eq!(
            trace.submitted_command_phases,
            vec![vec![
                DeviceCommandPhase::DynamicBinding,
                DeviceCommandPhase::Compute,
            ]]
        );
        assert_eq!(
            trace.submitted_commands,
            vec![vec![
                TestCommand::CoalescedProgramBinding,
                TestCommand::ReusableExecution,
            ]]
        );
        assert_eq!(trace.submitted_reusable_captures, vec![None]);
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::ReplayedOnly]
        );
    }
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.encode_calls, 0);
        assert_eq!(trace.reusable_binding_encode_calls, 2);
        assert_eq!(trace.program_binding_slots.len(), 2);
    }
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert!(!batch_identity
        .materialization_snapshot()
        .full_participant_projection());

    drop(handle);
    drop(topology);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn eager_boundary_preserves_adjacent_direct_segment_execution() {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_provider_behavior(
            false,
            ProviderBehavior::ProgramBindingFirstNodeEagerBoundary,
        ));
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
    assert_eq!(providers.len(), 2);
    let batch_identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("an eager boundary must retain typed partial program authority");
    let program = test_reusable_program(
        program_id,
        2,
        vec![0],
        vec![DeviceReusableExecutionSegment::new(0, 1, 2, 1).unwrap()],
        vec![1],
        vec![],
    );

    let handle = OperationDispatch::encode_and_submit_reusable_wave_with_inputs(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        &program,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.program_binding_coalesce_calls, 1);
        assert_eq!(trace.program_binding_input_counts, vec![2]);
        assert_eq!(trace.submitted_command_counts, vec![3]);
        assert_eq!(
            trace.submitted_commands,
            vec![vec![
                TestCommand::CoalescedProgramBinding,
                TestCommand::Provider,
                TestCommand::ReusableExecution,
            ]]
        );
        assert_eq!(trace.submitted_reusable_captures, vec![None]);
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::Adaptive]
        );
    }
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.encode_calls, 1);
        assert_eq!(trace.reusable_binding_encode_calls, 1);
        assert_eq!(trace.program_binding_slots.len(), 2);
    }
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));

    drop(handle);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_replay_submission_restores_state_and_enforces_replayed_compute() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
        0x42,
    );
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave has a reusable identity");
    let node_count = u32::try_from(providers.len()).unwrap();
    let program = test_reusable_program(
        program_id,
        node_count,
        vec![],
        vec![DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap()],
        (0..node_count).collect(),
        vec![],
    );

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
    let attribution = handle
        .attribution()
        .expect("determinism replay must retain actual-path attribution");
    assert_eq!(attribution.device().replayed_segments().len(), 1);
    assert_eq!(
        attribution.device().replayed_segments()[0]
            .logical_commands()
            .len(),
        providers.len()
    );

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::ReplayedOnly]
        );
        assert_eq!(
            trace.submitted_attribution_requirements,
            vec![DeviceSubmissionAttributionRequirement::LogicalExecutionPath]
        );
        assert!(trace
            .uploaded_payloads
            .iter()
            .any(|payload| !payload.is_empty() && payload.iter().all(|byte| *byte == 0x42)));
        let phases = &trace.submitted_command_phases[0];
        let first_compute = phases
            .iter()
            .position(|phase| *phase == DeviceCommandPhase::Compute)
            .expect("determinism replay has one compute command");
        assert!(phases[..first_compute]
            .iter()
            .all(|phase| *phase != DeviceCommandPhase::Compute));
        assert_eq!(phases[first_compute..], [DeviceCommandPhase::Compute]);
        assert_eq!(
            trace.submitted_commands[0][first_compute],
            TestCommand::ReusableExecution
        );
    }
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    let expected_readbacks = handle.readback_plan().collection_request().request_count();
    let evidence = handle.wait_into_evidence().unwrap();
    assert_eq!(
        evidence.expected_execution_path(),
        DeviceExecutionPath::Replayed
    );
    assert_eq!(evidence.physical_readbacks().len(), expected_readbacks);
    assert!(evidence
        .physical_readbacks()
        .iter()
        .all(|readback| !readback.bytes().is_empty() && readback.raw_sha256().len() == 64));
    let artifact =
        serde_json::to_value(evidence.into_artifact_execution("replay-00").unwrap()).unwrap();
    assert_eq!(artifact["mode"], "replay");
    assert_eq!(
        artifact["attribution"]["replayed_segments"][0]["logical_commands"]
            .as_array()
            .map(Vec::len),
        Some(providers.len())
    );
    assert!(artifact["attribution"]
        .get("reusable_executable_fingerprint")
        .is_none());

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_replay_accepts_only_live_declared_eager_boundaries() {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_determinism_provider_behavior(
            false,
            ProviderBehavior::ProgramBindingFirstNodeEagerBoundary,
        ));
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
    assert_eq!(providers.len(), 2);
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
    .expect("partial determinism wave has reusable authority");
    let program_fingerprint = program_id.fingerprint();
    let program = test_reusable_program(
        program_id,
        2,
        vec![0],
        vec![DeviceReusableExecutionSegment::new(0, 1, 2, 1).unwrap()],
        vec![1],
        vec![],
    );

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

    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(
            trace.submitted_compute_path_requirements,
            vec![DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries]
        );
        assert!(trace.submitted_commands.last().is_some_and(|commands| {
            commands.ends_with(&[
                TestCommand::CoalescedProgramBinding,
                TestCommand::Provider,
                TestCommand::ReusableExecution,
            ])
        }));
    }
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.encode_calls, 1);
        assert_eq!(trace.reusable_binding_encode_calls, 1);
    }

    let evidence = handle.wait_into_evidence().unwrap();
    assert_eq!(
        evidence.expected_compute_path_requirement(),
        DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
    );
    assert_eq!(
        evidence
            .declared_eager_boundary_node_ids()
            .iter()
            .map(NodeId::as_str)
            .collect::<Vec<_>>(),
        vec!["node.main"]
    );
    assert_eq!(
        evidence.reusable_program_fingerprint().as_deref(),
        Some(program_fingerprint.as_str())
    );
    let artifact =
        serde_json::to_value(evidence.into_artifact_execution("replay-00").unwrap()).unwrap();
    assert_eq!(
        artifact["compute_path_requirement"],
        "replayed_with_declared_eager_boundaries"
    );
    assert_eq!(
        artifact["declared_eager_boundary_node_ids"],
        json!(["node.main"])
    );
    assert_eq!(
        artifact["reusable_program_fingerprint"],
        program_fingerprint
    );
    assert_eq!(
        artifact["attribution"]["replayed_segments"][0]["reusable_program_fingerprint"],
        artifact["reusable_program_fingerprint"]
    );

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn determinism_replay_rejects_an_unresident_non_boundary_gap_before_submission() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
        0x42,
    );
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("determinism wave has reusable authority");
    let incomplete_program = test_reusable_program(
        program_id,
        2,
        vec![],
        vec![],
        vec![],
        vec![
            DeviceReusableExecutionProgramGap::new(
                0,
                DeviceReusableExecutionProgramGapReason::ProviderReplayKeyMissing,
            ),
            DeviceReusableExecutionProgramGap::new(
                1,
                DeviceReusableExecutionProgramGapReason::CapacityDeferred,
            ),
        ],
    );
    assert!(!incomplete_program.is_determinism_ready());
    assert!(!incomplete_program.has_resident_segments());
    assert_eq!(
        incomplete_program.state(),
        DeviceReusableExecutionProgramState::Partial
    );

    let error = match OperationDispatch::encode_and_submit_determinism_replayed_wave(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Replay,
        &restore,
        0xa5,
        &incomplete_program,
        wave,
        &lane,
        &reaper,
    ) {
        Ok(_) => panic!("an undeclared eager gap must fail before submission"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("2 non-resident replay-eligible gap"),
        "{message}"
    );
    assert!(message.contains("node_index=0"), "{message}");
    assert!(
        message.contains(&format!(
            "node_id={} provider_id={} operation_id={}",
            batch_identity.node_id_at(0).unwrap().as_str(),
            batch_identity.provider_id_at(0).unwrap().as_str(),
            batch_identity.operation_id_at(0).unwrap().as_str(),
        )),
        "{message}"
    );
    assert!(
        message.contains("reason=provider_replay_key_missing"),
        "{message}"
    );
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn product_dispatch_rejects_a_diagnostic_only_reusable_program_before_provider_encode() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave has reusable authority");
    let node_count = u32::try_from(providers.len()).unwrap();
    let diagnostic_program = test_reusable_program(
        program_id,
        node_count,
        vec![],
        vec![],
        vec![],
        (0..node_count)
            .map(|node_index| {
                DeviceReusableExecutionProgramGap::new(
                    node_index,
                    DeviceReusableExecutionProgramGapReason::CapacityDeferred,
                )
            })
            .collect(),
    );

    let error = OperationDispatch::encode_and_submit_reusable_wave_with_inputs(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        &diagnostic_program,
        wave,
        &lane,
        &reaper,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("product reusable execution requires at least one resident segment"));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);

    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn reusable_topology_states_cannot_alias_resident_program_authority() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();

    let dynamic_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("dynamic topology must produce a reusable program identity");

    *fixture.provider_behavior.lock().unwrap() = ProviderBehavior::Success;
    let static_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("static topology must retain the base reusable program identity");
    assert_ne!(dynamic_program_id, static_program_id);

    *fixture.provider_behavior.lock().unwrap() =
        ProviderBehavior::ProgramBindingFirstNodeEagerBoundary;
    let eager_boundary_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("an eager node boundary must preserve partial reusable program authority");
    assert_ne!(eager_boundary_program_id, dynamic_program_id);
    assert_ne!(eager_boundary_program_id, static_program_id);

    drop(providers);
    drop(wave);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn eager_only_provider_cannot_authorize_reusable_topology() {
    let (fixture, sequence, session, batch, step) =
        setup_with_fixture(fixture_with_provider_behavior_and_execution_semantics(
            false,
            ProviderBehavior::ProgramBinding,
            ProviderExecutionSemantics::bitwise_eager_only(),
            ExecutionDeterminismRequirement::BitwiseSameRuntime,
        ));
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture
        .plan
        .payload()
        .nodes()
        .iter()
        .map(|node| fixture.registry.bind(&fixture.resolved, node.id()).unwrap())
        .collect::<Vec<_>>();

    let error = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("without a bitwise eager-equivalence contract"));
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert!(fixture
        .runtime_trace
        .lock()
        .unwrap()
        .submitted_command_counts
        .is_empty());

    drop(providers);
    drop(wave);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn stale_reusable_topology_is_rejected_before_dispatch_or_encoding() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
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
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let batch_identity = OperationDispatch::bind_compiled_submission_wave_identity(
        &topology,
        active_bindings.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let live_program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        &providers,
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .expect("program-binding wave must have a reusable program identity");
    let stale_program_id = live_program_id.with_topology_fingerprint(
        DeviceReusableExecutionTopologyFingerprint::from_sha256([0xff; 32]),
    );
    let node_count = u32::try_from(providers.len()).unwrap();
    let stale_program = test_reusable_program(
        stale_program_id,
        node_count,
        vec![],
        vec![DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap()],
        (0..node_count).collect(),
        vec![],
    );

    let error = OperationDispatch::encode_and_submit_reusable_wave_with_inputs(
        &providers,
        &fixture.resolved,
        &batch_identity,
        active_bindings.iter(),
        DeviceTimingMode::Off,
        &[],
        &stale_program,
        wave,
        &lane,
        &reaper,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SubmissionWaveDispatchError::Contract(ref error)
            if error.to_string().contains(
                "reusable execution program differs from the exact wave topology"
            )
    ));
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 0);
        assert!(trace.submitted_commands.is_empty());
    }
    {
        let trace = fixture.provider_trace.lock().unwrap();
        assert_eq!(trace.encode_calls, 0);
        assert_eq!(trace.reusable_binding_encode_calls, 0);
    }
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);

    drop(stale_program);
    drop(topology);
    drop(providers);
    drop(active_bindings);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}
