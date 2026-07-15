mod vnext_event_contract;

use vnext_event_contract::*;

#[test]
fn vnext_event_sink_contract() {
    const EXPECTED_CASES: usize = 16;
    let mut passed = 0_usize;
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = execution_plan("sink", &operation_registry);
    let topology = TrustedExecutionTopology::from_plan(&plan).unwrap();
    let resolved = resolved_model_plan(&plan, "sink", &operation_registry);
    let ProvisionedRuntimePool {
        resources,
        runtime,
        evidence: _,
        journal: _,
        committed_snapshot: _,
    } = provision_runtime_pool(&plan, &topology, "sink");
    let SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    } = execute_sequence(
        &resources,
        &runtime,
        &resolved,
        &operation_registry,
        "run.sink",
        "request.sink",
        1,
    );
    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 1);
    let accepted = &journal[0];

    live_witness_emitter_contract(
        &mut passed,
        &resources,
        &runtime,
        &resolved,
        &operation_registry,
    );

    let sink = RecordingSink::default();
    check(
        &mut passed,
        sink.capture_policy() == ExecutionEventCapturePolicy::AllFrames,
    );
    check(
        &mut passed,
        ExecutionEventCapturePolicy::FirstFramePerRequest.captures_frame(0)
            && !ExecutionEventCapturePolicy::FirstFramePerRequest.captures_frame(1),
    );
    let mut emitter =
        ExecutionEventEmitter::new(&sink, active.run_id().clone(), active.request_id().clone());
    emitter
        .emit(
            accepted,
            &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        )
        .unwrap();
    check(
        &mut passed,
        sink.kinds.lock().unwrap().as_slice() == [ExecutionEventKind::RequestAccepted],
    );
    check(&mut passed, emitter.cursor().last_sequence() == 1);
    let shared_sink = std::sync::Arc::new(RecordingSink::default());
    let durable_sink: std::sync::Arc<dyn ExecutionEventSink> = shared_sink.clone();
    let mut durable_emitter = ExecutionEventEmitter::from_shared(
        durable_sink,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    durable_emitter
        .emit(
            accepted,
            &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        )
        .unwrap();
    check(
        &mut passed,
        shared_sink.kinds.lock().unwrap().as_slice() == [ExecutionEventKind::RequestAccepted],
    );
    let mut failed_emitter = ExecutionEventEmitter::new(
        &FailingSink,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    check(
        &mut passed,
        failed_emitter
            .emit(
                accepted,
                &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
            )
            .is_err(),
    );
    check(
        &mut passed,
        failed_emitter.cursor().last_sequence() == 0 && failed_emitter.sink_failed(),
    );
    check(
        &mut passed,
        failed_emitter
            .emit(
                accepted,
                &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
            )
            .is_err(),
    );
    let event_sink_source = include_str!("../src/vnext/event/sink.rs");
    check(
        &mut passed,
        event_sink_source.contains("pub struct EventEmissionPermit<'event>")
            && !event_sink_source.contains("pub fn new_event_emission_permit"),
    );

    close_plan_runtime(resources);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT EVENT SINK PASS: {passed}/{EXPECTED_CASES}");
}
