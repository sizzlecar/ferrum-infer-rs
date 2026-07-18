mod vnext_event_contract;

use vnext_event_contract::*;

#[test]
fn vnext_event_sink_contract() {
    const EXPECTED_CASES: usize = 27;
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
    let default_batch_sink = RecordingSink::default();
    let mut default_batch_emitter = ExecutionEventEmitter::new(
        &default_batch_sink,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    let default_batch_contexts = [
        TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        TrustedExecutionEventContext::bound(active.run_id(), active.request_id(), &topology),
    ];
    default_batch_emitter
        .emit_batch(&journal[..2], &default_batch_contexts)
        .unwrap();
    check(
        &mut passed,
        default_batch_sink.kinds.lock().unwrap().as_slice()
            == [
                ExecutionEventKind::RequestAccepted,
                ExecutionEventKind::PlanBuilt,
            ],
    );
    check(
        &mut passed,
        default_batch_emitter.cursor().last_sequence() == 2,
    );
    let batch_sink = BatchRecordingSink::default();
    let mut batch_emitter = ExecutionEventEmitter::new(
        &batch_sink,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    let invalid_contexts = [
        TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
    ];
    check(
        &mut passed,
        batch_emitter
            .emit_batch(&journal[..2], &invalid_contexts)
            .is_err(),
    );
    check(
        &mut passed,
        *batch_sink.batch_calls.lock().unwrap() == 0
            && batch_emitter.cursor().last_sequence() == 0
            && !batch_emitter.sink_failed(),
    );
    let contexts = [
        TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        TrustedExecutionEventContext::bound(active.run_id(), active.request_id(), &topology),
    ];
    batch_emitter.emit_batch(&journal[..2], &contexts).unwrap();
    check(
        &mut passed,
        *batch_sink.batch_calls.lock().unwrap() == 1
            && *batch_sink.record_calls.lock().unwrap() == 0,
    );
    check(
        &mut passed,
        batch_sink.kinds.lock().unwrap().as_slice()
            == [
                ExecutionEventKind::RequestAccepted,
                ExecutionEventKind::PlanBuilt,
            ],
    );
    check(&mut passed, batch_emitter.cursor().last_sequence() == 2);
    let mut failed_batch_emitter = ExecutionEventEmitter::new(
        &FailingSink,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    check(
        &mut passed,
        failed_batch_emitter
            .emit_batch(&journal[..2], &contexts)
            .is_err(),
    );
    check(
        &mut passed,
        failed_batch_emitter.cursor().last_sequence() == 0 && failed_batch_emitter.sink_failed(),
    );
    let disabled_sink = DisabledRecordingSink::default();
    let mut disabled_emitter = ExecutionEventEmitter::new(
        &disabled_sink,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    disabled_emitter
        .emit_batch(&journal[..2], &contexts)
        .unwrap();
    check(
        &mut passed,
        *disabled_sink.record_calls.lock().unwrap() == 0
            && *disabled_sink.batch_calls.lock().unwrap() == 0,
    );
    check(
        &mut passed,
        disabled_emitter.cursor().last_sequence() == 2 && !disabled_emitter.sink_failed(),
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
