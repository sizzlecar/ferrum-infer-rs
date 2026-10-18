mod vnext_event_contract;

use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use vnext_event_contract::*;

struct DeliverySink {
    policy: ExecutionEventDeliveryPolicy,
    failures: Mutex<VecDeque<Option<ExecutionEventSinkError>>>,
    calls: AtomicUsize,
}

impl DeliverySink {
    fn new(
        policy: ExecutionEventDeliveryPolicy,
        failures: Vec<Option<ExecutionEventSinkError>>,
    ) -> Self {
        Self {
            policy,
            failures: Mutex::new(failures.into()),
            calls: AtomicUsize::new(0),
        }
    }
}

impl ExecutionEventSink for DeliverySink {
    fn delivery_policy(&self) -> ExecutionEventDeliveryPolicy {
        self.policy
    }
    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        true
    }
    fn record(&self, _: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.failures
            .lock()
            .unwrap()
            .pop_front()
            .flatten()
            .map_or(Ok(()), Err)
    }
}

#[test]
fn optional_delivery_keeps_validation_atomic_and_required_failure_keeps_original_cause() {
    let runtime_catalog = catalog();
    let registry = make_operation_registry(&runtime_catalog);
    let plan = execution_plan("delivery", &registry);
    let topology = TrustedExecutionTopology::from_plan(&plan).unwrap();
    let resolved = resolved_model_plan(&plan, "delivery", &registry);
    let ProvisionedRuntimePool {
        resources, runtime, ..
    } = provision_runtime_pool(&plan, &topology, "delivery");
    let SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    } = execute_sequence(
        &resources,
        &runtime,
        &resolved,
        &registry,
        "run.delivery",
        "request.delivery",
        1,
    );
    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 1);
    let contexts = [
        TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        TrustedExecutionEventContext::bound(active.run_id(), active.request_id(), &topology),
    ];
    let transport = ExecutionEventSinkError::transport("queue is full");
    let contract = ExecutionEventSinkError::contract("profile projection rejected identity");

    let optional = DeliverySink::new(
        ExecutionEventDeliveryPolicy::OptionalTelemetry,
        vec![Some(transport.clone()), None],
    );
    let mut emitter = ExecutionEventEmitter::new(
        &optional,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    emitter
        .emit_batch(journal[..2].to_vec(), &contexts)
        .unwrap();
    assert_eq!(emitter.cursor().last_sequence(), 2);
    assert!(!emitter.sink_failed());
    assert_eq!(emitter.transport_failure(), Some(&transport));
    assert_eq!(
        optional.calls.load(Ordering::Relaxed),
        2,
        "a transport loss must not suppress the next destination call"
    );
    let error = emitter.emit(journal[0].clone(), &contexts[0]).unwrap_err();
    assert_eq!(error.kind(), ExecutionEventSinkErrorKind::Contract);
    assert_eq!(emitter.cursor().last_sequence(), 2);
    assert_eq!(
        optional.calls.load(Ordering::Relaxed),
        2,
        "degraded delivery still validates every candidate"
    );

    let strict = DeliverySink::new(
        ExecutionEventDeliveryPolicy::RequiredEvidence,
        vec![Some(transport.clone())],
    );
    let mut emitter = ExecutionEventEmitter::new(
        &strict,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    assert_eq!(
        emitter
            .emit_batch(journal[..2].to_vec(), &contexts)
            .unwrap_err(),
        transport
    );
    assert_eq!(emitter.cursor().last_sequence(), 0);
    assert!(emitter.sink_failed());
    assert_eq!(
        emitter.emit(journal[0].clone(), &contexts[0]).unwrap_err(),
        transport
    );
    assert_eq!(
        strict.calls.load(Ordering::Relaxed),
        1,
        "sealed strict delivery preserves its original cause"
    );

    let mixed = DeliverySink::new(
        ExecutionEventDeliveryPolicy::OptionalTelemetry,
        vec![Some(transport), Some(contract.clone())],
    );
    let mut emitter =
        ExecutionEventEmitter::new(&mixed, active.run_id().clone(), active.request_id().clone());
    assert_eq!(
        emitter
            .emit_batch(journal[..2].to_vec(), &contexts)
            .unwrap_err(),
        contract
    );
    assert_eq!(
        emitter.cursor().last_sequence(),
        0,
        "a mixed batch cannot commit a validated prefix"
    );
    assert!(emitter.sink_failed());
    assert_eq!(mixed.calls.load(Ordering::Relaxed), 2);

    let invalid = DeliverySink::new(ExecutionEventDeliveryPolicy::OptionalTelemetry, vec![]);
    let mut emitter = ExecutionEventEmitter::new(
        &invalid,
        active.run_id().clone(),
        active.request_id().clone(),
    );
    let invalid_contexts = [
        TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
        TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
    ];
    assert_eq!(
        emitter
            .emit_batch(journal[..2].to_vec(), &invalid_contexts)
            .unwrap_err()
            .kind(),
        ExecutionEventSinkErrorKind::Contract
    );
    assert_eq!(emitter.cursor().last_sequence(), 0);
    assert_eq!(invalid.calls.load(Ordering::Relaxed), 0);
    emitter
        .emit_batch(journal[..2].to_vec(), &contexts)
        .unwrap();
    assert_eq!(emitter.cursor().last_sequence(), 2);
    close_plan_runtime(resources);
}
