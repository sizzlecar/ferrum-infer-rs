use super::*;

pub(crate) fn pool_timestamp(sequence: u64) -> MonotonicTimestamp {
    MonotonicTimestamp {
        nanos_since_run_start: sequence * 100,
    }
}

pub(crate) fn provision_pool(
    plan: &ExecutionPlan,
    topology: &TrustedExecutionTopology,
    suffix: &str,
) -> (
    ResourceTransaction<TestDriver, TransactionCommitted>,
    Arc<TestRuntime>,
    ResourcePoolEvidence,
    Vec<ResourcePoolEvent>,
) {
    let (transaction, _, runtime) = transaction(
        plan,
        &format!("run.provision.{suffix}"),
        &format!("transaction.provision.{suffix}"),
        &format!("request.provision.{suffix}"),
        CommitBehavior::Valid,
    );
    let reserved = transaction.reserve().unwrap();
    let evidence =
        ResourcePoolEvidence::from_external(topology, reserved.admission(), reserved.identity())
            .unwrap();
    let opened = ResourcePoolEvent::opened(1, pool_timestamp(1), &evidence).unwrap();
    let reserve_event = ResourcePoolEvent::transition(
        2,
        pool_timestamp(2),
        &evidence,
        reserved.receipts().last().unwrap(),
        reserved.latest_transition_validation_context().unwrap(),
    )
    .unwrap();
    let committed = reserved.commit().unwrap();
    let commit_event = ResourcePoolEvent::transition(
        3,
        pool_timestamp(3),
        &evidence,
        committed.receipts().last().unwrap(),
        committed.latest_transition_validation_context().unwrap(),
    )
    .unwrap();
    (
        committed,
        runtime,
        evidence,
        vec![opened, reserve_event, commit_event],
    )
}

pub(crate) struct ProvisionedRuntimePool {
    pub(crate) resources: Arc<PlanRuntimeResources<TestRuntime>>,
    pub(crate) runtime: Arc<TestRuntime>,
    pub(crate) evidence: ResourcePoolEvidence,
    pub(crate) journal: Vec<ResourcePoolEvent>,
    pub(crate) committed_snapshot: ResourceLedgerSnapshot,
}

pub(crate) fn provision_runtime_pool(
    plan: &ExecutionPlan,
    topology: &TrustedExecutionTopology,
    suffix: &str,
) -> ProvisionedRuntimePool {
    let (committed, runtime, evidence, journal) = provision_pool(plan, topology, suffix);
    let pool_ids = committed
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool_id in pool_ids {
        committed
            .maintenance_controller()
            .initialize_pool(&pool_id)
            .unwrap();
    }
    let committed_snapshot = committed.ledger_snapshot();
    let resources = match committed.into_plan_runtime() {
        Ok(resources) => resources,
        Err(failure) => panic!("event plan runtime handoff failed: {}", failure.error()),
    };
    ProvisionedRuntimePool {
        resources,
        runtime,
        evidence,
        journal,
        committed_snapshot,
    }
}

pub(crate) fn provision_no_static_plan_runtime(
    plan: &ExecutionPlan,
) -> (Arc<PlanRuntimeResources<TestRuntime>>, Arc<TestRuntime>) {
    let (driver, _) = driver(plan, CommitBehavior::Valid);
    let runtime = Arc::clone(driver.runtime());
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(
            Arc::clone(driver.runtime()),
            id("request.event.no-static.provision"),
        )
        .unwrap()
        .into_parts();
    let resources = match provisioning {
        StaticProvisioning::NoStatic(no_static) => no_static.into_plan_runtime(),
        StaticProvisioning::Required(_) => {
            panic!("no-static event fixture unexpectedly requires static provisioning")
        }
    };
    (resources, runtime)
}

pub(crate) fn close_plan_runtime(
    resources: Arc<PlanRuntimeResources<TestRuntime>>,
) -> PlanRuntimeCloseReceipt {
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => receipt,
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("event plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!("event plan runtime close failed: {:?}", failure.failure()),
    }
}

pub(crate) struct SequenceEvidence {
    pub(crate) active: TrustedActiveSequenceBinding,
    pub(crate) completed: TrustedCompletedSequenceBinding,
    pub(crate) submissions: Vec<SubmittedOperationReceipt>,
    pub(crate) completions: Vec<OperationCompletionReceipt>,
}

pub(crate) struct FailureSequenceEvidence {
    pub(crate) active: TrustedActiveSequenceBinding,
    pub(crate) completed: Option<TrustedCompletedSequenceBinding>,
    pub(crate) aborted: Option<TrustedAbortedSequenceBinding>,
    pub(crate) submissions: Vec<SubmittedOperationReceipt>,
    pub(crate) completions: Vec<OperationCompletionReceipt>,
    pub(crate) journal: Vec<ExecutionEvent>,
}

#[derive(Clone, Copy)]
pub(crate) enum ReplayTerminalFixtureMode {
    ContractFailed,
    Drained,
    Quarantined,
    SubmissionIndeterminateDrained,
}

pub(crate) struct ReplayTerminalFailureEvidence {
    pub(crate) sequence: FailureSequenceEvidence,
    pub(crate) drains: Vec<CompletionDrainReceipt>,
    pub(crate) quarantines: Vec<CompletionQuarantineReceipt>,
}

pub(crate) fn suppress_expected_panic_hook<T>(operation: impl FnOnce() -> T) -> T {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
    std::panic::set_hook(previous);
    match outcome {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

pub(crate) fn logical_resources(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    run_id: &str,
    request_id: &str,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    let request = RequestResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let binding = plan_resources.trusted_runtime_binding().unwrap();
    let mut request_maintenance_attempts = 0;
    let request_resources = loop {
        match binding
            .try_admit_request(request.clone(), id(run_id), id(request_id))
            .unwrap()
        {
            RequestResourceAdmissionDecision::Admitted(resources) => break resources,
            RequestResourceAdmissionDecision::BackingDeferred(deferred) => {
                assert!(
                    request_maintenance_attempts < MAX_EVENT_MAINTENANCE_ATTEMPTS,
                    "event request admission did not converge after bounded maintenance"
                );
                request_maintenance_attempts += 1;
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            RequestResourceAdmissionDecision::Deferred(_) => {
                panic!("event fixture request admission deferred")
            }
            RequestResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("event fixture request admission rejected")
            }
        }
    };
    let sequence = SequenceResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let mut sequence_maintenance_attempts = 0;
    loop {
        match request_resources
            .try_admit_sequence(sequence.clone())
            .unwrap()
        {
            SequenceResourceAdmissionDecision::Admitted(resources) => break resources,
            SequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                assert!(
                    sequence_maintenance_attempts < MAX_EVENT_MAINTENANCE_ATTEMPTS,
                    "event sequence admission did not converge after bounded maintenance"
                );
                sequence_maintenance_attempts += 1;
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            SequenceResourceAdmissionDecision::Deferred(_) => {
                panic!("event fixture sequence admission deferred")
            }
            SequenceResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("event fixture sequence admission rejected")
            }
        }
    }
}

pub(crate) fn begin_single_participant_step(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![one_token_span()]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=MAX_EVENT_MAINTENANCE_ATTEMPTS {
        match batch.try_begin_step(request.clone()).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred)
                if attempt < MAX_EVENT_MAINTENANCE_ATTEMPTS =>
            {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            StepResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("event step backing did not converge after bounded maintenance")
            }
            StepResourceAdmissionDecision::Deferred(_) => {
                panic!("event single-participant step unexpectedly deferred")
            }
            StepResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("event single-participant step unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded event step admission always returns or panics")
}

pub(crate) fn admit_single_participant_invocation(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_id: &NodeId,
) -> InvocationResourceLease<TestRuntime> {
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        node_id.clone(),
        step.bind_all_invocation_work_shape(vec![one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=MAX_EVENT_MAINTENANCE_ATTEMPTS {
        match step.try_admit_invocation(request.clone()).unwrap() {
            InvocationResourceAdmissionDecision::Admitted(invocation) => return invocation,
            InvocationResourceAdmissionDecision::BackingDeferred(deferred)
                if attempt < MAX_EVENT_MAINTENANCE_ATTEMPTS =>
            {
                plan_resources.maintain_for_deferred(&deferred).unwrap();
            }
            InvocationResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("event invocation backing did not converge after bounded maintenance")
            }
            InvocationResourceAdmissionDecision::Deferred(_) => {
                panic!("event single-participant invocation unexpectedly deferred")
            }
            InvocationResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("event single-participant invocation unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded event invocation admission always returns or panics")
}
