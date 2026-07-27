use super::*;

#[derive(Default)]
pub(crate) struct RecordingSink {
    pub(crate) kinds: Mutex<Vec<ExecutionEventKind>>,
}

impl ExecutionEventSink for RecordingSink {
    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        true
    }

    fn record(&self, permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        self.kinds.lock().unwrap().push(permit.event().kind());
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct BatchRecordingSink {
    pub(crate) kinds: Mutex<Vec<ExecutionEventKind>>,
    pub(crate) record_calls: Mutex<usize>,
    pub(crate) batch_calls: Mutex<usize>,
}

impl ExecutionEventSink for BatchRecordingSink {
    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        true
    }

    fn record(&self, permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        *self.record_calls.lock().unwrap() += 1;
        self.kinds.lock().unwrap().push(permit.event().kind());
        Ok(())
    }

    fn record_batch(
        &self,
        permit: EventBatchEmissionPermit,
    ) -> Result<(), ExecutionEventSinkError> {
        *self.batch_calls.lock().unwrap() += 1;
        self.kinds
            .lock()
            .unwrap()
            .extend(permit.events().iter().map(ExecutionEvent::kind));
        Ok(())
    }
}

pub(crate) struct FailingSink;

impl ExecutionEventSink for FailingSink {
    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        true
    }

    fn record(&self, _permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        Err(ExecutionEventSinkError::new("injected sink failure"))
    }
}

#[derive(Default)]
pub(crate) struct DisabledRecordingSink {
    pub(crate) record_calls: Mutex<usize>,
    pub(crate) batch_calls: Mutex<usize>,
}

impl ExecutionEventSink for DisabledRecordingSink {
    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        false
    }

    fn record(&self, _permit: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        *self.record_calls.lock().unwrap() += 1;
        Ok(())
    }

    fn record_batch(
        &self,
        _permit: EventBatchEmissionPermit,
    ) -> Result<(), ExecutionEventSinkError> {
        *self.batch_calls.lock().unwrap() += 1;
        Ok(())
    }
}

pub(crate) fn base_parts(
    run_id: &RunId,
    request_id: &RequestIdentity,
    sequence: u64,
    span_id: impl Into<String>,
    parent_span_id: Option<SpanId>,
) -> ExecutionIdentityParts {
    ExecutionIdentityParts {
        version: EXECUTION_IDENTITY_VERSION,
        run_id: run_id.clone(),
        request_id: request_id.clone(),
        sequence,
        plan_id: None,
        plan_hash: None,
        frame_id: None,
        node_invocation_id: None,
        node_id: None,
        operation_id: None,
        provider_id: None,
        device_id: None,
        resource_pool_id: None,
        resource_pool_identity_fingerprint: None,
        provisioning_run_id: None,
        provisioning_request_id: None,
        transaction_id: None,
        active_sequence_slot: None,
        admission_generation: None,
        activation_epoch: None,
        runtime_implementation_fingerprint: None,
        active_sequence_fingerprint: None,
        completed_sequence_fingerprint: None,
        aborted_sequence_fingerprint: None,
        resource_id: None,
        resource_generation: None,
        resource_batch_fingerprint: None,
        span_id: id(span_id),
        parent_span_id,
        async_links: Vec::new(),
    }
}

pub(crate) fn bind_plan(
    mut parts: ExecutionIdentityParts,
    plan: &ExecutionPlan,
) -> ExecutionIdentityParts {
    parts.plan_id = Some(plan.payload().plan_id().clone());
    parts.plan_hash = Some(plan.plan_hash().clone());
    parts.device_id = Some(plan.payload().device_id().clone());
    parts.runtime_implementation_fingerprint = Some(
        plan.payload()
            .device_runtime_implementation_fingerprint()
            .to_owned(),
    );
    parts
}

pub(crate) fn bind_active(
    mut parts: ExecutionIdentityParts,
    active: &TrustedActiveSequenceBinding,
) -> ExecutionIdentityParts {
    let provisioning = active.static_provisioning_identity();
    parts.resource_pool_id = active.static_pool_id();
    parts.resource_pool_identity_fingerprint = active.static_pool_identity_fingerprint();
    parts.provisioning_run_id = provisioning.map(|identity| identity.run_id().clone());
    parts.provisioning_request_id = provisioning.map(|identity| identity.request_id().clone());
    parts.transaction_id = provisioning.map(|identity| identity.transaction_id().clone());
    parts.active_sequence_slot = Some(active.sequence_authority().sparse_id());
    parts.admission_generation = Some(active.sequence_authority().generation());
    parts.activation_epoch = Some(active.activation_epoch());
    debug_assert_eq!(
        parts.runtime_implementation_fingerprint.as_deref(),
        Some(active.runtime_implementation_fingerprint())
    );
    parts.active_sequence_fingerprint = Some(active.fingerprint().to_owned());
    parts
}

pub(crate) fn make_event(
    sequence: u64,
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    parts: ExecutionIdentityParts,
    detail: ExecutionEventDetail,
) -> ExecutionEvent {
    ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: sequence * 10,
        },
        phase,
        kind,
        ExecutionIdentityEnvelope::new(parts).unwrap(),
        detail,
    )
    .unwrap()
}

pub(crate) fn accepted_event(run: &RunId, request: &RequestIdentity) -> ExecutionEvent {
    make_event(
        1,
        ExecutionPhase::Resolution,
        ExecutionEventKind::RequestAccepted,
        base_parts(run, request, 1, "span.request", None),
        ExecutionEventDetail::None,
    )
}

pub(crate) fn plan_event(
    plan: &ExecutionPlan,
    run: &RunId,
    request: &RequestIdentity,
) -> ExecutionEvent {
    make_event(
        2,
        ExecutionPhase::Planning,
        ExecutionEventKind::PlanBuilt,
        bind_plan(
            base_parts(run, request, 2, "span.plan", Some(id("span.request"))),
            plan,
        ),
        ExecutionEventDetail::None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn frame_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    run: &RunId,
    request: &RequestIdentity,
    sequence: u64,
    frame: u64,
    kind: ExecutionEventKind,
) -> ExecutionEvent {
    let frame_id = ExecutionFrameId::try_from(frame).unwrap();
    let span: SpanId = id(format!("span.frame.{frame}"));
    let mut parts = bind_active(
        bind_plan(
            base_parts(
                run,
                request,
                sequence,
                span.as_str(),
                Some(id("span.request")),
            ),
            plan,
        ),
        active,
    );
    parts.frame_id = Some(frame_id);
    make_event(
        sequence,
        ExecutionPhase::Execution,
        kind,
        parts,
        ExecutionEventDetail::None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn node_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    run: &RunId,
    request: &RequestIdentity,
    sequence: u64,
    frame: u64,
    invocation: u64,
    node_index: usize,
    kind: ExecutionEventKind,
) -> ExecutionEvent {
    let node = &plan.payload().nodes()[node_index];
    let frame_span: SpanId = id(format!("span.frame.{frame}"));
    let node_span: SpanId = id(format!("span.frame.{frame}.node.{node_index}"));
    let operation_span: SpanId = id(format!("span.frame.{frame}.operation.{node_index}"));
    let (span, parent) = match kind {
        ExecutionEventKind::NodeStarted | ExecutionEventKind::NodeRetired => {
            (node_span, frame_span)
        }
        ExecutionEventKind::OperationSubmitted => (operation_span, node_span),
        _ => panic!("invalid node event kind"),
    };
    let mut parts = bind_active(
        bind_plan(
            base_parts(run, request, sequence, span.as_str(), Some(parent)),
            plan,
        ),
        active,
    );
    parts.frame_id = Some(ExecutionFrameId::try_from(frame).unwrap());
    parts.node_invocation_id = Some(NodeInvocationId::try_from(invocation).unwrap());
    parts.node_id = Some(node.id().clone());
    parts.operation_id = Some(node.operation_id().clone());
    parts.provider_id = Some(node.selection().selected_provider().clone());
    make_event(
        sequence,
        ExecutionPhase::Execution,
        kind,
        parts,
        ExecutionEventDetail::None,
    )
}

pub(crate) fn request_completed_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    completed: &TrustedCompletedSequenceBinding,
    run: &RunId,
    request: &RequestIdentity,
    sequence: u64,
) -> ExecutionEvent {
    let mut parts = bind_active(
        bind_plan(
            base_parts(run, request, sequence, "span.request", None),
            plan,
        ),
        active,
    );
    parts.completed_sequence_fingerprint = Some(completed.fingerprint().to_owned());
    make_event(
        sequence,
        ExecutionPhase::Completion,
        ExecutionEventKind::RequestCompleted,
        parts,
        ExecutionEventDetail::Counters {
            input: 11,
            output: 7,
        },
    )
}

pub(crate) fn sequence_completed_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    completed: &TrustedCompletedSequenceBinding,
    sequence: u64,
) -> ExecutionEvent {
    let mut parts = bind_active(
        bind_plan(
            base_parts(
                active.run_id(),
                active.request_id(),
                sequence,
                "span.sequence-completed",
                Some(id("span.request")),
            ),
            plan,
        ),
        active,
    );
    parts.completed_sequence_fingerprint = Some(completed.fingerprint().to_owned());
    make_event(
        sequence,
        ExecutionPhase::Completion,
        ExecutionEventKind::SequenceCompleted,
        parts,
        ExecutionEventDetail::None,
    )
}

pub(crate) fn sequence_aborted_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    aborted: &TrustedAbortedSequenceBinding,
    sequence: u64,
) -> ExecutionEvent {
    let mut parts = bind_active(
        bind_plan(
            base_parts(
                active.run_id(),
                active.request_id(),
                sequence,
                "span.sequence-aborted",
                Some(id("span.request")),
            ),
            plan,
        ),
        active,
    );
    parts.aborted_sequence_fingerprint = Some(aborted.fingerprint().to_owned());
    make_event(
        sequence,
        ExecutionPhase::Completion,
        ExecutionEventKind::SequenceAborted,
        parts,
        ExecutionEventDetail::None,
    )
}

pub(crate) fn operation_failure_event(
    failure: &IdentifiedFailure,
    sequence: u64,
) -> ExecutionEvent {
    let failed_operation = failure.identity().parts();
    let mut observation = failed_operation.clone();
    observation.sequence = sequence;
    observation.span_id = id(format!("{}.failure-observed", failed_operation.span_id));
    observation.parent_span_id = Some(failed_operation.span_id.clone());
    let identity = ExecutionIdentityEnvelope::new(observation).unwrap();
    ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: sequence * 10,
        },
        ExecutionPhase::Execution,
        ExecutionEventKind::FailureObserved,
        identity,
        ExecutionEventDetail::Failure(failure.clone()),
    )
    .unwrap()
}

pub(crate) fn planning_failure_event(
    plan: &ExecutionPlan,
    run: &RunId,
    request: &RequestIdentity,
    sequence: u64,
) -> (ExecutionEvent, IdentifiedFailure) {
    let identity = ExecutionIdentityEnvelope::new(bind_plan(
        base_parts(
            run,
            request,
            sequence,
            "span.planning-failure",
            Some(id("span.request")),
        ),
        plan,
    ))
    .unwrap();
    let failure = IdentifiedFailure::new(
        identity.clone(),
        FailureEnvelope::new(
            FailureDomain::Planning,
            "planning_fixture_failure",
            "injected planning failure",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let event = ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: sequence * 10,
        },
        ExecutionPhase::Planning,
        ExecutionEventKind::FailureObserved,
        identity,
        ExecutionEventDetail::Failure(failure.clone()),
    )
    .unwrap();
    (event, failure)
}

pub(crate) fn request_failed_terminal_event(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    completed: Option<&TrustedCompletedSequenceBinding>,
    aborted: Option<&TrustedAbortedSequenceBinding>,
    failure: &IdentifiedFailure,
    sequence: u64,
) -> ExecutionEvent {
    assert!(completed.is_some() ^ aborted.is_some());
    let mut parts = bind_active(
        bind_plan(
            base_parts(
                active.run_id(),
                active.request_id(),
                sequence,
                "span.request",
                None,
            ),
            plan,
        ),
        active,
    );
    parts.completed_sequence_fingerprint =
        completed.map(|binding| binding.fingerprint().to_owned());
    parts.aborted_sequence_fingerprint = aborted.map(|binding| binding.fingerprint().to_owned());
    make_event(
        sequence,
        ExecutionPhase::Completion,
        ExecutionEventKind::RequestFailed,
        parts,
        ExecutionEventDetail::FailureTerminal {
            first_failure_fingerprint: failure.fingerprint(),
        },
    )
}

pub(crate) fn request_journal(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    completed: &TrustedCompletedSequenceBinding,
    submissions: &[SubmittedOperationReceipt],
    completions: &[OperationCompletionReceipt],
    frames: u64,
) -> Vec<ExecutionEvent> {
    let run = active.run_id();
    let request = active.request_id();
    let mut events = vec![accepted_event(run, request), plan_event(plan, run, request)];
    let mut sequence = 3_u64;
    let mut invocation = 1_u64;
    let mut submission_index = 0_usize;
    let mut completion_index = 0_usize;
    for frame in 1..=frames {
        events.push(frame_event(
            plan,
            active,
            run,
            request,
            sequence,
            frame,
            ExecutionEventKind::FrameStarted,
        ));
        sequence += 1;
        for node_index in 0..plan.payload().nodes().len() {
            for kind in [
                ExecutionEventKind::NodeStarted,
                ExecutionEventKind::OperationSubmitted,
                ExecutionEventKind::NodeRetired,
            ] {
                let event = node_event(
                    plan, active, run, request, sequence, frame, invocation, node_index, kind,
                );
                if kind == ExecutionEventKind::OperationSubmitted {
                    assert_eq!(
                        submissions[submission_index].participants()[0].identity(),
                        event.identity()
                    );
                    submission_index += 1;
                } else if kind == ExecutionEventKind::NodeRetired {
                    assert!(same_operation_authority(
                        event.identity(),
                        completions[completion_index].participants()[0]
                            .submission()
                            .identity(),
                    ));
                    completion_index += 1;
                }
                events.push(event);
                sequence += 1;
            }
            invocation += 1;
        }
        events.push(frame_event(
            plan,
            active,
            run,
            request,
            sequence,
            frame,
            ExecutionEventKind::FrameCompleted,
        ));
        sequence += 1;
    }
    assert_eq!(submission_index, submissions.len());
    assert_eq!(completion_index, completions.len());
    events.push(sequence_completed_event(plan, active, completed, sequence));
    sequence += 1;
    events.push(request_completed_event(
        plan, active, completed, run, request, sequence,
    ));
    events
}

pub(crate) fn same_operation_authority(
    observation: &ExecutionIdentityEnvelope,
    operation: &ExecutionIdentityEnvelope,
) -> bool {
    let mut normalized = observation.parts().clone();
    normalized.sequence = operation.parts().sequence;
    normalized.span_id = operation.parts().span_id.clone();
    normalized.parent_span_id = operation.parts().parent_span_id.clone();
    normalized == *operation.parts()
}

pub(crate) fn event_context<'a>(
    event: &'a ExecutionEvent,
    topology: &'a TrustedExecutionTopology,
    active: &'a TrustedActiveSequenceBinding,
    completed: &'a TrustedCompletedSequenceBinding,
    submissions: &'a [SubmittedOperationReceipt],
    completions: &'a [OperationCompletionReceipt],
) -> TrustedExecutionEventContext<'a> {
    match event.kind() {
        ExecutionEventKind::RequestAccepted => {
            TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id())
        }
        ExecutionEventKind::PlanBuilt => {
            TrustedExecutionEventContext::bound(active.run_id(), active.request_id(), topology)
        }
        ExecutionEventKind::RequestFailed => {
            let failure = match event.detail() {
                ExecutionEventDetail::Failure(failure) => failure,
                _ => unreachable!(),
            };
            TrustedExecutionEventContext::failure(
                active.run_id(),
                active.request_id(),
                Some(topology),
                event
                    .identity()
                    .parts()
                    .active_sequence_slot
                    .is_some()
                    .then_some(active),
                failure,
            )
        }
        ExecutionEventKind::OperationSubmitted => {
            let receipt = submissions
                .iter()
                .find(|receipt| {
                    receipt
                        .participants()
                        .iter()
                        .any(|participant| participant.identity() == event.identity())
                })
                .expect("journal operation has external submission receipt");
            TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                topology,
                active,
                receipt,
            )
        }
        ExecutionEventKind::NodeRetired => {
            let completion = completions
                .iter()
                .flat_map(|receipt| receipt.participants())
                .find(|participant| {
                    same_operation_authority(event.identity(), participant.submission().identity())
                })
                .expect("retired node has external participant completion evidence");
            TrustedExecutionEventContext::node_retired(
                active.run_id(),
                active.request_id(),
                topology,
                active,
                completion,
            )
        }
        ExecutionEventKind::SequenceCompleted | ExecutionEventKind::RequestCompleted => {
            TrustedExecutionEventContext::completed(
                active.run_id(),
                active.request_id(),
                topology,
                active,
                completed,
            )
        }
        _ => TrustedExecutionEventContext::active(
            active.run_id(),
            active.request_id(),
            topology,
            active,
        ),
    }
}

pub(crate) fn observe_journal(
    journal: &[ExecutionEvent],
    topology: &TrustedExecutionTopology,
    active: &TrustedActiveSequenceBinding,
    completed: &TrustedCompletedSequenceBinding,
    submissions: &[SubmittedOperationReceipt],
    completions: &[OperationCompletionReceipt],
) -> Result<ExecutionEventCursor, VNextError> {
    let mut cursor =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in journal {
        cursor.observe_against(
            event,
            &event_context(event, topology, active, completed, submissions, completions),
        )?;
    }
    Ok(cursor)
}

pub(crate) fn observe_failure_journal(
    evidence: &FailureSequenceEvidence,
    topology: &TrustedExecutionTopology,
) -> Result<ExecutionEventCursor, VNextError> {
    let mut cursor = ExecutionEventCursor::new(
        evidence.active.run_id().clone(),
        evidence.active.request_id().clone(),
    );
    let mut first_failure: Option<IdentifiedFailure> = None;
    for event in &evidence.journal {
        let context = match event.kind() {
            ExecutionEventKind::RequestAccepted => TrustedExecutionEventContext::pre_plan(
                evidence.active.run_id(),
                evidence.active.request_id(),
            ),
            ExecutionEventKind::PlanBuilt => TrustedExecutionEventContext::bound(
                evidence.active.run_id(),
                evidence.active.request_id(),
                topology,
            ),
            ExecutionEventKind::OperationSubmitted => {
                let submission = evidence
                    .submissions
                    .iter()
                    .find(|receipt| {
                        receipt
                            .participants()
                            .iter()
                            .any(|participant| participant.identity() == event.identity())
                    })
                    .unwrap();
                TrustedExecutionEventContext::operation_submitted(
                    evidence.active.run_id(),
                    evidence.active.request_id(),
                    topology,
                    &evidence.active,
                    submission,
                )
            }
            ExecutionEventKind::NodeRetired => {
                let completion = evidence
                    .completions
                    .iter()
                    .flat_map(|receipt| receipt.participants())
                    .find(|participant| {
                        same_operation_authority(
                            event.identity(),
                            participant.submission().identity(),
                        )
                    })
                    .unwrap();
                TrustedExecutionEventContext::node_retired(
                    evidence.active.run_id(),
                    evidence.active.request_id(),
                    topology,
                    &evidence.active,
                    completion,
                )
            }
            ExecutionEventKind::FailureObserved => {
                let failure = match event.detail() {
                    ExecutionEventDetail::Failure(failure) => failure,
                    _ => unreachable!(),
                };
                TrustedExecutionEventContext::failure(
                    evidence.active.run_id(),
                    evidence.active.request_id(),
                    Some(topology),
                    Some(&evidence.active),
                    failure,
                )
            }
            ExecutionEventKind::SequenceCompleted => TrustedExecutionEventContext::completed(
                evidence.active.run_id(),
                evidence.active.request_id(),
                topology,
                &evidence.active,
                evidence.completed.as_ref().unwrap(),
            ),
            ExecutionEventKind::SequenceAborted => TrustedExecutionEventContext::aborted(
                evidence.active.run_id(),
                evidence.active.request_id(),
                topology,
                &evidence.active,
                evidence.aborted.as_ref().unwrap(),
            ),
            ExecutionEventKind::RequestFailed => {
                TrustedExecutionEventContext::failure_with_disposition(
                    evidence.active.run_id(),
                    evidence.active.request_id(),
                    topology,
                    &evidence.active,
                    evidence.completed.as_ref(),
                    evidence.aborted.as_ref(),
                    first_failure.as_ref().unwrap(),
                )
            }
            _ => TrustedExecutionEventContext::active(
                evidence.active.run_id(),
                evidence.active.request_id(),
                topology,
                &evidence.active,
            ),
        };
        cursor.observe_against(event, &context)?;
        if event.kind() == ExecutionEventKind::FailureObserved {
            let ExecutionEventDetail::Failure(failure) = event.detail() else {
                unreachable!()
            };
            first_failure = Some(failure.clone());
        }
    }
    Ok(cursor)
}
