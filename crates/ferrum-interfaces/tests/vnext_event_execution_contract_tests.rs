mod vnext_event_contract;

use vnext_event_contract::*;

#[test]
fn vnext_event_execution_contract() {
    const EXPECTED_CASES: usize = 54;
    let mut passed = 0_usize;
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = execution_plan("v4", &operation_registry);
    let topology = TrustedExecutionTopology::from_plan(&plan).unwrap();
    check(
        &mut passed,
        plan.payload().nodes().len() == 2
            && plan.payload().nodes()[1].dependencies() == [id("node.first")],
    );
    check(&mut passed, ExecutionFrameId::try_from(0).is_err());
    check(&mut passed, NodeInvocationId::try_from(0).is_err());
    check(
        &mut passed,
        topology.device_runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    );

    let resolved = resolved_model_plan(&plan, "v4", &operation_registry);
    let ProvisionedRuntimePool {
        resources: plan_resources,
        runtime: plan_runtime,
        evidence: _,
        journal: _,
        committed_snapshot: _,
    } = provision_runtime_pool(&plan, &topology, "v4");
    let SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    } = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.request.one",
        "request.one",
        2,
    );
    check(
        &mut passed,
        active.runtime_implementation_fingerprint()
            == topology.device_runtime_implementation_fingerprint(),
    );
    check(
        &mut passed,
        !serde_json::to_string(&active)
            .unwrap()
            .contains("runtime_type"),
    );
    let SequenceEvidence {
        active: active_two,
        completed: completed_two,
        submissions: submissions_two,
        completions: _,
    } = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.request.two",
        "request.two",
        1,
    );

    let completed_failure = execute_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.failure.completed",
        "request.failure.completed",
        true,
    );
    check(
        &mut passed,
        observe_failure_journal(&completed_failure, &topology)
            .unwrap()
            .is_terminal(),
    );
    let completed_failure_first = match completed_failure.journal[5].detail() {
        ExecutionEventDetail::Failure(failure) => failure,
        _ => unreachable!(),
    };
    check(
        &mut passed,
        matches!(
            completed_failure.journal.last().unwrap().detail(),
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint
            } if first_failure_fingerprint == &completed_failure_first.fingerprint()
        ),
    );

    let ProvisionedRuntimePool {
        resources: abort_plan_resources,
        runtime: abort_runtime,
        evidence: abort_pool_evidence,
        journal: abort_pool_prefix,
        committed_snapshot: _,
    } = provision_runtime_pool(&plan, &topology, "abort-v5");
    let (aborted_failure, _, _) = execute_failure_then_abort(
        &abort_plan_resources,
        &abort_runtime,
        abort_pool_prefix,
        abort_pool_evidence,
        &resolved,
        &operation_registry,
        "run.failure.aborted",
        "request.failure.aborted",
    );
    close_plan_runtime(plan_resources);
    close_plan_runtime(abort_plan_resources);
    check(
        &mut passed,
        observe_failure_journal(&aborted_failure, &topology)
            .unwrap()
            .is_terminal(),
    );
    check(
        &mut passed,
        aborted_failure.aborted.as_ref().unwrap().disposition()
            == ActiveSequenceAbortDisposition::SequenceSessionTerminalized,
    );
    let aborted_failure_first = match aborted_failure.journal[5].detail() {
        ExecutionEventDetail::Failure(failure) => failure,
        _ => unreachable!(),
    };
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(
            &serde_json::to_vec(&completed_failure.journal[5]).unwrap(),
        )
        .unwrap()
        .revalidate(&TrustedExecutionEventContext::failure(
            completed_failure.active.run_id(),
            completed_failure.active.request_id(),
            Some(&topology),
            Some(&completed_failure.active),
            completed_failure_first,
        ))
        .unwrap()
            == completed_failure.journal[5],
    );
    let failure_wire = serde_json::to_value(&completed_failure.journal[5]).unwrap();
    let mut event_unknown_top = failure_wire.clone();
    event_unknown_top["unknown_top"] = json!(true);
    let mut event_unknown_identity = failure_wire.clone();
    event_unknown_identity["identity"]["unknown_identity"] = json!(true);
    let mut event_unknown_detail = failure_wire.clone();
    event_unknown_detail["detail"]["failure"]["failure"]["unknown_nested"] = json!(true);
    for wire in [
        event_unknown_top,
        event_unknown_identity,
        event_unknown_detail,
    ] {
        check(
            &mut passed,
            ExecutionEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap()).is_err(),
        );
    }
    let mut event_unknown_variant =
        serde_json::to_value(completed_failure.journal.last().unwrap()).unwrap();
    event_unknown_variant["detail"]["failure_terminal"]["extra"] = json!(true);
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(&event_unknown_variant).unwrap())
            .is_err(),
    );
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(
            &serde_json::to_vec(&completed_failure.journal[5]).unwrap(),
        )
        .unwrap()
        .revalidate(&TrustedExecutionEventContext::failure(
            completed_failure.active.run_id(),
            completed_failure.active.request_id(),
            Some(&topology),
            Some(&completed_failure.active),
            aborted_failure_first,
        ))
        .is_err(),
    );

    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 2);
    let cursor = observe_journal(
        &journal,
        &topology,
        &active,
        &completed,
        &submissions,
        &completions,
    )
    .unwrap();
    check(
        &mut passed,
        cursor.is_terminal() && cursor.completed_frames() == 2,
    );
    let invocations = journal
        .iter()
        .filter(|event| event.kind() == ExecutionEventKind::NodeStarted)
        .map(|event| {
            (
                event.identity().parts().frame_id.unwrap().get(),
                event.identity().parts().node_id.clone().unwrap(),
                event.identity().parts().node_invocation_id.unwrap().get(),
            )
        })
        .collect::<Vec<_>>();
    check(
        &mut passed,
        invocations.len() == 4
            && invocations[0].1 == invocations[2].1
            && invocations[0].2 == 1
            && invocations[2].2 == 3,
    );

    let mut frame_jump =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..2] {
        frame_jump
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let jump = frame_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        3,
        2,
        ExecutionEventKind::FrameStarted,
    );
    check(
        &mut passed,
        frame_jump
            .observe_against(
                &jump,
                &event_context(
                    &jump,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && frame_jump.last_sequence() == 2,
    );

    let mut incomplete =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..3] {
        incomplete
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let premature = frame_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        4,
        1,
        ExecutionEventKind::FrameCompleted,
    );
    check(
        &mut passed,
        incomplete
            .observe_against(
                &premature,
                &event_context(
                    &premature,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && incomplete.last_sequence() == 3,
    );

    let mut cross_frame =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..11] {
        cross_frame
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let dependency_from_prior_frame = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        12,
        2,
        3,
        1,
        ExecutionEventKind::NodeStarted,
    );
    check(
        &mut passed,
        cross_frame
            .observe_against(
                &dependency_from_prior_frame,
                &event_context(
                    &dependency_from_prior_frame,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err(),
    );

    let mut duplicate_invocation =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..6] {
        duplicate_invocation
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let duplicate = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        7,
        1,
        1,
        1,
        ExecutionEventKind::NodeStarted,
    );
    check(
        &mut passed,
        duplicate_invocation
            .observe_against(
                &duplicate,
                &event_context(
                    &duplicate,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && duplicate_invocation.last_sequence() == 6,
    );

    let mut invocation_gap =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..3] {
        invocation_gap
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let gap = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        4,
        1,
        2,
        0,
        ExecutionEventKind::NodeStarted,
    );
    check(
        &mut passed,
        invocation_gap
            .observe_against(
                &gap,
                &event_context(
                    &gap,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && invocation_gap.last_sequence() == 3,
    );

    let mut duplicate_node =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..6] {
        duplicate_node
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let repeated_completed_node = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        7,
        1,
        2,
        0,
        ExecutionEventKind::NodeStarted,
    );
    check(
        &mut passed,
        duplicate_node
            .observe_against(
                &repeated_completed_node,
                &event_context(
                    &repeated_completed_node,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && duplicate_node.last_sequence() == 6,
    );

    let node_started = &journal[3];
    let mut node_prefix =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..3] {
        node_prefix
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    for (field, replacement) in [
        ("resource_pool_id", json!(999_999_u64)),
        ("resource_pool_identity_fingerprint", json!(sha('0'))),
        (
            "activation_epoch",
            json!(active.activation_epoch().saturating_add(1)),
        ),
        ("runtime_implementation_fingerprint", json!(sha('0'))),
        ("active_sequence_fingerprint", json!(sha('0'))),
        ("frame_id", json!(2)),
        ("node_invocation_id", json!(2)),
    ] {
        let mut wire = serde_json::to_value(node_started).unwrap();
        wire["identity"][field] = replacement;
        let decoded =
            ExecutionEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap()).unwrap();
        let mut tampered_cursor = node_prefix.clone();
        let revalidated = decoded.revalidate(&TrustedExecutionEventContext::active(
            active.run_id(),
            active.request_id(),
            &topology,
            &active,
        ));
        let rejected = match revalidated {
            Err(_) => true,
            Ok(tampered) => tampered_cursor
                .observe_against(
                    &tampered,
                    &TrustedExecutionEventContext::active(
                        active.run_id(),
                        active.request_id(),
                        &topology,
                        &active,
                    ),
                )
                .is_err(),
        };
        let mut valid_cursor = node_prefix.clone();
        let rejected = rejected
            && tampered_cursor.last_sequence() == 3
            && valid_cursor
                .observe_against(
                    node_started,
                    &event_context(
                        node_started,
                        &topology,
                        &active,
                        &completed,
                        &submissions,
                        &completions,
                    ),
                )
                .is_ok();
        assert!(
            rejected,
            "wire tamper `{field}` was accepted or mutated cursor"
        );
        passed += 1;
    }
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(node_started).unwrap()).unwrap();
    let unchanged = node_prefix.clone();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::active(
                active_two.run_id(),
                active_two.request_id(),
                &topology,
                &active_two,
            ))
            .is_err()
            && unchanged.last_sequence() == 3,
    );

    let operation_submitted = &journal[4];
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(operation_submitted).unwrap())
            .unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &submissions[0],
            ))
            .unwrap()
            == *operation_submitted,
    );
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(operation_submitted).unwrap())
            .unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::operation_submitted(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &submissions_two[0],
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(&vec![b' '; MAX_EXECUTION_EVENT_WIRE_BYTES + 1]).is_err(),
    );
    let mut plan_wire = serde_json::to_value(&journal[1]).unwrap();
    plan_wire["identity"]["runtime_implementation_fingerprint"] = Value::Null;
    check(
        &mut passed,
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(&plan_wire).unwrap())
            .unwrap()
            .revalidate(&TrustedExecutionEventContext::bound(
                active.run_id(),
                active.request_id(),
                &topology,
            ))
            .is_err(),
    );
    let mut device_parts = base_parts(
        active.run_id(),
        active.request_id(),
        1,
        "span.device-failure",
        None,
    );
    device_parts.device_id = Some(plan.payload().device_id().clone());
    check(
        &mut passed,
        ExecutionIdentityEnvelope::new(device_parts.clone()).is_err(),
    );
    device_parts.runtime_implementation_fingerprint = Some(
        plan.payload()
            .device_runtime_implementation_fingerprint()
            .to_owned(),
    );
    let device_identity = ExecutionIdentityEnvelope::new(device_parts).unwrap();
    check(
        &mut passed,
        IdentifiedFailure::new(
            device_identity,
            FailureEnvelope::new(
                FailureDomain::Device,
                "device_failed",
                "device failure",
                false,
            )
            .unwrap(),
        )
        .is_ok(),
    );

    let mut missing_submission =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..4] {
        missing_submission
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let node_without_submission = node_event(
        &plan,
        &active,
        active.run_id(),
        active.request_id(),
        5,
        1,
        1,
        0,
        ExecutionEventKind::NodeRetired,
    );
    check(
        &mut passed,
        missing_submission
            .observe_against(
                &node_without_submission,
                &event_context(
                    &node_without_submission,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .is_err()
            && missing_submission.last_sequence() == 4,
    );

    let sequence_completed = &journal[journal.len() - 2];
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(sequence_completed).unwrap()).unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::completed(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &completed,
            ))
            .unwrap()
            == *sequence_completed,
    );
    let decoded =
        ExecutionEvent::decode_untrusted(&serde_json::to_vec(sequence_completed).unwrap()).unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::completed(
                active.run_id(),
                active.request_id(),
                &topology,
                &active,
                &completed_two,
            ))
            .is_err(),
    );
    let mut before_sync =
        ExecutionEventCursor::new(active.run_id().clone(), active.request_id().clone());
    for event in &journal[..journal.len() - 2] {
        before_sync
            .observe_against(
                event,
                &event_context(
                    event,
                    &topology,
                    &active,
                    &completed,
                    &submissions,
                    &completions,
                ),
            )
            .unwrap();
    }
    let premature_terminal = request_completed_event(
        &plan,
        &active,
        &completed,
        active.run_id(),
        active.request_id(),
        before_sync.last_sequence() + 1,
    );
    check(
        &mut passed,
        before_sync
            .observe_against(
                &premature_terminal,
                &TrustedExecutionEventContext::completed(
                    active.run_id(),
                    active.request_id(),
                    &topology,
                    &active,
                    &completed,
                ),
            )
            .is_err()
            && !before_sync.is_terminal(),
    );

    let mut no_failure_disposition = ExecutionEventCursor::new(
        completed_failure.active.run_id().clone(),
        completed_failure.active.request_id().clone(),
    );
    for event in &completed_failure.journal[..6] {
        let context = match event.kind() {
            ExecutionEventKind::RequestAccepted => TrustedExecutionEventContext::pre_plan(
                completed_failure.active.run_id(),
                completed_failure.active.request_id(),
            ),
            ExecutionEventKind::PlanBuilt => TrustedExecutionEventContext::bound(
                completed_failure.active.run_id(),
                completed_failure.active.request_id(),
                &topology,
            ),
            ExecutionEventKind::OperationSubmitted => {
                TrustedExecutionEventContext::operation_submitted(
                    completed_failure.active.run_id(),
                    completed_failure.active.request_id(),
                    &topology,
                    &completed_failure.active,
                    &completed_failure.submissions[0],
                )
            }
            ExecutionEventKind::FailureObserved => TrustedExecutionEventContext::failure(
                completed_failure.active.run_id(),
                completed_failure.active.request_id(),
                Some(&topology),
                Some(&completed_failure.active),
                completed_failure_first,
            ),
            _ => TrustedExecutionEventContext::active(
                completed_failure.active.run_id(),
                completed_failure.active.request_id(),
                &topology,
                &completed_failure.active,
            ),
        };
        no_failure_disposition
            .observe_against(event, &context)
            .unwrap();
    }
    let terminal_without_disposition = request_failed_terminal_event(
        &plan,
        &completed_failure.active,
        completed_failure.completed.as_ref(),
        None,
        completed_failure_first,
        7,
    );
    check(
        &mut passed,
        no_failure_disposition
            .observe_against(
                &terminal_without_disposition,
                &TrustedExecutionEventContext::failure_with_disposition(
                    completed_failure.active.run_id(),
                    completed_failure.active.request_id(),
                    &topology,
                    &completed_failure.active,
                    completed_failure.completed.as_ref(),
                    None,
                    completed_failure_first,
                ),
            )
            .is_err()
            && !no_failure_disposition.is_terminal(),
    );

    let completed_terminal = completed_failure.journal.last().unwrap();
    let wrong_terminal_fingerprint = ExecutionEvent::new(
        completed_terminal.timestamp(),
        completed_terminal.phase(),
        ExecutionEventKind::RequestFailed,
        completed_terminal.identity().clone(),
        ExecutionEventDetail::FailureTerminal {
            first_failure_fingerprint: sha('0'),
        },
    )
    .unwrap();
    let mut terminal_prefix = observe_failure_journal(
        &FailureSequenceEvidence {
            active: completed_failure.active.clone(),
            completed: completed_failure.completed.clone(),
            aborted: None,
            submissions: completed_failure.submissions.clone(),
            completions: completed_failure.completions.clone(),
            journal: completed_failure.journal[..7].to_vec(),
        },
        &topology,
    )
    .unwrap();
    check(
        &mut passed,
        terminal_prefix
            .observe_against(
                &wrong_terminal_fingerprint,
                &TrustedExecutionEventContext::failure_with_disposition(
                    completed_failure.active.run_id(),
                    completed_failure.active.request_id(),
                    &topology,
                    &completed_failure.active,
                    completed_failure.completed.as_ref(),
                    None,
                    completed_failure_first,
                ),
            )
            .is_err(),
    );
    check(
        &mut passed,
        ExecutionEvent::new(
            MonotonicTimestamp {
                nanos_since_run_start: 80,
            },
            ExecutionPhase::Completion,
            ExecutionEventKind::RequestFailed,
            completed_failure.journal[5].identity().clone(),
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint: completed_failure_first.fingerprint(),
            },
        )
        .is_err(),
    );

    let mut non_active_failure =
        ExecutionEventCursor::new(active_two.run_id().clone(), active_two.request_id().clone());
    non_active_failure
        .observe_against(
            &accepted_event(active_two.run_id(), active_two.request_id()),
            &TrustedExecutionEventContext::pre_plan(active_two.run_id(), active_two.request_id()),
        )
        .unwrap();
    non_active_failure
        .observe_against(
            &plan_event(&plan, active_two.run_id(), active_two.request_id()),
            &TrustedExecutionEventContext::bound(
                active_two.run_id(),
                active_two.request_id(),
                &topology,
            ),
        )
        .unwrap();
    let (planning_failure, planning_failure_evidence) =
        planning_failure_event(&plan, active_two.run_id(), active_two.request_id(), 3);
    non_active_failure
        .observe_against(
            &planning_failure,
            &TrustedExecutionEventContext::failure(
                active_two.run_id(),
                active_two.request_id(),
                Some(&topology),
                None,
                &planning_failure_evidence,
            ),
        )
        .unwrap();
    let foreign_disposition = sequence_completed_event(&plan, &active_two, &completed_two, 4);
    check(
        &mut passed,
        non_active_failure
            .observe_against(
                &foreign_disposition,
                &TrustedExecutionEventContext::completed(
                    active_two.run_id(),
                    active_two.request_id(),
                    &topology,
                    &active_two,
                    &completed_two,
                ),
            )
            .is_err(),
    );
    let mut non_active_abort = ExecutionEventCursor::new(
        aborted_failure.active.run_id().clone(),
        aborted_failure.active.request_id().clone(),
    );
    let abort_accepted = accepted_event(
        aborted_failure.active.run_id(),
        aborted_failure.active.request_id(),
    );
    non_active_abort
        .observe_against(
            &abort_accepted,
            &TrustedExecutionEventContext::pre_plan(
                aborted_failure.active.run_id(),
                aborted_failure.active.request_id(),
            ),
        )
        .unwrap();
    let abort_plan = plan_event(
        &plan,
        aborted_failure.active.run_id(),
        aborted_failure.active.request_id(),
    );
    non_active_abort
        .observe_against(
            &abort_plan,
            &TrustedExecutionEventContext::bound(
                aborted_failure.active.run_id(),
                aborted_failure.active.request_id(),
                &topology,
            ),
        )
        .unwrap();
    let (abort_planning_failure, abort_planning_evidence) = planning_failure_event(
        &plan,
        aborted_failure.active.run_id(),
        aborted_failure.active.request_id(),
        3,
    );
    non_active_abort
        .observe_against(
            &abort_planning_failure,
            &TrustedExecutionEventContext::failure(
                aborted_failure.active.run_id(),
                aborted_failure.active.request_id(),
                Some(&topology),
                None,
                &abort_planning_evidence,
            ),
        )
        .unwrap();
    let foreign_abort = sequence_aborted_event(
        &plan,
        &aborted_failure.active,
        aborted_failure.aborted.as_ref().unwrap(),
        4,
    );
    check(
        &mut passed,
        non_active_abort
            .observe_against(
                &foreign_abort,
                &TrustedExecutionEventContext::aborted(
                    aborted_failure.active.run_id(),
                    aborted_failure.active.request_id(),
                    &topology,
                    &aborted_failure.active,
                    aborted_failure.aborted.as_ref().unwrap(),
                ),
            )
            .is_err(),
    );

    let preplan_run: RunId = id("run.failure.preplan");
    let preplan_request: RequestIdentity = id("request.failure.preplan");
    let preplan_identity = ExecutionIdentityEnvelope::new(base_parts(
        &preplan_run,
        &preplan_request,
        1,
        "span.request",
        None,
    ))
    .unwrap();
    let preplan_failure = IdentifiedFailure::new(
        preplan_identity.clone(),
        FailureEnvelope::new(
            FailureDomain::ModelResolution,
            "preplan_failure",
            "preplan failure",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let preplan_terminal = ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: 10,
        },
        ExecutionPhase::Resolution,
        ExecutionEventKind::RequestFailed,
        preplan_identity,
        ExecutionEventDetail::Failure(preplan_failure.clone()),
    )
    .unwrap();
    let mut preplan_cursor =
        ExecutionEventCursor::new(preplan_run.clone(), preplan_request.clone());
    check(
        &mut passed,
        preplan_cursor
            .observe_against(
                &preplan_terminal,
                &TrustedExecutionEventContext::failure(
                    &preplan_run,
                    &preplan_request,
                    None,
                    None,
                    &preplan_failure,
                ),
            )
            .is_ok()
            && preplan_cursor.is_terminal(),
    );

    let accepted = &journal[0];
    let mut wrong_cursor =
        ExecutionEventCursor::new(id("run.cursor.wrong"), active.request_id().clone());
    check(
        &mut passed,
        wrong_cursor
            .observe_against(
                accepted,
                &TrustedExecutionEventContext::pre_plan(active.run_id(), active.request_id()),
            )
            .is_err()
            && wrong_cursor.last_sequence() == 0,
    );
    let decoded = ExecutionEvent::decode_untrusted(&serde_json::to_vec(accepted).unwrap()).unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::pre_plan(
                active.run_id(),
                active.request_id(),
            ))
            .unwrap()
            == *accepted,
    );
    let mut wire = serde_json::to_value(accepted).unwrap();
    wire["identity"]["sequence"] = json!(0);
    let decoded = ExecutionEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap()).unwrap();
    check(
        &mut passed,
        decoded
            .revalidate(&TrustedExecutionEventContext::pre_plan(
                active.run_id(),
                active.request_id(),
            ))
            .is_err(),
    );

    let model_failure = FailureEnvelope::new(
        FailureDomain::ModelResolution,
        "model_resolution_failed",
        "model resolution failed",
        false,
    )
    .unwrap();
    check(
        &mut passed,
        IdentifiedFailure::new(accepted.identity().clone(), model_failure).is_ok(),
    );
    let wrong_domain = FailureEnvelope::new(
        FailureDomain::Operation,
        "operation_failed",
        "operation failed",
        false,
    )
    .unwrap();
    check(
        &mut passed,
        IdentifiedFailure::new(accepted.identity().clone(), wrong_domain.clone()).is_err(),
    );
    check(
        &mut passed,
        IdentifiedFailure::new(journal[4].identity().clone(), wrong_domain).is_ok(),
    );
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT EVENT EXECUTION PASS: {passed}/{EXPECTED_CASES}");
}
