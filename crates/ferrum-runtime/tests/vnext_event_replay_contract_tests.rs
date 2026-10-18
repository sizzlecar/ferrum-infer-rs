mod vnext_event_contract;

use vnext_event_contract::*;

#[test]
fn vnext_event_replay_contract() {
    const EXPECTED_CASES: usize = 47;
    let mut passed = 0_usize;
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = execution_plan("replay", &operation_registry);
    let topology = TrustedExecutionTopology::from_plan(&plan).unwrap();
    let resolved = resolved_model_plan(&plan, "replay", &operation_registry);
    let ProvisionedRuntimePool {
        resources: plan_resources,
        runtime: plan_runtime,
        evidence: pool_evidence,
        journal: pool_journal,
        committed_snapshot: _,
    } = provision_runtime_pool(&plan, &topology, "replay");
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
        "run.replay.one",
        "request.replay.one",
        2,
    );
    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 2);
    let SequenceEvidence {
        active: _,
        completed: completed_two,
        submissions: _,
        completions: completions_two,
    } = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.replay.two",
        "request.replay.two",
        1,
    );
    let completed_failure = execute_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.replay.failed",
        "request.replay.failed",
        true,
    );
    let completed_failure_first = match completed_failure.journal[5].detail() {
        ExecutionEventDetail::Failure(failure) => failure,
        _ => unreachable!(),
    };
    let succeeded_completion_failure = execute_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.replay.synthetic",
        "request.replay.synthetic",
        false,
    );
    plan_runtime.fail_next_fence();
    let failed_completion_success = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.replay.failed-completion",
        "request.replay.failed-completion",
        3,
    );
    let failed_completion_success_journal = request_journal(
        &plan,
        &failed_completion_success.active,
        &failed_completion_success.completed,
        &failed_completion_success.submissions,
        &failed_completion_success.completions,
        3,
    );
    let contract_terminal_failure = execute_terminal_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.replay.contract-terminal",
        "request.replay.contract-terminal",
        ReplayTerminalFixtureMode::ContractFailed,
    );
    let drained_terminal_failure = execute_terminal_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.replay.drained-terminal",
        "request.replay.drained-terminal",
        ReplayTerminalFixtureMode::Drained,
    );
    let quarantined_terminal_failure = execute_terminal_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.replay.quarantined-terminal",
        "request.replay.quarantined-terminal",
        ReplayTerminalFixtureMode::Quarantined,
    );
    let no_fence_drained_terminal_failure = execute_terminal_failure_then_complete(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.replay.no-fence-terminal",
        "request.replay.no-fence-terminal",
        ReplayTerminalFixtureMode::SubmissionIndeterminateDrained,
    );
    let ProvisionedRuntimePool {
        resources: abort_plan_resources,
        runtime: abort_runtime,
        evidence: abort_pool_evidence,
        journal: abort_pool_prefix,
        committed_snapshot: _,
    } = provision_runtime_pool(&plan, &topology, "replay-abort");
    let (aborted_failure, abort_pool_evidence, abort_pool_journal) = execute_failure_then_abort(
        &abort_plan_resources,
        &abort_runtime,
        abort_pool_prefix,
        abort_pool_evidence,
        &resolved,
        &operation_registry,
        "run.replay.aborted",
        "request.replay.aborted",
    );
    let close_receipt = close_plan_runtime(plan_resources);
    let abort_close_receipt = close_plan_runtime(abort_plan_resources);

    let replay_evidence = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    let replay = ReplayIdentity::from_evidence(&replay_evidence).unwrap();
    check(
        &mut passed,
        replay.terminal_identity() == journal.last().unwrap().identity()
            && replay.random_seed() == 9271,
    );
    let unvalidated =
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&replay).unwrap()).unwrap();
    check(
        &mut passed,
        unvalidated.revalidate(&replay_evidence).unwrap() == replay,
    );
    check(
        &mut passed,
        matches!(
            completed_failure.completions[0].disposition(),
            OperationCompletionDisposition::FailedButQuiescent(failures)
                if failures.as_slice() == std::slice::from_ref(completed_failure_first)
        ),
    );
    check(
        &mut passed,
        matches!(
            succeeded_completion_failure.completions[0].disposition(),
            OperationCompletionDisposition::Succeeded
        ),
    );
    check(
        &mut passed,
        matches!(
            failed_completion_success.completions[0].disposition(),
            OperationCompletionDisposition::FailedButQuiescent(_)
        ) && failed_completion_success.completions[1..]
            .iter()
            .all(|receipt| {
                matches!(
                    receipt.disposition(),
                    OperationCompletionDisposition::Succeeded
                )
            }),
    );

    let missing_operation_terminal = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions[..completions.len() - 1],
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&missing_operation_terminal).is_err(),
    );
    let mut duplicated_completions = completions.clone();
    duplicated_completions.push(completions[0].clone());
    let duplicate_operation_terminal = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &duplicated_completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&duplicate_operation_terminal).is_err(),
    );
    let mut unused_completions = completions.clone();
    unused_completions.push(failed_completion_success.completions[4].clone());
    let unused_operation_terminal = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &unused_completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&unused_operation_terminal).is_err(),
    );
    let succeeded_but_failed = ReplayEvidence::new(
        &resolved,
        b"successful fence reported failed",
        b"reversal state",
        9271,
        &succeeded_completion_failure.journal,
        &succeeded_completion_failure.active,
        succeeded_completion_failure.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &succeeded_completion_failure.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&succeeded_but_failed).is_err(),
    );
    let failed_but_succeeded = ReplayEvidence::new(
        &resolved,
        b"failed fence reported successful",
        b"reverse state",
        9271,
        &failed_completion_success_journal,
        &failed_completion_success.active,
        Some(&failed_completion_success.completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &failed_completion_success.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&failed_but_succeeded).is_err(),
    );

    let mismatched_failure = IdentifiedFailure::new(
        completed_failure_first.identity().clone(),
        FailureEnvelope::new(
            FailureDomain::Device,
            "different_terminal_failure",
            "journal failure differs from failed fence",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let mut mismatched_failure_journal = completed_failure.journal.clone();
    let observed_failure_event = &completed_failure.journal[5];
    mismatched_failure_journal[5] = ExecutionEvent::new(
        observed_failure_event.timestamp(),
        observed_failure_event.phase(),
        ExecutionEventKind::FailureObserved,
        observed_failure_event.identity().clone(),
        ExecutionEventDetail::Failure(mismatched_failure.clone()),
    )
    .unwrap();
    let terminal_failure_event = completed_failure.journal.last().unwrap();
    *mismatched_failure_journal.last_mut().unwrap() = ExecutionEvent::new(
        terminal_failure_event.timestamp(),
        terminal_failure_event.phase(),
        ExecutionEventKind::RequestFailed,
        terminal_failure_event.identity().clone(),
        ExecutionEventDetail::FailureTerminal {
            first_failure_fingerprint: mismatched_failure.fingerprint(),
        },
    )
    .unwrap();
    let mismatched_failed_completion = ReplayEvidence::new(
        &resolved,
        b"mismatched failed completion",
        b"mismatch state",
        9271,
        &mismatched_failure_journal,
        &completed_failure.active,
        completed_failure.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completed_failure.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&mismatched_failed_completion).is_err(),
    );

    let contract_terminal_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"contract terminal request",
        b"contract terminal state",
        9271,
        &contract_terminal_failure.sequence.journal,
        &contract_terminal_failure.sequence.active,
        contract_terminal_failure.sequence.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &contract_terminal_failure.sequence.completions,
        &contract_terminal_failure.drains,
        &contract_terminal_failure.quarantines,
        &pool_evidence,
        &pool_journal,
    );
    let contract_terminal_replay =
        ReplayIdentity::from_evidence(&contract_terminal_replay_evidence).unwrap();
    check(
        &mut passed,
        contract_terminal_replay.cleanup_status() == ReplayCleanupStatus::Completed,
    );
    let drained_terminal_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"drained terminal request",
        b"drained terminal state",
        9271,
        &drained_terminal_failure.sequence.journal,
        &drained_terminal_failure.sequence.active,
        drained_terminal_failure.sequence.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &drained_terminal_failure.sequence.completions,
        &drained_terminal_failure.drains,
        &drained_terminal_failure.quarantines,
        &pool_evidence,
        &pool_journal,
    );
    let drained_terminal_replay =
        ReplayIdentity::from_evidence(&drained_terminal_replay_evidence).unwrap();
    check(
        &mut passed,
        drained_terminal_replay.cleanup_status() == ReplayCleanupStatus::Completed,
    );
    let quarantined_terminal_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"quarantined terminal request",
        b"quarantined terminal state",
        9271,
        &quarantined_terminal_failure.sequence.journal,
        &quarantined_terminal_failure.sequence.active,
        quarantined_terminal_failure.sequence.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &quarantined_terminal_failure.sequence.completions,
        &quarantined_terminal_failure.drains,
        &quarantined_terminal_failure.quarantines,
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&quarantined_terminal_replay_evidence).is_err(),
    );
    let no_fence_terminal_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"no-fence terminal request",
        b"no-fence terminal state",
        9271,
        &no_fence_drained_terminal_failure.sequence.journal,
        &no_fence_drained_terminal_failure.sequence.active,
        no_fence_drained_terminal_failure
            .sequence
            .completed
            .as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &no_fence_drained_terminal_failure.sequence.completions,
        &no_fence_drained_terminal_failure.drains,
        &no_fence_drained_terminal_failure.quarantines,
        &pool_evidence,
        &pool_journal,
    );
    let no_fence_terminal_replay =
        ReplayIdentity::from_evidence(&no_fence_terminal_replay_evidence).unwrap();
    check(
        &mut passed,
        no_fence_terminal_replay.cleanup_status() == ReplayCleanupStatus::Completed,
    );
    check(
        &mut passed,
        no_fence_drained_terminal_failure
            .sequence
            .submissions
            .is_empty()
            && no_fence_drained_terminal_failure
                .sequence
                .completions
                .is_empty()
            && no_fence_drained_terminal_failure.drains.len() == 1
            && !no_fence_drained_terminal_failure.drains[0].had_submission_fence()
            && no_fence_drained_terminal_failure
                .sequence
                .journal
                .iter()
                .all(|event| event.kind() != ExecutionEventKind::OperationSubmitted),
    );
    let cross_type_slot_collision = ReplayEvidence::new(
        &resolved,
        b"cross-type slot collision",
        b"cross-type state",
        9271,
        &drained_terminal_failure.sequence.journal,
        &drained_terminal_failure.sequence.active,
        drained_terminal_failure.sequence.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &contract_terminal_failure.sequence.completions,
        &drained_terminal_failure.drains,
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&cross_type_slot_collision).is_err(),
    );
    let mut terminal_fingerprint_mutation = serde_json::to_value(&drained_terminal_replay).unwrap();
    terminal_fingerprint_mutation["operation_terminal_evidence_fingerprint"] = json!(sha('0'));
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(
            &serde_json::to_vec(&terminal_fingerprint_mutation).unwrap(),
        )
        .unwrap()
        .revalidate(&drained_terminal_replay_evidence)
        .is_err(),
    );
    let mut terminal_count_mutation = serde_json::to_value(&drained_terminal_replay).unwrap();
    terminal_count_mutation["operation_terminal_evidence_count"] = json!(2);
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&terminal_count_mutation).unwrap())
            .unwrap()
            .revalidate(&drained_terminal_replay_evidence)
            .is_err(),
    );
    let wrong_input = ReplayEvidence::new(
        &resolved,
        b"tampered request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&replay).unwrap())
            .unwrap()
            .revalidate(&wrong_input)
            .is_err(),
    );
    let wrong_state = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"tampered state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&replay).unwrap())
            .unwrap()
            .revalidate(&wrong_state)
            .is_err(),
    );
    let wrong_seed = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9272,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&replay).unwrap())
            .unwrap()
            .revalidate(&wrong_seed)
            .is_err(),
    );
    let truncated = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal[..journal.len() - 1],
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&truncated).is_err(),
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&vec![b' '; MAX_REPLAY_IDENTITY_WIRE_BYTES + 1]).is_err(),
    );
    let other_runtime_catalog = catalog();
    let other_operation_registry = make_operation_registry(&other_runtime_catalog);
    let other_plan = execution_plan("v4-other", &other_operation_registry);
    let other_resolved = resolved_model_plan(&other_plan, "v4-other", &other_operation_registry);
    let wrong_plan = ReplayEvidence::new(
        &other_resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&wrong_plan).is_err(),
    );
    let missing_completion = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        None,
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&missing_completion).is_err(),
    );
    let wrong_completion = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed_two),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&wrong_completion).is_err(),
    );
    let wrong_submissions = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions_two,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&wrong_submissions).is_err(),
    );
    let truncated_pool = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal[..2],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&truncated_pool).is_err(),
    );
    let reordered_pool_journal = vec![
        pool_journal[0].clone(),
        pool_journal[2].clone(),
        pool_journal[1].clone(),
    ];
    let reordered_pool = ReplayEvidence::new(
        &resolved,
        b"request bytes",
        b"initial state bytes",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
        &pool_evidence,
        &reordered_pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&reordered_pool).is_err(),
    );
    for field in [
        "resolved_plan_fingerprint",
        "request_journal_fingerprint",
        "active_sequence_fingerprint",
        "pool_journal_fingerprint",
    ] {
        let mut value = serde_json::to_value(&replay).unwrap();
        value[field] = json!(sha('0'));
        check(
            &mut passed,
            ReplayIdentity::decode_untrusted(&serde_json::to_vec(&value).unwrap())
                .unwrap()
                .revalidate(&replay_evidence)
                .is_err(),
        );
    }

    let completed_failure_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"failed completion request",
        b"failed completion state",
        9271,
        &completed_failure.journal,
        &completed_failure.active,
        completed_failure.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completed_failure.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    let completed_failure_replay =
        ReplayIdentity::from_evidence(&completed_failure_replay_evidence).unwrap();
    check(
        &mut passed,
        completed_failure_replay.cleanup_status() == ReplayCleanupStatus::Completed,
    );

    let aborted_replay_evidence = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&abort_close_receipt),
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal,
    );
    let aborted_replay = ReplayIdentity::from_evidence(&aborted_replay_evidence).unwrap();
    check(
        &mut passed,
        aborted_replay.cleanup_status() == ReplayCleanupStatus::SequenceQuiescent,
    );
    check(
        &mut passed,
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&aborted_replay).unwrap())
            .unwrap()
            .revalidate(&aborted_replay_evidence)
            .unwrap()
            == aborted_replay,
    );

    let pending_abort_evidence = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::AllowPending,
        ReplayPlanCleanupEvidence::Pending,
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal[..3],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&pending_abort_evidence)
            .unwrap()
            .cleanup_status()
            == ReplayCleanupStatus::CleanupPending,
    );
    let open_pool_clean_abort = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Pending,
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal[..3],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&open_pool_clean_abort).is_err(),
    );
    let quarantined_but_open_abort = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&abort_close_receipt),
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal[..abort_pool_journal.len() - 1],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&quarantined_but_open_abort).is_err(),
    );
    let incomplete_pool_pending = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::AllowPending,
        ReplayPlanCleanupEvidence::Pending,
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal[..abort_pool_journal.len() - 1],
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&incomplete_pool_pending).is_err(),
    );
    let wrong_abort_pool = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::AllowPending,
        ReplayPlanCleanupEvidence::Pending,
        &aborted_failure.completions,
        &[],
        &[],
        &pool_evidence,
        &pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&wrong_abort_pool).is_err(),
    );
    let reused_completion_receipt = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        completed_failure.completed.as_ref(),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&abort_close_receipt),
        &aborted_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&reused_completion_receipt).is_err(),
    );
    let reused_submission_receipt = ReplayEvidence::new(
        &resolved,
        b"aborted request",
        b"aborted state",
        9271,
        &aborted_failure.journal,
        &aborted_failure.active,
        None,
        aborted_failure.aborted.as_ref(),
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&abort_close_receipt),
        &completed_failure.completions,
        &[],
        &[],
        &abort_pool_evidence,
        &abort_pool_journal,
    );
    check(
        &mut passed,
        ReplayIdentity::from_evidence(&reused_submission_receipt).is_err(),
    );
    for field in [
        "aborted_sequence_fingerprint",
        "cleanup_status",
        "resource_pool_identity_fingerprint",
    ] {
        let mut wire = serde_json::to_value(&aborted_replay).unwrap();
        wire[field] = if field == "cleanup_status" {
            json!("completed")
        } else {
            json!(sha('0'))
        };
        check(
            &mut passed,
            ReplayIdentity::decode_untrusted(&serde_json::to_vec(&wire).unwrap())
                .unwrap()
                .revalidate(&aborted_replay_evidence)
                .is_err(),
        );
    }

    no_static_replay_requires_explicit_root_cleanup();
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT EVENT REPLAY PASS: {passed}/{EXPECTED_CASES}");
}

fn no_static_replay_requires_explicit_root_cleanup() {
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = no_static_execution_plan("no-static-replay", &operation_registry);
    assert!(plan.payload().memory().static_allocations().is_empty());
    let resolved = no_static_resolved_model_plan(&plan, "no-static-replay", &operation_registry);
    let (resources, runtime) = provision_no_static_plan_runtime(&plan);
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
        "run.no-static-replay",
        "request.no-static-replay",
        1,
    );
    assert!(active.static_pool_id().is_none());
    assert!(active.static_provisioning_identity().is_none());
    assert!(active.static_entries().is_empty());
    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 1);

    let missing_cleanup = ReplayEvidence::new_no_static(
        &resolved,
        b"no-static request",
        b"no-static initial state",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Pending,
        &completions,
        &[],
        &[],
    );
    assert!(ReplayIdentity::from_evidence(&missing_cleanup).is_err());

    let pending = ReplayEvidence::new_no_static(
        &resolved,
        b"no-static request",
        b"no-static initial state",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::AllowPending,
        ReplayPlanCleanupEvidence::Pending,
        &completions,
        &[],
        &[],
    );
    let pending_identity = ReplayIdentity::from_evidence(&pending).unwrap();
    assert_eq!(
        pending_identity.cleanup_status(),
        ReplayCleanupStatus::CleanupPending
    );
    assert!(pending_identity.pool_journal_fingerprint().is_none());
    assert!(pending_identity.plan_cleanup_fingerprint().is_none());

    let close_receipt = close_plan_runtime(resources);
    assert_eq!(close_receipt.released_static_resources(), 0);
    assert!(close_receipt.evidence().static_pool_identity().is_none());
    let clean = ReplayEvidence::new_no_static(
        &resolved,
        b"no-static request",
        b"no-static initial state",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&close_receipt),
        &completions,
        &[],
        &[],
    );
    let clean_identity = ReplayIdentity::from_evidence(&clean).unwrap();
    assert_eq!(
        clean_identity.cleanup_status(),
        ReplayCleanupStatus::Completed
    );
    assert!(clean_identity.pool_journal_fingerprint().is_none());
    assert!(clean_identity.plan_cleanup_fingerprint().is_some());
    assert_eq!(
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&clean_identity).unwrap())
            .unwrap()
            .revalidate(&clean)
            .unwrap(),
        clean_identity
    );
    let mut tampered_cleanup = serde_json::to_value(&clean_identity).unwrap();
    tampered_cleanup["plan_cleanup_fingerprint"] = json!(sha('0'));
    assert!(
        ReplayIdentity::decode_untrusted(&serde_json::to_vec(&tampered_cleanup).unwrap())
            .unwrap()
            .revalidate(&clean)
            .is_err()
    );

    let (foreign_resources, _) = provision_no_static_plan_runtime(&plan);
    let foreign_close = close_plan_runtime(foreign_resources);
    let foreign_cleanup = ReplayEvidence::new_no_static(
        &resolved,
        b"no-static request",
        b"no-static initial state",
        9271,
        &journal,
        &active,
        Some(&completed),
        None,
        ReplayCleanupRequirement::RequireClean,
        ReplayPlanCleanupEvidence::Closed(&foreign_close),
        &completions,
        &[],
        &[],
    );
    assert!(ReplayIdentity::from_evidence(&foreign_cleanup).is_err());
}
