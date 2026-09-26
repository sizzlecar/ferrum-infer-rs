use super::*;

fn case() -> manifest::Cohort {
    let mut case = crate::commands::calibrate_slo::tests::manifest()
        .training
        .remove(0);
    case.wave_plan = Some(manifest::WavePlan {
        prefill_chunks: Some(
            [16, 32, 64, 128]
                .map(|n| NonZeroU32::new(n).unwrap())
                .to_vec(),
        ),
        decode_routes: Some(vec![
            CalibrationDecodeRoute::Actual,
            CalibrationDecodeRoute::FullLogits,
        ]),
    });
    case
}

fn prefill(count: u32) -> Row {
    Row {
        request_id: RequestId::new(),
        owner_incarnation: 1,
        work_generation: 3,
        work: Work::Prefill {
            offset: 0,
            count,
            total_prompt_tokens: count + 17,
        },
    }
}

fn decode() -> Row {
    Row {
        request_id: RequestId::new(),
        owner_incarnation: 2,
        work_generation: 4,
        work: Work::Decode { kv_tokens: 20 },
    }
}

fn evidence(attempt: &Attempt) -> (Vec<ReportedRow>, HostStageEvidenceV1) {
    let rows = attempt
        .rows
        .iter()
        .map(|row| ReportedRow {
            row: row.clone(),
            full_logits: true,
        })
        .collect();
    let host = HostStageEvidenceV1 {
        schema_version: 1,
        call_id: 1,
        presubmit_prediction: None,
        prospective_capture: None,
        fingerprint: None,
        actual_shape: None,
        statistical_evidence: None,
        structured_evidence: None,
        prepare_started_at_ns: Some(1),
        executor_returned_at_ns: Some(2),
        rows: attempt
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| HostRowStageV1 {
                request_id: row.request_id.clone(),
                owner_incarnation: row.owner_incarnation,
                work_generation: row.work_generation,
                input_index: i as u32,
                actual_work: match row.work {
                    Work::Decode { kv_tokens } => HostStageWork::Decode { kv_tokens },
                    Work::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    } => HostStageWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    },
                },
                host_processing_ordinal: Some(i as u32),
                host_started_at_ns: Some(3),
                token_committed_at_ns: Some(4),
                output_published_at_ns: Some(5),
                completion_started_at_ns: None,
                settled_at_ns: Some(6),
                terminal: None,
                completeness: HostStageCompleteness::CompleteSingleWave,
            })
            .collect(),
        finalized_at_ns: Some(7),
        full_wall_ns: None,
        completeness: HostStageCompleteness::CompleteSingleWave,
    };
    // No synthetic timing bound, canonical shape, cost sample or queue receipt
    // is invented: the cursor only consumes exact successful work correlation.
    (rows, host)
}

fn complete(cursor: &mut Cursor<'_>, rows: Vec<Row>) -> Attempt {
    let attempt = cursor.begin_rows(rows).unwrap();
    let (actual, host) = evidence(&attempt);
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &actual,
            Some(&host)
        )
        .unwrap());
    attempt
}

#[test]
fn wave_plan_successful_work_drives_independent_cycles_and_mixed_advances_both() {
    let case = case();
    let mut cursor = Cursor::new(&case).unwrap().unwrap();
    assert_eq!(cursor.choice().prefill_chunk_tokens.get(), 16);
    // A clipped final chunk still consumes one successful prefill wave.
    let mut final_row = prefill(5);
    final_row.work = Work::Prefill {
        offset: 48,
        count: 5,
        total_prompt_tokens: 53,
    };
    complete(&mut cursor, vec![final_row]);
    assert_eq!(
        (
            cursor.choice().prefill_completed,
            cursor.choice().decode_completed
        ),
        (1, 0)
    );
    assert_eq!(cursor.choice().prefill_chunk_tokens.get(), 32);
    complete(&mut cursor, vec![decode(), decode()]);
    assert_eq!(
        cursor.choice().decode_route,
        CalibrationDecodeRoute::FullLogits
    );
    complete(&mut cursor, vec![prefill(32), decode()]);
    assert_eq!(
        (
            cursor.choice().prefill_completed,
            cursor.choice().decode_completed
        ),
        (2, 2)
    );
    assert_eq!(cursor.choice().decode_route, CalibrationDecodeRoute::Actual);
    complete(&mut cursor, vec![prefill(64)]);
    complete(&mut cursor, vec![prefill(128)]);
    assert_eq!(cursor.choice().prefill_chunk_tokens.get(), 16);
}

#[test]
fn wave_plan_blocked_withdrawn_failed_and_indeterminate_do_not_advance() {
    let case = case();
    let mut cursor = Cursor::new(&case).unwrap().unwrap();
    let before = cursor.choice();
    // Blocked/maintenance turn never calls reconcile. A retry selects the same
    // ordinals; its live frontier may legitimately be recaptured.
    let attempt = cursor.begin_rows(vec![prefill(16), decode()]).unwrap();
    for _ in 0..3 {
        assert!(!cursor
            .apply(
                &attempt,
                CalibrationSubmissionState::NotSubmitted,
                false,
                &[],
                None
            )
            .unwrap());
        assert_eq!(cursor.choice(), before);
    }
    for state in [
        CalibrationSubmissionState::InFlightUnknown,
        CalibrationSubmissionState::Submitted,
    ] {
        assert!(cursor.apply(&attempt, state, false, &[], None).is_err());
        assert_eq!(cursor.choice(), before);
    }
    let (rows, host) = evidence(&attempt);
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            true,
            &rows,
            Some(&host)
        )
        .is_err());
    assert_eq!(cursor.choice(), before);
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &rows,
            Some(&host)
        )
        .unwrap());
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &rows,
            Some(&host)
        )
        .is_err());
}

#[test]
fn wave_plan_requires_actual_owner_frontier_work_and_forced_full_policy() {
    let case = case();
    let mut cursor = Cursor::new(&case).unwrap().unwrap();
    complete(&mut cursor, vec![decode()]);
    let attempt = cursor.begin_rows(vec![decode(), prefill(16)]).unwrap();
    let (mut actual, mut host) = evidence(&attempt);
    let before = cursor.choice();
    actual[0].full_logits = false;
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &actual,
            Some(&host)
        )
        .is_err());
    actual[0].full_logits = true;
    actual[0].row.work_generation += 1;
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &actual,
            Some(&host)
        )
        .is_err());
    actual[0].row.work_generation -= 1;
    host.rows[1].actual_work = HostStageWork::Prefill {
        offset: 1,
        count: 16,
        total_prompt_tokens: 33,
    };
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &actual,
            Some(&host)
        )
        .is_err());
    assert_eq!(cursor.choice(), before);
    let (mut actual, mut host) = evidence(&attempt);
    actual.reverse(); // The engine is allowed to use actual resource order.
    host.rows.reverse();
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &actual,
            Some(&host)
        )
        .unwrap());
}

#[test]
fn wave_plan_requires_successful_terminal_handoff_but_not_cost_eligibility() {
    use ferrum_engine::continuous_engine::HostTerminalStageV1;
    use ferrum_interfaces::model_executor::ExecutorCompletionWork;
    let case = case();
    let mut cursor = Cursor::new(&case).unwrap().unwrap();
    let attempt = cursor.begin_rows(vec![decode()]).unwrap();
    let (actual, mut host) = evidence(&attempt);
    host.completeness = HostStageCompleteness::AdditionalOrUnknownWork;
    host.rows[0].completeness = HostStageCompleteness::AdditionalOrUnknownWork;
    host.rows[0].terminal = Some(HostTerminalStageV1 {
        finish_reason: ferrum_types::FinishReason::Length,
        generated_tokens: 73,
        through_output_ordinal: 73,
        output_failed: false,
        physical_failed: false,
        scheduler_failed: false,
        terminal_handoff_succeeded: false,
        pending_restore_removed: false,
        admission_cancellation_work: ExecutorCompletionWork::NoAdditionalWork,
        cache_completion_work: ExecutorCompletionWork::NoAdditionalWork,
        other_physical_resources: true,
        request_slot_closed: true,
        owner_matched: true,
    });
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &actual,
            Some(&host)
        )
        .is_err());
    assert_eq!(cursor.choice().decode_completed, 0);
    host.rows[0]
        .terminal
        .as_mut()
        .unwrap()
        .terminal_handoff_succeeded = true;
    // Additional successful cleanup can make an observation ineligible. It
    // must not secretly repeat the same route to wait for a Known prediction.
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &actual,
            Some(&host)
        )
        .unwrap());
}

#[test]
fn wave_plan_missing_failed_or_mismatched_host_evidence_never_counts_as_progress() {
    let case = case();
    for status in [
        HostStageCompleteness::MissingEvidence,
        HostStageCompleteness::Failed,
        HostStageCompleteness::IdentityMismatch,
        HostStageCompleteness::InvalidClock,
    ] {
        let mut cursor = Cursor::new(&case).unwrap().unwrap();
        let attempt = cursor.begin_rows(vec![decode()]).unwrap();
        let (actual, mut host) = evidence(&attempt);
        assert!(cursor
            .apply(
                &attempt,
                CalibrationSubmissionState::HostReconciled,
                false,
                &actual,
                None
            )
            .is_err());
        host.rows[0].completeness = status;
        assert!(cursor
            .apply(
                &attempt,
                CalibrationSubmissionState::HostReconciled,
                false,
                &actual,
                Some(&host)
            )
            .is_err());
        assert_eq!(cursor.choice().decode_completed, 0);
    }
}

#[test]
fn wave_plan_reset_static_axis_and_checked_mixed_ordinal_are_explicit() {
    let mut case = case();
    case.wave_plan.as_mut().unwrap().prefill_chunks = None;
    let mut cursor = Cursor::new(&case).unwrap().unwrap();
    assert_eq!(
        cursor.choice().prefill_chunk_tokens,
        case.prefill_chunk_tokens
    );
    complete(&mut cursor, vec![decode()]);
    // Fresh case invocation is also how phase and repetition scopes reset.
    assert_eq!(
        Cursor::new(&case)
            .unwrap()
            .unwrap()
            .choice()
            .decode_completed,
        0
    );
    cursor.decode_completed = u64::MAX;
    let attempt = cursor.begin_rows(vec![decode(), prefill(1)]).unwrap();
    let before = cursor.choice();
    let (actual, host) = evidence(&attempt);
    assert!(cursor
        .apply(
            &attempt,
            CalibrationSubmissionState::HostReconciled,
            false,
            &actual,
            Some(&host)
        )
        .is_err());
    assert_eq!(cursor.choice(), before);
    case.wave_plan = None;
    assert!(Cursor::new(&case).unwrap().is_none());
}
