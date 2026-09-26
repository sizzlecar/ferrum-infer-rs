//! Exercise the real one-owner admission protocol before a joint prefix wave.
//! No prepared receipt, failure flag or successful sampler result is injected.
use super::*;

fn joint_declaration(
    session: &CalibrationSession,
    source: &Source,
) -> (StructuredCalibrationGroupOptionsV2, StructuredPrefixPlanV5) {
    let (mut options, mut prefixes) = declaration(session, source);
    let child = &mut options.children[0];
    child.scope.owner.rows = 2;
    child.membership_rule.owner.rows = 2;
    let row = child.membership_rule.windows[0].rows[0].clone();
    child.membership_rule.windows[0].rows = vec![row; 2];
    for phase in &mut child.cohort_plan.phases {
        let request = phase[0].requests[0].clone();
        phase[0].requests = vec![request; 2];
    }
    for phase in &mut prefixes.phases {
        let declaration = phase[0].as_mut().unwrap();
        declaration.release_generated = 2;
        let policy = plan(session, &[10, 10]).tokenizer_policy_sha256;
        declaration.slots = vec![
            StructuredPrefixSlotV5 {
                tokenizer_policy_sha256: policy,
                token_ids: vec![TokenId::new(10), TokenId::new(10)],
                token_bytes: vec![b"a".to_vec(), b"a".to_vec()],
            },
            StructuredPrefixSlotV5 {
                tokenizer_policy_sha256: policy,
                token_ids: vec![TokenId::new(10), TokenId::new(11)],
                token_bytes: vec![b"a".to_vec(), vec![0xc3]],
            },
        ];
    }
    (options, prefixes)
}

async fn joint_session() -> (
    CalibrationSession,
    Arc<ControlledExecutor>,
    Source,
    Vec<RequestId>,
    Vec<CreditedOutputSession>,
) {
    let (mut session, executor) = prepared_session_with_width(2).await;
    let source = Source::new();
    let (options, prefixes) = joint_declaration(&session, &source);
    session
        .begin_structured_prefix_cost_group_v5(options, prefixes)
        .await
        .unwrap();
    session.begin_structured_cost_group_cohort_v2(0).unwrap();
    let mut ids = Vec::new();
    let mut outputs = Vec::new();
    for _ in 0..2 {
        let request = request(&session, 3);
        ids.push(request.id.clone());
        outputs.push(
            session
                .add_request(
                    request,
                    InferenceRequestContext::from_ingress(slo_clock_now()),
                    Arc::new(OutputProjectionContract::cli_text()),
                )
                .await
                .unwrap(),
        );
    }
    for id in &ids {
        ready(&session, id, false).await;
    }
    (session, executor, source, ids, outputs)
}

fn joint_prefill(session: &CalibrationSession, ids: &[RequestId]) -> Vec<CalibrationWork> {
    ids.iter()
        .map(|id| frontier(session, id).prefill_work(NonZeroU32::MIN).unwrap())
        .collect()
}

fn records(source: &Source) -> Vec<serde_json::Value> {
    std::fs::read_to_string(&source.0)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["record"].clone())
        .collect()
}

#[tokio::test]
async fn prefix_source5_admission_drains_before_joint_g2_release_and_full_length() {
    let (mut session, executor, source, ids, outputs) = joint_session().await;
    let mut consumers = tokio::task::JoinSet::new();
    for mut output in outputs {
        consumers.spawn(async move {
            let mut last = None;
            while let Some(frame) = output.frames.next().await {
                last = Some(frame.metadata().clone());
                drop(frame);
            }
            let completion = output.completion.await.unwrap();
            let OutputCompletion::Succeeded { reason, usage, .. } = completion.payload() else {
                panic!("the original complete request must reach Length");
            };
            assert_eq!(*reason, ferrum_types::FinishReason::Length);
            assert_eq!(usage.completion_tokens, 3);
            let last = last.unwrap();
            assert!(last.terminal);
            assert_eq!(last.generated_tokens, 3);
        });
    }
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 2);
    // These are real separate admission turns, just as in the CLI. Do not
    // offer the B2 wave after the first owner becomes selectable.
    admit(&mut session).await;
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 1);
    assert_eq!(session.engine.inner.scheduler.active_count(), 1);
    assert!(matches!(
        session.advance_structured_prefix_release_v5().unwrap(),
        PrefixReleaseProgressV5::Preparing
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    admit(&mut session).await;
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 0);
    assert_eq!(session.engine.inner.scheduler.active_count(), 2);
    assert!(matches!(
        session.step(CalibrationAction::AdmitOne).await.unwrap(),
        CalibrationTurn::Blocked(CalibrationBlockReason::AdmissionUnavailable)
    ));
    assert!(session
        .structured_cost_group_progress_v2()
        .unwrap()
        .iter()
        .all(|p| p.offered_attempts == 0));
    assert!(session.prefix_preparation_failure().is_none());
    let first = joint_prefill(&session, &ids);
    let report = wave(&mut session, &executor, first).await;
    assert!(report.error.is_none(), "{:?}", report.error);
    assert_eq!(
        report.submission,
        CalibrationSubmissionState::HostReconciled
    );
    assert!(report.structured_prepared_projection.is_none());
    for id in &ids {
        ready(&session, id, false).await;
        assert_eq!(frontier(&session, id).generated_tokens(), 1);
    }
    assert!(matches!(
        session.advance_structured_prefix_release_v5().unwrap(),
        PrefixReleaseProgressV5::Preparing
    ));
    let second = ids
        .iter()
        .map(|id| frontier(&session, id).decode_work().unwrap())
        .collect();
    let report = wave(&mut session, &executor, second).await;
    assert!(report.error.is_none(), "{:?}", report.error);
    assert_eq!(
        report.submission,
        CalibrationSubmissionState::HostReconciled
    );
    assert!(report.structured_prepared_projection.is_none());
    for id in &ids {
        ready(&session, id, false).await;
    }
    let PrefixReleaseProgressV5::Released { receipts } =
        session.advance_structured_prefix_release_v5().unwrap()
    else {
        panic!("both original actors must reach the common G2 frontier");
    };
    assert_eq!(receipts.len(), 2);
    for (slot, id) in ids.iter().enumerate() {
        let receipt = receipts
            .iter()
            .find(|r| &r.frontier.request_id == id)
            .unwrap();
        assert_eq!(receipt.frontier.generated_tokens, 2);
        assert_eq!(receipt.frontier.kv_tokens, 2);
        assert_eq!(
            receipt.frontier.pending_utf8,
            if slot == 0 { vec![] } else { vec![0xc3] }
        );
        assert_eq!(
            receipt.actor_applied_output_ordinal,
            receipt.frontier.output_accepted_ordinal
        );
        assert!(session.engine.inner.sequences.read()[id]
            .calibration_prefix
            .is_none());
    }
    let suffix = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                .unwrap()
        })
        .collect();
    let report = wave(&mut session, &executor, suffix).await;
    assert!(report.error.is_none(), "{:?}", report.error);
    assert!(report
        .structured_prepared_projection
        .as_ref()
        .unwrap()
        .error
        .is_some());
    // The controlled CPU lacks qualified numerical projection. Its true
    // failure is retained, even though preparation and complete output succeed.
    bounded(async {
        while let Some(result) = consumers.join_next().await {
            result.unwrap();
        }
    })
    .await;
    assert_eq!(executor.physical.load(Ordering::Acquire), 3);
    let artifact = session.finish_structured_cost_group_v2().await.unwrap();
    assert!(artifact.failure.is_some());
    assert!(artifact.children.iter().all(|child| child.model.is_none()));
    let source_records = records(&source);
    let completed: Vec<_> = source_records
        .iter()
        .filter(|r| r["kind"] == "preparation_completed")
        .collect();
    assert_eq!(completed.len(), 2);
    for (wave_index, record) in completed.iter().enumerate() {
        assert_eq!(record["offered"], wave_index + 1);
        assert_eq!(record["reconciled"], true);
        assert!(record["failure"].is_null());
        assert_eq!(record["rows"].as_array().unwrap().len(), 2);
        for row in record["rows"].as_array().unwrap() {
            assert_eq!(row["before"]["generated_tokens"], wave_index);
            assert_eq!(row["after"]["generated_tokens"], wave_index + 1);
        }
    }
    let offered = source_records
        .iter()
        .filter(|r| r["kind"] == "preparation_offered")
        .count();
    assert_eq!(offered, 2);
    assert_eq!(
        source_records
            .iter()
            .filter(|r| r["kind"] == "preparation_released")
            .count(),
        2
    );
    assert!(!source_records.iter().any(|r| r["kind"] == "phase_freeze"));
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_source5_offered_incomplete_cohort_remains_a_counted_fatal_attempt() {
    let (mut session, executor, source, ids, outputs) = joint_session().await;
    admit(&mut session).await;
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 1);
    let work = joint_prefill(&session, &ids);
    let turn = session.step(CalibrationAction::Wave(work)).await.unwrap();
    assert!(matches!(
        turn,
        CalibrationTurn::Blocked(CalibrationBlockReason::SelectionUnavailable(
            "calibration_exact_cohort_unavailable"
        ))
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let progress = session.structured_cost_group_progress_v2().unwrap();
    assert_eq!(progress.len(), 1);
    assert_eq!(progress[0].phase, StructuredCapturePhase::Failed);
    assert_eq!(progress[0].offered_attempts, 1);
    assert!(progress[0].failure.as_ref().unwrap().contains(
        "source5 preparation was not submitted: SelectionUnavailable(\"calibration_exact_cohort_unavailable\")"
    ));
    assert!(session.advance_structured_prefix_release_v5().is_err());
    let source_records = records(&source);
    assert_eq!(
        source_records
            .iter()
            .filter(|r| r["kind"] == "preparation_offered")
            .count(),
        1
    );
    assert!(source_records.iter().any(|r| r["kind"] == "phase_failed"));
    assert!(!source_records.iter().any(|r| matches!(
        r["kind"].as_str(),
        Some("preparation_completed" | "preparation_released" | "phase_freeze")
    )));
    drop(outputs);
    session.shutdown().await.unwrap();
}
