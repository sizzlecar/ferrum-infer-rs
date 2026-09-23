use super::*;

fn admit(s: &ContinuousBatchScheduler) {
    let receipt = s
        .prepare_dynamic_admission_observed(
            1,
            wake(),
            &mut |request| {
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                })
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(receipt.admitted(), 1);
}

fn publish_prefill(s: &ContinuousBatchScheduler, id: &RequestId, context: usize, previous: usize) {
    let output = s
        .prepare_prefill_output_publication(id, context, previous)
        .unwrap();
    s.mark_prefill_complete(id, context);
    s.publish_prefill_output_commit(&output, previous + 1)
        .unwrap();
}

#[tokio::test]
async fn recompute_context_includes_last_sample_and_never_double_counts_replayed_output() {
    let s = scheduler();
    // Product ingress declares its actual original tokenized prompt once.
    let request = InferenceRequest::new("four token prompt fixture", ModelId::new("fixture"))
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, serde_json::json!(4));
    let id = s.submit(request).await.unwrap();
    admit(&s);
    publish_prefill(&s, &id, 4, 0);
    assert!(s.defer_decode_to_waiting_for_capacity(&id, 1));
    let waiting = s.planning_state(limit(), wake()).unwrap();
    let row = &waiting.requests()[0];
    assert_eq!(row.recompute_target_tokens, Some(4));
    assert_eq!(row.committed_output_tokens, 1);
    assert_eq!(row.prefill_context_tokens, Some(5));
    assert_eq!(row.prefill_offset, 0);
    assert!(!row.readiness.ready());
    admit(&s);
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let action = |count| PlanningWorkAction::Prefill {
        offset: 0,
        count: NonZeroUsize::new(count).unwrap(),
    };
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, action(6))],
            &hint(),
            wake()
        )
        .unwrap(),
        PlanningSelectionOutcome::Rejected(_)
    ));
    assert!(snapshot.matches(&s.planning_state(limit(), wake()).unwrap()));
    let PlanningSelectionOutcome::Published { batch, .. } = s
        .try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, action(5))],
            &hint(),
            wake(),
        )
        .unwrap()
    else {
        panic!("the exact five-token rebuild must publish")
    };
    assert_eq!(
        (
            batch.requests[0].tokens_processed,
            batch.requests[0].tokens_to_process
        ),
        (0, Some(5))
    );
    publish_prefill(&s, &id, 5, 1);
    let decoded = s.planning_state(limit(), wake()).unwrap();
    assert_eq!(decoded.requests()[0].computed_tokens, 5);
    assert_eq!(decoded.requests()[0].committed_output_tokens, 2);
    let PlanningSelectionOutcome::Published { batch, .. } = s
        .try_select_planned_wave(
            &decoded,
            &[selected(&decoded, &id, PlanningWorkAction::Decode)],
            &hint(),
            wake(),
        )
        .unwrap()
    else {
        panic!("decode must continue after rebuild")
    };
    assert_eq!(
        (
            batch.requests[0].tokens_processed,
            batch.requests[0].tokens_to_process
        ),
        (5, Some(1))
    );
    s.update_decode_progress(&id, 3);
    assert!(s.defer_decode_to_waiting_for_capacity(&id, 1));
    admit(&s);
    let repeated = s.planning_state(limit(), wake()).unwrap();
    assert_eq!(repeated.requests()[0].recompute_target_tokens, Some(6));
    assert_eq!(repeated.requests()[0].prefill_context_tokens, Some(7));
    assert!(!s.mark_prefill_chunk_processed(&id, 7, 3));
    let partial = s.planning_state(limit(), wake()).unwrap();
    // The mutable last-prefill boundary is now seven, already containing
    // output. Adding cumulative outputs to it would incorrectly produce ten.
    assert_eq!(partial.requests()[0].prompt_tokens, Some(7));
    assert_eq!(partial.requests()[0].prefill_context_tokens, Some(7));
    assert_eq!(partial.requests()[0].prefill_offset, 3);
    assert_eq!(partial.requests()[0].recompute_target_tokens, Some(6));
}

#[tokio::test]
async fn recompute_without_original_prompt_evidence_cannot_authorize_a_full_span() {
    let s = scheduler();
    let id = s
        .submit(InferenceRequest::new(
            "unknown original tokens",
            ModelId::new("fixture"),
        ))
        .await
        .unwrap();
    admit(&s);
    publish_prefill(&s, &id, 4, 0);
    assert!(s.defer_decode_to_waiting_for_capacity(&id, 1));
    admit(&s);
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    assert_eq!(snapshot.requests()[0].recompute_target_tokens, Some(4));
    assert_eq!(snapshot.requests()[0].prefill_context_tokens, None);
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, prefill())],
            &hint(),
            wake()
        )
        .unwrap(),
        PlanningSelectionOutcome::Rejected("unknown full prefill boundary")
    ));
}

#[test]
fn rebuild_context_arithmetic_is_checked_without_inventing_prompt_evidence() {
    assert_eq!(
        projection::prefill_context(Some(usize::MAX), 0),
        Ok(Some(usize::MAX))
    );
    assert_eq!(
        projection::prefill_context(Some(usize::MAX), 1),
        Err(PlanningStateUnavailable::CounterExhausted)
    );
    assert_eq!(projection::prefill_context(None, usize::MAX), Ok(None));
}
