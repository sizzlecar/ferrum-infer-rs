//! Real scheduler/credited-output lifecycle; controlled execution is not GPU evidence.
use super::*;

async fn configured(
    mode: ferrum_types::PrefillDecodeExecution,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    completion_fixture_with_config(3, |config| {
        config.batching.prefill_decode_execution = mode;
        config.scheduler.active_decode_prefill_chunk = Some(2);
        config.scheduler.active_decode_prefill_token_budget = Some(3);
    })
    .await
}

#[tokio::test]
async fn unknown_completion_applies_one_aggregate_budget_across_real_prefill_owners() {
    let (engine, _, executor) = configured(ferrum_types::PrefillDecodeExecution::Mixed).await;
    let (decoder, mut decode_output) = request(&engine, 1, 6, None).await;
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 1).await;
    consume(&engine, &decoder, &mut decode_output).await;
    let (a, mut a_output) = request(&engine, 3, 4, None).await;
    let (b, mut b_output) = request(&engine, 3, 4, None).await;
    admit(&engine, 2).await;

    step(&engine, &executor, 3, 100).await;
    {
        let sequences = engine.inner.sequences.read();
        let mut progress = [
            sequences[&a].prefill_tokens_processed,
            sequences[&b].prefill_tokens_processed,
        ];
        progress.sort();
        assert_eq!(
            progress,
            [1, 2],
            "Q=3 is not multiplied by two prefill owners"
        );
        assert_eq!(sequences[&decoder].generated_tokens.len(), 2);
        assert!(!sequences[&a].prefill_complete && !sequences[&b].prefill_complete);
    }
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    consume(&engine, &decoder, &mut decode_output).await;
    step(&engine, &executor, 3, 100).await;
    {
        let sequences = engine.inner.sequences.read();
        assert!(sequences[&a].prefill_complete && sequences[&b].prefill_complete);
        assert_eq!(sequences[&a].prefill_tokens_processed, 3);
        assert_eq!(sequences[&b].prefill_tokens_processed, 3);
        assert_eq!(sequences[&decoder].generated_tokens.len(), 3);
    }
    consume(&engine, &decoder, &mut decode_output).await;
    consume(&engine, &a, &mut a_output).await;
    consume(&engine, &b, &mut b_output).await;
    drop(a_output);
    drop(b_output);
    cleanup(engine, decode_output).await;
}

#[tokio::test]
async fn split_completion_runs_one_phase_and_caps_prefill_with_a_ready_decoder() {
    let (engine, _, executor) = configured(ferrum_types::PrefillDecodeExecution::Split).await;
    let (decoder, mut decode_output) = request(&engine, 1, 6, None).await;
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 1).await;
    consume(&engine, &decoder, &mut decode_output).await;
    let (prefill, prefill_output) = request(&engine, 5, 4, None).await;
    admit(&engine, 1).await;

    // The existing completion rotation first services the decoder. Split does
    // not turn this one selected wave into two hidden physical submissions.
    step(&engine, &executor, 3, 100).await;
    assert_eq!(
        executor.submitted_requests.lock().last().unwrap(),
        &[decoder.clone()]
    );
    assert_eq!(
        engine.inner.sequences.read()[&prefill].prefill_tokens_processed,
        0
    );
    consume(&engine, &decoder, &mut decode_output).await;

    // The next fair turn is a P-only physical wave, but its ready domain still
    // contains the decoder. Neither active cap nor aggregate Q is bypassed.
    step(&engine, &executor, 3, 100).await;
    assert_eq!(
        executor.submitted_requests.lock().last().unwrap(),
        &[prefill.clone()]
    );
    assert_eq!(
        engine.inner.sequences.read()[&prefill].prefill_tokens_processed,
        2
    );
    assert_eq!(
        engine.inner.sequences.read()[&decoder]
            .generated_tokens
            .len(),
        2
    );
    assert_eq!(executor.physical.load(Ordering::Acquire), 3);
    drop(prefill_output);
    cleanup(engine, decode_output).await;
}

#[tokio::test]
async fn completion_publisher_policy_rejects_changed_amounts_without_acquiring_work() {
    let (engine, scheduler, executor) =
        configured(ferrum_types::PrefillDecodeExecution::Mixed).await;
    let (decoder, mut decode_output) = request(&engine, 1, 6, None).await;
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 1).await;
    consume(&engine, &decoder, &mut decode_output).await;
    let (a, a_output) = request(&engine, 4, 4, None).await;
    let (b, b_output) = request(&engine, 4, 4, None).await;
    admit(&engine, 2).await;
    let mut availability = Vec::new();
    let epochs = executor
        .write_execution_capacity_snapshot(&mut availability)
        .unwrap()
        .unwrap();
    let queue = scheduler
        .planning_state(
            NonZeroUsize::new(256).unwrap(),
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(
                    epochs.coordinator_id,
                    epochs.release_epoch,
                    epochs.capacity_epoch,
                    0,
                ),
                &availability,
            ),
        )
        .unwrap();
    let selection = |id: &RequestId, n: usize| PlanningWorkSelection {
        key: queue
            .requests()
            .iter()
            .find(|row| &row.key.request_id == id)
            .unwrap()
            .key
            .clone(),
        action: PlanningWorkAction::Prefill {
            offset: 0,
            count: NonZeroUsize::new(n).unwrap(),
        },
    };
    let valid = vec![selection(&a, 2), selection(&b, 1)];
    let per_request_bad = vec![selection(&a, 3)];
    let aggregate_bad = vec![selection(&a, 2), selection(&b, 2)];
    let proof = ControllerSafetyProof {
        recovery_peers: Vec::new(),
        protection: None,
        budget: ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
        queue,
        fences: Vec::new(),
    };
    let mut hint = ferrum_interfaces::BatchHint::simple(3);
    hint.max_tokens = 100;
    let before = executor.physical.load(Ordering::Acquire);
    // Exercise the same numeric verifier invoked just before real scheduler
    // publication. This proof is never passed to reserve/guard/dispatch.
    assert!(engine
        .inner
        .completion_work_policy_matches(&proof, &valid, &hint));
    assert!(engine.inner.completion_work_policy_matches(
        &proof,
        &valid.iter().cloned().rev().collect::<Vec<_>>(),
        &hint
    ));
    assert!(!engine
        .inner
        .completion_work_policy_matches(&proof, &per_request_bad, &hint));
    assert!(!engine
        .inner
        .completion_work_policy_matches(&proof, &aggregate_bad, &hint));
    hint.max_tokens = 2;
    assert!(!engine
        .inner
        .completion_work_policy_matches(&proof, &valid, &hint));
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    assert_eq!(
        engine.inner.sequences.read()[&a].prefill_tokens_processed,
        0
    );
    assert_eq!(
        engine.inner.sequences.read()[&b].prefill_tokens_processed,
        0
    );
    drop(a_output);
    drop(b_output);
    cleanup(engine, decode_output).await;
}
