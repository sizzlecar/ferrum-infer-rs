//! Ordinary no-wait traffic must create the cache it later consumes.
use super::*;

#[tokio::test]
async fn native_prefix_cpu_no_wait_business_capture_retired_source_restore_and_first_commit() {
    let (engine, executor) = startup_with_wait(None).await;
    assert!(engine
        .inner
        .config
        .scheduler
        .prefix_rendezvous_max_wait_ms
        .is_none());
    let before = copies(&executor).len();
    let runtime = engine.inner.cost_runtime.as_ref().unwrap().clone();
    let startup_epoch = runtime.prefix_cost_snapshot().unwrap().model_version();
    let (source, source_output) = submit(&engine, false).await;
    let mut capture_seen = false;
    let mut decisions = std::collections::VecDeque::new();
    tokio::time::timeout(Duration::from_secs(20), async {
        while engine.inner.sequences.read().contains_key(&source) {
            tick(&engine).await;
            let state = engine.inner.slo_controller.lock();
            assert!(
                state.prefix.is_none(),
                "no-wait cannot create a follower hold"
            );
            if decisions.len() == 128 {
                decisions.pop_front();
            }
            decisions.push_back((state.last_observation, state.last_audit));
            drop(state);
            capture_seen |= copies(&executor).len() == before + 1;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("no-wait source stalled: {decisions:?}"));
    source_output.await.unwrap();
    assert!(
        capture_seen,
        "ordinary no-wait source published no business checkpoint: {decisions:?}; {}",
        failure_state(&engine, &executor, &source)
    );
    assert_eq!(copies(&executor).len(), before + 1);
    assert!(!engine.inner.sequences.read().contains_key(&source));
    let business_capture = executor
        .evidence
        .prefix
        .as_ref()
        .unwrap()
        .publications
        .lock()
        .iter()
        .rev()
        .find(|record| record.domain.kind() == vnext::NativeCheckpointTransferKind::Capture)
        .expect("actual business Capture must publish full acknowledgement")
        .identity
        .clone();
    // The arrival after this observed update is ordinary workload timing. Only
    // the installed automatic worker may publish the new model; no manual drain.
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if runtime.prefix_cost_snapshot().is_some_and(|snapshot| {
                snapshot.current() && snapshot.model_version() > startup_epoch
            }) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("business Capture receipt was not automatically learned");

    // Only now does a new request arrive. There is no live producer to put in
    // a made-up rendezvous offer, and startup used different token contents.
    let (target, target_output) = submit(&engine, true).await;
    let mut restored = false;
    let mut first_commit_witness = false;
    decisions.clear();
    tokio::time::timeout(Duration::from_secs(20), async {
        while engine.inner.sequences.read().contains_key(&target) {
            let before_tokens = engine
                .inner
                .sequences
                .read()
                .get(&target)
                .map(|s| s.generated_tokens.len());
            tick(&engine).await;
            let state = engine.inner.slo_controller.lock();
            assert!(
                state.prefix.is_none(),
                "ready restore cannot create a timed hold"
            );
            let audit = state.last_audit;
            if decisions.len() == 128 {
                decisions.pop_front();
            }
            decisions.push_back((state.last_observation, audit));
            drop(state);
            if let Some(sequence) = engine.inner.sequences.read().get(&target) {
                if sequence.prefill_tokens_processed == BOUNDARY
                    && sequence.generated_tokens.is_empty()
                {
                    restored = true;
                    assert!(
                        executor.prefix_session(&source).is_none(),
                        "retired producer must have no native slot in the fresh target capture"
                    );
                    let records = executor
                        .evidence
                        .prefix
                        .as_ref()
                        .unwrap()
                        .publications
                        .lock();
                    let restore = records
                        .iter()
                        .rev()
                        .find(|record| {
                            record.domain.kind() == vnext::NativeCheckpointTransferKind::Restore
                        })
                        .expect("native Restore full acknowledgement");
                    assert!(
                        restore
                            .source
                            .as_ref()
                            .is_some_and(|origin| origin.same_transfer(&business_capture)),
                        "restore must use the business checkpoint's actual capture authority"
                    );
                    drop(records);
                    assert_eq!(copies(&executor).len(), before + 2);
                    assert_eq!(
                        executor.native_structured_history.lock()[&target].len(),
                        BOUNDARY
                    );
                }
                if before_tokens == Some(0) && !sequence.generated_tokens.is_empty() {
                    first_commit_witness = audit.is_some_and(|a| {
                        a.witness.is_some() && a.backend_submitted && a.host_reconciled
                    });
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
        "no-wait target stalled: restored={restored}, first={first_commit_witness}, {decisions:?}"
    )
    });
    target_output.await.unwrap();
    let actual = copies(&executor);
    // Preserve the live publication/revocation state before orderly shutdown.
    // This reads diagnostics only; it cannot re-run or qualify a prediction.
    let failed_state =
        (!restored || !first_commit_witness).then(|| failure_state(&engine, &executor, &target));
    engine.shutdown().await.unwrap();
    assert!(
        restored,
        "the retired business producer's checkpoint was not restored: {decisions:?}; before_shutdown={failed_state:?}; {}",
        prefix_journal_tail(&engine)
    );
    assert!(
        first_commit_witness,
        "restored target first commit lacked fresh full-queue proof: before_shutdown={failed_state:?}; {}",
        prefix_journal_tail(&engine)
    );
    assert_eq!(actual.len(), before + 2);
    assert!(actual[before..]
        .iter()
        .all(|bytes| bytes == &(BOUNDARY as u32).to_le_bytes()));
}
