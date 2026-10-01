use super::*;

#[tokio::test]
async fn source8_engine_original_deferred_preparation_retries_unchanged_frontier() {
    let (mut session, executor) = source8_session(3, true).await;
    let (ids, outputs) = admit_cohort(&mut session, 3, false).await;
    let before: Vec<_> = ids.iter().map(|id| frontier(&session, id)).collect();
    let rows: Vec<_> = before
        .iter()
        .map(|f| f.prefill_work(NonZeroU32::MIN).unwrap())
        .collect();
    // This unused CPU fixture has no admitted physical allocator frontier to
    // release. Use its real backend-owned maintenance ticket; the original
    // guarded maintenance method changes its advertised capacity epoch.
    executor.deferrals.capacity(&ids, Some(true));
    let deferred = wave(&mut session, &executor, rows).await;
    assert_eq!(
        deferred.submission,
        CalibrationSubmissionState::NotSubmitted
    );
    assert!(deferred.error.is_none(), "{deferred:?}");
    assert!(deferred.host_stages.is_none());
    let proof = deferred.no_submission_proof().unwrap_or_else(|| {
        panic!(
            "recorder-minted capacity proof missing: {deferred:?}; source={:?}",
            session
                .prepared_owner_capture
                .as_ref()
                .unwrap()
                .prepared_audit()
        )
    });
    assert_eq!(proof.fifo(), 1);
    assert_eq!(proof.participants().len(), ids.len());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    for (index, id) in ids.iter().enumerate() {
        let now = frontier(&session, id);
        assert_eq!(now.owner_incarnation(), before[index].owner_incarnation());
        assert_eq!(now.work_generation(), before[index].work_generation());
        assert_eq!(now.generated_tokens(), 0);
        assert_eq!(now.prefill_progress(), before[index].prefill_progress());
        assert_eq!(proof.participants()[index].request_id, *id);
    }
    assert_collecting(&session, 1, 1);
    assert!(matches!(
        session.advance_prepared_owner_prefix_release().unwrap(),
        PrefixReleaseProgressV5::Preparing
    ));
    // The next actual attempt gets its own original ticket. The deferred
    // attempt remains in the source and never becomes a completed sample.
    assert!(matches!(
        bounded(session.step(CalibrationAction::Maintenance))
            .await
            .unwrap(),
        CalibrationTurn::MaintenanceReconciled
    ));
    assert_eq!(
        executor.deferrals.maintenance_calls.load(Ordering::Acquire),
        1
    );
    assert_eq!(executor.deferrals.capacity_epoch.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_collecting(&session, 1, 1);
    let rows = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .prefill_work(NonZeroU32::MIN)
                .unwrap()
        })
        .collect();
    let actual = wave(&mut session, &executor, rows).await;
    assert!(actual.error.is_none(), "{actual:?}");
    assert_eq!(
        actual.submission,
        CalibrationSubmissionState::HostReconciled
    );
    assert!(actual.no_submission_proof().is_none());
    assert_eq!(actual.host_stage_queue.unwrap().accepted_ordinal, Some(2));
    assert!(actual.host_stages.as_ref().unwrap().call_id > proof.call_id());
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_collecting(&session, 2, 2);
    release(&mut session, &ids, 2).await;
    // This test ends at the proven retry/release boundary. Incomplete output
    // is cancelled and cannot install a model or a successful checkpoint.
    drop(outputs);
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    session.shutdown().await.unwrap();
}
