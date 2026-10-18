//! Real eager CPU execution and ownership at the completion-worker boundary.
//!
//! The barriers only schedule the existing worker. The work under cancellation
//! is the production embedding/head plan, with real admission, Step, submission,
//! reaper, and sequence registry owners. This fixture has no native resident
//! program or attention KV tensors.

use super::*;
use std::future::Future;
use std::pin::Pin;
use std::sync::mpsc;

type CpuRuntime = ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
type WorkerTask<'a> = Pin<Box<dyn Future<Output = ()> + 'a>>;

/// Holding the first worker job makes one manual poll a deterministic enqueue
/// or queue-capacity wait. Drop also opens the gate if an assertion fails.
struct WorkerGate<'a> {
    release: Option<mpsc::Sender<()>>,
    completion: WorkerTask<'a>,
}

impl<'a> WorkerGate<'a> {
    async fn closed(worker: &'a VNextCompletionWorker) -> Self {
        let (started_sender, started_receiver) = tokio::sync::oneshot::channel();
        let (release, released) = mpsc::channel();
        let mut completion: WorkerTask<'a> = Box::pin(async move {
            worker
                .execute(VNextCompletionTaskKind::PostSubmitDrain, move || {
                    let _ = started_sender.send(());
                    let _ = released.recv();
                })
                .await
                .unwrap();
        });
        assert!(futures::poll!(completion.as_mut()).is_pending());
        started_receiver.await.unwrap();
        Self {
            release: Some(release),
            completion,
        }
    }

    async fn open(mut self) {
        self.release.take().unwrap().send(()).unwrap();
        self.completion.as_mut().await;
    }
}

impl Drop for WorkerGate<'_> {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
    }
}

fn marker(worker: &VNextCompletionWorker) -> WorkerTask<'_> {
    Box::pin(async move {
        worker
            .execute(VNextCompletionTaskKind::PostSubmitDrain, || ())
            .await
            .unwrap();
    })
}

/// A result is sent before the worker's post-task recovery sweep. Two FIFO
/// barriers also cover cancellation whose wake could not enter a full queue.
async fn settle_worker(executor: &VNextModelExecutor<CpuRuntime>) {
    marker(&executor.completion_worker).await;
    marker(&executor.completion_worker).await;
}

fn sequence(
    executor: &VNextModelExecutor<CpuRuntime>,
    frontier: &ExecutionFrontier,
) -> Weak<VNextSequence<CpuRuntime>> {
    executor
        .sequences
        .lock()
        .token_index
        .get(frontier.sequence())
        .unwrap()
        .clone()
}

fn input(frontier: &ExecutionFrontier, token: u32) -> ExecutionWork<'_> {
    ExecutionWork::new(
        frontier,
        TokenWork::Decode {
            token: TokenId::new(token),
            position: frontier.consumed_tokens(),
        },
        &LogitsReturnPolicy::FullLogits,
    )
    .unwrap()
}

fn assert_no_live_resources(executor: &VNextModelExecutor<CpuRuntime>) {
    assert_eq!(executor.reaper.retained_count(), 0, "completion owners");
    assert_eq!(executor.reaper.quarantined_count(), 0);
    assert_eq!(executor.lane.in_flight_count(), 0, "submitted lane owners");
    assert_eq!(executor.sequences.lock().total_len(), 0, "registry owners");
    // A released Step/Invocation slot deliberately retains its backing in the
    // plan's lane-stable arena. Account for that exact residency separately
    // from live request resources; do not accept an arbitrary byte allowance.
    let pools = executor.plan_resources.dynamic_pool_status().unwrap();
    let mut cached_bytes = 0;
    for pool in pools.pools() {
        let occupancy = pool.live_occupancy();
        assert_eq!(occupancy.transient().total().claim_count(), 0, "{pool:?}");
        assert_eq!(
            occupancy.transient().total().physical_bytes(),
            0,
            "{pool:?}"
        );
        let stable = occupancy.lane_stable();
        assert_eq!(
            stable.total().claim_count(),
            stable.step().claim_count() + stable.invocation().claim_count(),
            "only Step/Invocation arena claims may remain: {pool:?}"
        );
        let bytes = stable.step().physical_bytes() + stable.invocation().physical_bytes();
        assert_eq!(stable.total().physical_bytes(), bytes, "{pool:?}");
        assert_eq!(pool.resident_bytes() - pool.free_bytes(), bytes, "{pool:?}");
        assert!(!pool.poisoned(), "{pool:?}");
        cached_bytes += bytes;
    }
    let resources = executor.plan_runtime_resource_snapshot().unwrap().unwrap();
    assert_eq!(resources.dynamic_used_bytes(), cached_bytes);
    assert_eq!(resources.pending_growth_bytes(), 0);
    assert_eq!(resources.quarantined_bytes(), 0);
    assert_eq!(
        executor.plan_resources.deferred_cleanup_status().pending(),
        0
    );
    assert_eq!(
        executor.completion_worker.metrics_snapshot()["pending_tasks"],
        0
    );
}

/// The retained arena is a plan cache, not leaked request ownership. Once the
/// real executor/lane is gone, explicit plan close must release every allocation.
fn close_fixture(fixture: Fixture<CpuRuntime>) {
    let runtime = Arc::clone(&fixture.executor.runtime);
    let resources = Arc::clone(&fixture.executor.plan_resources);
    drop(fixture);
    match PlanRuntimeResources::close(resources) {
        Ok(ferrum_runtime::PlanRuntimeCloseOutcome::Closed(_)) => {}
        Ok(ferrum_runtime::PlanRuntimeCloseOutcome::Referenced {
            strong_count,
            deferred_cleanup,
            ..
        }) => panic!(
            "closed cohort retained plan owners: {strong_count}, cleanup: {deferred_cleanup:?}"
        ),
        Err(failure) => panic!("plan close failed: {:?}", failure.failure()),
    }
    assert_eq!(runtime.resident_bytes(), 0, "all physical owners released");
}

async fn assert_cancelled(
    executor: &VNextModelExecutor<CpuRuntime>,
    inputs: &[ExecutionWork<'_>],
    owners: &[Weak<VNextSequence<CpuRuntime>>],
) {
    settle_worker(executor).await;
    assert_no_live_resources(executor);
    for owner in owners {
        assert!(
            owner.upgrade().is_none(),
            "cancelled incarnation is still retained"
        );
    }
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    assert!(executor.execute_tokens(inputs).await.is_err());
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before,
        "an abandoned cache must not submit or replay its work"
    );
    assert_no_live_resources(executor);
}

fn assert_submitted_waiting(
    executor: &VNextModelExecutor<CpuRuntime>,
    before: u64,
    owners: &[Weak<VNextSequence<CpuRuntime>>],
) {
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before + 1
    );
    assert_eq!(
        executor.reaper.retained_count(),
        1,
        "one real shared submission"
    );
    assert_eq!(executor.lane.in_flight_count(), 1);
    for owner in owners {
        let owner = owner.upgrade().unwrap();
        assert_eq!(
            *owner.tokens.lock(),
            [0],
            "no product frontier commit before readback"
        );
        assert!(
            owner.operation.try_lock().is_err(),
            "actual execution owns the sequence gate"
        );
    }
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_full_logits_retires_and_completes_real_cohort() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let executor = &fixture.executor;
    let ids = [RequestId::new(), RequestId::new()];
    let caches = [
        admit(&fixture, &ids[0]).await,
        admit(&fixture, &ids[1]).await,
    ];
    let owners = caches
        .iter()
        .map(|frontier| sequence(executor, frontier))
        .collect::<Vec<_>>();
    // Reverse caller order relative to admission; the independent numerical
    // oracle is the fixture's dyadic embedding row and identity dense head.
    let inputs = [input(&caches[1], 3), input(&caches[0], 1)];
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    let outputs = completed(executor.execute_tokens(&inputs).await.unwrap());
    assert_eq!(outputs.len(), 2);
    let expected = [[1.0_f32, 1.0009765625, -1.0, -2.0], [3.0, 2.0, 1.0, -4.0]];
    for (index, ((output, input), expected)) in
        outputs.iter().zip(&inputs).zip(expected).enumerate()
    {
        let ExecutorSamplingOutput::FullLogits(logits) = sampling(output) else {
            panic!("full-logits policy was not retained")
        };
        assert_eq!(
            logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected.map(f32::to_bits)
        );
        output.validate_for(index, input, VOCAB).unwrap();
        assert_eq!(output.next_frontier().sequence(), input.sequence());
        assert_eq!(output.next_frontier().consumed_tokens(), 2);
    }
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before + 1
    );
    assert_eq!(executor.reaper.retained_count(), 0);
    assert_eq!(executor.lane.in_flight_count(), 0);
    for (owner, token) in owners.iter().zip([1, 3]) {
        assert_eq!(*owner.upgrade().unwrap().tokens.lock(), [0, token]);
    }
    for output in &outputs {
        let outcome = executor
            .finish_tokens(
                output.next_frontier().sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            )
            .await
            .unwrap();
        assert!(outcome.error.is_none(), "{:?}", outcome.error);
        let ExecutionCloseDisposition::Ended(ended) = outcome.disposition else {
            panic!("quiescent actual cohort must end")
        };
        assert_eq!(ended.sequence(), output.next_frontier().sequence());
    }
    settle_worker(executor).await;
    assert_no_live_resources(executor);
    assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    assert!(executor.execute_tokens(&inputs).await.is_err());
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before
    );
    close_fixture(fixture);
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_failed_retirement_returns_exact_step_owner() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let executor = &fixture.executor;
    let id = RequestId::new();
    let cache = admit(&fixture, &id).await;
    let owner = sequence(executor, &cache);
    let sequence = owner.upgrade().unwrap();
    let tokens = [0, 1];
    executor
        .extend_sequence(
            &sequence,
            ResourceWorkShape::single(TokenSpanWork::from_token_ids(&tokens, 0..2).unwrap())
                .unwrap(),
        )
        .unwrap();
    let span = TokenSpanWork::from_token_ids(&tokens, 1..2).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&sequence.session)]).unwrap();
    let step = executor.begin_step(&batch, &sequence, &span).unwrap();
    let step_id = step.batch_step_id();
    let frames = step.participant_frames().collect::<Vec<_>>();
    let step_owner = Arc::downgrade(&step);
    let wave = executor.prepare_wave(&step, &sequence, &span).unwrap();
    let submitted_before = executor.metrics.submitted_waves.load(Ordering::Relaxed);

    // The real prepared wave, not an artificial Arc clone in this test, keeps
    // finalization from consuming its Step. Both failures must return that same
    // owner; converting only their error text would lose this retry authority.
    let failure = PreparedRetirement::Ordinary(step)
        .try_retire_normal()
        .expect_err("a live prepared wave must prevent normal retirement");
    assert!(matches!(
        failure.error(),
        VNextError::InvalidExecutionPlan { .. }
    ));
    let retirement = failure.into_retirement();
    let PreparedRetirement::Ordinary(recovered) = &retirement else {
        panic!("ordinary failure changed the authority kind")
    };
    assert!(Arc::ptr_eq(recovered, wave.step_resources()));
    assert!(Arc::ptr_eq(recovered, &step_owner.upgrade().unwrap()));
    assert_eq!(retirement.batch_step_id(), step_id);

    let failure = retirement
        .try_abort()
        .expect_err("a live prepared wave must also prevent abort finalization");
    assert!(matches!(
        failure.error(),
        VNextError::InvalidExecutionPlan { .. }
    ));
    let retirement = failure.into_retirement();
    let PreparedRetirement::Ordinary(recovered) = &retirement else {
        panic!("abort failure changed the authority kind")
    };
    assert!(Arc::ptr_eq(recovered, wave.step_resources()));
    assert!(Arc::ptr_eq(recovered, &step_owner.upgrade().unwrap()));

    drop(wave);
    let receipt = retirement.try_abort().unwrap();
    assert_eq!(receipt.batch_step_id(), step_id);
    assert_eq!(receipt.participants().len(), frames.len());
    for (participant, frame) in receipt.participants().iter().zip(frames) {
        assert_eq!(participant.assignment(), frame);
        assert_eq!(
            participant.disposition(),
            StepParticipantRetirementDisposition::Aborted
        );
    }
    assert!(
        step_owner.upgrade().is_none(),
        "the finalized Step owner was released"
    );
    assert_eq!(
        *sequence.tokens.lock(),
        [0],
        "unsubmitted work was not committed"
    );
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        submitted_before
    );
    drop(batch);
    drop(sequence);
    close(executor, cache.sequence()).await;
    settle_worker(executor).await;
    assert!(owner.upgrade().is_none());
    assert_no_live_resources(executor);
    close_fixture(fixture);
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_cancel_waiting_for_worker_capacity() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let executor = &fixture.executor;
    let id = RequestId::new();
    let cache = admit(&fixture, &id).await;
    let owners = [sequence(executor, &cache)];
    let inputs = [input(&cache, 1)];
    let gate = WorkerGate::closed(&executor.completion_worker).await;
    let mut queued = marker(&executor.completion_worker);
    assert!(futures::poll!(queued.as_mut()).is_pending());
    let scheduled = executor.completion_worker.metrics_snapshot()["scheduled_tasks"].clone();
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    let mut decode = Box::pin(executor.execute_tokens(&inputs));
    assert!(futures::poll!(decode.as_mut()).is_pending());
    assert_submitted_waiting(executor, before, &owners);
    assert_eq!(
        executor.completion_worker.metrics_snapshot()["scheduled_tasks"],
        scheduled,
        "the actual submitted wave is waiting for the occupied queue permit"
    );
    drop(decode);
    gate.open().await;
    queued.await;
    assert_cancelled(executor, &inputs, &owners).await;
    close_fixture(fixture);
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_cancel_queued_shared_cohort() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let executor = &fixture.executor;
    let ids = [RequestId::new(), RequestId::new()];
    let caches = [
        admit(&fixture, &ids[0]).await,
        admit(&fixture, &ids[1]).await,
    ];
    let owners = caches
        .iter()
        .map(|frontier| sequence(executor, frontier))
        .collect::<Vec<_>>();
    let inputs = [input(&caches[1], 3), input(&caches[0], 1)];
    let gate = WorkerGate::closed(&executor.completion_worker).await;
    let scheduled = executor.completion_worker.metrics_snapshot()["scheduled_tasks"]
        .as_u64()
        .unwrap();
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    let mut decode = Box::pin(executor.execute_tokens(&inputs));
    assert!(futures::poll!(decode.as_mut()).is_pending());
    assert_submitted_waiting(executor, before, &owners);
    assert_eq!(
        executor.completion_worker.metrics_snapshot()["scheduled_tasks"],
        scheduled + 1,
        "the shared wave's real readback job is queued behind the closed gate"
    );
    drop(decode);
    gate.open().await;
    assert_cancelled(executor, &inputs, &owners).await;
    close_fixture(fixture);
}

#[tokio::test]
async fn sampling_readback_cpu_lifecycle_cancel_delivered_unpolled_result() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let executor = &fixture.executor;
    let ids = [RequestId::new(), RequestId::new()];
    let caches = [
        admit(&fixture, &ids[0]).await,
        admit(&fixture, &ids[1]).await,
    ];
    let owners = caches
        .iter()
        .map(|frontier| sequence(executor, frontier))
        .collect::<Vec<_>>();
    let inputs = [input(&caches[1], 3), input(&caches[0], 1)];
    let gate = WorkerGate::closed(&executor.completion_worker).await;
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    let readbacks_before = executor.completion_worker.metrics_snapshot()["task_classes"]
        ["wave_readback"]["completed_tasks"]
        .as_u64()
        .unwrap();
    let mut decode = Box::pin(executor.execute_tokens(&inputs));
    assert!(futures::poll!(decode.as_mut()).is_pending());
    assert_submitted_waiting(executor, before, &owners);
    let mut delivered = marker(&executor.completion_worker);
    assert!(futures::poll!(delivered.as_mut()).is_pending());
    gate.open().await;
    delivered.await;
    // FIFO proves the actual result sender already ran; decode has not been
    // polled again and therefore cannot consume the delivered owner envelope.
    assert_eq!(
        executor.completion_worker.metrics_snapshot()["task_classes"]["wave_readback"]
            ["completed_tasks"],
        readbacks_before + 1
    );
    for owner in &owners {
        assert_eq!(*owner.upgrade().unwrap().tokens.lock(), [0]);
    }
    drop(decode);
    assert_cancelled(executor, &inputs, &owners).await;
    close_fixture(fixture);
}
