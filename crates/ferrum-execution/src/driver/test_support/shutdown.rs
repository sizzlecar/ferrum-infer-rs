//! Whole-driver assertions over the actual upper-layer CPU language fixture.
use super::*;
use crate::executor_lifecycle::{ExecutorOwner, ExecutorShutdownObserver, ExecutorShutdownPhase};

async fn retired(observer: &ExecutorShutdownObserver) {
    let mut observer = observer.clone();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if observer.snapshot().phase() == ExecutorShutdownPhase::Retired {
                return;
            }
            observer.changed().await;
        }
    })
    .await
    .expect("original program close must finish");
}

async fn live_sequence<R: DeviceRuntime>(executor: &ProgramExecutor<R>) -> ExecutionFrontier {
    let request = RequestId::new();
    let frontier = sampling::admit(executor, &request).await;
    let decoded = token_execution::execute_one(
        executor,
        &frontier,
        TokenWork::Decode {
            token: TokenId::new(3),
            position: frontier.consumed_tokens(),
        },
    )
    .await;
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) = decoded.product()
    else {
        panic!("shutdown fixture requires the real complete logits");
    };
    assert_eq!(logits, &[1.0, 1.0009765625, -1.0, -2.0]);
    decoded.next_frontier().clone()
}

pub async fn program_shutdown_retires_original_cpu_sequence_and_plan<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
) -> Arc<R> {
    let runtime = Arc::clone(&executor.runtime);
    let frontier = live_sequence(&executor).await;
    let sequence = executor.sequences.lock().token_index[frontier.sequence()].clone();
    let resources = Arc::downgrade(&executor.plan_resources);
    let reaper = Arc::downgrade(&executor.reaper);
    let worker = Arc::downgrade(&executor.completion_worker.recovery_wake());
    let owner = ExecutorOwner::new(Box::new(executor));
    let scope = owner.scope().unwrap();
    let observer = owner.begin_shutdown();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
    assert!(sequence.upgrade().is_some());
    drop(scope);
    retired(&observer).await;
    observer.snapshot().result().unwrap();
    assert!(sequence.upgrade().is_none());
    assert!(resources.upgrade().is_none());
    assert!(reaper.upgrade().is_none());
    assert!(
        worker.upgrade().is_none(),
        "original completion worker has joined"
    );
    drop(owner);
    runtime
}

struct ReleaseWorker(Option<std::sync::mpsc::Sender<()>>);

impl Drop for ReleaseWorker {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}

pub async fn program_shutdown_joins_cancelled_cpu_submission_after_waiter_drop<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
) -> Arc<R> {
    let runtime = Arc::clone(&executor.runtime);
    let frontier = live_sequence(&executor).await;
    let sequence = executor.sequences.lock().token_index[frontier.sequence()].clone();
    let resources = Arc::downgrade(&executor.plan_resources);
    let worker = Arc::downgrade(&executor.completion_worker.recovery_wake());
    let (started, began) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let release = ReleaseWorker(Some(release));
    let mut blocker = Box::pin(executor.completion_worker.execute(
        VNextCompletionTaskKind::PostSubmitDrain,
        move || {
            let _ = started.send(());
            let _ = released.recv();
        },
    ));
    assert!(futures::poll!(blocker.as_mut()).is_pending());
    began.await.unwrap();
    // Dropping this observer does not release the accepted worker job.
    drop(blocker);
    let work = [ExecutionWork::new(
        &frontier,
        TokenWork::Decode {
            token: TokenId::new(1),
            position: frontier.consumed_tokens(),
        },
        &LogitsReturnPolicy::FullLogits,
    )
    .unwrap()];
    let before = executor.metrics.submitted_waves.load(Ordering::Relaxed);
    let mut submitted = Box::pin(executor.execute_tokens(&work));
    assert!(futures::poll!(submitted.as_mut()).is_pending());
    assert_eq!(
        executor.metrics.submitted_waves.load(Ordering::Relaxed),
        before + 1
    );
    assert_eq!(executor.reaper.retained_count(), 1);
    assert_eq!(executor.lane.in_flight_count(), 1);
    drop(submitted);
    let owner = ExecutorOwner::new(Box::new(executor));
    let observer = owner.begin_shutdown();
    drop(owner);
    // Cancel the close wait while the real readback remains queued behind an
    // accepted task. The shutdown service must retain both original owners.
    assert!(
        tokio::time::timeout(Duration::from_millis(25), observer.wait())
            .await
            .is_err()
    );
    assert!(sequence.upgrade().is_some());
    assert!(resources.upgrade().is_some());
    assert!(worker.upgrade().is_some());
    drop(release);
    retired(&observer).await;
    observer.snapshot().result().unwrap();
    assert!(sequence.upgrade().is_none());
    assert!(resources.upgrade().is_none());
    assert!(worker.upgrade().is_none());
    runtime
}

pub async fn program_shutdown_retains_referenced_cpu_plan_until_original_retry<R: DeviceRuntime>(
    executor: ProgramExecutor<R>,
) -> Arc<R> {
    let runtime = Arc::clone(&executor.runtime);
    let _frontier = live_sequence(&executor).await;
    let external = Arc::clone(&executor.plan_resources);
    let resources = Arc::downgrade(&external);
    let worker = Arc::downgrade(&executor.completion_worker.recovery_wake());
    let owner = ExecutorOwner::new(Box::new(executor));
    let observer = owner.begin_shutdown();
    let report = tokio::time::timeout(Duration::from_secs(5), observer.wait())
        .await
        .unwrap();
    assert_eq!(report.phase(), ExecutorShutdownPhase::Retained);
    assert!(report
        .error()
        .unwrap()
        .to_string()
        .contains("program plan retains"));
    assert!(external.is_closing());
    assert!(resources.upgrade().is_some());
    assert!(
        worker.upgrade().is_none(),
        "worker joins before plan finalization"
    );
    drop(owner);
    drop(external);
    observer.retry();
    retired(&observer).await;
    assert!(
        resources.upgrade().is_none(),
        "retry closes that same plan allocation"
    );
    // Retirement is proved; the earlier failed close observation stays visible.
    assert!(observer.snapshot().result().is_err());
    runtime
}
