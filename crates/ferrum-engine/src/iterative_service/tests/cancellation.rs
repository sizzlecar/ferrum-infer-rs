use super::*;

fn config() -> ServiceConfig {
    ServiceConfig {
        batching: BatchPolicy::Configured,
        maximum_pending: 1,
        maximum_pending_bytes: 4096,
        max_num_seqs: 1,
        request_batch_max_wait_ms: 0,
    }
}

fn request(id: u64) -> TimelineRequest {
    TimelineRequest {
        id,
        frames: vec![10],
        transitions: 3,
    }
}

fn worker(first_boundary: Option<mpsc::Sender<()>>, release: mpsc::Receiver<()>) -> TimelineWorker {
    TimelineWorker {
        first_boundary,
        release,
        memberships: Arc::new(Mutex::new(Vec::new())),
        observed: Arc::new(Observed::default()),
        cancellation_seen: None,
    }
}

#[tokio::test]
async fn detached_cancellation_reclaims_a_queued_slot_without_cancelling_other_jobs() {
    let (reached, boundary) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let service = ServiceCore::start(Box::new(worker(Some(reached), released)), config()).unwrap();
    let active = service.try_submit(request(1)).unwrap();
    boundary.recv_timeout(Duration::from_secs(10)).unwrap();
    let queued = service.try_submit(request(2)).unwrap();
    let cancel = queued.cancellation_handle();
    let waiter = tokio::spawn(queued.wait());
    assert!(
        matches!(service.try_submit(request(3)), Err(e) if e.kind() == ServiceErrorKind::QueueFull)
    );
    cancel.clone().cancel();
    let replacement = service.try_submit(request(3)).unwrap();
    assert_eq!(
        waiter.await.unwrap().unwrap_err().kind(),
        ServiceErrorKind::Cancelled
    );
    // Merely releasing cancellation authority does not cancel an accepted job.
    drop(replacement.cancellation_handle());
    release.send(()).unwrap();
    assert_eq!(active.wait().await.unwrap().id, 1);
    assert_eq!(replacement.wait().await.unwrap().id, 3);
    assert!(service.shutdown().await.unwrap().failure.is_none());
}

#[tokio::test]
async fn detached_cancellation_waits_for_the_running_step_boundary() {
    let (reached, boundary) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let mut worker = worker(Some(reached), released);
    let (seen, cancellation_seen) = mpsc::channel();
    worker.cancellation_seen = Some(seen);
    let memberships = worker.memberships.clone();
    let service = ServiceCore::start(Box::new(worker), config()).unwrap();
    let active = service.try_submit(request(7)).unwrap();
    let cancel = active.cancellation_handle();
    let waiter = tokio::spawn(active.wait());
    boundary.recv_timeout(Duration::from_secs(10)).unwrap();
    cancel.cancel();
    assert!(!waiter.is_finished());
    assert_eq!(service.snapshot().active_requests, 1);
    release.send(()).unwrap();
    let result = tokio::time::timeout(Duration::from_secs(10), waiter)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.unwrap_err().kind(), ServiceErrorKind::Cancelled);
    assert!(
        cancellation_seen
            .recv_timeout(Duration::from_secs(10))
            .unwrap(),
        "long stage must observe the original Job cancellation through its prepared/request owner"
    );
    assert_eq!(*memberships.lock().unwrap(), vec![vec![(7, 0)]]);
    assert!(service.shutdown().await.unwrap().failure.is_none());
}

#[tokio::test]
async fn detached_cancellation_cannot_revoke_a_published_result_or_retain_the_service() {
    let (_, released) = mpsc::channel();
    let worker = worker(None, released);
    let observed = worker.observed.clone();
    let service = ServiceCore::unstarted(&worker, config(), None);
    let shared = Arc::downgrade(&service.shared);
    let job = service.try_submit(request(7)).unwrap();
    let cancel = job.cancellation_handle();
    drop(cancel.clone());
    let retired = service.run_to_completion(Box::new(worker));
    assert!(retired.failure.is_none());
    cancel.cancel();
    assert_eq!(
        job.wait().await.unwrap(),
        TimelineOutput {
            id: 7,
            frames: vec![16]
        }
    );
    cancel.cancel();
    assert_eq!(service.snapshot().executed_requests, 1);
    assert_eq!(
        observed
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(
                event,
                observations::LifecycleEvent::Finished {
                    outcome: observations::RequestOutcome::Completed,
                    ..
                }
            ))
            .count(),
        1
    );
    drop(retired);
    drop(service);
    assert!(shared.upgrade().is_none());
    cancel.cancel();
}

struct BlockedOutput(mpsc::Sender<()>);
impl OutputGate for BlockedOutput {
    fn try_acquire(&self) -> Result<Option<OutputLease>, OutputGateError> {
        self.0.send(()).unwrap();
        Ok(None)
    }
    fn subscribe_capacity(&self, _: std::task::Waker) -> OutputLease {
        // This resource never becomes available: only cancellation can wake
        // the admitted job, not a synthetic resource-refund notification.
        OutputLease::new(())
    }
}

#[tokio::test]
async fn detached_cancellation_wakes_an_output_admission_wait() {
    let (_, released) = mpsc::channel();
    let worker = worker(None, released);
    let memberships = worker.memberships.clone();
    let service = ServiceCore::start(Box::new(worker), config()).unwrap();
    let (attempted, attempts) = mpsc::channel();
    let (response, _lease) = oneshot::channel();
    let job = service
        .try_submit_request_with_output(
            request(7),
            Some(QueuedOutput {
                gate: Arc::new(BlockedOutput(attempted)),
                lease: None,
                response,
            }),
        )
        .unwrap();
    let cancel = job.cancellation_handle();
    let waiter = tokio::spawn(job.wait());
    attempts.recv_timeout(Duration::from_secs(10)).unwrap();
    cancel.cancel();
    let result = tokio::time::timeout(Duration::from_secs(10), waiter)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.unwrap_err().kind(), ServiceErrorKind::Cancelled);
    assert!(memberships.lock().unwrap().is_empty());
    assert!(service.shutdown().await.unwrap().failure.is_none());
}
