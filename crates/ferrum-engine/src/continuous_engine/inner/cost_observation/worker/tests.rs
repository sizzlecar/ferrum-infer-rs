use super::*;

#[tokio::test]
async fn shutdown_drains_work_published_after_the_last_empty_observation() {
    let pending = Arc::new(AtomicBool::new(false));
    let drained = Arc::new(AtomicBool::new(false));
    let (empty, observed_empty) = std::sync::mpsc::channel();
    let (resume, released) = std::sync::mpsc::channel();
    let work = pending.clone();
    let completed = drained.clone();
    let mut first = true;
    let worker = Arc::new(
        CostTrainingWorker::spawn(move || {
            if first {
                first = false;
                // Latch the empty result, then allow shutdown to publish work
                // before the worker inspects its stop flag.
                let more = work.load(Ordering::Acquire);
                empty.send(()).unwrap();
                released
                    .recv_timeout(std::time::Duration::from_secs(3))
                    .unwrap();
                return more;
            }
            if work.swap(false, Ordering::AcqRel) {
                completed.store(true, Ordering::Release);
            }
            false
        })
        .unwrap(),
    );
    worker.wake();
    observed_empty
        .recv_timeout(std::time::Duration::from_secs(3))
        .unwrap();
    pending.store(true, Ordering::Release);
    let stopping = worker.clone();
    let shutdown = tokio::spawn(async move { stopping.shutdown().await });
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while !worker.shutdown_started() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    resume.send(()).unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(3), shutdown)
            .await
            .unwrap()
            .unwrap()
    );
    assert!(drained.load(Ordering::Acquire));
    assert!(!pending.load(Ordering::Acquire));
}
