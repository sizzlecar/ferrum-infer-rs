use super::*;
use std::sync::{atomic::AtomicUsize, mpsc};

struct WaitingWorker {
    control: Option<TensorAdmissionControl>,
    entered: mpsc::Sender<TensorAdmissionControl>,
    calls: Arc<AtomicUsize>,
    closed: Arc<AtomicUsize>,
}

impl BatchWorker for WaitingWorker {
    type Output = u64;
    fn set_admission_control(&mut self, control: TensorAdmissionControl) {
        self.control = Some(control);
    }
    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        let control = self
            .control
            .take()
            .expect("worker receives an explicit batch control");
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            self.entered.send(control.clone()).unwrap();
            futures::executor::block_on(control.cancelled());
            return Err(ImageBatchError::Cancelled);
        }
        assert!(
            !control.is_cancelled(),
            "next independent batch needs a fresh control"
        );
        Ok(requests.into_iter().map(|request| request.seed).collect())
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn waiting_service(
    batch: usize,
) -> (
    ServiceCore<u64>,
    mpsc::Receiver<TensorAdmissionControl>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let (entered, observed) = mpsc::channel();
    let calls = Arc::new(AtomicUsize::new(0));
    let closed = Arc::new(AtomicUsize::new(0));
    let service = ServiceCore::start(
        Box::new(WaitingWorker {
            control: None,
            entered,
            calls: calls.clone(),
            closed: closed.clone(),
        }),
        ImageServiceConfig {
            maximum_pending: 4,
            maximum_prompt_bytes: 1024,
            max_num_seqs: batch,
            request_batch_max_wait_ms: if batch == 1 { 0 } else { 30_000 },
        },
    )
    .unwrap();
    (service, observed, calls, closed)
}

fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "independent request".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}

#[tokio::test]
async fn one_cancel_preserves_batch_geometry_all_cancel_wakes_and_next_batch_succeeds() {
    let (service, entered, calls, closed) = waiting_service(2);
    let first = service.try_submit(request(1)).unwrap();
    let second = service.try_submit(request(2)).unwrap();
    let control = entered.recv_timeout(Duration::from_secs(2)).unwrap();
    first.cancel();
    assert!(
        !control.is_cancelled(),
        "one remaining request still owns the original batch execution"
    );
    assert_eq!(service.snapshot().active_requests, 2);
    second.cancel();
    assert!(control.is_cancelled());
    assert_eq!(
        first.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert_eq!(
        second.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert!(service.snapshot().accepting);
    let third = service.try_submit(request(3)).unwrap();
    let fourth = service.try_submit(request(4)).unwrap();
    assert_eq!(third.wait().await.unwrap(), 3);
    assert_eq!(fourth.wait().await.unwrap(), 4);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    assert_eq!(closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn shutdown_wakes_active_wait_and_rejects_pending_without_reopening_admission() {
    let (service, entered, calls, closed) = waiting_service(1);
    let active = service.try_submit(request(1)).unwrap();
    let control = entered.recv_timeout(Duration::from_secs(2)).unwrap();
    let pending = service.try_submit(request(2)).unwrap();
    service.shared.stop();
    assert!(control.is_cancelled());
    assert_eq!(
        active.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert_eq!(
        pending.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Shutdown
    );
    let retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(closed.load(Ordering::SeqCst), 1);
    assert!(!service.snapshot().accepting);
}
