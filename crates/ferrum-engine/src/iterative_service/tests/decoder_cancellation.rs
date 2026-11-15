use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

/// A real held Output allocation represents a decoder phase after its
/// denoiser has retired. No native/GPU qualification is claimed by this fixture.
struct DecoderBoundaryWorker {
    timeline: TimelineWorker,
    host: HostMemoryAccount,
    entered: Option<mpsc::Sender<()>>,
    finish: mpsc::Receiver<()>,
    suspended: bool,
}
impl BatchWorker for DecoderBoundaryWorker {
    type Request = TimelineRequest;
    type Output = TimelineOutput;
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn DenoisingGenerator<Request = Self::Request, Output = Self::Output>> {
        Some(self)
    }
    fn execute(
        &mut self,
        _: BatchPreparation,
        _: Vec<Self::Request>,
    ) -> Result<Vec<Self::Output>, BatchError> {
        panic!("iterative protocol required")
    }
    fn compatible(&self, a: &Self::Request, b: &Self::Request) -> bool {
        self.timeline.compatible(a, b)
    }
    fn maximum_batch_size(&self, r: &Self::Request, configured: usize) -> usize {
        self.timeline.maximum_batch_size(r, configured)
    }
    fn shutdown_evidence(&mut self) -> Result<Option<Box<dyn Any + Send>>, GeneratorError> {
        self.timeline.shutdown_evidence()
    }
}
impl DenoisingGenerator for DecoderBoundaryWorker {
    type Request = TimelineRequest;
    type Output = TimelineOutput;
    fn supports_request(&mut self, r: &Self::Request) -> bool {
        self.timeline.supports_request(r)
    }
    fn prepare_request(&mut self, r: &Self::Request) -> Result<BatchPreparation, BatchError> {
        self.timeline.prepare_request(r)
    }
    fn bind_request_cancellation(&mut self, p: &mut BatchPreparation, c: Arc<AtomicBool>) {
        self.timeline.bind_request_cancellation(p, c);
    }
    fn try_admit_request(
        &mut self,
        p: &mut BatchPreparation,
        r: PreparationResources,
    ) -> Result<bool, BatchError> {
        self.timeline.try_admit_request(p, r)
    }
    fn start_request(
        &mut self,
        p: BatchPreparation,
        r: &Self::Request,
    ) -> Result<DenoisingRequest, BatchError> {
        self.timeline.start_request(p, r)
    }
    fn try_prepare_batch(
        &mut self,
        r: &mut [&mut DenoisingRequest],
        p: Option<&mut DenoisingBatch>,
    ) -> Result<Option<DenoisingBatch>, BatchError> {
        let next = self.timeline.try_prepare_batch(r, p)?;
        self.suspended = false;
        Ok(next)
    }
    fn pause_batch(
        &mut self,
        b: &mut DenoisingBatch,
        r: &mut [&mut DenoisingRequest],
    ) -> Result<(), BatchError> {
        self.timeline.pause_batch(b, r)
    }
    fn advance_batch(
        &mut self,
        b: &mut DenoisingBatch,
        r: &mut [&mut DenoisingRequest],
    ) -> Result<Vec<bool>, BatchError> {
        self.timeline.advance_batch(b, r)
    }
    fn batch_is_suspended(&mut self, _: &DenoisingBatch) -> bool {
        self.suspended
    }
    fn restore_suspended_batch(
        &mut self,
        b: &mut DenoisingBatch,
        r: &mut [&mut DenoisingRequest],
    ) -> Result<(), BatchError> {
        let actual = r
            .iter()
            .map(|r| r.get::<Lane>().map(|r| r.output.id))
            .collect::<Result<Vec<_>, _>>()?;
        if actual.is_empty() || &actual != b.get::<Vec<u64>>()? {
            return Err(BatchError::Rejected(
                "cannot restore absent or different execution members".into(),
            ));
        }
        self.suspended = false;
        Ok(())
    }
    fn finish_request(
        &mut self,
        r: &mut DenoisingRequest,
        b: &mut DenoisingBatch,
    ) -> Result<Self::Output, BatchError> {
        self.timeline.finish_request(r, b)
    }
    fn try_finish_request(
        &mut self,
        r: &mut DenoisingRequest,
        b: &mut DenoisingBatch,
    ) -> Result<Option<Self::Output>, BatchError> {
        if let Some(entered) = self.entered.take() {
            self.suspended = true;
            let owner = self
                .host
                .allocate_zeroed(HostMemoryClass::Output, 128)
                .unwrap();
            entered.send(()).unwrap();
            self.finish.recv_timeout(Duration::from_secs(10)).unwrap();
            let cancelled = r
                .get::<Lane>()?
                .cancelled
                .as_ref()
                .unwrap()
                .load(Ordering::Acquire);
            // None is published only after the actual phase resource retires.
            drop(owner);
            if cancelled {
                return Ok(None);
            }
        }
        self.timeline.finish_request(r, b).map(Some)
    }
}

#[tokio::test]
async fn last_decoder_cancelled_after_its_fence_does_not_restore_empty_members_or_stop_service() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 1024,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let (entered, at_decoder) = mpsc::channel();
    let (finish, finished) = mpsc::channel();
    let (_, release) = mpsc::channel();
    let memberships = Arc::new(Mutex::new(Vec::new()));
    let worker = DecoderBoundaryWorker {
        timeline: TimelineWorker {
            first_boundary: None,
            release,
            memberships: memberships.clone(),
            observed: Arc::new(Observed::default()),
            cancellation_seen: None,
        },
        host: host.clone(),
        entered: Some(entered),
        finish: finished,
        suspended: false,
    };
    let service = ServiceCore::start(
        Box::new(worker),
        ServiceConfig {
            batching: BatchPolicy::Auto,
            maximum_pending: 2,
            maximum_pending_bytes: 4096,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    let request = |id| TimelineRequest {
        id,
        frames: vec![10],
        transitions: 1,
    };
    let first = service.try_submit(request(1)).unwrap();
    at_decoder.recv_timeout(Duration::from_secs(10)).unwrap();
    first.cancel();
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        128,
        "cancellation must not retire a live decoder resource"
    );
    finish.send(()).unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(10), first.wait())
        .await
        .unwrap();
    assert_eq!(outcome.unwrap_err().kind(), ServiceErrorKind::Cancelled);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(service.snapshot().accepting);
    let second = service.try_submit(request(2)).unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(10), second.wait())
            .await
            .unwrap()
            .unwrap(),
        TimelineOutput {
            id: 2,
            frames: vec![11]
        }
    );
    assert_eq!(
        *memberships.lock().unwrap(),
        vec![vec![(1, 0)], vec![(2, 0)]]
    );
    assert!(service.shutdown().await.unwrap().failure.is_none());
}
