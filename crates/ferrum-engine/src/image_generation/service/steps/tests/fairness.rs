//! Candidate callbacks expose deterministic queue boundaries around the real
//! Stored DiT consumer. They never replace native admission or forward work.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hold {
    Prepare(u64),
    NativeCandidate(u64),
}

#[derive(Default)]
struct ControlState {
    entered: bool,
    released: bool,
    // The second field counts completed real Stored forwards, not callbacks.
    prepared: Vec<(u64, usize)>,
}

struct Control {
    hold: Hold,
    state: Mutex<ControlState>,
    changed: Condvar,
    arrived: tokio::sync::Notify,
}

impl Control {
    fn new(hold: Hold) -> Self {
        Self {
            hold,
            state: Mutex::new(ControlState::default()),
            changed: Condvar::new(),
            arrived: tokio::sync::Notify::new(),
        }
    }

    fn hold_if(&self, point: Hold) {
        if self.hold != point {
            return;
        }
        let mut state = self.state.lock().unwrap();
        if state.entered {
            return;
        }
        state.entered = true;
        self.arrived.notify_one();
        while !state.released {
            state = self.changed.wait(state).unwrap();
        }
    }

    async fn wait(&self) {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let ready = self.arrived.notified();
                if self.state.lock().unwrap().entered {
                    break;
                }
                ready.await;
            }
        })
        .await
        .expect("candidate reaches the controlled preparation boundary");
    }

    fn release(&self) {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .released = true;
        self.changed.notify_all();
    }

    fn prepared(&self) -> Vec<(u64, usize)> {
        self.state.lock().unwrap().prepared.clone()
    }
}

struct ReleaseControlOnDrop(Arc<Control>);
impl Drop for ReleaseControlOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

struct FairnessGenerator {
    inner: Preconditioned,
    control: Arc<Control>,
}

impl ImageGenerator for FairnessGenerator {
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        Some(self)
    }

    fn host_account(&self) -> &HostMemoryAccount {
        self.inner.host_account()
    }

    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        self.inner.capabilities()
    }

    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        unreachable!("declared step generator must use the actual step consumer")
    }

    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        ImageGenerator::shutdown(&mut self.inner)
    }
}

impl ImageDenoisingGenerator for FairnessGenerator {
    type Output = ImageGenerationOutput;

    fn prepare_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        let completed_forwards = self.inner.trace.lock().unwrap().len();
        self.control
            .state
            .lock()
            .unwrap()
            .prepared
            .push((request.seed, completed_forwards));
        self.control.hold_if(Hold::Prepare(request.seed));
        match request.seed {
            100 => Err(ImageBatchError::Rejected(
                "fixture rejects this candidate metadata".into(),
            )),
            101 => Err(ImageBatchError::Overloaded(
                "fixture cannot admit this candidate metadata".into(),
            )),
            _ => self.inner.prepare_request(request),
        }
    }

    fn try_admit_request(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.inner.try_admit_request(prepared, resources)
    }

    fn start_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        self.inner.start_request(prepared, request)
    }

    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut ImageDenoisingRequest],
        previous: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        let outcome = self.inner.try_prepare_batch(requests, previous)?;
        if outcome.is_some() && requests.len() > 1 {
            for request in requests.iter() {
                let (seed, _, _) = request
                    .get::<ContinuousRequest<Runtime>>()?
                    .fixture_snapshot();
                self.control.hold_if(Hold::NativeCandidate(seed));
            }
        }
        Ok(outcome)
    }

    fn pause_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        self.inner.pause_batch(batch, requests)
    }

    fn advance_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        self.inner.advance_batch(batch, requests)
    }

    fn finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        self.inner.finish_request(request, batch)
    }
}

fn start_controlled(
    hold: Hold,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<Control>,
) {
    let (pipeline, source) = fixture_pipeline();
    let runtime = pipeline.execution.runtime.clone();
    let boundary = Arc::new(Boundary::default());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let control = Arc::new(Control::new(hold));
    let service = ImageService::new(
        FairnessGenerator {
            inner: Preconditioned {
                pipeline,
                _source: source,
                boundary: boundary.clone(),
                trace: trace.clone(),
                denied_native_join: Arc::new(AtomicBool::new(false)),
            },
            control: control.clone(),
        },
        ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            maximum_pending: 4,
            maximum_pending_bytes: 1 << 18,
            maximum_prompt_bytes: 1024,
            max_num_seqs: 2,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    (service, boundary, trace, runtime, control)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_fairness_rejected_metadata_cannot_consume_the_next_active_forward() {
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    let (service, boundary, trace, runtime, control) = start_controlled(Hold::Prepare(100));
    let _release = ReleaseOnDrop(boundary.clone());
    let _release_control = ReleaseControlOnDrop(control.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let rejected = service.try_submit(request(100)).unwrap();
    boundary.release(1);
    control.wait().await;
    // The second invalid candidate is already queued before the first returns.
    let overloaded = service.try_submit(request(101)).unwrap();
    let b = service.try_submit(request(1)).unwrap();
    control.release();
    boundary.wait(2).await;
    boundary.release(2);
    let rejected = match rejected.wait().await {
        Err(error) => error,
        Ok(_) => panic!("rejected metadata must not execute"),
    };
    let overloaded = match overloaded.wait().await {
        Err(error) => error,
        Ok(_) => panic!("overloaded metadata must not execute"),
    };
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    assert_eq!([a.receipt.seed, b.receipt.seed], [0, 1]);
    check(&trace.lock().unwrap(), &independent);
    drop((a, b));
    finish(&service, &runtime).await;
    assert_eq!(rejected.kind(), ImageServiceErrorKind::Generation);
    assert_eq!(overloaded.kind(), ImageServiceErrorKind::QueueFull);
    let attempts = control
        .prepared()
        .into_iter()
        .filter(|(seed, _)| *seed == 100 || *seed == 101)
        .collect::<Vec<_>>();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0], (100, 1));
    assert_eq!(attempts[1].0, 101);
    assert!(
        attempts[1].1 >= 2,
        "the active real DiT must advance between candidate preparations: {attempts:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_fairness_cancelled_published_join_cannot_reset_the_active_forward_budget() {
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    let (service, boundary, trace, runtime, control) = start_controlled(Hold::NativeCandidate(1));
    let _release = ReleaseOnDrop(boundary.clone());
    let _release_control = ReleaseControlOnDrop(control.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    // This is after the actual (A,B) native admission succeeded, before the
    // worker receives its successful batch and notices B's cancellation.
    control.wait().await;
    b.cancel();
    let c = service.try_submit(request(2)).unwrap();
    control.release();
    boundary.wait(2).await;
    boundary.release(2);
    let cancelled = match b.wait().await {
        Err(error) => error,
        Ok(_) => panic!("cancelled joined request must not deliver output"),
    };
    let a = a.wait().await.unwrap();
    let c = c.wait().await.unwrap();
    assert_eq!([a.receipt.seed, c.receipt.seed], [0, 2]);
    check(&trace.lock().unwrap(), &independent);
    drop((a, c));
    finish(&service, &runtime).await;
    assert_eq!(cancelled.kind(), ImageServiceErrorKind::Cancelled);
    let attempts = control.prepared();
    let (_, completed_forwards) = attempts
        .iter()
        .find(|(seed, _)| *seed == 2)
        .expect("the subsequent FIFO candidate must eventually be prepared");
    assert!(
        *completed_forwards >= 2,
        "a published candidate cancellation must not renew attempts before the survivor advances: {attempts:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_fairness_legal_fifo_candidates_fill_the_original_batch_slots() {
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    let (service, boundary, trace, runtime, control) = start_controlled(Hold::Prepare(0));
    let _release = ReleaseOnDrop(boundary.clone());
    let _release_control = ReleaseControlOnDrop(control.clone());
    let a = service.try_submit(request(0)).unwrap();
    control.wait().await;
    let b = service.try_submit(request(1)).unwrap();
    control.release();
    boundary.wait(1).await;
    boundary.release(1);
    boundary.wait(2).await;
    boundary.release(2);
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    let trace = trace.lock().unwrap().clone();
    check(&trace, &independent);
    drop((a, b));
    finish(&service, &runtime).await;
    assert_eq!(control.prepared(), [(0, 0), (1, 0)]);
    assert_eq!(
        trace[0]
            .iter()
            .map(|(seed, progress, _)| (*seed, *progress))
            .collect::<Vec<_>>(),
        [(0, 1), (1, 1)],
        "legal FIFO candidates must still fill the original two free slots"
    );
}
