//! The actual service worker and Stored Reference DiT/Euler/decoder qualify
//! observation order. Initialization is the parent's preconditioned fixture;
//! this is neither an H3 model nor CUDA numerical qualification.
use super::*;

#[derive(Clone, Debug)]
enum Seen {
    Paused(Vec<(u64, usize)>),
    Advanced(Vec<(u64, usize)>),
    Exited(u64, usize, ImageDenoisingExitReason),
    Decoder(u64, usize),
}
fn metadata(request: &ImageDenoisingRequest) -> (u64, usize) {
    let (seed, step, _) = request
        .get::<ContinuousRequest<Runtime>>()
        .unwrap()
        .fixture_snapshot();
    (seed, step)
}
struct Observed {
    inner: Preconditioned,
    seen: Arc<Mutex<Vec<Seen>>>,
    deny_survivor: Arc<AtomicBool>,
}
impl ImageGenerator for Observed {
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
        panic!("observed requests must use the original step worker")
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        self.inner.shutdown()
    }
}
impl ImageDenoisingGenerator for Observed {
    type Output = ImageGenerationOutput;
    fn record_request_exit(
        &self,
        request: &ImageDenoisingRequest,
        reason: ImageDenoisingExitReason,
    ) {
        let (seed, step) = metadata(request);
        self.seen
            .lock()
            .unwrap()
            .push(Seen::Exited(seed, step, reason));
    }
    fn prepare_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.inner.prepare_request(request)
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
        // A controlled metadata-only admission refusal tests the worker's
        // finite survivor-failure branch. It never fabricates a forward or a
        // device-capacity proof, and leaves the original fenced owner intact.
        if previous.is_some() && requests.len() == 1 && self.deny_survivor.load(Ordering::Acquire) {
            return Ok(None);
        }
        self.inner.try_prepare_batch(requests, previous)
    }
    fn pause_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        self.inner.pause_batch(batch, requests)?;
        self.seen.lock().unwrap().push(Seen::Paused(
            requests.iter().map(|request| metadata(request)).collect(),
        ));
        Ok(())
    }
    fn advance_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        let result = self.inner.advance_batch(batch, requests)?;
        self.seen.lock().unwrap().push(Seen::Advanced(
            requests.iter().map(|request| metadata(request)).collect(),
        ));
        Ok(result)
    }
    fn finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        let (seed, step) = metadata(request);
        self.seen.lock().unwrap().push(Seen::Decoder(seed, step));
        self.inner.finish_request(request, batch)
    }
}

fn start_observed() -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Vec<Seen>>>,
    Arc<AtomicBool>,
    Arc<Runtime>,
) {
    let (pipeline, source) = fixture_pipeline();
    let runtime = pipeline.execution.runtime.clone();
    let boundary = Arc::new(Boundary::default());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let deny_survivor = Arc::new(AtomicBool::new(false));
    let service = ImageService::new(
        Observed {
            inner: Preconditioned {
                pipeline,
                _source: source,
                boundary: boundary.clone(),
                trace: Arc::new(Mutex::new(Vec::new())),
                denied_native_join: Arc::new(AtomicBool::new(false)),
            },
            seen: seen.clone(),
            deny_survivor: deny_survivor.clone(),
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
    (service, boundary, seen, deny_survivor, runtime)
}

fn assert_fenced_and_never_reentered(seen: &[Seen]) {
    for (index, event) in seen.iter().enumerate() {
        let Seen::Exited(seed, step, reason) = event else {
            continue;
        };
        let last_step = seen[..index]
            .iter()
            .rposition(
                |event| matches!(event, Seen::Advanced(rows) if rows.iter().any(|&(s,_)| s==*seed)),
            )
            .expect("exit follows an actual completed Stored forward");
        assert!(
            seen[last_step + 1..index].iter().any(|event| {
                matches!(event, Seen::Paused(rows) if rows.contains(&(*seed,*step)))
            }),
            "exit {seed}/{step}/{reason:?} must follow its real pause: {seen:?}"
        );
        assert!(
            !seen[index + 1..].iter().any(|event| {
                matches!(event, Seen::Advanced(rows) if rows.iter().any(|&(s,_)| s==*seed))
                    || matches!(event, Seen::Exited(s,_,_) if s==seed)
            }),
            "a withdrawn original request must not reenter or exit twice: {seen:?}"
        );
    }
    for (index, event) in seen.iter().enumerate() {
        if let Seen::Decoder(seed, step) = event {
            assert!(seen[..index].iter().any(|event| {
                matches!(event, Seen::Exited(s,p,ImageDenoisingExitReason::Completed) if s==seed && p==step)
            }), "solver completion is observed before actual decode: {seen:?}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_exit_observation_follows_real_pause_for_cancel_completion_and_refill() {
    let (service, boundary, seen, _, runtime) = start_observed();
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    // A has completed its last real forward, while B has only its first.
    // Cancellation observed at this fence wins over decoding A; B continues
    // with fresh C and neither the old A identity nor its callback reenters.
    a.cancel();
    let c = service.try_submit(request(2)).unwrap();
    boundary.release(2);
    assert_eq!(
        a.wait().await.err().expect("cancelled A").kind(),
        ImageServiceErrorKind::Cancelled
    );
    let b = b.wait().await.unwrap();
    let c = c.wait().await.unwrap();
    drop((b, c));
    finish(&service, &runtime).await;
    let seen = seen.lock().unwrap();
    assert_fenced_and_never_reentered(&seen);
    assert!(seen.iter().any(|event| matches!(
        event,
        Seen::Exited(0, 2, ImageDenoisingExitReason::Cancelled)
    )));
    for seed in [1, 2] {
        assert!(seen.iter().any(|event| matches!(event, Seen::Exited(s,2,ImageDenoisingExitReason::Completed) if *s==seed)));
    }
    assert!(seen.iter().any(
        |event| matches!(event, Seen::Advanced(rows) if rows.iter().map(|&(seed,_)|seed).eq([1,2]))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_exit_observation_reports_failed_survivor_without_poisoning_the_worker() {
    let (service, boundary, seen, deny_survivor, runtime) = start_observed();
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    deny_survivor.store(true, Ordering::Release);
    a.cancel();
    boundary.release(2);
    assert_eq!(
        a.wait().await.err().expect("cancelled A").kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert_eq!(
        b.wait().await.err().expect("unadmitted survivor B").kind(),
        ImageServiceErrorKind::Generation
    );
    deny_survivor.store(false, Ordering::Release);
    let c = service
        .try_submit(request(2))
        .unwrap()
        .wait()
        .await
        .unwrap();
    drop(c);
    finish(&service, &runtime).await;
    let seen = seen.lock().unwrap();
    assert_fenced_and_never_reentered(&seen);
    assert!(seen.iter().any(|event| matches!(
        event,
        Seen::Exited(0, 2, ImageDenoisingExitReason::Cancelled)
    )));
    assert!(seen
        .iter()
        .any(|event| matches!(event, Seen::Exited(1, 1, ImageDenoisingExitReason::Failed))));
    assert!(seen.iter().any(|event| matches!(
        event,
        Seen::Exited(2, 2, ImageDenoisingExitReason::Completed)
    )));
}
