use super::*;
use ferrum_runtime::{HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use std::sync::{atomic::AtomicUsize, mpsc};

fn account() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            64,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "edit".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}
fn edit(host: &HostMemoryAccount, seed: u64) -> QueuedRequest {
    let mut image = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap();
    image.as_mut().copy_from_slice(b"\x89PNG\r\n\x1a\n");
    QueuedRequest::Edit(
        ImageEditRequest {
            generation: request(seed),
            image: image.freeze(),
        },
        None,
    )
}
fn config() -> ImageServiceConfig {
    ImageServiceConfig {
        batching: ferrum_types::ImageBatchPolicy::Configured,
        maximum_pending: 4,
        maximum_pending_bytes: 1024 * 1024,
        maximum_prompt_bytes: 1024,
        max_num_seqs: 2,
        request_batch_max_wait_ms: 10,
    }
}
struct Worker {
    host: HostMemoryAccount,
    calls: Arc<Mutex<Vec<Vec<u64>>>>,
    entered: Option<mpsc::Sender<()>>,
    release: Option<mpsc::Receiver<()>>,
    failure: Option<Arc<AtomicUsize>>,
}
impl BatchWorker for Worker {
    type Output = u64;
    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        let seeds = requests.into_iter().map(|r| r.seed).collect::<Vec<_>>();
        self.calls.lock().unwrap().push(seeds.clone());
        Ok(seeds)
    }
    fn edit(&mut self, request: ImageEditRequest) -> Result<u64, ImageBatchError> {
        assert!(self.host.owns(&request.image));
        assert_eq!(request.image.bytes(), b"\x89PNG\r\n\x1a\n");
        self.calls
            .lock()
            .unwrap()
            .push(vec![request.generation.seed]);
        if let Some(entered) = self.entered.take() {
            entered.send(()).unwrap();
        }
        if let Some(release) = self.release.take() {
            release.recv().unwrap();
        }
        if let Some(owner) = self.failure.take() {
            return Err(ImageBatchError::Fatal(Box::new(Failure(owner))));
        }
        Ok(request.generation.seed)
    }
}
#[derive(Debug)]
struct Failure(Arc<AtomicUsize>);
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("original edit failure")
    }
}
impl Error for Failure {}
impl Drop for Failure {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn qwen_input_classification_requires_owner_free_preflight_and_preserves_host_errors() {
    use crate::image_generation::{ImageGenerationError, ImageGenerationFailure};
    use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
    use ferrum_models::qwen_image21::image_input::{ImageInputError, ManagedRgba8Image};
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1024,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut png = Vec::new();
    ferrum_output::png::write_rgba8_png(
        &mut png,
        ferrum_output::png::PngDimensions::new(1, 1).unwrap(),
        ferrum_output::png::PngLimits::default(),
        |writer| writer.write_all(&[1, 2, 3, 255]),
    )
    .unwrap();
    let make_error = |source: ImageInputError| ImageGenerationError::<ReferenceDeviceRuntime> {
        failure: ImageGenerationFailure::Preflight(
            anyhow::Error::new(source).context("decode input"),
        ),
        completed_stages: Vec::new(),
        completed_steps: Vec::new(),
    };
    for oversized in [false, true] {
        let mut wire = png.clone();
        if oversized {
            wire[16..20].copy_from_slice(&8192u32.to_be_bytes());
            wire[20..24].copy_from_slice(&8192u32.to_be_bytes());
        } else {
            wire[29] ^= 1;
        }
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, wire.len())
            .unwrap();
        bytes.as_mut().copy_from_slice(&wire);
        let bytes = bytes.freeze();
        let error =
            ManagedRgba8Image::decode_png(&host, &bytes, ImageInputLimits::default()).unwrap_err();
        if oversized {
            assert!(matches!(&error, ImageInputError::Limit(_)));
        } else {
            assert!(matches!(&error, ImageInputError::Png(_)));
        }
        assert!(matches!(
            qwen_batch_error(false, make_error(error)),
            ImageBatchError::InvalidInput(_)
        ));
    }
    let host_error = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1025)
        .unwrap_err();
    assert!(matches!(
        qwen_batch_error(false, make_error(ImageInputError::Host(host_error))),
        ImageBatchError::Rejected(_)
    ));
    assert!(matches!(
        qwen_batch_error(true, make_error(ImageInputError::Invalid("bad PNG"))),
        ImageBatchError::Fatal(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn active_edit_cancellation_keeps_encoded_owner_until_execution_and_does_not_mix_batches() {
    let host = account();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (entered, started) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let service = ServiceCore::start(
        Box::new(Worker {
            host: host.clone(),
            calls: calls.clone(),
            entered: Some(entered),
            release: Some(gate),
            failure: None,
        }),
        ImageServiceConfig {
            // The default edit cap is one: a lone edit must dispatch without
            // waiting for this generation batch deadline.
            request_batch_max_wait_ms: 30_000,
            ..config()
        },
    )
    .unwrap();
    let active = service.try_submit_request(edit(&host, 7)).unwrap();
    started.recv_timeout(Duration::from_secs(2)).unwrap();
    let a = service.try_submit(request(11)).unwrap();
    let b = service.try_submit(request(12)).unwrap();
    active.cancel();
    assert_eq!(host.usage().unwrap().allocated_bytes(), 8);
    assert_eq!(service.snapshot().active_requests, 1);
    release.send(()).unwrap();
    assert_eq!(
        active.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert_eq!(a.wait().await.unwrap(), 11);
    assert_eq!(b.wait().await.unwrap(), 12);
    assert_eq!(*calls.lock().unwrap(), vec![vec![7], vec![11, 12]]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let mut retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    drop(retired.worker.take());
}

#[tokio::test]
async fn pending_edit_cancel_and_fatal_edit_preserve_their_distinct_retirement_rules() {
    let host = account();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (entered, started) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let dropped = Arc::new(AtomicUsize::new(0));
    let service = ServiceCore::start(
        Box::new(Worker {
            host: host.clone(),
            calls,
            entered: Some(entered),
            release: Some(gate),
            failure: Some(dropped.clone()),
        }),
        config(),
    )
    .unwrap();
    let active = service.try_submit_request(edit(&host, 1)).unwrap();
    started.recv_timeout(Duration::from_secs(2)).unwrap();
    let pending = service.try_submit_request(edit(&host, 2)).unwrap();
    pending.cancel();
    let later = service.try_submit(request(3)).unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    assert_eq!(
        pending.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    release.send(()).unwrap();
    assert_eq!(
        active.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Generation
    );
    assert_eq!(
        later.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Stopped
    );
    let mut retired = service.shutdown().await.unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    match retired.failure.as_ref().unwrap() {
        ImageWorkerFailure::Generation(error) => assert!(error.downcast_ref::<Failure>().is_some()),
        _ => panic!("wrong error authority"),
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(retired.failure.take());
    drop(retired.worker.take());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

fn batch_account() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            128 * 1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 128 * 1024,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn png_edit(
    host: &HostMemoryAccount,
    seed: u64,
    pixel: u8,
    width: u32,
    height: u32,
) -> ImageEditRequest {
    let mut encoded = Vec::new();
    ferrum_output::png::write_rgba8_png(
        &mut encoded,
        ferrum_output::png::PngDimensions::new(u64::from(width), u64::from(height)).unwrap(),
        ferrum_output::png::PngLimits::default(),
        |writer| {
            for _ in 0..width * height {
                writer.write_all(&[pixel, 2, 3, 255])?;
            }
            Ok(())
        },
    )
    .unwrap();
    let mut image = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, encoded.len())
        .unwrap();
    image.as_mut().copy_from_slice(&encoded);
    ImageEditRequest {
        generation: request(seed),
        image: image.freeze(),
    }
}

#[derive(Clone, Copy)]
enum BatchFailure {
    None,
    Fatal,
    Panic,
    WrongCount,
}
struct EditBatchWorker {
    host: HostMemoryAccount,
    calls: Arc<Mutex<Vec<Vec<(u64, u8)>>>>,
    entered: Option<mpsc::Sender<()>>,
    release: Option<mpsc::Receiver<()>>,
    failure: BatchFailure,
    error_dropped: Arc<AtomicUsize>,
    worker_dropped: Arc<AtomicUsize>,
}
impl EditBatchWorker {
    fn new(host: &HostMemoryAccount) -> Self {
        Self {
            host: host.clone(),
            calls: Arc::new(Mutex::new(Vec::new())),
            entered: None,
            release: None,
            failure: BatchFailure::None,
            error_dropped: Arc::new(AtomicUsize::new(0)),
            worker_dropped: Arc::new(AtomicUsize::new(0)),
        }
    }
}
impl Drop for EditBatchWorker {
    fn drop(&mut self) {
        self.worker_dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl BatchWorker for EditBatchWorker {
    type Output = (u64, u8);
    fn maximum_edit_batch_size(&self) -> usize {
        2
    }
    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }
    fn compatible_edits(&self, a: &ImageEditRequest, b: &ImageEditRequest) -> bool {
        compatible_edit_geometry(a, b)
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        let outputs = requests
            .into_iter()
            .map(|r| (r.seed, 0))
            .collect::<Vec<_>>();
        self.calls.lock().unwrap().push(outputs.clone());
        Ok(outputs)
    }
    fn edit_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<Self::Output>, ImageBatchError> {
        if let Some(entered) = self.entered.take() {
            entered.send(()).unwrap();
        }
        if let Some(release) = self.release.take() {
            release.recv().unwrap();
        }
        match self.failure {
            BatchFailure::Fatal => {
                return Err(ImageBatchError::Fatal(Box::new(Failure(
                    self.error_dropped.clone(),
                ))))
            }
            BatchFailure::Panic => std::panic::panic_any(Failure(self.error_dropped.clone())),
            BatchFailure::WrongCount => return Ok(Vec::new()),
            BatchFailure::None => {}
        }
        let outputs = requests
            .into_iter()
            .map(|r| {
                // Read the actual independently owned reference, not a request ID
                // standing in for image payload. Full decode is deliberate here.
                let decoded =
                    ferrum_models::qwen_image21::image_input::ManagedRgba8Image::decode_png(
                        &self.host,
                        &r.image,
                        ImageInputLimits::default(),
                    )
                    .unwrap();
                (r.generation.seed, decoded.bytes().bytes()[0])
            })
            .collect::<Vec<_>>();
        self.calls.lock().unwrap().push(outputs.clone());
        Ok(outputs)
    }
}

#[tokio::test]
async fn edit_batches_preserve_distinct_images_fifo_cancellation_and_queue_quota() {
    let host = batch_account();
    let mut worker = EditBatchWorker::new(&host);
    let calls = worker.calls.clone();
    let (entered, started) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    worker.entered = Some(entered);
    worker.release = Some(gate);
    let service = ServiceCore::start(
        Box::new(worker),
        ImageServiceConfig {
            request_batch_max_wait_ms: 30_000,
            maximum_pending: 3,
            ..config()
        },
    )
    .unwrap();
    let a = service
        .try_submit_request(QueuedRequest::Edit(png_edit(&host, 11, 17, 2, 2), None))
        .unwrap();
    let b = service
        .try_submit_request(QueuedRequest::Edit(png_edit(&host, 29, 203, 4, 4), None))
        .unwrap();
    started.recv_timeout(Duration::from_secs(2)).unwrap();
    let active_bytes = host.usage().unwrap().committed_bytes();
    a.cancel();
    assert_eq!(service.snapshot().active_requests, 2);
    assert_eq!(host.usage().unwrap().committed_bytes(), active_bytes);
    let generated = service.try_submit(request(50)).unwrap();
    let pending = service
        .try_submit_request(QueuedRequest::Edit(png_edit(&host, 99, 5, 2, 2), None))
        .unwrap();
    let c = service
        .try_submit_request(QueuedRequest::Edit(png_edit(&host, 31, 41, 2, 2), None))
        .unwrap();
    assert_eq!(
        service
            .try_submit_request(QueuedRequest::Edit(png_edit(&host, 100, 6, 2, 2), None))
            .err()
            .unwrap()
            .kind(),
        ImageServiceErrorKind::QueueFull
    );
    let before_cancel = host.usage().unwrap().committed_bytes();
    pending.cancel();
    let d = service
        .try_submit_request(QueuedRequest::Edit(png_edit(&host, 32, 42, 2, 2), None))
        .unwrap();
    // Pruning returned the cancelled backing before admitting this replacement.
    assert_eq!(host.usage().unwrap().committed_bytes(), before_cancel);
    assert_eq!(
        pending.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    release.send(()).unwrap();
    assert_eq!(
        a.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert_eq!(b.wait().await.unwrap(), (29, 203));
    assert_eq!(generated.wait().await.unwrap(), (50, 0));
    assert_eq!(c.wait().await.unwrap(), (31, 41));
    assert_eq!(d.wait().await.unwrap(), (32, 42));
    assert_eq!(
        *calls.lock().unwrap(),
        vec![
            vec![(11, 17), (29, 203)],
            vec![(50, 0)],
            vec![(31, 41), (32, 42)]
        ]
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(service.shutdown().await.unwrap());
}

#[tokio::test]
async fn reference_geometry_and_output_shape_partition_edit_batches() {
    for different_output in [false, true] {
        let host = batch_account();
        let worker = EditBatchWorker::new(&host);
        let calls = worker.calls.clone();
        let service = ServiceCore::start(Box::new(worker), config()).unwrap();
        let first = png_edit(&host, 1, 11, 2, 2);
        let mut second = png_edit(&host, 2, 22, 2, if different_output { 2 } else { 4 });
        if different_output {
            second.generation.width = 64;
        }
        let a = service
            .try_submit_request(QueuedRequest::Edit(first, None))
            .unwrap();
        let b = service
            .try_submit_request(QueuedRequest::Edit(second, None))
            .unwrap();
        assert_eq!(a.wait().await.unwrap(), (1, 11));
        assert_eq!(b.wait().await.unwrap(), (2, 22));
        assert_eq!(*calls.lock().unwrap(), vec![vec![(1, 11)], vec![(2, 22)]]);
        drop(service.shutdown().await.unwrap());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn png_geometry_hint_is_bounded_and_never_substitutes_for_strict_decode() {
    use ferrum_models::qwen_image21::image_input::ManagedRgba8Image;
    let host = batch_account();
    let valid = png_edit(&host, 1, 7, 2, 2);
    assert_eq!(
        png_dimensions_hint(valid.image.bytes(), ImageInputLimits::default()),
        Some((2, 2))
    );
    for length in 0..33 {
        assert_eq!(
            png_dimensions_hint(&valid.image.bytes()[..length], ImageInputLimits::default()),
            None
        );
    }
    let mut corrupt = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, valid.image.len())
        .unwrap();
    corrupt.as_mut().copy_from_slice(valid.image.bytes());
    corrupt.as_mut()[29] ^= 1; // unchanged IHDR geometry, invalid CRC
    let corrupt = corrupt.freeze();
    assert_eq!(
        png_dimensions_hint(corrupt.bytes(), ImageInputLimits::default()),
        Some((2, 2))
    );
    assert!(ManagedRgba8Image::decode_png(&host, &corrupt, ImageInputLimits::default()).is_err());
    let invalid = match edit(&host, 2) {
        QueuedRequest::Edit(r, _) => r,
        _ => unreachable!(),
    };
    assert!(!compatible_edit_geometry(&valid, &invalid));
    assert!(!compatible_edit_geometry(&invalid, &invalid));
}

#[tokio::test]
async fn batch_edit_fatal_panic_and_wrong_output_count_preserve_worker_retirement() {
    for failure in [
        BatchFailure::Fatal,
        BatchFailure::Panic,
        BatchFailure::WrongCount,
    ] {
        let host = batch_account();
        let mut worker = EditBatchWorker::new(&host);
        worker.failure = failure;
        let error_dropped = worker.error_dropped.clone();
        let worker_dropped = worker.worker_dropped.clone();
        let service = ServiceCore::start(
            Box::new(worker),
            ImageServiceConfig {
                request_batch_max_wait_ms: 30_000,
                ..config()
            },
        )
        .unwrap();
        let a = service
            .try_submit_request(QueuedRequest::Edit(png_edit(&host, 1, 7, 2, 2), None))
            .unwrap();
        let b = service
            .try_submit_request(QueuedRequest::Edit(png_edit(&host, 2, 8, 2, 2), None))
            .unwrap();
        assert!(a.wait().await.is_err());
        assert!(b.wait().await.is_err());
        assert!(!service.snapshot().accepting);
        let mut retired = service.shutdown().await.unwrap();
        assert!(retired.failure.is_some());
        assert_eq!(worker_dropped.load(Ordering::SeqCst), 0);
        assert_eq!(error_dropped.load(Ordering::SeqCst), 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        drop(retired.failure.take());
        drop(retired.worker.take());
        assert_eq!(worker_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(
            error_dropped.load(Ordering::SeqCst),
            usize::from(!matches!(failure, BatchFailure::WrongCount))
        );
    }
}

struct LegacyGenerator {
    host: HostMemoryAccount,
    calls: Arc<AtomicUsize>,
}
impl ImageGenerator for LegacyGenerator {
    fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    fn supports_edits(&self) -> bool {
        true
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        Ok(ImageGeneratorCapabilities {
            maximum_batch_size: 4,
            maximum_width: 1024,
            maximum_height: 1024,
            dimension_multiple: 32,
        })
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        Err(ImageBatchError::Rejected(
            "unused generation fixture".into(),
        ))
    }
    fn edit(&mut self, _: ImageEditRequest) -> Result<ImageGenerationOutput, ImageBatchError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(ImageBatchError::Rejected(
            "owner-free legacy edit rejection".into(),
        ))
    }
}

#[tokio::test]
async fn legacy_generator_keeps_single_edit_cap_and_delegates_without_batch_wait() {
    let host = batch_account();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut generator = LegacyGenerator {
        host: host.clone(),
        calls: calls.clone(),
    };
    assert_eq!(generator.maximum_edit_batch_size(), 1);
    assert!(generator
        .edit_batch(vec![
            png_edit(&host, 1, 1, 2, 2),
            png_edit(&host, 2, 2, 2, 2)
        ])
        .is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let service = ImageService::new(
        generator,
        ImageServiceConfig {
            max_num_seqs: 4,
            request_batch_max_wait_ms: 30_000,
            ..config()
        },
    )
    .unwrap();
    assert_eq!(service.maximum_edit_batch_size(), 1);
    let job = service
        .try_submit_edit(png_edit(&host, 3, 3, 2, 2))
        .unwrap();
    let error = tokio::time::timeout(Duration::from_secs(2), job.wait())
        .await
        .unwrap()
        .err()
        .expect("legacy request rejection");
    assert_eq!(error.kind(), ImageServiceErrorKind::Generation);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(service.is_accepting());
    assert!(!service.shutdown().await.unwrap().has_failure());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
