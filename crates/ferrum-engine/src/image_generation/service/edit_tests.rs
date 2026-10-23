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
    QueuedRequest::Edit(ImageEditRequest {
        generation: request(seed),
        image: image.freeze(),
    })
}
fn config() -> ImageServiceConfig {
    ImageServiceConfig {
        maximum_pending: 4,
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
        config(),
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
