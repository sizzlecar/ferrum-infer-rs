//! Typed queue/ingress ownership, not image-edit numerical qualification.
//! A valid PNG is strictly decoded only at start; the subsequent preconditioned
//! Stored Qwen DiT/VAE fixture intentionally does not claim visual conditioning.
use super::*;
use crate::image_generation::{service::ImageRequestRef, ImageEditRequest};
use ferrum_models::qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image};
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Prepared,
    Started,
    Whole,
}
#[derive(Clone, Debug)]
struct Seen {
    phase: Phase,
    seed: u64,
    pointer: usize,
    digest: [u8; 32],
}
struct PreparedEdit {
    seed: u64,
    pointer: usize,
    digest: [u8; 32],
}
fn fingerprint(edit: &ImageEditRequest) -> PreparedEdit {
    PreparedEdit {
        seed: edit.generation.seed,
        pointer: edit.image.bytes().as_ptr() as usize,
        digest: Sha256::digest(edit.image.bytes()).into(),
    }
}
struct TypedEdit {
    inner: Preconditioned,
    enabled: bool,
    seen: Arc<Mutex<Vec<Seen>>>,
}
impl TypedEdit {
    fn observe(&self, phase: Phase, edit: &ImageEditRequest) {
        let f = fingerprint(edit);
        self.seen.lock().unwrap().push(Seen {
            phase,
            seed: f.seed,
            pointer: f.pointer,
            digest: f.digest,
        });
    }
}
impl ImageGenerator for TypedEdit {
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
    fn supports_edits(&self) -> bool {
        true
    }
    fn maximum_edit_batch_size(&self) -> usize {
        2
    }
    fn compatible_edits(&self, _: &ImageEditRequest, _: &ImageEditRequest) -> bool {
        true
    }
    fn edit_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        assert!(
            !self.enabled,
            "declared step editing must use the typed continuation entrypoint"
        );
        for edit in &requests {
            self.observe(Phase::Whole, edit);
        }
        Err(ImageBatchError::Rejected(
            "whole editing callback rejected its input without device work".into(),
        ))
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        unreachable!("declared step generation")
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        ImageGenerator::shutdown(&mut self.inner)
    }
}
impl ImageDenoisingGenerator for TypedEdit {
    type Output = ImageGenerationOutput;
    fn supports_image_request(&self, input: ImageRequestRef<'_>) -> bool {
        matches!(input, ImageRequestRef::Generate(_)) || self.enabled
    }
    fn prepare_request(
        &mut self,
        r: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.inner.prepare_request(r)
    }
    fn start_request(
        &mut self,
        p: ImageBatchPreparation,
        r: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        self.inner.start_request(p, r)
    }
    fn prepare_image_request(
        &mut self,
        input: ImageRequestRef<'_>,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        match input {
            ImageRequestRef::Generate(r) => self.prepare_request(r),
            ImageRequestRef::Edit(edit) => {
                assert!(self.enabled);
                assert!(self.host_account().owns(&edit.image));
                self.observe(Phase::Prepared, edit);
                Ok(ImageBatchPreparation::new(fingerprint(edit)))
            }
        }
    }
    fn start_image_request(
        &mut self,
        p: ImageBatchPreparation,
        input: ImageRequestRef<'_>,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        match input {
            ImageRequestRef::Generate(r) => self.start_request(p, r),
            ImageRequestRef::Edit(edit) => {
                assert!(self.enabled);
                let p = p.into_inner::<PreparedEdit>()?;
                let actual = fingerprint(edit);
                assert_eq!(
                    (p.seed, p.pointer, p.digest),
                    (actual.seed, actual.pointer, actual.digest)
                );
                let decoded = ManagedRgba8Image::decode_png(
                    self.host_account(),
                    &edit.image,
                    ImageInputLimits::default(),
                )
                .unwrap();
                assert_eq!((decoded.width(), decoded.height()), (2, 2));
                drop(decoded);
                self.observe(Phase::Started, edit);
                Ok(ImageDenoisingRequest::new(fixture_request(
                    &mut self.inner.pipeline,
                    edit.generation.seed,
                )))
            }
        }
    }
    fn try_admit_request(
        &mut self,
        p: &mut ImageBatchPreparation,
        r: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.inner.try_admit_request(p, r)
    }
    fn try_prepare_batch(
        &mut self,
        r: &mut [&mut ImageDenoisingRequest],
        p: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        self.inner.try_prepare_batch(r, p)
    }
    fn pause_batch(
        &mut self,
        b: &mut ImageDenoisingBatch,
        r: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        self.inner.pause_batch(b, r)
    }
    fn advance_batch(
        &mut self,
        b: &mut ImageDenoisingBatch,
        r: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        self.inner.advance_batch(b, r)
    }
    fn finish_request(
        &mut self,
        r: &mut ImageDenoisingRequest,
        b: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        self.inner.finish_request(r, b)
    }
}
fn edit(host: &HostMemoryAccount, seed: u64) -> ImageEditRequest {
    let mut png = Vec::new();
    let rgba = [17, 41, ((seed % 8) * 23) as u8, 255].repeat(4);
    write_rgba8_png(
        &mut png,
        PngDimensions::new(2, 2).unwrap(),
        PngLimits::new(8, 64).unwrap(),
        |sink| sink.write_all(&rgba),
    )
    .unwrap();
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, png.len())
        .unwrap();
    bytes.as_mut().copy_from_slice(&png);
    ImageEditRequest {
        generation: request(seed),
        image: bytes.freeze(),
    }
}
fn start_typed(
    enabled: bool,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<Mutex<Vec<Seen>>>,
) {
    let (pipeline, source) = fixture_pipeline();
    let runtime = pipeline.execution.runtime.clone();
    let boundary = Arc::new(Boundary::default());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let service = ImageService::new(
        TypedEdit {
            inner: Preconditioned {
                pipeline,
                _source: source,
                boundary: boundary.clone(),
                trace: trace.clone(),
                denied_native_join: Arc::new(AtomicBool::new(false)),
            },
            enabled,
            seen: seen.clone(),
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
    (service, boundary, trace, runtime, seen)
}

#[test]
fn default_qwen_step_edit_is_explicitly_rejected_without_any_submission() {
    let (mut pipeline, _source) = fixture_pipeline();
    let image = edit(pipeline.host_account(), 0);
    let before = pipeline.execution.runtime.snapshot();
    assert!(!ImageDenoisingGenerator::supports_image_request(
        &pipeline,
        ImageRequestRef::Edit(&image)
    ));
    assert!(matches!(
        ImageDenoisingGenerator::prepare_image_request(
            &mut pipeline,
            ImageRequestRef::Edit(&image)
        ),
        Err(ImageBatchError::Rejected(_))
    ));
    assert!(matches!(
        ImageDenoisingGenerator::start_image_request(
            &mut pipeline,
            ImageBatchPreparation::new(0u64),
            ImageRequestRef::Edit(&image)
        ),
        Err(ImageBatchError::Rejected(_))
    ));
    assert_eq!(
        pipeline.execution.runtime.snapshot().submissions,
        before.submissions
    );
    drop(image);
    pipeline.shutdown().unwrap();
    assert_eq!(
        pipeline.host_account().usage().unwrap().committed_bytes(),
        0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_edit_support_does_not_silently_opt_into_the_step_edit_protocol() {
    let (service, _boundary, _trace, runtime, seen) = start_typed(false);
    let before = runtime.snapshot().submissions;
    let image = edit(service.host_account(), 0);
    let original = fingerprint(&image);
    let result = service.try_submit_edit(image).unwrap().wait().await;
    assert_eq!(
        result.err().unwrap().kind(),
        ImageServiceErrorKind::Generation
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].phase, Phase::Whole);
    assert_eq!(
        (seen[0].pointer, seen[0].digest),
        (original.pointer, original.digest)
    );
    drop(seen);
    assert_eq!(runtime.snapshot().submissions, before);
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn typed_edit_keeps_the_original_png_through_prepare_start_cancel_and_refill() {
    let (service, boundary, trace, runtime, seen) = start_typed(true);
    let _release = ReleaseOnDrop(boundary.clone());
    let a = edit(service.host_account(), 0);
    let af = fingerprint(&a);
    let a = service.try_submit_edit(a).unwrap();
    boundary.wait(1).await;
    let b = edit(service.host_account(), 1);
    let bf = fingerprint(&b);
    let b = service.try_submit_edit(b).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    a.cancel();
    let c = edit(service.host_account(), 2);
    let cf = fingerprint(&c);
    let c = service.try_submit_edit(c).unwrap();
    boundary.release(2);
    assert_eq!(
        a.wait().await.err().unwrap().kind(),
        ImageServiceErrorKind::Cancelled
    );
    let b = b.wait().await.unwrap();
    let c = c.wait().await.unwrap();
    assert_eq!([b.receipt.seed, c.receipt.seed], [1, 2]);
    let seen = seen.lock().unwrap();
    for expected in [af, bf, cf] {
        let rows = seen
            .iter()
            .filter(|s| s.seed == expected.seed)
            .collect::<Vec<_>>();
        assert_eq!(
            rows.iter().map(|s| s.phase).collect::<Vec<_>>(),
            [Phase::Prepared, Phase::Started]
        );
        assert!(rows
            .iter()
            .all(|s| (s.pointer, s.digest) == (expected.pointer, expected.digest)));
    }
    drop(seen);
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    drop((b, c));
    finish(&service, &runtime).await;
}
