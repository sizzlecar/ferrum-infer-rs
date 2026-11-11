//! Actual Stored DiT/VAE service consumption with a larger, owned F32 RGB
//! postprocessing result. Initialization stays explicitly preconditioned, as
//! in the parent fixture; this is completion/Host qualification, not H3 GPU.
use super::*;
use crate::image_generation::{host_budget, QwenImage21T2iLayout, QwenImage21TransformerConfig};
use ferrum_models::image_pixels::{ImagePixelEncoding, OwnedImagePixels};
use ferrum_runtime::{ElementType, HostMemoryScope, ResolvedTensorLayout, ResolvedTensorSpec};
use std::sync::atomic::AtomicUsize;

#[derive(Default)]
struct CompletionWait {
    completed: AtomicUsize,
    blocked_after_output: AtomicBool,
    changed: tokio::sync::Notify,
}
impl CompletionWait {
    async fn after_first_output(&self) {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let changed = self.changed.notified();
                if self.blocked_after_output.load(Ordering::Acquire) {
                    break;
                }
                changed.await;
            }
        })
        .await
        .expect("second decoder reaches real Host capacity while first pixels are owned");
    }
}
struct F32Completion {
    inner: Preconditioned,
    maximum: usize,
    phase_bytes: usize,
    phase: Option<HostMemoryScope>,
    wait: Arc<CompletionWait>,
}
impl ImageGenerator for F32Completion {
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        Some(self)
    }
    fn host_account(&self) -> &HostMemoryAccount {
        self.inner.host_account()
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        let mut c = self.inner.capabilities()?;
        c.maximum_batch_size = self.maximum;
        Ok(c)
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        unreachable!("declared step consumer")
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        drop(self.phase.take());
        ImageGenerator::shutdown(&mut self.inner)
    }
}
impl ImageDenoisingGenerator for F32Completion {
    type Output = ImageGenerationOutput;
    fn prepare_request(
        &mut self,
        r: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.inner.prepare_request(r)
    }
    fn try_admit_request(
        &mut self,
        p: &mut ImageBatchPreparation,
        r: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.inner.try_admit_request(p, r)
    }
    fn start_request(
        &mut self,
        p: ImageBatchPreparation,
        r: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        self.inner.start_request(p, r)
    }
    fn try_prepare_batch(
        &mut self,
        r: &mut [&mut ImageDenoisingRequest],
        p: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        if self.phase.is_none() {
            self.phase = match self
                .host_account()
                .reserve_scope(&[(HostMemoryClass::Output, self.phase_bytes)])
            {
                Ok(scope) => Some(scope),
                Err(e) if e.is_temporary_capacity() => return Ok(None),
                Err(e) => return Err(ImageBatchError::Rejected(e.to_string())),
            };
        }
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
        self.try_finish_request(r, b)?.ok_or_else(|| {
            ImageBatchError::Overloaded(
                "real completion capacity is held by delivered pixels".into(),
            )
        })
    }
    fn try_finish_request(
        &mut self,
        r: &mut ImageDenoisingRequest,
        b: &mut ImageDenoisingBatch,
    ) -> Result<Option<Self::Output>, ImageBatchError> {
        let parent = self.host_account().clone();
        let old = self
            .phase
            .take()
            .expect("one shared decoder phase was admitted before publication");
        self.phase = Some(
            match parent.exchange_scope(old, &[(HostMemoryClass::Output, self.phase_bytes)]) {
                Ok(scope) => scope,
                Err(failure) => {
                    let (e, old) = failure.into_parts();
                    self.phase = Some(old);
                    if e.is_temporary_capacity() {
                        if self.wait.completed.load(Ordering::Acquire) > 0 {
                            self.wait
                                .blocked_after_output
                                .store(true, Ordering::Release);
                            self.wait.changed.notify_one();
                        }
                        return Ok(None);
                    }
                    return Err(ImageBatchError::Rejected(e.to_string()));
                }
            },
        );
        let host = self.phase.as_ref().unwrap().account();
        let spec = ResolvedTensorSpec::new(
            vec![1, 3, 32, 32],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let logical = spec.minimum_storage_bytes().unwrap() as usize;
        let mut pixels = host
            .allocate_zeroed(HostMemoryClass::Output, logical)
            .unwrap();
        // Real distinct transient backing coexists with the final result and
        // the model readback. Find its logical extent using actual padding.
        let available = self.phase_bytes - parent.backing_bytes(logical).unwrap();
        let (mut low, mut high) = (1usize, available);
        while low < high {
            let mid = low + (high - low + 1) / 2;
            if parent.backing_bytes(mid).unwrap() <= available {
                low = mid
            } else {
                high = mid - 1
            }
        }
        let scratch = host.allocate_zeroed(HostMemoryClass::Output, low).unwrap();
        let mut output = self.inner.finish_request(r, b)?;
        assert_eq!(output.rgba.tensor().dimensions(), [1, 4, 32, 32]);
        // Preserve the declared staged BF16 unit-domain conversion, then
        // store actual RGB planes as F32. The alpha plane is deliberately gone.
        for (source, target) in output.rgba.bytes().bytes()[..3 * 32 * 32 * 2]
            .chunks_exact(2)
            .zip(pixels.as_mut().chunks_exact_mut(4))
        {
            let value =
                half::bf16::from_bits(u16::from_le_bytes(source.try_into().unwrap())).to_f32();
            let clamped = half::bf16::from_f32(value.clamp(-1.0, 1.0));
            let half = half::bf16::from_f32(clamped.to_f32() * 0.5);
            let unit = half::bf16::from_f32(half.to_f32() + 0.5).to_f32();
            target.copy_from_slice(&unit.to_le_bytes());
        }
        drop(scratch);
        output.rgba = OwnedImagePixels::new(
            &parent,
            spec,
            pixels.freeze(),
            ImagePixelEncoding::F32UnitRgbRne,
        )
        .unwrap();
        self.wait.completed.fetch_add(1, Ordering::AcqRel);
        Ok(Some(output))
    }
}

fn start_completion(
    maximum: usize,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<CompletionWait>,
) {
    let (mut pipeline, source) = fixture_pipeline();
    pipeline.execution.config.maximum_batch_size = maximum;
    let host = pipeline.host_account();
    let decoder = pipeline
        .execution
        .sources
        .assets
        .decoder_batch(1, 2, 2)
        .unwrap();
    let original = host
        .backing_bytes(
            decoder
                .output_tensor()
                .unwrap()
                .minimum_storage_bytes()
                .unwrap() as usize,
        )
        .unwrap();
    let f32_pixels = host.backing_bytes(3 * 32 * 32 * 4).unwrap();
    let config = QwenImage21TransformerConfig::new(
        pipeline.execution.sources.assets.transformer.clone(),
        QwenImage21T2iLayout::new(2, 2, 2, vec![2; maximum]).unwrap(),
    )
    .unwrap();
    let prepared = pipeline.prepare_denoiser(&config, maximum).unwrap();
    let step_output = host_budget::sum_ports(
        host,
        host_budget::ports(&prepared).unwrap().outputs().iter(),
    )
    .unwrap();
    let phase_bytes = host.limits().class_bytes(HostMemoryClass::Output)
        - maximum * original
        - step_output
        - (f32_pixels - original) / 2;
    assert!(phase_bytes > f32_pixels);
    let runtime = pipeline.execution.runtime.clone();
    let boundary = Arc::new(Boundary::default());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let wait = Arc::new(CompletionWait::default());
    let service = ImageService::new(
        F32Completion {
            inner: Preconditioned {
                pipeline,
                _source: source,
                boundary: boundary.clone(),
                trace: trace.clone(),
                denied_native_join: Arc::new(AtomicBool::new(false)),
            },
            maximum,
            phase_bytes,
            phase: None,
            wait: wait.clone(),
        },
        ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            maximum_pending: 4,
            maximum_pending_bytes: 1 << 18,
            maximum_prompt_bytes: 1024,
            max_num_seqs: maximum,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    (service, boundary, trace, runtime, wait)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_terminal_decoder_has_one_shared_phase_and_delivers_before_next_canvas() {
    let (service, boundary, trace, runtime, wait) = start_completion(2);
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    boundary.release(2);
    // No outside reservation can release credit to produce this first image.
    let a = tokio::time::timeout(Duration::from_secs(15), a.wait())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(a.rgba.encoding(), ImagePixelEncoding::F32UnitRgbRne);
    wait.after_first_output().await;
    assert_eq!(service.snapshot().active_requests, 1);
    // Only these genuinely delivered pixels block the second completed lane.
    drop(a);
    let b = tokio::time::timeout(Duration::from_secs(15), b.wait())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(b.rgba.tensor().dimensions(), [1, 3, 32, 32]);
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    drop(b);
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_pending_front_does_not_retain_a_cancelled_completed_successors_credit() {
    let (service, boundary, trace, runtime, wait) = start_completion(3);
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    let c = service.try_submit(request(2)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    boundary.release(2);
    let a = tokio::time::timeout(Duration::from_secs(15), a.wait())
        .await
        .unwrap()
        .unwrap();
    wait.after_first_output().await;
    // Both later cursors actually completed. C sits behind pending B, and its
    // real persistent Output grant is sufficient to unblock B when retired.
    assert_eq!(
        trace
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .iter()
            .map(|(seed, step, _)| (*seed, *step))
            .collect::<Vec<_>>(),
        [(1, 2), (2, 2)]
    );
    c.cancel();
    assert_eq!(
        c.wait().await.err().unwrap().kind(),
        ImageServiceErrorKind::Cancelled
    );
    let b = tokio::time::timeout(Duration::from_secs(15), b.wait())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(b.receipt.seed, 1);
    assert_eq!(
        a.receipt.seed, 0,
        "first real canvas is still owned while B finishes"
    );
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    drop((a, b));
    finish(&service, &runtime).await;
}
