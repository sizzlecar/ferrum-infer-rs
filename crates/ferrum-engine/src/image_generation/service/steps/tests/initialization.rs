//! Real service scheduling, Stored DiT/Euler/decoder and Host ownership. Only
//! the initialization boundary is controlled/preconditioned like the parent;
//! these tests do not claim H3 causal-model or GPU numerical qualification.
use super::*;
use ferrum_runtime::{HostMemoryScope, ManagedHostBuffer};
use std::task::Waker;

#[derive(Default)]
struct State {
    begun: bool,
    advances: usize,
    waves: usize,
    cancelling: bool,
    allow_ready: bool,
    allow_cancel: bool,
    terminal: bool,
    phase_live: bool,
    phase_bytes: usize,
    drops: usize,
    eos_joined: bool,
    wake: Option<Waker>,
}
#[derive(Default)]
struct Control {
    state: Mutex<State>,
    changed: tokio::sync::Notify,
}
struct ReleaseInitializationOnDrop(Arc<Control>);
impl Drop for ReleaseInitializationOnDrop {
    fn drop(&mut self) {
        let wake = {
            let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
            state.allow_ready = true;
            state.allow_cancel = true;
            state.wake.take()
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}
impl Control {
    async fn wait(&self, predicate: impl Fn(&State) -> bool) {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let changed = self.changed.notified();
                if predicate(&self.state.lock().unwrap()) {
                    break;
                }
                changed.await;
            }
        })
        .await
        .expect("original service initialization boundary becomes observable");
    }
    fn release(&self, cancel: bool) {
        let wake = {
            let mut s = self.state.lock().unwrap();
            if cancel {
                s.allow_cancel = true;
            } else {
                s.allow_ready = true;
            }
            s.wake.take()
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}
struct Phase {
    seed: u64,
    remaining: usize,
    scope: Option<HostMemoryScope>,
    buffer: Option<ManagedHostBuffer>,
    control: Arc<Control>,
}
impl Phase {
    fn retire(&mut self) {
        drop(self.buffer.take());
        self.scope.take().unwrap().close().unwrap();
        let mut s = self.control.state.lock().unwrap();
        s.phase_live = false;
        s.terminal = true;
    }
}
impl Drop for Phase {
    fn drop(&mut self) {
        self.control.state.lock().unwrap().drops += 1;
    }
}
#[derive(Clone, Copy)]
enum Mode {
    Pending,
    NativePressure,
    Fatal,
    Rejected,
}
struct Phased {
    inner: Preconditioned,
    control: Arc<Control>,
    mode: Mode,
}
impl ImageGenerator for Phased {
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
        panic!("shared step consumer is required");
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        self.inner.shutdown()
    }
}
impl ImageDenoisingGenerator for Phased {
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
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        if *p.get_mut::<u64>()? == 1 && matches!(self.mode, Mode::NativePressure) {
            return Err(ImageBatchError::InitializationAwaitDenoisingProgress {
                diagnostic: "controlled zero-native initialization admission pressure".into(),
            });
        }
        self.inner.try_admit_request(p, resources)
    }
    fn start_request(
        &mut self,
        p: ImageBatchPreparation,
        r: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        self.inner.start_request(p, r)
    }
    fn begin_image_initialization(
        &mut self,
        p: ImageBatchPreparation,
        request: ImageRequestRef<'_>,
    ) -> Result<ImageInitializationOutcome, ImageBatchError> {
        let ImageRequestRef::Generate(r) = request else {
            unreachable!()
        };
        if r.seed != 1 {
            return self
                .start_request(p, r)
                .map(ImageInitializationOutcome::Ready);
        }
        assert_eq!(p.into_inner::<u64>()?, r.seed);
        let host = self.host_account();
        let bytes = host.backing_bytes(16).unwrap();
        let scope = host
            .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes)])
            .unwrap();
        let buffer = scope
            .account()
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
            .unwrap();
        {
            let mut s = self.control.state.lock().unwrap();
            s.begun = true;
            s.phase_live = true;
            s.phase_bytes = bytes;
        }
        self.control.changed.notify_one();
        Ok(ImageInitializationOutcome::Pending(
            ImageRequestInitialization::new(Phase {
                seed: r.seed,
                remaining: 2,
                scope: Some(scope),
                buffer: Some(buffer),
                control: self.control.clone(),
            }),
        ))
    }
    fn advance_image_initialization(
        &mut self,
        init: &mut ImageRequestInitialization,
        request: ImageRequestRef<'_>,
        cancelled: bool,
        wake: &Waker,
    ) -> Result<ImageInitializationProgress, ImageBatchError> {
        let p = init.get_mut::<Phase>()?;
        let ImageRequestRef::Generate(r) = request else {
            unreachable!()
        };
        assert_eq!(p.seed, r.seed);
        let mut s = self.control.state.lock().unwrap();
        s.advances += 1;
        if cancelled {
            s.cancelling = true;
            if !s.allow_cancel {
                s.wake = Some(wake.clone());
                drop(s);
                self.control.changed.notify_one();
                return Ok(ImageInitializationProgress::Pending {
                    made_progress: false,
                });
            }
            drop(s);
            p.retire();
            self.control.changed.notify_one();
            return Ok(ImageInitializationProgress::Cancelled);
        }
        if matches!(self.mode, Mode::Fatal) {
            return Err(ImageBatchError::Fatal(Box::new(service_error(
                ImageServiceErrorKind::Generation,
                "original initialization failure",
            ))));
        }
        if matches!(self.mode, Mode::Rejected) {
            drop(s);
            p.retire();
            self.control.changed.notify_one();
            return Ok(ImageInitializationProgress::Rejected {
                diagnostic: "quiescent initialization answer was rejected".into(),
            });
        }
        if p.remaining > 0 {
            p.buffer.as_mut().unwrap().as_mut()[p.remaining] = p.remaining as u8;
            p.remaining -= 1;
            s.waves += 1;
            drop(s);
            self.control.changed.notify_one();
            return Ok(ImageInitializationProgress::Pending {
                made_progress: true,
            });
        }
        if !s.allow_ready {
            s.wake = Some(wake.clone());
            drop(s);
            self.control.changed.notify_one();
            return Ok(ImageInitializationProgress::Pending {
                made_progress: false,
            });
        }
        drop(s);
        p.retire();
        let ready = self
            .inner
            .start_request(ImageBatchPreparation::new(p.seed), r)?;
        self.control.changed.notify_one();
        Ok(ImageInitializationProgress::Ready(ready))
    }
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut ImageDenoisingRequest],
        previous: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        if requests.iter().any(|r| {
            r.get::<ContinuousRequest<Runtime>>()
                .unwrap()
                .fixture_snapshot()
                .0
                == 1
        }) {
            let mut s = self.control.state.lock().unwrap();
            assert!(
                s.terminal,
                "a pending/EOS-less initialization must never enter a physical DiT batch"
            );
            s.eos_joined = true;
        }
        self.inner.try_prepare_batch(requests, previous)
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
    ) -> Result<ImageGenerationOutput, ImageBatchError> {
        self.inner.finish_request(request, batch)
    }
}

fn start(
    mode: Mode,
    allow_ready: bool,
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
    let control = Arc::new(Control::default());
    control.state.lock().unwrap().allow_ready = allow_ready;
    let service = ImageService::new(
        Phased {
            inner: Preconditioned {
                pipeline,
                _source: source,
                boundary: boundary.clone(),
                trace: trace.clone(),
                denied_native_join: Arc::new(AtomicBool::new(false)),
            },
            control: control.clone(),
            mode,
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
async fn pending_initialization_yields_to_real_survivor_and_joins_only_after_terminal_eos() {
    let (service, boundary, trace, runtime, control) = start(Mode::Pending, false);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    assert!(control.state.lock().unwrap().begun);
    assert!(trace
        .lock()
        .unwrap()
        .iter()
        .all(|row| row.iter().all(|(seed, _, _)| *seed == 0)));
    assert_eq!(service.snapshot().preparing_requests, 1);
    boundary.release(2);
    let a = a.wait().await.unwrap();
    control.wait(|s| s.waves == 2 && s.wake.is_some()).await;
    assert!(!control.state.lock().unwrap().eos_joined);
    assert_eq!(service.snapshot().active_requests, 0);
    control.release(false);
    let b = b.wait().await.unwrap();
    assert!(control.state.lock().unwrap().eos_joined);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn idle_progressing_initialization_finishes_without_an_unrelated_wake() {
    let (service, boundary, trace, runtime, control) = start(Mode::Pending, true);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    boundary.state.lock().unwrap().all = true;
    let output = service
        .try_submit(request(1))
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert!(control.state.lock().unwrap().eos_joined);
    assert_eq!(control.state.lock().unwrap().waves, 2);
    assert!(!trace.lock().unwrap().is_empty());
    drop(output);
    finish(&service, &runtime).await;
}

async fn cancel_phase(closing: bool) {
    let (service, boundary, trace, runtime, control) = start(Mode::Pending, false);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    boundary.state.lock().unwrap().all = true;
    let host = service.host_account().clone();
    let gate = Arc::new(HostOutputGate {
        host: host.clone(),
        logical: 16,
    });
    let job = service.try_submit_with_output(request(1), gate).unwrap();
    control.wait(|s| s.begun && s.wake.is_some()).await;
    let mut shutdown = if closing {
        Some(Box::pin(service.shutdown()))
    } else {
        job.cancel();
        None
    };
    if let Some(shutdown) = &mut shutdown {
        assert!(futures::poll!(shutdown.as_mut()).is_pending());
    }
    control.wait(|s| s.cancelling).await;
    assert_eq!(service.snapshot().preparing_requests, 1);
    assert_eq!(service.snapshot().active_requests, 0);
    assert!(control.state.lock().unwrap().phase_live);
    assert_eq!(control.state.lock().unwrap().drops, 0);
    assert!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output)
            >= host.backing_bytes(16).unwrap()
    );
    let mut waiting = Box::pin(job.wait());
    assert!(futures::poll!(waiting.as_mut()).is_pending());
    control.release(true);
    let error = waiting
        .await
        .err()
        .expect("cancelled initialization rejects");
    assert_eq!(
        error.kind(),
        if closing {
            ImageServiceErrorKind::Shutdown
        } else {
            ImageServiceErrorKind::Cancelled
        }
    );
    assert!(trace.lock().unwrap().is_empty());
    assert!(!control.state.lock().unwrap().phase_live);
    assert_eq!(control.state.lock().unwrap().drops, 1);
    if let Some(shutdown) = shutdown {
        let retired = shutdown.await.unwrap();
        assert!(!retired.has_failure());
        drop(retired);
    } else {
        finish(&service, &runtime).await;
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_initializer_keeps_output_and_preparing_credit_until_terminal() {
    cancel_phase(false).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn closing_drives_original_initializer_to_safe_terminal_before_returning() {
    cancel_phase(true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn idle_zero_native_initialization_pressure_has_a_finite_queuefull_result() {
    let (service, boundary, trace, runtime, control) = start(Mode::NativePressure, false);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    let error = service
        .try_submit(request(1))
        .unwrap()
        .wait()
        .await
        .err()
        .expect("idle native pressure rejects");
    assert_eq!(error.kind(), ImageServiceErrorKind::QueueFull);
    assert!(!control.state.lock().unwrap().begun);
    assert!(trace.lock().unwrap().is_empty());
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn zero_native_initialization_pressure_preserves_actual_survivor_progress() {
    let (service, boundary, trace, runtime, control) = start(Mode::NativePressure, false);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    assert!(!control.state.lock().unwrap().begun);
    boundary.release(2);
    let a = a.wait().await.unwrap();
    assert_eq!(
        b.wait()
            .await
            .err()
            .expect("unreleased native pressure rejects when idle")
            .kind(),
        ImageServiceErrorKind::QueueFull
    );
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    drop(a);
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn safe_initialization_rejection_preserves_real_dit_decode_and_future_requests() {
    let (service, boundary, trace, runtime, control) = start(Mode::Rejected, false);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    assert!(control.state.lock().unwrap().phase_live);
    boundary.release(2);
    let error = b.wait().await.err().expect("quiescent rejection rejects");
    assert_eq!(error.kind(), ImageServiceErrorKind::Generation);
    assert!(error
        .to_string()
        .contains("quiescent initialization answer"));
    let a = a.wait().await.unwrap();
    assert!(control.state.lock().unwrap().terminal);
    assert_eq!(control.state.lock().unwrap().drops, 1);
    let c = service
        .try_submit(request(2))
        .unwrap()
        .wait()
        .await
        .unwrap();
    let trace = trace.lock().unwrap().clone();
    assert!(trace.iter().flatten().all(|(seed, _, _)| *seed != 1));
    assert!(trace.iter().flatten().any(|(seed, _, _)| *seed == 2));
    check(
        &trace,
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    drop((a, c));
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fatal_initialization_keeps_the_exact_opaque_host_owner_in_retirement_custody() {
    let (service, boundary, trace, _runtime, control) = start(Mode::Fatal, false);
    let _release = ReleaseOnDrop(boundary.clone());
    let _phase_release = ReleaseInitializationOnDrop(control.clone());
    let host = service.host_account().clone();
    assert_eq!(
        service
            .try_submit(request(1))
            .unwrap()
            .wait()
            .await
            .err()
            .expect("fatal initialization rejects")
            .kind(),
        ImageServiceErrorKind::Generation
    );
    let retired = service.shutdown().await.unwrap();
    assert!(retired.has_failure());
    assert_eq!(control.state.lock().unwrap().drops, 0);
    assert!(control.state.lock().unwrap().phase_live);
    assert!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::InputPreprocess)
            >= control.state.lock().unwrap().phase_bytes
    );
    assert!(trace.lock().unwrap().is_empty());
    // An unattended failure receipt deliberately quarantines its native
    // owners. Explicitly transfer it, recover this controlled Host-only
    // phase, and prove that the exact original owner stayed recoverable.
    let (generator, failure) = retired.into_parts();
    let Some(ImageWorkerFailure::Generation(error)) = failure else {
        panic!("original generation failure is retained");
    };
    let mut owners = error.downcast::<super::super::FailedOwners>().unwrap();
    assert_eq!(owners._initializations.len(), 1);
    let mut phase = owners
        ._initializations
        .pop()
        .unwrap()
        .into_inner::<Phase>()
        .unwrap();
    phase.retire();
    drop((phase, owners, generator));
    assert_eq!(control.state.lock().unwrap().drops, 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
