//! Real worker, Stored Reference DiT/decoder and original Host owners. Only
//! the auxiliary phase is controlled; this is not a causal/GPU throughput test.
use super::*;
use ferrum_runtime::{HostMemoryScope, ManagedHostBuffer};
use std::task::Waker;
mod ready_capacity;

#[derive(Clone, Copy)]
enum Mode {
    Ready,
    Handoff,
    AuxiliaryTail,
    AuxiliaryUnknownProgress,
    Unknown,
    Duplicate,
    Omitted,
    EarlyHandoff,
    Fatal,
    FormatterPanic,
    Pressure,
    PreparationRejected,
    BeginRejected,
    BlockedDownstream,
    ReadyCapacity,
}
struct FormatterPanic;
impl fmt::Display for FormatterPanic {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("controlled original error formatter panic")
    }
}
impl fmt::Debug for FormatterPanic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl Error for FormatterPanic {}
#[derive(Default)]
struct State {
    metadata_entered: bool,
    maximum_probed: bool,
    prefix_cancel_wait: Option<(Arc<ResourceSignal>, u64)>,
    prefix_cancel_entered: bool,
    deferred_ready_join: bool,
    downstream_waited: bool,
    pressure: Option<ManagedHostBuffer>,
    capacity_subscription: Option<ImageOutputLease>,
    release_metadata: bool,
    allow_finish: bool,
    groups: Vec<Vec<(Uuid, u64)>>,
    snapshots: Vec<Vec<(Uuid, bool)>>,
    events: Vec<&'static str>,
    live: bool,
    drops: usize,
    phase_bytes: usize,
    wake: Option<Waker>,
    initialized: Vec<(u64, usize)>,
    cancelled_initializations: Vec<(u64, usize)>,
    native_pressure: Option<native_pressure::NativePressure>,
}
#[derive(Default)]
struct Control {
    state: Mutex<State>,
    gate: Condvar,
    changed: tokio::sync::Notify,
}
impl Control {
    async fn wait(&self, predicate: impl Fn(&State) -> bool) {
        let result = tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let changed = self.changed.notified();
                if predicate(&self.state.lock().unwrap()) {
                    break;
                }
                changed.await;
            }
        })
        .await;
        let diagnostic = {
            let state = self.state.lock().unwrap();
            format!(
                "metadata={}, probed={}, groups={:?}, events={:?}, live={}, snapshots={}, wake={}",
                state.metadata_entered,
                state.maximum_probed,
                state.groups,
                state.events,
                state.live,
                state.snapshots.len(),
                state.wake.is_some()
            )
        };
        assert!(
            result.is_ok(),
            "original cohort boundary becomes observable: {diagnostic}"
        );
    }
    fn metadata(&self) {
        self.state.lock().unwrap().release_metadata = true;
        self.gate.notify_all();
    }
    fn finish(&self) {
        let wake = {
            let mut s = self.state.lock().unwrap();
            s.allow_finish = true;
            s.wake.take()
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }
}
struct Release(Arc<Control>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.metadata();
        self.0.finish();
    }
}
struct Prepared {
    members: Vec<(Uuid, u64)>,
    phase: Option<HostMemoryScope>,
}
struct Phase {
    members: Vec<(Uuid, u64)>,
    scope: Option<HostMemoryScope>,
    bytes: Option<ManagedHostBuffer>,
    control: Arc<Control>,
}
impl Phase {
    fn retire(&mut self) {
        drop(self.bytes.take());
        if let Some(scope) = self.scope.take() {
            scope.close().unwrap();
        }
        let mut s = self.control.state.lock().unwrap();
        s.live = false;
        s.events.push("retired");
    }
}
impl Drop for Phase {
    fn drop(&mut self) {
        self.control.state.lock().unwrap().drops += 1;
    }
}
struct Cohorted {
    inner: Preconditioned,
    control: Arc<Control>,
    mode: Mode,
}
impl ImageGenerator for Cohorted {
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        Some(self)
    }
    fn host_account(&self) -> &HostMemoryAccount {
        self.inner.host_account()
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        let mut capabilities = self.inner.capabilities()?;
        capabilities.maximum_batch_size = self.inner.pipeline.execution.config.maximum_batch_size;
        Ok(capabilities)
    }
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool {
        let signal = {
            let mut state = self.control.state.lock().unwrap();
            let signal = state.prefix_cancel_wait.take();
            if signal.is_some() {
                assert_eq!((a.seed, b.seed), (1, 1));
                assert!(state.maximum_probed);
                state.prefix_cancel_entered = true;
                self.control.changed.notify_one();
            }
            signal
        };
        if let Some((signal, observed)) = signal {
            // The real prefix waiter has already read cancelled=false and
            // still owns Shared.state. Cancellation's resource notification
            // releases this boundary, before the real Condvar wait registers.
            signal.wait(observed);
        }
        true
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        panic!("real step consumer is required")
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        ImageGenerator::shutdown(&mut self.inner)
    }
}
impl ImageDenoisingGenerator for Cohorted {
    type Output = ImageGenerationOutput;
    fn initialization_uses_auxiliary_model(&self) -> bool {
        matches!(
            self.mode,
            Mode::AuxiliaryTail | Mode::AuxiliaryUnknownProgress
        )
    }
    fn remaining_denoising_steps(&self, request: &ImageDenoisingRequest) -> Option<usize> {
        if !matches!(self.mode, Mode::AuxiliaryTail | Mode::Handoff) {
            return None;
        }
        let (_, completed, _) = request
            .get::<ContinuousRequest<Runtime>>()
            .ok()?
            .fixture_snapshot();
        self.inner
            .pipeline
            .execution
            .sources
            .assets
            .schedule
            .steps()
            .len()
            .checked_sub(completed)
    }
    fn initialization_cohort_generator(
        &mut self,
    ) -> Option<&mut dyn ImageInitializationCohortGenerator> {
        Some(self)
    }
    fn prepare_request(
        &mut self,
        r: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        if r.seed == 1 {
            let mut s = self.control.state.lock().unwrap();
            s.metadata_entered = true;
            self.control.changed.notify_one();
            while !s.release_metadata {
                s = self.control.gate.wait(s).unwrap();
            }
        }
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
    fn advance_image_initialization(
        &mut self,
        owner: &mut ImageRequestInitialization,
        request: ImageRequestRef<'_>,
        cancelled: bool,
        wake: &Waker,
    ) -> Result<ImageInitializationProgress, ImageBatchError> {
        let seed = *owner.get::<u64>()?;
        if cancelled {
            self.control
                .state
                .lock()
                .unwrap()
                .cancelled_initializations
                .push((seed, self.inner.trace.lock().unwrap().len()));
            return Ok(ImageInitializationProgress::Cancelled);
        }
        let ImageRequestRef::Generate(request) = request else {
            unreachable!()
        };
        assert_eq!(seed, request.seed);
        if matches!(self.mode, Mode::BlockedDownstream) && seed == 2 {
            match self
                .host_account()
                .allocate_zeroed(HostMemoryClass::Output, 128 * 1024)
            {
                Ok(bytes) => drop(bytes),
                Err(error) if error.is_temporary_capacity() => {
                    let subscription =
                        ImageOutputLease::new(self.host_account().subscribe_capacity(wake.clone()));
                    let mut state = self.control.state.lock().unwrap();
                    state.downstream_waited = true;
                    state.capacity_subscription = Some(subscription);
                    drop(state);
                    self.control.changed.notify_one();
                    return Ok(ImageInitializationProgress::Pending {
                        made_progress: false,
                    });
                }
                Err(error) => return Err(ImageBatchError::Rejected(error.to_string())),
            }
        }
        let ready = self
            .inner
            .start_request(ImageBatchPreparation::new(seed), request)?;
        self.control
            .state
            .lock()
            .unwrap()
            .initialized
            .push((seed, self.inner.trace.lock().unwrap().len()));
        if matches!(self.mode, Mode::BlockedDownstream) && seed == 1 {
            let pressure = self
                .host_account()
                .allocate_zeroed(HostMemoryClass::Output, 192 * 1024)
                .unwrap();
            self.control.state.lock().unwrap().pressure = Some(pressure);
        }
        Ok(ImageInitializationProgress::Ready(ready))
    }
    fn try_prepare_batch(
        &mut self,
        r: &mut [&mut ImageDenoisingRequest],
        b: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        if matches!(self.mode, Mode::BlockedDownstream)
            && r.iter().any(|request| {
                request
                    .get::<crate::image_generation::denoising::ContinuousRequest<Runtime>>()
                    .unwrap()
                    .fixture_snapshot()
                    .0
                    == 1
            })
        {
            let mut state = self.control.state.lock().unwrap();
            if !state.deferred_ready_join {
                // A clean optional membership refusal leaves this actual Ready
                // owner pending while its original survivor can still execute.
                state.deferred_ready_join = true;
                return Ok(None);
            }
        }
        self.inner.try_prepare_batch(r, b)
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
        let result = self.inner.advance_batch(b, r)?;
        let pressure = {
            let mut state = self.control.state.lock().unwrap();
            state.events.push("commit");
            if matches!(self.mode, Mode::BlockedDownstream)
                && r.iter().any(|request| {
                    request
                        .get::<crate::image_generation::denoising::ContinuousRequest<Runtime>>()
                        .unwrap()
                        .fixture_snapshot()
                        .0
                        == 1
                })
            {
                state.pressure.take()
            } else {
                None
            }
        };
        // Only a real Stored DiT commit releases the original Output owner.
        drop(pressure);
        Ok(result)
    }
    fn finish_request(
        &mut self,
        r: &mut ImageDenoisingRequest,
        b: &mut ImageDenoisingBatch,
    ) -> Result<ImageGenerationOutput, ImageBatchError> {
        // This fixture's independently owned native pressure retires only
        // after an actual solver completes, before its larger decoder starts.
        let pressure = self.control.state.lock().unwrap().native_pressure.take();
        if let Some(pressure) = pressure {
            pressure.close();
        }
        self.inner.finish_request(r, b)
    }
}
impl ImageInitializationCohortGenerator for Cohorted {
    fn maximum_initialization_cohort_members(
        &mut self,
        p: &mut ImageBatchPreparation,
    ) -> Result<usize, ImageBatchError> {
        self.control.state.lock().unwrap().maximum_probed = true;
        self.control.changed.notify_one();
        Ok(if *p.get_mut::<u64>()? == 0 {
            0
        } else if matches!(self.mode, Mode::ReadyCapacity) {
            self.inner.pipeline.execution.config.maximum_batch_size
        } else {
            2
        })
    }
    fn initialization_cohort_compatible(
        &mut self,
        a: &mut ImageBatchPreparation,
        b: &mut ImageBatchPreparation,
    ) -> Result<bool, ImageBatchError> {
        Ok(*a.get_mut::<u64>()? > 0 && *b.get_mut::<u64>()? > 0)
    }
    fn prepare_initialization_cohort(
        &mut self,
        members: Vec<ImageInitializationCohortMember>,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        let members = members
            .into_iter()
            .map(|m| Ok((m.request_id, m.preparation.into_inner::<u64>()?)))
            .collect::<Result<Vec<_>, ImageBatchError>>()?;
        if matches!(self.mode, Mode::PreparationRejected)
            && members.iter().any(|(_, seed)| *seed == 1)
        {
            return Err(ImageBatchError::InvalidInput(
                "controlled invalid group metadata".into(),
            ));
        }
        Ok(ImageBatchPreparation::new(Prepared {
            members,
            phase: None,
        }))
    }
    fn try_admit_initialization_cohort_at_boundary(
        &mut self,
        p: &mut ImageBatchPreparation,
        _: Option<&mut ImageDenoisingBatch>,
        _: &mut [&mut ImageDenoisingRequest],
    ) -> Result<bool, ImageBatchError> {
        let p = p.get_mut::<Prepared>()?;
        if matches!(self.mode, Mode::Pressure) && p.members.iter().any(|(_, seed)| *seed == 1) {
            return Err(ImageBatchError::InitializationAwaitDenoisingProgress {
                diagnostic: "controlled zero-native pressure".into(),
            });
        }
        if p.phase.is_none() {
            let host = self.host_account();
            let bytes = host.backing_bytes(16).unwrap();
            p.phase = Some(
                host.reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes)])
                    .unwrap(),
            );
            self.control.state.lock().unwrap().phase_bytes = bytes;
        }
        Ok(true)
    }
    fn begin_initialization_cohort(
        &mut self,
        p: &mut ImageBatchPreparation,
        members: &[ImageInitializationCohortRequest<'_>],
    ) -> Result<ImageRequestInitialization, ImageBatchError> {
        let p = p.get_mut::<Prepared>()?;
        if matches!(self.mode, Mode::BeginRejected) && p.members.iter().any(|(_, seed)| *seed == 1)
        {
            return Err(ImageBatchError::InvalidInput(
                "controlled definitely unstarted cohort".into(),
            ));
        }
        assert_eq!(
            members.iter().map(|m| m.request_id).collect::<Vec<_>>(),
            p.members.iter().map(|(id, _)| *id).collect::<Vec<_>>()
        );
        let scope = p.phase.take().unwrap();
        let bytes = scope
            .account()
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
            .unwrap();
        {
            let mut s = self.control.state.lock().unwrap();
            s.groups.push(p.members.clone());
            s.events.push("begin");
            s.live = true;
        }
        self.control.changed.notify_one();
        Ok(ImageRequestInitialization::new(Phase {
            members: p.members.clone(),
            scope: Some(scope),
            bytes: Some(bytes),
            control: self.control.clone(),
        }))
    }
    fn advance_initialization_cohort(
        &mut self,
        owner: &mut ImageRequestInitialization,
        members: &[ImageInitializationCohortRequest<'_>],
        wake: &Waker,
    ) -> Result<ImageInitializationCohortProgress, ImageBatchError> {
        let phase = owner.get_mut::<Phase>()?;
        let mut s = self.control.state.lock().unwrap();
        s.snapshots.push(
            members
                .iter()
                .map(|m| (m.request_id, m.cancelled))
                .collect(),
        );
        if !s.allow_finish {
            s.wake = Some(wake.clone());
            drop(s);
            self.control.changed.notify_one();
            return Ok(ImageInitializationCohortProgress {
                made_progress: false,
                finished: false,
                completed: Vec::new(),
            });
        }
        drop(s);
        if matches!(self.mode, Mode::FormatterPanic) {
            return Err(ImageBatchError::Fatal(Box::new(FormatterPanic)));
        }
        if matches!(self.mode, Mode::Fatal) {
            return Err(ImageBatchError::Fatal(Box::new(service_error(
                ImageServiceErrorKind::Generation,
                "original shared initialization failure",
            ))));
        }
        assert_eq!(phase.members.len(), members.len());
        if matches!(
            self.mode,
            Mode::Unknown | Mode::Duplicate | Mode::Omitted | Mode::EarlyHandoff
        ) {
            let first = members[0].request_id;
            let completed = match self.mode {
                Mode::Unknown => vec![ImageInitializationCohortCompletion {
                    request_id: Uuid::new_v4(),
                    outcome: ImageInitializationCohortOutcome::Rejected {
                        diagnostic: "unknown".into(),
                    },
                }],
                Mode::Duplicate => (0..2)
                    .map(|_| ImageInitializationCohortCompletion {
                        request_id: first,
                        outcome: ImageInitializationCohortOutcome::Rejected {
                            diagnostic: "duplicate".into(),
                        },
                    })
                    .collect(),
                Mode::Omitted => Vec::new(),
                Mode::EarlyHandoff => vec![ImageInitializationCohortCompletion {
                    request_id: first,
                    outcome: ImageInitializationCohortOutcome::Pending(
                        ImageRequestInitialization::new(1u64),
                    ),
                }],
                _ => unreachable!(),
            };
            return Ok(ImageInitializationCohortProgress {
                made_progress: true,
                finished: !matches!(self.mode, Mode::EarlyHandoff),
                completed,
            });
        }
        phase.retire();
        let mut completed = Vec::new();
        // Reverse completion order deliberately; routing must use original UUIDs.
        for member in members.iter().rev() {
            let ImageRequestRef::Generate(request) = member.request else {
                unreachable!()
            };
            let outcome = if member.cancelled {
                ImageInitializationCohortOutcome::Cancelled
            } else if matches!(
                self.mode,
                Mode::Handoff
                    | Mode::AuxiliaryTail
                    | Mode::AuxiliaryUnknownProgress
                    | Mode::BlockedDownstream
                    | Mode::ReadyCapacity
            ) {
                ImageInitializationCohortOutcome::Pending(ImageRequestInitialization::new(
                    request.seed,
                ))
            } else {
                ImageInitializationCohortOutcome::Ready(
                    self.inner
                        .start_request(ImageBatchPreparation::new(request.seed), request)?,
                )
            };
            completed.push(ImageInitializationCohortCompletion {
                request_id: member.request_id,
                outcome,
            });
        }
        Ok(ImageInitializationCohortProgress {
            made_progress: true,
            finished: true,
            completed,
        })
    }
    fn initialization_cohort_blocks_denoising(&self, owner: &ImageRequestInitialization) -> bool {
        owner
            .get::<Phase>()
            .map_or(true, |phase| phase.scope.is_some())
    }
}
fn start_cohort(
    mode: Mode,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<Control>,
) {
    start_cohort_with_wait(mode, 0)
}
fn extended_schedule(pipeline: &mut QwenImage21Pipeline<Runtime>) {
    use ferrum_models::qwen_image21::sampling_schedule::QwenImage21SamplingSchedule;
    let index = serde_json::json!({"scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],
        "sample_sigmas":[1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.125]});
    let scheduler = serde_json::json!({"_class_name":"FlowMatchEulerDiscreteScheduler","_diffusers_version":"fixture",
        "base_image_seq_len":256,"base_shift":0.5,"invert_sigmas":false,"max_image_seq_len":8192,
        "max_shift":0.9,"num_train_timesteps":1000,"shift":1.0,"shift_terminal":null,
        "stochastic_sampling":false,"time_shift_type":"exponential","use_beta_sigmas":false,
        "use_dynamic_shifting":false,"use_exponential_sigmas":false,"use_karras_sigmas":false});
    Arc::get_mut(&mut pipeline.execution.sources.assets)
        .expect("fresh fixture owns its assets")
        .schedule = QwenImage21SamplingSchedule::parse(
        &serde_json::to_vec(&index).unwrap(),
        &serde_json::to_vec(&scheduler).unwrap(),
    )
    .unwrap();
}
fn start_cohort_with_wait(
    mode: Mode,
    wait_ms: u64,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<Control>,
) {
    start_cohort_with_policy(mode, wait_ms, ImageBatchPolicy::Auto)
}
fn start_cohort_with_policy(
    mode: Mode,
    wait_ms: u64,
    batching: ImageBatchPolicy,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<Control>,
) {
    start_cohort_with_capacity(mode, wait_ms, batching, None)
}
fn start_cohort_with_capacity(
    mode: Mode,
    wait_ms: u64,
    batching: ImageBatchPolicy,
    capacity: Option<usize>,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<Control>,
) {
    let (mut pipeline, source) = fixture_pipeline();
    let native_pressure = if matches!(mode, Mode::ReadyCapacity) {
        extended_schedule(&mut pipeline);
        let (bounded, pressure) = ready_capacity::bounded_pipeline(pipeline);
        pipeline = bounded;
        Some(pressure)
    } else {
        None
    };
    if matches!(mode, Mode::BlockedDownstream) {
        extended_schedule(&mut pipeline);
        pipeline.execution.config.maximum_batch_size = 3;
    }
    if let Some(capacity) = capacity {
        pipeline.execution.config.maximum_batch_size = capacity;
    }
    let runtime = pipeline.execution.runtime.clone();
    let boundary = Arc::new(Boundary::default());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let control = Arc::new(Control::default());
    control.state.lock().unwrap().native_pressure = native_pressure;
    let service = ImageService::new(
        Cohorted {
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
            batching,
            maximum_pending: 4,
            maximum_pending_bytes: 1 << 18,
            maximum_prompt_bytes: 1024,
            max_num_seqs: capacity.unwrap_or(
                if matches!(mode, Mode::BlockedDownstream | Mode::ReadyCapacity) {
                    3
                } else {
                    2
                },
            ),
            request_batch_max_wait_ms: wait_ms,
        },
    )
    .unwrap();
    (service, boundary, trace, runtime, control)
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auxiliary_final_step_admission_preserves_ready_joins_and_regroups_the_queued_prefix() {
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    for (mode, policy, defer_tail) in [
        (Mode::AuxiliaryTail, ImageBatchPolicy::Auto, true),
        (Mode::AuxiliaryTail, ImageBatchPolicy::Configured, false),
        (
            Mode::AuxiliaryUnknownProgress,
            ImageBatchPolicy::Auto,
            false,
        ),
        (Mode::Handoff, ImageBatchPolicy::Auto, false),
    ] {
        let (service, boundary, trace, runtime, control) =
            start_cohort_with_policy(mode, 0, policy);
        let _release = Release(control.clone());
        let _steps = ReleaseOnDrop(boundary.clone());
        boundary.state.lock().unwrap().all = true;
        let first = service.try_submit(request(1)).unwrap();
        control.wait(|state| state.metadata_entered).await;
        let rest = (2..=4)
            .map(|seed| service.try_submit(request(seed)).unwrap())
            .collect::<Vec<_>>();
        control.metadata();
        control.wait(|state| state.wake.is_some()).await;
        control.finish();
        let mut outputs = vec![first.wait().await.unwrap()];
        for response in rest {
            outputs.push(response.wait().await.unwrap());
        }
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.receipt.seed)
                .collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        let groups = control
            .state
            .lock()
            .unwrap()
            .groups
            .iter()
            .map(|group| group.iter().map(|(_, seed)| *seed).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(groups[0], [1, 2]);
        if defer_tail {
            assert_eq!(
                groups[1..],
                [vec![3, 4]],
                "finishing the resident tail frees both original slots"
            );
        } else {
            assert_eq!(
                groups[1],
                [3],
                "ordinary admission starts the next request before both slots are free"
            );
            assert!(groups
                .iter()
                .all(|group| !(group.contains(&3) && group.contains(&4))));
        }
        let actual = trace.lock().unwrap();
        check(&actual, &independent);
        assert!(
            actual
                .iter()
                .any(|wave| wave.len() == 2 && wave[0].1 != wave[1].1),
            "an already Ready request still joins its final-step peer"
        );
        drop(actual);
        drop(outputs);
        finish(&service, &runtime).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auxiliary_final_step_admission_retires_a_cancelled_waiter_without_losing_the_next_request()
{
    let (service, boundary, trace, runtime, control) = start_cohort(Mode::AuxiliaryTail);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    let first = service.try_submit(request(1)).unwrap();
    control.wait(|state| state.metadata_entered).await;
    let second = service.try_submit(request(2)).unwrap();
    let cancelled = service.try_submit(request(3)).unwrap();
    let next = service.try_submit(request(4)).unwrap();
    control.metadata();
    control.wait(|state| state.wake.is_some()).await;
    control.finish();
    boundary.wait(1).await;
    boundary.release(1);
    boundary.wait(2).await;
    cancelled.cancel();
    boundary.release(2);
    assert_eq!(
        cancelled
            .wait()
            .await
            .err()
            .expect("queued cancellation is delivered")
            .kind(),
        ImageServiceErrorKind::Cancelled
    );
    let outputs = [
        first.wait().await.unwrap(),
        second.wait().await.unwrap(),
        next.wait().await.unwrap(),
    ];
    assert_eq!(outputs.map(|output| output.receipt.seed), [1, 2, 4]);
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    assert!(control
        .state
        .lock()
        .unwrap()
        .groups
        .iter()
        .flatten()
        .all(|(_, seed)| *seed != 3));
    finish(&service, &runtime).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_seals_fifo_and_routes_reordered_results_to_real_outputs() {
    let (service, boundary, trace, runtime, control) = start_cohort(Mode::Ready);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    boundary.state.lock().unwrap().all = true;
    let a = service.try_submit(request(1)).unwrap();
    control.wait(|s| s.metadata_entered).await;
    let b = service.try_submit(request(2)).unwrap();
    let c = service.try_submit(request(3)).unwrap();
    control.metadata();
    control
        .wait(|s| s.groups.len() == 1 && s.wake.is_some())
        .await;
    assert_eq!(
        control.state.lock().unwrap().groups[0]
            .iter()
            .map(|(_, seed)| *seed)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(service.snapshot().preparing_requests, 2);
    control.finish();
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    let c = c.wait().await.unwrap();
    assert_eq!([a.receipt.seed, b.receipt.seed, c.receipt.seed], [1, 2, 3]);
    assert!(trace.lock().unwrap().iter().any(|wave| wave.len() == 2));
    assert_eq!(
        control
            .state
            .lock()
            .unwrap()
            .groups
            .iter()
            .map(|g| g.len())
            .collect::<Vec<_>>(),
        [2, 1]
    );
    drop((a, b, c));
    finish(&service, &runtime).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_arrivals_after_cohort_handoff_keep_the_cohort_initialization_protocol() {
    let (service, boundary, trace, runtime, control) =
        start_cohort_with_capacity(Mode::Handoff, 0, ImageBatchPolicy::Auto, Some(3));
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    boundary.state.lock().unwrap().all = true;
    let first = service.try_submit(request(1)).unwrap();
    control.wait(|state| state.metadata_entered).await;
    control.metadata();
    control
        .wait(|state| state.groups.len() == 1 && state.wake.is_some())
        .await;
    // These arrivals cannot join the sealed first phase. They must still use
    // the declared cohort protocol after its downstream Ready owner joins.
    let second = service.try_submit(request(2)).unwrap();
    let third = service.try_submit(request(3)).unwrap();
    control.finish();
    let outputs = [
        first.wait().await.unwrap(),
        second.wait().await.unwrap(),
        third.wait().await.unwrap(),
    ];
    assert_eq!(
        outputs
            .iter()
            .map(|output| output.receipt.seed)
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );
    let cohorts = control
        .state
        .lock()
        .unwrap()
        .groups
        .iter()
        .map(|group| group.iter().map(|(_, seed)| *seed).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(cohorts, [vec![1], vec![2, 3]]);
    assert!(trace.lock().unwrap().iter().any(|wave| wave.len() == 2));
    drop(outputs);
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_handoff_uses_original_pending_downstream_consumer() {
    let (service, boundary, trace, runtime, control) = start_cohort(Mode::Handoff);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    boundary.state.lock().unwrap().all = true;
    let a = service.try_submit(request(1)).unwrap();
    control.wait(|s| s.metadata_entered).await;
    let b = service.try_submit(request(2)).unwrap();
    control.metadata();
    control.wait(|s| s.wake.is_some()).await;
    control.finish();
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    assert_eq!([a.receipt.seed, b.receipt.seed], [1, 2]);
    assert!(!trace.lock().unwrap().is_empty());
    assert_eq!(control.state.lock().unwrap().groups.len(), 1);
    drop((a, b));
    finish(&service, &runtime).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_cancel_snapshot_keeps_shared_phase_and_original_output_credit(
) {
    let (service, boundary, trace, runtime, control) = start_cohort(Mode::Ready);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    boundary.state.lock().unwrap().all = true;
    let host = service.host_account().clone();
    let gate = Arc::new(HostOutputGate {
        host: host.clone(),
        logical: 16,
    });
    let a = service
        .try_submit_with_output(request(1), gate.clone())
        .unwrap();
    control.wait(|s| s.metadata_entered).await;
    let b = service.try_submit_with_output(request(2), gate).unwrap();
    control.metadata();
    control.wait(|s| s.wake.is_some()).await;
    a.cancel();
    control
        .wait(|s| {
            s.snapshots
                .last()
                .is_some_and(|v| v.iter().filter(|(_, c)| *c).count() == 1)
        })
        .await;
    assert_eq!(service.snapshot().preparing_requests, 2);
    assert!(trace.lock().unwrap().is_empty());
    assert!(control.state.lock().unwrap().live);
    assert_eq!(control.state.lock().unwrap().drops, 0);
    assert!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output)
            >= 32
    );
    control.finish();
    assert_eq!(
        a.wait().await.err().expect("cancelled request").kind(),
        ImageServiceErrorKind::Cancelled
    );
    let b = b.wait().await.unwrap();
    assert_eq!(b.receipt.seed, 2);
    drop(b);
    finish(&service, &runtime).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_protocol_failures_preserve_the_original_phase_and_job_backing(
) {
    for mode in [
        Mode::Unknown,
        Mode::Duplicate,
        Mode::Omitted,
        Mode::EarlyHandoff,
        Mode::Fatal,
        Mode::FormatterPanic,
    ] {
        let (service, boundary, trace, _runtime, control) = start_cohort(mode);
        let _release = Release(control.clone());
        let _steps = ReleaseOnDrop(boundary.clone());
        boundary.state.lock().unwrap().all = true;
        let host = service.host_account().clone();
        let gate = Arc::new(HostOutputGate {
            host: host.clone(),
            logical: 16,
        });
        let a = service
            .try_submit_with_output(request(1), gate.clone())
            .unwrap();
        control.wait(|s| s.metadata_entered).await;
        let b = service.try_submit_with_output(request(2), gate).unwrap();
        control.metadata();
        control.wait(|s| s.wake.is_some()).await;
        control.finish();
        for error in [
            a.wait().await.err().expect("protocol error"),
            b.wait().await.err().expect("peer protocol error"),
        ] {
            assert_eq!(error.kind(), ImageServiceErrorKind::Generation);
            let message = error.to_string();
            if matches!(mode, Mode::Fatal) {
                assert!(message.contains("original shared initialization failure"));
            } else if matches!(mode, Mode::FormatterPanic) {
                assert!(message.contains("error formatter panicked"));
            }
            assert!(message.contains("original shared recovery owner retained"));
        }
        let retired = service.shutdown().await.unwrap();
        assert!(retired.has_failure());
        assert!(trace.lock().unwrap().is_empty());
        assert_eq!(control.state.lock().unwrap().drops, 0);
        assert!(
            host.usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::Output)
                >= 32
        );
        let (generator, failure) = retired.into_parts();
        let Some(ImageWorkerFailure::Generation(error)) = failure else {
            panic!("shared failure retained")
        };
        let mut error = error
            .downcast::<super::super::cohorts::CohortFailure>()
            .unwrap();
        let source = Error::source(error.as_ref())
            .unwrap()
            .downcast_ref::<ImageWorkerFailure>()
            .expect("borrow the original typed worker failure");
        if matches!(mode, Mode::Fatal) {
            assert!(source
                .to_string()
                .contains("original shared initialization failure"));
        } else if matches!(mode, Mode::FormatterPanic) {
            let ImageWorkerFailure::Generation(source) = source else {
                panic!("formatter failure must not replace the original generation failure")
            };
            let retained = source
                .downcast_ref::<super::super::FailedOwners>()
                .expect("borrow the original step recovery owner");
            assert!(retained.source.is::<FormatterPanic>());
        }
        let mut phase = error._owner.take().unwrap().into_inner::<Phase>().unwrap();
        phase.retire();
        drop((phase, error, generator));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_zero_native_rejections_are_finite_and_leave_future_requests_usable(
) {
    for mode in [
        Mode::Pressure,
        Mode::PreparationRejected,
        Mode::BeginRejected,
    ] {
        let (service, boundary, _trace, runtime, control) = start_cohort(mode);
        let _release = Release(control.clone());
        let _steps = ReleaseOnDrop(boundary.clone());
        boundary.state.lock().unwrap().all = true;
        let a = service.try_submit(request(1)).unwrap();
        control.wait(|s| s.metadata_entered).await;
        control.metadata();
        let error = tokio::time::timeout(Duration::from_secs(15), a.wait())
            .await
            .unwrap()
            .err()
            .expect("finite rejection");
        assert_eq!(
            error.kind(),
            if matches!(mode, Mode::Pressure) {
                ImageServiceErrorKind::QueueFull
            } else {
                ImageServiceErrorKind::InvalidRequest
            }
        );
        assert!(control.state.lock().unwrap().groups.is_empty());
        control.finish();
        let b = service
            .try_submit(request(2))
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(b.receipt.seed, 2);
        drop(b);
        finish(&service, &runtime).await;
    }
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_retirement_gives_survivor_a_real_commit_before_next_auxiliary(
) {
    let (service, boundary, trace, runtime, control) = start_cohort(Mode::Ready);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    let c = service.try_submit(request(2)).unwrap();
    control.metadata();
    boundary.release(1);
    control
        .wait(|s| s.groups.len() == 1 && s.wake.is_some())
        .await;
    control.finish();
    boundary.wait(2).await;
    assert_eq!(
        control.state.lock().unwrap().groups.len(),
        1,
        "next auxiliary must not pass the survivor commit boundary"
    );
    assert!(trace.lock().unwrap()[1]
        .iter()
        .any(|(seed, _, _)| *seed == 0));
    boundary.release(2);
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    let c = c.wait().await.unwrap();
    let s = control.state.lock().unwrap();
    let second_begin = s
        .events
        .iter()
        .enumerate()
        .filter(|(_, e)| **e == "begin")
        .nth(1)
        .unwrap()
        .0;
    let retired = s.events.iter().position(|e| *e == "retired").unwrap();
    assert!(s.events[retired + 1..second_begin].contains(&"commit"));
    drop(s);
    drop((a, b, c));
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_closing_drives_every_member_to_safe_terminal() {
    let (service, boundary, trace, _runtime, control) = start_cohort(Mode::Ready);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    boundary.state.lock().unwrap().all = true;
    let host = service.host_account().clone();
    let gate = Arc::new(HostOutputGate {
        host: host.clone(),
        logical: 16,
    });
    let a = service
        .try_submit_with_output(request(1), gate.clone())
        .unwrap();
    control.wait(|s| s.metadata_entered).await;
    let b = service.try_submit_with_output(request(2), gate).unwrap();
    control.metadata();
    control.wait(|s| s.wake.is_some()).await;
    let mut shutdown = Box::pin(service.shutdown());
    assert!(futures::poll!(shutdown.as_mut()).is_pending());
    control
        .wait(|s| {
            s.snapshots
                .last()
                .is_some_and(|v| v.iter().all(|(_, cancelled)| *cancelled))
        })
        .await;
    assert_eq!(service.snapshot().preparing_requests, 2);
    assert!(control.state.lock().unwrap().live);
    assert_eq!(control.state.lock().unwrap().drops, 0);
    assert!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output)
            >= 32
    );
    control.finish();
    assert_eq!(
        a.wait().await.err().expect("closing first member").kind(),
        ImageServiceErrorKind::Shutdown
    );
    assert_eq!(
        b.wait().await.err().expect("closing second member").kind(),
        ImageServiceErrorKind::Shutdown
    );
    let retired = shutdown.await.unwrap();
    assert!(!retired.has_failure());
    drop(retired);
    assert!(trace.lock().unwrap().is_empty());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_auto_wait_collects_later_arrival_before_original_deadline() {
    let (service, boundary, _trace, runtime, control) = start_cohort_with_wait(Mode::Ready, 30_000);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    boundary.state.lock().unwrap().all = true;
    let a = service.try_submit(request(1)).unwrap();
    control.wait(|s| s.metadata_entered).await;
    control.metadata();
    control.wait(|s| s.maximum_probed).await;
    // This remains well inside the declared window: no auxiliary installation
    // may happen just because only the first request is currently present.
    assert!(tokio::time::timeout(
        Duration::from_millis(50),
        control.wait(|s| !s.groups.is_empty())
    )
    .await
    .is_err());
    let b = service.try_submit(request(2)).unwrap();
    control.wait(|s| s.wake.is_some()).await;
    assert_eq!(control.state.lock().unwrap().groups[0].len(), 2);
    control.finish();
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    assert_eq!([a.receipt.seed, b.receipt.seed], [1, 2]);
    drop((a, b));
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_zero_wait_does_not_hold_first_request_for_an_arrival() {
    let (service, boundary, _trace, runtime, control) = start_cohort_with_wait(Mode::Ready, 0);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    boundary.state.lock().unwrap().all = true;
    let a = service.try_submit(request(1)).unwrap();
    control.wait(|s| s.metadata_entered).await;
    control.metadata();
    control.wait(|s| s.wake.is_some()).await;
    assert_eq!(control.state.lock().unwrap().groups[0].len(), 1);
    let b = service.try_submit(request(2)).unwrap();
    control.finish();
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    drop((a, b));
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_does_not_apply_idle_wait_to_a_real_survivor() {
    let (service, boundary, trace, runtime, control) = start_cohort_with_wait(Mode::Ready, 30_000);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    control.metadata();
    boundary.release(1);
    // The ordinary test deadline is shorter than the declared idle window;
    // the survivor must not be parked merely to collect a hypothetical peer.
    control
        .wait(|s| s.groups.len() == 1 && s.wake.is_some())
        .await;
    control.finish();
    boundary.wait(2).await;
    assert!(trace.lock().unwrap()[1]
        .iter()
        .any(|(seed, _, _)| *seed == 0));
    boundary.release(2);
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    drop((a, b));
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_ready_lane_commits_before_later_downstream_needs_its_host_owner(
) {
    let (service, boundary, trace, runtime, control) = start_cohort(Mode::BlockedDownstream);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    let survivor = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let a = service.try_submit(request(1)).unwrap();
    let b = service.try_submit(request(2)).unwrap();
    control.metadata();
    boundary.release(1);
    control
        .wait(|s| s.groups.len() == 1 && s.wake.is_some())
        .await;
    assert_eq!(control.state.lock().unwrap().groups[0].len(), 2);
    control.finish();
    boundary.wait(2).await;
    boundary.release(2);
    let outputs = tokio::time::timeout(Duration::from_secs(15), async {
        let survivor = survivor.wait().await.unwrap();
        let a = a.wait().await.unwrap();
        let b = b.wait().await.unwrap();
        (survivor, a, b)
    })
    .await
    .expect("Ready lane must execute before its downstream peer needs the same Host capacity");
    let downstream_after_commits = {
        let state = control.state.lock().unwrap();
        assert!(!state.downstream_waited);
        state
            .initialized
            .iter()
            .find(|(seed, _)| *seed == 2)
            .expect("the later original initialization completes")
            .1
    };
    assert!(trace.lock().unwrap()[..downstream_after_commits]
        .iter()
        .any(|wave| wave
            .iter()
            .any(|(seed, progress, _)| *seed == 1 && *progress > 0)));
    assert!(control.state.lock().unwrap().deferred_ready_join);
    assert!(control.state.lock().unwrap().pressure.is_none());
    assert!(trace
        .lock()
        .unwrap()
        .iter()
        .any(|wave| wave.iter().any(|(seed, _, _)| *seed == 1)));
    drop(outputs);
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_cancel_and_close_wake_the_original_declared_wait() {
    for closing in [false, true] {
        let (service, boundary, _trace, runtime, control) =
            start_cohort_with_wait(Mode::Ready, 30_000);
        let _release = Release(control.clone());
        let _steps = ReleaseOnDrop(boundary.clone());
        boundary.state.lock().unwrap().all = true;
        let job = service.try_submit(request(1)).unwrap();
        control.wait(|s| s.metadata_entered).await;
        control.metadata();
        control.wait(|s| s.maximum_probed).await;
        let mut shutdown = if closing {
            Some(Box::pin(service.shutdown()))
        } else {
            job.cancel();
            None
        };
        let already_retired = match shutdown.as_mut() {
            Some(shutdown) => match futures::poll!(shutdown.as_mut()) {
                std::task::Poll::Ready(receipt) => Some(receipt),
                std::task::Poll::Pending => None,
            },
            None => None,
        };
        let error = tokio::time::timeout(Duration::from_secs(15), job.wait())
            .await
            .expect("withdrawal must wake the 30-second declared collection wait")
            .err()
            .expect("withdrawn job has no image output");
        assert_eq!(
            error.kind(),
            if closing {
                ImageServiceErrorKind::Shutdown
            } else {
                ImageServiceErrorKind::Cancelled
            }
        );
        assert!(
            control.state.lock().unwrap().groups.is_empty(),
            "withdrawn metadata must never acquire an auxiliary owner"
        );
        if let Some(shutdown) = shutdown {
            let retired = match already_retired {
                Some(receipt) => receipt,
                None => shutdown.await,
            }
            .unwrap();
            assert!(!retired.has_failure());
            drop(retired);
        } else {
            finish(&service, &runtime).await;
        }
    }
}

async fn withdraw_at_prefix_wait_boundary(drop_job: bool) {
    let (service, boundary, _trace, runtime, control) = start_cohort_with_wait(Mode::Ready, 30_000);
    let _release = Release(control.clone());
    let _steps = ReleaseOnDrop(boundary.clone());
    boundary.state.lock().unwrap().all = true;
    let mut job = service.try_submit(request(1)).unwrap();
    control.wait(|s| s.metadata_entered).await;
    let signal = job.inner.shared.resources.clone();
    struct ReleaseSignal(Arc<ResourceSignal>);
    impl Drop for ReleaseSignal {
        fn drop(&mut self) {
            self.0.notify();
        }
    }
    let _signal = ReleaseSignal(signal.clone());
    control.state.lock().unwrap().prefix_cancel_wait = Some((signal.clone(), signal.epoch()));
    control.metadata();
    control.wait(|s| s.prefix_cancel_entered).await;
    let response = job.inner.receiver.take().unwrap();
    let cancellation = tokio::task::spawn_blocking(move || {
        if drop_job {
            drop(job);
            None
        } else {
            job.cancel();
            Some(job)
        }
    });
    let error = tokio::time::timeout(Duration::from_secs(15), async {
        // Keep an explicitly cancelled handle alive until its original result;
        // a second notification from Drop must not repair a lost first wake.
        let retained = cancellation.await.unwrap();
        let result = response.await.unwrap();
        drop(retained);
        result
    })
    .await
    .expect("cancellation at the predicate/wait boundary must wake the original wait")
    .err()
    .expect("withdrawn job has no image output");
    assert_eq!(error.kind(), ImageServiceErrorKind::Cancelled);
    assert!(control.state.lock().unwrap().groups.is_empty());
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_cancel_at_prefix_wait_boundary_is_not_lost() {
    withdraw_at_prefix_wait_boundary(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_initialization_cohort_drop_at_prefix_wait_boundary_is_not_lost() {
    withdraw_at_prefix_wait_boundary(true).await;
}
