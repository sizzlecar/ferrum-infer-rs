//! Two-stage preparation. Raw ingress identity remains stable while the actual
//! downstream recipes are built only from a validated, retired expansion.
use super::super::expansion::{
    phase::ExpansionPhaseProgress, product::ProductExpansionPhase, ExpansionHostRequirements,
    NativeTextExpansion, PreparedTextExpansion,
};
use super::*;
use crate::image_generation::{
    service::{
        preparation_error, ImageBatchError, ImageBatchPreparation, ImageDenoisingRequest,
        ImageInitializationOutcome, ImageInitializationProgress, ImageRequestInitialization,
        ImageRequestRef,
    },
    ImageGenerationRequest,
};
use ferrum_execution::{executor_lifecycle::ExecutorShutdownPhase, RuntimeComposition};
use std::task::Waker;
mod cohort;
pub(super) use cohort::synchronous::PreparedRunInitialization;
pub(super) mod downstream;
mod effective;
mod ingress;
mod requirements;
mod vision;
use effective::EffectiveRequest;
use ingress::IngressIdentity;
use requirements::InitializationRequirements;
use vision::{PreparedVisual, VisionInitialization};

type InitResult<T> = std::result::Result<T, ImageBatchError>;

pub(super) enum PreparedInitialization<R: DeviceRuntime> {
    Raw(PreparedContinuousRequest<R>),
    Expansion(PreparedExpansion<R>),
}
pub(super) struct PreparedExpansion<R: DeviceRuntime> {
    correlation: Option<ferrum_types::ImageRequestCorrelation>,
    pipeline: Arc<()>,
    identity: IngressIdentity,
    job: u64,
    assets: Arc<NativeTextExpansion>,
    prepared: PreparedTextExpansion,
    requirements: InitializationRequirements,
    visual: Option<PreparedVisual<R>>,
    ingress_backing_bytes: usize,
    phase: Option<HostMemoryScope>,
    output_credit: usize,
    receipts: Vec<ImageStageReceipt>,
}
struct Initialization<R: DeviceRuntime> {
    correlation: Option<ferrum_types::ImageRequestCorrelation>,
    prompt_trace: Option<u64>,
    pipeline: Arc<()>,
    identity: IngressIdentity,
    job: u64,
    output_credit: usize,
    receipts: Vec<ImageStageReceipt>,
    state: InitializationState<R>,
}
enum InitializationState<R: DeviceRuntime> {
    Vision(VisionInitialization<R>),
    Expansion(ProductExpansionPhase<R>),
    PrepareDownstream {
        effective: EffectiveRequest,
    },
    PreparingDownstream(
        downstream::PreparationTask<(EffectiveRequest, PreparedContinuousRequest<R>)>,
    ),
    Downstream {
        effective: EffectiveRequest,
        prepared: PreparedContinuousRequest<R>,
    },
    Done,
}

struct ExpansionCleanupFailure {
    error: ferrum_types::FerrumError,
    cleanup: ferrum_execution::executor_lifecycle::ExecutorShutdownObserver,
}
impl std::fmt::Display for ExpansionCleanupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prompt expansion: {}; cleanup {:?}",
            self.error,
            self.cleanup.snapshot().phase()
        )
    }
}
impl std::error::Error for ExpansionCleanupFailure {}
impl std::fmt::Debug for ExpansionCleanupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn bind_initialization_correlation(
        &self,
        prepared: &mut ImageBatchPreparation,
        correlation: ferrum_types::ImageRequestCorrelation,
    ) {
        if let Ok(prepared) = prepared.get_mut::<PreparedInitialization<R>>() {
            match prepared {
                PreparedInitialization::Raw(prepared) => {
                    prepared.link_correlation(self, correlation)
                }
                PreparedInitialization::Expansion(prepared) => {
                    prepared.correlation = Some(correlation)
                }
            }
        }
    }
    /// The synchronous entrypoint drives exactly the same initialization
    /// transitions. The service uses individual advancements between forwards.
    pub(super) fn start_initialization_blocking(
        &mut self,
        prepared: ImageBatchPreparation,
        input: ImageRequestRef<'_>,
    ) -> InitResult<ImageDenoisingRequest> {
        let initialization = match self.begin_initialization(prepared, input)? {
            ImageInitializationOutcome::Ready(request) => return Ok(request),
            ImageInitializationOutcome::Pending(initialization) => initialization,
        };
        self.continue_initialization_blocking(initialization, input)
    }
    pub(super) fn continue_initialization_blocking(
        &mut self,
        mut initialization: ImageRequestInitialization,
        input: ImageRequestRef<'_>,
    ) -> InitResult<ImageDenoisingRequest> {
        struct ThreadWake(std::thread::Thread);
        impl std::task::Wake for ThreadWake {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }
        let wake = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let _subscription = self.execution.host.subscribe_capacity(wake.clone());
        loop {
            match self.advance_initialization(&mut initialization, input, false, &wake)? {
                ImageInitializationProgress::Ready(request) => return Ok(request),
                ImageInitializationProgress::Rejected { diagnostic } => {
                    return Err(ImageBatchError::Rejected(diagnostic))
                }
                ImageInitializationProgress::Cancelled => {
                    return Err(ImageBatchError::Rejected(
                        "image initialization cancelled".into(),
                    ))
                }
                ImageInitializationProgress::Pending {
                    made_progress: false,
                } => {
                    if matches!(
                        &initialization.get_mut::<Initialization<R>>()?.state,
                        InitializationState::Downstream { .. }
                    ) {
                        // A synchronous caller may itself hold older lanes or
                        // output credit. Waiting here cannot advance those
                        // owners; preserve the existing finite run admission.
                        return Err(ImageBatchError::Overloaded("expanded prompt cannot obtain image phase Host capacity in this synchronous call".into()));
                    }
                    std::thread::park();
                }
                ImageInitializationProgress::Pending {
                    made_progress: true,
                } => (),
            }
        }
    }
    fn expansion_composition(&self) -> RuntimeComposition<R> {
        RuntimeComposition::from_shared(
            self.execution.runtime.clone(),
            self.execution.registry.clone(),
            self.execution.materializers.clone(),
            self.execution.catalog.clone(),
        )
    }

    pub(super) fn prepare_initialization(
        &mut self,
        input: ImageRequestRef<'_>,
    ) -> InitResult<ImageBatchPreparation> {
        let Some(configured) = self.execution.sources.assets.expansion.as_ref() else {
            return self
                .prepare_continuous_image_request(input)
                .map(|p| ImageBatchPreparation::new(PreparedInitialization::Raw(p)))
                .map_err(preparation_error);
        };
        let request = match input {
            ImageRequestRef::Generate(request) => request,
            ImageRequestRef::Edit(request) => &request.generation,
        };
        let validate = || -> Result<_> {
            ensure!(!self.is_stopped(), "H3 image pipeline is stopped");
            super::super::request::validate_request_metadata(
                request,
                &self.execution.config,
                self.execution.sources.assets.spatial_scale(),
            )?;
            let mut options = configured.options.clone();
            // Reuse the actual image runtime ceiling, respecting a smaller
            // explicit causal limit. This creates no independent allocation.
            let capacity = usize::try_from(self.execution.config.device_capacity_bytes)?;
            options.engine.memory.usable_capacity_bytes = Some(
                options
                    .engine
                    .memory
                    .usable_capacity_bytes
                    .unwrap_or(capacity)
                    .min(capacity),
            );
            let composition = self.expansion_composition();
            let (prepared, visual) = match input {
                ImageRequestRef::Generate(_) => (
                    configured.assets.prepare_text(
                        &request.prompt,
                        request.width,
                        request.height,
                        options.max_new_tokens,
                        options.prefill_chunk_tokens,
                        options.engine,
                    )?,
                    None,
                ),
                ImageRequestRef::Edit(edit) => {
                    let visual = PreparedVisual::prepare(
                        &configured.assets,
                        &composition,
                        &self.execution.host,
                        &self.execution.config,
                        edit,
                    )?;
                    let prepared = configured.assets.prepare_edits(
                        &request.prompt,
                        (request.width, request.height),
                        &visual.grids(),
                        visual.plan().config().architecture().merge_size(),
                        &options,
                    )?;
                    (prepared, Some(visual))
                }
            };
            let causal = configured.assets.host_requirements(
                &composition,
                &prepared,
                &self.execution.host,
            )?;
            let visual_requirements = visual
                .as_ref()
                .map(|visual| visual.requirements(&self.execution.host))
                .transpose()?;
            let decoded_rgba = visual
                .as_ref()
                .map(|visual| visual.decoded_backing_bytes(&self.execution.host))
                .transpose()?
                .unwrap_or(0);
            let requirements =
                InitializationRequirements::new(causal, visual_requirements, decoded_rgba)?;
            let ingress_backing_bytes = match input {
                ImageRequestRef::Edit(request) => request.encoded_backing_bytes(),
                ImageRequestRef::Generate(_) => 0,
            };
            let mut peak = requirements.claims().to_vec();
            peak.push((HostMemoryClass::InputPreprocess, ingress_backing_bytes));
            self.execution
                .check_scope_capacity_with_retained_diagnostics(&peak)?;
            Ok((prepared, requirements, visual, ingress_backing_bytes))
        };
        let (prepared, requirements, visual, ingress_backing_bytes) =
            validate().map_err(preparation_error)?;
        let job = self.execution.next_job;
        self.execution.next_job = job
            .checked_add(1)
            .ok_or_else(|| ImageBatchError::Rejected("H3 request identity exhausted".into()))?;
        Ok(ImageBatchPreparation::new(
            PreparedInitialization::<R>::Expansion(PreparedExpansion {
                correlation: None,
                pipeline: self.execution.service_preparation_identity.clone(),
                identity: IngressIdentity::new(input),
                job,
                assets: configured.assets.clone(),
                prepared,
                requirements,
                visual,
                ingress_backing_bytes,
                phase: None,
                output_credit: 0,
                receipts: vec![],
            }),
        ))
    }

    pub(super) fn admit_initialization(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        output_credit: usize,
    ) -> InitResult<bool> {
        match prepared.get_mut::<PreparedInitialization<R>>()? {
            PreparedInitialization::Raw(prepared) => self
                .try_admit_continuous_request(prepared, output_credit)
                .map_err(preparation_error),
            PreparedInitialization::Expansion(prepared) => {
                if !Arc::ptr_eq(
                    &prepared.pipeline,
                    &self.execution.service_preparation_identity,
                ) {
                    return Err(ImageBatchError::Rejected(
                        "prompt preparation belongs to another pipeline".into(),
                    ));
                }
                if prepared.phase.is_some() {
                    return Ok(true);
                }
                let ready = self
                    .execution
                    .admit_auxiliary_peak(
                        prepared.requirements.startup_device_bytes,
                        &mut prepared.receipts,
                    )
                    .map_err(|failure| self.h3_step_error(failure))?;
                if !ready {
                    return Err(ImageBatchError::InitializationAwaitDenoisingProgress {
                        diagnostic: "prompt expansion is waiting for original active image owners to release capacity".into(),
                    });
                }
                self.reserve_expansion_host(prepared, output_credit)
            }
        }
    }

    fn reserve_expansion_host(
        &self,
        prepared: &mut PreparedExpansion<R>,
        output_credit: usize,
    ) -> InitResult<bool> {
        let claims = prepared.requirements.claims();
        let mut peak = claims.to_vec();
        peak.push((HostMemoryClass::Output, output_credit));
        peak.push((
            HostMemoryClass::InputPreprocess,
            prepared.ingress_backing_bytes,
        ));
        self.execution
            .check_scope_capacity_with_retained_diagnostics(&peak)
            .map_err(|e| preparation_error(e.into()))?;
        let phase = match self.execution.host.reserve_scope(&claims) {
            Ok(phase) => phase,
            Err(error) if error.is_temporary_capacity() => return Ok(false),
            Err(error) => return Err(preparation_error(error.into())),
        };
        prepared.output_credit = output_credit;
        prepared.phase = Some(phase);
        Ok(true)
    }

    pub(super) fn admit_initialization_at_boundary(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        output_credit: usize,
        batch: Option<&mut ContinuousBatch<R>>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> InitResult<bool> {
        let diagnostic = match self.admit_initialization(prepared, output_credit) {
            Err(ImageBatchError::InitializationAwaitDenoisingProgress { diagnostic }) => diagnostic,
            other => return other,
        };
        let waiting = || ImageBatchError::InitializationAwaitDenoisingProgress {
            diagnostic: diagnostic.clone(),
        };
        let Some(batch) = batch else {
            return Err(waiting());
        };
        if requests.is_empty() || batch.suspended {
            return Err(waiting());
        }
        let PreparedInitialization::Expansion(expansion) =
            prepared.get_mut::<PreparedInitialization<R>>()?
        else {
            return Err(waiting());
        };
        // Reserve the real auxiliary phase before destroying a runnable native
        // installation. The old cohort keeps its independent progress floor.
        if !self.reserve_expansion_host(expansion, output_credit)? {
            return Ok(false);
        }
        let suspended = self
            .suspend_continuous_batch(batch, requests)
            .map_err(|failure| self.suspension_error(failure))?;
        if !suspended {
            drop(expansion.phase.take());
            return Err(waiting());
        }
        let ready = self
            .execution
            .admit_auxiliary_peak(
                expansion.requirements.startup_device_bytes,
                &mut expansion.receipts,
            )
            .map_err(|failure| self.suspension_error(failure))?;
        if !ready {
            // No auxiliary native resources were acquired. Recreate the old
            // runnable cohort before returning an ordinary candidate refusal.
            drop(expansion.phase.take());
            self.restore_continuous_batch(batch, requests)
                .map_err(|failure| self.suspension_error(failure))?;
            return Err(waiting());
        }
        Ok(true)
    }

    pub(super) fn initialization_has_auxiliary_owner(
        &self,
        initialization: &ImageRequestInitialization,
    ) -> bool {
        initialization
            .get::<Initialization<R>>()
            .map_or(true, |initialization| {
                matches!(
                    initialization.state,
                    InitializationState::Vision(_) | InitializationState::Expansion(_)
                )
            })
    }

    pub(super) fn suspension_error(
        &mut self,
        failure: ImageGenerationFailure<R>,
    ) -> ImageBatchError {
        self.execution.stopped = true;
        ImageBatchError::Fatal(Box::new(crate::image_generation::ImageGenerationError {
            failure,
            completed_stages: vec![],
            completed_steps: vec![],
        }))
    }

    pub(super) fn begin_initialization(
        &mut self,
        prepared: ImageBatchPreparation,
        input: ImageRequestRef<'_>,
    ) -> InitResult<ImageInitializationOutcome> {
        match prepared.into_inner::<PreparedInitialization<R>>()? {
            PreparedInitialization::Raw(prepared) => self
                .start_continuous_image_request(prepared, input)
                .map(ImageDenoisingRequest::new)
                .map(ImageInitializationOutcome::Ready)
                .map_err(|e| self.h3_initialization_error(e)),
            PreparedInitialization::Expansion(mut prepared) => {
                if !Arc::ptr_eq(
                    &prepared.pipeline,
                    &self.execution.service_preparation_identity,
                ) || !prepared.identity.matches(input)
                {
                    return Err(ImageBatchError::Rejected(
                        "prompt initialization ingress identity changed".into(),
                    ));
                }
                let phase = prepared.phase.take().ok_or_else(|| {
                    ImageBatchError::Rejected(
                        "prompt initialization requires its original Host phase".into(),
                    )
                })?;
                let prompt_trace = self.trace_phase_started(
                    prepared.correlation,
                    None,
                    crate::image_generation::continuous_trace::ImageTracePhase::PromptExpansion,
                );
                let state = match prepared.visual {
                    Some(visual) => Ok(InitializationState::Vision(VisionInitialization::new(
                        prepared.assets,
                        prepared.prepared,
                        visual,
                        phase,
                        prepared.requirements,
                    ))),
                    None => self
                        .initialize_expansion(
                            prepared.assets,
                            prepared.prepared,
                            phase,
                            prepared.job,
                            prepared.correlation,
                        )
                        .map(InitializationState::Expansion),
                };
                let state = match state {
                    Ok(state) => state,
                    Err(error) => {
                        self.trace_phase_finished(prompt_trace, crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed);
                        return Err(error);
                    }
                };
                Ok(ImageInitializationOutcome::Pending(
                    ImageRequestInitialization::new(Initialization::<R> {
                        correlation: prepared.correlation,
                        prompt_trace,
                        pipeline: prepared.pipeline,
                        identity: prepared.identity,
                        job: prepared.job,
                        output_credit: prepared.output_credit,
                        receipts: prepared.receipts,
                        state,
                    }),
                ))
            }
        }
    }

    pub(super) fn advance_initialization(
        &mut self,
        opaque: &mut ImageRequestInitialization,
        input: ImageRequestRef<'_>,
        cancelled: bool,
        wake: &Waker,
    ) -> InitResult<ImageInitializationProgress> {
        self.advance_initialization_at_boundary(opaque, input, cancelled, wake, None)
    }

    pub(super) fn advance_initialization_at_boundary(
        &mut self,
        opaque: &mut ImageRequestInitialization,
        input: ImageRequestRef<'_>,
        cancelled: bool,
        wake: &Waker,
        membership: Option<(&mut ContinuousBatch<R>, &[&mut ContinuousRequest<R>])>,
    ) -> InitResult<ImageInitializationProgress> {
        let initialization = opaque.get_mut::<Initialization<R>>()?;
        if !Arc::ptr_eq(
            &initialization.pipeline,
            &self.execution.service_preparation_identity,
        ) || !initialization.identity.matches(input)
        {
            self.execution.stopped = true;
            return Err(ImageBatchError::Fatal(Box::new(
                ferrum_types::FerrumError::request_validation(
                    "live prompt initialization ingress identity changed",
                ),
            )));
        }
        match &mut initialization.state {
            InitializationState::Vision(_) => {
                let InitializationState::Vision(visual) =
                    std::mem::replace(&mut initialization.state, InitializationState::Done)
                else {
                    unreachable!()
                };
                if cancelled {
                    self.trace_phase_finished(initialization.prompt_trace.take(),crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Cancelled);
                    return Ok(ImageInitializationProgress::Cancelled);
                }
                match self.advance_visual_initialization(
                    visual,
                    initialization.job,
                    initialization.correlation,
                )? {
                    Ok(phase) => {
                        initialization.state = InitializationState::Expansion(phase);
                        Ok(ImageInitializationProgress::Pending {
                            made_progress: true,
                        })
                    }
                    Err(diagnostic) => {
                        self.trace_phase_finished(initialization.prompt_trace.take(),crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed);
                        Ok(ImageInitializationProgress::Rejected { diagnostic })
                    }
                }
            }
            InitializationState::Expansion(phase) => {
                match phase.advance(cancelled, wake) {
                    Ok(ExpansionPhaseProgress::Pending { made_progress }) => {
                        Ok(ImageInitializationProgress::Pending { made_progress })
                    }
                    Ok(ExpansionPhaseProgress::Cancelled) => {
                        self.trace_phase_finished(initialization.prompt_trace.take(),crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Cancelled);
                        initialization.state = InitializationState::Done;
                        Ok(ImageInitializationProgress::Cancelled)
                    }
                    Ok(ExpansionPhaseProgress::Expanded(text)) => {
                        self.trace_phase_finished(initialization.prompt_trace.take(),crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed);
                        let effective = EffectiveRequest::from_expanded(input, text);
                        let prepared = match self.prepare_continuous_image_request_with_job(
                            effective.as_ref(),
                            initialization.job,
                        ) {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                initialization.state = InitializationState::Done;
                                return Ok(ImageInitializationProgress::Rejected {
                                    diagnostic: format!(
                                        "expanded prompt cannot prepare image: {error:#}"
                                    ),
                                });
                            }
                        };
                        if let Some(correlation) = initialization.correlation {
                            prepared.link_correlation(self, correlation);
                        }
                        initialization.state = InitializationState::Downstream {
                            effective,
                            prepared,
                        };
                        Ok(ImageInitializationProgress::Pending {
                            made_progress: true,
                        })
                    }
                    Err(error) => {
                        self.trace_phase_finished(initialization.prompt_trace.take(),crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed);
                        let cleanup = phase.cleanup();
                        if cleanup.snapshot().phase() == ExecutorShutdownPhase::Retired {
                            initialization.state = InitializationState::Done;
                            Ok(ImageInitializationProgress::Rejected {
                                diagnostic: error.to_string(),
                            })
                        } else {
                            self.execution.stopped = true;
                            Err(ImageBatchError::Fatal(Box::new(ExpansionCleanupFailure {
                                error,
                                cleanup,
                            })))
                        }
                    }
                }
            }
            InitializationState::PrepareDownstream { .. } => {
                if cancelled {
                    initialization.state = InitializationState::Done;
                    return Ok(ImageInitializationProgress::Cancelled);
                }
                // Diagnostic callbacks stay on their original synchronous path.
                // A cohort may prepare one request while existing lanes execute.
                if self.main_text_diagnostics.is_none()
                    && self.main_vision_diagnostics.is_none()
                    && self.final_latent_diagnostics.is_none()
                    && self.dit_input_diagnostics.is_none()
                {
                    let Some(permit) = downstream::PreparationPermit::try_acquire(
                        &self.downstream_preparation,
                        Some(wake),
                    ) else {
                        return Ok(ImageInitializationProgress::Pending {
                            made_progress: false,
                        });
                    };
                    let InitializationState::PrepareDownstream { effective } =
                        std::mem::replace(&mut initialization.state, InitializationState::Done)
                    else {
                        unreachable!()
                    };
                    let fork = self.metadata_preparation_fork();
                    let job = initialization.job;
                    let task = permit
                        .spawn(move || {
                            let prepared = fork.prepare_continuous_image_request_with_job(
                                effective.as_ref(),
                                job,
                            )?;
                            Ok((effective, prepared))
                        })
                        .map_err(preparation_error)?;
                    initialization.state = InitializationState::PreparingDownstream(task);
                    return Ok(ImageInitializationProgress::Pending {
                        made_progress: true,
                    });
                }
                let InitializationState::PrepareDownstream { effective } =
                    std::mem::replace(&mut initialization.state, InitializationState::Done)
                else {
                    unreachable!()
                };
                if cancelled {
                    return Ok(ImageInitializationProgress::Cancelled);
                }
                let prepared = match self.prepare_continuous_image_request_with_job(
                    effective.as_ref(),
                    initialization.job,
                ) {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        return Ok(ImageInitializationProgress::Rejected {
                            diagnostic: format!("expanded prompt cannot prepare image: {error:#}"),
                        });
                    }
                };
                if let Some(correlation) = initialization.correlation {
                    prepared.link_correlation(self, correlation);
                }
                initialization.state = InitializationState::Downstream {
                    effective,
                    prepared,
                };
                Ok(ImageInitializationProgress::Pending {
                    made_progress: true,
                })
            }
            InitializationState::PreparingDownstream(task) => {
                // Cancellation waits for this metadata-only owner to retire;
                // no thread or source backing is detached from its request.
                if cancelled {
                    initialization.state = InitializationState::Done;
                    return Ok(ImageInitializationProgress::Cancelled);
                }
                match task.poll() {
                    std::task::Poll::Pending => Ok(ImageInitializationProgress::Pending {
                        made_progress: false,
                    }),
                    std::task::Poll::Ready(Ok((effective, prepared))) => {
                        if let Err(error) = prepared.validate_device_capacity(self) {
                            initialization.state = InitializationState::Done;
                            return Ok(ImageInitializationProgress::Rejected {
                                diagnostic: format!("expanded prompt image capacity: {error:#}"),
                            });
                        }
                        if let Some(correlation) = initialization.correlation {
                            prepared.link_correlation(self, correlation);
                        }
                        initialization.state = InitializationState::Downstream {
                            effective,
                            prepared,
                        };
                        Ok(ImageInitializationProgress::Pending {
                            made_progress: true,
                        })
                    }
                    std::task::Poll::Ready(Err(error)) => {
                        initialization.state = InitializationState::Done;
                        Ok(ImageInitializationProgress::Rejected {
                            diagnostic: format!("expanded prompt cannot prepare image: {error:#}"),
                        })
                    }
                }
            }
            InitializationState::Downstream { prepared, .. } => {
                if cancelled {
                    initialization.state = InitializationState::Done;
                    return Ok(ImageInitializationProgress::Cancelled);
                }
                let admitted = match self
                    .try_admit_continuous_request(prepared, initialization.output_credit)
                {
                    Ok(admitted) => admitted,
                    Err(error) => {
                        initialization.state = InitializationState::Done;
                        return Ok(ImageInitializationProgress::Rejected {
                            diagnostic: format!("expanded prompt image admission: {error:#}"),
                        });
                    }
                };
                if !admitted {
                    return Ok(ImageInitializationProgress::Pending {
                        made_progress: false,
                    });
                }
                if let Some((batch, members)) = membership {
                    batch.prepare_join(self, members, prepared.geometry());
                }
                let InitializationState::Downstream {
                    effective,
                    prepared,
                } = std::mem::replace(&mut initialization.state, InitializationState::Done)
                else {
                    unreachable!()
                };
                let mut request =
                    match self.start_continuous_image_request(prepared, effective.as_ref()) {
                        Ok(request) => request,
                        Err(error) => {
                            return match self.h3_initialization_error(error) {
                                fatal @ ImageBatchError::Fatal(_) => Err(fatal),
                                rejected => Ok(ImageInitializationProgress::Rejected {
                                    diagnostic: rejected.to_string(),
                                }),
                            }
                        }
                    };
                request
                    .stages
                    .splice(0..0, std::mem::take(&mut initialization.receipts));
                Ok(ImageInitializationProgress::Ready(
                    ImageDenoisingRequest::new(request),
                ))
            }
            InitializationState::Done => Err(ImageBatchError::Rejected(
                "image initialization is terminal".into(),
            )),
        }
    }
}
