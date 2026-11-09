//! Initialization keeps request-persistent and transient work authorization
//! separate before the first upload. Every model stage fences before return.
use super::super::editing::{PreparedReferenceEdit, ReferenceState};
use super::super::stages::InitialNoiseExecution;
use super::*;
use crate::image_generation::service::ImageRequestRef;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest, ImageStage};
use ferrum_models::p_h3_image::{PH3ImageEditPixels, PH3ImageRefinedTextCache};
use sha2::{Digest, Sha256};

pub(in crate::image_generation) struct PreparedContinuousRequest<R: DeviceRuntime> {
    pipeline: Arc<()>,
    identity: [u8; 32],
    prepared: super::super::stages::PreparedGeneration<R>,
    edit: Option<PreparedReferenceEdit<R>>,
    persistent_claims: [(HostMemoryClass, usize); 1],
    work_budget: ExecutionHostBudget,
    persistent: Option<HostMemoryScope>,
    work: Option<HostMemoryScope>,
    backing: Vec<HostBackingPreparation>,
    job: u64,
    main_vision_reuse: Option<super::super::vision::reuse::MainVisionReuseBinding>,
}
impl<R: DeviceRuntime> PreparedContinuousRequest<R> {
    pub(super) fn main_vision_output_capacity(&self, host: &HostMemoryAccount) -> Result<usize> {
        // This optional cache owns one Picture's readbacks. Multi-image edits
        // keep their independently admitted outputs on the normal execution path.
        if self
            .edit
            .as_ref()
            .is_some_and(|edit| !edit.additional.is_empty())
        {
            return Ok(0);
        }
        let Some(visual) = &self.prepared.conditioning.visual else {
            return Ok(0);
        };
        let ports = host_budget::ports(visual)?;
        ensure!(
            ports.outputs().len() == 4,
            "main vision requires four declared outputs"
        );
        ports.outputs().iter().try_fold(0usize, |sum, port| {
            ensure!(
                port.tensor().element_type() == ElementType::F32,
                "main vision reuse requires F32 outputs"
            );
            sum.checked_add(
                host.backing_bytes(usize::try_from(port.tensor().minimum_storage_bytes()?)?)?,
            )
            .context("main vision output capacity overflows")
        })
    }
    pub(super) fn bind_main_vision_reuse(
        &mut self,
        cohort: Arc<super::super::vision::reuse::CohortMainVisionReuse>,
    ) {
        if let Some(edit) = &self.edit {
            if !edit.additional.is_empty() {
                return;
            }
            self.main_vision_reuse = Some(super::super::vision::reuse::MainVisionReuseBinding {
                cohort,
                encoded: edit.encoded.clone(),
            });
        }
    }
    pub(super) fn link_correlation(
        &self,
        pipeline: &PH3ImagePipeline<R>,
        correlation: ferrum_types::ImageRequestCorrelation,
    ) {
        pipeline.trace_request_link(
            correlation,
            &self.prepared.conditioning.recipe.geometry.identity,
        );
    }
}
fn generation(request: ImageRequestRef<'_>) -> &ImageGenerationRequest {
    match request {
        ImageRequestRef::Generate(r) => r,
        ImageRequestRef::Edit(r) => &r.generation,
    }
}
pub(super) fn request_identity(input: ImageRequestRef<'_>) -> [u8; 32] {
    let request = generation(input);
    let mut digest = Sha256::new();
    digest.update((request.prompt.len() as u64).to_le_bytes());
    digest.update(request.prompt.as_bytes());
    digest.update(request.width.to_le_bytes());
    digest.update(request.height.to_le_bytes());
    digest.update(request.seed.to_le_bytes());
    digest.update(request.n.to_le_bytes());
    match input {
        ImageRequestRef::Generate(_) => digest.update([0]),
        ImageRequestRef::Edit(edit) => {
            digest.update([1]);
            digest.update((edit.image.len() as u64).to_le_bytes());
            digest.update(edit.image.bytes());
            if !edit.references.is_empty() {
                digest.update((edit.references.len() as u64).to_le_bytes());
                for image in &edit.references {
                    digest.update((image.len() as u64).to_le_bytes());
                    digest.update(image.bytes());
                }
            }
        }
    }
    digest.finalize().into()
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(in crate::image_generation) fn prepare_continuous_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<PreparedContinuousRequest<R>> {
        self.prepare_continuous_image_request(ImageRequestRef::Generate(request))
    }

    pub(in crate::image_generation) fn prepare_continuous_image_request(
        &mut self,
        input: ImageRequestRef<'_>,
    ) -> Result<PreparedContinuousRequest<R>> {
        ensure!(!self.is_stopped(), "H3 image pipeline is stopped");
        let job = self.execution.next_job;
        self.execution.next_job = job
            .checked_add(1)
            .context("H3 request identity exhausted")?;
        self.prepare_continuous_image_request_with_job(input, job)
    }

    pub(super) fn prepare_continuous_image_request_with_job(
        &self,
        input: ImageRequestRef<'_>,
        job: u64,
    ) -> Result<PreparedContinuousRequest<R>> {
        use super::super::trace::metadata_phase;
        let identity = RequestIdentity::new(format!("request.image.h3.{job}"))?;
        let (prepared, edit) = metadata_phase("request_recipe", || -> Result<_> {
            Ok(match input {
                ImageRequestRef::Generate(request) => {
                    (self.prepare_generation(request, identity)?, None)
                }
                ImageRequestRef::Edit(request) => {
                    let (prepared, edit) = self.prepare_reference_edit(request, identity)?;
                    (prepared, Some(edit))
                }
            })
        })?;
        let mut stages = vec![
            &prepared.conditioning.encoder,
            &prepared.conditioning.connector,
        ];
        stages.extend(prepared.noise.stage());
        stages.extend([
            &prepared.denoising.install.stage,
            &prepared.denoising.refine.stage,
            &prepared.denoising.steady.stage,
        ]);
        stages.extend(&prepared.decoder.stages);
        if let Some(visual) = &prepared.conditioning.visual {
            stages.push(visual);
        }
        if let Some(edit) = &edit {
            for image in edit.images() {
                stages.extend([&image.encoder.stage, &image.noise.stage]);
            }
        }
        stages.extend(&prepared.conditioning.additional_visuals);
        metadata_phase("device_capacity", || {
            self.execution.validate_capacity_with_prefix(&stages, None)
        })?;
        let requirements = metadata_phase("host_requirements", || {
            self.continuous_host_requirements(&prepared, edit.as_ref())
        })?;
        // A single accepted request must have a possible complete lifecycle,
        // even though phase work is reserved independently of persistent data.
        let mut whole = requirements.whole_request_budget()?.claims().to_vec();
        if let Some(edit) = &edit {
            whole.push((HostMemoryClass::InputPreprocess, edit.ingress_bytes()?));
        }
        self.execution
            .check_scope_capacity_with_retained_diagnostics(&whole)?;
        let persistent_claims = requirements.persistent_claims();
        let work_budget = requirements.initialization;
        let claims = persistent_claims
            .iter()
            .copied()
            .chain(work_budget.claims())
            .collect::<Vec<_>>();
        self.execution
            .check_scope_capacity_with_retained_diagnostics(&claims)?;
        Ok(PreparedContinuousRequest {
            pipeline: self.execution.service_preparation_identity.clone(),
            identity: request_identity(input),
            prepared,
            edit,
            persistent_claims,
            work_budget,
            persistent: None,
            work: None,
            backing: requirements.initialization_backing,
            job,
            main_vision_reuse: None,
        })
    }

    /// `false` leaves the exact metadata preparation intact. Existing active
    /// cohorts retain their separate forward credit while this optional join
    /// waits; no reservation/release loop creates its own capacity wakeups.
    pub(in crate::image_generation) fn try_admit_continuous_request(
        &self,
        prepared: &mut PreparedContinuousRequest<R>,
        retained_output_bytes: usize,
    ) -> Result<bool> {
        ensure!(
            Arc::ptr_eq(
                &prepared.pipeline,
                &self.execution.service_preparation_identity
            ),
            "H3 request preparation belongs to another execution owner"
        );
        if prepared.persistent.is_some() && prepared.work.is_some() {
            return Ok(true);
        }
        ensure!(
            prepared.persistent.is_none() && prepared.work.is_none(),
            "H3 request admission has only one of its required scopes"
        );
        let requirements =
            self.continuous_host_requirements(&prepared.prepared, prepared.edit.as_ref())?;
        ensure!(
            requirements.persistent_claims() == prepared.persistent_claims,
            "H3 persistent request storage changed after preparation"
        );
        prepared.work_budget = requirements.initialization;
        prepared.backing = requirements.initialization_backing;
        let claims = prepared
            .persistent_claims
            .iter()
            .copied()
            .chain(prepared.work_budget.claims())
            .collect::<Vec<_>>();
        let mut own_peak = claims.clone();
        if let Some(edit) = &prepared.edit {
            own_peak.push((HostMemoryClass::InputPreprocess, edit.ingress_backing_bytes));
        }
        own_peak.push((HostMemoryClass::Output, retained_output_bytes));
        if let Some(binding) = &prepared.main_vision_reuse {
            // Independent retained readbacks overlap the next request's TE
            // Output peak. This is real additional credit under the same cap.
            own_peak.push((HostMemoryClass::Output, binding.cohort.capacity()));
        }
        if let Some(binding) = &prepared.main_vision_reuse {
            binding.check_request_capacity(
                &self.execution.host,
                &own_peak,
                self.execution.retained_diagnostic_output_bytes(),
            )?;
        } else {
            self.execution
                .check_scope_capacity_with_retained_diagnostics(&own_peak)?;
        }
        let mut admission = claims.clone();
        if let Some(binding) = &prepared.main_vision_reuse {
            admission.push((
                HostMemoryClass::Output,
                binding.cohort.unreserved_capacity()?,
            ));
        }
        // Probe the entire first-use grant before reserving either request
        // scope; a temporary miss must not churn work credit/wakeups.
        match self.execution.host.check_scope_admission(&admission) {
            Ok(()) => (),
            Err(source) if source.is_temporary_capacity() => return Ok(false),
            Err(source) => return Err(source.into()),
        }
        let persistent = match self
            .execution
            .host
            .reserve_scope(&prepared.persistent_claims)
        {
            Ok(scope) => scope,
            Err(source) if source.is_temporary_capacity() => return Ok(false),
            Err(source) => return Err(source.into()),
        };
        let work = match self
            .execution
            .host
            .reserve_scope(&prepared.work_budget.claims())
        {
            Ok(scope) => scope,
            Err(source) if source.is_temporary_capacity() => return Ok(false),
            Err(source) => return Err(source.into()),
        };
        if let Some(binding) = &prepared.main_vision_reuse {
            if !binding.cohort.try_admit(
                &self.execution.host,
                &self.execution.service_preparation_identity,
            )? {
                return Ok(false);
            }
        }
        prepared.persistent = Some(persistent);
        prepared.work = Some(work);
        Ok(true)
    }

    pub(in crate::image_generation) fn start_continuous_request(
        &mut self,
        prepared: PreparedContinuousRequest<R>,
        request: &ImageGenerationRequest,
    ) -> std::result::Result<ContinuousRequest<R>, ImageGenerationError<R>> {
        self.start_continuous_image_request(prepared, ImageRequestRef::Generate(request))
    }

    pub(in crate::image_generation) fn start_continuous_image_request(
        &mut self,
        mut prepared: PreparedContinuousRequest<R>,
        input: ImageRequestRef<'_>,
    ) -> std::result::Result<ContinuousRequest<R>, ImageGenerationError<R>> {
        let empty = |source| ImageGenerationError {
            failure: ImageGenerationFailure::Preflight(source),
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        };
        ensure_prepared(self, &prepared, input).map_err(empty)?;
        let request = generation(input);
        let persistent = prepared
            .persistent
            .take()
            .expect("checked persistent scope");
        let work = prepared.work.take().expect("checked minimum work scope");
        // Decode/resize can fail on hostile or corrupt ingress. No stage,
        // native pin or device work exists yet, so these are local rejections.
        let pixels = match (&prepared.edit, input) {
            (Some(edit), ImageRequestRef::Edit(request)) => Some(
                self.preprocess_reference_edit(edit, request, &work.account())
                    .map_err(empty)?,
            ),
            (None, ImageRequestRef::Generate(_)) => None,
            _ => return Err(empty(anyhow::anyhow!("H3 prepared request kind changed"))),
        };
        let mut pins = Vec::new();
        let mut stages = Vec::new();
        self.execution.execution_host = Some(work.account());
        let result = self.initialize_continuous_streams(
            &prepared.prepared,
            prepared.edit.as_ref(),
            input,
            pixels,
            prepared.job,
            &persistent,
            &work,
            &mut pins,
            &mut stages,
            prepared.main_vision_reuse.as_ref(),
        );
        let result = result.and_then(|streams| {
            let shrink = (|| -> Result<()> {
                let steady = self.steady_host_requirements(&prepared.prepared.denoising.steady)?;
                self.execution
                    .host
                    .shrink_scope_unused(&work, &steady.claims())?;
                Ok(())
            })();
            shrink
                .map(|()| streams)
                .map_err(ImageGenerationFailure::Preflight)
        });
        self.execution.execution_host = None;
        let (image, audio, cache, reference, retained_conditioning) = match result {
            Ok(streams) => streams,
            Err(source) => {
                return Err(ImageGenerationError {
                    failure: ImageGenerationFailure::Pipeline {
                        source: Box::new(source),
                        stages: Vec::new(),
                        weights: pins,
                        host_scopes: vec![persistent, work],
                        host_backing: prepared.backing,
                    },
                    completed_stages: stages,
                    completed_steps: Vec::new(),
                });
            }
        };
        let restoration = retained_conditioning.map(|conditioning| suspension::Restoration {
            conditioning,
            install: prepared.prepared.denoising.install,
            refine: prepared.prepared.denoising.refine,
            work_budget: prepared.work_budget,
        });
        let parts = ContinuousRequestParts {
            geometry: prepared.prepared.conditioning.recipe.geometry,
            image,
            audio,
            cache,
            restoration,
            reference,
            persistent,
            minimum_work: work,
            weights: pins
                .pop()
                .expect("initialized streams retain their exact pin"),
            decoder: prepared.prepared.decoder,
            backing: prepared.backing,
            stages,
            job: prepared.job,
            seed: request.seed,
            width: request.width,
            height: request.height,
        };
        ContinuousRequest::bind(self, parts).map_err(|failure| {
            let parts = failure.parts;
            ImageGenerationError {
                failure: ImageGenerationFailure::Pipeline {
                    source: Box::new(ImageGenerationFailure::Preflight(failure.source)),
                    stages: Vec::new(),
                    weights: vec![parts.weights],
                    host_scopes: vec![parts.persistent, parts.minimum_work],
                    host_backing: parts.backing,
                },
                completed_stages: parts.stages,
                completed_steps: Vec::new(),
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn initialize_continuous_streams(
        &mut self,
        prepared: &super::super::stages::PreparedGeneration<R>,
        edit: Option<&PreparedReferenceEdit<R>>,
        input: ImageRequestRef<'_>,
        pixels: Option<Vec<PH3ImageEditPixels>>,
        job: u64,
        persistent: &HostMemoryScope,
        work: &HostMemoryScope,
        pins: &mut Vec<ImageWeightOwnerPin<R>>,
        stages: &mut Vec<ImageStageReceipt>,
        main_vision_reuse: Option<&super::super::vision::reuse::MainVisionReuseBinding>,
    ) -> std::result::Result<
        (
            ManagedHostTensor,
            ManagedHostTensor,
            PH3ImageRefinedTextCache,
            Vec<ReferenceState>,
            Option<ManagedHostTensor>,
        ),
        ImageGenerationFailure<R>,
    > {
        let request = generation(input);
        if self
            .execution
            .resident_stages
            .has_weights(ImageStage::Denoiser)
        {
            pins.push(
                self.execution
                    .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
                    .map_err(ImageGenerationFailure::Preflight)?,
            );
        }
        let (reference, vision) = match (edit, input, pixels) {
            (Some(edit), ImageRequestRef::Edit(request), Some(pixels)) => {
                let span = self.trace_phase_started(
                    None,
                    Some(&prepared.conditioning.recipe.geometry.identity),
                    crate::image_generation::continuous_trace::ImageTracePhase::ReferenceEncoder,
                );
                let initialized = self.initialize_reference_edit(
                    edit,
                    request,
                    pixels,
                    &persistent.account(),
                    job,
                    &prepared.conditioning.recipe.geometry.identity,
                    stages,
                );
                self.trace_phase_finished(
                    span,
                    if initialized.is_ok() {
                        crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed
                    } else {
                        crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed
                    },
                );
                let (reference, vision) = initialized?;
                (reference, Some(vision))
            }
            (None, ImageRequestRef::Generate(_), None) => (Vec::new(), None),
            _ => {
                return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                    "H3 prepared request kind changed"
                )))
            }
        };
        let conditioning = match vision.as_ref() {
            Some(vision) => self.execute_conditioning_with_images_reuse(
                &prepared.conditioning,
                Some(vision.as_slice()),
                job,
                stages,
                main_vision_reuse,
            )?,
            None => self.execute_conditioning(&prepared.conditioning, job, stages)?,
        };
        drop(vision);
        // A future cold installation has a different native transaction.
        // Retain the exact connector input in this request's actual persistent
        // grant so refinement can be executed again, never relabel a work view.
        let retained_conditioning = if self.execution.sources.assets.has_prompt_expansion() {
            let mut bytes = persistent
                .account()
                .allocate_zeroed(HostMemoryClass::InputPreprocess, conditioning.bytes().len())
                .map_err(|source| ImageGenerationFailure::Preflight(source.into()))?;
            bytes.as_mut().copy_from_slice(conditioning.bytes().bytes());
            Some(
                ManagedHostTensor::new(
                    conditioning.value_id().clone(),
                    conditioning.tensor().clone(),
                    bytes.freeze(),
                )
                .map_err(|source| ImageGenerationFailure::Preflight(source.into()))?,
            )
        } else {
            None
        };
        let noise_trace = self.trace_phase_started(
            None,
            Some(&prepared.conditioning.recipe.geometry.identity),
            crate::image_generation::continuous_trace::ImageTracePhase::Noise,
        );
        let (image, audio) = match &prepared.noise.execution {
            InitialNoiseExecution::NativeCuda(stage) => {
                let state = super::super::noise::managed_state(
                    &work.account(),
                    self.options
                        .initial_noise
                        .state(request.seed)
                        .map_err(ImageGenerationFailure::Preflight)?,
                )
                .map_err(ImageGenerationFailure::Preflight)?;
                let active = self.execution.start_stage(stage.clone(), job, stages)?;
                let Some(actor) = self.execution.serial_actor.as_mut() else {
                    return Err(active.processing(anyhow::anyhow!("H3 serial actor is closed")));
                };
                let (active, outputs) = active.execute_on_actor(actor, job, vec![state])?;
                let samples = (|| -> Result<_> {
                    ensure!(
                        outputs.len() == 1,
                        "H3 joint noise requires exactly one output"
                    );
                    prepared.noise.layout.samples(
                        &persistent.account(),
                        &outputs[0],
                        ProgramValueId::new("value.h3.sample.image")?,
                        ProgramValueId::new("value.h3.sample.audio")?,
                    )
                })();
                let samples = match samples {
                    Ok(samples) => samples,
                    Err(source) => return Err(active.processing(source)),
                };
                drop(outputs);
                self.execution.finish_stage(active, stages)?;
                samples
            }
            InitialNoiseExecution::CpuMt19937 => (|| -> Result<_> {
                prepared.noise.layout.cpu_samples(
                    &persistent.account(),
                    &work.account(),
                    request.seed,
                    ProgramValueId::new("value.h3.sample.image")?,
                    ProgramValueId::new("value.h3.sample.audio")?,
                )
            })()
            .map_err(ImageGenerationFailure::Preflight)?,
        };
        self.trace_phase_finished(
            noise_trace,
            crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed,
        );
        if pins.is_empty() {
            let active = self.execution.start_stage(
                prepared.denoising.install.stage.clone(),
                job,
                stages,
            )?;
            let pin = match self
                .execution
                .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
            {
                Ok(pin) => pin,
                Err(source) => return Err(active.processing(source)),
            };
            pins.push(pin);
            self.execution.finish_stage(active, stages)?;
        }
        let mut caches = self.execute_refinement(
            &prepared.denoising.refine,
            job,
            persistent,
            work.account(),
            &prepared.conditioning.recipe.geometry.identity,
            &conditioning,
            &pins[0],
            stages,
        )?;
        drop(conditioning);
        ensure_single_cache(&caches, &prepared.conditioning.recipe.geometry.identity)
            .map_err(ImageGenerationFailure::Preflight)?;
        Ok((
            image,
            audio,
            caches.pop().unwrap(),
            reference,
            retained_conditioning,
        ))
    }
}

fn ensure_prepared<R: DeviceRuntime>(
    pipeline: &PH3ImagePipeline<R>,
    prepared: &PreparedContinuousRequest<R>,
    request: ImageRequestRef<'_>,
) -> Result<()> {
    ensure!(
        match (&prepared.edit, request) {
            (None, ImageRequestRef::Generate(_)) => true,
            (Some(edit), ImageRequestRef::Edit(request)) =>
                edit.images().count() == request.image_count()
                    && edit
                        .images()
                        .zip(request.images())
                        .all(|(image, encoded)| image.encoded.same_region(encoded)),
            _ => false,
        },
        "H3 preparation must retain the original encoded image allocation and range"
    );
    ensure!(
        !pipeline.is_stopped()
            && pipeline.execution.execution_host.is_none()
            && Arc::ptr_eq(
                &prepared.pipeline,
                &pipeline.execution.service_preparation_identity
            )
            && prepared.identity == request_identity(request)
            && prepared.persistent.is_some()
            && prepared.work.is_some(),
        "H3 initialization requires this admitted request at a completed stage boundary"
    );
    Ok(())
}
fn ensure_single_cache(caches: &[PH3ImageRefinedTextCache], id: &RequestIdentity) -> Result<()> {
    ensure!(
        caches.len() == 1 && caches[0].key().request_identity() == id,
        "H3 initialization returned another request's refinement"
    );
    Ok(())
}
