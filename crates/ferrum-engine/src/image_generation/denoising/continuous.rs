//! Real Full-T2I continuations. Persistent request storage and step work use
//! separate grants from the existing Host parent; no alternate resource ledger.
use super::*;
use crate::image_generation::{
    host_budget::{self, ExecutionHostBudget},
    prepare::{batch_dimensions, PreparedRequest},
};
use sha2::{Digest, Sha256};

pub(crate) struct PreparedContinuousRequest<R: DeviceRuntime> {
    pipeline: Arc<()>,
    identity: [u8; 32],
    prepared: PreparedRequest<R>,
    persistent_claims: [(HostMemoryClass, usize); 2],
    persistent: Option<HostMemoryScope>,
}
pub(crate) struct ContinuousRequest<R: DeviceRuntime> {
    pipeline: Arc<()>,
    state: RequestState,
    text: ManagedHostTensor,
    transformer: QwenImage21TransformerConfig,
    decoder: QwenImage21VaeDecoderConfig,
    decode_stage: PreparedStage<R>,
    latent_shape: QwenImage21LatentShape,
    persistent: HostMemoryScope,
    minimum_work: Option<HostMemoryScope>,
    work_budget: ExecutionHostBudget,
    _backing: Vec<HostBackingPreparation>,
    stages: Vec<ImageStageReceipt>,
    steps: Vec<ImageDenoiseStepReceipt>,
    seed: u64,
    width: u32,
    height: u32,
    last_batch: (usize, usize),
    decoded: bool,
}
impl<R: DeviceRuntime> ContinuousRequest<R> {
    pub(crate) fn is_complete(&self) -> bool {
        self.state.solver.is_complete()
    }
    pub(crate) fn job(&self) -> u64 {
        self.state.id.job
    }
}
/// A compiled/admitted exact membership. It cannot be rebound to other request
/// objects, even when their physical shapes or parent account happen to match.
pub(crate) struct ContinuousBatch<R: DeviceRuntime> {
    pipeline: Arc<()>,
    members: Vec<RequestId>,
    config: QwenImage21TransformerConfig,
    prepared: PreparedStage<R>,
    active: Option<ActiveImageStage<R>>,
    scope: Option<HostMemoryScope>,
    stages: Vec<ImageStageReceipt>,
    job: u64,
    _backing: Vec<HostBackingPreparation>,
}

fn process<R: DeviceRuntime>(source: anyhow::Error) -> ImageGenerationFailure<R> {
    ImageGenerationFailure::Processing {
        source,
        active: None,
    }
}
fn resolved(spec: &ProgramTensorSpec) -> anyhow::Result<ResolvedTensorSpec> {
    Ok(ResolvedTensorSpec::new(
        spec.dimensions.clone(),
        spec.element_type,
        spec.layout.clone(),
    )?)
}
fn copy(host: &HostMemoryAccount, value: &ManagedHostTensor) -> anyhow::Result<ManagedHostTensor> {
    let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, value.bytes().len())?;
    bytes.as_mut().copy_from_slice(value.bytes().bytes());
    Ok(ManagedHostTensor::new(
        value.value_id().clone(),
        value.tensor().clone(),
        bytes.freeze(),
    )?)
}
fn request_identity(r: &ImageGenerationRequest) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(r.prompt.as_bytes());
    hash.update(r.width.to_le_bytes());
    hash.update(r.height.to_le_bytes());
    hash.update(r.seed.to_le_bytes());
    hash.update(r.n.to_le_bytes());
    hash.finalize().into()
}
fn continuation_output_budget<R: DeviceRuntime>(
    host: &HostMemoryAccount,
    p: &mut PreparedRequest<R>,
) -> anyhow::Result<()> {
    // The final decoder readback belongs to the request's persistent Output
    // grant. Initialization/step work only needs TE, noise or velocity, never
    // a duplicate future image reservation.
    p.host_budget.output = [&p.text_stage, &p.noise_stage, &p.denoise_stage]
        .into_iter()
        .map(|stage| host_budget::sum_ports(host, host_budget::ports(stage)?.outputs().iter()))
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("missing continuation output stages")?;
    Ok(())
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(crate) fn prepare_continuous_request(
        &self,
        request: &ImageGenerationRequest,
    ) -> anyhow::Result<PreparedContinuousRequest<R>> {
        let mut prepared =
            self.prepare_batch_with_host_admission(std::slice::from_ref(request), None, false)?;
        continuation_output_budget(&self.execution.host, &mut prepared)?;
        ensure!(
            self.execution.config.decoder_batch_policy == ImageDecoderBatchPolicy::SingleImage
                && prepared.prefix_decode.is_none()
                && prepared.edit.is_none()
                && prepared.transformer.execution_phase()
                    == ferrum_models::qwen_image21::QwenImage21TransformerPhase::Full,
            "step generation requires declared full text-to-image execution"
        );
        let specs = prepared.transformer.input_tensor_specs()?;
        let latent = specs
            .get(&id::<ProgramValueId>(
                QwenImage21TransformerFamily::LATENTS_ID,
            )?)
            .context("missing latent input")?;
        let text = specs
            .get(&id::<ProgramValueId>(
                QwenImage21TransformerFamily::TEXT_ID,
            )?)
            .context("missing text input")?;
        let samples = self
            .execution
            .host
            .backing_bytes(usize::try_from(latent.byte_len()?)?)?
            .checked_mul(2)
            .context("persistent sample slots overflow")?;
        let input = samples
            .checked_add(
                self.execution
                    .host
                    .backing_bytes(usize::try_from(text.byte_len()?)?)?,
            )
            .context("persistent conditioning extent overflows")?;
        let output = self.execution.host.backing_bytes(usize::try_from(
            prepared.decoder.output_tensor()?.minimum_storage_bytes()?,
        )?)?;
        let claims = [
            (HostMemoryClass::InputPreprocess, input),
            (HostMemoryClass::Output, output),
        ];
        self.execution.host.check_scope_capacity(&claims)?;
        Ok(PreparedContinuousRequest {
            pipeline: self.execution.service_preparation_identity.clone(),
            identity: request_identity(request),
            prepared,
            persistent_claims: claims,
            persistent: None,
        })
    }
    /// Pure Host admission. Both the request's future sample/image capacity
    /// and its initialization work are retained before the first submission.
    /// A temporary shortage leaves this exact preparation reusable.
    pub(crate) fn try_admit_continuous_request(
        &self,
        p: &mut PreparedContinuousRequest<R>,
    ) -> anyhow::Result<bool> {
        self.try_admit_continuous_request_with_output_credit(p, 0)
    }
    pub(crate) fn try_admit_continuous_request_with_output_credit(
        &self,
        p: &mut PreparedContinuousRequest<R>,
        retained_output_bytes: usize,
    ) -> anyhow::Result<bool> {
        ensure!(
            Arc::ptr_eq(&p.pipeline, &self.execution.service_preparation_identity),
            "continuation preparation belongs to another pipeline"
        );
        if p.persistent.is_some() && p.prepared.host_scope.is_some() {
            return Ok(true);
        }
        ensure!(
            p.persistent.is_none() && p.prepared.host_scope.is_none(),
            "partial continuation admission"
        );
        // Metadata may have been prepared before another request initialized
        // immutable conversion caches. Recalculate actual additional backing;
        // an old cold budget is not a second copy of those same cache entries.
        self.prepare_execution_host(&mut p.prepared)?;
        continuation_output_budget(&self.execution.host, &mut p.prepared)?;
        let claims = p
            .persistent_claims
            .iter()
            .copied()
            .chain(p.prepared.host_budget.claims())
            .collect::<Vec<_>>();
        // A selected request cannot await its own output gate. Other requests'
        // grants remain temporary pressure because they can finish independently.
        let mut own_peak = claims.clone();
        own_peak.push((HostMemoryClass::Output, retained_output_bytes));
        self.execution.host.check_scope_capacity(&own_peak)?;
        // Avoid repeatedly reserving then refunding only the persistent half
        // of a blocked admission: those self-generated refund events would
        // otherwise prevent an idle consumer from sleeping on real capacity.
        match self.execution.host.check_scope_admission(&claims) {
            Ok(()) => (),
            Err(e) if e.is_temporary_capacity() => return Ok(false),
            Err(e) => return Err(e.into()),
        }
        let persistent = match self.execution.host.reserve_scope(&p.persistent_claims) {
            Ok(scope) => scope,
            Err(e) if e.is_temporary_capacity() => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let work = match self.reserve_execution_host(&p.prepared) {
            Ok(scope) => scope,
            Err(e)
                if e.downcast_ref::<HostMemoryError>()
                    .is_some_and(HostMemoryError::is_temporary_capacity) =>
            {
                return Ok(false)
            }
            Err(e) => return Err(e),
        };
        p.persistent = Some(persistent);
        p.prepared.host_scope = Some(work);
        Ok(true)
    }
    /// All stages finish at their real fence before this request becomes
    /// visible to a denoising cohort. Callers must pause an existing cohort
    /// before entering here, so adaptive residency only evicts idle owners.
    pub(crate) fn start_continuous_request(
        &mut self,
        mut p: PreparedContinuousRequest<R>,
        request: &ImageGenerationRequest,
    ) -> Result<ContinuousRequest<R>, ImageGenerationFailure<R>> {
        if !Arc::ptr_eq(&p.pipeline, &self.execution.service_preparation_identity)
            || p.identity != request_identity(request)
        {
            return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "continuation preparation request/source identity differs"
            )));
        }
        let persistent = p
            .persistent
            .take()
            .context("continuation lacks persistent grant")
            .map_err(process)?;
        let work = p
            .prepared
            .host_scope
            .take()
            .context("continuation lacks initialization grant")
            .map_err(process)?;
        let job = self.execution.next_job;
        self.execution.next_job = self
            .execution
            .next_job
            .checked_add(1)
            .context("image job identity exhausted")
            .map_err(process)?;
        if let Some(candidate) = p.prepared.plan_cache_candidate.take() {
            candidate.commit();
        }
        self.execution.execution_host = Some(work.account());
        let result = self.initialize_continuous_request(p.prepared, request, job, persistent);
        self.execution.execution_host = None;
        match result {
            Ok(mut request) => {
                request.minimum_work = Some(work);
                Ok(request)
            }
            Err(e) => {
                drop(work);
                Err(e)
            }
        }
    }
    fn initialize_continuous_request(
        &mut self,
        p: PreparedRequest<R>,
        request: &ImageGenerationRequest,
        job: u64,
        persistent: HostMemoryScope,
    ) -> Result<ContinuousRequest<R>, ImageGenerationFailure<R>> {
        let mut stages = Vec::new();
        let inputs = p
            .prompts
            .managed_inputs(
                self.execution_host(),
                &self.execution.sources.assets.text,
                &self
                    .execution
                    .sources
                    .assets
                    .sources
                    .text_encoder
                    .tensor(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME)
                    .map_err(|e| process(e.into()))?,
            )
            .map_err(process)?
            .into_parts()
            .1;
        let text = self.start_stage(p.text_stage, job, &mut stages)?;
        let (text, outputs) = self.execute_stage(text, job, Vec::from(inputs))?;
        self.finish_stage(text, &mut stages)?;
        let hidden = only_output(outputs, Qwen3VLTextEncoderFamily::OUTPUT_ID).map_err(process)?;
        let features = p
            .prompts
            .crop_hidden(
                self.execution_host(),
                &self.execution.sources.assets.text,
                &hidden,
            )
            .map_err(process)?
            .into_tensor();
        drop(hidden);
        let noise = self.start_stage(p.noise_stage, job, &mut stages)?;
        let state = match noise_state(self.execution_host(), request.seed) {
            Ok(state) => state,
            Err(e) => return Err(noise.processing(e)),
        };
        let (noise, outputs) = self.execute_stage(noise, job, vec![state])?;
        self.finish_stage(noise, &mut stages)?;
        let output = only_output(outputs, NormalNoiseFamily::OUTPUT_ID).map_err(process)?;
        let latents = QwenImage21ManagedLatents::from_output(
            self.execution_host(),
            &output,
            p.latent_shape,
            QwenImage21LatentLayout::Nchw,
            QwenImage21LatentDomain::Normalized,
        )
        .map_err(process)?;
        let sample = latents
            .repack(
                self.execution_host(),
                QwenImage21LatentLayout::Packed,
                id(QwenImage21TransformerFamily::LATENTS_ID).map_err(process)?,
            )
            .map_err(process)?
            .into_input_tensor()
            .map_err(process)?;
        drop((output, latents));
        let host = persistent.account();
        let sample = copy(&host, &sample).map_err(process)?;
        let text = copy(&host, &features).map_err(process)?;
        let solver = ManagedBf16Euler::new(
            self.execution.sources.assets.schedule.solver().clone(),
            host,
            sample.tensor().clone(),
            self.profiles.euler,
        )
        .map_err(|e| process(e.into()))?;
        Ok(ContinuousRequest {
            pipeline: self.execution.service_preparation_identity.clone(),
            state: RequestState {
                id: RequestId {
                    job,
                    caller_index: 0,
                },
                sample,
                solver,
            },
            text,
            transformer: p.transformer,
            decoder: p.decoder,
            decode_stage: p.decode_stage,
            latent_shape: p.latent_shape,
            persistent,
            minimum_work: None,
            work_budget: p.host_budget,
            _backing: p.host_backing,
            stages,
            steps: Vec::new(),
            seed: request.seed,
            width: request.width,
            height: request.height,
            last_batch: (1, 0),
            decoded: false,
        })
    }
    /// Fresh configuration, Host grant and actual device admission precede
    /// publication. The caller keeps the old cohort's work grant until this
    /// succeeds; a capacity failure therefore cannot strand its active rows.
    pub(crate) fn try_prepare_continuous_batch(
        &mut self,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> Result<Option<ContinuousBatch<R>>, ImageGenerationFailure<R>> {
        self.prepare_continuous_replacement(requests, None)
    }
    pub(crate) fn try_replace_continuous_batch(
        &mut self,
        previous: &mut ContinuousBatch<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> Result<Option<ContinuousBatch<R>>, ImageGenerationFailure<R>> {
        self.prepare_continuous_replacement(requests, Some(previous))
    }
    fn prepare_continuous_replacement(
        &mut self,
        requests: &mut [&mut ContinuousRequest<R>],
        mut previous: Option<&mut ContinuousBatch<R>>,
    ) -> Result<Option<ContinuousBatch<R>>, ImageGenerationFailure<R>> {
        if previous.as_ref().is_some_and(|old| {
            old.active.is_some()
                || !Arc::ptr_eq(&old.pipeline, &self.execution.service_preparation_identity)
        }) {
            return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "pause the original cohort at its fence before exchanging step authorization"
            )));
        }
        let prepare = || -> anyhow::Result<_> {
            let candidate_cache = self.execution.plan_cache.begin_candidate()?;
            ensure!(
                !requests.is_empty() && requests.len() <= self.execution.config.maximum_batch_size,
                "invalid continuation cohort size"
            );
            let first = &*requests[0];
            let layout = first.transformer.layout();
            ensure!(
                requests.iter().all(|r| Arc::ptr_eq(
                    &r.pipeline,
                    &self.execution.service_preparation_identity
                ) && !r.is_complete()
                    && r.transformer.layout().latent_height() == layout.latent_height()
                    && r.transformer.layout().latent_width() == layout.latent_width()),
                "continuation cohort identity/geometry differs"
            );
            let lengths = requests
                .iter()
                .map(|r| r.transformer.layout().valid_text_lengths()[0])
                .collect::<Vec<_>>();
            let tokens = requests
                .iter()
                .map(|r| r.transformer.layout().text_tokens())
                .max()
                .unwrap();
            let config = QwenImage21TransformerConfig::new(
                self.execution.sources.assets.transformer.clone(),
                QwenImage21T2iLayout::new(
                    tokens,
                    layout.latent_height(),
                    layout.latent_width(),
                    lengths,
                )?,
            )?
            .with_weight_encodings(self.execution.sources.assets.transformer_encodings.clone())?;
            let stage = self.prepare_denoiser(&config, requests.len())?;
            self.validate_capacity_with_prefix(&[&stage], None)?;
            let ports = host_budget::ports(&stage)?;
            let mut budget = ExecutionHostBudget {
                input: host_budget::sum_ports(&self.execution.host, ports.inputs().iter())?,
                output: host_budget::sum_ports(&self.execution.host, ports.outputs().iter())?,
                ..Default::default()
            };
            // Initialization/decode are serialized at boundaries. These
            // immutable B1 stage budgets bound their existing staging/weight
            // conversions; cohort tensor inputs/outputs are calculated fresh.
            for r in requests.iter() {
                budget.input = budget.input.max(r.work_budget.input);
            }
            let mut backing = Vec::new();
            // Exact currently additional source backing for DiT and a B1
            // decoder. TE/noise have already fenced; future image Output is
            // separately granted per request and is never reserved here again.
            for required in [&stage, &first.decode_stage] {
                let (bytes, retention) = self.stage_host_backing(required)?;
                budget.weights = budget
                    .weights
                    .checked_add(bytes)
                    .context("step source backing overflows")?;
                backing.push(retention);
                if self.execution.runtime.requires_managed_static_staging() {
                    budget.staging = budget.staging.max(host_budget::staging_bytes(
                        &self.execution.host,
                        required,
                        self.execution.config.static_initialization,
                    )?);
                }
            }
            Ok((config, stage, budget, backing, candidate_cache.defer()?))
        };
        let (config, prepared, budget, backing, candidate_cache) =
            prepare().map_err(ImageGenerationFailure::Preflight)?;
        let exact_singleton = requests.len() == 1
            && requests[0].transformer == config
            && budget.input <= requests[0].work_budget.input
            && budget.output <= requests[0].work_budget.output
            && budget.staging <= requests[0].work_budget.staging
            && budget.weights <= requests[0].work_budget.weights
            && requests[0].minimum_work.is_some();
        let from_previous = previous.is_some();
        let from_initial = !from_previous && requests[0].minimum_work.is_some();
        let scope = if from_previous || from_initial && !exact_singleton {
            let old = if let Some(previous) = previous.as_deref_mut() {
                previous.scope.take()
            } else {
                requests[0].minimum_work.take()
            }
            .context("step work authority was already transferred")
            .map_err(process)?;
            match self.execution.host.exchange_scope(old, &budget.claims()) {
                Ok(scope) => scope,
                Err(failure) => {
                    let (error, old) = failure.into_parts();
                    if let Some(previous) = previous.as_deref_mut() {
                        previous.scope = Some(old);
                    } else {
                        requests[0].minimum_work = Some(old);
                    }
                    if error.is_temporary_capacity() {
                        return Ok(None);
                    }
                    return Err(ImageGenerationFailure::Preflight(error.into()));
                }
            }
        } else if exact_singleton {
            requests[0].minimum_work.take().unwrap()
        } else {
            match self.execution.host.reserve_scope(&budget.claims()) {
                Ok(s) => s,
                Err(e) if e.is_temporary_capacity() => return Ok(None),
                Err(e) => return Err(ImageGenerationFailure::Preflight(e.into())),
            }
        };
        let job = self.execution.next_job;
        self.execution.next_job = self
            .execution
            .next_job
            .checked_add(1)
            .context("image job identity exhausted")
            .map_err(process)?;
        self.execution.execution_host = Some(scope.account());
        let mut stages = Vec::new();
        let active = self.start_stage(prepared.clone(), job, &mut stages);
        self.execution.execution_host = None;
        let active = match active {
            Ok(active) => active,
            Err(e) => {
                if let Some(previous) = previous.as_deref_mut() {
                    previous.scope = Some(scope);
                } else if from_initial {
                    requests[0].minimum_work = Some(scope);
                }
                return Err(e);
            }
        };
        candidate_cache.commit();
        // No member loses its first-forward credit until the replacement has
        // actually initialized/admitted on the device and owns its work grant.
        for request in requests.iter_mut() {
            request.minimum_work.take();
        }
        Ok(Some(ContinuousBatch {
            pipeline: self.execution.service_preparation_identity.clone(),
            members: requests.iter().map(|r| r.state.id).collect(),
            config,
            prepared,
            active: Some(active),
            scope: Some(scope),
            stages,
            job,
            _backing: backing,
        }))
    }
}

#[cfg(test)]
mod tests;

// A deterministic preconditioned request isolates the service's real Stored
// DiT and VAE consumers from unsupported Reference Philox/TE initialization.
#[cfg(test)]
pub(in crate::image_generation) fn fixture_request(
    pipeline: &mut QwenImage21Pipeline<ferrum_kernels::backend::reference::ReferenceDeviceRuntime>,
    seed: u64,
) -> ContinuousRequest<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    let mut request = tests::request(pipeline, (seed % 2) as usize);
    request.seed = seed;
    request
}

#[cfg(test)]
impl ContinuousRequest<ferrum_kernels::backend::reference::ReferenceDeviceRuntime> {
    pub(in crate::image_generation) fn fixture_snapshot(&self) -> (u64, usize, Vec<u8>) {
        (
            self.seed,
            self.state.solver.step_index(),
            self.state.sample.bytes().bytes().to_vec(),
        )
    }
}

impl<R: DeviceRuntime> ContinuousBatch<R> {
    fn validate(&self, requests: &[&mut ContinuousRequest<R>]) -> anyhow::Result<()> {
        ensure!(
            requests.iter().map(|r| r.state.id).collect::<Vec<_>>() == self.members,
            "step preparation membership/order changed; prepare and admit a replacement"
        );
        Ok(())
    }
    /// Return the real execution owner to idle residency at a completed fence.
    /// Its original work grant stays held until a replacement has admitted.
    pub(crate) fn pause(
        &mut self,
        pipeline: &mut QwenImage21Pipeline<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> Result<(), ImageGenerationFailure<R>> {
        if !Arc::ptr_eq(
            &self.pipeline,
            &pipeline.execution.service_preparation_identity,
        ) {
            return Err(process(anyhow::anyhow!("foreign denoising cohort")));
        }
        self.validate(requests).map_err(process)?;
        if let Some(active) = self.active.take() {
            pipeline.finish_stage(active, &mut self.stages)?;
        }
        if !self.stages.is_empty() {
            for request in requests {
                request.stages.extend(self.stages.iter().cloned());
            }
            self.stages.clear();
        }
        Ok(())
    }
    pub(crate) fn advance(
        &mut self,
        pipeline: &mut QwenImage21Pipeline<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> Result<(), ImageGenerationFailure<R>> {
        if !Arc::ptr_eq(
            &self.pipeline,
            &pipeline.execution.service_preparation_identity,
        ) {
            return Err(process(anyhow::anyhow!("foreign denoising cohort")));
        }
        self.validate(requests).map_err(process)?;
        ensure_active(requests).map_err(process)?;
        pipeline.execution.execution_host = Some(
            self.scope
                .as_ref()
                .context("step work authority was transferred")
                .map_err(process)?
                .account(),
        );
        let result = self.advance_inner(pipeline, requests);
        pipeline.execution.execution_host = None;
        result
    }
    fn advance_inner(
        &mut self,
        pipeline: &mut QwenImage21Pipeline<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> Result<(), ImageGenerationFailure<R>> {
        let host = self
            .scope
            .as_ref()
            .context("step work authority was transferred")
            .map_err(process)?
            .account();
        let active = match self.active.take() {
            Some(active) => active,
            None => pipeline.start_stage(self.prepared.clone(), self.job, &mut self.stages)?,
        };
        let inputs = (|| -> anyhow::Result<_> {
            let specs = self.config.input_tensor_specs()?;
            let mut gather = |name: &str| -> anyhow::Result<ManagedHostTensor> {
                let spec = resolved(
                    specs
                        .get(&id::<ProgramValueId>(name)?)
                        .context("missing step input")?,
                )?;
                let mut allocation = host.allocate_zeroed(
                    HostMemoryClass::InputPreprocess,
                    usize::try_from(spec.minimum_storage_bytes()?)?,
                )?;
                let stride = allocation.as_mut().len() / requests.len();
                for (dest, r) in allocation
                    .as_mut()
                    .chunks_exact_mut(stride)
                    .zip(requests.iter())
                {
                    let input = if name == QwenImage21TransformerFamily::LATENTS_ID {
                        &r.state.sample
                    } else {
                        &r.text
                    };
                    ensure!(
                        host.owns(input.bytes())
                            && input.bytes().len() <= dest.len()
                            && input.value_id().as_str() == name
                            && input.tensor().element_type() == spec.element_type()
                            && input.tensor().layout() == &ResolvedTensorLayout::Contiguous
                            && input.tensor().dimensions().len() == 2
                            && input.tensor().dimensions()[1] == spec.dimensions()[1],
                        "request input differs from its gathered cohort port"
                    );
                    if name == QwenImage21TransformerFamily::LATENTS_ID {
                        ensure!(
                            input.bytes().len() == dest.len(),
                            "latent request extent differs"
                        );
                    }
                    dest[..input.bytes().len()].copy_from_slice(input.bytes().bytes());
                }
                Ok(ManagedHostTensor::new(
                    id(name)?,
                    spec,
                    allocation.freeze(),
                )?)
            };
            let times = requests
                .iter()
                .map(|r| {
                    r.state
                        .solver
                        .current_step()
                        .map(|s| s.timestep().as_f32())
                        .context("complete request in active step")
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(vec![
                gather(QwenImage21TransformerFamily::LATENTS_ID)?,
                gather(QwenImage21TransformerFamily::TEXT_ID)?,
                self.config
                    .managed_time_features_per_sample(&host, &times)?,
                self.config.managed_rotary_frequencies(&host)?,
            ])
        })();
        let inputs = match inputs {
            Ok(inputs) => inputs,
            Err(e) => return Err(active.processing(e)),
        };
        let (active, outputs) = pipeline.execute_stage(active, self.job, inputs)?;
        let successors = (|| -> anyhow::Result<_> {
            let velocity = only_output(outputs, QwenImage21TransformerFamily::VELOCITY_ID)?;
            let expected = self
                .config
                .input_tensor_specs()?
                .remove(&id::<ProgramValueId>(
                    QwenImage21TransformerFamily::LATENTS_ID,
                )?)
                .context("missing latent tensor")?;
            ensure!(
                velocity.tensor() == &resolved(&expected)?,
                "velocity differs from exact current membership"
            );
            let stride = requests[0].state.sample.bytes().len();
            // Prepare every request's successor using its own persistent
            // credit before publishing any cursor/sample mutation.
            let pending = requests
                .iter_mut()
                .enumerate()
                .map(|(index, r)| {
                    let row = velocity
                        .bytes()
                        .slice(index * stride..(index + 1) * stride)?;
                    Ok(r.state.solver.prepare_advance(
                        &r.state.sample,
                        r.state.sample.tensor(),
                        &row,
                    )?)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(pending.into_iter().map(|p| p.commit()).collect::<Vec<_>>())
        })();
        let successors = match successors {
            Ok(s) => s,
            Err(e) => return Err(active.processing(e)),
        };
        for (index, (r, (step, sample))) in requests.iter_mut().zip(successors).enumerate() {
            r.state.sample = sample;
            r.last_batch = (self.members.len(), index);
            r.steps.push(ImageDenoiseStepReceipt {
                index: step.index(),
                sigma: step.sigma(),
                next_sigma: step.next_sigma(),
                scheduler_timestep: step.timestep().as_f32(),
            });
        }
        self.active = Some(active);
        Ok(())
    }
}
fn ensure_active<R: DeviceRuntime>(requests: &[&mut ContinuousRequest<R>]) -> anyhow::Result<()> {
    ensure!(
        requests.iter().all(|r| !r.is_complete()),
        "complete request cannot consume another forward"
    );
    Ok(())
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    /// Decode into this request's already-reserved future Output grant. A
    /// delivered image cannot consume the remaining cohort's forward credit.
    /// The batch must be paused before switching model stages/residency.
    pub(crate) fn finish_continuous_request(
        &mut self,
        mut request: ContinuousRequest<R>,
        work: &ContinuousBatch<R>,
    ) -> Result<ImageGenerationOutput, ImageGenerationFailure<R>> {
        self.finish_continuous_request_borrowed(&mut request, work)
    }
    pub(crate) fn finish_continuous_request_borrowed(
        &mut self,
        request: &mut ContinuousRequest<R>,
        work: &ContinuousBatch<R>,
    ) -> Result<ImageGenerationOutput, ImageGenerationFailure<R>> {
        if !request.is_complete()
            || !Arc::ptr_eq(
                &request.pipeline,
                &self.execution.service_preparation_identity,
            )
            || !Arc::ptr_eq(&work.pipeline, &self.execution.service_preparation_identity)
            || request.decoded
        {
            return Err(process(anyhow::anyhow!(
                "incomplete or foreign continuation"
            )));
        }
        if work.active.is_some() {
            return Err(process(anyhow::anyhow!(
                "pause denoising at its fence before decoding"
            )));
        }
        self.execution.execution_host = Some(
            work.scope
                .as_ref()
                .context("decode work authority was transferred")
                .map_err(process)?
                .account(),
        );
        let result = (|| {
            let normalized = QwenImage21ManagedLatents::new(
                self.execution_host(),
                request.state.sample.clone(),
                request.latent_shape,
                QwenImage21LatentLayout::Packed,
                QwenImage21LatentDomain::Normalized,
            )
            .map_err(process)?;
            let input = normalized
                .denormalize_for_vae(
                    self.execution_host(),
                    &self.execution.sources.assets.statistics,
                )
                .map_err(process)?
                .into_input_tensor()
                .map_err(process)?;
            let active = self.start_stage(
                request.decode_stage.clone(),
                request.state.id.job,
                &mut request.stages,
            )?;
            let mut active = active;
            if let Err(e) = active
                .session
                .set_output_host_account(request.persistent.account())
            {
                return Err(active.processing(e.into()));
            }
            let Some(actor) = self.execution.serial_actor.as_mut() else {
                return Err(active.processing(anyhow::anyhow!("image serial actor is closed")));
            };
            let (active, outputs) =
                active.execute_on_actor(actor, request.state.id.job, vec![input])?;
            let rgba = match only_output(outputs, QwenImage21VaeDecoderFamily::OUTPUT_ID) {
                Ok(output) => output,
                Err(e) => return Err(active.processing(e)),
            };
            let checked = (|| -> anyhow::Result<OwnedImagePixels> {
                ensure!(
                    rgba.tensor() == &request.decoder.output_tensor()?,
                    "decoder output differs from request geometry"
                );
                OwnedImagePixels::new(
                    &self.execution.host,
                    rgba.tensor().clone(),
                    rgba.bytes().clone(),
                    ImagePixelEncoding::Bf16SignedRgbaStaged,
                )
            })();
            let rgba = match checked {
                Ok(pixels) => pixels,
                Err(e) => return Err(active.processing(e)),
            };
            self.finish_stage(active, &mut request.stages)?;
            request.decoded = true;
            Ok(ImageGenerationOutput {
                rgba,
                receipt: ImageGenerationReceipt {
                    job: request.state.id.job,
                    batch_size: request.last_batch.0,
                    batch_index: request.last_batch.1,
                    decoder_batch_policy: self.execution.config.decoder_batch_policy,
                    residency: self.execution.config.residency,
                    width: request.width,
                    height: request.height,
                    seed: request.seed,
                    stages: request.stages.clone(),
                    steps: request.steps.clone(),
                    runtime_fingerprint: self
                        .execution
                        .runtime
                        .descriptor()
                        .runtime_implementation_fingerprint
                        .clone(),
                    native_memory: self.execution.runtime.native_memory_snapshot(),
                    assets: self.execution.sources.assets.receipt().clone(),
                    euler_profile: self.profiles.euler.id(),
                    noise_layout_profile: None,
                },
            })
        })();
        self.execution.execution_host = None;
        result
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    /// Synchronous run consumes the same persistent requests and real mixed-time
    /// step executor as serving. The caller supplies all members up front.
    pub(crate) fn generate_continuous_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageGenerationError<R>> {
        let preflight = |e| ImageGenerationError {
            failure: ImageGenerationFailure::Preflight(e),
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        };
        if self.is_stopped() {
            return Err(ImageGenerationError {
                failure: ImageGenerationFailure::Unavailable,
                completed_stages: vec![],
                completed_steps: vec![],
            });
        }
        batch_dimensions(
            &requests,
            &self.execution.config,
            self.execution
                .sources
                .assets
                .spatial_scale()
                .map_err(preflight)?,
        )
        .map_err(preflight)?;
        let prepared = requests
            .iter()
            .map(|r| self.prepare_continuous_request(r))
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(preflight)?;
        let mut states = Vec::with_capacity(requests.len());
        let mut batch: Option<ContinuousBatch<R>> = None;
        let result = (|| -> Result<Vec<ImageGenerationOutput>, ImageGenerationFailure<R>> {
            for (mut prepared, request) in prepared.into_iter().zip(&requests) {
                if let Some(old) = batch.as_mut() {
                    old.pause(self, &mut states.iter_mut().collect::<Vec<_>>())?;
                }
                if !self
                    .try_admit_continuous_request(&mut prepared)
                    .map_err(ImageGenerationFailure::Preflight)?
                {
                    return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                        "initial continuation cannot obtain persistent and initialization capacity"
                    )));
                }
                states.push(self.start_continuous_request(prepared, request)?);
                let candidate = if let Some(previous) = batch.as_mut() {
                    self.try_replace_continuous_batch(
                        previous,
                        &mut states.iter_mut().collect::<Vec<_>>(),
                    )
                } else {
                    self.try_prepare_continuous_batch(&mut states.iter_mut().collect::<Vec<_>>())
                };
                let next = candidate?
                    .context("initial cohort cannot obtain its independent step work grant")
                    .map_err(ImageGenerationFailure::Preflight)?;
                // The previous owner is paused and keeps its grant until the
                // replacement's fresh compile + Host + device admission succeeds.
                batch = Some(next);
            }
            let mut batch = batch
                .take()
                .context("missing continuation cohort")
                .map_err(process)?;
            while states.iter().any(|r| !r.is_complete()) {
                batch.advance(self, &mut states.iter_mut().collect::<Vec<_>>())?;
            }
            batch.pause(self, &mut states.iter_mut().collect::<Vec<_>>())?;
            states
                .drain(..)
                .map(|r| self.finish_continuous_request(r, &batch))
                .collect()
        })();
        match result {
            Ok(outputs) => Ok(outputs),
            Err(failure) => {
                self.execution.stopped = !matches!(
                    failure,
                    ImageGenerationFailure::Preflight(_)
                        | ImageGenerationFailure::AllocationRejected { .. }
                        | ImageGenerationFailure::CapacityRejected { .. }
                        | ImageGenerationFailure::InitializationCapacityRejected { .. }
                );
                let stages = states
                    .iter()
                    .flat_map(|r| r.stages.iter().cloned())
                    .collect();
                let steps = states.first().map_or_else(Vec::new, |r| r.steps.clone());
                Err(ImageGenerationError {
                    failure,
                    completed_stages: stages,
                    completed_steps: steps,
                })
            }
        }
    }
}
