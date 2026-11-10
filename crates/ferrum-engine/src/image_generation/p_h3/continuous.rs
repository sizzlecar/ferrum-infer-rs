//! Request-persistent H3 streams and one actual packed DiT continuation.
//! Membership is sealed only after fresh compilation and real admission.
//! Every request keeps its own sampling Host authority across cohort changes.
use super::{
    editing::ReferenceState,
    request::{BatchGeometry, RequestGeometry},
    stages::{CompiledDecoder, CompiledSteady},
    PH3ImagePipeline,
};
use crate::image_generation::{
    host_budget::{self, ExecutionHostBudget},
    ActiveImageStage, ImageDenoiseStepReceipt, ImageGenerationFailure, ImageStageReceipt,
    ImageWeightOwnerKey, ImageWeightOwnerPin,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::p_h3_image::{
    PH3ImageDenoiserSample, PH3ImageFinalLayerFamily, PH3ImageManagedSampling,
    PH3ImageRefinedTextCache, PH3ImageVelocityPrecision,
};
use ferrum_runtime::*;
use std::sync::Arc;
mod budget;
mod continuation;
mod expansion;
pub(super) use expansion::downstream::PreparationSlot;
mod finish;
mod initialize;
mod native_completion;
#[cfg(test)]
pub(in crate::image_generation) use native_completion::clean_decoder_admission;
mod run;
mod service;
mod step;
mod suspension;
#[cfg(all(test, feature = "cuda"))]
mod tests;
mod trace;
pub(in crate::image_generation) use initialize::PreparedContinuousRequest;

/// Initialization has fenced before these owners are published. The scope
/// contains both image/audio successor slots plus this request's real cache;
/// minimum_work keeps at least its first actual forward admitted on Host.
struct ContinuousRequestParts<R: DeviceRuntime> {
    pub geometry: RequestGeometry,
    pub image: ManagedHostTensor,
    pub audio: ManagedHostTensor,
    pub cache: PH3ImageRefinedTextCache,
    pub restoration: Option<suspension::Restoration<R>>,
    pub reference: Vec<ReferenceState>,
    pub persistent: HostMemoryScope,
    pub minimum_work: HostMemoryScope,
    pub weights: ImageWeightOwnerPin<R>,
    pub decoder: CompiledDecoder<R>,
    pub backing: Vec<HostBackingPreparation>,
    pub stages: Vec<ImageStageReceipt>,
    pub job: u64,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
}

/// A failed bind returns every original owner to its initialization caller.
struct ContinuousRequestBindFailure<R: DeviceRuntime> {
    pub source: anyhow::Error,
    pub parts: ContinuousRequestParts<R>,
}

pub(in crate::image_generation) struct ContinuousRequest<R: DeviceRuntime> {
    pipeline: Arc<()>,
    geometry: RequestGeometry,
    image: ManagedHostTensor,
    audio: Option<ManagedHostTensor>,
    cache: Option<PH3ImageRefinedTextCache>,
    restoration: Option<suspension::Restoration<R>>,
    reference: Vec<ReferenceState>,
    sampling: Option<PH3ImageManagedSampling>,
    persistent: HostMemoryScope,
    minimum_work: Option<HostMemoryScope>,
    weights: Option<ImageWeightOwnerPin<R>>,
    decoder: CompiledDecoder<R>,
    backing: Vec<HostBackingPreparation>,
    stages: Vec<ImageStageReceipt>,
    steps: Vec<ImageDenoiseStepReceipt>,
    job: u64,
    seed: u64,
    width: u32,
    height: u32,
    last_batch: (usize, usize),
    decoded: bool,
}
impl<R: DeviceRuntime> ContinuousRequest<R> {
    fn bind(
        pipeline: &PH3ImagePipeline<R>,
        parts: ContinuousRequestParts<R>,
    ) -> std::result::Result<Self, ContinuousRequestBindFailure<R>> {
        let prepare = || -> Result<_> {
            let host = pipeline.execution.host_account();
            ensure!(
                host.same_parent(&parts.persistent.account())
                    && host.same_parent(&parts.minimum_work.account())
                    && parts.weights.key() == ImageWeightOwnerKey::Denoiser
                    && parts.cache.key().request_identity() == &parts.geometry.identity
                    && parts
                        .cache
                        .key()
                        .matches_snapshot(parts.weights.weights().evidence()),
                "H3 continuation scope, snapshot role or cache identity differs"
            );
            ensure!(
                parts.reference.len() == parts.geometry.references.len(),
                "H3 reference geometry and actual initialized image counts differ"
            );
            for (reference, extent) in parts.reference.iter().zip(&parts.geometry.references) {
                reference.validate(
                    *extent,
                    pipeline.execution.sources.assets.denoiser.transformer(),
                    host,
                    pipeline.options.edit_noise.augmentation(),
                )?;
            }
            for (sample, expected) in [
                (&parts.image, &parts.geometry.image_sample),
                (&parts.audio, &parts.geometry.audio_sample),
            ] {
                ensure!(
                    sample.tensor() == expected
                        && host.owns(sample.bytes())
                        && sample.bytes().class() == HostMemoryClass::InputPreprocess
                        && sample.bytes().len()
                            == usize::try_from(expected.minimum_storage_bytes()?)?,
                    "H3 continuation sample shape, owner or extent differs"
                );
            }
            ensure!(
                parts.image.value_id() != parts.audio.value_id(),
                "H3 continuation image/audio identities must differ"
            );
            PH3ImageManagedSampling::new_for_request(
                &pipeline.execution.sources.assets.schedule,
                parts.persistent.account(),
                parts.geometry.image_sample.clone(),
                parts.geometry.audio_sample.clone(),
                PH3ImageVelocityPrecision::Bf16,
                parts.geometry.identity.clone(),
            )
        };
        let sampling = match prepare() {
            Ok(sampling) => sampling,
            Err(source) => return Err(ContinuousRequestBindFailure { source, parts }),
        };
        let request = Self {
            pipeline: pipeline.execution.service_preparation_identity.clone(),
            geometry: parts.geometry,
            image: parts.image,
            audio: Some(parts.audio),
            cache: Some(parts.cache),
            restoration: parts.restoration,
            reference: parts.reference,
            sampling: Some(sampling),
            persistent: parts.persistent,
            minimum_work: Some(parts.minimum_work),
            weights: Some(parts.weights),
            decoder: parts.decoder,
            backing: parts.backing,
            stages: parts.stages,
            steps: Vec::new(),
            job: parts.job,
            seed: parts.seed,
            width: parts.width,
            height: parts.height,
            last_batch: (1, 0),
            decoded: false,
        };
        trace::initialized(pipeline, &request);
        Ok(request)
    }
    pub fn is_complete(&self) -> bool {
        self.sampling
            .as_ref()
            .is_none_or(PH3ImageManagedSampling::is_complete)
    }
    pub fn step_index(&self) -> usize {
        self.sampling
            .as_ref()
            .map_or(self.steps.len(), PH3ImageManagedSampling::step_index)
    }
    pub fn image(&self) -> &ManagedHostTensor {
        &self.image
    }
    pub fn persistent_scope(&self) -> &HostMemoryScope {
        &self.persistent
    }
    fn weights(&self) -> Result<&ImageWeightOwnerPin<R>> {
        self.weights
            .as_ref()
            .context("completed H3 request no longer pins denoiser weights")
    }
}

/// Fixed physical membership, including row/time maps and the real active
/// session. No equal-shaped replacement may reuse this sealed preparation.
pub(in crate::image_generation) struct ContinuousBatch<R: DeviceRuntime> {
    pipeline: Arc<()>,
    members: Vec<RequestIdentity>,
    prepared: CompiledSteady<R>,
    active: Option<ActiveImageStage<R>>,
    work: Option<HostMemoryScope>,
    work_budget: ExecutionHostBudget,
    phase_backing: std::collections::BTreeMap<String, HostBackingPreparation>,
    stages: Vec<ImageStageReceipt>,
    job: u64,
    suspended: bool,
    suspended_floor: Option<ExecutionHostBudget>,
    pending_membership:
        Option<continuation::PendingMembership<continuation::PreparedContinuation<R>>>,
}
fn process<R: DeviceRuntime>(source: anyhow::Error) -> ImageGenerationFailure<R> {
    ImageGenerationFailure::Processing {
        source,
        active: None,
    }
}

impl<R: DeviceRuntime> ContinuousBatch<R> {
    fn validate(
        &self,
        pipeline: &PH3ImagePipeline<R>,
        requests: &[&mut ContinuousRequest<R>],
        advancing: bool,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(
                &self.pipeline,
                &pipeline.execution.service_preparation_identity,
            ) && !requests.is_empty()
                && requests.len() == self.members.len()
                && requests.iter().zip(&self.members).all(|(r, id)| {
                    Arc::ptr_eq(&r.pipeline, &self.pipeline)
                        && &r.geometry.identity == id
                        && (!advancing || !r.is_complete())
                }),
            "H3 continuation membership, order, pipeline or live progress differs"
        );
        if advancing {
            ensure!(
                !self.suspended,
                "suspended H3 cohort cannot execute a denoising step"
            );
            let snapshot = requests[0].weights()?.weights().evidence();
            for request in requests {
                ensure!(
                    request.weights()?.weights().evidence() == snapshot,
                    "H3 cohort requests refer to different native weight snapshots"
                );
            }
        }
        Ok(())
    }

    /// Finish the original session at its completed fence before TE, decoder
    /// or membership replacement can change resident native owners.
    pub fn pause(
        &mut self,
        pipeline: &mut PH3ImagePipeline<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> std::result::Result<(), ImageGenerationFailure<R>> {
        self.validate(pipeline, requests, false).map_err(process)?;
        if let Some(active) = self.active.take() {
            pipeline.execution.finish_stage(active, &mut self.stages)?;
        }
        for request in requests {
            request.stages.extend(self.stages.iter().cloned());
            // Once all required forwards have fenced, this lane never needs
            // the DiT again. Release every completed lane together before
            // decoding any one of them; live lanes keep their own exact pins.
            if request.is_complete() {
                request.weights.take();
                request.audio.take();
                request.cache.take();
                request.restoration.take();
                request.reference.clear();
                request.sampling.take();
                // No completed lane can allocate another successor. Retain
                // only its real image charge for the decoder, not the unused
                // second slot or retired audio/refinement-cache entitlement.
                pipeline
                    .execution
                    .host
                    .shrink_scope_unused(
                        &request.persistent,
                        &[(HostMemoryClass::InputPreprocess, 0)],
                    )
                    .map_err(|source| process(source.into()))?;
            }
        }
        self.stages.clear();
        self.shrink_idle_work(pipeline).map_err(process)?;
        Ok(())
    }

    fn shrink_idle_work(&self, pipeline: &PH3ImagePipeline<R>) -> Result<()> {
        let scope = self
            .work
            .as_ref()
            .context("H3 work authority was transferred")?;
        pipeline
            .execution
            .host
            .shrink_scope_unused_to_at_most(scope, &self.work_budget.claims())?;
        Ok(())
    }

    /// One packed DenoiseRefinedText evaluation, with each member's own live
    /// timestep. All image/audio successors allocate from their unchanged
    /// persistent grants before any request cursor or sample is published.
    pub fn advance(
        &mut self,
        pipeline: &mut PH3ImagePipeline<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> std::result::Result<(), ImageGenerationFailure<R>> {
        self.validate(pipeline, requests, true).map_err(process)?;
        if pipeline.execution.execution_host.is_some() {
            return Err(process(anyhow::anyhow!(
                "another H3 stage still owns the shared execution Host view"
            )));
        }
        let host = self
            .work
            .as_ref()
            .context("H3 cohort work authority was transferred")
            .map_err(process)?
            .account();
        pipeline.execution.execution_host = Some(host.clone());
        let result = self.advance_inner(pipeline, requests, &host);
        pipeline.execution.execution_host = None;
        result
    }

    fn advance_inner(
        &mut self,
        pipeline: &mut PH3ImagePipeline<R>,
        requests: &mut [&mut ContinuousRequest<R>],
        host: &HostMemoryAccount,
    ) -> std::result::Result<(), ImageGenerationFailure<R>> {
        let active = match self.active.take() {
            Some(active) => active,
            None => pipeline.execution.start_stage(
                self.prepared.stage.clone(),
                self.job,
                &mut self.stages,
            )?,
        };
        let metadata = (|| -> Result<_> {
            let assets = &pipeline.execution.sources.assets;
            let geometry = BatchGeometry::new(
                assets.denoiser.transformer(),
                requests.iter().map(|r| &r.geometry).collect(),
                pipeline.execution.config.maximum_batch_size,
                assets.limits.denoiser.max_metadata_bytes,
            )?;
            let steps = requests
                .iter()
                .map(|r| {
                    Ok((
                        &r.geometry.identity,
                        r.sampling
                            .as_ref()
                            .context("retired H3 sampler")?
                            .current_step()
                            .context("completed H3 cohort member")?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let times = geometry.times(&steps, pipeline.options.edit_noise.augmentation())?;
            // This is a clone of the real sealed handle, never a new
            // installation, capacity grant or substitute snapshot.
            let weights = requests[0].weights()?.weights().clone();
            Ok((times, geometry.positions, weights))
        })();
        let (times, positions, weights) = match metadata {
            Ok(metadata) => metadata,
            Err(source) => return Err(active.processing(source)),
        };
        self.prepare_survivors(pipeline, requests);
        let diagnostic_identity = super::dit_input_diagnostics::DitInputIdentity {
            job: requests[0].job,
            seed: requests[0].seed,
            width: requests[0].width,
            height: requests[0].height,
        };
        let lanes = requests
            .iter_mut()
            .map(|r| {
                let ContinuousRequest {
                    geometry,
                    image,
                    audio,
                    sampling,
                    cache,
                    reference,
                    ..
                } = &mut **r;
                Ok(step::StepLane {
                    identity: &geometry.identity,
                    image,
                    audio: audio.as_mut().context("retired H3 audio stream")?,
                    sampling: sampling.as_mut().context("retired H3 sampler")?,
                    cache: cache.as_ref().context("retired H3 refinement cache")?,
                    references: reference.iter().map(ReferenceState::view).collect(),
                })
            })
            .collect::<Result<Vec<_>>>();
        let mut lanes = match lanes {
            Ok(lanes) => lanes,
            Err(source) => return Err(active.processing(source)),
        };
        let (active, steps) = step::execute(
            &mut pipeline.execution,
            &self.prepared,
            active,
            self.job,
            host,
            &weights,
            &times,
            &positions,
            &mut lanes,
            pipeline
                .dit_input_diagnostics
                .as_mut()
                .map(|d| (d, diagnostic_identity)),
        )?;
        drop(lanes);
        for (index, (request, step)) in requests.iter_mut().zip(steps).enumerate() {
            request.last_batch = (self.members.len(), index);
            request.steps.push(ImageDenoiseStepReceipt {
                index: step.index(),
                sigma: step.sigma(),
                next_sigma: step.next_sigma(),
                scheduler_timestep: step.timestep(),
            });
        }
        if let Err(source) = self.shrink_idle_work(pipeline) {
            return Err(active.processing(source));
        }
        self.active = Some(active);
        Ok(())
    }
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(in crate::image_generation) fn try_prepare_continuous_batch(
        &mut self,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> std::result::Result<Option<ContinuousBatch<R>>, ImageGenerationFailure<R>> {
        self.prepare_continuous_replacement(requests, None, None)
    }

    pub(in crate::image_generation) fn try_replace_continuous_batch(
        &mut self,
        previous: &mut ContinuousBatch<R>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> std::result::Result<Option<ContinuousBatch<R>>, ImageGenerationFailure<R>> {
        let prepared = previous
            .pending_membership
            .take()
            .map(|pending| {
                pending.finish_for(&requests.iter().map(|r| &r.geometry).collect::<Vec<_>>())
            })
            .transpose()
            .map_err(ImageGenerationFailure::Preflight)?
            .flatten();
        self.prepare_continuous_replacement(requests, Some(previous), prepared)
    }

    fn prepare_continuous_replacement(
        &mut self,
        requests: &mut [&mut ContinuousRequest<R>],
        mut previous: Option<&mut ContinuousBatch<R>>,
        prepared_candidate: Option<continuation::PreparedContinuation<R>>,
    ) -> std::result::Result<Option<ContinuousBatch<R>>, ImageGenerationFailure<R>> {
        if self.dit_input_diagnostics.is_some() && requests.len() != 1 {
            return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "DiT input capture requires B1"
            )));
        }
        if self.execution.execution_host.is_some() || self.execution.is_stopped() {
            return Err(ImageGenerationFailure::Unavailable);
        }
        if previous.as_ref().is_some_and(|old| {
            old.active.is_some()
                || old.suspended
                || !Arc::ptr_eq(&old.pipeline, &self.execution.service_preparation_identity)
        }) {
            return Err(process(anyhow::anyhow!(
                "pause the original H3 cohort before replacing membership"
            )));
        }
        let prepare = || -> Result<_> {
            ensure!(
                !requests.is_empty() && requests.len() <= self.execution.config.maximum_batch_size,
                "H3 cohort must contain a bounded nonempty live membership"
            );
            ensure!(
                requests.iter().all(|r| {
                    Arc::ptr_eq(&r.pipeline, &self.execution.service_preparation_identity)
                        && !r.is_complete()
                }),
                "H3 cohort has foreign, completed or differently installed requests"
            );
            let snapshot = requests[0].weights()?.weights().evidence();
            for request in requests.iter() {
                ensure!(
                    request.weights()?.weights().evidence() == snapshot,
                    "H3 cohort requests refer to different native weight snapshots"
                );
            }
            let continuation::PreparedContinuation {
                prepared,
                candidate_cache,
            } = match prepared_candidate {
                Some(candidate) => candidate,
                None => self
                    .compile_continuation_recipe(requests.iter().map(|r| &r.geometry).collect())?,
            };
            ensure!(
                prepared.recipe.request_ids()
                    == requests
                        .iter()
                        .map(|r| r.geometry.identity.clone())
                        .collect::<Vec<_>>(),
                "fresh H3 recipe differs from candidate request identities"
            );
            self.execution
                .validate_capacity_with_prefix(&[&prepared.stage], None)?;
            let (budget, phase_backing) =
                self.continuous_phase_budget(&prepared, requests, previous.as_deref())?;
            self.execution
                .check_scope_capacity_with_retained_diagnostics(&budget.claims())?;
            let job = self.execution.next_job;
            let next_job = job
                .checked_add(1)
                .context("H3 cohort job identity exhausted")?;
            Ok((
                prepared,
                budget,
                phase_backing,
                job,
                next_job,
                candidate_cache,
            ))
        };
        let (prepared, budget, mut phase_backing, job, next_job, candidate_cache) =
            prepare().map_err(ImageGenerationFailure::Preflight)?;

        // An initialized request or previous cohort already owns enough work
        // credit for progress. Never release that authority to the admission
        // queue while a replacement has not actually become runnable.
        let old = if let Some(old) = previous.as_deref_mut() {
            old.work.take()
        } else {
            requests[0].minimum_work.take()
        }
        .context("H3 continuation lost its minimum forward work authority")
        .map_err(process)?;
        let scope = match self.execution.host.exchange_scope(old, &budget.claims()) {
            Ok(scope) => scope,
            Err(failure) => {
                let (source, old) = failure.into_parts();
                restore_work(&mut previous, requests, old);
                if source.is_temporary_capacity() {
                    return Ok(None);
                }
                return Err(ImageGenerationFailure::Preflight(source.into()));
            }
        };
        self.execution.next_job = next_job;
        self.execution.execution_host = Some(scope.account());
        let mut stages = Vec::new();
        let started = self
            .execution
            .start_stage(prepared.stage.clone(), job, &mut stages);
        self.execution.execution_host = None;
        let active = match started {
            Ok(active) => active,
            Err(failure) => {
                // Exchange preserved at least the old unused credit. The
                // previous exact recipe and all request cursors stay intact,
                // even when native admission closed an idle old shape.
                restore_work(&mut previous, requests, scope);
                if let Some(previous) = previous.as_deref_mut() {
                    previous.stages.extend(stages);
                } else {
                    requests[0].stages.extend(stages);
                }
                return clean_capacity_failure(failure).map(|()| None);
            }
        };
        let attached = (|| -> Result<_> {
            prepared
                .recipe
                .validate_attached_session(&active.session, requests[0].weights()?.weights())
        })();
        if let Err(source) = attached {
            restore_work(&mut previous, requests, scope);
            if let Some(previous) = previous.as_deref_mut() {
                previous.stages.extend(stages);
            } else {
                requests[0].stages.extend(stages);
            }
            return Err(active.processing(source));
        }
        candidate_cache.commit();
        if let Err(source) = self
            .execution
            .host
            .shrink_scope_unused_to_at_most(&scope, &budget.claims())
        {
            restore_work(&mut previous, requests, scope);
            if let Some(previous) = previous.as_deref_mut() {
                previous.stages.extend(stages);
            } else {
                requests[0].stages.extend(stages);
            }
            return Err(active.processing(source.into()));
        }
        for request in requests.iter_mut() {
            request.minimum_work.take();
        }
        if let Some(previous) = previous.as_deref_mut() {
            for (key, retained) in std::mem::take(&mut previous.phase_backing) {
                phase_backing.entry(key).or_insert(retained);
            }
        }
        let batch = ContinuousBatch {
            pipeline: self.execution.service_preparation_identity.clone(),
            members: requests
                .iter()
                .map(|r| r.geometry.identity.clone())
                .collect(),
            prepared,
            active: Some(active),
            work: Some(scope),
            work_budget: budget,
            phase_backing,
            stages,
            job,
            suspended: false,
            suspended_floor: None,
            pending_membership: None,
        };
        trace::published(self, &batch);
        Ok(Some(batch))
    }
}

fn restore_work<R: DeviceRuntime>(
    previous: &mut Option<&mut ContinuousBatch<R>>,
    requests: &mut [&mut ContinuousRequest<R>],
    scope: HostMemoryScope,
) {
    if let Some(previous) = previous.as_deref_mut() {
        previous.work = Some(scope);
    } else {
        requests[0].minimum_work = Some(scope);
    }
}

/// Only acknowledged, pre-submission cleanup is temporary capacity. A failed
/// native release or any submitted/unknown execution keeps its real owner in
/// the ordinary failure and is never converted into a successful retry signal.
fn clean_capacity_failure<R: DeviceRuntime>(
    failure: ImageGenerationFailure<R>,
) -> std::result::Result<(), ImageGenerationFailure<R>> {
    use ferrum_execution::component::ComponentInitializationError;
    let failure = match failure.recover_allocation() {
        Ok(_) => return Ok(()),
        Err(failure) => failure,
    };
    match failure {
        ImageGenerationFailure::AllocationRejected { .. }
        | ImageGenerationFailure::CapacityRejected { .. }
        | ImageGenerationFailure::InitializationCapacityRejected { .. }
        | ImageGenerationFailure::Initialization {
            source: ComponentInitializationError::Contract(VNextError::DeviceCapacityUnavailable(_)),
            ..
        } => Ok(()),
        ImageGenerationFailure::Initialization {
            plan,
            source:
                ComponentInitializationError::Pool {
                    source: VNextError::DeviceCapacityUnavailable(_),
                    transaction,
                    ..
                },
        } => match transaction.release() {
            Ok(released) => {
                drop(released);
                Ok(())
            }
            Err(source) => Err(ImageGenerationFailure::InitializationRelease { plan, source }),
        },
        failure => Err(failure),
    }
}
