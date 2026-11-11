//! Complete single-request generation using the original execution owner.
//! Stages retain the shared runtime, actor and native ownership rules. This
//! direct entrypoint does not claim concurrent GPU qualification.
use super::{
    stages::{CompiledDecoder, CompiledDenoiser},
    PH3ImagePipeline,
};
use crate::image_generation::{
    ImageEditRequest, ImageGenerationError, ImageGenerationFailure, ImageGenerationOutput,
    ImageGenerationRequest, ImageStage, ImageStageReceipt, ImageWeightOwnerKey,
    ImageWeightOwnerPin,
};
use anyhow::{ensure, Result};
use ferrum_models::{
    image_pixels::OwnedImagePixels,
    p_h3_image::{
        PH3ImageRefinedTextCache, PH3ImageRefinementDestination, PH3ImageTextRefinementError,
    },
    p_h3_image_vae::PH3ImageVaePipelineError,
};
use ferrum_runtime::*;
use std::time::Instant;

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    /// The same persistent request lane and packed-step consumer used by the
    /// service worker, with one caller-supplied member.
    pub fn generate(
        &mut self,
        request: ImageGenerationRequest,
    ) -> std::result::Result<ImageGenerationOutput, ImageGenerationError<R>> {
        let mut outputs = self.generate_continuous_batch(vec![request])?;
        Ok(outputs
            .pop()
            .expect("one admitted H3 request returns one image"))
    }

    /// Ordered reference-image editing shares the same admitted request lane, actual
    /// packed DiT step and final decoder as generation and serving.
    pub fn edit(
        &mut self,
        request: ImageEditRequest,
    ) -> std::result::Result<ImageGenerationOutput, ImageGenerationError<R>> {
        let mut outputs = self.edit_continuous_batch(vec![request])?;
        Ok(outputs
            .pop()
            .expect("one admitted H3 edit returns one image"))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_refinement(
        &mut self,
        prepared: &CompiledDenoiser<R>,
        job: u64,
        scope: &HostMemoryScope,
        work_host: HostMemoryAccount,
        identity: &RequestIdentity,
        conditioning: &ManagedHostTensor,
        pin: &ImageWeightOwnerPin<R>,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<Vec<PH3ImageRefinedTextCache>, ImageGenerationFailure<R>> {
        let span = self.trace_phase_started(
            None,
            Some(identity),
            crate::image_generation::continuous_trace::ImageTracePhase::Refiner,
        );
        let result = self.execute_refinement_inner(
            prepared,
            job,
            scope,
            work_host,
            identity,
            conditioning,
            pin,
            receipts,
        );
        self.trace_phase_finished(
            span,
            if result.is_ok() {
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed
            } else {
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed
            },
        );
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_refinement_inner(
        &mut self,
        prepared: &CompiledDenoiser<R>,
        job: u64,
        scope: &HostMemoryScope,
        work_host: HostMemoryAccount,
        identity: &RequestIdentity,
        conditioning: &ManagedHostTensor,
        pin: &ImageWeightOwnerPin<R>,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<Vec<PH3ImageRefinedTextCache>, ImageGenerationFailure<R>> {
        let mut active = self
            .execution
            .start_stage(prepared.stage.clone(), job, receipts)?;
        let started = Instant::now();
        let result = (|| {
            let run = RunId::new(format!("run.image.h3.{job}.refine"))
                .map_err(anyhow::Error::from)
                .map_err(PH3ImageTextRefinementError::Preparation)?;
            let mut consumer = prepared
                .recipe
                .bind_refinement(&mut active.session, pin.weights(), work_host)
                .map_err(PH3ImageTextRefinementError::Preparation)?;
            consumer.execute_on_actor(
                self.execution.serial_actor.as_mut().expect("live actor"),
                run,
                identity.clone(),
                conditioning,
                &[PH3ImageRefinementDestination::new(identity, scope)],
            )
        })();
        active
            .receipt
            .execution_wall_durations_us
            .push(elapsed(started));
        if let Some(profile) = active.session.take_execution_profile() {
            active.receipt.device_profiles.push(profile);
        }
        let caches = match result {
            Ok(caches) => {
                active.receipt.completed_executions += 1;
                caches
            }
            Err(PH3ImageTextRefinementError::Preparation(source)) => {
                return Err(active.processing(source))
            }
            Err(PH3ImageTextRefinementError::Execution(source)) => {
                return Err(ImageGenerationFailure::Execution { source, active })
            }
        };
        self.execution.finish_stage(active, receipts)?;
        Ok(caches)
    }

    pub(super) fn execute_decoder(
        &mut self,
        prepared: &CompiledDecoder<R>,
        image: ManagedHostTensor,
        job: u64,
        identity: &RequestIdentity,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<OwnedImagePixels, ImageGenerationFailure<R>> {
        let span = self.trace_phase_started(
            None,
            Some(identity),
            crate::image_generation::continuous_trace::ImageTracePhase::Decoder,
        );
        let result = self.execute_decoder_inner(prepared, image, job, identity, receipts);
        self.trace_phase_finished(
            span,
            if result.is_ok() {
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed
            } else {
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed
            },
        );
        result
    }

    fn execute_decoder_inner(
        &mut self,
        prepared: &CompiledDecoder<R>,
        image: ManagedHostTensor,
        job: u64,
        identity: &RequestIdentity,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<OwnedImagePixels, ImageGenerationFailure<R>> {
        let host = self.execution.execution_host().clone();
        let normalized = (|| -> Result<_> {
            let spec = prepared.recipe.normalized_latent_spec()?;
            ensure!(
                matches!(image.tensor().dimensions(), [1, c, 1, h, w]
                if spec.dimensions() == [1, *c, *h, *w])
                    && image.tensor().element_type() == ElementType::F32
                    && image.tensor().layout() == &ResolvedTensorLayout::Contiguous
                    && image.tensor().minimum_storage_bytes()? == spec.minimum_storage_bytes()?,
                "H3 decoder requires the exact final single-frame sample"
            );
            Ok(ManagedHostTensor::new(
                image.value_id().clone(),
                spec,
                image.bytes().clone(),
            )?)
        })()
        .map_err(ImageGenerationFailure::Preflight)?;
        drop(image);
        let mut active: Option<crate::image_generation::ActiveImageStage<R>> = None;
        let mut active_shape = None;
        let mut weights = None;
        let mut completed_executions = 0usize;
        let output = (|| -> std::result::Result<_, ImageGenerationFailure<R>> {
            let mut consumer = prepared
                .recipe
                .bind(&normalized, identity.clone(), host.clone(), host.clone())
                .map_err(ImageGenerationFailure::Preflight)?;
            loop {
                let Some(group) =
                    consumer
                        .next_group()
                        .map_err(|source| ImageGenerationFailure::Processing {
                            source,
                            active: active.take(),
                        })?
                else {
                    break;
                };
                // Only the final traversal group can have a shorter shape.
                // Close the previous session before admitting its dynamic pools;
                // the original pin keeps their one static installation live.
                if active_shape != Some(group.tile_count()) {
                    if let Some(previous) = active.take() {
                        self.execution.finish_stage(previous, receipts)?;
                    }
                    let stage = prepared
                        .stage(group.tile_count())
                        .map_err(ImageGenerationFailure::Preflight)?;
                    active = Some(self.execution.start_stage(stage.clone(), job, receipts)?);
                    active_shape = Some(group.tile_count());
                    if prepared.shares_weights() && weights.is_none() {
                        weights = Some(
                            self.execution
                                .pin_weight_owner(ImageWeightOwnerKey::Decoder)
                                .map_err(|source| ImageGenerationFailure::Processing {
                                    source,
                                    active: active.take(),
                                })?,
                        );
                    }
                }
                let run = RunId::new(format!("run.image.h3.{job}.decode.{}", group.first_tile()))
                    .map_err(anyhow::Error::from)
                    .map_err(|source| ImageGenerationFailure::Processing {
                        source,
                        active: active.take(),
                    })?;
                let current = active.as_mut().expect("actual decoder shape is active");
                let before = consumer.executed_groups();
                let started = Instant::now();
                let result = consumer.advance_with_session(
                    &mut current.session,
                    self.execution.serial_actor.as_mut().expect("live actor"),
                    run,
                );
                current
                    .receipt
                    .execution_wall_durations_us
                    .push(elapsed(started));
                completed_executions = consumer.executed_groups();
                current.receipt.completed_executions += completed_executions - before;
                if let Some(profile) = current.session.take_execution_profile() {
                    current.receipt.device_profiles.push(profile);
                }
                match result {
                    Ok(true) => break,
                    Ok(false) => (),
                    Err(PH3ImageVaePipelineError::Preparation(source)) => {
                        return Err(ImageGenerationFailure::Processing {
                            source,
                            active: active.take(),
                        });
                    }
                    Err(PH3ImageVaePipelineError::Execution(source)) => {
                        return Err(ImageGenerationFailure::Execution {
                            source,
                            active: active.take().expect("failed decoder session"),
                        });
                    }
                }
            }
            let pixels =
                consumer
                    .finish()
                    .map_err(|source| ImageGenerationFailure::Processing {
                        source,
                        active: active.take(),
                    })?;
            if let Some(current) = active.take() {
                self.execution.finish_stage(current, receipts)?;
            }
            drop(weights.take());
            if prepared.shares_weights()
                && !self.execution.config.residency.retains(ImageStage::Decoder)
                && !self
                    .execution
                    .release_weight_owner(ImageWeightOwnerKey::Decoder, receipts)?
            {
                return Err(ImageGenerationFailure::Processing {
                    source: anyhow::anyhow!("cold decoder snapshot still has a live pin"),
                    active: None,
                });
            }
            Ok(pixels)
        })();
        match output {
            Ok(pixels) => Ok(pixels),
            Err(source) => {
                let source =
                    source.with_stage_execution_progress(ImageStage::Decoder, completed_executions);
                if active.is_none() && weights.is_none() {
                    Err(source)
                } else {
                    Err(ImageGenerationFailure::Pipeline {
                        source: Box::new(source),
                        stages: active.into_iter().collect(),
                        weights: weights.into_iter().collect(),
                        host_scopes: Vec::new(),
                        host_backing: Vec::new(),
                    })
                }
            }
        }
    }
}
fn elapsed(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}
