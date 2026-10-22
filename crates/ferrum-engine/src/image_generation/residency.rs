//! Two bounded, exact-plan session slots. Static and dynamic storage continue
//! to use the existing runtime authority; this module creates no allocator.
use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageResidencyPolicy {
    #[default]
    Cold,
    /// Retain DiT and VAE static state between batches, releasing idle dynamic
    /// backing after each stage. TE and noise remain cold. Different complete
    /// plans (including masks/profiles) are closed before replacement.
    DenoiserAndDecoder,
}

pub(super) struct ResidentStages<R: DeviceRuntime> {
    denoiser: Option<ActiveImageStage<R>>,
    decoder: Option<ActiveImageStage<R>>,
}
impl<R: DeviceRuntime> ResidentStages<R> {
    pub(super) fn new() -> Self {
        Self {
            denoiser: None,
            decoder: None,
        }
    }
    fn slot(&mut self, stage: ImageStage) -> Option<&mut Option<ActiveImageStage<R>>> {
        match stage {
            ImageStage::Denoiser => Some(&mut self.denoiser),
            ImageStage::Decoder => Some(&mut self.decoder),
            _ => None,
        }
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub fn residency_policy(&self) -> ImageResidencyPolicy {
        self.config.residency
    }

    pub(super) fn start_stage(
        &mut self,
        prepared: PreparedStage<R>,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        if self.config.residency == ImageResidencyPolicy::DenoiserAndDecoder {
            if let Some(previous) = self
                .resident_stages
                .slot(prepared.plan.stage)
                .and_then(Option::take)
            {
                let mut previous = previous;
                // These counters describe this acquisition, not the old batch.
                previous.receipt.completed_executions = 0;
                previous.receipt.execution_wall_durations_us.clear();
                previous.receipt.initialization_wall_duration_us = 0;
                previous.receipt.session_reused = true;
                if previous.receipt.plan.plan_hash == prepared.plan.plan_hash
                    && previous.receipt.plan.family_fingerprint == prepared.plan.family_fingerprint
                    && previous.receipt.plan.profile == prepared.plan.profile
                    && previous.receipt.plan.batch_size == prepared.plan.batch_size
                {
                    previous.receipt.plan = prepared.plan;
                    return Ok(previous);
                }
                // Publish an actual zero-execution eviction receipt. A failed
                // close owns the original session and prevents initialization.
                // The old plan was compiled during an earlier acquisition.
                previous.receipt.plan.compile_wall_duration_us = 0;
                stages.push(previous.close()?);
            }
        }
        let source = match prepared.plan.stage {
            ImageStage::TextEncoder => {
                ComponentWeightSource::Stored(self.assets.sources.text_encoder.as_ref())
            }
            ImageStage::Noise => ComponentWeightSource::None,
            ImageStage::Denoiser => {
                ComponentWeightSource::Stored(self.assets.sources.transformer.as_ref())
            }
            ImageStage::Decoder => ComponentWeightSource::Stored(self.assets.sources.vae.as_ref()),
        };
        prepared.start(self, job, source)
    }

    pub(super) fn finish_stage(
        &mut self,
        mut active: ActiveImageStage<R>,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<(), ImageGenerationFailure<R>> {
        if self.config.residency == ImageResidencyPolicy::DenoiserAndDecoder {
            if let Some(slot) = self.resident_stages.slot(active.receipt.plan.stage) {
                if slot.is_some() {
                    return Err(
                        active.processing(anyhow::anyhow!("resident stage slot already occupied"))
                    );
                }
                let idle_release = match active.session.try_release_idle_dynamic_backing() {
                    Ok(IdleDynamicReleaseOutcome::Released(receipt)) => receipt,
                    Ok(IdleDynamicReleaseOutcome::Busy(reason)) => {
                        return Err(active.processing(anyhow::anyhow!(
                            "stage is not idle for residency: {reason:?}"
                        )))
                    }
                    Err(error) => return Err(active.processing(error.into())),
                };
                stages.push(ImageStageReceipt {
                    execution: active.receipt.clone(),
                    close: None,
                    retirement: ImageStageRetirement::Resident { idle_release },
                    close_wall_duration_us: 0,
                });
                *slot = Some(active);
                return Ok(());
            }
        }
        stages.push(active.close()?);
        Ok(())
    }

    /// Stop admission and explicitly close retained sessions. On failure the
    /// returned error retains the original failed owner; the pipeline retains
    /// any other session not yet closed. Keeping only an error string loses
    /// that recovery authority. Already returned Host outputs remain valid.
    pub fn shutdown(&mut self) -> Result<Vec<ImageStageReceipt>, ImageGenerationError<R>> {
        self.stopped = true;
        let mut completed_stages = Vec::new();
        for stage in [ImageStage::Denoiser, ImageStage::Decoder] {
            if let Some(mut active) = self.resident_stages.slot(stage).and_then(Option::take) {
                active.receipt.completed_executions = 0;
                active.receipt.execution_wall_durations_us.clear();
                active.receipt.initialization_wall_duration_us = 0;
                active.receipt.plan.compile_wall_duration_us = 0;
                active.receipt.session_reused = true;
                match active.close() {
                    Ok(receipt) => completed_stages.push(receipt),
                    Err(failure) => {
                        return Err(ImageGenerationError {
                            failure,
                            completed_stages,
                            completed_steps: Vec::new(),
                        })
                    }
                }
            }
        }
        Ok(completed_stages)
    }
}
