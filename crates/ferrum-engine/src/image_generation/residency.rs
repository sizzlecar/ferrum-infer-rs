//! Two bounded session/snapshot slots. Compatible plans share the same sealed
//! physical weights; dynamic storage remains independently admitted.
use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageResidencyPolicy {
    #[default]
    Cold,
    /// Retain DiT and VAE static state between batches, releasing idle dynamic
    /// backing after each stage. TE and noise remain cold. Different complete
    /// plans (including masks/profiles) are closed before replacement. A new
    /// plan may explicitly bind the same readonly physical weight snapshot;
    /// incompatible weight contracts fail instead of being silently reused.
    DenoiserAndDecoder,
}

pub(super) struct ResidentStages<R: DeviceRuntime> {
    denoiser: Option<ActiveImageStage<R>>,
    decoder: Option<ActiveImageStage<R>>,
    denoiser_weights: Option<ResidentWeights<R>>,
    decoder_weights: Option<ResidentWeights<R>>,
}
struct ResidentWeights<R: DeviceRuntime> {
    weights: SharedComponentWeights<R>,
    origin: ImageStageExecutionReceipt,
}
impl<R: DeviceRuntime> ResidentStages<R> {
    pub(super) fn new() -> Self {
        Self {
            denoiser: None,
            decoder: None,
            denoiser_weights: None,
            decoder_weights: None,
        }
    }
    fn weights_slot(&mut self, stage: ImageStage) -> Option<&mut Option<ResidentWeights<R>>> {
        match stage {
            ImageStage::Denoiser => Some(&mut self.denoiser_weights),
            ImageStage::Decoder => Some(&mut self.decoder_weights),
            _ => None,
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
                previous.receipt.weights_reused = true;
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
            if let Some(weights) = self
                .resident_stages
                .weights_slot(prepared.plan.stage)
                .and_then(|slot| slot.as_ref())
                .map(|entry| entry.weights.clone())
            {
                // The runtime checks the actual Arc and full physical contract.
                // On failure the slot keeps its snapshot; no source read or
                // cold fallback can silently replace the selected installation.
                return prepared.start_attached(self, job, &weights);
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
        let stage = prepared.plan.stage;
        if self.config.residency == ImageResidencyPolicy::DenoiserAndDecoder
            && matches!(stage, ImageStage::Denoiser | ImageStage::Decoder)
        {
            let (active, weights) = prepared.start_shared(self, job, source)?;
            *self
                .resident_stages
                .weights_slot(stage)
                .expect("resident weight stage") = Some(ResidentWeights {
                weights,
                origin: active.receipt.clone(),
            });
            Ok(active)
        } else {
            prepared.start(self, job, source)
        }
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
                active.receipt.weights_reused = true;
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
        // Every session has retired before any final snapshot release. A live
        // failure/foreign execution owner makes close Referenced, never success.
        for stage in [ImageStage::Denoiser, ImageStage::Decoder] {
            if let Some(ResidentWeights {
                weights,
                mut origin,
            }) = self
                .resident_stages
                .weights_slot(stage)
                .and_then(Option::take)
            {
                origin.completed_executions = 0;
                origin.execution_wall_durations_us.clear();
                origin.initialization_wall_duration_us = 0;
                origin.plan.compile_wall_duration_us = 0;
                origin.session_reused = false;
                origin.weights_reused = true;
                let started = std::time::Instant::now();
                let failure = match weights.close() {
                    Ok(SharedComponentWeightsCloseOutcome::Closed(receipt)) => {
                        completed_stages.push(ImageStageReceipt {
                            execution: origin,
                            close: None,
                            retirement: ImageStageRetirement::SharedWeightsClosed { receipt },
                            close_wall_duration_us: u64::try_from(started.elapsed().as_micros())
                                .unwrap_or(u64::MAX),
                        });
                        continue;
                    }
                    Ok(outcome @ SharedComponentWeightsCloseOutcome::Referenced { .. }) => {
                        ImageGenerationFailure::SharedWeightsReferenced { stage, outcome }
                    }
                    Err(source) => ImageGenerationFailure::SharedWeightsClose { stage, source },
                };
                return Err(ImageGenerationError {
                    failure,
                    completed_stages,
                    completed_steps: Vec::new(),
                });
            }
        }
        Ok(completed_stages)
    }
}
