//! Bounded model session/snapshot slots. Compatible plans share the same sealed
//! physical weights; dynamic storage remains independently admitted.
use super::*;
use ferrum_quantization::bf16_whole_tensor_fp8::ExactBf16SafetensorsSource;

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
    /// Also retain the text encoder's session and readonly weight snapshot.
    /// All retained weights remain charged to the same device account during
    /// every other stage. Noise stays cold. This does not imply that all model
    /// weights and the active stage's dynamic backing fit a given device.
    AllModelComponents,
}

impl ImageResidencyPolicy {
    fn retains(self, stage: ImageStage) -> bool {
        match self {
            Self::Cold => false,
            Self::DenoiserAndDecoder => {
                matches!(stage, ImageStage::Denoiser | ImageStage::Decoder)
            }
            Self::AllModelComponents => !matches!(stage, ImageStage::Noise),
        }
    }
}

pub(super) struct ResidentStages<R: DeviceRuntime> {
    text_encoder: Option<ActiveImageStage<R>>,
    denoiser: Option<ActiveImageStage<R>>,
    decoder: Option<ActiveImageStage<R>>,
    text_encoder_weights: Option<ResidentWeights<R>>,
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
            text_encoder: None,
            denoiser: None,
            decoder: None,
            text_encoder_weights: None,
            denoiser_weights: None,
            decoder_weights: None,
        }
    }
    fn weights_slot(&mut self, stage: ImageStage) -> Option<&mut Option<ResidentWeights<R>>> {
        match stage {
            ImageStage::TextEncoder => Some(&mut self.text_encoder_weights),
            ImageStage::Denoiser => Some(&mut self.denoiser_weights),
            ImageStage::Decoder => Some(&mut self.decoder_weights),
            _ => None,
        }
    }
    fn slot(&mut self, stage: ImageStage) -> Option<&mut Option<ActiveImageStage<R>>> {
        match stage {
            ImageStage::TextEncoder => Some(&mut self.text_encoder),
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

    /// Batch planning evidence, not an allocation permit. All retained model
    /// installations coexist; only one stage's dynamic pools execute at once.
    /// The shared device ledger remains authoritative under outside contention.
    pub(super) fn validate_resident_capacity(
        &self,
        stages: &[&PreparedStage<R>],
    ) -> anyhow::Result<()> {
        if self.config.residency == ImageResidencyPolicy::Cold {
            return Ok(());
        }
        let mut retained_static = 0_u64;
        let mut active_peak = 0_u64;
        for stage in stages {
            let payload = stage.component.executable().execution_plan().payload();
            ensure!(
                payload.work_limits() == RuntimeWorkLimits::FixedTensor,
                "image resident capacity requires compiled fixed tensor plans"
            );
            let memory = payload.memory();
            let retained = self.config.residency.retains(stage.plan.stage);
            if retained {
                retained_static = retained_static
                    .checked_add(memory.static_bytes())
                    .context("resident static bytes overflow")?;
            }
            // Fixed demand includes actual compiled provider scratch, per-pool
            // alignment, and proven activation/invocation lifetime reuse. Do
            // not add the cuDNN workspace limit again or sum all dynamic plans.
            let active = memory
                .minimum_runnable_request_bytes()
                .checked_add(if retained { 0 } else { memory.static_bytes() })
                .context("active component bytes overflow")?;
            active_peak = active_peak.max(active);
        }
        let required = retained_static
            .checked_add(active_peak)
            .and_then(|n| n.checked_add(self.device_opaque_allowance_bytes))
            .context("resident batch capacity overflows")?;
        ensure!(required <= self.config.device_capacity_bytes,
            "resident image batch needs {required} device bytes (retained static {retained_static}, active stage peak {active_peak}, native allowance {}), exceeding total ceiling {}",
            self.device_opaque_allowance_bytes, self.config.device_capacity_bytes);
        Ok(())
    }

    pub(super) fn start_stage(
        &mut self,
        prepared: PreparedStage<R>,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        if self.config.residency.retains(prepared.plan.stage) {
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
        // Keep the archive's optional implicit dtype conversion out of the
        // managed FP8 conversion path. These wrappers borrow the original mmap
        // and stay alive throughout synchronous static initialization.
        let text_source = ExactBf16SafetensorsSource::new(&self.assets.sources.text_encoder);
        let dit_source = ExactBf16SafetensorsSource::new(&self.assets.sources.transformer);
        let profile = prepared.plan.profile.as_str();
        let source = match prepared.plan.stage {
            ImageStage::TextEncoder if profile == Qwen3VLTextEncoderFamily::CUDA_FP8_PROFILE_ID => {
                ComponentWeightSource::Stored(&text_source)
            }
            ImageStage::TextEncoder => {
                ComponentWeightSource::Stored(self.assets.sources.text_encoder.as_ref())
            }
            ImageStage::Noise => ComponentWeightSource::None,
            ImageStage::Denoiser
                if matches!(
                    profile,
                    QwenImage21TransformerFamily::CUDA_FP8_PROFILE_ID
                        | QwenImage21TransformerFamily::CUDA_FP8_FUSED_PROFILE_ID
                ) =>
            {
                ComponentWeightSource::Stored(&dit_source)
            }
            ImageStage::Denoiser => {
                ComponentWeightSource::Stored(self.assets.sources.transformer.as_ref())
            }
            ImageStage::Decoder => ComponentWeightSource::Stored(self.assets.sources.vae.as_ref()),
        };
        let stage = prepared.plan.stage;
        if self.config.residency.retains(stage) {
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
        if self.config.residency.retains(active.receipt.plan.stage) {
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
        for stage in [
            ImageStage::TextEncoder,
            ImageStage::Denoiser,
            ImageStage::Decoder,
        ] {
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
        for stage in [
            ImageStage::TextEncoder,
            ImageStage::Denoiser,
            ImageStage::Decoder,
        ] {
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
